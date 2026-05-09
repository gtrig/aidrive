"""
Gym-like environment wrapping Car, Track, LineTools and gate progression.
No pyglet window is opened; safe to run in a tight training loop.
"""

import math
import numpy as np
import shapely
from pathlib import Path

import config
from entities.car import Car
from entities.track import Track
from system.tools import LineTools
from system.track_registry import load as load_track_meta

# Sensor layout used for training and for the autonomous play mode.
TRAINING_SENSOR_LAYOUT = [(-90, 80), (-45, 100), (-20, 150), (0, 200), (20, 150), (45, 100), (90, 80)]

# Discrete action set: (acceleration, steering_degrees)
# Steering reduced to ±3° for more stable random exploration;
# ±7° added for tight corners.
ACTIONS = [
    (0.05,  0),   # 0 – throttle straight   (most likely during exploration)
    (0.0,   0),   # 1 – coast
    (-0.1,  0),   # 2 – brake
    (0, -3),       # 3 – turn left
    (0, +3),       # 4 – turn right
    (-0.1, -3),    # 5 – brake + turn left
    (-0.1, +3),    # 6 – brake + turn right
    (0, -6),       # 5 – turn sharp left
    (0, +6),       # 6 – turn sharp right
]

# Reward weights
R_GATE        =  5.0    # crossing the next gate forward
R_LAP         = 10.0    # flat bonus for completing a full lap
R_LAP_TIME    = 20.0    # additional time bonus: R_LAP_TIME * (1 - steps / LAP_REF_STEPS)
LAP_REF_STEPS = 2000    # reference lap (slowest acceptable); faster → more bonus
R_APPROACH    =  0.004  # per px closer to the next gate midpoint (dense signal)
R_SPEED       =  0.005  # per unit of speed on road
R_OFF_ROAD    = -5.0    # episode-ending penalty
R_TIME        = -0.01  # small per-step time pressure

MAX_STEPS     = 2000

# Normalisation constants
SENSOR_MAX   = 200.0    # longest sensor size
DIST_MAX     = 1500.0   # approx max gate-to-gate distance on track


class CarEnv:
    """
    Minimal gym-like interface:
        obs, info = env.reset()
        obs, reward, done, info = env.step(action)

    Observation vector (obs_dim = n_sensors + 1 + 3):
        [sensor_0..n_norm, speed_norm,
         dist_to_next_gate_norm, sin(heading_diff), cos(heading_diff)]
    """

    def __init__(
        self,
        sensor_layout=None,
        max_steps=MAX_STEPS,
        gates_path=None,
        track_npy=None,
        rand_start: bool = False,
        track_id: str = 'track1',
    ):
        self.sensor_layout = sensor_layout or TRAINING_SENSOR_LAYOUT
        self.max_steps = max_steps
        self.rand_start = rand_start
        self.track_id = track_id

        _meta = load_track_meta(track_id)
        self._start_x    = _meta.start_x
        self._start_y    = _meta.start_y
        self._start_heading = _meta.start_heading

        gates_file = Path(gates_path) if gates_path else _meta.gates_npy

        self.car = Car(
            x=self._start_x, y=self._start_y,
            speed=0, maxspeed=4,
            heading=self._start_heading,
            sensors=False,
            headless=True,
            sensor_layout=self.sensor_layout,
        )
        self.track = Track(headless=True, track_id=track_id)
        self.line_tools = LineTools(
            sensors=self.car.sensors,
            lines=self.track.lines,
        )

        self.gates = list(np.load(str(gates_file), allow_pickle=True))
        self.n_gates = len(self.gates)
        self._gate_normals = self._precompute_gate_normals(self.gates)
        self._gate_mids = [
            ((g[0][0] + g[1][0]) / 2.0, (g[0][1] + g[1][1]) / 2.0)
            for g in self.gates
        ]

        self._occupancy = self._build_occupancy_grid(
            self.track.road,
            config.window_width,
            config.window_height,
        )

        # obs: n_sensors + speed + dist_to_gate + sin(angle) + cos(angle)
        self.obs_dim   = len(self.sensor_layout) + 1 + 3
        self.n_actions = len(ACTIONS)

        self._step        = 0
        self._next_gate   = 0
        self._ep_gates    = 0   # gates crossed this episode
        self._lap_start_step = 0  # step when current lap began
        self._prev_pos    = (self._start_x, self._start_y)
        self._prev_dist   = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        if self.rand_start:
            # Place the car at the midpoint of a random gate, pointed in the
            # gate's forward direction.  This forces the agent to learn every
            # section of the track, not just the first corner.
            gate_idx = np.random.randint(0, self.n_gates)
            mid = self._gate_mids[gate_idx]
            nx, ny = self._gate_normals[gate_idx]
            start_x   = mid[0]
            start_y   = mid[1]
            # heading: atan2(nx, ny) gives degrees from +Y axis (same convention as car)
            start_hdg = math.degrees(math.atan2(nx, ny)) % 360
        else:
            gate_idx  = 0
            start_x   = self._start_x
            start_y   = self._start_y
            start_hdg = self._start_heading

        self.car.x           = start_x
        self.car.y           = start_y
        self.car.speed       = 1.5
        self.car.orientation = start_hdg
        self.car.acceleration = 0
        self.car.steering    = 0
        self.car.x_direction = 0
        self.car.y_direction = 1
        self.car.updateSensors()

        # MUST call getLinesInBox first to populate linesSample
        self.line_tools.updatePOI(self.car.x, self.car.y)
        self.line_tools.getLinesInBox()
        self.line_tools.getIntesections()

        self._step           = 0
        self._next_gate      = gate_idx
        self._ep_gates       = 0
        self._lap_start_step = 0
        self._prev_pos  = (self.car.x, self.car.y)
        self._prev_dist = self._dist_to_next_gate(self.car.x, self.car.y)

        return self._observation(), {}

    def step(self, action: int):
        accel, steer = ACTIONS[action]
        self.car.accelerate(accel)
        self.car.turn(steer)
        self.car.update_self()

        # MUST call getLinesInBox first so sensor intersection has nearby lines
        self.line_tools.updatePOI(self.car.x, self.car.y)
        self.line_tools.getLinesInBox()
        self.line_tools.getIntesections()

        reward = R_TIME
        done   = False
        info   = {}

        # off-road termination
        if not self._is_on_road(self.car.x, self.car.y):
            reward += R_OFF_ROAD
            done    = True
            info['reason']     = 'off_road'
            info['gates_hit']  = self._ep_gates
            self._prev_pos = (self.car.x, self.car.y)
            return self._observation(), reward, done, info

        # speed reward
        reward += R_SPEED * max(self.car.speed, 0)

        # dense approach reward: reward for getting closer to the next gate
        cur_pos  = (self.car.x, self.car.y)
        cur_dist = self._dist_to_next_gate(self.car.x, self.car.y)
        reward  += R_APPROACH * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        # gate crossing (forward direction only)
        # _next_gate wraps around so random-start episodes count a full loop
        # back to the starting gate as a lap, not just to the last gate index.
        gate   = self.gates[self._next_gate]
        normal = self._gate_normals[self._next_gate]
        if self._segment_crosses_gate(self._prev_pos, cur_pos, gate, normal):
            reward += R_GATE
            self._ep_gates  += 1
            self._next_gate  = (self._next_gate + 1) % self.n_gates
            self._prev_dist  = self._dist_to_next_gate(self.car.x, self.car.y)
            if self._ep_gates >= self.n_gates:
                lap_steps  = self._step - self._lap_start_step
                time_bonus = R_LAP_TIME * max(0.0, 1.0 - lap_steps / LAP_REF_STEPS)
                reward += R_LAP + time_bonus
                done    = True
                info['reason']          = 'lap_complete'
                info['gates_hit']       = self._ep_gates
                info['lap_steps']       = lap_steps
                info['lap_time_bonus']  = round(time_bonus, 3)

        self._prev_pos = cur_pos
        self._step    += 1

        if self._step >= self.max_steps:
            done = True
            info['reason']    = 'timeout'
            info['gates_hit'] = self._ep_gates

        return self._observation(), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dist_to_next_gate(self, x, y):
        if self._next_gate >= self.n_gates:
            return 0.0
        mx, my = self._gate_mids[self._next_gate]
        return math.hypot(x - mx, y - my)

    def _observation(self):
        """
        [sensor_0..n normalised, speed_norm,
         dist_to_next_gate_norm, sin(heading_diff), cos(heading_diff)]
        heading_diff = angle from car orientation to direction of next gate.
        """
        obs = [min(s.distance, SENSOR_MAX) / SENSOR_MAX for s in self.car.sensors]
        obs.append(self.car.speed / self.car.maxspeed)

        # navigation features
        if self._next_gate < self.n_gates:
            mx, my = self._gate_mids[self._next_gate]
            dx, dy = mx - self.car.x, my - self.car.y
            dist = math.hypot(dx, dy)
            obs.append(min(dist, DIST_MAX) / DIST_MAX)

            # angle from car heading to gate direction
            gate_angle   = math.degrees(math.atan2(dx, dy)) % 360
            heading_diff = math.radians((gate_angle - self.car.orientation) % 360)
            obs.append(math.sin(heading_diff))
            obs.append(math.cos(heading_diff))
        else:
            obs.extend([0.0, 0.0, 1.0])

        return np.array(obs, dtype=np.float32)

    def _is_on_road(self, px: float, py: float) -> bool:
        xi = max(0, min(int(px), self._occupancy.shape[1] - 1))
        yi = max(0, min(int(py), self._occupancy.shape[0] - 1))
        return bool(self._occupancy[yi, xi])

    @staticmethod
    def _build_occupancy_grid(road_polygon, width: int, height: int) -> np.ndarray:
        xs = np.arange(width,  dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(road_polygon, grid_x.ravel(), grid_y.ravel())
        return inside.reshape(height, width)

    @staticmethod
    def _precompute_gate_normals(gates):
        n    = len(gates)
        mids = [((g[0][0]+g[1][0])/2.0, (g[0][1]+g[1][1])/2.0) for g in gates]
        normals = []
        for i in range(n):
            prev_m = mids[(i-1) % n]
            next_m = mids[(i+1) % n]
            tdx = next_m[0] - prev_m[0]
            tdy = next_m[1] - prev_m[1]
            gdx = float(gates[i][1][0] - gates[i][0][0])
            gdy = float(gates[i][1][1] - gates[i][0][1])
            n1  = (-gdy, gdx)
            n2  = ( gdy, -gdx)
            normals.append(n1 if tdx*n1[0]+tdy*n1[1] >= tdx*n2[0]+tdy*n2[1] else n2)
        return normals

    @staticmethod
    def _segment_crosses_gate(p1, p2, gate, forward_normal):
        ax, ay = p1
        bx, by = p2
        cx, cy = gate[0][0], gate[0][1]
        dx, dy = gate[1][0], gate[1][1]

        def cross(ox, oy, qx, qy, rx, ry):
            return (qx-ox)*(ry-oy) - (qy-oy)*(rx-ox)

        d1 = cross(cx, cy, dx, dy, ax, ay)
        d2 = cross(cx, cy, dx, dy, bx, by)
        d3 = cross(ax, ay, bx, by, cx, cy)
        d4 = cross(ax, ay, bx, by, dx, dy)

        if not (((d1>0 and d2<0) or (d1<0 and d2>0)) and
                ((d3>0 and d4<0) or (d3<0 and d4>0))):
            return False

        mvx = bx - ax
        mvy = by - ay
        return (mvx * forward_normal[0] + mvy * forward_normal[1]) > 0
