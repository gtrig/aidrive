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
ACTIONS = [
    (0.05,  0),    # 0 – throttle straight
    (0.0,   0),    # 1 – coast
    (-0.1,  0),    # 2 – brake
    (0, -4),        # 3 – turn left
    (0, +4),        # 4 – turn right
    (0, -12),       # 5 – sharp left  (±6 was too weak for hairpins)
    (0, +12),       # 6 – sharp right
    (0.05, -4),     # 7 – throttle + turn left
    (0.05, +4),     # 8 – throttle + turn right
    (0.05, -12),    # 9 – throttle + sharp left
    (0.05, +12),    # 10 – throttle + sharp right
]

# Reward weights
R_GATE        =  5.0    # crossing the next gate forward
R_LAP         = 10.0    # flat bonus for completing a full lap
R_LAP_TIME    = 20.0    # additional time bonus: R_LAP_TIME * (1 - steps / LAP_REF_STEPS)
LAP_REF_STEPS = 2000    # reference lap (slowest acceptable); faster → more bonus
R_APPROACH    =  0.004  # per px closer to the next gate midpoint (dense signal)
R_FORWARD     =  0.02   # per px of movement along the track-forward gate normal
R_SPEED       =  0.008  # per unit of track-aligned forward speed
R_WRONG_WAY   =  0.015  # per unit of speed when heading opposes track forward
R_ON_ROAD     =  0.004  # per step while still on track (offsets time pressure)
R_OFF_ROAD    = -5.0    # episode-ending penalty
R_TIME        = -0.002  # mild per-step pressure (was -0.01 → -20/ep at timeout)

MAX_STEPS     = 2000

# Normalisation constants
SENSOR_MAX   = 200.0    # longest sensor size
DIST_MAX     = 1500.0   # approx max gate-to-gate distance on track

# Per-track occupancy grids built once and shared across all env workers.
_OCCUPANCY_CACHE: dict[str, np.ndarray] = {}


def get_shared_occupancy(track_id: str, road_polygon, width: int, height: int) -> np.ndarray:
    """Return a cached read-only occupancy grid for track_id."""
    if track_id not in _OCCUPANCY_CACHE:
        _OCCUPANCY_CACHE[track_id] = CarEnv._build_occupancy_grid(
            road_polygon, width, height,
        )
    return _OCCUPANCY_CACHE[track_id]


class CarEnv:
    """
    Minimal gym-like interface:
        obs, info = env.reset()
        obs, reward, done, info = env.step(action)

    Observation vector (obs_dim = n_sensors + 1 + 3):
        [sensor_0..n_norm, track_speed_norm,
         dist_to_next_gate_norm, sin(heading_diff), cos(heading_diff)]
        track_speed_norm is signed: positive along track forward, negative when
        the car faces / drives the wrong way.
    """

    def __init__(
        self,
        sensor_layout=None,
        max_steps=MAX_STEPS,
        gates_path=None,
        track_npy=None,
        rand_start: bool = False,
        track_id: str = 'track1',
        occupancy_grid: np.ndarray | None = None,
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

        # minspeed=0: default Car mins=-1 lets sustained braking enter reverse; dense
        # R_APPROACH then rewards sliding toward the gate midpoint without crossing it.
        self.car = Car(
            x=self._start_x, y=self._start_y,
            speed=0, maxspeed=4,
            minspeed=0,
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

        if occupancy_grid is not None:
            self._occupancy = occupancy_grid
        else:
            self._occupancy = get_shared_occupancy(
                track_id,
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

        reward = R_TIME + R_ON_ROAD
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

        alignment = self._track_alignment()
        speed = max(self.car.speed, 0)

        # Speed reward only when heading aligns with track forward — raw speed
        # rewarded the agent for throttling while facing backward.
        reward += R_SPEED * speed * max(alignment, 0.0)
        if alignment < 0.0 and speed > 0.0:
            reward -= R_WRONG_WAY * speed

        cur_pos  = (self.car.x, self.car.y)
        cur_dist = self._dist_to_next_gate(self.car.x, self.car.y)

        # Forward progress along the track direction (gate normal).  Rewards
        # driving the right way; penalises backing/sliding against track flow.
        nx, ny = self._gate_normals[self._next_gate]
        nlen = math.hypot(nx, ny) or 1.0
        mvx = cur_pos[0] - self._prev_pos[0]
        mvy = cur_pos[1] - self._prev_pos[1]
        forward_progress = (mvx * nx + mvy * ny) / nlen
        reward += R_FORWARD * forward_progress

        # Dense approach reward only when advancing along the track — otherwise
        # the agent can face backward, throttle "forward", and farm gate distance.
        if forward_progress > 0:
            reward += R_APPROACH * (self._prev_dist - cur_dist)
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

    def _track_alignment(self, gate_idx: int | None = None) -> float:
        """Cosine of angle between car heading and the track-forward gate normal."""
        idx = self._next_gate if gate_idx is None else gate_idx
        nx, ny = self._gate_normals[idx]
        nlen = math.hypot(nx, ny) or 1.0
        heading_rad = math.radians(self.car.orientation)
        fx = math.sin(heading_rad)
        fy = math.cos(heading_rad)
        return (fx * nx + fy * ny) / nlen

    def _dist_to_next_gate(self, x, y):
        if self._next_gate >= self.n_gates:
            return 0.0
        mx, my = self._gate_mids[self._next_gate]
        return math.hypot(x - mx, y - my)

    def _observation(self):
        """
        [sensor_0..n normalised, track_speed_norm,
         dist_to_next_gate_norm, sin(heading_diff), cos(heading_diff)]
        heading_diff = angle from car orientation to direction of next gate.
        """
        obs = [min(s.distance, SENSOR_MAX) / SENSOR_MAX for s in self.car.sensors]
        alignment = self._track_alignment()
        track_speed_norm = (self.car.speed * alignment) / self.car.maxspeed
        obs.append(float(np.clip(track_speed_norm, -1.0, 1.0)))

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
