"""
Visual play mode using the existing pyglet renderer.

Without a policy:   human keyboard control (same as magame.py)
With a policy:      the DQN agent drives; auto-resets and reloads the
                    newest checkpoint in models/ at the end of each episode.

Usage:
    .venv/bin/python play.py                          # human control
    .venv/bin/python play.py --policy models/dqn_best.pt
    .venv/bin/python play.py --policy models/          # watch latest in dir
    .venv/bin/python play.py --policy models/dqn_best.pt --allow-override
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyglet
from pyglet.window import key

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from entities.car import Car
from entities.track import Track
from system.component import Component
from system.tools import LineTools
from system.track_registry import load as load_track_meta, list_tracks
from env.car_env import TRAINING_SENSOR_LAYOUT, ACTIONS, SENSOR_MAX, CarEnv

RESET_PAUSE = 1.5   # seconds to freeze on screen after episode ends


def parse_args():
    p = argparse.ArgumentParser(description='Play / watch the trained DQN agent')
    p.add_argument('--policy',         type=str, default=None,
                   help='path to a .pt checkpoint OR a directory to watch for the newest one')
    p.add_argument('--models-dir',     type=str, default='models',
                   help='directory scanned for the latest checkpoint on each reset')
    p.add_argument('--allow-override', action='store_true',
                   help='keyboard overrides agent while keys are held')
    p.add_argument('--track',          type=str, default='track1',
                   help='which track to use (any folder in assets/tracks/)')
    return p.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def latest_checkpoint(models_dir: str) -> Path | None:
    """Return the most-recently-modified DQN .pt file in models_dir, or None.
    PPO checkpoints (ppo_*.pt) are excluded since play.py uses a DQNAgent."""
    pts = sorted(
        (p for p in Path(models_dir).glob('*.pt') if not p.name.startswith('ppo_')),
        key=lambda p: p.stat().st_mtime,
    )
    return pts[-1] if pts else None


def build_obs(car, henv, next_gate):
    """Sync car state into the headless env and return its _observation().
    This guarantees the play observation always matches training exactly."""
    henv.car.x           = car.x
    henv.car.y           = car.y
    henv.car.speed       = car.speed
    henv.car.orientation = car.orientation
    henv._next_gate      = next_gate
    for hs, vs in zip(henv.car.sensors, car.sensors):
        hs.distance = vs.distance
    return henv._observation()


def reset_car(car, sx=500, sy=90, sh=270):
    car.x = sx
    car.y = sy
    car.speed = 1.5    # match training start speed
    car.orientation = sh
    car.acceleration = 0
    car.steering = 0
    car.x_direction = 0
    car.y_direction = 1
    car.updateSensors()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    available = list_tracks()
    if args.track not in available:
        print(f'[play] unknown track "{args.track}". Available: {available}')
        raise SystemExit(1)

    _tmeta = load_track_meta(args.track)
    START_X_, START_Y_, START_HEADING_ = _tmeta.start_x, _tmeta.start_y, _tmeta.start_heading

    sensor_layout = TRAINING_SENSOR_LAYOUT if args.policy else [(-90, 45), (90, 45)]

    # ---- visual car & track ----
    car = Car(
        x=START_X_, y=START_Y_,
        speed=0, maxspeed=4,
        heading=START_HEADING_,
        sensors=True,
        headless=False,
        sensor_layout=sensor_layout,
    )
    track  = Track(headless=False, track_id=args.track)
    dlines = LineTools(sensors=car.sensors, lines=track.lines)
    gates  = np.load(str(_tmeta.gates_npy), allow_pickle=True)

    # headless env — always created for the occupancy grid and gate normals
    # so direction-gated checks apply in both AI and human modes
    _henv = CarEnv(sensor_layout=sensor_layout, track_id=args.track)

    def is_on_road(x, y):
        return _henv._is_on_road(x, y)

    # ---- agent ----
    agent = None
    loaded_from = None

    def load_agent(path: str | Path | None):
        nonlocal agent, loaded_from
        if path is None:
            return
        path = Path(path)
        if not path.exists():
            print(f'[play] checkpoint not found: {path}')
            return
        if path == loaded_from:
            return      # already up to date
        from agent.dqn import DQNAgent
        obs_dim = _henv.obs_dim
        if agent is None:
            agent = DQNAgent(obs_dim=obs_dim, n_actions=_henv.n_actions)
        try:
            agent.load(str(path))
        except ValueError as e:
            print(f'[play] {e}')
            print('[play] Skipping incompatible checkpoint.')
            return
        loaded_from = path
        print(f'[play] loaded {path.name}')
        try:
            window.set_caption(f'pyCar – AI  [{path.name}]')
        except NameError:
            pass   # window not yet created on the initial load call

    # ---- episode state ----
    state = {
        'next_gate':    0,
        'prev_pos':     (START_X_, START_Y_),
        'episode':      0,
        'gates_hit':    0,
        'step':         0,
        'paused':       False,
        'pause_until':  0.0,
        'status_text':  '',
        # lap timer (wall-clock seconds)
        'lap_start_t':  None,   # set in do_reset()
        'last_lap_t':   None,   # time of most recently completed lap
        'best_lap_t':   None,   # fastest lap so far (seconds)
    }

    gate_normals = _henv._gate_normals   # always available now

    def _gate_crossed(prev, cur, gate_idx):
        """Return True only when the car crosses gate gate_idx in the forward direction."""
        if gate_idx >= len(gates):
            return False
        return CarEnv._segment_crosses_gate(
            prev, cur, gates[gate_idx], gate_normals[gate_idx]
        )

    def end_episode(reason: str):
        state['episode'] += 1
        lap_note = ''
        if reason == 'lap complete!' and state['lap_start_t'] is not None:
            elapsed = time.time() - state['lap_start_t']
            state['last_lap_t'] = elapsed
            is_best = (state['best_lap_t'] is None or elapsed < state['best_lap_t'])
            if is_best:
                state['best_lap_t'] = elapsed
            lap_note = f'  {elapsed:.2f}s' + (' ★ best!' if is_best else '')
        state['status_text'] = (
            f'Episode {state["episode"]}  |  '
            f'Gates {state["gates_hit"]}/{len(gates)}  |  '
            f'{reason}{lap_note}  |  reloading…'
        )
        state['paused']      = True
        state['pause_until'] = time.time() + RESET_PAUSE

    def do_reset():
        reset_car(car, START_X_, START_Y_, START_HEADING_)
        dlines.updatePOI(car.x, car.y)
        dlines.getLinesInBox()
        dlines.getIntesections()
        state['next_gate']   = 0
        state['gates_hit']   = 0
        state['prev_pos']    = (car.x, car.y)
        state['step']        = 0
        state['paused']      = False
        state['lap_start_t'] = time.time()
        # reload newest checkpoint
        if args.policy:
            load_agent(latest_checkpoint(args.models_dir))

    # ---- window ----
    window = pyglet.window.Window(
        width=config.window_width,
        height=config.window_height,
        caption='pyCar – AI' if agent else 'pyCar – Human',
        resizable=False,
    )

    keys_held = set()

    # initial policy load (now that window exists for set_caption)
    if args.policy:
        p = Path(args.policy)
        load_agent(p if p.is_file() else latest_checkpoint(args.models_dir))

    # start the lap clock
    state['lap_start_t'] = time.time()

    # ---- draw ----
    def draw():
        window.clear()
        track.draw_self()

        for line in dlines.getLinesInBox():
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                ('v2f', (float(line[0][0]), float(line[0][1]),
                         float(line[1][0]), float(line[1][1]))),
                ('c3B', (255, 0, 0, 255, 0, 0)),
            )

        for idx, gate in enumerate(gates):
            # gate line
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                ('v2f', (gate[0][0], gate[0][1], gate[1][0], gate[1][1])),
                ('c3B', (0, 255, 0, 0, 255, 0)),
            )
            # direction arrow: small tick from midpoint toward forward normal
            mx = (gate[0][0] + gate[1][0]) / 2.0
            my = (gate[0][1] + gate[1][1]) / 2.0
            nx, ny = gate_normals[idx]
            length = (nx * nx + ny * ny) ** 0.5 or 1.0
            ax2 = mx + 10 * nx / length
            ay2 = my + 10 * ny / length
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                ('v2f', (mx, my, ax2, ay2)),
                ('c3B', (0, 200, 255, 0, 200, 255)),
            )

        count = 0
        for s in car.sensors:
            count += 1
            pyglet.text.Label(
                f'{s.offset}:{s.distance}',
                font_name='Times New Roman', font_size=10,
                x=30, y=window.height - 10 - (15 * count),
                anchor_x='left', anchor_y='center',
            ).draw()

        # top-right: mode + episode info
        mode = 'AI' if (agent and not keys_held) else 'HUMAN'
        pyglet.text.Label(
            f'{mode}  |  Ep {state["episode"]}  |  Gates {state["gates_hit"]}/{len(gates)}',
            font_name='Times New Roman', font_size=12,
            x=window.width - 10, y=window.height - 10,
            anchor_x='right', anchor_y='top',
        ).draw()

        # lap timer line (live clock + last/best lap)
        if not state['paused'] and state['lap_start_t'] is not None:
            live = time.time() - state['lap_start_t']
            timer_str = f'Lap  {live:.2f}s'
        elif state['last_lap_t'] is not None:
            timer_str = f'Last  {state["last_lap_t"]:.2f}s'
        else:
            timer_str = ''
        if state['best_lap_t'] is not None:
            timer_str += f'   Best  {state["best_lap_t"]:.2f}s ★'
        if timer_str:
            pyglet.text.Label(
                timer_str,
                font_name='Times New Roman', font_size=12,
                x=window.width - 10, y=window.height - 30,
                anchor_x='right', anchor_y='top',
                color=(100, 220, 100, 230),
            ).draw()

        # bottom-right: keybinding hints
        hints = 'SPACE – next episode  |  R – reset (with pause)  |  Arrows – drive'
        pyglet.text.Label(
            hints,
            font_name='Times New Roman', font_size=9,
            x=window.width - 10, y=10,
            anchor_x='right', anchor_y='bottom',
            color=(180, 180, 180, 200),
        ).draw()

        # centre banner during pause
        if state['paused']:
            pyglet.text.Label(
                state['status_text'],
                font_name='Times New Roman', font_size=14,
                bold=True,
                x=window.width // 2, y=window.height // 2,
                anchor_x='center', anchor_y='center',
                color=(255, 220, 50, 220),
            ).draw()

        car.draw_self()

    # ---- update ----
    def update(dt):
        if state['paused']:
            if time.time() >= state['pause_until']:
                do_reset()
            return

        # AI or human driving
        if agent and not keys_held:
            obs = build_obs(car, _henv, state['next_gate'])
            accel, steer = ACTIONS[agent.act(obs, eval=False)]
            car.accelerate(accel)
            car.turn(steer)

        car.update_self()
        dlines.updatePOI(car.x, car.y)
        dlines.getLinesInBox()
        dlines.getIntesections()
        state['step'] += 1

        cur = (car.x, car.y)

        # off-road check
        if not is_on_road(car.x, car.y):
            end_episode('off-road')
            return

        # gate progression — next_gate wraps so a full circuit back to the
        # starting gate counts as a lap, not just reaching the last gate index.
        if _gate_crossed(state['prev_pos'], cur, state['next_gate']):
            state['gates_hit'] += 1
            state['next_gate']  = (state['next_gate'] + 1) % len(gates)
            if state['gates_hit'] >= len(gates):
                end_episode('lap complete!')
                return

        state['prev_pos'] = cur

    # ---- events ----
    @window.event
    def on_draw():
        draw()

    @window.event
    def on_key_press(symbol, modifiers):
        keys_held.add(symbol)
        if symbol == key.UP:
            car.accelerate(0.05)
        elif symbol == key.LEFT:
            car.turn(-5)
        elif symbol == key.RIGHT:
            car.turn(5)
        elif symbol == key.DOWN:
            car.accelerate(-0.1)
        elif symbol == key.R:
            end_episode('manual reset')
        elif symbol == key.SPACE:
            # instant skip: increment counter, reload checkpoint, reset immediately
            state['episode'] += 1
            if args.policy:
                load_agent(latest_checkpoint(args.models_dir))
            do_reset()

    @window.event
    def on_key_release(symbol, modifiers):
        keys_held.discard(symbol)
        if symbol in (key.UP, key.DOWN):
            car.accelerate(0)
        elif symbol in (key.LEFT, key.RIGHT):
            car.turn(0)

    pyglet.clock.schedule_interval(update, 1 / 60.0)
    pyglet.app.run()


if __name__ == '__main__':
    main()
