"""
Headless DQN training loop with parallel environments.

Usage:
    .venv/bin/python train.py
    .venv/bin/python train.py --episodes 5000 --n-envs 8 --save-every 100
    .venv/bin/python train.py --load models/dqn_best.pt --episodes 1000
    .venv/bin/python train.py --device cpu  # force CPU
"""

import argparse
import csv
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.car_env import TRAINING_SENSOR_LAYOUT
from env.vec_env import VecEnv
from env.vec_env_shm import ShmVecEnv
from agent.dqn import DQNAgent
from system.track_registry import list_tracks


def parse_args():
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = argparse.ArgumentParser(description='Train DQN car agent')
    p.add_argument('--episodes',    type=int, default=3000,
                   help='number of training episodes (per virtual env)')
    p.add_argument('--n-envs',      type=int, default=4,
                   help='number of parallel environments')
    p.add_argument('--max-steps',   type=int, default=2000,
                   help='max steps per episode')
    p.add_argument('--update-every',type=int, default=4,
                   help='run one gradient update every N env steps')
    p.add_argument('--n-step',      type=int, default=3,
                   help='n-step return horizon (1 = plain 1-step DQN)')
    p.add_argument('--eps-decay',   type=int, default=12_000,
                   help='gradient steps to decay epsilon from 1.0 → 0.05')
    p.add_argument('--reset-eps',   action='store_true',
                   help='reset epsilon to 1.0 after loading a checkpoint (re-explore)')
    p.add_argument('--save-every',  type=int, default=100,
                   help='save checkpoint every N episodes')
    p.add_argument('--seed',        type=str, default=None,
                   help='random seed')
    p.add_argument('--load',        type=str, default=None,
                   help='checkpoint .pt file to resume from')
    p.add_argument('--runs-dir',    type=str, default='runs',
                   help='directory for CSV logs')
    p.add_argument('--models-dir',  type=str, default='models',
                   help='directory for checkpoints')
    p.add_argument('--device',      type=str, default=default_device,
                   help='torch device (cpu / cuda / mps)')
    p.add_argument('--vec-impl',    type=str, default='shm',
                   choices=['pipe', 'shm'],
                   help='VecEnv backend: pipe (original) or shm (async shared-memory)')
    p.add_argument('--rand-start',  action='store_true',
                   help='start each episode at a random gate instead of the fixed start position')
    p.add_argument('--track',       type=str, default='track1',
                   help='which track to train on (any folder in assets/tracks/)')
    args = p.parse_args()
    available = list_tracks()
    if args.track not in available:
        p.error(f'unknown track "{args.track}". Available: {available}')
    return args


class ProgressBar:
    """
    ANSI progress bar that stays at the bottom of terminal output.

    Pattern: the bar is printed without a trailing newline.  Every time a
    normal log line is emitted, we erase the bar (\\r\\033[K), print the line
    with \\n, then redraw the bar on the new current line.  This gives the
    visual effect of the bar always sitting beneath the scrolling log.

    Falls back to plain print() when stdout is not a TTY (e.g. piped to file).
    """

    _FILLED = '█'
    _EMPTY  = '░'

    def __init__(self, total: int):
        self.total   = total
        self.current = 0
        self._t0     = time.perf_counter()
        self._is_tty = sys.stdout.isatty()

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        s = max(0, int(seconds))
        h, rem = divmod(s, 3600)
        m, s   = divmod(rem, 60)
        if h:
            return f'{h}h{m:02d}m{s:02d}s'
        if m:
            return f'{m}m{s:02d}s'
        return f'{s}s'

    def _render(self, sps: float) -> str:
        elapsed = time.perf_counter() - self._t0
        pct     = self.current / max(self.total, 1)

        if pct > 0:
            eta = elapsed * (1.0 - pct) / pct
        else:
            eta = 0.0

        # right-hand stats string (fixed width)
        stats = (
            f' {self.current}/{self.total}'
            f'  {self._fmt_time(elapsed)} elapsed'
            f'  ETA {self._fmt_time(eta)}'
            f'  {sps:,.0f} sps '
        )
        pct_str = f'{pct * 100:5.1f}%'

        try:
            cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        except Exception:
            cols = 80

        bar_width = max(cols - len(pct_str) - len(stats) - 3, 4)
        filled    = round(bar_width * pct)
        bar       = self._FILLED * filled + self._EMPTY * (bar_width - filled)
        return f'{pct_str} |{bar}|{stats}'

    def update(self, current: int, sps: float):
        """Redraw the bar in place (no new log line)."""
        self.current = current
        if not self._is_tty:
            return
        sys.stdout.write(f'\r\033[K{self._render(sps)}')
        sys.stdout.flush()

    def log(self, msg: str, sps: float = 0.0):
        """Print a log line above the bar, then redraw the bar below it."""
        self.update(self.current, sps)   # keep current fresh
        if self._is_tty:
            sys.stdout.write(f'\r\033[K{msg}\n{self._render(sps)}')
            sys.stdout.flush()
        else:
            print(msg, flush=True)

    def close(self, msg: str = ''):
        """Erase the bar and optionally print a final message."""
        if self._is_tty:
            sys.stdout.write(f'\r\033[K')
            if msg:
                sys.stdout.write(msg + '\n')
            sys.stdout.flush()
        elif msg:
            print(msg, flush=True)


def set_seed(seed):
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    args = parse_args()
    set_seed(args.seed)

    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    print(f'Device   : {args.device}')
    print(f'Envs     : {args.n_envs} parallel  [{args.vec_impl}]')

    # ----------------------------------------------------------------
    # Build vectorised environment
    # ----------------------------------------------------------------
    print('Initialising environments (building occupancy grids)...')
    t_init = time.perf_counter()
    _env_kwargs = dict(sensor_layout=TRAINING_SENSOR_LAYOUT, max_steps=args.max_steps,
                       rand_start=args.rand_start, track_id=args.track)
    if args.vec_impl == 'shm':
        vec_env = ShmVecEnv(n_envs=args.n_envs, **_env_kwargs)
    else:
        vec_env = VecEnv(n_envs=args.n_envs, **_env_kwargs)
    print(f'  done in {time.perf_counter() - t_init:.1f}s')

    obs_dim   = vec_env.obs_dim
    n_actions = vec_env.n_actions

    # ----------------------------------------------------------------
    # Build agent
    # ----------------------------------------------------------------
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=args.device,
        eps_decay_steps=args.eps_decay,
        n_step=args.n_step,
    )

    if args.load:
        print(f'Resuming from {args.load}')
        try:
            agent.load(args.load)
        except ValueError as e:
            print(f'[train] {e}')
            print('[train] Starting from a fresh model instead.')
            args.load = None
        if args.reset_eps:
            agent._total_steps = 0
            print(f'  epsilon reset to 1.0 (will decay over {args.eps_decay} gradient steps)')

    # ----------------------------------------------------------------
    # CSV logging
    # ----------------------------------------------------------------
    run_id   = time.strftime('%Y%m%d_%H%M%S')
    csv_path = Path(args.runs_dir) / f'run_{run_id}.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['episode', 'total_reward', 'steps', 'gates_hit',
                     'reason', 'lap_steps', 'lap_time_bonus',
                     'epsilon', 'steps_per_sec'])

    print(f'obs_dim={obs_dim}  n_actions={n_actions}  '
          f'batch={agent.batch_size}  update_every={args.update_every}  '
          f'n_step={args.n_step}')
    print(f'Logs -> {csv_path}')
    print('-' * 70)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    obs = vec_env.reset()               # (n_envs, obs_dim)
    ep_rewards  = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps    = np.zeros(args.n_envs, dtype=np.int32)

    episode_count = 0
    global_steps  = 0
    best_reward   = float('-inf')
    t_start       = time.perf_counter()

    total_episodes = args.episodes * args.n_envs   # wall-clock equivalent
    bar = ProgressBar(total=total_episodes)

    # Whether this backend supports async send/recv pipelining
    _async = hasattr(vec_env, 'send_actions') and hasattr(vec_env, 'recv_results')

    # Pre-send the first batch of actions so the very first recv has results
    if _async:
        actions = agent.act_batch(obs)
        vec_env.send_actions(actions)

    while episode_count < total_episodes:
        if _async:
            # --- async pipeline: learn while workers are stepping ---
            # agent.learn() runs on GPU while workers compute env.step
            if global_steps % args.update_every == 0:
                agent.learn()
            # now collect the results workers computed in parallel
            next_obs, rewards, dones, infos = vec_env.recv_results()
            # immediately fire off the next batch of actions (non-blocking)
            next_actions = agent.act_batch(next_obs)
            vec_env.send_actions(next_actions)
        else:
            # --- synchronous pipeline (pipe backend) ---
            actions      = agent.act_batch(obs)
            next_obs, rewards, dones, infos = vec_env.step(actions)
            next_actions = actions   # used for buffer push below

        for i in range(args.n_envs):
            agent.buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i],
                              env_id=i)

        ep_rewards += rewards
        ep_steps   += 1
        global_steps += args.n_envs

        # --- log completed episodes ---
        for i in np.where(dones)[0]:
            episode_count += 1
            ep_r       = float(ep_rewards[i])
            ep_s       = int(ep_steps[i])
            ep_g       = int(infos[i].get('gates_hit', 0))
            reason     = infos[i].get('reason', 'unknown')
            lap_steps  = infos[i].get('lap_steps', '')
            lap_bonus  = infos[i].get('lap_time_bonus', '')
            eps        = agent.epsilon()
            elapsed    = max(time.perf_counter() - t_start, 1e-6)
            sps        = global_steps / elapsed

            writer.writerow([episode_count, f'{ep_r:.4f}', ep_s, ep_g,
                             reason, lap_steps, lap_bonus,
                             f'{eps:.4f}', f'{sps:.0f}'])

            if episode_count % 10 == 0:
                lap_tag = (f'  lap={lap_steps}st +{float(lap_bonus):.1f}'
                           if reason == 'lap_complete' and lap_bonus != '' else '')
                bar.log(
                    f'Ep {episode_count:5d} | reward {ep_r:8.2f} | '
                    f'steps {ep_s:5d} | gates {ep_g:3d} | '
                    f'{reason:<12s} | eps {eps:.3f} | {sps:.0f} sps{lap_tag}',
                    sps=sps,
                )

            if ep_r > best_reward:
                best_reward = ep_r
                agent.save(os.path.join(args.models_dir, 'dqn_best.pt'))
                bar.log(f'  -> best  {os.path.join(args.models_dir, "dqn_best.pt")}  (reward {ep_r:.2f})', sps=sps)

            if episode_count % args.save_every == 0:
                ckpt = os.path.join(args.models_dir, f'dqn_{episode_count}.pt')
                agent.save(ckpt)
                bar.log(f'  -> saved {ckpt}', sps=sps)

            # reset per-env accumulators
            ep_rewards[i] = 0.0
            ep_steps[i]   = 0

        # redraw bar every step (cheap — just overwrites the current line)
        elapsed = max(time.perf_counter() - t_start, 1e-6)
        bar.update(episode_count, sps=global_steps / elapsed)

        obs     = next_obs
        actions = next_actions   # for next buffer push iteration

        # --- gradient update every N env steps (synchronous path only) ---
        if not _async and global_steps % args.update_every == 0:
            agent.learn()

    # ----------------------------------------------------------------
    # Finalise
    # ----------------------------------------------------------------
    vec_env.close()
    csv_file.close()
    agent.save(os.path.join(args.models_dir, 'dqn_final.pt'))
    elapsed = time.perf_counter() - t_start
    bar.close(
        f'\nTraining complete in {elapsed:.0f}s  |  best reward: {best_reward:.2f}\n'
        f'Final checkpoint: {args.models_dir}/dqn_final.pt'
    )


if __name__ == '__main__':
    # 'spawn' can be used on Windows; 'fork' is Linux default and faster
    mp_ctx = __import__('multiprocessing')
    if mp_ctx.get_start_method(allow_none=True) is None:
        mp_ctx.set_start_method('fork')
    main()
