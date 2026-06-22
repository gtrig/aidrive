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
import threading
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

# Median episode length from runs/*.csv (~1.1k–1.9k steps); used to size ε decay
# instead of max_steps (2k), which made exploration collapse too early.
AVG_EP_STEPS = 1500
# Best session (run_20260509_171847) kept ε≥0.30 and reached ~44 gates/ep; runs that
# hit ε=0.05 early stagnated below 3 gates/ep.
EPS_DECAY_FRACTION = 0.55


def _is_better_episode(ep_g: int, ep_r: float, best_g: int, best_r: float) -> bool:
    """Prefer more gates; break ties with reward. Ignore 0-gate regressions."""
    if ep_g > best_g:
        return True
    if ep_g == best_g and ep_g > 0 and ep_r > best_r:
        return True
    return False


def evaluate_agent(agent: DQNAgent, env_kwargs: dict, n_episodes: int = 5) -> tuple[float, int]:
    """Greedy rollouts (no ε) — measures actual policy quality."""
    from env.car_env import CarEnv

    env = CarEnv(**env_kwargs)
    gates = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action = agent.act(obs, eval=True)
            obs, _, done, info = env.step(action)
        gates.append(int(info.get('gates_hit', 0)))
    return float(np.mean(gates)), max(gates)


def _resolve_checkpoint(models_dir: str, explicit: str | None) -> str | None:
    """Pick checkpoint: explicit --load, else dqn_best.pt, else dqn_final.pt."""
    if explicit:
        return explicit
    root = Path(models_dir)
    for name in ('dqn_best.pt', 'dqn_final.pt'):
        path = root / name
        if path.is_file():
            return str(path)
    return None


def parse_args():
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = argparse.ArgumentParser(description='Train DQN car agent')
    p.add_argument('--episodes',    type=int, default=3000,
                   help='number of training episodes (per virtual env)')
    p.add_argument('--n-envs',      type=int, default=8,
                   help='number of parallel environments')
    p.add_argument('--max-steps',   type=int, default=2000,
                   help='max steps per episode')
    p.add_argument('--update-every',type=int, default=4,
                   help='run one gradient update every N env steps')
    p.add_argument('--n-step',      type=int, default=3,
                   help='n-step return horizon (1 = plain 1-step DQN)')
    p.add_argument('--eps-decay',   type=int, default=None,
                   help='env steps to decay epsilon from 1.0 → eps-end '
                        '(default: ~55%% of expected total env steps for this run)')
    p.add_argument('--eps-end',     type=float, default=0.20,
                   help='minimum exploration rate (best runs stayed ≥0.30)')
    p.add_argument('--lr',          type=float, default=5e-4,
                   help='Adam learning rate')
    p.add_argument('--reset-eps',   action='store_true',
                   help='reset epsilon to 1.0 after loading a checkpoint (re-explore)')
    p.add_argument('--save-every',  type=int, default=100,
                   help='save checkpoint every N episodes')
    p.add_argument('--eval-every',  type=int, default=100,
                   help='run greedy eval every N completed episodes')
    p.add_argument('--eval-episodes', type=int, default=5,
                   help='greedy episodes per eval (used for dqn_best.pt)')
    p.add_argument('--seed',        type=str, default=None,
                   help='random seed')
    p.add_argument('--load',        type=str, default=None,
                   help='checkpoint .pt file to resume from (weights, optimizer, '
                        'exploration step count, CSV episode index, best metrics)')
    p.add_argument('--fresh',       action='store_true',
                   help='train from scratch (do not auto-load dqn_best.pt)')
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
                   help='start each episode at a random gate (use after basic driving works)')
    p.add_argument('--track',       type=str, default='track1',
                   help='which track to train on (any folder in assets/tracks/)')
    p.add_argument('--buffer-capacity', type=int, default=100_000,
                   help='replay buffer size (larger helps long runs; more RAM)')
    args = p.parse_args()
    available = list_tracks()
    if args.track not in available:
        p.error(f'unknown track "{args.track}". Available: {available}')

    # Size ε decay from recorded session episode lengths, not max_steps ceiling.
    if args.eps_decay is None:
        est_env_steps = args.episodes * args.n_envs * AVG_EP_STEPS
        args.eps_decay = max(100_000, int(est_env_steps * EPS_DECAY_FRACTION))

    if not args.fresh:
        args.load = _resolve_checkpoint(args.models_dir, args.load)

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


class AsyncSaver:
    """Write checkpoints on a background thread to avoid I/O spikes."""

    def __init__(self):
        self._thread: threading.Thread | None = None

    def save(self, agent: DQNAgent, path: str, **kwargs):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = threading.Thread(
            target=agent.save,
            args=(path,),
            kwargs=kwargs,
            daemon=True,
        )
        self._thread.start()

    def flush(self):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()


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
    _use_rand_start = args.rand_start
    _env_kwargs = dict(sensor_layout=TRAINING_SENSOR_LAYOUT, max_steps=args.max_steps,
                       rand_start=_use_rand_start, track_id=args.track)
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
        lr=args.lr,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay,
        n_step=args.n_step,
        buffer_capacity=args.buffer_capacity,
    )

    resume_base = 0
    if args.load:
        print(f'Resuming from {args.load}')
        try:
            agent.load(args.load)
            resume_base = agent.resume_global_episode
            bg = agent.resume_best_gates
            br = agent.resume_best_reward
            print(f'  checkpoint: completed_episodes={resume_base}  '
                  f'learn_steps={agent._learn_steps}  epsilon={agent.epsilon():.4f}  '
                  f'best_gates={bg if bg is not None else "?"}  '
                  f'best_reward={br if br is not None else "?"}')
        except ValueError as e:
            print(f'[train] {e}')
            print('[train] Starting from a fresh model instead.')
            args.load = None
        if args.reset_eps:
            agent._total_steps = 0
            agent._learn_steps = 0
            agent._explore_env_steps = 0
            print(f'  epsilon reset to 1.0 (will decay over {args.eps_decay} env steps)')
        elif agent._explore_env_steps > 0:
            # Extend ε horizon so resume does not snap to eps_end when the new
            # run's eps_decay is shorter than steps already explored.
            extended = agent._explore_env_steps + int(
                args.episodes * args.n_envs * AVG_EP_STEPS * EPS_DECAY_FRACTION
            )
            if extended > args.eps_decay:
                args.eps_decay = extended
                agent.eps_decay_steps = extended
                print(f'  extended eps_decay to {args.eps_decay} env steps for resume')

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
          f'batch={agent.batch_size}  buffer={args.buffer_capacity}  '
          f'update_every={args.update_every}  n_step={args.n_step}  '
          f'lr={args.lr}  eps_end={args.eps_end}  eps_decay={args.eps_decay}')
    print(f'Logs -> {csv_path}')
    print('-' * 70)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    obs = vec_env.reset()               # (n_envs, obs_dim)
    ep_rewards  = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps    = np.zeros(args.n_envs, dtype=np.int32)

    episode_count = resume_base
    global_steps  = agent._explore_env_steps if args.load else 0
    best_gates    = (
        agent.resume_best_gates
        if agent.resume_best_gates is not None
        else -1
    )
    best_reward   = (
        agent.resume_best_reward
        if agent.resume_best_reward is not None
        else float('-inf')
    )
    best_eval_gates = (
        agent.resume_best_eval_gates
        if agent.resume_best_eval_gates is not None
        else -1.0
    )
    t_start       = time.perf_counter()
    ckpt_saver    = AsyncSaver()

    total_episodes = args.episodes * args.n_envs   # completions in this session
    episode_goal   = resume_base + total_episodes
    bar = ProgressBar(total=total_episodes)
    _shm_views = args.vec_impl == 'shm'

    # Whether this backend supports async send/recv pipelining
    _async = hasattr(vec_env, 'send_actions') and hasattr(vec_env, 'recv_results')

    # Pre-send the first batch of actions so the very first recv has results
    if _async:
        actions = agent.act_batch(obs)
        vec_env.send_actions(actions)

    while episode_count < episode_goal:
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

        agent.buffer.push_batch(obs, actions, rewards, next_obs, dones)

        ep_rewards += rewards
        ep_steps   += 1
        global_steps += args.n_envs
        agent.set_exploration_env_steps(global_steps)

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

            if episode_count % args.eval_every == 0:
                eval_mean, eval_max = evaluate_agent(
                    agent, _env_kwargs, n_episodes=args.eval_episodes,
                )
                if eval_mean > best_eval_gates:
                    best_eval_gates = eval_mean
                    best_gates = eval_max
                    best_reward = ep_r
                    ckpt_saver.save(
                        agent,
                        os.path.join(args.models_dir, 'dqn_best.pt'),
                        global_episode=episode_count,
                        best_reward=best_reward,
                        best_gates=best_gates,
                        best_eval_gates=best_eval_gates,
                    )
                    bar.log(
                        f'  -> best  {os.path.join(args.models_dir, "dqn_best.pt")}  '
                        f'(eval gates {eval_mean:.1f} avg / {eval_max} max)',
                        sps=sps,
                    )

            if episode_count % args.save_every == 0:
                ckpt = os.path.join(args.models_dir, f'dqn_{episode_count}.pt')
                ckpt_saver.save(
                    agent,
                    ckpt,
                    global_episode=episode_count,
                    best_reward=best_reward,
                    best_gates=best_gates,
                    best_eval_gates=best_eval_gates,
                )
                bar.log(f'  -> saved {ckpt}', sps=sps)

            # reset per-env accumulators
            ep_rewards[i] = 0.0
            ep_steps[i]   = 0

        if episode_count > resume_base or global_steps % 50 == 0:
            elapsed = max(time.perf_counter() - t_start, 1e-6)
            bar.update(episode_count - resume_base, sps=global_steps / elapsed)

        # Retain a stable obs buffer when recv returns shared-memory views.
        obs     = np.array(next_obs, copy=True) if _shm_views else next_obs
        actions = next_actions   # for next buffer push iteration

        # --- gradient update every N env steps (synchronous path only) ---
        if not _async and global_steps % args.update_every == 0:
            agent.learn()

    # ----------------------------------------------------------------
    # Finalise
    # ----------------------------------------------------------------
    vec_env.close()
    csv_file.close()
    ckpt_saver.flush()
    eval_mean, eval_max = evaluate_agent(
        agent, _env_kwargs, n_episodes=args.eval_episodes,
    )
    agent.save(
        os.path.join(args.models_dir, 'dqn_final.pt'),
        global_episode=episode_count,
        best_reward=best_reward,
        best_gates=best_gates,
        best_eval_gates=max(best_eval_gates, eval_mean),
    )
    elapsed = time.perf_counter() - t_start
    bar.close(
        f'\nTraining complete in {elapsed:.0f}s  |  '
        f'best eval gates: {max(best_eval_gates, eval_mean):.1f}  '
        f'final eval: {eval_mean:.1f} avg / {eval_max} max\n'
        f'Final checkpoint: {args.models_dir}/dqn_final.pt'
    )


if __name__ == '__main__':
    # 'spawn' can be used on Windows; 'fork' is Linux default and faster
    mp_ctx = __import__('multiprocessing')
    if mp_ctx.get_start_method(allow_none=True) is None:
        mp_ctx.set_start_method('fork')
    main()
