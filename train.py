"""
PPO training loop with parallel environments.

Usage:
    .venv/bin/python train.py
    .venv/bin/python train.py --n-envs 8 --horizon 2048 --total-steps 5_000_000
    .venv/bin/python train.py --load models/ppo_best.pt --total-steps 2_000_000
    .venv/bin/python train.py --track track1 --rand-start
    .venv/bin/python train.py --fresh                 # ignore existing checkpoint
"""

import argparse
import csv
import math
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
from agent.ppo import PPOAgent
from agent.rollout_buffer import RolloutBuffer
from system.track_registry import list_tracks


# ──────────────────────────────────────────────────────────────────────────────
# Greedy evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    agent: PPOAgent,
    env_kwargs: dict,
    n_episodes: int = 5,
) -> tuple[float, float]:
    """Run n_episodes stochastic rollouts; return (mean_score, max_score).

    Score is the episode total reward.  We use the stochastic policy rather
    than greedy argmax because until the policy has sharpened, argmax on a
    near-uniform distribution picks the same action every step and crashes
    immediately.
    """
    from env.car_env import CarEnv

    env = CarEnv(**env_kwargs)
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_score = 0.0
        while not done:
            action = int(agent.act_batch(obs[None])[0][0])
            obs, reward, done, _info = env.step(action)
            ep_score += float(reward)
        scores.append(ep_score)
    return float(np.mean(scores)), float(max(scores))


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = argparse.ArgumentParser(description='Train PPO car agent')

    # environment
    p.add_argument('--track',       type=str,   default='track1',
                   help='track name (folder in assets/tracks/)')
    p.add_argument('--n-envs',      type=int,   default=8,
                   help='number of parallel environments')
    p.add_argument('--max-steps',   type=int,   default=2000,
                   help='max steps per episode')
    p.add_argument('--rand-start',  action='store_true',
                   help='randomise start position each episode')

    # PPO rollout
    p.add_argument('--total-steps', type=int,   default=5_000_000,
                   help='total environment steps to train for')
    p.add_argument('--horizon',     type=int,   default=2048,
                   help='rollout horizon (steps per env per update)')

    # PPO update
    p.add_argument('--epochs',      type=int,   default=4,
                   help='number of PPO update epochs per rollout')
    p.add_argument('--minibatches', type=int,   default=4,
                   help='number of minibatches per epoch')
    p.add_argument('--clip',        type=float, default=0.2,
                   help='PPO clip coefficient')
    p.add_argument('--gae-lambda',  type=float, default=0.95,
                   help='GAE lambda')
    p.add_argument('--gamma',       type=float, default=0.99,
                   help='discount factor')
    p.add_argument('--ent-coef',    type=float, default=0.10,
                   help='entropy bonus coefficient')
    p.add_argument('--lr',          type=float, default=3e-4,
                   help='Adam learning rate (linearly decayed to 0 over total-steps)')
    p.add_argument('--max-grad-norm', type=float, default=0.5,
                   help='gradient norm clipping')
    p.add_argument('--anneal-lr',   action='store_true', default=True,
                   help='linearly anneal learning rate to 0 over total-steps (default: on)')
    p.add_argument('--no-anneal-lr', dest='anneal_lr', action='store_false',
                   help='disable learning rate annealing')

    # misc
    p.add_argument('--seed',        type=str,   default=None)
    p.add_argument('--device',      type=str,   default=default_device)
    p.add_argument('--load',        type=str,   default=None,
                   help='.pt checkpoint to resume from')
    p.add_argument('--fresh',       action='store_true',
                   help='train from scratch, do not auto-load existing checkpoint')
    p.add_argument('--save-every',  type=int,   default=100,
                   help='save periodic checkpoint every N episodes')
    p.add_argument('--eval-every',  type=int,   default=100,
                   help='run greedy eval every N completed episodes')
    p.add_argument('--eval-episodes', type=int, default=5,
                   help='episodes per greedy eval')
    p.add_argument('--runs-dir',    type=str,   default='runs')
    p.add_argument('--models-dir',  type=str,   default='models')

    args = p.parse_args()

    available = list_tracks()
    if args.track not in available:
        p.error(f'unknown track "{args.track}". Available: {available}')

    # Auto-load best checkpoint unless --fresh
    if not args.fresh and args.load is None:
        candidate = Path(args.models_dir) / 'ppo_best.pt'
        if candidate.is_file():
            args.load = str(candidate)

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Progress bar (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

class ProgressBar:
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
        eta     = elapsed * (1.0 - pct) / pct if pct > 0 else 0.0
        stats   = (
            f' {self.current:,}/{self.total:,}'
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
        self.current = current
        if not self._is_tty:
            return
        sys.stdout.write(f'\r\033[K{self._render(sps)}')
        sys.stdout.flush()

    def log(self, msg: str, sps: float = 0.0):
        self.update(self.current, sps)
        if self._is_tty:
            sys.stdout.write(f'\r\033[K{msg}\n{self._render(sps)}')
            sys.stdout.flush()
        else:
            print(msg, flush=True)

    def close(self, msg: str = ''):
        if self._is_tty:
            sys.stdout.write('\r\033[K')
            if msg:
                sys.stdout.write(msg + '\n')
            sys.stdout.flush()
        elif msg:
            print(msg, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Async checkpoint saver
# ──────────────────────────────────────────────────────────────────────────────

class AsyncSaver:
    def __init__(self):
        self._thread: threading.Thread | None = None

    def save(self, agent: PPOAgent, path: str, **kwargs):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = threading.Thread(
            target=agent.save, args=(path,), kwargs=kwargs, daemon=True
        )
        self._thread.start()

    def flush(self):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.seed is not None:
        s = int(args.seed)
        random.seed(s); np.random.seed(s); torch.manual_seed(s)

    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    print(f'Device   : {args.device}')
    print(f'Envs     : {args.n_envs}  horizon={args.horizon}  '
          f'total_steps={args.total_steps:,}')

    # ── Vectorised environment ────────────────────────────────────────────────
    print('Initialising environments…')
    t_init = time.perf_counter()
    env_kwargs = dict(
        sensor_layout=TRAINING_SENSOR_LAYOUT,
        max_steps=args.max_steps,
        rand_start=args.rand_start,
        track_id=args.track,
    )
    vec_env = VecEnv(n_envs=args.n_envs, **env_kwargs)
    obs_dim   = vec_env.obs_dim
    n_actions = vec_env.n_actions
    print(f'  done in {time.perf_counter() - t_init:.1f}s  '
          f'obs_dim={obs_dim}  n_actions={n_actions}')

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=args.lr,
        clip_coef=args.clip,
        ent_coef=args.ent_coef,
        update_epochs=args.epochs,
        n_minibatches=args.minibatches,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )

    resume_episode = 0
    if args.load:
        print(f'Resuming from {args.load}')
        try:
            agent.load(args.load)
            resume_episode = agent.resume_global_episode
            print(f'  checkpoint: episodes={resume_episode}  '
                  f'updates={agent._total_updates}  '
                  f'env_steps={agent._total_env_steps:,}')
        except (ValueError, Exception) as e:
            print(f'[train] Could not load checkpoint: {e}')
            print('[train] Starting from scratch.')

    # ── CSV logging ───────────────────────────────────────────────────────────
    run_id   = time.strftime('%Y%m%d_%H%M%S')
    csv_path = Path(args.runs_dir) / f'run_{run_id}.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow([
        'episode', 'total_reward', 'steps', 'gates_hit',
        'reason', 'lap_steps',
        'pg_loss', 'v_loss', 'entropy', 'clip_frac',
        'env_steps', 'steps_per_sec',
    ])
    print(f'Logs  -> {csv_path}')
    print('-' * 70)

    # ── Rollout buffer ────────────────────────────────────────────────────────
    rollout = RolloutBuffer(
        horizon=args.horizon, n_envs=args.n_envs, obs_dim=obs_dim
    )

    # ── Training state ────────────────────────────────────────────────────────
    obs = vec_env.reset()                          # (n_envs, obs_dim)
    ep_rewards  = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps    = np.zeros(args.n_envs, dtype=np.int32)

    episode_count  = resume_episode
    global_steps   = agent._total_env_steps
    best_score: float = (
        float(agent.resume_best_score)
        if agent.resume_best_score is not None else float('-inf')
    )
    best_ckpt_path = Path(args.models_dir) / 'ppo_best.pt'

    def save_if_best(score: float, episode: int, label: str, sps: float) -> bool:
        """Save ppo_best.pt when score beats the running peak."""
        nonlocal best_score
        if score <= best_score:
            return False
        best_score = score
        ckpt_saver.save(
            agent,
            str(best_ckpt_path),
            global_episode=episode,
            best_score=best_score,
        )
        bar.log(f'  -> best  ppo_best.pt  ({label} score {score:.1f})', sps=sps)
        return True

    next_eval_episode = (
        ((resume_episode // args.eval_every) + 1) * args.eval_every
        if resume_episode > 0 else args.eval_every
    )

    t_start    = time.perf_counter()
    ckpt_saver = AsyncSaver()
    bar        = ProgressBar(total=args.total_steps)

    # most recent PPO stats (displayed in log lines between updates)
    last_stats: dict = {}

    while global_steps < args.total_steps:
        # ── LR annealing ──────────────────────────────────────────────────────
        if args.anneal_lr:
            frac = max(0.0, 1.0 - global_steps / args.total_steps)
            agent.set_lr(args.lr * frac)

        # ── Collect one rollout ───────────────────────────────────────────────
        for step in range(args.horizon):
            # Update obs normaliser stats before normalising
            agent.update_obs_stats(obs)

            actions, logps, values = agent.act_batch(obs)
            next_obs, rewards, dones, infos = vec_env.step(actions)

            rollout.add(step, obs, actions, logps, values, rewards, dones)

            ep_rewards += rewards
            ep_steps   += 1
            global_steps += args.n_envs
            agent._total_env_steps = global_steps

            elapsed = max(time.perf_counter() - t_start, 1e-6)
            sps     = global_steps / elapsed

            # Log completed episodes
            for i in np.where(dones)[0]:
                episode_count += 1
                ep_r   = float(ep_rewards[i])
                ep_s   = int(ep_steps[i])
                ep_g   = int(infos[i].get('gates_hit', 0))
                reason = infos[i].get('reason', 'unknown')
                lap_s  = infos[i].get('lap_steps', '')

                writer.writerow([
                    episode_count, f'{ep_r:.4f}', ep_s, ep_g, reason, lap_s,
                    f'{last_stats.get("pg_loss", 0):.5f}',
                    f'{last_stats.get("v_loss", 0):.5f}',
                    f'{last_stats.get("entropy", 0):.4f}',
                    f'{last_stats.get("clip_frac", 0):.4f}',
                    global_steps, f'{sps:.0f}',
                ])

                if episode_count % 10 == 0:
                    lap_tag = f'  lap={lap_s}st' if reason == 'lap_complete' and lap_s != '' else ''
                    bar.log(
                        f'Ep {episode_count:5d} | reward {ep_r:8.2f} | '
                        f'steps {ep_s:4d} | gates {ep_g:3d} | '
                        f'{reason:<12s} | {sps:.0f} sps{lap_tag}',
                        sps=sps,
                    )

                # Save whenever a rollout episode sets a new peak score.
                save_if_best(ep_r, episode_count, 'episode', sps)

                # Periodic save
                if episode_count % args.save_every == 0:
                    ckpt = os.path.join(args.models_dir, f'ppo_{episode_count}.pt')
                    ckpt_saver.save(
                        agent, ckpt,
                        global_episode=episode_count,
                        best_score=best_score,
                    )
                    bar.log(f'  -> saved {ckpt}', sps=sps)

                ep_rewards[i] = 0.0
                ep_steps[i]   = 0

            bar.update(global_steps, sps=sps)
            obs = next_obs

        # ── PPO update ────────────────────────────────────────────────────────
        last_values = agent.get_value(obs)
        rollout.compute_gae(last_values, gamma=args.gamma, lam=args.gae_lambda)
        last_stats = agent.update(rollout)

        # Periodic eval — save new peaks; keep training on current weights so
        # PPO can continue learning (mid-training rollback froze progress by
        # resetting to an early lucky best every eval).
        while episode_count >= next_eval_episode:
            eval_mean, eval_max = evaluate_agent(
                agent, env_kwargs, n_episodes=args.eval_episodes
            )
            elapsed = max(time.perf_counter() - t_start, 1e-6)
            eval_sps = global_steps / elapsed
            if not save_if_best(eval_max, episode_count, 'eval', eval_sps):
                bar.log(
                    f'  eval {eval_mean:.1f} avg / {eval_max:.1f} max  '
                    f'(best {best_score:.1f})',
                    sps=eval_sps,
                )
            next_eval_episode += args.eval_every

        # Warn if entropy collapses — indicates ent_coef may need to increase
        ent = last_stats.get('entropy', 0.0)
        ent_max = math.log(n_actions)
        if ent < 0.2 * ent_max:
            elapsed = max(time.perf_counter() - t_start, 1e-6)
            bar.log(
                f'  [WARN] entropy collapse: {ent:.3f} / {ent_max:.3f} '
                f'({100*ent/ent_max:.0f}% of max) — consider --ent-coef higher',
                sps=global_steps / elapsed,
            )

    # ── Finalise ──────────────────────────────────────────────────────────────
    vec_env.close()
    csv_file.close()
    ckpt_saver.flush()

    # Always finish from the best checkpoint, not the last (possibly worse) weights.
    if best_ckpt_path.is_file():
        agent.load_weights(str(best_ckpt_path))

    elapsed = time.perf_counter() - t_start
    eval_mean, eval_max = evaluate_agent(
        agent, env_kwargs, n_episodes=args.eval_episodes
    )
    agent.save(
        os.path.join(args.models_dir, 'ppo_final.pt'),
        global_episode=episode_count,
        best_score=max(best_score, eval_max),
    )
    bar.close(
        f'\nTraining complete in {elapsed:.0f}s  |  '
        f'best peak score: {max(best_score, eval_max):.1f}  '
        f'final eval: {eval_mean:.1f} avg / {eval_max:.1f} max\n'
        f'Final checkpoint: {args.models_dir}/ppo_final.pt'
    )


if __name__ == '__main__':
    import multiprocessing as _mp
    if _mp.get_start_method(allow_none=True) is None:
        _mp.set_start_method('fork')
    main()
