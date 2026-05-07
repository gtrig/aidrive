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
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.car_env import CarEnv, TRAINING_SENSOR_LAYOUT
from env.vec_env import VecEnv
from agent.dqn import DQNAgent


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
    return p.parse_args()


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

    print(f'Device : {args.device}')
    print(f'Envs   : {args.n_envs} parallel')

    # ----------------------------------------------------------------
    # Build vectorised environment
    # ----------------------------------------------------------------
    print('Initialising environments (building occupancy grids)...')
    t_init = time.perf_counter()
    vec_env = VecEnv(
        n_envs=args.n_envs,
        sensor_layout=TRAINING_SENSOR_LAYOUT,
        max_steps=args.max_steps,
    )
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
          f'batch={agent.batch_size}  update_every={args.update_every}')
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

    while episode_count < total_episodes:
        # --- collect one step across all envs ---
        actions = agent.act_batch(obs)
        next_obs, rewards, dones, infos = vec_env.step(actions)

        for i in range(args.n_envs):
            agent.buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

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
                lap_tag = (f'  lap={lap_steps}st +{lap_bonus:.1f}'
                           if reason == 'lap_complete' else '')
                print(
                    f'Ep {episode_count:5d} | reward {ep_r:8.2f} | '
                    f'steps {ep_s:5d} | gates {ep_g:3d} | '
                    f'{reason:<12s} | eps {eps:.3f} | {sps:.0f} sps{lap_tag}'
                )

            if ep_r > best_reward:
                best_reward = ep_r
                agent.save(os.path.join(args.models_dir, 'dqn_best.pt'))

            if episode_count % args.save_every == 0:
                ckpt = os.path.join(args.models_dir, f'dqn_{episode_count}.pt')
                agent.save(ckpt)
                print(f'  -> saved {ckpt}')

            # reset per-env accumulators
            ep_rewards[i] = 0.0
            ep_steps[i]   = 0

        obs = next_obs

        # --- gradient update every N env steps ---
        if global_steps % args.update_every == 0:
            agent.learn()

    # ----------------------------------------------------------------
    # Finalise
    # ----------------------------------------------------------------
    vec_env.close()
    csv_file.close()
    agent.save(os.path.join(args.models_dir, 'dqn_final.pt'))
    elapsed = time.perf_counter() - t_start
    print(f'\nTraining complete in {elapsed:.0f}s  |  best reward: {best_reward:.2f}')
    print(f'Final checkpoint: {args.models_dir}/dqn_final.pt')


if __name__ == '__main__':
    # 'spawn' can be used on Windows; 'fork' is Linux default and faster
    mp_ctx = __import__('multiprocessing')
    if mp_ctx.get_start_method(allow_none=True) is None:
        mp_ctx.set_start_method('fork')
    main()
