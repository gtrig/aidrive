"""
DQN agent: Q-network, target network, epsilon-greedy policy, training step.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from agent.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        # replay buffer — needs obs_dim for numpy pre-allocation
        buffer_capacity: int = 50_000,
        batch_size: int = 256,         # larger batches saturate GPU better
        # training
        gamma: float = 0.99,
        lr: float = 5e-4,
        # Soft target (Polyak) averaging — more stable than periodic hard sync
        # on long runs.  Set target_tau=0 to fall back to hard sync only.
        target_tau: float = 0.005,
        target_sync: int = 0,        # if >0 and target_tau==0: hard-copy every N learns
        grad_clip: float = 1.0,      # used with clip_grad_value_ (cheaper than norm)
        # exploration
        eps_start: float = 1.0,
        eps_end: float = 0.20,
        eps_decay_steps: int = 12_000,
        # n-step returns
        n_step: int = 1,
        # misc
        device: str = 'cpu',
    ):
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.target_sync = target_sync
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.eps_start = eps_start
        self.eps_end = eps_end
        # Env-step horizon for ε decay (set from train.py from session statistics).
        self.eps_decay_steps = eps_decay_steps
        self.n_step = n_step

        self.policy_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim, n_step=n_step, gamma=gamma)

        self._total_steps = 0
        self._learn_steps = 0
        self._explore_env_steps = 0
        # Filled by load(); train uses for episode / best-metric resume
        self.resume_global_episode = 0
        self.resume_best_reward: float | None = None
        self.resume_best_gates: int | None = None
        self.resume_best_eval_gates: float | None = None

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------------

    def set_exploration_env_steps(self, env_steps: int):
        """Drive ε from parallel env steps (not gradient steps)."""
        self._explore_env_steps = int(env_steps)

    def epsilon(self):
        progress = min(self._explore_env_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def act(self, obs: np.ndarray, eval: bool = False) -> int:
        """Single observation → single action."""
        if not eval and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            return int(self.policy_net(t).argmax(dim=1).item())

    def act_batch(self, obs: np.ndarray, eval: bool = False) -> np.ndarray:
        """Batch of observations (n_envs, obs_dim) → array of actions (n_envs,).
        Exploration is applied independently per environment."""
        n = len(obs)
        eps = self.epsilon() if not eval else 0.0
        with torch.no_grad():
            t = torch.from_numpy(obs).to(self.device)
            greedy = self.policy_net(t).argmax(dim=1)
            if eval or eps <= 0.0:
                return greedy.cpu().numpy()
            explore = torch.rand(n, device=self.device) < eps
            random_actions = torch.randint(
                0, self.n_actions, (n,), device=self.device,
            )
            actions = torch.where(explore, random_actions, greedy)
            return actions.cpu().numpy()

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def learn(self):
        """Sample one mini-batch and perform one gradient update.
        Returns the scalar loss, or None if the buffer is not full enough."""
        if len(self.buffer) < self.batch_size:
            return None

        self._total_steps += 1

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, device=str(self.device)
        )

        # Single policy forward for current and next states (Double DQN).
        batch = states.shape[0]
        all_q = self.policy_net(torch.cat([states, next_states], dim=0))
        q_values = all_q[:batch].gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = all_q[batch:].argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            # n-step gamma: γ^n is already baked into the n-step return stored in the
            # buffer, so we only need γ^n for the bootstrap term.
            gamma_n = self.gamma ** self.n_step
            targets = rewards + gamma_n * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # clip_grad_value_ only clamps individual parameter gradients —
        # no global norm computation needed, significantly cheaper than
        # clip_grad_norm_ for small networks.
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._learn_steps += 1

        with torch.no_grad():
            if self.target_tau > 0.0:
                tau = self.target_tau
                for tp, pp in zip(self.target_net.parameters(),
                                  self.policy_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)
            elif self.target_sync > 0 and self._learn_steps % self.target_sync == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, *, global_episode: int | None = None,
             best_reward: float | None = None, best_gates: int | None = None,
             best_eval_gates: float | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'obs_dim':     self.obs_dim,
            'n_actions':   self.n_actions,
            'n_step':      self.n_step,
            'policy_net':  self.policy_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'total_steps':       self._total_steps,
            'learn_steps':       self._learn_steps,
            'explore_env_steps': self._explore_env_steps,
        }
        if global_episode is not None:
            payload['global_episode'] = int(global_episode)
        if best_reward is not None and best_reward > float('-inf'):
            payload['best_reward'] = float(best_reward)
        if best_gates is not None:
            payload['best_gates'] = int(best_gates)
        if best_eval_gates is not None:
            payload['best_eval_gates'] = float(best_eval_gates)
        torch.save(payload, path)

    def load(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        ckpt_obs_dim = ckpt.get('obs_dim')
        ckpt_n_actions = ckpt.get('n_actions')
        if ckpt_obs_dim is not None and int(ckpt_obs_dim) != int(self.obs_dim):
            raise ValueError(
                f'Checkpoint obs_dim={ckpt_obs_dim} does not match current env '
                f'obs_dim={self.obs_dim}. Sensor layout changed; train/load a '
                f'compatible model.'
            )
        if ckpt_n_actions is not None and int(ckpt_n_actions) != int(self.n_actions):
            raise ValueError(
                f'Checkpoint n_actions={ckpt_n_actions} does not match current env '
                f'n_actions={self.n_actions}. Action space changed; train/load a '
                f'compatible model.'
            )
        try:
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
        except RuntimeError as e:
            raise ValueError(
                'Checkpoint tensor shapes are incompatible with current model '
                f'(obs_dim={self.obs_dim}, n_actions={self.n_actions}). '
                'This usually means the sensor layout changed.'
            ) from e
        self.optimizer.load_state_dict(ckpt['optimizer'])
        ls = int(ckpt.get('learn_steps', 0))
        ts = ckpt.get('total_steps')
        if ts is None:
            ts = ls
        else:
            ts = int(ts)
        self._learn_steps = ls
        self._total_steps = max(ts, ls)
        # ε is env-step based; fall back for checkpoints saved before this field.
        self._explore_env_steps = int(
            ckpt.get('explore_env_steps', self._total_steps * 8)
        )

        self.resume_global_episode = int(ckpt.get('global_episode', 0))
        br = ckpt.get('best_reward')
        self.resume_best_reward = float(br) if br is not None else None
        bg = ckpt.get('best_gates')
        self.resume_best_gates = int(bg) if bg is not None else None
        beg = ckpt.get('best_eval_gates')
        self.resume_best_eval_gates = float(beg) if beg is not None else None
