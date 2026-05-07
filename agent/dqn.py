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
        lr: float = 1e-3,
        target_sync: int = 1_000,
        grad_clip: float = 1.0,        # used with clip_grad_value_ (cheaper than norm)
        # exploration
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 12_000,
        # misc
        device: str = 'cpu',
    ):
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.policy_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim)

        self._total_steps = 0
        self._learn_steps = 0

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------------

    def epsilon(self):
        progress = min(self._total_steps / self.eps_decay_steps, 1.0)
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
            greedy = self.policy_net(t).argmax(dim=1).cpu().numpy()
        if eval:
            return greedy
        mask = np.random.random(n) < eps
        random_actions = np.random.randint(0, self.n_actions, size=n)
        return np.where(mask, random_actions, greedy)

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

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # clip_grad_value_ only clamps individual parameter gradients —
        # no global norm computation needed, significantly cheaper than
        # clip_grad_norm_ for small networks.
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._learn_steps += 1

        if self._learn_steps % self.target_sync == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'obs_dim':     self.obs_dim,
            'n_actions':   self.n_actions,
            'policy_net':  self.policy_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'total_steps': self._total_steps,
            'learn_steps': self._learn_steps,
        }, path)

    def load(self, path: str):
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
        self._total_steps = ckpt.get('total_steps', 0)
        self._learn_steps = ckpt.get('learn_steps', 0)
