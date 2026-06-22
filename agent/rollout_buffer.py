"""
Fixed-horizon rollout buffer for on-policy PPO training.

Stores one rollout of `horizon` steps across `n_envs` parallel environments,
computes Generalised Advantage Estimation (GAE-Lambda), and exposes the data
as flat numpy arrays ready for minibatch PPO updates.

Usage:
    buf = RolloutBuffer(horizon=2048, n_envs=8, obs_dim=11)
    # inside rollout collection loop:
    buf.add(step, obs, actions, logps, values, rewards, dones)
    # after collection is complete:
    buf.compute_gae(last_values, gamma=0.99, lam=0.95)
    # then pass buf to PPOAgent.update(buf)
"""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, horizon: int, n_envs: int, obs_dim: int):
        self.horizon  = horizon
        self.n_envs   = n_envs
        self.obs_dim  = obs_dim

        # Pre-allocate; all shaped (horizon, n_envs, ...)
        self.obs        = np.zeros((horizon, n_envs, obs_dim), dtype=np.float32)
        self.actions    = np.zeros((horizon, n_envs),          dtype=np.int32)
        self.logps      = np.zeros((horizon, n_envs),          dtype=np.float32)
        self.values     = np.zeros((horizon, n_envs),          dtype=np.float32)
        self.rewards    = np.zeros((horizon, n_envs),          dtype=np.float32)
        self.dones      = np.zeros((horizon, n_envs),          dtype=np.float32)

        # Filled by compute_gae()
        self.advantages = np.zeros((horizon, n_envs),          dtype=np.float32)
        self.returns    = np.zeros((horizon, n_envs),          dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────

    def add(
        self,
        step:    int,
        obs:     np.ndarray,   # (n_envs, obs_dim)
        actions: np.ndarray,   # (n_envs,)
        logps:   np.ndarray,   # (n_envs,)
        values:  np.ndarray,   # (n_envs,)
        rewards: np.ndarray,   # (n_envs,)
        dones:   np.ndarray,   # (n_envs,) bool or float
    ):
        self.obs[step]     = obs
        self.actions[step] = actions
        self.logps[step]   = logps
        self.values[step]  = values
        self.rewards[step] = rewards
        self.dones[step]   = dones.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        last_values: np.ndarray,  # (n_envs,) bootstrap value at step T
        gamma: float = 0.99,
        lam:   float = 0.95,
    ):
        """Compute GAE advantages and target returns in-place.

        GAE(γ, λ):
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + (γλ) * (1 - done_t) * A_{t+1}
        Returns = Advantages + Values (used as regression targets for V).
        """
        next_value   = last_values.copy()   # (n_envs,)
        next_adv     = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.horizon)):
            not_done   = 1.0 - self.dones[t]
            delta      = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            next_adv   = delta + gamma * lam * not_done * next_adv
            self.advantages[t] = next_adv
            next_value = self.values[t]

        self.returns = self.advantages + self.values
