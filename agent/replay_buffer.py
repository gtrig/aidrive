"""
Fixed-size circular replay buffer backed by pre-allocated numpy arrays.
Sampling uses np.random.choice which is ~5x faster than random.sample on a
Python list and avoids repeated np.array() conversions per sample.

Supports n-step returns: set n_step > 1 to buffer n transitions per env before
storing the compressed (s, a, R_n, s_n, done_n) tuple in the main ring buffer.
The n-step discounted return is:
    R_n = r_0 + γ*r_1 + γ²*r_2 + … + γ^(n-1)*r_{n-1}
and next_state / done refer to the state n steps ahead.
"""

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_step: int = 1, gamma: float = 0.99):
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.n_step   = n_step
        self.gamma    = gamma
        self._pos  = 0
        self._size = 0

        self._states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions     = np.zeros((capacity,),         dtype=np.int64)
        self._rewards     = np.zeros((capacity,),         dtype=np.float32)
        self._dones       = np.zeros((capacity,),         dtype=np.float32)

        # Per-env n-step deques: keyed by env_id (int).  Each deque holds
        # (state, action, reward, next_state, done) namedtuple-like tuples.
        self._nstep_buf: dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_nstep_deque(self, env_id: int) -> deque:
        if env_id not in self._nstep_buf:
            self._nstep_buf[env_id] = deque(maxlen=self.n_step)
        return self._nstep_buf[env_id]

    def _commit(self, state, action: int, reward: float, next_state, done: bool):
        """Write a fully-resolved transition into the ring buffer."""
        self._states[self._pos]      = state
        self._next_states[self._pos] = next_state
        self._actions[self._pos]     = action
        self._rewards[self._pos]     = reward
        self._dones[self._pos]       = float(done)
        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _flush_nstep(self, buf: deque):
        """Compute n-step return for the oldest transition in the deque and commit it."""
        if not buf:
            return
        s0, a0, _, _, _ = buf[0]
        R = 0.0
        g = 1.0
        for (_, _, r, ns, d) in buf:
            R += g * r
            g *= self.gamma
            if d:
                break
        # ns and d belong to the last transition in the deque
        *_, ns_last, d_last = buf[-1]
        self._commit(s0, a0, R, ns_last, d_last)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, state, action: int, reward: float, next_state, done: bool,
             env_id: int = 0):
        """Add a single transition, resolving n-step returns if needed."""
        if self.n_step == 1:
            self._commit(state, action, reward, next_state, done)
            return

        buf = self._get_nstep_deque(env_id)
        buf.append((state, action, reward, next_state, done))

        if len(buf) == self.n_step:
            self._flush_nstep(buf)

        # On episode end, flush remaining partial sequences
        if done:
            while len(buf) > 0:
                self._flush_nstep(buf)
                buf.popleft()

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Push one transition per env row; env_id is the batch index."""
        n = len(states)
        if self.n_step == 1:
            for i in range(n):
                self._commit(
                    states[i], int(actions[i]), float(rewards[i]),
                    next_states[i], bool(dones[i]),
                )
            return

        for i in range(n):
            self.push(
                states[i], int(actions[i]), float(rewards[i]),
                next_states[i], bool(dones[i]), env_id=i,
            )

    def sample(self, batch_size: int, device='cpu'):
        idx = np.random.choice(self._size, batch_size, replace=False)
        dev = torch.device(device)
        non_blocking = dev.type == 'cuda'
        return (
            torch.from_numpy(self._states[idx]).to(dev, non_blocking=non_blocking),
            torch.from_numpy(self._actions[idx]).to(dev, non_blocking=non_blocking),
            torch.from_numpy(self._rewards[idx]).to(dev, non_blocking=non_blocking),
            torch.from_numpy(self._next_states[idx]).to(dev, non_blocking=non_blocking),
            torch.from_numpy(self._dones[idx]).to(dev, non_blocking=non_blocking),
        )

    def __len__(self):
        return self._size
