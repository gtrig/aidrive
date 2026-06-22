"""
PPO (Proximal Policy Optimisation) agent with SEPARATE actor and critic networks.

Key design choice: the policy (actor) and value (critic) networks are fully
independent MLPs with their own Adam optimisers.  Sharing a trunk causes the
value loss (scale ~50-100) to dominate over the policy gradient (scale ~0.01),
swamping the entropy signal and causing periodic policy collapse.  With separate
networks each gradient update is clean and correctly scaled.

Public API (unchanged from the shared-trunk version):
    agent = PPOAgent(obs_dim, n_actions, device=...)
    actions, logps, values = agent.act_batch(obs_np)   # rollout collection
    action = agent.evaluate(obs_np)                    # greedy (eval/play)
    agent.update(rollout)                              # PPO gradient step
    agent.save(path)  /  agent.load(path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ──────────────────────────────────────────────────────────────────────────────
# Running statistics for observation normalisation
# ──────────────────────────────────────────────────────────────────────────────

class RunningMeanStd:
    """Welford online algorithm for mean and variance of a float vector."""

    def __init__(self, shape: tuple[int, ...]):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_n    = x.shape[0]
        total  = self.count + batch_n
        delta  = batch_mean - self.mean
        new_mean = self.mean + delta * batch_n / total
        m_a = self.var * self.count
        m_b = batch_var * batch_n
        new_var = (m_a + m_b + delta ** 2 * self.count * batch_n / total) / total
        self.mean  = new_mean
        self.var   = new_var
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state_dict(self) -> dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d: dict):
        self.mean  = np.array(d['mean'],  dtype=np.float64)
        self.var   = np.array(d['var'],   dtype=np.float64)
        self.count = float(d['count'])


# ──────────────────────────────────────────────────────────────────────────────
# Neural networks  (fully independent — no shared trunk)
# ──────────────────────────────────────────────────────────────────────────────

def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    """Orthogonal init — standard for PPO."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class PolicyNet(nn.Module):
    """Actor: obs → action logits."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),  nn.Tanh(),
            _layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action_logp_entropy(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ):
        logits = self(x)
        dist   = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class ValueNet(nn.Module):
    """Critic: obs → scalar value estimate."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),  nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1),        std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# PPO agent
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(
        self,
        obs_dim:    int,
        n_actions:  int,
        # network
        hidden:     int   = 128,
        # optimisation
        lr:         float = 3e-4,
        # PPO clip
        clip_coef:  float = 0.2,
        # loss coefficients
        ent_coef:   float = 0.10,
        # training
        update_epochs:    int   = 4,
        n_minibatches:    int   = 4,
        max_grad_norm:    float = 0.5,
        # misc
        device:     str   = 'cpu',
        norm_obs:   bool  = True,
    ):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.clip_coef = clip_coef
        self.ent_coef  = ent_coef
        self.update_epochs  = update_epochs
        self.n_minibatches  = n_minibatches
        self.max_grad_norm  = max_grad_norm
        self.device    = torch.device(device)
        self.norm_obs  = norm_obs

        # Fully separate networks — value gradients never touch the policy.
        self.policy_net = PolicyNet(obs_dim, n_actions, hidden).to(self.device)
        self.value_net  = ValueNet(obs_dim, hidden).to(self.device)

        self.actor_optimizer  = optim.Adam(
            self.policy_net.parameters(), lr=lr, eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.value_net.parameters(), lr=lr, eps=1e-5
        )

        self.obs_rms = RunningMeanStd((obs_dim,)) if norm_obs else None

        self._total_updates   = 0
        self._total_env_steps = 0
        self.resume_global_episode    = 0
        self.resume_best_score: float | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Observation normalisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def update_obs_stats(self, obs: np.ndarray):
        if self.obs_rms is not None:
            self.obs_rms.update(obs.reshape(-1, self.obs_dim))

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs)

    # ──────────────────────────────────────────────────────────────────────────
    # LR control (called by train.py for annealing)
    # ──────────────────────────────────────────────────────────────────────────

    def set_lr(self, lr: float):
        for pg in self.actor_optimizer.param_groups:
            pg['lr'] = lr
        for pg in self.critic_optimizer.param_groups:
            pg['lr'] = lr

    # ──────────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────────

    def act_batch(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stochastic actions for rollout collection.

        Returns:
            actions : (N,) int32
            logps   : (N,) float32
            values  : (N,) float32
        """
        obs_norm = self._normalize(obs)
        t = torch.from_numpy(obs_norm).to(self.device)
        with torch.no_grad():
            actions, logps, _ = self.policy_net.get_action_logp_entropy(t)
            values             = self.value_net(t)
        return (
            actions.cpu().numpy().astype(np.int32),
            logps.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
        )

    def evaluate(self, obs: np.ndarray) -> int:
        """Greedy (argmax) action for play / eval — no exploration."""
        obs_norm = self._normalize(obs)
        t = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(t)
        return int(logits.argmax(dim=-1).item())

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """Value estimates for bootstrapping the last rollout step."""
        obs_norm = self._normalize(obs)
        t = torch.from_numpy(obs_norm).to(self.device)
        with torch.no_grad():
            return self.value_net(t).cpu().numpy().astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # PPO update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, rollout) -> dict[str, float]:
        """Run K epochs of minibatch PPO over the provided rollout.

        Actor and critic are updated with SEPARATE backward passes so the
        value gradient never contaminates the policy network parameters.
        """
        pg_losses, v_losses, ent_losses, clip_fracs = [], [], [], []

        # Normalise observations with current running stats (consistent with
        # act_batch which also normalises before feeding to the networks).
        obs_flat      = self._normalize(rollout.obs.reshape(-1, self.obs_dim))
        actions_flat  = rollout.actions.reshape(-1)
        logps_flat    = rollout.logps.reshape(-1)
        advs_flat     = rollout.advantages.reshape(-1)
        returns_flat  = rollout.returns.reshape(-1)

        # Global advantage normalisation over the full rollout
        advs_flat = (advs_flat - advs_flat.mean()) / (advs_flat.std() + 1e-8)

        batch_size     = obs_flat.shape[0]
        minibatch_size = max(1, batch_size // self.n_minibatches)

        for _ in range(self.update_epochs):
            perm = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start: start + minibatch_size]

                mb_obs     = torch.from_numpy(obs_flat[idx]).to(self.device)
                mb_actions = torch.from_numpy(actions_flat[idx]).long().to(self.device)
                mb_logps   = torch.from_numpy(logps_flat[idx]).to(self.device)
                mb_advs    = torch.from_numpy(advs_flat[idx]).to(self.device)
                mb_returns = torch.from_numpy(returns_flat[idx]).to(self.device)

                # ── Actor update ──────────────────────────────────────────────
                _, newlogp, entropy = self.policy_net.get_action_logp_entropy(
                    mb_obs, mb_actions
                )
                logratio = newlogp - mb_logps
                ratio    = logratio.exp()

                pg_loss1 = -mb_advs * ratio
                pg_loss2 = -mb_advs * ratio.clamp(
                    1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()
                ent_loss = entropy.mean()
                actor_loss = pg_loss - self.ent_coef * ent_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                # ── Critic update ─────────────────────────────────────────────
                newvalue = self.value_net(mb_obs)
                v_loss   = 0.5 * (newvalue - mb_returns).pow(2).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                clip_fracs.append(clip_frac.item())

        self._total_updates += 1

        return {
            'pg_loss':   float(np.mean(pg_losses)),
            'v_loss':    float(np.mean(v_losses)),
            'entropy':   float(np.mean(ent_losses)),
            'clip_frac': float(np.mean(clip_fracs)),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(
        self,
        path: str,
        *,
        global_episode:  int   | None = None,
        best_score:      float | None = None,
        best_eval_score: float | None = None,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            'obs_dim':        self.obs_dim,
            'n_actions':      self.n_actions,
            'policy_net':     self.policy_net.state_dict(),
            'value_net':      self.value_net.state_dict(),
            'actor_optimizer':  self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_updates':    self._total_updates,
            'total_env_steps':  self._total_env_steps,
        }
        if self.obs_rms is not None:
            payload['obs_rms'] = self.obs_rms.state_dict()
        if global_episode is not None:
            payload['global_episode'] = int(global_episode)
        if best_score is not None:
            payload['best_score'] = float(best_score)
        if best_eval_score is not None:
            payload['best_eval_score'] = float(best_eval_score)
        torch.save(payload, path)

    def _read_checkpoint(self, path: str) -> dict:
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _apply_checkpoint(self, ckpt: dict):
        ckpt_obs_dim   = ckpt.get('obs_dim')
        ckpt_n_actions = ckpt.get('n_actions')
        if ckpt_obs_dim is not None and int(ckpt_obs_dim) != int(self.obs_dim):
            raise ValueError(
                f'Checkpoint obs_dim={ckpt_obs_dim} does not match '
                f'env obs_dim={self.obs_dim}.'
            )
        if ckpt_n_actions is not None and int(ckpt_n_actions) != int(self.n_actions):
            raise ValueError(
                f'Checkpoint n_actions={ckpt_n_actions} does not match '
                f'env n_actions={self.n_actions}.'
            )

        try:
            if 'policy_net' in ckpt:
                self.policy_net.load_state_dict(ckpt['policy_net'])
                self.value_net.load_state_dict(ckpt['value_net'])
            else:
                raise ValueError(
                    'Checkpoint uses old shared-trunk format; cannot load into '
                    'separate actor/critic architecture. Train from scratch.'
                )
        except RuntimeError as e:
            raise ValueError(
                f'Checkpoint tensor shapes are incompatible: {e}'
            ) from e

        if 'actor_optimizer' in ckpt:
            self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        if 'critic_optimizer' in ckpt:
            self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])

        if self.obs_rms is not None and 'obs_rms' in ckpt:
            self.obs_rms.load_state_dict(ckpt['obs_rms'])

    def load(self, path: str):
        ckpt = self._read_checkpoint(path)
        self._apply_checkpoint(ckpt)

        self._total_updates   = int(ckpt.get('total_updates', 0))
        self._total_env_steps = int(ckpt.get('total_env_steps', 0))
        self.resume_global_episode = int(ckpt.get('global_episode', 0))
        if 'best_score' in ckpt:
            self.resume_best_score = float(ckpt['best_score'])
        elif 'best_eval_score' in ckpt:
            # Legacy checkpoints stored eval mean under best_eval_score.
            self.resume_best_score = float(ckpt['best_eval_score'])
        else:
            self.resume_best_score = None

    def load_weights(self, path: str):
        """Restore network/optimizer state from a checkpoint.

        Training progress counters (env steps, update count) are preserved so
        LR schedules and logging continue from the current session.
        """
        env_steps = self._total_env_steps
        updates   = self._total_updates
        self._apply_checkpoint(self._read_checkpoint(path))
        self._total_env_steps = env_steps
        self._total_updates   = updates
