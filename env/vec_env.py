"""
Multiprocessing vectorised environment.

Spawns N worker subprocesses, each running one CarEnv instance.
The main process communicates with workers via multiprocessing Pipes.
Done environments are auto-reset so the caller always receives a valid
next observation.

Usage:
    from env.vec_env import VecEnv
    from env.car_env import TRAINING_SENSOR_LAYOUT

    vec = VecEnv(n_envs=4, sensor_layout=TRAINING_SENSOR_LAYOUT)
    obs = vec.reset()                        # (4, obs_dim) float32
    obs, rewards, dones, infos = vec.step(actions)   # actions: (4,) int
    vec.close()
"""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ------------------------------------------------------------------

def _worker(conn, env_kwargs: dict):
    """Subprocess target: owns one CarEnv, responds to commands over a Pipe."""
    # make sure project root is importable inside the subprocess
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from env.car_env import CarEnv
    env = CarEnv(**env_kwargs)

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == 'reset':
                obs, info = env.reset()
                conn.send((obs, info))
            elif cmd == 'step':
                obs, reward, done, info = env.step(payload)
                if done:
                    obs, _ = env.reset()   # auto-reset; return first obs of new ep
                conn.send((obs, reward, done, info))
            elif cmd == 'close':
                break
    finally:
        conn.close()


# ------------------------------------------------------------------
# VecEnv
# ------------------------------------------------------------------

class VecEnv:
    """
    Vectorised wrapper around N independent CarEnv instances.
    Workers are spawned with mp.Process so the GIL is not a bottleneck.
    """

    def __init__(self, n_envs: int, **env_kwargs):
        self.n_envs = n_envs
        self._parent_conns = []
        self._processes = []

        ctx = mp.get_context('fork')   # 'fork' is default on Linux; fast startup
        for _ in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child_conn, env_kwargs), daemon=True)
            p.start()
            child_conn.close()          # only keep parent end
            self._parent_conns.append(parent_conn)
            self._processes.append(p)

        # discover obs_dim from the first worker
        self._parent_conns[0].send(('reset', None))
        obs0, _ = self._parent_conns[0].recv()
        self.obs_dim  = obs0.shape[0]
        self.n_actions = len(__import__('env.car_env', fromlist=['ACTIONS']).ACTIONS)

        # finish resetting the remaining workers
        for conn in self._parent_conns[1:]:
            conn.send(('reset', None))
        for conn in self._parent_conns[1:]:
            conn.recv()

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns (n_envs, obs_dim) float32 array."""
        for conn in self._parent_conns:
            conn.send(('reset', None))
        results = [conn.recv() for conn in self._parent_conns]
        return np.stack([r[0] for r in results], axis=0)

    def step(self, actions: np.ndarray):
        """
        Send one action per env and collect results.

        Args:
            actions: int array of shape (n_envs,)

        Returns:
            obs     : (n_envs, obs_dim) float32  — first obs of new ep if done
            rewards : (n_envs,)         float32
            dones   : (n_envs,)         bool
            infos   : list of dicts
        """
        for conn, action in zip(self._parent_conns, actions):
            conn.send(('step', int(action)))

        results = [conn.recv() for conn in self._parent_conns]
        obs, rewards, dones, infos = zip(*results)

        return (
            np.stack(obs,     axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=bool),
            list(infos),
        )

    def close(self):
        """Terminate all worker processes cleanly."""
        for conn in self._parent_conns:
            try:
                conn.send(('close', None))
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
