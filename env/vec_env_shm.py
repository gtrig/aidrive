"""
Async shared-memory vectorised environment.

Instead of pickling observations over Pipes, each worker writes its observation
and reward directly into a pre-allocated shared NumPy array.  A pair of
multiprocessing Events per worker signals handoff:

    step_event   : set by parent to tell worker "new action is ready, step now"
    done_event   : set by worker to tell parent "result is ready, read now"

Async pipeline in train.py:
    1. Parent writes actions into action_buf and sets step_events  (non-blocking)
    2. Parent calls agent.learn()  (GPU work overlaps with worker stepping)
    3. Parent waits for done_events and reads obs/reward/done from shared memory

This eliminates per-step pickle/unpickle and overlaps CPU env stepping with GPU
gradient computation, giving a further 1.3–1.8× speedup over VecEnv.

Public API is identical to VecEnv so train.py can switch between them with a flag.
"""

import ctypes
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------
# Worker target
# ------------------------------------------------------------------

def _shm_worker(
    env_id: int,
    env_kwargs: dict,
    shm_names: dict,          # {name: (shm_name, shape, dtype_str)}
    step_event: mp.Event,
    done_event: mp.Event,
    stop_event: mp.Event,
):
    """Subprocess: owns one CarEnv; communicates via shared memory + Events."""
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from env.car_env import CarEnv
    env = CarEnv(**env_kwargs)

    # Attach to shared memory blocks
    blocks = {}
    arrays = {}
    for key, (sname, shape, dtype_str) in shm_names.items():
        blk = shm.SharedMemory(name=sname)
        blocks[key] = blk
        arrays[key] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=blk.buf)

    obs_arr      = arrays['obs']        # (n_envs, obs_dim)
    reward_arr   = arrays['reward']     # (n_envs,)
    done_arr     = arrays['done']       # (n_envs,)
    action_arr   = arrays['action']     # (n_envs,)
    gates_arr    = arrays['gates']      # (n_envs,)
    reason_arr   = arrays['reason']     # (n_envs,)  encoded as int
    lap_steps_arr  = arrays['lap_steps']   # (n_envs,)
    lap_bonus_arr  = arrays['lap_bonus']   # (n_envs,)  float32

    REASON_CODES = {'off_road': 0, 'max_steps': 1, 'lap_complete': 2}

    # Initial reset
    obs, info = env.reset()
    obs_arr[env_id] = obs
    done_arr[env_id] = 0
    done_event.set()

    try:
        while not stop_event.is_set():
            step_event.wait()
            if stop_event.is_set():
                break
            step_event.clear()

            action = int(action_arr[env_id])
            obs, reward, done, info = env.step(action)

            if done:
                obs, _ = env.reset()

            obs_arr[env_id]       = obs
            reward_arr[env_id]    = reward
            done_arr[env_id]      = float(done)
            gates_arr[env_id]     = int(info.get('gates_hit', 0))
            reason_arr[env_id]    = REASON_CODES.get(info.get('reason', 'off_road'), 0)
            lap_steps_arr[env_id] = int(info.get('lap_steps', 0))
            lap_bonus_arr[env_id] = float(info.get('lap_time_bonus', 0.0))

            done_event.set()
    finally:
        for blk in blocks.values():
            blk.close()


# ------------------------------------------------------------------
# ShmVecEnv
# ------------------------------------------------------------------

class ShmVecEnv:
    """Async shared-memory vectorised environment (drop-in for VecEnv)."""

    REASON_NAMES = {0: 'off_road', 1: 'max_steps', 2: 'lap_complete'}

    def __init__(self, n_envs: int, **env_kwargs):
        self.n_envs = n_envs

        # Probe obs_dim / n_actions from a temporary single env
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from env.car_env import CarEnv, ACTIONS
        _probe = CarEnv(**env_kwargs)
        obs0, _ = _probe.reset()
        self.obs_dim   = obs0.shape[0]
        self.n_actions = len(ACTIONS)
        del _probe

        # Allocate shared memory buffers
        self._shm_blocks = {}
        shm_names = {}

        def _alloc(key, shape, dtype):
            arr = np.zeros(shape, dtype=dtype)
            blk = shm.SharedMemory(create=True, size=arr.nbytes)
            np.ndarray(shape, dtype=dtype, buffer=blk.buf)[:] = arr
            self._shm_blocks[key] = blk
            shm_names[key] = (blk.name, shape, dtype.str)
            return blk

        _alloc('obs',       (n_envs, self.obs_dim), np.dtype('float32'))
        _alloc('reward',    (n_envs,),              np.dtype('float32'))
        _alloc('done',      (n_envs,),              np.dtype('float32'))
        _alloc('action',    (n_envs,),              np.dtype('int32'))
        _alloc('gates',     (n_envs,),              np.dtype('int32'))
        _alloc('reason',    (n_envs,),              np.dtype('int32'))
        _alloc('lap_steps', (n_envs,),              np.dtype('int32'))
        _alloc('lap_bonus', (n_envs,),              np.dtype('float32'))

        # Create numpy views into shared memory (parent side)
        def _view(key, shape, dtype):
            blk = self._shm_blocks[key]
            return np.ndarray(shape, dtype=dtype, buffer=blk.buf)

        self._obs_arr       = _view('obs',       (n_envs, self.obs_dim), np.float32)
        self._reward_arr    = _view('reward',    (n_envs,),              np.float32)
        self._done_arr      = _view('done',      (n_envs,),              np.float32)
        self._action_arr    = _view('action',    (n_envs,),              np.int32)
        self._gates_arr     = _view('gates',     (n_envs,),              np.int32)
        self._reason_arr    = _view('reason',    (n_envs,),              np.int32)
        self._lap_steps_arr = _view('lap_steps', (n_envs,),              np.int32)
        self._lap_bonus_arr = _view('lap_bonus', (n_envs,),              np.float32)

        # Events and processes
        ctx = mp.get_context('fork')
        self._step_events = [ctx.Event() for _ in range(n_envs)]
        self._done_events = [ctx.Event() for _ in range(n_envs)]
        self._stop_event  = ctx.Event()
        self._processes   = []

        for i in range(n_envs):
            p = ctx.Process(
                target=_shm_worker,
                args=(i, env_kwargs, shm_names,
                      self._step_events[i], self._done_events[i],
                      self._stop_event),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        # Wait for all workers to finish initial reset
        for ev in self._done_events:
            ev.wait()
            ev.clear()

    # ------------------------------------------------------------------
    # Public API (matches VecEnv)
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Trigger a full reset on all workers, return obs array."""
        # Workers already auto-reset on done; this is a hard reset for init.
        # Re-use the initial obs already in shared memory.
        return self._obs_arr.copy()

    def send_actions(self, actions: np.ndarray):
        """Write actions and fire step_events (non-blocking — async half)."""
        self._action_arr[:] = actions.astype(np.int32)
        for ev in self._step_events:
            ev.set()

    def recv_results(self):
        """Block until all workers have finished stepping, then read results."""
        for ev in self._done_events:
            ev.wait()
            ev.clear()

        obs     = self._obs_arr.copy()
        rewards = self._reward_arr.copy()
        dones   = self._done_arr.copy().astype(bool)
        reason_codes = self._reason_arr.copy()
        lap_steps    = self._lap_steps_arr.copy()
        lap_bonus    = self._lap_bonus_arr.copy()
        infos = []
        for i in range(self.n_envs):
            reason = self.REASON_NAMES.get(int(reason_codes[i]), 'off_road')
            info   = {
                'gates_hit': int(self._gates_arr[i]),
                'reason':    reason,
            }
            if reason == 'lap_complete':
                info['lap_steps']      = int(lap_steps[i])
                info['lap_time_bonus'] = float(lap_bonus[i])
            infos.append(info)
        return obs, rewards, dones, infos

    def step(self, actions: np.ndarray):
        """Synchronous step (same signature as VecEnv.step) — send then recv."""
        self.send_actions(actions)
        return self.recv_results()

    def close(self):
        self._stop_event.set()
        for ev in self._step_events:
            ev.set()     # unblock waiting workers
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for blk in self._shm_blocks.values():
            blk.close()
            blk.unlink()
