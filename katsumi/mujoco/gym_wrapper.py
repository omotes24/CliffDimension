"""Gymnasium wrapper around CliffEnv (for SAC / MBPO / DreamerV3 baselines on glacus).

Conditioning (T, m, phi0) is sampled per episode from the training sets of plan Sec. 11.1 unless fixed.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("pip install gymnasium") from e

from .env import CliffEnv, EnvParams
from .build_model import ModelParams

TRAIN_T = [8, 8.5, 9, 10.5, 11, 11.5, 12]
TRAIN_M = [60, 63, 66, 69, 72, 75]
VAL_T = [8.25, 8.75, 9.25, 10.75, 11.25, 11.75]
VAL_M = [61.5, 64.5, 67.5, 70.5, 73.5]


class CliffGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, T_set=None, m_set=None, phi0_set=None, fixed=None, env_params: EnvParams | None = None,
                 seed: int = 0):
        super().__init__()
        self.T_set = T_set or TRAIN_T
        self.m_set = m_set or TRAIN_M
        self.phi0_set = phi0_set                  # None -> Uniform[0, 2)
        self.fixed = fixed                        # dict(T=, m=, phi0=) overrides sampling
        self.rng = np.random.default_rng(seed)
        self.env = CliffEnv(env_params or EnvParams())
        self._envs_by_m = {}
        obs = self.env.reset()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.env.n_act,), dtype=np.float32)

    def _env_for(self, m):
        # one CliffEnv per body mass (model rebuild is expensive)
        if m not in self._envs_by_m:
            p = EnvParams(model=ModelParams(m=float(m)), T=self.env.p.T, phi0=self.env.p.phi0,
                          grasp=self.env.p.grasp, eps=self.env.p.eps)
            self._envs_by_m[m] = CliffEnv(p)
        return self._envs_by_m[m]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.fixed:
            T, m, phi0 = self.fixed["T"], self.fixed["m"], self.fixed.get("phi0", 0.0)
        else:
            T = float(self.rng.choice(self.T_set))
            m = float(self.rng.choice(self.m_set))
            phi0 = float(self.rng.uniform(0, 2)) if self.phi0_set is None else float(self.rng.choice(self.phi0_set))
        self.env = self._env_for(m)
        obs = self.env.reset(T=T, phi0=phi0)
        return obs, dict(T=T, m=m, phi0=phi0)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        a_env = a.copy()
        a_env[-2:] = 0.5 * (a[-2:] + 1.0)          # finger channels: [-1, 1] -> [0, 1]
        obs, r, term, trunc, info = self.env.step(a_env)
        return obs, float(r), bool(term), bool(trunc), info
