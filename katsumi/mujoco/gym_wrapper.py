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

TRAIN_T = [16, 17, 18, 21, 22, 23, 24]        # full periods (one-way 8..12 s, plan Sec. 11.1)
TRAIN_M = [60, 63, 66, 69, 72, 75]
VAL_T = [16.5, 17.5, 18.5, 21.5, 22.5, 23.5]
VAL_M = [61.5, 64.5, 67.5, 70.5, 73.5]


class CliffGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, T_set=None, m_set=None, phi0_set=None, fixed=None, env_params: EnvParams | None = None,
                 seed: int = 0, shaping: str = "none", w_dist: float = 20.0, w_face: float = 10.0):
        """shaping: "none" (plan reward only) or "margin" (adds -w_dist * d_min - w_face * (1 - facing) at the end of a
        failed episode, where d_min is the closest approach of the hand mid-point to B's tip after the release —
        the dense 'constraint margin' signal of the dual-world-model proposal)."""
        super().__init__()
        self.shaping, self.w_dist, self.w_face = shaping, w_dist, w_face
        self._d_min, self._face_best = np.inf, -1.0
        self.w_energy = 20.0                       # reward per metre of swing-energy height gained while on A
        self._E_prev, self._com_prev = None, None
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
            if phi0 is None:
                phi0 = float(self.rng.uniform(0, 1))
        else:
            T = float(self.rng.choice(self.T_set))
            m = float(self.rng.choice(self.m_set))
            phi0 = float(self.rng.uniform(0, 1)) if self.phi0_set is None else float(self.rng.choice(self.phi0_set))
        self.env = self._env_for(m)
        obs = self.env.reset(T=T, phi0=phi0)
        self._d_min, self._face_best = np.inf, -1.0
        self._E_prev, self._com_prev = None, None
        return obs, dict(T=T, m=m, phi0=phi0)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        a_env = a.copy()
        a_env[-2:] = 0.5 * (a[-2:] + 1.0)          # finger channels: [-1, 1] -> [0, 1]
        obs, r, term, trunc, info = self.env.step(a_env)
        if "energy" in self.shaping:
            # swing-up shaping (potential based): mechanical energy of the CoM relative to the hang, in metres of height,
            # rewarded while the athlete is still on A -- lets a policy discover gradual pumping like on a playground swing
            env = self.env
            m_ = env.model
            if getattr(self, "_hum_ids", None) is None or self._hum_model is not m_:
                root = m_.body_rootid[env.bid["pelvis"]]
                self._hum_ids = np.array([b for b in range(m_.nbody) if m_.body_rootid[b] == root])
                self._hum_mass = m_.body_mass[self._hum_ids]
                self._hum_model = m_
            c = (self._hum_mass[:, None] * env.data.xipos[self._hum_ids]).sum(0) / self._hum_mass.sum()
            if self._com_prev is not None and not env.released_A:
                v = (c - self._com_prev) / env.p.control_dt
                E = 0.5 * float(v @ v) / 9.81 + float(c[2])
                if self._E_prev is not None:
                    r = r + self.w_energy * (E - self._E_prev)
                self._E_prev = E
            self._com_prev = c
        if "margin" in self.shaping:
            env = self.env
            if env.released_A:
                pB = env.cliffs["B"][0]
                mid = 0.5 * (env.data.site_xpos[env.hands["left"].site] + env.data.site_xpos[env.hands["right"].site])
                dist = float(np.linalg.norm(mid - pB))
                if dist < self._d_min:
                    self._d_min = dist
                    self._face_best = float(env.data.site_xmat[env.chest_site].reshape(3, 3)[:, 2][0])
            if (term or trunc) and not info.get("success", False):
                dm = self._d_min if np.isfinite(self._d_min) else 3.0
                r = r - self.w_dist * dm - self.w_face * (1.0 - self._face_best)
                info = dict(info, d_min=dm, face_best=self._face_best)
        return obs, float(r), bool(term), bool(trunc), info
