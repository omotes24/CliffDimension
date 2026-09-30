"""Landing reflex: a fixed flight / hold controller shared by all swing-and-release methods.

After the release the policy under test no longer acts; the reflex tracks the flight joint-angle profile of a
reference oracle solution (nearest body mass) as a function of the time since the release, corrects the hand
position / velocity relative to B's tip towards the reference (least squares on the relative joints; the CoM is
ballistic so only hand-vs-CoM geometry can be changed), and after the catch tracks the reference hold profile.
This isolates the scientific question (swing pumping + release timing) from the catch mechanics, which are the
same for every method.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .. import device
from ..planar.model import NTH, NQ

P_REL = np.array([[1, -1, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], float)


def _rel(th):
    return np.array([th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]])


class LandingReflex:
    def __init__(self, refs, kp=1.0, kd=0.1, reach_gain=1.0, kp_hold=None, kd_hold=None):
        """refs: list of oracle solution dicts (any bodies); the nearest mass is used for a given episode.
        kp, kd: flight tracking gains; kp_hold, kd_hold: hold-phase gains (default: the flight gains)."""
        self.refs = [r for r in refs if r.get("ok")]
        assert self.refs, "no converged reference solutions"
        self.kp, self.kd, self.reach_gain = kp, kd, reach_gain
        self.kp_hold = kp if kp_hold is None else kp_hold; self.kd_hold = kd if kd_hold is None else kd_hold
        self.r = None

    @classmethod
    def from_glob(cls, pattern, **kw):
        refs = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(pattern))]
        return cls(refs, **kw)

    def select(self, m, T=None, plan=None):
        """plan: a controller's own multi-phase plan (solution dict with F_X/F_U/H_X/H_U and an absolute t_l), used
        instead of the nearest stored reference when given (oracle-in-the-loop MPC)."""
        if plan is not None and plan.get("ok") and "F_X" in plan:
            self.r = plan
        else:
            ms = np.array([r["m"] for r in self.refs]); Ts = np.array([r["T"] for r in self.refs])
            d = np.abs(ms - m) + (0.5 * np.abs(Ts - T) if T is not None else 0.0)
            self.r = self.refs[int(np.argmin(d))]
        r = self.r
        self.tauF = np.linspace(0, r["d_f"], r["F_X"].shape[1])
        self.tauH = np.linspace(0, r["params"]["T_hold"], r["H_X"].shape[1])
        self.tauC = np.linspace(0, r["d_c"], r["C_X"].shape[1]) if ("C_X" in r and r.get("catch_model") == "compliant") else None
        self.dvF = device.device_state(r["t_l"] + self.tauF, r["T"], r["params"]["eps"])
        return self.r

    def _interp(self, tau, taus, arr):
        tau = min(max(tau, taus[0]), taus[-1])
        return np.array([np.interp(tau, taus, row) for row in arr])

    def flight(self, env):
        r = self.r
        tau = env.t - env.t_release
        u = self._interp(tau, self.tauF, r["F_U"])
        th_ref = self._interp(tau, self.tauF, r["F_X"][2:NQ]); thd_ref = self._interp(tau, self.tauF, r["F_X"][NQ + 2:])
        if self.reach_gain > 0:
            # reference hand position / velocity relative to B's tip at the same time-since-release
            hand_ref = self._interp(tau, self.tauF, r["F_X"][0:2]); vhand_ref = self._interp(tau, self.tauF, r["F_X"][NQ:NQ + 2])
            pB_ref = np.array([np.interp(tau, self.tauF, self.dvF["pB"][:, i]) for i in range(2)])
            vB_ref = np.array([np.interp(tau, self.tauF, self.dvF["vB"][:, i]) for i in range(2)])
            dv = env.dev(env.t)
            e = (hand_ref - pB_ref) - (env.q[0:2] - dv["pB"])
            ev = (vhand_ref - vB_ref) - (env.qd[0:2] - dv["vB"])
            q = env.q.copy(); c0 = env.ch.com(q)
            J = np.zeros((2, NTH))
            for i in range(NTH):
                qq = q.copy(); qq[2 + i] += 1e-5
                J[:, i] = -(env.ch.com(qq) - c0) / 1e-5
            Jr = J @ P_REL
            W = np.diag([1.0, 1.0, 0.3, 0.3])
            Minv = W @ Jr.T @ np.linalg.inv(Jr @ W @ Jr.T + 1e-4 * np.eye(2))
            th_ref = th_ref + P_REL @ np.clip(Minv @ (self.reach_gain * e), -0.5, 0.5)
            thd_ref = thd_ref + P_REL @ np.clip(Minv @ (self.reach_gain * ev), -3.0, 3.0)
        return self._pd(u, th_ref, thd_ref, env)

    def hold(self, env):
        r = self.r
        tau = env.t - env.t_catch
        if self.tauC is not None and tau < self.tauC[-1]:
            u = self._interp(tau, self.tauC, r["C_U"])
            th_ref = self._interp(tau, self.tauC, r["C_X"][2:NQ]); thd_ref = self._interp(tau, self.tauC, r["C_X"][NQ + 2:])
        else:
            tau_h = tau - (self.tauC[-1] if self.tauC is not None else 0.0)
            u = self._interp(tau_h, self.tauH, r["H_U"])
            th_ref = self._interp(tau_h, self.tauH, r["H_X"][:NTH]); thd_ref = self._interp(tau_h, self.tauH, r["H_X"][NTH:])
        return self._pd(u, th_ref, thd_ref, env, self.kp_hold, self.kd_hold)

    def _pd(self, u, th_ref, thd_ref, env, kp=None, kd=None):
        kp = self.kp if kp is None else kp; kd = self.kd if kd is None else kd
        th, thd = env.q[2:], env.qd[2:]
        return np.clip(u + kp * (_rel(th_ref) - _rel(th)) + kd * (_rel(thd_ref) - _rel(thd)), -1, 1)

    def __call__(self, env):
        """Torques for the current mode (F or C); None while on A."""
        if env.mode == "F":
            return self.flight(env)
        if env.mode == "C":
            return self.hold(env)
        return None


class ReflexEnv:
    """Environment wrapper: the agent acts only while hooked on A (4 torques + release trigger); after the release the
    landing reflex produces the torques. Observation / reward / termination unchanged."""

    def __init__(self, env, reflex: LandingReflex):
        self.env, self.reflex = env, reflex
        self.observation_space, self.action_space = env.observation_space, env.action_space

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.env.flight_plan = None
        self.reflex.select(self.env.body.m, self.env.T)
        return obs, info

    def step(self, action):
        a = np.array(action, float)
        if self.env.mode != "A":
            plan = getattr(self.env, "flight_plan", None)
            if plan is not None and plan is not self.reflex.r:      # a controller handed over its own flight / hold plan
                self.reflex.select(self.env.body.m, self.env.T, plan=plan)
            a[:4] = self.reflex(self.env)
            a[4] = -1.0
        return self.env.step(a)

    def __getattr__(self, k):
        return getattr(self.env, k)
