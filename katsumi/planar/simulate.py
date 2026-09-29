"""Forward re-integration (RK4, 1 ms) of NLP solutions and release-window analysis.

Two uses
  1. Verification: integrate the open-loop controls of a collocation solution with a fine RK4
     step and report the state mismatch at the phase ends (numerical-accuracy audit).
  2. Release-window W (plan Sec. 13.2): shift the release command by delta in [-0.2, 0.2] s,
     keep every joint under closed-loop PD tracking of the planned joint trajectory (torque
     saturated), let the fingers close when the hand enters the hook region of B, and integrate the
     catch with a compliant, capacity-limited grasp model. Success = hold for T_hold without the
     grasp being released and without leaving the model's admissible set.
"""
from __future__ import annotations

import numpy as np

from .. import device
from .anthro import Body, G
from .model import PlanarChain, NQ, NTH, NTAU


# ---------------------------------------------------------------------------- helpers ---
def _hs_interp(vals_k, vals_m, dur, t):
    """Quadratic (Hermite-Simpson) interpolation of a control given knot and mid-point values."""
    N = vals_m.shape[1]
    dt = dur / N
    s = np.clip(t / dt, 0, N - 1e-12)
    k = int(np.floor(s))
    tau = s - k                       # in [0,1)
    a, b, c = vals_k[:, k], vals_m[:, k], vals_k[:, k + 1]
    # Lagrange basis on {0, 1/2, 1}
    return a * (2 * tau - 1) * (tau - 1) + b * 4 * tau * (1 - tau) + c * tau * (2 * tau - 1)


def _state_interp(X, dur, t):
    """Linear interpolation of the planned state (used as PD reference)."""
    N = X.shape[1] - 1
    dt = dur / N
    s = np.clip(t / dt, 0, N - 1e-12)
    k = int(np.floor(s))
    a = s - k
    return (1 - a) * X[:, k] + a * X[:, k + 1]


def rk4(f, x, t, dt):
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Resim:
    def __init__(self, body: Body, res: dict, dt: float = 1e-3):
        self.body = body
        self.res = res
        self.chain = PlanarChain(body)
        self.dt = dt
        self.T = res["T"]
        self.eps = res["params"]["eps"]
        self.tau_cap = body.tau_cap

    def dev(self, t):
        return device.device_state(t, self.T, self.eps)

    # --- open-loop verification ------------------------------------------------------
    def verify(self):
        """Integrate each phase open loop from the planned initial state; return end-state errors."""
        r, ch, dt = self.res, self.chain, self.dt
        out = {}
        # swing
        d_s = r["d_s"]
        xs = r["S_X"][:, 0].copy()
        def fS(t, x):
            dv = self.dev(r["t_s0"] + t)
            u = _hs_interp(r["S_U"], r["S_Um"], d_s, t)
            q = np.concatenate([dv["pA"], x[:NTH]]); qd = np.concatenate([dv["vA"], x[NTH:]])
            thdd, _ = ch.pinned(q, qd, u * self.tau_cap, dv["aA"])
            return np.concatenate([x[NTH:], thdd])
        t = 0.0
        n = int(round(d_s / dt))
        h = d_s / n
        for _ in range(n):
            xs = rk4(fS, xs, t, h); t += h
        out["S_pos_err"] = np.abs(xs[:NTH] - r["S_X"][:NTH, -1]).max()
        out["S_vel_err"] = np.abs(xs[NTH:] - r["S_X"][NTH:, -1]).max()
        # flight
        d_f = r["d_f"]
        xf = r["F_X"][:, 0].copy()
        def fF(t, x):
            u = _hs_interp(r["F_U"], r["F_Um"], d_f, t)
            return np.concatenate([x[NQ:], ch.qdd_free(x[:NQ], x[NQ:], u * self.tau_cap)])
        t = 0.0
        n = int(round(d_f / dt)); h = d_f / n
        for _ in range(n):
            xf = rk4(fF, xf, t, h); t += h
        out["F_hand_err"] = np.abs(xf[:2] - r["F_X"][:2, -1]).max()
        out["F_pos_err"] = np.abs(xf[:NQ] - r["F_X"][:NQ, -1]).max()
        out["F_vel_err"] = np.abs(xf[NQ:] - r["F_X"][NQ:, -1]).max()
        # hold
        Th = r["params"]["T_hold"]
        xh = r["H_X"][:, 0].copy()
        Rmax = 0.0
        def fH(t, x):
            dv = self.dev(r["t_h0"] + t)
            u = _hs_interp(r["H_U"], r["H_Um"], Th, t)
            q = np.concatenate([dv["pB"], x[:NTH]]); qd = np.concatenate([dv["vB"], x[NTH:]])
            thdd, _ = ch.pinned(q, qd, u * self.tau_cap, dv["aB"])
            return np.concatenate([x[NTH:], thdd])
        t = 0.0
        n = int(round(Th / dt)); h = Th / n
        for _ in range(n):
            xh = rk4(fH, xh, t, h); t += h
        out["H_pos_err"] = np.abs(xh[:NTH] - r["H_X"][:NTH, -1]).max()
        out["H_vel_err"] = np.abs(xh[NTH:] - r["H_X"][NTH:, -1]).max()
        return out

    # --- metrics of the planned solution ------------------------------------------------
    def metrics(self):
        r, ch = self.res, self.chain
        m = {}
        dvl = self.dev(r["t_l"])
        q = np.concatenate([dvl["pA"], r["S_X"][:NTH, -1]]); qd = np.concatenate([dvl["vA"], r["S_X"][NTH:, -1]])
        m["v0"] = np.array(ch.f_comd(q, qd)).ravel()
        m["G_release"] = ch.com(q)
        m["h_release"] = float(dvl["h"])
        dvc = self.dev(r["t_c"])
        XF = r["F_X"]
        m["x_catch"] = float(dvc["x"])
        m["v_rel_catch"] = XF[NQ:NQ + 2, -1] - dvc["vB"]
        m["v_rel_norm"] = float(np.linalg.norm(m["v_rel_catch"]))
        m["G_catch"] = ch.com(XF[:NQ, -1])
        m["vG_catch"] = np.array(ch.f_comd(XF[:NQ, -1], XF[NQ:, -1])).ravel()
        m["impulse_catch"] = r["Lam"] * self.body.f_cap
        m["impulse_norm"] = float(np.linalg.norm(m["impulse_catch"]))
        m["U_catch"] = float(r["U_catch"])
        m["U_peak_swing"] = float(np.sqrt((r["S_R"] ** 2).sum(0)).max())
        m["U_peak_hold"] = float(np.sqrt((r["H_R"] ** 2).sum(0)).max())
        m["direction"] = "outbound" if r["phi_l"] < 1.0 else "return"
        # joint effort split
        for ph in ("S", "F", "H"):
            m[f"tau_rms_{ph}"] = float(np.sqrt((r[ph + "_U"] ** 2).mean()))
        return m
