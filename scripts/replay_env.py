"""Replay an optimiser solution in the planar environment (oracle <-> environment consistency check).

The swing torques (S_U), the release time t_l, the flight torques (F_U), the compliant-catch torques (C_U, when
present) and the hold torques (H_U) of a solution pickle are applied as piecewise-linear commands, optionally with
a weak PD tracking of the reference joint angles (normalised gains kp [1/rad], kd [s/rad]; the pumped swing is
open-loop sensitive, so a pure feed-forward replay diverges after a few seconds from the 38 ms collocation mesh
alone). The release trigger fires at the first control step >= t_l. Reports the joint-angle error along the swing,
whether the catch happened, and U_peak of the environment vs U* of the optimiser (the environment's catch is
compliant; the optimiser's may be the rigid impact model).

usage: python scripts/replay_env.py results/v3/results/grid/sol_ref_T18_m66_phi0.250.pkl [--kp 1 --kd 0.1]
"""
import argparse, pickle, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.env import PlanarCliffEnv
from katsumi.planar.model import NTH, NQ


def _rel(th):
    return np.array([th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]])


class OracleTracker:
    """Feed-forward torques + PD tracking of an oracle solution (usable as a policy in the environment)."""

    def __init__(self, r, kp=1.0, kd=0.1):
        self.r, self.kp, self.kd = r, kp, kd
        self.t_l, self.d_f, self.t_h0, self.t_c = r["t_l"], r["d_f"], r["t_h0"], r["t_c"]
        N_S = r["S_U"].shape[1] - 1
        self.tS = r["t_s0"] + np.linspace(0, r["d_s"], N_S + 1)
        self.tF = self.t_l + np.linspace(0, r["d_f"], r["F_U"].shape[1])
        self.tH = self.t_h0 + np.linspace(0, r["params"]["T_hold"], r["H_U"].shape[1])
        self.tC = (r["t_c"] + np.linspace(0, r["d_c"], r["C_U"].shape[1])) if ("C_U" in r and r.get("catch_model") == "compliant") else None

    def ref(self, t):
        """(u_ff, th_ref, thd_ref) at absolute time t."""
        r = self.r
        if t < self.t_l:
            tt, U, X, off = self.tS, r["S_U"], r["S_X"], 0
        elif t < self.t_l + self.d_f:
            tt, U, X, off = self.tF, r["F_U"], r["F_X"], 2
        elif self.tC is not None and t < self.t_h0:
            tt, U, X, off = self.tC, r["C_U"], r["C_X"], 2
        else:
            tt, U, X, off = self.tH, r["H_U"], r["H_X"], 0
        t = min(max(t, tt[0]), tt[-1])
        u = np.array([np.interp(t, tt, row) for row in U])
        n = X.shape[0] // 2
        th = np.array([np.interp(t, tt, row) for row in X[off:n]])
        thd = np.array([np.interp(t, tt, row) for row in X[n + off:]])
        return u, th, thd

    def __call__(self, env):
        t = env.t
        u, th_ref, thd_ref = self.ref(t)
        th, thd = env.q[2:], env.qd[2:]
        u = u + self.kp * (_rel(th_ref) - _rel(th)) + self.kd * (_rel(thd_ref) - _rel(thd))
        a = np.zeros(5)
        a[:4] = np.clip(u, -1, 1)
        a[4] = 1.0 if (t >= self.t_l - 1e-9 and not env.released) else -1.0
        return a


def replay(r, control_dt=0.002, sub_dt=0.002, kp=1.0, kd=0.1, verbose=True, **env_kw):
    T, m = r["T"], r["m"]
    body_kw = r.get("body_kw", {}) or {}
    env = PlanarCliffEnv(T=T, m=m, body_kw=body_kw, control_dt=control_dt, sub_dt=sub_dt, **env_kw)
    pol = OracleTracker(r, kp, kd)
    t_s0 = r["t_s0"]
    obs, info = env.reset(options=dict(t0=t_s0, state=(r["S_X"][:NTH, 0], r["S_X"][NTH:, 0])))
    errs = []; U_env = []
    done = False; k = 0
    while not done:
        a = pol(env)
        obs, rew, term, trunc, inf = env.step(a)
        done = term or trunc
        U_env.append((env.t - t_s0, env.U, env.mode))
        _, th_ref, _ = pol.ref(env.t)
        errs.append((env.t - t_s0, np.abs(env.q[2:] - th_ref).max(), env.mode))
        k += 1
    res = env.result
    e = np.array([(a_, b_) for a_, b_, mo in errs if mo == "A"]) if errs else np.zeros((1, 2))
    out = dict(reason=res["reason"], success=res["success"], U_peak_env=res["U_peak"], U_star=r["U_peak"],
               U_catch_star=r.get("U_catch"), max_th_err_swing=float(e[:, 1].max()), th_err_at_release=float(e[-1, 1]),
               release_env=res["release_time"], release_star=r["t_l"] - t_s0, catch_env=res["catch_time"],
               catch_star=r["t_c"] - t_s0, d_min=res["d_min"], steps=k,
               U_swing_env=max([u for _, u, mo in U_env if mo == "A"] or [0]),
               U_swing_star=float(np.sqrt((r["S_R"] ** 2).sum(0)).max()),
               U_hold_env=max([u for _, u, mo in U_env if mo == "C"] or [0]),
               U_hold_star=float(np.sqrt((r["H_R"] ** 2).sum(0)).max()), U_imp_star=r.get("U_imp"))
    if verbose:
        for kk, v in out.items():
            print(f"  {kk:18s} {v}")
    return out, env, U_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="+")
    ap.add_argument("--sub-dt", type=float, default=0.002)
    ap.add_argument("--control-dt", type=float, default=0.002)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.1)
    a = ap.parse_args()
    for f in a.pkl:
        r = pickle.load(open(f, "rb"))
        print(f, "ok" if r.get("ok") else "NOT CONVERGED")
        replay(r, control_dt=a.control_dt, sub_dt=a.sub_dt, kp=a.kp, kd=a.kd)


if __name__ == "__main__":
    main()
