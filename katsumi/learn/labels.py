"""Dense dual labels along oracle trajectories.

One trajectory-optimisation solution labels every knot of its swing phase: the multipliers of the state-continuity
defects give the costate p(t_k) = -nu_k (validated against the initial-condition multiplier dJ*/dx0), the value-to-go
is the objective minus the effort accrued before the knot (valid while the capacity peak lies ahead), the remaining
swing time is t_l - t_k and the required capacity is U_peak. A dataset of from-state solves therefore contains
~N_S labelled states per solve instead of one.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from ..planar.model import NTH


def trajectory_labels(sol, min_tau=0.05):
    """Rows (dicts) for the swing knots of one solution; [] when the multipliers are missing."""
    d = sol.get("duals") or {}
    nu = (d.get("defect_mult") or {}).get("S")
    if nu is None or not sol.get("ok"):
        return []
    X, U, R = np.array(sol["S_X"]), np.array(sol["S_U"]), np.array(sol["S_R"])
    N = X.shape[1] - 1
    dt = sol["d_s"] / N
    p = sol["params"]
    w_E, t_ref = p.get("w_E", 1.0), p.get("t_ref", 1.0)
    Uk = np.sqrt((R ** 2).sum(0))
    eff_k = 0.25 * (U ** 2).sum(0) + Uk ** 2                         # effort integrand at the knots
    # Simpson with interpolated mid-points (piecewise-linear commands / forces)
    Rm = 0.5 * (R[:, :-1] + R[:, 1:]); Um = 0.5 * (U[:, :-1] + U[:, 1:])
    eff_m = 0.25 * (Um ** 2).sum(0) + (Rm ** 2).sum(0)
    seg = dt / 6 * (eff_k[:-1] + 4 * eff_m + eff_k[1:])
    accrued = np.concatenate([[0.0], np.cumsum(seg)])                  # effort accrued before knot k
    # the value-to-go formula holds while the capacity peak still lies ahead
    k_peak = int(np.argmax(Uk)) if Uk.max() >= sol["U_peak"] - 1e-3 else N
    # nu_k belongs to the interval [t_k, t_{k+1}] (~ costate at its mid-point): extrapolate to the knot
    P = np.empty_like(nu)
    P[:, :-1] = -(1.5 * nu[:, :-1] - 0.5 * nu[:, 1:]); P[:, -1] = -nu[:, -1]
    rows = []
    body_kw = sol.get("body_kw", {}) or {}
    for k in range(min(N, k_peak + 1)):
        tau = sol["d_s"] - k * dt
        if tau < min_tau:
            break
        rows.append(dict(T=sol["T"], m=sol["m"], stature=body_kw.get("stature", 1.75), cap_scale=body_kw.get("cap_scale", 1.0),
                         t0=sol["t_s0"] + k * dt, x=X[:, k].copy(), J=sol["J"] - w_E * accrued[k] / t_ref, U=sol["U_peak"], tau=tau,
                         p=P[:, k].copy(), u=U[:, k].copy(), knot=k, dense=1))
    return rows


def dense_rows_from_dir(sol_dir, pattern="sol_*.pkl", min_tau=0.05):
    rows = []
    for f in sorted(glob.glob(os.path.join(sol_dir, pattern))):
        try:
            sol = pickle.load(open(f, "rb"))
        except Exception:
            continue
        rr = trajectory_labels(sol, min_tau)
        for r in rr:
            r["src"] = os.path.basename(f)
            r["ref"] = os.path.basename(sol.get("ref", f))
            r["level"] = sol.get("level", 0.0)
        rows += rr
    return rows
