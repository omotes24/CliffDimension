"""Common definitions of the experiment suite.

Evaluation rules (fixed for all experiments; a change of anything here makes a new experiment):
  * F_REF = 1300 N is the DISPLAY reference of the grip load, U(t) = |R(t)| / F_REF.
  * F_MAX is the ACTUAL two-hand capacity allowed in a condition. Trials are run with the generous cap U_CAP_RUN and the
    success at every level F_MAX in F_MAX_LEVELS is decided afterwards from the recorded load history
    (success(F_MAX) = 2 s hold reached AND max_t U(t) <= F_MAX / F_REF). This is exact: the trajectory up to the first
    exceedance of a level does not depend on the cap, and an exceedance is a failure at that level.
  * Main metric: rest start (static hang at the swing start of the reference), release before the deadline, catch of B,
    and a 2 s hold while the grip capacity, joint and contact conditions hold. Wall contact during the swing or the flight
    (penetration > 2 cm) is a failure; bracing against B's wall panel while hooked is allowed (contact model).
  * Trials started in the middle of the swing (perturbed states) are diagnostic ("recovery") trials: separate tables.
  * Every trial records the first failure type and its time.
"""
from __future__ import annotations

import copy
import os
import pickle
import subprocess
import time

import numpy as np

from .. import device
from ..planar.anthro import make_body, G
from ..planar.model import PlanarChain, NQ, NTH, NTAU
from ..planar.nlp import PlanarNLP, ReducedParams

F_REF = 1300.0
F_MAX_LEVELS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)        # multiples of F_REF
F_MAX_HEADLINE = 2.0                                          # 2600 N: the capacity used in the headline numbers
U_CAP_RUN = 2.5                                               # cap applied while running (trials continue up to here)

# executor fixed for the suite (experiment 2 re-examines these values; the suite's main runs use them)
ENV_KW = dict(control_dt=0.002, sub_dt=0.0005, hook_cap_B=0.1, hook_cap=0.1, hold="zoh", U_cap=U_CAP_RUN,
              k_wall=1e5, d_wall=2e3, wall_fail_C=0.10, joint_stop_k=15.0, stop_damp=0.01, K_att=40000.0, D_att=1500.0,
              hook_tol=0.025, catch_box=(0.03, -0.005, 0.05), u_rate=10.0, release_thresh=0.8)
REFLEX_KW = dict(kp=1.0, kd=0.1, reach_gain=0.25, kp_hold=3.0, kd_hold=0.3, hold_mode="track")
TRACK_KW = dict(kp=1.0, kd=0.1)                               # swing feed-forward + PD tracking gains of the replays

FAILURES = ("slipped off A", "no release", "missed B", "hit B's face", "catch velocity", "lost hook", "grip capacity exceeded",
            "hit wall", "held B")


def git_hash(root):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def config_dict(root):
    b = make_body(66.0)
    return dict(F_REF=F_REF, F_MAX_LEVELS=list(F_MAX_LEVELS), F_MAX_HEADLINE=F_MAX_HEADLINE, U_CAP_RUN=U_CAP_RUN, ENV_KW=ENV_KW,
                REFLEX_KW=REFLEX_KW, TRACK_KW=TRACK_KW, git=git_hash(root),
                device=dict(X0=device.X0, XT=device.XT, H0=device.H0, D_LEDGE=device.D_LEDGE, LEDGE_HEIGHT=device.LEDGE_HEIGHT,
                            WALL_BOTTOM_OFFSET=device.WALL_BOTTOM_OFFSET, eps=0.20),
                body=dict(tau_cap=b.tau_cap.tolist(), f_cap=b.f_cap, mu_out=b.mu_out, mu_in=b.mu_in, qd_max=b.qd_max,
                          clearance=b.clearance.tolist(), forearm_clearance=b.forearm_clearance.tolist(), stature_ref=1.75),
                nlp_defaults={k: (list(v) if isinstance(v, tuple) else v) for k, v in ReducedParams().__dict__.items() if k != "x0"},
                success_rule="rest start; release before the deadline; catch; 2 s hold; U(t) <= F_MAX / F_REF for all t; "
                             "no wall penetration > 2 cm while swinging / flying; bracing on B's panel allowed")


def success_levels(res):
    """success(F_MAX) for every level from a trial result (dict from env.result)."""
    held = res.get("reason") == "held B"
    return {lv: bool(held and res["U_peak"] <= lv + 1e-9) for lv in F_MAX_LEVELS}


def first_failure(res):
    """(type, time) of the first failure at the headline capacity: an exceedance of F_MAX_HEADLINE that happened before the
    terminal event is the first failure."""
    t_exc = (res.get("t_exceed") or {}).get(F_MAX_HEADLINE)
    if t_exc is not None and (res.get("reason") == "held B" or t_exc < res["t"] - 1e-9):
        return "grip capacity exceeded", t_exc
    if res.get("reason") == "held B":
        return "none", None
    return res.get("reason"), res.get("t")


# ------------------------------------------------------------------------------------------------------------------
# solution metrics
# ------------------------------------------------------------------------------------------------------------------
def _phase_arrays(r, ph):
    if ph == "S":
        return r["S_X"], r["t_s0"], r["d_s"], "A"
    if ph == "F":
        return r["F_X"], r["t_l"], r["d_f"], "F"
    if ph == "C" and "C_X" in r and r.get("catch_model") == "compliant":
        return r["C_X"], r["t_c"], r["d_c"], "C"
    if ph == "H":
        return r["H_X"], r["t_h0"], r["params"]["T_hold"], "C"
    return None


def wall_clearance(chain, q, dv):
    """Minimum signed clearance [m] of the body points to the two wall bodies (negative = penetration)."""
    body = chain.body
    P = np.hstack([q[0:2, None], chain.f_tips(q), chain.f_forearm(q)])
    clear = np.concatenate([[0.0], body.clearance, body.forearm_clearance])
    h, xB = float(dv["h"]), float(dv["x"])
    best = np.inf
    for i in range(P.shape[1]):
        x, y = P[0, i], P[1, i]
        aA = x - (-device.D_LEDGE + clear[i]); bA = (h - device.WALL_BOTTOM_OFFSET) - y
        aB = (xB + device.D_LEDGE - clear[i]) - x; bB = -device.WALL_BOTTOM_OFFSET - y
        best = min(best, max(aA, bA), max(aB, bB))
    return float(best)


def solution_metrics(r, body=None):
    """Scalar metrics of a multi-phase solution (for the solver experiments)."""
    body = body or make_body(r["m"], **(r.get("body_kw", {}) or {}))
    chain = PlanarChain(body)
    p = r["params"]
    out = dict(ok=bool(r.get("ok")), status=r.get("status"), iters=r.get("iters"), U_peak=float(r["U_peak"]), effort=float(r["effort"]),
               J=float(r.get("J", p.get("w_U", 10.0) * r["U_peak"] + p.get("w_E", 1.0) * r["effort"])), d_s=float(r["d_s"]),
               d_f=float(r["d_f"]), phi_l=float(r["phi_l"]), phi_c=float(r.get("phi_c", np.nan)), t_l=float(r["t_l"]),
               U_catch=float(r.get("U_catch", np.nan)), U_imp=float(r.get("U_imp", np.nan)) if r.get("U_imp") is not None else np.nan,
               req_cap_N=float(r["U_peak"] * body.f_cap), req_cap_BW=float(r["U_peak"] * body.f_cap / (body.m * G)))
    # phase peak loads
    out["U_swing"] = float(np.sqrt((r["S_R"] ** 2).sum(0)).max())
    out["U_hold"] = float(np.sqrt((r["H_R"] ** 2).sum(0)).max())
    out["U_catchphase"] = float(np.sqrt((r["C_R"] ** 2).sum(0)).max()) if "C_R" in r else np.nan
    # catch relative velocity (hand vs B at the end of the flight)
    T, eps = r["T"], p["eps"]
    dvc = device.device_state(r["t_c"], T, eps)
    vrel = r["F_X"][NQ:NQ + 2, -1] - dvc["vB"]
    out["vrel_x"], out["vrel_y"], out["vrel_norm"] = float(vrel[0]), float(vrel[1]), float(np.linalg.norm(vrel))
    # minimum wall clearance / joint margin per phase
    lo_neg = np.array(p.get("rel_lo_face_neg", (0.0, -4.2, -2.1, 0.0))); hi_neg = np.array(p.get("rel_hi_face_neg", (2.6, 0.6, 0.35, 2.6)))
    A_REL = np.array([[1, -1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1]], float)
    for ph in ("S", "F", "C", "H"):
        arr = _phase_arrays(r, ph)
        if arr is None:
            continue
        X, t0, dur, mode = arr
        N = X.shape[1] - 1
        wc, jm = np.inf, np.inf
        for k in range(N + 1):
            t = t0 + dur * k / N
            dv = device.device_state(t, T, eps)
            if mode == "A":
                q = np.concatenate([dv["pA"], X[:NTH, k]]); lo, hi = lo_neg, hi_neg
            elif mode == "C":
                q = np.concatenate([dv["pB"], X[:NTH, k]]) if X.shape[0] == 2 * NTH else X[:NQ, k]; lo, hi = -hi_neg, -lo_neg
            else:
                q = X[:NQ, k]; lo, hi = np.minimum(lo_neg, -hi_neg), np.maximum(hi_neg, -lo_neg)
            wc = min(wc, wall_clearance(chain, q, dv))
            rel = A_REL @ q[2:]
            jm = min(jm, float(np.min(np.minimum(rel - lo, hi - rel))))
        out[f"wall_min_{ph}"] = wc; out[f"joint_margin_{ph}_deg"] = float(np.degrees(jm))
    out["wall_min"] = float(min(v for k, v in out.items() if k.startswith("wall_min_")))
    out["joint_margin_deg"] = float(min(v for k, v in out.items() if k.startswith("joint_margin_")))
    out["u_max"] = float(np.abs(r["S_U"]).max())
    return out


# ------------------------------------------------------------------------------------------------------------------
# solver helpers
# ------------------------------------------------------------------------------------------------------------------
COLD_GUESSES = ((3.0, 0.6, "quadratic"), (4.5, 0.6, "quadratic"), (8.0, 0.6, "linear"), (11.0, 0.7, "linear"), (2.0, 0.6, "quadratic"))


def ref_params(r, drop=("fixed_release_phase", "wait_in_cost", "from_state", "x0", "u0", "u0_dt", "cone_abs")):
    kw = dict(r["params"])
    for q in drop:
        kw.pop(q, None)
    return kw


def solve_variant(ref, overrides=None, body_kw=None, fn=None, max_cpu=1800.0, max_iter=3000, tol=1e-4, cold=True, verbose=False):
    """Re-solve the full problem of a reference solution (same T, m, release phase) with parameter overrides, warm-started
    from the reference; cold starts as fall-back. Returns the solution dict (cached in `fn`)."""
    if fn and os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    kw = ref_params(ref); kw.update(overrides or {})
    phi = float(kw.pop("fixed_release_phase", ref["phi_l"]) or ref["phi_l"])
    bk = dict(ref.get("body_kw", {}) or {}); bk.update(body_kw or {})
    body = make_body(ref["m"], **bk)
    p = ReducedParams(fixed_release_phase=phi, wait_in_cost=False, **kw)
    t0 = time.time(); best = None; tried = []
    nlp = PlanarNLP(body, ref["T"], phi, p)
    try:
        nlp.set_initial(prev=ref)
    except Exception:
        nlp.set_initial(d_w=max(0.0, ref["T"] - ref["d_s"]), d_s=ref["d_s"])
    r = nlp.solve(print_level=5 if verbose else 0, max_iter=max_iter, tol=tol, max_cpu_time=max_cpu, sensitivities=True)
    r["cold_start"] = 0; tried.append(r)
    if r["ok"]:
        best = r
    elif cold:
        for d_s, amp, pump in COLD_GUESSES:
            nlp = PlanarNLP(body, ref["T"], phi, p)
            nlp.set_initial(d_w=max(0.0, ref["T"] - d_s), d_s=d_s, swing_amp=amp, pump=pump)
            r = nlp.solve(print_level=0, max_iter=max_iter, tol=tol, max_cpu_time=max_cpu, sensitivities=True)
            r["cold_start"] = 1; tried.append(r)
            if r["ok"]:
                best = r; break
    if best is None:
        best = min(tried, key=lambda x: (x.get("catch_gap", 9.0) if np.isfinite(x.get("catch_gap", 9.0) or 9.0) else 9.0, x.get("U_peak", 9)))
    best["solve_s"] = time.time() - t0; best["body_kw"] = bk; best["overrides"] = dict(overrides or {}); best["ref_file"] = ref.get("ref_file")
    for key in list(best.keys()):
        if key.endswith("_Am") or key.endswith("_Um") or key.endswith("_Rm"):
            best.pop(key)
    if fn:
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        pickle.dump(best, open(fn, "wb"))
    return best


def solve_from_state(ref, x0, t0, u0=None, u0_dt=0.002, overrides=None, warm_k=None, fn=None, max_cpu=600.0, max_iter=2000,
                     tol=1e-4, N_min=30, verbose=False, acceptable_tol=1e-3, warm=None):
    """Remaining-problem (cost-to-go) solve from the swing state x0 at absolute time t0 (free release time, no wait),
    warm-started from the tail of the reference from knot warm_k (nearest knot when None), or from `warm` (a from-state
    solution on the same mesh, e.g. the base solve of a finite-difference pair). Returns the solution with duals
    (costate dJ*/dx0, dJ*/dt0, multiplier categories)."""
    if fn and os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    r = ref
    T, m = r["T"], r["m"]
    N = r["S_X"].shape[1] - 1
    tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
    if warm_k is None:
        warm_k = int(np.clip(np.searchsorted(tS, t0), 0, N - 1))
    rem = max(r["d_s"] - (tS[warm_k] - r["t_s0"]), 0.3)
    kw = ref_params(r, drop=("fixed_release_phase", "wait_in_cost", "from_state", "x0", "N_S", "d_s_bounds", "u0", "u0_dt", "cone_abs"))
    kw.update(overrides or {})
    N_S = int(np.clip(round(N * rem / r["d_s"]) + 10, N_min, 180)) if warm is None else warm["S_X"].shape[1] - 1
    p = ReducedParams(fixed_release_phase=None, wait_in_cost=False, from_state=True, x0=tuple(np.asarray(x0, float)), N_S=N_S,
                      d_s_bounds=(0.02, T), u0=(None if u0 is None else tuple(np.asarray(u0, float))), u0_dt=u0_dt, **kw)
    body_kw = r.get("body_kw", {}) or {}
    nlp = PlanarNLP(make_body(m, **body_kw), T, t0 / T, p)
    if warm is not None:
        prev = dict(warm); prev["d_w"] = 0.0
    else:
        prev = dict(r)
        for key in ("S_X", "S_A", "S_U", "S_R", "S_Am", "S_Um", "S_Rm"):
            if key in r:
                prev[key] = r[key][:, warm_k:]
        prev["d_s"] = rem; prev["d_w"] = 0.0
    try:
        nlp.set_initial(prev=prev)
    except Exception:
        nlp.set_initial(d_w=0.0, d_s=rem)
    t1 = time.time()
    out = nlp.solve(print_level=5 if verbose else 0, max_iter=max_iter, tol=tol, max_cpu_time=max_cpu, sensitivities=True,
                    acceptable_tol=acceptable_tol)
    out["solve_s"] = time.time() - t1; out["t0_abs"] = float(t0); out["x0"] = np.asarray(x0, float).tolist(); out["u0"] = None if u0 is None else list(u0)
    out["body_kw"] = body_kw; out["overrides"] = dict(overrides or {})
    for key in list(out.keys()):
        if key.endswith("_Am") or key.endswith("_Um") or key.endswith("_Rm"):
            out.pop(key)
    if fn:
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        pickle.dump(out, open(fn, "wb"))
    return out


def ref_state(r, k):
    """(x_k, t_k, u_k) at swing knot k of a reference solution."""
    N = r["S_X"].shape[1] - 1
    tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
    return r["S_X"][:, k].copy(), float(tS[k]), r["S_U"][:, k].copy()


def accrued_effort(r):
    """Effort accrued before every swing knot (Simpson with interpolated mid-points), as used by the trajectory labels."""
    X, U, R = np.array(r["S_X"]), np.array(r["S_U"]), np.array(r["S_R"])
    N = X.shape[1] - 1
    dt = r["d_s"] / N
    Uk = np.sqrt((R ** 2).sum(0))
    eff_k = 0.25 * (U ** 2).sum(0) + Uk ** 2
    Rm = 0.5 * (R[:, :-1] + R[:, 1:]); Um = 0.5 * (U[:, :-1] + U[:, 1:])
    eff_m = 0.25 * (Um ** 2).sum(0) + (Rm ** 2).sum(0)
    seg = dt / 6 * (eff_k[:-1] + 4 * eff_m + eff_k[1:])
    return np.concatenate([[0.0], np.cumsum(seg)]), Uk


def load_ref(fn):
    r = pickle.load(open(fn, "rb"))
    r["ref_file"] = fn
    return r
