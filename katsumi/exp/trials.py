"""Trial runners of the experiment suite.

replay_trial      : a stored reference solution is executed (swing: feed-forward + PD; after the release the shared landing
                    reflex), optionally from a later phase, with state perturbations and release-time errors.
controller_trial  : a closed-loop controller (BC, tracking MPC, dual-field MPC, oracle MPC) from a rest or perturbed start.
Both return one flat row (dict) with the frozen success rule applied at every capacity level.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from .. import device
from ..planar.env import PlanarCliffEnv
from ..planar.model import NQ, NTH, NTAU
from ..learn.reflex import LandingReflex, ReflexEnv
from .common import ENV_KW, REFLEX_KW, TRACK_KW, F_MAX_LEVELS, success_levels, first_failure

P_REL = np.array([[1, -1, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], float)
REL_LO = np.array([0.0, -4.2, -2.1, 0.0]); REL_HI = np.array([2.6, 0.60, 0.35, 2.6])


def _rel(th):
    return np.array([th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]])


class RefTracker:
    """Feed-forward torques of a solution + PD tracking of its swing joint angles (the reflex handles flight / hold).
    lead: feed-forward sampling offset (control_dt / 2 = mid-point sampling); dt_release: release-time error added to
    the planned release; exact: release scheduled at the (shifted) planned time inside the control period."""

    def __init__(self, r, control_dt, kp=1.0, kd=0.1, lead=None, dt_release=0.0, exact=True):
        self.r = r; self.kp, self.kd = kp, kd
        self.lead = control_dt / 2 if lead is None else lead
        N = r["S_U"].shape[1] - 1
        self.tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
        self.t_rel = r["t_l"] + dt_release
        self.exact = exact
        self.scheduled = False

    def __call__(self, env):
        t = env.t
        if self.exact and not self.scheduled:
            env.schedule_release(self.t_rel); self.scheduled = True
        tc = min(max(t, self.tS[0]), self.tS[-1]); tm = min(max(t + self.lead, self.tS[0]), self.tS[-1])
        u = np.array([np.interp(tm, self.tS, row) for row in self.r["S_U"]])
        th_ref = np.array([np.interp(tc, self.tS, row) for row in self.r["S_X"][:NTH]])
        thd_ref = np.array([np.interp(tc, self.tS, row) for row in self.r["S_X"][NTH:]])
        u = u + self.kp * (_rel(th_ref) - _rel(env.th)) + self.kd * (_rel(thd_ref) - _rel(env.thd))
        a = np.zeros(NTAU + 1); a[:NTAU] = np.clip(u, -1, 1)
        a[NTAU] = -1.0 if self.exact else (1.0 if (t >= self.t_rel - 1e-9 and not env.released) else -1.0)
        return a


def perturb_rest_state(th, thd, sig_th, sig_thd, rng, max_tries=100, clear_fn=None):
    """Perturb a swing state in relative-joint coordinates (inside the joint ranges and, when clear_fn is given, clear
    of the walls: clear_fn(th) -> signed clearance [m])."""
    for _ in range(max_tries):
        rel = _rel(th) + rng.normal(0, sig_th, 4); reld = _rel(thd) + rng.normal(0, sig_thd, 4)
        th2 = th[2] + rng.normal(0, sig_th); thd2 = thd[2] + rng.normal(0, sig_thd)      # base = trunk angle (th = th_trunk + P rel)
        if np.all(rel >= REL_LO + 0.01) and np.all(rel <= REL_HI - 0.01):
            th_n = th2 + P_REL @ rel; thd_n = thd2 + P_REL @ reld
            if clear_fn is None or clear_fn(th_n) > 0.0:
                return th_n, thd_n
    return th.copy(), thd.copy()


def make_env(r, env_kw=None, reflex_refs=None, reflex_kw=None):
    ek = dict(ENV_KW); ek.update(env_kw or {})
    rk = dict(REFLEX_KW); rk.update(reflex_kw or {})
    env = PlanarCliffEnv(T=r["T"], m=r["m"], body_kw=r.get("body_kw", {}) or {}, **ek)
    reflex = LandingReflex(reflex_refs or [r], **rk)
    return ReflexEnv(env, reflex)


def start_options(r, start="rest", sig_th=0.0, sig_thd=0.0, seed=0, k=None):
    """reset() options for a trial start: 'rest' (static hang at the swing start), 'knot' (swing knot k), 'flight'
    (state right after the planned release), 'catch' (start of the compliant catch phase or the post-impact hold start),
    'hold' (start of the hold phase). Perturbations apply to the rest / knot starts."""
    rng = np.random.default_rng(seed)
    if start in ("rest", "knot"):
        if start == "rest":
            th, thd, t0 = np.zeros(NTH), np.zeros(NTH), r["t_s0"]
        else:
            N = r["S_X"].shape[1] - 1
            tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
            th, thd, t0 = r["S_X"][:NTH, k].copy(), r["S_X"][NTH:, k].copy(), float(tS[k])
        if sig_th > 0 or sig_thd > 0:
            from ..planar.anthro import make_body
            from ..planar.model import PlanarChain
            from .common import wall_clearance
            chain = PlanarChain(make_body(r["m"], **(r.get("body_kw", {}) or {})))
            dv = device.device_state(t0, r["T"], r["params"]["eps"])
            th, thd = perturb_rest_state(th, thd, sig_th, sig_thd, rng, clear_fn=lambda th_: wall_clearance(chain, np.concatenate([dv["pA"], th_]), dv))
        return dict(t0=t0, state=(th, thd))
    if start == "flight":
        return dict(t0=r["t_l"], mode="F", q=r["F_X"][:NQ, 0], qd=r["F_X"][NQ:, 0], t_release=r["t_l"])
    if start == "catch":
        if "C_X" in r and r.get("catch_model") == "compliant":
            return dict(t0=r["t_c"], mode="C", q=r["C_X"][:NQ, 0], qd=r["C_X"][NQ:, 0], off=0.0, t_release=r["t_l"], t_catch=r["t_c"])
        dv = device.device_state(r["t_h0"], r["T"], r["params"]["eps"])
        q = np.concatenate([dv["pB"], r["H_X"][:NTH, 0]]); qd = np.concatenate([dv["vB"], r["H_X"][NTH:, 0]])
        return dict(t0=r["t_h0"], mode="C", q=q, qd=qd, off=0.0, t_release=r["t_l"], t_catch=r["t_c"])
    if start == "hold":
        dv = device.device_state(r["t_h0"], r["T"], r["params"]["eps"])
        q = np.concatenate([dv["pB"], r["H_X"][:NTH, 0]]); qd = np.concatenate([dv["vB"], r["H_X"][NTH:, 0]])
        return dict(t0=r["t_h0"], mode="C", q=q, qd=qd, off=0.0, t_release=r["t_l"], t_catch=r["t_c"])
    raise ValueError(start)


def finish_row(row, env, res, t_wall, swing_err=None):
    row.update(reason=res["reason"], t=res["t"], success_held=int(res["reason"] == "held B"), U_peak=res["U_peak"],
               U_peak_A=res.get("U_peak_A"), U_peak_C=res.get("U_peak_C"), E=res["E"], d_min=(res["d_min"] if np.isfinite(res["d_min"]) else np.nan),
               released=int(res.get("release_time") is not None), caught=int(res.get("catch_time") is not None),
               release_time=res.get("release_time"), catch_time=res.get("catch_time"), catch_off=res.get("catch_off"),
               catch_hx=(res.get("catch_hand") or (np.nan, np.nan))[0], catch_hy=(res.get("catch_hand") or (np.nan, np.nan))[1],
               jr_max=res.get("jr_max"), wall_max=res.get("wall_max"), slip=res.get("slip"), wall_s=round(t_wall, 1),
               swing_err=swing_err)
    ff, tf = first_failure(res)
    row["first_failure"], row["t_first_failure"] = ff, tf
    for lv, ok in success_levels(res).items():
        row[f"succ_{lv:g}"] = int(ok)
    te = res.get("t_exceed") or {}
    for lv in F_MAX_LEVELS:
        row[f"t_exc_{lv:g}"] = te.get(lv, np.nan)
    return row


def replay_trial(r, control_dt=None, sub_dt=None, lead=None, exact=True, start="rest", sig_th=0.0, sig_thd=0.0, seed=0, knot=None,
                 dt_release=0.0, reflex_refs=None, env_kw=None, reflex_kw=None, track_kw=None, max_time=40.0):
    """Execute a reference solution in the environment and return one row."""
    ek = dict(env_kw or {})
    if control_dt is not None: ek["control_dt"] = control_dt
    if sub_dt is not None: ek["sub_dt"] = sub_dt
    env = make_env(r, ek, reflex_refs, reflex_kw)
    tk = dict(TRACK_KW); tk.update(track_kw or {})
    pol = RefTracker(r, env.control_dt, kp=tk["kp"], kd=tk["kd"], lead=lead, dt_release=dt_release, exact=exact)
    opts = start_options(r, start, sig_th, sig_thd, seed, knot)
    env.reset(options=opts)
    t0 = time.time()
    N = r["S_X"].shape[1] - 1; tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
    done = False; errs = []; steps = 0; max_steps = int(max_time / env.control_dt)
    while not done and steps < max_steps:
        a = pol(env.env) if env.mode == "A" else np.zeros(NTAU + 1)
        obs, rew, term, trunc, inf = env.step(a); done = term or trunc; steps += 1
        if env.mode == "A":
            th_ref = np.array([np.interp(env.t, tS, row) for row in r["S_X"][:NTH]])
            errs.append(float(np.abs(env.q[2:] - th_ref).max()))
    res = env.env.result or dict(success=False, reason="max_steps", t=env.t - env.t0, U_peak=env.U_peak, E=env.E, d_min=env.d_min)
    row = dict(ref=os.path.basename(r.get("ref_file", "") or ""), T=r["T"], m=r["m"], phi=round(r["phi_l"], 4), catch_model=r.get("catch_model", "impact"),
               U_star=r["U_peak"], control_dt=env.control_dt, sub_dt=env.sub_dt, lead=(env.control_dt / 2 if lead is None else lead), exact=int(exact),
               start=start, knot=knot, sig_th=sig_th, sig_thd=sig_thd, seed=seed, dt_release=dt_release)
    return finish_row(row, env, res, time.time() - t0, swing_err=(max(errs) if errs else np.nan))


def controller_trial(ctrl, r, start="rest", sig_th=0.0, sig_thd=0.0, seed=0, knot=None, reflex_refs=None, env_kw=None, reflex_kw=None,
                     max_time=40.0, label=""):
    """Closed-loop trial of a controller object (callable env -> action with reset()); the landing reflex takes over after
    the release (a controller may hand over its own plan through env.flight_plan)."""
    env = make_env(r, env_kw, reflex_refs, reflex_kw)
    opts = start_options(r, start, sig_th, sig_thd, seed, knot)
    env.reset(options=opts)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    t0 = time.time(); done = False; steps = 0; max_steps = int(max_time / env.control_dt)
    while not done and steps < max_steps:
        a = ctrl(env.env) if env.mode == "A" else np.zeros(NTAU + 1)
        obs, rew, term, trunc, inf = env.step(a); done = term or trunc; steps += 1
    res = env.env.result or dict(success=False, reason="max_steps", t=env.t - env.t0, U_peak=env.U_peak, E=env.E, d_min=env.d_min)
    row = dict(ctrl=label, ref=os.path.basename(r.get("ref_file", "") or ""), T=r["T"], m=r["m"], phi=round(r["phi_l"], 4), U_star=r["U_peak"],
               start=start, knot=knot, sig_th=sig_th, sig_thd=sig_thd, seed=seed, control_dt=env.control_dt,
               n_fail=getattr(ctrl, "n_fail", None), n_solve=getattr(ctrl, "n_solve", None), solve_s=getattr(ctrl, "solve_s", None))
    return finish_row(row, env, res, time.time() - t0)
