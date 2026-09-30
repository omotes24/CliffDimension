"""Executor decomposition: replay oracle solutions in the environment (feed-forward + weak PD, landing reflex
shared with the closed-loop evaluation) and tabulate, per reference, whether the swing was executed to the release,
whether the catch happened and whether the hold survived -- for several control periods / feed-forward samplings.

usage: python scripts/replay_table.py --refs "results/v3/results/grid/sol_ref_T18_m*_phi0.250.pkl" \
           --control-dt 0.02 0.01 0.004 --out results/figs/replay_table.csv
"""
import argparse, glob, os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_env import replay, OracleTracker
from katsumi.learn.reflex import LandingReflex, ReflexEnv
from katsumi.planar.env import PlanarCliffEnv
from katsumi.planar.model import NTH


def replay_reflex(r, refs, control_dt, sub_dt, kp, kd, lead, reach, hook_cap_B, kp_hold, kd_hold, hold_mode="track"):
    """Swing by feed-forward + PD of the solution; flight / hold by the landing reflex of the closed-loop evaluation."""
    env = ReflexEnv(PlanarCliffEnv(T=r["T"], m=r["m"], body_kw=r.get("body_kw", {}) or {}, control_dt=control_dt, sub_dt=sub_dt, hook_cap_B=hook_cap_B),
                    LandingReflex(refs, reach_gain=reach, kp_hold=kp_hold, kd_hold=kd_hold, hold_mode=hold_mode))
    pol = OracleTracker(r, kp, kd, 0.0, lead=lead)
    env.reset(options=dict(t0=r["t_s0"], state=(np.zeros(NTH), np.zeros(NTH))))
    N = r["S_X"].shape[1] - 1; tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
    done = False; errs = []; U_swing = 0.0; U_hold = 0.0
    while not done:
        a = pol(env.env)
        obs, rew, term, trunc, inf = env.step(a); done = term or trunc
        if env.mode == "A":
            th_ref = np.array([np.interp(env.t, tS, row) for row in r["S_X"][:NTH]])
            errs.append(np.abs(env.q[2:] - th_ref).max()); U_swing = max(U_swing, env.U)
        elif env.mode == "C":
            U_hold = max(U_hold, env.U)
    out = dict(U_swing_env=U_swing, U_swing_star=float(np.sqrt((r["S_R"] ** 2).sum(0)).max()), U_hold_env=U_hold,
               max_th_err_swing=float(max(errs)) if errs else float("nan"))
    return out, env.env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="results/v3/results/grid/sol_ref_T18_m*_phi0.250.pkl")
    ap.add_argument("--control-dt", type=float, nargs="+", default=[0.02, 0.01, 0.004])
    ap.add_argument("--lead", type=str, nargs="+", default=["half"], help="feed-forward sampling: 'zero' (interval start) or 'half' (mid-point)")
    ap.add_argument("--reach", type=float, nargs="+", default=[0.0])
    ap.add_argument("--hook-cap-B", type=float, default=0.1)
    ap.add_argument("--sub-dt", type=float, default=0.001)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.1)
    ap.add_argument("--kp-hold", type=float, default=3.0); ap.add_argument("--kd-hold", type=float, default=0.3)
    ap.add_argument("--reflex-refs", default=None, help="glob of solutions for the landing reflex (default: the replayed solution itself)")
    ap.add_argument("--out", default="results/figs/replay_table.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(a.refs))
    reflex_refs = [pickle.load(open(g, "rb")) for g in sorted(glob.glob(a.reflex_refs))] if a.reflex_refs else None
    rows = []
    for f in files:
        r = pickle.load(open(f, "rb"))
        if not r.get("ok"):
            continue
        for cdt in a.control_dt:
            for lead in a.lead:
                for reach in a.reach:
                    ld = 0.0 if lead == "zero" else cdt / 2
                    sdt = min(a.sub_dt, cdt)
                    out, env = replay_reflex(r, reflex_refs or [r], cdt, sdt, a.kp, a.kd, ld, reach, a.hook_cap_B, a.kp_hold, a.kd_hold)
                    res = env.result
                    rows.append(dict(ref=os.path.basename(f), T=r["T"], m=r["m"], phi=round(r["phi_l"], 4), catch_model=r.get("catch_model", "impact"),
                                     control_dt=cdt, lead=lead, reach=reach, reason=res["reason"], success=int(res["success"]),
                                     released=int(res.get("release_time") is not None), caught=int(res.get("catch_time") is not None),
                                     U_swing=out["U_swing_env"], U_star_swing=out["U_swing_star"], U_hold=out["U_hold_env"], U_peak=res["U_peak"],
                                     U_star=r["U_peak"], th_err=out["max_th_err_swing"], d_min=res["d_min"] if np.isfinite(res["d_min"]) else np.nan,
                                     t_end=res["t"], slip=float(getattr(env, "slip", 0.0))))
                    print(f"{os.path.basename(f)[8:-4]:22s} dt={cdt:<6g} lead={lead:4s} reach={reach:g}: {res['reason']:24s} rel={rows[-1]['released']} catch={rows[-1]['caught']} "
                          f"U_swing={out['U_swing_env']:.3f}/{out['U_swing_star']:.3f} th_err={out['max_th_err_swing']:.3f} d_min={rows[-1]['d_min']:.3f} U_hold={out['U_hold_env']:.3f}", flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d.to_csv(a.out, index=False)
    g = d.groupby(["control_dt", "lead", "reach"]).agg(n=("success", "size"), released=("released", "mean"), caught=("caught", "mean"),
                                                        success=("success", "mean"), th_err=("th_err", "median"), U_swing=("U_swing", "median"),
                                                        U_star=("U_star_swing", "median")).reset_index()
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
