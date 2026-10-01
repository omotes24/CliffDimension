"""Experiment 6: the central comparison -- the same MPC, data and architecture with and without the dual labels.

Methods (all with the same executor and the same shared landing reflex driven by the nearest stored reference):
  REPLAY  : reference feed-forward + PD tracking (the plan replayed)
  BC      : behaviour cloning (release timer shared)
  TRACK   : reference-tracking MPC (no learned value; releases at the reference's release time)
  VMPC    : MPC with the value-only ensemble as terminal cost        } identical horizon, re-plan period, margins,
  SMPC    : MPC with the Sobolev ensemble as terminal cost           } OOD penalty, solver budget, tracking, tau / U predictor
  ORACLE  : oracle re-optimisation every 0.5 s with the shared reflex (comparable condition)
  ORACLE+ : oracle re-optimisation handing its own flight / hold plan to the reflex (reference condition, extra information)
Main trials: rest starts with weak perturbations (sigma 0.01 rad, 0.1 rad/s; seed 0 unperturbed), every learning seed.
Diagnostic trials: perturbed starts in the middle of the swing (recovery), reported separately.

usage: python scripts/exp/exp6_compare.py --grid results/grid --models results/suite/models/full --out results/suite/exp6 --workers 16
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import load_ref, F_MAX_LEVELS, F_MAX_HEADLINE
from katsumi.exp.trials import replay_trial, controller_trial, make_env
from katsumi.exp.util import pool_map, bootstrap_ci

MPC_KW = dict(H=15, control_dt=0.02, replan=2, beta=2.0, beta_d=1.0, w_tau=20.0, kp_track=1.0, kd_track=0.1, U_cap_max=F_MAX_HEADLINE, max_cpu=30.0)


def build_controller(name, r, models_file, reflex_refs, mpc_kw=None):
    """mpc_kw: overrides of MPC_KW (e.g. the horizon ablation H = 40 -> 0.8 s)."""
    import torch; torch.set_num_threads(1)
    MPC_KW = {**globals()["MPC_KW"], **(mpc_kw or {})}
    from katsumi.learn.dual_field import load_models, BCController
    from katsumi.learn.swing_mpc import SwingMPC
    from katsumi.learn.oracle_mpc import OracleMPC
    env = make_env(r, reflex_refs=reflex_refs)
    body, chain, T = env.env.body, env.env.ch, r["T"]
    bk = r.get("body_kw", {}) or {}
    st, cs = bk.get("stature", 1.75), bk.get("cap_scale", 1.0)
    M = load_models(models_file) if models_file else None
    if name == "BC":
        return BCController(M["bc"], M["ens"][0], body, T, control_dt=0.002, stature=st, cap_scale=cs)
    if name == "TRACK":
        return SwingMPC(None, body, chain, T, track_ref=r, stature=st, cap_scale=cs, **MPC_KW)
    if name == "VMPC":
        return SwingMPC(M["ens0"], body, chain, T, density=M["density"], timer_net=M["ens"][0], stature=st, cap_scale=cs, **MPC_KW)
    if name == "SMPC":
        return SwingMPC(M["ens"], body, chain, T, density=M["density"], timer_net=M["ens"][0], stature=st, cap_scale=cs, **MPC_KW)
    if name == "ORACLE":
        return OracleMPC(r, body, T, replan_dt=0.5, control_dt=0.002, hand_over_plan=False, U_cap_max=F_MAX_HEADLINE)
    if name == "ORACLE+":
        return OracleMPC(r, body, T, replan_dt=0.5, control_dt=0.002, hand_over_plan=True, U_cap_max=F_MAX_HEADLINE)
    raise ValueError(name)


def task(t):
    r = load_ref(t["ref"])
    reflex_refs = [load_ref(f) for f in t["reflex_refs"]]
    if t["method"] == "REPLAY":
        row = replay_trial(r, start=t["start"], sig_th=t["sig_th"], sig_thd=t["sig_thd"], seed=t["seed"], knot=t.get("knot"), reflex_refs=reflex_refs)
        row["ctrl"] = "REPLAY"
    else:
        c = build_controller(t["method"], r, t.get("models_file"), reflex_refs, mpc_kw=t.get("mpc_kw"))
        row = controller_trial(c, r, start=t["start"], sig_th=t["sig_th"], sig_thd=t["sig_thd"], seed=t["seed"], knot=t.get("knot"), reflex_refs=reflex_refs, label=t["method"])
    row.update(task_key=t["task_key"], method=t["method"], learn_seed=t.get("learn_seed", -1), group=t["group"], models_file=os.path.basename(t.get("models_file") or ""))
    return row


def summarise(df, out):
    cols = [f"succ_{lv:g}" for lv in F_MAX_LEVELS]
    main = df[df["group"] == "main"]
    g = main.groupby("method")
    summ = g[cols + ["released", "caught", "success_held", "U_peak"]].mean(); summ["n"] = g.size()
    summ["n_learn_seeds"] = g["learn_seed"].nunique()
    summ.to_csv(os.path.join(out, "summary_main.csv")); print(summ.round(3).to_string())
    # per body and per learning seed
    main.groupby(["method", "m"])[cols].mean().to_csv(os.path.join(out, "summary_by_body.csv"))
    main.groupby(["method", "learn_seed"])[cols].mean().to_csv(os.path.join(out, "summary_by_seed.csv"))
    # paired difference Sobolev - value-only at the headline capacity
    key = f"succ_{F_MAX_HEADLINE:g}"
    a_ = main[main["method"] == "SMPC"].set_index(["ref", "seed", "learn_seed"])[key]
    b_ = main[main["method"] == "VMPC"].set_index(["ref", "seed", "learn_seed"])[key]
    j = a_.to_frame("S").join(b_.to_frame("V"), how="inner")
    res = {}
    if len(j):
        d = (j["S"] - j["V"]).values
        lo, hi = bootstrap_ci(d)
        res = dict(n_pairs=int(len(d)), delta_S=float(d.mean()), ci95=[lo, hi], S_sobolev=float(j["S"].mean()), S_value_only=float(j["V"].mean()),
                   per_seed={int(s): float(gg["S"].mean() - gg["V"].mean()) for s, gg in j.groupby(level="learn_seed")})
    json.dump(res, open(os.path.join(out, "delta_S.json"), "w"), indent=1); print(json.dumps(res, indent=1))
    # failure types
    ft = main.groupby(["method", "first_failure"]).size().unstack(fill_value=0); ft.to_csv(os.path.join(out, "first_failures.csv")); print(ft.to_string())
    rec = df[df["group"] == "recovery"]
    if len(rec):
        rs = rec.groupby("method")[cols + ["released", "caught"]].mean(); rs["n"] = rec.groupby("method").size()
        rs.to_csv(os.path.join(out, "summary_recovery.csv")); print(rs.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--models", default="results/suite/models/full")
    ap.add_argument("--out", default="results/suite/exp6"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--m", type=float, nargs="+", default=[60.0, 66.0, 69.0, 72.0]); ap.add_argument("--T", type=float, nargs="+", default=[18.0])
    ap.add_argument("--phi", type=float, default=0.25)
    ap.add_argument("--n-perturb", type=int, default=10); ap.add_argument("--sig-th", type=float, default=0.01); ap.add_argument("--sig-thd", type=float, default=0.1)
    ap.add_argument("--methods", default="REPLAY,BC,TRACK,VMPC,SMPC,ORACLE,ORACLE+")
    ap.add_argument("--recovery-knots", type=int, default=5)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--mpc-H", type=int, default=None, help="MPC horizon in plan steps of 20 ms (default 15 = 0.3 s)")
    ap.add_argument("--mpc-max-cpu", type=float, default=None); ap.add_argument("--learn-seeds", type=int, default=None, help="use only the first n learning seeds")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, "trials.csv")
    if a.summary_only:
        summarise(pd.read_csv(out_csv), a.out); return
    refs = [os.path.join(a.grid, f"sol_ref_T{T:g}_m{m:g}_phi{a.phi:.3f}.pkl") for T in a.T for m in a.m]
    refs = [f for f in refs if os.path.exists(f)]
    model_files = sorted(glob.glob(os.path.join(a.models, "models_seed*.pt")))[:a.learn_seeds]
    mpc_kw = {k: v for k, v in (("H", a.mpc_H), ("max_cpu", a.mpc_max_cpu)) if v is not None}
    reflex_refs = refs                                     # the shared reflex knows the stored references (nearest body is used)
    methods = a.methods.split(",")
    tasks = []
    for f in refs:
        base = os.path.basename(f)[8:-4]
        for seed in range(a.n_perturb):
            st, sd = (0.0, 0.0) if seed == 0 else (a.sig_th, a.sig_thd)
            for meth in methods:
                files = model_files if meth in ("BC", "VMPC", "SMPC") else [None]
                for mf in files:
                    ls = int(os.path.basename(mf)[11:-3]) if mf else -1
                    tasks.append(dict(task_key=f"main_{base}_{meth}_s{seed}_l{ls}", ref=f, reflex_refs=reflex_refs, method=meth, start="rest", sig_th=st, sig_thd=sd,
                                      seed=seed, models_file=mf, learn_seed=ls, group="main"))
        # recovery diagnostics: perturbed starts in the middle of the swing (first learning seed only)
        r = load_ref(f); N = r["S_X"].shape[1] - 1
        for j, fr in enumerate(np.linspace(0.2, 0.8, a.recovery_knots)):
            k = int(round(fr * N))
            for meth in [m_ for m_ in methods if m_ in ("BC", "VMPC", "SMPC", "ORACLE", "REPLAY")]:
                mf = model_files[0] if (meth in ("BC", "VMPC", "SMPC") and model_files) else None
                tasks.append(dict(task_key=f"rec_{base}_{meth}_k{k}", ref=f, reflex_refs=reflex_refs, method=meth, start="knot", knot=k, sig_th=0.03, sig_thd=0.2,
                                  seed=100 + j, models_file=mf, learn_seed=(0 if mf else -1), group="recovery"))
    # cheap methods first, then the heavy ones
    order = {"REPLAY": 0, "BC": 1, "TRACK": 2, "VMPC": 3, "SMPC": 3, "ORACLE": 4, "ORACLE+": 4}
    for t in tasks:
        t["mpc_kw"] = mpc_kw
    tasks.sort(key=lambda t: order[t["method"]])
    df = pool_map(task, tasks, a.workers, out_csv, "exp6")
    if len(df):
        summarise(df, a.out)


if __name__ == "__main__":
    main()
