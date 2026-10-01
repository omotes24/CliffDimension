"""Experiment 3: do margins on the plan side raise the closed-loop success rate? (intervention on the 2 x 2 design)

  A: no margin, nominal plan           B: constraint margins (wall +2/5 mm, joint range -2/5 deg, cone -5/10 %)
  C: release-time error scenarios      D: margins + scenarios
Only the planner is made conservative; the environment (physics, actual capacity) is unchanged. Each variant is executed
unperturbed, with release-time errors not used in planning (-3, -1, +1, +3 ms) and with weak / strong state perturbations
of the rest start (10 seeds each). Reported: success vs allowed capacity, exceedance rate, capacity and effort increments.

usage: python scripts/exp/exp3_margins.py --grid results/grid --comp results/comp_t18 --out results/suite/exp3 --workers 12
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import load_ref, solve_variant, solution_metrics, F_MAX_LEVELS
from katsumi.exp.trials import replay_trial
from katsumi.exp.util import pool_map, find_refs

DEG = np.pi / 180
VARIANTS = {
    "A_base": dict(),
    "B_wall2mm": dict(wall_margin=0.002), "B_wall5mm": dict(wall_margin=0.005),
    "B_joint2deg": dict(joint_margin=2 * DEG), "B_joint5deg": dict(joint_margin=5 * DEG),
    "B_cone5": dict(mu_scale=0.95), "B_cone10": dict(mu_scale=0.90),
    "B_all_small": dict(wall_margin=0.002, joint_margin=2 * DEG, mu_scale=0.95),
    "B_all_large": dict(wall_margin=0.005, joint_margin=5 * DEG, mu_scale=0.90),
    "C_robust2ms": dict(robust_deltas=(-0.002, 0.0, 0.002)),
    "D_small_robust": dict(wall_margin=0.002, joint_margin=2 * DEG, mu_scale=0.95, robust_deltas=(-0.002, 0.0, 0.002)),
    "D_large_robust": dict(wall_margin=0.005, joint_margin=5 * DEG, mu_scale=0.90, robust_deltas=(-0.002, 0.0, 0.002)),
}
PERTURB = {"none": (0.0, 0.0, 1), "weak": (0.01, 0.1, 10), "strong": (0.03, 0.2, 10)}
DT_EVAL = (-0.003, -0.001, 0.001, 0.003)


def solve_task(t):
    ref = load_ref(t["ref"])
    fn = os.path.join(t["out"], "plans", os.path.basename(t["ref"]).replace(".pkl", f"_{t['variant']}.pkl"))
    r = solve_variant(ref, overrides=VARIANTS[t["variant"]], fn=fn, max_cpu=t["max_cpu"])
    met = solution_metrics(r) if r.get("ok") else dict(ok=False, status=r.get("status"))
    met.update(task_key=t["task_key"], ref=os.path.basename(t["ref"]), variant=t["variant"], plan_file=fn, solve_s=r.get("solve_s"),
               U_peak_base=ref["U_peak"], effort_base=ref["effort"])
    return met


def replay_task(t):
    r = load_ref(t["plan_file"])
    row = replay_trial(r, start="rest", sig_th=t["sig_th"], sig_thd=t["sig_thd"], seed=t["seed"], dt_release=t["dt_release"])
    row.update(task_key=t["task_key"], variant=t["variant"], perturb=t["perturb"], base_ref=t["base_ref"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--comp", default="results/comp_t18")
    ap.add_argument("--out", default="results/suite/exp3")
    ap.add_argument("--m", type=float, nargs="+", default=[60.0, 66.0, 75.0]); ap.add_argument("--phi", type=float, nargs="+", default=[0.25])
    ap.add_argument("--models", nargs="+", default=["compliant", "impact"])
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--max-cpu", type=float, default=2400.0)
    ap.add_argument("--variants", default=None, help="comma list of variants (default: all)"); ap.add_argument("--n-seeds", type=int, default=None)
    a = ap.parse_args()
    variants = a.variants.split(",") if a.variants else list(VARIANTS)
    if a.n_seeds is not None:
        for k in PERTURB:
            PERTURB[k] = (PERTURB[k][0], PERTURB[k][1], min(PERTURB[k][2], a.n_seeds))
    os.makedirs(a.out, exist_ok=True)
    refs = []
    if "compliant" in a.models: refs += find_refs(a.comp, [18.0], a.m, a.phi, tag="comp")
    if "impact" in a.models: refs += find_refs(a.grid, [18.0], a.m, a.phi, tag="ref")
    tasks = [dict(task_key=f"{os.path.basename(f)}_{v}", ref=f, variant=v, out=a.out, max_cpu=a.max_cpu) for f in refs for v in variants]
    ds = pool_map(solve_task, tasks, a.workers, os.path.join(a.out, "plans.csv"), "exp3-plans")
    plans = ds[ds["ok"] == True] if "ok" in ds else ds
    tasks = []
    for pr in plans.itertuples():
        for pname, (st, sd, n) in PERTURB.items():
            for seed in range(n):
                dts = DT_EVAL if pname == "none" else (0.0,)
                for dt in ((0.0,) + tuple(dts)) if pname == "none" else dts:
                    tasks.append(dict(task_key=f"{pr.ref}_{pr.variant}_{pname}_{seed}_{dt}", plan_file=pr.plan_file, variant=pr.variant, perturb=pname,
                                      sig_th=st, sig_thd=sd, seed=seed, dt_release=dt, base_ref=pr.ref))
    df = pool_map(replay_task, tasks, a.workers, os.path.join(a.out, "trials.csv"), "exp3-trials")
    if len(df):
        cols = [f"succ_{lv:g}" for lv in F_MAX_LEVELS]
        agg = df.groupby(["catch_model", "variant", "perturb"])[cols + ["released", "caught", "success_held", "U_peak"]].mean()
        agg["n"] = df.groupby(["catch_model", "variant", "perturb"]).size()
        agg.to_csv(os.path.join(a.out, "success_by_variant.csv"))
        print(agg.round(3).to_string()[:8000])


if __name__ == "__main__":
    main()
