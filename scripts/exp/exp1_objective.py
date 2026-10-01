"""Experiment 1: does the optimiser really compute the "required grip capacity"?

For 12 conditions (T = 18 s, m in {60, 66, 75} kg, phi_l in {0.25, 0.3125, 0.375, 0.75}) the full problem is solved with
  current   : J = 10 U_peak + E_eff
  w1 / w100 : J =  1 U_peak + E_eff,  J = 100 U_peak + E_eff
  lexico    : stage 1 minimises U_peak (w_U = 100, w_E = 0.01) -> U_min; stage 2 minimises E_eff under U_peak <= 1.01 U_min
with the same warm start (the reference solution of the condition or its nearest phase neighbour) and the same budget.
Recorded: U_peak, E_eff, release time, catch relative velocity, per-phase peak loads, min wall clearance, min joint margin.

usage: python scripts/exp/exp1_objective.py --grid results/grid --out results/suite/exp1 --workers 12
"""
import argparse, glob, os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import load_ref, solve_variant, solution_metrics
from katsumi.exp.util import pool_map

OBJECTIVES = {"current": dict(w_U=10.0, w_E=1.0), "w1": dict(w_U=1.0, w_E=1.0), "w100": dict(w_U=100.0, w_E=1.0),
              "lexico1": dict(w_U=100.0, w_E=0.01)}


def nearest_ref(grid, T, m, phi):
    fs = sorted(glob.glob(os.path.join(grid, f"sol_ref_T{T:g}_m{m:g}_phi*.pkl")))
    best, bd = None, 9
    for f in fs:
        ph = float(f.split("_phi")[1][:-4])
        d = abs(ph - phi)
        if d < bd:
            r = pickle.load(open(f, "rb"))
            if r.get("ok"):
                best, bd = f, d
    return best


def task(t):
    ref = load_ref(t["ref"])
    out = {}
    base_fn = os.path.join(t["out"], f"sol_{t['cond']}_{t['obj']}.pkl")
    ov = dict(OBJECTIVES[t["obj"]]); ov["fixed_release_phase"] = t["phi"]
    # the reference may be a phase neighbour: the phase is fixed to the condition's phase
    ref = dict(ref); ref["phi_l"] = t["phi"]
    r1 = solve_variant(ref, overrides=ov, fn=base_fn, max_cpu=t["max_cpu"])
    rows = []
    met = solution_metrics(r1); met.update(task_key=t["task_key"], cond=t["cond"], obj=t["obj"], T=ref["T"], m=ref["m"], phi=t["phi"], stage=1,
                                           solve_s=r1.get("solve_s"), ref_used=os.path.basename(t["ref"]))
    rows.append(met)
    if t["obj"] == "lexico1" and r1.get("ok"):
        # stage 2: minimise the effort under U_peak <= 1.01 U_min (warm start: the stage-1 solution)
        fn2 = os.path.join(t["out"], f"sol_{t['cond']}_lexico2.pkl")
        r2 = solve_variant(r1, overrides=dict(w_U=0.0, w_E=1.0, U_max=1.01 * r1["U_peak"], fixed_release_phase=t["phi"]), fn=fn2, max_cpu=t["max_cpu"], cold=False)
        met2 = solution_metrics(r2); met2.update(task_key=t["task_key"] + "_s2", cond=t["cond"], obj="lexico2", T=ref["T"], m=ref["m"], phi=t["phi"], stage=2,
                                                 solve_s=r2.get("solve_s"), U_min_stage1=r1["U_peak"], ref_used=os.path.basename(t["ref"]))
        rows.append(met2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--out", default="results/suite/exp1")
    ap.add_argument("--T", type=float, nargs="+", default=[18.0])
    ap.add_argument("--m", type=float, nargs="+", default=[60.0, 66.0, 75.0])
    ap.add_argument("--phi", type=float, nargs="+", default=[0.25, 0.3125, 0.375, 0.75])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-cpu", type=float, default=2400.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = []
    for T in a.T:
        for m in a.m:
            for phi in a.phi:
                ref = nearest_ref(a.grid, T, m, phi)
                if ref is None:
                    print("no reference for", T, m, phi); continue
                cond = f"T{T:g}_m{m:g}_phi{phi:.4f}"
                for obj in OBJECTIVES:
                    tasks.append(dict(task_key=f"{cond}_{obj}", ref=ref, cond=cond, obj=obj, phi=phi, out=a.out, max_cpu=a.max_cpu))
    df = pool_map(task, tasks, a.workers, os.path.join(a.out, "table.csv"), "exp1")
    if len(df) and "U_peak" in df:
        piv = df.pivot_table(index="cond", columns="obj", values="U_peak")
        print(piv.round(3).to_string())
        piv.to_csv(os.path.join(a.out, "U_peak_by_objective.csv"))
        pe = df.pivot_table(index="cond", columns="obj", values="effort"); pe.to_csv(os.path.join(a.out, "effort_by_objective.csv"))


if __name__ == "__main__":
    main()
