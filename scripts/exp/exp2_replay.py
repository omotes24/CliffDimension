"""Experiment 2: where does the replay of an oracle trajectory break? (no learner involved)

Main condition: compliant-catch solutions executed with the compliant environment; impact-model solutions executed with
the same environment are the model-difference condition. Three time resolutions are varied separately:
  mesh   : the collocation mesh of the solution (0.5x, 1x, 2x intervals; re-solved, warm-started)   [replay fixed]
  sub_dt : the integration step 1 / 0.5 / 0.25 ms                                                      [control 2 ms]
  ctrl   : the control period 1 / 2 / 5 / 10 / 20 ms                                                    [sub_dt 0.25 ms]
plus the release switching (exact scheduled time vs rounded to the control period) and phase-wise starts (rest / flight /
catch / hold, with the reference's internal state).

usage: python scripts/exp/exp2_replay.py --grid results/grid --comp results/comp_t18 --out results/suite/exp2 --workers 12
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import load_ref, solve_variant
from katsumi.exp.trials import replay_trial
from katsumi.exp.util import pool_map, find_refs

MESH = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}


def mesh_task(t):
    ref = load_ref(t["ref"])
    f = MESH[t["mesh"]]
    if f == 1.0:
        return dict(task_key=t["task_key"], ref=t["ref"], mesh=t["mesh"], out_file=t["ref"], ok=int(bool(ref.get("ok"))), U_peak=ref["U_peak"])
    p = ref["params"]
    ov = dict(N_S=int(round(p["N_S"] * f)), N_F=int(round(p["N_F"] * f)), N_H=int(round(p["N_H"] * f)), N_C=int(round(p.get("N_C", 40) * f)))
    fn = os.path.join(t["out"], "mesh", os.path.basename(t["ref"]).replace(".pkl", f"_mesh{t['mesh']}.pkl"))
    r = solve_variant(ref, overrides=ov, fn=fn, max_cpu=t["max_cpu"])
    return dict(task_key=t["task_key"], ref=t["ref"], mesh=t["mesh"], out_file=fn, ok=int(bool(r.get("ok"))), U_peak=r["U_peak"], d_s=r["d_s"],
                solve_s=r.get("solve_s"), iters=r.get("iters"))


def replay_task(t):
    r = load_ref(t["file"])
    row = replay_trial(r, control_dt=t["control_dt"], sub_dt=t["sub_dt"], exact=t["exact"], start=t["start"], dt_release=t.get("dt_release", 0.0))
    row.update(task_key=t["task_key"], group=t["group"], mesh=t.get("mesh", "1x"), ref_file=os.path.basename(t["file"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--comp", default="results/comp_t18")
    ap.add_argument("--out", default="results/suite/exp2")
    ap.add_argument("--m", type=float, nargs="+", default=[60.0, 66.0, 75.0])
    ap.add_argument("--phi", type=float, nargs="+", default=[0.25])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-cpu", type=float, default=2400.0)
    ap.add_argument("--skip-mesh", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs_comp = find_refs(a.comp, [18.0], a.m, a.phi, tag="comp")
    refs_imp = find_refs(a.grid, [18.0], a.m, a.phi, tag="ref")
    refs = [(f, "compliant") for f in refs_comp] + [(f, "impact") for f in refs_imp]
    print("references:", [os.path.basename(f) for f, _ in refs])
    # ---- (a) mesh variants (solver) --------------------------------------------------------------------------
    mesh_files = {}
    if not a.skip_mesh:
        tasks = [dict(task_key=f"{os.path.basename(f)}_{mk}", ref=f, mesh=mk, out=a.out, max_cpu=a.max_cpu) for f, _ in refs for mk in MESH]
        dm = pool_map(mesh_task, tasks, a.workers, os.path.join(a.out, "mesh_solves.csv"), "exp2-mesh")
        for r in dm.itertuples():
            if getattr(r, "ok", 0):
                mesh_files[(r.ref, r.mesh)] = r.out_file
    # ---- (b)-(e) replays -------------------------------------------------------------------------------------
    tasks = []
    for f, model in refs:
        base = os.path.basename(f)
        # integration step at control 2 ms
        for sd in (0.001, 0.0005, 0.00025):
            tasks.append(dict(task_key=f"{base}_sub{sd}", file=f, control_dt=0.002, sub_dt=sd, exact=True, start="rest", group="sub_dt"))
        # control period at sub 0.25 ms
        for cd in (0.001, 0.002, 0.005, 0.01, 0.02):
            tasks.append(dict(task_key=f"{base}_ctrl{cd}", file=f, control_dt=cd, sub_dt=0.00025, exact=True, start="rest", group="control_dt"))
            tasks.append(dict(task_key=f"{base}_ctrl{cd}_rounded", file=f, control_dt=cd, sub_dt=0.00025, exact=False, start="rest", group="release_rounding"))
        # phase-wise starts at the fine setting
        for st in ("rest", "flight", "catch", "hold"):
            tasks.append(dict(task_key=f"{base}_start_{st}", file=f, control_dt=0.002, sub_dt=0.00025, exact=True, start=st, group="phase_start"))
        # mesh variants executed with the fine setting
        for mk in MESH:
            mf = f if mk == "1x" else mesh_files.get((f, mk))
            if mf:
                tasks.append(dict(task_key=f"{base}_mesh{mk}", file=mf, control_dt=0.002, sub_dt=0.00025, exact=True, start="rest", group="mesh", mesh=mk))
    df = pool_map(replay_task, tasks, a.workers, os.path.join(a.out, "replays.csv"), "exp2-replay")
    # ---- convergence summary -----------------------------------------------------------------------------------
    if len(df):
        df["catch_model"] = df["catch_model"].fillna("impact")
        summ = []
        for (model, group), g in df.groupby(["catch_model", "group"]):
            key = {"sub_dt": "sub_dt", "control_dt": "control_dt", "release_rounding": "control_dt", "mesh": "mesh", "phase_start": "start"}[group]
            for ref_file, gg in g.groupby("ref_file"):
                gg = gg.sort_values(key) if key != "mesh" else gg
                for r in gg.itertuples():
                    summ.append(dict(catch_model=model, group=group, ref=ref_file, setting=getattr(r, key), reason=r.reason, U_peak=r.U_peak,
                                     catch_hx=r.catch_hx, catch_hy=r.catch_hy, released=r.released, caught=r.caught, held=r.success_held,
                                     swing_err=r.swing_err, first_failure=r.first_failure, t_first_failure=r.t_first_failure))
        pd.DataFrame(summ).to_csv(os.path.join(a.out, "summary.csv"), index=False)
        print(pd.DataFrame(summ).to_string(index=False)[:6000])


if __name__ == "__main__":
    main()
