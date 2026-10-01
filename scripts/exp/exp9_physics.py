"""Experiment 9: the physical conclusions checked by re-computation and intervention (the 864-case grid itself runs in the
grid pipeline; this script adds the interventions).

9.2  "the hold dominates": the capacity constraint of ONE phase (swing / catch / hold) is relaxed by 10 % and the problem
     re-solved; hold-time variants 0.5 / 1 / 3 s (the main success rule stays 2 s).
9.3  capability sensitivity by re-optimisation: joint torque capacity x (0.99, 1.01, 0.95, 1.05, 0.9, 1.1) for all joints and
     +-5 % per joint (elbow / shoulder / hip / knee); dJ/dln tau and dU_peak/dln tau reported separately and compared with
     the multiplier-based shadow price of the base solution.
9.4  catch-model dependence: impact averaging time 25 / 50 / 100 ms; compliant stiffness x0.5 / x2 and damping x0.5 / x2;
     the window-averaged load (common 50 ms window) is reported next to the instantaneous peak of the compliant model.

usage: python scripts/exp/exp9_physics.py --grid results/grid --comp results/comp_t18 --out results/suite/exp9 --workers 16
"""
import argparse, json, os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import load_ref, solve_variant, solution_metrics
from katsumi.exp.util import pool_map, find_refs


def task(t):
    ref = load_ref(t["ref"])
    r = solve_variant(ref, overrides=t.get("ov"), body_kw=t.get("body_kw"), fn=t["fn"], max_cpu=t["max_cpu"])
    met = solution_metrics(r) if r.get("ok") else dict(ok=False, status=r.get("status"), U_peak=r.get("U_peak"), effort=r.get("effort"))
    met.update(task_key=t["task_key"], part=t["part"], variant=t["name"], ref=os.path.basename(t["ref"]), m=ref["m"], T=ref["T"], solve_s=r.get("solve_s"),
               U_base=ref["U_peak"], effort_base=ref["effort"], J_base=ref.get("J", 10 * ref["U_peak"] + ref["effort"]))
    sens = (r.get("duals") or {}).get("dJ_dlnp", {}) if r.get("ok") else {}
    met["sens_tau_cap"] = float(np.sum(sens.get("tau_cap", [np.nan])))
    met["sens_tau_joint"] = json.dumps([float(v) for v in sens.get("tau_cap", [])])
    # window-averaged catch load of the compliant model (50 ms moving average of |C_R|) next to the instantaneous peak
    if r.get("ok") and "C_R" in r and r.get("catch_model") == "compliant":
        U = np.sqrt((r["C_R"] ** 2).sum(0)); n = len(U); dt = r["d_c"] / max(n - 1, 1)
        w = max(1, int(round(0.05 / dt)))
        avg = np.convolve(U, np.ones(w) / w, mode="valid")
        met["U_catch_inst"] = float(U.max()); met["U_catch_avg50ms"] = float(avg.max())
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--comp", default="results/comp_t18"); ap.add_argument("--out", default="results/suite/exp9")
    ap.add_argument("--m", type=float, nargs="+", default=[60.0, 66.0, 75.0]); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--max-cpu", type=float, default=2400.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs = find_refs(a.grid, [18.0], a.m, [0.25]); comps = find_refs(a.comp, [18.0], a.m, [0.25], tag="comp")
    tasks = []
    for f in refs:
        b = os.path.basename(f)[8:-4]
        # 9.2 phase relaxation / hold time
        for ph, nm in (("S", "swing"), ("catch", "catch"), ("H", "hold")):
            tasks.append(dict(task_key=f"{b}_relax_{nm}", part="9.2", name=f"relax_{nm}_10pct", ref=f, ov=dict(cap_relax=((ph, 1.1),)), fn=os.path.join(a.out, "sol", f"{b}_relax_{nm}.pkl"), max_cpu=a.max_cpu))
        for th in (0.5, 1.0, 3.0):
            tasks.append(dict(task_key=f"{b}_hold{th}", part="9.2", name=f"T_hold={th}", ref=f, ov=dict(T_hold=th, N_H=max(20, int(30 * th))), fn=os.path.join(a.out, "sol", f"{b}_hold{th}.pkl"), max_cpu=a.max_cpu))
        tasks.append(dict(task_key=f"{b}_base", part="base", name="base", ref=f, ov={}, fn=os.path.join(a.out, "sol", f"{b}_base.pkl"), max_cpu=a.max_cpu))
        # 9.3 capability sensitivity
        for s in (0.99, 1.01, 0.95, 1.05, 0.9, 1.1):
            tasks.append(dict(task_key=f"{b}_tau_all{s}", part="9.3", name=f"tau_all x{s}", ref=f, ov=dict(tau_scale=(s, s, s, s)), fn=os.path.join(a.out, "sol", f"{b}_tau_all{s}.pkl"), max_cpu=a.max_cpu))
        for j, jn in enumerate(("elbow", "shoulder", "hip", "knee")):
            for s in (0.95, 1.05):
                ts = [1.0] * 4; ts[j] = s
                tasks.append(dict(task_key=f"{b}_tau_{jn}{s}", part="9.3", name=f"tau_{jn} x{s}", ref=f, ov=dict(tau_scale=tuple(ts)), fn=os.path.join(a.out, "sol", f"{b}_tau_{jn}{s}.pkl"), max_cpu=a.max_cpu))
        # 9.4 impact averaging time
        for dc in (0.025, 0.1):
            tasks.append(dict(task_key=f"{b}_dcatch{dc}", part="9.4", name=f"delta_catch={dc}", ref=f, ov=dict(delta_catch=dc), fn=os.path.join(a.out, "sol", f"{b}_dcatch{dc}.pkl"), max_cpu=a.max_cpu))
    for f in comps:
        b = os.path.basename(f)[9:-4]
        tasks.append(dict(task_key=f"comp_{b}_base", part="9.4", name="compliant base", ref=f, ov={}, fn=os.path.join(a.out, "sol", f"comp_{b}_base.pkl"), max_cpu=a.max_cpu))
        for K in (20000.0, 80000.0):
            tasks.append(dict(task_key=f"comp_{b}_K{K:g}", part="9.4", name=f"K_catch={K:g}", ref=f, ov=dict(K_catch=K), fn=os.path.join(a.out, "sol", f"comp_{b}_K{K:g}.pkl"), max_cpu=a.max_cpu))
        for D in (750.0, 3000.0):
            tasks.append(dict(task_key=f"comp_{b}_D{D:g}", part="9.4", name=f"D_catch={D:g}", ref=f, ov=dict(D_catch=D), fn=os.path.join(a.out, "sol", f"comp_{b}_D{D:g}.pkl"), max_cpu=a.max_cpu))
    if a.limit:
        tasks = tasks[:a.limit]
    df = pool_map(task, tasks, a.workers, os.path.join(a.out, "table.csv"), "exp9")
    if len(df):
        cols = [c for c in ("part", "variant", "m", "ok", "U_peak", "U_base", "effort", "effort_base", "J", "J_base", "U_catch_inst", "U_catch_avg50ms", "sens_tau_cap") if c in df]
        print(df[cols].round(4).to_string(index=False)[:12000])
        # 9.3 finite-difference sensitivities vs the multiplier price (all joints)
        rows = []
        for m, g in df[df["part"].isin(["9.3", "base"])].groupby("m"):
            base = g[g["variant"] == "base"]
            if not len(base) or not base["ok"].iloc[0]:
                continue
            J0, U0, S0 = float(base["J"].iloc[0]), float(base["U_peak"].iloc[0]), float(base["sens_tau_cap"].iloc[0])
            for s in (0.01, 0.05, 0.1):
                up = g[g["variant"] == f"tau_all x{1 + s}"]; dn = g[g["variant"] == f"tau_all x{1 - s:g}"]
                if len(up) and len(dn) and up["ok"].iloc[0] and dn["ok"].iloc[0]:
                    dl = np.log(1 + s) - np.log(1 - s)
                    rows.append(dict(m=m, step=s, dJ_dlntau_fd=(float(up["J"].iloc[0]) - float(dn["J"].iloc[0])) / dl, dJ_dlntau_mult=S0,
                                     dU_dlntau_fd=(float(up["U_peak"].iloc[0]) - float(dn["U_peak"].iloc[0])) / dl, elasticity_U=(float(up["U_peak"].iloc[0]) - float(dn["U_peak"].iloc[0])) / dl / U0))
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "capability_sensitivity.csv"), index=False); print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
