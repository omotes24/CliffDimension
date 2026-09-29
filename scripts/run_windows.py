"""Release-window analysis (plan Sec. 13.2) for grid solutions.

For each selected solution: shift the release command by delta in [-0.2, 0.2] s (0.005 s steps near 0,
0.01 s elsewhere), re-integrate with closed-loop joint tracking and the compliant finite-capacity catch,
and report (i) the catch window (deltas for which the hands hook B), (ii) the load ratio of the
re-simulated execution to the planned U*, (iii) the success window W for capacity margins.

usage: python scripts/run_windows.py --grid results/grid --tag ref --out results/windows [--best-only]
"""
import argparse, glob, os, sys, pickle, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.anthro import make_body
from katsumi.planar.simulate import release_window, window_width

MARGINS = (1.0, 1.1, 1.25, 1.5, 2.0)


def analyse(pkl, deltas, modes=("plan", "hang")):
    r = pickle.load(open(pkl, "rb"))
    body = make_body(r["m"])
    out = dict(T=r["T"], m=r["m"], phi_l=r["phi_l"], U_star=r["U_peak"], file=os.path.basename(pkl))
    for mode in modes:
        win = release_window(body, r, deltas=deltas, cap_margins=MARGINS, hold_mode=mode)
        caught = [d for d, v in win.items() if v["caught"]]
        out[f"{mode}_catch_window_s"] = (max(caught) - min(caught) + (deltas[1] - deltas[0])) if caught else 0.0
        out[f"{mode}_catch_deltas"] = caught
        for c in MARGINS:
            out[f"{mode}_W_{c}"] = window_width(win, c)
        v0 = win.get(0.0)
        if v0 is not None:
            out[f"{mode}_U0_max"] = v0["U_max"]
            out[f"{mode}_U0_catch"] = v0.get("U_catch", np.nan)
            out[f"{mode}_U0_hold"] = v0.get("U_hold", np.nan)
            out[f"{mode}_reason0"] = v0["reason"]
        out[f"{mode}_detail"] = {str(k): dict(caught=v["caught"], reason=v["reason"], U_max=v["U_max"]) for k, v in win.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--out", default="results/windows")
    ap.add_argument("--best-only", action="store_true", help="only the minimum-U* phase of each (T, m)")
    ap.add_argument("--fine", type=float, default=0.03, help="half-width of the fine (5 ms) region around 0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.grid, f"rows_{a.tag}_T*_m*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["phi_l"] = df["phi_l"].round(4)
    df = df.sort_values("ok", ascending=False).drop_duplicates(["T", "m", "phi_l"], keep="first")
    df = df[df["ok"] == 1]
    if a.best_only:
        idx = df.groupby(["T", "m"])["U_peak"].idxmin()
        df = df.loc[idx]
    deltas = np.unique(np.round(np.concatenate([np.arange(-0.20, 0.2001, 0.01), np.arange(-a.fine, a.fine + 1e-9, 0.005)]), 3))
    rows = []
    for r in df.itertuples():
        pkl = os.path.join(a.grid, f"sol_{a.tag}_T{r.T:g}_m{r.m:g}_phi{r.phi_l:.3f}.pkl")
        if not os.path.exists(pkl):
            continue
        res = analyse(pkl, deltas)
        rows.append(res)
        print({k: v for k, v in res.items() if not k.endswith("detail") and not k.endswith("deltas")}, flush=True)
        json.dump(rows, open(os.path.join(a.out, f"windows_{a.tag}.json"), "w"), indent=1, default=float)
    summ = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("detail") and not k.endswith("deltas")} for r in rows])
    summ.to_csv(os.path.join(a.out, f"windows_{a.tag}.csv"), index=False)
    print(summ)


if __name__ == "__main__":
    main()
