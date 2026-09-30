"""Release windows at 0.5 ms resolution (all 54 best-phase solutions) vs the 5 ms grid.

Reads results/windows/windows_ref_fine.json (run_windows --dt-fine 0.0005 --suffix _fine) and windows_ref.json.
Outputs: results/figs/windows_fine.pdf/png (histogram of catch windows + example), results/figs/windows_fine.json.

usage: python scripts/analyze_windows_fine.py --out results/figs
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", default="results/windows/windows_ref_fine.json")
    ap.add_argument("--coarse", default="results/windows/windows_ref.json")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    F = json.load(open(a.fine))
    C = {w["file"]: w for w in json.load(open(a.coarse))} if os.path.exists(a.coarse) else {}
    rows = []
    for w in F:
        det = w["plan_detail"]
        ds = np.array(sorted(float(k) for k in det))
        caught = np.array([det[str(d) if str(d) in det else f"{d}"]["caught"] for d in ds])
        held = np.array([det[str(d) if str(d) in det else f"{d}"]["reason"] == "held" for d in ds])
        fine = np.abs(ds) <= 0.04 + 1e-9
        dsf, cf = ds[fine], caught[fine]
        # catch window = extent of caught deltas within the fine region (+ one step)
        win = (dsf[cf].max() - dsf[cf].min() + 0.0005) if cf.any() else 0.0
        # first/last caught delta
        rows.append(dict(T=w["T"], m=w["m"], phi=w["phi_l"], U_star=w["U_star"], window_ms=1e3 * win,
                         d_lo_ms=1e3 * dsf[cf].min() if cf.any() else np.nan, d_hi_ms=1e3 * dsf[cf].max() if cf.any() else np.nan,
                         n_caught=int(cf.sum()), U0=w.get("plan_U0_max", np.nan), ratio=w.get("plan_U0_max", np.nan) / w["U_star"],
                         reason0=w.get("plan_reason0"), coarse_ms=1e3 * C[w["file"]]["plan_catch_window_s"] if w["file"] in C else np.nan,
                         hang_window_ms=1e3 * w.get("hang_catch_window_s", np.nan)))
    df = pd.DataFrame(rows).sort_values(["T", "m"])
    df.to_csv(os.path.join(a.out, "windows_fine.csv"), index=False)
    summ = dict(n=int(len(df)), window_median_ms=float(df["window_ms"].median()), window_min_ms=float(df["window_ms"].min()),
                window_max_ms=float(df["window_ms"].max()), window_mean_ms=float(df["window_ms"].mean()),
                ratio_median=float(df["ratio"].median()), ratio_max=float(df["ratio"].max()),
                coarse_median_ms=float(df["coarse_ms"].median()) if df["coarse_ms"].notna().any() else None,
                center_median_ms=float((0.5 * (df["d_lo_ms"] + df["d_hi_ms"])).median()),
                n_held=int((df["reason0"] == "held").sum()))
    json.dump(summ, open(os.path.join(a.out, "windows_fine.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(df.round(2).to_string(index=False))
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.hist(df["window_ms"], bins=np.arange(0, df["window_ms"].max() + 1.0, 0.5), color=CAT[0], alpha=0.9)
    ax.set_xlabel("catch window (0.5 ms grid) [ms]"); ax.set_ylabel("conditions"); ax.set_title("54 best-phase plans", fontsize=8.5)
    ax = axes[1]
    ax.scatter(df["m"] + (df["T"] - 20) * 0.15, df["window_ms"], c=[CAT[1] if p < 0.5 else CAT[2] for p in df["phi"]], s=14)
    ax.set_xlabel("body mass m [kg] (jitter = T)"); ax.set_ylabel("catch window [ms]")
    ax.set_title("orange: outbound release, green: return", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "windows_fine.pdf")); fig.savefig(os.path.join(a.out, "windows_fine.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
