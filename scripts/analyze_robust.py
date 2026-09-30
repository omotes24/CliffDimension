"""Required capacity vs release-timing tolerance (robust scenario optimisation) and its re-simulated verification.

Reads results/robust/rows_rob*_T20_m66_phi*.csv (+ pickles) written by run_grid with robust_deltas, the nominal
grid solutions, and (optionally) fine re-simulated windows of the robust plans (results/windows_robust/*.json).
Outputs results/figs/robust_pareto.pdf/png, paper/tab_robust.tex, results/figs/robust.json.

usage: python scripts/analyze_robust.py --out results/figs
"""
import argparse, glob, json, os, pickle, re, sys
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
    ap.add_argument("--robust", default="results/robust")
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--windows", default="results/windows_robust")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.robust, "sol_rob*_T20_m66_phi*.pkl"))):
        r = pickle.load(open(f, "rb"))
        tag = re.search(r"sol_(rob\d+(?:cl)?)_", os.path.basename(f)).group(1)
        dw = int(re.search(r"rob(\d+)", tag).group(1)) / 1000
        rows.append(dict(tag=tag, dw_ms=1e3 * dw, closed_loop=tag.endswith("cl"), phi=round(r["phi_l"], 4), ok=bool(r["ok"]),
                         U_peak=r["U_peak"], effort=r["effort"], d_f=r["d_f"], file=os.path.basename(f),
                         scen_Ucatch=[s["U_catch"] for s in r.get("scen", [])]))
    # nominal (dw = 0) from the grid
    for ph in (0.25, 0.6875):
        f = os.path.join(a.grid, f"sol_ref_T20_m66_phi{ph:.3f}.pkl")
        if os.path.exists(f):
            r = pickle.load(open(f, "rb"))
            for cl in (False, True):
                rows.append(dict(tag="nominal", dw_ms=0.0, closed_loop=cl, phi=round(r["phi_l"], 4), ok=bool(r["ok"]), U_peak=r["U_peak"],
                                 effort=r["effort"], d_f=r["d_f"], file=os.path.basename(f), scen_Ucatch=[]))
    df = pd.DataFrame(rows)
    # re-simulated windows of the robust plans (fine delta grid), if available
    win = {}
    for f in glob.glob(os.path.join(a.windows, "windows_*_fine.json")) + glob.glob(os.path.join(a.windows, "windows_*.json")):
        for w in json.load(open(f)):
            win[w["file"]] = w
    df["catch_window_ms"] = [1e3 * win[fn]["plan_catch_window_s"] if fn in win else np.nan for fn in df["file"]]
    df["W_1.0_ms"] = [1e3 * win[fn].get("plan_W_1.0", np.nan) if fn in win else np.nan for fn in df["file"]]
    df["W_1.1_ms"] = [1e3 * win[fn].get("plan_W_1.1", np.nan) if fn in win else np.nan for fn in df["file"]]
    df = df.sort_values(["phi", "closed_loop", "dw_ms"])
    df.to_csv(os.path.join(a.out, "robust.csv"), index=False)
    print(df[["tag", "phi", "closed_loop", "dw_ms", "ok", "U_peak", "d_f", "catch_window_ms", "W_1.0_ms", "W_1.1_ms"]].round(4).to_string(index=False))
    ok = df[df["ok"]]
    # ---- figure ---------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for i, (ph, cl) in enumerate([(0.25, False), (0.6875, False), (0.25, True), (0.6875, True)]):
        d = ok[(ok["phi"] == ph) & (ok["closed_loop"] == cl)].sort_values("dw_ms")
        if len(d):
            ax.plot(d["dw_ms"], d["U_peak"] * 1300 / (66 * 9.81), "-o" if not cl else "--s", ms=4, lw=1.4, color=CAT[i % 2],
                    label=f"$\\phi_\\ell$={ph:.3f}, {'closed loop' if cl else 'open loop'}")
    ax.set_xlabel(r"designed release tolerance $\pm\delta_w$ [ms]"); ax.set_ylabel("required capacity [BW]")
    ax.set_title("capacity–tolerance trade-off (T = 20 s, m = 66 kg)", fontsize=8.5); ax.legend(fontsize=6.5)
    ax = axes[1]
    for i, (ph, cl) in enumerate([(0.25, False), (0.6875, False), (0.25, True), (0.6875, True)]):
        d = ok[(ok["phi"] == ph) & (ok["closed_loop"] == cl)].sort_values("dw_ms")
        if len(d) and d["catch_window_ms"].notna().any():
            ax.plot(d["dw_ms"], d["catch_window_ms"], "-o" if not cl else "--s", ms=4, lw=1.4, color=CAT[i % 2],
                    label=f"$\\phi_\\ell$={ph:.3f}, {'closed loop' if cl else 'open loop'}")
    lim = max(45, ok["dw_ms"].max() * 2.2 if len(ok) else 45)
    ax.plot([0, lim / 2], [0, lim], ":", color=INK2, lw=0.9, label=r"$2\delta_w$ (designed)")
    ax.set_xlabel(r"designed release tolerance $\pm\delta_w$ [ms]"); ax.set_ylabel("re-simulated catch window [ms]")
    ax.set_title("verification (0.5 ms re-integration)", fontsize=8.5); ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "robust_pareto.pdf")); fig.savefig(os.path.join(a.out, "robust_pareto.png"), dpi=200)
    plt.close(fig)
    # ---- table + numbers --------------------------------------------------------------------------------
    with open(os.path.join("paper", "tab_robust.tex"), "w") as f:
        f.write("\\begin{tabular}{rlrrrrr}\n\\toprule\n$\\phi_\\ell$ & 飛行制御 & $\\pm\\delta_w$ [ms] & $U^*$ & 必要容量 [BW] & 捕捉窓 [ms] & $W_{1.0}$ [ms] \\\\\n\\midrule\n")
        for r in ok.itertuples():
            f.write(f"{r.phi:.3f} & {'閉' if r.closed_loop else '開'} & {r.dw_ms:.0f} & {r.U_peak:.3f} & {r.U_peak * 1300 / (66 * 9.81):.2f} & "
                    f"{'--' if np.isnan(r.catch_window_ms) else f'{r.catch_window_ms:.1f}'} & {'--' if np.isnan(r._12) else f'{r._12:.1f}'} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    summ = {}
    for ph in (0.25, 0.6875):
        for cl in (False, True):
            d = ok[(ok["phi"] == ph) & (ok["closed_loop"] == cl)].sort_values("dw_ms")
            if len(d) >= 2:
                base = d[d["dw_ms"] == 0]["U_peak"].values
                base = base[0] if len(base) else d["U_peak"].iloc[0]
                slope = np.polyfit(d["dw_ms"], d["U_peak"] / base - 1, 1)[0] * 10  # per 10 ms
                summ[f"phi{ph}_{'cl' if cl else 'ol'}"] = dict(slope_pct_per_10ms=100 * slope, U=d["U_peak"].tolist(), dw=d["dw_ms"].tolist(),
                                                              win=d["catch_window_ms"].tolist())
    json.dump(summ, open(os.path.join(a.out, "robust.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
