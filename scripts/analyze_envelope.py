"""Lower-envelope report: how much did multi-start + continuation lower the phase curves, and how rough are they?

Reads results/grid/envelope_ref.json (written by envelope_grid.py) and the current grid rows.
Outputs: results/figs/envelope.pdf/png (before / after curves), paper/tab_envelope.tex, numbers in envelope.json.

usage: python scripts/analyze_envelope.py --grid results/grid --out results/figs
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    log = json.load(open(a.log or os.path.join(a.grid, "envelope_ref.json")))
    conds = [tuple(c) for c in log["conditions"]]
    imps = pd.DataFrame(log["improvements"]) if log["improvements"] else pd.DataFrame(columns=["T", "m", "phi", "old", "new", "kind"])
    # before/after curves: "before" = first 'old' value seen per case (or current if never improved)
    rows = []
    for (T, m) in conds:
        fs = glob.glob(os.path.join(a.grid, f"rows_ref_T{T:g}_m{m:g}*.csv"))
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d["phi_l"] = d["phi_l"].round(4)
        d = d.sort_values("ok", ascending=False).drop_duplicates(["phi_l"]).sort_values("phi_l")
        for r in d.itertuples():
            first = imps[(imps["T"] == T) & (imps["m"] == m) & (np.isclose(imps["phi"], r.phi_l, atol=2e-3))]
            before = float(first["old"].iloc[0]) if len(first) and first["old"].iloc[0] is not None and not np.isnan(first["old"].iloc[0]) else (r.U_peak if r.ok else np.nan)
            rows.append(dict(T=T, m=m, phi=r.phi_l, before=before, after=r.U_peak if r.ok else np.nan, ok=r.ok))
    df = pd.DataFrame(rows)
    df["rel"] = 1 - df["after"] / df["before"]
    summ = dict(n_cases=int(len(df)), n_improved=int((df["rel"] > 1e-3).sum()), mean_rel=float(df["rel"].mean()),
                median_rel=float(df["rel"].median()), max_rel=float(df["rel"].max()),
                p90_rel=float(df["rel"].quantile(0.9)), passes=log.get("passes"), summary_log=log.get("summary"),
                roughness_before=log.get("roughness_before"), roughness_after=log.get("roughness_after"))
    # improvements by kind
    if len(imps):
        imps["rel"] = 1 - imps["new"] / imps["old"].astype(float)
        summ["by_kind"] = imps.groupby(imps["kind"].str.split(" ").str[0])["rel"].agg(["count", "mean", "max"]).to_dict()
    # valley comparison after the envelope: outbound vs return minima
    val = []
    for (T, m), d in df.groupby(["T", "m"]):
        o = d[d["phi"] < 0.5]["after"].min(); r_ = d[d["phi"] >= 0.5]["after"].min()
        val.append(dict(T=T, m=m, out_min=o, ret_min=r_, diff_pct=100 * (r_ - o) / o))
    val = pd.DataFrame(val)
    summ["valley_diff_pct_max"] = float(val["diff_pct"].abs().max()); summ["valley_diff_pct_mean"] = float(val["diff_pct"].abs().mean())
    summ["n_return_best"] = int((val["ret_min"] < val["out_min"]).sum()); summ["n_cond"] = int(len(val))
    json.dump(summ, open(os.path.join(a.out, "envelope.json"), "w"), indent=1, default=float)
    df.to_csv(os.path.join(a.out, "envelope_cases.csv"), index=False)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("roughness_before", "roughness_after")}, indent=1, default=float))
    # figure: before / after for the 12 conditions
    Ts = sorted(set(c[0] for c in conds)); ms = sorted(set(c[1] for c in conds))
    fig, axes = plt.subplots(len(ms), len(Ts), figsize=(2.6 * len(Ts), 1.9 * len(ms)), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for i, m in enumerate(ms):
        for j, T in enumerate(Ts):
            ax = axes[i, j]
            d = df[(df["T"] == T) & (df["m"] == m)].sort_values("phi")
            ax.plot(d["phi"], d["before"], "-", color="#c9c9c4", lw=1.3, label="single start")
            ax.plot(d["phi"], d["after"], "-o", color=CAT[0], ms=2.5, lw=1.3, label="envelope")
            ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
            ax.set_title(f"T = {T:g} s, m = {m:g} kg", fontsize=8)
            if i == len(ms) - 1:
                ax.set_xlabel(r"$\phi_\ell$")
            if j == 0:
                ax.set_ylabel(r"$U^*$ [$F_{cap}$]")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "envelope.pdf")); fig.savefig(os.path.join(a.out, "envelope.png"), dpi=200)
    plt.close(fig)
    with open(os.path.join("paper", "tab_envelope.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n$T$ [s] & $m$ [kg] & 往路谷 & 復路谷 & 差 [\\%] & 粗さ 前$\\to$後 [\\%] \\\\\n\\midrule\n")
        for r in val.itertuples():
            rb = log.get("roughness_before", {}).get(f"{r.T:g},{r.m:g}", float("nan")); ra = log.get("roughness_after", {}).get(f"{r.T:g},{r.m:g}", float("nan"))
            f.write(f"{r.T:g} & {r.m:g} & {r.out_min:.3f} & {r.ret_min:.3f} & {r.diff_pct:+.1f} & {100*rb:.2f}$\\to${100*ra:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
