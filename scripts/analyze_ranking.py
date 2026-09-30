"""Is the RANKING of release phases preserved when the model changes?

Compares the reference phase curve U*(phi) (results/grid, tag ref) with the curves of model variants:
  * one-factor variants at 16 phases (results/sens16: mu_out, joint capacity, eps),
  * compliant catch inside the optimisation (results/compliant, tag comp),
  * body-size variants (results/build: stature, arm length; reference = same T, m in results/grid),
  * mesh refinement is reported from numbers when available.
Metrics per variant: Spearman rho, Kendall tau, argmin phase (and whether it lies in the same valley),
rank of the reference's best phase under the variant, and the relative change of the min.

usage: python scripts/analyze_ranking.py --out results/figs
"""
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#52514e"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

LABEL = {"mu060": r"$\mu_{out}=0.6$", "mu150": r"$\mu_{out}=1.5$", "cap070": "joint cap. ×0.7", "cap130": "joint cap. ×1.3",
         "eps010": r"$\epsilon=0.10$ s", "eps040": r"$\epsilon=0.40$ s", "comp": "compliant catch",
         "stature160": "H = 1.60 m", "stature165": "H = 1.65 m", "stature170": "H = 1.70 m", "stature180": "H = 1.80 m",
         "stature185": "H = 1.85 m", "arm095": "arm ×0.95", "arm105": "arm ×1.05", "dc030": r"$\Delta_c=30$ ms",
         "dc100": r"$\Delta_c=100$ ms", "vrel3": r"$|v_{rel}|\le3$", "vrel5": r"$|v_{rel}|\le5$"}


def load_curve(files):
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["phi_l"] = d["phi_l"].round(4)
    d = d.sort_values("ok", ascending=False).drop_duplicates(["T", "m", "phi_l"])
    d = d[d["ok"] == 1].sort_values("phi_l")
    return d.set_index("phi_l")["U_peak"]


def compare(ref, var):
    common = ref.index.intersection(var.index)
    if len(common) < 6:
        return None
    r, v = ref.loc[common].values, var.loc[common].values
    rho = stats.spearmanr(r, v).correlation
    tau = stats.kendalltau(r, v).correlation
    phi_ref = common[np.argmin(r)]; phi_var = common[np.argmin(v)]
    same_valley = (phi_ref < 0.5) == (phi_var < 0.5)
    rank_of_ref_best = int(stats.rankdata(v)[np.argmin(r)])
    # within the reference's valley: does the variant keep the same best phase?
    return dict(n=len(common), spearman=rho, kendall=tau, phi_best_ref=phi_ref, phi_best_var=phi_var, same_valley=same_valley,
                rank_ref_best_in_var=rank_of_ref_best, min_ref=r.min(), min_var=v.min(), rel_min=(v.min() / r.min() - 1),
                ptov_var=(v.max() / v.min() - 1), ptov_ref=(r.max() / r.min() - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    curves = {}
    # one-factor variants at (20, 60)
    ref60 = load_curve(glob.glob(os.path.join(a.grid, "rows_ref_T20_m60*.csv")))
    for f in sorted(glob.glob("results/sens16/rows_*_T20_m60*.csv")):
        tag = re.match(r"rows_(.+?)_T20", os.path.basename(f)).group(1)
        var = load_curve([f])
        if var is None or ref60 is None:
            continue
        c = compare(ref60, var)
        if c:
            rows.append(dict(group="one-factor (T=20 s, m=60 kg)", tag=tag, label=LABEL.get(tag, tag), **c))
            curves[("m60", tag)] = (ref60, var)
    # compliant catch at (20, 60) and (20, 66)
    for m in (60, 66):
        ref = load_curve(glob.glob(os.path.join(a.grid, f"rows_ref_T20_m{m}*.csv")))
        fs = glob.glob(f"results/compliant/rows_comp_T20_m{m}*.csv")
        var = load_curve(fs)
        if ref is not None and var is not None:
            c = compare(ref, var)
            if c:
                rows.append(dict(group=f"catch model (T=20 s, m={m} kg)", tag=f"comp_m{m}", label=f"compliant catch, m={m}", **c))
                curves[(f"m{m}", "comp")] = (ref, var)
    # body size at (20, 66)
    ref66 = load_curve(glob.glob(os.path.join(a.grid, "rows_ref_T20_m66*.csv")))
    for f in sorted(glob.glob("results/build/rows_*_T20_m66*.csv")):
        tag = re.match(r"rows_(.+?)_T20", os.path.basename(f)).group(1)
        var = load_curve([f])
        if var is None or ref66 is None:
            continue
        c = compare(ref66, var)
        if c:
            rows.append(dict(group="body size (T=20 s, m=66 kg)", tag=tag, label=LABEL.get(tag, tag), **c))
            curves[("m66", tag)] = (ref66, var)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "ranking.csv"), index=False)
    if len(df):
        print(df[["group", "label", "n", "spearman", "kendall", "phi_best_ref", "phi_best_var", "same_valley",
                  "rank_ref_best_in_var", "rel_min", "ptov_var"]].round(3).to_string(index=False))
    # ---- figure: normalised curves ---------------------------------------------------------------------------
    groups = [("m60", [k[1] for k in curves if k[0] == "m60" and k[1] != "comp"], "one-factor variants (T = 20 s, m = 60 kg)"),
              ("m66", [k[1] for k in curves if k[0] == "m66"], "body size (T = 20 s, m = 66 kg)"),
              ("comp", None, "catch model")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=False)
    for ax, (g, tags, title) in zip(axes, groups):
        if g == "comp":
            items = [(k, curves[k]) for k in curves if k[1] == "comp"]
            for i, (k, (ref, var)) in enumerate(items):
                ax.plot(ref.index, ref.values / ref.min(), "-", color=CAT[i], lw=1.2, alpha=0.5, label=f"rigid impact, {k[0]}")
                ax.plot(var.index, var.values / var.min(), "--o", color=CAT[i], ms=2.5, lw=1.2, label=f"compliant, {k[0]}")
        else:
            ref = None
            for i, tag in enumerate(tags):
                ref, var = curves[(g, tag)]
                ax.plot(var.index, var.values / var.min(), "-o", ms=2.2, lw=1.1, color=CAT[i % len(CAT)], label=LABEL.get(tag, tag))
            if ref is not None:
                ax.plot(ref.index, ref.values / ref.min(), "k-", lw=2.0, alpha=0.7, label="reference")
        ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
        ax.set_title(title, fontsize=9); ax.set_xlabel(r"release phase $\phi_\ell$")
        ax.legend(fontsize=6, ncol=2)
    axes[0].set_ylabel(r"$U^*(\phi_\ell)\,/\,\min_\phi U^*$")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "ranking.pdf")); fig.savefig(os.path.join(a.out, "ranking.png"), dpi=200)
    plt.close(fig)
    # LaTeX table
    if len(df):
        with open(os.path.join("paper", "tab_ranking.tex"), "w") as f:
            f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n変種 & $\\rho_S$ & $\\tau_K$ & 最良 $\\phi_\\ell$（基準→変種） & 同じ谷 & 基準最良の順位 & $\\Delta\\min$ [\\%] \\\\\n\\midrule\n")
            for r in df.itertuples():
                f.write(f"{r.label} & {r.spearman:.2f} & {r.kendall:.2f} & {r.phi_best_ref:.3f}$\\to${r.phi_best_var:.3f} & "
                        f"{'○' if r.same_valley else '×'} & {r.rank_ref_best_in_var}/{r.n} & {100 * r.rel_min:+.1f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    print("ranking written")


if __name__ == "__main__":
    main()
