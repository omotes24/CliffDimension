"""Figures / tables for the learning experiments (RL baselines, dual-field fits, closed-loop comparison).

usage: python scripts/analyze_learning.py --rl results/rl_planar --dfl results/dfl --out results/figs
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#52514e", "#d63a8a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
LABEL = {"none": "sparse", "energy": "+energy", "margin+energy": "+margin+energy", "dual+energy": "+dual+energy",
         "dual+margin+energy": "+dual+margin+energy", "dual": "+dual"}


def rl_curves(rl_dir, out, window=2000):
    runs = sorted(glob.glob(os.path.join(rl_dir, "ppo_*_s*")))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    rows = []
    for i, d in enumerate(runs):
        f = os.path.join(d, "episodes.csv")
        if not os.path.exists(f):
            continue
        e = pd.read_csv(f)
        tag = re.match(r"ppo_(.+)_s\d+", os.path.basename(d)).group(1)
        e["steps"] = (e["t"] / 0.02).cumsum()                       # environment steps (control_dt = 20 ms)
        e["succ"] = e["success"].rolling(window, min_periods=100).mean()
        e["len"] = e["t"].rolling(window, min_periods=100).mean()
        e["rel"] = e["release_time"].notna().astype(float).rolling(window, min_periods=100).mean()
        axes[0].plot(e["steps"] / 1e6, e["succ"], color=CAT[i % len(CAT)], lw=1.2, label=LABEL.get(tag, tag))
        axes[1].plot(e["steps"] / 1e6, e["len"], color=CAT[i % len(CAT)], lw=1.2)
        axes[2].plot(e["steps"] / 1e6, e["rel"], color=CAT[i % len(CAT)], lw=1.2)
        last = e.tail(2000)
        rows.append(dict(run=tag, episodes=len(e), steps=int(e["steps"].iloc[-1]), success_all=e["success"].mean(),
                         success_last2000=last["success"].mean(), n_success=int(e["success"].sum()),
                         U_peak_success=float(e.loc[e["success"] == 1, "U_peak"].median()) if e["success"].sum() else np.nan,
                         released_last2000=last["release_time"].notna().mean(),
                         reasons_last2000=json.dumps(last["reason"].value_counts().head(4).to_dict(), ensure_ascii=False)))
    axes[0].set_ylabel("success rate (rolling)"); axes[1].set_ylabel("episode length [s]"); axes[2].set_ylabel("released fraction")
    for ax in axes:
        ax.set_xlabel("environment steps [M]")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "rl_curves.pdf")); fig.savefig(os.path.join(out, "rl_curves.png"), dpi=200); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "rl_summary.csv"), index=False)
    if len(df):
        print(df.to_string(index=False))
    return df


def dfl_results(dfl_dir, out):
    its = sorted(glob.glob(os.path.join(dfl_dir, "it*")), key=lambda s: int(re.search(r"it(\d+)", s).group(1)))
    fits, summ = [], []
    for d in its:
        it = int(re.search(r"it(\d+)", d).group(1))
        fj = os.path.join(d, "fit.json"); sj = os.path.join(d, "summary.csv"); ej = os.path.join(d, "episodes.csv")
        if os.path.exists(fj):
            f = json.load(open(fj))
            for r in f["fits"]:
                r = dict(r); r["it"] = it; fits.append(r)
        if os.path.exists(ej):
            e = pd.read_csv(ej); e["it"] = it
            g = e.groupby(["group", "ctrl"]).agg(n=("success", "size"), success=("success", "mean"), U_peak=("U_peak", "median"),
                                                U_star=("U_star", "median"),
                                                rel_err=("release_err", lambda s: np.nanmedian(np.abs(pd.to_numeric(s, errors="coerce"))))).reset_index()
            g["it"] = it
            summ.append(g)
    fits = pd.DataFrame(fits); summ = pd.concat(summ, ignore_index=True) if summ else pd.DataFrame()
    fits.to_csv(os.path.join(out, "dfl_fits.csv"), index=False); summ.to_csv(os.path.join(out, "dfl_closedloop.csv"), index=False)
    if len(fits):
        print(fits[["it", "model", "split", "n", "r2_V", "r2_p", "r2_tau", "r2_U", "r2_u"]].round(3).to_string(index=False))
    if len(summ):
        print(summ.round(3).to_string(index=False))
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
        for j, grp in enumerate(["train", "holdout"]):
            ax = axes[j]
            for i, ctrl in enumerate(sorted(summ["ctrl"].unique())):
                s = summ[(summ["group"] == grp) & (summ["ctrl"] == ctrl)].sort_values("it")
                ax.plot(s["it"], s["success"], "-o", ms=3, color=CAT[i], label=ctrl)
            ax.set_title(f"{grp} bodies", fontsize=9); ax.set_xlabel("DAgger iteration"); ax.set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("closed-loop success rate"); axes[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "dfl_closedloop.pdf")); fig.savefig(os.path.join(out, "dfl_closedloop.png"), dpi=200); plt.close(fig)
    # LaTeX tables
    with open(os.path.join("paper", "tab_dfl_fit.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n反復 & モデル & 分割 & $R^2(V)$ & $R^2(\\bm p)$ & $R^2(\\tau)$ & $R^2(U^*)$ \\\\\n\\midrule\n")
        for r in fits.itertuples():
            f.write(f"{r.it} & {r.model} & {r.split} & {r.r2_V:.3f} & {r.r2_p:.3f} & {r.r2_tau:.3f} & {r.r2_U:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join("paper", "tab_dfl_closedloop.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n反復 & 制御器 & 身体 & 成功率 & $\\Upk$（中央値） & $|\\Delta t_\\ell|$ [s] \\\\\n\\midrule\n")
        for r in summ.itertuples():
            f.write(f"{r.it} & {r.ctrl} & {r.group} & {100 * r.success:.0f}\\% ({r.n}) & {r.U_peak:.2f} / $U^*$={r.U_star:.2f} & {r.rel_err:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return fits, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default="results/rl_planar")
    ap.add_argument("--dfl", default="results/dfl")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("paper", exist_ok=True)
    if os.path.isdir(a.rl):
        rl_curves(a.rl, a.out)
    if os.path.isdir(a.dfl):
        dfl_results(a.dfl, a.out)


if __name__ == "__main__":
    main()
