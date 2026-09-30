"""Figures / tables for the learning experiments (RL baselines, dual-field fits, closed-loop comparison).

usage: python scripts/analyze_learning.py --rl results/rl_planar --dfl results/dfl --out results/figs
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#52514e", "#d63a8a", "#17becf", "#8c564b", "#bcbd22"]
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
        dm = pd.to_numeric(e["d_min"], errors="coerce").where(e["release_time"].notna())
        e["dmin"] = dm.rolling(window, min_periods=50).median()
        ls = "--" if tag.endswith("_rsi") else "-"
        lab = LABEL.get(tag.replace("_rsi", ""), tag) + (" (RSI)" if tag.endswith("_rsi") else "")
        axes[0].plot(e["steps"] / 1e6, e["dmin"], ls, color=CAT[i % len(CAT)], lw=1.2, label=lab)
        axes[1].plot(e["steps"] / 1e6, e["len"], ls, color=CAT[i % len(CAT)], lw=1.2)
        axes[2].plot(e["steps"] / 1e6, e["rel"], ls, color=CAT[i % len(CAT)], lw=1.2)
        last = e.tail(2000)
        rows.append(dict(run=tag, episodes=len(e), steps=int(e["steps"].iloc[-1]), success_all=e["success"].mean(),
                         success_last2000=last["success"].mean(), n_success=int(e["success"].sum()),
                         U_peak_success=float(e.loc[e["success"] == 1, "U_peak"].median()) if e["success"].sum() else np.nan,
                         released_last2000=last["release_time"].notna().mean(),
                         reasons_last2000=json.dumps(last["reason"].value_counts().head(4).to_dict(), ensure_ascii=False)))
    axes[0].set_ylabel("closest approach to B [m]\n(released episodes, rolling median)"); axes[1].set_ylabel("episode length [s]"); axes[2].set_ylabel("released fraction")
    axes[0].set_ylim(0, 2.6)
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
    """Fit tables from results/dfl/it*/fit.json; closed-loop table from every results/dfl/*/episodes.csv."""
    its = sorted(glob.glob(os.path.join(dfl_dir, "it*")), key=lambda s: int(re.search(r"it(\d+)", s).group(1)))
    fits = []
    for d in its:
        it = int(re.search(r"it(\d+)", d).group(1))
        fj = os.path.join(d, "fit.json")
        if os.path.exists(fj):
            f = json.load(open(fj))
            for r in f["fits"]:
                r = dict(r); r["it"] = it; fits.append(r)
    fits = pd.DataFrame(fits)
    eps = []
    RUN_SUFFIX = {"dfl_h40": "-H40", "dfl_h25": "-H25", "dfl_tau": "-tau", "dfl_tau_h40": "-tau-H40", "dfl_tau2": "-tau2", "dfl_tau3": "-tau3"}
    for f in sorted(glob.glob(os.path.join(dfl_dir, "*", "episodes.csv"))):
        e = pd.read_csv(f); run = os.path.basename(os.path.dirname(f)); e["run"] = run
        if run in RUN_SUFFIX:
            e["ctrl"] = e["ctrl"].astype(str) + RUN_SUFFIX[run]
        eps.append(e)
    eps = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    fits.to_csv(os.path.join(out, "dfl_fits.csv"), index=False)
    if len(fits):
        cols = [c for c in ["it", "model", "split", "n", "r2_V", "r2_p", "rho_p", "sign_p", "r2_tau", "r2_U", "r2_u"] if c in fits.columns]
        print(fits[cols].round(3).to_string(index=False))
    summ = pd.DataFrame()
    if len(eps):
        if "released" not in eps.columns:
            eps["released"] = np.nan
        eps["released"] = eps["released"].where(eps["released"].notna(),
            (eps["release_err"].notna() | eps["reason"].isin(["held B", "missed B", "hit B's face", "lost hook", "cone exceeded on B"])).astype(float))
        if "caught" not in eps.columns:
            eps["caught"] = eps["reason"].isin(["held B", "lost hook", "grip capacity exceeded"]).astype(float) * eps["released"]
        if "d_min" not in eps.columns:
            eps["d_min"] = np.nan
        g = eps.groupby(["ctrl"])
        summ = g.agg(n=("success", "size"), success=("success", "mean"), released=("released", "mean"), caught=("caught", "mean"),
                     d_min_med=("d_min", "median"), t_med=("t", "median"), U_med=("U_peak", "median"), U_star=("U_star", "median")).reset_index()
        summ["main_failure"] = [eps[eps["ctrl"] == c]["reason"].value_counts().index[0] for c in summ["ctrl"]]
        summ["n_slip"] = [int((eps[eps["ctrl"] == c]["reason"] == "slipped off A").sum()) for c in summ["ctrl"]]
        summ["n_wall"] = [int((eps[eps["ctrl"] == c]["reason"] == "hit wall").sum()) for c in summ["ctrl"]]
        summ["n_cap"] = [int((eps[eps["ctrl"] == c]["reason"] == "grip capacity exceeded").sum()) for c in summ["ctrl"]]
        print(summ.round(3).to_string(index=False))
        eps.to_csv(os.path.join(out, "dfl_episodes_all.csv"), index=False)
        summ.to_csv(os.path.join(out, "dfl_closedloop.csv"), index=False)
    # LaTeX tables
    if len(fits):
        with open(os.path.join("paper", "tab_dfl_fit.tex"), "w") as f:
            f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\nモデル & 分割 & $n$ & $R^2(V)$ & $R^2(\\tau)$ & $\\rho_S(\\bm p)$ & 符号一致$(\\bm p)$ & $R^2(\\bm u)$ \\\\\n\\midrule\n")
            last_it = fits["it"].max()
            for r in fits[fits["it"] == last_it].itertuples():
                rho = getattr(r, "rho_p", float("nan")); sg = getattr(r, "sign_p", float("nan"))
                f.write(f"{r.model} & {r.split} & {r.n} & {r.r2_V:.2f} & {r.r2_tau:.2f} & {rho:.2f} & {sg:.2f} & {r.r2_u:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    NAMES = {"BC": "行動模倣（BC）", "DFL": "双対場 MPC（$V$ 終端，0.3\\,s）", "DFL-H40": "双対場 MPC（$V$ 終端，0.8\\,s）",
             "DFL-tau": "双対場 MPC（$V$＋$\\tau$ 進行，0.3\\,s）", "DFL-tau-H40": "双対場 MPC（$V$＋$\\tau$ 進行，0.8\\,s）",
             "DFL-tau2": "双対場 MPC（$V$＋$\\tau$ 進行，初期制約緩和）", "DFL-tau3": "双対場 MPC（$V$＋$\\tau$ 進行，計画の PD 追従）",
             "DFL-noSob": "双対場 MPC（値のみ）", "ORACLE": "オラクル MPC"}
    REASON = {"slipped off A": "A で滑り", "hit wall": "壁に接触", "grip capacity exceeded": "容量超過", "missed B": "B を逃す",
              "hit B's face": "B 前面に衝突", "no release": "離手せず", "lost hook": "フック喪失", "held B": "成功"}
    with open(os.path.join("paper", "tab_dfl_closedloop.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrl}\n\\toprule\n制御器 & $n$ & 離手 & 捕捉 & 保持（成功） & 最接近中央値 [cm] & 生存時間中央値 [s] & $\\Upk$ 中央値 & 主な失敗 \\\\\n\\midrule\n")
        for r in summ.itertuples():
            dm = "---" if not np.isfinite(r.d_min_med) else f"{100 * r.d_min_med:.1f}"
            f.write(f"{NAMES.get(r.ctrl, r.ctrl)} & {r.n} & {100 * r.released:.0f}\\% & {100 * r.caught:.0f}\\% & {100 * r.success:.0f}\\% & {dm} & {r.t_med:.1f} & "
                    f"{r.U_med:.2f}（$U^*$={r.U_star:.2f}） & {REASON.get(r.main_failure, r.main_failure)} \\\\\n")
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
