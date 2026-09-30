"""LaTeX macros / tables for the dual-field and learning sections (paper/numbers_dfl.tex, paper/tab_prices.tex,
paper/tab_window.tex).

usage: python scripts/paper_numbers_dfl.py --data results/dual_field --dfl results/dfl --rl results/rl_planar --grid results/grid
"""
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/dual_field")
    ap.add_argument("--dfl", default="results/dfl")
    ap.add_argument("--rl", default="results/rl_planar")
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    M = {}
    # ---- dual dataset ---------------------------------------------------------------------------------------
    rows_csv = os.path.join(a.data, "rows.csv")
    if os.path.exists(rows_csv):
        d = pd.read_csv(rows_csv)
        ok = d[d["ok"] == 1]
        M["dflNsolve"] = str(len(d)); M["dflNok"] = str(len(ok)); M["dflNref"] = str(d["ref"].nunique())
        n_dense = 0
        for f in glob.glob(os.path.join(a.data, "sol_*.pkl")):
            pass
        # dense rows: from the last fit log if present
        for lg in sorted(glob.glob("logs/dfl_it*.log")):
            m = re.search(r"dense rows: (\d+)", open(lg).read())
            if m:
                n_dense = int(m.group(1))
        M["dflNrows"] = str(n_dense + len(ok)) if n_dense else str(len(ok))
        el = -ok["sens_tau_cap"] / ok["J"]                    # elasticity d ln J / d ln tau_cap (negative -> report magnitude)
        M["elastTau"] = f"{np.median(ok['sens_tau_cap'] / ok['J']):.2f}"
        M["elastTauMean"] = f"{np.mean(ok['sens_tau_cap'] / ok['J']):.2f}"
        M["elastTauTen"] = f"{-10 * np.median(ok['sens_tau_cap'] / ok['J']):.0f}"
        cap = ok[["cap_cap_H", "cap_cap_catch", "cap_cap_S"]].mean()
        tot = cap.sum()
        M["capHoldShare"] = f"{100 * cap['cap_cap_H'] / tot:.0f}"; M["capCatchShare"] = f"{100 * cap['cap_cap_catch'] / tot:.0f}"
        M["capSwingShare"] = f"{100 * cap['cap_cap_S'] / tot:.0f}"
        with open(os.path.join(a.out, "tab_prices.tex"), "w") as f:
            f.write("\\begin{tabular}{lrrr}\n\\toprule\nパラメータ $p$ & $\\partial J^*/\\partial\\ln p$（中央値） & 平均 & 弾性 $\\partial\\ln J^*/\\partial\\ln p$（中央値） \\\\\n\\midrule\n")
            names = {"sens_tau_cap": "関節トルク容量 $\\tau^{\\rm cap}$", "sens_mu_out": "引っ掛かり係数 $\\mu_{\\rm out}$", "sens_mu_in": "$\\mu_{\\rm in}$",
                     "sens_qd_max": "関節角速度上限", "sens_v_rel_max": "捕捉相対速度上限", "sens_v_away_max": "壁から離れる速度上限"}
            for c, nm in names.items():
                if c in ok:
                    f.write(f"{nm} & {ok[c].median():.3f} & {ok[c].mean():.3f} & {np.median(ok[c] / ok['J']):.3f} \\\\\n")
            f.write("\\midrule\n容量乗数の配分 & 保持 %.0f\\%% & 捕捉 %.0f\\%% & 振り %.0f\\%% \\\\\n" % (100 * cap['cap_cap_H'] / tot, 100 * cap['cap_cap_catch'] / tot, 100 * cap['cap_cap_S'] / tot))
            f.write("\\bottomrule\n\\end{tabular}\n")
    # ---- fits ---------------------------------------------------------------------------------------------
    its = sorted(glob.glob(os.path.join(a.dfl, "it*/fit.json")), key=lambda s: int(re.search(r"it(\d+)", s).group(1)))
    if its:
        f = json.load(open(its[-1]))
        fits = pd.DataFrame(f["fits"])
        def get(model, split, key):
            r = fits[(fits["model"] == model) & (fits["split"] == split)]
            return float(r[key].iloc[0]) if len(r) else float("nan")
        M["fitVhold"] = f"{get('DFL(Sobolev)', 'holdout', 'r2_V'):.2f}"; M["fitTauhold"] = f"{get('DFL(Sobolev)', 'holdout', 'r2_tau'):.2f}"
        M["fitPhold"] = f"{get('DFL(Sobolev)', 'holdout', 'r2_p'):.2f}"; M["fitPholdNoSob"] = f"{get('DFL(value-only)', 'holdout', 'r2_p'):.2f}"
        M["fitUhold"] = f"{get('DFL(Sobolev)', 'holdout', 'r2_u'):.2f}"
    # ---- RL -----------------------------------------------------------------------------------------------
    n_ep = 0; n_s = 0
    for f in glob.glob(os.path.join(a.rl, "ppo_*/episodes.csv")):
        e = pd.read_csv(f); n_ep += len(e); n_s += int(e["success"].sum())
    if n_ep:
        M["rlEpisodesAll"] = str(n_ep); M["rlSuccessAll"] = str(n_s)
    # ---- v3 window from the grid ------------------------------------------------------------------------------
    files = sorted(glob.glob(os.path.join(a.grid, "rows_ref_T*_m*.csv")))
    if files:
        g = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
        g["phi_l"] = g["phi_l"].round(4)
        g = g.sort_values("ok", ascending=False).drop_duplicates(["T", "m", "phi_l"])
        okg = g[(g["ok"] == 1) & (g["U_peak"] < 2.0)]
        if len(okg):
            win = okg[(okg["phi_l"] >= 0.2) & (okg["phi_l"] <= 0.35)]
            M["vthreecapmin"] = f"{win['U_peak'].min():.2f}"; M["vthreecapmax"] = f"{win['U_peak'].max():.2f}"
            best = okg.loc[okg.groupby(["T", "m"])["U_peak"].idxmin()]
            with open(os.path.join(a.out, "tab_window.tex"), "w") as f:
                f.write("\\begin{tabular}{rrrrr}\n\\toprule\n$T$ [s] & $m$ [kg] & 最良 $\\phi_\\ell$ & $\\Upk$ & BW \\\\\n\\midrule\n")
                for r in best.sort_values(["T", "m"]).itertuples():
                    f.write(f"{r.T:g} & {r.m:g} & {r.phi_l:.3f} & {r.U_peak:.2f} & {r.req_cap_BW:.2f} \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(a.out, "numbers_dfl.tex"), "w") as f:
        for k, v in M.items():
            f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n" if False else f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps(M, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
