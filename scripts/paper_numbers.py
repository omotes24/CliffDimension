"""Write LaTeX macros with the headline numbers of the Stage-1 experiments (paper/numbers.tex)
and the LaTeX tables used in the manuscript (paper/tab_*.tex)."""
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi import device


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--windows", default="results/windows/windows_ref.csv")
    ap.add_argument("--sens", default="results/figs/sensitivity.csv")
    ap.add_argument("--audit", default="results/figs/audit.json")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--build", default="results/build")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.grid, f"rows_{a.tag}_T*_m*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["phi_l"] = df["phi_l"].round(4)
    df = df.sort_values("ok", ascending=False).drop_duplicates(["T", "m", "phi_l"], keep="first")
    ok = df[df["ok"] == 1].copy()
    ok["outbound"] = ok["phi_l"] < 0.5
    macros = {}
    macros["ncases"] = str(len(df))
    macros["nok"] = str(len(ok))
    macros["capmin"] = fmt(ok["req_cap_BW"].min())
    macros["capmax"] = fmt(ok["req_cap_BW"].max())
    macros["capminN"] = fmt(ok["req_cap_N"].min(), 0)
    macros["capmaxN"] = fmt(ok["req_cap_N"].max(), 0)
    # best per (T, m)
    rows = []
    for (T, m), d in ok.groupby(["T", "m"]):
        b = d.loc[d["req_cap_BW"].idxmin()]
        o = d[d["outbound"]]
        r = d[~d["outbound"]]
        near_cap = d.loc[d["phi_l"].idxmin(), "req_cap_BW"] if len(d) else np.nan
        rows.append(dict(T=T, m=m, best_phi=b["phi_l"], best_cap=b["req_cap_BW"], best_capN=b["req_cap_N"],
                         best_dir="outbound" if b["phi_l"] < 0.5 else "return",
                         out_cap=o["req_cap_BW"].min() if len(o) else np.nan, out_phi=o.loc[o["req_cap_BW"].idxmin(), "phi_l"] if len(o) else np.nan,
                         ret_cap=r["req_cap_BW"].min() if len(r) else np.nan, ret_phi=r.loc[r["req_cap_BW"].idxmin(), "phi_l"] if len(r) else np.nan,
                         near_cap=near_cap, tau=b["d_f"], v0x=b["v0x"], v0y=b["v0y"], x_catch=b["x_catch"],
                         h_rel=b["h_release"], E=b["effort"], d_s=b["d_s"], vrel=b["vrel_norm"], worst_cap=d["req_cap_BW"].max()))
    s = pd.DataFrame(rows)
    macros["bestphimin"] = fmt(s["best_phi"].min(), 3)
    macros["bestphimax"] = fmt(s["best_phi"].max(), 3)
    macros["outphimin"] = fmt(s["out_phi"].min(), 2); macros["outphimax"] = fmt(s["out_phi"].max(), 2)
    macros["retphimin"] = fmt(s["ret_phi"].min(), 2); macros["retphimax"] = fmt(s["ret_phi"].max(), 2)
    macros["ptov"] = fmt(100 * ((s["worst_cap"] - s["best_cap"]) / s["best_cap"]).mean(), 0)
    macros["nret"] = str(int((s["best_dir"] == "return").sum())) if "best_dir" in s else "--"
    macros["ncond"] = str(len(s))
    macros["Tmin"] = fmt(s["T"].min(), 0); macros["Tmax"] = fmt(s["T"].max(), 0)
    macros["mmin"] = fmt(s["m"].min(), 0); macros["mmax"] = fmt(s["m"].max(), 0)
    macros["nT"] = str(s["T"].nunique()); macros["nm"] = str(s["m"].nunique())
    macros["bestcapmin"] = fmt(s["best_cap"].min()); macros["bestcapmax"] = fmt(s["best_cap"].max())
    macros["hOnediffmax"] = fmt(100 * (s["ret_cap"] - s["out_cap"]).abs().max() / s["out_cap"].mean(), 1)
    macros["hTwopenalty"] = fmt(100 * ((s["near_cap"] - s["best_cap"]) / s["best_cap"]).mean(), 1)
    macros["hTwopenaltymax"] = fmt(100 * ((s["near_cap"] - s["best_cap"]) / s["best_cap"]).max(), 1)
    macros["taumin"] = fmt(ok["d_f"].min()); macros["taumax"] = fmt(ok["d_f"].max())
    macros["dsmin"] = fmt(ok["d_s"].min(), 1); macros["dsmax"] = fmt(ok["d_s"].max(), 1)
    macros["vrelmin"] = fmt(ok["vrel_norm"].min(), 1); macros["vrelmax"] = fmt(ok["vrel_norm"].max(), 1)
    macros["capsixty"] = fmt(s[s["m"] == 60]["best_cap"].mean()); macros["capseventyfive"] = fmt(s[s["m"] == 75]["best_cap"].mean())
    # audit
    if os.path.exists(a.audit):
        au = json.load(open(a.audit))
        macros["resSmax"] = fmt(au["hs_residual_max"]["S"], 1)
        macros["verSpos"] = fmt(1e3 * au["reintegration_err_max"]["S_pos_rad"], 1)
        macros["verFhand"] = fmt(1e3 * au["reintegration_err_max"]["F_hand_m"], 2)
        macros["solvemed"] = fmt(au["solve_time_s"]["median"] / 60, 1)
    # windows
    if os.path.exists(a.windows):
        w = pd.read_csv(a.windows)
        macros["catchwinmax"] = fmt(1e3 * w["plan_catch_window_s"].max(), 0)
        macros["catchwinmed"] = fmt(1e3 * w["plan_catch_window_s"].median(), 0)
        macros["loadratio"] = fmt((w["plan_U0_max"] / w["U_star"]).median(), 2)
        macros["loadratiomax"] = fmt((w["plan_U0_max"] / w["U_star"]).max(), 2)
        macros["nwin"] = str(len(w))
        wt = w[["T", "m", "phi_l", "U_star", "plan_catch_window_s", "plan_U0_catch", "plan_U0_hold", "plan_U0_max", "plan_reason0", "hang_U0_max", "hang_reason0"]].copy()
        with open(os.path.join(a.out, "tab_windows.tex"), "w") as f:
            f.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n$T$ & $m$ & $\\phi_\\ell$ & $U^*$ & 捕捉窓 [ms] & $U_{\\rm catch}$ & $U_{\\rm hold}$ \\\\\n\\midrule\n")
            for r in wt.itertuples():
                f.write(f"{r.T:g} & {r.m:g} & {r.phi_l:.3f} & {r.U_star:.2f} & {1e3*r.plan_catch_window_s:.0f} & "
                        f"{r.plan_U0_catch:.2f} & {r.plan_U0_hold:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    # sensitivity table
    if os.path.exists(a.sens):
        se = pd.read_csv(a.sens)
        with open(os.path.join(a.out, "tab_sens.tex"), "w") as f:
            f.write("\\begin{tabular}{lrrr}\n\\toprule\n要因 & 最小容量 [BW] & 平均容量 [BW] & 最良位相 \\\\\n\\midrule\n")
            for r in se.itertuples():
                f.write(f"{r.factor} & {r.min_cap_BW:.2f} & {r.mean_cap_BW:.2f} & {r.best_phi:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    # body-size sweep table (results/build): stature / arm-length ratio at T = 20 s, m = 66 kg
    build_dir = a.build
    if os.path.isdir(build_dir):
        tags = sorted(set(re.match(r"rows_(.+?)_T", os.path.basename(f)).group(1)
                          for f in glob.glob(os.path.join(build_dir, "rows_*_T*.csv"))))
        brow = []
        ref_row = ok[(ok["T"] == 20.0) & (ok["m"] == 66.0)]
        if len(ref_row):
            bb = ref_row.loc[ref_row["req_cap_BW"].idxmin()]
            brow.append(("1.75", "1.00", bb["req_cap_BW"], ref_row["req_cap_BW"].mean(), bb["phi_l"], bb["d_f"], len(ref_row)))
        for tag in tags:
            fs = glob.glob(os.path.join(build_dir, f"rows_{tag}_T*.csv"))
            d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
            d["phi_l"] = d["phi_l"].round(4)
            d = d.sort_values("ok", ascending=False).drop_duplicates(["T", "m", "phi_l"])
            d = d[d["ok"] == 1]
            if len(d) == 0:
                continue
            bb = d.loc[d["req_cap_BW"].idxmin()]
            mt = re.match(r"stature(\d+)", tag)
            ma = re.match(r"arm(\d+)", tag)
            H = f"{int(mt.group(1))/100:.2f}" if mt else "1.75"
            A = f"{int(ma.group(1))/100:.2f}" if ma else "1.00"
            brow.append((H, A, bb["req_cap_BW"], d["req_cap_BW"].mean(), bb["phi_l"], bb["d_f"], len(d)))
        if brow:
            brow.sort(key=lambda r: (r[1], r[0]))
            with open(os.path.join(a.out, "tab_build.tex"), "w") as f:
                f.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n身長 [m] & 腕長比 & 最小容量 [BW] & 平均容量 [BW] & 最良 $\\phi_\\ell$ & $\\tau$ [s] & 収束数 \\\\\n\\midrule\n")
                for H, A, cmin, cmean, ph, tau, n in brow:
                    f.write(f"{H} & {A} & {cmin:.2f} & {cmean:.2f} & {ph:.3f} & {tau:.2f} & {n} \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
    # main table: best per (T, m)
    s_tab = s[s["T"].isin([16, 19, 20, 24]) & s["m"].isin([60, 75])] if len(s) > 12 else s
    with open(os.path.join(a.out, "tab_best.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n$T$ [s] & $m$ [kg] & 往路谷 & 復路谷 & $\\phi_\\ell=0$ & $\\tau$ [s] \\\\\n\\midrule\n")
        for r in s_tab.itertuples():
            f.write(f"{r.T:g} & {r.m:g} & {r.out_cap:.2f} ({r.out_phi:.2f}) & "
                    f"{r.ret_cap:.2f} ({r.ret_phi:.2f}) & {r.near_cap:.2f} & {r.tau:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for k in ["resSmax", "verSpos", "verFhand", "solvemed", "catchwinmax", "catchwinmed", "loadratio", "loadratiomax", "nwin"]:
        macros.setdefault(k, "--")
    with open(os.path.join(a.out, "numbers.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    s.to_csv(os.path.join(a.out, "best_per_condition.csv"), index=False)
    print(json.dumps(macros, indent=1, ensure_ascii=False))
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
