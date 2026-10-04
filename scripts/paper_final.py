"""Figures, tables and LaTeX macros of the final manuscript from the experiment suite (results/suite) and the polished
864-case grid (results/grid, results/grid_prepolish).

  python scripts/paper_final.py --parts all        -> results/figs/*.pdf, paper/tab_x*.tex, paper/numbers_final.tex

Every number quoted in the text is a macro written here, so the manuscript follows the data.
"""
import argparse, glob, json, os, pickle, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import figstyle as fs
from figstyle import plt, C, CAT, INK, INK2, GRID, MUTED, METHOD_COL, METHOD_JP

S = os.path.join(ROOT, "results", "suite")              # working copy synced from hades (untracked)
if not os.path.isdir(S):
    S = os.path.join(ROOT, "results", "suite_final")    # tracked snapshot of the tables (CSV / JSON)
FIG = os.path.join(ROOT, "results", "figs")
PAPER = os.path.join(ROOT, "paper")
MAC = {}
LEVELS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)


def mac(name, val, nd=2):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MAC[name] = (f"{val:.{nd}f}" if isinstance(val, (float, np.floating)) else str(val))


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def tab(name, header, rows, align=None):
    align = align or ("l" + "r" * (len(header) - 1))
    with open(os.path.join(PAPER, name + ".tex"), "w") as f:
        f.write("\\begin{tabular}{" + align + "}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(("\\midrule\n" if r == "---" else " & ".join(str(x) for x in r) + " \\\\\n"))
        f.write("\\bottomrule\n\\end{tabular}\n")


def rd(*p):
    return pd.read_csv(os.path.join(S, *p))


# ===================================================================================================== experiment 1
def exp1():
    U = rd("exp1", "U_peak_by_objective.csv").set_index("cond"); E = rd("exp1", "effort_by_objective.csv").set_index("cond")
    ok = U[(U["lexico1"] < 2.0)].index                                  # conditions inside the feasible region (U <= 2)
    names = {"lexico1": "容量最優先（$\\min U$）", "lexico2": "辞書式（$U \\leq 1.01\\,\\hat{U}$ の下で努力最小）", "w100": "$w_U=100$", "current": "$w_U=10$（本稿）", "w1": "$w_U=1$"}
    order = ["lexico1", "lexico2", "w100", "current", "w1"]
    dU = {k: 100 * (U.loc[ok, k] / U.loc[ok, "lexico1"] - 1) for k in order}
    dE = {k: 100 * (E.loc[ok, k] / E.loc[ok, "current"] - 1) for k in order}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for i, k in enumerate(order):
        col = CAT[i]
        ax[0].scatter(dU[k], [i] * len(ok), s=22, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax[0].plot([dU[k].min(), dU[k].max()], [i, i], color=col, lw=1.2, zorder=2)
        ax[1].scatter(dE[k], [i] * len(ok), s=22, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax[1].plot([dE[k].min(), dE[k].max()], [i, i], color=col, lw=1.2, zorder=2)
    for a, xl in zip(ax, ("必要把持容量 $U_{\\mathrm{peak}}$ の増分 [%]（容量最優先解に対して）", "努力 $E_{\\mathrm{eff}}$ の増分 [%]（$w_U=10$ に対して）")):
        a.set_yticks(range(len(order))); a.set_yticklabels([names[k] for k in order] if a is ax[0] else []); a.invert_yaxis(); a.set_xlabel(xl)
        a.axvline(0, color=MUTED, lw=0.8)
    fig.subplots_adjust(wspace=0.06)
    fs.save(fig, FIG, "x1_objective")
    mac("objNcond", len(ok)); mac("objNall", len(U))
    mac("objCurMax", dU["current"].max(), 1); mac("objCurMed", dU["current"].median(), 1)
    mac("objWhundredMax", dU["w100"].max(), 1); mac("objWoneMax", dU["w1"].max(), 1); mac("objWoneMed", dU["w1"].median(), 1)
    mac("objLexEffMin", dE["lexico1"].min(), 0); mac("objLexEffMax", dE["lexico1"].max(), 0)
    mac("objLexTwoEffMed", dE["lexico2"].median(), 0)
    rows = []
    for c in U.index:
        T, m, ph = re.match(r"T(\d+)_m(\d+)_phi([0-9.]+)", c).groups()
        f = lambda v: "---" if not np.isfinite(v) else f"{v:.3f}"
        rows.append([m, f"{float(ph):.3f}"] + [f(U.loc[c, k]) for k in order] + [f"{E.loc[c, k]:.2f}" if np.isfinite(E.loc[c, k]) else "---" for k in ("lexico1", "current", "w1")])
    tab("tab_x1", ["$m$ [kg]", "$\\phi_\\ell$", "容量最優先", "辞書式", "$w_U{=}100$", "$w_U{=}10$", "$w_U{=}1$", "$E$：容量最優先", "$E$：$w_U{=}10$", "$E$：$w_U{=}1$"], rows, "rr|rrrrr|rrr")


# ===================================================================================================== experiment 9
def exp9():
    d = rd("exp9", "table.csv")
    d = d[d["ok"] == True].copy()
    base = d[d["variant"] == "base"].set_index("m")["U_peak"]; cbase = d[d["variant"] == "compliant base"].set_index("m")["U_peak"]
    d["rel"] = np.where(d["variant"].str.contains("catch=|compliant"), d["m"].map(cbase), d["m"].map(base))
    d.loc[d["variant"].str.startswith("delta_catch"), "rel"] = d["m"].map(base)
    d["dU"] = 100 * (d["U_peak"] / d["rel"] - 1)
    ms = [60.0, 66.0, 75.0]; mcol = {60.0: C["blue"], 66.0: C["orange"], 75.0: C["aqua"]}

    def panel(ax, variants, labels, title, xlab="$U_{\\mathrm{peak}}$ の変化 [%]"):
        for i, v in enumerate(variants):
            for j, m in enumerate(ms):
                r = d[(d["variant"] == v) & (d["m"] == m)]
                if len(r):
                    ax.barh(i + (j - 1) * 0.25, r["dU"].iloc[0], height=0.22, color=mcol[m], edgecolor="white", linewidth=0.8)
                else:
                    ax.text(0.0, i + (j - 1) * 0.25, " 解なし", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(variants))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, loc="left"); ax.set_xlabel(xlab); ax.grid(axis="y", visible=False)

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9))
    panel(ax[0], ["relax_swing_10pct", "relax_catch_10pct", "relax_hold_10pct", "T_hold=0.5", "T_hold=1.0", "T_hold=3.0"],
          ["振りの容量 +10%", "捕捉の容量 +10%", "保持の容量 +10%", "保持 0.5 s", "保持 1 s", "保持 3 s"], "(a) 相ごとの介入")
    panel(ax[1], ["tau_all x0.9", "tau_all x0.95", "tau_all x1.05", "tau_all x1.1", "tau_elbow x0.95", "tau_shoulder x0.95", "tau_hip x0.95", "tau_knee x0.95"],
          ["全関節 ×0.90", "全関節 ×0.95", "全関節 ×1.05", "全関節 ×1.10", "肘 ×0.95", "肩 ×0.95", "股 ×0.95", "膝 ×0.95"], "(b) 関節トルク容量")
    panel(ax[2], ["delta_catch=0.025", "delta_catch=0.1", "K_catch=20000", "K_catch=80000", "D_catch=750", "D_catch=3000"],
          ["$\\Delta_c$=25 ms", "$\\Delta_c$=100 ms", "剛性 ×0.5", "剛性 ×2", "減衰 ×0.5", "減衰 ×2"], "(c) 捕捉モデル")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=mcol[m]) for m in ms], labels=[f"$m$={m:g} kg" for m in ms], loc="lower left", fontsize=7)
    fig.subplots_adjust(wspace=0.75)
    fs.save(fig, FIG, "x9_interventions")
    g = lambda v: d[d["variant"] == v]["dU"]
    mac("relHoldMin", -g("relax_hold_10pct").max(), 1); mac("relHoldMax", -g("relax_hold_10pct").min(), 1)
    mac("relSwingMax", -g("relax_swing_10pct").min(), 1); mac("relCatchMax", -g("relax_catch_10pct").min(), 1)
    mac("baseResolveMax", -g("base").min() if len(g("base")) else 0.0, 2)
    mac("holdHalfMin", -g("T_hold=0.5").max(), 1); mac("holdHalfMax", -g("T_hold=0.5").min(), 1)
    mac("holdOneMax", -g("T_hold=1.0").min(), 1); mac("holdThreeMax", g("T_hold=3.0").abs().max(), 1)
    mac("dcatchMax", pd.concat([g("delta_catch=0.025"), g("delta_catch=0.1")]).abs().max(), 1)
    comp = 100 * (cbase / base - 1)
    mac("compVsImpactMin", -comp.max(), 0); mac("compVsImpactMax", -comp.min(), 0)
    mac("kHalfMin", -g("K_catch=20000").max(), 0); mac("kHalfMax", -g("K_catch=20000").min(), 0)
    mac("kDoubleMin", g("K_catch=80000").min(), 0); mac("kDoubleMax", g("K_catch=80000").max(), 0)
    cs = rd("exp9", "capability_sensitivity.csv")
    rows = [[f"{r.m:g}", f"{100 * r.step:g}", f"{r.dJ_dlntau_fd:.2f}", f"{r.dJ_dlntau_mult:.2f}", f"{r.elasticity_U:.2f}"] for r in cs.itertuples()]
    tab("tab_x9sens", ["$m$ [kg]", "刻み [\\%]", "有限差分 $\\partial J^*/\\partial\\ln\\tau^{\\rm cap}$", "乗数", "$U_{\\rm peak}$ の弾性"], rows)
    lo = cs[cs["m"] < 70]
    mac("elastUmin", lo["elasticity_U"].abs().min(), 2); mac("elastUmax", lo["elasticity_U"].abs().max(), 2)
    mac("sensRatioMin", (lo["dJ_dlntau_fd"] / lo["dJ_dlntau_mult"]).min(), 2); mac("sensRatioMax", (lo["dJ_dlntau_fd"] / lo["dJ_dlntau_mult"]).max(), 2)
    el = d[(d["variant"] == "tau_elbow x0.95") & (d["m"] == 75.0)]["dU"]
    mac("elbowHeavy", el.iloc[0] if len(el) else np.nan, 1)
    win = d[d["U_catch_avg50ms"].notna()] if "U_catch_avg50ms" in d else d.iloc[:0]
    mac("winVsInstMax", 100 * (1 - win["U_catch_avg50ms"] / win["U_catch_inst"]).max() if len(win) else 0.0, 0)


# ===================================================================================================== experiment 2
def exp2():
    r = rd("exp2", "replays.csv")
    r["setting"] = np.where(r.group == "mesh", r["mesh"].astype(str), np.where(r.group == "phase_start", r["start"].astype(str),
                            np.where(r.group == "sub_dt", (1000 * r["sub_dt"]).map("{:g}".format), (1000 * r["control_dt"]).map("{:g}".format))))
    g = r.groupby(["catch_model", "group", "setting"]).agg(n=("succ_2", "size"), released=("released", "sum"), caught=("caught", "sum"), held=("success_held", "sum"),
                                                           s2=("succ_2", "sum"), err=("swing_err", "median")).reset_index()
    panels = [("control_dt", ["1", "2", "5", "10", "20"], "制御周期 [ms]\n（離手時刻は厳密）"), ("release_rounding", ["1", "2", "5", "10", "20"], "制御周期 [ms]\n（離手を周期格子に丸め）"),
              ("sub_dt", ["1", "0.5", "0.25"], "積分刻み [ms]"), ("mesh", ["0.5x", "1x", "2x"], "計画メッシュ"), ("phase_start", ["rest", "flight", "catch", "hold"], "開始相")]
    jp = {"rest": "静止", "flight": "飛行", "catch": "捕捉", "hold": "保持", "0.5x": "0.5×", "1x": "1×", "2x": "2×"}
    fig, ax = plt.subplots(2, 5, figsize=(7.1, 3.3), sharey=True, gridspec_kw=dict(width_ratios=[5, 5, 3, 3, 4]))
    for row, cm in enumerate(("impact", "compliant")):
        for col, (grp, sets, xl) in enumerate(panels):
            a = ax[row, col]; gg = g[(g.catch_model == cm) & (g.group == grp)].set_index("setting")
            x = np.arange(len(sets))
            for k, (key, colr, lab) in enumerate((("released", fs.NEUTRAL, "離手"), ("caught", C["blue"], "捕捉"), ("s2", fs.GOOD, "成功（2 s 保持，$F_{\\max}$=2600 N）"))):
                a.bar(x + (k - 1) * 0.27, [gg.loc[s, key] if s in gg.index else 0 for s in sets], width=0.25, color=colr, edgecolor="white", linewidth=0.8, label=lab)
            a.set_xticks(x); a.set_xticklabels([jp.get(s, s) for s in sets]); a.set_ylim(0, 3.25); a.set_yticks([0, 1, 2, 3]); a.grid(axis="x", visible=False)
            if row == 1:
                a.set_xlabel(xl)
            if col == 0:
                a.set_ylabel(("剛体衝突" if cm == "impact" else "順応捕捉") + "の参照解\n本数（3 体格中）")
    ax[0, 0].legend(loc="upper center", bbox_to_anchor=(2.9, 1.42), ncol=3)
    fig.subplots_adjust(wspace=0.12, hspace=0.18)
    fs.save(fig, FIG, "x2_replay")
    gi = g.set_index(["catch_model", "group", "setting"])
    v = lambda cm, grp, s, key="s2": int(gi.loc[(cm, grp, s), key]) if (cm, grp, s) in gi.index else 0
    for cm, tag in (("impact", "Imp"), ("compliant", "Comp")):
        mac(f"rep{tag}Two", v(cm, "control_dt", "2")); mac(f"rep{tag}Five", v(cm, "control_dt", "5")); mac(f"rep{tag}Ten", v(cm, "control_dt", "10"))
        mac(f"rep{tag}TwentyRel", v(cm, "control_dt", "20", "released")); mac(f"rep{tag}RoundTwo", v(cm, "release_rounding", "2"))
        mac(f"rep{tag}MeshHalf", v(cm, "mesh", "0.5x")); mac(f"rep{tag}MeshOne", v(cm, "mesh", "1x")); mac(f"rep{tag}MeshTwo", v(cm, "mesh", "2x"))
        mac(f"rep{tag}MeshTwoHeld", v(cm, "mesh", "2x", "held")); mac(f"rep{tag}MeshOneHeld", v(cm, "mesh", "1x", "held"))
        mac(f"rep{tag}StartCatch", v(cm, "phase_start", "catch")); mac(f"rep{tag}StartFlight", v(cm, "phase_start", "flight"))
        mac(f"rep{tag}TenCaught", v(cm, "control_dt", "10", "caught"))
    e = lambda cm, s: 100 * float(gi.loc[(cm, "mesh", s), "err"])
    mac("meshErrHalf", e("compliant", "0.5x"), 1); mac("meshErrOne", e("compliant", "1x"), 1); mac("meshErrTwo", e("compliant", "2x"), 1)
    ms = rd("exp2", "mesh_solves.csv")
    ms["mesh"] = ms["task_key"].str.extract(r"_(0\.5x|1x|2x)$"); ms["ref"] = ms["task_key"].str.replace(r"\.pkl_.*$", "", regex=True)
    pv = ms.pivot_table(index="ref", columns="mesh", values="U_peak")
    mac("meshUhalf", (100 * (pv["0.5x"] / pv["2x"] - 1)).abs().max(), 0); mac("meshUone", (100 * (pv["1x"] / pv["2x"] - 1)).abs().max(), 1)
    sub = r[r.group == "sub_dt"].pivot_table(index="ref_file", columns="sub_dt", values="U_peak")
    mac("subdtUspread", (100 * (sub.max(1) / sub.min(1) - 1)).max(), 0)
    rows = []
    for cm, cmj in (("impact", "剛体衝突"), ("compliant", "順応捕捉")):
        for grp, sets, xl in panels:
            for s in sets:
                if (cm, grp, s) in gi.index:
                    q = gi.loc[(cm, grp, s)]
                    rows.append([cmj, xl.split("\n")[0] + ("（丸め）" if grp == "release_rounding" else ""), jp.get(s, s), int(q.n), int(q.released), int(q.caught), int(q.held), int(q.s2),
                                 "---" if not np.isfinite(q.err) else f"{100 * q.err:.1f}"])
        rows.append("---")
    tab("tab_x2", ["参照解", "分解の軸", "設定", "$n$", "離手", "捕捉", "2 s 保持", "成功", "振り追従誤差 [$10^{-2}$ rad]"], rows[:-1], "lllrrrrrr")


PARTS = dict(exp1=exp1, exp9=exp9, exp2=exp2)

# ===================================================================================================== experiment 3
VAR_JP = {"A_base": "余裕なし（解き直し）", "B_wall2mm": "壁 2 mm", "B_wall5mm": "壁 5 mm", "B_joint2deg": "関節 2°", "B_joint5deg": "関節 5°", "B_cone5": "錐 5%",
          "B_cone10": "錐 10%", "B_all_small": "小さい余裕すべて", "B_all_large": "大きい余裕すべて", "C_robust2ms": "離手 ±2 ms のシナリオ",
          "D_small_robust": "小さい余裕＋シナリオ", "D_large_robust": "大きい余裕＋シナリオ"}


def exp3():
    p = rd("exp3", "plans.csv"); t = rd("exp3", "trials.csv")
    base = p[p.variant == "A_base"].set_index("ref")["U_peak"]
    p["dU"] = 100 * (p["U_peak"] / p["ref"].map(base) - 1)
    okp = p[p["ok"] == True]
    lv_cols = [f"succ_{lv:g}" for lv in LEVELS]
    groups = [("壁との隙間", "B_wall2mm", "B_wall5mm", "2 mm", "5 mm"), ("関節可動域", "B_joint2deg", "B_joint5deg", "2°", "5°"), ("摩擦・引っ掛かり錐", "B_cone5", "B_cone10", "5%", "10%"),
              ("すべて", "B_all_small", "B_all_large", "小", "大")]
    fig, ax = plt.subplots(2, 4, figsize=(7.1, 3.5), sharex=True, sharey=True)
    for row, (pert, ttl) in enumerate((("none", "状態の摂動なし（離手時刻誤差 0，±1，±3 ms）"), ("weak", "弱い摂動（$\sigma_\theta$=0.01 rad，$\sigma_{\dot\theta}$=0.1 rad/s）"))):
        for col, (name, v1, v2, l1, l2) in enumerate(groups):
            a = ax[row, col]
            for v, colr, lab, ls in (("A_base", MUTED, "余裕なし", "-"), (v1, C["blue"], l1, "-"), (v2, C["orange"], l2, "-")):
                g = t[(t.variant == v) & (t.perturb == pert)]
                if len(g):
                    a.plot(LEVELS, g[lv_cols].mean().values, color=colr, marker="o", ms=3.5, lw=1.5, ls=ls, label=f"{lab}（$n$={len(g)}）")
            a.set_ylim(-0.03, 1.0); a.set_xticks([1.0, 1.5, 2.0, 2.5])
            if row == 0:
                a.set_title(name)
            a.legend(loc="upper left", fontsize=6.3, handlelength=1.2)
            if col == 0:
                a.set_ylabel(("摂動なし" if row == 0 else "弱い摂動") + "\n成功率")
            if row == 1:
                a.set_xlabel("許容容量 $F_{\max}/F_{\mathrm{ref}}$")
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    fs.save(fig, FIG, "x3_margins")
    rows = []
    for v in VAR_JP:
        pv = p[p.variant == v]
        if not len(pv):
            continue
        cells = [VAR_JP[v], f"{int((pv['ok'] == True).sum())}/{len(pv)}"]
        o = okp[okp.variant == v]
        cells.append("---" if not len(o) or v == "A_base" else f"{o['dU'].min():+.1f}〜{o['dU'].max():+.1f}")
        for pert in ("none", "weak", "strong"):
            g = t[(t.variant == v) & (t.perturb == pert)]
            cells.append("---" if not len(g) else f"{pct(g['succ_2'].mean())}（{len(g)}）")
        rows.append(cells)
    tab("tab_x3", ["計画の余裕", "求解成功", "$\\Upk$ の増分 [\\%]", "成功率：摂動なし", "弱い摂動", "強い摂動"], rows, "lrrrrr")
    sv = lambda v, pert: t[(t.variant == v) & (t.perturb == pert)]["succ_2"].mean()
    mac("marBaseNone", pct(sv("A_base", "none"))); mac("marBaseWeak", pct(sv("A_base", "weak"))); mac("marBaseStrong", pct(sv("A_base", "strong")))
    best = max(((sv(v, "weak"), v) for v in VAR_JP if len(t[(t.variant == v) & (t.perturb == "weak")])), key=lambda x: x[0])
    mac("marBestWeak", pct(best[0])); MAC["marBestWeakName"] = VAR_JP[best[1]]
    mac("marStrongMax", pct(max(sv(v, "strong") for v in VAR_JP if len(t[(t.variant == v) & (t.perturb == "strong")]))))
    rob = p[p.variant.isin(["C_robust2ms", "D_small_robust", "D_large_robust"])]
    mac("robN", len(rob)); mac("robOk", int((rob["ok"] == True).sum())); mac("robInfeasible", int(rob["status"].str.contains("Infeasible").sum()))
    mac("robTimeout", int(rob["status"].str.contains("CpuTime").sum()))
    for v, nm in (("B_wall2mm", "marWallTwo"), ("B_wall5mm", "marWallFive"), ("B_joint2deg", "marJointTwo"), ("B_joint5deg", "marJointFive"), ("B_cone10", "marConeTen"), ("B_all_large", "marAllLarge"), ("B_all_small", "marAllSmall")):
        o = okp[okp.variant == v]["dU"]
        mac(nm + "Med", o.median(), 1); mac(nm + "Max", o.max(), 1)
    # release-timing error alone (base plans)
    g = t[(t.variant == "A_base") & (t.perturb == "none")].groupby("dt_release")["succ_2"].mean()
    mac("dtZero", pct(g.get(0.0, np.nan))); mac("dtOne", pct(np.mean([g.get(-0.001, np.nan), g.get(0.001, np.nan)]))); mac("dtThree", pct(np.mean([g.get(-0.003, np.nan), g.get(0.003, np.nan)])))


# ===================================================================================================== experiment 4
def exp4():
    u = rd("exp4", "u0_J.csv"); u = u[(u[["minus", "none", "plus", "ref"]] < 20).all(axis=1)]
    spread = (u[["minus", "none", "plus", "ref"]].max(axis=1) - u[["minus", "none", "plus", "ref"]].min(axis=1))
    mac("uzeroSpreadMed", spread.median(), 3); mac("uzeroSpreadMax", spread.max(), 3); mac("uzeroRelMax", 100 * (spread / u["none"]).max(), 1)
    tr = rd("exp4", "trajectory_labels.csv"); tr = tr[tr["ok"] == 1]
    ts = json.load(open(os.path.join(S, "exp4", "trajectory_summary.json")))
    mac("trajN", ts["n"]); mac("trajVmed", 100 * ts["V_rel_err_median"], 2); mac("trajVmax", 100 * ts["V_rel_err_max"], 1); mac("trajTau", ts["tau_err_median_s"], 3)
    mac("trajCos", ts["p_cos_median"], 2); mac("trajPrel", 100 * ts["p_rel_err_median"], 0)
    ng = json.load(open(os.path.join(S, "exp4_newmodels", "net_gradient_test.json")))
    mac("netGradErr", 100 * ng["autograd_vs_fd_rel_median"], 1); mac("netLabelSign", 100 * ng["label_vs_net_sign_agreement"], 0)
    f1 = json.load(open(os.path.join(S, "exp4", "fd_summary.json")))
    mac("fdLooseRel", f1["rel_err_median"], 2); mac("fdLooseSign", 100 * f1["sign_agreement"], 0)
    fb = rd("exp4", "fd_tight_base.csv"); fd = rd("exp4", "fd_check_tight_merged.csv") if os.path.exists(os.path.join(S, "exp4", "fd_check_tight_merged.csv")) else None
    okb = fb[fb["ok"] == 1]
    mac("fdBaseOk", len(okb)); mac("fdBaseAll", len(fb)); mac("fdNoiseMed", 1000 * okb["noise"].median(), 1); mac("fdNoiseMax", 1000 * okb["noise"].max(), 1)
    mac("fdRepCos", okb["p_rep_cos"].min(), 4)
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.0]))
    a = ax[0]
    for k_, colr, lab in (("none", MUTED, "指定なし"), ("ref", C["blue"], "参照の指令"), ("plus", C["orange"], "+0.3"), ("minus", C["aqua"], "−0.3")):
        a.plot(u["k"], u[k_] - u["none"], marker="o", ms=3.5, lw=1.2, color=colr, label=lab)
    a.set_xlabel("振りの節点番号"); a.set_ylabel("$J^*$ の差（指定なしを基準）"); a.set_title("(a) 直前の指令への依存", loc="left"); a.legend(fontsize=6.3, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    a = ax[1]
    if fd is not None and len(fd):
        epss = sorted(fd["eps"].unique()); cols = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(epss)))
        for e, colr in zip(epss, cols):
            g = fd[fd["eps"] == e]
            a.scatter(g["D_dual"], g["D_fd"], s=14, color=colr, edgecolor="white", linewidth=0.5, label=f"$\\delta$={e:g}", zorder=3)
        lim = float(np.nanpercentile(np.abs(fd["D_dual"]), 99)) * 1.4
        a.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8); a.set_xlim(-lim, lim)
        big = fd[fd["eps"] >= 0.03]
        yl = float(np.nanpercentile(np.abs(big["D_fd"]), 97)) * 1.3 if len(big) else lim * 4
        a.set_ylim(-yl, yl)
        a.legend(fontsize=6.3, ncol=2, loc="upper left", handletextpad=0.1, columnspacing=0.6)
        rows = []
        for e, g in fd.groupby("eps"):
            g2 = g[g["resolvable"] == 1]
            rows.append([f"{e:g}", len(g), int(g["resolvable"].sum()), f"{g['rel_err'].median():.2f}", ("---" if not len(g2) else f"{g2['rel_err'].median():.2f}"),
                         pct(g["sign_ok"].mean()), ("---" if len(g) < 3 else f"{np.corrcoef(g['D_dual'], g['D_fd'])[0, 1]:.2f}"), pct(g["active_set_changed"].mean())])
        tab("tab_x4fd", ["刻み $\\delta$", "$n$", "分解可能", "相対誤差（中央値）", "同（分解可能のみ）", "符号一致 [\\%]", "相関", "活性集合の変化 [\\%]"], rows)
        bigg = fd[fd["eps"] >= 0.1]
        if len(bigg) > 3:
            mac("fdBigCorr", float(np.corrcoef(bigg["D_dual"], bigg["D_fd"])[0, 1]), 2); mac("fdBigSign", pct(bigg["sign_ok"].mean())); mac("fdBigRel", bigg["rel_err"].median(), 2)
            mac("fdBigN", len(bigg)); mac("fdBigResolvable", int(bigg["resolvable"].sum()))
        small = fd[fd["eps"] <= 0.011]
        mac("fdSmallCorr", float(np.corrcoef(small["D_dual"], small["D_fd"])[0, 1]), 2); mac("fdSmallSign", pct(small["sign_ok"].mean())); mac("fdSmallN", len(small))
        mac("fdSmallResolvable", int(small["resolvable"].sum()))
    if fd is not None and len(fd) and (fd["eps"] >= 0.1).sum() > 3:
        bigg = fd[fd["eps"] >= 0.1]; cr = float(np.corrcoef(bigg["D_dual"], bigg["D_fd"])[0, 1]); sg = bigg["sign_ok"].mean(); nres = int(bigg["resolvable"].sum())
        if cr > 0.7 and sg > 0.8:
            t_ = ("刻みを 0.1 以上にすると信号がばらつきを上回り（\\fdBigN 組中 \\fdBigResolvable 組），costate による方向微分と有限差分の相関は \\fdBigCorr，符号の一致は \\fdBigSign\\,\\%になる．"
                  "解き直しのばらつきを超える大きさの変化については，costate は価値の変化を正しく予測している．")
        else:
            t_ = ("刻みを 0.1 以上にすると，\\fdBigN 組中 \\fdBigResolvable 組で信号がばらつきを上回るが，costate による方向微分と有限差分の相関は \\fdBigCorr，符号の一致は \\fdBigSign\\,\\%にとどまる．"
                  "この刻みでは活性な制約の組が変わる場合が多く，一次の予測である costate と有限の変化は一致しない．"
                  "したがって本稿では，costate ラベルの正しさを有限差分によって確認できていない．確認できたのは，乗数が解き直しで再現することと，次に述べる軌道ラベルとの整合だけである．")
        open(os.path.join(PAPER, "fd_paragraph.tex"), "w").write(t_ + "\n")
    a.set_xlabel("乗数による方向微分 $p^{*\\top}d$"); a.set_ylabel("有限差分 $\\Delta J^*/2\\delta$"); a.set_title("(b) costate と有限差分", loc="left")
    a = ax[2]
    a.scatter(tr["V_resolve"], tr["V_label"], s=14, color=C["blue"], edgecolor="white", linewidth=0.5, zorder=3)
    lo, hi = tr["V_resolve"].min() - 0.1, tr["V_resolve"].max() + 0.1
    a.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8); a.set_xlabel("独立な解き直しの $V^*$"); a.set_ylabel("軌道ラベルの $V$"); a.set_title("(c) 軌道ラベルの価値", loc="left")
    fig.subplots_adjust(wspace=0.42)
    fs.save(fig, FIG, "x4_audit")


# ===================================================================================================== experiment 5
def exp5():
    r = rd("exp5", "ranking_scores.csv"); c = rd("exp5", "candidates.csv")
    g = r.groupby(["model", "ood", "H"]).agg(best=("best_pick", "mean"), regret=("regret", "mean"), rho=("rho", "mean"), n=("n", "size")).reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.2))
    lab = {("value-only", 0): ("値のみ", C["orange"], ""), ("value-only", 1): ("値のみ＋分布外罰則", C["orange"], "//"), ("Sobolev", 0): ("Sobolev", C["blue"], ""), ("Sobolev", 1): ("Sobolev＋分布外罰則", C["blue"], "//")}
    for j, (key, yl) in enumerate((("best", "最良候補を選ぶ割合"), ("rho", "順位相関（Spearman）"), ("regret", "選択の後悔 $\\Delta Q$（平均）"))):
        a = ax[j]
        for i, ((mdl, ood), (nm, colr, hatch)) in enumerate(lab.items()):
            for h_i, H in enumerate((0.3, 0.8)):
                v = g[(g.model == mdl) & (g.ood == ood) & (g.H == H)][key]
                a.bar(h_i + (i - 1.5) * 0.2, v.iloc[0] if len(v) else 0, width=0.18, color=colr, alpha=0.55 if ood else 1.0, edgecolor="white", linewidth=0.8, hatch=hatch,
                      label=nm if h_i == 0 else None)
        a.set_xticks([0, 1]); a.set_xticklabels(["地平 0.3 s", "地平 0.8 s"]); a.set_ylabel(yl); a.grid(axis="x", visible=False)
        if key == "best":
            a.axhline(1 / 8, color=MUTED, lw=0.8, ls="--"); a.text(0.5, 1 / 8 + 0.015, "無作為", fontsize=6.5, color=INK2, ha="center")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=6.5)
    fig.subplots_adjust(wspace=0.38)
    fs.save(fig, FIG, "x5_ranking")
    gi = g.set_index(["model", "ood", "H"])
    for mdl, tag in (("Sobolev", "Sob"), ("value-only", "Val")):
        for H, ht in ((0.3, "Short"), (0.8, "Long")):
            mac(f"rank{tag}{ht}", pct(gi.loc[(mdl, 0, H), "best"])); mac(f"rank{tag}{ht}Ood", pct(gi.loc[(mdl, 1, H), "best"])); mac(f"rankRho{tag}{ht}", gi.loc[(mdl, 0, H), "rho"], 2)
            mac(f"rankReg{tag}{ht}", gi.loc[(mdl, 0, H), "regret"], 3)
    cs = c[c.get("solved", 0) == 1]
    sp = cs.groupby(["state", "H"])["Q_ref"].agg(lambda x: x.max() - x.min()).reset_index()
    sp = sp[cs.groupby(["state", "H"]).size().values >= 3]
    mac("candN", int((c.get("feasible", 0) == 1).sum())); mac("candSolved", len(cs))
    mac("spreadShortMed", sp[sp.H == 0.3]["Q_ref"].median(), 3); mac("spreadLongMed", sp[sp.H == 0.8]["Q_ref"].median(), 3)
    mac("spreadShortMin", sp[sp.H == 0.3]["Q_ref"].min(), 3); mac("spreadShortMax", sp[sp.H == 0.3]["Q_ref"].max(), 3)
    mac("spreadLongMin", sp[sp.H == 0.8]["Q_ref"].min(), 2); mac("spreadLongMax", sp[sp.H == 0.8]["Q_ref"].max(), 2)
    rows = []
    for (mdl, ood), (nm, _, _) in lab.items():
        for H in (0.3, 0.8):
            q = gi.loc[(mdl, ood, H)]
            rows.append([nm, f"{H:g}", int(q["n"]), pct(q["best"]), f"{q['rho']:.2f}", f"{q['regret']:.3f}"])
    tab("tab_x5", ["終端価値", "地平 [s]", "状態$\\times$シード", "最良候補 [\\%]", "順位相関", "後悔 $\\Delta Q$"], rows, "lrrrrr")


# ===================================================================================================== experiment 6
FF_JP = {"none": "成功", "slipped off A": "A から滑落", "hit wall": "壁に接触", "no release": "離手せず（期限）", "missed B": "B を掴み損ね", "hit B's face": "B の前面に衝突",
         "lost hook": "保持中に指が外れる", "grip capacity exceeded": "把持容量超過"}
FF_COL = {"none": fs.GOOD, "slipped off A": C["blue"], "hit wall": C["orange"], "no release": C["yellow"], "missed B": C["aqua"], "hit B's face": C["magenta"],
          "lost hook": C["violet"], "grip capacity exceeded": C["red"]}
ORDER = ["REPLAY", "BC", "TRACK", "VMPC", "SMPC", "ORACLE", "ORACLE+"]


def exp6():
    t = rd("exp6", "trials.csv"); t = t[t["error"].isna()] if "error" in t else t
    m = t[t.group == "main"]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.75), gridspec_kw=dict(width_ratios=[1.25, 0.9, 1.35]))
    a = ax[0]; y = np.arange(len(ORDER))
    for k, (key, colr, lab) in enumerate((("released", fs.NEUTRAL, "離手"), ("caught", C["blue"], "捕捉"), ("succ_2", fs.GOOD, "成功"))):
        v = [m[m.method == o][key].mean() for o in ORDER]
        a.barh(y + (k - 1) * 0.27, v, height=0.25, color=colr, edgecolor="white", linewidth=0.8, label=lab)
    a.set_yticks(y); a.set_yticklabels([f"{METHOD_JP[o]}（{len(m[m.method == o])}）" for o in ORDER]); a.invert_yaxis(); a.set_xlim(0, 1.0); a.set_xlabel("割合"); a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6.5); a.set_title("(a) 到達した段階", loc="left")
    a = ax[1]
    for o in ORDER:
        v = [m[m.method == o][f"succ_{lv:g}"].mean() for lv in LEVELS]
        if max(v) > 0:
            a.plot(LEVELS, v, marker="o", ms=3.5, color=METHOD_COL[o], label=METHOD_JP[o])
    a.plot(LEVELS, [0] * len(LEVELS), color=INK2, lw=1.0, label="他の 5 手法（すべて 0）")
    a.set_xlabel("許容容量 $F_{\\max}/F_{\\mathrm{ref}}$"); a.set_ylabel("成功率"); a.set_ylim(-0.02, 0.5); a.set_xticks([1.0, 1.5, 2.0, 2.5]); a.legend(loc="upper left", fontsize=6.3); a.set_title("(b) 許容容量と成功率", loc="left")
    a = ax[2]
    ff = m.groupby(["method", "first_failure"]).size().unstack(fill_value=0).reindex(ORDER).fillna(0)
    ff = ff.div(ff.sum(axis=1), axis=0); left = np.zeros(len(ORDER))
    for key in FF_JP:
        if key in ff:
            a.barh(y, ff[key].values, left=left, height=0.6, color=FF_COL[key], edgecolor="white", linewidth=1.0, label=FF_JP[key]); left += ff[key].values
    a.set_yticks(y); a.set_yticklabels([o for o in ORDER], fontsize=6.5); a.invert_yaxis(); a.set_xlim(0, 1); a.set_xlabel("最初の失敗の内訳"); a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.3, columnspacing=0.8, handlelength=1.0); a.set_title("(c) 最初の失敗", loc="left")
    fig.subplots_adjust(wspace=0.30)
    fs.save(fig, FIG, "x6_compare")
    rows = []
    for o in ORDER:
        g = m[m.method == o]
        top = g[g.first_failure != "none"]["first_failure"].value_counts()
        rows.append([METHOD_JP[o], len(g), int(g["learn_seed"].nunique()) if o in ("BC", "VMPC", "SMPC") else "---", pct(g["released"].mean(), 1), pct(g["caught"].mean(), 1), pct(g["success_held"].mean(), 1)] +
                    [pct(g[f"succ_{lv:g}"].mean(), 1) for lv in (1.0, 1.5, 2.0, 2.5)] + [f"{g['U_peak'].median():.2f}", (FF_JP.get(top.index[0], top.index[0]) + f"（{top.iloc[0]}）") if len(top) else "---"])
    tab("tab_x6", ["手法", "$n$", "学習シード", "離手", "捕捉", "2 s 保持", "$S_{1.0}$", "$S_{1.5}$", "$S_{2.0}$", "$S_{2.5}$", "$\\Upk$ 中央値", "最多の失敗"], rows, "lrrrrrrrrrrl")
    dS = json.load(open(os.path.join(S, "exp6", "delta_S.json")))
    mac("deltaS", 100 * dS["delta_S"], 1); mac("deltaSlo", 100 * dS["ci95"][0], 1); mac("deltaShi", 100 * dS["ci95"][1], 1); mac("deltaSpairs", dS["n_pairs"])
    for o, tag in (("SMPC", "Smpc"), ("VMPC", "Vmpc"), ("BC", "Bc"), ("REPLAY", "Replay"), ("ORACLE", "Oracle"), ("ORACLE+", "OraclePlus"), ("TRACK", "Track")):
        g = m[m.method == o]
        mac(f"cl{tag}N", len(g)); mac(f"cl{tag}Rel", pct(g["released"].mean(), 1)); mac(f"cl{tag}Catch", pct(g["caught"].mean(), 1)); mac(f"cl{tag}Succ", pct(g["succ_2"].mean(), 1))
        mac(f"cl{tag}SuccInt", pct(g["succ_2"].mean(), 0))
    # release-rate difference with a paired bootstrap over (ref, perturbation seed, learning seed)
    sm = m[m.method == "SMPC"].set_index(["ref", "seed", "learn_seed"])["released"]; vm = m[m.method == "VMPC"].set_index(["ref", "seed", "learn_seed"])["released"]
    dd = (sm - vm).dropna().values; rng = np.random.default_rng(0)
    bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
    mac("relDiff", 100 * dd.mean(), 1); mac("relDiffLo", 100 * np.percentile(bs, 2.5), 1); mac("relDiffHi", 100 * np.percentile(bs, 97.5), 1)
    by = m[m.method.isin(["SMPC", "VMPC"])].groupby(["method", "learn_seed"])["released"].mean().unstack()
    MAC["relSmpcSeeds"] = " / ".join(pct(v) for v in by.loc["SMPC"].values); MAC["relVmpcSeeds"] = " / ".join(pct(v) for v in by.loc["VMPC"].values)
    bb = m[m.method.isin(["SMPC", "VMPC"])].groupby(["method", "m"])["released"].mean().unstack()
    MAC["relSmpcBodies"] = " / ".join(pct(v) for v in bb.loc["SMPC"].values); MAC["relVmpcBodies"] = " / ".join(pct(v) for v in bb.loc["VMPC"].values)
    MAC["clBodies"] = " / ".join(f"{b:g}" for b in bb.columns)
    for o, tag in (("SMPC", "Smpc"), ("VMPC", "Vmpc"), ("ORACLE", "Oracle"), ("TRACK", "Track"), ("REPLAY", "Replay")):
        g = m[m.method == o]["first_failure"].value_counts()
        for key, kt in (("no release", "NoRel"), ("slipped off A", "Slip"), ("hit wall", "Wall"), ("hit B's face", "Face"), ("missed B", "Miss")):
            mac(f"ff{tag}{kt}", int(g.get(key, 0)))
    r = t[t.group == "recovery"]
    rows = [[METHOD_JP[o], len(g), pct(g["released"].mean()), pct(g["caught"].mean()), pct(g["succ_2"].mean())] for o in ORDER for g in [r[r.method == o]] if len(g)]
    tab("tab_x6rec", ["手法", "$n$", "離手 [\\%]", "捕捉 [\\%]", "成功 [\\%]"], rows)
    for o, tag in (("SMPC", "Smpc"), ("VMPC", "Vmpc"), ("TRACK", "Track")):
        g = m[(m.method == o) & (m.released == 1)]; ffr = g["first_failure"].value_counts()
        mac(f"rel{tag}N", len(g)); mac(f"rel{tag}Dmin", g["d_min"].median(), 1); mac(f"rel{tag}Wall", int(ffr.get("hit wall", 0))); mac(f"rel{tag}Miss", int(ffr.get("missed B", 0)))
    wall = m.groupby("method")["wall_s"].median()
    mac("wallSmpc", wall["SMPC"] / 60, 0); mac("wallOracle", wall["ORACLE"] / 60, 0)
    h = rd("exp6_h40", "trials.csv"); h = h[h["error"].isna()] if "error" in h else h
    for o, tag in (("SMPC", "Smpc"), ("VMPC", "Vmpc")):
        g = h[h.method == o]
        mac(f"long{tag}N", len(g)); mac(f"long{tag}Rel", pct(g["released"].mean())); mac(f"long{tag}Catch", pct(g["caught"].mean()))


# ===================================================================================================== experiment 7
def exp7():
    lc = rd("exp7", "learning_curve.csv"); lc = lc[lc.method.isin(["SMPC", "VMPC"])].copy()
    base = pd.read_csv(os.path.join(S, "dual_field_rows_frozen.csv")); bok = base[(base["ok"] == 1) & (base["m"] != 69)]
    n_tot = len(bok)                                         # labelled solves available for training (held-out body excluded)
    lc["x"] = lc["budget"].where(lc["budget"] > 0, n_tot)
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))
    for meth in ("VMPC", "SMPC"):
        for coll, ls, mk in (("reference", "-", "o"), ("visited", "none", "D")):
            g = lc[(lc.method == meth) & (lc.collection == coll)].sort_values("x")
            if len(g):
                ax.plot(g["x"], g["released"], color=METHOD_COL[meth], marker=mk, ms=5 if mk == "o" else 6.5, ls=ls, markeredgecolor="white", markeredgewidth=0.8,
                        label=f"{METHOD_JP[meth]}：{'参照近傍' if coll == 'reference' else '訪問状態を含む'}")
    ax.set_xscale("log"); ax.set_xticks([100, 300, 600, n_tot]); ax.set_xticklabels(["100", "300", "600", f"{n_tot}"]); ax.minorticks_off()
    ax.set_xlabel("学習に使った求解の数"); ax.set_ylabel("離手率"); ax.set_ylim(-0.02, 0.45); ax.legend(fontsize=6.3, loc="upper left")
    fs.save(fig, FIG, "x7_data")
    li = lc.set_index(["collection", "budget", "method"])
    v = lambda c_, b, m_: pct(li.loc[(c_, b, m_), "released"], 1) if (c_, b, m_) in li.index else "---"
    for b, bt in ((100, "Hundred"), (300, "ThreeHundred"), (600, "SixHundred")):
        mac(f"data{bt}Smpc", v("reference", b, "SMPC")); mac(f"data{bt}Vmpc", v("reference", b, "VMPC"))
    mac("dataVisitedSmpc", v("visited", 600, "SMPC")); mac("dataVisitedVmpc", v("visited", 600, "VMPC")); mac("dataNall", n_tot)
    mac("dataAnyCatch", pct(lc["caught"].max()))
    # composition of the visited-state training set (same selection as train_fields.py: sorted sources, rng(0) shuffle, first 600)
    vis = pd.read_csv(os.path.join(S, "exp7", "dagger_data", "rows.csv")); vok = vis[vis["ok"] == 1]
    src = sorted(set("sol_" + bok["tag"] + ".pkl") | set("sol_" + vok["tag"] + ".pkl")); rng = np.random.default_rng(0); rng.shuffle(src)
    nv = len(set(src[:600]) & set("sol_" + vok["tag"] + ".pkl"))
    mac("daggerAll", len(vis)); mac("daggerOk", len(vok)); mac("dataVisitedN", nv); mac("dataVisitedRefN", 600 - nv)
    rows = [[("参照近傍" if r_.collection == "reference" else "訪問状態を含む"), (f"{int(r_.budget)}" if r_.budget > 0 else f"{n_tot}"), METHOD_JP[r_.method], int(r_.n), pct(r_.released, 1), pct(r_.caught, 1), pct(r_.succ_2, 1)]
            for r_ in lc.sort_values(["collection", "x", "method"]).itertuples()]
    tab("tab_x7", ["データの集め方", "求解数", "手法", "$n$", "離手 [\\%]", "捕捉 [\\%]", "成功 [\\%]"], rows, "lrlrrrr")


# ===================================================================================================== experiment 8 + fits
def exp8():
    fits = []
    for f in sorted(glob.glob(os.path.join(S, "models", "full", "fit_seed*.json"))):
        j = json.load(open(f)); fits += j["fits"]; info = j["info"]
    F = pd.DataFrame(fits)
    mac("fitNtrainS", info["n_train"]); mac("fitNvalS", info["n_val"]); mac("fitNholdS", info["n_test"]); mac("fitNseed", F["seed"].nunique())
    rows = []
    for split, sj in (("train", "学習"), ("val", "未知軌道"), ("holdout", "未知体格（$m$=69 kg）")):
        for mdl, mj, tag in (("DFL(Sobolev)", "Sobolev", "Sob"), ("DFL(value-only)", "値のみ", "Val")):
            g = F[(F.split == split) & (F.model == mdl)]
            if not len(g):
                continue
            f_ = lambda c_: f"{g[c_].mean():.2f}"
            rows.append([sj, mj, f_("r2_V"), f_("r2_tau"), f_("r2_U"), f_("rho_p"), f_("sign_p"), f"{g['rmse_tau'].mean():.2f}"])
            st = {"train": "Train", "val": "Val", "holdout": "Hold"}[split]
            mac(f"fit{tag}{st}V", g["r2_V"].mean(), 2); mac(f"fit{tag}{st}Rho", g["rho_p"].mean(), 2); mac(f"fit{tag}{st}U", g["r2_U"].mean(), 2); mac(f"fit{tag}{st}Tau", g["r2_tau"].mean(), 2)
            mac(f"fit{tag}{st}Vmin", g["r2_V"].min(), 2); mac(f"fit{tag}{st}Vmax", g["r2_V"].max(), 2)
        rows.append("---")
    tab("tab_xfit", ["評価集合", "学習", "$R^2(V)$", "$R^2(t_{\\rm go})$", "$R^2(U)$", "costate $\\rho_S$", "符号一致", "$t_{\\rm go}$ RMSE [s]"], rows[:-1], "llrrrrrr")
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0, 1.1]))
    for j, (nm, ttl) in enumerate((("mass", "(a) 質量の外挿"), ("period", "(b) 周期の外挿"))):
        d = rd("exp8", f"fit_by_condition_{nm}.csv")
        g = d.groupby(["model", "train"])[["r2_V", "rho_p"]].mean()
        a = ax[j]
        for i, (mdl, colr, mj) in enumerate((("value-only", C["orange"], "値のみ"), ("Sobolev", C["blue"], "Sobolev"))):
            for k_, (key, hatch) in enumerate((("r2_V", ""), ("rho_p", "//"))):
                vals = [g.loc[(mdl, True), key], g.loc[(mdl, False), key]]
                a.bar(np.arange(2) + (2 * i + k_ - 1.5) * 0.2, vals, width=0.18, color=colr, alpha=1.0 if k_ == 0 else 0.55, hatch=hatch, edgecolor="white", linewidth=0.8,
                      label=mj + "：" + ("$R^2(V)$" if k_ == 0 else "costate $\\rho_S$"))
        a.set_xticks([0, 1]); a.set_xticklabels(["学習条件", "未学習条件"]); a.set_ylim(0, 1.0); a.set_title(ttl, loc="left"); a.grid(axis="x", visible=False)
        tg = "Mass" if nm == "mass" else "Period"
        mac(f"gen{tg}SobIn", g.loc[("Sobolev", True), "r2_V"], 2); mac(f"gen{tg}SobOut", g.loc[("Sobolev", False), "r2_V"], 2)
        mac(f"gen{tg}ValIn", g.loc[("value-only", True), "r2_V"], 2); mac(f"gen{tg}ValOut", g.loc[("value-only", False), "r2_V"], 2)
        mac(f"gen{tg}SobRho", g.loc[("Sobolev", False), "rho_p"], 2); mac(f"gen{tg}ValRho", g.loc[("value-only", False), "rho_p"], 2)
    ax[0].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(1.15, -0.16), ncol=2, handlelength=1.2)
    b = rd("exp8", "bodies.csv"); a = ax[2]
    st = b[b["stature"].notna() & (b["ok"] == True)].sort_values("stature"); cp = b[b["cap_scale"].notna() & (b["ok"] == True)].sort_values("cap_scale")
    ref_U = pickle.load(open(os.path.join(ROOT, "results", "grid", "sol_ref_T18_m66_phi0.250.pkl"), "rb"))["U_peak"]
    sx = list(st["stature"]) + [1.75]; sy = list(st["U_peak"]) + [ref_U]; o = np.argsort(sx)
    a.plot(np.array(sx)[o], np.array(sy)[o], marker="o", ms=3.5, color=C["blue"]); a.set_xlabel("身長 [m]"); a.set_ylabel("$U_{\\mathrm{peak}}$"); a.set_title("(c) 体格", loc="left")
    a2 = a.inset_axes([0.45, 0.45, 0.52, 0.5])
    cx = list(cp["cap_scale"]) + [1.0]; cy = list(cp["U_peak"]) + [ref_U]; o = np.argsort(cx)
    a2.plot(np.array(cx)[o], np.array(cy)[o], marker="o", ms=3, color=C["orange"]); a2.set_xlabel("トルク容量の倍率", fontsize=6.5, labelpad=1); a2.tick_params(labelsize=6)
    for x_, y_ in zip(st["stature"], st["U_peak"]):
        MAC["bodyH" + {1.60: "OneSixty", 1.65: "OneSixtyFive", 1.70: "OneSeventy", 1.80: "OneEighty", 1.85: "OneEightyFive", 1.90: "OneNinety"}[round(x_, 2)]] = f"{y_:.2f}"
    mac("bodyCapLo", cp[cp.cap_scale == 0.9]["U_peak"].iloc[0], 2); mac("bodyCapHi", cp[cp.cap_scale == 1.2]["U_peak"].iloc[0], 2); mac("bodyRefU", ref_U, 2)
    e = rd("exp8", "envelope.csv"); e = e[e["ok"] == 1]; a = ax[3]
    et = e[e["dlnp_tau"] != 0].sort_values("dlnp_tau")
    a.plot(100 * (np.exp(et["dlnp_tau"]) - 1), et["err_none"], marker="o", ms=3.5, color=MUTED, label="補正なし")
    a.plot(100 * (np.exp(et["dlnp_tau"]) - 1), et["err_envelope"], marker="o", ms=3.5, color=C["blue"], label="包絡線定理")
    a.set_xlabel("トルク容量の変化 [%]"); a.set_ylabel("$J^*$ の予測誤差"); a.set_title("(d) 一次転移", loc="left"); a.legend(fontsize=6.3)
    fig.subplots_adjust(wspace=0.55)
    fs.save(fig, FIG, "x8_general")
    sm = et[np.abs(et["dlnp_tau"]) <= 0.055]; lg = et[(np.abs(et["dlnp_tau"]) > 0.055) & (np.abs(et["dlnp_tau"]) < 0.12)]
    mac("envSmallMax", sm["err_envelope"].max(), 3); mac("envSmallNoneMax", sm["err_none"].max(), 2); mac("envTenMax", lg["err_envelope"].max(), 2); mac("envTenNoneMax", lg["err_none"].max(), 2)
    em = e[e["dlnp_mu"] != 0]
    mac("envMuWorse", int((em["err_envelope"] > em["err_none"]).sum())); mac("envMuN", len(em))
    cl = rd("exp8", "closedloop_bodies.csv"); cl = cl[cl["error"].isna()] if "error" in cl else cl
    g = cl.groupby("wrong_condition")[["released", "caught", "succ_2"]].mean()
    mac("newBodyRel", pct(g.loc[0, "released"])); mac("newBodyRelWrong", pct(g.loc[1, "released"])); mac("newBodyCatch", pct(g["caught"].max())); mac("newBodyN", len(cl))


PARTS.update(exp3=exp3, exp4=exp4, exp5=exp5, exp6=exp6, exp7=exp7, exp8=exp8)



# ===================================================================================================== schematics / solution
def _ref():
    r = pickle.load(open(os.path.join(ROOT, "results", "grid", "sol_ref_T18_m66_phi0.250.pkl"), "rb"))
    from katsumi.planar.anthro import make_body
    from katsumi.planar.model import PlanarChain
    return r, PlanarChain(make_body(r["m"]))


def schem():
    """Planar model with every symbol of the notation table, and the ledge cross-section with the grasp cone."""
    from katsumi import device
    r, ch = _ref()
    T, eps = r["T"], r["params"]["eps"]
    dv = device.device_state(0.12 * T, T, eps)
    q = np.concatenate([dv["pA"], [0.12, 0.30, 0.52, 0.86, 0.38]])       # an illustrative swing posture
    P = np.array(ch.f_points(q))                               # 2 x 6: hand, elbow, shoulder, hip, knee, ankle
    fig = plt.figure(figsize=(7.1, 3.5))
    ax = fig.add_axes([0.0, 0.0, 0.64, 1.0]); ax.set_aspect("equal"); ax.axis("off")
    h, xB = dv["h"], dv["x"]; d, hL, fb = device.D_LEDGE, device.LEDGE_HEIGHT, device.FACE_BELOW
    wall = "#c9ced4"
    ax.add_patch(plt.Rectangle((-d - 0.09, h - hL - fb), 0.09, hL + fb + 0.55, color=wall)); ax.add_patch(plt.Rectangle((-d, h - hL), d, hL, color="#6c757d"))
    ax.add_patch(plt.Rectangle((xB + d, -hL - fb), 0.09, hL + fb + 0.55, color=wall)); ax.add_patch(plt.Rectangle((xB, -hL), d, hL, color="#6c757d"))
    ax.plot(P[0], P[1], color=INK, lw=3.0, solid_capstyle="round", zorder=3); ax.scatter(P[0], P[1], s=26, color="white", edgecolor=INK, linewidth=1.2, zorder=4)
    names = ["手（掛かり線）$\\mathbf{p}_h$", "肘：$\\tau_1$", "肩：$\\tau_2$", "股：$\\tau_3$", "膝：$\\tau_4$", "足先"]
    for i, nm in enumerate(names):
        ax.text(P[0, i] + 0.09, P[1, i] + (0.07 if i == 0 else 0.0), nm, fontsize=7.5, ha="left", va="center")
    mid = 0.5 * (P[:, :-1] + P[:, 1:])
    lnames = ["$\\ell_1$：前腕＋手", "$\\ell_2$：上腕", "$\\ell_3$：体幹＋頭", "$\\ell_4$：大腿", "$\\ell_5$：下腿"]
    for i in range(5):
        ax.text(mid[0, i] - (0.08 if i < 4 else 0.24), mid[1, i] - (0.03 if i < 4 else 0.10), lnames[i], fontsize=7, ha="right", va="center", color=INK2)
    # link angle theta_4 measured from the downward vertical (drawn at the hip)
    s_ = P[:, 3]; th4 = q[5]
    ax.plot([s_[0], s_[0]], [s_[1], s_[1] - 0.48], color=MUTED, lw=0.8, ls="--")
    arc = np.linspace(0, th4, 30); ax.plot(s_[0] + 0.33 * np.sin(arc), s_[1] - 0.33 * np.cos(arc), color=C["blue"], lw=1.2)
    ax.text(s_[0] + 0.13, s_[1] - 0.44, "$\\theta_4$", fontsize=8.5, ha="center")
    # hand reaction force
    ax.annotate("", xy=(P[0, 0] - 0.10, P[1, 0] + 0.42), xytext=(P[0, 0], P[1, 0]), arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.6))
    ax.text(P[0, 0] + 0.02, P[1, 0] + 0.36, "突起反力 $\\mathbf{R}=(R_x,R_y)$", fontsize=7.5, ha="left")
    # device quantities
    yt = h + 0.82
    ax.annotate("", xy=(xB, yt), xytext=(0, yt), arrowprops=dict(arrowstyle="<->", color=C["blue"], lw=1.0)); ax.text(xB / 2, yt + 0.05, "先端間距離 $x(t)$", ha="center", fontsize=8)
    ax.plot([0, 0], [h, yt], color=MUTED, lw=0.6, ls=":"); ax.plot([xB, xB], [0, yt], color=MUTED, lw=0.6, ls=":")
    xl = -0.62
    ax.annotate("", xy=(xl, h), xytext=(xl, 0), arrowprops=dict(arrowstyle="<->", color=C["blue"], lw=1.0)); ax.text(xl - 0.04, h / 2, "高低差\n$h(t)$", ha="right", va="center", fontsize=8)
    ax.plot([xl, xB + 0.15], [0, 0], color=MUTED, lw=0.6, ls=":"); ax.plot([xl, 0], [h, h], color=MUTED, lw=0.6, ls=":")
    ax.text(-0.10, h + 0.60, "クリフ A", ha="right", fontsize=8); ax.text(-0.02, h + 0.03, "$\\mathbf{p}_A=(0,h)$", ha="right", va="bottom", fontsize=7.5)
    ax.text(xB + 0.16, 0.60, "クリフ B", ha="left", fontsize=8); ax.text(xB - 0.03, 0.04, "$\\mathbf{p}_B=(x,0)$", ha="right", va="bottom", fontsize=7.5)
    o = np.array([xB - 0.55, -1.15])
    for v, nm in (((0.3, 0), "$x$"), ((0, 0.3), "$y$")):
        ax.annotate("", xy=o + np.array(v), xytext=o, arrowprops=dict(arrowstyle="->", color=INK, lw=1.0)); ax.text(*(o + 1.3 * np.array(v)), nm, fontsize=8.5, ha="center", va="center")
    ax.set_xlim(-1.45, xB + 0.75); ax.set_ylim(-1.40, h + 1.02)
    # ---- ledge cross-section with the cone
    a2 = fig.add_axes([0.655, 0.03, 0.345, 0.90]); a2.set_aspect("equal"); a2.axis("off")
    cm = 100.0
    a2.add_patch(plt.Rectangle((-8, -(hL + fb) * cm), 5, (hL + fb) * cm + 4, color=wall)); a2.add_patch(plt.Rectangle((-3, -hL * cm), 3, hL * cm, color="#6c757d"))
    a2.annotate("", xy=(0, -6.3), xytext=(-3, -6.3), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8)); a2.text(-1.5, -8.2, "3 cm", ha="center", fontsize=7)
    a2.annotate("", xy=(1.3, 0), xytext=(1.3, -5), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8)); a2.text(1.9, -2.5, "5 cm", va="center", fontsize=7)
    a2.annotate("", xy=(-9.2, -5), xytext=(-9.2, -20), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8)); a2.text(-9.9, -12.5, "本体前面 15 cm", va="center", ha="right", fontsize=7, rotation=90)
    a2.text(-5.5, -22.3, "この下は開放空間", ha="center", fontsize=7, color=INK2)
    a2.text(-5.5, -14, "壁", ha="center", fontsize=7.5); a2.text(-1.5, -2.6, "突起", ha="center", fontsize=6.5, color="white")
    mu_o, mu_i, L = 1.0, 2.0, 11.0
    e1 = L * np.array([-mu_o, 1.0]) / np.hypot(mu_o, 1); e2 = L * np.array([mu_i, 1.0]) / np.hypot(mu_i, 1)
    a2.add_patch(plt.Polygon(np.array([[0, 0], e1, e2]), color=C["blue"], alpha=0.18, lw=0, zorder=5))
    a2.plot([0, e1[0]], [0, e1[1]], color=C["blue"], lw=1.0, zorder=6); a2.plot([0, e2[0]], [0, e2[1]], color=C["blue"], lw=1.0, zorder=6)
    a2.text(e1[0] - 0.3, e1[1] + 0.8, "$R_x=-\\mu_{\\mathrm{out}}R_y$", fontsize=7, ha="center"); a2.text(e2[0] - 1.0, e2[1] + 0.8, "$R_x=\\mu_{\\mathrm{in}}R_y$", fontsize=7, ha="center")
    a2.text(1.3, 5.3, "許容される\n反力 $\\mathbf{R}$", fontsize=7, ha="center", zorder=7)
    a2.scatter([0], [0], s=14, color=C["red"], zorder=8); a2.text(0.9, 0.3, "掛かり線", fontsize=6.5, va="bottom")
    a2.set_xlim(-14, 13); a2.set_ylim(-24.5, 12); a2.set_title("突起の断面と摩擦・引っ掛かり錐", fontsize=8)
    fs.save(fig, FIG, "model_schematic")


def sol():
    """Time histories of the reference solution: grip utilisation, joint torques, joint ranges, wall clearance."""
    from katsumi import device
    from katsumi.exp.common import wall_clearance
    r, ch = _ref()
    T, eps = r["T"], r["params"]["eps"]; p = r["params"]
    lo = np.array(p["rel_lo_face_neg"]); hi = np.array(p["rel_hi_face_neg"])
    segs = []
    for ph, t0, dur in (("S", r["t_s0"], r["d_s"]), ("F", r["t_l"], r["d_f"]), ("H", r["t_h0"], p["T_hold"])):
        X = r[ph + "_X"]; n = X.shape[1]; tt = t0 + np.linspace(0, dur, n) - r["t_s0"]
        U = np.hypot(r[ph + "_R"][0], r[ph + "_R"][1]) if ph != "F" else np.zeros(n)
        tau = r[ph + "_U"]
        th = X[:5] if ph != "F" else X[2:7]
        rel = np.vstack([th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]])
        if ph == "H":
            L_, H_ = -hi, -lo
        elif ph == "S":
            L_, H_ = lo, hi
        else:
            L_, H_ = np.minimum(lo, -hi), np.maximum(hi, -lo)
        frac = np.minimum(rel - L_[:, None], H_[:, None] - rel)                 # distance to the nearer joint limit [rad]
        cl = []
        for i in range(n):
            ta = t0 + dur * i / (n - 1); dv = device.device_state(ta, T, eps)
            q = np.concatenate([dv["pA"], th[:, i]]) if ph == "S" else (np.concatenate([dv["pB"], th[:, i]]) if ph == "H" else X[:7, i])
            cl.append(wall_clearance(ch, q, dv))
        segs.append((ph, tt, U, tau, frac, np.array(cl)))
    fig, ax = plt.subplots(4, 1, figsize=(7.1, 4.6), sharex=True)
    shade = {"S": "#ffffff", "F": "#f3ede6", "H": "#eaf1f7"}
    jn = ["肘", "肩", "股", "膝"]
    for ph, tt, U, tau, frac, cl in segs:
        for a in ax:
            a.axvspan(tt[0], tt[-1], color=shade[ph], lw=0)
        ax[0].plot(tt, U, color=C["blue"])
        for j in range(4):
            ax[1].plot(tt, tau[j], color=CAT[j], lw=1.2, label=jn[j] if ph == "S" else None)
            ax[2].plot(tt, np.degrees(frac[j]), color=CAT[j], lw=1.2)
        ax[3].plot(tt, 100 * cl, color=C["blue"])
    ax[0].axhline(r["U_peak"], color=MUTED, lw=0.8, ls="--"); ax[0].text(0.05, r["U_peak"] + 0.04, f"$U_{{\\mathrm{{peak}}}}$={r['U_peak']:.2f}", fontsize=7, color=INK2)
    ax[0].set_ylabel("把持利用率 $U$"); ax[0].set_ylim(0, 1.35)
    ax[1].set_ylabel("関節トルク $\\tau_j/\\tau_j^{\\mathrm{cap}}$"); ax[1].set_ylim(-1.1, 1.1); ax[1].legend(ncol=4, loc="lower left", fontsize=6.5)
    ax[2].set_ylabel("可動域端まで [度]"); ax[2].set_ylim(-2, 60)
    ax[3].set_ylabel("壁との隙間 [cm]"); ax[3].set_ylim(-1, 30); ax[3].set_xlabel("振り開始からの時間 [s]")
    tS, tF, tH = segs[0][1], segs[1][1], segs[2][1]
    for x_, nm in ((0.5 * (tS[0] + tS[-1]), "振り（クリフ A に拘束）"), (0.5 * (tF[0] + tF[-1]), "飛行"), (0.5 * (tH[0] + tH[-1]), "保持（クリフ B に拘束）")):
        ax[0].text(x_, 1.42, nm, ha="center", fontsize=7.5, clip_on=False)
    fig.subplots_adjust(hspace=0.12)
    fs.save(fig, FIG, "solution_timeseries")
    Us = segs[0][2]; Uh = segs[2][2]
    mac("refUswingMax", Us.max(), 2); mac("refUholdMax", Uh.max(), 2); mac("refUpeak", r["U_peak"], 2); mac("refDs", r["d_s"], 2); mac("refDf", r["d_f"], 2)
    mac("refDw", r["d_w"], 1); mac("refImp", r["U_imp"], 2)
    tH_ = segs[2][1] - segs[2][1][0]
    mac("refHoldPeakT", float(tH_[np.argmax(Uh >= 0.999 * Uh.max())]), 2); mac("refHoldPlateau", float((Uh >= 0.995 * r["U_peak"]).mean() * p["T_hold"]), 2)
    mac("refSwingAtPeak", float((Us >= 0.995 * r["U_peak"]).mean() * r["d_s"]), 2)
    mac("refMinClear", 100 * min(seg[5].min() for seg in segs), 1); mac("refTorqueSat", pct(np.mean(np.abs(segs[0][3]) > 0.99)))


PARTS.update(schem=schem, sol=sol)



# ===================================================================================================== 864-case grid
PAT = re.compile(r"sol_ref_T(?P<T>[0-9.]+)_m(?P<m>[0-9.]+)_phi(?P<phi>[0-9.]+)\.pkl$")


def grid_summary(gdir, cache):
    if os.path.exists(cache) and os.path.getmtime(cache) > max(os.path.getmtime(f) for f in glob.glob(os.path.join(gdir, "sol_ref_*.pkl"))):
        return pd.read_csv(cache)
    rows = []
    for f in sorted(glob.glob(os.path.join(gdir, "sol_ref_T*_m*_phi*.pkl"))):
        mm = PAT.search(os.path.basename(f)); r = pickle.load(open(f, "rb"))
        ok = bool(r.get("ok")); g = 9.81
        row = dict(T=float(mm["T"]), m=float(mm["m"]), phi=float(mm["phi"]), ok=int(ok), status=r.get("status"), U=float(r.get("U_peak", np.nan)), effort=float(r.get("effort", np.nan)),
                   d_s=float(r.get("d_s", np.nan)), d_f=float(r.get("d_f", np.nan)), phi_c=float(r.get("phi_c", np.nan)), polished=int("polished_from" in r), solve_s=float(r.get("solve_s", np.nan)))
        try:
            row.update(U_swing=float(np.hypot(r["S_R"][0], r["S_R"][1]).max()), U_hold=float(np.hypot(r["H_R"][0], r["H_R"][1]).max()), U_catch=float(r.get("U_catch", np.nan)), U_imp=float(r.get("U_imp", np.nan)))
        except Exception:
            pass
        row["BW"] = row["U"] * 1300.0 / (row["m"] * g)
        rows.append(row)
    d = pd.DataFrame(rows); d.to_csv(cache, index=False)
    return d


def grid():
    d = grid_summary(os.path.join(ROOT, "results", "grid"), os.path.join(FIG, "grid_final.csv"))
    pre_dir = os.path.join(ROOT, "results", "grid_prepolish")
    pre = grid_summary(pre_dir, os.path.join(FIG, "grid_prepolish.csv")) if os.path.isdir(pre_dir) else None
    d["Uok"] = np.where(d["ok"] == 1, d["U"], np.nan)
    ms = sorted(d["m"].unique()); Ts = sorted(d["T"].unique()); phis = sorted(d["phi"].unique())
    mac("gridN", len(d)); mac("gridNok", int(d["ok"].sum())); mac("gridNT", len(Ts)); mac("gridNm", len(ms)); mac("gridNphi", len(phis))
    # ---- required capacity vs release phase, one panel per body mass
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 3.9), sharex=True, sharey=True)
    for a, m in zip(ax.ravel(), ms):
        g = d[d.m == m]
        a.axvspan(0.5, 1.0, color="#f1efe9", lw=0)
        for T in Ts:
            gg = g[g["T"] == T].sort_values("phi")
            a.plot(gg["phi"], gg["Uok"], color=C["blue"], lw=0.7, alpha=0.45, marker="o", ms=1.8)
        med = g.groupby("phi")["Uok"].min()
        a.plot(med.index, med.values, color=INK, lw=1.4, label="周期についての最小")
        bad = g.groupby("phi")["ok"].sum(); bad = bad[bad == 0]
        a.scatter(bad.index, [1.35] * len(bad), marker="x", s=18, color=fs.BAD, zorder=4, label="全周期で解なし")
        a.axhline(1.0, color=MUTED, lw=0.7, ls="--"); a.axhline(2.0, color=MUTED, lw=0.7, ls="--")
        a.set_yscale("log"); a.set_ylim(0.9, 6.5); a.set_yticks([1, 2, 3, 4, 6]); a.set_yticklabels(["1", "2", "3", "4", "6"]); a.minorticks_off()
        a.set_title(f"$m$ = {m:g} kg", loc="left"); a.set_xlim(-0.03, 0.97); a.set_xticks([0, 0.25, 0.5, 0.75])
    for a in ax[1]:
        a.set_xlabel("離手位相 $\\phi_\\ell$")
    for a in ax[:, 0]:
        a.set_ylabel("$U_{\\mathrm{peak}}$")
    ax[1, 2].legend(fontsize=6.3, loc="center left", bbox_to_anchor=(0.30, 0.42)); ax[0, 2].text(0.75, 1.03, "$F_{\\mathrm{ref}}$", fontsize=6.5, color=INK2, ha="center", va="bottom"); ax[0, 2].text(0.75, 2.06, "$F_{\\max}$", fontsize=6.5, color=INK2, ha="center", va="bottom")
    ax[0, 1].text(0.75, 5.2, "復路", fontsize=7, color=INK2, ha="center"); ax[0, 1].text(0.22, 5.2, "往路", fontsize=7, color=INK2, ha="center")
    fig.subplots_adjust(wspace=0.06, hspace=0.22)
    fs.save(fig, FIG, "grid_capacity")
    # ---- map: best capacity over the period
    best = d.groupby(["m", "phi"])["Uok"].min().unstack()
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.35), gridspec_kw=dict(width_ratios=[1.75, 1.0]))
    a = ax[0]
    im = a.imshow(np.log(best.values), aspect="auto", cmap="Blues", vmin=0.0, vmax=np.log(4.0), origin="lower")
    a.set_xticks(range(len(best.columns))); a.set_xticklabels([f"{p:.2f}".rstrip("0").rstrip(".") if i % 2 == 0 else "" for i, p in enumerate(best.columns)])
    a.set_yticks(range(len(ms))); a.set_yticklabels([f"{m:g}" for m in ms]); a.grid(False)
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            v = best.values[i, j]
            a.text(j, i, "×" if not np.isfinite(v) else f"{v:.2f}"[:4], ha="center", va="center", fontsize=5.6, color="white" if (np.isfinite(v) and v > 2.2) else INK)
    a.set_xlabel("離手位相 $\\phi_\\ell$"); a.set_ylabel("体重 $m$ [kg]"); a.set_title("(a) 必要把持容量 $U_{\\mathrm{peak}}$（周期についての最小，× は解なし）", loc="left")
    a = ax[1]
    feas = d[(d["ok"] == 1) & (d["U"] <= 2.0)].groupby("phi").size().reindex(phis, fill_value=0)
    if pre is not None:
        fp = pre[(pre["ok"] == 1) & (pre["U"] <= 2.0)].groupby("phi").size().reindex(phis, fill_value=0)
        a.bar(np.arange(len(phis)) - 0.2, fp.values, width=0.38, color=fs.NEUTRAL, edgecolor="white", linewidth=0.6, label="連鎖掃引のみ")
        a.bar(np.arange(len(phis)) + 0.2, feas.values, width=0.38, color=C["blue"], edgecolor="white", linewidth=0.6, label="格子近傍からの連続法の後")
        a.legend(fontsize=6.3, loc="upper right")
    else:
        a.bar(np.arange(len(phis)), feas.values, width=0.6, color=C["blue"])
    a.set_xticks(range(0, len(phis), 4)); a.set_xticklabels([f"{phis[i]:.2f}" for i in range(0, len(phis), 4)]); a.set_xlabel("離手位相 $\\phi_\\ell$"); a.set_ylabel("$U_{\\mathrm{peak}}\\leq2$ の条件数（54 中）".replace("\\leq", "\\leq "))
    a.set_title("(b) 連続法の効果", loc="left"); a.grid(axis="x", visible=False); a.set_ylim(0, 62)
    fig.subplots_adjust(wspace=0.22)
    fs.save(fig, FIG, "grid_map")
    # ---- numbers
    okd = d[d["ok"] == 1]
    mac("gridFeasTwo", int((okd["U"] <= 2.0).sum())); mac("gridFeasOne", int((okd["U"] <= 1.5).sum()))
    if pre is not None:
        pm = pre.set_index(["T", "m", "phi"]); dm = d.set_index(["T", "m", "phi"])
        mac("preFeasTwo", int(((pre["ok"] == 1) & (pre["U"] <= 2.0)).sum())); mac("preNok", int(pre["ok"].sum()))
        both = dm.join(pm[["ok", "U"]], rsuffix="_pre")
        imp = both[(both["ok"] == 1) & ((both["ok_pre"] == 0) | (both["U"] < both["U_pre"] - 1e-3))]
        mac("polImproved", len(imp)); mac("polBig", int(((imp["ok_pre"] == 1) & (imp["U_pre"] > 2.0) & (imp["U"] <= 2.0)).sum())); mac("polNewOk", int((imp["ok_pre"] == 0).sum()))
    wide = best.copy()
    feas_phi = [p for p in phis if (best[p] <= 2.0).all()]            # phases feasible (U <= 2) for every body mass
    part_phi = [p for p in phis if (best[p] <= 2.0).any() and not (best[p] <= 2.0).all()]
    none_phi = [p for p in phis if not (best[p] <= 2.0).any()]
    f3 = lambda L: "，".join(f"{p:.3f}".rstrip("0") for p in L) if L else "なし"
    MAC["phiAllFeasible"] = f3(feas_phi); MAC["phiPartFeasible"] = f3(part_phi); MAC["phiNoneFeasible"] = f3(none_phi)
    mac("nPhiAllFeasible", len(feas_phi)); mac("nPhiNone", len(none_phi))
    bm = best.min(axis=1)
    rows = []
    for m in ms:
        row = best.loc[m]; lo = row.min(); win = [p for p in phis if row[p] <= 1.10 * lo]
        out = row[[p for p in phis if p < 0.5]].min(); ret = row[[p for p in phis if p >= 0.5]].min()
        nfe = int((row <= 2.0).sum())
        rows.append([f"{m:g}", f"{lo:.3f}", f"{lo * 1300:.0f}", f"{lo * 1300 / (m * 9.81):.2f}", f"{out:.3f}", f"{ret:.3f}", f"{nfe}/{len(phis)}", f"{min(win):.3f}〜{max(win):.3f}（{len(win)}）"])
        tg = {60.0: "Sixty", 63.0: "SixtyThree", 66.0: "SixtySix", 69.0: "SixtyNine", 72.0: "SeventyTwo", 75.0: "SeventyFive"}[m]
        mac(f"cap{tg}", lo, 2); mac(f"capBW{tg}", lo * 1300 / (m * 9.81), 2); mac(f"capRet{tg}", ret, 2); mac(f"capOut{tg}", out, 2)
    tab("tab_grid", ["$m$ [kg]", "最小 $\\Upk$", "必要容量 [N]", "体重比 [BW]", "往路の最小", "復路の最小", "$\\Upk\\le2$ の位相数", "最小の 1.1 倍以内の位相（個数）"], rows, "rrrrrrrl")
    mac("capBWmin", (bm * 1300 / (bm.index.values * 9.81)).min(), 2); mac("capBWmax", (bm * 1300 / (bm.index.values * 9.81)).max(), 2)
    # dependence on the period at phi = 0.25
    q = okd[okd["phi"] == 0.25].groupby("m")["U"].agg(["min", "median", "max"])
    sp = okd[(okd["phi"] == 0.25) & (okd["U"] < 2)].groupby("m")["U"].agg(lambda x: 100 * (x.max() / x.min() - 1))
    mac("periodSpreadMax", sp.max(), 1); mac("periodSpreadMed", sp.median(), 1)
    # which phase of the motion carries the peak (best solutions with U <= 2)
    lowU = okd[okd["U"] <= 2.0]
    if "U_hold" in lowU:
        mac("holdAtPeak", pct((lowU["U_hold"] >= 0.995 * lowU["U"]).mean())); mac("swingAtPeak", pct((lowU["U_swing"] >= 0.995 * lowU["U"]).mean()))
        mac("impShareMed", pct((lowU["U_imp"] / lowU["U"]).median()))
    mac("flightMin", lowU["d_f"].quantile(0.05), 2); mac("flightMax", lowU["d_f"].quantile(0.95), 2); mac("swingDurMin", lowU["d_s"].quantile(0.05), 1); mac("swingDurMax", lowU["d_s"].quantile(0.95), 1)
    # ---- text of the result paragraph (statements chosen from the data)
    out_ph = [p for p in phis if p < 0.5]; ret_ph = [p for p in phis if p >= 0.5]
    f3 = lambda p: f"{p:.3f}".rstrip("0").rstrip(".") if p > 0 else "0"
    rng_ = lambda L: (f"{f3(min(L))}$〜${f3(max(L))}" if len(L) > 1 else f3(L[0])) if L else "\\text{なし}"
    all_out = [p for p in out_ph if (best[p] <= 2.0).all()]; all_ret = [p for p in ret_ph if (best[p] <= 2.0).all()]
    any_ret = [p for p in ret_ph if (best[p] <= 2.0).any()]
    dead = [p for p in phis if not (best[p] <= 2.0).any()]
    ret_m = [m for m in ms if (best.loc[m, ret_ph] <= 2.0).any()]
    txt = []
    txt.append("\\textbf{実行可能な位相．}見出しの許容容量 $\\Upk\\le2$ で解が得られる離手位相は，往路では $\\phi_\\ell=" + rng_(all_out) + "$ が全体重に共通である"
               + ("．" if len(all_out) else "（共通の位相はない）．"))
    if len(all_ret) == 0 and len(ret_m) < len(ms):
        txt.append("復路で $\\Upk\\le2$ の解が得られたのは $m=$" + "，".join(f"{m:g}" for m in ret_m) + "\\,kg だけで，位相は $\\phi_\\ell=" + rng_(any_ret) + "$ である．")
    elif len(all_ret):
        txt.append("復路でも $\\phi_\\ell=" + rng_(all_ret) + "$ が全体重で実行可能である．")
        txt.append("復路の必要容量は往路とほぼ同じで，体重ごとの最小値の比（復路／往路）は " + f"{(best[ret_ph].min(axis=1) / best[out_ph].min(axis=1)).min():.2f}〜{(best[ret_ph].min(axis=1) / best[out_ph].min(axis=1)).max():.2f}" + " である．"
                   "復路の位相 $1-\\phi$ では，装置は往路の位相 $\\phi$ と同じ位置にあり，動く向きだけが逆である．")
    else:
        txt.append("復路では $\\phi_\\ell=" + rng_(any_ret) + "$ で一部の体重に解がある．")
    if dead:
        txt.append("$\\phi_\\ell=" + rng_(dead) + "$ では，どの体重でも $\\Upk\\le2$ の解は得られなかった．B が最も遠い $\\phi=0.5$ の前後で，先端間距離は 2.6〜2.7\\,m になり，"
                   "A は最も低い位置にある．片側からの振りでは，この距離を越える離手の速度と高さが得られない．")
    txt.append("\n\\textbf{体重と装置周期．}必要把持容量の最小値は，$m=60$\\,kg の \\capSixty から $m=75$\\,kg の \\capSeventyFive へ体重とともに増える"
               "（体重比では \\capBWmin〜\\capBWmax\\,BW）．関節トルクの上限を体重によらず固定しているので，重い体ほど振りに使える余力が小さく，体重比でも要求が増える．"
               "装置周期への依存は小さい．$\\phi_\\ell=0.25$ では，9 通りの周期の間の $\\Upk$ の違いは最大 \\periodSpreadMax\\,\\%（中央値 \\periodSpreadMed\\,\\%）である．"
               "装置の速さは 0.08〜0.11\\,m/s で，離手のときの体の速さ（数 m/s）に比べて小さい．飛行時間は \\flightMin〜\\flightMax\\,s で，その間に装置が動く距離は 5\\,cm 以下である．"
               "効くのは離手の瞬間の装置の位置であって，速さではない．")
    txt.append("\n\\textbf{最大負荷が生じる相．}$\\Upk\\le2$ で解けた \\gridFeasTwo 条件のうち，保持の最大負荷が $\\Upk$ に達しているものは \\holdAtPeak\\,\\%，"
               "振りの最大負荷が $\\Upk$ に達しているものは \\swingAtPeak\\,\\%である．多くの解で両方が同時に上限に触れている．"
               "どちらが必要容量を決めているかは，到達しているかどうかでは分からない．これを第~\\ref{sec:interventions}節で調べる．")
    open(os.path.join(PAPER, "grid_paragraph.tex"), "w").write("\n".join(txt) + "\n")
    st = d["status"].value_counts()
    mac("gridInfeasible", int(st.get("Infeasible_Problem_Detected", 0))); mac("gridTimeout", int(st.get("Maximum_CpuTime_Exceeded", 0) + st.get("Maximum_Iterations_Exceeded", 0) + st.get("Restoration_Failed", 0)))


def data():
    d = pd.read_csv(os.path.join(S, "dual_field_rows_frozen.csv")); ok = d[d["ok"] == 1]
    mac("dsNsolve", len(d)); mac("dsNok", len(ok)); mac("dsNref", ok["ref"].nunique())
    tot = ok[["cap_cap_H", "cap_cap_S", "cap_cap_catch"]].sum(axis=1)
    mac("shareHold", pct((ok["cap_cap_H"] / tot).mean())); mac("shareCatch", pct((ok["cap_cap_catch"] / tot).mean())); mac("shareSwing", pct((ok["cap_cap_S"] / tot).mean(), 1))
    el = ok["sens_tau_cap"] / ok["J"]
    mac("elastTauMed", el.median(), 2); mac("elastTauMean", el.mean(), 2); mac("elastMuOutMed", (ok["sens_mu_out"] / ok["J"]).median(), 3)
    mac("priceSpeedMax", max(ok["sens_qd_max"].abs().max(), ok["sens_v_rel_max"].abs().max(), ok["sens_v_away_max"].abs().max()), 3)
    names = {"sens_tau_cap": "関節トルク容量 $\\tau^{\\rm cap}$", "sens_mu_out": "引っ掛かり係数 $\\mu_{\\rm out}$", "sens_mu_in": "摩擦係数 $\\mu_{\\rm in}$", "sens_qd_max": "関節角速度上限 $\\dot q_{\\max}$",
             "sens_v_rel_max": "捕捉相対速度上限 $v_{\\rm rel}^{\\max}$", "sens_v_away_max": "壁から離れる速度上限 $v_{\\rm away}^{\\max}$"}
    rows = [[names[c], f"{ok[c].median():.3f}", f"{ok[c].mean():.3f}", f"{ok[c].quantile(0.1):.3f}", f"{(ok[c] / ok['J']).median():.3f}", pct((ok[c].abs() > 1e-6).mean())] for c in names]
    tab("tab_prices2", ["パラメータ $p$", "$\\partial J^*/\\partial\\ln p$ 中央値", "平均", "10\\%点", "弾性（中央値）", "拘束的な割合 [\\%]"], rows)
    lam = [c for c in ok.columns if c.startswith("lam_")]
    sh = ok[lam].abs().mean().sort_values(ascending=False)
    MAC["lamTop"] = "，".join(f"{k[4:]}" for k in sh.index[:3])


PARTS.update(grid=grid, data=data)



def dev():
    """Device motion: tip distance and height difference over one period, and the device speeds."""
    from katsumi import device
    T, eps = 20.0, 0.2
    ph = np.linspace(0, 1, 801); st = [device.device_state(p * T, T, eps) for p in ph]
    x = np.array([s_["x"] for s_ in st]); h = np.array([s_["h"] for s_ in st]); vx = np.array([s_["vB"][0] for s_ in st]); vh = np.array([s_["vA"][1] for s_ in st])
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.75))
    for a in ax:
        a.axvspan(0.5, 1.0, color="#f1efe9", lw=0); a.set_xlabel("装置位相 $\\phi$"); a.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); a.set_xticklabels(["0", "", "0.5", "", "1"])
    ax[0].plot(ph, x, color=C["blue"], label="$x(t)$"); ax[0].plot(ph, h, color=C["orange"], label="$h(t)$"); ax[0].set_ylabel("[m]"); ax[0].set_ylim(-0.1, 3.3)
    ax[0].legend(loc="upper center", ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=0.8); ax[0].text(0.25, 1.25, "往路", ha="center", fontsize=6.5, color=INK2); ax[0].text(0.75, 1.25, "復路", ha="center", fontsize=6.5, color=INK2)
    tri = np.where(ph < 0.5, 0.9 / (T / 2), -0.9 / (T / 2))
    ax[1].plot(ph, tri, color=MUTED, lw=1.0, ls="--", label="三角波"); ax[1].plot(ph, vx, color=C["blue"], label="$\\dot x$"); ax[1].plot(ph, vh, color=C["orange"], label="$\\dot h$")
    ax[1].set_ylabel("[m/s]"); ax[1].set_ylim(-0.13, 0.19); ax[1].legend(loc="upper center", ncol=3, fontsize=6.3, handlelength=1.0, columnspacing=0.6)
    fig.subplots_adjust(wspace=0.55)
    fs.save(fig, FIG, "device_motion")


PARTS.update(dev=dev)


def write_macros():
    with open(os.path.join(PAPER, "numbers_final.tex"), "w") as f:
        for k in sorted(MAC):
            f.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{MAC[k]}}}\n")
    print(f"{len(MAC)} macros -> paper/numbers_final.tex")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="all")
    a = ap.parse_args()
    fs.setup(); os.makedirs(FIG, exist_ok=True)
    parts = list(PARTS) if a.parts == "all" else a.parts.split(",")
    prev = os.path.join(PAPER, "numbers_final.json")
    if os.path.exists(prev) and a.parts != "all":
        MAC.update(json.load(open(prev)))
    for p in parts:
        PARTS[p]()
    json.dump(MAC, open(prev, "w"), indent=0, ensure_ascii=False)
    write_macros()


if __name__ == "__main__":
    main()
