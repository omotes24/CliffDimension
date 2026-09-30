"""Dual-variable report: WHEN the grip capacity binds and WHAT capability has value.

Reads results/duals/dual_*.pkl (run_duals.py). Produces
  * cap_mult stacked bars over phi: fraction of the capacity multiplier mass in swing / catch / hold
    (the multipliers of U <= U_peak sum to w_U by stationarity; their distribution = equioscillation weights),
  * shadow prices dU*/d ln p = (dJ*/d ln p) / w_U for tau_cap (elbow, shoulder, hip, knee), mu_out, v_rel_max, qd_max,
  * FD validation table from results/duals_fd (if present),
  * multiplier mass per constraint category (what limits the motion besides the grip).

usage: python scripts/analyze_duals.py --duals results/duals --fd results/duals_fd --out results/figs
"""
import argparse, glob, json, os, pickle, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#52514e", "#d63a6a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
PNAMES = {"tau_cap[0]": "elbow torque", "tau_cap[1]": "shoulder torque", "tau_cap[2]": "hip torque", "tau_cap[3]": "knee torque",
          "mu_out": r"$\mu_{out}$ (hook)", "mu_in": r"$\mu_{in}$", "v_rel_max": r"$|v_{rel}|$ bound", "v_away_max": r"$v_{away}$ bound",
          "qd_max": "joint speed bound"}


def load(duals_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(duals_dir, "dual_*.pkl"))):
        r = pickle.load(open(f, "rb"))
        d = r.get("duals")
        if not r.get("ok") or not d or "cap_mult" not in d:
            continue
        cm = {k: float(np.nansum(v)) for k, v in d["cap_mult"].items()}
        tot = sum(cm.values())
        row = dict(T=r["T"], m=r["m"], phi=round(r["phi_l"], 4), U_peak=r["U_peak"], cap_total=tot,
                   f_swing=(cm.get("cap_S", 0) + cm.get("cap_wait", 0)) / tot, f_catch=cm.get("cap_catch", 0) / tot,
                   f_hold=cm.get("cap_H", 0) / tot, f_C=cm.get("cap_C", 0) / tot)
        for name, vals in d["dJ_dlnp"].items():
            vals = np.ravel(vals)
            if len(vals) > 1:
                for i, v in enumerate(vals):
                    row[f"s_{name}[{i}]"] = float(v) / d["w_U"]
            else:
                row[f"s_{name}"] = float(vals[0]) / d["w_U"]
        # multiplier mass per category (inequalities only)
        for cat, c in d["categories"].items():
            row[f"lam_{cat}"] = c["abs_sum"]
        # time-resolved hold multipliers (first 0.5 s vs rest)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duals", default="results/duals")
    ap.add_argument("--fd", default="results/duals_fd")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    df = load(a.duals)
    if not len(df):
        print("no duals"); return
    df = df.sort_values(["m", "phi"])
    df.to_csv(os.path.join(a.out, "duals.csv"), index=False)
    ms = sorted(df["m"].unique())
    # ---- figure: (a) where the capacity binds, (b) shadow prices ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ax = axes[0]
    d = df[df["m"] == ms[-1]]
    w = 1.0 / 16 * 0.85
    bottom = np.zeros(len(d))
    for key, lab, col in (("f_swing", "swing", CAT[0]), ("f_catch", "catch pulse", CAT[1]), ("f_hold", "hold (post-catch swing)", CAT[2])):
        ax.bar(d["phi"], d[key], width=w, bottom=bottom, color=col, label=lab)
        bottom += d[key].values
    ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
    ax.set_xlabel(r"release phase $\phi_\ell$"); ax.set_ylabel("share of capacity multiplier mass")
    ax.set_title(f"where $U = U_{{peak}}$ binds (T = 20 s, m = {ms[-1]:g} kg)", fontsize=8.5); ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1)
    ax = axes[1]
    keys = [("s_tau_cap[1]", "shoulder torque"), ("s_tau_cap[0]", "elbow torque"), ("s_mu_out", r"$\mu_{out}$ (hook)"),
            ("s_tau_cap[2]", "hip torque"), ("s_tau_cap[3]", "knee torque"), ("s_v_rel_max", r"$|v_{rel}|$ bound"), ("s_qd_max", "joint speed")]
    for i, (k, lab) in enumerate(keys):
        if k in d:
            ax.plot(d["phi"], -100 * d[k], "-o", ms=2.5, lw=1.2, color=CAT[i % len(CAT)], label=lab)
    ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
    ax.set_xlabel(r"release phase $\phi_\ell$"); ax.set_ylabel(r"$-\partial U^*/\partial \ln p$ [% of $F_{cap}$ per +100 %]")
    ax.set_title("shadow price of each capability", fontsize=8.5); ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "duals.pdf")); fig.savefig(os.path.join(a.out, "duals.png"), dpi=200)
    plt.close(fig)
    # ---- summary numbers --------------------------------------------------------------------------------
    summ = {}
    for m in ms:
        d = df[df["m"] == m]
        summ[f"m{m:g}"] = dict(n=int(len(d)), f_swing_mean=float(d["f_swing"].mean()), f_catch_mean=float(d["f_catch"].mean()),
                              f_hold_mean=float(d["f_hold"].mean()),
                              shoulder_pct=float(-100 * d["s_tau_cap[1]"].mean()), elbow_pct=float(-100 * d["s_tau_cap[0]"].mean()),
                              hip_pct=float(-100 * d["s_tau_cap[2]"].mean()), knee_pct=float(-100 * d["s_tau_cap[3]"].mean()),
                              mu_out_pct=float(-100 * d["s_mu_out"].mean()), vrel_pct=float(-100 * d["s_v_rel_max"].mean()),
                              qd_pct=float(-100 * d["s_qd_max"].mean()))
    # FD validation
    fd_rows = []
    for f in glob.glob(os.path.join(a.fd, "duals_*.json")):
        for r in json.load(open(f)):
            for k, v in (r.get("fd") or {}).items():
                fd_rows.append(dict(phi=round(r["phi_l"], 4), param=k, dual=v["dJ_dlnp_dual"], fd=v["dJ_dlnp"], dU_fd=v["dU_dlnp"], ok=v["ok"]))
    if fd_rows:
        fd = pd.DataFrame(fd_rows)
        fd.to_csv(os.path.join(a.out, "duals_fd.csv"), index=False)
        summ["fd"] = fd.to_dict("records")
        with open(os.path.join("paper", "tab_duals_fd.tex"), "w") as fh:
            fh.write("\\begin{tabular}{llrrr}\n\\toprule\n$\\phi_\\ell$ & パラメータ & $\\partial J^*/\\partial\\ln p$（双対） & 同（差分） & $\\partial U^*/\\partial\\ln p$（差分） \\\\\n\\midrule\n")
            for r in fd.itertuples():
                fh.write(f"{r.phi:.3f} & {PNAMES.get(r.param, r.param)} & {r.dual:+.2f} & {r.fd:+.2f} & {r.dU_fd:+.3f} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")
    json.dump(summ, open(os.path.join(a.out, "duals_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "fd"}, indent=1))
    if fd_rows:
        print(pd.DataFrame(fd_rows).round(3).to_string(index=False))
    # multiplier mass per category, averaged (inequalities)
    lam_cols = [c for c in df.columns if c.startswith("lam_") and c not in ("lam_dyn", "lam_link", "lam_init", "lam_pin", "lam_impact")]
    print(df[lam_cols].mean().sort_values(ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
