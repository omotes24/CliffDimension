"""Release-phase vs required capacity for the v3 model (from solution pickles; grid in progress)."""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#52514e"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def main(grid="results/v3/results/grid", out="results/figs"):
    rows = []
    for f in sorted(glob.glob(os.path.join(grid, "sol_ref_*.pkl"))):
        r = pickle.load(open(f, "rb"))
        rows.append((r["T"], r["m"], round(r["phi_l"], 4), bool(r.get("ok")), r["U_peak"], r.get("catch_gap", np.nan)))
    Ts = sorted(set(r[0] for r in rows)); ms = sorted(set(r[1] for r in rows))
    fig, axes = plt.subplots(1, len(Ts), figsize=(2.6 * len(Ts), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, T in zip(axes, Ts):
        ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
        ax.axhspan(2.0, 6.5, color="#fff3ee", zorder=0)
        for i, m in enumerate(ms):
            sel = sorted([r for r in rows if r[0] == T and r[1] == m], key=lambda r: r[2])
            ok = [r for r in sel if r[3] and r[4] < 2.0]
            sup = [r for r in sel if r[3] and r[4] >= 2.0]
            bad = [r for r in sel if not r[3]]
            if ok:
                ax.plot([r[2] for r in ok], [r[4] for r in ok], "-o", ms=3, lw=1.1, color=CAT[i % len(CAT)], label=f"{m:g} kg")
            if sup:
                ax.plot([r[2] for r in sup], [r[4] for r in sup], "^", ms=4, color=CAT[i % len(CAT)], alpha=0.8)
            if bad:
                ax.plot([r[2] for r in bad], [6.0] * len(bad), "x", ms=5, color=CAT[i % len(CAT)], alpha=0.8)
        ax.set_title(f"T = {T:g} s", fontsize=9); ax.set_xlabel(r"release phase $\phi_\ell$"); ax.set_xlim(0, 1); ax.set_ylim(0.8, 6.4)
    axes[0].set_ylabel(r"required two-hand capacity $U_{\rm peak}$ [$F_{\rm cap}$ = 1300 N]")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    axes[-1].text(0.98, 6.1, "x: infeasible (unreachable)", fontsize=7, ha="right", color=INK2)
    axes[-1].text(0.98, 2.15, "triangles: > 2 (superhuman)", fontsize=7, ha="right", color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reqcap_vs_phase.pdf")); fig.savefig(os.path.join(out, "reqcap_vs_phase.png"), dpi=200)
    print("written", len(rows), "cases")


if __name__ == "__main__":
    main(*sys.argv[1:])
