"""Figures and tables from the Stage-1 release-phase grid.

usage: python scripts/analyze_grid.py --grid results/grid --tag ref --out results/figs
Works on partial results (reads rows_<tag>_T*_m*.csv).
"""
import argparse, glob, os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi import device

# palette (dataviz reference instance): categorical slots 1-3, single-hue sequential, diverging pair
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#e8f0fb", "#9dc1ee", "#2a78d6", "#0f3f7a"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f2f0", "#eb6834"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

G = 9.81


def load(grid, tag):
    files = sorted(glob.glob(os.path.join(grid, f"rows_{tag}_T*_m*.csv")))
    if not files:
        raise SystemExit("no rows files")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["phi_l"] = df["phi_l"].round(4)
    df = df.sort_values("ok", ascending=False).drop_duplicates(["T", "m", "phi_l"], keep="first")
    for c in ["T", "m", "phi_l", "U_peak", "req_cap_N", "req_cap_BW", "effort", "d_s", "d_f", "phi_c", "x_catch",
              "h_release", "v0x", "v0y", "vrel_norm", "impulse_norm", "U_catch", "U_swing", "U_hold",
              "res_S_max", "res_F_max", "res_H_max", "ver_S_pos", "ver_F_hand", "ver_H_pos"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ok"] = df["ok"].astype(int)
    df["outbound"] = df["phi_l"] < 0.5
    return df


def fig_reqcap_vs_phase(df, out):
    Ts = sorted(df["T"].unique())
    ms = sorted(df["m"].unique())
    fig, axes = plt.subplots(1, len(Ts), figsize=(3.4 * len(Ts), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, T in zip(axes, Ts):
        for j, m in enumerate(ms):
            d = df[(df["T"] == T) & (df["m"] == m)].sort_values("phi_l")
            ok = d["ok"] == 1
            ax.plot(d["phi_l"][ok], d["req_cap_BW"][ok], "-o", color=CAT[j % 3], ms=3.5, lw=1.6, label=f"m = {m:g} kg")
            if (~ok).any():
                ax.plot(d["phi_l"][~ok], np.full((~ok).sum(), ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 3.0), "x", color=CAT[j % 3], ms=5)
        ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
        ax.text(0.25, 0.97, "outbound (A↓, B→far)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=INK2)
        ax.text(0.75, 0.97, "return (A↑, B→near)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=INK2)
        ax.set_title(f"T = {T:g} s (one-way {T/2:g} s)", fontsize=10)
        ax.set_xlabel("release phase φ_ℓ = (t_ℓ mod T)/T")
        ax.set_xlim(-0.025, 1.0)
    axes[0].set_ylabel("required two-hand grip capacity [BW]")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reqcap_vs_phase.png"), dpi=200)
    fig.savefig(os.path.join(out, "reqcap_vs_phase.pdf"))
    plt.close(fig)


def fig_heatmaps(df, out, m_sel=None):
    ms = sorted(df["m"].unique())
    m_sel = m_sel or ms[len(ms) // 2]
    d = df[df["m"] == m_sel]
    Ts = sorted(d["T"].unique())
    phis = sorted(d["phi_l"].unique())
    quantities = [("req_cap_BW", "required capacity [BW]"), ("effort", "E_eff [-]"), ("d_f", "flight time τ [s]"),
                  ("x_catch", "tip distance at catch x(t_c) [m]")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, (q, lab) in zip(axes, quantities):
        M = np.full((len(Ts), len(phis)), np.nan)
        for i, T in enumerate(Ts):
            for j, ph in enumerate(phis):
                r = d[(d["T"] == T) & (d["phi_l"] == ph) & (d["ok"] == 1)]
                if len(r):
                    M[i, j] = r[q].values[0]
        dphi = 0.5 * (phis[1] - phis[0]) if len(phis) > 1 else 0.03
        im = ax.imshow(M, aspect="auto", cmap=SEQ, origin="lower",
                       extent=[phis[0] - dphi, phis[-1] + dphi, -0.5, len(Ts) - 0.5])
        ax.set_yticks(range(len(Ts)))
        ax.set_yticklabels([f"{T:g}" for T in Ts])
        ax.set_xlabel("release phase φ_ℓ")
        ax.set_title(lab, fontsize=9.5)
        ax.grid(False)
        for i in range(len(Ts)):
            for j in range(len(phis)):
                if np.isnan(M[i, j]):
                    ax.text(phis[j], i, "×", ha="center", va="center", color=INK2, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("T [s]")
    fig.suptitle(f"m = {m_sel:g} kg   (× = no feasible solution found)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"heatmaps_m{m_sel:g}.png"), dpi=200)
    plt.close(fig)


def fig_tm_maps(df, out):
    """T x m maps of the minimum-over-phase quantities (plan Sec. 15.1: T x m heat maps)."""
    ok = df[df["ok"] == 1]
    if ok["T"].nunique() < 2 or ok["m"].nunique() < 2:
        return None
    rows = []
    for (T, m), d in ok.groupby(["T", "m"]):
        b = d.loc[d["req_cap_BW"].idxmin()]
        rows.append(dict(T=T, m=m, cap=b["req_cap_BW"], capN=b["req_cap_N"], phi=b["phi_l"], tau=b["d_f"], E=b["effort"],
                         n_ok=len(d), out_cap=d[d["outbound"]]["req_cap_BW"].min() if d["outbound"].any() else np.nan,
                         ret_cap=d[~d["outbound"]]["req_cap_BW"].min() if (~d["outbound"]).any() else np.nan))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out, "tm_map.csv"), index=False)
    Ts = sorted(s["T"].unique()); ms = sorted(s["m"].unique())
    quantities = [("cap", "min required capacity [BW]", SEQ), ("phi", "best release phase φ_ℓ", SEQ),
                  ("tau", "flight time τ at best phase [s]", SEQ), ("E", "E_eff at best phase [-]", SEQ)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (q, lab, cmap) in zip(axes, quantities):
        M = np.full((len(ms), len(Ts)), np.nan)
        for r in s.itertuples():
            M[ms.index(r.m), Ts.index(r.T)] = getattr(r, q)
        im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(Ts))); ax.set_xticklabels([f"{T:g}" for T in Ts], fontsize=7.5)
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([f"{m:g}" for m in ms], fontsize=7.5)
        ax.set_xlabel("period T [s]  (one-way T/2)"); ax.set_title(lab, fontsize=9.5); ax.grid(False)
        if len(Ts) * len(ms) <= 60:
            for i in range(len(ms)):
                for j in range(len(Ts)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("body mass m [kg]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tm_maps.png"), dpi=200)
    fig.savefig(os.path.join(out, "tm_maps.pdf"))
    plt.close(fig)
    # outbound - return difference map (diverging)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    M = np.full((len(ms), len(Ts)), np.nan)
    for r in s.itertuples():
        M[ms.index(r.m), Ts.index(r.T)] = 100 * (r.ret_cap - r.out_cap) / r.out_cap
    v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    im = ax.imshow(M, aspect="auto", cmap=DIV, origin="lower", vmin=-v, vmax=v)
    ax.set_xticks(range(len(Ts))); ax.set_xticklabels([f"{T:g}" for T in Ts], fontsize=7.5)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([f"{m:g}" for m in ms], fontsize=7.5)
    ax.set_xlabel("period T [s]"); ax.set_ylabel("m [kg]"); ax.grid(False)
    ax.set_title("H1: (best return − best outbound) / best outbound [%]", fontsize=8.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tm_h1_diff.png"), dpi=200)
    plt.close(fig)
    return s


def fig_h1_h2(df, out):
    """H1: outbound vs return best capacity; H2: shortest-distance phase vs minimum-load phase."""
    rows = []
    for (T, m), d in df[df["ok"] == 1].groupby(["T", "m"]):
        o = d[d["outbound"]]
        r = d[~d["outbound"]]
        best_o = o.loc[o["req_cap_BW"].idxmin()] if len(o) else None
        best_r = r.loc[r["req_cap_BW"].idxmin()] if len(r) else None
        best = d.loc[d["req_cap_BW"].idxmin()]
        # shortest tip-to-tip distance at release (x = 1.8 at phi = 0 / 1)
        d = d.assign(x_release=[device.device_state(ph * T, T, 0.2)["x"] for ph in d["phi_l"]])
        nearest = d.loc[d["x_release"].idxmin()]
        rows.append(dict(T=T, m=m,
                         best_phi=best["phi_l"], best_cap=best["req_cap_BW"], best_dir="outbound" if best["outbound"] else "return",
                         out_cap=None if best_o is None else best_o["req_cap_BW"], out_phi=None if best_o is None else best_o["phi_l"],
                         ret_cap=None if best_r is None else best_r["req_cap_BW"], ret_phi=None if best_r is None else best_r["phi_l"],
                         nearest_phi=nearest["phi_l"], nearest_cap=nearest["req_cap_BW"], nearest_x=nearest["x_release"],
                         best_x=float(device.device_state(best["phi_l"] * T, T, 0.2)["x"]),
                         best_E=best["effort"], best_tau=best["d_f"], best_v0x=best["v0x"], best_v0y=best["v0y"]))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out, "summary_h1_h2.csv"), index=False)
    if len(s) == 0:
        return s
    s = s.astype({c: float for c in ["out_cap", "ret_cap", "out_phi", "ret_phi"]})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    ax = axes[0]
    x = np.arange(len(s))
    ax.bar(x - 0.18, s["out_cap"].fillna(0), width=0.34, color=CAT[0], label="best outbound release")
    ax.bar(x + 0.18, s["ret_cap"].fillna(0), width=0.34, color=CAT[1], label="best return release")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{r.T:g}\nm{r.m:g}" for r in s.itertuples()], fontsize=7)
    ax.set_ylabel("min required capacity [BW]")
    ax.set_title("H1: outbound vs return", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_ylim(0, max(1.05 * np.nanmax(s[["out_cap", "ret_cap"]].values), 1))
    ax = axes[1]
    ax.scatter(s["nearest_x"], s["nearest_cap"], color=CAT[1], s=28, label="release at shortest distance")
    ax.scatter(s["best_x"], s["best_cap"], color=CAT[0], s=28, label="release at minimum load")
    for r in s.itertuples():
        ax.plot([r.nearest_x, r.best_x], [r.nearest_cap, r.best_cap], color=GRID, lw=1, zorder=0)
    ax.set_xlabel("tip-to-tip distance at release [m]")
    ax.set_ylabel("required capacity [BW]")
    ax.set_title("H2: shortest distance ≠ minimum load", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "h1_h2.png"), dpi=200)
    fig.savefig(os.path.join(out, "h1_h2.pdf"))
    plt.close(fig)
    return s


def fig_phi0_strategy(df, out, cap_factor=1.25, T_sel=20.0, m_sel=60.0):
    """Given a start phase phi0 and an athlete whose capacity is cap_factor x 1300 N, which release phase
    minimises effort incl. waiting?  E_total = E_eff(phi_l) + U_wait^2 * wait, wait = ((phi_l - phi0 - d_s/T) mod 1) * T."""
    d = df[(df["T"] == T_sel) & (df["m"] == m_sel) & (df["ok"] == 1)].sort_values("phi_l")
    if len(d) == 0:
        return
    f_cap = 1300.0 * cap_factor
    U_wait = m_sel * G / f_cap
    feas = d[d["U_peak"] * 1300.0 <= f_cap]
    phi0s = np.linspace(0, 1, 33)
    best_phi, best_E, waits = [], [], []
    for p0 in phi0s:
        cands = []
        for r in feas.itertuples():
            swing_phase = r.d_s / T_sel
            wait = ((r.phi_l - swing_phase - p0) % 1.0) * T_sel
            cands.append((r.effort + U_wait ** 2 * wait, r.phi_l, wait))
        if cands:
            e, ph, w = min(cands)
            best_phi.append(ph); best_E.append(e); waits.append(w)
        else:
            best_phi.append(np.nan); best_E.append(np.nan); waits.append(np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(d["phi_l"], d["req_cap_BW"], "-o", color=CAT[0], ms=3)
    axes[0].axhline(f_cap / (m_sel * G), color=CAT[1], lw=1.2, ls="--")
    axes[0].text(0.02, f_cap / (m_sel * G) + 0.02, f"athlete capacity {cap_factor:g}×1300 N", color=CAT[1], fontsize=7.5)
    axes[0].set_xlabel("release phase φ_ℓ"); axes[0].set_ylabel("required capacity [BW]")
    axes[1].plot(phi0s, best_phi, "-", color=CAT[0]); axes[1].set_xlabel("start phase φ₀"); axes[1].set_ylabel("chosen release phase φ_ℓ")
    axes[2].plot(phi0s, best_E, "-", color=CAT[0], label="E_eff incl. waiting"); axes[2].plot(phi0s, waits, "-", color=CAT[1], label="wait [s]")
    axes[2].set_xlabel("start phase φ₀"); axes[2].legend(fontsize=7.5)
    fig.suptitle(f"T = {T_sel:g} s, m = {m_sel:g} kg: strategy vs start phase", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"phi0_strategy_T{T_sel:g}_m{m_sel:g}.png"), dpi=200)
    plt.close(fig)


def audit_table(df, out):
    a = df[df["ok"] == 1]
    rep = dict(n_cases=int(len(df)), n_ok=int(a.shape[0]),
               hs_residual_max=dict(S=float(a["res_S_max"].max()), F=float(a["res_F_max"].max()), H=float(a["res_H_max"].max())),
               hs_residual_p95_median=dict(S=float(a["res_S_p95"].median()), F=float(a["res_F_p95"].median()), H=float(a["res_H_p95"].median())),
               reintegration_err_max=dict(S_pos_rad=float(a["ver_S_pos"].max()), F_hand_m=float(a["ver_F_hand"].max()), H_pos_rad=float(a["ver_H_pos"].max())),
               solve_time_s=dict(median=float(pd.to_numeric(df["solve_s"], errors="coerce").median()),
                                 max=float(pd.to_numeric(df["solve_s"], errors="coerce").max())),
               iters_median=float(pd.to_numeric(df["iters"], errors="coerce").median()))
    json.dump(rep, open(os.path.join(out, "audit.json"), "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.grid, a.tag)
    df.to_csv(os.path.join(a.out, f"grid_{a.tag}_all.csv"), index=False)
    print(df.groupby(["T", "m"])["ok"].agg(["count", "sum"]))
    fig_reqcap_vs_phase(df, a.out)
    for m in sorted(df["m"].unique()):
        fig_heatmaps(df, a.out, m_sel=m)
    s = fig_h1_h2(df, a.out)
    print(s)
    tm = fig_tm_maps(df, a.out)
    if tm is not None:
        print(tm.round(3).to_string(index=False))
    for T in sorted(df["T"].unique()):
        for m in sorted(df["m"].unique()):
            fig_phi0_strategy(df, a.out, T_sel=T, m_sel=m)
    print(json.dumps(audit_table(df, a.out), indent=1))


if __name__ == "__main__":
    main()
