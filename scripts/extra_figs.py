"""Additional manuscript figures: device motion, release-window outcome, sensitivity bars, 3-D search history,
body-size sweep.

usage: python scripts/extra_figs.py --out results/figs
"""
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi import device

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def fig_device(out, T=20.0, eps=0.2):
    t = np.linspace(0, T, 2001)
    s = device.device_state(t, T, eps)
    s3 = device.device_state(t, T, eps, smooth=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    ax.plot(t / T, s["x"], color=CAT[0], lw=1.8, label="x(t): tip-to-tip distance")
    ax.plot(t / T, s["h"], color=CAT[1], lw=1.8, label="h(t): height of A above B")
    ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
    ax.set_xlabel("device phase φ = (t mod T)/T"); ax.set_ylabel("[m]"); ax.legend(fontsize=7.5, loc="center right")
    ax.text(0.25, 2.62, "outbound", ha="center", fontsize=7.5, color=INK2); ax.text(0.75, 2.62, "return", ha="center", fontsize=7.5, color=INK2)
    ax.set_ylim(-0.1, 2.9)
    ax = axes[1]
    ax.plot(t / T, s["xd"], color=CAT[0], lw=1.8, label="ẋ (smoothed, ε = 0.2 s)")
    ax.plot(t / T, s3["xd"], color=CAT[0], lw=1.0, ls="--", label="ẋ (triangle wave)")
    ax.plot(t / T, s["hd"], color=CAT[1], lw=1.8, label="ḣ (smoothed)")
    ax.axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
    ax.set_xlabel("device phase φ"); ax.set_ylabel("[m/s]"); ax.legend(fontsize=6.5, loc="center", ncol=1)
    ax.set_ylim(-0.13, 0.13)
    fig.suptitle(f"device motion for T = {T:g} s (one-way {T/2:g} s)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "device_motion.pdf")); fig.savefig(os.path.join(out, "device_motion.png"), dpi=200)
    plt.close(fig)


def fig_windows(out, path="results/windows/windows_ref.json"):
    if not os.path.exists(path):
        return
    W = json.load(open(path))
    n = len(W)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.3 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {"held": CAT[2], "cone exceeded": CAT[3], "lost hook": CAT[1], "hit wall": INK2, "missed": "#c9c9c4"}
    for ax, w in zip(axes, W):
        det = w["plan_detail"]
        ds = sorted(float(k) for k in det)
        for d in ds:
            v = det[str(d) if str(d) in det else f"{d}"]
            r = v["reason"]
            key = "held" if r == "held" else ("cone exceeded" if r.startswith("cone") else ("lost hook" if r.startswith("lost")
                  else ("hit wall" if r.startswith("hit") else "missed")))
            ax.plot([1e3 * d], [v["U_max"]], "o" if v["caught"] else "x", color=colors[key], ms=4 if v["caught"] else 3.5)
        ax.axhline(w["U_star"], color=CAT[0], lw=1.0, ls="--")
        ax.set_title(f"T={w['T']:g} s, m={w['m']:g} kg, φ_ℓ={w['phi_l']:.3f}", fontsize=8)
        ax.set_xlim(-45, 45)
    for ax in axes[(n - 1) // cols * cols:]:
        ax.set_xlabel("release shift δ [ms]")
    for ax in axes[::cols]:
        ax.set_ylabel("re-integrated U_max")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[k], label=k) for k in ("held", "cone exceeded", "lost hook")] + \
              [plt.Line2D([], [], marker="x", ls="", color=colors[k], label=k) for k in ("hit wall", "missed")] + \
              [plt.Line2D([], [], color=CAT[0], ls="--", label="planned U*")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "windows.pdf")); fig.savefig(os.path.join(out, "windows.png"), dpi=200)
    plt.close(fig)


def fig_sensitivity(out, path="results/figs/sensitivity.csv"):
    if not os.path.exists(path):
        return
    s = pd.read_csv(path)
    ref = s[s["tag"] == "ref"]["min_cap_BW"].values[0]
    s = s[s["tag"] != "ref"].copy()
    s["rel"] = 100 * (s["min_cap_BW"] / ref - 1)
    short = {"eps010": "ε = 0.10 s", "eps040": "ε = 0.40 s", "cap070": "joint capacity ×0.7", "cap130": "joint capacity ×1.3",
             "dc030": "Δc = 30 ms", "dc100": "Δc = 100 ms", "vrel3": "|v_rel| ≤ 3 m/s", "vrel5": "|v_rel| ≤ 5 m/s",
             "mu060": "μ_out = 0.6", "mu150": "μ_out = 1.5"}
    s["label"] = s["tag"].map(short)
    s = s.sort_values("rel")
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.barh(s["label"], s["rel"], color=[CAT[1] if v > 0 else CAT[0] for v in s["rel"]], height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(f"change of min. required capacity [%]\n(reference {ref:.2f} BW)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sensitivity.pdf")); fig.savefig(os.path.join(out, "sensitivity.png"), dpi=200)
    plt.close(fig)


def fig_search3d(out, paths=("results/search3d/T20_m66_init.json",)):
    hs = [(p, json.load(open(p))) for p in paths if os.path.exists(p)]
    if not hs:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    for i, (p, d) in enumerate(hs):
        h = np.array(d["history"])
        ax.plot(h[:, 0], h[:, 1], color=CAT[i], lw=1.6, label=os.path.basename(p).replace(".json", "") + " (best)")
        ax.plot(h[:, 0], h[:, 2], color=CAT[i], lw=0.9, alpha=0.5, label="population mean")
    ax.axhline(1000, color=INK2, lw=0.8, ls="--"); ax.text(2, 1010, "failure floor (1000 + 100·d_min + …)", fontsize=7, color=INK2)
    ax.set_xlabel("CMA-ES generation"); ax.set_ylabel("cost"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "search3d.pdf")); fig.savefig(os.path.join(out, "search3d.png"), dpi=200)
    plt.close(fig)


def fig_build(out, build_dir="results/build", grid_dir="results/grid"):
    files = glob.glob(os.path.join(build_dir, "rows_*_T20_m66*.csv"))
    if not files:
        return
    rows = []
    for f in files:
        tag = re.match(r"rows_(.+?)_T20", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        d["tag"] = tag
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["phi_l"] = d["phi_l"].round(4)
    d = d.sort_values("ok", ascending=False).drop_duplicates(["tag", "phi_l"])
    d = d[d["ok"] == 1]
    ref_files = glob.glob(os.path.join(grid_dir, "rows_ref_T20_m66*.csv"))
    if ref_files:
        r = pd.concat([pd.read_csv(f) for f in ref_files], ignore_index=True)
        r = r[r["ok"] == 1].copy(); r["tag"] = "stature175"; r["phi_l"] = r["phi_l"].round(4)
        d = pd.concat([d, r.drop_duplicates(["phi_l"])], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    st = sorted([t for t in d["tag"].unique() if t.startswith("stature")], key=lambda t: int(t[7:]))
    cmap = plt.get_cmap("Blues")
    for i, t in enumerate(st):
        dd = d[d["tag"] == t].sort_values("phi_l")
        axes[0].plot(dd["phi_l"], dd["req_cap_BW"], "-o", ms=2.5, lw=1.4, color=cmap(0.35 + 0.6 * i / max(len(st) - 1, 1)),
                     label=f"H = {int(t[7:])/100:.2f} m")
    axes[0].axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
    axes[0].set_xlabel("release phase φ_ℓ"); axes[0].set_ylabel("required capacity [BW]"); axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("stature (T = 20 s, m = 66 kg)", fontsize=9)
    arm = [("arm095", "arm ×0.95"), ("stature175", "arm ×1.00"), ("arm105", "arm ×1.05")]
    for i, (t, lab) in enumerate(arm):
        dd = d[d["tag"] == t].sort_values("phi_l")
        if len(dd):
            axes[1].plot(dd["phi_l"], dd["req_cap_BW"], "-o", ms=2.5, lw=1.4, color=CAT[i], label=lab)
    axes[1].axvspan(0.5, 1.0, color="#f2f2f0", zorder=0)
    axes[1].set_xlabel("release phase φ_ℓ"); axes[1].legend(fontsize=7); axes[1].set_title("arm-length ratio", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "build.pdf")); fig.savefig(os.path.join(out, "build.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_device(a.out)
    fig_windows(a.out)
    fig_sensitivity(a.out)
    fig_search3d(a.out)
    fig_build(a.out)
    print("extra figures written to", a.out)


if __name__ == "__main__":
    main()
