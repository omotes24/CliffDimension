"""Plotting of reduced-model solutions (stick figures, forces, utilisation)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import device
from .anthro import Body
from .model import PlanarChain, NQ, NTH


def _phase_frames(res, body, n_per_phase=(10, 8, 8)):
    """Return list of (t_abs, q, phase) sampled along the solution."""
    frames = []
    T, eps = res["T"], res["params"]["eps"]
    for ph, n, dur, t0 in (("S", n_per_phase[0], res["d_s"], res["t_s0"]),
                           ("F", n_per_phase[1], res["d_f"], res["t_l"]),
                           ("H", n_per_phase[2], res["params"]["T_hold"], res["t_h0"])):
        X = res[ph + "_X"]
        N = X.shape[1] - 1
        for i in np.linspace(0, N, n).round().astype(int):
            t = t0 + dur * i / N
            dv = device.device_state(t, T, eps)
            if ph in ("S",):
                q = np.concatenate([dv["pA"], X[:NTH, i]])
            elif ph == "H":
                q = np.concatenate([dv["pB"], X[:NTH, i]])
            else:
                q = X[:NQ, i]
            frames.append((t, q, ph))
    return frames


def draw_device(ax, t, T, eps, alpha=1.0):
    dv = device.device_state(t, T, eps)
    h, x = float(dv["h"]), float(dv["x"])
    d = device.D_LEDGE
    # A: ledge [-d, 0] x [h-0.05, h], face below to h-0.20, wall above
    ax.add_patch(plt.Rectangle((-d, h - 0.05), d, 0.05, color="#245B76", alpha=alpha))
    ax.add_patch(plt.Rectangle((-0.5, h - 0.20), 0.5 - d, 1.2, color="#CBD1D6", alpha=0.6 * alpha))
    ax.add_patch(plt.Rectangle((x, -0.05), d, 0.05, color="#245B76", alpha=alpha))
    ax.add_patch(plt.Rectangle((x + d, -0.20), 0.5 - d, 1.2, color="#CBD1D6", alpha=0.6 * alpha))


def plot_solution(res, body: Body, path: str, title: str = ""):
    chain = PlanarChain(body)
    T, eps = res["T"], res["params"]["eps"]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    frames = _phase_frames(res, body)
    colors = {"S": "#245B76", "F": "#9E4D3F", "H": "#3B7A57"}
    for j, (t, q, ph) in enumerate(frames):
        P = chain.points(q)
        ax.plot(P[0], P[1], "-o", color=colors[ph], ms=2.5, lw=1.4, alpha=0.35 + 0.65 * (j / len(frames)))
    draw_device(ax, res["t_l"], T, eps, alpha=0.5)
    draw_device(ax, res["t_c"], T, eps, alpha=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 3.4)
    ax.set_ylim(-2.6, 1.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or f"T={res['T']:.2f} s, m={res['m']:.0f} kg, phi0={res['phi0']:.2f}: "
                 f"release phi={res['phi_l']:.3f}, tau={res['d_f']:.3f} s, U_peak={res['U_peak']:.3f}, "
                 f"U_catch={res['U_catch']:.3f}, E_eff={res['effort']:.2f}")
    for ph, lab in (("S", "swing on A"), ("F", "flight"), ("H", "hold on B")):
        ax.plot([], [], color=colors[ph], label=lab)
    ax.legend(loc="lower right", fontsize=8)

    # utilisation U over time
    ax2 = fig.add_subplot(gs[1, :])
    tt, UU = [], []
    for ph, dur, t0 in (("S", res["d_s"], res["t_s0"]), ("H", res["params"]["T_hold"], res["t_h0"])):
        R = res[ph + "_R"]
        N = R.shape[1] - 1
        t = t0 + dur * np.arange(N + 1) / N - res["t0"]
        U = np.sqrt((R ** 2).sum(0))
        ax2.plot(t, U, color=colors[ph], lw=1.5)
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.plot([res["t_c"] - res["t0"]], [res["U_catch"]], "v", color="#A76A3E", ms=8,
             label=f"catch impulse / ({res['params']['delta_catch']*1e3:.0f} ms F_cap)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.axvspan(res["t_l"] - res["t0"], res["t_c"] - res["t0"], color=colors["F"], alpha=0.15)
    ax2.set_ylabel("U = |R| / F_cap")
    ax2.set_xlabel("time since start [s]")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    # torques
    ax3 = fig.add_subplot(gs[2, :])
    names = ["elbow", "shoulder", "hip", "knee"]
    for ph, dur, t0 in (("S", res["d_s"], res["t_s0"]), ("F", res["d_f"], res["t_l"]),
                        ("H", res["params"]["T_hold"], res["t_h0"])):
        U = res[ph + "_U"]
        N = U.shape[1] - 1
        t = t0 + dur * np.arange(N + 1) / N - res["t0"]
        for j in range(4):
            ax3.plot(t, U[j], color=f"C{j}", lw=1.0, label=names[j] if ph == "S" else None)
    ax3.axvspan(res["t_l"] - res["t0"], res["t_c"] - res["t0"], color=colors["F"], alpha=0.15)
    ax3.set_ylabel("tau / tau_cap")
    ax3.set_xlabel("time since start [s]")
    ax3.set_ylim(-1.05, 1.05)
    ax3.legend(loc="upper right", fontsize=8, ncol=4)
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_solution_compact(res, body: Body, path: str, title: str = ""):
    """Two-panel version for the manuscript: stick figures + grasp utilisation."""
    chain = PlanarChain(body)
    T, eps = res["T"], res["params"]["eps"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.1), gridspec_kw=dict(height_ratios=[2.3, 1]))
    frames = _phase_frames(res, body, n_per_phase=(8, 7, 7))
    colors = {"S": "#245B76", "F": "#9E4D3F", "H": "#3B7A57"}
    for j, (t, q, ph) in enumerate(frames):
        P = chain.points(q)
        ax.plot(P[0], P[1], "-o", color=colors[ph], ms=2.2, lw=1.3, alpha=0.35 + 0.65 * (j / len(frames)))
    draw_device(ax, res["t_l"], T, eps, alpha=0.5)
    draw_device(ax, res["t_c"], T, eps, alpha=1.0)
    ax.set_aspect("equal"); ax.set_xlim(-0.6, 3.4); ax.set_ylim(-2.3, 1.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title, fontsize=9)
    for ph, lab in (("S", "swing on A"), ("F", "flight"), ("H", "hold on B")):
        ax.plot([], [], color=colors[ph], label=lab)
    ax.legend(loc="lower right", fontsize=7.5)
    for ph, dur, t0 in (("S", res["d_s"], res["t_s0"]), ("H", res["params"]["T_hold"], res["t_h0"])):
        R = res[ph + "_R"]; N = R.shape[1] - 1
        t = t0 + dur * np.arange(N + 1) / N - res["t_s0"]
        ax2.plot(t, np.sqrt((R ** 2).sum(0)), color=colors[ph], lw=1.5)
    ax2.axhline(res["U_peak"], color="k", lw=0.8, ls="--")
    ax2.axvspan(res["t_l"] - res["t_s0"], res["t_c"] - res["t_s0"], color=colors["F"], alpha=0.15)
    ax2.plot([res["t_c"] - res["t_s0"]], [res["U_catch"]], "v", color="#A76A3E", ms=7)
    ax2.set_ylabel("U = |R| / F_cap"); ax2.set_xlabel("time since swing start [s]"); ax2.set_ylim(0, 1.25 * res["U_peak"])
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
