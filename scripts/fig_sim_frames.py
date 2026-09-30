"""Simulation film strips for the manuscript: (top) the oracle solution, (bottom) the same solution executed in the
compiled environment (2 ms executor, landing reflex), as stick-figure snapshots with the device at each instant.

usage: python scripts/fig_sim_frames.py --ref results/v3/results/grid/sol_ref_T18_m66_phi0.250.pkl --out results/figs/sim_frames.pdf
"""
import argparse, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from katsumi import device
from katsumi.planar.plots import draw_device
from katsumi.planar.model import NQ, NTH
from katsumi.planar.env import PlanarCliffEnv
from katsumi.learn.reflex import LandingReflex, ReflexEnv
from replay_env import OracleTracker

COL = {"A": "#245B76", "F": "#9E4D3F", "C": "#3B7A57"}


def oracle_frames(r, times):
    """q(t) of the solution at absolute times (swing / flight / hold phases)."""
    T, eps = r["T"], r["params"]["eps"]
    out = []
    for t in times:
        if t < r["t_l"]:
            X, t0, dur = r["S_X"], r["t_s0"], r["d_s"]; ph = "A"
        elif t < r["t_c"]:
            X, t0, dur = r["F_X"], r["t_l"], r["d_f"]; ph = "F"
        else:
            X, t0, dur = r["H_X"], r["t_h0"], r["params"]["T_hold"]; ph = "C"
        N = X.shape[1] - 1
        tt = t0 + np.linspace(0, dur, N + 1)
        x = np.array([np.interp(t, tt, row) for row in X])
        dv = device.device_state(t, T, eps)
        if ph == "A":
            q = np.concatenate([dv["pA"], x[:NTH]])
        elif ph == "C":
            q = np.concatenate([dv["pB"], x[:NTH]])
        else:
            q = x[:NQ]
        out.append((t, q, ph))
    return out


def env_frames(r, control_dt=0.002, sub_dt=0.001):
    env = ReflexEnv(PlanarCliffEnv(T=r["T"], m=r["m"], body_kw=r.get("body_kw", {}) or {}, control_dt=control_dt, sub_dt=sub_dt),
                    LandingReflex([r], reach_gain=0.25, kp_hold=3.0, kd_hold=0.3))
    pol = OracleTracker(r, 1.0, 0.1, 0.0, lead=control_dt / 2)
    env.reset(options=dict(t0=r["t_s0"], state=(np.zeros(NTH), np.zeros(NTH))))
    log = []
    done = False
    while not done:
        a = pol(env.env)
        obs, rew, term, trunc, inf = env.step(a); done = term or trunc
        log.append((env.t, env.q.copy(), env.mode, env.U))
    return log, env.env.result


def draw(ax, chain, t, q, ph, T, eps, title):
    draw_device(ax, t, T, eps, alpha=1.0)
    P = chain.points(q)
    ax.plot(P[0], P[1], "-o", color=COL[ph], ms=2.5, lw=1.8)
    ax.set_aspect("equal"); ax.set_xlim(-0.7, 3.4); ax.set_ylim(-2.1, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/v3/results/grid/sol_ref_T18_m66_phi0.250.pkl")
    ap.add_argument("--out", default="results/figs/sim_frames.pdf")
    a = ap.parse_args()
    r = pickle.load(open(a.ref, "rb"))
    T, eps = r["T"], r["params"]["eps"]
    from katsumi.planar.anthro import make_body
    from katsumi.planar.model import PlanarChain
    chain = PlanarChain(make_body(r["m"], **(r.get("body_kw", {}) or {})))
    # snapshot instants: 5 through the swing, release, mid-flight, catch, +0.3 s, +1.5 s of the hold
    ts = r["t_s0"]; tl = r["t_l"]; tc = r["t_c"]
    times = [ts + f * (tl - ts) for f in (0.0, 0.5, 0.75, 0.9)] + [tl, 0.5 * (tl + tc), tc, tc + 0.3, tc + 1.5]
    labels = ["振り開始", "振り 50%", "振り 75%", "振り 90%", "離手", "飛行", "捕捉", "保持 +0.3 s", "保持 +1.5 s"]
    log, res = env_frames(r)
    fig, axes = plt.subplots(2, len(times), figsize=(7.4, 2.5))
    for j, (t, q, ph) in enumerate(oracle_frames(r, times)):
        draw(axes[0, j], chain, t, q, ph, T, eps, f"{labels[j]}\n$t$={t - ts:.2f} s")
    # environment: nearest logged instant (the release / catch instants of the executed episode)
    tlog = np.array([l[0] for l in log])
    t_rel = next((l[0] for l in log if l[2] != "A"), None); t_cat = next((l[0] for l in log if l[2] == "C"), None)
    env_times = [ts + f * ((t_rel or tl) - ts) for f in (0.0, 0.5, 0.75, 0.9)] + [t_rel or tl, 0.5 * ((t_rel or tl) + (t_cat or tc)),
                                                                                    t_cat or tc, (t_cat or tc) + 0.3, (t_cat or tc) + 1.5]
    for j, t in enumerate(env_times):
        k = int(np.clip(np.searchsorted(tlog, t), 0, len(log) - 1))
        tk, q, mode, U = log[k]
        draw(axes[1, j], chain, tk, q, mode, T, eps, f"環境 $t$={tk - ts:.2f} s\n$U$={U:.2f}")
    axes[0, 0].set_ylabel("オラクル解", fontsize=8); axes[1, 0].set_ylabel("環境で実行", fontsize=8)
    fig.suptitle(f"$T$={T:g} s, $m$={r['m']:g} kg, $\\phi_\\ell$={r['phi_l']:.3f}: {res['reason']} (U_peak={res['U_peak']:.2f}, U*={r['U_peak']:.2f})", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200); fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("saved", a.out, res["reason"])


if __name__ == "__main__":
    main()
