"""Dual DAgger: roll out the Pontryagin (dual-field) controller in the environment, collect the visited swing states and
write them as from-state oracle jobs; the oracle labels them (dual_field_data.py --states) and the field is retrained.

    python scripts/dfl_dagger.py --models results/dfl/models.pt --data results/dual_field \
        --refs "results/grid/sol_ref_T1[6789]_m*_phi*.pkl" --out results/dual_field/states_it1.json --episodes 2 --stride 0.25

Each visited state is paired with the reference solution of its episode and the knot whose remaining swing time is
closest to the field's own tau estimate (warm start for the oracle).
"""
import argparse, glob, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from katsumi.planar.env import PlanarCliffEnv
from katsumi.planar.model import NTH
from katsumi.learn.dual_field import DualFieldNet, PontryaginController, make_features
from katsumi.learn.swing_mpc import SwingMPC
from katsumi.learn.reflex import LandingReflex, ReflexEnv


def load_ensemble(path):
    ck = torch.load(path, weights_only=False)
    sds = ck.get("ens") or [ck["dfl"]]
    nets = []
    for sd in sds:
        lin = [k for k in sd if k.endswith(".weight")]
        width = sd[lin[0]].shape[0]; depth = len(lin) - 1
        n_ = DualFieldNet(ck["mu"].numpy(), ck["sd"].numpy(), ck["y_mu"].numpy(), ck["y_sd"].numpy(), width=width, depth=depth)
        n_.load_state_dict(sd); n_.eval(); nets.append(n_)
    return nets


def load_field(path, key="dfl"):
    ck = torch.load(path, weights_only=False)
    sd = ck[key]
    lin = [k for k in sd if k.endswith(".weight")]
    width = sd[lin[0]].shape[0]; depth = len(lin) - 1
    net = DualFieldNet(ck["mu"].numpy(), ck["sd"].numpy(), ck["y_mu"].numpy(), ck["y_sd"].numpy(), width=width, depth=depth)
    net.load_state_dict(sd); net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2, help="episodes per reference (rest start + perturbed starts)")
    ap.add_argument("--stride", type=float, default=0.25, help="seconds between recorded states")
    ap.add_argument("--noise", type=float, default=0.0, help="exploration noise on the torques")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-states", type=int, default=400)
    ap.add_argument("--H", type=int, default=25)
    ap.add_argument("--beta", type=float, default=2.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    nets = load_ensemble(a.models)
    refs = [f for f in sorted(glob.glob(a.refs)) if pickle.load(open(f, "rb")).get("ok")]
    reflex = LandingReflex([pickle.load(open(f, "rb")) for f in refs])
    jobs = []
    summary = []
    for f in refs:
        r = pickle.load(open(f, "rb"))
        if r["U_peak"] > 2.5:
            continue
        T, m = r["T"], r["m"]
        body_kw = r.get("body_kw", {}) or {}
        env = ReflexEnv(PlanarCliffEnv(T=T, m=m, body_kw=body_kw), reflex)
        ctrl = SwingMPC(nets, env.env.body, env.env.ch, T, H=a.H, stature=body_kw.get("stature", 1.75), cap_scale=body_kw.get("cap_scale", 1.0), beta=a.beta)
        N = r["S_X"].shape[1] - 1
        tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
        rem_ref = r["d_s"] - (tS - r["t_s0"])
        starts = [dict(t0=r["t_s0"], state=(np.zeros(NTH), np.zeros(NTH)))]
        for j in range(a.episodes - 1):
            k = int(rng.integers(int(0.1 * N), int(0.8 * N)))
            x = r["S_X"][:, k] + np.concatenate([rng.normal(0, 0.03, NTH), rng.normal(0, 0.2, NTH)])
            starts.append(dict(t0=float(tS[k]), state=(x[:NTH], x[NTH:])))
        for si, st in enumerate(starts):
            obs, info = env.reset(options=st)
            ctrl.reset()
            t_next = env.t
            done = False; n = 0
            while not done and n < 2000:
                a_ = ctrl(env.env)
                if a.noise > 0:
                    a_[:4] = np.clip(a_[:4] + rng.normal(0, a.noise, 4), -1, 1)
                if env.mode == "A" and env.t >= t_next - 1e-9:
                    tau_hat = ctrl.last.get("tau", 1.0)
                    k = int(np.argmin(np.abs(rem_ref - max(tau_hat, 0.05))))
                    jobs.append(dict(ref=f, t0=float(env.t), x0=np.concatenate([env.th, env.thd]).tolist(), k=k, rem=float(max(tau_hat, 0.1)),
                                     rep=len(jobs), level=-1.0, tag=f"{os.path.basename(f)[4:-4]}_dag{len(jobs):05d}"))
                    t_next += a.stride
                obs, rew, term, trunc, inf = env.step(a_)
                done = term or trunc; n += 1
            res = env.result
            summary.append(dict(ref=os.path.basename(f), start=si, reason=res["reason"], success=res["success"], U_peak=res["U_peak"], t=res["t"]))
            print(f"[{os.path.basename(f)[8:-4]} start{si}] {res['reason']:26s} U={res['U_peak']:.3f} t={res['t']:.2f}  states so far {len(jobs)}", flush=True)
        if len(jobs) >= a.max_states:
            break
    rng.shuffle(jobs)
    jobs = jobs[:a.max_states]
    json.dump(jobs, open(a.out, "w"))
    json.dump(summary, open(a.out.replace(".json", "_rollouts.json"), "w"), indent=1)
    print("wrote", len(jobs), "states ->", a.out, " success rate", np.mean([s["success"] for s in summary]))


if __name__ == "__main__":
    main()
