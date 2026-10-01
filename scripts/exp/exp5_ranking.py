"""Experiment 5: can the learned value rank short action candidates correctly? (decision quality, not regression error)

For 10 development states, 8 feasible candidate control sequences per horizon (0.3 s and 0.8 s) are generated around the
oracle control (smooth offsets, simulated in the environment). The reference Q of a candidate is its running cost plus the
value of an independent remaining-problem solve from its terminal state; the learned Q replaces that value by the network
prediction (value-only / Sobolev, with and without the out-of-distribution penalty). Reported: best-candidate pick rate,
Spearman rank correlation, selection regret R = Q_ref(chosen) - min Q_ref.

usage: python scripts/exp/exp5_ranking.py --grid results/grid --models results/suite/models/full --out results/suite/exp5 --workers 12
"""
import argparse, glob, json, os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import load_ref, solve_from_state, ref_state, ENV_KW
from katsumi.exp.util import pool_map, save_json
from katsumi.planar.env import PlanarCliffEnv
from katsumi.planar.model import NTH, NTAU

HORIZONS = (0.3, 0.8)
N_CAND = 8


def simulate(ref, oracle, x, t0, H, offset, control_dt=0.002):
    """Apply the oracle control (interpolated) + offset over H seconds from (x, t0); returns (ok, x_H, cost, U_max)."""
    env = PlanarCliffEnv(T=ref["T"], m=ref["m"], body_kw=ref.get("body_kw", {}) or {}, **{k: v for k, v in ENV_KW.items() if k not in ("control_dt",)}, control_dt=control_dt)
    env.reset(options=dict(t0=t0, state=(x[:NTH], x[NTH:])))
    N = oracle["S_U"].shape[1] - 1
    tS = t0 + np.linspace(0, oracle["d_s"], N + 1)
    env.u_prev = np.array([np.interp(t0, tS, row) for row in oracle["S_U"]])
    n = int(round(H / control_dt)); Umax = 0.0
    for i in range(n):
        tm = min(env.t + control_dt / 2, tS[-1])
        u = np.array([np.interp(tm, tS, row) for row in oracle["S_U"]]) + offset(env.t - t0)
        a = np.concatenate([np.clip(u, -1, 1), [-1.0]])
        obs, rew, term, trunc, inf = env.step(a)
        Umax = max(Umax, env.U)
        if term or trunc or env.mode != "A":
            return False, None, None, Umax
    p = ref["params"]
    return True, np.concatenate([env.th, env.thd]), p.get("w_E", 1.0) * env.E / p.get("t_ref", 1.0), Umax


def make_offsets(rng, H, n):
    """Smooth random torque offsets: sum of two sinusoids per joint with amplitudes 0.1 - 0.3, zero at t = 0."""
    offs = [lambda tau: np.zeros(NTAU)]
    for _ in range(n - 1):
        a1 = rng.uniform(0.1, 0.3, NTAU) * rng.choice([-1, 1], NTAU); a2 = rng.uniform(0.0, 0.15, NTAU) * rng.choice([-1, 1], NTAU)
        w1 = np.pi / H; w2 = 2 * np.pi / H * rng.uniform(0.8, 1.5)
        offs.append(lambda tau, a1=a1, a2=a2, w1=w1, w2=w2: a1 * np.sin(w1 * tau) + a2 * np.sin(w2 * tau))
    return offs


def task(t):
    ref = load_ref(t["ref"]); x, t0, _ = ref_state(ref, t["k"])
    out_dir = os.path.join(t["out"], "solves"); os.makedirs(out_dir, exist_ok=True)
    tag = f"{os.path.basename(t['ref'])[8:-4]}_k{t['k']:03d}"
    oracle = solve_from_state(ref, x, t0, warm_k=t["k"], fn=os.path.join(out_dir, f"{tag}_oracle.pkl"), max_cpu=t["max_cpu"])
    rows = []
    if not oracle.get("ok"):
        return [dict(task_key=t["task_key"], state=tag, ok=0)]
    rng = np.random.default_rng(t["seed"])
    for H in HORIZONS:
        offs = make_offsets(rng, H, N_CAND); tries = 0; i = 0
        while i < N_CAND and tries < 4 * N_CAND:
            ok, xH, cost, Umax = simulate(ref, oracle, x, t0, H, offs[i])
            tries += 1
            if not ok:
                if i == 0:
                    rows.append(dict(task_key=f"{t['task_key']}_H{H}_c{i}", state=tag, H=H, cand=i, feasible=0)); i += 1
                else:
                    offs[i] = make_offsets(rng, H, 2)[1]
                continue
            s = solve_from_state(oracle, xH, t0 + H, fn=os.path.join(out_dir, f"{tag}_H{H}_c{i}.pkl"), max_cpu=t["max_cpu"])
            rows.append(dict(task_key=f"{t['task_key']}_H{H}_c{i}", state=tag, ref=os.path.basename(t["ref"]), T=ref["T"], m=ref["m"], k=t["k"], t0=t0, H=H, cand=i,
                             feasible=1, cost=cost, Umax_h=Umax, solved=int(bool(s.get("ok"))), V_ref=s.get("J") if s.get("ok") else np.nan,
                             U_ref=s.get("U_peak") if s.get("ok") else np.nan, xH=json.dumps(np.round(xH, 6).tolist()), tH=t0 + H,
                             V_oracle_state=oracle["J"], Q_ref=(cost + s["J"]) if s.get("ok") else np.nan))
            i += 1
    return rows


def score_models(df, model_dirs, out):
    import torch
    from katsumi.learn.dual_field import load_models, make_features
    df = df[(df.get("feasible", 1) == 1)].copy()
    res = []
    for md in model_dirs:
        for fn in sorted(glob.glob(os.path.join(md, "models_seed*.pt"))):
            M = load_models(fn)
            for mname, nets in (("value-only", M["ens0"]), ("Sobolev", M["ens"])):
                for ood in (0, 1):
                    for (state, H), g in df.groupby(["state", "H"]):
                        g = g[g["solved"] == 1]
                        if len(g) < 3:
                            continue
                        X = np.stack([np.array(json.loads(s)) for s in g["xH"]])
                        z = torch.tensor(np.stack([make_features(X[i], g["tH"].iloc[i], g["T"].iloc[i], g["m"].iloc[i])[0] for i in range(len(g))]), dtype=torch.float32)
                        with torch.no_grad():
                            V = np.mean([n_(z)[0].numpy().ravel() for n_ in nets], 0)
                            if ood and M["density"] is not None:
                                w, means, pc, c0 = M["density"]
                                zs = ((z - nets[0].mu) / nets[0].sd).numpy()
                                ll = []
                                for k in range(len(w)):
                                    dz = (zs - means[k]) @ pc[k]
                                    ll.append(np.log(w[k]) + np.sum(np.log(np.diag(pc[k]))) - 0.5 * (dz ** 2).sum(1))
                                nll = -np.logaddexp.reduce(np.array(ll), 0)
                                V = V + np.maximum(nll - c0, 0) ** 2 / 10.0
                        Qhat = g["cost"].values + V; Qref = g["Q_ref"].values
                        ih = int(np.argmin(Qhat)); ib = int(np.argmin(Qref))
                        from scipy.stats import spearmanr
                        rho = spearmanr(Qhat, Qref).correlation if len(g) > 2 else np.nan
                        res.append(dict(model_dir=os.path.basename(md.rstrip("/")), seed=os.path.basename(fn), model=mname, ood=ood, state=state, H=H, n=len(g),
                                        best_pick=int(ih == ib), regret=float(Qref[ih] - Qref[ib]), rho=float(rho), unsolved=int((df[(df["state"] == state) & (df["H"] == H)]["solved"] == 0).sum())))
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(out, "ranking_scores.csv"), index=False)
    if len(r):
        summ = r.groupby(["model_dir", "model", "ood", "H"]).agg(best_pick=("best_pick", "mean"), regret_mean=("regret", "mean"), regret_median=("regret", "median"),
                                                                   rho=("rho", "mean"), n=("n", "size")).reset_index()
        summ.to_csv(os.path.join(out, "ranking_summary.csv"), index=False); print(summ.round(3).to_string(index=False))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--models", nargs="*", default=["results/suite/models/full"])
    ap.add_argument("--out", default="results/suite/exp5"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--max-cpu", type=float, default=900.0)
    ap.add_argument("--score-only", action="store_true"); ap.add_argument("--n-states", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    states = []
    for m, fr in ((60, (0.3, 0.55, 0.8)), (66, (0.15, 0.3, 0.55, 0.8)), (72, (0.3, 0.55, 0.8))):
        f = os.path.join(a.grid, f"sol_ref_T18_m{m}_phi0.250.pkl")
        if os.path.exists(f):
            r = pickle.load(open(f, "rb")); N = r["S_X"].shape[1] - 1
            for fr_ in fr:
                states.append((f, int(round(fr_ * N))))
    if a.n_states:
        states = states[:a.n_states]
    tasks = [dict(task_key=f"s{i:02d}", ref=f, k=k, out=a.out, max_cpu=a.max_cpu, seed=i) for i, (f, k) in enumerate(states)]
    if not a.score_only:
        df = pool_map(task, tasks, a.workers, os.path.join(a.out, "candidates.csv"), "exp5")
    else:
        df = pd.read_csv(os.path.join(a.out, "candidates.csv"))
    if len(df):
        score_models(df, a.models, a.out)


if __name__ == "__main__":
    main()
