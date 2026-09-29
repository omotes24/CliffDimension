"""Frozen-policy evaluation on the condition grid (plan Sec. 8.3, 11, 12.3) — for glacus.

Supports (a) SB3 SAC checkpoints, (b) CMA-ES spline policies (results/search3d/*.pkl).
Per condition (T, m): 8 phase bins x n trials, success rate with the one-sided 95 % Clopper-Pearson lower
bound (eq. 19), CVaR_0.90 of U_peak with explicit failure penalty, mean E_eff over all trials, and the
success-only physical quantities in a separate table.

usage: python scripts/eval_policy.py --sac results/sac/seed0/sac_final.zip --T 9.5 10 --m 60 61 ... --n 25
       python scripts/eval_policy.py --cma results/search3d/T10_m66.pkl --n 5
"""
import argparse, os, sys, json, pickle
import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.mujoco.env import CliffEnv, EnvParams
from katsumi.mujoco.build_model import ModelParams


def clopper_pearson_lower(k, n, alpha=0.05):
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def cvar(values, q=0.90):
    v = np.sort(np.asarray(values))
    k = max(int(np.ceil((1 - q) * len(v))), 1)
    return float(v[-k:].mean())


def run_trials(policy_fn, T, m, n_per_bin, seed=0, fail_penalty_U=3.0):
    env = CliffEnv(EnvParams(model=ModelParams(m=m), T=T))
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(8):
        for i in range(n_per_bin):
            phi0 = (b + rng.uniform(0, 1)) * 0.125          # 8 bins over one full period
            obs = env.reset(T=T, phi0=phi0)
            t = 0.0
            while True:
                a = policy_fn(obs, t, env)
                obs, r, term, trunc, info = env.step(a)
                t = env.t_elapsed
                if term or trunc:
                    break
            res = env.result
            rows.append(dict(T=T, m=m, phi0=phi0, success=res["success"], reason=res["reason"], U_peak=res["U_peak"],
                             E_eff=res["E_eff"], release_time=res["release_time"], catch_time=res["catch_time"]))
    k = sum(r["success"] for r in rows)
    n = len(rows)
    U_risk = [r["U_peak"] if r["success"] else max(r["U_peak"], fail_penalty_U) for r in rows]
    summary = dict(T=T, m=m, n=n, k=k, p_hat=k / n, p_lower95=clopper_pearson_lower(k, n),
                   cvar90_U=cvar(U_risk), mean_E=float(np.mean([r["E_eff"] for r in rows])),
                   success_only=dict(mean_U=float(np.mean([r["U_peak"] for r in rows if r["success"]])) if k else None,
                                     mean_E=float(np.mean([r["E_eff"] for r in rows if r["success"]])) if k else None))
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sac", type=str, default="")
    ap.add_argument("--cma", type=str, default="")
    ap.add_argument("--T", type=float, nargs="+", default=[19.0, 20.0])
    ap.add_argument("--m", type=float, nargs="+", default=[60, 66, 75])
    ap.add_argument("--n", type=int, default=5, help="trials per phase bin (25 -> 200 per condition)")
    ap.add_argument("--out", type=str, default="results/eval/eval.json")
    a = ap.parse_args()
    if a.sac:
        from stable_baselines3 import SAC
        model = SAC.load(a.sac)
        def policy_fn(obs, t, env):
            act, _ = model.predict(obs, deterministic=True)
            act = np.asarray(act, float)
            act[-2:] = 0.5 * (act[-2:] + 1)
            return act
    elif a.cma:
        from katsumi.mujoco.search import SplinePolicy, SearchParams
        d = pickle.load(open(a.cma, "rb"))
        sp = SearchParams(**d["sp"])
        x = d["search"]["x_best"]
        pol_cache = {}
        def policy_fn(obs, t, env):
            if id(env) not in pol_cache:
                pol_cache[id(env)] = SplinePolicy(env, sp)
            return pol_cache[id(env)].action(x, t)
    else:
        raise SystemExit("--sac or --cma required")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    summaries, all_rows = [], []
    for T in a.T:
        for m in a.m:
            s, rows = run_trials(policy_fn, T, m, a.n)
            summaries.append(s)
            all_rows += rows
            print(s, flush=True)
    json.dump(dict(summary=summaries, trials=all_rows), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
