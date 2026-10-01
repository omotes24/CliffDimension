"""Experiment 7: separate the effect of the amount of data from the effect of how it is collected.

Stage A (collection: reference neighbourhood): train value-only / Sobolev nets with oracle budgets of 100 / 300 / all
   independent solves (parent-trajectory-level split, held-out body excluded) and evaluate both with the same MPC.
Stage B (collection: visited states): roll out the current controllers (both nets, training conditions only), collect the
   visited swing states every `--stride` s, relabel them with the oracle (budget --dagger-budget solves), retrain both nets
   on base + visited data and evaluate. The learning curve is success vs oracle calls (failed solves count).

usage: python scripts/exp/exp7_data.py --grid results/grid --data results/dual_field --out results/suite/exp7 --workers 16
"""
import argparse, glob, json, os, pickle, subprocess, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "exp"))
from katsumi.exp.common import load_ref, F_MAX_LEVELS
from katsumi.exp.trials import make_env, start_options
from katsumi.exp.util import pool_map
from katsumi.planar.model import NTH, NTAU

PY = sys.executable


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc != 0:
        print(f"  command failed (rc={rc}), see {log}", flush=True)
    return rc


def collect_task(t):
    """Roll out a controller from a (perturbed) rest start and record the visited swing states every stride seconds."""
    from exp6_compare import build_controller
    r = load_ref(t["ref"])
    reflex_refs = [load_ref(f) for f in t["reflex_refs"]]
    c = build_controller(t["method"], r, t["models_file"], reflex_refs)
    env = make_env(r, reflex_refs=reflex_refs)
    env.reset(options=start_options(r, "rest", t["sig_th"], t["sig_thd"], t["seed"])); c.reset()
    N = r["S_X"].shape[1] - 1; tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
    states = []; next_t = env.t; done = False; steps = 0
    while not done and steps < int(30 / env.control_dt):
        if env.mode == "A" and env.t >= next_t - 1e-9:
            k = int(np.clip(np.searchsorted(tS, env.t), 0, N - 1))
            states.append(dict(ref=t["ref"], t0=float(env.t), x0=np.concatenate([env.th, env.thd]).tolist(), k=k, rem=float(max(r["d_s"] - (env.t - r["t_s0"]), 0.3)),
                               rep=len(states), level=-1, tag=f"{os.path.basename(t['ref'])[4:-4]}_{t['method']}_s{t['seed']}_v{len(states):03d}"))
            next_t += t["stride"]
        a = c(env.env) if env.mode == "A" else np.zeros(NTAU + 1)
        obs, rew, term, trunc, inf = env.step(a); done = term or trunc; steps += 1
    return dict(task_key=t["task_key"], n_states=len(states), reason=(env.env.result or {}).get("reason"), states=json.dumps(states))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--data", default="results/dual_field")
    ap.add_argument("--out", default="results/suite/exp7"); ap.add_argument("--models-full", default="results/suite/models/full")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--budgets", type=int, nargs="+", default=[100, 300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--n-perturb", type=int, default=5)
    ap.add_argument("--train-m", type=float, nargs="+", default=[60.0, 66.0, 72.0]); ap.add_argument("--eval-m", type=float, nargs="+", default=[60.0, 66.0, 69.0, 72.0])
    ap.add_argument("--stride", type=float, default=0.25); ap.add_argument("--dagger-budget", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=6000); ap.add_argument("--stage", default="A,B")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); log = os.path.join(a.out, "exp7.log")
    seeds = [str(s) for s in a.seeds]
    evals = []
    if "A" in a.stage:
        for B in a.budgets:
            md = os.path.join(a.out, f"models_budget{B}")
            run([PY, "scripts/exp/train_fields.py", "--data", a.data, "--out", md, "--seeds", *seeds, "--budget", str(B), "--epochs", str(a.epochs)], log)
            run([PY, "scripts/exp/exp6_compare.py", "--grid", a.grid, "--models", md, "--out", os.path.join(a.out, f"eval_budget{B}"), "--workers", str(a.workers),
                 "--m", *[f"{m:g}" for m in a.eval_m], "--n-perturb", str(a.n_perturb), "--methods", "VMPC,SMPC", "--recovery-knots", "0"], log)
            evals.append(("reference", B, os.path.join(a.out, f"eval_budget{B}")))
        # the full data set (all solves) is the last point of the curve: reuse the main models / evaluation when present
        full_eval = os.path.join(os.path.dirname(a.out), "exp6")
        if os.path.exists(os.path.join(full_eval, "trials.csv")):
            evals.append(("reference", -1, full_eval))
    if "B" in a.stage:
        # 1. visited states of both controllers on the training conditions (first learning seed)
        mf = sorted(glob.glob(os.path.join(a.models_full, "models_seed*.pt")))
        if not mf:
            print("no full models for the DAgger stage"); return
        refs = [os.path.join(a.grid, f"sol_ref_T18_m{m:g}_phi0.250.pkl") for m in a.train_m]
        refs = [f for f in refs if os.path.exists(f)]
        tasks = [dict(task_key=f"{os.path.basename(f)}_{meth}_s{s}", ref=f, reflex_refs=refs, method=meth, models_file=mf[0], sig_th=(0.0 if s == 0 else 0.01),
                      sig_thd=(0.0 if s == 0 else 0.1), seed=s, stride=a.stride) for f in refs for meth in ("VMPC", "SMPC") for s in range(3)]
        dc = pool_map(collect_task, tasks, a.workers, os.path.join(a.out, "visited_rollouts.csv"), "exp7-collect")
        states = []
        for s in dc["states"].dropna():
            states += json.loads(s)
        rng = np.random.default_rng(0); rng.shuffle(states)
        states = states[:a.dagger_budget]
        for i, s in enumerate(states):
            s["rep"] = i
        sf = os.path.join(a.out, "visited_states.json"); json.dump(states, open(sf, "w"))
        print(f"visited states to relabel: {len(states)}", flush=True)
        # 2. oracle relabels (failed solves count towards the budget)
        dd = os.path.join(a.out, "dagger_data")
        run([PY, "scripts/dual_field_data.py", "--states", sf, "--out", dd, "--workers", str(a.workers), "--max-cpu", "600"], log)
        # 3. retrain: base budget B0 + visited vs base budget B0 + the same number of further reference solves
        B0 = a.dagger_budget
        md_v = os.path.join(a.out, f"models_visited{B0}")
        run([PY, "scripts/exp/train_fields.py", "--data", a.data, "--extra", dd, "--out", md_v, "--seeds", *seeds, "--budget", str(2 * B0), "--epochs", str(a.epochs)], log)
        md_r = os.path.join(a.out, f"models_budget{2 * B0}")
        run([PY, "scripts/exp/train_fields.py", "--data", a.data, "--out", md_r, "--seeds", *seeds, "--budget", str(2 * B0), "--epochs", str(a.epochs)], log)
        for name, md in (("visited", md_v), ("reference", md_r)):
            ev = os.path.join(a.out, f"eval_{name}{2 * B0}")
            run([PY, "scripts/exp/exp6_compare.py", "--grid", a.grid, "--models", md, "--out", ev, "--workers", str(a.workers), "--m", *[f"{m:g}" for m in a.eval_m],
                 "--n-perturb", str(a.n_perturb), "--methods", "VMPC,SMPC", "--recovery-knots", "0"], log)
            evals.append((name, 2 * B0, ev))
    # learning curve table
    rows = []
    for coll, B, ev in evals:
        fn = os.path.join(ev, "trials.csv")
        if not os.path.exists(fn):
            continue
        d = pd.read_csv(fn); d = d[d["group"] == "main"]
        info = {}
        for mf in glob.glob(os.path.join(os.path.dirname(fn).replace("eval_", "models_"), "fit_seed0.json")):
            info = json.load(open(mf)).get("info", {})
        for meth, g in d.groupby("method"):
            rows.append(dict(collection=coll, budget=B, n_solves=info.get("n_solves"), method=meth, n=len(g), **{f"succ_{lv:g}": g[f"succ_{lv:g}"].mean() for lv in F_MAX_LEVELS},
                             released=g["released"].mean(), caught=g["caught"].mean()))
    if rows:
        lc = pd.DataFrame(rows); lc.to_csv(os.path.join(a.out, "learning_curve.csv"), index=False); print(lc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
