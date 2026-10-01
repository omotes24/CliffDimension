"""Train the dual-field networks for the experiment suite (GPU when available).

Per seed: an ensemble of Sobolev nets (value + costate), an ensemble of value-only nets (same architecture, alpha = 0),
a behaviour-cloning net and the Gaussian-mixture density of the training features. The split is by parent reference
trajectory (all perturbed / dense states of a parent stay together) and by held-out body mass. `--budget` limits the number
of independent oracle solves (solution files) of the training set (experiment 7); `--extra` adds further data directories
(visited-state relabels) whose solves count towards the budget as well.

usage: python scripts/exp/train_fields.py --data results/dual_field --out results/suite/models/full --seeds 0 1 2
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.learn.dual_field import load_dataset, train_dual_field, train_bc, fit_density
from dfl_experiment import split, report_fit


def load_many(dirs, dense=True):
    ds, Fs, Ys = [], [], []
    for i, dd in enumerate(dirs):
        d, F, Y = load_dataset(dd, dense=dense)
        d = d.copy(); d["data_dir"] = dd
        ds.append(d); Fs.append(F); Ys.append(Y)
    import pandas as pd
    d = pd.concat(ds, ignore_index=True)
    F = np.concatenate(Fs)
    Y = {k: np.concatenate([y[k] for y in Ys]) for k in Ys[0]}
    return d, F, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/dual_field"); ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default="results/suite/models/full")
    ap.add_argument("--holdout-m", type=float, default=69.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=6000); ap.add_argument("--width", type=int, default=384); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--ensemble", type=int, default=4); ap.add_argument("--gmm", type=int, default=24)
    ap.add_argument("--budget", type=int, default=None, help="max number of independent solves (solution files) in the training set")
    ap.add_argument("--train-refs", default=None, help="json list of parent references allowed for training (others -> test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, F, Y = load_many([a.data] + list(a.extra))
    tr, va, te = split(d, a.holdout_m, seed=0)
    if a.train_refs:
        allowed = set(json.load(open(a.train_refs)))
        keep = d["ref"].isin(allowed).values
        tr = np.array([i for i in tr if keep[i]]); va = np.array([i for i in va if keep[i]])
        te = np.array(sorted(set(range(len(d))) - set(tr) - set(va)))
    n_solves = None
    if a.budget is not None:
        srcs = d["src"].values if "src" in d else d["tag"].values
        tr_src = sorted(set(srcs[tr]) | set(srcs[va]))
        rng = np.random.default_rng(0); rng.shuffle(tr_src)
        use = set(tr_src[:a.budget]); n_solves = len(use)
        tr = np.array([i for i in tr if srcs[i] in use]); va = np.array([i for i in va if srcs[i] in use])
    info = dict(n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)), budget=a.budget, n_solves=n_solves,
                device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")), data=[a.data] + list(a.extra))
    print(json.dumps(info), flush=True)
    kw = dict(epochs=a.epochs, log_every=2000, width=a.width, depth=a.depth, lr=a.lr, verbose=False)
    for seed in a.seeds:
        fn = os.path.join(a.out, f"models_seed{seed}.pt")
        if os.path.exists(fn):
            print("exists", fn); continue
        t0 = time.time()
        ens = [train_dual_field(F, Y, tr, va, alpha=1.0, seed=seed * 100 + j, **kw)[0] for j in range(a.ensemble)]
        ens0 = [train_dual_field(F, Y, tr, va, alpha=0.0, seed=seed * 100 + j, **kw)[0] for j in range(a.ensemble)]
        bc, _ = train_bc(F, Y, tr, va, seed=seed, **kw)
        dens = fit_density(F[tr], ens[0].mu.numpy(), ens[0].sd.numpy(), n_components=a.gmm, seed=seed)
        fits = []
        for name, idx in (("train", tr), ("val", va), ("holdout", te)):
            if len(idx) == 0:
                continue
            f1 = report_fit(ens[0], bc, F, Y, idx, name); f1["model"] = "DFL(Sobolev)"; f1["seed"] = seed
            f0 = report_fit(ens0[0], bc, F, Y, idx, name); f0["model"] = "DFL(value-only)"; f0["seed"] = seed
            fits += [f1, f0]
        torch.save(dict(dfl=ens[0].state_dict(), dfl0=ens0[0].state_dict(), bc=bc.state_dict(), mu=ens[0].mu, sd=ens[0].sd, y_mu=ens[0].y_mu, y_sd=ens[0].y_sd,
                        ens=[n_.state_dict() for n_ in ens], ens0=[n_.state_dict() for n_ in ens0], density=dens, info=info, fits=fits), fn)
        json.dump(dict(info=info, fits=fits), open(os.path.join(a.out, f"fit_seed{seed}.json"), "w"), indent=1)
        print(f"seed {seed}: {time.time() - t0:.0f}s", json.dumps([{k: round(v, 3) if isinstance(v, float) else v for k, v in f.items() if k in ('model', 'split', 'r2_V', 'rho_p', 'r2_tau')} for f in fits if f['split'] == 'holdout']), flush=True)


if __name__ == "__main__":
    main()
