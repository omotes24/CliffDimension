"""Dual-field learning experiment: fit the dual field / BC on the oracle dataset, report generalisation, and evaluate
the derived controllers closed-loop in the planar environment (with the shared landing reflex).

    python scripts/dfl_experiment.py --data results/dual_field --refs "results/v3/results/grid/sol_ref_T1[6789]_m*_phi*.pkl" \
        --holdout-m 69 --out results/dfl --episodes 40

Splits: rows of the held-out body mass are the transfer test set; the remaining rows are split 85/15 (by reference
solution) into train / validation. Models: DFL (Sobolev, alpha=1), DFL value-only (alpha=0), BC. Closed-loop
evaluation: for each start (reference swing start at rest + perturbed states 1-3 s into the swing) every controller
runs one episode; success rate, U_peak, release-time error and swing-peak U are reported per body group.
"""
import argparse, glob, json, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from katsumi.planar.env import PlanarCliffEnv
from katsumi.planar.anthro import make_body
from katsumi.planar.model import PlanarChain, NTH
from katsumi.learn.dual_field import (load_rows, load_dataset, train_dual_field, train_bc, r2, PontryaginController, BCController, run_episode,
                                      make_features)
from katsumi.learn.reflex import LandingReflex, ReflexEnv
from katsumi.learn.swing_mpc import SwingMPC


def split(d, holdout_m, seed=0):
    rng = np.random.default_rng(seed)
    test = (d["m"].values == holdout_m)
    refs = sorted(set(d.loc[~test, "ref"]))
    rng.shuffle(refs)
    n_va = max(1, int(round(0.15 * len(refs))))
    va_refs = set(refs[:n_va])
    va = (~test) & d["ref"].isin(va_refs).values
    tr = (~test) & ~va
    return np.where(tr)[0], np.where(va)[0], np.where(test)[0]


def report_fit(net, bc, F, Y, idx, name):
    z = torch.tensor(F[idx], dtype=torch.float32)
    V, U, tau, p = net.value_and_costate(z)
    V, U, tau, p = V.detach().numpy(), U.detach().numpy(), tau.detach().numpy(), p.detach().numpy()
    with torch.no_grad():
        u = bc(z).numpy()
    out = dict(split=name, n=len(idx), r2_V=r2(Y["J"][idx], V), r2_U=r2(Y["U"][idx], U), r2_tau=r2(Y["tau"][idx], tau),
               r2_p=float(np.mean([r2(Y["p"][idx][:, i], p[:, i]) for i in range(2 * NTH)])),
               r2_p_th=float(np.mean([r2(Y["p"][idx][:, i], p[:, i]) for i in range(NTH)])),
               r2_p_thd=float(np.mean([r2(Y["p"][idx][:, i], p[:, i]) for i in range(NTH, 2 * NTH)])),
               r2_u=float(np.mean([r2(Y["u"][idx][:, i], u[:, i]) for i in range(4)])),
               rmse_tau=float(np.sqrt(np.mean((Y["tau"][idx] - tau) ** 2))), rmse_U=float(np.sqrt(np.mean((Y["U"][idx] - U) ** 2))))
    return out


def eval_controllers(nets, refs, d, out_dir, episodes_per_ref=6, seed=0, control_dt=0.02, holdout_m=None, controller="mpc", H_mpc=25):
    rng = np.random.default_rng(seed)
    rows = []
    reflex = LandingReflex([pickle.load(open(f, "rb")) for f in refs])
    for f in refs:
        r = pickle.load(open(f, "rb"))
        if not r.get("ok") or r["U_peak"] > 2.5:
            continue
        T, m = r["T"], r["m"]
        body_kw = r.get("body_kw", {}) or {}
        env = ReflexEnv(PlanarCliffEnv(T=T, m=m, body_kw=body_kw, control_dt=control_dt), reflex)
        body, chain = env.env.body, env.env.ch
        N = r["S_X"].shape[1] - 1
        tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
        starts = [dict(t0=r["t_s0"], state=(np.zeros(NTH), np.zeros(NTH)), kind="rest")]
        for j in range(episodes_per_ref - 1):
            k = int(rng.integers(int(0.1 * N), int(0.8 * N)))
            x = r["S_X"][:, k] + np.concatenate([rng.normal(0, 0.03, NTH), rng.normal(0, 0.2, NTH)])
            starts.append(dict(t0=float(tS[k]), state=(x[:NTH], x[NTH:]), kind=f"perturbed_k{k}"))
        ctrls = {}
        for name, (net, bc) in nets.items():
            st_, cs_ = body_kw.get("stature", 1.75), body_kw.get("cap_scale", 1.0)
            if bc is None:
                if controller == "mpc":
                    ctrls[name] = SwingMPC(net, body, chain, T, H=H_mpc, control_dt=control_dt, stature=st_, cap_scale=cs_)
                else:
                    ctrls[name] = PontryaginController(net, body, chain, T, control_dt=control_dt, stature=st_, cap_scale=cs_)
            else:
                ctrls[name] = BCController(bc, net, body, T, control_dt=control_dt, stature=st_, cap_scale=cs_)
        for st in starts:
            kind = st.pop("kind")
            for name, c in ctrls.items():
                t1 = time.time()
                res = run_episode(env, c, dict(st))
                rel_err = (res["release_time"] + (st["t0"] - r["t_s0"]) - r["d_s"]) if res.get("release_time") is not None else None
                rows.append(dict(ref=os.path.basename(f), T=T, m=m, group="holdout" if m == holdout_m else "train", start=kind, ctrl=name,
                                 success=int(res["success"]), reason=res["reason"], U_peak=res["U_peak"], U_swing=res["U_swing"],
                                 U_star=r["U_peak"], release_err=rel_err, t=res["t"], wall_s=round(time.time() - t1, 1)))
                print(f"[{os.path.basename(f)[8:-4]} {kind:14s} {name:10s}] {res['reason']:26s} U={res['U_peak']:.3f} (U*={r['U_peak']:.3f}) "
                      f"rel_err={rel_err if rel_err is None else round(rel_err, 3)} {time.time() - t1:.1f}s", flush=True)
            st["kind"] = kind
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/dual_field")
    ap.add_argument("--refs", default="results/v3/results/grid/sol_ref_T1[6789]_m*_phi*.pkl")
    ap.add_argument("--holdout-m", type=float, default=69.0)
    ap.add_argument("--out", default="results/dfl")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--controller", default="mpc", choices=["mpc", "pmp"], help="dual-field controller for the closed loop")
    ap.add_argument("--H", type=int, default=25, help="MPC horizon (control periods)")
    ap.add_argument("--episodes", type=int, default=6, help="episodes per reference solution (1 rest start + perturbed starts)")
    ap.add_argument("--eval-refs", default=None, help="glob of references used for the closed-loop evaluation (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-dense", action="store_true", help="point labels only (no trajectory labels from the defect multipliers)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, F, Y = load_dataset(a.data, dense=not a.no_dense)
    tr, va, te = split(d, a.holdout_m, a.seed)
    print(f"rows: train {len(tr)} val {len(va)} holdout(m={a.holdout_m:g}) {len(te)}  (dense rows: {int(d['dense'].sum())})", flush=True)
    t0 = time.time()
    kw = dict(epochs=a.epochs, seed=a.seed, log_every=1000, width=a.width, depth=a.depth, lr=a.lr)
    net, h1 = train_dual_field(F, Y, tr, va, alpha=1.0, **kw)
    print(f"DFL trained {time.time() - t0:.0f}s", flush=True)
    net0, h2 = train_dual_field(F, Y, tr, va, alpha=0.0, **kw)
    bc, h3 = train_bc(F, Y, tr, va, **kw)
    fits = []
    for name, idx in (("train", tr), ("val", va), ("holdout", te)):
        if len(idx) == 0:
            continue
        f1 = report_fit(net, bc, F, Y, idx, name); f1["model"] = "DFL(Sobolev)"
        f0 = report_fit(net0, bc, F, Y, idx, name); f0["model"] = "DFL(value-only)"
        fits += [f1, f0]
        print(json.dumps(f1), "\n", json.dumps(f0), flush=True)
    json.dump(dict(fits=fits, hist_sobolev=h1, hist_value=h2, hist_bc=h3, n=dict(train=len(tr), val=len(va), holdout=len(te))),
              open(os.path.join(a.out, "fit.json"), "w"), indent=1)
    torch.save(dict(dfl=net.state_dict(), dfl0=net0.state_dict(), bc=bc.state_dict(), mu=net.mu, sd=net.sd, y_mu=net.y_mu, y_sd=net.y_sd),
               os.path.join(a.out, "models.pt"))
    if a.no_eval:
        return
    refs = sorted(glob.glob(a.eval_refs or a.refs))
    nets = {"DFL": (net, None), "DFL-noSob": (net0, None), "BC": (net, bc)}
    rows = eval_controllers(nets, refs, d, a.out, episodes_per_ref=a.episodes, seed=a.seed, holdout_m=a.holdout_m, controller=a.controller, H_mpc=a.H)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby(["group", "ctrl"]).agg(n=("success", "size"), success=("success", "mean"), U_peak=("U_peak", "median"),
                                            U_star=("U_star", "median"), rel_err_abs=("release_err", lambda s: np.nanmedian(np.abs(s.astype(float))))).reset_index()
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
