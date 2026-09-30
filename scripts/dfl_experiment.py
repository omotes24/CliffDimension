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
                                      make_features, DualFieldNet, BCNet, fit_density)
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
    from scipy import stats
    P = Y["p"][idx]
    # robust costate metrics: Spearman rank correlation per component and the sign agreement (the costate is heavy
    # tailed near active constraints, which dominates a plain R^2)
    rho = [stats.spearmanr(P[:, i], p[:, i]).correlation for i in range(2 * NTH)]
    sign = [np.mean(np.sign(P[:, i]) == np.sign(p[:, i])) for i in range(2 * NTH)]
    # R^2 on the central 90% of |p| per component
    r2c = []
    for i in range(2 * NTH):
        q = np.quantile(np.abs(P[:, i]), 0.95); msk = np.abs(P[:, i]) <= q
        r2c.append(r2(P[msk, i], p[msk, i]))
    out = dict(split=name, n=len(idx), r2_V=r2(Y["J"][idx], V), r2_U=r2(Y["U"][idx], U), r2_tau=r2(Y["tau"][idx], tau),
               r2_p=float(np.mean([r2(P[:, i], p[:, i]) for i in range(2 * NTH)])),
               r2_p_th=float(np.mean([r2(P[:, i], p[:, i]) for i in range(NTH)])),
               r2_p_thd=float(np.mean([r2(P[:, i], p[:, i]) for i in range(NTH, 2 * NTH)])),
               r2_p_c90=float(np.mean(r2c)), rho_p=float(np.nanmean(rho)), sign_p=float(np.mean(sign)),
               rho_p_thd=float(np.nanmean(rho[NTH:])), sign_p_thd=float(np.mean(sign[NTH:])),
               r2_u=float(np.mean([r2(Y["u"][idx][:, i], u[:, i]) for i in range(4)])),
               rmse_tau=float(np.sqrt(np.mean((Y["tau"][idx] - tau) ** 2))), rmse_U=float(np.sqrt(np.mean((Y["U"][idx] - U) ** 2))))
    return out


def _load_models(path):
    ck = torch.load(path, weights_only=False)

    def mk(sd):
        lin = [k for k in sd if k.endswith(".weight")]
        n_ = DualFieldNet(ck["mu"].numpy(), ck["sd"].numpy(), ck["y_mu"].numpy(), ck["y_sd"].numpy(), width=sd[lin[0]].shape[0], depth=len(lin) - 1)
        n_.load_state_dict(sd); n_.eval(); return n_
    ens = [mk(sd) for sd in (ck.get("ens") or [ck["dfl"]])]
    net0 = mk(ck["dfl0"])
    dens = ck.get("density")
    density = (np.array(dens["w"]), np.array(dens["means"]), np.array(dens["prec_chol"]), dens["c0"]) if dens else None
    lin = [k for k in ck["bc"] if k.endswith(".weight")]
    bc = BCNet(ck["mu"].numpy(), ck["sd"].numpy(), width=ck["bc"][lin[0]].shape[0], depth=len(lin) - 1); bc.load_state_dict(ck["bc"]); bc.eval()
    return ens, net0, bc, density


def eval_task(args):
    """One (reference, start, controller) episode; models are loaded from disk in the worker."""
    f, st, kind, name, models_path, refs, control_dt, controller, H_mpc, beta, holdout_m, beta_d, replan, exec_kw = args
    exec_kw = dict(exec_kw or {})
    mpc_dt = exec_kw.pop("mpc_dt", 0.02); exec_refs = exec_kw.pop("exec_refs", None); reach = exec_kw.pop("reach", 0.25)
    w_tau = exec_kw.pop("w_tau", 0.0)
    kp_hold = exec_kw.pop("kp_hold", 3.0); kd_hold = exec_kw.pop("kd_hold", 0.3)
    torch.set_num_threads(1)
    ens, net0, bc, density = _load_models(models_path) if (models_path and os.path.exists(models_path)) else ([None], None, None, None)
    r = pickle.load(open(f, "rb"))
    T, m = r["T"], r["m"]
    body_kw = r.get("body_kw", {}) or {}
    reflex_refs = sorted(glob.glob(exec_refs)) if exec_refs else refs
    reflex = LandingReflex([pickle.load(open(g, "rb")) for g in reflex_refs], reach_gain=reach, kp_hold=kp_hold, kd_hold=kd_hold)
    env = ReflexEnv(PlanarCliffEnv(T=T, m=m, body_kw=body_kw, control_dt=control_dt, **exec_kw), reflex)
    body, chain = env.env.body, env.env.ch
    st_, cs_ = body_kw.get("stature", 1.75), body_kw.get("cap_scale", 1.0)
    if name == "BC":
        c = BCController(bc, ens[0], body, T, control_dt=control_dt, stature=st_, cap_scale=cs_)
    elif name == "ORACLE":
        from katsumi.learn.oracle_mpc import OracleMPC
        c = OracleMPC(r, body, T, replan_dt=0.5, control_dt=control_dt)
    else:
        net = ens if name == "DFL" else net0
        if controller == "mpc":
            c = SwingMPC(net, body, chain, T, H=H_mpc, control_dt=mpc_dt, stature=st_, cap_scale=cs_, beta=beta, density=density,
                         beta_d=beta_d, replan=replan, w_tau=w_tau)
        else:
            c = PontryaginController(net[0] if isinstance(net, list) else net, body, chain, T, control_dt=control_dt, stature=st_, cap_scale=cs_)
    t1 = time.time()
    res = run_episode(env, c, dict(st))
    rel_err = (res["release_time"] + (st["t0"] - r["t_s0"]) - r["d_s"]) if res.get("release_time") is not None else None
    d_min = res.get("d_min", float("inf"))
    row = dict(ref=os.path.basename(f), T=T, m=m, group="holdout" if m == holdout_m else "train", start=kind, ctrl=name,
               success=int(res["success"]), reason=res["reason"], U_peak=res["U_peak"], U_swing=res["U_swing"], U_star=r["U_peak"],
               release_err=rel_err, released=int(res.get("release_time") is not None), caught=int(res.get("catch_time") is not None),
               d_min=(d_min if np.isfinite(d_min) else np.nan), t=res["t"], wall_s=round(time.time() - t1, 1), mpc_fail=getattr(c, "n_fail", 0),
               n_solve=getattr(c, "n_solve", 0), solve_s=round(getattr(c, "solve_s", 0.0), 1), control_dt=control_dt)
    print(f"[{os.path.basename(f)[8:-4]} {kind:14s} {name:10s}] {res['reason']:26s} U={res['U_peak']:.3f} (U*={r['U_peak']:.3f}) "
          f"rel_err={rel_err if rel_err is None else round(rel_err, 3)} d_min={d_min:.3f} t={res['t']:.2f} {time.time() - t1:.0f}s", flush=True)
    return row


def eval_controllers(models_path, refs, out_dir, episodes_per_ref=6, seed=0, control_dt=0.004, holdout_m=None, controller="mpc", H_mpc=25,
                     beta=2.0, workers=8, ctrl_names=("DFL", "DFL-noSob", "BC"), beta_d=1.0, replan=2, exec_kw=None):
    rng = np.random.default_rng(seed)
    tasks = []
    for f in refs:
        r = pickle.load(open(f, "rb"))
        if not r.get("ok") or r["U_peak"] > 1.6:                 # feasible for a human grip (< U_cap of the environment)
            continue
        N = r["S_X"].shape[1] - 1
        tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
        starts = [(dict(t0=r["t_s0"], state=(np.zeros(NTH), np.zeros(NTH))), "rest")]
        for j in range(episodes_per_ref - 1):
            k = int(rng.integers(int(0.1 * N), int(0.8 * N)))
            x = r["S_X"][:, k] + np.concatenate([rng.normal(0, 0.03, NTH), rng.normal(0, 0.2, NTH)])
            starts.append((dict(t0=float(tS[k]), state=(x[:NTH], x[NTH:])), f"perturbed_k{k}"))
        for st, kind in starts:
            for name in ctrl_names:
                tasks.append((f, st, kind, name, models_path, refs, control_dt, controller, H_mpc, beta, holdout_m, beta_d, replan, exec_kw))
    import multiprocessing as mp
    rows = []
    ctx = mp.get_context("forkserver")
    with ctx.Pool(workers) as pool:
        for row in pool.imap_unordered(eval_task, tasks):
            rows.append(row)
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
    ap.add_argument("--ensemble", type=int, default=4, help="number of Sobolev fields (terminal value = mean + beta std)")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--beta-d", type=float, default=1.0, help="out-of-distribution (GMM negative log-density) penalty weight")
    ap.add_argument("--replan", type=int, default=2, help="MPC re-planning interval (control periods)")
    ap.add_argument("--gmm", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ctrls", default="DFL,BC", help="controllers to evaluate (DFL, DFL-noSob, BC)")
    ap.add_argument("--eval-only", default=None, help="models.pt to evaluate (skip training)")
    ap.add_argument("--episodes", type=int, default=6, help="episodes per reference solution (1 rest start + perturbed starts)")
    ap.add_argument("--eval-refs", default=None, help="glob of references used for the closed-loop evaluation (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-dense", action="store_true", help="point labels only (no trajectory labels from the defect multipliers)")
    ap.add_argument("--control-dt", type=float, default=0.002, help="environment control period [s] (executor rate)")
    ap.add_argument("--sub-dt", type=float, default=0.001, help="environment integration step [s]")
    ap.add_argument("--mpc-dt", type=float, default=0.02, help="dual-field MPC plan period [s] (horizon H * mpc_dt)")
    ap.add_argument("--reach", type=float, default=0.25, help="landing-reflex flight reach-correction gain")
    ap.add_argument("--kp-hold", type=float, default=3.0); ap.add_argument("--kd-hold", type=float, default=0.3)
    ap.add_argument("--hook-cap-B", type=float, default=0.1, help="finger-hook capacity on B (fraction of f_cap)")
    ap.add_argument("--w-tau", type=float, default=0.0, help="MPC progress term on the time-to-release field")
    ap.add_argument("--exec-refs", default=None, help="glob of oracle solutions for the landing reflex (default: the evaluation references)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    exec_kw = dict(reach=a.reach, hook_cap_B=a.hook_cap_B, mpc_dt=a.mpc_dt, exec_refs=a.exec_refs, sub_dt=a.sub_dt, kp_hold=a.kp_hold, kd_hold=a.kd_hold,
                   w_tau=a.w_tau)
    if a.eval_only:
        refs = sorted(glob.glob(a.eval_refs or a.refs))
        rows = eval_controllers(a.eval_only, refs, a.out, episodes_per_ref=a.episodes, seed=a.seed, holdout_m=a.holdout_m, control_dt=a.control_dt,
                                controller=a.controller, H_mpc=a.H, beta=a.beta, workers=a.workers, ctrl_names=tuple(a.ctrls.split(",")),
                                beta_d=a.beta_d, replan=a.replan, exec_kw=exec_kw)
        import pandas as pd
        df = pd.DataFrame(rows)
        summ = df.groupby(["group", "ctrl"]).agg(n=("success", "size"), success=("success", "mean"), U_peak=("U_peak", "median"),
                                                U_star=("U_star", "median")).reset_index()
        print(summ.to_string(index=False)); summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
        return
    d, F, Y = load_dataset(a.data, dense=not a.no_dense)
    tr, va, te = split(d, a.holdout_m, a.seed)
    print(f"rows: train {len(tr)} val {len(va)} holdout(m={a.holdout_m:g}) {len(te)}  (dense rows: {int(d['dense'].sum())})", flush=True)
    t0 = time.time()
    kw = dict(epochs=a.epochs, log_every=1000, width=a.width, depth=a.depth, lr=a.lr)
    ens = []
    for j in range(a.ensemble):
        n_, h1 = train_dual_field(F, Y, tr, va, alpha=1.0, seed=a.seed + j, **kw)
        ens.append(n_)
        print(f"DFL member {j} trained {time.time() - t0:.0f}s", flush=True)
    net = ens[0]
    net0, h2 = train_dual_field(F, Y, tr, va, alpha=0.0, seed=a.seed, **kw)
    bc, h3 = train_bc(F, Y, tr, va, seed=a.seed, **kw)
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
    dens = fit_density(F[tr], net.mu.numpy(), net.sd.numpy(), n_components=a.gmm, seed=a.seed)
    print("density: median nll %.2f, c0 %.2f" % (dens["nll_median"], dens["c0"]), flush=True)
    torch.save(dict(dfl=net.state_dict(), dfl0=net0.state_dict(), bc=bc.state_dict(), mu=net.mu, sd=net.sd, y_mu=net.y_mu, y_sd=net.y_sd,
                    ens=[n_.state_dict() for n_ in ens], density=dens), os.path.join(a.out, "models.pt"))
    if a.no_eval:
        return
    refs = sorted(glob.glob(a.eval_refs or a.refs))
    rows = eval_controllers(os.path.join(a.out, "models.pt"), refs, a.out, episodes_per_ref=a.episodes, seed=a.seed, holdout_m=a.holdout_m,
                            control_dt=a.control_dt, controller=a.controller, H_mpc=a.H, beta=a.beta, workers=a.workers,
                            ctrl_names=tuple(a.ctrls.split(",")), beta_d=a.beta_d, replan=a.replan, exec_kw=exec_kw)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby(["group", "ctrl"]).agg(n=("success", "size"), success=("success", "mean"), U_peak=("U_peak", "median"),
                                            U_star=("U_star", "median"), rel_err_abs=("release_err", lambda s: np.nanmedian(np.abs(s.astype(float))))).reset_index()
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
