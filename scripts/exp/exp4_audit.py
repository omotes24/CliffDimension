"""Experiment 4: audit of the dual labels and of the value definition (before adding data).

4.1  state sufficiency: same (theta, theta_dot) at the same time with different previous commands u0 -> does the remaining
     problem (value, release time, first action) change? plus the past-peak-load check z_t vs the remaining U*.
4.2  finite-difference check of the costate at 12 states x 3 directions x 3 step sizes (central differences in normalised
     coordinates), with active-set / branch-switch bookkeeping.
4.3  numerical test of the network gradient (autograd in raw coordinates vs finite differences of the de-standardised
     forward pass; the Sobolev target p is the raw dJ/dx).
4.4  trajectory labels vs independent re-solves at 30 points (value-to-go, U*, tau, costate from defect multipliers).

usage: python scripts/exp/exp4_audit.py --grid results/grid --data results/dual_field --models results/dfl/it0/models.pt --out results/suite/exp4 --workers 12
"""
import argparse, glob, json, os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import load_ref, solve_from_state, ref_state, accrued_effort
from katsumi.exp.util import pool_map, save_json
from katsumi.planar.model import NTH

NSTATE = 12


def pick_knots(r, n=NSTATE):
    N = r["S_X"].shape[1] - 1
    return [int(round(f * N)) for f in np.linspace(0.05, 0.9, n)]


def state_scales(data_dir):
    """Per-coordinate scales of the swing state (std over the oracle dataset) for normalised perturbations."""
    rows = os.path.join(data_dir, "rows.csv")
    if os.path.exists(rows):
        d = pd.read_csv(rows)
        s = np.array([d[f"x{i}"].std() for i in range(2 * NTH)])
        if np.all(np.isfinite(s)) and np.all(s > 0):
            return s
    return np.array([0.3] * NTH + [1.5] * NTH)


# ---------------------------------------------------------------- 4.1 -------------------------------------------------
def task_u0(t):
    ref = load_ref(t["ref"]); x, t0, u = ref_state(ref, t["k"])
    rows = []
    for name, u0 in (("none", None), ("ref", u), ("plus", np.clip(u + 0.3, -1, 1)), ("minus", np.clip(u - 0.3, -1, 1))):
        fn = os.path.join(t["out"], "u0", f"k{t['k']:03d}_{name}.pkl")
        s = solve_from_state(ref, x, t0, u0=u0, u0_dt=0.02, warm_k=t["k"], fn=fn, max_cpu=t["max_cpu"])
        rows.append(dict(task_key=f"{t['task_key']}_{name}", k=t["k"], t_in=t0 - ref["t_s0"], u0=name, ok=int(bool(s.get("ok"))), J=s.get("J"),
                         U=s.get("U_peak"), d_s=s.get("d_s"), effort=s.get("effort"), u_first=json.dumps(np.round(s["S_U"][:, 0], 3).tolist()) if s.get("ok") else None,
                         u0_vals=json.dumps(None if u0 is None else np.round(u0, 3).tolist())))
    return rows


# ---------------------------------------------------------------- 4.2 -------------------------------------------------
def task_fd(t):
    ref = load_ref(t["ref"]); x, t0, u = ref_state(ref, t["k"])
    d = np.array(t["dir"]); sc = np.array(t["scales"]); eps = t["eps"]
    out = []
    base = solve_from_state(ref, x, t0, warm_k=t["k"], fn=os.path.join(t["out"], "fd", f"k{t['k']:03d}_base.pkl"), max_cpu=t["max_cpu"])
    if not base.get("ok"):
        return dict(task_key=t["task_key"], k=t["k"], ok=0)
    p = np.array(base["duals"]["costate_x0"])
    dx = eps * sc * d                                            # raw perturbation of a unit step in normalised coordinates
    sp = solve_from_state(ref, x + dx, t0, warm_k=t["k"], fn=os.path.join(t["out"], "fd", f"{t['task_key']}_p.pkl"), max_cpu=t["max_cpu"])
    sm = solve_from_state(ref, x - dx, t0, warm_k=t["k"], fn=os.path.join(t["out"], "fd", f"{t['task_key']}_m.pkl"), max_cpu=t["max_cpu"])
    row = dict(task_key=t["task_key"], k=t["k"], t_in=t0 - ref["t_s0"], dir=t["dir_name"], eps=eps, ok=int(sp.get("ok") and sm.get("ok")),
               J0=base["J"], Jp=sp.get("J"), Jm=sm.get("J"), d_s0=base["d_s"], d_sp=sp.get("d_s"), d_sm=sm.get("d_s"))
    if row["ok"]:
        D_fd = (sp["J"] - sm["J"]) / (2 * eps)                   # derivative per unit normalised step
        D_dual = float(p @ (sc * d))
        row.update(D_fd=D_fd, D_dual=D_dual, abs_err=abs(D_fd - D_dual), rel_err=abs(D_fd - D_dual) / max(abs(D_fd), 1e-6),
                   sign_ok=int(np.sign(D_fd) == np.sign(D_dual)))
        # active set: constraint categories with multiplier mass above tolerance
        def active(s):
            return {c for c, v in (s["duals"].get("categories") or {}).items() if v.get("abs_sum", 0) > 1e-5}
        row["active_set_changed"] = int(active(sp) != active(sm))
        row["branch_switch"] = int(abs(sp["d_s"] - sm["d_s"]) > 0.3 or abs(sp["J"] - sm["J"]) > 10 * eps * max(abs(D_fd), 1.0) + 0.5)
        # one-sided checks for the record
        row["D_fwd"] = (sp["J"] - base["J"]) / eps; row["D_bwd"] = (base["J"] - sm["J"]) / eps
    return row


# ---------------------------------------------------------------- 4.3 -------------------------------------------------
def net_gradient_test(models_path, data_dir, n=200, seed=0):
    import torch
    from katsumi.learn.dual_field import load_models, load_dataset
    net = load_models(models_path)["ens"][0]
    d, F, Y = load_dataset(data_dir)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=min(n, len(F)), replace=False)
    z = torch.tensor(F[idx], dtype=torch.float32)
    V, U, tau, p = net.value_and_costate(z)
    p = p.detach().numpy()
    # finite differences of the raw forward pass in raw coordinates
    h = 1e-3 * np.maximum(net.sd.numpy()[:2 * NTH], 1e-3)
    pf = np.zeros_like(p)
    with torch.no_grad():
        for i in range(2 * NTH):
            zp = z.clone(); zp[:, i] += float(h[i]); zm = z.clone(); zm[:, i] -= float(h[i])
            pf[:, i] = ((net(zp)[0] - net(zm)[0]) / (2 * h[i])).numpy().ravel()
    rel = np.abs(p - pf) / (np.abs(pf) + 1e-3)
    # the standardised-gradient identity: dVbar/dxbar_i = (s_i / s_V) dV/dx_i
    sV = float(net.y_sd[0]); s = net.sd.numpy()[:2 * NTH]
    pbar = p * s / sV
    lab = Y["p"][idx]
    return dict(n=int(len(idx)), autograd_vs_fd_rel_median=float(np.median(rel)), autograd_vs_fd_rel_max=float(rel.max()),
                label_units="raw dJ/dx (same as autograd p)", pbar_scale_check=dict(s_V=sV, s_x=s.tolist(), pbar_abs_median=float(np.median(np.abs(pbar)))),
                label_vs_net_sign_agreement=float(np.mean(np.sign(lab) == np.sign(p))), notes="loss uses p in raw units with a robust MAD scale; no s_i/s_V factor is needed")


# ---------------------------------------------------------------- 4.4 -------------------------------------------------
def task_traj(t):
    """One reference: a from-start solve with duals gives the trajectory labels; 10 knots are re-solved independently."""
    from katsumi.learn.labels import trajectory_labels
    ref = load_ref(t["ref"])
    x0, t0, _ = ref_state(ref, 0)
    full = solve_from_state(ref, x0, t0, warm_k=0, fn=os.path.join(t["out"], "traj", os.path.basename(t["ref"]).replace(".pkl", "_full.pkl")), max_cpu=t["max_cpu"])
    rows = []
    if not full.get("ok"):
        return [dict(task_key=t["task_key"], ok=0)]
    labs = {row["knot"]: row for row in trajectory_labels(full)}
    acc, Uk = accrued_effort(full)
    N = full["S_X"].shape[1] - 1
    tS = full["t_s0"] + np.linspace(0, full["d_s"], N + 1) if "t_s0" in full else t0 + np.linspace(0, full["d_s"], N + 1)
    for k in [int(round(f * N)) for f in np.linspace(0.05, 0.9, 10)]:
        x = full["S_X"][:, k]; tk = float(tS[k])
        s = solve_from_state(full, x, tk, warm_k=k, fn=os.path.join(t["out"], "traj", os.path.basename(t["ref"]).replace(".pkl", f"_k{k:03d}.pkl")), max_cpu=t["max_cpu"])
        lab = labs.get(k)
        row = dict(task_key=f"{t['task_key']}_k{k}", ref=os.path.basename(t["ref"]), k=k, t_in=tk - t0, ok=int(bool(s.get("ok"))),
                   V_label=(lab["J"] if lab else np.nan), V_resolve=s.get("J"), U_label=full["U_peak"], U_resolve=s.get("U_peak"),
                   tau_label=(lab["tau"] if lab else np.nan), tau_resolve=s.get("d_s"), z_t=float(Uk[:k + 1].max()),
                   past_peak_binding=int(float(Uk[:k + 1].max()) > (s.get("U_peak") or 9) + 1e-3))
        if lab is not None and s.get("ok") and "costate_x0" in (s.get("duals") or {}):
            pl = np.array(lab["p"]); pr = np.array(s["duals"]["costate_x0"])
            row.update(p_cos=float(pl @ pr / (np.linalg.norm(pl) * np.linalg.norm(pr) + 1e-12)), p_rel_err=float(np.linalg.norm(pl - pr) / (np.linalg.norm(pr) + 1e-9)),
                       p_sign_agree=float(np.mean(np.sign(pl) == np.sign(pr))))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--data", default="results/dual_field")
    ap.add_argument("--models", default="results/dfl/it0/models.pt"); ap.add_argument("--out", default="results/suite/exp4")
    ap.add_argument("--ref", default=None, help="reference for 4.1 / 4.2 (default: T18 m66 phi 0.25)")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--max-cpu", type=float, default=900.0)
    ap.add_argument("--parts", default="1,2,3,4"); ap.add_argument("--n-states", type=int, default=NSTATE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref = a.ref or os.path.join(a.grid, "sol_ref_T18_m66_phi0.250.pkl")
    r = load_ref(ref); knots = pick_knots(r, a.n_states)
    parts = set(a.parts.split(","))
    if "1" in parts:
        tasks = [dict(task_key=f"u0_k{k:03d}", ref=ref, k=k, out=a.out, max_cpu=a.max_cpu) for k in knots]
        d1 = pool_map(task_u0, tasks, a.workers, os.path.join(a.out, "u0_dependence.csv"), "exp4.1")
        if len(d1):
            piv = d1.pivot_table(index="k", columns="u0", values="J"); piv.to_csv(os.path.join(a.out, "u0_J.csv")); print(piv.round(4).to_string())
    if "2" in parts:
        sc = state_scales(a.data); rng = np.random.default_rng(0)
        dirs = {"theta1": np.eye(2 * NTH)[0], "thetad1": np.eye(2 * NTH)[NTH]}
        rd = rng.normal(size=2 * NTH); dirs["random"] = rd / np.linalg.norm(rd)
        tasks = []
        for k in knots:
            for dn, d in dirs.items():
                for eps in (1e-3, 3e-3, 1e-2):
                    tasks.append(dict(task_key=f"fd_k{k:03d}_{dn}_{eps:g}", ref=ref, k=k, dir=d.tolist(), dir_name=dn, eps=eps, scales=sc.tolist(), out=a.out, max_cpu=a.max_cpu))
        d2 = pool_map(task_fd, tasks, a.workers, os.path.join(a.out, "fd_check.csv"), "exp4.2")
        if len(d2) and "rel_err" in d2:
            ok = d2[d2["ok"] == 1]
            summ = dict(n=int(len(ok)), rel_err_median=float(ok["rel_err"].median()), rel_err_p90=float(ok["rel_err"].quantile(0.9)), rel_err_max=float(ok["rel_err"].max()),
                        sign_agreement=float(ok["sign_ok"].mean()), active_set_changed=float(ok["active_set_changed"].mean()), branch_switch=float(ok["branch_switch"].mean()),
                        by_eps={f"{e:g}": dict(median=float(g["rel_err"].median()), p90=float(g["rel_err"].quantile(0.9))) for e, g in ok.groupby("eps")},
                        by_dir={dn: float(g["rel_err"].median()) for dn, g in ok.groupby("dir")},
                        smooth_branch_only=dict(n=int(((ok["branch_switch"] == 0) & (ok["active_set_changed"] == 0)).sum()),
                                                rel_err_median=float(ok[(ok["branch_switch"] == 0) & (ok["active_set_changed"] == 0)]["rel_err"].median())))
            save_json(summ, os.path.join(a.out, "fd_summary.json")); print(json.dumps(summ, indent=1))
    if "3" in parts and os.path.exists(a.models):
        res = net_gradient_test(a.models, a.data); save_json(res, os.path.join(a.out, "net_gradient_test.json")); print(json.dumps(res, indent=1)[:1500])
    if "4" in parts:
        refs = [os.path.join(a.grid, f"sol_ref_T18_m{m}_phi0.250.pkl") for m in (60, 66, 72)]
        tasks = [dict(task_key=f"traj_{os.path.basename(f)}", ref=f, out=a.out, max_cpu=a.max_cpu) for f in refs if os.path.exists(f)]
        d4 = pool_map(task_traj, tasks, a.workers, os.path.join(a.out, "trajectory_labels.csv"), "exp4.4")
        if len(d4) and "V_resolve" in d4:
            ok = d4[d4["ok"] == 1].copy()
            ok["V_rel_err"] = (ok["V_label"] - ok["V_resolve"]).abs() / ok["V_resolve"].abs()
            ok["tau_err"] = (ok["tau_label"] - ok["tau_resolve"]).abs()
            summ = dict(n=int(len(ok)), V_rel_err_median=float(ok["V_rel_err"].median()), V_rel_err_max=float(ok["V_rel_err"].max()),
                        tau_err_median_s=float(ok["tau_err"].median()), U_resolve_vs_label_median=float((ok["U_resolve"] - ok["U_label"]).abs().median()),
                        p_cos_median=float(ok["p_cos"].median()) if "p_cos" in ok else None, p_rel_err_median=float(ok["p_rel_err"].median()) if "p_rel_err" in ok else None,
                        past_peak_binding_frac=float(ok["past_peak_binding"].mean()))
            save_json(summ, os.path.join(a.out, "trajectory_summary.json")); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
