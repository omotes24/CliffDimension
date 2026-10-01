"""Experiment 8: generalisation to unseen body / device conditions and transfer by sensitivities.

8a  fit-level splits on the existing data set: interpolation vs extrapolation in body mass (train 60/66/72 -> test 63/69 | 75)
    and in the device period (train 16/18 -> test 17 | 19); metrics per held-out condition for value-only and Sobolev nets.
8b  new bodies (stature 1.60-1.90 m, joint-capability 0.8-1.2 at T = 18 s, m = 66 kg, phi = 0.25): references + oracle labels,
    nets trained with the interpolation design, fit per condition; closed loop on the new bodies with the correct condition
    input (conditional generalisation) and with a deliberately wrong input (model error), separately.
8c  envelope-theorem transfer: predicted J(b0 + db) = J(b0) + dJ/dln p * dln p versus re-solves for the torque capacity and the
    hook coefficient (+-1/5/10/20 %), compared with "no correction" and with the conditional network.

usage: python scripts/exp/exp8_general.py --grid results/grid --data results/dual_field --out results/suite/exp8 --workers 16
"""
import argparse, glob, json, os, pickle, subprocess, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "exp"))
from katsumi.exp.common import load_ref, solve_variant, solution_metrics, F_MAX_LEVELS
from katsumi.exp.util import pool_map, save_json
PY = sys.executable


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)


def fit_by_condition(models_dir, data_dir, extra_dirs=()):
    """Per-(T, m, stature, cap_scale) fit metrics of both nets from the saved models."""
    import torch
    from scipy import stats
    from katsumi.learn.dual_field import load_models
    from train_fields import load_many
    d, F, Y = load_many([data_dir] + list(extra_dirs))
    rows = []
    for fn in sorted(glob.glob(os.path.join(models_dir, "models_seed*.pt"))):
        M = load_models(fn)
        for mname, net in (("value-only", M["ens0"][0]), ("Sobolev", M["ens"][0])):
            z = torch.tensor(F, dtype=torch.float32)
            V, U, tau, p = net.value_and_costate(z)
            V, tau, p = V.detach().numpy().ravel(), tau.detach().numpy().ravel(), p.detach().numpy()
            keys = ["T", "m"] + [c for c in ("stature", "cap_scale") if c in d.columns]
            for key, g in d.groupby(keys):
                idx = g.index.values
                if len(idx) < 20:
                    continue
                r2v = 1 - np.sum((V[idx] - Y["J"][idx]) ** 2) / max(np.sum((Y["J"][idx] - Y["J"][idx].mean()) ** 2), 1e-9)
                rho = np.nanmedian([stats.spearmanr(p[idx, i], Y["p"][idx, i]).correlation for i in range(p.shape[1])])
                sign = float(np.mean(np.sign(p[idx]) == np.sign(Y["p"][idx])))
                rows.append(dict(models=os.path.basename(fn), model=mname, **dict(zip(keys, key)), n=len(idx), r2_V=float(r2v), rmse_V=float(np.sqrt(np.mean((V[idx] - Y["J"][idx]) ** 2))),
                                 rmse_tau=float(np.sqrt(np.mean((tau[idx] - Y["tau"][idx]) ** 2))), rho_p=float(rho), sign_p=sign))
    return pd.DataFrame(rows)


def body_task(t):
    ref = load_ref(t["ref"])
    r = solve_variant(ref, body_kw=t["body_kw"], fn=t["fn"], max_cpu=t["max_cpu"])
    met = solution_metrics(r) if r.get("ok") else dict(ok=False, status=r.get("status"))
    met.update(task_key=t["task_key"], fn=t["fn"], **{k: v for k, v in t["body_kw"].items()})
    return met


def closedloop_task(t):
    """SMPC on a new body; with wrong=True the controller is conditioned on a 10 % different stature / capability."""
    import copy
    from exp6_compare import build_controller
    from katsumi.exp.trials import controller_trial
    r_ = load_ref(t["ref"]); r_c = copy.deepcopy(r_)
    if t["wrong"]:
        r_c["body_kw"] = {k: v * 1.1 for k, v in (r_c.get("body_kw", {}) or {}).items()}     # the controller's belief only
    c = build_controller("SMPC", r_c, t["models_file"], [r_])
    row = controller_trial(c, r_, start="rest", sig_th=(0 if t["seed"] == 0 else 0.01), sig_thd=(0 if t["seed"] == 0 else 0.1), seed=t["seed"], reflex_refs=[r_], label="SMPC")
    row.update(task_key=t["task_key"], body=os.path.basename(t["ref"]), wrong_condition=int(t["wrong"]))
    return row


def envelope_task(t):
    ref = load_ref(t["ref"])
    base = solve_variant(ref, fn=os.path.join(t["out"], "env", "base.pkl"), max_cpu=t["max_cpu"])
    r = solve_variant(ref, overrides=t["ov"], fn=t["fn"], max_cpu=t["max_cpu"])
    sens = (base.get("duals") or {}).get("dJ_dlnp", {})
    pred = base["J"] + t["dlnp_tau"] * float(np.sum(sens.get("tau_cap", [0]))) + t["dlnp_mu"] * float(np.sum(sens.get("mu_out", [0])) + np.sum(sens.get("mu_in", [0])))
    return dict(task_key=t["task_key"], variant=t["name"], ok=int(bool(r.get("ok"))), J_base=base["J"], J_resolve=r.get("J"), J_pred_envelope=pred,
                err_envelope=abs(pred - r["J"]) if r.get("ok") else np.nan, err_none=abs(base["J"] - r["J"]) if r.get("ok") else np.nan,
                U_base=base["U_peak"], U_resolve=r.get("U_peak"), dlnp_tau=t["dlnp_tau"], dlnp_mu=t["dlnp_mu"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--data", default="results/dual_field"); ap.add_argument("--out", default="results/suite/exp8")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--max-cpu", type=float, default=2400.0); ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--parts", default="a,b,c")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); log = os.path.join(a.out, "exp8.log")
    parts = a.parts.split(",")
    d = pd.read_csv(os.path.join(a.data, "rows.csv"))
    refs_all = sorted(set(d["ref"]))
    # ---- 8a: splits on the existing data -------------------------------------------------------------------------
    if "a" in parts:
        designs = {"mass": lambda f: float(f.split("_m")[1].split("_")[0]) in (60.0, 66.0, 72.0),
                   "period": lambda f: float(f.split("_T")[1].split("_")[0]) in (16.0, 18.0)}
        for name, allow in designs.items():
            md = os.path.join(a.out, f"models_split_{name}")
            tr = [f for f in refs_all if allow(f)]
            json.dump(tr, open(os.path.join(a.out, f"train_refs_{name}.json"), "w"))
            if len(glob.glob(os.path.join(md, "models_seed*.pt"))) < 2:
                run([PY, "scripts/exp/train_fields.py", "--data", a.data, "--out", md, "--seeds", "0", "1", "--epochs", str(a.epochs), "--holdout-m", "-1",
                     "--train-refs", os.path.join(a.out, f"train_refs_{name}.json")], log)
            fc = fit_by_condition(md, a.data)
            if len(fc):
                fc["train"] = [allow(f"sol_ref_T{T:g}_m{m:g}_phi0.250.pkl") for T, m in zip(fc["T"], fc["m"])]
                fc.to_csv(os.path.join(a.out, f"fit_by_condition_{name}.csv"), index=False)
                print(fc.groupby(["model", "train"])[["r2_V", "rmse_V", "rmse_tau", "rho_p", "sign_p"]].mean().round(3).to_string())
    # ---- 8b: new bodies (stature, capability) -------------------------------------------------------------------
    if "b" in parts:
        ref = os.path.join(a.grid, "sol_ref_T18_m66_phi0.250.pkl")
        bodies = [dict(stature=s) for s in (1.60, 1.65, 1.70, 1.80, 1.85, 1.90)] + [dict(cap_scale=c) for c in (0.8, 0.9, 0.95, 1.05, 1.1, 1.2)]
        tasks = [dict(task_key=json.dumps(bk), ref=ref, body_kw=bk, fn=os.path.join(a.out, "bodies", "sol_" + "_".join(f"{k}{v:.2f}" for k, v in bk.items()) + ".pkl"),
                      max_cpu=a.max_cpu) for bk in bodies]
        db = pool_map(body_task, tasks, a.workers, os.path.join(a.out, "bodies.csv"), "exp8-bodies")
        print(db[[c for c in ("stature", "cap_scale", "ok", "U_peak", "effort", "d_s") if c in db]].round(3).to_string(index=False))
        # oracle labels along the new references (stride 15, levels 0 / 1) -> extra data set
        okf = [f for f in db[db["ok"] == True]["fn"]] if "ok" in db else []
        dd = os.path.join(a.out, "bodies_data")
        if okf and not os.path.exists(os.path.join(dd, "rows.csv")):
            run([PY, "scripts/dual_field_data.py", "--refs", *okf, "--out", dd, "--stride", "15", "--levels", "0", "1", "--reps", "1", "--workers", str(a.workers), "--max-cpu", "600"], log)
        # interpolation design: train on stature {1.65, 1.75(base data), 1.85} and cap {0.9, 1.0, 1.1}; test the rest
        if os.path.exists(os.path.join(dd, "rows.csv")):
            db2 = pd.read_csv(os.path.join(dd, "rows.csv"))
            import re
            def allowed(ref_name):
                ms = re.search(r"stature([0-9.]+?)(?:_|\.pkl)", ref_name); mc = re.search(r"cap_scale([0-9.]+?)(?:_|\.pkl)", ref_name)
                st = round(float(ms.group(1)), 2) if ms else 1.75; cp = round(float(mc.group(1)), 2) if mc else 1.0
                return st in (1.65, 1.75, 1.85) and cp in (0.9, 1.0, 1.1)
            tr = sorted(set(refs_all) | {r_ for r_ in set(db2["ref"]) if allowed(r_)})
            json.dump(tr, open(os.path.join(a.out, "train_refs_bodies.json"), "w"))
            md = os.path.join(a.out, "models_bodies")
            if len(glob.glob(os.path.join(md, "models_seed*.pt"))) < 2:
                run([PY, "scripts/exp/train_fields.py", "--data", a.data, "--extra", dd, "--out", md, "--seeds", "0", "1", "--epochs", str(a.epochs), "--holdout-m", "-1",
                     "--train-refs", os.path.join(a.out, "train_refs_bodies.json")], log)
            fc = fit_by_condition(md, a.data, [dd])
            if len(fc):
                fc.to_csv(os.path.join(a.out, "fit_by_condition_bodies.csv"), index=False)
                print(fc.groupby(["model", "stature", "cap_scale"])[["r2_V", "rmse_tau", "rho_p"]].mean().round(3).to_string()[:4000])
            # closed loop on the new bodies: correct conditioning vs wrong conditioning (model error), SMPC only, 3 perturbations
            mf = sorted(glob.glob(os.path.join(md, "models_seed*.pt")))
            cf = os.path.join(a.out, "closedloop_bodies.csv")
            if os.path.exists(cf):                                  # rows of the earlier sequential run: give them task keys
                old = pd.read_csv(cf)
                if "task_key" not in old.columns and len(old):
                    old["task_key"] = [f"{b}_w{int(w)}_s{int(sd)}" for b, w, sd in zip(old["body"], old["wrong_condition"], old["seed"])]
                    old.to_csv(cf, index=False)
            if mf and okf:
                tasks = [dict(task_key=f"{os.path.basename(f)}_w{int(wrong)}_s{sd}", ref=f, wrong=wrong, seed=sd, models_file=mf[0])
                         for f in okf for wrong in (False, True) for sd in range(3)]
                dc = pool_map(closedloop_task, tasks, a.workers, cf, "exp8-closedloop")
                if len(dc):
                    print(dc.groupby(["wrong_condition"])[[f"succ_{lv:g}" for lv in F_MAX_LEVELS] + ["released", "caught"]].mean().round(3).to_string())
    # ---- 8c: envelope transfer -----------------------------------------------------------------------------------
    if "c" in parts:
        ref = os.path.join(a.grid, "sol_ref_T18_m66_phi0.250.pkl")
        tasks = []
        for s in (0.99, 1.01, 0.95, 1.05, 0.9, 1.1, 0.8, 1.2):
            tasks.append(dict(task_key=f"tau{s}", name=f"tau_cap x{s}", ref=ref, ov=dict(tau_scale=(s, s, s, s)), dlnp_tau=float(np.log(s)), dlnp_mu=0.0,
                              fn=os.path.join(a.out, "env", f"tau{s}.pkl"), out=a.out, max_cpu=a.max_cpu))
        for s in (0.9, 1.1, 0.8, 1.2):
            tasks.append(dict(task_key=f"mu{s}", name=f"mu x{s}", ref=ref, ov=dict(mu_scale=s), dlnp_tau=0.0, dlnp_mu=float(np.log(s)),
                              fn=os.path.join(a.out, "env", f"mu{s}.pkl"), out=a.out, max_cpu=a.max_cpu))
        de = pool_map(envelope_task, tasks, a.workers, os.path.join(a.out, "envelope.csv"), "exp8-envelope")
        if len(de):
            print(de[["variant", "ok", "J_base", "J_resolve", "J_pred_envelope", "err_envelope", "err_none"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
