"""Dual variables / shadow prices of the capability parameters for saved grid solutions.

For every selected solution the NLP is re-solved from the saved solution (warm start, tight tolerance) with
the capability parameters exposed as Opti parameters; the result stores
  * the multiplier mass per constraint category (what binds),
  * the time-resolved multipliers of the capacity epigraph constraints (WHEN the grip capacity binds:
    swing / catch / hold),
  * dJ*/d ln p for p in {tau_cap (elbow, shoulder, hip, knee), mu_out, mu_in, v_rel_max, v_away_max, qd_max}.
Optionally (`--fd`) the sensitivities are validated by finite differences (+10 % re-solves).

usage: python scripts/run_duals.py --grid results/grid --tag ref --T 20 --m 66 --out results/duals --workers 8 [--fd]
"""
import argparse, glob, json, os, pickle, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.anthro import make_body
from katsumi.planar.nlp import PlanarNLP, ReducedParams

FD_PARAMS = [("tau_cap", 0), ("tau_cap", 1), ("tau_cap", 2), ("tau_cap", 3), ("mu_out", None), ("v_rel_max", None),
             ("qd_max", None)]


def params_from(r, **over):
    kw = dict(r["params"])
    for k in ("fixed_release_phase", "wait_in_cost"):
        kw.pop(k, None)
    kw = {k: v for k, v in kw.items() if k in ReducedParams.__dataclass_fields__}
    kw.update(over)
    return ReducedParams(fixed_release_phase=r["phi_l"], wait_in_cost=False, **kw)


def job(args):
    f, outdir, fd, step = args
    r = pickle.load(open(f, "rb"))
    body = make_body(r["m"], **r.get("body_kw", {}))
    t0 = time.time()
    nlp = PlanarNLP(body, r["T"], r["phi_l"], params_from(r))
    nlp.set_initial(prev=r)
    base = nlp.solve(print_level=0, max_iter=2000, tol=1e-6, max_cpu_time=1500, sensitivities=True)
    base["solve_s"] = time.time() - t0
    base["param_tag"] = r.get("param_tag", "ref")
    base["body_kw"] = r.get("body_kw", {})
    res = dict(file=os.path.basename(f), T=r["T"], m=r["m"], phi_l=r["phi_l"], ok=base["ok"], U_peak=base["U_peak"],
               U_peak_saved=r["U_peak"], effort=base["effort"], duals=base.get("duals"))
    if fd and base["ok"]:
        J0 = nlp.p.w_U * base["U_peak"] + nlp.p.w_E * base["effort"]
        fdres = {}
        for name, idx in FD_PARAMS:
            b2 = make_body(r["m"], **r.get("body_kw", {}))
            over = {}
            if name == "tau_cap":
                b2.tau_cap = b2.tau_cap.copy(); b2.tau_cap[idx] *= (1 + step)
            elif name == "mu_out":
                b2.mu_out *= (1 + step)
            elif name == "qd_max":
                b2.qd_max *= (1 + step)
            elif name == "v_rel_max":
                over["v_rel_max"] = r["params"]["v_rel_max"] * (1 + step)
            n2 = PlanarNLP(b2, r["T"], r["phi_l"], params_from(r, **over))
            n2.set_initial(prev=base)
            o2 = n2.solve(print_level=0, max_iter=3000, tol=1e-6, max_cpu_time=1500)
            J1 = nlp.p.w_U * o2["U_peak"] + nlp.p.w_E * o2["effort"]
            key = name if idx is None else f"{name}[{idx}]"
            fdres[key] = dict(ok=o2["ok"], dU_dlnp=(o2["U_peak"] - base["U_peak"]) / np.log(1 + step),
                              dJ_dlnp=(J1 - J0) / np.log(1 + step),
                              dJ_dlnp_dual=float(np.ravel(base["duals"]["dJ_dlnp"][name])[idx if idx is not None else 0]))
        res["fd"] = fdres
    pickle.dump(base, open(os.path.join(outdir, os.path.basename(f).replace("sol_", "dual_")), "wb"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--T", type=float, nargs="+", default=[20])
    ap.add_argument("--m", type=float, nargs="+", default=[66])
    ap.add_argument("--phis", type=float, nargs="*", default=None)
    ap.add_argument("--out", default="results/duals")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--step", type=float, default=0.10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = []
    for T in a.T:
        for m in a.m:
            fs = sorted(glob.glob(os.path.join(a.grid, f"sol_{a.tag}_T{T:g}_m{m:g}_phi*.pkl")))
            if a.phis:
                fs = [f for f in fs if any(abs(float(f.split("_phi")[1][:-4]) - ph) < 2e-3 for ph in a.phis)]
            files += fs
    print(len(files), "solutions", flush=True)
    rows = []
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(job, [(f, a.out, a.fd, a.step) for f in files]):
            rows.append(res)
            d = res.get("duals") or {}
            cm = {k: round(sum(v), 3) for k, v in d.get("cap_mult", {}).items()}
            print(res["file"], res["ok"], round(res["U_peak"], 4), "cap_mult", cm,
                  "dJ/dlnp", {k: [round(float(x), 3) for x in np.ravel(v)] for k, v in d.get("dJ_dlnp", {}).items()},
                  ("FD " + json.dumps(res["fd"], default=float)) if "fd" in res else "", flush=True)
            json.dump(rows, open(os.path.join(a.out, f"duals_{a.tag}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
