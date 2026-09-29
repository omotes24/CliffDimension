"""Polish a release-phase grid: re-solve local-optimum outliers with warm starts from their phase
neighbours (cold-started parallel runs occasionally land in a worse basin).

A case is an outlier if it failed or if U_peak exceeds the better neighbour (cyclic in phi) by more than
`--tol` (relative). It is re-solved from the better neighbour's solution and replaced when improved.

usage: python scripts/polish_grid.py --grid results/grid --tag ref --workers 20 --passes 2
"""
import argparse, glob, os, pickle, re, sys, json, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.anthro import make_body
from katsumi.planar.nlp import PlanarNLP, ReducedParams


def load_all(grid, tag):
    sols = {}
    for f in glob.glob(os.path.join(grid, f"sol_{tag}_T*_m*_phi*.pkl")):
        r = pickle.load(open(f, "rb"))
        sols[(r["T"], r["m"], round(r["phi_l"], 4))] = (f, r)
    return sols


def body_kwargs(r):
    """Reconstruct make_body kwargs from a saved solution (params saved under 'body_kw' when available)."""
    return r.get("body_kw", {})


def resolve(args):
    f, r, prev, tag = args
    kw = dict(r["params"])
    for k in ("fixed_release_phase", "wait_in_cost"):
        kw.pop(k, None)
    p = ReducedParams(fixed_release_phase=r["phi_l"], wait_in_cost=False, **kw)
    body = make_body(r["m"], **body_kwargs(r))
    t0 = time.time()
    nlp = PlanarNLP(body, r["T"], r["phi_l"], p)
    try:
        nlp.set_initial(prev=prev)
    except Exception:
        nlp.set_initial(d_w=r["T"] - prev["d_s"], d_s=prev["d_s"])
    new = nlp.solve(print_level=0, max_iter=2500, tol=1e-4, max_cpu_time=900)
    new["solve_s"] = time.time() - t0
    new["cold_start"] = 0
    new["param_tag"] = tag
    new["polished"] = True
    improved = new["ok"] and (not r.get("ok") or new["U_peak"] < r["U_peak"] - 1e-4)
    if improved:
        pickle.dump(new, open(f, "wb"))
    return (os.path.basename(f), r.get("U_peak"), new["U_peak"], new["ok"], improved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args()
    for ps in range(a.passes):
        sols = load_all(a.grid, a.tag)
        jobs = []
        by_cond = {}
        for (T, m, ph), (f, r) in sols.items():
            by_cond.setdefault((T, m), []).append((ph, f, r))
        for (T, m), lst in by_cond.items():
            lst.sort()
            n = len(lst)
            for i, (ph, f, r) in enumerate(lst):
                nb = [lst[(i - 1) % n], lst[(i + 1) % n]]
                nb_ok = [x for x in nb if x[2].get("ok")]
                if not nb_ok:
                    continue
                best_nb = min(nb_ok, key=lambda x: x[2]["U_peak"])
                if (not r.get("ok")) or r["U_peak"] > (1 + a.tol) * best_nb[2]["U_peak"]:
                    jobs.append((f, r, best_nb[2], a.tag))
        print(f"pass {ps}: {len(jobs)} outliers of {len(sols)}", flush=True)
        if not jobs:
            break
        with mp.Pool(a.workers) as pool:
            for res in pool.imap_unordered(resolve, jobs):
                print(res, flush=True)
    # regenerate row CSVs
    os.system(f"{sys.executable} scripts/run_grid.py --out {a.grid} --tag {a.tag} --workers {a.workers} "
              f"--T {' '.join(str(t) for t in sorted(set(k[0] for k in load_all(a.grid, a.tag))))} "
              f"--m {' '.join(str(m) for m in sorted(set(k[1] for k in load_all(a.grid, a.tag))))} --nphi 16 > /dev/null 2>&1")


if __name__ == "__main__":
    main()
