"""Experiment 9.1: completion of the 864-case grid by continuation from lattice neighbours.

The grid runs sweep the release phase of one (T, m) condition as a chain; once the chain passes the unreachable phases
around the far point of cliff B it stays in a high-load basin (U_peak ~ 3) for the rest of the cycle, although the same
phase is solved with U_peak ~ 1.1 for a neighbouring period. This script re-solves every case whose required capacity
exceeds that of a lattice neighbour by more than `--tol` (or that failed), warm-started from that neighbour:

    neighbours of (T, m, phi) = the same (m, phi) at any other period, the two adjacent phases (cyclic), the two
    adjacent masses.

A solution is replaced only when the re-solve converges with a lower U_peak. Each (case, neighbour) pair is tried once;
passes are repeated until nothing is left. The pre-polish solutions are kept in `--backup` and the log records every
replacement, so that the effect of the continuation is reported next to the result.

usage: python scripts/exp/exp9_polish.py --grid results/grid --tag ref --workers 22 --passes 8
"""
import argparse, glob, json, os, pickle, re, shutil, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.planar.anthro import make_body
from katsumi.planar.nlp import PlanarNLP, ReducedParams

PAT = re.compile(r"sol_(?P<tag>\w+?)_T(?P<T>[0-9.]+)_m(?P<m>[0-9.]+)_phi(?P<phi>[0-9.]+)\.pkl$")


def load_all(grid, tag):
    """{(T, m, phi_target)} -> (file, ok, U_peak); the key uses the target phase of the file name."""
    sols = {}
    for f in glob.glob(os.path.join(grid, f"sol_{tag}_T*_m*_phi*.pkl")):
        mm = PAT.search(os.path.basename(f))
        if not mm:
            continue
        try:
            r = pickle.load(open(f, "rb"))
        except Exception:
            continue
        sols[(float(mm["T"]), float(mm["m"]), float(mm["phi"]))] = (f, bool(r.get("ok")), float(r.get("U_peak", np.inf)))
    return sols


def resolve(job):
    f, f_nb, phi, max_cpu = job
    r = pickle.load(open(f, "rb")); prev = dict(pickle.load(open(f_nb, "rb")))
    kw = dict(r["params"])
    for k in ("fixed_release_phase", "wait_in_cost"):
        kw.pop(k, None)
    body_kw = r.get("body_kw", {}) or {}
    p = ReducedParams(fixed_release_phase=phi, wait_in_cost=False, **kw)
    body = make_body(r["m"], **body_kw)
    t0 = time.time()
    nlp = PlanarNLP(body, r["T"], phi, p)
    # the neighbour may belong to another period / phase: keep its swing and shift the wait so that the release phase fits
    try:
        prev["d_w"] = max(r["T"] - prev["d_s"], 0.0)          # the problem starts one period before the release phase
        nlp.set_initial(prev=prev)
    except Exception:
        nlp.set_initial(d_w=max(r["T"] - prev["d_s"], 0.0), d_s=prev["d_s"])
    new = nlp.solve(print_level=0, max_iter=2500, tol=1e-4, max_cpu_time=max_cpu)
    new["solve_s"] = time.time() - t0; new["cold_start"] = 0; new["param_tag"] = r.get("param_tag", "ref"); new["body_kw"] = body_kw
    new["polished_from"] = os.path.basename(f_nb)
    old_ok, old_U = bool(r.get("ok")), float(r.get("U_peak", np.inf))
    improved = bool(new["ok"]) and ((not old_ok) or new["U_peak"] < old_U - 1e-4)
    if improved:
        pickle.dump(new, open(f, "wb"))
    return dict(file=os.path.basename(f), frm=os.path.basename(f_nb), old_ok=int(old_ok), old_U=old_U, new_ok=int(bool(new["ok"])), new_U=float(new["U_peak"]),
                status=new["status"], improved=int(improved), solve_s=round(new["solve_s"], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid"); ap.add_argument("--tag", default="ref")
    ap.add_argument("--backup", default="results/grid_prepolish"); ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--passes", type=int, default=8); ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--max-cpu", type=float, default=900.0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nb-max", type=float, default=2.0, help="only neighbours with U_peak below this value are used as warm starts (a low-load "
                                                             "solution is propagated; neighbours that sit in the high-load basin themselves are not)")
    a = ap.parse_args()
    if a.backup and not os.path.exists(a.backup):
        shutil.copytree(a.grid, a.backup)
    log_fn = os.path.join(a.grid, "polish_log.jsonl"); tried_fn = os.path.join(a.grid, "polish_tried.json")
    tried = set(tuple(x) for x in json.load(open(tried_fn))) if os.path.exists(tried_fn) else set()
    for ps in range(a.passes):
        sols = load_all(a.grid, a.tag)
        Ts = sorted({k[0] for k in sols}); ms = sorted({k[1] for k in sols}); phis = sorted({k[2] for k in sols})
        jobs = []
        for (T, m, ph), (f, ok, U) in sols.items():
            i, j = phis.index(ph), ms.index(m)
            nb = [(T2, m, ph) for T2 in Ts if T2 != T]
            nb += [(T, m, phis[(i - 1) % len(phis)]), (T, m, phis[(i + 1) % len(phis)])]
            nb += [(T, ms[j2], ph) for j2 in (j - 1, j + 1) if 0 <= j2 < len(ms)]
            cand = [(sols[k][2], sols[k][0]) for k in nb if k in sols and sols[k][1] and sols[k][2] <= a.nb_max]
            cand = [c for c in sorted(cand) if ((not ok) or U > (1 + a.tol) * c[0]) and (os.path.basename(f), os.path.basename(c[1])) not in tried]
            if cand:
                jobs.append((f, cand[0][1], ph, a.max_cpu))
        if a.limit:
            jobs = jobs[:a.limit]
        n_ok = sum(1 for v in sols.values() if v[1])
        print(f"pass {ps}: {len(jobs)} cases to re-solve of {len(sols)} ({n_ok} converged)", flush=True)
        if not jobs:
            break
        n_imp = 0
        with mp.get_context("forkserver").Pool(a.workers) as pool:
            for res in pool.imap_unordered(resolve, jobs):
                tried.add((res["file"], res["frm"])); n_imp += res["improved"]
                res["pass"] = ps
                with open(log_fn, "a") as fl:
                    fl.write(json.dumps(res) + "\n")
                json.dump(sorted(tried), open(tried_fn, "w"))
                print(f"  {res['file']} <- {res['frm']}: U {res['old_U']:.3f} -> {res['new_U']:.3f} ok={res['new_ok']} improved={res['improved']} {res['solve_s']}s", flush=True)
        print(f"pass {ps}: {n_imp} of {len(jobs)} improved", flush=True)
    print("polish finished", flush=True)


if __name__ == "__main__":
    main()
