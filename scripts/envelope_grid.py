"""Lower envelope of the release-phase curve: multi-start + warm-start continuation.

The NLP is non-convex, so a single cold start per phase gives only an UPPER bound of the required
capacity. This script tightens the bound for selected (T, m) conditions by

  1. `--cold K` additional cold starts per phase with different swing guesses (d_s, swing_amp),
  2. repeated continuation passes along phi (forward and backward, cyclic), each case being
     re-solved from its neighbour's solution and replaced whenever U_peak improves,

until a pass improves nothing by more than `--tol` (relative). The pickles in `--grid` are updated in
place (same file names) so that all downstream scripts see the envelope; a log records every
improvement and the smoothness of the curve before / after.

usage: python scripts/envelope_grid.py --grid results/grid --tag ref --T 16 19 20 24 --m 60 66 75 \
           --cold 3 --passes 4 --workers 22
"""
import argparse, glob, json, os, pickle, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.anthro import make_body
from katsumi.planar.nlp import PlanarNLP, ReducedParams

COLD_GUESSES = [(2.5, 0.4), (3.5, 0.8), (4.5, 0.6), (5.5, 0.5), (2.0, 0.9), (4.0, 0.3)]


def sol_path(grid, tag, T, m, ph):
    return os.path.join(grid, f"sol_{tag}_T{T:g}_m{m:g}_phi{ph:.3f}.pkl")


def load_cond(grid, tag, T, m):
    out = {}
    for f in glob.glob(os.path.join(grid, f"sol_{tag}_T{T:g}_m{m:g}_phi*.pkl")):
        r = pickle.load(open(f, "rb"))
        out[round(r["phi_l"], 4)] = r
    return out


def make_nlp(r):
    kw = dict(r["params"])
    for k in ("fixed_release_phase", "wait_in_cost"):
        kw.pop(k, None)
    p = ReducedParams(fixed_release_phase=r["phi_l"], wait_in_cost=False, **kw)
    body = make_body(r["m"], **r.get("body_kw", {}))
    return PlanarNLP(body, r["T"], r["phi_l"], p)


def _finish(new, r, kind):
    new["param_tag"] = r.get("param_tag", "ref")
    new["body_kw"] = r.get("body_kw", {})
    new["start_kind"] = kind
    return new


def cold_job(args):
    f, r, d_s, amp = args
    t0 = time.time()
    nlp = make_nlp(r)
    nlp.set_initial(d_w=max(0.0, r["T"] - d_s), d_s=d_s, swing_amp=amp)
    new = nlp.solve(print_level=0, max_iter=2500, tol=1e-4, max_cpu_time=900)
    new["solve_s"] = time.time() - t0
    new["cold_start"] = 1
    return f, _finish(new, r, f"cold d_s={d_s} amp={amp}")


def warm_job(args):
    f, r, prev, kind = args
    t0 = time.time()
    nlp = make_nlp(r)
    try:
        nlp.set_initial(prev=prev)
    except Exception:
        nlp.set_initial(d_w=r["T"] - prev["d_s"], d_s=prev["d_s"])
    new = nlp.solve(print_level=0, max_iter=2500, tol=1e-4, max_cpu_time=900)
    new["solve_s"] = time.time() - t0
    new["cold_start"] = 0
    return f, _finish(new, r, kind)


def better(new, old, tol):
    if not new.get("ok"):
        return False
    if not old.get("ok"):
        return True
    return new["U_peak"] < old["U_peak"] * (1 - tol)


def roughness(sols):
    """mean |second difference| / mean value over the cyclic phase curve (converged cases only)."""
    phs = sorted(sols)
    u = np.array([sols[p]["U_peak"] if sols[p].get("ok") else np.nan for p in phs])
    if np.isnan(u).any() or len(u) < 3:
        return float("nan")
    d2 = u[(np.arange(len(u)) + 1) % len(u)] - 2 * u + u[np.arange(len(u)) - 1]
    return float(np.mean(np.abs(d2)) / np.mean(u))


def continuation_pass(cond, sols, grid, tag, direction, tol, log):
    """Sequential cyclic pass over the phases of one condition (runs inside one worker)."""
    T, m = cond
    phs = sorted(sols)
    order = phs if direction > 0 else phs[::-1]
    # start from the best converged case so that the first warm start is a good one
    k0 = int(np.argmin([sols[p]["U_peak"] if sols[p].get("ok") else 9 for p in order]))
    order = order[k0:] + order[:k0]
    n_imp = 0
    prev = sols[order[0]]
    for ph in order[1:] + [order[0]]:
        r = sols[ph]
        if not prev.get("ok"):
            prev = r if r.get("ok") else prev
            continue
        f = sol_path(grid, tag, T, m, ph)
        _, new = warm_job((f, r, prev, f"continuation {'fwd' if direction > 0 else 'bwd'}"))
        if better(new, r, tol):
            log.append(dict(T=T, m=m, phi=ph, old=r.get("U_peak"), new=new["U_peak"], kind=new["start_kind"]))
            pickle.dump(new, open(f, "wb"))
            sols[ph] = new
            n_imp += 1
            print(f"  [{T:g},{m:g}] phi={ph:.4f} {r.get('U_peak')} -> {new['U_peak']:.4f} ({new['start_kind']})", flush=True)
        prev = sols[ph] if sols[ph].get("ok") else prev
    return n_imp


def pass_worker(args):
    cond, grid, tag, direction, tol = args
    sols = load_cond(grid, tag, *cond)
    log = []
    n = continuation_pass(cond, sols, grid, tag, direction, tol, log)
    return cond, n, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--T", type=float, nargs="+", default=[16, 19, 20, 24])
    ap.add_argument("--m", type=float, nargs="+", default=[60, 66, 75])
    ap.add_argument("--cold", type=int, default=3, help="extra cold starts per phase")
    ap.add_argument("--passes", type=int, default=4, help="max continuation passes (alternating direction)")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    conds = [(T, m) for T in a.T for m in a.m]
    logpath = a.log or os.path.join(a.grid, f"envelope_{a.tag}.json")
    log = dict(conditions=conds, cold=a.cold, improvements=[], roughness_before={}, roughness_after={}, passes=[])
    before = {}
    for c in conds:
        s = load_cond(a.grid, a.tag, *c)
        before[c] = {ph: (r["U_peak"] if r.get("ok") else None) for ph, r in s.items()}
        log["roughness_before"][f"{c[0]:g},{c[1]:g}"] = roughness(s)
    # ---- 1. multi-start (embarrassingly parallel) --------------------------------------------------
    if a.cold > 0:
        jobs = []
        for c in conds:
            s = load_cond(a.grid, a.tag, *c)
            for ph, r in s.items():
                for (d_s, amp) in COLD_GUESSES[:a.cold]:
                    jobs.append((sol_path(a.grid, a.tag, c[0], c[1], ph), r, d_s, amp))
        print(f"multi-start: {len(jobs)} cold solves", flush=True)
        n_imp = 0
        with mp.Pool(a.workers) as pool:
            for f, new in pool.imap_unordered(cold_job, jobs):
                old = pickle.load(open(f, "rb"))
                if better(new, old, a.tol):
                    n_imp += 1
                    log["improvements"].append(dict(T=new["T"], m=new["m"], phi=round(new["phi_l"], 4), old=old.get("U_peak"),
                                                    new=new["U_peak"], kind=new["start_kind"]))
                    pickle.dump(new, open(f, "wb"))
                    print(f"  improved {os.path.basename(f)}: {old.get('U_peak')} -> {new['U_peak']:.4f} ({new['start_kind']})", flush=True)
        log["passes"].append(dict(kind="multistart", improved=n_imp))
        json.dump(log, open(logpath, "w"), indent=1, default=float)
    # ---- 2. continuation passes (conditions in parallel, phases sequential) ---------------------------
    direction = 1
    for ps in range(a.passes):
        with mp.Pool(min(a.workers, len(conds))) as pool:
            res = pool.map(pass_worker, [(c, a.grid, a.tag, direction, a.tol) for c in conds])
        n_tot = sum(n for _, n, _ in res)
        for _, _, lg in res:
            log["improvements"].extend(lg)
        log["passes"].append(dict(kind=f"continuation {'fwd' if direction > 0 else 'bwd'}", improved=n_tot))
        print(f"pass {ps} ({'fwd' if direction > 0 else 'bwd'}): {n_tot} improvements", flush=True)
        json.dump(log, open(logpath, "w"), indent=1, default=float)
        direction = -direction
        if n_tot == 0 and ps >= 1:
            break
    after = {}
    for c in conds:
        s = load_cond(a.grid, a.tag, *c)
        after[c] = {ph: (r["U_peak"] if r.get("ok") else None) for ph, r in s.items()}
        log["roughness_after"][f"{c[0]:g},{c[1]:g}"] = roughness(s)
    rel = []
    for c in conds:
        for ph in before[c]:
            b0, a0 = before[c][ph], after[c].get(ph)
            if b0 and a0:
                rel.append((b0 - a0) / b0)
    log["summary"] = dict(n_cases=len(rel), n_improved=int(np.sum(np.array(rel) > a.tol)),
                          mean_rel_decrease=float(np.mean(rel)) if rel else None,
                          max_rel_decrease=float(np.max(rel)) if rel else None)
    json.dump(log, open(logpath, "w"), indent=1, default=float)
    print(json.dumps(log["summary"], indent=1))
    # regenerate the row CSVs for the touched conditions
    Ts = " ".join(f"{T:g}" for T in a.T); ms = " ".join(f"{m:g}" for m in a.m)
    os.system(f"{sys.executable} scripts/run_grid.py --out {a.grid} --tag {a.tag} --workers {a.workers} --T {Ts} --m {ms} --nphi 16 > /dev/null 2>&1")


if __name__ == "__main__":
    main()
