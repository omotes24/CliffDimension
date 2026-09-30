"""Oracle dataset for dual-field learning: cost-to-go solves from states along / around reference trajectories.

For every converged reference solution (results/<grid>/sol_<tag>_T*_m*_phi*.pkl) the swing is sampled at `--stride`
knots; at each sampled time t_k the swing state x_k is perturbed with `--levels` noise scales (0 = on the reference)
and a from-state NLP (free release time, no wait) is solved, warm-started from the tail of the reference. Recorded
per sample: the value J* (optimiser objective), U*, the remaining swing time d_s* (time-to-release), the costate
dJ*/dx0, dJ*/dt0, the multiplier mass of every constraint category, the capacity (epigraph) multipliers per phase,
and the first controls u*(t0). Solutions are pickled (--out/sol_*.pkl) and a flat table is written to --out/rows.csv.

usage (hades): python scripts/dual_field_data.py --refs "results/grid/sol_ref_T18_m*_phi0.250.pkl" --out results/dual_field \
                    --stride 10 --levels 0 0.5 1 2 --workers 12
"""
import argparse, glob, json, os, pickle, sys, time, traceback
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.anthro import make_body
from katsumi.planar.nlp import PlanarNLP, ReducedParams
from katsumi.planar.model import NTH

SIG_TH, SIG_THD = 0.06, 0.40        # perturbation scales per unit level [rad], [rad/s]
CATS = ["cone_A", "cone_B", "cone_C", "cone_impact", "joint_range", "joint_speed", "wall", "approach", "catch_geom", "catch_vel",
        "hook", "settle", "rate", "torque", "duration", "flight_inv"]


REL_LO = np.array([0.0, -4.2, -2.1, 0.0]); REL_HI = np.array([2.6, 0.60, 0.35, 2.6])     # facing -x (swing on A)
P_REL = np.array([[1, -1, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], float)   # th = th2*1 + P rel


def perturb_state(x, lv, rng, body, t0, T, eps, qd_max=10.0, margin=0.02):
    """Perturb a swing state in relative-joint coordinates (kept inside the optimiser's joint ranges, incl. the
    joint-speed limit) plus a whole-body rotation; reject perturbations that put a body point inside a wall."""
    from katsumi import device
    from katsumi.planar.model import PlanarChain
    th, thd = x[:NTH], x[NTH:]
    rel = np.array([th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]])
    reld = np.array([thd[0] - thd[1], thd[2] - thd[1], thd[3] - thd[2], thd[4] - thd[3]])
    chain = PlanarChain(body)
    dv = device.device_state(t0, T, eps)
    for _ in range(20):
        rel_n = np.clip(rel + rng.normal(0, SIG_TH * lv, 4), REL_LO + margin, REL_HI - margin)
        reld_n = np.clip(reld + rng.normal(0, SIG_THD * lv, 4), -0.9 * qd_max, 0.9 * qd_max)
        th2 = th[2] + rng.normal(0, SIG_TH * lv); thd2 = thd[2] + rng.normal(0, SIG_THD * lv)
        th_n = th2 + P_REL @ rel_n; thd_n = thd2 + P_REL @ reld_n
        q = np.concatenate([dv["pA"], th_n])
        P = np.array(chain.f_tips(q)); Pf = np.array(chain.f_forearm(q))
        pts = np.hstack([q[0:2, None], P, Pf])
        clear = np.concatenate([[0.0], body.clearance, body.forearm_clearance])
        ok = True
        for i in range(pts.shape[1]):
            xx, yy = pts[0, i], pts[1, i]
            aA = xx - (-device.D_LEDGE + clear[i]); bA = (dv["h"] - device.WALL_BOTTOM_OFFSET) - yy
            aB = (dv["x"] + device.D_LEDGE - clear[i]) - xx; bB = -device.WALL_BOTTOM_OFFSET - yy
            if max(aA, bA) < 0.005 or max(aB, bB) < 0.005:
                ok = False
        if ok:
            return np.concatenate([th_n, thd_n])
    return None


def sample_jobs(ref_files, stride, levels, n_rep, seed, t_margin):
    rng = np.random.default_rng(seed)
    jobs = []
    for f in ref_files:
        r = pickle.load(open(f, "rb"))
        if not r.get("ok") or r["U_peak"] > 2.5:
            continue
        body = make_body(r["m"], **(r.get("body_kw", {}) or {}))
        N = r["S_X"].shape[1] - 1
        tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
        ks = list(range(0, N + 1, stride))
        ks = [k for k in ks if r["d_s"] - (tS[k] - r["t_s0"]) > t_margin]     # leave >= t_margin s before the release
        for k in ks:
            for lv in levels:
                reps = 1 if lv == 0 else n_rep
                for j in range(reps):
                    if lv > 0:
                        xn = perturb_state(r["S_X"][:, k], lv, rng, body, float(tS[k]), r["T"], r["params"]["eps"], body.qd_max)
                        if xn is None:
                            continue
                        d = xn - r["S_X"][:, k]
                    else:
                        d = np.zeros(2 * NTH)
                    jobs.append(dict(ref=f, k=int(k), level=float(lv), rep=j, delta=d.tolist()))
    return jobs


def job(args):
    jb, outdir, max_cpu, N_min = args
    r = pickle.load(open(jb["ref"], "rb"))
    T, m = r["T"], r["m"]
    N = r["S_X"].shape[1] - 1
    k = jb["k"]
    tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
    if "x0" in jb:                                   # visited state (DAgger): explicit state / time, warm start from knot k
        t0 = float(jb["t0"]); x0 = np.array(jb["x0"], float)
        rem = float(jb.get("rem", r["d_s"] - (tS[k] - r["t_s0"])))
        tag = jb.get("tag", f"{os.path.basename(jb['ref'])[4:-4]}_v{jb['rep']:04d}")
    else:
        t0 = float(tS[k]); x0 = r["S_X"][:, k] + np.array(jb["delta"])
        rem = r["d_s"] - (t0 - r["t_s0"])
        tag = f"{os.path.basename(jb['ref'])[4:-4]}_k{k:03d}_l{jb['level']:g}_r{jb['rep']}"
    fn = os.path.join(outdir, f"sol_{tag}.pkl")
    row = dict(tag=tag, ref=os.path.basename(jb["ref"]), T=T, m=m, k=k, level=jb["level"], rep=jb["rep"], t0=t0, phase=(t0 % T) / T,
               t_in=t0 - r["t_s0"], rem_ref=rem, U_ref=r["U_peak"], ok=0)
    for i in range(2 * NTH):
        row[f"x{i}"] = float(x0[i])
    body_kw = r.get("body_kw", {}) or {}
    row["stature"] = body_kw.get("stature", 1.75); row["cap_scale"] = body_kw.get("cap_scale", 1.0)
    try:
        if os.path.exists(fn):
            out = pickle.load(open(fn, "rb"))
        else:
            kw = dict(r["params"])
            for q in ("fixed_release_phase", "wait_in_cost", "from_state", "x0", "N_S", "d_s_bounds"):
                kw.pop(q, None)
            N_S = int(np.clip(round(N * rem / r["d_s"]) + 10, N_min, 180))
            p = ReducedParams(fixed_release_phase=None, wait_in_cost=False, from_state=True, x0=tuple(x0), N_S=N_S,
                              d_s_bounds=(0.02, T), **kw)
            nlp = PlanarNLP(make_body(m, **body_kw), T, t0 / T, p)
            prev = dict(r)
            for key in ("S_X", "S_A", "S_U", "S_R", "S_Am", "S_Um", "S_Rm"):
                if key in r:
                    prev[key] = r[key][:, k:]
            prev["d_s"] = rem; prev["d_w"] = 0.0
            nlp.set_initial(prev=prev)
            t1 = time.time()
            out = nlp.solve(print_level=0, max_iter=2000, tol=1e-4, max_cpu_time=max_cpu, sensitivities=True)
            out["solve_s"] = time.time() - t1
            out["ref"] = jb["ref"]; out["k"] = k; out["level"] = jb["level"]; out["delta"] = jb.get("delta", (x0 - r["S_X"][:, k]).tolist()); out["body_kw"] = body_kw; out["t0_abs"] = t0
            # keep the pickle small: drop the mid-point arrays
            for key in list(out.keys()):
                if key.endswith("_Am") or key.endswith("_Um") or key.endswith("_Rm"):
                    out.pop(key)
            pickle.dump(out, open(fn, "wb"))
        row.update(ok=int(out["ok"]), status=out["status"], iters=out["iters"], solve_s=round(out.get("solve_s", 0), 1),
                   J=out["J"], U=out["U_peak"], d_s=out["d_s"], d_f=out["d_f"], effort=out["effort"], U_catch=out["U_catch"])
        for i in range(4):
            row[f"u{i}"] = float(out["S_U"][i, 0])
        d = out.get("duals") or {}
        if d and "costate_x0" in d:
            for i, v in enumerate(d["costate_x0"]):
                row[f"p{i}"] = v
            row["dJ_dt0"] = d["dJ_dt0"]
            cats = d.get("categories", {})
            for c in CATS:
                row["lam_" + c] = cats.get(c, {}).get("abs_sum", 0.0)
            for c, v in d.get("cap_mult", {}).items():
                row["cap_" + c] = sum(v)
            for name, v in d.get("dJ_dlnp", {}).items():
                row["sens_" + name] = sum(v)
    except Exception:
        traceback.print_exc()
        row["status"] = "exception"
    print(f"[{tag}] ok={row['ok']} U={row.get('U', float('nan')):.3f} d_s={row.get('d_s', float('nan')):.2f} "
          f"J={row.get('J', float('nan')):.3f} it={row.get('iters', '')} {row.get('solve_s', '')}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", default=[], help="glob(s) of reference solution pickles")
    ap.add_argument("--states", default=None, help="JSON list of visited states {ref, t0, x0, k, rem, rep, level} (DAgger)")
    ap.add_argument("--out", default="results/dual_field")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--reps", type=int, default=1, help="perturbation draws per (knot, level > 0)")
    ap.add_argument("--t-margin", type=float, default=0.15, help="skip samples closer than this to the reference release [s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cpu", type=float, default=900)
    ap.add_argument("--N-min", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--table", default="rows.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(set(sum([glob.glob(g) for g in a.refs], [])))
    jobs = json.load(open(a.states)) if a.states else sample_jobs(files, a.stride, a.levels, a.reps, a.seed, a.t_margin)
    if not a.no_shuffle:                      # interleave references so that partial tables already cover all bodies
        np.random.default_rng(a.seed).shuffle(jobs)
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(files)} references, {len(jobs)} from-state solves", flush=True)
    json.dump(jobs, open(os.path.join(a.out, "jobs.json" if not a.states else "jobs_states.json"), "w"))
    rows = []
    import csv
    table = os.path.join(a.out, a.table)
    if os.path.exists(table):
        rows = list(csv.DictReader(open(table)))                 # existing rows are kept (one row per tag, see below)
    with mp.Pool(a.workers) as pool:
        for row in pool.imap_unordered(job, [(jb, a.out, a.max_cpu, a.N_min) for jb in jobs]):
            rows = [r for r in rows if r["tag"] != row["tag"]] + [row]          # one row per tag
            keys = sorted(set().union(*[r.keys() for r in rows]))
            with open(table, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("done", len(rows), "ok", sum(int(r["ok"]) for r in rows))


if __name__ == "__main__":
    main()
