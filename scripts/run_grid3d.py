"""Spatial (twisting, two-hand) reduced model: release-phase sweeps for a strategy.

A strategy = (grip width on A, release order, catch order). Every (T, m, phi) job is warm-started from the
planar solution of the same condition (results/grid) or, if `--init3d DIR` holds a spatial solution of the same
(T, m, phi) (any tag), from that one. Solutions are pickled to --out; rows are appended to a CSV.

usage: python scripts/run_grid3d.py --T 20 --m 66 --phis 0.25 0.6875 --grip 0.39 --release none --catch none \
           --tag wide_sim --out results/spatial --workers 2
"""
import argparse, csv, glob, json, os, pickle, sys, time, traceback
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.anthro import make_body, G
from katsumi.spatial.anthro3d import make_body3d
from katsumi.spatial.nlp3d import SpatialNLP, Params3D
from katsumi.spatial.model3d import NQ

STAGE_A_ITERS = 300

FIELDS = ["tag", "T", "m", "phi_l", "ok", "status", "iters", "solve_s", "U_peak", "req_hand_N", "req_hand_BW", "req_two_hand_BW",
          "effort", "d_s", "d_r", "d_f", "d_c", "dB", "yB", "grip_A", "release_first", "catch_first",
          "U_S_L", "U_S_R", "U_H_L", "U_H_R", "U_S1", "U_C1", "catch_L_first", "catch_R_first", "catch_L_hold", "catch_R_hold",
          "yaw_release", "yaw_catch", "yawrate_release", "Lz_release", "Lxy_release", "couple_impulse", "asym_swing", "asym_hold",
          "stature", "cap_scale", "init_from"]


def metrics(r, body3d, chain):
    """Derived quantities for the strategy analysis."""
    from katsumi import device
    out = {}
    fh = body3d.f_hand
    XS = r["S_X"]; RS = r["S_R"]
    q_rel = (r["S1_X"][:, -1] if "S1_X" in r else XS[:, -1])[:NQ]
    qd_rel = (r["S1_X"][:, -1] if "S1_X" in r else XS[:, -1])[NQ:]
    out["yaw_release"] = float(q_rel[3] - np.pi)
    out["yawrate_release"] = float(qd_rel[3])
    L = np.array(chain.f_Lg(q_rel, qd_rel)).ravel()
    out["Lz_release"] = float(L[2]); out["Lxy_release"] = float(np.hypot(L[0], L[1]))
    qc = r["F_X"][:NQ, -1]
    out["yaw_catch"] = float(qc[3] % (2 * np.pi))
    # contact couple about the vertical axis during the swing: sum (R_xR - R_xL) * dA/2 dt (N m s), last 1.0 s
    N = XS.shape[1] - 1; dt = r["d_s"] / N
    dA = r["params"]["grip_A"]
    k0 = max(0, int(N - 1.0 / dt))
    couple = np.sum((RS[3, k0:] - RS[0, k0:]) * fh * (dA / 2) * dt)
    out["couple_impulse"] = float(couple)
    UL = np.sqrt((RS[0:3] ** 2).sum(0)); UR = np.sqrt((RS[3:6] ** 2).sum(0))
    out["asym_swing"] = float(np.max(np.abs(UL - UR)) / max(np.max(np.maximum(UL, UR)), 1e-9))
    RH = r["H_R"]; UL = np.sqrt((RH[0:3] ** 2).sum(0)); UR = np.sqrt((RH[3:6] ** 2).sum(0))
    out["asym_hold"] = float(np.max(np.abs(UL - UR)) / max(np.max(np.maximum(UL, UR)), 1e-9))
    return out


def job(args):
    T, m, phi, grip, rel, cat, tag, outdir, init_dir, init3d, body_kw, maxit, max_cpu = args
    fn = os.path.join(outdir, f"sol3d_{tag}_T{T:g}_m{m:g}_phi{phi:.3f}.pkl")
    row = {k: "" for k in FIELDS}
    row.update(tag=tag, T=T, m=m, phi_l=round(phi, 4), grip_A=grip, release_first=rel or "", catch_first=cat or "",
               stature=body_kw.get("stature", 1.75), cap_scale=body_kw.get("cap_scale", 1.0))
    try:
        if os.path.exists(fn):
            r = pickle.load(open(fn, "rb"))
            body3d = make_body3d(m, **body_kw)
            nlp = None
        else:
            body3d = make_body3d(m, **body_kw)
            p = Params3D(fixed_release_phase=phi, grip_A=grip, release_first=rel, catch_first=cat)
            nlp = SpatialNLP(body3d, T, phi, p)
            init_from = "cold"
            prev = None
            if init3d:
                cands = sorted(glob.glob(os.path.join(init3d, f"sol3d_*_T{T:g}_m{m:g}_phi{phi:.3f}.pkl")))
                cands = [c for c in cands if pickle.load(open(c, "rb")).get("ok")]
                if cands:
                    # prefer the same tag, else any converged spatial solution of this condition
                    same = [c for c in cands if f"sol3d_{tag}_" in c]
                    prev = pickle.load(open((same or cands)[0], "rb")); init_from = os.path.basename((same or cands)[0])
            if prev is not None:
                nlp.set_initial_from_prev(prev)
            else:
                pf = os.path.join(init_dir, f"sol_ref_T{T:g}_m{m:g}_phi{phi:.3f}.pkl")
                if not os.path.exists(pf):
                    fs = sorted(glob.glob(os.path.join(init_dir, f"sol_ref_T{T:g}_m*_phi{phi:.3f}.pkl")))
                    pf = fs[0] if fs else None
                sol = pickle.load(open(pf, "rb"))
                nlp.set_initial_from_planar(sol, make_body(sol["m"]))
                init_from = os.path.basename(pf)
            t0 = time.time()
            # two-stage solve: L-BFGS (cheap iterations) to feasibility, then exact Hessian from that point
            if prev is None or not prev.get("ok"):
                rA = nlp.solve(print_level=0, max_iter=STAGE_A_ITERS, tol=1e-4, max_cpu_time=max_cpu / 3, hessian="limited-memory")
                nlp = SpatialNLP(body3d, T, phi, p)
                nlp.set_initial_from_prev(rA)
            r = nlp.solve(print_level=0, max_iter=maxit, tol=1e-4, max_cpu_time=max_cpu, hessian="exact")
            if not r["ok"]:                      # fall back: continue with L-BFGS from the best point so far
                nlp = SpatialNLP(body3d, T, phi, p)
                nlp.set_initial_from_prev(r)
                r2 = nlp.solve(print_level=0, max_iter=maxit, tol=1e-4, max_cpu_time=max_cpu, hessian="limited-memory")
                if r2["ok"] or r2["U_peak"] < r["U_peak"]:
                    r = r2
            r["solve_s"] = time.time() - t0
            r["tag"] = tag; r["init_from"] = init_from; r["body_kw"] = body_kw
            pickle.dump(r, open(fn, "wb"))
        ch = SpatialNLP.__new__(SpatialNLP)  # light chain access without rebuilding the NLP
        from katsumi.spatial.model3d import SpatialChain
        chain = SpatialChain(body3d)
        mg = m * G
        row.update(ok=int(r["ok"]), status=r["status"], iters=r["iters"], solve_s=round(r.get("solve_s", 0), 1), U_peak=r["U_peak"],
                   req_hand_N=r["U_peak"] * body3d.f_hand, req_hand_BW=r["U_peak"] * body3d.f_hand / mg,
                   req_two_hand_BW=2 * r["U_peak"] * body3d.f_hand / mg, effort=r["effort"], d_s=r["d_s"], d_r=r["d_r"], d_f=r["d_f"],
                   d_c=r["d_c"], dB=float(r["dB"]), yB=float(r["yB"]), init_from=r.get("init_from", ""))
        for k in ("U_S", "U_H"):
            if k in r:
                row[k + "_L"], row[k + "_R"] = r[k]
        if "U_S1" in r:
            row["U_S1"] = r["U_S1"][0]
        if "U_C1" in r:
            row["U_C1"] = r["U_C1"][0]
        cl = r.get("catch_loads", {})
        row.update(catch_L_first=cl.get("L_first", ""), catch_R_first=cl.get("R_first", ""), catch_L_hold=cl.get("L_hold", ""), catch_R_hold=cl.get("R_hold", ""))
        row.update(metrics(r, body3d, chain))
    except Exception:
        traceback.print_exc()
        row.update(ok=0, status="exception")
    print(f"[{tag}] T={T} m={m} phi={phi:.3f} ok={row['ok']} U_hand={row['U_peak']} ({row['req_two_hand_BW']} BW two-hand eq.) "
          f"d_r={row['d_r']} d_c={row['d_c']} dB={row['dB']} it={row['iters']} {row['solve_s']}s {row['status']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+", default=[20])
    ap.add_argument("--m", type=float, nargs="+", default=[66])
    ap.add_argument("--phis", type=float, nargs="+", default=[0.25, 0.6875])
    ap.add_argument("--grip", type=float, default=0.39)
    ap.add_argument("--release", default="none", choices=["none", "L", "R"])
    ap.add_argument("--catch", default="none", choices=["none", "L", "R"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/spatial")
    ap.add_argument("--init", default="results/grid", help="planar solutions for the initial guess")
    ap.add_argument("--init3d", default=None, help="directory with spatial solutions to warm start from")
    ap.add_argument("--body", default="{}", help="JSON kwargs for make_body3d (stature, cap_scale, arm_scale, ...)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--maxit", type=int, default=4000)
    ap.add_argument("--max-cpu", type=float, default=4 * 3600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    body_kw = json.loads(a.body)
    rel = None if a.release == "none" else a.release
    cat = None if a.catch == "none" else a.catch
    jobs = [(T, m, ph, a.grip, rel, cat, a.tag, a.out, a.init, a.init3d, body_kw, a.maxit, a.max_cpu)
            for T in a.T for m in a.m for ph in a.phis]
    rows = []
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            with open(os.path.join(a.out, f"rows3d_{a.tag}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
