"""Stage-1 main experiment: release-phase map of the reduced model.

For every (T, m) (T = full device period) the NLP is solved for a grid of release phases phi_l in [0, 1) with the
"required grip capacity" objective (minimise U_peak, effort as regulariser). Results are
appended to a CSV and each solution is pickled, so partial runs are usable.

usage: python scripts/run_grid.py --out results/grid --T 16 19 20 24 --m 60 75 --nphi 16 --workers 2
"""
import argparse, os, sys, time, pickle, json, csv, traceback
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.planar.anthro import make_body, G
from katsumi.planar.nlp import PlanarNLP, ReducedParams
from katsumi.planar.simulate import Resim

FIELDS = ["T", "m", "phi_l", "ok", "status", "iters", "solve_s", "U_peak", "req_cap_N", "req_cap_BW", "effort",
          "d_s", "d_f", "phi_c", "x_catch", "h_release", "v0x", "v0y", "vrel_x", "vrel_y", "vrel_norm",
          "impulse_norm", "U_catch", "U_swing", "U_hold", "direction",
          "res_S_p95", "res_F_p95", "res_H_p95", "res_S_max", "res_F_max", "res_H_max",
          "ver_S_pos", "ver_F_hand", "ver_H_pos", "cold_start", "param_tag"]


def hs_residuals(r, body):
    from scripts.check_solution import hs_residuals as _hs  # noqa
    return _hs(r, body)


def solve_case(T, m, phi_l, prev, params_kw, tag, outdir, d_s_guesses=(3.0, 2.0, 4.0)):
    body = make_body(m, **params_kw.pop("body_kw", {}))
    p = ReducedParams(fixed_release_phase=phi_l, wait_in_cost=False, **params_kw)
    best = None
    tried = []
    t0 = time.time()
    if prev is not None:
        nlp = PlanarNLP(body, T, phi_l, p)
        nlp.set_initial(prev=prev)
        r = nlp.solve(print_level=0, max_iter=2500, tol=1e-4, max_cpu_time=900)
        r["cold_start"] = 0
        tried.append(r)
        if r["ok"]:
            best = r
    if best is None:
        for d_s in d_s_guesses:
            nlp = PlanarNLP(body, T, phi_l, p)
            nlp.set_initial(d_w=T - d_s, d_s=d_s)
            r = nlp.solve(print_level=0, max_iter=2500, tol=1e-4, max_cpu_time=900)
            r["cold_start"] = 1
            tried.append(r)
            if r["ok"]:
                if best is None or r["U_peak"] < best["U_peak"] - 1e-3:
                    best = r
                break
    if best is None:  # keep the least-infeasible attempt for diagnostics
        best = min(tried, key=lambda x: x.get("U_peak", 9))
    best["solve_s"] = time.time() - t0
    best["param_tag"] = tag
    return best, body


def row_from(r, body):
    row = {k: "" for k in FIELDS}
    rs = Resim(body, r)
    row.update(T=r["T"], m=r["m"], phi_l=round(r["phi_l"], 4), ok=int(r["ok"]), status=r["status"], iters=r["iters"],
               solve_s=round(r["solve_s"], 1), U_peak=r["U_peak"], req_cap_N=r["U_peak"] * body.f_cap,
               req_cap_BW=r["U_peak"] * body.f_cap / (body.m * G), effort=r["effort"], d_s=r["d_s"], d_f=r["d_f"],
               phi_c=r["phi_c"], cold_start=r.get("cold_start", ""), param_tag=r.get("param_tag", ""))
    try:
        mtr = rs.metrics()
        row.update(x_catch=mtr["x_catch"], h_release=mtr["h_release"], v0x=mtr["v0"][0], v0y=mtr["v0"][1],
                   vrel_x=mtr["v_rel_catch"][0], vrel_y=mtr["v_rel_catch"][1], vrel_norm=mtr["v_rel_norm"],
                   impulse_norm=mtr["impulse_norm"], U_catch=mtr["U_catch"], U_swing=mtr["U_peak_swing"],
                   U_hold=mtr["U_peak_hold"], direction=mtr["direction"])
        res = hs_residuals(r, body)
        row.update(res_S_p95=res["S"]["p95"], res_F_p95=res["F"]["p95"], res_H_p95=res["H"]["p95"],
                   res_S_max=res["S"]["max"], res_F_max=res["F"]["max"], res_H_max=res["H"]["max"])
        ver = rs.verify()
        row.update(ver_S_pos=ver["S_pos_err"], ver_F_hand=ver["F_hand_err"], ver_H_pos=ver["H_pos_err"])
    except Exception as e:  # diagnostics must never kill the run
        row["status"] += f" [metrics failed: {e}]"
    return row


def worker(args):
    T, m, phis, params_kw, tag, outdir = args
    prev = None
    rows = []
    for phi_l in phis:
        fn = os.path.join(outdir, f"sol_{tag}_T{T:g}_m{m:g}_phi{phi_l:.3f}.pkl")
        try:
            if os.path.exists(fn):
                r = pickle.load(open(fn, "rb"))
                body = make_body(m, **dict(params_kw).get("body_kw", {}))
            else:
                r, body = solve_case(T, m, phi_l, prev, dict(params_kw), tag, outdir)
                pickle.dump(r, open(fn, "wb"))
            row = row_from(r, body)
            prev = r if r["ok"] else prev
        except Exception:
            traceback.print_exc()
            row = {k: "" for k in FIELDS}
            row.update(T=T, m=m, phi_l=phi_l, ok=0, status="exception", param_tag=tag)
        rows.append(row)
        rows_name = f"rows_{tag}_T{T:g}_m{m:g}.csv" if len(phis) > 1 else f"rows_{tag}_T{T:g}_m{m:g}_phi{phi_l:.3f}.csv"
        with open(os.path.join(outdir, rows_name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[{tag}] T={T} m={m} phi_l={phi_l:.3f} ok={row['ok']} U_peak={row['U_peak']} "
              f"E={row['effort']} it={row['iters']} {row['solve_s']}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grid")
    ap.add_argument("--T", type=float, nargs="+", default=[16, 19, 20, 24])
    ap.add_argument("--m", type=float, nargs="+", default=[60, 67.5, 75])
    ap.add_argument("--nphi", type=int, default=16)
    ap.add_argument("--phis", type=float, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--params", default="{}", help="JSON of ReducedParams overrides (+ body_kw)")
    ap.add_argument("--split-phases", action="store_true",
                    help="one job per (T, m, phi): no warm start along phi, but embarrassingly parallel (many cores)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    params_kw = json.loads(a.params)
    phis = a.phis if a.phis else list(np.arange(a.nphi) / a.nphi)
    if a.split_phases:
        jobs = [(T, m, [ph], params_kw, a.tag, a.out) for T in a.T for m in a.m for ph in phis]
    else:
        jobs = [(T, m, phis, params_kw, a.tag, a.out) for T in a.T for m in a.m]
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            all_rows = pool.map(worker, jobs, chunksize=1)
    else:
        all_rows = [worker(j) for j in jobs]
    rows = [r for rr in all_rows for r in rr]
    with open(os.path.join(a.out, f"grid_{a.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
