"""CMA-ES trajectory search in the 3-D MuJoCo environment.

usage:
  python scripts/search_3d.py --T 10 --m 66 --phi0 0.2 --planar results/grid/sol_ref_T10_m67.5_phi0.500.pkl \
         --gens 200 --pop 24 --workers 8 --out results/search3d/T10_m66
The planar solution provides the initial sagittal joint trajectory and release time; phi0 is chosen so that
the device phase at the start equals the planar swing start (phi0 = phi_l - d_s / T).
"""
import argparse, os, sys, pickle, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from katsumi.mujoco.env import CliffEnv, EnvParams
from katsumi.mujoco.build_model import ModelParams
from katsumi.mujoco.search import SearchParams, SplinePolicy, run_cmaes, rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--m", type=float, default=66.0)
    ap.add_argument("--phi0", type=float, default=None)
    ap.add_argument("--planar", type=str, default="")
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--horizon", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/search3d/run")
    ap.add_argument("--fcap", type=float, default=None, help="per-hand grip capacity [N] (curriculum; default 650)")
    ap.add_argument("--hook-tol", type=float, default=None, help="hook tolerance beyond the tip [m] (curriculum)")
    ap.add_argument("--x0", type=str, default="", help="warm start from a previous search .pkl (x_best)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sp = SearchParams(horizon=a.horizon, popsize=a.pop, generations=a.gens, sigma0=a.sigma)
    ep = EnvParams(model=ModelParams(m=a.m), T=a.T)
    if a.fcap is not None:
        ep.grasp.f_cap_hand = a.fcap
    if a.hook_tol is not None:
        ep.grasp.hook_tol = a.hook_tol
    env = CliffEnv(ep)
    phi0 = a.phi0
    x0 = None
    if a.planar:
        env.reset(T=a.T, m_body=a.m, phi0=0.0)
        pol = SplinePolicy(env, sp)
        x0 = pol.init_from_planar(a.planar)
        if phi0 is None:
            phi0 = (pol.t_start_device % a.T) / a.T
        print(f"planar init: device start phase {phi0:.3f}, release at t={x0[pol.K*pol.nu:pol.K*pol.nu+2]}")
    if a.x0:
        prev = pickle.load(open(a.x0, "rb"))
        x0 = prev["search"]["x_best"]
        if phi0 is None:
            phi0 = prev["phi0"]
    if phi0 is None:
        phi0 = 0.0
    ep.phi0 = phi0
    # evaluate the initial candidate once (diagnostic)
    env.reset(T=a.T, m_body=a.m, phi0=phi0)
    pol = SplinePolicy(env, sp)
    if x0 is None:
        x0 = pol.default_x()
    t = time.time()
    r0 = rollout(env, pol, x0, a.T, a.m, phi0, sp)
    print("initial candidate:", {k: v for k, v in r0.items() if k not in ("frames", "log")}, f"({time.time()-t:.1f}s)")
    res = run_cmaes(ep, sp, a.T, a.m, phi0, x0=x0, workers=a.workers, seed=a.seed)
    rb = rollout(env, pol, res["x_best"], a.T, a.m, phi0, sp, record=True)
    print("best:", {k: v for k, v in rb.items() if k not in ("frames", "log")})
    pickle.dump(dict(search=res, best_rollout={k: v for k, v in rb.items() if k != "log"},
                     T=a.T, m=a.m, phi0=phi0, sp=sp.__dict__, planar=a.planar), open(a.out + ".pkl", "wb"))
    json.dump(dict(cost=res["cost_best"], success=rb["success"], U_peak=rb["U_peak"], E_eff=rb["E_eff"],
                   d_min=float(rb["d_min"]), reason=rb["reason"], history=res["history"]), open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
