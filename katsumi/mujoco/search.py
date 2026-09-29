"""Physics-based trajectory search for the 3-D environment (plan Sec. 12.1 "物理ベース探索",
promoted to the main 3-D method; RL is the comparison).

Open-loop policy = per-joint spline of PD targets (K knots over the horizon) + per-hand release
time on A + close-command time for B. The joint targets are tracked by the torque-saturated PD
servos of the environment, the fingers by the finite-capacity grasp model, so every candidate is a
full physical trial of the environment (same success / failure / load logging as for RL).

Cost (minimised by CMA-ES)
    fail :  1000 + 100 * d_min + 50 * (1 - cos(facing error at closest approach)) + 20 * (hands not on B)
    success: w_U * U_peak + w_E * E_eff   (plan Sec. 8 objectives)
Initialisation from the Stage-1 planar solution (retargeted sagittal joints) is optional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pickle
import numpy as np

from .env import CliffEnv, EnvParams
from .. import device


@dataclass
class SearchParams:
    horizon: float = 6.0          # planned duration after the start [s] (episode may end earlier)
    knot_dt: float = 0.2
    w_U: float = 10.0
    w_E: float = 1.0
    sigma0: float = 0.15
    popsize: int = 16
    generations: int = 50
    symmetric: bool = False       # tie left/right sagittal joints (reduces the search space)
    turn_window: float = 0.4      # [s] after release used by the planar retargeting to flip the facing convention


SAGITTAL = ["shoulder_flex", "elbow", "hip_flex", "knee"]


class SplinePolicy:
    """Piecewise-linear joint targets in normalised action space + grasp timing."""

    def __init__(self, env: CliffEnv, sp: SearchParams):
        self.env, self.sp = env, sp
        self.K = int(round(sp.horizon / sp.knot_dt)) + 1
        self.nu = env.model.nu
        self.n_params = self.K * self.nu + 3      # + t_rel_L, t_rel_R, t_close

    def unpack(self, x):
        knots = np.clip(x[: self.K * self.nu].reshape(self.K, self.nu), -1, 1)
        t_rel = np.clip(x[self.K * self.nu: self.K * self.nu + 2], 0.05, self.sp.horizon)
        t_close = np.clip(x[self.K * self.nu + 2], 0.0, 1.5)   # delay after the LAST release before closing
        return knots, t_rel, t_close

    def action(self, x, t):
        knots, t_rel, t_close = self.unpack(x)
        s = np.clip(t / self.sp.knot_dt, 0, self.K - 1 - 1e-9)
        k = int(np.floor(s))
        a = s - k
        targets = (1 - a) * knots[k] + a * knots[k + 1]
        t_last = max(t_rel)
        g = np.array([1.0 if t < t_rel[i] else (0.0 if t < t_last + t_close else 1.0) for i in range(2)])
        return np.concatenate([targets, g])

    def default_x(self):
        a0 = self.env.joint_targets_from_qpos()
        x = np.zeros(self.n_params)
        x[: self.K * self.nu] = np.tile(a0, self.K)
        x[self.K * self.nu: self.K * self.nu + 2] = 2.0
        x[self.K * self.nu + 2] = 0.1
        return x

    # -------------------------------------------------------------- planar retargeting ------
    def init_from_planar(self, sol_pkl: str):
        """Retarget a Stage-1 planar solution (relative joint angles) onto the sagittal joints."""
        r = pickle.load(open(sol_pkl, "rb"))
        from ..planar.anthro import make_body
        from ..planar.model import PlanarChain, NTH, NQ
        chain = PlanarChain(make_body(r["m"]))
        x = self.default_x()
        knots = x[: self.K * self.nu].reshape(self.K, self.nu)
        env = self.env
        names = env.act_names
        idx = {n: i for i, n in enumerate(names)}
        lo, hi = env.jrange[:, 0], env.jrange[:, 1]
        t_start = r["t_s0"]                 # the 3-D episode starts at the beginning of the swing
        t_rel = r["t_l"] - t_start
        for k in range(self.K):
            t = t_start + k * self.sp.knot_dt
            if t <= r["t_l"]:
                X = r["S_X"]; dur = r["d_s"]; t0 = r["t_s0"]
                th = _interp_cols(X[:NTH], dur, t - t0)
                facing = -1.0
            elif t <= r["t_c"]:
                X = r["F_X"]; dur = r["d_f"]; t0 = r["t_l"]
                th = _interp_cols(X[2:NQ], dur, t - t0)
                facing = -1.0 + 2.0 * min(1.0, (t - r["t_l"]) / self.sp.turn_window)
            else:
                X = r["H_X"]; dur = r["params"]["T_hold"]; t0 = r["t_h0"]
                th = _interp_cols(X[:NTH], dur, min(t - t0, dur))
                facing = +1.0
            e, psi, chi, kap = th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]
            # facing -1 (chest to A, -x):  flex = pi + psi, elbow = e, hip = -chi, knee = kap ; facing +1: mirrored
            sflex = np.pi + facing * (-psi) if facing > 0 else np.pi + psi
            elbow = e * (-facing)
            hipf = chi * facing
            knee = kap * (-facing)
            vals = dict(shoulder_flex=sflex, elbow=elbow, hip_flex=hipf, knee=knee)
            for side in ("left", "right"):
                for jn, v in vals.items():
                    i = idx[f"{jn}_{side}"]
                    v = float(np.clip(v, lo[i], hi[i]))
                    knots[k, i] = 2 * (v - lo[i]) / (hi[i] - lo[i]) - 1
        x[: self.K * self.nu] = knots.ravel()
        x[self.K * self.nu: self.K * self.nu + 2] = t_rel
        x[self.K * self.nu + 2] = 0.05
        self.t_start_device = t_start
        return x


def _interp_cols(X, dur, t):
    N = X.shape[1] - 1
    s = np.clip(t / dur * N, 0, N - 1e-9)
    k = int(np.floor(s))
    a = s - k
    return (1 - a) * X[:, k] + a * X[:, k + 1]


# ------------------------------------------------------------------ evaluation --------------
def rollout(env: CliffEnv, policy: SplinePolicy, x, T, m, phi0, sp: SearchParams, record=False):
    env.reset(T=T, m_body=m, phi0=phi0)
    d_min = np.inf
    face_best = -1.0
    frames = []
    t = 0.0
    info = {}
    while True:
        a = policy.action(x, t)
        obs, r, term, trunc, info = env.step(a)
        t = env.t_elapsed
        if env.released_A:
            pB = env.cliffs["B"][0]
            mid = 0.5 * (env.data.site_xpos[env.hands["left"].site] + env.data.site_xpos[env.hands["right"].site])
            dist = np.linalg.norm(mid - pB)
            if dist < d_min:
                d_min = dist
                chest = env.data.site_xmat[env.chest_site].reshape(3, 3)[:, 2]
                face_best = float(chest[0])
        if record:
            frames.append(env.data.qpos.copy())
        if term or trunc:
            break
    res = env.result or {}
    success = bool(res.get("success", False))
    if success:
        cost = sp.w_U * res["U_peak"] + sp.w_E * res["E_eff"]
    else:
        on_B = sum(h.attached == "B" for h in env.hands.values())
        dm = d_min if np.isfinite(d_min) else 3.0
        cost = 1000.0 + 100.0 * dm + 50.0 * (1 - face_best) + 20.0 * (2 - on_B)
    out = dict(cost=cost, success=success, reason=res.get("reason", info.get("reason", "")),
               U_peak=res.get("U_peak", env.U_peak), E_eff=res.get("E_eff", env.E_eff), d_min=d_min,
               face=face_best, t_end=t, release_time=res.get("release_time"), catch_time=res.get("catch_time"))
    if record:
        out["frames"] = np.array(frames)
        out["log"] = env.log
    return out


_worker_env = None


def _worker_init(env_params_bytes):
    global _worker_env
    p = pickle.loads(env_params_bytes)
    _worker_env = CliffEnv(p)


def _worker_eval(args):
    x, T, m, phi0, sp = args
    pol = SplinePolicy(_worker_env, sp)
    return rollout(_worker_env, pol, x, T, m, phi0, sp)["cost"]


def run_cmaes(env_params: EnvParams, sp: SearchParams, T, m, phi0, x0=None, workers=1, seed=0, log=print,
              sigma0=None):
    import cma
    import multiprocessing as mp
    env = CliffEnv(env_params)
    env.reset(T=T, m_body=m, phi0=phi0)
    pol = SplinePolicy(env, sp)
    if x0 is None:
        x0 = pol.default_x()
    es = cma.CMAEvolutionStrategy(x0, sigma0 or sp.sigma0, {"popsize": sp.popsize, "seed": seed, "verbose": -9})
    pool = None
    if workers > 1:
        pool = mp.Pool(workers, initializer=_worker_init, initargs=(pickle.dumps(env_params),))
    best = (np.inf, None, None)
    hist = []
    for g in range(sp.generations):
        X = es.ask()
        args = [(np.asarray(x), T, m, phi0, sp) for x in X]
        if pool is not None:
            costs = pool.map(_worker_eval, args)
        else:
            costs = [rollout(env, pol, x, T, m, phi0, sp)["cost"] for x in X]
        es.tell(X, costs)
        i = int(np.argmin(costs))
        if costs[i] < best[0]:
            best = (costs[i], np.array(X[i]), g)
        hist.append((g, float(np.min(costs)), float(np.mean(costs))))
        log(f"gen {g:3d} best {np.min(costs):9.2f} mean {np.mean(costs):9.2f} overall {best[0]:9.2f}")
    if pool is not None:
        pool.close()
    return dict(x_best=best[1], cost_best=best[0], gen_best=best[2], history=hist, policy_K=pol.K)
