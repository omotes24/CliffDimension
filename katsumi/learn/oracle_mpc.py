"""Oracle-in-the-loop MPC: re-solve the remaining trajectory optimisation from the current swing state every
`replan_dt` seconds and track the plan with a weak PD until the next re-plan; release when the plan says so.

This is the exact (non-amortised) counterpart of the dual-field controller: it shows what closed-loop re-planning
with the oracle achieves on the same environment, and its cost (one NLP solve per re-plan).
"""
from __future__ import annotations

import time

import numpy as np

from ..planar.anthro import make_body
from ..planar.nlp import PlanarNLP, ReducedParams
from ..planar.model import NTH, NQ, NTAU


def _rel(th):
    return np.array([th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]])


class OracleMPC:
    def __init__(self, ref, body, T, replan_dt=0.5, kp=1.0, kd=0.1, max_cpu=240.0, tol=1e-4, N_min=30, verbose=False,
                 d_s_min=0.05, release_slack=0.5, control_dt=0.02, mu_margin=0.9):
        """mu_margin: the plans use a tightened hook/friction cone (mu_out * mu_margin) so that tracking errors do not
        push the executed hand force out of the true cone (constraint tightening)."""
        import copy
        self.ref, self.T = ref, float(T)
        self.body = copy.deepcopy(body); self.body.mu_out = body.mu_out * mu_margin
        self.replan_dt, self.kp, self.kd, self.max_cpu, self.tol, self.N_min = replan_dt, kp, kd, max_cpu, tol, N_min
        self.verbose, self.d_s_min, self.release_slack, self.dt = verbose, d_s_min, release_slack, control_dt
        self.reset()

    def reset(self):
        self.plan = None
        self.t_plan = -np.inf
        self.n_solve = 0; self.n_fail = 0; self.solve_s = 0.0
        self.last = {}

    # ------------------------------------------------------------------------------------------------
    def _solve(self, x, t):
        r = self.ref
        p_ref = dict(r["params"])
        for q in ("fixed_release_phase", "wait_in_cost", "from_state", "x0", "N_S", "d_s_bounds"):
            p_ref.pop(q, None)
        # remaining time estimate for the mesh: previous plan or the reference (time to its release from a similar state)
        if self.plan is not None:
            rem_est = max(self.plan["t_l"] - t, 0.3)
            prev = self.plan["sol"]
            k0 = int(np.clip(np.searchsorted(prev["tS"], t), 0, prev["S_X"].shape[1] - 2))
        else:
            N = r["S_X"].shape[1] - 1
            tS = r["t_s0"] + np.linspace(0, r["d_s"], N + 1)
            # nearest reference state (normalised) to warm start
            Z = r["S_X"].T; d = np.linalg.norm((Z - x) / np.array([0.3] * NTH + [1.5] * NTH), axis=1)
            k0 = int(np.argmin(d))
            rem_est = max(r["d_s"] - (tS[k0] - r["t_s0"]), 0.3)
            prev = dict(r); prev["tS"] = tS
        N_S = int(np.clip(round(150 * rem_est / 5.7) + 10, self.N_min, 150))
        p = ReducedParams(fixed_release_phase=None, wait_in_cost=False, from_state=True, x0=tuple(x), N_S=N_S,
                          d_s_bounds=(self.d_s_min, self.T), **p_ref)
        nlp = PlanarNLP(self.body, self.T, t / self.T, p)
        warm = dict(prev)
        for key in ("S_X", "S_A", "S_U", "S_R", "S_Am", "S_Um", "S_Rm"):
            if key in prev:
                warm[key] = prev[key][:, k0:]
        warm["d_s"] = rem_est; warm["d_w"] = 0.0
        try:
            nlp.set_initial(prev=warm)
        except Exception:
            nlp.set_initial(d_w=0.0, d_s=rem_est)
        t0 = time.time()
        sol = nlp.solve(print_level=0, max_iter=1500, tol=self.tol, max_cpu_time=self.max_cpu)
        self.solve_s += time.time() - t0; self.n_solve += 1
        if not sol["ok"]:
            self.n_fail += 1
        N = sol["S_X"].shape[1] - 1
        sol["tS"] = t + np.linspace(0, sol["d_s"], N + 1)
        if self.verbose:
            print(f"   [oracle-mpc] t={t:.2f} ok={sol['ok']} U*={sol['U_peak']:.3f} d_s={sol['d_s']:.2f} it={sol['iters']} {time.time() - t0:.0f}s", flush=True)
        return sol

    def act(self, th, thd, t):
        x = np.concatenate([th, thd])
        if self.plan is None or (t - self.t_plan >= self.replan_dt - 1e-9 and (self.plan["t_l"] - t) > self.replan_dt):
            sol = self._solve(x, t)
            self.t_plan = t                                   # (retry after replan_dt even when the solve failed)
            if sol["ok"] or self.plan is None:
                self.plan = dict(sol=sol, t_l=t + sol["d_s"], ok=sol["ok"])
        pl = self.plan["sol"]
        tt = pl["tS"]
        tc = min(max(t, tt[0]), tt[-1])
        u = np.array([np.interp(tc, tt, row) for row in pl["S_U"]])
        th_ref = np.array([np.interp(tc, tt, row) for row in pl["S_X"][:NTH]]); thd_ref = np.array([np.interp(tc, tt, row) for row in pl["S_X"][NTH:]])
        u = np.clip(u + self.kp * (_rel(th_ref) - _rel(th)) + self.kd * (_rel(thd_ref) - _rel(thd)), -1, 1)
        release = (self.plan["t_l"] - t) <= self.release_slack * self.dt
        self.last = dict(tau=self.plan["t_l"] - t, U=pl["U_peak"], ok=self.plan["ok"])
        return u, release

    def __call__(self, env):
        u, release = self.act(env.th, env.thd, env.t)
        a = np.zeros(NTAU + 1); a[:NTAU] = u; a[NTAU] = 1.0 if release else -1.0
        return a
