"""Fast planar Cliff-Dimension environment (the reduced model as a Gymnasium environment).

Same physics as the trajectory optimiser (PlanarChain, device kinematics, hook/friction cone, compliant catch of
simulate.release_window, forearm-face clearance, optimiser joint ranges as soft stops), integrated by the compiled
sub-step functions of `fastsim` (several thousand control steps per second on one core). The optimiser is therefore
an exact oracle for this environment: its solutions give the minimum required grip capacity U* and the Lagrange
multipliers ("prices") of every physical constraint, against which learning methods can be measured.

Episode: start hanging on A at device phase phi0; actions = 4 normalised joint torques (elbow, shoulder, hip,
knee) + a release trigger (a[4] > 0 releases, once). The hands hook B automatically when the hook line enters
the catch box (from `hook_tol` = 2.5 cm behind the tip, the reach of the distal phalanges, to the wall; height
-0.5 .. +5 cm) with an admissible approach (fingers close reflexively). Success = holding B for `hold_time`
with the grip inside the admissible set. Failures: no release within one period, hand falls past B, hits B's
face, slip on A (grip outside the cone for 30 ms), grip capacity exceeded, hook lost on B (the fingers slide along
the ledge whenever the demanded force leaves the friction/hook cone and lose the edge beyond `hook_tol`), body
through a wall.

Reward (plan Sec. 10.2): -w_U dU_peak - w_E dE - w_time dt per step, +/-100 at the end. Optional shaping:
  "margin"  : -w_dist * d_min (closest approach of the hook line to B's tip after the release, at failure)
  "energy"  : potential-based swing-energy term while on A (lets a policy discover pumping)
  "dual"    : Lagrangian penalty  -scale * sum_k lambda_k * violation_k  with the constraint prices lambda_k taken
              from the physics oracle (run_duals.py), violations measured in the oracle's own constraint units
Shaping strings can be combined with "+" (e.g. "dual+energy").

reset(options=...) accepts phi0, T, m, body_kw and an initial swing state `state=(th, thd)` at time `t0` (absolute
device time), so that episodes can start anywhere along an oracle trajectory.
"""
from __future__ import annotations

import json
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("pip install gymnasium") from e

from .. import device
from .anthro import G, make_body
from .model import PlanarChain, NQ, NTH, NTAU
from .fastsim import FastSim, REL_LO_NEG, REL_HI_NEG  # noqa: F401  (re-exported)

DEFAULT_PRICES = dict(cap=1.0, cone=0.3, joint_speed=0.05, joint_range=0.5, wall=1.0, catch=1.0)


class PlanarCliffEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, T=20.0, m=66.0, phi0=None, body_kw=None, control_dt=0.02, sub_dt=0.002, hold_time=2.0,
                 shaping="none", prices=None, price_scale=10.0, U_target=1.2, U_cap=2.0, w_U=3.0, w_E=3.0, w_time=0.01,
                 w_dist=20.0, w_energy=20.0, eps=0.20, K_att=40000.0, D_att=1500.0, hook_tol=0.025,
                 catch_box=(0.03, -0.005, 0.05), seed=0, T_set=None, m_set=None, body_set=None, joint_stop_k=15.0,
                 stop_damp=0.01, ramp_att=0.01, obs_body=True, cache_dir=None):
        super().__init__()
        self.T_fixed, self.m_fixed, self.phi0_fixed = T, m, phi0
        self.T_set, self.m_set, self.body_set = T_set, m_set, body_set
        self.body_kw = dict(body_kw or {})
        self.control_dt, self.sub_dt, self.hold_time, self.eps = control_dt, sub_dt, hold_time, eps
        self.n_sub = int(round(control_dt / sub_dt))
        self.shaping = shaping
        self.prices = dict(DEFAULT_PRICES); self.prices.update(prices or {})
        self.price_scale, self.U_target, self.U_cap = price_scale, U_target, U_cap
        self.w_U, self.w_E, self.w_time, self.w_dist, self.w_energy = w_U, w_E, w_time, w_dist, w_energy
        self.K_att, self.D_att, self.hook_tol, self.catch_box = K_att, D_att, hook_tol, catch_box
        self.joint_stop_k, self.ramp_att, self.stop_damp = joint_stop_k, ramp_att, stop_damp
        self.obs_body = obs_body
        self.cache_dir = cache_dir
        self.rng = np.random.default_rng(seed)
        self._sims = {}
        self._set_body(m, self.body_kw)
        self.T = float(T)
        self._blank()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._obs().shape[0],), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(NTAU + 1,), dtype=np.float32)

    # ------------------------------------------------------------------ setup ----------------
    def _set_body(self, m, body_kw):
        key = (float(m), tuple(sorted(body_kw.items())))
        if key not in self._sims:
            body = make_body(float(m), **body_kw)
            chain = PlanarChain(body)
            sim = FastSim(body, chain, self.sub_dt, self.n_sub, self.eps, self.K_att, self.D_att, self.joint_stop_k,
                          ramp_att=self.ramp_att, stop_damp=self.stop_damp, cache_dir=self.cache_dir)
            self._sims[key] = (body, chain, sim)
        self.body, self.ch, self.sim = self._sims[key]
        self.cur_body_kw = dict(body_kw)
        self.tau_cap = self.body.tau_cap
        self.f_cap = self.body.f_cap

    def _blank(self):
        self.mode = "A"
        self.th = np.zeros(NTH); self.thd = np.zeros(NTH)
        self.q = np.zeros(NQ); self.qd = np.zeros(NQ)
        self.t = 0.0; self.t0 = 0.0
        self.released = False; self.t_release = None; self.t_catch = None; self.off = 0.0
        self.U = 0.0; self.U_peak = 0.0; self.E = 0.0
        self.d_min = np.inf; self.cone_viol_time = 0.0; self.E_swing_prev = None; self.slip = 0.0
        self.viol_acc = dict(cap=0.0, cone=0.0, joint_speed=0.0, joint_range=0.0, wall=0.0)
        self.done = False; self.result = None

    def dev(self, t):
        return device.device_state(t, self.T, self.eps)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o = options or {}
        if self.body_set:
            bk = dict(self.body_set[int(self.rng.integers(len(self.body_set)))])
            m = float(bk.pop("m", self.m_fixed))
        else:
            bk = dict(self.body_kw)
            m = float(self.rng.choice(self.m_set)) if self.m_set else float(self.m_fixed)
        if "m" in o:
            m = float(o["m"])
        if "body_kw" in o:
            bk = dict(o["body_kw"])
        T = float(self.rng.choice(self.T_set)) if self.T_set else float(self.T_fixed)
        T = float(o.get("T", T))
        self._set_body(m, bk)
        self._blank()
        self.T = T
        if "t0" in o:
            self.t0 = float(o["t0"]); phi0 = (self.t0 % T) / T
        else:
            phi0 = o.get("phi0", self.phi0_fixed)
            phi0 = float(self.rng.uniform(0, 1)) if phi0 is None else float(phi0)
            self.t0 = phi0 * T
        self.t = self.t0
        if "state" in o:
            th, thd = o["state"]
            self.th, self.thd = np.array(th, float).copy(), np.array(thd, float).copy()
        dv = self.dev(self.t)
        self.q = np.concatenate([dv["pA"], self.th]); self.qd = np.concatenate([dv["vA"], self.thd])
        # U at the start (static hang / current pinned state), so that U_peak is meaningful from step 0
        _, R = self.ch.pinned(self.q, self.qd, np.zeros(NTAU), dv["aA"])
        self.U = float(np.linalg.norm(R) / self.f_cap); self.U_peak = self.U
        return self._obs(), dict(T=T, m=m, phi0=phi0, body_kw=bk)

    # ------------------------------------------------------------------ helpers --------------
    @staticmethod
    def _rel(th):
        return np.array([th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3]])

    def _cone_excess_A(self, R):
        """Distance of the hand force on A from the admissible cone, normalised (0 inside); R: (2, n)."""
        rx, ry = R[0] / self.f_cap, R[1] / self.f_cap
        ryp = np.maximum(ry, 0)
        exc = np.maximum.reduce([-ry, -(rx + self.body.mu_out * ryp), rx - self.body.mu_in * ryp, np.zeros_like(rx)])
        return exc

    def _qd_excess(self, QD):
        """Joint-speed excess beyond qd_max, summed over joints; QD: (5, n) link rates."""
        rel = np.abs(np.vstack([QD[0] - QD[1], QD[2] - QD[1], QD[3] - QD[2], QD[4] - QD[3]]))
        return np.maximum(rel - self.body.qd_max, 0).sum(0)

    def _cone_timer(self, exc, thresh, limit):
        """Sequential violation timer over a sub-step array; returns the first index where it exceeds `limit` or -1."""
        ct = self.cone_viol_time
        for i, e in enumerate(exc):
            ct = ct + self.sub_dt if e > thresh else 0.0
            if ct > limit:
                self.cone_viol_time = ct
                return i
        self.cone_viol_time = ct
        return -1

    # ------------------------------------------------------------------ step -----------------
    def step(self, action):
        assert not self.done, "call reset()"
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        u = a[:NTAU]
        U_peak0, E0 = self.U_peak, self.E
        term, reason = False, ""
        wall_pen = 0.0; jr_max = 0.0; cone_exc = 0.0; qd_exc = 0.0
        if self.mode == "A" and not self.released and a[NTAU] > 0.0:
            self.released = True; self.t_release = self.t; self.mode = "F"
        remaining = self.n_sub
        Gc = Gv = None
        while remaining > 0 and not term:
            k = remaining
            tt = self.t + self.sub_dt * np.arange(k)
            TT = np.full(k, self.T); tauM = np.tile(u[:, None], (1, k))
            if self.mode == "A":
                f = self.sim.fA if k == self.n_sub else None
                if f is not None:
                    X, R, jr, wall, GC, GV = [np.array(o) for o in f(np.concatenate([self.th, self.thd]), tt, TT, tauM)]
                else:
                    X, R, jr, wall, GC, GV = self._loop(self.sim.fA1, np.concatenate([self.th, self.thd]), tt, u, None, 6)
                jr = jr.ravel(); wall = wall.ravel()
                Uarr = np.linalg.norm(R, axis=0) / self.f_cap
                exc = self._cone_excess_A(R)
                qde = self._qd_excess(X[NTH:])
                i_end = k - 1; sub_term = False
                i_slip = self._cone_timer(exc, 0.02, 0.03)
                cands = []
                if i_slip >= 0: cands.append((i_slip, "slipped off A"))
                i_cap = np.argmax(Uarr > self.U_cap) if (Uarr > self.U_cap).any() else -1
                if i_cap >= 0: cands.append((i_cap, "grip capacity exceeded"))
                i_w = np.argmax(wall > 0.02) if (wall > 0.02).any() else -1
                if i_w >= 0: cands.append((i_w, "hit wall"))
                if cands:
                    i_end, reason = min(cands); sub_term = True
                self._commit(Uarr[:i_end + 1], u, exc[:i_end + 1], qde[:i_end + 1], jr[:i_end + 1], wall[:i_end + 1])
                cone_exc = max(cone_exc, float(exc[:i_end + 1].max())); qd_exc = max(qd_exc, float(qde[:i_end + 1].max()))
                jr_max = max(jr_max, float(jr[:i_end + 1].max())); wall_pen = max(wall_pen, float(wall[:i_end + 1].max()))
                self.th, self.thd = X[:NTH, i_end].copy(), X[NTH:, i_end].copy()
                self.t += self.sub_dt * (i_end + 1)
                dv = self.dev(self.t)
                self.q = np.concatenate([dv["pA"], self.th]); self.qd = np.concatenate([dv["vA"], self.thd])
                Gc, Gv = GC[:, i_end], GV[:, i_end]
                term = sub_term
                remaining = 0 if not sub_term else 0
            elif self.mode == "F":
                f = self.sim.fF if k == self.n_sub else None
                if f is not None:
                    X, jr, wall = [np.array(o) for o in f(np.concatenate([self.q, self.qd]), tt, TT, tauM)]
                else:
                    X, jr, wall = self._loop(self.sim.fF1, np.concatenate([self.q, self.qd]), tt, u, None, 3)
                jr = jr.ravel(); wall = wall.ravel()
                te = tt + self.sub_dt
                dvs = self.dev(te)
                hx, hy = X[0], X[1]
                xB = np.atleast_1d(dvs["x"]); vB = np.atleast_2d(dvs["vB"]).reshape(-1, 2)
                vrel = X[NQ:NQ + 2].T - vB                      # (k, 2)
                d = np.hypot(hx - xB, hy)
                approach_ok = (vrel[:, 1] <= 0.05) & (vrel[:, 0] >= -0.55)
                in_height = (self.catch_box[1] <= hy) & (hy <= self.catch_box[2])
                catch_a = (xB - self.hook_tol <= hx) & (hx <= xB + device.D_LEDGE) & in_height & approach_ok
                past = (hx > xB + device.D_LEDGE) & (hy > -device.WALL_BOTTOM_OFFSET)
                catch_b = past & in_height & approach_ok
                face = past & ~(in_height & approach_ok)
                missed = (hy < -1.0) & (X[NQ + 1] < 0)
                wallhit = wall > 0.02
                qde = self._qd_excess(X[NQ + 2:])
                events = []
                for cond, name in ((catch_a | catch_b, "catch"), (face, "hit B's face"), (missed, "missed B"), (wallhit, "hit wall")):
                    if cond.any():
                        events.append((int(np.argmax(cond)), name))
                i_end = k - 1; ev = None
                if events:
                    i_end, ev = min(events)
                self.d_min = min(self.d_min, float(d[:i_end + 1].min()))
                self._commit(np.zeros(i_end + 1), u, np.zeros(i_end + 1), qde[:i_end + 1], jr[:i_end + 1], wall[:i_end + 1])
                qd_exc = max(qd_exc, float(qde[:i_end + 1].max()))
                jr_max = max(jr_max, float(jr[:i_end + 1].max())); wall_pen = max(wall_pen, float(wall[:i_end + 1].max()))
                self.q, self.qd = X[:NQ, i_end].copy(), X[NQ:, i_end].copy()
                self.t += self.sub_dt * (i_end + 1)
                remaining -= i_end + 1
                if ev == "catch":
                    # the fingers curl over the edge where they land (distal phalanges reach up to hook_tol in front
                    # of the tip); the hook line is attached there and the spring-damper reacts to later motion
                    self.mode = "C"; self.t_catch = self.t
                    if catch_b[i_end] and not catch_a[i_end]:
                        self.off = float(device.D_LEDGE); self.q[0] = xB[i_end] + device.D_LEDGE
                    else:
                        self.off = float(np.clip(hx[i_end] - xB[i_end], -self.hook_tol, device.D_LEDGE))
                elif ev is not None:
                    term, reason = True, ev
            else:  # C: hooked on B (attachment `off` is part of the integrated state; it slides when the cone is exceeded)
                xc = np.concatenate([self.q, self.qd, [self.off]])
                f = self.sim.fC if k == self.n_sub else None
                if f is not None:
                    X, Fa, Fraw, sep, jr, wall, slip = [np.array(o) for o in f(xc, tt, TT, tauM, np.full(k, self.hook_tol))]
                else:
                    X, Fa, Fraw, sep, jr, wall, slip = self._loop(self.sim.fC1, xc, tt, u, self.hook_tol, 7)
                sep = sep.ravel(); jr = jr.ravel(); wall = wall.ravel(); slip = slip.ravel()
                Uarr = np.linalg.norm(Fa, axis=0) / self.f_cap
                exc = np.linalg.norm(Fraw - Fa, axis=0) / self.f_cap
                qde = self._qd_excess(X[NQ + 2:2 * NQ])
                offs = X[2 * NQ]
                cands = []
                lost = (sep > 0.08) | (offs <= -self.hook_tol + 1e-9)        # hand away from the ledge / slid off the tip
                if lost.any(): cands.append((int(np.argmax(lost)), "lost hook"))
                if (Uarr > self.U_cap).any(): cands.append((int(np.argmax(Uarr > self.U_cap)), "grip capacity exceeded"))
                if (wall > 0.02).any(): cands.append((int(np.argmax(wall > 0.02)), "hit wall"))
                te = tt + self.sub_dt
                held = te - self.t_catch >= self.hold_time
                if held.any(): cands.append((int(np.argmax(held)), "held B"))
                i_end = k - 1
                if cands:
                    i_end, reason = min(cands); term = True
                self._commit(Uarr[:i_end + 1], u, exc[:i_end + 1], qde[:i_end + 1], jr[:i_end + 1], wall[:i_end + 1])
                self.slip += float(slip[:i_end + 1].sum())
                cone_exc = max(cone_exc, float(exc[:i_end + 1].max())); qd_exc = max(qd_exc, float(qde[:i_end + 1].max()))
                jr_max = max(jr_max, float(jr[:i_end + 1].max())); wall_pen = max(wall_pen, float(wall[:i_end + 1].max()))
                self.q, self.qd = X[:NQ, i_end].copy(), X[NQ:2 * NQ, i_end].copy()
                self.off = float(offs[i_end])
                self.t += self.sub_dt * (i_end + 1)
                remaining = 0
        trunc = False
        if not term and not self.released and self.t - self.t0 >= self.T:
            term, reason = True, "no release"
        if not term and self.t - self.t0 > self.T + 4.0:
            trunc, reason = True, "time limit"
        success = reason == "held B"
        # ---- reward ---------------------------------------------------------------------------
        r = -self.w_U * (self.U_peak - U_peak0) - self.w_E * (self.E - E0) - self.w_time * self.control_dt
        if "energy" in self.shaping and self.mode == "A" and Gc is not None:
            Es = 0.5 * float(Gv @ Gv) / G + float(Gc[1])
            if self.E_swing_prev is not None:
                r += self.w_energy * (Es - self.E_swing_prev)
            self.E_swing_prev = Es
        if "dual" in self.shaping:
            pr = self.prices
            lag = (pr["cap"] * max(0.0, self.U - self.U_target) + pr["cone"] * cone_exc + pr["joint_speed"] * qd_exc
                   + pr["joint_range"] * jr_max + pr["wall"] * wall_pen)
            r -= self.price_scale * lag * self.control_dt / 0.02
            for k_, v in (("cap", max(0.0, self.U - self.U_target)), ("cone", cone_exc), ("joint_speed", qd_exc), ("joint_range", jr_max), ("wall", wall_pen)):
                self.viol_acc[k_] += v
        if term or trunc:
            r += 100.0 if success else -100.0
            if ("margin" in self.shaping or "dual" in self.shaping) and not success and self.released:
                dm = self.d_min if np.isfinite(self.d_min) else 3.0
                w = self.w_dist if "margin" in self.shaping else self.price_scale * self.prices["catch"]
                r -= w * dm
            self.done = True
            self.result = dict(success=success, reason=reason, t=self.t - self.t0, U_peak=self.U_peak, E=self.E,
                               d_min=self.d_min, release_time=(self.t_release - self.t0) if self.released else None,
                               catch_time=(self.t_catch - self.t0) if self.t_catch else None,
                               phi_release=((self.t_release % self.T) / self.T) if self.released else None,
                               T=self.T, m=self.body.m, viol=dict(self.viol_acc), slip=self.slip, off=self.off)
        info = dict(success=success, reason=reason, U_peak=self.U_peak, E=self.E, mode=self.mode, d_min=self.d_min)
        return self._obs(), float(r), bool(term), bool(trunc), info

    def _loop(self, f1, x, tt, u, off, n_out):
        """Single-substep function looped in Python (only used for the partial control steps at mode changes)."""
        outs = [[] for _ in range(n_out)]
        for ti in tt:
            res = f1(x, ti, self.T, u) if off is None else f1(x, ti, self.T, u, off)
            res = [np.array(o) for o in res]
            x = res[0].ravel()
            for j in range(n_out):
                outs[j].append(res[j].reshape(-1, 1))
        return [np.hstack(o) for o in outs]

    def _commit(self, Uarr, u, exc, qde, jr, wall):
        self.U = float(Uarr[-1]); self.U_peak = max(self.U_peak, float(Uarr.max()))
        self.E += self.sub_dt * float(np.sum(0.25 * float(u @ u) + Uarr ** 2))

    # ------------------------------------------------------------------ observation ----------
    def _obs(self):
        dv = self.dev(self.t)
        ph = (self.t % self.T) / self.T
        q, qd = self.q, self.qd
        rel = self._rel(q[2:]); reld = self._rel(qd[2:])
        hand_rel_B = np.array([q[0] - dv["x"], q[1]]); vrel_B = qd[0:2] - dv["vB"]
        mode = np.array([self.mode == "A", self.mode == "F", self.mode == "C"], float)
        t_rel = (self.t - self.t_release) if self.released else 0.0
        parts = [q[2:], qd[2:] / 5.0, rel, reld / 5.0, [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), dv["h"], dv["x"] - 2.25],
                 hand_rel_B / 2.0, vrel_B / 3.0, mode, [t_rel, self.U, self.U_peak, (self.t - self.t0) / self.T]]
        if self.obs_body:
            bk = self.cur_body_kw
            parts.append([self.T / 20.0 - 1.0, self.body.m / 66.0 - 1.0, bk.get("stature", 1.75) / 1.75 - 1.0,
                          bk.get("cap_scale", 1.0) - 1.0])
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------ convenience -----------
    def state(self):
        """Current swing state (th, thd) and absolute time (valid while hooked on A)."""
        return self.th.copy(), self.thd.copy(), self.t


def prices_from_duals(dual_json, default=None):
    """Constraint prices from a run_duals.py JSON (mean multiplier mass per inequality category, normalised so that
    the capacity price is 1 = the epigraph weight w_U)."""
    rows = json.load(open(dual_json))
    acc = {}
    n = 0
    for r in rows:
        d = r.get("duals")
        if not r.get("ok") or not d:
            continue
        cats = d["categories"]
        cap = sum(sum(v) for v in d["cap_mult"].values()) or 1.0
        acc["cone"] = acc.get("cone", 0) + (cats.get("cone_A", {}).get("abs_sum", 0) + cats.get("cone_B", {}).get("abs_sum", 0)) / cap
        acc["joint_speed"] = acc.get("joint_speed", 0) + cats.get("joint_speed", {}).get("abs_sum", 0) / cap
        acc["joint_range"] = acc.get("joint_range", 0) + cats.get("joint_range", {}).get("abs_sum", 0) / cap
        acc["wall"] = acc.get("wall", 0) + cats.get("wall", {}).get("abs_sum", 0) / cap
        acc["catch"] = acc.get("catch", 0) + (cats.get("catch_geom", {}).get("abs_sum", 0) + cats.get("catch_vel", {}).get("abs_sum", 0)) / cap
        n += 1
    out = dict(default or DEFAULT_PRICES)
    if n:
        out.update({k: v / n for k, v in acc.items()}); out["cap"] = 1.0
    return out
