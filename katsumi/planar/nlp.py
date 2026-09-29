"""Multi-phase direct collocation (Hermite-Simpson) for the planar reduced model.

Phases (absolute device time t; the athlete starts at device phase phi0, i.e. t0 = phi0 * T)
    W  wait      : static hang on A, duration d_w >= 0 (no dynamics; grip effort accumulates)
    S  swing     : pinned on A, duration d_s, joint torques build the swing
    F  flight    : free chain, duration d_f, only internal torques
    C  catch     : hands inside the hook region of B, finite hand force R within the grasp set
    H  hold      : pinned on B for T_hold = 2.0 s (success criterion of the plan, Sec. 7.3)

Decision variables: durations, states at knots, normalised torques u = tau / tau_cap at knots and
interval mid-points, normalised hand force during C, and the epigraph variable U_peak.

Anti-exploit safeguards (the optimiser must not gain momentum from discretisation error):
    * Hermite-Simpson (4th order) with dt <= ~0.04 s
    * in flight, the CoM must follow the analytic parabola within `tol_com` [m] / `tol_comv` [m/s]
      and the angular momentum about the CoM must be conserved within `tol_L` (plan Sec. 13.3)
    * every solution is re-integrated with RK4 at 1 ms in `simulate.py` and the mismatch reported

Objective  J = w_E * E_eff + w_U * U_peak + w_s * smoothness
    E_eff = (1/t_ref) int [ (1/4) sum_j u_j^2 + U^2 ] dt,  U = |R| / f_cap  (plan eq. 15)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import casadi as ca

from .. import device
from .anthro import Body, G
from .model import PlanarChain, NQ, NTH, NTAU


@dataclass
class ReducedParams:
    eps: float = 0.20            # device turnaround smoothing [s]
    T_hold: float = 2.0          # required hold time on B [s]
    release_deadline_factor: float = 2.0   # release before 2T (plan Sec. 7.3)
    N_S: int = 150
    N_F: int = 30
    N_H: int = 60
    d_s_bounds: tuple = (0.3, 6.0)
    d_f_bounds: tuple = (0.10, 1.20)
    # catch = inelastic impact of the hands on the ledge corner of B (rigid limit), see _build
    v_rel_max: float = 4.0            # sanity bound on |hand velocity relative to B| at contact [m/s]
    v_away_max: float = 0.5           # allowed hand velocity component away from B's wall at contact [m/s]
    delta_catch: float = 0.05         # equivalent duration used to convert the catch impulse to a force [s]
    approach_clear: float = 0.05      # the hand may be below the grip plane only if > 5 cm in front of the tip
    wall_smooth: float = 0.01         # smooth-max width for wall disjunctions [m]
    edge_radius: float = device.EDGE_RADIUS_DEFAULT
    tol_com: float = 5e-3             # flight CoM position band vs analytic parabola [m] (post-hoc audit is 1 mm)
    tol_comv: float = 2e-2            # flight CoM velocity band [m/s]
    tol_L: float = 0.10               # flight angular momentum band [kg m^2/s]
    u_rate: float = 10.0              # max |d(tau/tau_cap)/dt| [1/s]  (full range in 0.1 s)
    t_ref: float = 1.0
    w_E: float = 1.0
    w_U: float = 10.0
    U_max: float = 2.5                # upper bound on U_peak (2.5 -> required capacity up to 2.5 x f_cap is explored)
    fixed_release_phase: float | None = None   # if set: release exactly at device phase phi_l (t_l = (phi_l + 2) T)
    wait_in_cost: bool = True
    w_smooth: float = 1e-3
    # joint ranges [rad] for the "facing -x" convention (elbow, shoulder, hip, knee); mirrored when facing +x
    rel_lo_face_neg: tuple = (0.0, -4.2, -2.1, 0.0)
    rel_hi_face_neg: tuple = (2.6, 0.26, 0.35, 2.6)


def smax(a, b, delta):
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + delta ** 2))


def _ge0(opti, expr):
    """opti.subject_to(expr >= 0) that tolerates constant expressions (checks and skips them)."""
    if isinstance(expr, ca.MX) and expr.is_constant():
        val = float(ca.Function("c", [], [expr])()["o0"])
        assert val >= -1e-9, f"constant constraint violated: {val}"
        return
    opti.subject_to(expr >= 0)


class PlanarNLP:
    def __init__(self, body: Body, T: float, phi0: float, params: ReducedParams | None = None):
        self.body = body
        self.T = float(T)
        self.phi0 = float(phi0)
        self.p = params or ReducedParams()
        self.chain = PlanarChain(body)
        self._build()

    # ------------------------------------------------------------------ helpers -------
    def _dev(self, t):
        return device.device_ca(t, self.T, self.p.eps)

    def _rel_bounds(self, facing):
        lo = np.array(self.p.rel_lo_face_neg)
        hi = np.array(self.p.rel_hi_face_neg)
        if facing == "neg":
            return lo, hi
        if facing == "pos":
            return -hi, -lo
        return np.minimum(lo, -hi), np.maximum(hi, -lo)  # union (flight)

    def _wall_constraints(self, opti, q, dev):
        """Body points must stay out of both wall bodies (smooth disjunction)."""
        p = self.p
        pts = self.chain.f_tips(q)                # 2 x 5 clearance points
        hand = q[0:2]
        allP = ca.horzcat(hand, pts)
        clear = np.concatenate([[0.0], self.body.clearance])
        h = dev["h"]
        xB = dev["x"]
        for i in range(allP.shape[1]):
            x, y = allP[0, i], allP[1, i]
            aA = x - (-device.D_LEDGE + clear[i])                  # in front of A's face
            bA = (h - device.WALL_BOTTOM_OFFSET) - y               # below A's body
            _ge0(opti, smax(aA, bA, p.wall_smooth))
            aB = (xB + device.D_LEDGE - clear[i]) - x              # in front of B's face
            bB = (-device.WALL_BOTTOM_OFFSET) - y                  # below B's body
            _ge0(opti, smax(aB, bB, p.wall_smooth))

    def _no_approach_from_below(self, opti, q, dev):
        hand = q[0:2]
        xB = dev["x"]
        _ge0(opti, smax(hand[1], (xB - self.p.approach_clear) - hand[0], self.p.wall_smooth))

    def _grasp_cone(self, opti, r, side):
        """r: normalised force on the athlete R / f_cap. side 'A': wall on -x; 'B': wall on +x."""
        b = self.body
        opti.subject_to(r[1] >= 0)
        if side == "A":
            opti.subject_to(r[0] >= -b.mu_out * r[1])   # body swung to +x (away from A) -> R_x < 0, friction-limited
            opti.subject_to(r[0] <= b.mu_in * r[1])
        else:
            opti.subject_to(r[0] <= b.mu_out * r[1])    # body swung to -x (away from B) -> R_x > 0
            opti.subject_to(r[0] >= -b.mu_in * r[1])

    def _joint_limits(self, opti, q, qd, facing):
        lo, hi = self._rel_bounds(facing)
        opti.subject_to(opti.bounded(lo, self.chain.f_rel(q), hi))
        opti.subject_to(opti.bounded(-self.body.qd_max, self.chain.B.T @ qd, self.body.qd_max))

    # ------------------------------------------------------------------ generic HS phase
    def _hs_phase(self, name, N, dur, t_start, nx, na, nu, nr, residual, path, effort, U_of=None,
                  r_is_control=False):
        """Hermite-Simpson (compressed) phase with IMPLICIT dynamics.

        State x = (pos, vel) with len(vel) = na; accelerations A and the hand force Rv are
        decision variables; residual(t, x, a, u, r) = M qdd + h - B tau - J^T R must vanish at
        knots and mid-points. No matrix inverse appears in the NLP graph (fast, well conditioned).

        path(t, x, a, u, r, at_knot) -> adds path constraints
        effort(t, x, a, u, r)        -> scalar integrand
        U_of(t, x, a, u, r)          -> grasp utilisation U (bounded by U_peak) or None
        """
        opti = self.opti
        p = self.p
        X = opti.variable(nx, N + 1)
        A = opti.variable(na, N + 1)
        Am = opti.variable(na, N)
        U = opti.variable(nu, N + 1)
        Um = 0.5 * (U[:, :-1] + U[:, 1:])          # piecewise-linear torque commands
        dt = dur / N
        # activation-rate limit on the torque commands (anti-sawtooth, physiological)
        opti.subject_to(opti.bounded(-p.u_rate * dt, ca.vec(U[:, 1:] - U[:, :-1]), p.u_rate * dt))
        if nr == 0:
            Rv = ca.MX.zeros(1, N + 1)
            Rm = ca.MX.zeros(1, N)
        else:
            Rv = opti.variable(nr, N + 1)
            Rm = opti.variable(nr, N)
        F = []
        E = []
        for k in range(N + 1):
            tk = t_start + dt * k
            F.append(ca.vertcat(X[na:, k], A[:, k]))
            opti.subject_to(residual(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]) == 0)
            E.append(effort(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]))
            path(tk, X[:, k], A[:, k], U[:, k], Rv[:, k], True)
            if U_of is not None:
                opti.subject_to(U_of(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]) <= self.U_peak)
        eff = 0
        smooth = 0
        for k in range(N):
            tm = t_start + dt * (k + 0.5)
            xm = 0.5 * (X[:, k] + X[:, k + 1]) + dt / 8 * (F[k] - F[k + 1])
            fm = ca.vertcat(xm[na:], Am[:, k])
            opti.subject_to(residual(tm, xm, Am[:, k], Um[:, k], Rm[:, k]) == 0)
            opti.subject_to(X[:, k + 1] - X[:, k] - dt / 6 * (F[k] + 4 * fm + F[k + 1]) == 0)
            em = effort(tm, xm, Am[:, k], Um[:, k], Rm[:, k])
            eff += dt / 6 * (E[k] + 4 * em + E[k + 1])
            path(tm, xm, Am[:, k], Um[:, k], Rm[:, k], False)
            if U_of is not None:
                opti.subject_to(U_of(tm, xm, Am[:, k], Um[:, k], Rm[:, k]) <= self.U_peak)
            smooth += ca.sumsqr(U[:, k + 1] - U[:, k])
        self.vars[name + "_X"] = X
        self.vars[name + "_A"] = A
        self.vars[name + "_Am"] = Am
        self.vars[name + "_U"] = U
        self.vars[name + "_Um"] = Um
        self.vars[name + "_R"] = Rv
        self.vars[name + "_Rm"] = Rm
        self.effort_terms.append(eff / self.p.t_ref)
        self.smooth_terms.append(smooth)
        return X, U, Um, Rv

    # ------------------------------------------------------------------ build ---------
    def _build(self):
        p, b, T = self.p, self.body, self.T
        chain = self.chain
        opti = ca.Opti()
        self.opti = opti
        self.vars = {}
        self.effort_terms = []
        self.smooth_terms = []
        tau_cap = ca.DM(b.tau_cap)
        f_cap = b.f_cap

        t0 = self.phi0 * T
        d_w = opti.variable()
        d_s = opti.variable()
        d_f = opti.variable()
        opti.subject_to(d_w >= 0)
        opti.subject_to(opti.bounded(p.d_s_bounds[0], d_s, p.d_s_bounds[1]))
        opti.subject_to(opti.bounded(p.d_f_bounds[0], d_f, p.d_f_bounds[1]))
        opti.subject_to(d_w + d_s <= p.release_deadline_factor * T)
        U_peak = opti.variable()
        self.U_peak = U_peak
        opti.subject_to(opti.bounded(0, U_peak, p.U_max))

        t_s0 = t0 + d_w
        t_l = t_s0 + d_s
        t_c = t_l + d_f

        # ---- W: wait (static hang): effort with the exact pivot acceleration --------------------
        nW = 24
        Uw2 = 0
        for k in range(nW + 1):
            tk = t0 + d_w * k / nW
            dv = self._dev(tk)
            Rw = b.m * (G + dv["aA"][1])
            wgt = 0.5 if k in (0, nW) else 1.0
            Uw2 += wgt * (Rw / f_cap) ** 2
        self.wait_effort = (d_w / nW) * Uw2 / p.t_ref
        if p.wait_in_cost:
            self.effort_terms.append(self.wait_effort)
        a_peak = device.H0 * 1.5 / ((T - p.eps) * p.eps)          # peak |h_ddot|
        opti.subject_to(U_peak >= b.m * (G + a_peak) / f_cap)
        if p.fixed_release_phase is not None:
            # release exactly at device phase phi_l, one period after the (irrelevant) start phase
            opti.subject_to(t_l == (p.fixed_release_phase + 2.0) * T)

        # ---- S: swing on A ----------------------------------------------------------------------
        Bm = ca.DM(chain.B)
        JT = ca.vertcat(ca.DM.eye(2), ca.DM.zeros(NTH, 2))

        def qqd_pinned(t, x, which):
            dv = self._dev(t)
            pp = dv["pA"] if which == "A" else dv["pB"]
            vv = dv["vA"] if which == "A" else dv["vB"]
            aa = dv["aA"] if which == "A" else dv["aB"]
            q = ca.vertcat(pp, x[0:NTH])
            qd = ca.vertcat(vv, x[NTH:])
            return dv, q, qd, aa

        def resS(t, x, a, u, r):
            dv, q, qd, aa = qqd_pinned(t, x, "A")
            qdd = ca.vertcat(aa, a)
            return (chain.f_M(q) @ qdd + chain.f_h(q, qd) - Bm @ (u * tau_cap) - JT @ (r * f_cap)) / f_cap

        def U_S(t, x, a, u, r):
            return ca.sqrt(r[0] ** 2 + r[1] ** 2 + 1e-8)

        def pathS(t, x, a, u, r, at_knot):
            dv, q, qd, aa = qqd_pinned(t, x, "A")
            self._grasp_cone(opti, r, "A")
            if at_knot:
                self._joint_limits(opti, q, qd, "neg")
                self._wall_constraints(opti, q, dv)

        def effS(t, x, a, u, r):
            return 0.25 * ca.sumsqr(u) + U_S(t, x, a, u, r) ** 2

        XS, US, UmS, RS = self._hs_phase("S", p.N_S, d_s, t_s0, 2 * NTH, NTH, NTAU, 2, resS, pathS, effS, U_S)
        opti.subject_to(opti.bounded(-1, ca.vec(US), 1))
        opti.subject_to(opti.bounded(-1, ca.vec(UmS), 1))
        opti.subject_to(XS[:, 0] == 0)

        # ---- F: flight ---------------------------------------------------------------------------
        dvl = self._dev(t_l)
        q_rel = ca.vertcat(dvl["pA"], XS[0:NTH, -1])
        qd_rel = ca.vertcat(dvl["vA"], XS[NTH:, -1])
        G0 = chain.f_com(q_rel)
        V0 = chain.f_comd(q_rel, qd_rel)
        L0 = chain.f_Lg(q_rel, qd_rel)
        gvec = ca.DM([0.0, -G])

        def resF(t, x, a, u, r):
            q, qd = x[0:NQ], x[NQ:]
            return (chain.f_M(q) @ a + chain.f_h(q, qd) - Bm @ (u * tau_cap)) / f_cap

        def pathF(t, x, a, u, r, at_knot):
            q, qd = x[0:NQ], x[NQ:]
            dv = self._dev(t)
            if at_knot:
                self._joint_limits(opti, q, qd, "union")
                self._wall_constraints(opti, q, dv)
                self._no_approach_from_below(opti, q, dv)
                # physical invariants of free flight (anti-exploit, plan Sec. 13.3 tolerances)
                tau_ = t - t_l
                Gk = chain.f_com(q)
                Vk = chain.f_comd(q, qd)
                opti.subject_to(opti.bounded(-p.tol_com, Gk - (G0 + V0 * tau_ + 0.5 * gvec * tau_ ** 2), p.tol_com))
                opti.subject_to(opti.bounded(-p.tol_comv, Vk - (V0 + gvec * tau_), p.tol_comv))
                Lk = chain.f_Lg(q, qd)
                opti.subject_to(opti.bounded(-p.tol_L, Lk - L0, p.tol_L))

        def effF(t, x, a, u, r):
            return 0.25 * ca.sumsqr(u)

        XF, UF, UmF, _ = self._hs_phase("F", p.N_F, d_f, t_l, 2 * NQ, NQ, NTAU, 0, resF, pathF, effF, None)
        opti.subject_to(opti.bounded(-1, ca.vec(UF), 1))
        opti.subject_to(opti.bounded(-1, ca.vec(UmF), 1))
        opti.subject_to(XF[0:2, 0] == dvl["pA"])
        opti.subject_to(XF[NQ:NQ + 2, 0] == dvl["vA"])
        opti.subject_to(XF[2:NQ, 0] == XS[0:NTH, -1])
        opti.subject_to(XF[NQ + 2:, 0] == XS[NTH:, -1])

        # ---- catch: inelastic impact of the hands on B's tip --------------------------------------
        # pre-impact state = end of flight; hand position must coincide with the tip, the hand must
        # move onto the supporting surfaces (down / towards the wall), then M (qd+ - qd-) = J^T Lam.
        dvc = self._dev(t_c)
        qF_end, qdF_end = XF[0:NQ, -1], XF[NQ:, -1]
        opti.subject_to(qF_end[0:2] == dvc["pB"])
        vrel = qdF_end[0:2] - dvc["vB"]
        opti.subject_to(ca.sumsqr(vrel) <= p.v_rel_max ** 2)
        opti.subject_to(vrel[1] <= 0.0)                       # moving down onto the grip surface
        opti.subject_to(vrel[0] >= -p.v_away_max)             # not flying away from the wall
        thd_plus = opti.variable(NTH)
        Lam = opti.variable(2)                                # impulse on the athlete [N s] / f_cap
        dqd = ca.vertcat(dvc["vB"] - qdF_end[0:2], thd_plus - qdF_end[2:])
        imp_res = chain.f_M(qF_end) @ dqd - JT @ (Lam * f_cap)
        opti.subject_to(imp_res[0:2] / f_cap == 0)
        opti.subject_to(imp_res[2:] / 10.0 == 0)
        self._grasp_cone(opti, Lam, "B")
        U_imp = ca.sqrt(Lam[0] ** 2 + Lam[1] ** 2 + 1e-8) / p.delta_catch    # impulse spread over delta_catch, / f_cap
        self.vars["thd_plus"] = thd_plus
        self.vars["Lam"] = Lam
        self.U_imp = U_imp
        d_c = 0.0
        t_h0 = t_c

        # ---- H: hold on B ------------------------------------------------------------------------
        def resH(t, x, a, u, r):
            dv, q, qd, aa = qqd_pinned(t, x, "B")
            qdd = ca.vertcat(aa, a)
            return (chain.f_M(q) @ qdd + chain.f_h(q, qd) - Bm @ (u * tau_cap) - JT @ (r * f_cap)) / f_cap

        def U_H(t, x, a, u, r):
            return ca.sqrt(r[0] ** 2 + r[1] ** 2 + 1e-8)

        def pathH(t, x, a, u, r, at_knot):
            dv, q, qd, aa = qqd_pinned(t, x, "B")
            self._grasp_cone(opti, r, "B")
            if at_knot:
                self._joint_limits(opti, q, qd, "pos")
                self._wall_constraints(opti, q, dv)

        def effH(t, x, a, u, r):
            return 0.25 * ca.sumsqr(u) + U_H(t, x, a, u, r) ** 2

        XH, UH, UmH, RH = self._hs_phase("H", p.N_H, p.T_hold, t_h0, 2 * NTH, NTH, NTAU, 2, resH, pathH, effH, U_H)
        opti.subject_to(opti.bounded(-1, ca.vec(UH), 1))
        opti.subject_to(opti.bounded(-1, ca.vec(UmH), 1))
        opti.subject_to(XH[0:NTH, 0] == qF_end[2:])
        opti.subject_to(XH[NTH:, 0] == thd_plus)
        # catch load = impulse pulse superposed on the sustained tension right after the catch (Sec. 6.3 / 6.4)
        U_H0 = ca.sqrt(RH[0, 0] ** 2 + RH[1, 0] ** 2 + 1e-8)
        U_catch = U_imp + U_H0
        opti.subject_to(U_catch <= U_peak)
        self.effort_terms.append(U_imp ** 2 * p.delta_catch / p.t_ref)
        self.U_catch = U_catch
        # muscle activation continuity across the phase boundaries (rate limit applies within phases)
        opti.subject_to(UF[:, 0] == US[:, -1])
        opti.subject_to(UH[:, 0] == UF[:, -1])

        self.durs = dict(d_w=d_w, d_s=d_s, d_f=d_f)
        self.effort = sum(self.effort_terms)
        self.smooth = sum(self.smooth_terms)
        self.times = dict(t0=t0, t_s0=t_s0, t_l=t_l, t_c=t_c)
        opti.minimize(p.w_E * self.effort + p.w_U * U_peak + p.w_smooth * self.smooth)

    # ------------------------------------------------------------------ initial guess --
    def set_initial(self, d_w=0.0, d_s=3.0, d_f=0.5, swing_amp=0.6, prev=None):
        opti, p, T = self.opti, self.p, self.T
        if prev is not None:
            for name, v in self.vars.items():
                if not v.is_symbolic():
                    continue
                opti.set_initial(v, prev[name])
            for name, v in self.durs.items():
                opti.set_initial(v, prev[name])
            opti.set_initial(self.U_peak, prev["U_peak"])
            return
        opti.set_initial(self.durs["d_w"], d_w)
        opti.set_initial(self.durs["d_s"], d_s)
        opti.set_initial(self.durs["d_f"], d_f)
        opti.set_initial(self.U_peak, 0.9)
        t0 = self.phi0 * T
        t_l = t0 + d_w + d_s
        t_c = t_l + d_f
        # swing: growing oscillation ending with the body ahead (+x) and moving +x
        N = p.N_S
        s = np.linspace(0, 1, N + 1)
        w = 2 * np.pi / 2.2
        th = swing_amp * s ** 2 * np.sin(w * d_s * s)
        thd = np.gradient(th, d_s / N)
        gain = np.array([[0.8], [1.0], [1.0], [1.2], [1.3]])
        XS = np.vstack([np.tile(th, (NTH, 1)) * gain, np.tile(thd, (NTH, 1)) * gain])
        opti.set_initial(self.vars["S_X"], XS)
        opti.set_initial(self.vars["S_U"], 0.0)
        # flight: hand parabola from A tip to B tip
        dA = device.device_state(t_l, T, p.eps)
        dB = device.device_state(t_c, T, p.eps)
        N = p.N_F
        tt = np.linspace(0, d_f, N + 1)
        v0x = (dB["x"] - dA["pA"][0]) / d_f
        v0y = (0.0 - dA["h"] + 0.5 * G * d_f ** 2) / d_f
        XF = np.zeros((2 * NQ, N + 1))
        XF[0] = dA["pA"][0] + v0x * tt
        XF[1] = dA["h"] + v0y * tt - 0.5 * G * tt ** 2
        XF[2:NQ] = np.linspace(XS[0:NTH, -1], np.zeros(NTH), N + 1).T
        XF[NQ] = v0x
        XF[NQ + 1] = v0y - G * tt
        opti.set_initial(self.vars["F_X"], XF)
        opti.set_initial(self.vars["F_U"], 0.0)
        opti.set_initial(self.vars["thd_plus"], 0.0)
        opti.set_initial(self.vars["Lam"], np.array([0.0, 0.1]))
        opti.set_initial(self.vars["H_X"], 0.0)
        opti.set_initial(self.vars["H_U"], 0.0)
        # accelerations / forces: gravity-consistent starting values
        for ph in ("S", "F", "H"):
            opti.set_initial(self.vars[ph + "_A"], 0.0)
            opti.set_initial(self.vars[ph + "_Am"], 0.0)
        opti.set_initial(self.vars["F_A"][1, :], -G)
        opti.set_initial(self.vars["F_Am"][1, :], -G)
        w = self.body.m * G / self.body.f_cap
        for ph in ("S", "H"):
            R0 = np.zeros((2, self.vars[ph + "_R"].shape[1])); R0[1] = w
            opti.set_initial(self.vars[ph + "_R"], R0)
            R0 = np.zeros((2, self.vars[ph + "_Rm"].shape[1])); R0[1] = w
            opti.set_initial(self.vars[ph + "_Rm"], R0)

    # ------------------------------------------------------------------ solve ----------
    def solve(self, max_iter=3000, print_level=0, tol=1e-5, max_cpu_time=1200.0):
        opts = {"expand": True, "ipopt.max_iter": max_iter, "ipopt.print_level": print_level, "print_time": 0,
                "ipopt.tol": tol, "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 10,
                "ipopt.acceptable_constr_viol_tol": 1e-6,
                "ipopt.mu_strategy": "adaptive", "ipopt.linear_solver": "mumps",
                "ipopt.max_cpu_time": max_cpu_time, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)
        try:
            sol = self.opti.solve()
            ok = True
            status = sol.stats()["return_status"]
        except RuntimeError as e:  # infeasible / max iter: still extract the debug values
            sol = self.opti.debug
            ok = False
            status = self.opti.debug.stats().get("return_status", str(e)[:60])
        out = self.extract(sol)
        out["ok"] = ok and status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        out["status"] = status
        out["iters"] = sol.stats().get("iter_count", -1)
        return out

    def extract(self, sol):
        v = {k: np.array(sol.value(x)) for k, x in self.vars.items()}
        for k, x in self.durs.items():
            v[k] = float(sol.value(x))
        v["U_peak"] = float(sol.value(self.U_peak))
        v["U_catch"] = float(sol.value(self.U_catch))
        v["U_imp"] = float(sol.value(self.U_imp))
        v["wait_effort"] = float(sol.value(self.wait_effort))
        v["effort"] = float(sol.value(self.effort))
        v["d_c"] = 0.0
        v["T"], v["phi0"], v["m"] = self.T, self.phi0, self.body.m
        v["params"] = asdict(self.p)
        t0 = self.phi0 * self.T
        v["t0"] = t0
        v["t_s0"] = t0 + v["d_w"]
        v["t_l"] = v["t_s0"] + v["d_s"]
        v["t_c"] = v["t_l"] + v["d_f"]
        v["t_h0"] = v["t_c"]
        v["phi_l"] = (v["t_l"] % (2 * self.T)) / self.T
        v["phi_c"] = (v["t_c"] % (2 * self.T)) / self.T
        return v
