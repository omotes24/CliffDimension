"""Multi-phase direct collocation (Hermite-Simpson) for the planar reduced model.

Phases (absolute device time t; T = full device period; the athlete starts at phase phi0 in [0,1), t0 = phi0 * T)
    W  wait      : static hang on A, duration d_w >= 0 (no dynamics; grip effort accumulates)
    S  swing     : pinned on A, duration d_s, joint torques build the swing
    F  flight    : free chain, duration d_f, only internal torques
    catch        : either an inelastic IMPACT of the hook line on B's tip (rigid limit, default) or a
                   COMPLIANT catch phase C (finite-stiffness spring-damper between the hook line and the
                   ledge, Hunt-Crossley-like damping ramp, cone constraints on the demanded force)
    H  hold      : pinned on B for T_hold = 2.0 s (success criterion of the plan, Sec. 7.3)

Decision variables: durations, states at knots, normalised torques u = tau / tau_cap at knots and
interval mid-points, hand forces in the pinned phases, and the epigraph variable U_peak.

Robust (release-timing) variant: `robust_deltas = (-dw, ..., 0, ..., +dw)` builds K scenarios that share the
swing up to t_l - dw and a common release segment of length 2 dw; scenario k releases at t_l + delta_k.
The flight torque profile is shared by all scenarios (open loop, no hand correction) unless
`robust_shared_flight = False`; catch and hold are scenario specific (the catch is felt). One epigraph
U_peak bounds every scenario, so the optimum is the capacity that makes EVERY release in [-dw, dw] succeed.

Capability parameters (joint torque caps, mu_out, mu_in, v_rel_max, v_away_max, qd_max) are Opti
parameters, so the solution carries the Lagrange multipliers and the parametric sensitivities
dJ*/d ln p ("shadow prices" of the capabilities). Constraints are registered with a category so that the
multiplier mass per category (what limits the motion, and when) can be reported.

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
    release_deadline_factor: float = 1.0   # release within one full period T (plan Sec. 7.3: 2 x one-way time)
    N_S: int = 180
    N_F: int = 30
    N_H: int = 60
    d_s_bounds: tuple = (0.3, 12.0)   # up to 12 s of swing (gradual pumping over several cycles is allowed)
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
    U_max: float = 6.0                # upper bound on U_peak (large, so that "needs superhuman grip" and "unreachable" are distinct)
    fixed_release_phase: float | None = None   # if set: release exactly at device phase phi_l in [0,1) (t_l = (phi_l + 1) T)
    wait_in_cost: bool = True
    w_smooth: float = 1e-3
    # joint ranges [rad] for the "facing -x" convention (elbow, shoulder, hip, knee); mirrored when facing +x
    rel_lo_face_neg: tuple = (0.0, -4.2, -2.1, 0.0)
    rel_hi_face_neg: tuple = (2.6, 0.60, 0.35, 2.6)   # shoulder hyperextension in the hang up to 35 deg (arched swing)
    # ---- catch model -------------------------------------------------------------------------------
    catch_model: str = "impact"       # "impact" (rigid inelastic) or "compliant" (finite stiffness phase C)
    K_catch: float = 40000.0          # [N/m]  hook-line / ledge stiffness (as in simulate.release_window)
    D_catch: float = 1500.0           # [N s/m] damping at full ramp
    ramp_catch: float = 0.01          # [m] Hunt-Crossley-like ramp: damping grows with |deflection| / ramp
    d_c: float = 0.10                 # duration of the compliant catch phase [s]
    N_C: int = 40
    hook_loss: float = 0.06           # max |deflection| during C (hook not lost) [m]
    settle_v: float = 0.30            # max |hand velocity rel. to B| at the end of C [m/s]
    # ---- robust release timing ---------------------------------------------------------------------
    robust_deltas: tuple = ()         # e.g. (-0.02, 0.0, 0.02): scenarios of the release-time error [s]
    robust_shared_flight: bool = True # same open-loop flight torques for all scenarios
    robust_nsub: int = 2              # HS intervals between consecutive scenario release knots
    N_H_scen: int = 40                # hold intervals of the non-nominal scenarios
    # ---- from-state (cost-to-go) solves ------------------------------------------------------------
    from_state: bool = False          # start the swing at time t0 from the given state x0 (no wait phase); the
                                      # multipliers of the initial condition give the costate dJ*/dx0
    x0: tuple = (0.0,) * 10           # initial swing state (th, thd) when from_state


def smax(a, b, delta):
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + delta ** 2))


def _is_const(expr):
    return isinstance(expr, ca.MX) and expr.is_constant()


class PlanarNLP:
    def __init__(self, body: Body, T: float, phi0: float, params: ReducedParams | None = None):
        self.body = body
        self.T = float(T)
        self.phi0 = float(phi0)
        self.p = params or ReducedParams()
        self.chain = PlanarChain(body)
        self._build()

    # ------------------------------------------------------------------ constraint registry ---
    def _con(self, expr, cat, rows=None):
        """opti.subject_to with a category tag (for multiplier accounting).

        rows: canonical row count (default numel); a double inequality with non-constant bounds is
        converted by Opti into two rows, which `_bounded` accounts for."""
        self.opti.subject_to(expr)
        self.cons.append((cat, expr.numel() if rows is None else rows))
        if cat.startswith("cap_"):
            self.cap_cons.append((cat, expr))
        return expr

    def _ge0(self, expr, cat):
        """expr >= 0 that tolerates constant expressions (checks and skips them)."""
        if _is_const(expr):
            val = float(ca.Function("c", [], [expr])()["o0"])
            assert val >= -1e-9, f"constant constraint violated: {val}"
            return
        self._con(expr >= 0, cat)

    def _bounded(self, lo, expr, hi, cat):
        def const(b):
            return not isinstance(b, (ca.MX, ca.SX)) or b.is_constant()
        c = self.opti.bounded(lo, expr, hi)
        self._con(c, cat, rows=c.numel() if (const(lo) and const(hi)) else 2 * c.numel())

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

    def _wall_constraints(self, q, dev):
        """Body points must stay out of both wall bodies (smooth disjunction)."""
        p = self.p
        pts = self.chain.f_tips(q)                # 2 x 5 clearance points
        hand = q[0:2]
        allP = ca.horzcat(hand, pts, self.chain.f_forearm(q))
        clear = np.concatenate([[0.0], self.body.clearance, self.body.forearm_clearance])
        h = dev["h"]
        xB = dev["x"]
        for i in range(allP.shape[1]):
            x, y = allP[0, i], allP[1, i]
            aA = x - (-device.D_LEDGE + clear[i])                  # in front of A's face
            bA = (h - device.WALL_BOTTOM_OFFSET) - y               # below A's body
            self._ge0(smax(aA, bA, p.wall_smooth), "wall")
            aB = (xB + device.D_LEDGE - clear[i]) - x              # in front of B's face
            bB = (-device.WALL_BOTTOM_OFFSET) - y                  # below B's body
            self._ge0(smax(aB, bB, p.wall_smooth), "wall")

    def _no_approach_from_below(self, q, dev):
        hand = q[0:2]
        xB = dev["x"]
        self._ge0(smax(hand[1], (xB - self.p.approach_clear) - hand[0], self.p.wall_smooth), "approach")

    def _grasp_cone(self, r, side, cat=None):
        """r: normalised force on the athlete R / f_cap. side 'A': wall on -x; 'B': wall on +x."""
        mu_out, mu_in = self.par["mu_out"], self.par["mu_in"]
        cat = cat or ("cone_" + side)
        self._con(r[1] >= 0, cat)
        if side == "A":
            self._con(r[0] >= -mu_out * r[1], cat)   # body swung to +x (away from A) -> R_x < 0, friction-limited
            self._con(r[0] <= mu_in * r[1], cat)
        else:
            self._con(r[0] <= mu_out * r[1], cat)    # body swung to -x (away from B) -> R_x > 0
            self._con(r[0] >= -mu_in * r[1], cat)

    def _joint_limits(self, q, qd, facing):
        lo, hi = self._rel_bounds(facing)
        self._bounded(lo, self.chain.f_rel(q), hi, "joint_range")
        qdm = self.par["qd_max"]
        self._con(self.chain.B.T @ qd <= qdm, "joint_speed")
        self._con(self.chain.B.T @ qd >= -qdm, "joint_speed")

    # ------------------------------------------------------------------ generic HS phase
    def _hs_phase(self, name, N, dur, t_start, nx, na, nu, nr, residual, path, effort, U_of=None,
                  U_shared=None):
        """Hermite-Simpson (compressed) phase with IMPLICIT dynamics.

        State x = (pos, vel) with len(vel) = na; accelerations A and the hand force Rv are
        decision variables; residual(t, x, a, u, r) = M qdd + h - B tau - J^T R must vanish at
        knots and mid-points. No matrix inverse appears in the NLP graph (fast, well conditioned).

        path(t, x, a, u, r, at_knot) -> adds path constraints
        effort(t, x, a, u, r)        -> scalar integrand
        U_of(t, x, a, u, r)          -> grasp utilisation U (bounded by U_peak) or None
        U_shared                     -> reuse an existing control matrix (robust scenarios)
        """
        opti = self.opti
        p = self.p
        X = opti.variable(nx, N + 1)
        A = opti.variable(na, N + 1)
        Am = opti.variable(na, N)
        if U_shared is None:
            U = opti.variable(nu, N + 1)
            dt = dur / N
            # activation-rate limit on the torque commands (anti-sawtooth, physiological)
            self._bounded(-p.u_rate * dt, ca.vec(U[:, 1:] - U[:, :-1]), p.u_rate * dt, "rate")
            self._bounded(-1, ca.vec(U), 1, "torque")
        else:
            U = U_shared
            dt = dur / N
        Um = 0.5 * (U[:, :-1] + U[:, 1:])          # piecewise-linear torque commands
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
            self._con(residual(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]) == 0, "dyn")
            E.append(effort(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]))
            path(tk, X[:, k], A[:, k], U[:, k], Rv[:, k], True)
            if U_of is not None:
                self._con(U_of(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]) <= self.U_peak, "cap_" + name[0])
        eff = 0
        smooth = 0
        for k in range(N):
            tm = t_start + dt * (k + 0.5)
            xm = 0.5 * (X[:, k] + X[:, k + 1]) + dt / 8 * (F[k] - F[k + 1])
            fm = ca.vertcat(xm[na:], Am[:, k])
            self._con(residual(tm, xm, Am[:, k], Um[:, k], Rm[:, k]) == 0, "dyn")
            self._con(X[:, k + 1] - X[:, k] - dt / 6 * (F[k] + 4 * fm + F[k + 1]) == 0, "dyn")
            em = effort(tm, xm, Am[:, k], Um[:, k], Rm[:, k])
            eff += dt / 6 * (E[k] + 4 * em + E[k + 1])
            path(tm, xm, Am[:, k], Um[:, k], Rm[:, k], False)
            if U_of is not None:
                self._con(U_of(tm, xm, Am[:, k], Um[:, k], Rm[:, k]) <= self.U_peak, "cap_" + name[0])
            if U_shared is None:
                smooth += ca.sumsqr(U[:, k + 1] - U[:, k])
        self.vars[name + "_X"] = X
        self.vars[name + "_A"] = A
        self.vars[name + "_Am"] = Am
        self.vars[name + "_U"] = U
        self.vars[name + "_Um"] = Um
        self.vars[name + "_R"] = Rv
        self.vars[name + "_Rm"] = Rm
        self._eff_sink.append(eff / self.p.t_ref)
        self.smooth_terms.append(smooth)
        return X, U, Um, Rv

    # ------------------------------------------------------------------ build ---------
    def _build(self):
        p, b, T = self.p, self.body, self.T
        chain = self.chain
        opti = ca.Opti()
        self.opti = opti
        self.vars = {}
        self.cons = []
        self.cap_cons = []
        self.effort_terms = []
        self._eff_sink = self.effort_terms
        self.smooth_terms = []
        # capability parameters (values set in solve(); multipliers give their shadow prices)
        self.par = dict(tau_cap=opti.parameter(NTAU), mu_out=opti.parameter(), mu_in=opti.parameter(),
                        v_rel_max=opti.parameter(), v_away_max=opti.parameter(), qd_max=opti.parameter())
        self.par_values = dict(tau_cap=np.array(b.tau_cap, float), mu_out=float(b.mu_out), mu_in=float(b.mu_in),
                               v_rel_max=float(p.v_rel_max), v_away_max=float(p.v_away_max), qd_max=float(b.qd_max))
        tau_cap = self.par["tau_cap"]
        f_cap = b.f_cap

        deltas = list(p.robust_deltas) if len(p.robust_deltas) else [0.0]
        deltas = sorted(deltas)
        K = len(deltas)
        dw = max(abs(d) for d in deltas) if K > 1 else 0.0
        if K > 1:
            assert abs(deltas[0] + dw) < 1e-12 and abs(deltas[-1] - dw) < 1e-12, "robust_deltas must span [-dw, dw]"
            assert any(abs(d) < 1e-12 for d in deltas), "robust_deltas must contain the nominal 0"
            assert np.allclose(np.diff(deltas), np.diff(deltas)[0]), "robust_deltas must be evenly spaced"
        self.K, self.deltas, self.dw = K, deltas, dw

        # start time and initial swing state as parameters: their multipliers give dJ*/dt0 (= -Hamiltonian) and the
        # costate dJ*/dx0 at the start of the swing (envelope theorem)
        self.par_t0 = opti.parameter()
        self.par_x0 = opti.parameter(2 * NTH)
        self.par_values["t0"] = self.phi0 * T
        self.par_values["x0"] = np.array(p.x0 if p.from_state else np.zeros(2 * NTH), float)
        t0 = self.par_t0
        d_w = opti.variable()
        d_s = opti.variable()
        if p.from_state:
            self._con(d_w == 0, "duration")
        else:
            self._con(d_w >= 0, "duration")
        self._bounded(p.d_s_bounds[0], d_s, p.d_s_bounds[1], "duration")
        self._con(d_w + d_s + dw <= p.release_deadline_factor * T, "duration")
        U_peak = opti.variable()
        self.U_peak = U_peak
        self._bounded(0, U_peak, p.U_max, "cap_bound")

        t_s0 = t0 + d_w
        t_l = t_s0 + d_s                    # nominal release

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
        a_peak = device.H0 * 1.5 / ((0.5 * T - p.eps) * p.eps)    # peak |h_ddot| (one-way time T/2)
        self._con(U_peak >= b.m * (G + a_peak) / f_cap, "cap_wait")
        if p.fixed_release_phase is not None:
            # release exactly at device phase phi_l, one period after the (irrelevant) start phase
            self._con(t_l == (p.fixed_release_phase + 1.0) * T, "duration")

        # ---- S: swing on A ----------------------------------------------------------------------
        Bm = ca.DM(chain.B)
        JT = ca.vertcat(ca.DM.eye(2), ca.DM.zeros(NTH, 2))

        def qqd_pinned(t, x, which, off=None):
            dv = self._dev(t)
            pp = dv["pA"] if which == "A" else dv["pB"]
            vv = dv["vA"] if which == "A" else dv["vB"]
            aa = dv["aA"] if which == "A" else dv["aB"]
            if off is not None:
                pp = pp + off
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
            self._grasp_cone(r, "A")
            if at_knot:
                self._joint_limits(q, qd, "neg")
                self._wall_constraints(q, dv)

        def effS(t, x, a, u, r):
            return 0.25 * ca.sumsqr(u) + U_S(t, x, a, u, r) ** 2

        d_sa = d_s - dw                      # shared swing before the earliest release
        XS, US, UmS, RS = self._hs_phase("S", p.N_S, d_sa, t_s0, 2 * NTH, NTH, NTAU, 2, resS, pathS, effS, U_S)
        self._con(XS[:, 0] - self.par_x0 == 0, "init")     # (expression form keeps the parameter inside g for dJ/dx0)
        # release segment (robust): shared pinned swing of fixed length 2 dw; knots = scenario releases
        if K > 1:
            N_b = p.robust_nsub * (K - 1)
            XSb, USb, UmSb, RSb = self._hs_phase("Sb", N_b, 2 * dw, t_l - dw, 2 * NTH, NTH, NTAU, 2, resS, pathS, effS, U_S)
            self._con(XSb[:, 0] == XS[:, -1], "link")
            self._con(USb[:, 0] == US[:, -1], "link")
            self._con(RSb[:, 0] == RS[:, -1], "link")
            rel_states = [(XSb[:, k * p.robust_nsub], USb[:, k * p.robust_nsub], t_l + deltas[k]) for k in range(K)]
        else:
            rel_states = [(XS[:, -1], US[:, -1], t_l)]

        # ---- per scenario: flight -> catch -> hold ----------------------------------------------------
        gvec = ca.DM([0.0, -G])

        def resF(t, x, a, u, r):
            q, qd = x[0:NQ], x[NQ:]
            return (chain.f_M(q) @ a + chain.f_h(q, qd) - Bm @ (u * tau_cap)) / f_cap

        def effF(t, x, a, u, r):
            return 0.25 * ca.sumsqr(u)

        def resH(t, x, a, u, r, off=None):
            dv, q, qd, aa = qqd_pinned(t, x, "B", off)
            qdd = ca.vertcat(aa, a)
            return (chain.f_M(q) @ qdd + chain.f_h(q, qd) - Bm @ (u * tau_cap) - JT @ (r * f_cap)) / f_cap

        def U_H(t, x, a, u, r):
            return ca.sqrt(r[0] ** 2 + r[1] ** 2 + 1e-8)

        def effH(t, x, a, u, r):
            return 0.25 * ca.sumsqr(u) + U_H(t, x, a, u, r) ** 2

        self.scen = []
        U_F_shared = None
        scen_efforts = []
        for k, (x_rel, u_rel, t_lk) in enumerate(rel_states):
            sfx = "" if K == 1 else f"_{k}"
            nominal = (K == 1) or abs(deltas[k]) < 1e-12
            eff_list = []
            self._eff_sink = eff_list
            d_f = opti.variable()
            self._bounded(p.d_f_bounds[0], d_f, p.d_f_bounds[1], "duration")
            t_c = t_lk + d_f
            dvl = self._dev(t_lk)
            q_rel = ca.vertcat(dvl["pA"], x_rel[0:NTH])
            qd_rel = ca.vertcat(dvl["vA"], x_rel[NTH:])
            G0 = chain.f_com(q_rel)
            V0 = chain.f_comd(q_rel, qd_rel)
            L0 = chain.f_Lg(q_rel, qd_rel)

            def pathF(t, x, a, u, r, at_knot, t_lk=t_lk, G0=G0, V0=V0, L0=L0):
                q, qd = x[0:NQ], x[NQ:]
                dv = self._dev(t)
                if at_knot:
                    self._joint_limits(q, qd, "union")
                    self._wall_constraints(q, dv)
                    self._no_approach_from_below(q, dv)
                    # physical invariants of free flight (anti-exploit, plan Sec. 13.3 tolerances)
                    tau_ = t - t_lk
                    Gk = chain.f_com(q)
                    Vk = chain.f_comd(q, qd)
                    self._bounded(-p.tol_com, Gk - (G0 + V0 * tau_ + 0.5 * gvec * tau_ ** 2), p.tol_com, "flight_inv")
                    self._bounded(-p.tol_comv, Vk - (V0 + gvec * tau_), p.tol_comv, "flight_inv")
                    Lk = chain.f_Lg(q, qd)
                    self._bounded(-p.tol_L, Lk - L0, p.tol_L, "flight_inv")

            XF, UF, UmF, _ = self._hs_phase("F" + sfx, p.N_F, d_f, t_lk, 2 * NQ, NQ, NTAU, 0, resF, pathF, effF, None,
                                            U_shared=U_F_shared)
            if p.robust_shared_flight and U_F_shared is None:
                U_F_shared = UF
            self._con(XF[0:2, 0] == dvl["pA"], "link")
            self._con(XF[NQ:NQ + 2, 0] == dvl["vA"], "link")
            self._con(XF[2:NQ, 0] == x_rel[0:NTH], "link")
            self._con(XF[NQ + 2:, 0] == x_rel[NTH:], "link")
            # muscle activation continuity at the release (rate-limit consistent for shifted scenarios)
            if nominal or not p.robust_shared_flight:
                self._con(UF[:, 0] == u_rel, "link")
            else:
                self._bounded(-p.u_rate * abs(deltas[k]), UF[:, 0] - u_rel, p.u_rate * abs(deltas[k]), "link")

            # ---- catch ------------------------------------------------------------------------------
            dvc = self._dev(t_c)
            qF_end, qdF_end = XF[0:NQ, -1], XF[NQ:, -1]
            self._con(qF_end[0:2] == dvc["pB"], "catch_geom")
            vrel = qdF_end[0:2] - dvc["vB"]
            self._con(ca.sumsqr(vrel) <= self.par["v_rel_max"] ** 2, "catch_vel")
            self._con(vrel[1] <= 0.0, "catch_vel")                       # moving down onto the grip surface
            self._con(vrel[0] >= -self.par["v_away_max"], "catch_vel")   # not flying away from the wall
            N_H = p.N_H if nominal else p.N_H_scen
            if p.catch_model == "impact":
                thd_plus = opti.variable(NTH)
                Lam = opti.variable(2)                                # impulse on the athlete [N s] / f_cap
                dqd = ca.vertcat(dvc["vB"] - qdF_end[0:2], thd_plus - qdF_end[2:])
                imp_res = chain.f_M(qF_end) @ dqd - JT @ (Lam * f_cap)
                self._con(imp_res[0:2] / f_cap == 0, "impact")
                self._con(imp_res[2:] / 10.0 == 0, "impact")
                self._grasp_cone(Lam, "B", "cone_impact")
                U_imp = ca.sqrt(Lam[0] ** 2 + Lam[1] ** 2 + 1e-8) / p.delta_catch    # impulse spread over delta_catch, / f_cap
                self.vars["thd_plus" + sfx] = thd_plus
                self.vars["Lam" + sfx] = Lam
                t_h0 = t_c
                off_H = None
                th_h0, thd_h0 = qF_end[2:], thd_plus
                eff_list.append(U_imp ** 2 * p.delta_catch / p.t_ref)
                u_h0 = UF[:, -1]
            else:
                # ---- C: compliant catch (free chain + spring-damper hand force towards B's moving tip) ----
                def RC_of(t, x):
                    q, qd = x[0:NQ], x[NQ:]
                    dv = self._dev(t)
                    dp = q[0:2] - dv["pB"]
                    dvv = qd[0:2] - dv["vB"]
                    ramp = ca.tanh(ca.sqrt(ca.sumsqr(dp) + 1e-10) / p.ramp_catch)
                    return (-p.K_catch * dp - p.D_catch * ramp * dvv) / f_cap, dp, dvv

                def resC(t, x, a, u, r):
                    q, qd = x[0:NQ], x[NQ:]
                    Rn, _, _ = RC_of(t, x)
                    return (chain.f_M(q) @ a + chain.f_h(q, qd) - Bm @ (u * tau_cap) - JT @ (Rn * f_cap)) / f_cap

                def U_C(t, x, a, u, r):
                    Rn, _, _ = RC_of(t, x)
                    return ca.sqrt(Rn[0] ** 2 + Rn[1] ** 2 + 1e-8)

                def pathC(t, x, a, u, r, at_knot):
                    q, qd = x[0:NQ], x[NQ:]
                    dv = self._dev(t)
                    Rn, dp, dvv = RC_of(t, x)
                    self._grasp_cone(Rn, "B", "cone_C")                   # demanded force inside the admissible cone
                    self._con(ca.sumsqr(dp) <= p.hook_loss ** 2, "hook")
                    if at_knot:
                        self._joint_limits(q, qd, "pos")
                        self._wall_constraints(q, dv)

                def effC(t, x, a, u, r):
                    return 0.25 * ca.sumsqr(u) + U_C(t, x, a, u, r) ** 2

                XC, UC, UmC, _ = self._hs_phase("C" + sfx, p.N_C, p.d_c, t_c, 2 * NQ, NQ, NTAU, 0, resC, pathC, effC, U_C)
                self._con(XC[:, 0] == XF[:, -1], "link")
                self._con(UC[:, 0] == UF[:, -1], "link")
                t_h0 = t_c + p.d_c
                dvh = self._dev(t_h0)
                qC_end, qdC_end = XC[0:NQ, -1], XC[NQ:, -1]
                off_H = qC_end[0:2] - dvh["pB"]                       # static deflection kept during the hold
                vres = qdC_end[0:2] - dvh["vB"]
                self._con(ca.sumsqr(vres) <= p.settle_v ** 2, "settle")
                # residual (small) inelastic impact when the hand is pinned at its deflected position
                thd_plus = opti.variable(NTH)
                Lam = opti.variable(2)
                dqd = ca.vertcat(-vres, thd_plus - qdC_end[2:])
                imp_res = chain.f_M(qC_end) @ dqd - JT @ (Lam * f_cap)
                self._con(imp_res[0:2] / f_cap == 0, "impact")
                self._con(imp_res[2:] / 10.0 == 0, "impact")
                U_imp = ca.sqrt(Lam[0] ** 2 + Lam[1] ** 2 + 1e-8) / p.delta_catch
                self.vars["thd_plus" + sfx] = thd_plus
                self.vars["Lam" + sfx] = Lam
                th_h0, thd_h0 = qC_end[2:], thd_plus
                u_h0 = UC[:, -1]

            # ---- H: hold on B -------------------------------------------------------------------------
            def resHk(t, x, a, u, r, off_H=off_H):
                return resH(t, x, a, u, r, off_H)

            def pathHk(t, x, a, u, r, at_knot, off_H=off_H):
                dv, q, qd, aa = qqd_pinned(t, x, "B", off_H)
                self._grasp_cone(r, "B")
                if at_knot:
                    self._joint_limits(q, qd, "pos")
                    self._wall_constraints(q, dv)

            XH, UH, UmH, RH = self._hs_phase("H" + sfx, N_H, p.T_hold, t_h0, 2 * NTH, NTH, NTAU, 2, resHk, pathHk, effH, U_H)
            self._con(XH[0:NTH, 0] == th_h0, "link")
            self._con(XH[NTH:, 0] == thd_h0, "link")
            self._con(UH[:, 0] == u_h0, "link")
            # catch load = impulse pulse superposed on the sustained tension right after the catch (Sec. 6.3 / 6.4)
            U_H0 = ca.sqrt(RH[0, 0] ** 2 + RH[1, 0] ** 2 + 1e-8)
            U_catch = U_imp + U_H0
            self._con(U_catch <= U_peak, "cap_catch")
            scen_efforts.append(sum(eff_list))
            self.scen.append(dict(k=k, delta=deltas[k], nominal=nominal, d_f=d_f, t_l=t_lk, t_c=t_c, t_h0=t_h0,
                                  U_imp=U_imp, U_catch=U_catch, sfx=sfx))
        self._eff_sink = self.effort_terms
        # effort: shared part + mean over scenarios
        self.effort = sum(self.effort_terms) + sum(scen_efforts) / K
        self.smooth = sum(self.smooth_terms)
        self.durs = dict(d_w=d_w, d_s=d_s)
        for s in self.scen:
            self.durs["d_f" + s["sfx"]] = s["d_f"]
        nom = [s for s in self.scen if s["nominal"]][0]
        self.nom = nom
        self.U_catch, self.U_imp = nom["U_catch"], nom["U_imp"]
        self.times = dict(t0=t0, t_s0=t_s0, t_l=t_l, t_c=nom["t_c"])
        opti.minimize(p.w_E * self.effort + p.w_U * U_peak + p.w_smooth * self.smooth)

    # ------------------------------------------------------------------ initial guess --
    def set_initial(self, d_w=0.0, d_s=3.0, d_f=0.5, swing_amp=0.6, prev=None, swing_period=2.2, pump="quadratic"):
        """Cold start: a growing oscillation of all links (period `swing_period`) over the swing; `pump` = "quadratic"
        (amplitude ~ s^2, i.e. a late build-up) or "linear" (a swing-like gradual build-up over many cycles)."""
        opti, p, T = self.opti, self.p, self.T
        if prev is not None:
            self._set_initial_from_prev(prev, d_f)
            return
        opti.set_initial(self.durs["d_w"], d_w)
        opti.set_initial(self.durs["d_s"], d_s)
        for s in self.scen:
            opti.set_initial(s["d_f"], d_f)
        opti.set_initial(self.U_peak, 0.9)
        t0 = self.phi0 * T
        t_l = t0 + d_w + d_s
        x0 = np.array(p.x0, float) if p.from_state else np.zeros(2 * NTH)
        # swing: growing oscillation ending with the body ahead (+x) and moving +x
        N = p.N_S
        s_ = np.linspace(0, 1, N + 1)
        w = 2 * np.pi / swing_period
        env = s_ ** 2 if pump == "quadratic" else s_
        th = swing_amp * env * np.sin(w * d_s * s_)
        thd = np.gradient(th, d_s / N)
        gain = np.array([[0.8], [1.0], [1.0], [1.2], [1.3]])
        XS = np.vstack([np.tile(th, (NTH, 1)) * gain, np.tile(thd, (NTH, 1)) * gain])
        XS += x0[:, None] * (1 - s_)[None, :]                     # from-state: blend the given start into the guess
        opti.set_initial(self.vars["S_X"], XS)
        opti.set_initial(self.vars["S_U"], 0.0)
        if "Sb_X" in self.vars:
            opti.set_initial(self.vars["Sb_X"], np.tile(XS[:, -1:], (1, self.vars["Sb_X"].shape[1])))
            opti.set_initial(self.vars["Sb_U"], 0.0)
        wgt = self.body.m * G / self.body.f_cap
        for s in self.scen:
            sfx = s["sfx"]
            t_lk = t_l + s["delta"]
            t_c = t_lk + d_f
            dA = device.device_state(t_lk, T, p.eps)
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
            opti.set_initial(self.vars["F" + sfx + "_X"], XF)
            opti.set_initial(self.vars["F" + sfx + "_U"], 0.0)
            opti.set_initial(self.vars["thd_plus" + sfx], 0.0)
            opti.set_initial(self.vars["Lam" + sfx], np.array([0.0, 0.1]))
            if "C" + sfx + "_X" in self.vars:
                XC = np.tile(XF[:, -1:], (1, p.N_C + 1))
                XC[NQ:NQ + 2] = 0.0
                opti.set_initial(self.vars["C" + sfx + "_X"], XC)
                opti.set_initial(self.vars["C" + sfx + "_U"], 0.0)
                opti.set_initial(self.vars["C" + sfx + "_A"], 0.0)
                opti.set_initial(self.vars["C" + sfx + "_Am"], 0.0)
            opti.set_initial(self.vars["H" + sfx + "_X"], 0.0)
            opti.set_initial(self.vars["H" + sfx + "_U"], 0.0)
            for ph in ("F" + sfx, "H" + sfx):
                opti.set_initial(self.vars[ph + "_A"], 0.0)
                opti.set_initial(self.vars[ph + "_Am"], 0.0)
            opti.set_initial(self.vars["F" + sfx + "_A"][1, :], -G)
            opti.set_initial(self.vars["F" + sfx + "_Am"][1, :], -G)
            R0 = np.zeros((2, self.vars["H" + sfx + "_R"].shape[1])); R0[1] = wgt
            opti.set_initial(self.vars["H" + sfx + "_R"], R0)
            R0 = np.zeros((2, self.vars["H" + sfx + "_Rm"].shape[1])); R0[1] = wgt
            opti.set_initial(self.vars["H" + sfx + "_Rm"], R0)
        for ph in ("S",) + (("Sb",) if "Sb_X" in self.vars else ()):
            opti.set_initial(self.vars[ph + "_A"], 0.0)
            opti.set_initial(self.vars[ph + "_Am"], 0.0)
            R0 = np.zeros((2, self.vars[ph + "_R"].shape[1])); R0[1] = wgt
            opti.set_initial(self.vars[ph + "_R"], R0)
            R0 = np.zeros((2, self.vars[ph + "_Rm"].shape[1])); R0[1] = wgt
            opti.set_initial(self.vars[ph + "_Rm"], R0)

    def _set_initial_from_prev(self, prev, d_f_default=0.5):
        """Warm start from a saved solution (nominal, robust or compliant).

        Arrays are resampled along the knot axis when the mesh differs; scenario copies are initialised from
        the nominal solution; a compliant phase missing in `prev` starts from the flight end state; the
        robust release segment from the swing end state."""
        opti, p = self.opti, self.p

        def resample(arr, shape):
            arr = np.array(arr, float)
            if arr.shape == tuple(shape):
                return arr
            if arr.ndim == 2 and arr.shape[0] == shape[0] and arr.shape[1] > 1 and shape[1] > 1:
                s_old = np.linspace(0, 1, arr.shape[1]); s_new = np.linspace(0, 1, shape[1])
                return np.vstack([np.interp(s_new, s_old, row) for row in arr])
            if arr.ndim == 1 and arr.size == shape[0] and shape[1] == 1:
                return arr[:, None]
            return None

        def base_name(name):
            for s_ in self.scen:
                sfx = s_["sfx"]
                if not sfx:
                    continue
                for ph in ("F", "H", "C"):
                    if name.startswith(ph + sfx + "_"):
                        return name.replace(sfx, "", 1)
                if name in ("thd_plus" + sfx, "Lam" + sfx):
                    return name[: -len(sfx)]
            return name

        for name, v in self.vars.items():
            if not v.is_symbolic():
                continue
            shape = tuple(v.shape)
            base = base_name(name)
            if name.startswith("S_") and ("Sa_" + name[2:]) in prev:
                src = prev["Sa_" + name[2:]]
            else:
                src = prev.get(name, prev.get(base))
            arr = resample(src, shape) if src is not None else None
            if arr is not None:
                opti.set_initial(v, arr)
                continue
            kind = base.split("_")[-1]
            if base.startswith("C"):                       # compliant phase absent in prev: start at the flight end
                if kind == "X" and "F_X" in prev:
                    opti.set_initial(v, np.tile(np.array(prev["F_X"])[:, -1:], (1, shape[1])))
                elif kind == "U" and "F_U" in prev:
                    opti.set_initial(v, np.tile(np.array(prev["F_U"])[:, -1:], (1, shape[1])))
                else:
                    opti.set_initial(v, 0.0)
            elif base.startswith("Sb"):                    # robust release segment: continue the swing end
                srcname = "S_" + kind
                if srcname in prev:
                    opti.set_initial(v, np.tile(np.array(prev[srcname])[:, -1:], (1, shape[1])))
            else:
                opti.set_initial(v, 0.0)
        for name, v in self.durs.items():
            if name in prev:
                opti.set_initial(v, prev[name])
            elif name.startswith("d_f"):
                opti.set_initial(v, prev.get("d_f", d_f_default))
        opti.set_initial(self.U_peak, prev["U_peak"])

    # ------------------------------------------------------------------ solve ----------
    def solve(self, max_iter=3000, print_level=0, tol=1e-5, max_cpu_time=1200.0, sensitivities=False):
        opts = {"expand": True, "ipopt.max_iter": max_iter, "ipopt.print_level": print_level, "print_time": 0,
                "ipopt.tol": tol, "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 10,
                "ipopt.acceptable_constr_viol_tol": 1e-6,
                "ipopt.mu_strategy": "adaptive", "ipopt.linear_solver": "mumps",
                "ipopt.max_cpu_time": max_cpu_time, "ipopt.sb": "yes"}
        for k, v in self.par.items():
            self.opti.set_value(v, self.par_values[k])
        self.opti.set_value(self.par_t0, self.par_values["t0"])
        self.opti.set_value(self.par_x0, self.par_values["x0"])
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
        if sensitivities and out["ok"]:
            try:
                out["duals"] = self.dual_report(sol)
            except Exception as e:  # diagnostics must never kill a solve
                out["duals"] = dict(error=str(e)[:200])
        return out

    def dual_report(self, sol):
        """Multiplier mass per constraint category and parametric sensitivities dJ*/d ln p (envelope theorem)."""
        opti = self.opti
        lam = np.array(sol.value(opti.lam_g)).ravel()
        rep = dict(categories={}, n_lam=int(lam.size))
        i = 0
        for cat, n in self.cons:
            seg = lam[i:i + n]
            i += n
            c = rep["categories"].setdefault(cat, dict(n=0, abs_sum=0.0, max_abs=0.0, n_active=0))
            c["n"] += n
            c["abs_sum"] += float(np.abs(seg).sum())
            c["max_abs"] = max(c["max_abs"], float(np.abs(seg).max()) if n else 0.0)
            c["n_active"] += int((np.abs(seg) > 1e-6).sum())
        rep["rows_consistent"] = bool(i == lam.size)
        # time-resolved multipliers of the capacity epigraph constraints (equioscillation weights)
        rep["cap_mult"] = {}
        for cat, expr in self.cap_cons:
            try:
                d = float(np.abs(np.array(sol.value(opti.dual(expr))).ravel()).sum())
            except Exception:
                d = float("nan")
            rep["cap_mult"].setdefault(cat, []).append(d)
        # parametric sensitivities: dJ*/dp = dL/dp at the solution; reported per unit log-parameter
        L = opti.f + ca.dot(opti.lam_g, opti.g)
        sens = {}
        for name, pv in self.par.items():
            g = sol.value(ca.jacobian(L, pv))
            gradp = (g.toarray() if hasattr(g, "toarray") else np.array(g)).ravel()
            val = np.atleast_1d(np.array(self.par_values[name], float))
            sens[name] = (gradp * val).tolist()          # dJ/d ln p  (= p dJ/dp)
        rep["dJ_dlnp"] = sens
        rep["w_U"] = self.p.w_U
        # costate at the start of the swing and the time sensitivity (dJ*/dt0 = -H along the optimal trajectory)
        g = sol.value(ca.jacobian(L, self.par_x0))
        rep["costate_x0"] = (g.toarray() if hasattr(g, "toarray") else np.array(g)).ravel().tolist()
        g = sol.value(ca.jacobian(L, self.par_t0))
        rep["dJ_dt0"] = float(np.array(g).ravel()[0])
        rep["J"] = float(sol.value(opti.f))
        return rep

    def _compliant_loads(self, v, sfx=""):
        """Hand-force history of the compliant catch phase from the knot states (numpy, same law as the NLP)."""
        p, f_cap = self.p, self.body.f_cap
        X = np.array(v["C" + sfx + "_X"])
        N = X.shape[1] - 1
        t_c = v["t_c"]
        U = np.zeros(N + 1)
        Rh = np.zeros((2, N + 1))
        for k in range(N + 1):
            dv = device.device_state(t_c + p.d_c * k / N, self.T, p.eps)
            dp = X[0:2, k] - dv["pB"]
            dvv = X[NQ:NQ + 2, k] - dv["vB"]
            ramp = np.tanh(np.linalg.norm(dp) / p.ramp_catch)
            R = -p.K_catch * dp - p.D_catch * ramp * dvv
            Rh[:, k] = R
            U[k] = np.linalg.norm(R) / f_cap
        dt = p.d_c / N
        w = max(int(round(0.01 / dt)), 1)
        U10 = np.convolve(U, np.ones(w) / w, mode="valid").max() if N + 1 >= w else U.max()
        return dict(C_R=Rh / f_cap, C_Uhist=U, U_catch_C=float(U.max()), U_catch_10ms=float(U10),
                    U_catch_impact_equiv=float(v["U_catch"]), U_catch=float(U.max()))

    def _merge_swing(self, v, k_nom):
        """Concatenate S (duration d_s - dw) and the first k_nom intervals of Sb (dt = 2dw/N_b) and resample onto a
        uniform mesh with the same number of intervals (cubic Hermite for the states, linear for torques/forces)."""
        N_S = self.p.N_S
        dw = self.dw
        d_sa = v["d_s"] - dw
        tS = np.linspace(0.0, d_sa, N_S + 1)
        dtb = 2 * dw / (self.p.robust_nsub * (self.K - 1))
        tB = d_sa + dtb * np.arange(1, k_nom + 1)
        tk = np.concatenate([tS, tB])                                   # knot times (non-uniform)
        X = np.hstack([v["Sa_X"], v["Sb_X"][:, 1:k_nom + 1]])
        A = np.hstack([v["Sa_A"], v["Sb_A"][:, 1:k_nom + 1]])
        U = np.hstack([v["Sa_U"], v["Sb_U"][:, 1:k_nom + 1]])
        R = np.hstack([v["Sa_R"], v["Sb_R"][:, 1:k_nom + 1]])
        n = NTH
        Nn = N_S + k_nom
        tu = np.linspace(0.0, tk[-1], Nn + 1)
        def hermite(t):
            j = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, len(tk) - 2)
            h = tk[j + 1] - tk[j]; u = (t - tk[j]) / h
            p0, p1 = X[:n, j], X[:n, j + 1]; v0, v1 = X[n:, j], X[n:, j + 1]; a0, a1 = A[:, j], A[:, j + 1]
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
            pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
            vel = h00 * v0 + h10 * h * a0 + h01 * v1 + h11 * h * a1
            acc = (1 - u) * a0 + u * a1
            return pos, vel, acc
        Xu = np.zeros((2 * n, Nn + 1)); Au = np.zeros((n, Nn + 1))
        for i, t in enumerate(tu):
            pos, vel, acc = hermite(min(t, tk[-1]))
            Xu[:n, i], Xu[n:, i], Au[:, i] = pos, vel, acc
        Uu = np.vstack([np.interp(tu, tk, row) for row in U])
        Ru = np.vstack([np.interp(tu, tk, row) for row in R])
        tm = 0.5 * (tu[:-1] + tu[1:])
        Amu = np.zeros((n, Nn))
        for i, t in enumerate(tm):
            Amu[:, i] = hermite(t)[2]
        v["S_X"], v["S_A"], v["S_Am"], v["S_U"] = Xu, Au, Amu, Uu
        v["S_Um"] = 0.5 * (Uu[:, :-1] + Uu[:, 1:])
        v["S_R"], v["S_Rm"] = Ru, 0.5 * (Ru[:, :-1] + Ru[:, 1:])

    def extract(self, sol):
        v = {k: np.array(sol.value(x)) for k, x in self.vars.items()}
        for k, x in self.durs.items():
            v[k] = float(sol.value(x))
        if "d_f" not in v:                       # robust: nominal scenario's flight time under the plain key
            v["d_f"] = v["d_f" + self.nom["sfx"]]
        v["U_peak"] = float(sol.value(self.U_peak))
        v["U_catch"] = float(sol.value(self.U_catch))
        v["U_imp"] = float(sol.value(self.U_imp))
        v["wait_effort"] = float(sol.value(self.wait_effort))
        v["effort"] = float(sol.value(self.effort))
        v["d_c"] = self.p.d_c if self.p.catch_model == "compliant" else 0.0
        v["T"], v["phi0"], v["m"] = self.T, self.phi0, self.body.m
        v["params"] = asdict(self.p)
        v["x0"] = np.array(self.par_values["x0"], float)
        v["J"] = float(sol.value(self.opti.f))
        t0 = self.phi0 * self.T
        v["t0"] = t0
        v["t_s0"] = t0 + v["d_w"]
        v["t_l"] = v["t_s0"] + v["d_s"]
        v["t_c"] = v["t_l"] + v["d_f"]
        v["t_h0"] = v["t_c"] + v["d_c"]
        v["phi_l"] = (v["t_l"] % self.T) / self.T
        v["phi_c"] = (v["t_c"] % self.T) / self.T
        v["T_convention"] = "period"
        v["catch_model"] = self.p.catch_model
        # reach diagnostic: distance between the hook line and B's tip at the end of the flight (0 when feasible)
        try:
            dvc = device.device_state(v["t_c"], self.T, self.p.eps)
            v["catch_gap"] = float(np.linalg.norm(np.array(v["F_X"])[0:2, -1] - dvc["pB"]))
        except Exception:
            v["catch_gap"] = float("nan")
        if self.p.catch_model == "compliant":
            v.update(self._compliant_loads(v, self.nom["sfx"] if self.K > 1 else ""))
        if self.K > 1:
            # nominal scenario under the plain keys, all scenarios in "scen"
            nom = self.nom
            sfx = nom["sfx"]
            for key in ("F_X", "F_A", "F_Am", "F_U", "F_Um", "H_X", "H_A", "H_Am", "H_U", "H_Um", "H_R", "H_Rm",
                        "C_X", "C_A", "C_Am", "C_U", "C_Um", "thd_plus", "Lam"):
                src = (key + sfx) if key in ("thd_plus", "Lam") else key.replace("_", sfx + "_", 1)
                if src in v:
                    v[key] = v[src]
            # merged nominal swing (S + Sb up to the nominal release knot) on a uniform mesh under the plain keys;
            # the raw S variables stay available as "Sa_*" for warm starts
            k_nom = int(round((0.0 - self.deltas[0]) / (self.deltas[1] - self.deltas[0]))) * self.p.robust_nsub
            for kind in ("X", "A", "Am", "U", "Um", "R", "Rm"):
                v["Sa_" + kind] = v["S_" + kind]
            self._merge_swing(v, k_nom)
            v["scen"] = [dict(k=s["k"], delta=s["delta"], d_f=float(sol.value(s["d_f"])),
                              U_catch=float(sol.value(s["U_catch"])), U_imp=float(sol.value(s["U_imp"])))
                         for s in self.scen]
            v["robust_dw"] = self.dw
        return v
