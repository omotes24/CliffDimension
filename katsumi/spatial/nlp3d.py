"""Multi-phase Hermite-Simpson collocation for the SPATIAL reduced model (twist axis, two hands).

Phases (absolute device time t, T = full period, start phase phi0)
    W    wait        static two-hand hang on A (grip effort only)
    S    swing       both hands pinned on A (grip width d_A), duration d_s
    S1   one-hand    (optional, release_first in {"L","R"}) the other hand still on A, duration d_r in [0, dr_max]
    F    flight      free 14-DoF chain, duration d_f
    catch 1          inelastic impact of the first hand (catch_first) or of both hands on B
    C1   one-hand    (optional) only the first hand on B, duration d_c in [0, dc_max]
    catch 2          impact of the second hand (impulses on both hands)
    H    hold        both hands pinned on B for T_hold

Hands are point contacts on the ledge line; the ledge exerts a 3-D force per hand inside an admissible set
(support R_z >= 0, hook/friction cone in x with mu_out / mu_in, lateral friction |R_y| <= mu_lat R_z).
Per-hand grip utilisation U_i = |R_i| / f_hand; the epigraph U_peak bounds U_i at every collocation point,
the catch loads of both hands, and is the quantity minimised ("required per-hand grip capacity").

Pinning is imposed at the position level (hand point = ledge point) at knots and mid-points, with the
constraint forces as decision variables in the implicit dynamics residual.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import casadi as ca

from .. import device
from ..planar.anthro import G
from .anthro3d import Body3D
from .model3d import SpatialChain, NQ, NTAU, IDX


@dataclass
class Params3D:
    eps: float = 0.20
    T_hold: float = 1.5
    release_deadline_factor: float = 1.0
    N_S: int = 100
    N_F: int = 20
    N_H: int = 30
    N_1: int = 8                       # intervals of the one-hand phases
    d_s_bounds: tuple = (0.3, 6.0)
    d_f_bounds: tuple = (0.10, 1.20)
    dr_max: float = 0.30               # max release stagger [s]
    dc_max: float = 0.30               # max catch stagger [s]
    release_first: str | None = None   # None = simultaneous release
    catch_first: str | None = None     # None = simultaneous catch
    grip_A: float = 0.39               # hand spacing on A [m] (0.39 = shoulder width, 0.10 = hands together)
    grip_B_bounds: tuple = (0.10, 0.60)
    yB_bounds: tuple = (-0.40, 0.40)   # lateral position of the catch on B [m]
    yaw_tol: float = np.deg2rad(20.0)  # |yaw error| at the catch (facing B)
    v_rel_max: float = 4.0
    v_away_max: float = 0.5
    delta_catch: float = 0.05
    approach_clear: float = 0.05
    wall_smooth: float = 0.01
    tol_com: float = 5e-3
    tol_comv: float = 2e-2
    tol_L: float = 0.15
    u_rate: float = 10.0
    t_ref: float = 1.0
    w_E: float = 1.0
    w_U: float = 10.0
    w_smooth: float = 1e-3
    U_max: float = 3.5
    fixed_release_phase: float | None = None
    wait_in_cost: bool = False
    # joint ranges (flex, abd, elbow) per arm, hip, knee; trunk roll / pitch / rate bounds
    arm_lo: tuple = (-0.9, -0.6, 0.0)
    arm_hi: tuple = (3.4, 1.8, 2.6)
    hip_range: tuple = (-0.35, 2.1)
    knee_range: tuple = (0.0, 2.6)
    roll_max: float = 1.2
    pitch_max: float = 2.5
    trunk_rate_max: float = 15.0
    clearances: dict | None = None


def smax(a, b, delta):
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + delta ** 2))


def _is_const(expr):
    return isinstance(expr, ca.MX) and expr.is_constant()


CLEAR_DEFAULT = dict(hand_L=0.0, hand_R=0.0, wrist_L=0.03, wrist_R=0.03, elbow_L=0.06, elbow_R=0.06,
                     shoulder_L=0.12, shoulder_R=0.12, head=0.12, S0=0.12, hip=0.12, knee=0.07, ankle=0.05, toe=0.05)


class SpatialNLP:
    def __init__(self, body: Body3D, T: float, phi0: float, params: Params3D | None = None):
        self.body = body
        self.T = float(T)
        self.phi0 = float(phi0)
        self.p = params or Params3D()
        self.chain = SpatialChain(body)
        self._build()

    # ------------------------------------------------------------------ registry / helpers ---
    def _con(self, expr, cat, rows=None):
        self.opti.subject_to(expr)
        self.cons.append((cat, expr.numel() if rows is None else rows))
        if cat.startswith("cap_"):
            self.cap_cons.append((cat, expr))

    def _ge0(self, expr, cat):
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

    def _dev(self, t):
        d = device.device_ca(t, self.T, self.p.eps)
        return d

    def ledge_point(self, dv, which, y):
        """3-D point of the ledge line of cliff A (x=0, z=h) or B (x=x_B, z=0) at lateral coordinate y, + velocity/acc."""
        if which == "A":
            pos = ca.vertcat(0.0, y, dv["h"])
            vel = ca.vertcat(0.0, 0.0, dv["vA"][1])
            acc = ca.vertcat(0.0, 0.0, dv["aA"][1])
        else:
            pos = ca.vertcat(dv["x"], y, 0.0)
            vel = ca.vertcat(dv["vB"][0], 0.0, 0.0)
            acc = ca.vertcat(dv["aB"][0], 0.0, 0.0)
        return pos, vel, acc

    def _cone(self, r, side, cat):
        """r = R / f_hand (world). side 'A': wall at -x of the hands; 'B': wall at +x."""
        mu_out, mu_in, mu_lat = self.par["mu_out"], self.par["mu_in"], self.par["mu_lat"]
        self._con(r[2] >= 0, cat)
        if side == "A":
            self._con(r[0] >= -mu_out * r[2], cat)
            self._con(r[0] <= mu_in * r[2], cat)
        else:
            self._con(r[0] <= mu_out * r[2], cat)
            self._con(r[0] >= -mu_in * r[2], cat)
        self._con(r[1] <= mu_lat * r[2], cat)
        self._con(r[1] >= -mu_lat * r[2], cat)

    def _joint_limits(self, q, qd):
        p = self.p
        lo = list(p.arm_lo) * 2 + [p.hip_range[0], p.knee_range[0]]
        hi = list(p.arm_hi) * 2 + [p.hip_range[1], p.knee_range[1]]
        self._bounded(np.array(lo), q[6:], np.array(hi), "joint_range")
        self._bounded(-p.roll_max, q[4], p.roll_max, "trunk_range")
        self._bounded(-p.pitch_max, q[5], p.pitch_max, "trunk_range")
        qdm = self.par["qd_max"]
        self._con(qd[6:] <= qdm, "joint_speed")
        self._con(qd[6:] >= -qdm, "joint_speed")
        self._bounded(-p.trunk_rate_max, qd[3:6], p.trunk_rate_max, "trunk_range")

    def _wall_constraints(self, q, dv):
        p = self.p
        P = self.chain.f_points(q)
        names = self.chain.point_names
        clear = dict(CLEAR_DEFAULT)
        if p.clearances:
            clear.update(p.clearances)
        h, xB = dv["h"], dv["x"]
        for i, nm in enumerate(names):
            c = clear.get(nm, 0.05)
            x, z = P[0, i], P[2, i]
            aA = x - (-device.D_LEDGE + c)
            bA = (h - device.WALL_BOTTOM_OFFSET) - z
            self._ge0(smax(aA, bA, p.wall_smooth), "wall")
            aB = (xB + device.D_LEDGE - c) - x
            bB = (-device.WALL_BOTTOM_OFFSET) - z
            self._ge0(smax(aB, bB, p.wall_smooth), "wall")

    def _no_approach_from_below(self, hand, dv):
        self._ge0(smax(hand[2], (dv["x"] - self.p.approach_clear) - hand[0], self.p.wall_smooth), "approach")

    def _wrist(self, q, sides):
        dL, dR, e1, e2, e3 = self.chain.f_dirs(q)
        s = float(np.sin(self.body.wrist_dev_max))
        for side, d in (("L", dL), ("R", dR)):
            if side in sides:
                self._bounded(-s, d[1], s, "wrist")

    # ------------------------------------------------------------------ generic HS phase -----
    def _hs_phase(self, name, N, dur, t_start, nr, residual, path, effort, U_list):
        """x = (q, qd) (28), accelerations A (14), torques U (8), hand forces R (nr = 0, 3 or 6)."""
        opti, p = self.opti, self.p
        nx, na, nu = 2 * NQ, NQ, NTAU
        X = opti.variable(nx, N + 1)
        A = opti.variable(na, N + 1)
        Am = opti.variable(na, N)
        U = opti.variable(nu, N + 1)
        dt = dur / N
        self._bounded(-p.u_rate * dt, ca.vec(U[:, 1:] - U[:, :-1]), p.u_rate * dt, "rate")
        self._bounded(-1, ca.vec(U), 1, "torque")
        Um = 0.5 * (U[:, :-1] + U[:, 1:])
        if nr == 0:
            Rv = ca.MX.zeros(1, N + 1); Rm = ca.MX.zeros(1, N)
        else:
            Rv = opti.variable(nr, N + 1); Rm = opti.variable(nr, N)
        F, E = [], []
        for k in range(N + 1):
            tk = t_start + dt * k
            F.append(ca.vertcat(X[na:, k], A[:, k]))
            self._con(residual(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]) == 0, "dyn")
            E.append(effort(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]))
            path(tk, X[:, k], A[:, k], U[:, k], Rv[:, k], True)
            for Uf in U_list:
                self._con(Uf(tk, X[:, k], A[:, k], U[:, k], Rv[:, k]) <= self.U_peak, "cap_" + name[0])
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
            for Uf in U_list:
                self._con(Uf(tm, xm, Am[:, k], Um[:, k], Rm[:, k]) <= self.U_peak, "cap_" + name[0])
            smooth += ca.sumsqr(U[:, k + 1] - U[:, k])
        self.vars[name + "_X"] = X; self.vars[name + "_A"] = A; self.vars[name + "_Am"] = Am
        self.vars[name + "_U"] = U; self.vars[name + "_Um"] = Um; self.vars[name + "_R"] = Rv; self.vars[name + "_Rm"] = Rm
        self.effort_terms.append(eff / p.t_ref)
        self.smooth_terms.append(smooth)
        self.phase_meta[name] = dict(N=N, nr=nr)
        return X, U, Um, Rv

    # ------------------------------------------------------------------ build -----------------
    def _build(self):
        p, b, T = self.p, self.body, self.T
        ch = self.chain
        opti = ca.Opti()
        self.opti = opti
        self.vars, self.cons, self.cap_cons, self.effort_terms, self.smooth_terms, self.phase_meta = {}, [], [], [], [], {}
        self.par = dict(tau_cap=opti.parameter(NTAU), mu_out=opti.parameter(), mu_in=opti.parameter(),
                        mu_lat=opti.parameter(), v_rel_max=opti.parameter(), v_away_max=opti.parameter(),
                        qd_max=opti.parameter())
        self.par_values = dict(tau_cap=np.array(b.tau_cap, float), mu_out=float(b.mu_out), mu_in=float(b.mu_in),
                               mu_lat=float(b.mu_lat), v_rel_max=float(p.v_rel_max), v_away_max=float(p.v_away_max),
                               qd_max=float(b.qd_max))
        tau_cap = self.par["tau_cap"]
        fh = b.f_hand
        Bm = ca.DM(ch.B)

        t0 = self.phi0 * T
        d_w = opti.variable(); d_s = opti.variable(); d_f = opti.variable()
        d_r = opti.variable() if p.release_first else 0.0
        d_c = opti.variable() if p.catch_first else 0.0
        self._con(d_w >= 0, "duration")
        self._bounded(p.d_s_bounds[0], d_s, p.d_s_bounds[1], "duration")
        self._bounded(p.d_f_bounds[0], d_f, p.d_f_bounds[1], "duration")
        if p.release_first:
            self._bounded(0.0, d_r, p.dr_max, "duration")
        if p.catch_first:
            self._bounded(0.0, d_c, p.dc_max, "duration")
        self._con(d_w + d_s + d_r <= p.release_deadline_factor * T, "duration")
        U_peak = opti.variable()
        self.U_peak = U_peak
        self._bounded(0, U_peak, p.U_max, "cap_bound")
        dB = opti.variable(); yB = opti.variable()
        self._bounded(p.grip_B_bounds[0], dB, p.grip_B_bounds[1], "grip")
        self._bounded(p.yB_bounds[0], yB, p.yB_bounds[1], "grip")
        self.vars["dB"] = dB; self.vars["yB"] = yB

        t_s0 = t0 + d_w
        t_r1 = t_s0 + d_s               # first release (both, or the release_first hand)
        t_l = t_r1 + d_r                # last hand leaves A
        t_c = t_l + d_f                 # first catch
        t_h0 = t_c + d_c                # both hands on B
        if p.fixed_release_phase is not None:
            self._con(t_l == (p.fixed_release_phase + 1.0) * T, "duration")

        # lateral hand coordinates on A (facing A: left hand at -y) and on B (facing B: left hand at +y)
        yA = {"L": -0.5 * p.grip_A, "R": 0.5 * p.grip_A}
        yBs = {"L": yB + 0.5 * dB, "R": yB - 0.5 * dB}

        # ---- W: wait -------------------------------------------------------------------------------
        nW = 24
        Uw2 = 0
        for k in range(nW + 1):
            dv = self._dev(t0 + d_w * k / nW)
            Rw = 0.5 * b.m * (G + dv["aA"][1]) / fh
            Uw2 += (0.5 if k in (0, nW) else 1.0) * Rw ** 2
        self.wait_effort = (d_w / nW) * Uw2 / p.t_ref
        if p.wait_in_cost:
            self.effort_terms.append(self.wait_effort)
        a_peak = device.H0 * 1.5 / ((0.5 * T - p.eps) * p.eps)
        self._con(U_peak >= 0.5 * b.m * (G + a_peak) / fh, "cap_wait")

        # ---- generic pieces -------------------------------------------------------------------------
        def residual_of(attached, which, ys):
            """attached: list of sides pinned to cliff `which` at lateral coordinates ys[side]."""
            def res(t, x, a, u, r):
                q, qd = x[0:NQ], x[NQ:]
                RL = ca.MX.zeros(3); RR = ca.MX.zeros(3)
                for i, s in enumerate(attached):
                    if s == "L":
                        RL = r[3 * i:3 * i + 3] * fh
                    else:
                        RR = r[3 * i:3 * i + 3] * fh
                return ch.f_res(q, qd, a, u * tau_cap, RL, RR) / fh
            return res

        def U_of_hand(i):
            def Uf(t, x, a, u, r):
                return ca.sqrt(ca.sumsqr(r[3 * i:3 * i + 3]) + 1e-8)
            return Uf

        def effort_of(n_att):
            def eff(t, x, a, u, r):
                e = ca.sumsqr(u) / NTAU
                for i in range(n_att):
                    e = e + 0.5 * ca.sumsqr(r[3 * i:3 * i + 3])
                return e
            return eff

        def path_of(attached, which, ys, free_hands_check=()):
            side_cone = which

            def path(t, x, a, u, r, at_knot):
                q, qd = x[0:NQ], x[NQ:]
                dv = self._dev(t)
                hL, hR = ch.f_hands(q)
                H = {"L": hL, "R": hR}
                for i, s in enumerate(attached):
                    pos, vel, acc = self.ledge_point(dv, which, ys[s])
                    self._con(H[s] - pos == 0, "pin")
                    self._cone(r[3 * i:3 * i + 3], side_cone, "cone_" + which)
                if at_knot:
                    self._joint_limits(q, qd)
                    self._wall_constraints(q, dv)
                    self._wrist(q, attached)
                    for s in free_hands_check:
                        self._no_approach_from_below(H[s], dv)
            return path

        # ---- S: swing, both hands on A ---------------------------------------------------------------
        XS, US, UmS, RS = self._hs_phase("S", p.N_S, d_s, t_s0, 6, residual_of(["L", "R"], "A", yA),
                                         path_of(["L", "R"], "A", yA), effort_of(2), [U_of_hand(0), U_of_hand(1)])
        # start at rest, facing A, trunk upright, legs straight (arm angles / position by the pin constraints)
        self._con(XS[NQ:, 0] == 0, "init")
        self._con(XS[3, 0] == np.pi, "init")
        self._con(XS[4:6, 0] == 0, "init")
        self._con(XS[12:14, 0] == 0, "init")
        x_last, u_last, t_last = XS[:, -1], US[:, -1], t_r1
        # ---- S1: one hand remains on A ---------------------------------------------------------------
        if p.release_first:
            keep = "R" if p.release_first == "L" else "L"
            XS1, US1, UmS1, RS1 = self._hs_phase("S1", p.N_1, d_r, t_r1, 3, residual_of([keep], "A", yA),
                                                 path_of([keep], "A", yA), effort_of(1), [U_of_hand(0)])
            self._con(XS1[:, 0] == XS[:, -1], "link")
            self._con(US1[:, 0] == US[:, -1], "link")
            x_last, u_last, t_last = XS1[:, -1], US1[:, -1], t_l
        # ---- F: flight ------------------------------------------------------------------------------
        q_rel, qd_rel = x_last[0:NQ], x_last[NQ:]
        G0 = ch.f_com(q_rel); V0 = ch.f_comd(q_rel, qd_rel); L0 = ch.f_Lg(q_rel, qd_rel)
        gvec = ca.DM([0.0, 0.0, -G])

        def pathF(t, x, a, u, r, at_knot):
            q, qd = x[0:NQ], x[NQ:]
            dv = self._dev(t)
            if at_knot:
                self._joint_limits(q, qd)
                self._wall_constraints(q, dv)
                hL, hR = ch.f_hands(q)
                self._no_approach_from_below(hL, dv)
                self._no_approach_from_below(hR, dv)
                tau_ = t - t_l
                self._bounded(-p.tol_com, ch.f_com(q) - (G0 + V0 * tau_ + 0.5 * gvec * tau_ ** 2), p.tol_com, "flight_inv")
                self._bounded(-p.tol_comv, ch.f_comd(q, qd) - (V0 + gvec * tau_), p.tol_comv, "flight_inv")
                self._bounded(-p.tol_L, ch.f_Lg(q, qd) - L0, p.tol_L, "flight_inv")

        XF, UF, UmF, _ = self._hs_phase("F", p.N_F, d_f, t_l, 0, residual_of([], None, {}), pathF, effort_of(0), [])
        self._con(XF[:, 0] == x_last, "link")
        self._con(UF[:, 0] == u_last, "link")

        # ---- catch 1 -------------------------------------------------------------------------------
        def catch_conditions(q, qd, dv, side, y):
            hL, hR = ch.f_hands(q)
            JL, JR = ch.f_J(q)
            H = {"L": hL, "R": hR}; J = {"L": JL, "R": JR}
            pos, vel, acc = self.ledge_point(dv, "B", y)
            self._con(H[side] - pos == 0, "catch_geom")
            vrel = J[side] @ qd - vel
            self._con(ca.sumsqr(vrel) <= self.par["v_rel_max"] ** 2, "catch_vel")
            self._con(vrel[2] <= 0.0, "catch_vel")
            self._con(vrel[0] >= -self.par["v_away_max"], "catch_vel")
            return vel

        def impact(q, qd_minus, attached_before, new_sides, vels, tag):
            """Inelastic impact: M (qd+ - qd-) = sum_i J_i^T Lam_i, J_i qd+ = v_i for all attached hands."""
            qd_plus = opti.variable(NQ)
            JL, JR = ch.f_J(q)
            J = {"L": JL, "R": JR}
            sides = list(attached_before) + list(new_sides)
            Lams = {}
            rhs = 0
            for s in sides:
                Lam = opti.variable(3)
                Lams[s] = Lam
                rhs = rhs + J[s].T @ (Lam * fh)
                self._con(J[s] @ qd_plus - vels[s] == 0, "impact")
                self._cone(Lam, "B", "cone_impact")
            res = ch.f_M(q) @ (qd_plus - qd_minus) - rhs
            self._con(res / fh == 0, "impact")
            self.vars["qd_plus_" + tag] = qd_plus
            for s, Lam in Lams.items():
                self.vars[f"Lam_{tag}_{s}"] = Lam
            return qd_plus, Lams

        dvc = self._dev(t_c)
        qF, qdF = XF[0:NQ, -1], XF[NQ:, -1]
        # facing B at the first catch
        dL_, dR_, e1c, e2c, e3c = ch.f_dirs(qF)
        self._con(e1c[0] >= np.cos(p.yaw_tol), "facing")
        self.catch_loads = {}
        if p.catch_first:
            c1 = p.catch_first
            c2 = "R" if c1 == "L" else "L"
            v1 = catch_conditions(qF, qdF, dvc, c1, yBs[c1])
            self._wrist(qF, [c1])
            qd1, Lam1 = impact(qF, qdF, [], [c1], {c1: v1}, "c1")
            x_c1 = ca.vertcat(qF, qd1)
            XC1, UC1, UmC1, RC1 = self._hs_phase("C1", p.N_1, d_c, t_c, 3, residual_of([c1], "B", yBs),
                                                 path_of([c1], "B", yBs, free_hands_check=[c2]), effort_of(1), [U_of_hand(0)])
            self._con(XC1[:, 0] == x_c1, "link")
            self._con(UC1[:, 0] == UF[:, -1], "link")
            U_imp1 = ca.sqrt(ca.sumsqr(Lam1[c1]) + 1e-8) / p.delta_catch
            U_c1 = U_imp1 + ca.sqrt(ca.sumsqr(RC1[0:3, 0]) + 1e-8)
            self._con(U_c1 <= U_peak, "cap_catch")
            self.catch_loads[c1 + "_first"] = U_c1
            self.effort_terms.append(U_imp1 ** 2 * p.delta_catch / p.t_ref)
            # second catch at the end of C1
            dvh = self._dev(t_h0)
            qC, qdC = XC1[0:NQ, -1], XC1[NQ:, -1]
            v2 = catch_conditions(qC, qdC, dvh, c2, yBs[c2])
            self._wrist(qC, [c2])
            pos1, vel1, acc1 = self.ledge_point(dvh, "B", yBs[c1])
            qd2, Lam2 = impact(qC, qdC, [c1], [c2], {c1: vel1, c2: v2}, "c2")
            x_h0 = ca.vertcat(qC, qd2)
            u_h0 = UC1[:, -1]
            imp_second = {c1: Lam2[c1], c2: Lam2[c2]}
        else:
            vL = catch_conditions(qF, qdF, dvc, "L", yBs["L"])
            vR = catch_conditions(qF, qdF, dvc, "R", yBs["R"])
            self._wrist(qF, ["L", "R"])
            qd2, Lam2 = impact(qF, qdF, [], ["L", "R"], {"L": vL, "R": vR}, "c2")
            x_h0 = ca.vertcat(qF, qd2)
            u_h0 = UF[:, -1]
            imp_second = {"L": Lam2["L"], "R": Lam2["R"]}

        # ---- H: hold, both hands on B --------------------------------------------------------------
        XH, UH, UmH, RH = self._hs_phase("H", p.N_H, p.T_hold, t_h0, 6, residual_of(["L", "R"], "B", yBs),
                                         path_of(["L", "R"], "B", yBs), effort_of(2), [U_of_hand(0), U_of_hand(1)])
        self._con(XH[:, 0] == x_h0, "link")
        self._con(UH[:, 0] == u_h0, "link")
        for i, s in enumerate(["L", "R"]):
            U_imp = ca.sqrt(ca.sumsqr(imp_second[s]) + 1e-8) / p.delta_catch
            U_cs = U_imp + ca.sqrt(ca.sumsqr(RH[3 * i:3 * i + 3, 0]) + 1e-8)
            self._con(U_cs <= U_peak, "cap_catch")
            self.catch_loads[s + "_hold"] = U_cs
            self.effort_terms.append(U_imp ** 2 * p.delta_catch / p.t_ref)

        self.durs = dict(d_w=d_w, d_s=d_s, d_f=d_f)
        if p.release_first:
            self.durs["d_r"] = d_r
        if p.catch_first:
            self.durs["d_c"] = d_c
        self.effort = sum(self.effort_terms)
        self.smooth = sum(self.smooth_terms)
        self.times = dict(t0=t0, t_s0=t_s0, t_r1=t_r1, t_l=t_l, t_c=t_c, t_h0=t_h0)
        opti.minimize(p.w_E * self.effort + p.w_U * U_peak + p.w_smooth * self.smooth)

    # ------------------------------------------------------------------ initial guess ---------
    def set_initial_from_planar(self, sol, body_planar, twist_dir=+1.0):
        """Map a planar solution (hands = base, 5 links) onto the spatial model as the initial guess."""
        from ..planar.model import PlanarChain, NTH
        opti, p, T = self.opti, self.p, self.T
        chp = PlanarChain(body_planar)
        b = self.body

        def map_state(q_p, psi, s_twist=None):
            P = chp.points(q_p)                          # 2 x 6: hand, elbow, shoulder, hip, knee, ansi
            th = q_p[2:]
            S0 = np.array([P[0, 2], 0.0, P[1, 2]])
            if s_twist is None:
                theta = th[2] if np.cos(psi) < 0 else -th[2]
            else:                                        # continuous pitch guess while the yaw turns (flight)
                theta = th[2] * np.cos(np.pi * s_twist)
            Rtr = np.array(ca.DM(ca.vertcat(ca.horzcat(np.cos(psi), -np.sin(psi), 0), ca.horzcat(np.sin(psi), np.cos(psi), 0), ca.horzcat(0, 0, 1))))
            Rtr = Rtr @ np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            d_ua = np.array([-np.sin(th[1]), 0.0, np.cos(th[1])])       # shoulder -> elbow
            v = Rtr.T @ d_ua
            alpha = np.arctan2(v[0], v[2])
            Rsh = Rtr @ np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]])
            d_fa = np.array([-np.sin(th[0]), 0.0, np.cos(th[0])])       # elbow -> hand
            w2 = Rsh.T @ d_fa
            gamma = np.clip(np.arctan2(w2[0], w2[2]), 0.0, 2.6)
            d_th = np.array([np.sin(th[3]), 0.0, -np.cos(th[3])])
            v3 = Rtr.T @ d_th
            chi = np.arctan2(v3[0], -v3[2])
            d_sk = np.array([np.sin(th[4]), 0.0, -np.cos(th[4])])
            v4 = Rtr.T @ d_sk
            kappa = np.clip(chi - np.arctan2(v4[0], -v4[2]), 0.0, 2.6)
            q = np.zeros(NQ)
            q[0:3] = S0; q[3] = psi; q[4] = 0.0; q[5] = theta
            q[6:9] = [alpha, 0.0, gamma]; q[9:12] = [alpha, 0.0, gamma]; q[12] = chi; q[13] = kappa
            return q

        def phase_guess(Xp, dur, psi_fun, hand_of_t, twist=False):
            """Xp: planar state columns (q_p or (th, thd) with hand from device); returns X (28 x N+1)."""
            N = Xp.shape[1] - 1
            Q = np.zeros((NQ, N + 1))
            for k in range(N + 1):
                Q[:, k] = map_state(hand_of_t(k), psi_fun(k / N), (k / N) if twist else None)
            Qd = np.gradient(Q, dur / N, axis=1) if N > 0 else np.zeros_like(Q)
            Qd[6:] = np.clip(Qd[6:], -0.8 * b.qd_max, 0.8 * b.qd_max)
            Qd[3:6] = np.clip(Qd[3:6], -0.8 * p.trunk_rate_max, 0.8 * p.trunk_rate_max)
            return np.vstack([Q, Qd])

        eps = sol["params"]["eps"]
        # swing
        XSp = sol["S_X"]; d_s = sol["d_s"]; NSp = XSp.shape[1] - 1
        def handS(k):
            t = sol["t_s0"] + d_s * k / NSp
            dv = device.device_state(t, T, eps)
            return np.concatenate([dv["pA"], XSp[:NTH, k]])
        XS = phase_guess(XSp, d_s, lambda s: np.pi, handS)
        XS = self._resample(XS, p.N_S + 1)
        XS[NQ:, 0] = 0.0                                             # starts at rest
        opti.set_initial(self.vars["S_X"], XS)
        opti.set_initial(self.vars["S_U"], self._resample(np.vstack([sol["S_U"][1], sol["S_U"][1] * 0, sol["S_U"][0],
                                                                       sol["S_U"][1], sol["S_U"][1] * 0, sol["S_U"][0],
                                                                       sol["S_U"][2], sol["S_U"][3]]), p.N_S + 1))
        RS = np.array(sol["S_R"]) * body_planar.f_cap / (2 * b.f_hand)   # per hand, (x, y) -> (x, 0, z)
        R6 = np.zeros((6, RS.shape[1])); R6[0] = RS[0]; R6[2] = RS[1]; R6[3] = RS[0]; R6[5] = RS[1]
        opti.set_initial(self.vars["S_R"], self._resample(R6, p.N_S + 1))
        opti.set_initial(self.vars["S_Rm"], self._resample(R6, p.N_S))
        # flight: twist from pi to pi + twist_dir * pi
        XFp = sol["F_X"]; d_f = sol["d_f"]; NFp = XFp.shape[1] - 1
        psi_end = np.pi + twist_dir * np.pi
        smooth = lambda s: 3 * s ** 2 - 2 * s ** 3                    # zero yaw rate at release and catch
        XF = phase_guess(XFp, d_f, lambda s: np.pi + twist_dir * np.pi * smooth(s), lambda k: XFp[:, k][:7], twist=True)
        XF = self._resample(XF, p.N_F + 1)
        opti.set_initial(self.vars["F_X"], XF)
        opti.set_initial(self.vars["F_U"], 0.0)
        # hold
        XHp = sol["H_X"]; Th = sol["params"]["T_hold"]; NHp = XHp.shape[1] - 1
        def handH(k):
            t = sol["t_h0"] + Th * k / NHp
            dv = device.device_state(t, T, eps)
            return np.concatenate([dv["pB"], XHp[:NTH, k]])
        XH = phase_guess(XHp, Th, lambda s: psi_end, handH)
        XH = self._resample(XH, p.N_H + 1)
        opti.set_initial(self.vars["H_X"], XH)
        opti.set_initial(self.vars["H_U"], 0.0)
        RH = np.array(sol["H_R"]) * body_planar.f_cap / (2 * b.f_hand)
        R6 = np.zeros((6, RH.shape[1])); R6[0] = RH[0]; R6[2] = RH[1]; R6[3] = RH[0]; R6[5] = RH[1]
        opti.set_initial(self.vars["H_R"], self._resample(R6, p.N_H + 1))
        opti.set_initial(self.vars["H_Rm"], self._resample(R6, p.N_H))
        # one-hand phases: tile
        if "S1_X" in self.vars:
            opti.set_initial(self.vars["S1_X"], np.tile(XS[:, -1:], (1, p.N_1 + 1)))
            opti.set_initial(self.vars["S1_R"], np.tile(np.array(self._resample(R6, p.N_S + 1))[0:3, -1:] * 2, (1, p.N_1 + 1)))
            opti.set_initial(self.vars["S1_Rm"], 0.0)
        if "C1_X" in self.vars:
            opti.set_initial(self.vars["C1_X"], np.tile(XH[:, :1], (1, p.N_1 + 1)))
            opti.set_initial(self.vars["C1_R"], np.tile(R6[0:3, :1] * 2, (1, p.N_1 + 1)))
            opti.set_initial(self.vars["C1_Rm"], 0.0)
        for name, v in self.vars.items():
            if name.endswith("_A") or name.endswith("_Am"):
                opti.set_initial(v, 0.0)
            if name.startswith("qd_plus"):
                opti.set_initial(v, XH[NQ:, 0])
            if name.startswith("Lam_"):
                opti.set_initial(v, np.array([0.0, 0.0, 0.1]))
        opti.set_initial(self.vars["F_A"][2, :], -G); opti.set_initial(self.vars["F_Am"][2, :], -G)
        opti.set_initial(self.durs["d_w"], sol["d_w"]); opti.set_initial(self.durs["d_s"], d_s); opti.set_initial(self.durs["d_f"], d_f)
        if "d_r" in self.durs:
            opti.set_initial(self.durs["d_r"], 0.02)
        if "d_c" in self.durs:
            opti.set_initial(self.durs["d_c"], 0.02)
        opti.set_initial(self.U_peak, sol["U_peak"] * 1.2)
        opti.set_initial(self.vars["dB"], b.shoulder_width); opti.set_initial(self.vars["yB"], 0.0)

    def set_initial_from_prev(self, prev):
        opti = self.opti
        for name, v in self.vars.items():
            if not v.is_symbolic():
                continue
            if name in prev:
                arr = np.array(prev[name], float)
                shape = tuple(v.shape)
                if arr.shape == shape:
                    opti.set_initial(v, arr)
                elif arr.ndim == 2 and arr.shape[0] == shape[0]:
                    opti.set_initial(v, self._resample(arr, shape[1]))
        for name, v in self.durs.items():
            if name in prev:
                opti.set_initial(v, prev[name])
        opti.set_initial(self.U_peak, prev["U_peak"])

    @staticmethod
    def _resample(arr, ncol):
        arr = np.atleast_2d(np.array(arr, float))
        if arr.shape[1] == ncol:
            return arr
        s_old = np.linspace(0, 1, arr.shape[1]); s_new = np.linspace(0, 1, ncol)
        return np.vstack([np.interp(s_new, s_old, row) for row in arr])

    # ------------------------------------------------------------------ solve -------------------
    def solve(self, max_iter=3000, print_level=0, tol=1e-4, max_cpu_time=3600.0, sensitivities=False, expand=False,
              hessian="limited-memory"):
        opts = {"expand": expand, "ipopt.max_iter": max_iter, "ipopt.hessian_approximation": hessian,
                "ipopt.limited_memory_max_history": 30, "ipopt.print_level": print_level, "print_time": 0,
                "ipopt.tol": tol, "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 10,
                "ipopt.acceptable_constr_viol_tol": 1e-6, "ipopt.mu_strategy": "adaptive",
                "ipopt.linear_solver": "mumps", "ipopt.max_cpu_time": max_cpu_time, "ipopt.sb": "yes"}
        for k, v in self.par.items():
            self.opti.set_value(v, self.par_values[k])
        self.opti.solver("ipopt", opts)
        try:
            sol = self.opti.solve()
            ok = True
            status = sol.stats()["return_status"]
        except RuntimeError as e:
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
            except Exception as e:
                out["duals"] = dict(error=str(e)[:200])
        return out

    def dual_report(self, sol):
        opti = self.opti
        lam = np.array(sol.value(opti.lam_g)).ravel()
        rep = dict(categories={}, n_lam=int(lam.size))
        i = 0
        for cat, n in self.cons:
            seg = lam[i:i + n]; i += n
            c = rep["categories"].setdefault(cat, dict(n=0, abs_sum=0.0, max_abs=0.0, n_active=0))
            c["n"] += n; c["abs_sum"] += float(np.abs(seg).sum())
            c["max_abs"] = max(c["max_abs"], float(np.abs(seg).max()) if n else 0.0)
            c["n_active"] += int((np.abs(seg) > 1e-6).sum())
        rep["rows_consistent"] = bool(i == lam.size)
        rep["cap_mult"] = {}
        for cat, expr in self.cap_cons:
            try:
                d = float(np.abs(np.array(sol.value(opti.dual(expr))).ravel()).sum())
            except Exception:
                d = float("nan")
            rep["cap_mult"].setdefault(cat, []).append(d)
        L = opti.f + ca.dot(opti.lam_g, opti.g)
        sens = {}
        for name, pv in self.par.items():
            g = sol.value(ca.jacobian(L, pv))
            gradp = (g.toarray() if hasattr(g, "toarray") else np.array(g)).ravel()
            sens[name] = (gradp * np.atleast_1d(np.array(self.par_values[name], float))).tolist()
        rep["dJ_dlnp"] = sens
        rep["w_U"] = self.p.w_U
        return rep

    def extract(self, sol):
        v = {k: np.array(sol.value(x)) for k, x in self.vars.items()}
        for k, x in self.durs.items():
            v[k] = float(sol.value(x))
        v.setdefault("d_r", 0.0); v.setdefault("d_c", 0.0)
        v["U_peak"] = float(sol.value(self.U_peak))
        v["effort"] = float(sol.value(self.effort))
        v["catch_loads"] = {k: float(sol.value(e)) for k, e in self.catch_loads.items()}
        v["T"], v["phi0"], v["m"] = self.T, self.phi0, self.body.m
        v["params"] = asdict(self.p)
        t0 = self.phi0 * self.T
        v["t0"] = t0; v["t_s0"] = t0 + v["d_w"]; v["t_r1"] = v["t_s0"] + v["d_s"]; v["t_l"] = v["t_r1"] + v["d_r"]
        v["t_c"] = v["t_l"] + v["d_f"]; v["t_h0"] = v["t_c"] + v["d_c"]
        v["phi_l"] = (v["t_l"] % self.T) / self.T
        v["phi_c"] = (v["t_c"] % self.T) / self.T
        v["model"] = "spatial14"
        # per-hand peak utilisation by phase
        for ph, nr in ((k, m["nr"]) for k, m in self.phase_meta.items()):
            if nr:
                R = np.array(v[ph + "_R"]).reshape(nr, -1)
                v["U_" + ph] = [float(np.sqrt((R[3 * i:3 * i + 3] ** 2).sum(0)).max()) for i in range(nr // 3)]
        return v
