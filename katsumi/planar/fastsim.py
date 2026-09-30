"""Compiled sub-step integrators for the planar Cliff-Dimension environment.

The environment physics (pinned swing on A, free flight, compliant hold on B; soft joint stops; wall
penetration and cone diagnostics) is written once symbolically in CasADi, unrolled over `n_sub` RK4
sub-steps with `mapaccum`, code-generated to C and compiled into a shared library that is cached on disk.
One control step of the environment is then a single call into compiled code (thousands of environment
steps per second on one core instead of ~50 with per-substep Python calls).

Functions (all take the device period T as an input, so one library serves every T):
    A(x, t, T, tau)      : x = (th, thd) on A       -> X (10 x n), R (2 x n), jr (n), wall (n), Gc (2 x n), Gv (2 x n)
    F(x, t, T, tau)      : x = (q, qd) in flight    -> X (14 x n), jr (n), wall (n)
    C(x, t, T, tau, hook_tol) : x = (q, qd, off) hooked on B -> X (15 x n), Fa (2 x n), Fraw (2 x n), sep (n), jr (n),
                           wall (n), slip (n)   [off = hook attachment rel. to B's tip; slides when the cone is exceeded]
with n = n_sub for the batched versions and n = 1 for the single-substep versions (used across mode changes).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import casadi as ca
import numpy as np

from .. import device
from .anthro import G
from .model import NQ, NTH, NTAU

# joint ranges of the optimiser for the "facing -x" convention (elbow, shoulder, hip, knee)
REL_LO_NEG = np.array([0.0, -4.2, -2.1, 0.0])
REL_HI_NEG = np.array([2.6, 0.60, 0.35, 2.6])


def rel_bounds(facing):
    if facing == "neg":
        return REL_LO_NEG, REL_HI_NEG
    if facing == "pos":
        return -REL_HI_NEG, -REL_LO_NEG
    return np.minimum(REL_LO_NEG, -REL_HI_NEG), np.maximum(REL_HI_NEG, -REL_LO_NEG)   # union (flight)


def _rel(th):
    return ca.vertcat(th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3])


class FastSim:
    def __init__(self, body, chain, sub_dt=0.002, n_sub=10, eps=0.20, K_att=40000.0, D_att=1500.0,
                 joint_stop_k=15.0, ramp_att=0.01, stop_damp=0.01, cache_dir=None, verbose=False):
        self.body, self.chain = body, chain
        self.sub_dt, self.n_sub, self.eps = float(sub_dt), int(n_sub), float(eps)
        self.K_att, self.D_att, self.joint_stop_k, self.ramp_att = float(K_att), float(D_att), float(joint_stop_k), float(ramp_att)
        self.stop_damp = float(stop_damp)
        self.cache_dir = cache_dir or os.environ.get("KATSUMI_FASTSIM_CACHE",
                                                     os.path.join(os.path.expanduser("~"), ".cache", "katsumi_fastsim"))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.key = self._key()
        self.lib = os.path.join(self.cache_dir, f"fastsim_{self.key}.so")
        if not os.path.exists(self.lib):
            self._build_and_compile(verbose)
        n = self.n_sub
        self.fA1 = ca.external("A1", self.lib); self.fF1 = ca.external("F1", self.lib); self.fC1 = ca.external("C1", self.lib)
        self.fA = ca.external(f"A1_acc{n}", self.lib) if n > 1 else self.fA1
        self.fF = ca.external(f"F1_acc{n}", self.lib) if n > 1 else self.fF1
        self.fC = ca.external(f"C1_acc{n}", self.lib) if n > 1 else self.fC1
        self.f_aux = ca.external("aux", self.lib)

    # ------------------------------------------------------------------------------------------------
    def _key(self):
        b = self.body
        parts = [b.m, b.lengths, b.masses, b.coms, b.inertias, b.tips, b.clearance, b.forearm_points, b.forearm_clearance,
                 b.tau_cap, b.f_cap, b.mu_out, b.mu_in, self.sub_dt, self.n_sub, self.eps, self.K_att, self.D_att,
                 self.joint_stop_k, self.ramp_att, self.stop_damp, REL_LO_NEG, REL_HI_NEG, device.D_LEDGE, device.WALL_BOTTOM_OFFSET, ca.__version__, "v5"]
        s = "|".join(np.array2string(np.atleast_1d(np.asarray(p, float)), precision=10) if not isinstance(p, str) else p
                     for p in parts)
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def _limit_torque(self, th, thd, lo, hi):
        k, tc = self.joint_stop_k, ca.DM(self.body.tau_cap)
        rel, reld = _rel(th), _rel(thd)
        over, under = rel - ca.DM(hi), ca.DM(lo) - rel
        # unilateral stops: a stop only pushes back into the range (never pulls), damping included
        push_hi = ca.fmax(ca.if_else(over > 0, over + self.stop_damp * reld, 0.0), 0.0)
        push_lo = ca.fmax(ca.if_else(under > 0, under - self.stop_damp * reld, 0.0), 0.0)
        tq = -k * tc * push_hi + k * tc * push_lo
        tq = ca.fmin(ca.fmax(tq, -2 * tc), 2 * tc)
        jr = ca.sum1(ca.fmax(over, 0)) + ca.sum1(ca.fmax(under, 0))
        return tq, jr

    def _wall_pen(self, q, dv):
        ch, b = self.chain, self.body
        P = ch.f_tips(q); Pf = ch.f_forearm(q)
        pts = ca.horzcat(q[0:2], P, Pf)
        clear = np.concatenate([[0.0], b.clearance, b.forearm_clearance])
        h, xB = dv["h"], dv["x"]
        pen = 0.0
        for i in range(pts.shape[1]):
            x, y = pts[0, i], pts[1, i]
            aA = x - (-device.D_LEDGE + clear[i]); bA = (h - device.WALL_BOTTOM_OFFSET) - y
            aB = (xB + device.D_LEDGE - clear[i]) - x; bB = -device.WALL_BOTTOM_OFFSET - y
            pen = ca.fmax(pen, ca.fmax(-ca.fmax(aA, bA), -ca.fmax(aB, bB)))
        return pen

    def _hand_force_B(self, q, qd, dv, off):
        b = self.body
        dp = q[0:2] - (dv["pB"] + ca.vertcat(off, 0.0)); dvv = qd[0:2] - dv["vB"]
        ramp = ca.tanh(ca.sqrt(ca.sumsqr(dp) + 1e-12) / self.ramp_att)      # Hunt-Crossley-like ramp, as in the NLP
        F = -self.K_att * dp - self.D_att * ramp * dvv
        Fy = ca.fmax(F[1], 0.0)
        Fx = ca.fmin(ca.fmax(F[0], -b.mu_in * Fy), b.mu_out * Fy)
        return ca.vertcat(Fx, Fy), F, ca.sqrt(ca.sumsqr(dp) + 1e-12)

    @staticmethod
    def _rk4(f, x, t, dt):
        k1 = f(t, x); k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1); k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2); k4 = f(t + dt, x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build_and_compile(self, verbose):
        ch, b, dt = self.chain, self.body, self.sub_dt
        tc = ca.DM(b.tau_cap)
        t = ca.SX.sym("t"); T = ca.SX.sym("T"); tau = ca.SX.sym("tau", NTAU)
        dev = lambda tt: device.device_ca(tt, T, self.eps)
        loN, hiN = rel_bounds("neg"); loU, hiU = rel_bounds("union"); loP, hiP = rel_bounds("pos")
        # ---- A: pinned on A ------------------------------------------------------------------------------
        xA = ca.SX.sym("xA", 2 * NTH)

        def fA_rhs(tt, x):
            th, thd = x[:NTH], x[NTH:]
            dv = dev(tt)
            q = ca.vertcat(dv["pA"], th); qd = ca.vertcat(dv["vA"], thd)
            tq, _ = self._limit_torque(th, thd, loN, hiN)
            thdd, _ = ch.f_pinned(q, qd, tau * tc + tq, dv["aA"])
            return ca.vertcat(thd, thdd)

        xn = self._rk4(fA_rhs, xA, t, dt)
        tn = t + dt
        dv = dev(tn)
        q = ca.vertcat(dv["pA"], xn[:NTH]); qd = ca.vertcat(dv["vA"], xn[NTH:])
        tq, jr = self._limit_torque(xn[:NTH], xn[NTH:], loN, hiN)
        _, R = ch.f_pinned(q, qd, tau * tc + tq, dv["aA"])
        A1 = ca.Function("A1", [xA, t, T, tau], [xn, R, jr, self._wall_pen(q, dv), ch.f_com(q), ch.f_comd(q, qd)],
                         ["x", "t", "T", "tau"], ["X", "R", "jr", "wall", "Gc", "Gv"])
        # ---- F: free flight ------------------------------------------------------------------------------
        xF = ca.SX.sym("xF", 2 * NQ)

        def fF_rhs(tt, x):
            q, qd = x[:NQ], x[NQ:]
            tq, _ = self._limit_torque(q[2:], qd[2:], loU, hiU)
            return ca.vertcat(qd, ch.f_qdd_free(q, qd, tau * tc + tq, ca.DM.zeros(2)))

        xn = self._rk4(fF_rhs, xF, t, dt)
        dv = dev(t + dt)
        _, jr = self._limit_torque(xn[2:NQ], xn[NQ + 2:], loU, hiU)
        F1 = ca.Function("F1", [xF, t, T, tau], [xn, jr, self._wall_pen(xn[:NQ], dv)], ["x", "t", "T", "tau"], ["X", "jr", "wall"])
        # ---- C: hooked on B (spring-damper hand force, Coulomb-limited with sliding of the hook point) ----
        # state xC = (q, qd, off): off = attachment point of the hook line relative to B's tip (x). When the demanded
        # tangential force leaves the admissible cone the fingers slide along the ledge: the attachment is
        # return-mapped so that the spring force sits exactly on the cone (elastic-plastic friction model).
        xC = ca.SX.sym("xC", 2 * NQ + 1)
        hook_tol = ca.SX.sym("hook_tol")

        def fC_rhs(tt, x):
            q, qd, o = x[:NQ], x[NQ:2 * NQ], x[2 * NQ]
            Fa, _, _ = self._hand_force_B(q, qd, dev(tt), o)
            tq, _ = self._limit_torque(q[2:], qd[2:], loP, hiP)
            return ca.vertcat(qd, ch.f_qdd_free(q, qd, tau * tc + tq, Fa), 0.0)

        xn = self._rk4(fC_rhs, xC, t, dt)
        dv = dev(t + dt)
        o0 = xn[2 * NQ]
        Fa, Fraw, sep = self._hand_force_B(xn[:NQ], xn[NQ:2 * NQ], dv, o0)
        dp = xn[0:2] - (dv["pB"] + ca.vertcat(o0, 0.0)); dvv = xn[NQ:NQ + 2] - dv["vB"]
        ramp = ca.tanh(ca.sqrt(ca.sumsqr(dp) + 1e-12) / self.ramp_att)
        # return mapping of the attachment: spring x-force equal to the admissible (clipped) one
        o_new = xn[0] - dv["pB"][0] + (Fa[0] + self.D_att * ramp * dvv[0]) / self.K_att
        o_new = ca.if_else(ca.fabs(Fraw[0] - Fa[0]) > 1e-9, o_new, o0)
        o_new = ca.if_else(Fa[1] > 1e-9, o_new, xn[0] - dv["pB"][0])       # no normal load: the fingers hover over the ledge
        o_new = ca.fmin(ca.fmax(o_new, -hook_tol), device.D_LEDGE)
        slip = ca.fabs(o_new - o0)
        xn = ca.vertcat(xn[:2 * NQ], o_new)
        _, jr = self._limit_torque(xn[2:NQ], xn[NQ + 2:2 * NQ], loP, hiP)
        C1 = ca.Function("C1", [xC, t, T, tau, hook_tol], [xn, Fa, Fraw, sep, jr, self._wall_pen(xn[:NQ], dv), slip],
                         ["x", "t", "T", "tau", "hook_tol"], ["X", "Fa", "Fraw", "sep", "jr", "wall", "slip"])
        # ---- auxiliary: CoM / CoM velocity of a full state (for shaping / diagnostics) ---------------------
        aux = ca.Function("aux", [xF], [ch.f_com(xF[:NQ]), ch.f_comd(xF[:NQ], xF[NQ:]), ch.f_rel(xF[:NQ])], ["x"], ["Gc", "Gv", "rel"])
        n = self.n_sub
        funcs = [A1, F1, C1, aux]
        if n > 1:
            funcs += [A1.mapaccum(n), F1.mapaccum(n), C1.mapaccum(n)]      # exported as <name>_acc<n>
        cg = ca.CodeGenerator(f"fastsim_{self.key}", dict(with_header=False, casadi_real="double", casadi_int="long long int"))
        for f in funcs:
            cg.add(f)
        src = cg.generate(self.cache_dir + os.sep)
        cc = os.environ.get("CC", "gcc")
        cmd = [cc, "-O1", "-fPIC", "-shared", "-o", self.lib, src, "-lm"]
        if verbose:
            print("compiling", " ".join(cmd), file=sys.stderr, flush=True)
        subprocess.run(cmd, check=True)
        try:
            os.remove(src)
        except OSError:
            pass
