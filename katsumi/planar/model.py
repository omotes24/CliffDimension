"""Planar 5-link chain with the hands (grip point) as the floating base.

Generalised coordinates  q = (x_h, y_h, th1, th2, th3, th4, th5)
    (x_h, y_h) : grip point (both hands lumped) in the world frame
    th_i       : absolute angle of link i from the downward vertical, positive towards +x
Link direction  d(th) = (sin th, -cos th)  (th = 0: link hangs straight down)

Relative joint angles (used for ranges, torques and effort)
    e   = th1 - th2   elbow          (facing -x: flexion > 0)
    psi = th3 - th2   shoulder       (0 = arms overhead)
    chi = th4 - th3   hip
    kap = th5 - th4   knee
Joint torques tau = (tau_e, tau_psi, tau_chi, tau_kap) act on the relative angles (virtual work),
so the generalised force is B tau with B = (d rel / d q)^T.

Equations of motion  M(q) qdd + h(q, qd) = B tau + J^T R,   J = d p_h / d q = [I_2  0]
    R : force exerted by the ledge on the hands (world frame). In the pinned phases the hand
        coordinates are prescribed by the device and R is the constraint force (rows 1-2).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .anthro import Body, G

NQ = 7
NTH = 5
NTAU = 4


def dvec(th):
    return ca.vertcat(ca.sin(th), -ca.cos(th))


class PlanarChain:
    def __init__(self, body: Body):
        self.body = body
        L = body.lengths
        Mi = body.masses
        Ci = body.coms
        Ii = body.inertias
        tips = body.tips

        q = ca.SX.sym("q", NQ)
        qd = ca.SX.sym("qd", NQ)
        ph = q[0:2]
        th = q[2:7]
        thd = qd[2:7]

        # joint positions P0 = hand, P1 = elbow, P2 = shoulder, P3 = hip, P4 = knee, P5 = ankle
        P = [ph]
        for i in range(NTH):
            P.append(P[-1] + L[i] * dvec(th[i]))
        # clearance points (link tips; for the shank the hanging toe) + points along the forearm: the forearm
        # hangs against the cliff face while hooked, so it must never pass behind the face plane (the elbow and
        # everything below it are under the board and free)
        Ptip = [P[i] + tips[i] * dvec(th[i]) for i in range(NTH)]
        Pfa = [P[0] + d * dvec(th[0]) for d in body.forearm_points]
        # link CoMs
        C = [P[i] + Ci[i] * dvec(th[i]) for i in range(NTH)]
        Cd = [ca.jtimes(C[i], q, qd) for i in range(NTH)]
        Ttot = 0
        V = 0
        for i in range(NTH):
            Ttot += 0.5 * Mi[i] * ca.dot(Cd[i], Cd[i]) + 0.5 * Ii[i] * thd[i] ** 2
            V += Mi[i] * G * C[i][1]
        mtot = float(np.sum(Mi))
        Gc = sum(Mi[i] * C[i] for i in range(NTH)) / mtot
        Gcd = ca.jtimes(Gc, q, qd)

        Mmat = ca.hessian(Ttot, qd)[0]
        Mmat = 0.5 * (Mmat + Mmat.T)
        hvec = ca.jtimes(Mmat @ qd, q, qd) - ca.gradient(Ttot, q) + ca.gradient(V, q)

        rel = ca.vertcat(th[0] - th[1], th[2] - th[1], th[3] - th[2], th[4] - th[3])
        Bmat = ca.jacobian(rel, q).T  # 7 x 4

        # angular momentum about the CoM (out-of-plane component)
        Lg = 0
        for i in range(NTH):
            r = C[i] - Gc
            v = Cd[i] - Gcd
            Lg += Ii[i] * thd[i] + Mi[i] * (r[0] * v[1] - r[1] * v[0])

        self.q, self.qd = q, qd
        self.f_M = ca.Function("M", [q], [Mmat])
        self.f_h = ca.Function("h", [q, qd], [hvec])
        self.f_B = ca.Function("B", [], [Bmat])
        self.B = np.array(ca.DM(Bmat))
        self.f_points = ca.Function("points", [q], [ca.horzcat(*P)])          # 2 x 6
        self.f_tips = ca.Function("tips", [q], [ca.horzcat(*Ptip)])           # 2 x 5
        self.f_forearm = ca.Function("forearm", [q], [ca.horzcat(*Pfa)])      # 2 x n_forearm_points
        self.f_com = ca.Function("com", [q], [Gc])
        self.f_comd = ca.Function("comd", [q, qd], [Gcd])
        self.f_rel = ca.Function("rel", [q], [rel])
        self.f_energy = ca.Function("energy", [q, qd], [Ttot, V])
        self.f_Lg = ca.Function("Lg", [q, qd], [Lg])
        self.mtot = mtot

        # --- symbolic dynamics helpers ---------------------------------------------------
        tau = ca.SX.sym("tau", NTAU)
        R = ca.SX.sym("R", 2)
        # free flight / catch (external hand force R)
        rhs = Bmat @ tau - hvec + ca.vertcat(R, ca.SX.zeros(NTH))
        qdd_free = ca.solve(Mmat, rhs)
        self.f_qdd_free = ca.Function("qdd_free", [q, qd, tau, R], [qdd_free])
        # pinned: hand acceleration prescribed
        ah = ca.SX.sym("ah", 2)
        Mtt = Mmat[2:, 2:]
        Mth = Mmat[2:, 0:2]
        Mht = Mmat[0:2, 2:]
        Mhh = Mmat[0:2, 0:2]
        thdd = ca.solve(Mtt, (Bmat @ tau - hvec)[2:] - Mth @ ah)
        Rpin = Mhh @ ah + Mht @ thdd + hvec[0:2]
        self.f_pinned = ca.Function("pinned", [q, qd, tau, ah], [thdd, Rpin])

    # convenience numpy wrappers -----------------------------------------------------------
    def points(self, q):
        return np.array(self.f_points(q))

    def com(self, q):
        return np.array(self.f_com(q)).ravel()

    def qdd_free(self, q, qd, tau, R=(0.0, 0.0)):
        return np.array(self.f_qdd_free(q, qd, tau, R)).ravel()

    def pinned(self, q, qd, tau, ah):
        thdd, R = self.f_pinned(q, qd, tau, ah)
        return np.array(thdd).ravel(), np.array(R).ravel()

    def joint_state_from_theta(self, ph, vh, th, thd):
        q = np.concatenate([np.asarray(ph, float), np.asarray(th, float)])
        qd = np.concatenate([np.asarray(vh, float), np.asarray(thd, float)])
        return q, qd
