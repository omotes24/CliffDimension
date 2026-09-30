"""Spatial (3-D) reduced model with a twist axis and two independent hands.

Seven rigid bodies (trunk+head, upper arm L/R, forearm+hand L/R, thigh pair, shank+foot pair), 14 DoF

    q = ( p (3)            world position of the shoulder-line centre S0
          psi, phi, theta   trunk orientation, R_tr = Rz(psi) Rx(phi) Ry(theta)   (yaw = twist, roll, pitch)
          aL, bL, gL         left  shoulder flexion, abduction, elbow flexion
          aR, bR, gR         right shoulder flexion, abduction, elbow flexion
          chi, kappa )       hip flexion, knee flexion (legs move as a pair)

World frame: +x from A towards B, +y lateral (along the ledges), +z up.  Trunk frame: e1 forward (chest),
e2 left, e3 up. Reference posture (all angles 0): trunk upright facing +x, arms straight overhead, legs down.
Facing cliff A means psi = pi; facing cliff B means psi = 0 (mod 2 pi).

Equations of motion  M(q) qdd + h(q, qd) = B tau + J_L^T R_L + J_R^T R_R
    tau = 8 joint torques acting directly on the joint coordinates (B = [0; I_8]),
    R_L, R_R = world-frame forces of the ledge on the left / right hand (hook line = point contact).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from ..planar.anthro import G
from .anthro3d import Body3D

NQ = 14
NTAU = 8
IDX = dict(p=slice(0, 3), eul=slice(3, 6), armL=slice(6, 9), armR=slice(9, 12), hip=12, knee=13)
JOINT_NAMES = ["sh_flex_L", "sh_abd_L", "elbow_L", "sh_flex_R", "sh_abd_R", "elbow_R", "hip", "knee"]


def Rx(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, c, -s), ca.horzcat(0, s, c))


def Ry(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(ca.horzcat(c, 0, s), ca.horzcat(0, 1, 0), ca.horzcat(-s, 0, c))


def Rz(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))


def vee(S):
    return ca.vertcat(S[2, 1], S[0, 2], S[1, 0])


def skew(v):
    return ca.vertcat(ca.horzcat(0, -v[2], v[1]), ca.horzcat(v[2], 0, -v[0]), ca.horzcat(-v[1], v[0], 0))


class SpatialChain:
    def __init__(self, body: Body3D):
        self.body = body
        sg = body.segs
        w = body.shoulder_width
        q = ca.SX.sym("q", NQ)
        qd = ca.SX.sym("qd", NQ)
        p = q[0:3]
        psi, phi, th = q[3], q[4], q[5]
        e1, e2, e3 = ca.DM([1, 0, 0]), ca.DM([0, 1, 0]), ca.DM([0, 0, 1])
        R_tr = Rz(psi) @ Rx(phi) @ Ry(th)

        bodies = []   # (mass, com_world, R_seg (3rd axis = long axis), I_local)
        pts = {}
        S0 = p
        pts["S0"] = S0
        pts["head"] = S0 + R_tr @ (body.head_top * e3)
        # trunk + head
        c_tr = S0 + R_tr @ sg["trunk"].com
        bodies.append((sg["trunk"].mass, c_tr, R_tr, sg["trunk"].inertia))
        # arms
        hands, elbows, shoulders, wrists, dfa = {}, {}, {}, {}, {}
        for side, sgn, sl in (("L", 1.0, IDX["armL"]), ("R", -1.0, IDX["armR"])):
            a, b, g = q[sl.start], q[sl.start + 1], q[sl.start + 2]
            sh = S0 + R_tr @ (sgn * 0.5 * w * e2)
            R_sh = R_tr @ Ry(a) @ Rx(-sgn * b)         # positive b = abduction (away from the midline)
            d_ua = R_sh @ e3
            L_ua = sg["ua_" + side].length
            el = sh + L_ua * d_ua
            R_fa = R_sh @ Ry(g)
            d_fa_ = R_fa @ e3
            L1 = sg["fa_" + side].length
            hd = el + L1 * d_fa_
            c_ua = sh + R_sh @ sg["ua_" + side].com
            c_fa = el + R_fa @ sg["fa_" + side].com
            bodies.append((sg["ua_" + side].mass, c_ua, R_sh, sg["ua_" + side].inertia))
            bodies.append((sg["fa_" + side].mass, c_fa, R_fa, sg["fa_" + side].inertia))
            shoulders[side], elbows[side], hands[side], dfa[side] = sh, el, hd, d_fa_
            wrists[side] = hd - 0.10 * d_fa_
            pts["shoulder_" + side] = sh
            pts["elbow_" + side] = el
            pts["wrist_" + side] = wrists[side]
            pts["hand_" + side] = hd
        # legs (pair)
        chi, kap = q[12], q[13]
        hip = S0 + R_tr @ (-sg["trunk"].length * e3)
        R_th = R_tr @ Ry(-chi)                         # 3rd axis = -(thigh direction)
        d_th = -(R_th @ e3)
        knee = hip + sg["thigh"].length * d_th
        R_sk = R_tr @ Ry(-chi + kap)
        d_sk = -(R_sk @ e3)
        ankle = knee + sg["shank"].length * d_sk
        toe = ankle + 0.10 * (body.stature / 1.75) * d_sk
        c_th = hip + sg["thigh"].com[2] * d_th
        c_sk = knee + sg["shank"].com[2] * d_sk
        bodies.append((sg["thigh"].mass, c_th, R_th, sg["thigh"].inertia))
        bodies.append((sg["shank"].mass, c_sk, R_sk, sg["shank"].inertia))
        pts.update(hip=hip, knee=knee, ankle=ankle, toe=toe)

        # ---- energies / mass matrix from body Jacobians (lean SX graph: first derivatives only) -------------
        V = 0
        mtot = 0
        Gc = 0
        Mmat = ca.SX.zeros(NQ, NQ)
        Jw_list = []
        for (mi, ci, Ri, Ii) in bodies:
            Jv = ca.jacobian(ci, q)                                    # 3 x NQ
            Rd = ca.reshape(ca.jtimes(ca.vec(Ri), q, qd), 3, 3)
            om = vee(Ri.T @ Rd)                                        # linear in qd
            Jw = ca.jacobian(om, qd)                                   # 3 x NQ (function of q only)
            Jw_list.append(Jw)
            Mmat = Mmat + mi * (Jv.T @ Jv) + Jw.T @ (ca.DM(Ii) @ Jw)
            V += mi * G * ci[2]
            mtot += mi
            Gc = Gc + mi * ci
        Mmat = 0.5 * (Mmat + Mmat.T)
        Gc = Gc / mtot
        Gcd = ca.jtimes(Gc, q, qd)
        Ttot = 0.5 * ca.dot(qd, Mmat @ qd)
        # h = Mdot qd - dT/dq + dV/dq  (Mdot qd via a directional derivative of M qd along qd)
        hvec = ca.jtimes(Mmat @ qd, q, qd) - ca.gradient(Ttot, q) + ca.gradient(V, q)
        # angular momentum about the CoM (world frame)
        Lg = ca.SX.zeros(3)
        for (mi, ci, Ri, Ii), Jw in zip(bodies, Jw_list):
            cd = ca.jtimes(ci, q, qd)
            om = Jw @ qd
            Lg += mi * ca.cross(ci - Gc, cd - Gcd) + Ri @ (ca.DM(Ii) @ om)
        # hand Jacobians
        JL = ca.jacobian(hands["L"], q)
        JR = ca.jacobian(hands["R"], q)
        # torque map: joints are the last 8 coordinates
        Bmat = ca.vertcat(ca.DM.zeros(6, NTAU), ca.DM.eye(NTAU))

        self.q, self.qd = q, qd
        self.mtot = float(mtot)
        self.B = np.array(ca.DM(Bmat))
        self.f_M = ca.Function("M3", [q], [Mmat])
        self.f_h = ca.Function("h3", [q, qd], [hvec])
        self.f_com = ca.Function("com3", [q], [Gc])
        self.f_comd = ca.Function("comd3", [q, qd], [Gcd])
        self.f_Lg = ca.Function("Lg3", [q, qd], [Lg])
        self.f_energy = ca.Function("energy3", [q, qd], [Ttot, V])
        self.f_hands = ca.Function("hands3", [q], [hands["L"], hands["R"]])
        self.f_handsd = ca.Function("handsd3", [q, qd], [JL @ qd, JR @ qd])
        self.f_J = ca.Function("J3", [q], [JL, JR])
        self.point_names = list(pts.keys())
        self.f_points = ca.Function("points3", [q], [ca.horzcat(*[pts[k] for k in self.point_names])])
        self.f_dirs = ca.Function("dirs3", [q], [dfa["L"], dfa["R"], R_tr @ e1, R_tr @ e2, R_tr @ e3])
        # symbolic helpers for simulation: free acceleration with hand forces
        tau = ca.SX.sym("tau", NTAU)
        RL = ca.SX.sym("RL", 3)
        RR = ca.SX.sym("RR", 3)
        rhs = Bmat @ tau - hvec + JL.T @ RL + JR.T @ RR
        self.f_qdd_free = ca.Function("qdd_free3", [q, qd, tau, RL, RR], [ca.solve(Mmat, rhs)])
        # implicit dynamics residual as ONE embedded function (its Jacobian is generated once and reused per node)
        acc = ca.SX.sym("acc", NQ)
        # reverse-mode AD for the embedded call (14 outputs << 56 inputs): 3x faster NLP Jacobians
        self.f_res = ca.Function("res3", [q, qd, acc, tau, RL, RR], [Mmat @ acc - rhs], {"ad_weight": 1.0, "ad_weight_sp": 1.0})
        # constraint-consistent acceleration for a set of pinned hands (index-1 with Baumgarte stabilisation)
        Jd_L = ca.reshape(ca.jtimes(ca.vec(JL), q, qd), 3, NQ)
        Jd_R = ca.reshape(ca.jtimes(ca.vec(JR), q, qd), 3, NQ)
        self.f_Jd = ca.Function("Jd3", [q, qd], [Jd_L, Jd_R])

    # -------------------------------------------------------------------- numpy wrappers ------------
    def points(self, q):
        P = np.array(self.f_points(q))
        return {k: P[:, i] for i, k in enumerate(self.point_names)}

    def hands(self, q):
        L, R = self.f_hands(q)
        return np.array(L).ravel(), np.array(R).ravel()

    def qdd_pinned(self, q, qd, tau, pinned, acc_targets, alpha=20.0, beta=100.0):
        """Acceleration and hand forces with `pinned` in {"L","R"} held at moving points.

        acc_targets: dict side -> (pos_target, vel_target, acc_target) of the ledge point.
        Baumgarte terms remove drift in the forward simulation used for verification."""
        M = np.array(self.f_M(q)); h = np.array(self.f_h(q, qd)).ravel()
        JL, JR = self.f_J(q); JdL, JdR = self.f_Jd(q, qd)
        J = {"L": np.array(JL), "R": np.array(JR)}; Jd = {"L": np.array(JdL), "R": np.array(JdR)}
        hl, hr = self.hands(q)
        hp = {"L": hl, "R": hr}
        hv = {"L": J["L"] @ qd, "R": J["R"] @ qd}
        sides = list(pinned)
        n = len(sides)
        Jc = np.vstack([J[s] for s in sides]) if n else np.zeros((0, NQ))
        rhs_c = []
        for s in sides:
            pt, vt, at = acc_targets[s]
            rhs_c.append(at - Jd[s] @ qd - 2 * alpha * (hv[s] - vt) - beta ** 2 * (hp[s] - pt) * 0 - beta * (hp[s] - pt))
        rhs_c = np.concatenate(rhs_c) if n else np.zeros(0)
        f = self.B @ tau - h
        K = np.block([[M, -Jc.T], [Jc, np.zeros((3 * n, 3 * n))]])
        sol = np.linalg.solve(K, np.concatenate([f, rhs_c]))
        qdd = sol[:NQ]
        Rf = {s: sol[NQ + 3 * i: NQ + 3 * i + 3] for i, s in enumerate(sides)}
        return qdd, Rf
