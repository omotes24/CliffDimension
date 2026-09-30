"""Gradient-based dual-field MPC for the pinned swing (IPOPT, implicit Hermite-Simpson as in the oracle).

Horizon H control periods (20 ms each, one HS interval per period). Decision variables: swing states, accelerations
(knots and mid-points), hand forces, torque commands. Constraints and running cost are those of the oracle's swing
phase (implicit dynamics residual, grip cone, capacity <= U_theta F_cap, joint ranges / speeds, walls, torque bounds,
activation rate). The dual field enters as the terminal cost: the value network is embedded symbolically (SiLU MLP in
CasADi, exact derivatives), so the MPC minimises  running cost + V_theta(x_H, t_H).  The release fires when
tau_theta <= dt/2. Warm start: the previous plan shifted by one period.
"""
from __future__ import annotations

import numpy as np
import casadi as ca
import torch

from .. import device
from ..planar.model import NTH, NQ, NTAU
from ..planar.fastsim import rel_bounds
from .dual_field import make_features, body_feats, A_REL

JT = ca.vertcat(ca.DM.eye(2), ca.DM.zeros(NTH, 2))


def smax(a, b, delta):
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + delta ** 2))


def mlp_casadi(net, z):
    """Symbolic forward pass of DualFieldNet.body (Linear/SiLU stack) on a standardised MX input z; returns (V, U, tau)
    de-standardised."""
    layers = [m for m in net.body.net if isinstance(m, torch.nn.Linear)]
    h = z
    for i, lin in enumerate(layers):
        W = ca.DM(lin.weight.detach().numpy().astype(float)); b = ca.DM(lin.bias.detach().numpy().astype(float))
        h = W @ h + b
        if i < len(layers) - 1:
            h = h / (1 + ca.exp(-h))                                     # SiLU
    y_mu = ca.DM(net.y_mu.numpy().astype(float)); y_sd = ca.DM(net.y_sd.numpy().astype(float))
    out = h * y_sd + y_mu
    return out[0], out[1], out[2]


class SwingMPC:
    def __init__(self, net, body, chain, T, H=25, control_dt=0.02, eps=0.20, w_E=1.0, t_ref=1.0, u_rate=10.0, U_margin=1.0,
                 stature=1.75, cap_scale=1.0, release_slack=0.5, wall_smooth=0.01, w_smooth=1e-3, max_iter=80, verbose=False,
                 terminal_weight=1.0, beta=2.0, replan=1, density=None, beta_d=1.0):
        """net: a DualFieldNet or a list of them (ensemble: terminal value = mean + beta * std, pessimistic where the
        members disagree, i.e. away from the oracle data)."""
        self.nets = list(net) if isinstance(net, (list, tuple)) else [net]
        self.net = self.nets[0]
        self.beta = beta
        self.replan = replan
        self.density, self.beta_d = density, beta_d
        self.body, self.ch, self.T = body, chain, float(T)
        self.H, self.dt, self.eps = H, control_dt, eps
        self.w_E, self.t_ref, self.u_rate, self.U_margin = w_E, t_ref, u_rate, U_margin
        self.stature, self.cap_scale, self.release_slack = stature, cap_scale, release_slack
        self.wall_smooth, self.w_smooth, self.max_iter, self.verbose = wall_smooth, w_smooth, max_iter, verbose
        self.terminal_weight = terminal_weight
        self.lo, self.hi = rel_bounds("neg")
        self._build()
        self.reset()

    # ------------------------------------------------------------------------------------------------
    def _dev(self, t):
        return device.device_ca(t, self.T, self.eps)

    def _build(self):
        b, ch, T, H, dt = self.body, self.ch, self.T, self.H, self.dt
        opti = ca.Opti()
        X = opti.variable(2 * NTH, H + 1); A = opti.variable(NTH, H + 1); Am = opti.variable(NTH, H)
        U = opti.variable(NTAU, H + 1); R = opti.variable(2, H + 1); Rm = opti.variable(2, H)
        x0 = opti.parameter(2 * NTH); t0 = opti.parameter(); u_prev = opti.parameter(NTAU); Ucap = opti.parameter()
        self.par = dict(x0=x0, t0=t0, u_prev=u_prev, Ucap=Ucap)
        tau_cap = ca.DM(b.tau_cap); f_cap = b.f_cap; Bm = ca.DM(ch.B)
        clear = np.concatenate([[0.0], b.clearance, b.forearm_clearance])

        def res(t, x, a, u, r):
            dv = self._dev(t)
            q = ca.vertcat(dv["pA"], x[:NTH]); qd = ca.vertcat(dv["vA"], x[NTH:])
            qdd = ca.vertcat(dv["aA"], a)
            return (ch.f_M(q) @ qdd + ch.f_h(q, qd) - Bm @ (u * tau_cap) - JT @ (r * f_cap)) / f_cap, q, dv

        def cone(r):
            opti.subject_to(r[1] >= 0)
            opti.subject_to(r[0] >= -b.mu_out * r[1]); opti.subject_to(r[0] <= b.mu_in * r[1])

        def eff(u, r):
            return 0.25 * ca.sumsqr(u) + ca.sumsqr(r)

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(-self.u_rate * dt, U[:, 0] - u_prev, self.u_rate * dt))
        E = []; F = []
        for k in range(H + 1):
            tk = t0 + k * dt
            rk, q, dv = res(tk, X[:, k], A[:, k], U[:, k], R[:, k])
            opti.subject_to(rk == 0)
            F.append(ca.vertcat(X[NTH:, k], A[:, k]))
            E.append(eff(U[:, k], R[:, k]))
            cone(R[:, k])
            opti.subject_to(ca.sumsqr(R[:, k]) <= Ucap ** 2)
            rel = ca.DM(A_REL) @ X[:NTH, k]; reld = ca.DM(A_REL) @ X[NTH:, k]
            opti.subject_to(opti.bounded(self.lo, rel, self.hi))
            opti.subject_to(opti.bounded(-b.qd_max, reld, b.qd_max))
            if k > 0:                                                     # walls (the initial state is given)
                pts = ca.horzcat(ch.f_tips(q), ch.f_forearm(q))
                for i in range(pts.shape[1]):
                    xx, yy = pts[0, i], pts[1, i]
                    aA = xx - (-device.D_LEDGE + clear[i + 1]); bA = (dv["h"] - device.WALL_BOTTOM_OFFSET) - yy
                    aB = (dv["x"] + device.D_LEDGE - clear[i + 1]) - xx; bB = -device.WALL_BOTTOM_OFFSET - yy
                    opti.subject_to(smax(aA, bA, self.wall_smooth) >= 0)
                    opti.subject_to(smax(aB, bB, self.wall_smooth) >= 0)
            opti.subject_to(opti.bounded(-1, U[:, k], 1))
            if k > 0:
                opti.subject_to(opti.bounded(-self.u_rate * dt, U[:, k] - U[:, k - 1], self.u_rate * dt))
        J = 0
        for k in range(H):
            tm = t0 + (k + 0.5) * dt
            xm = 0.5 * (X[:, k] + X[:, k + 1]) + dt / 8 * (F[k] - F[k + 1])
            fm = ca.vertcat(xm[NTH:], Am[:, k])
            um = 0.5 * (U[:, k] + U[:, k + 1])
            rm_, _, _ = res(tm, xm, Am[:, k], um, Rm[:, k])
            opti.subject_to(rm_ == 0)
            opti.subject_to(X[:, k + 1] - X[:, k] - dt / 6 * (F[k] + 4 * fm + F[k + 1]) == 0)
            cone(Rm[:, k])
            opti.subject_to(ca.sumsqr(Rm[:, k]) <= Ucap ** 2)
            J += dt / 6 * (E[k] + 4 * eff(um, Rm[:, k]) + E[k + 1]) * self.w_E / self.t_ref
            J += self.w_smooth * ca.sumsqr(U[:, k + 1] - U[:, k])
        # terminal: the dual field's value at (x_H, t_H)
        tH = t0 + H * dt
        ph = ca.fmod(tH, T) / T
        zf = ca.vertcat(X[:, H], ca.sin(2 * np.pi * ph), ca.cos(2 * np.pi * ph), ca.DM(body_feats(T, b.m, self.stature, self.cap_scale)))
        Vs = []
        for n_ in self.nets:
            mu = ca.DM(n_.mu.numpy().astype(float)); sd = ca.DM(n_.sd.numpy().astype(float))
            V_, _, _ = mlp_casadi(n_, (zf - mu) / sd)
            Vs.append(V_)
        Vmean = sum(Vs) / len(Vs)
        if len(Vs) > 1:
            Vvar = sum((V_ - Vmean) ** 2 for V_ in Vs) / len(Vs)
            VH = Vmean + self.beta * ca.sqrt(Vvar + 1e-6)
        else:
            VH = Vmean
        # out-of-distribution penalty: Gaussian-mixture density of the oracle data in the standardised feature space
        self.nll_expr = None
        if self.density is not None:
            w, means, prec_chol, c0 = self.density
            zs = (zf - ca.DM(self.net.mu.numpy().astype(float))) / ca.DM(self.net.sd.numpy().astype(float))
            terms = []
            for k in range(len(w)):
                Lk = ca.DM(prec_chol[k])                               # z-space: (z - mu)^T P (z - mu) with P = L L^T
                dz = Lk.T @ (zs - ca.DM(means[k]))
                logdet = float(np.sum(np.log(np.diag(prec_chol[k]))))
                terms.append(np.log(w[k]) + logdet - 0.5 * ca.sumsqr(dz))
            tv = ca.vertcat(*terms)
            mx = ca.mmax(tv)
            logp = mx + ca.log(ca.sum1(ca.exp(tv - mx)))
            nll = -logp
            self.nll_expr = nll
            VH = VH + self.beta_d * ca.fmax(nll - c0, 0.0) ** 2 / 10.0
        J += self.terminal_weight * VH
        self.VH_expr = VH
        opti.minimize(J)
        opts = {"expand": False, "ipopt.print_level": 5 if self.verbose else 0, "print_time": 0, "ipopt.max_iter": self.max_iter,
                "ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 5, "ipopt.mu_strategy": "adaptive",
                "ipopt.linear_solver": "mumps", "ipopt.sb": "yes", "ipopt.max_cpu_time": 10.0, "ipopt.warm_start_init_point": "yes",
                "ipopt.warm_start_bound_push": 1e-6, "ipopt.warm_start_mult_bound_push": 1e-6, "ipopt.mu_init": 1e-3}
        opti.solver("ipopt", opts)
        self.opti = opti
        self.v = dict(X=X, A=A, Am=Am, U=U, R=R, Rm=Rm)

    # ------------------------------------------------------------------------------------------------
    def reset(self):
        self.u_prev = np.zeros(NTAU)
        self.sol = None
        self.last = {}
        self.n_fail = 0
        self.lam_g = None
        self.plan_left = 0

    def field(self, x, t):
        z = torch.tensor(make_features(x, t, self.T, self.body.m, self.stature, self.cap_scale), dtype=torch.float32)
        with torch.no_grad():
            outs = [n_(z) for n_ in self.nets]
        V = np.mean([float(o[0][0]) for o in outs]); U = np.mean([float(o[1][0]) for o in outs]); tau = np.mean([float(o[2][0]) for o in outs])
        return V, U, tau

    def _guess(self, x, t):
        """Initial guess: shift the previous solution or hold the state (static accelerations from the dynamics)."""
        H = self.H
        if self.sol is not None:
            g = {k: np.array(v) for k, v in self.sol.items()}
            sft = self.replan
            for k in ("X", "A", "U", "R", "Am", "Rm"):
                g[k] = np.concatenate([g[k][:, sft:], np.repeat(g[k][:, -1:], sft, 1)], 1)
            g["X"][:, 0] = x
            return g
        Xg = np.tile(x[:, None], (1, H + 1)); Ug = np.tile(self.u_prev[:, None], (1, H + 1))
        Ag = np.zeros((NTH, H + 1)); Rg = np.zeros((2, H + 1))
        for k in range(H + 1):
            dv = device.device_state(t + k * self.dt, self.T, self.eps)
            q = np.concatenate([dv["pA"], x[:NTH]]); qd = np.concatenate([dv["vA"], x[NTH:]])
            thdd, Rk = self.ch.pinned(q, qd, self.u_prev * self.body.tau_cap, dv["aA"])
            Ag[:, k] = thdd; Rg[:, k] = Rk / self.body.f_cap
        return dict(X=Xg, A=Ag, Am=Ag[:, :H], U=Ug, R=Rg, Rm=Rg[:, :H])

    def act(self, th, thd, t):
        x = np.concatenate([th, thd])
        V0, U0, tau0 = self.field(x, t)
        release = tau0 <= self.release_slack * self.dt
        dv = device.device_state(t, self.T, self.eps)
        q = np.concatenate([dv["pA"], th]); qd = np.concatenate([dv["vA"], thd])
        _, Rnow = self.ch.pinned(q, qd, self.u_prev * self.body.tau_cap, dv["aA"])
        Ucap = max(self.U_margin * U0, np.linalg.norm(Rnow) / self.body.f_cap + 0.05, 0.6)
        opti = self.opti
        opti.set_value(self.par["x0"], x); opti.set_value(self.par["t0"], t); opti.set_value(self.par["u_prev"], self.u_prev)
        opti.set_value(self.par["Ucap"], Ucap)
        g = self._guess(x, t)
        for k, v in self.v.items():
            opti.set_initial(v, g[k])
        if self.lam_g is not None:
            try:
                opti.set_initial(opti.lam_g, self.lam_g)
            except Exception:
                pass
        ok = True
        try:
            sol = opti.solve()
            val = lambda e: sol.value(e)
            self.lam_g = np.array(sol.value(opti.lam_g))
            stats = sol.stats()
        except RuntimeError:
            val = lambda e: opti.debug.value(e)
            ok = False; self.n_fail += 1
            stats = opti.debug.stats()
        self.sol = {k: np.array(val(v)) for k, v in self.v.items()}
        u = np.clip(self.sol["U"][:, 0], -1, 1)
        u = np.clip(u, self.u_prev - self.u_rate * self.dt, self.u_prev + self.u_rate * self.dt)
        self.u_prev = u
        self.last = dict(V=V0, U=U0, tau=tau0, ok=ok, iters=stats.get("iter_count", -1), VH=float(val(self.VH_expr)), Ucap=Ucap,
                         nll=float(val(self.nll_expr)) if self.nll_expr is not None else float("nan"))
        return u, release

    def __call__(self, env):
        # re-plan every `replan` control periods; in between apply the stored plan (shifted)
        if self.plan_left > 0 and self.sol is not None:
            k = self.replan - self.plan_left
            u = np.clip(self.sol["U"][:, k], -1, 1)
            u = np.clip(u, self.u_prev - self.u_rate * self.dt, self.u_prev + self.u_rate * self.dt)
            self.u_prev = u
            _, _, tau0 = self.field(np.concatenate([env.th, env.thd]), env.t)
            release = tau0 <= self.release_slack * self.dt
            self.plan_left -= 1
        else:
            u, release = self.act(env.th, env.thd, env.t)
            self.plan_left = self.replan - 1
        a = np.zeros(NTAU + 1); a[:NTAU] = u; a[NTAU] = 1.0 if release else -1.0
        return a
