"""Dual-field MPC: short-horizon optimal control of the pinned swing with the learned dual field as terminal condition.

Pointwise Hamiltonian minimisation (Pontryagin controller) is ill-conditioned on this task: over one control period
the torques barely change the value, so a ~10% costate error flips the minimiser. Integrating the exact rigid-body
dynamics over a short horizon H (0.3-0.5 s) restores conditioning; the dual field enters at the end of the horizon
as a linearised terminal cost  V(x_H) ~ V_theta(xbar) + p_theta(xbar)^T (x_H - xbar)  (xbar = terminal state of the
previous plan; one re-linearisation per control step) plus the running cost of the optimiser. Path constraints
(torque bounds, activation rate, grip cone, capacity <= U_theta F_cap, joint ranges / speeds, walls) are those of
the oracle. Multiple shooting, RK4 with 2 sub-steps per 20 ms, CasADi Opti / IPOPT, warm-started by shifting.
"""
from __future__ import annotations

import numpy as np
import casadi as ca
import torch

from .. import device
from ..planar.model import NTH, NQ, NTAU
from ..planar.fastsim import rel_bounds
from .dual_field import make_features, A_REL


def smax(a, b, delta):
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + delta ** 2))


class DualFieldMPC:
    def __init__(self, net, body, chain, T, H=20, control_dt=0.02, n_sub=2, eps=0.20, w_E=1.0, t_ref=1.0, u_rate=10.0,
                 U_margin=1.0, stature=1.75, cap_scale=1.0, release_slack=0.5, rho=0.0, wall_smooth=0.01, ipopt_iter=60,
                 verbose=False):
        self.net, self.body, self.ch, self.T = net, body, chain, float(T)
        self.H, self.dt, self.n_sub, self.eps = H, control_dt, n_sub, eps
        self.w_E, self.t_ref, self.u_rate, self.U_margin = w_E, t_ref, u_rate, U_margin
        self.stature, self.cap_scale, self.release_slack, self.rho = stature, cap_scale, release_slack, rho
        self.wall_smooth = wall_smooth
        self.lo, self.hi = rel_bounds("neg")
        self.ipopt_iter = ipopt_iter
        self.verbose = verbose
        self._build()
        self.reset()

    # ------------------------------------------------------------------------------------------------
    def _build(self):
        b, ch, T = self.body, self.ch, self.T
        H, dt, ns = self.H, self.dt, self.n_sub
        tc = ca.DM(b.tau_cap)
        opti = ca.Opti()
        X = opti.variable(2 * NTH, H + 1)
        U = opti.variable(NTAU, H)
        x0 = opti.parameter(2 * NTH); t0 = opti.parameter()
        u_prev = opti.parameter(NTAU)
        p_term = opti.parameter(2 * NTH); x_term = opti.parameter(2 * NTH); Ucap = opti.parameter()
        self.par = dict(x0=x0, t0=t0, u_prev=u_prev, p_term=p_term, x_term=x_term, Ucap=Ucap)
        Bm = ca.DM(ch.B)

        def rhs(t, x, u):
            dv = device.device_ca(t, T, self.eps)
            q = ca.vertcat(dv["pA"], x[:NTH]); qd = ca.vertcat(dv["vA"], x[NTH:])
            thdd, R = ch.f_pinned(q, qd, u * tc, dv["aA"])
            return ca.vertcat(x[NTH:], thdd), R, q, qd, dv

        opti.subject_to(X[:, 0] == x0)
        J = 0
        hs = dt / ns
        for k in range(H):
            xk = X[:, k]; uk = U[:, k]; tk = t0 + k * dt
            # RK4 sub-steps
            xs = xk
            for j in range(ns):
                ts = tk + j * hs
                k1, _, _, _, _ = rhs(ts, xs, uk); k2, _, _, _, _ = rhs(ts + 0.5 * hs, xs + 0.5 * hs * k1, uk)
                k3, _, _, _, _ = rhs(ts + 0.5 * hs, xs + 0.5 * hs * k2, uk); k4, _, _, _, _ = rhs(ts + hs, xs + hs * k3, uk)
                xs = xs + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == xs)
            # running cost and path constraints at the knot (hand force from the pinned dynamics)
            _, R, q, qd, dv = rhs(tk, xk, uk)
            r = R / b.f_cap
            J += dt * self.w_E / self.t_ref * (0.25 * ca.sumsqr(uk) + ca.sumsqr(r))
            opti.subject_to(r[1] >= 0)
            opti.subject_to(r[0] >= -b.mu_out * r[1]); opti.subject_to(r[0] <= b.mu_in * r[1])
            opti.subject_to(ca.sumsqr(r) <= Ucap ** 2)
            rel = ca.DM(A_REL) @ xk[:NTH]; reld = ca.DM(A_REL) @ xk[NTH:]
            opti.subject_to(opti.bounded(self.lo + 0.01, rel, self.hi - 0.01))
            opti.subject_to(opti.bounded(-b.qd_max, reld, b.qd_max))
            # walls (clearance points, as in the oracle)
            pts = ca.horzcat(q[0:2], ch.f_tips(q), ch.f_forearm(q))
            clear = np.concatenate([[0.0], b.clearance, b.forearm_clearance])
            for i in range(1, pts.shape[1]):                                  # (the hand itself is pinned on A)
                xx, yy = pts[0, i], pts[1, i]
                aA = xx - (-device.D_LEDGE + clear[i]); bA = (dv["h"] - device.WALL_BOTTOM_OFFSET) - yy
                aB = (dv["x"] + device.D_LEDGE - clear[i]) - xx; bB = -device.WALL_BOTTOM_OFFSET - yy
                opti.subject_to(smax(aA, bA, self.wall_smooth) >= 0)
                opti.subject_to(smax(aB, bB, self.wall_smooth) >= 0)
            opti.subject_to(opti.bounded(-1, uk, 1))
            prev = u_prev if k == 0 else U[:, k - 1]
            opti.subject_to(opti.bounded(-self.u_rate * dt, uk - prev, self.u_rate * dt))
        # terminal: linearised dual field (+ optional proximal term)
        xH = X[:, H]
        J += ca.dot(p_term, xH - x_term) + 0.5 * self.rho * ca.sumsqr(xH - x_term)
        opti.minimize(J)
        opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": self.ipopt_iter, "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 5, "ipopt.warm_start_init_point": "yes",
                "ipopt.mu_init": 1e-2, "ipopt.linear_solver": "mumps", "ipopt.sb": "yes", "ipopt.max_cpu_time": 5.0}
        opti.solver("ipopt", opts)
        self.opti, self.X, self.U = opti, X, U

    # ------------------------------------------------------------------------------------------------
    def field(self, x, t):
        z = torch.tensor(make_features(x, t, self.T, self.body.m, self.stature, self.cap_scale), dtype=torch.float32)
        V, U, tau, p = self.net.value_and_costate(z)
        return float(V[0]), float(U[0]), float(tau[0]), p[0].detach().numpy()

    def reset(self):
        self.u_prev = np.zeros(NTAU)
        self.Xg = None; self.Ug = None
        self.last = {}
        self.n_fail = 0

    def rollout_guess(self, x, t, Useq):
        """Integrate the guess with the same RK4 (numpy via the chain functions) to get the terminal state."""
        xs = x.copy(); hs = self.dt / self.n_sub
        tc = self.body.tau_cap
        Xs = [xs.copy()]
        for k in range(self.H):
            u = Useq[:, k]
            for j in range(self.n_sub):
                ts = t + k * self.dt + j * hs

                def f(tt, xx):
                    dv = device.device_state(tt, self.T, self.eps)
                    q = np.concatenate([dv["pA"], xx[:NTH]]); qd = np.concatenate([dv["vA"], xx[NTH:]])
                    thdd, _ = self.ch.pinned(q, qd, u * tc, dv["aA"])
                    return np.concatenate([xx[NTH:], thdd])
                k1 = f(ts, xs); k2 = f(ts + 0.5 * hs, xs + 0.5 * hs * k1); k3 = f(ts + 0.5 * hs, xs + 0.5 * hs * k2); k4 = f(ts + hs, xs + hs * k3)
                xs = xs + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Xs.append(xs.copy())
        return np.stack(Xs, 1)

    def act(self, th, thd, t):
        x = np.concatenate([th, thd])
        V0, U0, tau0, p0 = self.field(x, t)
        self.last = dict(V=V0, U=U0, tau=tau0)
        release = tau0 <= self.release_slack * self.dt
        # warm start: shift the previous plan
        if self.Ug is None:
            Ug = np.tile(self.u_prev[:, None], (1, self.H))
        else:
            Ug = np.concatenate([self.Ug[:, 1:], self.Ug[:, -1:]], 1)
        Xg = self.rollout_guess(x, t, Ug)
        xH = Xg[:, -1]; tH = t + self.H * self.dt
        VH, UH, tauH, pH = self.field(xH, tH)
        Ucap = max(self.U_margin * max(U0, UH), np.linalg.norm(self.dyn_R(x, t)) / self.body.f_cap + 0.05, 0.6)
        opti = self.opti
        opti.set_value(self.par["x0"], x); opti.set_value(self.par["t0"], t); opti.set_value(self.par["u_prev"], self.u_prev)
        opti.set_value(self.par["p_term"], pH); opti.set_value(self.par["x_term"], xH); opti.set_value(self.par["Ucap"], Ucap)
        opti.set_initial(self.X, Xg); opti.set_initial(self.U, Ug)
        try:
            sol = opti.solve()
            Uopt = np.array(sol.value(self.U)).reshape(NTAU, self.H); Xopt = np.array(sol.value(self.X))
            ok = True
        except RuntimeError:
            Uopt = np.array(opti.debug.value(self.U)).reshape(NTAU, self.H); Xopt = np.array(opti.debug.value(self.X))
            ok = False; self.n_fail += 1
        u = np.clip(Uopt[:, 0], -1, 1)
        u = np.clip(u, self.u_prev - self.u_rate * self.dt, self.u_prev + self.u_rate * self.dt)
        self.Ug, self.Xg = Uopt, Xopt
        self.u_prev = u
        self.last.update(ok=ok, VH=VH, tauH=tauH)
        return u, release

    def dyn_R(self, x, t):
        dv = device.device_state(t, self.T, self.eps)
        q = np.concatenate([dv["pA"], x[:NTH]]); qd = np.concatenate([dv["vA"], x[NTH:]])
        _, R = self.ch.pinned(q, qd, self.u_prev * self.body.tau_cap, dv["aA"])
        return R

    def __call__(self, env):
        u, release = self.act(env.th, env.thd, env.t)
        a = np.zeros(NTAU + 1); a[:NTAU] = u; a[NTAU] = 1.0 if release else -1.0
        return a


class DualFieldMPPI:
    """Sampling MPC (MPPI) over a short horizon with the compiled swing simulator and the dual field as terminal value.

    cost(sample) = sum_k dt [w_E/t_ref (0.25|u|^2 + U^2)] + penalties(cone, capacity > U_theta, joint range/speed, wall)
                   + V_theta(x_H, t_H)
    Torque sequences are sampled around the shifted previous mean, rate-limited and clipped; K samples are rolled
    out in one call of the mapped compiled function (fastsim A, 10 sub-steps per control step).
    """

    def __init__(self, net, body, chain, sim, T, H=20, K=64, iters=2, sigma=0.2, lam=0.3, control_dt=0.02, eps=0.20,
                 w_E=1.0, t_ref=1.0, u_rate=10.0, U_margin=1.0, stature=1.75, cap_scale=1.0, release_slack=0.5,
                 pen=dict(cone=20.0, cap=20.0, joint=20.0, wall=200.0, speed=2.0), seed=0, terminal="value"):
        self.net, self.body, self.ch, self.sim, self.T = net, body, chain, sim, float(T)
        self.H, self.K, self.iters, self.sigma, self.lam = H, K, iters, sigma, lam
        self.dt, self.eps, self.w_E, self.t_ref, self.u_rate, self.U_margin = control_dt, eps, w_E, t_ref, u_rate, U_margin
        self.stature, self.cap_scale, self.release_slack, self.pen = stature, cap_scale, release_slack, dict(pen)
        self.terminal = terminal
        self.lo, self.hi = rel_bounds("neg")
        self.fmap = sim.fA.map(K)
        self.n_sub = sim.n_sub
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.u_prev = np.zeros(NTAU)
        self.mu = None
        self.last = {}

    def field_batch(self, X, t):
        """V, U, tau for a batch of states X (n, 10) at time t."""
        F = make_features(X, np.full(X.shape[0], t), self.T, self.body.m, self.stature, self.cap_scale)
        with torch.no_grad():
            V, U, tau = self.net(torch.tensor(F, dtype=torch.float32))
        return V.numpy(), U.numpy(), tau.numpy()

    def rollout_cost(self, x, t, Useq, Ucap):
        """Useq: (K, H, 4). Returns cost (K,), terminal states (K, 10)."""
        K, H = Useq.shape[0], Useq.shape[1]
        ns = self.n_sub
        X = np.tile(x[:, None], (1, K))
        cost = np.zeros(K)
        f_cap = self.body.f_cap
        for k in range(H):
            tk = t + k * self.dt
            tt = np.tile(tk + self.sim.sub_dt * np.arange(ns), K)
            TT = np.full(ns * K, self.T)
            tau = np.repeat(Useq[:, k, :].T, ns, axis=1)                 # (4, ns*K)
            out = self.fmap(X, tt.reshape(1, -1), TT.reshape(1, -1), tau)
            Xs = np.array(out[0]); R = np.array(out[1]); jr = np.array(out[2]).ravel(); wall = np.array(out[3]).ravel()
            Xs = Xs.reshape(2 * NTH, K, ns); R = R.reshape(2, K, ns); jr = jr.reshape(K, ns); wall = wall.reshape(K, ns)
            Uk = np.sqrt((R ** 2).sum(0)) / f_cap                         # (K, ns)
            rx, ry = R[0] / f_cap, R[1] / f_cap
            cone = np.maximum.reduce([-ry, -(rx + self.body.mu_out * np.maximum(ry, 0)), rx - self.body.mu_in * np.maximum(ry, 0), np.zeros_like(rx)])
            u = Useq[:, k, :]
            cost += self.dt * self.w_E / self.t_ref * (0.25 * (u ** 2).sum(1) + (Uk ** 2).mean(1))
            cost += self.pen["cone"] * cone.sum(1) * self.sim.sub_dt + self.pen["cap"] * np.maximum(Uk - Ucap, 0).sum(1) * self.sim.sub_dt
            cost += self.pen["joint"] * jr.sum(1) * self.sim.sub_dt + self.pen["wall"] * wall.sum(1) * self.sim.sub_dt
            reld = np.abs(A_REL @ Xs[NTH:, :, -1])                          # (4, K)
            cost += self.pen["speed"] * np.maximum(reld - self.body.qd_max, 0).sum(0) * self.dt
            X = Xs[:, :, -1]
        return cost, X.T

    def act(self, th, thd, t):
        """CEM / MPPI hybrid: K samples around the shifted mean with rate-smooth (integrated) noise, elite refit,
        `iters` rounds with shrinking spread; the mean is re-weighted with MPPI weights in the last round."""
        x = np.concatenate([th, thd])
        V0, U0, tau0 = self.field_batch(x[None, :], t)
        V0, U0, tau0 = float(V0[0]), float(U0[0]), float(tau0[0])
        release = tau0 <= self.release_slack * self.dt
        H, K = self.H, self.K
        if self.mu is None:
            mu = np.tile(self.u_prev, (H, 1))
        else:
            mu = np.concatenate([self.mu[1:], self.mu[-1:]], 0)
        Ucap = max(self.U_margin * U0, 0.6)
        du = self.u_rate * self.dt
        sig = np.full((H, NTAU), self.sigma)
        n_elite = max(K // 8, 4)
        for it in range(self.iters):
            # smooth noise: random walk in the command (respects the rate limit statistically)
            steps = self.rng.normal(0, 1.0, (K, H, NTAU)) * (sig[None] / np.sqrt(H))
            eps = np.cumsum(steps, axis=1) + self.rng.normal(0, 0.3, (K, 1, NTAU)) * sig[None, :1]
            eps[0] = 0.0                                                # the mean itself
            Useq = mu[None] + eps
            prev = np.tile(self.u_prev, (K, 1))
            for k in range(H):
                Useq[:, k] = np.clip(np.clip(Useq[:, k], prev - du, prev + du), -1, 1)
                prev = Useq[:, k]
            cost, XH = self.rollout_cost(x, t, Useq, Ucap)
            if self.terminal == "value":
                VH, _, _ = self.field_batch(XH, t + H * self.dt)
                cost = cost + VH
            elif callable(self.terminal):
                cost = cost + self.terminal(XH, t + H * self.dt)
            elite = np.argsort(cost)[:n_elite]
            if it < self.iters - 1:
                mu = Useq[elite].mean(0); sig = Useq[elite].std(0) + 0.02
            else:
                lam = max(self.lam, 0.5 * (cost[elite].std() + 1e-6))
                w = np.exp(-(cost[elite] - cost[elite].min()) / lam); w /= w.sum()
                mu = np.tensordot(w, Useq[elite], axes=(0, 0))
        self.mu = mu
        u = np.clip(mu[0], self.u_prev - du, self.u_prev + du)
        self.u_prev = u
        self.last = dict(V=V0, U=U0, tau=tau0, cost_min=float(cost.min()), ess=float(1.0 / (w ** 2).sum()))
        return u, release

    def __call__(self, env):
        u, release = self.act(env.th, env.thd, env.t)
        a = np.zeros(NTAU + 1); a[:NTAU] = u; a[NTAU] = 1.0 if release else -1.0
        return a
