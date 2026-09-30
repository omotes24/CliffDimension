"""Dual-field learning (DFL) on the planar Cliff-Dimension task.

The physics oracle (trajectory optimisation) labels swing states (x, t; body) with the DUAL of the task:
    V*(x, t)          optimal cost-to-go (optimiser objective)
    p*(x, t) = dV*/dx costate (multipliers of the initial condition; envelope theorem)
    dV*/dt
    U*(x, t)          required grip capacity from here
    tau*(x, t)        optimal remaining swing time (time-to-release)
    lambda_k          multiplier mass of every constraint category (prices)
A dual-field network V_theta(x, t, body) is trained with a Sobolev loss (value + costate); the controller is then
obtained WITHOUT policy learning, pointwise, from Pontryagin's minimum principle:
    u(x,t) = argmin_u  l(x,u) + p_theta(x,t)^T f(x,u,t)   s.t. torque bounds, grip cone, |R| <= U_theta f_cap,
                                                              joint-range look-ahead, torque-rate limit
using the known rigid-body dynamics (control-affine). The release fires when tau_theta <= dt/2.
Baselines: behaviour cloning of the oracle torques (primal) with the same release timer.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from .. import device
from ..planar.anthro import G, make_body
from ..planar.model import PlanarChain, NTH, NQ, NTAU
from ..planar.fastsim import rel_bounds

A_REL = np.array([[1, -1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1]], float)   # rel = A th


# ---------------------------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------------------------
def body_feats(T, m, stature=1.75, cap_scale=1.0):
    return np.array([T / 20.0 - 1.0, m / 66.0 - 1.0, stature / 1.75 - 1.0, cap_scale - 1.0], float)


def make_features(x, t, T, m, stature=1.75, cap_scale=1.0):
    """x: (..., 10) swing state, t: (...,) absolute time. Returns (..., 16) features (raw units; standardised in the net)."""
    x = np.atleast_2d(np.asarray(x, float)); t = np.atleast_1d(np.asarray(t, float))
    ph = (t % T) / T
    tf = np.stack([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)], -1)
    bf = np.broadcast_to(body_feats(T, m, stature, cap_scale), (x.shape[0], 4))
    return np.concatenate([x, tf, bf], -1)


def load_rows(csv_path):
    """Point labels (one per from-state solve) from the dataset table."""
    import pandas as pd
    d = pd.read_csv(csv_path)
    d = d[d["ok"] == 1].copy()
    X = d[[f"x{i}" for i in range(2 * NTH)]].values
    F = np.stack([make_features(X[i], d["t0"].values[i], d["T"].values[i], d["m"].values[i], d["stature"].values[i], d["cap_scale"].values[i])[0]
                  for i in range(len(d))])
    P = d[[f"p{i}" for i in range(2 * NTH)]].values
    Y = dict(J=d["J"].values, U=d["U"].values, tau=d["d_s"].values, u=d[[f"u{i}" for i in range(4)]].values, p=P,
             dJdt=d["dJ_dt0"].values)
    d["dense"] = 0
    return d, F, Y


def load_dataset(data_dir, dense=True, min_tau=0.05):
    """Point labels (rows.csv) + dense trajectory labels (defect multipliers of every solution pickle)."""
    import pandas as pd
    from .labels import dense_rows_from_dir
    d, F, Y = load_rows(os.path.join(data_dir, "rows.csv"))
    if not dense:
        return d, F, Y
    rows = [r for r in dense_rows_from_dir(data_dir, min_tau=min_tau) if r["knot"] > 0]      # knot 0 = the point label
    if not rows:
        return d, F, Y
    Xd = np.stack([r["x"] for r in rows])
    Fd = np.stack([make_features(r["x"], r["t0"], r["T"], r["m"], r["stature"], r["cap_scale"])[0] for r in rows])
    Yd = dict(J=np.array([r["J"] for r in rows]), U=np.array([r["U"] for r in rows]), tau=np.array([r["tau"] for r in rows]),
              u=np.stack([r["u"] for r in rows]), p=np.stack([r["p"] for r in rows]), dJdt=np.full(len(rows), np.nan))
    dd = pd.DataFrame(dict(ref=[r["ref"] for r in rows], T=[r["T"] for r in rows], m=[r["m"] for r in rows], level=[r["level"] for r in rows],
                           t0=[r["t0"] for r in rows], dense=1, src=[r["src"] for r in rows], knot=[r["knot"] for r in rows]))
    for i in range(2 * NTH):
        dd[f"x{i}"] = Xd[:, i]
    d_all = pd.concat([d, dd], ignore_index=True)
    F_all = np.concatenate([F, Fd])
    Y_all = {k: np.concatenate([Y[k], Yd[k]]) for k in Y}
    Y_all["w"] = np.concatenate([np.full(len(d), 5.0), np.ones(len(dd))])       # exact point costates count more
    return d_all, F_all, Y_all


# ---------------------------------------------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_in, n_out, width=256, depth=3, act=nn.SiLU):
        super().__init__()
        layers = []; d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]; d = width
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class DualFieldNet(nn.Module):
    """z (16 features) -> (V, U, tau). p = dV/dx by autograd (in raw state units)."""

    def __init__(self, mu, sd, y_mu, y_sd, width=256, depth=3):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32)); self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.register_buffer("y_mu", torch.tensor(y_mu, dtype=torch.float32)); self.register_buffer("y_sd", torch.tensor(y_sd, dtype=torch.float32))
        self.body = MLP(mu.shape[0], 3, width, depth)

    def forward(self, z):
        out = self.body((z - self.mu) / self.sd) * self.y_sd + self.y_mu
        return out[:, 0], out[:, 1], out[:, 2]            # V, U, tau

    def value_and_costate(self, z):
        z = z.clone().requires_grad_(True)
        V, U, tau = self.forward(z)
        p = torch.autograd.grad(V.sum(), z, create_graph=True)[0][:, :2 * NTH]
        return V, U, tau, p


class BCNet(nn.Module):
    def __init__(self, mu, sd, width=256, depth=3):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32)); self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.body = MLP(mu.shape[0], NTAU, width, depth)

    def forward(self, z):
        return torch.tanh(self.body((z - self.mu) / self.sd))


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_dual_field(F, Y, idx_tr, idx_va, epochs=4000, lr=2e-3, alpha=1.0, width=256, depth=3, seed=0, verbose=True, log_every=500,
                     batch=8192, weight_decay=0.0):
    """Sobolev training: value + costate (+ U, tau heads). alpha = 0 gives the value-only ablation.
    `epochs` counts passes when the training set fits in one batch, otherwise optimisation steps."""
    torch.manual_seed(seed)
    dev = _device()
    mu, sd = F[idx_tr].mean(0), F[idx_tr].std(0) + 1e-6
    ymat = np.stack([Y["J"], Y["U"], Y["tau"]], 1)
    y_mu, y_sd = ymat[idx_tr].mean(0), ymat[idx_tr].std(0) + 1e-6
    net = DualFieldNet(mu, sd, y_mu, y_sd, width, depth).to(dev)
    Ft = torch.tensor(F, dtype=torch.float32, device=dev); yt = torch.tensor(ymat, dtype=torch.float32, device=dev)
    pt = torch.tensor(Y["p"], dtype=torch.float32, device=dev)
    # robust per-component scale of the costate (median absolute deviation): the loss must resolve the typical
    # (small) costates, not only the outliers near active constraints
    pr = Y["p"][idx_tr]
    mad = np.median(np.abs(pr - np.median(pr, 0)), 0) * 1.4826
    p_sd = torch.tensor(np.maximum(mad, 0.05 * (pr.std(0) + 1e-6)) + 1e-6, dtype=torch.float32, device=dev)
    wt = torch.tensor(Y.get("w", np.ones(len(F))), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = torch.tensor(idx_tr, device=dev); va = torch.tensor(idx_va, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []

    def losses(ix):
        V, U, tau, p = net.value_and_costate(Ft[ix])
        w = wt[ix] / wt[ix].mean()
        lv = (w * ((V - yt[ix, 0]) / net.y_sd[0]).pow(2)).mean(); lu = (w * ((U - yt[ix, 1]) / net.y_sd[1]).pow(2)).mean()
        lt = (w * ((tau - yt[ix, 2]) / net.y_sd[2]).pow(2)).mean()
        ep = ((p - pt[ix]) / p_sd).pow(2).mean(1)
        lp = (w * torch.clamp(ep, max=100.0)).mean()                     # clip outliers (near-constraint spikes)
        return lv, lu, lt, lp

    for ep in range(epochs):
        net.train()
        ix = tr if len(tr) <= batch else tr[torch.randint(0, len(tr), (batch,), generator=g).to(dev)]
        lv, lu, lt, lp = losses(ix)
        loss = lv + lu + lt + alpha * lp
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep % log_every == 0 or ep == epochs - 1):
            net.eval()
            vv = va if len(va) <= 4 * batch else va[:4 * batch]
            lvv, luv, ltv, lpv = losses(vv)
            r = dict(ep=ep, loss=float(loss), lv=float(lv), lu=float(lu), lt=float(lt), lp=float(lp),
                     va_V=float(lvv), va_U=float(luv), va_tau=float(ltv), va_p=float(lpv))
            hist.append(r)
            if verbose:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    return net.cpu(), hist


def train_bc(F, Y, idx_tr, idx_va, epochs=4000, lr=2e-3, width=256, depth=3, seed=0, verbose=True, log_every=500, batch=8192):
    torch.manual_seed(seed)
    dev = _device()
    mu, sd = F[idx_tr].mean(0), F[idx_tr].std(0) + 1e-6
    net = BCNet(mu, sd, width, depth).to(dev)
    Ft = torch.tensor(F, dtype=torch.float32, device=dev); ut = torch.tensor(Y["u"], dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = torch.tensor(idx_tr, device=dev); va = torch.tensor(idx_va, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    for ep in range(epochs):
        net.train()
        ix = tr if len(tr) <= batch else tr[torch.randint(0, len(tr), (batch,), generator=g).to(dev)]
        loss = (net(Ft[ix]) - ut[ix]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % log_every == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                r = dict(ep=ep, loss=float(loss), va_u=float((net(Ft[va]) - ut[va]).pow(2).mean()))
            hist.append(r)
            if verbose:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    return net.cpu(), hist


def r2(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return 1 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


# ---------------------------------------------------------------------------------------------------------------
# controllers
# ---------------------------------------------------------------------------------------------------------------
class SwingDynamics:
    """Control-affine pinned-swing dynamics of the environment's chain: thdd = c + D u, hand force R = R0 + Ru u."""

    def __init__(self, body, chain):
        self.body, self.ch = body, chain
        self.Bm = chain.B * body.tau_cap[None, :]           # 7 x 4 (torque per unit command)

    def affine(self, th, thd, t, T, eps=0.20):
        dv = device.device_state(t, T, eps)
        q = np.concatenate([dv["pA"], th]); qd = np.concatenate([dv["vA"], thd])
        M = np.array(self.ch.f_M(q)); h = np.array(self.ch.f_h(q, qd)).ravel()
        aA = dv["aA"]
        Mtt, Mth, Mht, Mhh = M[2:, 2:], M[2:, :2], M[:2, 2:], M[:2, :2]
        Mtt_inv = np.linalg.inv(Mtt)
        c = Mtt_inv @ (-h[2:] - Mth @ aA)
        D = Mtt_inv @ self.Bm[2:, :]
        R0 = Mhh @ aA + Mht @ c + h[:2]
        Ru = Mht @ D
        return c, D, R0, Ru


class PontryaginController:
    """Swing controller from a dual field: pointwise constrained Hamiltonian minimisation (SLSQP, 4 variables)."""

    def __init__(self, net, body, chain, T, control_dt=0.02, w_E=1.0, t_ref=1.0, u_rate=10.0, U_margin=1.0,
                 eps=0.20, stature=1.75, cap_scale=1.0, release_slack=0.5, tau_min_release=0.0):
        self.net, self.body, self.T, self.dt = net, body, T, control_dt
        self.dyn = SwingDynamics(body, chain)
        self.w_E, self.t_ref, self.u_rate, self.U_margin, self.eps = w_E, t_ref, u_rate, U_margin, eps
        self.stature, self.cap_scale = stature, cap_scale
        self.release_slack = release_slack
        self.lo, self.hi = rel_bounds("neg")
        self.u_prev = np.zeros(NTAU)
        self.last = {}

    def field(self, th, thd, t):
        z = torch.tensor(make_features(np.concatenate([th, thd]), t, self.T, self.body.m, self.stature, self.cap_scale), dtype=torch.float32)
        V, U, tau, p = self.net.value_and_costate(z)
        return float(V[0]), float(U[0]), float(tau[0]), p[0].detach().numpy()

    def act(self, th, thd, t):
        from scipy.optimize import minimize
        V, Uhat, tau, p = self.field(th, thd, t)
        self.last = dict(V=V, U=Uhat, tau=tau)
        release = tau <= self.release_slack * self.dt
        c, D, R0, Ru = self.dyn.affine(th, thd, t, self.T, self.eps)
        f_cap = self.body.f_cap
        p_thd = p[NTH:]
        kE = self.w_E / self.t_ref
        rel = A_REL @ th; reld = A_REL @ thd
        mu_out, mu_in = self.body.mu_out, self.body.mu_in
        Ucap = max(self.U_margin * Uhat, np.linalg.norm(R0) / f_cap + 0.05, 0.6)

        def H(u):
            R = R0 + Ru @ u
            return kE * (0.25 * u @ u + R @ R / f_cap ** 2) + p_thd @ (D @ u)

        def dH(u):
            R = R0 + Ru @ u
            return kE * (0.5 * u + 2 * Ru.T @ R / f_cap ** 2) + D.T @ p_thd

        dt = self.dt
        cons = [
            dict(type="ineq", fun=lambda u: (R0 + Ru @ u)[1] / f_cap),                                   # Ry >= 0
            dict(type="ineq", fun=lambda u: ((R0 + Ru @ u)[0] + mu_out * (R0 + Ru @ u)[1]) / f_cap),    # cone
            dict(type="ineq", fun=lambda u: (mu_in * (R0 + Ru @ u)[1] - (R0 + Ru @ u)[0]) / f_cap),
            dict(type="ineq", fun=lambda u: Ucap ** 2 - ((R0 + Ru @ u) @ (R0 + Ru @ u)) / f_cap ** 2),  # capacity
            dict(type="ineq", fun=lambda u: (self.hi - 0.01) - (rel + reld * dt + 0.5 * (A_REL @ (c + D @ u)) * dt ** 2)),
            dict(type="ineq", fun=lambda u: (rel + reld * dt + 0.5 * (A_REL @ (c + D @ u)) * dt ** 2) - (self.lo + 0.01)),
            dict(type="ineq", fun=lambda u: self.u_rate * dt - (u - self.u_prev)),
            dict(type="ineq", fun=lambda u: self.u_rate * dt + (u - self.u_prev)),
            dict(type="ineq", fun=lambda u: 0.95 * self.body.qd_max - (reld + (A_REL @ (c + D @ u)) * dt)),   # joint-speed look-ahead
            dict(type="ineq", fun=lambda u: 0.95 * self.body.qd_max + (reld + (A_REL @ (c + D @ u)) * dt)),
        ]
        u0 = np.clip(self.u_prev, -1, 1)
        res = minimize(H, u0, jac=dH, bounds=[(-1, 1)] * NTAU, constraints=cons, method="SLSQP", options=dict(maxiter=60, ftol=1e-8))
        u = np.clip(res.x, -1, 1) if res.success or np.isfinite(res.fun) else u0
        # keep the rate limit even if SLSQP returned an infeasible point
        u = np.clip(u, self.u_prev - self.u_rate * dt, self.u_prev + self.u_rate * dt)
        self.u_prev = u
        return u, release

    def __call__(self, env):
        th, thd, t = env.th, env.thd, env.t
        u, release = self.act(th, thd, t)
        a = np.zeros(NTAU + 1); a[:NTAU] = u; a[NTAU] = 1.0 if release else -1.0
        return a

    def reset(self):
        self.u_prev = np.zeros(NTAU)


class BCController:
    """Behaviour-cloned swing torques (primal) with the same release timer (tau head of the dual net)."""

    def __init__(self, bc_net, timer_net, body, T, control_dt=0.02, u_rate=10.0, stature=1.75, cap_scale=1.0, release_slack=0.5):
        self.bc, self.timer, self.body, self.T, self.dt = bc_net, timer_net, body, T, control_dt
        self.u_rate, self.stature, self.cap_scale, self.release_slack = u_rate, stature, cap_scale, release_slack
        self.u_prev = np.zeros(NTAU)

    def act(self, th, thd, t):
        z = torch.tensor(make_features(np.concatenate([th, thd]), t, self.T, self.body.m, self.stature, self.cap_scale), dtype=torch.float32)
        with torch.no_grad():
            u = self.bc(z)[0].numpy()
            _, _, tau = self.timer(z)
        u = np.clip(u, self.u_prev - self.u_rate * self.dt, self.u_prev + self.u_rate * self.dt)
        self.u_prev = u
        return u, float(tau[0]) <= self.release_slack * self.dt

    def __call__(self, env):
        u, release = self.act(env.th, env.thd, env.t)
        a = np.zeros(NTAU + 1); a[:NTAU] = u; a[NTAU] = 1.0 if release else -1.0
        return a

    def reset(self):
        self.u_prev = np.zeros(NTAU)


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def run_episode(env, controller, start, max_steps=None):
    """start: dict for env.reset(options=...). Returns env.result plus the controller's last field values."""
    obs, info = env.reset(options=start)
    if max_steps is None:
        max_steps = int(round(40.0 / env.control_dt))
    controller.reset()
    done = False; k = 0; U_swing = 0.0
    while not done and k < max_steps:
        a = controller(env.env if hasattr(env, "env") else env)
        obs, r, term, trunc, inf = env.step(a)
        if env.mode == "A":
            U_swing = max(U_swing, env.U)
        done = term or trunc; k += 1
    res = dict(env.result or dict(success=False, reason="max_steps", U_peak=env.U_peak, t=env.t - env.t0))
    res["U_swing"] = U_swing
    res.update({f"field_{k_}": v for k_, v in getattr(controller, "last", {}).items()})
    return res


def fit_density(F, mu, sd, n_components=24, seed=0):
    """Gaussian mixture of the standardised oracle features (for the out-of-distribution penalty of the MPC).
    Returns (weights, means, precisions_cholesky, c0) with c0 = 95th percentile of the negative log-density on the data."""
    from sklearn.mixture import GaussianMixture
    Z = (F - mu) / sd
    gm = GaussianMixture(n_components=n_components, covariance_type="full", reg_covar=1e-3, random_state=seed, max_iter=200).fit(Z)
    nll = -gm.score_samples(Z)
    return dict(w=gm.weights_.tolist(), means=gm.means_.tolist(), prec_chol=gm.precisions_cholesky_.tolist(), c0=float(np.quantile(nll, 0.95)),
                nll_median=float(np.median(nll)))
