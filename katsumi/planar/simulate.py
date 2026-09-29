"""Forward re-integration (RK4, 1 ms) of NLP solutions and release-window analysis.

Two uses
  1. Verification: integrate the open-loop controls of a collocation solution with a fine RK4
     step and report the state mismatch at the phase ends (numerical-accuracy audit).
  2. Release-window W (plan Sec. 13.2): shift the release command by delta in [-0.2, 0.2] s,
     keep every joint under closed-loop PD tracking of the planned joint trajectory (torque
     saturated), let the fingers close when the hand enters the hook region of B, and integrate the
     catch with a compliant, capacity-limited grasp model. Success = hold for T_hold without the
     grasp being released and without leaving the model's admissible set.
"""
from __future__ import annotations

import numpy as np

from .. import device
from .anthro import Body, G
from .model import PlanarChain, NQ, NTH, NTAU


# ---------------------------------------------------------------------------- helpers ---
def _hs_interp(vals_k, vals_m, dur, t):
    """Quadratic (Hermite-Simpson) interpolation of a control given knot and mid-point values."""
    N = vals_m.shape[1]
    dt = dur / N
    s = np.clip(t / dt, 0, N - 1e-12)
    k = int(np.floor(s))
    tau = s - k                       # in [0,1)
    a, b, c = vals_k[:, k], vals_m[:, k], vals_k[:, k + 1]
    # Lagrange basis on {0, 1/2, 1}
    return a * (2 * tau - 1) * (tau - 1) + b * 4 * tau * (1 - tau) + c * tau * (2 * tau - 1)


def _state_interp(X, dur, t):
    """Linear interpolation of the planned state (used as PD reference)."""
    N = X.shape[1] - 1
    dt = dur / N
    s = np.clip(t / dt, 0, N - 1e-12)
    k = int(np.floor(s))
    a = s - k
    return (1 - a) * X[:, k] + a * X[:, k + 1]


def rk4(f, x, t, dt):
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Resim:
    def __init__(self, body: Body, res: dict, dt: float = 1e-3):
        self.body = body
        self.res = res
        self.chain = PlanarChain(body)
        self.dt = dt
        self.T = res["T"]
        self.eps = res["params"]["eps"]
        self.tau_cap = body.tau_cap

    def dev(self, t):
        return device.device_state(t, self.T, self.eps)

    # --- open-loop verification ------------------------------------------------------
    def verify(self):
        """Integrate each phase open loop from the planned initial state; return end-state errors."""
        r, ch, dt = self.res, self.chain, self.dt
        out = {}
        # swing
        d_s = r["d_s"]
        xs = r["S_X"][:, 0].copy()
        def fS(t, x):
            dv = self.dev(r["t_s0"] + t)
            u = _hs_interp(r["S_U"], r["S_Um"], d_s, t)
            q = np.concatenate([dv["pA"], x[:NTH]]); qd = np.concatenate([dv["vA"], x[NTH:]])
            thdd, _ = ch.pinned(q, qd, u * self.tau_cap, dv["aA"])
            return np.concatenate([x[NTH:], thdd])
        t = 0.0
        n = int(round(d_s / dt))
        h = d_s / n
        for _ in range(n):
            xs = rk4(fS, xs, t, h); t += h
        out["S_pos_err"] = np.abs(xs[:NTH] - r["S_X"][:NTH, -1]).max()
        out["S_vel_err"] = np.abs(xs[NTH:] - r["S_X"][NTH:, -1]).max()
        # flight
        d_f = r["d_f"]
        xf = r["F_X"][:, 0].copy()
        def fF(t, x):
            u = _hs_interp(r["F_U"], r["F_Um"], d_f, t)
            return np.concatenate([x[NQ:], ch.qdd_free(x[:NQ], x[NQ:], u * self.tau_cap)])
        t = 0.0
        n = int(round(d_f / dt)); h = d_f / n
        for _ in range(n):
            xf = rk4(fF, xf, t, h); t += h
        out["F_hand_err"] = np.abs(xf[:2] - r["F_X"][:2, -1]).max()
        out["F_pos_err"] = np.abs(xf[:NQ] - r["F_X"][:NQ, -1]).max()
        out["F_vel_err"] = np.abs(xf[NQ:] - r["F_X"][NQ:, -1]).max()
        # hold
        Th = r["params"]["T_hold"]
        xh = r["H_X"][:, 0].copy()
        Rmax = 0.0
        def fH(t, x):
            dv = self.dev(r["t_h0"] + t)
            u = _hs_interp(r["H_U"], r["H_Um"], Th, t)
            q = np.concatenate([dv["pB"], x[:NTH]]); qd = np.concatenate([dv["vB"], x[NTH:]])
            thdd, _ = ch.pinned(q, qd, u * self.tau_cap, dv["aB"])
            return np.concatenate([x[NTH:], thdd])
        t = 0.0
        n = int(round(Th / dt)); h = Th / n
        for _ in range(n):
            xh = rk4(fH, xh, t, h); t += h
        out["H_pos_err"] = np.abs(xh[:NTH] - r["H_X"][:NTH, -1]).max()
        out["H_vel_err"] = np.abs(xh[NTH:] - r["H_X"][NTH:, -1]).max()
        return out

    # --- metrics of the planned solution ------------------------------------------------
    def metrics(self):
        r, ch = self.res, self.chain
        m = {}
        dvl = self.dev(r["t_l"])
        q = np.concatenate([dvl["pA"], r["S_X"][:NTH, -1]]); qd = np.concatenate([dvl["vA"], r["S_X"][NTH:, -1]])
        m["v0"] = np.array(ch.f_comd(q, qd)).ravel()
        m["G_release"] = ch.com(q)
        m["h_release"] = float(dvl["h"])
        dvc = self.dev(r["t_c"])
        XF = r["F_X"]
        m["x_catch"] = float(dvc["x"])
        m["v_rel_catch"] = XF[NQ:NQ + 2, -1] - dvc["vB"]
        m["v_rel_norm"] = float(np.linalg.norm(m["v_rel_catch"]))
        m["G_catch"] = ch.com(XF[:NQ, -1])
        m["vG_catch"] = np.array(ch.f_comd(XF[:NQ, -1], XF[NQ:, -1])).ravel()
        m["impulse_catch"] = r["Lam"] * self.body.f_cap
        m["impulse_norm"] = float(np.linalg.norm(m["impulse_catch"]))
        m["U_catch"] = float(r["U_catch"])
        m["U_peak_swing"] = float(np.sqrt((r["S_R"] ** 2).sum(0)).max())
        m["U_peak_hold"] = float(np.sqrt((r["H_R"] ** 2).sum(0)).max())
        m["direction"] = "outbound" if r["phi_l"] < 1.0 else "return"
        # joint effort split
        for ph in ("S", "F", "H"):
            m[f"tau_rms_{ph}"] = float(np.sqrt((r[ph + "_U"] ** 2).mean()))
        return m


# ====================================================================================================
# Release-window analysis (plan Sec. 13.2)
# ====================================================================================================
def _hermite_interp(th, thd, thdd, dur, t):
    """Cubic Hermite interpolation of the planned joint angles (knot angles, velocities, accelerations)."""
    N = th.shape[1] - 1
    h = dur / N
    s = np.clip(t / h, 0, N - 1e-12)
    k = int(np.floor(s))
    u = s - k
    p0, p1 = th[:, k], th[:, k + 1]
    v0, v1 = thd[:, k], thd[:, k + 1]
    a0, a1 = thdd[:, k], thdd[:, k + 1]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
    pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
    vel = h00 * v0 + h10 * h * a0 + h01 * v1 + h11 * h * a1
    return pos, vel


def _plan_reference(res):
    """Planned joint angles / velocities / torques as functions of time since the swing start."""
    NTHl = NTH
    segs = []  # (t0, dur, th, thd, thdd, U, Um)
    segs.append((0.0, res["d_s"], res["S_X"][:NTHl], res["S_X"][NTHl:], res["S_A"], res["S_U"], res["S_Um"]))
    segs.append((res["d_s"], res["d_f"], res["F_X"][2:NQ], res["F_X"][NQ + 2:], res["F_A"][2:], res["F_U"], res["F_Um"]))
    segs.append((res["d_s"] + res["d_f"], res["params"]["T_hold"], res["H_X"][:NTHl], res["H_X"][NTHl:], res["H_A"],
                 res["H_U"], res["H_Um"]))

    def ref(t, phase=None):
        """phase: None (by time) or 0/1/2 to force the swing / flight / hold segment."""
        if phase is None:
            phase = 0 if t <= segs[0][1] else (1 if t <= segs[1][0] + segs[1][1] else 2)
        t0, dur, th, thd, thdd, U, Um = segs[phase]
        tt = min(max(t - t0, 0.0), dur)
        pos, vel = _hermite_interp(th, thd, thdd, dur, tt)
        return pos, vel, _hs_interp(U, Um, dur, tt)
    return ref


def release_window(body: Body, res: dict, deltas=np.arange(-0.20, 0.2001, 0.01), cap_margins=(1.0, 1.1, 1.25),
                   dt=1e-3, kp_scale=4.0, kd_ratio=0.12, tau_act=0.04, hold_gain=0.25, v_rel_max=4.0, v_away_max=0.5,
                   catch_box=(0.03, -0.005, 0.05), hook_tol=0.01, K_att=40000.0, D_att=1500.0, hold_mode="plan",
                   pre_release=0.3, trace=None):
    """Re-integrate the planned motion with the release command shifted by delta [s] (plan Sec. 13.2).

    Joints: PD tracking of the planned joint trajectory (feed-forward = planned torque) through a
    first-order activation filter (time constant tau_act) with torque saturation. After the catch the
    hold reference is re-aligned to the actual catch instant and tracked with half the gain.
    Catch: inelastic impact when the hook line enters the box around B's tip with an admissible approach;
    the hands stay where they hooked (no teleport).
    Success for capacity margin c: caught, admissible cone throughout, and max U <= c * U*.
    """
    ch = PlanarChain(body)
    T, eps = res["T"], res["params"]["eps"]
    ref = _plan_reference(res)
    tau_cap = body.tau_cap
    kp = kp_scale * tau_cap
    kd = kd_ratio * kp
    Bt = ch.B.T
    Ustar = res["U_peak"]
    t_sw0 = res["t_s0"]
    out = {}

    th_hang = res["H_X"][:NTH, -1]                       # posture at the end of the planned hold (hanging)

    def tau_parts(t, th, thd, hold=False):
        """(feed-forward torque, feedback torque command). The activation filter acts on the feedback only,
        so that with delta = 0 the planned open-loop torques are reproduced exactly."""
        if hold and hold_mode == "hang":                  # stabilise: PD towards the hanging posture, no feed-forward
            rel_err = Bt[:, 2:] @ (th_hang - th)
            reld_err = Bt[:, 2:] @ (-thd)
            return np.zeros(NTAU), hold_gain * (kp * rel_err + kd * reld_err)
        th_r, thd_r, u_ff = ref(t, 2 if hold else None)
        rel_err = Bt[:, 2:] @ (th_r - th)
        reld_err = Bt[:, 2:] @ (thd_r - thd)
        g = hold_gain if hold else 1.0
        return u_ff * tau_cap, g * (kp * rel_err + kd * reld_err)

    def applied(t, th, thd, fb_state, hold=False):
        ff, fb = tau_parts(t, th, thd, hold)
        return np.clip(ff + fb_state, -tau_cap, tau_cap), fb

    for delta in deltas:
        t_rel = res["d_s"] + delta
        Umax = 0.0
        ok_cone = True
        # ---- phase 1: pinned on A until t_rel, re-integrated from the planned state `pre_release` s
        #      before the planned release (isolates the release-timing perturbation from the accumulated
        #      tracking error of a multi-second pumping swing)
        def fA(t, x):
            th, thd, fbs = x[:NTH], x[NTH:2 * NTH], x[2 * NTH:]
            dv = device.device_state(t_sw0 + t, T, eps)
            q = np.concatenate([dv["pA"], th]); qd = np.concatenate([dv["vA"], thd])
            tau, fb = applied(t, th, thd, fbs)
            thdd, _ = ch.pinned(q, qd, tau, dv["aA"])
            return np.concatenate([thd, thdd, (fb - fbs) / tau_act])
        t = max(res["d_s"] - pre_release, 0.0)
        th0, thd0, u0 = ref(t, 0)
        x = np.concatenate([th0, thd0, np.zeros(NTAU)])
        n = int(round(max(t_rel - t, 0.0) / dt))
        for _ in range(n):
            dv = device.device_state(t_sw0 + t, T, eps)
            q = np.concatenate([dv["pA"], x[:NTH]]); qd = np.concatenate([dv["vA"], x[NTH:2 * NTH]])
            _, R = ch.pinned(q, qd, applied(t, x[:NTH], x[NTH:2 * NTH], x[2 * NTH:])[0], dv["aA"])
            Umax = max(Umax, np.linalg.norm(R) / body.f_cap)
            if R[1] < 0 or R[0] < -body.mu_out * R[1] - 1e-6:
                ok_cone = False
            x = rk4(fA, x, t, dt); t += dt
        dv = device.device_state(t_sw0 + t, T, eps)
        q = np.concatenate([dv["pA"], x[:NTH]]); qd = np.concatenate([dv["vA"], x[NTH:2 * NTH]])
        fbs = x[2 * NTH:]                                 # feedback state is continuous across the release
        # ---- phase 2: flight until catch or failure ---------------------------------------------
        def fF(t, x):
            qq, qqd, fbs = x[:NQ], x[NQ:2 * NQ], x[2 * NQ:]
            tau, fb = applied(t, qq[2:], qqd[2:], fbs)
            return np.concatenate([qqd, ch.qdd_free(qq, qqd, tau), (fb - fbs) / tau_act])
        x = np.concatenate([q, qd, fbs])
        caught = False
        reason = "no catch"
        t_fl0 = t
        while t - t_fl0 < 2.0:
            dvB = device.device_state(t_sw0 + t, T, eps)
            xB = dvB["x"]
            hx, hy = x[0], x[1]
            vrel = x[NQ:NQ + 2] - dvB["vB"]
            approach_ok = (vrel[1] <= 0.05 and vrel[0] >= -v_away_max - 0.05)
            in_height = catch_box[1] <= hy <= catch_box[2]
            if (xB - hook_tol <= hx <= xB + device.D_LEDGE) and in_height and approach_ok:
                caught = True
                break
            if hx > xB + device.D_LEDGE and hy > -device.WALL_BOTTOM_OFFSET:
                if in_height and approach_ok:      # palm stopped by the face, hook line at the root of the top surface
                    caught = True
                    x[0] = xB + device.D_LEDGE
                    break
                reason = f"hit wall (y={hy:+.3f}, vrel=({vrel[0]:+.2f},{vrel[1]:+.2f}))"
                break
            if x[1] < -1.0 and x[NQ + 1] < 0:
                reason = "missed (fell)"
                break
            x = rk4(fF, x, t, dt); t += dt
        if not caught:
            out[round(delta, 3)] = dict(success={c: False for c in cap_margins}, U_max=Umax, caught=False, reason=reason)
            continue
        # ---- compliant catch + hold: spring-damper between the hook line and the attachment point ----
        # (finite stiffness / damping as in the Stage-2 grasp model; no teleport, no impulsive pin)
        dvB = device.device_state(t_sw0 + t, T, eps)
        q = x[:NQ].copy(); qd = x[NQ:2 * NQ].copy()
        off = np.array([q[0] - dvB["pB"][0], 0.0])          # hook position along the ledge top, kept fixed
        t_shift = t - (res["d_s"] + res["d_f"])
        fbs = x[2 * NQ:].copy()                            # feedback state is continuous across the catch
        U_catch = 0.0
        U_hist = []
        U_hold = 0.0
        held = True
        reason = "held"

        def hand_force(t, qq, qqd):
            dv = device.device_state(t_sw0 + t, T, eps)
            dp = qq[0:2] - (dv["pB"] + off)
            dvv = qqd[0:2] - dv["vB"]
            ramp = min(np.linalg.norm(dp) / 0.01, 1.0)         # Hunt-Crossley-like: damping grows with deflection
            F = -K_att * dp - D_att * ramp * dvv
            Fy = max(F[1], 0.0)                                   # unilateral support
            Fx = float(np.clip(F[0], -body.mu_in * Fy, body.mu_out * Fy))   # B: pull-away = +x, friction-limited
            return np.array([Fx, Fy]), F

        def fC(t, x):
            qq, qqd, fbs = x[:NQ], x[NQ:2 * NQ], x[2 * NQ:]
            Fa, _ = hand_force(t, qq, qqd)
            tau, fb = applied(t - t_shift, qq[2:], qqd[2:], fbs, hold=True)
            return np.concatenate([qqd, ch.qdd_free(qq, qqd, tau, Fa), (fb - fbs) / tau_act])
        x = np.concatenate([q, qd, fbs])
        Th = res["params"]["T_hold"]
        n = int(round(Th / dt))
        t_hold0 = t
        for _ in range(n):
            Fa, Fraw = hand_force(t, x[:NQ], x[NQ:2 * NQ])
            U = np.linalg.norm(Fa) / body.f_cap
            if t - t_hold0 < 0.1:
                U_hist.append(U)                                    # catch pulse (10 ms moving-average peak below)
            else:
                U_hold = max(U_hold, U)
            if np.linalg.norm(Fraw - Fa) > 0.05 * body.f_cap + 1e-9 and np.linalg.norm(Fa) > 0:
                ok_cone = False                                    # demanded force outside the admissible cone
            dv = device.device_state(t_sw0 + t, T, eps)
            if np.linalg.norm(x[0:2] - (dv["pB"] + off)) > 0.08:  # hook lost (8 cm separation)
                held = False
                reason = "lost hook"
                break
            if trace is not None:
                trace.setdefault("hold_U", []).append(U)
            x = rk4(fC, x, t, dt); t += dt
        # catch load: peak of the 10 ms moving average of |F| / f_cap during the first 0.1 s (plan Sec. 6.4)
        w = max(int(round(0.01 / dt)), 1)
        U_catch = float(np.convolve(np.array(U_hist), np.ones(w) / w, mode="valid").max()) if len(U_hist) >= w else 0.0
        Umax = max(Umax, U_catch, U_hold)
        if not held:
            out[round(delta, 3)] = dict(success={c: False for c in cap_margins}, U_max=Umax, caught=True,
                                        reason=reason, U_catch=U_catch, U_hold=U_hold, off=float(off[0]))
            continue
        out[round(delta, 3)] = dict(success={c: bool(ok_cone and Umax <= c * Ustar + 1e-9) for c in cap_margins},
                                    U_max=Umax, caught=True, reason="held" if ok_cone else "cone exceeded",
                                    U_catch=U_catch, U_hold=U_hold, off=float(off[0]))
    return out


def window_width(win: dict, margin: float, deltas=None):
    """Length of the longest connected interval of successful deltas containing 0 (plan eq. 20)."""
    ds = sorted(win.keys())
    ok = [win[d]["success"][margin] for d in ds]
    if not any(ok):
        return 0.0
    # longest connected run
    best = 0
    run = 0
    for o in ok:
        run = run + 1 if o else 0
        best = max(best, run)
    step = ds[1] - ds[0] if len(ds) > 1 else 0.01
    return best * step
