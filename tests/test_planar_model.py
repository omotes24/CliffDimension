"""Unit tests for the device trajectory and the planar chain dynamics (plan Sec. 13.3 analogues)."""
import numpy as np
import casadi as ca
import pytest

from katsumi import device
from katsumi.planar.anthro import make_body, G
from katsumi.planar.model import PlanarChain


# ------------------------------------------------------------------ device -------------
@pytest.mark.parametrize("T", [16.0, 19.0, 20.0, 24.0])
@pytest.mark.parametrize("eps", [0.10, 0.20, 0.40])
def test_device_endpoints_and_period(T, eps):
    """T is the full period: x(0)=1.80, x(T/2)=2.70, x(T)=1.80."""
    for smooth in (True, False):
        s0 = device.device_state(0.0, T, eps, smooth)
        sT = device.device_state(0.5 * T, T, eps, smooth)
        s2 = device.device_state(T, T, eps, smooth)
        assert abs(s0["x"] - 1.80) < 1e-12 and abs(s0["h"] - 0.90) < 1e-12
        assert abs(sT["x"] - 2.70) < 1e-9 and abs(sT["h"] - 0.0) < 1e-9
        assert abs(s2["x"] - 1.80) < 1e-9 and abs(s2["h"] - 0.90) < 1e-9
        t = np.linspace(0, 3 * T, 4001)
        a = device.device_state(t, T, eps, smooth)
        b = device.device_state(t + T, T, eps, smooth)
        assert np.allclose(a["x"], b["x"], atol=1e-9)
        # root-to-root distance = tip-to-tip + 2d
        assert np.allclose(a["x"] + 2 * device.D_LEDGE, a["x"] + 0.060)


def test_smooth_profile_is_c1_and_matches_numeric_derivative():
    T, eps = 20.0, 0.2
    t = np.linspace(0, T, 200001)
    s = device.device_state(t, T, eps, True)
    dt = t[1] - t[0]
    xd_num = np.gradient(s["x"], dt)
    assert np.max(np.abs(xd_num[5:-5] - s["xd"][5:-5])) < 1e-4
    xdd_num = np.gradient(s["xd"], dt)
    assert np.max(np.abs(xdd_num[5:-5] - s["xdd"][5:-5])) < 1e-2
    # end-point speed is zero, mid-travel speed is 0.9/(T/2-eps)
    assert abs(s["xd"][0]) < 1e-12
    assert abs(np.max(s["xd"]) - 0.9 / (0.5 * T - eps)) < 1e-9


def test_casadi_device_matches_numpy():
    T, eps = 19.0, 0.2
    t = ca.SX.sym("t")
    d = device.device_ca(t, T, eps)
    f = ca.Function("f", [t], [d["x"], d["h"], d["vB"][0], d["vA"][1], d["aB"][0]])
    for tt in np.linspace(0.0, 2 * T, 97):
        x, h, vb, va, ab = [float(v) for v in f(tt)]
        s = device.device_state(tt, T, eps, True)
        assert abs(x - s["x"]) < 1e-12 and abs(h - s["h"]) < 1e-12
        assert abs(vb - s["xd"]) < 1e-12 and abs(va - s["hd"]) < 1e-12 and abs(ab - s["xdd"]) < 1e-10


# ------------------------------------------------------------------ dynamics -----------
def _rk4(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def test_static_hang_force_equals_weight():
    body = make_body(70.0)
    chain = PlanarChain(body)
    q = np.zeros(7)
    qd = np.zeros(7)
    thdd, R = chain.pinned(q, qd, np.zeros(4), np.array([0.0, 0.0]))
    assert np.allclose(thdd, 0, atol=1e-9)
    assert np.allclose(R, [0.0, 70.0 * G], atol=1e-9)
    # accelerating pivot upward adds m*a
    thdd, R = chain.pinned(q, qd, np.zeros(4), np.array([0.0, 0.5]))
    assert np.allclose(R, [0.0, 70.0 * (G + 0.5)], atol=1e-9)


def test_free_flight_com_is_ballistic_and_angular_momentum_conserved():
    body = make_body(66.0)
    chain = PlanarChain(body)
    rng = np.random.default_rng(0)
    q = np.concatenate([[0.3, 1.2], rng.uniform(-0.8, 0.8, 5)])
    qd = np.concatenate([[2.4, 1.0], rng.uniform(-3, 3, 5)])
    tau_fun = lambda t: 80.0 * np.sin(np.array([3.0, 5.0, 2.0, 4.0]) * t + np.arange(4))  # internal torques only

    def f(t):
        def rhs(x):
            return np.concatenate([x[7:], chain.qdd_free(x[:7], x[7:], tau_fun(t))])
        return rhs

    x = np.concatenate([q, qd])
    dt = 1e-4
    t = 0.0
    G0 = chain.com(q)
    V0 = np.array(chain.f_comd(q, qd)).ravel()
    L0 = float(chain.f_Lg(q, qd))
    for _ in range(10000):  # 1 s
        x = _rk4(f(t), x, dt)
        t += dt
    Gt = chain.com(x[:7])
    Ga = G0 + V0 * t + 0.5 * np.array([0.0, -G]) * t ** 2
    assert np.linalg.norm(Gt - Ga) < 1e-3, ("CoM error [m]", np.linalg.norm(Gt - Ga))   # <= 1 mm / 1 s
    Lt = float(chain.f_Lg(x[:7], x[7:]))
    assert abs(Lt - L0) <= 1e-3 * max(abs(L0), 1.0), (L0, Lt)


def test_free_flight_energy_conserved_without_torque():
    body = make_body(66.0)
    chain = PlanarChain(body)
    q = np.array([0.0, 1.0, 0.3, -0.2, 0.1, -0.4, 0.5])
    qd = np.array([1.0, 2.0, 1.0, -2.0, 0.5, 3.0, -1.0])
    x = np.concatenate([q, qd])
    rhs = lambda x: np.concatenate([x[7:], chain.qdd_free(x[:7], x[7:], np.zeros(4))])
    E0 = sum(float(v) for v in chain.f_energy(q, qd))
    for _ in range(5000):
        x = _rk4(rhs, x, 1e-4)
    E1 = sum(float(v) for v in chain.f_energy(x[:7], x[7:]))
    assert abs(E1 - E0) < 1e-6 * abs(E0) + 1e-6


def test_pinned_dynamics_consistent_with_free_dynamics():
    """Pinned solution must equal free dynamics driven by the reported constraint force."""
    body = make_body(75.0)
    chain = PlanarChain(body)
    rng = np.random.default_rng(1)
    q = np.concatenate([[0.0, 0.9], rng.uniform(-0.5, 0.5, 5)])
    qd = np.concatenate([[0.0, -0.09], rng.uniform(-2, 2, 5)])
    tau = rng.uniform(-50, 50, 4)
    ah = np.array([0.0, -0.3])
    thdd, R = chain.pinned(q, qd, tau, ah)
    qdd = chain.qdd_free(q, qd, tau, R)
    assert np.allclose(qdd[:2], ah, atol=1e-9)
    assert np.allclose(qdd[2:], thdd, atol=1e-9)


def test_pendulum_period_small_amplitude():
    """Rigid straight chain swinging as a compound pendulum about the fixed hand."""
    body = make_body(66.0)
    chain = PlanarChain(body)
    # lock joints by huge PD (emulate rigid body), small amplitude
    th0 = 0.02
    q = np.concatenate([[0.0, 0.0], th0 * np.ones(5)])
    qd = np.zeros(7)
    x = np.concatenate([q, qd])
    kp, kd = 2e5, 2e3
    def rhs(x):
        rel = np.array(chain.f_rel(x[:7])).ravel()
        reld = chain.B.T @ x[7:]
        tau = -kp * rel - kd * reld
        thdd, _ = chain.pinned(x[:7], x[7:], tau, np.zeros(2))
        return np.concatenate([np.zeros(2), x[9:], np.zeros(2), thdd])
    dt = 2e-5
    t = 0.0
    prev = th0
    crossings = []
    for _ in range(int(3.0 / dt)):
        x = _rk4(rhs, x, dt)
        t += dt
        if prev > 0 and x[2] <= 0:
            crossings.append(t)
        prev = x[2]
    # compound pendulum: T = 2 pi sqrt(I_O / (m g l_com))
    P = chain.points(q)
    I_O = 0.0
    lcom = 0.0
    for i, l in enumerate(body.links):
        c = P[:, i] + l.com * np.array([np.sin(th0), -np.cos(th0)])
        r2 = np.sum(c ** 2)
        I_O += l.inertia + l.mass * r2
        lcom += l.mass * np.sqrt(r2)
    lcom /= body.m
    Tp = 2 * np.pi * np.sqrt(I_O / (body.m * G * lcom))
    assert len(crossings) >= 2
    assert abs((crossings[1] - crossings[0]) - Tp) < 0.01 * Tp
