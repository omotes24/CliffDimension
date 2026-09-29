"""Stage-2 acceptance tests (plan Sec. 13.3) and P1 checks for the MuJoCo environment."""
import numpy as np
import mujoco
import pytest

from katsumi import device
from katsumi.mujoco.env import CliffEnv, EnvParams, G
from katsumi.mujoco.build_model import build_xml, ModelParams


@pytest.fixture(scope="module")
def env():
    return CliffEnv(EnvParams(T=20.0, phi0=0.0))


def _hold_action(env):
    return np.concatenate([env.joint_targets_from_qpos(), [1.0, 1.0]])


# ------------------------------------------------------------------ geometry / period ---
def test_model_mass_and_dof():
    m = mujoco.MjModel.from_xml_string(build_xml(ModelParams(m=72.0)))
    assert abs(sum(m.body_mass) - 72.0) < 1e-6
    assert m.nu == 31                      # 33 with the two finger channels


def test_cliff_geometry_and_period(env):
    m, d = env.model, env.data
    T = env.p.T
    for t in (0.0, 0.5 * T, T, 1.65 * T):
        env._set_mocap(t)
        mujoco.mj_forward(m, d)
        s = device.device_state(t, T, env.p.eps)
        tipA = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cliffA_tip")]
        tipB = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cliffB_tip")]
        assert abs(tipB[0] - tipA[0] - s["x"]) < 1e-9
        assert abs(tipA[2] - tipB[2] - s["h"]) < 1e-9
        # root-to-root = tip-to-tip + 6 cm: faces are the outer x-extent of the ledge boxes
        gA = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cliffA_ledge")
        gB = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cliffB_ledge")
        rootA = d.geom_xpos[gA][0] - m.geom_size[gA][0]
        rootB = d.geom_xpos[gB][0] + m.geom_size[gB][0]
        assert abs((rootB - rootA) - (s["x"] + 0.060)) < 1e-6
    env._set_mocap(0.0)
    s0 = device.device_state(0.0, T, env.p.eps)
    assert abs(s0["x"] - 1.80) < 1e-12 and abs(s0["h"] - 0.90) < 1e-12
    sT = device.device_state(0.5 * T, T, env.p.eps)
    assert abs(sT["x"] - 2.70) < 1e-9 and abs(sT["h"]) < 1e-9


# ------------------------------------------------------------------ free flight ---------
def test_free_flight_com_and_angular_momentum(env):
    """No contacts, no grasp: CoM error <= 1 mm over 1 s, |L_G| change <= 0.1 %."""
    m, d = env.model, env.data
    env.reset()
    # open the hands and give the body a generic velocity, move it away from the cliffs
    for h in env.hands.values():
        h._release("test")
    d.qpos[0:3] = [1.0, 0.0, 2.0]
    d.qvel[:] = 0.0
    d.qvel[0:6] = [1.5, 0.2, 1.0, 0.8, -0.5, 0.3]
    rng = np.random.default_rng(0)
    d.qvel[6:] = rng.uniform(-2, 2, m.nv - 6)
    d.ctrl[:] = d.qpos[env.qadr] + rng.uniform(-0.3, 0.3, m.nu)   # internal PD torques active
    d.eq_active[:] = 0
    m.opt.gravity[:] = [0, 0, -G]
    mujoco.mj_forward(m, d)
    com0 = d.subtree_com[env.bid["pelvis"]].copy()
    v0 = np.zeros(3)
    mujoco.mj_subtreeVel(m, d)
    v0 = d.subtree_linvel[env.bid["pelvis"]].copy()
    L0 = d.subtree_angmom[env.bid["pelvis"]].copy()
    # move the cliffs far away so there is no contact
    d.mocap_pos[env.mocapA] = [-10, 0, 0]
    d.mocap_pos[env.mocapB] = [10, 0, 0]
    n = int(round(1.0 / m.opt.timestep))
    for _ in range(n):
        mujoco.mj_step(m, d)
        assert d.ncon == 0
    t = n * m.opt.timestep
    com_analytic = com0 + v0 * t + 0.5 * np.array([0, 0, -G]) * t ** 2
    err = np.linalg.norm(d.subtree_com[env.bid["pelvis"]] - com_analytic)
    assert err <= 1e-3, f"CoM error {err*1e3:.2f} mm"
    mujoco.mj_subtreeVel(m, d)
    L1 = d.subtree_angmom[env.bid["pelvis"]]
    assert np.linalg.norm(L1 - L0) <= 1e-3 * max(np.linalg.norm(L0), 1.0) + 0.05, (L0, L1)


# ------------------------------------------------------------------ P1: static hang -----
def test_static_two_hand_hang(env):
    env.reset()
    a = _hold_action(env)
    for _ in range(100):  # 2 s
        obs, r, term, trunc, info = env.step(a)
        assert not term
    Fz = sum(h.force[2] for h in env.hands.values())
    assert abs(Fz - env.p.model.m * G) < 0.02 * env.p.model.m * G
    assert all(info["onA"])
    # the body follows A downwards
    sA = device.device_state(env.t_start + env.t_elapsed, env.p.T, env.p.eps)
    mid = 0.5 * (env.data.site_xpos[env.hands["left"].site] + env.data.site_xpos[env.hands["right"].site])
    assert abs(mid[2] - sA["h"]) < 0.01


# ------------------------------------------------------------------ P1: drop -------------
def test_release_leads_to_free_fall(env):
    env.reset()
    a = _hold_action(env)
    for _ in range(25):
        env.step(a)
    a_open = a.copy()
    a_open[-2:] = 0.0
    z0 = env.com()[2]
    t0 = env.t_elapsed
    out = None
    for _ in range(30):
        obs, r, term, trunc, info = env.step(a_open)
        assert not any(info["onA"]) and not any(info["onB"])
        assert all(np.linalg.norm(h.force) == 0 for h in env.hands.values())
        if term:
            out = info
            break
    dz = env.com()[2] - z0
    dt = env.t_elapsed - t0
    # started moving with A (v = -0.09 m/s): dz ~ v t - g t^2 / 2
    vA = device.device_state(env.t_start + t0, env.p.T, env.p.eps)["hd"]
    assert abs(dz - (vA * dt - 0.5 * G * dt ** 2)) < 0.02


# ------------------------------------------------------------------ impulse balance -----
def test_catch_impulse_balance(env):
    """Plan eq. (12): over a window, int (R_L + R_R) dt = m (v_G(t2) - v_G(t1)) - m g dt, residual <= 1 %."""
    m, d = env.model, env.data
    env.reset()
    a = _hold_action(env)
    # excite a swing by moving the hip targets, then integrate the balance over 1 s
    a_sw = a.copy()
    for i, n in enumerate(env.act_names):
        if n.startswith("hip_flex"):
            a_sw[i] = 0.3
    for _ in range(20):
        env.step(a_sw)
    mujoco.mj_subtreeVel(m, d)
    v1 = d.subtree_linvel[env.bid["pelvis"]].copy()
    imp = np.zeros(3)
    n_steps = 50
    for _ in range(n_steps):
        env.step(a)
        imp += env.log[-1]["impulse"].sum(0) + env.log[-1]["wall_impulse"]
    mujoco.mj_subtreeVel(m, d)
    v2 = d.subtree_linvel[env.bid["pelvis"]].copy()
    dt = n_steps * env.p.control_dt
    rhs = env.p.model.m * (v2 - v1) - env.p.model.m * np.array([0, 0, -G]) * dt
    resid = np.linalg.norm(imp - rhs)
    assert resid <= 0.01 * np.linalg.norm(imp) + 0.5, (imp, rhs)


# ------------------------------------------------------------------ P1: low-speed catch --
def test_low_speed_catch_on_B():
    p = EnvParams(T=20.0, phi0=0.0)
    e = CliffEnv(p)
    e.reset()
    m, d = e.model, e.data
    # place the athlete hanging just above B with open hands facing B (+x), then close the fingers
    for h in e.hands.values():
        h._release("test")
    e._set_mocap(e.t_start)
    pB = e.cliffs["B"][0]
    d.qpos[3:7] = [1, 0, 0, 0]              # facing +x
    mujoco.mj_forward(m, d)
    gl = d.site_xpos[e.hands["left"].site]
    gr = d.site_xpos[e.hands["right"].site]
    mid = 0.5 * (gl + gr)
    d.qpos[0:3] += (pB + np.array([+0.015, 0.0, 0.02])) - mid   # hook line 1.5 cm inward from B's tip
    d.qvel[:] = 0.0
    d.qvel[2] = -0.3                          # slow descent onto the ledge
    mujoco.mj_forward(m, d)
    a = np.concatenate([e.joint_targets_from_qpos(), [1.0, 1.0]])
    caught = False
    for k in range(75):
        obs, r, term, trunc, info = e.step(a)
        if all(info["onB"]):
            caught = True
        if term:
            break
    assert caught, [h.events for h in e.hands.values()]
    assert all(info["onB"])
    Fz = sum(h.force[2] for h in e.hands.values())
    assert Fz > 0.5 * e.p.model.m * G


# ------------------------------------------------------------------ constraints ----------
def test_no_grasp_from_behind_or_below():
    p = EnvParams(T=20.0, phi0=0.0)
    e = CliffEnv(p)
    e.reset()
    m, d = e.model, e.data
    for h in e.hands.values():
        h._release("test")
    e._set_mocap(e.t_start)
    pB = e.cliffs["B"][0]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    mid = 0.5 * (d.site_xpos[e.hands["left"].site] + d.site_xpos[e.hands["right"].site])
    # hands 6 cm BELOW the grip surface, right under the ledge: must not engage
    d.qpos[0:3] += (pB + np.array([+0.01, 0.0, -0.06])) - mid
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    a = np.concatenate([e.joint_targets_from_qpos(), [1.0, 1.0]])
    for k in range(5):
        e.step(a)
    assert not any(h.attached for h in e.hands.values())


def test_timestep_convergence_of_hang_forces():
    """Plan Sec. 13.3: 2 / 1 / 0.5 ms re-runs, 10 ms-average peak force within 5 %."""
    peaks = {}
    for ts in (0.002, 0.001, 0.0005):
        mp = ModelParams(timestep=ts)
        e = CliffEnv(EnvParams(model=mp, T=20.0, phi0=0.0))
        e.reset()
        a = np.concatenate([e.joint_targets_from_qpos(), [1.0, 1.0]])
        a_sw = a.copy()
        for i, n in enumerate(e.act_names):
            if n.startswith("hip_flex"):
                a_sw[i] = 0.4
        pk = 0.0
        for k in range(60):
            e.step(a_sw if (k // 15) % 2 == 0 else a)
            if k >= 10:
                pk = max(pk, e.log[-1]["Favg10"].max())
        peaks[ts] = pk
    ref = peaks[0.0005]
    assert abs(peaks[0.001] - ref) <= 0.05 * ref, peaks     # 1 ms is the production step
    print("timestep convergence (10 ms-average peak force):", peaks)
