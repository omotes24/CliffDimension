"""Cliff Dimension MuJoCo environment (Stage 2).

Control period 20 ms, physics 1 ms (plan Sec. 6.4). Actions: 31 joint targets in [-1, 1] mapped to
the joint ranges (PD position servos with torque saturation) + 2 finger close commands in [0, 1].
State / observation follow plan eq. (13); start / success / failure follow Sec. 7.3; learning reward
follows Sec. 10.2. All load quantities of Sec. 6.4 / 8.1 are logged per control step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import mujoco

from .. import device
from .build_model import build_xml, ModelParams
from .grasp import HandGrasp, GraspParams

G = 9.81


@dataclass
class EnvParams:
    model: ModelParams = field(default_factory=ModelParams)
    grasp: GraspParams = field(default_factory=GraspParams)
    T: float = 20.0                   # full device period [s] (one-way 10 s)
    eps: float = 0.20
    phi0: float = 0.0
    control_dt: float = 0.02
    hold_time: float = 2.0
    episode_extra: float = 4.0        # episode limit T + 4 s (T = full period)
    release_deadline_factor: float = 1.0
    fail_height: float = -4.0         # virtual failure plane (plan Sec. 7.3)
    chest_angle_start: float = 20.0
    chest_angle_catch: float = 30.0
    illegal_support: bool = True      # leg / trunk contact with the cliffs = failure
    # learning reward (plan Sec. 10.2)
    w_U: float = 3.0
    w_E: float = 3.0
    w_time: float = 0.01
    t_ref: float = 1.0
    shaping: float = 0.0              # optional hand-reach potential shaping weight (0 = off)
    obs_noise: float = 0.0


class CliffEnv:
    def __init__(self, p: EnvParams | None = None):
        self.p = p or EnvParams()
        self.xml = build_xml(self.p.model)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = int(round(self.p.control_dt / m.opt.timestep))
        # joints / actuators
        self.act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.act_joint = np.array([m.actuator_trnid[i, 0] for i in range(m.nu)])
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.act_joint])
        self.vadr = np.array([m.jnt_dofadr[j] for j in self.act_joint])
        self.jrange = np.array([m.jnt_range[j] for j in self.act_joint])   # degrees converted at compile -> radians
        self.tau_cap = np.array([m.actuator_forcerange[i, 1] for i in range(m.nu)])
        self.n_act = m.nu + 2
        # bodies / sites
        self.bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                    ["pelvis", "torso", "head", "cliffA", "cliffB", "hand_left", "hand_right",
                     "forearm_left", "forearm_right", "upper_arm_left", "upper_arm_right"]}
        self.mocapA = m.body_mocapid[self.bid["cliffA"]]
        self.mocapB = m.body_mocapid[self.bid["cliffB"]]
        self.chest_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "chest")
        self.hands = {"left": HandGrasp(m, self.data, "left", self.p.grasp),
                      "right": HandGrasp(m, self.data, "right", self.p.grasp)}
        cliff_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] in (self.bid["cliffA"], self.bid["cliffB"])]
        self.cliff_geoms = set(cliff_geoms)
        arm_bodies = {self.bid[k] for k in ["hand_left", "hand_right", "forearm_left", "forearm_right",
                                           "upper_arm_left", "upper_arm_right"]}
        self.illegal_geoms = {i for i in range(m.ngeom)
                              if m.geom_bodyid[i] not in arm_bodies and i not in self.cliff_geoms
                              and m.geom_contype[i] != 0}
        self.rng = np.random.default_rng(0)
        self.log = []

    # ------------------------------------------------------------------ device -------------
    def _device(self, t):
        s = device.device_state(t, self.p.T, self.p.eps)
        pA = np.array([0.0, 0.0, float(s["h"])])
        pB = np.array([float(s["x"]), 0.0, 0.0])
        vA = np.array([0.0, 0.0, float(s["hd"])])
        vB = np.array([float(s["xd"]), 0.0, 0.0])
        return pA, vA, pB, vB, float(s["phi"])

    def _set_mocap(self, t):
        pA, vA, pB, vB, phi = self._device(t)
        self.data.mocap_pos[self.mocapA] = pA
        self.data.mocap_pos[self.mocapB] = pB
        self.cliffs = {"A": (pA, vA, +1), "B": (pB, vB, -1)}
        self.phi = phi
        return pA, vA, pB, vB

    # ------------------------------------------------------------------ reset --------------
    def reset(self, T=None, m_body=None, phi0=None, seed=None, fatigue0=0.0):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if T is not None:
            self.p.T = float(T)
        if phi0 is not None:
            self.p.phi0 = float(phi0)
        if m_body is not None and abs(m_body - self.p.model.m) > 1e-9:
            self.p.model.m = float(m_body)
            self.__init__(self.p)
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        self.t_start = self.p.phi0 * self.p.T
        d.time = 0.0
        self._set_mocap(self.t_start)
        # hanging posture: arms overhead, facing -x (chest towards A)
        q = np.zeros(m.nq)
        q[3:7] = [0.0, 0.0, 0.0, 1.0]          # yaw 180 deg: quaternion (w,x,y,z) = (0,0,0,1)
        for i, name in enumerate(self.act_names):
            if name.startswith("shoulder_flex"):
                q[self.qadr[i]] = np.pi
        hip = 0.0
        for _ in range(6):                      # slight hip flexion so that the CoM hangs below the hands
            for i, name in enumerate(self.act_names):
                if name.startswith("hip_flex"):
                    q[self.qadr[i]] = hip
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            gl = d.site_xpos[self.hands["left"].site]
            gr = d.site_xpos[self.hands["right"].site]
            mid = 0.5 * (gl + gr)
            dx = self.com()[0] - mid[0]         # >0: CoM on the +x (world) side of the hands
            hip += dx / 0.18                    # hip flexion moves the legs towards body +x = world -x
        mid = 0.5 * (d.site_xpos[self.hands["left"].site] + d.site_xpos[self.hands["right"].site])
        pA = self.cliffs["A"][0]
        target = pA + np.array([-0.015, 0.0, 0.0])  # hook line 1.5 cm inward from the tip (depth)
        d.qpos[0:3] += target - mid
        mujoco.mj_forward(m, d)
        # start moving with A
        d.qvel[:] = 0.0
        d.qvel[0:3] = self.cliffs["A"][1]
        mujoco.mj_forward(m, d)
        for h in self.hands.values():
            h.reset(fatigue0)
            h.force_attach("A", self.cliffs, depth=0.015)
        self.ctrl_target = np.array([d.qpos[a] for a in self.qadr])
        self.g_cmd = np.array([1.0, 1.0])
        self.t_elapsed = 0.0
        self.U_peak = 0.0
        self.E_eff = 0.0
        self.hold_timer = 0.0
        self.released_A = False
        self.first_contact_B_time = None
        self.release_time = None
        self.done = False
        self.result = None
        self.log = []
        self.contact_flags = set()
        return self._obs()

    # ------------------------------------------------------------------ step ---------------
    def step(self, action):
        assert not self.done, "call reset()"
        a = np.asarray(action, float)
        a_j = np.clip(a[: self.model.nu], -1, 1)
        g = np.clip(a[self.model.nu:], 0, 1)
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        self.ctrl_target = lo + 0.5 * (a_j + 1) * (hi - lo)
        self.data.ctrl[:] = self.ctrl_target
        self.g_cmd = g
        dt = self.model.opt.timestep
        Fmax = np.zeros(2)
        F_hist = []
        tau_sq = 0.0
        Wp = Wn = 0.0
        impulse = np.zeros((2, 3))
        wall_impulse = np.zeros(3)
        illegal = False
        min_z = np.inf
        for k in range(self.n_sub):
            t_abs = self.t_start + self.data.time
            self._set_mocap(t_abs)
            for i, (side, h) in enumerate(self.hands.items()):
                h.pre_step(self.cliffs, g[i], dt)
            # integrator switch: RK4 in free flight (4th-order ballistic accuracy, plan Sec. 13.3),
            # implicit Euler while grasp constraints / contacts act (exact constraint-force readings)
            constrained = any(h.attached for h in self.hands.values()) or self.data.ncon > 0
            self.model.opt.integrator = (mujoco.mjtIntegrator.mjINT_IMPLICITFAST if constrained
                                         else mujoco.mjtIntegrator.mjINT_RK4)
            mujoco.mj_step(self.model, self.data)
            # efc_force / contacts left by mj_step are the forces applied during this step (exact under
            # the semi-implicit integrator); read them before any further forward pass
            wall_F = self._cliff_contact_force()
            wall_impulse += wall_F * dt
            for i, (side, h) in enumerate(self.hands.items()):
                F = h.post_step(self.cliffs, dt)
                Fn = np.linalg.norm(F)
                Fmax[i] = max(Fmax[i], Fn)
                impulse[i] += F * dt
            F_hist.append([np.linalg.norm(h.force) for h in self.hands.values()])
            # effort (plan eq. 15): joint torques / capacity
            tau = self.data.actuator_force
            tau_sq += float(np.mean((tau / self.tau_cap) ** 2)) * dt
            qd = self.data.qvel[self.vadr]
            pw = tau * qd
            Wp += float(np.sum(np.clip(pw, 0, None))) * dt
            Wn += float(np.sum(np.clip(-pw, 0, None))) * dt
            if self.p.illegal_support and self._illegal_contact():
                illegal = True
            min_z = min(min_z, float(self.data.xpos[self.bid["pelvis"], 2]))
        mujoco.mj_forward(self.model, self.data)          # fresh kinematics for the observation
        F_hist = np.array(F_hist)
        # 10 ms moving-average peak
        w = max(int(round(0.01 / dt)), 1)
        if len(F_hist) >= w:
            kern = np.ones(w) / w
            Favg = np.array([np.convolve(F_hist[:, i], kern, mode="valid").max() for i in range(2)])
        else:
            Favg = F_hist.max(0)
        U_now = np.array([h.U for h in self.hands.values()])
        caps = np.array([h.capacity() for h in self.hands.values()])
        U_step = float(np.max(Fmax / np.maximum(caps, 1e-6)))
        dU = max(U_step - self.U_peak, 0.0)
        self.U_peak += dU
        # effort increment: torque term + grasp utilisation term, integrated over the control step
        U_int = float(np.mean((F_hist / np.maximum(caps, 1e-6)) ** 2)) * self.p.control_dt if len(F_hist) else 0.0
        dE = (tau_sq + U_int) / self.p.t_ref
        self.E_eff += dE
        self.t_elapsed += self.p.control_dt
        # ---- task logic ------------------------------------------------------------------
        onA = [h.attached == "A" for h in self.hands.values()]
        onB = [h.attached == "B" for h in self.hands.values()]
        if not any(onA) and not self.released_A:
            self.released_A = True
            self.release_time = self.t_elapsed
        if any(onB) and self.first_contact_B_time is None:
            self.first_contact_B_time = self.t_elapsed
        chest = self.data.site_xmat[self.chest_site].reshape(3, 3)[:, 2]
        facing_B = chest[0] >= np.cos(np.deg2rad(self.p.chest_angle_catch))
        if all(onB) and self.released_A and facing_B:
            self.hold_timer += self.p.control_dt
        else:
            self.hold_timer = 0.0
        terminated = False
        success = False
        reason = ""
        if self.hold_timer >= self.p.hold_time - 1e-9:
            terminated, success, reason = True, True, "held B"
        elif min_z < self.p.fail_height:
            terminated, reason = True, "fell"
        elif illegal:
            terminated, reason = True, "illegal support"
        elif (not self.released_A) and self.t_elapsed > self.p.release_deadline_factor * self.p.T:
            terminated, reason = True, "release deadline"
        elif self.released_A and not any(onA) and not any(onB) and \
                self.data.xpos[self.bid["pelvis"], 2] < -1.5 and self.data.qvel[2] < 0 and \
                self.data.site_xpos[self.hands["left"].site][2] < -0.6:
            terminated, reason = True, "dropped"
        truncated = (not terminated) and self.t_elapsed >= self.p.release_deadline_factor * self.p.T + self.p.episode_extra
        if truncated:
            reason = "time limit"
        # ---- reward (plan Sec. 10.2) -----------------------------------------------------
        r = -self.p.w_U * dU - self.p.w_E * dE - self.p.w_time * self.p.control_dt / self.p.t_ref
        if terminated or truncated:
            r += 100.0 if success else -100.0
            self.done = True
            self.result = dict(success=success, reason=reason, t=self.t_elapsed, U_peak=self.U_peak,
                               E_eff=self.E_eff, release_time=self.release_time,
                               catch_time=self.first_contact_B_time)
        self.log.append(dict(t=self.t_elapsed, Fmax=Fmax.copy(), Favg10=Favg.copy(), impulse=impulse.copy(),
                             U=U_now.copy(), U_peak=self.U_peak, E=self.E_eff, Wp=Wp, Wn=Wn,
                             onA=onA, onB=onB, phi=self.phi, slip=[h.slip for h in self.hands.values()],
                             wall_impulse=wall_impulse,
                             pelvis=self.data.xpos[self.bid["pelvis"]].copy(),
                             tau=self.data.actuator_force.copy()))
        info = dict(success=success, reason=reason, U_peak=self.U_peak, E_eff=self.E_eff, onA=onA, onB=onB)
        return self._obs(), r, terminated, truncated, info

    def _cliff_contact_force(self):
        """Total contact force [N, world] exerted by the cliff bodies on the humanoid."""
        d, m = self.data, self.model
        tot = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self.cliff_geoms or c.geom2 in self.cliff_geoms:
                mujoco.mj_contactForce(m, d, i, f6)
                R = c.frame.reshape(3, 3)          # rows: normal, tangent1, tangent2
                fw = R.T @ f6[:3]                  # force on geom2 along the contact frame -> world
                # MuJoCo convention: the normal points from geom1 to geom2; force acts on geom2 (- on geom1)
                sign = -1.0 if c.geom2 in self.cliff_geoms else 1.0
                tot += sign * fw
        return tot

    def _illegal_contact(self):
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.cliff_geoms and g2 in self.illegal_geoms) or (g2 in self.cliff_geoms and g1 in self.illegal_geoms):
                return True
        return False

    # ------------------------------------------------------------------ observation --------
    def com(self):
        return self.data.subtree_com[self.bid["pelvis"]].copy()

    def _obs(self):
        d, m, p = self.data, self.model, self.p
        pA, vA, pB, vB = self.cliffs["A"][0], self.cliffs["A"][1], self.cliffs["B"][0], self.cliffs["B"][1]
        com = self.com()
        qj = d.qpos[self.qadr]
        qdj = d.qvel[self.vadr]
        gl = d.site_xpos[self.hands["left"].site]
        gr = d.site_xpos[self.hands["right"].site]
        mode = lambda h: [h.attached == "A", h.attached is None, h.attached == "B"]
        obs = np.concatenate([
            d.qpos[0:3] - pB, d.qpos[3:7], d.qvel[0:6],
            qj, qdj,
            gl - com, gr - com, com - pA, com - pB,
            pA, pB, vA, vB, [np.sin(2 * np.pi * self.phi), np.cos(2 * np.pi * self.phi), p.T, p.model.m],
            mode(self.hands["left"]), mode(self.hands["right"]),
            self.hands["left"].force / p.grasp.f_cap_hand, self.hands["right"].force / p.grasp.f_cap_hand,
            [self.hands["left"].U, self.hands["right"].U, self.hands["left"].slip, self.hands["right"].slip],
            self.g_cmd, [self.hands["left"].fatigue, self.hands["right"].fatigue],
            [self.t_elapsed, p.release_deadline_factor * p.T + p.episode_extra - self.t_elapsed, self.U_peak],
        ]).astype(np.float32)
        if p.obs_noise > 0:
            obs = obs + self.rng.normal(0, p.obs_noise, obs.shape).astype(np.float32)
        return obs

    @property
    def obs_dim(self):
        return self._obs().shape[0]

    # ------------------------------------------------------------------ helpers ------------
    def joint_targets_from_qpos(self, qpos=None):
        """Map a joint configuration to the normalised action (useful for holding the current pose)."""
        q = self.data.qpos if qpos is None else qpos
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        return np.clip(2 * (q[self.qadr] - lo) / (hi - lo) - 1, -1, 1)

    def render_frames(self, width=640, height=480, every=1, camera=None):
        """Offscreen rendering helper (requires EGL/OSMesa); returns a list of RGB arrays from the log."""
        raise NotImplementedError("use scripts/render_episode.py")
