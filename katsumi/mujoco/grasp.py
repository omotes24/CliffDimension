"""Finite-capacity two-hand grasp approximation (plan Sec. 6.1), implemented with MuJoCo soft
`connect` equality constraints (implicit, unconditionally stable) plus an explicit admissible-wrench
check after every physics step.

Each hand is represented by its hook line (site `grip_<side>`: the PIP crease that lies over the
ledge edge) and a palm normal (site `palm_<side>`, local +z).

Engagement ("掛かり") requires, once the close command g >= 0.5 has been held for `close_time`:
    * the hook line inside the hook region above the ledge top surface
      (x over the 3 cm top within `hook_tol` beyond the tip, z within [-z_below, z_above], |y| inside the width)
    * the palm facing the wall (angle <= `palm_angle_max`)
    * the hand not moving upward relative to the ledge faster than `v_up_max`
Then a soft point constraint between the hook line and the attachment point on the ledge top surface
is activated. After each step the constraint force F (on the hand, world frame) is checked against
the admissible set of the hand:
    F_z (support from the top surface)            >= 0            (unilateral; sustained pull-down = lift-off)
    F away from the wall (friction / hook)        <= mu_out * F_z
    F towards the wall (finger tips on the face)  <= mu_in  * F_z
    |F_y| (lateral friction)                      <= mu_lat * F_z
    |F|                                           <= F_cap(f, depth)  capacity (fatigue-, depth-dependent)
Excess force makes the attachment slide (viscous-plastic slip, `slip_coef`); accumulated slip beyond
`slip_max`, loss of the geometric hook (anchor beyond the tip) or lift-off release the grasp ("剥離").
No teleporting of the hand, no unlimited constraint, no suction from behind.

Fatigue (plan eq. 16):  f_dot = k_load (1 - f) U^p - k_rec f [1 - U]_+ ,  F_cap = (1 - alpha_f f) F_cap0.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mujoco

from .. import device


@dataclass
class GraspParams:
    f_cap_hand: float = 650.0      # per-hand capacity [N] (absolute; plan Sec. 5.3 main comparison)
    solref_time: float = 0.008     # constraint time constant [s] (finite stiffness)
    mu_out: float = 1.0            # pulls away from the wall
    mu_in: float = 2.0             # pulls towards the wall (finger tips against the face)
    mu_lat: float = 0.8            # lateral friction
    hook_tol: float = 0.012        # hook line may be up to this beyond the tip [m]
    z_above: float = 0.045         # hook line height window above the top surface at engagement [m]
    z_below: float = 0.020
    palm_angle_max: float = 70.0   # [deg]
    v_up_max: float = 0.5          # [m/s]
    slip_coef: float = 3000.0      # viscous-plastic slip: d(slip)/dt = excess_force / slip_coef [m/s per N]
    slip_max: float = 0.020        # [m]
    liftoff_time: float = 0.010    # sustained pull-down duration that releases the grasp [s]
    depth_full: float = 0.015      # hook depth giving full capacity [m]; shallower hooks are weaker
    close_time: float = 0.06       # finger closing latency [s]
    # fatigue
    k_load: float = 0.0
    k_rec: float = 0.0
    p_fat: float = 2.0
    alpha_f: float = 0.5


class HandGrasp:
    def __init__(self, model, data, side: str, gp: GraspParams):
        self.m, self.d, self.side, self.gp = model, data, side, gp
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"grip_{side}")
        self.palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"palm_{side}")
        self.body = model.site_bodyid[self.site]
        self.eq = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{side}_{n}") for n in ("A", "B")}
        self.ledge_halfwidth = model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cliffA_ledge")][1]
        self.site_local = model.site_pos[self.site].copy()
        for e in self.eq.values():
            model.eq_solref[e] = [gp.solref_time, 1.0]
        self.reset()

    def reset(self, fatigue0: float = 0.0):
        for e in self.eq.values():
            self.d.eq_active[e] = 0
        self.attached = None        # None | "A" | "B"
        self.p_loc = np.zeros(3)     # attachment point in the cliff frame (tip at origin)
        self.slip = 0.0
        self.depth = 0.0
        self.fatigue = fatigue0
        self.force = np.zeros(3)     # constraint force on the hand [N] (world)
        self.U = 0.0
        self.close_timer = 0.0
        self.lift_timer = 0.0
        self.last_release_reason = ""
        self.events = []

    # ------------------------------------------------------------------------------------------
    def capacity(self):
        gp = self.gp
        depth_factor = float(np.clip(self.depth / gp.depth_full, 0.3, 1.0)) if self.attached else 1.0
        return gp.f_cap_hand * (1.0 - gp.alpha_f * self.fatigue) * depth_factor

    def _site_state(self):
        p = self.d.site_xpos[self.site].copy()
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.site, v6, 0)
        v = v6[3:].copy()
        R = self.d.site_xmat[self.palm].reshape(3, 3)
        palm_n = R[:, 2].copy()
        return p, v, palm_n

    def _activate(self, name, p_loc):
        e = self.eq[name]
        self.m.eq_data[e, 0:3] = self.site_local
        self.m.eq_data[e, 3:6] = p_loc
        self.d.eq_active[e] = 1
        self.attached = name
        self.p_loc = np.array(p_loc, float)

    def pre_step(self, cliffs: dict, g_cmd: float, dt: float):
        """Engagement check (called before mj_step). cliffs: {"A": (pos, vel, facing), "B": (...)}."""
        gp = self.gp
        if self.attached is not None:
            if g_cmd < 0.5:
                self._release("open")
            return
        if g_cmd < 0.5:
            self.close_timer = 0.0
            return
        self.close_timer += dt
        if self.close_timer < gp.close_time:
            return
        p, v, palm_n = self._site_state()
        for name, (pc, vc, s) in cliffs.items():
            x, y, z = p - pc
            lo, hi = sorted([-s * device.D_LEDGE, s * gp.hook_tol])
            if not (lo - 1e-9 <= x <= hi + 1e-9):
                continue
            if not (-gp.z_below <= z <= gp.z_above):
                continue
            if abs(y) > self.ledge_halfwidth - 0.03:
                continue
            wall_dir = np.array([-s, 0.0, 0.0])
            if np.dot(palm_n, wall_dir) < np.cos(np.deg2rad(gp.palm_angle_max)):
                continue
            if (v - vc)[2] > gp.v_up_max:
                continue
            self._activate(name, [x, y, 0.0])
            self.depth = float(-s * x)
            self.slip = 0.0
            self.lift_timer = 0.0
            self.events.append(("engage", name, self.d.time, self.depth))
            return

    def post_step(self, cliffs: dict, dt: float):
        """Read the constraint force, enforce the admissible set (slip / release). Call after mj_step."""
        gp = self.gp
        self.force = np.zeros(3)
        if self.attached is None:
            self.U = 0.0
            self._fatigue_step(0.0, dt)
            return self.force
        e = self.eq[self.attached]
        d = self.d
        rows = np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == e))[0]
        F = d.efc_force[rows].copy() if len(rows) == 3 else np.zeros(3)
        pc, vc, s = cliffs[self.attached]
        # unilateral support
        if F[2] < -1.0:
            self.lift_timer += dt
            if self.lift_timer > gp.liftoff_time:
                self._release("lift-off")
                return self.force
        else:
            self.lift_timer = 0.0
        Fz = max(F[2], 0.0)
        excess = 0.0
        slip_dir = np.zeros(3)
        away = -s * F[0]                      # on the hand: force towards the wall (= resists the pull away)
        if away > gp.mu_out * Fz:
            excess = max(excess, away - gp.mu_out * Fz)
            slip_dir[0] += s                  # the hand slides away from the wall (towards +s)
        if -away > gp.mu_in * Fz:
            excess = max(excess, -away - gp.mu_in * Fz)
            slip_dir[0] -= s
        if abs(F[1]) > gp.mu_lat * Fz:
            excess = max(excess, abs(F[1]) - gp.mu_lat * Fz)
            slip_dir[1] -= np.sign(F[1])
        cap = self.capacity()
        mag = float(np.linalg.norm(F))
        if mag > cap:
            excess = max(excess, mag - cap)
            slip_dir -= F / (mag + 1e-9)
        if excess > 0:
            ds = excess / gp.slip_coef * dt
            self.slip += ds
            n = slip_dir / (np.linalg.norm(slip_dir) + 1e-9)
            self.p_loc = self.p_loc + ds * n
            self.p_loc[2] = 0.0
            self.m.eq_data[e, 3:6] = self.p_loc
            self.depth = float(-s * self.p_loc[0])
            if self.slip > gp.slip_max or self.depth < -gp.hook_tol:
                self._release("capacity/friction slip" if self.slip > gp.slip_max else "slipped off tip")
                return self.force
        self.force = F
        self.U = mag / max(cap, 1e-6)
        self._fatigue_step(self.U, dt)
        return self.force

    def _fatigue_step(self, U, dt):
        gp = self.gp
        if gp.k_load > 0 or gp.k_rec > 0:
            f = self.fatigue
            fd = gp.k_load * (1 - f) * U ** gp.p_fat - gp.k_rec * f * max(1 - U, 0.0)
            self.fatigue = float(np.clip(f + fd * dt, 0.0, 1.0))

    def _release(self, reason):
        self.events.append(("release", self.attached, self.d.time, reason))
        for e in self.eq.values():
            self.d.eq_active[e] = 0
        self.last_release_reason = reason
        self.attached = None
        self.U = 0.0
        self.force = np.zeros(3)
        self.slip = 0.0
        self.close_timer = 0.0
        self.lift_timer = 0.0

    def force_attach(self, name, cliffs, depth=0.015):
        """Used at reset: start the episode already holding cliff `name`."""
        pc, vc, s = cliffs[name]
        p, _, _ = self._site_state()
        rel = p - pc
        self._activate(name, [-s * depth, rel[1], 0.0])
        self.depth = depth
        self.slip = 0.0
        self.close_timer = self.gp.close_time
