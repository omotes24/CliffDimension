"""Programmatic MuJoCo model for the Cliff Dimension task (Stage 2).

World frame (MuJoCo, z-up)      plan frame (Katsumi_Cliff_Dimension.pdf)
    x  : from A towards B       =  x
    y  : along the ledges       =  z (lateral)
    z  : up                     =  y

Humanoid: 33 actuated hinge DoF as in plan Sec. 5.1
    trunk 3 (abdomen x/y/z) + neck 2, per arm: shoulder 3, elbow 1, forearm pronation 1, wrist 2,
    per leg: hip 3, knee 1, ankle 2.  The finger grasp is modelled outside the XML (grasp.py) and
    commanded through two extra action channels (g_L, g_R).
Segment masses / inertias follow de Leva (1996) male values scaled to the requested body mass m;
segment lengths are for a 1.75 m stature (scaled by `stature`).

Cliffs A and B are mocap bodies (kinematic, driven from device.py). Each carries
    * the 3 cm x 5 cm ledge (long axis y, width W), with the tip upper edge rounded by `edge_radius`
      (user request: slightly rounded so it does not dig into the hand), implemented as a cylinder
    * the front face 15 cm below the ledge underside (the plate that limits wrist motion)
    * a wall above the ledge
Below the face the space is open (the body may swing under the cliff), as in the section drawing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .. import device

# de Leva 1996 (male): mass fraction, length [m] at 1.741 m, CoM fraction from proximal, r_sag, r_trans, r_long
DELEVA = {
    "head":      (0.0694, 0.2429, 0.5002, 0.303, 0.315, 0.261),
    "upper_trunk": (0.1596, 0.1707, 0.2999, 0.505, 0.320, 0.465),
    "mid_trunk": (0.1633, 0.2155, 0.4502, 0.482, 0.383, 0.468),
    "lower_trunk": (0.1117, 0.1457, 0.6115, 0.615, 0.551, 0.587),
    "upper_arm": (0.0271, 0.2817, 0.5772, 0.285, 0.269, 0.158),
    "forearm":   (0.0162, 0.2689, 0.4574, 0.276, 0.265, 0.121),
    "hand":      (0.0061, 0.0862, 0.7900, 0.628, 0.513, 0.401),
    "thigh":     (0.1416, 0.4222, 0.4095, 0.329, 0.329, 0.149),
    "shank":     (0.0433, 0.4340, 0.4459, 0.255, 0.249, 0.103),
    "foot":      (0.0137, 0.2581, 0.4415, 0.257, 0.245, 0.124),
}
STATURE_REF = 1.741


@dataclass
class ModelParams:
    m: float = 66.0                 # body mass [kg]
    stature: float = 1.75           # [m]
    cap_scale: float = 1.0          # joint torque capacity scale
    kp_scale: float = 1.0           # PD stiffness scale
    timestep: float = 0.001
    ledge_width: float = 1.20       # along y [m] (to be measured; independent parameter)
    edge_radius: float = device.EDGE_RADIUS_DEFAULT
    wall_height_above: float = 0.80
    face_thickness: float = 0.05
    friction: float = 0.8
    # joint torque capacities [N m] (per joint, one side)
    tau_cap: dict = field(default_factory=lambda: dict(
        abdomen_x=150, abdomen_y=250, abdomen_z=120, neck_y=30, neck_z=30,
        shoulder_flex=90, shoulder_abd=80, shoulder_rot=40, elbow=70, pronation=15,
        wrist_flex=20, wrist_dev=15, hip_flex=200, hip_abd=150, hip_rot=80, knee=180,
        ankle_pf=120, ankle_inv=40))
    # PD gains [N m / rad] and damping ratio-ish [N m s / rad]
    kp: dict = field(default_factory=lambda: dict(
        abdomen_x=300, abdomen_y=400, abdomen_z=250, neck_y=40, neck_z=40,
        shoulder_flex=200, shoulder_abd=180, shoulder_rot=80, elbow=150, pronation=30,
        wrist_flex=40, wrist_dev=30, hip_flex=400, hip_abd=300, hip_rot=150, knee=350,
        ankle_pf=200, ankle_inv=80))
    kv_ratio: float = 0.08          # kv = kv_ratio * kp  [s]

    def scale_len(self, L):
        return L * self.stature / STATURE_REF


def _inertial(p: ModelParams, seg: str, direction: np.ndarray, extra_mass_frac=0.0, com_override=None):
    """Return (mass, com_xyz, diag_inertia_xyz) for a segment whose long axis is `direction`
    (unit vector from proximal to distal in the body frame) with the proximal joint at the body origin."""
    frac, L0, cfrac, rs, rt, rl = DELEVA[seg]
    L = p.scale_len(L0)
    mass = (frac + extra_mass_frac) * p.m
    com = direction * cfrac * L if com_override is None else np.asarray(com_override)
    # inertia about CoM in a frame with the long axis = local z; MuJoCo needs fullinertia or diag+quat.
    Il = mass * (rl * L) ** 2
    Is = mass * (rs * L) ** 2
    It = mass * (rt * L) ** 2
    return mass, com, (Is, It, Il), L


def _quat_from_z(direction):
    """Quaternion (w,x,y,z) rotating local +z onto `direction`."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-9:
        return (1, 0, 0, 0) if c > 0 else (0, 1, 0, 0)
    s = np.sqrt((1 + c) * 2)
    q = np.array([s / 2, v[0] / s, v[1] / s, v[2] / s])
    return tuple(q / np.linalg.norm(q))


def _fmt(v):
    return " ".join(f"{x:.5g}" for x in np.atleast_1d(v))


def _inertial_xml(mass, com, diag, direction):
    q = _quat_from_z(direction)
    return (f'<inertial pos="{_fmt(com)}" quat="{_fmt(q)}" mass="{mass:.4f}" '
            f'diaginertia="{diag[0]:.5g} {diag[1]:.5g} {diag[2]:.5g}"/>')


def _capsule(name, a, b, r, cls=None, extra=""):
    c = f' class="{cls}"' if cls else ""
    return f'<geom name="{name}" type="capsule" fromto="{_fmt(a)} {_fmt(b)}" size="{r:.4g}"{c} {extra}/>'


def _joint(name, axis, rng_deg, kind):
    lo, hi = rng_deg
    return (f'<joint name="{name}" type="hinge" axis="{_fmt(axis)}" range="{lo} {hi}" '
            f'class="{kind}"/>')


def build_xml(p: ModelParams | None = None) -> str:
    p = p or ModelParams()
    S = p.scale_len
    # --- segment geometry ----------------------------------------------------------------------
    L_ua, L_fa, L_hand = S(0.2817), S(0.2689), S(0.0862)
    L_th, L_sh, L_ft = S(0.4222), S(0.4340), S(0.2581)
    L_lt, L_mt, L_ut, L_hd = S(0.1457), S(0.2155), S(0.1707), S(0.2429)
    hip_half = S(0.09)          # half hip width
    sh_half = S(0.19)           # half shoulder width
    grip_from_wrist = S(0.095)  # wrist centre -> PIP hook line (along the hand axis)
    grip_palm_offset = S(0.045) # hook line lies on the palm side (local -x: palm faces backward in the zero pose,
                                # i.e. forward/towards the wall once the arm is flexed overhead)

    down = np.array([0, 0, -1.0])
    up = np.array([0, 0, 1.0])
    fwd = np.array([1.0, 0, 0])

    acts = []
    joints_meta = []

    def J(name, axis, rng, cap_key):
        joints_meta.append((name, cap_key))
        return _joint(name, axis, rng, "hinge")

    def act(name, cap_key):
        acts.append((name, cap_key))

    # ---- lower trunk / pelvis (root) ----------------------------------------------------------
    mass, com, diag, _ = _inertial(p, "lower_trunk", up, com_override=[-0.01, 0, 0.5 * L_lt * 0.6])
    pelvis_inertial = _inertial_xml(mass, com, diag, up)
    # ---- torso (mid + upper trunk) ------------------------------------------------------------
    m_mt, com_mt, d_mt, _ = _inertial(p, "mid_trunk", up)
    m_ut, com_ut, d_ut, _ = _inertial(p, "upper_trunk", up)
    com_ut = com_ut + np.array([0, 0, L_mt])
    m_t = m_mt + m_ut
    com_t = (m_mt * com_mt + m_ut * com_ut) / m_t
    # combine inertias about the common CoM (parallel axis along z only, both aligned with z)
    def shift(diag, dz):
        return (diag[0] + dz ** 2, diag[1] + dz ** 2, diag[2])
    Imt = np.array(d_mt) + np.array(shift((0, 0, 0), 0)) * 0
    d_t = tuple(np.array(d_mt) * 1 + 0)  # placeholder to keep structure simple
    Ixx = d_mt[0] + m_mt * (com_mt[2] - com_t[2]) ** 2 + d_ut[0] + m_ut * (com_ut[2] - com_t[2]) ** 2
    Iyy = d_mt[1] + m_mt * (com_mt[2] - com_t[2]) ** 2 + d_ut[1] + m_ut * (com_ut[2] - com_t[2]) ** 2
    Izz = d_mt[2] + d_ut[2]
    torso_inertial = _inertial_xml(m_t, com_t, (Ixx, Iyy, Izz), up)
    torso_len = L_mt + L_ut
    # ---- head ---------------------------------------------------------------------------------
    m_h, com_h, d_h, _ = _inertial(p, "head", up)
    com_h = up * (1 - 0.5002) * L_hd  # CoM measured from the vertex in de Leva; from the neck: 50% of length
    head_inertial = _inertial_xml(m_h, com_h, d_h, up)

    def arm(side):
        sgn = 1.0 if side == "left" else -1.0
        y0 = sgn * sh_half
        m_ua, com_ua, d_ua, _ = _inertial(p, "upper_arm", down)
        m_fa, com_fa, d_fa, _ = _inertial(p, "forearm", down)
        m_hn, com_hn, d_hn, _ = _inertial(p, "hand", down)
        abd_axis = np.array([-sgn, 0, 0])   # positive = away from the body
        rot_axis = np.array([0, 0, -sgn])   # positive = external rotation
        xml = f'''
      <body name="upper_arm_{side}" pos="0 {y0:.4f} {torso_len - 0.02:.4f}">
        {J(f"shoulder_flex_{side}", [0, -1, 0], (-60, 180), "shoulder_flex")}
        {J(f"shoulder_abd_{side}", abd_axis, (-40, 180), "shoulder_abd")}
        {J(f"shoulder_rot_{side}", rot_axis, (-90, 90), "shoulder_rot")}
        {_inertial_xml(m_ua, com_ua, d_ua, down)}
        {_capsule(f"upper_arm_{side}", [0, 0, -0.03], [0, 0, -L_ua + 0.03], 0.040, "body")}
        <body name="forearm_{side}" pos="0 0 {-L_ua:.4f}">
          {J(f"elbow_{side}", [0, -1, 0], (0, 150), "elbow")}
          {J(f"pronation_{side}", [0, 0, -1], (-80, 80), "pronation")}
          {_inertial_xml(m_fa, com_fa, d_fa, down)}
          {_capsule(f"forearm_{side}", [0, 0, -0.02], [0, 0, -L_fa + 0.07], 0.031, "body")}
          {_capsule(f"wrist_{side}", [0, 0, -L_fa + 0.07], [0, 0, -L_fa + 0.01], 0.022, "body")}
          <body name="hand_{side}" pos="0 0 {-L_fa:.4f}">
            {J(f"wrist_flex_{side}", [0, -1, 0], (-70, 70), "wrist_flex")}
            {J(f"wrist_dev_{side}", [sgn, 0, 0], (-25, 30), "wrist_dev")}
            {_inertial_xml(m_hn, com_hn, d_hn, down)}
            {_capsule(f"hand_{side}", [0, 0, -0.01], [0, 0, -L_hand], 0.030, "hand")}
            <site name="grip_{side}" pos="{-grip_palm_offset:.4f} 0 {-grip_from_wrist:.4f}" size="0.008" rgba="1 0 0 1"/>
            <site name="palm_{side}" pos="-0.02 0 {-0.5 * grip_from_wrist:.4f}" size="0.006" rgba="0 1 0 1" zaxis="-1 0 0"/>
          </body>
        </body>
      </body>'''
        return xml

    def leg(side):
        sgn = 1.0 if side == "left" else -1.0
        y0 = sgn * hip_half
        m_th, com_th, d_th, _ = _inertial(p, "thigh", down)
        m_sh, com_sh, d_sh, _ = _inertial(p, "shank", down)
        m_ft, com_ft, d_ft, _ = _inertial(p, "foot", fwd, com_override=[0.4415 * L_ft - 0.05, 0, -0.03])
        abd_axis = np.array([-sgn, 0, 0])
        return f'''
      <body name="thigh_{side}" pos="0 {y0:.4f} 0">
        {J(f"hip_flex_{side}", [0, -1, 0], (-25, 125), "hip_flex")}
        {J(f"hip_abd_{side}", abd_axis, (-30, 45), "hip_abd")}
        {J(f"hip_rot_{side}", [0, 0, -sgn], (-40, 40), "hip_rot")}
        {_inertial_xml(m_th, com_th, d_th, down)}
        {_capsule(f"thigh_{side}", [0, 0, -0.04], [0, 0, -L_th + 0.04], 0.065, "body")}
        <body name="shin_{side}" pos="0 0 {-L_th:.4f}">
          {J(f"knee_{side}", [0, 1, 0], (0, 150), "knee")}
          {_inertial_xml(m_sh, com_sh, d_sh, down)}
          {_capsule(f"shin_{side}", [0, 0, -0.03], [0, 0, -L_sh + 0.03], 0.045, "body")}
          <body name="foot_{side}" pos="0 0 {-L_sh:.4f}">
            {J(f"ankle_pf_{side}", [0, 1, 0], (-30, 50), "ankle_pf")}
            {J(f"ankle_inv_{side}", [sgn, 0, 0], (-30, 30), "ankle_inv")}
            {_inertial_xml(m_ft, com_ft, d_ft, fwd)}
            <geom name="foot_{side}" type="box" pos="{0.4 * L_ft - 0.05:.4f} 0 -0.03" size="{0.5 * L_ft:.4f} 0.045 0.02" class="body"/>
          </body>
        </body>
      </body>'''

    # actuators: one position servo per hinge
    for name, key in joints_meta:
        pass  # collected below after body definition (joints_meta is filled while formatting)

    body_xml = f'''
    <body name="pelvis" pos="0 0 1.0" childclass="body">
      <freejoint name="root"/>
      {pelvis_inertial}
      <geom name="pelvis" type="capsule" fromto="0 {-hip_half:.4f} 0 0 {hip_half:.4f} 0" size="0.085" class="body"/>
      <site name="pelvis_site" pos="0 0 0" size="0.01"/>
      <body name="torso" pos="0 0 {L_lt * 0.6:.4f}">
        {J("abdomen_x", [1, 0, 0], (-35, 35), "abdomen_x")}
        {J("abdomen_y", [0, 1, 0], (-45, 70), "abdomen_y")}
        {J("abdomen_z", [0, 0, 1], (-45, 45), "abdomen_z")}
        {torso_inertial}
        {_capsule("torso_lower", [0, -0.07, 0.05], [0, 0.07, 0.05], 0.075, "body")}
        {_capsule("torso_upper", [0, -0.10, torso_len - 0.07], [0, 0.10, torso_len - 0.07], 0.075, "body")}
        <site name="chest" pos="0.10 0 {torso_len - 0.10:.4f}" size="0.01" zaxis="1 0 0"/>
        <body name="head" pos="0 0 {torso_len + 0.02:.4f}">
          {J("neck_y", [0, 1, 0], (-45, 45), "neck_y")}
          {J("neck_z", [0, 0, 1], (-70, 70), "neck_z")}
          {head_inertial}
          <geom name="head" type="sphere" pos="0 0 {0.55 * L_hd:.4f}" size="0.095" class="body"/>
        </body>
        {arm("right")}
        {arm("left")}
      </body>
      {leg("right")}
      {leg("left")}
    </body>'''

    act_xml = []
    for name, key in joints_meta:
        cap = p.tau_cap[key] * p.cap_scale
        kp = p.kp[key] * p.kp_scale
        kv = kp * p.kv_ratio
        act_xml.append(f'<position name="{name}" joint="{name}" kp="{kp:.4g}" kv="{kv:.4g}" '
                       f'forcerange="{-cap:.4g} {cap:.4g}" ctrlrange="-3.5 3.5"/>')

    # ---- cliffs ---------------------------------------------------------------------------------
    d = device.D_LEDGE
    hL = device.LEDGE_HEIGHT
    fb = device.FACE_BELOW
    W = p.ledge_width
    r = p.edge_radius
    th = p.face_thickness
    Hup = p.wall_height_above

    def cliff(name, facing):
        """facing = +1: A (tip points towards +x, wall on the -x side); -1: B (mirror)."""
        s = facing
        # ledge: tip at local x = 0 (tip face at x=0), root at x = -s*d. grip surface z = 0 (top), ledge z in [-hL, 0]
        x_root = -s * d
        parts = []
        # main box (behind the rounded edge)
        cx = -s * (d + r) / 2  # centre between x=-s*r and x=-s*d... compute explicitly:
        xa, xb = sorted([-s * r, -s * d])
        parts.append(f'<geom name="{name}_ledge" type="box" pos="{(xa + xb) / 2:.4f} 0 {-hL / 2:.4f}" '
                     f'size="{(xb - xa) / 2:.4f} {W / 2:.4f} {hL / 2:.4f}" class="cliff"/>')
        # rounded tip edge: cylinder along y at (x=-s*r, z=-r), radius r
        parts.append(f'<geom name="{name}_edge" type="cylinder" pos="{-s * r:.4f} 0 {-r:.4f}" '
                     f'size="{r:.4f} {W / 2:.4f}" zaxis="0 1 0" class="cliff"/>')
        # tip face below the rounding: box x in [-s*r, 0], z in [-hL, -r]
        xa, xb = sorted([0.0, -s * r])
        parts.append(f'<geom name="{name}_tipface" type="box" pos="{(xa + xb) / 2:.4f} 0 {-(hL + r) / 2:.4f}" '
                     f'size="{(xb - xa) / 2 + 1e-4:.4f} {W / 2:.4f} {(hL - r) / 2:.4f}" class="cliff"/>')
        # front face plate below the ledge: x in [root - s*th, root], z in [-(hL+fb), -hL]
        xa, xb = sorted([x_root, x_root - s * th])
        parts.append(f'<geom name="{name}_face" type="box" pos="{(xa + xb) / 2:.4f} 0 {-(hL + fb / 2):.4f}" '
                     f'size="{(xb - xa) / 2:.4f} {W / 2:.4f} {fb / 2:.4f}" class="cliff"/>')
        # wall above the ledge
        parts.append(f'<geom name="{name}_wall" type="box" pos="{(xa + xb) / 2:.4f} 0 {Hup / 2:.4f}" '
                     f'size="{(xb - xa) / 2:.4f} {W / 2:.4f} {Hup / 2:.4f}" class="cliff"/>')
        parts.append(f'<site name="{name}_tip" pos="0 0 0" size="0.01" rgba="0 0 1 1"/>')
        inner = "\n        ".join(parts)
        return f'''
    <body name="{name}" mocap="true" pos="0 0 0">
        {inner}
    </body>'''

    xml = f'''<mujoco model="cliff_dimension">
  <compiler angle="degree" inertiafromgeom="false" autolimits="true"/>
  <option timestep="{p.timestep}" gravity="0 0 -9.81" integrator="RK4" cone="elliptic" impratio="10"/>
  <default>
    <default class="body">
      <geom type="capsule" condim="3" friction="{p.friction} 0.005 0.0001" solimp="0.9 0.99 0.001" solref="0.008 1"
            rgba="0.8 0.6 0.4 1" group="1" contype="1" conaffinity="2"/>
      <joint type="hinge" damping="0.5" armature="0.005" limited="true" solimplimit="0 0.99 0.01"/>
    </default>
    <default class="hand">
      <geom type="capsule" contype="0" conaffinity="0" rgba="0.9 0.7 0.5 1" group="1"/>
    </default>
    <default class="hinge">
      <joint type="hinge" damping="0.5" armature="0.005" limited="true" solimplimit="0 0.99 0.01"/>
    </default>
    <default class="cliff">
      <geom condim="3" friction="{p.friction} 0.005 0.0001" solimp="0.95 0.99 0.001" solref="0.005 1"
            rgba="0.75 0.78 0.80 1" contype="2" conaffinity="1"/>
    </default>
  </default>
  <worldbody>
    <light pos="0 -3 4" dir="0 0.5 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="water" type="plane" pos="0 0 -4.0" size="6 3 0.1" rgba="0.3 0.5 0.8 0.3" contype="0" conaffinity="0"/>
    {cliff("cliffA", +1)}
    {cliff("cliffB", -1)}
    {body_xml}
  </worldbody>
  <actuator>
    {chr(10).join("    " + a for a in act_xml)}
  </actuator>
  <equality>
    <connect name="grasp_left_A" body1="hand_left" body2="cliffA" anchor="0 0 0" active="false" solref="0.008 1"/>
    <connect name="grasp_left_B" body1="hand_left" body2="cliffB" anchor="0 0 0" active="false" solref="0.008 1"/>
    <connect name="grasp_right_A" body1="hand_right" body2="cliffA" anchor="0 0 0" active="false" solref="0.008 1"/>
    <connect name="grasp_right_B" body1="hand_right" body2="cliffB" anchor="0 0 0" active="false" solref="0.008 1"/>
  </equality>
</mujoco>
'''
    return xml


JOINT_ORDER = None  # filled lazily by env from the compiled model


if __name__ == "__main__":
    import mujoco
    xml = build_xml(ModelParams())
    m = mujoco.MjModel.from_xml_string(xml)
    print("nq", m.nq, "nv", m.nv, "nu", m.nu, "nbody", m.nbody)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print("total mass", sum(m.body_mass), "subtree mass pelvis", m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')])
