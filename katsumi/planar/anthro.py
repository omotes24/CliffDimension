"""Anthropometric parameters for the planar 5-link chain.

Segment masses / CoM positions / radii of gyration follow de Leva (1996), adult male, and are
scaled so that the total mass equals the requested body mass m (plan Sec. 5.3: same skeleton,
link masses summed to m, inertias scaled consistently). Segment lengths are for a 1.75 m stature
and are kept fixed across m (stature variation is a separate experiment axis).

Chain (proximal = towards the hands, which are the base of the chain):
    L1 : grip point -> elbow      (both forearms + both hands, lumped)
    L2 : elbow -> shoulder        (both upper arms)
    L3 : shoulder -> hip          (trunk + head)
    L4 : hip -> knee              (both thighs)
    L5 : knee -> ankle            (both shanks + feet); clearance point at the hanging toe tip

All angles are absolute link angles measured from the downward vertical, positive towards +x.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

G = 9.81


@dataclass
class Link:
    name: str
    length: float      # joint-to-joint length [m]
    mass: float        # [kg]
    com: float         # CoM offset from the proximal end along the link [m] (may be negative)
    inertia: float     # about the CoM [kg m^2]
    tip: float         # distance from the proximal end to the clearance point used for wall checks [m]


@dataclass
class Body:
    m: float
    links: list = field(default_factory=list)
    # joint capacities (both limbs combined) [N m]; plan Sec. 5.1 / 13.1 capability scaling
    tau_cap: np.ndarray = field(default_factory=lambda: np.array([120.0, 160.0, 300.0, 240.0]))
    qd_max: float = 10.0                   # joint angular velocity limit [rad/s]
    # grasp (both hands combined)
    f_cap: float = 2 * 650.0               # total grip capacity |R| <= f_cap [N]  (absolute, not scaled with m)
    mu_out: float = 1.0                    # friction/hook coefficient against pulls away from the wall
    mu_in: float = 2.0                     # coefficient against pulls towards the wall (fingertips on the face)
    clearance: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.12, 0.12, 0.07, 0.10]))

    @property
    def lengths(self):
        return np.array([l.length for l in self.links])

    @property
    def masses(self):
        return np.array([l.mass for l in self.links])

    @property
    def coms(self):
        return np.array([l.com for l in self.links])

    @property
    def inertias(self):
        return np.array([l.inertia for l in self.links])

    @property
    def tips(self):
        return np.array([l.tip for l in self.links])


def _combine(parts):
    """parts: list of (mass, com_offset, inertia_about_own_com). Returns (mass, com, inertia)."""
    M = sum(p[0] for p in parts)
    c = sum(p[0] * p[1] for p in parts) / M
    I = sum(p[2] + p[0] * (p[1] - c) ** 2 for p in parts)
    return M, c, I


def make_body(m: float, cap_scale: float = 1.0, grip_scale: float = 1.0, grip_offset: float = 0.12) -> Body:
    """Build the lumped planar body for total mass m [kg].

    cap_scale  : multiplies joint torque capacities (plan Sec. 13.1: {0.7, 1.0, 1.3})
    grip_scale : multiplies grip capacity
    grip_offset: distance from the wrist centre to the ledge contact line on the fingers (PIP) [m]
    """
    # de Leva 1996, male: mass fraction, length [m] for 1.75 m, CoM fraction from proximal, radius of gyration (sagittal)
    f_hand, L_hand, c_hand, k_hand = 0.0061, 0.0862, 0.7900, 0.628   # length: wrist -> 3rd metacarpale
    f_fa, L_fa, c_fa, k_fa = 0.0162, 0.2689, 0.4574, 0.276
    f_ua, L_ua, c_ua, k_ua = 0.0271, 0.2817, 0.5772, 0.285
    f_tr, L_tr, c_tr, k_tr = 0.4346, 0.5319, 0.4486, 0.372
    f_hd, L_hd, c_hd, k_hd = 0.0694, 0.2429, 0.5976, 0.362
    f_th, L_th, c_th, k_th = 0.1416, 0.4222, 0.4095, 0.329
    f_sh, L_sh, c_sh, k_sh = 0.0433, 0.4340, 0.4459, 0.255
    f_ft, L_ft, c_ft, k_ft = 0.0137, 0.2581, 0.4415, 0.257
    total_frac = 2 * (f_hand + f_fa + f_ua + f_th + f_sh + f_ft) + f_tr + f_hd
    scale = m / total_frac  # forces exact total mass m (de Leva fractions sum to ~1.0)

    def seg(frac, L, k):
        mass = 2 * frac * scale
        return mass, mass * (k * L) ** 2

    # L1: grip -> elbow. hand CoM lies slightly beyond the grip line (towards fingertips): negative offset
    m_h, I_h = seg(f_hand, L_hand, k_hand)
    m_f, I_f = seg(f_fa, L_fa, k_fa)
    L1 = grip_offset + L_fa
    com_hand = c_hand * L_hand - grip_offset          # from wrist towards fingertips, relative to the grip line
    com_fa = grip_offset + (1 - c_fa) * L_fa          # forearm CoM measured from the elbow -> from the grip
    M1, C1, I1 = _combine([(m_h, -com_hand, I_h), (m_f, com_fa, I_f)])
    # L2: elbow -> shoulder
    m_u, I_u = seg(f_ua, L_ua, k_ua)
    L2 = L_ua
    M2, C2, I2 = m_u, (1 - c_ua) * L_ua, I_u
    # L3: shoulder -> hip (trunk + head). cervicale ~0.05 above the shoulder-joint level.
    L3 = 0.50
    m_t = f_tr * scale
    I_t = m_t * (k_tr * L_tr) ** 2
    m_hd = f_hd * scale
    I_hd = m_hd * (k_hd * L_hd) ** 2
    com_tr = c_tr * L_tr - 0.05
    com_hd = -(0.05 + (1 - c_hd) * L_hd)              # head CoM above the shoulder line
    M3, C3, I3 = _combine([(m_t, com_tr, I_t), (m_hd, com_hd, I_hd)])
    # L4: hip -> knee
    m_thigh, I_thigh = seg(f_th, L_th, k_th)
    L4 = L_th
    M4, C4, I4 = m_thigh, c_th * L_th, I_thigh
    # L5: knee -> ankle, feet hanging (plantar-flexed) beyond the ankle
    m_s, I_s = seg(f_sh, L_sh, k_sh)
    m_ft, I_ft = seg(f_ft, L_ft, k_ft)
    L5 = L_sh
    M5, C5, I5 = _combine([(m_s, c_sh * L_sh, I_s), (m_ft, L_sh + 0.08, I_ft)])

    links = [
        Link("forearm+hand", L1, M1, C1, I1, tip=L1),
        Link("upper arm", L2, M2, C2, I2, tip=L2),
        Link("trunk+head", L3, M3, C3, I3, tip=L3),
        Link("thigh", L4, M4, C4, I4, tip=L4),
        Link("shank+foot", L5, M5, C5, I5, tip=L5 + 0.10),
    ]
    body = Body(m=m, links=links)
    body.tau_cap = body.tau_cap * cap_scale
    body.f_cap = body.f_cap * grip_scale
    assert abs(sum(l.mass for l in links) - m) < 1e-9
    return body


if __name__ == "__main__":
    b = make_body(66.0)
    for l in b.links:
        print(f"{l.name:14s} L={l.length:.3f} m={l.mass:6.2f} c={l.com:+.3f} I={l.inertia:.4f}")
    print("total", sum(l.mass for l in b.links), "reach hand->toe", sum(b.lengths) + 0.10)
