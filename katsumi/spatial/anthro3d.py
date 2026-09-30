"""Anthropometry for the spatial (twisting) reduced model.

Seven rigid bodies: trunk+head, left / right upper arm, left / right forearm+hand, thigh pair, shank+foot pair.
Segment masses, CoM positions and the three radii of gyration (sagittal, transverse, longitudinal) follow
de Leva (1996, adult male) and are scaled so that the total mass equals m; lengths scale with stature/1.75
(arm segments additionally by arm_scale). Leg pairs are lumped into one body each whose inertia includes the
lateral separation of the two legs (this is most of the body's inertia about its long axis).

Capabilities are given PER SIDE for the arms (shoulder flexion / abduction, elbow) and for the PAIR for the
legs (hip, knee); the grip capacity is PER HAND (f_hand, default 650 N = half of the planar 1300 N).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..planar.anthro import G, STATURE_REF


@dataclass
class Seg:
    name: str
    mass: float
    com: np.ndarray        # CoM in the segment frame [m] (frame origin = proximal joint)
    inertia: np.ndarray    # 3x3 inertia about the CoM in the segment frame [kg m^2]
    length: float          # proximal -> distal joint [m]


@dataclass
class Body3D:
    m: float
    stature: float
    segs: dict
    shoulder_width: float          # biacromial distance [m]
    leg_sep: float                 # lateral distance between the two legs' axes [m]
    grip_offset: float             # wrist centre -> hook line [m]
    tau_cap: np.ndarray = field(default_factory=lambda: np.array([80.0, 60.0, 60.0, 80.0, 60.0, 60.0, 300.0, 240.0]))
    #                       (shoulder flex L, abd L, elbow L, shoulder flex R, abd R, elbow R, hip pair, knee pair) [N m]
    qd_max: float = 10.0
    f_hand: float = 650.0          # grip capacity per hand [N]
    mu_out: float = 1.0
    mu_in: float = 2.0
    mu_lat: float = 0.8            # friction along the ledge
    wrist_dev_max: float = np.deg2rad(35.0)   # max angle between the forearm and the plane normal to the ledge
    head_top: float = 0.30         # shoulder line -> top of head [m]

    @property
    def mtot(self):
        return sum(s.mass for s in self.segs.values())


def _rod_inertia(mass, L, k_sag, k_tr, k_long):
    """Diagonal inertia about the CoM in a frame whose 3rd axis is the segment's long axis."""
    return np.diag([mass * (k_sag * L) ** 2, mass * (k_tr * L) ** 2, mass * (k_long * L) ** 2])


def make_body3d(m: float, stature: float = STATURE_REF, arm_scale: float = 1.0, cap_scale: float = 1.0,
                grip_scale: float = 1.0, mu_out: float | None = None, grip_offset: float = 0.12,
                cap_with_size: bool = False) -> Body3D:
    ks = stature / STATURE_REF
    ka = ks * arm_scale
    # de Leva 1996 (male): mass fraction, length for 1.75 m, CoM fraction (from proximal), k_sag, k_trans, k_long
    D = dict(
        hand=(0.0061, 0.0862 * ka, 0.7900, 0.628, 0.513, 0.401),
        fa=(0.0162, 0.2689 * ka, 0.4574, 0.276, 0.265, 0.121),
        ua=(0.0271, 0.2817 * ka, 0.5772, 0.285, 0.269, 0.158),
        tr=(0.4346, 0.5319 * ks, 0.4486, 0.372, 0.347, 0.191),
        hd=(0.0694, 0.2429 * ks, 0.5976, 0.362, 0.376, 0.312),
        th=(0.1416, 0.4222 * ks, 0.4095, 0.329, 0.329, 0.149),
        sh=(0.0433, 0.4340 * ks, 0.4459, 0.255, 0.249, 0.103),
        ft=(0.0137, 0.2581 * ks, 0.4415, 0.257, 0.245, 0.124),
    )
    total_frac = 2 * (D["hand"][0] + D["fa"][0] + D["ua"][0] + D["th"][0] + D["sh"][0] + D["ft"][0]) + D["tr"][0] + D["hd"][0]
    scale = m / total_frac
    grip_offset = grip_offset * ka
    segs = {}

    def one(frac, L, c, ks_, kt_, kl_):
        mass = frac * scale
        return mass, c * L, _rod_inertia(mass, L, ks_, kt_, kl_)

    # ---- upper arm (per side): frame origin = shoulder, long axis = 3rd axis pointing towards the elbow
    mu_, cu, Iu = one(*D["ua"])
    for side in ("L", "R"):
        segs["ua_" + side] = Seg("upper arm " + side, mu_, np.array([0, 0, cu]), Iu, D["ua"][1])
    # ---- forearm + hand (per side): origin = elbow, 3rd axis towards the hook line (length L_fa + grip_offset)
    mf, cf, If = one(*D["fa"])
    mh, ch_, Ih = one(*D["hand"])
    L_fa = D["fa"][1]
    L1 = L_fa + grip_offset
    c_hand = L_fa + ch_                           # hand CoM measured from the elbow (wrist at L_fa)
    mfh = mf + mh
    cfh = (mf * cf + mh * c_hand) / mfh
    Ifh = If + Ih + mf * np.diag([(cf - cfh) ** 2, (cf - cfh) ** 2, 0]) + mh * np.diag([(c_hand - cfh) ** 2, (c_hand - cfh) ** 2, 0])
    for side in ("L", "R"):
        segs["fa_" + side] = Seg("forearm+hand " + side, mfh, np.array([0, 0, cfh]), Ifh, L1)
    # ---- trunk + head: origin = shoulder-line centre, 3rd axis UP (towards the head); hip at -L_tr
    mt, ct, It = one(*D["tr"])
    mhd, chd, Ihd = one(*D["hd"])
    L_tr = 0.50 * ks                             # shoulder line -> hip joint centre
    z_tr = -(ct - 0.05 * ks)                     # trunk CoM below the shoulder line (cervicale ~5 cm above)
    z_hd = 0.05 * ks + (1 - chd) * D["hd"][1]    # head CoM above the shoulder line
    mth = mt + mhd
    zc = (mt * z_tr + mhd * z_hd) / mth
    Ith = It + Ihd + mt * np.diag([(z_tr - zc) ** 2, (z_tr - zc) ** 2, 0]) + mhd * np.diag([(z_hd - zc) ** 2, (z_hd - zc) ** 2, 0])
    segs["trunk"] = Seg("trunk+head", mth, np.array([0, 0, zc]), Ith, L_tr)
    # ---- thigh pair: origin = hip centre, 3rd axis towards the knee; two rods separated laterally
    leg_sep = 0.18 * ks
    mth1, cth, Ith1 = one(*D["th"])
    Ipair = 2 * Ith1 + 2 * mth1 * np.diag([(leg_sep / 2) ** 2, 0, (leg_sep / 2) ** 2])   # separation along the 2nd (lateral) axis
    segs["thigh"] = Seg("thigh pair", 2 * mth1, np.array([0, 0, cth]), Ipair, D["th"][1])
    # ---- shank + foot pair: origin = knee, 3rd axis towards the ankle; feet hanging beyond the ankle
    msh, csh, Ish = one(*D["sh"])
    mft, cft, Ift = one(*D["ft"])
    L_sh = D["sh"][1]
    c_ft = L_sh + 0.08 * ks
    msf = msh + mft
    csf = (msh * csh + mft * c_ft) / msf
    Isf = Ish + Ift + msh * np.diag([(csh - csf) ** 2, (csh - csf) ** 2, 0]) + mft * np.diag([(c_ft - csf) ** 2, (c_ft - csf) ** 2, 0])
    Ipair = 2 * Isf + 2 * msf * np.diag([(leg_sep / 2) ** 2, 0, (leg_sep / 2) ** 2])
    segs["shank"] = Seg("shank+foot pair", 2 * msf, np.array([0, 0, csf]), Ipair, L_sh)

    body = Body3D(m=m, stature=stature, segs=segs, shoulder_width=0.39 * ks, leg_sep=leg_sep, grip_offset=grip_offset)
    if cap_with_size:
        sf = ks ** 2 * (m / 66.0)
        cap_scale, grip_scale = cap_scale * sf, grip_scale * sf
    body.tau_cap = body.tau_cap * cap_scale
    body.f_hand = body.f_hand * grip_scale
    body.head_top = 0.30 * ks
    if mu_out is not None:
        body.mu_out = mu_out
    assert abs(body.mtot - m) < 1e-9, body.mtot
    return body


if __name__ == "__main__":
    b = make_body3d(66.0)
    for k, s in b.segs.items():
        print(f"{k:8s} m={s.mass:6.2f} L={s.length:.3f} com={s.com.round(3)} I=diag{np.diag(s.inertia).round(4)}")
    print("total", b.mtot)
