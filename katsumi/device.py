"""Device (cliff A / cliff B) kinematics.

Coordinates follow the plan (Katsumi_Cliff_Dimension.pdf, Sec. 2-3):
    +x : from A towards B          (horizontal)
    +y : vertically upward
    z  : along the ledge long axis (not used in the planar model)

Tip points
    p_A(t) = (0,     h(t))      A moves vertically
    p_B(t) = (x(t),  0   )      B moves horizontally, its grip surface is y = 0

Specified end-point conditions (Table in Sec. 1)
    x(0) = 1.80, x(T) = 2.70, x(2T) = 1.80   [m]
    h(0) = 0.90, h(T) = 0.00, h(2T) = 0.90   [m]
    root-to-root horizontal distance = x(t) + 2 d,  d = 0.030 m

Two progress functions rho(phi) in [0, 1] are provided
    * triangle wave (eq. 2)      -- exact position reference, velocity jumps at turnarounds
    * smoothed wave  (eq. 4)     -- finite acceleration at turnarounds, parameter eps

Both are provided as numpy and CasADi implementations that share the same formulas.
"""
from __future__ import annotations

import numpy as np

try:  # CasADi is optional for the pure-numpy users of this module
    import casadi as ca
except Exception:  # pragma: no cover
    ca = None

X0, XT = 1.80, 2.70          # tip-to-tip horizontal distance at phi = 0 and phi = 1
H0 = 0.90                    # height difference at phi = 0
AMP = XT - X0                # 0.90 m amplitude of both motions
D_LEDGE = 0.030              # ledge depth (projection normal to the face) [m]
LEDGE_HEIGHT = 0.050         # vertical height of the ledge [m]            (Cliff_Front_Section_Corrected.tex)
FACE_BELOW = 0.150           # face from ledge underside to body lower edge [m]
WALL_BOTTOM_OFFSET = LEDGE_HEIGHT + FACE_BELOW   # 0.20 m below the grip surface the body ends; below is free space
EDGE_RADIUS_DEFAULT = 0.005  # fillet radius on the tip upper edge (user request: slightly rounded, 5 mm default)


# ----------------------------------------------------------------------------
# progress function rho and its time derivatives
# ----------------------------------------------------------------------------
def _r(q):
    return 3 * q ** 2 - 2 * q ** 3


def _r_int(q):
    # integral_0^q r(a) da
    return q ** 3 - 0.5 * q ** 4


def _dr(q):
    return 6 * q - 6 * q ** 2


def rho_triangle(u, T):
    """Triangle wave progress, eq. (2). u = t mod 2T. Returns rho, drho/dt, d2rho/dt2 (=0)."""
    phi = u / T
    outbound = phi <= 1.0
    rho = np.where(outbound, phi, 2 - phi)
    drho = np.where(outbound, 1.0 / T, -1.0 / T)
    return rho, drho, np.zeros_like(rho)


def _rho_half_smooth_np(u, T, eps):
    """Smoothed outbound half period, 0 <= u <= T (eq. 4). Returns rho, rho_dot, rho_ddot."""
    k = 1.0 / (T - eps)
    s1 = u / eps
    s2 = (T - u) / eps
    seg1 = u < eps
    seg3 = u > T - eps
    rho = np.where(seg1, k * eps * _r_int(s1),
                   np.where(seg3, 1.0 - k * eps * _r_int(s2), k * (u - 0.5 * eps)))
    drho = np.where(seg1, k * _r(s1), np.where(seg3, k * _r(s2), k))
    ddrho = np.where(seg1, k * _dr(s1) / eps, np.where(seg3, -k * _dr(s2) / eps, 0.0))
    return rho, drho, ddrho


def rho_smooth(u, T, eps):
    """Smoothed progress over the full period 0 <= u < 2T (outbound then return)."""
    u = np.asarray(u, dtype=float)
    outbound = u <= T
    ub = np.where(outbound, u, 2 * T - u)
    rho, drho, ddrho = _rho_half_smooth_np(ub, T, eps)
    sign = np.where(outbound, 1.0, -1.0)
    return rho, sign * drho, ddrho  # d/dt of rho(2T-u) = -rho'(2T-u); second derivative sign cancels


def device_state(t, T, eps=0.20, smooth=True):
    """Return dict with positions/velocities/accelerations of tips A and B at absolute time(s) t.

    x  : tip-to-tip horizontal distance,   h : height of A above B's grip surface
    """
    t = np.asarray(t, dtype=float)
    u = np.mod(t, 2 * T)
    if smooth:
        rho, drho, ddrho = rho_smooth(u, T, eps)
    else:
        rho, drho, ddrho = rho_triangle(u, T)
    x = X0 + AMP * rho
    h = H0 * (1.0 - rho)
    return dict(
        phi=u / T, rho=rho,
        x=x, xd=AMP * drho, xdd=AMP * ddrho,
        h=h, hd=-H0 * drho, hdd=-H0 * ddrho,
        pA=np.stack([np.zeros_like(x), h], -1), vA=np.stack([np.zeros_like(x), -H0 * drho], -1),
        aA=np.stack([np.zeros_like(x), -H0 * ddrho], -1),
        pB=np.stack([x, np.zeros_like(x)], -1), vB=np.stack([AMP * drho, np.zeros_like(x)], -1),
        aB=np.stack([AMP * ddrho, np.zeros_like(x)], -1),
    )


# ----------------------------------------------------------------------------
# CasADi versions (symbolic time), same formulas
# ----------------------------------------------------------------------------
def rho_smooth_ca(t, T, eps):
    """CasADi symbolic smoothed progress. Returns (rho, rho_dot, rho_ddot) as functions of absolute time t."""
    assert ca is not None
    u = ca.fmod(t, 2 * T)
    outbound = u <= T
    ub = ca.if_else(outbound, u, 2 * T - u)
    k = 1.0 / (T - eps)
    s1 = ub / eps
    s2 = (T - ub) / eps
    seg1 = ub < eps
    seg3 = ub > T - eps
    rho = ca.if_else(seg1, k * eps * _r_int(s1),
                     ca.if_else(seg3, 1.0 - k * eps * _r_int(s2), k * (ub - 0.5 * eps)))
    drho = ca.if_else(seg1, k * _r(s1), ca.if_else(seg3, k * _r(s2), k))
    ddrho = ca.if_else(seg1, k * _dr(s1) / eps, ca.if_else(seg3, -k * _dr(s2) / eps, 0.0))
    sign = ca.if_else(outbound, 1.0, -1.0)
    return rho, sign * drho, ddrho


def device_ca(t, T, eps):
    """CasADi: positions / velocities / accelerations of the tips as functions of absolute time."""
    rho, drho, ddrho = rho_smooth_ca(t, T, eps)
    x = X0 + AMP * rho
    h = H0 * (1.0 - rho)
    pA = ca.vertcat(0.0, h)
    vA = ca.vertcat(0.0, -H0 * drho)
    aA = ca.vertcat(0.0, -H0 * ddrho)
    pB = ca.vertcat(x, 0.0)
    vB = ca.vertcat(AMP * drho, 0.0)
    aB = ca.vertcat(AMP * ddrho, 0.0)
    return dict(x=x, h=h, pA=pA, vA=vA, aA=aA, pB=pB, vB=vB, aB=aB, rho=rho, drho=drho)
