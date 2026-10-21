"""Coupled geomechanics: does injection pressure reactivate a nearby fault?

In-situ stress (compression positive) in principal axes x = SHmax,
y = Shmin, z = vertical, with hydrostatic initial pore pressure. During
injection two things change on a fault inside the reservoir:

1. pore pressure rises by dp, lowering the effective normal stress
   sigma_n' = sigma_n - p  (the direct, destabilising effect);
2. the reservoir rock expands against lateral confinement, raising the
   total horizontal stresses by  A * dp,  A = alpha (1 - 2 nu) / (1 - nu)
   (the poroelastic stress path -- the coupling term).

Mohr-Coulomb (cohesionless) slip criterion on the fault plane:

    CFF = tau - mu_f * (sigma_n - p) >= 0   ->  reactivation

`critical_dp` solves for the smallest pressure increase that makes CFF
reach zero (first crossing, vectorised over many faults), and
`allowable_dp` also applies the hydraulic-fracturing limit.
"""

from __future__ import annotations

import numpy as np

G = 9.81
RHO_ROCK = 2450.0
RHO_WATER = 1050.0
DP_CAP = 80e6  # Pa; faults not reactivated below this are reported at the cap


def fault_normal(strike_deg, dip_deg):
    """Unit normal of a plane whose strike is measured from SHmax (x axis)."""
    th, de = np.radians(strike_deg), np.radians(dip_deg)
    s = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1)            # strike vector
    d = np.stack([-np.sin(th) * np.cos(de), np.cos(th) * np.cos(de), np.sin(de)], -1)  # dip vector
    n = np.cross(s, d)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def in_situ(depth, shmin_ratio, shmax_ratio):
    """Total principal stresses (SHmax, Shmin, Sv) and pore pressure, Pa."""
    sv = RHO_ROCK * G * depth
    p0 = RHO_WATER * G * depth
    return np.stack([shmax_ratio * sv, shmin_ratio * sv, sv], -1), p0


def stress_path_coefficient(nu, alpha=1.0):
    return alpha * (1.0 - 2.0 * nu) / (1.0 - nu)


def traction(principal, n):
    """Normal and shear stress on the plane with normal n (principal axes)."""
    t = principal * n
    sn = np.sum(t * n, -1)
    tau = np.sqrt(np.maximum(np.sum(t * t, -1) - sn ** 2, 0.0))
    return sn, tau


def cff(dp, principal, p0, n, mu_f, A):
    """Coulomb failure function after a pore-pressure increase dp."""
    S = principal + (A * dp)[..., None] * np.array([1.0, 1.0, 0.0])
    sn, tau = traction(S, n)
    return tau - mu_f * (sn - (p0 + dp))


def slip_tendency(principal, p0, n):
    sn, tau = traction(principal, n)
    return tau / (sn - p0)


def critical_dp(principal, p0, n, mu_f, A, iters: int = 50, n_scan: int = 400):
    """Smallest dp >= 0 with CFF >= 0, vectorised over faults.

    CFF is NOT monotone in dp in general: with a strong stress path the
    horizontal stress can overtake the vertical one, the resolved shear
    stress passes through zero and slip reverses sense. So the first
    crossing is bracketed by a coarse scan and then refined by bisection.
    Returns 0 for faults already critically stressed and DP_CAP for faults
    not reactivated below the cap.
    """
    principal = np.asarray(principal, float)
    N = principal.shape[0]
    p0 = np.broadcast_to(np.asarray(p0, float), (N,))
    mu_f = np.broadcast_to(np.asarray(mu_f, float), (N,))
    A = np.broadcast_to(np.asarray(A, float), (N,))
    grid = np.linspace(0.0, DP_CAP, n_scan)
    f = np.stack([cff(np.full(N, g), principal, p0, n, mu_f, A) for g in grid], 1)  # (N, n_scan)
    crossed = f >= 0
    any_cross = crossed.any(1)
    first = np.argmax(crossed, 1)
    hi = grid[first]
    lo = grid[np.maximum(first - 1, 0)]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        go_up = cff(mid, principal, p0, n, mu_f, A) < 0
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    out = 0.5 * (lo + hi)
    out = np.where(first == 0, 0.0, out)
    return np.where(any_cross, out, DP_CAP)


def fracture_dp(principal, p0, A):
    """Pressure increase at which pore pressure reaches the minimum total
    stress (hydraulic fracturing). Each principal stress evolves along the
    stress path (horizontal ones rise by A*dp, the vertical one does not),
    so the stress that becomes minimum can change during injection:

        dp_frac = min( (Sh - p0)/(1 - A),  (SH - p0)/(1 - A),  Sv - p0 )
    """
    principal = np.asarray(principal, float)
    A = np.asarray(A, float)
    horiz = (np.minimum(principal[..., 0], principal[..., 1]) - p0) / (1.0 - A)
    vert = principal[..., 2] - p0
    return np.minimum(np.minimum(horiz, vert), DP_CAP)


def allowable_dp(principal, p0, n, mu_f, A, frac_safety: float = 0.9):
    """Allowable pressure increase = min(fault reactivation, 90% of the
    hydraulic-fracturing limit). Returns (dp_allow, governed_by_fault)."""
    slip = critical_dp(principal, p0, n, mu_f, A)
    frac = frac_safety * fracture_dp(principal, p0, A)
    return np.minimum(slip, frac), slip <= frac
