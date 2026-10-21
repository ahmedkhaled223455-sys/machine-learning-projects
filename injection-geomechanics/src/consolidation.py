"""Terzaghi 1D consolidation -- the classic coupled flow-deformation benchmark.

A saturated layer of thickness H is loaded instantly; the load is first
carried by excess pore pressure u0, which dissipates as fluid drains
through the top (u = 0 at z = 0) while the base is impermeable
(du/dz = 0 at z = H):

    du/dt = c_v d2u/dz2,     c_v = k / (mu_f * m_v)

Numerical solution: finite differences, Crank-Nicolson in time with a few
backward-Euler start-up steps (Rannacher smoothing) to damp the
oscillations CN produces from the discontinuous initial condition.
Verified against the analytical Fourier-series solution.
"""

from __future__ import annotations

import numpy as np


def analytic_pressure(z_over_H, T, u0=1.0, n_terms=200):
    """Excess pore pressure u(z, t) / u0 at time factor T = c_v t / H^2."""
    z = np.asarray(z_over_H, dtype=float)
    m = np.arange(n_terms)
    M = np.pi * (2 * m + 1) / 2.0
    terms = (2.0 / M)[:, None] * np.sin(np.outer(M, z)) * np.exp(-(M ** 2) * T)[:, None]
    return u0 * terms.sum(0)


def analytic_degree(T, n_terms=200):
    """Average degree of consolidation U(T) = settlement / final settlement."""
    m = np.arange(n_terms)
    M = np.pi * (2 * m + 1) / 2.0
    T = np.atleast_1d(np.asarray(T, dtype=float))
    return 1.0 - (2.0 / M ** 2 * np.exp(-np.outer(T, M ** 2))).sum(1)


def _thomas(a, b, c, d):
    """Tridiagonal solve; a = sub-, b = main, c = super-diagonal."""
    n = len(b)
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_numeric(T_end: float, nz: int = 50, nt: int = 200, u0: float = 1.0,
                  startup_steps: int = 4):
    """Return (z/H nodes, u at T_end) on a uniform grid. Node 0 is the drained top."""
    dz = 1.0 / nz
    z = np.linspace(0.0, 1.0, nz + 1)
    u = np.full(nz + 1, u0)
    u[0] = 0.0
    dt = T_end / nt
    n = nz  # unknowns at nodes 1..nz (node nz has a ghost node for du/dz = 0)

    def step(u, dt, theta):
        r = dt / dz ** 2
        a = np.full(n, -theta * r)
        b = np.full(n, 1 + 2 * theta * r)
        c = np.full(n, -theta * r)
        a[-1] = -2 * theta * r  # ghost node mirror at the impermeable base
        ui = u[1:]
        left = np.concatenate([[0.0], ui[:-1]])           # u_{i-1} (u_0 = 0)
        right = np.concatenate([ui[1:], [ui[-2]]])         # u_{i+1} (mirror at base)
        rhs = ui + (1 - theta) * r * (left - 2 * ui + right)
        new = u.copy()
        new[1:] = _thomas(a, b, c, rhs)
        new[0] = 0.0
        return new

    # Rannacher start-up: backward Euler half steps, then Crank-Nicolson
    for _ in range(startup_steps):
        u = step(u, dt / 2, 1.0)
    for _ in range(nt - startup_steps // 2):
        u = step(u, dt, 0.5)
    return z, u


def degree_from_profile(z, u, u0=1.0):
    """U = 1 - mean(u)/u0, trapezoidal average over the layer."""
    mean_u = np.sum((u[1:] + u[:-1]) / 2 * np.diff(z)) / (z[-1] - z[0])
    return 1.0 - mean_u / u0
