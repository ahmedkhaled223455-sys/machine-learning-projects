"""Pressure build-up around an injection well.

Single-phase radial diffusion in a confined reservoir of thickness h:

    phi * mu * c_t * dp/dt = (1/r) d/dr ( k(r) r dp/dr )

* `theis_dp`      -- analytical Theis line-source solution (homogeneous,
                     infinite-acting reservoir), the verification target.
* `simulate`      -- batched finite-volume solver (log-spaced radial grid,
                     implicit Euler, Thomas algorithm vectorised across many
                     reservoirs at once). Handles what Theis cannot: a
                     two-zone permeability (near-well damaged/stimulated
                     zone) and a closed outer boundary (compartmentalised
                     reservoir), where pressure builds above the
                     infinite-acting prediction.

The equation is linear in the injection rate, so the solver works with a
unit rate and results are scaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from expint import exp1

MU_BRINE = 5e-4      # Pa s
C_TOTAL = 1e-9       # 1/Pa, rock + fluid compressibility
R_WELL = 0.1         # m
YEAR = 365.25 * 24 * 3600.0


def diffusivity(k, phi, mu=MU_BRINE, ct=C_TOTAL):
    return k / (phi * mu * ct)


def theis_dp(r, t, Q, k, h, phi, mu=MU_BRINE, ct=C_TOTAL):
    """Pressure increase (Pa) at radius r (m), time t (s), rate Q (m3/s)."""
    u = np.asarray(r, float) ** 2 / (4.0 * diffusivity(k, phi, mu, ct) * np.asarray(t, float))
    return Q * mu / (4.0 * np.pi * k * h) * exp1(u)


@dataclass
class Grid:
    r_faces: np.ndarray   # (N, n+1)
    r_cent: np.ndarray    # (N, n)


def make_grid(r_outer: np.ndarray, n_cells: int = 240) -> Grid:
    """Log-spaced cells from the well radius to each reservoir's outer radius."""
    r_outer = np.atleast_1d(np.asarray(r_outer, float))
    s = np.linspace(0.0, 1.0, n_cells + 1)
    faces = R_WELL * (r_outer[:, None] / R_WELL) ** s[None, :]
    cent = np.sqrt(faces[:, :-1] * faces[:, 1:])
    return Grid(faces, cent)


def _thomas_batched(a, b, c, d):
    """Solve many tridiagonal systems at once; arrays shaped (N, n)."""
    N, n = b.shape
    cp = np.empty((N, n))
    dp = np.empty((N, n))
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, n):
        den = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / den
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / den
    x = np.empty((N, n))
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def simulate(t_end, h, phi, k_outer, k_inner=None, r_inner=None, r_outer=None,
             n_cells: int = 240, n_steps: int = 300, mu=MU_BRINE, ct=C_TOTAL,
             startup_steps: int = 20):
    """Unit-rate (1 m3/s) pressure increase at t_end for a batch of reservoirs.

    All parameters broadcast to shape (N,). r_outer = None means an
    effectively infinite reservoir (outer radius 50x the diffusion length,
    no-flow there). Returns (Grid, dp) with dp shaped (N, n_cells), Pa per m3/s.
    """
    h, phi, k_outer = (np.atleast_1d(np.asarray(v, float)) for v in (h, phi, k_outer))
    N = max(len(h), len(phi), len(k_outer))
    h, phi, k_outer = (np.broadcast_to(v, (N,)) for v in (h, phi, k_outer))
    k_inner = k_outer if k_inner is None else np.broadcast_to(np.asarray(k_inner, float), (N,))
    r_inner = np.zeros(N) if r_inner is None else np.broadcast_to(np.asarray(r_inner, float), (N,))
    if r_outer is None:
        r_outer = 50.0 * np.sqrt(diffusivity(np.maximum(k_outer, k_inner), phi, mu, ct) * t_end)
    r_outer = np.broadcast_to(np.asarray(r_outer, float), (N,))

    g = make_grid(r_outer, n_cells)
    rf, rc = g.r_faces, g.r_cent
    k_cell = np.where(rc < r_inner[:, None], k_inner[:, None], k_outer[:, None])
    vol = np.pi * (rf[:, 1:] ** 2 - rf[:, :-1] ** 2) * h[:, None]
    acc = vol * phi[:, None] * ct                              # storage per cell
    # transmissibility between cells i and i+1 (harmonic, radial)
    k_l, k_r = k_cell[:, :-1], k_cell[:, 1:]
    r_face = rf[:, 1:-1]
    res = (np.log(r_face / rc[:, :-1]) / k_l + np.log(rc[:, 1:] / r_face) / k_r)
    T = 2.0 * np.pi * h[:, None] / (mu * res)                  # (N, n-1)

    # geometric time steps spanning 1e-8 * t_end .. t_end; the growth ratio
    # shrinks as n_steps grows, so refining in time actually reduces error
    edges = t_end * np.concatenate([[0.0], np.logspace(-8, 0, n_steps)])
    dts = np.diff(edges)

    p = np.zeros((N, n_cells))
    q = np.zeros((N, n_cells))
    q[:, 0] = 1.0                                              # unit injection into the well cell

    def flux_div(p):
        """(A p)_i = sum of transmissibility-weighted pressure differences."""
        out = np.zeros_like(p)
        dpf = T * (p[:, 1:] - p[:, :-1])
        out[:, :-1] += dpf
        out[:, 1:] -= dpf
        return out

    # theta-scheme: backward Euler for the first steps (damps start-up
    # transients), Crank-Nicolson afterwards (second order in time)
    for j, dt in enumerate(dts):
        theta = 1.0 if j < startup_steps else 0.5
        a = np.zeros((N, n_cells))
        c = np.zeros((N, n_cells))
        b = acc / dt
        a[:, 1:] = -theta * T
        c[:, :-1] = -theta * T
        b[:, :-1] += theta * T
        b[:, 1:] += theta * T
        rhs = acc / dt * p + (1 - theta) * flux_div(p) + q
        p = _thomas_batched(a, b, c, rhs)
    return g, p


def interp_at(g: Grid, field: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Log-radius linear interpolation of a cell field at radius r (per sample)."""
    r = np.broadcast_to(np.asarray(r, float), (field.shape[0],))
    lr = np.log(g.r_cent)
    out = np.empty(field.shape[0])
    for i in range(field.shape[0]):
        out[i] = np.interp(np.log(r[i]), lr[i], field[i])
    return out
