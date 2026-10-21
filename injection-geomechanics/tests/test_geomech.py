"""Verification tests: analytic benchmarks, convergence orders, geomechanics
invariants, and surrogate/ROM sanity checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import consolidation as tz  # noqa: E402
from expint import exp1  # noqa: E402
from fault_stability import (DP_CAP, allowable_dp, cff, critical_dp,  # noqa: E402
                             fault_normal, fracture_dp, in_situ,
                             stress_path_coefficient, traction)
from radial_flow import (YEAR, diffusivity, interp_at, simulate,  # noqa: E402
                         theis_dp)
from rom import PODRBF  # noqa: E402
from surrogate import RidgeSurrogate, physics_unit_dp  # noqa: E402


# ---------------- exponential integral ----------------

def test_exp1_reference_values():
    ref = {0.001: 6.331539364136149, 0.1: 1.8229239584193906, 1.0: 0.21938393439552029,
           5.0: 0.0011482955912753257, 10.0: 4.156968929685324e-06}
    x = np.array(list(ref))
    assert np.allclose(exp1(x), list(ref.values()), rtol=1e-12)


def test_exp1_continuous_at_branch_switch():
    a, b = exp1(np.array([1.0 - 1e-9, 1.0 + 1e-9]))
    assert abs(a - b) < 1e-8


def test_exp1_rejects_nonpositive():
    with pytest.raises(ValueError):
        exp1(np.array([0.0]))


# ---------------- Terzaghi consolidation ----------------

def test_terzaghi_matches_analytic():
    for T in (0.05, 0.2, 0.5, 1.0):
        z, u = tz.solve_numeric(T, 50, 200)
        assert np.abs(u - tz.analytic_pressure(z, T)).max() < 2e-4


def test_terzaghi_second_order_convergence():
    errs = []
    for nz in (10, 20, 40, 80):
        z, u = tz.solve_numeric(0.2, nz, nz * 4)
        errs.append(np.abs(u - tz.analytic_pressure(z, 0.2)).max())
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert (orders > 1.9).all()


def test_terzaghi_degree_of_consolidation():
    assert np.isclose(tz.analytic_degree(0.197)[0], 0.5, atol=2e-3)   # textbook T50 = 0.197
    z, u = tz.solve_numeric(0.5, 80, 320)
    assert abs(tz.degree_from_profile(z, u) - tz.analytic_degree(0.5)[0]) < 1e-3


# ---------------- radial injection flow ----------------

@pytest.fixture(scope="module")
def homogeneous():
    k = np.array([1e-14, 1e-13, 1e-15])
    h = np.array([50.0, 100.0, 150.0])
    phi = np.array([0.2, 0.15, 0.1])
    g, dp = simulate(10 * YEAR, h, phi, k)
    return k, h, phi, g, dp


def test_simulator_matches_theis(homogeneous):
    k, h, phi, g, dp = homogeneous
    for r in (200.0, 1000.0, 5000.0):
        num = interp_at(g, dp, r)
        th = theis_dp(r, 10 * YEAR, 1.0, k, h, phi)
        assert np.abs(num / th - 1).max() < 2e-3


def test_simulator_converges_with_refinement():
    k, h, phi = np.array([1e-14]), np.array([80.0]), np.array([0.15])
    errs = []
    for nc, ns in ((60, 75), (120, 150), (240, 300)):
        g, dp = simulate(YEAR, h, phi, k, n_cells=nc, n_steps=ns)
        errs.append(abs(interp_at(g, dp, 500.0)[0] / theis_dp(500.0, YEAR, 1, k, h, phi)[0] - 1))
    assert errs[0] > errs[1] > errs[2]


def test_mass_balance_in_closed_reservoir():
    """Closed reservoir: injected volume = storage increase (phi c_t V dp)."""
    k, h, phi, Re, t = np.array([1e-13]), np.array([50.0]), np.array([0.2]), np.array([3000.0]), 5 * YEAR
    g, dp = simulate(t, h, phi, k, r_outer=Re)
    vol = np.pi * (g.r_faces[:, 1:] ** 2 - g.r_faces[:, :-1] ** 2) * h[:, None]
    stored = np.sum(vol * phi[:, None] * 1e-9 * dp)
    assert np.isclose(stored, 1.0 * t, rtol=1e-6)


def test_closed_boundary_raises_pressure():
    k, h, phi = np.array([1e-13]), np.array([50.0]), np.array([0.2])
    g1, open_ = simulate(10 * YEAR, h, phi, k)
    g2, closed = simulate(10 * YEAR, h, phi, k, r_outer=np.array([4000.0]))
    assert interp_at(g2, closed, 1000.0)[0] > 1.5 * interp_at(g1, open_, 1000.0)[0]


def test_physics_estimate_close_to_simulator():
    rng = np.random.default_rng(0)
    N = 60
    k = 10 ** rng.uniform(-15, -13, N)
    h, phi = rng.uniform(20, 150, N), rng.uniform(0.08, 0.25, N)
    Re = 10 ** rng.uniform(np.log10(3000), np.log10(30000), N)
    r = rng.uniform(300, 2500, N)
    g, dp = simulate(10 * YEAR, h, phi, k, r_outer=Re)
    est = physics_unit_dp(r, 10 * YEAR, k, k, np.full(N, 10.0), h, phi, Re)
    assert np.median(np.abs(est / interp_at(g, dp, r) - 1)) < 0.01


# ---------------- geomechanics ----------------

def test_fault_normal_is_unit_and_orthogonal_to_strike():
    n = fault_normal(np.array([0.0, 30.0, 75.0]), np.array([45.0, 60.0, 80.0]))
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)
    strike = np.stack([np.cos(np.radians([0, 30, 75])), np.sin(np.radians([0, 30, 75])), np.zeros(3)], -1)
    assert np.allclose(np.sum(n * strike, -1), 0.0)


def test_traction_invariants():
    S = np.array([[50e6, 40e6, 60e6]])
    n = fault_normal(np.array([20.0]), np.array([55.0]))
    sn, tau = traction(S, n)
    assert S.min() <= sn[0] <= S.max()
    assert tau[0] <= (S.max() - S.min()) / 2 + 1e-6      # never exceeds the Mohr-circle radius


def test_critical_dp_matches_mohr_circle_without_coupling():
    mu = 0.6
    dip = 45 + np.degrees(np.arctan(mu)) / 2                 # optimally oriented normal fault
    n = fault_normal(np.array([0.0]), np.array([dip]))
    S, p0 = in_situ(np.array([2500.0]), np.array([0.7]), np.array([0.85]))
    s1, s3 = S[0, 2], S[0, 1]
    analytic = (s1 + s3) / 2 - (s1 - s3) / 2 / np.sin(np.arctan(mu)) - p0[0]
    assert np.isclose(critical_dp(S, p0, n, mu, 0.0)[0], analytic, rtol=1e-6)


def test_coupled_cff_matches_hand_calculation():
    """Fault striking along SHmax: its normal lies in the Shmin-vertical plane,
    so sigma_n and tau follow 2D formulas; only Shmin feels the stress path."""
    dip = 60.0
    n = fault_normal(np.array([0.0]), np.array([dip]))
    S, p0 = in_situ(np.array([2500.0]), np.array([0.7]), np.array([0.85]))
    A, mu, dp = 0.6, 0.6, 3e6
    ny2, nz2 = n[0, 1] ** 2, n[0, 2] ** 2
    sh, sv = S[0, 1] + A * dp, S[0, 2]              # vertical stress is NOT raised
    sn = sh * ny2 + sv * nz2
    tau = abs(sv - sh) * np.sqrt(ny2 * nz2)
    expect = tau - mu * (sn - (p0[0] + dp))
    got = cff(np.array([dp]), S, p0, n, np.array([mu]), np.array([A]))[0]
    assert np.isclose(got, expect, rtol=1e-9)


def test_critical_dp_is_first_crossing():
    rng = np.random.default_rng(1)
    N = 300
    S, p0 = in_situ(rng.uniform(1500, 3500, N), rng.uniform(0.55, 0.75, N), rng.uniform(0.75, 0.95, N))
    n = fault_normal(rng.uniform(0, 90, N), rng.uniform(40, 80, N))
    mu, A = rng.uniform(0.5, 0.8, N), stress_path_coefficient(rng.uniform(0.2, 0.3, N))
    dpc = critical_dp(S, p0, n, mu, A)
    inner = (dpc > 0) & (dpc < DP_CAP)
    assert inner.any()
    assert np.allclose(cff(dpc[inner], S[inner], p0[inner], n[inner], mu[inner], A[inner]), 0.0, atol=1e3)
    for frac in (0.25, 0.5, 0.9):        # stable everywhere below the reported value
        assert (cff(frac * dpc[inner], S[inner], p0[inner], n[inner], mu[inner], A[inner]) < 0).all()


def test_already_critical_fault_returns_zero():
    S, p0 = in_situ(np.array([2500.0]), np.array([0.5]), np.array([0.8]))
    n = fault_normal(np.array([0.0]), np.array([60.0]))
    assert critical_dp(S, p0, n, 0.6, 0.0)[0] == 0.0


def test_fracture_limit_tracks_changing_minimum_stress():
    S, p0 = in_situ(np.array([2500.0]), np.array([0.7]), np.array([0.85]))
    A = stress_path_coefficient(0.25)
    # horizontal route: (Sh - p0)/(1 - A); vertical route: Sv - p0 -- the smaller wins
    expect = min((S[0, 1] - p0[0]) / (1 - A), S[0, 2] - p0[0])
    assert np.isclose(fracture_dp(S, p0, A)[0], expect)


def test_allowable_is_min_of_limits():
    rng = np.random.default_rng(2)
    N = 200
    S, p0 = in_situ(rng.uniform(1500, 3500, N), rng.uniform(0.55, 0.75, N), rng.uniform(0.75, 0.95, N))
    n = fault_normal(rng.uniform(0, 90, N), rng.uniform(40, 80, N))
    mu, A = rng.uniform(0.5, 0.8, N), stress_path_coefficient(0.25)
    allow, by_fault = allowable_dp(S, p0, n, mu, A, 0.9)
    assert np.allclose(allow, np.minimum(critical_dp(S, p0, n, mu, A), 0.9 * fracture_dp(S, p0, A)))
    assert by_fault.any() and (~by_fault).any()


# ---------------- surrogates and ROM ----------------

def test_ridge_recovers_quadratic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = 1 + 2 * X[:, 0] - X[:, 1] * X[:, 2] + 0.5 * X[:, 2] ** 2
    m = RidgeSurrogate(1e-8).fit(X, y)
    assert np.allclose(m.predict(X), y, atol=1e-6)


def test_pod_rbf_reconstructs_smooth_family():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 50)
    P = rng.uniform(0.5, 2.0, (300, 2))
    fields = np.sin(np.outer(P[:, 0], x) * 3) * P[:, 1:2]
    rom = PODRBF(n_modes=6, length_scale=0.5).fit(P[:250], fields[:250])
    assert rom.energy[5] > 0.999
    assert np.abs(rom.predict(P[250:]) - fields[250:]).max() < 0.05


def test_diffusivity_units():
    assert np.isclose(diffusivity(1e-14, 0.2), 1e-14 / (0.2 * 5e-4 * 1e-9))
