"""Tests for the media mix model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm import MediaMixModel, _ridge, adstock, hill_transform, saturation
from generate_data import MMM_CHANNELS, _simulate_mmm


def test_adstock_carryover():
    x = np.array([1.0, 0.0, 0.0])
    y = adstock(x, 0.5)
    assert y[0] == 1.0 and y[1] == 0.5 and y[2] == 0.25   # geometric decay


def test_adstock_zero_theta_is_identity():
    x = np.array([2.0, 5.0, 1.0])
    assert np.allclose(adstock(x, 0.0), x)


def test_saturation_is_increasing_and_bounded():
    s = saturation(np.array([0.0, 1.0, 10.0, 1e6]), half=5.0)
    assert s[0] == 0.0
    assert np.all(np.diff(s) > 0)         # monotone increasing
    assert np.all(s < 1.0)                # bounded below 1


def test_ridge_recovers_linear_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    true = np.array([2.0, -1.0, 0.5])
    y = 4.0 + X @ true + rng.normal(0, 0.01, 500)
    intercept, coef = _ridge(X, y, lam=1e-6)
    assert np.allclose(coef, true, atol=0.05)
    assert abs(intercept - 4.0) < 0.05


def test_hill_transform_self_calibrates_at_median():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    # with theta=0 (no carryover), half defaults to the median (3.0),
    # so the median input maps to 0.5
    t = hill_transform(x, theta=0.0)
    assert abs(t[2] - 0.5) < 1e-6


def test_mmm_recovers_contribution_shares_and_roas():
    df, true_shares, true_roas = _simulate_mmm(250, np.random.default_rng(20))
    spend = {c: df[f"spend_{c}"].to_numpy() for c in MMM_CHANNELS}
    sales = df["sales"].to_numpy()
    controls = {"season": df["season"].to_numpy(), "trend": df["trend"].to_numpy()}

    model = MediaMixModel(MMM_CHANNELS).fit(spend, sales, controls)
    shares = model.contribution_shares(spend, controls)
    roas = model.roas(spend, controls)

    mae = np.mean([abs(shares[c] - true_shares[c]) for c in MMM_CHANNELS])
    assert model.r2(spend, sales, controls) > 0.8
    assert mae < 0.06
    # ROAS ordering recovered: search most efficient, tv least
    assert max(roas, key=roas.get) == "search"
    assert min(roas, key=roas.get) == "tv"
