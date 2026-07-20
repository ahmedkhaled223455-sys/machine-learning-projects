"""Tests for causal campaign-lift estimation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lift import bootstrap_ci, did_estimate, permutation_test
from generate_data import _simulate_campaign


def _panel_with_lift(lift, seed=0, n_geos=40, noise=30.0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_geos):
        treated = g < n_geos // 2
        base = rng.normal(1000, 100)
        for period in ("pre", "post"):
            for _ in range(4):
                mean = base + (50.0 if period == "post" else 0.0)  # common trend
                if treated and period == "post":
                    mean += lift
                rows.append({"geo": f"g{g}", "period": period,
                             "treated": treated, "sales": rng.normal(mean, noise)})
    return pd.DataFrame(rows)


def test_did_recovers_known_lift():
    df = _panel_with_lift(150.0, seed=1)
    est = did_estimate(df)
    assert abs(est - 150.0) < 30.0        # within noise


def test_did_near_zero_without_effect():
    df = _panel_with_lift(0.0, seed=2)
    assert abs(did_estimate(df)) < 30.0


def test_permutation_significant_when_effect_present():
    df = _panel_with_lift(150.0, seed=3)
    assert permutation_test(df, n_perm=2000, seed=0) < 0.05


def test_permutation_not_significant_without_effect():
    df = _panel_with_lift(0.0, seed=4, noise=60.0)
    assert permutation_test(df, n_perm=2000, seed=0) > 0.05


def test_bootstrap_ci_covers_truth():
    df = _panel_with_lift(150.0, seed=5)
    lo, hi = bootstrap_ci(df, n_boot=2000, seed=0)
    assert lo < 150.0 < hi
    assert lo < hi


def test_full_campaign_simulation_recovers_120():
    df = _simulate_campaign(np.random.default_rng(20))
    assert abs(did_estimate(df) - 120.0) < 30.0
