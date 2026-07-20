"""Tests for return anomaly detection and regime-break detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly import (dispersion_change_point, regime_breaks,
                     returns_from_prices, robust_return_flags)


def test_robust_flags_catch_a_spike():
    dates = pd.bdate_range("2022-01-03", periods=100)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, 100))}, index=dates)
    prices.iloc[50, 0] *= 2.0                      # bad tick
    flags = robust_return_flags(returns_from_prices(prices), k=8.0)
    flagged_dates = {d for _, d in flags}
    assert str(dates[50].date()) in flagged_dates


def test_robust_flags_quiet_on_clean_series():
    dates = pd.bdate_range("2022-01-03", periods=200)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))}, index=dates)
    assert robust_return_flags(returns_from_prices(prices), k=8.0) == []


def test_dispersion_change_point_locates_vol_shift():
    rng = np.random.default_rng(2)
    r = np.concatenate([rng.normal(0, 0.01, 150), rng.normal(0, 0.04, 150)])
    dates = pd.bdate_range("2022-01-03", periods=300)
    s = pd.Series(r, index=dates)
    date, ratio = dispersion_change_point(s)
    assert ratio > 2.0
    break_pos = dates.get_loc(pd.to_datetime(date))
    assert 120 < break_pos < 180                   # near the true split (150)


def test_regime_breaks_isolates_the_right_ticker():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2022-01-03", periods=300)
    calm = rng.normal(0, 0.01, 300)
    shift = np.concatenate([rng.normal(0, 0.01, 150), rng.normal(0, 0.04, 150)])
    returns = pd.DataFrame({"CALM": calm, "SHIFT": shift}, index=dates)
    breaks = regime_breaks(returns, ratio_threshold=2.0)
    assert "SHIFT" in breaks and "CALM" not in breaks
