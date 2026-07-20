"""Tests for the deterministic data-correctness checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import checks


def _panel():
    dates = pd.bdate_range("2022-01-03", periods=6)
    df = pd.DataFrame({"AAA": [100, 101, 101, 102, 103, 104.0],
                       "BBB": [50, 51, 52, 53, 54, 55.0]}, index=dates)
    df.iloc[4, df.columns.get_loc("BBB")] = np.nan   # missing
    return df, dates


def test_missing_values():
    df, dates = _panel()
    assert checks.missing_values(df) == [("BBB", str(dates[4].date()))]


def test_stale_prices_flags_repeat():
    df, dates = _panel()
    # AAA repeats 101 at index 2
    assert ("AAA", str(dates[2].date())) in checks.stale_prices(df)


def test_range_violations():
    df, _ = _panel()
    df.iloc[0, 0] = -5.0
    viol = checks.range_violations(df)
    assert any(tk == "AAA" for tk, _ in viol)


def test_date_gaps_detects_removed_day():
    df, dates = _panel()
    df2 = df.drop(dates[3])              # remove a business day
    gaps = checks.date_gaps(df2)
    assert str(dates[3].date()) in gaps
