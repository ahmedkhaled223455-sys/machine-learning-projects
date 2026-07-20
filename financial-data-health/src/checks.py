"""Deterministic data-correctness checks over a price panel.

These are the cheap, precise structural checks that catch the common feed
failures directly -- missing cells, stuck (stale) prices, non-positive
values, and missing trading days. Each returns a plain list of findings
so the monitor can aggregate and score them.

The panel is a DataFrame indexed by date, one column per ticker.
"""

from __future__ import annotations

import pandas as pd


def missing_values(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Cells that are NaN -- a dropped value in the feed."""
    out = []
    for tk in df.columns:
        for date in df.index[df[tk].isna()]:
            out.append((tk, str(date.date())))
    return out


def stale_prices(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Consecutive identical prices -- a stuck feed. Continuous prices
    essentially never repeat exactly, so an exact repeat is a strong
    stale signal."""
    out = []
    for tk in df.columns:
        s = df[tk]
        repeated = (s == s.shift(1)) & s.notna()
        for date in s.index[repeated]:
            out.append((tk, str(date.date())))
    return out


def range_violations(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Non-positive prices -- impossible for equities."""
    out = []
    for tk in df.columns:
        bad = df[tk] <= 0
        for date in df.index[bad.fillna(False)]:
            out.append((tk, str(date.date())))
    return out


def date_gaps(df: pd.DataFrame) -> list[str]:
    """Business days missing from the index -- a whole absent feed day."""
    full = pd.bdate_range(df.index.min(), df.index.max())
    missing = full.difference(df.index)
    return [str(d.date()) for d in missing]
