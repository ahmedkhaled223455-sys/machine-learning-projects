"""Statistical anomaly detection on returns.

Two kinds of "abnormal production behavior" the structural checks can't
express:

- Point outliers (bad ticks): a single return so far from the local
  distribution it's almost certainly a bad print. Flagged with a robust
  z-score -- median and MAD instead of mean and std, so the outliers
  don't inflate the very threshold meant to catch them.

- Volatility regime breaks: a ticker whose return distribution
  structurally changes (e.g. volatility jumps and stays high). Found with
  a variance change-point scan: the split of the series that maximizes
  the before/after variance ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def returns_from_prices(df: pd.DataFrame) -> pd.DataFrame:
    # fill_method=None: don't forward-fill missing prices into fake 0 returns.
    return df.pct_change(fill_method=None)


def robust_return_flags(returns: pd.DataFrame, k: float = 8.0
                        ) -> list[tuple[str, str]]:
    """Flag returns more than k robust-sigmas from the ticker's median."""
    out = []
    for tk in returns.columns:
        r = returns[tk]
        med = r.median()
        mad = (r - med).abs().median() * MAD_TO_SIGMA
        if mad == 0 or np.isnan(mad):
            continue
        z = (r - med) / mad
        for date in r.index[z.abs() > k]:
            out.append((tk, str(date.date())))
    return out


def _mad(a: np.ndarray) -> float:
    return float(np.median(np.abs(a - np.median(a))))


def dispersion_change_point(series: pd.Series, min_frac: float = 0.15):
    """Return (break_date, ratio) for the split maximizing the before/after
    ratio of *robust* dispersion (MAD). Using MAD instead of variance keeps
    a single bad tick from masquerading as a regime change -- the outlier
    barely moves the median absolute deviation."""
    r = series.dropna()
    vals = r.to_numpy()
    n = len(vals)
    if n < 40:
        return None, 1.0
    lo, hi = int(min_frac * n), int((1 - min_frac) * n)
    best_ratio, best_i = 1.0, None
    for i in range(lo, hi):
        s1, s2 = _mad(vals[:i]), _mad(vals[i:])
        if s1 <= 0 or s2 <= 0:
            continue
        ratio = max(s1, s2) / min(s1, s2)
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    if best_i is None:
        return None, 1.0
    return str(r.index[best_i].date()), best_ratio


def regime_breaks(returns: pd.DataFrame, ratio_threshold: float = 2.0
                  ) -> dict[str, dict]:
    """Per-ticker volatility-regime break, reported when the robust
    dispersion ratio across the best split exceeds a threshold."""
    out = {}
    for tk in returns.columns:
        date, ratio = dispersion_change_point(returns[tk])
        if date is not None and ratio >= ratio_threshold:
            out[tk] = {"date": date, "ratio": round(float(ratio), 2)}
    return out
