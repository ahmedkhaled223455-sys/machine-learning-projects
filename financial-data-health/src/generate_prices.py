"""Generate a synthetic multi-ticker price panel with injected data-health
defects and known ground truth.

Production market-data feeds break in characteristic ways, and this
recreates them so detectors can be validated against a known answer:
  - missing values   (a feed drops a cell)
  - stale prices     (a feed sticks, repeating the last value)
  - outlier bad ticks(a fat-finger / bad print: price jumps and reverts)
  - missing dates    (a whole day is absent from the feed)
  - volatility regime break (a ticker's behavior structurally changes --
    "abnormal production behavior" rather than a point error)

Writes data/prices.csv (corrupted, as a feed would deliver it) and
data/ground_truth.json.

Run:  python src/generate_prices.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 7
TICKERS = ["ACO", "BLT", "CDX", "DRN", "EGL", "FTX"]
N_DAYS = 504
MU, SIGMA = 0.0003, 0.012
REGIME_TICKER = "CDX"
REGIME_BREAK = 300          # index; second half has elevated volatility
REGIME_VOL_MULT = 3.0


def _clean_prices(rng):
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS)
    data = {}
    for tk in TICKERS:
        r = rng.normal(MU, SIGMA, N_DAYS)
        if tk == REGIME_TICKER:
            r[REGIME_BREAK:] = rng.normal(MU, SIGMA * REGIME_VOL_MULT,
                                          N_DAYS - REGIME_BREAK)
        data[tk] = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(data, index=dates)


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = _clean_prices(rng)
    dates = df.index
    gt = {"missing": [], "stale": [], "outlier": [], "missing_dates": [],
          "regime_break": {"ticker": REGIME_TICKER,
                           "date": str(dates[REGIME_BREAK].date())}}

    # --- missing values: drop random cells -------------------------------
    for _ in range(10):
        i = rng.integers(1, N_DAYS); tk = rng.choice(TICKERS)
        df.iloc[i, df.columns.get_loc(tk)] = np.nan
        gt["missing"].append([tk, str(dates[i].date())])

    # --- stale runs: repeat the previous price ---------------------------
    for _ in range(3):
        tk = rng.choice(TICKERS)
        start = int(rng.integers(20, N_DAYS - 10))
        run = int(rng.integers(3, 6))
        stuck = df.iloc[start - 1, df.columns.get_loc(tk)]
        for k in range(run):
            df.iloc[start + k, df.columns.get_loc(tk)] = stuck
            gt["stale"].append([tk, str(dates[start + k].date())])

    # --- outlier bad ticks: spike then revert ----------------------------
    for _ in range(8):
        i = int(rng.integers(1, N_DAYS)); tk = str(rng.choice(TICKERS))
        factor = float(rng.choice([0.5, 1.8, 2.5]))
        df.iloc[i, df.columns.get_loc(tk)] *= factor
        gt["outlier"].append([tk, str(dates[i].date())])

    # --- missing dates: remove whole rows from the feed ------------------
    drop_idx = sorted(rng.choice(range(30, N_DAYS - 30), size=4, replace=False))
    gt["missing_dates"] = [str(dates[i].date()) for i in drop_idx]
    df = df.drop(df.index[drop_idx])

    df.to_csv(DATA / "prices.csv", index_label="date")
    (DATA / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    print(f"wrote {len(df)} rows x {len(TICKERS)} tickers")
    print(f"injected: {len(gt['missing'])} missing, {len(gt['stale'])} stale, "
          f"{len(gt['outlier'])} outliers, {len(gt['missing_dates'])} missing dates, "
          f"1 regime break ({REGIME_TICKER})")


if __name__ == "__main__":
    main()
