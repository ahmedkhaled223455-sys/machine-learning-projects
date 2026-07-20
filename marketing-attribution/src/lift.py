"""Campaign lift: did a campaign *cause* incremental sales?

Uses a geo holdout / difference-in-differences (DiD) design: some markets
get the campaign (treated), others don't (control), and we compare the
pre-to-post change in the treated markets against the same change in the
control markets. Subtracting the control change removes any trend or
seasonality that would have happened anyway, isolating the campaign's
causal effect.

Significance comes from a permutation test (shuffle which markets were
treated and see how often chance reproduces the observed effect), and
uncertainty from a bootstrap confidence interval over markets. Same
causal toolkit as the student-outcomes project, applied to marketing.

Data model: a long panel with columns geo, period ('pre'/'post'),
treated (bool), sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_mean(df: pd.DataFrame, treated: bool, period: str) -> float:
    sub = df[(df["treated"] == treated) & (df["period"] == period)]
    return float(sub["sales"].mean())


def did_estimate(df: pd.DataFrame) -> float:
    """Difference-in-differences point estimate of the campaign lift."""
    treated_change = _group_mean(df, True, "post") - _group_mean(df, True, "pre")
    control_change = _group_mean(df, False, "post") - _group_mean(df, False, "pre")
    return treated_change - control_change


def _geo_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per geo: its pre->post change and treatment flag."""
    rows = []
    for geo, g in df.groupby("geo"):
        pre = g[g["period"] == "pre"]["sales"].mean()
        post = g[g["period"] == "post"]["sales"].mean()
        rows.append({"geo": geo, "treated": bool(g["treated"].iloc[0]),
                     "change": post - pre})
    return pd.DataFrame(rows)


def permutation_test(df: pd.DataFrame, n_perm: int = 5000,
                     seed: int = 0) -> float:
    """Two-sided p-value: how often shuffled treatment matches the effect."""
    geo = _geo_level(df)
    changes = geo["change"].to_numpy()
    treated_mask = geo["treated"].to_numpy()
    n_treated = int(treated_mask.sum())

    observed = (changes[treated_mask].mean() - changes[~treated_mask].mean())
    rng = np.random.default_rng(seed)
    n = len(changes)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        t = perm[:n_treated]
        c = perm[n_treated:]
        stat = changes[t].mean() - changes[c].mean()
        if abs(stat) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the DiD lift, resampling geos."""
    geo = _geo_level(df)
    treated = geo[geo["treated"]]["change"].to_numpy()
    control = geo[~geo["treated"]]["change"].to_numpy()
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        t = rng.choice(treated, size=len(treated), replace=True)
        c = rng.choice(control, size=len(control), replace=True)
        stats[b] = t.mean() - c.mean()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def analyze_campaign(df: pd.DataFrame, seed: int = 0) -> dict:
    return {
        "did_lift": did_estimate(df),
        "p_value": permutation_test(df, seed=seed),
        "ci95": bootstrap_ci(df, seed=seed),
    }
