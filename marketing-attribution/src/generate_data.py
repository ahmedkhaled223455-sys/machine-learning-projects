"""Generate synthetic marketing data with KNOWN ground truth.

Real marketing data has no ground truth -- you can never prove an
attribution or mix model is correct because the true channel
contributions are unobservable. Synthetic data flips that: because we set
the true effects, we can check whether each method recovers them. That is
the whole point of this project's validation.

Writes three datasets plus the ground-truth answers:
  data/journeys.jsonl   -- customer paths for attribution
  data/mmm_weekly.csv   -- weekly spend + sales for the mix model
  data/campaign_geo.csv -- geo pre/post panel for causal lift
  data/ground_truth.json

Run:  python src/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from attribution import markov_attribution
from mmm import hill_transform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 20


# ---------------------------------------------------------------------------
# 1. Attribution journeys from a known Markov chain
# ---------------------------------------------------------------------------
ATTR_CHANNELS = ["search", "social", "email", "display"]
# states: start, search, social, email, display, conversion, null
_TRUE_ROWS = {
    "start":   {"search": .30, "social": .30, "email": .20, "display": .20},
    "search":  {"conversion": .25, "social": .20, "email": .10, "null": .45},
    "social":  {"conversion": .15, "email": .20, "search": .10, "null": .55},
    "email":   {"conversion": .30, "search": .10, "null": .60},
    "display": {"conversion": .08, "social": .20, "null": .72},
}


def _true_transition():
    states = ["start"] + ATTR_CHANNELS + ["conversion", "null"]
    idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for s, row in _TRUE_ROWS.items():
        for dst, p in row.items():
            P[idx[s], idx[dst]] = p
    P[idx["conversion"], idx["conversion"]] = 1.0
    P[idx["null"], idx["null"]] = 1.0
    return P, states, idx


def _simulate_journeys(n, rng):
    P, states, idx = _true_transition()
    inv = {i: s for s, i in idx.items()}
    journeys = []
    for _ in range(n):
        s = idx["start"]
        path = []
        for _ in range(20):                       # length cap
            s = rng.choice(len(states), p=P[s])
            name = inv[s]
            if name == "conversion":
                journeys.append((path, True)); break
            if name == "null":
                journeys.append((path, False)); break
            path.append(name)
        else:
            journeys.append((path, False))
    return journeys


# ---------------------------------------------------------------------------
# 2. MMM weekly spend + sales
# ---------------------------------------------------------------------------
MMM_CHANNELS = ["tv", "search", "social", "email"]
TRUE_THETA = {"tv": .60, "search": .20, "social": .40, "email": .10}
# Betas and spend scales set so the four channels have clearly tiered
# true efficiency (ROAS ~ search 15 > email 8 > social 6 > tv 3), which a
# working mix model should be able to recover and rank.
TRUE_BETA = {"tv": 700.0, "search": 1300.0, "social": 520.0, "email": 360.0}
SPEND_SCALE = {"tv": 120.0, "search": 45.0, "social": 45.0, "email": 22.0}


def _simulate_mmm(weeks, rng):
    t = np.arange(weeks)
    season = np.sin(2 * np.pi * t / 52.0)
    trend = t.astype(float)
    # Gamma spend: positive, high week-to-week variation, independent across
    # channels -> the dynamic range and low collinearity that let a mix model
    # actually separate channel effects.
    spend = {ch: rng.gamma(2.0, SPEND_SCALE[ch] / 2.0, weeks) for ch in MMM_CHANNELS}
    contributions = {}
    # Modest base/season/trend so channels drive most of the variance.
    sales = np.full(weeks, 800.0) + 120.0 * season + 0.6 * trend
    for ch in MMM_CHANNELS:
        sat = hill_transform(spend[ch], TRUE_THETA[ch])
        contributions[ch] = TRUE_BETA[ch] * sat
        sales = sales + contributions[ch]
    sales = sales + rng.normal(0, 80, weeks)

    total_contrib = sum(c.sum() for c in contributions.values())
    true_shares = {ch: float(contributions[ch].sum() / total_contrib)
                   for ch in MMM_CHANNELS}
    true_roas = {ch: float(contributions[ch].sum() / spend[ch].sum())
                 for ch in MMM_CHANNELS}
    df = pd.DataFrame({"week": t, "season": season, "trend": trend, "sales": sales})
    for ch in MMM_CHANNELS:
        df[f"spend_{ch}"] = spend[ch]
    return df, true_shares, true_roas


# ---------------------------------------------------------------------------
# 3. Campaign geo panel with a known lift
# ---------------------------------------------------------------------------
TRUE_LIFT = 120.0
N_GEOS = 60


def _simulate_campaign(rng):
    rows = []
    for g in range(N_GEOS):
        treated = g < N_GEOS // 2
        base = rng.normal(1000, 120)
        common_post_trend = 40.0                  # happens to everyone
        for period in ("pre", "post"):
            for _ in range(4):                    # 4 weeks per period
                mean = base + (common_post_trend if period == "post" else 0.0)
                if treated and period == "post":
                    mean += TRUE_LIFT
                rows.append({"geo": f"geo_{g:02d}", "period": period,
                             "treated": treated,
                             "sales": rng.normal(mean, 60)})
    return pd.DataFrame(rows)


def main():
    DATA.mkdir(exist_ok=True)
    # Independent seeds per dataset -- each simulation is reproducible on its
    # own, rather than depending on how many draws an earlier one consumed.
    journeys = _simulate_journeys(8000, np.random.default_rng(SEED))
    with open(DATA / "journeys.jsonl", "w") as f:
        for path, conv in journeys:
            f.write(json.dumps({"path": path, "converted": conv}) + "\n")

    true_P, states, _ = _true_transition()
    true_attr = markov_attribution(true_P, states, ATTR_CHANNELS)

    mmm_df, true_shares, true_roas = _simulate_mmm(250, np.random.default_rng(SEED))
    mmm_df.to_csv(DATA / "mmm_weekly.csv", index=False)

    campaign_df = _simulate_campaign(np.random.default_rng(SEED))
    campaign_df.to_csv(DATA / "campaign_geo.csv", index=False)

    ground_truth = {
        "attribution_true_shares": true_attr,
        "mmm_true_contribution_shares": true_shares,
        "mmm_true_roas": true_roas,
        "campaign_true_lift": TRUE_LIFT,
    }
    (DATA / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    n_conv = sum(1 for _, c in journeys if c)
    print(f"journeys: {len(journeys)} ({n_conv} conversions, "
          f"{100*n_conv/len(journeys):.1f}%)")
    print(f"mmm: {len(mmm_df)} weeks | campaign: {len(campaign_df)} rows")
    print("true attribution shares:",
          {k: round(v, 3) for k, v in true_attr.items()})


if __name__ == "__main__":
    main()
