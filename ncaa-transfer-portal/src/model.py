"""Projecting a player's production at a new program.

Two ways to answer "what will this transfer give us?":

**Baseline (what gets quoted):** he averaged 14.2 a game, so expect about
14.2. This implicitly assumes the new program plays at the same pace,
against the same quality of opponent, and gives him the same role.

**Context-aware:** strip his old context out to get a schedule-adjusted
per-100-possession rate, then re-express that rate in the new program's
pace, schedule, and expected role.

The second is only worth the effort if it actually predicts better -- which
is what the analysis measures, and where the gap is widest for players who
changed teams.

Note the assumption in the projection: expected minutes at the new program
are treated as known (a coach recruiting a transfer has a role in mind).
Projecting the role too would add its own error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adjust import project_points_per_game


def baseline_projection(prior_season: pd.DataFrame) -> pd.Series:
    """Carry last season's scoring average forward unchanged."""
    return prior_season["points_per_game"]


def context_projection(prior_season: pd.DataFrame, new_context: pd.DataFrame) -> pd.Series:
    """Re-express last season's adjusted efficiency in the new context."""
    return project_points_per_game(
        prior_season["adjusted_efficiency"].to_numpy(),
        new_context["pace"].to_numpy(),
        new_context["minutes_per_game"].to_numpy(),
        new_context["sos"].to_numpy(),
    )


def rmse(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.nanmean((actual - predicted) ** 2)))


def mae(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.nanmean(np.abs(actual - predicted)))


def correlation(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def undervalued(prior_season: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Players whose raw scoring most understates their adjusted efficiency.

    Ranks each player on both measures; a large positive gap means the box
    score sells him short -- slow pace, limited minutes, or a hard schedule.
    """
    df = prior_season.copy()
    df["raw_rank"] = df["points_per_game"].rank(ascending=False)
    df["adj_rank"] = df["adjusted_efficiency"].rank(ascending=False)
    df["rank_gap"] = df["raw_rank"] - df["adj_rank"]
    cols = ["player_id", "team_id", "points_per_game", "minutes_per_game",
            "pace", "sos", "adjusted_efficiency", "raw_rank", "adj_rank", "rank_gap"]
    return df.sort_values("rank_gap", ascending=False).head(top_n)[cols]
