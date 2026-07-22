"""Context adjustment: turn raw box-score averages into a comparable rate.

Points per game is the number everyone quotes and the one that misleads
most. It bundles together:

  - **pace**: a team playing 76 possessions per 40 minutes hands its
    players ~20% more scoring chances than one playing 63
  - **minutes**: a starter's raw total dwarfs a bench player's at equal skill
  - **schedule**: efficiency drops against better opponents

Dividing by possessions played removes the first two, and a schedule term
removes the third:

    points_per_100 = points_per_game / (pace * minutes / 40) * 100
    adjusted       = points_per_100 + SOS_COEF * strength_of_schedule

What's left is much closer to the player's underlying scoring ability, which
is what actually transfers when he changes programs.
"""

from __future__ import annotations

import pandas as pd

# Efficiency lost per unit of schedule strength (recovered by adding it back).
SOS_COEF = 1.6


def possessions_played(pace, minutes_per_game):
    """Possessions a player is on the floor for, per game."""
    return pace * (minutes_per_game / 40.0)


def add_context(seasons: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Join team pace/schedule onto each player-season."""
    return seasons.merge(teams, on="team_id", how="left", validate="many_to_one")


def add_adjusted_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add possessions, per-100 rate, and schedule-adjusted efficiency."""
    out = df.copy()
    out["possessions"] = possessions_played(out["pace"], out["minutes_per_game"])
    out["points_per_100"] = out["points_per_game"] / out["possessions"] * 100.0
    out["adjusted_efficiency"] = out["points_per_100"] + SOS_COEF * out["sos"]
    return out


def project_points_per_game(adjusted_efficiency, pace, minutes_per_game, sos):
    """Invert the adjustment: what this efficiency yields in a given context.

    This is the transfer question -- take what a player showed at his old
    program and express it in the new program's pace, role, and schedule.
    """
    pp100 = adjusted_efficiency - SOS_COEF * sos
    return pp100 * possessions_played(pace, minutes_per_game) / 100.0
