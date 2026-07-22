"""Integrity checks across the player-season feed.

Scouting data arrives from several sources and disagrees: the same
player-season appears twice, minutes exceed the length of a game, box-score
fields go missing or negative. Every one of those silently corrupts a
player-value model, so they get caught and quarantined before any modeling.
"""

from __future__ import annotations

import pandas as pd

GAME_MINUTES = 40.0        # regulation; overtime raises per-game averages slightly
MINUTES_CEILING = 40.0


def find_issues(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return a frame of offending rows for each class of problem."""
    dup_mask = df.duplicated(subset=["player_id", "season"], keep=False)
    return {
        "duplicate_player_seasons": df[dup_mask],
        "impossible_minutes": df[df["minutes_per_game"] > MINUTES_CEILING],
        "negative_points": df[df["points_per_game"] < 0],
        "missing_points": df[df["points_per_game"].isna()],
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop unusable rows; keep the first copy of any duplicated player-season."""
    before = len(df)
    out = df.copy()

    out = out[~(out["minutes_per_game"] > MINUTES_CEILING)]
    out = out[~(out["points_per_game"] < 0)]
    out = out[out["points_per_game"].notna()]
    out = out.drop_duplicates(subset=["player_id", "season"], keep="first")

    removed = {
        "rows_in": before,
        "rows_out": len(out),
        "rows_removed": before - len(out),
    }
    return out.reset_index(drop=True), removed


def report(df: pd.DataFrame) -> str:
    issues = find_issues(df)
    lines = ["Data integrity check:"]
    for name, rows in issues.items():
        lines.append(f"  {name:26s} {len(rows):3d} rows")
    return "\n".join(lines)
