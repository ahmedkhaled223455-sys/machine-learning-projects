"""Tests for the context adjustment, integrity checks, and projections."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_checks
import model
from adjust import (add_adjusted_metrics, add_context, possessions_played,
                    project_points_per_game, SOS_COEF)


# --- context adjustment ---------------------------------------------------

def test_possessions_scale_with_pace_and_minutes():
    # 40 minutes at 70 pace = the team's full possession count
    assert possessions_played(70, 40) == 70
    assert possessions_played(70, 20) == 35        # half the minutes, half the chances
    assert possessions_played(80, 20) == 40        # faster team, more chances


def test_per_100_undoes_pace_and_minutes():
    """Two players with identical efficiency but different context should
    have different PPG and the same per-100 rate."""
    seasons = pd.DataFrame({
        "player_id": ["A", "B"], "season": [2024, 2024],
        "team_id": ["fast", "slow"],
        "minutes_per_game": [30.0, 30.0],
        # same 20 pts/100: fast team yields more raw points
        "points_per_game": [20.0 * (76 * 30 / 40) / 100, 20.0 * (64 * 30 / 40) / 100],
    })
    teams = pd.DataFrame({"team_id": ["fast", "slow"], "pace": [76.0, 64.0],
                          "sos": [0.0, 0.0]})
    out = add_adjusted_metrics(add_context(seasons, teams))
    assert out.loc[0, "points_per_game"] > out.loc[1, "points_per_game"]
    assert abs(out.loc[0, "points_per_100"] - out.loc[1, "points_per_100"]) < 1e-9


def test_schedule_adjustment_credits_tougher_opponents():
    seasons = pd.DataFrame({
        "player_id": ["A", "B"], "season": [2024, 2024],
        "team_id": ["hard", "easy"],
        "minutes_per_game": [30.0, 30.0], "points_per_game": [15.0, 15.0],
    })
    teams = pd.DataFrame({"team_id": ["hard", "easy"], "pace": [70.0, 70.0],
                          "sos": [1.5, -1.5]})
    out = add_adjusted_metrics(add_context(seasons, teams))
    # identical raw production, but the tougher schedule earns a higher rating
    assert out.loc[0, "adjusted_efficiency"] > out.loc[1, "adjusted_efficiency"]


def test_projection_inverts_the_adjustment():
    """Adjusting then projecting back into the same context is a round trip."""
    ppg, pace, minutes, sos = 14.0, 72.0, 28.0, 0.4
    pp100 = ppg / possessions_played(pace, minutes) * 100
    adjusted = pp100 + SOS_COEF * sos
    assert abs(project_points_per_game(adjusted, pace, minutes, sos) - ppg) < 1e-9


# --- integrity ------------------------------------------------------------

def _dirty_frame():
    return pd.DataFrame({
        "player_id": ["A", "A", "B", "C", "D"],
        "season": [2024, 2024, 2024, 2024, 2024],
        "team_id": ["T1"] * 5,
        "minutes_per_game": [30.0, 30.0, 55.0, 25.0, 20.0],   # 55 > game length
        "points_per_game": [12.0, 12.0, 9.0, -3.0, np.nan],   # negative, missing
    })


def test_find_issues_flags_each_problem():
    issues = data_checks.find_issues(_dirty_frame())
    assert len(issues["duplicate_player_seasons"]) == 2       # both copies
    assert len(issues["impossible_minutes"]) == 1
    assert len(issues["negative_points"]) == 1
    assert len(issues["missing_points"]) == 1


def test_clean_removes_bad_rows_and_dedupes():
    clean, stats = data_checks.clean(_dirty_frame())
    assert stats["rows_removed"] == 4          # dup + impossible + negative + missing
    assert len(clean) == 1 and clean.loc[0, "player_id"] == "A"
    assert clean["player_id"].is_unique


# --- metrics and projections ---------------------------------------------

def test_error_metrics():
    assert model.rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert model.mae([1, 2], [2, 4]) == 1.5
    assert abs(model.correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12


def test_correlation_ignores_missing():
    assert abs(model.correlation([1, 2, 3, np.nan], [2, 4, 6, 5]) - 1.0) < 1e-12


def test_undervalued_ranks_context_suppressed_players_first():
    df = pd.DataFrame({
        "player_id": ["star", "hidden"], "team_id": ["fast", "slow"],
        "points_per_game": [18.0, 6.0], "minutes_per_game": [32.0, 14.0],
        "pace": [78.0, 62.0], "sos": [-1.0, 1.5],
        "adjusted_efficiency": [20.0, 28.0],
    })
    board = model.undervalued(df, top_n=1)
    # the low-scoring player on a slow team with a hard schedule is the find
    assert board.iloc[0]["player_id"] == "hidden"


def test_adjustment_helps_transfers_not_stayers():
    """The signature of a correct context adjustment: it matters when the
    context changes, and does nothing when it doesn't."""
    import json
    metrics_path = Path(__file__).resolve().parents[1] / "results" / "metrics.json"
    if not metrics_path.exists():
        import pytest
        pytest.skip("run src/run_analysis.py first")
    m = json.loads(metrics_path.read_text())["projection_rmse"]
    assert m["transfers only"]["improvement_pct"] > 5.0
    assert abs(m["stayers only"]["improvement_pct"]) < 1.0
