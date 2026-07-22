"""Simulate two seasons of college basketball player-seasons, with transfers.

The simulation encodes the actual arithmetic of basketball production:

    points per game = points per 100 possessions x possessions played / 100
    possessions played = team pace x (minutes played / 40)

so a player's raw scoring average confounds three things -- how good he is,
how fast his team plays, and how much he plays -- plus how tough his
schedule was. Each player has a latent `true_talent` (points per 100
possessions above average) that is never observed directly.

Between season 1 and season 2 a share of players enter the transfer
portal and move to a team with different pace, schedule strength, and
role. That is the setting the analysis has to handle: the same player
produces different raw numbers at a different program.

A few rows are also corrupted (duplicates, impossible minutes, negatives,
missing values) so the integrity checks have real work to do.

Run:  python src/generate_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DATA

SEED = 12
N_TEAMS = 40
N_PLAYERS = 600
TRANSFER_RATE = 0.25
BASE_PP100 = 18.0          # league-average points per 100 possessions for a rotation player


def _make_teams(rng) -> pd.DataFrame:
    return pd.DataFrame({
        "team_id": [f"T{i:02d}" for i in range(N_TEAMS)],
        # possessions per 40 minutes: slow grinders to run-and-gun
        "pace": rng.uniform(62, 78, N_TEAMS).round(1),
        # schedule strength: positive = tougher opponents, suppresses efficiency
        "sos": rng.normal(0, 1, N_TEAMS).round(2),
    })


def _season_rows(players, teams, team_of, season, rng) -> pd.DataFrame:
    t = teams.set_index("team_id")
    rows = []
    for p, tid in zip(players.itertuples(), team_of):
        pace = float(t.loc[tid, "pace"])
        sos = float(t.loc[tid, "sos"])

        # Role: better players earn more minutes, with noise.
        minutes = np.clip(18 + 4.5 * p.true_talent + rng.normal(0, 3.5), 8, 36)

        # Efficiency: talent, minus a penalty for tougher schedule, plus noise.
        pp100 = BASE_PP100 + p.true_talent * 3.2 - 1.6 * sos + rng.normal(0, 1.6)
        pp100 = max(pp100, 2.0)

        possessions = pace * (minutes / 40.0)
        ppg = pp100 * possessions / 100.0

        rows.append({
            "player_id": p.player_id, "season": season, "team_id": tid,
            "minutes_per_game": round(float(minutes), 1),
            "points_per_game": round(float(ppg), 1),
            "games": int(rng.integers(24, 34)),
        })
    return pd.DataFrame(rows)


def _corrupt(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    """Inject the data problems a real multi-source feed would have."""
    df = df.copy()
    truth = {"duplicates": [], "impossible_minutes": [], "negative_points": [],
             "missing_points": []}

    dup_idx = rng.choice(df.index, size=6, replace=False)
    truth["duplicates"] = df.loc[dup_idx, "player_id"].tolist()
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    for i in rng.choice(df.index, size=5, replace=False):
        df.loc[i, "minutes_per_game"] = float(rng.uniform(41, 55))   # > game length
        truth["impossible_minutes"].append(str(df.loc[i, "player_id"]))

    for i in rng.choice(df.index, size=4, replace=False):
        df.loc[i, "points_per_game"] = -abs(float(rng.uniform(1, 5)))
        truth["negative_points"].append(str(df.loc[i, "player_id"]))

    for i in rng.choice(df.index, size=7, replace=False):
        df.loc[i, "points_per_game"] = np.nan
        truth["missing_points"].append(str(df.loc[i, "player_id"]))

    return df, truth


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    teams = _make_teams(rng)
    players = pd.DataFrame({
        "player_id": [f"P{i:04d}" for i in range(N_PLAYERS)],
        "true_talent": rng.normal(0, 1, N_PLAYERS).round(3),
    })

    team_s1 = rng.choice(teams.team_id, N_PLAYERS)
    transferred = rng.random(N_PLAYERS) < TRANSFER_RATE
    team_s2 = np.where(transferred, rng.choice(teams.team_id, N_PLAYERS), team_s1)

    s1 = _season_rows(players, teams, team_s1, 2024, rng)
    s2 = _season_rows(players, teams, team_s2, 2025, rng)
    seasons = pd.concat([s1, s2], ignore_index=True)

    dirty, corruption = _corrupt(seasons, rng)

    teams.to_csv(DATA / "teams.csv", index=False)
    dirty.to_csv(DATA / "player_seasons.csv", index=False)
    players.assign(transferred=transferred).to_csv(DATA / "ground_truth.csv", index=False)
    pd.DataFrame([{"issue": k, "n": len(v)} for k, v in corruption.items()]) \
        .to_csv(DATA / "corruption_log.csv", index=False)

    print(f"{N_PLAYERS} players x 2 seasons ({len(seasons)} clean rows -> "
          f"{len(dirty)} rows after corruption)")
    print(f"transfers: {int(transferred.sum())} ({transferred.mean():.0%})")
    print("injected issues:", {k: len(v) for k, v in corruption.items()})


if __name__ == "__main__":
    main()
