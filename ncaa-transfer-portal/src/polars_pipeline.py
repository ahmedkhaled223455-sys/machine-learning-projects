"""The same clean-join-adjust pipeline in polars.

polars expressions replace pandas' row-wise thinking with a lazy query plan:
filters, the team join, and the derived rate columns are declared once and
executed in one pass. On a real multi-season scouting feed (millions of
player-game rows) that difference matters; here it's the same result,
written the way a polars codebase would.

Run:  python src/polars_pipeline.py
"""

from __future__ import annotations

import polars as pl

from paths import DATA

MINUTES_CEILING = 40.0
SOS_COEF = 1.6


def build(seasons_csv=None, teams_csv=None) -> pl.DataFrame:
    seasons = pl.read_csv(seasons_csv or (DATA / "player_seasons.csv"))
    teams = pl.read_csv(teams_csv or (DATA / "teams.csv"))

    return (
        seasons.lazy()
        # integrity: drop impossible, negative, and missing rows, then dedupe
        .filter(pl.col("minutes_per_game") <= MINUTES_CEILING)
        .filter(pl.col("points_per_game") >= 0)
        .filter(pl.col("points_per_game").is_not_null())
        .unique(subset=["player_id", "season"], keep="first")
        # context
        .join(teams.lazy(), on="team_id", how="left")
        # rate metrics
        .with_columns(
            (pl.col("pace") * pl.col("minutes_per_game") / 40.0).alias("possessions")
        )
        .with_columns(
            (pl.col("points_per_game") / pl.col("possessions") * 100.0)
            .alias("points_per_100")
        )
        .with_columns(
            (pl.col("points_per_100") + SOS_COEF * pl.col("sos"))
            .alias("adjusted_efficiency")
        )
        .sort("adjusted_efficiency", descending=True)
        .collect()
    )


def main():
    df = build()
    print(f"polars pipeline: {df.height} clean player-seasons, "
          f"{df.width} columns")
    print("\nTop 10 by schedule-adjusted efficiency:")
    top = df.select(["player_id", "season", "team_id", "points_per_game",
                     "minutes_per_game", "pace", "sos", "adjusted_efficiency"]).head(10)
    print(top)

    per_season = (df.group_by("season")
                  .agg(pl.col("adjusted_efficiency").mean().round(2).alias("mean_adj_eff"),
                       pl.col("points_per_game").mean().round(2).alias("mean_ppg"),
                       pl.len().alias("players"))
                  .sort("season"))
    print("\nBy season:")
    print(per_season)


if __name__ == "__main__":
    main()
