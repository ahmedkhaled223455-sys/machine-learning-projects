"""End-to-end transfer-portal analysis.

  1. integrity-check and clean the multi-source player-season feed
  2. adjust raw box-score averages for pace, role, and schedule
  3. show which measure actually tracks underlying talent
  4. project season-2 production and compare against the naive baseline,
     overall and for the players who transferred
  5. surface the transfers whose raw numbers most understate them

Run:  python src/generate_data.py && python src/run_analysis.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import data_checks
import model
from adjust import add_adjusted_metrics, add_context
from paths import DATA, RESULTS

S1, S2 = 2024, 2025


def load():
    seasons = pd.read_csv(DATA / "player_seasons.csv")
    teams = pd.read_csv(DATA / "teams.csv")
    truth = pd.read_csv(DATA / "ground_truth.csv")
    return seasons, teams, truth


def main():
    RESULTS.mkdir(exist_ok=True)
    seasons, teams, truth = load()

    # --- 1. integrity ---------------------------------------------------
    print(data_checks.report(seasons))
    clean, stats = data_checks.clean(seasons)
    print(f"  cleaned: {stats['rows_in']} -> {stats['rows_out']} rows "
          f"({stats['rows_removed']} removed)\n")

    # --- 2. context adjustment ------------------------------------------
    df = add_adjusted_metrics(add_context(clean, teams))
    s1 = df[df.season == S1].merge(truth, on="player_id")
    s2 = df[df.season == S2].merge(truth, on="player_id")

    # --- 3. which measure tracks talent? --------------------------------
    corr_raw = model.correlation(s1["points_per_game"], s1["true_talent"])
    corr_adj = model.correlation(s1["adjusted_efficiency"], s1["true_talent"])
    print("Which season-1 measure tracks underlying talent?")
    print(f"  points per game        r = {corr_raw:.3f}")
    print(f"  adjusted efficiency    r = {corr_adj:.3f}")
    print(f"  -> adjustment improves correlation by {corr_adj - corr_raw:+.3f}\n")

    # --- 4. project season 2 --------------------------------------------
    pair = s1.merge(s2, on="player_id", suffixes=("_s1", "_s2"))
    prior = pair.rename(columns={"points_per_game_s1": "points_per_game",
                                 "adjusted_efficiency_s1": "adjusted_efficiency"})
    new_ctx = pair.rename(columns={"pace_s2": "pace", "sos_s2": "sos",
                                   "minutes_per_game_s2": "minutes_per_game"})
    actual = pair["points_per_game_s2"]

    base_pred = model.baseline_projection(prior)
    # Honest variant: adjust for the new pace and schedule, but assume the
    # player keeps his old role (no future information used).
    same_role_ctx = new_ctx.copy()
    same_role_ctx["minutes_per_game"] = pair["minutes_per_game_s1"].to_numpy()
    role_agnostic_pred = model.context_projection(prior, same_role_ctx)
    # Upper bound: the coach's intended role at the new program is known.
    ctx_pred = model.context_projection(prior, new_ctx)

    transferred = pair["transferred_s1"].to_numpy(dtype=bool)
    results = {}
    for label, mask in (("all players", slice(None)), ("transfers only", transferred),
                        ("stayers only", ~transferred)):
        r = {
            "n": int(len(actual[mask])),
            "baseline_rmse": model.rmse(actual[mask], base_pred[mask]),
            "pace_sos_only_rmse": model.rmse(actual[mask], role_agnostic_pred[mask]),
            "known_role_rmse": model.rmse(actual[mask], ctx_pred[mask]),
        }
        r["improvement_pct"] = 100 * (1 - r["pace_sos_only_rmse"] / r["baseline_rmse"])
        r["improvement_known_role_pct"] = 100 * (1 - r["known_role_rmse"] / r["baseline_rmse"])
        results[label] = r

    print("Projecting season-2 scoring (RMSE, points per game):")
    print(f"{'group':16s}{'n':>5s}{'baseline':>10s}{'pace+SOS':>10s}"
          f"{'better':>8s}{'+known role':>13s}{'better':>8s}")
    for label, r in results.items():
        print(f"{label:16s}{r['n']:5d}{r['baseline_rmse']:10.2f}"
              f"{r['pace_sos_only_rmse']:10.2f}{r['improvement_pct']:7.1f}%"
              f"{r['known_role_rmse']:13.2f}{r['improvement_known_role_pct']:7.1f}%")
    print("  (pace+SOS uses no future information; known-role additionally assumes")
    print("   the intended minutes at the new program, which a coach would specify)")

    # --- 5. undervalued transfers ---------------------------------------
    s1_transfers = s1[s1["transferred"]]
    board = model.undervalued(s1_transfers, top_n=6)
    print("\nMost undervalued transfers (raw scoring understates them):")
    print(f"{'player':9s}{'ppg':>6s}{'min':>6s}{'pace':>7s}{'sos':>7s}"
          f"{'adj_eff':>9s}{'rank gap':>10s}")
    for r in board.itertuples():
        print(f"{r.player_id:9s}{r.points_per_game:6.1f}{r.minutes_per_game:6.1f}"
              f"{r.pace:7.1f}{r.sos:7.2f}{r.adjusted_efficiency:9.1f}"
              f"{r.rank_gap:10.0f}")

    make_figures(s1, pair, base_pred, ctx_pred, transferred)

    metrics = {
        "integrity": stats,
        "talent_correlation": {"points_per_game": corr_raw,
                               "adjusted_efficiency": corr_adj},
        "projection_rmse": results,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    board.to_csv(RESULTS / "undervalued_transfers.csv", index=False)
    print(f"\nSaved metrics, board, and figures to {RESULTS}")


def make_figures(s1, pair, base_pred, ctx_pred, transferred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(s1["true_talent"], s1["points_per_game"], s=10,
                    alpha=0.5, color="#8896AB")
    axes[0].set_xlabel("true talent (unobserved)")
    axes[0].set_ylabel("points per game")
    axes[0].set_title(f"Raw scoring vs talent "
                      f"(r={model.correlation(s1['points_per_game'], s1['true_talent']):.2f})")
    axes[1].scatter(s1["true_talent"], s1["adjusted_efficiency"], s=10,
                    alpha=0.5, color="#31708E")
    axes[1].set_xlabel("true talent (unobserved)")
    axes[1].set_ylabel("adjusted efficiency (pts/100, schedule-adjusted)")
    axes[1].set_title(f"Adjusted efficiency vs talent "
                      f"(r={model.correlation(s1['adjusted_efficiency'], s1['true_talent']):.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "talent_recovery.png", dpi=150)
    plt.close(fig)

    actual = pair["points_per_game_s2"]
    base_err = (base_pred - actual)[transferred]
    ctx_err = (pd.Series(ctx_pred, index=actual.index) - actual)[transferred]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(base_err, bins=30, alpha=0.6, label="baseline (carry PPG forward)",
            color="#B1624E")
    ax.hist(ctx_err, bins=30, alpha=0.6, label="context-adjusted projection",
            color="#31708E")
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("projection error (predicted - actual PPG)")
    ax.set_ylabel("transfers")
    ax.set_title("Projecting transfers into a new program")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "transfer_projection_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
