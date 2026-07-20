"""Run all three analyses and validate each against known ground truth.

Because the data was generated with set effects (see generate_data.py),
every method's output is checked against the truth it should recover:
- attribution: does the Markov model's channel credit match the true
  removal effects (and beat last-touch)?
- MMM: do recovered contribution shares and ROAS ranking match the truth?
- lift: does difference-in-differences recover the injected campaign lift,
  with a CI that covers it?

Run:  python src/generate_data.py && python src/run_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attribution import (build_transition_matrix, heuristic_attribution,
                         markov_attribution)
from lift import analyze_campaign
from mmm import MediaMixModel
from generate_data import ATTR_CHANNELS, MMM_CHANNELS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_journeys():
    out = []
    for line in (DATA / "journeys.jsonl").read_text().splitlines():
        d = json.loads(line)
        out.append((d["path"], d["converted"]))
    return out


def _mae(a: dict, b: dict) -> float:
    return float(np.mean([abs(a[k] - b[k]) for k in a]))


def _spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_attribution(truth) -> dict:
    journeys = load_journeys()
    P, states = build_transition_matrix(journeys, ATTR_CHANNELS)
    markov = markov_attribution(P, states, ATTR_CHANNELS)
    models = {m: heuristic_attribution(journeys, ATTR_CHANNELS, m)
              for m in ("first", "last", "linear", "time_decay", "position")}
    models["markov"] = markov
    true = truth["attribution_true_shares"]
    print("\n=== Multi-touch attribution (credit share by channel) ===")
    header = "channel".ljust(9) + "".join(m.rjust(11) for m in
                                           ["true", "last", "markov"])
    print(header)
    for ch in ATTR_CHANNELS:
        print(ch.ljust(9) + f"{true[ch]:11.3f}{models['last'][ch]:11.3f}"
              f"{markov[ch]:11.3f}")
    mae_markov = _mae(markov, true)
    mae_last = _mae(models["last"], true)
    print(f"MAE vs truth  ->  markov {mae_markov:.3f} | last-touch {mae_last:.3f}")
    return {"true": true, "models": models,
            "mae_markov": mae_markov, "mae_last": mae_last}


def run_mmm(truth) -> dict:
    df = pd.read_csv(DATA / "mmm_weekly.csv")
    spend = {ch: df[f"spend_{ch}"].to_numpy() for ch in MMM_CHANNELS}
    sales = df["sales"].to_numpy()
    controls = {"season": df["season"].to_numpy(), "trend": df["trend"].to_numpy()}

    model = MediaMixModel(MMM_CHANNELS).fit(spend, sales, controls)
    shares = model.contribution_shares(spend, controls)
    roas = model.roas(spend, controls)
    r2 = model.r2(spend, sales, controls)

    true_shares = truth["mmm_true_contribution_shares"]
    true_roas = truth["mmm_true_roas"]
    print("\n=== Media mix model (contribution share by channel) ===")
    print("channel".ljust(9) + "true".rjust(9) + "recovered".rjust(11)
          + "ROAS".rjust(9) + "trueROAS".rjust(10))
    for ch in MMM_CHANNELS:
        print(ch.ljust(9) + f"{true_shares[ch]:9.3f}{shares[ch]:11.3f}"
              f"{roas[ch]:9.2f}{true_roas[ch]:10.2f}")
    mae = _mae(shares, true_shares)
    roas_corr = _spearman([true_roas[c] for c in MMM_CHANNELS],
                          [roas[c] for c in MMM_CHANNELS])
    print(f"model R^2 {r2:.3f} | contribution-share MAE {mae:.3f} | "
          f"ROAS rank correlation {roas_corr:.2f}")
    return {"true_shares": true_shares, "recovered_shares": shares,
            "roas": roas, "true_roas": true_roas, "r2": r2, "mae": mae,
            "roas_spearman": roas_corr,
            "theta": model.theta_}


def run_lift(truth) -> dict:
    df = pd.read_csv(DATA / "campaign_geo.csv")
    res = analyze_campaign(df, seed=1)
    true_lift = truth["campaign_true_lift"]
    lo, hi = res["ci95"]
    covers = lo <= true_lift <= hi
    print("\n=== Campaign lift (difference-in-differences) ===")
    print(f"true lift {true_lift:.1f} | DiD estimate {res['did_lift']:.1f} "
          f"| 95% CI [{lo:.1f}, {hi:.1f}] | p={res['p_value']:.4f}")
    print(f"CI covers truth: {covers}")
    return {**res, "true_lift": true_lift, "ci_covers_truth": bool(covers)}


def make_figures(attr, mmm_res, lift_res):
    RESULTS.mkdir(exist_ok=True)
    # attribution: true vs last vs markov
    x = np.arange(len(ATTR_CHANNELS)); w = 0.27
    plt.figure(figsize=(7, 4))
    plt.bar(x - w, [attr["true"][c] for c in ATTR_CHANNELS], w, label="true")
    plt.bar(x, [attr["models"]["last"][c] for c in ATTR_CHANNELS], w, label="last-touch")
    plt.bar(x + w, [attr["models"]["markov"][c] for c in ATTR_CHANNELS], w, label="markov")
    plt.xticks(x, ATTR_CHANNELS); plt.ylabel("credit share")
    plt.title("Attribution: last-touch bias vs. data-driven vs. truth")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "attribution_comparison.png", dpi=150); plt.close()

    # mmm: true vs recovered shares
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, [mmm_res["true_shares"][c] for c in MMM_CHANNELS], w, label="true")
    plt.bar(x + w / 2, [mmm_res["recovered_shares"][c] for c in MMM_CHANNELS], w,
            label="recovered")
    plt.xticks(x, MMM_CHANNELS); plt.ylabel("contribution share")
    plt.title(f"MMM contribution recovery (R^2={mmm_res['r2']:.2f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "mmm_recovery.png", dpi=150); plt.close()


def main():
    truth = json.loads((DATA / "ground_truth.json").read_text())
    attr = run_attribution(truth)
    mmm_res = run_mmm(truth)
    lift_res = run_lift(truth)
    make_figures(attr, mmm_res, lift_res)

    RESULTS.mkdir(exist_ok=True)
    metrics = {
        "attribution": {"mae_markov": attr["mae_markov"],
                        "mae_last": attr["mae_last"]},
        "mmm": {"r2": mmm_res["r2"], "contribution_share_mae": mmm_res["mae"],
                "roas_spearman": mmm_res["roas_spearman"]},
        "lift": {"did_lift": lift_res["did_lift"], "true_lift": lift_res["true_lift"],
                 "p_value": lift_res["p_value"], "ci95": lift_res["ci95"],
                 "ci_covers_truth": lift_res["ci_covers_truth"]},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics + figures to {RESULTS}")


if __name__ == "__main__":
    main()
