"""Run the full data-health check over the price panel, score it against
ground truth, and emit a monitoring report.

This is the "automate the analysis for ongoing monitoring" piece: one
call runs every structural check and anomaly detector, summarizes
per-ticker health, and (because the defects are known here) reports
detection recall/precision so the monitor itself is validated.

Run:  python src/generate_prices.py && python src/monitor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import checks
from anomaly import regime_breaks, returns_from_prices, robust_return_flags
from generate_prices import REGIME_TICKER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_prices() -> pd.DataFrame:
    return pd.read_csv(DATA / "prices.csv", index_col="date", parse_dates=True)


def run_health_check(df: pd.DataFrame) -> dict:
    returns = returns_from_prices(df)
    return {
        "missing": checks.missing_values(df),
        "stale": checks.stale_prices(df),
        "range": checks.range_violations(df),
        "date_gaps": checks.date_gaps(df),
        "outliers": robust_return_flags(returns),
        "regimes": regime_breaks(returns),
    }


def _recall(detected: set, injected: list) -> dict:
    inj = {tuple(x) for x in injected}
    hit = len(inj & detected)
    return {"injected": len(inj), "detected": hit,
            "recall": round(hit / len(inj), 3) if inj else 1.0}


def evaluate(df: pd.DataFrame, flags: dict, gt: dict) -> dict:
    missing = _recall(set(flags["missing"]), gt["missing"])
    stale = _recall(set(flags["stale"]), gt["stale"])
    gaps = _recall({(d,) for d in flags["date_gaps"]},
                   [[d] for d in gt["missing_dates"]])

    # Outliers: a bad tick shows up on its day and/or the revert next day.
    idx = list(df.index)
    pos = {str(d.date()): i for i, d in enumerate(idx)}
    flagged = set(flags["outliers"])
    caught = 0
    for tk, d in gt["outlier"]:
        near = {(tk, d)}
        if d in pos and pos[d] + 1 < len(idx):
            near.add((tk, str(idx[pos[d] + 1].date())))
        if near & flagged:
            caught += 1
    n_out = len(gt["outlier"])
    outlier = {"injected": n_out, "detected": caught,
               "recall": round(caught / n_out, 3) if n_out else 1.0,
               "total_flags": len(flagged)}

    detected_regimes = flags["regimes"]
    regime_ok = REGIME_TICKER in detected_regimes
    regime = {"true_ticker": gt["regime_break"]["ticker"],
              "true_date": gt["regime_break"]["date"],
              "detected": detected_regimes,
              "correct_ticker": regime_ok,
              "no_false_regimes": set(detected_regimes) <= {REGIME_TICKER}}
    return {"missing": missing, "stale": stale, "date_gaps": gaps,
            "outlier": outlier, "regime": regime}


def write_report(ev: dict, flags: dict):
    RESULTS.mkdir(exist_ok=True)
    lines = ["# Data-health monitor report", ""]
    for name in ("missing", "stale", "date_gaps", "outlier"):
        e = ev[name]
        lines.append(f"- **{name}**: caught {e['detected']}/{e['injected']} "
                     f"(recall {e['recall']:.0%})")
    r = ev["regime"]
    lines.append(f"- **regime break**: true {r['true_ticker']} @ {r['true_date']}; "
                 f"detected {r['detected']}; "
                 f"correct={r['correct_ticker']}, no false alarms={r['no_false_regimes']}")
    (RESULTS / "health_report.md").write_text("\n".join(lines))
    (RESULTS / "metrics.json").write_text(json.dumps(ev, indent=2))


def make_figure(df: pd.DataFrame, flags: dict):
    RESULTS.mkdir(exist_ok=True)
    returns = returns_from_prices(df)
    r = returns[REGIME_TICKER]
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, r, lw=0.7, color="#31708E")
    reg = flags["regimes"].get(REGIME_TICKER)
    if reg:
        plt.axvline(pd.to_datetime(reg["date"]), color="#B1624E", ls="--",
                    label=f"detected regime break (var ratio {reg['ratio']})")
    plt.title(f"{REGIME_TICKER} daily returns -- volatility regime break")
    plt.ylabel("return"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "regime_break.png", dpi=150)
    plt.close()


def main():
    df = load_prices()
    gt = json.loads((DATA / "ground_truth.json").read_text())
    flags = run_health_check(df)
    ev = evaluate(df, flags, gt)

    print("=== Data-health monitor ===")
    for name in ("missing", "stale", "date_gaps", "outlier"):
        e = ev[name]
        print(f"  {name:11s} recall {e['recall']:.0%} "
              f"({e['detected']}/{e['injected']})")
    r = ev["regime"]
    print(f"  regime      {r['detected']} | correct={r['correct_ticker']} "
          f"no-false-alarms={r['no_false_regimes']}")
    print(f"  outlier flags total: {ev['outlier']['total_flags']}")

    write_report(ev, flags)
    make_figure(df, flags)
    print(f"\nSaved report + figure to {RESULTS}")


if __name__ == "__main__":
    main()
