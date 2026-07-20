"""Integration test: the monitor recovers the injected defects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import generate_prices
import monitor

ROOT = Path(__file__).resolve().parents[1]


def test_monitor_recovers_injected_defects():
    generate_prices.main()                      # writes data/ deterministically
    df = monitor.load_prices()
    gt = json.loads((ROOT / "data" / "ground_truth.json").read_text())
    flags = monitor.run_health_check(df)
    ev = monitor.evaluate(df, flags, gt)

    # structural checks are exact; anomaly recall should be perfect here too
    assert ev["missing"]["recall"] == 1.0
    assert ev["stale"]["recall"] == 1.0
    assert ev["date_gaps"]["recall"] == 1.0
    assert ev["outlier"]["recall"] == 1.0

    # regime break: right ticker, no false alarms on the others
    assert ev["regime"]["correct_ticker"] is True
    assert ev["regime"]["no_false_regimes"] is True
