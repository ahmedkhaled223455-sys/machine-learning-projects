import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_metrics import QUANTILES, Z  # noqa: E402
from retrain_policy import (CalibrationMonitor, ModelRegistry,  # noqa: E402
                            promotion_decision)


def _stream(n=3000, h=4, bias_from=None, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, h))
    if bias_from is not None:
        y[bias_from:] += 3.0  # truth moves away from the forecasts
    q = np.broadcast_to(Z, (n, h, len(QUANTILES))).copy()
    return y, q


def test_monitor_quiet_when_calibrated():
    y, q = _stream()
    assert CalibrationMonitor().run(y, q)["trigger_at"] is None


def test_monitor_triggers_after_drift():
    y, q = _stream(bias_from=1500)
    trig = CalibrationMonitor().run(y, q)["trigger_at"]
    assert trig is not None and 1500 < trig <= 2300


GOOD = {"in_dist": {"pinball": 0.020, "coverage90": 0.90}}
CHAMP = {"in_dist": {"pinball": 0.025, "coverage90": 0.90}}


def test_promote_better_and_calibrated():
    assert promotion_decision(GOOD, CHAMP)["promote"]


def test_reject_miscalibrated_even_if_sharper():
    bad = {"in_dist": {"pinball": 0.010, "coverage90": 0.70}}
    d = promotion_decision(bad, CHAMP)
    assert not d["promote"] and "coverage" in d["reasons"][0]


def test_reject_not_better():
    same = {"in_dist": {"pinball": 0.0249, "coverage90": 0.90}}
    assert not promotion_decision(same, CHAMP)["promote"]


def test_registry_promote_and_rollback(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register("v1", CHAMP)
    reg.promote("v1")
    reg.register("v2", GOOD)
    reg.promote("v2")
    assert reg.state["live"] == "v2"
    assert ModelRegistry(tmp_path / "reg.json").state["live"] == "v2"  # persisted
    assert reg.rollback() == "v1"
    assert reg.rollback() is None and reg.state["live"] == "v1"
