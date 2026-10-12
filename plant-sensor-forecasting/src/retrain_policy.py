"""Production guard-rails: calibration monitoring, retraining triggers,
and champion/challenger promotion with rollback.

* ``CalibrationMonitor`` watches live forecasts against the truth that
  arrives later. When rolling 90%-interval coverage stays below a floor,
  it fires a retraining trigger. This is how slow sensor drift -- which the
  input layer cannot repair -- gets caught.
* ``promotion_decision`` only lets a retrained model replace the current
  one if it is better on the backtest *and* keeps calibration within
  tolerance in every scenario; otherwise the champion stays.
* ``ModelRegistry`` records every version and which one is live, so a bad
  rollout can be rolled back to the previous champion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fc_metrics import HI, LO


@dataclass
class CalibrationMonitor:
    window: int = 500          # forecasts per rolling check
    floor: float = 0.80        # minimum acceptable 90%-interval coverage
    consecutive: int = 3       # checks below floor before triggering
    step: int = 100            # evaluate every `step` forecasts

    def run(self, y: np.ndarray, q: np.ndarray) -> dict:
        """y: (N, H) truths, q: (N, H, Q) forecasts, in time order.
        Returns rolling coverage and the first trigger index (or None)."""
        inside = ((y >= q[..., LO]) & (y <= q[..., HI])).mean(axis=1)  # (N,)
        checks, below, trigger = [], 0, None
        for end in range(self.window, len(inside) + 1, self.step):
            cov = float(inside[end - self.window:end].mean())
            checks.append((end, round(cov, 4)))
            below = below + 1 if cov < self.floor else 0
            if trigger is None and below >= self.consecutive:
                trigger = end
        return {"trigger_at": trigger, "checks": checks,
                "min_coverage": min(c for _, c in checks) if checks else None}


def promotion_decision(challenger: dict, champion: dict | None,
                       margin: float = 0.01, coverage_tol: float = 0.05,
                       nominal: float = 0.90) -> dict:
    """challenger / champion: {scenario: {"pinball": .., "coverage90": ..}}.

    Promote only if (a) every scenario's coverage is within ``coverage_tol``
    of nominal and (b) mean pinball improves on the champion by at least
    ``margin`` (relative). With no champion, (a) alone decides."""
    reasons = []
    for scen, m in challenger.items():
        if abs(m["coverage90"] - nominal) > coverage_tol:
            reasons.append(f"{scen}: coverage {m['coverage90']:.3f} outside "
                           f"{nominal}±{coverage_tol}")
    if champion is not None:
        ch = np.mean([m["pinball"] for m in challenger.values()])
        cur = np.mean([champion[s]["pinball"] for s in challenger])
        if ch > cur * (1 - margin):
            reasons.append(f"pinball {ch:.4f} not {margin:.0%} better than "
                           f"champion {cur:.4f}")
    return {"promote": not reasons, "reasons": reasons}


class ModelRegistry:
    """Tiny file-backed registry: versions with metrics, plus a live pointer
    and the history of live versions (for rollback)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"versions": {}, "live": None, "history": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))

    def register(self, version: str, metrics: dict) -> None:
        self.state["versions"][version] = metrics
        self._save()

    def live_metrics(self) -> dict | None:
        live = self.state["live"]
        return self.state["versions"][live] if live else None

    def promote(self, version: str) -> None:
        if self.state["live"]:
            self.state["history"].append(self.state["live"])
        self.state["live"] = version
        self._save()

    def rollback(self) -> str | None:
        if not self.state["history"]:
            return None
        self.state["live"] = self.state["history"].pop()
        self._save()
        return self.state["live"]
