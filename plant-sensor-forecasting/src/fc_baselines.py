"""Classical baselines. Every learned model has to beat these to matter.

Each baseline produces a point forecast; ``fc_metrics.from_residuals``
turns it into quantiles using its own validation residuals, so the
comparison with the neural models is probabilistic, not just point-wise.

* persistence  -- the level stays where it was
* holt         -- Holt's linear-trend exponential smoothing on the window
* arx          -- direct multi-horizon linear model on recent sensors and
                  the known future controls (ridge least squares)
* physics      -- first-principles mass balance with the nominal valve
                  coefficient and no disturbance (``plant_sim``)
"""

from __future__ import annotations

import numpy as np

from fc_windows import H, Windows


def persistence(w: Windows) -> np.ndarray:
    return np.repeat(w.past_level[:, -1:], H, axis=1)


def holt(w: Windows, alpha: float = 0.5, beta: float = 0.1) -> np.ndarray:
    x = w.past_level
    level = x[:, 0].copy()
    trend = x[:, 1] - x[:, 0]
    for t in range(1, x.shape[1]):
        prev = level
        level = alpha * x[:, t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    steps = np.arange(1, H + 1)[None, :]
    return level[:, None] + trend[:, None] * steps


def _arx_features(w: Windows) -> np.ndarray:
    last = w.past_level[:, -1:]
    return np.hstack([
        np.ones((len(w), 1)),
        last,
        w.past_level[:, -8:] - last,
        w.past_fin[:, -8:],
        w.fut_fin,
        w.fut_u,
    ])


class ARX:
    """Direct multi-horizon ridge regression predicting change from the
    last observed level."""

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self.coef = None

    def fit(self, w: Windows) -> "ARX":
        X = _arx_features(w)
        Y = w.target - w.past_level[:, -1:]
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.coef = np.linalg.solve(A, X.T @ Y)
        return self

    def predict(self, w: Windows) -> np.ndarray:
        return w.past_level[:, -1:] + _arx_features(w) @ self.coef


def physics(w: Windows) -> np.ndarray:
    return w.phys.copy()
