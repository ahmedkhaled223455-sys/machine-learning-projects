"""Classical baselines every learned forecaster must beat."""

from __future__ import annotations

import numpy as np

from traj_dataset import F_STEPS
from traj_sim import DT

_T = DT * np.arange(1, F_STEPS + 1)  # future times (s)


def constant_velocity(hist: np.ndarray, fit_steps: int = 5) -> np.ndarray:
    """Average velocity over the last `fit_steps` steps, extrapolated."""
    xy = hist[:, 0, :, :2]
    v = (xy[:, -1] - xy[:, -1 - fit_steps]) / (fit_steps * DT)
    return xy[:, -1][:, None] + v[:, None] * _T[None, :, None]


def constant_acceleration(hist: np.ndarray, fit_steps: int = 10) -> np.ndarray:
    """Least-squares quadratic fit to the last `fit_steps` positions."""
    xy = hist[:, 0, -fit_steps:, :2]                 # (N, S, 2)
    t = DT * (np.arange(fit_steps) - (fit_steps - 1))  # ends at t = 0
    X = np.stack([np.ones_like(t), t, 0.5 * t ** 2], 1)  # (S, 3)
    coef, *_ = np.linalg.lstsq(X, xy.transpose(1, 0, 2).reshape(fit_steps, -1), rcond=None)
    coef = coef.reshape(3, len(xy), 2)              # (3, N, 2)
    Tf = np.stack([np.ones_like(_T), _T, 0.5 * _T ** 2], 1)  # (F, 3)
    return np.einsum("fc,cnd->nfd", Tf, coef)


class RidgeForecaster:
    """Linear map from the target's own history to its future (no neighbours)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    @staticmethod
    def _features(hist):
        return np.concatenate([hist[:, 0].reshape(len(hist), -1), np.ones((len(hist), 1))], 1)

    def fit(self, hist, fut):
        X = self._features(hist)
        Y = fut.reshape(len(fut), -1)
        reg = self.alpha * np.eye(X.shape[1])
        reg[-1, -1] = 0.0
        self.W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
        return self

    def predict(self, hist):
        return (self._features(hist) @ self.W).reshape(len(hist), F_STEPS, 2)
