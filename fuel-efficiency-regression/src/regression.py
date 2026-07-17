"""Linear models and evaluation utilities implemented from scratch.

- LinearRegression: ordinary least squares via the pseudo-inverse
  (numerically safer than inverting X'X directly).
- RidgeRegression: closed-form L2-regularized solution; the intercept
  is deliberately NOT penalized.
- kfold_indices / regression_metrics: cross-validation and scoring.
"""

from __future__ import annotations

import numpy as np


class LinearRegression:
    def fit(self, X, y):
        Xb = np.column_stack([np.ones(len(X)), X])
        self.coef_ = np.linalg.pinv(Xb) @ y
        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(X)), X])
        return Xb @ self.coef_


class RidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xb = np.column_stack([np.ones(len(X)), X])
        d = Xb.shape[1]
        P = np.eye(d)
        P[0, 0] = 0.0                      # do not penalize the intercept
        self.coef_ = np.linalg.solve(Xb.T @ Xb + self.alpha * P, Xb.T @ y)
        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(X)), X])
        return Xb @ self.coef_


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot,
    }


def kfold_indices(n: int, k: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for chunk in np.array_split(perm, k):
        val = np.sort(chunk)
        train = np.setdiff1d(np.arange(n), val)
        yield train, val


class Standardizer:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.std_
