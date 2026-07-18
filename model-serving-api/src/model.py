"""From-scratch logistic regression + preprocessing.

Same model class and data-quality handling as
../diabetes-risk-prediction/src/models.py -- this project's job isn't to
reinvent the model, it's to show how you'd take a from-scratch model like
it and put it behind a real HTTP API. Kept as a standalone copy so this
project runs on its own, without importing across project folders.
"""

from __future__ import annotations

import numpy as np

# Column order the CSV (and every downstream array) uses.
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# Columns where a raw value of 0 encodes "not recorded", not "measured as
# zero" -- the same data-quality issue documented in diabetes-risk-prediction.
# Applied identically at train time and inference time to avoid train/serve
# skew: a client that submits glucose=0 is saying "unknown", not "zero".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ZERO_AS_MISSING_IDX = [FEATURES.index(c) for c in ZERO_AS_MISSING]


class LogisticRegressionGD:
    """L2-regularized logistic regression via full-batch gradient descent.

    Expects standardized inputs (zero mean, unit variance per feature).
    """

    def __init__(self, lr: float = 0.1, epochs: int = 3000, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)


class MedianImputer:
    """Fill NaNs with per-column medians learned on training data only."""

    def fit(self, X):
        self.medians_ = np.nanmedian(np.asarray(X, dtype=float), axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = self.medians_[j]
        return X


class Standardizer:
    """Column-wise z-scoring; parameters learned on training data only."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_


def mark_missing(X: np.ndarray) -> np.ndarray:
    """Replace biologically-impossible zeros with NaN, on a copy."""
    X = np.asarray(X, dtype=float).copy()
    for j in ZERO_AS_MISSING_IDX:
        X[X[:, j] == 0, j] = np.nan
    return X


def stratified_split(y, test_frac: float = 0.2, seed: int = 42):
    """Class-preserving train/test index split."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    test_idx = []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_test = int(round(test_frac * len(idx)))
        test_idx.append(idx[:n_test])
    test_idx = np.sort(np.concatenate(test_idx))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return train_idx, test_idx


def roc_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted, s_sorted = y_true[order], scores[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    distinct = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tps, fps = tps[distinct], fps[distinct]
    P, N = y_true.sum(), len(y_true) - y_true.sum()
    tpr = np.r_[0.0, tps / max(P, 1)]
    fpr = np.r_[0.0, fps / max(N, 1)]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tpr, fpr))
