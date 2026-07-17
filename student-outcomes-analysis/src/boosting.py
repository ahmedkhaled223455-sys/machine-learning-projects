"""Gradient-boosted regression trees implemented from scratch.

- RegressionTree: CART-style tree with vectorized best-split search
  (maximizes SSE reduction using cumulative sums over sorted values).
- GradientBoostingRegressor: squared-loss boosting = sequentially fit
  shallow trees to the current residuals, shrink by a learning rate.
- RidgeRegression and k-fold utilities for baselines.
"""

from __future__ import annotations

import numpy as np


class RegressionTree:
    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def _best_split(self, X, y):
        n, d = X.shape
        y_sum = y.sum()
        best = None                     # (gain, feature, threshold)
        base_sse_term = y_sum ** 2 / n  # constant part; gain compares child terms
        for j in range(d):
            order = np.argsort(X[:, j], kind="mergesort")
            xs, ys = X[order, j], y[order]
            cs = np.cumsum(ys)
            n_left = np.arange(1, n)
            sum_left = cs[:-1]
            n_right = n - n_left
            sum_right = y_sum - sum_left
            valid = ((xs[1:] != xs[:-1])
                     & (n_left >= self.min_samples_leaf)
                     & (n_right >= self.min_samples_leaf))
            if not valid.any():
                continue
            # SSE reduction = sum_l^2/n_l + sum_r^2/n_r - sum^2/n
            gain = sum_left ** 2 / n_left + sum_right ** 2 / n_right - base_sse_term
            gain[~valid] = -np.inf
            i = int(np.argmax(gain))
            if gain[i] > 0 and (best is None or gain[i] > best[0]):
                best = (float(gain[i]), int(j), float(0.5 * (xs[i] + xs[i + 1])))
        return best

    def _build(self, X, y, depth):
        node = {"value": float(y.mean()), "n": len(y)}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        _, j, thr = split
        mask = X[:, j] <= thr
        node.update({"feature": j, "threshold": thr,
                     "left": self._build(X[mask], y[mask], depth + 1),
                     "right": self._build(X[~mask], y[~mask], depth + 1)})
        return node

    def fit(self, X, y):
        self.tree_ = self._build(np.asarray(X, float), np.asarray(y, float), 0)
        return self

    def _predict_one(self, x):
        node = self.tree_
        while "feature" in node:
            node = (node["left"] if x[node["feature"]] <= node["threshold"]
                    else node["right"])
        return node["value"]

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._predict_one(x) for x in X])


class GradientBoostingRegressor:
    """Squared-loss gradient boosting: each tree fits the residuals."""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05,
                 max_depth: int = 3, min_samples_leaf: int = 10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.init_ = float(y.mean())
        F = np.full(len(y), self.init_)
        self.trees_ = []
        self.val_curve_ = []
        F_val = (np.full(len(y_val), self.init_)
                 if X_val is not None else None)
        for _ in range(self.n_estimators):
            residual = y - F
            tree = RegressionTree(self.max_depth, self.min_samples_leaf)
            tree.fit(X, residual)
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            if X_val is not None:
                F_val += self.learning_rate * tree.predict(X_val)
                self.val_curve_.append(
                    float(np.sqrt(np.mean((np.asarray(y_val, float) - F_val) ** 2))))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(len(X), self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred


class RidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xb = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        P = np.eye(Xb.shape[1])
        P[0, 0] = 0.0
        self.coef_ = np.linalg.solve(Xb.T @ Xb + self.alpha * P,
                                     Xb.T @ np.asarray(y, float))
        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        return Xb @ self.coef_


class MeanBaseline:
    def fit(self, X, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


def kfold_indices(n: int, k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    for chunk in np.array_split(rng.permutation(n), k):
        val = np.sort(chunk)
        yield np.setdiff1d(np.arange(n), val), val


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(np.mean(np.abs(resid))),
            "r2": 1.0 - float(np.sum(resid ** 2)) / ss_tot}
