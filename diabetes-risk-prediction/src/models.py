"""Classification models implemented from scratch with NumPy.

Includes:
- LogisticRegressionGD: L2-regularized logistic regression trained by
  full-batch gradient descent.
- DecisionTree: CART-style binary tree with vectorized split search
  (Gini impurity).
- RandomForest: bagged decision trees with per-split feature subsampling.

No scikit-learn is used anywhere in this project; the goal is to make
every modeling decision explicit and inspectable.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
class LogisticRegressionGD:
    """L2-regularized logistic regression via full-batch gradient descent.

    Expects standardized inputs (zero mean, unit variance per feature);
    the learning rate below is tuned for that scale.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 2000, l2: float = 1e-3,
                 seed: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # Clip to avoid overflow in exp for extreme logits.
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y                       # dL/dz for log-loss
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history.append(loss + 0.5 * self.l2 * np.sum(self.w ** 2))
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Decision tree (CART, Gini impurity)
# ---------------------------------------------------------------------------
class DecisionTree:
    """Binary classification tree with vectorized best-split search.

    For each candidate feature the samples are sorted once and the Gini
    impurity of every possible threshold is computed with cumulative sums,
    so split search is O(n log n) per feature rather than O(n^2).
    """

    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 5,
                 max_features: str | int | None = None, seed: int = 0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.tree_ = None

    # -- split search -------------------------------------------------------
    def _best_split(self, X, y):
        n, d = X.shape
        if self.max_features is None:
            feat_idx = np.arange(d)
        else:
            k = (int(np.sqrt(d)) if self.max_features == "sqrt"
                 else int(self.max_features))
            k = max(1, min(k, d))
            feat_idx = self.rng.choice(d, size=k, replace=False)

        best = None  # (weighted_gini, feature, threshold)
        min_leaf = self.min_samples_leaf
        for j in feat_idx:
            order = np.argsort(X[:, j], kind="mergesort")
            xs, ys = X[order, j], y[order]
            cum_pos = np.cumsum(ys)
            total_pos = cum_pos[-1]

            n_left = np.arange(1, n)                # samples in left child
            pos_left = cum_pos[:-1]
            n_right = n - n_left
            pos_right = total_pos - pos_left

            valid = (xs[1:] != xs[:-1]) & (n_left >= min_leaf) & (n_right >= min_leaf)
            if not valid.any():
                continue

            pl = pos_left / n_left
            pr = pos_right / n_right
            gini_left = 1.0 - pl ** 2 - (1.0 - pl) ** 2
            gini_right = 1.0 - pr ** 2 - (1.0 - pr) ** 2
            weighted = (n_left * gini_left + n_right * gini_right) / n
            weighted[~valid] = np.inf

            i = int(np.argmin(weighted))
            if best is None or weighted[i] < best[0]:
                thr = 0.5 * (xs[i] + xs[i + 1])
                best = (weighted[i], int(j), float(thr))
        return best

    # -- recursive construction --------------------------------------------
    def _build(self, X, y, depth):
        node = {"n": len(y), "p": float(y.mean()) if len(y) else 0.0}
        if (depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf
                or node["p"] in (0.0, 1.0)):
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        _, j, thr = split
        mask = X[:, j] <= thr
        node.update({
            "feature": j, "threshold": thr,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        })
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.tree_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x):
        node = self.tree_
        while "feature" in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["p"]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------
class RandomForest:
    """Bootstrap-aggregated decision trees (majority soft vote)."""

    def __init__(self, n_trees: int = 100, max_depth: int = 6,
                 min_samples_leaf: int = 5, seed: int = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for t in range(self.n_trees):
            idx = rng.integers(0, n, size=n)      # bootstrap sample
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features="sqrt", seed=self.seed + t + 1)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
