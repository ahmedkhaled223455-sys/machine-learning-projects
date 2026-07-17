"""Evaluation utilities implemented from scratch: metrics, ROC/AUC,
stratified splitting and k-fold cross-validation, and standardization.
"""

from __future__ import annotations

import numpy as np

# numpy renamed trapz -> trapezoid in 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# -- basic metrics ----------------------------------------------------------
def confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, fn, tn


def classification_metrics(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# -- ROC / AUC --------------------------------------------------------------
def roc_curve(y_true, scores):
    """FPR/TPR at every distinct score threshold (descending)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    # Keep only the last point of each tied-score block.
    distinct = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tps, fps = tps[distinct], fps[distinct]

    P = y_true.sum()
    N = len(y_true) - P
    tpr = np.r_[0.0, tps / max(P, 1)]
    fpr = np.r_[0.0, fps / max(N, 1)]
    return fpr, tpr


def auc(y_true, scores):
    fpr, tpr = roc_curve(y_true, scores)
    return float(_trapz(tpr, fpr))


# -- splitting --------------------------------------------------------------
def stratified_split(y, test_frac: float = 0.2, seed: int = 42):
    """Class-preserving train/test index split."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        n_test = int(round(test_frac * len(idx)))
        test_idx.append(idx[:n_test])
    test_idx = np.sort(np.concatenate(test_idx))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return train_idx, test_idx


def stratified_kfold(y, k: int = 5, seed: int = 42):
    """Yield (train_idx, val_idx) pairs with per-class balanced folds."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk.tolist())
    all_idx = np.arange(len(y))
    for i in range(k):
        val_idx = np.sort(np.array(folds[i]))
        train_idx = np.setdiff1d(all_idx, val_idx)
        yield train_idx, val_idx


# -- preprocessing ----------------------------------------------------------
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

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MedianImputer:
    """Fill NaNs with per-column medians learned on training data only."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.medians_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = self.medians_[j]
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
