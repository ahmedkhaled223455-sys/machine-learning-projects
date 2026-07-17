"""K-means clustering and validation indices implemented from scratch.

- KMeans with k-means++ initialization and multiple restarts
- inertia (within-cluster sum of squares) for elbow analysis
- silhouette score (exact, O(n^2)) for model selection
- PCA via SVD for low-dimensional visualization
"""

from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(self, k: int, n_init: int = 10, max_iter: int = 300,
                 tol: float = 1e-6, seed: int = 0):
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _init_pp(self, X, rng):
        """k-means++: spread initial centroids proportionally to squared
        distance from the nearest already-chosen centroid."""
        n = len(X)
        centers = [X[rng.integers(n)]]
        for _ in range(1, self.k):
            d2 = np.min(
                ((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            probs = d2 / d2.sum()
            centers.append(X[rng.choice(n, p=probs)])
        return np.array(centers)

    def _lloyd(self, X, centers):
        for _ in range(self.max_iter):
            d2 = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
            labels = d2.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(self.k):
                pts = X[labels == j]
                if len(pts):
                    new_centers[j] = pts.mean(axis=0)
            shift = np.abs(new_centers - centers).max()
            centers = new_centers
            if shift < self.tol:
                break
        d2 = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        return centers, labels, inertia

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            centers = self._init_pp(X, rng)
            centers, labels, inertia = self._lloyd(X, centers)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        d2 = ((np.asarray(X, float)[:, None, :] - self.centers_[None]) ** 2).sum(-1)
        return d2.argmin(axis=1)


def silhouette_score(X, labels) -> float:
    """Mean silhouette over all samples: (b - a) / max(a, b), where a is
    the mean intra-cluster distance and b the mean distance to the
    nearest other cluster."""
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    sil = np.zeros(n)
    uniq = np.unique(labels)
    for i in range(n):
        same = labels == labels[i]
        n_same = same.sum()
        a = D[i, same].sum() / (n_same - 1) if n_same > 1 else 0.0
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            b = min(b, D[i, labels == c].mean())
        sil[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(sil.mean())


def pca_project(X, n_components: int = 2):
    """Project standardized data onto its top principal components."""
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_explained = (S ** 2) / np.sum(S ** 2)
    return Xc @ Vt[:n_components].T, var_explained[:n_components]
