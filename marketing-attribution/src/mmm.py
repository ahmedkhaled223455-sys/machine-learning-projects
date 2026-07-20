"""Media mix modeling (MMM): attribute sales to channel spend.

MMM regresses a sales time series on channel spend to estimate each
channel's incremental contribution and return on ad spend (ROAS). Two
transforms make it realistic, and both are implemented here:

- adstock (geometric carryover): today's spend keeps working for weeks,
  y_t = x_t + theta * y_{t-1}.
- saturation (diminishing returns): the tenth impression is worth less
  than the first; a Hill curve x^a / (x^a + g^a) bends the response.

The regression is ridge, solved from scratch in closed form
(w = (X'X + lambda I)^-1 X'y) on standardized features -- no scikit-learn.
The adstock rate is chosen per channel by a small grid search that
maximizes fit, as in practice.
"""

from __future__ import annotations

import numpy as np


def adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """Geometric carryover: y_t = x_t + theta * y_{t-1}."""
    y = np.empty_like(x, dtype=float)
    carry = 0.0
    for t in range(len(x)):
        carry = x[t] + theta * carry
        y[t] = carry
    return y


def saturation(x: np.ndarray, half: float, shape: float = 1.0) -> np.ndarray:
    """Hill saturation: x^shape / (x^shape + half^shape), in [0, 1)."""
    x = np.asarray(x, dtype=float)
    xs = np.power(np.maximum(x, 0.0), shape)
    return xs / (xs + half ** shape + 1e-12)


def hill_transform(spend: np.ndarray, theta: float,
                   half: float | None = None) -> np.ndarray:
    """Adstock then saturate. If no half-saturation point is given, use the
    median of the adstocked series -- a self-calibrating choice so the
    saturation curve sits in the informative range of the actual spend."""
    a = adstock(np.asarray(spend, dtype=float), theta)
    if half is None:
        half = float(np.median(a)) or 1.0
    return saturation(a, half)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float):
    """Closed-form ridge on standardized X; intercept not penalized.

    Returns (intercept, coef) in the ORIGINAL feature scale.
    """
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    ybar = y.mean()
    yc = y - ybar
    d = Xs.shape[1]
    w_std = np.linalg.solve(Xs.T @ Xs + lam * np.eye(d), Xs.T @ yc)
    coef = w_std / sigma
    intercept = ybar - mu @ coef
    return intercept, coef


class MediaMixModel:
    def __init__(self, channels: list[str], adstock_grid=None,
                 sat_half: float | None = None, lam: float = 1.0):
        self.channels = list(channels)
        self.adstock_grid = (np.linspace(0.0, 0.8, 9)
                             if adstock_grid is None else np.asarray(adstock_grid))
        self.sat_half = sat_half          # None -> per-channel median (self-calibrating)
        self.lam = lam
        self.theta_: dict[str, float] = {}
        self.intercept_ = 0.0
        self.coef_: np.ndarray | None = None
        self.extra_names_: list[str] = []

    def _transform_channel(self, spend: np.ndarray, theta: float) -> np.ndarray:
        return hill_transform(spend, theta, self.sat_half)

    def fit(self, spend: dict[str, np.ndarray], sales: np.ndarray,
            controls: dict[str, np.ndarray] | None = None) -> "MediaMixModel":
        controls = controls or {}
        self.extra_names_ = list(controls)
        # Pick each channel's adstock by best univariate correlation with sales.
        for ch in self.channels:
            best_theta, best_corr = self.adstock_grid[0], -np.inf
            for theta in self.adstock_grid:
                t = self._transform_channel(spend[ch], theta)
                if t.std() == 0:
                    continue
                corr = abs(np.corrcoef(t, sales)[0, 1])
                if corr > best_corr:
                    best_corr, best_theta = corr, theta
            self.theta_[ch] = float(best_theta)

        X = self._design(spend, controls)
        self.intercept_, self.coef_ = _ridge(X, sales, self.lam)
        return self

    def _design(self, spend, controls) -> np.ndarray:
        cols = [self._transform_channel(spend[ch], self.theta_[ch])
                for ch in self.channels]
        cols += [np.asarray(controls[name], float) for name in self.extra_names_]
        return np.column_stack(cols)

    def contributions(self, spend: dict[str, np.ndarray],
                      controls: dict[str, np.ndarray] | None = None) -> dict[str, float]:
        """Total sales contribution attributed to each channel."""
        controls = controls or {}
        out = {}
        for j, ch in enumerate(self.channels):
            t = self._transform_channel(spend[ch], self.theta_[ch])
            out[ch] = float(self.coef_[j] * t.sum())
        return out

    def contribution_shares(self, spend, controls=None) -> dict[str, float]:
        contrib = self.contributions(spend, controls)
        pos = {k: max(v, 0.0) for k, v in contrib.items()}
        total = sum(pos.values()) or 1.0
        return {k: v / total for k, v in pos.items()}

    def roas(self, spend: dict[str, np.ndarray],
             controls: dict[str, np.ndarray] | None = None) -> dict[str, float]:
        """Return on ad spend: contributed sales per dollar spent."""
        contrib = self.contributions(spend, controls)
        return {ch: contrib[ch] / max(spend[ch].sum(), 1e-9)
                for ch in self.channels}

    def predict(self, spend, controls=None) -> np.ndarray:
        return self.intercept_ + self._design(spend, controls or {}) @ self.coef_

    def r2(self, spend, sales, controls=None) -> float:
        pred = self.predict(spend, controls)
        ss_res = float(((sales - pred) ** 2).sum())
        ss_tot = float(((sales - sales.mean()) ** 2).sum()) or 1.0
        return 1.0 - ss_res / ss_tot
