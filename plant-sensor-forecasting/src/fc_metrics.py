"""Probabilistic forecast scoring and post-hoc calibration.

All forecasts are quantile forecasts with shape (N, H, Q) for the levels
in ``QUANTILES``. Scores:

* mae          -- error of the median forecast (point accuracy)
* pinball      -- mean quantile (pinball) loss over all quantiles; a proper
                  scoring rule, so it rewards honest uncertainty
* coverage90   -- share of truths inside the [5%, 95%] interval (target 0.90)
* width90      -- mean width of that interval (sharpness)
* calib_error  -- mean |observed - nominal| frequency across quantile levels
* infeasible   -- share of forecasts with any quantile outside the tank's
                  physical range [0, L_MAX]
"""

from __future__ import annotations

import numpy as np

from plant_sim import L_MAX

QUANTILES = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
MEDIAN = 2
LO, HI = 0, 4


def pinball(y: np.ndarray, q: np.ndarray, taus: np.ndarray = QUANTILES) -> float:
    diff = y[..., None] - q                       # (N, H, Q)
    loss = np.maximum(taus * diff, (taus - 1.0) * diff)
    return float(loss.mean())


def score(y: np.ndarray, q: np.ndarray) -> dict:
    med = q[..., MEDIAN]
    lo, hi = q[..., LO], q[..., HI]
    observed = (y[..., None] <= q).mean(axis=(0, 1))
    return {
        "mae": float(np.abs(y - med).mean()),
        "pinball": pinball(y, q),
        "coverage90": float(((y >= lo) & (y <= hi)).mean()),
        "width90": float((hi - lo).mean()),
        "calib_error": float(np.abs(observed - QUANTILES).mean()),
        "infeasible": float(((q < 0.0) | (q > L_MAX)).any(axis=(1, 2)).mean()),
    }


def mae_by_horizon(y: np.ndarray, q: np.ndarray) -> list[float]:
    return np.abs(y - q[..., MEDIAN]).mean(axis=0).round(4).tolist()


def fit_quantile_shift(y_val: np.ndarray, q_val: np.ndarray,
                       taus: np.ndarray = QUANTILES) -> np.ndarray:
    """Split-conformal style recalibration: for each horizon h and quantile
    tau, the additive shift that makes the tau-quantile hit tau on held-out
    validation data. Returns shifts of shape (H, Q)."""
    resid = y_val[..., None] - q_val               # (N, H, Q)
    shifts = np.empty(resid.shape[1:])
    for j, tau in enumerate(taus):
        shifts[:, j] = np.quantile(resid[:, :, j], tau, axis=0)
    return shifts


def apply_shift(q: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Apply recalibration shifts and restore quantile ordering."""
    return np.sort(q + shifts[None, :, :], axis=-1)


Z = np.array([-1.6449, -0.6745, 0.0, 0.6745, 1.6449])  # N(0,1) at QUANTILES


def combine_ensemble(member_q: np.ndarray) -> np.ndarray:
    """Fold a deep ensemble (M, N, H, Q) into one quantile forecast.

    Total variance = mean aleatoric variance (each member's own 90%
    interval, converted to a sigma) + epistemic variance (spread of the
    member medians). Members that agree give the average member interval;
    members that disagree -- typically outside the training regime -- give
    a wider one."""
    med = member_q[..., MEDIAN]                                   # (M, N, H)
    sigma_alea = (member_q[..., HI] - member_q[..., LO]) / (2 * 1.6449)
    var = (sigma_alea ** 2).mean(axis=0) + med.var(axis=0)
    return med.mean(axis=0)[..., None] + np.sqrt(var)[..., None] * Z


def from_residuals(point: np.ndarray, resid_val: np.ndarray,
                   taus: np.ndarray = QUANTILES) -> np.ndarray:
    """Turn a point forecaster into a quantile forecaster using the
    empirical per-horizon distribution of its validation residuals."""
    qs = np.quantile(resid_val, taus, axis=0).T     # (H, Q)
    return point[..., None] + qs[None, :, :]
