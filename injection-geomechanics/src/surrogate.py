"""Surrogates for the maximum safe injection rate.

* `RidgeSurrogate` -- generic regression (standardised features plus
  squares and pairwise products) on whatever features it is given.
* `physics_estimate` -- closed-form estimate of the safe rate: Theis while
  infinite-acting, pseudo-steady state once the closed boundary is felt,
  and a Hawkins skin for the near-well zone. It is imperfect in the
  transition between regimes and in how the near-well zone affects the
  far field.
* The hybrid model = physics estimate x a learned correction factor, fit
  on dimensionless groups (the same idea as a physics-hybrid forecaster:
  learn only what the simple physics misses).
"""

from __future__ import annotations

import numpy as np

from expint import exp1
from radial_flow import C_TOTAL, MU_BRINE, R_WELL, diffusivity


def poly_features(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    cross = [X[:, i] * X[:, j] for i in range(n) for j in range(i, n)]
    return np.column_stack([X] + cross)


class RidgeSurrogate:
    def __init__(self, alpha: float = 1e-3, quadratic: bool = True):
        self.alpha, self.quadratic = alpha, quadratic

    def _design(self, X):
        Z = (X - self.mu) / self.sd
        F = poly_features(Z) if self.quadratic else Z
        return np.column_stack([np.ones(len(F)), F])

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        D = self._design(X)
        reg = self.alpha * len(D) * np.eye(D.shape[1])
        reg[0, 0] = 0.0
        self.w = np.linalg.solve(D.T @ D + reg, D.T @ y)
        return self

    def predict(self, X):
        return self._design(X) @ self.w


PSS_SWITCH = 0.2  # dimensionless time D t / Re^2 where the boundary takes over;
                  # calibrated against the simulator on a separate calibration set


def physics_unit_dp(r, t, k_outer, k_inner, r_inner, h, phi, r_e, mu=MU_BRINE, ct=C_TOTAL):
    """Closed-form unit-rate pressure (Pa per m3/s) at radius r.

    Infinite-acting (Theis) while the pressure front has not felt the closed
    boundary; pseudo-steady state for a bounded circular reservoir afterwards:
        dp = mu/(2 pi k h) [2 D t / Re^2 + ln(Re/r) - 3/4 + r^2 / (2 Re^2)]
    At the well, a Hawkins skin s = (k_outer/k_inner - 1) ln(r_inner / r_w)
    represents the near-well damaged or stimulated zone.
    """
    r = np.asarray(r, float)
    D = diffusivity(k_outer, phi, mu, ct)
    base = mu / (2 * np.pi * k_outer * h)
    theis = 0.5 * base * exp1(r ** 2 / (4 * D * t))
    pss = base * (2 * D * t / r_e ** 2 + np.log(r_e / r) - 0.75 + r ** 2 / (2 * r_e ** 2))
    dp = np.where(D * t / r_e ** 2 < PSS_SWITCH, theis, pss)
    at_well = np.isclose(r, R_WELL)
    skin = (k_outer / k_inner - 1.0) * np.log(np.maximum(r_inner, R_WELL) / R_WELL)
    return np.where(at_well, dp + base * skin, dp)


def physics_estimate(t, r_fault, dp_slip, dp_frac_allow, k_outer, k_inner, r_inner, h, phi, r_e):
    """Safe rate (m3/s) from the closed-form physics: min over fault and wellbore limits."""
    q_fault = dp_slip / physics_unit_dp(r_fault, t, k_outer, k_inner, r_inner, h, phi, r_e)
    q_well = dp_frac_allow / physics_unit_dp(np.full_like(r_fault, R_WELL), t, k_outer,
                                             k_inner, r_inner, h, phi, r_e)
    return np.minimum(q_fault, q_well)
