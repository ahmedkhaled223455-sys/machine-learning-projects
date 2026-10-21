"""Reduced-order model of the pressure field: POD + RBF interpolation.

Snapshots of the (log, normalised) unit-rate pressure profile on a common
radial grid are compressed with proper orthogonal decomposition (SVD);
the few POD coefficients are then interpolated across parameter space
with Gaussian radial basis functions. A new reservoir's full pressure
profile is reconstructed in microseconds instead of a full simulation.
"""

from __future__ import annotations

import numpy as np


class PODRBF:
    def __init__(self, n_modes: int = 8, length_scale: float = 1.0, ridge: float = 1e-8):
        self.n_modes = n_modes
        self.length_scale = length_scale
        self.ridge = ridge

    def fit(self, params: np.ndarray, fields: np.ndarray):
        self.p_mean, self.p_std = params.mean(0), params.std(0)
        X = (params - self.p_mean) / self.p_std
        self.f_mean = fields.mean(0)
        U, s, Vt = np.linalg.svd(fields - self.f_mean, full_matrices=False)
        self.energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        self.modes = Vt[: self.n_modes]                       # (r, n_grid)
        coef = (fields - self.f_mean) @ self.modes.T          # (N, r)
        K = self._kernel(X, X)
        self.X = X
        self.W = np.linalg.solve(K + self.ridge * np.eye(len(X)), coef)
        return self

    def _kernel(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * self.length_scale ** 2))

    def coefficients(self, params):
        X = (params - self.p_mean) / self.p_std
        return self._kernel(X, self.X) @ self.W

    def predict(self, params):
        return self.f_mean + self.coefficients(params) @ self.modes

    def project(self, fields):
        """Best possible reconstruction with the retained modes (truncation error only)."""
        c = (fields - self.f_mean) @ self.modes.T
        return self.f_mean + c @ self.modes
