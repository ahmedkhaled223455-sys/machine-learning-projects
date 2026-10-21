"""Exponential integral E1(x), the well function of the Theis solution.

Implemented from first principles (no SciPy): power series for x <= 1 and
a continued fraction (modified Lentz) for x > 1. Accurate to ~1e-13
relative error over the range used here; checked against reference values
in the tests.
"""

from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649015329


def _series(x: np.ndarray) -> np.ndarray:
    # E1(x) = -gamma - ln x - sum_{k>=1} (-x)^k / (k * k!)
    total = np.zeros_like(x)
    term = np.ones_like(x)
    for k in range(1, 40):
        term = term * (-x) / k
        total = total + term / k
    return -EULER_GAMMA - np.log(x) - total


def _continued_fraction(x: np.ndarray) -> np.ndarray:
    # E1(x) = exp(-x) * 1/(x + 1 - 1^2/(x + 3 - 2^2/(x + 5 - ...)))
    tiny = 1e-300
    b = x + 1.0
    c = np.full_like(x, 1.0 / tiny)
    d = 1.0 / b
    h = d.copy()
    for i in range(1, 80):
        a = -float(i * i)
        b = b + 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        h = h * c * d
    return h * np.exp(-x)


def exp1(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("E1 is defined here for x > 0")
    out = np.empty_like(x)
    small = x <= 1.0
    out[small] = _series(x[small])
    out[~small] = _continued_fraction(x[~small])
    return out
