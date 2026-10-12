"""Sensor-fault injection and a robust input layer.

Real plant sensors fail in a handful of recognisable ways. We inject four
of them into the *level sensor* of a test series and then try to repair
the model inputs before forecasting:

* dropout  -- missing readings (NaN)
* spike    -- isolated wild values
* stuck    -- the sensor freezes at one value for a while
* drift    -- a slowly growing offset (calibration drift)

The repair layer handles the first three from the data alone. Drift is
deliberately *not* repaired: a single slowly drifting sensor is
indistinguishable from a real slow change without redundant measurements.
It is detected downstream instead, by the calibration monitor in
``retrain_policy.py``. Ground truth is never touched -- only model inputs.
"""

from __future__ import annotations

import warnings

import numpy as np


def inject_faults(level_meas: np.ndarray, seed: int = 0,
                  dropout_frac: float = 0.03, n_spikes: int = 160,
                  n_stuck: int = 20, stuck_len: int = 30,
                  drift_total: float = 0.0) -> tuple[np.ndarray, dict]:
    """Return a corrupted copy of the level sensor and a fault log."""
    rng = np.random.default_rng(seed)
    x = level_meas.astype(float).copy()
    n = len(x)
    log = {"dropout": [], "spike": [], "stuck": []}

    for _ in range(n_stuck):
        s = int(rng.integers(0, n - stuck_len))
        x[s:s + stuck_len] = x[s]
        log["stuck"].append((s, s + stuck_len))

    spikes = rng.choice(n, size=n_spikes, replace=False)
    x[spikes] += rng.choice([-1.0, 1.0], size=n_spikes) * rng.uniform(0.5, 1.5, n_spikes)
    log["spike"] = sorted(int(i) for i in spikes)

    drop = rng.random(n) < dropout_frac
    x[drop] = np.nan
    log["dropout"] = np.flatnonzero(drop).tolist()

    if drift_total:
        x = x + np.linspace(0.0, drift_total, n)
    return x, log


def _mad(v: np.ndarray) -> float:
    return float(np.median(np.abs(v - np.median(v)))) * 1.4826


def clean_window(window: np.ndarray, spike_z: float = 6.0,
                 stuck_min: int = 6) -> np.ndarray:
    """Repair one input window of level readings.

    1. Spikes: points whose deviation from a rolling median exceeds
       ``spike_z`` robust standard deviations become missing.
    2. Stuck runs: >= ``stuck_min`` identical consecutive readings are
       physically implausible for a noisy sensor, so all but the first
       become missing.
    3. Missing values are linearly interpolated (edge values carried).
    """
    w = window.astype(float).copy()
    n = len(w)

    # stuck detection (exact repeats can only come from a frozen sensor)
    run_start = 0
    for i in range(1, n + 1):
        if i == n or not (w[i] == w[i - 1]):
            if i - run_start >= stuck_min:
                w[run_start + 1:i] = np.nan
            run_start = i

    # spike detection against a centred rolling median
    finite = np.isfinite(w)
    if finite.sum() >= 5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            med = np.array([np.nanmedian(w[max(0, i - 3):i + 4]) for i in range(n)])
        resid = w - med
        scale = _mad(resid[np.isfinite(resid)]) or 1e-6
        w[np.abs(resid) > spike_z * max(scale, 0.01)] = np.nan

    # interpolate
    finite = np.isfinite(w)
    if not finite.any():
        return np.zeros(n)
    idx = np.arange(n)
    return np.interp(idx, idx[finite], w[finite])
