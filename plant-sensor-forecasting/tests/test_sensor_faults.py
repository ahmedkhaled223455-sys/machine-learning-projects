import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_faults import clean_window, inject_faults  # noqa: E402


def _signal(n=32, seed=0):
    rng = np.random.default_rng(seed)
    return 2.0 + 0.01 * np.arange(n) + 0.02 * rng.standard_normal(n)


def test_clean_signal_is_left_alone():
    x = _signal()
    assert np.allclose(clean_window(x), x)


def test_spike_removed():
    x = _signal()
    bad = x.copy()
    bad[20] += 1.0
    out = clean_window(bad)
    assert abs(out[20] - x[20]) < 0.1


def test_spike_in_last_position_removed():
    x = _signal()
    bad = x.copy()
    bad[-1] -= 1.2
    assert abs(clean_window(bad)[-1] - x[-1]) < 0.1


def test_stuck_run_replaced_by_interpolation():
    x = _signal()
    bad = x.copy()
    bad[10:22] = bad[10]
    out = clean_window(bad)
    assert np.abs(out[10:22] - x[10:22]).max() < 0.08


def test_dropouts_filled():
    x = _signal()
    bad = x.copy()
    bad[[3, 4, 15, 31]] = np.nan
    out = clean_window(bad)
    assert np.isfinite(out).all()
    assert np.abs(out - x).max() < 0.08


def test_injection_logs_and_drift():
    x = _signal(2000)
    bad, log = inject_faults(x, seed=1, drift_total=0.4)
    assert len(log["spike"]) == 160 and len(log["stuck"]) == 20
    assert np.isnan(bad).any()
    # drift reaches the requested size at the end
    assert np.nanmean(bad[-50:] - x[-50:]) > 0.3
