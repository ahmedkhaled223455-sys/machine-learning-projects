import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_metrics import (QUANTILES, Z, apply_shift, combine_ensemble,  # noqa: E402
                        fit_quantile_shift, pinball, score)


def _gaussian_forecast(n=20000, h=4, bias=0.0, sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, h)) * sd
    q = np.broadcast_to(bias + Z, (n, h, len(QUANTILES))).copy()
    return y, q


def test_pinball_hand_example():
    y = np.array([[1.0]])
    q = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0]]])
    # loss_tau = tau * 1 for each quantile -> mean of taus = 0.5
    assert np.isclose(pinball(y, q), QUANTILES.mean())


def test_calibrated_forecast_scores_calibrated():
    y, q = _gaussian_forecast()
    s = score(y, q)
    assert abs(s["coverage90"] - 0.90) < 0.01
    assert s["calib_error"] < 0.01


def test_shift_recalibration_removes_bias():
    y_val, q_val = _gaussian_forecast(bias=0.8, seed=1)
    y_te, q_te = _gaussian_forecast(bias=0.8, seed=2)
    before = score(y_te, q_te)
    after = score(y_te, apply_shift(q_te, fit_quantile_shift(y_val, q_val)))
    assert before["calib_error"] > 0.1
    assert after["calib_error"] < 0.02
    assert after["pinball"] < before["pinball"]


def test_agreeing_ensemble_matches_member():
    _, q = _gaussian_forecast(n=10)
    members = np.stack([q, q, q])
    assert np.allclose(combine_ensemble(members), q, atol=1e-3)


def test_disagreeing_ensemble_is_wider():
    _, q = _gaussian_forecast(n=10)
    members = np.stack([q - 2.0, q, q + 2.0])
    comb = combine_ensemble(members)
    assert (comb[..., 4] - comb[..., 0]).mean() > (q[..., 4] - q[..., 0]).mean() * 1.5
