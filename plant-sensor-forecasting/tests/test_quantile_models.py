"""PyTorch model tests. Skipped automatically where torch is not installed
(the repo-wide lint job); the project's own CI job installs torch."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_metrics import QUANTILES, pinball  # noqa: E402
from fc_windows import H, make_windows  # noqa: E402
from plant_sim import L_MAX, simulate  # noqa: E402
from quantile_models import (QuantileForecaster, QuantileMLP,  # noqa: E402
                             TrainConfig, build_inputs, feasibility_penalty,
                             pinball_loss)


def test_quantiles_never_cross():
    torch.manual_seed(0)
    net = QuantileMLP(10)
    q = net(torch.randn(256, 10) * 5).detach().numpy()
    assert q.shape == (256, H, len(QUANTILES))
    assert (np.diff(q, axis=-1) > 0).all()


def test_torch_pinball_matches_numpy():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((50, H)).astype(np.float32)
    q = np.sort(rng.standard_normal((50, H, len(QUANTILES))), axis=-1).astype(np.float32)
    t = float(pinball_loss(torch.from_numpy(y), torch.from_numpy(q)))
    assert np.isclose(t, pinball(y, q), rtol=1e-5)


def test_feasibility_penalty():
    inside = torch.full((4, H, 5), L_MAX / 2)
    assert float(feasibility_penalty(inside)) == 0.0
    outside = torch.full((4, H, 5), L_MAX + 1.0)
    assert float(feasibility_penalty(outside)) > 0.9


def test_hybrid_inputs_include_physics():
    w = make_windows(simulate(300, seed=0))
    x_pure, base_pure = build_inputs(w, hybrid=False)
    x_hyb, base_hyb = build_inputs(w, hybrid=True)
    assert x_hyb.shape[1] == x_pure.shape[1] + H
    assert np.allclose(base_hyb, w.phys)


def test_training_learns_and_beats_untrained():
    train = make_windows(simulate(4000, seed=1))
    val = make_windows(simulate(1500, seed=2))
    model = QuantileForecaster(hybrid=True, config=TrainConfig(max_epochs=10, batch=128,
                                                                     patience=10, seed=0))
    model.fit(train, val)
    assert model.history[-1] <= model.history[0] or min(model.history) < model.history[0]
    q = model.predict(val)
    assert q.shape == (len(val), H, len(QUANTILES))
    # better than predicting the physics rollout with zero spread
    flat = np.repeat(val.phys[..., None], len(QUANTILES), axis=-1)
    assert pinball(val.target, q) < pinball(val.target, flat)
