"""Simulator, dataset, metrics, baselines, and NumPy attention reference."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention_ref import multihead_attention, softmax  # noqa: E402
from traj_baselines import (RidgeForecaster, constant_acceleration,  # noqa: E402
                            constant_velocity)
from traj_dataset import (F_STEPS, H_STEPS, MAX_NEIGHBORS,  # noqa: E402
                          build_dataset)
from traj_metrics import min_ade, min_fde, miss_rate, top1  # noqa: E402
from traj_sim import CAR_LEN, DT, LANE_W, N_LANES, SimConfig, simulate  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    return simulate(SimConfig(), np.random.default_rng(0))


@pytest.fixture(scope="module")
def data():
    return build_dataset(8, seed=0)


# ---------------- simulator ----------------

def test_scene_shapes_and_bounds(scene):
    pos = scene["pos"]
    cfg = SimConfig()
    assert pos.shape == (cfg.steps, cfg.n_cars, 2)
    assert np.isfinite(pos).all()
    assert pos[..., 1].min() >= -1e-9 and pos[..., 1].max() <= (N_LANES - 1) * LANE_W + 1e-9
    speeds = np.diff(pos[..., 0], axis=0) / DT
    assert speeds.min() >= 0 and speeds.max() < 40


def test_cars_settle_on_lane_centres(scene):
    y = scene["pos"][..., 1]
    off = np.abs(y / LANE_W - np.round(y / LANE_W))
    steady = ~scene["lane_change"]
    assert off[steady].max() < 1e-9


def test_lane_changes_happen_and_move_one_lane():
    rng = np.random.default_rng(1)
    moved = 0
    for _ in range(5):
        sc = simulate(SimConfig(), rng)
        y = sc["pos"][..., 1]
        steps = np.abs(np.diff(np.round(y / LANE_W), axis=0))
        assert steps.max() <= 1  # never jumps more than one lane per step
        moved += int(sc["lane_change"].any(0).sum())
    assert moved > 0


def test_no_collisions_in_steady_lanes(scene):
    pos, lc = scene["pos"], scene["lane_change"]
    lanes = np.round(pos[..., 1] / LANE_W)
    for t in range(0, len(pos), 10):
        for ln in range(N_LANES):
            idx = np.where((lanes[t] == ln) & ~lc[t])[0]
            xs = np.sort(pos[t, idx, 0])
            if len(xs) > 1:
                assert (np.diff(xs) - CAR_LEN).min() > 0


# ---------------- dataset ----------------

def test_dataset_shapes(data):
    n = len(data["fut"])
    assert data["hist"].shape == (n, 1 + MAX_NEIGHBORS, H_STEPS, 4)
    assert data["fut"].shape == (n, F_STEPS, 2)
    assert data["mask"][:, 0].all()


def test_target_frame_origin_and_padding(data):
    assert np.allclose(data["hist"][:, 0, -1, :2], 0.0)
    pad = ~data["mask"]
    assert np.all(data["hist"][pad] == 0.0)
    # the first future step is one 0.1 s step away (< 4 m at highway speed + noise)
    assert np.linalg.norm(data["fut"][:, 0], axis=-1).max() < 4.0


def test_lane_change_share_is_meaningful(data):
    assert 0.05 < data["lc"].mean() < 0.6


# ---------------- metrics ----------------

def test_metrics_perfect_and_best_of_k():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(10, F_STEPS, 2))
    assert min_ade(true, true) == 0 and min_fde(true, true) == 0 and miss_rate(true, true) == 0
    far = true[:, None] + 10.0
    pred = np.concatenate([far, true[:, None], far], 1)  # correct mode in the middle
    assert min_ade(pred, true) == 0 and miss_rate(pred, true) == 0
    scores = np.tile([0.0, 5.0, 0.0], (10, 1))
    assert np.allclose(top1(pred, scores), true)


def test_miss_rate_threshold():
    true = np.zeros((2, F_STEPS, 2))
    pred = true.copy()
    pred[0, -1, 0] = 1.9
    pred[1, -1, 0] = 2.1
    assert miss_rate(pred, true) == 0.5


# ---------------- baselines ----------------

def _hist_from(fn):
    t = DT * (np.arange(H_STEPS) - (H_STEPS - 1))
    h = np.zeros((1, 1 + MAX_NEIGHBORS, H_STEPS, 4))
    h[0, 0, :, 0], h[0, 0, :, 1] = fn(t)
    tf = DT * np.arange(1, F_STEPS + 1)
    fx, fy = fn(tf)
    return h, np.stack([fx, fy], -1)[None]


def test_constant_velocity_exact_on_straight_line():
    h, fut = _hist_from(lambda t: (25.0 * t, 0.5 * t))
    assert np.allclose(constant_velocity(h), fut)


def test_constant_acceleration_exact_on_parabola():
    h, fut = _hist_from(lambda t: (25.0 * t + 0.5 * 1.2 * t ** 2, -0.3 * t ** 2))
    assert np.allclose(constant_acceleration(h), fut, atol=1e-8)


def test_ridge_beats_constant_velocity_on_simulated_data(data):
    train = build_dataset(20, seed=5)
    ridge = RidgeForecaster(1.0).fit(train["hist"], train["fut"])
    assert min_ade(ridge.predict(data["hist"]), data["fut"]) < min_ade(constant_velocity(data["hist"]), data["fut"])


# ---------------- attention reference ----------------

def test_softmax_and_masking():
    rng = np.random.default_rng(0)
    B, Lq, Lk, d, h = 2, 3, 5, 8, 2
    W = [rng.normal(size=(d, d)) for _ in range(4)]
    b = [rng.normal(size=d) for _ in range(4)]
    mask = np.array([[1, 1, 0, 1, 0], [1, 0, 0, 0, 0]], bool)
    out, w = multihead_attention(rng.normal(size=(B, Lq, d)), rng.normal(size=(B, Lk, d)),
                                 *W, *b, n_heads=h, key_mask=mask)
    assert out.shape == (B, Lq, d)
    assert np.allclose(w.sum(-1), 1.0)
    assert np.all(w[:, :, :, ~mask[0]][0] == 0)
    assert np.allclose(w[1, :, :, 0], 1.0)  # only one valid key
    assert np.allclose(softmax(np.array([1.0, 1.0])), 0.5)
