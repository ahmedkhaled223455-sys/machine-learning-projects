import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_windows import H, W, make_windows, physics_rollout_batch  # noqa: E402
from plant_sim import CV_TRUE, physics_rollout, simulate  # noqa: E402


def test_targets_are_true_future_levels():
    s = simulate(400, seed=0)
    w = make_windows(s)
    t = W  # first window: "now" is index W
    assert np.allclose(w.target[0], s.level_true[t:t + H])
    assert np.allclose(w.past_level[0], s.level_meas[t - W:t])


def test_inputs_do_not_see_the_future():
    s = simulate(400, seed=0)
    base = make_windows(s)
    s.level_meas[W:] += 100.0        # corrupt everything from "now" onward
    s.level_true[W:] += 100.0
    s.disturbance[:] = 99.0
    w = make_windows(s)
    assert np.allclose(w.past_level[0], base.past_level[0])
    assert np.allclose(w.phys[0], base.phys[0])
    assert np.allclose(w.fut_fin[0], base.fut_fin[0])


def test_control_alignment_with_true_physics():
    # true Cv, starting from the true level, no disturbance -> exact match
    s = simulate(400, seed=2)
    s.disturbance[:] = 0.0
    lvl = s.level_true.copy()
    from plant_sim import step_level
    for t in range(1, len(lvl)):
        lvl[t] = step_level(lvl[t - 1], s.fin_true[t - 1], s.u[t - 1], 0.0)
    s.level_true[:] = lvl
    w = make_windows(s)
    start = np.array([lvl[t - 1] for t in range(W, len(lvl) - H)])
    roll = physics_rollout_batch(start, w.fut_fin, w.fut_u, cv=CV_TRUE)
    assert np.allclose(roll, w.target)


def test_batch_rollout_matches_scalar():
    s = simulate(300, seed=5)
    w = make_windows(s)
    one = physics_rollout(w.past_level[10, -1], w.fut_fin[10], w.fut_u[10])
    assert np.allclose(one, w.phys[10])
