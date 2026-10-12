import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_sim import (CV_TRUE, L_MAX, SHIFT_TARGETS, physics_rollout,  # noqa: E402
                       simulate, step_level)


def test_step_obeys_mass_balance():
    # inflow 0.5, outflow 0.55 * 0.6 * sqrt(4) = 0.66 -> level drops 0.16
    assert np.isclose(step_level(4.0, 0.5, 0.6, 0.0, cv=CV_TRUE), 4.0 - 0.16)


def test_level_stays_in_tank():
    assert step_level(0.01, 0.0, 1.0, -1.0) == 0.0
    assert step_level(L_MAX, 5.0, 0.0, 0.0) == L_MAX


def test_true_physics_reproduces_noise_free_plant():
    s = simulate(500, seed=3)
    # rebuild the plant with the true Cv and the true disturbance
    level = s.level_true[0]
    for t in range(1, 500):
        level = step_level(level, s.fin_true[t - 1], s.u[t - 1], s.disturbance[t - 1])
        assert np.isclose(level, s.level_true[t])


def test_regime_shift_leaves_training_range():
    train = simulate(20000, seed=1)
    shift = simulate(8000, SHIFT_TARGETS, seed=4)
    assert (shift.level_true > train.level_true.max()).mean() > 0.8


def test_nominal_physics_is_biased_but_bounded():
    s = simulate(2000, seed=7)
    roll = physics_rollout(s.level_true[100], s.fin_true[100:116], s.u[100:116])
    assert np.all((roll >= 0) & (roll <= L_MAX))
