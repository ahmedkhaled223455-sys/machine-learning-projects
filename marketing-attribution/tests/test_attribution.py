"""Tests for multi-touch attribution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attribution import (build_transition_matrix, conversion_probability,
                         heuristic_attribution, markov_attribution,
                         removal_effect, _touch_weights)
from generate_data import ATTR_CHANNELS, _simulate_journeys, _true_transition


def test_touch_weights_sum_to_one():
    for model in ("first", "last", "linear", "time_decay", "position"):
        for n in (1, 2, 3, 5):
            w = _touch_weights(n, model)
            assert abs(w.sum() - 1.0) < 1e-9


def test_first_and_last_put_all_weight_on_ends():
    assert _touch_weights(4, "first")[0] == 1.0
    assert _touch_weights(4, "last")[-1] == 1.0


def test_conversion_probability_simple_chain():
    # start -> conversion w.p. 0.5, start -> null w.p. 0.5
    states = ["start", "conversion", "null"]
    P = np.array([[0.0, 0.5, 0.5],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    assert abs(conversion_probability(P, states) - 0.5) < 1e-9


def test_removal_effect_nonneg_and_transition_rows_normalized():
    journeys = [(["search", "email"], True), (["social"], False),
                (["display", "search"], True)]
    P, states = build_transition_matrix(journeys, ATTR_CHANNELS)
    assert np.allclose(P.sum(axis=1), 1.0)
    for ch in ATTR_CHANNELS:
        assert removal_effect(P, states, ch) >= -1e-9


def test_markov_recovers_true_shares_from_journeys():
    true_P, states, _ = _true_transition()
    true_shares = markov_attribution(true_P, states, ATTR_CHANNELS)

    journeys = _simulate_journeys(6000, np.random.default_rng(0))
    P_hat, states_hat = build_transition_matrix(journeys, ATTR_CHANNELS)
    est = markov_attribution(P_hat, states_hat, ATTR_CHANNELS)

    mae = np.mean([abs(est[c] - true_shares[c]) for c in ATTR_CHANNELS])
    assert mae < 0.03
    # display is the weakest channel in the generating chain
    assert min(true_shares, key=true_shares.get) == "display"


def test_last_touch_is_biased_relative_to_markov():
    # Last-touch over-credits the closing channel; check it disagrees with the
    # data-driven model on the known-biased channel (email closes many paths).
    journeys = _simulate_journeys(6000, np.random.default_rng(1))
    P_hat, states_hat = build_transition_matrix(journeys, ATTR_CHANNELS)
    markov = markov_attribution(P_hat, states_hat, ATTR_CHANNELS)
    last = heuristic_attribution(journeys, ATTR_CHANNELS, "last")
    assert last["email"] > markov["email"]
