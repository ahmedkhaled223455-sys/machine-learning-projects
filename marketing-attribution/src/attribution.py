"""Multi-touch attribution: heuristic rules vs. a data-driven Markov model.

Heuristic models (first-/last-touch, linear, time-decay, position-based)
distribute conversion credit by simple rules. They're the industry
default and they're biased: last-touch, for example, always over-credits
whatever channel closed the sale and ignores the channels that opened it.

The data-driven alternative here is the **Markov removal-effect** model.
Customer journeys are modeled as a Markov chain over channel states with
absorbing "conversion" and "null" states. A channel's importance is its
*removal effect*: how much the overall conversion probability drops when
that channel is taken out of the graph. This attributes credit by causal
contribution to conversion, not by position in the path.

The functions operate on a transition matrix, so the same code computes
the true removal effects from a known generating chain (for validation)
and the estimated ones from observed journeys.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# A journey is (path, converted): path is a list of channel names in order.
Journey = tuple


def build_transition_matrix(journeys: list, channels: list[str]):
    """Estimate the Markov transition matrix from observed journeys.

    States: 'start', each channel, 'conversion' (absorbing), 'null'
    (absorbing). Returns (P, states).
    """
    states = ["start"] + list(channels) + ["conversion", "null"]
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=float)

    for path, converted in journeys:
        prev = "start"
        for ch in path:
            counts[idx[prev], idx[ch]] += 1
            prev = ch
        counts[idx[prev], idx["conversion" if converted else "null"]] += 1

    # Absorbing states loop to themselves.
    counts[idx["conversion"], idx["conversion"]] = 1.0
    counts[idx["null"], idx["null"]] = 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0            # unreachable rows -> harmless
    return counts / row_sums, states


def conversion_probability(P: np.ndarray, states: list[str]) -> float:
    """Probability of reaching 'conversion' from 'start' (absorption)."""
    conv, null = states.index("conversion"), states.index("null")
    transient = [i for i in range(len(states)) if i not in (conv, null)]
    start_pos = transient.index(states.index("start"))

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, [conv, null])]
    fundamental = np.linalg.pinv(np.eye(len(transient)) - Q)
    absorb = fundamental @ R          # rows: transient, cols: [conversion, null]
    return float(absorb[start_pos, 0])


def removal_effect(P: np.ndarray, states: list[str], channel: str) -> float:
    """Drop in conversion probability when `channel` is removed from the graph."""
    base = conversion_probability(P, states)
    if base == 0:
        return 0.0
    c = states.index(channel)
    null = states.index("null")
    P2 = P.copy()
    # Redirect every transition INTO the channel to 'null' instead.
    P2[:, null] += P2[:, c]
    P2[:, c] = 0.0
    removed = conversion_probability(P2, states)
    return 1.0 - removed / base


def markov_attribution(P: np.ndarray, states: list[str],
                       channels: list[str]) -> dict[str, float]:
    """Conversion credit per channel, proportional to removal effects."""
    effects = {ch: max(removal_effect(P, states, ch), 0.0) for ch in channels}
    total = sum(effects.values())
    if total == 0:
        return {ch: 0.0 for ch in channels}
    return {ch: effects[ch] / total for ch in channels}


# ---------------------------------------------------------------------------
# Heuristic attribution models
# ---------------------------------------------------------------------------
def heuristic_attribution(journeys: list, channels: list[str],
                          model: str) -> dict[str, float]:
    """Credit shares per channel under a rule-based model.

    model in {first, last, linear, time_decay, position}.
    """
    credit: dict[str, float] = defaultdict(float)
    for path, converted in journeys:
        if not converted or not path:
            continue
        weights = _touch_weights(len(path), model)
        for ch, w in zip(path, weights):
            credit[ch] += w
    total = sum(credit.values())
    if total == 0:
        return {ch: 0.0 for ch in channels}
    return {ch: credit.get(ch, 0.0) / total for ch in channels}


def _touch_weights(n: int, model: str) -> np.ndarray:
    if model == "first":
        w = np.zeros(n); w[0] = 1.0
    elif model == "last":
        w = np.zeros(n); w[-1] = 1.0
    elif model == "linear":
        w = np.ones(n)
    elif model == "time_decay":
        w = 2.0 ** np.arange(n)               # later touches weigh more
    elif model == "position":
        if n == 1:
            w = np.array([1.0])
        elif n == 2:
            w = np.array([0.5, 0.5])
        else:
            w = np.full(n, 0.2 / (n - 2))
            w[0] = w[-1] = 0.4                 # U-shaped: 40% first, 40% last
    else:
        raise ValueError(f"unknown model '{model}'")
    return w / w.sum()
