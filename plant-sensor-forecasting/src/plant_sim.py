"""Simulated process unit: a gravity-drained surge tank.

Physics (mass balance, one step = one minute):

    A * dL/dt = F_in + d(t) - F_out,      F_out = Cv * u * sqrt(L)

* L      tank level [m], physically bounded to [0, L_MAX]
* F_in   inflow [m^3/min], a planned (known-in-advance) control
* u      outlet valve opening in [0, 1], a planned control
* Cv     valve coefficient. Operators only know a nominal value; the true
         plant is 10% different, so a physics-only model is biased.
* d(t)   unmeasured disturbance inflow (AR(1)), never observed.

Sensors report a noisy level and a noisy inflow. The valve command is known
exactly (it is what the control system sent). Ground truth -- the true
level -- is kept separately so forecasts can be scored against reality,
not against a noisy sensor.

Operating regimes are set by a plan of target levels. Training data covers
targets in [1.0, 3.0] m; the regime-shift test operates at [3.5, 5.0] m,
outside anything the learned models have seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AREA = 1.0          # m^2
L_MAX = 6.0         # m, tank height
CV_NOMINAL = 0.50   # what the engineers believe
CV_TRUE = 0.55      # what the plant actually does
LEVEL_NOISE = 0.02  # m, level sensor noise (1 sigma)
FLOW_NOISE = 0.01   # m^3/min, inflow sensor noise (1 sigma)

TRAIN_TARGETS = (1.0, 3.0)
SHIFT_TARGETS = (3.5, 5.0)


@dataclass
class PlantSeries:
    level_true: np.ndarray   # (T,) true level
    level_meas: np.ndarray   # (T,) level sensor
    fin_true: np.ndarray     # (T,) planned inflow (applied exactly)
    fin_meas: np.ndarray     # (T,) inflow sensor
    u: np.ndarray            # (T,) valve command (known exactly)
    disturbance: np.ndarray  # (T,) unmeasured inflow


def step_level(level: float, fin: float, u: float, d: float,
               cv: float = CV_TRUE) -> float:
    """One explicit-Euler minute of the mass balance, clipped to the tank."""
    fout = cv * u * np.sqrt(max(level, 0.0))
    nxt = level + (fin + d - fout) / AREA
    return float(min(max(nxt, 0.0), L_MAX))


def make_plan(n: int, targets: tuple[float, float], rng: np.random.Generator,
              hold: tuple[int, int] = (40, 120)) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant inflow and valve schedule that steers toward random
    target levels inside ``targets``. The valve is sized with the *nominal*
    Cv, as a real operator would, so the plant settles near -- not exactly
    at -- each target."""
    fin = np.empty(n)
    u = np.empty(n)
    t = 0
    while t < n:
        span = int(rng.integers(hold[0], hold[1] + 1))
        f = rng.uniform(0.4, 0.9)
        target = rng.uniform(*targets)
        valve = np.clip(f / (CV_NOMINAL * np.sqrt(target)), 0.2, 1.0)
        fin[t:t + span] = f
        u[t:t + span] = valve
        t += span
    return fin, u


def simulate(n: int, targets: tuple[float, float] = TRAIN_TARGETS,
             seed: int = 0) -> PlantSeries:
    rng = np.random.default_rng(seed)
    fin, u = make_plan(n, targets, rng)
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = 0.95 * d[t - 1] + 0.01 * rng.standard_normal()
    level = np.empty(n)
    level[0] = float(np.mean(targets))
    for t in range(1, n):
        level[t] = step_level(level[t - 1], fin[t - 1], u[t - 1], d[t - 1])
    level_meas = level + LEVEL_NOISE * rng.standard_normal(n)
    fin_meas = fin + FLOW_NOISE * rng.standard_normal(n)
    return PlantSeries(level, level_meas, fin, fin_meas, u, d)


def physics_rollout(level0: float, fin_future: np.ndarray, u_future: np.ndarray,
                    cv: float = CV_NOMINAL) -> np.ndarray:
    """Forecast the level by integrating the mass balance with the nominal
    valve coefficient and no disturbance -- what a first-principles model
    would predict. Returns levels at t+1 .. t+H."""
    out = np.empty(len(fin_future))
    level = level0
    for h in range(len(fin_future)):
        level = step_level(level, fin_future[h], u_future[h], 0.0, cv=cv)
        out[h] = level
    return out
