"""Turn a plant series into supervised forecasting windows.

At decision time t a model may use:
  * the last W measured levels and inflows (sensors, possibly faulty),
  * the current and planned inflow / valve commands (known controls).
It must predict the TRUE level at t+1 .. t+H.

Nothing from the future of the sensors, the true level, or the unmeasured
disturbance ever enters the inputs -- the window builder is the single
place that enforces that, and ``tests/test_fc_windows.py`` checks it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plant_sim import AREA, CV_NOMINAL, L_MAX, PlantSeries
from sensor_faults import clean_window

W = 32   # lookback (minutes)
H = 16   # horizon (minutes)


@dataclass
class Windows:
    past_level: np.ndarray   # (N, W) sensor levels after the input layer
    past_fin: np.ndarray     # (N, W) sensor inflows
    fut_fin: np.ndarray      # (N, H) planned inflow
    fut_u: np.ndarray        # (N, H) planned valve opening
    phys: np.ndarray         # (N, H) physics-only forecast
    target: np.ndarray       # (N, H) true level

    def __len__(self) -> int:
        return len(self.target)


def physics_rollout_batch(level0: np.ndarray, fut_fin: np.ndarray,
                          fut_u: np.ndarray, cv: float = CV_NOMINAL) -> np.ndarray:
    """Vectorised nominal mass-balance rollout for many windows at once."""
    level = np.clip(level0.astype(float), 0.0, L_MAX)
    out = np.empty_like(fut_fin, dtype=float)
    for h in range(fut_fin.shape[1]):
        fout = cv * fut_u[:, h] * np.sqrt(np.maximum(level, 0.0))
        level = np.clip(level + (fut_fin[:, h] - fout) / AREA, 0.0, L_MAX)
        out[:, h] = level
    return out


def make_windows(series: PlantSeries, level_input: np.ndarray | None = None,
                 clean: bool = False, stride: int = 1) -> Windows:
    """Build windows. ``level_input`` overrides the level sensor (e.g. a
    fault-injected copy); ``clean`` runs the robust input layer on each
    window before it reaches any model."""
    lvl = series.level_meas if level_input is None else level_input
    n = len(lvl)
    starts = np.arange(W, n - H, stride)  # t = start index = "now" + 1
    past_level = np.stack([lvl[t - W:t] for t in starts])
    if clean:
        past_level = np.stack([clean_window(w) for w in past_level])
    else:
        # even without the repair layer, a model cannot ingest NaN; carry
        # the last finite value forward (the naive industry default)
        past_level = _ffill_rows(past_level)
    past_fin = np.stack([series.fin_meas[t - W:t] for t in starts])
    # level[t] is produced by the controls applied at t-1, so the commands
    # that drive the next H levels are those at t-1 .. t+H-2 (the current
    # command plus the plan) -- all known at decision time.
    fut_fin = np.stack([series.fin_true[t - 1:t - 1 + H] for t in starts])
    fut_u = np.stack([series.u[t - 1:t - 1 + H] for t in starts])
    target = np.stack([series.level_true[t:t + H] for t in starts])
    phys = physics_rollout_batch(past_level[:, -1], fut_fin, fut_u)
    return Windows(past_level, past_fin, fut_fin, fut_u, phys, target)


def _ffill_rows(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for r in range(out.shape[0]):
        row = out[r]
        if np.isnan(row).any():
            finite = np.isfinite(row)
            if not finite.any():
                out[r] = 0.0
                continue
            first = row[finite][0]
            last = first
            for i in range(len(row)):
                if np.isfinite(row[i]):
                    last = row[i]
                else:
                    row[i] = last
    return out
