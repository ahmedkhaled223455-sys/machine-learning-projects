"""Turn simulated scenes into agent-centric prediction samples.

Each sample: a *target* car plus up to M nearest neighbours, observed for
H_STEPS (2 s), and the target's true future for F_STEPS (3 s). Everything is
expressed in the target's frame at the last observed step (target at the
origin). Observed positions carry sensor noise; the future target is the
true trajectory.

Arrays
  hist   (N, A, H_STEPS, 4)  x, y, vx, vy  (A = 1 + M; agent 0 = target)
  mask   (N, A) bool          True where the agent slot is real
  fut    (N, F_STEPS, 2)      target future x, y
  lc     (N,) bool            target changes lanes during the future window
"""

from __future__ import annotations

import numpy as np

from traj_sim import DT, SimConfig, simulate

H_STEPS = 20
F_STEPS = 30
MAX_NEIGHBORS = 8
NEIGHBOR_RANGE = 60.0  # metres, longitudinal
POS_NOISE = 0.1        # metres, observation noise on history


def velocities(xy: np.ndarray) -> np.ndarray:
    """Finite-difference velocity along the time axis (axis -2)."""
    v = np.zeros_like(xy)
    v[..., 1:, :] = (xy[..., 1:, :] - xy[..., :-1, :]) / DT
    v[..., 0, :] = v[..., 1, :]
    return v


def samples_from_scene(scene: dict, rng: np.random.Generator, t0_per_car: int = 3) -> dict:
    pos = scene["pos"]                 # (steps, n, 2)
    lcf = scene["lane_change"]         # (steps, n)
    steps, n, _ = pos.shape
    A = 1 + MAX_NEIGHBORS
    out = {"hist": [], "mask": [], "fut": [], "lc": []}
    noisy = pos + rng.normal(0.0, POS_NOISE, pos.shape)
    for car in range(n):
        t0s = rng.integers(H_STEPS - 1, steps - F_STEPS, t0_per_car)
        for t0 in t0s:
            origin = noisy[t0, car]
            hsl = slice(t0 - H_STEPS + 1, t0 + 1)
            # neighbours: nearest by longitudinal distance within range
            d = np.abs(pos[t0, :, 0] - pos[t0, car, 0])
            d[car] = np.inf
            order = [j for j in np.argsort(d) if d[j] <= NEIGHBOR_RANGE][:MAX_NEIGHBORS]
            agents = [car] + order
            hist = np.zeros((A, H_STEPS, 4))
            mask = np.zeros(A, dtype=bool)
            for slot, j in enumerate(agents):
                xy = noisy[hsl, j] - origin
                hist[slot, :, :2] = xy
                hist[slot, :, 2:] = velocities(xy)
                mask[slot] = True
            fut = pos[t0 + 1: t0 + 1 + F_STEPS, car] - origin
            out["hist"].append(hist)
            out["mask"].append(mask)
            out["fut"].append(fut)
            out["lc"].append(bool(lcf[t0 + 1: t0 + 1 + F_STEPS, car].any()))
    return {k: np.asarray(v) for k, v in out.items()}


def build_dataset(n_scenes: int, seed: int, cfg: SimConfig | None = None,
                  t0_per_car: int = 3) -> dict:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    parts = [samples_from_scene(simulate(cfg, rng), rng, t0_per_car) for _ in range(n_scenes)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def dense_traffic_config() -> SimConfig:
    """Distribution shift for robustness testing: more cars, slower traffic."""
    return SimConfig(n_cars=26, road_len=300.0, v0_range=(18.0, 28.0))
