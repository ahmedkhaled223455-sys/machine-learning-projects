"""Multi-lane highway traffic simulator with known ground truth.

Cars follow the Intelligent Driver Model (IDM) longitudinally and make
discretionary lane changes (a simplified MOBIL rule: change when the target
lane lets you go meaningfully faster and the new follower is not forced to
brake hard). Some cars also slow down unexpectedly. Both behaviours are
*decided stochastically*, so the same 2-second history can be followed by
several plausible futures -- which is exactly why motion forecasters predict
multiple modes.

Coordinates: x along the road (m), y across it (m, lane centres at
lane * LANE_W). Time step DT = 0.1 s (10 Hz, typical of AV logs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 0.1
LANE_W = 3.7
N_LANES = 3


@dataclass
class SimConfig:
    n_cars: int = 16
    road_len: float = 320.0          # initial spawn segment (m)
    v0_range: tuple = (24.0, 33.0)   # desired speeds (m/s)
    init_speed_frac: tuple = (0.75, 1.0)
    steps: int = 260                 # recorded steps (26 s)
    warmup: int = 50                 # steps simulated before recording
    lc_hazard: float = 0.03          # per-step chance a car evaluates a lane change
    lc_gain: float = 0.3             # acceleration advantage needed (m/s^2)
    lc_random: float = 0.002         # per-step chance of an unmotivated lane change
    lc_duration: tuple = (3.0, 5.0)  # seconds to complete a lane change
    slow_hazard: float = 0.003       # per-step chance of a slowdown event
    slow_factor: tuple = (0.5, 0.8)  # new desired speed fraction
    safe_decel: float = 4.0          # max braking imposed on new follower (m/s^2)


# IDM parameters
A_MAX = 1.5
B_COMF = 2.0
S0 = 2.0
DELTA = 4.0
CAR_LEN = 4.8


def idm_accel(v, v0, gap, dv, T):
    """IDM acceleration. gap = bumper-to-bumper distance to leader (inf if none),
    dv = v - v_leader (closing speed)."""
    s_star = S0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(A_MAX * B_COMF)))
    free = 1.0 - (v / v0) ** DELTA
    inter = np.where(np.isfinite(gap), (s_star / np.maximum(gap, 0.1)) ** 2, 0.0)
    return A_MAX * (free - inter)


def _leader_follower(x, lanes, i, lane):
    """Indices of the closest car ahead / behind car i in `lane` (-1 if none)."""
    same = np.where((lanes == lane) & (np.arange(len(x)) != i))[0]
    if len(same) == 0:
        return -1, -1
    d = x[same] - x[i]
    ahead = same[d > 0]
    behind = same[d <= 0]
    lead = ahead[np.argmin(x[ahead])] if len(ahead) else -1
    foll = behind[np.argmax(x[behind])] if len(behind) else -1
    return lead, foll


def _accel_with(x, v, v0, T, i, lead):
    if lead < 0:
        return float(idm_accel(v[i], v0[i], np.inf, 0.0, T[i]))
    gap = x[lead] - x[i] - CAR_LEN
    return float(idm_accel(v[i], v0[i], gap, v[i] - v[lead], T[i]))


def simulate(cfg: SimConfig, rng: np.random.Generator) -> dict:
    """Run one scene. Returns positions (steps, n, 2), lane-change flags
    (steps, n) marking steps during which each car is changing lanes, and
    per-car parameters."""
    n = cfg.n_cars
    lanes = rng.integers(0, N_LANES, n)
    # spawn without overlaps: sort positions per lane and enforce spacing
    x = np.sort(rng.uniform(0, cfg.road_len, n))
    for lane in range(N_LANES):
        idx = np.where(lanes == lane)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if x[b] - x[a] < 25.0:
                x[b] = x[a] + 25.0
    v0 = rng.uniform(*cfg.v0_range, n)
    v0_nominal = v0.copy()
    v = v0 * rng.uniform(*cfg.init_speed_frac, n)
    T = rng.uniform(1.2, 1.8, n)
    y = lanes * LANE_W

    lc_from = lanes.astype(float).copy()   # lane index at start of a change
    lc_to = lanes.astype(float).copy()
    lc_t = np.zeros(n)                     # elapsed time in current change
    lc_dur = np.ones(n)
    changing = np.zeros(n, dtype=bool)

    total = cfg.warmup + cfg.steps
    pos = np.zeros((cfg.steps, n, 2))
    flags = np.zeros((cfg.steps, n), dtype=bool)

    for step in range(total):
        acc = np.zeros(n)
        for i in range(n):
            lead, _ = _leader_follower(x, lanes, i, lanes[i])
            acc[i] = _accel_with(x, v, v0, T, i, lead)

        # behaviour decisions
        for i in range(n):
            if rng.random() < cfg.slow_hazard:
                v0[i] = v0_nominal[i] * rng.uniform(*cfg.slow_factor)
            elif v0[i] < v0_nominal[i] and rng.random() < 0.01:
                v0[i] = v0_nominal[i]  # resume
            if changing[i]:
                continue
            motivated = rng.random() < cfg.lc_hazard
            random_lc = rng.random() < cfg.lc_random
            if not (motivated or random_lc):
                continue
            options = [ln for ln in (lanes[i] - 1, lanes[i] + 1) if 0 <= ln < N_LANES]
            rng.shuffle(options)
            for ln in options:
                lead, foll = _leader_follower(x, lanes, i, ln)
                # spatial safety: room in the target lane
                if lead >= 0 and x[lead] - x[i] - CAR_LEN < 8.0:
                    continue
                if foll >= 0 and x[i] - x[foll] - CAR_LEN < 8.0:
                    continue
                # new follower must not brake harder than safe_decel
                if foll >= 0:
                    gap_f = x[i] - x[foll] - CAR_LEN
                    a_f = float(idm_accel(v[foll], v0[foll], gap_f, v[foll] - v[i], T[foll]))
                    if a_f < -cfg.safe_decel:
                        continue
                gain = _accel_with(x, v, v0, T, i, lead) - acc[i]
                if random_lc or gain > cfg.lc_gain:
                    changing[i] = True
                    lc_from[i] = lanes[i]
                    lc_to[i] = ln
                    lc_t[i] = 0.0
                    lc_dur[i] = rng.uniform(*cfg.lc_duration)
                    lanes[i] = ln  # occupy the target lane for car-following
                    break

        # integrate longitudinal motion
        v = np.maximum(0.0, v + acc * DT)
        x = x + v * DT

        # lateral motion: smooth (cosine) profile between lane centres
        lc_t = np.where(changing, lc_t + DT, lc_t)
        frac = np.clip(lc_t / lc_dur, 0.0, 1.0)
        s = 0.5 - 0.5 * np.cos(np.pi * frac)
        y = np.where(changing, (lc_from + (lc_to - lc_from) * s) * LANE_W, lanes * LANE_W)
        done = changing & (frac >= 1.0)
        changing = changing & ~done

        if step >= cfg.warmup:
            k = step - cfg.warmup
            pos[k, :, 0] = x
            pos[k, :, 1] = y
            flags[k] = changing

    return {"pos": pos, "lane_change": flags, "v0": v0_nominal}
