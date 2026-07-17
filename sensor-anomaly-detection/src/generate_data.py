"""Physics-motivated vibration data simulator for a rotating machine.

Why simulated data? Labeled run-to-failure vibration data is scarce and
expensive; simulation lets us control ground truth (fault presence and
severity) and study detector behavior systematically. The generator
mimics the classic signature of a rolling-element bearing defect:

- Healthy signal: shaft fundamental (~29.7 Hz) plus decaying harmonics,
  broadband measurement noise, small run-to-run amplitude/speed jitter.
- Faulty signal: healthy signal + periodic impulse train at the
  bearing-pass frequency (3.585x shaft speed, an outer-race-style ratio),
  where each impact excites an exponentially decaying high-frequency
  structural resonance (~900 Hz). Impulse amplitude scales with a
  continuous severity parameter.

Outputs data/vibration_windows.npz with train (healthy-only) and test
(mixed) windows, labels, and severities.

Run:  python src/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

FS = 5000          # sampling rate [Hz]
DURATION = 1.0     # window length [s]
N = int(FS * DURATION)
SHAFT_HZ = 29.7    # nominal shaft speed
FAULT_RATIO = 3.585   # bearing defect frequency / shaft frequency
RESONANCE_HZ = 900    # structural resonance excited by impacts
SEED = 7


def healthy_window(rng) -> np.ndarray:
    t = np.arange(N) / FS
    speed = SHAFT_HZ * (1 + rng.normal(0, 0.01))     # +/-1% speed jitter
    x = np.zeros(N)
    for h, amp in enumerate([1.0, 0.45, 0.2], start=1):
        a = amp * (1 + rng.normal(0, 0.08))
        x += a * np.sin(2 * np.pi * h * speed * t + rng.uniform(0, 2 * np.pi))
    x += rng.normal(0, 0.35, N)                       # broadband noise
    return x


def add_bearing_fault(x, severity: float, rng) -> np.ndarray:
    """Superimpose an impulse train whose energy scales with severity."""
    t = np.arange(N) / FS
    speed = SHAFT_HZ * (1 + rng.normal(0, 0.01))
    f_fault = FAULT_RATIO * speed
    period = 1.0 / f_fault
    x = x.copy()
    # Decaying-resonance kernel for a single impact.
    tk = np.arange(int(0.01 * FS)) / FS               # 10 ms kernel
    kernel = np.exp(-tk / 0.002) * np.sin(2 * np.pi * RESONANCE_HZ * tk)
    start = rng.uniform(0, period)
    impact_times = np.arange(start, DURATION, period)
    for ti in impact_times:
        i0 = int(ti * FS)
        amp = severity * (1 + rng.normal(0, 0.15))    # impact-to-impact scatter
        seg = min(len(kernel), N - i0)
        if seg > 0:
            x[i0:i0 + seg] += amp * kernel[:seg]
    return x


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    n_train_healthy = 300
    n_test_healthy = 150
    n_test_faulty = 150

    X_train = np.stack([healthy_window(rng) for _ in range(n_train_healthy)])

    test_windows, test_labels, test_sev = [], [], []
    for _ in range(n_test_healthy):
        test_windows.append(healthy_window(rng))
        test_labels.append(0)
        test_sev.append(0.0)
    for _ in range(n_test_faulty):
        sev = rng.uniform(0.2, 1.5)                   # mild -> severe
        test_windows.append(add_bearing_fault(healthy_window(rng), sev, rng))
        test_labels.append(1)
        test_sev.append(sev)

    X_test = np.stack(test_windows)
    y_test = np.array(test_labels)
    sev_test = np.array(test_sev)

    np.savez_compressed(
        OUT / "vibration_windows.npz",
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test, severity_test=sev_test,
        fs=FS, shaft_hz=SHAFT_HZ, fault_ratio=FAULT_RATIO,
    )
    print(f"train (healthy only): {X_train.shape}")
    print(f"test: {X_test.shape}  faulty fraction {y_test.mean():.2f}")
    print(f"saved -> {OUT / 'vibration_windows.npz'}")


if __name__ == "__main__":
    main()
