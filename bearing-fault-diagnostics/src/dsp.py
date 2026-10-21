"""Condition-monitoring signal processing, implemented with NumPy FFTs.

* `analytic_signal` / `envelope`      -- Hilbert transform via the FFT
* `bandpass`                          -- zero-phase FFT-mask band-pass
* `envelope_spectrum`                 -- spectrum of the squared envelope
* `stft`, `spectral_kurtosis`         -- Antoni's STFT-based spectral kurtosis
* `best_band`                         -- kurtogram-style search over window
                                         lengths for the most impulsive band
* `time_synchronous_average`          -- angle-domain resampling with a known
                                         shaft speed, averaged over revolutions
* `tsa_residual`                      -- signal minus its shaft-synchronous part
"""

from __future__ import annotations

import numpy as np


def analytic_signal(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h, axis=-1)


def envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(analytic_signal(x))


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1 / fs)
    X[..., (f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n, axis=-1)


def amplitude_spectrum(x: np.ndarray, fs: float):
    n = x.shape[-1]
    return np.fft.rfftfreq(n, 1 / fs), np.abs(np.fft.rfft(x * np.hanning(n), axis=-1)) * 2 / n


def envelope_spectrum(x: np.ndarray, fs: float):
    """Spectrum of the squared envelope (mean removed) -- shows impact repetition rates."""
    e2 = envelope(x) ** 2
    e2 = e2 - e2.mean(-1, keepdims=True)
    return amplitude_spectrum(e2, fs)


def stft(x: np.ndarray, nwin: int, hop: int | None = None):
    hop = hop or nwin // 4
    w = np.hanning(nwin)
    starts = np.arange(0, len(x) - nwin + 1, hop)
    frames = np.stack([x[s:s + nwin] * w for s in starts])
    return np.fft.rfft(frames, axis=-1)                    # (frames, freqs)


def spectral_kurtosis(x: np.ndarray, fs: float, nwin: int):
    """SK(f) = <|X|^4> / <|X|^2>^2 - 2  (zero for stationary Gaussian noise,
    large in bands dominated by repetitive transients)."""
    S = stft(x, nwin)
    p = np.abs(S) ** 2
    sk = (p ** 2).mean(0) / (p.mean(0) ** 2 + 1e-30) - 2.0
    return np.fft.rfftfreq(nwin, 1 / fs), sk


def best_band(x: np.ndarray, fs: float, windows=(16, 32, 64, 128, 256), f_min: float = 200.0,
              min_bandwidth: float = 0.0):
    """Kurtogram-style search: for each window length, the frequency bin with the
    highest spectral kurtosis defines a band [f - fs/nwin, f + fs/nwin]; return
    the (lo, hi, sk) of the best band overall.

    min_bandwidth: demodulation bands narrower than ~2-3x the fault repetition
    rate cannot show that rate in the envelope, so such windows are skipped.
    """
    best = (0.0, fs / 2, -np.inf)
    for nw in windows:
        bw = fs / nw
        if 2 * bw < min_bandwidth:
            continue
        f, sk = spectral_kurtosis(x, fs, nw)
        ok = (f - bw >= f_min) & (f + bw <= fs / 2)
        if not ok.any():
            continue
        i = np.argmax(np.where(ok, sk, -np.inf))
        if sk[i] > best[2]:
            best = (f[i] - bw, f[i] + bw, float(sk[i]))
    return best


def time_synchronous_average(x: np.ndarray, fs: float, shaft_hz: float, samples_per_rev: int = 256):
    """Resample to the shaft-angle domain (known constant speed) and average
    complete revolutions. Returns (tsa_one_rev, n_revs)."""
    n_revs = int(len(x) / fs * shaft_hz)
    angle_t = np.arange(n_revs * samples_per_rev) / (samples_per_rev * shaft_hz)  # times of each angle sample
    t = np.arange(len(x)) / fs
    xa = np.interp(angle_t, t, x)
    return xa.reshape(n_revs, samples_per_rev).mean(0), n_revs


def tsa_residual(x: np.ndarray, fs: float, shaft_hz: float, samples_per_rev: int = 256):
    """Remove everything synchronous with the shaft (harmonics, gear mesh) by
    subtracting the periodic extension of the TSA, mapped back to time."""
    tsa, _ = time_synchronous_average(x, fs, shaft_hz, samples_per_rev)
    t = np.arange(len(x)) / fs
    phase = (t * shaft_hz) % 1.0 * samples_per_rev
    ext = np.concatenate([tsa, tsa[:1]])
    sync = np.interp(phase, np.arange(samples_per_rev + 1), ext)
    return x - sync
