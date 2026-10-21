"""From vibration record to fault evidence.

`fault_indicators` measures, for each candidate fault (outer, inner, ball),
how far the spectrum rises at the fault frequency and its 2nd/3rd harmonics
above the local noise floor (peak in a +-2% window / median of the
surrounding +-15%, ignoring that window). The preprocessing chain is
configurable so each step can be ablated:

    "raw_spectrum"   amplitude spectrum of the raw signal (no demodulation)
    "envelope_full"  squared-envelope spectrum of the full-band signal
    "envelope_sk"    + band-pass on the spectral-kurtosis-selected band
    "full"           + time-synchronous-average residual first
                       (removes shaft harmonics and gear mesh)
"""

from __future__ import annotations

import numpy as np

from bearing import FS
from dsp import (amplitude_spectrum, bandpass, best_band, envelope_spectrum,
                 tsa_residual)

FAULTS = ("outer", "inner", "ball")
METHODS = ("raw_spectrum", "envelope_full", "envelope_sk", "full")


def _floor(f, A, fc, win=0.02, ring=0.15):
    w = np.abs(f - fc) <= win * fc
    nb = (np.abs(f - fc) <= ring * fc) & ~w
    return float(np.median(A[nb])) + 1e-12 if nb.any() else None


def _peak_ratio(f, A, fc, win=0.02, ring=0.15):
    w = np.abs(f - fc) <= win * fc
    fl = _floor(f, A, fc, win, ring)
    if not w.any() or fl is None:
        return 1.0
    return float(A[w].max() / fl)


def _sideband_ratio(f, A, fc, f_mod, halfwidth_hz=1.0, ring=0.15):
    """Sideband evidence at fc +- f_mod, relative to what noise alone gives.

    The max over a small window divided by the median floor is > 1 even for pure
    noise (~2x), which would bias every tie toward the modulated fault types.
    So the ratio is divided by the same statistic measured at many window
    positions in the surrounding ring: ~1 for noise, >1 for real sidebands.
    """
    fl = _floor(f, A, fc)
    if fl is None:
        return 1.0
    df = f[1] - f[0]
    half = max(int(round(halfwidth_hz / df)), 1)
    ring_idx = np.where(np.abs(f - fc) <= ring * fc)[0]
    noise_max = [A[max(i - half, 0): i + half + 1].max() / fl for i in ring_idx[::2 * half + 1]]
    baseline = float(np.median(noise_max)) if noise_max else 1.0
    vals = []
    for s in (fc - f_mod, fc + f_mod):
        w = np.abs(f - s) <= halfwidth_hz
        vals.append(A[w].max() / fl if w.any() else 1.0)
    return float(np.mean(vals)) / baseline


BANDWIDTH_FACTOR = 3.0  # demodulation band >= 3x the highest candidate fault frequency


def spectrum_for(x, shaft_hz, method, fs=FS, max_fault_hz=0.0):
    if method == "raw_spectrum":
        return amplitude_spectrum(x, fs)
    if method == "envelope_full":
        return envelope_spectrum(x, fs)
    y = tsa_residual(x, fs, shaft_hz) if method == "full" else x
    lo, hi, _ = best_band(y, fs, min_bandwidth=BANDWIDTH_FACTOR * max_fault_hz)
    return envelope_spectrum(bandpass(y, fs, lo, hi), fs)


def fault_evidence(x, shaft_hz, defect_orders, ftf_order, method="full", harmonics=(1, 2, 3), fs=FS):
    """Returns (ind, mod):
    ind -- (3 faults x n_harmonics) peak-to-floor ratios at the fault frequency harmonics
    mod -- (3,) modulation-sideband ratios: inner at BPFI +- shaft speed, ball at
           2*BSF +- cage frequency; a fixed outer-race defect is unmodulated (ratio 1).
    """
    max_fault = max(defect_orders[k] for k in FAULTS) * shaft_hz
    f, A = spectrum_for(x, shaft_hz, method, fs, max_fault)
    ind = np.array([[_peak_ratio(f, A, h * defect_orders[k] * shaft_hz) for h in harmonics]
                    for k in FAULTS])
    def sb(order, f_mod):  # geometric mean over the first two harmonics (less noisy)
        return float(np.exp(np.mean([np.log(_sideband_ratio(f, A, h * order * shaft_hz, f_mod))
                                     for h in (1, 2)])))

    mod = np.array([1.0, sb(defect_orders["inner"], shaft_hz), sb(defect_orders["ball"], ftf_order * shaft_hz)])
    return ind, mod


def fault_indicators(x, shaft_hz, defect_orders, method="full", harmonics=(1, 2, 3), fs=FS):
    """Harmonic evidence only (kept for the ablation and simple use)."""
    max_fault = max(defect_orders[k] for k in FAULTS) * shaft_hz
    f, A = spectrum_for(x, shaft_hz, method, fs, max_fault)
    return np.array([[_peak_ratio(f, A, h * defect_orders[k] * shaft_hz) for h in harmonics]
                     for k in FAULTS])


TIE_MARGIN = 0.25       # harmonic scores this close (log units) are treated as a tie
SIDEBAND_EVIDENCE = 0.4 # log sideband ratio a modulated fault must show to win a tie


def rule_diagnosis(ind: np.ndarray, threshold: float, mod: np.ndarray | None = None,
                   tie_margin: float = TIE_MARGIN, sideband_evidence: float = SIDEBAND_EVIDENCE) -> int:
    """0 = healthy, 1/2/3 = outer/inner/ball.

    Detection: harmonic evidence (mean log ratio over harmonics) above threshold.
    Identification: the strongest harmonic evidence, unless several candidates
    tie (coinciding fault frequencies). Then a modulated fault (inner: shaft-rate
    sidebands; ball: cage-rate sidebands) must show sideband evidence to win;
    otherwise the unmodulated explanation (outer race) is preferred.
    """
    score = np.log(ind).mean(-1)
    if score.max() < threshold:
        return 0
    top = int(np.argmax(score))
    if mod is None:
        return 1 + top
    tied = np.where(score >= score[top] - tie_margin)[0]
    if len(tied) == 1:
        return 1 + top
    sbs = np.log(np.maximum(mod, 1e-12))
    modulated = [k for k in tied if k != 0 and sbs[k] >= sideband_evidence]
    if modulated:
        return 1 + max(modulated, key=lambda k: sbs[k])
    return 1 if 0 in tied else 1 + top


def statistical_features(x, fs=FS):
    """Generic condition indicators (no bearing kinematics): RMS, peak, crest
    factor, kurtosis, skewness, spectral centroid, 8 log-spaced band energies."""
    c = x - x.mean()
    rms = np.sqrt(np.mean(c ** 2))
    peak = np.abs(c).max()
    kurt = np.mean(c ** 4) / np.mean(c ** 2) ** 2 - 3
    skew = np.mean(c ** 3) / np.mean(c ** 2) ** 1.5
    f, A = amplitude_spectrum(c, fs)
    p = A ** 2
    tot = p.sum()
    edges = np.geomspace(10, fs / 2, 9)
    bands = [p[(f >= lo) & (f < hi)].sum() / tot for lo, hi in zip(edges[:-1], edges[1:])]
    return np.array([np.log(rms), np.log(peak), peak / rms, kurt, skew, (p * f).sum() / tot] + bands)


class SoftmaxClassifier:
    """Multinomial logistic regression, full-batch gradient descent with L2."""

    def __init__(self, l2=1e-2, lr=0.5, iters=3000):
        self.l2, self.lr, self.iters = l2, lr, iters

    def fit(self, X, y, n_classes):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        Y = np.eye(n_classes)[y]
        self.W = np.zeros((Z.shape[1], n_classes))
        for _ in range(self.iters):
            P = self._softmax(Z @ self.W)
            grad = Z.T @ (P - Y) / len(Z) + self.l2 * np.vstack([np.zeros((1, n_classes)), self.W[1:]])
            self.W -= self.lr * grad
        return self

    @staticmethod
    def _softmax(S):
        S = S - S.max(1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(1, keepdims=True)

    def predict_proba(self, X):
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return self._softmax(Z @ self.W)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
