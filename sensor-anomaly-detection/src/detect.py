"""Unsupervised fault detection on vibration windows.

The detectors are trained on healthy data ONLY (the realistic industrial
setting: faults are rare and unlabeled), then scored on a mixed test set:

1. Mahalanobis distance in feature space (regularized covariance).
2. PCA reconstruction error (subspace of healthy variation).
3. kNN novelty score (mean distance to the 5 nearest healthy windows).

Features per window combine time-domain statistics (RMS, kurtosis, crest
factor) with frequency-domain energies (log-spaced bands of the FFT
power spectrum) -- impact trains from bearing defects raise kurtosis and
high-frequency band energy long before overall RMS changes.

Run:  python src/generate_data.py && python src/detect.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vibration_windows.npz"
RESULTS = ROOT / "results"

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(X: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    n = X.shape[1]
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    peak = np.max(np.abs(X), axis=1)
    crest = peak / rms
    centered = X - X.mean(axis=1, keepdims=True)
    std = centered.std(axis=1)
    kurt = np.mean(centered ** 4, axis=1) / std ** 4 - 3.0     # excess kurtosis
    skew = np.mean(centered ** 3, axis=1) / std ** 3

    # Power spectrum and log-spaced band energies (fractions of total).
    psd = np.abs(np.fft.rfft(X, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    edges = np.geomspace(10, fs / 2, 9)                        # 8 bands
    band_feats, names = [], []
    total = psd.sum(axis=1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        band_feats.append(psd[:, m].sum(axis=1) / total)
        names.append(f"band_{int(lo)}-{int(hi)}Hz")
    centroid = (psd * freqs).sum(axis=1) / total

    F = np.column_stack([rms, peak, crest, kurt, skew, centroid] + band_feats)
    feat_names = ["rms", "peak", "crest_factor", "kurtosis", "skewness",
                  "spectral_centroid"] + names
    return F, feat_names


# ---------------------------------------------------------------------------
# Detectors (fit on healthy data only)
# ---------------------------------------------------------------------------
class MahalanobisDetector:
    def fit(self, F):
        self.mu = F.mean(axis=0)
        cov = np.cov(F, rowvar=False)
        cov += 1e-6 * np.eye(cov.shape[0]) * np.trace(cov)     # regularize
        self.inv = np.linalg.inv(cov)
        return self

    def score(self, F):
        d = F - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.inv, d))


class PCAReconstruction:
    def __init__(self, var_keep: float = 0.95):
        self.var_keep = var_keep

    def fit(self, F):
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Z = (F - self.mu) / self.sd
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        var = S ** 2 / (len(F) - 1)
        k = int(np.searchsorted(np.cumsum(var) / var.sum(), self.var_keep) + 1)
        self.k = k
        self.V = Vt[:k].T                                      # d x k
        return self

    def score(self, F):
        Z = (F - self.mu) / self.sd
        recon = (Z @ self.V) @ self.V.T
        return np.sqrt(np.sum((Z - recon) ** 2, axis=1))


class KNNNovelty:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, F):
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.ref = (F - self.mu) / self.sd
        return self

    def score(self, F):
        Z = (F - self.mu) / self.sd
        d2 = ((Z[:, None, :] - self.ref[None, :, :]) ** 2).sum(axis=2)
        d2.sort(axis=1)
        return np.sqrt(d2[:, :self.k]).mean(axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def roc_auc(y, scores):
    order = np.argsort(-scores, kind="mergesort")
    ys = np.asarray(y)[order]
    ss = np.asarray(scores)[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    distinct = np.r_[ss[1:] != ss[:-1], True]
    tps, fps = tps[distinct], fps[distinct]
    P, Nn = ys.sum(), len(ys) - ys.sum()
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / Nn]
    return fpr, tpr, float(_trapz(tpr, fpr))


def main():
    RESULTS.mkdir(exist_ok=True)
    if not DATA.exists():
        raise SystemExit("Run src/generate_data.py first.")
    z = np.load(DATA)
    fs = float(z["fs"])
    X_train, X_test = z["X_train"], z["X_test"]
    y_test, sev = z["y_test"], z["severity_test"]

    F_train, feat_names = extract_features(X_train, fs)
    F_test, _ = extract_features(X_test, fs)
    print(f"features: {F_train.shape[1]} -> {feat_names}")

    detectors = {
        "mahalanobis": MahalanobisDetector(),
        "pca_reconstruction": PCAReconstruction(var_keep=0.95),
        "knn_novelty": KNNNovelty(k=5),
    }

    summary = {}
    plt.figure(figsize=(5.5, 5))
    for name, det in detectors.items():
        det.fit(F_train)
        s_test = det.score(F_test)
        fpr, tpr, aucv = roc_auc(y_test, s_test)
        # Threshold = 99th percentile of healthy-train scores.
        thr = float(np.percentile(det.score(F_train), 99))
        pred = (s_test >= thr).astype(int)
        detection = float(pred[y_test == 1].mean())
        false_alarm = float(pred[y_test == 0].mean())
        summary[name] = {"auc": round(aucv, 4),
                         "threshold_p99": thr,
                         "detection_rate": round(detection, 4),
                         "false_alarm_rate": round(false_alarm, 4)}
        print(f"{name:20s} AUC {aucv:.3f} | detect {detection:.1%}"
              f" | false alarm {false_alarm:.1%}")
        plt.plot(fpr, tpr, label=f"{name} (AUC={aucv:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Fault detection ROC (trained on healthy data only)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS / "roc_detectors.png", dpi=150)
    plt.close()

    # Detection rate vs severity for the best detector.
    best = max(summary, key=lambda k: summary[k]["auc"])
    det = detectors[best]
    thr = summary[best]["threshold_p99"]
    s_test = det.score(F_test)
    faulty = y_test == 1
    bins = np.linspace(0.2, 1.5, 7)
    centers, rates = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = faulty & (sev >= lo) & (sev < hi)
        if m.sum():
            centers.append(0.5 * (lo + hi))
            rates.append(float((s_test[m] >= thr).mean()))
    plt.figure(figsize=(6, 4))
    plt.plot(centers, rates, "o-")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Fault severity (impulse amplitude)")
    plt.ylabel("Detection rate @ 1% false alarms")
    plt.title(f"Sensitivity vs fault severity — {best}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "detection_vs_severity.png", dpi=150)
    plt.close()

    # Example signals + spectra.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    t = np.arange(X_test.shape[1]) / fs
    healthy_i = int(np.where(y_test == 0)[0][0])
    faulty_i = int(np.where((y_test == 1) & (sev > 1.0))[0][0])
    for row, (idx, label) in enumerate([(healthy_i, "healthy"),
                                        (faulty_i, f"faulty (sev={sev[faulty_i]:.2f})")]):
        x = X_test[idx]
        axes[row, 0].plot(t[:1000], x[:1000], lw=0.6)
        axes[row, 0].set_title(f"{label} — time domain (first 0.2 s)")
        axes[row, 0].set_xlabel("time [s]")
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        psd = np.abs(np.fft.rfft(x)) ** 2
        axes[row, 1].semilogy(freqs, psd, lw=0.6)
        axes[row, 1].set_title(f"{label} — power spectrum")
        axes[row, 1].set_xlabel("frequency [Hz]")
    plt.tight_layout()
    plt.savefig(RESULTS / "example_signals.png", dpi=150)
    plt.close()

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump({"detectors": summary, "best": best,
                   "n_features": F_train.shape[1]}, f, indent=2)
    print(f"\nBest detector: {best}. Saved metrics and figures to {RESULTS}")


if __name__ == "__main__":
    main()
