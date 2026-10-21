"""Standard motion-forecasting metrics (as in Argoverse / Waymo benchmarks).

pred: (N, K, F, 2) candidate trajectories, true: (N, F, 2).
"""

from __future__ import annotations

import numpy as np

MISS_THRESHOLD = 2.0  # metres at the final step


def _errors(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    if pred.ndim == 3:
        pred = pred[:, None]
    return np.linalg.norm(pred - true[:, None], axis=-1)  # (N, K, F)


def min_ade(pred, true) -> float:
    return float(_errors(pred, true).mean(-1).min(-1).mean())


def min_fde(pred, true) -> float:
    return float(_errors(pred, true)[..., -1].min(-1).mean())


def miss_rate(pred, true, thr: float = MISS_THRESHOLD) -> float:
    return float((_errors(pred, true)[..., -1].min(-1) > thr).mean())


def summarize(pred, true) -> dict:
    return {"minADE": min_ade(pred, true), "minFDE": min_fde(pred, true),
            "MR": miss_rate(pred, true)}


def top1(pred: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Most likely mode per sample -> (N, F, 2)."""
    return pred[np.arange(len(pred)), scores.argmax(-1)]
