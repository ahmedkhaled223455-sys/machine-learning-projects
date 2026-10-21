"""NumPy reference implementation of multi-head scaled dot-product attention.

Used as ground truth in tests for the PyTorch module, so the attention math
is verified independently of the framework.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def multihead_attention(xq, xkv, Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads, key_mask=None):
    """xq (B, Lq, d), xkv (B, Lk, d); W* are (d, d) applied as x @ W.T + b
    (the torch.nn.Linear convention). key_mask (B, Lk) bool, True = keep.
    Returns output (B, Lq, d) and attention weights (B, h, Lq, Lk)."""
    B, Lq, d = xq.shape
    Lk = xkv.shape[1]
    dh = d // n_heads

    def proj(x, W, b, L):
        return (x @ W.T + b).reshape(B, L, n_heads, dh).transpose(0, 2, 1, 3)

    q, k, v = proj(xq, Wq, bq, Lq), proj(xkv, Wk, bk, Lk), proj(xkv, Wv, bv, Lk)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
    if key_mask is not None:
        scores = np.where(key_mask[:, None, None, :], scores, -1e30)
    w = softmax(scores, -1)
    out = (w @ v).transpose(0, 2, 1, 3).reshape(B, Lq, d)
    return out @ Wo.T + bo, w
