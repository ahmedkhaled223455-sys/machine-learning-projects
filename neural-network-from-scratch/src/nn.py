"""A multilayer perceptron implemented from first principles in NumPy.

- tanh hidden activations, sigmoid output, binary cross-entropy loss
- backpropagation derived and implemented by hand
- mini-batch SGD with momentum and L2 weight decay
- numerical gradient checking (central differences) to verify backprop

The design goal is transparency: every gradient in this file can be
traced back to the chain rule, and `gradient_check` proves the
implementation is correct to ~1e-8 relative error.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class MLP:
    def __init__(self, layer_sizes: list[int], seed: int = 0,
                 l2: float = 1e-4):
        """layer_sizes e.g. [2, 32, 32, 1] -> two hidden layers of 32."""
        self.sizes = layer_sizes
        self.l2 = l2
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier/Glorot initialization (suits tanh).
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    # -- forward ------------------------------------------------------------
    def forward(self, X):
        """Return output probabilities and per-layer activation cache."""
        a = np.asarray(X, float)
        cache = [a]
        L = len(self.W)
        for l in range(L - 1):
            a = np.tanh(a @ self.W[l] + self.b[l])
            cache.append(a)
        p = _sigmoid(cache[-1] @ self.W[-1] + self.b[-1]).ravel()
        return p, cache

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # -- loss and gradients ---------------------------------------------------
    def loss(self, X, y):
        p, _ = self.forward(X)
        eps = 1e-12
        data = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.5 * self.l2 * sum(float(np.sum(W ** 2)) for W in self.W)
        return data + reg

    def gradients(self, X, y):
        """Backpropagation. Returns (dW list, db list)."""
        y = np.asarray(y, float)
        n = len(y)
        p, cache = self.forward(X)
        L = len(self.W)
        dW = [None] * L
        db = [None] * L

        # Output layer: dL/dz = (p - y)/n  (sigmoid + BCE simplification).
        delta = ((p - y) / n)[:, None]
        dW[-1] = cache[-1].T @ delta + self.l2 * self.W[-1]
        db[-1] = delta.sum(axis=0)

        for l in range(L - 2, -1, -1):
            delta = (delta @ self.W[l + 1].T) * (1.0 - cache[l + 1] ** 2)  # tanh'
            dW[l] = cache[l].T @ delta + self.l2 * self.W[l]
            db[l] = delta.sum(axis=0)
        return dW, db

    # -- training -------------------------------------------------------------
    def fit(self, X, y, lr: float = 0.1, epochs: int = 2000,
            batch_size: int = 64, momentum: float = 0.9, seed: int = 0,
            verbose_every: int | None = None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(seed)
        vW = [np.zeros_like(W) for W in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        self.loss_history = []
        n = len(y)
        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                dW, db = self.gradients(X[idx], y[idx])
                for l in range(len(self.W)):
                    vW[l] = momentum * vW[l] - lr * dW[l]
                    vb[l] = momentum * vb[l] - lr * db[l]
                    self.W[l] += vW[l]
                    self.b[l] += vb[l]
            self.loss_history.append(self.loss(X, y))
            if verbose_every and (epoch + 1) % verbose_every == 0:
                acc = (self.predict(X) == y).mean()
                print(f"  epoch {epoch + 1:5d} loss {self.loss_history[-1]:.4f}"
                      f" acc {acc:.3f}")
        return self


def gradient_check(model: MLP, X, y, n_checks: int = 60,
                   eps: float = 1e-6, seed: int = 0) -> float:
    """Compare analytic gradients to central finite differences.

    Returns the maximum relative error over randomly sampled parameters.
    A correct implementation lands around 1e-9..1e-7.
    """
    rng = np.random.default_rng(seed)
    dW, db = model.gradients(X, y)
    max_rel = 0.0
    params = [(model.W, dW), (model.b, db)]
    for tensors, grads in params:
        for t, g in zip(tensors, grads):
            flat_t = t.reshape(-1)
            flat_g = g.reshape(-1)
            for _ in range(max(1, n_checks // (2 * len(tensors)))):
                i = rng.integers(len(flat_t))
                orig = flat_t[i]
                flat_t[i] = orig + eps
                lp = model.loss(X, y)
                flat_t[i] = orig - eps
                lm = model.loss(X, y)
                flat_t[i] = orig
                num = (lp - lm) / (2 * eps)
                denom = max(abs(num) + abs(flat_g[i]), 1e-12)
                max_rel = max(max_rel, abs(num - flat_g[i]) / denom)
    return max_rel
