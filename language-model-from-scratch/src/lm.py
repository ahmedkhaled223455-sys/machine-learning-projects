"""Character-level language models implemented from scratch in NumPy.

Two model families, sharing one vocabulary/tokenization layer:

- NGramLM: count-based model of order n with add-k smoothing and
  recursive backoff to shorter contexts. The classical baseline.
- NeuralLM: a fixed-context neural language model (Bengio et al., 2003
  architecture): learned character embeddings -> concatenation ->
  tanh hidden layer -> softmax over the vocabulary. Trained with
  mini-batch SGD + momentum; backpropagation (including gradients
  into the embedding table) written out by hand.

These are the same primitives that underlie modern LLMs — tokenization,
embeddings, next-token prediction, temperature/top-k sampling, and
perplexity evaluation — at a scale where every moving part is visible.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.size = len(chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def _sample_from(probs: np.ndarray, temperature: float, top_k: int | None,
                 rng) -> int:
    """Temperature + optional top-k sampling from a probability vector."""
    logp = np.log(probs + 1e-12) / max(temperature, 1e-6)
    p = np.exp(logp - logp.max())
    if top_k is not None and top_k < len(p):
        cutoff = np.sort(p)[-top_k]
        p = np.where(p >= cutoff, p, 0.0)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


# ---------------------------------------------------------------------------
# N-gram model with add-k smoothing and backoff
# ---------------------------------------------------------------------------
class NGramLM:
    def __init__(self, order: int, vocab_size: int, k: float = 0.5):
        assert order >= 1
        self.order = order
        self.V = vocab_size
        self.k = k
        # counts[m] maps a length-m context tuple -> np.array of next-char counts
        self.counts = [dict() for _ in range(order)]

    def fit(self, ids: np.ndarray):
        for m in range(self.order):
            table = self.counts[m]
            for i in range(m, len(ids)):
                ctx = tuple(ids[i - m:i])
                row = table.get(ctx)
                if row is None:
                    row = np.zeros(self.V, dtype=np.float64)
                    table[ctx] = row
                row[ids[i]] += 1
        return self

    def _probs(self, context: tuple) -> np.ndarray:
        """Distribution over the next char, backing off to shorter
        contexts (down to the unigram) whenever a context is unseen."""
        for m in range(min(len(context), self.order - 1), -1, -1):
            ctx = context[len(context) - m:] if m else ()
            row = self.counts[m].get(tuple(ctx))
            if row is not None:
                return (row + self.k) / (row.sum() + self.k * self.V)
        return np.full(self.V, 1.0 / self.V)

    def perplexity(self, ids: np.ndarray) -> float:
        n = self.order - 1
        nll = 0.0
        count = 0
        for i in range(len(ids)):
            ctx = tuple(ids[max(0, i - n):i])
            p = self._probs(ctx)[ids[i]]
            nll -= np.log(p)
            count += 1
        return float(np.exp(nll / count))

    def generate(self, vocab: Vocab, prompt: str, n_chars: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        ids = list(vocab.encode(prompt))
        for _ in range(n_chars):
            ctx = tuple(ids[-(self.order - 1):]) if self.order > 1 else ()
            probs = self._probs(ctx)
            ids.append(_sample_from(probs, temperature, top_k, rng))
        return vocab.decode(ids)


# ---------------------------------------------------------------------------
# Neural language model (fixed context window)
# ---------------------------------------------------------------------------
class NeuralLM:
    def __init__(self, vocab_size: int, context: int = 12, emb_dim: int = 24,
                 hidden: int = 128, seed: int = 0):
        self.V, self.C, self.E, self.H = vocab_size, context, emb_dim, hidden
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0, 0.1, (vocab_size, emb_dim))
        lim1 = np.sqrt(6.0 / (context * emb_dim + hidden))
        self.W1 = rng.uniform(-lim1, lim1, (context * emb_dim, hidden))
        self.b1 = np.zeros(hidden)
        lim2 = np.sqrt(6.0 / (hidden + vocab_size))
        self.W2 = rng.uniform(-lim2, lim2, (hidden, vocab_size))
        self.b2 = np.zeros(vocab_size)

    # -- forward / loss -----------------------------------------------------
    def _forward(self, ctx_ids: np.ndarray):
        B = len(ctx_ids)
        X = self.emb[ctx_ids].reshape(B, self.C * self.E)
        h = np.tanh(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        expl = np.exp(logits)
        probs = expl / expl.sum(axis=1, keepdims=True)
        return X, h, probs

    def loss(self, ctx_ids, targets) -> float:
        _, _, probs = self._forward(ctx_ids)
        return float(-np.mean(np.log(probs[np.arange(len(targets)), targets]
                                     + 1e-12)))

    # -- training -----------------------------------------------------------
    def fit(self, ids: np.ndarray, epochs: int = 10, batch_size: int = 128,
            lr: float = 0.1, momentum: float = 0.9, seed: int = 0,
            val_ids: np.ndarray | None = None, verbose: bool = True):
        # Build (context, target) pairs with a sliding window.
        C = self.C
        contexts = np.lib.stride_tricks.sliding_window_view(ids[:-1], C)
        targets = ids[C:]
        n = len(targets)
        rng = np.random.default_rng(seed)
        params = [self.emb, self.W1, self.b1, self.W2, self.b2]
        vel = [np.zeros_like(p) for p in params]
        self.history = {"train_loss": [], "val_ppl": []}

        for epoch in range(epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                ctx, tgt = contexts[idx], targets[idx]
                B = len(idx)
                X, h, probs = self._forward(ctx)
                epoch_loss += -np.sum(
                    np.log(probs[np.arange(B), tgt] + 1e-12))

                # backprop
                dlogits = probs
                dlogits[np.arange(B), tgt] -= 1.0
                dlogits /= B
                dW2 = h.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dh = (dlogits @ self.W2.T) * (1.0 - h ** 2)
                dW1 = X.T @ dh
                db1 = dh.sum(axis=0)
                dX = (dh @ self.W1.T).reshape(B, self.C, self.E)
                demb = np.zeros_like(self.emb)
                np.add.at(demb, ctx, dX)

                for p, v, g in zip(params, vel,
                                   [demb, dW1, db1, dW2, db2]):
                    v *= momentum
                    v -= lr * g
                    p += v

            self.history["train_loss"].append(epoch_loss / n)
            if val_ids is not None:
                ppl = self.perplexity(val_ids)
                self.history["val_ppl"].append(ppl)
                if verbose:
                    print(f"  epoch {epoch + 1:2d} | train loss "
                          f"{epoch_loss / n:.3f} | val ppl {ppl:.2f}")
        return self

    def perplexity(self, ids: np.ndarray, batch_size: int = 4096) -> float:
        C = self.C
        contexts = np.lib.stride_tricks.sliding_window_view(ids[:-1], C)
        targets = ids[C:]
        nll = 0.0
        for start in range(0, len(targets), batch_size):
            ctx = contexts[start:start + batch_size]
            tgt = targets[start:start + batch_size]
            _, _, probs = self._forward(ctx)
            nll -= np.sum(np.log(probs[np.arange(len(tgt)), tgt] + 1e-12))
        return float(np.exp(nll / len(targets)))

    def generate(self, vocab: Vocab, prompt: str, n_chars: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        ids = list(vocab.encode(prompt))
        while len(ids) < self.C:                     # left-pad with spaces
            ids = [vocab.stoi.get(" ", 0)] + ids
        for _ in range(n_chars):
            ctx = np.array(ids[-self.C:])[None, :]
            _, _, probs = self._forward(ctx)
            ids.append(_sample_from(probs[0], temperature, top_k, rng))
        return vocab.decode(ids)
