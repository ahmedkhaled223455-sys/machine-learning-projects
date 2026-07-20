"""Embedders: turn text into vectors for similarity search.

`TfidfEmbedder` is implemented from scratch with NumPy -- dependency-free,
fully reproducible, and a genuinely strong keyword-retrieval baseline. It
fits a vocabulary and IDF weights on the corpus, then maps any text to an
L2-normalized TF-IDF vector so cosine similarity is a plain dot product.

The `Embedder` protocol is the swap point: a production system would drop
in a dense semantic embedder (Voyage AI, OpenAI, or a local
sentence-transformers model) behind the same `fit`/`embed` interface. The
rest of the pipeline -- vector store, retriever, RAG -- is embedder-
agnostic. `ApiEmbedder` sketches that adapter.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfEmbedder:
    """From-scratch TF-IDF vectorizer with L2-normalized output."""

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        n_docs = len(corpus)
        df: Counter = Counter()
        tokenized = [set(tokenize(d)) for d in corpus]
        for toks in tokenized:
            df.update(toks)
        vocab = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab_ = {t: i for i, t in enumerate(vocab)}
        # Smoothed IDF (as in scikit-learn): log((1+N)/(1+df)) + 1.
        self.idf_ = np.array(
            [math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in vocab],
            dtype=np.float64)
        return self

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("call fit() before embed()")
        vecs = np.zeros((len(texts), len(self.vocab_)), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for tok, c in counts.items():
                j = self.vocab_.get(tok)
                if j is not None:
                    vecs[row, j] = c
            vecs[row] *= self.idf_
        # L2-normalize so cosine similarity == dot product.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    @property
    def dim(self) -> int:
        return len(self.vocab_)


class ApiEmbedder:
    """Adapter for a real embedding API (e.g. Voyage/OpenAI).

    Sketch only -- shows where a dense semantic embedder plugs in behind
    the same interface. Left un-wired so the project runs with no key.
    """

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def fit(self, corpus: list[str]):
        return self  # API embedders are stateless; nothing to fit

    def embed_batch(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            "Wire this to your embedding provider's batch endpoint, then "
            "L2-normalize the returned vectors.")
