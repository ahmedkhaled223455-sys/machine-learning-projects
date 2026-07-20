"""A minimal in-memory vector store: the core of what a vector database
does, written out so it's inspectable.

Stores an (n x d) matrix of L2-normalized embeddings plus per-vector
metadata, and answers nearest-neighbor queries by cosine similarity
(a dot product, since vectors are normalized). This is an exact,
brute-force index -- O(n) per query, perfect for a small corpus.

A production vector database (FAISS, Chroma, pgvector) swaps this exact
scan for an approximate-nearest-neighbor index (HNSW/IVF) to stay fast at
millions of vectors, trading a little recall for speed. The interface --
add vectors + metadata, search by similarity -- is the same.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    score: float
    metadata: dict


class VectorStore:
    def __init__(self):
        self._vectors: np.ndarray | None = None   # (n, d), L2-normalized
        self._metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        self._vectors = (vectors if self._vectors is None
                         else np.vstack([self._vectors, vectors]))
        self._metadata.extend(metadata)

    def __len__(self) -> int:
        return len(self._metadata)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchHit]:
        if self._vectors is None or len(self._metadata) == 0:
            return []
        # Cosine similarity == dot product for normalized vectors.
        scores = self._vectors @ query_vector
        k = min(k, len(scores))
        # argpartition for the top-k, then sort just those by score desc.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchHit(score=float(scores[i]), metadata=self._metadata[i])
                for i in top]

    # -- persistence: a real store survives process restarts ---------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self._vectors)
        path.with_suffix(".json").write_text(json.dumps(self._metadata))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        store._vectors = np.load(path.with_suffix(".npy"))
        store._metadata = json.loads(path.with_suffix(".json").read_text())
        return store
