"""Tests for the NumPy cosine vector store."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vector_store import VectorStore


def _normalize(v):
    return v / np.linalg.norm(v)


def _store():
    vecs = np.array([_normalize(v) for v in
                     [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]])
    meta = [{"doc_id": "x"}, {"doc_id": "y"}, {"doc_id": "z"}]
    s = VectorStore()
    s.add(vecs, meta)
    return s


def test_search_returns_nearest_first():
    s = _store()
    q = _normalize(np.array([1.0, 0.0]))
    hits = s.search(q, k=2)
    assert hits[0].metadata["doc_id"] == "x"     # exact match
    assert hits[1].metadata["doc_id"] == "z"     # closest of the rest
    assert hits[0].score >= hits[1].score


def test_len_and_mismatch_guard():
    s = _store()
    assert len(s) == 3
    try:
        s.add(np.zeros((2, 2)), [{"doc_id": "a"}])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_search_empty_store():
    assert VectorStore().search(np.array([1.0, 0.0]), k=3) == []


def test_save_load_roundtrip(tmp_path):
    s = _store()
    path = tmp_path / "index"
    s.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 3
    q = _normalize(np.array([0.0, 1.0]))
    assert loaded.search(q, k=1)[0].metadata["doc_id"] == "y"
