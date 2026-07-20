"""Tests for the from-scratch TF-IDF embedder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedder import TfidfEmbedder, tokenize


def test_tokenize():
    assert tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]


def test_vectors_are_l2_normalized():
    emb = TfidfEmbedder().fit(["the cat sat", "the dog ran", "a cat meowed"])
    v = emb.embed("the cat sat")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_identical_text_has_cosine_one():
    emb = TfidfEmbedder().fit(["alpha beta", "beta gamma", "gamma delta"])
    a = emb.embed("alpha beta")
    b = emb.embed("alpha beta")
    assert abs(float(a @ b) - 1.0) < 1e-9


def test_common_word_downweighted_vs_rare_word():
    # "the" appears in every doc (high df -> low idf); "zebra" in one.
    emb = TfidfEmbedder().fit(["the zebra", "the cat", "the dog"])
    idf_the = emb.idf_[emb.vocab_["the"]]
    idf_zebra = emb.idf_[emb.vocab_["zebra"]]
    assert idf_zebra > idf_the


def test_disjoint_texts_are_orthogonal():
    emb = TfidfEmbedder().fit(["apple banana", "carrot potato"])
    sim = float(emb.embed("apple banana") @ emb.embed("carrot potato"))
    assert abs(sim) < 1e-9


def test_embed_before_fit_raises():
    try:
        TfidfEmbedder().embed("x")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
