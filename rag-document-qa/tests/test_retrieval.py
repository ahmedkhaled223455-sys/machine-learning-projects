"""Retrieval-quality test: recall@k on the eval set (no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from eval_rag import evaluate_retrieval, load_questions


def test_recall_is_high(retriever):
    result = evaluate_retrieval(retriever, load_questions(), ks=(1, 3))
    # Distinct-topic corpus + TF-IDF -> retrieval should be essentially
    # perfect; guard well above chance (1/14 docs) without being brittle.
    assert result["recall_at"][1] >= 0.9
    assert result["recall_at"][3] >= 0.9
    assert result["recall_at"][3] >= result["recall_at"][1]


def test_specific_question_retrieves_gold(retriever):
    hits = retriever.retrieve("How does MinHash find duplicate documents?", k=3)
    assert hits[0]["doc_id"] == "minhash"
