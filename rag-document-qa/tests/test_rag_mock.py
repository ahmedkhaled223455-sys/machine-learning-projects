"""Grounded-generation test with a scripted MockLLM (no API key).

Verifies the RAG assembly: real retrieval feeds the prompt, the prompt
carries the retrieved passages and the grounding instruction, and the
result reports the sources the answer is based on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation import MockLLM
from rag import SYSTEM, answer, build_prompt


def test_build_prompt_contains_passages_and_instruction():
    passages = [{"doc_id": "minhash", "chunk_id": 0, "text": "MinHash estimates Jaccard."},
                {"doc_id": "tfidf", "chunk_id": 1, "text": "TF-IDF weights terms."}]
    prompt = build_prompt("What is MinHash?", passages)
    assert "[1]" in prompt and "[2]" in prompt
    assert "MinHash estimates Jaccard." in prompt
    assert "source: minhash" in prompt
    assert "cite" in SYSTEM.lower()


def test_answer_uses_retrieval_and_returns_sources(retriever):
    mock = MockLLM(["MinHash estimates Jaccard similarity via hashing [1]."])
    result = answer("How does MinHash detect duplicates?", retriever, mock, k=3)

    assert "[1]" in result["answer"]
    assert len(result["sources"]) == 3
    # top source should be the MinHash doc (real retrieval ran)
    assert result["sources"][0]["doc_id"] == "minhash"
    # the prompt the LLM saw actually contained retrieved context
    user_prompt = mock.calls[0]["user"]
    assert "Context passages:" in user_prompt
    assert "minhash" in user_prompt
