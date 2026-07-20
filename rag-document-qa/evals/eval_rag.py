"""Evaluate the RAG system.

Two layers, matching the two halves of RAG:

1. Retrieval quality (recall@k) -- does the passage that actually
   contains the answer show up in the top-k retrieved chunks? This needs
   NO llm, so it runs anywhere and gives an honest, reproducible number
   for the retrieval half of the system.

2. Answer quality (optional, --answers) -- runs grounded generation
   against a real model and checks the answer contains the expected fact
   and cites a source. Requires ANTHROPIC_API_KEY.

Run:
    python evals/eval_rag.py            # retrieval recall@k (no key)
    python evals/eval_rag.py --answers  # + answer grading (needs key)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "eval_questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def evaluate_retrieval(retriever, questions: list[dict], ks=(1, 3)) -> dict:
    """recall@k = fraction of questions whose gold doc is in the top-k."""
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    per_q = []
    for item in questions:
        retrieved = retriever.retrieve(item["question"], k=max_k)
        docs = [r["doc_id"] for r in retrieved]
        ranks = [i for i, d in enumerate(docs) if d == item["gold_doc"]]
        rank = ranks[0] if ranks else None
        for k in ks:
            if rank is not None and rank < k:
                hits_at[k] += 1
        per_q.append({"question": item["question"], "gold": item["gold_doc"],
                      "top_docs": docs, "gold_rank": rank})
    n = len(questions)
    return {"n": n, "recall_at": {k: round(hits_at[k] / n, 3) for k in ks},
            "per_question": per_q}


def evaluate_answers(retriever, questions: list[dict], llm) -> dict:
    from rag import answer
    n_ok = 0
    per_q = []
    for item in questions:
        result = answer(item["question"], retriever, llm, k=3)
        text = result["answer"].lower()
        ok = all(kw.lower() in text for kw in item.get("answer_contains", []))
        cited = "[" in result["answer"]
        n_ok += ok and cited
        per_q.append({"question": item["question"], "ok": ok and cited,
                      "answer": result["answer"]})
    return {"n": len(questions), "n_ok": n_ok,
            "accuracy": round(n_ok / len(questions), 3), "per_question": per_q}


def main():
    from retriever import build_default_retriever

    questions = load_questions()
    retriever = build_default_retriever()

    ret = evaluate_retrieval(retriever, questions)
    print(f"Retrieval over {ret['n']} questions:")
    for k, v in ret["recall_at"].items():
        print(f"  recall@{k}: {v:.0%}")
    misses = [q for q in ret["per_question"] if q["gold_rank"] is None]
    if misses:
        print(f"  misses (gold not in top-{max(ret['recall_at'])}):")
        for m in misses:
            print(f"    - {m['question']}  (top: {m['top_docs']})")

    if "--answers" in sys.argv:
        from generation import AnthropicLLM
        ans = evaluate_answers(retriever, questions, AnthropicLLM())
        print(f"\nGrounded answers: {ans['n_ok']}/{ans['n']} correct + cited "
              f"({ans['accuracy']:.0%})")
        for q in ans["per_question"]:
            print(f"  [{'PASS' if q['ok'] else 'FAIL'}] {q['question']}")


if __name__ == "__main__":
    main()
