"""Retrieval-augmented generation: retrieve context, then answer from it.

The generation step is deliberately grounded: the prompt hands the model
numbered context passages and instructs it to answer ONLY from them, cite
the passages it used as [1], [2], and say it doesn't know if the answer
isn't present. That instruction is the main lever against hallucination
in a RAG system -- the model is told the retrieved context is the only
source of truth.

Returns the answer plus the source passages, so every answer is auditable
back to the documents it came from.

Run against real Claude (needs ANTHROPIC_API_KEY):
    python src/rag.py "Why can gradient boosting overfit small datasets?"
"""

from __future__ import annotations

from retriever import Retriever, build_default_retriever

SYSTEM = """You are a question-answering assistant. Answer the user's
question using ONLY the numbered context passages provided. Cite the
passages you use with bracketed numbers like [1] or [2]. If the answer is
not contained in the context, reply exactly: "I don't know based on the
provided context." Keep the answer to a few sentences."""


def build_prompt(question: str, passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(f"[{i}] (source: {p['doc_id']})\n{p['text']}")
    context = "\n\n".join(blocks)
    return (f"Context passages:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer using only the context above, with citations.")


def answer(question: str, retriever: Retriever, llm, k: int = 3) -> dict:
    passages = retriever.retrieve(question, k=k)
    prompt = build_prompt(question, passages)
    response = llm.complete(SYSTEM, prompt)
    return {
        "question": question,
        "answer": response,
        "sources": [{"doc_id": p["doc_id"], "chunk_id": p["chunk_id"],
                     "score": p["score"]} for p in passages],
    }


def main():
    import sys

    from generation import AnthropicLLM

    retriever = build_default_retriever()
    question = (" ".join(sys.argv[1:])
                or "What problem do vector databases solve?")
    result = answer(question, retriever, AnthropicLLM())
    print(f"Q: {result['question']}\n")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['doc_id']}#{s['chunk_id']} (score {s['score']:.3f})")


if __name__ == "__main__":
    main()
