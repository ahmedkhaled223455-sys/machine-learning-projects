"""Retriever: builds the index from a document corpus and returns the
most relevant chunks for a query.

Ties the pieces together -- load docs, chunk them, fit the embedder,
embed every chunk, and load them into the vector store -- then answers
retrieve(query, k) by embedding the query and searching. This is the
"retrieval" in retrieval-augmented generation, and it needs no LLM, so
its quality can be measured directly (see evals/).
"""

from __future__ import annotations

from pathlib import Path

from chunking import chunk_documents
from embedder import TfidfEmbedder
from vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"


def load_corpus(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    """Return {doc_id: text} for every .md file in the corpus."""
    return {p.stem: p.read_text(encoding="utf-8")
            for p in sorted(docs_dir.glob("*.md"))}


class Retriever:
    def __init__(self, embedder=None, chunk_size: int = 80, overlap: int = 20):
        self.embedder = embedder or TfidfEmbedder()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = VectorStore()

    def build(self, documents: dict[str, str]) -> "Retriever":
        chunks = chunk_documents(documents, size=self.chunk_size,
                                 overlap=self.overlap)
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed_batch(texts)
        metadata = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text}
                    for c in chunks]
        self.store.add(vectors, metadata)
        return self

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        qvec = self.embedder.embed(query)
        hits = self.store.search(qvec, k=k)
        return [{**h.metadata, "score": h.score} for h in hits]


def build_default_retriever(docs_dir: Path = DOCS_DIR) -> Retriever:
    return Retriever().build(load_corpus(docs_dir))


if __name__ == "__main__":
    import sys

    r = build_default_retriever()
    query = " ".join(sys.argv[1:]) or "How does MinHash find duplicate documents?"
    print(f"Query: {query}\n")
    for hit in r.retrieve(query, k=3):
        print(f"[{hit['doc_id']}#{hit['chunk_id']}] score={hit['score']:.3f}")
        print(f"  {hit['text'][:140]}...\n")
