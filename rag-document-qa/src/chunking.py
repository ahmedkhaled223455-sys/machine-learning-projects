"""Split documents into overlapping word-window chunks.

Chunking matters in RAG: too large and a retrieved passage buries the
relevant sentence in noise; too small and it loses the context needed to
answer. Overlap keeps a fact that straddles a boundary intact in at least
one chunk. Word-window chunking (fixed size, fixed stride) is simple and
deterministic, which also makes it testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def _words(text: str) -> list[str]:
    return text.split()


def chunk_text(text: str, doc_id: str, size: int = 80,
               overlap: int = 20) -> list[Chunk]:
    """Split one document's text into overlapping word windows."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    # Collapse whitespace so chunks read cleanly.
    words = _words(re.sub(r"\s+", " ", text).strip())
    if not words:
        return []
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks


def chunk_documents(documents: dict[str, str], size: int = 80,
                    overlap: int = 20) -> list[Chunk]:
    """Chunk a {doc_id: text} mapping into a flat list of chunks."""
    out = []
    for doc_id, text in documents.items():
        out.extend(chunk_text(text, doc_id, size=size, overlap=overlap))
    return out
