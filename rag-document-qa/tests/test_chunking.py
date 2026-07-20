"""Tests for word-window chunking."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import chunk_documents, chunk_text


def test_windows_and_overlap():
    text = " ".join(str(i) for i in range(200))   # 200 "words"
    chunks = chunk_text(text, "doc", size=80, overlap=20)
    # step = 60 -> windows at 0-79, 60-139, 120-199; the third reaches the
    # end, so no redundant tail chunk is emitted.
    assert [c.chunk_id for c in chunks] == [0, 1, 2]
    w0 = chunks[0].text.split()
    w1 = chunks[1].text.split()
    assert len(w0) == 80
    # last 20 of chunk0 overlap first 20 of chunk1
    assert w0[60:80] == w1[0:20]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", "d", size=10, overlap=10)


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", "d") == []


def test_chunk_documents_flattens():
    docs = {"a": "one two three four five", "b": "six seven eight"}
    chunks = chunk_documents(docs, size=3, overlap=1)
    assert {c.doc_id for c in chunks} == {"a", "b"}
    assert all(c.text for c in chunks)
