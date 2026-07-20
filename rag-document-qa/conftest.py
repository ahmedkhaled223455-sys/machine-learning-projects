"""Shared test setup: make src/ importable and build the retriever once."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retriever import build_default_retriever  # noqa: E402


@pytest.fixture(scope="session")
def retriever():
    return build_default_retriever()
