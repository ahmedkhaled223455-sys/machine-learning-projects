"""Test setup: make src/ importable and provide a freshly-built SQLite
database for the tool and agent tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import build_db  # noqa: E402


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("data") / "students.db"
    build_db(path)
    return str(path)
