"""Test setup: make src/ importable and build a fresh database per test."""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grants_auth import authenticate  # noqa: E402
from grants_service import GrantsService  # noqa: E402
from grants_store import build_db, connect  # noqa: E402

FIXED_TODAY = date(2026, 11, 10)


class GrantsEnv:
    def __init__(self, tmp_path):
        self.db = str(Path(tmp_path) / "grants.db")
        self.keys = build_db(self.db)
        self.conn = connect(self.db)

    def service(self, label: str, today: date = FIXED_TODAY) -> GrantsService:
        return GrantsService(self.conn, authenticate(self.conn, self.keys[label]), today=lambda: today)


@pytest.fixture
def grants_env(tmp_path):
    env = GrantsEnv(tmp_path)
    yield env
    env.conn.close()
