"""Unit tests for the agent's tools -- pure SQLite, no model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools import dispatch_tool, get_schema, run_sql


def test_schema_lists_table_and_columns(db_path):
    schema = get_schema(db_path)
    assert "students" in schema
    for col in ("math_score", "test_prep", "parental_education"):
        assert col in schema


def test_run_sql_count(db_path):
    out = json.loads(run_sql(db_path, "SELECT COUNT(*) AS n FROM students"))
    assert out["rows"][0][0] == 1000


def test_run_sql_average(db_path):
    out = json.loads(run_sql(
        db_path, "SELECT ROUND(AVG(math_score),1) FROM students"))
    assert out["rows"][0][0] == 66.1


def test_run_sql_rejects_non_select(db_path):
    assert run_sql(db_path, "DROP TABLE students").startswith("ERROR")
    assert run_sql(db_path, "UPDATE students SET math_score=0").startswith("ERROR")


def test_run_sql_rejects_multiple_statements(db_path):
    out = run_sql(db_path, "SELECT 1; DROP TABLE students")
    assert out.startswith("ERROR")


def test_run_sql_reports_sql_errors(db_path):
    assert "SQL ERROR" in run_sql(db_path, "SELECT nope FROM students")


def test_run_sql_truncates_large_results(db_path):
    out = json.loads(run_sql(db_path, "SELECT * FROM students"))
    assert len(out["rows"]) == 50
    assert "truncated" in out.get("note", "")


def test_dispatch_routes_tools(db_path):
    assert "students" in dispatch_tool("get_schema", {}, db_path)
    assert dispatch_tool("nope", {}, db_path).startswith("ERROR")
