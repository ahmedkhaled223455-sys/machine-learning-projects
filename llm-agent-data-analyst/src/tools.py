"""The tools the agent can call, plus their JSON schemas.

This is the function-calling surface: TOOL_SCHEMAS is exactly the
`tools=[...]` payload the Anthropic Messages API expects, and
dispatch_tool() executes a tool the model asked for and returns a string
result to feed back as a tool_result. Keeping tools pure (SQLite only,
no LLM) means they're fully unit-testable without any model.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

# Anthropic tool-use schemas (name / description / input_schema).
TOOL_SCHEMAS = [
    {
        "name": "get_schema",
        "description": (
            "Return the schema of the 'students' table: column names and "
            "types, plus a few example values. Call this first to learn "
            "what columns exist before writing SQL."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_sql",
        "description": (
            "Run a single read-only SQL SELECT query against the SQLite "
            "'students' table and return the result rows as JSON. Only "
            "SELECT queries are allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A SQLite SELECT query over the students table.",
                }
            },
            "required": ["query"],
        },
    },
]

MAX_ROWS = 50


def get_schema(db_path: str | Path) -> str:
    con = sqlite3.connect(str(db_path))
    cols = con.execute("PRAGMA table_info(students)").fetchall()
    lines = ["table students ("]
    for _, name, ctype, *_ in cols:
        sample = con.execute(
            f"SELECT DISTINCT {name} FROM students LIMIT 3").fetchall()
        vals = ", ".join(str(s[0]) for s in sample)
        lines.append(f"  {name} {ctype}  -- e.g. {vals}")
    lines.append(")")
    con.close()
    return "\n".join(lines)


def run_sql(db_path: str | Path, query: str) -> str:
    q = query.strip().rstrip(";")
    # Read-only guard: single SELECT statement only.
    if ";" in q:
        return "ERROR: only a single statement is allowed."
    if not q.lower().startswith(("select", "with")):
        return "ERROR: only read-only SELECT queries are allowed."
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(q)
        headers = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        con.close()
        return f"SQL ERROR: {e}"
    con.close()
    truncated = len(rows) > MAX_ROWS
    out = {"columns": headers,
           "rows": [list(r) for r in rows[:MAX_ROWS]]}
    if truncated:
        out["note"] = f"result truncated to first {MAX_ROWS} rows"
    return json.dumps(out)


def dispatch_tool(name: str, tool_input: dict, db_path: str | Path) -> str:
    """Execute a tool the model requested; always returns a string."""
    if name == "get_schema":
        return get_schema(db_path)
    if name == "run_sql":
        return run_sql(db_path, tool_input.get("query", ""))
    return f"ERROR: unknown tool '{name}'"
