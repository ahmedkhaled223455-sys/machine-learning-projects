"""Execute every query in sql/queries.sql against the SQLite database and
save the results as a Markdown report.

Run:  python src/build_database.py && python src/run_sql_analysis.py
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "students.db"
SQL = ROOT / "sql" / "queries.sql"
OUT = ROOT / "results" / "sql_results.md"


def load_queries() -> list[tuple[str, str]]:
    text = SQL.read_text()
    blocks = re.split(r"--\s*name:\s*", text)[1:]
    queries = []
    for block in blocks:
        title, _, body = block.partition("\n")
        sql = body.strip().rstrip(";")
        queries.append((title.strip(), sql))
    return queries


def main():
    if not DB.exists():
        raise SystemExit("Run src/build_database.py first.")
    OUT.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    lines = ["# SQL analysis results", ""]
    for title, sql in load_queries():
        df = pd.read_sql(sql, con)
        lines += [f"## {title}", "", df.to_markdown(index=False), ""]
        print(f"[{title}]")
        print(df.to_string(index=False), "\n")
    con.close()
    OUT.write_text("\n".join(lines))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
