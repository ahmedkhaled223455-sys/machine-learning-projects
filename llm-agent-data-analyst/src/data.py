"""Build a small SQLite database the agent can query with its tools.

The dataset is the same 1,000-student exam-scores table used by
../student-outcomes-analysis, so the agent's answers can be checked
against numbers that project already verified. Column names are
snake_cased so the text-to-SQL is clean.

Run:  python src/data.py   (writes data/students.db)
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "students.csv"
DB = ROOT / "data" / "students.db"

COLUMNS = {
    "gender": "gender",
    "race/ethnicity": "race_ethnicity",
    "parental level of education": "parental_education",
    "lunch": "lunch",
    "test preparation course": "test_prep",
    "math score": "math_score",
    "reading score": "reading_score",
    "writing score": "writing_score",
}
INT_COLS = {"math_score", "reading_score", "writing_score"}


def build_db(db_path: Path = DB) -> Path:
    rows = list(csv.DictReader(CSV.open()))
    cols = list(COLUMNS.values())

    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS students")
    coldefs = ", ".join(
        f"{c} {'INTEGER' if c in INT_COLS else 'TEXT'}" for c in cols)
    con.execute(f"CREATE TABLE students ({coldefs})")
    placeholders = ", ".join("?" for _ in cols)
    for r in rows:
        values = []
        for raw, c in COLUMNS.items():
            v = r[raw]
            values.append(int(v) if c in INT_COLS else v)
        con.execute(f"INSERT INTO students VALUES ({placeholders})", values)
    con.commit()
    con.close()
    return db_path


def main():
    build_db()
    con = sqlite3.connect(DB)
    n = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    con.close()
    print(f"built {DB} with {n} rows")


if __name__ == "__main__":
    main()
