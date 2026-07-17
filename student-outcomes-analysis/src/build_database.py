"""Load the student performance CSV into a SQLite database.

Column names are normalized to snake_case so downstream SQL is clean.
Run:  python src/build_database.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

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


def main():
    df = pd.read_csv(CSV).rename(columns=COLUMNS)
    con = sqlite3.connect(DB)
    df.to_sql("students", con, if_exists="replace", index=False)
    n = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    cols = [r[1] for r in con.execute("PRAGMA table_info(students)")]
    con.close()
    print(f"wrote {DB.name}: {n} rows")
    print("columns:", ", ".join(cols))


if __name__ == "__main__":
    main()
