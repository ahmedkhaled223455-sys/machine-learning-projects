"""PySpark port of the student-outcomes SQL analysis.

Same business questions as ../sql/queries.sql -- validated against
SQLite and Snowflake in ../results/sql_results.md -- now expressed with
the PySpark DataFrame API and Window functions, run locally via Spark's
local[*] mode. Same data, same logic, a third engine: SQLite for a dev
laptop, Snowflake for a cloud warehouse, Spark for distributed compute.

Mixes both common PySpark styles on purpose: most queries use the
DataFrame API (`groupBy().agg()`), one uses `spark.sql()` against a
registered temp view, and the last uses `pyspark.sql.window.Window` for
a partitioned top-N -- the same query that used `ROW_NUMBER() OVER
(PARTITION BY ...)` in the SQL versions.

Run:  python pyspark/spark_analysis.py   (from the project root)
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "students.csv"
RESULTS = ROOT / "results" / "pyspark_results.md"

# Same rename map as src/build_database.py, for a consistent schema
# across the SQLite, Snowflake, and Spark versions of this analysis.
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


def fmt(v) -> str:
    """1-decimal formatting for floats (avoids float repr noise like
    69.70000000000001 in the rendered table); everything else as-is."""
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def to_markdown(spark_df, cols: list[str]) -> str:
    """Render a small Spark DataFrame as a markdown table. Collects to
    the driver -- fine at this scale (a few dozen rows), not how you'd
    render a 10-billion-row result in production."""
    rows = spark_df.collect()
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def main():
    spark = (SparkSession.builder
             .appName("student-outcomes-analysis")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(str(CSV), header=True, inferSchema=True)
    for old, new in COLUMNS.items():
        df = df.withColumnRenamed(old, new)
    df = (df.withColumn("math_score", F.col("math_score").cast("int"))
            .withColumn("reading_score", F.col("reading_score").cast("int"))
            .withColumn("writing_score", F.col("writing_score").cast("int")))
    df.cache()

    sections = []

    # ---- Overall summary (DataFrame API) -----------------------------
    overall = df.agg(
        F.count("*").alias("students"),
        F.round(F.avg("math_score"), 1).alias("avg_math"),
        F.round(F.avg("reading_score"), 1).alias("avg_reading"),
        F.round(F.avg("writing_score"), 1).alias("avg_writing"),
    )
    sections.append(("Overall summary", overall,
                      ["students", "avg_math", "avg_reading", "avg_writing"]))

    # ---- Average scores by test-prep completion (DataFrame API) ------
    by_prep = (df.groupBy("test_prep")
               .agg(F.count("*").alias("n"),
                    F.round(F.avg("math_score"), 1).alias("avg_math"),
                    F.round(F.avg("reading_score"), 1).alias("avg_reading"),
                    F.round(F.avg("writing_score"), 1).alias("avg_writing"))
               .orderBy("test_prep"))
    sections.append(("Average scores by test-prep completion", by_prep,
                      ["test_prep", "n", "avg_math", "avg_reading", "avg_writing"]))

    # ---- Average scores by lunch status (DataFrame API) --------------
    by_lunch = (df.groupBy("lunch")
                .agg(F.count("*").alias("n"),
                     F.round(F.avg("math_score"), 1).alias("avg_math"),
                     F.round(F.avg("reading_score"), 1).alias("avg_reading"),
                     F.round(F.avg("writing_score"), 1).alias("avg_writing")))
    sections.append(("Average scores by lunch status (socioeconomic proxy)",
                      by_lunch, ["lunch", "n", "avg_math", "avg_reading", "avg_writing"]))

    # ---- Prep effect within each lunch group -- via spark.sql() ------
    df.createOrReplaceTempView("students")
    interaction = spark.sql("""
        SELECT lunch, test_prep,
               COUNT(*) AS n,
               ROUND(AVG(math_score), 1) AS avg_math,
               ROUND(AVG(writing_score), 1) AS avg_writing
        FROM students
        GROUP BY lunch, test_prep
        ORDER BY lunch, test_prep
    """)
    sections.append(("Prep effect within each lunch group (interaction check)",
                      interaction, ["lunch", "test_prep", "n", "avg_math", "avg_writing"]))

    # ---- Prep completion rate by parental education (DataFrame API) --
    by_educ = (df.groupBy("parental_education")
               .agg(F.count("*").alias("n"),
                    F.round(100 * F.avg(F.when(F.col("test_prep") == "completed", 1)
                                        .otherwise(0)), 1).alias("pct_completed_prep"),
                    F.round(F.avg("math_score"), 1).alias("avg_math"))
               .orderBy(F.desc("pct_completed_prep")))
    sections.append(("Prep completion rate by parental education (selection check)",
                      by_educ, ["parental_education", "n", "pct_completed_prep", "avg_math"]))

    # ---- Share of at-risk students (math < 60) by prep status ---------
    at_risk = (df.groupBy("test_prep")
               .agg(F.count("*").alias("n"),
                    F.sum(F.when(F.col("math_score") < 60, 1).otherwise(0)).alias("below_60"),
                    F.round(100 * F.avg(F.when(F.col("math_score") < 60, 1)
                                        .otherwise(0)), 1).alias("pct_below_60")))
    sections.append(("Share of at-risk students (math < 60) by prep status",
                      at_risk, ["test_prep", "n", "below_60", "pct_below_60"]))

    # ---- Top 3 math scores per parental-education group (Window) -----
    w = Window.partitionBy("parental_education").orderBy(F.desc("math_score"))
    top3 = (df.withColumn("rk", F.row_number().over(w))
            .filter(F.col("rk") <= 3)
            .select("parental_education", "gender", "math_score")
            .orderBy("parental_education", F.desc("math_score")))
    sections.append(("Top 3 math scores within each parental-education group (window function)",
                      top3, ["parental_education", "gender", "math_score"]))

    # ---- render + save --------------------------------------------------
    lines = ["# PySpark analysis results", "",
             "Same queries as `sql/queries.sql`, run through the PySpark "
             "DataFrame API and `Window` functions instead of raw SQL. "
             "Cross-check against `results/sql_results.md` -- every number "
             "below should match exactly (same data, same logic).", ""]
    for title, spark_df, cols in sections:
        print(f"\n## {title}")
        spark_df.show(20, truncate=False)
        lines.append(f"## {title}\n")
        lines.append(to_markdown(spark_df, cols))
        lines.append("")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text("\n".join(lines))
    print(f"\nSaved {RESULTS}")

    spark.stop()


if __name__ == "__main__":
    main()
