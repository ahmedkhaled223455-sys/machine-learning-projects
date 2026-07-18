-- Snowflake version of the student-outcomes analysis.
--
-- Setup (free 30-day Snowflake trial at signup.snowflake.com):
--   1. In Snowsight, run the DDL below.
--   2. Load data/students.csv via Data > Add Data > Load into a Table
--      (target STUDENTS; the load wizard maps headers automatically —
--      rename columns to match the DDL, or load then rename).
--   3. Run the analysis queries. Results should match results/sql_results.md
--      exactly (same data, same logic — SQLite and Snowflake both support
--      these window functions).

-- ---------- DDL ----------
CREATE WAREHOUSE IF NOT EXISTS ANALYSIS_WH
    WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60;
CREATE DATABASE IF NOT EXISTS EDU;
CREATE SCHEMA IF NOT EXISTS EDU.OUTCOMES;
USE EDU.OUTCOMES;

CREATE OR REPLACE TABLE STUDENTS (
    GENDER              VARCHAR,
    RACE_ETHNICITY      VARCHAR,
    PARENTAL_EDUCATION  VARCHAR,
    LUNCH               VARCHAR,
    TEST_PREP           VARCHAR,
    MATH_SCORE          INTEGER,
    READING_SCORE       INTEGER,
    WRITING_SCORE       INTEGER
);

-- ---------- Analysis ----------

-- Average scores by test-prep completion
SELECT TEST_PREP,
       COUNT(*)                    AS N,
       ROUND(AVG(MATH_SCORE), 1)    AS AVG_MATH,
       ROUND(AVG(READING_SCORE), 1) AS AVG_READING,
       ROUND(AVG(WRITING_SCORE), 1) AS AVG_WRITING
FROM STUDENTS
GROUP BY TEST_PREP
ORDER BY TEST_PREP;

-- Prep completion rate by parental education (selection check)
SELECT PARENTAL_EDUCATION,
       COUNT(*) AS N,
       ROUND(100 * AVG(IFF(TEST_PREP = 'completed', 1, 0)), 1)
           AS PCT_COMPLETED_PREP,
       ROUND(AVG(MATH_SCORE), 1) AS AVG_MATH
FROM STUDENTS
GROUP BY PARENTAL_EDUCATION
ORDER BY PCT_COMPLETED_PREP DESC;

-- Share of at-risk students (math < 60) by prep status
SELECT TEST_PREP,
       COUNT(*) AS N,
       ROUND(100 * AVG(IFF(MATH_SCORE < 60, 1, 0)), 1) AS PCT_BELOW_60
FROM STUDENTS
GROUP BY TEST_PREP;

-- Top 3 math scores within each parental-education group (window function)
SELECT PARENTAL_EDUCATION, GENDER, MATH_SCORE
FROM (
    SELECT PARENTAL_EDUCATION, GENDER, MATH_SCORE,
           ROW_NUMBER() OVER (
               PARTITION BY PARENTAL_EDUCATION
               ORDER BY MATH_SCORE DESC) AS RK
    FROM STUDENTS
)
WHERE RK <= 3
ORDER BY PARENTAL_EDUCATION, MATH_SCORE DESC;
