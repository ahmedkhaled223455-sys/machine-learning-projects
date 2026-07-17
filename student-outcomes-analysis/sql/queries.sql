-- name: Overall summary
SELECT COUNT(*) AS students,
       ROUND(AVG(math_score), 1) AS avg_math,
       ROUND(AVG(reading_score), 1) AS avg_reading,
       ROUND(AVG(writing_score), 1) AS avg_writing
FROM students;

-- name: Average scores by test-prep completion
SELECT test_prep,
       COUNT(*) AS n,
       ROUND(AVG(math_score), 1) AS avg_math,
       ROUND(AVG(reading_score), 1) AS avg_reading,
       ROUND(AVG(writing_score), 1) AS avg_writing
FROM students
GROUP BY test_prep
ORDER BY test_prep;

-- name: Average scores by lunch status (socioeconomic proxy)
SELECT lunch,
       COUNT(*) AS n,
       ROUND(AVG(math_score), 1) AS avg_math,
       ROUND(AVG(reading_score), 1) AS avg_reading,
       ROUND(AVG(writing_score), 1) AS avg_writing
FROM students
GROUP BY lunch;

-- name: Prep effect within each lunch group (interaction check)
SELECT lunch,
       test_prep,
       COUNT(*) AS n,
       ROUND(AVG(math_score), 1) AS avg_math,
       ROUND(AVG(writing_score), 1) AS avg_writing
FROM students
GROUP BY lunch, test_prep
ORDER BY lunch, test_prep;

-- name: Prep completion rate by parental education (selection check)
SELECT parental_education,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(CASE WHEN test_prep = 'completed' THEN 1 ELSE 0 END), 1)
           AS pct_completed_prep,
       ROUND(AVG(math_score), 1) AS avg_math
FROM students
GROUP BY parental_education
ORDER BY pct_completed_prep DESC;

-- name: Share of at-risk students (math < 60) by prep status
SELECT test_prep,
       COUNT(*) AS n,
       SUM(CASE WHEN math_score < 60 THEN 1 ELSE 0 END) AS below_60,
       ROUND(100.0 * AVG(CASE WHEN math_score < 60 THEN 1 ELSE 0 END), 1)
           AS pct_below_60
FROM students
GROUP BY test_prep;

-- name: Top 3 math scores within each parental-education group (window function)
SELECT parental_education, gender, math_score
FROM (
    SELECT parental_education, gender, math_score,
           ROW_NUMBER() OVER (
               PARTITION BY parental_education
               ORDER BY math_score DESC
           ) AS rk
    FROM students
)
WHERE rk <= 3
ORDER BY parental_education, math_score DESC;
