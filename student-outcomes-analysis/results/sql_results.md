# SQL analysis results

## Overall summary

|   students |   avg_math |   avg_reading |   avg_writing |
|-----------:|-----------:|--------------:|--------------:|
|       1000 |       66.1 |          69.2 |          68.1 |

## Average scores by test-prep completion

| test_prep   |   n |   avg_math |   avg_reading |   avg_writing |
|:------------|----:|-----------:|--------------:|--------------:|
| completed   | 358 |       69.7 |          73.9 |          74.4 |
| none        | 642 |       64.1 |          66.5 |          64.5 |

## Average scores by lunch status (socioeconomic proxy)

| lunch        |   n |   avg_math |   avg_reading |   avg_writing |
|:-------------|----:|-----------:|--------------:|--------------:|
| free/reduced | 355 |       58.9 |          64.7 |          63   |
| standard     | 645 |       70   |          71.7 |          70.8 |

## Prep effect within each lunch group (interaction check)

| lunch        | test_prep   |   n |   avg_math |   avg_writing |
|:-------------|:------------|----:|-----------:|--------------:|
| free/reduced | completed   | 131 |       63   |          70.4 |
| free/reduced | none        | 224 |       56.5 |          58.7 |
| standard     | completed   | 227 |       73.5 |          76.8 |
| standard     | none        | 418 |       68.1 |          67.6 |

## Prep completion rate by parental education (selection check)

| parental_education   |   n |   pct_completed_prep |   avg_math |
|:---------------------|----:|---------------------:|-----------:|
| some high school     | 179 |                 43   |       63.5 |
| bachelor's degree    | 118 |                 39   |       69.4 |
| associate's degree   | 222 |                 36.9 |       67.9 |
| some college         | 226 |                 34.1 |       67.1 |
| master's degree      |  59 |                 33.9 |       69.7 |
| high school          | 196 |                 28.6 |       62.1 |

## Share of at-risk students (math < 60) by prep status

| test_prep   |   n |   below_60 |   pct_below_60 |
|:------------|----:|-----------:|---------------:|
| completed   | 358 |         87 |           24.3 |
| none        | 642 |        236 |           36.8 |

## Top 3 math scores within each parental-education group (window function)

| parental_education   | gender   |   math_score |
|:---------------------|:---------|-------------:|
| associate's degree   | male     |          100 |
| associate's degree   | female   |          100 |
| associate's degree   | male     |           98 |
| bachelor's degree    | female   |          100 |
| bachelor's degree    | male     |          100 |
| bachelor's degree    | female   |           99 |
| high school          | female   |           99 |
| high school          | male     |           94 |
| high school          | male     |           90 |
| master's degree      | male     |           95 |
| master's degree      | female   |           94 |
| master's degree      | female   |           92 |
| some college         | female   |          100 |
| some college         | male     |          100 |
| some college         | male     |          100 |
| some high school     | female   |           97 |
| some high school     | male     |           94 |
| some high school     | male     |           94 |
