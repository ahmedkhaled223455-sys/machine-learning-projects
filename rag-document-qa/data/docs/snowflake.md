# Snowflake

Snowflake is a cloud data warehouse that separates storage from compute,
so data is stored once and multiple independent compute clusters can query
it without contention. It runs on AWS, Azure, or GCP and scales compute up
or down on demand.

Snowflake speaks standard SQL and supports window functions, semi-
structured JSON data, and pay-per-second compute billing. Separating
storage and compute means heavy analytical queries do not block each other
and costs track actual usage.
