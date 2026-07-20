# Apache Airflow

Apache Airflow orchestrates data pipelines as directed acyclic graphs
(DAGs), where each node is a task and edges encode dependencies. Airflow
schedules tasks, retries failures, and provides a UI to monitor runs,
making complex multi-step workflows reproducible and observable.

Airflow is used for ETL when a pipeline has ordered steps that must run on
a schedule, such as extract-then-transform-then-load jobs. Because
dependencies are explicit in the DAG, tasks run in the correct order and
independent tasks can run in parallel.
