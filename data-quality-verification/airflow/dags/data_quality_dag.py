"""Airflow DAG: daily data-quality verification run.

Orchestrates the pipeline as two dependent tasks:
    generate_dataset >> run_verification

Setup (using the official Airflow docker-compose quickstart,
https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/):

1. Follow the quickstart to get Airflow running with docker compose.
2. Copy this file into the quickstart's dags/ folder.
3. Mount this project into the containers by adding under x-airflow-common
   volumes in the quickstart's docker-compose.yaml:
       - /path/to/data-quality-verification:/opt/airflow/project
4. In the Airflow UI (http://localhost:8080), unpause `data_quality_pipeline`
   and trigger a run; both tasks should go green and results/ appears
   in the project folder.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

default_args = {
    "owner": "ahmed",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="data_quality_pipeline",
    description="Generate marketplace records and run layered quality checks",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["data-quality"],
) as dag:

    generate_dataset = BashOperator(
        task_id="generate_dataset",
        bash_command=f"cd {PROJECT} && python src/generate_dataset.py",
    )

    run_verification = BashOperator(
        task_id="run_verification",
        bash_command=f"cd {PROJECT} && python src/run_pipeline.py",
    )

    generate_dataset >> run_verification
