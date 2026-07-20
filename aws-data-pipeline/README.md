# AWS Data Pipeline (S3 → transform → DynamoDB, serverless-style)

An event-driven ETL pipeline on AWS primitives: raw sales events land in
an **S3** bucket, a **Lambda**-style handler validates and aggregates
them, and results are written to a **curated S3** document plus a
**DynamoDB** table for fast key-value lookups. Built with **boto3**,
endpoint-configurable so the identical code runs against a local
emulator or real AWS.

## Honest note on how this is verified

Standing up real AWS needs an account and credentials, which aren't part
of this repo. So the AWS integration is verified the way teams actually
test cloud code without touching production:

- **moto** — an in-process mock of the AWS APIs. The end-to-end pipeline
  test (`tests/test_pipeline_moto.py`) runs the real boto3 code against
  mocked S3 + DynamoDB and asserts the results. This runs in ordinary CI
  with no Docker and is the authoritative check.
- **LocalStack** — a containerized AWS emulator (`docker-compose.yml`)
  for running the pipeline against real HTTP AWS APIs on `localhost`, no
  account required.

The boto3 code is real and unchanged; only the endpoint differs. Point
`AWS_ENDPOINT_URL` at LocalStack, or unset it and supply real
credentials, and the same pipeline runs against AWS proper.

## Architecture

```
producer ─▶ S3 raw (sales-raw/events/*.jsonl)
                     │  S3 ObjectCreated → triggers
                     ▼
             Lambda: handler.lambda_handler
                     │  (pipeline.run)
             read events ─▶ validate + aggregate (pure Python)
                     │
      ┌──────────────┴───────────────┐
      ▼                              ▼
 S3 curated                     DynamoDB
 (aggregates/latest.json)       (category_totals, key=category)
```

The design keeps a hard line between **pure logic** (`transform.py`: no
boto3, unit-tested anywhere) and **AWS glue** (`aws_io.py`,
`pipeline.py`: thin, mock-tested). That separation is what makes a Lambda
testable without the Lambda runtime.

## What it demonstrates

- **S3** as a data lake: raw landing zone + curated output, paginated
  reads, JSON object writes (`aws_io.py`).
- **DynamoDB** (AWS's managed NoSQL): table with a partition key, upserts,
  key lookups and scans, with `Decimal` handling for numeric attributes.
- **Lambda** packaging: a thin `handler.lambda_handler(event, context)`
  wrapping the pipeline, ready to wire to an S3-trigger notification.
- **Portable boto3**: one `AWS_ENDPOINT_URL` indirection covers moto,
  LocalStack, and real AWS.
- **Data-quality discipline** carried into the cloud: malformed events
  (missing category, non-positive quantity, non-numeric price) are
  dropped before aggregation, not silently summed.

## Verified results (sample data, seed=11)

`generate_events.py` produces 2,000 raw events with ~5% deliberately
malformed. The transform layer's ground truth (asserted in
`tests/test_transform.py` on a small inline set, and reproduced here on
the full file):

```
2000 raw events | 101 dropped as invalid | 1899 valid
total revenue $1,570,165.15 across 1899 orders, over 30 days

by category:
  electronics  revenue $1,210,508.65  units 1950  orders 646
  home         revenue $  189,814.05  units  956  orders 324
  clothing     revenue $  136,309.13  units 1449  orders 472
  books        revenue $   33,533.32  units 1370  orders 457
```

## Run against LocalStack

```bash
docker compose up -d
pip install -r requirements.txt
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
python src/generate_events.py     # build sample raw data
python src/setup_resources.py     # create buckets + table
python src/load_raw.py            # land events in S3
python src/pipeline.py            # run ETL; writes curated S3 + DynamoDB
docker compose down
```

## Test

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

`test_transform.py` is pure Python (runs anywhere). `test_pipeline_moto.py`
mocks AWS in-process with `@mock_aws` — no Docker, no account — and
asserts the curated S3 document and the DynamoDB rollups end to end. The
project's CI job runs both on every push.

## Limitations / production notes

- Resource creation here is imperative boto3 for a one-command demo; in
  production this would be **Terraform** or **CloudFormation/CDK**.
- Re-aggregates the whole prefix per run — fine at this scale; a real
  high-volume pipeline would process only the triggering object and
  update DynamoDB incrementally (atomic `UpdateItem` counters).
- No IAM policies, DLQ, or partitioned/`Parquet` curated output; those
  are the natural next layers for a production deployment.
