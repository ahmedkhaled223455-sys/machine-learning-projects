"""ETL orchestration: S3 raw events -> transform -> curated S3 + DynamoDB.

This is the thin glue around the pure logic in transform.py:
  1. read newline-delimited JSON events from s3://RAW_BUCKET/events/
  2. validate + aggregate (pure Python, no AWS)
  3. write the aggregate document to s3://CURATED_BUCKET/aggregates/latest.json
  4. upsert one item per category into the DynamoDB table for fast
     key-value lookups (e.g. a dashboard reading current category totals)

Returns a small summary dict (also used as the Lambda response).

Run (needs resources + raw data loaded; see README):
    python src/pipeline.py
"""

from __future__ import annotations

import aws_io
from config import (CURATED_BUCKET, CURATED_KEY, RAW_BUCKET, RAW_PREFIX,
                    TABLE_NAME)
from transform import process


def run(raw_bucket: str = RAW_BUCKET, curated_bucket: str = CURATED_BUCKET,
        table_name: str = TABLE_NAME) -> dict:
    raw = aws_io.read_events(raw_bucket, RAW_PREFIX)
    aggregates, dropped = process(raw)

    # curated aggregate document -> S3
    doc = {**aggregates, "events_read": len(raw), "events_dropped": dropped}
    aws_io.put_json(curated_bucket, CURATED_KEY, doc)

    # per-category rollups -> DynamoDB
    for category, metrics in aggregates["by_category"].items():
        aws_io.upsert_category_total(table_name, category, metrics)

    summary = {
        "events_read": len(raw),
        "events_dropped": dropped,
        "categories_written": len(aggregates["by_category"]),
        "total_revenue": aggregates["totals"]["revenue"],
        "curated_key": f"s3://{curated_bucket}/{CURATED_KEY}",
    }
    return summary


def main():
    summary = run()
    print("pipeline complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
