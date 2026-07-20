"""End-to-end pipeline test against mocked AWS (moto).

`mock_aws` intercepts every boto3 S3 and DynamoDB call in-process, so
this exercises the real pipeline code -- read from S3, transform, write
curated S3, upsert DynamoDB -- with zero real infrastructure. This is
the authoritative check that the boto3 glue in aws_io.py / pipeline.py is
correct; it runs in ordinary CI with just `pip install boto3 moto`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

moto = pytest.importorskip("moto", reason="moto not installed")
from moto import mock_aws  # noqa: E402

import aws_io  # noqa: E402
import pipeline  # noqa: E402
import setup_resources  # noqa: E402
from config import CURATED_BUCKET, CURATED_KEY, RAW_BUCKET, RAW_PREFIX, TABLE_NAME  # noqa: E402

RAW_EVENTS = [
    {"event_id": "E1", "date": "2024-01-01", "category": "books",
     "quantity": 2, "unit_price": 10.0},
    {"event_id": "E2", "date": "2024-01-01", "category": "electronics",
     "quantity": 1, "unit_price": 100.0},
    {"event_id": "E3", "date": "2024-01-02", "category": "electronics",
     "quantity": 3, "unit_price": 50.0},
    {"event_id": "E4", "date": "2024-01-02", "quantity": 1,
     "unit_price": 20.0},                     # invalid: missing category
    {"event_id": "E5", "date": "2024-01-02", "category": "home",
     "quantity": 0, "unit_price": 20.0},      # invalid: qty 0
]


def _seed_raw():
    """Create buckets + table, then land raw events in S3."""
    setup_resources.main()
    body = "\n".join(json.dumps(e) for e in RAW_EVENTS).encode("utf-8")
    aws_io.s3_client().put_object(
        Bucket=RAW_BUCKET, Key=RAW_PREFIX + "events.jsonl", Body=body)


@mock_aws
def test_pipeline_writes_curated_s3_and_dynamodb():
    _seed_raw()

    summary = pipeline.run()
    assert summary["events_read"] == 5
    assert summary["events_dropped"] == 2
    assert summary["categories_written"] == 2
    assert summary["total_revenue"] == 270.0

    # curated aggregate document landed in S3
    doc = aws_io.get_json(CURATED_BUCKET, CURATED_KEY)
    assert doc["totals"] == {"revenue": 270.0, "units": 6, "orders": 3}
    assert doc["events_dropped"] == 2

    # per-category rollups queryable in DynamoDB by key
    elec = aws_io.get_category_total(TABLE_NAME, "electronics")
    assert elec == {"category": "electronics", "revenue": 250.0,
                    "units": 4, "orders": 2}
    books = aws_io.get_category_total(TABLE_NAME, "books")
    assert books["revenue"] == 20.0
    assert aws_io.get_category_total(TABLE_NAME, "home") is None  # all dropped

    assert {c["category"] for c in aws_io.scan_categories(TABLE_NAME)} == {
        "books", "electronics"}


@mock_aws
def test_lambda_handler_returns_200():
    import handler
    _seed_raw()
    resp = handler.lambda_handler({"Records": [{"s3": {}}]}, None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["total_revenue"] == 270.0
    assert body["invoked_with_records"] == 1
