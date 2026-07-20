"""Thin boto3 layer for S3 and DynamoDB.

Every client/resource is built through _endpoint(), which honors the
AWS_ENDPOINT_URL environment variable. That single indirection is what
lets the identical code run against:
  - moto            (in-process mock, unit tests -- no endpoint needed)
  - LocalStack      (AWS_ENDPOINT_URL=http://localhost:4566)
  - real AWS        (no AWS_ENDPOINT_URL; normal credentials/region)
with no code change -- the standard pattern for portable AWS code.

DynamoDB numbers are stored as Decimal (the SDK rejects float), so
revenue is converted on write and back to float on read.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _endpoint() -> dict:
    url = os.environ.get("AWS_ENDPOINT_URL")
    return {"endpoint_url": url} if url else {}


def s3_client():
    return boto3.client("s3", region_name=REGION, **_endpoint())


def dynamodb_resource():
    return boto3.resource("dynamodb", region_name=REGION, **_endpoint())


# --- S3 ------------------------------------------------------------------
def put_json(bucket: str, key: str, obj) -> None:
    s3_client().put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def get_json(bucket: str, key: str):
    body = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def list_keys(bucket: str, prefix: str = "") -> list[str]:
    s3 = s3_client()
    keys, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return keys


def read_events(bucket: str, prefix: str) -> list[dict]:
    """Read every newline-delimited-JSON object under a prefix."""
    s3 = s3_client()
    events = []
    for key in list_keys(bucket, prefix):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        events += [json.loads(line) for line in body.splitlines() if line.strip()]
    return events


# --- DynamoDB ------------------------------------------------------------
def _to_decimal(x):
    return Decimal(str(x)) if isinstance(x, float) else x


def upsert_category_total(table_name: str, category: str, metrics: dict) -> None:
    table = dynamodb_resource().Table(table_name)
    table.put_item(Item={
        "category": category,
        "revenue": _to_decimal(metrics["revenue"]),
        "units": metrics["units"],
        "orders": metrics["orders"],
    })


def get_category_total(table_name: str, category: str) -> dict | None:
    table = dynamodb_resource().Table(table_name)
    item = table.get_item(Key={"category": category}).get("Item")
    if item is None:
        return None
    return {"category": item["category"],
            "revenue": float(item["revenue"]),
            "units": int(item["units"]),
            "orders": int(item["orders"])}


def scan_categories(table_name: str) -> list[dict]:
    table = dynamodb_resource().Table(table_name)
    items = table.scan().get("Items", [])
    return [{"category": i["category"], "revenue": float(i["revenue"]),
             "units": int(i["units"]), "orders": int(i["orders"])}
            for i in items]
