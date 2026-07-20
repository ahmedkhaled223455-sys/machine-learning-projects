"""AWS Lambda entry point.

Package this directory as a Lambda deployment (handler =
`handler.lambda_handler`) and it becomes an event-driven ETL: wire an S3
`ObjectCreated` notification on the raw bucket's `events/` prefix to this
function, and every upload triggers the aggregation and refreshes the
curated S3 document + DynamoDB rollups.

The handler stays deliberately thin -- parse the event, call the shared
pipeline, return a response -- with all real work in pipeline.py /
transform.py. That separation is what makes the logic testable without
the Lambda runtime (see tests/), which is exactly how it's verified here.
"""

from __future__ import annotations

import json

from pipeline import run


def lambda_handler(event, context=None):
    # An S3-trigger event carries bucket/key records; we don't need them
    # to re-aggregate the whole prefix, but log how we were invoked.
    n_records = len(event.get("Records", [])) if isinstance(event, dict) else 0
    summary = run()
    summary["invoked_with_records"] = n_records
    return {"statusCode": 200, "body": json.dumps(summary)}


if __name__ == "__main__":
    # Local smoke invocation with an empty event.
    print(lambda_handler({}))
