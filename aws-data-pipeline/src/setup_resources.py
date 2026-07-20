"""Create the S3 buckets and DynamoDB table (idempotent).

For LocalStack or a fresh AWS account. In production this would be
Terraform or CloudFormation, not imperative boto3 -- but for a local /
CI integration run, creating resources in code keeps the demo one
command. The DDL-equivalent intent (bucket names, table key schema) is
identical either way.

Run:  python src/setup_resources.py
"""

from __future__ import annotations

from botocore.exceptions import ClientError

from aws_io import REGION, dynamodb_resource, s3_client
from config import CURATED_BUCKET, RAW_BUCKET, TABLE_NAME


def ensure_bucket(name: str) -> None:
    s3 = s3_client()
    try:
        s3.head_bucket(Bucket=name)
        print(f"bucket exists: {name}")
    except ClientError:
        # us-east-1 must NOT pass a LocationConstraint; every other region must.
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=name)
        else:
            s3.create_bucket(
                Bucket=name,
                CreateBucketConfiguration={"LocationConstraint": REGION})
        print(f"created bucket: {name}")


def ensure_table(name: str) -> None:
    ddb = dynamodb_resource()
    existing = [t.name for t in ddb.tables.all()]
    if name in existing:
        print(f"table exists: {name}")
        return
    table = ddb.create_table(
        TableName=name,
        KeySchema=[{"AttributeName": "category", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "category", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    print(f"created table: {name}")


def main():
    ensure_bucket(RAW_BUCKET)
    ensure_bucket(CURATED_BUCKET)
    ensure_table(TABLE_NAME)
    print("resources ready")


if __name__ == "__main__":
    main()
