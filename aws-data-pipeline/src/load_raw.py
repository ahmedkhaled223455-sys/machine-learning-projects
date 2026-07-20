"""Upload the locally-generated events.jsonl into the S3 raw bucket,
simulating the upstream producer that lands data in the lake.

Run (after generate_events.py and setup_resources.py):
    python src/load_raw.py
"""

from __future__ import annotations

from pathlib import Path

from aws_io import s3_client
from config import RAW_BUCKET, RAW_PREFIX

ROOT = Path(__file__).resolve().parents[1]


def main():
    body = (ROOT / "data" / "events.jsonl").read_bytes()
    key = RAW_PREFIX + "events.jsonl"
    s3_client().put_object(Bucket=RAW_BUCKET, Key=key, Body=body)
    print(f"uploaded {len(body)} bytes to s3://{RAW_BUCKET}/{key}")


if __name__ == "__main__":
    main()
