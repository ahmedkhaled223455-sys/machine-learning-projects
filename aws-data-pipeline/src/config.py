"""Resource names, overridable by environment variable so the same code
targets different buckets/tables per environment (dev/CI/prod)."""

import os

RAW_BUCKET = os.environ.get("RAW_BUCKET", "sales-raw")
CURATED_BUCKET = os.environ.get("CURATED_BUCKET", "sales-curated")
TABLE_NAME = os.environ.get("TABLE_NAME", "category_totals")

RAW_PREFIX = "events/"
CURATED_KEY = "aggregates/latest.json"
