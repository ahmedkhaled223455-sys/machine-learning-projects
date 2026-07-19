"""Ingest the generated JSONL documents into MongoDB and build indexes.

Idempotent: drops and recreates the two collections so re-running gives
a clean load. Then creates the indexes the analytical queries rely on --
including a multikey index on the `tags` array and a compound index --
and prints each collection's index list so you can see what was built.

Run (needs a reachable Mongo -- see docker-compose.yml):
    python src/ingest.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pymongo import ASCENDING, TEXT

from db import get_db

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    db = get_db()

    products = load_jsonl(DATA / "products.jsonl")
    suppliers = load_jsonl(DATA / "suppliers.jsonl")

    db.products.drop()
    db.suppliers.drop()
    db.products.insert_many(products)
    db.suppliers.insert_many(suppliers)
    print(f"inserted {db.products.count_documents({})} products, "
          f"{db.suppliers.count_documents({})} suppliers")

    # Indexes ---------------------------------------------------------------
    # unique business key
    db.products.create_index([("product_id", ASCENDING)], unique=True)
    # single-field: category is the most common filter/group key
    db.products.create_index([("category", ASCENDING)])
    # multikey: indexing an array field indexes every element -- makes
    # tag lookups ($in / equality on tags) index-backed
    db.products.create_index([("tags", ASCENDING)])
    # compound: supports "filter by category, sort by price" in one index
    db.products.create_index([("category", ASCENDING), ("price", ASCENDING)])
    # text index for keyword search over product names
    db.products.create_index([("name", TEXT)])
    db.suppliers.create_index([("supplier_id", ASCENDING)], unique=True)

    print("\nproducts indexes:")
    for name, spec in db.products.index_information().items():
        print(f"  {name}: {spec['key']}")
    print("suppliers indexes:")
    for name, spec in db.suppliers.index_information().items():
        print(f"  {name}: {spec['key']}")


if __name__ == "__main__":
    main()
