"""Integration tests for the aggregation queries.

These need a real MongoDB (the aggregation framework, $lookup, and
explain() are Mongo behaviors, not worth faking). If no Mongo is
reachable via MONGO_URI, the whole module is skipped -- so the repo-wide
`pytest` run stays green on machines without a database, while the
dedicated CI job (which starts a mongo service container) runs them for
real.

Expected values are ground truth: computed independently in plain Python
from the same seeded data (SEED=7 in generate_data.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pymongo", reason="pymongo not installed")

from pymongo.errors import PyMongoError  # noqa: E402

import ingest  # noqa: E402
import queries  # noqa: E402
from db import get_client, get_db  # noqa: E402


@pytest.fixture(scope="module")
def db():
    client = get_client()
    try:
        client.admin.command("ping")
    except PyMongoError:
        pytest.skip("no MongoDB reachable at MONGO_URI")
    # Load fresh, seeded data + indexes.
    ingest.main()
    return get_db(client)


def test_catalog_summary_counts_and_order(db):
    rows = queries.catalog_summary(db)
    by_cat = {r["_id"]: r for r in rows}
    assert by_cat["electronics"]["n"] == 209
    assert by_cat["books"]["n"] == 147
    assert by_cat["clothing"]["n"] == 141
    assert by_cat["home"]["n"] == 103
    # sorted by inventory_value desc -- electronics dominates
    assert [r["_id"] for r in rows][0] == "electronics"
    assert by_cat["electronics"]["avg_price"] == pytest.approx(610.97, abs=0.5)


def test_total_products(db):
    assert db.products.count_documents({}) == 600


def test_rating_leaderboard(db):
    top = queries.rating_leaderboard(db, min_reviews=3, limit=5)
    assert 0 < len(top) <= 5
    assert top[0]["avg_rating"] == pytest.approx(5.0)
    assert all(r["n_reviews"] >= 3 for r in top)
    # sorted by avg_rating desc
    ratings = [r["avg_rating"] for r in top]
    assert ratings == sorted(ratings, reverse=True)


def test_verified_rating_counts(db):
    res = queries.verified_rating(db)
    assert res["verified"]["n"] == 1081
    assert res["unverified"]["n"] == 474
    assert res["verified"]["n"] + res["unverified"]["n"] == 1555
    assert res["verified"]["avg_rating"] == pytest.approx(3.927, abs=0.01)


def test_schema_flexibility_exists(db):
    pres = queries.specs_field_presence(db)
    assert pres["author"] == 147      # books only
    assert pres["voltage"] == 209     # electronics only
    assert pres["weight_kg"] == 103   # home only
    # 'material' spans TWO category schemas (clothing + home) -- the
    # whole point of schema flexibility in a document store
    assert pres["material"] == 141 + 103


def test_tag_popularity(db):
    tags = queries.tag_popularity(db, limit=6)
    assert len(tags) == 6
    counts = [t["n"] for t in tags]
    assert counts == sorted(counts, reverse=True)
    top = {t["_id"]: t["n"] for t in tags}
    assert top.get("refurbished") == 79


def test_lookup_supplier_country(db):
    rows = queries.products_by_supplier_country(db)
    by_country = {r["_id"]: r["n"] for r in rows}
    assert sum(by_country.values()) == 600     # every product joins a supplier
    assert rows[0]["_id"] == "US"
    assert by_country["US"] == 223


def test_category_query_uses_index(db):
    idx = queries.category_query_uses_index(db)
    assert idx is not None            # not a collection scan
    assert "category" in idx
