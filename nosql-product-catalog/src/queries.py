"""Analytical queries over the catalog, using MongoDB's aggregation
framework -- the document-database analog of SQL GROUP BY / JOIN.

Each query is a function returning plain Python so tests can assert on
the results; main() runs them all and prints. The pipelines exercise the
operators an interviewer looks for: $group, $unwind (over embedded
arrays), $match, $project with $size/$avg, $lookup (the Mongo join),
$exists on nested fields, and an index-usage check via explain().

Run (after ingest.py):  python src/queries.py
"""

from __future__ import annotations

from db import get_db


# --- 1. catalog summary by category ($group + computed field) -------------
def catalog_summary(db) -> list[dict]:
    return list(db.products.aggregate([
        {"$group": {
            "_id": "$category",
            "n": {"$sum": 1},
            "avg_price": {"$avg": "$price"},
            "inventory_value": {"$sum": {"$multiply": ["$price", "$inventory"]}},
        }},
        {"$sort": {"inventory_value": -1}},
    ]))


# --- 2. top-rated products with >=3 reviews ($project $size/$avg) ----------
def rating_leaderboard(db, min_reviews: int = 3, limit: int = 5) -> list[dict]:
    return list(db.products.aggregate([
        {"$project": {
            "_id": 0,
            "product_id": 1,
            "category": 1,
            "n_reviews": {"$size": "$reviews"},
            "avg_rating": {"$avg": "$reviews.rating"},
        }},
        {"$match": {"n_reviews": {"$gte": min_reviews}}},
        {"$sort": {"avg_rating": -1, "n_reviews": -1}},
        {"$limit": limit},
    ]))


# --- 3. verified vs unverified review rating ($unwind embedded array) -----
def verified_rating(db) -> dict:
    rows = db.products.aggregate([
        {"$unwind": "$reviews"},
        {"$group": {
            "_id": "$reviews.verified_purchase",
            "n": {"$sum": 1},
            "avg_rating": {"$avg": "$reviews.rating"},
        }},
    ])
    return {("verified" if r["_id"] else "unverified"):
            {"n": r["n"], "avg_rating": r["avg_rating"]} for r in rows}


# --- 4. schema flexibility: $exists on heterogeneous nested specs ----------
def specs_field_presence(db, fields=("author", "voltage", "material",
                                     "weight_kg")) -> dict:
    return {f: db.products.count_documents({f"specs.{f}": {"$exists": True}})
            for f in fields}


# --- 5. tag popularity (multikey array via $unwind) -----------------------
def tag_popularity(db, limit: int = 6) -> list[dict]:
    return list(db.products.aggregate([
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "n": {"$sum": 1}}},
        {"$sort": {"n": -1, "_id": 1}},
        {"$limit": limit},
    ]))


# --- 6. products per supplier country ($lookup = the Mongo join) ----------
def products_by_supplier_country(db) -> list[dict]:
    return list(db.products.aggregate([
        {"$lookup": {
            "from": "suppliers",
            "localField": "supplier_id",
            "foreignField": "supplier_id",
            "as": "supplier",
        }},
        {"$unwind": "$supplier"},
        {"$group": {"_id": "$supplier.country", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
    ]))


def _find_index_name(plan) -> str | None:
    """Recursively pull the indexName out of an explain() winning plan
    (its nesting varies across Mongo versions)."""
    if isinstance(plan, dict):
        if "indexName" in plan:
            return plan["indexName"]
        for v in plan.values():
            found = _find_index_name(v)
            if found:
                return found
    elif isinstance(plan, list):
        for v in plan:
            found = _find_index_name(v)
            if found:
                return found
    return None


def category_query_uses_index(db) -> str | None:
    """Confirm a category filter + price sort is served by the compound
    (category, price) index, not a collection scan + in-memory sort. The
    sort makes index use deterministic (the planner won't choose a
    blocking sort when a supporting index exists)."""
    exp = db.products.find({"category": "electronics"}).sort("price", 1).explain()
    return _find_index_name(exp.get("queryPlanner", {}).get("winningPlan", {}))


def main():
    db = get_db()

    print("=== 1. catalog summary by category ===")
    for r in catalog_summary(db):
        print(f"  {r['_id']:12s} n={r['n']:3d} "
              f"avg_price={r['avg_price']:8.2f} "
              f"inventory_value={r['inventory_value']:14.2f}")

    print("\n=== 2. top-rated products (>=3 reviews) ===")
    for r in rating_leaderboard(db):
        print(f"  {r['product_id']} {r['category']:12s} "
              f"avg={r['avg_rating']:.2f} n={r['n_reviews']}")

    print("\n=== 3. verified vs unverified review rating ===")
    for k, v in verified_rating(db).items():
        print(f"  {k:12s} n={v['n']:4d} avg_rating={v['avg_rating']:.3f}")

    print("\n=== 4. schema flexibility ($exists on nested specs) ===")
    for f, n in specs_field_presence(db).items():
        print(f"  specs.{f:12s} present in {n} docs")

    print("\n=== 5. tag popularity ===")
    for r in tag_popularity(db):
        print(f"  {r['_id']:14s} {r['n']}")

    print("\n=== 6. products by supplier country ($lookup) ===")
    for r in products_by_supplier_country(db):
        print(f"  {r['_id']} {r['n']}")

    print("\n=== index check ===")
    idx = category_query_uses_index(db)
    print(f"  category filter uses index: {idx}")


if __name__ == "__main__":
    main()
