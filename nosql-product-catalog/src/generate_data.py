"""Generate a semi-structured product catalog: exactly the kind of data
a document database exists for.

Two collections' worth of documents are written as JSONL:

- products.jsonl : one document per product. Every product shares a few
  common fields, but the nested `specs` sub-document is DIFFERENT per
  category (a book has an author and page count; a monitor has voltage
  and refresh rate). Each product also embeds a variable-length array of
  review sub-documents. Representing this in a relational schema would
  take a sparse mega-table or several join tables; as documents it's
  natural -- that's the point.
- suppliers.jsonl : one document per supplier, REFERENCED by products
  via `supplier_id` (not embedded). Products embed what's read together
  (reviews, specs) and reference what's shared across many products
  (suppliers) -- the core document-modeling decision, demonstrated both
  ways so the `$lookup` in queries.py has something to join.

Pure standard library, deterministic under SEED. No database needed for
this step.

Run:  python src/generate_data.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 7
N_PRODUCTS = 600
N_SUPPLIERS = 30

random.seed(SEED)

CATEGORIES = ["books", "electronics", "clothing", "home"]

BRANDS = {
    "books": ["Vellum House", "Northwind Press", "Paper Lantern", "Quill & Co"],
    "electronics": ["Nyx", "Voltcraft", "Lumen", "Corebit", "Pulse"],
    "clothing": ["Northface Trail", "Loomstate", "Aria", "Bastion"],
    "home": ["Hearth", "Nestwell", "Terra", "Oaks"],
}

TAG_POOL = {
    "books": ["fiction", "nonfiction", "bestseller", "hardcover", "signed",
              "translated", "award-winner"],
    "electronics": ["wireless", "4k", "usb-c", "refurbished", "gaming",
                    "energy-star", "portable"],
    "clothing": ["organic", "waterproof", "slim-fit", "unisex", "recycled",
                 "limited"],
    "home": ["handmade", "eco", "flat-pack", "ceramic", "stackable"],
}

REVIEW_SNIPPETS = [
    "Exactly as described.", "Better than expected.", "Would buy again.",
    "Arrived damaged.", "Great value for the price.", "Runs a little small.",
    "Stopped working after a week.", "Beautiful build quality.",
    "Does the job.", "Overpriced for what it is.",
]


def make_specs(category: str) -> dict:
    """Category-specific nested sub-document -- heterogeneous by design."""
    if category == "books":
        return {
            "author": random.choice(
                ["A. Rehman", "J. Okafor", "M. Silva", "L. Zhang", "P. Duval"]),
            "pages": random.randint(120, 940),
            "language": random.choice(["en", "en", "en", "fr", "es"]),
            "isbn": "978-" + "".join(str(random.randint(0, 9)) for _ in range(10)),
            "format": random.choice(["hardcover", "paperback", "ebook"]),
        }
    if category == "electronics":
        return {
            "voltage": random.choice([5, 12, 110, 220]),
            "wattage": random.choice([15, 45, 65, 150, 300]),
            "warranty_months": random.choice([6, 12, 24, 36]),
            "refresh_hz": random.choice([60, 75, 120, 144, 240]),
        }
    if category == "clothing":
        return {
            "material": random.choice(["cotton", "wool", "polyester", "linen"]),
            "sizes": random.sample(["XS", "S", "M", "L", "XL", "XXL"],
                                   k=random.randint(2, 5)),
            "care": random.choice(["machine wash", "hand wash", "dry clean"]),
        }
    # home
    return {
        "material": random.choice(["oak", "ceramic", "steel", "bamboo"]),
        "weight_kg": round(random.uniform(0.3, 22.0), 1),
        "dimensions_cm": [random.randint(10, 120) for _ in range(3)],
        "assembly_required": random.choice([True, False]),
    }


def make_reviews(category: str) -> list[dict]:
    n = random.choices([0, 1, 2, 3, 4, 5, 8], weights=[3, 5, 7, 6, 4, 3, 1])[0]
    reviews = []
    for _ in range(n):
        # Ratings skew positive, as real review data does.
        rating = random.choices([1, 2, 3, 4, 5], weights=[1, 1, 2, 4, 6])[0]
        reviews.append({
            "user": f"u{random.randint(1000, 9999)}",
            "rating": rating,
            "text": random.choice(REVIEW_SNIPPETS),
            "verified_purchase": random.random() < 0.7,
        })
    return reviews


def make_products(suppliers: list[dict]) -> list[dict]:
    base = datetime(2024, 1, 1)
    supplier_ids = [s["supplier_id"] for s in suppliers]
    products = []
    for i in range(1, N_PRODUCTS + 1):
        category = random.choices(CATEGORIES, weights=[3, 4, 3, 2])[0]
        price = round({
            "books": random.uniform(6, 45),
            "electronics": random.uniform(20, 1200),
            "clothing": random.uniform(12, 180),
            "home": random.uniform(9, 400),
        }[category], 2)
        doc = {
            "product_id": f"P{i:05d}",
            "name": f"{random.choice(BRANDS[category])} {category[:-1].title()} {i}",
            "category": category,
            "brand": random.choice(BRANDS[category]),
            "price": price,
            "currency": "USD",
            "in_stock": random.random() < 0.85,
            "inventory": random.randint(0, 500),
            "supplier_id": random.choice(supplier_ids),   # REFERENCE
            "tags": random.sample(TAG_POOL[category],
                                  k=random.randint(1, min(4, len(TAG_POOL[category])))),
            "specs": make_specs(category),                # EMBEDDED, heterogeneous
            "reviews": make_reviews(category),            # EMBEDDED array
            "created_at": (base + timedelta(days=random.randint(0, 540))).isoformat(),
        }
        products.append(doc)
    return products


def make_suppliers() -> list[dict]:
    countries = ["US", "US", "CA", "DE", "CN", "MX", "IN"]
    suppliers = []
    for i in range(1, N_SUPPLIERS + 1):
        suppliers.append({
            "supplier_id": f"S{i:03d}",
            "name": f"Supplier {i}",
            "country": random.choice(countries),
            "rating": round(random.uniform(2.5, 5.0), 1),
            "active": random.random() < 0.9,
        })
    return suppliers


def main():
    DATA.mkdir(exist_ok=True)
    suppliers = make_suppliers()
    products = make_products(suppliers)

    with open(DATA / "suppliers.jsonl", "w") as f:
        for s in suppliers:
            f.write(json.dumps(s) + "\n")
    with open(DATA / "products.jsonl", "w") as f:
        for p in products:
            f.write(json.dumps(p) + "\n")

    n_reviews = sum(len(p["reviews"]) for p in products)
    by_cat = {c: sum(1 for p in products if p["category"] == c) for c in CATEGORIES}
    print(f"wrote {len(products)} products, {len(suppliers)} suppliers, "
          f"{n_reviews} embedded reviews")
    print("by category:", by_cat)


if __name__ == "__main__":
    main()
