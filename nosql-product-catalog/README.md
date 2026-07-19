# NoSQL Product Catalog (MongoDB)

A document-database project built around the thing document stores are
actually *for*: **heterogeneous, nested, schema-flexible data**. A
product catalog where every category carries different attributes, each
product embeds a variable-length array of reviews, and suppliers are
referenced rather than embedded — then analyzed with MongoDB's
**aggregation framework** (the document-world analog of SQL `GROUP BY` /
`JOIN`), backed by the right **indexes**.

This is the NoSQL counterpart to this repo's relational work
([student-outcomes-analysis](../student-outcomes-analysis/) runs the same
analysis across SQLite, Snowflake, and PySpark). The point here is
knowing *when a document model fits* and how to query it well — not
treating Mongo as a table with extra steps.

## Why a document model here

The catalog is deliberately the kind of data that fights a relational
schema:

- **Heterogeneous by category.** A book's `specs` sub-document has
  `author`, `pages`, `isbn`; an electronics item has `voltage`,
  `wattage`, `refresh_hz`; clothing has `sizes` (an array). In SQL that's
  either a sparse mega-table of mostly-NULLs or a pile of per-category
  join tables. As documents it's just... the document.
- **Embedded arrays.** Each product embeds its `reviews` (`rating`,
  `verified_purchase`, text) and its `tags`. These are read with the
  product and belong to it → **embed** them.
- **Shared entities.** Suppliers are referenced across many products →
  store once in a `suppliers` collection and **reference** by
  `supplier_id`, joined at query time with `$lookup`.

Embedding what's read together, referencing what's shared, is *the* core
MongoDB data-modeling decision — this project shows both on purpose.

## Data (`src/generate_data.py`, standard library only)

600 products across 4 categories, 30 suppliers, 1,555 embedded reviews;
deterministic under a fixed seed.

```
by category: books 147, electronics 209, clothing 141, home 103
```

## Pipeline

```bash
docker compose up -d                 # local MongoDB (see docker-compose.yml)
pip install -r requirements.txt
python src/generate_data.py          # build JSONL documents
python src/ingest.py                 # load + create indexes
python src/queries.py                # run the aggregation analytics
```

`src/ingest.py` bulk-loads both collections and builds the indexes the
queries rely on: a single-field index on `category`, a **multikey** index
on the `tags` array, a **compound** `(category, price)` index, a **text**
index on `name`, and unique indexes on the business keys.

## Queries (`src/queries.py`) and verified results

Each is an aggregation pipeline; the numbers below are asserted in
`tests/test_queries.py` against ground truth computed independently in
plain Python from the same seeded data.

**1. Catalog summary by category** (`$group` + `$multiply` for inventory value)

| category | n | avg price | inventory value |
|---|---:|---:|---:|
| electronics | 209 | \$610.97 | \$31,730,963 |
| home | 103 | \$209.14 | \$6,176,667 |
| clothing | 141 | \$91.62 | \$3,341,004 |
| books | 147 | \$26.69 | \$907,519 |

**2. Top-rated products** (`$project` with `$size`/`$avg` over the embedded
review array, filter to ≥3 reviews, sort, limit).

**3. Verified vs unverified reviews** (`$unwind` the embedded array, then
`$group`): 1,081 verified reviews (avg 3.93) vs 474 unverified (avg 3.98)
— barely different, a small honest finding rather than a manufactured one.

**4. Schema flexibility** (`$exists` on nested `specs` fields):

| `specs` field | documents |
|---|---:|
| `author` | 147 (books only) |
| `voltage` | 209 (electronics only) |
| `weight_kg` | 103 (home only) |
| `material` | **244** (clothing *and* home) |

`material` appearing in 244 documents — across two different category
schemas — is the schema-flexibility point in one number: same field name,
different document shapes, one query.

**5. Tag popularity** (`$unwind` the multikey `tags` array, `$group`): led
by `refurbished` (79), `portable` (76), `gaming` (75).

**6. Products per supplier country** (`$lookup` — the Mongo join — from
products to the referenced `suppliers` collection): US 223, CA 118, MX 85,
DE 67, IN 59, CN 48 (sums to all 600).

**Index check.** `queries.py` also runs `explain()` on a category filter
and confirms the winning plan uses the `category` index (an `IXSCAN`, not
a collection scan) — verifying the indexes from `ingest.py` are actually
doing work.

## Test

```bash
python -m pytest tests/ -v
```

Integration tests run against a real MongoDB and assert the exact counts
above. With no database reachable they **skip** (not fail), so the
repo-wide test run stays green anywhere; the project's CI job spins up a
`mongo:7` service container and runs them for real.

## Portability / limitations

- `MONGO_URI` drives the connection, so the identical code runs against
  local Docker, CI, or a managed cluster (Atlas) with no change.
- Reviews are embedded, which is right for read-with-product access but
  caps unbounded growth — a product with millions of reviews would move
  to a referenced `reviews` collection (the "subset" / "extended
  reference" pattern).
- No sharding or replica-set config here; this is a single-node modeling
  and query project, not an ops/scaling one.
