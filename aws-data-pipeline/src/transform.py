"""Pure transform logic -- no boto3, no AWS, no I/O.

Keeping the business logic in dependency-free functions is deliberate:
it's unit-testable anywhere (no mock, no network), and the AWS glue in
pipeline.py stays a thin shell around it. This is the same reason a
well-structured Lambda keeps its handler thin and its logic pure.

Run standalone to see the aggregates on the generated sample:
    python src/transform.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REQUIRED = ("event_id", "date", "category", "quantity", "unit_price")


def validate(event: dict) -> bool:
    """True if the event is well-formed enough to aggregate."""
    if not all(k in event for k in REQUIRED):
        return False
    # bool is a subclass of int in Python, so guard it explicitly on both
    # numeric fields -- True/False must not pass as 1/0.
    if isinstance(event["quantity"], bool) or not isinstance(event["quantity"], int):
        return False
    if event["quantity"] <= 0:
        return False
    if isinstance(event["unit_price"], bool) or not isinstance(
            event["unit_price"], (int, float)):
        return False
    if event["unit_price"] <= 0:
        return False
    return True


def clean(events: list[dict]) -> tuple[list[dict], int]:
    """Split raw events into (valid, n_dropped)."""
    valid = [e for e in events if validate(e)]
    return valid, len(events) - len(valid)


def aggregate(events: list[dict]) -> dict:
    """Aggregate valid events into per-category and per-day rollups.

    Returns a dict with:
      by_category : {category: {revenue, units, orders}}
      by_date     : {date: revenue}
      totals      : {revenue, units, orders}
    """
    by_cat = defaultdict(lambda: {"revenue": 0.0, "units": 0, "orders": 0})
    by_date = defaultdict(float)
    for e in events:
        line = e["quantity"] * e["unit_price"]
        c = by_cat[e["category"]]
        c["revenue"] += line
        c["units"] += e["quantity"]
        c["orders"] += 1
        by_date[e["date"]] += line

    by_cat = {k: {"revenue": round(v["revenue"], 2),
                  "units": v["units"], "orders": v["orders"]}
              for k, v in by_cat.items()}
    by_date = {k: round(v, 2) for k, v in sorted(by_date.items())}
    totals = {
        "revenue": round(sum(v["revenue"] for v in by_cat.values()), 2),
        "units": sum(v["units"] for v in by_cat.values()),
        "orders": sum(v["orders"] for v in by_cat.values()),
    }
    return {"by_category": by_cat, "by_date": by_date, "totals": totals}


def process(raw_events: list[dict]) -> tuple[dict, int]:
    """Full transform: validate then aggregate. Returns (aggregates, n_dropped)."""
    valid, dropped = clean(raw_events)
    return aggregate(valid), dropped


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data" / "events.jsonl"
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    agg, dropped = process(raw)
    print(f"{len(raw)} raw events | {dropped} dropped as invalid")
    print(f"total revenue ${agg['totals']['revenue']:,.2f} across "
          f"{agg['totals']['orders']} valid orders")
    print("by category:")
    for cat, v in sorted(agg["by_category"].items(),
                         key=lambda t: -t[1]["revenue"]):
        print(f"  {cat:12s} revenue ${v['revenue']:>12,.2f} "
              f"units {v['units']:>5d} orders {v['orders']:>5d}")
