"""Generate raw sales events -- the kind of newline-delimited JSON that
lands in an S3 'raw' bucket from an upstream producer.

Deliberately includes a slice of malformed events (missing category,
non-positive quantity, non-numeric price) so the transform layer has
real validation work to do -- the same data-quality discipline as the
rest of this repo, applied to a cloud landing zone.

Pure standard library, deterministic under SEED.

Run:  python src/generate_events.py
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 11
N_EVENTS = 2000

random.seed(SEED)

CATEGORIES = ["books", "electronics", "clothing", "home"]
REGIONS = ["us-east", "us-west", "eu", "apac"]
PRICE = {"books": (6, 45), "electronics": (20, 1200),
         "clothing": (12, 180), "home": (9, 400)}


def main():
    DATA.mkdir(exist_ok=True)
    base = date(2024, 1, 1)
    events = []
    for i in range(1, N_EVENTS + 1):
        cat = random.choices(CATEGORIES, weights=[3, 4, 3, 2])[0]
        lo, hi = PRICE[cat]
        ev = {
            "event_id": f"E{i:06d}",
            "date": (base + timedelta(days=random.randint(0, 29))).isoformat(),
            "product_id": f"P{random.randint(1, 600):05d}",
            "category": cat,
            "quantity": random.randint(1, 5),
            "unit_price": round(random.uniform(lo, hi), 2),
            "region": random.choice(REGIONS),
        }
        # Inject ~5% malformed records for the validation layer to catch.
        r = random.random()
        if r < 0.02:
            ev.pop("category")                       # missing required field
        elif r < 0.035:
            ev["quantity"] = random.choice([0, -1])  # non-positive quantity
        elif r < 0.05:
            ev["unit_price"] = "N/A"                 # non-numeric price
        events.append(ev)

    with open(DATA / "events.jsonl", "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    print(f"wrote {len(events)} raw events to {DATA / 'events.jsonl'}")


if __name__ == "__main__":
    main()
