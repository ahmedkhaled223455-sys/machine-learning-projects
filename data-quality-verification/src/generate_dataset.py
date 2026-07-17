"""Generate a marketplace-style dataset with labeled quality defects.

Clean product-listing records are generated from templates, then a
controlled fraction is corrupted with realistic failure modes — each
record carries ground-truth defect labels so the verification pipeline
in run_pipeline.py can be scored honestly.

Failure modes injected (one or more per corrupted record):
  exact_duplicate     verbatim copy of another record
  near_duplicate      copy with small token edits (evades exact matching)
  missing_field       required field dropped
  wrong_type          numeric field stored as text like "N/A"
  out_of_range        impossible numeric value (price x100 unit error)
  encoding_artifact   mojibake sequences ("Ã©", "â€™") in text
  gibberish_text      description replaced by random characters
  repeated_boilerplate  description replaced by spammy repetition
  category_typo       category label misspelled / cased wrong
  stale_timestamp     timestamp far outside the collection window

Run:  python src/generate_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
SEED = 11
N_CLEAN = 4000
N_CORRUPT = 700

CATEGORIES = ["electronics", "home", "outdoors", "toys", "office", "grocery"]
ADJ = ["durable", "compact", "premium", "lightweight", "ergonomic",
       "wireless", "portable", "adjustable", "stainless", "rechargeable",
       "waterproof", "foldable", "classic", "modern", "heavy-duty"]
NOUN = ["backpack", "kettle", "lamp", "keyboard", "tent", "blender",
        "notebook", "speaker", "chair", "monitor", "bottle", "charger",
        "drill", "router", "desk"]
VERB = ["features", "includes", "offers", "provides", "comes with"]
TAIL = ["a two-year warranty", "an ergonomic grip", "fast shipping",
        "a machine-washable cover", "energy-efficient operation",
        "tool-free assembly", "a compact folding design",
        "an extended battery", "a reinforced frame", "quiet operation"]

PRICE_RANGE = {"electronics": (15, 900), "home": (8, 400),
               "outdoors": (10, 600), "toys": (5, 120),
               "office": (4, 350), "grocery": (2, 60)}


def make_clean(rng) -> dict:
    cat = str(rng.choice(CATEGORIES))
    lo, hi = PRICE_RANGE[cat]
    desc = (f"This {rng.choice(ADJ)} {rng.choice(NOUN)} "
            f"{rng.choice(VERB)} {rng.choice(TAIL)} and "
            f"{rng.choice(TAIL)}. The {rng.choice(ADJ)} design "
            f"{rng.choice(VERB)} {rng.choice(TAIL)}.")
    return {
        "record_id": None,                     # assigned later
        "category": cat,
        "price_usd": round(float(rng.uniform(lo, hi)), 2),
        "stock": int(rng.integers(0, 500)),
        "timestamp": int(rng.integers(1_735_689_600, 1_751_241_600)),  # 2025H1
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Corruption operators (each returns the defect label it applied)
# ---------------------------------------------------------------------------
def corrupt(record: dict, pool: list[dict], rng) -> list[str]:
    ops = rng.choice(["exact_duplicate", "near_duplicate", "missing_field",
                      "wrong_type", "out_of_range", "encoding_artifact",
                      "gibberish_text", "repeated_boilerplate",
                      "category_typo", "stale_timestamp"])
    if ops == "exact_duplicate":
        src = pool[rng.integers(len(pool))]
        record.update({k: src[k] for k in src if k != "record_id"})
    elif ops == "near_duplicate":
        src = pool[rng.integers(len(pool))]
        record.update({k: src[k] for k in src if k != "record_id"})
        words = record["description"].split()
        i = int(rng.integers(len(words)))
        words[i] = str(rng.choice(ADJ))
        record["description"] = " ".join(words)
        record["price_usd"] = round(record["price_usd"] * 1.01, 2)
    elif ops == "missing_field":
        field = str(rng.choice(["price_usd", "category", "timestamp"]))
        record[field] = None
    elif ops == "wrong_type":
        record["price_usd"] = str(rng.choice(["N/A", "call for price", "-", "TBD"]))
    elif ops == "out_of_range":
        record["price_usd"] = round(float(record["price_usd"]) * 100, 2)
    elif ops == "encoding_artifact":
        junk = str(rng.choice(["Ã©", "â€™", "Ã¼", "â€œ", "�"]))
        words = record["description"].split()
        for _ in range(3):
            i = int(rng.integers(len(words)))
            words[i] = words[i][:2] + junk + words[i][2:]
        record["description"] = " ".join(words)
    elif ops == "gibberish_text":
        chars = list("abcdefghijklmnopqrstuvwxyz ")
        record["description"] = "".join(
            rng.choice(chars, size=int(rng.integers(60, 140))))
    elif ops == "repeated_boilerplate":
        phrase = str(rng.choice(["BEST PRICE GUARANTEED", "buy now",
                                 "limited offer", "click here"]))
        record["description"] = " ".join([phrase] * int(rng.integers(8, 20)))
    elif ops == "category_typo":
        cat = record["category"]
        record["category"] = str(rng.choice(
            [cat.upper(), cat[:-1], cat + "s", " " + cat]))
    elif ops == "stale_timestamp":
        record["timestamp"] = int(rng.choice([946_684_800,      # year 2000
                                              2_082_758_400]))  # year 2036
    return [str(ops)]


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    clean = [make_clean(rng) for _ in range(N_CLEAN)]

    corrupted = []
    labels = {}
    for _ in range(N_CORRUPT):
        rec = make_clean(rng)
        defects = corrupt(rec, clean, rng)
        corrupted.append(rec)
        labels[id(rec)] = defects

    records = clean + corrupted
    order = rng.permutation(len(records))
    dataset, truth = [], []
    for new_id, idx in enumerate(order):
        rec = records[idx]
        rec["record_id"] = f"R{new_id:05d}"
        dataset.append(rec)
        truth.append({"record_id": rec["record_id"],
                      "defects": labels.get(id(rec), [])})

    (OUT / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in dataset))
    (OUT / "ground_truth.jsonl").write_text(
        "\n".join(json.dumps(t) for t in truth))
    n_bad = sum(1 for t in truth if t["defects"])
    print(f"wrote {len(dataset)} records ({n_bad} corrupted, "
          f"{n_bad / len(dataset):.1%}) -> {OUT}")


if __name__ == "__main__":
    main()
