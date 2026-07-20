"""Unit tests for the pure transform logic -- no AWS, no mock, runs
anywhere. Small inline dataset with hand-computed expectations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transform import aggregate, clean, process, validate

EVENTS = [
    {"event_id": "E1", "date": "2024-01-01", "category": "books",
     "quantity": 2, "unit_price": 10.0, "region": "eu"},
    {"event_id": "E2", "date": "2024-01-01", "category": "electronics",
     "quantity": 1, "unit_price": 100.0, "region": "eu"},
    {"event_id": "E3", "date": "2024-01-02", "category": "electronics",
     "quantity": 3, "unit_price": 50.0, "region": "us-east"},
    {"event_id": "E4", "date": "2024-01-02",   # invalid: missing category
     "quantity": 1, "unit_price": 20.0},
    {"event_id": "E5", "date": "2024-01-02", "category": "home",
     "quantity": 0, "unit_price": 20.0},        # invalid: non-positive qty
    {"event_id": "E6", "date": "2024-01-02", "category": "home",
     "quantity": 1, "unit_price": "N/A"},        # invalid: non-numeric price
]


def test_validate_flags_bad_records():
    assert validate(EVENTS[0]) is True
    assert validate(EVENTS[3]) is False   # missing category
    assert validate(EVENTS[4]) is False   # qty 0
    assert validate(EVENTS[5]) is False   # price "N/A"


def test_validate_rejects_bool_quantity():
    # bool is a subclass of int in Python -- guard against True/False sneaking
    # through as quantity 1/0.
    bad = dict(EVENTS[0], quantity=True)
    assert validate(bad) is False


def test_clean_counts():
    valid, dropped = clean(EVENTS)
    assert len(valid) == 3
    assert dropped == 3


def test_aggregate_by_category():
    valid, _ = clean(EVENTS)
    agg = aggregate(valid)
    assert agg["by_category"]["books"] == {"revenue": 20.0, "units": 2, "orders": 1}
    assert agg["by_category"]["electronics"] == {
        "revenue": 250.0, "units": 4, "orders": 2}


def test_aggregate_totals_and_dates():
    agg, dropped = process(EVENTS)
    assert dropped == 3
    assert agg["totals"] == {"revenue": 270.0, "units": 6, "orders": 3}
    # two distinct dates among the valid events
    assert set(agg["by_date"]) == {"2024-01-01", "2024-01-02"}
    assert agg["by_date"]["2024-01-01"] == 120.0   # 20 + 100
    assert agg["by_date"]["2024-01-02"] == 150.0   # 3 * 50
