"""Tests for the from-scratch geodesy -- NumPy only, no geospatial deps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geo_core import (bounding_box, circuity, compass_point, haversine,
                      initial_bearing, nearest_station)
from validate_distances import load_segments, load_stations, validate


def test_haversine_matches_known_distances():
    # Published great-circle distances; spherical model is within ~0.5%.
    assert abs(float(haversine(33.9425, -118.4081, 40.6413, -73.7781)) - 2475) < 25
    assert abs(float(haversine(47.6062, -122.3321, 45.5152, -122.6784)) - 145) < 5
    assert abs(float(haversine(29.7604, -95.3698, 29.9511, -90.0715)) - 318) < 5


def test_haversine_zero_and_symmetry():
    assert float(haversine(40.0, -100.0, 40.0, -100.0)) == 0.0
    ab = float(haversine(40.0, -100.0, 45.0, -110.0))
    ba = float(haversine(45.0, -110.0, 40.0, -100.0))
    assert abs(ab - ba) < 1e-9


def test_haversine_is_vectorized():
    d = haversine([40.0, 41.0], [-100.0, -101.0], [45.0, 46.0], [-110.0, -111.0])
    assert isinstance(d, np.ndarray) and d.shape == (2,) and np.all(d > 0)


def test_bearing_cardinal_directions():
    assert abs(float(initial_bearing(0, 0, 10, 0)) - 0) < 1e-6      # due north
    assert abs(float(initial_bearing(0, 0, 0, 10)) - 90) < 1e-6     # due east
    assert compass_point(0) == "N"
    assert compass_point(90) == "E"
    assert compass_point(225) == "SW"


def test_bounding_box():
    box = bounding_box([10, 20, 30], [-100, -110, -90])
    assert box["min_lat"] == 10 and box["max_lat"] == 30
    assert box["min_lon"] == -110 and box["max_lon"] == -90


def test_nearest_station_picks_closest():
    names = ["A", "B", "C"]
    lats = [40.0, 34.0, 47.0]
    lons = [-100.0, -118.0, -122.0]
    name, miles, bearing = nearest_station(34.5, -118.5, lats, lons, names)
    assert name == "B"
    assert miles < 60
    assert 0 <= bearing <= 360


def test_circuity_flags_impossible_mileage():
    # track shorter than the straight line is impossible
    assert float(circuity(90, 100)) < 1.0
    assert float(circuity(115, 100)) == 1.15


def test_network_data_passes_validation():
    report = validate(load_stations(), load_segments())
    assert report["n"] >= 25
    assert report["impossible"] == []            # no segment beats a straight line
    assert 1.0 < report["median_circuity"] < 1.5  # realistic corridor circuity


def test_validator_detects_a_bad_segment():
    stations = {"X": (40.0, -100.0), "Y": (45.0, -110.0)}
    true_gc = float(haversine(40.0, -100.0, 45.0, -110.0))
    segments = [{"from": "X", "to": "Y", "miles": true_gc * 0.8}]   # impossible
    report = validate(stations, segments)
    assert len(report["impossible"]) == 1
