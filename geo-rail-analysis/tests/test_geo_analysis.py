"""Tests for the GeoPandas/Shapely layer.

Skipped when geopandas isn't installed, so the repo-wide test run stays
green on a machine without the geospatial stack; the project's CI job
installs it and runs these for real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("geopandas", reason="geopandas not installed")
pytest.importorskip("shapely", reason="shapely not installed")

import geo_analysis as ga  # noqa: E402
from geo_core import haversine  # noqa: E402


@pytest.fixture(scope="module")
def network():
    stations = ga.load_stations()
    segments = ga.load_segments(stations)
    return stations, segments


def test_geodataframes_load_with_crs(network):
    stations, segments = network
    assert len(stations) >= 20
    assert len(segments) >= 25
    assert stations.crs.to_string() == ga.WGS84
    assert segments.geometry.geom_type.eq("LineString").all()


def test_projected_length_is_plausible(network):
    _, segments = network
    miles = ga.network_length_miles(segments)
    # straight-line total across ~30 continental segments
    assert 5_000 < miles < 15_000


def test_projected_length_differs_from_degree_length(network):
    """Guard against the classic bug: measuring length in degrees."""
    _, segments = network
    degrees = float(segments.length.sum())      # meaningless as a distance
    miles = ga.network_length_miles(segments)
    assert miles > degrees * 10                  # wildly different scales


def test_nearest_station_matches_haversine(network):
    stations, _ = network
    lat, lon = 35.5, -112.5
    name, miles = ga.nearest_station(stations, lat, lon)
    row = stations[stations.station == name].iloc[0]
    expected = float(haversine(lat, lon, row.lat, row.lon))
    assert abs(miles - expected) < 25            # projection vs sphere
    assert name in set(stations.station)


def test_corridor_buffer_contains_its_endpoints(network):
    stations, segments = network
    near = ga.stations_near_corridor(stations, segments, ("Los Angeles", "Barstow"), 60)
    found = set(near.station)
    assert "Los Angeles" in found and "Barstow" in found


def test_plot_writes_an_image(network, tmp_path):
    stations, segments = network
    out = ga.plot_network(stations, segments, out=tmp_path / "map.png")
    assert out.exists() and out.stat().st_size > 5_000
