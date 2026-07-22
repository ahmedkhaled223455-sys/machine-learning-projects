"""Geodesy from first principles -- NumPy only, no geospatial dependencies.

Everything here is the math a geospatial stack does under the hood:
great-circle distance, initial bearing, bounding boxes, and
nearest-neighbour search over points on a sphere. Implementing it directly
makes the assumptions explicit (spherical earth, degrees vs radians) and
means the core analysis runs anywhere, with GeoPandas/Shapely layered on
top for real geometry operations (see geo_analysis.py).
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_MILES = 3958.7613


def haversine(lat1, lon1, lat2, lon2, radius: float = EARTH_RADIUS_MILES):
    """Great-circle distance between points, in miles. Vectorized."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, dtype=float),
                                              np.asarray(lon1, dtype=float),
                                              np.asarray(lat2, dtype=float),
                                              np.asarray(lon2, dtype=float)))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Compass bearing (degrees, 0-360) when departing point 1 toward point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, dtype=float),
                                              np.asarray(lon1, dtype=float),
                                              np.asarray(lat2, dtype=float),
                                              np.asarray(lon2, dtype=float)))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def compass_point(bearing: float) -> str:
    """Bearing to a 8-point compass label, for human-readable output."""
    points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return points[int((bearing + 22.5) % 360 // 45)]


def bounding_box(lats, lons) -> dict:
    """Min/max extent of a set of points -- the network's territory."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    return {"min_lat": float(lats.min()), "max_lat": float(lats.max()),
            "min_lon": float(lons.min()), "max_lon": float(lons.max())}


def nearest_station(lat: float, lon: float, station_lats, station_lons,
                    names) -> tuple[str, float, float]:
    """Closest station to an arbitrary point.

    Returns (name, distance_miles, bearing_degrees) -- the question asked
    when an incident is reported at a set of coordinates.
    """
    d = haversine(lat, lon, np.asarray(station_lats), np.asarray(station_lons))
    i = int(np.argmin(d))
    b = float(initial_bearing(lat, lon, station_lats[i], station_lons[i]))
    return names[i], float(d[i]), b


def circuity(track_miles, great_circle_miles):
    """Ratio of routed track distance to straight-line distance.

    Always >= 1 for real track: rails curve around terrain. A value below 1
    is physically impossible and means the tabulated mileage is wrong.
    """
    gc = np.asarray(great_circle_miles, dtype=float)
    return np.asarray(track_miles, dtype=float) / np.where(gc == 0, np.nan, gc)
