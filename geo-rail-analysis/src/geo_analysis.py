"""Geospatial analysis with GeoPandas / Shapely.

Where geo_core.py does the spherical math by hand, this module uses the
standard geospatial stack for real geometry work:

- build a GeoDataFrame of stations (Points) and segments (LineStrings)
- **reproject before measuring**: lengths and buffers computed in degrees
  are meaningless, so everything metric happens in an equal-area/-distance
  projection (US National Atlas, EPSG:2163) and is converted to miles
- buffer a corridor and spatially join to find stations inside it
- plot the network to a map image

Requires geopandas + shapely (see requirements.txt). The core distance
analysis in validate_distances.py runs without them.

Run:  python src/geo_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import geopandas as gpd
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

WGS84 = "EPSG:4326"          # lat/lon degrees -- storage, not measurement
US_EQUAL_AREA = "EPSG:2163"  # US National Atlas Equal Area (meters)
METERS_PER_MILE = 1609.344


def load_stations() -> gpd.GeoDataFrame:
    df = pd.read_csv(DATA / "stations.csv")
    geometry = [Point(lon, lat) for lat, lon in zip(df.lat, df.lon)]  # x=lon, y=lat
    return gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)


def load_segments(stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    coords = {r.station: (r.geometry.x, r.geometry.y) for r in stations.itertuples()}
    df = pd.read_csv(DATA / "segments.csv")
    df = df[df["from"].isin(coords) & df["to"].isin(coords)].copy()
    geometry = [LineString([coords[a], coords[b]])
                for a, b in zip(df["from"], df["to"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)


def network_length_miles(segments: gpd.GeoDataFrame) -> float:
    """Total straight-line network length, measured in a projected CRS."""
    return float(segments.to_crs(US_EQUAL_AREA).length.sum() / METERS_PER_MILE)


def stations_near_corridor(stations: gpd.GeoDataFrame, segments: gpd.GeoDataFrame,
                           corridor: tuple[str, str], radius_miles: float
                           ) -> gpd.GeoDataFrame:
    """Stations within `radius_miles` of a given corridor.

    Buffering happens in the projected CRS -- buffering degrees would give a
    distance that changes with latitude.
    """
    a, b = corridor
    mask = (((segments["from"] == a) & (segments["to"] == b))
            | ((segments["from"] == b) & (segments["to"] == a)))
    line = segments[mask].to_crs(US_EQUAL_AREA)
    if line.empty:
        return stations.iloc[0:0]
    buffered = line.buffer(radius_miles * METERS_PER_MILE).union_all()
    projected = stations.to_crs(US_EQUAL_AREA)
    return stations[projected.within(buffered).values]


def nearest_station(stations: gpd.GeoDataFrame, lat: float, lon: float) -> tuple[str, float]:
    """Nearest station to a point, with distance in miles (projected CRS)."""
    point = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(US_EQUAL_AREA).iloc[0]
    projected = stations.to_crs(US_EQUAL_AREA)
    distances = projected.distance(point) / METERS_PER_MILE
    i = int(distances.idxmin())
    return str(stations.loc[i, "station"]), float(distances.loc[i])


def plot_network(stations: gpd.GeoDataFrame, segments: gpd.GeoDataFrame,
                 out: Path = None) -> Path:
    out = out or (RESULTS / "network_map.png")
    RESULTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    segments.plot(ax=ax, linewidth=1.2, color="#31708E", zorder=1)
    stations.plot(ax=ax, color="#B1624E", markersize=28, zorder=2)
    for r in stations.itertuples():
        ax.annotate(r.station, (r.geometry.x, r.geometry.y),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_title("Rail network (synthetic topology, real station coordinates)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    stations = load_stations()
    segments = load_segments(stations)

    print(f"Loaded {len(stations)} stations, {len(segments)} segments "
          f"(CRS {stations.crs})")
    print(f"Total network length: {network_length_miles(segments):,.0f} straight-line miles")

    bounds = stations.total_bounds  # minx, miny, maxx, maxy
    print(f"Territory extent: lon {bounds[0]:.1f} to {bounds[2]:.1f}, "
          f"lat {bounds[1]:.1f} to {bounds[3]:.1f}")

    near = stations_near_corridor(stations, segments, ("Los Angeles", "Barstow"), 60)
    print(f"Stations within 60 mi of the LA-Barstow corridor: "
          f"{', '.join(near.station) if len(near) else 'none'}")

    # An incident reported at coordinates -> closest terminal.
    incident = (35.5, -112.5)   # northern Arizona
    name, miles = nearest_station(stations, *incident)
    print(f"Incident at {incident}: nearest terminal is {name} ({miles:.0f} mi)")

    path = plot_network(stations, segments)
    print(f"Saved map to {path}")


if __name__ == "__main__":
    main()
