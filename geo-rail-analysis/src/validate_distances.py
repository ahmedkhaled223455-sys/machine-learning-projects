"""Validate tabulated segment mileages against geography.

Track distance can never be shorter than the great-circle distance between
its endpoints -- rails curve around terrain, they don't tunnel through the
earth. So the ratio of tabulated miles to great-circle miles (the
"circuity factor") must be >= 1, and in practice sits around 1.1-1.4 for
real corridors.

Anything below 1.0 is physically impossible and means the mileage table is
wrong. This check found exactly that in this repo's own network data
(Billings-Minneapolis listed at 700 mi against a 740 mi great-circle
distance), which is why the data now reads 850.

Run:  python src/validate_distances.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geo_core import circuity, haversine

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Below this, the mileage is impossible; above this, it's implausibly winding.
MIN_CIRCUITY = 1.0
MAX_PLAUSIBLE_CIRCUITY = 2.0


def load_stations(path: Path = DATA / "stations.csv") -> dict[str, tuple[float, float]]:
    with open(path) as f:
        return {r["station"]: (float(r["lat"]), float(r["lon"]))
                for r in csv.DictReader(f)}


def load_segments(path: Path = DATA / "segments.csv") -> list[dict]:
    with open(path) as f:
        return [{"from": r["from"], "to": r["to"], "miles": float(r["miles"])}
                for r in csv.DictReader(f)]


def validate(stations: dict, segments: list[dict]) -> dict:
    """Attach great-circle distance and circuity to each segment, and flag."""
    rows, impossible, implausible = [], [], []
    for seg in segments:
        a, b = seg["from"], seg["to"]
        if a not in stations or b not in stations:
            continue                      # unknown station: out of scope here
        gc = float(haversine(*stations[a], *stations[b]))
        c = float(circuity(seg["miles"], gc))
        row = {**seg, "great_circle_miles": gc, "circuity": c}
        rows.append(row)
        if c < MIN_CIRCUITY:
            impossible.append(row)
        elif c > MAX_PLAUSIBLE_CIRCUITY:
            implausible.append(row)
    ratios = np.array([r["circuity"] for r in rows])
    return {"segments": rows,
            "impossible": impossible,
            "implausible": implausible,
            "median_circuity": float(np.median(ratios)) if len(ratios) else float("nan"),
            "n": len(rows)}


def plot_circuity_map(stations: dict, report: dict, out: Path = None) -> Path:
    """Draw the network in lat/lon, coloring each segment by how indirect it
    is. Pure matplotlib, so this runs without the geospatial stack."""
    out = out or (RESULTS / "circuity_map.png")
    RESULTS.mkdir(exist_ok=True)
    ratios = [r["circuity"] for r in report["segments"]]
    norm = plt.Normalize(min(ratios), max(ratios))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9, 6))
    for r in report["segments"]:
        (lat1, lon1), (lat2, lon2) = stations[r["from"]], stations[r["to"]]
        ax.plot([lon1, lon2], [lat1, lat2], lw=2.0,
                color=cmap(norm(r["circuity"])), zorder=1)
    lons = [lon for _, lon in stations.values()]
    lats = [lat for lat, _ in stations.values()]
    ax.scatter(lons, lats, s=26, color="#B1624E", zorder=2)
    for name, (lat, lon) in stations.items():
        ax.annotate(name, (lon, lat), xytext=(3, 3),
                    textcoords="offset points", fontsize=7)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="circuity (track miles / great-circle miles)")
    ax.set_title("Rail network by route directness\n"
                 "(synthetic topology, real station coordinates)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    stations = load_stations()
    segments = load_segments()
    report = validate(stations, segments)

    print(f"Validated {report['n']} segments against great-circle distance")
    print(f"  median circuity factor: {report['median_circuity']:.2f}")
    print(f"  physically impossible (< {MIN_CIRCUITY}): {len(report['impossible'])}")
    print(f"  implausibly winding (> {MAX_PLAUSIBLE_CIRCUITY}): {len(report['implausible'])}")
    for r in report["impossible"] + report["implausible"]:
        print(f"    {r['from']} -> {r['to']}: {r['miles']:.0f} mi track vs "
              f"{r['great_circle_miles']:.0f} mi great-circle "
              f"(circuity {r['circuity']:.2f})")

    longest = max(report["segments"], key=lambda r: r["circuity"])
    print(f"  most indirect segment: {longest['from']} -> {longest['to']} "
          f"(circuity {longest['circuity']:.2f})")

    path = plot_circuity_map(stations, report)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
