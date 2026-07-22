# Rail Network Route Optimizer (Java)

A freight-routing tool over a rail network: **Dijkstra shortest-path search
on a weighted graph**, with two competing planning metrics (distance vs.
transit time) and **disruption rerouting** when a segment goes out of
service. Written in plain **Java** — no frameworks, no external
dependencies, builds with nothing but a JDK.

This is the operations-research side of the portfolio: graph algorithms and
network optimization rather than statistical modeling.

## Why two metrics matter

The cheapest route by mileage is often not the fastest, and a planner needs
both. The tool reports each and flags when they disagree:

```
$ java -cp out RouteOptimizer "Los Angeles" "Las Vegas"
Network: 22 stations, 30 segments
Planning Los Angeles -> Las Vegas

Shortest by distance:
  Los Angeles -> Las Vegas  [270 mi, 9.0 h]
Fastest by transit time:
  Los Angeles -> Barstow -> Las Vegas  [275 mi, 7.0 h]
  (different route -- the shortest path is not the fastest)
```

Five extra miles buys two hours, because the direct corridor is slower per
mile. That trade-off is invisible if you only optimize distance.

## Disruption rerouting

Every run closes the first segment of the optimal route and re-plans — the
question an operations team asks when a segment goes down:

```
$ java -cp out RouteOptimizer Seattle Chicago
Shortest by distance:
  Seattle -> Spokane -> Billings -> Minneapolis -> Chicago  [2010 mi, 48.5 h]

Disruption: segment Seattle <-> Spokane closed
  Detour: Seattle -> Portland -> Boise -> Salt Lake City -> Denver -> Kansas City -> Chicago  [2575 mi, 62.0 h]
  Added distance: 565 mi (+28.1%)
```

Losing one segment costs 565 miles here — the kind of number that justifies
redundancy on a corridor. `Graph.copy()` makes the scenario
non-destructive, so the base network is never mutated.

> Segment mileages in `data/network.csv` are validated against great-circle
> distance by the companion [geo-rail-analysis](../geo-rail-analysis/)
> project — which caught one segment tabulated *shorter* than a straight
> line and led to the correction now in this data.

## Design

| File | Role |
|---|---|
| `src/Graph.java` | Undirected weighted graph; adjacency lists, dual weights (miles/hours), segment removal for outages, deep copy for scenarios. |
| `src/Dijkstra.java` | Priority-queue shortest path, O(E log V), with a selectable metric; returns the path plus **both** cost totals. |
| `src/RailNetwork.java` | CSV loader that skips malformed rows rather than failing the load. |
| `src/RouteOptimizer.java` | CLI: plan, compare metrics, simulate a closure. |
| `tests/NetworkTests.java` | 23 assertion-based tests, no test framework required. |

## Run

```bash
javac -d out src/*.java tests/*.java

java -cp out NetworkTests                              # 23 passed, 0 failed
java -cp out RouteOptimizer Seattle Chicago
java -cp out RouteOptimizer "Los Angeles" "New Orleans"
```

## Data

`data/network.csv` is a **synthetic, illustrative** network of 22 western
U.S. cities and 30 segments (`from,to,miles,hours`) — representative
corridors and plausible distances, not any carrier's actual network or
proprietary data.

## Limitations / next steps

- Static weights: no capacity, congestion, or time-of-day variation; a real
  planner would model segment capacity and yard dwell time.
- Single-commodity point-to-point routing — no car-blocking, train makeup,
  or multi-leg scheduling.
- Dijkstra assumes non-negative weights (true here); A\* with a geographic
  heuristic would speed up search on a continental-scale network.
