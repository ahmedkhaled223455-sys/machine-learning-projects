import java.util.*;

/**
 * Dijkstra's shortest-path search over the rail network.
 *
 * <p>Uses a priority queue keyed on tentative cost, so the complexity is
 * O(E log V). The planning metric is selectable: the cheapest route by
 * distance is often not the fastest by transit time, and reporting both is
 * the point of the tool.
 */
public final class Dijkstra {

    /** What we are minimizing. */
    public enum Metric { MILES, HOURS }

    /** The outcome of a search: the path plus both cost totals along it. */
    public static final class Result {
        public final boolean found;
        public final List<String> path;
        public final double totalMiles;
        public final double totalHours;

        Result(boolean found, List<String> path, double totalMiles, double totalHours) {
            this.found = found;
            this.path = path;
            this.totalMiles = totalMiles;
            this.totalHours = totalHours;
        }

        static Result notFound() {
            return new Result(false, Collections.emptyList(), 0, 0);
        }

        /** Cost under the metric that was optimized. */
        public double cost(Metric metric) {
            return metric == Metric.MILES ? totalMiles : totalHours;
        }

        @Override
        public String toString() {
            if (!found) {
                return "no route";
            }
            return String.join(" -> ", path)
                    + String.format("  [%.0f mi, %.1f h]", totalMiles, totalHours);
        }
    }

    private Dijkstra() {
    }

    /** Lowest-cost route from origin to destination under the given metric. */
    public static Result shortestPath(Graph graph, String origin, String destination, Metric metric) {
        if (!graph.contains(origin) || !graph.contains(destination)) {
            return Result.notFound();
        }
        if (origin.equals(destination)) {
            return new Result(true, List.of(origin), 0, 0);
        }

        Map<String, Double> best = new HashMap<>();
        Map<String, String> previous = new HashMap<>();
        Set<String> settled = new HashSet<>();
        PriorityQueue<String> queue =
                new PriorityQueue<>(Comparator.comparingDouble(s -> best.getOrDefault(s, Double.MAX_VALUE)));

        best.put(origin, 0.0);
        queue.add(origin);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            if (!settled.add(current)) {
                continue;                       // stale queue entry
            }
            if (current.equals(destination)) {
                break;
            }
            for (Graph.Edge edge : graph.neighbors(current)) {
                if (settled.contains(edge.to)) {
                    continue;
                }
                double candidate = best.get(current) + edge.weight(metric);
                if (candidate < best.getOrDefault(edge.to, Double.MAX_VALUE)) {
                    best.put(edge.to, candidate);
                    previous.put(edge.to, current);
                    queue.add(edge.to);         // re-inserted; stale copies skipped above
                }
            }
        }

        if (!best.containsKey(destination)) {
            return Result.notFound();
        }
        return buildResult(graph, origin, destination, previous);
    }

    /** Walk the predecessor chain back to the origin and total both metrics. */
    private static Result buildResult(Graph graph, String origin, String destination,
                                      Map<String, String> previous) {
        LinkedList<String> path = new LinkedList<>();
        for (String at = destination; at != null; at = previous.get(at)) {
            path.addFirst(at);
            if (at.equals(origin)) {
                break;
            }
        }
        double miles = 0;
        double hours = 0;
        for (int i = 0; i < path.size() - 1; i++) {
            String a = path.get(i);
            String b = path.get(i + 1);
            for (Graph.Edge e : graph.neighbors(a)) {
                if (e.to.equals(b)) {
                    miles += e.miles;
                    hours += e.hours;
                    break;
                }
            }
        }
        return new Result(true, path, miles, hours);
    }
}
