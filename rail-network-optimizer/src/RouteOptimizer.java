import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * CLI: plan a freight route across the network and stress-test it.
 *
 * <p>Reports the shortest route by distance and the fastest by transit time
 * (they are frequently different routes), then simulates a closure on the
 * first segment of the shortest route and re-plans, reporting the detour cost
 * -- the question an operations team actually asks when a segment goes down.
 *
 * <pre>
 *   java -cp out RouteOptimizer Seattle Chicago
 * </pre>
 */
public final class RouteOptimizer {

    public static void main(String[] args) throws Exception {
        Path csv = Paths.get("data", "network.csv");
        Graph network = RailNetwork.fromCsv(csv);

        String origin = args.length > 0 ? args[0] : "Seattle";
        String destination = args.length > 1 ? args[1] : "Chicago";

        System.out.printf("Network: %d stations, %d segments%n",
                network.stations().size(), network.segmentCount());
        System.out.printf("Planning %s -> %s%n%n", origin, destination);

        Dijkstra.Result byMiles = Dijkstra.shortestPath(network, origin, destination,
                Dijkstra.Metric.MILES);
        Dijkstra.Result byHours = Dijkstra.shortestPath(network, origin, destination,
                Dijkstra.Metric.HOURS);

        if (!byMiles.found) {
            System.out.println("No route exists between those stations.");
            return;
        }

        System.out.println("Shortest by distance:");
        System.out.println("  " + byMiles);
        System.out.println("Fastest by transit time:");
        System.out.println("  " + byHours);
        System.out.println(byMiles.path.equals(byHours.path)
                ? "  (same route under both metrics)"
                : "  (different route -- the shortest path is not the fastest)");

        simulateDisruption(network, byMiles, origin, destination);
    }

    /** Close the first segment of the optimal route and re-plan. */
    private static void simulateDisruption(Graph network, Dijkstra.Result baseline,
                                           String origin, String destination) {
        if (baseline.path.size() < 2) {
            return;
        }
        String a = baseline.path.get(0);
        String b = baseline.path.get(1);

        Graph disrupted = network.copy();
        disrupted.removeSegment(a, b);

        System.out.printf("%nDisruption: segment %s <-> %s closed%n", a, b);
        Dijkstra.Result detour = Dijkstra.shortestPath(disrupted, origin, destination,
                Dijkstra.Metric.MILES);
        if (!detour.found) {
            System.out.println("  No alternative route -- the network is cut.");
            return;
        }
        System.out.println("  Detour: " + detour);
        System.out.printf("  Added distance: %.0f mi (+%.1f%%)%n",
                detour.totalMiles - baseline.totalMiles,
                100.0 * (detour.totalMiles - baseline.totalMiles) / baseline.totalMiles);
    }
}
