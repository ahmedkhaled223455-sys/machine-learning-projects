import java.nio.file.Paths;
import java.util.List;

/**
 * Assertion-based tests -- no external test framework, so the project builds
 * and runs with nothing but a JDK.
 *
 * <pre>
 *   javac -d out src/*.java tests/*.java &amp;&amp; java -cp out NetworkTests
 * </pre>
 */
public final class NetworkTests {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        shortestPathPicksLowerCostRoute();
        metricChangesTheChosenRoute();
        graphIsUndirected();
        unreachableStationReportsNoRoute();
        removingSegmentForcesDetour();
        copyDoesNotMutateOriginal();
        csvLoadsExpectedNetwork();

        System.out.printf("%n%d passed, %d failed%n", passed, failed);
        if (failed > 0) {
            System.exit(1);
        }
    }

    // --- tests ------------------------------------------------------------

    /** A direct 100-mile link should beat a 2-hop 150-mile detour. */
    private static void shortestPathPicksLowerCostRoute() {
        Graph g = new Graph();
        g.addSegment("A", "B", 100, 2.0);
        g.addSegment("A", "C", 60, 1.0);
        g.addSegment("C", "B", 90, 1.0);

        Dijkstra.Result r = Dijkstra.shortestPath(g, "A", "B", Dijkstra.Metric.MILES);
        check("direct route chosen by miles", r.found && r.path.equals(List.of("A", "B")));
        check("miles total correct", r.totalMiles == 100.0);
    }

    /** Optimizing hours instead of miles should take the longer-but-faster path. */
    private static void metricChangesTheChosenRoute() {
        Graph g = new Graph();
        g.addSegment("A", "B", 100, 5.0);          // short but slow
        g.addSegment("A", "C", 60, 1.0);
        g.addSegment("C", "B", 90, 1.0);           // longer but faster overall

        Dijkstra.Result miles = Dijkstra.shortestPath(g, "A", "B", Dijkstra.Metric.MILES);
        Dijkstra.Result hours = Dijkstra.shortestPath(g, "A", "B", Dijkstra.Metric.HOURS);

        check("miles picks direct", miles.path.equals(List.of("A", "B")));
        check("hours picks via C", hours.path.equals(List.of("A", "C", "B")));
        check("hours total correct", hours.totalHours == 2.0);
    }

    private static void graphIsUndirected() {
        Graph g = new Graph();
        g.addSegment("X", "Y", 10, 1.0);
        check("edge traversable forward", Dijkstra
                .shortestPath(g, "X", "Y", Dijkstra.Metric.MILES).found);
        check("edge traversable backward", Dijkstra
                .shortestPath(g, "Y", "X", Dijkstra.Metric.MILES).found);
        check("segment counted once", g.segmentCount() == 1);
    }

    private static void unreachableStationReportsNoRoute() {
        Graph g = new Graph();
        g.addSegment("A", "B", 10, 1.0);
        g.addSegment("C", "D", 10, 1.0);          // separate component
        check("disconnected -> no route",
                !Dijkstra.shortestPath(g, "A", "D", Dijkstra.Metric.MILES).found);
        check("unknown station -> no route",
                !Dijkstra.shortestPath(g, "A", "Nowhere", Dijkstra.Metric.MILES).found);
    }

    private static void removingSegmentForcesDetour() {
        Graph g = new Graph();
        g.addSegment("A", "B", 100, 2.0);
        g.addSegment("A", "C", 60, 1.0);
        g.addSegment("C", "B", 90, 1.0);

        check("segment removed", g.removeSegment("A", "B"));
        Dijkstra.Result r = Dijkstra.shortestPath(g, "A", "B", Dijkstra.Metric.MILES);
        check("detour found", r.found && r.path.equals(List.of("A", "C", "B")));
        check("detour costs more", r.totalMiles == 150.0);
    }

    private static void copyDoesNotMutateOriginal() {
        Graph g = new Graph();
        g.addSegment("A", "B", 100, 2.0);
        g.addSegment("A", "C", 60, 1.0);
        g.addSegment("C", "B", 90, 1.0);

        Graph copy = g.copy();
        copy.removeSegment("A", "B");
        check("copy has segment removed", !Dijkstra
                .shortestPath(copy, "A", "B", Dijkstra.Metric.MILES).path.equals(List.of("A", "B")));
        check("original untouched", Dijkstra
                .shortestPath(g, "A", "B", Dijkstra.Metric.MILES).path.equals(List.of("A", "B")));
        check("copy preserves segment count", copy.segmentCount() == 2);
    }

    private static void csvLoadsExpectedNetwork() throws Exception {
        Graph g = RailNetwork.fromCsv(Paths.get("data", "network.csv"));
        check("stations loaded", g.stations().size() >= 20);
        check("segments loaded", g.segmentCount() >= 25);
        check("known station present", g.contains("Chicago"));

        Dijkstra.Result r = Dijkstra.shortestPath(g, "Seattle", "Chicago", Dijkstra.Metric.MILES);
        check("cross-network route found", r.found);
        check("route starts at origin", r.path.get(0).equals("Seattle"));
        check("route ends at destination", r.path.get(r.path.size() - 1).equals("Chicago"));
        check("route has positive cost", r.totalMiles > 0 && r.totalHours > 0);
    }

    // --- helper -----------------------------------------------------------

    private static void check(String name, boolean condition) {
        if (condition) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            System.out.println("  FAIL  " + name);
        }
    }
}
