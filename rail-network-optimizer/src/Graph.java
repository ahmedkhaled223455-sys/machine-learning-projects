import java.util.*;

/**
 * An undirected weighted graph of a rail network.
 *
 * <p>Nodes are stations/terminals; edges are track segments carrying two
 * weights that a planner actually trades off: distance (miles) and transit
 * time (hours). Segments can be removed at runtime to model a closure or
 * maintenance outage, which is what makes rerouting analysis possible.
 */
public class Graph {

    /** A track segment between two stations. */
    public static final class Edge {
        public final String from;
        public final String to;
        public final double miles;
        public final double hours;

        public Edge(String from, String to, double miles, double hours) {
            this.from = from;
            this.to = to;
            this.miles = miles;
            this.hours = hours;
        }

        /** Weight of this segment under the chosen planning metric. */
        public double weight(Dijkstra.Metric metric) {
            return metric == Dijkstra.Metric.MILES ? miles : hours;
        }

        @Override
        public String toString() {
            return from + " -> " + to + " (" + miles + " mi, " + hours + " h)";
        }
    }

    private final Map<String, List<Edge>> adjacency = new HashMap<>();

    /** Add a segment in both directions (track is traversable either way). */
    public void addSegment(String a, String b, double miles, double hours) {
        adjacency.computeIfAbsent(a, k -> new ArrayList<>()).add(new Edge(a, b, miles, hours));
        adjacency.computeIfAbsent(b, k -> new ArrayList<>()).add(new Edge(b, a, miles, hours));
    }

    /**
     * Remove a segment in both directions -- models a closure or outage.
     *
     * @return true if a segment was actually removed
     */
    public boolean removeSegment(String a, String b) {
        boolean removed = false;
        if (adjacency.containsKey(a)) {
            removed |= adjacency.get(a).removeIf(e -> e.to.equals(b));
        }
        if (adjacency.containsKey(b)) {
            removed |= adjacency.get(b).removeIf(e -> e.to.equals(a));
        }
        return removed;
    }

    /** Outgoing segments from a station (empty list if unknown). */
    public List<Edge> neighbors(String station) {
        return adjacency.getOrDefault(station, Collections.emptyList());
    }

    public Set<String> stations() {
        return Collections.unmodifiableSet(adjacency.keySet());
    }

    public boolean contains(String station) {
        return adjacency.containsKey(station);
    }

    /** Number of undirected segments (each stored twice internally). */
    public int segmentCount() {
        int directed = adjacency.values().stream().mapToInt(List::size).sum();
        return directed / 2;
    }

    /** A deep copy, so disruption scenarios don't mutate the base network. */
    public Graph copy() {
        Graph g = new Graph();
        for (List<Edge> edges : adjacency.values()) {
            for (Edge e : edges) {
                // addSegment inserts both directions; only add each pair once
                if (e.from.compareTo(e.to) < 0) {
                    g.addSegment(e.from, e.to, e.miles, e.hours);
                }
            }
        }
        return g;
    }
}
