import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Loads a rail network from CSV.
 *
 * <p>Expected header: {@code from,to,miles,hours}. Malformed or short rows are
 * skipped rather than aborting the load -- a real feed will have imperfect
 * rows, and a routing tool should still come up.
 */
public final class RailNetwork {

    private RailNetwork() {
    }

    public static Graph fromCsv(Path csv) throws IOException {
        Graph graph = new Graph();
        List<String> lines = Files.readAllLines(csv);
        int skipped = 0;

        for (int i = 1; i < lines.size(); i++) {          // skip header
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] parts = line.split(",");
            if (parts.length < 4) {
                skipped++;
                continue;
            }
            try {
                String from = parts[0].trim();
                String to = parts[1].trim();
                double miles = Double.parseDouble(parts[2].trim());
                double hours = Double.parseDouble(parts[3].trim());
                if (from.isEmpty() || to.isEmpty() || miles <= 0 || hours <= 0) {
                    skipped++;
                    continue;
                }
                graph.addSegment(from, to, miles, hours);
            } catch (NumberFormatException e) {
                skipped++;
            }
        }
        if (skipped > 0) {
            System.err.println("warning: skipped " + skipped + " malformed row(s)");
        }
        return graph;
    }
}
