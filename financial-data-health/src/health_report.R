# Data-health report in R -- the same robust return-outlier check as the
# Python monitor, to show the analysis reproduced in R (SQL/Pandas/R being
# the tabular stack this is aimed at).
#
# Base R only (no packages), so it runs on a stock R install. Reads the
# panel the Python generator wrote, computes per-ticker return summaries,
# flags outliers with a robust MAD z-score (base R mad() is already scaled
# by 1.4826, matching the Python detector), and saves a plot.
#
# Run from the project root, after `python src/generate_prices.py`:
#   Rscript src/health_report.R

prices <- read.csv("data/prices.csv", stringsAsFactors = FALSE)
dates <- as.Date(prices$date)
tickers <- setdiff(names(prices), "date")

K <- 8  # robust-sigma threshold, same as the Python monitor

cat("=== Per-ticker return summary ===\n")
cat(sprintf("%-6s %8s %8s %8s %8s\n", "ticker", "mean", "sd", "min", "max"))

all_flags <- data.frame(ticker = character(), date = character(),
                        z = numeric(), stringsAsFactors = FALSE)

for (tk in tickers) {
  p <- prices[[tk]]
  r <- c(NA, diff(p) / head(p, -1))          # simple returns
  rr <- r[is.finite(r)]
  cat(sprintf("%-6s %8.4f %8.4f %8.4f %8.4f\n",
              tk, mean(rr), sd(rr), min(rr), max(rr)))

  med <- median(rr)
  s <- mad(r, na.rm = TRUE)                   # base R mad() ~ 1.4826 * MAD
  if (s > 0) {
    z <- (r - med) / s
    idx <- which(abs(z) > K)
    if (length(idx) > 0) {
      all_flags <- rbind(all_flags,
                         data.frame(ticker = tk,
                                    date = as.character(dates[idx]),
                                    z = round(z[idx], 1),
                                    stringsAsFactors = FALSE))
    }
  }
}

cat(sprintf("\n=== Outlier flags (|robust z| > %d): %d ===\n", K, nrow(all_flags)))
if (nrow(all_flags) > 0) {
  print(all_flags[order(-abs(all_flags$z)), ], row.names = FALSE)
}

# Save a returns plot for the regime-break ticker.
if (!dir.exists("results")) dir.create("results")
tk <- "CDX"
p <- prices[[tk]]
r <- c(NA, diff(p) / head(p, -1))
png("results/r_returns_CDX.png", width = 900, height = 400)
plot(dates, r, type = "l", col = "#31708E",
     main = paste(tk, "daily returns (R)"), xlab = "date", ylab = "return")
dev.off()

cat("\nSaved results/r_returns_CDX.png\n")
