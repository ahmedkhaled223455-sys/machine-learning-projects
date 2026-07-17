"""Customer segmentation on the Mall Customers dataset.

Cluster 200 customers on Age, AnnualIncome and SpendingScore
(standardized so no feature dominates the Euclidean metric), select k
with both the elbow curve and silhouette score, then profile each
segment so the clusters can be acted on by a marketing team.

Run:  python src/segment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kmeans import KMeans, pca_project, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mall_customers.csv"
RESULTS = ROOT / "results"
FEATURES = ["Age", "AnnualIncome_k", "SpendingScore"]
SEED = 0


def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    print(f"{len(df)} customers | features: {FEATURES}")

    X = df[FEATURES].to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd

    # ---- model selection: elbow + silhouette -----------------------------
    ks = range(2, 11)
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(k=k, n_init=10, seed=SEED).fit(Z)
        s = silhouette_score(Z, km.labels_)
        inertias.append(km.inertia_)
        silhouettes.append(s)
        print(f"k={k}: inertia {km.inertia_:8.1f} silhouette {s:.3f}")
    best_k = list(ks)[int(np.argmax(silhouettes))]
    print(f"selected k={best_k} (max silhouette)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(ks), inertias, "o-")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("inertia (WCSS)")
    axes[0].set_title("Elbow curve")
    axes[1].plot(list(ks), silhouettes, "o-", color="#B1624E")
    axes[1].axvline(best_k, color="gray", ls=":")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("mean silhouette")
    axes[1].set_title("Silhouette analysis")
    plt.tight_layout()
    plt.savefig(RESULTS / "model_selection.png", dpi=150)
    plt.close()

    # ---- final clustering -------------------------------------------------
    km = KMeans(k=best_k, n_init=20, seed=SEED).fit(Z)
    df["segment"] = km.labels_

    # Profile table in original units.
    profile = df.groupby("segment").agg(
        size=("segment", "size"),
        mean_age=("Age", "mean"),
        mean_income_k=("AnnualIncome_k", "mean"),
        mean_spending=("SpendingScore", "mean"),
        pct_female=("Gender", lambda g: 100 * (g == "Female").mean()),
    ).round(1)
    profile.to_csv(RESULTS / "segment_profiles.csv")
    print("\nSegment profiles:")
    print(profile.to_string())

    # ---- visualizations ---------------------------------------------------
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(7, 5.5))
    for c in range(best_k):
        m = df["segment"] == c
        plt.scatter(df.loc[m, "AnnualIncome_k"], df.loc[m, "SpendingScore"],
                    s=26, color=cmap(c), label=f"segment {c} (n={m.sum()})")
    centers_orig = km.centers_ * sd + mu
    plt.scatter(centers_orig[:, 1], centers_orig[:, 2], marker="X", s=180,
                c="black", label="centroids")
    plt.xlabel("annual income [k$]")
    plt.ylabel("spending score (1-100)")
    plt.title(f"Customer segments (k={best_k})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / "segments_income_spending.png", dpi=150)
    plt.close()

    proj, var = pca_project(Z, 2)
    plt.figure(figsize=(6.5, 5))
    for c in range(best_k):
        m = df["segment"] == c
        plt.scatter(proj[m, 0], proj[m, 1], s=26, color=cmap(c),
                    label=f"segment {c}")
    plt.xlabel(f"PC1 ({var[0]:.0%} var)")
    plt.ylabel(f"PC2 ({var[1]:.0%} var)")
    plt.title("Segments in PCA space (all three features)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / "segments_pca.png", dpi=150)
    plt.close()

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump({
            "k_selected": int(best_k),
            "silhouette_by_k": {str(k): round(s, 4)
                                for k, s in zip(ks, silhouettes)},
            "inertia_by_k": {str(k): round(i, 1)
                             for k, i in zip(ks, inertias)},
            "final_silhouette": round(silhouette_score(Z, km.labels_), 4),
        }, f, indent=2)
    print(f"\nSaved profiles, metrics and figures to {RESULTS}")


if __name__ == "__main__":
    main()
