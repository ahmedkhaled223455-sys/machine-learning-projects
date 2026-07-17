"""Does completing the test-preparation course move exam scores?

An observational-efficacy analysis, the kind product teams run before
(or instead of) a randomized experiment:

1. Naive comparison of completers vs non-completers per subject,
   with permutation tests (from scratch) and Cohen's d effect sizes.
2. Selection check: is prep completion correlated with socioeconomic
   proxies (lunch status, parental education)? If it is, the naive
   comparison is confounded.
3. Regression adjustment: OLS with demographic controls, prep
   coefficient bootstrapped for a 95% CI.
4. Honest causal framing: this is observational data; the analysis
   quantifies association and confounding risk, and states what a
   randomized rollout would add.

Run:  python src/build_database.py && python src/efficacy_analysis.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "students.db"
RESULTS = ROOT / "results"
SUBJECTS = ["math_score", "reading_score", "writing_score"]
SEED = 42


# ---------------------------------------------------------------------------
# Statistics from scratch
# ---------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10000,
                     seed: int = SEED) -> float:
    """Two-sided p-value for difference in means under random relabeling."""
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:na].mean() - perm[na:].mean()) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)          # add-one: p is never exactly 0


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return np.linalg.pinv(Xb) @ y


def bootstrap_coef(X: np.ndarray, y: np.ndarray, coef_idx: int,
                   n_boot: int = 1000, seed: int = SEED):
    """95% percentile CI for one OLS coefficient (index into X columns)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    coefs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        coefs[i] = ols_fit(X[idx], y[idx])[coef_idx + 1]   # +1 skips intercept
    return float(np.percentile(coefs, 2.5)), float(np.percentile(coefs, 97.5))


# ---------------------------------------------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    if not DB.exists():
        raise SystemExit("Run src/build_database.py first.")
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM students", con)
    con.close()

    prep = (df["test_prep"] == "completed").to_numpy()
    print(f"{len(df)} students | prep completed: {prep.sum()} "
          f"({prep.mean():.1%})\n")

    # ---- 1. naive comparison + permutation tests -------------------------
    naive = {}
    for subj in SUBJECTS:
        a = df.loc[prep, subj].to_numpy(float)       # completers
        b = df.loc[~prep, subj].to_numpy(float)      # non-completers
        diff = a.mean() - b.mean()
        p = permutation_test(a, b)
        d = cohens_d(a, b)
        naive[subj] = {"diff": round(float(diff), 2),
                       "cohens_d": round(d, 3),
                       "perm_p": round(p, 5)}
        print(f"{subj:14s} completers {a.mean():5.1f} vs {b.mean():5.1f} | "
              f"diff {diff:+.1f} | d={d:.2f} | perm p={p:.4f}")

    # ---- 2. selection check ----------------------------------------------
    print("\nPrep completion rate by group (selection check):")
    balance = {}
    for col in ["lunch", "parental_education"]:
        rates = df.groupby(col)["test_prep"].apply(
            lambda s: (s == "completed").mean())
        balance[col] = {k: round(float(v), 3) for k, v in rates.items()}
        spread = rates.max() - rates.min()
        print(f"  {col}: {balance[col]}  (spread {spread:.1%})")

    # ---- 3. regression adjustment ----------------------------------------
    dummies = pd.get_dummies(
        df[["gender", "race_ethnicity", "parental_education", "lunch"]],
        drop_first=True).astype(float)
    X = np.column_stack([prep.astype(float), dummies.to_numpy()])

    adjusted = {}
    for subj in SUBJECTS:
        y = df[subj].to_numpy(float)
        coef = ols_fit(X, y)[1]                       # prep coefficient
        lo, hi = bootstrap_coef(X, y, coef_idx=0)
        adjusted[subj] = {"adj_effect": round(float(coef), 2),
                          "ci95": [round(lo, 2), round(hi, 2)]}
        print(f"adjusted prep effect on {subj}: {coef:+.1f} "
              f"[95% CI {lo:+.1f}, {hi:+.1f}]")

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    bins = np.arange(0, 105, 5)
    for ax, subj in zip(axes, SUBJECTS):
        ax.hist(df.loc[~prep, subj], bins=bins, alpha=0.55, density=True,
                label="no prep", color="#8896AB")
        ax.hist(df.loc[prep, subj], bins=bins, alpha=0.55, density=True,
                label="prep completed", color="#B1624E")
        ax.set_title(subj.replace("_", " "))
        ax.set_xlabel("score")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS / "score_distributions.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(SUBJECTS))
    naive_effects = [naive[s]["diff"] for s in SUBJECTS]
    adj_effects = [adjusted[s]["adj_effect"] for s in SUBJECTS]
    err = np.array([[adjusted[s]["adj_effect"] - adjusted[s]["ci95"][0]
                     for s in SUBJECTS],
                    [adjusted[s]["ci95"][1] - adjusted[s]["adj_effect"]
                     for s in SUBJECTS]])
    ax.bar(xs - 0.18, naive_effects, width=0.36, label="naive difference",
           color="#8896AB")
    ax.bar(xs + 0.18, adj_effects, width=0.36, yerr=err, capsize=4,
           label="adjusted (OLS + bootstrap 95% CI)", color="#B1624E")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("_score", "") for s in SUBJECTS])
    ax.set_ylabel("test-prep effect [points]")
    ax.set_title("Estimated effect of completing test preparation")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS / "prep_effect_estimates.png", dpi=150)
    plt.close()

    with open(RESULTS / "efficacy_metrics.json", "w") as f:
        json.dump({"naive": naive, "balance_check": balance,
                   "adjusted": adjusted,
                   "n_students": len(df),
                   "prep_completion_rate": round(float(prep.mean()), 3)},
                  f, indent=2)
    print(f"\nSaved figures and efficacy_metrics.json to {RESULTS}")
    print("\nCaveat: observational data. The balance check bounds obvious "
          "selection on measured demographics; unmeasured confounders "
          "(e.g. motivation) remain. A randomized rollout would settle it.")


if __name__ == "__main__":
    main()
