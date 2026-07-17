"""Predict math scores: how much signal is in demographics alone?

Two feature sets are compared deliberately:

- A. Demographics + prep only — the honest "what can we know upfront"
  model. Expect modest performance; that finding matters.
- B. Adding reading/writing scores — near-collinear outcomes measured
  at the same time. Performance jumps, but using them as "predictors"
  answers a different (and mostly useless) product question. Included
  to make the leakage discussion concrete.

Models: mean baseline, ridge regression, gradient-boosted trees — all
implemented from scratch (see boosting.py). 5-fold cross-validation,
permutation feature importance for the GBM.

Run:  python src/build_database.py && python src/predict_scores.py
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

from boosting import (GradientBoostingRegressor, MeanBaseline,
                      RidgeRegression, kfold_indices, regression_metrics)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "students.db"
RESULTS = ROOT / "results"
SEED = 42
TARGET = "math_score"


def make_features(df: pd.DataFrame, include_other_scores: bool):
    cat = pd.get_dummies(
        df[["gender", "race_ethnicity", "parental_education", "lunch",
            "test_prep"]], drop_first=True).astype(float)
    names = list(cat.columns)
    X = cat.to_numpy()
    if include_other_scores:
        X = np.column_stack([X, df[["reading_score", "writing_score"]]])
        names += ["reading_score", "writing_score"]
    return X, names


def cv_compare(X, y, k=5):
    models = {
        "baseline_mean": lambda: MeanBaseline(),
        "ridge": lambda: RidgeRegression(alpha=1.0),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=10),
    }
    out = {}
    for name, factory in models.items():
        folds = []
        for tr, va in kfold_indices(len(y), k=k, seed=SEED):
            model = factory().fit(X[tr], y[tr])
            folds.append(regression_metrics(y[va], model.predict(X[va])))
        out[name] = {m: round(float(np.mean([f[m] for f in folds])), 3)
                     for m in folds[0]}
    return out


def permutation_importance(model, X, y, names, n_repeats=10, seed=SEED):
    rng = np.random.default_rng(seed)
    base = regression_metrics(y, model.predict(X))["rmse"]
    imp = []
    for j in range(X.shape[1]):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(regression_metrics(y, model.predict(Xp))["rmse"] - base)
        imp.append(float(np.mean(deltas)))
    return sorted(zip(names, imp), key=lambda t: -t[1])


def main():
    RESULTS.mkdir(exist_ok=True)
    if not DB.exists():
        raise SystemExit("Run src/build_database.py first.")
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM students", con)
    con.close()
    y = df[TARGET].to_numpy(float)

    report = {}
    preds_for_plot = {}
    for variant, include in [("demographics_only", False),
                             ("with_other_scores", True)]:
        X, names = make_features(df, include)
        report[variant] = cv_compare(X, y)
        print(f"\n[{variant}] ({X.shape[1]} features)")
        for m, s in report[variant].items():
            print(f"  {m:18s} RMSE {s['rmse']:6.2f}  MAE {s['mae']:6.2f}  "
                  f"R2 {s['r2']:6.3f}")

        # one 80/20 split for plots / importance
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(y))
        te, tr = perm[:200], perm[200:]
        gbm = GradientBoostingRegressor(200, 0.05, 3, 10).fit(X[tr], y[tr])
        preds_for_plot[variant] = (y[te], gbm.predict(X[te]))
        if not include:
            importance = permutation_importance(gbm, X[te], y[te], names)

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (variant, (yt, yp)) in zip(axes, preds_for_plot.items()):
        r2 = regression_metrics(yt, yp)["r2"]
        ax.scatter(yt, yp, s=16, alpha=0.6)
        lims = [0, 105]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel("actual math score")
        ax.set_ylabel("predicted")
        ax.set_title(f"{variant} (holdout R2={r2:.2f})")
    plt.tight_layout()
    plt.savefig(RESULTS / "prediction_scatter.png", dpi=150)
    plt.close()

    top = importance[:10]
    plt.figure(figsize=(7, 4.5))
    plt.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1],
             color="#31708E")
    plt.xlabel("RMSE increase when feature is shuffled")
    plt.title("Permutation importance — GBM, demographics only")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=150)
    plt.close()

    with open(RESULTS / "prediction_metrics.json", "w") as f:
        json.dump({"cv": report,
                   "top_importance": [{"feature": n, "rmse_increase": round(v, 3)}
                                      for n, v in importance[:8]]}, f, indent=2)
    print(f"\nSaved figures and prediction_metrics.json to {RESULTS}")


if __name__ == "__main__":
    main()
