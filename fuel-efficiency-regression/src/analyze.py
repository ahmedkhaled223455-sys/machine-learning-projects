"""Fuel-efficiency (MPG) prediction on the classic Auto MPG dataset.

Pipeline
--------
1. Load data; audit and impute the 6 missing horsepower values
   (median, learned on each training fold only).
2. Feature engineering: one-hot encode origin, add power-to-weight
   ratio, standardize numeric features.
3. 10-fold cross-validation comparing: mean-prediction baseline,
   OLS, and ridge regression across a regularization sweep.
4. Diagnostics on a held-out split: predicted-vs-actual, residuals vs
   fitted (checks the linearity assumption), coefficient magnitudes.

Run:  python src/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regression import (LinearRegression, RidgeRegression, Standardizer,
                        kfold_indices, regression_metrics)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "auto-mpg.csv"
RESULTS = ROOT / "results"
SEED = 42

NUMERIC = ["cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "power_to_weight"]


def load_features():
    df = pd.read_csv(DATA)
    n_missing = int(df["horsepower"].isna().sum())
    dummies = pd.get_dummies(df["origin"], prefix="origin", drop_first=True)
    return df, dummies, n_missing


def build_matrix(df, dummies, hp_median):
    d = df.copy()
    d["horsepower"] = d["horsepower"].fillna(hp_median)
    d["power_to_weight"] = d["horsepower"] / d["weight"]
    X_num = d[NUMERIC].to_numpy(float)
    X_cat = dummies.to_numpy(float)
    return X_num, X_cat


def cv_evaluate(df, dummies, model_factory, k=10):
    """Cross-validate with fold-local imputation and standardization."""
    y = df["mpg"].to_numpy(float)
    metrics = []
    for tr, va in kfold_indices(len(df), k=k, seed=SEED):
        hp_median = df.iloc[tr]["horsepower"].median()
        Xn, Xc = build_matrix(df, dummies, hp_median)
        sc = Standardizer().fit(Xn[tr])
        X = np.column_stack([sc.transform(Xn), Xc])
        model = model_factory().fit(X[tr], y[tr])
        metrics.append(regression_metrics(y[va], model.predict(X[va])))
    return {m: float(np.mean([f[m] for f in metrics])) for m in metrics[0]}


class MeanBaseline:
    def fit(self, X, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


def main():
    RESULTS.mkdir(exist_ok=True)
    df, dummies, n_missing = load_features()
    print(f"{len(df)} cars | horsepower missing: {n_missing}")

    results = {"baseline_mean": cv_evaluate(df, dummies, MeanBaseline),
               "ols": cv_evaluate(df, dummies, LinearRegression)}
    print(f"[CV] baseline  RMSE {results['baseline_mean']['rmse']:.2f}")
    print(f"[CV] OLS       RMSE {results['ols']['rmse']:.2f} "
          f"R2 {results['ols']['r2']:.3f}")

    # Ridge sweep.
    alphas = np.geomspace(1e-3, 1e3, 13)
    sweep = []
    for a in alphas:
        m = cv_evaluate(df, dummies, lambda a=a: RidgeRegression(alpha=a))
        sweep.append(m["rmse"])
    best_alpha = float(alphas[int(np.argmin(sweep))])
    results["ridge"] = cv_evaluate(
        df, dummies, lambda: RidgeRegression(alpha=best_alpha))
    results["ridge"]["alpha"] = best_alpha
    print(f"[CV] ridge(a={best_alpha:g}) RMSE {results['ridge']['rmse']:.2f} "
          f"R2 {results['ridge']['r2']:.3f}")

    plt.figure(figsize=(6, 4))
    plt.semilogx(alphas, sweep, "o-")
    plt.axvline(best_alpha, color="gray", ls=":")
    plt.xlabel("ridge alpha")
    plt.ylabel("10-fold CV RMSE [mpg]")
    plt.title("Regularization sweep")
    plt.tight_layout()
    plt.savefig(RESULTS / "ridge_sweep.png", dpi=150)
    plt.close()

    # ---- held-out diagnostics (single 80/20 split) -----------------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(df))
    n_test = int(0.2 * len(df))
    te, tr = perm[:n_test], perm[n_test:]

    hp_median = df.iloc[tr]["horsepower"].median()
    Xn, Xc = build_matrix(df, dummies, hp_median)
    sc = Standardizer().fit(Xn[tr])
    X = np.column_stack([sc.transform(Xn), Xc])
    y = df["mpg"].to_numpy(float)

    model = RidgeRegression(alpha=best_alpha).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    holdout = regression_metrics(y[te], pred)
    print(f"[holdout] ridge RMSE {holdout['rmse']:.2f} "
          f"MAE {holdout['mae']:.2f} R2 {holdout['r2']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y[te], pred, s=18, alpha=0.7)
    lims = [y[te].min() - 2, y[te].max() + 2]
    axes[0].plot(lims, lims, "k--", lw=0.8)
    axes[0].set_xlabel("actual mpg")
    axes[0].set_ylabel("predicted mpg")
    axes[0].set_title(f"Held-out predictions (R2={holdout['r2']:.3f})")
    resid = y[te] - pred
    axes[1].scatter(pred, resid, s=18, alpha=0.7)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("fitted mpg")
    axes[1].set_ylabel("residual")
    axes[1].set_title("Residuals vs fitted")
    plt.tight_layout()
    plt.savefig(RESULTS / "holdout_diagnostics.png", dpi=150)
    plt.close()

    feat_names = NUMERIC + list(dummies.columns)
    coefs = model.coef_[1:]                     # drop intercept
    order = np.argsort(np.abs(coefs))
    plt.figure(figsize=(7, 4.5))
    plt.barh(np.array(feat_names)[order], coefs[order], color="#B1624E")
    plt.xlabel("ridge coefficient (numeric features standardized)")
    plt.title("What drives fuel efficiency?")
    plt.tight_layout()
    plt.savefig(RESULTS / "coefficients.png", dpi=150)
    plt.close()

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump({"cv": results, "holdout_ridge": holdout,
                   "missing_horsepower": n_missing}, f, indent=2)
    print(f"Saved metrics and figures to {RESULTS}")


if __name__ == "__main__":
    main()
