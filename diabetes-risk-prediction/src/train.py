"""End-to-end training pipeline for diabetes risk prediction.

Steps
-----
1. Load the Pima Indians Diabetes dataset and audit data quality
   (several columns use 0 to encode "missing", which is biologically
   impossible for glucose, blood pressure, skin thickness, insulin, BMI).
2. Stratified 80/20 train/test split.
3. 5-fold cross-validation on the training set comparing logistic
   regression and a random forest. All preprocessing (median imputation,
   standardization) is fit inside each fold to avoid leakage.
4. Refit both models on the full training set, evaluate once on the
   held-out test set, and save metrics + figures.

Run:  python src/train.py   (from the project root)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import (MedianImputer, Standardizer, auc,
                        classification_metrics, roc_curve,
                        stratified_kfold, stratified_split)
from models import LogisticRegressionGD, RandomForest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pima-indians-diabetes.csv"
RESULTS = ROOT / "results"

# Columns where a raw value of 0 actually means "not recorded".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SEED = 42


def load_and_audit():
    df = pd.read_csv(DATA)
    audit = {}
    for col in ZERO_AS_MISSING:
        n_zero = int((df[col] == 0).sum())
        audit[col] = {"zeros_as_missing": n_zero,
                      "pct": round(100 * n_zero / len(df), 1)}
        df[col] = df[col].replace(0, np.nan)
    return df, audit


def make_models():
    return {
        "logistic_regression": lambda: LogisticRegressionGD(
            lr=0.1, epochs=3000, l2=1e-3),
        "random_forest": lambda: RandomForest(
            n_trees=100, max_depth=6, min_samples_leaf=5, seed=SEED),
    }


def prep(X_train, X_apply):
    """Fit imputer+scaler on X_train only, apply to both."""
    imp = MedianImputer().fit(X_train)
    Xt, Xa = imp.transform(X_train), imp.transform(X_apply)
    sc = Standardizer().fit(Xt)
    return sc.transform(Xt), sc.transform(Xa)


def cross_validate(X, y, k=5):
    out = {}
    for name, factory in make_models().items():
        aucs, f1s = [], []
        for tr, va in stratified_kfold(y, k=k, seed=SEED):
            Xtr, Xva = prep(X[tr], X[va])
            model = factory().fit(Xtr, y[tr])
            proba = model.predict_proba(Xva)
            aucs.append(auc(y[va], proba))
            f1s.append(classification_metrics(y[va], (proba >= 0.5).astype(int))["f1"])
        out[name] = {"cv_auc_mean": float(np.mean(aucs)),
                     "cv_auc_std": float(np.std(aucs)),
                     "cv_f1_mean": float(np.mean(f1s))}
        print(f"[CV] {name:20s} AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}"
              f" | F1 {np.mean(f1s):.3f}")
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    df, audit = load_and_audit()
    print("Data quality audit (zeros treated as missing):")
    for col, info in audit.items():
        print(f"  {col:15s} {info['zeros_as_missing']:4d} ({info['pct']}%)")

    feature_names = [c for c in df.columns if c != "Outcome"]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["Outcome"].to_numpy(dtype=int)
    print(f"\n{len(df)} patients | positive rate {y.mean():.1%}")

    tr_idx, te_idx = stratified_split(y, test_frac=0.2, seed=SEED)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_test, y_test = X[te_idx], y[te_idx]
    print(f"train {len(tr_idx)} / test {len(te_idx)}")

    cv_results = cross_validate(X_train, y_train, k=5)

    # ---- final fit + held-out evaluation --------------------------------
    Xtr, Xte = prep(X_train, X_test)
    holdout, curves = {}, {}
    models = {}
    for name, factory in make_models().items():
        model = factory().fit(Xtr, y_train)
        models[name] = model
        proba = model.predict_proba(Xte)
        m = classification_metrics(y_test, (proba >= 0.5).astype(int))
        m["auc"] = auc(y_test, proba)
        holdout[name] = m
        curves[name] = roc_curve(y_test, proba)
        print(f"[holdout] {name:20s} AUC {m['auc']:.3f} acc {m['accuracy']:.3f}"
              f" precision {m['precision']:.3f} recall {m['recall']:.3f}"
              f" F1 {m['f1']:.3f}")

    # ---- figures ----------------------------------------------------------
    plt.figure(figsize=(5.5, 5))
    for name, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={holdout[name]['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Held-out ROC — diabetes risk prediction")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS / "roc_holdout.png", dpi=150)
    plt.close()

    lr = models["logistic_regression"]
    order = np.argsort(lr.w)
    plt.figure(figsize=(6.5, 4))
    plt.barh(np.array(feature_names)[order], lr.w[order], color="#31708E")
    plt.xlabel("Coefficient (standardized features)")
    plt.title("Logistic regression drivers of diabetes risk")
    plt.tight_layout()
    plt.savefig(RESULTS / "lr_coefficients.png", dpi=150)
    plt.close()

    # Threshold sweep for the better model (by holdout AUC).
    best = max(holdout, key=lambda k: holdout[k]["auc"])
    proba = models[best].predict_proba(Xte)
    ths = np.linspace(0.1, 0.9, 33)
    prec, rec = [], []
    for t in ths:
        m = classification_metrics(y_test, (proba >= t).astype(int))
        prec.append(m["precision"])
        rec.append(m["recall"])
    plt.figure(figsize=(6, 4))
    plt.plot(ths, prec, label="precision")
    plt.plot(ths, rec, label="recall")
    plt.axvline(0.5, color="gray", ls=":", lw=1)
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title(f"Precision/recall vs threshold — {best}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "threshold_tradeoff.png", dpi=150)
    plt.close()

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump({"data_audit": audit, "cv": cv_results,
                   "holdout": holdout, "best_model": best}, f, indent=2)
    print(f"\nSaved metrics and figures to {RESULTS}")


if __name__ == "__main__":
    main()
