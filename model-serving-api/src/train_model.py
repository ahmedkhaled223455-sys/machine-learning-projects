"""Train the logistic-regression risk model and save inference artifacts
for the FastAPI service.

Reports a leakage-safe holdout AUC (impute/scale fit on the training
split only), then refits the same model on the full dataset -- standard
practice: report the honest holdout number, ship the model trained on
every available row.

Run:  python src/train_model.py   (from the project root)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import (FEATURES, LogisticRegressionGD, MedianImputer,
                   Standardizer, mark_missing, roc_auc, stratified_split)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pima-indians-diabetes.csv"
MODEL_DIR = ROOT / "model"
SEED = 42


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["Outcome"].to_numpy(dtype=int)
    print(f"{len(df)} patients | positive rate {y.mean():.1%}")

    # ---- leakage-safe holdout evaluation -----------------------------
    tr_idx, te_idx = stratified_split(y, test_frac=0.2, seed=SEED)
    Xtr_raw, Xte_raw = mark_missing(X[tr_idx]), mark_missing(X[te_idx])
    imp = MedianImputer().fit(Xtr_raw)
    Xtr, Xte = imp.transform(Xtr_raw), imp.transform(Xte_raw)
    sc = Standardizer().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    holdout_model = LogisticRegressionGD(lr=0.1, epochs=3000, l2=1e-3)
    holdout_model.fit(Xtr, y[tr_idx])
    holdout_auc = roc_auc(y[te_idx], holdout_model.predict_proba(Xte))
    print(f"Holdout AUC (20% held out, never used for fitting): "
          f"{holdout_auc:.3f}")

    # ---- refit on the full dataset for the deployed artifact ----------
    X_full_raw = mark_missing(X)
    imp_full = MedianImputer().fit(X_full_raw)
    X_full = imp_full.transform(X_full_raw)
    sc_full = Standardizer().fit(X_full)
    X_full = sc_full.transform(X_full)

    final_model = LogisticRegressionGD(lr=0.1, epochs=3000, l2=1e-3)
    final_model.fit(X_full, y)

    np.savez(
        MODEL_DIR / "artifacts.npz",
        w=final_model.w, b=final_model.b,
        medians=imp_full.medians_, means=sc_full.mean_, stds=sc_full.std_,
        holdout_auc=holdout_auc,
    )
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump({"features": FEATURES, "holdout_auc": round(holdout_auc, 4),
                   "n_train_full": len(df)}, f, indent=2)
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
