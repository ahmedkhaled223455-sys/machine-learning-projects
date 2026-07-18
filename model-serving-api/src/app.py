"""FastAPI service serving the from-scratch logistic-regression diabetes
risk model.

Run (from the project root, after `python src/train_model.py` has
produced model/artifacts.npz):

    uvicorn src.app:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from .model import mark_missing
from .schemas import PatientFeatures, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "model" / "artifacts.npz"
THRESHOLD = 0.5

app = FastAPI(
    title="Diabetes Risk Prediction API",
    description=(
        "Serves a from-scratch logistic regression (no scikit-learn) "
        "trained on the Pima Indians Diabetes dataset. Full training and "
        "evaluation writeup lives in ../diabetes-risk-prediction."
    ),
    version="1.0.0",
)

_state = {"w": None, "b": None, "medians": None, "means": None, "stds": None,
          "holdout_auc": None}


def _load_artifacts() -> bool:
    if not ARTIFACTS.exists():
        return False
    data = np.load(ARTIFACTS)
    _state.update({
        "w": data["w"], "b": float(data["b"]),
        "medians": data["medians"], "means": data["means"], "stds": data["stds"],
        "holdout_auc": float(data["holdout_auc"]),
    })
    return True


# Load at import time so both `uvicorn src.app:app` and the test client
# (which imports this module directly) see a ready model without relying
# on framework-specific startup-event wiring.
_load_artifacts()


@app.get("/")
def root():
    return {"service": "diabetes-risk-prediction-api", "docs": "/docs",
            "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok" if _state["w"] is not None else "model_not_loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    if _state["w"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded -- run `python src/train_model.py` first.",
        )

    x = np.array([[
        patient.pregnancies, patient.glucose, patient.blood_pressure,
        patient.skin_thickness, patient.insulin, patient.bmi,
        patient.diabetes_pedigree_function, patient.age,
    ]], dtype=float)

    # Same preprocessing as training: a submitted 0 for these columns
    # means "not recorded", not "measured as zero" -- median-impute it
    # rather than feeding a biologically impossible value to the model.
    x = mark_missing(x)
    for j in range(x.shape[1]):
        if np.isnan(x[0, j]):
            x[0, j] = _state["medians"][j]
    x = (x - _state["means"]) / _state["stds"]

    z = x @ _state["w"] + _state["b"]
    proba = float(1.0 / (1.0 + np.exp(-np.clip(z[0], -30, 30))))

    return PredictionResponse(
        risk_probability=round(proba, 4),
        risk_label="high" if proba >= THRESHOLD else "low",
        threshold=THRESHOLD,
        model_holdout_auc=round(_state["holdout_auc"], 4),
    )
