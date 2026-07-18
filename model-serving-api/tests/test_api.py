"""API tests using FastAPI's TestClient (starlette test client over httpx).

Assumes `python src/train_model.py` has already produced
model/artifacts.npz -- the CI workflow runs that first.

Run:  python -m pytest tests/ -v   (from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for `import src`

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

VALID_PATIENT = {
    "pregnancies": 2, "glucose": 130, "blood_pressure": 70,
    "skin_thickness": 25, "insulin": 90, "bmi": 32.5,
    "diabetes_pedigree_function": 0.45, "age": 35,
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "diabetes-risk-prediction-api"


def test_health_reports_model_loaded():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_patient():
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_label"] in ("high", "low")
    expected_label = "high" if body["risk_probability"] >= body["threshold"] else "low"
    assert body["risk_label"] == expected_label


def test_predict_missing_field_returns_422():
    bad = dict(VALID_PATIENT)
    del bad["glucose"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_out_of_range_returns_422():
    bad = dict(VALID_PATIENT, age=-5)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_zero_glucose_is_treated_as_missing_not_literal_zero():
    # 0 encodes "not recorded" in this dataset (see README) -- the API
    # must impute it the same way training did, not score a literal zero.
    zero_glucose = dict(VALID_PATIENT, glucose=0)
    r = client.post("/predict", json=zero_glucose)
    assert r.status_code == 200
    assert 0.0 <= r.json()["risk_probability"] <= 1.0


def test_high_risk_profile_scores_above_low_risk_profile():
    low_risk = dict(VALID_PATIENT, glucose=85, bmi=22.0, age=25,
                    diabetes_pedigree_function=0.15, pregnancies=0)
    high_risk = dict(VALID_PATIENT, glucose=190, bmi=42.0, age=55,
                     diabetes_pedigree_function=1.8, pregnancies=6)
    p_low = client.post("/predict", json=low_risk).json()["risk_probability"]
    p_high = client.post("/predict", json=high_risk).json()["risk_probability"]
    assert p_high > p_low
