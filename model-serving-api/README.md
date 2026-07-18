# Model Serving API

A FastAPI REST service that puts a trained model behind a real HTTP
endpoint: request validation, structured responses, health checks,
automated API tests, and a Dockerfile — the deployment half of the
modeling story in [`../diabetes-risk-prediction`](../diabetes-risk-prediction/).

This portfolio's stated principle is "no ML libraries, everything from
scratch" — that's about *modeling*. FastAPI is a serving/HTTP framework,
not a modeling shortcut: the model underneath is the same from-scratch
NumPy logistic regression, just wrapped for real traffic instead of a
one-off script.

## Why this exists as its own project

`diabetes-risk-prediction` answers "does the model work?" (cross-validation,
holdout AUC, ROC curves). This project answers the next question a data
engineering role actually asks: "how does someone else's code call it?" —
request schemas, input validation, error responses, and tests that would
catch a regression before it reached a client.

## The model

Logistic regression (gradient descent, L2), trained on the Pima Indians
Diabetes dataset (768 patients). Same data-quality handling as the
research project: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
and `BMI` use `0` to encode "not recorded" (biologically impossible as a
true zero), so those are converted to `NaN` and median-imputed — with the
median learned on training data only.

```
768 patients | positive rate 34.9%
Holdout AUC (20% held out, never used for fitting): 0.836
```

`src/train_model.py` reports that holdout number, then refits the same
model on the full 768 rows for the artifact the API actually serves —
report the honest number, ship the model trained on everything available.

**Train/serve consistency matters here**: the API applies the exact same
zero-as-missing imputation to incoming requests as training used. A client
that sends `glucose: 0` is saying "I don't have this value," and the
model treats it that way — not as a literal, impossible zero. Skipping
this at inference time is a classic way real systems quietly degrade.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | `{"status": "ok"}` once the model is loaded |
| `/predict` | POST | Patient features → risk probability |
| `/docs` | GET | Interactive Swagger UI (auto-generated) |

Request/response schemas are Pydantic models (`src/schemas.py`) with
range validation on every field (e.g. `age` must be 1–120) — malformed
input gets a `422` with a field-level error, not a silent bad prediction.

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pregnancies":2,"glucose":130,"blood_pressure":70,
       "skin_thickness":25,"insulin":90,"bmi":32.5,
       "diabetes_pedigree_function":0.45,"age":35}'
```

```json
{
  "risk_probability": 0.331,
  "risk_label": "low",
  "threshold": 0.5,
  "model_holdout_auc": 0.836
}
```

(Verified by hand against the trained artifact — a clearly low-risk
profile scores 0.018 and a clearly high-risk one scores 0.987, so the
model's ranking behaves sensibly, not just its plumbing.)

## Run

```bash
pip install -r requirements.txt
python src/train_model.py            # writes model/artifacts.npz (~2 s)
uvicorn src.app:app --reload --port 8000
# then open http://localhost:8000/docs
```

## Test

```bash
python -m pytest tests/ -v
```

Covers: valid prediction shape, `422` on a missing field, `422` on an
out-of-range value, the zero-as-missing imputation path, and a
monotonicity check (a clearly high-risk profile must score above a
clearly low-risk one) — testing the model's behavior through the API,
not just that the endpoint returns *some* 200.

## Docker

```bash
docker build -t diabetes-risk-api .
docker run -p 8000:8000 diabetes-risk-api
```

The image trains the model at build time, so the container is ready to
serve as soon as it starts — no separate training step at deploy time.

## Limitations / what a real production version would add

- No auth, no rate limiting — fine for a portfolio demo, not for a
  public endpoint.
- Single global model, no versioning or A/B rollout; a model registry
  (MLflow, as used in [`../data-quality-verification`](../data-quality-verification/))
  would track which artifact is actually serving.
- No request/response logging or drift monitoring — the natural next
  layer once this is behind real traffic.
- Synchronous single-record inference only; a batch `/predict_batch`
  endpoint would matter at higher volume.
