# Plant sensor forecasting — probabilistic, physics-informed, fault-robust

Multi-horizon **probabilistic forecasting** of a process unit's level from
noisy, faulty plant sensors, built the way a production model has to be:
**PyTorch** quantile networks, a **physics-hybrid** that respects the mass
balance, a **deep ensemble** for epistemic uncertainty, post-hoc
**calibration**, a robust **sensor-fault input layer**, honest
**back-testing against classical baselines**, and **CI/CD with a
champion/challenger promotion gate, rollback, and a drift-triggered
retraining monitor**.

## The plant

A gravity-drained surge tank, one step per minute:

```
A dL/dt = F_in + d(t) - Cv * u * sqrt(L)
```

| quantity | what the model gets |
|---|---|
| level `L` | noisy sensor (σ = 2 cm) — and in fault scenarios, a broken one |
| inflow `F_in` | noisy sensor history + the known plan |
| valve `u` | known commands (current + plan) |
| valve coefficient `Cv` | only a **nominal** value; the true plant is 10% different |
| disturbance `d(t)` | **never observed** (AR(1) unmeasured inflow) |

The target is the **true** level 1–16 minutes ahead, never the noisy sensor.
Training data covers target levels 1–3 m; the **regime-shift** test runs the
plant at levels 93% of which lie above anything seen in training.

## Models

| model | what it is |
|---|---|
| persistence, Holt | classical time-series baselines |
| ARX | direct multi-horizon ridge regression on recent sensors + known future controls |
| physics | first-principles mass balance with nominal `Cv`, no disturbance |
| `pure_net` / `pure_ensemble` | PyTorch quantile MLP predicting change from the last level (all dynamics learned) |
| `hybrid_net` / `hybrid_ensemble` | same network predicting a **correction to the physics rollout** |

Shared design of the PyTorch models (`src/quantile_models.py`):

* **Five quantiles per horizon step**, made non-crossing by construction
  (median + cumulative softplus gaps).
* **Loss** = pinball loss + a soft penalty on physically infeasible levels.
* **Deep ensemble** (5 seeds); `combine_ensemble` adds member disagreement
  (epistemic) to each member's own spread (aleatoric).
* Early stopping on validation pinball; every model, including the
  baselines, is **recalibrated on held-out validation data** (per-horizon,
  per-quantile shift) so the comparison is fair.

## Scenarios

* `in_dist` — same regime as training
* `regime_shift` — levels outside the training range
* `faults_naive` — level sensor with 3% dropouts, 160 spikes, 20 stuck runs; inputs only forward-filled
* `faults_cleaned` — same faults through the robust input layer (rolling-median spike removal, stuck-run detection, interpolation)
* `drift` — a slowly drifting sensor, which the input layer deliberately **cannot** repair from one sensor; the **calibration monitor** has to catch it

## Results

Classical baselines (from `python src/backtest.py` without torch; MAE and
pinball in metres, 90% interval should cover ~0.90):

| model | scenario | MAE | pinball | 90% coverage |
|---|---|---|---|---|
| ARX | in_dist | 0.074 | 0.023 | 0.907 |
| ARX | **regime_shift** | 0.212 | 0.078 | **0.428** |
| physics | regime_shift | 0.150 | 0.048 | 0.768 |
| ARX | faults_naive | 0.116 | 0.042 | 0.819 |
| ARX | faults_cleaned | 0.087 | 0.029 | 0.864 |

What the baselines already show:

* **The best in-distribution model is the worst out of it.** ARX is the
  strongest baseline in regime, then its 90% interval covers only 43% of
  outcomes once the plant leaves the training range — confidently wrong.
* **The robust input layer cuts the fault damage by about a third**
  (ARX pinball 0.042 → 0.029 on the same faulty sensor).
* **Drift is caught by monitoring, not by cleaning.** On clean data the
  rolling-coverage monitor never fires; with a drifting level sensor it
  fires at forecast 1,600 as rolling 90% coverage collapses to 10%.

The PyTorch pure/hybrid single models and ensembles are trained and scored
in CI (job *Train + backtest + promotion gate*); the full table, per-horizon
errors, monitor log, promotion decision, and a regime-shift fan chart are
written to `results/` and uploaded as the `forecasting-results` artifact.

## Production pieces

* `retrain_policy.CalibrationMonitor` — rolling coverage on live forecasts;
  fires a retraining trigger after 3 consecutive checks below 80%.
* `retrain_policy.promotion_decision` — a challenger replaces the live model
  only if it is ≥1% better on pinball **and** keeps 90% coverage within
  ±5 points in every gated scenario (in-distribution and faulty sensors).
* `retrain_policy.ModelRegistry` — versions, live pointer, rollback.
* `Dockerfile` — runs the test suite at build time, then the full
  train → backtest → gate pipeline.
* CI: unit tests (including PyTorch model tests), training + backtest +
  gate, and a Docker build.

## Run

```bash
pip install -r requirements.txt
pip install -r requirements-torch.txt      # CPU PyTorch
python -m pytest tests
python src/backtest.py
# or: docker build -t plant-forecasting . && docker run plant-forecasting
```

## Limitations

* The plant is simulated so that ground truth, faults, and regime changes
  are known exactly; real plants add unmodelled dynamics and delayed labels.
* One process unit and one physics law; foundation-scale models must share
  structure across many units.
* The fault layer repairs one sensor from its own history; drift needs
  redundant sensors or physics residuals to localise, which is why it is
  handled by monitoring here.
