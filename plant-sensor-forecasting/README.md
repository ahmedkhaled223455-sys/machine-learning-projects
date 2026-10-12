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

From the CI run (*Train + backtest + promotion gate*). Pinball loss and MAE
in metres (lower is better); a calibrated 90% interval covers ~0.90.

| model | in-regime pinball | regime-shift pinball | regime-shift 90% coverage | faulty sensor (cleaned) pinball |
|---|---|---|---|---|
| persistence | 0.0358 | 0.0400 | 0.887 | 0.0421 |
| Holt | 0.0441 | 0.0466 | 0.920 | 0.0496 |
| ARX (best classical) | 0.0231 | 0.0781 | **0.428** | 0.0287 |
| physics only | 0.0335 | 0.0482 | 0.768 | 0.0350 |
| pure net | 0.0230 | 0.0344 | 0.890 | 0.0274 |
| pure ensemble | 0.0225 | 0.0357 | 0.900 | 0.0268 |
| hybrid net | 0.0228 | 0.0321 | 0.897 | 0.0270 |
| **hybrid ensemble** | **0.0226** | **0.0316** | **0.902** | **0.0266** |

Full table (MAE, interval width, infeasibility, faults with and without
cleaning): `results/backtest.md`, produced by `python src/backtest.py`.

What it shows -- including where the fancy models *don't* win:

* **In regime, nothing beats a good linear model by much.** The networks
  edge ARX by only ~2--3% on pinball; most of the remaining error is the
  unmeasured disturbance, which no model can see. Reported as is.
* **Out of regime is where it matters.** ARX's 90% interval covers only 43%
  of outcomes once the plant leaves the training range -- confidently
  wrong. The hybrid ensemble keeps coverage at 0.90 with **60% lower pinball
  than ARX** and 34% lower than physics alone, and beats the pure network
  (0.0316 vs 0.0344): learning *corrections to physics* extrapolates better
  than learning the dynamics from scratch.
* **The ensemble adds little here.** It slightly helps the hybrid and
  slightly hurts the pure model under shift -- recalibration, not member
  disagreement, does most of the work on this problem.
* **Cleaning the inputs matters more than the model.** On the same faulty
  sensor, the robust input layer cuts the hybrid ensemble's pinball from
  0.0351 to 0.0266 (−24%). Coverage under faults is still 0.85, below the
  nominal 0.90 -- a known gap.
* **Drift is caught by monitoring.** On clean data the rolling-coverage
  monitor never fires (minimum 0.82); with a drifting level sensor it fires
  at forecast 1,600 as coverage collapses to 0.11.
* **The gate promoted the hybrid ensemble over ARX** -- better pinball on the
  gated scenarios with coverage inside tolerance -- and the registry keeps
  ARX as the rollback target.

Training: 5-member ensembles of each family train in ~10 s each on a CI CPU
(early stopping after 12--18 epochs); the full pipeline runs in ~30 s.

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
