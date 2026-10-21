# Trajectory prediction with a transformer — multimodal, social, benchmarked

Predict where a car on a highway will be over the next **3 seconds** from
**2 seconds** of noisy observations of it and its neighbours — the
behaviour-prediction problem at the heart of autonomous driving. Built with
a **PyTorch transformer** (temporal + social attention, learned mode
queries), trained with the **winner-takes-all multimodal objective**, and
benchmarked honestly against classical baselines and ablations with the
standard metrics (**minADE / minFDE / miss rate**).

## Why multimodal

Simulated drivers decide *stochastically* when to change lanes or slow
down. The same 2-second history can be followed by keeping the lane,
moving left, or moving right — a single "average" prediction lands between
lanes and is wrong for all of them. The model therefore outputs **K = 6
candidate futures plus a confidence for each**, and is scored on the best
candidate (minADE@6) as well as its single most confident one (@1).

## Data

`src/traj_sim.py` — a 3-lane highway simulator with known ground truth:

| piece | what it does |
|---|---|
| car-following | Intelligent Driver Model (IDM), per-driver desired speed and headway |
| lane changes | simplified MOBIL rule: change when the target lane is meaningfully faster and the new follower is not forced to brake hard; plus rare unmotivated changes; 3–5 s smooth lateral profile |
| slowdowns | random events that drop a driver's desired speed, then recover |
| sensing | 10 Hz, 0.1 m position noise on observed history; the target is the *true* future |

`src/traj_dataset.py` builds agent-centric samples: the target plus up to
8 nearest neighbours (within 60 m), in the target's frame at the last
observed step, with a padding mask. Train 24,000 / test 4,800 samples, plus
a **dense-traffic shift set** (more, slower cars) never seen in training.
About 18% of test samples contain a lane change in the future window.

## Model (`src/traj_model.py`)

```
history (B, agents, 20 steps, [x, y, vx, vy])
  -> linear embed + learned time encoding + agent-type embedding
  -> temporal transformer   (self-attention over each agent's 20 steps)
  -> social transformer     (self-attention across agents, key-padding mask)
  -> mode decoder           (6 learned queries: self-attention, then
                             cross-attention to the scene tokens)
  -> per mode: 30-step trajectory residual + confidence logit
  -> trajectory = constant-velocity rollout + residual
```

Design choices, each tested:

- **Attention written from primitives** and verified against an independent
  NumPy reference (`src/attention_ref.py`) and PyTorch's
  `scaled_dot_product_attention`.
- **No positional encoding across agents**, so predictions are invariant to
  neighbour ordering by construction (tested by permuting slots).
- **Key-padding mask** for empty neighbour slots: filling them with garbage
  leaves predictions unchanged (tested).
- **Physics anchor**: the network predicts residuals on a constant-velocity
  rollout; with the residual head zeroed the model reproduces the CV
  baseline exactly (tested).
- **Winner-takes-all loss**: smooth-L1 on the mode closest to the truth plus
  cross-entropy teaching the confidences which mode that was.

## Baselines and ablations

| model | purpose |
|---|---|
| constant velocity | the classic kinematic baseline |
| constant acceleration | quadratic fit — shows how noise hurts higher-order extrapolation |
| ridge regression | strong linear baseline on the target's own history |
| transformer, K = 6 | full model |
| transformer, K = 6, no social | ablation: neighbours masked out — does interaction matter? |
| transformer, K = 1 | ablation: single-mode — does multimodality matter? |

## Results

Produced by CI (`python src/train_eval.py`, CPU, fixed seeds); full tables
in `results/results.md`, raw numbers in `results/metrics.json`, example
predictions in `results/examples.png`.

<!-- RESULTS -->
_Results are written by the CI run and copied here._
<!-- /RESULTS -->

## Run it

```bash
pip install -r requirements.txt
pip install -r requirements-torch.txt      # CPU PyTorch
python -m pytest -v tests
python src/train_eval.py                   # ~20-30 min on CPU
python src/train_eval.py --quick           # small smoke run
```

Or with Docker (runs the tests during the build):

```bash
docker build -t traj-transformer .
docker run --rm -v "$PWD/results:/app/results" traj-transformer
```

## Limits

The data is simulated: straight highway, three lanes, no on-ramps, no map
input, and drivers that follow known rules. That is deliberate — it gives
exact ground truth and controlled multimodality — but real driving logs
are messier, and a production model would add map/lane-graph encoding,
more agent types, and much larger scale.
