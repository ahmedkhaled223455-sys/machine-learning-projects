# Machine Learning Projects — From First Principles

A portfolio of machine learning projects where the core algorithms —
logistic regression, decision trees, random forests, k-means++, PCA,
anomaly detectors, and a neural network with backpropagation — are
**implemented from scratch in NumPy** and validated on real datasets.

The premise: anyone can call `model.fit()`. These projects demonstrate
what happens inside it — every gradient, split criterion, and distance
computation is written out, tested, and verified (the neural network's
backprop is checked against numerical gradients to ~1e-8 relative error).

## Projects

| Project | Task | Data | Headline result |
|---------|------|------|-----------------|
| [Diabetes risk prediction](diabetes-risk-prediction/) | Binary classification | Pima Indians Diabetes (768 patients, real) | Holdout AUC **0.836** (logistic regression), leakage-safe CV pipeline |
| [Sensor anomaly detection](sensor-anomaly-detection/) | Unsupervised fault detection | Simulated bearing vibration (physics-based generator) | AUC **0.976**, 88% detection at 0.7% false alarms |
| [Fuel efficiency regression](fuel-efficiency-regression/) | Regression | Auto MPG (398 cars, real) | 10-fold CV R² **0.817**, RMSE 3.27 vs 7.79 baseline |
| [Neural network from scratch](neural-network-from-scratch/) | Nonlinear classification | Two moons / two spirals (synthetic) | Spirals **100%** test acc vs 62% linear baseline; gradient check 6e-8 |
| [Customer segmentation](customer-segmentation/) | Clustering | Mall Customers (200 customers, real) | k=6 segments, silhouette 0.427, actionable profiles |

## Design principles

- **No ML libraries.** Only NumPy, pandas (data handling) and Matplotlib
  (plots). Everything else — models, metrics, cross-validation,
  standardization — is implemented in the repo.
- **Leakage-safe evaluation.** Imputation and scaling are always fit on
  training folds only. Every headline number is computed on data the
  model never saw.
- **Honest results.** Metrics come from running the code in this repo,
  and known limitations are documented in each project README.
- **Reproducible.** Fixed seeds; each project runs end-to-end in under
  ten seconds on a laptop.

## Setup

```bash
pip install numpy pandas matplotlib
cd diabetes-risk-prediction   # or any project folder
python src/train.py           # entry point named in each README
```

Python ≥ 3.10. Each project also lists its own `requirements.txt`.

## License

MIT — see [LICENSE](LICENSE).
