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
| [Student outcomes analysis](student-outcomes-analysis/) | SQL + causal-flavored efficacy analysis | Students Performance (1,000 students) | Test-prep effect +5.5 to +10.1 points (permutation p<0.001, bootstrap CIs); same analysis ported to SQLite, Snowflake, PySpark, and a live [Tableau dashboard](https://public.tableau.com/app/profile/ahmed.khaled5730/viz/StudentOutcomesDashboard_17844221900050/DoesTestPrepWorkStudentOutcomesDashboard); from-scratch gradient boosting |
| [Language model from scratch](language-model-from-scratch/) | Character-level language modeling | tiny-shakespeare (public domain) | Neural LM beats smoothed n-grams on held-out perplexity; embeddings, backprop, temperature/top-k sampling from scratch |
| [Data-quality verification](data-quality-verification/) | Automated QC with human-in-the-loop routing | 4,700 marketplace records, 10 injected failure modes | **99.3% defect recall, 0.13% leakage** at 8.5% review load — rules + robust statistics + MinHash dedup + LM perplexity filtering |
| [Model serving API](model-serving-api/) | REST API for real-time inference | Pima Indians Diabetes (768 patients, real) | FastAPI service (validation, `/predict`, automated tests, Dockerfile) wrapping the from-scratch logistic regression at holdout AUC **0.836** |
| [NoSQL product catalog](nosql-product-catalog/) | Document-store modeling + aggregation | Synthetic catalog (600 heterogeneous product docs, 1,555 embedded reviews) | **MongoDB** — embedded vs referenced modeling, aggregation-framework analytics (`$group`/`$unwind`/`$lookup`), multikey + compound indexes; CI runs it against a live Mongo service |

## Design principles

- **No ML libraries.** Only NumPy, pandas (data handling) and Matplotlib
  (plots). Everything else — models, metrics, cross-validation,
  standardization — is implemented in the repo. (FastAPI in
  [`model-serving-api`](model-serving-api/) is the one exception, and it's
  a serving/HTTP framework, not a modeling shortcut — the model behind
  its `/predict` endpoint is the same from-scratch NumPy code.)
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
