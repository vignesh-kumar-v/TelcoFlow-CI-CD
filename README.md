# TelcoFlow — Customer Churn Prediction Platform

[![ML Pipeline](https://github.com/vignesh-kumar-v/TelcoFlow-CI-CD/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/vignesh-kumar-v/TelcoFlow-CI-CD/actions/workflows/ml-pipeline.yml)
![tests](https://img.shields.io/badge/tests-177-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-yellow)

An end-to-end ML platform that predicts telecom customer churn. Features are
engineered in SQL, models are compared and tuned with Optuna, runs are tracked
and versioned in MLflow, the winner is served over FastAPI, and the whole thing
is orchestrated with Airflow, containerised, deployable to Kubernetes, and
exercised by GitHub Actions on every push.

## Results

Four models compete on test ROC-AUC over a stratified 80/10/10 split
(4,225 train / 1,409 validation / 1,409 test). Any of them can be deployed — the
winner's feature shape is recorded at training time and reproduced at serving.

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** *(deployed)* | **0.8458** | 0.508 | **0.794** | 0.619 |
| Tuned LightGBM (Optuna, 50 trials) | 0.8436 | 0.546 | 0.765 | **0.637** |
| LightGBM | 0.8350 | 0.773 | 0.246 | 0.373 |
| XGBoost | 0.8310 | 0.518 | 0.770 | 0.619 |

- **79.4% recall on churners** — the metric that matters when a missed churner
  is a lost customer and a false positive costs one retention offer.
- **Optuna tuning lifted LightGBM from 0.8350 to 0.8436**, and SQL feature
  engineering lifted the winner from 0.8422 to **0.8458**.
- Untuned LightGBM shows the danger of ranking on AUC alone: 0.773 precision but
  **24.6% recall**, so it misses three churners in four at the 0.5 threshold.

**Top churn drivers** (SHAP on the deployed model): tenure, contract type,
fiber-optic internet, monthly charges, and the engineered `avg_monthly_spend` —
one of the SQL-derived features earning a top-six slot on its own merits.

<p align="center">
  <img src="docs/images/shap-summary.png" alt="SHAP summary plot — feature impact on churn prediction" width="620">
</p>

**Sharpest segment finding** — month-to-month customers in their first six
months on premium plans churn at **77.1%**, nearly **3× the 26.5% base rate**:

| Contract | Tenure | Spend | Customers | Churn rate |
|---|---|---|---|---|
| Month-to-month | 0–6m | premium | 118 | **77.1%** |
| Month-to-month | 0–6m | high | 524 | 72.0% |
| Month-to-month | 1–2y | premium | 177 | 56.5% |

A simulated retention campaign on the 585 highest-risk customers produced a
**+11.6pp retention lift (p = 0.0035)** — see [A/B Test](#ab-test-simulation).

## Problem Statement

Customer churn costs telecom companies billions annually. This project builds a
reproducible pipeline that ingests raw customer records into a database,
engineers features in SQL, trains and compares multiple classifiers, serves
predictions in batch and in real time, watches for data drift, and closes the
loop by measuring whether acting on those predictions actually retains anyone.

## Architecture

```
                     ┌──────────────────────────────────────────┐
  Raw CSV ─────────> │ SQL layer  (SQLite / Postgres / BigQuery) │
                     │  01_clean_customers.sql                  │
                     │  02_feature_engineering.sql              │
                     │  03_churn_analytics.sql                  │
                     └─────────────────┬────────────────────────┘
                                       │ cleaned_data.parquet
                                       ▼
                     ┌──────────────────────────────────────────┐
                     │ Train: LR / LightGBM / XGBoost / Tuned    │───> MLflow
                     │ Optuna tuning · SHAP · best model wins    │     tracking
                     └─────────────────┬────────────────────────┘     + registry
                                       │
                     artifacts/<ts>/   │   data/processed/scoring_batch.parquet
                       model.joblib    │            (held-out split)
                       preprocessor    │
                       train_info.json ▼
              ┌────────────────────────┴───────────────────┐
              ▼                                            ▼
   ┌─────────────────────┐                    ┌───────────────────────┐
   │ Batch scoring       │                    │ FastAPI  /predict     │
   │ + drift detection   │                    │ real-time inference   │
   └──────────┬──────────┘                    └───────────────────────┘
              │ predictions
              ▼
   ┌─────────────────────┐
   │ A/B test simulation │  did contacting them actually help?
   └─────────────────────┘

  Orchestration: Airflow DAG with quality gates   Deployment: Docker / Kubernetes
```

## Quick Start

```bash
git clone https://github.com/vignesh-kumar-v/TelcoFlow-CI-CD.git
cd TelcoFlow-CI-CD

python -m venv venv && source venv/bin/activate
make install

make pipeline        # SQL features -> train -> score -> A/B test
make test
```

`make help` lists every target.

## Pipeline Stages

| Stage | Command | What it does |
|-------|---------|--------------|
| **SQL features** | `make sql-features` | Loads raw records into the DB, cleans and derives features in versioned SQL, exports to parquet |
| **Validate** (alt) | `make validate` | Pure-pandas cleaning path, no database required |
| **Train** | `make train` | Compares 4 models, tunes with Optuna, runs SHAP, logs to MLflow, registers the winner |
| **Score** | `make score` | Scores the held-out batch, writes predictions and a drift report |
| **A/B test** | `make ab-test` | Simulates a retention campaign and tests it for significance |
| **Serve** | `make api` | FastAPI real-time predictions on port 8000 |

## SQL Feature Engineering

Cleaning and feature derivation live in `sql/`, not in pandas. The backend is
chosen by one environment variable — the same SQL runs on both:

```bash
make sql-features                                    # SQLite (default)

docker compose up -d                                 # or Postgres
export DB_URL=postgresql+psycopg2://telco:telco@localhost:5432/telco
make sql-features
```

Derived features:

| Feature | Definition | Why |
|---------|-----------|-----|
| `num_addon_services` | Count of the 6 optional services taken | Churn concentrates among customers paying a lot for very little |
| `avg_monthly_spend` | `TotalCharges / tenure` | Lifetime billing rate, which differs from the current price after any repricing |
| `charges_ratio` | `MonthlyCharges / avg_monthly_spend` | Above 1 means a recent price increase — a classic churn trigger |
| `tenure_bucket` | 0-6m / 6-12m / 1-2y / 2-4y / 4y+ | Churn risk is heavily front-loaded |
| `spend_bucket` | low / medium / high / premium | Price tier |

`03_churn_analytics.sql` aggregates churn by segment. The result is a real
finding, not decoration — the worst segment churns at nearly 3× the base rate:

| Contract | Tenure | Spend | Customers | Churn rate |
|---|---|---|---|---|
| Month-to-month | 0-6m | premium | 118 | **77.1%** |
| Month-to-month | 0-6m | high | 524 | 72.0% |
| Month-to-month | 1-2y | premium | 177 | 56.5% |

*(base rate across all customers: 26.5%)*

**Training/serving parity.** Training features are built in the warehouse, but
an API request never passes through it, so the same derivations exist in Python
(`features.compute_engineered_features`). `tests/test_sql.py` asserts the two
implementations agree row-for-row across all 7,043 customers — divergence there
would be training/serving skew, where the model scores live traffic on
differently-computed inputs.

## Model Selection

Scores are in [Results](#results). The point here is that **any of the four can
be deployed**, which requires reproducing three different feature shapes at
serving time:

| Model | Feature shape it consumes |
|-------|--------------------------|
| Logistic Regression | Dense one-hot matrix (standardised) |
| LightGBM | Native pandas `category` columns |
| XGBoost | Integer category codes |
| Tuned LightGBM | Native categoricals, Optuna-optimised |

Training records which shape the winner needs in `train_info.json`, and both
serving paths transform features accordingly. Category levels are pinned at fit
time, so a scoring batch that happens to omit a category cannot silently shift
every remaining code.

## Experiment Tracking

Every run logs parameters, per-model metrics, each Optuna trial (as a nested
run) and artifacts to MLflow; the winner is registered as a new version of
`telco-churn-classifier`.

```bash
make mlflow-ui                    # http://localhost:5000
make mlflow-ui MLFLOW_PORT=5001   # macOS binds 5000 to AirPlay Receiver
```

Each Optuna trial is a nested run, so the whole search is inspectable rather
than collapsed into a single best-params line — 50 trials spanning validation
AUC 0.8126 to 0.8397 below:

![MLflow runs with nested Optuna trials](docs/images/mlflow-runs.png)

The winner is registered as a new version, each annotated with the algorithm
and score at registration time:

![MLflow model registry versions](docs/images/mlflow-registry.png)

Tracking is best-effort: if MLflow is unavailable the run logs a warning and
training completes anyway. A metrics sink should not be able to take down the
pipeline.

## Drift Detection

Batch scoring runs against the **held-out** split, not the training data, so
drift numbers are meaningful. Numeric features are compared on mean/std shift
(>10%), categoricals on Total Variation Distance (>0.1).

```bash
make score          # clean holdout    -> 0/7 numeric, 0/17 categorical drifted
make score-drift    # perturbed batch  -> 3/7 numeric, 1/17 categorical drifted
```

## A/B Test Simulation

Scoring customers is half the problem; the business question is whether acting
on the scores retains anyone. High-risk customers are assigned to a retention
offer or no contact, **stratified by predicted-risk decile** so both arms carry
the same baseline risk, then evaluated:

- Two-proportion z-test and chi-square on retention
- Welch's t-test on net revenue per customer
- 95% confidence intervals, Cohen's h/d, and the minimum detectable effect
- An **A/A negative control** that must come out non-significant

```
A/B test — retention offer (20% assumed churn reduction)
Metric               Treatment  Control  Difference  p-value  Verdict
Retention rate           41.4%    29.8%      +11.6%   0.0035  significant
Net revenue/customer      $325     $263        +$62   0.1036  not significant

A/A negative control — no treatment effect injected
Retention rate           26.2%    23.4%       +2.8%   0.4301  not significant
```

The revenue result being *non*-significant while retention is significant is the
honest readout: a $50 offer against high-variance revenue needs a larger sample.
Randomisation is checked too — mean predicted risk was 0.7351 (treatment) vs
0.7347 (control), p = 0.968.

**This is a simulation, not a live experiment.** The outcome model assumes the
treatment reduces each customer's predicted churn probability by a fixed
relative amount. It demonstrates the analysis machinery — stratification,
significance testing, power, negative controls — not a measured business result.

## API

```bash
make api          # or: make docker-run
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness — 200 whenever the process is up |
| `GET` | `/ready` | Readiness — 503 until a model is loaded |
| `POST` | `/predict` | Churn probability and risk band for one customer |
| `GET` | `/model/info` | Deployed version, type, metrics and top SHAP features |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customerID":"7590-VHVEG","gender":"Female","SeniorCitizen":0,"Partner":"Yes",
       "Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service",
       "InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
       "Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":29.85,"TotalCharges":29.85}'
```

```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.8414804970801406,
  "risk_category": "High",
  "model_version": "20260830-124941",
  "timestamp": "2026-08-30T12:51:43.594095"
}
```

Pydantic validates every field before the model is touched, so malformed
requests get a 422 rather than an opaque 500. Interactive docs are generated at
`/docs`:

![FastAPI interactive docs returning a live prediction](docs/images/api-docs.png)

Liveness and readiness are separate on purpose: probes check status codes, not
bodies, so a 200 response saying "no model" would still get the pod added to the
load balancer. `/ready` returns 503 instead, and retries loading — so a pod that
starts before training finishes becomes ready on its own, without a restart.

## Orchestration (Airflow)

```
ingest_sql → train → evaluate_gate ─┬─> promote_model → batch_score → drift_gate → ab_test
                                    └─> reject_model  (fails the run)
```

`evaluate_gate` branches on test ROC-AUC against a threshold, so a regression
stops at `reject_model` instead of quietly shipping; `drift_gate` fails the run
when the scored batch has drifted. Gate logic lives in `src/telco_churn/gates.py`
— stdlib only, unit-tested without an Airflow install.

Airflow pins its own dependency versions, so it gets a separate environment:

```bash
make airflow-install    # isolated venv, official Airflow constraints
make airflow-run        # http://localhost:8080
```

Verified by running the DAG end to end on Airflow 2.10.3: all 8 tasks succeeded,
the gate branched to `promote_model` and correctly skipped `reject_model`, and
raising the threshold above the achievable AUC flipped it to a failed run with
`Model rejected: ROC-AUC 0.8458 is below the 0.99 threshold`.

The scheduler environment has **no ML dependencies at all** — verified by
importing `gates.py` there with pandas, scikit-learn, LightGBM, XGBoost, SHAP
and MLflow all absent. The gates are stdlib-only; everything heavier is shelled
out to the project venv.

Both verdict tasks run with `retries=0`. A quality gate reaching a verdict on a
finished model is deterministic, so retrying reaches the same conclusion and
only delays the alert — the transient-failure tasks around them keep `retries=2`.

## Docker

```bash
make docker-build     # 1.69GB
make docker-run       # API on :8000
```

The image derives a runtime subset from `requirements.txt` rather than keeping a
second dependency list: `xgboost` becomes `xgboost-cpu` (the Linux xgboost wheel
hard-depends on 457MB of CUDA this image never runs) and notebook/test packages
are dropped. Verified to produce byte-identical predictions to a host run.

## Kubernetes

Enable Kubernetes in Docker Desktop (Settings → Kubernetes), or use minikube —
see [`k8s/README.md`](k8s/README.md) for both paths.

```bash
docker build -t telco-churn-mlops:latest .
make k8s-deploy && make k8s-status
```

Deployment (2 replicas, startup/readiness/liveness probes, zero-downtime
rollout) + Service, a training Job, and a nightly scoring CronJob, sharing a
persistent artifacts volume.

Verified end to end on a live cluster: the training Job writes the model to the
shared PVC, the API pods sit at `0/1` and out of the Service until it appears,
then reach `1/1` with zero restarts, and the CronJob scores the held-out batch
using the model the Job produced.

See [`k8s/README.md`](k8s/README.md) for probe design, the `:latest` image trap,
and the multi-node storage caveat.

## BigQuery

```bash
gcloud auth application-default login    # one-time

make bq-load          # cleaned data -> BigQuery
make bq-features      # query engineered features back out
make bq-analytics     # churn by segment, computed in BigQuery
make bq-predictions   # publish scored customers
make bq-all           # load, analyse and publish
```

`GCP_PROJECT` defaults to the active gcloud project, and nothing contacts
Google without it — the rest of the pipeline runs unchanged with no cloud
credentials.

The same feature definitions run in three places, and the results are asserted
equal, not assumed:

| Backend | Where features are built |
|---|---|
| SQLite / Postgres | `sql/02_feature_engineering.sql` |
| BigQuery | `FEATURE_QUERY` in `bigquery_loader.py` |
| Python (serving) | `features.compute_engineered_features` |

`test_bigquery_features_match_local_computation` checks the warehouse output
against the Python serving path over all 7,043 rows; `TestTrainServeParity`
does the same for the local SQL. Verified exact — zero difference across every
engineered column.

Reads go through pandas-gbq and the BigQuery Storage API, streaming results
over gRPC rather than paging REST.

## Tests

```bash
make test               # everything
make unit-test          # no artifacts needed — runs on a cold checkout
make integration-test   # asserts against a completed pipeline run
```

**177 tests** — 34 unit, 30 Kubernetes manifest, 25 SQL, 22 A/B, 21
orchestration, 17 BigQuery, 16 integration, 12 API. Integration tests **skip**
rather than fail when no pipeline has run, so a fresh clone is green:
**155 pass, 22 skip, 0 fail** with no artifacts on disk. Notable coverage:

- Category codes stay stable when a scoring batch omits a category
- The deployed model is the comparison winner (not a filtered subset's)
- All three feature shapes are servable from the real fitted artifact
- SQL, BigQuery and Python feature implementations agree row-for-row
- A/A false-positive rate stays near α across 120 simulations
- k8s manifests: claim references resolve, Service selector matches pod labels
- The Airflow DAG imports no ML dependencies

## CI/CD

Four parallel jobs on every push and PR: unit tests (fast feedback) → full
pipeline + integration tests, alongside manifest/DAG validation and a Docker
build. Pip downloads and Docker layers are cached, trained models and reports
are uploaded as workflow artifacts, and the model comparison table is written to
the run summary.

## Project Structure

```
├── .github/workflows/ml-pipeline.yml  # 4 parallel CI jobs
├── airflow/dags/telco_churn_dag.py    # orchestration with quality gates
├── k8s/                               # Deployment, Service, Job, CronJob
├── sql/                               # versioned cleaning + feature SQL
├── notebooks/eda.ipynb                # exploratory analysis
├── src/telco_churn/
│   ├── features.py                    # feature defs + fitted transform
│   ├── inference.py                   # shared serving path
│   ├── gates.py                       # promotion / drift predicates (stdlib only)
│   ├── db.py, sql_features.py         # SQL layer
│   ├── train.py, batch_score.py       # training and batch scoring
│   ├── tracking.py                    # MLflow
│   ├── ab_test.py                     # experiment simulation
│   ├── bigquery_loader.py             # warehouse path
│   └── api.py                         # FastAPI
├── tests/                             # 177 tests
├── Dockerfile / .dockerignore         # 1.69GB runtime image
├── docker-compose.yml                 # Postgres + MLflow server
└── requirements.txt / requirements-airflow.txt
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 customers, 20 features, 26.5% churn rate. The 11 records with a blank
`TotalCharges` are all `tenure = 0` (new customers, never billed) and are
median-imputed.

## Limitations and next steps

Known constraints, stated plainly:

- **The dataset is a static snapshot.** There is no time dimension, so the
  train/test split cannot be temporal. A production churn model should be
  validated on a forward time window, since customer mix drifts.
- **The A/B test is simulated.** Outcomes are drawn from the model's own
  predicted probabilities, so it validates the analysis, not the intervention.
- **Feature logic exists in three implementations** (local SQL, BigQuery SQL,
  Python). Parity tests catch divergence, but a single source — dbt models, or
  a feature store — would remove the duplication entirely.
- **The artifact store is a shared filesystem.** That works on a single node;
  a multi-node cluster needs ReadWriteMany or, better, serving pulling from the
  MLflow registry rather than a mounted volume.
- **Decision threshold is fixed at 0.5.** With a 2.7:1 imbalance, the operating
  point should be chosen from the cost of a missed churner versus a wasted
  retention offer, not left at the default.
- **No model monitoring in production.** Drift is computed at scoring time, but
  there is no alerting or automated retraining trigger wired to it.

## Tech Stack

**ML:** LightGBM · XGBoost · scikit-learn · Optuna · SHAP
**Serving:** FastAPI · Uvicorn · Pydantic
**Data:** pandas · NumPy · PyArrow · SQLAlchemy · SQLite/Postgres · BigQuery
**MLOps:** MLflow · Airflow · Docker · Kubernetes · GitHub Actions
**Stats:** SciPy

## License

[MIT](LICENSE)
