# TelcoFlow — Customer Churn Prediction Pipeline

An end-to-end machine learning pipeline that predicts customer churn for a telecom provider. Built with LightGBM, served via FastAPI, containerized with Docker, orchestrated via Make, and automated through GitHub Actions CI/CD.

## Problem Statement

Customer churn costs telecom companies billions annually. This project builds a reproducible ML pipeline that ingests raw customer data, validates and cleans it, trains a binary classifier, generates batch predictions with churn probabilities, and monitors for data drift — all runnable with a single command.

## Project Structure

```
TelcoFlow-CI-CD/
├── .github/workflows/
│   └── ml-pipeline.yml           # CI/CD: validate → train → score → test
├── Dockerfile                    # Multi-stage Docker build
├── Makefile                      # Pipeline orchestration
├── requirements.txt              # Pinned Python dependencies
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── src/
│   ├── __init__.py
│   └── telco_churn/
│       ├── __init__.py
│       ├── validate_and_clean.py # Stage 1: Schema validation & data cleaning
│       ├── train.py              # Stage 2: Model training & evaluation
│       ├── batch_score.py        # Stage 3: Batch inference & drift detection
│       └── api.py                # FastAPI real-time prediction endpoint
└── tests/
    ├── __init__.py
    ├── test_pipeline.py          # Integration tests
    └── test_unit.py              # Unit tests
```

**Generated at runtime (gitignored):**
```
├── data/processed/               # Cleaned parquet data
├── artifacts/                    # Timestamped model artifacts (model.joblib, metrics.json)
├── outputs/                      # Batch predictions (parquet + csv)
└── reports/                      # Drift analysis reports (json)
```

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  Raw CSV     │────>│  Validate &  │────>│  Train LightGBM  │────>│  Batch   │
│  (7,043 rows)│     │  Clean       │     │  + Evaluate      │     │  Score   │
└──────────────┘     └──────┬───────┘     └───────┬──────────┘     └────┬─────┘
                            │                     │                     │
                     cleaned_data.parquet   model.joblib          predictions.csv
                                            metrics.json          drift_report.json
```

| Stage | Script | What It Does |
|-------|--------|--------------|
| **Validate & Clean** | `validate_and_clean.py` | Schema validation against 21 expected columns, `TotalCharges` type coercion, median imputation for missing values, binary target encoding (`Yes/No` → `1/0`) |
| **Train** | `train.py` | Stratified 80/10/10 split, OrdinalEncoder for 16 categorical features, LightGBM with early stopping on validation AUC, saves model + preprocessor + metrics as timestamped artifacts |
| **Batch Score** | `batch_score.py` | Loads latest trained model, generates churn probabilities for all customers, outputs predictions in Parquet & CSV, runs numeric feature drift detection (>10% mean shift triggers warning) |

## Quick Start

```bash
# Clone
git clone https://github.com/vignesh-kumar-v/TelcoFlow-CI-CD.git
cd TelcoFlow-CI-CD

# Setup
python -m venv venv
source venv/bin/activate
make install

# Run the full pipeline
make validate
make train
make score

# Run tests
make test
make unit-test

# Clean bytecode
make clean
```

## Docker

```bash
# Build the image
make docker-build

# Run batch scoring
docker run --rm \
  -v $(pwd)/data:/home/mluser/app/data \
  -v $(pwd)/outputs:/home/mluser/app/outputs \
  -v $(pwd)/artifacts:/home/mluser/app/artifacts \
  telco-churn-mlops make score

# Run the API server
make docker-run
```

## API

FastAPI serves real-time churn predictions on port 8000.

```bash
# Start locally
make api

# Start via Docker
make docker-run
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check for load balancers |
| `POST` | `/predict` | Predict churn for a single customer |
| `GET` | `/model/info` | Model version and performance metrics |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

**Example response:**
```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.435,
  "risk_category": "Medium",
  "timestamp": "2026-02-01T19:00:00"
}
```

## Makefile Targets

| Target | Command | Description |
|--------|---------|-------------|
| `install` | `make install` | Install Python dependencies |
| `validate` | `make validate` | Run data validation and cleaning |
| `train` | `make train` | Train LightGBM model |
| `score` | `make score` | Generate batch predictions |
| `test` | `make test` | Run all tests |
| `unit-test` | `make unit-test` | Run unit tests only |
| `api` | `make api` | Start FastAPI server locally |
| `docker-build` | `make docker-build` | Build Docker image |
| `docker-run` | `make docker-run` | Run API server in Docker |
| `clean` | `make clean` | Remove bytecode and caches |

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM (Gradient Boosted Decision Trees) |
| Max Estimators | 500 |
| Learning Rate | 0.05 |
| Num Leaves | 31 |
| Early Stopping | 50 rounds (validation AUC) |
| Preprocessing | OrdinalEncoder (handles unseen categories) |
| Split Strategy | Stratified 80/10/10 (train/val/test) |
| Random State | 42 |

## Results

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.82** |
| Precision | 0.57 |
| Recall | 0.57 |
| F1-Score | 0.57 |

Evaluated on a held-out test set of 1,409 customers (20% of data) with stratified sampling to preserve class distribution.

## Dataset

The [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle.

- **Records:** 7,043 customers
- **Features:** 20 (demographics, account info, service subscriptions)
- **Target:** Churn (binary — 26.5% positive class)

| Feature Type | Count | Examples |
|-------------|-------|----------|
| Categorical | 16 | gender, Contract, InternetService, PaymentMethod |
| Numeric | 3 | tenure, MonthlyCharges, TotalCharges |
| Identifier | 1 | customerID |

## CI/CD

GitHub Actions runs the full pipeline on every push to `main` and on pull requests:

```
Install dependencies → Validate data → Train model → Batch score → Run tests
```

See [`.github/workflows/ml-pipeline.yml`](.github/workflows/ml-pipeline.yml) for the workflow definition.

## Tech Stack

- **ML:** LightGBM, scikit-learn
- **API:** FastAPI, Uvicorn, Pydantic
- **Data:** pandas, NumPy, PyArrow
- **CLI:** Typer, Rich
- **Serialization:** joblib, Parquet
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Orchestration:** GNU Make

## License

[MIT](LICENSE)
