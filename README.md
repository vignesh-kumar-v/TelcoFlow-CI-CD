# TelcoFlow — Customer Churn Prediction Pipeline

An end-to-end machine learning pipeline that predicts customer churn for a telecom provider. Compares Logistic Regression, LightGBM, and XGBoost models with Optuna hyperparameter tuning, SHAP explainability, and data drift detection. Built with FastAPI, containerized with Docker, orchestrated via Make, and automated through GitHub Actions CI/CD.

## Problem Statement

Customer churn costs telecom companies billions annually. This project builds a reproducible ML pipeline that ingests raw customer data, validates and cleans it, trains and compares multiple classifiers, generates batch predictions with churn probabilities, and monitors for data drift — all runnable with a single command.

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
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   └── telco_churn/
│       ├── __init__.py
│       ├── validate_and_clean.py # Stage 1: Schema validation & data cleaning
│       ├── train.py              # Stage 2: Multi-model training, Optuna tuning & SHAP
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
├── artifacts/                    # Timestamped model artifacts
│   └── <timestamp>/
│       ├── model.joblib          # Best model
│       ├── preprocessor.joblib   # Fitted preprocessor
│       ├── train_info.json       # Feature stats & categorical distributions
│       ├── metrics.json          # Best model metrics + SHAP importance
│       ├── model_comparison.json # All models' metrics
│       └── shap_summary.png     # SHAP feature importance plot
├── outputs/                      # Batch predictions (parquet + csv)
└── reports/                      # Drift analysis reports (json)
```

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌──────────┐
│  Raw CSV     │────>│  Validate &  │────>│  Train & Compare  │────>│  Batch   │
│  (7,043 rows)│     │  Clean       │     │  LR/LGBM/XGB/Tuned│     │  Score   │
└──────────────┘     └──────┬───────┘     └───────┬───────────┘     └────┬─────┘
                            │                     │                      │
                     cleaned_data.parquet   model.joblib           predictions.csv
                                            model_comparison.json  drift_report.json
                                            shap_summary.png
```

| Stage | Script | What It Does |
|-------|--------|--------------|
| **Validate & Clean** | `validate_and_clean.py` | Schema validation against 21 expected columns, `TotalCharges` type coercion, median imputation for missing values, binary target encoding (`Yes/No` → `1/0`) |
| **Train** | `train.py` | Stratified 80/10/10 split, proper categorical encoding (OneHot for LR, native category for LGBM), class imbalance handling, multi-model comparison (LR, LightGBM, XGBoost, Tuned LightGBM), Optuna hyperparameter tuning, SHAP feature importance |
| **Batch Score** | `batch_score.py` | Loads best model, generates churn probabilities, outputs predictions in Parquet & CSV, runs numeric + categorical drift detection (mean/std percentage thresholds + Total Variation Distance) |

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
make train         # Full training with Optuna tuning (50 trials)
make score

# Or run without tuning (faster)
make train-quick

# Run tests
make test
make unit-test

# Clean bytecode
make clean
```

## Model Comparison

The training pipeline automatically compares 4 models and selects the best by ROC-AUC:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline with balanced class weights |
| **LightGBM** | Gradient boosted trees with native categorical handling |
| **XGBoost** | Gradient boosted trees with encoded categoricals |
| **Tuned LightGBM** | LightGBM with Optuna-optimized hyperparameters |

## Feature Importance

SHAP (SHapley Additive exPlanations) analysis runs on the best tree-based model, producing:
- `shap_summary.png` — beeswarm plot showing feature impact on predictions
- `shap_feature_importance` in `metrics.json` — ranked mean absolute SHAP values

## EDA Notebook

`notebooks/eda.ipynb` provides exploratory analysis including:
- Dataset overview and missing value analysis
- Class distribution visualization (26.5% churn rate)
- Numeric feature distributions by churn status
- Categorical feature churn rate analysis
- Correlation analysis with point-biserial correlations

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
| `train` | `make train` | Train all models with Optuna tuning |
| `train-quick` | `make train-quick` | Train all models without Optuna tuning |
| `tune` | `make tune` | Train with 100 Optuna trials |
| `score` | `make score` | Generate batch predictions + drift report |
| `test` | `make test` | Run all tests |
| `unit-test` | `make unit-test` | Run unit tests only |
| `api` | `make api` | Start FastAPI server locally |
| `docker-build` | `make docker-build` | Build Docker image |
| `docker-run` | `make docker-run` | Run API server in Docker |
| `clean` | `make clean` | Remove bytecode and caches |

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithms | LightGBM, XGBoost, Logistic Regression |
| Max Estimators | 500 (default), up to 1000 (tuned) |
| Learning Rate | 0.05 (default), 0.01–0.3 (tuned) |
| Num Leaves | 31 (default), 15–127 (tuned) |
| Early Stopping | 50 rounds (validation AUC) |
| Class Imbalance | `scale_pos_weight` (tree models), `class_weight="balanced"` (LR) |
| Preprocessing | OrdinalEncoder (Contract), category dtype (nominals), OneHotEncoder (LR) |
| Split Strategy | Stratified 80/10/10 (train/val/test) |
| Tuning | Optuna (50 trials default, configurable via `--n-trials`) |
| Explainability | SHAP TreeExplainer on best model |

## Dataset

The [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle.

- **Records:** 7,043 customers
- **Features:** 20 (demographics, account info, service subscriptions)
- **Target:** Churn (binary — 26.5% positive class)

| Feature Type | Count | Examples |
|-------------|-------|----------|
| Nominal | 14 | gender, Partner, InternetService, PaymentMethod |
| Ordinal | 1 | Contract (Month-to-month < One year < Two year) |
| Numeric | 4 | SeniorCitizen, tenure, MonthlyCharges, TotalCharges |
| Identifier | 1 | customerID (dropped before training) |

## CI/CD

GitHub Actions runs the full pipeline on every push to `main` and on pull requests:

```
Install dependencies → Validate data → Train (--skip-tuning) → Batch score → Run tests
```

See [`.github/workflows/ml-pipeline.yml`](.github/workflows/ml-pipeline.yml) for the workflow definition.

## Tech Stack

- **ML:** LightGBM, XGBoost, scikit-learn, Optuna, SHAP
- **API:** FastAPI, Uvicorn, Pydantic
- **Data:** pandas, NumPy, PyArrow
- **CLI:** Typer, Rich
- **Serialization:** joblib, Parquet
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Orchestration:** GNU Make

## License

[MIT](LICENSE)
