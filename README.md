# Telecom Customer Churn Prediction

A machine learning pipeline to predict customer churn for a telecommunications company using LightGBM.

## Project Structure

```
TelecomChurn/
├── .github/workflows/
│   └── ml-pipeline.yml        # CI/CD pipeline (validate → train → score → test)
├── src/telco_churn/
│   ├── validate_and_clean.py  # Data validation and cleaning
│   ├── train.py               # Model training (LightGBM)
│   ├── batch_score.py         # Batch scoring with drift detection
│   └── Makefile               # Task runner
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv  # Raw dataset (7,043 customers, 21 features)
└── ML/
    └── AdaBoost.py            # Custom AdaBoost implementation
```

## Pipeline

The ML pipeline consists of three stages:

1. **Validate & Clean** (`validate_and_clean.py`) — Schema validation, type conversion, missing value imputation, target encoding
2. **Train** (`train.py`) — 80/10/10 train/val/test split, OrdinalEncoder for categoricals, LightGBM with early stopping
3. **Batch Score** (`batch_score.py`) — Generate churn probabilities for all customers, detect feature drift (>10% threshold)

## Setup

```bash
python -m venv venv
source venv/bin/activate
cd src/telco_churn
make install
```

## Usage

```bash
cd src/telco_churn

# Run full pipeline
make validate
make train
make score

# Run tests
make test
```

## Key Dependencies

- pandas, numpy
- scikit-learn
- lightgbm
- typer, rich
- joblib

## Dataset

The Telco Customer Churn dataset contains 7,043 customer records with 20 features including demographics, account information, and service subscriptions. The target variable is binary churn (Yes/No).

## Model Performance

- **Algorithm:** LightGBM Classifier (500 estimators, lr=0.05, early stopping at 50 rounds)
- **ROC-AUC:** 0.82
- **Precision / Recall / F1:** 0.57
