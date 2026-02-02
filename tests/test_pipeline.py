import json
from pathlib import Path
import pandas as pd


def _get_latest_run():
    artifacts_dir = Path("artifacts")
    runs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    assert len(runs) > 0, "No training runs found in artifacts/"
    return sorted(runs, key=lambda x: x.name, reverse=True)[0]


def test_imports():
    from src.telco_churn import validate_and_clean, train, batch_score
    assert True


def test_data_exists():
    raw_path = Path("data/raw/Telco-Customer-Churn.csv")
    assert raw_path.exists(), f"Raw data not found at {raw_path}"
    assert raw_path.stat().st_size > 0, "Raw data file is empty"

def test_cleaned_data_exists():
    clean_path = Path("data/processed/cleaned_data.parquet")
    assert clean_path.exists(), f"Cleaned data not found at {clean_path}"
    df = pd.read_parquet(clean_path)
    assert len(df) > 0, "Cleaned data is empty"
    assert "Churn" in df.columns, "Target column 'Churn' not found"

def test_artifacts_exist():
    latest_run = _get_latest_run()
    required_files = ["model.joblib", "preprocessor.joblib", "train_info.json", "metrics.json"]
    for file in required_files:
        file_path = latest_run / file
        assert file_path.exists(), f"Missing artifact: {file_path}"

def test_outputs_exist():
    outputs_dir = Path("outputs")
    assert outputs_dir.exists(), "Outputs directory not found"
    parquet_files = list(outputs_dir.glob("predictions_*.parquet"))
    csv_files = list(outputs_dir.glob("predictions_*.csv"))
    assert len(parquet_files) > 0, "No Parquet prediction files found"
    assert len(csv_files) > 0, "No CSV prediction files found"
    latest_parquet = sorted(parquet_files, key=lambda x: x.name, reverse=True)[0]
    df_pred = pd.read_parquet(latest_parquet)
    assert len(df_pred) > 0, "Predictions file is empty"
    assert "churn_probability" in df_pred.columns, "Predictions missing 'churn_probability' column"
    assert "prediction" in df_pred.columns, "Predictions missing 'prediction' column"


def test_model_comparison_exists():
    latest_run = _get_latest_run()
    mc_path = latest_run / "model_comparison.json"
    assert mc_path.exists(), f"model_comparison.json not found in {latest_run}"
    with open(mc_path) as f:
        mc = json.load(f)
    assert len(mc) >= 3, "model_comparison.json should have at least 3 models"
    for model_name, metrics in mc.items():
        assert "roc_auc" in metrics, f"{model_name} missing roc_auc"


def test_shap_plot_exists():
    latest_run = _get_latest_run()
    shap_path = latest_run / "shap_summary.png"
    assert shap_path.exists(), f"shap_summary.png not found in {latest_run}"
    assert shap_path.stat().st_size > 0, "shap_summary.png is empty"


def test_train_info_has_feature_stats():
    latest_run = _get_latest_run()
    with open(latest_run / "train_info.json") as f:
        train_info = json.load(f)
    assert "feature_stats" in train_info, "train_info.json missing feature_stats"
    assert "categorical_distributions" in train_info, "train_info.json missing categorical_distributions"
    assert len(train_info["feature_stats"]) > 0, "feature_stats is empty"
    assert len(train_info["categorical_distributions"]) > 0, "categorical_distributions is empty"
    assert "customerID" not in train_info["feature_names"], "customerID should not be in feature_names"


def test_drift_report_has_numeric_and_categorical():
    reports_dir = Path("reports")
    assert reports_dir.exists(), "Reports directory not found"
    report_files = list(reports_dir.glob("drift_report_*.json"))
    assert len(report_files) > 0, "No drift reports found"
    latest_report = sorted(report_files, key=lambda x: x.name, reverse=True)[0]
    with open(latest_report) as f:
        report = json.load(f)
    assert "numeric_drift" in report, "Drift report missing numeric_drift section"
    assert "categorical_drift" in report, "Drift report missing categorical_drift section"
    assert len(report["numeric_drift"]) > 0, "numeric_drift is empty"
    assert len(report["categorical_drift"]) > 0, "categorical_drift is empty"
