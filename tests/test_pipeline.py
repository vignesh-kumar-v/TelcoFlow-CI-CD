from pathlib import Path
import pandas as pd

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
    artifacts_dir = Path("src/telco_churn/artifacts")
    assert artifacts_dir.exists(), "Artifacts directory not found"
    runs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    assert len(runs) > 0, "No training runs found in artifacts/"
    latest_run = sorted(runs, key=lambda x: x.name, reverse=True)[0]
    required_files = ["model.joblib", "preprocessor.joblib", "train_info.json", "metrics.json"]
    for file in required_files:
        file_path = latest_run / file
        assert file_path.exists(), f"Missing artifact: {file_path}"

def test_outputs_exist():
    outputs_dir = Path("src/telco_churn/outputs")
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