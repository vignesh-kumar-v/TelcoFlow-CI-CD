import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import typer
from rich.console import Console
from sklearn.metrics import classification_report, roc_auc_score

console = Console()

def get_latest_artifact_path(artifacts_dir: Path):
    runs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not runs:
        raise FileNotFoundError(f"No artifact runs found in {artifacts_dir}")
    latest_run = sorted(runs, key=lambda x: x.name, reverse=True)[0]
    console.log(f"[blue]Using latest artifact: {latest_run.name}")
    return latest_run

def load_artifacts(run_dir: Path):
    console.log("[blue]Loading artifacts...")
    model = joblib.load(run_dir / "model.joblib")
    preprocessor = joblib.load(run_dir / "preprocessor.joblib")
    with open(run_dir / "train_info.json", "r") as f:
        train_info = json.load(f)
    console.log("[green]Artifacts loaded successfully")
    return model, preprocessor, train_info

def load_scoring_data(data_path: Path) -> pd.DataFrame:
    console.log(f"[blue]Loading scoring data from {data_path}...")
    df = pd.read_parquet(data_path)
    console.log(f"[green]Loaded {len(df)} rows to score")
    return df

def generate_predictions(model, preprocessor, df: pd.DataFrame) -> pd.DataFrame:
    console.log("[blue]Generating predictions...")
    X = df.drop(columns=["Churn"], errors="ignore")
    X_processed = preprocessor.transform(X)
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = df.copy()
    predictions["churn_probability"] = probabilities
    predictions["prediction"] = (probabilities >= 0.5).astype(int)
    console.log(f"[green]Generated predictions for {len(predictions)} customers")
    return predictions

def save_predictions(predictions: pd.DataFrame, outputs_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = outputs_dir / f"predictions_{timestamp}.parquet"
    predictions.to_parquet(parquet_path, index=False)
    console.log(f"[green]Saved Parquet: {parquet_path}")
    csv_path = outputs_dir / f"predictions_{timestamp}.csv"
    predictions.to_csv(csv_path, index=False)
    console.log(f"[green]Saved CSV: {csv_path}")
    return parquet_path, csv_path


def generate_drift_report(scoring_df: pd.DataFrame, train_info: dict, reports_dir: Path):
    console.log("[blue]Generating drift report...")
    train_features = train_info["feature_names"]
    scoring_features = [col for col in scoring_df.columns if col != "Churn"]
    missing_features = set(train_features) - set(scoring_features)
    extra_features = set(scoring_features) - set(train_features)
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "n_scoring_samples": len(scoring_df),
        "n_training_samples": train_info["n_samples"],
        "missing_features": list(missing_features),
        "extra_features": list(extra_features),
        "feature_drift": {}
    }
    numeric_features = scoring_df[train_features].select_dtypes(include=[np.number]).columns
    for feature in numeric_features:
        train_mean = train_info.get(f"{feature}_mean", None)
        scoring_mean = scoring_df[feature].mean()
        if train_mean is not None:
            drift_pct = abs(scoring_mean - train_mean) / train_mean * 100
        else:
            drift_pct = 0
        drift_report["feature_drift"][feature] = {
            "train_mean": train_mean,
            "scoring_mean": scoring_mean,
            "drift_percent": drift_pct
        }
        if drift_pct > 10:
            console.log(f"[yellow]Warning: {feature} drift = {drift_pct:.1f}%")

    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = reports_dir / f"drift_report_{timestamp}.json"
    
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)
    
    console.log(f"[green]Drift report saved: {report_path}")
    console.rule("[bold cyan]Drift Analysis Summary")
    console.print(f"Scoring samples: {drift_report['n_scoring_samples']}")
    console.print(f"Training samples: {drift_report['n_training_samples']}")
    console.print(f"Features with >10% drift: {sum(1 for f in drift_report['feature_drift'].values() if f['drift_percent'] > 10)}")
    
    return drift_report

def main(
    scoring_data_path: Path = Path("/home/vignesh/TelecomChurn/data/processed/cleaned_data.parquet"),
    artifacts_dir: Path = Path("artifacts"),
    outputs_dir: Path = Path("outputs"),
    reports_dir: Path = Path("reports")):

    console.rule("[bold magenta]Telco Churn: Batch Scoring Pipeline")
    latest_run_dir = get_latest_artifact_path(artifacts_dir)
    model, preprocessor, train_info = load_artifacts(latest_run_dir)
    scoring_df = load_scoring_data(scoring_data_path)
    predictions_df = generate_predictions(model, preprocessor, scoring_df)
    parquet_path, csv_path = save_predictions(predictions_df, outputs_dir)
    drift_report = generate_drift_report(scoring_df, train_info, reports_dir)
    console.rule("[bold green]Batch scoring completed!")
    console.rule("[bold cyan]Top 10 Churn Risks")
    top_churners = predictions_df.nlargest(10, "churn_probability")[["customerID", "churn_probability"]]
    console.print(top_churners.to_string(index=False))

if __name__ == "__main__":
    typer.run(main)