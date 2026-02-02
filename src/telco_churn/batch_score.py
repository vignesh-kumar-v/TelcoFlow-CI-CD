import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import typer
from rich.console import Console
from src.telco_churn.train import (
    DROP_COLS, NOMINAL_FEATURES, ORDINAL_FEATURES, NUMERIC_FEATURES,
    prepare_data_lgbm,
)

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
    X = df.drop(columns=["Churn"] + DROP_COLS, errors="ignore")
    X_processed, _ = prepare_data_lgbm(X, preprocessor)
    X_pred = X_processed.copy()
    for col in X_pred.columns:
        if hasattr(X_pred[col], "cat"):
            X_pred[col] = X_pred[col].cat.codes
    try:
        probabilities = model.predict_proba(X_processed)[:, 1]
    except Exception:
        probabilities = model.predict_proba(X_pred)[:, 1]
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

    scoring_X = scoring_df.drop(columns=["Churn"] + DROP_COLS, errors="ignore")
    train_features = train_info["feature_names"]
    missing_features = set(train_features) - set(scoring_X.columns)
    extra_features = set(scoring_X.columns) - set(train_features)

    feature_stats = train_info.get("feature_stats", {})
    categorical_distributions = train_info.get("categorical_distributions", {})

    numeric_drift = {}
    for feature in NUMERIC_FEATURES:
        if feature not in scoring_X.columns or feature not in feature_stats:
            continue
        train_stats = feature_stats[feature]
        scoring_mean = float(scoring_X[feature].mean())
        scoring_std = float(scoring_X[feature].std())
        train_mean = train_stats["mean"]
        train_std = train_stats["std"]

        mean_drift_pct = abs(scoring_mean - train_mean) / max(abs(train_mean), 1e-8) * 100
        std_drift_pct = abs(scoring_std - train_std) / max(abs(train_std), 1e-8) * 100

        drifted = mean_drift_pct > 10 or std_drift_pct > 10
        numeric_drift[feature] = {
            "train_mean": train_mean,
            "scoring_mean": scoring_mean,
            "mean_drift_percent": round(mean_drift_pct, 2),
            "train_std": train_std,
            "scoring_std": scoring_std,
            "std_drift_percent": round(std_drift_pct, 2),
            "drifted": drifted,
        }
        if drifted:
            console.log(f"[yellow]Warning: {feature} numeric drift detected (mean={mean_drift_pct:.1f}%, std={std_drift_pct:.1f}%)")

    categorical_drift = {}
    for feature in NOMINAL_FEATURES + ORDINAL_FEATURES:
        if feature not in scoring_X.columns or feature not in categorical_distributions:
            continue
        train_dist = categorical_distributions[feature]
        scoring_dist = scoring_X[feature].value_counts(normalize=True)
        scoring_dist = {str(k): float(v) for k, v in scoring_dist.items()}

        all_categories = set(list(train_dist.keys()) + list(scoring_dist.keys()))
        tvd = 0.5 * sum(
            abs(train_dist.get(cat, 0.0) - scoring_dist.get(cat, 0.0))
            for cat in all_categories
        )

        new_categories = list(set(scoring_dist.keys()) - set(train_dist.keys()))
        missing_categories = list(set(train_dist.keys()) - set(scoring_dist.keys()))

        drifted = tvd > 0.1
        categorical_drift[feature] = {
            "tvd": round(tvd, 4),
            "drifted": drifted,
            "new_categories": new_categories,
            "missing_categories": missing_categories,
        }
        if drifted:
            console.log(f"[yellow]Warning: {feature} categorical drift detected (TVD={tvd:.4f})")

    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "n_scoring_samples": len(scoring_df),
        "n_training_samples": train_info["n_samples"],
        "missing_features": list(missing_features),
        "extra_features": list(extra_features),
        "numeric_drift": numeric_drift,
        "categorical_drift": categorical_drift,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = reports_dir / f"drift_report_{timestamp}.json"

    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    console.log(f"[green]Drift report saved: {report_path}")
    console.rule("[bold cyan]Drift Analysis Summary")
    console.print(f"Scoring samples: {drift_report['n_scoring_samples']}")
    console.print(f"Training samples: {drift_report['n_training_samples']}")
    n_numeric_drifted = sum(1 for v in numeric_drift.values() if v["drifted"])
    n_categorical_drifted = sum(1 for v in categorical_drift.values() if v["drifted"])
    console.print(f"Numeric features with drift: {n_numeric_drifted}/{len(numeric_drift)}")
    console.print(f"Categorical features with drift: {n_categorical_drifted}/{len(categorical_drift)}")

    return drift_report


def main(
    scoring_data_path: Path = Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
    artifacts_dir: Path = Path.cwd() / "artifacts",
    outputs_dir: Path = Path.cwd() / "outputs",
    reports_dir: Path = Path.cwd() / "reports"):

    console.rule("[bold magenta]Telco Churn: Batch Scoring Pipeline")
    latest_run_dir = get_latest_artifact_path(artifacts_dir)
    model, preprocessor, train_info = load_artifacts(latest_run_dir)
    scoring_df = load_scoring_data(scoring_data_path)
    predictions_df = generate_predictions(model, preprocessor, scoring_df)
    parquet_path, csv_path = save_predictions(predictions_df, outputs_dir)
    drift_report = generate_drift_report(scoring_df, train_info, reports_dir)
    console.rule("[bold green]Batch scoring completed!")
    console.rule("[bold cyan]Top 10 Churn Risks")
    if "customerID" in predictions_df.columns:
        top_churners = predictions_df.nlargest(10, "churn_probability")[["customerID", "churn_probability"]]
    else:
        top_churners = predictions_df.nlargest(10, "churn_probability")[["churn_probability"]]
    console.print(top_churners.to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
