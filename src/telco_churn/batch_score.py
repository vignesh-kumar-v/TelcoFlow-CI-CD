import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from src.telco_churn.features import active_features, drop_non_features
from src.telco_churn.inference import LoadedModel, get_latest_artifact_path

console = Console()

MEAN_DRIFT_THRESHOLD = 10.0   # percent change vs training mean
STD_DRIFT_THRESHOLD = 10.0    # percent change vs training std
TVD_THRESHOLD = 0.1           # total variation distance for categoricals


def load_scoring_data(data_path: Path) -> pd.DataFrame:
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"No scoring data at {data_path}. Run `make train` to produce the "
            f"held-out batch, or pass --scoring-data-path."
        )
    console.log(f"[blue]Loading scoring data from {data_path}...")
    df = pd.read_parquet(data_path)
    console.log(f"[green]Loaded {len(df)} rows to score")
    return df


def simulate_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Perturb a batch so the drift detector has something real to catch.

    Shifts the charge/tenure distributions and re-weights the contract mix
    toward month-to-month, which is what a genuine change in customer intake
    would look like.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    console.log("[yellow]Simulating drift on the scoring batch...")

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"] * rng.normal(1.25, 0.05, len(df))
    if "tenure" in df.columns:
        df["tenure"] = (df["tenure"] * 0.6).round().astype(int)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"] * 0.7
    if "Contract" in df.columns:
        flip = rng.random(len(df)) < 0.35
        df.loc[flip, "Contract"] = "Month-to-month"

    return df


def generate_predictions(loaded: LoadedModel, df: pd.DataFrame) -> pd.DataFrame:
    console.log(
        f"[blue]Generating predictions with {loaded.model_name} "
        f"({loaded.model_type})..."
    )
    probabilities = loaded.predict_proba(df)
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

    scoring_X = drop_non_features(scoring_df)
    train_features = train_info["feature_names"]
    missing_features = set(train_features) - set(scoring_X.columns)
    extra_features = set(scoring_X.columns) - set(train_features)

    feature_stats = train_info.get("feature_stats", {})
    categorical_distributions = train_info.get("categorical_distributions", {})
    nominal, ordinal, numeric = active_features(scoring_X)

    numeric_drift = {}
    for feature in numeric:
        if feature not in feature_stats:
            continue
        train_stats = feature_stats[feature]
        scoring_mean = float(scoring_X[feature].mean())
        scoring_std = float(scoring_X[feature].std())
        train_mean = train_stats["mean"]
        train_std = train_stats["std"]

        mean_drift_pct = abs(scoring_mean - train_mean) / max(abs(train_mean), 1e-8) * 100
        std_drift_pct = abs(scoring_std - train_std) / max(abs(train_std), 1e-8) * 100

        drifted = mean_drift_pct > MEAN_DRIFT_THRESHOLD or std_drift_pct > STD_DRIFT_THRESHOLD
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
            console.log(
                f"[yellow]Warning: {feature} numeric drift detected "
                f"(mean={mean_drift_pct:.1f}%, std={std_drift_pct:.1f}%)"
            )

    categorical_drift = {}
    for feature in nominal + ordinal:
        if feature not in categorical_distributions:
            continue
        train_dist = categorical_distributions[feature]
        scoring_dist = scoring_X[feature].value_counts(normalize=True)
        scoring_dist = {str(k): float(v) for k, v in scoring_dist.items()}

        all_categories = set(train_dist) | set(scoring_dist)
        tvd = 0.5 * sum(
            abs(train_dist.get(cat, 0.0) - scoring_dist.get(cat, 0.0))
            for cat in all_categories
        )

        drifted = tvd > TVD_THRESHOLD
        categorical_drift[feature] = {
            "tvd": round(tvd, 4),
            "drifted": drifted,
            "new_categories": sorted(set(scoring_dist) - set(train_dist)),
            "missing_categories": sorted(set(train_dist) - set(scoring_dist)),
        }
        if drifted:
            console.log(f"[yellow]Warning: {feature} categorical drift detected (TVD={tvd:.4f})")

    n_numeric_drifted = sum(1 for v in numeric_drift.values() if v["drifted"])
    n_categorical_drifted = sum(1 for v in categorical_drift.values() if v["drifted"])

    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "n_scoring_samples": len(scoring_df),
        "n_training_samples": train_info["n_samples"],
        "missing_features": sorted(missing_features),
        "extra_features": sorted(extra_features),
        "thresholds": {
            "mean_drift_percent": MEAN_DRIFT_THRESHOLD,
            "std_drift_percent": STD_DRIFT_THRESHOLD,
            "tvd": TVD_THRESHOLD,
        },
        "drift_detected": bool(n_numeric_drifted or n_categorical_drifted),
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
    console.print(f"Numeric features with drift: {n_numeric_drifted}/{len(numeric_drift)}")
    console.print(f"Categorical features with drift: {n_categorical_drifted}/{len(categorical_drift)}")

    return drift_report


def main(
    scoring_data_path: Path = Path.cwd() / "data" / "processed" / "scoring_batch.parquet",
    artifacts_dir: Path = Path.cwd() / "artifacts",
    outputs_dir: Path = Path.cwd() / "outputs",
    reports_dir: Path = Path.cwd() / "reports",
    simulate_drift_flag: bool = typer.Option(
        False, "--simulate-drift", help="Perturb the batch to demonstrate drift detection"
    ),
):
    console.rule("[bold magenta]Telco Churn: Batch Scoring Pipeline")

    latest_run_dir = get_latest_artifact_path(artifacts_dir)
    console.log(f"[blue]Using latest artifact: {latest_run_dir.name}")
    loaded = LoadedModel.load(latest_run_dir)

    scoring_df = load_scoring_data(scoring_data_path)
    if simulate_drift_flag:
        scoring_df = simulate_drift(scoring_df)

    predictions_df = generate_predictions(loaded, scoring_df)
    save_predictions(predictions_df, outputs_dir)
    generate_drift_report(scoring_df, loaded.train_info, reports_dir)

    console.rule("[bold green]Batch scoring completed!")
    console.rule("[bold cyan]Top 10 Churn Risks")
    cols = ["customerID", "churn_probability"] if "customerID" in predictions_df.columns \
        else ["churn_probability"]
    console.print(predictions_df.nlargest(10, "churn_probability")[cols].to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
