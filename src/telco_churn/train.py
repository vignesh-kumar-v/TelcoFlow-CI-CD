import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
import typer
from rich.console import Console
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import lightgbm
from sklearn.metrics import classification_report, roc_auc_score

console = Console()

def load_clean_data(clean_path: Path):
    console.log(f"[blue]Loading cleaned data from {clean_path}...")
    df = pd.read_parquet(clean_path)
    console.log(f"[green]Loaded {len(df)} rows")
    return df

def split_data(df: pd.DataFrame, target_col: str="Churn", test_size: float=0.2, val_size: float=0.2, random_state: int=42):
    console.log(f"[blue]Splitting data (test_size={test_size}, val_size={val_size})...")
    df_train_val, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    df_train, df_val = train_test_split(df_train_val, test_size=val_size/(1-test_size), random_state=random_state, stratify=df_train_val[target_col])
    console.log(f"[green]Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    return df_train, df_val, df_test

def create_preprocessor(categorical_features: list):
    console.log(f"[blue]Creating preprocessor for {len(categorical_features)} categorical features...")
    preprocessor = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    return preprocessor

def train_model(X_train, y_train, X_val, y_val, categorical_features: list):
    console.log(f"[blue]Training LightGBM model...")
    model = lightgbm.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        early_stopping_round=50,
        verbose=-1,
        categorical_feature=categorical_features
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
    )
    console.log(f"[green]Model trained with {model.best_iteration_} iterations")
    return model

def evaluate_model(model, X_test, y_test):
    console.log(f"[blue]Evaluating on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    aus_score = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "roc_auc": aus_score,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "support": report["1"]["support"]
    }
    console.log(f"[green]Test ROC-AUC: {aus_score:.4f}")
    return metrics

def save_artifacts(model, preprocessor, metrics: dict, df_train: pd.DataFrame, artifacts_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"[blue]Saving artifacts to {run_dir}...")
    joblib.dump(model, run_dir / "model.joblib")
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    train_info = {
        "n_samples": len(df_train),
        "n_features": len(df_train.columns) - 1,
        "feature_names": df_train.drop(columns=["Churn"]).columns.tolist(),
        "train_date": timestamp
    }
    with open(run_dir / "train_info.json", "w") as f:
        json.dump(train_info, f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    console.log(f"[green]Artifacts saved!")

def main(clean_path: Path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "cleaned_data.parquet",
    artifacts_dir: Path = Path(__file__).resolve().parent / "artifacts",
    random_state: int = 42):
    console.rule("[bold magenta]Telco Churn: Training Pipeline")
    df = load_clean_data(clean_path)
    df_train, df_val, df_test = split_data(df, random_state=random_state)
    X_train = df_train.drop(columns=["Churn"])
    y_train = df_train["Churn"]
    X_val = df_val.drop(columns=["Churn"])
    y_val = df_val["Churn"]
    X_test = df_test.drop(columns=["Churn"])
    y_test = df_test["Churn"]

    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    console.log(f"[blue]Categorical features: {categorical_features}")
    preprocessor = create_preprocessor(categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    model = train_model(
        X_train_processed, y_train,
        X_val_processed, y_val,
        categorical_features=list(range(len(categorical_features)))
    )

    metrics = evaluate_model(model, X_test_processed, y_test)
    save_artifacts(model, preprocessor, metrics, df_train, artifacts_dir)
    console.rule("[bold]Training completed successfully!")

if __name__ == "__main__":
    typer.run(main)