import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import typer
import xgboost as xgb
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.telco_churn.features import (  # noqa: E402
    CONTRACT_ORDER, DROP_COLS, LINEAR, MODEL_TYPE_BY_NAME, NOMINAL_FEATURES,
    NUMERIC_FEATURES, ORDINAL_FEATURES, TARGET, TREE_CODES, TREE_NATIVE,
    FeaturePipeline, active_features, create_lgbm_preprocessor,
    create_lr_preprocessor, drop_non_features, prepare_data_lgbm, prepare_data_lr,
)
from src.telco_churn.tracking import ExperimentTracker  # noqa: E402

console = Console()

# Re-exported so `from src.telco_churn.train import NOMINAL_FEATURES` keeps working
# for existing callers and tests; the definitions now live in features.py.
__all__ = [
    "DROP_COLS", "NOMINAL_FEATURES", "ORDINAL_FEATURES", "NUMERIC_FEATURES",
    "CONTRACT_ORDER", "create_lgbm_preprocessor", "create_lr_preprocessor",
    "prepare_data_lgbm", "prepare_data_lr", "compute_scale_pos_weight",
]


def load_clean_data(clean_path: Path):
    console.log(f"[blue]Loading cleaned data from {clean_path}...")
    df = pd.read_parquet(clean_path)
    console.log(f"[green]Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def split_data(df: pd.DataFrame, target_col: str = TARGET, test_size: float = 0.2,
               val_size: float = 0.2, random_state: int = 42):
    console.log(f"[blue]Splitting data (test_size={test_size}, val_size={val_size})...")
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=df_train_val[target_col]
    )
    console.log(f"[green]Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    return df_train, df_val, df_test


def compute_scale_pos_weight(y):
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos


def evaluate_model(model, X_test, y_test, model_name: str = "model"):
    console.log(f"[blue]Evaluating {model_name} on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "roc_auc": auc_score,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "support": report["1"]["support"],
    }
    console.log(f"[green]{model_name} Test ROC-AUC: {auc_score:.4f}")
    return metrics


def train_logistic_regression(X_train, y_train, X_val, y_val):
    console.log("[blue]Training Logistic Regression...")
    # lbfgs needs the one-hot matrix scaled to converge; without it the solver
    # hits max_iter and warns on every run.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)
    return model


def train_lgbm(X_train, y_train, X_val, y_val, cat_indices, scale_pos_weight, params=None):
    console.log("[blue]Training LightGBM...")
    default_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
    )
    if params:
        default_params.update(params)
    model = lightgbm.LGBMClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lightgbm.early_stopping(50, verbose=False), lightgbm.log_evaluation(period=0)],
        categorical_feature=cat_indices,
    )
    console.log(f"[green]LightGBM trained with {model.best_iteration_} iterations")
    return model


def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    console.log("[blue]Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=50,
        enable_categorical=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    console.log(f"[green]XGBoost trained with {model.best_iteration} iterations")
    return model


def tune_lgbm_optuna(X_train, y_train, X_val, y_val, cat_indices, scale_pos_weight,
                     n_trials=50, tracker: "ExperimentTracker | None" = None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    console.log(f"[blue]Running Optuna hyperparameter tuning ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = train_lgbm(X_train, y_train, X_val, y_val, cat_indices,
                           scale_pos_weight, params=params)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if tracker is not None:
            tracker.log_trial(trial.number, params, val_auc)
        return val_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    console.log(f"[green]Best trial AUC: {study.best_value:.4f}")
    console.log(f"[green]Best params: {study.best_params}")
    return study.best_params, study


def _coerce_binary_shap(shap_values):
    """SHAP returns per-class values in several shapes across versions/models."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


def generate_shap_analysis(model, X, run_dir: Path, model_type: str,
                           feature_names=None):
    """Feature importance for the deployed model, whatever type it is."""
    import shap

    console.log("[blue]Generating SHAP feature importance...")

    if model_type == LINEAR:
        # `model` is a StandardScaler -> LogisticRegression pipeline; explain the
        # classifier over the scaled matrix it actually sees.
        scaler, classifier = model[0], model[-1]
        X_scaled = scaler.transform(X)
        explainer = shap.LinearExplainer(classifier, X_scaled)
        shap_values = _coerce_binary_shap(explainer.shap_values(X_scaled))
        X_display = pd.DataFrame(X_scaled, columns=feature_names)
    else:
        explainer = shap.TreeExplainer(model)
        X_display = X.copy()
        for col in X_display.columns:
            if isinstance(X_display[col].dtype, pd.CategoricalDtype):
                X_display[col] = X_display[col].cat.codes
        source = X if model_type == TREE_NATIVE else X_display
        shap_values = _coerce_binary_shap(
            explainer.shap_values(source, check_additivity=False)
        )

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, show=False)
    plt.tight_layout()
    plt.savefig(run_dir / "shap_summary.png", dpi=150)
    plt.close("all")
    console.log("[green]SHAP summary plot saved")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(X_display.columns.tolist(), mean_abs_shap.tolist()))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def compute_feature_stats(X: pd.DataFrame):
    _, _, numeric = active_features(X)
    return {
        col: {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "min": float(X[col].min()),
            "max": float(X[col].max()),
        }
        for col in numeric
    }


def compute_categorical_distributions(X: pd.DataFrame):
    nominal, ordinal, _ = active_features(X)
    distributions = {}
    for col in nominal + ordinal:
        dist = X[col].value_counts(normalize=True)
        distributions[col] = {str(k): float(v) for k, v in dist.items()}
    return distributions


def save_artifacts(model, pipeline: FeaturePipeline, metrics: dict, df_train: pd.DataFrame,
                   artifacts_dir: Path, model_name: str, model_type: str,
                   model_comparison: dict = None, shap_importance: dict = None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"[blue]Saving artifacts to {run_dir}...")

    joblib.dump(model, run_dir / "model.joblib")
    joblib.dump(pipeline, run_dir / "preprocessor.joblib")

    X_train = drop_non_features(df_train)
    nominal, ordinal, numeric = active_features(X_train)

    train_info = {
        "n_samples": len(df_train),
        "n_features": len(X_train.columns),
        "feature_names": X_train.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": nominal + ordinal,
        "train_date": timestamp,
        # Recorded so serving transforms features the way this model was trained.
        "model_name": model_name,
        "model_type": model_type,
        "feature_stats": compute_feature_stats(X_train),
        "categorical_distributions": compute_categorical_distributions(X_train),
    }
    with open(run_dir / "train_info.json", "w") as f:
        json.dump(train_info, f, indent=2)

    if shap_importance:
        metrics = {**metrics, "shap_feature_importance": shap_importance}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if model_comparison:
        with open(run_dir / "model_comparison.json", "w") as f:
            json.dump(model_comparison, f, indent=2)

    console.log("[green]Artifacts saved!")
    return run_dir


def save_scoring_batch(df_test: pd.DataFrame, scoring_path: Path):
    """Persist the held-out test split as the batch-scoring input.

    Scoring the training data would make drift detection meaningless, since the
    baseline in train_info.json is computed from that same data.
    """
    scoring_path.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_parquet(scoring_path, index=False)
    console.log(f"[green]Saved {len(df_test)} held-out rows to {scoring_path}")


def main(
    clean_path: Path = Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
    artifacts_dir: Path = Path.cwd() / "artifacts",
    scoring_path: Path = Path.cwd() / "data" / "processed" / "scoring_batch.parquet",
    random_state: int = 42,
    skip_tuning: bool = typer.Option(False, "--skip-tuning", help="Skip Optuna hyperparameter tuning"),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials"),
    track: bool = typer.Option(True, "--track/--no-track", help="Log the run to MLflow"),
    register: bool = typer.Option(True, "--register/--no-register", help="Register the winner in the MLflow registry"),
):
    console.rule("[bold magenta]Telco Churn: Training Pipeline")

    df = load_clean_data(clean_path)
    df_train, df_val, df_test = split_data(df, random_state=random_state)

    X_train_raw = drop_non_features(df_train)
    X_val_raw = drop_non_features(df_val)
    X_test_raw = drop_non_features(df_test)
    y_train, y_val, y_test = df_train[TARGET], df_val[TARGET], df_test[TARGET]

    spw = compute_scale_pos_weight(y_train)
    console.log(f"[blue]scale_pos_weight: {spw:.2f}")

    # One fitted pipeline serves every model shape.
    pipeline = FeaturePipeline().fit(X_train_raw)
    cat_indices = pipeline.cat_indices

    X_train_tree = pipeline.transform_tree_native(X_train_raw)
    X_val_tree = pipeline.transform_tree_native(X_val_raw)
    X_test_tree = pipeline.transform_tree_native(X_test_raw)

    X_train_codes = pipeline.transform_tree_codes(X_train_raw)
    X_val_codes = pipeline.transform_tree_codes(X_val_raw)
    X_test_codes = pipeline.transform_tree_codes(X_test_raw)

    X_train_lin = pipeline.transform_linear(X_train_raw)
    X_val_lin = pipeline.transform_linear(X_val_raw)
    X_test_lin = pipeline.transform_linear(X_test_raw)

    tracker = ExperimentTracker(enabled=track)
    tracker.start_run(
        params={
            "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
            "scale_pos_weight": round(float(spw), 4),
            "random_state": random_state,
            "tuning": "optuna" if not skip_tuning else "none",
            "n_trials": 0 if skip_tuning else n_trials,
            "n_features": X_train_raw.shape[1],
        }
    )

    # Each entry: (model, metrics, test matrix used for SHAP, feature names)
    candidates = {}

    lr_model = train_logistic_regression(X_train_lin, y_train, X_val_lin, y_val)
    lr_metrics = evaluate_model(lr_model, X_test_lin, y_test, "Logistic Regression")
    candidates["logistic_regression"] = (
        lr_model, lr_metrics, X_test_lin,
        pipeline.lr_transformer.get_feature_names_out().tolist(),
    )

    lgbm_model = train_lgbm(X_train_tree, y_train, X_val_tree, y_val, cat_indices, spw)
    lgbm_metrics = evaluate_model(lgbm_model, X_test_tree, y_test, "LightGBM")
    candidates["lightgbm"] = (lgbm_model, lgbm_metrics, X_test_tree, None)

    xgb_model = train_xgboost(X_train_codes, y_train, X_val_codes, y_val, spw)
    xgb_metrics = evaluate_model(xgb_model, X_test_codes, y_test, "XGBoost")
    candidates["xgboost"] = (xgb_model, xgb_metrics, X_test_codes, None)

    if not skip_tuning:
        best_params, study = tune_lgbm_optuna(
            X_train_tree, y_train, X_val_tree, y_val, cat_indices, spw,
            n_trials=n_trials, tracker=tracker,
        )
        tuned_model = train_lgbm(X_train_tree, y_train, X_val_tree, y_val,
                                 cat_indices, spw, params=best_params)
        tuned_metrics = evaluate_model(tuned_model, X_test_tree, y_test, "Tuned LightGBM")
        candidates["tuned_lightgbm"] = (tuned_model, tuned_metrics, X_test_tree, None)
        tracker.log_params({f"best_{k}": v for k, v in best_params.items()})
    else:
        console.log("[yellow]Skipping Optuna tuning (--skip-tuning)")

    model_comparison = {name: m for name, (_, m, _, _) in candidates.items()}
    tracker.log_model_comparison(model_comparison)

    # Every candidate is deployable: the feature pipeline can reproduce whichever
    # shape the winner needs, so selection is purely on ROC-AUC.
    best_name = max(candidates, key=lambda k: candidates[k][1]["roc_auc"])
    best_model, best_metrics, best_X_test, best_feature_names = candidates[best_name]
    best_type = MODEL_TYPE_BY_NAME[best_name]
    console.log(
        f"[bold green]Best model: {best_name} "
        f"(AUC={best_metrics['roc_auc']:.4f}, type={best_type}) — deploying this one"
    )

    run_dir = save_artifacts(
        best_model, pipeline, best_metrics, df_train, artifacts_dir,
        model_name=best_name, model_type=best_type,
        model_comparison=model_comparison,
    )

    shap_importance = generate_shap_analysis(
        best_model, best_X_test, run_dir, best_type, feature_names=best_feature_names
    )
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        metrics_out = json.load(f)
    metrics_out["shap_feature_importance"] = shap_importance
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    save_scoring_batch(df_test, scoring_path)

    tracker.log_best_model(
        model=best_model, model_name=best_name, model_type=best_type,
        metrics=best_metrics, run_dir=run_dir, register=register,
    )
    tracker.end_run()

    console.rule("[bold]Training completed successfully!")


if __name__ == "__main__":
    typer.run(main)
