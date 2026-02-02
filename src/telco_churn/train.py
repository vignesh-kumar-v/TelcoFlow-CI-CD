import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import typer
from rich.console import Console
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

console = Console()

DROP_COLS = ["customerID"]

NOMINAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "PaymentMethod",
]

ORDINAL_FEATURES = ["Contract"]
CONTRACT_ORDER = [["Month-to-month", "One year", "Two year"]]

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_clean_data(clean_path: Path):
    console.log(f"[blue]Loading cleaned data from {clean_path}...")
    df = pd.read_parquet(clean_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    console.log(f"[green]Loaded {len(df)} rows (dropped {DROP_COLS})")
    return df


def split_data(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2,
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


def create_lgbm_preprocessor():
    encoder = OrdinalEncoder(
        categories=CONTRACT_ORDER,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    return encoder


def create_lr_preprocessor():
    transformer = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_FEATURES),
            ("ordinal", OrdinalEncoder(categories=CONTRACT_ORDER, handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return transformer


def prepare_data_lgbm(X: pd.DataFrame, encoder, fit: bool = False):
    X = X.copy()
    if fit:
        X[ORDINAL_FEATURES] = encoder.fit_transform(X[ORDINAL_FEATURES])
    else:
        X[ORDINAL_FEATURES] = encoder.transform(X[ORDINAL_FEATURES])
    for col in NOMINAL_FEATURES:
        X[col] = X[col].astype("category")
    cat_indices = [X.columns.get_loc(c) for c in NOMINAL_FEATURES + ORDINAL_FEATURES]
    return X, cat_indices


def prepare_data_lr(X: pd.DataFrame, transformer, fit: bool = False):
    if fit:
        return transformer.fit_transform(X)
    return transformer.transform(X)


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
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
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
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    for col in NOMINAL_FEATURES:
        if hasattr(X_train_xgb[col], "cat"):
            X_train_xgb[col] = X_train_xgb[col].cat.codes
            X_val_xgb[col] = X_val_xgb[col].cat.codes
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
    model.fit(
        X_train_xgb, y_train,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False,
    )
    console.log(f"[green]XGBoost trained with {model.best_iteration} iterations")
    return model


def tune_lgbm_optuna(X_train, y_train, X_val, y_val, cat_indices, scale_pos_weight, n_trials=50):
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
        model = train_lgbm(X_train, y_train, X_val, y_val, cat_indices, scale_pos_weight, params=params)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    console.log(f"[green]Best trial AUC: {study.best_value:.4f}")
    console.log(f"[green]Best params: {study.best_params}")
    return study.best_params


def generate_shap_analysis(model, X_test, run_dir: Path, is_lgbm: bool = False):
    import shap
    console.log("[blue]Generating SHAP feature importance...")
    X_shap = X_test.copy()
    X_display = X_test.copy()
    for col in X_display.columns:
        if hasattr(X_display[col], "cat"):
            X_display[col] = X_display[col].cat.codes
    if is_lgbm:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap, check_additivity=False)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_display, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, show=False)
    plt.tight_layout()
    plt.savefig(run_dir / "shap_summary.png", dpi=150)
    plt.close("all")
    console.log(f"[green]SHAP summary plot saved")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(X_display.columns.tolist(), mean_abs_shap.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance


def compute_feature_stats(X: pd.DataFrame):
    stats = {}
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            stats[col] = {
                "mean": float(X[col].mean()),
                "std": float(X[col].std()),
                "min": float(X[col].min()),
                "max": float(X[col].max()),
            }
    return stats


def compute_categorical_distributions(X: pd.DataFrame):
    distributions = {}
    for col in NOMINAL_FEATURES + ORDINAL_FEATURES:
        if col in X.columns:
            dist = X[col].value_counts(normalize=True)
            distributions[col] = {str(k): float(v) for k, v in dist.items()}
    return distributions


def save_artifacts(model, preprocessor, metrics: dict, df_train: pd.DataFrame,
                   artifacts_dir: Path, model_comparison: dict = None,
                   shap_importance: dict = None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"[blue]Saving artifacts to {run_dir}...")

    joblib.dump(model, run_dir / "model.joblib")
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")

    X_train = df_train.drop(columns=["Churn"], errors="ignore")
    feature_stats = compute_feature_stats(X_train)
    categorical_distributions = compute_categorical_distributions(X_train)

    train_info = {
        "n_samples": len(df_train),
        "n_features": len(X_train.columns),
        "feature_names": X_train.columns.tolist(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": NOMINAL_FEATURES + ORDINAL_FEATURES,
        "train_date": timestamp,
        "feature_stats": feature_stats,
        "categorical_distributions": categorical_distributions,
    }
    with open(run_dir / "train_info.json", "w") as f:
        json.dump(train_info, f, indent=2)

    if shap_importance:
        metrics["shap_feature_importance"] = shap_importance
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if model_comparison:
        with open(run_dir / "model_comparison.json", "w") as f:
            json.dump(model_comparison, f, indent=2)

    console.log(f"[green]Artifacts saved!")
    return run_dir


def main(
    clean_path: Path = Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
    artifacts_dir: Path = Path.cwd() / "artifacts",
    random_state: int = 42,
    skip_tuning: bool = typer.Option(False, "--skip-tuning", help="Skip Optuna hyperparameter tuning"),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials"),
):
    console.rule("[bold magenta]Telco Churn: Training Pipeline")

    df = load_clean_data(clean_path)
    df_train, df_val, df_test = split_data(df, random_state=random_state)

    X_train = df_train.drop(columns=["Churn"])
    y_train = df_train["Churn"]
    X_val = df_val.drop(columns=["Churn"])
    y_val = df_val["Churn"]
    X_test = df_test.drop(columns=["Churn"])
    y_test = df_test["Churn"]

    spw = compute_scale_pos_weight(y_train)
    console.log(f"[blue]scale_pos_weight: {spw:.2f}")

    lgbm_encoder = create_lgbm_preprocessor()
    X_train_lgbm, cat_indices = prepare_data_lgbm(X_train, lgbm_encoder, fit=True)
    X_val_lgbm, _ = prepare_data_lgbm(X_val, lgbm_encoder)
    X_test_lgbm, _ = prepare_data_lgbm(X_test, lgbm_encoder)

    lr_transformer = create_lr_preprocessor()
    X_train_lr = prepare_data_lr(X_train, lr_transformer, fit=True)
    X_val_lr = prepare_data_lr(X_val, lr_transformer)
    X_test_lr = prepare_data_lr(X_test, lr_transformer)

    model_comparison = {}

    lr_model = train_logistic_regression(X_train_lr, y_train, X_val_lr, y_val)
    lr_metrics = evaluate_model(lr_model, X_test_lr, y_test, "Logistic Regression")
    model_comparison["logistic_regression"] = lr_metrics

    lgbm_model = train_lgbm(X_train_lgbm, y_train, X_val_lgbm, y_val, cat_indices, spw)
    lgbm_metrics = evaluate_model(lgbm_model, X_test_lgbm, y_test, "LightGBM")
    model_comparison["lightgbm"] = lgbm_metrics

    xgb_model = train_xgboost(X_train_lgbm, y_train, X_val_lgbm, y_val, spw)
    X_test_xgb = X_test_lgbm.copy()
    for col in NOMINAL_FEATURES:
        if hasattr(X_test_xgb[col], "cat"):
            X_test_xgb[col] = X_test_xgb[col].cat.codes
    xgb_metrics = evaluate_model(xgb_model, X_test_xgb, y_test, "XGBoost")
    model_comparison["xgboost"] = xgb_metrics

    if not skip_tuning:
        best_params = tune_lgbm_optuna(X_train_lgbm, y_train, X_val_lgbm, y_val, cat_indices, spw, n_trials=n_trials)
        tuned_model = train_lgbm(X_train_lgbm, y_train, X_val_lgbm, y_val, cat_indices, spw, params=best_params)
        tuned_metrics = evaluate_model(tuned_model, X_test_lgbm, y_test, "Tuned LightGBM")
        model_comparison["tuned_lightgbm"] = tuned_metrics
    else:
        console.log("[yellow]Skipping Optuna tuning (--skip-tuning)")
        tuned_model = None
        tuned_metrics = None

    all_candidates = {
        "logistic_regression": (lr_model, lr_metrics),
        "lightgbm": (lgbm_model, lgbm_metrics),
        "xgboost": (xgb_model, xgb_metrics),
    }
    if tuned_model is not None:
        all_candidates["tuned_lightgbm"] = (tuned_model, tuned_metrics)

    overall_best = max(all_candidates, key=lambda k: all_candidates[k][1]["roc_auc"])
    console.log(f"[blue]Overall best model: {overall_best} (AUC={all_candidates[overall_best][1]['roc_auc']:.4f})")

    tree_candidates = {k: v for k, v in all_candidates.items() if k in ("lightgbm", "tuned_lightgbm", "xgboost")}
    best_name = max(tree_candidates, key=lambda k: tree_candidates[k][1]["roc_auc"])
    best_model, best_metrics = tree_candidates[best_name]
    console.log(f"[bold green]Deployed model: {best_name} (AUC={best_metrics['roc_auc']:.4f})")

    run_dir = save_artifacts(best_model, lgbm_encoder, best_metrics, df_train, artifacts_dir,
                             model_comparison=model_comparison, shap_importance=None)

    shap_X = X_test_xgb if best_name == "xgboost" else X_test_lgbm
    is_lgbm = best_name in ("lightgbm", "tuned_lightgbm")
    shap_importance = generate_shap_analysis(best_model, shap_X, run_dir, is_lgbm=is_lgbm)
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        metrics_out = json.load(f)
    metrics_out["shap_feature_importance"] = shap_importance
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    console.rule("[bold]Training completed successfully!")


if __name__ == "__main__":
    typer.run(main)
