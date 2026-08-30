"""Integration tests over the artifacts a real pipeline run leaves behind.

These assert against files on disk, so they skip (rather than fail) on a cold
checkout. Run `make pipeline` first to exercise them for real.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.telco_churn.features import (
    ENGINEERED_FEATURES, MODEL_TYPE_BY_NAME, drop_non_features,
)
from src.telco_churn.inference import LoadedModel

pytestmark = pytest.mark.integration

RUN_HINT = "no completed pipeline run — run `make pipeline` first"


def _latest_run():
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        pytest.skip(RUN_HINT)
    runs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not runs:
        pytest.skip(RUN_HINT)
    return sorted(runs, key=lambda x: x.name, reverse=True)[0]


@pytest.fixture(scope="module")
def latest_run():
    return _latest_run()


@pytest.fixture(scope="module")
def loaded(latest_run):
    return LoadedModel.load(latest_run)


def test_imports():
    from src.telco_churn import api, batch_score, train, validate_and_clean  # noqa: F401


def test_data_exists():
    raw_path = Path("data/raw/Telco-Customer-Churn.csv")
    assert raw_path.exists(), f"Raw data not found at {raw_path}"
    assert raw_path.stat().st_size > 0, "Raw data file is empty"


def test_cleaned_data_exists():
    clean_path = Path("data/processed/cleaned_data.parquet")
    if not clean_path.exists():
        pytest.skip(RUN_HINT)
    df = pd.read_parquet(clean_path)
    assert len(df) > 0, "Cleaned data is empty"
    assert "Churn" in df.columns, "Target column 'Churn' not found"
    assert df["Churn"].isin([0, 1]).all(), "Churn should be encoded to 0/1"


def test_artifacts_exist(latest_run):
    for file in ("model.joblib", "preprocessor.joblib", "train_info.json", "metrics.json"):
        assert (latest_run / file).exists(), f"Missing artifact: {latest_run / file}"


def test_train_info_records_model_type(latest_run):
    """Serving needs to know which feature shape the deployed model expects."""
    with open(latest_run / "train_info.json") as f:
        train_info = json.load(f)
    assert train_info["model_name"] in MODEL_TYPE_BY_NAME
    assert train_info["model_type"] == MODEL_TYPE_BY_NAME[train_info["model_name"]]


def test_deployed_model_is_the_best_scoring_one(latest_run):
    """The saved model must be the comparison winner, not a filtered subset's."""
    with open(latest_run / "model_comparison.json") as f:
        comparison = json.load(f)
    with open(latest_run / "train_info.json") as f:
        deployed = json.load(f)["model_name"]
    best = max(comparison, key=lambda k: comparison[k]["roc_auc"])
    assert deployed == best, (
        f"Deployed {deployed} but {best} scored higher "
        f"({comparison[best]['roc_auc']:.4f} vs {comparison[deployed]['roc_auc']:.4f})"
    )


def test_model_round_trips_and_predicts(loaded):
    """The saved model + pipeline must actually score raw data together."""
    clean_path = Path("data/processed/cleaned_data.parquet")
    if not clean_path.exists():
        pytest.skip(RUN_HINT)
    sample = pd.read_parquet(clean_path).head(20)
    probabilities = loaded.predict_proba(sample)
    assert len(probabilities) == len(sample)
    assert ((probabilities >= 0) & (probabilities <= 1)).all(), "probabilities out of range"


def test_every_model_shape_is_servable(loaded):
    """The saved pipeline must be able to feed whichever model won.

    The original bug was that only LightGBM-shaped features were reproducible
    at serving time, so an XGBoost or Logistic winner could not be deployed.
    This asserts all three shapes work from the real fitted artifact.
    """
    from src.telco_churn.features import LINEAR, TREE_CODES, TREE_NATIVE

    clean_path = Path("data/processed/cleaned_data.parquet")
    if not clean_path.exists():
        pytest.skip(RUN_HINT)
    sample = drop_non_features(pd.read_parquet(clean_path).head(50))

    native = loaded.pipeline.transform_for(sample, TREE_NATIVE)
    codes = loaded.pipeline.transform_for(sample, TREE_CODES)
    linear = loaded.pipeline.transform_for(sample, LINEAR)

    assert len(native) == len(codes) == linear.shape[0] == 50
    assert isinstance(native["gender"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_integer_dtype(codes["gender"])
    assert linear.shape[1] > sample.shape[1]


def test_api_payload_reproduces_training_features(loaded):
    """A raw API-shaped record must yield every feature the model was trained on.

    Guards the training/serving boundary: training features come from SQL, but a
    single prediction request never touches the warehouse.
    """
    clean_path = Path("data/processed/cleaned_data.parquet")
    if not clean_path.exists():
        pytest.skip(RUN_HINT)

    from src.telco_churn.features import ensure_engineered_features

    # Only the fields the API accepts — engineered columns deliberately dropped.
    raw_only = pd.read_parquet(clean_path).head(5).drop(
        columns=ENGINEERED_FEATURES, errors="ignore"
    )
    rebuilt = ensure_engineered_features(raw_only, loaded.train_info["feature_names"])
    for feature in loaded.train_info["feature_names"]:
        assert feature in rebuilt.columns, f"serving cannot produce {feature}"

    probabilities = loaded.predict_proba(raw_only)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_scoring_batch_is_held_out(loaded):
    """Batch scoring must run on data the model never trained on."""
    scoring_path = Path("data/processed/scoring_batch.parquet")
    if not scoring_path.exists():
        pytest.skip(RUN_HINT)
    scoring = pd.read_parquet(scoring_path)
    assert len(scoring) > 0
    assert len(scoring) < loaded.train_info["n_samples"], \
        "scoring batch should be the held-out split, not the training set"


def test_outputs_exist():
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        pytest.skip(RUN_HINT)
    parquet_files = list(outputs_dir.glob("predictions_*.parquet"))
    csv_files = list(outputs_dir.glob("predictions_*.csv"))
    assert parquet_files, "No Parquet prediction files found"
    assert csv_files, "No CSV prediction files found"

    latest = sorted(parquet_files, key=lambda x: x.name, reverse=True)[0]
    df_pred = pd.read_parquet(latest)
    assert len(df_pred) > 0, "Predictions file is empty"
    assert "churn_probability" in df_pred.columns
    assert "prediction" in df_pred.columns
    assert df_pred["churn_probability"].between(0, 1).all()
    assert df_pred["prediction"].isin([0, 1]).all()


def test_model_comparison_exists(latest_run):
    with open(latest_run / "model_comparison.json") as f:
        comparison = json.load(f)
    assert len(comparison) >= 3, "should compare at least 3 models"
    for model_name, metrics in comparison.items():
        assert "roc_auc" in metrics, f"{model_name} missing roc_auc"
        assert 0 <= metrics["roc_auc"] <= 1


def test_shap_plot_exists(latest_run):
    shap_path = latest_run / "shap_summary.png"
    assert shap_path.exists(), f"shap_summary.png not found in {latest_run}"
    assert shap_path.stat().st_size > 0, "shap_summary.png is empty"


def test_shap_importance_recorded(latest_run):
    with open(latest_run / "metrics.json") as f:
        metrics = json.load(f)
    importance = metrics.get("shap_feature_importance", {})
    assert importance, "metrics.json missing shap_feature_importance"
    assert all(v >= 0 for v in importance.values()), "SHAP importance must be non-negative"


def test_train_info_has_feature_stats(latest_run):
    with open(latest_run / "train_info.json") as f:
        train_info = json.load(f)
    assert train_info["feature_stats"], "feature_stats is empty"
    assert train_info["categorical_distributions"], "categorical_distributions is empty"
    assert "customerID" not in train_info["feature_names"]
    assert "Churn" not in train_info["feature_names"]


def test_drift_report_has_numeric_and_categorical():
    reports_dir = Path("reports")
    if not reports_dir.exists():
        pytest.skip(RUN_HINT)
    report_files = list(reports_dir.glob("drift_report_*.json"))
    if not report_files:
        pytest.skip(RUN_HINT)
    latest_report = sorted(report_files, key=lambda x: x.name, reverse=True)[0]
    with open(latest_report) as f:
        report = json.load(f)
    assert report["numeric_drift"], "numeric_drift is empty"
    assert report["categorical_drift"], "categorical_drift is empty"
    assert "drift_detected" in report
    assert "thresholds" in report
