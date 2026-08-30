"""Single prediction path shared by batch scoring and the API.

Training records which model won and therefore which feature shape it expects,
so both serving entry points transform features the same way rather than trying
one shape and falling back to another on exception.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.telco_churn.features import (
    FeaturePipeline, drop_non_features, ensure_engineered_features,
)

MODEL_FILE = "model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"
TRAIN_INFO_FILE = "train_info.json"
METRICS_FILE = "metrics.json"


def get_latest_artifact_path(artifacts_dir: Path) -> Path:
    """Newest training run. Run dirs are timestamps, so lexical sort is chronological."""
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(
            f"No artifacts directory at {artifacts_dir}. Run `make train` first."
        )
    runs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not runs:
        raise FileNotFoundError(
            f"No artifact runs found in {artifacts_dir}. Run `make train` first."
        )
    return sorted(runs, key=lambda x: x.name, reverse=True)[0]


class LoadedModel:
    """A deployed model plus everything needed to feed it correctly."""

    def __init__(self, model, pipeline: FeaturePipeline, train_info: dict, run_dir: Path):
        self.model = model
        self.pipeline = pipeline
        self.train_info = train_info
        self.run_dir = run_dir

    @property
    def model_type(self) -> str:
        return self.train_info["model_type"]

    @property
    def model_name(self) -> str:
        return self.train_info.get("model_name", "unknown")

    @property
    def version(self) -> str:
        return self.run_dir.name

    @classmethod
    def load(cls, run_dir: Path) -> "LoadedModel":
        run_dir = Path(run_dir)
        model = joblib.load(run_dir / MODEL_FILE)
        pipeline = joblib.load(run_dir / PREPROCESSOR_FILE)
        with open(run_dir / TRAIN_INFO_FILE) as f:
            train_info = json.load(f)
        return cls(model, pipeline, train_info, run_dir)

    @classmethod
    def load_latest(cls, artifacts_dir: Path) -> "LoadedModel":
        return cls.load(get_latest_artifact_path(artifacts_dir))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Churn probability for each row of a raw frame.

        Derived columns are recomputed when the model expects them but the
        caller supplied only raw fields — which is exactly the API's case.
        """
        X = ensure_engineered_features(df, self.train_info.get("feature_names", []))
        X = self.pipeline.transform_for(drop_non_features(X), self.model_type)
        return self.model.predict_proba(X)[:, 1]

    def metrics(self) -> dict:
        path = self.run_dir / METRICS_FILE
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)


def risk_category(probability: float, high: float = 0.6, medium: float = 0.4) -> str:
    if probability > high:
        return "High"
    if probability > medium:
        return "Medium"
    return "Low"
