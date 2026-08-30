"""MLflow experiment tracking and model registry.

Every training run logs its parameters, per-model metrics, Optuna trials and
artifacts; the winning model is then registered as a new version in the MLflow
Model Registry so promotion is a registry action rather than someone reading
timestamps out of `artifacts/`.

Tracking is best-effort by design. If MLflow is missing or the backend is
unreachable the tracker degrades to a no-op and training still completes — a
metrics sink should never be able to take down the pipeline.

The default backend is SQLite, because the plain-file store cannot serve the
model registry. Point MLFLOW_TRACKING_URI at a real server to share results.
"""

import os
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "telco-churn"
REGISTERED_MODEL_NAME = "telco-churn-classifier"


class ExperimentTracker:
    """Thin wrapper over MLflow that never raises into the training pipeline."""

    def __init__(self, enabled: bool = True, tracking_uri: str | None = None,
                 experiment: str | None = None):
        self.enabled = enabled
        self.mlflow = None
        self.run = None
        self.registered_version = None

        if not enabled:
            return

        try:
            import mlflow
        except ImportError:
            console.log("[yellow]MLflow not installed — skipping experiment tracking")
            self.enabled = False
            return

        self.mlflow = mlflow
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        name = experiment or os.getenv("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT)
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(name)
            self.uri = uri
            console.log(f"[blue]MLflow tracking to {uri} (experiment: {name})")
        except Exception as exc:
            console.log(f"[yellow]MLflow unavailable ({exc}) — continuing untracked")
            self.enabled = False

    def _guard(self, action, label: str):
        """Run an MLflow call, downgrading any failure to a warning."""
        if not self.enabled:
            return None
        try:
            return action()
        except Exception as exc:
            console.log(f"[yellow]MLflow {label} failed: {exc}")
            return None

    def start_run(self, params: dict | None = None):
        def _start():
            self.run = self.mlflow.start_run()
            if params:
                self.mlflow.log_params(params)
            return self.run

        return self._guard(_start, "start_run")

    def log_params(self, params: dict):
        return self._guard(lambda: self.mlflow.log_params(params), "log_params")

    def log_trial(self, trial_number: int, params: dict, value: float):
        """Record one Optuna trial as a nested run so the search is inspectable."""
        def _log():
            with self.mlflow.start_run(nested=True, run_name=f"optuna-trial-{trial_number}"):
                self.mlflow.log_params(params)
                self.mlflow.log_metric("val_roc_auc", value)

        return self._guard(_log, "log_trial")

    def log_model_comparison(self, comparison: dict):
        """Flatten every candidate's metrics onto the parent run."""
        def _log():
            for model_name, metrics in comparison.items():
                for metric_name, value in metrics.items():
                    self.mlflow.log_metric(f"{model_name}.{metric_name}", float(value))

        return self._guard(_log, "log_model_comparison")

    def log_best_model(self, model, model_name: str, model_type: str, metrics: dict,
                       run_dir: Path, register: bool = True):
        def _log():
            self.mlflow.set_tags({
                "best_model": model_name,
                "model_type": model_type,
                "artifact_run": run_dir.name,
            })
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(f"best.{metric_name}", float(value))

            for artifact in ("shap_summary.png", "metrics.json", "model_comparison.json",
                             "train_info.json"):
                path = run_dir / artifact
                if path.exists():
                    self.mlflow.log_artifact(str(path))

            kwargs = {"name": "model"}
            if register:
                kwargs["registered_model_name"] = REGISTERED_MODEL_NAME
            try:
                info = self.mlflow.sklearn.log_model(model, **kwargs)
            except TypeError:
                # MLflow < 3 used `artifact_path` instead of `name`.
                kwargs.pop("name")
                kwargs["artifact_path"] = "model"
                info = self.mlflow.sklearn.log_model(model, **kwargs)

            if register:
                self._record_version(model_name, metrics)
            return info

        return self._guard(_log, "log_best_model")

    def _record_version(self, model_name: str, metrics: dict):
        """Find the version just created and annotate it with why it won."""
        try:
            client = self.mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if not versions:
                return
            latest = max(versions, key=lambda v: int(v.version))
            self.registered_version = latest.version
            client.update_model_version(
                name=REGISTERED_MODEL_NAME,
                version=latest.version,
                description=(
                    f"{model_name} — test ROC-AUC {metrics.get('roc_auc', float('nan')):.4f}"
                ),
            )
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME, latest.version, "algorithm", model_name
            )
            console.log(
                f"[green]Registered {REGISTERED_MODEL_NAME} version {latest.version} "
                f"({model_name})"
            )
        except Exception as exc:
            console.log(f"[yellow]Model registration bookkeeping failed: {exc}")

    def end_run(self):
        def _end():
            self.mlflow.end_run()
            self.run = None

        return self._guard(_end, "end_run")
