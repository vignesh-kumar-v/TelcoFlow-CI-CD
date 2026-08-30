"""Quality gates between pipeline stages.

Producing a model is not the same as that model being fit to deploy, and
producing scores is not the same as those scores being trustworthy. These
predicates make both decisions explicit and reviewable.

Deliberately stdlib-only: the Airflow scheduler imports this from its own
environment, which does not have the ML dependencies installed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MIN_ROC_AUC = 0.78


def latest_run_dir(artifacts_dir: Path) -> Path:
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"No artifacts directory at {artifacts_dir}")
    runs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not runs:
        raise FileNotFoundError(f"No training runs in {artifacts_dir}")
    return sorted(runs, key=lambda d: d.name, reverse=True)[0]


@dataclass
class ModelVerdict:
    passed: bool
    roc_auc: float
    model_name: str
    threshold: float
    run: str

    @property
    def reason(self) -> str:
        comparison = ">=" if self.passed else "<"
        return (
            f"{self.model_name} scored ROC-AUC {self.roc_auc:.4f} "
            f"{comparison} threshold {self.threshold}"
        )


def evaluate_model(run_dir: Path, min_roc_auc: float = DEFAULT_MIN_ROC_AUC) -> ModelVerdict:
    """Decide whether a finished training run is fit to promote."""
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    with open(run_dir / "train_info.json") as f:
        train_info = json.load(f)

    roc_auc = float(metrics["roc_auc"])
    return ModelVerdict(
        passed=roc_auc >= min_roc_auc,
        roc_auc=roc_auc,
        model_name=train_info.get("model_name", "unknown"),
        threshold=min_roc_auc,
        run=run_dir.name,
    )


def find_drifted_features(reports_dir: Path) -> list[str]:
    """Features flagged in the most recent drift report, newest report wins."""
    reports = sorted(Path(reports_dir).glob("drift_report_*.json"), reverse=True)
    if not reports:
        return []
    with open(reports[0]) as f:
        report = json.load(f)
    return (
        [k for k, v in report.get("numeric_drift", {}).items() if v.get("drifted")]
        + [k for k, v in report.get("categorical_drift", {}).items() if v.get("drifted")]
    )


def write_promotion_marker(run_dir: Path, verdict: ModelVerdict, dag_run: str,
                           promoted_at: str) -> dict:
    """Stamp an approved run so it is auditable after the fact."""
    marker = {
        "promoted_at": promoted_at,
        "run": verdict.run,
        "model_name": verdict.model_name,
        "roc_auc": verdict.roc_auc,
        "threshold": verdict.threshold,
        "dag_run": dag_run,
    }
    (Path(run_dir) / "PROMOTED.json").write_text(json.dumps(marker, indent=2))
    return marker
