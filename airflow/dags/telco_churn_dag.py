"""Airflow DAG for the Telco churn pipeline.

    ingest_sql -> train -> evaluate_gate -> [promote_model | reject_model]
                                                  |
                                            batch_score -> drift_gate -> ab_test

The interesting part is `evaluate_gate`: training producing a model is not the
same as that model being fit to deploy. The gate reads the run's metrics and
branches, so a regression below MIN_ROC_AUC stops at `reject_model` instead of
quietly shipping. `drift_gate` does the same for the scored batch.

Heavy ML steps shell out to the project venv rather than importing the training
code, because Airflow pins its own dependency versions and is installed into a
separate environment (see requirements-airflow.txt).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG

# Operator paths moved in Airflow 3; support both.
try:
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import (
        BranchPythonOperator, PythonOperator,
    )
except ImportError:  # Airflow 2.x
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator

# Where the project lives and which interpreter runs it. Override with Airflow
# Variables or env vars when the scheduler runs outside the repo.
PROJECT_DIR = os.getenv("TELCO_PROJECT_DIR", str(Path(__file__).resolve().parents[2]))
PYTHON_BIN = os.getenv("TELCO_PYTHON_BIN", f"{PROJECT_DIR}/venv/bin/python")

MIN_ROC_AUC = float(os.getenv("TELCO_MIN_ROC_AUC", "0.78"))

# The gate predicates are stdlib-only, so the scheduler can import them without
# the ML dependencies. Everything heavier is shelled out to PYTHON_BIN.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.telco_churn.gates import (  # noqa: E402
    evaluate_model, find_drifted_features, latest_run_dir, write_promotion_marker,
)

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


def evaluate_gate(**context) -> str:
    """Deploy only if the new model clears the quality bar."""
    run_dir = latest_run_dir(Path(PROJECT_DIR) / "artifacts")
    verdict = evaluate_model(run_dir, MIN_ROC_AUC)

    context["ti"].xcom_push(key="run_dir", value=str(run_dir))
    context["ti"].xcom_push(key="roc_auc", value=verdict.roc_auc)
    context["ti"].xcom_push(key="model_name", value=verdict.model_name)

    print(verdict.reason)
    return "promote_model" if verdict.passed else "reject_model"


def promote_model(**context) -> dict:
    """Record the approved run. Downstream serving reads the latest artifact."""
    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="evaluate_gate", key="run_dir"))
    verdict = evaluate_model(run_dir, MIN_ROC_AUC)

    marker = write_promotion_marker(
        run_dir, verdict,
        dag_run=str(context["run_id"]),
        promoted_at=datetime.utcnow().isoformat(),
    )
    print(f"Promoted {verdict.model_name} from {verdict.run} (ROC-AUC {verdict.roc_auc:.4f})")
    return marker


def reject_model(**context) -> None:
    roc_auc = context["ti"].xcom_pull(task_ids="evaluate_gate", key="roc_auc")
    raise ValueError(
        f"Model rejected: ROC-AUC {roc_auc:.4f} is below the {MIN_ROC_AUC} threshold. "
        "The previous model stays live."
    )


def drift_gate(**context) -> None:
    """Fail loudly on drift so the alert is visible in the Airflow UI."""
    drifted = find_drifted_features(Path(PROJECT_DIR) / "reports")
    if drifted:
        # Surfaced as a failure rather than a silent log line: drifted inputs
        # mean the scores are no longer trustworthy.
        raise ValueError(f"Drift detected in {len(drifted)} feature(s): {', '.join(drifted)}")
    print("No drift detected")


with DAG(
    dag_id="telco_churn_pipeline",
    description="Ingest, train, gate, deploy, score and evaluate the churn model",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "churn", "telco"],
) as dag:

    ingest_sql = BashOperator(
        task_id="ingest_sql",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_BIN} -m src.telco_churn.sql_features",
        doc_md="Load the raw CSV into the database and build features in SQL.",
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON_BIN} -m src.telco_churn.train --skip-tuning"
        ),
        doc_md="Train and compare all candidates; the winner is saved and registered.",
    )

    gate = BranchPythonOperator(
        task_id="evaluate_gate",
        python_callable=evaluate_gate,
        doc_md=f"Promote only when test ROC-AUC >= {MIN_ROC_AUC}.",
    )

    promote = PythonOperator(task_id="promote_model", python_callable=promote_model)

    # retries=0: rejection is a deterministic verdict on a finished model, not a
    # transient error. Re-running it reaches the same conclusion and, with the
    # default retry_delay, would sit on the alert for ten minutes first.
    reject = PythonOperator(
        task_id="reject_model", python_callable=reject_model, retries=0,
    )

    batch_score = BashOperator(
        task_id="batch_score",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_BIN} -m src.telco_churn.batch_score",
        doc_md="Score the held-out batch and write the drift report.",
    )

    # Same reasoning: drift is a verdict on a written report, not a flaky call.
    check_drift = PythonOperator(
        task_id="drift_gate", python_callable=drift_gate, retries=0,
    )

    ab_test = BashOperator(
        task_id="ab_test",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_BIN} -m src.telco_churn.ab_test",
        doc_md="Simulate the retention campaign and evaluate significance.",
    )

    ingest_sql >> train >> gate >> [promote, reject]
    promote >> batch_score >> check_drift >> ab_test
