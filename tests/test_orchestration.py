"""Tests for the pipeline quality gates and the Airflow DAG.

The gate logic is stdlib-only and tested directly. The DAG itself is checked
structurally with the ast module so CI validates it without installing Airflow;
a real DagBag parse runs too when Airflow happens to be present.
"""

import ast
import json
from pathlib import Path

import pytest

from src.telco_churn.gates import (
    DEFAULT_MIN_ROC_AUC, evaluate_model, find_drifted_features, latest_run_dir,
    write_promotion_marker,
)

DAG_FILE = Path("airflow/dags/telco_churn_dag.py")


def make_run(tmp_path: Path, roc_auc: float, name: str = "lightgbm") -> Path:
    run_dir = tmp_path / "artifacts" / "20260101-120000"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"roc_auc": roc_auc}))
    (run_dir / "train_info.json").write_text(json.dumps({"model_name": name}))
    return run_dir


class TestRunDiscovery:
    def test_picks_the_newest_run(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        for name in ("20260101-090000", "20260101-120000", "20251231-235959"):
            (artifacts / name).mkdir(parents=True)
        assert latest_run_dir(artifacts).name == "20260101-120000"

    def test_ignores_non_timestamp_directories(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        (artifacts / "20260101-090000").mkdir(parents=True)
        (artifacts / "scratch").mkdir()
        assert latest_run_dir(artifacts).name == "20260101-090000"

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            latest_run_dir(tmp_path / "nope")

    def test_empty_directory_raises(self, tmp_path):
        (tmp_path / "artifacts").mkdir()
        with pytest.raises(FileNotFoundError, match="No training runs"):
            latest_run_dir(tmp_path / "artifacts")


class TestModelGate:
    def test_good_model_passes(self, tmp_path):
        verdict = evaluate_model(make_run(tmp_path, 0.84), min_roc_auc=0.78)
        assert verdict.passed
        assert verdict.roc_auc == 0.84

    def test_weak_model_is_rejected(self, tmp_path):
        verdict = evaluate_model(make_run(tmp_path, 0.61), min_roc_auc=0.78)
        assert not verdict.passed
        assert "< threshold" in verdict.reason

    def test_exact_threshold_passes(self, tmp_path):
        """The bar is inclusive; a model exactly at threshold should ship."""
        assert evaluate_model(make_run(tmp_path, 0.78), min_roc_auc=0.78).passed

    def test_verdict_reports_the_model_name(self, tmp_path):
        verdict = evaluate_model(make_run(tmp_path, 0.9, name="tuned_lightgbm"))
        assert verdict.model_name == "tuned_lightgbm"
        assert "tuned_lightgbm" in verdict.reason

    def test_default_threshold_is_sane(self):
        assert 0.5 < DEFAULT_MIN_ROC_AUC < 1.0


class TestPromotionMarker:
    def test_marker_is_written_and_auditable(self, tmp_path):
        run_dir = make_run(tmp_path, 0.84)
        verdict = evaluate_model(run_dir)
        marker = write_promotion_marker(
            run_dir, verdict, dag_run="manual__2026", promoted_at="2026-01-01T00:00:00"
        )
        written = json.loads((run_dir / "PROMOTED.json").read_text())
        assert written == marker
        assert written["roc_auc"] == 0.84
        assert written["dag_run"] == "manual__2026"


class TestDriftGate:
    def _write_report(self, tmp_path, numeric, categorical, name="drift_report_1.json"):
        reports = tmp_path / "reports"
        reports.mkdir(exist_ok=True)
        (reports / name).write_text(json.dumps({
            "numeric_drift": numeric, "categorical_drift": categorical
        }))
        return reports

    def test_clean_report_yields_no_features(self, tmp_path):
        reports = self._write_report(
            tmp_path, {"tenure": {"drifted": False}}, {"Contract": {"drifted": False}}
        )
        assert find_drifted_features(reports) == []

    def test_drifted_features_are_listed(self, tmp_path):
        reports = self._write_report(
            tmp_path,
            {"tenure": {"drifted": True}, "MonthlyCharges": {"drifted": False}},
            {"Contract": {"drifted": True}},
        )
        assert set(find_drifted_features(reports)) == {"tenure", "Contract"}

    def test_missing_reports_directory_is_not_an_error(self, tmp_path):
        assert find_drifted_features(tmp_path / "absent") == []

    def test_newest_report_wins(self, tmp_path):
        self._write_report(tmp_path, {"tenure": {"drifted": True}}, {},
                           name="drift_report_20260101-000000.json")
        reports = self._write_report(tmp_path, {"tenure": {"drifted": False}}, {},
                                     name="drift_report_20260102-000000.json")
        assert find_drifted_features(reports) == []


class TestDagStructure:
    """Structural checks that need no Airflow install."""

    @pytest.fixture(scope="class")
    def tree(self):
        if not DAG_FILE.exists():
            pytest.skip("DAG file not found")
        return ast.parse(DAG_FILE.read_text())

    def test_dag_file_is_valid_python(self, tree):
        assert tree is not None

    def test_declares_expected_tasks(self, tree):
        source = DAG_FILE.read_text()
        for task_id in ("ingest_sql", "train", "evaluate_gate", "promote_model",
                        "reject_model", "batch_score", "drift_gate", "ab_test"):
            assert f'"{task_id}"' in source, f"DAG missing task {task_id}"

    def test_gate_callables_are_defined(self, tree):
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert {"evaluate_gate", "promote_model", "reject_model", "drift_gate"} <= defined

    def test_declares_dependencies(self):
        """A DAG with no >> edges would run every task in parallel."""
        assert ">>" in DAG_FILE.read_text()

    def test_verdict_tasks_do_not_retry(self):
        """Gate failures are deterministic verdicts, not transient errors.

        Regression test: reject_model inherited retries=2 from default_args, so
        a rejected model was re-evaluated twice — same answer each time — and
        the alert was delayed by two retry_delays.
        """
        source = DAG_FILE.read_text()
        for task_id in ("reject_model", "drift_gate"):
            start = source.index(f'task_id="{task_id}"')
            window = source[start:start + 200]
            assert "retries=0" in window, f"{task_id} should not retry a verdict"

    def test_does_not_import_ml_dependencies(self, tree):
        """The scheduler venv has no ML libs; heavy work is shelled out."""
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not ({"pandas", "sklearn", "lightgbm", "xgboost", "shap"} & imported)


class TestDagParsesInAirflow:
    def test_dagbag_loads_without_errors(self):
        # Probe a submodule, not `airflow`: the repo's own airflow/ directory
        # resolves as a namespace package when Airflow itself is not installed.
        pytest.importorskip("airflow.models", reason="Airflow is installed separately")
        from airflow.models import DagBag

        dagbag = DagBag(dag_folder=str(DAG_FILE.parent), include_examples=False)
        assert not dagbag.import_errors, dagbag.import_errors
        dag = dagbag.get_dag("telco_churn_pipeline")
        assert dag is not None
        assert len(dag.tasks) == 8
