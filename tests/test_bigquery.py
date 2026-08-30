"""BigQuery layer tests.

The client is mocked so these run in CI with no credentials. Anything needing a
real project is marked `gcp` and skipped unless GCP_PROJECT is set.
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import typer

from src.telco_churn.bigquery_loader import (
    ANALYTICS_QUERY, CLEANED_TABLE, FEATURE_QUERY, get_project, load_dataframe,
    run_query, table_id,
)


class TestProjectResolution:
    def test_explicit_project_wins(self):
        assert get_project("explicit-project") == "explicit-project"

    def test_falls_back_to_environment(self, monkeypatch):
        monkeypatch.setenv("GCP_PROJECT", "env-project")
        assert get_project() == "env-project"

    def test_missing_project_raises_with_guidance(self, monkeypatch):
        monkeypatch.delenv("GCP_PROJECT", raising=False)
        with pytest.raises(typer.BadParameter, match="gcloud auth"):
            get_project()

    def test_table_id_is_fully_qualified(self):
        assert table_id("proj", "ds", "tbl") == "proj.ds.tbl"


class TestQueries:
    """The SQL must be valid-looking and mirror the local feature definitions."""

    def test_feature_query_defines_every_engineered_column(self):
        sql = FEATURE_QUERY.format(table="p.d.t")
        for col in ("num_addon_services", "avg_monthly_spend", "charges_ratio",
                    "tenure_bucket", "spend_bucket"):
            assert col in sql, f"feature query missing {col}"

    def test_feature_query_guards_division_by_zero(self):
        """tenure = 0 customers must not blow up the query."""
        assert "SAFE_DIVIDE" in FEATURE_QUERY

    def test_analytics_query_composes_the_feature_query(self):
        sql = ANALYTICS_QUERY.format(feature_query=FEATURE_QUERY.format(table="p.d.t"))
        assert "p.d.t" in sql
        assert "GROUP BY" in sql
        assert "HAVING COUNT(*) >= 20" in sql

    def test_table_reference_is_interpolated(self):
        assert "`proj.ds.tbl`" in FEATURE_QUERY.format(table="proj.ds.tbl")


class TestLoad:
    def test_load_waits_for_the_job(self):
        """job.result() must be awaited, or load errors pass silently."""
        client = MagicMock()
        job = MagicMock()
        client.load_table_from_dataframe.return_value = job
        client.get_table.return_value = MagicMock(num_rows=3, schema=[1, 2])

        df = pd.DataFrame({"a": [1, 2, 3]})
        load_dataframe(client, df, "p.d.t")

        job.result.assert_called_once()
        client.load_table_from_dataframe.assert_called_once()

    def test_load_defaults_to_truncate(self):
        client = MagicMock()
        client.get_table.return_value = MagicMock(num_rows=1, schema=[1])
        load_dataframe(client, pd.DataFrame({"a": [1]}), "p.d.t")
        job_config = client.load_table_from_dataframe.call_args.kwargs["job_config"]
        assert job_config.write_disposition == "WRITE_TRUNCATE"

    def test_write_mode_is_configurable(self):
        client = MagicMock()
        client.get_table.return_value = MagicMock(num_rows=1, schema=[1])
        load_dataframe(client, pd.DataFrame({"a": [1]}), "p.d.t", write_mode="WRITE_APPEND")
        job_config = client.load_table_from_dataframe.call_args.kwargs["job_config"]
        assert job_config.write_disposition == "WRITE_APPEND"


class TestQueryExecution:
    def test_run_query_reads_via_pandas_gbq(self):
        client = MagicMock()
        client.project = "proj"
        expected = pd.DataFrame({"churn_rate": [0.77, 0.5]})
        with patch("pandas_gbq.read_gbq", return_value=expected) as read_gbq:
            result = run_query(client, "SELECT 1")
        pd.testing.assert_frame_equal(result, expected)
        assert read_gbq.call_args.args[0] == "SELECT 1"
        assert read_gbq.call_args.kwargs["project_id"] == "proj"

    def test_falls_back_to_the_client_without_pandas_gbq(self):
        """The loader must still work when the optional accelerator is absent."""
        import builtins

        client = MagicMock()
        expected = pd.DataFrame({"churn_rate": [0.42]})
        client.query.return_value.result.return_value.to_dataframe.return_value = expected

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "pandas_gbq":
                raise ImportError("pandas_gbq not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", blocked):
            result = run_query(client, "SELECT 1")

        pd.testing.assert_frame_equal(result, expected)
        client.query.assert_called_once_with("SELECT 1")


@pytest.mark.gcp
@pytest.mark.skipif(not os.getenv("GCP_PROJECT"), reason="needs GCP_PROJECT and credentials")
class TestAgainstRealBigQuery:
    """End-to-end against a real project. Opt-in via GCP_PROJECT."""

    def test_client_connects(self):
        from src.telco_churn.bigquery_loader import get_client
        client = get_client()
        assert client.project == os.environ["GCP_PROJECT"]

    def test_feature_query_runs(self):
        from src.telco_churn.bigquery_loader import get_client
        client = get_client()
        source = table_id(client.project, "telco_churn", CLEANED_TABLE)
        df = run_query(client, FEATURE_QUERY.format(table=source) + " LIMIT 10")
        assert len(df) <= 10
        assert "num_addon_services" in df.columns

    def test_feature_query_is_idempotent(self):
        """Must run against a table that already carries engineered columns.

        Regression test: selecting `*` alongside the derived columns emitted
        each of them twice and BigQuery rejected the query as ambiguous.
        """
        from src.telco_churn.bigquery_loader import get_client
        client = get_client()
        source = table_id(client.project, "telco_churn", CLEANED_TABLE)
        df = run_query(client, FEATURE_QUERY.format(table=source) + " LIMIT 5")
        assert not df.columns.duplicated().any(), "feature query produced duplicate columns"

    def test_bigquery_features_match_local_computation(self):
        """BigQuery SQL and the Python serving path must agree exactly.

        The warehouse builds training features; Python builds them for live
        requests. Divergence here is training/serving skew against real data.
        """
        import pandas as pd

        from src.telco_churn.bigquery_loader import get_client
        from src.telco_churn.features import (
            ENGINEERED_NOMINAL, ENGINEERED_NUMERIC, compute_engineered_features,
        )

        client = get_client()
        source = table_id(client.project, "telco_churn", CLEANED_TABLE)
        bq = run_query(client, FEATURE_QUERY.format(table=source))
        bq = bq.sort_values("customerID").reset_index(drop=True)

        local = compute_engineered_features(
            bq.drop(columns=ENGINEERED_NUMERIC + ENGINEERED_NOMINAL)
        )

        for col in ENGINEERED_NUMERIC:
            pd.testing.assert_series_equal(
                bq[col].astype(float), local[col].astype(float),
                check_names=False, rtol=1e-9,
            )
        for col in ENGINEERED_NOMINAL:
            assert bq[col].astype(str).tolist() == list(local[col].astype(str))
