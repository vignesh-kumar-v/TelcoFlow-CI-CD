"""Tests for the SQL feature-engineering layer.

Runs against a throwaway SQLite file, so these are real end-to-end SQL tests
that need no services and run in CI.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.telco_churn.db import execute_script, get_engine, read_sql_file, split_statements
from src.telco_churn.features import (
    ENGINEERED_NOMINAL, ENGINEERED_NUMERIC, compute_engineered_features,
)
from src.telco_churn.sql_features import (
    CANONICAL_COLUMNS, LOWER_TO_CANONICAL, export_features, ingest_raw,
    run_transformations,
)
from src.telco_churn.validate_and_clean import clean_data

RAW_CSV = Path("data/raw/Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """A populated, fully transformed throwaway database."""
    if not RAW_CSV.exists():
        pytest.skip("raw data not available")
    db_path = tmp_path_factory.mktemp("sqldb") / "test.db"
    eng = get_engine(f"sqlite:///{db_path}")
    ingest_raw(RAW_CSV, eng)
    run_transformations(eng)
    return eng


class TestStatementSplitting:
    """Semicolons inside comments and strings must not split a statement."""

    def test_semicolon_in_comment_is_not_a_split(self):
        sql = "-- a note; with a semicolon\nSELECT 1;"
        assert len(split_statements(sql)) == 1

    def test_semicolon_in_string_literal_is_not_a_split(self):
        assert len(split_statements("SELECT 'a;b' AS x;")) == 1

    def test_multiple_statements_split(self):
        assert len(split_statements("SELECT 1; SELECT 2;")) == 2

    def test_comment_only_script_yields_nothing(self):
        assert split_statements("-- just a comment\n") == []


class TestShippedSql:
    def test_every_sql_file_parses(self):
        for name in ("01_clean_customers.sql", "02_feature_engineering.sql",
                     "03_churn_analytics.sql"):
            assert split_statements(read_sql_file(name)), f"{name} produced no statements"


class TestCleaning:
    def test_row_count_preserved(self, engine):
        raw = pd.read_sql("SELECT COUNT(*) AS n FROM raw_customers", engine)["n"][0]
        clean = pd.read_sql("SELECT COUNT(*) AS n FROM clean_customers", engine)["n"][0]
        assert raw == clean == 7043

    def test_total_charges_is_numeric_with_no_nulls(self, engine):
        df = pd.read_sql("SELECT totalcharges FROM clean_customers", engine)
        assert pd.api.types.is_numeric_dtype(df["totalcharges"])
        assert df["totalcharges"].isna().sum() == 0

    def test_churn_encoded_to_binary(self, engine):
        df = pd.read_sql("SELECT DISTINCT churn FROM clean_customers", engine)
        assert set(df["churn"]) == {0, 1}

    def test_sql_cleaning_matches_pandas_cleaning(self, engine):
        """The SQL path must reproduce the pandas path it replaces.

        Same median imputation, same target encoding — otherwise switching
        backends would silently retrain the model on different data.
        """
        sql_df = pd.read_sql(
            "SELECT customerid, totalcharges, churn FROM clean_customers ORDER BY customerid",
            engine,
        )
        pandas_df = clean_data(pd.read_csv(RAW_CSV)).sort_values("customerID")

        assert sql_df["churn"].tolist() == pandas_df["Churn"].tolist()
        pd.testing.assert_series_equal(
            sql_df["totalcharges"].reset_index(drop=True),
            pandas_df["TotalCharges"].reset_index(drop=True),
            check_names=False, rtol=1e-9,
        )

    def test_blank_total_charges_got_the_median(self, engine):
        """The 11 tenure=0 rows should carry the imputed median, not zero."""
        df = pd.read_sql(
            "SELECT totalcharges FROM clean_customers WHERE tenure = 0", engine
        )
        assert len(df) == 11
        assert (df["totalcharges"] > 0).all()
        assert df["totalcharges"].nunique() == 1  # all got the same median


class TestFeatureEngineering:
    def test_engineered_columns_present(self, engine):
        df = pd.read_sql("SELECT * FROM customer_features LIMIT 1", engine)
        for col in ENGINEERED_NUMERIC + ENGINEERED_NOMINAL:
            assert col in df.columns, f"missing engineered column {col}"

    def test_addon_count_in_valid_range(self, engine):
        df = pd.read_sql("SELECT num_addon_services FROM customer_features", engine)
        assert df["num_addon_services"].between(0, 6).all()

    def test_no_internet_service_not_counted_as_addon(self, engine):
        """'No internet service' is a placeholder, not a declined add-on."""
        df = pd.read_sql(
            "SELECT num_addon_services FROM customer_features "
            "WHERE internetservice = 'No'", engine
        )
        assert (df["num_addon_services"] == 0).all()

    def test_avg_monthly_spend_defined_for_zero_tenure(self, engine):
        """Division by tenure must not produce NULL/inf for new customers."""
        df = pd.read_sql(
            "SELECT avg_monthly_spend, monthlycharges FROM customer_features "
            "WHERE tenure = 0", engine
        )
        assert df["avg_monthly_spend"].notna().all()
        pd.testing.assert_series_equal(
            df["avg_monthly_spend"], df["monthlycharges"], check_names=False
        )

    def test_no_nulls_in_engineered_columns(self, engine):
        df = pd.read_sql("SELECT * FROM customer_features", engine)
        for col in ENGINEERED_NUMERIC + ENGINEERED_NOMINAL:
            assert df[col].isna().sum() == 0, f"{col} contains NULLs"

    def test_buckets_cover_every_row(self, engine):
        df = pd.read_sql(
            "SELECT tenure_bucket, spend_bucket FROM customer_features", engine
        )
        assert set(df["tenure_bucket"]) <= {"0-6m", "6-12m", "1-2y", "2-4y", "4y+"}
        assert set(df["spend_bucket"]) <= {"low", "medium", "high", "premium"}


class TestAnalytics:
    def test_segment_report_built(self, engine):
        df = pd.read_sql("SELECT * FROM churn_by_segment", engine)
        assert len(df) > 0
        assert df["churn_rate"].between(0, 1).all()
        assert (df["customers"] >= 20).all(), "HAVING clause should drop tiny segments"

    def test_month_to_month_churns_more_than_two_year(self, engine):
        """Sanity check that the engineered segments separate the target."""
        df = pd.read_sql(
            "SELECT contract, SUM(churned) * 1.0 / SUM(customers) AS rate "
            "FROM churn_by_segment GROUP BY contract", engine
        ).set_index("contract")["rate"]
        assert df["Month-to-month"] > df.get("Two year", 0)


class TestTrainServeParity:
    """The Python feature code must reproduce the SQL exactly.

    Training features are built in the warehouse; serving features are built in
    Python from a single API payload. Any disagreement is training/serving skew
    — the model would score live traffic on differently-computed inputs.
    """

    @pytest.fixture(scope="class")
    def both(self, engine):
        sql_df = pd.read_sql(
            "SELECT * FROM customer_features ORDER BY customerid", engine
        ).rename(columns=LOWER_TO_CANONICAL)
        raw = clean_data(pd.read_csv(RAW_CSV)).sort_values("customerID").reset_index(drop=True)
        python_df = compute_engineered_features(raw)
        return sql_df.reset_index(drop=True), python_df

    def test_addon_counts_match(self, both):
        sql_df, python_df = both
        pd.testing.assert_series_equal(
            sql_df["num_addon_services"].astype(int),
            python_df["num_addon_services"].astype(int),
            check_names=False,
        )

    def test_avg_monthly_spend_matches(self, both):
        sql_df, python_df = both
        pd.testing.assert_series_equal(
            sql_df["avg_monthly_spend"].astype(float),
            python_df["avg_monthly_spend"].astype(float),
            check_names=False, rtol=1e-9,
        )

    def test_charges_ratio_matches(self, both):
        sql_df, python_df = both
        pd.testing.assert_series_equal(
            sql_df["charges_ratio"].astype(float),
            python_df["charges_ratio"].astype(float),
            check_names=False, rtol=1e-9,
        )

    def test_tenure_buckets_match(self, both):
        sql_df, python_df = both
        assert sql_df["tenure_bucket"].tolist() == list(python_df["tenure_bucket"])

    def test_spend_buckets_match(self, both):
        """Boundary values decide bucket edges; < vs <= must agree with the SQL."""
        sql_df, python_df = both
        assert sql_df["spend_bucket"].tolist() == list(python_df["spend_bucket"])


class TestExport:
    def test_export_produces_model_ready_parquet(self, engine, tmp_path):
        out = tmp_path / "features.parquet"
        df = export_features(engine, out)
        assert out.exists()
        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) == 7043
        # Canonical mixed-case names, not the lower-case DB names.
        assert "customerID" in df.columns
        assert "MonthlyCharges" in df.columns

    def test_export_is_idempotent(self, engine, tmp_path):
        """Re-running the transformations must not duplicate or drop rows."""
        run_transformations(engine)
        df = export_features(engine, tmp_path / "again.parquet")
        assert len(df) == 7043
