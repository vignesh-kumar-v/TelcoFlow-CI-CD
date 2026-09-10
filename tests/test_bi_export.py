"""Tests for the BI extract layer.

A dashboard is only as good as the extract under it, and the failure mode is
silent: a mislabelled column or a broken join produces a chart that renders
perfectly and says something false. These check the shapes and the joins.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from xml.etree import ElementTree as ET

from src.telco_churn.bi_export import (
    DATA_DICTIONARY, MASTER_FILE, build_master_table, coerce_binary_shap,
    generate_tableau_workbook, write_data_dictionary,
)
RUN_HINT = "no completed pipeline run — run `make pipeline` first"


def make_customers(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n)],
        "churn_probability": rng.uniform(0, 1, n),
        "prediction": rng.integers(0, 2, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "annual_revenue": rng.uniform(240, 1440, n),
        "risk_decile": rng.integers(1, 11, n),
        "segment": rng.integers(0, 3, n),
    })


def make_shap_long(customers, features=("tenure", "Contract", "MonthlyCharges")):
    rows = []
    for i, cid in enumerate(customers["customerID"]):
        for rank, feature in enumerate(features, start=1):
            value = (-1) ** rank * (len(features) - rank + 1) * 0.1
            rows.append({
                "customerID": cid, "feature": feature, "shap_value": value,
                "feature_value": float(i), "abs_shap": abs(value),
                "direction": "increases churn" if value >= 0 else "decreases churn",
                "rank_within_customer": rank,
            })
    return pd.DataFrame(rows)


class TestShapCoercion:
    def test_handles_the_list_form(self):
        values = [np.zeros((3, 2)), np.ones((3, 2))]
        assert np.array_equal(coerce_binary_shap(values), np.ones((3, 2)))

    def test_handles_the_three_dimensional_form(self):
        values = np.stack([np.zeros((3, 2)), np.ones((3, 2))], axis=2)
        assert np.array_equal(coerce_binary_shap(values), np.ones((3, 2)))

    def test_passes_two_dimensional_values_through(self):
        values = np.ones((4, 5))
        assert np.array_equal(coerce_binary_shap(values), values)


class TestMasterTable:
    def test_stays_one_row_per_customer(self):
        """The long SHAP table must be pivoted in, not joined row-wise."""
        customers = make_customers()
        master = build_master_table(customers, make_shap_long(customers))
        assert len(master) == len(customers)
        assert master["customerID"].is_unique

    def test_top_drivers_are_pivoted_into_columns(self):
        customers = make_customers()
        master = build_master_table(customers, make_shap_long(customers), top_k=3)
        for rank in (1, 2, 3):
            for suffix in ("feature", "shap", "direction"):
                assert f"driver_{rank}_{suffix}" in master.columns

    def test_drivers_are_ordered_by_importance(self):
        customers = make_customers()
        master = build_master_table(customers, make_shap_long(customers))
        assert (master["driver_1_shap"].abs() >= master["driver_2_shap"].abs()).all()
        assert (master["driver_2_shap"].abs() >= master["driver_3_shap"].abs()).all()

    def test_drivers_match_the_long_table(self):
        """A mismatch here is a chart that confidently attributes the wrong cause."""
        customers = make_customers()
        shap_long = make_shap_long(customers)
        master = build_master_table(customers, shap_long)

        expected = shap_long[shap_long["rank_within_customer"] == 1] \
            .set_index("customerID")["feature"]
        actual = master.set_index("customerID")["driver_1_feature"]
        assert (actual == expected.reindex(actual.index)).all()

    def test_survives_an_empty_shap_table(self):
        customers = make_customers()
        master = build_master_table(customers, pd.DataFrame())
        assert len(master) == len(customers)

    def test_direction_agrees_with_the_sign(self):
        customers = make_customers()
        master = build_master_table(customers, make_shap_long(customers))
        increases = master["driver_1_direction"] == "increases churn"
        assert (master.loc[increases, "driver_1_shap"] >= 0).all()
        assert (master.loc[~increases, "driver_1_shap"] < 0).all()


class TestTableauWorkbook:
    def test_is_well_formed_xml(self, tmp_path):
        master = make_customers()
        path = generate_tableau_workbook(master, tmp_path, tmp_path / "wb.twb")
        ET.parse(path)

    def test_declares_every_column(self, tmp_path):
        master = make_customers()
        path = generate_tableau_workbook(master, tmp_path, tmp_path / "wb.twb")
        declared = {c.get("name") for c in ET.parse(path).getroot().iter("column")
                    if c.get("ordinal") is not None}
        assert declared == set(master.columns)

    def test_points_at_the_master_extract(self, tmp_path):
        master = make_customers()
        path = generate_tableau_workbook(master, tmp_path, tmp_path / "wb.twb")
        connection = next(
            c for c in ET.parse(path).getroot().iter("connection")
            if c.get("class") == "textscan"
        )
        assert connection.get("filename") == MASTER_FILE
        assert Path(connection.get("directory")).resolve() == tmp_path.resolve()

    def test_identifier_columns_are_dimensions_not_measures(self, tmp_path):
        """Otherwise Tableau sums risk_decile, which is meaningless."""
        master = make_customers()
        path = generate_tableau_workbook(master, tmp_path, tmp_path / "wb.twb")
        roles = {
            c.get("name"): c.get("role")
            for c in ET.parse(path).getroot().iter("column") if c.get("role")
        }
        assert roles["[risk_decile]"] == "dimension"
        assert roles["[segment]"] == "dimension"
        assert roles["[churn_probability]"] == "measure"

    def test_numeric_and_string_columns_are_typed(self, tmp_path):
        master = make_customers()
        path = generate_tableau_workbook(master, tmp_path, tmp_path / "wb.twb")
        types = {c.get("name"): c.get("datatype")
                 for c in ET.parse(path).getroot().iter("column")
                 if c.get("ordinal") is not None}
        assert types["customerID"] == "string"
        assert types["churn_probability"] == "real"


class TestDataDictionary:
    def test_documents_each_exported_table(self, tmp_path):
        tables = {"dim_customer": make_customers(), "fct_shap_global": make_customers()}
        content = write_data_dictionary(tables, tmp_path / "DATA_DICTIONARY.md").read_text()
        for name in tables:
            assert f"`{name}.csv`" in content

    def test_every_documented_table_has_a_description(self):
        assert all(v.strip() for v in DATA_DICTIONARY.values())

    def test_lists_column_names(self, tmp_path):
        tables = {"dim_customer": make_customers()}
        content = write_data_dictionary(tables, tmp_path / "d.md").read_text()
        assert "`churn_probability`" in content


@pytest.mark.integration
class TestExportedFiles:
    """Assertions against a real `make bi-export` run."""

    @staticmethod
    def _export_dir():
        path = Path("outputs/bi")
        if not path.exists() or not (path / MASTER_FILE).exists():
            pytest.skip("no BI export — run `make bi-export` first")
        return path

    def test_master_has_one_row_per_scored_customer(self):
        export = self._export_dir()
        master = pd.read_csv(export / MASTER_FILE)
        customers = pd.read_csv(export / "dim_customer.csv")
        assert len(master) == len(customers)
        assert master["customerID"].is_unique

    def test_risk_category_agrees_with_the_probability(self):
        master = pd.read_csv(self._export_dir() / MASTER_FILE)
        assert (master.loc[master["risk_category"] == "High", "churn_probability"] > 0.6).all()
        assert (master.loc[master["risk_category"] == "Low", "churn_probability"] <= 0.4).all()

    def test_revenue_at_risk_is_the_product_it_claims_to_be(self):
        master = pd.read_csv(self._export_dir() / MASTER_FILE)
        expected = master["annual_revenue"] * master["churn_probability"]
        assert master["revenue_at_risk"].to_numpy() == pytest.approx(expected.to_numpy())

    def test_prediction_outcome_matches_the_confusion_matrix(self):
        master = pd.read_csv(self._export_dir() / MASTER_FILE)
        if "prediction_outcome" not in master.columns:
            pytest.skip("no ground truth in the scored batch")
        tp = master[master["prediction_outcome"] == "true positive"]
        fn = master[master["prediction_outcome"] == "false negative"]
        assert ((tp["prediction"] == 1) & (tp["actual_churn"] == 1)).all()
        assert ((fn["prediction"] == 0) & (fn["actual_churn"] == 1)).all()

    def test_shap_long_table_covers_every_feature_per_customer(self):
        export = self._export_dir()
        shap_long = pd.read_csv(export / "fct_shap_customer.csv")
        per_customer = shap_long.groupby("customerID").size()
        assert per_customer.nunique() == 1, "every customer needs the same feature set"

    def test_shap_ranks_are_consistent_with_magnitudes(self):
        shap_long = pd.read_csv(self._export_dir() / "fct_shap_customer.csv")
        first = shap_long[shap_long["customerID"] == shap_long["customerID"].iloc[0]]
        ordered = first.sort_values("rank_within_customer")["abs_shap"].to_numpy()
        assert (np.diff(ordered) <= 1e-9).all()

    def test_causal_estimates_separate_adjusted_from_naive(self):
        path = self._export_dir() / "fct_causal_estimates.csv"
        if not path.exists():
            pytest.skip("no causal report — run `make causal` first")
        estimates = pd.read_csv(path)
        assert set(estimates["adjusted"].unique()) == {True, False}
        assert estimates["analysis"].nunique() == 3
