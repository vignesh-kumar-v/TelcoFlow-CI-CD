"""Tests for the logistic regression interpretation layer.

The whole table rests on two claims: that the unpenalised reference-coded model
is a fair description of what is deployed, and that its standard errors are
computed on a full-rank design. Both are checked here.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.telco_churn.features import TARGET, drop_non_features
from src.telco_churn.inference import LoadedModel
from src.telco_churn.interpret import (
    VIF_WARNING, build_inference_matrix, fit_inference_model,
    variance_inflation_factors, wald_inference,
)

RUN_HINT = "no completed pipeline run — run `make pipeline` first"


def make_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(20, 8000, n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "gender": rng.choice(["Male", "Female"], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
    }), (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)


class TestInferenceMatrix:
    def test_drops_one_level_per_categorical(self):
        """Reference coding is what makes the unpenalised fit identifiable."""
        df, _ = make_frame()
        design, _ = build_inference_matrix(df)
        contract_terms = [c for c in design.columns if c.startswith("Contract")]
        assert len(contract_terms) == 2, "3 levels must produce 2 terms"

    def test_matrix_is_full_rank(self):
        """The serving matrix is not; that is the reason this one exists."""
        df, _ = make_frame()
        design, _ = build_inference_matrix(df)
        assert np.linalg.matrix_rank(design.to_numpy(dtype=float)) == design.shape[1]

    def test_serving_matrix_is_rank_deficient_by_comparison(self):
        """Documents the problem: every level kept, so the dummies are collinear."""
        from src.telco_churn.features import FeaturePipeline
        df, _ = make_frame()
        pipeline = FeaturePipeline().fit(df)
        served = pipeline.transform_linear(df)
        assert np.linalg.matrix_rank(served) < served.shape[1]

    def test_numeric_columns_pass_through_unchanged(self):
        df, _ = make_frame()
        design, _ = build_inference_matrix(df)
        assert design["tenure"].to_numpy() == pytest.approx(df["tenure"].to_numpy())


class TestWaldInference:
    def test_odds_ratio_is_the_exponentiated_coefficient(self):
        df, y = make_frame()
        design, _ = build_inference_matrix(df)
        model, scaler, X_scaled = fit_inference_model(design, y)
        table = wald_inference(model, X_scaled, design, scaler)
        assert table["odds_ratio"].to_numpy() == pytest.approx(
            np.exp(table["coefficient"].to_numpy())
        )

    def test_confidence_interval_brackets_the_odds_ratio(self):
        df, y = make_frame()
        design, _ = build_inference_matrix(df)
        model, scaler, X_scaled = fit_inference_model(design, y)
        table = wald_inference(model, X_scaled, design, scaler)
        assert (table["or_ci_low"] <= table["odds_ratio"]).all()
        assert (table["odds_ratio"] <= table["or_ci_high"]).all()

    def test_significance_agrees_with_the_interval(self):
        """A significant term whose interval spans an odds ratio of 1 is a bug."""
        df, y = make_frame()
        design, _ = build_inference_matrix(df)
        model, scaler, X_scaled = fit_inference_model(design, y)
        table = wald_inference(model, X_scaled, design, scaler)
        for _, row in table.iterrows():
            spans_one = row["or_ci_low"] <= 1.0 <= row["or_ci_high"]
            assert row["significant"] != spans_one

    def test_recovers_a_known_log_odds_effect(self):
        """The estimator has to be right before its intervals mean anything."""
        rng = np.random.default_rng(4)
        n = 20000
        x = rng.normal(size=n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 + 1.5 * x)))).astype(int)
        design = pd.DataFrame({"x": x})
        model, scaler, X_scaled = fit_inference_model(design, y)
        table = wald_inference(model, X_scaled, design, scaler)
        # class_weight="balanced" shifts the intercept, not the slope.
        assert table.loc[0, "coefficient"] == pytest.approx(1.5, rel=0.15)

    def test_standard_errors_shrink_with_sample_size(self):
        errors = []
        for n in (500, 8000):
            df, y = make_frame(n=n, seed=1)
            design, _ = build_inference_matrix(df)
            model, scaler, X_scaled = fit_inference_model(design, y)
            errors.append(wald_inference(model, X_scaled, design, scaler)["std_error"].mean())
        assert errors[1] < errors[0]


class TestVif:
    def test_independent_columns_score_near_one(self):
        rng = np.random.default_rng(0)
        design = pd.DataFrame(rng.normal(size=(2000, 4)), columns=list("abcd"))
        assert variance_inflation_factors(design).max() < 1.5

    def test_collinear_columns_are_flagged(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=2000)
        design = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.01, 2000),
                               "c": rng.normal(size=2000)})
        vif = variance_inflation_factors(design)
        assert vif["a"] > VIF_WARNING
        assert vif["c"] < 2

    def test_never_returns_below_one(self):
        rng = np.random.default_rng(2)
        design = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
        assert (variance_inflation_factors(design) >= 1.0).all()


@pytest.mark.integration
class TestAgainstTheDeployedModel:
    """The claim that makes the coefficients worth reading."""

    @staticmethod
    def _loaded():
        artifacts = Path("artifacts")
        if not artifacts.exists():
            pytest.skip(RUN_HINT)
        runs = [d for d in artifacts.iterdir() if d.is_dir() and d.name[0].isdigit()]
        if not runs:
            pytest.skip(RUN_HINT)
        return LoadedModel.load(sorted(runs, key=lambda d: d.name, reverse=True)[0])

    def test_inference_model_ranks_customers_like_the_deployed_one(self):
        from src.telco_churn.features import LINEAR
        from src.telco_churn.interpret import check_agreement

        loaded = self._loaded()
        if loaded.model_type != LINEAR:
            pytest.skip("the deployed model is not the linear one")

        clean = Path("data/processed/cleaned_data.parquet")
        if not clean.exists():
            pytest.skip(RUN_HINT)
        df = pd.read_parquet(clean)

        design, _ = build_inference_matrix(drop_non_features(df))
        model, scaler, _ = fit_inference_model(design, df[TARGET].astype(int).to_numpy())
        agreement = check_agreement(loaded, df, model, scaler, design)

        assert agreement["spearman_rho"] > 0.95, (
            "the coefficients would describe a model that is not serving"
        )
        assert agreement["auc_gap"] < 0.02
