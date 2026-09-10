"""Tests for the causal inference module.

Causal estimates cannot be checked against a held-out set — the counterfactual
is never observed. So the estimators are tested the only way they can be: on
data whose true effect is known by construction, where each one has to recover
that effect *and* the naive comparison it replaces has to fail to.

The regression primitives are checked against textbook cases first, because an
estimator built on a wrong OLS is wrong everywhere.
"""

import numpy as np
import pandas as pd
import pytest

from src.telco_churn.causal import (
    SMD_THRESHOLD, TREATMENT_COLUMN, build_confounder_matrix, check_overlap,
    difference_in_differences, e_value, estimate_propensity,
    instrumental_variables_analysis, inverse_probability_weights, ipw_ate,
    match_nearest_neighbour, att_from_matches, ols, prepare_observational,
    parallel_trends_test, simulate_did_panel, simulate_encouragement,
    standardised_mean_differences, two_stage_least_squares,
)
from src.telco_churn.features import TARGET


def make_scored(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n)],
        "churn_probability": rng.uniform(0.15, 0.85, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
    })


class TestOls:
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        n = 4000
        x = rng.normal(size=n)
        y = 2.0 + 3.0 * x + rng.normal(0, 1, n)
        result = ols(y, np.column_stack([np.ones(n), x]), ["const", "x"])
        assert result.coefficients[0] == pytest.approx(2.0, abs=0.1)
        assert result.coefficients[1] == pytest.approx(3.0, abs=0.1)

    def test_confidence_interval_brackets_the_truth(self):
        rng = np.random.default_rng(1)
        n = 4000
        x = rng.normal(size=n)
        y = 1.0 + 0.5 * x + rng.normal(0, 1, n)
        term = ols(y, np.column_stack([np.ones(n), x]), ["const", "x"]).term("x")
        assert term["ci_95"][0] < 0.5 < term["ci_95"][1]

    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(2)
        n = 2000
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        assert not ols(y, np.column_stack([np.ones(n), x]), ["const", "x"]).term("x")["significant"]

    def test_robust_errors_grow_under_heteroskedasticity(self):
        """HC1 exists for this case; classical errors would understate it."""
        rng = np.random.default_rng(3)
        n = 3000
        x = rng.uniform(0.1, 3, n)
        y = 1 + 2 * x + rng.normal(0, 1, n) * x       # variance rises with x
        robust = ols(y, np.column_stack([np.ones(n), x]), ["const", "x"]).std_errors[1]

        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        resid = y - X @ beta
        classical = np.sqrt(
            (resid @ resid / (n - 2)) * np.linalg.pinv(X.T @ X)[1, 1]
        )
        assert robust > classical


class TestTwoStageLeastSquares:
    def test_recovers_the_truth_where_naive_ols_cannot(self):
        rng = np.random.default_rng(0)
        n = 6000
        z = rng.normal(size=n)
        confounder = rng.normal(size=n)
        d = 0.8 * z + confounder + rng.normal(0, 0.5, n)
        y = 1.0 + 2.0 * d + 3.0 * confounder + rng.normal(0, 1, n)
        exog = np.ones((n, 1))

        naive = ols(y, np.column_stack([d, exog]), ["d", "const"]).coefficients[0]
        iv, diagnostics = two_stage_least_squares(y, d, z, exog, ["d", "const"])

        assert iv.coefficients[0] == pytest.approx(2.0, abs=0.15)
        assert abs(naive - 2.0) > 1.0, "the naive estimate should be badly biased"
        assert not diagnostics["weak_instrument"]

    def test_weak_instrument_is_flagged(self):
        rng = np.random.default_rng(1)
        n = 2000
        z = rng.normal(size=n)
        d = 0.005 * z + rng.normal(size=n)          # almost no first stage
        y = 2.0 * d + rng.normal(size=n)
        _, diagnostics = two_stage_least_squares(y, d, z, np.ones((n, 1)), ["d", "const"])
        assert diagnostics["weak_instrument"]
        assert diagnostics["first_stage_f_statistic"] < 10


class TestObservationalSetup:
    def test_ineligible_customers_are_excluded(self):
        """Customers with no internet cannot buy Tech Support."""
        df = pd.DataFrame({
            TREATMENT_COLUMN: ["Yes", "No", "No internet service"],
            TARGET: [0, 1, 1], "tenure": [10, 20, 30],
            "TotalCharges": [100.0, 200.0, 300.0], "MonthlyCharges": [10.0, 10.0, 10.0],
            **{c: ["No", "No", "No internet service"] for c in
               ("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "StreamingTV", "StreamingMovies")},
        })
        prepared = prepare_observational(df)
        assert len(prepared) == 2
        assert "No internet service" not in prepared[TREATMENT_COLUMN].to_numpy()

    def test_addon_count_excludes_the_treatment_itself(self):
        """num_addon_services counts Tech Support, so it is a mechanical control."""
        df = pd.DataFrame({
            TREATMENT_COLUMN: ["Yes"], TARGET: [0], "tenure": [10],
            "TotalCharges": [100.0], "MonthlyCharges": [10.0],
            "OnlineSecurity": ["Yes"], "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"], "StreamingTV": ["No"], "StreamingMovies": ["No"],
        })
        prepared = prepare_observational(df)
        assert prepared["num_other_addons"].iloc[0] == 2
        assert prepared["num_addon_services"].iloc[0] == 3

    def test_mediator_is_absent_from_the_primary_specification(self):
        """MonthlyCharges is caused by the treatment; controlling for it is a bad control."""
        df = prepare_observational(pd.DataFrame({
            TREATMENT_COLUMN: ["Yes", "No"] * 10, TARGET: [0, 1] * 10,
            "tenure": list(range(1, 21)), "TotalCharges": [100.0] * 20,
            "MonthlyCharges": [50.0] * 20, "Contract": ["Month-to-month"] * 20,
            "InternetService": ["DSL"] * 20, "PaymentMethod": ["Mailed check"] * 20,
            "PaperlessBilling": ["No"] * 20, "Partner": ["No"] * 20,
            "Dependents": ["No"] * 20, "SeniorCitizen": [0] * 20, "gender": ["Male"] * 20,
            "PhoneService": ["Yes"] * 20, "MultipleLines": ["No"] * 20,
            **{c: ["No"] * 20 for c in ("OnlineSecurity", "OnlineBackup",
                                        "DeviceProtection", "StreamingTV", "StreamingMovies")},
        }))
        assert "MonthlyCharges" not in build_confounder_matrix(df).columns
        assert "MonthlyCharges" in build_confounder_matrix(df, include_mediators=True).columns


class TestPropensityMethods:
    """Confounded data with a known effect: adjustment must beat the raw gap."""

    @staticmethod
    def confounded(n=4000, effect=-0.10, seed=0):
        rng = np.random.default_rng(seed)
        confounder = rng.normal(size=n)
        # Low-risk customers select into treatment, so the raw gap overstates it.
        treated = (rng.random(n) < 1 / (1 + np.exp(-1.2 * confounder))).astype(int)
        risk = np.clip(0.45 - 0.15 * confounder + effect * treated, 0.01, 0.99)
        outcome = (rng.random(n) < risk).astype(int)
        X = pd.DataFrame({"confounder": confounder, "noise": rng.normal(size=n)})
        return X, treated, outcome

    def test_matching_beats_the_naive_difference(self):
        X, treated, outcome = self.confounded()
        naive = outcome[treated == 1].mean() - outcome[treated == 0].mean()
        propensity = estimate_propensity(X, treated)
        pairs, _ = match_nearest_neighbour(propensity, treated)
        att = att_from_matches(outcome, pairs)
        assert abs(att - (-0.10)) < abs(naive - (-0.10))
        assert att == pytest.approx(-0.10, abs=0.05)

    def test_ipw_recovers_the_effect(self):
        X, treated, outcome = self.confounded()
        propensity = estimate_propensity(X, treated)
        ate = ipw_ate(outcome, treated, inverse_probability_weights(propensity, treated))
        assert ate == pytest.approx(-0.10, abs=0.05)

    def test_matching_improves_covariate_balance(self):
        X, treated, outcome = self.confounded()
        propensity = estimate_propensity(X, treated)
        before = standardised_mean_differences(X, treated).abs().max()
        pairs, _ = match_nearest_neighbour(propensity, treated)
        rows = [i for pair in pairs for i in pair]
        after = standardised_mean_differences(
            X.iloc[rows].reset_index(drop=True), treated[rows]
        ).abs().max()
        assert before > SMD_THRESHOLD, "the fixture should start imbalanced"
        assert after < before

    def test_weighting_improves_covariate_balance(self):
        X, treated, outcome = self.confounded()
        propensity = estimate_propensity(X, treated)
        before = standardised_mean_differences(X, treated).abs().max()
        weighted = standardised_mean_differences(
            X, treated, weights=inverse_probability_weights(propensity, treated)
        ).abs().max()
        assert weighted < before

    def test_no_true_effect_yields_no_estimated_effect(self):
        """The placebo case: adjustment must not manufacture an effect."""
        X, treated, outcome = self.confounded(effect=0.0, seed=5)
        propensity = estimate_propensity(X, treated)
        pairs, _ = match_nearest_neighbour(propensity, treated)
        assert att_from_matches(outcome, pairs) == pytest.approx(0.0, abs=0.04)

    def test_matching_is_without_replacement(self):
        """A control matched to several treated units would dominate the estimate."""
        X, treated, _ = self.confounded(n=1500)
        propensity = estimate_propensity(X, treated)
        pairs, _ = match_nearest_neighbour(propensity, treated)
        controls = [c for _, c in pairs]
        assert len(controls) == len(set(controls))

    def test_caliper_leaves_hopeless_units_unmatched(self):
        """A treated unit with no comparable control should be dropped, not forced."""
        propensity = np.concatenate([np.full(20, 0.99), np.full(20, 0.01)])
        treated = np.concatenate([np.ones(20, int), np.zeros(20, int)])
        _, info = match_nearest_neighbour(propensity, treated, caliper_sd=0.01)
        assert info["n_unmatched_treated"] > 0

    def test_overlap_drops_units_outside_common_support(self):
        propensity = np.array([0.001, 0.4, 0.5, 0.6, 0.999])
        treated = np.array([1, 1, 0, 0, 0])
        overlap = check_overlap(propensity, treated)
        assert overlap["n_dropped"] > 0
        assert overlap["n_in_support"] < overlap["n_total"]


class TestEValue:
    def test_larger_effects_need_stronger_confounders(self):
        assert e_value(3.0) > e_value(1.5)

    def test_no_effect_needs_no_confounder(self):
        assert e_value(1.0) == pytest.approx(1.0)

    def test_is_symmetric_in_the_direction_of_effect(self):
        assert e_value(2.0) == pytest.approx(e_value(0.5))


class TestEncouragementDesign:
    def test_2sls_recovers_the_complier_effect(self):
        scored = make_scored(n=6000)
        design = simulate_encouragement(scored, seed=3)
        result = instrumental_variables_analysis(design)
        truth = float(design["true_complier_effect"].iloc[0])
        assert result["two_stage_least_squares"]["coefficient"] == pytest.approx(truth, abs=0.05)

    def test_2sls_is_less_biased_than_naive_ols_across_seeds(self):
        """One draw proves nothing about an estimator; the average bias does."""
        scored = make_scored(n=4000)
        naive_bias, iv_bias = [], []
        for seed in range(8):
            design = simulate_encouragement(scored, seed=seed)
            result = instrumental_variables_analysis(design)
            truth = float(design["true_complier_effect"].iloc[0])
            naive_bias.append(result["naive_ols"]["coefficient"] - truth)
            iv_bias.append(result["two_stage_least_squares"]["coefficient"] - truth)
        assert abs(np.mean(iv_bias)) < abs(np.mean(naive_bias))
        assert abs(np.mean(iv_bias)) < 0.03

    def test_the_instrument_is_strong(self):
        design = simulate_encouragement(make_scored(n=4000), seed=1)
        result = instrumental_variables_analysis(design)
        assert not result["first_stage"]["weak_instrument"]

    def test_itt_is_attenuated_toward_zero_by_non_compliance(self):
        """ITT answers a different question, and the dilution is the reason."""
        design = simulate_encouragement(make_scored(n=4000), seed=2)
        result = instrumental_variables_analysis(design)
        assert abs(result["intention_to_treat"]["coefficient"]) < \
            abs(result["two_stage_least_squares"]["coefficient"])

    def test_no_defiers_are_generated(self):
        """Monotonicity is an assumption; the simulation must honour it."""
        design = simulate_encouragement(make_scored(n=3000), seed=4)
        offered = design[design["offered"] == 1]["redeemed"].mean()
        not_offered = design[design["offered"] == 0]["redeemed"].mean()
        assert offered > not_offered


class TestDifferenceInDifferences:
    def test_recovers_the_effect_where_naive_comparisons_fail(self):
        panel = simulate_did_panel(make_scored(n=6000), seed=1)
        result = difference_in_differences(panel)
        truth = panel.attrs["true_effect"]

        assert result["did_estimate"] == pytest.approx(truth, abs=0.03)
        assert abs(result["naive_post_period_comparison"] - truth) > abs(
            result["did_estimate"] - truth
        )
        assert abs(result["naive_before_after_comparison"] - truth) > abs(
            result["did_estimate"] - truth
        )

    def test_regression_matches_the_two_by_two_table(self):
        """The interaction coefficient is the double difference, by construction."""
        panel = simulate_did_panel(make_scored(n=4000), seed=2)
        result = difference_in_differences(panel)
        # did_estimate is rounded for display; the regression term is not.
        assert result["regression"]["coefficient"] == pytest.approx(
            result["did_estimate"], abs=1e-4
        )

    def test_no_effect_yields_no_estimate(self):
        panel = simulate_did_panel(make_scored(n=6000), treatment_effect=0.0, seed=3)
        result = difference_in_differences(panel)
        assert result["did_estimate"] == pytest.approx(0.0, abs=0.03)
        assert not result["regression"]["significant"]

    def test_parallel_trends_hold_before_treatment(self):
        panel = simulate_did_panel(make_scored(n=6000), seed=4)
        assert parallel_trends_test(panel)["trends_parallel"]

    def test_broken_pre_trends_are_detected(self):
        """The check that stops DiD from crediting a pre-existing divergence."""
        rng = np.random.default_rng(9)
        rows = []
        for period in (-2, -1, 0):
            for group in (0, 1):
                # The treated group is already diverging, before anything happens.
                rate = 0.3 + (0.12 * group * (period + 2))
                n = 3000
                rows.append(pd.DataFrame({
                    "period": period, "post": int(period == 0), "treated_group": group,
                    "churned": (rng.random(n) < rate).astype(int),
                }))
        panel = pd.concat(rows, ignore_index=True)
        assert not parallel_trends_test(panel)["trends_parallel"]

    def test_single_pre_period_cannot_be_tested(self):
        panel = simulate_did_panel(make_scored(n=500), n_pre_periods=1, seed=5)
        assert not parallel_trends_test(panel)["testable"]
