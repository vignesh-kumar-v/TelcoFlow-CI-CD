"""Tests for the A/B test harness.

The statistics are the product here, so these check the estimators against
known answers and verify the harness does not manufacture effects.
"""

import numpy as np
import pandas as pd
import pytest

from src.telco_churn.ab_test import (
    ALPHA, assign_groups, chi_square_test, minimum_detectable_effect,
    run_experiment, simulate_outcomes, two_proportion_z_test,
)


def make_scored(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n)],
        "churn_probability": rng.uniform(0.5, 0.95, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
    })


class TestTwoProportionZTest:
    def test_identical_proportions_give_no_effect(self):
        result = two_proportion_z_test(50, 100, 50, 100)
        assert result["absolute_difference"] == 0
        assert result["z_statistic"] == 0
        assert result["p_value"] == 1.0
        assert not result["significant"]

    def test_large_difference_is_significant(self):
        result = two_proportion_z_test(90, 100, 40, 100)
        assert result["significant"]
        assert result["p_value"] < 0.001
        assert result["absolute_difference"] == pytest.approx(0.5)

    def test_confidence_interval_brackets_the_difference(self):
        result = two_proportion_z_test(60, 100, 40, 100)
        low, high = result["ci_95"]
        assert low < result["absolute_difference"] < high

    def test_significant_result_excludes_zero_from_ci(self):
        """A significant p-value and a CI containing zero would contradict."""
        result = two_proportion_z_test(90, 100, 40, 100)
        low, high = result["ci_95"]
        assert low > 0 or high < 0

    def test_nonsignificant_result_includes_zero_in_ci(self):
        result = two_proportion_z_test(51, 100, 50, 100)
        low, high = result["ci_95"]
        assert low <= 0 <= high

    def test_cohens_h_sign_follows_the_difference(self):
        assert two_proportion_z_test(70, 100, 50, 100)["cohens_h"] > 0
        assert two_proportion_z_test(50, 100, 70, 100)["cohens_h"] < 0

    def test_larger_sample_shrinks_the_interval(self):
        small = two_proportion_z_test(60, 100, 50, 100)["ci_95"]
        large = two_proportion_z_test(600, 1000, 500, 1000)["ci_95"]
        assert (large[1] - large[0]) < (small[1] - small[0])


class TestChiSquare:
    def test_agrees_with_z_test_on_significance(self):
        """For a 2x2 table the two tests are equivalent; they must not disagree."""
        df = pd.DataFrame({
            "group": ["treatment"] * 200 + ["control"] * 200,
            "churned": [0] * 150 + [1] * 50 + [0] * 90 + [1] * 110,
        })
        chi = chi_square_test(df)
        z = two_proportion_z_test(150, 200, 90, 200)
        assert chi["significant"] == z["significant"]
        assert chi["p_value"] == pytest.approx(z["p_value"], abs=1e-3)

    def test_cramers_v_within_unit_range(self):
        df = pd.DataFrame({
            "group": ["treatment"] * 100 + ["control"] * 100,
            "churned": [0] * 60 + [1] * 40 + [0] * 50 + [1] * 50,
        })
        assert 0 <= chi_square_test(df)["cramers_v"] <= 1


class TestAssignment:
    def test_arms_are_balanced_in_size(self):
        assigned = assign_groups(make_scored(), risk_threshold=0.5, seed=1)
        counts = assigned["group"].value_counts()
        assert abs(counts["treatment"] - counts["control"]) <= 10

    def test_stratification_balances_baseline_risk(self):
        """Both arms must carry the same baseline risk, or the readout is confounded."""
        assigned = assign_groups(make_scored(n=1200), risk_threshold=0.5, seed=2)
        means = assigned.groupby("group")["churn_probability"].mean()
        assert abs(means["treatment"] - means["control"]) < 0.02

    def test_threshold_filters_population(self):
        assigned = assign_groups(make_scored(), risk_threshold=0.8, seed=3)
        assert (assigned["churn_probability"] >= 0.8).all()

    def test_empty_target_population_raises(self):
        with pytest.raises(ValueError, match="No customers scored"):
            assign_groups(make_scored(), risk_threshold=0.999999, seed=4)

    def test_assignment_is_reproducible(self):
        a = assign_groups(make_scored(), 0.5, seed=7)["group"].tolist()
        b = assign_groups(make_scored(), 0.5, seed=7)["group"].tolist()
        assert a == b


class TestSimulation:
    def test_treatment_reduces_churn_when_effect_injected(self):
        assigned = assign_groups(make_scored(n=3000), 0.5, seed=5)
        outcomes = simulate_outcomes(assigned, effect=0.5, offer_cost=0, seed=5)
        rates = outcomes.groupby("group")["churned"].mean()
        assert rates["treatment"] < rates["control"]

    def test_zero_effect_leaves_arms_equivalent(self):
        assigned = assign_groups(make_scored(n=3000), 0.5, seed=6)
        outcomes = simulate_outcomes(assigned, effect=0.0, offer_cost=0, seed=6)
        rates = outcomes.groupby("group")["churned"].mean()
        assert abs(rates["treatment"] - rates["control"]) < 0.05

    def test_offer_cost_only_charged_to_treatment(self):
        assigned = assign_groups(make_scored(), 0.5, seed=8)
        outcomes = simulate_outcomes(assigned, effect=0.2, offer_cost=50.0, seed=8)
        gap = outcomes["revenue"] - outcomes["revenue_net"]
        assert (gap[outcomes["group"] == "treatment"] == 50.0).all()
        assert (gap[outcomes["group"] == "control"] == 0.0).all()

    def test_retained_is_complement_of_churned(self):
        assigned = assign_groups(make_scored(), 0.5, seed=9)
        outcomes = simulate_outcomes(assigned, effect=0.2, offer_cost=0, seed=9)
        assert (outcomes["retained"] + outcomes["churned"] == 1).all()


class TestStatisticalIntegrity:
    """The harness must find effects only when effects exist."""

    def test_aa_false_positive_rate_near_alpha(self):
        """Across many seeds with no injected effect, ~alpha should be significant.

        A materially higher rate would mean the harness fabricates effects.
        """
        assigned = assign_groups(make_scored(n=800), 0.5, seed=11)
        significant = sum(
            run_experiment(assigned, effect=0.0, offer_cost=0, seed=s)[1]["retention"]["significant"]
            for s in range(120)
        )
        false_positive_rate = significant / 120
        assert false_positive_rate < 0.15, (
            f"A/A significant {false_positive_rate:.1%} of the time; expected ~{ALPHA:.0%}"
        )

    def test_large_effect_is_reliably_detected(self):
        assigned = assign_groups(make_scored(n=1500), 0.5, seed=12)
        detected = sum(
            run_experiment(assigned, effect=0.5, offer_cost=0, seed=s)[1]["retention"]["significant"]
            for s in range(20)
        )
        assert detected >= 18, f"only detected a large effect {detected}/20 times"


class TestPower:
    def test_mde_shrinks_with_sample_size(self):
        assert minimum_detectable_effect(2000, 0.3) < minimum_detectable_effect(200, 0.3)

    def test_mde_is_a_positive_proportion(self):
        mde = minimum_detectable_effect(500, 0.3)
        assert 0 < mde < 1
