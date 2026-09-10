"""Tests for the pre-registered power analysis.

The closed-form power formulas are the product here, so these check them against
known behaviour, against Monte Carlo simulation of the same estimator, and
against the A/B harness that consumes them.
"""

import json

import pytest

from src.telco_churn.ab_test import minimum_detectable_effect, two_proportion_z_test
from src.telco_churn.power import (
    ALPHA, PreRegistration, holm_bonferroni, interpret_result,
    mde_for_sample_size, power_for_sample_size, required_sample_size_means,
    required_sample_size_proportions, simulate_power, _z_test_p_value,
)


def make_prereg(available, required=1000, mde=0.05, baseline=0.265):
    return PreRegistration(
        experiment="test", primary_metric="retention rate", baseline_rate=baseline,
        mde_absolute=mde, alpha=ALPHA, power_target=0.8,
        required_n_per_arm=required, available_n_per_arm=available,
    )


class TestSampleSize:
    def test_smaller_effects_need_more_customers(self):
        """The core relationship: halving the MDE roughly quadruples the sample."""
        big = required_sample_size_proportions(0.3, 0.10)
        small = required_sample_size_proportions(0.3, 0.05)
        assert small > big
        assert 3.5 < small / big < 4.5

    def test_higher_power_needs_more_customers(self):
        assert (required_sample_size_proportions(0.3, 0.05, power=0.9)
                > required_sample_size_proportions(0.3, 0.05, power=0.8))

    def test_stricter_alpha_needs_more_customers(self):
        assert (required_sample_size_proportions(0.3, 0.05, alpha=0.01)
                > required_sample_size_proportions(0.3, 0.05, alpha=0.05))

    def test_matches_the_textbook_case(self):
        """0.5 -> 0.6 at 80% power and alpha 0.05 is a standard worked example."""
        n = required_sample_size_proportions(0.5, 0.1, alpha=0.05, power=0.8)
        assert 380 <= n <= 410

    def test_rejects_impossible_inputs(self):
        with pytest.raises(ValueError):
            required_sample_size_proportions(0.0, 0.05)
        with pytest.raises(ValueError):
            required_sample_size_proportions(0.3, 0.0)

    def test_continuous_metric_scales_with_variance(self):
        assert (required_sample_size_means(sd=200, mde=50)
                > required_sample_size_means(sd=100, mde=50))


class TestPowerAndMde:
    def test_power_rises_with_sample_size(self):
        powers = [power_for_sample_size(n, 0.265, 0.05) for n in (100, 500, 2000, 10000)]
        assert powers == sorted(powers)
        assert powers[-1] > 0.99

    def test_required_n_delivers_the_target_power(self):
        """The two formulas must be consistent, or one of them is wrong."""
        n = required_sample_size_proportions(0.265, 0.05, power=0.8)
        assert power_for_sample_size(n, 0.265, 0.05) == pytest.approx(0.8, abs=0.02)

    def test_mde_inverts_the_power_calculation(self):
        n = 1000
        mde = mde_for_sample_size(n, 0.3)
        assert power_for_sample_size(n, 0.3, mde) == pytest.approx(0.8, abs=0.03)

    def test_agrees_with_the_ab_harness_mde(self):
        """ab_test reports an MDE too; it must be this same formula, not a copy.

        The harness rounds to four places for display, hence the tolerance.
        """
        for n, baseline in ((300, 0.265), (1000, 0.4), (2500, 0.15)):
            assert (mde_for_sample_size(n, baseline)
                    == pytest.approx(minimum_detectable_effect(n, baseline), abs=1e-4))


class TestSimulationAgreement:
    """The analytic formula is an approximation. It has to match the real test."""

    @pytest.mark.parametrize("n,baseline,mde", [
        (300, 0.265, 0.05), (1000, 0.3, 0.05), (500, 0.5, 0.10),
    ])
    def test_analytic_power_matches_monte_carlo(self, n, baseline, mde):
        analytic = power_for_sample_size(n, baseline, mde)
        simulated = simulate_power(n, baseline, mde, n_simulations=3000, seed=7)
        assert abs(analytic - simulated) < 0.05, (
            f"analytic {analytic:.3f} vs simulated {simulated:.3f}"
        )

    def test_internal_z_test_matches_the_ab_harness(self):
        """The simulator carries its own z-test; it must be the same test."""
        for args in ((60, 100, 50, 100), (150, 400, 120, 400), (9, 50, 4, 50)):
            assert (_z_test_p_value(*args)
                    == pytest.approx(two_proportion_z_test(*args)["p_value"], abs=1e-6))

    def test_no_effect_rejects_at_about_alpha(self):
        """With no effect injected, rejection rate is the false-positive rate."""
        rate = simulate_power(500, 0.3, 0.0, n_simulations=3000, seed=11)
        assert rate < 0.09, f"false-positive rate {rate:.3f} well above alpha"


class TestHolmBonferroni:
    def test_smallest_p_gets_the_strictest_threshold(self):
        result = holm_bonferroni({"a": 0.001, "b": 0.04})
        assert result["a"]["adjusted_threshold"] == pytest.approx(0.025)
        assert result["b"]["adjusted_threshold"] == pytest.approx(0.05)

    def test_stops_rejecting_after_the_first_failure(self):
        """Holm is a step-down procedure: one failure ends the sequence."""
        result = holm_bonferroni({"a": 0.03, "b": 0.04})
        assert not result["a"]["significant_after_correction"]
        assert not result["b"]["significant_after_correction"]

    def test_is_less_conservative_than_bonferroni(self):
        result = holm_bonferroni({"a": 0.02, "b": 0.9, "c": 0.95})
        # Bonferroni would need p < 0.0167; Holm's first threshold is the same,
        # but the second and third are looser.
        assert result["b"]["adjusted_threshold"] > 0.05 / 3

    def test_strong_result_survives_correction(self):
        result = holm_bonferroni({"retention": 0.0035, "revenue": 0.1036})
        assert result["retention"]["significant_after_correction"]
        assert not result["revenue"]["significant_after_correction"]


class TestPreRegistration:
    def test_under_powered_study_is_flagged(self):
        prereg = make_prereg(available=292, required=1292)
        assert not prereg.is_adequately_powered
        assert prereg.achieved_power < 0.5

    def test_adequately_powered_study_is_flagged(self):
        prereg = make_prereg(available=1500, required=1292)
        assert prereg.is_adequately_powered
        assert prereg.achieved_power > 0.8

    def test_detectable_mde_exceeds_the_planned_one_when_under_powered(self):
        prereg = make_prereg(available=292, required=1292, mde=0.05)
        assert prereg.detectable_mde > prereg.mde_absolute

    def test_round_trips_through_disk(self, tmp_path):
        original = make_prereg(available=292)
        original.write(tmp_path)
        restored = PreRegistration.load(tmp_path)
        assert restored.required_n_per_arm == original.required_n_per_arm
        assert restored.baseline_rate == original.baseline_rate

    def test_missing_registration_returns_none(self, tmp_path):
        assert PreRegistration.load(tmp_path) is None

    def test_serialised_form_carries_the_commitments(self, tmp_path):
        """A pre-registration without a decision rule is not one."""
        payload = json.loads((make_prereg(292).write(tmp_path)).read_text())
        for key in ("hypothesis", "decision_rule", "stopping_rule", "analysis_plan"):
            assert key in payload
        assert "interim" in payload["stopping_rule"].lower()


class TestInterpretation:
    """The distinction the whole module exists to draw."""

    def test_non_significant_and_under_powered_is_inconclusive(self):
        result = interpret_result(make_prereg(292, 1292), observed_effect=0.02, p_value=0.4)
        assert result["verdict"] == "inconclusive"
        assert "not a null result" in result["reading"]

    def test_non_significant_and_well_powered_is_a_null(self):
        result = interpret_result(make_prereg(2000, 1292), observed_effect=0.002, p_value=0.8)
        assert result["verdict"] == "null"
        assert "evidence against" in result["reading"]

    def test_significant_result_in_a_small_study_warns_about_inflation(self):
        result = interpret_result(make_prereg(292, 1292), observed_effect=0.116, p_value=0.003)
        assert result["verdict"] == "significant"
        assert "exaggerate" in result["reading"]

    def test_significant_and_powered_carries_no_caveat(self):
        result = interpret_result(make_prereg(2000, 1292), observed_effect=0.06, p_value=0.001)
        assert result["verdict"] == "significant"
        assert "exaggerate" not in result["reading"]
