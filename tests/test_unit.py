"""Unit tests: no artifacts, no I/O, safe to run on a cold checkout."""

import numpy as np
import pandas as pd
import pytest

from src.telco_churn.batch_score import simulate_drift
from src.telco_churn.features import (
    DROP_COLS, LINEAR, MODEL_TYPE_BY_NAME, NOMINAL_FEATURES, NUMERIC_FEATURES,
    ORDINAL_FEATURES, TREE_CODES, TREE_NATIVE, FeaturePipeline, active_features,
    create_lgbm_preprocessor, create_lr_preprocessor, drop_non_features,
)
from src.telco_churn.inference import risk_category
from src.telco_churn.train import _coerce_binary_shap, compute_scale_pos_weight
from src.telco_churn.validate_and_clean import clean_data, validate_schema


def _cycle(values, n):
    """Repeat `values` to exactly length n, so every column lines up."""
    return [values[i % len(values)] for i in range(n)]


def make_frame(n=6):
    """Minimal frame carrying every feature the pipeline expects.

    n must be a multiple of 6 so each categorical column sees all its levels.
    """
    assert n % 6 == 0, "n must be a multiple of 6"
    rng = np.random.default_rng(0)
    data = {col: rng.choice(["Yes", "No"], n).tolist() for col in NOMINAL_FEATURES}
    data["MultipleLines"] = _cycle(["No", "Yes", "No phone service"], n)
    data["InternetService"] = _cycle(["DSL", "Fiber optic", "No"], n)
    data["Contract"] = _cycle(["Month-to-month", "One year", "Two year"], n)
    data["SeniorCitizen"] = _cycle([0, 1], n)
    data["tenure"] = list(range(1, n + 1))
    data["MonthlyCharges"] = [50.0 + i for i in range(n)]
    data["TotalCharges"] = [500.0 + 10 * i for i in range(n)]
    data["customerID"] = [f"ID-{i}" for i in range(n)]
    data["Churn"] = _cycle([0, 1], n)
    return pd.DataFrame(data)


class TestDataValidation:
    def test_validate_schema_missing_columns(self):
        df = pd.DataFrame({"customerID": [1], "gender": ["Male"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_validate_schema_correct_columns(self):
        validate_schema(make_frame())

    def test_clean_data_fixes_total_charges(self):
        df = pd.DataFrame({"customerID": ["1234"], "TotalCharges": [" 123.45 "], "Churn": ["Yes"]})
        df_clean = clean_data(df)
        assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
        assert df_clean["TotalCharges"].iloc[0] == 123.45

    def test_clean_data_encodes_target(self):
        df = pd.DataFrame({
            "customerID": ["1234", "5678"],
            "TotalCharges": [100, 200],
            "Churn": ["Yes", "No"],
        })
        assert clean_data(df)["Churn"].tolist() == [1, 0]

    def test_blank_total_charges_filled_with_median(self):
        """The 11 tenure=0 rows arrive as blank strings, not NaN."""
        df = pd.DataFrame({
            "customerID": list("abcd"),
            "TotalCharges": ["100", "200", " ", "400"],
            "Churn": ["No"] * 4,
        })
        df_clean = clean_data(df)
        assert df_clean["TotalCharges"].isna().sum() == 0
        assert df_clean["TotalCharges"].iloc[2] == 200.0  # median of 100/200/400


class TestPreprocessing:
    def test_lgbm_preprocessor_handles_unknown_contract(self):
        encoder = create_lgbm_preprocessor()
        encoder.fit(pd.DataFrame({"Contract": ["Month-to-month", "One year", "Two year"]}))
        assert encoder.transform(pd.DataFrame({"Contract": ["Unknown"]}))[0, 0] == -1

    def test_lr_preprocessor_expands_columns(self):
        transformer = create_lr_preprocessor()
        df = make_frame()
        result = transformer.fit_transform(df)
        n_input = len(NOMINAL_FEATURES) + len(ORDINAL_FEATURES) + len(NUMERIC_FEATURES)
        assert result.shape[1] > n_input

    def test_drop_non_features_removes_id_and_target(self):
        X = drop_non_features(make_frame())
        assert "customerID" not in X.columns
        assert "Churn" not in X.columns

    def test_active_features_partitions_columns(self):
        nominal, ordinal, numeric = active_features(drop_non_features(make_frame()))
        assert set(nominal) == set(NOMINAL_FEATURES)
        assert ordinal == ORDINAL_FEATURES
        assert set(numeric) == set(NUMERIC_FEATURES)


class TestFeaturePipeline:
    """The pipeline must produce identical encodings for train and serve."""

    def test_category_codes_are_stable_across_batches(self):
        """A batch missing a category must not shift the remaining codes.

        Regression test: `astype("category")` infers levels per-frame, so a
        scoring batch without "No phone service" used to renumber every other
        level and silently feed the model wrong values.
        """
        train = drop_non_features(make_frame())
        pipeline = FeaturePipeline().fit(train)

        full = pipeline.transform_tree_codes(train)
        code_for_yes = full.loc[train["MultipleLines"] == "Yes", "MultipleLines"].iloc[0]

        # Same rows, minus every "No phone service" record.
        subset = train[train["MultipleLines"] != "No phone service"]
        partial = pipeline.transform_tree_codes(subset)
        assert partial.loc[subset["MultipleLines"] == "Yes", "MultipleLines"].iloc[0] == code_for_yes

    def test_unseen_category_becomes_missing_not_a_wrong_code(self):
        train = drop_non_features(make_frame())
        pipeline = FeaturePipeline().fit(train)
        novel = train.copy()
        novel.loc[:, "PaymentMethod"] = "Crypto"
        assert pipeline.transform_tree_codes(novel)["PaymentMethod"].eq(-1).all()

    def test_transform_for_dispatches_by_model_type(self):
        train = drop_non_features(make_frame())
        pipeline = FeaturePipeline().fit(train)

        native = pipeline.transform_for(train, TREE_NATIVE)
        assert isinstance(native["gender"].dtype, pd.CategoricalDtype)

        codes = pipeline.transform_for(train, TREE_CODES)
        assert pd.api.types.is_integer_dtype(codes["gender"])

        linear = pipeline.transform_for(train, LINEAR)
        assert isinstance(linear, np.ndarray)
        assert linear.shape[1] > train.shape[1]

    def test_unknown_model_type_rejected(self):
        pipeline = FeaturePipeline().fit(drop_non_features(make_frame()))
        with pytest.raises(ValueError, match="Unknown model_type"):
            pipeline.transform_for(drop_non_features(make_frame()), "nonsense")

    def test_column_order_does_not_change_encoding(self):
        train = drop_non_features(make_frame())
        pipeline = FeaturePipeline().fit(train)
        shuffled = train[list(reversed(train.columns.tolist()))]
        pd.testing.assert_frame_equal(
            pipeline.transform_tree_codes(train), pipeline.transform_tree_codes(shuffled)
        )

    def test_missing_feature_raises(self):
        train = drop_non_features(make_frame())
        pipeline = FeaturePipeline().fit(train)
        with pytest.raises(ValueError, match="missing trained features"):
            pipeline.transform_tree_native(train.drop(columns=["tenure"]))


class TestModelTypeRouting:
    def test_every_candidate_has_a_model_type(self):
        """Any model the trainer can pick must have a serving shape defined."""
        expected = {"logistic_regression", "lightgbm", "tuned_lightgbm", "xgboost"}
        assert expected == set(MODEL_TYPE_BY_NAME)

    def test_model_types_are_known_values(self):
        assert set(MODEL_TYPE_BY_NAME.values()) <= {TREE_NATIVE, TREE_CODES, LINEAR}


class TestFeatureDropping:
    def test_customer_id_in_drop_cols(self):
        assert "customerID" in DROP_COLS

    def test_drop_cols_not_in_feature_lists(self):
        all_features = NOMINAL_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES
        for col in DROP_COLS:
            assert col not in all_features


class TestClassImbalance:
    def test_scale_pos_weight_calculation(self):
        assert compute_scale_pos_weight(pd.Series([0, 0, 0, 1])) == 3.0

    def test_scale_pos_weight_balanced(self):
        assert compute_scale_pos_weight(pd.Series([0, 0, 1, 1])) == 1.0


class TestShapCoercion:
    """SHAP returns per-class values in different shapes by version and model."""

    def test_list_of_arrays_takes_positive_class(self):
        values = [np.zeros((3, 2)), np.ones((3, 2))]
        assert np.array_equal(_coerce_binary_shap(values), np.ones((3, 2)))

    def test_three_dim_array_takes_positive_class(self):
        values = np.stack([np.zeros((3, 2)), np.ones((3, 2))], axis=-1)
        assert np.array_equal(_coerce_binary_shap(values), np.ones((3, 2)))

    def test_two_dim_array_passes_through(self):
        values = np.ones((3, 2))
        assert np.array_equal(_coerce_binary_shap(values), values)


class TestRiskCategory:
    @pytest.mark.parametrize("prob,expected", [
        (0.95, "High"), (0.61, "High"), (0.6, "Medium"),
        (0.5, "Medium"), (0.41, "Medium"), (0.4, "Low"), (0.01, "Low"),
    ])
    def test_boundaries(self, prob, expected):
        assert risk_category(prob) == expected


class TestDriftSimulation:
    def test_simulation_actually_shifts_distributions(self):
        df = make_frame(12)
        drifted = simulate_drift(df)
        assert drifted["MonthlyCharges"].mean() > df["MonthlyCharges"].mean()
        assert drifted["tenure"].mean() < df["tenure"].mean()

    def test_simulation_is_deterministic(self):
        df = make_frame(12)
        pd.testing.assert_frame_equal(simulate_drift(df, seed=7), simulate_drift(df, seed=7))

    def test_simulation_does_not_mutate_input(self):
        df = make_frame(12)
        before = df.copy()
        simulate_drift(df)
        pd.testing.assert_frame_equal(df, before)


if __name__ == "__main__":
    pytest.main([__file__])
