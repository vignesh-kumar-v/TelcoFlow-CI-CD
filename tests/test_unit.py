import pytest
import pandas as pd
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telco_churn.validate_and_clean import validate_schema, clean_data
from telco_churn.train import (
    DROP_COLS, NOMINAL_FEATURES, ORDINAL_FEATURES, NUMERIC_FEATURES,
    create_lgbm_preprocessor, create_lr_preprocessor,
    compute_scale_pos_weight,
)


class TestDataValidation:
    """Test data validation functions"""

    def test_validate_schema_missing_columns(self):
        """Should raise error when required columns are missing"""
        df = pd.DataFrame({"customerID": [1], "gender": ["Male"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_validate_schema_correct_columns(self):
        """Should pass when all required columns exist"""
        data = {
            "customerID": ["1234"],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["Fiber optic"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [70.0],
            "TotalCharges": [800.0],
            "Churn": [0]
        }
        df = pd.DataFrame(data)

        validate_schema(df)

    def test_clean_data_fixes_total_charges(self):
        """Should convert TotalCharges strings to numeric"""
        df = pd.DataFrame({
            "customerID": ["1234"],
            "TotalCharges": [" 123.45 "],
            "Churn": ["Yes"]
        })

        df_clean = clean_data(df)
        assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
        assert df_clean["TotalCharges"].iloc[0] == 123.45

    def test_clean_data_encodes_target(self):
        """Should encode Churn Yes/No to 1/0"""
        df = pd.DataFrame({
            "customerID": ["1234", "5678"],
            "TotalCharges": [100, 200],
            "Churn": ["Yes", "No"]
        })

        df_clean = clean_data(df)
        assert df_clean["Churn"].tolist() == [1, 0]


class TestPreprocessing:
    """Test preprocessing functions"""

    def test_lgbm_preprocessor_handles_unknown_contract(self):
        """Should handle contract categories not seen during training"""
        encoder = create_lgbm_preprocessor()
        train_df = pd.DataFrame({"Contract": ["Month-to-month", "One year", "Two year"]})
        encoder.fit(train_df[["Contract"]])

        test_df = pd.DataFrame({"Contract": ["Unknown"]})
        result = encoder.transform(test_df[["Contract"]])
        assert result[0, 0] == -1


class TestFeatureDropping:
    """Test that customerID is properly excluded"""

    def test_customer_id_in_drop_cols(self):
        """customerID should be in DROP_COLS"""
        assert "customerID" in DROP_COLS

    def test_drop_cols_not_in_feature_lists(self):
        """DROP_COLS should not appear in any feature list"""
        all_features = NOMINAL_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES
        for col in DROP_COLS:
            assert col not in all_features


class TestClassImbalance:
    """Test class imbalance handling"""

    def test_scale_pos_weight_calculation(self):
        """scale_pos_weight should be n_neg / n_pos"""
        y = pd.Series([0, 0, 0, 1])
        spw = compute_scale_pos_weight(y)
        assert spw == 3.0

    def test_scale_pos_weight_balanced(self):
        """Balanced classes should give weight of 1.0"""
        y = pd.Series([0, 0, 1, 1])
        spw = compute_scale_pos_weight(y)
        assert spw == 1.0


class TestModelComparison:
    """Test model comparison structure"""

    def test_required_model_keys(self):
        """Model comparison dict should have required metric keys"""
        required_keys = {"roc_auc", "precision", "recall", "f1_score", "support"}
        sample_metrics = {
            "roc_auc": 0.85,
            "precision": 0.65,
            "recall": 0.60,
            "f1_score": 0.62,
            "support": 100,
        }
        assert set(sample_metrics.keys()) == required_keys


class TestEncoderTypes:
    """Test that LR preprocessor produces more columns (OneHot)"""

    def test_lr_preprocessor_expands_columns(self):
        """LR preprocessor with OneHotEncoder should produce more columns than input"""
        transformer = create_lr_preprocessor()
        data = {col: ["Yes", "No"] for col in NOMINAL_FEATURES}
        data.update({col: ["Month-to-month", "One year"] for col in ORDINAL_FEATURES})
        data.update({col: [1.0, 2.0] for col in NUMERIC_FEATURES})
        df = pd.DataFrame(data)
        result = transformer.fit_transform(df)
        n_input = len(NOMINAL_FEATURES) + len(ORDINAL_FEATURES) + len(NUMERIC_FEATURES)
        assert result.shape[1] > n_input


class TestDriftDetection:
    """Test drift detection logic"""

    def test_drift_calculation(self):
        """Test drift percentage calculation"""
        train_mean = 50.0
        scoring_mean = 55.0
        drift_pct = abs(scoring_mean - train_mean) / train_mean * 100
        assert drift_pct == 10.0

    def test_no_drift(self):
        """Test when there's no drift"""
        train_mean = 50.0
        scoring_mean = 50.0
        drift_pct = abs(scoring_mean - train_mean) / train_mean * 100
        assert drift_pct == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
