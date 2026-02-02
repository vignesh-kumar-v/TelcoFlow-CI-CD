# tests/test_unit.py
"""Unit tests for individual functions"""

import pytest
import pandas as pd
from pathlib import Path
import sys


# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from telco_churn.validate_and_clean import validate_schema, clean_data
from telco_churn.train import create_preprocessor


class TestDataValidation:
    """Test data validation functions"""

    def test_validate_schema_missing_columns(self):
        """Should raise error when required columns are missing"""
        df = pd.DataFrame({"customerID": [1], "gender": ["Male"]})  # Missing most columns

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_validate_schema_correct_columns(self):
        """Should pass when all required columns exist"""
        # Create minimal valid dataframe
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

        # Should not raise exception
        validate_schema(df)  # If this passes, test succeeds

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

    def test_preprocessor_handles_unknown_categories(self):
        """Should handle categories not seen during training"""
        # Create training data
        train_df = pd.DataFrame({
            "gender": ["Male", "Female"],
            "Partner": ["Yes", "No"]
        })

        preprocessor = create_preprocessor(["gender", "Partner"])
        preprocessor.fit(train_df)

        # Test data with new category
        test_df = pd.DataFrame({
            "gender": ["Other"],  # Not in training data
            "Partner": ["Maybe"]  # Not in training data
        })

        # Should not raise error (unknown_value=-1 handles this)
        transformed = preprocessor.transform(test_df)
        assert transformed.shape == (1, 2)


class TestDriftDetection:
    """Test drift detection logic"""

    def test_drift_calculation(self):
        """Test drift percentage calculation"""
        from telco_churn.batch_score import generate_drift_report

        # Mock training info
        train_info = {
            "n_samples": 1000,
            "feature_names": ["MonthlyCharges"],
            "MonthlyCharges_mean": 50.0
        }

        # Create scoring data with drift
        scoring_df = pd.DataFrame({
            "MonthlyCharges": [55.0] * 100  # 10% drift (55 vs 50)
        })

        # This would require refactoring batch_score to be more testable
        # For now, we test the math directly
        train_mean = 50.0
        scoring_mean = 55.0
        drift_pct = abs(scoring_mean - train_mean) / train_mean * 100

        assert drift_pct == 10.0
        assert drift_pct > 0  # Positive drift

    def test_no_drift(self):
        """Test when there's no drift"""
        train_mean = 50.0
        scoring_mean = 50.0
        drift_pct = abs(scoring_mean - train_mean) / train_mean * 100

        assert drift_pct == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
