"""API contract tests.

Schema validation and health reporting are exercised without a model; the
prediction path needs a trained artifact and is marked integration.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.telco_churn.api import app

VALID_CUSTOMER = {
    "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def has_model() -> bool:
    artifacts = Path("artifacts")
    return artifacts.exists() and any(
        d.is_dir() and d.name[0].isdigit() for d in artifacts.iterdir()
    )


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    """Liveness stays green whenever the process is up."""

    def test_health_is_200_regardless_of_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body

    def test_health_reports_model_presence(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is has_model()
        if has_model():
            assert body["model_version"]
            assert body["model_name"]


class TestReadiness:
    """Readiness must gate on the model, by status code."""

    def test_readiness_status_code_reflects_model(self, client):
        """A 200 body saying "no_model" would still be admitted to the Service.

        Regression test: readiness previously pointed at /health, which returns
        200 unconditionally, so modelless pods received live traffic.
        """
        response = client.get("/ready")
        if has_model():
            assert response.status_code == 200
            assert response.json()["status"] == "ready"
        else:
            assert response.status_code == 503

    def test_readiness_is_a_distinct_endpoint(self, client):
        assert client.get("/ready").status_code in {200, 503}
        assert client.get("/health").status_code == 200


class TestValidation:
    """Bad payloads must be rejected by the schema, before touching the model."""

    def test_missing_field_rejected(self, client):
        payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_tenure_rejected(self, client):
        assert client.post("/predict", json={**VALID_CUSTOMER, "tenure": -5}).status_code == 422

    def test_out_of_range_senior_citizen_rejected(self, client):
        assert client.post("/predict", json={**VALID_CUSTOMER, "SeniorCitizen": 7}).status_code == 422

    def test_negative_charges_rejected(self, client):
        assert client.post("/predict", json={**VALID_CUSTOMER, "MonthlyCharges": -1}).status_code == 422

    def test_wrong_type_rejected(self, client):
        assert client.post(
            "/predict", json={**VALID_CUSTOMER, "MonthlyCharges": "not-a-number"}
        ).status_code == 422


@pytest.mark.integration
class TestPrediction:
    def test_predict_returns_valid_response(self, client):
        if not has_model():
            pytest.skip("no trained model — run `make pipeline` first")
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["customerID"] == VALID_CUSTOMER["customerID"]
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_category"] in {"Low", "Medium", "High"}
        assert body["model_version"]

    def test_unseen_category_does_not_crash(self, client):
        """Serving must survive a category the model never saw in training."""
        if not has_model():
            pytest.skip("no trained model — run `make pipeline` first")
        response = client.post(
            "/predict", json={**VALID_CUSTOMER, "PaymentMethod": "Crypto wallet"}
        )
        assert response.status_code == 200, response.text
        assert 0.0 <= response.json()["churn_probability"] <= 1.0

    def test_model_info_reports_deployed_model(self, client):
        if not has_model():
            pytest.skip("no trained model — run `make pipeline` first")
        body = client.get("/model/info").json()
        assert body["model_version"]
        assert body["model_type"] in {"tree_native", "tree_codes", "linear"}
        assert "roc_auc" in body["metrics"]
        # SHAP importance is summarised, not dumped wholesale into the response.
        assert "shap_feature_importance" not in body["metrics"]
        assert isinstance(body["top_features"], list)
