"""FastAPI endpoint for real-time churn predictions.

Online inference for immediate decisions — e.g. deciding whether to offer a
retention discount while a customer is still on the phone.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from src.telco_churn.inference import LoadedModel, risk_category

console = Console()

# Populated during startup; None means the app is up but has no model to serve.
loaded_model: LoadedModel | None = None


def _load_model() -> "LoadedModel | None":
    """Load the newest model, returning None when none exists yet.

    Retried on demand rather than only at startup: in Kubernetes the API pods
    come up in parallel with the training Job, so the artifact volume is often
    still empty at boot. Without the retry those pods would never serve until
    someone restarted them.
    """
    global loaded_model
    if loaded_model is not None:
        return loaded_model
    try:
        loaded_model = LoadedModel.load_latest(Path.cwd() / "artifacts")
        console.log(
            f"[green]Model loaded from {loaded_model.version} "
            f"({loaded_model.model_name}, {loaded_model.model_type})"
        )
    except FileNotFoundError as exc:
        loaded_model = None
        console.log(f"[yellow]No model available yet: {exc}")
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup rather than on every request.

    A missing model is not fatal: the app still starts so the readiness probe
    can hold the pod out of the load balancer, instead of crash-looping.
    """
    global loaded_model
    console.log("[blue]Loading latest model...")
    _load_model()
    yield
    loaded_model = None


app = FastAPI(title="Telco Churn Prediction API", version="1.0.0", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    """Input contract for a single prediction.

    Pydantic validates types and ranges before anything reaches the model, so
    malformed requests fail with a 422 instead of an opaque 500.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No", "tenure": 1,
            "PhoneService": "No", "MultipleLines": "No phone service",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85, "TotalCharges": 29.85,
        }
    })

    customerID: str
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    customerID: str
    churn_probability: float
    risk_category: str
    model_version: str
    timestamp: str


def _require_model() -> LoadedModel:
    model = _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model available. Run `make train` to produce one.",
        )
    return model


@app.get("/health")
def health_check():
    """Liveness probe: is the process up?

    Always 200 while the app is running. Deliberately does not consider the
    model — a liveness failure restarts the container, and restarting will not
    conjure a model that has not been trained yet.
    """
    model = loaded_model
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model.version if model else None,
        "model_name": model.model_name if model else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/ready")
def readiness_check():
    """Readiness probe: can this pod actually serve predictions?

    Returns 503 without a model, so Kubernetes keeps the pod out of the Service
    endpoints instead of routing traffic that can only fail. Probes look at the
    status code, not the body — a 200 saying "no_model" would still be admitted.
    """
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded yet")
    return {
        "status": "ready",
        "model_version": model.version,
        "model_name": model.model_name,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(features: CustomerFeatures):
    """Churn probability for a single customer.

    Feature transformation is driven by the model type recorded at training
    time, so this path is identical to batch scoring regardless of which
    algorithm won.
    """
    model = _require_model()
    try:
        df = pd.DataFrame([features.model_dump()])
        probability = float(model.predict_proba(df)[0])
    except Exception as exc:
        console.log(f"[red]Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        customerID=features.customerID,
        churn_probability=probability,
        risk_category=risk_category(probability),
        model_version=model.version,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model/info")
def model_info():
    """Which model version is live, and how it scored — useful when debugging prod."""
    model = _require_model()
    metrics = model.metrics()
    return {
        "model_version": model.version,
        "model_name": model.model_name,
        "model_type": model.model_type,
        "n_training_samples": model.train_info.get("n_samples"),
        "features": model.train_info.get("feature_names", []),
        "metrics": {k: v for k, v in metrics.items() if k != "shap_feature_importance"},
        "top_features": list(metrics.get("shap_feature_importance", {}))[:10],
        "loaded_at": datetime.now().isoformat(),
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Launch the API server."""
    import uvicorn

    console.rule("[bold magenta]Starting Telco Churn Prediction API")
    console.log(f"[blue]API docs: http://localhost:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
