# src/telco_churn/api.py
"""
FastAPI endpoint for real-time churn predictions
What it does: Accepts customer features, returns churn probability
Why it matters: Online inference for immediate decisions (e.g., during customer service call)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime
import typer
from rich.console import Console

from src.telco_churn.train import (
    prepare_data_lgbm, DROP_COLS, NOMINAL_FEATURES, ORDINAL_FEATURES,
)

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")
console = Console()

# Global variables for model and preprocessor
model = None
preprocessor = None
latest_run_dir = None


class CustomerFeatures(BaseModel):
    """Pydantic model for input validation

    Why: Automatically validates data types, ranges, and required fields
    Prevents bad data from crashing the API
    """
    customerID: str = Field(..., example="1234-ABCD")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=820.5)


@app.on_event("startup")
def load_model():
    """Load latest model on startup

    Why: Model loads once when API starts, not on every request
    Makes API responses fast (<100ms)
    """
    global model, preprocessor, latest_run_dir

    console.log("[blue]Loading latest model...")
    artifacts_dir = Path.cwd() / "artifacts"

    # Get latest run
    runs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not runs:
        raise RuntimeError("No trained models found")

    latest_run_dir = sorted(runs, key=lambda x: x.name, reverse=True)[0]

    # Load model and preprocessor
    model = joblib.load(latest_run_dir / "model.joblib")
    preprocessor = joblib.load(latest_run_dir / "preprocessor.joblib")

    console.log(f"[green]Model loaded from {latest_run_dir.name}")


@app.get("/health")
def health_check():
    """Health check endpoint

    Why: Load balancers and monitoring tools use this to check if API is alive
    """
    return {
        "status": "healthy",
        "model_version": latest_run_dir.name if latest_run_dir else "none",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=dict)
def predict_churn(features: CustomerFeatures):
    """
    Predict churn probability for a single customer

    Why: Real-time inference for immediate actions (e.g., offer retention discount during call)
    """
    try:
        # Convert Pydantic model to DataFrame
        df = pd.DataFrame([features.dict()])

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # Apply preprocessing (same as training!)
        X_processed, _ = prepare_data_lgbm(df, preprocessor)

        # Get prediction probability
        probability = model.predict_proba(X_processed)[0, 1]

        return {
            "customerID": features.customerID,
            "churn_probability": float(probability),
            "risk_category": "High" if probability > 0.6 else "Medium" if probability > 0.4 else "Low",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        # Log error and return meaningful message
        console.log(f"[red]Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
def model_info():
    """Get model metadata

    Why: Helps debug which model version is running in production
    """
    if not latest_run_dir:
        return {"error": "No model loaded"}

    metrics_path = latest_run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = pd.read_json(metrics_path, typ="series").to_dict()
    else:
        metrics = {}

    return {
        "model_version": latest_run_dir.name,
        "metrics": metrics,
        "loaded_at": datetime.now().isoformat()
    }


def main():
    """Launch API server"""
    import uvicorn

    console.rule("[bold magenta]Starting Telco Churn Prediction API")
    console.log(f"[blue]Model: {latest_run_dir.name if latest_run_dir else 'None'}")
    console.log(f"[blue]API docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    typer.run(main)
