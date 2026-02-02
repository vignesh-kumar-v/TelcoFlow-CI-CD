import pandas as pd
from pathlib import Path
import typer
from rich.console import Console

console = Console()

def load_raw_data(raw_path: Path):
    console.log(f"[bold blue] Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    console.log(f"[green]Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def validate_schema(df: pd.DataFrame):
    required_columns = {
        "customerID": "object",
        "gender": "object",
        "SeniorCitizen": "int64",
        "Partner": "object",
        "Dependents": "object",
        "tenure": "int64",
        "PhoneService": "object",
        "MultipleLines": "object",
        "InternetService": "object",
        "OnlineSecurity": "object",
        "OnlineBackup": "object",
        "DeviceProtection": "object",
        "TechSupport": "object",
        "StreamingTV": "object",
        "StreamingMovies": "object",
        "Contract": "object",
        "PaperlessBilling": "object",
        "PaymentMethod": "object",
        "MonthlyCharges": "float64",
        "TotalCharges": "object",
        "Churn": "object"
    }
    console.log("[bold blue]Validating schema...")
    missing_cols = set(required_columns.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for col, expected_dtype in required_columns.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            console.log(f"[yellow]Warning: {col} dtype is {actual_dtype}, expected {expected_dtype}")
    console.log("[green]Schema validation passed")

def clean_data(df: pd.DataFrame):
    console.log("[bold blue]Cleaning data...")
    df['TotalCharges'] = pd.to_numeric(df["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    median_charges = df["TotalCharges"].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_charges)
    console.log(f"[yellow]Filled {df['TotalCharges'].isna().sum()} missing TotalCharges with median {median_charges:.2f}")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    console.log("[green]Encoded target variable: Yes->1, No->0")
    console.log(f"[blue]Final shape: {df.shape}")
    console.log(f"[blue]Missing values per column:\n{df.isnull().sum().to_dict()}")
    return df

def save_clean_data(df: pd.DataFrame, processed_path: Path):
    console.log(f"[bold blue]Saving cleaned data to {processed_path}...")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    console.log(f"[green]Saved {len(df)} rows to {processed_path}")

def main(
    raw_path: Path=Path.cwd() / "data" / "raw" / "Telco-Customer-Churn.csv",
    processed_path: Path=Path.cwd() / "data" / "processed" / "cleaned_data.parquet"
    ):
    console.rule("[bold magenta]Telco churn: Data validation and cleaning")
    df = load_raw_data(raw_path)
    validate_schema(df)
    df_clean = clean_data(df)
    save_clean_data(df_clean, processed_path)
    console.rule("[bold green]Pipeline completed successfully!")

if __name__ == "__main__":
    typer.run(main)