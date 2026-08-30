"""BigQuery load and query layer.

Moves the cleaned dataset off local disk and into a warehouse, so feature reads
are a query rather than a parquet file. The local SQLite/Postgres path and this
one produce the same columns; which one feeds training is a deployment choice.

    export GCP_PROJECT=your-project-id
    gcloud auth application-default login

    make bq-load          # cleaned data -> BigQuery
    make bq-features      # query engineered features back out
    make bq-analytics     # churn-by-segment aggregation, computed in BigQuery

Every entry point requires GCP_PROJECT; nothing here contacts Google unless it
is set, so the rest of the pipeline runs unchanged without cloud credentials.
"""

import os
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.telco_churn.features import (
    NOMINAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES, TARGET,
)

console = Console()
app = typer.Typer(help="BigQuery load and query commands")

DEFAULT_DATASET = "telco_churn"
CLEANED_TABLE = "cleaned_customers"
PREDICTIONS_TABLE = "churn_predictions"

# Base columns are listed explicitly rather than selected with *, so the query
# is idempotent: running it against a table that already carries the engineered
# columns would otherwise emit each of them twice and fail as ambiguous.
BASE_COLUMNS = (
    ["customerID"] + NOMINAL_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES + [TARGET]
)
_BASE_SELECT = ",\n  ".join(BASE_COLUMNS)

# Engineered features, expressed in BigQuery Standard SQL. Mirrors
# sql/02_feature_engineering.sql so both backends produce the same columns.
FEATURE_QUERY = f"""
SELECT
  {_BASE_SELECT},
""" + """
  (CASE WHEN OnlineSecurity   = 'Yes' THEN 1 ELSE 0 END)
+ (CASE WHEN OnlineBackup     = 'Yes' THEN 1 ELSE 0 END)
+ (CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END)
+ (CASE WHEN TechSupport      = 'Yes' THEN 1 ELSE 0 END)
+ (CASE WHEN StreamingTV      = 'Yes' THEN 1 ELSE 0 END)
+ (CASE WHEN StreamingMovies  = 'Yes' THEN 1 ELSE 0 END) AS num_addon_services,

  COALESCE(SAFE_DIVIDE(TotalCharges, tenure), MonthlyCharges) AS avg_monthly_spend,

  SAFE_DIVIDE(
    MonthlyCharges,
    COALESCE(SAFE_DIVIDE(TotalCharges, tenure), MonthlyCharges)
  ) AS charges_ratio,

  CASE
    WHEN tenure <= 6  THEN '0-6m'
    WHEN tenure <= 12 THEN '6-12m'
    WHEN tenure <= 24 THEN '1-2y'
    WHEN tenure <= 48 THEN '2-4y'
    ELSE '4y+'
  END AS tenure_bucket,

  CASE
    WHEN MonthlyCharges <  35 THEN 'low'
    WHEN MonthlyCharges <  65 THEN 'medium'
    WHEN MonthlyCharges <  90 THEN 'high'
    ELSE 'premium'
  END AS spend_bucket
FROM `{table}`
"""

ANALYTICS_QUERY = """
WITH features AS (
{feature_query}
)
SELECT
  Contract,
  tenure_bucket,
  spend_bucket,
  COUNT(*)                     AS customers,
  SUM(Churn)                   AS churned,
  ROUND(AVG(Churn), 4)         AS churn_rate,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM features
GROUP BY Contract, tenure_bucket, spend_bucket
HAVING COUNT(*) >= 20
ORDER BY churn_rate DESC
LIMIT 20
"""


def get_project(project: str | None = None) -> str:
    resolved = project or os.getenv("GCP_PROJECT")
    if not resolved:
        raise typer.BadParameter(
            "No GCP project. Set GCP_PROJECT or pass --project.\n"
            "  export GCP_PROJECT=your-project-id\n"
            "  gcloud auth application-default login"
        )
    return resolved


def get_client(project: str | None = None):
    """BigQuery client, with a readable message when credentials are absent."""
    from google.cloud import bigquery

    resolved = get_project(project)
    try:
        return bigquery.Client(project=resolved)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create a BigQuery client for project {resolved!r}: {exc}\n"
            "Run `gcloud auth application-default login` first."
        ) from exc


def table_id(project: str, dataset: str, table: str) -> str:
    return f"{project}.{dataset}.{table}"


def ensure_dataset(client, dataset: str, location: str = "US"):
    """Create the dataset when missing so a first run works on a fresh project."""
    from google.cloud import bigquery

    dataset_ref = f"{client.project}.{dataset}"
    try:
        client.get_dataset(dataset_ref)
        console.log(f"[blue]Dataset {dataset_ref} already exists")
    except Exception:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = location
        client.create_dataset(ds, exists_ok=True)
        console.log(f"[green]Created dataset {dataset_ref} in {location}")
    return dataset_ref


def load_dataframe(client, df: pd.DataFrame, destination: str, write_mode: str = "WRITE_TRUNCATE"):
    """Load a frame into BigQuery and block until the job finishes."""
    from google.cloud import bigquery

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_mode,
        autodetect=True,
    )
    console.log(f"[blue]Loading {len(df)} rows into {destination}...")
    job = client.load_table_from_dataframe(df, destination, job_config=job_config)
    job.result()  # wait for completion; surfaces load errors here

    table = client.get_table(destination)
    console.log(f"[green]Loaded {table.num_rows} rows, {len(table.schema)} columns into {destination}")
    return table


def run_query(client, sql: str) -> pd.DataFrame:
    """Run a query and return the results.

    Reads go through pandas-gbq, which is the path google-cloud-bigquery now
    directs DataFrame access to and which streams results over the BigQuery
    Storage API instead of paging REST. Falls back to the plain client when
    pandas-gbq is not installed.
    """
    console.log("[blue]Running BigQuery query...")
    try:
        import pandas_gbq

        df = pandas_gbq.read_gbq(sql, project_id=client.project, progress_bar_type=None)
    except ImportError:
        df = client.query(sql).result().to_dataframe()
    console.log(f"[green]Query returned {len(df)} rows")
    return df


@app.command("load")
def load(
    source: Path = typer.Option(
        Path.cwd() / "data" / "processed" / "cleaned_data.parquet", help="Parquet to upload"
    ),
    project: str = typer.Option(None, help="GCP project (defaults to GCP_PROJECT)"),
    dataset: str = typer.Option(DEFAULT_DATASET),
    table: str = typer.Option(CLEANED_TABLE),
    location: str = typer.Option("US"),
):
    """Load the cleaned dataset into BigQuery."""
    console.rule("[bold magenta]BigQuery: Load")
    if not source.exists():
        raise typer.BadParameter(f"No data at {source}. Run `make validate` first.")

    client = get_client(project)
    ensure_dataset(client, dataset, location)
    df = pd.read_parquet(source)
    destination = table_id(client.project, dataset, table)
    load_dataframe(client, df, destination)
    console.rule("[bold green]Load complete!")


@app.command("features")
def features(
    output: Path = typer.Option(
        Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
        help="Where to write the queried features",
    ),
    project: str = typer.Option(None),
    dataset: str = typer.Option(DEFAULT_DATASET),
    table: str = typer.Option(CLEANED_TABLE),
):
    """Query engineered features out of BigQuery into the training parquet."""
    console.rule("[bold magenta]BigQuery: Feature Query")
    client = get_client(project)
    source = table_id(client.project, dataset, table)
    df = run_query(client, FEATURE_QUERY.format(table=source))

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    console.log(f"[green]Wrote {len(df)} rows, {len(df.columns)} columns to {output}")
    console.rule("[bold green]Feature query complete!")


@app.command("analytics")
def analytics(
    project: str = typer.Option(None),
    dataset: str = typer.Option(DEFAULT_DATASET),
    table: str = typer.Option(CLEANED_TABLE),
):
    """Compute churn-by-segment in BigQuery and print the top segments."""
    console.rule("[bold magenta]BigQuery: Churn Analytics")
    client = get_client(project)
    source = table_id(client.project, dataset, table)
    sql = ANALYTICS_QUERY.format(feature_query=FEATURE_QUERY.format(table=source))
    df = run_query(client, sql)

    report = Table(title="Churn rate by segment (BigQuery)")
    for col in ("Contract", "tenure_bucket", "spend_bucket", "customers", "churn_rate"):
        report.add_column(col)
    for _, row in df.iterrows():
        report.add_row(
            str(row["Contract"]), str(row["tenure_bucket"]), str(row["spend_bucket"]),
            str(int(row["customers"])), f"{float(row['churn_rate']):.1%}",
        )
    console.print(report)


@app.command("upload-predictions")
def upload_predictions(
    outputs_dir: Path = typer.Option(Path.cwd() / "outputs"),
    project: str = typer.Option(None),
    dataset: str = typer.Option(DEFAULT_DATASET),
    table: str = typer.Option(PREDICTIONS_TABLE),
):
    """Publish the latest batch predictions to BigQuery for downstream consumers."""
    console.rule("[bold magenta]BigQuery: Publish Predictions")
    files = sorted(Path(outputs_dir).glob("predictions_*.parquet"), reverse=True)
    if not files:
        raise typer.BadParameter(f"No predictions in {outputs_dir}. Run `make score` first.")

    client = get_client(project)
    ensure_dataset(client, dataset)
    df = pd.read_parquet(files[0])
    load_dataframe(client, df, table_id(client.project, dataset, table))
    console.rule("[bold green]Predictions published!")


if __name__ == "__main__":
    app()
