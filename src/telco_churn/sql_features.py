"""SQL-based ingest, cleaning and feature engineering.

Replaces the pure-pandas path in validate_and_clean.py: raw records land in a
real table, the cleaning and derived features are expressed as versioned SQL
under sql/, and the result is exported back to the same parquet the training
stage already consumes.

    make sql-features                      # SQLite, no services needed
    DB_URL=postgresql+psycopg2://... make sql-features

Column names are lower-cased on the way in so the SQL is portable between
SQLite and Postgres, then mapped back to the mixed-case names the model expects
on the way out.
"""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.telco_churn.db import (
    FEATURE_TABLE, RAW_TABLE, execute_script, get_engine, read_sql_file,
    table_row_count,
)
from src.telco_churn.features import (
    ENGINEERED_NOMINAL, ENGINEERED_NUMERIC, NOMINAL_FEATURES, NUMERIC_FEATURES,
    ORDINAL_FEATURES, TARGET,
)
from src.telco_churn.validate_and_clean import validate_schema

console = Console()

SCRIPTS = ["01_clean_customers.sql", "02_feature_engineering.sql", "03_churn_analytics.sql"]

# The DB works in lower case; the model expects these exact names.
CANONICAL_COLUMNS = (
    ["customerID"] + NOMINAL_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES
    + [TARGET] + ENGINEERED_NUMERIC + ENGINEERED_NOMINAL
)
LOWER_TO_CANONICAL = {c.lower(): c for c in CANONICAL_COLUMNS}


def ingest_raw(csv_path: Path, engine, table: str = RAW_TABLE) -> int:
    """Load the raw CSV into the database, replacing any previous load."""
    console.log(f"[blue]Ingesting {csv_path} into {table}...")
    df = pd.read_csv(csv_path)
    validate_schema(df)
    df.columns = [c.lower() for c in df.columns]
    df.to_sql(table, engine, if_exists="replace", index=False)
    count = table_row_count(engine, table)
    console.log(f"[green]Ingested {count} rows into {table}")
    return count


def run_transformations(engine) -> None:
    for script in SCRIPTS:
        console.log(f"[blue]Running {script}...")
        n = execute_script(engine, read_sql_file(script))
        console.log(f"[green]{script}: {n} statement(s) executed")


def export_features(engine, output_path: Path, table: str = FEATURE_TABLE) -> pd.DataFrame:
    """Read the engineered table back out to the parquet training consumes."""
    console.log(f"[blue]Exporting {table} to {output_path}...")
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    df = df.rename(columns={c: LOWER_TO_CANONICAL.get(c, c) for c in df.columns})

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"SQL output is missing expected columns: {missing}")
    df = df[CANONICAL_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    console.log(f"[green]Exported {len(df)} rows, {len(df.columns)} columns")
    return df


def show_segment_report(engine, limit: int = 10) -> None:
    """Print the highest-churn segments produced by the analytics SQL."""
    df = pd.read_sql(
        f"SELECT * FROM churn_by_segment ORDER BY churn_rate DESC LIMIT {limit}", engine
    )
    if df.empty:
        return
    table = Table(title=f"Top {len(df)} churn segments (min 20 customers)")
    for col in ("contract", "tenure_bucket", "spend_bucket", "customers", "churn_rate"):
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(
            str(row["contract"]), str(row["tenure_bucket"]), str(row["spend_bucket"]),
            str(int(row["customers"])), f"{float(row['churn_rate']):.1%}",
        )
    console.print(table)


def main(
    raw_path: Path = Path.cwd() / "data" / "raw" / "Telco-Customer-Churn.csv",
    output_path: Path = Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
    db_url: str = typer.Option(None, "--db-url", help="Overrides the DB_URL env var"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Reuse the existing raw table"),
):
    console.rule("[bold magenta]Telco Churn: SQL Feature Engineering")
    engine = get_engine(db_url)

    if not skip_ingest:
        ingest_raw(raw_path, engine)
    run_transformations(engine)
    export_features(engine, output_path)
    show_segment_report(engine)

    console.rule("[bold green]SQL feature pipeline completed!")


if __name__ == "__main__":
    typer.run(main)
