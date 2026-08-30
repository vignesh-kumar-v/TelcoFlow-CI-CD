"""Database connectivity for the SQL feature-engineering stage.

The backend is chosen by the DB_URL environment variable, so the same SQL runs
against either engine:

    DB_URL=sqlite:///data/telco.db                        (default, no services)
    DB_URL=postgresql+psycopg2://telco:telco@localhost:5432/telco   (docker compose up -d)

Everything under sql/ is written in portable SQL for that reason — no
dialect-specific functions, and median is computed with an ORDER BY/LIMIT trick
rather than percentile_cont, which SQLite does not have.
"""

import os
from pathlib import Path

from rich.console import Console
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

console = Console()

DEFAULT_DB_URL = "sqlite:///data/telco.db"
RAW_TABLE = "raw_customers"
FEATURE_TABLE = "customer_features"
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def get_db_url() -> str:
    return os.getenv("DB_URL", DEFAULT_DB_URL)


def get_engine(db_url: str | None = None) -> Engine:
    url = db_url or get_db_url()
    if url.startswith("sqlite:///"):
        # SQLAlchemy will not create the parent directory for a file-backed DB.
        db_path = Path(url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    console.log(f"[blue]Database: {_redact(url)}")
    return create_engine(url)


def _redact(url: str) -> str:
    """Hide the password before a connection string reaches the logs."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def read_sql_file(name: str) -> str:
    path = SQL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text()


def split_statements(sql: str) -> list[str]:
    """Split a script into statements.

    Uses sqlparse rather than splitting on ";" because semicolons also appear
    inside comments and string literals, where they do not end a statement.
    """
    import sqlparse

    statements = []
    for raw in sqlparse.split(sql):
        # Strip comments so an all-comment trailing chunk is not run as SQL.
        stripped = sqlparse.format(raw, strip_comments=True).strip()
        if stripped:
            statements.append(stripped)
    return statements


def execute_script(engine: Engine, sql: str):
    """Run a multi-statement script as a single transaction."""
    statements = split_statements(sql)
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    return len(statements)


def table_row_count(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
