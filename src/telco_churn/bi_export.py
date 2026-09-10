"""BI extract layer: the pipeline's results, shaped for Tableau or Power BI.

Everything this project produces lands in JSON reports and parquet files, which
is right for the pipeline and wrong for anyone who wants to look at it. A BI tool
wants flat, typed, denormalised tables with stable column names — so this module
publishes one.

Two shapes are written, because dashboards need both:

  * a **star schema** (`dim_*` / `fct_*`) for anyone joining tables themselves, and
  * **`telcoflow_master.csv`**, one denormalised row per customer with the score,
    the segment, the SHAP contributions and the outcome already joined, so a
    dashboard can be built without configuring a single relationship.

The per-customer SHAP export is the reason this is worth doing rather than just
screenshotting a summary plot. `fct_shap_customer.csv` is long-format — one row
per customer per feature — which is what lets a dashboard answer "why is *this*
customer high risk?" instead of only "what matters on average?". That is the
question a retention agent actually has, and it is the one a static SHAP summary
plot cannot answer.

Every table is regenerated from whatever the latest pipeline run produced, so
the dashboard follows the model rather than freezing one run's numbers.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.telco_churn.features import LINEAR, TARGET, TREE_NATIVE, drop_non_features
from src.telco_churn.inference import LoadedModel, get_latest_artifact_path, risk_category

console = Console()

EXPORT_DIRNAME = "bi"
MASTER_FILE = "telcoflow_master.csv"

# Per-customer SHAP is long-format and multiplies rows by features, so it is
# capped: a dashboard needs enough customers to explore, not all of them.
MAX_SHAP_CUSTOMERS = 1500


def coerce_binary_shap(shap_values):
    """SHAP returns per-class values in several shapes across versions and models."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


def compute_customer_shap(loaded: LoadedModel, df: pd.DataFrame):
    """Per-customer SHAP contributions for the deployed model, whatever type it is.

    Returns (values, feature_names, display_frame) where `display_frame` holds
    the feature values each contribution belongs to, so the dashboard can show
    "tenure = 2 months contributed +0.14" rather than a bare number.
    """
    import shap

    X_raw = drop_non_features(df)
    X = loaded.pipeline.transform_for(X_raw, loaded.model_type)

    if loaded.model_type == LINEAR:
        scaler, classifier = loaded.model[0], loaded.model[-1]
        X_scaled = scaler.transform(X)
        explainer = shap.LinearExplainer(classifier, X_scaled)
        values = coerce_binary_shap(explainer.shap_values(X_scaled))
        names = loaded.pipeline.lr_transformer.get_feature_names_out().tolist()
        display = pd.DataFrame(X, columns=names, index=df.index)
    else:
        explainer = shap.TreeExplainer(loaded.model)
        display = X.copy()
        for col in display.columns:
            if isinstance(display[col].dtype, pd.CategoricalDtype):
                display[col] = display[col].cat.codes
        source = X if loaded.model_type == TREE_NATIVE else display
        values = coerce_binary_shap(explainer.shap_values(source, check_additivity=False))
        names = display.columns.tolist()

    return np.asarray(values), names, display


def _clean_names(names: list) -> list:
    """Strip ColumnTransformer prefixes so the dashboard shows readable labels."""
    return [n.split("__", 1)[-1] for n in names]


def build_customer_table(loaded: LoadedModel, predictions: pd.DataFrame,
                         segments: pd.DataFrame = None) -> pd.DataFrame:
    """One row per scored customer: attributes, score, risk band, segment, outcome."""
    df = predictions.copy()
    df["risk_category"] = df["churn_probability"].apply(risk_category)
    df["risk_decile"] = pd.qcut(
        df["churn_probability"], q=10, labels=False, duplicates="drop"
    ) + 1
    df["model_version"] = loaded.version
    df["model_name"] = loaded.model_name

    if TARGET in df.columns:
        df["actual_churn"] = df[TARGET].astype(int)
        # Whether the 0.5 threshold got this customer right, so the dashboard can
        # show where the model fails rather than only where it scores.
        df["prediction_outcome"] = np.select(
            [
                (df["prediction"] == 1) & (df["actual_churn"] == 1),
                (df["prediction"] == 1) & (df["actual_churn"] == 0),
                (df["prediction"] == 0) & (df["actual_churn"] == 1),
            ],
            ["true positive", "false positive", "false negative"],
            default="true negative",
        )

    df["annual_revenue"] = df["MonthlyCharges"] * 12
    df["revenue_at_risk"] = df["annual_revenue"] * df["churn_probability"]

    if segments is not None and "customerID" in df.columns:
        df = df.merge(segments, on="customerID", how="left")
    return df


def build_shap_tables(loaded: LoadedModel, predictions: pd.DataFrame,
                      max_customers: int = MAX_SHAP_CUSTOMERS):
    """Global feature importance, and the long per-customer contribution table."""
    sample = predictions.head(max_customers).reset_index(drop=True)
    values, names, display = compute_customer_shap(loaded, sample)
    names = _clean_names(names)

    mean_abs = np.abs(values).mean(axis=0)
    # Sign of the average contribution: does more of this feature push toward
    # churn or away from it? The magnitude alone cannot say.
    mean_signed = values.mean(axis=0)
    global_table = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
        "direction": np.where(mean_signed >= 0, "increases churn", "decreases churn"),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    global_table["rank"] = global_table.index + 1

    ids = sample["customerID"] if "customerID" in sample.columns \
        else pd.Series([f"row-{i}" for i in range(len(sample))])
    long = pd.DataFrame({
        "customerID": np.repeat(ids.to_numpy(), len(names)),
        "feature": np.tile(names, len(sample)),
        "shap_value": values.reshape(-1),
        "feature_value": display.to_numpy(dtype=float).reshape(-1),
    })
    long["abs_shap"] = long["shap_value"].abs()
    long["direction"] = np.where(long["shap_value"] >= 0, "increases churn", "decreases churn")

    # Rank within each customer, so a dashboard can filter to "this customer's
    # top 5 drivers" with a simple predicate instead of a table calculation.
    long["rank_within_customer"] = (
        long.groupby("customerID")["abs_shap"].rank(ascending=False, method="first").astype(int)
    )
    return global_table, long


def _latest_report(reports_dir: Path, prefix: str):
    files = sorted(Path(reports_dir).glob(f"{prefix}_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def build_result_tables(reports_dir: Path, run_dir: Path) -> dict:
    """Flatten the JSON reports into the long tables a BI tool can chart.

    Each analysis writes a nested JSON document, which is the right shape for a
    report and the wrong shape for a bar chart. These are all reduced to the same
    tidy form — one row per estimate, with its interval — so a single dashboard
    sheet can render results from four different analyses.
    """
    tables = {}

    comparison_path = Path(run_dir) / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)
        tables["fct_model_comparison"] = pd.DataFrame([
            {"model": name, **{k: v for k, v in metrics.items() if k != "support"},
             "deployed": name == json.load(open(Path(run_dir) / "train_info.json"))["model_name"]}
            for name, metrics in comparison.items()
        ]).sort_values("roc_auc", ascending=False)

    ab = _latest_report(reports_dir, "ab_test")
    if ab:
        rows = []
        for label, block in (("A/B test", ab["ab_test"]), ("A/A control", ab["aa_control"])):
            retention, revenue = block["retention"], block["revenue"]
            rows.append({
                "run": label, "metric": "retention rate",
                "treatment": retention["rate_treatment"], "control": retention["rate_control"],
                "difference": retention["absolute_difference"],
                "ci_low": retention["ci_95"][0], "ci_high": retention["ci_95"][1],
                "p_value": retention["p_value"], "significant": retention["significant"],
                "effect_size": retention["cohens_h"], "effect_size_name": "Cohen's h",
            })
            rows.append({
                "run": label, "metric": "net revenue per customer",
                "treatment": revenue["mean_treatment"], "control": revenue["mean_control"],
                "difference": revenue["difference"],
                "ci_low": revenue["ci_95"][0], "ci_high": revenue["ci_95"][1],
                "p_value": revenue["p_value"], "significant": revenue["significant"],
                "effect_size": revenue["cohens_d"], "effect_size_name": "Cohen's d",
            })
        tables["fct_ab_test"] = pd.DataFrame(rows)

        verdict = ab.get("pre_registered_verdict")
        prereg = ab.get("pre_registration")
        if verdict and prereg:
            tables["fct_power"] = pd.DataFrame([{
                "experiment": prereg.get("experiment"),
                "pre_registered_mde": verdict["pre_registered_mde"],
                "observed_effect": verdict["observed_effect"],
                "p_value": verdict["p_value"],
                "achieved_power": verdict["achieved_power"],
                "required_n_per_arm": verdict["required_n_per_arm"],
                "available_n_per_arm": verdict["available_n_per_arm"],
                "adequately_powered": verdict["adequately_powered"],
                "verdict": verdict["verdict"],
                "detectable_mde_at_available_n": prereg.get("detectable_mde_at_target_power"),
            }])

    segmentation = _latest_report(reports_dir, "segmentation")
    if segmentation:
        segments = pd.DataFrame(segmentation["segments"])
        segments["distinguishing_features"] = segments["distinguishing_features"].apply(
            lambda d: ", ".join(f"{k} ({v:+.1f} SD)" for k, v in d.items())
        )
        tables["dim_segment"] = segments
        tables["fct_k_selection"] = pd.DataFrame(segmentation["k_selection"]["scores"])

    causal = _latest_report(reports_dir, "causal")
    if causal:
        rows = []
        primary = causal["observational"]["primary"]
        rows.append({
            "analysis": "Propensity scoring", "estimator": "Naive difference",
            "estimate": primary["naive_difference"], "ci_low": None, "ci_high": None,
            "adjusted": False,
            "note": "confounded by who buys Tech Support",
        })
        for name, e in primary["estimates"].items():
            rows.append({
                "analysis": "Propensity scoring", "estimator": name.replace("_", " "),
                "estimate": e["estimate"], "ci_low": e["ci_95"][0], "ci_high": e["ci_95"][1],
                "adjusted": True, "note": e["method"],
            })

        iv = causal["instrumental_variables"]
        for name, key, adjusted in (
            ("Naive OLS", "naive_ols", False),
            ("Intention to treat", "intention_to_treat", True),
            ("2SLS / LATE", "two_stage_least_squares", True),
        ):
            block = iv[key]
            rows.append({
                "analysis": "Instrumental variables", "estimator": name,
                "estimate": block["coefficient"],
                "ci_low": block["ci_95"][0], "ci_high": block["ci_95"][1],
                "adjusted": adjusted, "note": "",
            })
        rows.append({
            "analysis": "Instrumental variables", "estimator": "True effect (constructed)",
            "estimate": iv["true_complier_effect"], "ci_low": None, "ci_high": None,
            "adjusted": True, "note": "the estimand, computed from the DGP",
        })

        did = causal["difference_in_differences"]
        for name, value, adjusted in (
            ("Naive post-period", did["naive_post_period_comparison"], False),
            ("Naive before/after", did["naive_before_after_comparison"], False),
            ("Difference-in-differences", did["did_estimate"], True),
            ("True effect (constructed)", did["true_effect"], True),
        ):
            rows.append({
                "analysis": "Difference-in-differences", "estimator": name,
                "estimate": value,
                "ci_low": did["regression"]["ci_95"][0] if name.startswith("Difference") else None,
                "ci_high": did["regression"]["ci_95"][1] if name.startswith("Difference") else None,
                "adjusted": adjusted, "note": "",
            })
        tables["fct_causal_estimates"] = pd.DataFrame(rows)

        balance = primary["balance"]["per_covariate"]
        tables["fct_covariate_balance"] = pd.DataFrame([
            {"covariate": k, "smd_before": v["before"], "smd_after": v["after_matching"],
             "balanced_after": abs(v["after_matching"]) <= primary["balance"]["threshold"]}
            for k, v in balance.items()
        ]).sort_values("smd_before", key=lambda s: s.abs(), ascending=False)

    coefficients = _latest_report(reports_dir, "coefficients")
    if coefficients:
        tables["fct_logistic_coefficients"] = pd.DataFrame(coefficients["coefficients"])

    return tables


def build_master_table(customers: pd.DataFrame, shap_long: pd.DataFrame,
                       top_k: int = 3) -> pd.DataFrame:
    """One denormalised row per customer, with the top SHAP drivers pivoted in.

    Exists so a dashboard can be built from a single file with no joins. The top
    drivers are widened into columns rather than left long, because a customer
    detail view wants "driver 1, driver 2, driver 3" as fields.
    """
    master = customers.copy()
    if shap_long.empty or "customerID" not in master.columns:
        return master

    top = shap_long[shap_long["rank_within_customer"] <= top_k]
    for rank in range(1, top_k + 1):
        slice_ = top[top["rank_within_customer"] == rank].set_index("customerID")
        master[f"driver_{rank}_feature"] = master["customerID"].map(slice_["feature"])
        master[f"driver_{rank}_shap"] = master["customerID"].map(slice_["shap_value"])
        master[f"driver_{rank}_direction"] = master["customerID"].map(slice_["direction"])
    return master


DATA_DICTIONARY = {
    "telcoflow_master": (
        "One row per scored customer. Attributes, churn score, risk band, segment, "
        "revenue at risk and the top three SHAP drivers, already joined. Start here "
        "— it needs no relationships configured."
    ),
    "dim_customer": "One row per scored customer, without the SHAP columns.",
    "fct_shap_customer": (
        "Long format: one row per customer per feature. Filter on "
        "rank_within_customer <= 5 for a per-customer explanation view."
    ),
    "fct_shap_global": "Mean absolute SHAP per feature, with rank and direction.",
    "dim_segment": "K-means segment profiles: size, churn rate, value and revenue at risk.",
    "fct_k_selection": "Silhouette, inertia and Davies-Bouldin per candidate k.",
    "fct_model_comparison": "Test metrics for every candidate model; `deployed` flags the winner.",
    "fct_ab_test": "A/B and A/A results per metric, with confidence intervals and effect sizes.",
    "fct_power": "Pre-registered design against what the experiment actually achieved.",
    "fct_causal_estimates": (
        "Every causal estimate on one scale, tagged by analysis. `adjusted` "
        "separates the confounded comparisons from the corrected ones."
    ),
    "fct_covariate_balance": "Standardised mean differences before and after matching.",
    "fct_logistic_coefficients": "Odds ratios with confidence intervals, p-values and VIF.",
}


_TABLEAU_TYPES = {
    "i": ("integer", "quantitative", "measure"),
    "u": ("integer", "quantitative", "measure"),
    "f": ("real", "quantitative", "measure"),
    "b": ("boolean", "nominal", "dimension"),
    "M": ("date", "ordinal", "dimension"),
}


def _tableau_column_type(series: pd.Series):
    return _TABLEAU_TYPES.get(series.dtype.kind, ("string", "nominal", "dimension"))


def generate_tableau_workbook(master: pd.DataFrame, export_dir: Path,
                              out_path: Path) -> Path:
    """Emit a .twb that opens with the master extract connected and typed.

    A Tableau workbook is XML, so the tedious half — pointing at the file,
    declaring forty-odd columns and marking each one dimension or measure — can
    be generated. The sheets cannot, and are not attempted: a dashboard is a
    design task, and dashboards/README.md walks through building it.

    The CSV path is absolute and baked in at generation time, because that is
    how Tableau stores a text connection. Moving the repo means regenerating
    this file, or just opening the CSV directly.
    """
    from xml.etree import ElementTree as ET

    connection_id = "textscan.telcoflow"
    stem = Path(MASTER_FILE).stem

    workbook = ET.Element("workbook", {
        "source-build": "2023.3.0", "source-platform": "mac", "version": "18.1",
        "xmlns:user": "http://www.tableausoftware.com/xml/user",
    })
    datasources = ET.SubElement(workbook, "datasources")
    datasource = ET.SubElement(datasources, "datasource", {
        "caption": "TelcoFlow master", "inline": "true",
        "name": "federated.telcoflow", "version": "18.1",
    })
    connection = ET.SubElement(datasource, "connection", {"class": "federated"})

    named_connections = ET.SubElement(connection, "named-connections")
    named_connection = ET.SubElement(named_connections, "named-connection", {
        "caption": stem, "name": connection_id,
    })
    ET.SubElement(named_connection, "connection", {
        "class": "textscan",
        "directory": str(Path(export_dir).resolve()),
        "filename": MASTER_FILE,
        "password": "", "server": "",
    })

    relation = ET.SubElement(connection, "relation", {
        "connection": connection_id, "name": MASTER_FILE,
        "table": f"[{stem}#csv]", "type": "table",
    })
    columns = ET.SubElement(relation, "columns", {
        "character-set": "UTF-8", "header": "yes", "locale": "en_US", "separator": ",",
    })
    for ordinal, name in enumerate(master.columns):
        datatype, _, _ = _tableau_column_type(master[name])
        ET.SubElement(columns, "column", {
            "datatype": datatype, "name": name, "ordinal": str(ordinal),
        })

    # Role metadata: without it every numeric column arrives as a measure and
    # identifiers like risk_decile get summed.
    for name in master.columns:
        datatype, dtype_role, role = _tableau_column_type(master[name])
        if name in ("risk_decile", "segment", "prediction", "actual_churn"):
            role, dtype_role = "dimension", "ordinal"
        ET.SubElement(datasource, "column", {
            "caption": name.replace("_", " ").title(),
            "datatype": datatype, "name": f"[{name}]",
            "role": role, "type": dtype_role,
        })

    ET.SubElement(workbook, "worksheets")
    ET.SubElement(workbook, "dashboards")
    ET.SubElement(workbook, "windows")

    tree = ET.ElementTree(workbook)
    ET.indent(tree, space="  ")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def write_data_dictionary(tables: dict, out_path: Path) -> Path:
    """Markdown reference so the extracts are usable without reading this file."""
    lines = [
        "# TelcoFlow BI extracts",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from the latest pipeline run.",
        "Regenerate with `make bi-export` after any run — these files follow the model.",
        "",
        "| Table | Rows | Columns | What it holds |",
        "|---|---:|---:|---|",
    ]
    for name, df in sorted(tables.items()):
        description = DATA_DICTIONARY.get(name, "")
        lines.append(f"| `{name}.csv` | {len(df):,} | {len(df.columns)} | {description} |")

    lines += ["", "## Columns", ""]
    for name, df in sorted(tables.items()):
        lines.append(f"### `{name}.csv`")
        lines.append("")
        lines.append("| Column | Type |")
        lines.append("|---|---|")
        for col in df.columns:
            lines.append(f"| `{col}` | {_tableau_column_type(df[col])[0]} |")
        lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def main(
    artifacts_dir: Path = Path.cwd() / "artifacts",
    outputs_dir: Path = Path.cwd() / "outputs",
    reports_dir: Path = Path.cwd() / "reports",
    export_dir: Path = Path.cwd() / "outputs" / EXPORT_DIRNAME,
    dashboards_dir: Path = Path.cwd() / "dashboards",
    max_shap_customers: int = typer.Option(MAX_SHAP_CUSTOMERS,
                                           help="Cap on customers in the long SHAP table"),
):
    console.rule("[bold magenta]Telco Churn: BI Extract Export")

    run_dir = get_latest_artifact_path(Path(artifacts_dir))
    loaded = LoadedModel.load(run_dir)
    console.log(f"[blue]Deployed model: {loaded.model_name} ({loaded.version})")

    prediction_files = sorted(Path(outputs_dir).glob("predictions_*.parquet"), reverse=True)
    if not prediction_files:
        raise FileNotFoundError(f"No predictions in {outputs_dir}. Run `make score` first.")
    predictions = pd.read_parquet(prediction_files[0])
    console.log(f"[green]Loaded {len(predictions):,} scored customers")

    segments_path = Path(outputs_dir) / "customer_segments.parquet"
    segments = pd.read_parquet(segments_path) if segments_path.exists() else None
    if segments is None:
        console.log("[yellow]No segment assignments — run `make segment` to include them")

    customers = build_customer_table(loaded, predictions, segments)

    console.log("[blue]Computing per-customer SHAP contributions...")
    shap_global, shap_long = build_shap_tables(loaded, predictions, max_shap_customers)
    console.log(
        f"[green]{len(shap_long):,} contribution rows across "
        f"{shap_long['customerID'].nunique():,} customers and "
        f"{shap_global.shape[0]} features"
    )

    tables = {
        "dim_customer": customers,
        "fct_shap_global": shap_global,
        "fct_shap_customer": shap_long,
        **build_result_tables(reports_dir, run_dir),
    }
    tables["telcoflow_master"] = build_master_table(customers, shap_long)

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(export_dir / f"{name}.csv", index=False)

    summary = Table(title=f"BI extracts written to {export_dir}")
    summary.add_column("Table")
    summary.add_column("Rows", justify="right")
    summary.add_column("Columns", justify="right")
    for name, df in sorted(tables.items()):
        summary.add_row(f"{name}.csv", f"{len(df):,}", str(len(df.columns)))
    console.print(summary)

    dictionary = write_data_dictionary(tables, export_dir / "DATA_DICTIONARY.md")
    console.log(f"[green]Data dictionary: {dictionary}")

    workbook = generate_tableau_workbook(
        tables["telcoflow_master"], export_dir, Path(dashboards_dir) / "telcoflow.twb"
    )
    console.log(f"[green]Tableau workbook scaffold: {workbook}")
    console.print(
        f"\nOpen [bold]{workbook.name}[/bold] in Tableau, or connect directly to "
        f"[bold]{MASTER_FILE}[/bold] — see dashboards/README.md for the build."
    )

    console.rule("[bold green]BI export completed!")


if __name__ == "__main__":
    typer.run(main)
