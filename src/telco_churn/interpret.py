"""Interpretable coefficients for the logistic regression baseline.

Logistic regression already competes in `train.py` and, on this dataset, wins:
it is the deployed model. But it is deployed as a *scorer*, and the reason to
keep a linear model around when gradient boosting is available is not its AUC —
it is that every coefficient is a statement someone can argue with. "Fiber-optic
internet multiplies the odds of churn by 2.4, holding contract and tenure fixed"
is a claim a retention manager can act on or dispute. SHAP explains what the
model did; an odds ratio explains what the model *believes*.

Two details make the table honest rather than decorative:

  * **The deployed model is L2-penalised, and penalised coefficients have no
    valid Wald standard errors.** The penalty shrinks estimates toward zero by
    an amount that the usual variance formula does not know about, so the
    intervals would be wrong. The inference model is therefore refit without a
    penalty, and its agreement with the deployed model is measured rather than
    assumed — if the two rank customers differently, the table describes a model
    that is not in production.

  * **The serving design matrix one-hot encodes every level with no reference
    category.** That is fine under L2, which handles the resulting collinearity,
    but it makes the unpenalised information matrix singular and the standard
    errors meaningless. The inference matrix uses reference coding, so each
    coefficient reads as "relative to the omitted level" — which is what an odds
    ratio has to mean to be interpretable at all.

Variance inflation factors are reported alongside, because a coefficient with a
VIF of 30 is not a finding, it is two collinear columns splitting one effect.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.telco_churn.features import (
    LINEAR, TARGET, active_features, drop_non_features,
)
from src.telco_churn.inference import LoadedModel, get_latest_artifact_path

console = Console()

ALPHA = 0.05
# Above this, a coefficient is being split across collinear columns rather than
# measuring anything on its own.
VIF_WARNING = 10.0


def build_inference_matrix(X: pd.DataFrame):
    """Reference-coded design matrix: one omitted level per categorical.

    The serving encoder keeps every level, which is what makes the unpenalised
    fit singular. Dropping one level per feature restores full rank and gives
    each coefficient a referent.
    """
    nominal, ordinal, numeric = active_features(X)
    transformer = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), nominal),
            ("ordinal", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), ordinal),
            ("numeric", "passthrough", numeric),
        ]
    )
    matrix = transformer.fit_transform(X)
    names = [n.split("__", 1)[-1] for n in transformer.get_feature_names_out()]
    return pd.DataFrame(matrix, columns=names, index=X.index), transformer


def fit_inference_model(design: pd.DataFrame, y: np.ndarray, random_state: int = 42):
    """Unpenalised logistic regression, on standardised inputs for convergence."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(design)
    # C=inf rather than penalty=None: the same unpenalised fit, and `penalty` is
    # deprecated from scikit-learn 1.8. Internally 1.8 translates C=inf back into
    # penalty=None and warns that it is therefore ignoring C — which is what was
    # asked for, so the notice is suppressed rather than passed on to callers.
    model = LogisticRegression(
        C=np.inf, max_iter=5000, solver="lbfgs",
        class_weight="balanced", random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*penalty=None will ignore.*", category=UserWarning
        )
        model.fit(X_scaled, y)
    return model, scaler, X_scaled


def wald_inference(model, X_scaled: np.ndarray, design: pd.DataFrame,
                   scaler: StandardScaler) -> pd.DataFrame:
    """Odds ratios with Wald standard errors from the observed information matrix.

    Cov(beta) = (X' W X)^-1 with W = diag(p(1-p)); the intercept column has to
    be included or every standard error is computed against the wrong model.
    Coefficients are returned on the original feature scale — a log-odds change
    per unit of `tenure`, not per standard deviation — while the standardised
    version is kept for comparing magnitudes across features.
    """
    probabilities = model.predict_proba(X_scaled)[:, 1]
    weights = probabilities * (1 - probabilities)

    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    information = X_design.T @ (X_design * weights[:, None])
    covariance = np.linalg.pinv(information)
    se_scaled = np.sqrt(np.clip(np.diag(covariance), 0, None))[1:]

    coef_scaled = model.coef_[0]
    # Undo standardisation: a coefficient per raw unit is what reads as a
    # meaningful odds ratio.
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef_raw = coef_scaled / scale
    se_raw = se_scaled / scale

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_scaled > 0, coef_scaled / se_scaled, 0.0)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z)))
    crit = stats.norm.ppf(1 - ALPHA / 2)

    return pd.DataFrame({
        "feature": design.columns,
        "coefficient": coef_raw,
        "std_error": se_raw,
        "odds_ratio": np.exp(coef_raw),
        "or_ci_low": np.exp(coef_raw - crit * se_raw),
        "or_ci_high": np.exp(coef_raw + crit * se_raw),
        "z_statistic": z,
        "p_value": p_values,
        # Comparable across features on different scales: the log-odds shift per
        # standard deviation of the input.
        "standardised_coefficient": coef_scaled,
        "significant": p_values < ALPHA,
    })


def variance_inflation_factors(design: pd.DataFrame) -> pd.Series:
    """VIF per column, from the inverse correlation matrix.

    Computed via the correlation matrix rather than by regressing each column on
    the rest: same answer, one matrix inversion instead of forty-odd fits.
    """
    corr = np.corrcoef(design.to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    try:
        inverse = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=design.columns)
    return pd.Series(np.clip(np.diag(inverse), 1.0, None), index=design.columns)


def check_agreement(deployed: LoadedModel, df: pd.DataFrame, model, scaler,
                    design: pd.DataFrame) -> dict:
    """Does the inference model describe the model that is actually serving?

    Compared on predictions rather than coefficients, because the two live in
    different column spaces — full dummy coding versus reference coding — so
    their coefficient vectors are not comparable term by term. What must match
    is the ranking of customers, since that is what the deployed model is for.
    """
    from sklearn.metrics import roc_auc_score

    deployed_p = deployed.predict_proba(df)
    inference_p = model.predict_proba(scaler.transform(design))[:, 1]
    y = df[TARGET].astype(int).to_numpy()

    return {
        "pearson_r": round(float(np.corrcoef(deployed_p, inference_p)[0, 1]), 4),
        "spearman_rho": round(float(stats.spearmanr(deployed_p, inference_p).statistic), 4),
        "deployed_auc": round(float(roc_auc_score(y, deployed_p)), 4),
        "inference_auc": round(float(roc_auc_score(y, inference_p)), 4),
        "auc_gap": round(float(abs(roc_auc_score(y, deployed_p)
                                   - roc_auc_score(y, inference_p))), 4),
        "mean_abs_probability_gap": round(float(np.abs(deployed_p - inference_p).mean()), 4),
    }


def main(
    clean_path: Path = Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
    artifacts_dir: Path = Path.cwd() / "artifacts",
    reports_dir: Path = Path.cwd() / "reports",
    top_n: int = typer.Option(15, help="Rows to print (the full table is written to reports/)"),
    random_state: int = typer.Option(42, help="Random seed"),
):
    console.rule("[bold magenta]Telco Churn: Logistic Regression Interpretation")

    run_dir = get_latest_artifact_path(Path(artifacts_dir))
    deployed = LoadedModel.load(run_dir)
    console.log(f"[blue]Deployed model: {deployed.model_name} ({deployed.model_type})")

    df = pd.read_parquet(clean_path)
    X = drop_non_features(df)
    y = df[TARGET].astype(int).to_numpy()

    design, _ = build_inference_matrix(X)
    console.log(
        f"[blue]Reference-coded design: {design.shape[1]} terms "
        f"(the serving matrix keeps every level and is rank-deficient)"
    )

    model, scaler, X_scaled = fit_inference_model(design, y, random_state=random_state)
    coefficients = wald_inference(model, X_scaled, design, scaler)
    coefficients["vif"] = variance_inflation_factors(design).to_numpy()

    if deployed.model_type == LINEAR:
        agreement = check_agreement(deployed, df, model, scaler, design)
        console.log(
            f"[green]Agreement with the deployed model: Spearman "
            f"{agreement['spearman_rho']:.4f}, AUC {agreement['deployed_auc']:.4f} "
            f"vs {agreement['inference_auc']:.4f} (gap {agreement['auc_gap']:.4f})"
        )
        if agreement["spearman_rho"] < 0.95:
            console.log(
                "[yellow]The inference model ranks customers differently from the "
                "deployed one — read these coefficients as a description of the "
                "features, not of what is serving."
            )
    else:
        agreement = None
        console.log(
            f"[yellow]The deployed model is {deployed.model_name}, not the linear "
            f"one. This table interprets logistic regression as the transparent "
            f"baseline alongside it, not the model in production."
        )

    ranked = coefficients.reindex(
        coefficients["odds_ratio"].apply(lambda o: abs(np.log(o))).sort_values(ascending=False).index
    )

    table = Table(title=f"Churn odds ratios — top {top_n} by effect size")
    table.add_column("Feature", min_width=30, no_wrap=True)
    table.add_column("Odds ratio", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p", justify="right")
    table.add_column("VIF", justify="right")
    table.add_column("Reading")

    for _, row in ranked.head(top_n).iterrows():
        direction = "raises churn odds" if row["odds_ratio"] > 1 else "lowers churn odds"
        note = "" if row["significant"] else "[yellow]not significant"
        if row["vif"] > VIF_WARNING:
            note = f"[yellow]collinear (VIF {row['vif']:.0f})"
        table.add_row(
            row["feature"][:30],
            f"{row['odds_ratio']:.3f}",
            f"[{row['or_ci_low']:.2f}, {row['or_ci_high']:.2f}]",
            f"{row['p_value']:.4f}" if row["p_value"] >= 1e-4 else "<0.0001",
            f"{row['vif']:.1f}",
            note or direction,
        )
    console.print(table)

    # Headline only from terms that are both significant and not collinear.
    # Quoting an odds ratio whose VIF was just flagged as uninterpretable would
    # contradict the diagnostic printed two lines above it.
    clean = ranked[ranked["significant"] & (ranked["vif"] <= VIF_WARNING)]
    strongest = clean.iloc[0] if len(clean) else ranked.iloc[0]
    direction = "raises" if strongest["odds_ratio"] > 1 else "lowers"
    console.print(
        f"\n[bold]{strongest['feature']}[/bold] carries the largest cleanly identified "
        f"effect: holding every other feature fixed, it {direction} the odds of churn "
        f"by a factor of [bold]{strongest['odds_ratio']:.2f}[/bold] "
        f"(95% CI {strongest['or_ci_low']:.2f}-{strongest['or_ci_high']:.2f}). "
        f"Terms above VIF {VIF_WARNING:.0f} are excluded from this headline."
    )

    high_vif = coefficients[coefficients["vif"] > VIF_WARNING]
    if len(high_vif):
        console.log(
            f"[yellow]{len(high_vif)} term(s) above VIF {VIF_WARNING:.0f} — their "
            f"individual coefficients split one shared effect and should not be "
            f"read separately: {', '.join(high_vif['feature'].head(5))}"
        )
    else:
        console.log(f"[green]No term exceeds VIF {VIF_WARNING:.0f}")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "logistic_coefficients.csv"
    ranked.to_csv(csv_path, index=False)

    report = {
        "timestamp": datetime.now().isoformat(),
        "artifact_run": run_dir.name,
        "deployed_model": deployed.model_name,
        "inference_model": {
            "penalty": "none (C=inf)",
            "coding": "reference (first level dropped per categorical)",
            "n_terms": int(design.shape[1]),
            "n_samples": int(len(design)),
            "note": (
                "Refit without a penalty because L2-shrunk coefficients have no "
                "valid Wald standard errors, and with reference coding because the "
                "serving matrix is rank-deficient without a penalty."
            ),
        },
        "agreement_with_deployed": agreement,
        "n_significant": int(coefficients["significant"].sum()),
        "n_high_vif": int(len(high_vif)),
        "coefficients": json.loads(ranked.to_json(orient="records")),
    }
    report_path = reports_dir / f"coefficients_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.log(f"[green]Coefficient table saved: {csv_path.name}, {report_path.name}")

    console.rule("[bold green]Interpretation completed!")


if __name__ == "__main__":
    typer.run(main)
