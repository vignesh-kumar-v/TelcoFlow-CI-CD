"""Retention-campaign A/B test simulation.

Scoring customers is only half the problem: the business question is whether
acting on those scores actually retains anyone, and whether the observed
difference is real or noise.

This module takes the batch-scored customers, targets the high-risk ones,
randomly assigns them to a retention offer (treatment) or no contact (control),
simulates outcomes, and evaluates the result the way a real experiment readout
would:

  * two-proportion z-test and chi-square on the churn rate
  * Welch's t-test on retained revenue per customer
  * confidence interval and effect size, not just a p-value
  * an A/A negative control, which must come out non-significant
  * the readout judged against the pre-registered design in reports/, so a
    non-significant result in an under-powered study is reported as
    inconclusive rather than as evidence of no effect

The A/A run matters: a test that finds an effect where none was injected is
measuring a bug in the harness, not a treatment.

Assignment is stratified by predicted-risk decile. Simple random assignment can
leave the arms imbalanced on baseline risk, which is precisely the variable the
outcome depends on most.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from scipy import stats

from src.telco_churn.power import (
    PreRegistration, holm_bonferroni, interpret_result, mde_for_sample_size,
)

console = Console()

ALPHA = 0.05


def load_latest_predictions(outputs_dir: Path) -> pd.DataFrame:
    files = sorted(Path(outputs_dir).glob("predictions_*.parquet"), reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No predictions in {outputs_dir}. Run `make score` first."
        )
    console.log(f"[blue]Loading predictions from {files[0].name}...")
    return pd.read_parquet(files[0])


def assign_groups(df: pd.DataFrame, risk_threshold: float, seed: int, n_strata: int = 10):
    """Target high-risk customers and split them into balanced arms.

    Returns the targeted frame with a `group` column.
    """
    targeted = df[df["churn_probability"] >= risk_threshold].copy()
    if targeted.empty:
        raise ValueError(f"No customers scored at or above {risk_threshold}")

    rng = np.random.default_rng(seed)

    # Stratify on risk so both arms carry the same baseline churn pressure.
    targeted["risk_stratum"] = pd.qcut(
        targeted["churn_probability"], q=min(n_strata, targeted["churn_probability"].nunique()),
        labels=False, duplicates="drop",
    )
    targeted["group"] = "control"
    for _, idx in targeted.groupby("risk_stratum").groups.items():
        idx = np.array(idx)
        shuffled = rng.permutation(idx)
        treated = shuffled[: len(shuffled) // 2]
        targeted.loc[treated, "group"] = "treatment"

    console.log(
        f"[green]Targeted {len(targeted)} customers at risk >= {risk_threshold} "
        f"({(targeted['group'] == 'treatment').sum()} treatment / "
        f"{(targeted['group'] == 'control').sum()} control)"
    )
    return targeted


def simulate_outcomes(targeted: pd.DataFrame, effect: float, offer_cost: float,
                      seed: int) -> pd.DataFrame:
    """Draw churn outcomes, with the offer reducing risk by `effect` (relative).

    The model's predicted probability is treated as each customer's true
    baseline risk; the treatment multiplies it by (1 - effect). `effect = 0`
    gives an A/A test.
    """
    rng = np.random.default_rng(seed)
    df = targeted.copy()

    baseline = df["churn_probability"].to_numpy()
    multiplier = np.where(df["group"].eq("treatment"), 1.0 - effect, 1.0)
    true_risk = np.clip(baseline * multiplier, 0.0, 1.0)

    df["churned"] = (rng.random(len(df)) < true_risk).astype(int)
    df["retained"] = 1 - df["churned"]

    # Annual revenue kept, less the cost of the offer for the treated arm.
    monthly = df["MonthlyCharges"] if "MonthlyCharges" in df.columns else pd.Series(70.0, index=df.index)
    df["revenue"] = df["retained"] * monthly * 12
    df["revenue_net"] = df["revenue"] - np.where(df["group"].eq("treatment"), offer_cost, 0.0)
    return df


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int):
    """Z-test for a difference in two proportions, with a 95% CI."""
    p_a, p_b = success_a / n_a, success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # CI uses the unpooled SE: the pooled form belongs to the null hypothesis.
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - ALPHA / 2)
    return {
        "rate_treatment": round(p_a, 4),
        "rate_control": round(p_b, 4),
        "absolute_difference": round(p_a - p_b, 4),
        "relative_lift": round((p_a - p_b) / p_b, 4) if p_b else None,
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "ci_95": [round(float((p_a - p_b) - crit * se_diff), 4),
                  round(float((p_a - p_b) + crit * se_diff), 4)],
        # Cohen's h: effect size for proportions, independent of sample size.
        "cohens_h": round(float(
            2 * np.arcsin(np.sqrt(p_a)) - 2 * np.arcsin(np.sqrt(p_b))
        ), 4),
        "significant": bool(p_value < ALPHA),
    }


def chi_square_test(df: pd.DataFrame):
    table = pd.crosstab(df["group"], df["churned"])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    return {
        "contingency_table": table.to_dict(),
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": int(dof),
        # Phi coefficient: Cramer's V for a 2x2 table.
        "cramers_v": round(float(np.sqrt(chi2 / n)), 4),
        "significant": bool(p_value < ALPHA),
    }


def revenue_t_test(df: pd.DataFrame, column: str = "revenue_net"):
    """Welch's t-test — the arms need not share a variance."""
    treatment = df.loc[df["group"] == "treatment", column]
    control = df.loc[df["group"] == "control", column]
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    n_t, n_c = len(treatment), len(control)
    var_t, var_c = treatment.var(ddof=1), control.var(ddof=1)
    pooled_sd = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
    se_diff = np.sqrt(var_t / n_t + var_c / n_c)
    dof = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    )
    crit = stats.t.ppf(1 - ALPHA / 2, dof)
    diff = treatment.mean() - control.mean()

    return {
        "mean_treatment": round(float(treatment.mean()), 2),
        "mean_control": round(float(control.mean()), 2),
        "difference": round(float(diff), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": round(float(dof), 1),
        "ci_95": [round(float(diff - crit * se_diff), 2),
                  round(float(diff + crit * se_diff), 2)],
        "cohens_d": round(float(diff / pooled_sd), 4) if pooled_sd > 0 else 0.0,
        "significant": bool(p_value < ALPHA),
    }


def minimum_detectable_effect(n_per_arm: int, baseline_rate: float, power: float = 0.8):
    """Smallest absolute change this sample size could reliably detect.

    Delegates to the power module so the readout and the pre-registration cannot
    drift apart — two implementations of this formula quietly disagreeing would
    have the experiment report one detectable effect and plan against another.
    """
    return round(mde_for_sample_size(n_per_arm, baseline_rate, ALPHA, power), 4)


def run_experiment(targeted: pd.DataFrame, effect: float, offer_cost: float, seed: int):
    outcomes = simulate_outcomes(targeted, effect, offer_cost, seed)

    treatment = outcomes[outcomes["group"] == "treatment"]
    control = outcomes[outcomes["group"] == "control"]

    retention = two_proportion_z_test(
        int(treatment["retained"].sum()), len(treatment),
        int(control["retained"].sum()), len(control),
    )
    return outcomes, {
        "n_treatment": len(treatment),
        "n_control": len(control),
        "injected_effect": effect,
        "retention": retention,
        "chi_square": chi_square_test(outcomes),
        "revenue": revenue_t_test(outcomes),
        "minimum_detectable_effect": minimum_detectable_effect(
            min(len(treatment), len(control)),
            float(control["retained"].mean()),
        ),
    }


def print_readout(result: dict, title: str):
    retention, revenue = result["retention"], result["revenue"]
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Treatment", justify="right")
    table.add_column("Control", justify="right")
    table.add_column("Difference", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Verdict")

    table.add_row(
        "Retention rate",
        f"{retention['rate_treatment']:.1%}", f"{retention['rate_control']:.1%}",
        f"{retention['absolute_difference']:+.1%}", f"{retention['p_value']:.4f}",
        "[green]significant" if retention["significant"] else "[yellow]not significant",
    )
    table.add_row(
        "Net revenue/customer",
        f"${revenue['mean_treatment']:,.0f}", f"${revenue['mean_control']:,.0f}",
        f"${revenue['difference']:+,.0f}", f"{revenue['p_value']:.4f}",
        "[green]significant" if revenue["significant"] else "[yellow]not significant",
    )
    console.print(table)
    console.print(
        f"  Retention 95% CI: [{retention['ci_95'][0]:+.1%}, {retention['ci_95'][1]:+.1%}]"
        f"   Cohen's h: {retention['cohens_h']:.3f}"
        f"   MDE @80% power: {result['minimum_detectable_effect']:.1%}"
    )


def print_prereg_verdict(interpretation: dict, corrected: dict) -> None:
    """Report the readout against the design that was committed before it ran."""
    colour = {"significant": "green", "null": "blue", "inconclusive": "yellow"}[
        interpretation["verdict"]
    ]
    console.print(
        f"\n[bold]Pre-registered verdict:[/bold] [{colour}]"
        f"{interpretation['verdict'].upper()}[/{colour}]"
    )
    console.print(f"  {interpretation['reading']}")
    console.print(
        f"  Planned MDE {interpretation['pre_registered_mde']:.1%} · "
        f"power {interpretation['achieved_power']:.0%} · "
        f"n {interpretation['available_n_per_arm']:,}/"
        f"{interpretation['required_n_per_arm']:,} per arm"
    )

    console.print("\n[bold]Holm-Bonferroni across the metric family:[/bold]")
    for name, r in corrected.items():
        mark = "[green]holds" if r["significant_after_correction"] else "[yellow]does not hold"
        console.print(
            f"  {name:<24} p={r['p_value']:.4f}  "
            f"threshold={r['adjusted_threshold']:.4f}  {mark}"
        )


def main(
    outputs_dir: Path = Path.cwd() / "outputs",
    reports_dir: Path = Path.cwd() / "reports",
    risk_threshold: float = typer.Option(0.5, help="Target customers at or above this churn score"),
    effect: float = typer.Option(0.20, help="Relative churn reduction from the offer"),
    offer_cost: float = typer.Option(50.0, help="Cost of the retention offer per treated customer"),
    seed: int = typer.Option(42, help="Random seed"),
):
    console.rule("[bold magenta]Telco Churn: Retention A/B Test Simulation")

    predictions = load_latest_predictions(outputs_dir)
    targeted = assign_groups(predictions, risk_threshold, seed)

    # Balance check: if the arms differ on baseline risk, the readout is suspect.
    balance = targeted.groupby("group")["churn_probability"].mean()
    _, balance_p = stats.ttest_ind(
        targeted.loc[targeted["group"] == "treatment", "churn_probability"],
        targeted.loc[targeted["group"] == "control", "churn_probability"],
        equal_var=False,
    )
    console.log(
        f"[blue]Randomisation check — mean predicted risk "
        f"treatment {balance.get('treatment', float('nan')):.4f} vs "
        f"control {balance.get('control', float('nan')):.4f} (p={balance_p:.3f})"
    )

    outcomes, ab_result = run_experiment(targeted, effect, offer_cost, seed)
    print_readout(ab_result, f"A/B test — retention offer ({effect:.0%} assumed churn reduction)")

    # Negative control: same machinery, no injected effect.
    console.log("[blue]Running A/A negative control...")
    _, aa_result = run_experiment(targeted, 0.0, offer_cost, seed + 1)
    print_readout(aa_result, "A/A negative control — no treatment effect injected")

    if aa_result["retention"]["significant"]:
        console.log(
            "[yellow]A/A control came out significant — expected roughly "
            f"{ALPHA:.0%} of the time by chance; re-run with another seed."
        )
    else:
        console.log("[green]A/A control is non-significant, as it should be")

    # Judge the result against the design committed before any outcome existed.
    # Without this, a non-significant readout is indistinguishable from a study
    # that never had the power to find the effect in the first place.
    prereg = PreRegistration.load(reports_dir)
    interpretation, corrected = None, None
    if prereg is None:
        console.log(
            "[yellow]No pre-registration found — run `make prereg` before the "
            "experiment to get a design-aware readout."
        )
    else:
        interpretation = interpret_result(
            prereg,
            ab_result["retention"]["absolute_difference"],
            ab_result["retention"]["p_value"],
        )
        corrected = holm_bonferroni(
            {
                "retention rate": ab_result["retention"]["p_value"],
                "net revenue/customer": ab_result["revenue"]["p_value"],
            },
            alpha=prereg.alpha,
        )
        print_prereg_verdict(interpretation, corrected)

    treated = outcomes[outcomes["group"] == "treatment"]
    net_gain = ab_result["revenue"]["difference"] * len(treated)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "risk_threshold": risk_threshold,
            "assumed_effect": effect,
            "offer_cost": offer_cost,
            "alpha": ALPHA,
            "seed": seed,
        },
        "population": {
            "scored_customers": len(predictions),
            "targeted_customers": len(targeted),
            "randomisation_balance_p_value": round(float(balance_p), 4),
        },
        "ab_test": ab_result,
        "aa_control": aa_result,
        "pre_registration": prereg.to_dict() if prereg else None,
        "pre_registered_verdict": interpretation,
        "multiple_comparisons": corrected,
        "business_impact": {
            "net_revenue_gain_treated_arm": round(float(net_gain), 2),
            "campaign_cost": round(float(offer_cost * len(treated)), 2),
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"ab_test_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(
        f"\nProjected net revenue from treating {len(treated)} customers: "
        f"[bold]${net_gain:,.0f}[/bold]"
    )
    console.log(f"[green]A/B test report saved: {report_path}")
    console.rule("[bold green]A/B test simulation completed!")


if __name__ == "__main__":
    typer.run(main)
