"""Pre-registered power analysis for the retention experiment.

Reporting a confidence interval after the fact answers "was this significant?".
Power analysis answers the question that should have been asked first: "how many
customers does this experiment need before it can detect the effect we care
about?"

The ordering matters, and it is enforced here rather than assumed. Computing
power from the *observed* effect after the readout is post-hoc power, which is a
deterministic (and therefore useless) function of the p-value: a non-significant
result always yields low observed power, so it can never tell you anything the
p-value did not. The only informative version fixes the effect size in advance.

So the design is committed to disk before any outcome is drawn:

    make prereg      -> reports/preregistration.json   (design, MDE, required n)
    make ab-test     -> reads it back and reports the result against that plan

Committing first also makes the honest readout possible. A non-significant
result in an under-powered study is not evidence of no effect, and the only way
to tell those apart is to know what the study could have detected before it ran.

Analytic power is cross-checked against Monte Carlo simulation of the same
estimator the A/B harness uses, because a closed form that disagrees with the
test it claims to describe is worth nothing.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()

ALPHA = 0.05
POWER_TARGET = 0.80

# The smallest retention improvement worth a campaign. Set from the economics,
# not from the data: a $50 offer against ~$840 of annual revenue needs roughly a
# 6pp lift to break even, so 5pp is the point below which we would not act.
DEFAULT_MDE = 0.05

PREREG_FILE = "preregistration.json"


def _z(alpha: float, power: float, two_sided: bool = True):
    """Critical value and power quantile for the given error rates."""
    z_alpha = stats.norm.ppf(1 - (alpha / 2 if two_sided else alpha))
    z_beta = stats.norm.ppf(power)
    return z_alpha, z_beta


def required_sample_size_proportions(baseline_rate: float, mde: float,
                                     alpha: float = ALPHA, power: float = POWER_TARGET,
                                     two_sided: bool = True) -> int:
    """Customers per arm needed to detect an absolute `mde` change in a rate.

    Uses the standard two-proportion formula, where the null term is pooled and
    the alternative term is not — under H0 both arms share a rate, under H1 they
    do not, and using one variance for both understates the requirement.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if mde <= 0:
        raise ValueError(f"mde must be positive, got {mde}")

    p1 = baseline_rate
    p2 = min(max(baseline_rate + mde, 1e-9), 1 - 1e-9)
    p_bar = (p1 + p2) / 2

    z_alpha, z_beta = _z(alpha, power, two_sided)
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(numerator / (p2 - p1) ** 2))


def power_for_sample_size(n_per_arm: int, baseline_rate: float, mde: float,
                          alpha: float = ALPHA, two_sided: bool = True) -> float:
    """Probability this many customers per arm detects an `mde` change."""
    if n_per_arm < 1:
        return 0.0
    p1 = baseline_rate
    p2 = min(max(baseline_rate + mde, 1e-9), 1 - 1e-9)
    p_bar = (p1 + p2) / 2

    z_alpha, _ = _z(alpha, 0.5, two_sided)
    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
    se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_arm)
    z_beta = (abs(p2 - p1) - z_alpha * se_null) / se_alt
    return float(stats.norm.cdf(z_beta))


def mde_for_sample_size(n_per_arm: int, baseline_rate: float,
                        alpha: float = ALPHA, power: float = POWER_TARGET) -> float:
    """Smallest absolute change this sample size can detect at the given power."""
    z_alpha, z_beta = _z(alpha, power)
    se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n_per_arm)
    return float((z_alpha + z_beta) * se)


def required_sample_size_means(sd: float, mde: float, alpha: float = ALPHA,
                               power: float = POWER_TARGET) -> int:
    """Customers per arm needed to detect an `mde` shift in a continuous metric.

    Used for net revenue per customer, whose variance is what makes that arm of
    the readout so much harder to move than retention.
    """
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    z_alpha, z_beta = _z(alpha, power)
    return int(np.ceil(2 * ((z_alpha + z_beta) * sd / mde) ** 2))


def power_curve(baseline_rate: float, mde: float, n_grid, alpha: float = ALPHA):
    """Power as a function of sample size, for the design-decision plot."""
    return [
        {"n_per_arm": int(n),
         "power": round(power_for_sample_size(int(n), baseline_rate, mde, alpha), 4)}
        for n in n_grid
    ]


def _z_test_p_value(success_a: int, n_a: int, success_b: int, n_b: int) -> float:
    """Pooled two-proportion z-test p-value.

    Deliberately duplicated in miniature rather than imported from ab_test:
    ab_test imports this module, and the simulator has to be independent of the
    thing it validates anyway. tests/test_power.py asserts the two agree.
    """
    p_a, p_b = success_a / n_a, success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (p_a - p_b) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def simulate_power(n_per_arm: int, baseline_rate: float, mde: float,
                   alpha: float = ALPHA, n_simulations: int = 2000,
                   seed: int = 42) -> float:
    """Monte Carlo power: how often the actual test rejects, over many draws.

    The analytic formula is a normal approximation to a binomial experiment.
    This runs the experiment instead, and the two are asserted to agree — a
    closed form that disagrees with the estimator it describes is not a power
    calculation, it is a guess.
    """
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(n_per_arm, min(baseline_rate + mde, 1.0), n_simulations)
    control = rng.binomial(n_per_arm, baseline_rate, n_simulations)
    rejections = sum(
        _z_test_p_value(int(t), n_per_arm, int(c), n_per_arm) < alpha
        for t, c in zip(treatment, control)
    )
    return rejections / n_simulations


def holm_bonferroni(p_values: dict, alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni correction over a family of tests.

    The readout reports retention and revenue from one experiment. Testing two
    metrics at alpha=0.05 each gives the family a ~10% false-positive rate, so
    the primary metric is designated in advance and the family is corrected.
    Holm is uniformly more powerful than Bonferroni at the same error rate.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    results, still_rejecting = {}, True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        if p >= threshold:
            still_rejecting = False
        results[name] = {
            "p_value": round(float(p), 6),
            "adjusted_threshold": round(float(threshold), 6),
            "significant_after_correction": bool(still_rejecting),
        }
    return results


@dataclass
class PreRegistration:
    """The experiment's design, fixed before any outcome is observed."""

    experiment: str
    primary_metric: str
    baseline_rate: float
    mde_absolute: float
    alpha: float
    power_target: float
    required_n_per_arm: int
    available_n_per_arm: int
    secondary_metrics: list = field(default_factory=list)
    registered_at: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def required_n_total(self) -> int:
        return self.required_n_per_arm * 2

    @property
    def is_adequately_powered(self) -> bool:
        return self.available_n_per_arm >= self.required_n_per_arm

    @property
    def achieved_power(self) -> float:
        return power_for_sample_size(
            self.available_n_per_arm, self.baseline_rate, self.mde_absolute, self.alpha
        )

    @property
    def detectable_mde(self) -> float:
        """What this sample can actually detect, as opposed to what we wanted."""
        return mde_for_sample_size(
            self.available_n_per_arm, self.baseline_rate, self.alpha, self.power_target
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "required_n_total": self.required_n_total,
            "achieved_power": round(self.achieved_power, 4),
            "detectable_mde_at_target_power": round(self.detectable_mde, 4),
            "adequately_powered": self.is_adequately_powered,
            "hypothesis": {
                "null": f"The retention offer does not change {self.primary_metric}.",
                "alternative": (
                    f"The retention offer changes {self.primary_metric} by at least "
                    f"{self.mde_absolute:.1%} in absolute terms."
                ),
                "direction": "two-sided",
            },
            "decision_rule": (
                f"Reject the null if the two-proportion z-test on {self.primary_metric} "
                f"gives p < {self.alpha}, with secondary metrics Holm-corrected across "
                f"the family. A non-significant result at n < {self.required_n_per_arm} "
                f"per arm is reported as inconclusive, not as evidence of no effect."
            ),
            "stopping_rule": (
                "No interim analyses and no optional stopping: a single readout at "
                "the planned sample size. Peeking inflates the false-positive rate "
                "well above alpha."
            ),
            "analysis_plan": {
                "primary_test": "two-proportion z-test (pooled null variance)",
                "confirmatory_test": "chi-square test of independence",
                "secondary_test": "Welch's t-test on net revenue per customer",
                "assignment": "1:1, stratified by predicted-risk decile",
                "negative_control": "an A/A run with no injected effect must be non-significant",
                "multiple_comparisons": "Holm-Bonferroni across the metric family",
            },
        }

    def write(self, reports_dir: Path) -> Path:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / PREREG_FILE
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, reports_dir: Path) -> "PreRegistration | None":
        """Read back the committed design, or None if the experiment was not registered."""
        path = Path(reports_dir) / PREREG_FILE
        if not path.exists():
            return None
        with open(path) as f:
            raw = json.load(f)
        fields = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**fields)


def interpret_result(prereg: PreRegistration, observed_effect: float,
                     p_value: float) -> dict:
    """Read the result against the committed plan, not against hindsight.

    The distinction this exists to draw: "we found no effect" and "this study
    could not have found the effect we cared about" produce the same p-value and
    mean completely different things.
    """
    significant = p_value < prereg.alpha
    powered = prereg.is_adequately_powered

    if significant:
        verdict = "significant"
        reading = (
            f"Detected a {observed_effect:+.1%} change in {prereg.primary_metric} "
            f"(p={p_value:.4f})."
        )
        if not powered:
            reading += (
                " The study was under-powered for the pre-registered MDE, so this "
                "estimate is likely inflated — statistically significant effects "
                "found in small samples exaggerate the true effect size."
            )
    elif powered:
        verdict = "null"
        reading = (
            f"No effect detected, and the study was powered to find one of "
            f"{prereg.mde_absolute:.1%}. This is evidence against an effect that large."
        )
    else:
        verdict = "inconclusive"
        reading = (
            f"No effect detected, but the study had only {prereg.achieved_power:.0%} "
            f"power against the pre-registered {prereg.mde_absolute:.1%} MDE. "
            f"It needed {prereg.required_n_per_arm:,} per arm and had "
            f"{prereg.available_n_per_arm:,}. This is inconclusive, not a null result."
        )

    return {
        "verdict": verdict,
        "reading": reading,
        "pre_registered_mde": prereg.mde_absolute,
        "observed_effect": round(float(observed_effect), 4),
        "p_value": round(float(p_value), 6),
        "achieved_power": round(prereg.achieved_power, 4),
        "required_n_per_arm": prereg.required_n_per_arm,
        "available_n_per_arm": prereg.available_n_per_arm,
        "adequately_powered": powered,
    }


def plot_power_curve(baseline_rate: float, mde: float, available_n: int,
                     required_n: int, alpha: float, out_path: Path) -> Path:
    """The design plot: how power grows with sample size, and where we sit on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_grid = np.unique(np.linspace(50, max(required_n * 1.6, available_n * 2), 200).astype(int))
    curves = {
        f"MDE {m:.0%}": [power_for_sample_size(int(n), baseline_rate, m, alpha) for n in n_grid]
        for m in (mde, mde * 1.5, mde * 2)
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, ys in curves.items():
        ax.plot(n_grid, ys, linewidth=2, label=label)

    ax.axhline(POWER_TARGET, color="grey", linestyle="--", linewidth=1)
    ax.text(n_grid[-1], POWER_TARGET + 0.015, "80% power target",
            ha="right", fontsize=9, color="grey")
    ax.axvline(required_n, color="#2a9d8f", linestyle=":", linewidth=1.6)
    ax.text(required_n, 0.04, f" required n={required_n:,}",
            fontsize=9, color="#2a9d8f")
    ax.axvline(available_n, color="#e76f51", linestyle=":", linewidth=1.6)
    ax.text(available_n, 0.9, f" available n={available_n:,}",
            fontsize=9, color="#e76f51")

    ax.set_xlabel("Customers per arm")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Power to detect a retention lift (baseline {baseline_rate:.1%}, alpha={alpha})"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _baseline_from_predictions(outputs_dir: Path, risk_threshold: float):
    """Expected retention and arm size for the targeted population.

    Both come from the scored batch, which exists before any outcome is drawn —
    the pre-registration must not touch experiment results.
    """
    import pandas as pd

    files = sorted(Path(outputs_dir).glob("predictions_*.parquet"), reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No predictions in {outputs_dir}. Run `make score` first."
        )
    predictions = pd.read_parquet(files[0])
    targeted = predictions[predictions["churn_probability"] >= risk_threshold]
    if targeted.empty:
        raise ValueError(f"No customers scored at or above {risk_threshold}")

    baseline_retention = float(1 - targeted["churn_probability"].mean())
    revenue_sd = float((targeted["MonthlyCharges"] * 12).std()) \
        if "MonthlyCharges" in targeted.columns else 0.0
    return baseline_retention, len(targeted) // 2, revenue_sd


def main(
    outputs_dir: Path = Path.cwd() / "outputs",
    reports_dir: Path = Path.cwd() / "reports",
    docs_dir: Path = Path.cwd() / "docs" / "images",
    risk_threshold: float = typer.Option(0.5, help="Target customers at or above this churn score"),
    mde: float = typer.Option(DEFAULT_MDE, help="Smallest retention lift worth acting on"),
    alpha: float = typer.Option(ALPHA, help="Significance level"),
    power: float = typer.Option(POWER_TARGET, help="Target power"),
    n_simulations: int = typer.Option(2000, help="Monte Carlo draws for the power check"),
):
    console.rule("[bold magenta]Telco Churn: Pre-Registered Power Analysis")

    baseline, available_n, revenue_sd = _baseline_from_predictions(outputs_dir, risk_threshold)
    console.log(
        f"[blue]Targeted population: {available_n * 2} customers "
        f"({available_n} per arm), expected retention {baseline:.1%}"
    )

    required_n = required_sample_size_proportions(baseline, mde, alpha, power)
    analytic = power_for_sample_size(available_n, baseline, mde, alpha)
    simulated = simulate_power(available_n, baseline, mde, alpha, n_simulations)

    console.log(
        f"[blue]Validating the closed form against {n_simulations:,} simulated experiments..."
    )
    gap = abs(analytic - simulated)
    if gap > 0.05:
        console.log(
            f"[yellow]Analytic and simulated power differ by {gap:.3f} — "
            "the normal approximation is straining at this sample size."
        )
    else:
        console.log(f"[green]Analytic {analytic:.3f} vs simulated {simulated:.3f} — agrees")

    # Revenue is the secondary metric and the harder one: a $50 shift against
    # annual revenue with this much spread needs a far larger sample.
    revenue_mde = 50.0
    required_n_revenue = (
        required_sample_size_means(revenue_sd, revenue_mde, alpha, power)
        if revenue_sd > 0 else None
    )

    prereg = PreRegistration(
        experiment="retention_offer_v1",
        primary_metric="retention rate",
        baseline_rate=round(baseline, 4),
        mde_absolute=mde,
        alpha=alpha,
        power_target=power,
        required_n_per_arm=required_n,
        available_n_per_arm=available_n,
        secondary_metrics=["net revenue per customer"],
        registered_at=datetime.now().isoformat(),
        notes={
            "mde_rationale": (
                "A $50 offer against roughly $840 of annual revenue per retained "
                "customer breaks even near a 6pp lift, so 5pp is the point below "
                "which the campaign would not be run regardless of significance."
            ),
            "baseline_source": (
                "Mean predicted retention among targeted customers, taken from the "
                "scored batch before any outcome is drawn."
            ),
            "analytic_power_at_available_n": round(analytic, 4),
            "simulated_power_at_available_n": round(simulated, 4),
            "simulation_draws": n_simulations,
            "revenue_sd": round(revenue_sd, 2),
            "revenue_mde_dollars": revenue_mde,
            "required_n_per_arm_revenue": required_n_revenue,
        },
    )

    table = Table(title="Pre-registered design")
    table.add_column("Parameter")
    table.add_column("Value", justify="right")
    table.add_row("Primary metric", prereg.primary_metric)
    table.add_row("Baseline retention", f"{baseline:.1%}")
    table.add_row("Minimum detectable effect", f"{mde:.1%} absolute")
    table.add_row("Alpha / power target", f"{alpha} / {power:.0%}")
    table.add_row("Required n per arm", f"{required_n:,}")
    table.add_row("Available n per arm", f"{available_n:,}")
    table.add_row("Power at available n", f"{analytic:.1%}")
    table.add_row("Detectable MDE at available n", f"{prereg.detectable_mde:.1%}")
    if required_n_revenue:
        table.add_row("Required n per arm (revenue, $50)", f"{required_n_revenue:,}")
    console.print(table)

    if prereg.is_adequately_powered:
        console.log(f"[green]Adequately powered: {available_n:,} >= {required_n:,} per arm")
    else:
        console.log(
            f"[yellow]Under-powered by design: {available_n:,} of the {required_n:,} "
            f"per arm needed. A non-significant readout will be reported as "
            f"inconclusive, not as a null result."
        )

    path = prereg.write(reports_dir)
    console.log(f"[green]Pre-registration written: {path}")

    plot_path = plot_power_curve(
        baseline, mde, available_n, required_n, alpha, Path(docs_dir) / "power-curve.png"
    )
    console.log(f"[green]Power curve saved: {plot_path}")
    console.rule("[bold green]Power analysis completed!")


if __name__ == "__main__":
    typer.run(main)
