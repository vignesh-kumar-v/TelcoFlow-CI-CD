"""Causal inference: what actually *causes* churn, as distinct from what predicts it.

The rest of this project is predictive. A model that ranks churn risk well is
answering "who is likely to leave?", and that question is enough to decide whom
to call. It is not enough to decide *what to offer them*, because the features
that predict churn are not the features that change it. Customers on two-year
contracts churn less, but a customer who would have stayed anyway is also the
customer most willing to sign a two-year contract — so the raw gap between
contract types is mostly selection, and acting on it would waste the budget.

Four analyses, each answering a question prediction cannot:

  1. **Propensity scoring** (matching, IPW, AIPW) — the effect of taking Tech
     Support on churn, from observational data, adjusting for who selects into
     it. Includes the overlap check, covariate balance before and after, and a
     doubly-robust estimator that is consistent if *either* the treatment model
     or the outcome model is right.
  2. **Instrumental variables** (2SLS) — the retention campaign as an
     encouragement design. Customers are randomly *offered* the retention deal,
     but only some redeem it, and the ones who redeem are not a random subset.
     Comparing redeemers to non-redeemers is confounded; the random offer is an
     instrument that recovers the effect on compliers.
  3. **Difference-in-differences** — a two-period rollout where the treated
     group already had higher churn before anything happened, so the post-period
     comparison is biased and the pre/post difference removes it. Includes the
     parallel-trends test the design actually rests on.
  4. **Refutation** — placebo treatments, subset stability, and an E-value for
     how strong an unmeasured confounder would have to be to explain the result
     away. An estimate nobody tried to break is not evidence.

Estimators 2 and 3 run on constructed data, and that is deliberate rather than a
shortcut: the dataset is a static snapshot with no time dimension and no
experiment, so there is no DiD or IV to be had in it. Building the data-
generating process means the true effect is known, which turns each estimator
into a testable claim — 2SLS must recover the injected complier effect where
naive OLS does not, and DiD must recover the injected rollout effect where the
post-period comparison does not. tests/test_causal.py asserts exactly that.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.telco_churn.features import (  # noqa: E402
    ADDON_SERVICE_COLUMNS, TARGET, compute_engineered_features,
)
from src.telco_churn.inference import LoadedModel  # noqa: E402

console = Console()

ALPHA = 0.05
RANDOM_STATE = 42

# The observational question: does buying Tech Support keep customers?
TREATMENT_COLUMN = "TechSupport"
TREATMENT_VALUE = "Yes"

# Confounders for the primary specification.
#
# MonthlyCharges is deliberately absent. Tech Support is a paid add-on, so it
# *raises* the monthly bill: conditioning on the bill would block part of the
# very effect being estimated. It is a mediator, not a confounder, and
# controlling for it is the classic bad-control mistake. It is added back in a
# labelled sensitivity specification so the difference is visible rather than
# assumed away.
CONFOUNDERS_NUMERIC = ["tenure", "num_other_addons"]
CONFOUNDERS_CATEGORICAL = [
    "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
    "Partner", "Dependents", "SeniorCitizen", "gender", "PhoneService",
    "MultipleLines",
]
MEDIATOR_CONTROLS = ["MonthlyCharges"]

# |SMD| below this is the conventional line for "balanced enough to compare".
SMD_THRESHOLD = 0.1


# --- Regression primitives ----------------------------------------------
#
# Implemented directly rather than pulled from statsmodels: the project pins its
# dependencies from a verified environment, and OLS with heteroskedasticity-
# robust errors plus two-stage least squares is about sixty lines. Binary
# outcomes and group-level treatment both produce non-constant error variance,
# so classical standard errors would be wrong here by default.

@dataclass
class RegressionResult:
    names: list
    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    conf_int: np.ndarray
    n: int

    def term(self, name: str) -> dict:
        i = self.names.index(name)
        return {
            "coefficient": round(float(self.coefficients[i]), 6),
            "std_error": round(float(self.std_errors[i]), 6),
            "t_statistic": round(float(self.t_stats[i]), 4),
            "p_value": round(float(self.p_values[i]), 6),
            "ci_95": [round(float(self.conf_int[i, 0]), 6),
                      round(float(self.conf_int[i, 1]), 6)],
            "significant": bool(self.p_values[i] < ALPHA),
        }


def _inference(beta, cov, names, n, k) -> RegressionResult:
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    dof = max(n - k, 1)
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    crit = stats.t.ppf(1 - ALPHA / 2, dof)
    ci = np.column_stack([beta - crit * se, beta + crit * se])
    return RegressionResult(names, beta, se, t, p, ci, n)


def ols(y: np.ndarray, X: np.ndarray, names: list) -> RegressionResult:
    """OLS with HC1 heteroskedasticity-robust standard errors.

    HC1 rather than classical: with a binary outcome the error variance is
    p(1-p), which varies with the fitted value by construction, so the
    homoskedastic formula understates the uncertainty.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtX_inv @ meat @ XtX_inv * (n / max(n - k, 1))
    return _inference(beta, cov, names, n, k)


def two_stage_least_squares(y: np.ndarray, endog: np.ndarray, instruments: np.ndarray,
                            exog: np.ndarray, names: list) -> tuple:
    """2SLS: project the endogenous regressor onto the instruments, then regress.

    `exog` must already carry the intercept. Standard errors use residuals from
    the *structural* equation — computed with the actual endogenous regressor,
    not its fitted values — because the second-stage residuals from the
    projection understate the true variance.
    """
    y = np.asarray(y, dtype=float)
    endog = np.asarray(endog, dtype=float).reshape(len(y), -1)
    instruments = np.asarray(instruments, dtype=float).reshape(len(y), -1)
    exog = np.asarray(exog, dtype=float)

    Z = np.hstack([instruments, exog])          # full instrument set
    endog_hat = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ endog

    X_hat = np.hstack([endog_hat, exog])
    X_true = np.hstack([endog, exog])
    n, k = X_hat.shape

    XtX_inv = np.linalg.pinv(X_hat.T @ X_hat)
    beta = XtX_inv @ X_hat.T @ y
    resid = y - X_true @ beta                   # structural residuals

    meat = (X_hat * resid[:, None]).T @ (X_hat * resid[:, None])
    cov = XtX_inv @ meat @ XtX_inv * (n / max(n - k, 1))
    second_stage = _inference(beta, cov, names, n, k)

    # First stage, and the F-statistic on the excluded instruments. Below ~10
    # the instrument is weak and 2SLS is biased toward the OLS estimate it was
    # meant to fix, so this is a precondition, not a diagnostic afterthought.
    first_stage_names = [f"z{i}" for i in range(instruments.shape[1])] + \
                        [f"x{i}" for i in range(exog.shape[1])]
    first_stage = ols(endog[:, 0], Z, first_stage_names)

    restricted = exog @ np.linalg.pinv(exog.T @ exog) @ exog.T @ endog[:, 0]
    ssr_restricted = float(((endog[:, 0] - restricted) ** 2).sum())
    ssr_full = float(((endog[:, 0] - Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ endog[:, 0]) ** 2).sum())
    q = instruments.shape[1]
    f_stat = ((ssr_restricted - ssr_full) / q) / (ssr_full / max(n - Z.shape[1], 1))

    return second_stage, {
        "first_stage_coefficient": round(float(first_stage.coefficients[0]), 6),
        "first_stage_f_statistic": round(float(f_stat), 2),
        "weak_instrument": bool(f_stat < 10),
    }


# --- Propensity scoring -------------------------------------------------

def prepare_observational(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the population that could actually take the treatment.

    Customers without internet cannot buy Tech Support, and their "No internet
    service" value is not a declined add-on — it is ineligibility. Leaving them
    in the control arm compares subscribers against people who were never
    offered the product, which is a comparison of two different populations
    rather than an effect.
    """
    if "num_addon_services" not in df.columns:
        df = compute_engineered_features(df)

    eligible = df[df[TREATMENT_COLUMN].isin(["Yes", "No"])].copy()

    # num_addon_services counts Tech Support itself, so it is mechanically
    # determined by the treatment. Recount without it.
    others = [c for c in ADDON_SERVICE_COLUMNS if c != TREATMENT_COLUMN]
    eligible["num_other_addons"] = sum(
        (eligible[col] == TREATMENT_VALUE).astype(int) for col in others
    )
    eligible["treated"] = (eligible[TREATMENT_COLUMN] == TREATMENT_VALUE).astype(int)
    eligible["outcome"] = eligible[TARGET].astype(int)
    return eligible.reset_index(drop=True)


def build_confounder_matrix(df: pd.DataFrame, include_mediators: bool = False):
    """Design matrix of confounders, one-hot encoded with a reference level."""
    numeric = list(CONFOUNDERS_NUMERIC) + (list(MEDIATOR_CONTROLS) if include_mediators else [])
    parts = [df[numeric].astype(float).reset_index(drop=True)]
    for col in CONFOUNDERS_CATEGORICAL:
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
        parts.append(dummies.astype(float).reset_index(drop=True))
    return pd.concat(parts, axis=1)


def estimate_propensity(X: pd.DataFrame, treated: np.ndarray,
                        random_state: int = RANDOM_STATE) -> np.ndarray:
    """P(treated | confounders), the score that makes the arms comparable."""
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(scaler.fit_transform(X), treated)
    return model.predict_proba(scaler.transform(X))[:, 1]


def check_overlap(propensity: np.ndarray, treated: np.ndarray,
                  trim: float = 0.02) -> dict:
    """Common support: units with no counterpart cannot be matched, only extrapolated.

    Anyone whose score falls outside the region both arms occupy is dropped.
    Keeping them would mean the estimate rests on the model's guess about a
    customer type that only ever appears in one arm.
    """
    treated_scores = propensity[treated == 1]
    control_scores = propensity[treated == 0]
    low = max(treated_scores.min(), control_scores.min(), trim)
    high = min(treated_scores.max(), control_scores.max(), 1 - trim)
    in_support = (propensity >= low) & (propensity <= high)
    return {
        "support_range": [round(float(low), 4), round(float(high), 4)],
        "n_total": int(len(propensity)),
        "n_in_support": int(in_support.sum()),
        "n_dropped": int((~in_support).sum()),
        "share_dropped": round(float((~in_support).mean()), 4),
        "mask": in_support,
    }


def standardised_mean_differences(X: pd.DataFrame, treated: np.ndarray,
                                  weights: np.ndarray = None) -> pd.Series:
    """Covariate imbalance on a scale that does not depend on sample size.

    A t-test would conflate imbalance with n; SMD does not, which is why it is
    the standard balance diagnostic. |SMD| < 0.1 is the conventional target.
    """
    w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=float)
    t, c = treated == 1, treated == 0

    def wmean(v, mask):
        return float(np.average(v[mask], weights=w[mask]))

    def wvar(v, mask):
        m = wmean(v, mask)
        return float(np.average((v[mask] - m) ** 2, weights=w[mask]))

    out = {}
    for col in X.columns:
        v = X[col].to_numpy(dtype=float)
        pooled_sd = np.sqrt((wvar(v, t) + wvar(v, c)) / 2)
        out[col] = 0.0 if pooled_sd == 0 else (wmean(v, t) - wmean(v, c)) / pooled_sd
    return pd.Series(out)


def match_nearest_neighbour(propensity: np.ndarray, treated: np.ndarray,
                            caliper_sd: float = 0.2):
    """Greedy 1:1 matching on the propensity logit, within a caliper.

    Matching on the logit rather than the raw score keeps distances meaningful
    near 0 and 1, where the probability scale is compressed. The caliper caps
    how bad a match is allowed to be: without it, a treated unit with no
    plausible counterpart still gets one, and its "effect" is pure extrapolation.
    Matching is without replacement, so no control unit props up several treated
    units and silently dominates the estimate.
    """
    eps = 1e-6
    p = np.clip(propensity, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    caliper = caliper_sd * float(np.std(logit))

    treated_idx = np.flatnonzero(treated == 1)
    control_idx = np.flatnonzero(treated == 0)

    # Match the hardest treated units first: once controls are consumed, the
    # units matched last get the worst remaining partners, and the ones with the
    # fewest candidates are the ones that would otherwise go unmatched.
    order = treated_idx[np.argsort(-logit[treated_idx])]
    available = list(control_idx)
    available_logits = logit[control_idx].tolist()

    pairs, unmatched = [], 0
    for t_i in order:
        if not available:
            unmatched += len(order) - len(pairs) - unmatched
            break
        distances = np.abs(np.asarray(available_logits) - logit[t_i])
        j = int(np.argmin(distances))
        if distances[j] > caliper:
            unmatched += 1
            continue
        pairs.append((int(t_i), int(available[j])))
        available.pop(j)
        available_logits.pop(j)

    return pairs, {
        "caliper_logit": round(float(caliper), 4),
        "n_treated": int(len(treated_idx)),
        "n_matched_pairs": int(len(pairs)),
        "n_unmatched_treated": int(unmatched),
        "match_rate": round(len(pairs) / max(len(treated_idx), 1), 4),
    }


def att_from_matches(outcome: np.ndarray, pairs: list) -> float:
    """Average treatment effect on the treated, from matched pairs."""
    if not pairs:
        return float("nan")
    t_idx = [p[0] for p in pairs]
    c_idx = [p[1] for p in pairs]
    return float(np.mean(outcome[t_idx] - outcome[c_idx]))


def inverse_probability_weights(propensity: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """Stabilised IPW weights.

    Stabilisation multiplies by the marginal treatment probability, which keeps
    a unit with a near-zero propensity from acquiring an enormous weight and
    single-handedly setting the estimate.
    """
    p = np.clip(propensity, 0.01, 0.99)
    marginal = float(np.mean(treated))
    return np.where(treated == 1, marginal / p, (1 - marginal) / (1 - p))


def ipw_ate(outcome: np.ndarray, treated: np.ndarray, weights: np.ndarray) -> float:
    t, c = treated == 1, treated == 0
    return float(
        np.average(outcome[t], weights=weights[t])
        - np.average(outcome[c], weights=weights[c])
    )


def aipw_ate(outcome: np.ndarray, treated: np.ndarray, propensity: np.ndarray,
             X: pd.DataFrame, random_state: int = RANDOM_STATE) -> float:
    """Augmented IPW — the doubly-robust estimator.

    Consistent if *either* the propensity model or the outcome model is right,
    not both. When matching and IPW disagree, this is the one to trust, because
    it does not stake the answer on a single modelling choice being correct.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    p = np.clip(propensity, 0.01, 0.99)

    predictions = {}
    for arm in (0, 1):
        mask = treated == arm
        if len(np.unique(outcome[mask])) < 2:
            predictions[arm] = np.full(len(outcome), float(outcome[mask].mean()))
            continue
        model = LogisticRegression(max_iter=2000, random_state=random_state)
        model.fit(X_scaled[mask], outcome[mask])
        predictions[arm] = model.predict_proba(X_scaled)[:, 1]

    influence = (
        predictions[1] - predictions[0]
        + treated * (outcome - predictions[1]) / p
        - (1 - treated) * (outcome - predictions[0]) / (1 - p)
    )
    return float(np.mean(influence))


def bootstrap_ci(estimator, n: int, n_boot: int = 200, seed: int = RANDOM_STATE):
    """Percentile bootstrap CI, for estimators with no closed-form variance.

    Matching in particular has no usable analytic standard error: the matched
    pairs are constructed from the data, so treating them as fixed understates
    the uncertainty.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            value = estimator(idx)
        except Exception:
            continue
        if value is not None and np.isfinite(value):
            draws.append(value)
    if len(draws) < 20:
        return [float("nan"), float("nan")], float("nan")
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return [round(float(lo), 4), round(float(hi), 4)], round(float(np.std(draws)), 4)


def e_value(risk_ratio: float) -> float:
    """How strong an unmeasured confounder would have to be to explain this away.

    Reported on the risk-ratio scale: an E-value of 2.0 means a lurking variable
    would need to at least double both the chance of treatment and the chance of
    the outcome, above and beyond every covariate already adjusted for. It turns
    "but what about unobserved confounding?" from an unanswerable objection into
    a number a domain expert can judge.
    """
    rr = risk_ratio if risk_ratio >= 1 else 1 / risk_ratio
    return float(rr + np.sqrt(rr * (rr - 1)))


# --- Instrumental variables: the encouragement design -------------------

def simulate_encouragement(scored: pd.DataFrame, complier_effect: float = -0.15,
                           base_uptake: float = 0.45, always_taker_rate: float = 0.08,
                           seed: int = RANDOM_STATE) -> pd.DataFrame:
    """A randomised offer that customers can decline — the realistic campaign.

    Real retention campaigns do not assign treatment, they assign *eligibility*.
    The offer is mailed at random (Z), but redemption (D) is the customer's
    choice, and engaged customers redeem more. That choice is what breaks the
    naive comparison: redeemers were already less likely to leave, so the gap
    between redeemers and non-redeemers overstates the offer by the amount that
    engagement predicts retention.

    The construction below encodes exactly that. `engagement` raises redemption
    and lowers churn independently, so it confounds D with Y while leaving Z
    untouched — which is what makes Z a valid instrument and OLS a biased one.
    """
    rng = np.random.default_rng(seed)
    df = scored.copy().reset_index(drop=True)
    n = len(df)

    engagement = rng.normal(0, 1, n)
    offered = rng.integers(0, 2, n)                                  # Z: randomised

    uptake_probability = np.clip(base_uptake + 0.20 * engagement, 0.02, 0.95)
    redeems_if_offered = rng.random(n) < uptake_probability

    # Always-takers seek the deal out whether or not it was mailed to them, and
    # they are the most engaged customers of all. This is what makes the treated
    # group non-random even among customers who were never offered anything.
    always_take_probability = np.clip(always_taker_rate + 0.07 * engagement, 0.0, 0.6)
    always_takes = rng.random(n) < always_take_probability

    # Monotonicity: no defiers. Nobody redeems only when *not* offered.
    redeemed = np.where(offered == 1, redeems_if_offered | always_takes, always_takes).astype(int)

    baseline_risk = df["churn_probability"].to_numpy()
    untreated_risk = np.clip(baseline_risk - 0.09 * engagement, 0.01, 0.99)
    treated_risk = np.clip(baseline_risk + complier_effect - 0.09 * engagement, 0.01, 0.99)

    true_risk = np.where(redeemed == 1, treated_risk, untreated_risk)
    churned = (rng.random(n) < true_risk).astype(int)

    # The estimand 2SLS actually targets, computed from the data-generating
    # process rather than assumed to equal the injected parameter. Two things
    # separate them: risk is bounded, so the offer cannot take a customer at 8%
    # risk down by 15 points, and LATE is the effect on *compliers* — customers
    # who redeem when offered and not otherwise — rather than on everyone.
    # Benchmarking against the injected parameter would make a correct estimator
    # look biased.
    compliers = redeems_if_offered & ~always_takes
    realised = float(np.mean((treated_risk - untreated_risk)[compliers])) \
        if compliers.any() else complier_effect

    df["offered"] = offered
    df["redeemed"] = redeemed
    df["churned"] = churned
    df["engagement"] = engagement          # unobserved in the analysis; kept for the test
    df["is_complier"] = compliers.astype(int)
    df["injected_effect"] = complier_effect
    df["true_complier_effect"] = round(realised, 4)
    return df


def instrumental_variables_analysis(df: pd.DataFrame) -> dict:
    """ITT, first stage, Wald/LATE and 2SLS, against the biased naive comparison."""
    y = df["churned"].to_numpy(dtype=float)
    d = df["redeemed"].to_numpy(dtype=float)
    z = df["offered"].to_numpy(dtype=float)
    const = np.ones((len(df), 1))

    # What comparing redeemers to non-redeemers would tell you. It is wrong.
    naive = ols(y, np.column_stack([d, const]), ["redeemed", "const"])

    # Intention to treat: the effect of *being offered*, which is randomised and
    # therefore unbiased — but it answers a different question, diluted by
    # everyone who ignored the offer.
    itt = ols(y, np.column_stack([z, const]), ["offered", "const"])
    first_stage = ols(d, np.column_stack([z, const]), ["offered", "const"])

    compliance = float(first_stage.coefficients[0])
    wald = float(itt.coefficients[0] / compliance) if compliance else float("nan")

    tsls, diagnostics = two_stage_least_squares(y, d, z, const, ["redeemed", "const"])

    return {
        "n": int(len(df)),
        "naive_ols": naive.term("redeemed"),
        "intention_to_treat": itt.term("offered"),
        "first_stage": {**first_stage.term("offered"), **diagnostics},
        "compliance_rate": round(compliance, 4),
        "wald_estimator_late": round(wald, 4),
        "two_stage_least_squares": tsls.term("redeemed"),
        "assumptions": {
            "relevance": (
                f"First-stage F = {diagnostics['first_stage_f_statistic']:.1f} "
                f"({'weak' if diagnostics['weak_instrument'] else 'strong'}; the "
                f"conventional threshold is 10)."
            ),
            "exclusion": (
                "The offer can only affect churn through redemption. It holds here "
                "by construction; in a real campaign, an offer letter that reminds "
                "an inattentive customer their contract exists would violate it."
            ),
            "monotonicity": (
                "No defiers: nobody redeems only when not offered. Enforced in the "
                "simulation and untestable in the field."
            ),
            "interpretation": (
                "2SLS identifies the effect on compliers — customers who redeem "
                "when offered and not otherwise — which is not the population "
                "average, and is the group a campaign can actually move."
            ),
        },
    }


# --- Difference-in-differences ------------------------------------------

def simulate_did_panel(scored: pd.DataFrame, treatment_effect: float = -0.08,
                       group_gap: float = 0.10, common_shock: float = 0.05,
                       n_pre_periods: int = 2, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """A two-group, multi-period rollout where the groups were never equal.

    The campaign goes to the higher-churn region first, which is what actually
    happens — you pilot where the problem is. That makes the post-period
    treated-vs-control comparison biased by `group_gap`, and a before-and-after
    comparison within the treated group biased by `common_shock`. Only the
    double difference removes both.

    Extra pre-periods are generated so parallel trends can be *tested* rather
    than asserted, which is the assumption the whole design rests on.
    """
    rng = np.random.default_rng(seed)
    base = scored.reset_index(drop=True)
    n = len(base)

    treated_group = rng.integers(0, 2, n)
    customer_effect = rng.normal(0, 0.05, n)

    rows = []
    realised_effect = treatment_effect
    periods = list(range(-n_pre_periods, 1))            # ..., -1, 0 = post
    for period in periods:
        post = int(period == 0)
        untreated = np.clip(
            base["churn_probability"].to_numpy() * 0.35
            + group_gap * treated_group                  # the level difference
            + common_shock * post                        # hits both groups equally
            + customer_effect,
            0.01, 0.99,
        )
        risk = np.clip(untreated + treatment_effect * treated_group * post, 0.01, 0.99)
        if post:
            # As with LATE above: churn risk is bounded, so the realised effect
            # among the treated is what the estimator can recover, not the
            # injected parameter.
            treated_mask = treated_group == 1
            realised_effect = float(np.mean((risk - untreated)[treated_mask]))
        rows.append(pd.DataFrame({
            "customer_index": np.arange(n),
            "period": period,
            "post": post,
            "treated_group": treated_group,
            "churned": (rng.random(n) < risk).astype(int),
        }))

    panel = pd.concat(rows, ignore_index=True)
    panel.attrs["true_effect"] = round(realised_effect, 4)
    panel.attrs["injected_effect"] = treatment_effect
    return panel


def difference_in_differences(panel: pd.DataFrame) -> dict:
    """The 2x2 table, the regression with an interaction, and the trends test."""
    main = panel[panel["period"].isin([-1, 0])]

    cells = main.groupby(["treated_group", "post"])["churned"].mean().unstack()
    treated_change = float(cells.loc[1, 1] - cells.loc[1, 0])
    control_change = float(cells.loc[0, 1] - cells.loc[0, 0])

    y = main["churned"].to_numpy(dtype=float)
    g = main["treated_group"].to_numpy(dtype=float)
    t = main["post"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(main)), g, t, g * t])
    model = ols(y, X, ["const", "treated_group", "post", "did"])

    # The two comparisons DiD exists to replace, so the bias is visible.
    naive_post = float(cells.loc[1, 1] - cells.loc[0, 1])
    naive_prepost = treated_change

    return {
        "means": {
            "treated_pre": round(float(cells.loc[1, 0]), 4),
            "treated_post": round(float(cells.loc[1, 1]), 4),
            "control_pre": round(float(cells.loc[0, 0]), 4),
            "control_post": round(float(cells.loc[0, 1]), 4),
        },
        "treated_change": round(treated_change, 4),
        "control_change": round(control_change, 4),
        "did_estimate": round(treated_change - control_change, 4),
        "regression": model.term("did"),
        "naive_post_period_comparison": round(naive_post, 4),
        "naive_before_after_comparison": round(naive_prepost, 4),
        "parallel_trends": parallel_trends_test(panel),
    }


def parallel_trends_test(panel: pd.DataFrame) -> dict:
    """The assumption DiD stands on, tested on the periods before treatment.

    If the treated and control groups were already diverging beforehand, the
    post-period gap is that pre-existing trend continuing, and DiD attributes it
    to the campaign. This cannot prove parallel trends — only that they had not
    visibly broken before treatment — which is the strongest evidence the design
    admits.
    """
    pre = panel[panel["period"] < 0]
    if pre["period"].nunique() < 2:
        return {"testable": False,
                "reason": "needs at least two pre-treatment periods"}

    gaps = (
        pre.groupby(["period", "treated_group"])["churned"].mean()
        .unstack().pipe(lambda d: d[1] - d[0])
    )
    y = pre["churned"].to_numpy(dtype=float)
    g = pre["treated_group"].to_numpy(dtype=float)
    t = pre["period"].to_numpy(dtype=float)
    placebo = ols(y, np.column_stack([np.ones(len(pre)), g, t, g * t]),
                  ["const", "treated_group", "period", "pre_trend"])
    term = placebo.term("pre_trend")

    return {
        "testable": True,
        "pre_period_gaps": {int(k): round(float(v), 4) for k, v in gaps.items()},
        "gap_drift": round(float(gaps.max() - gaps.min()), 4),
        "placebo_interaction": term,
        "trends_parallel": not term["significant"],
        "reading": (
            "The treated-control gap does not move significantly before treatment, "
            "so the design's key assumption survives the strongest test available."
            if not term["significant"] else
            "The groups were already diverging before treatment — the DiD estimate "
            "picks up that trend and should not be read as an effect."
        ),
    }


# --- Observational pipeline and refutation ------------------------------

@dataclass
class ObservationalResult:
    spec: str
    n: int
    naive_difference: float
    estimates: dict
    balance: dict
    overlap: dict
    matching: dict
    diagnostics: dict = field(default_factory=dict)


def run_observational(df: pd.DataFrame, include_mediators: bool = False,
                      n_boot: int = 200, seed: int = RANDOM_STATE) -> ObservationalResult:
    """Estimate the Tech Support effect three ways and report whether they agree."""
    X_all = build_confounder_matrix(df, include_mediators=include_mediators)
    treated_all = df["treated"].to_numpy()
    outcome_all = df["outcome"].to_numpy()

    naive = float(outcome_all[treated_all == 1].mean() - outcome_all[treated_all == 0].mean())

    propensity_all = estimate_propensity(X_all, treated_all, random_state=seed)
    overlap = check_overlap(propensity_all, treated_all)
    mask = overlap.pop("mask")

    X = X_all[mask].reset_index(drop=True)
    treated = treated_all[mask]
    outcome = outcome_all[mask]
    propensity = estimate_propensity(X, treated, random_state=seed)

    smd_before = standardised_mean_differences(X, treated)

    pairs, match_info = match_nearest_neighbour(propensity, treated)
    att = att_from_matches(outcome, pairs)

    matched_rows = [i for pair in pairs for i in pair]
    smd_after = standardised_mean_differences(
        X.iloc[matched_rows].reset_index(drop=True), treated[matched_rows]
    )

    weights = inverse_probability_weights(propensity, treated)
    ate_ipw = ipw_ate(outcome, treated, weights)
    smd_weighted = standardised_mean_differences(X, treated, weights=weights)
    ate_aipw = aipw_ate(outcome, treated, propensity, X, random_state=seed)

    n = len(outcome)

    def boot_matching(idx):
        p = estimate_propensity(X.iloc[idx].reset_index(drop=True), treated[idx], random_state=seed)
        b_pairs, _ = match_nearest_neighbour(p, treated[idx])
        return att_from_matches(outcome[idx], b_pairs)

    def boot_ipw(idx):
        p = estimate_propensity(X.iloc[idx].reset_index(drop=True), treated[idx], random_state=seed)
        return ipw_ate(outcome[idx], treated[idx], inverse_probability_weights(p, treated[idx]))

    def boot_aipw(idx):
        Xi = X.iloc[idx].reset_index(drop=True)
        p = estimate_propensity(Xi, treated[idx], random_state=seed)
        return aipw_ate(outcome[idx], treated[idx], p, Xi, random_state=seed)

    ci_match, se_match = bootstrap_ci(boot_matching, n, n_boot=n_boot, seed=seed)
    ci_ipw, se_ipw = bootstrap_ci(boot_ipw, n, n_boot=n_boot, seed=seed + 1)
    ci_aipw, se_aipw = bootstrap_ci(boot_aipw, n, n_boot=n_boot, seed=seed + 2)

    # Risk ratio for the E-value, from the matched pairs.
    t_rate = float(outcome[[p[0] for p in pairs]].mean()) if pairs else float("nan")
    c_rate = float(outcome[[p[1] for p in pairs]].mean()) if pairs else float("nan")
    risk_ratio = t_rate / c_rate if c_rate else float("nan")

    estimates = {
        "matching_att": {
            "estimate": round(att, 4), "ci_95": ci_match, "bootstrap_se": se_match,
            "method": "1:1 nearest-neighbour on the propensity logit, caliper 0.2 SD",
        },
        "ipw_ate": {
            "estimate": round(ate_ipw, 4), "ci_95": ci_ipw, "bootstrap_se": se_ipw,
            "method": "stabilised inverse probability weighting",
        },
        "aipw_ate": {
            "estimate": round(ate_aipw, 4), "ci_95": ci_aipw, "bootstrap_se": se_aipw,
            "method": "augmented IPW (doubly robust)",
        },
    }
    spread = max(e["estimate"] for e in estimates.values()) - \
        min(e["estimate"] for e in estimates.values())

    return ObservationalResult(
        spec="with mediator controls" if include_mediators else "primary",
        n=n,
        naive_difference=round(naive, 4),
        estimates=estimates,
        balance={
            "max_abs_smd_before": round(float(smd_before.abs().max()), 4),
            "max_abs_smd_after_matching": round(float(smd_after.abs().max()), 4),
            "max_abs_smd_after_weighting": round(float(smd_weighted.abs().max()), 4),
            "n_imbalanced_before": int((smd_before.abs() > SMD_THRESHOLD).sum()),
            "n_imbalanced_after_matching": int((smd_after.abs() > SMD_THRESHOLD).sum()),
            "threshold": SMD_THRESHOLD,
            "balanced": bool(smd_after.abs().max() <= SMD_THRESHOLD),
            "per_covariate": {
                col: {"before": round(float(smd_before[col]), 4),
                      "after_matching": round(float(smd_after[col]), 4)}
                for col in X.columns
            },
        },
        overlap=overlap,
        matching=match_info,
        diagnostics={
            "selection_bias_removed": round(naive - att, 4),
            "estimator_spread": round(float(spread), 4),
            "estimators_agree": bool(spread < 0.03),
            "risk_ratio": round(float(risk_ratio), 4) if np.isfinite(risk_ratio) else None,
            "e_value": round(e_value(risk_ratio), 3) if np.isfinite(risk_ratio) else None,
            "effective_sample_size": int(round(weights.sum() ** 2 / (weights ** 2).sum())),
        },
    )


def refute(df: pd.DataFrame, observed_att: float, n_placebo: int = 12,
           seed: int = RANDOM_STATE) -> dict:
    """Try to break the estimate. An estimate nobody attacked is not evidence.

    Two refuters. A placebo scrambles the treatment assignment: the whole
    pipeline should then find nothing, and if it still finds an effect the
    machinery is manufacturing one. A subset refuter re-estimates on random
    70% samples: a result that swings wildly across subsamples was driven by a
    handful of rows.
    """
    rng = np.random.default_rng(seed)
    X = build_confounder_matrix(df)
    outcome = df["outcome"].to_numpy()
    real_treated = df["treated"].to_numpy()

    placebo_estimates = []
    for _ in range(n_placebo):
        fake = rng.permutation(real_treated)
        p = estimate_propensity(X, fake, random_state=seed)
        pairs, _ = match_nearest_neighbour(p, fake)
        value = att_from_matches(outcome, pairs)
        if np.isfinite(value):
            placebo_estimates.append(float(value))

    subset_estimates = []
    for _ in range(n_placebo):
        idx = rng.choice(len(df), size=int(0.7 * len(df)), replace=False)
        Xi = X.iloc[idx].reset_index(drop=True)
        p = estimate_propensity(Xi, real_treated[idx], random_state=seed)
        pairs, _ = match_nearest_neighbour(p, real_treated[idx])
        value = att_from_matches(outcome[idx], pairs)
        if np.isfinite(value):
            subset_estimates.append(float(value))

    placebo_mean = float(np.mean(placebo_estimates)) if placebo_estimates else float("nan")
    placebo_sd = float(np.std(placebo_estimates)) if placebo_estimates else float("nan")
    subset_mean = float(np.mean(subset_estimates)) if subset_estimates else float("nan")
    subset_sd = float(np.std(subset_estimates)) if subset_estimates else float("nan")

    # The placebo must land near zero, and the real estimate must sit far
    # outside the range placebos produce.
    placebo_passes = abs(placebo_mean) < max(0.02, 2 * placebo_sd / np.sqrt(max(len(placebo_estimates), 1)))
    separated = abs(observed_att - placebo_mean) > 3 * (placebo_sd or 1e-9)
    subset_passes = abs(subset_mean - observed_att) < max(0.02, 2 * subset_sd)

    return {
        "placebo_treatment": {
            "n_runs": len(placebo_estimates),
            "mean_estimate": round(placebo_mean, 4),
            "sd": round(placebo_sd, 4),
            "passes": bool(placebo_passes),
            "well_separated_from_observed": bool(separated),
            "reading": (
                "Randomly reassigned treatment produces no effect, and the real "
                "estimate sits far outside the placebo range."
                if placebo_passes and separated else
                "The placebo run found an effect where none exists — the estimate "
                "is an artifact of the pipeline, not of the treatment."
            ),
        },
        "random_subset": {
            "n_runs": len(subset_estimates),
            "mean_estimate": round(subset_mean, 4),
            "sd": round(subset_sd, 4),
            "passes": bool(subset_passes),
            "reading": (
                "The estimate is stable across 70% subsamples."
                if subset_passes else
                "The estimate moves materially across subsamples; a small number "
                "of observations is driving it."
            ),
        },
        "all_passed": bool(placebo_passes and separated and subset_passes),
    }


def plot_causal_diagnostics(result: ObservationalResult, out_path: Path) -> Path:
    """Love plot and the estimator comparison — the two charts that carry the argument."""
    balance = result.balance["per_covariate"]
    order = sorted(balance, key=lambda c: abs(balance[c]["before"]), reverse=True)[:14]
    before = [balance[c]["before"] for c in order][::-1]
    after = [balance[c]["after_matching"] for c in order][::-1]
    labels = [c if len(c) <= 26 else c[:24] + ".." for c in order][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(labels))
    ax1.scatter(before, y, s=45, color="#e76f51", label="Before matching")
    ax1.scatter(after, y, s=45, color="#2a9d8f", label="After matching")
    for i in range(len(labels)):
        ax1.plot([before[i], after[i]], [i, i], color="grey", alpha=0.35, linewidth=1)
    for line in (-SMD_THRESHOLD, SMD_THRESHOLD):
        ax1.axvline(line, color="grey", linestyle="--", linewidth=1)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("Standardised mean difference")
    ax1.set_title("Covariate balance (dashed lines: the |SMD| < 0.1 target)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2, axis="x")

    names = ["Naive\ndifference"] + [
        k.replace("_", "\n") for k in result.estimates
    ]
    values = [result.naive_difference] + [e["estimate"] for e in result.estimates.values()]
    errors = [[0, 0]] + [
        [e["estimate"] - e["ci_95"][0], e["ci_95"][1] - e["estimate"]]
        for e in result.estimates.values()
    ]
    colours = ["#e76f51"] + ["#264653"] * len(result.estimates)

    ax2.bar(names, values, color=colours, alpha=0.85)
    ax2.errorbar(names, values, yerr=np.array(errors).T, fmt="none",
                 ecolor="black", capsize=4, linewidth=1.2)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Effect on churn probability")
    ax2.set_title("Naive comparison vs adjusted estimates")
    ax2.tick_params(axis="x", labelsize=8.5)
    ax2.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _print_observational(result: ObservationalResult) -> None:
    table = Table(title=f"Effect of Tech Support on churn — {result.spec} specification")
    table.add_column("Estimator")
    table.add_column("Effect", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Reading")

    table.add_row(
        "Naive difference", f"{result.naive_difference:+.1%}", "—",
        "[yellow]confounded by who buys it",
    )
    for name, e in result.estimates.items():
        ci = f"[{e['ci_95'][0]:+.1%}, {e['ci_95'][1]:+.1%}]" \
            if np.isfinite(e["ci_95"][0]) else "—"
        excludes_zero = np.isfinite(e["ci_95"][0]) and (e["ci_95"][0] > 0 or e["ci_95"][1] < 0)
        table.add_row(
            name.replace("_", " "), f"{e['estimate']:+.1%}", ci,
            "[green]excludes zero" if excludes_zero else "[yellow]includes zero",
        )
    console.print(table)

    b, d = result.balance, result.diagnostics
    console.print(
        f"  Balance: {b['n_imbalanced_before']} covariates above |SMD| 0.1 before "
        f"matching, {b['n_imbalanced_after_matching']} after "
        f"(max {b['max_abs_smd_before']:.3f} -> {b['max_abs_smd_after_matching']:.3f})"
    )
    console.print(
        f"  Overlap: dropped {result.overlap['n_dropped']} of "
        f"{result.overlap['n_total']} outside common support · "
        f"matched {result.matching['n_matched_pairs']} pairs "
        f"({result.matching['match_rate']:.0%} of treated)"
    )
    console.print(
        f"  Selection bias removed: {d['selection_bias_removed']:+.1%} · "
        f"estimator spread {d['estimator_spread']:.3f} "
        f"({'agree' if d['estimators_agree'] else 'disagree'})"
        + (f" · E-value {d['e_value']:.2f}" if d.get("e_value") else "")
    )


def score_population(df: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    """Predicted risk for the whole customer base.

    The designed studies below simulate campaigns, and a campaign runs against
    the full base rather than the model's held-out test split. Scoring all 7,043
    customers also gives those estimators enough sample to separate a biased
    estimate from an unbiased one — at 1,409 rows the confidence intervals are
    wide enough to swallow the very bias the section exists to show.
    """
    loaded = LoadedModel.load_latest(Path(artifacts_dir))
    scored = df.copy()
    scored["churn_probability"] = loaded.predict_proba(df)
    return scored


def main(
    clean_path: Path = Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
    artifacts_dir: Path = Path.cwd() / "artifacts",
    reports_dir: Path = Path.cwd() / "reports",
    docs_dir: Path = Path.cwd() / "docs" / "images",
    n_boot: int = typer.Option(200, help="Bootstrap resamples for the confidence intervals"),
    seed: int = typer.Option(RANDOM_STATE, help="Random seed"),
):
    console.rule("[bold magenta]Telco Churn: Causal Inference")

    df = prepare_observational(pd.read_parquet(clean_path))
    console.log(
        f"[green]{len(df):,} customers eligible for Tech Support "
        f"({int(df['treated'].sum()):,} subscribed) — customers without internet "
        f"excluded as ineligible"
    )

    console.rule("[bold cyan]1. Propensity scoring — does Tech Support reduce churn?")
    console.log("[blue]Estimating propensity, checking overlap, matching and weighting...")
    primary = run_observational(df, include_mediators=False, n_boot=n_boot, seed=seed)
    _print_observational(primary)

    console.log("[blue]Sensitivity: adding MonthlyCharges, which Tech Support partly causes...")
    mediated = run_observational(df, include_mediators=True, n_boot=max(n_boot // 4, 50), seed=seed)
    console.print(
        f"  Controlling for the mediator moves the ATT from "
        f"{primary.estimates['matching_att']['estimate']:+.1%} to "
        f"{mediated.estimates['matching_att']['estimate']:+.1%} — the gap is the "
        f"portion of the effect that runs through the bill, which a confounder-only "
        f"adjustment should not remove."
    )

    console.rule("[bold cyan]2. Instrumental variables — the offer nobody has to accept")
    scored = score_population(df, artifacts_dir)

    encouraged = simulate_encouragement(scored, seed=seed)
    iv = instrumental_variables_analysis(encouraged)
    true_effect = float(encouraged["true_complier_effect"].iloc[0])

    iv_table = Table(title="Encouragement design (the true complier effect is known here)")
    iv_table.add_column("Estimator")
    iv_table.add_column("Effect on churn", justify="right")
    iv_table.add_column("95% CI", justify="right")
    iv_table.add_column("Reading")
    iv_table.add_row(
        "Naive OLS (redeemers vs not)", f"{iv['naive_ols']['coefficient']:+.1%}",
        f"[{iv['naive_ols']['ci_95'][0]:+.1%}, {iv['naive_ols']['ci_95'][1]:+.1%}]",
        "[yellow]confounded by who redeems",
    )
    iv_table.add_row(
        "Intention to treat (offered)", f"{iv['intention_to_treat']['coefficient']:+.1%}",
        f"[{iv['intention_to_treat']['ci_95'][0]:+.1%}, {iv['intention_to_treat']['ci_95'][1]:+.1%}]",
        "unbiased, but diluted by non-compliance",
    )
    iv_table.add_row(
        "2SLS / LATE (compliers)", f"{iv['two_stage_least_squares']['coefficient']:+.1%}",
        f"[{iv['two_stage_least_squares']['ci_95'][0]:+.1%}, {iv['two_stage_least_squares']['ci_95'][1]:+.1%}]",
        "[green]recovers the truth",
    )
    iv_table.add_row(
        "[bold]True effect on compliers", f"[bold]{true_effect:+.1%}", "—",
        "the estimand, computed from the DGP",
    )
    console.print(iv_table)
    console.print(
        f"  Compliance {iv['compliance_rate']:.1%} · "
        f"first-stage F {iv['first_stage']['first_stage_f_statistic']:.0f} "
        f"({'weak' if iv['first_stage']['weak_instrument'] else 'strong instrument'}) · "
        f"Wald estimate {iv['wald_estimator_late']:+.1%}"
    )

    console.rule("[bold cyan]3. Difference-in-differences — a rollout to the worse region")
    panel = simulate_did_panel(scored, seed=seed)
    did = difference_in_differences(panel)
    did_true = panel.attrs["true_effect"]

    did_table = Table(title="Churn rate by group and period")
    did_table.add_column("Group")
    did_table.add_column("Pre", justify="right")
    did_table.add_column("Post", justify="right")
    did_table.add_column("Change", justify="right")
    m = did["means"]
    did_table.add_row("Treated", f"{m['treated_pre']:.1%}", f"{m['treated_post']:.1%}",
                      f"{did['treated_change']:+.1%}")
    did_table.add_row("Control", f"{m['control_pre']:.1%}", f"{m['control_post']:.1%}",
                      f"{did['control_change']:+.1%}")
    did_table.add_row("[bold]Difference-in-differences", "", "",
                      f"[bold]{did['did_estimate']:+.1%}")
    console.print(did_table)
    console.print(
        f"  Naive post-period comparison: {did['naive_post_period_comparison']:+.1%} "
        f"(biased — the treated region already churned more)\n"
        f"  Naive before/after in treated:  {did['naive_before_after_comparison']:+.1%} "
        f"(biased — a shock hit both groups)\n"
        f"  DiD: {did['did_estimate']:+.1%} against a true effect of {did_true:+.1%} "
        f"(regression p={did['regression']['p_value']:.4f})"
    )
    pt = did["parallel_trends"]
    if pt["testable"]:
        console.print(
            f"  [{'green' if pt['trends_parallel'] else 'yellow'}]Parallel trends: "
            f"{pt['reading']}"
        )

    console.rule("[bold cyan]4. Refutation — trying to break the result")
    refutation = refute(df, primary.estimates["matching_att"]["estimate"], seed=seed)
    for name, r in (("Placebo treatment", refutation["placebo_treatment"]),
                    ("Random 70% subset", refutation["random_subset"])):
        mark = "[green]passed" if r["passes"] else "[yellow]FAILED"
        console.print(f"  {name:<20} {r['mean_estimate']:+.4f} (sd {r['sd']:.4f})  {mark}")
        console.print(f"    {r['reading']}")

    plot_path = plot_causal_diagnostics(primary, Path(docs_dir) / "causal-diagnostics.png")
    console.log(f"[green]Diagnostics plot saved: {plot_path.name}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "observational": {
            "question": "Does subscribing to Tech Support reduce churn?",
            "treatment": f"{TREATMENT_COLUMN} == {TREATMENT_VALUE!r}",
            "eligible_population": "customers with internet service",
            "primary": {
                "spec": primary.spec, "n": primary.n,
                "naive_difference": primary.naive_difference,
                "estimates": primary.estimates, "balance": primary.balance,
                "overlap": primary.overlap, "matching": primary.matching,
                "diagnostics": primary.diagnostics,
            },
            "mediator_sensitivity": {
                "spec": mediated.spec,
                "estimates": mediated.estimates,
                "note": (
                    "MonthlyCharges is caused in part by the treatment, so this "
                    "specification is reported for contrast, not as the estimate."
                ),
            },
            "refutation": refutation,
        },
        "instrumental_variables": {**iv, "true_complier_effect": true_effect},
        "difference_in_differences": {**did, "true_effect": did_true},
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"causal_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.log(f"[green]Causal report saved: {report_path}")

    console.rule("[bold green]Causal analysis completed!")


if __name__ == "__main__":
    typer.run(main)
