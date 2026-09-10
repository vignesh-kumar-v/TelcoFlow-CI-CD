# TelcoFlow — Customer Churn Prediction Platform

[![ML Pipeline](https://github.com/vignesh-kumar-v/TelcoFlow-CI-CD/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/vignesh-kumar-v/TelcoFlow-CI-CD/actions/workflows/ml-pipeline.yml)
![tests](https://img.shields.io/badge/tests-294-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-yellow)

An end-to-end ML platform that predicts telecom customer churn — and then asks
the questions prediction cannot answer. Features are engineered in SQL, models
are compared and tuned with Optuna, runs are tracked and versioned in MLflow,
the winner is served over FastAPI, and the whole thing is orchestrated with
Airflow, containerised, deployable to Kubernetes, and exercised by GitHub
Actions on every push.

On top of the model sit the analytics that decide what to *do* with it: a
customer segmentation, an experiment with its power analysis pre-registered
before the readout, causal estimates of what actually moves churn as opposed to
what merely predicts it, interpretable odds ratios, and a BI extract layer that
feeds a Tableau dashboard.

## Results

Four models compete on test ROC-AUC over a stratified 80/10/10 split
(4,225 train / 1,409 validation / 1,409 test). Any of them can be deployed — the
winner's feature shape is recorded at training time and reproduced at serving.

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** *(deployed)* | **0.8458** | 0.508 | **0.794** | 0.619 |
| Tuned LightGBM (Optuna, 50 trials) | 0.8436 | 0.546 | 0.765 | **0.637** |
| LightGBM | 0.8350 | 0.773 | 0.246 | 0.373 |
| XGBoost | 0.8310 | 0.518 | 0.770 | 0.619 |

- **79.4% recall on churners** — the metric that matters when a missed churner
  is a lost customer and a false positive costs one retention offer.
- **Optuna tuning lifted LightGBM from 0.8350 to 0.8436**, and SQL feature
  engineering lifted the winner from 0.8422 to **0.8458**.
- Untuned LightGBM shows the danger of ranking on AUC alone: 0.773 precision but
  **24.6% recall**, so it misses three churners in four at the 0.5 threshold.

Four findings the model itself cannot produce:

| Question | Answer | Where |
|---|---|---|
| Who are these customers? | 3 segments; **New and Unsettled** holds 3,240 customers at 46.2% churn and **$107k/month of exposed revenue** | [Segmentation](#customer-segmentation) |
| Could the experiment have found what we cared about? | No — it needed **1,292 per arm** and had 292, so it was powered for 10.2pp, not the 5pp that mattered | [Power analysis](#power-analysis) |
| Does Tech Support *cause* lower churn? | Yes, but the raw −26.5% gap is mostly selection. The adjusted effect is **−6.0%** (95% CI −8.0 to −2.6) | [Causal inference](#causal-inference) |
| What does the model believe? | A two-year contract multiplies churn odds by **0.20** (95% CI 0.16–0.27), holding everything else fixed | [Interpretation](#model-interpretation) |

**Top churn drivers** (SHAP on the deployed model): tenure, contract type,
fiber-optic internet, monthly charges, and the engineered `avg_monthly_spend` —
one of the SQL-derived features earning a top-six slot on its own merits.

<p align="center">
  <img src="docs/images/shap-summary.png" alt="SHAP summary plot — feature impact on churn prediction" width="620">
</p>

**Sharpest segment finding** — month-to-month customers in their first six
months on premium plans churn at **77.1%**, nearly **3× the 26.5% base rate**:

| Contract | Tenure | Spend | Customers | Churn rate |
|---|---|---|---|---|
| Month-to-month | 0–6m | premium | 118 | **77.1%** |
| Month-to-month | 0–6m | high | 524 | 72.0% |
| Month-to-month | 1–2y | premium | 177 | 56.5% |

A simulated retention campaign on the 585 highest-risk customers produced a
**+11.6pp retention lift (p = 0.0035)** — see [A/B Test](#ab-test-simulation).

## Problem Statement

Customer churn costs telecom companies billions annually. This project builds a
reproducible pipeline that ingests raw customer records into a database,
engineers features in SQL, trains and compares multiple classifiers, serves
predictions in batch and in real time, watches for data drift, and closes the
loop by measuring whether acting on those predictions actually retains anyone.

Prediction is where most churn projects stop, and it is not where the decisions
are. Knowing who will leave does not say who they are, what would keep them, or
whether the experiment testing that could ever have answered the question — so
segmentation, causal inference and a pre-registered power analysis sit on top of
the model, and a BI layer puts the results in front of the people spending the
retention budget.

## Architecture

```
                     ┌──────────────────────────────────────────┐
  Raw CSV ─────────> │ SQL layer  (SQLite / Postgres / BigQuery) │
                     │  01_clean_customers.sql                  │
                     │  02_feature_engineering.sql              │
                     │  03_churn_analytics.sql                  │
                     └─────────────────┬────────────────────────┘
                                       │ cleaned_data.parquet
                                       ▼
                     ┌──────────────────────────────────────────┐
                     │ Train: LR / LightGBM / XGBoost / Tuned    │───> MLflow
                     │ Optuna tuning · SHAP · best model wins    │     tracking
                     └─────────────────┬────────────────────────┘     + registry
                                       │
                     artifacts/<ts>/   │   data/processed/scoring_batch.parquet
                       model.joblib    │            (held-out split)
                       preprocessor    │
                       train_info.json ▼
              ┌────────────────────────┴───────────────────┐
              ▼                                            ▼
   ┌─────────────────────┐                    ┌───────────────────────┐
   │ Batch scoring       │                    │ FastAPI  /predict     │
   │ + drift detection   │                    │ real-time inference   │
   └──────────┬──────────┘                    └───────────────────────┘
              │ predictions
              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Analytics — the questions a churn score cannot answer         │
   │                                                               │
   │  pre-register ─> A/B test    what could this experiment find? │
   │  segmentation                who are these customers?         │
   │  interpretation              what does the model believe?     │
   │  causal inference            what actually *causes* churn?    │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ BI extracts -> Tableau  │
                     └─────────────────────────┘

  Orchestration: Airflow DAG with quality gates   Deployment: Docker / Kubernetes
```

## Quick Start

```bash
git clone https://github.com/vignesh-kumar-v/TelcoFlow-CI-CD.git
cd TelcoFlow-CI-CD

python -m venv venv && source venv/bin/activate
make install

make pipeline        # SQL -> train -> score -> experiment -> analysis -> BI
make test
```

`make help` lists every target.

## Pipeline Stages

| Stage | Command | What it does |
|-------|---------|--------------|
| **SQL features** | `make sql-features` | Loads raw records into the DB, cleans and derives features in versioned SQL, exports to parquet |
| **Validate** (alt) | `make validate` | Pure-pandas cleaning path, no database required |
| **Train** | `make train` | Compares 4 models, tunes with Optuna, runs SHAP, logs to MLflow, registers the winner |
| **Score** | `make score` | Scores the held-out batch, writes predictions and a drift report |
| **Pre-register** | `make prereg` | Commits the experiment design — MDE, alpha, power, required n — before any outcome exists |
| **A/B test** | `make ab-test` | Simulates a retention campaign and judges it against that committed plan |
| **Segment** | `make segment` | Clusters the customer base and profiles each segment's revenue at risk |
| **Interpret** | `make interpret` | Odds ratios with confidence intervals and VIF for the linear baseline |
| **Causal** | `make causal` | Propensity scoring, instrumental variables, DiD and refutation tests |
| **BI export** | `make bi-export` | Publishes the Tableau extracts and workbook scaffold |
| **Serve** | `make api` | FastAPI real-time predictions on port 8000 |

`make analysis` runs the three analysis stages; `make pipeline` runs everything.

## SQL Feature Engineering

Cleaning and feature derivation live in `sql/`, not in pandas. The backend is
chosen by one environment variable — the same SQL runs on both:

```bash
make sql-features                                    # SQLite (default)

docker compose up -d                                 # or Postgres
export DB_URL=postgresql+psycopg2://telco:telco@localhost:5432/telco
make sql-features
```

Derived features:

| Feature | Definition | Why |
|---------|-----------|-----|
| `num_addon_services` | Count of the 6 optional services taken | Churn concentrates among customers paying a lot for very little |
| `avg_monthly_spend` | `TotalCharges / tenure` | Lifetime billing rate, which differs from the current price after any repricing |
| `charges_ratio` | `MonthlyCharges / avg_monthly_spend` | Above 1 means a recent price increase — a classic churn trigger |
| `tenure_bucket` | 0-6m / 6-12m / 1-2y / 2-4y / 4y+ | Churn risk is heavily front-loaded |
| `spend_bucket` | low / medium / high / premium | Price tier |

`03_churn_analytics.sql` aggregates churn by segment. The result is a real
finding, not decoration — the worst segment churns at nearly 3× the base rate:

| Contract | Tenure | Spend | Customers | Churn rate |
|---|---|---|---|---|
| Month-to-month | 0-6m | premium | 118 | **77.1%** |
| Month-to-month | 0-6m | high | 524 | 72.0% |
| Month-to-month | 1-2y | premium | 177 | 56.5% |

*(base rate across all customers: 26.5%)*

**Training/serving parity.** Training features are built in the warehouse, but
an API request never passes through it, so the same derivations exist in Python
(`features.compute_engineered_features`). `tests/test_sql.py` asserts the two
implementations agree row-for-row across all 7,043 customers — divergence there
would be training/serving skew, where the model scores live traffic on
differently-computed inputs.

## Model Selection

Scores are in [Results](#results). The point here is that **any of the four can
be deployed**, which requires reproducing three different feature shapes at
serving time:

| Model | Feature shape it consumes |
|-------|--------------------------|
| Logistic Regression | Dense one-hot matrix (standardised) |
| LightGBM | Native pandas `category` columns |
| XGBoost | Integer category codes |
| Tuned LightGBM | Native categoricals, Optuna-optimised |

Training records which shape the winner needs in `train_info.json`, and both
serving paths transform features accordingly. Category levels are pinned at fit
time, so a scoring batch that happens to omit a category cannot silently shift
every remaining code.

## Experiment Tracking

Every run logs parameters, per-model metrics, each Optuna trial (as a nested
run) and artifacts to MLflow; the winner is registered as a new version of
`telco-churn-classifier`.

```bash
make mlflow-ui                    # http://localhost:5000
make mlflow-ui MLFLOW_PORT=5001   # macOS binds 5000 to AirPlay Receiver
```

Each Optuna trial is a nested run, so the whole search is inspectable rather
than collapsed into a single best-params line — 50 trials spanning validation
AUC 0.8126 to 0.8397 below:

![MLflow runs with nested Optuna trials](docs/images/mlflow-runs.png)

The winner is registered as a new version, each annotated with the algorithm
and score at registration time:

![MLflow model registry versions](docs/images/mlflow-registry.png)

Tracking is best-effort: if MLflow is unavailable the run logs a warning and
training completes anyway. A metrics sink should not be able to take down the
pipeline.

## Drift Detection

Batch scoring runs against the **held-out** split, not the training data, so
drift numbers are meaningful. Numeric features are compared on mean/std shift
(>10%), categoricals on Total Variation Distance (>0.1).

```bash
make score          # clean holdout    -> 0/7 numeric, 0/17 categorical drifted
make score-drift    # perturbed batch  -> 3/7 numeric, 1/17 categorical drifted
```

## Power Analysis

Reporting a confidence interval after the fact answers "was this significant?".
Power analysis answers the question that should have been asked first: **how many
customers does this experiment need before it can detect the effect we care
about?**

Order matters, and it is enforced rather than assumed. Computing power from the
*observed* effect afterwards is post-hoc power — a deterministic function of the
p-value, so it can never say anything the p-value did not. The design is
therefore committed to disk before any outcome is drawn:

```bash
make prereg       # writes reports/preregistration.json — MDE, alpha, required n
make ab-test      # reads it back and reports the result against that plan
```

The Airflow DAG orders these the same way, and a test asserts it: `prereg` after
`ab_test` would be a design written with knowledge of the result.

| Parameter | Value |
|---|---|
| Primary metric | Retention rate |
| Baseline retention (targeted population) | 26.5% |
| Minimum detectable effect | **5.0pp absolute** |
| Alpha / power target | 0.05 / 80% |
| **Required n per arm** | **1,292** |
| Available n per arm | 292 |
| Power at the available n | **26%** |
| Smallest effect this sample *can* detect | 10.2pp |

The MDE is set from the economics, not from the data: a $50 offer against roughly
$840 of annual revenue breaks even near a 6pp lift, so 5pp is the point below
which the campaign would not run regardless of significance.

![Power curve](docs/images/power-curve.png)

**The closed form is checked against the estimator it describes.** Analytic power
at n=292 is 0.264; simulating 2,000 experiments through the same two-proportion
z-test the harness uses gives 0.278. A power formula that disagrees with its own
test is not a calculation, it is a guess — so `tests/test_power.py` asserts the
two agree across several designs.

Revenue, the secondary metric, needs 446 per arm to detect a $50 shift. That it
needs *fewer* than retention is not a mistake — a $50 move against that spread is
a larger standardised effect than 5pp is against a 26.5% rate.

## A/B Test Simulation

Scoring customers is half the problem; the business question is whether acting
on the scores retains anyone. High-risk customers are assigned to a retention
offer or no contact, **stratified by predicted-risk decile** so both arms carry
the same baseline risk, then evaluated:

- Two-proportion z-test and chi-square on retention
- Welch's t-test on net revenue per customer
- 95% confidence intervals, Cohen's h/d, and the minimum detectable effect
- **Holm-Bonferroni** across the metric family, so testing two metrics at
  alpha = 0.05 each does not give the readout a ~10% false-positive rate
- An **A/A negative control** that must come out non-significant

```
A/B test — retention offer (20% assumed churn reduction)
Metric               Treatment  Control  Difference  p-value  Verdict
Retention rate           41.4%    29.8%      +11.6%   0.0035  significant
Net revenue/customer      $325     $263        +$62   0.1036  not significant

A/A negative control — no treatment effect injected
Retention rate           26.2%    23.4%       +2.8%   0.4301  not significant

Pre-registered verdict: SIGNIFICANT
  Detected a +11.6% change in retention rate (p=0.0035). The study was
  under-powered for the pre-registered MDE, so this estimate is likely
  inflated — statistically significant effects found in small samples
  exaggerate the true effect size.
  Planned MDE 5.0% · power 26% · n 292/1,292 per arm

Holm-Bonferroni across the metric family:
  retention rate           p=0.0035  threshold=0.0250  holds
  net revenue/customer     p=0.1036  threshold=0.0500  does not hold
```

The pre-registration is what makes that last block possible. The observed 11.6pp
lift clears significance and survives multiplicity correction — but it sits just
above the 10.2pp this sample could reliably detect, which is the signature of the
winner's curse: an effect only detectable because it came in large. The honest
readout says so, rather than quoting +11.6pp as a planning number.

It also separates two things a p-value alone conflates. A non-significant result
in a study powered to find the effect is evidence against it; the same p-value
from an under-powered study is **inconclusive**. `interpret_result` returns
`significant` / `null` / `inconclusive` accordingly, and each case is tested.

Randomisation is checked too — mean predicted risk was 0.7351 (treatment) vs
0.7347 (control), p = 0.968.

**This is a simulation, not a live experiment.** The outcome model assumes the
treatment reduces each customer's predicted churn probability by a fixed
relative amount. It demonstrates the analysis machinery — stratification,
pre-registration, significance testing, power, multiplicity correction, negative
controls — not a measured business result.

## Customer Segmentation

The supervised model answers "who is about to churn?". It does not answer "who
*are* these customers?" — and a retention budget is allocated across segments,
not across 7,043 individual probabilities.

```bash
make segment
```

K-means over 15 behavioural and value features, standardised. Three things keep
this from being decoration:

- **Churn is excluded from the clustering matrix.** Segments are built from
  behaviour alone and churn rate is a *post-hoc* profile. Feeding the target in
  would produce clusters that predict churn by construction and say nothing about
  who the customers are. A test asserts the target never reaches the matrix.
- **k is chosen, not assumed.** Silhouette, Davies-Bouldin and Calinski-Harabasz
  are swept over k = 2..10, with a minimum-segment-size constraint so the winner
  is not a k that isolates a handful of outliers. Inertia is reported for the
  elbow but cannot pick k on its own, since it falls monotonically.
- **The structure is validated.** Ward hierarchical clustering is fitted at the
  same k and compared by Adjusted Rand Index, and the labels are bootstrapped.

k = 3 won on silhouette (0.229). **Ward agreement ARI 0.725, bootstrap ARI
0.991** — two algorithms optimising different objectives land on substantially
the same partition, and the labels survive resampling.

| Segment | Customers | Churn | Tenure | Monthly | Add-ons | Revenue at risk |
|---|---:|---:|---:|---:|---:|---:|
| **New and Unsettled** | 3,240 (46%) | **46.2%** | 17m | $72 | 1.7 | **$107,169/mo** |
| Premium Loyalists | 2,114 (30%) | 11.4% | 58m | $88 | 4.1 | $21,187/mo |
| Locked-In Core | 1,689 (24%) | 7.8% | 30m | $23 | 0.1 | $2,972/mo |

![Choosing k: elbow and silhouette](docs/images/segmentation-k-selection.png)

![Segment profile and where retention spend should go](docs/images/segmentation-profile.png)

The ranking column is `monthly_revenue_at_risk` — segment size × churn rate ×
value. Ranking by churn rate alone would point at the same segment for the wrong
reason; ranking by value alone would point at Premium Loyalists, who are not
leaving. Segment names are derived from the profile rather than hardcoded per
cluster id, because k-means numbers its clusters arbitrarily and a fixed mapping
would silently mislabel everything on the next refit.

Silhouette of 0.23 is modest, which is normal for mixed customer data — the
segments are real and reproducible, not sharply separated islands. The PCA panel
above shows exactly that.

## Model Interpretation

Logistic regression already competes in training and, on this dataset, wins. The
reason to keep a linear model around when gradient boosting is available is not
its AUC — it is that every coefficient is a claim someone can argue with. SHAP
explains what the model *did*; an odds ratio explains what it *believes*.

```bash
make interpret
```

| Feature | Odds ratio | 95% CI | p | VIF |
|---|---:|---:|---:|---:|
| **Contract: Two year** | **0.204** | 0.16–0.27 | <0.0001 | 2.7 |
| Contract: One year | 0.500 | 0.42–0.59 | <0.0001 | 1.6 |
| Tenure bucket 1–2y | 0.445 | 0.33–0.59 | <0.0001 | 2.6 |
| Paperless billing | 1.380 | 1.21–1.58 | <0.0001 | 1.2 |
| Payment: Electronic check | 1.341 | 1.13–1.59 | 0.0007 | 2.0 |

A two-year contract multiplies the odds of churn by 0.20 — a five-fold
reduction — holding every other feature fixed. Two details make that table
honest rather than decorative:

- **The deployed model is L2-penalised, and penalised coefficients have no valid
  Wald standard errors.** The penalty shrinks estimates by an amount the variance
  formula does not know about. The inference model is refit unpenalised, and its
  agreement with what is serving is *measured*: Spearman **0.9976** on predicted
  probabilities, AUC 0.8506 vs 0.8513. If those diverged, the table would be
  describing a model that is not in production, and a test enforces the bound.
- **The serving design matrix one-hot encodes every level with no reference
  category** — fine under L2, but it makes the unpenalised information matrix
  singular and the standard errors meaningless. The inference matrix uses
  reference coding, so each coefficient reads as "relative to the omitted level".

**VIF is reported alongside, and it changes the reading.** `InternetService: Fiber
optic` shows the largest raw odds ratio in the table (8.16) — and a VIF of 149,
meaning it is splitting one effect across collinear columns. Its interval spans
1.9 to 35.2, which is not a finding. The headline deliberately excludes any term
above VIF 10, so the module never quotes a number its own diagnostic just
invalidated.

## Causal Inference

Everything above is predictive. A model that ranks churn risk well is answering
"who is likely to leave?" — enough to decide whom to call, not enough to decide
*what to offer them*, because the features that predict churn are not the
features that change it.

```bash
make causal
```

### 1. Does Tech Support reduce churn? (propensity scoring)

Customers with Tech Support churn 26.5 percentage points less. Almost none of
that is the product.

| Estimator | Effect on churn | 95% CI |
|---|---:|---:|
| Naive difference | −26.5% | *confounded* |
| Matching ATT | **−6.0%** | −8.0 to −2.6 |
| IPW ATE | −8.2% | −10.8 to −5.6 |
| AIPW (doubly robust) | −6.7% | −9.3 to −4.2 |

**20.5 of the 26.5 points were selection.** The three estimators span 2.2 points,
and AIPW is consistent if *either* the treatment model or the outcome model is
right — so the answer does not rest on one modelling choice.

![Covariate balance and estimator comparison](docs/images/causal-diagnostics.png)

The design decisions matter more than the estimator here:

- **The eligible population is restricted to customers with internet.** Customers
  without it cannot buy Tech Support; their `"No internet service"` value is
  ineligibility, not a declined add-on. Leaving them in the control arm compares
  subscribers against people who were never offered the product.
- **`num_addon_services` counts Tech Support itself**, so it is mechanically
  determined by the treatment. The confounder set uses a recount that excludes it.
- **`MonthlyCharges` is deliberately not a confounder.** Tech Support is a paid
  add-on, so it *raises* the bill: conditioning on the bill blocks part of the
  effect being estimated. It is a mediator, and controlling for it moves the ATT
  from −6.0% to −11.4% — reported as a labelled sensitivity spec so the
  difference is visible rather than assumed away.
- **Overlap is enforced**: 39 units outside common support are dropped rather
  than extrapolated over. Matching is 1:1 on the propensity logit, within a
  0.2 SD caliper, without replacement.
- **Balance is checked, not hoped for**: 11 covariates started above |SMD| 0.1
  (max 0.85); one remains after matching (max 0.16).

**Refutation.** An estimate nobody attacked is not evidence. Randomly permuting
the treatment gives −0.006 (sd 0.009) — the pipeline finds nothing where nothing
exists, and the real estimate sits far outside that range. Re-estimating on
random 70% subsamples gives −0.051, so no small group of rows is driving it. The
**E-value is 1.94**: an unmeasured confounder would need to nearly double both
the chance of treatment and the chance of churn, beyond every covariate already
adjusted for, to explain the result away.

### 2. Instrumental variables — the offer nobody has to accept

Real campaigns do not assign treatment, they assign *eligibility*. The offer is
mailed at random (Z), but redemption (D) is the customer's choice, and engaged
customers redeem more — so comparing redeemers to non-redeemers overstates the
offer by however much engagement predicts retention.

| Estimator | Effect on churn | 95% CI | |
|---|---:|---:|---|
| Naive OLS (redeemers vs not) | −20.8% | −23.6 to −18.1 | confidently wrong |
| Intention to treat (offered) | −6.8% | −9.4 to −4.2 | unbiased, diluted by non-compliance |
| **2SLS / LATE (compliers)** | **−16.3%** | −22.5 to −10.2 | recovers the truth |
| True effect on compliers | −12.5% | — | the estimand, from the DGP |

Compliance 41.8%, first-stage **F = 1,480** (the weak-instrument threshold is 10).
The naive interval *excludes* the true effect; the 2SLS interval covers it. Across
8 seeds, mean bias is **+0.004 for 2SLS against −0.065 for naive OLS** — which is
the right way to demonstrate an estimator property, since one draw proves nothing.

The benchmark is computed from the data-generating process rather than assumed
equal to the injected parameter, because two things separate them: churn risk is
bounded, so the offer cannot take a customer at 8% risk down by 15 points; and
LATE is the effect on *compliers*, not on everyone. Benchmarking against the
injected parameter would make a correct estimator look biased.

### 3. Difference-in-differences — a rollout to the worse region

The campaign goes to the higher-churn region first, which is what actually
happens — you pilot where the problem is. That makes the post-period comparison
biased by the pre-existing gap, and a before/after comparison biased by a shock
that hit both groups.

| Group | Pre | Post | Change |
|---|---:|---:|---:|
| Treated | 26.5% | 23.7% | −2.9% |
| Control | 16.9% | 21.5% | +4.6% |
| **Difference-in-differences** | | | **−7.4%** |

Against a true effect of −7.98%. The two comparisons DiD replaces get it wrong in
both directions: the naive post-period comparison says **+2.2%** (the campaign
made things worse) and the naive before/after says −2.9%.

**Parallel trends are tested, not asserted.** Extra pre-treatment periods are
generated so the treated–control gap can be checked for drift before treatment;
the placebo interaction comes out at p = 0.83. This cannot *prove* parallel
trends — only that they had not visibly broken beforehand — which is the
strongest evidence the design admits. A test constructs a panel with deliberately
diverging pre-trends and asserts the check catches it.

### Why these two are constructed data

The dataset is a static snapshot with no time dimension and no experiment, so
there is no DiD or IV to be had in it. Building the data-generating process means
the true effect is *known*, which turns each estimator into a testable claim
rather than a number nobody can check — 2SLS must recover the complier effect
where naive OLS does not, DiD must recover the rollout effect where the
post-period comparison does not. `tests/test_causal.py` asserts exactly that,
including the null cases where each estimator must find nothing.

The propensity-scoring analysis in §1 is the one that runs on the real 7,043
customers.

## Dashboard (Tableau)

```bash
make bi-export
```

Everything the pipeline produces lands in JSON reports and parquet — right for
the pipeline, wrong for anyone who wants to look at it. This publishes twelve
flat, typed, BI-ready extracts to `outputs/bi/`, in two shapes:

- a **star schema** (`dim_*` / `fct_*`) for anyone joining tables themselves, and
- **`telcoflow_master.csv`**, one denormalised row per customer with the score,
  risk band, segment, revenue at risk and top three SHAP drivers already joined,
  so a dashboard can be built without configuring a single relationship.

| Extract | Rows | What it holds |
|---|---:|---|
| `telcoflow_master.csv` | 1,409 | One row per scored customer — start here |
| `fct_shap_customer.csv` | 77,495 | Long format: one row per customer per feature |
| `fct_causal_estimates.csv` | 12 | Every causal estimate on one scale, naive vs adjusted |
| `fct_ab_test.csv` | 4 | A/B and A/A results with intervals and effect sizes |
| `dim_segment.csv` | 3 | Segment profiles and revenue at risk |
| `fct_logistic_coefficients.csv` | 40 | Odds ratios, intervals, p-values and VIF |

The per-customer SHAP export is the reason this beats screenshotting a summary
plot. Long format lets a dashboard answer "why is *this* customer high risk?"
instead of only "what matters on average?" — which is the question a retention
agent actually has, and the one a static SHAP plot cannot answer.

`dashboards/telcoflow.twb` is a generated Tableau scaffold with the extract
connected, all 47 columns typed, and identifier-like fields (`risk_decile`,
`segment`) marked as dimensions so Tableau does not sum them. It is generated
XML, not a workbook opened and verified by hand — if Tableau objects to it,
`File → Open → telcoflow_master.csv` loses nothing but the typing. It is
gitignored because it stores an absolute path to `outputs/bi`; `make bi-export`
regenerates it per machine.

[`dashboards/README.md`](dashboards/README.md) has the full build: calculated
fields with Tableau syntax, five sheet specifications, layout, and publish steps.
`outputs/bi/DATA_DICTIONARY.md` documents every table and column.

## API

```bash
make api          # or: make docker-run
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness — 200 whenever the process is up |
| `GET` | `/ready` | Readiness — 503 until a model is loaded |
| `POST` | `/predict` | Churn probability and risk band for one customer |
| `GET` | `/model/info` | Deployed version, type, metrics and top SHAP features |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customerID":"7590-VHVEG","gender":"Female","SeniorCitizen":0,"Partner":"Yes",
       "Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service",
       "InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
       "Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":29.85,"TotalCharges":29.85}'
```

```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.8414804970801406,
  "risk_category": "High",
  "model_version": "20260830-124941",
  "timestamp": "2026-08-30T12:51:43.594095"
}
```

Pydantic validates every field before the model is touched, so malformed
requests get a 422 rather than an opaque 500. Interactive docs are generated at
`/docs`:

![FastAPI interactive docs returning a live prediction](docs/images/api-docs.png)

Liveness and readiness are separate on purpose: probes check status codes, not
bodies, so a 200 response saying "no model" would still get the pod added to the
load balancer. `/ready` returns 503 instead, and retries loading — so a pod that
starts before training finishes becomes ready on its own, without a restart.

## Orchestration (Airflow)

```
ingest_sql → train → evaluate_gate ─┬─> promote_model → batch_score → drift_gate ─┐
                                    └─> reject_model  (fails the run)              │
                                                                                   │
        ┌──────────────────────────────────────────────────────────────────────────┘
        ├─> prereg → ab_test ──┐
        ├─> segment ───────────┤
        ├─> interpret ─────────┼─> bi_export
        └─> causal ────────────┘
```

`evaluate_gate` branches on test ROC-AUC against a threshold, so a regression
stops at `reject_model` instead of quietly shipping; `drift_gate` fails the run
when the scored batch has drifted. Gate logic lives in `src/telco_churn/gates.py`
— stdlib only, unit-tested without an Airflow install.

`prereg` sits deliberately upstream of `ab_test`: the design is committed before
any outcome is drawn, so the readout is judged against a plan rather than against
hindsight. Ordering them the other way would make the pre-registration
meaningless, and `test_preregistration_runs_before_the_experiment` asserts it.
The three analyses fan out in parallel — they read the same scored batch and do
not depend on each other — and `bi_export` fans back in, because the dashboard
extracts include results from all four.

Airflow pins its own dependency versions, so it gets a separate environment:

```bash
make airflow-install    # isolated venv, official Airflow constraints
make airflow-run        # http://localhost:8080
```

Verified by running the DAG end to end on Airflow 2.10.3: all 8 tasks succeeded,
the gate branched to `promote_model` and correctly skipped `reject_model`, and
raising the threshold above the achievable AUC flipped it to a failed run with
`Model rejected: ROC-AUC 0.8458 is below the 0.99 threshold`.

The scheduler environment has **no ML dependencies at all** — verified by
importing `gates.py` there with pandas, scikit-learn, LightGBM, XGBoost, SHAP
and MLflow all absent. The gates are stdlib-only; everything heavier is shelled
out to the project venv.

Both verdict tasks run with `retries=0`. A quality gate reaching a verdict on a
finished model is deterministic, so retrying reaches the same conclusion and
only delays the alert — the transient-failure tasks around them keep `retries=2`.

## Docker

```bash
make docker-build     # 1.69GB
make docker-run       # API on :8000
```

The image derives a runtime subset from `requirements.txt` rather than keeping a
second dependency list: `xgboost` becomes `xgboost-cpu` (the Linux xgboost wheel
hard-depends on 457MB of CUDA this image never runs) and notebook/test packages
are dropped. Verified to produce byte-identical predictions to a host run.

## Kubernetes

Enable Kubernetes in Docker Desktop (Settings → Kubernetes), or use minikube —
see [`k8s/README.md`](k8s/README.md) for both paths.

```bash
make docker-build            # tags both the pinned version and :latest
make k8s-deploy && make k8s-status
```

Build with `make docker-build` rather than a bare `docker build`: the manifests
pin an immutable tag (`VERSION` in the Makefile), and an image tagged only
`:latest` will not satisfy them.

Deployment (2 replicas, startup/readiness/liveness probes, zero-downtime
rollout) + Service, a training Job, and a nightly scoring CronJob, sharing a
persistent artifacts volume.

Verified end to end on a live cluster: the training Job writes the model to the
shared PVC, the API pods sit at `0/1` and out of the Service until it appears,
then reach `1/1` with zero restarts, and the CronJob scores the held-out batch
using the model the Job produced.

See [`k8s/README.md`](k8s/README.md) for probe design, the `:latest` image trap,
and the multi-node storage caveat.

## BigQuery

```bash
gcloud auth application-default login    # one-time

make bq-load          # cleaned data -> BigQuery
make bq-features      # query engineered features back out
make bq-analytics     # churn by segment, computed in BigQuery
make bq-predictions   # publish scored customers
make bq-all           # load, analyse and publish
```

`GCP_PROJECT` defaults to the active gcloud project, and nothing contacts
Google without it — the rest of the pipeline runs unchanged with no cloud
credentials.

The same feature definitions run in three places, and the results are asserted
equal, not assumed:

| Backend | Where features are built |
|---|---|
| SQLite / Postgres | `sql/02_feature_engineering.sql` |
| BigQuery | `FEATURE_QUERY` in `bigquery_loader.py` |
| Python (serving) | `features.compute_engineered_features` |

`test_bigquery_features_match_local_computation` checks the warehouse output
against the Python serving path over all 7,043 rows; `TestTrainServeParity`
does the same for the local SQL. Verified exact — zero difference across every
engineered column.

Reads go through pandas-gbq and the BigQuery Storage API, streaming results
over gRPC rather than paging REST.

## Tests

```bash
make test               # everything
make unit-test          # no artifacts needed — runs on a cold checkout
make integration-test   # asserts against a completed pipeline run
```

**294 tests** — 34 unit, 31 causal, 30 Kubernetes manifest, 29 power, 25 SQL,
24 BI export, 23 orchestration, 22 A/B, 18 segmentation, 17 BigQuery, 16
integration, 13 interpretation, 12 API. Integration tests **skip** rather than
fail when no pipeline has run, so a fresh clone is green: **264 pass, 30 skip,
0 fail** with no artifacts on disk. Notable coverage:

- Category codes stay stable when a scoring batch omits a category
- The deployed model is the comparison winner (not a filtered subset's)
- All three feature shapes are servable from the real fitted artifact
- SQL, BigQuery and Python feature implementations agree row-for-row
- A/A false-positive rate stays near α across 120 simulations
- Analytic power matches Monte Carlo across several designs, and the two
  independent z-test implementations agree
- 2SLS beats naive OLS on mean bias across seeds; DiD recovers a known effect
  where both naive comparisons fail; each estimator finds nothing when there is
  nothing to find
- Broken pre-trends are detected, so DiD cannot credit a pre-existing divergence
- The clustering matrix never contains the target, and pure noise is *not*
  reported as stable structure
- The unpenalised inference matrix is full rank while the serving matrix is not
- The Airflow DAG imports no ML dependencies, and `prereg` precedes `ab_test`

## CI/CD

Four parallel jobs on every push and PR: unit tests (fast feedback) → full
pipeline + integration tests, alongside manifest/DAG validation and a Docker
build. Pip downloads and Docker layers are cached; trained models, reports and
the BI extracts are uploaded as workflow artifacts, so the Tableau extracts can
be downloaded straight from a CI run without executing anything locally.

The run summary carries the model comparison table plus the headline from each
analysis — the experiment's pre-registered verdict, the most exposed segment,
and the causal estimate against the naive comparison it corrects:

```
SIGNIFICANT — effect +11.6%, p=0.0035, power 26% against a pre-registered
5.0% MDE (292/1,292 per arm).

3 segments; most exposed is New and Unsettled — 3,240 customers at 46.2%
churn, $107,169/month at risk.

Tech Support ATT -6.0% (95% CI -8.0% to -2.6%) against a naive gap of
-26.5% — -20.5% of it was selection.
```

## Project Structure

```
├── .github/workflows/ml-pipeline.yml  # 4 parallel CI jobs
├── airflow/dags/telco_churn_dag.py    # orchestration with quality gates
├── dashboards/README.md               # Tableau build guide
├── k8s/                               # Deployment, Service, Job, CronJob
├── sql/                               # versioned cleaning + feature SQL
├── notebooks/eda.ipynb                # exploratory analysis
├── src/telco_churn/
│   ├── features.py                    # feature defs + fitted transform
│   ├── inference.py                   # shared serving path
│   ├── gates.py                       # promotion / drift predicates (stdlib only)
│   ├── db.py, sql_features.py         # SQL layer
│   ├── train.py, batch_score.py       # training and batch scoring
│   ├── tracking.py                    # MLflow
│   ├── power.py                       # pre-registration and power analysis
│   ├── ab_test.py                     # experiment simulation and readout
│   ├── segmentation.py                # k-means segments + stability validation
│   ├── interpret.py                   # odds ratios, Wald intervals, VIF
│   ├── causal.py                      # propensity scoring, IV/2SLS, DiD, refutation
│   ├── bi_export.py                   # Tableau / Power BI extracts
│   ├── bigquery_loader.py             # warehouse path
│   └── api.py                         # FastAPI
├── tests/                             # 294 tests
├── Dockerfile / .dockerignore         # 1.69GB runtime image
├── docker-compose.yml                 # Postgres + MLflow server
└── requirements.txt / requirements-airflow.txt
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 customers, 20 features, 26.5% churn rate. The 11 records with a blank
`TotalCharges` are all `tenure = 0` (new customers, never billed) and are
median-imputed.

## Limitations and next steps

Known constraints, stated plainly:

- **The dataset is a static snapshot.** There is no time dimension, so the
  train/test split cannot be temporal. A production churn model should be
  validated on a forward time window, since customer mix drifts.
- **The A/B test is simulated.** Outcomes are drawn from the model's own
  predicted probabilities, so it validates the analysis, not the intervention.
- **The IV and DiD analyses run on constructed data**, for the same reason: the
  snapshot contains no experiment and no time dimension. They demonstrate that
  the estimators recover a known effect where the naive comparisons do not. Only
  the propensity-scoring analysis estimates a causal effect from the real 7,043
  customers — and it rests on conditional ignorability, which is an assumption
  about unobserved confounding that no test can verify. The E-value quantifies
  how strong such a confounder would need to be (1.94); it does not rule one out.
- **The segments are reproducible but not sharply separated.** Silhouette of 0.23
  is normal for mixed customer data and means the boundaries are soft. Treat the
  profiles as a way to allocate budget, not as natural kinds.
- **The Tableau workbook is a generated scaffold**, not a designed dashboard. The
  extracts and the build guide are the deliverable; the sheets are a manual step.
- **Feature logic exists in three implementations** (local SQL, BigQuery SQL,
  Python). Parity tests catch divergence, but a single source — dbt models, or
  a feature store — would remove the duplication entirely.
- **The artifact store is a shared filesystem.** That works on a single node;
  a multi-node cluster needs ReadWriteMany or, better, serving pulling from the
  MLflow registry rather than a mounted volume.
- **Decision threshold is fixed at 0.5.** With a 2.7:1 imbalance, the operating
  point should be chosen from the cost of a missed churner versus a wasted
  retention offer, not left at the default.
- **No model monitoring in production.** Drift is computed at scoring time, but
  there is no alerting or automated retraining trigger wired to it.
- **The experiment is under-powered by design**, because the targeted population
  is what it is. Detecting the 5pp effect that matters needs 1,292 customers per
  arm against the 292 available, so the campaign would have to run across several
  scoring cycles before the readout could settle the question.

## Tech Stack

**ML:** LightGBM · XGBoost · scikit-learn · Optuna · SHAP
**Serving:** FastAPI · Uvicorn · Pydantic
**Data:** pandas · NumPy · PyArrow · SQLAlchemy · SQLite/Postgres · BigQuery
**MLOps:** MLflow · Airflow · Docker · Kubernetes · GitHub Actions
**Statistics:** SciPy · power analysis and pre-registration · Holm-Bonferroni
**Causal inference:** propensity score matching · IPW · AIPW · 2SLS · DiD ·
placebo and E-value sensitivity
**Analytics:** k-means and Ward clustering · PCA · Tableau

## License

[MIT](LICENSE)
