# TelcoFlow churn dashboard (Tableau)

The pipeline publishes BI-ready extracts to `outputs/bi/`. This is the build for
a four-sheet dashboard over them.

```bash
make bi-export      # regenerates every extract from the latest pipeline run
```

## Connect

Two ways in, and the second is the reliable one:

1. **`dashboards/telcoflow.twb`** — a generated scaffold with the master extract
   already connected, all 47 columns typed, and identifier-like fields marked as
   dimensions rather than measures. It stores an absolute path to
   `outputs/bi/telcoflow_master.csv`, so regenerate it (`make bi-export`) if you
   move the repo. It is generated, not opened and verified by a human — if
   Tableau objects to it, use the next option and lose nothing but the typing.
2. **File → Open → `outputs/bi/telcoflow_master.csv`.** One flat table, no
   relationships to configure. Everything below can be built from it alone.

Add the other extracts as separate data sources when you want the results
sheets: `fct_causal_estimates.csv`, `fct_ab_test.csv`, `fct_shap_global.csv`.

## What is in the master extract

One row per scored customer, at `outputs/bi/telcoflow_master.csv`:

| Field group | Columns |
|---|---|
| Identity and attributes | `customerID`, `Contract`, `tenure`, `InternetService`, `PaymentMethod`, … |
| Model output | `churn_probability`, `prediction`, `risk_category`, `risk_decile`, `model_version` |
| Outcome | `actual_churn`, `prediction_outcome` (true/false positive/negative) |
| Value | `MonthlyCharges`, `annual_revenue`, `revenue_at_risk` |
| Segment | `segment`, `segment_name` |
| Explanation | `driver_1_feature` / `_shap` / `_direction`, and the same for drivers 2 and 3 |

`revenue_at_risk` is `annual_revenue × churn_probability` — expected revenue
lost, which is the number that should drive where retention spend goes. Ranking
by `churn_probability` alone sends the budget to cheap customers who were leaving
anyway.

`outputs/bi/DATA_DICTIONARY.md` lists every table and column.

## Calculated fields

Create these first; three of the four sheets use them.

```
// Risk band, ordered — the raw risk_category sorts alphabetically
[Risk Band]
IF [Churn Probability] >= 0.6 THEN "1 - High"
ELSEIF [Churn Probability] >= 0.4 THEN "2 - Medium"
ELSE "3 - Low"
END

// Model quality at the current threshold
[Correct Prediction]
IIF([Prediction] = [Actual Churn], 1, 0)

// Campaign economics: does treating this customer pay?
[Expected Offer Value]
[Revenue At Risk] * 0.20 - 50      // 20% assumed churn reduction, $50 offer

// Cumulative capture for the gain curve
[Running Churners]
RUNNING_SUM(SUM([Actual Churn]))
```

## Sheets

**1 — Risk overview.** Bar chart, `Risk Band` on columns, `SUM(Revenue At Risk)`
on rows, `COUNTD(customerID)` on label. Colour by `Risk Band` (red → amber →
grey). This is the headline: how much money sits in each band, not how many
customers.

**2 — Segment scatter.** `AVG(Monthly Charges)` on columns, `AVG(Churn
Probability)` on rows, `Segment Name` on colour, `COUNTD(customerID)` on size,
`Segment Name` on label. Add a reference line at the average churn probability.
Bubbles above the line and to the right are where retention spend earns the most.

**3 — Model performance.** `Risk Decile` on columns (as a dimension, descending),
`AVG(Actual Churn)` on rows as bars. A model that ranks well makes this a
staircase. Add `prediction_outcome` on colour in a second sheet to show *where*
the model is wrong, which the AUC number cannot.

**4 — Customer explanation.** Filter to one customer with a `customerID`
parameter. Bars for `driver_1_shap`, `driver_2_shap`, `driver_3_shap`, coloured
by direction (red = increases churn, blue = decreases). Label with the driver
feature names. This is what makes the dashboard usable by a retention agent
rather than only by an analyst.

For a fuller version, connect `fct_shap_customer.csv` instead, filter
`rank_within_customer <= 5`, and put `feature` on rows with `shap_value` on
columns — a proper diverging bar chart of one customer's five biggest drivers.

**5 (optional) — Causal results.** From `fct_causal_estimates.csv`: `estimator`
on rows, `estimate` on columns, `analysis` as the row grouping, `adjusted` on
colour. Add `ci_low`/`ci_high` as a reference band. It shows the naive
comparisons next to the corrected ones, which is the entire point of that
analysis.

## Assemble

New dashboard, 1200 × 900, tiled. Sheet 1 top-left, sheet 2 top-right, sheet 3
bottom-left, sheet 4 bottom-right. Add `Risk Band`, `Contract` and `Segment Name`
as dashboard-wide filters (right-click each → *Apply to Worksheets* → *All Using
This Data Source*).

## Publish

Tableau Public: *Server → Tableau Public → Save to Tableau Public*. Note that
**everything published to Tableau Public is publicly visible** — that is fine
here, since the dataset is the public Kaggle Telco set and carries no real
customer data.

Then screenshot the dashboard to `docs/images/tableau-dashboard.png` and link it
from the main README, next to the SHAP and MLflow screenshots.
