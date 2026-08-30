-- Stage 2: derive model features in SQL.
--
-- These are aggregations the raw columns do not express directly:
--
--   num_addon_services  how many of the six optional services a customer takes.
--                       Churn concentrates among customers paying a high bill
--                       while subscribing to almost nothing.
--   avg_monthly_spend   lifetime billing rate (totalcharges / tenure), which
--                       differs from monthlycharges whenever pricing changed.
--   charges_ratio       current price vs lifetime average; > 1 means a recent
--                       increase, a classic churn trigger.
--   tenure_bucket       lifecycle stage; churn risk is heavily front-loaded.
--   spend_bucket        price tier.
--
-- 'No internet service' is deliberately not counted as a subscription: it is a
-- placeholder for customers without internet, not a declined add-on.

DROP TABLE IF EXISTS customer_features;

CREATE TABLE customer_features AS
WITH addons AS (
    SELECT
        customerid,
        (CASE WHEN onlinesecurity   = 'Yes' THEN 1 ELSE 0 END)
      + (CASE WHEN onlinebackup     = 'Yes' THEN 1 ELSE 0 END)
      + (CASE WHEN deviceprotection = 'Yes' THEN 1 ELSE 0 END)
      + (CASE WHEN techsupport      = 'Yes' THEN 1 ELSE 0 END)
      + (CASE WHEN streamingtv      = 'Yes' THEN 1 ELSE 0 END)
      + (CASE WHEN streamingmovies  = 'Yes' THEN 1 ELSE 0 END) AS num_addon_services
    FROM clean_customers
)
SELECT
    c.*,
    a.num_addon_services,

    -- NULLIF guards the tenure = 0 rows; they fall back to the current rate.
    COALESCE(
        c.totalcharges / NULLIF(CAST(c.tenure AS DOUBLE PRECISION), 0),
        c.monthlycharges
    ) AS avg_monthly_spend,

    c.monthlycharges / NULLIF(
        COALESCE(
            c.totalcharges / NULLIF(CAST(c.tenure AS DOUBLE PRECISION), 0),
            c.monthlycharges
        ), 0
    ) AS charges_ratio,

    CASE
        WHEN c.tenure <= 6  THEN '0-6m'
        WHEN c.tenure <= 12 THEN '6-12m'
        WHEN c.tenure <= 24 THEN '1-2y'
        WHEN c.tenure <= 48 THEN '2-4y'
        ELSE '4y+'
    END AS tenure_bucket,

    CASE
        WHEN c.monthlycharges <  35 THEN 'low'
        WHEN c.monthlycharges <  65 THEN 'medium'
        WHEN c.monthlycharges <  90 THEN 'high'
        ELSE 'premium'
    END AS spend_bucket
FROM clean_customers c
JOIN addons a ON a.customerid = c.customerid;
