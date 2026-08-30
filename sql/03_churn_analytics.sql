-- Stage 3 (reporting): churn rates by segment.
--
-- Not consumed by the model — this is the analyst-facing view that explains
-- where churn concentrates, and it doubles as a sanity check that the
-- engineered buckets actually separate the target.

DROP TABLE IF EXISTS churn_by_segment;

CREATE TABLE churn_by_segment AS
SELECT
    contract,
    tenure_bucket,
    spend_bucket,
    COUNT(*)                                        AS customers,
    SUM(churn)                                      AS churned,
    ROUND(AVG(CAST(churn AS DOUBLE PRECISION)), 4)  AS churn_rate,
    ROUND(AVG(monthlycharges), 2)                   AS avg_monthly_charges,
    ROUND(AVG(CAST(num_addon_services AS DOUBLE PRECISION)), 2) AS avg_addons
FROM customer_features
GROUP BY contract, tenure_bucket, spend_bucket
HAVING COUNT(*) >= 20
ORDER BY churn_rate DESC;
