-- Stage 1: clean the raw ingest into a typed, analysis-ready table.
--
-- Replaces the pandas cleaning step. Two things the raw CSV gets wrong:
--   * total_charges arrives as text, because 11 rows (all tenure = 0, i.e. brand
--     new customers who have not been billed yet) hold a single space instead
--     of a number. Those become NULL here and are imputed with the median.
--   * churn is Yes/No text and needs to be a 0/1 integer target.
--
-- Identifiers are lower case throughout: Postgres folds unquoted names to lower
-- case while SQLite preserves them, so the ingest normalises column names and
-- the export maps them back to the mixed-case names the model expects.
--
-- Median uses an ORDER BY/LIMIT window rather than percentile_cont, which
-- SQLite does not implement.

DROP TABLE IF EXISTS clean_customers;

CREATE TABLE clean_customers AS
WITH typed AS (
    SELECT
        customerid,
        gender,
        seniorcitizen,
        partner,
        dependents,
        tenure,
        phoneservice,
        multiplelines,
        internetservice,
        onlinesecurity,
        onlinebackup,
        deviceprotection,
        techsupport,
        streamingtv,
        streamingmovies,
        contract,
        paperlessbilling,
        paymentmethod,
        monthlycharges,
        -- Blank strings become NULL so they can be imputed rather than
        -- silently casting to zero.
        CAST(NULLIF(TRIM(CAST(totalcharges AS VARCHAR(64))), '') AS DOUBLE PRECISION)
            AS totalcharges,
        CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END AS churn
    FROM raw_customers
),
-- Portable median: average the middle one (odd n) or two (even n) values.
median_charges AS (
    SELECT AVG(totalcharges) AS median_total_charges
    FROM (
        SELECT totalcharges
        FROM typed
        WHERE totalcharges IS NOT NULL
        ORDER BY totalcharges
        LIMIT 2 - (SELECT COUNT(totalcharges) FROM typed) % 2
        OFFSET (SELECT (COUNT(totalcharges) - 1) / 2 FROM typed)
    ) AS middle_values
)
SELECT
    t.customerid,
    t.gender,
    t.seniorcitizen,
    t.partner,
    t.dependents,
    t.tenure,
    t.phoneservice,
    t.multiplelines,
    t.internetservice,
    t.onlinesecurity,
    t.onlinebackup,
    t.deviceprotection,
    t.techsupport,
    t.streamingtv,
    t.streamingmovies,
    t.contract,
    t.paperlessbilling,
    t.paymentmethod,
    t.monthlycharges,
    COALESCE(t.totalcharges, m.median_total_charges) AS totalcharges,
    t.churn
FROM typed t
CROSS JOIN median_charges m;
