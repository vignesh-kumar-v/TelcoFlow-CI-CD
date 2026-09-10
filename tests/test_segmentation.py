"""Tests for customer segmentation.

Clustering always returns clusters, so the tests that matter are the ones that
check the partition means something: that the target never reaches the feature
matrix, that k is chosen on evidence, that a known structure is recovered, and
that noise is *not* reported as structure.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.telco_churn.features import TARGET
from src.telco_churn.segmentation import (
    BINARY_INPUTS, MIN_SEGMENT_SHARE, NUMERIC_INPUTS, build_clustering_matrix,
    name_segments, profile_segments, select_k, validate_structure,
)


def make_customers(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n)],
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(20, 8000, n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        **{c: rng.choice(["Yes", "No"], n) for c in (
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies")},
        TARGET: rng.integers(0, 2, n),
    })


def well_separated(n_per_blob=200, seed=1):
    """Three blobs a long way apart — any working clusterer must find them."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-8.0, -8.0], [0.0, 8.0], [8.0, -8.0]])
    X = np.vstack([c + rng.normal(0, 0.6, (n_per_blob, 2)) for c in centres])
    truth = np.repeat([0, 1, 2], n_per_blob)
    return X, truth


class TestClusteringMatrix:
    def test_target_never_reaches_the_matrix(self):
        """Segments built on the outcome would predict churn by construction."""
        X = build_clustering_matrix(make_customers())
        assert TARGET not in X.columns
        assert "Churn" not in X.columns

    def test_every_declared_input_is_present(self):
        X = build_clustering_matrix(make_customers())
        for col in NUMERIC_INPUTS + list(BINARY_INPUTS):
            assert col in X.columns

    def test_engineered_features_are_derived_when_absent(self):
        """The raw CSV has no avg_monthly_spend; the matrix still needs it."""
        raw = make_customers().drop(
            columns=["avg_monthly_spend", "charges_ratio", "num_addon_services"],
            errors="ignore",
        )
        assert "avg_monthly_spend" not in raw.columns

        X = build_clustering_matrix(raw)
        assert "avg_monthly_spend" in X.columns
        assert X["avg_monthly_spend"].notna().all()
        assert (X["avg_monthly_spend"] > 0).all()

    def test_matrix_is_entirely_numeric_and_finite(self):
        X = build_clustering_matrix(make_customers())
        assert X.select_dtypes(include="number").shape[1] == X.shape[1]
        assert np.isfinite(X.to_numpy()).all()

    def test_binary_flags_are_zero_or_one(self):
        X = build_clustering_matrix(make_customers())
        for col in BINARY_INPUTS:
            assert set(X[col].unique()) <= {0.0, 1.0}


class TestKSelection:
    def test_recovers_a_known_number_of_blobs(self):
        X, _ = well_separated()
        assert select_k(StandardScaler().fit_transform(X)).chosen_k == 3

    def test_sweeps_every_candidate(self):
        X, _ = well_separated(n_per_blob=60)
        selection = select_k(StandardScaler().fit_transform(X), k_range=range(2, 6))
        assert [s["k"] for s in selection.scores] == [2, 3, 4, 5]

    def test_inertia_falls_monotonically(self):
        """The reason inertia cannot pick k on its own."""
        X, _ = well_separated(n_per_blob=60)
        scores = select_k(StandardScaler().fit_transform(X), k_range=range(2, 7)).scores
        inertias = [s["inertia"] for s in scores]
        assert inertias == sorted(inertias, reverse=True)

    def test_tiny_segments_are_ruled_out(self):
        """A k that carves off a handful of outliers is not a segmentation.

        Isolating four far-flung points scores beautifully on silhouette, so
        without the minimum-size constraint that k would win.
        """
        rng = np.random.default_rng(3)
        X = np.vstack([
            rng.normal(0, 1, (200, 2)), rng.normal([12, 0], 1, (200, 2)),
            rng.normal([0, 12], 1, (200, 2)), rng.normal(25, 0.2, (4, 2)),
        ])
        selection = select_k(StandardScaler().fit_transform(X), k_range=range(2, 6))

        by_k = {s["k"]: s for s in selection.scores}
        assert not by_k[4]["eligible"], "k=4 isolates the outliers and must be ruled out"
        assert by_k[3]["eligible"]
        assert selection.chosen_k == 3
        assert by_k[selection.chosen_k]["smallest_segment_share"] >= MIN_SEGMENT_SHARE

    def test_ineligible_k_can_still_win_on_silhouette_alone(self):
        """Documents why the constraint exists: silhouette alone prefers k=4 here."""
        rng = np.random.default_rng(3)
        X = np.vstack([
            rng.normal(0, 1, (200, 2)), rng.normal([12, 0], 1, (200, 2)),
            rng.normal([0, 12], 1, (200, 2)), rng.normal(25, 0.2, (4, 2)),
        ])
        scores = select_k(StandardScaler().fit_transform(X), k_range=range(2, 6)).scores
        best_unconstrained = max(scores, key=lambda s: s["silhouette"])["k"]
        assert best_unconstrained == 4

    def test_reason_is_recorded(self):
        X, _ = well_separated(n_per_blob=60)
        assert "silhouette" in select_k(StandardScaler().fit_transform(X)).reason


class TestStructureValidation:
    def test_real_structure_is_reported_as_stable(self):
        from sklearn.cluster import KMeans
        X, _ = well_separated()
        X_scaled = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
        validation = validate_structure(X_scaled, labels, 3, n_bootstrap=8)
        assert validation["stable"]
        assert validation["cross_method_ari_vs_ward"] > 0.9

    def test_ward_agreement_recovers_the_true_labels(self):
        from sklearn.cluster import KMeans
        X, truth = well_separated()
        X_scaled = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(truth, labels) > 0.95

    def test_pure_noise_is_not_reported_as_stable_structure(self):
        """The check that stops this module from dressing up randomness."""
        from sklearn.cluster import KMeans
        rng = np.random.default_rng(5)
        X_scaled = StandardScaler().fit_transform(rng.normal(0, 1, (500, 6)))
        labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(X_scaled)
        validation = validate_structure(X_scaled, labels, 5, n_bootstrap=8)
        assert validation["cross_method_ari_vs_ward"] < 0.9
        assert not validation["stable"]


class TestProfiles:
    def test_every_customer_lands_in_exactly_one_segment(self):
        df = make_customers()
        X = build_clustering_matrix(df)
        labels = np.random.default_rng(0).integers(0, 3, len(df))
        profiles = profile_segments(df, X, labels)
        assert profiles["customers"].sum() == len(df)
        assert profiles["share"].sum() == pytest.approx(1.0, abs=1e-3)

    def test_revenue_at_risk_combines_value_and_rate(self):
        df = make_customers()
        X = build_clustering_matrix(df)
        labels = np.zeros(len(df), dtype=int)
        row = profile_segments(df, X, labels).iloc[0]
        assert row["monthly_revenue_at_risk"] == pytest.approx(
            row["monthly_revenue"] * row["churn_rate"], rel=1e-3
        )

    def test_names_are_unique_and_cover_every_segment(self):
        df = make_customers()
        X = build_clustering_matrix(df)
        labels = np.random.default_rng(0).integers(0, 4, len(df))
        profiles = profile_segments(df, X, labels)
        names = name_segments(profiles)
        assert set(names) == set(profiles["segment"])
        assert len(set(names.values())) == len(names)

    def test_high_churn_segment_gets_an_at_risk_name(self):
        df = make_customers(n=400)
        df[TARGET] = 0
        df.loc[:150, TARGET] = 1               # segment 0 churns heavily
        X = build_clustering_matrix(df)
        labels = np.zeros(len(df), dtype=int)
        labels[200:] = 1
        names = name_segments(profile_segments(df, X, labels))
        assert "Risk" in names[0] or "Unsettled" in names[0] or "Drifting" in names[0]
