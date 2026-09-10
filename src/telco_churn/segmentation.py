"""Unsupervised customer segmentation.

The supervised model answers "who is about to churn?". It does not answer "who
are these customers?" — and a retention budget is allocated across segments, not
across 7,043 individual probabilities. This module clusters the customer base on
behaviour and value, then profiles what each cluster is worth and how fast it is
leaking.

Three things keep this from being decoration:

  * **Churn is excluded from the clustering matrix.** Segments are built from
    behaviour alone; churn rate is a *post-hoc* profile of each segment. Feeding
    the target in would produce clusters that predict churn by construction and
    say nothing about who the customers are.
  * **k is chosen, not assumed.** Silhouette, Davies-Bouldin and Calinski-
    Harabasz are swept over a range of k, with a minimum-segment-size constraint
    so the winner is not a k that isolates a handful of outliers.
  * **The structure is validated, not asserted.** Ward hierarchical clustering
    is fitted at the same k and compared to k-means by Adjusted Rand Index, and
    the labels are bootstrapped. Two different algorithms agreeing, and labels
    surviving resampling, is the difference between real structure and a
    partition of noise.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.telco_churn.features import (  # noqa: E402
    TARGET, compute_engineered_features,
)

console = Console()

K_RANGE = range(2, 11)
RANDOM_STATE = 42

# Behaviour and value only. Every column here is something the customer chose or
# accrued; none of them is the outcome.
NUMERIC_INPUTS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "charges_ratio", "num_addon_services",
]

# Binary commitment/behaviour flags, already on a 0/1 scale.
BINARY_INPUTS = {
    "is_month_to_month": ("Contract", "Month-to-month"),
    "is_two_year": ("Contract", "Two year"),
    "has_fiber": ("InternetService", "Fiber optic"),
    "no_internet": ("InternetService", "No"),
    "is_paperless": ("PaperlessBilling", "Yes"),
    "pays_by_echeck": ("PaymentMethod", "Electronic check"),
    "has_partner": ("Partner", "Yes"),
    "has_dependents": ("Dependents", "Yes"),
}

# A segment smaller than this is an outlier pocket, not a market to act on.
MIN_SEGMENT_SHARE = 0.03


def build_clustering_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the behavioural feature matrix, with no target leakage.

    Engineered columns are recomputed when the caller passed raw data, so this
    works on either the SQL output or the raw CSV.
    """
    if not all(c in df.columns for c in ("avg_monthly_spend", "charges_ratio")):
        df = compute_engineered_features(df)

    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_INPUTS:
        X[col] = pd.to_numeric(df[col], errors="coerce")
    for name, (col, value) in BINARY_INPUTS.items():
        X[name] = (df[col] == value).astype(float)
    X["senior"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").astype(float)

    if TARGET in X.columns:  # defensive: the target must never reach the clusterer
        raise AssertionError("target column leaked into the clustering matrix")

    return X.fillna(X.median(numeric_only=True))


@dataclass
class KSelection:
    """The sweep over k and the choice it justifies."""

    scores: list
    chosen_k: int
    reason: str


def select_k(X_scaled: np.ndarray, k_range=K_RANGE, random_state: int = RANDOM_STATE,
             min_share: float = MIN_SEGMENT_SHARE) -> KSelection:
    """Sweep k and pick the best silhouette among usably-balanced partitions.

    Silhouette decides because it measures separation per point and is
    comparable across k; inertia is reported for the elbow but cannot pick k on
    its own, since it falls monotonically. Partitions with a segment below
    `min_share` are recorded but not eligible — carving off 40 outliers scores
    well and is not a segmentation.
    """
    n = len(X_scaled)
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        sizes = np.bincount(labels, minlength=k)
        smallest_share = float(sizes.min() / n)
        scores.append({
            "k": int(k),
            "inertia": round(float(km.inertia_), 2),
            "silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_scaled, labels)), 2),
            "smallest_segment_share": round(smallest_share, 4),
            "eligible": bool(smallest_share >= min_share),
        })

    eligible = [s for s in scores if s["eligible"]]
    pool = eligible or scores
    best = max(pool, key=lambda s: s["silhouette"])
    reason = (
        f"highest silhouette ({best['silhouette']:.4f}) among partitions whose "
        f"smallest segment holds at least {min_share:.0%} of customers"
    )
    if not eligible:
        reason = (
            f"highest silhouette ({best['silhouette']:.4f}); no k met the "
            f"{min_share:.0%} minimum-segment constraint"
        )
    return KSelection(scores=scores, chosen_k=best["k"], reason=reason)


def validate_structure(X_scaled: np.ndarray, labels: np.ndarray, k: int,
                       n_bootstrap: int = 25, random_state: int = RANDOM_STATE) -> dict:
    """Check the partition is real structure rather than a slice through noise.

    Two independent checks: a different algorithm (Ward linkage, which optimises
    a different objective) should recover roughly the same partition, and the
    labels should survive resampling the customer base.
    """
    ward_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
    cross_method_ari = float(adjusted_rand_score(labels, ward_labels))

    rng = np.random.default_rng(random_state)
    n = len(X_scaled)
    aris = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        resampled = KMeans(n_clusters=k, n_init=10, random_state=random_state) \
            .fit_predict(X_scaled[idx])
        aris.append(adjusted_rand_score(labels[idx], resampled))

    return {
        "cross_method_ari_vs_ward": round(cross_method_ari, 4),
        "bootstrap_ari_mean": round(float(np.mean(aris)), 4),
        "bootstrap_ari_std": round(float(np.std(aris)), 4),
        "bootstrap_iterations": n_bootstrap,
        # 0.5 is the conventional line between "unstable" and "reproducible".
        "stable": bool(np.mean(aris) >= 0.5 and cross_method_ari >= 0.5),
    }


def profile_segments(df: pd.DataFrame, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """What each segment is, what it is worth, and how fast it is leaking."""
    profile = df.copy()
    if "num_addon_services" not in profile.columns:
        profile = compute_engineered_features(profile)
    profile["segment"] = labels

    rows = []
    total = len(profile)
    global_means = X.mean()
    global_sds = X.std().replace(0, np.nan)

    for segment in sorted(profile["segment"].unique()):
        mask = profile["segment"] == segment
        members = profile[mask]
        segment_means = X[mask.to_numpy()].mean()

        # Standardised distance from the overall average: what makes this
        # segment distinctive, on a scale comparable across features.
        z = ((segment_means - global_means) / global_sds).dropna()
        distinguishing = z.reindex(z.abs().sort_values(ascending=False).index).head(4)

        churn_rate = (
            float(members[TARGET].mean()) if TARGET in members.columns else float("nan")
        )
        monthly_revenue = float(members["MonthlyCharges"].sum())

        rows.append({
            "segment": int(segment),
            "customers": int(len(members)),
            "share": round(len(members) / total, 4),
            "churn_rate": round(churn_rate, 4),
            "avg_tenure": round(float(members["tenure"].mean()), 1),
            "avg_monthly_charges": round(float(members["MonthlyCharges"].mean()), 2),
            "avg_total_charges": round(float(members["TotalCharges"].mean()), 2),
            "avg_addons": round(float(members["num_addon_services"].mean()), 2),
            "pct_month_to_month": round(float((members["Contract"] == "Month-to-month").mean()), 4),
            "pct_fiber": round(float((members["InternetService"] == "Fiber optic").mean()), 4),
            "monthly_revenue": round(monthly_revenue, 2),
            "monthly_revenue_share": round(monthly_revenue / float(profile["MonthlyCharges"].sum()), 4),
            # Revenue sitting in the churn-risk pool: size x rate x value. This
            # is the number that should drive where retention spend goes.
            "monthly_revenue_at_risk": round(monthly_revenue * churn_rate, 2),
            "distinguishing_features": {k: round(float(v), 2) for k, v in distinguishing.items()},
        })

    return pd.DataFrame(rows)


def name_segments(profiles: pd.DataFrame) -> dict:
    """Give each segment a label a stakeholder can act on.

    Derived from the profile rather than hardcoded per cluster id, because
    k-means numbers its clusters arbitrarily — a fixed mapping would silently
    mislabel everything the next time the model is refit.
    """
    base_churn = float((profiles["churn_rate"] * profiles["customers"]).sum()
                       / profiles["customers"].sum())
    median_tenure = float(profiles["avg_tenure"].median())
    median_spend = float(profiles["avg_monthly_charges"].median())

    names = {}
    for _, row in profiles.iterrows():
        high_churn = row["churn_rate"] > base_churn * 1.25
        low_churn = row["churn_rate"] < base_churn * 0.6
        long_tenure = row["avg_tenure"] > median_tenure
        high_spend = row["avg_monthly_charges"] > median_spend
        committed = row["pct_month_to_month"] < 0.5

        if high_churn and high_spend:
            name = "High-Value At Risk"
        elif high_churn and not long_tenure:
            name = "New and Unsettled"
        elif high_churn:
            name = "Drifting Regulars"
        elif low_churn and committed and high_spend:
            name = "Premium Loyalists"
        elif low_churn and committed:
            name = "Locked-In Core"
        elif not high_spend and not long_tenure:
            name = "Budget Newcomers"
        elif high_spend:
            name = "Established Spenders"
        else:
            name = "Steady Mainstream"

        # k-means can produce two segments matching the same rule; keep names unique.
        if name in names.values():
            name = f"{name} ({int(row['segment'])})"
        names[int(row["segment"])] = name
    return names


def plot_k_selection(scores: list, chosen_k: int, out_path: Path) -> Path:
    """Elbow and silhouette side by side — the evidence behind the choice of k."""
    ks = [s["k"] for s in scores]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(ks, [s["inertia"] for s in scores], marker="o", color="#264653", linewidth=2)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow — falls monotonically, so it cannot pick k alone")
    ax1.grid(alpha=0.25)

    ax2.plot(ks, [s["silhouette"] for s in scores], marker="o", color="#2a9d8f", linewidth=2)
    ax2.axvline(chosen_k, color="#e76f51", linestyle=":", linewidth=1.8)
    ax2.text(chosen_k, min(s["silhouette"] for s in scores), f" chosen k={chosen_k}",
             fontsize=9, color="#e76f51")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette — comparable across k, so this decides")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_segments(X_scaled: np.ndarray, labels: np.ndarray, profiles: pd.DataFrame,
                  names: dict, out_path: Path) -> Path:
    """PCA projection plus the chart that actually drives budget: churn vs value."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    palette = plt.get_cmap("tab10")

    for segment in sorted(np.unique(labels)):
        mask = labels == segment
        ax1.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.45,
                    color=palette(segment % 10), label=names[int(segment)])
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax1.set_title(f"Segments in the first two components ({explained:.0%} of variance)")
    ax1.legend(fontsize=8, markerscale=2, loc="best")
    ax1.grid(alpha=0.2)

    base_churn = float((profiles["churn_rate"] * profiles["customers"]).sum()
                       / profiles["customers"].sum())
    sizes = profiles["customers"] / profiles["customers"].max() * 1600
    ax2.scatter(profiles["avg_monthly_charges"], profiles["churn_rate"],
                s=sizes, alpha=0.6,
                color=[palette(int(s) % 10) for s in profiles["segment"]])
    for _, row in profiles.iterrows():
        ax2.annotate(names[int(row["segment"])],
                     (row["avg_monthly_charges"], row["churn_rate"]),
                     textcoords="offset points", xytext=(0, 14),
                     ha="center", fontsize=8.5)
    ax2.axhline(base_churn, color="grey", linestyle="--", linewidth=1)
    ax2.text(profiles["avg_monthly_charges"].min(), base_churn + 0.008,
             f"base rate {base_churn:.1%}", fontsize=8.5, color="grey")
    ax2.set_xlabel("Average monthly charges ($)")
    ax2.set_ylabel("Churn rate")
    ax2.set_title("Where retention spend should go (bubble size = customers)")
    ax2.grid(alpha=0.25)
    # The bubbles sit at the data limits and their labels are offset upward, so
    # the default margins clip both.
    ax2.margins(x=0.18, y=0.22)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(
    clean_path: Path = Path.cwd() / "data" / "processed" / "cleaned_data.parquet",
    outputs_dir: Path = Path.cwd() / "outputs",
    reports_dir: Path = Path.cwd() / "reports",
    docs_dir: Path = Path.cwd() / "docs" / "images",
    k: int = typer.Option(None, help="Force a value of k instead of selecting one"),
    n_bootstrap: int = typer.Option(25, help="Bootstrap resamples for the stability check"),
    random_state: int = typer.Option(RANDOM_STATE, help="Random seed"),
):
    console.rule("[bold magenta]Telco Churn: Customer Segmentation")

    df = pd.read_parquet(clean_path)
    console.log(f"[green]Loaded {len(df):,} customers")

    X = build_clustering_matrix(df)
    console.log(f"[blue]Clustering on {X.shape[1]} behavioural features (target excluded)")

    # k-means minimises squared Euclidean distance, so an unscaled TotalCharges
    # in the thousands would decide every cluster on its own.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k is None:
        console.log(f"[blue]Sweeping k over {K_RANGE.start}-{K_RANGE.stop - 1}...")
        selection = select_k(X_scaled, random_state=random_state)
        chosen_k = selection.chosen_k
        console.log(f"[green]Selected k={chosen_k} — {selection.reason}")
    else:
        selection = select_k(X_scaled, k_range=range(k, k + 1), random_state=random_state)
        chosen_k = k
        console.log(f"[yellow]Using forced k={k}")

    kmeans = KMeans(n_clusters=chosen_k, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)

    console.log("[blue]Validating structure against Ward linkage and bootstrap resamples...")
    validation = validate_structure(X_scaled, labels, chosen_k,
                                    n_bootstrap=n_bootstrap, random_state=random_state)
    if validation["stable"]:
        console.log(
            f"[green]Structure holds — Ward agreement ARI "
            f"{validation['cross_method_ari_vs_ward']:.3f}, bootstrap ARI "
            f"{validation['bootstrap_ari_mean']:.3f}"
        )
    else:
        console.log(
            f"[yellow]Segments are unstable — Ward ARI "
            f"{validation['cross_method_ari_vs_ward']:.3f}, bootstrap ARI "
            f"{validation['bootstrap_ari_mean']:.3f}. Treat the profiles as indicative."
        )

    profiles = profile_segments(df, X, labels)
    names = name_segments(profiles)
    profiles.insert(1, "name", profiles["segment"].map(names))

    table = Table(title=f"Customer segments (k={chosen_k})")
    for col in ("Segment", "Customers", "Share", "Churn", "Tenure", "Monthly", "Addons", "Rev at risk"):
        table.add_column(col, justify="right" if col != "Segment" else "left")
    for _, row in profiles.sort_values("monthly_revenue_at_risk", ascending=False).iterrows():
        table.add_row(
            row["name"], f"{row['customers']:,}", f"{row['share']:.1%}",
            f"{row['churn_rate']:.1%}", f"{row['avg_tenure']:.0f}m",
            f"${row['avg_monthly_charges']:.0f}", f"{row['avg_addons']:.1f}",
            f"${row['monthly_revenue_at_risk']:,.0f}",
        )
    console.print(table)

    worst = profiles.loc[profiles["monthly_revenue_at_risk"].idxmax()]
    console.print(
        f"\n[bold]{worst['name']}[/bold] holds the most revenue at risk: "
        f"{worst['customers']:,} customers churning at {worst['churn_rate']:.1%} "
        f"on ${worst['avg_monthly_charges']:.0f}/month "
        f"= [bold]${worst['monthly_revenue_at_risk']:,.0f}/month[/bold] exposed."
    )

    assignments = df[["customerID"]].copy() if "customerID" in df.columns else pd.DataFrame(index=df.index)
    assignments["segment"] = labels
    assignments["segment_name"] = assignments["segment"].map(names)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(outputs_dir / "customer_segments.parquet", index=False)
    assignments.to_csv(outputs_dir / "customer_segments.csv", index=False)
    console.log(f"[green]Segment assignments saved to {outputs_dir}/customer_segments.*")

    k_plot = plot_k_selection(selection.scores, chosen_k, Path(docs_dir) / "segmentation-k-selection.png")
    seg_plot = plot_segments(X_scaled, labels, profiles, names, Path(docs_dir) / "segmentation-profile.png")
    console.log(f"[green]Plots saved: {k_plot.name}, {seg_plot.name}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_customers": len(df),
        "features_used": X.columns.tolist(),
        "target_excluded": True,
        "k_selection": {
            "range": [K_RANGE.start, K_RANGE.stop - 1],
            "chosen_k": chosen_k,
            "reason": selection.reason,
            "scores": selection.scores,
        },
        "validation": validation,
        "segments": json.loads(profiles.to_json(orient="records")),
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"segmentation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.log(f"[green]Segmentation report saved: {report_path}")

    console.rule("[bold green]Segmentation completed!")


if __name__ == "__main__":
    typer.run(main)
