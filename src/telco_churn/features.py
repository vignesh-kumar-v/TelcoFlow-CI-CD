"""Feature definitions and the fitted transform shared by training and serving.

Each candidate model consumes the same raw columns in a different shape:

  - LightGBM   : pandas ``category`` columns, handled natively
  - XGBoost    : integer category codes (trained with ``enable_categorical=False``)
  - Logistic   : dense one-hot matrix from a fitted ``ColumnTransformer``

``FeaturePipeline`` fits every encoder once at training time and can then
reproduce any of those three shapes, so serving code never has to guess which
one the deployed model expects.

The category level lists are pinned at fit time on purpose. ``astype("category")``
infers levels from whatever frame it is handed, so a scoring batch that happens
not to contain (say) ``"No phone service"`` would shift every subsequent code by
one and silently feed the model mislabelled features.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

DROP_COLS = ["customerID"]

NOMINAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "PaymentMethod",
]

ORDINAL_FEATURES = ["Contract"]
CONTRACT_ORDER = [["Month-to-month", "One year", "Two year"]]

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"

# Engineered columns produced by the SQL layer. Optional: the pipeline picks them
# up when present so the pure-pandas path keeps working unchanged.
ENGINEERED_NUMERIC = ["num_addon_services", "avg_monthly_spend", "charges_ratio"]
ENGINEERED_NOMINAL = ["tenure_bucket", "spend_bucket"]
ENGINEERED_FEATURES = ENGINEERED_NUMERIC + ENGINEERED_NOMINAL

# Optional services counted by num_addon_services. "No internet service" is a
# placeholder for customers without internet, not a declined add-on, so only an
# explicit "Yes" counts.
ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_BUCKETS = [(6, "0-6m"), (12, "6-12m"), (24, "1-2y"), (48, "2-4y")]
TENURE_BUCKET_DEFAULT = "4y+"
SPEND_BUCKETS = [(35, "low"), (65, "medium"), (90, "high")]
SPEND_BUCKET_DEFAULT = "premium"

# Model types, i.e. the three feature shapes above.
TREE_NATIVE = "tree_native"      # LightGBM
TREE_CODES = "tree_codes"        # XGBoost
LINEAR = "linear"                # Logistic Regression

MODEL_TYPE_BY_NAME = {
    "lightgbm": TREE_NATIVE,
    "tuned_lightgbm": TREE_NATIVE,
    "xgboost": TREE_CODES,
    "logistic_regression": LINEAR,
}


def active_features(df: pd.DataFrame):
    """Split the columns actually present into (nominal, ordinal, numeric).

    Engineered SQL columns are included when the frame carries them.
    """
    nominal = [c for c in NOMINAL_FEATURES + ENGINEERED_NOMINAL if c in df.columns]
    ordinal = [c for c in ORDINAL_FEATURES if c in df.columns]
    numeric = [c for c in NUMERIC_FEATURES + ENGINEERED_NUMERIC if c in df.columns]
    return nominal, ordinal, numeric


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers and the target, leaving only model inputs."""
    return df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Python mirror of sql/02_feature_engineering.sql.

    Needed at serving time: a single API request never passes through the
    warehouse, so the same derived columns have to be computable from the raw
    fields. Any divergence between this and the SQL is training/serving skew,
    which is why tests/test_sql.py asserts the two agree row for row.
    """
    out = df.copy()

    out["num_addon_services"] = sum(
        (out[col] == "Yes").astype(int) for col in ADDON_SERVICE_COLUMNS
    )

    tenure = out["tenure"].astype(float)
    # tenure == 0 would divide by zero; those customers fall back to the
    # current rate, matching COALESCE(..., monthlycharges) in the SQL.
    lifetime_rate = out["TotalCharges"].astype(float) / tenure.replace(0, np.nan)
    out["avg_monthly_spend"] = lifetime_rate.fillna(out["MonthlyCharges"].astype(float))

    out["charges_ratio"] = (
        out["MonthlyCharges"].astype(float)
        / out["avg_monthly_spend"].replace(0, np.nan)
    )

    out["tenure_bucket"] = _bucketize(tenure, TENURE_BUCKETS, TENURE_BUCKET_DEFAULT)
    out["spend_bucket"] = _bucketize(
        out["MonthlyCharges"].astype(float), SPEND_BUCKETS, SPEND_BUCKET_DEFAULT,
        inclusive=False,
    )
    return out


def _bucketize(values, buckets, default, inclusive=True):
    """Apply CASE WHEN-style thresholds in order.

    `inclusive` picks <= (tenure) versus < (charges), matching the SQL.
    """
    conditions = [
        values <= edge if inclusive else values < edge for edge, _ in buckets
    ]
    return np.select(conditions, [label for _, label in buckets], default=default)


def ensure_engineered_features(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Add engineered columns when the model needs them and they are absent.

    Lets one code path serve raw API payloads and warehouse-built batches alike.
    """
    if any(col in required for col in ENGINEERED_FEATURES) and not all(
        col in df.columns for col in ENGINEERED_FEATURES
    ):
        return compute_engineered_features(df)
    return df


def create_lgbm_preprocessor():
    """Ordinal encoder for ``Contract``, whose levels have a real order."""
    return OrdinalEncoder(
        categories=CONTRACT_ORDER,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )


def create_lr_preprocessor(nominal=None, ordinal=None, numeric=None):
    """One-hot + ordinal + passthrough matrix for the linear model."""
    return ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             NOMINAL_FEATURES if nominal is None else nominal),
            ("ordinal", OrdinalEncoder(categories=CONTRACT_ORDER,
                                       handle_unknown="use_encoded_value", unknown_value=-1),
             ORDINAL_FEATURES if ordinal is None else ordinal),
            ("numeric", "passthrough", NUMERIC_FEATURES if numeric is None else numeric),
        ]
    )


def prepare_data_lgbm(X: pd.DataFrame, encoder, fit: bool = False, categories: dict | None = None):
    """Encode ``Contract`` and cast nominals to ``category``.

    Kept as a standalone function for the unit tests and for callers that only
    need the LightGBM shape. ``categories`` pins the level lists; without it the
    levels are inferred from ``X``, which is only safe at fit time.
    """
    X = X.copy()
    ordinal = [c for c in ORDINAL_FEATURES if c in X.columns]
    if ordinal:
        if fit:
            X[ordinal] = encoder.fit_transform(X[ordinal])
        else:
            X[ordinal] = encoder.transform(X[ordinal])
    nominal = [c for c in NOMINAL_FEATURES + ENGINEERED_NOMINAL if c in X.columns]
    for col in nominal:
        if categories is not None and col in categories:
            X[col] = pd.Categorical(X[col], categories=categories[col])
        else:
            X[col] = X[col].astype("category")
    cat_indices = [X.columns.get_loc(c) for c in nominal + ordinal]
    return X, cat_indices


def prepare_data_lr(X: pd.DataFrame, transformer, fit: bool = False):
    if fit:
        return transformer.fit_transform(X)
    return transformer.transform(X)


@dataclass
class FeaturePipeline:
    """Every encoder the pipeline needs, fitted once and saved as one artifact."""

    contract_encoder: OrdinalEncoder = field(default_factory=create_lgbm_preprocessor)
    lr_transformer: ColumnTransformer | None = None
    categories: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    cat_indices: list = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> "FeaturePipeline":
        nominal, ordinal, numeric = active_features(X)

        # Pin category levels from the training frame so codes stay stable.
        self.categories = {
            col: sorted(X[col].dropna().unique().tolist()) for col in nominal
        }
        if ordinal:
            self.contract_encoder.fit(X[ordinal])

        self.lr_transformer = create_lr_preprocessor(nominal, ordinal, numeric)
        self.lr_transformer.fit(X)

        self.feature_names = X.columns.tolist()
        self.cat_indices = [X.columns.get_loc(c) for c in nominal + ordinal]
        return self

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder/restrict incoming columns to the fitted training layout.

        Guards against a caller passing columns in a different order, which
        would otherwise line features up against the wrong model inputs.
        """
        if not self.feature_names:
            return X
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Scoring data is missing trained features: {missing}")
        return X[self.feature_names]

    def transform_tree_native(self, X: pd.DataFrame) -> pd.DataFrame:
        """LightGBM shape: pinned ``category`` dtype columns."""
        X, _ = prepare_data_lgbm(self._align(X), self.contract_encoder,
                                 categories=self.categories)
        return X

    def transform_tree_codes(self, X: pd.DataFrame) -> pd.DataFrame:
        """XGBoost shape: integer codes from the pinned category levels."""
        X = self.transform_tree_native(X)
        for col in X.columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].cat.codes
        return X

    def transform_linear(self, X: pd.DataFrame) -> np.ndarray:
        """Logistic Regression shape: dense one-hot matrix."""
        return self.lr_transformer.transform(self._align(X))

    def transform_for(self, X: pd.DataFrame, model_type: str):
        if model_type == TREE_NATIVE:
            return self.transform_tree_native(X)
        if model_type == TREE_CODES:
            return self.transform_tree_codes(X)
        if model_type == LINEAR:
            return self.transform_linear(X)
        raise ValueError(f"Unknown model_type: {model_type!r}")
