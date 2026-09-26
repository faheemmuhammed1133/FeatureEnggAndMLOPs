"""Leakage-aware preprocessing and candidate classifiers for session outcomes."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Revenue"

# PageValues is deliberately omitted from the model contract. It is a Google Analytics
# page-value metric that can encode information closely tied to the transaction outcome.
# See README.md and the notebook for the leakage discussion.
NUMERIC_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]

# These integer columns are identifiers/codes for categories, not measurements with a
# meaningful numeric distance. One-hot encoding avoids treating (for example) browser 8 as
# four times browser 2.
CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
EXCLUDED_COLUMNS = ["Revenue", "PageValues"]


def build_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Build preprocessing that is fitted only when the containing pipeline is fitted."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_logistic_pipeline() -> Pipeline:
    """Interpretable, regularized baseline with balanced class weights."""
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])


def build_forest_pipeline() -> Pipeline:
    """Nonlinear comparison model; scaling is unnecessary for a tree model."""
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ])


def build_candidates() -> dict[str, Pipeline]:
    """Return modest, reproducible candidates for validation-set comparison."""
    return {
        "balanced_logistic_regression": build_logistic_pipeline(),
        "balanced_random_forest": build_forest_pipeline(),
    }
