"""Lightweight feature contract shared by training, API, and UI code."""

TARGET_COLUMN = "Revenue"

# PageValues is a Google Analytics metric close to the outcome and is excluded to reduce
# target-proxy leakage in the deployable feature contract.
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

# Operating systems, browsers, regions, and traffic types are category codes, not ordered
# measurements; the model treats them as categorical features.
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

INTEGER_COLUMNS = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

FLOAT_COLUMNS = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]
