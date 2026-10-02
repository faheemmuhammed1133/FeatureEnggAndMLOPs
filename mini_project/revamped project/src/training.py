"""Training workflow shared by the notebook and the command-line entry point."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.shopper_pipeline import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "shopper_purchase_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the supplied CSV without changing its row population."""
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + EXCLUDED_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Revenue contains missing labels; resolve them before training.")
    return frame


def split_dataset(frame: pd.DataFrame, *, random_state: int = 42) -> dict[str, object]:
    """Create stratified 60/20/20 train, validation, and test partitions."""
    X = frame[FEATURE_COLUMNS].copy()
    y = frame[TARGET_COLUMN].astype(int).copy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_train_validation": X_train_val,
        "y_train_validation": y_train_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def classification_metrics(y_true, probabilities, *, threshold: float = 0.5) -> dict[str, float]:
    """Compute threshold-free ranking metrics and threshold-dependent metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def best_f1_threshold(y_true, probabilities) -> float:
    """Select a decision threshold on validation predictions, never on the test set."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def train_and_save(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    *,
    random_state: int = 42,
) -> dict[str, object]:
    """Compare candidates on validation data, select a threshold, then save the model."""
    frame = load_dataset(data_path)
    partitions = split_dataset(frame, random_state=random_state)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(partitions["X_train"], partitions["y_train"])
    dummy_validation = dummy.predict_proba(partitions["X_validation"])[:, 1]
    dummy_metrics = classification_metrics(
        partitions["y_validation"], dummy_validation, threshold=0.5
    )

    candidate_pipelines = build_candidates()
    validation_results: dict[str, dict[str, float]] = {}
    for name, candidate in candidate_pipelines.items():
        candidate.fit(partitions["X_train"], partitions["y_train"])
        probabilities = candidate.predict_proba(partitions["X_validation"])[:, 1]
        validation_results[name] = classification_metrics(
            partitions["y_validation"], probabilities, threshold=0.5
        )

    selected_name = max(
        validation_results,
        key=lambda name: validation_results[name]["average_precision"],
    )
    selected_candidate = candidate_pipelines[selected_name]
    validation_probabilities = selected_candidate.predict_proba(
        partitions["X_validation"]
    )[:, 1]
    decision_threshold = best_f1_threshold(
        partitions["y_validation"], validation_probabilities
    )
    selected_validation_metrics = classification_metrics(
        partitions["y_validation"],
        validation_probabilities,
        threshold=decision_threshold,
    )

    # Refit the selected recipe on train + validation after model/threshold choices are
    # complete. The held-out test partition stays untouched until the final report below.
    selected_candidate.fit(
        partitions["X_train_validation"], partitions["y_train_validation"]
    )
    test_probabilities = selected_candidate.predict_proba(partitions["X_test"])[:, 1]
    test_metrics = classification_metrics(
        partitions["y_test"], test_probabilities, threshold=decision_threshold
    )

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_candidate, model_path)

    metadata: dict[str, object] = {
        "model_name": selected_name,
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "positive_label": True,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "excluded_feature_reason": {
            "PageValues": "Potential target proxy; excluded from the deployable feature contract.",
            "Revenue": "Target label, not an input feature.",
        },
        "selection_metric": "validation_average_precision",
        "decision_threshold_selection": "validation_max_f1",
        "decision_threshold": decision_threshold,
        "rows": int(len(frame)),
        "positive_rate": float(frame[TARGET_COLUMN].astype(bool).mean()),
        "exact_duplicate_rows_kept": int(frame.duplicated().sum()),
        "split_rows": {
            "train": int(len(partitions["X_train"])),
            "validation": int(len(partitions["X_validation"])),
            "test": int(len(partitions["X_test"])),
        },
        "dummy_validation_metrics": dummy_metrics,
        "candidate_validation_metrics": validation_results,
        "selected_validation_metrics": selected_validation_metrics,
        "final_test_metrics": test_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Selected model: {selected_name}")
    print(f"Decision threshold (validation max F1): {decision_threshold:.4f}")
    print("Validation metrics:", json.dumps(selected_validation_metrics, indent=2))
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    return metadata
