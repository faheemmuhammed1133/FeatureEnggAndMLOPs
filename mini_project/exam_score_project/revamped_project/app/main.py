"""FastAPI prediction service backed by the saved sklearn pipeline."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.schema import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "shopper_purchase_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"

pipeline = None
model_metadata: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, model_metadata
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Trained model artifacts are missing. Run `python train.py` from the project "
            "directory before starting the API."
        )
    pipeline = joblib.load(MODEL_PATH)
    model_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    yield
    pipeline = None
    model_metadata = {}


app = FastAPI(
    title="Online Shopper Purchase Intention API",
    description=(
        "Estimates the probability that a summarized online shopping session has the "
        "Revenue outcome. PageValues is excluded from the model input contract because it "
        "may act as a target proxy. This educational model is not a real-time intervention "
        "system; see the project README for scope and data limitations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ShopperSessionInput(BaseModel):
    """Session inputs; blank CSV cells are passed to the pipeline imputers."""

    Administrative: int | None = Field(default=None, ge=0, description="Administrative pages viewed")
    Administrative_Duration: float | None = Field(default=None, ge=0, description="Time on administrative pages")
    Informational: int | None = Field(default=None, ge=0, description="Informational pages viewed")
    Informational_Duration: float | None = Field(default=None, ge=0, description="Time on informational pages")
    ProductRelated: int | None = Field(default=None, ge=0, description="Product-related pages viewed")
    ProductRelated_Duration: float | None = Field(default=None, ge=0, description="Time on product-related pages")
    BounceRates: float | None = Field(default=None, ge=0, le=1)
    ExitRates: float | None = Field(default=None, ge=0, le=1)
    SpecialDay: float | None = Field(default=None, ge=0, le=1)
    Month: str | None = Field(default=None, min_length=1, max_length=12)
    OperatingSystems: int | None = Field(default=None, ge=1)
    Browser: int | None = Field(default=None, ge=1)
    Region: int | None = Field(default=None, ge=1)
    TrafficType: int | None = Field(default=None, ge=1)
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"] | None = None
    Weekend: bool | None = None

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 25.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 18,
                "ProductRelated_Duration": 620.5,
                "BounceRates": 0.01,
                "ExitRates": 0.03,
                "SpecialDay": 0.0,
                "Month": "May",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        },
    )


class PredictionOutput(BaseModel):
    purchase_probability: float
    predicted_purchase: bool
    decision_threshold: float
    model_version: str


class BatchPredictionInput(BaseModel):
    """One bounded chunk of sessions for vectorized batch inference."""

    sessions: list[ShopperSessionInput] = Field(min_length=1, max_length=500)


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_version": model_metadata.get("model_version"),
    }


@app.get("/model-info")
def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return {
        "model_name": model_metadata.get("model_name"),
        "model_version": model_metadata.get("model_version"),
        "decision_threshold": model_metadata.get("decision_threshold"),
        "excluded_columns": model_metadata.get("excluded_columns"),
        "final_test_metrics": model_metadata.get("final_test_metrics"),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(session: ShopperSessionInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_frame = pd.DataFrame([session.model_dump()], columns=FEATURE_COLUMNS)
    input_frame = input_frame.replace({None: np.nan})
    probability = float(pipeline.predict_proba(input_frame)[0, 1])
    threshold = float(model_metadata["decision_threshold"])
    return PredictionOutput(
        purchase_probability=round(probability, 4),
        predicted_purchase=probability >= threshold,
        decision_threshold=round(threshold, 4),
        model_version=str(model_metadata.get("model_version", "unknown")),
    )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchPredictionInput):
    """Score a bounded batch in one vectorized model call."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_frame = pd.DataFrame(
        [session.model_dump() for session in batch.sessions],
        columns=FEATURE_COLUMNS,
    ).replace({None: np.nan})
    probabilities = pipeline.predict_proba(input_frame)[:, 1]
    threshold = float(model_metadata["decision_threshold"])
    version = str(model_metadata.get("model_version", "unknown"))
    return BatchPredictionOutput(
        predictions=[
            PredictionOutput(
                purchase_probability=round(float(probability), 4),
                predicted_purchase=bool(probability >= threshold),
                decision_threshold=round(threshold, 4),
                model_version=version,
            )
            for probability in probabilities
        ]
    )
