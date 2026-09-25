"""FastAPI prediction service backed by the saved sklearn pipeline."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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
    """Session inputs accepted by the deployed model; no target or PageValues field."""

    Administrative: int = Field(ge=0, description="Administrative pages viewed")
    Administrative_Duration: float = Field(ge=0, description="Time on administrative pages")
    Informational: int = Field(ge=0, description="Informational pages viewed")
    Informational_Duration: float = Field(ge=0, description="Time on informational pages")
    ProductRelated: int = Field(ge=0, description="Product-related pages viewed")
    ProductRelated_Duration: float = Field(ge=0, description="Time on product-related pages")
    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    SpecialDay: float = Field(ge=0, le=1)
    Month: str = Field(min_length=1, max_length=12)
    OperatingSystems: int = Field(ge=1)
    Browser: int = Field(ge=1)
    Region: int = Field(ge=1)
    TrafficType: int = Field(ge=1)
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"]
    Weekend: bool

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

    input_frame = pd.DataFrame([session.model_dump()])
    probability = float(pipeline.predict_proba(input_frame)[0, 1])
    threshold = float(model_metadata["decision_threshold"])
    return PredictionOutput(
        purchase_probability=round(probability, 4),
        predicted_purchase=probability >= threshold,
        decision_threshold=round(threshold, 4),
        model_version=str(model_metadata.get("model_version", "unknown")),
    )
