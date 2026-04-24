"""
app/routes/predict.py
──────────────────────
Prediction routes — fully documented for Swagger / OpenAPI.

Endpoints
─────────
POST /predict          — single prediction with model selector
GET  /models           — list available (loaded) models
GET  /health           — liveness + readiness check
"""

from __future__ import annotations

import time
import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.services.model_loader import (
    model_registry,
    MODEL_LSTM, MODEL_GAN, MODEL_XGBOOST, MODEL_LIGHTGBM,
)
from app.utils.helpers import (
    preprocess_tabular,
    build_single_sequence,
    apply_threshold,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────

ModelName = Literal["lstm", "gan", "xgboost", "lightgbm"]


class TelemetryRow(BaseModel):
    """One 5-minute telemetry reading from the grid sensor."""

    bucket_5m: str = Field(
        ...,
        description="Timestamp of the 5-minute bucket (ISO-8601, e.g. '2026-04-01T08:00:00')",
        examples=["2026-04-01T08:00:00"],
    )
    site_id: int = Field(..., description="Encoded site identifier (integer)", examples=[0])

    # ── Three-phase measurements ──────────────────────────────
    voltageA:      float = Field(..., description="Phase-A voltage (V)",   examples=[232.1])
    voltageB:      float = Field(..., description="Phase-B voltage (V)",   examples=[231.5])
    voltageC:      float = Field(..., description="Phase-C voltage (V)",   examples=[230.8])
    current1A:     float = Field(..., description="Phase-A current (A)",   examples=[12.3])
    current1B:     float = Field(..., description="Phase-B current (A)",   examples=[11.9])
    current1C:     float = Field(..., description="Phase-C current (A)",   examples=[12.7])
    activePower1A: float = Field(..., description="Phase-A active power (W)", examples=[2850.0])
    activePower1B: float = Field(..., description="Phase-B active power (W)", examples=[2760.0])
    activePower1C: float = Field(..., description="Phase-C active power (W)", examples=[2930.0])
    frequency:     float = Field(..., description="Grid frequency (Hz)",   examples=[50.02])


class PredictRequest(BaseModel):
    """
    Payload for a single outage-prediction request.

    * **model**     — which model to use for inference.
    * **threshold** — decision boundary; lower = more sensitive to outages.
    * **rows**      — ordered telemetry rows (oldest → newest).
                      LSTM & GAN require ≥ `sequence_len` rows (default 12).
                      XGBoost & LightGBM need at least 1 row (uses the last one).
    """

    model: ModelName = Field(
        ...,
        description="Model to use for prediction",
        examples=["xgboost"],
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Decision threshold (overrides server default if provided)",
        examples=[0.30],
    )
    rows: list[TelemetryRow] = Field(
        ...,
        min_length=1,
        description="Chronologically ordered telemetry rows (oldest → newest)",
    )

    @model_validator(mode="after")
    def check_sequence_length(self) -> "PredictRequest":
        if self.model in (MODEL_LSTM, MODEL_GAN):
            if len(self.rows) < settings.sequence_len:
                raise ValueError(
                    f"Models 'lstm' and 'gan' require at least "
                    f"{settings.sequence_len} telemetry rows "
                    f"(received {len(self.rows)})."
                )
        return self


class PredictResponse(BaseModel):
    """Prediction result with full diagnostic detail."""

    model:       ModelName = Field(..., description="Model used for inference")
    probability: float     = Field(..., description="Predicted probability of outage in next 30 min")
    prediction:  int       = Field(..., description="Binary label: 1 = outage predicted, 0 = normal")
    threshold:   float     = Field(..., description="Decision threshold applied")
    latency_ms:  float     = Field(..., description="Inference latency in milliseconds")
    message:     str       = Field(..., description="Human-readable interpretation")


class ModelInfo(BaseModel):
    name: str
    description: str
    requires_sequence: bool
    min_rows: int


class ModelsResponse(BaseModel):
    available: list[ModelInfo]
    unavailable: list[str]
    sequence_len: int = settings.sequence_len


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    scaler_ready: bool
    feature_count: int


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_MODEL_DESCRIPTIONS = {
    MODEL_LSTM:     "Bidirectional LSTM (4 layers, 128→128→64→32 units)",
    MODEL_GAN:      "GAN — CNN discriminator classification head (LSTM generator trained adversarially)",
    MODEL_XGBOOST:  "XGBoost gradient boosted trees",
    MODEL_LIGHTGBM: "LightGBM gradient boosted trees",
}

_SEQUENCE_MODELS = {MODEL_LSTM, MODEL_GAN}


def _rows_to_dataframe(rows: list[TelemetryRow]) -> pd.DataFrame:
    """Convert validated Pydantic rows to a pandas DataFrame."""
    records = [r.model_dump() for r in rows]
    df = pd.DataFrame(records)
    df["bucket_5m"] = pd.to_datetime(df["bucket_5m"])
    df.sort_values(["site_id", "bucket_5m"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _predict_tabular(model_name: str, df: pd.DataFrame, threshold: float) -> tuple[float, int]:
    """Run inference with a sklearn-compatible (XGBoost / LightGBM) model."""
    X = preprocess_tabular(df, model_registry.feature_names, model_registry.scaler)
    model = model_registry.get(model_name)
    proba = float(model.predict_proba(X)[-1, 1])  # use last row's probability
    pred  = apply_threshold(proba, threshold)
    return proba, pred


def _predict_sequence(model_name: str, df: pd.DataFrame, threshold: float) -> tuple[float, int]:
    """Run inference with an LSTM or GAN model."""
    seq, ok = build_single_sequence(
        df,
        model_registry.feature_names,
        model_registry.scaler,
        settings.sequence_len,
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"After feature engineering and NaN removal, fewer than "
                f"{settings.sequence_len} rows remain. "
                f"Provide more historical rows (recommend ≥ {settings.sequence_len + 6})."
            ),
        )
    model = model_registry.get(model_name)
    raw = model.predict(seq, verbose=0)  # (1, 1) or (1,)
    proba = float(np.array(raw).flatten()[0])
    pred  = apply_threshold(proba, threshold)
    return proba, pred


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run outage prediction",
    description="""
Predict whether the power grid at a telecom site will go down in the **next 30 minutes**.

### Model options
| Model      | Type         | Min rows | Notes |
|------------|--------------|----------|-------|
| `xgboost`  | Tree ensemble | 12+      | Fast; trained on top-20 features |
| `lightgbm` | Tree ensemble | 12+      | Fast; same feature set as XGBoost |
| `lstm`     | Deep learning | 23+     | Bidirectional LSTM; best on sequential patterns |
| `gan`      | Deep learning | 23+     | CNN discriminator from adversarial training |

### Threshold
Lower threshold → higher recall (fewer missed outages, more false alarms).  
Default is **0.30** (tuned for telecom sites where missed outages are costly).

### Row ordering
Provide rows in **chronological order** (oldest → newest).
The last `sequence_len` rows are used.
""",
    tags=["Prediction"],
)
async def predict(body: PredictRequest) -> PredictResponse:
    # Check model availability
    if body.model not in model_registry.available_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{body.model}' is not loaded. "
                f"Available: {model_registry.available_models}"
            ),
        )

    threshold = body.threshold if body.threshold is not None else settings.default_threshold
    df = _rows_to_dataframe(body.rows)

    t0 = time.perf_counter()
    try:
        if body.model in _SEQUENCE_MODELS:
            proba, pred = _predict_sequence(body.model, df, threshold)
        else:
            proba, pred = _predict_tabular(body.model, df, threshold)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for model '%s'", body.model)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    message = (
        "⚡ Outage likely in the next 30 minutes — alert recommended."
        if pred == 1
        else "✅ No outage predicted in the next 30 minutes."
    )

    return PredictResponse(
        model=body.model,
        probability=round(proba, 6),
        prediction=pred,
        threshold=threshold,
        latency_ms=round(latency_ms, 2),
        message=message,
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models",
    description="Returns all models that are currently loaded and ready for inference.",
    tags=["Models"],
)
async def list_models() -> ModelsResponse:
    all_names = {"lstm", "gan", "xgboost", "lightgbm"}
    available = [
        ModelInfo(
            name=name,
            description=_MODEL_DESCRIPTIONS.get(name, ""),
            requires_sequence=name in _SEQUENCE_MODELS,
            min_rows=settings.sequence_len if name in _SEQUENCE_MODELS else 1,
        )
        for name in model_registry.available_models
    ]
    unavailable = sorted(all_names - set(model_registry.available_models))
    return ModelsResponse(available=available, unavailable=unavailable)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Liveness and readiness probe — returns 200 when the service is ready.",
    tags=["Health"],
)
async def health() -> HealthResponse:
    try:
        scaler_ready = model_registry.scaler is not None
    except RuntimeError:
        scaler_ready = False

    return HealthResponse(
        status="ready" if model_registry.is_ready else "loading",
        models_loaded=model_registry.available_models,
        scaler_ready=scaler_ready,
        feature_count=len(model_registry.feature_names),
    )
