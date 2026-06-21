"""Mathematically strict data contracts for the Stateless GBDT Ensemble Matrix Calculator.

All request/response shapes are defined here with Pydantic v2. The validation layer is the
first line of defense: malformed dimensionality, non-finite values (``NaN`` / ``±inf``) and
type mismatches are rejected *before* they ever reach the numerical inference core, so the
matrix engine can assume it always receives a clean, finite, correctly-sized feature vector.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Global model contract constants
# ---------------------------------------------------------------------------
# The model was trained on a 10-dimensional structured numeric feature space.
# This is a hard contract: when ``validate_dims`` is enabled the request is rejected
# unless exactly EXPECTED_FEATURE_DIM features are supplied.
EXPECTED_FEATURE_DIM: int = 10

# Number of independent cross-validation folds blended at inference time.
NUM_FOLDS: int = 7


class CalcRequest(BaseModel):
    """Inbound inference request.

    Attributes
    ----------
    features:
        The structured numeric feature vector fed into the ensemble. Every element must be a
        finite real number (no ``NaN``, no ``±inf``). When ``validate_dims`` is true the vector
        length must equal :data:`EXPECTED_FEATURE_DIM`.
    model_version:
        Identifier of the model weight set to evaluate against (e.g. ``"v1.0.0"``). The engine
        is stateless and seeds its deterministic mock weights from this string, so two requests
        with the same version + features always yield identical predictions.
    validate_dims:
        When true (default) enforce strict dimensionality checking against the trained contract.
        When false the engine adapts to the supplied length (useful for experimentation), while
        still rejecting non-finite values.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        # ``model_`` is a protected namespace in Pydantic v2; we intentionally expose
        # ``model_version`` as part of the public API contract, so disable the guard.
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "features": [0.12, -0.34, 0.56, 0.78, -0.9, 0.21, 0.43, -0.65, 0.87, 0.09],
                "model_version": "v1.0.0",
                "validate_dims": True,
            }
        },
    )

    features: List[float] = Field(
        ...,
        min_length=1,
        description=(
            "Structured numeric feature vector. Each value must be finite (no NaN/inf). "
            f"When validate_dims is true the length must equal {EXPECTED_FEATURE_DIM}."
        ),
    )
    model_version: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Model weight-set identifier, e.g. 'v1.0.0'.",
    )
    validate_dims: bool = Field(
        default=True,
        description="Enforce strict dimensionality checking against the trained feature contract.",
    )

    @field_validator("features")
    @classmethod
    def _reject_non_finite(cls, value: List[float]) -> List[float]:
        """Reject vectors containing ``NaN``, ``+inf`` or ``-inf``.

        This guarantees the numerical core never has to defend against non-finite inputs,
        preventing silent ``NaN`` propagation through the ensemble arithmetic.
        """
        bad_positions = [i for i, v in enumerate(value) if not math.isfinite(v)]
        if bad_positions:
            raise ValueError(
                "features must contain only finite numbers; "
                f"non-finite values found at indices {bad_positions}"
            )
        return value

    @model_validator(mode="after")
    def _enforce_dimensionality(self) -> "CalcRequest":
        """Enforce the trained dimensionality contract when requested."""
        if self.validate_dims and len(self.features) != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"feature dimensionality mismatch: expected {EXPECTED_FEATURE_DIM} "
                f"features but received {len(self.features)} (set validate_dims=false to bypass)"
            )
        return self


class CalcResponse(BaseModel):
    """Outbound inference result.

    Attributes
    ----------
    status:
        Always ``"success"`` for a 200 response.
    ensemble_prediction:
        The blended (mean) prediction across all :data:`NUM_FOLDS` folds, squashed into the
        ``[0.0, 1.0]`` probability range.
    prediction_variance:
        Population variance of the per-fold predictions. A small number indicates the folds
        agree (high confidence); a large number indicates epistemic uncertainty.
    fold_predictions:
        The raw per-fold predictions used to compute the blend, exposed for transparency and
        downstream uncertainty analysis.
    latency_ms:
        Server-side processing latency in milliseconds.
    timestamp:
        UTC timestamp marking when the response was produced.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "success",
                "ensemble_prediction": 0.7421,
                "prediction_variance": 0.0034,
                "fold_predictions": [0.71, 0.75, 0.73, 0.74, 0.76, 0.72, 0.73],
                "latency_ms": 0.83,
                "timestamp": "2026-06-21T12:00:00+00:00",
            }
        },
    )

    status: str = Field(default="success", description="Result status, always 'success' on 200.")
    ensemble_prediction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean blended probability across all folds, in [0.0, 1.0].",
    )
    prediction_variance: float = Field(
        ...,
        ge=0.0,
        description="Population variance of the per-fold predictions (model uncertainty).",
    )
    fold_predictions: List[float] = Field(
        ...,
        description="Raw per-fold prediction values used to build the ensemble.",
    )
    latency_ms: float = Field(..., ge=0.0, description="Processing latency in milliseconds.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of response generation.",
    )


class HealthResponse(BaseModel):
    """System health probe payload for liveness/readiness checks."""

    status: str = Field(default="ok", description="'ok' when the service is live.")
    service: str = Field(default="stateless-gbdt-ensemble-matrix-calculator")
    expected_feature_dim: int = Field(default=EXPECTED_FEATURE_DIM)
    num_folds: int = Field(default=NUM_FOLDS)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for 4xx/5xx conditions."""

    status: str = Field(default="error")
    detail: str = Field(..., description="Human-readable description of what went wrong.")
