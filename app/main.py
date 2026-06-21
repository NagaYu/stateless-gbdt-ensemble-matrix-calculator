"""Stateless GBDT Ensemble Matrix Calculator — pure REST inference API.

A headless (no UI), 100% stateless inference service. It exposes a single calculation endpoint
plus a health probe. There is no database, no external AI provider and no persistent state: every
response is computed in-memory from the request payload alone, which makes the service trivially
horizontally scalable and free of any storage-layer operational overhead.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.matrix_engine import ENGINE
from app.schemas import (
    CalcRequest,
    CalcResponse,
    ErrorResponse,
    HealthResponse,
)

app = FastAPI(
    title="Stateless GBDT Ensemble Matrix Calculator",
    version="1.0.0",
    description=(
        "A 100% stateless, zero-storage inference API. It evaluates a structured feature vector "
        "against a deterministic 7-fold Gradient-Boosted ensemble using nothing but in-memory "
        "NumPy linear algebra — no database, no external AI service, no disk I/O at request time. "
        "Returns the blended ensemble prediction together with the per-fold variance as a rigorous "
        "uncertainty estimate."
    ),
    contact={"name": "Stateless GBDT Ensemble Matrix Calculator"},
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# CORS — permit local cross-origin clients to call the API safely.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers — guarantee structured JSON errors, never raw 500s.
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert Pydantic validation failures into clean HTTP 400 responses.

    Bad dimensionality, non-finite feature values (NaN/inf) and type mismatches all surface here
    as a 400 Bad Request with a precise, machine-readable detail string — the request never
    reaches the inference core.
    """
    messages = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        messages.append(f"{location or 'request'}: {err.get('msg', 'invalid input')}")
    payload = ErrorResponse(detail="; ".join(messages) or "validation error")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=payload.model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort safety net: surface unexpected errors as structured 500s, never a raw stack.

    In practice the validation layer and the finite-safe numerical core mean this should not be
    reachable, but it guarantees the server stays up and responds with a clean envelope no matter
    what.
    """
    payload = ErrorResponse(detail=f"internal inference error: {type(exc).__name__}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness / readiness probe",
)
async def health() -> HealthResponse:
    """Report service health for orchestration probes (Docker/Kubernetes).

    Returns a lightweight payload describing the model contract (expected feature dimensionality
    and fold count). Because the service is stateless, liveness and readiness are equivalent: if
    the process answers, it is ready to serve traffic.
    """
    return HealthResponse()


@app.post(
    "/api/v1/calculate",
    response_model=CalcResponse,
    tags=["inference"],
    summary="Run a 7-fold ensemble inference pass",
    responses={
        400: {"model": ErrorResponse, "description": "Validation error (bad dims / non-finite values)"},
        500: {"model": ErrorResponse, "description": "Unexpected internal inference error"},
    },
)
async def calculate(payload: CalcRequest) -> CalcResponse:
    """Evaluate a feature vector against the stateless 7-fold ensemble.

    The request body is validated strictly: the feature vector must contain only finite numbers,
    and (when ``validate_dims`` is true) must match the trained dimensionality. The validated
    vector is then pushed through all 7 deterministic folds in a single vectorised matrix
    operation. The response carries the blended ensemble prediction, the per-fold variance
    (an uncertainty estimate), the raw fold predictions, and the processing latency.

    This call is fully self-contained: no database, no external service, no disk access.
    """
    start = time.perf_counter()

    result = ENGINE.predict(features=payload.features, model_version=payload.model_version)

    latency_ms = (time.perf_counter() - start) * 1000.0

    return CalcResponse(
        status="success",
        ensemble_prediction=result.ensemble_prediction,
        prediction_variance=result.prediction_variance,
        fold_predictions=result.fold_predictions,
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc),
    )
