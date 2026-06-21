"""End-to-end verification suite for the Stateless GBDT Ensemble Matrix Calculator.

Run with:  python -m pytest -q   (or simply execute this file: python tests/test_api.py)

The suite exercises the schema validation layer, the numerical engine and the HTTP surface,
including the adversarial NaN / inf / wrong-dimension cases that must never crash the server.
"""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from app.main import app
from app.matrix_engine import ENGINE
from app.schemas import EXPECTED_FEATURE_DIM, NUM_FOLDS

client = TestClient(app)

VALID_FEATURES = [0.12, -0.34, 0.56, 0.78, -0.9, 0.21, 0.43, -0.65, 0.87, 0.09]


# ---------------------------------------------------------------------------
# Engine-level checks
# ---------------------------------------------------------------------------
def test_engine_determinism():
    r1 = ENGINE.predict(VALID_FEATURES, "v1.0.0")
    r2 = ENGINE.predict(VALID_FEATURES, "v1.0.0")
    assert r1.fold_predictions == r2.fold_predictions
    assert r1.ensemble_prediction == r2.ensemble_prediction


def test_engine_fold_count_and_ranges():
    r = ENGINE.predict(VALID_FEATURES, "v1.0.0")
    assert len(r.fold_predictions) == NUM_FOLDS
    assert all(0.0 <= p <= 1.0 for p in r.fold_predictions)
    assert 0.0 <= r.ensemble_prediction <= 1.0
    assert r.prediction_variance >= 0.0


def test_engine_variance_matches_numpy():
    import numpy as np

    r = ENGINE.predict(VALID_FEATURES, "v1.0.0")
    expected_mean = float(np.mean(r.fold_predictions))
    expected_var = float(np.var(r.fold_predictions, ddof=0))
    assert math.isclose(r.ensemble_prediction, expected_mean, rel_tol=1e-9)
    assert math.isclose(r.prediction_variance, expected_var, rel_tol=1e-9)


def test_engine_extreme_but_finite_values_no_overflow():
    extreme = [1e6, -1e6, 1e8, -1e8, 1e6, -1e6, 1e8, -1e8, 0.0, 1.0]
    r = ENGINE.predict(extreme, "v1.0.0")
    assert all(math.isfinite(p) for p in r.fold_predictions)
    assert math.isfinite(r.ensemble_prediction)


def test_engine_version_changes_weights():
    r1 = ENGINE.predict(VALID_FEATURES, "v1.0.0")
    r2 = ENGINE.predict(VALID_FEATURES, "v2.0.0")
    assert r1.fold_predictions != r2.fold_predictions


# ---------------------------------------------------------------------------
# HTTP-level checks
# ---------------------------------------------------------------------------
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["expected_feature_dim"] == EXPECTED_FEATURE_DIM
    assert body["num_folds"] == NUM_FOLDS


def test_calculate_success():
    resp = client.post(
        "/api/v1/calculate",
        json={"features": VALID_FEATURES, "model_version": "v1.0.0", "validate_dims": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert 0.0 <= body["ensemble_prediction"] <= 1.0
    assert body["prediction_variance"] >= 0.0
    assert len(body["fold_predictions"]) == NUM_FOLDS
    assert body["latency_ms"] >= 0.0
    assert "timestamp" in body


def test_calculate_wrong_dimension_returns_400():
    resp = client.post(
        "/api/v1/calculate",
        json={"features": [0.1, 0.2, 0.3], "model_version": "v1.0.0", "validate_dims": True},
    )
    assert resp.status_code == 400
    assert "dimensionality" in resp.json()["detail"].lower()


def test_calculate_wrong_dimension_bypassed():
    resp = client.post(
        "/api/v1/calculate",
        json={"features": [0.1, 0.2, 0.3], "model_version": "v1.0.0", "validate_dims": False},
    )
    assert resp.status_code == 200
    assert len(resp.json()["fold_predictions"]) == NUM_FOLDS


def test_calculate_nan_returns_400():
    # JSON has no NaN literal; Pydantic also rejects the string. Use Python float nan via the
    # engine-bypassing path is not possible over HTTP, so we send the JSON token "NaN" which
    # FastAPI's parser accepts as float('nan') — and our finite validator must reject it.
    resp = client.post(
        "/api/v1/calculate",
        content='{"features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,NaN], "model_version": "v1.0.0", "validate_dims": true}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "finite" in resp.json()["detail"].lower()


def test_calculate_inf_returns_400():
    resp = client.post(
        "/api/v1/calculate",
        content='{"features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,Infinity], "model_version": "v1.0.0", "validate_dims": true}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "finite" in resp.json()["detail"].lower()


def test_calculate_missing_field_returns_400():
    resp = client.post("/api/v1/calculate", json={"features": VALID_FEATURES})
    assert resp.status_code == 400


def test_calculate_extra_field_forbidden():
    resp = client.post(
        "/api/v1/calculate",
        json={
            "features": VALID_FEATURES,
            "model_version": "v1.0.0",
            "validate_dims": True,
            "rogue": 123,
        },
    )
    assert resp.status_code == 400


def test_calculate_wrong_type_returns_400():
    resp = client.post(
        "/api/v1/calculate",
        json={"features": "not-a-list", "model_version": "v1.0.0", "validate_dims": True},
    )
    assert resp.status_code == 400


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
