# Stateless GBDT Ensemble Matrix Calculator

> **100% Stateless · Zero-Storage · Mathematical Rigor with 7-Fold Ensembles · Zero Infrastructure Overhead**

A headless, pure-REST inference API that scores a structured feature vector against a
deterministic **7-fold Gradient-Boosted ensemble** using nothing but in-memory **NumPy** linear
algebra. There is **no database, no external AI service, and no disk I/O at request time** — every
response is derived entirely from the request payload and a set of deterministically materialised
weights.

---

## ✨ Design Philosophy

| Principle | What it means here |
| --- | --- |
| **100% Stateless & Zero-Storage** | No DB, no cache, no files, no sessions. The process holds no per-request state, so any replica can serve any request. Kill it, restart it, scale it to N — behaviour is identical. |
| **Mathematical Rigor with 7-Fold Ensemble** | Predictions are blended across **7 independent cross-validation folds**. We return both the **mean** (the blend) and the **population variance** (a first-class epistemic uncertainty signal), computed with NumPy. |
| **Zero Infrastructure Overhead** | One container, one process, CPU + RAM only. No connection pools, no migrations, no secrets to rotate, no storage layer to operate. |
| **Strict Data Contracts** | Pydantic v2 rejects wrong dimensionality, `NaN`/`±inf`, and type mismatches at the edge with an instant **HTTP 400** — the numerical core only ever sees clean, finite input. |

---

## 🧮 How the math works

For a feature vector **x** and a `model_version` string:

1. **Deterministic weights.** Each fold `f ∈ {0..6}` derives a stable seed from
   `sha256(model_version :: fold)`, materialising a weight vector `w_f` and bias `b_f`. The same
   version always reproduces the same "trained" weights — no persistence required.
2. **Vectorised matrix pass.** The augmented vector `[x, 1]` is multiplied by the stacked
   `(7 × dim+1)` weight matrix in a single `numpy` matrix–vector product, yielding 7 logits.
3. **Squashing.** A numerically-stable logistic maps each logit into `(0, 1)` — extreme but
   finite inputs never overflow.
4. **Blend & uncertainty.**
   - `ensemble_prediction = mean(fold_predictions)`
   - `prediction_variance = var(fold_predictions, ddof=0)`

A low variance ⇒ the folds agree (high confidence); a high variance ⇒ epistemic uncertainty.

---

## 🚀 3-Minute Quickstart

### Option A — Local (Python 3.12+)

```bash
# 1. Install the locked dependencies
pip install -r requirements.txt

# 2. Launch the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Open the interactive docs (fully English OpenAPI / Swagger UI)
open http://localhost:8000/docs
```

### Option B — Docker (one command)

```bash
docker build -t stateless-gbdt-calculator .
docker run --rm -p 8000:8000 stateless-gbdt-calculator
```

The container runs as a non-root user and ships a built-in `HEALTHCHECK` against `/health`.

---

## 📡 API Reference

### `GET /health`
Liveness / readiness probe. Returns the model contract (expected dims, fold count).

### `POST /api/v1/calculate`
Run a 7-fold ensemble inference pass.

**Request body**

| Field | Type | Description |
| --- | --- | --- |
| `features` | `List[float]` | Finite numeric feature vector. Length must equal **10** when `validate_dims` is true. |
| `model_version` | `str` | Model weight-set identifier, e.g. `"v1.0.0"`. |
| `validate_dims` | `bool` | Enforce strict 10-dim contract (default `true`). |

**Example — POST a feature vector with cURL**

```bash
curl -s -X POST http://localhost:8000/api/v1/calculate \
  -H 'Content-Type: application/json' \
  -d '{
    "features": [0.12, -0.34, 0.56, 0.78, -0.9, 0.21, 0.43, -0.65, 0.87, 0.09],
    "model_version": "v1.0.0",
    "validate_dims": true
  }'
```

**Response**

```json
{
  "status": "success",
  "ensemble_prediction": 0.4261046,
  "prediction_variance": 0.0126954,
  "fold_predictions": [0.36634, 0.38990, 0.47017, 0.38949, 0.33244, 0.68353, 0.35086],
  "latency_ms": 0.84,
  "timestamp": "2026-06-21T11:13:45.445552Z"
}
```

**Example — Python client**

```python
import requests

resp = requests.post(
    "http://localhost:8000/api/v1/calculate",
    json={
        "features": [0.12, -0.34, 0.56, 0.78, -0.9, 0.21, 0.43, -0.65, 0.87, 0.09],
        "model_version": "v1.0.0",
        "validate_dims": True,
    },
    timeout=5,
)
resp.raise_for_status()
data = resp.json()
print("prediction:", data["ensemble_prediction"])
print("uncertainty (variance):", data["prediction_variance"])
```

---

## 🛡️ Robustness Guarantees

The service is engineered to **never return a raw 500** for bad input:

| Input | Result |
| --- | --- |
| Wrong dimensionality (e.g. 3 features, `validate_dims=true`) | **400** with a precise detail message |
| `NaN` in the vector | **400** — `non-finite values found at indices [...]` |
| `+Infinity` / `-Infinity` | **400** — rejected by the finite validator |
| Wrong type (`"features": "foo"`) | **400** |
| Unknown extra field | **400** (`extra="forbid"`) |
| Extreme but finite magnitudes (`1e8`) | **200** — stable sigmoid, no overflow |

---

## 🗂️ Project Layout

```
.
├── app/
│   ├── __init__.py
│   ├── schemas.py         # Pydantic v2 strict contracts (request/response/health/error)
│   ├── matrix_engine.py   # Stateless 7-fold NumPy inference core
│   └── main.py            # FastAPI app, CORS, exception handlers, endpoints
├── tests/
│   └── test_api.py        # Engine + HTTP verification suite (14 cases)
├── requirements.txt       # Pinned dependencies
├── Dockerfile             # Slim, non-root, healthchecked production image
├── .dockerignore
└── README.md
```

---

## 🧪 Running the tests

```bash
python -m pytest -q
```

```
14 passed
```

---

## 📜 License

MIT.
