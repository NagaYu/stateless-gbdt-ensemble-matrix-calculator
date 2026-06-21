"""Stateless 7-fold ensemble inference core — 100% in-memory, zero I/O.

Design philosophy
-----------------
This module is the mathematical heart of the calculator. It performs **no** network calls,
opens **no** sockets, touches **no** database and reads **no** files at request time. Every
prediction is produced purely from CPU + RAM using ``numpy`` linear algebra.

The model emulates a Gradient-Boosted Decision Tree ensemble trained with 7-fold cross
validation. Each of the 7 folds is represented by an independent, deterministically generated
weight matrix (a compact "mock decision layer"). At inference time the feature vector is pushed
through all 7 folds, producing 7 independent scalar predictions. We then report:

* ``ensemble_prediction`` — the mean of the 7 fold predictions (the blend), and
* ``prediction_variance`` — the population variance of the 7 fold predictions (the uncertainty).

Determinism: fold weights are seeded from a hash of ``model_version``. The same version always
materialises the same weights, so the engine behaves like a fixed trained artifact without ever
persisting anything to disk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np

from app.schemas import NUM_FOLDS


@dataclass(frozen=True)
class EnsembleResult:
    """Immutable container for a completed ensemble inference pass."""

    ensemble_prediction: float
    prediction_variance: float
    fold_predictions: List[float]


def _stable_seed(model_version: str, fold_index: int) -> int:
    """Derive a stable 32-bit seed from the model version string and fold index.

    Using a cryptographic digest (rather than Python's salted ``hash``) guarantees the seed is
    reproducible across processes and interpreter restarts — essential for a stateless service
    where every replica must agree on the "trained" weights without sharing storage.
    """
    digest = hashlib.sha256(f"{model_version}::fold-{fold_index}".encode("utf-8")).digest()
    # Take the first 4 bytes as an unsigned 32-bit integer (numpy's legacy seed range).
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic squashing into (0, 1).

    The piecewise formulation avoids ``exp`` overflow for large-magnitude logits, so extreme
    (but finite) feature values never raise ``OverflowError`` or emit warnings.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    negative = ~positive
    # For x >= 0: 1 / (1 + exp(-x))
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    # For x < 0: exp(x) / (1 + exp(x)) — keeps exponent argument negative.
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)
    return out


class StatelessEnsembleEngine:
    """In-memory 7-fold ensemble predictor.

    Parameters
    ----------
    num_folds:
        Number of independent cross-validation folds to blend. Defaults to the project-wide
        :data:`app.schemas.NUM_FOLDS` (7).
    """

    def __init__(self, num_folds: int = NUM_FOLDS) -> None:
        if num_folds < 2:
            raise ValueError("num_folds must be >= 2 for a meaningful ensemble.")
        self.num_folds = num_folds

    # ------------------------------------------------------------------
    # Weight materialisation
    # ------------------------------------------------------------------
    def _build_fold_weights(self, model_version: str, dim: int) -> List[np.ndarray]:
        """Materialise one weight vector + bias per fold, deterministically.

        Each fold ``f`` gets:
          * a weight vector ``w_f`` of length ``dim`` drawn from a fold-seeded normal, and
          * a scalar bias ``b_f``.

        The slight per-fold variation (different seeds) is what produces *diversity* across the
        ensemble — without diversity the variance would collapse to zero and the uncertainty
        signal would be meaningless.
        """
        weights: List[np.ndarray] = []
        for fold_index in range(self.num_folds):
            rng = np.random.default_rng(_stable_seed(model_version, fold_index))
            # Weights scaled by 1/sqrt(dim) to keep logits in a sane range regardless of dim.
            w = rng.normal(loc=0.0, scale=1.0 / np.sqrt(dim), size=dim)
            b = rng.normal(loc=0.0, scale=0.25)
            # Pack weights + bias into a single (dim + 1,) vector; last element is the bias.
            weights.append(np.concatenate([w, np.asarray([b], dtype=np.float64)]))
        return weights

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, features: List[float], model_version: str) -> EnsembleResult:
        """Run a full 7-fold ensemble inference pass.

        Parameters
        ----------
        features:
            Finite numeric feature vector. The caller (schema layer) guarantees finiteness;
            this method defensively re-checks to remain safe even if invoked directly.
        model_version:
            Determines which deterministic weight set is materialised.

        Returns
        -------
        EnsembleResult
            The blended prediction, the per-fold variance and the raw fold predictions.

        Raises
        ------
        ValueError
            If the feature vector is empty or contains non-finite values.
        """
        vector = np.asarray(features, dtype=np.float64)

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("features must be a non-empty 1-D numeric vector.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("features must contain only finite numbers (no NaN/inf).")

        dim = vector.size
        fold_weights = self._build_fold_weights(model_version, dim)

        # Augment the feature vector with a constant 1.0 to absorb the per-fold bias term,
        # turning each fold prediction into a single dot product: logit = [features, 1] · [w, b].
        augmented = np.concatenate([vector, np.asarray([1.0], dtype=np.float64)])

        # Stack all fold weight vectors into a (num_folds, dim + 1) matrix and compute every
        # fold's logit in one vectorised matrix-vector product. This is the core "matrix" step.
        weight_matrix = np.vstack(fold_weights)
        logits = weight_matrix @ augmented  # shape: (num_folds,)

        # Squash each logit into a probability in (0, 1).
        fold_predictions = np.asarray(_sigmoid(logits), dtype=np.float64)

        # Blend (mean) and uncertainty (population variance, ddof=0).
        ensemble_prediction = float(np.mean(fold_predictions))
        prediction_variance = float(np.var(fold_predictions, ddof=0))

        # Clamp the blend into [0, 1] to satisfy the response contract against float drift.
        ensemble_prediction = float(np.clip(ensemble_prediction, 0.0, 1.0))
        prediction_variance = float(max(prediction_variance, 0.0))

        return EnsembleResult(
            ensemble_prediction=ensemble_prediction,
            prediction_variance=prediction_variance,
            fold_predictions=[float(p) for p in fold_predictions],
        )


# Module-level singleton: the engine holds no per-request state, so a single instance is
# safely shared across all requests/threads (it only reads from immutable inputs).
ENGINE = StatelessEnsembleEngine()
