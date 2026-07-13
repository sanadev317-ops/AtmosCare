"""Hybrid stacking and lightweight calibration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from ..utils import get_logger

logger = get_logger(__name__)


def _to_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def align_predictions(
    gru_pred: np.ndarray,
    sarima_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    gru = _to_1d(gru_pred)
    sarima = _to_1d(sarima_pred)
    arrays = [gru, sarima]
    if y_true is not None:
        arrays.append(_to_1d(y_true))

    n = min(len(arr) for arr in arrays)
    if n == 0:
        raise ValueError(
            "No overlapping samples available for alignment. "
            "Ensure GRU, SARIMA, and target arrays are non-empty."
        )
    gru = gru[:n]
    sarima = sarima[:n]
    if y_true is None:
        return gru, sarima, None
    return gru, sarima, _to_1d(y_true)[:n]


def search_best_alpha(
    gru_pred: np.ndarray,
    sarima_pred: np.ndarray,
    y_true: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    grid = grid if grid is not None else np.linspace(0.0, 1.0, 101)
    gru, sarima, y = align_predictions(gru_pred, sarima_pred, y_true)

    best_alpha = 0.5
    best_rmse = float("inf")
    curve = []
    for alpha in grid:
        hybrid = alpha * gru + (1.0 - alpha) * sarima
        rmse = float(np.sqrt(mean_squared_error(y, hybrid)))
        curve.append({"alpha": float(alpha), "rmse": rmse})
        if rmse < best_rmse:
            best_alpha = float(alpha)
            best_rmse = rmse

    return {"best_alpha": best_alpha, "best_rmse": best_rmse, "curve": curve}


def compute_confidence(
    prediction: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    rolling_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    pred = _to_1d(prediction)
    if lower_bound is not None and upper_bound is not None:
        lower = _to_1d(lower_bound)
        upper = _to_1d(upper_bound)
        width = np.abs(upper - lower)
        scale = np.maximum(np.abs(pred), 1.0)
        return np.clip(1.0 / (1.0 + width / scale), 0.0, 1.0)
    if rolling_std is not None:
        std = np.maximum(_to_1d(rolling_std), 1e-6)
        return np.clip(1.0 / (1.0 + std), 0.0, 1.0)
    return np.full_like(pred, 0.5, dtype=float)


class StackingModel:
    """Simple stacking regressor for GRU + SARIMA outputs."""

    def __init__(self, model_type: str = "ridge"):
        self.model_type = model_type
        self.model = Ridge(alpha=1.0) if model_type.lower() == "ridge" else LinearRegression()
        self.is_fitted = False
        self.validation_metrics: Dict[str, float] = {}

    def fit(self, X_meta: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        X = np.asarray(X_meta, dtype=float)
        y = _to_1d(y_true)
        if len(X) != len(y):
            n = min(len(X), len(y))
            X = X[:n]
            y = y[:n]

        if len(X) < 5:
            raise ValueError("Not enough samples to fit stacking model.")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        val_pred = self.model.predict(X_val)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
            "mae": float(mean_absolute_error(y_val, val_pred)),
        }
        self.validation_metrics = metrics
        return metrics

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        X = np.asarray(X_meta, dtype=float)
        if not self.is_fitted:
            raise RuntimeError("StackingModel is not fitted.")
        return np.asarray(self.model.predict(X), dtype=float)

    def fit_or_refit(self, X_meta: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        metrics = self.fit(X_meta, y_true)
        return {"metrics": metrics, "model_type": self.model_type}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "is_fitted": self.is_fitted, "model_type": self.model_type}, path)
        return path

    @classmethod
    def load(cls, path: str) -> "StackingModel":
        state = joblib.load(path)
        obj = cls(state.get("model_type", "ridge"))
        obj.model = state["model"]
        obj.is_fitted = bool(state.get("is_fitted", True))
        return obj
