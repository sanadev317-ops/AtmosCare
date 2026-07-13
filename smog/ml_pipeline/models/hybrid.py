# ============================================================================
# HYBRID ENSEMBLE MODULE
# ============================================================================
"""Hybrid ensemble combining GRU and SARIMA."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger

logger = get_logger(__name__)


class HybridPredictor:
    """Blend GRU and SARIMA predictions using weighted ensemble."""
    
    def __init__(self, config: Dict):
        """
        Initialize hybrid predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.hybrid_config = config.get("hybrid_model", {})
        
        self.gru_weight = self.hybrid_config.get("gru_weight", 0.7)
        self.sarima_weight = self.hybrid_config.get("sarima_weight", 0.3)
        self.method = self.hybrid_config.get("blending_method", "weighted_average")
        
        # Normalize weights
        total = self.gru_weight + self.sarima_weight
        self.gru_weight /= total
        self.sarima_weight /= total
        self.soft_floor = float(self.hybrid_config.get("soft_floor", 1e-3))
        
        logger.info(f"Hybrid initialized: GRU={self.gru_weight:.2f}, "
                   f"SARIMA={self.sarima_weight:.2f}, method={self.method}")
    
    def blend(
        self,
        gru_pred: np.ndarray,
        sarima_pred: np.ndarray,
        sarima_r2: Optional[float] = None,
        rmse_gru: Optional[float] = None,
        rmse_sarima: Optional[float] = None,
    ) -> np.ndarray:
        """
        Blend two predictions.

        Args:
            gru_pred: GRU predictions
            sarima_pred: SARIMA predictions
            sarima_r2: ignored; retained for API compatibility
            rmse_gru: Validation RMSE for GRU (optional)
            rmse_sarima: Validation RMSE for SARIMA (optional)

        Returns:
            Blended predictions
        """
        n = min(len(gru_pred), len(sarima_pred))
        gru_p = gru_pred[:n]
        sarima_p = sarima_pred[:n]

        self.gru_weight = 0.7
        self.sarima_weight = 0.3
        logger.info("Hybrid enforced weighted blend: GRU=0.700 SARIMAX=0.300")

        if self.method in {
            "weighted_average",
            "validation_weighted_average",
            "learned_weighted_average",
            "adaptive_gated_residual",
        }:
            blended = (self.gru_weight * gru_p + self.sarima_weight * sarima_p)
        elif self.method == "max":
            blended = np.maximum(gru_p, sarima_p)
        elif self.method == "min":
            blended = np.minimum(gru_p, sarima_p)
        elif self.method == "median":
            blended = np.median([gru_p, sarima_p], axis=0)
        else:
            logger.warning(f"Unknown blending method: {self.method}. Using weighted average.")
            blended = (self.gru_weight * gru_p + self.sarima_weight * sarima_p)

        return np.maximum(blended, self.soft_floor)
    
    def set_weights(self, gru_weight: float, sarima_weight: float):
        """Update blend weights."""
        total = gru_weight + sarima_weight
        self.gru_weight = gru_weight / total
        self.sarima_weight = sarima_weight / total
        
        logger.info(f"Weights updated: GRU={self.gru_weight:.2f}, "
                   f"SARIMA={self.sarima_weight:.2f}")


class AdaptiveFusionGater:
    """Lightweight dynamic gate for SARIMAX residual fusion."""

    def __init__(self, config: Dict):
        self.config = config
        self.hybrid_config = config.get("hybrid_model", {})
        self.gate_method = self.hybrid_config.get("gate_method", "logistic_regression")
        self.gate_threshold = float(self.hybrid_config.get("gate_threshold", 0.5))
        self.random_state = int(config.get("reproducibility", {}).get("seed", 42))

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self._fitted = False
        self._is_constant = False
        self._constant_weight = self.gate_threshold
        self._feature_count = 0

    @staticmethod
    def _to_matrix(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        if isinstance(X, pd.Series):
            return X.to_numpy(dtype=float).reshape(-1, 1)
        return np.asarray(X, dtype=float)

    def fit(self, X, y) -> "AdaptiveFusionGater":
        X_mat = self._to_matrix(X)
        y_arr = np.asarray(y).astype(int).ravel()

        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(-1, 1)

        self._feature_count = X_mat.shape[1]
        if len(y_arr) == 0:
            self._constant_weight = self.gate_threshold
            self._fitted = True
            self._is_constant = True
            return self

        unique = np.unique(y_arr)
        if len(unique) < 2:
            self._constant_weight = float(np.clip(np.mean(y_arr), 0.0, 1.0))
            self._fitted = True
            self._is_constant = True
            logger.warning(
                "Adaptive gate received a single class. Falling back to constant weight "
                f"{self._constant_weight:.3f}"
            )
            return self

        X_scaled = self.scaler.fit_transform(X_mat)
        self.model.fit(X_scaled, y_arr)
        self._fitted = True
        self._is_constant = False

        logger.info(
            "Adaptive gate fitted with logistic regression on %d samples and %d features",
            len(y_arr),
            self._feature_count,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_mat = self._to_matrix(X)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(-1, 1)

        if not self._fitted:
            return np.full(len(X_mat), self._constant_weight, dtype=float)

        if self._is_constant or self._feature_count == 0:
            return np.full(len(X_mat), self._constant_weight, dtype=float)

        X_scaled = self.scaler.transform(X_mat)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return np.clip(proba, 0.0, 1.0)

    def predict_weights(self, X) -> np.ndarray:
        weight_gru = self.predict_proba(X)
        weight_sarima = 1.0 - weight_gru
        return np.column_stack([weight_sarima, weight_gru])

    def combine(self, sarimax_pred, gru_residual_pred, X) -> np.ndarray:
        """Combine baseline and residual predictions using adaptive gate weights."""
        sarimax_pred = np.asarray(sarimax_pred, dtype=float)
        gru_residual_pred = np.asarray(gru_residual_pred, dtype=float)
        n = min(len(sarimax_pred), len(gru_residual_pred))
        weights = self.predict_proba(X)[:n]
        final_pred = sarimax_pred[:n] + weights * gru_residual_pred[:n]
        return np.maximum(final_pred, float(self.hybrid_config.get("soft_floor", 1e-3)))
