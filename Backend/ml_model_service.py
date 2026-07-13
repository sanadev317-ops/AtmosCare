"""
ML Model Service - UNIFIED ENSEMBLE (GRU + SARIMA Stacking)

Provides AI-driven AQI predictions using:
1. GRU Seq2Seq model (primary for sequence prediction)
2. SARIMA model (secondary for seasonal patterns)
3. Stacking meta-model (Ridge regression combining both)

Singleton service with model caching and error handling.
"""

import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

from .pipeline_service import AirQualityInferenceService, _patch_pandas_stringarray_compat

logger = logging.getLogger(__name__)


class MLModelService:
    """Unified ML model service with ensemble stacking."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModelService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.gru = None
            self.sarima = None
            self.stacking_model = None
            self.scaler_X = None
            self.scaler_y = None
            self.pca = None
            self.is_loaded = False
            self.model_version = "unknown"
            self.pipeline = AirQualityInferenceService()
            self.initialized = True
            logger.info("MLModelService initialized")

    def _find_model_file(self, pattern: str) -> Path:
        """Find model file by pattern."""
        search_paths = [
            Path("."),
            Path("artifacts/models"),
            Path("smog/artifacts/models"),
            Path("../artifacts/models"),
            Path("../smog/artifacts/models"),
            Path("../../artifacts/models"),
            Path("../../smog/artifacts/models"),
            Path("Backend"),
        ]
        
        for search_path in search_paths:
            matches = list(search_path.glob(pattern))
            if matches:
                return matches[0]
        
        return None

    def _load_model_components(self):
        """Load all model components from disk."""
        try:
            # 1. Load GRU model
            gru_path = self._find_model_file("gru_aqi_model.keras")
            if gru_path and gru_path.exists():
                self.gru = tf.keras.models.load_model(gru_path)
                logger.info(f"✓ GRU model loaded from {gru_path}")
            else:
                logger.warning("GRU model not found, will attempt fallback")
            
            # 2. Load SARIMA model (if available)
            sarima_path = self._find_model_file("**/sarima*.pkl")
            if sarima_path and sarima_path.exists():
                try:
                    _patch_pandas_stringarray_compat()
                    self.sarima = joblib.load(sarima_path)
                    logger.info(f"✓ SARIMA model loaded from {sarima_path}")
                except Exception as e:
                    logger.warning(f"SARIMA load failed: {e}")
            
            # 3. Load Stacking meta-model (PREFERRED over GRU alone)
            stacking_path = self._find_model_file("**/stacking_model.pkl")
            if stacking_path and stacking_path.exists():
                try:
                    self.stacking_model = joblib.load(stacking_path)
                    self.model_version = "stacking_v1"
                    logger.info(f"✓ Stacking meta-model loaded from {stacking_path}")
                except Exception as e:
                    logger.warning(f"Stacking model load failed: {e}")
            
            # 4. Load scalers
            scaler_X_path = self._find_model_file("scaler_X.pkl")
            scaler_y_path = self._find_model_file("scaler_y.pkl")
            
            if scaler_X_path and scaler_X_path.exists():
                self.scaler_X = joblib.load(scaler_X_path)
                logger.debug("✓ X scaler loaded")
            
            if scaler_y_path and scaler_y_path.exists():
                self.scaler_y = joblib.load(scaler_y_path)
                logger.debug("✓ Y scaler loaded")
            
            # 5. Load PCA (optional)
            pca_path = self._find_model_file("pca_model.pkl")
            if pca_path and pca_path.exists():
                try:
                    self.pca = joblib.load(pca_path)
                    logger.debug("✓ PCA model loaded")
                except:
                    logger.debug("PCA not available (optional)")
            
            # Get feature order for consistency
            self.runtime_feature_order = list(
                getattr(
                    self.scaler_X,
                    "feature_names_in_",
                    ["pm25", "pm10", "temperature", "wind_speed", "hour", "day", "month", "weekday"],
                )
            ) if self.scaler_X else None
            
            logger.info(f"Model version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Model component loading error: {e}")
            raise

    def load_model(self) -> bool:
        """
        Load all available models.
        
        Returns:
            bool: True if at least one model loaded successfully
        """
        try:
            self._load_model_components()
            pipeline_status = self.pipeline.load()
            self.gru = self.pipeline.gru or self.gru
            self.sarima = self.pipeline.sarima_results or self.sarima
            self.stacking_model = self.pipeline.stacking_model or self.stacking_model
            self.scaler_X = self.pipeline.gru_scaler_x or self.scaler_X
            self.scaler_y = self.pipeline.gru_scaler_y or self.scaler_y
            if self.stacking_model is not None:
                self.model_version = "stacking_v1"
            elif self.gru is not None:
                self.model_version = "gru_v1"
            self.runtime_feature_order = list(self.pipeline.gru_feature_names)
            
            # Check what's available
            has_stacking = self.stacking_model is not None
            has_gru = self.gru is not None
            has_sarima = self.sarima is not None
            
            if has_stacking:
                logger.info("✓ ENSEMBLE MODE: Stacking meta-model active")
                self.is_loaded = True
            elif has_gru:
                logger.info("✓ FALLBACK: GRU model active (stacking not available)")
                self.is_loaded = True
            else:
                logger.error("✗ No models available")
                self.is_loaded = False
                return False
            
            logger.info(f"Model status - Stacking: {has_stacking}, GRU: {has_gru}, SARIMA: {has_sarima}")
            return True
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.is_loaded = False
            return False

    def build_features(self, pm25, pm10, temperature=None, wind_speed=None, hour=None, day=None, month=None, weekday=None) -> pd.DataFrame:
        """Build feature vector for model input."""
        now = datetime.now()
        row = {
            "pm25": pm25 or 0,
            "pm10": pm10 or 0,
            "temperature": temperature or 20,
            "wind_speed": wind_speed or 0,
            "hour": hour if hour is not None else now.hour,
            "day": day if day is not None else now.day,
            "month": month if month is not None else now.month,
            "weekday": weekday if weekday is not None else now.weekday(),
        }

        if self.runtime_feature_order:
            ordered = {c: row.get(c, 0) for c in self.runtime_feature_order}
            return pd.DataFrame([ordered], columns=self.runtime_feature_order, dtype=np.float32)
        else:
            return pd.DataFrame([row], dtype=np.float32)

    def prepare_sequence(self, X_scaled: np.ndarray, seq_len: int = 24) -> np.ndarray:
        """Prepare 3D sequence for GRU input."""
        if len(X_scaled.shape) == 2:
            return np.repeat(X_scaled[np.newaxis, :, :], seq_len, axis=1)
        return X_scaled

    def predict_with_stacking(self, gru_pred, sarima_pred) -> float:
        """
        Use stacking meta-model to combine GRU and SARIMA predictions.
        
        Args:
            gru_pred: GRU prediction
            sarima_pred: SARIMA prediction
        
        Returns:
            Combined prediction
        """
        try:
            if self.stacking_model is None:
                # Fallback: simple average
                return (gru_pred + sarima_pred) / 2
            
            # Use stacking model
            meta_input = np.array([[gru_pred, sarima_pred]], dtype=np.float32)
            stacking_pred = self.stacking_model.predict(meta_input)
            
            return float(stacking_pred[0]) if isinstance(stacking_pred, np.ndarray) else float(stacking_pred)
        except Exception as e:
            logger.warning(f"Stacking prediction failed: {e}, using average")
            return (gru_pred + sarima_pred) / 2

    def _compute_aqi_gru(self, pm25, pm10, temperature, wind_speed) -> float:
        """Compute AQI using GRU model."""
        try:
            if self.pipeline is not None:
                event = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "pm25": pm25,
                    "pm10": pm10,
                    "temperature": temperature,
                    "wind_speed": wind_speed,
                    "humidity": 0.0,
                }
                return float(self.pipeline._predict_gru([event], event))

            return None
        
        except Exception as e:
            logger.error(f"GRU prediction failed: {e}")
            return None

    def _compute_aqi_sarima(self, pm25, pm10) -> float:
        """Compute AQI using SARIMA model."""
        try:
            if self.pipeline is not None:
                event = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "pm25": pm25,
                    "pm10": pm10,
                    "temperature": 0.0,
                    "wind_speed": 0.0,
                    "humidity": 0.0,
                }
                sarima_pred, _ = self.pipeline._predict_sarima([event], steps=30)
                return float(sarima_pred)

            return None
        
        except Exception as e:
            logger.error(f"SARIMA prediction failed: {e}")
            return None

    def predict(self, pm25, pm10, temperature=None, wind_speed=None, humidity=None) -> float:
        """
        Predict AQI using best available model (stacking > GRU > fallback).
        """
        if not self.is_loaded and not self.load_model():
            return None

        try:
            if hasattr(self.pipeline, "predict_from_measurements"):
                result = self.pipeline.predict_from_measurements(
                    pm2_5=pm25,
                    pm10=pm10,
                    temperature=temperature,
                    humidity=humidity,
                    wind_speed=wind_speed,
                )
                return result.prediction

            seed_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "pm25": pm25,
                "pm10": pm10,
                "temperature": temperature,
                "wind_speed": wind_speed,
                "humidity": humidity or 0.0,
                "gas_level": pm25,
            }
            result = self.pipeline.predict([seed_event], seed_event)
            return result.prediction
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return (pm25 * 0.6 + pm10 * 0.4)  # Fallback calculation

    def predict_batch(self, pm25_list, pm10_list, temp_list=None, wind_list=None):
        """
        Predict AQI for multiple data points.
        
        Args:
            pm25_list: List of PM2.5 values
            pm10_list: List of PM10 values
            temp_list: List of temperatures (optional)
            wind_list: List of wind speeds (optional)
        
        Returns:
            List of AQI predictions
        """
        predictions = []
        for i in range(len(pm25_list)):
            temp = temp_list[i] if temp_list else None
            wind = wind_list[i] if wind_list else None
            pred = self.predict(pm25_list[i], pm10_list[i], temp, wind)
            predictions.append(pred)
        return predictions

    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "model_version": self.model_version,
            "stacking_available": self.stacking_model is not None,
            "gru_available": self.gru is not None,
            "sarima_available": self.sarima is not None,
            "scalers_available": self.scaler_X is not None and self.scaler_y is not None,
            "is_loaded": self.is_loaded
        }


# Global singleton instance
_model_service = None


def get_model_service() -> MLModelService:
    """Get or create the ML model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = MLModelService()
        _model_service.load_model()
    return _model_service


def init_model_service() -> MLModelService:
    """Initialize (or reinitialize) model service."""
    global _model_service
    _model_service = MLModelService()
    _model_service.load_model()
    return _model_service
