"""Production real-time forecasting orchestration on top of existing predictors."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..inference.predictor import Predictor
from ..utils import get_logger
from .device_buffer import DeviceBufferManager
from .mongo_store import MongoStore
from .stacking import StackingModel, compute_confidence, search_best_alpha

logger = get_logger(__name__)


class RealtimeForecastSystem:
    """Combine GRU + SARIMA + stacking + MongoDB persistence."""

    def __init__(self, config: Dict[str, Any], model_run_dir: str):
        self.config = config
        self.model_run_dir = model_run_dir
        self.predictor = Predictor(config, model_run_dir)
        self.mongo = MongoStore(config)
        self.buffer = DeviceBufferManager(
            maxlen=int(config.get("realtime", {}).get("buffer_length", 60))
        )
        self.stacking_path = config.get("realtime", {}).get(
            "stacking_model_path",
            "artifacts/models/stacking_model.pkl",
        )
        self.alpha = float(config.get("realtime", {}).get("default_alpha", 0.5))
        self.stacking = self._load_stacking_model()

    def _load_stacking_model(self) -> StackingModel:
        try:
            return StackingModel.load(self.stacking_path)
        except Exception:
            return StackingModel("ridge")

    @staticmethod
    def _to_frame(sequence: pd.DataFrame) -> pd.DataFrame:
        if sequence is None or len(sequence) == 0:
            raise ValueError("Input sequence is empty.")
        if not isinstance(sequence, pd.DataFrame):
            sequence = pd.DataFrame(sequence)
        if not isinstance(sequence.index, pd.DatetimeIndex):
            sequence = sequence.copy()
            sequence.index = pd.to_datetime(sequence.index, errors="coerce")
        return sequence.sort_index()

    def align_predictions(self, gru_pred, sarima_pred, y_true=None):
        gru = np.asarray(gru_pred, dtype=float).ravel()
        sarima = np.asarray(sarima_pred, dtype=float).ravel()
        n = min(len(gru), len(sarima), len(y_true) if y_true is not None else len(gru))
        gru = gru[:n]
        sarima = sarima[:n]
        if y_true is None:
            return gru, sarima, None
        return gru, sarima, np.asarray(y_true, dtype=float).ravel()[:n]

    def auto_weighted_hybrid(self, gru_pred, sarima_pred, y_true):
        result = search_best_alpha(gru_pred, sarima_pred, y_true)
        self.alpha = result["best_alpha"]
        return result

    def build_meta_features(self, gru_pred, sarima_pred) -> np.ndarray:
        gru, sarima, _ = self.align_predictions(gru_pred, sarima_pred)
        return np.column_stack([gru, sarima])

    def predict_from_sequence(
        self,
        sequence: pd.DataFrame,
        n_steps: int = 30,
        device_id: Optional[str] = None,
        store: bool = True,
    ) -> Dict[str, Any]:
        sequence = self._to_frame(sequence)
        predictions = self.predictor.predict(sequence, n_steps=n_steps)

        if "gru" not in predictions or "sarima" not in predictions:
            raise RuntimeError("GRU and SARIMA predictions are required.")

        gru_pred = np.asarray(predictions["gru"], dtype=float).ravel()
        sarima_pred = np.asarray(predictions["sarima"], dtype=float).ravel()
        gru_pred, sarima_pred, _ = self.align_predictions(gru_pred, sarima_pred)
        hybrid_pred = self.alpha * gru_pred + (1.0 - self.alpha) * sarima_pred
        meta_features = self.build_meta_features(gru_pred, sarima_pred)

        if self.stacking.is_fitted:
            stacking_pred = self.stacking.predict(meta_features)
        else:
            stacking_pred = hybrid_pred

        gru_lower = np.asarray(predictions.get("gru_lower", gru_pred), dtype=float).ravel()[: len(stacking_pred)]
        gru_upper = np.asarray(predictions.get("gru_upper", gru_pred), dtype=float).ravel()[: len(stacking_pred)]
        confidence = compute_confidence(stacking_pred, gru_lower, gru_upper)

        payload = {
            "timestamp": datetime.utcnow(),
            "device_id": device_id,
            "gru_pred": gru_pred.tolist(),
            "sarima_pred": sarima_pred.tolist(),
            "hybrid_pred": hybrid_pred.tolist(),
            "stacking_pred": stacking_pred.tolist(),
            "confidence": confidence.tolist(),
            "alpha": float(self.alpha),
        }

        prediction_id = None
        if store:
            prediction_id = self.mongo.store_prediction(payload)
            payload["prediction_id"] = prediction_id

        return payload

    def ingest_iot(self, sample: Dict[str, Any], api_device_id: Optional[str] = None) -> Dict[str, Any]:
        device_id = api_device_id or sample.get("device_id")
        if not device_id:
            raise ValueError("device_id is required.")

        sample = dict(sample)
        sample["device_id"] = device_id
        sample.setdefault("timestamp", datetime.utcnow())
        self.mongo.upsert_device(device_id=device_id, status="active")
        self.mongo.store_iot_data(sample)
        self.buffer.append(device_id, sample)

        recent = self.buffer.get_recent_frame(device_id, limit=self.predictor.seq_len)
        if len(recent) < self.predictor.seq_len:
            history = self.mongo.get_recent_iot_data(device_id, limit=self.predictor.seq_len)
            if history:
                self.buffer.preload_from_history(device_id, history)
                recent = self.buffer.get_recent_frame(device_id, limit=self.predictor.seq_len)

        if len(recent) < self.predictor.seq_len:
            raise RuntimeError("Insufficient buffer length for prediction.")

        pred = self.predict_from_sequence(recent, n_steps=self.predictor.forecast_horizon, device_id=device_id)
        actual = sample.get("actual_pm2_5")
        if actual is not None and pred.get("prediction_id"):
            stacking_pred = np.asarray(pred.get("stacking_pred", []), dtype=float).ravel()
            error = float(abs(float(actual) - stacking_pred[0])) if len(stacking_pred) else None
            self.mongo.update_prediction_actual(pred["prediction_id"], actual, error=error)
        return pred

    def history(self, device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self.mongo.get_predictions(device_id, limit=limit)

    def health(self) -> Dict[str, Any]:
        return {
            "predictor": bool(self.predictor),
            "mongo": self.mongo.health(),
            "buffer_devices": len(self.buffer._buffers),
            "stacking_loaded": bool(self.stacking.is_fitted),
        }

    def continual_learning_step(self, device_id: str, recent_y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        history = self.mongo.get_predictions(device_id, limit=200)
        if not history:
            return {"updated": False, "reason": "no_history"}

        X_rows = []
        y_rows = []
        for doc in history:
            if "actual" not in doc:
                continue
            gru = doc.get("gru_pred", [])
            sarima = doc.get("sarima_pred", [])
            actual = doc.get("actual")
            if actual is None:
                continue
            gru_arr = np.asarray(gru, dtype=float).ravel()
            sarima_arr = np.asarray(sarima, dtype=float).ravel()
            actual_arr = np.asarray(actual, dtype=float).ravel()
            if len(actual_arr) == 0 or len(gru_arr) == 0 or len(sarima_arr) == 0:
                continue
            X_rows.append(np.array([[gru_arr[0], sarima_arr[0]]], dtype=float))
            y_rows.append(np.array([actual_arr[0]], dtype=float))

        if not X_rows:
            return {"updated": False, "reason": "no_labeled_predictions"}

        X_meta = np.vstack(X_rows)
        y_true = np.concatenate(y_rows)
        alpha_result = self.auto_weighted_hybrid(X_meta[:, 0], X_meta[:, 1], y_true)
        metrics = self.stacking.fit(X_meta, y_true)
        self.stacking.save(self.stacking_path)
        return {"updated": True, "alpha": alpha_result["best_alpha"], "metrics": metrics}
