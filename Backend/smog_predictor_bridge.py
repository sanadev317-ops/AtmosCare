"""Bridge AtmosCare inference to the latest smog pipeline models."""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SMOG_ROOT = WORKSPACE_ROOT / "smog"
MODEL_DIR = SMOG_ROOT / "artifacts" / "models"
CONFIG_PATH = SMOG_ROOT / "ml_pipeline" / "config" / "config.yaml"

GRU_FEATURE_NAMES = [
    "pm2_5_log",
    "no2",
    "so2",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "traffic_combined",
    "crop_burning_intensity",
    "pm_lag7",
    "pm_lag14",
    "pm_roll7",
    "pm_roll14",
    "pm_diff1",
    "pm_diff3",
    "pm_zscore7",
    "no2_o3",
    "temp_wind",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]


def _discover_latest(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except Exception:
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _parse_timestamp(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _event_to_pm25(event: Dict[str, Any]) -> float:
    gas_level = _safe_float(event.get("gas_level"))
    pm25 = _safe_float(event.get("pm25"), gas_level)
    pm25 = _safe_float(event.get("pm2_5"), pm25)
    return max(0.0, pm25)


class SmogPredictorBridge:
    """Wraps smog/ml_pipeline Predictor for AtmosCare runtime inference."""

    def __init__(self) -> None:
        self.predictor = None
        self.stacking_model = None
        self.seq_len = 60
        self.forecast_horizon = 30
        self.model_version = "unknown"
        self._loaded = False
        self._sarima_exog_features: List[str] = []
        self._gru_scaler = None

    def load(self) -> Dict[str, Any]:
        status = {"gru": False, "sarima": False, "stacking": False, "bridge": False}

        if not SMOG_ROOT.exists():
            logger.warning("Smog pipeline root not found: %s", SMOG_ROOT)
            return status

        if str(SMOG_ROOT) not in sys.path:
            sys.path.insert(0, str(SMOG_ROOT))

        try:
            from ml_pipeline.utils import load_config
            from ml_pipeline.inference.predictor import Predictor

            config = load_config(str(CONFIG_PATH))
            self.predictor = Predictor(config, str(MODEL_DIR))
            self.seq_len = int(getattr(self.predictor, "seq_len", 60))
            self.forecast_horizon = int(getattr(self.predictor, "forecast_horizon", 30))

            scaler_features = list(
                getattr(getattr(self.predictor, "sarima_scaler", None), "feature_names_in_", [])
            )
            if scaler_features:
                self._sarima_exog_features = scaler_features
                self.predictor._sarima_feature_columns = scaler_features
                if self.predictor.sarima is not None:
                    self.predictor.sarima._feature_columns = scaler_features

            status["gru"] = self.predictor.gru is not None
            status["sarima"] = self.predictor.sarima is not None

            meta_path = _discover_latest(MODEL_DIR / "hybrid", "meta_model_XGBoost_*.pkl")
            if meta_path:
                self.stacking_model = joblib.load(meta_path)
                status["stacking"] = True
                self.model_version = meta_path.stem

            gru_scaler_path = _discover_latest(MODEL_DIR / "gru_optimized", "gru_scaler.pkl")
            if gru_scaler_path and self.predictor is not None:
                try:
                    from ml_pipeline.data import DataPreprocessor

                    gru_scaler = joblib.load(gru_scaler_path)
                    gru_scaler.feature_names_in_ = np.array(GRU_FEATURE_NAMES, dtype=object)
                    gru_scaler.n_features_in_ = len(GRU_FEATURE_NAMES)
                    self._gru_scaler = gru_scaler
                    self.predictor.preprocessor = DataPreprocessor.from_scaler(
                        config,
                        scaler=gru_scaler,
                        numeric_cols=GRU_FEATURE_NAMES,
                        feature_cols=GRU_FEATURE_NAMES,
                        cities=[],
                        target=config.get("features", {}).get("target", "pm2_5"),
                    )
                    self.predictor._gru_scaler_path = str(gru_scaler_path)
                except Exception:
                    logger.warning("Could not initialize GRU preprocessor from scaler", exc_info=True)

            metadata_path = _discover_latest(MODEL_DIR / "gru_optimized", "gru_optimized_*_metadata.json")
            if metadata_path:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                self.seq_len = int(metadata.get("sequence_length", self.seq_len))
                self.forecast_horizon = int(metadata.get("forecast_horizon", self.forecast_horizon))

            self._loaded = status["gru"] or status["sarima"]
            status["bridge"] = self._loaded
            logger.info(
                "Smog bridge loaded (GRU=%s, SARIMA=%s, stacking=%s, version=%s)",
                status["gru"],
                status["sarima"],
                status["stacking"],
                self.model_version,
            )
        except Exception as exc:
            logger.exception("Failed to load smog predictor bridge: %s", exc)

        return status

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _rows_from_buffer(self, buffer: Sequence[Dict[str, Any]], seed_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in buffer or []:
            ts = _parse_timestamp(item.get("timestamp"))
            pm25 = _event_to_pm25(item)
            pm10 = _safe_float(item.get("pm10"), pm25 * 1.15)
            temp = _safe_float(item.get("temperature"), 25.0)
            humidity = _safe_float(item.get("humidity"), 45.0)
            wind = _safe_float(item.get("wind_speed"), 2.0)
            rows.append(
                {
                    "timestamp": ts,
                    "pm2_5": pm25,
                    "pm10": pm10,
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind,
                }
            )

        if not rows:
            ts = _parse_timestamp(seed_event.get("timestamp"))
            pm25 = _event_to_pm25(seed_event)
            rows.append(
                {
                    "timestamp": ts,
                    "pm2_5": pm25,
                    "pm10": _safe_float(seed_event.get("pm10"), pm25 * 1.15),
                    "temperature": _safe_float(seed_event.get("temperature"), 25.0),
                    "humidity": _safe_float(seed_event.get("humidity"), 45.0),
                    "wind_speed": _safe_float(seed_event.get("wind_speed"), 2.0),
                }
            )

        while len(rows) < self.seq_len:
            rows.insert(0, dict(rows[0]))
        return rows[-self.seq_len :]

    def _engineer_gru_frame(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        pm_series = pd.Series([row["pm2_5"] for row in rows], dtype=float)
        engineered: List[Dict[str, float]] = []

        for idx, row in enumerate(rows):
            pm25 = float(row["pm2_5"])
            temp = float(row["temperature"])
            humidity = float(row["humidity"])
            wind = float(row["wind_speed"])
            ts = row["timestamp"]

            no2 = pm25 * 0.20
            so2 = pm25 * 0.25
            o3_est = pm25 * 0.10

            lag7 = float(pm_series.iloc[max(0, idx - 7)]) if idx > 0 else pm25
            lag14 = float(pm_series.iloc[max(0, idx - 14)]) if idx > 0 else pm25
            roll7 = float(pm_series.iloc[max(0, idx - 6) : idx + 1].mean())
            roll14 = float(pm_series.iloc[max(0, idx - 13) : idx + 1].mean())
            diff1 = pm25 - float(pm_series.iloc[max(0, idx - 1)]) if idx > 0 else 0.0
            diff3 = pm25 - float(pm_series.iloc[max(0, idx - 3)]) if idx > 0 else 0.0
            window = pm_series.iloc[max(0, idx - 6) : idx + 1]
            zscore7 = float((pm25 - window.mean()) / (window.std() + 1e-6)) if len(window) > 1 else 0.0

            doy = ts.timetuple().tm_yday
            engineered.append(
                {
                    "pm2_5_log": math.log1p(max(0.0, pm25)),
                    "no2": no2,
                    "so2": so2,
                    "temperature_2m_mean": temp,
                    "relative_humidity_2m_mean": humidity,
                    "wind_speed_10m_max": wind,
                    "traffic_combined": pm25 * 0.15,
                    "crop_burning_intensity": pm25 * 0.12,
                    "pm_lag7": lag7,
                    "pm_lag14": lag14,
                    "pm_roll7": roll7,
                    "pm_roll14": roll14,
                    "pm_diff1": diff1,
                    "pm_diff3": diff3,
                    "pm_zscore7": zscore7,
                    "no2_o3": no2 * o3_est,
                    "temp_wind": temp * wind,
                    "month_sin": math.sin(2 * math.pi * ts.month / 12.0),
                    "month_cos": math.cos(2 * math.pi * ts.month / 12.0),
                    "doy_sin": math.sin(2 * math.pi * doy / 365.0),
                    "doy_cos": math.cos(2 * math.pi * doy / 365.0),
                }
            )

        return pd.DataFrame(engineered, columns=GRU_FEATURE_NAMES).astype(np.float32)

    def _buffer_to_dataframe(self, buffer: Sequence[Dict[str, Any]], seed_event: Dict[str, Any]) -> pd.DataFrame:
        rows = self._rows_from_buffer(buffer, seed_event)
        gru_frame = self._engineer_gru_frame(rows)

        payload = pd.DataFrame(
            {
                "pm2_5": [row["pm2_5"] for row in rows],
                **{col: gru_frame[col].tolist() for col in GRU_FEATURE_NAMES},
            },
            index=pd.DatetimeIndex([row["timestamp"] for row in rows], name="datetime"),
        )

        # Supplementary columns used by SARIMA exog projection.
        payload["temperature_2m_mean"] = [row["temperature"] for row in rows]
        payload["relative_humidity_2m_mean"] = [row["humidity"] for row in rows]
        payload["wind_speed_10m_max"] = [row["wind_speed"] for row in rows]
        payload["industrial_index"] = [row["pm2_5"] * 0.9 for row in rows]
        payload["so2"] = [row["pm2_5"] * 0.25 for row in rows]
        payload["no2"] = [row["pm2_5"] * 0.20 for row in rows]
        payload["crop_burning_intensity"] = [row["pm2_5"] * 0.12 for row in rows]
        payload["month_sin"] = gru_frame["month_sin"].tolist()
        payload["month_cos"] = gru_frame["month_cos"].tolist()
        return payload.sort_index()

    def _decode_gru_outputs(self, scaled_values: np.ndarray) -> np.ndarray:
        if self._gru_scaler is None:
            return np.asarray(scaled_values, dtype=float)

        decoded: List[float] = []
        for value in np.asarray(scaled_values, dtype=float).ravel():
            dummy = np.zeros((1, len(GRU_FEATURE_NAMES)), dtype=float)
            dummy[0, 0] = float(value)
            log_pm = float(self._gru_scaler.inverse_transform(dummy)[0, 0])
            decoded.append(max(0.0, math.expm1(log_pm)))
        return np.asarray(decoded, dtype=float)

    def _predict_gru(self, frame: pd.DataFrame, n_steps: int) -> np.ndarray:
        if self.predictor is None or self.predictor.gru is None or self._gru_scaler is None:
            return np.array([], dtype=float)

        feature_frame = frame[GRU_FEATURE_NAMES].astype(np.float32)
        scaled = self._gru_scaler.transform(feature_frame)
        seq = np.expand_dims(scaled[-self.seq_len :], axis=0).astype(np.float32)
        raw = np.ravel(self.predictor.gru.predict(seq))
        return self._decode_gru_outputs(raw)[:n_steps]

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 500.0) -> float:
        return float(max(low, min(high, value)))

    def _stack_predictions(self, gru_pred: float, sarima_pred: float) -> float:
        if self.stacking_model is None:
            return self._clamp((gru_pred + sarima_pred) / 2.0)
        try:
            stacked = self.stacking_model.predict(np.array([[gru_pred, sarima_pred]], dtype=np.float32))
            return self._clamp(float(np.ravel(stacked)[0]))
        except Exception:
            return self._clamp((gru_pred + sarima_pred) / 2.0)

    def predict(
        self,
        buffer: Sequence[Dict[str, Any]],
        seed_event: Optional[Dict[str, Any]] = None,
        n_steps: int = 30,
    ) -> Dict[str, Any]:
        if not self._loaded or self.predictor is None:
            raise RuntimeError("Smog predictor bridge is not loaded")

        seed_event = seed_event or (buffer[-1] if buffer else {})
        frame = self._buffer_to_dataframe(buffer, seed_event)

        if self._sarima_exog_features:
            self.predictor._sarima_feature_columns = self._sarima_exog_features
            if self.predictor.sarima is not None:
                self.predictor.sarima._feature_columns = self._sarima_exog_features

        gru_arr = self._predict_gru(frame, n_steps=n_steps)
        sarima_predictions = self.predictor.predict(
            frame,
            n_steps=n_steps,
            use_gru=False,
            use_sarima=True,
            use_hybrid=False,
        )
        sarima_arr = np.asarray(sarima_predictions.get("sarima", []), dtype=float).ravel()

        pm25 = _event_to_pm25(seed_event)
        gru_pred = float(gru_arr[0]) if len(gru_arr) else self._clamp(pm25 * 1.05)
        sarima_pred = float(sarima_arr[0]) if len(sarima_arr) else self._clamp(pm25 * 0.95)
        final_pred = self._stack_predictions(gru_pred, sarima_pred)

        agreement = 1.0 - min(1.0, abs(gru_pred - sarima_pred) / 500.0)
        coverage = min(1.0, len(buffer) / float(self.seq_len))
        confidence = float(max(0.1, min(0.99, 0.45 + (0.35 * agreement) + (0.20 * coverage))))

        gru_forecast = [self._clamp(float(v)) for v in np.asarray(gru_arr, dtype=float).ravel().tolist()]
        sarima_forecast = [self._clamp(float(v)) for v in sarima_arr.tolist()] or [sarima_pred]

        feature_row = frame.iloc[-1].to_dict() if len(frame) else {}
        base_result = {
            "prediction": final_pred,
            "confidence": confidence,
            "gru_prediction": gru_pred,
            "sarima_prediction": sarima_pred,
            "gru_forecast": gru_forecast,
            "sarima_forecast": sarima_forecast,
            "stacking_input": [gru_pred, sarima_pred],
            "model_status": {
                "gru_loaded": self.predictor.gru is not None,
                "sarima_loaded": self.predictor.sarima is not None,
                "stacking_loaded": self.stacking_model is not None,
                "gru_sequence_length": self.seq_len,
                "model_version": self.model_version,
                "engine": "smog_bridge",
            },
        }

        try:
            from Backend.analytics_service import enrich_prediction_analytics

            analytics = enrich_prediction_analytics(
                base_result,
                feature_row=feature_row,
                stack_fn=self._stack_predictions,
            )
            base_result.update(analytics)
        except Exception:
            pass

        return base_result

    def predict_from_measurements(
        self,
        pm2_5: float,
        pm10: Optional[float] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        wind_speed: Optional[float] = None,
    ) -> Dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc),
            "pm2_5": pm2_5,
            "pm25": pm2_5,
            "gas_level": pm2_5,
            "pm10": pm10 if pm10 is not None else pm2_5 * 1.15,
            "temperature": temperature if temperature is not None else 25.0,
            "humidity": humidity if humidity is not None else 45.0,
            "wind_speed": wind_speed if wind_speed is not None else 2.0,
            "source": "api",
        }
        return self.predict(buffer=[event], seed_event=event)
