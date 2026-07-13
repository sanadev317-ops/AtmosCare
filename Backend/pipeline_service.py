from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from Backend.smog_predictor_bridge import SmogPredictorBridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOG_ROOT = PROJECT_ROOT / "smog"
SARIMA_ROOT = SMOG_ROOT / "artifacts" / "models" / "sarima_optimized"
STACKING_PATH = SMOG_ROOT / "artifacts" / "models" / "stacking_model.pkl"
GRU_PATH = PROJECT_ROOT / "gru_aqi_model.keras"
GRU_SCALER_X_PATH = PROJECT_ROOT / "scaler_X.pkl"
GRU_SCALER_Y_PATH = PROJECT_ROOT / "scaler_y.pkl"


def _discover_latest(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


SARIMA_MODEL_PATH = _discover_latest(SARIMA_ROOT, "sarima_*.pkl")
SARIMA_SCALER_PATH = SARIMA_ROOT / "sarima_scaler.joblib"


def _patch_pandas_stringarray_compat() -> None:
    """Patch pandas StringArray for cross-version pickle compatibility."""
    try:
        from pandas.core.arrays.string_ import StringArray

        if getattr(StringArray, "_sarima_compat_patched", False):
            return

        original_setstate = StringArray.__setstate__

        def compat_setstate(self, state):
            if isinstance(state, tuple):
                if len(state) == 2:
                    dtype, values = state
                    attrs = {}
                elif len(state) == 3:
                    dtype, values, attrs = state
                else:
                    return original_setstate(self, state)

                type(self).__init__(self, np.asarray(values, dtype=object), copy=False)

                try:
                    self._dtype = dtype
                except Exception:
                    pass

                if isinstance(attrs, dict):
                    for key, value in attrs.items():
                        try:
                            setattr(self, key, value)
                        except Exception:
                            pass

                return None

            return original_setstate(self, state)

        StringArray.__setstate__ = compat_setstate
        StringArray._sarima_compat_patched = True
    except Exception:
        pass


@dataclass
class InferenceResult:
    prediction: float
    confidence: float
    gru_prediction: float
    sarima_prediction: float
    sarima_forecast: List[float]
    stacking_input: List[float]
    model_status: Dict[str, Any]
    gru_forecast: Optional[List[float]] = None
    forecast_7d: Optional[List[Dict[str, Any]]] = None
    smog_sources: Optional[Dict[str, Any]] = None
    health: Optional[Dict[str, Any]] = None


class AirQualityInferenceService:
    """Runtime-only inference layer for GRU, SARIMA, and stacking."""

    def __init__(self) -> None:
        self.smog_bridge = SmogPredictorBridge()
        self.use_smog_bridge = False
        self.is_loaded = False
        self.model_version = "unknown"
        self.gru = None
        self.gru_scaler_x = None
        self.gru_scaler_y = None
        self.sarima_results = None
        self.sarima_scaler = None
        self.stacking_model = None
        self.stacking_is_fitted = False
        self.gru_sequence_length = 24
        self.gru_feature_names = [
            "pm25",
            "pm10",
            "temperature",
            "wind_speed",
            "hour",
            "day",
            "month",
            "weekday",
        ]
        self.sarima_feature_names = [
            "temperature_2m_mean",
            "industrial_index",
            "so2",
            "no2",
            "relative_humidity_2m_mean",
            "stagnation_index",
            "crop_burning_intensity",
            "month_sin",
            "month_cos",
        ]

    def load(self) -> Dict[str, Any]:
        status = {
            "gru": False,
            "sarima": False,
            "stacking": False,
            "engine": "legacy",
        }

        bridge_status = self.smog_bridge.load()
        if bridge_status.get("bridge"):
            self.use_smog_bridge = True
            self.is_loaded = True
            self.model_version = self.smog_bridge.model_version
            status.update(bridge_status)
            status["engine"] = "smog_bridge"
            return status

        if GRU_PATH.exists():
            self.gru = tf.keras.models.load_model(GRU_PATH, compile=False)
            self.gru_sequence_length = int(self.gru.input_shape[1] or 24)
            status["gru"] = True

        if GRU_SCALER_X_PATH.exists():
            self.gru_scaler_x = joblib.load(GRU_SCALER_X_PATH)
        if GRU_SCALER_Y_PATH.exists():
            self.gru_scaler_y = joblib.load(GRU_SCALER_Y_PATH)

        if SARIMA_MODEL_PATH and SARIMA_MODEL_PATH.exists():
            _patch_pandas_stringarray_compat()
            self.sarima_results = SARIMAXResults.load(SARIMA_MODEL_PATH)
            status["sarima"] = True
        if SARIMA_SCALER_PATH.exists():
            self.sarima_scaler = joblib.load(SARIMA_SCALER_PATH)
            if hasattr(self.sarima_scaler, "feature_names_in_"):
                self.sarima_feature_names = list(self.sarima_scaler.feature_names_in_)

        if STACKING_PATH.exists():
            stacking_obj = joblib.load(STACKING_PATH)
            if isinstance(stacking_obj, dict) and "model" in stacking_obj:
                self.stacking_model = stacking_obj["model"]
                self.stacking_is_fitted = bool(stacking_obj.get("is_fitted", True))
                status["stacking"] = True

        self.is_loaded = any(status.get(key) for key in ("gru", "sarima", "stacking"))
        if self.is_loaded:
            self.model_version = "legacy_stacking_v1"
        return status

    def predict_from_measurements(
        self,
        pm2_5: float,
        pm10: Optional[float] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        wind_speed: Optional[float] = None,
    ) -> InferenceResult:
        event = {
            "timestamp": datetime.now(timezone.utc),
            "pm25": pm2_5,
            "pm2_5": pm2_5,
            "gas_level": pm2_5,
            "pm10": pm10 if pm10 is not None else pm2_5 * 1.15,
            "temperature": temperature if temperature is not None else 25.0,
            "humidity": humidity if humidity is not None else 45.0,
            "wind_speed": wind_speed if wind_speed is not None else 2.0,
            "source": "api",
        }
        return self.predict(buffer=[event], seed_event=event)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        # FIX: the old check `isinstance(value, float) and math.isnan(value)`
        # only caught NaN when the input was ALREADY a Python float. A value
        # arriving as a string ("NaN", "Infinity" — common from JSON payloads
        # or values round-tripped through MongoDB) skipped the guard entirely
        # and `float("NaN")` / `float("Infinity")` returned an actual nan/inf
        # untouched. It also never checked for inf at all. That bad value
        # then propagated through _make_sarima_frame()'s 30-step extrapolation
        # and crashed SARIMAX.get_forecast() with "exog contains inf or nans".
        # Now: always attempt the conversion first, then validate the RESULT.
        if value is None:
            return default
        try:
            result = float(value)
        except Exception:
            return default
        if math.isnan(result) or math.isinf(result):
            return default
        return result

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 500.0) -> float:
        return float(max(low, min(high, value)))

    @staticmethod
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

    def _event_to_row(self, event: Dict[str, Any]) -> Dict[str, Any]:
        ts = self._parse_timestamp(event.get("timestamp"))
        temperature = self._safe_float(event.get("temperature"))
        humidity = self._safe_float(event.get("humidity"))
        wind_speed = self._safe_float(event.get("wind_speed"))
        gas_level = self._safe_float(event.get("gas_level"), self._safe_float(event.get("pm25")))
        pm25 = self._safe_float(event.get("pm25"), gas_level)
        pm10 = self._safe_float(event.get("pm10"), max(pm25 * 1.15, gas_level * 1.15))

        return {
            "timestamp": ts,
            "pm25": pm25,
            "pm10": pm10,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
        }

    def _make_gru_frame(self, buffer: Sequence[Dict[str, Any]], seed_event: Dict[str, Any]) -> pd.DataFrame:
        rows = [self._event_to_row(item) for item in buffer] if buffer else []
        if not rows:
            rows = [self._event_to_row(seed_event)]

        while len(rows) < self.gru_sequence_length:
            rows.insert(0, rows[0])
        rows = rows[-self.gru_sequence_length :]

        frame = pd.DataFrame(
            [
                {
                    "pm25": row["pm25"],
                    "pm10": row["pm10"],
                    "temperature": row["temperature"],
                    "wind_speed": row["wind_speed"],
                    "hour": row["timestamp"].hour,
                    "day": row["timestamp"].day,
                    "month": row["timestamp"].month,
                    "weekday": row["timestamp"].weekday(),
                }
                for row in rows
            ],
            columns=self.gru_feature_names,
        )
        return frame.astype(np.float32)

    def _make_sarima_frame(self, buffer: Sequence[Dict[str, Any]], steps: int = 30) -> pd.DataFrame:
        rows = [self._event_to_row(item) for item in buffer] if buffer else []
        if not rows:
            rows = [self._event_to_row(self.synthetic_fallback())]

        latest = rows[-1]
        frame_rows: List[Dict[str, float]] = []
        base_ts = latest["timestamp"]
        for i in range(steps):
            ts = base_ts + timedelta(hours=i + 1)
            temp = latest["temperature"] + (0.12 * i)
            humidity = max(0.0, min(100.0, latest["humidity"] - (0.08 * i)))
            wind_speed = max(0.0, latest["wind_speed"] + (0.03 * i))
            gas_level = max(0.0, latest["pm25"] + (0.15 * i))

            frame_rows.append(
                {
                    "temperature_2m_mean": temp,
                    "industrial_index": gas_level * 0.9,
                    "so2": gas_level * 0.25,
                    "no2": gas_level * 0.2,
                    "relative_humidity_2m_mean": humidity,
                    "stagnation_index": max(0.0, 100.0 - (wind_speed * 8.0)),
                    "crop_burning_intensity": gas_level * 0.12,
                    "month_sin": math.sin(2 * math.pi * ts.month / 12.0),
                    "month_cos": math.cos(2 * math.pi * ts.month / 12.0),
                }
            )

        frame = pd.DataFrame(frame_rows, columns=self.sarima_feature_names).astype(np.float32)
        if self.sarima_scaler is not None:
            transformed = self.sarima_scaler.transform(frame)
            frame = pd.DataFrame(transformed, columns=self.sarima_feature_names)

        # FIX (defense-in-depth): even with _safe_float hardened above, don't
        # let any NaN/Inf that slips in another way (e.g. a scaler edge case,
        # or overflow from 30 steps of extrapolation on an extreme reading)
        # reach statsmodels and 500 the whole /predict endpoint. Replace any
        # bad cell with that column's own finite mean, or 0.0 if the whole
        # column is bad.
        if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
            for col in frame.columns:
                col_vals = frame[col].to_numpy(dtype=np.float64)
                bad = ~np.isfinite(col_vals)
                if bad.any():
                    finite_vals = col_vals[np.isfinite(col_vals)]
                    fill_value = float(finite_vals.mean()) if finite_vals.size else 0.0
                    col_vals[bad] = fill_value
                    frame[col] = col_vals.astype(np.float32)

        return frame

    def synthetic_fallback(self, device_id: str = "api-fallback") -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        base = random.uniform(110.0, 220.0)
        from Backend.air_quality_service import estimate_secondary_pollutants

        secondary = estimate_secondary_pollutants(base)
        return {
            "device_id": device_id,
            "timestamp": now,
            "temperature": round(random.uniform(20.0, 36.0), 2),
            "humidity": round(random.uniform(25.0, 80.0), 2),
            "gas_level": round(base, 2),
            "pm2_5": round(base, 2),
            "wind_speed": round(random.uniform(0.2, 7.5), 2),
            "o3": secondary["o3"],
            "no2": secondary["no2"],
            "co": secondary["co"],
            "source": "synthetic_fallback",
        }

    def _predict_gru(self, buffer: Sequence[Dict[str, Any]], seed_event: Dict[str, Any]) -> float:
        if self.gru is None:
            return self._clamp(self._safe_float(seed_event.get("gas_level")) * 1.1)

        frame = self._make_gru_frame(buffer, seed_event)
        if self.gru_scaler_x is not None:
            scaled = self.gru_scaler_x.transform(frame)
        else:
            scaled = frame.to_numpy(dtype=np.float32)

        seq = scaled.reshape(1, self.gru_sequence_length, -1)
        pred = self.gru.predict(seq, verbose=0)
        pred = np.ravel(pred)
        if self.gru_scaler_y is not None:
            pred = np.ravel(self.gru_scaler_y.inverse_transform(pred.reshape(-1, 1)))
        return self._clamp(float(pred[0]))

    def _predict_sarima(self, buffer: Sequence[Dict[str, Any]], steps: int = 30) -> Tuple[float, List[float]]:
        if self.sarima_results is None:
            latest = self._event_to_row(buffer[-1] if buffer else self.synthetic_fallback())
            proxy = self._clamp(latest["pm25"] * 0.95 + latest["pm10"] * 0.05)
            return proxy, [proxy for _ in range(steps)]

        try:
            exog = self._make_sarima_frame(buffer, steps=steps)
            forecast = self.sarima_results.get_forecast(steps=steps, exog=exog)
            sarima_values = np.ravel(forecast.predicted_mean)
            sarima_values = [self._clamp(float(v)) for v in sarima_values]
            return sarima_values[0], sarima_values
        except Exception as e:
            # FIX: a SARIMA failure (bad exog, statsmodels internal error, etc.)
            # used to propagate all the way up and 500 the entire /predict
            # endpoint. Degrade to the same proxy forecast used when no SARIMA
            # model is loaded at all, so the dashboard still gets a usable
            # (GRU-only-influenced) response instead of an error page.
            import logging
            logging.getLogger(__name__).error(f"SARIMA forecast failed, falling back to proxy: {e}")
            latest = self._event_to_row(buffer[-1] if buffer else self.synthetic_fallback())
            proxy = self._clamp(latest["pm25"] * 0.95 + latest["pm10"] * 0.05)
            return proxy, [proxy for _ in range(steps)]

    def _stack_predictions(self, gru_pred: float, sarima_pred: float) -> float:
        gru_pred = float(np.ravel([gru_pred])[0])
        sarima_pred = float(np.ravel([sarima_pred])[0])
        if self.stacking_model is None:
            return self._clamp((gru_pred + sarima_pred) / 2.0)
        stacked = self.stacking_model.predict(np.array([[gru_pred, sarima_pred]], dtype=np.float32))
        stacked = np.ravel(stacked)
        return self._clamp(float(stacked[0]))

    def _compute_confidence(self, gru_pred: float, sarima_pred: float, buffer_size: int) -> float:
        agreement = 1.0 - min(1.0, abs(gru_pred - sarima_pred) / 500.0)
        coverage = min(1.0, buffer_size / float(self.gru_sequence_length))
        confidence = 0.45 + (0.35 * agreement) + (0.20 * coverage)
        return float(max(0.1, min(0.99, confidence)))

    def predict(self, buffer: Sequence[Dict[str, Any]], seed_event: Optional[Dict[str, Any]] = None) -> InferenceResult:
        if self.use_smog_bridge and self.smog_bridge.is_loaded:
            seed_event = seed_event or (buffer[-1] if buffer else self.synthetic_fallback())
            bridge_result = self.smog_bridge.predict(buffer=buffer, seed_event=seed_event)
            return InferenceResult(
                prediction=bridge_result["prediction"],
                confidence=bridge_result["confidence"],
                gru_prediction=bridge_result["gru_prediction"],
                sarima_prediction=bridge_result["sarima_prediction"],
                sarima_forecast=bridge_result["sarima_forecast"],
                stacking_input=bridge_result["stacking_input"],
                model_status=bridge_result["model_status"],
                gru_forecast=bridge_result.get("gru_forecast"),
                forecast_7d=bridge_result.get("forecast_7d"),
                smog_sources=bridge_result.get("smog_sources"),
                health=bridge_result.get("health"),
            )

        seed_event = seed_event or (buffer[-1] if buffer else self.synthetic_fallback())
        seed_event = self._event_to_row(seed_event)

        gru_pred = self._predict_gru(buffer, seed_event)
        sarima_pred, sarima_forecast = self._predict_sarima(buffer, steps=30)
        final_pred = self._stack_predictions(gru_pred, sarima_pred)
        confidence = self._compute_confidence(gru_pred, sarima_pred, len(buffer))

        return InferenceResult(
            prediction=final_pred,
            confidence=confidence,
            gru_prediction=gru_pred,
            sarima_prediction=sarima_pred,
            sarima_forecast=sarima_forecast,
            stacking_input=[gru_pred, sarima_pred],
            model_status=self.model_status(),
        )

    def model_status(self) -> Dict[str, Any]:
        if self.use_smog_bridge:
            predictor = self.smog_bridge.predictor
            return {
                "gru_loaded": bool(predictor and predictor.gru is not None),
                "sarima_loaded": bool(predictor and predictor.sarima is not None),
                "stacking_loaded": self.smog_bridge.stacking_model is not None,
                "gru_sequence_length": self.smog_bridge.seq_len,
                "model_version": self.smog_bridge.model_version,
                "engine": "smog_bridge",
            }
        return {
            "gru_loaded": self.gru is not None,
            "sarima_loaded": self.sarima_results is not None,
            "stacking_loaded": self.stacking_model is not None,
            "gru_sequence_length": self.gru_sequence_length,
            "gru_feature_count": len(self.gru_feature_names),
            "sarima_feature_count": len(self.sarima_feature_names),
            "engine": "legacy",
        }
