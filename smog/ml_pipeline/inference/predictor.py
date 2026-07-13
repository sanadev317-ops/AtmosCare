# ============================================================================
# INFERENCE / PREDICTION MODULE
# ============================================================================
"""Production inference and predictions."""

import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..models import GRUModel, SARIMAModel, HybridPredictor
from ..data import DataPreprocessor
from ..utils import get_logger

logger = get_logger(__name__)

_PM25_MIN = 0.0
_PM25_MAX = 500.0

# ---------------------------------------------------------------------------
# Alias map: canonical SARIMA feature → candidate column names in the dataset.
# Checked left-to-right; first match wins.
# ---------------------------------------------------------------------------
_SARIMA_ALIASES: Dict[str, List[str]] = {
    "no2":                    ["no2", "nitrogen_dioxide"],
    "so2":                    ["so2", "sulphur_dioxide"],
    "co":                     ["co", "carbon_monoxide"],
    "o3":                     ["o3", "ozone"],
    "temperature_2m_mean":    ["temperature_2m_mean", "temperature"],
    "wind_speed_10m_max":     ["wind_speed_10m_max", "wind_speed"],
    "pressure_msl_mean":      ["pressure_msl_mean", "pressure"],
    "crop_burning_intensity": ["crop_burning_intensity"],
    "traffic_combined":       ["traffic_combined"],
    "stagnation_index":       ["stagnation_index"],
    "pm_lag1":                ["pm_lag1"],
    "pm_lag3":                ["pm_lag3"],
    "pm_lag7":                ["pm_lag7"],
}


def _resolve_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the first alias that exists as a column in *df*, or None."""
    for alias in _SARIMA_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


class Predictor:
    """Production predictor for PM2.5 forecasting."""

    def __init__(self, config: Dict, run_dir: str):
        self.config = config
        self.run_dir = run_dir
        self.target = config.get("features", {}).get("target", "pm2_5")
        self.forecast_horizon = config.get("features", {}).get("forecast_horizon", 6)
        self.seq_len = config.get("gru_model", {}).get("sequence_length", 60)
        self._sarima_feature_columns: Optional[List[str]] = None
        self._gru_scaler_path: Optional[str] = None

        # ── GRU preprocessor ──────────────────────────────────────────────
        preprocessor_path = self._discover_file("preprocessor.pkl")
        if preprocessor_path and os.path.exists(preprocessor_path):
            self.preprocessor = DataPreprocessor.load(preprocessor_path, config)
            logger.info("Preprocessor loaded")
        else:
            self.preprocessor = None
            self._gru_scaler_path = self._discover_file(
                "gru_scaler.pkl", "gru_*scaler*.pkl", "gru_*scaler*.joblib"
            )
            logger.warning("Preprocessor not found")
            if self._gru_scaler_path:
                logger.warning("GRU scaler artifact found at %s", self._gru_scaler_path)
            else:
                logger.warning("GRU scaler artifact not found")

        # ── GRU model ─────────────────────────────────────────────────────
        self.gru: Optional[GRUModel] = None
        gru_path = self._discover_file(
            "gru_optimized/best_*.keras",
            "gru_optimized/gru_optimized_*.keras",
            "gru_optimized/*.keras",
            "gru_model.keras",
        )
        if gru_path and os.path.exists(gru_path):
            self.gru = GRUModel.load(gru_path, config)
            self.seq_len = int(getattr(self.gru, "sequence_length", self.seq_len))
            self.forecast_horizon = int(getattr(self.gru, "forecast_horizon", self.forecast_horizon))
            logger.info("GRU model loaded")

        # ── SARIMA model ──────────────────────────────────────────────────
        self.sarima: Optional[SARIMAModel] = None
        sarima_path = self._discover_file(
            "sarima_optimized/sarima_*.pkl",
            "sarima_optimized/*.pkl",
            "sarimax_model.pkl",
            "sarima_model.pkl",
        )
        if sarima_path and os.path.exists(sarima_path):
            try:
                self.sarima = SARIMAModel.load(sarima_path, config)
                logger.info("SARIMAX model loaded from %s", sarima_path)
            except Exception:
                self.sarima = None
                logger.exception("SARIMAX model could not be loaded from %s", sarima_path)
        else:
            logger.warning("SARIMAX model file not found under %s", run_dir)

        # ── SARIMA scaler ─────────────────────────────────────────────────
        self.sarima_scaler = None
        sarima_scaler_path = self._discover_file(
            "sarima_optimized/sarima_scaler.joblib",
            "sarima_optimized/sarima_scaler.pkl",
        )
        if sarima_scaler_path and os.path.exists(sarima_scaler_path):
            try:
                self.sarima_scaler = joblib.load(sarima_scaler_path)
                fitted_names = list(getattr(self.sarima_scaler, "feature_names_in_", []))
                if fitted_names:
                    self._sarima_feature_columns = fitted_names
                    logger.info(
                        "SARIMA scaler loaded  features(%d)=%s",
                        len(fitted_names), fitted_names,
                    )
                else:
                    logger.warning(
                        "SARIMA scaler loaded but has no feature_names_in_; "
                        "column order will be inferred from alias map."
                    )
            except Exception:
                logger.exception("SARIMA scaler could not be loaded from %s", sarima_scaler_path)
        else:
            logger.warning("SARIMA scaler file not found under %s", run_dir)

        # Prefer feature column order stored on the model itself.
        if self.sarima is not None and getattr(self.sarima, "_feature_columns", None):
            self._sarima_feature_columns = self.sarima._feature_columns
            logger.info(
                "Using SARIMA model's own feature column order: %s",
                self._sarima_feature_columns,
            )

        logger.info(
            "Predictor ready — GRU=%s  SARIMA=%s  scaler=%s  sarima_features=%s",
            self.gru is not None,
            self.sarima is not None,
            self.sarima_scaler is not None,
            self._sarima_feature_columns,
        )

        # ── Hybrid blender ────────────────────────────────────────────────
        self.hybrid = HybridPredictor(config)

    # ── file discovery ────────────────────────────────────────────────────

    def _discover_file(self, *patterns: str) -> Optional[str]:
        base = Path(self.run_dir)
        candidates = []
        for pattern in patterns:
            candidates.extend(base.rglob(pattern))
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

    # ── GRU helpers ───────────────────────────────────────────────────────

    def _inverse_horizon(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return self.preprocessor.inverse_transform_target(values)
        if values.ndim == 2:
            return np.stack(
                [self.preprocessor.inverse_transform_target(values[:, i])
                 for i in range(values.shape[1])],
                axis=1,
            )
        raise ValueError(f"Unsupported prediction shape: {values.shape}")

    def _ensure_preprocessor(self, recent_data: pd.DataFrame) -> bool:
        if self.preprocessor is not None:
            return True
        if not self._gru_scaler_path:
            return False
        try:
            scaler = joblib.load(self._gru_scaler_path)
            if not hasattr(scaler, "transform"):
                logger.error("GRU scaler artifact is not a valid scaler object")
                return False
            numeric_cols = recent_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_names = list(getattr(scaler, "feature_names_in_", []))
            if feature_names and all(n in recent_data.columns for n in feature_names):
                numeric_cols = feature_names
            expected = getattr(scaler, "n_features_in_", None)
            if expected is not None and len(numeric_cols) != int(expected):
                logger.warning(
                    "GRU scaler expects %d features; data has %d numeric cols",
                    expected, len(numeric_cols),
                )
                return False
            self.preprocessor = DataPreprocessor.from_scaler(
                self.config,
                scaler=scaler,
                numeric_cols=numeric_cols,
                feature_cols=numeric_cols,
                cities=[],
                target=self.target,
            )
            logger.info("Reconstructed inference preprocessor from GRU scaler artifact")
            return True
        except Exception:
            logger.exception("Failed to reconstruct preprocessor from GRU scaler")
            return False

    # ── SARIMA exog helpers ───────────────────────────────────────────────

    def _project_series(self, series: pd.Series, n_steps: int) -> np.ndarray:
        """Linear-trend extrapolation for a single feature series."""
        values = pd.to_numeric(series, errors="coerce").ffill().bfill().to_numpy(dtype=float)
        if len(values) == 0:
            return np.zeros(n_steps, dtype=float)
        window = values[-min(len(values), 7):]
        slope = (window[-1] - window[0]) / max(len(window) - 1, 1) if len(window) >= 2 else 0.0
        last = float(window[-1])
        return np.array([last + slope * (i + 1) for i in range(n_steps)], dtype=float)

    def _future_index(self, recent_data: pd.DataFrame, n_steps: int) -> pd.DatetimeIndex:
        for col in ("date", "timestamp", "datetime"):
            if col in recent_data.columns:
                dt = pd.to_datetime(recent_data[col], errors="coerce").dropna()
                if not dt.empty:
                    return pd.date_range(
                        start=dt.iloc[-1] + pd.Timedelta(days=1),
                        periods=n_steps, freq="D",
                    )
        if isinstance(recent_data.index, pd.DatetimeIndex) and len(recent_data.index) > 0:
            return pd.date_range(
                start=recent_data.index[-1] + pd.Timedelta(days=1),
                periods=n_steps, freq="D",
            )
        return pd.date_range("2000-01-01", periods=n_steps, freq="D")

    def _get_pm_series(self, recent_data: pd.DataFrame) -> np.ndarray:
        """Return the PM2.5 array from recent_data, or empty array if absent."""
        for candidate in (self.target, "pm2_5", "pm25", "pm2.5"):
            if candidate in recent_data.columns:
                return (
                    pd.to_numeric(recent_data[candidate], errors="coerce")
                    .ffill().bfill().to_numpy(dtype=float)
                )
        return np.array([])

    def _build_sarima_exog(self, recent_data: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        """Build and scale the SARIMA exogenous matrix for *n_steps* future periods.

        Column construction rules
        -------------------------
        1. Column order is driven by ``self._sarima_feature_columns`` (exact
           order from the fitted scaler / stored model).
        2. Every feature is resolved via ``_resolve_col`` which checks all
           known aliases, tolerating naming differences between training data
           and inference data.
        3. Lag features (pm_lag1/3/7) are seeded from *actual* most-recent
           PM2.5 observations — NOT projected.  This was the root cause of
           SARIMA producing ~250 µg/m³ when truth was ~170.
        4. Pre/post-scale stats are logged so distribution shifts are visible.
        """
        # ── column order ──────────────────────────────────────────────────
        if self._sarima_feature_columns:
            feature_columns = list(self._sarima_feature_columns)
        else:
            feature_columns = list(_SARIMA_ALIASES.keys())
            logger.warning(
                "_sarima_feature_columns not set; using alias-map keys: %s",
                feature_columns,
            )

        # ── actual PM2.5 for lag seeding ──────────────────────────────────
        pm_arr = self._get_pm_series(recent_data)

        def _lag(k: int) -> float:
            if len(pm_arr) >= k:
                return float(pm_arr[-k])
            return float(pm_arr[0]) if len(pm_arr) > 0 else 0.0

        # ── pre-compute wind/pressure for stagnation index ────────────────
        wind_src = _resolve_col(recent_data, "wind_speed_10m_max")
        pres_src = _resolve_col(recent_data, "pressure_msl_mean")
        wind_proj = (
            self._project_series(recent_data[wind_src], n_steps)
            if wind_src else np.ones(n_steps, dtype=float)
        )
        pres_proj = (
            self._project_series(recent_data[pres_src], n_steps)
            if pres_src else np.full(n_steps, 1013.0, dtype=float)
        )

        # ── build columns ─────────────────────────────────────────────────
        raw: Dict[str, np.ndarray] = {}
        for col in feature_columns:

            if col == "pm_lag1":
                raw[col] = np.full(n_steps, _lag(1))
                continue
            if col == "pm_lag3":
                raw[col] = np.full(n_steps, _lag(3))
                continue
            if col == "pm_lag7":
                raw[col] = np.full(n_steps, _lag(7))
                continue

            if col == "stagnation_index":
                raw[col] = np.clip(
                    pres_proj / (np.maximum(wind_proj, 0.1) + 5.0), 0.0, 20.0
                )
                continue

            if col == "crop_burning_intensity":
                src = _resolve_col(recent_data, col)
                if src:
                    raw[col] = self._project_series(recent_data[src], n_steps)
                else:
                    pm10_src = _resolve_col(recent_data, "pm10")
                    raw[col] = (
                        np.maximum(
                            self._project_series(recent_data[pm10_src], n_steps) * 0.15, 0.0
                        ) if pm10_src else np.zeros(n_steps, dtype=float)
                    )
                continue

            if col == "traffic_combined":
                src = _resolve_col(recent_data, col)
                raw[col] = (
                    self._project_series(recent_data[src], n_steps)
                    if src else np.zeros(n_steps, dtype=float)
                )
                continue

            # standard feature — resolve via alias map
            src = _resolve_col(recent_data, col)
            if src:
                raw[col] = self._project_series(recent_data[src], n_steps)
            else:
                logger.warning(
                    "SARIMA exog: '%s' not found (aliases=%s) → zero-filled.",
                    col, _SARIMA_ALIASES.get(col, [col]),
                )
                raw[col] = np.zeros(n_steps, dtype=float)

        # ── assemble ──────────────────────────────────────────────────────
        exog_raw = pd.DataFrame(raw, columns=feature_columns).astype(float)

        logger.info(
            "SARIMA exog pre-scale  shape=%s  mean=%s  std=%s",
            exog_raw.shape,
            exog_raw.mean().round(3).to_dict(),
            exog_raw.std().round(3).to_dict(),
        )

        # ── scale ─────────────────────────────────────────────────────────
        if self.sarima_scaler is not None:
            try:
                exog_scaled = pd.DataFrame(
                    self.sarima_scaler.transform(exog_raw),
                    columns=feature_columns,
                )
            except Exception:
                logger.exception(
                    "sarima_scaler.transform failed — using raw exog (accuracy degraded)"
                )
                exog_scaled = exog_raw
        else:
            logger.warning("No SARIMA scaler — passing raw exog to model.")
            exog_scaled = exog_raw

        logger.info(
            "SARIMA exog post-scale shape=%s  mean=%s",
            exog_scaled.shape,
            exog_scaled.mean().round(3).to_dict(),
        )
        return exog_scaled

    # ── fallback forecast ─────────────────────────────────────────────────

    def _fallback_sarima_forecast(self, recent_data: pd.DataFrame, n_steps: int) -> np.ndarray:
        """Trend-based fallback when the SARIMA model file is unavailable or fails."""
        pm_arr = self._get_pm_series(recent_data)
        if len(pm_arr) == 0:
            logger.warning("Fallback SARIMA: no PM2.5 data found — returning zeros.")
            return np.zeros(n_steps, dtype=float)
        tail = pm_arr[-min(len(pm_arr), 12):]
        slope = (float(tail[-1]) - float(tail[0])) / max(len(tail) - 1, 1)
        last = float(tail[-1])
        forecast = np.array([last + slope * (i + 1) for i in range(n_steps)], dtype=float)
        return np.clip(forecast, _PM25_MIN, _PM25_MAX)

    # ── public predict ────────────────────────────────────────────────────

    def predict(
        self,
        recent_data: pd.DataFrame,
        n_steps: int = 24,
        use_gru: bool = True,
        use_sarima: bool = True,
        use_hybrid: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts from all registered models.

        SARIMA is guaranteed to produce a non-empty result: if the primary
        model call fails for any reason, the trend-based fallback fires, and
        if that also fails, zeros are returned.  The pipeline will therefore
        never see an empty ``predictions["sarima"]``.

        All arrays are clipped to [0, 500] µg/m³.
        """
        predictions: Dict[str, np.ndarray] = {}

        # ── GRU preprocessor (needed only for GRU) ────────────────────────
        preprocessor_ready = (
            self.preprocessor is not None
            or self._ensure_preprocessor(recent_data)
        )
        if not preprocessor_ready:
            logger.error(
                "GRU preprocessor unavailable — GRU will be skipped; "
                "SARIMA will still run."
            )

        # ── GRU ───────────────────────────────────────────────────────────
        if use_gru and self.gru is not None and preprocessor_ready:
            try:
                recent_scaled = self.preprocessor.transform(recent_data)
                if len(recent_scaled) >= self.seq_len:
                    X_pred = np.expand_dims(
                        recent_scaled.iloc[-self.seq_len:].values, axis=0
                    ).astype(np.float32)
                    logger.info("GRU input shape: %s", X_pred.shape)

                    unc = self.gru.predict_with_uncertainty(X_pred, n_samples=50)
                    y_pred  = self._inverse_horizon(unc["prediction"])
                    y_lower = self._inverse_horizon(unc["lower_bound"])
                    y_upper = self._inverse_horizon(unc["upper_bound"])

                    predictions["gru"] = np.clip(
                        np.asarray(y_pred, dtype=float).squeeze(), _PM25_MIN, _PM25_MAX
                    )
                    predictions["gru_lower"] = np.asarray(y_lower, dtype=float).squeeze()
                    predictions["gru_upper"] = np.asarray(y_upper, dtype=float).squeeze()
                    logger.info("GRU prediction: %.2f ug/m3", float(np.ravel(y_pred)[0]))
                else:
                    logger.warning(
                        "Not enough history for GRU (%d/%d rows)",
                        len(recent_scaled), self.seq_len,
                    )
            except Exception:
                logger.exception("GRU prediction failed")

        # ── SARIMA (guaranteed non-empty) ─────────────────────────────────
        if use_sarima:
            sarima_result: Optional[np.ndarray] = None

            # Primary path: real SARIMA model
            if self.sarima is not None:
                try:
                    exog = self._build_sarima_exog(recent_data, n_steps)
                    raw_pred = self.sarima.predict(n_steps, exog=exog)
                    sarima_result = np.clip(
                        np.asarray(raw_pred, dtype=float).squeeze(), _PM25_MIN, _PM25_MAX
                    )
                    logger.info("SARIMAX predictions: %d steps", len(sarima_result))
                except Exception:
                    logger.exception(
                        "SARIMA primary prediction failed — activating fallback"
                    )

            # Fallback path: trend forecast
            if sarima_result is None or len(np.asarray(sarima_result).ravel()) == 0:
                reason = "model not loaded" if self.sarima is None else "prediction exception"
                logger.warning("SARIMA fallback activated (%s)", reason)
                try:
                    sarima_result = self._fallback_sarima_forecast(recent_data, n_steps)
                    logger.info(
                        "Fallback SARIMA: %d steps  mean=%.2f",
                        len(sarima_result), float(np.mean(sarima_result)),
                    )
                except Exception:
                    logger.exception("Fallback SARIMA also failed — using zeros")
                    sarima_result = np.zeros(n_steps, dtype=float)

            predictions["sarima"] = np.asarray(sarima_result, dtype=float).ravel()

        # ── Hybrid ────────────────────────────────────────────────────────
        if use_hybrid and "gru" in predictions and "sarima" in predictions:
            try:
                y_hybrid = self.hybrid.blend(predictions["gru"], predictions["sarima"])
                predictions["hybrid"] = np.clip(
                    np.asarray(y_hybrid, dtype=float).squeeze(), _PM25_MIN, _PM25_MAX
                )
                logger.info("Hybrid prediction: %.2f ug/m3", float(np.ravel(y_hybrid)[0]))
            except Exception:
                logger.exception("Hybrid blend failed")

        # ── summary log ───────────────────────────────────────────────────
        for key, arr in predictions.items():
            arr_r = np.asarray(arr).ravel()
            if len(arr_r) > 0:
                logger.info(
                    "predictions['%s']: len=%d  min=%.3f  max=%.3f  mean=%.3f",
                    key, len(arr_r), arr_r.min(), arr_r.max(), arr_r.mean(),
                )
            else:
                logger.warning("predictions['%s']: EMPTY", key)

        return predictions

    def predict_hourly(self, recent_data: pd.DataFrame) -> Dict[str, float]:
        """Single-step forecast convenience wrapper."""
        preds = self.predict(recent_data, n_steps=1)
        return {
            model: float(np.asarray(pred).ravel()[0])
            for model, pred in preds.items()
            if len(np.asarray(pred).ravel()) > 0
        }