# ============================================================================
# SARIMA MODEL MODULE
# ============================================================================
"""SARIMA training and inference for PM2.5 forecasting."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from ..utils import ensure_dir, get_logger

logger = get_logger(__name__)

DEFAULT_FEATURE_SET = [
    "no2",
    "so2",
    "co",
    "o3",
    "temperature_2m_mean",
    "wind_speed_10m_max",
    "crop_burning_intensity",
    "traffic_combined",
    "stagnation_index",
    "pressure_msl_mean",
]
DEFAULT_TARGET = "pm2_5"

# Physical upper bound for PM2.5 (µg/m³). Values above this indicate a
# runaway SARIMA forecast and will be clipped at inference time.
PM25_PHYSICAL_MAX = 500.0
PM25_PHYSICAL_MIN = 0.0


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


class SARIMAModel:
    """Encapsulate SARIMA training, persistence, and prediction.

    Key design decisions
    --------------------
    * Lag features (pm_lag1, pm_lag3, pm_lag7) are part of the exog matrix so
      that the scaler sees them during training.  At inference the caller must
      supply those lags from the *actual* most-recent PM2.5 values – not from
      the training set tail.
    * Predictions are clipped to [PM25_PHYSICAL_MIN, PM25_PHYSICAL_MAX] to
      guard against diverging seasonal extrapolation.
    * ``prepare_inference_exog`` is a dedicated helper that correctly builds
      the exog block for forecasting without touching the training scaler.
    """

    # ------------------------------------------------------------------ #
    #  Column layout expected by the scaler / model                        #
    # ------------------------------------------------------------------ #
    # These are appended to DEFAULT_FEATURE_SET in prepare_dataset.
    LAG_COLUMNS: List[str] = ["pm_lag1", "pm_lag3", "pm_lag7"]

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        sarima_cfg = config.get("sarima_model", {})

        self.order = tuple(sarima_cfg.get("order", [2, 1, 2]))
        self.seasonal_order = tuple(sarima_cfg.get("seasonal_order", [0, 1, 1, 7]))
        self.enforce_stationarity = bool(sarima_cfg.get("enforce_stationarity", False))
        self.enforce_invertibility = bool(sarima_cfg.get("enforce_invertibility", False))
        self.maxiter = int(sarima_cfg.get("maxiter", 300))
        self.method = sarima_cfg.get("method", "lbfgs")

        self.model = None
        self.results = None

        # Stored feature column order so inference can reproduce the exact
        # column layout the scaler was fitted on.
        self._feature_columns: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, y_train: pd.Series, exog: pd.DataFrame) -> "SARIMAXResults":
        if y_train is None or exog is None:
            raise ValueError("Both y_train and exog must be provided to fit SARIMA.")

        # Remember column order for inference validation.
        self._feature_columns = list(exog.columns)

        self.model = SARIMAX(
            y_train,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.results = self.model.fit(method=self.method, maxiter=self.maxiter, disp=False)
        logger.info(
            "SARIMA trained: order=%s seasonal_order=%s maxiter=%d method=%s features=%s",
            self.order,
            self.seasonal_order,
            self.maxiter,
            self.method,
            self._feature_columns,
        )
        return self.results

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, steps: int, exog: pd.DataFrame) -> pd.Series:
        """Return *clipped* PM2.5 forecasts for ``steps`` future periods.

        Parameters
        ----------
        steps:
            Number of future time steps to forecast.
        exog:
            Future exogenous features, shape (steps, n_features).  Columns
            must match the layout used during training (see
            ``_feature_columns``).

        Returns
        -------
        pd.Series
            Forecast values clipped to physical bounds.
        """
        if self.results is None:
            raise RuntimeError("SARIMA model is not fitted or loaded.")

        # ---- column-order / shape validation --------------------------------
        if self._feature_columns is not None:
            missing = [c for c in self._feature_columns if c not in exog.columns]
            if missing:
                raise ValueError(
                    f"SARIMA inference exog is missing columns: {missing}. "
                    f"Expected: {self._feature_columns}"
                )
            # Reorder to match training layout exactly.
            exog = exog[self._feature_columns]

        if len(exog) != steps:
            raise ValueError(
                f"exog has {len(exog)} rows but steps={steps}. They must match."
            )

        # ---- log stats so mismatches surface in logs ------------------------
        logger.info(
            "SARIMA exog stats (mean): %s",
            exog.mean().round(3).to_dict(),
        )
        logger.info(
            "SARIMA exog stats (std):  %s",
            exog.std().round(3).to_dict(),
        )

        forecast = self.results.get_forecast(steps=steps, exog=exog)
        raw = forecast.predicted_mean

        clipped = raw.clip(lower=PM25_PHYSICAL_MIN, upper=PM25_PHYSICAL_MAX)
        n_clipped = int((raw != clipped).sum())
        if n_clipped > 0:
            logger.warning(
                "SARIMA: clipped %d/%d forecast values outside [%.0f, %.0f]. "
                "Raw min=%.2f max=%.2f – check exog distribution shift.",
                n_clipped,
                steps,
                PM25_PHYSICAL_MIN,
                PM25_PHYSICAL_MAX,
                float(raw.min()),
                float(raw.max()),
            )

        return clipped

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> str:
        ensure_dir(os.path.dirname(path) or ".")
        if self.results is None:
            raise RuntimeError("SARIMA model is not fitted.")
        self.results.save(path)
        logger.info("SARIMA model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str, config: Optional[Dict] = None) -> "SARIMAModel":
        if not os.path.exists(path):
            raise FileNotFoundError(f"SARIMA model not found: {path}")

        obj = cls(config)

        # Cross-version pandas pickle compatibility fix.
        _patch_pandas_stringarray_compat()

        obj.results = SARIMAXResults.load(path)
        obj.model = obj.results.model

        # Recover feature column order from the stored model data if available.
        try:
            obj._feature_columns = list(obj.model.data.param_names)
        except Exception:
            pass

        logger.info("SARIMA model loaded from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    #  Dataset helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def prepare_dataset(
        df: pd.DataFrame,
        feature_set: Optional[Tuple[str, ...]] = None,
        target: str = DEFAULT_TARGET,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build (X_exog, y) from a training DataFrame.

        Lag columns (pm_lag1, pm_lag3, pm_lag7) are appended *after*
        ``feature_set`` so the column order is deterministic.
        """
        feature_set = list(feature_set) if feature_set is not None else list(DEFAULT_FEATURE_SET)
        df = df.copy()

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        df[target] = df[target].astype(float)

        # Build lag features from the target column.
        df["pm_lag1"] = df[target].shift(1)
        df["pm_lag3"] = df[target].shift(3)
        df["pm_lag7"] = df[target].shift(7)

        df = df.asfreq("D").ffill().bfill()

        feature_columns = feature_set + SARIMAModel.LAG_COLUMNS

        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required SARIMA features: {missing}")

        X = df[feature_columns].astype(float)
        y = df[target].astype(float)

        logger.info(
            "SARIMA dataset prepared: X=%s y=%s columns=%s",
            X.shape,
            y.shape,
            feature_columns,
        )
        return X, y

    @staticmethod
    def prepare_inference_exog(
        context_df: pd.DataFrame,
        steps: int,
        scaler: StandardScaler,
        feature_set: Optional[List[str]] = None,
        target: str = DEFAULT_TARGET,
        last_known_pm: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Build and scale the exog matrix for forecasting.

        This method is the *correct* way to construct the exog block at
        inference time.  It mirrors ``prepare_dataset`` exactly so that the
        scaler transformation is consistent.

        Parameters
        ----------
        context_df:
            DataFrame containing the most recent rows (at least ``steps``
            rows) with all exog feature columns *and* the target column.
        steps:
            Number of future periods to forecast.
        scaler:
            The ``StandardScaler`` fitted on training exog data.
        feature_set:
            Ordered list of base feature names (defaults to
            ``DEFAULT_FEATURE_SET``).
        target:
            Name of the PM2.5 target column.
        last_known_pm:
            Array of the most-recent actual PM2.5 values used to seed the lag
            features.  Must have at least 7 elements (oldest → newest).  If
            ``None``, lags are derived from ``context_df[target]``.

        Returns
        -------
        pd.DataFrame
            Scaled exog of shape (steps, n_features) ready for
            ``SARIMAModel.predict``.
        """
        feature_set = feature_set if feature_set is not None else list(DEFAULT_FEATURE_SET)

        # Take the last `steps` rows of base features.
        if len(context_df) < steps:
            raise ValueError(
                f"context_df has only {len(context_df)} rows but steps={steps}."
            )
        exog_base = context_df[feature_set].tail(steps).copy().reset_index(drop=True)

        # Seed lag features from real observations, not from model output.
        if last_known_pm is not None:
            pm_arr = np.asarray(last_known_pm, dtype=float)
            if len(pm_arr) < 7:
                raise ValueError(
                    "last_known_pm must have at least 7 elements to seed lag features."
                )
            # For a multi-step forecast we use the *last* known value for all
            # steps (a reasonable approximation; true rolling lags would
            # require iterative forecasting).
            exog_base["pm_lag1"] = pm_arr[-1]
            exog_base["pm_lag3"] = pm_arr[-3]
            exog_base["pm_lag7"] = pm_arr[-7]
        else:
            if target not in context_df.columns:
                raise ValueError(
                    f"Target column '{target}' not found in context_df and "
                    "last_known_pm was not provided."
                )
            pm_tail = context_df[target].tail(steps + 7).values.astype(float)
            exog_base["pm_lag1"] = pm_tail[-steps - 1] if len(pm_tail) > steps else pm_tail[0]
            exog_base["pm_lag3"] = pm_tail[-steps - 3] if len(pm_tail) > steps + 2 else pm_tail[0]
            exog_base["pm_lag7"] = pm_tail[-steps - 7] if len(pm_tail) > steps + 6 else pm_tail[0]

        col_order = feature_set + SARIMAModel.LAG_COLUMNS
        exog_base = exog_base[col_order].astype(float)

        logger.info(
            "Inference exog (pre-scale) stats – mean: %s",
            exog_base.mean().round(3).to_dict(),
        )

        scaled = SARIMAModel.rescale(exog_base, scaler)

        logger.info(
            "Inference exog (post-scale) stats – mean: %s",
            scaled.mean().round(3).to_dict(),
        )

        return scaled

    @staticmethod
    def build_scaler(X_train: pd.DataFrame) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(X_train)
        logger.info(
            "SARIMA scaler fitted on %d samples, %d features: %s",
            len(X_train),
            X_train.shape[1],
            list(X_train.columns),
        )
        return scaler

    @staticmethod
    def rescale(exog: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.transform(exog),
            index=exog.index,
            columns=exog.columns,
        )