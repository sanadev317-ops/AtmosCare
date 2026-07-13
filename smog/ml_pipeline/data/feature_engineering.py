# ============================================================================

# FEATURE ENGINEERING MODULE (IMPROVED)

# ============================================================================

"""Feature engineering with data leakage prevention and improved signal strength."""

from typing import Dict, List
import numpy as np
import pandas as pd

from ..utils import get_logger, compute_aqi

logger = get_logger(__name__)

class FeatureEngineer:

    def __init__(self, config: Dict):
        self.config = config
        self.feature_config = config.get("features", {})

        self.target = self.feature_config.get("target", "pm2_5")
        self.city_col = config.get("data", {}).get("city_column", "city")

        self.lag_steps = self.feature_config.get("lag_steps", [1, 3, 6, 12, 24])
        self.rolling_windows = self.feature_config.get("rolling_windows", [6, 12, 24])
        self.weekly_fourier_order = int(self.feature_config.get("weekly_fourier_order", 3))

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Starting feature engineering: {df.shape}")

        df = self._add_cyclical_features(df)
        df = self._add_domain_flags(df)
        df = self._add_regime_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_interaction_terms(df)

        if "AQI" not in df.columns:
            df["AQI"] = df[self.target].apply(compute_aqi)

        df = self._drop_engineering_nans(df)

        logger.info(f"Feature engineering complete: {df.shape}")
        return df

    # =========================================================================
    # TEMPORAL FEATURES (IMPROVED)
    # =========================================================================
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding temporal features")

        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        # cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # explicit categorical signal (IMPORTANT)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # weekly Fourier terms to help capture 7-day smog patterns
        hour_of_week = df.index.dayofweek * 24 + df.index.hour
        for k in range(1, self.weekly_fourier_order + 1):
            df[f"week_sin_{k}"] = np.sin(2 * np.pi * k * hour_of_week / 168.0)
            df[f"week_cos_{k}"] = np.cos(2 * np.pi * k * hour_of_week / 168.0)

        return df

    # =========================================================================
    # REGIME FEATURES
    # =========================================================================
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding regime awareness features")

        humidity = df["humidity"] if "humidity" in df.columns else pd.Series(0.0, index=df.index)
        wind_speed = df["wind_speed"] if "wind_speed" in df.columns else pd.Series(0.0, index=df.index)
        precipitation = df["precipitation"] if "precipitation" in df.columns else pd.Series(0.0, index=df.index)

        humidity = humidity.fillna(0.0)
        wind_speed = wind_speed.fillna(0.0)
        precipitation = precipitation.fillna(0.0)

        rain_dispersion = (precipitation > 0.1).astype(int)
        smog_trapped = (
            (humidity >= 70.0) &
            (wind_speed <= 2.5) &
            (precipitation <= 0.1)
        ).astype(int)
        dry_stable = (
            (humidity <= 45.0) &
            (wind_speed >= 3.0) &
            (precipitation <= 0.1)
        ).astype(int)

        regime_code = np.zeros(len(df), dtype=int)
        regime_code = np.where(rain_dispersion.values == 1, 1, regime_code)
        regime_code = np.where(smog_trapped.values == 1, 2, regime_code)
        regime_code = np.where(dry_stable.values == 1, 3, regime_code)

        df["regime_rain_dispersion"] = rain_dispersion
        df["regime_smog_trapped"] = smog_trapped
        df["regime_dry_stable"] = dry_stable
        df["regime_transition"] = ((regime_code == 0) & (precipitation <= 0.1)).astype(int)
        df["regime_code"] = regime_code

        return df

    # =========================================================================
    # DOMAIN FLAGS
    # =========================================================================
    def _add_domain_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding domain flags")

        df["smog_season"] = df.index.month.isin([10, 11, 12, 1, 2]).astype(int)

        df["rush_hour"] = (
            df.index.hour.isin(range(8, 11)) |
            df.index.hour.isin(range(17, 21))
        ).astype(int)

        return df

    # =========================================================================
    # LAG FEATURES (IMPROVED)
    # =========================================================================
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding lag features")

        if self.city_col in df.columns and df[self.city_col].nunique() > 1:
            grouper = df.groupby(self.city_col, sort=False)

            # pm2_5 lags
            for lag in self.lag_steps:
                df[f"pm2_5_lag{lag}"] = grouper[self.target].shift(lag)

            # exogenous lags
            exog_cols = ["temperature", "humidity", "wind_speed", "pressure"]

            for col in exog_cols:
                if col in df.columns:
                    df[f"{col}_lag6"] = grouper[col].shift(6)
                    df[f"{col}_lag12"] = grouper[col].shift(12)

        else:
            for lag in self.lag_steps:
                df[f"pm2_5_lag{lag}"] = df[self.target].shift(lag)

            exog_cols = ["temperature", "humidity", "wind_speed", "pressure"]

            for col in exog_cols:
                if col in df.columns:
                    df[f"{col}_lag6"] = df[col].shift(6)
                    df[f"{col}_lag12"] = df[col].shift(12)

        return df

    # =========================================================================
    # ROLLING FEATURES (REDUCED NOISE)
    # =========================================================================
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding rolling features")

        if self.city_col in df.columns and df[self.city_col].nunique() > 1:

            def _rolling_stats(g):
                for window in self.rolling_windows:
                    g[f"pm2_5_roll{window}_mean"] = g[self.target].rolling(window, min_periods=1).mean()
                    g[f"pm2_5_roll{window}_std"] = g[self.target].rolling(window, min_periods=1).std()
                return g

            df = df.groupby(self.city_col, group_keys=False, sort=False).apply(
                _rolling_stats,
                include_groups=False,
            )

        else:
            for window in self.rolling_windows:
                df[f"pm2_5_roll{window}_mean"] = df[self.target].rolling(window, min_periods=1).mean()
                df[f"pm2_5_roll{window}_std"] = df[self.target].rolling(window, min_periods=1).std()

        return df

    # =========================================================================
    # INTERACTIONS
    # =========================================================================
    def _add_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding interaction terms")

        if "humidity" in df.columns:
            df["humidity_x_pm25"] = (df["humidity"] * df[self.target]) / 100

        if "wind_speed" in df.columns:
            ws = df["wind_speed"].clip(lower=0.1)
            df["wind_x_pm25"] = ws * df[self.target]

            if "humidity" in df.columns:
                df["stagnation_idx"] = df["humidity"] / ws

        if "nh3" in df.columns:
            df["nh3_x_pm25"] = df["nh3"] * df[self.target]

            if "temperature" in df.columns:
                df["temp_x_nh3"] = df["temperature"] * df["nh3"]

        if "temperature" in df.columns and "so2" in df.columns:
            df["temp_x_so2"] = df["temperature"] * df["so2"]

        if "pressure" in df.columns:
            df["pressure_dev"] = df["pressure"] - 1013.0

        if "humidity" in df.columns:
            df["humidity_sq"] = df["humidity"] ** 2

        if "wind_speed" in df.columns:
            df["wind_speed_sq"] = df["wind_speed"] ** 2

        if "temperature" in df.columns:
            df["temperature_sq"] = df["temperature"] ** 2

        return df

    # =========================================================================
    # CLEANUP
    # =========================================================================
    def _drop_engineering_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        na_cols = (
            [f"pm2_5_lag{lag}" for lag in self.lag_steps] +
            [f"pm2_5_roll{w}_{stat}" for w in self.rolling_windows for stat in ["mean", "std"]]
        )

        na_cols = [c for c in na_cols if c in df.columns]

        before = len(df)
        df = df.dropna(subset=na_cols + [self.target])
        after = len(df)

        logger.info(f"Dropped {before - after} rows with NaN in engineered features")
        return df

    # =========================================================================
    # FEATURE LIST
    # =========================================================================
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        exclude = [self.city_col, self.target, "AQI", "datetime"]
        features = [c for c in df.columns if c not in exclude]

        logger.info(f"Final features ({len(features)}): {features[:10]}...")
        return features
