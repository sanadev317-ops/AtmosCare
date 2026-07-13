# ============================================================================
# DATA LOADING MODULE
# ============================================================================
"""Data loading and basic preprocessing."""

import os
from typing import Dict

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and validate data from various sources."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get("data", {})
        self.city_col = self.data_config.get("city_column", "city")
        self.target = config.get("features", {}).get("target", "pm2_5")
        self.required_cols = ["timestamp", "city", "pm2_5", "pm10"]

    def load_data(self) -> pd.DataFrame:
        """
        Load data from primary source or fallback.

        Returns:
            DataFrame indexed by timestamp.
        """
        primary_path = self.data_config.get("raw_path")
        fallback_path = self.data_config.get("fallback_path")
        dataset_name = os.path.basename(primary_path) if primary_path else "primary dataset"

        if primary_path and os.path.exists(primary_path):
            logger.info(f"Loading primary dataset: {primary_path}")
            try:
                df = self._load_and_preprocess(primary_path)
                logger.info("Primary dataset loaded successfully")
                logger.info(f"Using {dataset_name}")
                logger.info(f"Shape: {df.shape}")
                return df
            except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
                logger.warning(f"Primary load failed: {exc}. Trying fallback...")
        else:
            logger.warning("Primary dataset file not found. Trying fallback...")

        if fallback_path and os.path.exists(fallback_path):
            logger.info(f"Loading fallback dataset: {fallback_path}")
            df = self._load_and_preprocess(fallback_path)
            logger.info(f"Fallback data loaded: {df.shape}")
            return df

        raise FileNotFoundError(
            f"No dataset found.\nPrimary: {primary_path}\nFallback: {fallback_path}"
        )

    def _load_and_preprocess(self, path: str) -> pd.DataFrame:
        """Load and apply basic preprocessing."""
        try:
            df = pd.read_csv(path, on_bad_lines="error")
        except TypeError:
            df = pd.read_csv(path)

        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(".", "", regex=False)
        )

        print(df.columns.tolist())
        print(df.shape)
        logger.info(f"Normalized columns: {df.columns.tolist()}")
        logger.info(f"Normalized shape: {df.shape}")

        self._validate_required_columns(df)

        df = self._ensure_datetime_index(df)
        df = self._apply_sanity_checks(df)
        df = self._fill_missing_values(df)

        logger.info(f"Preprocessed shape: {df.shape}")
        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Raise a clear error if core columns are missing after normalization."""
        missing = [col for col in self.required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                "Primary dataset is missing required columns after normalization: "
                f"{missing}. Available columns: {df.columns.tolist()}"
            )

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a timestamp index, chronologically sorted."""
        if "timestamp" not in df.columns:
            raise ValueError(
                "Missing required 'timestamp' column after normalization. "
                f"Available columns: {df.columns.tolist()}"
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[~df["timestamp"].isna()].set_index("timestamp").sort_index()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()

        logger.info(f"DateTime range: {df.index.min()} -> {df.index.max()}")
        return df

    def _apply_sanity_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physical sanity checks."""
        before = len(df)

        if "pm2_5" in df.columns:
            df = df[df["pm2_5"] >= 0]

        if "humidity" in df.columns:
            df = df[df["humidity"].between(0, 100)]

        if "temperature" in df.columns:
            df = df[df["temperature"].between(-25, 60)]

        removed = before - len(df)
        if removed > 0:
            logger.info(f"Sanity check removed {removed} rows")

        if "co" in df.columns and pd.api.types.is_numeric_dtype(df["co"]) and df["co"].mean() > 50:
            df["co"] = df["co"] / 1000.0
            logger.info("CO converted ug/m3 -> mg/m3")

        if "pm10" in df.columns:
            cap99 = float(df["pm10"].quantile(0.99))
            n_cap = int((df["pm10"] > cap99).sum())
            df["pm10"] = df["pm10"].clip(upper=cap99)
            if n_cap > 0:
                logger.info(f"PM10 capped at 99th percentile ({cap99:.1f}), {n_cap} rows clipped")

        if "pm2_5" in df.columns:
            cap99 = float(df["pm2_5"].quantile(0.99))
            n_cap = int((df["pm2_5"] > cap99).sum())
            df["pm2_5"] = df["pm2_5"].clip(upper=cap99)
            if n_cap > 0:
                logger.info(f"PM2.5 capped at 99th percentile ({cap99:.1f}), {n_cap} rows clipped")

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values per city."""
        if self.city_col in df.columns and len(df[self.city_col].unique()) > 1:
            def _fill_city(g):
                city = g.name
                g = g.sort_index()
                g = g.asfreq("h")
                g[self.city_col] = city
                g = g.ffill(limit=3)
                num_cols = g.select_dtypes(include=[np.number]).columns
                g[num_cols] = g[num_cols].interpolate(method="time", limit=6)
                return g

            df = df.groupby(self.city_col, group_keys=False).apply(_fill_city, include_groups=False)
        else:
            df = df.sort_index().asfreq("h")
            df = df.ffill(limit=3)
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].interpolate(method="time", limit=6)

        df = df.dropna(subset=[self.target])

        logger.info(f"After filling: {df.shape}, missing: {df.isna().sum().sum()}")
        return df

    def get_city_data(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """Extract data for a specific city."""
        if self.city_col not in df.columns:
            return df

        city_df = df[df[self.city_col] == city].copy()
        if city_df.empty:
            logger.warning(f"No data for city: {city}")
        else:
            logger.info(f"Extracted {len(city_df)} rows for {city}")
        return city_df

    def get_city_list(self, df: pd.DataFrame) -> list:
        """Get list of unique cities."""
        if self.city_col not in df.columns:
            return []

        cities = sorted(df[self.city_col].unique().tolist())
        logger.info(f"Cities found: {cities}")
        return cities
