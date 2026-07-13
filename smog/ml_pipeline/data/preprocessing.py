# ============================================================================

# DATA PREPROCESSING MODULE (CORRECTED)

# ============================================================================

"""Data preprocessing and scaling."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.feature_config = config.get("features", {})
        self.target = self.feature_config.get("target", "pm2_5")

        # 🔥 robust handling (lowercase safe)
        self.city_col = config.get("data", {}).get("city_column", "city")

        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.feature_cols = []
        self.city_ohe_cols = []
        self.cities = []
        self.fitted = False

    @classmethod
    def from_scaler(
        cls,
        config: Dict,
        scaler: StandardScaler,
        numeric_cols: list,
        feature_cols: Optional[list] = None,
        cities: Optional[list] = None,
        fitted: bool = True,
        target: Optional[str] = None,
    ) -> "DataPreprocessor":
        """Build a lightweight preprocessor wrapper from a saved scaler artifact."""
        obj = cls(config)
        obj.scaler = scaler
        scaler_features = list(getattr(scaler, "feature_names_in_", []))
        obj.numeric_cols = scaler_features if scaler_features else list(numeric_cols)
        obj.feature_cols = list(feature_cols) if feature_cols is not None else list(obj.numeric_cols)
        obj.city_ohe_cols = []
        obj.cities = list(cities) if cities is not None else []
        obj.fitted = fitted
        if target is not None:
            obj.target = target
        return obj
    
    # =========================================================================
    # SPLIT
    # =========================================================================
    def split_data(
        self,
        df: pd.DataFrame,
        strategy: str = "chronological",
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df = df.sort_index()

        if self.city_col in df.columns and df[self.city_col].nunique() > 1:
            train_parts, val_parts, test_parts = [], [], []

            for city, g in df.groupby(self.city_col, sort=False):
                g = g.sort_index()
                n = len(g)

                train_end = int(n * (1 - test_size - val_size))
                val_end = int(n * (1 - test_size))

                train_parts.append(g.iloc[:train_end])
                val_parts.append(g.iloc[train_end:val_end])
                test_parts.append(g.iloc[val_end:])

            train = pd.concat(train_parts).sort_index()
            val = pd.concat(val_parts).sort_index()
            test = pd.concat(test_parts).sort_index()

            logger.info(
                f"Per-city split: train={len(train)}, val={len(val)}, test={len(test)}"
            )
        else:
            n = len(df)
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))

            train = df.iloc[:train_end]
            val = df.iloc[train_end:val_end]
            test = df.iloc[val_end:]

            logger.info(
                f"Chronological split: train={len(train)}, val={len(val)}, test={len(test)}"
            )

        return train, val, test

    # =========================================================================
    # SCALING
    # =========================================================================
    def scale_data(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # ---------------------------------------------------------
        # Extract city column
        # ---------------------------------------------------------
        if self.city_col in train.columns:
            train_city = train[self.city_col].astype(str)
            val_city = val[self.city_col].astype(str)
            test_city = test[self.city_col].astype(str)

            all_cities = pd.concat([train_city, val_city, test_city]).unique()
            self.cities = sorted([str(c).strip() for c in all_cities if str(c).strip()])

            train_num = train.drop(columns=[self.city_col])
            val_num = val.drop(columns=[self.city_col])
            test_num = test.drop(columns=[self.city_col])
        else:
            train_city = val_city = test_city = None
            self.cities = []
            train_num = train
            val_num = val
            test_num = test

        # ---------------------------------------------------------
        # KEEP ONLY NUMERIC COLUMNS (CRITICAL)
        # ---------------------------------------------------------
        self.numeric_cols = train_num.select_dtypes(include=[np.number]).columns.tolist()

        train_num = train_num[self.numeric_cols]

        # ---------------------------------------------------------
        # ALIGN VAL & TEST TO TRAIN FEATURES (CRITICAL FIX)
        # ---------------------------------------------------------
        val_num = val_num.reindex(columns=self.numeric_cols, fill_value=0)
        test_num = test_num.reindex(columns=self.numeric_cols, fill_value=0)

        # ---------------------------------------------------------
        # CLEAN NaN / INF (IMPORTANT)
        # ---------------------------------------------------------
        train_num = train_num.replace([np.inf, -np.inf], np.nan).fillna(0)
        val_num = val_num.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_num = test_num.replace([np.inf, -np.inf], np.nan).fillna(0)

        # ---------------------------------------------------------
        # FIT SCALER ONLY ON TRAIN
        # ---------------------------------------------------------
        self.scaler.fit(train_num)
        self.fitted = True

        logger.info(f"Scaler fitted on {len(self.numeric_cols)} numeric columns")

        # ---------------------------------------------------------
        # TRANSFORM
        # ---------------------------------------------------------
        train_scaled = self._scale_df(train_num, train_num.index)
        val_scaled = self._scale_df(val_num, val_num.index)
        test_scaled = self._scale_df(test_num, test_num.index)

        # ---------------------------------------------------------
        # ADD CITY ENCODING
        # ---------------------------------------------------------
        if train_city is not None:
            train_scaled = self._add_city_encoding(train_scaled, train_city)
            val_scaled = self._add_city_encoding(val_scaled, val_city)
            test_scaled = self._add_city_encoding(test_scaled, test_city)

        self.feature_cols = list(train_scaled.columns)

        logger.info(f"Scaling complete. Features: {len(self.feature_cols)}")

        return train_scaled, val_scaled, test_scaled

    def _scale_df(self, df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, index=index, columns=self.numeric_cols)

    def _add_city_encoding(self, df: pd.DataFrame, city_series: pd.Series) -> pd.DataFrame:
        ohe = pd.get_dummies(
            pd.Categorical(city_series.astype(str), categories=self.cities),
            prefix="city",
            dtype=float,
        )
        ohe.columns = [c.replace(" ", "_") for c in ohe.columns]
        ohe.index = df.index

        self.city_ohe_cols = list(ohe.columns)

        return pd.concat([df, ohe], axis=1)

    # =========================================================================
    # TRANSFORM (INFERENCE)
    # =========================================================================
    def transform(self, df: pd.DataFrame, city: Optional[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted.")

        df = df.copy()

        if self.city_col in df.columns:
            city_series = df[self.city_col].astype(str)
            df = df.drop(columns=[self.city_col])
        else:
            city_series = pd.Series([city or "Unknown"] * len(df), index=df.index)

        df = df.reindex(columns=self.numeric_cols, fill_value=0.0)

        scaled = self._scale_df(df, df.index)

        if self.cities:
            scaled = self._add_city_encoding(scaled, city_series)

        for col in self.feature_cols:
            if col not in scaled.columns:
                scaled[col] = 0.0

        return scaled[self.feature_cols]

    # =========================================================================
    # INVERSE SCALING (🔥 FIXED PROPERLY)
    # =========================================================================
    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted.")

        if self.target not in self.numeric_cols:
            logger.warning("Target not in numeric columns. Returning as-is.")
            return scaled_values

        target_idx = self.numeric_cols.index(self.target)

        dummy = np.zeros((len(scaled_values), len(self.numeric_cols)))
        dummy[:, target_idx] = scaled_values

        unscaled = self.scaler.inverse_transform(dummy)
        return unscaled[:, target_idx]

    # ✅ NEW: alias methods (fix your pipeline crash)
    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return self.inverse_transform_target(values)

    def inverse_target_scale(self, values: np.ndarray) -> np.ndarray:
        return self.inverse_transform_target(values)

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    def save(self, path: str):
        state = {
            "scaler": self.scaler,
            "numeric_cols": self.numeric_cols,
            "feature_cols": self.feature_cols,
            "city_ohe_cols": self.city_ohe_cols,
            "cities": self.cities,
            "fitted": self.fitted,
            "target": self.target,
        }
        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str, config: Dict) -> "DataPreprocessor":
        state = joblib.load(path)

        obj = cls(config)
        obj.scaler = state["scaler"]
        obj.numeric_cols = state["numeric_cols"]
        obj.feature_cols = state["feature_cols"]
        obj.city_ohe_cols = state.get("city_ohe_cols", [])
        obj.cities = state.get("cities", [])
        obj.fitted = state.get("fitted", True)
        obj.target = state.get("target", "pm2_5")

        logger.info(f"Preprocessor loaded from {path}")
        return obj


