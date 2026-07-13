# ============================================================================
# HELPER UTILITIES
# ============================================================================
"""Common utility functions for the pipeline."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from .logger import get_logger
from .tensorflow_runtime import configure_tensorflow_cpu

logger = get_logger(__name__)


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except Exception:
        pass

    env_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(Path(__file__).resolve().parents[2], ".env"),
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        configure_tensorflow_cpu()
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception as exc:
            logger.debug("TensorFlow CPU-only device configuration skipped: %s", exc)
        tf.random.set_seed(seed)
        logger.info(f"TensorFlow seed set to {seed}")
    except ImportError:
        logger.debug("TensorFlow not available")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _load_env_file()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Load optional .env overrides when available.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    database = config.setdefault("mongodb", {})
    database_uri = (
        os.getenv("DATABASE_URI")
        or os.getenv("MONGODB_URI")
        or os.getenv("MONGO_URI")
    )
    database_name = (
        os.getenv("DATABASE_NAME")
        or os.getenv("MONGODB_DATABASE")
        or os.getenv("MONGO_DB")
    )
    database_enabled = os.getenv("DATABASE_ENABLED")
    skip_database_check = os.getenv("SKIP_DATABASE_CHECK")
    skip_mongo_check = os.getenv("SKIP_MONGO_CHECK")

    if database_uri:
        database["uri"] = database_uri
    if database_name:
        database["database"] = database_name
    if database_enabled is not None:
        database["enabled"] = database_enabled.strip().lower() in {"1", "true", "yes", "on"}
    elif skip_database_check is not None:
        database["enabled"] = skip_database_check.strip().lower() not in {"1", "true", "yes", "on"}
    elif skip_mongo_check is not None:
        database["enabled"] = skip_mongo_check.strip().lower() not in {"1", "true", "yes", "on"}

    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_json(data: Dict[str, Any], path: str, indent: int = 2):
    """Save dictionary to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info(f"Saved: {path}")


def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    logger.info(f"Loaded: {path}")
    return data


def save_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    logger.info(f"Saved CSV: {path} ({len(df)} rows)")


def load_csv(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Load CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path, parse_dates=parse_dates, on_bad_lines="skip")
    logger.info(f"Loaded CSV: {path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_aqi(pm25: float) -> int:
    """
    Compute AQI from PM2.5 concentration (µg/m³).
    
    Args:
        pm25: PM2.5 concentration
        
    Returns:
        AQI value
    """
    if pd.isna(pm25):
        return 0
    
    breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500, 301, 500),
    ]
    
    for lo, hi, ilo, ihi in breakpoints:
        if lo <= pm25 <= hi:
            return int(((ihi - ilo) / (hi - lo)) * (pm25 - lo) + ilo)
    
    return 500


def aqi_category(pm25: float) -> str:
    """Get AQI category from PM2.5."""
    breakpoints = {
        "Good": (0, 12),
        "Moderate": (12, 35.4),
        "Unhealthy-SG": (35.4, 55.4),
        "Unhealthy": (55.4, 150.4),
        "Very Unhealthy": (150.4, 250.4),
        "Hazardous": (250.4, 500),
    }
    
    for cat, (lo, hi) in breakpoints.items():
        if lo <= pm25 < hi:
            return cat
    
    return "Hazardous"


def format_metrics(metrics: Dict[str, float], label: str = "Model") -> str:
    """Format metrics dictionary for display."""
    lines = [f"\n{'─'*50}\n  {label}\n{'─'*50}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:>10} : {v:.4f}")
        else:
            lines.append(f"  {k:>10} : {v}")
    lines.append(f"{'─'*50}")
    return "\n".join(lines)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def get_artifact_path(artifact_name: str, config: Dict[str, Any]) -> str:
    """Get path for artifact (model, checkpoint, etc.)."""
    model_dir = config.get("output", {}).get("model_dir", "artifacts/models")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(model_dir, artifact_name)
