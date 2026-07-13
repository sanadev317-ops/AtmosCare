# ============================================================================
# TENSORFLOW RUNTIME CONFIGURATION
# ============================================================================
"""TensorFlow runtime defaults for this project."""

from __future__ import annotations

import os


def configure_tensorflow_cpu() -> None:
    """Force TensorFlow to use CPU and keep accelerator loader noise out of logs."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
