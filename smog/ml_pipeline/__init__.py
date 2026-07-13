# ============================================================================
# ML Pipeline Package
# ============================================================================
"""Production-grade ML pipeline for smog (PM2.5) prediction."""

from .utils.tensorflow_runtime import configure_tensorflow_cpu

configure_tensorflow_cpu()

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

__all__ = [
    "data",
    "models",
    "inference",
    "services",
    "api",
    "config",
    "utils",
]


def __getattr__(name):
    """Lazily import subpackages to avoid pulling heavy optional deps at import time."""
    if name in __all__:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
