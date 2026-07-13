#!/usr/bin/env python
# ============================================================================
# FASTAPI SERVER
# ============================================================================
"""
Run the FastAPI inference server.

Usage:
    python ml_pipeline/run_api.py
    
    Or with custom config/models:
    python ml_pipeline/run_api.py --config path/to/config.yaml --models path/to/models
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from ml_pipeline.api import create_app
from ml_pipeline.utils import get_logger

logger = get_logger(__name__)


def main():
    """Run FastAPI server."""
    parser = argparse.ArgumentParser(description="Smog Prediction API Server")
    parser.add_argument(
        "--config",
        type=str,
        default="ml_pipeline/config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="artifacts/models",
        help="Path to models directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers"
    )
    
    args = parser.parse_args()
    
    logger.info("Creating FastAPI application...")
    app = create_app(config_path=args.config, model_run_dir=args.models)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Docs available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
