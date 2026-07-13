# ============================================================================
# LOGGING UTILITY
# ============================================================================
"""Structured logging for the ML pipeline."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


class PipelineLogger:
    """Centralized logger for the ML pipeline."""
    
    _instances = {}
    
    def __init__(self, name: str, log_level: str = "INFO", 
                 log_file: Optional[str] = None, log_dir: str = "logs"):
        """
        Initialize logger.
        
        Args:
            name: Logger name (usually __name__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            return

        # Windows consoles often default to a legacy code page.
        # Reconfigure the standard streams so UTF-8 messages do not crash logging.
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass
        
        # Console handler
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if path provided)
        if log_file:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10485760,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls, name: str, log_level: str = "INFO",
                   log_file: Optional[str] = None, log_dir: str = "logs"):
        """Get or create logger instance (singleton pattern)."""
        if name not in cls._instances:
            cls._instances[name] = cls(name, log_level, log_file, log_dir)
        return cls._instances[name].logger
    
    def get(self):
        """Return the logger instance."""
        return self.logger


def get_logger(name: str, log_level: str = "INFO", 
               log_file: Optional[str] = "pipeline.log") -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Log file name
        
    Returns:
        Logger instance
    """
    return PipelineLogger.get_logger(name, log_level, log_file)
