import sys
import json
import psutil
import logging
from typing import Optional
from datetime import datetime


class MemoryFormatter(logging.Formatter):
    """Custom formatter that includes memory usage in every log message."""

    def format(self, record):
        # Get memory info
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        memory_percent = memory.percent

        # Add memory info to the record
        record.memory_mb = f"{memory_mb:.0f}MB"
        record.memory_percent = f"{memory_percent:.1f}%"

        return super().format(record)


def setup_logger(
    name: Optional[str] = None, level: str = "INFO", format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger that works well with Google Cloud Logging.
    Cloud Run automatically captures stdout/stderr and sends to Cloud Logging

    Args:
        name: Logger name (defaults to calling module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string

    Return:
        Configured logger instance
    """
    # Default format that includes memory usage
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[MEM: %(memory_mb)s/%(memory_percent)s] - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )

    # Create logger
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create console handler that writes to stdout (Cloud Run captures this)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Create custom formatter with memory usage
    formatter = MemoryFormatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False

    return logger


class StructuredLogger:
    """
    Structured logger that outputs JSON logs.
    Useful for advanced log analysis in Google Cloud Logging.
    """
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = setup_logger(name, level, "%(message)s")

    def _get_memory_info(self):
        """Get current memory usage."""
        memory = psutil.virtual_memory()
        return {
            "memory_used_mb": round(memory.used / 1024 / 1024, 1),
            "memory_percent": round(memory.percent, 1),
            "memory_available_mb": round(memory.available / 1024 / 1024, 1)
        }

    def _log(self, level: str, message: str, **kwargs):
        """Log a structured message."""
        log_entry = {
            "timestamp": datetime.now().isoformat() + "Z",
            "severity": level.upper(),
            "message": message,
            **self._get_memory_info(),  # Always include memory info
            **kwargs,
        }

        # Log as JSON string
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))

    def info(self, message: str, **kwargs):
        self._log("info", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("error", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("warning", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("debug", message, **kwargs)


default_logger = setup_logger("cloudrun", level="INFO")


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a logger for a specific module."""
    return setup_logger(name, level)
