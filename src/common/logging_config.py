"""
Centralized logging configuration for the project.
This module provides a consistent logging setup across all Python files.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: str = None,
    log_format: str = None
) -> logging.Logger:
    """
    Set up and return a logger with consistent configuration.

    Args:
        name: Name of the logger (typically __name__ from the calling module)
        level: Logging level (default: logging.INFO)
        log_file: Optional path to log file. If None, logs only to console.
        log_format: Custom log format string. If None, uses default format.

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger with the default configuration.
    Convenience function for simple use cases.

    Args:
        name: Name of the logger (typically __name__ from the calling module)

    Returns:
        Configured logger instance
    """
    return setup_logger(name=name)


# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = None  # Set to a path like 'logs/app.log' to enable file logging
