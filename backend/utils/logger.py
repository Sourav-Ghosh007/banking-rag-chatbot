"""
Logger Setup
============
Simple explanation:
- Instead of every file doing its own logging setup, we configure it once here
- Every log line will show: timestamp | level | filename | message
- Example: 2025-06-01 10:23:45 | INFO | chat.py | User asked about loan EMI

Just call: logger = logging.getLogger(__name__) in any file
Then call setup_logging() once in main.py at startup
"""

import logging
import sys
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the entire application.
    Call this once in main.py when the app starts.

    log_level: "DEBUG", "INFO", "WARNING", "ERROR"
    """
    # Log format: time | level | file:line | message
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Get the numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),   # print to terminal
        ],
    )

    # Quieten noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging configured at level: %s", log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function — get a logger for a module.
    Usage: logger = get_logger(__name__)
    """
    return logging.getLogger(name)
