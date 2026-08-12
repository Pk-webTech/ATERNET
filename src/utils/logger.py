"""
Project-wide logger factory. Writes to both console and logs/aternet.log.
"""

import logging
from pathlib import Path

from config.logging_config import LOG_LEVEL, LOG_FORMAT, DATE_FORMAT, LOG_FILE_NAME
from config.paths import LOGS_DIR

_CONFIGURED_LOGGERS = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(LOGS_DIR) / LOG_FILE_NAME)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If filesystem isn't writable for some reason, fall back to console-only.
        pass

    _CONFIGURED_LOGGERS.add(name)
    return logger
