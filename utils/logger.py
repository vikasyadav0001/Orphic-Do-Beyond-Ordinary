import logging
import sys


def get_logger(name:str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicates)
    logger.propagate = False

    return logger