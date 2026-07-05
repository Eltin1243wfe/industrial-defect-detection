"""
Basic logger setup. I got tired of print() everywhere so pulled this out early on.
Writes to console + a log file per run so I can go back and check what happened
after the fact (useful when a training run dies overnight).
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        # avoid duplicate handlers if this gets called more than once for the same name
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}_{timestamp}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
