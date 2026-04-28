import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, MutableMapping

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

@contextmanager
def Timer(
    label: str,
    sink: MutableMapping[str, float] | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[None]:
    
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if sink is not None:
            sink[label] = round(elapsed, 4)
        if logger is not None:
            logger.info("[timing] %-12s %.3fs", label, elapsed)

def project_root() -> Path:
   
    return Path(__file__).resolve().parent.parent

def ensure_dir(path: Path) -> Path:
   
    path.mkdir(parents=True, exist_ok=True)
    return path