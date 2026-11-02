"""Logging configuration for the prunestack framework.

Provides a centralized ``setup_logging`` function that configures
console and file handlers with appropriate formatting.

Usage
-----
::

    from prunestack.utils.logging import setup_logging
    setup_logging(verbose=True, log_file="experiment.log")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal



# Format strings
_CONSOLE_FMT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
_FILE_FMT = "%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    log_file: str | Path | None = None,
    level: int | None = None,
) -> logging.Logger:
    """Configure the prunestack root logger.

    Parameters
    ----------
    verbose : bool
        If True, set console level to DEBUG.
    quiet : bool
        If True, set console level to WARNING.
    log_file : str or Path, optional
        Path to a log file. If provided, a FileHandler is added at DEBUG level.
    level : int, optional
        Explicit log level (overrides verbose/quiet).

    Returns
    -------
    logging.Logger
        The configured ``prunestack`` root logger.
    """
    root = logging.getLogger("prunestack")

    # Determine console level
    if level is not None:
        console_level = level
    elif quiet:
        console_level = logging.WARNING
    elif verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO

    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Replace (and close) handlers from a previous configuration
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # File handler (always DEBUG)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the prunestack namespace.

    Parameters
    ----------
    name : str
        Module or component name, e.g. ``"data.loader"``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(f"prunestack.{name}")
