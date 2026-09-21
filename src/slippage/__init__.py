"""Optimal execution and transaction cost analysis."""

from __future__ import annotations

from .exceptions import (
    CalibrationError,
    ConvergenceError,
    InsufficientDataError,
    NoVolumeError,
    SlippageError,
    ValidationError,
)
from .types import Bar, Fill, Order, Side

__version__ = "0.1.0"

__all__ = [
    "Bar",
    "CalibrationError",
    "ConvergenceError",
    "Fill",
    "InsufficientDataError",
    "NoVolumeError",
    "Order",
    "Side",
    "SlippageError",
    "ValidationError",
    "__version__",
]
