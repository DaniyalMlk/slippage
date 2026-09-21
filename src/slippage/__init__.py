"""Optimal execution and transaction cost analysis."""

from __future__ import annotations

from .benchmarks import Benchmark, Score, benchmark_price, order_window, score_order
from .costs import cost_bps, cost_currency, cost_per_share
from .exceptions import (
    CalibrationError,
    ConvergenceError,
    InsufficientDataError,
    NoVolumeError,
    SlippageError,
    ValidationError,
)
from .series import BarSeries
from .types import Bar, Fill, Order, Side

__version__ = "0.1.0"

__all__ = [
    "Bar",
    "BarSeries",
    "Benchmark",
    "CalibrationError",
    "ConvergenceError",
    "Fill",
    "InsufficientDataError",
    "NoVolumeError",
    "Order",
    "Score",
    "Side",
    "SlippageError",
    "ValidationError",
    "__version__",
    "benchmark_price",
    "cost_bps",
    "cost_currency",
    "cost_per_share",
    "order_window",
    "score_order",
]
