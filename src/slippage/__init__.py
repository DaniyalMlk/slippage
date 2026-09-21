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
from .shortfall import (
    DelayBasis,
    FillAttribution,
    ShortfallBreakdown,
    attribute_fills,
    implementation_shortfall,
    participation_rate,
    shortfall_from_market,
)
from .types import Bar, Fill, Order, Side

__version__ = "0.1.0"

__all__ = [
    "Bar",
    "BarSeries",
    "Benchmark",
    "CalibrationError",
    "ConvergenceError",
    "DelayBasis",
    "Fill",
    "FillAttribution",
    "InsufficientDataError",
    "NoVolumeError",
    "Order",
    "Score",
    "ShortfallBreakdown",
    "Side",
    "SlippageError",
    "ValidationError",
    "__version__",
    "attribute_fills",
    "benchmark_price",
    "cost_bps",
    "cost_currency",
    "cost_per_share",
    "implementation_shortfall",
    "order_window",
    "participation_rate",
    "score_order",
    "shortfall_from_market",
]
