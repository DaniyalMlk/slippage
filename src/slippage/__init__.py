"""Optimal execution and transaction cost analysis."""

from __future__ import annotations

from .benchmarks import Benchmark, Score, benchmark_price, order_window, score_order
from .calibration import (
    Estimate,
    ExecutionSample,
    LinearTemporaryFit,
    PowerLawFit,
    fit_linear_temporary,
    fit_permanent,
    fit_power_law,
    samples_from_orders,
)
from .costs import cost_bps, cost_currency, cost_per_share
from .exceptions import (
    CalibrationError,
    ConvergenceError,
    IdentifiabilityWarning,
    InsufficientDataError,
    NoVolumeError,
    SlippageError,
    ValidationError,
)
from .execution import (
    ExecutionProblem,
    HalfLifeSensitivity,
    Trajectory,
    closed_form_moments,
    efficient_frontier,
    half_life_sensitivity,
    linear_trajectory,
    optimal_trajectory,
    schedule_moments,
    trajectory_from_trades,
)
from .impact import (
    ImpactModel,
    LinearImpact,
    PowerLawImpact,
    ScheduleCost,
    SquareRootLaw,
    schedule_cost,
    uniform_schedule,
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
    "Estimate",
    "ExecutionProblem",
    "ExecutionSample",
    "Fill",
    "FillAttribution",
    "HalfLifeSensitivity",
    "IdentifiabilityWarning",
    "ImpactModel",
    "InsufficientDataError",
    "LinearImpact",
    "LinearTemporaryFit",
    "NoVolumeError",
    "Order",
    "PowerLawFit",
    "PowerLawImpact",
    "ScheduleCost",
    "Score",
    "ShortfallBreakdown",
    "Side",
    "SlippageError",
    "SquareRootLaw",
    "Trajectory",
    "ValidationError",
    "__version__",
    "attribute_fills",
    "benchmark_price",
    "closed_form_moments",
    "cost_bps",
    "cost_currency",
    "cost_per_share",
    "efficient_frontier",
    "fit_linear_temporary",
    "fit_permanent",
    "fit_power_law",
    "half_life_sensitivity",
    "implementation_shortfall",
    "linear_trajectory",
    "optimal_trajectory",
    "order_window",
    "participation_rate",
    "samples_from_orders",
    "schedule_cost",
    "schedule_moments",
    "score_order",
    "shortfall_from_market",
    "trajectory_from_trades",
    "uniform_schedule",
]
