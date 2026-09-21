"""Benchmark prices an execution can be scored against.

The window used for an interval benchmark defaults to the *life of the order*:
from arrival until the end of the bar containing the last fill. Scoring against
a full-day VWAP when the order only traded for twenty minutes flatters or
punishes the execution for hours it had no part in, which is the most common
way a VWAP number ends up meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .costs import cost_bps, cost_currency
from .exceptions import ValidationError
from .series import BarSeries
from .types import Order

__all__ = ["Benchmark", "Score", "benchmark_price", "order_window", "score_order"]


class Benchmark(Enum):
    """Reference prices supported by :func:`benchmark_price`."""

    DECISION = "decision"
    """Price when the trade was decided; the reference for total shortfall."""

    ARRIVAL = "arrival"
    """Price when the order reached the market; the reference for trading cost."""

    INTERVAL_VWAP = "interval_vwap"
    """Volume-weighted average price over the order's life."""

    INTERVAL_TWAP = "interval_twap"
    """Time-weighted average price over the order's life."""

    CLOSE = "close"
    """Close of the last bar in the window."""

    OPEN = "open"
    """Open of the first bar in the window."""

    @property
    def is_interval(self) -> bool:
        """Whether the benchmark depends on a window rather than a single moment."""
        return self in {
            Benchmark.INTERVAL_VWAP,
            Benchmark.INTERVAL_TWAP,
            Benchmark.CLOSE,
            Benchmark.OPEN,
        }


def order_window(order: Order, series: BarSeries) -> tuple[datetime, datetime]:
    """The half-open window over which an order was live.

    Starts at arrival. Ends at the end of the bar containing the last fill, so
    that bar is included by the half-open windowing rule. An order with no
    fills is scored over the remainder of the series, which is the interval it
    failed to trade in.
    """
    start = order.arrival_time
    last = order.last_fill_time
    end = series.end if last is None else last + series.bar_duration
    if end <= start:
        end = start + series.bar_duration
    return start, end


def benchmark_price(
    order: Order,
    series: BarSeries,
    benchmark: Benchmark = Benchmark.ARRIVAL,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> float:
    """Evaluate ``benchmark`` for ``order`` against ``series``.

    ``start`` and ``end`` override the default order window; supplying them for
    a point-in-time benchmark such as arrival has no effect.
    """
    default_start, default_end = order_window(order, series)
    window_start = default_start if start is None else start
    window_end = default_end if end is None else end

    if benchmark is Benchmark.DECISION:
        if order.decision_price is not None:
            return order.decision_price
        return series.price_at(order.decision_time)
    if benchmark is Benchmark.ARRIVAL:
        return series.price_at(order.arrival_time)
    if benchmark is Benchmark.INTERVAL_VWAP:
        return series.vwap(window_start, window_end)
    if benchmark is Benchmark.INTERVAL_TWAP:
        return series.twap(window_start, window_end)
    if benchmark is Benchmark.CLOSE:
        return series.close_price(window_start, window_end)
    if benchmark is Benchmark.OPEN:
        return series.open_price(window_start, window_end)
    raise ValidationError(f"unsupported benchmark {benchmark!r}")


@dataclass(frozen=True)
class Score:
    """An executed order measured against one benchmark.

    Only the *filled* quantity is scored here. What happened to the unfilled
    remainder is opportunity cost, which belongs to implementation shortfall
    and needs a decision price and a final price that a single benchmark
    comparison does not have.
    """

    benchmark: Benchmark
    benchmark_price: float
    average_price: float
    filled_quantity: float
    cost_bps: float
    cost_currency: float


def score_order(
    order: Order,
    series: BarSeries,
    benchmark: Benchmark = Benchmark.ARRIVAL,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Score:
    """Score the filled part of ``order`` against ``benchmark``.

    Raises :class:`~slippage.exceptions.ValidationError` for an order with no
    fills: there is no execution to score.
    """
    reference = benchmark_price(order, series, benchmark, start=start, end=end)
    average = order.average_price
    return Score(
        benchmark=benchmark,
        benchmark_price=reference,
        average_price=average,
        filled_quantity=order.filled_quantity,
        cost_bps=cost_bps(order.side, average, reference),
        cost_currency=cost_currency(order.side, order.filled_quantity, average, reference),
    )
