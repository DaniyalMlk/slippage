"""Benchmark prices an execution can be scored against.

The window used for an interval benchmark defaults to the *life of the order*:
from arrival until the end of the bar containing the last fill. Scoring against
a full-day VWAP when the order only traded for twenty minutes flatters or
punishes the execution for hours it had no part in, which is the most common
way a VWAP number ends up meaningless.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .exceptions import ValidationError
from .series import BarSeries
from .types import Order

__all__ = ["Benchmark", "benchmark_price", "order_window"]


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
