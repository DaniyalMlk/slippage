"""Benchmark prices an execution can be scored against.

The window used for an interval benchmark defaults to the *life of the order*:
from arrival until the end of the bar containing the last fill. Scoring against
a full-day VWAP when the order only traded for twenty minutes flatters or
punishes the execution for hours it had no part in, which is the most common
way a VWAP number ends up meaningless.

**Whose volume is in it is the other convention, and it is not the window.**
Exchange volume includes the order's own prints, so a VWAP built from those
bars is a benchmark the order helped set. Over any window the market VWAP
decomposes as::

    vwap_market = p * own_average + (1 - p) * vwap_ex_own

with ``p`` the order's share of the window's volume, so::

    own_average - vwap_market = (1 - p) * (own_average - vwap_ex_own)

and a cost measured against the market VWAP is ``(1 - p)`` times the cost
against a benchmark the order had no part in. At 28% participation the number
is 28% too small. At 100% -- a block crossed while nothing else traded -- the
market VWAP *is* the order's own average and the measured cost is exactly zero
whatever was paid. :attr:`Benchmark.INTERVAL_VWAP_EX_OWN` removes the order's
own fills from the benchmark; :func:`own_share` reports the decomposition.

What that correction assumes is worth saying plainly, because only half of it
is certain. The *volume* a bar records does include the order's shares, so
subtracting them from the denominator is right. The *notional* does not exist
in OHLCV data and is estimated here from the bar's typical price, so the
numerator's correction is as good as that estimate and no better. The identity
above then holds exactly as arithmetic -- it is the definition rearranged --
and the size of what it corrects is a fact about the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .costs import cost_bps, cost_currency
from .exceptions import ValidationError
from .series import BarSeries
from .types import Order

__all__ = [
    "Benchmark",
    "OwnShare",
    "Score",
    "benchmark_price",
    "order_window",
    "own_share",
    "score_order",
]


class Benchmark(Enum):
    """Reference prices supported by :func:`benchmark_price`."""

    DECISION = "decision"
    """Price when the trade was decided; the reference for total shortfall."""

    ARRIVAL = "arrival"
    """Price when the order reached the market; the reference for trading cost."""

    INTERVAL_VWAP = "interval_vwap"
    """Volume-weighted average price over the order's life.

    Built from the bars as recorded, so it includes the order's own prints.
    """

    INTERVAL_VWAP_EX_OWN = "interval_vwap_ex_own"
    """The same, with the order's own fills taken out of the benchmark.

    What the rest of the market paid over the window the order traded in. See
    the module documentation for the decomposition and for what the correction
    assumes.
    """

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
            Benchmark.INTERVAL_VWAP_EX_OWN,
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
    end = series.end if last is None else series.end_of_bar_containing(last)
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
    if benchmark is Benchmark.INTERVAL_VWAP_EX_OWN:
        return own_share(order, series, start=window_start, end=window_end).ex_own_vwap
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


@dataclass(frozen=True)
class OwnShare:
    """How much of a window's volume and notional was one order's own.

    Attributes:
        quantity: Shares the order filled inside the window.
        notional: What it paid for them.
        market_quantity: Volume the bars record over the window, which
            includes ``quantity``.
        market_notional: The bars' notional over the window, estimated from
            their typical prices, which includes ``notional`` to the extent
            that estimate is good.
        start: Window start, inclusive.
        end: Window end, exclusive.
    """

    quantity: float
    notional: float
    market_quantity: float
    market_notional: float
    start: datetime
    end: datetime

    @property
    def rate(self) -> float:
        """The order's share of the window's volume: the ``p`` of the decomposition.

        Measured over the *benchmark's* window, which is not the window
        :func:`slippage.shortfall.participation_rate` uses -- that one starts
        at the bar containing arrival rather than at arrival, deliberately. The
        two differ whenever arrival falls part-way through a bar, and only this
        one makes the decomposition hold.
        """
        return self.quantity / self.market_quantity

    @property
    def average_price(self) -> float:
        """What the order paid, over the window."""
        return self.notional / self.quantity

    @property
    def market_vwap(self) -> float:
        """The benchmark as recorded, own prints included."""
        return self.market_notional / self.market_quantity

    @property
    def ex_own_vwap(self) -> float:
        """The benchmark with the order's own fills removed."""
        return (self.market_notional - self.notional) / (self.market_quantity - self.quantity)


def own_share(
    order: Order,
    series: BarSeries,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> OwnShare:
    """Decompose a window's volume into the order's own part and everyone else's.

    The window defaults to :func:`order_window`, so that this and the VWAP it
    corrects are measured over the same bars. Passing a window that is not the
    benchmark's own gives a number that is correct about that window and says
    nothing about the benchmark.

    Raises:
        ValidationError: If the order has no fills inside the window; if its
            fills account for every share the bars record, which leaves no
            benchmark that is not the order itself; or if they account for more
            than the bars record, which means the bars do not include the
            prints and the correction would be wrong rather than unnecessary.
    """
    window_start, window_end = order_window(order, series)
    if start is not None:
        window_start = start
    if end is not None:
        window_end = end
    quantity = 0.0
    notional = 0.0
    for fill in order.fills:
        if window_start <= fill.timestamp < window_end:
            quantity += fill.quantity
            notional += fill.quantity * fill.price
    if quantity <= 0.0:
        raise ValidationError(
            f"order filled nothing inside [{window_start!r}, {window_end!r}), so there is "
            "no own share to remove from the benchmark"
        )
    market_quantity = series.total_volume(window_start, window_end)
    market_notional = series.vwap(window_start, window_end) * market_quantity
    if quantity > market_quantity:
        raise ValidationError(
            f"the order filled {quantity:g} shares inside the window but the bars record "
            f"only {market_quantity:g} over it. The bars do not include these prints -- a "
            "different venue, a different symbol, or a feed that reports only what it "
            "matched -- so removing the order from them would subtract volume that was "
            "never added"
        )
    if quantity == market_quantity:
        raise ValidationError(
            f"the order's {quantity:g} shares are the whole of the {market_quantity:g} the "
            "bars record over its window, so the market VWAP is the order's own average "
            "price and measuring one against the other gives zero whatever was paid. There "
            "is no benchmark here that is not the order itself"
        )
    return OwnShare(
        quantity=quantity,
        notional=notional,
        market_quantity=market_quantity,
        market_notional=market_notional,
        start=window_start,
        end=window_end,
    )
