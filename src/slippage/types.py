"""Value types for orders, fills and market data.

Sign convention
---------------

Exactly one sign convention runs through the library, and it is carried by
:class:`Side`. ``Side.BUY.sign`` is ``+1`` and ``Side.SELL.sign`` is ``-1``.

That single number does two jobs:

* **Cost.** ``sign * (execution_price - benchmark)`` is positive whenever the
  trade did worse than the benchmark — a buy filled above it, or a sell filled
  below it. Cost functions therefore never branch on the side.
* **Direction.** ``sign * quantity`` is the signed order flow, which is what
  every impact model needs: buying pushes the price up, selling pushes it down.

The two uses agreeing is not a coincidence. Impact moves the price against the
trader, so the direction that makes impact positive is the direction that makes
cost positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .exceptions import ValidationError

__all__ = ["Side", "Fill", "Order", "Bar"]

# Quantities are floats because odd lots, currency notionals and fractional
# shares all occur; comparisons therefore need a tolerance rather than ``==``.
_QTY_TOL = 1e-9


class Side(Enum):
    """The direction of an order, and the sign convention that follows from it."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """``+1`` for a buy, ``-1`` for a sell."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        """The other side."""
        return Side.SELL if self is Side.BUY else Side.BUY

    @classmethod
    def parse(cls, value: str | Side) -> Side:
        """Accept ``"B"``, ``"buy"``, ``"SELL"`` and friends."""
        if isinstance(value, Side):
            return value
        text = value.strip().lower()
        if text in {"b", "buy", "bought", "long", "+1"}:
            return cls.BUY
        if text in {"s", "sell", "sold", "short", "-1"}:
            return cls.SELL
        raise ValidationError(f"cannot interpret {value!r} as a side")

    def __str__(self) -> str:
        return self.value


def _check_positive(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    if value <= 0.0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    return float(value)


def _check_non_negative(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    if value < 0.0:
        raise ValidationError(f"{name} must be non-negative, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Fill:
    """A single execution against a parent order.

    ``quantity`` is unsigned: the direction lives on the parent
    :class:`Order`, so a fill on its own is never ambiguous about whether a
    negative number means a sell or a correction.
    """

    timestamp: datetime
    quantity: float
    price: float
    commission: float = 0.0
    venue: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", _check_positive("fill quantity", self.quantity))
        object.__setattr__(self, "price", _check_positive("fill price", self.price))
        object.__setattr__(
            self, "commission", _check_non_negative("commission", self.commission)
        )

    @property
    def notional(self) -> float:
        """Gross traded value, excluding commission."""
        return self.quantity * self.price


@dataclass(frozen=True)
class Order:
    """A parent order and the fills it received.

    ``decision_time`` is when the portfolio manager committed to the trade and
    ``arrival_time`` is when the order reached the market. The gap between them
    is where delay cost accrues, which is why they are separate fields rather
    than one timestamp: collapsing them makes delay cost unmeasurable, and
    delay cost is frequently the largest component of implementation shortfall.
    """

    symbol: str
    side: Side
    quantity: float
    decision_time: datetime
    arrival_time: datetime
    fills: tuple[Fill, ...] = field(default_factory=tuple)
    decision_price: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValidationError("symbol must not be empty")
        object.__setattr__(self, "side", Side.parse(self.side))
        object.__setattr__(self, "quantity", _check_positive("order quantity", self.quantity))
        if self.arrival_time < self.decision_time:
            raise ValidationError(
                f"arrival_time {self.arrival_time!r} precedes "
                f"decision_time {self.decision_time!r}"
            )
        fills = tuple(sorted(self.fills, key=lambda f: f.timestamp))
        object.__setattr__(self, "fills", fills)
        for fill in fills:
            if fill.timestamp < self.decision_time:
                raise ValidationError(
                    f"fill at {fill.timestamp!r} precedes decision_time "
                    f"{self.decision_time!r}"
                )
        filled = sum(f.quantity for f in fills)
        if filled > self.quantity + _QTY_TOL:
            raise ValidationError(
                f"fills total {filled} which exceeds order quantity {self.quantity}"
            )
        if self.decision_price is not None:
            object.__setattr__(
                self, "decision_price", _check_positive("decision_price", self.decision_price)
            )

    # -- derived quantities -------------------------------------------------

    @property
    def filled_quantity(self) -> float:
        return sum(f.quantity for f in self.fills)

    @property
    def unfilled_quantity(self) -> float:
        remaining = self.quantity - self.filled_quantity
        return 0.0 if remaining < _QTY_TOL else remaining

    @property
    def is_complete(self) -> bool:
        return self.unfilled_quantity == 0.0

    @property
    def fill_rate(self) -> float:
        """Fraction of the target quantity that was executed, in ``[0, 1]``."""
        return self.filled_quantity / self.quantity

    @property
    def average_price(self) -> float:
        """Quantity-weighted average execution price.

        Raises
        ------
        ValidationError
            If the order has no fills, in which case no average exists. A
            sentinel such as zero or NaN would flow into a cost figure and
            quietly corrupt an aggregate.
        """
        if not self.fills:
            raise ValidationError(f"order on {self.symbol} has no fills to average")
        return sum(f.notional for f in self.fills) / self.filled_quantity

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def first_fill_time(self) -> datetime | None:
        return self.fills[0].timestamp if self.fills else None

    @property
    def last_fill_time(self) -> datetime | None:
        return self.fills[-1].timestamp if self.fills else None

    @property
    def signed_quantity(self) -> float:
        """Target quantity with the side's sign applied."""
        return self.side.sign * self.quantity

    def with_fills(self, fills: tuple[Fill, ...]) -> Order:
        """Return a copy carrying a different set of fills."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            decision_time=self.decision_time,
            arrival_time=self.arrival_time,
            fills=fills,
            decision_price=self.decision_price,
        )


@dataclass(frozen=True)
class Bar:
    """An OHLCV bar. ``timestamp`` is the *start* of the interval."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _check_positive(name, getattr(self, name)))
        object.__setattr__(self, "volume", _check_non_negative("volume", self.volume))
        if self.high < self.low:
            raise ValidationError(f"high {self.high} is below low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValidationError(
                f"open {self.open} lies outside the range [{self.low}, {self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValidationError(
                f"close {self.close} lies outside the range [{self.low}, {self.high}]"
            )

    @property
    def typical_price(self) -> float:
        """``(high + low + close) / 3``.

        The conventional stand-in for the average traded price within a bar
        when tick data is unavailable. Using the close instead would bias any
        VWAP computed from bars towards the end of each interval.
        """
        return (self.high + self.low + self.close) / 3.0

    @property
    def notional(self) -> float:
        return self.typical_price * self.volume
