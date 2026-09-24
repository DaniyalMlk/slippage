"""Implementation shortfall and its attribution.

Implementation shortfall (Perold, 1988) is the difference between the return of
a *paper* portfolio, which trades the whole order instantly at the decision
price for free, and the *real* one. For an order of target size ``X`` with side
sign ``s``, fills ``(q_i, p_i)`` totalling ``Q``, decision price ``P_d`` and a
final price ``P_n`` at which the unfilled remainder is marked, it is::

    IS = s * (sum(q_i * p_i) - Q * P_d)      # executed shares vs paper
       + s * (X - Q) * (P_n - P_d)           # shares never executed
       + explicit costs

This module splits that total into the pieces a trader can act on:

``delay``
    The market moving between the decision and the order reaching the market.
    A process cost: it is fixed by how fast orders get from the portfolio
    manager to the desk, not by how the desk trades.
``trading``
    Execution relative to the arrival price. Optionally split further into the
    half-spread paid for immediacy and the remainder, which is impact and
    timing.
``opportunity``
    The move on shares that were never executed.
``explicit``
    Commissions and fees, which are known in advance and kept apart from the
    implicit costs above so they cannot mask them.

Delay basis
-----------

Where the unexecuted shares' delay belongs is a convention, not a fact. The
*order* basis (Kissell's expanded shortfall, the default) charges delay on the
full target, since the whole order sat idle during the delay, and measures
opportunity cost from arrival. The *executed* basis charges delay only on the
shares that traded and measures opportunity cost from the decision price. Both
split the same total; only the boundary between delay and opportunity moves.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .costs import BPS_PER_UNIT
from .exceptions import SlippageError, ValidationError
from .series import BarSeries
from .types import Order, Side

__all__ = [
    "DelayBasis",
    "FillAttribution",
    "ShortfallBreakdown",
    "attribute_fills",
    "implementation_shortfall",
    "participation_rate",
    "shortfall_from_market",
    "shortfall_from_totals",
]

# Relative tolerance for the internal check that the components sum to the
# total. Both sides are computed from the same inputs by different routes, so
# any disagreement beyond rounding is a bug rather than a data problem. It is
# relative to the size of the *terms*, not of the result: when the components
# nearly cancel, the total can be far smaller than the rounding error in each.
_SUM_RTOL = 1e-12

# Relative slack on the fills-against-target comparison. A tape summed in
# floating point can exceed its own target by an ulp or two without anybody
# having overfilled anything, and refusing that would be refusing correct data.
_FILL_RTOL = 1e-12


class DelayBasis(Enum):
    """Which shares delay cost is charged on. See the module docstring."""

    ORDER = "order"
    EXECUTED = "executed"


@dataclass(frozen=True)
class ShortfallBreakdown:
    """Implementation shortfall in currency, split into its components.

    Every component follows the library's sign convention: positive is a cost.
    ``spread`` and ``impact`` are ``None`` unless a half-spread was supplied, in
    which case they partition ``trading`` exactly.
    """

    side: Side
    target_quantity: float
    filled_quantity: float
    decision_price: float
    arrival_price: float
    final_price: float
    average_price: float | None
    delay: float
    trading: float
    opportunity: float
    commission: float
    fees: float
    spread: float | None = None
    impact: float | None = None
    delay_basis: DelayBasis = DelayBasis.ORDER

    # -- aggregates ---------------------------------------------------------

    @property
    def explicit(self) -> float:
        return self.commission + self.fees

    @property
    def implicit(self) -> float:
        return self.delay + self.trading + self.opportunity

    @property
    def total(self) -> float:
        return self.implicit + self.explicit

    @property
    def paper_notional(self) -> float:
        """The value the paper portfolio traded: ``X * P_d``.

        The denominator for every basis-point figure, so that components of the
        same order are expressed on one scale and add up in bps as they do in
        currency.
        """
        return self.target_quantity * self.decision_price

    def bps(self, amount: float) -> float:
        """Express a currency amount in basis points of the paper notional."""
        return BPS_PER_UNIT * amount / self.paper_notional

    @property
    def total_bps(self) -> float:
        return self.bps(self.total)

    def components(self) -> dict[str, float]:
        """The additive components, in currency, in presentation order."""
        return {
            "delay": self.delay,
            "trading": self.trading,
            "opportunity": self.opportunity,
            "commission": self.commission,
            "fees": self.fees,
        }

    def components_bps(self) -> dict[str, float]:
        return {name: self.bps(value) for name, value in self.components().items()}

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return iter(self.components().items())


def _check_price(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be a positive finite price, got {value!r}")
    return float(value)


def _check_quantity(name: str, value: float, *, positive: bool = False) -> float:
    """A quantity that is finite, unsigned, and optionally not zero.

    Quantities are unsigned everywhere in this library; the side carries the
    sign. A negative one here would flip a component silently rather than fail.
    """
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError(f"{name} must be a non-negative finite number, got {value!r}")
    if positive and value == 0.0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    return float(value)


def shortfall_from_totals(
    *,
    side: Side,
    quantity: float,
    filled_quantity: float,
    executed_notional: float,
    decision_price: float,
    arrival_price: float,
    final_price: float,
    commission: float = 0.0,
    fees: float = 0.0,
    half_spread: float | None = None,
    delay_basis: DelayBasis = DelayBasis.ORDER,
) -> ShortfallBreakdown:
    """Decompose a shortfall from order totals rather than from a fill tape.

    This is the whole arithmetic of the decomposition, and it is deliberately
    the version that takes the least. The Perold identity reads the side, the
    two quantities, what the executed shares cost in total, three prices and the
    explicit costs; it does not read *when* anything happened. Fills at
    one-minute and at six-hour spacings give an identical breakdown to every
    digit, so requiring timestamps to get one asks a caller to invent data the
    answer does not depend on.

    Two callers have only totals. A broker's TCA extract gives side, quantity,
    filled quantity and an average price, with no individual fills to hang a
    time on. And anything reconstructing an order from a wire format has to
    fabricate timestamps to satisfy :class:`~slippage.types.Order` — which is
    worse than merely inconvenient, because the fabricated values look like data
    to everything downstream.

    :func:`implementation_shortfall` is a thin wrapper over this for the case
    where the fills *are* in hand, so the identity lives in one place and the
    two cannot disagree.

    Parameters
    ----------
    side
        Direction of the parent order. Every component changes sign with it.
    quantity
        Target size, unsigned.
    filled_quantity
        How much of it executed, unsigned. The remainder is charged as
        opportunity cost. It may not exceed ``quantity``.
    executed_notional
        Total cash value of the fills, ``sum(q_i * p_i)``, unsigned. Given as a
        total rather than as an average price because an average would have to
        be reconstructed into one anyway, and a weighted average supplied by
        hand is a common place for a rounding error to enter.
    decision_price, arrival_price, final_price
        The three prices the decomposition is measured between.
    commission
        Commission on the executed shares, in currency.
    fees
        Explicit costs beyond commission (exchange, clearing, taxes).
    half_spread
        Half the quoted spread at arrival, in price units. When given, trading
        cost is split into ``spread`` and ``impact``.
    delay_basis
        Which shares delay is charged on; see the module docstring.
    """
    p_d = _check_price("decision_price", decision_price)
    p_0 = _check_price("arrival_price", arrival_price)
    p_n = _check_price("final_price", final_price)
    x = _check_quantity("quantity", quantity, positive=True)
    q = _check_quantity("filled_quantity", filled_quantity)
    notional = _check_quantity("executed_notional", executed_notional)
    if not math.isfinite(commission) or commission < 0.0:
        raise ValidationError(f"commission must be non-negative, got {commission!r}")
    if not math.isfinite(fees) or fees < 0.0:
        raise ValidationError(f"fees must be non-negative, got {fees!r}")
    if half_spread is not None and (not math.isfinite(half_spread) or half_spread < 0.0):
        raise ValidationError(f"half_spread must be non-negative, got {half_spread!r}")
    if q > x * (1.0 + _FILL_RTOL):
        raise ValidationError(
            f"filled_quantity {q!r} exceeds quantity {x!r}: an order cannot be overfilled, "
            "and taken at face value this gives a negative unfilled quantity and an "
            "opportunity cost with the wrong sign"
        )
    q = min(q, x)

    s = side.sign
    unfilled = x - q

    trading = s * (notional - q * p_0)
    if delay_basis is DelayBasis.ORDER:
        delay = s * x * (p_0 - p_d)
        opportunity = s * unfilled * (p_n - p_0)
    else:
        delay = s * q * (p_0 - p_d)
        opportunity = s * unfilled * (p_n - p_d)

    spread: float | None = None
    impact: float | None = None
    if half_spread is not None:
        spread = q * half_spread
        impact = trading - spread

    breakdown = ShortfallBreakdown(
        side=side,
        target_quantity=x,
        filled_quantity=q,
        decision_price=p_d,
        arrival_price=p_0,
        final_price=p_n,
        average_price=notional / q if q > 0.0 else None,
        delay=delay,
        trading=trading,
        opportunity=opportunity,
        commission=float(commission),
        fees=float(fees),
        spread=spread,
        impact=impact,
        delay_basis=delay_basis,
    )

    # The direct Perold formula, computed independently of the split.
    direct = s * (notional - q * p_d) + s * unfilled * (p_n - p_d) + commission + fees
    scale = x * max(p_d, p_0, p_n) + notional + commission + fees
    if abs(breakdown.total - direct) > _SUM_RTOL * scale:
        raise SlippageError(
            f"shortfall components sum to {breakdown.total!r} but the direct formula "
            f"gives {direct!r}; this is a bug"
        )
    return breakdown


def implementation_shortfall(
    order: Order,
    *,
    arrival_price: float,
    final_price: float,
    decision_price: float | None = None,
    fees: float = 0.0,
    half_spread: float | None = None,
    delay_basis: DelayBasis = DelayBasis.ORDER,
) -> ShortfallBreakdown:
    """Decompose the implementation shortfall of ``order``.

    A thin wrapper over :func:`shortfall_from_totals`, which holds the
    arithmetic. This one exists because an :class:`~slippage.types.Order` is
    what the rest of the library passes around, and totalling its fills at every
    call site would be repetitive and easy to get subtly wrong.

    Parameters
    ----------
    order
        The parent order and its fills. Only the side, the quantities, the fill
        prices and the commissions are read: the timestamps on the order and on
        its fills play no part in the decomposition. If you have totals rather
        than fills, call :func:`shortfall_from_totals` directly rather than
        inventing times to build an order out of.
    arrival_price
        Price when the order reached the market.
    final_price
        Price at which the unfilled remainder is marked: typically the close,
        or the price when the order was cancelled.
    decision_price
        Overrides ``order.decision_price``. One of the two must be given.
    fees
        Explicit costs beyond per-fill commission (exchange, clearing, taxes).
    half_spread
        Half the quoted spread at arrival, in price units. When given, trading
        cost is split into ``spread`` (the fixed price of immediacy on every
        executed share) and ``impact`` (everything else).
    delay_basis
        Which shares delay is charged on; see the module docstring.
    """
    if decision_price is None:
        decision_price = order.decision_price
    if decision_price is None:
        raise ValidationError(
            "a decision price is required: set Order.decision_price or pass decision_price"
        )

    return shortfall_from_totals(
        side=order.side,
        quantity=order.quantity,
        filled_quantity=order.filled_quantity,
        executed_notional=sum(f.notional for f in order.fills),
        decision_price=decision_price,
        arrival_price=arrival_price,
        final_price=final_price,
        commission=order.total_commission,
        fees=fees,
        half_spread=half_spread,
        delay_basis=delay_basis,
    )


def shortfall_from_market(
    order: Order,
    series: BarSeries,
    *,
    final_price: float | None = None,
    fees: float = 0.0,
    half_spread: float | None = None,
    delay_basis: DelayBasis = DelayBasis.ORDER,
) -> ShortfallBreakdown:
    """:func:`implementation_shortfall` with prices read from market data.

    The decision price falls back to the market at ``decision_time``, arrival is
    the market at ``arrival_time``, and the final price defaults to the last
    close in the series — the mark for any shares left unexecuted at the end of
    the horizon.
    """
    decision = (
        order.decision_price
        if order.decision_price is not None
        else series.price_at(order.decision_time)
    )
    return implementation_shortfall(
        order,
        arrival_price=series.price_at(order.arrival_time),
        final_price=series[-1].close if final_price is None else final_price,
        decision_price=decision,
        fees=fees,
        half_spread=half_spread,
        delay_basis=delay_basis,
    )


# -- per-fill attribution ---------------------------------------------------


@dataclass(frozen=True)
class FillAttribution:
    """One fill's contribution to trading cost and its footprint in the market."""

    timestamp: datetime
    quantity: float
    price: float
    trading_cost: float
    cost_bps: float
    cumulative_quantity: float
    cumulative_fraction: float
    bar_volume: float
    participation: float | None


def attribute_fills(
    order: Order, series: BarSeries, *, arrival_price: float | None = None
) -> list[FillAttribution]:
    """Break trading cost down fill by fill.

    ``trading_cost`` is measured against the arrival price, so the fills'
    costs sum exactly to :attr:`ShortfallBreakdown.trading`. ``participation``
    is the fill's share of the volume printed in the bar it landed in, and is
    ``None`` for a bar with no recorded volume rather than infinite.
    """
    p_0 = series.price_at(order.arrival_time) if arrival_price is None else arrival_price
    _check_price("arrival_price", p_0)
    s = order.side.sign
    cumulative = 0.0
    rows: list[FillAttribution] = []
    for fill in order.fills:
        cumulative += fill.quantity
        bar = series.bar_containing(fill.timestamp)
        cost = s * fill.quantity * (fill.price - p_0)
        rows.append(
            FillAttribution(
                timestamp=fill.timestamp,
                quantity=fill.quantity,
                price=fill.price,
                trading_cost=cost,
                cost_bps=BPS_PER_UNIT * s * (fill.price - p_0) / p_0,
                cumulative_quantity=cumulative,
                cumulative_fraction=cumulative / order.quantity,
                bar_volume=bar.volume,
                participation=fill.quantity / bar.volume if bar.volume > 0.0 else None,
            )
        )
    return rows


def participation_rate(order: Order, series: BarSeries) -> float:
    """Executed quantity as a fraction of market volume over the order's life.

    The window runs from the bar containing arrival to the bar containing the
    last fill, inclusive. The arrival bar is included even when arrival falls
    part-way through it, because excluding it would drop the volume the order
    actually traded against in its first minutes.
    """
    if not order.fills:
        return 0.0
    first_bar = series.bar_containing(order.arrival_time)
    last_fill = order.last_fill_time
    assert last_fill is not None
    end = series.end_of_bar_containing(last_fill)
    volume = series.total_volume(first_bar.timestamp, end)
    if volume <= 0.0:
        raise ValidationError("no market volume recorded over the order's life")
    return order.filled_quantity / volume
