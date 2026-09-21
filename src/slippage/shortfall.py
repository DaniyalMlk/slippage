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
from enum import Enum

from .costs import BPS_PER_UNIT
from .exceptions import SlippageError, ValidationError
from .series import BarSeries
from .types import Order, Side

__all__ = [
    "DelayBasis",
    "ShortfallBreakdown",
    "implementation_shortfall",
    "shortfall_from_market",
]

# Relative tolerance for the internal check that the components sum to the
# total. Both sides are computed from the same inputs by different routes, so
# any disagreement beyond rounding is a bug rather than a data problem.
_SUM_RTOL = 1e-9


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

    Parameters
    ----------
    order
        The parent order and its fills.
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
    p_d = _check_price("decision_price", decision_price)
    p_0 = _check_price("arrival_price", arrival_price)
    p_n = _check_price("final_price", final_price)
    if not math.isfinite(fees) or fees < 0.0:
        raise ValidationError(f"fees must be non-negative, got {fees!r}")
    if half_spread is not None and (not math.isfinite(half_spread) or half_spread < 0.0):
        raise ValidationError(f"half_spread must be non-negative, got {half_spread!r}")

    s = order.side.sign
    x = order.quantity
    q = order.filled_quantity
    unfilled = order.unfilled_quantity
    notional = sum(f.notional for f in order.fills)

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
        side=order.side,
        target_quantity=x,
        filled_quantity=q,
        decision_price=p_d,
        arrival_price=p_0,
        final_price=p_n,
        average_price=order.average_price if order.fills else None,
        delay=delay,
        trading=trading,
        opportunity=opportunity,
        commission=order.total_commission,
        fees=float(fees),
        spread=spread,
        impact=impact,
        delay_basis=delay_basis,
    )

    # The direct Perold formula, computed independently of the split.
    direct = s * (notional - q * p_d) + s * unfilled * (p_n - p_d) + order.total_commission + fees
    scale = max(abs(direct), x * p_d * 1e-12, 1e-12)
    if abs(breakdown.total - direct) > _SUM_RTOL * scale:
        raise SlippageError(
            f"shortfall components sum to {breakdown.total!r} but the direct formula "
            f"gives {direct!r}; this is a bug"
        )
    return breakdown


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
