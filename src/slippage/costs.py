"""Signed transaction costs.

A positive number always means the trade did *worse* than the reference. That
holds for both sides because every formula multiplies by :attr:`Side.sign`
rather than branching, so there is exactly one place where the convention could
be wrong and exactly one place to test it.
"""

from __future__ import annotations

from .exceptions import ValidationError
from .types import Side

__all__ = [
    "BPS_PER_UNIT",
    "cost_bps",
    "cost_currency",
    "cost_per_share",
    "from_bps",
    "to_bps",
]

BPS_PER_UNIT = 10_000.0
"""Basis points in one unit of relative price."""


def cost_per_share(side: Side, execution_price: float, reference_price: float) -> float:
    """Signed cost of one share against ``reference_price``, in price units."""
    return side.sign * (execution_price - reference_price)


def cost_currency(
    side: Side, quantity: float, execution_price: float, reference_price: float
) -> float:
    """Signed cost of ``quantity`` shares, in currency."""
    if quantity < 0.0:
        raise ValidationError(f"quantity must be unsigned, got {quantity!r}")
    return quantity * cost_per_share(side, execution_price, reference_price)


def cost_bps(side: Side, execution_price: float, reference_price: float) -> float:
    """Signed cost in basis points of the reference price.

    Normalising by the reference rather than the execution price keeps the
    denominator independent of how well the trade went, so two executions of
    the same order are compared on the same scale.
    """
    if reference_price <= 0.0:
        raise ValidationError(
            f"reference price must be positive to express cost in bps, got {reference_price!r}"
        )
    return BPS_PER_UNIT * cost_per_share(side, execution_price, reference_price) / reference_price


def to_bps(fraction: float) -> float:
    """Convert a relative move to basis points."""
    return fraction * BPS_PER_UNIT


def from_bps(bps: float) -> float:
    """Convert basis points to a relative move."""
    return bps / BPS_PER_UNIT
