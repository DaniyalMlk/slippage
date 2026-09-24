"""The totals-based decomposition, and its agreement with the fill-based one.

There is one arithmetic and two ways into it, so almost everything worth
asserting here is that the two ways agree. The interesting part is the property
test: whatever order, fills and prices hypothesis can build, totalling the fills
by hand and calling the new entry point has to give the same breakdown, field
for field, as handing the order to the old one. If that ever stops holding, one
of the two has grown a behaviour the other has not.

The rest are the checks the new signature needs on its own account, because it
accepts numbers where the old one accepted an object that had already validated
them: an unsigned quantity really is unsigned, an overfill is refused, and the
timestamps the old path carries genuinely make no difference to the answer.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slippage.exceptions import ValidationError
from slippage.shortfall import (
    DelayBasis,
    ShortfallBreakdown,
    implementation_shortfall,
    shortfall_from_totals,
)
from slippage.types import Fill, Order, Side

START = datetime(2026, 3, 2, 14, 30)


def order_from(
    side: Side,
    quantity: float,
    fills: list[tuple[float, float, float]],
    *,
    spacing: int = 1,
    decision_price: float | None = None,
) -> Order:
    """An order whose fill times are spaced by ``spacing`` minutes."""
    return Order(
        symbol="TEST",
        side=side,
        quantity=quantity,
        decision_time=START - timedelta(minutes=10),
        arrival_time=START,
        decision_price=decision_price,
        fills=tuple(
            Fill(
                timestamp=START + timedelta(minutes=spacing * (index + 1)),
                quantity=q,
                price=p,
                commission=c,
            )
            for index, (q, p, c) in enumerate(fills)
        ),
    )


WORKED = [(6000.0, 50.12, 60.0), (1000.0, 50.20, 10.0)]


# -- the two routes agree ----------------------------------------------------


def test_the_two_entry_points_give_the_same_breakdown() -> None:
    """The hand-worked order, both ways, field for field."""
    order = order_from(Side.BUY, 10_000.0, WORKED, decision_price=50.00)

    from_order = implementation_shortfall(order, arrival_price=50.05, final_price=50.40, fees=12.5)
    from_totals = shortfall_from_totals(
        side=order.side,
        quantity=order.quantity,
        filled_quantity=order.filled_quantity,
        executed_notional=sum(fill.notional for fill in order.fills),
        decision_price=50.00,
        arrival_price=50.05,
        final_price=50.40,
        commission=order.total_commission,
        fees=12.5,
    )
    assert dataclasses.asdict(from_order) == dataclasses.asdict(from_totals)


@settings(max_examples=250, deadline=None)
@given(
    side=st.sampled_from(Side),
    quantity=st.floats(min_value=100.0, max_value=5e6),
    fraction=st.floats(min_value=0.0, max_value=1.0),
    decision=st.floats(min_value=1.0, max_value=500.0),
    arrival=st.floats(min_value=1.0, max_value=500.0),
    final=st.floats(min_value=1.0, max_value=500.0),
    average=st.floats(min_value=1.0, max_value=500.0),
    commission=st.floats(min_value=0.0, max_value=5000.0),
    fees=st.floats(min_value=0.0, max_value=5000.0),
    basis=st.sampled_from(DelayBasis),
)
def test_the_two_routes_agree_on_anything_hypothesis_can_build(
    side: Side,
    quantity: float,
    fraction: float,
    decision: float,
    arrival: float,
    final: float,
    average: float,
    commission: float,
    fees: float,
    basis: DelayBasis,
) -> None:
    """One arithmetic, two doors into it.

    Written as a property rather than as a handful of cases because the thing at
    risk is not a wrong number in one place — it is the two paths drifting apart
    at some corner nobody enumerated, which is exactly what a wrapper invites.
    """
    filled = quantity * fraction
    order = order_from(side, quantity, [(filled, average, commission)] if filled > 0 else [])

    from_totals = shortfall_from_totals(
        side=side,
        quantity=quantity,
        filled_quantity=filled,
        executed_notional=filled * average,
        decision_price=decision,
        arrival_price=arrival,
        final_price=final,
        commission=commission if filled > 0 else 0.0,
        fees=fees,
        delay_basis=basis,
    )
    from_order = implementation_shortfall(
        order,
        decision_price=decision,
        arrival_price=arrival,
        final_price=final,
        fees=fees,
        delay_basis=basis,
    )

    for field in dataclasses.fields(from_totals):
        left = getattr(from_totals, field.name)
        right = getattr(from_order, field.name)
        if isinstance(left, float) and isinstance(right, float):
            assert left == pytest.approx(right, rel=1e-12, abs=1e-9), field.name
        else:
            assert left == right, field.name


def test_the_fill_timestamps_make_no_difference_to_the_answer() -> None:
    """The claim the new entry point rests on, asserted rather than assumed.

    If spacing the same fills a minute apart and six hours apart ever produced
    different breakdowns, requiring timestamps would be justified and this whole
    function would be wrong.
    """

    def decompose(spacing: int) -> object:
        return dataclasses.asdict(
            implementation_shortfall(
                order_from(Side.BUY, 10_000.0, WORKED, spacing=spacing),
                decision_price=50.00,
                arrival_price=50.05,
                final_price=50.40,
            )
        )

    assert decompose(1) == decompose(360)


# -- the new signature's own checks ------------------------------------------


def test_the_components_still_sum_to_the_total() -> None:
    """The Perold cross-check lives in the totals function now, so it runs here."""
    result = shortfall_from_totals(
        side=Side.SELL,
        quantity=10_000.0,
        filled_quantity=7000.0,
        executed_notional=7000.0 * 49.88,
        decision_price=50.00,
        arrival_price=49.95,
        final_price=49.60,
        commission=70.0,
        fees=12.5,
    )
    components = sum(value for _, value in result)
    assert components == pytest.approx(result.total, rel=1e-12)


def test_the_average_price_is_derived_from_the_notional() -> None:
    """The caller gives a total; the average comes back, so nothing is lost."""
    result = shortfall_from_totals(
        side=Side.BUY,
        quantity=1000.0,
        filled_quantity=800.0,
        executed_notional=800.0 * 12.345,
        decision_price=12.0,
        arrival_price=12.1,
        final_price=12.5,
    )
    assert result.average_price == pytest.approx(12.345)


def test_an_unfilled_order_has_no_average_price() -> None:
    """Nothing traded, so there is no price to report — not a zero."""
    result = shortfall_from_totals(
        side=Side.BUY,
        quantity=1000.0,
        filled_quantity=0.0,
        executed_notional=0.0,
        decision_price=12.0,
        arrival_price=12.1,
        final_price=12.5,
    )
    assert result.average_price is None
    assert result.trading == pytest.approx(0.0)


def test_overfilling_is_refused_with_both_numbers() -> None:
    with pytest.raises(ValidationError) as raised:
        shortfall_from_totals(
            side=Side.BUY,
            quantity=1000.0,
            filled_quantity=1500.0,
            executed_notional=1500.0 * 12.0,
            decision_price=12.0,
            arrival_price=12.1,
            final_price=12.5,
        )
    assert "1500" in str(raised.value)
    assert "1000" in str(raised.value)


def test_a_fill_total_a_hair_over_its_target_is_accepted() -> None:
    """Summing a tape in floating point can overshoot by an ulp.

    Refusing that would be refusing correct data, so the comparison carries a
    relative slack and the quantity is clamped rather than rejected.
    """
    result = shortfall_from_totals(
        side=Side.BUY,
        quantity=1000.0,
        filled_quantity=1000.0 * (1 + 1e-15),
        executed_notional=1000.0 * 12.0,
        decision_price=12.0,
        arrival_price=12.1,
        final_price=12.5,
    )
    assert result.filled_quantity == pytest.approx(1000.0)


@pytest.mark.parametrize(
    "field",
    ["quantity", "filled_quantity", "executed_notional", "commission", "fees"],
)
def test_a_negative_quantity_is_refused_by_name(field: str) -> None:
    """Quantities are unsigned here; the side carries the sign.

    A negative one would flip a component silently rather than fail, and the
    result would look like an ordinary breakdown of a different trade.
    """
    arguments: dict[str, object] = {
        "side": Side.BUY,
        "quantity": 1000.0,
        "filled_quantity": 800.0,
        "executed_notional": 9600.0,
        "decision_price": 12.0,
        "arrival_price": 12.1,
        "final_price": 12.5,
        "commission": 5.0,
        "fees": 1.0,
    }
    arguments[field] = -1.0
    with pytest.raises(ValidationError) as raised:
        shortfall_from_totals(**arguments)  # type: ignore[arg-type]
    assert field in str(raised.value)


def test_a_zero_target_quantity_is_refused() -> None:
    """There is no order to decompose, and every basis-point figure would divide by zero."""
    with pytest.raises(ValidationError):
        shortfall_from_totals(
            side=Side.BUY,
            quantity=0.0,
            filled_quantity=0.0,
            executed_notional=0.0,
            decision_price=12.0,
            arrival_price=12.1,
            final_price=12.5,
        )


def test_the_delay_basis_still_reattributes_without_changing_the_total() -> None:
    """The convention survives the refactor, on the totals path."""

    def decompose(basis: DelayBasis) -> ShortfallBreakdown:
        return shortfall_from_totals(
            side=Side.BUY,
            quantity=10_000.0,
            filled_quantity=7000.0,
            executed_notional=7000.0 * 50.12,
            decision_price=50.00,
            arrival_price=50.05,
            final_price=50.40,
            delay_basis=basis,
        )

    on_order = decompose(DelayBasis.ORDER)
    on_executed = decompose(DelayBasis.EXECUTED)

    assert on_order.delay != pytest.approx(on_executed.delay)
    assert on_order.total == pytest.approx(on_executed.total, rel=1e-12)
