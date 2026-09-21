from __future__ import annotations

import math

import pytest
from conftest import minute

from slippage.exceptions import ValidationError
from slippage.types import Bar, Fill, Order, Side


class TestSide:
    def test_signs_are_opposite_units(self) -> None:
        assert Side.BUY.sign == 1
        assert Side.SELL.sign == -1

    def test_opposite_is_an_involution(self) -> None:
        for side in Side:
            assert side.opposite.opposite is side

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("b", Side.BUY),
            ("BUY", Side.BUY),
            (" Bought ", Side.BUY),
            ("long", Side.BUY),
            ("s", Side.SELL),
            ("Sell", Side.SELL),
            ("SHORT", Side.SELL),
            ("-1", Side.SELL),
        ],
    )
    def test_parse_accepts_common_spellings(self, text: str, expected: Side) -> None:
        assert Side.parse(text) is expected

    def test_parse_rejects_nonsense(self) -> None:
        with pytest.raises(ValidationError, match="cannot interpret"):
            Side.parse("sideways")

    def test_parse_is_idempotent_on_enum(self) -> None:
        assert Side.parse(Side.SELL) is Side.SELL


class TestFill:
    def test_notional_excludes_commission(self) -> None:
        fill = Fill(timestamp=minute(0), quantity=100.0, price=25.0, commission=1.0)
        assert fill.notional == pytest.approx(2_500.0)

    @pytest.mark.parametrize("quantity", [0.0, -1.0, math.nan, math.inf])
    def test_quantity_must_be_positive_and_finite(self, quantity: float) -> None:
        with pytest.raises(ValidationError):
            Fill(timestamp=minute(0), quantity=quantity, price=10.0)

    @pytest.mark.parametrize("price", [0.0, -10.0, math.nan])
    def test_price_must_be_positive(self, price: float) -> None:
        with pytest.raises(ValidationError):
            Fill(timestamp=minute(0), quantity=1.0, price=price)

    def test_commission_may_be_zero_but_not_negative(self) -> None:
        Fill(timestamp=minute(0), quantity=1.0, price=10.0, commission=0.0)
        with pytest.raises(ValidationError, match="non-negative"):
            Fill(timestamp=minute(0), quantity=1.0, price=10.0, commission=-0.01)


class TestOrder:
    def test_average_price_is_quantity_weighted(self, simple_order: Order) -> None:
        # 400 @ 100.10 and 600 @ 100.20 -> 100.16, not the unweighted 100.15.
        assert simple_order.average_price == pytest.approx(100.16)

    def test_derived_quantities(self, simple_order: Order) -> None:
        assert simple_order.filled_quantity == pytest.approx(1_000.0)
        assert simple_order.unfilled_quantity == 0.0
        assert simple_order.is_complete
        assert simple_order.fill_rate == pytest.approx(1.0)
        assert simple_order.total_commission == pytest.approx(10.0)
        assert simple_order.signed_quantity == pytest.approx(1_000.0)

    def test_sell_order_has_negative_signed_quantity(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.SELL,
            quantity=500.0,
            decision_time=minute(0),
            arrival_time=minute(0),
        )
        assert order.signed_quantity == pytest.approx(-500.0)

    def test_fills_are_sorted_on_construction(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=300.0,
            decision_time=minute(0),
            arrival_time=minute(0),
            fills=(
                Fill(timestamp=minute(5), quantity=100.0, price=10.0),
                Fill(timestamp=minute(1), quantity=200.0, price=11.0),
            ),
        )
        assert [f.timestamp for f in order.fills] == [minute(1), minute(5)]
        assert order.first_fill_time == minute(1)
        assert order.last_fill_time == minute(5)

    def test_partial_fill_reports_remainder(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=1_000.0,
            decision_time=minute(0),
            arrival_time=minute(0),
            fills=(Fill(timestamp=minute(1), quantity=250.0, price=10.0),),
        )
        assert order.unfilled_quantity == pytest.approx(750.0)
        assert order.fill_rate == pytest.approx(0.25)
        assert not order.is_complete

    def test_floating_point_overfill_within_tolerance_is_accepted(self) -> None:
        # Three fills of a third each do not sum to exactly one in binary.
        third = 1.0 / 3.0
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=1.0,
            decision_time=minute(0),
            arrival_time=minute(0),
            fills=tuple(Fill(timestamp=minute(i), quantity=third, price=10.0) for i in range(3)),
        )
        assert order.is_complete
        assert order.unfilled_quantity == 0.0

    def test_overfill_beyond_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds order quantity"):
            Order(
                symbol="ACME",
                side=Side.BUY,
                quantity=100.0,
                decision_time=minute(0),
                arrival_time=minute(0),
                fills=(Fill(timestamp=minute(1), quantity=100.5, price=10.0),),
            )

    def test_arrival_before_decision_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="precedes"):
            Order(
                symbol="ACME",
                side=Side.BUY,
                quantity=100.0,
                decision_time=minute(5),
                arrival_time=minute(1),
            )

    def test_fill_before_decision_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="precedes decision_time"):
            Order(
                symbol="ACME",
                side=Side.BUY,
                quantity=100.0,
                decision_time=minute(5),
                arrival_time=minute(5),
                fills=(Fill(timestamp=minute(1), quantity=10.0, price=10.0),),
            )

    def test_average_price_of_unfilled_order_raises(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=100.0,
            decision_time=minute(0),
            arrival_time=minute(0),
        )
        with pytest.raises(ValidationError, match="no fills"):
            _ = order.average_price

    def test_empty_symbol_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="symbol"):
            Order(
                symbol="",
                side=Side.BUY,
                quantity=1.0,
                decision_time=minute(0),
                arrival_time=minute(0),
            )

    def test_side_is_parsed_from_a_string(self) -> None:
        order = Order(
            symbol="ACME",
            side="sell",  # type: ignore[arg-type]
            quantity=1.0,
            decision_time=minute(0),
            arrival_time=minute(0),
        )
        assert order.side is Side.SELL

    def test_with_fills_preserves_the_rest(self, simple_order: Order) -> None:
        replaced = simple_order.with_fills((Fill(timestamp=minute(3), quantity=100.0, price=99.0),))
        assert replaced.symbol == simple_order.symbol
        assert replaced.decision_price == simple_order.decision_price
        assert replaced.filled_quantity == pytest.approx(100.0)


class TestBar:
    def test_typical_price(self) -> None:
        bar = Bar(timestamp=minute(0), open=10.0, high=12.0, low=9.0, close=11.0, volume=100.0)
        assert bar.typical_price == pytest.approx((12.0 + 9.0 + 11.0) / 3.0)
        assert bar.notional == pytest.approx(bar.typical_price * 100.0)

    def test_zero_volume_is_allowed(self) -> None:
        bar = Bar(timestamp=minute(0), open=10.0, high=10.0, low=10.0, close=10.0, volume=0.0)
        assert bar.notional == 0.0

    def test_inverted_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="below low"):
            Bar(timestamp=minute(0), open=10.0, high=9.0, low=11.0, close=10.0, volume=1.0)

    def test_open_outside_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="open"):
            Bar(timestamp=minute(0), open=13.0, high=12.0, low=9.0, close=11.0, volume=1.0)

    def test_close_outside_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="close"):
            Bar(timestamp=minute(0), open=10.0, high=12.0, low=9.0, close=8.0, volume=1.0)
