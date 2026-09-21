from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slippage.costs import cost_bps, cost_currency, cost_per_share, from_bps, to_bps
from slippage.exceptions import ValidationError
from slippage.types import Side

prices = st.floats(min_value=0.01, max_value=1e5, allow_nan=False, allow_infinity=False)


class TestSignConvention:
    @pytest.mark.parametrize(
        ("side", "execution", "reference", "expected"),
        [
            (Side.BUY, 101.0, 100.0, 1.0),    # paid up: a cost
            (Side.BUY, 99.0, 100.0, -1.0),    # bought cheap: a saving
            (Side.SELL, 99.0, 100.0, 1.0),    # sold cheap: a cost
            (Side.SELL, 101.0, 100.0, -1.0),  # sold rich: a saving
            (Side.BUY, 100.0, 100.0, 0.0),
            (Side.SELL, 100.0, 100.0, 0.0),
        ],
    )
    def test_positive_means_worse_than_reference(
        self, side: Side, execution: float, reference: float, expected: float
    ) -> None:
        assert cost_per_share(side, execution, reference) == pytest.approx(expected)

    @given(execution=prices, reference=prices)
    def test_buy_and_sell_costs_are_exact_negatives(
        self, execution: float, reference: float
    ) -> None:
        assert cost_per_share(Side.BUY, execution, reference) == -cost_per_share(
            Side.SELL, execution, reference
        )


class TestUnits:
    def test_currency_cost_scales_with_quantity(self) -> None:
        assert cost_currency(Side.BUY, 1_000.0, 50.05, 50.0) == pytest.approx(50.0)

    def test_currency_cost_rejects_signed_quantity(self) -> None:
        with pytest.raises(ValidationError, match="unsigned"):
            cost_currency(Side.SELL, -100.0, 50.0, 50.0)

    def test_bps_are_relative_to_the_reference(self) -> None:
        # Ten cents on a fifty dollar stock is twenty basis points.
        assert cost_bps(Side.BUY, 50.10, 50.0) == pytest.approx(20.0)
        assert cost_bps(Side.SELL, 49.90, 50.0) == pytest.approx(20.0)

    def test_bps_reject_non_positive_reference(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            cost_bps(Side.BUY, 1.0, 0.0)

    @given(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
    def test_bps_round_trip(self, fraction: float) -> None:
        assert from_bps(to_bps(fraction)) == pytest.approx(fraction, abs=1e-15)

    @given(execution=prices, reference=prices, quantity=st.floats(min_value=1.0, max_value=1e6))
    def test_bps_and_currency_agree(
        self, execution: float, reference: float, quantity: float
    ) -> None:
        currency = cost_currency(Side.BUY, quantity, execution, reference)
        bps = cost_bps(Side.BUY, execution, reference)
        assert currency == pytest.approx(bps / 1e4 * reference * quantity, rel=1e-9, abs=1e-9)
