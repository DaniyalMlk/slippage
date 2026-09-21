from __future__ import annotations

from datetime import datetime, timedelta
from typing import TypedDict

import pytest
from conftest import make_bar, minute
from hypothesis import given, settings
from hypothesis import strategies as st

from slippage.exceptions import ValidationError
from slippage.series import BarSeries
from slippage.shortfall import (
    DelayBasis,
    implementation_shortfall,
    shortfall_from_market,
)
from slippage.types import Fill, Order, Side


def worked_order(side: Side = Side.BUY) -> Order:
    """The hand-worked example: 10,000 shares, 7,000 done in two fills.

    For a buy the prices drift up; for a sell they are mirrored about 50.00 so
    every component comes out identical.
    """
    s = side.sign
    return Order(
        symbol="ACME",
        side=side,
        quantity=10_000.0,
        decision_time=minute(0),
        arrival_time=minute(1),
        fills=(
            Fill(timestamp=minute(2), quantity=3_000.0, price=50.0 + s * 0.20, commission=30.0),
            Fill(timestamp=minute(5), quantity=4_000.0, price=50.0 + s * 0.30, commission=40.0),
        ),
        decision_price=50.0,
    )


class Prices(TypedDict):
    arrival_price: float
    final_price: float


def worked_prices(side: Side) -> Prices:
    s = side.sign
    return {"arrival_price": 50.0 + s * 0.10, "final_price": 50.0 + s * 0.50}


class TestWorkedExample:
    """Every figure below was computed by hand before the code was run.

    Paper portfolio: buy 10,000 at 50.00 = 500,000.
    Executed: 3,000 @ 50.20 + 4,000 @ 50.30 = 351,800 for 7,000 shares.
    Arrival 50.10, final 50.50, commission 70, fees 5.

    delay        10,000 x (50.10 - 50.00)            = 1,000.00   20.0 bps
    trading      351,800 - 7,000 x 50.10              = 1,100.00   22.0 bps
    opportunity  3,000 x (50.50 - 50.10)              = 1,200.00   24.0 bps
    commission                                        =    70.00    1.4 bps
    fees                                              =     5.00    0.1 bps
    total                                             = 3,375.00   67.5 bps
    """

    @pytest.mark.parametrize("side", list(Side))
    def test_order_basis_components(self, side: Side) -> None:
        result = implementation_shortfall(worked_order(side), fees=5.0, **worked_prices(side))
        assert result.delay == pytest.approx(1_000.0)
        assert result.trading == pytest.approx(1_100.0)
        assert result.opportunity == pytest.approx(1_200.0)
        assert result.commission == pytest.approx(70.0)
        assert result.fees == pytest.approx(5.0)
        assert result.total == pytest.approx(3_375.0)
        assert result.paper_notional == pytest.approx(500_000.0)
        assert result.total_bps == pytest.approx(67.5)

    @pytest.mark.parametrize("side", list(Side))
    def test_components_in_bps_add_up(self, side: Side) -> None:
        result = implementation_shortfall(worked_order(side), fees=5.0, **worked_prices(side))
        bps = result.components_bps()
        assert bps == pytest.approx(
            {"delay": 20.0, "trading": 22.0, "opportunity": 24.0, "commission": 1.4, "fees": 0.1}
        )
        assert sum(bps.values()) == pytest.approx(result.total_bps)

    @pytest.mark.parametrize("side", list(Side))
    def test_executed_basis_moves_only_the_delay_opportunity_boundary(self, side: Side) -> None:
        # Delay on the 7,000 executed shares only: 700. The 300 of delay on the
        # unexecuted 3,000 moves into opportunity: 3,000 x 0.50 = 1,500.
        result = implementation_shortfall(
            worked_order(side),
            fees=5.0,
            delay_basis=DelayBasis.EXECUTED,
            **worked_prices(side),
        )
        assert result.delay == pytest.approx(700.0)
        assert result.trading == pytest.approx(1_100.0)
        assert result.opportunity == pytest.approx(1_500.0)
        assert result.total == pytest.approx(3_375.0)

    def test_half_spread_partitions_trading_cost(self) -> None:
        result = implementation_shortfall(
            worked_order(), half_spread=0.02, **worked_prices(Side.BUY)
        )
        assert result.spread == pytest.approx(7_000.0 * 0.02)
        assert result.impact == pytest.approx(1_100.0 - 140.0)
        assert result.spread is not None and result.impact is not None
        assert result.spread + result.impact == pytest.approx(result.trading)

    def test_without_half_spread_the_split_is_absent(self) -> None:
        result = implementation_shortfall(worked_order(), **worked_prices(Side.BUY))
        assert result.spread is None
        assert result.impact is None

    def test_average_price_is_reported(self) -> None:
        result = implementation_shortfall(worked_order(), **worked_prices(Side.BUY))
        assert result.average_price == pytest.approx(351_800.0 / 7_000.0)


class TestEdgeCases:
    def test_unfilled_order_is_all_delay_and_opportunity(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=1_000.0,
            decision_time=minute(0),
            arrival_time=minute(1),
            decision_price=20.0,
        )
        result = implementation_shortfall(order, arrival_price=20.10, final_price=20.40)
        assert result.trading == 0.0
        assert result.average_price is None
        assert result.delay == pytest.approx(100.0)
        assert result.opportunity == pytest.approx(300.0)
        assert result.total == pytest.approx(1_000.0 * 0.40)

    def test_complete_order_has_no_opportunity_cost(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.SELL,
            quantity=100.0,
            decision_time=minute(0),
            arrival_time=minute(0),
            fills=(Fill(timestamp=minute(1), quantity=100.0, price=9.95),),
            decision_price=10.0,
        )
        result = implementation_shortfall(order, arrival_price=10.0, final_price=5.0)
        # A crash after the order completed is not the trader's cost.
        assert result.opportunity == 0.0
        assert result.total == pytest.approx(5.0)

    def test_favourable_moves_give_negative_components(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=100.0,
            decision_time=minute(0),
            arrival_time=minute(0),
            fills=(Fill(timestamp=minute(1), quantity=100.0, price=9.90),),
            decision_price=10.0,
        )
        result = implementation_shortfall(order, arrival_price=10.0, final_price=10.0)
        assert result.trading == pytest.approx(-10.0)
        assert result.total_bps == pytest.approx(-100.0)

    def test_decision_price_argument_overrides_the_order(self) -> None:
        result = implementation_shortfall(
            worked_order(), decision_price=50.10, **worked_prices(Side.BUY)
        )
        assert result.delay == pytest.approx(0.0)

    def test_missing_decision_price_raises(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=100.0,
            decision_time=minute(0),
            arrival_time=minute(0),
        )
        with pytest.raises(ValidationError, match="decision price is required"):
            implementation_shortfall(order, arrival_price=10.0, final_price=10.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"arrival_price": 0.0, "final_price": 1.0},
            {"arrival_price": 1.0, "final_price": float("nan")},
            {"arrival_price": 1.0, "final_price": 1.0, "fees": -1.0},
            {"arrival_price": 1.0, "final_price": 1.0, "half_spread": -0.01},
        ],
    )
    def test_bad_inputs_are_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            implementation_shortfall(worked_order(), **kwargs)  # type: ignore[arg-type]


# -- invariants ---------------------------------------------------------------

price = st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)


@st.composite
def orders(draw: st.DrawFn) -> Order:
    side = draw(st.sampled_from(list(Side)))
    target = draw(st.floats(min_value=100.0, max_value=1e6))
    fractions = draw(st.lists(st.floats(min_value=0.01, max_value=1.0), max_size=6))
    fill_rate = draw(st.floats(min_value=0.0, max_value=1.0))
    total = sum(fractions)
    fills = tuple(
        Fill(
            timestamp=datetime(2026, 1, 5, 10) + timedelta(minutes=i),
            quantity=target * fill_rate * f / total,
            price=draw(price),
            commission=draw(st.floats(min_value=0.0, max_value=50.0)),
        )
        for i, f in enumerate(fractions)
        if target * fill_rate * f / total > 0.0
    )
    return Order(
        symbol="X",
        side=side,
        quantity=target,
        decision_time=datetime(2026, 1, 5, 9, 30),
        arrival_time=datetime(2026, 1, 5, 9, 45),
        fills=fills,
        decision_price=draw(price),
    )


@settings(max_examples=300)
@given(order=orders(), arrival=price, final=price, fees=st.floats(min_value=0.0, max_value=100.0))
def test_both_delay_bases_split_the_same_total(
    order: Order, arrival: float, final: float, fees: float
) -> None:
    by_order = implementation_shortfall(
        order, arrival_price=arrival, final_price=final, fees=fees, delay_basis=DelayBasis.ORDER
    )
    by_exec = implementation_shortfall(
        order, arrival_price=arrival, final_price=final, fees=fees, delay_basis=DelayBasis.EXECUTED
    )
    assert by_order.total == pytest.approx(by_exec.total, rel=1e-9, abs=1e-6)
    assert by_order.trading == by_exec.trading
    assert by_order.delay + by_order.opportunity == pytest.approx(
        by_exec.delay + by_exec.opportunity, rel=1e-9, abs=1e-6
    )


@settings(max_examples=300)
@given(order=orders(), arrival=price, final=price)
def test_shortfall_is_paper_minus_real(order: Order, arrival: float, final: float) -> None:
    """Perold's definition, computed as two portfolio values rather than a formula.

    For a buy the paper portfolio pays X * P_d for X shares worth X * P_n;
    the real one pays the fills plus commission for Q shares worth Q * P_n and
    leaves the rest in cash. Shortfall is the difference in final wealth.
    """
    result = implementation_shortfall(order, arrival_price=arrival, final_price=final)
    s = order.side.sign
    assert order.decision_price is not None
    paper = s * order.quantity * (final - order.decision_price)
    spent = sum(f.notional for f in order.fills)
    real = s * (order.filled_quantity * final - spent) - order.total_commission
    assert result.total == pytest.approx(paper - real, rel=1e-9, abs=1e-6)


# -- market data helpers ------------------------------------------------------


@pytest.fixture
def trending() -> BarSeries:
    return BarSeries(
        [make_bar(n, 50.0 + 0.05 * n, open_=50.0 + 0.05 * n, volume=20_000.0) for n in range(10)]
    )


class TestFromMarket:
    def test_prices_come_from_the_series(self, trending: BarSeries) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=1_000.0,
            decision_time=minute(0),
            arrival_time=minute(2),
            fills=(Fill(timestamp=minute(3), quantity=600.0, price=50.20),),
        )
        result = shortfall_from_market(order, trending)
        assert result.decision_price == pytest.approx(trending[0].open)
        assert result.arrival_price == pytest.approx(trending[2].open)
        assert result.final_price == pytest.approx(trending[-1].close)
        assert result.delay == pytest.approx(1_000.0 * (trending[2].open - trending[0].open))

    def test_recorded_decision_price_wins(self, trending: BarSeries) -> None:
        order = worked_order()
        result = shortfall_from_market(order, trending, final_price=51.0)
        assert result.decision_price == 50.0
        assert result.final_price == 51.0
