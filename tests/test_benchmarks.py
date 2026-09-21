from __future__ import annotations

import pytest
from conftest import make_bar, minute

from slippage.benchmarks import Benchmark, benchmark_price, order_window
from slippage.series import BarSeries
from slippage.types import Bar, Fill, Order, Side


@pytest.fixture
def series(rising_bars: list[Bar]) -> BarSeries:
    return BarSeries(rising_bars)


class TestOrderWindow:
    def test_window_runs_from_arrival_to_the_end_of_the_last_fill_bar(
        self, simple_order: Order, series: BarSeries
    ) -> None:
        start, end = order_window(simple_order, series)
        assert start == minute(1)
        # Last fill is at minute 4, so the window must include the bar
        # starting at minute 4 and therefore end at minute 5.
        assert end == minute(5)

    def test_unfilled_order_is_scored_over_the_rest_of_the_series(self, series: BarSeries) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=100.0,
            decision_time=minute(0),
            arrival_time=minute(3),
        )
        start, end = order_window(order, series)
        assert (start, end) == (minute(3), series.end)

    def test_fill_in_the_arrival_bar_still_gives_a_non_empty_window(
        self, series: BarSeries
    ) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=100.0,
            decision_time=minute(2),
            arrival_time=minute(2),
            fills=(Fill(timestamp=minute(2), quantity=100.0, price=100.3),),
        )
        start, end = order_window(order, series)
        assert end > start
        assert len(series.window(start, end)) == 1


class TestBenchmarkPrice:
    def test_arrival_is_the_open_of_the_arrival_bar(
        self, simple_order: Order, series: BarSeries
    ) -> None:
        assert benchmark_price(simple_order, series, Benchmark.ARRIVAL) == pytest.approx(
            series[1].open
        )

    def test_decision_prefers_the_recorded_decision_price(
        self, simple_order: Order, series: BarSeries
    ) -> None:
        assert simple_order.decision_price == 100.0
        assert benchmark_price(simple_order, series, Benchmark.DECISION) == 100.0

    def test_decision_falls_back_to_the_market_at_decision_time(self, series: BarSeries) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=100.0,
            decision_time=minute(3),
            arrival_time=minute(5),
        )
        assert benchmark_price(order, series, Benchmark.DECISION) == pytest.approx(series[3].open)

    def test_interval_vwap_uses_only_the_order_window(
        self, simple_order: Order, series: BarSeries
    ) -> None:
        expected = series.vwap(minute(1), minute(5))
        assert benchmark_price(simple_order, series, Benchmark.INTERVAL_VWAP) == pytest.approx(
            expected
        )
        # And that is materially different from the full-series VWAP on a
        # trending day, which is the reason the window matters.
        assert abs(expected - series.vwap()) > 0.1

    def test_interval_twap_close_and_open(self, simple_order: Order, series: BarSeries) -> None:
        assert benchmark_price(simple_order, series, Benchmark.INTERVAL_TWAP) == pytest.approx(
            series.twap(minute(1), minute(5))
        )
        assert benchmark_price(simple_order, series, Benchmark.CLOSE) == pytest.approx(
            series[4].close
        )
        assert benchmark_price(simple_order, series, Benchmark.OPEN) == pytest.approx(
            series[1].open
        )

    def test_explicit_window_overrides_the_default(
        self, simple_order: Order, series: BarSeries
    ) -> None:
        full_day = benchmark_price(
            simple_order, series, Benchmark.INTERVAL_VWAP, start=series.start, end=series.end
        )
        assert full_day == pytest.approx(series.vwap())

    def test_is_interval_classification(self) -> None:
        assert not Benchmark.ARRIVAL.is_interval
        assert not Benchmark.DECISION.is_interval
        assert Benchmark.INTERVAL_VWAP.is_interval
        assert Benchmark.CLOSE.is_interval

    def test_flat_market_makes_every_benchmark_agree(self) -> None:
        bars = BarSeries([make_bar(n, 50.0) for n in range(6)])
        order = Order(
            symbol="ACME",
            side=Side.SELL,
            quantity=10.0,
            decision_time=minute(0),
            arrival_time=minute(1),
            fills=(Fill(timestamp=minute(3), quantity=10.0, price=49.9),),
        )
        prices = {b: benchmark_price(order, bars, b) for b in Benchmark}
        for price in prices.values():
            assert price == pytest.approx(50.0)
