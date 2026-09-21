from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import make_bar, minute

from slippage.exceptions import InsufficientDataError, NoVolumeError, ValidationError
from slippage.series import BarSeries
from slippage.types import Bar


class TestConstruction:
    def test_bars_are_sorted(self) -> None:
        series = BarSeries([make_bar(3, 101.0), make_bar(1, 100.0), make_bar(2, 100.5)])
        assert [b.timestamp for b in series] == [minute(1), minute(2), minute(3)]

    def test_empty_series_is_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            BarSeries([])

    def test_duplicate_timestamps_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            BarSeries([make_bar(1, 100.0), make_bar(1, 101.0)])

    def test_span_uses_inferred_duration(self, flat_bars: list[Bar]) -> None:
        series = BarSeries(flat_bars)
        assert series.start == minute(0)
        assert series.bar_duration == timedelta(minutes=1)
        assert series.end == minute(10)

    def test_single_bar_series_has_nominal_duration(self) -> None:
        series = BarSeries([make_bar(0, 100.0)])
        assert series.bar_duration == timedelta(minutes=1)
        assert series.twap() == pytest.approx(series[0].typical_price)


class TestWindowing:
    def test_window_is_half_open_on_bar_starts(self, flat_bars: list[Bar]) -> None:
        series = BarSeries(flat_bars)
        sub = series.window(minute(2), minute(5))
        assert [b.timestamp for b in sub] == [minute(2), minute(3), minute(4)]

    def test_empty_window_raises(self, flat_bars: list[Bar]) -> None:
        series = BarSeries(flat_bars)
        with pytest.raises(InsufficientDataError, match="no bars in window"):
            series.window(minute(20), minute(30))

    def test_open_ended_window_covers_the_series(self, flat_bars: list[Bar]) -> None:
        series = BarSeries(flat_bars)
        assert len(series.window()) == len(series)


class TestPriceAt:
    def test_price_at_is_the_open_of_the_bar_in_progress(self) -> None:
        bars = [
            Bar(timestamp=minute(n), open=100.0 + n, high=110.0 + n, low=99.0 + n,
                close=100.5 + n, volume=1_000.0)
            for n in range(5)
        ]
        series = BarSeries(bars)
        assert series.price_at(minute(2)) == pytest.approx(102.0)
        # Thirty seconds into the third bar is still the third bar.
        assert series.price_at(minute(2) + timedelta(seconds=30)) == pytest.approx(102.0)

    def test_price_before_the_series_is_the_first_open(self, rising_bars: list[Bar]) -> None:
        series = BarSeries(rising_bars)
        assert series.price_at(minute(-5)) == pytest.approx(series[0].open)

    def test_price_after_the_series_is_the_last_close(self, rising_bars: list[Bar]) -> None:
        series = BarSeries(rising_bars)
        assert series.price_at(minute(500)) == pytest.approx(series[-1].close)


class TestAggregates:
    def test_vwap_of_a_flat_series_is_the_level(self, flat_bars: list[Bar]) -> None:
        series = BarSeries(flat_bars)
        # Typical price of a flat bar is (100.05 + 99.95 + 100) / 3 = 100.
        assert series.vwap() == pytest.approx(100.0)

    def test_vwap_weights_by_volume(self) -> None:
        series = BarSeries(
            [make_bar(0, 100.0, spread=0.0, volume=1_000.0),
             make_bar(1, 110.0, spread=0.0, volume=9_000.0)]
        )
        # 10% of the volume at 100 and 90% at 110.
        assert series.vwap() == pytest.approx(109.0)

    def test_twap_ignores_volume(self) -> None:
        series = BarSeries(
            [make_bar(0, 100.0, spread=0.0, volume=1_000.0),
             make_bar(1, 110.0, spread=0.0, volume=9_000.0)]
        )
        assert series.twap() == pytest.approx(105.0)

    def test_twap_weights_by_bar_span(self) -> None:
        # A one-minute bar at 100 followed by a ten-minute bar at 110: the
        # unweighted mean would be 105, the time-weighted answer is far higher.
        bars = [
            make_bar(0, 100.0, spread=0.0),
            make_bar(1, 110.0, spread=0.0),
            make_bar(11, 110.0, spread=0.0),
        ]
        series = BarSeries(bars)
        assert series.twap(minute(0), minute(11)) == pytest.approx(
            (100.0 * 60 + 110.0 * 600) / 660
        )

    def test_a_window_keeps_the_parent_bar_spans(self) -> None:
        # Regression: slicing used to re-infer spans from the slice alone, so a
        # window ending just before a long gap forgot that its final bar
        # covered that gap and reported the unweighted mean instead.
        bars = [
            make_bar(0, 100.0, spread=0.0),
            make_bar(1, 110.0, spread=0.0),
            make_bar(11, 110.0, spread=0.0),
        ]
        series = BarSeries(bars)
        sliced = series.window(minute(0), minute(11))
        assert sliced.twap() == pytest.approx(series.twap(minute(0), minute(11)))
        assert sliced.twap() == pytest.approx((100.0 * 60 + 110.0 * 600) / 660)

    def test_vwap_without_volume_raises_rather_than_falling_back(self) -> None:
        series = BarSeries([make_bar(0, 100.0, volume=0.0), make_bar(1, 101.0, volume=0.0)])
        with pytest.raises(NoVolumeError, match="undefined"):
            series.vwap()
        # TWAP remains well defined, which is the point of the distinction.
        assert series.twap() == pytest.approx(100.5)

    def test_total_volume_over_a_window(self, flat_bars: list[Bar]) -> None:
        series = BarSeries(flat_bars)
        assert series.total_volume(minute(0), minute(3)) == pytest.approx(30_000.0)

    def test_open_and_close_of_a_window(self, rising_bars: list[Bar]) -> None:
        series = BarSeries(rising_bars)
        assert series.open_price(minute(2), minute(5)) == pytest.approx(rising_bars[2].open)
        assert series.close_price(minute(2), minute(5)) == pytest.approx(rising_bars[4].close)

    def test_vwap_lies_between_the_extremes(self, rising_bars: list[Bar]) -> None:
        series = BarSeries(rising_bars)
        typicals = [b.typical_price for b in series]
        assert min(typicals) <= series.vwap() <= max(typicals)
        assert min(typicals) <= series.twap() <= max(typicals)
