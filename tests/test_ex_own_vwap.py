"""The VWAP benchmark, with and without the order's own prints in it.

Most of these are identities rather than tolerances. The decomposition is the
definition of the correction rearranged, so anything but equality here would be
an implementation error rather than an approximation, and a tolerance would
hide it.

The one that is not an identity is the ratio in basis points, and why it is not
is worth stating: ``cost_bps`` normalises by the reference price, and the two
references differ, so the ``(1 - p)`` relation that is exact in currency is
exact in basis points only up to the ratio of the two benchmarks. That ratio is
one plus the slippage, so the gap is of the order of the thing being measured
-- fractions of a basis point on a basis-point number, and measured here rather
than waved at.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from conftest import make_bar, minute

from slippage.benchmarks import (
    Benchmark,
    OwnShare,
    benchmark_price,
    order_window,
    own_share,
    score_order,
)
from slippage.exceptions import ValidationError
from slippage.series import BarSeries
from slippage.shortfall import participation_rate
from slippage.types import Fill, Order, Side


def _series(volumes: list[float], closes: list[float]) -> BarSeries:
    """One-minute bars with flat prices, so the typical price is the close exactly."""
    return BarSeries(
        [
            make_bar(n, close, spread=0.0, volume=volume)
            for n, (volume, close) in enumerate(zip(volumes, closes, strict=True))
        ]
    )


def _order(
    fills: list[tuple[datetime, float, float]],
    *,
    side: Side = Side.BUY,
    arrival: datetime | None = None,
) -> Order:
    return Order(
        symbol="ACME",
        side=side,
        quantity=sum(quantity for _, quantity, _ in fills),
        decision_time=minute(0),
        arrival_time=minute(1) if arrival is None else arrival,
        fills=tuple(
            Fill(timestamp=timestamp, quantity=quantity, price=price)
            for timestamp, quantity, price in fills
        ),
        decision_price=100.0,
    )


@pytest.fixture
def book() -> tuple[Order, BarSeries]:
    """A buy taking a bit over a quarter of five minutes of volume, paying up for it.

    The order arrives half a minute into the first bar, which is on purpose:
    that is what makes the benchmark's window and the window
    ``participation_rate`` uses differ, and one of the tests below is about
    exactly that.
    """
    series = _series(
        [1400.0, 1300.0, 1300.0, 1000.0, 1000.0],
        [100.00, 100.10, 100.15, 100.20, 100.30],
    )
    order = _order(
        [
            (minute(1) + timedelta(seconds=30), 400.0, 100.20),
            (minute(2) + timedelta(seconds=10), 300.0, 100.30),
            (minute(3) + timedelta(seconds=5), 300.0, 100.25),
        ],
        arrival=minute(0) + timedelta(seconds=30),
    )
    return order, series


# --- the decomposition ------------------------------------------------------


def test_the_market_vwap_is_the_blend_of_the_two(book: tuple[Order, BarSeries]) -> None:
    """``vwap_market == p * own_average + (1 - p) * vwap_ex_own``, exactly.

    The whole argument rests on this one line, so it is asserted as the
    equality it is rather than to a tolerance.
    """
    order, series = book
    share = own_share(order, series)
    blend = share.rate * share.average_price + (1.0 - share.rate) * share.ex_own_vwap
    assert blend == pytest.approx(share.market_vwap, rel=1e-15, abs=0.0)
    assert share.market_vwap == pytest.approx(series.vwap(*order_window(order, series)))


def test_the_benchmark_sits_between_the_market_and_the_order(
    book: tuple[Order, BarSeries],
) -> None:
    """A buy that paid up pulls the market VWAP above where the rest of the market was."""
    order, series = book
    share = own_share(order, series)
    assert share.ex_own_vwap < share.market_vwap < share.average_price
    assert share.rate == pytest.approx(1000.0 / 3600.0)


def test_the_currency_cost_is_exactly_one_minus_p_times_the_honest_one(
    book: tuple[Order, BarSeries],
) -> None:
    """In currency the relation is an identity, because the quantity is the same both times."""
    order, series = book
    share = own_share(order, series)
    market = score_order(order, series, Benchmark.INTERVAL_VWAP)
    honest = score_order(order, series, Benchmark.INTERVAL_VWAP_EX_OWN)
    assert market.cost_currency / honest.cost_currency == pytest.approx(
        1.0 - share.rate, rel=1e-12, abs=0.0
    )
    assert market.cost_currency < honest.cost_currency


def test_the_basis_point_ratio_is_one_minus_p_up_to_the_denominators(
    book: tuple[Order, BarSeries],
) -> None:
    """And the gap is of the order of the slippage, which is what makes it harmless.

    ``cost_bps`` divides by the reference price, and the two references differ
    by the slippage itself, so the ratio is ``(1 - p)`` times ``vwap_ex_own /
    vwap_market``. Here that is 3.8e-04 in relative terms -- a figure worth
    measuring rather than assuming, since it is the only place the clean
    identity does not hold.
    """
    order, series = book
    share = own_share(order, series)
    market = score_order(order, series, Benchmark.INTERVAL_VWAP)
    honest = score_order(order, series, Benchmark.INTERVAL_VWAP_EX_OWN)
    ratio = market.cost_bps / honest.cost_bps
    assert ratio == pytest.approx(
        (1.0 - share.rate) * share.ex_own_vwap / share.market_vwap, rel=1e-12
    )
    relative = abs(ratio - (1.0 - share.rate)) / (1.0 - share.rate)
    assert 1e-05 < relative < 1e-03


def test_the_correction_is_worth_a_third_of_the_reported_number(
    book: tuple[Order, BarSeries],
) -> None:
    """The measurement, on a 28% participation rate: 9.9 bps reported against 13.7 real."""
    order, series = book
    market = score_order(order, series, Benchmark.INTERVAL_VWAP)
    honest = score_order(order, series, Benchmark.INTERVAL_VWAP_EX_OWN)
    assert market.cost_bps == pytest.approx(9.90, abs=0.01)
    assert honest.cost_bps == pytest.approx(13.72, abs=0.01)
    assert honest.cost_bps - market.cost_bps == pytest.approx(3.81, abs=0.01)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_the_relation_holds_on_both_sides(side: Side) -> None:
    """Every cost here multiplies by the side's sign, so the ratio cannot know the side.

    Worth asserting anyway: the correction changes the benchmark, and a sign
    error in a benchmark is the kind that shows up on one side only.
    """
    series = _series([1200.0, 1200.0, 1200.0], [100.00, 100.10, 100.20])
    price = 100.20 if side is Side.BUY else 99.90
    order = _order([(minute(1) + timedelta(seconds=20), 600.0, price)], side=side)
    share = own_share(order, series)
    market = score_order(order, series, Benchmark.INTERVAL_VWAP)
    honest = score_order(order, series, Benchmark.INTERVAL_VWAP_EX_OWN)
    assert market.cost_currency / honest.cost_currency == pytest.approx(1.0 - share.rate, rel=1e-12)
    # Positive is worse than the reference on *both* sides -- that is the
    # convention costs.py exists to keep in one place. A buy paying above the
    # VWAP and a sell printing below it are both positive, and getting that
    # backwards in an expectation is easier than getting it backwards in the
    # code, which is why this is spelled out.
    assert market.cost_currency > 0.0
    assert honest.cost_currency > market.cost_currency


# --- the limit, and the refusals -------------------------------------------


def test_an_order_that_was_the_whole_market_scores_exactly_zero_and_is_refused() -> None:
    """Both halves, in that order, because the second is only defensible given the first.

    One bar, one price, and the order is every share of it. The market VWAP is
    then the order's own average to the last bit and the cost it reports is
    exactly zero -- not small. So there is nothing to correct and nothing to
    correct it with, and asking for the ex-own benchmark has to raise rather
    than divide by the zero shares left.
    """
    series = BarSeries(
        [make_bar(0, 100.00, spread=0.0, volume=500.0), make_bar(1, 100.50, spread=0.0, volume=0.0)]
    )
    order = Order(
        symbol="ACME",
        side=Side.BUY,
        quantity=500.0,
        decision_time=minute(0),
        arrival_time=minute(0),
        fills=(Fill(timestamp=minute(0) + timedelta(seconds=10), quantity=500.0, price=100.00),),
    )
    assert score_order(order, series, Benchmark.INTERVAL_VWAP).cost_bps == 0.0
    with pytest.raises(ValidationError, match="no benchmark here that is not the order itself"):
        own_share(order, series)
    with pytest.raises(ValidationError, match="whole of the"):
        benchmark_price(order, series, Benchmark.INTERVAL_VWAP_EX_OWN)


def test_more_own_volume_than_the_bars_record_is_a_data_problem_not_a_correction() -> None:
    """Which is why it has its own message naming the likely causes.

    Subtracting the order from bars that never contained it would take out
    volume nobody put in, and the result would be wrong in the direction that
    looks like a better execution.
    """
    series = _series([300.0, 300.0], [100.00, 100.10])
    order = _order([(minute(0) + timedelta(seconds=30), 1000.0, 100.05)], arrival=minute(0))
    with pytest.raises(ValidationError, match="bars do not include these prints"):
        own_share(order, series)


def test_no_fills_inside_the_window_is_refused() -> None:
    series = _series([1000.0] * 4, [100.0, 100.1, 100.2, 100.3])
    order = _order([(minute(1), 500.0, 100.1)])
    with pytest.raises(ValidationError, match="no own share to remove"):
        own_share(order, series, start=minute(2), end=minute(4))


def test_an_unfilled_order_has_no_own_share() -> None:
    series = _series([1000.0] * 4, [100.0, 100.1, 100.2, 100.3])
    order = Order(
        symbol="ACME",
        side=Side.BUY,
        quantity=500.0,
        decision_time=minute(0),
        arrival_time=minute(1),
    )
    with pytest.raises(ValidationError, match="no own share to remove"):
        own_share(order, series)


# --- the window the correction has to use -----------------------------------


def test_the_participation_rate_is_measured_over_a_different_window(
    book: tuple[Order, BarSeries],
) -> None:
    """And it is the benchmark's window that makes the decomposition hold.

    ``participation_rate`` starts at the bar containing arrival, deliberately,
    so that the volume the order traded against in its first minutes is not
    dropped. ``order_window`` starts at arrival itself. With a mid-bar arrival
    the two differ -- 0.200 against 0.278 here -- and only the second satisfies
    the identity, so the correction cannot be built on the first.
    """
    order, series = book
    share = own_share(order, series)
    reported = participation_rate(order, series)
    assert reported == pytest.approx(0.200, abs=0.001)
    assert share.rate == pytest.approx(0.278, abs=0.001)
    assert share.rate > reported

    # The identity, attempted with the other rate, does not hold.
    wrong = reported * share.average_price + (1.0 - reported) * share.ex_own_vwap
    assert abs(wrong - share.market_vwap) > 0.001


def test_the_two_windows_agree_when_arrival_lands_on_a_bar_boundary() -> None:
    """Which is the other half: the discrepancy is the mid-bar arrival, not the definitions."""
    series = _series([1000.0] * 5, [100.0, 100.1, 100.2, 100.3, 100.4])
    order = _order([(minute(1) + timedelta(seconds=10), 400.0, 100.15)], arrival=minute(1))
    assert own_share(order, series).rate == pytest.approx(participation_rate(order, series))


def test_an_explicit_window_overrides_the_default(book: tuple[Order, BarSeries]) -> None:
    order, series = book
    narrow = own_share(order, series, start=minute(2), end=minute(4))
    assert narrow.start == minute(2)
    assert narrow.end == minute(4)
    assert narrow.quantity == 600.0
    # Not a larger rate: the narrow window drops 400 of the order's shares and
    # 1,400 of the market's, so it goes the other way -- 0.261 against 0.278.
    # A window is not a dial on participation, which is the reason to assert
    # the figure rather than a direction.
    assert narrow.rate == pytest.approx(600.0 / 2300.0)
    assert narrow.rate < own_share(order, series).rate


# --- wiring -----------------------------------------------------------------


def test_the_new_benchmark_is_an_interval_benchmark() -> None:
    assert Benchmark.INTERVAL_VWAP_EX_OWN.is_interval
    assert Benchmark("interval_vwap_ex_own") is Benchmark.INTERVAL_VWAP_EX_OWN


def test_benchmark_price_and_score_order_agree_with_own_share(
    book: tuple[Order, BarSeries],
) -> None:
    order, series = book
    share = own_share(order, series)
    assert benchmark_price(order, series, Benchmark.INTERVAL_VWAP_EX_OWN) == share.ex_own_vwap
    assert (
        score_order(order, series, Benchmark.INTERVAL_VWAP_EX_OWN).benchmark_price
        == share.ex_own_vwap
    )


def test_own_share_reports_the_window_it_used(book: tuple[Order, BarSeries]) -> None:
    order, series = book
    share = own_share(order, series)
    assert (share.start, share.end) == order_window(order, series)
    assert isinstance(share, OwnShare)
