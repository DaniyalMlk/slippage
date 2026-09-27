"""Tests for post-trade mark-outs.

Two kinds of test, and the split matters. On a book generated with no market and
no idiosyncratic noise the measurement is exact arithmetic, so the persisted move
at each horizon has to equal the impact the generator put there to nine decimal
places. That pins the signs, the windows and the decomposition, which is
everything that can be wrong without being noticed.

The rest is statistics, and the statistics are the reason the module exists in
the shape it does. A mark-out at an hour is a few basis points of signal under
about twenty-seven of market move, so the per-order figure is noise and the
aggregate needs the market taken out of it. Those tests compare the adjusted and
unadjusted estimators against a known truth rather than asserting either one is
close to it, because "close" on one seed is not evidence and the comparison is.

``examples/mark_outs.py`` runs the same comparison over twenty-five books and
prints the table the README quotes.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from slippage.calibration import fit_permanent
from slippage.exceptions import InsufficientDataError, ValidationError
from slippage.reversion import (
    DEFAULT_HORIZONS,
    IMPACT_FLOOR_BPS,
    permanent_moves_from_orders,
    price_reversion,
    reversion_profile,
)
from slippage.series import BarSeries
from slippage.synthetic import DecayingBook, decaying_book
from slippage.types import Bar, Fill, Order, Side

HOUR = timedelta(hours=1)
OPEN = datetime(2026, 3, 2, 9, 30)
MINUTE = timedelta(minutes=1)


def quiet_book(
    *,
    seed: int = 3,
    orders: int = 4,
    permanent_bps: float = 4.0,
    temporary_bps: float = 6.0,
) -> DecayingBook:
    """A book with the market and the idiosyncratic noise switched off.

    Where the measurement is exact arithmetic rather than an estimate.
    """
    return decaying_book(
        np.random.default_rng(seed),
        orders=orders,
        permanent_bps=permanent_bps,
        temporary_bps=temporary_bps,
        market_volatility_bps=0.0,
        idiosyncratic_volatility_bps=0.0,
    )


def flat_series(
    *, minutes: int = 120, price: float = 100.0, start: datetime = OPEN
) -> BarSeries:
    return BarSeries(
        [
            Bar(start + i * MINUTE, price, price * 1.001, price * 0.999, price, 5_000.0)
            for i in range(minutes)
        ]
    )


def order_on(series: BarSeries, *, side: Side = Side.BUY, minutes: int = 5) -> Order:
    arrival = series.start + 10 * MINUTE
    return Order(
        symbol="ACME",
        side=side,
        quantity=1_000.0,
        decision_time=arrival,
        arrival_time=arrival,
        fills=tuple(
            Fill(
                timestamp=arrival + i * MINUTE + timedelta(seconds=30),
                quantity=200.0,
                price=series.price_at(arrival + i * MINUTE),
            )
            for i in range(minutes)
        ),
    )


# -- exact, on a book with no noise in it -------------------------------------


def test_the_persisted_move_is_the_impact_the_generator_put_there() -> None:
    """Exact arithmetic, so asserted exactly.

    The generator builds impact linearly in the executed fraction and decays it
    exponentially afterwards, and the mark-out reads it straight back off the
    prints. Anything wrong with the sign, the completion window or the
    decomposition breaks this by basis points, not by rounding.
    """
    book = quiet_book()
    for order in book.orders.values():
        profile = price_reversion(order, book.bars[order.symbol])
        assert profile.impact_bps == pytest.approx(
            book.permanent_bps + book.temporary_bps, abs=1e-9
        )
        for mark in profile.marks:
            expected = book.permanent_bps + book.temporary_bps * 2.0 ** (
                -mark.horizon.total_seconds()
                / book.half_life.total_seconds()
            )
            assert mark.permanent_bps == pytest.approx(expected, abs=1e-9)
            assert mark.reverted_bps == pytest.approx(
                profile.impact_bps - expected, abs=1e-9
            )


def test_the_decomposition_adds_up_at_every_horizon() -> None:
    """``impact = permanent + reverted``, by construction and therefore always.

    Worth asserting because the two are computed from different intervals and a
    benchmark subtracted from one and not the other would break it while leaving
    both looking like basis points.
    """
    book = decaying_book(np.random.default_rng(11), orders=6)
    for order in book.orders.values():
        for benchmark in (None, book.index):
            profile = price_reversion(
                order, book.bars[order.symbol], benchmark=benchmark
            )
            for mark in profile.marks:
                assert mark.permanent_bps + mark.reverted_bps == pytest.approx(
                    profile.impact_bps, abs=1e-12
                )
                assert mark.impact_bps == pytest.approx(profile.impact_bps)


def test_a_sell_is_measured_the_same_way_up() -> None:
    """Both sides are in the book and the alternation is the point.

    An estimator that forgot to sign for the side would return about zero on
    average over a book with equal sides, which looks like "no reversion" rather
    than like a bug. So the sides are checked to agree order by order instead.
    """
    book = quiet_book(orders=6)
    buys = [
        price_reversion(order, book.bars[order.symbol]).at(HOUR).permanent_bps
        for order in book.orders.values()
        if order.side is Side.BUY
    ]
    sells = [
        price_reversion(order, book.bars[order.symbol]).at(HOUR).permanent_bps
        for order in book.orders.values()
        if order.side is Side.SELL
    ]
    assert buys and sells
    assert all(value > 0.0 for value in buys + sells)
    assert statistics_close(buys + sells)


def statistics_close(values: list[float]) -> bool:
    return max(values) - min(values) < 1e-9


def test_a_price_that_does_not_move_reverts_nothing() -> None:
    series = flat_series()
    profile = price_reversion(order_on(series), series, horizons=(timedelta(minutes=5),))
    assert profile.impact_bps == pytest.approx(0.0, abs=1e-12)
    assert profile.at(timedelta(minutes=5)).reverted_bps == pytest.approx(0.0, abs=1e-12)
    # And the fraction is withheld rather than reported as zero over zero.
    assert profile.at(timedelta(minutes=5)).reverted_fraction is None


def test_a_fraction_is_withheld_when_the_impact_is_inside_the_noise() -> None:
    """The denominator is a measured move, so a small one makes the ratio a lie.

    At a tenth of a basis point of impact, a tenth of a basis point of reversion
    is a reverted fraction of one — and at a hundredth it is a fraction of ten.
    Neither is a statement about the order.
    """
    book = quiet_book(permanent_bps=0.1, temporary_bps=0.1)
    order = next(iter(book.orders.values()))
    profile = price_reversion(order, book.bars[order.symbol])
    assert abs(profile.impact_bps) < IMPACT_FLOOR_BPS
    assert all(mark.reverted_fraction is None for mark in profile.marks)
    bigger = quiet_book(permanent_bps=4.0, temporary_bps=6.0)
    order = next(iter(bigger.orders.values()))
    profile = price_reversion(order, bigger.bars[order.symbol])
    fraction = profile.at(HOUR).reverted_fraction
    assert fraction is not None
    assert fraction == pytest.approx(0.6, abs=0.01)


# -- the market, which is bigger than the thing being measured -----------------


def test_a_pure_market_move_is_removed_and_shows_up_without_the_benchmark() -> None:
    """The failure this measurement has by default, and the fix, in one test.

    The book here has no impact at all: every print is the market factor times
    beta. Unadjusted, the mark-outs report basis points of impact and reversion
    that are entirely somebody else's move. Adjusted, they report zero to
    floating point.
    """
    book = decaying_book(
        np.random.default_rng(5),
        orders=8,
        permanent_bps=0.0,
        temporary_bps=0.0,
        idiosyncratic_volatility_bps=0.0,
        market_volatility_bps=4.0,
    )
    unadjusted = reversion_profile(list(book.orders.values()), book.bars)
    adjusted = reversion_profile(
        list(book.orders.values()), book.bars, benchmark=book.index
    )
    assert adjusted.benchmark_adjusted
    assert not unadjusted.benchmark_adjusted
    for point in adjusted.points:
        assert point.mean_permanent_bps == pytest.approx(0.0, abs=1e-9)
        assert point.mean_reverted_bps == pytest.approx(0.0, abs=1e-9)
    assert max(abs(point.mean_permanent_bps) for point in unadjusted.points) > 1.0


def test_a_beta_scales_what_is_removed_and_is_exact_only_at_one() -> None:
    """Which is a property of subtracting simple returns, and worth knowing.

    At a beta of one the stock's price is the benchmark's times a constant, so the
    two returns are identical and the subtraction cancels to 2e-12. At any other
    beta it cancels only to first order, because a simple return is not linear in
    the underlying move: the residual is of the order of the square of the market
    move, measured at 0.038 basis points here against a market that wandered
    thirty. Small against any impact worth measuring, and not zero, so the test
    asserts the size rather than pretending it is.

    Using the wrong beta is the failure that matters, and it is two orders of
    magnitude larger: 8.0 basis points on the same book.
    """
    book = decaying_book(
        np.random.default_rng(6),
        orders=6,
        permanent_bps=0.0,
        temporary_bps=0.0,
        idiosyncratic_volatility_bps=0.0,
        market_volatility_bps=4.0,
        beta=0.5,
    )
    residuals = [
        abs(
            price_reversion(
                order, book.bars[order.symbol], benchmark=book.index, beta=0.5
            ).impact_bps
        )
        for order in book.orders.values()
    ]
    assert max(residuals) < 0.05
    mistaken = [
        abs(
            price_reversion(
                order, book.bars[order.symbol], benchmark=book.index, beta=1.0
            ).impact_bps
        )
        for order in book.orders.values()
    ]
    assert max(mistaken) > 100.0 * max(residuals)


def test_a_beta_with_no_benchmark_is_refused() -> None:
    """Silently ignoring it would leave the caller believing they had adjusted."""
    series = flat_series()
    with pytest.raises(ValidationError, match="no benchmark to apply it to"):
        price_reversion(order_on(series), series, beta=0.6)


def test_the_benchmark_adjustment_earns_its_place_on_one_book() -> None:
    """The comparison, on a single seed, against a truth both estimators know.

    The claim being tested is not that the adjusted figure is accurate — it is
    that it is sharper, which is a comparison and survives one seed in a way an
    accuracy claim does not. ``examples/mark_outs.py`` runs it over twenty-five.
    """
    book = decaying_book(np.random.default_rng(1000))
    orders = list(book.orders.values())
    unadjusted = reversion_profile(orders, book.bars)
    adjusted = reversion_profile(orders, book.bars, benchmark=book.index)
    assert adjusted.points[-1].standard_error < 0.5 * unadjusted.points[-1].standard_error
    assert adjusted.impact_standard_error < 0.5 * unadjusted.impact_standard_error
    # And *not* that the adjusted point estimate is the closer one on this seed.
    # It is not: 9.51 against 9.91 for a truth of 10. With standard errors of 0.40
    # and 1.69 that is exactly the kind of thing one draw does, which is why the
    # accuracy comparison belongs in the example over twenty-five books and the
    # precision comparison belongs here.
    assert abs(unadjusted.mean_impact_bps - 10.0) < unadjusted.impact_standard_error


# -- horizons past the end of the data ----------------------------------------


def test_a_horizon_past_the_series_is_flagged_rather_than_read_off_the_close() -> None:
    """``price_at`` clamps, which is right for a lookup and wrong for a mark-out.

    Without the flag, "the data stops here" and "the price stopped moving" are the
    same number, and the second one is the more flattering.
    """
    series = flat_series(minutes=30)
    profile = price_reversion(
        order_on(series),
        series,
        horizons=(timedelta(minutes=5), timedelta(minutes=60)),
    )
    near, far = profile.marks
    assert near.observed
    assert not far.observed
    assert far.moment > series.end
    assert far.price == pytest.approx(series.price_at(series.end))
    assert profile.observed == (near,)


def test_aggregation_counts_only_the_observed_marks() -> None:
    book = decaying_book(np.random.default_rng(8), orders=6)
    horizons = (*DEFAULT_HORIZONS, timedelta(hours=9))
    profile = reversion_profile(list(book.orders.values()), book.bars, horizons=horizons)
    assert profile.at(timedelta(minutes=5)).orders == 6
    unreachable = profile.at(timedelta(hours=9))
    assert unreachable.orders == 0
    assert math.isnan(unreachable.mean_permanent_bps)
    assert math.isinf(unreachable.standard_error)


def test_orders_that_never_traded_are_skipped_not_refused() -> None:
    """A parent order with no fills is a normal line on a blotter."""
    book = decaying_book(np.random.default_rng(9), orders=4)
    orders = list(book.orders.values())
    empty = Order(
        symbol=orders[0].symbol,
        side=Side.BUY,
        quantity=100.0,
        decision_time=orders[0].decision_time,
        arrival_time=orders[0].arrival_time,
    )
    profile = reversion_profile([*orders, empty], book.bars)
    assert profile.orders == 4
    with pytest.raises(InsufficientDataError, match="no completion time"):
        price_reversion(empty, book.bars[empty.symbol])
    with pytest.raises(InsufficientDataError, match="no orders with fills"):
        reversion_profile([empty], book.bars)


# -- the fitted decay ---------------------------------------------------------


def test_the_fitted_half_life_recovers_a_noiseless_decay() -> None:
    """Separable least squares on a curve that is exactly the model.

    With no noise the five points lie on the curve being fitted, so the half-life
    comes back to a fraction of a second and the asymptote and amplitude to a
    thousandth of a basis point. That is the test of the fitter; the noisy case
    is the test of the estimator.
    """
    book = quiet_book(orders=6)
    profile = reversion_profile(list(book.orders.values()), book.bars)
    decay = profile.decay()
    assert decay.half_life.total_seconds() == pytest.approx(
        book.half_life.total_seconds(), abs=1.0
    )
    assert decay.asymptote_bps == pytest.approx(book.permanent_bps, abs=1e-3)
    assert decay.amplitude_bps == pytest.approx(book.temporary_bps, abs=1e-3)
    assert decay.r_squared > 0.9999
    assert decay.points == 5
    fraction = decay.permanent_fraction
    assert fraction is not None
    assert fraction == pytest.approx(0.4, abs=1e-4)


def test_a_curve_that_does_not_decay_is_refused_rather_than_fitted() -> None:
    """An impact that grows with the horizon is information, not a half-life.

    The book here has a negative temporary component, so the price keeps moving
    away after the order finishes. A fitter that returned a half-life for it
    would be describing continuation as decay.
    """
    book = quiet_book(orders=6, permanent_bps=10.0, temporary_bps=-6.0)
    profile = reversion_profile(list(book.orders.values()), book.bars)
    with pytest.raises(InsufficientDataError, match="does not decay"):
        profile.decay()


def test_three_horizons_are_needed_for_three_parameters() -> None:
    book = quiet_book(orders=4)
    profile = reversion_profile(
        list(book.orders.values()),
        book.bars,
        horizons=(timedelta(minutes=1), timedelta(minutes=5)),
    )
    with pytest.raises(InsufficientDataError, match="at least three horizons"):
        profile.decay()


def test_the_permanent_fraction_is_withheld_on_a_negligible_total() -> None:
    book = quiet_book(orders=4, permanent_bps=0.1, temporary_bps=0.2)
    decay = reversion_profile(
        list(book.orders.values()), book.bars
    ).decay()
    assert decay.permanent_fraction is None


# -- the pair fit_permanent wanted --------------------------------------------


def test_the_moves_feed_fit_permanent_and_recover_its_coefficient() -> None:
    """The gap this module was written to close.

    ``fit_permanent`` documented its second argument as the moves that persisted
    and left the caller to walk the bars, choose a horizon and sign for the side.
    With the market removed, the coefficient it recovers is within a tenth of the
    truth and its t-statistic is nearly five.
    """
    book = decaying_book(np.random.default_rng(1000))
    orders = list(book.orders.values())
    quantities, moves = permanent_moves_from_orders(
        orders, book.bars, horizon=timedelta(minutes=45), benchmark=book.index
    )
    assert len(quantities) == len(moves) == len(orders)
    assert all(quantity > 0.0 for quantity in quantities)
    estimate = fit_permanent(quantities, moves)
    # Truth: permanent_bps of the arrival price per whole order, and every order
    # is the same size in the generator's distribution only on average, so the
    # comparison is against the regression's own scale rather than a constant.
    assert estimate.value > 0.0
    assert estimate.t_stat > 3.0
    unadjusted = fit_permanent(
        *permanent_moves_from_orders(orders, book.bars, horizon=timedelta(minutes=45))
    )
    assert unadjusted.t_stat < estimate.t_stat


def test_the_moves_are_signed_as_a_cost_and_the_quantities_are_not() -> None:
    """The convention the other fits in ``calibration`` use.

    A buy that pushed the price up and a sell that pushed it down both contribute
    positively, so both sides can go into one regression. Signing the quantity
    instead would give the same gamma and a different intercept story, and would
    disagree with ``fit_linear_temporary`` sitting next to it.
    """
    book = quiet_book(orders=6)
    quantities, moves = permanent_moves_from_orders(
        book.orders.values(),
        book.bars,
        horizon=timedelta(minutes=45),
    )
    assert all(quantity > 0.0 for quantity in quantities)
    assert all(move > 0.0 for move in moves)
    order = next(iter(book.orders.values()))
    profile = price_reversion(
        order, book.bars[order.symbol], horizons=(timedelta(minutes=45),)
    )
    assert profile.permanent_per_share(timedelta(minutes=45)) == pytest.approx(
        profile.at(timedelta(minutes=45)).permanent_bps / 1e4 * profile.arrival_price
    )


def test_orders_whose_mark_out_falls_off_the_end_are_left_out() -> None:
    """Rather than contributing a zero move standing in for no measurement."""
    book = decaying_book(np.random.default_rng(13), orders=6)
    with pytest.raises(InsufficientDataError, match="bars end too soon"):
        permanent_moves_from_orders(
            book.orders.values(), book.bars, horizon=timedelta(hours=9)
        )


# -- refusals -----------------------------------------------------------------


def test_the_boundary_conditions() -> None:
    series = flat_series()
    order = order_on(series)
    with pytest.raises(ValidationError, match="at least one horizon"):
        price_reversion(order, series, horizons=())
    with pytest.raises(ValidationError, match="is positive"):
        price_reversion(order, series, horizons=(timedelta(0),))
    with pytest.raises(ValidationError, match="is positive"):
        price_reversion(order, series, horizons=(-HOUR,))
    with pytest.raises(ValidationError, match="no mark-out at"):
        price_reversion(order, series, horizons=(HOUR,)).at(timedelta(minutes=5))
    with pytest.raises(ValidationError, match="no bars for"):
        reversion_profile([order], {})
    with pytest.raises(ValidationError, match="no bars for"):
        permanent_moves_from_orders([order], {}, horizon=HOUR)
    profile = reversion_profile([order], {"ACME": series})
    with pytest.raises(ValidationError, match="no mark-out at"):
        profile.at(timedelta(hours=3))


def test_a_benchmark_may_be_one_series_or_one_per_symbol() -> None:
    book = decaying_book(np.random.default_rng(14), orders=4)
    orders = list(book.orders.values())
    shared = reversion_profile(orders, book.bars, benchmark=book.index)
    per_symbol = reversion_profile(
        orders, book.bars, benchmark={order.symbol: book.index for order in orders}
    )
    assert shared.mean_impact_bps == pytest.approx(per_symbol.mean_impact_bps)


def test_the_realised_cost_is_reported_adjusted_and_not() -> None:
    """What was paid is a fact; what it would have been is a counterfactual.

    Both are useful and they are different numbers, so both are fields. With no
    benchmark they are equal, which is the case that would hide a mix-up.
    """
    book = decaying_book(np.random.default_rng(15), orders=4)
    order = next(iter(book.orders.values()))
    plain = price_reversion(order, book.bars[order.symbol])
    assert plain.realised_bps == pytest.approx(plain.adjusted_realised_bps)
    adjusted = price_reversion(order, book.bars[order.symbol], benchmark=book.index)
    assert adjusted.realised_bps == pytest.approx(plain.realised_bps)
    assert adjusted.adjusted_realised_bps != pytest.approx(adjusted.realised_bps)


def test_the_headline_fraction_walks_back_past_horizons_with_no_data() -> None:
    """A NaN here survives a JSON round trip and fails a strict parser downstream.

    The longest horizon asked for is routinely past the end of the bars, where
    there is no mean to take a fraction of. Returning NaN would be read back by
    ``json.loads`` without complaint and rejected by anything stricter, so the
    fraction comes from the last horizon that had orders in it.
    """
    book = decaying_book(np.random.default_rng(21), orders=6)
    horizons = (*DEFAULT_HORIZONS, timedelta(hours=9))
    profile = reversion_profile(list(book.orders.values()), book.bars, horizons=horizons)
    assert profile.points[-1].orders == 0
    fraction = profile.reverted_fraction
    assert fraction is not None
    assert not math.isnan(fraction)
    assert fraction == pytest.approx(
        profile.at(timedelta(minutes=60)).reverted_fraction(profile.mean_impact_bps)
    )
