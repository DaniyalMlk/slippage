"""The options nothing had ever chosen, each exercised by a test that can fail.

``tests/test_options.py`` asks mechanically whether every documented option is
supplied somewhere. These are the answers for the six it found unsupplied, and
each one checks what the option *governs* rather than merely passing it: a call
that supplies an argument and asserts nothing would satisfy the gate and prove
nothing, which is the failure mode a gate like that invites.
"""

from __future__ import annotations

import itertools
import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from slippage.exceptions import ValidationError
from slippage.routing import (
    Allocation,
    Outflow,
    RoutingProblem,
    Venue,
    fill_distribution,
    marginal_posted,
    venue_fill,
)
from slippage.synthetic import (
    LATEST_START,
    LONGEST_EXECUTION,
    SESSION_MINUTES,
    SESSION_OPEN,
    decaying_book,
    synthetic_book,
)


class TestTheExecutionWindow:
    """``decaying_book(execution_minutes=)``: the length of the order, in bars."""

    @pytest.mark.parametrize("minutes", [1, 5, 20, 120, LONGEST_EXECUTION])
    def test_the_window_is_the_number_of_fills_and_the_span_they_cover(self, minutes: int) -> None:
        book = decaying_book(
            np.random.default_rng(3),
            orders=4,
            execution_minutes=minutes,
            market_volatility_bps=0.0,
            idiosyncratic_volatility_bps=0.0,
        )
        for order in book.orders.values():
            assert len(order.fills) == minutes
            span = order.fills[-1].timestamp - order.fills[0].timestamp
            assert span == timedelta(minutes=minutes - 1)
            # Every fill prices off a bar inside the session, which is the
            # thing the cap exists to guarantee.
            assert order.fills[-1].timestamp < book.bars[order.symbol].end
            assert sum(f.quantity for f in order.fills) == pytest.approx(order.quantity)

    @pytest.mark.parametrize("minutes", [LONGEST_EXECUTION + 1, 500, 10_000])
    def test_a_window_that_would_not_fit_is_refused_with_the_limit_named(
        self, minutes: int
    ) -> None:
        with pytest.raises(ValidationError, match=f"caps it at {LONGEST_EXECUTION}"):
            decaying_book(np.random.default_rng(3), orders=4, execution_minutes=minutes)

    @pytest.mark.parametrize("minutes", [0, -1, -400])
    def test_the_other_end_of_the_guard_is_unchanged(self, minutes: int) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            decaying_book(np.random.default_rng(3), orders=4, execution_minutes=minutes)

    @pytest.mark.parametrize("seed", range(8))
    def test_the_limit_does_not_depend_on_the_draw(self, seed: int) -> None:
        # What was wrong before was not only the missing guard but that the
        # failure moved with the seed: the window was measured against the start
        # drawn rather than the latest one available. At the cap every seed has
        # to work, and one past it every seed has to be refused.
        decaying_book(np.random.default_rng(seed), orders=12, execution_minutes=LONGEST_EXECUTION)
        with pytest.raises(ValidationError):
            decaying_book(
                np.random.default_rng(seed), orders=12, execution_minutes=LONGEST_EXECUTION + 1
            )

    def test_the_cap_is_the_arithmetic_and_not_a_margin(self) -> None:
        # The constant is derivable, so it is derived here rather than copied:
        # an order starting as late as the generator allows reads bars up to
        # start + execution_minutes - 1, and that index has to be in the
        # session. One more minute would not fit.
        latest = LATEST_START - 1
        assert latest + LONGEST_EXECUTION - 1 == SESSION_MINUTES - 1
        assert latest + (LONGEST_EXECUTION + 1) - 1 == SESSION_MINUTES


class TestTheSessionDate:
    """``synthetic_book(day=)`` and ``decaying_book(day=)``: which session this is."""

    @pytest.mark.parametrize("day", [date(2024, 1, 2), date(2026, 3, 2), date(2031, 12, 31)])
    def test_every_timestamp_the_synthetic_book_emits_is_on_that_session(self, day: date) -> None:
        book = synthetic_book(np.random.default_rng(4), symbols=3, orders=12, day=day)
        opening = datetime.combine(day, SESSION_OPEN)
        close = opening + timedelta(minutes=SESSION_MINUTES)
        for symbol, series in book.bars.items():
            assert series.start == opening, symbol
            assert series.end <= close + timedelta(minutes=1)
            assert series.start.date() == day
            assert (series.end - timedelta(minutes=1)).date() == day
        for order in book.orders.values():
            assert order.decision_time.date() == day
            assert order.arrival_time.date() == day
            for fill in order.fills:
                assert opening <= fill.timestamp < close

    @pytest.mark.parametrize("day", [date(2024, 1, 2), date(2026, 3, 2), date(2031, 12, 31)])
    def test_the_decaying_book_moves_its_index_series_too(self, day: date) -> None:
        # The index is the half of this generator a caller is most likely to
        # pass somewhere else -- it is what the benchmark adjustment reads -- so
        # an index left on the default date while the orders moved would be a
        # mark-out measured against another day.
        book = decaying_book(np.random.default_rng(5), orders=6, day=day)
        opening = datetime.combine(day, SESSION_OPEN)
        assert book.index.start == opening
        assert (book.index.end - timedelta(minutes=1)).date() == day
        for symbol, series in book.bars.items():
            assert series.start == opening, symbol
        for order in book.orders.values():
            assert order.arrival_time.date() == day
            assert all(f.timestamp.date() == day for f in order.fills)

    def test_the_date_moves_the_prints_and_nothing_else(self) -> None:
        # Two sessions a year apart, same seed: every price identical, every
        # timestamp shifted by exactly the gap between the two openings. A
        # generator that fed the date into its own randomness would fail the
        # first half, and one that dated only its bars would fail the second.
        early = decaying_book(np.random.default_rng(8), orders=5, day=date(2026, 3, 2))
        late = decaying_book(np.random.default_rng(8), orders=5, day=date(2027, 3, 2))
        shift = datetime.combine(date(2027, 3, 2), SESSION_OPEN) - datetime.combine(
            date(2026, 3, 2), SESSION_OPEN
        )
        assert shift == timedelta(days=365)
        assert early.orders.keys() == late.orders.keys()
        for symbol in early.orders:
            before, after = early.orders[symbol], late.orders[symbol]
            assert after.arrival_time - before.arrival_time == shift
            assert [f.price for f in after.fills] == [f.price for f in before.fills]
            assert [f.timestamp + shift for f in before.fills] == [f.timestamp for f in after.fills]

    def test_a_session_opening_is_not_midnight(self) -> None:
        # Guards the assertions above against a generator that dropped the time
        # of day: every timestamp would still land on the right date.
        assert time(0, 0) != SESSION_OPEN
        book = synthetic_book(np.random.default_rng(1), symbols=2, orders=4, day=date(2026, 3, 2))
        assert book.bars["SYM00"].start.time() == SESSION_OPEN


class TestTheImpactCoefficient:
    """``synthetic_book(impact_y=)``: the prefactor of the generating law."""

    def test_the_fills_carry_the_coefficient_they_were_generated_with(self) -> None:
        # Read off the fills rather than through a fit. Each fill pays
        # `impact_y * sigma * sqrt(done / V)` plus a half spread over the
        # unaffected mid, so the first fill of an order -- where `done` is that
        # fill's own quantity -- pins the coefficient directly.
        for y in (0.0, 0.2, 0.7, 2.5):
            book = synthetic_book(np.random.default_rng(6), symbols=4, orders=20, impact_y=y)
            assert book.impact_y == y
            checked = 0
            for order in book.orders.values():
                if not order.fills:
                    continue
                fill = order.fills[0]
                series = book.bars[order.symbol]
                mid = series.price_at(fill.timestamp.replace(second=0, microsecond=0))
                sigma = book.daily_volatility[order.symbol]
                volume = book.daily_volume[order.symbol]
                expected = mid * (
                    1.0 + order.side.sign * (y * sigma * math.sqrt(fill.quantity / volume) + 0.0002)
                )
                # Exactly, including the generator's rounding to four places:
                # the fill price it writes is a price a tape could carry, and
                # reproducing it to the tick is a stronger claim than a
                # tolerance that would also pass on a wrong coefficient at a
                # small enough sigma.
                assert fill.price == round(expected, 4)
                checked += 1
            assert checked > 10

    def test_the_generator_is_linear_in_the_coefficient(self) -> None:
        # Worth stating because it is what a parametrised recovery test would
        # not tell you: the same seed at two coefficients gives fill prices that
        # differ by exactly the impact term, so a fit run at 0.1 is the fit run
        # at 0.7 shifted by a constant. Two coefficients that *look* like two
        # cases are one case, and a recovery check gains nothing by repeating
        # over them.
        zero = synthetic_book(np.random.default_rng(6), symbols=4, orders=20, impact_y=0.0)
        one = synthetic_book(np.random.default_rng(6), symbols=4, orders=20, impact_y=1.0)
        half = synthetic_book(np.random.default_rng(6), symbols=4, orders=20, impact_y=0.5)
        spread = 0.0
        for symbol, order in zero.orders.items():
            for a, b, c in zip(
                order.fills,
                one.orders[symbol].fills,
                half.orders[symbol].fills,
                strict=True,
            ):
                # Within the four-place rounding the generator applies, which is
                # the only reason this is not an equality.
                assert c.price == pytest.approx((a.price + b.price) / 2.0, abs=1e-4)
                assert a.quantity == b.quantity == c.quantity
                assert a.timestamp == b.timestamp == c.timestamp
                spread = max(spread, abs(b.price - a.price))
        # The rounding is 1e-4 and the impact is basis points of a price in the
        # tens, so the prices being compared differ by far more than the
        # tolerance above -- otherwise the interpolation would hold for a
        # generator that ignored the coefficient entirely.
        assert spread > 1e-2

    def test_a_coefficient_of_zero_leaves_the_half_spread_and_nothing_else(self) -> None:
        book = synthetic_book(np.random.default_rng(6), symbols=4, orders=20, impact_y=0.0)
        for order in book.orders.values():
            series = book.bars[order.symbol]
            for fill in order.fills:
                mid = series.price_at(fill.timestamp.replace(second=0, microsecond=0))
                assert fill.price == round(mid * (1.0 + order.side.sign * 0.0002), 4)


class TestTheLeftOutVenue:
    """``fill_distribution(exclude=)``: the leave-one-out passive fill."""

    @staticmethod
    def problem() -> tuple[RoutingProblem, Allocation]:
        venues = (
            Venue("A", Outflow((0.4, 0.35, 0.25)), rebate=0.2),
            Venue("B", Outflow((0.55, 0.3, 0.15))),
            Venue("C", Outflow((0.25, 0.3, 0.25, 0.2)), rebate=-0.05),
        )
        problem = RoutingProblem(
            target=5, venues=venues, taker_cost=1.0, sweep_cost=2.2, unwind_cost=0.4
        )
        return problem, Allocation(taker=0, posted=(2, 2, 1))

    @pytest.mark.parametrize("left_out", [0, 1, 2])
    def test_it_is_the_convolution_of_the_venues_that_are_left(self, left_out: int) -> None:
        # The reference is built here by enumerating the product of the other
        # venues' supports, which shares no arithmetic with the module's
        # convolution.
        problem, allocation = self.problem()
        kept = [
            venue_fill(venue.outflow, allocation.posted[index]).probabilities
            for index, venue in enumerate(problem.venues)
            if index != left_out
        ]
        expected = [0.0] * (sum(len(side) for side in kept) + 1)
        for combination in itertools.product(*(enumerate(side) for side in kept)):
            total = sum(lots for lots, _ in combination)
            weight = 1.0
            for _, probability in combination:
                weight *= probability
            expected[total] += weight
        got = fill_distribution(problem, allocation, exclude=left_out).probabilities
        assert len(got) <= len(expected)
        for lots, probability in enumerate(got):
            assert probability == pytest.approx(expected[lots], abs=1e-12)
        assert sum(got) == pytest.approx(1.0, abs=1e-12)

    def test_leaving_a_venue_out_is_not_the_same_distribution(self) -> None:
        # Otherwise the row above would pass against a function that ignored
        # its argument. The whole distribution shifts down, so the mean is the
        # thing to compare.
        problem, allocation = self.problem()
        whole = fill_distribution(problem, allocation)

        def mean(probabilities: tuple[float, ...]) -> float:
            return sum(lots * p for lots, p in enumerate(probabilities))

        for left_out in range(len(problem.venues)):
            without = fill_distribution(problem, allocation, exclude=left_out)
            assert mean(without.probabilities) < mean(whole.probabilities) - 1e-6

    def test_an_index_outside_the_venues_leaves_every_venue_in(self) -> None:
        # `exclude` is matched by equality against each index, so a value that
        # is not an index excludes nothing. Documented by being asserted: it is
        # the behaviour `marginal_posted` relies on never needing.
        problem, allocation = self.problem()
        whole = fill_distribution(problem, allocation).probabilities
        for outside in (len(problem.venues), 99, -1):
            assert fill_distribution(problem, allocation, exclude=outside).probabilities == whole

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_the_marginal_is_written_in_terms_of_this_distribution(self, index: int) -> None:
        # The claim the docstring makes, asserted: conditional on the extra lot
        # being reached, what the other venues do is the leave-one-out
        # distribution, and the marginal's reach probability is read off it.
        problem, allocation = self.problem()
        others = fill_distribution(problem, allocation, exclude=index)
        marginal = marginal_posted(problem, allocation, index)
        venue = problem.venues[index]
        reached = venue.outflow.tail(allocation.posted[index] + 1)
        assert marginal.reached == pytest.approx(reached, abs=1e-12)
        assert sum(others.probabilities) == pytest.approx(1.0, abs=1e-12)


class TestTheCoverageWidth:
    """``SimulatedValue.covers(errors=)``: how many standard errors wide."""

    @staticmethod
    def value(mean: float = 10.0, standard_error: float = 0.5, draws: int = 1000):  # type: ignore[no-untyped-def]
        from slippage.adaptive import SimulatedValue

        return SimulatedValue(mean=mean, standard_error=standard_error, draws=draws)

    @pytest.mark.parametrize("errors", [0.5, 1.0, 2.0, 3.0, 10.0])
    def test_the_interval_is_exactly_that_many_standard_errors_wide(self, errors: float) -> None:
        simulated = self.value()
        edge = simulated.mean + errors * simulated.standard_error
        assert simulated.covers(edge, errors=errors)
        assert simulated.covers(edge - 1e-9, errors=errors)
        assert not simulated.covers(edge + 1e-9, errors=errors)
        # And symmetrically below, which a one-sided comparison would fail.
        assert simulated.covers(simulated.mean - errors * simulated.standard_error, errors=errors)
        assert not simulated.covers(
            simulated.mean - errors * simulated.standard_error - 1e-9, errors=errors
        )

    def test_a_wider_interval_covers_everything_a_narrower_one_does(self) -> None:
        simulated = self.value()
        for value in (9.0, 9.6, 10.0, 10.4, 11.5):
            covered = [simulated.covers(value, errors=k) for k in (0.5, 1.0, 2.0, 3.0, 10.0)]
            assert covered == sorted(covered), value
        # A spread, so that a width which did nothing would not pass: the
        # narrowest of these intervals rejects a value the widest accepts.
        assert not simulated.covers(11.4, errors=0.5)
        assert simulated.covers(11.4, errors=10.0)

    @pytest.mark.parametrize("errors", [0.5, 3.0, 1e6])
    def test_no_width_rescues_an_interval_that_was_never_measured(self, errors: float) -> None:
        # The documented refusal, at every width: a single draw has no standard
        # error, and an interval of infinite width containing everything would
        # read the weakest evidence as the strongest.
        single = self.value(standard_error=math.inf, draws=1)
        assert not single.covers(10.0, errors=errors)
        assert not single.covers(1e9, errors=errors)
