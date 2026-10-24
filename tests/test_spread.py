"""Tests for spread estimation from bars.

Each estimator is first checked against its own definition on a series built
by hand, where the answer is known without simulating anything: a price
alternating between a fixed bid and a fixed ask with no efficient-price
movement at all has exactly the spread between them, and every one of the
three has to say so.

The rest is the measurement, and it is pinned with the same seeds the module
docstring quotes. The published biases are reproduced exactly because the
simulation is deterministic in its seed; where a bias is not distinguishable
from zero at the number of runs used, the test asserts that rather than
asserting a figure.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import ClassVar

import pytest

from slippage.exceptions import InsufficientDataError, ValidationError
from slippage.spread import (
    Accuracy,
    Estimator,
    SpreadEstimate,
    abdi_ranaldo_spread,
    corwin_schultz_spread,
    estimate_spread,
    midrange,
    roll_spread,
    score_estimators,
    simulate_quotes,
)
from slippage.types import Bar

DAILY_VOLATILITY = 0.02


def bouncing(days: int, *, spread: float = 0.01, seed: int = 1) -> list[Bar]:
    """A still efficient price and nothing but an independent bounce.

    The limit all three estimators are exact in. With the efficient price
    held essentially fixed and enough trades a day for both sides to be hit,
    the high is the ask, the low is the bid, and every close sits on one of
    them independently of the last.

    Corwin-Schultz and Abdi-Ranaldo are then exact in closed form, which is
    worth writing down because it is why this is a definition check and not
    another simulation. With ``r = log(ask/bid)``, a still price gives
    ``beta = 2 r^2`` and ``gamma = r^2``, so Corwin-Schultz's ``alpha``
    collapses to ``r`` and its spread to ``2(e^r - 1)/(1 + e^r)``, which is
    ``r`` to second order. Abdi-Ranaldo's two deviations are the same close
    against two identical midranges, so their product is the squared
    half-spread every time and the root is the spread. Only Roll keeps a
    sampling error, because its covariance is an average over signs that
    actually vary.
    """
    return simulate_quotes(days=days, spread=spread, volatility=1e-7, ticks=40, seed=seed)


def alternating(days: int, *, mid: float = 100.0, spread: float = 0.01) -> list[Bar]:
    """A bounce that is not independent: strictly bid, ask, bid, ask.

    Not the Roll model. Successive changes are perfectly anti-correlated
    rather than independent, so their covariance is the full squared spread
    rather than a quarter of it, and Roll reports twice the truth.
    """
    half = 0.5 * spread * mid
    bid, ask = mid - half, mid + half
    start = datetime(2024, 1, 1)
    return [
        Bar(
            timestamp=start + timedelta(days=day),
            open=bid if day % 2 else ask,
            high=ask,
            low=bid,
            close=ask if day % 2 else bid,
            volume=1.0,
        )
        for day in range(days)
    ]


class TestAgainstTheDefinitions:
    """A hand-built series where the spread is not in doubt."""

    def test_roll_recovers_a_pure_bounce(self) -> None:
        # Roll averages over signs that vary, so it keeps a sampling error of
        # order one over the root of the sample even when nothing else moves.
        got = roll_spread(bouncing(4000))
        assert got.spread == pytest.approx(0.01, rel=0.03)
        assert got.rejected == 0
        assert got.estimator is Estimator.ROLL

    def test_a_dependent_bounce_doubles_roll(self) -> None:
        # The model's assumption is that the side of each trade is
        # independent of the last. Make it strictly alternate and the
        # covariance is the whole squared spread rather than a quarter of
        # it, so the estimator reports twice the truth -- exactly, which is
        # worth pinning because it is a silent failure in the field.
        got = roll_spread(alternating(200, spread=0.01))
        assert got.spread == pytest.approx(0.02, rel=1e-4)
        assert got.rejected == 0

    def test_corwin_schultz_recovers_a_pure_bounce(self) -> None:
        got = corwin_schultz_spread(bouncing(200))
        assert got.spread == pytest.approx(0.01, rel=1e-3)
        assert got.rejected == 0
        assert got.windows == 199

    def test_abdi_ranaldo_recovers_a_pure_bounce(self) -> None:
        got = abdi_ranaldo_spread(bouncing(200))
        assert got.spread == pytest.approx(0.01, rel=1e-3)
        assert got.rejected == 0

    @pytest.mark.parametrize("spread", [0.0005, 0.002, 0.01, 0.05])
    def test_the_range_estimators_are_exact_in_the_still_limit(self, spread: float) -> None:
        bars = bouncing(200, spread=spread)
        for one in (Estimator.CORWIN_SCHULTZ, Estimator.ABDI_RANALDO):
            assert estimate_spread(bars, one).spread == pytest.approx(spread, rel=2e-3)
        # Roll on the same bars is unbiased and not exact.
        assert roll_spread(bars).spread == pytest.approx(spread, rel=0.2)

    def test_the_midrange_is_the_log_midpoint(self) -> None:
        bar = alternating(2)[0]
        assert midrange(bar) == pytest.approx(
            0.5 * (math.log(bar.high) + math.log(bar.low)), rel=0.0
        )
        assert math.exp(midrange(bar)) == pytest.approx(math.sqrt(bar.high * bar.low), rel=1e-15)

    def test_roll_against_its_formula_written_out(self) -> None:
        bars = simulate_quotes(days=40, spread=0.01, volatility=0.02, ticks=20, seed=3)
        changes = [
            math.log(bars[index + 1].close) - math.log(bars[index].close)
            for index in range(len(bars) - 1)
        ]
        mean = math.fsum(changes) / len(changes)
        pairs = len(changes) - 1
        covariance = (
            math.fsum(
                (changes[index] - mean) * (changes[index + 1] - mean) for index in range(pairs)
            )
            / pairs
        )
        got = roll_spread(bars, floor=False)
        want = -2.0 * math.copysign(math.sqrt(abs(covariance)), covariance)
        assert got.unfloored == pytest.approx(want, rel=1e-14)

    def test_the_half_spread_is_what_placement_takes(self) -> None:
        got = roll_spread(bouncing(100))
        assert got.half_spread == pytest.approx(0.5 * got.spread, rel=0.0)
        assert got.basis_points == pytest.approx(got.spread * 10000.0, rel=0.0)


class TestTheRefusals:
    """A positive covariance and a negative product are answers, not errors."""

    def test_a_trending_price_defeats_roll(self) -> None:
        # Positive serial correlation in the efficient price makes the sample
        # covariance positive, and there is no real root of a positive number
        # negated. The estimator says zero and records that it refused.
        start = datetime(2024, 1, 1)
        level = 100.0
        bars = []
        for day in range(60):
            # Runs of five days up then five down: successive changes agree
            # with each other four times in five, which is the opposite of
            # what a bounce does.
            level *= 1.002 if (day // 5) % 2 == 0 else 1.0 / 1.002
            bars.append(
                Bar(
                    timestamp=start + timedelta(days=day),
                    open=level,
                    high=level * 1.0005,
                    low=level * 0.9995,
                    close=level,
                    volume=1.0,
                )
            )
        got = roll_spread(bars)
        assert got.spread == 0.0
        assert got.rejected == 1
        assert got.rejection_rate == 1.0
        assert got.unfloored < 0.0

    def test_the_unfloored_value_is_kept(self) -> None:
        bars = simulate_quotes(
            days=60, spread=0.0002, volatility=DAILY_VOLATILITY, ticks=78, seed=17
        )
        floored = roll_spread(bars)
        signed = roll_spread(bars, floor=False)
        assert floored.spread >= 0.0
        assert floored.unfloored == pytest.approx(signed.spread, rel=0.0)
        assert signed.unfloored == pytest.approx(signed.spread, rel=0.0)

    def test_flooring_lift_reports_what_the_flooring_did(self) -> None:
        raw = corwin_schultz_spread(
            simulate_quotes(days=120, spread=0.0005, volatility=DAILY_VOLATILITY, ticks=78, seed=23)
        )
        assert raw.unfloored < 0.0 < raw.spread
        # Nothing positive underneath, so the reported spread is not a
        # raised estimate: it is the flooring and nothing else.
        assert math.isinf(raw.flooring_lift)

    def test_a_clean_series_floors_nothing(self) -> None:
        bars = bouncing(150)
        for one in Estimator:
            got = estimate_spread(bars, one)
            assert got.rejected == 0
            assert got.flooring_lift == pytest.approx(0.0, abs=1e-12)


class TestTheSimulation:
    """The model the estimators are derived under, and nothing else."""

    def test_it_is_deterministic_in_its_seed(self) -> None:
        first = simulate_quotes(days=10, spread=0.01, volatility=0.02, ticks=50, seed=9)
        second = simulate_quotes(days=10, spread=0.01, volatility=0.02, ticks=50, seed=9)
        assert [bar.close for bar in first] == [bar.close for bar in second]
        other = simulate_quotes(days=10, spread=0.01, volatility=0.02, ticks=50, seed=10)
        assert [bar.close for bar in first] != [bar.close for bar in other]

    def test_the_range_widens_with_the_spread(self) -> None:
        ranges = []
        for spread in (0.0, 0.005, 0.02, 0.05):
            bars = simulate_quotes(days=200, spread=spread, volatility=0.001, ticks=100, seed=4)
            ranges.append(math.fsum(math.log(bar.high / bar.low) for bar in bars) / len(bars))
        assert ranges == sorted(ranges)

    def test_the_observed_range_grows_with_the_tick_count(self) -> None:
        # The reason the high-low estimators depend on how often the market
        # trades: a path sampled at more points has a wider observed extreme,
        # and the estimators are derived from the continuous range.
        widths = []
        for ticks in (4, 20, 100, 500):
            bars = simulate_quotes(days=300, spread=0.0, volatility=0.02, ticks=ticks, seed=6)
            widths.append(math.fsum(math.log(bar.high / bar.low) for bar in bars) / len(bars))
        assert widths == sorted(widths)
        assert widths[-1] > 1.5 * widths[0]

    def test_a_zero_spread_leaves_the_close_inside_the_range(self) -> None:
        for bar in simulate_quotes(days=30, spread=0.0, volatility=0.02, ticks=40, seed=8):
            assert bar.low <= bar.close <= bar.high
            assert bar.low <= bar.open <= bar.high


class TestAccuracy:
    """Scored against a spread that is known, with the published figures."""

    @pytest.mark.parametrize(
        ("spread", "wanted"),
        [
            (0.001, {"roll": 350.99, "corwin-schultz": 627.33, "abdi-ranaldo": 119.53}),
            (0.005, {"roll": 14.43, "corwin-schultz": 80.14, "abdi-ranaldo": -13.47}),
            (0.020, {"roll": 0.33, "corwin-schultz": -3.65, "abdi-ranaldo": -0.30}),
        ],
    )
    def test_the_published_table(self, spread: float, wanted: dict[str, float]) -> None:
        rows = score_estimators(
            days=252,
            spread=spread,
            volatility=DAILY_VOLATILITY,
            runs=200,
            ticks=78,
            seed=1000,
        )
        for row in rows:
            assert isinstance(row, Accuracy)
            assert row.bias * 100.0 == pytest.approx(wanted[row.estimator.value], abs=0.02)

    def test_at_a_realistic_ratio_the_unfloored_estimate_is_negative(self) -> None:
        # The claim the module exists to make. A 10bp spread against a 2%
        # daily volatility is an ordinary liquid stock, and two of the three
        # estimators carry no information about it whatsoever: everything
        # they report is the flooring.
        rows = score_estimators(
            days=252,
            spread=0.001,
            volatility=DAILY_VOLATILITY,
            runs=200,
            ticks=78,
            seed=1000,
        )
        unfloored = {row.estimator.value: row.unfloored_mean for row in rows}
        assert unfloored["corwin-schultz"] == pytest.approx(-4.31e-4, abs=2e-6)
        assert unfloored["abdi-ranaldo"] == pytest.approx(-1.08e-4, abs=2e-6)
        assert unfloored["roll"] == pytest.approx(7.91e-4, abs=2e-6)
        for row in rows:
            assert row.mean > 0.0
            assert row.refusal_rate > 0.4

    def test_the_bias_shrinks_as_the_spread_becomes_resolvable(self) -> None:
        worst = []
        for spread in (0.001, 0.005, 0.020):
            rows = score_estimators(
                days=252,
                spread=spread,
                volatility=DAILY_VOLATILITY,
                runs=200,
                ticks=78,
                seed=1000,
            )
            worst.append(max(abs(row.bias) for row in rows))
        assert worst == sorted(worst, reverse=True)

    def test_unbiased_and_wide_against_tight_and_wrong(self) -> None:
        rows = {
            row.estimator.value: row
            for row in score_estimators(
                days=252,
                spread=0.020,
                volatility=DAILY_VOLATILITY,
                runs=200,
                ticks=78,
                seed=1000,
            )
        }
        standard_error = {name: row.rmse / math.sqrt(row.runs) for name, row in rows.items()}
        # Roll's bias is inside one standard error, so 200 runs cannot
        # distinguish it from zero. Corwin-Schultz's is nine of them.
        assert abs(rows["roll"].mean - 0.020) < standard_error["roll"]
        assert abs(rows["corwin-schultz"].mean - 0.020) > 5.0 * standard_error["corwin-schultz"]
        # And the ordering of dispersion runs the other way.
        assert rows["roll"].rmse > 2.5 * rows["corwin-schultz"].rmse
        assert rows["roll"].rmse > 2.5 * rows["abdi-ranaldo"].rmse

    def test_roll_stops_refusing_once_the_spread_is_large(self) -> None:
        rows = {
            row.estimator.value: row
            for row in score_estimators(
                days=252,
                spread=0.020,
                volatility=DAILY_VOLATILITY,
                runs=200,
                ticks=78,
                seed=1000,
            )
        }
        assert rows["roll"].refusal_rate == 0.0


class TestTheTickCount:
    """Corwin-Schultz is calibrated to a trading frequency."""

    PUBLISHED: ClassVar[dict[int, float]] = {
        8: -45.50,
        20: -22.76,
        39: -11.57,
        78: -3.88,
        195: 3.67,
        390: 6.98,
    }

    def scored(self, ticks: int) -> dict[str, Accuracy]:
        return {
            row.estimator.value: row
            for row in score_estimators(
                days=252,
                spread=0.020,
                volatility=DAILY_VOLATILITY,
                runs=100,
                ticks=ticks,
                seed=2000,
            )
        }

    @pytest.mark.parametrize("ticks", sorted(PUBLISHED))
    def test_the_published_column(self, ticks: int) -> None:
        got = self.scored(ticks)["corwin-schultz"]
        assert got.bias * 100.0 == pytest.approx(self.PUBLISHED[ticks], abs=0.02)

    def test_it_is_monotone_and_changes_sign(self) -> None:
        biases = [self.scored(ticks)["corwin-schultz"].bias for ticks in sorted(self.PUBLISHED)]
        assert biases == sorted(biases)
        assert biases[0] < 0.0 < biases[-1]
        assert max(biases) - min(biases) > 0.5

    def test_roll_does_not_depend_on_it_in_any_way_the_runs_can_see(self) -> None:
        for ticks in (8, 78, 390):
            row = self.scored(ticks)["roll"]
            standard_error = row.rmse / math.sqrt(row.runs)
            assert abs(row.mean - row.truth) < 2.5 * standard_error

    def test_abdi_ranaldo_has_the_same_dependence_an_order_smaller(self) -> None:
        biases = [self.scored(ticks)["abdi-ranaldo"].bias for ticks in sorted(self.PUBLISHED)]
        corwin = [self.scored(ticks)["corwin-schultz"].bias for ticks in sorted(self.PUBLISHED)]
        assert max(biases) - min(biases) < 0.1 * (max(corwin) - min(corwin))


class TestRefusalsOfInput:
    """What the module declines to answer."""

    def test_too_few_bars_for_a_covariance(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least three bars"):
            roll_spread(alternating(2))

    def test_too_few_bars_for_a_range_comparison(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least two bars"):
            corwin_schultz_spread(alternating(1))
        with pytest.raises(InsufficientDataError, match="at least two bars"):
            abdi_ranaldo_spread(alternating(1))

    def test_a_simulation_of_one_day(self) -> None:
        with pytest.raises(ValidationError, match="at least two days"):
            simulate_quotes(days=1, spread=0.01, volatility=0.02)

    def test_a_simulation_of_one_tick(self) -> None:
        with pytest.raises(ValidationError, match="at least two ticks"):
            simulate_quotes(days=10, spread=0.01, volatility=0.02, ticks=1)

    def test_a_negative_spread(self) -> None:
        with pytest.raises(ValidationError, match="a spread is not negative"):
            simulate_quotes(days=10, spread=-0.01, volatility=0.02)

    def test_a_volatility_of_nothing(self) -> None:
        with pytest.raises(ValidationError, match="volatility is positive"):
            simulate_quotes(days=10, spread=0.01, volatility=0.0)

    def test_no_runs(self) -> None:
        with pytest.raises(ValidationError, match="at least one run"):
            score_estimators(days=10, spread=0.01, volatility=0.02, runs=0)

    def test_negative_window_counts(self) -> None:
        with pytest.raises(ValidationError, match="cannot be negative"):
            SpreadEstimate(
                estimator=Estimator.ROLL,
                spread=0.01,
                windows=-1,
                rejected=0,
                unfloored=0.01,
            )


# The standard error, which used to be the rmse's rather than the dispersion's.
#
# rmse is measured about the truth, so it carries the bias; a standard error of
# the mean is made of the sampling half alone. The identity between them is
# exact, so it is asserted as an equality.


class TestStandardError:
    def test_the_rmse_splits_exactly_into_bias_and_dispersion(self) -> None:
        rows = score_estimators(
            days=252,
            spread=0.001,
            volatility=DAILY_VOLATILITY,
            runs=200,
            ticks=78,
            seed=1000,
        )
        for row in rows:
            population = row.dispersion**2 * (row.runs - 1) / row.runs
            assert row.rmse**2 == pytest.approx(
                (row.mean - row.truth) ** 2 + population, rel=1e-12, abs=0.0
            )

    def test_the_standard_error_is_the_dispersion_over_the_root_of_the_runs(self) -> None:
        rows = score_estimators(
            days=252,
            spread=0.001,
            volatility=DAILY_VOLATILITY,
            runs=200,
            ticks=78,
            seed=1000,
        )
        for row in rows:
            assert row.standard_error == pytest.approx(
                row.dispersion / math.sqrt(row.runs), rel=1e-12, abs=0.0
            )

    def test_the_old_reading_overstated_it_where_the_bias_was_large(self) -> None:
        """And agreed with the right one where it was not, which is why it stood.

        Both halves matter. At a spread-to-volatility ratio of 0.05 the biased
        estimator's standard error was overstated twelvefold; at 1.0, where the
        estimators are nearly unbiased, the two readings are within a third of
        each other. A test of the first alone would not say that the old
        formula was wrong rather than merely different.
        """
        inflations = {}
        for spread in (0.001, 0.020):
            rows = score_estimators(
                days=252,
                spread=spread,
                volatility=DAILY_VOLATILITY,
                runs=200,
                ticks=78,
                seed=1000,
            )
            inflations[spread] = {
                row.estimator.value: (row.rmse / math.sqrt(row.runs)) / row.standard_error
                for row in rows
            }
        assert inflations[0.001]["corwin-schultz"] == pytest.approx(12.50, abs=0.02)
        assert max(inflations[0.020].values()) < 1.35
        # The estimator whose bias is worst is the one most overstated, which
        # is the mechanism rather than a coincidence of this parameterisation.
        worst = max(inflations[0.001], key=lambda name: inflations[0.001][name])
        assert worst == "corwin-schultz"

    def test_a_single_run_has_no_dispersion_to_report(self) -> None:
        rows = score_estimators(
            days=60,
            spread=0.01,
            volatility=DAILY_VOLATILITY,
            runs=1,
            ticks=78,
            seed=7,
        )
        for row in rows:
            assert row.dispersion == 0.0
            assert row.standard_error == 0.0
            # And the rmse is then the bias exactly, there being nothing else.
            assert row.rmse == pytest.approx(abs(row.mean - row.truth), rel=1e-12, abs=0.0)
