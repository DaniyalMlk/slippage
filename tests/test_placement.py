"""Limit order placement: three exact identities, and a biased simulation.

The closed forms here have unusually sharp things to be checked against.

The fill probability is exactly twice the probability of merely ending below
the limit, which is the reflection principle and is a statement about an
integer. Conditional on filling, the expected terminal mid is exactly the
limit price, which is a statement about a ratio of two closed forms that have
to cancel. And the whole driftless expected cost is ``(h+f)(1-p) - r p`` with
the distance appearing nowhere else, which is a statement about every other
term cancelling against the adverse selection.

The simulation is the independent route, and it is *biased*: a path checked at
its own time steps misses the excursions between them. The tests therefore
measure the bias and its rate rather than asserting agreement, and use the
continuity correction where it applies — which is to the probability and not
to the mean.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from slippage.exceptions import ValidationError
from slippage.placement import (
    MONITORING_BETA,
    Placement,
    evaluate,
    frontier,
    moments,
    monitoring_shift,
    simulate_placement,
)

VOL = 0.20
YEAR = 1.0
DAY = 1.0 / 252.0


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def market(distance: float, horizon: float = DAY, drift: float = 0.0) -> Placement:
    """A five basis point half spread, a basis point of fee and of rebate."""
    return Placement(
        distance=distance,
        horizon=horizon,
        volatility=VOL,
        drift=drift,
        half_spread=0.0005,
        taker_fee=0.0001,
        maker_rebate=0.0001,
    )


class TestFillProbability:
    """The running minimum, not the terminal value, and the factor is exactly two."""

    @pytest.mark.parametrize("distance", [0.05, 0.1, 0.2, 0.4])
    def test_the_factor_is_exactly_two(self, distance: float) -> None:
        placement = Placement(distance=distance, horizon=YEAR, volatility=VOL)
        running = evaluate(placement).fill_probability
        terminal = normal_cdf(-distance / placement.deviation)
        assert running / terminal == pytest.approx(2.0, abs=1e-12)

    @pytest.mark.parametrize("distance", [0.01, 0.05, 0.2, 0.5])
    def test_it_is_two_phi_of_the_standardised_distance(self, distance: float) -> None:
        placement = Placement(distance=distance, horizon=YEAR, volatility=VOL)
        expected = 2.0 * normal_cdf(-placement.standardised_distance)
        assert evaluate(placement).fill_probability == pytest.approx(expected, rel=1e-14)

    def test_it_falls_with_the_distance_and_rises_with_the_horizon(self) -> None:
        by_distance = [
            evaluate(market(mult * market(1.0).deviation)).fill_probability
            for mult in (0.25, 0.5, 1.0, 2.0, 4.0)
        ]
        assert by_distance == sorted(by_distance, reverse=True)
        by_horizon = [
            evaluate(market(0.01, horizon=horizon)).fill_probability
            for horizon in (DAY / 4, DAY, 4 * DAY, 16 * DAY)
        ]
        assert by_horizon == sorted(by_horizon)

    def test_an_adverse_drift_lowers_it(self) -> None:
        deviation = market(1.0, horizon=DAY).deviation
        probabilities = [
            evaluate(market(deviation, drift=multiple * deviation / DAY)).fill_probability
            for multiple in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_it_is_a_probability_everywhere_tested(self) -> None:
        for distance in (1e-6, 0.001, 0.05, 0.5, 5.0):
            for drift in (-2.0, -0.1, 0.0, 0.1, 2.0):
                placement = Placement(distance=distance, horizon=YEAR, volatility=VOL, drift=drift)
                probability = evaluate(placement).fill_probability
                assert 0.0 <= probability <= 1.0


class TestAdverseSelection:
    """Conditional on filling, the mid ends up exactly at the limit price."""

    @pytest.mark.parametrize("distance", [0.01, 0.05, 0.2, 0.5])
    def test_the_conditional_mid_is_the_limit_price(self, distance: float) -> None:
        """Exactly, because ``2 b Phi(b/s)`` over ``2 Phi(b/s)`` is ``b``."""
        placement = Placement(distance=distance, horizon=YEAR, volatility=VOL)
        assert evaluate(placement).mid_if_filled == pytest.approx(-distance, rel=1e-12)

    @pytest.mark.parametrize("distance", [0.01, 0.05, 0.2])
    def test_so_a_fill_saves_nothing_against_the_terminal_mid(self, distance: float) -> None:
        placement = Placement(distance=distance, horizon=YEAR, volatility=VOL)
        outcome = evaluate(placement)
        # Paid: the limit price. Worth: the mid at the horizon. The two agree.
        assert outcome.cost_if_filled - outcome.mid_if_filled == pytest.approx(0.0, abs=1e-14)

    def test_a_miss_means_the_price_went_up(self) -> None:
        for distance in (0.01, 0.05, 0.2):
            outcome = evaluate(Placement(distance=distance, horizon=YEAR, volatility=VOL))
            assert outcome.mid_if_unfilled > 0.0
            # And the two conditional means average back to zero, because the
            # unconditional mid is a martingale.
            probability = outcome.fill_probability
            assert probability * outcome.mid_if_filled + (
                1.0 - probability
            ) * outcome.mid_if_unfilled == pytest.approx(0.0, abs=1e-14)

    def test_with_a_drift_they_average_to_the_drift(self) -> None:
        for drift in (-0.4, -0.05, 0.05, 0.4):
            placement = Placement(distance=0.05, horizon=YEAR, volatility=VOL, drift=drift)
            outcome = evaluate(placement)
            probability = outcome.fill_probability
            blended = (
                probability * outcome.mid_if_filled + (1.0 - probability) * outcome.mid_if_unfilled
            )
            assert blended == pytest.approx(drift * YEAR, abs=1e-13)

    def test_an_adverse_drift_lifts_the_conditional_mid_off_the_limit(self) -> None:
        """The cancellation is a martingale property and nothing more."""
        mids = []
        for drift in (0.0, 0.05, 0.15):
            placement = Placement(distance=0.05, horizon=YEAR, volatility=VOL, drift=drift)
            mids.append(evaluate(placement).mid_if_filled)
        assert mids[0] == pytest.approx(-0.05, rel=1e-12)
        assert mids == sorted(mids)
        assert mids[-1] > -0.05


class TestExpectedCost:
    """The distance enters only through the fill probability."""

    @pytest.mark.parametrize("multiple", [0.01, 0.25, 0.5, 1.0, 2.0, 4.0])
    def test_the_driftless_cost_is_the_spread_times_the_miss(self, multiple: float) -> None:
        placement = market(multiple * market(1.0).deviation)
        outcome = evaluate(placement)
        crossing = placement.half_spread + placement.taker_fee
        closed = crossing * (1.0 - outcome.fill_probability) - (
            placement.maker_rebate * outcome.fill_probability
        )
        assert outcome.expected_cost == pytest.approx(closed, rel=1e-13)

    def test_both_the_mean_and_the_deviation_rise_with_the_distance(self) -> None:
        """So there is no frontier: resting deeper is worse on both counts.

        Measured over a day at 20% volatility with a five basis point half
        spread: the mean rises from -0.94 to +6.00 basis points and the
        standard deviation from 16.4 to 126.0.
        """
        deviation = market(1.0).deviation
        outcomes = frontier(
            market(deviation),
            [multiple * deviation for multiple in (0.01, 0.25, 0.5, 1.0, 2.0, 4.0)],
        )
        means = [outcome.expected_cost for outcome in outcomes]
        spreads = [outcome.cost_deviation for outcome in outcomes]
        assert means == sorted(means)
        assert spreads == sorted(spreads)
        assert means[0] * 1e4 == pytest.approx(-0.9441, abs=0.001)
        assert means[-1] * 1e4 == pytest.approx(5.9996, abs=0.001)
        assert spreads[0] * 1e4 == pytest.approx(16.4195, abs=0.01)
        assert spreads[-1] * 1e4 == pytest.approx(125.9896, abs=0.01)

    def test_at_the_touch_the_order_earns_the_rebate_and_nothing_else(self) -> None:
        tiny = market(1e-9)
        outcome = evaluate(tiny)
        assert outcome.fill_probability == pytest.approx(1.0, abs=1e-6)
        assert outcome.expected_cost == pytest.approx(-tiny.maker_rebate, abs=1e-8)
        assert outcome.cost_deviation < 1e-5

    def test_driftless_resting_never_loses_and_only_drift_reverses_that(self) -> None:
        """Delay is free under a martingale, so any chance of a fill is upside.

        The first draft of this test asserted that resting far out *loses* to
        crossing immediately. It does not, and the reason is the whole point:
        the expected cost is ``(h+f)(1-p) - r p``, which is below ``h+f`` for
        any positive ``p``. Four standard deviations out the advantage is
        4.4e-08, vanishing but still positive. What makes resting lose is a
        drift, and then the loss converges to the drift over the horizon.
        """
        deviation = market(1.0).deviation
        multiples = (0.01, 0.5, 1.0, 2.0, 4.0)
        advantages = [evaluate(market(multiple * deviation)).advantage for multiple in multiples]
        assert all(advantage > 0.0 for advantage in advantages)
        assert advantages == sorted(advantages, reverse=True)
        assert advantages[-1] < 1e-7
        # Far out the strategy *is* crossing, just later.
        far = evaluate(market(4.0 * deviation))
        assert far.expected_cost == pytest.approx(far.crossing_cost, rel=1e-3)

        drift = deviation / DAY
        drifting = [
            evaluate(market(multiple * deviation, drift=drift)).advantage for multiple in multiples
        ]
        assert drifting[-1] < 0.0
        # And the loss converges to what the mid did while the order waited.
        assert drifting[-1] == pytest.approx(-drift * DAY, rel=0.01)

    def test_the_drift_cost_of_resting(self) -> None:
        """What being picked off costs, and how it scales.

        Out at one standard deviation the cost is linear in the drift — 117,
        127 and 130 basis points per standard deviation of it across three
        intervals — while at the touch it grows faster than linearly from a
        base of -0.94 basis points.
        """
        deviation = market(1.0).deviation
        out = []
        touch = []
        for multiple in (0.0, 0.5, 1.0, 2.0):
            drift = multiple * deviation / DAY
            out.append(evaluate(market(deviation, drift=drift)).expected_cost * 1e4)
            touch.append(evaluate(market(0.01 * deviation, drift=drift)).expected_cost * 1e4)
        assert out[1] == pytest.approx(62.2566, abs=0.01)
        assert out[2] == pytest.approx(125.6227, abs=0.01)
        assert touch[1] == pytest.approx(0.4458, abs=0.001)
        assert touch[2] == pytest.approx(2.6919, abs=0.001)
        # Forty-seven times as expensive to rest at one standard deviation as
        # at the touch, once the drift is a standard deviation per horizon.
        assert out[2] / touch[2] == pytest.approx(46.7, abs=0.2)
        # Linear out at one standard deviation: the per-unit-drift increments
        # agree to within ten per cent of each other.
        slopes = [
            (out[1] - out[0]) / 0.5,
            (out[2] - out[1]) / 0.5,
            (out[3] - out[2]) / 1.0,
        ]
        assert max(slopes) / min(slopes) < 1.15
        # And super-linear at the touch: each increment is larger than the last
        # per unit of drift.
        touch_slopes = [
            (touch[1] - touch[0]) / 0.5,
            (touch[2] - touch[1]) / 0.5,
            (touch[3] - touch[2]) / 1.0,
        ]
        assert touch_slopes == sorted(touch_slopes)


class TestMoments:
    def test_the_second_moment_bounds_the_first(self) -> None:
        """Cauchy-Schwarz on the indicator: ``E[X 1]**2 <= E[X**2 1] P``."""
        for distance in (0.01, 0.05, 0.3):
            for drift in (-0.2, 0.0, 0.2):
                joint = moments(
                    Placement(distance=distance, horizon=YEAR, volatility=VOL, drift=drift)
                )
                assert joint.second >= 0.0
                assert joint.first**2 <= joint.second * joint.probability + 1e-18

    def test_the_driftless_first_moment_is_twice_the_barrier_times_phi(self) -> None:
        for distance in (0.01, 0.05, 0.2, 0.5):
            placement = Placement(distance=distance, horizon=YEAR, volatility=VOL)
            joint = moments(placement)
            expected = -2.0 * distance * normal_cdf(-placement.standardised_distance)
            assert joint.first == pytest.approx(expected, rel=1e-13)

    def test_a_vanishing_distance_recovers_the_unconditional_moments(self) -> None:
        placement = Placement(distance=1e-12, horizon=YEAR, volatility=VOL, drift=0.05)
        joint = moments(placement)
        assert joint.probability == pytest.approx(1.0, abs=1e-9)
        assert joint.first == pytest.approx(0.05 * YEAR, abs=1e-9)
        assert joint.second == pytest.approx((0.05 * YEAR) ** 2 + VOL * VOL * YEAR, abs=1e-9)


class TestSimulation:
    """The independent route, and the bias that comes with it."""

    def test_the_corrected_formula_matches_the_simulation(self) -> None:
        placement = Placement(distance=0.05, horizon=YEAR, volatility=VOL)
        drawn = simulate_placement(
            placement, paths=60_000, steps=2_000, rng=np.random.default_rng(17)
        )
        shift = monitoring_shift(VOL, YEAR, 2_000)
        corrected = evaluate(placement.at(0.05 + shift)).fill_probability
        # Within three standard errors of the simulated frequency, which the
        # uncorrected formula is not: it sits about eleven away.
        assert abs(corrected - drawn.fill_probability) < 3.0 * drawn.fill_error
        uncorrected = evaluate(placement).fill_probability
        assert abs(uncorrected - drawn.fill_probability) > 5.0 * drawn.fill_error

    def test_the_bias_halves_as_the_step_count_quadruples(self) -> None:
        """The square-root rate, measured rather than quoted."""
        placement = Placement(distance=0.05, horizon=YEAR, volatility=VOL)
        exact = evaluate(placement).fill_probability
        biases = []
        for steps in (250, 1_000, 4_000):
            drawn = simulate_placement(
                placement, paths=60_000, steps=steps, rng=np.random.default_rng(3)
            )
            biases.append(exact / drawn.fill_probability - 1.0)
        assert all(bias > 0.0 for bias in biases)
        assert biases == sorted(biases, reverse=True)
        for earlier, later in pairwise(biases):
            assert 1.6 < earlier / later < 2.6

    def test_the_correction_does_not_fix_the_mean_cost(self) -> None:
        """Its bias is in the conditional mid, not in the probability.

        The gap between the shifted formula's mean and the simulated one is
        many standard errors at a modest step count and shrinks with the step
        count, which is the signature of a bias the shift does not touch.
        """
        placement = market(0.05, horizon=YEAR)
        gaps = []
        for steps in (250, 1_000):
            drawn = simulate_placement(
                placement, paths=60_000, steps=steps, rng=np.random.default_rng(3)
            )
            shift = monitoring_shift(VOL, YEAR, steps)
            shifted = evaluate(placement.at(0.05 + shift))
            gaps.append(abs(shifted.expected_cost - drawn.expected_cost) / drawn.cost_error)
        assert gaps[0] > 5.0
        assert gaps[1] < gaps[0]

    def test_the_cost_deviation_agrees_without_any_correction(self) -> None:
        """It is dominated by the unfilled paths, which the monitor sees exactly."""
        placement = market(0.05, horizon=YEAR)
        drawn = simulate_placement(
            placement, paths=60_000, steps=2_000, rng=np.random.default_rng(23)
        )
        exact = evaluate(placement).cost_deviation
        assert exact == pytest.approx(drawn.cost_deviation, rel=0.03)

    def test_a_drifting_mid_simulates_too(self) -> None:
        placement = market(0.05, horizon=YEAR, drift=0.08)
        drawn = simulate_placement(
            placement, paths=40_000, steps=4_000, rng=np.random.default_rng(5)
        )
        shift = monitoring_shift(VOL, YEAR, 4_000)
        corrected = evaluate(placement.at(0.05 + shift)).fill_probability
        assert abs(corrected - drawn.fill_probability) < 4.0 * drawn.fill_error
        assert drawn.paths == 40_000
        assert drawn.steps == 4_000

    def test_the_shift_constant_and_its_scaling(self) -> None:
        assert abs(MONITORING_BETA - 0.58259716) < 1e-8
        assert monitoring_shift(VOL, YEAR, 1) == pytest.approx(MONITORING_BETA * VOL)
        # Falls as the square root of the step count.
        first = monitoring_shift(VOL, YEAR, 100)
        second = monitoring_shift(VOL, YEAR, 400)
        assert first / second == pytest.approx(2.0, rel=1e-12)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [({"paths": 0}, "paths must be at least one"), ({"steps": 0}, "steps")],
    )
    def test_refuses_a_degenerate_draw(self, kwargs: dict[str, int], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            simulate_placement(market(0.05), **kwargs)  # type: ignore[arg-type]

    def test_one_path_has_no_sample_deviation(self) -> None:
        drawn = simulate_placement(market(0.05), paths=1, steps=10, rng=np.random.default_rng(1))
        assert drawn.cost_deviation == 0.0
        assert drawn.cost_error == 0.0


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("distance", 0.0, "distance must be a positive"),
            ("distance", -0.01, "distance must be a positive"),
            ("horizon", 0.0, "horizon must be a positive"),
            ("volatility", 0.0, "volatility must be a positive"),
            ("volatility", math.inf, "volatility must be a positive"),
            ("drift", math.nan, "drift must be a finite"),
            ("half_spread", -0.001, "half_spread must be a non-negative"),
            ("taker_fee", -0.001, "taker_fee must be a non-negative"),
            ("maker_rebate", -0.001, "maker_rebate must be a non-negative"),
        ],
    )
    def test_refuses_bad_inputs(self, field: str, value: float, message: str) -> None:
        arguments: dict[str, float] = {
            "distance": 0.01,
            "horizon": DAY,
            "volatility": VOL,
        }
        arguments[field] = value
        with pytest.raises(ValidationError, match=message):
            Placement(**arguments)

    def test_at_keeps_the_market_and_changes_the_distance(self) -> None:
        original = market(0.01, drift=0.03)
        moved = original.at(0.02)
        assert moved.distance == 0.02
        assert moved.horizon == original.horizon
        assert moved.volatility == original.volatility
        assert moved.drift == original.drift
        assert moved.half_spread == original.half_spread
        assert moved.taker_fee == original.taker_fee
        assert moved.maker_rebate == original.maker_rebate

    def test_an_empty_frontier_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least one distance"):
            frontier(market(0.01), [])

    def test_monitoring_shift_validates(self) -> None:
        with pytest.raises(ValidationError, match="volatility"):
            monitoring_shift(0.0, YEAR, 10)
        with pytest.raises(ValidationError, match="horizon"):
            monitoring_shift(VOL, 0.0, 10)
        with pytest.raises(ValidationError, match="steps"):
            monitoring_shift(VOL, YEAR, 0)

    def test_the_standardised_distance_is_the_only_thing_that_matters(self) -> None:
        """Two markets at the same standardised distance fill alike."""
        first = Placement(distance=0.02, horizon=YEAR, volatility=0.2)
        second = Placement(distance=0.01, horizon=YEAR / 4, volatility=0.2)
        assert first.standardised_distance == pytest.approx(second.standardised_distance)
        assert evaluate(first).fill_probability == pytest.approx(
            evaluate(second).fill_probability, rel=1e-14
        )
