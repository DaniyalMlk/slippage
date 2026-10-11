"""What a schedule built on the wrong parameters costs."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from slippage import (
    ExecutionProblem,
    LinearImpact,
    Misspecification,
    continuous_regret,
    misspecification,
    optimal_trajectory,
    robust_urgency,
    schedule_dependent_objective,
    trajectory_from_ratio,
    urgency_number,
    urgency_ratio,
)
from slippage.exceptions import ValidationError

LAMBDA = 2e-6


def problem(
    *,
    volatility: float = 0.3,
    eta: float = 2e-6,
    gamma: float = 2.5e-7,
    periods: int = 50,
    epsilon: float = 0.0,
    horizon: float = 5.0,
) -> ExecutionProblem:
    return ExecutionProblem(
        quantity=1e6,
        horizon=horizon,
        periods=periods,
        volatility=volatility,
        impact=LinearImpact(gamma=gamma, eta=eta, epsilon=epsilon),
    )


def believing(error: float, **kwargs: float) -> ExecutionProblem:
    """A trader who believes the urgency ratio is ``1 + error`` times the truth.

    Moved through the volatility, which is the same as moving it through the
    impact coefficient — the invariance below is what says so.
    """
    base = problem(**kwargs)  # type: ignore[arg-type]
    return problem(volatility=base.volatility * math.sqrt(1.0 + error), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the invariance that makes this one-dimensional


def test_the_schedule_sees_only_lambda_sigma_squared_over_eta_tilde() -> None:
    """Scale the variance rate and the net impact together and nothing moves."""
    base = problem()
    factor = 3.0
    scaled = ExecutionProblem(
        quantity=base.quantity,
        horizon=base.horizon,
        periods=base.periods,
        volatility=base.volatility * math.sqrt(factor),
        impact=LinearImpact(gamma=base.impact.gamma * factor, eta=base.impact.eta * factor),
    )
    assert scaled.eta_tilde == pytest.approx(factor * base.eta_tilde, rel=1e-15)
    assert urgency_ratio(scaled, LAMBDA) == pytest.approx(urgency_ratio(base, LAMBDA), rel=1e-15)
    first = optimal_trajectory(base, LAMBDA).trades
    second = optimal_trajectory(scaled, LAMBDA).trades
    worst = max(abs(a - b) for a, b in zip(first, second, strict=True))
    assert worst / base.quantity < 1e-13


def test_the_fixed_per_share_cost_does_not_enter_at_all() -> None:
    """``epsilon`` is paid by every schedule, so it cannot change the regret."""
    bare = misspecification(believing(0.3), problem(), LAMBDA)
    with_spread = misspecification(believing(0.3, epsilon=0.02), problem(epsilon=0.02), LAMBDA)
    assert with_spread.regret_fraction == pytest.approx(bare.regret_fraction, rel=1e-13)
    assert with_spread.forecast_fraction == pytest.approx(bare.forecast_fraction, rel=1e-13)
    # And the subtraction is doing work: leaving the fixed term in would change it.
    trades = optimal_trajectory(problem(epsilon=0.02), LAMBDA).trades
    kept = schedule_dependent_objective(problem(epsilon=0.02), trades, LAMBDA)
    assert kept == pytest.approx(schedule_dependent_objective(problem(), trades, LAMBDA), rel=1e-13)


def test_an_error_in_impact_and_an_error_in_volatility_are_the_same_error() -> None:
    """The practical reading of the invariance."""
    truth = problem()
    through_volatility = misspecification(believing(0.25), truth, LAMBDA)
    # eta_tilde believed 1 / 1.25 times the truth gives the same ratio error.
    target = truth.eta_tilde / 1.25
    eta = target + 0.5 * truth.impact.gamma * truth.tau
    through_impact = misspecification(problem(eta=eta), truth, LAMBDA)
    assert through_impact.ratio_error == pytest.approx(through_volatility.ratio_error, rel=1e-12)
    assert through_impact.regret_fraction == pytest.approx(
        through_volatility.regret_fraction, rel=1e-11
    )


# --------------------------------------------------------------------------- #
# the two orders


def test_the_regret_is_never_negative_and_is_zero_only_when_right() -> None:
    truth = problem()
    exact = misspecification(problem(), truth, LAMBDA)
    assert exact.regret == 0.0
    assert exact.forecast_fraction == pytest.approx(0.0, abs=1e-15)
    assert exact.forecast_to_regret == math.inf
    for error in (-0.9, -0.5, -0.1, 0.1, 0.5, 2.0, 10.0):
        for periods in (5, 50, 500):
            got = misspecification(
                believing(error, periods=periods), problem(periods=periods), LAMBDA
            )
            assert got.regret >= 0.0
            assert got.regret_fraction >= 0.0


def test_the_schedule_is_wrong_to_second_order_and_the_forecast_to_first() -> None:
    """Both exponents measured by a log-log slope, not assumed."""
    truth = problem()
    errors = [0.08, 0.04, 0.02, 0.01, 0.005]
    regrets, forecasts = [], []
    for error in errors:
        got = misspecification(believing(error), truth, LAMBDA)
        regrets.append(got.regret_fraction)
        forecasts.append(abs(got.forecast_fraction))

    def slope(values: list[float]) -> float:
        return (math.log(values[0]) - math.log(values[-1])) / (
            math.log(errors[0]) - math.log(errors[-1])
        )

    assert slope(regrets) == pytest.approx(2.0, abs=0.02)
    assert slope(forecasts) == pytest.approx(1.0, abs=0.02)


def test_the_gap_between_the_two_orders_widens_as_the_error_shrinks() -> None:
    """A 10% error makes the forecast 3.4% wrong and the trades 0.04% wrong."""
    truth = problem()
    ratios = [
        misspecification(believing(error), truth, LAMBDA).forecast_to_regret
        for error in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    ]
    assert all(later > earlier for earlier, later in pairwise(ratios))
    tenth = misspecification(believing(0.1), truth, LAMBDA)
    assert tenth.forecast_fraction == pytest.approx(0.0335, abs=5e-4)
    assert tenth.regret_fraction == pytest.approx(3.78e-4, rel=0.02)
    assert tenth.forecast_to_regret == pytest.approx(88.0, abs=2.0)


# --------------------------------------------------------------------------- #
# the continuous closed form


def test_the_closed_form_is_minimised_at_the_truth_where_it_is_y_coth_y() -> None:
    for horizon in (0.5, 2.0, 5.0):
        for ratio in (0.01, 0.1, 1.0, 9.0):
            assert continuous_regret(ratio, ratio, horizon) < 1e-14
            y = math.sqrt(ratio) * horizon
            # Perturbing either way is worse, and the minimum's value is the
            # Almgren-Chriss objective y coth y, which the regret is relative to.
            assert continuous_regret(ratio * 1.01, ratio, horizon) > 0.0
            assert continuous_regret(ratio / 1.01, ratio, horizon) > 0.0
            assert y / math.tanh(y) > 1.0


def test_the_discrete_regret_converges_to_the_closed_form_like_one_over_the_count() -> None:
    error = 0.3
    gaps = []
    for periods in (50, 100, 200, 400, 800):
        truth = problem(periods=periods)
        got = misspecification(believing(error, periods=periods), truth, LAMBDA)
        limit = continuous_regret(got.believed_ratio, got.true_ratio, truth.horizon)
        gaps.append(got.regret_fraction / limit - 1.0)
    assert all(gap > 0.0 for gap in gaps)
    halvings = [a / b for a, b in pairwise(gaps)]
    assert all(1.8 < step < 2.2 for step in halvings), halvings


def test_the_closed_form_survives_an_urgent_trader() -> None:
    """``sinh`` overflows above 350; the ratio is taken in the exponents there."""
    assert continuous_regret(1.0, 1.0, 1000.0) == 0.0
    big = continuous_regret(4.0, 1.0, 1000.0)
    assert math.isfinite(big)
    # The urgent limit, exactly: (sqrt(k) - 1)**2 / (2 sqrt(k)) for k = 4.
    assert big == pytest.approx((2.0 - 1.0) ** 2 / (2.0 * 2.0), rel=1e-12)


# --------------------------------------------------------------------------- #
# how much it can ever cost


def curvature(y: float) -> float:
    """``regret_fraction / error**2`` in the limit of a small error, at urgency ``y``."""
    ratio = (y / 5.0) ** 2  # horizon of 5
    # A symmetric difference, which removes the cubic term. The step has to grow
    # for a patient trader: the regret there is y**4 / 45 times it squared, so at
    # y = 0.05 and a step of 1e-4 the whole quantity falls below the round-off of
    # the ratio it is computed from and comes back as exactly zero.
    step = 1e-4 if y >= 0.5 else 1e-2
    above = continuous_regret(ratio * (1.0 + step), ratio, 5.0)
    below = continuous_regret(ratio * (1.0 - step), ratio, 5.0)
    return (above + below) / (2.0 * step**2)


def test_the_cost_of_being_wrong_has_a_ceiling_of_exactly_one_eighth() -> None:
    """Rising in urgency and saturating at 1/8, which is a bound not a value."""
    rising = [curvature(y) for y in (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 8.0)]
    assert all(later > earlier for earlier, later in pairwise(rising))
    assert all(value < 0.125 for value in rising)
    for y in (12.0, 30.0, 200.0, 2000.0):
        assert curvature(y) == pytest.approx(0.125, rel=1e-7)


def test_a_patient_trader_is_almost_immune() -> None:
    """The coefficient falls like ``y**4 / 45``, so a long horizon is its own hedge."""
    for y in (0.05, 0.1, 0.2):
        assert curvature(y) == pytest.approx(y**4 / 45.0, rel=0.03)
    # Four decades of coefficient between a patient trader and an urgent one.
    assert curvature(10.0) / curvature(0.1) > 1e4


def test_a_factor_of_two_error_costs_six_per_cent_at_worst() -> None:
    urgent = continuous_regret(2.0, 1.0, 1000.0)
    assert urgent == pytest.approx(0.0607, abs=1e-4)
    # And believing the ratio half what it is costs the same, exactly, in that
    # limit: the closed form is symmetric under k -> 1 / k.
    assert continuous_regret(0.5, 1.0, 1000.0) == pytest.approx(urgent, rel=1e-12)


def test_the_symmetry_in_the_ratio_is_broken_at_finite_urgency() -> None:
    """Overestimating costs more, because the curvature rises with urgency."""
    truth = problem()
    ratio = urgency_ratio(truth, LAMBDA)
    horizon = truth.horizon
    assert urgency_number(truth, LAMBDA) == pytest.approx(1.5047, abs=1e-3)
    for factor in (1.5, 2.0, 3.0):
        too_fast = continuous_regret(ratio * factor, ratio, horizon)
        too_slow = continuous_regret(ratio / factor, ratio, horizon)
        # Believing the ratio too large is worse: it lands the trader in the
        # steeper part of the curvature.
        assert too_fast > too_slow
    assert continuous_regret(ratio * 1.5, ratio, horizon) / continuous_regret(
        ratio / 1.5, ratio, horizon
    ) == pytest.approx(1.70, abs=0.03)
    # Read the other way: with the belief fixed, a faster truth costs more, and
    # that is the asymmetry the minimax responds to.
    assert continuous_regret(ratio, ratio * 1.5, horizon) / continuous_regret(
        ratio, ratio / 1.5, horizon
    ) == pytest.approx(1.47, abs=0.03)


# --------------------------------------------------------------------------- #
# the robust schedule


def test_the_robust_urgency_beats_both_midpoints_and_leans_fast() -> None:
    truth = problem()
    ratio = urgency_ratio(truth, LAMBDA)
    got = robust_urgency(truth, LAMBDA, 0.5 * ratio, 2.0 * ratio)
    assert got.ratio > ratio  # above the geometric midpoint, which is the truth here
    assert got.worst_regret < got.geometric_worst
    assert got.worst_regret < got.arithmetic_worst
    assert got.worst_regret == pytest.approx(0.0192, abs=5e-4)
    assert got.geometric_worst == pytest.approx(0.0256, abs=5e-4)
    assert got.gain_over_geometric == pytest.approx(1.34, abs=0.05)
    # The worst case falls at an end of the range, as a monotone-in-each-direction
    # regret requires.
    assert got.worst_at in (
        pytest.approx(0.5 * ratio, rel=1e-9),
        pytest.approx(2.0 * ratio, rel=1e-9),
    )


def test_being_robust_is_worth_more_the_wider_the_uncertainty() -> None:
    truth = problem()
    ratio = urgency_ratio(truth, LAMBDA)
    narrow = robust_urgency(truth, LAMBDA, 0.8 * ratio, 1.25 * ratio)
    wide = robust_urgency(truth, LAMBDA, 0.1 * ratio, 10.0 * ratio)
    assert narrow.worst_regret < wide.worst_regret
    assert narrow.gain_over_geometric < wide.gain_over_geometric
    assert wide.worst_regret == pytest.approx(0.233, abs=0.01)
    assert wide.geometric_worst == pytest.approx(0.474, abs=0.01)
    # Over a factor of a quarter either way the worst case is still under 2%,
    # which is the phase's point: this is a second-order problem.
    assert narrow.worst_regret < 0.02


def test_the_robust_schedule_can_be_built_and_scored() -> None:
    truth = problem()
    ratio = urgency_ratio(truth, LAMBDA)
    got = robust_urgency(truth, LAMBDA, 0.5 * ratio, 2.0 * ratio)
    built = trajectory_from_ratio(truth, LAMBDA, got.ratio)
    assert sum(built.trades) == pytest.approx(truth.quantity, rel=1e-12)
    assert built.holdings[0] == truth.quantity
    assert built.holdings[-1] == 0.0
    # At the true ratio it is the optimal trajectory itself.
    exact = trajectory_from_ratio(truth, LAMBDA, ratio)
    reference = optimal_trajectory(truth, LAMBDA)
    assert exact.trades == pytest.approx(reference.trades, rel=1e-12)
    assert exact.expected_cost == pytest.approx(reference.expected_cost, rel=1e-12)
    assert exact.variance == pytest.approx(reference.variance, rel=1e-12)
    # And it is worse than optimal under the truth, as a robust schedule is.
    assert built.objective(LAMBDA) > reference.objective(LAMBDA)


# --------------------------------------------------------------------------- #
# refusals


def test_bad_arguments_are_refused() -> None:
    truth = problem()
    with pytest.raises(ValidationError, match="risk-neutral"):
        urgency_ratio(truth, 0.0)
    with pytest.raises(ValidationError, match="which order"):
        misspecification(problem(periods=20), problem(periods=50), LAMBDA)
    with pytest.raises(ValidationError, match="which order"):
        misspecification(
            ExecutionProblem(
                quantity=2e6,
                horizon=5.0,
                periods=50,
                volatility=0.3,
                impact=LinearImpact(gamma=2.5e-7, eta=2e-6),
            ),
            truth,
            LAMBDA,
        )
    with pytest.raises(ValidationError):
        continuous_regret(1.0, 1.0, 0.0)
    with pytest.raises(ValidationError):
        continuous_regret(-1.0, 1.0, 1.0)
    with pytest.raises(ValidationError):
        robust_urgency(truth, LAMBDA, 0.0, 1.0)
    with pytest.raises(ValidationError):
        robust_urgency(truth, LAMBDA, 1.0, 0.5)
    with pytest.raises(ValidationError):
        robust_urgency(truth, LAMBDA, 0.1, 1.0, grid=1)
    with pytest.raises(ValidationError):
        trajectory_from_ratio(truth, LAMBDA, -1.0)


def test_the_result_is_a_frozen_record() -> None:
    got = misspecification(believing(0.2), problem(), LAMBDA)
    assert isinstance(got, Misspecification)
    with pytest.raises(AttributeError):
        got.regret = 0.0  # type: ignore[misc]
