"""Optimal execution against a price forecast.

The checks that carry the weight are exact rather than toleranced. With no
forecast the solver has to reproduce :func:`execution.optimal_trajectory`'s
closed form, which shares no code with it. The response kernel has to equal the
Green's function of the Almgren-Chriss operator, built from the same urgency
``kappa`` the forecast-blind problem already has. The value of a forecast has to
equal ``mu' G mu / 2``, which makes it quadratic in the forecast and therefore
the same for a forecast and its negative. And the first-order conditions have
to vanish, which a search could not promise.

What is measured rather than derived is the economics: how strong a reversing
forecast has to be before the optimum asks to trade backwards, what the
forecast is worth in basis points, and how much of that a tuned proportional
tilt recovers -- which turns out to be nearly all of it except on a sharp
signal.
"""

from __future__ import annotations

import math

import pytest

from slippage.alpha import (
    ForecastSchedule,
    forecast_objective,
    forecast_schedule,
    forecast_value,
    proportional_tilt,
    response_kernel,
    spike_response,
)
from slippage.exceptions import ValidationError
from slippage.execution import ExecutionProblem, optimal_trajectory
from slippage.impact import LinearImpact

RISK = 2e-6
PRICE = 50.0


def problem(periods: int = 20) -> ExecutionProblem:
    return ExecutionProblem(
        quantity=1_000_000.0,
        horizon=1.0,
        periods=periods,
        volatility=0.9,
        impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6),
    )


def decaying(periods: int = 20, size: float = 0.02) -> list[float]:
    """Two cents a period at the start, falling linearly to zero."""
    return [size * (1.0 - index / periods) for index in range(periods)]


# -- validation ---------------------------------------------------------------


def test_the_wrong_number_of_drifts_is_refused() -> None:
    with pytest.raises(ValidationError, match="one per period"):
        forecast_schedule(problem(), RISK, [0.01] * 19)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_drift_that_is_not_a_price_move_is_refused(bad: float) -> None:
    drifts = decaying()
    drifts[7] = bad
    with pytest.raises(ValidationError, match="not a price move"):
        forecast_schedule(problem(), RISK, drifts)


@pytest.mark.parametrize("risk", [-1e-6, math.nan])
def test_a_negative_risk_aversion_is_refused(risk: float) -> None:
    with pytest.raises(ValidationError, match="risk aversion"):
        forecast_schedule(problem(), risk, decaying())
    with pytest.raises(ValidationError, match="risk aversion"):
        response_kernel(problem(), risk)


def test_a_saving_in_basis_points_needs_a_price() -> None:
    result = forecast_schedule(problem(), RISK, decaying())
    with pytest.raises(ValidationError, match="price must be positive"):
        result.saving_bps(0.0)


# -- the reductions that are exact --------------------------------------------


@pytest.mark.parametrize("risk", [0.0, 1e-7, RISK, 1e-4])
def test_no_forecast_reproduces_the_closed_form(risk: float) -> None:
    """The one check that goes through nothing of this module's.

    ``optimal_trajectory`` builds the sinh profile directly from ``kappa``; the
    solver here knows nothing about sinh and inverts a tridiagonal matrix.
    """
    task = problem()
    solved = forecast_schedule(task, risk, [0.0] * task.periods)
    closed = optimal_trajectory(task, risk)
    for mine, theirs in zip(solved.trajectory.holdings, closed.holdings, strict=True):
        assert mine == pytest.approx(theirs, rel=1e-10, abs=1e-6)
    assert solved.objective == pytest.approx(closed.objective(risk), rel=1e-12)
    assert solved.saving == pytest.approx(0.0, abs=1e-6)
    assert solved.forecast_cost == 0.0


def test_the_final_periods_drift_changes_nothing_at_all() -> None:
    """Exactly nothing: by then there is nothing left to trade."""
    task = problem()
    flat = forecast_schedule(task, RISK, [0.0] * task.periods)
    drifts = [0.0] * task.periods
    drifts[-1] = 10.0
    loud = forecast_schedule(task, RISK, drifts)
    assert loud.trajectory.holdings == flat.trajectory.holdings
    assert loud.objective == flat.objective


def test_the_first_order_conditions_vanish() -> None:
    """Which a search could not promise, and is why this is a solve."""
    task = problem()
    drifts = decaying()
    result = forecast_schedule(task, RISK, drifts)
    held = list(result.trajectory.holdings)
    impact = 2.0 * task.eta_tilde / task.tau
    risk = 2.0 * RISK * task.volatility**2 * task.tau
    scale = impact * task.quantity
    for index in range(1, task.periods):
        gradient = (
            impact * (2.0 * held[index] - held[index - 1] - held[index + 1])
            + risk * held[index]
            + drifts[index - 1]
        )
        assert abs(gradient) < 1e-9 * scale


def test_a_single_period_order_cannot_be_rescheduled() -> None:
    task = problem(periods=1)
    result = forecast_schedule(task, RISK, [0.05])
    assert result.trajectory.trades == (task.quantity,)
    assert result.saving == pytest.approx(0.0, abs=1e-9)


# -- the response kernel ------------------------------------------------------


@pytest.mark.parametrize("risk", [0.0, 1e-7, RISK, 1e-4])
@pytest.mark.parametrize("period", [1, 5, 10, 17])
def test_the_spike_response_is_the_green_function(risk: float, period: int) -> None:
    """Closed form against solver, with nothing shared but the problem.

    The kernel is built from ``ExecutionProblem.kappa``, so the smoothing a
    forecast receives is the urgency the forecast-blind schedule already had
    rather than a parameter of its own.
    """
    task = problem()
    measured = spike_response(task, risk, period, 1.0)
    kernel = response_kernel(task, risk)
    scale = max(abs(value) for value in measured)
    for index, value in enumerate(measured):
        assert value == pytest.approx(-kernel[index][period], abs=1e-13 * scale + 1e-9)


def test_the_kernel_is_triangular_at_a_zero_risk_aversion() -> None:
    """The Green's function of a discrete Laplacian, and not an exponential.

    A tent, which is the shape a spike response actually has -- the intuition
    that a signal decays geometrically away from its own period is wrong even
    in sign of curvature.
    """
    task = problem()
    kernel = response_kernel(task, 0.0)
    scale = 2.0 * task.eta_tilde / task.tau
    for row in range(1, task.periods):
        for column in range(1, task.periods):
            low, high = min(row, column), max(row, column)
            assert kernel[row][column] == pytest.approx(
                low * (task.periods - high) / (scale * task.periods), rel=1e-13
            )
    middle = spike_response(task, 0.0, 10, 1.0)
    steps = [middle[index + 1] - middle[index] for index in range(10)]
    assert max(steps) - min(steps) < 1e-9 * abs(steps[0])


def test_the_kernel_is_symmetric_and_vanishes_on_the_boundary() -> None:
    task = problem()
    kernel = response_kernel(task, RISK)
    assert all(value == 0.0 for value in kernel[0])
    assert all(value == 0.0 for value in kernel[task.periods])
    assert all(row[0] == 0.0 and row[task.periods] == 0.0 for row in kernel)
    for row in range(task.periods + 1):
        for column in range(task.periods + 1):
            assert kernel[row][column] == pytest.approx(kernel[column][row], rel=1e-14)


def test_a_spike_outside_the_interior_is_refused() -> None:
    task = problem()
    with pytest.raises(ValidationError, match="nothing left to trade"):
        spike_response(task, RISK, task.periods, 1.0)
    with pytest.raises(ValidationError, match="periods 1 to"):
        spike_response(task, RISK, 0, 1.0)


def test_the_response_does_not_depend_on_the_order_size() -> None:
    """The system is linear, so the adjustment is the same at any size.

    True for linear impact, which is what this inherits, and worth asserting
    because the opposite -- a bigger order deserving a bigger tilt -- is the
    natural assumption.
    """
    small = ExecutionProblem(
        quantity=10_000.0,
        horizon=1.0,
        periods=20,
        volatility=0.9,
        impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6),
    )
    large = problem()
    drifts = decaying()
    for task in (small, large):
        baseline = forecast_schedule(task, RISK, [0.0] * task.periods)
        tilted = forecast_schedule(task, RISK, drifts)
        shift = [
            after - before
            for after, before in zip(
                tilted.trajectory.holdings, baseline.trajectory.holdings, strict=True
            )
        ]
        if task is small:
            reference = shift
        else:
            for mine, theirs in zip(shift, reference, strict=True):
                assert mine == pytest.approx(theirs, rel=1e-8, abs=1e-6)


# -- the value of a forecast --------------------------------------------------


@pytest.mark.parametrize("risk", [0.0, RISK, 1e-4])
def test_the_closed_form_value_matches_the_solved_saving(risk: float) -> None:
    task = problem()
    drifts = decaying()
    assert forecast_value(task, risk, drifts) == pytest.approx(
        forecast_schedule(task, risk, drifts).saving, rel=1e-9
    )


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 3.0])
def test_the_value_is_quadratic_in_the_forecast(scale: float) -> None:
    task = problem()
    drifts = decaying()
    base = forecast_value(task, RISK, drifts)
    scaled = forecast_value(task, RISK, [scale * value for value in drifts])
    assert scaled == pytest.approx(scale * scale * base, rel=1e-11)
    solved = forecast_schedule(task, RISK, [scale * value for value in drifts]).saving
    assert solved == pytest.approx(scale * scale * base, rel=1e-9)


def test_a_forecast_and_its_negative_are_worth_exactly_the_same() -> None:
    """Which is not what anybody expects of a trading signal.

    "Hurry" and "wait" produce schedules that move in opposite directions and
    improvements that are identical, because the value is a quadratic form.
    """
    task = problem()
    drifts = decaying()
    forwards = forecast_schedule(task, RISK, drifts)
    backwards = forecast_schedule(task, RISK, [-value for value in drifts])
    assert backwards.saving == pytest.approx(forwards.saving, rel=1e-11)
    assert forwards.forecast_cost * backwards.forecast_cost < 0.0
    first = forwards.trajectory.holdings
    second = backwards.trajectory.holdings
    flat = forecast_schedule(task, RISK, [0.0] * task.periods).trajectory.holdings
    for index in range(1, task.periods):
        assert (first[index] - flat[index]) * (second[index] - flat[index]) < 0.0


def test_the_value_is_non_negative_for_any_forecast() -> None:
    """The kernel is positive definite, so a forecast never hurts.

    Swept over profiles rather than asserted from the form, because the sign
    of the quadratic form is the thing that would break if the operator were
    assembled wrongly.
    """
    task = problem()
    profiles = [
        decaying(),
        [-value for value in decaying()],
        [0.02 * (-1) ** index for index in range(task.periods)],
        [0.02 if index == 10 else 0.0 for index in range(task.periods)],
        [0.0] * task.periods,
        [0.01 * math.sin(index) for index in range(task.periods)],
    ]
    for drifts in profiles:
        assert forecast_value(task, RISK, drifts) >= -1e-12
        assert forecast_schedule(task, RISK, drifts).saving >= -1e-6


def test_the_objective_of_the_optimum_beats_every_schedule_tried() -> None:
    """An optimum is the cheapest thing, which is checkable without theory."""
    task = problem()
    drifts = decaying()
    best = forecast_schedule(task, RISK, drifts)
    rivals = [
        list(optimal_trajectory(task, RISK).trades),
        [task.quantity / task.periods] * task.periods,
        list(optimal_trajectory(task, 1e-5).trades),
        list(optimal_trajectory(task, 1e-8).trades),
        proportional_tilt(task, RISK, drifts, 0.04),
        proportional_tilt(task, RISK, drifts, 0.5),
    ]
    for trades in rivals:
        assert forecast_objective(task, RISK, trades, drifts) >= best.objective - 1e-6


def test_forecast_objective_rejects_a_schedule_that_is_not_the_order() -> None:
    task = problem()
    with pytest.raises(ValidationError):
        forecast_objective(task, RISK, [1.0] * task.periods, decaying())


# -- the economics, measured --------------------------------------------------


def test_the_optimum_is_reluctant_to_round_trip() -> None:
    """Impact is quadratic and the forecast is linear, so buying back is dear.

    Measured on a forecast that reverses halfway: the first negative trade
    appears at 97.7 cents a share a period, which is 4.86 times a period's
    volatility. The guess before measuring was that a round trip would appear
    at a fraction of one.
    """
    task = problem()
    half = task.periods // 2

    def reversing(size: float) -> list[float]:
        return [size if index < half else -size for index in range(task.periods)]

    low, high = 0.0, 5.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if forecast_schedule(task, RISK, reversing(middle)).round_trips:
            high = middle
        else:
            low = middle
    period_volatility = task.volatility * math.sqrt(task.tau)
    assert high == pytest.approx(0.977, rel=0.02)
    assert high / period_volatility == pytest.approx(4.86, rel=0.02)
    assert not forecast_schedule(task, RISK, reversing(high * 0.99)).round_trips
    beyond = forecast_schedule(task, RISK, reversing(high * 1.1))
    assert beyond.round_trips
    assert beyond.worst_trade < 0.0
    assert min(beyond.trajectory.trades) == beyond.worst_trade


def test_a_round_tripping_schedule_reports_rather_than_prices_itself() -> None:
    """LinearImpact.temporary refuses a negative rate, and it is right to.

    A negative rate is not a smaller cost, it is a different trade. So the
    impact cost and variance come back as not-a-number and the objective is
    taken from the quadratic form, which is defined there.
    """
    task = problem()
    half = task.periods // 2
    drifts = [3.0 if index < half else -3.0 for index in range(task.periods)]
    result = forecast_schedule(task, RISK, drifts)
    assert result.round_trips
    assert math.isnan(result.trajectory.expected_cost)
    assert math.isnan(result.trajectory.variance)
    assert math.isfinite(result.objective)
    assert result.objective < result.blind_objective
    assert result.objective == pytest.approx(
        result.blind_objective - forecast_value(task, RISK, drifts), rel=1e-9
    )


def test_what_the_forecast_is_worth_and_what_a_tilt_recovers() -> None:
    """Measured, and the answer argues against the solver on a smooth signal.

    On a decaying forecast a tuned proportional tilt recovers 99.3% of a
    saving that is only eight hundredths of a basis point to begin with.
    """
    task = problem()
    drifts = decaying()
    best = forecast_schedule(task, RISK, drifts)
    assert best.saving_bps(PRICE) == pytest.approx(0.0801, rel=0.03)
    assert best.saving > 0.0
    recovered = _best_tilt(task, drifts, best)
    assert recovered == pytest.approx(0.9929, rel=0.01)


def test_the_tilt_fails_where_the_smoothing_is_the_whole_point() -> None:
    """A single-period spike, which is exactly what a tilt cannot spread.

    63% recovered against 99% on a smooth forecast. So the case for solving
    this exactly is sharp isolated signals, and it is weak otherwise -- worth
    knowing before building the solver into anything.
    """
    task = problem()
    spike = [0.02 if index == 10 else 0.0 for index in range(task.periods)]
    best = forecast_schedule(task, RISK, spike)
    assert _best_tilt(task, spike, best) == pytest.approx(0.633, rel=0.03)

    alternating = [0.02 * (-1) ** index for index in range(task.periods)]
    smooth = forecast_schedule(task, RISK, alternating)
    assert _best_tilt(task, alternating, smooth) > 0.98


def _best_tilt(task: ExecutionProblem, drifts: list[float], best: ForecastSchedule) -> float:
    """Share of the optimum's saving a tuned proportional tilt recovers."""
    lowest = math.inf
    for step in range(-400, 401):
        trades = proportional_tilt(task, RISK, drifts, step / 200.0)
        lowest = min(lowest, forecast_objective(task, RISK, trades, drifts))
    return (best.blind_objective - lowest) / best.saving


def test_a_tilt_of_zero_is_the_blind_schedule_and_a_flat_forecast_changes_nothing() -> None:
    task = problem()
    blind = list(optimal_trajectory(task, RISK).trades)
    assert proportional_tilt(task, RISK, decaying(), 0.0) == pytest.approx(blind)
    assert proportional_tilt(task, RISK, [0.0] * task.periods, 0.5) == pytest.approx(blind)


def test_a_tilt_still_executes_the_whole_order() -> None:
    task = problem()
    for strength in (-2.0, -0.3, 0.3, 1.0, 2.0, 5.0):
        trades = proportional_tilt(task, RISK, decaying(), strength)
        assert math.fsum(trades) == pytest.approx(task.quantity, rel=1e-12)
        assert all(trade >= 0.0 for trade in trades)


def test_the_forecast_cost_is_the_drift_paid_on_what_is_still_held() -> None:
    """Which is the reading the telescoping gives, checked against the sum."""
    task = problem()
    drifts = decaying()
    result = forecast_schedule(task, RISK, drifts)
    held = result.trajectory.holdings
    direct = math.fsum(held[index + 1] * drifts[index] for index in range(task.periods - 1))
    assert result.forecast_cost == pytest.approx(direct, rel=1e-12)
    # And against the untelescoped form: each trade pays the drift accumulated
    # before it.
    cumulative = 0.0
    other = 0.0
    for index, trade in enumerate(result.trajectory.trades):
        other += trade * cumulative
        cumulative += drifts[index]
    assert other == pytest.approx(direct, rel=1e-9)


def test_a_fixed_per_share_cost_is_schedule_independent_until_it_is_not() -> None:
    """Which is what makes the quadratic solve the right problem, usually.

    ``epsilon X`` is the same for every schedule that trades one way, so it
    drops out of the optimisation. A round trip breaks that: the shares traded
    backwards and then forwards again each pay it, so the objective understates
    the real cost and the optimum is no longer the real optimum. Reported
    rather than silently wrong.
    """
    charged = ExecutionProblem(
        quantity=1_000_000.0,
        horizon=1.0,
        periods=20,
        volatility=0.9,
        impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.01),
    )
    free = problem()
    drifts = decaying()

    # No round trip: epsilon shifts both objectives by the same constant, so
    # the saving is identical and the schedule is untouched.
    with_fee = forecast_schedule(charged, RISK, drifts)
    without = forecast_schedule(free, RISK, drifts)
    assert with_fee.understated_fixed_cost == 0.0
    assert with_fee.saving == pytest.approx(without.saving, rel=1e-9)
    assert with_fee.objective - without.objective == pytest.approx(
        0.01 * charged.quantity, rel=1e-9
    )
    for mine, theirs in zip(with_fee.trajectory.holdings, without.trajectory.holdings, strict=True):
        assert mine == pytest.approx(theirs, rel=1e-10, abs=1e-6)

    # A round trip: now it is not a constant, and the amount by which the
    # objective is short is reported.
    half = charged.periods // 2
    reversing = [3.0 if index < half else -3.0 for index in range(charged.periods)]
    tripped = forecast_schedule(charged, RISK, reversing)
    assert tripped.round_trips
    assert tripped.understated_fixed_cost > 0.0
    backwards = -math.fsum(trade for trade in tripped.trajectory.trades if trade < 0.0)
    assert tripped.understated_fixed_cost == pytest.approx(2.0 * 0.01 * backwards)
    assert forecast_schedule(free, RISK, reversing).understated_fixed_cost == 0.0
