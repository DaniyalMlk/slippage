"""Optimal execution against a price forecast.

:func:`~slippage.execution.optimal_trajectory` assumes the price is a
martingale. The only reason to trade early is then risk, and the schedule
depends on the order and the stock but never on a view. Every desk with a
short-horizon forecast trades against one, and this module is where it goes.

The objective stays quadratic, which is the whole reason this is a solve and
not a search. Write the schedule as the remaining holdings ``x_1 .. x_{N-1}``
with ``x_0 = X`` and ``x_N = 0``: the "everything must trade" constraint is then
satisfied by construction and there is nothing to enforce. Impact and risk are
the terms Almgren and Chriss already use. And the forecast contributes a term
that **telescopes**::

    sum_k n_k D_{k-1}  =  sum_{k=1}^{N-1} x_k mu_k

where ``mu_k`` is the drift during period ``k`` and ``D_k`` its cumulative sum.
The reading is exact rather than approximate: the shares traded in period
``k + 1`` or later are precisely the ``x_k`` still outstanding after period
``k``, and every one of them pays period ``k``'s drift. So the forecast enters
*diagonally*, the Hessian stays tridiagonal, and the first-order conditions are
a linear system solved by one pass of the Thomas algorithm.

Two consequences fall out of the algebra before any measurement.

**The forecast adjustment does not depend on the order size.** The system is
linear, so the solution is the no-forecast trajectory plus a response that is
linear in the forecast alone. Doubling the order doubles the baseline schedule
and leaves the adjustment where it was -- for linear impact, which is the model
this inherits.

**The drift in the final period cannot move the schedule.** By then there is
nothing left to trade, so ``mu_N`` never appears. :func:`forecast_schedule`
accepts it and ignores it, and the tests check that rather than trusting the
index arithmetic.

Three things are exact rather than measured, and they are what this module is
worth.

**The response kernel is the Green's function of the Almgren-Chriss
operator.** The first-order conditions are ``a (2I - S) x + c x = -mu``, whose
homogeneous solutions are the ``sinh`` profiles that problem already solves
for, so the inverse is known in closed form -- see :func:`response_kernel`,
built from the *same* urgency ``kappa`` as the forecast-blind schedule. The
solver agrees with it to between 3.4e-15 and 2.4e-14 relative, at risk
aversions from zero up to ``kappa T = 5.7``. So the smoothing a forecast
receives is not a new parameter.

It is also not an exponential decay, which is the shape one expects. At a zero
risk aversion the ``sinh`` degenerate to their arguments and the kernel is
exactly **triangular** -- a tent, the Green's function of a discrete
Laplacian -- and at a realistic risk aversion it is a mild deformation of one:
a spike in the middle of twenty periods moves the holdings by 46, 93, 139, 186
... 476 shares on the way in and symmetrically out, where a geometric decay
would have been 476, 306, 196, 126.

**The value of a forecast has a closed form too**, ``mu' G mu / 2``, with no
solve in it: :func:`forecast_value`, agreeing with the solved objectives to
1.4e-12. Two consequences follow from the form rather than from any
measurement. The value is **quadratic** in the forecast -- doubling it
quadruples the value, confirmed to twelve figures. And the value is therefore
**identical for a forecast and its negative**: a signal saying "hurry" and one
saying "wait" are worth exactly the same, to twelve figures, even though the
schedules they produce move in opposite directions. That is not what anybody
expects of a trading signal.

**The drift in the final period cannot move the schedule**, because by then
there is nothing left to trade, and ``mu_N`` genuinely never appears.
:func:`forecast_schedule` accepts it and ignores it, exactly.

Measured on a million shares over one day in twenty periods, volatility 0.9
dollars per share per root day, ``eta = 2.5e-6``, ``gamma = 2.5e-7``, risk
aversion 2e-6 -- where one period's volatility is 20.1 cents a share:

* **The optimum is far more reluctant to round-trip than one would guess.**
  With a forecast that reverses halfway -- against the trade in the first half,
  with it in the second -- the unconstrained optimum first asks for a negative
  trade at a drift of **97.7 cents per share per period**, which is 4.86 times
  a period's volatility. Impact is quadratic and the forecast is linear, so
  buying shares back has to overcome a cost that rises faster than the reason
  to. When it does happen :class:`ForecastSchedule` reports it, because
  :meth:`~slippage.impact.LinearImpact.temporary` refuses a negative rate
  outright and a schedule nothing can price is worse than a flag.
* **The forecast is worth very little, and a tuned heuristic captures almost
  all of it.** Against a forecast of 2 cents per period decaying linearly to
  zero, the optimum beats the forecast-blind schedule by 0.0801 basis points of
  the order's value at fifty dollars a share. Trading in proportion to the
  signal, with its strength tuned, recovers **99.29%** of that.
* **Except where the smoothing is the whole point.** On a single-period spike
  the same tuned heuristic recovers only **63.3%**, because the optimum spreads
  the response over the horizon and a proportional tilt cannot. On an
  alternating forecast it recovers 99.17% and on the reversing one 90.6%. So
  the case for solving the problem exactly is sharp, isolated signals, and on a
  smooth forecast it is weak -- which is worth knowing before building the
  solver into anything.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .exceptions import ValidationError
from .execution import ExecutionProblem, Trajectory, optimal_trajectory, trajectory_from_trades

__all__ = [
    "ForecastSchedule",
    "forecast_objective",
    "forecast_schedule",
    "forecast_value",
    "proportional_tilt",
    "response_kernel",
    "spike_response",
]


def _check_forecast(problem: ExecutionProblem, drifts: Sequence[float]) -> list[float]:
    """Validate a per-period drift sequence and return it as a list.

    One value per period, in price units per share per period -- not per unit
    time. A forecast is a view about the next interval, which is how a desk
    states one, and converting it to a rate here would mean undoing that.
    """
    if len(drifts) != problem.periods:
        raise ValidationError(
            f"expected {problem.periods} drifts, one per period, got {len(drifts)}"
        )
    for index, value in enumerate(drifts):
        if not math.isfinite(value):
            raise ValidationError(f"drift {index} is {value!r}, which is not a price move")
    return list(drifts)


@dataclass(frozen=True)
class ForecastSchedule:
    """An optimal schedule under a forecast, and what it cost to get there.

    Attributes:
        trajectory: The schedule, its expected impact cost and its variance.
            The impact cost and variance are the forecast-blind quantities, as
            :func:`~slippage.execution.schedule_moments` computes them; the
            drift's contribution is :attr:`forecast_cost`.
        forecast_cost: ``sum_k x_k mu_k``, the drift paid on shares still
            outstanding. Negative when the forecast is in the trade's favour.
        objective: Impact plus risk plus drift -- the quantity minimised.
        blind_objective: The same quantity for the forecast-blind schedule,
            evaluated under the same drift. The difference is what the forecast
            was worth.
        round_trips: True when some period's trade came out negative, which is
            the optimum asking to trade against the order's own direction. A
            desk usually cannot, and the impact model refuses a negative rate,
            so this is reported rather than clipped away.
        worst_trade: The most negative trade, or zero when there is none.
    """

    trajectory: Trajectory
    forecast_cost: float
    objective: float
    blind_objective: float
    round_trips: bool
    worst_trade: float

    @property
    def saving(self) -> float:
        """How much the forecast was worth, in the objective's own units."""
        return self.blind_objective - self.objective

    def saving_bps(self, price: float) -> float:
        """The saving in basis points of the order's value at ``price``."""
        if not math.isfinite(price) or price <= 0.0:
            raise ValidationError(f"price must be positive, got {price!r}")
        return 1e4 * self.saving / (price * self.trajectory.holdings[0])


def forecast_objective(
    problem: ExecutionProblem,
    risk_aversion: float,
    trades: Sequence[float],
    drifts: Sequence[float],
) -> float:
    """``E + lambda V + sum_k x_k mu_k`` for an arbitrary schedule.

    Here so that the optimum can be checked against other schedules under the
    same forecast, including the forecast-blind one -- which is the comparison
    that says what the forecast was worth, rather than comparing two objectives
    that do not mean the same thing.
    """
    drift = _check_forecast(problem, drifts)
    trajectory = trajectory_from_trades(problem, list(trades))
    held = trajectory.holdings
    carried = math.fsum(held[index + 1] * drift[index] for index in range(problem.periods - 1))
    return trajectory.objective(risk_aversion) + carried


def _solve_tridiagonal(diagonal: list[float], off: float, right: list[float]) -> list[float]:
    """Thomas algorithm for a symmetric tridiagonal system with constant off-diagonal.

    The matrix is ``2 (eta_tilde / tau) * (2 I - S) + 2 lambda sigma^2 tau I``
    for the shift ``S``, so it is symmetric, positive definite and weakly
    diagonally dominant with equality only at a zero risk aversion -- where it
    is still non-singular, because the end conditions pin both ends of the
    schedule. One forward sweep and one back substitution; no pivoting, which
    the dominance is what licenses.
    """
    size = len(diagonal)
    if size == 0:
        return []
    pivots = [0.0] * size
    values = [0.0] * size
    pivots[0] = diagonal[0]
    values[0] = right[0]
    for index in range(1, size):
        factor = off / pivots[index - 1]
        pivots[index] = diagonal[index] - factor * off
        values[index] = right[index] - factor * values[index - 1]
    solution = [0.0] * size
    solution[-1] = values[-1] / pivots[-1]
    for index in range(size - 2, -1, -1):
        solution[index] = (values[index] - off * solution[index + 1]) / pivots[index]
    return solution


def forecast_schedule(
    problem: ExecutionProblem,
    risk_aversion: float,
    drifts: Sequence[float],
) -> ForecastSchedule:
    """The exactly optimal schedule under a per-period price forecast.

    Args:
        problem: The order, the horizon and the linear impact model.
        risk_aversion: ``lambda``, in inverse currency, as elsewhere.
        drifts: Expected price change in each period, in price per share. One
            per period; the last is accepted and ignored, because by the final
            period there is nothing left to trade.

    Raises:
        ValidationError: for a wrong number of drifts, a non-finite drift, or a
            negative risk aversion.
    """
    drift = _check_forecast(problem, drifts)
    if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
        raise ValidationError(f"risk aversion must be non-negative, got {risk_aversion!r}")

    periods = problem.periods
    quantity = problem.quantity
    tau = problem.tau
    impact = 2.0 * problem.eta_tilde / tau
    risk = 2.0 * risk_aversion * problem.volatility**2 * tau

    interior = periods - 1
    if interior <= 0:
        # One period: everything trades at once and no forecast can change it.
        trades = [quantity]
    else:
        diagonal = [2.0 * impact + risk] * interior
        right = [-drift[index] for index in range(interior)]
        right[0] += impact * quantity
        holdings = _solve_tridiagonal(diagonal, -impact, right)
        trades = [quantity - holdings[0]]
        trades.extend(holdings[index] - holdings[index + 1] for index in range(interior - 1))
        trades.append(holdings[-1])

    worst = min(trades)
    round_trips = worst < 0.0
    if round_trips:
        # schedule_moments would price this through LinearImpact.temporary,
        # which refuses a negative rate -- correctly, since a negative rate is
        # not a smaller cost but a different trade. So the trajectory is built
        # without the impact cost and the objective is computed from the
        # quadratic form directly.
        trajectory = Trajectory(
            times=tuple(problem.times()),
            holdings=tuple(_holdings_of(quantity, trades)),
            trades=tuple(trades),
            expected_cost=math.nan,
            variance=math.nan,
            risk_aversion=risk_aversion,
        )
        held = trajectory.holdings
        objective = (
            0.5 * impact * math.fsum((held[k] - held[k + 1]) ** 2 for k in range(periods))
            + 0.5 * risk * math.fsum(held[k + 1] ** 2 for k in range(periods))
            + math.fsum(held[k + 1] * drift[k] for k in range(periods - 1))
        )
        carried = math.fsum(held[k + 1] * drift[k] for k in range(periods - 1))
    else:
        trajectory = trajectory_from_trades(problem, trades, risk_aversion=risk_aversion)
        carried = math.fsum(trajectory.holdings[k + 1] * drift[k] for k in range(periods - 1))
        objective = trajectory.objective(risk_aversion) + carried

    blind = optimal_trajectory(problem, risk_aversion)
    blind_objective = forecast_objective(problem, risk_aversion, blind.trades, drift)
    return ForecastSchedule(
        trajectory=trajectory,
        forecast_cost=carried,
        objective=objective,
        blind_objective=blind_objective,
        round_trips=round_trips,
        worst_trade=min(worst, 0.0),
    )


def _holdings_of(quantity: float, trades: Sequence[float]) -> list[float]:
    holdings = [quantity]
    for trade in trades:
        holdings.append(holdings[-1] - trade)
    holdings[-1] = 0.0
    return holdings


def response_kernel(problem: ExecutionProblem, risk_aversion: float) -> list[list[float]]:
    """The Green's function of the Almgren-Chriss operator, in closed form.

    The first-order conditions are ``a (2I - S) x + c x = -mu`` for the shift
    ``S``, which is the discrete operator whose homogeneous solutions are the
    ``sinh`` profiles Almgren and Chriss already solve for. So its inverse is
    known::

        G(k, j) = sinh(kappa tau min) sinh(kappa tau (N - max))
                  / (a sinh(kappa tau) sinh(kappa tau N))

    with ``a = 2 eta_tilde / tau`` and ``kappa`` the *same* urgency
    :meth:`~slippage.execution.ExecutionProblem.kappa` returns. The smoothing
    a forecast gets is therefore not a new parameter: it is the one the
    forecast-blind problem already had.

    At a zero risk aversion the ``sinh`` degenerate to their arguments and the
    kernel is exactly triangular, ``min (N - max) / (a N)`` -- the Green's
    function of a discrete Laplacian. Which is why the response to a spike
    looks like a tent and not like an exponential decay, and why no decay
    constant is exposed here.

    Returned as an ``(N + 1) x (N + 1)`` matrix indexed by period, with the
    boundary rows and columns zero, so that ``G[k][j]`` lines up with the
    holdings. Nothing in it comes from the solver, which is the point: it is
    what the solver is checked against.
    """
    if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
        raise ValidationError(f"risk aversion must be non-negative, got {risk_aversion!r}")
    periods = problem.periods
    tau = problem.tau
    scale = 2.0 * problem.eta_tilde / tau
    kappa = problem.kappa(risk_aversion)
    kernel = [[0.0] * (periods + 1) for _ in range(periods + 1)]
    for row in range(1, periods):
        for column in range(1, periods):
            low, high = min(row, column), max(row, column)
            if kappa == 0.0:
                kernel[row][column] = low * (periods - high) / (scale * periods)
            else:
                kernel[row][column] = (
                    math.sinh(kappa * tau * low)
                    * math.sinh(kappa * tau * (periods - high))
                    / (scale * math.sinh(kappa * tau) * math.sinh(kappa * tau * periods))
                )
    return kernel


def forecast_value(
    problem: ExecutionProblem, risk_aversion: float, drifts: Sequence[float]
) -> float:
    """What a forecast is worth, in closed form: ``mu' G mu / 2``.

    No solve in it. Two consequences the tests pin down, both exact rather than
    measured:

    * The value is **quadratic** in the forecast, so doubling it quadruples the
      value.
    * The value is therefore **identical for a forecast and its negative**. A
      signal saying "hurry" and one saying "wait" are worth exactly the same,
      which is not what anybody expects of a trading signal -- the schedule
      moves in opposite directions and the improvement does not.
    """
    drift = _check_forecast(problem, drifts)
    kernel = response_kernel(problem, risk_aversion)
    total = 0.0
    for row in range(1, problem.periods):
        for column in range(1, problem.periods):
            total += drift[row - 1] * kernel[row][column] * drift[column - 1]
    return 0.5 * total


def spike_response(
    problem: ExecutionProblem, risk_aversion: float, period: int, size: float = 1.0
) -> list[float]:
    """The change in holdings caused by a forecast spike in one period.

    The system is linear, so this is a column of its inverse and depends on
    neither the order size nor the rest of the forecast. It is the kernel that
    says how a signal is smoothed, and its decay is the Almgren-Chriss urgency
    rather than a parameter of its own.

    Args:
        problem: The order and the impact model.
        risk_aversion: ``lambda``.
        period: Which period carries the spike, from 1 to ``periods - 1``.
            Period ``periods`` has no effect and is refused rather than
            silently returning zeros.
        size: Height of the spike, in price per share.
    """
    if not 1 <= period <= problem.periods - 1:
        raise ValidationError(
            f"a spike moves the schedule only in periods 1 to {problem.periods - 1}, "
            f"got {period}. The final period's drift cannot change anything: there "
            "is nothing left to trade by then."
        )
    drifts = [0.0] * problem.periods
    drifts[period - 1] = size
    with_spike = forecast_schedule(problem, risk_aversion, drifts)
    flat = forecast_schedule(problem, risk_aversion, [0.0] * problem.periods)
    return [
        after - before
        for after, before in zip(
            with_spike.trajectory.holdings, flat.trajectory.holdings, strict=True
        )
    ]


def proportional_tilt(
    problem: ExecutionProblem,
    risk_aversion: float,
    drifts: Sequence[float],
    strength: float,
) -> list[float]:
    """The heuristic: tilt each period's quantity by the signal's share.

    Trades the forecast-blind schedule adjusted by ``strength`` times each
    period's share of the signal, renormalised to the order size. It is what
    gets built when the forecast arrives and the optimiser does not, and it is
    here to be measured against the optimum rather than dismissed.

    ``strength`` of zero returns the blind schedule. The sign is such that a
    positive drift -- a price moving against a buyer -- pulls quantity earlier.
    """
    drift = _check_forecast(problem, drifts)
    blind = list(optimal_trajectory(problem, risk_aversion).trades)
    scale = math.fsum(abs(value) for value in drift)
    if scale == 0.0:
        return blind
    cumulative = 0.0
    weights = []
    for value in drift:
        cumulative += value
        weights.append(cumulative)
    centre = math.fsum(weights) / len(weights)
    tilted = [
        base * (1.0 - strength * (weight - centre) / scale)
        for base, weight in zip(blind, weights, strict=True)
    ]
    tilted = [max(value, 0.0) for value in tilted]
    total = math.fsum(tilted)
    if total <= 0.0:  # pragma: no cover - needs every period clipped to zero
        return blind
    return [value * problem.quantity / total for value in tilted]
