"""What a schedule built on the wrong parameters costs.

An execution schedule is computed from an impact coefficient and a volatility
that were estimated from data, and they are wrong. This module says by how much
that matters, and the answer has a shape worth stating before any of the
arithmetic.

**The schedule sees one number.** Writing out the mean-variance objective of an
arbitrary schedule under linear impact::

    E + lambda V = epsilon X + gamma X**2 / 2            (schedule-independent)
                   + (eta_tilde / tau) sum_k n_k**2
                   + lambda sigma**2 tau sum_k x_k**2

Everything that depends on the schedule is ``eta_tilde`` times
``sum n**2 / tau + r tau sum x**2`` with ``r = lambda sigma**2 / eta_tilde``. So
the trades depend on the three parameters and the risk aversion only through
``r``: two sets of beliefs with the same ratio produce the *same schedule*, bit
for bit, and the fixed per-share cost ``epsilon`` never enters. A
three-dimensional question about misspecification is a one-dimensional one about
``r``, and :func:`urgency_ratio` is the only number worth getting right.

**The forecast is wrong to first order; the schedule is wrong to second.** The
schedule sits at a minimum of the objective, so moving it by a first-order
amount changes the objective by a second-order one. The cost *prediction* enjoys
no such protection. Getting the ratio wrong by 10% makes the forecast 3.4% wrong
and the trades 0.038% wrong — a factor of ninety — and the gap widens as the
error shrinks, because it is the gap between the two orders.

**How much it can ever cost.** In the urgent limit the relative regret has a
closed form, ``(sqrt(k) - 1)**2 / (2 sqrt(k))`` for a ratio believed ``k`` times
its true value, whose expansion is ``epsilon**2 / 8``. That ``1/8`` is an upper
bound approached from below: expanding ``F`` below in both arguments gives
``(a**2 - y**2)**2 / 45`` for a patient trader, so the coefficient falls like
``y**4 / 45`` in ``y = sqrt(r) T``, the horizon measured in half-lives. A trader
working an order over many half-lives is almost immune to getting the parameters
wrong; an urgent one is not, and even then a factor-of-two error costs 6.1% of
the schedule-dependent objective.

**The asymmetry, and which way it runs.** The urgent-limit form is symmetric
under ``k -> 1 / k``, but at finite urgency the coefficient *rises* with
urgency, so the costly configuration is always the more urgent one — and that
settles both asymmetries at once. For a fixed truth, believing the ratio 1.5
times too large costs 1.70 times what believing it 1.5 times too small does,
because the first lands the trader in the steeper regime. And for a fixed
belief, a truth 1.5 times higher costs 1.47 times a truth 1.5 times lower, which
is why :func:`robust_urgency` finds its worst case at the fast end of a range of
possible truths and leans fast to meet it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .exceptions import ValidationError
from .execution import ExecutionProblem, Trajectory, optimal_trajectory, schedule_moments

__all__ = [
    "Misspecification",
    "RobustUrgency",
    "continuous_regret",
    "misspecification",
    "robust_urgency",
    "schedule_dependent_objective",
    "trajectory_from_ratio",
    "urgency_number",
    "urgency_ratio",
]


def _check_risk_aversion(risk_aversion: float) -> float:
    if not math.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValidationError(
            f"risk aversion must be positive here, got {risk_aversion!r}: a "
            "risk-neutral trader's schedule is the straight line whatever the "
            "parameters are, so there is nothing to misspecify"
        )
    return risk_aversion


def urgency_ratio(problem: ExecutionProblem, risk_aversion: float) -> float:
    """``lambda sigma**2 / eta_tilde``: everything the schedule sees.

    The units are one over time squared, and its square root is the
    continuous-time ``kappa``, so its reciprocal square root is the trade's
    half-life.
    """
    _check_risk_aversion(risk_aversion)
    return risk_aversion * problem.volatility**2 / problem.eta_tilde


def urgency_number(problem: ExecutionProblem, risk_aversion: float) -> float:
    """``sqrt(urgency_ratio) * horizon``: the horizon in continuous-time half-lives.

    This is the single dimensionless number the regret's size depends on. Below
    about 0.3 the trader is patient enough that misspecification is nearly free;
    above about 12 the regret has reached its ceiling.
    """
    return math.sqrt(urgency_ratio(problem, risk_aversion)) * problem.horizon


def schedule_dependent_objective(
    problem: ExecutionProblem, trades: tuple[float, ...] | list[float], risk_aversion: float
) -> float:
    """``E + lambda V`` less the part no schedule can change.

    ``epsilon X + gamma X**2 / 2`` is paid by every schedule that executes the
    order, so leaving it in the denominator of a relative regret would make the
    regret look smaller for a wider spread. It is also exactly the term a
    hand-rolled quadratic objective drops by accident, so it is subtracted here
    rather than never added.
    """
    expected, variance = schedule_moments(problem, trades)
    quantity = problem.quantity
    fixed = problem.impact.epsilon * quantity + 0.5 * problem.impact.gamma * quantity**2
    return expected + risk_aversion * variance - fixed


def _with_ratio(problem: ExecutionProblem, risk_aversion: float, ratio: float) -> ExecutionProblem:
    """The same problem seen by a trader who believes the urgency ratio is ``ratio``.

    Only the volatility is moved, because only the ratio matters: scaling
    ``sigma**2`` and ``eta_tilde`` by the same factor leaves the trades
    unchanged, and a test holds the library to that.
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValidationError(f"ratio must be positive and finite, got {ratio!r}")
    volatility = math.sqrt(ratio * problem.eta_tilde / risk_aversion)
    return ExecutionProblem(
        quantity=problem.quantity,
        horizon=problem.horizon,
        periods=problem.periods,
        volatility=volatility,
        impact=problem.impact,
    )


def trajectory_from_ratio(
    problem: ExecutionProblem, risk_aversion: float, ratio: float
) -> Trajectory:
    """The schedule of a trader who believes the urgency ratio is ``ratio``.

    The moments attached to it are the *true* ones, computed under ``problem``,
    which is the point: the trader chose the trades on one belief and will pay
    for them under another.
    """
    _check_risk_aversion(risk_aversion)
    believed = _with_ratio(problem, risk_aversion, ratio)
    trades = optimal_trajectory(believed, risk_aversion).trades
    expected, variance = schedule_moments(problem, trades)
    holdings = [problem.quantity]
    for trade in trades:
        holdings.append(holdings[-1] - trade)
    holdings[-1] = 0.0
    return Trajectory(
        times=tuple(problem.times()),
        holdings=tuple(holdings),
        trades=trades,
        expected_cost=expected,
        variance=variance,
        risk_aversion=risk_aversion,
        kappa=believed.kappa(risk_aversion),
    )


@dataclass(frozen=True)
class Misspecification:
    """What a schedule built on the wrong parameters costs, and what it predicted.

    Attributes:
        believed_ratio: ``lambda sigma**2 / eta_tilde`` as the trader believed it.
        true_ratio: The same under the true parameters.
        ratio_error: ``believed / true - 1``. The only error that matters.
        forecast: Schedule-dependent objective the trader predicted, which is
            their own schedule scored under their own beliefs.
        realised: The same schedule scored under the truth.
        optimal: The best any schedule achieves under the truth.
        regret: ``realised - optimal``. Non-negative by construction.
        regret_fraction: ``regret / optimal``, the number to quote.
        forecast_fraction: ``forecast / realised - 1``, how wrong the prediction
            was. Compare the two: the first is second order in the parameter
            error and the second is first order.
    """

    believed_ratio: float
    true_ratio: float
    ratio_error: float
    forecast: float
    realised: float
    optimal: float
    regret: float
    regret_fraction: float
    forecast_fraction: float

    @property
    def forecast_to_regret(self) -> float:
        """How many times larger the forecast error is than the regret.

        Diverges as the parameter error shrinks, which is the whole point. It is
        infinite when the parameters are right and the regret is exactly zero.
        """
        if self.regret_fraction == 0.0:
            return math.inf
        return abs(self.forecast_fraction) / self.regret_fraction


def misspecification(
    believed: ExecutionProblem, true: ExecutionProblem, risk_aversion: float
) -> Misspecification:
    """Score the schedule built from ``believed`` against the world ``true``.

    The risk aversion is a preference rather than a parameter, so it is the same
    in both: a trader does not misestimate how much they dislike risk.

    Args:
        believed: The problem as the trader thinks it is.
        true: The problem as it is.
        risk_aversion: ``lambda``, positive.

    Raises:
        ValidationError: If the two problems describe different orders — a
            different quantity, horizon or interval count is a different
            problem, not a misspecified one.
    """
    _check_risk_aversion(risk_aversion)
    for name in ("quantity", "horizon", "periods"):
        if getattr(believed, name) != getattr(true, name):
            raise ValidationError(
                f"the two problems differ in {name} "
                f"({getattr(believed, name)!r} against {getattr(true, name)!r}); "
                "misspecification is about the parameters, not about which order "
                "is being executed"
            )
    believed_trades = optimal_trajectory(believed, risk_aversion).trades
    optimal_trades = optimal_trajectory(true, risk_aversion).trades
    forecast = schedule_dependent_objective(believed, believed_trades, risk_aversion)
    realised = schedule_dependent_objective(true, believed_trades, risk_aversion)
    best = schedule_dependent_objective(true, optimal_trades, risk_aversion)
    believed_r = urgency_ratio(believed, risk_aversion)
    true_r = urgency_ratio(true, risk_aversion)
    regret = max(realised - best, 0.0)
    return Misspecification(
        believed_ratio=believed_r,
        true_ratio=true_r,
        ratio_error=believed_r / true_r - 1.0,
        forecast=forecast,
        realised=realised,
        optimal=best,
        regret=regret,
        regret_fraction=regret / best,
        forecast_fraction=forecast / realised - 1.0,
    )


def continuous_regret(believed_ratio: float, true_ratio: float, horizon: float) -> float:
    """The relative regret in continuous time, in closed form.

    With holdings ``x(t) = X sinh(a (T - t) / T) / sinh(a)`` for a believed
    ``a = sqrt(believed_ratio) T``, the schedule-dependent objective is
    ``eta_tilde X**2 / T`` times::

        F(a, y) = [ (a**2 - y**2) / 2 + (a**2 + y**2) sinh(2a) / (4a) ] / sinh(a)**2

    where ``y = sqrt(true_ratio) T``. It is minimised at ``a == y``, where it is
    ``y coth(y)`` — the Almgren-Chriss result — and the relative regret is
    ``F(a, y) / (y coth y) - 1``.

    This is the ``periods -> infinity`` limit of :func:`misspecification`'s
    ``regret_fraction``, which it is checked against: the discrete figure
    approaches it like one over the interval count.
    """
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValidationError(f"horizon must be positive and finite, got {horizon!r}")
    for name, value in (("believed_ratio", believed_ratio), ("true_ratio", true_ratio)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValidationError(f"{name} must be positive and finite, got {value!r}")
    a = math.sqrt(believed_ratio) * horizon
    y = math.sqrt(true_ratio) * horizon
    return max(_shape(a, y) / (y / math.tanh(y)) - 1.0, 0.0)


_LARGE = 350.0
"""Above this, ``sinh`` overflows and the ratio is taken in the exponents."""


def _shape(a: float, y: float) -> float:
    """``F(a, y)``, written so that an urgent trader does not overflow it."""
    if a > _LARGE:
        # sinh(2a) / sinh(a)**2 -> 2 / sinh(a) -> 0 for the first term, and
        # (a**2 + y**2) sinh(2a) / (4 a sinh(a)**2) -> (a**2 + y**2) / (2a).
        return (a * a + y * y) / (2.0 * a)
    sinh_a = math.sinh(a)
    return ((a * a - y * y) / 2.0 + (a * a + y * y) * math.sinh(2.0 * a) / (4.0 * a)) / (
        sinh_a * sinh_a
    )


@dataclass(frozen=True)
class RobustUrgency:
    """The urgency that minimises the worst regret over a range of possible truths.

    Attributes:
        ratio: The urgency ratio to build the schedule from.
        worst_regret: Largest ``regret_fraction`` it suffers anywhere in the
            range.
        worst_at: The true ratio at which that worst case falls.
        geometric_worst: Worst case of the geometric midpoint of the range,
            which is the natural guess.
        arithmetic_worst: Worst case of the arithmetic midpoint, which is what a
            point estimate in the middle of the parameter's range gives.
    """

    ratio: float
    worst_regret: float
    worst_at: float
    geometric_worst: float
    arithmetic_worst: float

    @property
    def gain_over_geometric(self) -> float:
        """How many times smaller the robust worst case is than the geometric guess."""
        if self.worst_regret == 0.0:
            return math.inf
        return self.geometric_worst / self.worst_regret


def robust_urgency(
    problem: ExecutionProblem,
    risk_aversion: float,
    low: float,
    high: float,
    *,
    grid: int = 201,
) -> RobustUrgency:
    """Minimise the worst-case regret over true ratios in ``[low, high]``.

    The worst case of a given belief is a maximum over the range, taken on a
    grid; the belief is then chosen by a golden-section search on that maximum,
    which is legitimate because the worst case is unimodal in the belief — it
    rises on both sides of wherever it is least, since moving the belief towards
    one end of the range moves it away from the other.

    The answer lands *above* the geometric midpoint of the range, because the
    regret's coefficient rises with urgency and so the binding case is always
    the fast end of the range. It is worth a real amount: over a factor of two
    either way the worst case falls from 2.56% of the schedule-dependent
    objective to 1.92%, and over a factor of ten either way from 47.4% to
    23.3%.

    Args:
        problem: The order. Only its horizon and interval count matter here.
        risk_aversion: ``lambda``, positive.
        low: Smallest possible true urgency ratio.
        high: Largest possible true urgency ratio.
        grid: Points at which the worst case is evaluated, spaced evenly in the
            logarithm of the ratio.
    """
    _check_risk_aversion(risk_aversion)
    if not math.isfinite(low) or low <= 0.0:
        raise ValidationError(f"low must be positive and finite, got {low!r}")
    if not math.isfinite(high) or high <= low:
        raise ValidationError(f"high must exceed low, got {high!r} against {low!r}")
    if grid < 2:
        raise ValidationError(f"grid must be at least two points, got {grid!r}")
    step = (math.log(high) - math.log(low)) / (grid - 1)
    candidates = [math.exp(math.log(low) + step * index) for index in range(grid)]

    def worst(belief: float) -> tuple[float, float]:
        at, value = candidates[0], -math.inf
        for truth in candidates:
            here = misspecification(
                _with_ratio(problem, risk_aversion, belief),
                _with_ratio(problem, risk_aversion, truth),
                risk_aversion,
            ).regret_fraction
            if here > value:
                at, value = truth, here
        return value, at

    left, right = math.log(low), math.log(high)
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    first, second = right - invphi * (right - left), left + invphi * (right - left)
    value_first, value_second = worst(math.exp(first))[0], worst(math.exp(second))[0]
    for _ in range(80):
        if right - left < 1e-12 * max(1.0, abs(left)):
            break
        if value_first < value_second:
            right, second, value_second = second, first, value_first
            first = right - invphi * (right - left)
            value_first = worst(math.exp(first))[0]
        else:
            left, first, value_first = first, second, value_second
            second = left + invphi * (right - left)
            value_second = worst(math.exp(second))[0]
    best = math.exp(0.5 * (left + right))
    best_worst, best_at = worst(best)
    return RobustUrgency(
        ratio=best,
        worst_regret=best_worst,
        worst_at=best_at,
        geometric_worst=worst(math.sqrt(low * high))[0],
        arithmetic_worst=worst(0.5 * (low + high))[0],
    )
