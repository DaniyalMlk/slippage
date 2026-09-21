"""Optimal execution trajectories under linear impact (Almgren and Chriss, 2000).

A trader must execute ``X`` shares over a horizon ``T`` split into ``N``
intervals of length ``tau``. Trading fast pays temporary impact; trading slowly
leaves the position exposed to price risk for longer. For a risk aversion
``lambda`` the schedule minimising ``E[cost] + lambda * Var[cost]`` under
linear impact is, in holdings,

    x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T),

where ``kappa`` solves ``2 / tau**2 * (cosh(kappa * tau) - 1) = lambda *
sigma**2 / eta_tilde`` and ``eta_tilde = eta - gamma * tau / 2``. As ``lambda``
goes to zero ``kappa`` does too and the trajectory becomes a straight line: the
risk-neutral trader works the order evenly. ``1 / kappa`` is the trade's
half-life, the time over which a risk-averse trader works off most of the
position.

Two computational choices matter here.

* **Moments come from direct summation over the schedule**, not from the
  published closed forms. The closed forms contain ``sinh(2 kappa T)`` and
  overflow for aggressive traders long before the trajectory itself becomes
  hard to compute; the tests check the two against each other where both are
  finite.
* **The trajectory is evaluated as a ratio of exponentials** when ``kappa T``
  is large, so an impatient trader's schedule does not become ``inf / inf``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .exceptions import ValidationError
from .impact import LinearImpact, schedule_cost

__all__ = [
    "ExecutionProblem",
    "HalfLifeSensitivity",
    "Trajectory",
    "closed_form_moments",
    "efficient_frontier",
    "half_life_sensitivity",
    "linear_trajectory",
    "optimal_trajectory",
    "schedule_moments",
    "trajectory_from_trades",
]

# Beyond this, sinh(kappa T) is evaluated through exponentials to avoid
# overflow; below it the direct form is accurate and clearer.
_DIRECT_SINH_LIMIT = 20.0


@dataclass(frozen=True)
class ExecutionProblem:
    """The inputs that define a liquidation or acquisition problem.

    Parameters
    ----------
    quantity
        Shares to execute, ``X``. Unsigned: buying and selling are symmetric.
    horizon
        Total time available, ``T``, in the time unit of ``volatility`` and of
        the impact model's ``eta``.
    periods
        Number of trading intervals, ``N``.
    volatility
        Price volatility ``sigma`` in price units per square root of the time
        unit — dollars per share per root day, for instance, not a percentage.
    impact
        A :class:`~slippage.impact.LinearImpact` model.
    """

    quantity: float
    horizon: float
    periods: int
    volatility: float
    impact: LinearImpact

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantity) or self.quantity <= 0.0:
            raise ValidationError(f"quantity must be positive, got {self.quantity!r}")
        if not math.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValidationError(f"horizon must be positive, got {self.horizon!r}")
        if self.periods < 1:
            raise ValidationError(f"periods must be at least 1, got {self.periods!r}")
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValidationError(f"volatility must be non-negative, got {self.volatility!r}")
        if self.eta_tilde <= 0.0:
            raise ValidationError(
                f"eta - gamma * tau / 2 = {self.eta_tilde:.4g} is not positive: permanent "
                "impact outweighs temporary impact at this interval length, so trading "
                "everything in one interval would look cheapest. Use more periods."
            )

    @property
    def tau(self) -> float:
        """Length of one trading interval."""
        return self.horizon / self.periods

    @property
    def eta_tilde(self) -> float:
        """Temporary impact net of the permanent impact within an interval."""
        return self.impact.eta - 0.5 * self.impact.gamma * self.tau

    def times(self) -> list[float]:
        """Interval boundaries ``t_0 = 0, ..., t_N = T``."""
        return [j * self.tau for j in range(self.periods + 1)]

    def kappa(self, risk_aversion: float) -> float:
        """Urgency ``kappa`` for a risk aversion ``lambda``.

        Solves ``2 / tau**2 * (cosh(kappa tau) - 1) = lambda sigma**2 / eta_tilde``
        exactly, as ``arccosh(1 + z) = log1p(z + sqrt(z (z + 2)))`` with
        ``z = kappa_tilde**2 tau**2 / 2``, which keeps full precision when
        ``z`` is tiny rather than rounding ``1 + z`` to one.
        """
        if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
            raise ValidationError(f"risk aversion must be non-negative, got {risk_aversion!r}")
        kappa_tilde_sq = risk_aversion * self.volatility**2 / self.eta_tilde
        z = 0.5 * kappa_tilde_sq * self.tau**2
        return math.log1p(z + math.sqrt(z * (z + 2.0))) / self.tau


@dataclass(frozen=True)
class Trajectory:
    """A schedule and its expected cost and variance.

    ``holdings`` has ``N + 1`` entries from ``X`` down to zero; ``trades`` has
    ``N``, the amount executed in each interval.
    """

    times: tuple[float, ...]
    holdings: tuple[float, ...]
    trades: tuple[float, ...]
    expected_cost: float
    variance: float
    risk_aversion: float | None = None
    kappa: float | None = None

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def objective(self, risk_aversion: float) -> float:
        """``E + lambda V``, the quantity the optimal trajectory minimises."""
        return self.expected_cost + risk_aversion * self.variance

    @property
    def half_life(self) -> float:
        """``1 / kappa``; infinite for a risk-neutral schedule."""
        if self.kappa is None or self.kappa == 0.0:
            return math.inf
        return 1.0 / self.kappa


def schedule_moments(problem: ExecutionProblem, trades: Sequence[float]) -> tuple[float, float]:
    """Expected cost and variance of an arbitrary schedule.

    Expected cost is the impact cost from :func:`~slippage.impact.schedule_cost`.
    Variance comes from the price diffusing while shares are still held: the
    ``x_k`` shares held through interval ``k`` carry ``sigma**2 tau x_k**2``.
    """
    if len(trades) != problem.periods:
        raise ValidationError(f"expected {problem.periods} trades, got {len(trades)}")
    total = sum(trades)
    if not math.isclose(total, problem.quantity, rel_tol=1e-9, abs_tol=1e-9):
        raise ValidationError(f"trades sum to {total}, not the quantity {problem.quantity}")
    cost = schedule_cost(problem.impact, trades, problem.tau).total
    held = problem.quantity
    variance = 0.0
    for n in trades:
        held -= n
        variance += held * held
    variance *= problem.volatility**2 * problem.tau
    return cost, variance


def trajectory_from_trades(
    problem: ExecutionProblem,
    trades: Sequence[float],
    *,
    risk_aversion: float | None = None,
    kappa: float | None = None,
) -> Trajectory:
    """Wrap a schedule of trades with its holdings and moments."""
    expected, variance = schedule_moments(problem, trades)
    holdings = [problem.quantity]
    for n in trades:
        holdings.append(holdings[-1] - n)
    holdings[-1] = 0.0
    return Trajectory(
        times=tuple(problem.times()),
        holdings=tuple(holdings),
        trades=tuple(trades),
        expected_cost=expected,
        variance=variance,
        risk_aversion=risk_aversion,
        kappa=kappa,
    )


def _sinh_ratio(kappa: float, remaining: float, horizon: float) -> float:
    """``sinh(kappa * remaining) / sinh(kappa * horizon)`` without overflow."""
    if kappa == 0.0:
        return remaining / horizon
    if kappa * horizon < _DIRECT_SINH_LIMIT:
        return math.sinh(kappa * remaining) / math.sinh(kappa * horizon)
    # sinh(a)/sinh(b) = exp(a - b) * (1 - exp(-2a)) / (1 - exp(-2b))
    a = kappa * remaining
    b = kappa * horizon
    return math.exp(a - b) * (-math.expm1(-2.0 * a)) / (-math.expm1(-2.0 * b))


def optimal_trajectory(problem: ExecutionProblem, risk_aversion: float) -> Trajectory:
    """The Almgren-Chriss optimal schedule for a risk aversion ``lambda``.

    ``risk_aversion`` is in inverse currency: ``lambda = 1e-6`` means one
    dollar of expected cost is worth a million dollars squared of variance.
    """
    kappa = problem.kappa(risk_aversion)
    horizon = problem.horizon
    holdings = [
        problem.quantity * _sinh_ratio(kappa, horizon - t, horizon) for t in problem.times()
    ]
    holdings[-1] = 0.0
    # The trades telescope to exactly X - 0. An earlier version pushed the
    # summation residue into the last trade, which for an impatient trader is
    # itself ~1e-10 and could be driven negative.
    trades = [max(holdings[j] - holdings[j + 1], 0.0) for j in range(problem.periods)]
    return trajectory_from_trades(problem, trades, risk_aversion=risk_aversion, kappa=kappa)


def linear_trajectory(problem: ExecutionProblem) -> Trajectory:
    """The risk-neutral schedule: equal trades in every interval."""
    return optimal_trajectory(problem, 0.0)


def closed_form_moments(problem: ExecutionProblem, risk_aversion: float) -> tuple[float, float]:
    """Expected cost and variance from Almgren and Chriss (2000), equation (20).

    Kept as an independent check on :func:`schedule_moments`. It overflows for
    large ``kappa T`` (it contains ``sinh(2 kappa T)``), which is why the
    library's trajectories do not rely on it.
    """
    x = problem.quantity
    gamma = problem.impact.gamma
    epsilon = problem.impact.epsilon
    eta_t = problem.eta_tilde
    sigma = problem.volatility
    tau = problem.tau
    horizon = problem.horizon
    kappa = problem.kappa(risk_aversion)
    fixed = 0.5 * gamma * x * x + epsilon * x
    if kappa == 0.0:
        n = problem.periods
        expected = fixed + eta_t * x * x / horizon
        variance = sigma**2 * x * x * horizon * (1.0 - 1.0 / n) * (1.0 - 0.5 / n) / 3.0
        return expected, variance
    sinh_kt = math.sinh(kappa * horizon)
    sinh_ktau = math.sinh(kappa * tau)
    expected = fixed + eta_t * x * x * (
        math.tanh(0.5 * kappa * tau)
        * (tau * math.sinh(2.0 * kappa * horizon) + 2.0 * horizon * sinh_ktau)
        / (2.0 * tau * tau * sinh_kt**2)
    )
    variance = (
        0.5
        * sigma**2
        * x
        * x
        * (tau * sinh_kt * math.cosh(kappa * (horizon - tau)) - horizon * sinh_ktau)
        / (sinh_kt**2 * sinh_ktau)
    )
    return expected, variance


def efficient_frontier(
    problem: ExecutionProblem, risk_aversions: Iterable[float]
) -> list[Trajectory]:
    """Optimal trajectories across a range of risk aversions, in the order given.

    Plotting ``variance`` against ``expected_cost`` traces the efficient
    frontier: no schedule has both lower expected cost and lower variance than
    a point on it.
    """
    return [optimal_trajectory(problem, lam) for lam in risk_aversions]


@dataclass(frozen=True)
class HalfLifeSensitivity:
    """Elasticities of the trade's half-life ``1 / kappa`` to each input.

    Each figure is ``d log(half-life) / d log(parameter)``: an elasticity of
    ``-0.5`` means a 10% rise in the parameter shortens the half-life by about
    5%. In the continuous-time limit these are exactly ``-1/2`` for risk
    aversion, ``-1`` for volatility and ``+1/2`` for temporary impact, and the
    discrete values approach them as the interval shrinks. Permanent impact
    enters only through ``eta_tilde`` and so vanishes in that limit.
    """

    half_life: float
    risk_aversion: float
    volatility: float
    eta: float
    gamma: float


def half_life_sensitivity(problem: ExecutionProblem, risk_aversion: float) -> HalfLifeSensitivity:
    """Analytic elasticities of the half-life, from the discrete kappa relation.

    With ``K = lambda sigma**2 / eta_tilde`` and ``cosh(kappa tau) = 1 + K tau**2 / 2``,
    implicit differentiation gives the elasticity of ``kappa`` in ``K`` as
    ``K tau / (2 kappa sinh(kappa tau))``. The chain rule through ``K`` then
    gives each parameter's elasticity; the half-life's are their negatives.

    Raises :class:`~slippage.exceptions.ValidationError` for zero risk aversion,
    where the half-life is infinite and has no elasticity.
    """
    if risk_aversion <= 0.0:
        raise ValidationError("the half-life of a risk-neutral schedule is infinite")
    kappa = problem.kappa(risk_aversion)
    tau = problem.tau
    k_value = risk_aversion * problem.volatility**2 / problem.eta_tilde
    kappa_in_k = k_value * tau / (2.0 * kappa * math.sinh(kappa * tau))
    eta_t = problem.eta_tilde
    return HalfLifeSensitivity(
        half_life=1.0 / kappa,
        risk_aversion=-kappa_in_k,
        volatility=-2.0 * kappa_in_k,
        eta=kappa_in_k * problem.impact.eta / eta_t,
        gamma=-kappa_in_k * 0.5 * problem.impact.gamma * tau / eta_t,
    )
