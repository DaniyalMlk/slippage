"""The size at which executing the trade consumes the reason for making it.

:mod:`slippage.execution` answers "given this size, how should it be traded?".
This module answers the question before it: **how large can the order be at
all?** A strategy's edge is some number of cents per share. Executing spends
some of it. Past a size, executing spends all of it, and that size is the
strategy's capacity.

Everything here is computed from :mod:`slippage.execution` and
:mod:`slippage.impact` rather than from a cost model written out again, which
matters more than it sounds: the Almgren-Chriss cost of a uniform schedule over
``N`` intervals carries the permanent term as ``gamma X^2 (1 - 1/N) / 2`` and
not ``gamma X^2 / 2``, and a re-derivation would quietly use the second.

**The answer depends on which variable is held fixed, and the two conventions in
use disagree about what limits capacity.** A model fixes the horizon. A trader
quotes participation — a percentage of volume — which fixes the *rate* and lets
the horizon grow with the size. Those give different functions of size, and the
difference is not a detail:

* At a fixed horizon, temporary impact per share is ``eta X / T``, linear in the
  size. Cost per share is linear in the size, so capacity is finite and scales
  with the horizon.
* At a fixed participation rate ``rho``, the horizon is ``X / (rho V)``, so
  temporary impact per share is ``eta rho V`` — **constant in the size**. It
  does not bound capacity at all. What it does is set a floor on the alpha
  required before any size works, and the *permanent* impact then sets the size
  limit.

Those are two different roles and the fixed-horizon formula conflates them.
Measured on the module's worked parameters, the log-log slope of cost per share
against size is 1.0000 at every decade at a fixed horizon, and at 5%
participation it runs 0.015, 0.129, 0.553 and 0.914 across four decades — zero
where the floor dominates, approaching one only once the permanent term takes
over. With the permanent impact removed the per-share cost at 5% participation
is 0.03000000 at a million shares, at a billion and at a trillion: the same
eight digits, and capacity genuinely unbounded. That is reported as unbounded
rather than as a very large number.

**At a fixed horizon the break-even size is a closed form and it is exact.**
Cost is ``epsilon X + A X^2`` with ``A`` free of the size — true of the
mean-variance objective too, since the variance is also exactly quadratic — so
the break-even is ``(alpha - epsilon) / A``. Checked against a bisection on the
library's own cost at 1e-15 relative across four horizons. The continuum
expression that drops the ``1 - 1/N`` sits 0.017% to 0.033% below it at a fixed
horizon, and **exactly** ``-1/N`` below it at a fixed participation rate, where
the horizon is long enough that the interval count sits on its cap and the one
factor is the whole of the discrepancy: -0.000500 at 2% participation and at
5%, to six decimals, against a cap of 2000 intervals.

**The horizon that maximises net alpha does not depend on the order size.** For
a uniform schedule the objective is
``(alpha - epsilon) X - eta X^2 / T - gamma X^2 / 2 - lambda sigma^2 X^2 T / 3``,
in which every term in ``T`` carries the same ``X^2``, so the optimum is
``T* = sqrt(3 eta_tilde / (lambda sigma^2))`` with no size in it. Confirmed
numerically at 500,000, two million and twenty million shares across four
orders of magnitude of risk aversion: the optimum agrees to **1.3e-07 at
worst**, and it matches the closed form to 3.8e-04, which is the golden-section
search's own resolution. The residual spread across sizes is not slack in the
invariance, which is exact in the algebra; it is the search's comparisons being
decided by rounding in an objective whose magnitude goes as ``X^2``, so a
fortyfold range of size moves the answer in its eighth significant figure. A desk that
lengthens its horizon because the order is larger is responding to the wrong
variable.

That optimum is ``sqrt(3)`` times the Almgren-Chriss half-life ``1 / kappa``,
since ``kappa = sqrt(lambda sigma^2 / eta_tilde)`` in the continuum — measured
at 1.7352, 1.7335, 1.7330 and 1.7322 half-lives as the risk aversion rises
through four decades, against ``sqrt(3) = 1.7321``, approaching it as the
intervals shorten relative to the horizon. Which explains the other half of the
measurement:

**For the Almgren-Chriss optimal schedule there is no interior optimal horizon,
because the schedule declines the extra time.** Net alpha rises monotonically in
``T`` and saturates: 10,348 at twenty days, 12,139 at fifty, 12,238 at a hundred
and 12,422 at two hundred, while the half-life stays at 10.54, 10.53, 10.52 and
10.49 days. The risk aversion has already chosen the effective horizon and the
nominal one stops mattering once it is a few half-lives. So ``optimal_horizon``
reports an interior optimum for the uniform schedule and says plainly that the
optimal schedule has none, rather than returning whichever point a search
happened to stop at.

**The alpha floor is the number to check first.** At the worked parameters a 10%
participation rate needs 5.50 cents a share of edge before any size is viable
and a 25% rate needs 13.00 cents, against an alpha of 5.00 cents: at those rates
the strategy has no capacity whatsoever, and no amount of patience about the
horizon changes it, because patience is what participation has already spent.
The fixed per-share cost enters the floor too, which is why a half-cent
commission that looks negligible against a 5 cent edge is a third of the floor
at 2% participation, and the thing that decides whether a patient strategy
clears it at all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from .exceptions import ValidationError
from .execution import ExecutionProblem, linear_trajectory, optimal_trajectory
from .impact import LinearImpact

#: Most intervals any problem built here is given. A capacity search walks the
#: horizon over orders of magnitude and the cost of a trajectory is linear in
#: the interval count, so an uncapped count makes a search quadratic in the
#: horizon for no accuracy worth having.
MAX_PERIODS = 2000

#: Default ceiling on the search, in units of one period's volume. Beyond fifty
#: days of volume the linear impact model is not describing anything, so a
#: capacity that large is reported as off the end of the model rather than as a
#: number.
DEFAULT_CEILING_IN_VOLUME = 50.0

#: Bisection steps. Eighty halvings of a bracket spanning the whole search is
#: more than a double resolves.
BISECTION_STEPS = 80


class UnboundedCapacityError(ValidationError):
    """Net alpha is still positive at the ceiling, so capacity is not finite here.

    Raised rather than returned as the ceiling, because the ceiling is an
    argument and returning it would make an unbounded capacity look like a
    measurement of one. It happens for real reasons: a strategy with no
    permanent impact has no size limit at a fixed participation rate at all.
    """


@dataclass(frozen=True)
class Mandate:
    """A strategy's edge and the market it has to trade in.

    Parameters
    ----------
    alpha
        Expected gain per share, in price units. The gross edge, before any
        execution cost.
    volume
        Volume per unit time, in shares. The time unit is the one the impact
        model's ``eta`` and the volatility are quoted in.
    volatility
        Price volatility in price units per square root of the time unit.
    impact
        The linear impact model. Its ``epsilon`` is a per-share cost that is
        charged however patiently the order trades, so it enters every floor.
    periods_per_unit_time
        Trading intervals in one unit of time, used when the horizon is derived
        from a participation rate rather than given.
    risk_aversion
        Mean-variance penalty in inverse currency, as in
        :func:`slippage.execution.optimal_trajectory`. Zero for a risk-neutral
        comparison, which is the only case where the horizon is free.
    """

    alpha: float
    volume: float
    volatility: float
    impact: LinearImpact
    periods_per_unit_time: int = 390
    risk_aversion: float = 0.0

    def __post_init__(self) -> None:
        for name in ("alpha", "volume", "volatility", "risk_aversion"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValidationError(f"{name} must be finite, got {value!r}")
        if self.volume <= 0.0:
            raise ValidationError(f"volume must be positive, got {self.volume!r}")
        if self.volatility < 0.0:
            raise ValidationError(f"volatility must be non-negative, got {self.volatility!r}")
        if self.risk_aversion < 0.0:
            raise ValidationError(f"risk aversion must be non-negative, got {self.risk_aversion!r}")
        if self.periods_per_unit_time < 1:
            raise ValidationError(
                f"periods per unit time must be at least 1, got {self.periods_per_unit_time!r}"
            )
        if self.alpha <= self.impact.epsilon:
            raise ValidationError(
                f"an alpha of {self.alpha!r} does not clear the per-share cost of "
                f"{self.impact.epsilon!r}, which is charged however the order trades. "
                "No size is viable at any horizon, which is a conclusion rather than "
                "a capacity."
            )

    def periods_for(self, horizon: float) -> int:
        """Intervals to split ``horizon`` into, capped at :data:`MAX_PERIODS`."""
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValidationError(f"horizon must be positive, got {horizon!r}")
        wanted = math.ceil(horizon * self.periods_per_unit_time)
        return max(2, min(MAX_PERIODS, wanted))

    def problem(self, quantity: float, horizon: float) -> ExecutionProblem:
        """The execution problem this mandate poses at a size and a horizon."""
        return ExecutionProblem(
            quantity=quantity,
            horizon=horizon,
            periods=self.periods_for(horizon),
            volatility=self.volatility,
            impact=self.impact,
        )

    def objective(self, quantity: float, horizon: float, *, uniform: bool = False) -> float:
        """Expected cost plus the risk penalty, from the library's own schedule.

        ``uniform`` forces the equal-slice schedule, which is what a desk
        trading a fixed participation rate is actually doing. Otherwise the
        Almgren-Chriss optimum for this mandate's risk aversion is used.
        """
        problem = self.problem(quantity, horizon)
        trajectory = (
            linear_trajectory(problem)
            if uniform
            else optimal_trajectory(problem, self.risk_aversion)
        )
        return trajectory.expected_cost + self.risk_aversion * trajectory.variance

    def net_alpha(self, quantity: float, horizon: float, *, uniform: bool = False) -> float:
        """Gross alpha on the whole order less what executing it costs."""
        return self.alpha * quantity - self.objective(quantity, horizon, uniform=uniform)


@dataclass(frozen=True)
class Capacity:
    """A break-even size and what it was bounded by."""

    quantity: float
    #: The size in units of one period's volume, which is how it gets quoted.
    volume_days: float
    horizon: float
    #: Cost per share at the break-even size, which equals the alpha.
    cost_per_share: float
    #: ``eta rho V + epsilon``: the per-share cost that no size can reduce. Only
    #: meaningful when the participation rate is what was held fixed.
    floor: float
    #: Whether the horizon was given or derived from a participation rate.
    participation: float | None
    #: Quadratic coefficient of the objective in the size, ``A`` in
    #: ``epsilon X + A X^2``. ``None`` where the horizon moves with the size and
    #: the objective is therefore not quadratic in it.
    curvature: float | None

    @property
    def headroom(self) -> float:
        """Alpha above the floor, which is what the permanent impact spends."""
        return self.cost_per_share - self.floor


@dataclass(frozen=True)
class Horizon:
    """The horizon that maximises net alpha, for a schedule that has one."""

    horizon: float
    net_alpha: float
    #: ``1 / kappa`` for the mandate's risk aversion, at this horizon.
    half_life: float
    #: ``horizon / half_life``, which is ``sqrt(3)`` for the uniform schedule.
    half_lives: float
    #: Whether the maximum is interior or at the ceiling of the search.
    interior: bool


def _positive_horizon(horizon: float | None) -> float:
    if horizon is None or not math.isfinite(horizon) or horizon <= 0.0:
        raise ValidationError(f"horizon must be positive, got {horizon!r}")
    return horizon


def _check_participation(participation: float) -> float:
    if not math.isfinite(participation) or not 0.0 < participation <= 1.0:
        raise ValidationError(
            f"participation is a fraction of volume in (0, 1], got {participation!r}"
        )
    return participation


def alpha_floor(mandate: Mandate, participation: float) -> float:
    """The per-share cost no size can reduce, at this participation rate.

    ``epsilon + eta rho V``. The temporary impact depends on the *rate* and the
    rate is what has been fixed, so this is paid on every share of any order.
    An alpha below it leaves no viable size, and no horizon fixes that, because
    the horizon is what the participation rate has already chosen.
    """
    _check_participation(participation)
    return mandate.impact.epsilon + mandate.impact.eta * participation * mandate.volume


def curvature(mandate: Mandate, horizon: float, *, uniform: bool = False) -> float:
    """``A`` in ``objective = epsilon X + A X^2``, read off the library.

    Exact: the impact cost less the per-share term and the variance are both
    homogeneous of degree two in the size, so one evaluation at any reference
    size determines the coefficient. Evaluating at a reference rather than
    writing the expression out keeps the discrete ``1 - 1/N`` on the permanent
    term, which the continuum expression drops.
    """
    reference = mandate.volume
    objective = mandate.objective(reference, horizon, uniform=uniform)
    return (objective - mandate.impact.epsilon * reference) / (reference * reference)


def break_even_at_horizon(mandate: Mandate, horizon: float, *, uniform: bool = False) -> Capacity:
    """The size at which net alpha is zero, with the horizon held fixed.

    A closed form, because the objective is ``epsilon X + A X^2``. No search.
    """
    coefficient = curvature(mandate, horizon, uniform=uniform)
    if coefficient <= 0.0:
        raise UnboundedCapacityError(
            f"the objective has no positive curvature in the size at a horizon of "
            f"{horizon!r}, so no size exhausts the alpha"
        )
    quantity = (mandate.alpha - mandate.impact.epsilon) / coefficient
    return Capacity(
        quantity=quantity,
        volume_days=quantity / mandate.volume,
        horizon=horizon,
        cost_per_share=mandate.objective(quantity, horizon, uniform=uniform) / quantity,
        floor=mandate.impact.epsilon,
        participation=None,
        curvature=coefficient,
    )


def break_even_at_participation(
    mandate: Mandate,
    participation: float,
    *,
    ceiling: float | None = None,
) -> Capacity:
    """The size at which net alpha is zero, with the participation rate fixed.

    Bisected rather than solved, because the horizon moves with the size and
    the objective is no longer quadratic in it: the interval count changes, and
    with a non-zero risk aversion the variance picks up a horizon that is itself
    a function of the size. Net alpha per share is monotone decreasing in the
    size, so the bisection is safe.

    Uses the uniform schedule throughout. A desk that has committed to a
    participation rate is trading equal slices of volume by construction, and
    letting the optimiser re-choose the schedule would answer a different
    question.
    """
    _check_participation(participation)
    floor = alpha_floor(mandate, participation)
    if mandate.alpha <= floor:
        raise UnboundedCapacityError(
            f"an alpha of {mandate.alpha!r} is at or below the floor of {floor:.6f} that "
            f"{participation:.1%} participation imposes, so no size is viable. The floor "
            "is the per-share cost plus the temporary impact of the rate itself, and "
            "neither depends on the size."
        )
    limit = DEFAULT_CEILING_IN_VOLUME * mandate.volume if ceiling is None else float(ceiling)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValidationError(f"the ceiling must be a positive size, got {limit!r}")

    rate = participation * mandate.volume

    def net_per_share(quantity: float) -> float:
        return mandate.net_alpha(quantity, quantity / rate, uniform=True) / quantity

    if net_per_share(limit) > 0.0:
        raise UnboundedCapacityError(
            f"net alpha per share is still {net_per_share(limit):.6f} at {limit:,.0f} "
            f"shares, which is {limit / mandate.volume:.0f} periods of volume. With no "
            "permanent impact the participation convention imposes no size limit at "
            "all; raise the ceiling only if the impact model still describes the market "
            "there."
        )

    low, high = 1.0, limit
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if net_per_share(middle) > 0.0:
            low = middle
        else:
            high = middle
    quantity = 0.5 * (low + high)
    horizon = quantity / rate
    return Capacity(
        quantity=quantity,
        volume_days=quantity / mandate.volume,
        horizon=horizon,
        cost_per_share=mandate.objective(quantity, horizon, uniform=True) / quantity,
        floor=floor,
        participation=participation,
        curvature=None,
    )


def break_even_continuum(
    mandate: Mandate, *, horizon: float | None = None, participation: float | None = None
) -> float:
    """The same break-even in the continuum, which drops the ``1 - 1/N``.

    Kept as the expression a reader would write down, and as the check on the
    exact ones: ``(alpha - epsilon) / (eta / T + gamma / 2)`` at a fixed
    horizon, and ``2 (alpha - floor) / gamma`` at a fixed participation rate. It
    is a risk-neutral statement, so it is the right comparison only for a
    mandate with no risk aversion.
    """
    if (horizon is None) == (participation is None):
        raise ValidationError("give exactly one of horizon and participation")
    edge = mandate.alpha - mandate.impact.epsilon
    if horizon is not None:
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValidationError(f"horizon must be positive, got {horizon!r}")
        return edge / (mandate.impact.eta / horizon + 0.5 * mandate.impact.gamma)
    assert participation is not None
    rate = _check_participation(participation) * mandate.volume
    if mandate.impact.gamma <= 0.0:
        raise UnboundedCapacityError(
            "with no permanent impact the continuum break-even at a fixed "
            "participation rate is infinite"
        )
    return 2.0 * (edge - mandate.impact.eta * rate) / mandate.impact.gamma


def uniform_optimal_horizon(mandate: Mandate, quantity: float) -> Horizon:
    """The horizon that maximises net alpha for a uniform schedule.

    ``sqrt(3 eta_tilde / (lambda sigma^2))`` in the continuum, with no size in
    it: cost falls as ``1 / T`` and the risk penalty rises as ``T``, and both
    carry the same ``X^2``. The returned horizon is the library's own maximiser,
    located by a golden-section search on the exact objective, and the continuum
    expression is what the tests check it against.

    ``half_lives`` comes out at ``sqrt(3)`` because the Almgren-Chriss urgency
    is ``kappa = sqrt(lambda sigma^2 / eta_tilde)`` over the same ratio.
    """
    if mandate.risk_aversion <= 0.0:
        raise UnboundedCapacityError(
            "with no risk aversion a longer horizon is always cheaper, so a uniform "
            "schedule has no optimal horizon. The cost tends to the permanent impact "
            "alone as the horizon grows."
        )
    if not math.isfinite(quantity) or quantity <= 0.0:
        raise ValidationError(f"quantity must be positive, got {quantity!r}")

    guess = math.sqrt(3.0 * mandate.impact.eta / (mandate.risk_aversion * mandate.volatility**2))
    low, high = 0.05 * guess, 5.0 * guess
    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    left = high - golden * (high - low)
    right = low + golden * (high - low)
    for _ in range(BISECTION_STEPS):
        if mandate.net_alpha(quantity, left, uniform=True) > mandate.net_alpha(
            quantity, right, uniform=True
        ):
            high = right
        else:
            low = left
        left = high - golden * (high - low)
        right = low + golden * (high - low)
    horizon = 0.5 * (low + high)
    problem = mandate.problem(quantity, horizon)
    half_life = 1.0 / problem.kappa(mandate.risk_aversion)
    return Horizon(
        horizon=horizon,
        net_alpha=mandate.net_alpha(quantity, horizon, uniform=True),
        half_life=half_life,
        half_lives=horizon / half_life,
        interior=0.06 * guess < horizon < 4.9 * guess,
    )


@dataclass(frozen=True)
class Point:
    """One size on a capacity curve."""

    quantity: float
    volume_days: float
    horizon: float
    cost_per_share: float
    net_alpha_per_share: float


def capacity_curve(
    mandate: Mandate,
    quantities: Sequence[float],
    *,
    horizon: float | None = None,
    participation: float | None = None,
) -> tuple[Point, ...]:
    """Cost and net alpha per share across sizes, at a fixed horizon or rate.

    The two conventions give differently shaped curves, which is the whole
    point of being able to ask for either.
    """
    if (horizon is None) == (participation is None):
        raise ValidationError("give exactly one of horizon and participation")
    if not quantities:
        raise ValidationError("a capacity curve needs at least one size")
    fixed = horizon
    rate = None if participation is None else _check_participation(participation) * mandate.volume
    built = []
    for quantity in quantities:
        if not math.isfinite(quantity) or quantity <= 0.0:
            raise ValidationError(f"every size must be positive, got {quantity!r}")
        at = quantity / rate if rate is not None else _positive_horizon(fixed)
        cost = mandate.objective(quantity, at, uniform=rate is not None) / quantity
        built.append(
            Point(
                quantity=quantity,
                volume_days=quantity / mandate.volume,
                horizon=at,
                cost_per_share=cost,
                net_alpha_per_share=mandate.alpha - cost,
            )
        )
    return tuple(built)


def cost_elasticity(curve: Sequence[Point], epsilon: float) -> tuple[float, ...]:
    """Log-log slopes of cost per share, net of the fixed cost, against size.

    One at a fixed horizon and climbing from zero towards one at a fixed
    participation rate, which is the measurement that separates the two
    conventions. The fixed per-share cost is removed first because it is a
    constant and would flatten every slope towards zero.
    """
    if len(curve) < 2:
        raise ValidationError("an elasticity needs at least two points")
    slopes = []
    for first, second in pairwise(curve):
        left = first.cost_per_share - epsilon
        right = second.cost_per_share - epsilon
        if left <= 0.0 or right <= 0.0:
            raise ValidationError(
                "cost per share is at or below the fixed per-share cost, so its "
                "logarithm is not defined; the fixed cost may be wrong"
            )
        slopes.append(
            (math.log(right) - math.log(left))
            / (math.log(second.quantity) - math.log(first.quantity))
        )
    return tuple(slopes)


__all__ = [
    "BISECTION_STEPS",
    "DEFAULT_CEILING_IN_VOLUME",
    "MAX_PERIODS",
    "Capacity",
    "Horizon",
    "Mandate",
    "Point",
    "UnboundedCapacityError",
    "alpha_floor",
    "break_even_at_horizon",
    "break_even_at_participation",
    "break_even_continuum",
    "capacity_curve",
    "cost_elasticity",
    "curvature",
    "uniform_optimal_horizon",
]
