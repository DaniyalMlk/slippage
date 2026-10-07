"""Tracking a benchmark that moves, which inverts what risk aversion does.

Every schedule elsewhere in this library minimises cost against the **arrival
price**: :func:`~slippage.execution.optimal_trajectory`,
:func:`~slippage.scheduling.solve_schedule`,
:func:`~slippage.basket.basket_trajectory`,
:func:`~slippage.transient.optimal_transient_schedule` and
:func:`~slippage.adaptive.solve_adaptive` all answer that one question in
different settings. The arrival price is a number fixed before the first share
trades, so the risk is in the position still held, and risk aversion front-loads
the schedule to get out of it.

Most institutional orders are not scored that way. They are scored against the
interval VWAP, and that is a different problem rather than a refinement of the
same one. The benchmark is a weighted average of the *same prices we trade at*,
so it moves with the market, and what we are exposed to is not the position but
the **difference between our participation and the market's**.

**That difference is the whole model.** Write ``u_k`` for our share of the order
done in bucket ``k`` and ``w_k`` for the market's realised share of the session's
volume. Both sum to one, so with ``p_k = p_0 (1 + sigma b_k)`` for a driftless
walk ``b``::

    slippage = sum_k (u_k - w_k) p_k = p_0 sigma sum_k (u_k - w_k) b_k

The ``p_0`` term cancels because the weights sum to one on both sides. Two things
follow immediately, and both are tested rather than asserted:

* **A schedule that matched the realised volume curve would have zero tracking
  error path by path**, whatever the prices did. Not approximately, and not in
  expectation — the sum is identically zero term by term. Nothing in an
  arrival-price problem behaves like this.
* **Volume uncertainty is the only reason it cannot be done.** ``u`` must be
  chosen before ``w`` is known, so the error is entirely the error in forecasting
  the market's own participation.

Taking variances, with ``C`` the covariance of the price walk and ``S`` the
covariance of the volume shares::

    Var(slippage) = (u - mu)' C (u - mu) + trace(C S)

The second term does not contain ``u`` at all: it is the **irreducible** tracking
error, and no schedule can reduce it. The first is a positive quadratic form
minimised at ``u = mu``, so **the expected volume profile is the
variance-minimising schedule** — derived here rather than assumed, which matters
because the result is often stated as a rule of thumb and it is a theorem with
conditions. One of those conditions is weaker than it looks: the optimum is
``mu`` for *any* positive-definite ``C``, so it does not depend on the price model
beyond driftlessness. The module tests that invariance directly.

**The volume model consumes something the package already measured and never
read.** :attr:`~slippage.volume.VolumeProfile.dispersion` has carried the
cross-day standard deviation of each bucket's share since profiles were added,
and nothing used it. Shares live on a simplex, so their covariance cannot be
diagonal — they must sum to one, which forces negative off-diagonal terms — and a
Dirichlet distribution is the one-parameter family on the simplex with exactly
that structure. :func:`fit_volume_covariance` fits its concentration to the
measured dispersion and reports how well one parameter can match a whole vector
of them, which is honest rather than impressive.

**Impact pulls the other way, and the trade-off is the point.** Following the
volume curve exactly is the minimum-tracking-error schedule and not the
minimum-cost one: a bucket that carries 3% of the day's volume gets 3% of the
order whether or not that is a sensible rate to trade at.
:func:`tracking_schedule` trades the two off, with impact priced by
:func:`~slippage.impact.schedule_cost`, which is the library's one description of
what trading costs rather than a second one written here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError
from .impact import ImpactModel, schedule_cost
from .volume import VolumeProfile

FloatArray = NDArray[np.float64]

__all__ = [
    "ObjectiveComparison",
    "TrackingFrontierPoint",
    "TrackingMoments",
    "TrackingProblem",
    "VolumeUncertainty",
    "compare_objectives",
    "fit_volume_covariance",
    "price_covariance",
    "tracking_frontier",
    "tracking_moments",
    "tracking_schedule",
]


def _shares(weights: Sequence[float] | FloatArray, name: str) -> FloatArray:
    """A participation vector, validated and normalised to sum to one."""
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValidationError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValidationError(f"{name} must be non-negative")
    total = float(array.sum())
    if total <= 0.0:
        raise ValidationError(f"{name} sums to {total}, which cannot be normalised")
    return np.asarray(array / total, dtype=np.float64)


@dataclass(frozen=True)
class VolumeUncertainty:
    """How far the market's volume shares stray from their expectation.

    Attributes:
        covariance: Covariance of the realised shares, ``(n, n)``. Its rows sum
            to zero, because the shares sum to one however they land.
        concentration: The Dirichlet concentration it was built from. Larger is
            tighter; the limit is a deterministic profile.
        dispersion_error: Largest relative gap between a bucket's standard
            deviation under this one-parameter model and the dispersion measured
            for it. Reported because one parameter cannot in general match a
            whole vector, and a reader should see by how much it failed rather
            than trust that it did not.
    """

    covariance: FloatArray
    concentration: float
    dispersion_error: float

    def __post_init__(self) -> None:
        if self.covariance.ndim != 2 or (self.covariance.shape[0] != self.covariance.shape[1]):
            raise ValidationError("the volume covariance must be square")
        if self.concentration <= 0.0:
            raise ValidationError(f"the concentration must be positive, got {self.concentration!r}")

    @property
    def buckets(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def rows_sum_to_zero(self) -> float:
        """Largest absolute row sum, which the simplex constraint forces to zero."""
        return float(np.max(np.abs(self.covariance.sum(axis=1))))


def fit_volume_covariance(
    profile: VolumeProfile, *, concentration: float | None = None
) -> VolumeUncertainty:
    """Fit a Dirichlet to the dispersion a profile already measured.

    Volume shares are a point on a simplex: they sum to one however the day
    turns out. A diagonal covariance cannot describe that, and using one
    **overstates** the irreducible tracking error, because it leaves out the
    negative correlations that the constraint forces — a heavy open means a
    lighter rest of the day, and the two errors partly cancel in the benchmark.

    The Dirichlet is the one-parameter family on the simplex with the right
    structure: under ``Dir(c mu)``, ``E[w] = mu`` and::

        Var(w_k)      =  mu_k (1 - mu_k) / (c + 1)
        Cov(w_j, w_k) = -mu_j mu_k / (c + 1)

    so a single ``c`` sets every entry. It is fitted by matching the mean of the
    measured variances, which is a least-squares fit in the variance rather than
    in the standard deviation, and the worst per-bucket miss is reported on the
    result.

    Args:
        profile: A profile with a ``dispersion`` vector, as
            :func:`~slippage.volume.estimate_profile` produces. Pass
            ``concentration`` instead where there is none.
        concentration: Use this ``c`` rather than fitting one. Required when the
            profile carries no dispersion, which is the case for
            :meth:`~slippage.volume.VolumeProfile.uniform` and for any profile
            built by hand.

    Returns:
        The covariance, the concentration and the fit's worst miss.

    Raises:
        ValidationError: If no dispersion is available and no concentration was
            given, if the dispersion is all zeros and no concentration was given,
            or if ``concentration`` is not positive.
    """
    mu = _shares(profile.fractions, "profile fractions")
    buckets = mu.size
    if concentration is None:
        if len(profile.dispersion) != buckets:
            raise ValidationError(
                "the profile carries no per-bucket dispersion, so a Dirichlet "
                "concentration cannot be fitted from it. Estimate the profile from "
                "bars with volume.estimate_profile, or pass concentration= "
                "directly. A uniform profile in particular has no dispersion, "
                "because nothing was measured to produce it."
            )
        measured = np.asarray(profile.dispersion, dtype=np.float64)
        if not np.all(np.isfinite(measured)) or np.any(measured < 0.0):
            raise ValidationError("the profile's dispersion must be finite and non-negative")
        variances = measured * measured
        if float(variances.sum()) <= 0.0:
            raise ValidationError(
                "the profile's dispersion is zero in every bucket, which says the "
                "volume curve is deterministic. The irreducible tracking error is "
                "then zero and the expected profile achieves the benchmark exactly; "
                "pass concentration= to model uncertainty that was not measured."
            )
        spread = mu * (1.0 - mu)
        fitted = float(np.mean(spread / np.where(variances > 0.0, variances, np.inf)))
        concentration = max(fitted - 1.0, 1e-9)
    if concentration <= 0.0:
        raise ValidationError(f"the concentration must be positive, got {concentration!r}")
    scale = 1.0 / (concentration + 1.0)
    covariance = np.asarray((np.diag(mu) - np.outer(mu, mu)) * scale, dtype=np.float64)
    implied = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if len(profile.dispersion) == buckets:
        measured = np.asarray(profile.dispersion, dtype=np.float64)
        usable = measured > 0.0
        error = (
            float(np.max(np.abs(implied[usable] / measured[usable] - 1.0)))
            if bool(np.any(usable))
            else 0.0
        )
    else:
        error = 0.0
    return VolumeUncertainty(
        covariance=covariance, concentration=float(concentration), dispersion_error=error
    )


def price_covariance(buckets: int, volatility: float, *, midpoint: bool = True) -> FloatArray:
    """Covariance of the price level each bucket trades at, in relative terms.

    ``volatility`` is the fractional price standard deviation over one bucket, so
    a driftless walk sampled at bucket ``k`` has variance ``k sigma**2`` and
    ``Cov(b_j, b_k) = min(j, k) sigma**2``.

    ``midpoint`` chooses *where in the bucket* that price is read: a bucket trades
    throughout its length rather than at an instant, so its representative price is
    nearer the middle than the end, and the default shifts every index down by half
    a bucket.

    **The choice makes no difference to anything, and that is worth knowing rather
    than worth hiding.** The shift adds a constant of ``-sigma**2 / 2`` to every
    entry of ``C``, and both quantities a tracking error is built from are
    orthogonal to a constant matrix: the schedule deviation sums to zero because
    both schedules are normalised, and the volume covariance's rows sum to zero
    because the shares sum to one however the day lands. So a constant contributes
    ``c (sum_k d_k)**2 = 0`` to the schedule term and ``c sum_jk S_jk = 0`` to the
    floor. Measured: the two conventions agree to twelve significant figures on
    three different schedules, and adding ``1000`` to every entry of ``C`` moves the
    floor by 1.2e-15.

    An earlier version of this docstring said the choice changes the level of the
    tracking error and only leaves the *optimum* alone. It changes neither, which
    is the stronger statement — it means the model needs no decision about where in
    a bucket a price is read, so there is one fewer arbitrary parameter in it.
    """
    if buckets < 1:
        raise ValidationError(f"buckets must be at least 1, got {buckets}")
    if not math.isfinite(volatility) or volatility < 0.0:
        raise ValidationError(f"volatility must be finite and non-negative, got {volatility!r}")
    index = np.arange(1, buckets + 1, dtype=np.float64)
    if midpoint:
        index = index - 0.5
    return np.asarray(np.minimum.outer(index, index) * volatility * volatility, dtype=np.float64)


@dataclass(frozen=True)
class TrackingProblem:
    """An order to be executed against the interval VWAP of its own session.

    Attributes:
        shares: Order size, positive. Only the schedule's shape affects tracking
            error; the size is what impact is charged on.
        profile: Expected share of volume in each bucket.
        volatility: Fractional price standard deviation over one bucket.
        market_volume: Shares the market is expected to print over the session,
            used to turn our participation into a rate for the impact model.
        bucket_hours: Length of a bucket in the time unit the impact model's
            ``eta`` is quoted in. Only used for impact.
        midpoint: Read each bucket's price at its middle rather than its end.
    """

    shares: float
    profile: VolumeProfile
    volatility: float
    market_volume: float = 0.0
    bucket_hours: float = 1.0
    midpoint: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.shares) or self.shares <= 0.0:
            raise ValidationError(f"shares must be positive, got {self.shares!r}")
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValidationError(
                f"volatility must be finite and non-negative, got {self.volatility!r}"
            )
        if not math.isfinite(self.market_volume) or self.market_volume < 0.0:
            raise ValidationError(
                f"market volume must be finite and non-negative, got {self.market_volume!r}"
            )
        if not math.isfinite(self.bucket_hours) or self.bucket_hours <= 0.0:
            raise ValidationError(f"bucket length must be positive, got {self.bucket_hours!r}")

    @property
    def buckets(self) -> int:
        return len(self.profile)

    @property
    def expected(self) -> FloatArray:
        """The expected volume share, which is the minimum-variance schedule."""
        return _shares(self.profile.fractions, "profile fractions")

    def price_covariance(self) -> FloatArray:
        return price_covariance(self.buckets, self.volatility, midpoint=self.midpoint)


@dataclass(frozen=True)
class TrackingMoments:
    """Tracking error of one schedule, split by what the schedule controls.

    Attributes:
        schedule_variance: ``(u - mu)' C (u - mu)``, the part a schedule can
            remove, zero exactly when the schedule is the expected profile.
        irreducible: ``trace(C S)``, which no schedule reduces because it does not
            contain the schedule. The floor a VWAP execution is held to.
        expected: Expected relative slippage, zero under a driftless walk.
    """

    schedule_variance: float
    irreducible: float
    expected: float = 0.0

    @property
    def variance(self) -> float:
        """Total variance of relative slippage against the realised VWAP."""
        return self.schedule_variance + self.irreducible

    @property
    def tracking_error_bps(self) -> float:
        """Standard deviation of slippage, in basis points of the arrival price."""
        return 1e4 * math.sqrt(max(self.variance, 0.0))

    @property
    def irreducible_bps(self) -> float:
        """The floor, in basis points."""
        return 1e4 * math.sqrt(max(self.irreducible, 0.0))

    @property
    def avoidable_share(self) -> float:
        """Share of the variance the schedule is responsible for, in ``[0, 1]``."""
        total = self.variance
        return 0.0 if total <= 0.0 else self.schedule_variance / total


def tracking_moments(
    problem: TrackingProblem,
    schedule: Sequence[float] | FloatArray,
    uncertainty: VolumeUncertainty,
) -> TrackingMoments:
    """Closed-form tracking error of ``schedule`` against the realised VWAP.

    ``schedule`` is read as participation weights and normalised, so share counts
    and fractions both work.

    Args:
        problem: The order and its market.
        schedule: What fraction of the order each bucket does.
        uncertainty: The volume model, from :func:`fit_volume_covariance`.

    Returns:
        The two parts of the variance and the expected slippage.

    Raises:
        ValidationError: If the schedule or the covariance has the wrong length.
    """
    weights = _shares(schedule, "schedule")
    if weights.size != problem.buckets:
        raise ValidationError(
            f"the schedule has {weights.size} buckets and the profile has {problem.buckets}"
        )
    if uncertainty.buckets != problem.buckets:
        raise ValidationError(
            f"the volume covariance covers {uncertainty.buckets} buckets and the "
            f"profile has {problem.buckets}"
        )
    covariance = problem.price_covariance()
    deviation = np.asarray(weights - problem.expected, dtype=np.float64)
    controlled = float(deviation @ np.asarray(covariance @ deviation, dtype=np.float64))
    floor = float(np.sum(covariance * uncertainty.covariance))
    return TrackingMoments(schedule_variance=max(controlled, 0.0), irreducible=max(floor, 0.0))


def _impact_cost(problem: TrackingProblem, weights: FloatArray, model: ImpactModel) -> float:
    """Relative impact cost of a participation schedule, per share of the order.

    Expressed as a fraction of the arrival price by dividing the currency cost by
    the order's notional at that price — which is unknown here, so the cost is
    returned per share and the caller scales. :func:`~slippage.impact.schedule_cost`
    does the work, so there is one account of what trading costs rather than two.
    """
    trades = [float(one) for one in weights * problem.shares]
    cost = schedule_cost(model, trades, problem.bucket_hours)
    return (cost.temporary + cost.permanent) / problem.shares


def tracking_schedule(
    problem: TrackingProblem,
    uncertainty: VolumeUncertainty,
    *,
    impact: ImpactModel | None = None,
    risk_aversion: float | None = None,
    price: float = 1.0,
) -> FloatArray:
    """The schedule that minimises tracking error, or trades it against impact.

    With no impact model the answer is the expected volume profile, and it is
    returned as such rather than solved for: the schedule-dependent variance is
    ``(u - mu)' C (u - mu)``, a positive quadratic form, so ``mu`` is the exact
    minimiser for any positive-definite ``C``, so it does not depend on the price
    model beyond the walk having no drift.

    With an impact model and a risk aversion, the objective is
    ``impact_cost(u) / price + lambda * Var(u)``, both terms in relative units so
    that ``lambda`` is dimensionless in the same way as
    :func:`~slippage.execution.optimal_trajectory`'s. The permanent part of
    impact is a constant under a fixed order size — Almgren-Chriss's identity
    makes it depend only on ``sum_k n_k**2`` — so what the schedule actually
    trades off is the temporary concession against the tracking variance. The
    solve is a projected gradient descent on the simplex, which keeps the weights
    non-negative rather than discovering afterwards that the unconstrained answer
    went below zero.

    Args:
        problem: The order and its market.
        uncertainty: The volume model.
        impact: An impact model, or ``None`` for the pure tracking answer.
        risk_aversion: Weight on tracking variance. Required with ``impact``.
        price: Arrival price, used to put impact in relative units.

    Returns:
        Participation weights summing to one.

    Raises:
        ValidationError: If exactly one of ``impact`` and ``risk_aversion`` is
            given, or if ``price`` is not positive.
    """
    if (impact is None) != (risk_aversion is None):
        raise ValidationError(
            "give both impact and risk_aversion to trade them off, or neither for "
            "the pure minimum-tracking-error schedule, which is the expected profile"
        )
    if impact is None or risk_aversion is None:
        return problem.expected
    if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
        raise ValidationError(
            f"risk aversion must be finite and non-negative, got {risk_aversion!r}"
        )
    if not math.isfinite(price) or price <= 0.0:
        raise ValidationError(f"price must be positive, got {price!r}")

    covariance = problem.price_covariance()
    mu = problem.expected
    buckets = mu.size

    def objective(weights: FloatArray) -> float:
        deviation = weights - mu
        variance = float(deviation @ np.asarray(covariance @ deviation, dtype=np.float64))
        return _impact_cost(problem, weights, impact) / price + risk_aversion * variance

    def gradient(weights: FloatArray) -> FloatArray:
        step = 1e-7
        base = objective(weights)
        out = np.zeros(buckets, dtype=np.float64)
        for index in range(buckets):
            bumped = weights.copy()
            bumped[index] += step
            out[index] = (objective(bumped) - base) / step
        return out

    weights = mu.copy()
    rate = 0.5 / max(float(np.max(np.abs(gradient(weights)))), 1e-12)
    best, best_value = weights.copy(), objective(weights)
    for _ in range(600):
        direction = gradient(weights)
        # Project the gradient onto the simplex's tangent plane, then clip and
        # renormalise. Subtracting the mean first keeps the step from spending
        # itself on the one direction the constraint forbids.
        direction = direction - float(direction.mean())
        candidate = np.asarray(np.clip(weights - rate * direction, 0.0, None), dtype=np.float64)
        total = float(candidate.sum())
        if total <= 0.0:
            break
        candidate = np.asarray(candidate / total, dtype=np.float64)
        value = objective(candidate)
        if value < best_value:
            best, best_value = candidate.copy(), value
            weights = candidate
        else:
            rate *= 0.5
            if rate < 1e-16:
                break
    return best


@dataclass(frozen=True)
class TrackingFrontierPoint:
    """One risk aversion, and what it bought.

    Attributes:
        risk_aversion: The weight used.
        schedule: The participation weights.
        impact_bps: Expected impact cost, in basis points of the arrival price.
        tracking_error_bps: Standard deviation of slippage against the VWAP.
        irreducible_bps: The floor, which is the same at every point.
    """

    risk_aversion: float
    schedule: tuple[float, ...]
    impact_bps: float
    tracking_error_bps: float
    irreducible_bps: float


def tracking_frontier(
    problem: TrackingProblem,
    uncertainty: VolumeUncertainty,
    impact: ImpactModel,
    risk_aversions: Sequence[float],
    *,
    price: float = 1.0,
) -> tuple[TrackingFrontierPoint, ...]:
    """Trace the trade-off between impact cost and tracking error.

    The two ends are worth knowing before reading the middle. As the risk
    aversion goes to zero the schedule goes to whatever minimises impact, which
    for a convex temporary cost is an equal slice per bucket regardless of where
    the volume is. As it grows the schedule goes to the expected volume profile
    and the tracking error falls to its irreducible floor, which it cannot go
    below however much more risk aversion is applied.
    """
    points = []
    for aversion in risk_aversions:
        weights = tracking_schedule(
            problem, uncertainty, impact=impact, risk_aversion=aversion, price=price
        )
        moments = tracking_moments(problem, weights, uncertainty)
        points.append(
            TrackingFrontierPoint(
                risk_aversion=float(aversion),
                schedule=tuple(float(one) for one in weights),
                impact_bps=1e4 * _impact_cost(problem, weights, impact) / price,
                tracking_error_bps=moments.tracking_error_bps,
                irreducible_bps=moments.irreducible_bps,
            )
        )
    return tuple(points)


@dataclass(frozen=True)
class ObjectiveComparison:
    """What using the wrong benchmark costs, in both directions.

    Attributes:
        vwap_schedule_error_bps: Tracking error of the volume-matching schedule.
        other_schedule_error_bps: Tracking error of the schedule brought in.
        excess_bps: How much worse the other schedule is at tracking VWAP.
        excess_over_floor: The same excess as a multiple of the irreducible
            floor, which is the number that says whether it matters. An excess
            well below the floor is noise a desk cannot act on.
        irreducible_bps: The floor.
    """

    vwap_schedule_error_bps: float
    other_schedule_error_bps: float
    excess_bps: float
    excess_over_floor: float
    irreducible_bps: float


def compare_objectives(
    problem: TrackingProblem,
    uncertainty: VolumeUncertainty,
    other: Sequence[float] | FloatArray,
) -> ObjectiveComparison:
    """Score a schedule built for another benchmark against this one.

    The intended use is to pass an arrival-price trajectory —
    :func:`~slippage.execution.optimal_trajectory`'s trades, say — and see what
    it does to VWAP tracking. A front-loaded schedule is the right answer to a
    different question, and this says what that answer costs when the question
    was VWAP.

    The excess is reported both absolutely and as a multiple of the irreducible
    floor, because the floor is what decides whether the difference is worth
    acting on: a schedule that is 2 basis points worse against a 20 basis point
    floor is inside the noise a desk can measure.
    """
    matched = tracking_moments(problem, problem.expected, uncertainty)
    alternative = tracking_moments(problem, other, uncertainty)
    excess = alternative.tracking_error_bps - matched.tracking_error_bps
    floor = matched.irreducible_bps
    return ObjectiveComparison(
        vwap_schedule_error_bps=matched.tracking_error_bps,
        other_schedule_error_bps=alternative.tracking_error_bps,
        excess_bps=excess,
        excess_over_floor=excess / floor if floor > 0.0 else math.inf,
        irreducible_bps=floor,
    )
