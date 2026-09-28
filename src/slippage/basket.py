"""Liquidating a basket, where the schedules are not separable.

:func:`~slippage.execution.optimal_trajectory` solves one name at a time.
Almost nothing is executed one name at a time, and a portfolio liquidation is not
a collection of single-asset problems for two reasons that change the answer
rather than refine it.

**Risk is a matrix.** The holdings still on the book through an interval carry
``x' Sigma x``, not a sum of ``sigma_i^2 x_i^2``. A hedged long/short pair carries
almost nothing while both legs are on and a great deal once one of them is gone,
so the schedule that minimises cost plus risk keeps them on *together*. Two
independent single-asset solutions cannot express that, and the way they get it
wrong is the dangerous direction: each leg is worked at the pace its own
volatility justifies, the faster one finishes first, and the book is left
outright at the end of the trade.

**Impact is a matrix too.** Trading one name moves the others — through the risk
model everyone else is hedging with, if nothing else. With a non-diagonal
temporary impact matrix the schedules couple even when the returns are
uncorrelated, which is the case that shows the coupling is about impact and not
only about risk.

The solution is exact rather than numerical, and the transformation that makes it
so is worth stating because it is also what makes the answer interpretable.
Writing ``E`` for the effective temporary impact matrix ``Eta - Gamma tau / 2``,
substitute ``y = E^(1/2) x``. The cost term becomes a plain sum of squared
increments of ``y`` and the risk term becomes ``y' A y`` with

    A = E^(-1/2) Sigma E^(-1/2)

which is symmetric and positive definite. Diagonalise it, ``A = U diag(mu) U'``,
and in each eigen-coordinate the problem is the scalar Almgren-Chriss problem
with unit impact and variance ``mu_i``: one urgency ``kappa_i`` solving
``2 / tau^2 (cosh(kappa_i tau) - 1) = lambda mu_i``, and a ``sinh`` trajectory.
Transform back for the holdings.

So a basket is liquidated along eigen-directions, each at its own pace, and
``mu_i`` is that direction's *risk per unit of impact*. The direction with the
largest ``mu`` is the one the trader is most anxious about and it is worked off
fastest; a direction with ``mu`` near zero is nearly costless to hold and is left
until the end. On a correlated long/short pair those two directions are the net
and the spread, which is exactly why the pair should be unwound as a pair.

**Measured, on the case the module exists for.** A long/short pair, 100,000
shares each, liquidated over a day in 20 intervals, returns correlated at 0.9, at
a risk aversion that makes ``kappa T`` about three. Against solving the two legs
as independent single-asset problems and scoring both schedules against the full
matrices, the joint solution saves **23.8%** of the objective with identical
legs, **12.1%** when the short leg is four times as expensive to trade, and
**6.1%** at ten times. With the returns uncorrelated and the legs identical the
two solutions coincide exactly, to the last bit, which is the check that the
saving is the correlation and not an artefact of the solver.

Two things about that are the opposite of what one would assume, and both are
recorded because the assumption is the natural one.

**The joint solution is riskier moment to moment, not safer.** Its peak
one-interval variance is 1.19 times the independent solution's on the symmetric
pair and 1.10 times on the illiquid one. A hedged pair is cheap to hold, so the
optimal schedule holds it *longer* and pays less impact. It is the cheaper
schedule at the same risk aversion, not the calmer one, and the risk aversion is
where the preference lives.

**Where the independent solution is actually wrong is the shape, and by an order
of magnitude.** It works each leg at the pace that leg's own liquidity justifies,
so on a pair whose legs differ in liquidity the liquid one finishes first and the
book is left outright in the middle of the trade. With the short leg four times as
expensive, the independent schedule reaches a net exposure of 18,961 shares of a
100,000-share pair against the joint solution's 2,025 — a factor of 9.4, against
an objective difference of 12.1%. On the symmetric pair both schedules keep the
hedge exactly and that metric is a ratio of two rounding errors, which
:meth:`BasketComparison.exposure_ratio` refuses rather than reports.

Sign convention: holdings are signed, positive long. A basket with a short leg
has a negative holding for it, and the trades that liquidate it are positive
buys. Costs are positive numbers of currency in every case, since impact is paid
whichever way the trade goes.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError
from .execution import ExecutionProblem, optimal_trajectory
from .impact import LinearImpact

FloatArray = NDArray[np.float64]
Floats = Sequence[float] | NDArray[np.floating[Any]]

#: Smallest eigenvalue, relative to the largest, that a matrix may have and still
#: be treated as positive definite. A matrix at the edge of singularity gives an
#: inverse square root with entries of order the reciprocal of that ratio, and the
#: trajectory it produces is arithmetic on rounding error rather than a schedule.
#: Relative rather than absolute, because an impact matrix in currency per share
#: squared and a covariance matrix in daily variance differ by many orders of
#: magnitude and an absolute floor would mean something different to each.
CONDITION_FLOOR = 1e-12

#: Most assets a basket will accept. The eigen-decomposition is cubic and the
#: schedule is ``periods * assets^2``, so this is about runtime rather than about
#: the method, and it says so.
MAX_BASKET_ASSETS = 200


def _as_matrix(values: Any, size: int, name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValidationError(
            f"{name} must be {size} by {size} for a basket of {size} assets, "
            f"got shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValidationError(f"{name} has a non-finite entry")
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    scale = float(np.max(np.abs(matrix))) or 1.0
    if asymmetry > 1e-9 * scale:
        raise ValidationError(
            f"{name} is not symmetric: it differs from its transpose by "
            f"{asymmetry:.6g}, which is {asymmetry / scale:.3g} of its largest "
            f"entry. A quadratic form only sees the symmetric part, so an "
            f"asymmetric matrix is a statement the arithmetic cannot honour."
        )
    return (matrix + matrix.T) / 2.0


def _eigendecompose(
    matrix: FloatArray, name: str, *, require_definite: bool = True
) -> tuple[FloatArray, FloatArray]:
    """Ascending eigenvalues and their vectors, or a refusal naming the matrix."""
    values, vectors = np.linalg.eigh(matrix)
    largest = float(values[-1])
    if largest <= 0.0:
        raise ValidationError(
            f"{name} has no positive eigenvalue, so it is not a positive definite matrix at all"
        )
    smallest = float(values[0])
    if require_definite and smallest <= CONDITION_FLOOR * largest:
        raise ValidationError(
            f"{name} is not positive definite to working precision: its smallest "
            f"eigenvalue is {smallest:.6g} against a largest of {largest:.6g}, a "
            f"ratio of {smallest / largest:.3g} against a floor of "
            f"{CONDITION_FLOOR:g}. An inverse square root of it would be "
            f"arithmetic on rounding error. Two assets with identical holdings "
            f"and identical impact are the usual cause; drop one and trade the "
            f"combination."
        )
    return values, vectors


def _inverse_square_roots(matrix: FloatArray, name: str) -> tuple[FloatArray, FloatArray]:
    """``M^(1/2)`` and ``M^(-1/2)`` for a symmetric positive definite ``M``."""
    values, vectors = _eigendecompose(matrix, name)
    roots = np.sqrt(values)
    forward = (vectors * roots) @ vectors.T
    inverse = (vectors / roots) @ vectors.T
    return forward, inverse


@dataclass(frozen=True)
class BasketProblem:
    """A basket to liquidate, and the matrices that decide how.

    ``holdings`` are signed and positive long. ``temporary`` and ``permanent`` are
    impact matrices: the cost of trading ``n`` shares in an interval of length
    ``tau`` is ``n' temporary n / tau``, and the permanent cost of the whole
    liquidation is ``holdings' permanent holdings / 2``. ``covariance`` is the
    covariance of returns *in currency per share over one unit of the horizon's
    clock*, so that ``x' covariance x * tau`` is a variance of currency.

    The single-asset problem this generalises writes those three as ``eta``,
    ``gamma`` and ``volatility ** 2``. A one-asset basket built from those numbers
    reproduces it exactly, which is the property the tests lean on hardest.
    """

    holdings: tuple[float, ...]
    temporary: tuple[tuple[float, ...], ...]
    permanent: tuple[tuple[float, ...], ...]
    covariance: tuple[tuple[float, ...], ...]
    horizon: float
    periods: int
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        size = len(self.holdings)
        if size == 0:
            raise ValidationError("a basket with no assets in it")
        if size > MAX_BASKET_ASSETS:
            raise ValidationError(
                f"{size} assets is above the {MAX_BASKET_ASSETS} this will "
                f"accept: the eigen-decomposition is cubic in the count"
            )
        if not all(math.isfinite(value) for value in self.holdings):
            raise ValidationError("a holding is not finite")
        if self.periods < 1:
            raise ValidationError(f"a liquidation needs at least one interval, got {self.periods}")
        if not math.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValidationError(f"the horizon must be positive and finite, got {self.horizon!r}")
        if self.names and len(self.names) != size:
            raise ValidationError(f"{len(self.names)} names for {size} assets")

    @property
    def assets(self) -> int:
        return len(self.holdings)

    @property
    def tau(self) -> float:
        return self.horizon / self.periods

    def labels(self) -> tuple[str, ...]:
        return self.names or tuple(f"asset{index}" for index in range(self.assets))

    def times(self) -> list[float]:
        return [index * self.tau for index in range(self.periods + 1)]

    def holdings_vector(self) -> FloatArray:
        return np.asarray(self.holdings, dtype=np.float64)

    def temporary_matrix(self) -> FloatArray:
        return _as_matrix(self.temporary, self.assets, "the temporary impact matrix")

    def permanent_matrix(self) -> FloatArray:
        return _as_matrix(self.permanent, self.assets, "the permanent impact matrix")

    def covariance_matrix(self) -> FloatArray:
        return _as_matrix(self.covariance, self.assets, "the covariance matrix")

    def effective_temporary(self) -> FloatArray:
        """``Eta - Gamma tau / 2``, the matrix the schedule actually sees.

        The same correction as the scalar ``eta_tilde``, and for the same reason:
        half of an interval's permanent impact is paid by the shares traded within
        that interval, so it offsets the temporary term rather than sitting
        outside the optimisation. Getting this wrong is invisible at a short
        interval and changes the urgency at a long one.
        """
        return self.temporary_matrix() - 0.5 * self.tau * self.permanent_matrix()

    @classmethod
    def from_single(
        cls,
        problem: ExecutionProblem,
        *,
        name: str = "asset0",
    ) -> BasketProblem:
        """The one-asset basket equivalent to a scalar problem.

        Exists for the tests rather than for callers, and it is the most valuable
        thing in this module for confidence: the basket solver has to reproduce
        the scalar solver on it, and the two share no arithmetic at all.
        """
        return cls(
            holdings=(problem.quantity,),
            temporary=((problem.impact.eta,),),
            permanent=((problem.impact.gamma,),),
            covariance=((problem.volatility**2,),),
            horizon=problem.horizon,
            periods=problem.periods,
            names=(name,),
        )


@dataclass(frozen=True)
class Direction:
    """One eigen-direction of the transformed problem, and its pace.

    These are what a basket is really liquidated along. ``weights`` is the
    direction in *share* space, normalised, so a long/short pair's directions read
    as a net and a spread rather than as coordinates of an intermediate basis.
    """

    #: Risk per unit of impact along this direction: the eigenvalue of
    #: ``E^(-1/2) Sigma E^(-1/2)``. Large means anxious, and worked off first.
    risk_per_impact: float
    #: Urgency, from the scalar relation at that eigenvalue.
    kappa: float
    #: Signed holdings along this direction at the start, in shares.
    initial: float
    weights: tuple[float, ...]

    @property
    def half_life(self) -> float:
        """``1 / kappa``, or infinity for a direction with no urgency."""
        if self.kappa <= 0.0:
            return math.inf
        return 1.0 / self.kappa


@dataclass(frozen=True)
class BasketTrajectory:
    """A basket schedule and what it costs."""

    times: tuple[float, ...]
    #: ``periods + 1`` rows of holdings, one per time, each a vector over assets.
    holdings: tuple[tuple[float, ...], ...]
    #: ``periods`` rows of trades, signed so that holdings plus trades telescope
    #: to zero. A positive trade reduces a long and a negative one covers a short.
    trades: tuple[tuple[float, ...], ...]
    expected_cost: float
    variance: float
    risk_aversion: float
    directions: tuple[Direction, ...]
    names: tuple[str, ...]

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def objective(self) -> float:
        return self.expected_cost + self.risk_aversion * self.variance

    def peak_interval_variance(self) -> float:
        """The largest one-interval variance the schedule carries after the start.

        The first interval is excluded and that is the whole reason this is a
        method rather than a one-liner: every complete liquidation of the same
        basket holds the same position through its first interval, so a maximum
        that includes it is the same number for every schedule and distinguishes
        nothing. A first draft of this did include it and read 1.000 for every
        comparison, which looked like agreement rather than like a broken metric.
        """
        covariance = np.asarray(self.covariance_used, dtype=np.float64)
        tau = self.times[1] - self.times[0] if len(self.times) > 1 else 0.0
        worst = 0.0
        for row in self.holdings[1:-1]:
            vector = np.asarray(row, dtype=np.float64)
            worst = max(worst, float(vector @ covariance @ vector) * tau)
        return worst

    def peak_exposure(self, weights: Floats) -> float:
        """The largest absolute exposure the schedule takes along ``weights``.

        The starting position is excluded, for the same reason the first interval
        is excluded above: every schedule of the same basket starts there, so
        including it makes a common number the maximum and hides the difference.
        """
        direction = np.asarray(weights, dtype=np.float64)
        if direction.shape != (len(self.names),):
            raise ValidationError(f"{direction.shape} weights for {len(self.names)} assets")
        return max(
            abs(float(np.asarray(row, dtype=np.float64) @ direction)) for row in self.holdings[1:]
        )

    def riskiest_direction(self) -> tuple[float, ...]:
        """The eigen-direction with the most risk per unit of impact.

        A property of the solution rather than of the basket, so a schedule built
        another way does not have one.
        """
        if not self.directions:
            raise ValidationError(
                "this schedule has no eigen-directions: they come from the solver, "
                "and this one was built another way"
            )
        return self.directions[0].weights

    #: The covariance the moments were computed against, carried so
    #: :meth:`worst_interval_variance` does not need the problem again.
    covariance_used: tuple[tuple[float, ...], ...] = ()


def basket_moments(
    problem: BasketProblem, trades: Sequence[Sequence[float]]
) -> tuple[float, float]:
    """Expected cost and variance of an arbitrary basket schedule.

    By direct summation over the schedule, exactly as the single-asset case does,
    rather than from the closed form the solution is derived through. That is the
    point: the solver and this share no arithmetic, so the solver being optimal is
    checkable rather than assumed.

    Expected cost is the temporary impact of each interval's trade plus the
    permanent impact of the whole liquidation. The permanent term is
    path-independent for a complete liquidation — every share is sold exactly once
    and the drift it leaves behind does not care when — which is why it can sit
    outside the optimisation and why it cancels out of any comparison between two
    schedules of the same basket.
    """
    if len(trades) != problem.periods:
        raise ValidationError(f"expected {problem.periods} intervals of trades, got {len(trades)}")
    holdings = problem.holdings_vector()
    temporary = problem.temporary_matrix()
    permanent = problem.permanent_matrix()
    covariance = problem.covariance_matrix()
    tau = problem.tau

    effective = temporary - 0.5 * tau * permanent
    total = np.zeros(problem.assets)
    cost = 0.0
    variance = 0.0
    remaining = holdings.copy()
    for index, row in enumerate(trades):
        trade = np.asarray(row, dtype=np.float64)
        if trade.shape != (problem.assets,):
            raise ValidationError(
                f"interval {index} has {trade.shape} trades for {problem.assets} assets"
            )
        cost += float(trade @ effective @ trade) / tau
        total += trade
        remaining = remaining - trade
        variance += float(remaining @ covariance @ remaining)
    variance *= tau

    residual = float(np.max(np.abs(total - holdings)))
    scale = float(np.max(np.abs(holdings))) or 1.0
    if residual > 1e-6 * scale:
        raise ValidationError(
            f"the trades do not liquidate the basket: the largest asset is left "
            f"with {residual:.6g} shares of {scale:.6g}"
        )
    cost += 0.5 * float(holdings @ permanent @ holdings)
    return cost, variance


def basket_trajectory(problem: BasketProblem, risk_aversion: float) -> BasketTrajectory:
    """The cost-and-risk optimal liquidation of a basket.

    Exact, through the transformation in the module docstring. The eigen-directions
    and their half-lives come back with the schedule, because a per-asset schedule
    alone does not show *why* it has the shape it has and the directions do.
    """
    if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
        raise ValidationError(
            f"risk aversion must be non-negative and finite, got {risk_aversion!r}"
        )
    effective = problem.effective_temporary()
    forward, inverse = _inverse_square_roots(effective, "the effective temporary impact matrix")
    covariance = problem.covariance_matrix()
    transformed = inverse @ covariance @ inverse
    # Symmetrised again: the product of three symmetric matrices is symmetric in
    # exact arithmetic and off by rounding in floating point, and `eigh` reads only
    # the lower triangle, so an unsymmetrised argument quietly discards half of it.
    transformed = (transformed + transformed.T) / 2.0
    eigenvalues, eigenvectors = _eigendecompose(
        transformed, "the transformed risk matrix", require_definite=False
    )

    tau = problem.tau
    holdings = problem.holdings_vector()
    start = forward @ holdings
    coordinates = eigenvectors.T @ start

    kappas = []
    for value in eigenvalues:
        mu = max(float(value), 0.0)
        z = 0.5 * risk_aversion * mu * tau * tau
        kappas.append(math.log1p(z + math.sqrt(z * (z + 2.0))) / tau)

    times = problem.times()
    rows: list[tuple[float, ...]] = []
    for time in times:
        scaled = []
        for index, kappa in enumerate(kappas):
            scaled.append(coordinates[index] * _sinh_ratio(kappa, problem.horizon, time))
        vector = inverse @ (eigenvectors @ np.asarray(scaled, dtype=np.float64))
        rows.append(tuple(float(value) for value in vector))
    # The last row is zero by construction; set it exactly so a downstream sum
    # telescopes rather than leaving a residue the size of the rounding.
    rows[-1] = tuple(0.0 for _ in range(problem.assets))
    # And the first is the holdings exactly, for the same reason: the transform
    # and its inverse are not exact inverses in floating point.
    rows[0] = tuple(float(value) for value in holdings)

    trades: list[tuple[float, ...]] = []
    for before, after in itertools.pairwise(rows):
        trades.append(tuple(b - a for b, a in zip(before, after, strict=True)))

    cost, variance = basket_moments(problem, trades)
    directions = []
    for index, kappa in enumerate(kappas):
        # Back into share space, and normalised, so the direction reads as a
        # portfolio rather than as a coordinate in the transformed basis.
        share_space = inverse @ eigenvectors[:, index]
        norm = float(np.linalg.norm(share_space)) or 1.0
        directions.append(
            Direction(
                risk_per_impact=max(float(eigenvalues[index]), 0.0),
                kappa=kappa,
                initial=float(coordinates[index]),
                weights=tuple(float(value / norm) for value in share_space),
            )
        )
    return BasketTrajectory(
        times=tuple(times),
        holdings=tuple(rows),
        trades=tuple(trades),
        expected_cost=cost,
        variance=variance,
        risk_aversion=risk_aversion,
        directions=tuple(sorted(directions, key=lambda one: -one.risk_per_impact)),
        names=problem.labels(),
        covariance_used=tuple(
            tuple(float(value) for value in row) for row in problem.covariance_matrix()
        ),
    )


def _sinh_ratio(kappa: float, horizon: float, time: float) -> float:
    """``sinh(kappa (T - t)) / sinh(kappa T)``, without overflowing.

    At ``kappa`` of zero this is the straight line ``(T - t) / T`` and the ratio
    is ``0 / 0``, so the limit is taken rather than computed. At large ``kappa T``
    both terms overflow, so the ratio is evaluated as exponentials of the
    difference, which is what it always was.
    """
    if kappa <= 0.0:
        return (horizon - time) / horizon
    remaining = horizon - time
    if kappa * horizon > 350.0:
        # sinh(a)/sinh(b) -> exp(a - b) * (1 - exp(-2a)) / (1 - exp(-2b))
        return (
            math.exp(kappa * (remaining - horizon))
            * (-math.expm1(-2.0 * kappa * remaining))
            / (-math.expm1(-2.0 * kappa * horizon))
        )
    return math.sinh(kappa * remaining) / math.sinh(kappa * horizon)


def basket_frontier(
    problem: BasketProblem, risk_aversions: Iterable[float]
) -> list[BasketTrajectory]:
    """One solution per risk aversion, in the order given."""
    return [basket_trajectory(problem, value) for value in risk_aversions]


def independent_trajectories(problem: BasketProblem, risk_aversion: float) -> BasketTrajectory:
    """The same basket solved one asset at a time, for comparison.

    This is what a trader who has a single-asset solver and a list of names does,
    and it is the thing the joint solution should be measured against. Each leg is
    given its own volatility from the diagonal of the covariance matrix and its own
    impact from the diagonals of the impact matrices, and the cross terms are
    dropped — which is exactly the information the joint solution uses.

    The result is scored against the *full* matrices, so its cost and variance are
    what the schedule would really have incurred rather than what its own
    assumptions predicted. A comparison that let each schedule be judged by its own
    model would be no comparison at all.
    """
    temporary = problem.temporary_matrix()
    permanent = problem.permanent_matrix()
    covariance = problem.covariance_matrix()
    columns: list[list[float]] = []
    for index in range(problem.assets):
        quantity = problem.holdings[index]
        if quantity == 0.0:
            columns.append([0.0] * problem.periods)
            continue
        single = ExecutionProblem(
            quantity=abs(quantity),
            horizon=problem.horizon,
            periods=problem.periods,
            volatility=math.sqrt(float(covariance[index][index])),
            impact=LinearImpact(
                gamma=float(permanent[index][index]),
                eta=float(temporary[index][index]),
            ),
        )
        solved = optimal_trajectory(single, risk_aversion)
        sign = 1.0 if quantity > 0.0 else -1.0
        columns.append([sign * value for value in solved.trades])
    trades = tuple(
        tuple(columns[asset][period] for asset in range(problem.assets))
        for period in range(problem.periods)
    )
    cost, variance = basket_moments(problem, trades)
    holdings_rows = [tuple(problem.holdings)]
    for row in trades:
        previous = holdings_rows[-1]
        holdings_rows.append(tuple(p - t for p, t in zip(previous, row, strict=True)))
    holdings_rows[-1] = tuple(0.0 for _ in range(problem.assets))
    return BasketTrajectory(
        times=tuple(problem.times()),
        holdings=tuple(holdings_rows),
        trades=trades,
        expected_cost=cost,
        variance=variance,
        risk_aversion=risk_aversion,
        directions=(),
        names=problem.labels(),
        covariance_used=tuple(tuple(float(value) for value in row) for row in covariance),
    )


def hedge_direction(problem: BasketProblem) -> tuple[float, ...]:
    """The riskiest direction the basket starts flat in.

    A schedule cannot be blamed for the exposure it inherits, only for the
    exposure it creates — and the exposure it creates lives in the directions the
    basket had none in to begin with. So: maximise ``w' Sigma w`` over unit
    vectors ``w`` orthogonal to the initial holdings, which is the top eigenvector
    of the covariance projected onto that orthogonal complement.

    On a dollar-neutral long/short pair the complement of ``(1, -1)`` is spanned by
    ``(1, 1)``, so this returns the net of the two legs and the metric built on it
    is "how much of the pair does the schedule ever leave unhedged". That it comes
    out as the obvious thing on the obvious case is the point of deriving it rather
    than special-casing it: a book with six legs and two hedges has the same
    question and no obvious answer.
    """
    holdings = problem.holdings_vector()
    covariance = problem.covariance_matrix()
    size = problem.assets
    if size < 2:
        raise ValidationError(
            "a one-asset basket has no direction orthogonal to its own holdings, "
            "so there is no exposure it could create rather than inherit"
        )
    norm = float(np.linalg.norm(holdings))
    if norm == 0.0:
        raise ValidationError("a basket with nothing in it starts flat everywhere")
    unit = holdings / norm
    projector = np.eye(size) - np.outer(unit, unit)
    projected = projector @ covariance @ projector
    projected = (projected + projected.T) / 2.0
    _, vectors = np.linalg.eigh(projected)
    direction = vectors[:, -1]
    # Re-project, since the eigenvector of a numerically projected matrix is only
    # orthogonal to the holdings up to rounding, and sign it so the result is
    # reproducible rather than up to the sign `eigh` happens to return.
    direction = projector @ direction
    length = float(np.linalg.norm(direction)) or 1.0
    direction = direction / length
    if float(direction[int(np.argmax(np.abs(direction)))]) < 0.0:
        direction = -direction
    return tuple(float(value) for value in direction)


@dataclass(frozen=True)
class BasketComparison:
    """The joint solution against the independent one, scored the same way."""

    problem: BasketProblem
    joint: BasketTrajectory
    independent: BasketTrajectory

    @property
    def objective_saving(self) -> float:
        """Fraction of the independent objective the joint solution saves.

        Both objectives include the permanent impact term, which is identical for
        any complete liquidation of the same basket — so this understates the
        difference in the part that is actually being optimised. That is the
        conservative direction and it is the honest one to quote, since the
        permanent cost is real money the trader pays either way.
        """
        reference = self.independent.objective
        if reference == 0.0:
            raise ValidationError(
                "the independent objective is zero, so a relative saving is a "
                "division by zero; compare the levels instead"
            )
        return (reference - self.joint.objective) / reference

    @property
    def peak_risk_ratio(self) -> float:
        """Independent peak interval variance over the joint solution's.

        **Below one, and that is not a defect.** Measured on a correlated pair, the
        joint solution carries 1.19 times the independent one's peak interval
        variance in the symmetric case and 1.10 in the illiquid-leg case. It is
        *deliberately* riskier moment to moment: a hedged pair is cheap to hold, so
        the optimal schedule holds it longer and pays less impact. A reader
        expecting the joint solution to be the safer one has the trade backwards —
        it is the cheaper one at the same risk aversion, and the risk aversion is
        where the preference lives.
        """
        joint = self.joint.peak_interval_variance()
        if joint == 0.0:
            raise ValidationError(
                "the joint solution carries no interval risk after the start, so "
                "the ratio is a division by zero"
            )
        return self.independent.peak_interval_variance() / joint

    def exposure_ratio(self, weights: Floats | None = None) -> float:
        """How much more the independent schedule leaves exposed, at its worst.

        Measured along :func:`hedge_direction` by default — the riskiest direction
        the basket starts flat in — because that is the exposure a schedule
        *creates* rather than inherits. Pass ``weights`` for a direction of your
        own.

        This is where the independent solution's mistake shows, and it is an order
        of magnitude larger than the difference in the totals. On a correlated pair
        whose short leg is four times as expensive to trade, the independent
        schedule reaches a net exposure of 18,961 shares of a 100,000-share pair
        against the joint solution's 2,025 — a factor of 9.4, against an objective
        difference of 12.05%. It works each leg at the pace that leg's own
        liquidity justifies, so the liquid one finishes first and the book is left
        outright in the middle of the trade.
        """
        direction = hedge_direction(self.problem) if weights is None else weights
        joint = self.joint.peak_exposure(direction)
        # Relative, not a test against zero. Two schedules that both keep a hedge
        # exactly still leave exposures of order `eps * size` behind, so an
        # absolute guard does not fire and the ratio comes back as one rounding
        # error over another — 0.43 on a symmetric pair where both are flat, which
        # reads as a finding and is noise.
        size = max(abs(value) for value in self.problem.holdings)
        if joint <= 1e-9 * size:
            raise ValidationError(
                f"the joint schedule's largest exposure along this direction is "
                f"{joint:.3g} against a basket of {size:.3g} shares, which is "
                f"rounding rather than exposure: on a symmetric pair both "
                f"schedules keep the hedge exactly and the ratio is one rounding "
                f"error over another. Compare the objectives instead."
            )
        return self.independent.peak_exposure(direction) / joint


def compare_to_independent(problem: BasketProblem, risk_aversion: float) -> BasketComparison:
    """Solve the basket both ways and score both against the full matrices."""
    return BasketComparison(
        problem=problem,
        joint=basket_trajectory(problem, risk_aversion),
        independent=independent_trajectories(problem, risk_aversion),
    )
