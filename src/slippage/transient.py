"""Impact that decays at a rate, rather than instantly or never.

:mod:`slippage.reversion` measures how much of an order's impact came back and
over what half-life. Nothing else in this library can produce that curve. The
rate models in :mod:`slippage.impact` charge for the current trading *speed*, so
their temporary impact is gone the instant the speed is zero, and ``gamma`` never
decays at all. A desk that has measured a reverted fraction of 0.6 with a
half-life of eleven minutes has learned something no scheduler here could use.

This module is the missing middle. Impact is a **decay kernel**, or propagator:
each share traded pushes the price by ``G(0)`` and that push decays as ``G(lag)``
while the rest of the order works. ``G(inf)`` is the permanent floor, so the two
existing descriptions are the two limits of this one.

Four things are worth stating.

**The cost is a quadratic form, not a simulation.** Slice ``k`` pays the impact
already in the price from every earlier slice, plus half of its own::

    cost = sum_k n_k [ sum_{j<k} n_j G(t_k - t_j) + n_k G(0) / 2 ]
         = n' K n / 2,     K_ij = G(|t_i - t_j|)

The second line is the first, rearranged. Having it as a symmetric form is what
makes everything below closed form instead of iterative, and the tests check the
two against each other by walking the price path trade by trade.

**The cost-minimising schedule is a linear solve.** Minimising ``n'Kn/2`` subject
to ``sum n = X`` gives ``n* = X K^-1 1 / (1' K^-1 1)`` and a cost of
``X^2 / (2 * 1' K^-1 1)``. No optimiser, no starting values, no convergence to
report. For an exponential kernel the answer comes out as the shape Obizhaeva and
Wang (2013) derive in continuous time: a block at the start, a constant rate
through the middle, a block at the end. That is not imposed anywhere here -- it
falls out of the solve, which is the most convincing form the result can take.

**A kernel that is not positive definite pays you to trade.** If ``K`` is
negative on some zero-sum direction then there is a round trip -- buy some, sell
the same amount back -- whose impact cost is *negative*, and an optimiser handed
that model finds profit in its own market impact. So it is checked rather than
assumed, and :func:`manipulation_round_trip` returns the offending trip rather
than a boolean, because "your kernel is inadmissible" is much less use than the
twelve numbers that break it.

Being a decreasing function of lag is **not** sufficient. A kernel that falls
slowly and then drops off a shoulder is strictly decreasing at every lag and
still admits a round trip whose cost is ``-2.35``: twelve slices, the largest of
them one share, in a kernel whose instantaneous impact is one. Complete
monotonicity is the property that works, because it makes the kernel a mixture of
exponentials and every exponential is positive definite on its own.

**The gain from scheduling well is not monotone in resilience**, and both limits
are the reason. As resilience goes to infinity the kernel becomes diagonal, the
cost becomes ``sum n_k^2`` and the uniform schedule is already optimal. As it goes
to zero the kernel becomes constant, the cost becomes ``X^2 G(0) / 2`` and *every*
schedule is optimal. The saving therefore peaks somewhere in between: on a
pure-transient kernel over eight slices it is worth 7.04% at a half-life of 35%
of the horizon, and the peak is a function of ``resilience * horizon`` alone --
identical at eight slices over eight minutes and eight over thirty. It does drift
with how finely the horizon is sliced, from 3.57% at four slices to 11.46% at
sixty-four, because the solution wants two instantaneous blocks and a finer grid
approximates them better.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .exceptions import CalibrationError, ValidationError
from .reversion import Decay

__all__ = [
    "DecayKernel",
    "ExponentialDecay",
    "PowerLawDecay",
    "RoundTrip",
    "TransientSchedule",
    "cost_matrix",
    "impact_path",
    "manipulation_round_trip",
    "optimal_transient_schedule",
    "residual_impact",
    "transient_cost",
]

FloatArray = NDArray[np.float64]

#: Relative tolerance for the positive-definiteness check. The cost matrix
#: restricted to zero-sum directions is singular by construction -- the
#: all-ones direction is projected out -- so its smallest eigenvalue is zero up
#: to rounding, and on a well-behaved power-law kernel that rounding comes out
#: *negative*, at around -1e-16 times the matrix scale. A test against zero
#: would reject it. The threshold is therefore relative to the largest
#: eigenvalue, which is the only scale in the problem.
DEFINITENESS_TOLERANCE = 1e-10


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be a positive finite number, got {value!r}")
    return value


def _non_negative(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError(f"{name} must be a non-negative finite number, got {value!r}")
    return value


@runtime_checkable
class DecayKernel(Protocol):
    """A propagator: what one share's push is worth ``lag`` later.

    :meth:`value` is the *total* including the permanent floor, so ``value(0)``
    is the instantaneous impact per share and ``permanent`` is where it is
    heading. A caller may supply their own, which is why nothing here trusts a
    kernel to be admissible.
    """

    @property
    def permanent(self) -> float:
        """The floor: impact per share that never decays."""

    def value(self, lag: float) -> float:
        """Impact per share remaining ``lag`` after the trade."""


@dataclass(frozen=True)
class ExponentialDecay:
    """``G(lag) = permanent + transient * exp(-resilience * lag)``.

    Obizhaeva and Wang's model: a limit order book that is eaten by a trade and
    refills at a constant rate. ``resilience`` is per unit of the same time
    ``tau`` is measured in, and mixing the two units is the easiest way to be
    wrong by orders of magnitude, so :attr:`half_life` is provided to state the
    parameter in a form that can be checked by eye against a mark-out curve.

    Parameters
    ----------
    permanent
        Price per share that never comes back.
    transient
        Price per share that decays. ``permanent + transient`` is the
        instantaneous impact of a share.
    resilience
        Decay rate per unit time. Larger is faster.
    """

    permanent: float
    transient: float
    resilience: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "permanent", _non_negative("permanent", self.permanent))
        object.__setattr__(self, "transient", _non_negative("transient", self.transient))
        object.__setattr__(self, "resilience", _positive("resilience", self.resilience))
        if self.permanent + self.transient <= 0.0:
            raise ValidationError(
                "permanent and transient impact are both zero, so this kernel says "
                "trading is free; that is not a model of anything"
            )

    @property
    def half_life(self) -> float:
        """Time for the decaying part to halve, in the kernel's own time unit."""
        return math.log(2.0) / self.resilience

    def value(self, lag: float) -> float:
        if lag < 0.0:
            raise ValidationError(f"lag must be non-negative, got {lag!r}")
        return self.permanent + self.transient * math.exp(-self.resilience * lag)

    @classmethod
    def from_measured_decay(
        cls,
        decay: Decay,
        *,
        quantity: float,
        price: float,
        unit: timedelta,
    ) -> ExponentialDecay:
        """Invert a mark-out curve from :meth:`~slippage.reversion.DecayProfile.decay`.

        The measured curve is in basis points of price for orders of a typical
        size, and a kernel is in price per share, so the inversion needs the size
        and the price it was measured at::

            permanent = asymptote_bps / 1e4 * price / quantity
            transient = amplitude_bps / 1e4 * price / quantity
            resilience = ln 2 / half_life

        ``unit`` is the time unit the resulting ``resilience`` should be quoted
        in, and it has to be the unit ``tau`` will be given in.

        **The resilience comes back exactly; the amplitude does not.** A mark-out
        is taken from the moment the order *completed*, by which time everything
        except the last slice has already been decaying. For an exponential
        kernel the horizon factors out of every term at once, so the measured
        curve is still a clean exponential with the right rate -- the half-life
        is recovered from a worked order as exactly as from a block. What shrinks
        is the amplitude, by

            (1/N) (1 - exp(-rho T)) / (1 - exp(-rho T/N))

        for a uniform schedule of ``N`` slices over a horizon ``T``. Over two
        half-lives that is 0.589 at eight slices and 0.541 in the limit of
        continuous trading, so this inversion reads ``transient`` back about 40%
        low on a worked order and is exact only for a single block. The factor is
        derived and checked against the model's own path in the tests.
        """
        quantity = _positive("quantity", quantity)
        price = _positive("price", price)
        seconds = unit.total_seconds()
        if seconds <= 0.0:
            raise ValidationError(f"unit must be a positive duration, got {unit!r}")
        life = decay.half_life.total_seconds() / seconds
        if life <= 0.0:
            raise CalibrationError(
                f"a half-life of {decay.half_life} is not positive in units of "
                f"{unit}, so it implies no decay rate"
            )
        scale = price / (1e4 * quantity)
        return cls(
            permanent=max(decay.asymptote_bps, 0.0) * scale,
            transient=max(decay.amplitude_bps, 0.0) * scale,
            resilience=math.log(2.0) / life,
        )


@dataclass(frozen=True)
class PowerLawDecay:
    """``G(lag) = permanent + transient * (1 + lag / scale) ** -exponent``.

    The propagator Bouchaud, Gefen, Potters and Wyart (2004) fit to order flow,
    where impact decays far more slowly than an exponential and the memory of a
    trade is never quite gone. It is completely monotone for any positive
    exponent, hence positive definite, hence admissible -- which is not a
    property to take for granted, and is checked on it like anything else.

    ``scale`` sets where the decay begins to bite and shares ``tau``'s time unit.
    """

    permanent: float
    transient: float
    exponent: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "permanent", _non_negative("permanent", self.permanent))
        object.__setattr__(self, "transient", _non_negative("transient", self.transient))
        object.__setattr__(self, "exponent", _positive("exponent", self.exponent))
        object.__setattr__(self, "scale", _positive("scale", self.scale))
        if self.permanent + self.transient <= 0.0:
            raise ValidationError(
                "permanent and transient impact are both zero, so this kernel says "
                "trading is free; that is not a model of anything"
            )

    def value(self, lag: float) -> float:
        if lag < 0.0:
            raise ValidationError(f"lag must be non-negative, got {lag!r}")
        # A float raised to a float widens to Any under the stubs, so the
        # narrowing happens here rather than at the call sites.
        shape = float((1.0 + lag / self.scale) ** -self.exponent)
        return self.permanent + self.transient * shape


def cost_matrix(kernel: DecayKernel, periods: int, tau: float) -> FloatArray:
    """``K_ij = G(|t_i - t_j|)`` for ``periods`` slices spaced ``tau`` apart.

    Symmetric by construction, because the cost of a schedule cannot depend on
    which of two slices is called earlier once the lag between them is fixed.
    """
    if periods < 1:
        raise ValidationError(f"periods must be at least 1, got {periods}")
    tau = _positive("tau", tau)
    lags = np.abs(np.arange(periods)[:, None] - np.arange(periods)[None, :]) * tau
    values = np.empty_like(lags, dtype=np.float64)
    for index, lag in enumerate(np.unique(lags)):
        values[lags == lag] = kernel.value(float(lag))
        del index
    return values


def _checked_trades(trades: Sequence[float]) -> FloatArray:
    array = np.asarray(trades, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValidationError("trades must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValidationError("every trade must be finite")
    return array


def transient_cost(kernel: DecayKernel, trades: Sequence[float], tau: float) -> float:
    """Expected impact cost of ``trades``, in currency: ``n' K n / 2``.

    Signed sizes are allowed, unlike the rate models in :mod:`slippage.impact`,
    because a round trip is the thing a kernel has to be tested against and a
    round trip needs one of each sign.
    """
    array = _checked_trades(trades)
    matrix = cost_matrix(kernel, array.size, tau)
    return float(0.5 * array @ matrix @ array)


def impact_path(kernel: DecayKernel, trades: Sequence[float], tau: float) -> list[float]:
    """Impact already in the price when each slice executes.

    One number per slice, the first of them zero because nothing has traded yet.
    This is what makes :func:`transient_cost` checkable against something other
    than itself: the cost is each slice times this, plus half of its own
    instantaneous impact.

    It deliberately stops at the last slice. A version of this returning one
    extra entry "for after the order" was wrong in a way that read as right --
    the extra entry came out one interval late, so it understated the impact at
    completion by exactly one period of decay, 16% on the grid it was checked
    on. What happens after the order is :func:`residual_impact`, which takes the
    horizons it is asked about rather than implying one.
    """
    array = _checked_trades(trades)
    matrix = cost_matrix(kernel, array.size, tau)
    return [float(matrix[index, :index] @ array[:index]) for index in range(array.size)]


def residual_impact(
    kernel: DecayKernel,
    trades: Sequence[float],
    tau: float,
    horizons: Sequence[float],
) -> list[float]:
    """Impact still in the price at each horizon *after the last slice*.

    Horizon zero is the impact at completion, including the last slice's own
    full instantaneous push. This is the mark-out curve the model predicts, so
    it is the quantity to compare against
    :meth:`~slippage.reversion.DecayProfile.decay` -- the loop this module
    exists to close.

    For an exponential kernel the curve is exactly
    ``permanent * X + A * exp(-resilience * h)``, because the horizon factors out
    of every term at once. So a mark-out measured on *any* schedule recovers the
    resilience exactly, and only the amplitude ``A`` depends on how the order was
    worked.
    """
    array = _checked_trades(trades)
    tau = _positive("tau", tau)
    last = (array.size - 1) * tau
    values: list[float] = []
    for horizon in horizons:
        if horizon < 0.0:
            raise ValidationError(
                f"a horizon of {horizon!r} is before the order finished; measure the "
                "path during the order with impact_path instead"
            )
        lags = last + horizon - np.arange(array.size) * tau
        values.append(
            float(
                sum(
                    float(size) * kernel.value(float(lag))
                    for size, lag in zip(array, lags, strict=True)
                )
            )
        )
    return values


@dataclass(frozen=True)
class RoundTrip:
    """A zero-sum schedule whose impact cost is negative.

    Attributes:
        trades: Signed sizes, summing to zero. Scaled so the largest is one, so
            the cost scales with the square of whatever size it is run at.
        cost: ``n' K n / 2`` for those trades. Negative, which is the problem.
        tau: The spacing it was found at.
    """

    trades: tuple[float, ...]
    cost: float
    tau: float


def manipulation_round_trip(kernel: DecayKernel, periods: int, tau: float) -> RoundTrip | None:
    """Find a profitable round trip, or return ``None`` if there is none.

    A round trip is a schedule summing to zero: buy some, sell the same back.
    Its cost is ``n'Kn/2``, so a profitable one exists exactly when ``K`` is
    negative somewhere on the subspace ``1'n = 0``. Projecting that subspace out
    and taking the smallest eigenvalue finds the cheapest direction there is, so
    this does not search -- it either produces the worst case or proves there
    isn't one.

    The threshold is relative to the largest eigenvalue rather than against
    zero. The projected matrix is singular by construction, so its smallest
    eigenvalue is zero up to rounding, and on a power-law kernel that rounding
    is *negative* at around -1e-16 of the scale. Against zero, this function
    would report a free lunch of 1e-16 on a perfectly good model.
    """
    matrix = cost_matrix(kernel, periods, tau)
    if periods < 2:
        return None
    projector = np.eye(periods) - np.ones((periods, periods)) / periods
    projected = projector @ matrix @ projector
    eigenvalues, vectors = np.linalg.eigh(projected)
    scale = float(np.max(np.abs(np.linalg.eigvalsh(matrix))))
    if eigenvalues[0] >= -DEFINITENESS_TOLERANCE * scale:
        return None
    direction = vectors[:, 0]
    # The eigenvector of the projected matrix already lies in the zero-sum
    # subspace up to rounding; subtracting the mean removes that rounding rather
    # than trusting it, because a trip that does not quite close is not a round
    # trip and its cost would include a real position.
    direction = direction - direction.mean()
    largest = float(np.max(np.abs(direction)))
    if largest <= 0.0:  # pragma: no cover - a zero eigenvector is not returned by eigh
        return None
    direction = direction / largest
    return RoundTrip(
        trades=tuple(float(one) for one in direction),
        cost=float(0.5 * direction @ matrix @ direction),
        tau=tau,
    )


@dataclass(frozen=True)
class TransientSchedule:
    """The cost-minimising schedule under a kernel, and what it is worth.

    Attributes:
        trades: The schedule, summing to the quantity asked for.
        cost: Its impact cost.
        uniform_cost: What trading at a constant rate would have cost.
        saving: ``1 - cost / uniform_cost``, the fraction saved. Zero at both
            extremes of resilience and largest in between.
        front_load: The first slice as a multiple of the uniform slice. Above
            one for any finite resilience, and the visible half of the
            block-rate-block shape.
    """

    trades: tuple[float, ...]
    cost: float
    uniform_cost: float
    saving: float
    front_load: float


def optimal_transient_schedule(
    kernel: DecayKernel, quantity: float, periods: int, tau: float
) -> TransientSchedule:
    """Minimise impact cost over schedules summing to ``quantity``.

    ``n* = X K^-1 1 / (1' K^-1 1)``, from the Lagrangian of a symmetric
    quadratic form under one linear constraint. Solved, not searched: there is
    nothing for an optimiser to add and a good deal for it to get wrong, and the
    tests compare this against SLSQP on the same problem, where the closed form
    is the better answer by 0.27% at low resilience because the optimiser has
    not converged.

    The schedule may contain slices larger than the quantity and, for a kernel
    with enough curvature, slices of the wrong sign -- the unconstrained
    solution is allowed to sell during a buy programme if the kernel says the
    price will be pushed back up in time to profit from it. That is a property of
    the model rather than of this function, so it is reported rather than
    clipped; :func:`~slippage.scheduling.solve_schedule` is where constraints
    belong.
    """
    quantity = _positive("quantity", quantity)
    if periods < 1:
        raise ValidationError(f"periods must be at least 1, got {periods}")
    trip = manipulation_round_trip(kernel, periods, tau)
    if trip is not None:
        raise ValidationError(
            f"this kernel admits a round trip of {periods} slices costing "
            f"{trip.cost!r}, so the cheapest schedule is unbounded: add the trip to "
            "any schedule and the cost falls. The trip is "
            f"{tuple(round(one, 4) for one in trip.trades)}"
        )
    matrix = cost_matrix(kernel, periods, tau)
    ones = np.ones(periods)
    weights = np.linalg.solve(matrix, ones)
    total = float(ones @ weights)
    if total <= 0.0:  # pragma: no cover - positive definiteness makes this positive
        raise ValidationError(
            f"1'K^-1 1 came out as {total!r}, which a positive definite kernel cannot produce"
        )
    trades = quantity * weights / total
    cost = float(0.5 * quantity**2 / total)
    uniform = np.full(periods, quantity / periods)
    uniform_cost = float(0.5 * uniform @ matrix @ uniform)
    return TransientSchedule(
        trades=tuple(float(one) for one in trades),
        cost=cost,
        uniform_cost=uniform_cost,
        saving=1.0 - cost / uniform_cost,
        front_load=float(trades[0]) / float(quantity / periods),
    )
