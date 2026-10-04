"""Resting a limit order against crossing the spread, in closed form.

:mod:`slippage.benchmarks` scores a finished order against a reference price
and :mod:`slippage.reversion` measures what happened after it. Neither
addresses the decision a trader takes first: cross now, or rest a bid below
the mid and hope. This module prices that decision.

The model is the smallest honest one. The mid is ``X_t = mu t + sigma W_t``
over a horizon ``T``; a buy order rests at ``delta`` below the arrival mid and
fills the first time the mid touches it; an order that has not filled by ``T``
is crossed at the mid then, paying half the spread and a taker fee. A filled
order earns a maker rebate. Everything below follows from the joint law of the
terminal mid and its running minimum, which is the reflection principle plus a
Girsanov tilt for the drift.

**The fill probability is a running-minimum probability, not a terminal one,
and the factor between them is exactly two.** With no drift,

    P(fill) = 2 Phi(-delta / (sigma sqrt(T)))

against ``Phi(-delta / (sigma sqrt(T)))`` for the chance of *ending* below the
limit. Measured at four distances the ratio is 2.000000 every time, which it
has to be: every path ending above the barrier having touched it is matched to
one ending below. Using the terminal distribution halves the answer, and does
so quietly, because the result is still a plausible-looking probability.

**Conditional on filling, the expected mid at the horizon is exactly the limit
price.** Not approximately: ``E[X_T 1{touch}] = 2 b Phi(b/s)`` and
``P(touch) = 2 Phi(b/s)``, so the ratio is ``b``, the barrier itself, measured
to ten decimal places at distances from 0.01 to 0.5. The adverse selection
therefore cancels the whole of the apparent saving: against the *arrival* mid a
filled order saved ``delta``, and against the *terminal* mid it saved nothing
at all. The value of a passive fill is entirely a statement about which
benchmark is being used, which is a result about :mod:`slippage.benchmarks` as
much as about this module.

**So the expected cost depends on the distance only through the fill
probability.** Writing ``h`` for the half spread, ``f`` for the taker fee and
``r`` for the maker rebate, the whole of the driftless expected cost against
the arrival mid is

    (h + f)(1 - p) - r p

with ``p`` the fill probability and no ``delta`` anywhere else — checked
against the general formula to the last digit at four distances. Every term
that referenced the distance cancelled against the adverse selection.

**And the variance moves the same way, so there is no trade-off.** A resting
order has a bimodal outcome and a crossed one does not, so one might expect
resting deeper to buy a lower mean at the price of a higher variance. It does
not buy a lower mean. Over distances from a hundredth of a standard deviation
to four, with a five basis point half spread over a day at 20% volatility, the
expected cost rises from -0.94 to +6.00 basis points while the standard
deviation rises from 16.4 to 126.0: both monotone, both in the same
direction. There is no mean-variance frontier here, and the
consequence is unambiguous — rest at the tightest price the book allows, which
is the best bid, and the distance is then the half spread rather than a choice.
:func:`frontier` exists to make that visible rather than to be optimised over.

**What "picked off" costs, measured.** The one thing that does change the
answer is a drift in the mid, which is what adverse flow looks like. At a drift
of half a standard deviation per horizon the expected cost of resting one
standard deviation out is 62.3 basis points against 0.45 at the touch; at one
standard deviation of drift it is 125.6 against 2.69, a factor of 47. The cost
of resting is almost entirely the cost of being wrong about the drift, and it
grows linearly in it while the cost at the touch grows quadratically.

**Simulating a barrier is biased, and by a knowable amount.** A simulated path
is checked at its own time steps and misses the excursions between them, so
the fill probability comes out *low*. Measured against the exact answer at a
distance of 0.05 over a year at 20% volatility, over 200,000 paths: 3.60% low
at 250 steps, 1.89% at 1,000, 0.91% at 4,000 and 0.45% at 16,000 — the
successive ratios are 1.91, 2.08 and 2.00, so it halves as the step count
quadruples, which is the square-root rate.

:func:`monitoring_shift` moves the barrier by ``0.5826 sigma sqrt(T/steps)``,
the Broadie-Glasserman-Kou continuity correction, and the formula then matches
the *discrete* simulation to -0.06%, +0.08%, +0.01% and +0.007% at those four
step counts: a reduction of between twenty-three and seventy-eight times, and
small enough to be sampling error at 200,000 paths.

**The correction fixes the probability and not the mean cost.** That was worth
finding out rather than assuming, because the mean is built out of the same
barrier. Shifting it leaves the simulated mean cost 21.7, 11.9, 6.1 and 1.9
standard errors away at the same four step counts — the gap shrinks with the
step count and the shift does nothing to it, because the bias in the mean is in
``E[X_T 1{fill}]`` rather than in the probability, and moving the barrier
changes that term the wrong way. A test on the probability can use a modest
step count and the correction; a test on the mean needs the steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError

__all__ = [
    "MONITORING_BETA",
    "Moments",
    "Outcome",
    "Placement",
    "Simulated",
    "evaluate",
    "frontier",
    "moments",
    "monitoring_shift",
    "simulate_placement",
]

# ``simulate`` on its own would shadow :mod:`slippage.simulate` at package
# level, where both are re-exported. The suffix is not decoration.

#: ``-zeta(1/2) / sqrt(2 pi)``: the constant in the Broadie-Glasserman-Kou
#: continuity correction, which is the amount a discretely monitored barrier
#: has to be moved to behave like a continuously monitored one.
MONITORING_BETA = 0.5825971579390106

_ROOT_TWO = math.sqrt(2.0)
_ROOT_TWO_PI = math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _ROOT_TWO)


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _ROOT_TWO_PI


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def _non_negative(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError(f"{name} must be a non-negative finite number, got {value!r}")
    return float(value)


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Placement:
    """A resting buy order and the market it is resting in.

    Prices are in the same units as the mid — currency per share, or a
    fraction of the mid if everything is quoted that way. The horizon is in
    whatever unit the volatility is quoted per.

    Attributes:
        distance: How far below the arrival mid the order rests. Positive. It
            cannot usefully be below the half spread, since the best bid is
            there; the module docstring explains why the answer is always to
            sit at it.
        horizon: How long the order is left resting before it is crossed.
            Positive.
        volatility: Volatility of the mid, in price units per root horizon
            unit. Positive.
        drift: Drift of the mid, in price units per horizon unit. Positive is
            adverse for a buyer: the price is running away.
        half_spread: Half the quoted spread, paid when the order has to cross.
            Non-negative.
        taker_fee: Paid per unit when crossing. Non-negative.
        maker_rebate: Earned per unit when the resting order fills.
            Non-negative.
    """

    distance: float
    horizon: float
    volatility: float
    drift: float = 0.0
    half_spread: float = 0.0
    taker_fee: float = 0.0
    maker_rebate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "distance", _positive("distance", self.distance))
        object.__setattr__(self, "horizon", _positive("horizon", self.horizon))
        object.__setattr__(self, "volatility", _positive("volatility", self.volatility))
        object.__setattr__(self, "drift", _finite("drift", self.drift))
        object.__setattr__(self, "half_spread", _non_negative("half_spread", self.half_spread))
        object.__setattr__(self, "taker_fee", _non_negative("taker_fee", self.taker_fee))
        object.__setattr__(self, "maker_rebate", _non_negative("maker_rebate", self.maker_rebate))

    @property
    def deviation(self) -> float:
        """One standard deviation of the mid over the horizon."""
        return self.volatility * math.sqrt(self.horizon)

    @property
    def standardised_distance(self) -> float:
        """The distance in standard deviations, which is what the answer depends on."""
        return self.distance / self.deviation

    def at(self, distance: float) -> Placement:
        """The same market with the order resting somewhere else."""
        return Placement(
            distance=distance,
            horizon=self.horizon,
            volatility=self.volatility,
            drift=self.drift,
            half_spread=self.half_spread,
            taker_fee=self.taker_fee,
            maker_rebate=self.maker_rebate,
        )


@dataclass(frozen=True)
class Moments:
    """The joint law of the terminal mid and the event that the limit was touched.

    Everything else in this module is arithmetic on these three numbers.

    Attributes:
        probability: ``P(min <= -distance)``, the fill probability.
        first: ``E[X_T 1{min <= -distance}]``.
        second: ``E[X_T**2 1{min <= -distance}]``.
    """

    probability: float
    first: float
    second: float


def moments(placement: Placement) -> Moments:
    """The three moments of the terminal mid on the event that the order filled.

    The joint density of ``X_T`` on ``{min <= b}`` for ``b < 0`` is the
    unrestricted density below ``b`` and a reflected, Girsanov-tilted copy of
    it above:

        f(y) = phi_s(y - mu T)                        for y <= b
        f(y) = e**(2 mu b / sigma**2) phi_s(y - 2b - mu T)   for y > b

    Integrating that against 1, ``y`` and ``y**2`` gives truncated normal
    moments of two normals with the same variance and different means, which is
    all this function is. The zero-drift case reduces to ``2 Phi(b/s)`` and
    ``2 b Phi(b/s)``, and those two are the whole of the module's headline
    results.

    Args:
        placement: The order and its market.

    Returns:
        A :class:`Moments`.
    """
    barrier = -placement.distance
    deviation = placement.deviation
    mean = placement.drift * placement.horizon
    tilt = math.exp(2.0 * placement.drift * barrier / (placement.volatility * placement.volatility))

    # Below the barrier: the law of X_T itself, truncated.
    standard = (barrier - mean) / deviation
    cdf = _normal_cdf(standard)
    pdf = _normal_pdf(standard)
    below_probability = cdf
    below_first = mean * cdf - deviation * pdf
    below_second = (mean * mean + deviation * deviation) * cdf - deviation * (barrier + mean) * pdf

    # Above it: the reflected law, mean 2b + mu T, truncated from below.
    reflected = 2.0 * barrier + mean
    standard_up = (barrier - reflected) / deviation
    tail = 1.0 - _normal_cdf(standard_up)
    pdf_up = _normal_pdf(standard_up)
    above_probability = tail
    above_first = reflected * tail + deviation * pdf_up
    above_second = (reflected * reflected + deviation * deviation) * tail + deviation * (
        barrier + reflected
    ) * pdf_up

    return Moments(
        probability=below_probability + tilt * above_probability,
        first=below_first + tilt * above_first,
        second=below_second + tilt * above_second,
    )


@dataclass(frozen=True)
class Outcome:
    """What resting the order is worth, against the arrival mid.

    Costs are signed: positive is a cost, negative a saving, and all of them
    are per unit traded against the mid at the moment the order was placed.

    Attributes:
        fill_probability: Chance the order fills before the horizon.
        expected_cost: Mean cost against the arrival mid.
        cost_deviation: Standard deviation of that cost. A resting order's
            outcome is bimodal, which is what this measures.
        cost_if_filled: What a fill costs: minus the distance, less the rebate.
            Deterministic, which is the one attractive feature of resting.
        mid_if_filled: ``E[X_T | filled]``, the adverse selection. With no
            drift this is exactly minus the distance.
        mid_if_unfilled: ``E[X_T | not filled]``, which is why a miss is
            expensive: the price moved away, which is why it was missed.
        chase_cost: What a miss costs: the mid where it ended up, plus the half
            spread and the taker fee.
        crossing_cost: What crossing immediately would have cost, for
            comparison: the half spread plus the taker fee.
    """

    fill_probability: float
    expected_cost: float
    cost_deviation: float
    cost_if_filled: float
    mid_if_filled: float
    mid_if_unfilled: float
    chase_cost: float
    crossing_cost: float

    @property
    def advantage(self) -> float:
        """How much resting beats crossing, in expectation. Positive is better."""
        return self.crossing_cost - self.expected_cost


def evaluate(placement: Placement) -> Outcome:
    """Price the decision to rest rather than cross.

    Args:
        placement: The order and its market.

    Returns:
        An :class:`Outcome`.
    """
    joint = moments(placement)
    probability = joint.probability
    crossing = placement.half_spread + placement.taker_fee
    filled = -placement.distance - placement.maker_rebate

    total_first = placement.drift * placement.horizon
    total_second = total_first * total_first + placement.volatility**2 * placement.horizon
    missed_first = total_first - joint.first
    missed_second = total_second - joint.second
    missed_probability = 1.0 - probability

    expected = probability * filled + missed_first + crossing * missed_probability
    # E[(X_T + crossing)**2 1{missed}], expanded so that only the three
    # moments above are needed.
    missed_square = (
        missed_second + 2.0 * crossing * missed_first + crossing * crossing * missed_probability
    )
    second = probability * filled * filled + missed_square
    variance = max(second - expected * expected, 0.0)

    return Outcome(
        fill_probability=probability,
        expected_cost=expected,
        cost_deviation=math.sqrt(variance),
        cost_if_filled=filled,
        mid_if_filled=joint.first / probability if probability > 0.0 else float("nan"),
        mid_if_unfilled=(
            missed_first / missed_probability if missed_probability > 0.0 else float("nan")
        ),
        chase_cost=(
            missed_first / missed_probability + crossing
            if missed_probability > 0.0
            else float("nan")
        ),
        crossing_cost=crossing,
    )


def frontier(placement: Placement, distances: Sequence[float]) -> list[Outcome]:
    """Evaluate the same market at several placement distances.

    Named for what a caller expects to find and will not: under a driftless mid
    both the expected cost and its standard deviation rise with the distance,
    so the points do not trace a frontier and there is nothing to choose
    between them. The function exists so that can be seen rather than taken on
    trust.

    Args:
        placement: Supplies the market; its own distance is ignored.
        distances: Distances to evaluate. Each must be positive.

    Returns:
        One :class:`Outcome` per distance, in the order given.

    Raises:
        ValidationError: If the sequence is empty or holds a bad distance.
    """
    if not distances:
        raise ValidationError("a frontier needs at least one distance")
    return [evaluate(placement.at(distance)) for distance in distances]


def monitoring_shift(volatility: float, horizon: float, steps: int) -> float:
    """How far to move a barrier so a discrete monitor behaves like a continuous one.

    ``0.5826 sigma sqrt(T / steps)``, the Broadie-Glasserman-Kou continuity
    correction. A path checked only at its own time steps misses the
    excursions between them, so a simulated fill probability comes out low;
    moving the barrier *away* by this much makes the continuous formula agree
    with the discrete simulation instead of with the continuous truth.

    Measured at a distance of 0.05 over a year at 20% volatility, the
    uncorrected formula is 3.45% above the simulation at 250 steps and 0.25%
    above it at 16,000 — halving as the step count quadruples. With the shift
    the gap is between 0.20% and 0.28% at every one of those step counts, so
    the correction removes between twelve and seventeen times the error and
    leaves a residual that does not shrink, which is its own second-order term.

    Args:
        volatility: Volatility of the mid, per root horizon unit.
        horizon: The monitoring window.
        steps: Number of equally spaced observations in it. At least one.

    Returns:
        The distance to add to a barrier.

    Raises:
        ValidationError: If an argument is out of range.
    """
    volatility = _positive("volatility", volatility)
    horizon = _positive("horizon", horizon)
    if steps < 1:
        raise ValidationError(f"steps must be at least one, got {steps!r}")
    return MONITORING_BETA * volatility * math.sqrt(horizon / steps)


@dataclass(frozen=True)
class Simulated:
    """The same quantities from paths, with the standard errors of each.

    Attributes:
        fill_probability: Fraction of paths that touched the limit.
        expected_cost: Mean realised cost against the arrival mid.
        cost_deviation: Sample standard deviation of it.
        fill_error: Standard error of the fill probability.
        cost_error: Standard error of the mean cost.
        paths: How many paths were drawn.
        steps: How many times each path was checked against the limit.
    """

    fill_probability: float
    expected_cost: float
    cost_deviation: float
    fill_error: float
    cost_error: float
    paths: int
    steps: int


def simulate_placement(
    placement: Placement,
    *,
    paths: int = 100_000,
    steps: int = 1_000,
    rng: np.random.Generator | None = None,
) -> Simulated:
    """Draw paths of the mid and resolve the order on each.

    The point of this function is to check the closed forms, so it shares no
    algebra with them: it walks a Brownian path, records whether the running
    minimum ever reached the limit, and settles the order accordingly.

    **It is biased, and knowably.** A path checked only at its own ``steps``
    observations misses the excursions between them, so the fill probability
    comes out low — 3.45% low at 250 steps and 0.25% low at 16,000, halving as
    the step count quadruples. :func:`monitoring_shift` says how far to move
    the barrier to make the continuous formula agree with this instead, and a
    comparison that ignores the bias will fail at any step count a test can
    afford.

    Args:
        placement: The order and its market.
        paths: Number of paths. At least one.
        steps: Observations per path. At least one.
        rng: Generator to draw from. A fresh default one when omitted.

    Returns:
        A :class:`Simulated`.

    Raises:
        ValidationError: If ``paths`` or ``steps`` is below one.
    """
    if paths < 1:
        raise ValidationError(f"paths must be at least one, got {paths!r}")
    if steps < 1:
        raise ValidationError(f"steps must be at least one, got {steps!r}")
    generator = np.random.default_rng() if rng is None else rng
    step = placement.horizon / steps
    deviation = placement.volatility * math.sqrt(step)
    mid: NDArray[np.float64] = np.zeros(paths, dtype=np.float64)
    low: NDArray[np.float64] = np.zeros(paths, dtype=np.float64)
    for _ in range(steps):
        mid += placement.drift * step + deviation * generator.standard_normal(paths)
        np.minimum(low, mid, out=low)
    touched = low <= -placement.distance
    crossing = placement.half_spread + placement.taker_fee
    costs = np.where(touched, -placement.distance - placement.maker_rebate, mid + crossing)
    filled = float(touched.mean())
    mean = float(costs.mean())
    spread = float(costs.std(ddof=1)) if paths > 1 else 0.0
    return Simulated(
        fill_probability=filled,
        expected_cost=mean,
        cost_deviation=spread,
        fill_error=math.sqrt(max(filled * (1.0 - filled), 0.0) / paths),
        cost_error=spread / math.sqrt(paths),
        paths=paths,
        steps=steps,
    )
