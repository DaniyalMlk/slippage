"""Splitting one order across venues, where the liquidity is random.

:mod:`slippage.placement` prices resting a limit order against crossing the
spread at *a* venue, and :mod:`slippage.volume` and :mod:`slippage.scheduling`
decide when to trade. Neither answers the first question a router in a
fragmented market has to answer: where to put the quantity. This module
answers it.

The model is the smallest one that still has the decision in it. A buyer wants
``target`` lots. They may cross for ``taker`` lots immediately at a known cost
per lot, and post ``posted[i]`` lots passively at each venue. At venue ``i``
the liquidity that arrives while the order rests is a random number of lots
``A_i``, drawn from a distribution the caller supplies, and the passive fill
there is ``min(posted[i], A_i)``. Venues are independent. Whatever is still
unfilled at the end is swept at a worse price than the opening cross, and an
over-execution is unwound at a cost.

Everything below is exact arithmetic on a lattice. Nothing here is simulated
and nothing is a normal approximation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .exceptions import ValidationError

__all__ = [
    "MAX_VENUES",
    "Allocation",
    "FillDistribution",
    "MarginalLot",
    "Outflow",
    "RoutingCost",
    "RoutingProblem",
    "Venue",
    "allocation_cost",
    "fill_distribution",
    "marginal_cross",
    "marginal_posted",
    "venue_fill",
]

# The exhaustive reference router enumerates the product of the venues'
# supports, so the number of venues is capped where that stops being a
# reference and becomes a hang. Real routing tables are shorter than this.
MAX_VENUES = 12


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value!r}")
    return value


@dataclass(frozen=True)
class Outflow:
    """How much liquidity a resting order at one venue gets to trade against.

    ``probabilities[k]`` is the probability that exactly ``k`` lots become
    available while the order rests, so the support is ``0`` to
    ``len(probabilities) - 1``. A discrete distribution rather than a
    continuous one because the quantity the cost depends on is a tail of a
    *sum* of these, and a sum of lattice variables lives on a lattice: giving
    it a density would be a modelling error dressed up as a convenience.

    The tails are summed from the top. ``P(A >= k)`` written as
    ``1 - P(A <= k - 1)`` is zero to the last bit once the true tail falls
    below ``1e-16``, and the tail at the far end of the support is exactly
    where a router's marginal decisions are taken.
    """

    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValidationError("an outflow needs at least one probability")
        for index, probability in enumerate(self.probabilities):
            _finite(f"probabilities[{index}]", probability)
            if probability < 0.0:
                raise ValidationError(
                    f"probabilities[{index}] must be non-negative, got {probability!r}"
                )
        total = math.fsum(self.probabilities)
        if abs(total - 1.0) > 1e-9:
            raise ValidationError(f"probabilities must sum to 1, got {total!r}")

    @property
    def support(self) -> int:
        """The largest number of lots that can arrive.

        Trailing zeros are ignored, so posting beyond this is provably
        pointless rather than merely unlikely to help: it is the natural cap
        on the search in :func:`route`.
        """
        for index in range(len(self.probabilities) - 1, -1, -1):
            if self.probabilities[index] > 0.0:
                return index
        return 0

    def tail(self, lots: int) -> float:
        """``P(A >= lots)``, summed from the top.

        Zero lots is certain, and anything past the support is impossible.
        """
        _non_negative_int("lots", lots)
        if lots == 0:
            return 1.0
        if lots >= len(self.probabilities):
            return 0.0
        return math.fsum(self.probabilities[lots:])

    def expected_fill(self, posted: int) -> float:
        """``E[min(posted, A)]``, by the layer-cake identity.

        ``E[min(L, A)] = sum_{k=1}^{L} P(A >= k)``: a sum of tails, each of
        which is itself summed from the top. Forming ``min`` pointwise and
        averaging gives the same answer where both are accurate and loses the
        far tail where only this one is.
        """
        _non_negative_int("posted", posted)
        upper = min(posted, len(self.probabilities) - 1)
        return math.fsum(self.tail(lots) for lots in range(1, upper + 1))

    @property
    def mean(self) -> float:
        """``E[A]``."""
        return self.expected_fill(self.support)

    @classmethod
    def from_counts(cls, counts: Sequence[float]) -> Outflow:
        """An empirical outflow, from a histogram of lots observed per interval."""
        if not counts:
            raise ValidationError("an outflow needs at least one count")
        for index, count in enumerate(counts):
            _finite(f"counts[{index}]", count)
            if count < 0.0:
                raise ValidationError(f"counts[{index}] must be non-negative, got {count!r}")
        total = math.fsum(counts)
        if total <= 0.0:
            raise ValidationError("counts must not be all zero")
        return cls(tuple(count / total for count in counts))

    @classmethod
    def point(cls, lots: int) -> Outflow:
        """A venue whose liquidity is known: exactly ``lots`` arrive."""
        _non_negative_int("lots", lots)
        return cls(tuple(1.0 if k == lots else 0.0 for k in range(lots + 1)))

    @classmethod
    def uniform(cls, support: int) -> Outflow:
        """Every outcome from zero to ``support`` lots equally likely."""
        _non_negative_int("support", support)
        weight = 1.0 / (support + 1)
        return cls(tuple(weight for _ in range(support + 1)))

    @classmethod
    def geometric(cls, mean: float, support: int) -> Outflow:
        """A geometric outflow truncated at ``support``, with the given mean.

        The untruncated geometric is the maximum-entropy distribution on the
        non-negative integers with a given mean, which makes it the honest
        default for a venue the caller knows one number about. Truncating it
        lowers the mean, so the ratio is solved for rather than taken from the
        textbook formula: the truncated mean rises monotonically from ``0`` at
        a ratio of zero to ``support / 2`` at a ratio of one, and a mean
        outside that range cannot be matched at this support.
        """
        _finite("mean", mean)
        _non_negative_int("support", support)
        if support == 0:
            raise ValidationError("a geometric outflow needs a support of at least one lot")
        if not 0.0 < mean < support / 2.0:
            raise ValidationError(
                f"mean must lie in (0, {support / 2.0}) for a support of {support} lots, "
                f"got {mean!r}"
            )

        def truncated_mean(ratio: float) -> float:
            weights = [ratio**k for k in range(support + 1)]
            total = math.fsum(weights)
            return math.fsum(k * weights[k] for k in range(support + 1)) / total

        low, high = 0.0, 1.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            if truncated_mean(middle) < mean:
                low = middle
            else:
                high = middle
        ratio = 0.5 * (low + high)
        weights = [ratio**k for k in range(support + 1)]
        total = math.fsum(weights)
        return cls(tuple(weight / total for weight in weights))


@dataclass(frozen=True)
class Venue:
    """One place the order can rest, and what resting there pays.

    ``rebate`` is per lot and signed the way a trader thinks of it: positive
    is money received for providing liquidity, and a venue that charges its
    makers takes a negative rebate. It is deliberately not folded into the
    sweep cost, because the two enter the marginal value of a posted lot in
    different places -- the rebate is earned whenever the lot fills, and the
    sweep is saved only when the lot fills *and* was needed.
    """

    name: str
    outflow: Outflow
    rebate: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("a venue needs a name")
        _finite("rebate", self.rebate)


@dataclass(frozen=True)
class RoutingProblem:
    """A quantity to buy, the venues it can rest at, and what each outcome costs.

    All four costs are per lot and positive-is-worse, in the same price units.

    * ``taker_cost`` is paid on the opening cross: half the spread plus the
      taker fee.
    * ``sweep_cost`` is paid on whatever is still unfilled when the order's
      time is up. It is the opening cross plus whatever the delay cost, so a
      problem where it is *below* ``taker_cost`` is one where there was never
      any reason to cross at the start; that is allowed, and the router will
      say so by taking nothing.
    * ``unwind_cost`` is paid per lot on an over-execution. Posting across
      several venues risks filling at more than one of them, and a model with
      no price on that would post everywhere.

    ``lot`` is the number of shares in a lot, used only to report costs per
    share. The optimisation is over integers because shares are integers, and
    rounding a continuous answer is measurably not the same thing.
    """

    target: int
    venues: tuple[Venue, ...]
    taker_cost: float
    sweep_cost: float
    unwind_cost: float
    lot: float = 1.0

    def __post_init__(self) -> None:
        if self.target < 1:
            raise ValidationError(f"target must be at least one lot, got {self.target!r}")
        if not self.venues:
            raise ValidationError("a routing problem needs at least one venue")
        if len(self.venues) > MAX_VENUES:
            raise ValidationError(
                f"at most {MAX_VENUES} venues, got {len(self.venues)}; the exhaustive "
                "reference router enumerates the product of their supports"
            )
        names = [venue.name for venue in self.venues]
        if len(set(names)) != len(names):
            raise ValidationError(f"venue names must be distinct, got {names}")
        for field_name in ("taker_cost", "sweep_cost", "unwind_cost"):
            value = float(getattr(self, field_name))
            _finite(field_name, value)
            if value < 0.0:
                raise ValidationError(f"{field_name} must be non-negative, got {value!r}")
        _finite("lot", self.lot)
        if self.lot <= 0.0:
            raise ValidationError(f"lot must be positive, got {self.lot!r}")

    @property
    def caps(self) -> tuple[int, ...]:
        """The largest posted quantity worth considering at each venue.

        Past a venue's support the tail is zero, so an extra lot there cannot
        fill: it changes neither the rebate earned nor the sweep avoided. The
        cap is therefore exact rather than a tolerance, which is what makes
        the exhaustive router a reference and not a wider guess.
        """
        return tuple(venue.outflow.support for venue in self.venues)


@dataclass(frozen=True)
class Allocation:
    """How many lots to cross for, and how many to post at each venue."""

    taker: int
    posted: tuple[int, ...]

    def __post_init__(self) -> None:
        _non_negative_int("taker", self.taker)
        for index, lots in enumerate(self.posted):
            _non_negative_int(f"posted[{index}]", lots)

    @property
    def total_posted(self) -> int:
        return sum(self.posted)

    @property
    def committed(self) -> int:
        """Lots the order has put at risk: the cross plus everything posted.

        Exceeds the target whenever the router is prepared to pay for an
        over-execution to raise the chance of a complete fill.
        """
        return self.taker + self.total_posted

    def with_lot_at(self, index: int, change: int) -> Allocation:
        """This allocation with ``change`` lots added at one venue."""
        posted = list(self.posted)
        posted[index] += change
        return Allocation(taker=self.taker, posted=tuple(posted))


@dataclass(frozen=True)
class FillDistribution:
    """The exact distribution of a total passive fill, on the lot lattice.

    ``probabilities[k]`` is the probability of filling exactly ``k`` lots
    across every venue the order rested at. It is a convolution of the
    per-venue fill distributions, computed in full: a normal approximation to
    this sum would be a reasonable thing to plot and the wrong thing to
    optimise over, because the two quantities a router pays for are tails of
    it at a specific lattice point.

    The two one-sided expectations are each built from the end of the
    distribution that is accurate there -- the shortfall from the cumulative
    probabilities at the bottom, the excess from the tails at the top -- which
    is the layer-cake identity used twice in opposite directions.
    """

    probabilities: tuple[float, ...]

    @property
    def support(self) -> int:
        """The largest total fill with any probability of occurring."""
        return len(self.probabilities) - 1

    def cumulative(self, lots: int) -> float:
        """``P(F <= lots)``, summed from the bottom."""
        if lots < 0:
            return 0.0
        if lots >= len(self.probabilities):
            return 1.0
        return math.fsum(self.probabilities[: lots + 1])

    def tail(self, lots: int) -> float:
        """``P(F >= lots)``, summed from the top."""
        if lots <= 0:
            return 1.0
        if lots >= len(self.probabilities):
            return 0.0
        return math.fsum(self.probabilities[lots:])

    @property
    def mean(self) -> float:
        """``E[F]``."""
        return math.fsum(lots * probability for lots, probability in enumerate(self.probabilities))

    def expected_shortfall(self, lots: int) -> float:
        """``E[(lots - F)^+]``: how much, on average, is left to sweep."""
        if lots <= 0:
            return 0.0
        return math.fsum(self.cumulative(level) for level in range(lots))

    def expected_excess(self, lots: int) -> float:
        """``E[(F - lots)^+]``: how much, on average, has to be unwound."""
        _non_negative_int("lots", lots)
        return math.fsum(self.tail(level) for level in range(lots + 1, len(self.probabilities)))


def venue_fill(outflow: Outflow, posted: int) -> FillDistribution:
    """The distribution of ``min(posted, A)`` at one venue.

    Mass piles up at ``posted`` itself: every outcome in which at least that
    much liquidity arrived fills the whole posted quantity and no more, so the
    top atom is the outflow's tail rather than its density there.
    """
    _non_negative_int("posted", posted)
    if posted == 0:
        return FillDistribution((1.0,))
    cap = min(posted, outflow.support)
    probabilities = [
        outflow.probabilities[lots] if lots < len(outflow.probabilities) else 0.0
        for lots in range(cap)
    ]
    probabilities.append(outflow.tail(cap))
    return FillDistribution(tuple(probabilities))


def _convolve(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    out = [0.0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a == 0.0:
            continue
        for j, b in enumerate(right):
            out[i + j] += a * b
    return tuple(out)


def fill_distribution(
    problem: RoutingProblem, allocation: Allocation, exclude: int | None = None
) -> FillDistribution:
    """The total passive fill across every venue, exactly.

    With ``exclude`` set, the venue at that index is left out. That
    leave-one-out distribution is not a diagnostic: it is what the closed-form
    marginal value of a posted lot is written in terms of, because conditional
    on the extra lot being reached the other venues are independent of
    everything that decided it.
    """
    _check_shape(problem, allocation)
    total: tuple[float, ...] = (1.0,)
    for index, venue in enumerate(problem.venues):
        if index == exclude:
            continue
        total = _convolve(total, venue_fill(venue.outflow, allocation.posted[index]).probabilities)
    return FillDistribution(total)


def _check_shape(problem: RoutingProblem, allocation: Allocation) -> None:
    if len(allocation.posted) != len(problem.venues):
        raise ValidationError(
            f"allocation covers {len(allocation.posted)} venues, problem has {len(problem.venues)}"
        )


@dataclass(frozen=True)
class RoutingCost:
    """What an allocation is expected to cost, by where the cost comes from.

    Every field is in currency, being a per-lot cost multiplied by a number
    of lots. ``rebate`` is signed as a cost like the rest, so a venue that
    pays its makers contributes a negative number and the four components add
    to ``total`` without a sign convention of their own.
    """

    cross: float
    rebate: float
    sweep: float
    unwind: float
    expected_passive_fill: float
    expected_shortfall: float
    expected_excess: float
    target: int
    lot: float

    @property
    def total(self) -> float:
        return self.cross + self.rebate + self.sweep + self.unwind

    @property
    def per_share(self) -> float:
        """The expected cost per share of the target quantity."""
        return self.total / (self.target * self.lot)

    @property
    def completion(self) -> float:
        """The expected fraction of the target filled without sweeping."""
        return 1.0 - self.expected_shortfall / self.target


def allocation_cost(problem: RoutingProblem, allocation: Allocation) -> RoutingCost:
    """The exact expected cost of an allocation.

    Four terms, and the third and fourth are the reason the whole fill
    distribution had to be computed rather than its mean: a sweep is paid on
    ``E[(need - F)^+]`` and an unwind on ``E[(F - need)^+]``, and those two
    are not functions of ``E[F]``.
    """
    _check_shape(problem, allocation)
    total_fill = fill_distribution(problem, allocation)
    need = problem.target - allocation.taker
    if need >= 0:
        shortfall = total_fill.expected_shortfall(need)
        excess = total_fill.expected_excess(need)
    else:
        # Crossing for more than the target over-executes before a single
        # passive lot fills, and every passive fill adds to that.
        shortfall = 0.0
        excess = float(-need) + total_fill.mean
    rebate = -math.fsum(
        venue.rebate * venue.outflow.expected_fill(allocation.posted[index])
        for index, venue in enumerate(problem.venues)
    )
    return RoutingCost(
        cross=problem.taker_cost * allocation.taker,
        rebate=rebate,
        sweep=problem.sweep_cost * shortfall,
        unwind=problem.unwind_cost * excess,
        expected_passive_fill=total_fill.mean,
        expected_shortfall=shortfall,
        expected_excess=excess,
        target=problem.target,
        lot=problem.lot,
    )


@dataclass(frozen=True)
class MarginalLot:
    """What one more lot, posted or crossed, does to the expected cost.

    ``reached`` is the probability that the lot trades at all; for a crossed
    lot it is one. ``needed`` is the probability that it was *also* wanted --
    that the rest of the order had not already filled the target without it --
    computed conditional on the lot being reached, which is exact here because
    the venues are independent.

    The three cost fields are changes in the corresponding components of
    :class:`RoutingCost` and add to :attr:`change`. A negative change is an
    improvement.
    """

    reached: float
    needed: float
    rebate: float
    sweep: float
    unwind: float

    @property
    def change(self) -> float:
        """The change in total expected cost. Negative is an improvement."""
        return self.rebate + self.sweep + self.unwind


def marginal_posted(problem: RoutingProblem, allocation: Allocation, index: int) -> MarginalLot:
    """One more lot posted at venue ``index``, in closed form.

    The whole of it is

        reached * (-rebate - sweep_cost * needed + unwind_cost * (1 - needed))

    with ``reached = P(A_i >= posted_i + 1)`` and ``needed`` the probability
    that the other venues and the cross together leave room for the lot. Two
    things follow from that form and neither is obvious from a search.

    **A posted lot is worth the probability that it is both reached and
    needed**, and those are probabilities of different things: ``reached``
    belongs to the venue, ``needed`` belongs to every *other* venue. So the
    rebate is earned on the first alone and the sweep is saved only on both.

    **Two venues paying the same rebate are ranked by their tail probability
    at the next lot, not by their average liquidity.** The venue index enters
    only through ``reached`` and the rebate; ``needed`` is a property of the
    leave-one-out distribution and of how much is already posted here. A
    venue with a high mean and a thin tail at the margin is the wrong place
    for the next lot even though it is the right place for the first.
    """
    _check_shape(problem, allocation)
    if not 0 <= index < len(problem.venues):
        raise ValidationError(f"venue index {index} out of range for {len(problem.venues)} venues")
    venue = problem.venues[index]
    posted = allocation.posted[index]
    reached = venue.outflow.tail(posted + 1)
    others = fill_distribution(problem, allocation, exclude=index)
    room = problem.target - allocation.taker - posted - 1
    needed = others.cumulative(room)
    return MarginalLot(
        reached=reached,
        needed=needed,
        rebate=-venue.rebate * reached,
        sweep=-problem.sweep_cost * reached * needed,
        unwind=problem.unwind_cost * reached * (1.0 - needed),
    )


def marginal_cross(problem: RoutingProblem, allocation: Allocation) -> MarginalLot:
    """One more lot crossed at the start, in closed form.

    A crossed lot is reached with certainty, so its only uncertainty is
    whether it was needed -- and that is read off the *full* passive
    distribution rather than a leave-one-out one, because crossing competes
    with every venue at once.
    """
    _check_shape(problem, allocation)
    passive = fill_distribution(problem, allocation)
    room = problem.target - allocation.taker - 1
    needed = passive.cumulative(room)
    return MarginalLot(
        reached=1.0,
        needed=needed,
        rebate=problem.taker_cost,
        sweep=-problem.sweep_cost * needed,
        unwind=problem.unwind_cost * (1.0 - needed),
    )
