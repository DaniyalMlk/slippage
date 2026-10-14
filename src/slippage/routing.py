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
    "Outflow",
]


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
