"""Market impact models.

Two levels of description are provided, because they answer different
questions.

**Rate models** (:class:`LinearImpact`, :class:`PowerLawImpact`) describe the
cost of trading at a given *speed*. They are what a scheduler needs: the
expected cost of any discrete schedule follows from them, which is how
:func:`schedule_cost` works and how the optimal trajectories are derived.

**The square-root law** (:class:`SquareRootLaw`) describes the impact of a whole
metaorder as a function of its *size* relative to daily volume. It is the most
robust empirical regularity in the microstructure literature, and it is what a
pre-trade estimate for a single order usually wants.

Conventions
-----------

Quantities are unsigned share counts and every cost is positive, because impact
is symmetric: buying ``X`` costs what selling ``X`` costs. Time is measured in
whatever unit ``tau`` is expressed in, and ``eta`` must be quoted per that same
unit — mixing a per-day ``eta`` with a ``tau`` in minutes is the easiest way to
be wrong by three orders of magnitude, so the unit is part of each model's
docstring rather than a silent assumption.

Why permanent impact is always linear
-------------------------------------

Huberman and Stanzl (2004) show that permanent impact which is not linear in
quantity admits price manipulation: a round trip that buys fast and sells slow
(or the reverse) makes money in expectation. A model with that property would
let an optimiser "find" profit in its own impact. Both rate models therefore
take a single permanent coefficient ``gamma`` and have no way to express
anything else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .costs import BPS_PER_UNIT
from .exceptions import ValidationError

__all__ = [
    "ImpactModel",
    "LinearImpact",
    "PowerLawImpact",
    "ScheduleCost",
    "SquareRootLaw",
    "schedule_cost",
    "uniform_schedule",
]


def _non_negative(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError(f"{name} must be a non-negative finite number, got {value!r}")
    return float(value)


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


@runtime_checkable
class ImpactModel(Protocol):
    """Anything that prices trading at a rate, with linear permanent impact."""

    @property
    def gamma(self) -> float:
        """Permanent impact: price shift per share traded."""
        ...

    @property
    def epsilon(self) -> float:
        """Fixed cost per share: half the spread plus per-share fees."""
        ...

    def temporary(self, rate: float) -> float:
        """Per-share price concession for trading at ``rate`` shares per unit time."""
        ...


@dataclass(frozen=True)
class LinearImpact:
    """Almgren and Chriss (2000): ``g(v) = gamma v`` and ``h(v) = epsilon + eta v``.

    Parameters
    ----------
    gamma
        Permanent impact, in price per share.
    eta
        Temporary impact, in price per (share per unit time). The time unit
        must match the ``tau`` used with the model.
    epsilon
        Fixed per-share cost, charged only when shares actually trade.
    """

    gamma: float
    eta: float
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", _non_negative("gamma", self.gamma))
        object.__setattr__(self, "eta", _non_negative("eta", self.eta))
        object.__setattr__(self, "epsilon", _non_negative("epsilon", self.epsilon))

    def temporary(self, rate: float) -> float:
        if rate < 0.0:
            raise ValidationError(f"rate must be unsigned, got {rate!r}")
        if rate == 0.0:
            return 0.0
        return self.epsilon + self.eta * rate


@dataclass(frozen=True)
class PowerLawImpact:
    """Temporary impact ``h(v) = epsilon + eta v**beta`` with linear permanent impact.

    ``beta = 1`` recovers :class:`LinearImpact`. Empirical estimates of the
    exponent for equities cluster around 0.5 to 0.6 (Almgren, Thum, Hauptmann
    and Li, 2005, report 0.6). The per-slice cost ``n h(n / tau)`` is convex in
    ``n`` for any ``beta > 0``, which is what makes the scheduling problem well
    posed.
    """

    gamma: float
    eta: float
    beta: float
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", _non_negative("gamma", self.gamma))
        object.__setattr__(self, "eta", _non_negative("eta", self.eta))
        object.__setattr__(self, "epsilon", _non_negative("epsilon", self.epsilon))
        beta = _positive("beta", self.beta)
        if beta > 3.0:
            raise ValidationError(f"beta above 3 is not a plausible impact exponent, got {beta}")
        object.__setattr__(self, "beta", beta)

    def temporary(self, rate: float) -> float:
        if rate < 0.0:
            raise ValidationError(f"rate must be unsigned, got {rate!r}")
        if rate == 0.0:
            return 0.0
        return self.epsilon + self.eta * float(rate**self.beta)


# -- schedule costs ---------------------------------------------------------


@dataclass(frozen=True)
class ScheduleCost:
    """Expected cost of a schedule, in currency, split by source."""

    temporary: float
    permanent: float

    @property
    def total(self) -> float:
        return self.temporary + self.permanent


def uniform_schedule(quantity: float, periods: int) -> list[float]:
    """``quantity`` split into ``periods`` equal slices."""
    _non_negative("quantity", quantity)
    if periods < 1:
        raise ValidationError(f"periods must be at least 1, got {periods}")
    return [quantity / periods] * periods


def schedule_cost(model: ImpactModel, trades: Sequence[float], tau: float) -> ScheduleCost:
    """Expected impact cost of executing ``trades`` in consecutive intervals of ``tau``.

    Trade ``k`` executes at the pre-trade price moved by the permanent impact
    of every earlier trade, less its own temporary concession::

        temporary = sum_k n_k * h(n_k / tau)
        permanent = sum_k n_k * gamma * (n_1 + ... + n_{k-1})

    For linear permanent impact the second sum is
    ``gamma * (X**2 - sum_k n_k**2) / 2``, the Almgren-Chriss identity, which
    the tests check rather than assume.
    """
    tau = _positive("tau", tau)
    temporary = 0.0
    permanent = 0.0
    done = 0.0
    for index, n in enumerate(trades):
        if not math.isfinite(n) or n < 0.0:
            raise ValidationError(f"trade {index} must be a non-negative size, got {n!r}")
        temporary += n * model.temporary(n / tau)
        permanent += n * model.gamma * done
        done += n
    return ScheduleCost(temporary=temporary, permanent=permanent)


# -- metaorder-level law ----------------------------------------------------


@dataclass(frozen=True)
class SquareRootLaw:
    """Metaorder impact ``I = Y * sigma * (Q / V) ** delta``.

    ``sigma`` is daily volatility as a fraction, ``Q`` the order size and ``V``
    daily volume, so ``I`` is a fraction of price. ``Y`` is of order one in
    most published estimates and ``delta = 0.5`` is the square-root law
    proper.

    ``I`` is the *peak* impact, reached when the last share trades. An order
    executed at a constant rate pays on average the mean of the impact path,
    ``I * (t / T) ** delta`` over ``[0, T]``, which is ``I / (1 + delta)`` —
    two thirds of the peak under the square root. Quoting the peak as the
    cost overstates it by half.
    """

    y: float = 1.0
    delta: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", _non_negative("y", self.y))
        object.__setattr__(self, "delta", _positive("delta", self.delta))

    def peak_impact(self, quantity: float, daily_volume: float, daily_volatility: float) -> float:
        """Peak impact as a fraction of price."""
        _non_negative("quantity", quantity)
        volume = _positive("daily_volume", daily_volume)
        vol = _non_negative("daily_volatility", daily_volatility)
        return self.y * vol * float((quantity / volume) ** self.delta)

    def expected_cost(self, quantity: float, daily_volume: float, daily_volatility: float) -> float:
        """Average impact paid per share at a constant rate, as a fraction of price."""
        return self.peak_impact(quantity, daily_volume, daily_volatility) / (1.0 + self.delta)

    def expected_cost_bps(
        self, quantity: float, daily_volume: float, daily_volatility: float
    ) -> float:
        return BPS_PER_UNIT * self.expected_cost(quantity, daily_volume, daily_volatility)
