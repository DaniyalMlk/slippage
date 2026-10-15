"""Estimating the spread from bars, because nobody has the quotes.

:mod:`slippage.placement` takes ``half_spread`` as an argument.
:mod:`slippage.shortfall` takes a spread assumption and charges it as a cost
component. Neither says where the number comes from, and it is usually not
available: cost analysis is run over bars, and the spread that was paid has
to be inferred from prices.

Three published estimators do that. This module implements all three and then
spends most of its effort on the question the papers do not answer, which is
when any of them can be believed.

## The three

**Roll (1984).** An efficient price moving as a random walk, plus a bounce
between bid and ask that is equally likely either way, gives successive
observed changes a covariance of exactly ``-(s/2)^2``. Invert it.

**Corwin and Schultz (2012).** The daily high-low range has two parts: how far
the efficient price travelled, which grows with the interval, and the spread,
which does not. Comparing one day's range against two days' separates them.

**Abdi and Ranaldo (2017).** The midrange estimates the day's efficient price
and barely carries the bounce; the close carries all of it. The close's
deviation from today's midrange and from tomorrow's share the bounce and
nothing else.

## All three refuse about half the time, and the refusal is the estimate

Each has a failure mode that is a sign rather than an error: Roll's sample
covariance can be positive, Corwin-Schultz's two-day windows can come out
negative, Abdi-Ranaldo's mean product can be negative. The published treatment
is to set those to zero, which is a **one-sided operation applied to a
two-sided sampling error**, so it cannot be neutral.

:func:`score_estimators` runs the model the estimators are derived under —
:func:`simulate_quotes` — so each can be scored against a spread that is
actually known. Over 200 runs of 252 days at 78 ticks a day, with a 2% daily
volatility:

| spread / volatility | estimator | mean | bias | before flooring |
|---|---|---|---|---|
| 0.05 | Roll | 45.1bp | +351% | 7.9bp |
| 0.05 | Corwin-Schultz | 72.7bp | +627% | **-4.3bp** |
| 0.05 | Abdi-Ranaldo | 22.0bp | +120% | **-1.1bp** |
| 0.25 | Roll | 57.2bp | +14% | 29.7bp |
| 0.25 | Corwin-Schultz | 90.1bp | +80% | 24.3bp |
| 0.25 | Abdi-Ranaldo | 43.3bp | -13% | 34.4bp |
| 1.00 | Roll | 200.7bp | +0.3% | 200.7bp |
| 1.00 | Corwin-Schultz | 192.7bp | -3.7% | 167.3bp |
| 1.00 | Abdi-Ranaldo | 199.4bp | -0.3% | 199.4bp |

The first block is the one worth reading. A 10bp spread against a 2% daily
volatility is an ordinary large-cap stock, and at that ratio the unfloored
estimates are **negative** for two of the three: the estimators carry no
information about the spread at all, and every basis point of the number they
report is the flooring. They are not noisy there. They are empty, and they
return a confident positive number anyway.

By a ratio of one they are all usable, and they are not interchangeable:
Roll's bias is 0.3% against a standard error of 1.4%, so it is unbiased as far
as 200 runs can tell, but its dispersion is 39.4bp against Abdi-Ranaldo's 13.2
and Corwin-Schultz's 11.4. Corwin-Schultz has the tightest dispersion and a
bias of -3.7% that *is* resolvable at a standard error of 0.4%. Unbiased and
wide, or tight and wrong by a measurable amount.

## Corwin-Schultz is calibrated to a trading frequency, not to a market

The range of a path sampled at finitely many points is not the range of the
path, and the high-low estimators are derived from the latter. Holding the
spread at one daily volatility and varying only how many trades a day the
high and low are the extremes of:

| ticks a day | Roll | Corwin-Schultz | Abdi-Ranaldo |
|---|---|---|---|
| 8 | -0.0% | **-45.5%** | -5.8% |
| 20 | -1.4% | -22.8% | -2.5% |
| 39 | -3.9% | -11.6% | -1.3% |
| 78 | -4.3% | -3.9% | -1.8% |
| 195 | +1.6% | +3.7% | -0.6% |
| 390 | -1.5% | **+7.0%** | -1.4% |

Roll does not depend on the tick count in any way 100 runs can see — its
standard error there is about 2.2% and nothing in its column exceeds twice
that. Corwin-Schultz moves monotonically across fifty-two percentage points
and passes through zero somewhere near a hundred trades a day. So the
estimator is right at one sampling frequency, and which frequency that is has
nothing to do with the market being measured. Abdi-Ranaldo has the same
dependence with a tenth of the range.

None of this is visible in a point estimate, which is the reason
:class:`SpreadEstimate` carries the rejection count and the unfloored value
rather than only the number.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .exceptions import InsufficientDataError, ValidationError
from .types import Bar

__all__ = [
    "Accuracy",
    "Estimator",
    "SpreadEstimate",
    "abdi_ranaldo_spread",
    "corwin_schultz_spread",
    "estimate_spread",
    "midrange",
    "roll_spread",
    "score_estimators",
    "simulate_quotes",
]

#: The Corwin-Schultz constant ``3 - 2 sqrt(2)``, written once.
_CS = 3.0 - 2.0 * math.sqrt(2.0)


class Estimator(str, Enum):
    """Which spread estimator produced a number."""

    ROLL = "roll"
    """Roll (1984): the serial covariance of price changes."""

    CORWIN_SCHULTZ = "corwin-schultz"
    """Corwin and Schultz (2012): two days of high-low ranges."""

    ABDI_RANALDO = "abdi-ranaldo"
    """Abdi and Ranaldo (2017): the close against two midranges."""


@dataclass(frozen=True)
class SpreadEstimate:
    """A proportional spread, and what the estimator had to throw away."""

    estimator: Estimator
    #: The full spread as a fraction of price. Multiply by 10,000 for basis
    #: points, halve it for the half-spread :mod:`slippage.placement` wants.
    spread: float
    #: Windows the estimator could form.
    windows: int
    #: Windows it could not use: Roll's positive covariances, Corwin-Schultz's
    #: negative two-day estimates, Abdi-Ranaldo's negative products.
    rejected: int
    #: The estimate before any negative values were floored at zero. Equal to
    #: :attr:`spread` when nothing was floored, and the honest number to look
    #: at when something was: flooring is a one-sided operation and biases the
    #: average up by exactly the mass it removes. It can be negative, and when
    #: it is, the reported spread is entirely an artefact of the flooring.
    unfloored: float

    def __post_init__(self) -> None:
        if self.windows < 0 or self.rejected < 0:
            raise ValidationError("window counts cannot be negative")

    @property
    def half_spread(self) -> float:
        """What :func:`slippage.placement.evaluate` takes, in price terms.

        Proportional, so multiply by the price to get a currency half-spread.
        """
        return 0.5 * self.spread

    @property
    def basis_points(self) -> float:
        return self.spread * 10000.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.windows if self.windows else 0.0

    @property
    def flooring_lift(self) -> float:
        """How much flooring the negatives raised the answer, as a fraction."""
        if self.unfloored <= 0.0:
            # The whole of the reported number is the flooring: there was no
            # positive estimate underneath it to raise.
            return math.inf if self.spread > 0.0 else 0.0
        return self.spread / self.unfloored - 1.0


def midrange(bar: Bar) -> float:
    """``(log high + log low) / 2``: the day's log midpoint estimate.

    Abdi and Ranaldo's ``eta``. It estimates the day's average efficient log
    price, and it does so without using the close, which is the whole point:
    the close carries the bounce and the midrange carries much less of it.
    """
    return 0.5 * (math.log(bar.high) + math.log(bar.low))


def _closes(bars: Sequence[Bar]) -> list[float]:
    if len(bars) < 3:
        raise InsufficientDataError(
            f"a serial covariance needs at least three bars, got {len(bars)}"
        )
    return [bar.close for bar in bars]


def roll_spread(bars: Sequence[Bar], *, floor: bool = True) -> SpreadEstimate:
    """Roll's estimator: ``2 sqrt(-cov(dp_t, dp_{t-1}))``.

    An efficient price that moves as a random walk, plus a bounce between bid
    and ask that is equally likely in either direction, gives successive
    observed price changes a covariance of exactly ``-(s/2)^2``. Invert that
    and the spread falls out of nothing but a price series.

    The estimator has one failure mode and it is not rare: a *sample*
    covariance can come out positive, and then there is no real square root.
    What the literature does is set the estimate to zero, which is a one-sided
    operation on a quantity whose sampling error is two-sided, so it biases
    the average down. ``floor=False`` returns the signed root instead —
    negative when the covariance was positive — so the two can be compared.
    """
    closes = _closes(bars)
    changes = [
        math.log(closes[index + 1]) - math.log(closes[index]) for index in range(len(closes) - 1)
    ]
    pairs = len(changes) - 1
    mean = math.fsum(changes) / len(changes)
    covariance = (
        math.fsum((changes[index] - mean) * (changes[index + 1] - mean) for index in range(pairs))
        / pairs
    )
    signed = -2.0 * math.copysign(math.sqrt(abs(covariance)), covariance)
    value = max(signed, 0.0) if floor else signed
    return SpreadEstimate(
        estimator=Estimator.ROLL,
        spread=value,
        windows=1,
        rejected=1 if covariance >= 0.0 else 0,
        unfloored=signed,
    )


def corwin_schultz_spread(bars: Sequence[Bar], *, floor: bool = True) -> SpreadEstimate:
    """The high-low estimator of Corwin and Schultz.

    The daily high-low range has two parts: how far the efficient price
    travelled, which grows with the length of the interval, and the spread,
    which does not. Comparing one day's range with the range over two days
    separates them, because only the first part scales.

    Each two-day window gives its own estimate and they are averaged. A
    sizeable share of them come out negative — the variance of a single
    window's range is large next to the spread it is trying to resolve — and
    the published treatment sets those to zero before averaging.
    :attr:`SpreadEstimate.unfloored` keeps the average without that step.
    """
    if len(bars) < 2:
        raise InsufficientDataError(
            f"the high-low estimator needs at least two bars, got {len(bars)}"
        )
    raw: list[float] = []
    for index in range(len(bars) - 1):
        first, second = bars[index], bars[index + 1]
        beta = math.log(first.high / first.low) ** 2 + math.log(second.high / second.low) ** 2
        high = max(first.high, second.high)
        low = min(first.low, second.low)
        gamma = math.log(high / low) ** 2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / _CS - math.sqrt(gamma / _CS)
        raw.append(2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha)))
    unfloored = math.fsum(raw) / len(raw)
    used = [max(one, 0.0) for one in raw] if floor else raw
    return SpreadEstimate(
        estimator=Estimator.CORWIN_SCHULTZ,
        spread=math.fsum(used) / len(used),
        windows=len(raw),
        rejected=sum(1 for one in raw if one < 0.0),
        unfloored=unfloored,
    )


def abdi_ranaldo_spread(bars: Sequence[Bar], *, floor: bool = True) -> SpreadEstimate:
    """The close-high-low estimator of Abdi and Ranaldo.

    The midrange is an estimate of the day's efficient price that barely
    carries the bounce; the close carries all of it. So the close's deviation
    from *today's* midrange and from *tomorrow's* share the bounce and nothing
    else, and four times the expected product of the two is the squared
    spread.

    It needs no range comparison across days and so does not inherit
    Corwin-Schultz's sensitivity to how long the interval is. It inherits the
    other problem: an average of products can be negative, and taking the root
    of a negative is the same refusal in different clothing.
    """
    if len(bars) < 2:
        raise InsufficientDataError(
            f"the close-high-low estimator needs at least two bars, got {len(bars)}"
        )
    products: list[float] = []
    for index in range(len(bars) - 1):
        close = math.log(bars[index].close)
        products.append(4.0 * (close - midrange(bars[index])) * (close - midrange(bars[index + 1])))
    mean = math.fsum(products) / len(products)
    signed = math.copysign(math.sqrt(abs(mean)), mean)
    value = max(signed, 0.0) if floor else signed
    return SpreadEstimate(
        estimator=Estimator.ABDI_RANALDO,
        spread=value,
        windows=len(products),
        rejected=sum(1 for one in products if one < 0.0),
        unfloored=signed,
    )


def estimate_spread(
    bars: Sequence[Bar], estimator: Estimator = Estimator.ABDI_RANALDO, *, floor: bool = True
) -> SpreadEstimate:
    """Whichever estimator was asked for."""
    if estimator is Estimator.ROLL:
        return roll_spread(bars, floor=floor)
    if estimator is Estimator.CORWIN_SCHULTZ:
        return corwin_schultz_spread(bars, floor=floor)
    return abdi_ranaldo_spread(bars, floor=floor)


# -- a market with a spread that is known -------------------------------------


def simulate_quotes(
    *,
    days: int,
    spread: float,
    volatility: float,
    ticks: int = 390,
    price: float = 100.0,
    drift: float = 0.0,
    seed: int = 0,
) -> list[Bar]:
    """Bars from an efficient random walk with a bid-ask bounce on top.

    Exactly the model all three estimators are derived under, so a run of this
    is the only place their answers can be compared against a spread that is
    known rather than estimated. ``ticks`` is how many trades a day the high
    and the low are the extremes *of*, which matters: the observed range of a
    path sampled at finitely many points is not the range of the path, and the
    high-low estimators are built on the latter.
    """
    if days < 2:
        raise ValidationError(f"at least two days are needed, got {days}")
    if ticks < 2:
        raise ValidationError(f"at least two ticks a day are needed, got {ticks}")
    if spread < 0.0:
        raise ValidationError(f"a spread is not negative, got {spread}")
    if volatility <= 0.0:
        raise ValidationError(f"volatility is positive, got {volatility}")
    rng = random.Random(seed)
    per_tick = volatility / math.sqrt(ticks)
    half = 0.5 * spread
    log_price = math.log(price)
    start = datetime(2024, 1, 1)
    bars: list[Bar] = []
    for day in range(days):
        observed: list[float] = []
        for _ in range(ticks):
            log_price += drift / ticks + rng.gauss(0.0, per_tick)
            side = 1.0 if rng.random() < 0.5 else -1.0
            observed.append(math.exp(log_price) * (1.0 + side * half))
        bars.append(
            Bar(
                timestamp=start + timedelta(days=day),
                open=observed[0],
                high=max(observed),
                low=min(observed),
                close=observed[-1],
                volume=1.0,
            )
        )
    return bars


@dataclass(frozen=True)
class Accuracy:
    """One estimator scored against a spread that is known."""

    estimator: Estimator
    runs: int
    truth: float
    mean: float
    #: Root mean squared error, in the same units as the spread.
    rmse: float
    #: ``mean / truth - 1``.
    bias: float
    #: The share of runs in which the estimator refused — a positive serial
    #: covariance, or a negative mean product.
    refusal_rate: float
    #: The mean before negatives were floored, which is what the bias would
    #: have been without that step.
    unfloored_mean: float


def score_estimators(
    *,
    days: int,
    spread: float,
    volatility: float,
    runs: int = 200,
    ticks: int = 390,
    drift: float = 0.0,
    seed: int = 0,
) -> tuple[Accuracy, ...]:
    """Run the simulation ``runs`` times and score all three estimators.

    Deterministic in ``seed``: each run uses ``seed + index``, so the same
    arguments give the same table.
    """
    if runs < 1:
        raise ValidationError(f"at least one run is needed, got {runs}")
    collected: dict[Estimator, list[SpreadEstimate]] = {one: [] for one in Estimator}
    for index in range(runs):
        bars = simulate_quotes(
            days=days,
            spread=spread,
            volatility=volatility,
            ticks=ticks,
            drift=drift,
            seed=seed + index,
        )
        for one in Estimator:
            collected[one].append(estimate_spread(bars, one))
    out: list[Accuracy] = []
    for one, results in collected.items():
        values = [result.spread for result in results]
        mean = math.fsum(values) / len(values)
        rmse = math.sqrt(math.fsum((value - spread) ** 2 for value in values) / len(values))
        refused = math.fsum(result.rejection_rate for result in results) / len(results)
        out.append(
            Accuracy(
                estimator=one,
                runs=runs,
                truth=spread,
                mean=mean,
                rmse=rmse,
                bias=mean / spread - 1.0 if spread > 0.0 else math.inf,
                refusal_rate=refused,
                unfloored_mean=math.fsum(result.unfloored for result in results) / len(results),
            )
        )
    return tuple(out)
