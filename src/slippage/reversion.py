"""What happened to the price after the order finished.

The rest of this library is built on a model with two impact terms. Temporary
impact is the cost of demanding liquidity now and it decays once the demand
stops; permanent impact is the information the trade revealed and it does not.
Every schedule in :mod:`slippage.execution` is a trade between them, and
:func:`~slippage.calibration.fit_permanent` needs the permanent part measured
before it can estimate anything.

Nothing measured it. A total cost against arrival contains both, and a report
quoting that number cannot tell a desk which one it is paying — which is the only
thing the number is useful for, because one of the two responds to trading slower
and the other does not.

A mark-out is the measurement. Take the price when the order completed, look
again some minutes later, and split the move over the order window into the part
that persisted and the part that came back:

    impact    = sign * (completion / arrival - 1)
    permanent = sign * (mark / arrival - 1)
    reverted  = impact - permanent

signed so that positive is unfavourable for the side traded. The reverted
fraction is ``reverted / impact``, and on data generated from the model it
recovers the split the data was generated with.

Three things make this measurement harder than the arithmetic suggests, and each
is handled explicitly rather than by default.

**The market move is larger than the impact.** At any horizon worth measuring, a
stock's own drift swamps a few basis points of decay: over thirty minutes a 30%
annual volatility is about 25 basis points of noise against impact of perhaps
ten. Reading a trend as reversion is the default failure of this measurement, not
an edge case. So a benchmark series can be passed and its move subtracted with a
beta, and the result records whether that was done. Measured over twenty-five
synthetic books in ``examples/mark_outs.py``, the adjustment roughly triples the
precision of every figure here and takes the fitted permanent-impact coefficient
from a t-statistic below 2 in sixteen of the twenty-five books to above 3 in all
of them.

The subtraction is on simple returns, which is exact when a stock moves with the
benchmark one for one and first-order otherwise: the residual is of the order of
the square of the market move, 0.04 basis points against a market that wandered
thirty. That is negligible against any impact worth measuring and it is not zero.
Using the wrong beta is the error that matters — a beta of one on a stock whose
beta is a half leaves 8 basis points, two hundred times the rounding.

**A single order measures almost nothing.** The noise above does not cancel
within one order; it cancels across many. So the per-order figure exists and the
aggregate is what carries a standard error, and :class:`DecayPoint` reports both
the mean and the error so a reverted fraction can be read against it.

**A horizon past the end of the data is not an observation.**
:meth:`~slippage.series.BarSeries.price_at` clamps to the final close, which is
the right behaviour for a price lookup and the wrong one here: it silently turns
"we have no data that far out" into "the price did not move". Every mark-out
carries an ``observed`` flag, aggregation counts only the observed ones, and the
count is in the result.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from .exceptions import InsufficientDataError, ValidationError
from .series import BarSeries
from .types import Order, Side

#: Mark-out horizons used when none are given. Chosen to span the decay rather
#: than to sample it evenly: most of the reversion in equities happens in the
#: first few minutes, so the grid is denser there, and the long horizons are for
#: the asymptote rather than for the shape.
DEFAULT_HORIZONS: tuple[timedelta, ...] = (
    timedelta(minutes=1),
    timedelta(minutes=5),
    timedelta(minutes=15),
    timedelta(minutes=30),
    timedelta(minutes=60),
)

#: Below this, in basis points, the impact over the order window is treated as
#: indistinguishable from zero and no reverted *fraction* is reported. The
#: fraction divides by it, so a near-zero denominator turns a tenth of a basis
#: point of noise into a reverted fraction of forty.
IMPACT_FLOOR_BPS = 0.5


def _bps(value: float) -> float:
    return value * 1e4


def _return(series: BarSeries, start: datetime, end: datetime) -> float:
    first = series.price_at(start)
    if first <= 0.0:
        raise ValidationError(f"price at {start.isoformat()} is {first!r}; prices are positive")
    return series.price_at(end) / first - 1.0


@dataclass(frozen=True)
class MarkOut:
    """The price at one horizon after completion, and what it implies."""

    horizon: timedelta
    moment: datetime
    price: float
    #: False when ``moment`` is past the end of the series, in which case
    #: ``price`` is the final close and the figures below are not measurements.
    observed: bool
    #: Signed move from arrival to this mark-out, in basis points of arrival.
    #: The part of the impact that persisted.
    permanent_bps: float
    #: The part that came back, ``impact - permanent``. Positive means the price
    #: moved back towards where it started.
    reverted_bps: float
    #: Impact over the order window, repeated here so a mark-out can be read
    #: on its own.
    impact_bps: float

    @property
    def reverted_fraction(self) -> float | None:
        """``reverted / impact``, or ``None`` when the impact is too small to divide by.

        ``None`` rather than a large number: the ratio's denominator is a
        measured move, and at a tenth of a basis point of impact the fraction is
        a statement about the noise.
        """
        if abs(self.impact_bps) < IMPACT_FLOOR_BPS:
            return None
        return self.reverted_bps / self.impact_bps


@dataclass(frozen=True)
class Reversion:
    """One order's mark-out profile."""

    symbol: str
    side: Side
    quantity: float
    arrival_time: datetime
    arrival_price: float
    completion_time: datetime
    completion_price: float
    average_price: float
    #: Signed cost of the fills against arrival, in basis points. Not adjusted
    #: for the benchmark even when one was given: what was paid is a fact rather
    #: than a counterfactual. :attr:`adjusted_realised_bps` is the comparable
    #: figure.
    realised_bps: float
    #: The same cost with the benchmark's move over the order window removed.
    #: Equal to :attr:`realised_bps` when no benchmark was given.
    adjusted_realised_bps: float
    #: Signed move from arrival to completion, benchmark-adjusted. The quantity
    #: the mark-outs decompose.
    impact_bps: float
    marks: tuple[MarkOut, ...]
    #: Whether a benchmark was subtracted, and with what beta.
    benchmark_adjusted: bool
    beta: float

    def at(self, horizon: timedelta) -> MarkOut:
        for mark in self.marks:
            if mark.horizon == horizon:
                return mark
        raise ValidationError(
            f"no mark-out at {horizon}; this profile has "
            f"{', '.join(str(mark.horizon) for mark in self.marks)}"
        )

    @property
    def observed(self) -> tuple[MarkOut, ...]:
        return tuple(mark for mark in self.marks if mark.observed)

    def permanent_per_share(self, horizon: timedelta) -> float:
        """The persisted move in price units, signed as a cost.

        The shape :func:`~slippage.calibration.fit_permanent` wants, for one
        order. Price units rather than basis points because the fit is through
        the origin in shares against price, and a coefficient fitted on one and
        used on the other is wrong by the arrival price.
        """
        return self.at(horizon).permanent_bps / 1e4 * self.arrival_price


def price_reversion(
    order: Order,
    series: BarSeries,
    *,
    horizons: Sequence[timedelta] = DEFAULT_HORIZONS,
    benchmark: BarSeries | None = None,
    beta: float = 1.0,
) -> Reversion:
    """Mark out one order at each horizon after its last fill.

    The order window runs from ``arrival_time`` to the end of the bar containing
    the last fill — the same window :func:`~slippage.calibration.samples_from_orders`
    uses, because a mark-out measured from a different completion point than the
    cost it decomposes does not decompose it.

    Delay is excluded on purpose: impact is measured from arrival, not from the
    decision, so whatever the market did before the order reached it is not
    attributed to the order. That is
    :class:`~slippage.shortfall.DelayBasis`'s question and it has its own answer.
    """
    if not order.fills:
        raise InsufficientDataError(
            f"{order.symbol}: an order with no fills has no completion time to mark out from"
        )
    if not horizons:
        raise ValidationError("at least one horizon is needed")
    for horizon in horizons:
        if horizon <= timedelta(0):
            raise ValidationError(f"a mark-out horizon is positive, got {horizon!r}")
    if benchmark is None and beta != 1.0:
        raise ValidationError(f"a beta of {beta!r} was given with no benchmark to apply it to")
    last = order.last_fill_time
    assert last is not None  # guaranteed by the fills check above
    completion = series.end_of_bar_containing(last)
    arrival_price = series.price_at(order.arrival_time)
    if arrival_price <= 0.0:
        raise ValidationError(
            f"{order.symbol}: arrival price is {arrival_price!r}; prices are positive"
        )
    sign = order.side.sign

    def adjusted(end: datetime) -> float:
        move = _return(series, order.arrival_time, end)
        if benchmark is not None:
            move -= beta * _return(benchmark, order.arrival_time, end)
        return sign * move

    impact_bps = _bps(adjusted(completion))
    realised = sign * (order.average_price / arrival_price - 1.0)
    adjusted_realised = realised
    if benchmark is not None:
        adjusted_realised -= sign * beta * _return(benchmark, order.arrival_time, completion)

    marks: list[MarkOut] = []
    for horizon in sorted(set(horizons)):
        moment = completion + horizon
        permanent_bps = _bps(adjusted(moment))
        marks.append(
            MarkOut(
                horizon=horizon,
                moment=moment,
                price=series.price_at(moment),
                # `price_at` clamps to the final close past the end of the
                # series, so the flag is the only thing distinguishing a price
                # that did not move from a price nobody recorded.
                observed=moment < series.end,
                permanent_bps=permanent_bps,
                reverted_bps=impact_bps - permanent_bps,
                impact_bps=impact_bps,
            )
        )
    return Reversion(
        symbol=order.symbol,
        side=order.side,
        quantity=order.filled_quantity,
        arrival_time=order.arrival_time,
        arrival_price=arrival_price,
        completion_time=completion,
        completion_price=series.price_at(completion),
        average_price=order.average_price,
        realised_bps=_bps(realised),
        adjusted_realised_bps=_bps(adjusted_realised),
        impact_bps=impact_bps,
        marks=tuple(marks),
        benchmark_adjusted=benchmark is not None,
        beta=beta if benchmark is not None else 1.0,
    )


# -- across orders -------------------------------------------------------------


def _mean_and_error(values: Sequence[float]) -> tuple[float, float]:
    count = len(values)
    mean = math.fsum(values) / count
    if count < 2:
        return mean, math.inf
    variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance / count)


@dataclass(frozen=True)
class DecayPoint:
    """One horizon of the aggregate mark-out curve."""

    horizon: timedelta
    #: Orders with an observed mark-out at this horizon. Falls as the horizon
    #: lengthens and the data runs out, which is why it is reported per point
    #: rather than once for the profile.
    orders: int
    mean_permanent_bps: float
    #: Standard error of :attr:`mean_permanent_bps`. Usually larger than the
    #: quantity itself on fewer than a hundred orders, and saying so is the
    #: point of reporting it.
    standard_error: float
    mean_reverted_bps: float
    reverted_standard_error: float

    def reverted_fraction(self, mean_impact_bps: float) -> float | None:
        if abs(mean_impact_bps) < IMPACT_FLOOR_BPS:
            return None
        return self.mean_reverted_bps / mean_impact_bps


@dataclass(frozen=True)
class Decay:
    """An exponential fitted to the aggregate mark-out curve."""

    #: Time for the reverting part to halve.
    half_life: timedelta
    #: Where the curve is heading: the impact that does not come back.
    asymptote_bps: float
    #: How much of the impact at completion is the decaying part.
    amplitude_bps: float
    #: Fraction of the variance in the fitted points the curve explains.
    r_squared: float
    points: int

    @property
    def permanent_fraction(self) -> float | None:
        """The asymptote as a fraction of the impact at completion."""
        total = self.asymptote_bps + self.amplitude_bps
        if abs(total) < IMPACT_FLOOR_BPS:
            return None
        return self.asymptote_bps / total


@dataclass(frozen=True)
class DecayProfile:
    """Mark-outs aggregated across orders."""

    orders: int
    mean_impact_bps: float
    impact_standard_error: float
    mean_realised_bps: float
    points: tuple[DecayPoint, ...]
    benchmark_adjusted: bool

    def at(self, horizon: timedelta) -> DecayPoint:
        for point in self.points:
            if point.horizon == horizon:
                return point
        raise ValidationError(
            f"no mark-out at {horizon}; this profile has "
            f"{', '.join(str(point.horizon) for point in self.points)}"
        )

    @property
    def reverted_fraction(self) -> float | None:
        """The fraction that came back by the longest horizon *with observations*.

        The longest horizon asked for is often past the end of the bars, where
        there is no mean to take a fraction of — so this walks back to the last
        one that had orders in it rather than returning a NaN that would survive
        a JSON round trip and fail a strict parser at the far end.

        The headline number, and the one least defensible on a short sample: it is
        a ratio of two noisy means and the denominator is the smaller of the two.
        """
        for point in reversed(self.points):
            if point.orders > 0:
                return point.reverted_fraction(self.mean_impact_bps)
        return None

    def decay(self) -> Decay:
        """Fit ``permanent(h) = a + b exp(-h / tau)`` to the curve.

        Separable least squares: for a fixed ``tau`` the curve is linear in ``a``
        and ``b``, so those come out in closed form and only ``tau`` needs
        searching. That is the whole reason to write it this way rather than
        reaching for a general optimiser — there are no starting values to get
        wrong and no local minima in two of the three parameters.

        Refuses rather than returning a half-life when the curve is not
        decaying. An impact that grows with the horizon is a real thing to
        observe — it is what information looks like — and a negative amplitude
        dressed up as a half-life would hide it.
        """
        usable = [point for point in self.points if point.orders > 0]
        if len(usable) < 3:
            raise InsufficientDataError(
                f"a three-parameter curve needs at least three horizons with "
                f"observations, got {len(usable)}"
            )
        times = [point.horizon.total_seconds() for point in usable]
        values = [point.mean_permanent_bps for point in usable]
        span = times[-1] - times[0]
        if span <= 0.0:  # pragma: no cover - horizons are a sorted set
            raise ValidationError("the horizons do not span any time")

        def solve(tau: float) -> tuple[float, float, float]:
            basis = [math.exp(-time / tau) for time in times]
            count = len(times)
            mean_basis = math.fsum(basis) / count
            mean_value = math.fsum(values) / count
            centred = math.fsum((one - mean_basis) ** 2 for one in basis)
            if centred <= 0.0:
                return mean_value, 0.0, math.inf
            slope = (
                math.fsum(
                    (one - mean_basis) * (value - mean_value)
                    for one, value in zip(basis, values, strict=True)
                )
                / centred
            )
            intercept = mean_value - slope * mean_basis
            residual = math.fsum(
                (value - intercept - slope * one) ** 2
                for one, value in zip(basis, values, strict=True)
            )
            return intercept, slope, residual

        # Golden section on log tau, because the parameter is a time scale and a
        # search linear in it spends most of its evaluations at the wrong end.
        low, high = math.log(span / 100.0), math.log(span * 100.0)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        left, right = high - ratio * (high - low), low + ratio * (high - low)
        value_left, value_right = solve(math.exp(left))[2], solve(math.exp(right))[2]
        for _ in range(200):
            if high - low < 1e-10 * (abs(low) + abs(high)):
                break
            if value_left < value_right:
                high, right, value_right = right, left, value_left
                left = high - ratio * (high - low)
                value_left = solve(math.exp(left))[2]
            else:
                low, left, value_left = left, right, value_right
                right = low + ratio * (high - low)
                value_right = solve(math.exp(right))[2]
        tau = math.exp((low + high) / 2.0)
        asymptote, amplitude, residual = solve(tau)
        if amplitude <= 0.0:
            raise InsufficientDataError(
                f"the mark-out curve does not decay: the fitted amplitude is "
                f"{amplitude:.4g} basis points, so the impact at {usable[-1].horizon} "
                f"is at least as large as at completion. That is what information "
                f"looks like rather than a fit to report a half-life from."
            )
        mean_value = math.fsum(values) / len(values)
        total = math.fsum((value - mean_value) ** 2 for value in values)
        return Decay(
            half_life=timedelta(seconds=tau * math.log(2.0)),
            asymptote_bps=asymptote,
            amplitude_bps=amplitude,
            r_squared=1.0 - residual / total if total > 0.0 else 1.0,
            points=len(usable),
        )


def reversion_profile(
    orders: Iterable[Order],
    series: Mapping[str, BarSeries],
    *,
    horizons: Sequence[timedelta] = DEFAULT_HORIZONS,
    benchmark: Mapping[str, BarSeries] | BarSeries | None = None,
    beta: float = 1.0,
) -> DecayProfile:
    """Aggregate mark-outs across orders, with standard errors.

    ``benchmark`` may be one series for everything — an index, which is the
    usual case — or one per symbol. Orders with no fills are skipped rather than
    refused, matching :func:`~slippage.calibration.samples_from_orders`: a parent
    order that never traded is a normal thing to find in a day's blotter.
    """
    profiles: list[Reversion] = []
    for order in orders:
        if not order.fills:
            continue
        try:
            bars = series[order.symbol]
        except KeyError:
            raise ValidationError(
                f"no bars for {order.symbol!r}; a mark-out needs the symbol's own series"
            ) from None
        against = benchmark[order.symbol] if isinstance(benchmark, Mapping) else benchmark
        profiles.append(
            price_reversion(order, bars, horizons=horizons, benchmark=against, beta=beta)
        )
    if not profiles:
        raise InsufficientDataError("no orders with fills to mark out")

    impact_mean, impact_error = _mean_and_error([one.impact_bps for one in profiles])
    realised_mean = math.fsum(one.realised_bps for one in profiles) / len(profiles)
    points: list[DecayPoint] = []
    for horizon in sorted(set(horizons)):
        permanents = [one.at(horizon).permanent_bps for one in profiles if one.at(horizon).observed]
        if not permanents:
            points.append(
                DecayPoint(
                    horizon=horizon,
                    orders=0,
                    mean_permanent_bps=math.nan,
                    standard_error=math.inf,
                    mean_reverted_bps=math.nan,
                    reverted_standard_error=math.inf,
                )
            )
            continue
        reverted = [one.at(horizon).reverted_bps for one in profiles if one.at(horizon).observed]
        permanent_mean, permanent_error = _mean_and_error(permanents)
        reverted_mean, reverted_error = _mean_and_error(reverted)
        points.append(
            DecayPoint(
                horizon=horizon,
                orders=len(permanents),
                mean_permanent_bps=permanent_mean,
                standard_error=permanent_error,
                mean_reverted_bps=reverted_mean,
                reverted_standard_error=reverted_error,
            )
        )
    return DecayProfile(
        orders=len(profiles),
        mean_impact_bps=impact_mean,
        impact_standard_error=impact_error,
        mean_realised_bps=realised_mean,
        points=tuple(points),
        benchmark_adjusted=benchmark is not None,
    )


def permanent_moves_from_orders(
    orders: Iterable[Order],
    series: Mapping[str, BarSeries],
    *,
    horizon: timedelta,
    benchmark: Mapping[str, BarSeries] | BarSeries | None = None,
    beta: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Quantities and persisted moves, in the shape ``fit_permanent`` takes.

    Which was the gap this module exists to close.
    :func:`~slippage.calibration.fit_permanent` documents its second argument as
    "the price changes that persisted after each order completed, typically
    measured well after the last fill" and left the caller to walk the bars, pick
    a horizon, and sign the move for the side. Getting any of the three wrong
    produces a gamma with the right units and the wrong value.

    Quantities are unsigned and moves are signed as a cost, which is the
    convention the other fits in that module use: a buy that pushed the price up
    and a sell that pushed it down both contributed positively to gamma.

    Orders whose mark-out at ``horizon`` falls past the end of their bars are
    skipped, because the alternative is a zero move standing in for a
    measurement nobody has.
    """
    quantities: list[float] = []
    moves: list[float] = []
    for order in orders:
        if not order.fills:
            continue
        try:
            bars = series[order.symbol]
        except KeyError:
            raise ValidationError(
                f"no bars for {order.symbol!r}; a mark-out needs the symbol's own series"
            ) from None
        against = benchmark[order.symbol] if isinstance(benchmark, Mapping) else benchmark
        profile = price_reversion(order, bars, horizons=(horizon,), benchmark=against, beta=beta)
        if not profile.at(horizon).observed:
            continue
        quantities.append(profile.quantity)
        moves.append(profile.permanent_per_share(horizon))
    if not quantities:
        raise InsufficientDataError(
            f"no order had an observed mark-out {horizon} after it completed; the "
            "bars end too soon for this horizon"
        )
    return quantities, moves
