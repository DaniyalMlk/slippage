"""Synthetic books of orders and market data with a known impact law.

For examples, tests and trying the command line without real data. Every order
is executed against its own symbol's minute bars and pays impact that follows
a square-root law with a known prefactor, so calibration run on the output can
be checked against the truth.

The bars are the *unaffected* market: the orders' impact shows in their fill
prices, not in the prints around them. That keeps the generated arrival and
VWAP benchmarks independent of the orders being scored, which is the
assumption a TCA report makes about real data too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError
from .series import BarSeries
from .types import Bar, Fill, Order, Side

__all__ = ["DecayingBook", "SyntheticBook", "decaying_book", "synthetic_book"]

SESSION_OPEN = time(9, 30)
SESSION_MINUTES = 390


@dataclass(frozen=True)
class SyntheticBook:
    """Orders, the bars they traded against, and the truth behind them."""

    orders: dict[str, Order]
    bars: dict[str, BarSeries]
    daily_volume: dict[str, float]
    daily_volatility: dict[str, float]
    impact_y: float
    """Prefactor of the square-root law the fills were generated from."""


def _intraday_shape() -> NDArray[np.float64]:
    """A U-shaped minute-by-minute volume curve summing to one."""
    t = np.linspace(0.0, 1.0, SESSION_MINUTES)
    shape = 1.0 + 10.0 * (t - 0.5) ** 2
    normalised: NDArray[np.float64] = shape / shape.sum()
    return normalised


def _session(
    rng: np.random.Generator, day: date, price: float, vol: float, volume: float
) -> BarSeries:
    start = datetime.combine(day, SESSION_OPEN)
    step = vol / math.sqrt(SESSION_MINUTES)
    shape = _intraday_shape()
    bars = []
    level = price
    for minute in range(SESSION_MINUTES):
        move = level * step * rng.standard_normal()
        close = max(level + move, 0.01)
        wiggle = level * step * abs(rng.standard_normal()) * 0.5
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=minute),
                open=round(level, 4),
                high=round(max(level, close) + wiggle, 4),
                low=round(max(min(level, close) - wiggle, 0.01), 4),
                close=round(close, 4),
                volume=float(round(volume * shape[minute] * rng.lognormal(0.0, 0.3))),
            )
        )
        level = close
    return BarSeries(bars)


def synthetic_book(
    rng: np.random.Generator,
    *,
    symbols: int = 8,
    orders: int = 60,
    impact_y: float = 0.7,
    day: date = date(2026, 3, 2),
) -> SyntheticBook:
    """Generate a day's book of orders across several symbols.

    Order sizes range from 0.01% to 2% of daily volume, worked at 5-20%
    participation. About one order in ten is cut short at 60-90% filled.
    Each fill pays half a spread plus square-root impact on the size done so
    far, ``impact_y * sigma * sqrt(done / V)``, on top of the unaffected price.
    """
    names = [f"SYM{i:02d}" for i in range(symbols)]
    prices = {s: float(rng.uniform(20.0, 200.0)) for s in names}
    vols = {s: float(rng.uniform(0.012, 0.035)) for s in names}
    volumes = {s: float(rng.uniform(2e6, 2e7)) for s in names}
    bars = {s: _session(rng, day, prices[s], vols[s], volumes[s]) for s in names}

    book: dict[str, Order] = {}
    for k in range(orders):
        symbol = names[int(rng.integers(symbols))]
        series = bars[symbol]
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        size_fraction = float(math.exp(rng.uniform(math.log(1e-4), math.log(2e-2))))
        quantity = float(max(round(size_fraction * volumes[symbol], -2), 100.0))
        participation = float(rng.uniform(0.05, 0.20))
        per_minute = participation * volumes[symbol] / SESSION_MINUTES
        minutes = max(1, min(math.ceil(quantity / per_minute), 240))
        decision_minute = int(rng.integers(0, SESSION_MINUTES - minutes - 20))
        delay = int(rng.integers(0, 15))
        decision = series[decision_minute].timestamp
        arrival = decision + timedelta(minutes=delay)
        fill_rate = 1.0 if rng.random() > 0.1 else float(rng.uniform(0.6, 0.9))
        target = quantity * fill_rate

        slices = max(1, minutes // 5)
        slice_size = target / slices
        fills = []
        done = 0.0
        for j in range(slices):
            moment = arrival + timedelta(minutes=5 * j + int(rng.integers(0, 5)))
            if moment >= series.end:
                break
            size = float(round(slice_size, 0)) if j < slices - 1 else float(round(target - done))
            if size <= 0.0:
                continue
            done += size
            mid = series.price_at(moment)
            impact = impact_y * vols[symbol] * math.sqrt(done / volumes[symbol])
            half_spread = 0.0002
            price = mid * (1.0 + side.sign * (impact + half_spread))
            fills.append(
                Fill(
                    timestamp=moment,
                    quantity=size,
                    price=round(price, 4),
                    commission=round(0.002 * size, 2),
                )
            )
        order_id = f"ORD{k:04d}"
        book[order_id] = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            decision_time=decision,
            arrival_time=arrival,
            fills=tuple(fills),
            decision_price=series.price_at(decision),
        )
    return SyntheticBook(
        orders=book,
        bars=bars,
        daily_volume=volumes,
        daily_volatility=vols,
        impact_y=impact_y,
    )


@dataclass(frozen=True)
class DecayingBook:
    """Orders whose impact is *in the prints*, with the decay it was built from.

    :func:`synthetic_book` deliberately keeps impact out of the bars: its prints
    are the unaffected market and the impact lives only in the fill prices. That
    is the right shape for calibrating a cost model against an arrival benchmark
    and it is useless for measuring a mark-out, because a mark-out reads the
    impact off the prints. So this is a second generator rather than an option on
    the first, and the difference between them is the assumption each one is
    testing.
    """

    orders: dict[str, Order]
    bars: dict[str, BarSeries]
    #: One index series, moving with every symbol by its beta. Passing it to
    #: :func:`~slippage.reversion.reversion_profile` is what the benchmark
    #: adjustment is for.
    index: BarSeries
    #: Impact at completion that never comes back, in basis points.
    permanent_bps: float
    #: Impact at completion that decays, in basis points.
    temporary_bps: float
    #: Half-life of that decay.
    half_life: timedelta
    #: Beta of every symbol to :attr:`index`.
    beta: float


def _impact_bps(
    moment: datetime,
    start: datetime,
    finish: datetime,
    permanent: float,
    temporary: float,
    decay: float,
) -> float:
    """Impact in basis points at ``moment``, building then decaying.

    Linear in the executed fraction while trading, because impact accumulates
    with the quantity done rather than with the clock; exponential in the elapsed
    time afterwards, because that is the shape a mark-out curve is fitted with
    and a generator that built one shape to be measured by another would be
    testing the fitter against itself.
    """
    if moment < start:
        return 0.0
    if moment <= finish:
        return (permanent + temporary) * (moment - start) / (finish - start)
    elapsed = (moment - finish).total_seconds()
    return permanent + temporary * math.exp(-elapsed / decay)


def decaying_book(
    rng: np.random.Generator,
    *,
    orders: int = 120,
    permanent_bps: float = 4.0,
    temporary_bps: float = 6.0,
    half_life: timedelta = timedelta(seconds=208),
    market_volatility_bps: float = 3.0,
    idiosyncratic_volatility_bps: float = 1.0,
    beta: float = 1.0,
    execution_minutes: int = 20,
    day: date = date(2026, 3, 2),
) -> DecayingBook:
    """A book whose prints carry impact that decays to a known asymptote.

    Every symbol's log price is a market factor times ``beta``, plus its own
    noise, plus the order's own impact signed for the side. Sides alternate, so
    an estimator that forgot to sign the move for the side returns zero on
    average rather than something plausible.

    ``market_volatility_bps`` and ``idiosyncratic_volatility_bps`` are per
    minute. The defaults are the interesting case rather than a quiet one: over
    the eighty minutes from arrival to the last mark-out the market factor
    accumulates about 27 basis points against ten of impact, so the measurement
    only works once the factor is removed.
    """
    if execution_minutes < 1:
        raise ValidationError(f"execution_minutes must be positive, got {execution_minutes!r}")
    if half_life <= timedelta(0):
        raise ValidationError(f"half_life must be positive, got {half_life!r}")
    decay = half_life.total_seconds() / math.log(2.0)
    opening = datetime.combine(day, SESSION_OPEN)
    minute = timedelta(minutes=1)
    factor = np.cumsum(rng.normal(0.0, market_volatility_bps, SESSION_MINUTES))

    index_bars = []
    for i in range(SESSION_MINUTES):
        price = 1000.0 * (1.0 + factor[i] / 1e4)
        index_bars.append(
            Bar(opening + i * minute, price, price * 1.002, price * 0.998, price, 1e7)
        )
    index = BarSeries(index_bars)

    book: dict[str, Order] = {}
    bars: dict[str, BarSeries] = {}
    for k in range(orders):
        symbol = f"SYM{k:04d}"
        side = Side.BUY if k % 2 == 0 else Side.SELL
        base = float(rng.uniform(20.0, 200.0))
        quantity = float(round(rng.uniform(20_000.0, 80_000.0), -2))
        noise = np.cumsum(rng.normal(0.0, idiosyncratic_volatility_bps, SESSION_MINUTES))
        start_minute = int(rng.integers(10, 40))
        start = opening + start_minute * minute
        finish = start + execution_minutes * minute
        prices: list[float] = []
        series_bars = []
        for i in range(SESSION_MINUTES):
            moment = opening + i * minute
            impact = side.sign * _impact_bps(
                moment, start, finish, permanent_bps, temporary_bps, decay
            )
            price = base * (1.0 + (impact + beta * factor[i] + noise[i]) / 1e4)
            prices.append(price)
            series_bars.append(Bar(moment, price, price * 1.002, price * 0.998, price, 20_000.0))
        bars[symbol] = BarSeries(series_bars)
        fills = tuple(
            Fill(
                timestamp=start + i * minute + timedelta(seconds=30),
                quantity=quantity / execution_minutes,
                price=round(prices[start_minute + i], 4),
            )
            for i in range(execution_minutes)
        )
        book[symbol] = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            decision_time=start,
            arrival_time=start,
            fills=fills,
        )
    return DecayingBook(
        orders=book,
        bars=bars,
        index=index,
        permanent_bps=permanent_bps,
        temporary_bps=temporary_bps,
        half_life=half_life,
        beta=beta,
    )
