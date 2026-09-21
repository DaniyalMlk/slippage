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

from .series import BarSeries
from .types import Bar, Fill, Order, Side

__all__ = ["SyntheticBook", "synthetic_book"]

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
