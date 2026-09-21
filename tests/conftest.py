from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from slippage.types import Bar, Fill, Order, Side

START = datetime(2026, 3, 2, 9, 30)


def minute(n: int) -> datetime:
    return START + timedelta(minutes=n)


def make_bar(
    n: int,
    close: float,
    *,
    open_: float | None = None,
    spread: float = 0.05,
    volume: float = 10_000.0,
) -> Bar:
    """A bar whose typical price is close to ``close``, for readable fixtures."""
    o = close if open_ is None else open_
    high = max(o, close) + spread
    low = min(o, close) - spread
    return Bar(timestamp=minute(n), open=o, high=high, low=low, close=close, volume=volume)


@pytest.fixture
def flat_bars() -> list[Bar]:
    """Ten one-minute bars at a constant 100.00 with constant volume."""
    return [make_bar(n, 100.0) for n in range(10)]


@pytest.fixture
def rising_bars() -> list[Bar]:
    """Ten one-minute bars rising by 0.10 a minute from 100.00."""
    return [
        make_bar(n, 100.0 + 0.10 * n, open_=100.0 + 0.10 * (n - 1) if n else 100.0)
        for n in range(10)
    ]


@pytest.fixture
def simple_order() -> Order:
    return Order(
        symbol="ACME",
        side=Side.BUY,
        quantity=1_000.0,
        decision_time=minute(0),
        arrival_time=minute(1),
        fills=(
            Fill(timestamp=minute(2), quantity=400.0, price=100.10, commission=4.0),
            Fill(timestamp=minute(4), quantity=600.0, price=100.20, commission=6.0),
        ),
        decision_price=100.0,
    )
