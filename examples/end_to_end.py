"""From raw executions to a scheduling recommendation.

1. Write a synthetic book to CSV and read it back, as raw fills would arrive.
2. Report implementation shortfall across the book.
3. Calibrate a square-root impact law from the executions.
4. Turn the law into a rate model and schedule a new order under a
   participation cap, in half-hour buckets across one session.

Run with ``python examples/end_to_end.py``.
"""

from __future__ import annotations

import tempfile
import warnings
from datetime import time, timedelta
from pathlib import Path

import numpy as np

from slippage import (
    IdentifiabilityWarning,
    PowerLawImpact,
    build_report,
    estimate_profile,
    fit_power_law,
    load_bars,
    load_orders,
    participation_caps,
    samples_from_orders,
    solve_schedule,
    synthetic_book,
    write_bars,
    write_orders,
)

SYMBOL = "SYM00"


def main() -> None:
    book = synthetic_book(np.random.default_rng(21), symbols=6, orders=400)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        write_orders(book.orders, root / "orders.csv", root / "fills.csv")
        write_bars(book.bars, root / "bars.csv")
        orders = load_orders(root / "orders.csv", root / "fills.csv")
        bars = load_bars(root / "bars.csv")

    # 1. What did the book cost?
    report = build_report(orders, bars, fees_per_share=0.0005)
    total = report.total()
    print(f"book: {total.orders} orders, shortfall {total.total_bps:+.2f} bps of paper notional")
    for name, value in total.bps().items():
        print(f"  {name:<12}{value:+8.2f} bps")
    print(f"  outliers flagged: {len(report.outliers())}")

    # 2. What does impact look like? Fit the square-root law on trading cost.
    samples = samples_from_orders(
        orders.values(), bars, book.daily_volume, book.daily_volatility, time_unit=timedelta(days=1)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IdentifiabilityWarning)
        fit = fit_power_law(
            [s.participation for s in samples],
            [s.cost_fraction for s in samples],
            [s.volatility for s in samples],
            delta=0.5,
        )
    print(f"\nsquare-root law: Y = {fit.y.value:.3f} +/- {fit.y.std_error:.3f}")

    # 3. Schedule a new sale of 5% of daily volume in one symbol.
    volume = book.daily_volume[SYMBOL]
    sigma = book.daily_volatility[SYMBOL]
    price = bars[SYMBOL][-1].close
    quantity = round(0.05 * volume, -3)
    # A rate model whose full-day uniform execution costs what the fitted law
    # predicts for the whole order: eta * (X / 1 day) ** 0.5 = Y sigma P sqrt(X / V).
    eta = fit.y.value * sigma * price / volume**0.5
    impact = PowerLawImpact(gamma=0.0, eta=eta, beta=0.5, epsilon=0.0002 * price)
    profile = estimate_profile(
        [bars[SYMBOL]], session_open=time(9, 30), bucket=timedelta(minutes=30), buckets=13
    )
    caps = participation_caps(profile.expected_volume(volume), 0.10)
    plan = solve_schedule(
        quantity=quantity,
        horizon=1.0,
        periods=13,
        volatility=sigma * price,
        impact=impact,
        risk_aversion=3e-7,
        lot_size=1_000.0,
        max_trade=caps,
    )
    print(f"\nrecommended schedule for {quantity:,.0f} {SYMBOL} at a 10% participation cap:")
    print(f"{'bucket':>8}{'expected vol':>14}{'cap':>10}{'trade':>10}")
    start = 9 * 60 + 30
    for k, (v, c, n) in enumerate(
        zip(profile.expected_volume(volume), caps, plan.trades, strict=True)
    ):
        clock = start + 30 * k
        binding = "  cap binds" if c - n < 1_000.0 else ""
        print(f"{clock // 60:>5}:{clock % 60:02d}{v:>14,.0f}{c:>10,.0f}{n:>10,.0f}{binding}")


if __name__ == "__main__":
    main()
