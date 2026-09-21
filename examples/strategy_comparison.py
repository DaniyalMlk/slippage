"""Compare TWAP, VWAP, POV and the optimal schedule by simulation.

Sixty days of synthetic history with a U-shaped intraday volume curve give a
volume profile; four schedules sell 500,000 shares over one session in
half-hour buckets; the simulator prices each under the same linear impact
model and the same random price paths.

The impact model assumes liquidity is constant through the day, as Almgren
and Chriss do. That is exactly the assumption VWAP exists to exploit: VWAP
concentrates trading at the open and close because that is when the market can
absorb it. Under a volume-blind model the same concentration looks like pure
cost, and the comparison below shows it. Read VWAP's row as the price of that
assumption, not as a verdict on VWAP.

Run with ``python examples/strategy_comparison.py``.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np

from slippage import (
    Bar,
    BarSeries,
    ExecutionProblem,
    LinearImpact,
    estimate_profile,
    optimal_trajectory,
    pov_schedule,
    simulate_costs,
    twap_schedule,
    vwap_schedule,
)

OPEN = time(9, 30)
BUCKET = timedelta(minutes=30)
TRUE_SHAPE = np.array([12, 8, 6, 5, 4.5, 4, 4, 4, 4.5, 5, 6, 9, 14], dtype=float)
TRUE_SHAPE /= TRUE_SHAPE.sum()
DAILY_VOLUME = 5e6


def history(rng: np.random.Generator, days: int = 60) -> list[BarSeries]:
    sessions = []
    for d in range(days):
        shape = TRUE_SHAPE * rng.lognormal(0.0, 0.25, TRUE_SHAPE.size)
        shape /= shape.sum()
        start = datetime.combine(date(2026, 1, 5) + timedelta(days=d), OPEN)
        sessions.append(
            BarSeries(
                Bar(start + BUCKET * b, 50.0, 50.1, 49.9, 50.0, DAILY_VOLUME * f)
                for b, f in enumerate(shape)
            )
        )
    return sessions


def main() -> None:
    rng = np.random.default_rng(2026)
    profile = estimate_profile(history(rng), session_open=OPEN, bucket=BUCKET, buckets=13)

    # One session is the horizon; time is measured in days.
    problem = ExecutionProblem(
        quantity=500_000,
        horizon=1.0,
        periods=13,
        volatility=0.95,
        impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.0625),
    )
    pov = pov_schedule(500_000, profile.expected_volume(DAILY_VOLUME), 0.12)
    schedules = {
        "TWAP": twap_schedule(500_000, 13),
        "VWAP": vwap_schedule(500_000, profile),
        "POV 12%": list(pov.trades),
        "optimal": list(optimal_trajectory(problem, 1e-6).trades),
    }

    print(f"{'strategy':<9}{'mean':>10}{'sd':>10}{'95th pct':>11}{'unfilled':>10}")
    for name, trades in schedules.items():
        dist = simulate_costs(
            problem.impact,
            trades,
            tau=problem.tau,
            volatility=problem.volatility,
            paths=50_000,
            rng=np.random.default_rng(7),
            antithetic=True,
        )
        unfilled = max(problem.quantity - sum(trades), 0.0)
        print(
            f"{name:<9}{dist.mean:>10,.0f}{dist.std:>10,.0f}"
            f"{dist.quantile(0.95):>11,.0f}{unfilled:>10,.0f}"
        )


if __name__ == "__main__":
    main()
