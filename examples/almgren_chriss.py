"""Optimal liquidation under the Almgren-Chriss (2000) example parameters.

One million shares of a $50 stock over five days, 30% annual volatility, a
1/8 spread and impact calibrated to five million shares of daily volume.

Run with ``python examples/almgren_chriss.py``.
"""

from __future__ import annotations

from slippage import ExecutionProblem, LinearImpact, half_life_sensitivity, optimal_trajectory

PROBLEM = ExecutionProblem(
    quantity=1_000_000,
    horizon=5.0,  # days
    periods=5,
    volatility=0.95,  # $/share/day^0.5
    impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.0625),
)


def main() -> None:
    print(f"{'lambda':>8} {'kappa':>7} {'half-life':>10} {'E[cost]':>10} {'sd':>10} {'day 1':>8}")
    for lam in (0.0, 1e-7, 1e-6, 1e-5):
        t = optimal_trajectory(PROBLEM, lam)
        print(
            f"{lam:>8g} {PROBLEM.kappa(lam):>7.3f} {t.half_life:>9.2f}d "
            f"{t.expected_cost:>10,.0f} {t.std:>10,.0f} {t.trades[0]:>8,.0f}"
        )
    s = half_life_sensitivity(PROBLEM, 1e-6)
    print()
    print(f"half-life elasticities at lambda = 1e-6 (half-life {s.half_life:.2f} days):")
    print(f"  risk aversion {s.risk_aversion:+.3f}   volatility {s.volatility:+.3f}")
    print(f"  eta           {s.eta:+.3f}   gamma      {s.gamma:+.3f}")


if __name__ == "__main__":
    main()
