"""A schedule under a participation cap and a power-law impact model.

The closed-form trajectory front-loads the order. Capping participation at 5%
of each period's expected volume forces it to spread out, and the dynamic
programme finds the best schedule that respects the cap. The impact model is a
square-root law in trading rate, which the closed form cannot handle at all.

Run with ``python examples/constrained_schedule.py``.
"""

from __future__ import annotations

from slippage import PowerLawImpact, participation_caps, solve_schedule

# Ten half-day periods over five days; volume is heavier at the open and close
# of each day.
EXPECTED_VOLUME = [3.0e6, 2.0e6] * 5
IMPACT = PowerLawImpact(gamma=2.5e-7, eta=1.2e-3, beta=0.5, epsilon=0.0625)


def main() -> None:
    common = {
        "quantity": 1_000_000.0,
        "horizon": 5.0,
        "periods": 10,
        "volatility": 0.95,
        "impact": IMPACT,
        "risk_aversion": 1e-6,
        "lot_size": 1_000.0,
    }
    free = solve_schedule(**common)  # type: ignore[arg-type]
    caps = participation_caps(EXPECTED_VOLUME, 0.05)
    capped = solve_schedule(**common, max_trade=caps)  # type: ignore[arg-type]

    print(f"{'period':>6} {'volume':>10} {'cap':>9} {'free':>9} {'capped':>9}")
    for k, (v, c, a, b) in enumerate(
        zip(EXPECTED_VOLUME, caps, free.trades, capped.trades, strict=True)
    ):
        print(f"{k:>6} {v:>10,.0f} {c:>9,.0f} {a:>9,.0f} {b:>9,.0f}")
    print(f"\nobjective E + lambda V: free {free.objective:,.0f}, capped {capped.objective:,.0f}")

    # Suppose the first two periods fill only 100,000 shares between them.
    left = 900_000.0
    print("\nafter falling behind, the policy's catch-up from period 2:")
    print([f"{n:,.0f}" for n in capped.trades_from(2, left)])


if __name__ == "__main__":
    main()
