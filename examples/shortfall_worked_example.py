"""Implementation shortfall, decomposed by hand and by the library.

A portfolio manager decides to buy 10,000 shares at 50.00. The order reaches the
desk at 50.10, fills 3,000 at 50.20 and 4,000 at 50.30, and the remaining 3,000
are cancelled with the stock at 50.50. Commission is a cent a share; exchange
fees come to five dollars.

Run with ``python examples/shortfall_worked_example.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from slippage import DelayBasis, Fill, Order, Side, implementation_shortfall

T0 = datetime(2026, 3, 2, 9, 30)

ORDER = Order(
    symbol="ACME",
    side=Side.BUY,
    quantity=10_000,
    decision_time=T0,
    arrival_time=T0 + timedelta(minutes=12),
    fills=(
        Fill(T0 + timedelta(minutes=15), 3_000, 50.20, commission=30.0),
        Fill(T0 + timedelta(minutes=40), 4_000, 50.30, commission=40.0),
    ),
    decision_price=50.00,
)

# The same arithmetic the library performs, written out.
BY_HAND = {
    "delay": 10_000 * (50.10 - 50.00),
    "trading": (3_000 * 50.20 + 4_000 * 50.30) - 7_000 * 50.10,
    "opportunity": 3_000 * (50.50 - 50.10),
    "commission": 70.0,
    "fees": 5.0,
}


def main() -> None:
    result = implementation_shortfall(
        ORDER, arrival_price=50.10, final_price=50.50, fees=5.0, half_spread=0.02
    )
    print(f"{'component':<14}{'by hand':>12}{'library':>12}{'bps':>9}")
    for name, value in result.components().items():
        print(f"{name:<14}{BY_HAND[name]:>12,.2f}{value:>12,.2f}{result.bps(value):>9.2f}")
    print(
        f"{'total':<14}{sum(BY_HAND.values()):>12,.2f}{result.total:>12,.2f}"
        f"{result.total_bps:>9.2f}"
    )
    assert result.spread is not None and result.impact is not None
    print()
    print(
        f"trading cost splits into {result.spread:,.2f} of half-spread "
        f"and {result.impact:,.2f} of impact"
    )

    executed = implementation_shortfall(
        ORDER,
        arrival_price=50.10,
        final_price=50.50,
        fees=5.0,
        delay_basis=DelayBasis.EXECUTED,
    )
    print(
        f"on the executed basis: delay {executed.delay:,.2f}, "
        f"opportunity {executed.opportunity:,.2f}, total {executed.total:,.2f}"
    )


if __name__ == "__main__":
    main()
