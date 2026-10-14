"""Where to put one order when every venue's liquidity is random.

Three venues, each expecting two lots while the order rests, and a buyer who
wants nine. The script shows the three things the module is for.

First, that a posted lot is worth the probability that it is both reached and
needed, and that those are probabilities of different things.

Second, that venues are therefore ranked by their liquidity at the margin and
not by their average liquidity -- a venue that always has exactly three lots
beats one with nearly twice the mean for the first three lots, and loses to it
for the fourth.

Third, that how far past the target to post is a question about the unwind
cost alone, and the answer ranges over more than a factor of two.

Run with ``python examples/venue_routing.py``.
"""

from __future__ import annotations

from slippage import (
    Allocation,
    Outflow,
    RoutingProblem,
    Venue,
    allocation_cost,
    exhaustive_route,
    marginal_posted,
    proportional_allocation,
    route,
)

# Two venues known only by their average liquidity, and one whose history is a
# histogram: it is empty three times in ten and deep the rest of the time.
VENUES = (
    Venue("lit-a", Outflow.geometric(2.0, 8)),
    Venue("lit-b", Outflow.geometric(2.0, 8), rebate=0.10),
    Venue("dark", Outflow.from_counts([30, 0, 0, 0, 0, 0, 0, 0, 70]), rebate=0.05),
)


def problem(unwind: float) -> RoutingProblem:
    return RoutingProblem(
        target=9,
        venues=VENUES,
        taker_cost=1.00,  # half the spread plus the taker fee, per lot
        sweep_cost=2.00,  # crossing later, after the price has moved
        unwind_cost=unwind,
        lot=100.0,  # shares
    )


def main() -> None:
    base = problem(0.50)
    routed = route(base)

    print("nine lots to buy; three venues each expecting two to five lots")
    print()
    header = f"{'venue':<8}{'E[A]':>7}{'posted':>8}{'reached':>9}{'needed':>8}{'next lot':>10}"
    print(header)
    print("-" * len(header))
    for index, venue in enumerate(base.venues):
        lot = marginal_posted(base, routed.allocation, index)
        print(
            f"{venue.name:<8}{venue.outflow.mean:>7.2f}{routed.allocation.posted[index]:>8}"
            f"{lot.reached:>9.3f}{lot.needed:>8.3f}{lot.change:>+10.4f}"
        )
    print("-" * len(header))
    print("reached belongs to the venue; needed belongs to every other one")
    print()

    best = exhaustive_route(base)
    baseline = allocation_cost(base, proportional_allocation(base))
    print(
        f"routed {routed.cost.per_share:+.5f} per share, "
        f"exhaustive {best.cost.per_share:+.5f}, "
        f"gap {routed.cost.total - best.cost.total:+.2e}"
    )
    print(
        f"splitting the nine lots by average liquidity: {baseline.per_share:+.5f} per share, "
        f"{100 * (baseline.per_share / routed.cost.per_share - 1):.1f}% worse"
    )
    print()

    # The ordering flip, on two venues built so that the mean and the margin
    # disagree. One always has exactly three lots; the other has eight lots
    # seven times in ten and nothing otherwise.
    certain = Outflow.from_counts([0, 0, 0, 1])
    occasional = Outflow.from_counts([30, 0, 0, 0, 0, 0, 0, 0, 70])
    flip = RoutingProblem(
        target=8,
        venues=(Venue("certain", certain), Venue("occasional", occasional)),
        taker_cost=1.0,
        sweep_cost=2.0,
        unwind_cost=0.6,
        lot=100.0,
    )
    by_margin = route(flip)
    by_mean = allocation_cost(flip, Allocation(0, (0, 8)))
    print(
        f"certain: mean {certain.mean:.2f}, tail at the first lot {certain.tail(1):.2f}; "
        f"occasional: mean {occasional.mean:.2f}, tail {occasional.tail(1):.2f}"
    )
    print(
        f"by the margin {by_margin.allocation.posted} at {by_margin.cost.per_share:.5f} "
        f"per share; by the mean (0, 8) at {by_mean.per_share:.5f}, "
        f"{100 * (by_mean.per_share / by_margin.cost.per_share - 1):.0f}% more"
    )
    print()

    print("how far past the target to post, against what an over-execution costs")
    print(f"{'unwind':>8}{'committed':>11}{'x target':>10}{'cost/share':>12}")
    for unwind in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        each = problem(unwind)
        answer = route(each)
        print(
            f"{unwind:>8.2f}{answer.allocation.committed:>11}"
            f"{answer.allocation.committed / each.target:>10.2f}"
            f"{answer.cost.per_share:>12.5f}"
        )


if __name__ == "__main__":
    main()
