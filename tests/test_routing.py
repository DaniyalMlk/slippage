"""Routing one order across venues with random available liquidity.

The tests that matter here are the ones with an independent reference. The
fill distribution is checked against brute-force enumeration over the product
of the venues' supports; the closed-form marginal against the difference of
two exactly computed costs; and the router against an exhaustive search over
every integer allocation. Nothing in this file compares the module to itself.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from slippage.exceptions import ValidationError
from slippage.routing import (
    MAX_VENUES,
    Allocation,
    FillDistribution,
    Outflow,
    RoutingProblem,
    Venue,
    allocation_cost,
    compare_to_proportional,
    exhaustive_route,
    fill_distribution,
    marginal_cross,
    marginal_posted,
    proportional_allocation,
    route,
    venue_fill,
)


def _problem(**overrides: object) -> RoutingProblem:
    defaults: dict[str, object] = {
        "target": 7,
        "venues": (
            Venue("A", Outflow.uniform(4), rebate=0.2),
            Venue("B", Outflow.geometric(1.5, 6)),
            Venue("C", Outflow.from_counts([2.0, 1.0, 1.0, 1.0, 3.0, 1.0])),
        ),
        "taker_cost": 1.0,
        "sweep_cost": 2.2,
        "unwind_cost": 0.4,
    }
    defaults.update(overrides)
    return RoutingProblem(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Outflow
# --------------------------------------------------------------------------


def test_outflow_rejects_what_cannot_be_a_distribution() -> None:
    with pytest.raises(ValidationError, match="at least one probability"):
        Outflow(())
    with pytest.raises(ValidationError, match="non-negative"):
        Outflow((1.2, -0.2))
    with pytest.raises(ValidationError, match="sum to 1"):
        Outflow((0.3, 0.3))
    with pytest.raises(ValidationError, match="finite"):
        Outflow((math.inf, 0.0))


def test_support_ignores_trailing_zeros() -> None:
    # The cap on the search has to be the last lot that can actually arrive,
    # not the length of whatever array the caller passed.
    assert Outflow((0.5, 0.5, 0.0, 0.0)).support == 1
    assert Outflow.point(0).support == 0


def test_tail_is_one_at_zero_and_zero_past_the_support() -> None:
    outflow = Outflow.uniform(3)
    assert outflow.tail(0) == 1.0
    assert outflow.tail(4) == 0.0
    assert outflow.tail(3) == pytest.approx(0.25)
    with pytest.raises(ValidationError, match="non-negative"):
        outflow.tail(-1)


def test_tail_survives_where_one_minus_the_cdf_does_not() -> None:
    # A distribution whose top atom is far below the resolution of 1.0. The
    # complement of the cumulative probability is exactly zero there; the tail
    # summed from the top is the atom itself.
    tiny = 1e-20
    outflow = Outflow((1.0 - tiny, tiny))
    assert outflow.tail(1) == pytest.approx(tiny, rel=1e-12)
    assert 1.0 - outflow.probabilities[0] == 0.0


def test_expected_fill_matches_the_definition() -> None:
    rng = random.Random(11)
    for _ in range(200):
        size = rng.randint(1, 9)
        outflow = Outflow.from_counts([rng.random() for _ in range(size + 1)])
        for posted in range(size + 3):
            direct = math.fsum(
                min(posted, lots) * probability
                for lots, probability in enumerate(outflow.probabilities)
            )
            assert outflow.expected_fill(posted) == pytest.approx(direct, abs=1e-14)


def test_geometric_matches_its_requested_mean_after_truncation() -> None:
    for support in (2, 5, 20):
        for mean in (0.2, 0.7, support / 2.0 - 0.05):
            outflow = Outflow.geometric(mean, support)
            assert outflow.mean == pytest.approx(mean, abs=1e-9)
            assert outflow.support == support


def test_geometric_refuses_a_mean_the_support_cannot_carry() -> None:
    # The truncated mean tops out at support/2, reached only in the uniform
    # limit. Asking for more is not a convergence problem, it is impossible.
    with pytest.raises(ValidationError, match="must lie in"):
        Outflow.geometric(2.5, 4)
    with pytest.raises(ValidationError, match="must lie in"):
        Outflow.geometric(0.0, 4)
    with pytest.raises(ValidationError, match="support of at least one lot"):
        Outflow.geometric(0.5, 0)
    assert Outflow.geometric(1.99, 4).mean == pytest.approx(1.99, abs=1e-9)


def test_from_counts_normalises_and_refuses_an_empty_histogram() -> None:
    outflow = Outflow.from_counts([3.0, 1.0])
    assert outflow.probabilities == pytest.approx((0.75, 0.25))
    with pytest.raises(ValidationError, match="all zero"):
        Outflow.from_counts([0.0, 0.0])
    with pytest.raises(ValidationError, match="at least one count"):
        Outflow.from_counts([])


# --------------------------------------------------------------------------
# The fill distribution, against brute force
# --------------------------------------------------------------------------


def _brute_force_fill(problem: RoutingProblem, allocation: Allocation) -> list[float]:
    """Enumerate the product of the venues' supports. The independent reference."""
    supports = [venue.outflow.probabilities for venue in problem.venues]
    masses: dict[int, float] = {}
    for outcome in itertools.product(*[range(len(support)) for support in supports]):
        probability = 1.0
        for support, lots in zip(supports, outcome, strict=True):
            probability *= support[lots]
        if probability == 0.0:
            continue
        filled = sum(min(allocation.posted[index], lots) for index, lots in enumerate(outcome))
        masses[filled] = masses.get(filled, 0.0) + probability
    return [masses.get(lots, 0.0) for lots in range(max(masses) + 1)]


def test_fill_distribution_agrees_with_brute_force_enumeration() -> None:
    rng = random.Random(12)
    worst = 0.0
    for _ in range(300):
        problem = _problem(
            venues=tuple(
                Venue(
                    chr(65 + index),
                    Outflow.from_counts([rng.random() for _ in range(rng.randint(2, 6))]),
                )
                for index in range(rng.randint(1, 3))
            )
        )
        allocation = Allocation(
            taker=rng.randint(0, problem.target),
            posted=tuple(rng.randint(0, cap + 1) for cap in problem.caps),
        )
        computed = fill_distribution(problem, allocation).probabilities
        reference = _brute_force_fill(problem, allocation)
        padded = [*reference, *([0.0] * (len(computed) - len(reference)))]
        for left, right in zip(computed, padded, strict=True):
            worst = max(worst, abs(left - right))
    assert worst < 1e-15, worst


def test_venue_fill_piles_the_tail_onto_the_posted_quantity() -> None:
    outflow = Outflow.from_counts([1.0, 1.0, 1.0, 1.0, 1.0])
    filled = venue_fill(outflow, 2)
    assert filled.probabilities == pytest.approx((0.2, 0.2, 0.6))
    assert venue_fill(outflow, 0).probabilities == (1.0,)
    # Posting past the support cannot change the distribution.
    assert venue_fill(outflow, 9).probabilities == pytest.approx(
        venue_fill(outflow, 4).probabilities
    )


def test_one_sided_expectations_match_their_definitions() -> None:
    rng = random.Random(13)
    for _ in range(200):
        weights = [rng.random() for _ in range(rng.randint(1, 10))]
        total = math.fsum(weights)
        distribution = FillDistribution(tuple(weight / total for weight in weights))
        for level in range(len(weights) + 2):
            shortfall = math.fsum(
                max(level - lots, 0) * probability
                for lots, probability in enumerate(distribution.probabilities)
            )
            excess = math.fsum(
                max(lots - level, 0) * probability
                for lots, probability in enumerate(distribution.probabilities)
            )
            assert distribution.expected_shortfall(level) == pytest.approx(shortfall, abs=1e-14)
            assert distribution.expected_excess(level) == pytest.approx(excess, abs=1e-14)


def test_shortfall_and_excess_differ_by_the_mean() -> None:
    # E[(q-F)^+] - E[(F-q)^+] = q - E[F], an identity with nothing of the
    # module's in it, so it checks both one-sided sums at once.
    rng = random.Random(14)
    for _ in range(300):
        weights = [rng.random() for _ in range(rng.randint(1, 12))]
        total = math.fsum(weights)
        distribution = FillDistribution(tuple(weight / total for weight in weights))
        level = rng.randint(0, 14)
        difference = distribution.expected_shortfall(level) - distribution.expected_excess(level)
        assert difference == pytest.approx(level - distribution.mean, abs=1e-13)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_cost_components_add_to_the_total() -> None:
    problem = _problem()
    cost = allocation_cost(problem, Allocation(2, (3, 2, 1)))
    assert cost.total == pytest.approx(cost.cross + cost.rebate + cost.sweep + cost.unwind)
    assert cost.per_share == pytest.approx(cost.total / problem.target)


def test_crossing_the_whole_target_costs_exactly_the_cross() -> None:
    problem = _problem()
    cost = allocation_cost(problem, Allocation(problem.target, (0, 0, 0)))
    assert cost.sweep == 0.0
    assert cost.unwind == 0.0
    assert cost.total == pytest.approx(problem.taker_cost * problem.target)
    assert cost.completion == pytest.approx(1.0)


def test_doing_nothing_costs_the_whole_sweep() -> None:
    problem = _problem()
    cost = allocation_cost(problem, Allocation(0, (0, 0, 0)))
    assert cost.total == pytest.approx(problem.sweep_cost * problem.target)
    assert cost.completion == 0.0


def test_crossing_past_the_target_over_executes_before_anything_fills() -> None:
    problem = _problem(unwind_cost=0.5)
    cost = allocation_cost(problem, Allocation(problem.target + 3, (0, 0, 0)))
    assert cost.expected_shortfall == 0.0
    assert cost.expected_excess == pytest.approx(3.0)
    assert cost.total == pytest.approx(problem.taker_cost * (problem.target + 3) + 0.5 * 3.0)


def test_an_allocation_of_the_wrong_width_is_an_error() -> None:
    problem = _problem()
    with pytest.raises(ValidationError, match="covers 2 venues"):
        allocation_cost(problem, Allocation(0, (1, 1)))


# --------------------------------------------------------------------------
# The marginal, against differenced costs
# --------------------------------------------------------------------------


def test_marginal_posted_agrees_with_the_differenced_cost() -> None:
    rng = random.Random(15)
    worst = 0.0
    for _ in range(400):
        problem = _problem(
            target=rng.randint(1, 9),
            taker_cost=rng.uniform(0.1, 2.0),
            sweep_cost=rng.uniform(0.1, 4.0),
            unwind_cost=rng.uniform(0.0, 2.0),
        )
        allocation = Allocation(
            taker=rng.randint(0, problem.target),
            posted=tuple(rng.randint(0, cap) for cap in problem.caps),
        )
        base = allocation_cost(problem, allocation).total
        for index in range(len(problem.venues)):
            closed = marginal_posted(problem, allocation, index).change
            differenced = allocation_cost(problem, allocation.with_lot_at(index, 1)).total - base
            worst = max(worst, abs(closed - differenced))
        crossed = marginal_cross(problem, allocation).change
        differenced = (
            allocation_cost(problem, Allocation(allocation.taker + 1, allocation.posted)).total
            - base
        )
        worst = max(worst, abs(crossed - differenced))
    assert worst < 1e-12, worst


def test_the_marginal_factorises_the_way_the_closed_form_says() -> None:
    problem = _problem()
    allocation = Allocation(1, (2, 3, 1))
    for index, venue in enumerate(problem.venues):
        marginal = marginal_posted(problem, allocation, index)
        assert marginal.reached == pytest.approx(venue.outflow.tail(allocation.posted[index] + 1))
        assert marginal.rebate == pytest.approx(-venue.rebate * marginal.reached)
        bracket = (
            -venue.rebate
            - problem.sweep_cost * marginal.needed
            + problem.unwind_cost * (1.0 - marginal.needed)
        )
        assert marginal.change == pytest.approx(marginal.reached * bracket)


def test_a_crossed_lot_is_reached_with_certainty() -> None:
    marginal = marginal_cross(_problem(), Allocation(0, (0, 0, 0)))
    assert marginal.reached == 1.0
    assert marginal.rebate == pytest.approx(_problem().taker_cost)


def test_a_lot_past_the_support_is_worth_exactly_nothing() -> None:
    problem = _problem()
    allocation = Allocation(0, problem.caps)
    for index in range(len(problem.venues)):
        marginal = marginal_posted(problem, allocation, index)
        assert marginal.reached == 0.0
        assert marginal.change == 0.0


def test_marginal_posted_checks_its_index() -> None:
    with pytest.raises(ValidationError, match="out of range"):
        marginal_posted(_problem(), Allocation(0, (0, 0, 0)), 3)


# --------------------------------------------------------------------------
# The router, against exhaustive search
# --------------------------------------------------------------------------


def _random_problem(rng: random.Random, venues: int, support: int, target: int) -> RoutingProblem:
    chosen: list[Venue] = []
    for index in range(rng.randint(1, venues)):
        size = rng.randint(1, support)
        kind = rng.choice(("uniform", "geometric", "counts", "point"))
        if kind == "uniform":
            outflow = Outflow.uniform(size)
        elif kind == "geometric" and size >= 2:
            outflow = Outflow.geometric(rng.uniform(0.1, size / 2.0 - 0.01), size)
        elif kind == "point":
            outflow = Outflow.point(size)
        else:
            outflow = Outflow.from_counts([rng.random() for _ in range(size + 1)])
        chosen.append(
            Venue(chr(65 + index), outflow, rebate=rng.choice((0.0, 0.0, rng.uniform(-0.1, 0.3))))
        )
    taker = rng.uniform(0.2, 2.0)
    return RoutingProblem(
        target=rng.randint(1, target),
        venues=tuple(chosen),
        taker_cost=taker,
        sweep_cost=taker * rng.uniform(0.5, 3.0),
        unwind_cost=rng.uniform(0.0, 2.0),
    )


@pytest.mark.parametrize(
    ("seed", "venues", "support", "target", "trials"),
    [(20261014, 3, 5, 8, 500), (77, 4, 4, 6, 400), (991, 2, 8, 10, 400)],
)
def test_the_default_radius_finds_the_exhaustive_optimum_every_time(
    seed: int, venues: int, support: int, target: int, trials: int
) -> None:
    rng = random.Random(seed)
    checked = 0
    for _ in range(trials):
        problem = _random_problem(rng, venues, support, target)
        try:
            best = exhaustive_route(problem, max_states=600_000)
        except ValidationError:
            continue
        checked += 1
        routed = route(problem)
        assert routed.cost.total <= best.cost.total + 1e-12, (
            problem,
            routed.allocation,
            best.allocation,
        )
    assert checked > trials // 2


def test_a_one_lot_neighbourhood_is_measurably_not_enough() -> None:
    # The claim in the module docstring, pinned as a band rather than a digit:
    # a single-lot neighbourhood misses optima, and the default does not.
    rng = random.Random(20261014)
    narrow_misses = 0
    checked = 0
    worst_narrow = 0.0
    for _ in range(600):
        problem = _random_problem(rng, 3, 5, 8)
        try:
            best = exhaustive_route(problem, max_states=600_000)
        except ValidationError:
            continue
        checked += 1
        scale = problem.sweep_cost * problem.target
        narrow = route(problem, radius=1).cost.total - best.cost.total
        if narrow > 1e-12:
            narrow_misses += 1
            worst_narrow = max(worst_narrow, narrow / scale)
        assert route(problem).cost.total <= best.cost.total + 1e-12
    rate = narrow_misses / checked
    assert 0.005 < rate < 0.10, rate
    assert 0.01 < worst_narrow < 0.5, worst_narrow


def test_the_coordinated_direction_is_what_makes_a_small_radius_work() -> None:
    # A venue with certain liquidity holds the whole target at a strict
    # single-lot local optimum while two rebate-paying venues sit empty. Every
    # one-lot move away is worse; the two-lot move is better.
    problem = RoutingProblem(
        target=5,
        venues=(
            Venue("A", Outflow((0.44552184144964074, 0.5544781585503593)), rebate=0.2676440581),
            Venue("B", Outflow((0.0, 0.0, 0.0, 0.0, 0.0, 1.0))),
            Venue("C", Outflow((1 / 3, 1 / 3, 1 / 3)), rebate=0.2866128008),
        ),
        taker_cost=0.827,
        sweep_cost=0.641,
        unwind_cost=0.380,
    )
    trapped = Allocation(0, (0, 5, 0))
    assert allocation_cost(problem, trapped).total == pytest.approx(0.0, abs=1e-12)
    for single in ((1, 4, 0), (0, 4, 1), (1, 5, 0), (0, 5, 1)):
        assert allocation_cost(problem, Allocation(0, single)).total > 0.0
    assert allocation_cost(problem, Allocation(0, (1, 4, 1))).total < -0.1
    assert route(problem).cost.total == pytest.approx(
        exhaustive_route(problem).cost.total, abs=1e-12
    )


def test_route_rejects_a_radius_below_one() -> None:
    with pytest.raises(ValidationError, match="at least one lot"):
        route(_problem(), radius=0)


def test_exhaustive_route_refuses_rather_than_hangs() -> None:
    problem = _problem(venues=tuple(Venue(chr(65 + i), Outflow.uniform(20)) for i in range(5)))
    with pytest.raises(ValidationError, match="exceeds max_states"):
        exhaustive_route(problem, max_states=1000)


def test_routing_reports_the_venue_it_posted_at_by_name() -> None:
    routed = route(_problem())
    assert routed.venue_names == ("A", "B", "C")
    assert routed.posted_at("B") == routed.allocation.posted[1]
    with pytest.raises(ValidationError, match="no venue named"):
        routed.posted_at("Z")


# --------------------------------------------------------------------------
# The findings in the module docstring
# --------------------------------------------------------------------------


def test_average_liquidity_is_the_wrong_ranking_and_the_cost_is_sixty_per_cent() -> None:
    # One venue always has exactly three lots; the other has eight lots seven
    # times in ten and nothing otherwise. The second has the higher mean and
    # the first is where the first lot belongs.
    deep = Outflow.from_counts([0.0, 0.0, 0.0, 1.0])
    thin = Outflow.from_counts([30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.0])
    assert thin.mean > deep.mean
    assert deep.tail(1) > thin.tail(1)
    problem = RoutingProblem(
        target=8,
        venues=(Venue("deep", deep), Venue("thin", thin)),
        taker_cost=1.0,
        sweep_cost=2.0,
        unwind_cost=0.6,
    )
    routed = route(problem)
    assert routed.allocation == Allocation(0, (3, 5))
    assert routed.cost.per_share == pytest.approx(0.375, abs=1e-9)
    by_mean = allocation_cost(problem, Allocation(0, (0, 8)))
    assert by_mean.per_share == pytest.approx(0.600, abs=1e-9)
    assert by_mean.per_share / routed.cost.per_share == pytest.approx(1.6, abs=1e-6)


def test_how_much_to_over_post_falls_as_the_unwind_rises() -> None:
    committed: list[int] = []
    for unwind in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        problem = RoutingProblem(
            target=9,
            venues=tuple(Venue(f"V{index}", Outflow.geometric(2.0, 8)) for index in range(3)),
            taker_cost=1.0,
            sweep_cost=2.0,
            unwind_cost=unwind,
        )
        committed.append(route(problem).allocation.committed)
    assert committed == [21, 18, 18, 15, 12, 10]
    # Monotone, and the end points are the claim: 2.33 times the target when
    # over-execution is free, 1.11 times when it costs twice the sweep.
    assert all(later <= earlier for earlier, later in itertools.pairwise(committed))
    assert committed[0] / 9 == pytest.approx(21 / 9, abs=1e-9)


def test_fragmentation_is_cheaper_with_the_liquidity_held_fixed() -> None:
    costs: dict[int, float] = {}
    for count in (1, 2, 3, 6):
        mean = 6.0 / count
        support = max(2, 24 // count)
        venues = tuple(
            Venue(f"V{index}", Outflow.geometric(mean, support)) for index in range(count)
        )
        assert math.fsum(venue.outflow.mean for venue in venues) == pytest.approx(6.0, abs=1e-8)
        problem = RoutingProblem(
            target=9, venues=venues, taker_cost=1.0, sweep_cost=2.0, unwind_cost=0.5
        )
        costs[count] = route(problem).cost.per_share
    assert costs[1] > costs[2] > costs[3] > costs[6]
    assert costs[1] == pytest.approx(0.81306, abs=5e-5)
    assert costs[6] == pytest.approx(0.64873, abs=5e-5)
    assert 1.0 - costs[6] / costs[1] == pytest.approx(0.202, abs=5e-3)


def test_the_proportional_baseline_costs_a_quarter_to_a_half_more() -> None:
    rng = random.Random(5)
    savings: list[float] = []
    already_optimal = 0
    for _ in range(400):
        count = rng.randint(2, 4)
        venues = []
        for index in range(count):
            support = rng.randint(2, 6)
            outflow = rng.choice(
                (
                    Outflow.uniform(support),
                    Outflow.geometric(rng.uniform(0.3, support / 2.0 - 0.01), support),
                    Outflow.from_counts([rng.random() for _ in range(support + 1)]),
                )
            )
            venues.append(
                Venue(chr(65 + index), outflow, rebate=rng.choice((0.0, rng.uniform(0.0, 0.2))))
            )
        taker = rng.uniform(0.3, 2.0)
        problem = RoutingProblem(
            target=rng.randint(3, 10),
            venues=tuple(venues),
            taker_cost=taker,
            sweep_cost=taker * rng.uniform(1.0, 3.0),
            unwind_cost=rng.uniform(0.1, 1.5),
        )
        comparison = compare_to_proportional(problem)
        assert comparison.saving >= -1e-12
        savings.append(comparison.fraction)
        if comparison.fraction <= 1e-12:
            already_optimal += 1
    savings.sort()
    median = savings[len(savings) // 2]
    assert 0.20 < median < 0.40, median
    assert 0.0 < already_optimal / len(savings) < 0.10


def test_proportional_allocation_posts_the_target_and_respects_the_caps() -> None:
    problem = _problem(target=6)
    allocation = proportional_allocation(problem)
    assert allocation.taker == 0
    assert allocation.total_posted == 6
    assert all(lots <= cap for lots, cap in zip(allocation.posted, problem.caps, strict=True))
    # A target larger than the whole routing table can hold posts everything.
    tight = _problem(target=12, venues=(Venue("A", Outflow.uniform(2)),))
    assert proportional_allocation(tight).posted == (2,)


def test_a_comparison_against_a_free_baseline_refuses_to_take_a_fraction() -> None:
    problem = _problem(taker_cost=0.0, sweep_cost=0.0, unwind_cost=0.0)
    comparison = compare_to_proportional(problem)
    assert comparison.proportional.total <= 0.0
    if comparison.proportional.per_share == 0.0:
        with pytest.raises(ValidationError, match="costs nothing"):
            _ = comparison.fraction


# --------------------------------------------------------------------------
# Problem validation
# --------------------------------------------------------------------------


def test_routing_problem_validation() -> None:
    venue = Venue("A", Outflow.uniform(3))
    with pytest.raises(ValidationError, match="at least one lot"):
        _problem(target=0)
    with pytest.raises(ValidationError, match="at least one venue"):
        _problem(venues=())
    with pytest.raises(ValidationError, match="must be distinct"):
        _problem(venues=(venue, Venue("A", Outflow.uniform(2))))
    with pytest.raises(ValidationError, match="sweep_cost must be non-negative"):
        _problem(sweep_cost=-1.0)
    with pytest.raises(ValidationError, match="lot must be positive"):
        _problem(lot=0.0)
    with pytest.raises(ValidationError, match=r"at most 12 venues"):
        _problem(
            venues=tuple(Venue(f"V{index}", Outflow.uniform(2)) for index in range(MAX_VENUES + 1))
        )


def test_a_lot_size_only_changes_the_per_share_figure() -> None:
    allocation = Allocation(2, (2, 2, 2))
    one = allocation_cost(_problem(), allocation)
    hundred = allocation_cost(_problem(lot=100.0), allocation)
    assert hundred.total == pytest.approx(one.total)
    assert hundred.per_share == pytest.approx(one.per_share / 100.0)


def test_venue_validation() -> None:
    with pytest.raises(ValidationError, match="needs a name"):
        Venue("", Outflow.uniform(2))
    with pytest.raises(ValidationError, match="rebate must be finite"):
        Venue("A", Outflow.uniform(2), rebate=math.nan)
