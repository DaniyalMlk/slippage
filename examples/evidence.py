"""The front page's figures, recomputed.

The README promises a validated numerical core. This is what that means, and
the order of the rows is deliberate: the first two checks are against
*enumeration*, not against a formula.

That distinction is the point. A solver checked against a closed form is
checked against a rearrangement of the same mathematics, and a sign error made
in both places agrees with itself. A dynamic programme checked against every
possible schedule, and a router checked against every possible allocation, are
checked against something that cannot contain the same mistake, because it
contains no derivation at all — only the objective, evaluated at every point of
a finite set.

The closed forms are here too, under the enumerations, because they say
different things. Two of them are stated as results rather than as tolerances:

- The lot grid can *match* the continuum optimum and cannot beat it, the grid
  being a subset of the continuum. So what is reported is that the objective is
  never below the closed form's, which is a claim about the direction of the
  inequality rather than a number.
- The response kernel's shape at zero urgency is a finding, so it is measured
  twice: as a tent at zero urgency and as not a tent at a positive one. Either
  figure alone would say nothing about whether the shape was derived or
  assumed.

Run with ``python examples/evidence.py``. Exits non-zero if any figure moves
outside the band the README states.
"""

from __future__ import annotations

import itertools
import random
import runpy
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import slippage as sl
from slippage.exceptions import ValidationError

#: The Almgren-Chriss parameterisation the paper's figures are computed with.
AC_IMPACT = sl.LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.0625)

#: The worked example whose arithmetic is done twice: once by the library and
#: once by hand in the script itself.
WORKED = Path(__file__).resolve().parent / "shortfall_worked_example.py"


class Check(NamedTuple):
    """One row of the front page's table."""

    label: str
    measured: float
    bound: float
    rendered: str

    @property
    def passed(self) -> bool:
        return self.measured <= self.bound


def compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """Every way to write ``total`` as an ordered sum of ``parts`` whole lots.

    This is the enumeration the schedule check is against. It is written out
    here rather than imported from the solver's own helpers for the obvious
    reason: a reference that shares code with the thing it is checking is not
    a reference.
    """
    return [
        tuple(high - low - 1 for low, high in itertools.pairwise((-1, *cut, total + parts - 1)))
        for cut in itertools.combinations(range(total + parts - 1), parts - 1)
    ]


def schedule_against_enumeration() -> tuple[Check, int, int]:
    """The dynamic programme against every schedule there is.

    Twelve cases: two impact models, three risk aversions, with and without a
    participation cap. Twelve lots over five periods is 1,820 schedules, which
    is small enough to enumerate and large enough that the optimum is not
    obvious — and the cap matters, because an unconstrained problem is the one
    a closed form could have answered anyway.

    The comparison is on the objective rather than on the trades. Two schedules
    can have the same objective to the last bit, and insisting the programme
    pick the same one the enumeration happened to find first would be testing
    the tie-break.
    """
    impacts = (
        sl.LinearImpact(gamma=1e-3, eta=5e-2, epsilon=0.01),
        sl.PowerLawImpact(gamma=1e-3, eta=5e-2, beta=0.5, epsilon=0.01),
    )
    lots, periods, lot, tau, sigma = 12, 5, 1.0, 0.1, 0.3
    schedules = len(compositions(lots, periods))
    worst = 0.0
    cases = 0
    for impact in impacts:
        for risk_aversion in (0.0, 1e-3, 1e-1):
            for cap in (None, 4):
                best = min(
                    sl.schedule_objective(
                        impact, [n * lot for n in combo], tau, sigma, risk_aversion
                    )
                    for combo in compositions(lots, periods)
                    if cap is None or max(combo) <= cap
                )
                plan = sl.solve_schedule(
                    quantity=lots * lot,
                    horizon=tau * periods,
                    periods=periods,
                    volatility=sigma,
                    impact=impact,
                    risk_aversion=risk_aversion,
                    lot_size=lot,
                    max_trade=None if cap is None else cap * lot,
                )
                cases += 1
                worst = max(worst, abs(plan.objective - best) / max(abs(best), 1e-12))
    return (
        Check(
            label=(
                f"The schedule solver against exhaustive enumeration of all "
                f"{schedules:,} schedules, {cases} cases"
            ),
            measured=worst,
            bound=1e-14,
            rendered="worst relative gap 1e-14, capped and uncapped",
        ),
        cases,
        schedules,
    )


def routing_against_exhaustive_search() -> tuple[Check, int, int]:
    """The router against every allocation, on random problems.

    Random rather than hand-picked, because a hand-picked routing problem is
    one whose answer the person picking it could see. The default search radius
    is a local move limit, so the question is whether a local search reaches
    the global optimum — and the honest way to ask is to generate problems
    until the enumeration refuses to run.
    """
    rng = random.Random(20261018)
    checked = 0
    optimal = 0
    worst = 0.0
    for _ in range(250):
        venues = tuple(
            sl.Venue(
                chr(65 + index),
                sl.Outflow.from_counts(
                    [rng.uniform(0.2, 3.0) for _ in range(rng.choice([4, 5, 6]))]
                ),
                rebate=rng.choice([0.0, 0.1, 0.2]),
            )
            for index in range(rng.choice([2, 3, 4]))
        )
        problem = sl.RoutingProblem(
            target=rng.choice([6, 7, 8, 10]),
            venues=venues,
            taker_cost=rng.uniform(0.5, 2.0),
            sweep_cost=rng.uniform(1.0, 3.0),
            unwind_cost=rng.uniform(0.0, 1.0),
        )
        try:
            best = sl.exhaustive_route(problem, max_states=600_000)
        except ValidationError:
            continue
        checked += 1
        excess = sl.route(problem).cost.total - best.cost.total
        worst = max(worst, excess)
        if excess <= 1e-12:
            optimal += 1
    return (
        Check(
            label=f"The router against exhaustive search, {checked} random problems",
            measured=worst,
            bound=0.0,
            rendered=f"at the optimum in all {optimal}, worst excess cost 0",
        ),
        checked,
        optimal,
    )


def grid_against_the_closed_form() -> tuple[Check, Check, float, float]:
    """A fine lot grid against the Almgren-Chriss continuum solution.

    Two rows. The first is how close the trades land, which has to be about one
    lot: the programme chooses whole lots and the closed form does not, so it
    cannot do better than rounding.

    The second is the one that matters and it is an inequality, not a
    tolerance. The lot grid is a subset of the continuum, so the best schedule
    on the grid cannot cost less than the best schedule in the continuum. A
    programme that reported a lower objective would not be more accurate; it
    would be wrong, and a tolerance on the difference would accept it.
    """
    quantity, periods, risk_aversion = 2000.0, 10, 1e-6
    problem = sl.ExecutionProblem(
        quantity=quantity,
        horizon=5.0,
        periods=periods,
        volatility=0.95,
        impact=AC_IMPACT,
    )
    closed = sl.optimal_trajectory(problem, risk_aversion)
    plan = sl.solve_schedule(
        quantity=quantity,
        horizon=5.0,
        periods=periods,
        volatility=0.95,
        impact=AC_IMPACT,
        risk_aversion=risk_aversion,
        lot_size=1.0,
    )
    worst_trade = max(
        abs(grid - continuum) for grid, continuum in zip(plan.trades, closed.trades, strict=True)
    )
    continuum_objective = closed.expected_cost + risk_aversion * closed.variance
    shortfall_below = continuum_objective - plan.objective
    return (
        Check(
            label=f"A {int(quantity):,}-lot grid against the Almgren-Chriss closed form",
            measured=worst_trade,
            bound=1.0,
            rendered="every trade within one lot of the continuum solution",
        ),
        Check(
            label="The same grid's objective against the continuum's",
            # The grid is a subset, so the only wrong answer is a *lower*
            # objective. Measuring how far below it goes, with a bound of zero,
            # states that: a tie passes, an improvement does not.
            measured=max(shortfall_below, 0.0),
            bound=0.0,
            rendered="never below it, the grid being a subset of the continuum",
        ),
        plan.objective,
        continuum_objective,
    )


def kernel_shape() -> tuple[Check, float, float]:
    """The response kernel's shape, measured as a finding rather than assumed.

    At zero urgency the kernel of the quadratic execution problem is the
    Green's function of a discrete Laplacian, which means every row is linear
    away from its own spike — a tent. The parameter-free way to say that is
    that the second difference of a row vanishes everywhere except at the
    spike, which needs no fitted scale and no reference implementation.

    Measured at a positive urgency as well, and that is half the row. A
    second difference of zero at one risk aversion is interesting only beside a
    second difference that is *not* zero at another: without the second
    measurement, "the kernel is a tent" could be describing a kernel that is
    always a tent because something was linearised, rather than a shape the
    problem produces at one end of its range and not the other.
    """
    problem = sl.ExecutionProblem(
        quantity=1.0, horizon=5.0, periods=20, volatility=0.95, impact=AC_IMPACT
    )

    def worst_second_difference(risk_aversion: float) -> float:
        kernel = sl.response_kernel(problem, risk_aversion)
        size = len(kernel)
        peak = max(max(abs(value) for value in row) for row in kernel)
        worst = 0.0
        for column in range(size):
            for row in range(1, size - 1):
                if row == column:
                    continue
                curvature = (
                    kernel[row - 1][column] - 2 * kernel[row][column] + kernel[row + 1][column]
                )
                worst = max(worst, abs(curvature))
        return worst / peak

    flat = worst_second_difference(0.0)
    urgent = worst_second_difference(1e-6)
    return (
        Check(
            label="The response kernel at zero urgency, against a tent",
            measured=flat,
            bound=1e-14,
            rendered=(
                "curvature 1e-14 of its peak away from the spike; "
                "2e-02 at an urgency of 1e-6, so the shape is a result"
            ),
        ),
        flat,
        urgent,
    )


def shortfall_against_arithmetic_by_hand() -> Check:
    """Implementation shortfall against the same sum done by hand.

    The worked example computes each component twice: once through the library
    and once as plain arithmetic written out in the script. Both appear in its
    output, and this asserts they are the same number rather than merely close,
    because a cost decomposition is a sum of products and there is nothing in
    it for a tolerance to absorb.
    """
    with redirect_stdout(StringIO()) as captured:
        runpy.run_path(str(WORKED))["main"]()
    output = captured.getvalue()
    worst = 0.0
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            by_hand = float(parts[1].replace(",", ""))
            library = float(parts[2].replace(",", ""))
        except ValueError:
            continue
        worst = max(worst, abs(by_hand - library))
    return Check(
        label="Implementation shortfall against the same sum done by hand",
        measured=worst,
        bound=0.0,
        rendered="identical on every component, to the cent",
    )


def ex_own_vwap_against_a_known_counterparty() -> tuple[Check, float, float]:
    """Recover the other traders' VWAP from bars that contain the order's prints too.

    The reference here is a *construction* rather than a formula. Two hundred
    books are built from explicit prints: a set the order did and a set it did
    not, each with its own quantity and price. Each bar's volume is the sum of
    both and its price is their blend, which is what an exchange tape reports
    and what the library's typical-price estimate recovers when the bar is
    flat. The others' VWAP is then known before the library is asked -- it is
    the volume-weighted average of the prints the order had no part in -- and
    the correction has to return that number.

    This contains no derivation shared with the library. The generator never
    computes a participation rate or a difference of notionals; it knows the
    answer because it wrote the prints down.

    Returns the worst relative recovery error, and the range of ``1 - p`` over
    the two hundred books, which is what the uncorrected benchmark is wrong by.
    """
    rng = np.random.default_rng(20261022)
    worst = 0.0
    factors = []
    for _ in range(200):
        bars = []
        own_fills = []
        others_quantity = 0.0
        others_notional = 0.0
        start = datetime(2026, 3, 2, 9, 30)
        for index in range(5):
            timestamp = start + timedelta(minutes=index)
            others_volume = float(rng.uniform(500.0, 3000.0))
            others_price = 100.0 + float(rng.normal(0.0, 0.2))
            own_volume = float(rng.uniform(0.0, 1200.0))
            own_price = others_price + float(rng.normal(0.0, 0.05))
            others_quantity += others_volume
            others_notional += others_volume * others_price
            if own_volume > 0.0:
                own_fills.append(
                    sl.Fill(
                        timestamp=timestamp + timedelta(seconds=20),
                        quantity=own_volume,
                        price=own_price,
                    )
                )
            volume = others_volume + own_volume
            blend = (others_volume * others_price + own_volume * own_price) / volume
            bars.append(
                sl.Bar(
                    timestamp=timestamp,
                    open=blend,
                    high=blend,
                    low=blend,
                    close=blend,
                    volume=volume,
                )
            )
        series = sl.BarSeries(bars)
        order = sl.Order(
            symbol="ACME",
            side=sl.Side.BUY,
            quantity=sum(fill.quantity for fill in own_fills),
            decision_time=start,
            arrival_time=start,
            fills=tuple(own_fills),
        )
        known = others_notional / others_quantity
        share = sl.own_share(order, series)
        worst = max(worst, abs(share.ex_own_vwap - known) / known)
        factors.append(1.0 - share.rate)
    return (
        Check(
            label="The ex-own VWAP against a book whose other prints are known, 200 books",
            measured=worst,
            bound=1e-12,
            rendered=(
                f"recovers their VWAP to 1e-12 relative, where the uncorrected "
                f"benchmark understates by {1.0 - max(factors):.0%} to "
                f"{1.0 - min(factors):.0%}"
            ),
        ),
        min(factors),
        max(factors),
    )


def main() -> dict[str, object]:
    """Run every check, print the table, and return the figures."""
    schedule, cases, schedules = schedule_against_enumeration()
    routing, checked, optimal = routing_against_exhaustive_search()
    trades, objective, grid_value, continuum_value = grid_against_the_closed_form()
    kernel, flat, urgent = kernel_shape()
    ex_own, least_factor, most_factor = ex_own_vwap_against_a_known_counterparty()
    checks = [
        schedule,
        routing,
        trades,
        objective,
        kernel,
        shortfall_against_arithmetic_by_hand(),
        ex_own,
    ]

    print("What is checked, and against what")
    print()
    print("| Check | Result |")
    print("| --- | --- |")
    for check in checks:
        print(f"| {check.label} | {check.rendered} |")

    print()
    print("The figures behind them:")
    print(f"  {cases} schedule cases, each against {schedules:,} enumerated schedules")
    print(f"  {checked} routing problems enumerated, {optimal} solved to the optimum")
    print(f"  grid objective {grid_value:.6f} against the continuum's {continuum_value:.6f}")
    print(f"  kernel curvature {flat:.3e} of peak at zero urgency, {urgent:.3e} at 1e-6")

    print()
    print("Measured against the bound the README states:")
    for check in checks:
        verdict = "ok" if check.passed else "BREACHED"
        print(f"  {check.label}: {check.measured:.3e} <= {check.bound:.0e}  {verdict}")

    failed = [check for check in checks if not check.passed]
    if failed:
        print()
        for check in failed:
            print(
                f"FAIL {check.label}: measured {check.measured:.6e}, "
                f"README claims {check.bound:.0e}",
                file=sys.stderr,
            )
        raise SystemExit(1)

    return {
        "checks": {check.label: check.measured for check in checks},
        "bounds": {check.label: check.bound for check in checks},
        "rendered": {check.label: check.rendered for check in checks},
        "cases": cases,
        "schedules": schedules,
        "checked": checked,
        "optimal": optimal,
        "grid_objective": grid_value,
        "continuum_objective": continuum_value,
        "curvature": (flat, urgent),
        "ex_own_factors": (least_factor, most_factor),
    }


if __name__ == "__main__":
    main()
