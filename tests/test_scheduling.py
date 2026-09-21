from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slippage.exceptions import ValidationError
from slippage.execution import ExecutionProblem, optimal_trajectory
from slippage.impact import ImpactModel, LinearImpact, PowerLawImpact
from slippage.scheduling import (
    participation_caps,
    schedule_objective,
    solve_schedule,
)

AC_IMPACT = LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.0625)


def compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """Every way to write ``total`` as an ordered sum of ``parts`` non-negative integers."""
    return [
        tuple(b - a - 1 for a, b in itertools.pairwise((-1, *cut, total + parts - 1)))
        for cut in itertools.combinations(range(total + parts - 1), parts - 1)
    ]


def brute_force(
    impact: ImpactModel,
    lots: int,
    lot: float,
    periods: int,
    tau: float,
    sigma: float,
    lam: float,
    *,
    cap: int | None = None,
    floor: int | None = None,
    already_done: float = 0.0,
) -> float:
    best = math.inf
    for combo in compositions(lots, periods):
        remaining = lots
        ok = True
        for n in combo:
            if cap is not None and n > cap:
                ok = False
            if floor is not None and n < min(floor, remaining):
                ok = False
            remaining -= n
        if not ok:
            continue
        value = schedule_objective(
            impact, [n * lot for n in combo], tau, sigma, lam, already_done=already_done
        )
        best = min(best, value)
    return best


class TestCompositions:
    def test_counts_and_sums(self) -> None:
        combos = compositions(4, 3)
        assert len(combos) == math.comb(6, 2)
        assert all(sum(c) == 4 and len(c) == 3 for c in combos)
        assert len(set(combos)) == len(combos)


class TestAgainstBruteForce:
    @pytest.mark.parametrize(
        "impact",
        [
            LinearImpact(gamma=1e-3, eta=5e-2, epsilon=0.01),
            PowerLawImpact(gamma=1e-3, eta=2e-1, beta=0.5, epsilon=0.01),
            PowerLawImpact(gamma=0.0, eta=1e-2, beta=1.7),
        ],
        ids=["linear", "sqrt", "convex"],
    )
    @pytest.mark.parametrize("lam", [0.0, 1e-3, 1e-1])
    def test_matches_exhaustive_search(self, impact: ImpactModel, lam: float) -> None:
        lot, lots, periods, horizon, sigma = 10.0, 8, 4, 2.0, 0.5
        plan = solve_schedule(
            quantity=lot * lots,
            horizon=horizon,
            periods=periods,
            volatility=sigma,
            impact=impact,
            risk_aversion=lam,
            lot_size=lot,
        )
        tau = horizon / periods
        expected = brute_force(impact, lots, lot, periods, tau, sigma, lam)
        assert plan.objective == pytest.approx(expected, rel=1e-12)
        assert schedule_objective(impact, plan.trades, tau, sigma, lam) == pytest.approx(expected)

    @settings(max_examples=40, deadline=None)
    @given(
        lots=st.integers(min_value=1, max_value=9),
        periods=st.integers(min_value=1, max_value=4),
        cap=st.integers(min_value=1, max_value=9),
        floor=st.integers(min_value=0, max_value=4),
        lam=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_constrained_problems_match_exhaustive_search(
        self, lots: int, periods: int, cap: int, floor: int, lam: float
    ) -> None:
        impact = PowerLawImpact(gamma=1e-3, eta=3e-2, beta=0.6, epsilon=0.02)
        lot, horizon, sigma = 5.0, 1.0, 0.3
        tau = horizon / periods
        expected = brute_force(impact, lots, lot, periods, tau, sigma, lam, cap=cap, floor=floor)
        if math.isinf(expected):
            with pytest.raises(ValidationError, match="no schedule completes"):
                solve_schedule(
                    quantity=lot * lots,
                    horizon=horizon,
                    periods=periods,
                    volatility=sigma,
                    impact=impact,
                    risk_aversion=lam,
                    lot_size=lot,
                    max_trade=cap * lot,
                    min_trade=floor * lot,
                )
            return
        plan = solve_schedule(
            quantity=lot * lots,
            horizon=horizon,
            periods=periods,
            volatility=sigma,
            impact=impact,
            risk_aversion=lam,
            lot_size=lot,
            max_trade=cap * lot,
            min_trade=floor * lot,
        )
        assert plan.objective == pytest.approx(expected, rel=1e-12, abs=1e-12)


class TestAgainstClosedForm:
    def test_slack_constraints_reproduce_almgren_chriss(self) -> None:
        problem = ExecutionProblem(
            quantity=1e6, horizon=5.0, periods=10, volatility=0.95, impact=AC_IMPACT
        )
        lam = 1e-6
        closed = optimal_trajectory(problem, lam)
        lot = 1e6 / 2000
        plan = solve_schedule(
            quantity=1e6,
            horizon=5.0,
            periods=10,
            volatility=0.95,
            impact=AC_IMPACT,
            risk_aversion=lam,
            lot_size=lot,
        )
        for dp, cf in zip(plan.trades, closed.trades, strict=True):
            assert abs(dp - cf) <= lot
        # The grid is a subset of the continuum, so the programme can only do
        # as well or slightly worse than the closed form, never better.
        assert plan.objective >= closed.objective(lam) * (1 - 1e-12)
        assert plan.objective == pytest.approx(closed.objective(lam), rel=1e-5)

    def test_risk_neutral_power_law_trades_evenly(self) -> None:
        # With no risk term the uniform schedule minimises any convex
        # temporary cost (Jensen). Linear permanent cost, gamma (X^2 - sum n^2)
        # / 2, leans the other way and rewards concentration, but with gamma
        # this small the temporary term decides.
        plan = solve_schedule(
            quantity=12_000.0,
            horizon=1.0,
            periods=6,
            volatility=0.5,
            impact=PowerLawImpact(gamma=1e-7, eta=1e-3, beta=0.5),
            risk_aversion=0.0,
            lot_size=100.0,
        )
        assert plan.trades == [2_000.0] * 6


class TestConstraints:
    @staticmethod
    def solve(**overrides: object) -> list[float]:
        kwargs: dict[str, object] = {
            "quantity": 1e6,
            "horizon": 5.0,
            "periods": 10,
            "volatility": 0.95,
            "impact": AC_IMPACT,
            "risk_aversion": 1e-6,
            "lot_size": 1_000.0,
        }
        kwargs.update(overrides)
        return solve_schedule(**kwargs).trades  # type: ignore[arg-type]

    def test_binding_cap_is_respected(self) -> None:
        free = self.solve()
        assert free[0] > 150_000
        capped = self.solve(max_trade=150_000.0)
        assert max(capped) <= 150_000.0
        assert sum(capped) == pytest.approx(1e6)
        # The capped plan fills the cap early rather than spreading evenly.
        assert capped[0] == 150_000.0

    def test_per_period_caps(self) -> None:
        caps = [50_000.0] * 5 + [300_000.0] * 5
        trades = self.solve(max_trade=caps)
        assert all(n <= c for n, c in zip(trades, caps, strict=True))

    def test_floor_is_respected(self) -> None:
        trades = self.solve(risk_aversion=0.0, min_trade=120_000.0)
        remaining = 1e6
        for n in trades:
            assert n >= min(120_000.0, remaining) - 1e-9
            remaining -= n

    def test_lot_size_is_respected(self) -> None:
        trades = self.solve(lot_size=25_000.0)
        assert all(n / 25_000.0 == round(n / 25_000.0) for n in trades)

    def test_infeasible_caps_are_reported(self) -> None:
        with pytest.raises(ValidationError, match="at most 500000 shares"):
            self.solve(max_trade=50_000.0)

    def test_quantity_must_fill_whole_lots(self) -> None:
        with pytest.raises(ValidationError, match="whole number"):
            self.solve(quantity=1_000_500.0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"quantity": -1.0},
            {"horizon": 0.0},
            {"periods": 0},
            {"volatility": -1.0},
            {"risk_aversion": -1e-6},
            {"lot_size": 0.0},
            {"max_trade": [1.0, 2.0]},
            {"min_trade": -5.0},
        ],
    )
    def test_invalid_inputs(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            self.solve(**overrides)


class TestParticipationCaps:
    def test_caps_scale_the_volume_profile(self) -> None:
        assert participation_caps([100_000.0, 50_000.0, 0.0], 0.1) == [10_000.0, 5_000.0, 0.0]

    @pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
    def test_rate_bounds(self, rate: float) -> None:
        with pytest.raises(ValidationError):
            participation_caps([1.0], rate)

    def test_negative_volume(self) -> None:
        with pytest.raises(ValidationError):
            participation_caps([-1.0], 0.1)


class TestReoptimisation:
    @staticmethod
    def plan(**overrides: object):  # type: ignore[no-untyped-def]
        kwargs: dict[str, object] = {
            "quantity": 200.0,
            "horizon": 4.0,
            "periods": 8,
            "volatility": 0.4,
            "impact": PowerLawImpact(gamma=2e-4, eta=2e-2, beta=0.6, epsilon=0.01),
            "risk_aversion": 5e-3,
            "lot_size": 10.0,
        }
        kwargs.update(overrides)
        return solve_schedule(**kwargs)  # type: ignore[arg-type]

    def test_following_the_plan_is_time_consistent(self) -> None:
        plan = self.plan()
        trades = plan.trades
        remaining = plan.quantity
        for k in range(plan.periods):
            assert plan.trades_from(k, remaining) == trades[k:]
            remaining -= trades[k]

    def test_off_plan_state_gets_the_optimal_catch_up(self) -> None:
        plan = self.plan()
        # Three periods in, only 50 shares have been done, whatever the plan
        # intended; the policy must give the best schedule for the rest.
        catch_up = plan.trades_from(3, 150.0)
        assert sum(catch_up) == pytest.approx(150.0)
        tau = plan.tau
        impact = PowerLawImpact(gamma=2e-4, eta=2e-2, beta=0.6, epsilon=0.01)
        best = brute_force(impact, 15, 10.0, 5, tau, 0.4, 5e-3, already_done=50.0)
        got = schedule_objective(impact, catch_up, tau, 0.4, 5e-3, already_done=50.0)
        assert got == pytest.approx(best, rel=1e-12)
        assert plan.values[3, 15] == pytest.approx(best, rel=1e-12)

    def test_behind_schedule_trades_harder(self) -> None:
        plan = self.plan()
        on_plan_left = plan.quantity - sum(plan.trades[:2])
        behind = plan.next_trade(2, on_plan_left + 50.0)
        on_plan = plan.next_trade(2, on_plan_left)
        assert behind > on_plan

    def test_state_queries(self) -> None:
        plan = self.plan(max_trade=30.0)
        assert plan.is_feasible(0, 200.0)
        # Six periods of at most 30 cannot finish 200.
        assert not plan.is_feasible(2, 200.0)
        with pytest.raises(ValidationError, match="satisfies the constraints"):
            plan.next_trade(2, 200.0)
        assert plan.is_feasible(8, 0.0)
        with pytest.raises(ValidationError, match="whole number of lots"):
            plan.next_trade(0, 15.0)
        with pytest.raises(ValidationError, match="period"):
            plan.next_trade(8, 0.0)


def test_nothing_left_costs_nothing() -> None:
    plan = solve_schedule(
        quantity=100.0,
        horizon=1.0,
        periods=5,
        volatility=0.2,
        impact=LinearImpact(gamma=0.0, eta=1e-2),
        risk_aversion=0.1,
        lot_size=10.0,
    )
    assert np.all(plan.values[:, 0] == 0.0)
