from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slippage.exceptions import ValidationError
from slippage.execution import (
    ExecutionProblem,
    _sinh_ratio,
    closed_form_moments,
    linear_trajectory,
    optimal_trajectory,
    schedule_moments,
    trajectory_from_trades,
)
from slippage.impact import LinearImpact


def almgren_chriss_example(periods: int = 5) -> ExecutionProblem:
    """Table 1 of Almgren and Chriss (2000).

    One million shares at $50 over five days; 30% annual volatility, so
    sigma = 0.95 $/share/day^0.5; a 1/8 spread, so epsilon = 1/16; gamma and
    eta calibrated to 5 million shares of daily volume.
    """
    return ExecutionProblem(
        quantity=1e6,
        horizon=5.0,
        periods=periods,
        volatility=0.95,
        impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.0625),
    )


class TestProblem:
    def test_derived_quantities(self) -> None:
        problem = almgren_chriss_example()
        assert problem.tau == 1.0
        assert problem.eta_tilde == pytest.approx(2.5e-6 - 0.5 * 2.5e-7)
        assert problem.times() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_published_urgency(self) -> None:
        # The paper: "kappa ~ 0.6/day, so kappa T ~ 3" at lambda = 1e-6.
        problem = almgren_chriss_example()
        kappa = problem.kappa(1e-6)
        assert kappa == pytest.approx(0.6, abs=0.01)
        assert kappa * problem.horizon == pytest.approx(3.0, abs=0.05)

    def test_kappa_solves_the_discrete_relation(self) -> None:
        problem = almgren_chriss_example(periods=20)
        for lam in (1e-9, 1e-7, 1e-6, 1e-4):
            kappa = problem.kappa(lam)
            lhs = 2.0 / problem.tau**2 * (math.cosh(kappa * problem.tau) - 1.0)
            rhs = lam * problem.volatility**2 / problem.eta_tilde
            assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_kappa_keeps_precision_for_tiny_risk_aversion(self) -> None:
        # Naively, 1 + z rounds to 1 and arccosh returns zero; the continuous
        # limit kappa ~ sqrt(lambda sigma^2 / eta_tilde) must survive.
        problem = almgren_chriss_example()
        lam = 1e-20
        expected = math.sqrt(lam * problem.volatility**2 / problem.eta_tilde)
        assert problem.kappa(lam) == pytest.approx(expected, rel=1e-6)
        assert problem.kappa(0.0) == 0.0

    def test_permanent_impact_dominating_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Use more periods"):
            ExecutionProblem(
                quantity=1e6,
                horizon=5.0,
                periods=1,
                volatility=0.95,
                impact=LinearImpact(gamma=1e-5, eta=1e-6),
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"quantity": 0.0},
            {"horizon": -1.0},
            {"periods": 0},
            {"volatility": -0.1},
            {"volatility": math.nan},
        ],
    )
    def test_invalid_inputs(self, kwargs: dict[str, float]) -> None:
        base: dict[str, object] = {
            "quantity": 1e6,
            "horizon": 5.0,
            "periods": 5,
            "volatility": 0.95,
            "impact": LinearImpact(gamma=2.5e-7, eta=2.5e-6),
        }
        base.update(kwargs)
        with pytest.raises(ValidationError):
            ExecutionProblem(**base)  # type: ignore[arg-type]

    def test_negative_risk_aversion_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="risk aversion"):
            almgren_chriss_example().kappa(-1e-6)


class TestRiskNeutral:
    def test_linear_schedule_trades_evenly(self) -> None:
        trajectory = linear_trajectory(almgren_chriss_example())
        assert trajectory.trades == pytest.approx((2e5,) * 5)
        assert trajectory.holdings == pytest.approx((1e6, 8e5, 6e5, 4e5, 2e5, 0.0))
        assert trajectory.half_life == math.inf

    def test_linear_moments_match_the_hand_formulae(self) -> None:
        # E = gamma X^2 / 2 + epsilon X + eta_tilde X^2 / T
        #   = 125,000 + 62,500 + 475,000 = 662,500
        # V = sigma^2 X^2 T (1 - 1/N)(1 - 1/2N) / 3
        problem = almgren_chriss_example()
        trajectory = linear_trajectory(problem)
        assert trajectory.expected_cost == pytest.approx(662_500.0)
        assert trajectory.variance == pytest.approx(0.95**2 * 1e12 * 5.0 * 0.8 * 0.9 / 3.0)

    def test_linear_schedule_minimises_expected_cost(self) -> None:
        problem = almgren_chriss_example()
        best = linear_trajectory(problem).expected_cost
        rng = np.random.default_rng(0)
        for _ in range(200):
            weights = rng.dirichlet(np.ones(problem.periods))
            cost, _ = schedule_moments(problem, list(weights * problem.quantity))
            assert cost >= best - 1e-6


class TestOptimal:
    @pytest.mark.parametrize("lam", [1e-8, 1e-7, 1e-6, 1e-5])
    @pytest.mark.parametrize("periods", [5, 20, 100])
    def test_summed_moments_match_the_published_closed_form(self, lam: float, periods: int) -> None:
        problem = almgren_chriss_example(periods)
        trajectory = optimal_trajectory(problem, lam)
        expected, variance = closed_form_moments(problem, lam)
        assert trajectory.expected_cost == pytest.approx(expected, rel=1e-9)
        assert trajectory.variance == pytest.approx(variance, rel=1e-9)

    def test_closed_form_reduces_to_the_linear_case(self) -> None:
        problem = almgren_chriss_example()
        assert closed_form_moments(problem, 0.0) == pytest.approx(
            (linear_trajectory(problem).expected_cost, linear_trajectory(problem).variance)
        )

    def test_holdings_follow_the_sinh_formula(self) -> None:
        problem = almgren_chriss_example(periods=10)
        lam = 2e-6
        trajectory = optimal_trajectory(problem, lam)
        kappa = problem.kappa(lam)
        for t, x in zip(trajectory.times, trajectory.holdings, strict=True):
            assert x == pytest.approx(
                1e6 * math.sinh(kappa * (5.0 - t)) / math.sinh(kappa * 5.0), abs=1e-6
            )
        assert trajectory.half_life == pytest.approx(1.0 / kappa)

    def test_trades_sum_exactly_and_decrease(self) -> None:
        trajectory = optimal_trajectory(almgren_chriss_example(periods=50), 1e-6)
        assert sum(trajectory.trades) == pytest.approx(1e6, abs=1e-6)
        assert all(a >= b for a, b in pairwise(trajectory.trades))
        assert all(n >= 0.0 for n in trajectory.trades)
        assert trajectory.holdings[0] == 1e6
        assert trajectory.holdings[-1] == 0.0

    @settings(max_examples=100, deadline=None)
    @given(
        lam=st.floats(min_value=1e-8, max_value=1e-4),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        scale=st.floats(min_value=1e-4, max_value=0.2),
    )
    def test_optimal_beats_perturbations_of_itself(
        self, lam: float, seed: int, scale: float
    ) -> None:
        problem = almgren_chriss_example(periods=10)
        best = optimal_trajectory(problem, lam)
        rng = np.random.default_rng(seed)
        # Multiplicative noise keeps every trade non-negative, then rescale so
        # the perturbed schedule still executes exactly X.
        noisy = np.array(best.trades) * np.exp(rng.normal(0.0, scale, problem.periods))
        trades = list(noisy * problem.quantity / noisy.sum())
        cost, variance = schedule_moments(problem, trades)
        assert cost + lam * variance >= best.objective(lam) * (1 - 1e-12)

    def test_more_risk_aversion_trades_faster(self) -> None:
        problem = almgren_chriss_example(periods=10)
        patient = optimal_trajectory(problem, 1e-8)
        urgent = optimal_trajectory(problem, 1e-5)
        assert urgent.trades[0] > patient.trades[0]
        assert urgent.half_life < patient.half_life

    def test_very_impatient_trader_does_not_overflow(self) -> None:
        # kappa T ~ 2,000 overflows sinh; the exponential form must not.
        problem = almgren_chriss_example(periods=500)
        trajectory = optimal_trajectory(problem, 1e3)
        assert problem.kappa(1e3) * problem.horizon > 710
        assert all(math.isfinite(x) for x in trajectory.holdings)
        assert trajectory.trades[0] == pytest.approx(1e6, rel=1e-3)
        with pytest.raises(OverflowError):
            closed_form_moments(problem, 1e3)

    def test_an_impatient_schedule_has_no_negative_trades(self) -> None:
        # Regression: the summation residue used to be pushed into the last
        # trade, which for an impatient trader is ~1e-10 and went negative.
        problem = almgren_chriss_example(periods=100)
        trajectory = optimal_trajectory(problem, 1.02e-3)
        assert min(trajectory.trades) >= 0.0
        assert sum(trajectory.trades) == pytest.approx(problem.quantity, rel=1e-12)

    @pytest.mark.parametrize("kappa_t", [5.0, 19.99, 20.01, 50.0, 300.0])
    def test_exponential_form_matches_sinh_where_both_are_finite(self, kappa_t: float) -> None:
        # Below kappa T = 20 the ratio is taken directly and above it through
        # exponentials; both must agree with the textbook form, and with each
        # other across the switch.
        horizon = 5.0
        kappa = kappa_t / horizon
        for remaining in (0.0, 0.01, 1.0, 2.5, 4.99, 5.0):
            direct = math.sinh(kappa * remaining) / math.sinh(kappa * horizon)
            assert _sinh_ratio(kappa, remaining, horizon) == pytest.approx(
                direct, rel=1e-12, abs=1e-300
            )


class TestArbitrarySchedules:
    def test_trajectory_from_trades(self) -> None:
        problem = almgren_chriss_example()
        trajectory = trajectory_from_trades(problem, [5e5, 5e5, 0.0, 0.0, 0.0])
        assert trajectory.holdings == (1e6, 5e5, 0.0, 0.0, 0.0, 0.0)
        # Only the first interval carries risk: 500,000 shares held through it.
        assert trajectory.variance == pytest.approx(0.95**2 * 1.0 * 5e5**2)

    def test_wrong_length_or_total_is_rejected(self) -> None:
        problem = almgren_chriss_example()
        with pytest.raises(ValidationError, match="expected 5 trades"):
            schedule_moments(problem, [1e6])
        with pytest.raises(ValidationError, match="not the quantity"):
            schedule_moments(problem, [1e5] * 5)
