from __future__ import annotations

import math

import numpy as np
import pytest

from slippage.exceptions import ValidationError
from slippage.execution import ExecutionProblem, Trajectory, optimal_trajectory
from slippage.impact import LinearImpact, PowerLawImpact, schedule_cost
from slippage.simulate import CostDistribution, simulate_costs, simulate_prices

PROBLEM = ExecutionProblem(
    quantity=1e6,
    horizon=5.0,
    periods=5,
    volatility=0.95,
    impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.0625),
)
Z95 = 1.6448536269514722


@pytest.fixture(scope="module")
def trajectory() -> Trajectory:
    return optimal_trajectory(PROBLEM, 1e-6)


def simulate(trades: list[float] | tuple[float, ...], **kwargs: object) -> CostDistribution:
    options: dict[str, object] = {
        "tau": PROBLEM.tau,
        "volatility": PROBLEM.volatility,
        "paths": 200_000,
        "rng": np.random.default_rng(42),
    }
    options.update(kwargs)
    return simulate_costs(PROBLEM.impact, list(trades), **options)  # type: ignore[arg-type]


class TestAgainstClosedForm:
    def test_mean_and_variance(self, trajectory: Trajectory) -> None:
        dist = simulate(trajectory.trades)
        assert abs(dist.mean - trajectory.expected_cost) < 4 * dist.std_error
        # The sample variance of n normals has relative error sqrt(2 / (n - 1)).
        rel = math.sqrt(2 / (dist.paths - 1))
        assert dist.std**2 == pytest.approx(trajectory.variance, rel=4 * rel)

    def test_tail_matches_the_normal_distribution(self, trajectory: Trajectory) -> None:
        # Cost is linear in Gaussian shocks, so it is exactly normal.
        dist = simulate(trajectory.trades)
        analytic_q = trajectory.expected_cost + Z95 * trajectory.std
        analytic_es = trajectory.expected_cost + trajectory.std * math.exp(-(Z95**2) / 2) / (
            0.05 * math.sqrt(2 * math.pi)
        )
        assert dist.quantile(0.95) == pytest.approx(analytic_q, rel=0.01)
        assert dist.expected_shortfall(0.95) == pytest.approx(analytic_es, rel=0.01)

    def test_drift_shifts_the_mean_by_the_held_position(self, trajectory: Trajectory) -> None:
        mu = 0.05
        held = sum(PROBLEM.quantity - sum(trajectory.trades[: k + 1]) for k in range(5))
        dist = simulate(trajectory.trades, drift=mu, antithetic=True)
        assert dist.mean == pytest.approx(trajectory.expected_cost - mu * PROBLEM.tau * held)

    def test_power_law_mean_is_the_schedule_cost(self) -> None:
        impact = PowerLawImpact(gamma=2.5e-7, eta=1e-3, beta=0.6, epsilon=0.0625)
        trades = [3e5, 3e5, 2e5, 1e5, 1e5]
        dist = simulate_costs(
            impact,
            trades,
            tau=1.0,
            volatility=0.95,
            paths=1_000,
            antithetic=True,
            rng=np.random.default_rng(0),
        )
        assert dist.mean == pytest.approx(schedule_cost(impact, trades, 1.0).total, rel=1e-12)


class TestAntithetic:
    def test_pairs_make_the_mean_exact(self, trajectory: Trajectory) -> None:
        dist = simulate(trajectory.trades, paths=1_000, antithetic=True)
        assert dist.mean == pytest.approx(trajectory.expected_cost, rel=1e-12)
        assert dist.std_error == pytest.approx(0.0, abs=1e-6)
        # The spread of individual paths is untouched.
        assert dist.std == pytest.approx(trajectory.std, rel=0.1)

    def test_needs_an_even_number_of_paths(self, trajectory: Trajectory) -> None:
        with pytest.raises(ValidationError, match="even"):
            simulate(trajectory.trades, paths=101, antithetic=True)

    def test_barely_helps_the_tail(self, trajectory: Trajectory) -> None:
        rng = np.random.default_rng(7)

        def spread(antithetic: bool) -> float:
            q = [
                simulate(trajectory.trades, paths=4_000, rng=rng, antithetic=antithetic).quantile(
                    0.95
                )
                for _ in range(150)
            ]
            return float(np.std(q))

        ratio = spread(True) / spread(False)
        assert 0.7 < ratio < 1.1


class TestPricePaths:
    def test_costs_rebuilt_from_fills_match(self, trajectory: Trajectory) -> None:
        mid, fills = simulate_prices(
            PROBLEM.impact,
            list(trajectory.trades),
            arrival_price=50.0,
            tau=PROBLEM.tau,
            volatility=PROBLEM.volatility,
            paths=500,
            rng=np.random.default_rng(3),
        )
        assert mid.shape == (500, 6)
        assert fills.shape == (500, 5)
        rebuilt = (50.0 - fills) @ np.array(trajectory.trades)
        direct = simulate_costs(
            PROBLEM.impact,
            list(trajectory.trades),
            tau=PROBLEM.tau,
            volatility=PROBLEM.volatility,
            paths=500,
            rng=np.random.default_rng(3),
        )
        assert rebuilt == pytest.approx(direct.costs, rel=1e-9, abs=1e-6)

    def test_permanent_impact_is_left_in_the_price(self) -> None:
        mid, _ = simulate_prices(
            LinearImpact(gamma=1e-6, eta=1e-6),
            [1e5, 1e5],
            arrival_price=50.0,
            tau=1.0,
            volatility=0.0,
            paths=2,
        )
        assert mid[0] == pytest.approx([50.0, 49.9, 49.8])

    def test_arrival_price_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            simulate_prices(PROBLEM.impact, [1.0], arrival_price=0.0, tau=1.0, volatility=1.0)


class TestDistribution:
    def test_summary_statistics(self) -> None:
        dist = CostDistribution(costs=np.arange(1.0, 101.0, dtype=np.float64))
        assert dist.paths == 100
        assert dist.mean == pytest.approx(50.5)
        assert dist.quantile(0.5) == pytest.approx(50.5)
        assert dist.expected_shortfall(0.9) == pytest.approx(
            np.mean(np.arange(91.0, 101.0)), rel=0.02
        )
        with pytest.raises(ValidationError):
            dist.quantile(1.5)

    @pytest.mark.parametrize(
        "kwargs",
        [{"tau": 0.0}, {"volatility": -1.0}, {"paths": 1}],
    )
    def test_invalid_inputs(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            simulate([1.0, 2.0], **kwargs)

    def test_negative_trades_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            simulate([1.0, -2.0])
