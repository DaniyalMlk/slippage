from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slippage.exceptions import ValidationError
from slippage.impact import (
    ImpactModel,
    LinearImpact,
    PowerLawImpact,
    SquareRootLaw,
    schedule_cost,
    uniform_schedule,
)

schedules = st.lists(st.floats(min_value=0.0, max_value=1e5), min_size=1, max_size=40)


class TestModels:
    def test_both_rate_models_satisfy_the_protocol(self) -> None:
        assert isinstance(LinearImpact(gamma=1e-6, eta=1e-5), ImpactModel)
        assert isinstance(PowerLawImpact(gamma=1e-6, eta=1e-5, beta=0.6), ImpactModel)

    def test_no_cost_for_not_trading(self) -> None:
        # The fixed per-share cost is charged on traded shares, not on idle
        # intervals, otherwise a schedule with more periods would look dearer.
        model = LinearImpact(gamma=0.0, eta=1.0, epsilon=0.05)
        assert model.temporary(0.0) == 0.0
        assert model.temporary(2.0) == pytest.approx(2.05)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gamma": -1.0, "eta": 1.0},
            {"gamma": 1.0, "eta": -1.0},
            {"gamma": 1.0, "eta": 1.0, "epsilon": math.nan},
        ],
    )
    def test_linear_rejects_negative_or_nan(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            LinearImpact(**kwargs)

    @pytest.mark.parametrize("beta", [0.0, -0.5, 3.5, math.inf])
    def test_power_law_rejects_implausible_exponents(self, beta: float) -> None:
        with pytest.raises(ValidationError):
            PowerLawImpact(gamma=0.0, eta=1.0, beta=beta)

    def test_negative_rate_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unsigned"):
            LinearImpact(gamma=0.0, eta=1.0).temporary(-1.0)
        with pytest.raises(ValidationError, match="unsigned"):
            PowerLawImpact(gamma=0.0, eta=1.0, beta=0.5).temporary(-1.0)


class TestUniformScheduleIntegrals:
    """Table of cases where the cost integrates by hand."""

    @pytest.mark.parametrize(
        ("gamma", "eta", "epsilon", "x", "n", "tau"),
        [
            (0.0, 1e-6, 0.0, 1e6, 1, 1.0),
            (2.5e-7, 2.5e-6, 0.0625, 1e6, 5, 1.0),
            (2.5e-7, 2.5e-6, 0.0625, 1e6, 50, 0.1),
            (1e-8, 3e-7, 0.01, 2.5e5, 13, 1.0 / 13),
        ],
    )
    def test_linear(
        self, gamma: float, eta: float, epsilon: float, x: float, n: int, tau: float
    ) -> None:
        model = LinearImpact(gamma=gamma, eta=eta, epsilon=epsilon)
        cost = schedule_cost(model, uniform_schedule(x, n), tau)
        assert cost.temporary == pytest.approx(epsilon * x + eta * x**2 / (n * tau))
        assert cost.permanent == pytest.approx(gamma * (x**2 - x**2 / n) / 2)

    @pytest.mark.parametrize("beta", [0.25, 0.5, 0.6, 1.0, 1.5])
    @pytest.mark.parametrize(("x", "n", "tau"), [(1e5, 1, 1.0), (1e5, 10, 0.5), (3e4, 7, 2.0)])
    def test_power_law(self, beta: float, x: float, n: int, tau: float) -> None:
        eta, epsilon = 1e-3, 0.02
        model = PowerLawImpact(gamma=0.0, eta=eta, beta=beta, epsilon=epsilon)
        cost = schedule_cost(model, uniform_schedule(x, n), tau)
        # n slices of x/n at rate x/(n tau): x * (epsilon + eta * (x/(n tau))**beta).
        assert cost.temporary == pytest.approx(
            epsilon * x + eta * x ** (1 + beta) / (n * tau) ** beta
        )

    def test_slower_is_cheaper_in_temporary_cost(self) -> None:
        model = PowerLawImpact(gamma=0.0, eta=1e-3, beta=0.6)
        fast = schedule_cost(model, uniform_schedule(1e5, 5), 1.0).temporary
        slow = schedule_cost(model, uniform_schedule(1e5, 20), 1.0).temporary
        # Four times the horizon cuts the temporary cost by 4 ** beta.
        assert fast / slow == pytest.approx(4**0.6)


class TestScheduleIdentities:
    @settings(max_examples=200)
    @given(trades=schedules, gamma=st.floats(min_value=0.0, max_value=1e-4))
    def test_almgren_chriss_permanent_identity(self, trades: list[float], gamma: float) -> None:
        cost = schedule_cost(LinearImpact(gamma=gamma, eta=0.0), trades, 1.0)
        x = sum(trades)
        expected = gamma * (x**2 - sum(n * n for n in trades)) / 2
        assert cost.permanent == pytest.approx(expected, rel=1e-9, abs=1e-9)

    @settings(max_examples=200)
    @given(
        trades=schedules,
        eta=st.floats(min_value=0.0, max_value=1e-3),
        epsilon=st.floats(min_value=0.0, max_value=0.1),
        tau=st.floats(min_value=0.01, max_value=10.0),
    )
    def test_unit_exponent_power_law_is_the_linear_model(
        self, trades: list[float], eta: float, epsilon: float, tau: float
    ) -> None:
        linear = schedule_cost(LinearImpact(gamma=1e-7, eta=eta, epsilon=epsilon), trades, tau)
        power = schedule_cost(
            PowerLawImpact(gamma=1e-7, eta=eta, beta=1.0, epsilon=epsilon), trades, tau
        )
        assert power.total == pytest.approx(linear.total, rel=1e-12, abs=1e-9)

    @settings(max_examples=200)
    @given(
        weights=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=20),
        beta=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_uniform_minimises_temporary_cost_for_fixed_horizon(
        self, weights: list[float], beta: float
    ) -> None:
        # Jensen: n h(n/tau) is convex in n, so spreading evenly is optimal
        # when only temporary impact matters.
        total = sum(weights)
        if total <= 0.0:
            return
        x = 1e4
        trades = [x * w / total for w in weights]
        model = PowerLawImpact(gamma=0.0, eta=1e-3, beta=beta)
        uneven = schedule_cost(model, trades, 1.0).temporary
        even = schedule_cost(model, uniform_schedule(x, len(trades)), 1.0).temporary
        assert even <= uneven * (1 + 1e-12)

    def test_reordering_changes_nothing_for_linear_permanent_impact(self) -> None:
        # Linear permanent cost depends only on the multiset of trade sizes.
        model = LinearImpact(gamma=1e-6, eta=1e-5, epsilon=0.01)
        a = schedule_cost(model, [100.0, 400.0, 250.0], 1.0)
        b = schedule_cost(model, [250.0, 100.0, 400.0], 1.0)
        assert a.total == pytest.approx(b.total)

    def test_negative_trade_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="trade 1"):
            schedule_cost(LinearImpact(gamma=0.0, eta=1.0), [1.0, -1.0], 1.0)

    def test_non_positive_tau_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tau"):
            schedule_cost(LinearImpact(gamma=0.0, eta=1.0), [1.0], 0.0)

    def test_uniform_schedule_validation(self) -> None:
        assert uniform_schedule(10.0, 4) == [2.5] * 4
        with pytest.raises(ValidationError):
            uniform_schedule(10.0, 0)


class TestSquareRootLaw:
    @pytest.mark.parametrize(
        ("y", "delta", "q", "v", "sigma", "expected"),
        [
            (1.0, 0.5, 1e5, 1e7, 0.02, 0.02 * 0.1),  # 1% of ADV, 2% vol -> 20 bps
            (0.8, 0.5, 4e5, 1e7, 0.02, 0.8 * 0.02 * 0.2),
            (1.0, 0.6, 1e6, 1e7, 0.015, 0.015 * 0.1**0.6),
            (1.0, 0.5, 0.0, 1e7, 0.02, 0.0),
        ],
    )
    def test_peak_impact_table(
        self, y: float, delta: float, q: float, v: float, sigma: float, expected: float
    ) -> None:
        assert SquareRootLaw(y=y, delta=delta).peak_impact(q, v, sigma) == pytest.approx(expected)

    def test_four_times_the_size_doubles_the_impact(self) -> None:
        law = SquareRootLaw()
        assert law.peak_impact(4e5, 1e7, 0.02) / law.peak_impact(1e5, 1e7, 0.02) == pytest.approx(
            2.0
        )

    @pytest.mark.parametrize("delta", [0.3, 0.5, 0.6, 1.0])
    def test_average_cost_is_the_mean_of_the_impact_path(self, delta: float) -> None:
        # Integrate I(t) = peak * (t/T)**delta numerically and compare.
        law = SquareRootLaw(y=0.9, delta=delta)
        peak = law.peak_impact(2e5, 1e7, 0.025)
        t = (np.arange(200_000) + 0.5) / 200_000
        numeric = float(np.mean(peak * t**delta))
        assert law.expected_cost(2e5, 1e7, 0.025) == pytest.approx(numeric, rel=1e-6)

    def test_square_root_average_is_two_thirds_of_peak(self) -> None:
        law = SquareRootLaw()
        assert law.expected_cost(1e5, 1e7, 0.02) == pytest.approx(
            2.0 / 3.0 * law.peak_impact(1e5, 1e7, 0.02)
        )
        assert law.expected_cost_bps(1e5, 1e7, 0.02) == pytest.approx(20.0 * 2.0 / 3.0)

    @pytest.mark.parametrize(
        ("q", "v", "sigma"), [(-1.0, 1e6, 0.02), (1.0, 0.0, 0.02), (1.0, 1e6, -0.1)]
    )
    def test_invalid_inputs(self, q: float, v: float, sigma: float) -> None:
        with pytest.raises(ValidationError):
            SquareRootLaw().peak_impact(q, v, sigma)
