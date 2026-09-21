from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pytest
from conftest import make_bar, minute

from slippage.calibration import (
    Estimate,
    fit_linear_temporary,
    fit_permanent,
    fit_power_law,
    samples_from_orders,
)
from slippage.exceptions import CalibrationError, IdentifiabilityWarning, InsufficientDataError
from slippage.impact import LinearImpact, SquareRootLaw
from slippage.series import BarSeries
from slippage.types import Fill, Order, Side


def linear_sample(
    rng: np.random.Generator, n: int, *, eta: float = 2e-6, epsilon: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    rates = rng.uniform(1e3, 5e4, n)
    costs = epsilon + eta * rates + rng.normal(0.0, 0.005, n)
    return rates, costs


def power_sample(
    rng: np.random.Generator,
    n: int,
    *,
    y: float = 0.8,
    delta: float = 0.5,
    low: float = 1e-4,
    high: float = 0.1,
    noise: float = 5e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.exp(rng.uniform(np.log(low), np.log(high), n))
    sigma = rng.uniform(0.01, 0.04, n)
    costs = y * sigma * x**delta + rng.normal(0.0, noise, n)
    return x, costs, sigma


class TestEstimate:
    def test_interval_and_coverage(self) -> None:
        estimate = Estimate(1.0, 0.1)
        low, high = estimate.interval()
        assert low == pytest.approx(1.0 - 0.196, abs=1e-3)
        assert high == pytest.approx(1.0 + 0.196, abs=1e-3)
        assert estimate.covers(1.15)
        assert not estimate.covers(1.25)
        assert estimate.t_stat == pytest.approx(10.0)

    def test_t_stat_of_an_exact_value(self) -> None:
        assert Estimate(2.0, 0.0).t_stat == float("inf")
        assert Estimate(0.0, 0.0).t_stat == 0.0


class TestLinearTemporary:
    def test_recovers_known_coefficients(self) -> None:
        rng = np.random.default_rng(7)
        rates, costs = linear_sample(rng, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("error", IdentifiabilityWarning)
            fit = fit_linear_temporary(rates, costs)
        assert fit.warnings == ()
        assert abs(fit.eta.value - 2e-6) < 3 * fit.eta.std_error
        assert abs(fit.epsilon.value - 0.01) < 3 * fit.epsilon.std_error
        assert fit.n == 400
        assert 0.9 < fit.r_squared < 1.0
        model = fit.to_model(gamma=1e-7)
        assert isinstance(model, LinearImpact)
        assert model.gamma == 1e-7

    def test_standard_errors_are_honest(self) -> None:
        # Over repeated samples, the 95% interval should contain the truth
        # about 95% of the time. Too narrow would mean overconfident errors.
        rng = np.random.default_rng(11)
        trials = 400
        hits = sum(
            fit_linear_temporary(*linear_sample(rng, 60)).eta.covers(2e-6) for _ in range(trials)
        )
        assert 0.91 <= hits / trials <= 0.985

    def test_narrow_rates_warn(self) -> None:
        rng = np.random.default_rng(3)
        rates = rng.uniform(1e4, 1.01e4, 200)
        costs = 0.01 + 2e-6 * rates + rng.normal(0.0, 0.005, 200)
        with pytest.warns(IdentifiabilityWarning, match="cannot be told apart"):
            fit = fit_linear_temporary(rates, costs)
        assert any("cannot be told apart" in w for w in fit.warnings)

    def test_few_observations_warn(self) -> None:
        with pytest.warns(IdentifiabilityWarning, match="only 5 observations"):
            fit_linear_temporary([1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.0, 3.2, 3.9, 5.1])

    def test_negative_slope_is_refused_as_a_model(self) -> None:
        rates = np.linspace(1.0, 100.0, 50)
        costs = 1.0 - 0.001 * rates
        fit = fit_linear_temporary(rates, costs)
        assert fit.eta.value < 0
        with pytest.raises(CalibrationError, match="reward trading faster"):
            fit.to_model()

    @pytest.mark.parametrize(
        ("rates", "costs", "error"),
        [
            ([1.0, 2.0], [1.0, 2.0], InsufficientDataError),
            ([1.0, 2.0, 3.0], [1.0, 2.0], CalibrationError),
            ([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0], CalibrationError),
            ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], CalibrationError),
        ],
    )
    def test_bad_input(self, rates: list[float], costs: list[float], error: type) -> None:
        with pytest.raises(error):
            fit_linear_temporary(rates, costs)


class TestPermanent:
    def test_recovers_gamma(self) -> None:
        rng = np.random.default_rng(5)
        q = rng.uniform(1e3, 1e5, 300)
        moves = 3e-7 * q + rng.normal(0.0, 0.005, 300)
        gamma = fit_permanent(q, moves)
        assert gamma.covers(3e-7, z=3.0)
        assert gamma.std_error > 0.0

    def test_exact_data_has_zero_error(self) -> None:
        gamma = fit_permanent([1.0, 2.0, 4.0], [0.5, 1.0, 2.0])
        assert gamma.value == pytest.approx(0.5)
        assert gamma.std_error == pytest.approx(0.0, abs=1e-12)

    def test_degenerate_input(self) -> None:
        with pytest.raises(CalibrationError, match="zero"):
            fit_permanent([0.0, 0.0], [1.0, 2.0])
        with pytest.raises(InsufficientDataError):
            fit_permanent([1.0], [1.0])


class TestPowerLaw:
    def test_recovers_the_square_root_law(self) -> None:
        rng = np.random.default_rng(2026)
        x, costs, sigma = power_sample(rng, 500)
        with warnings.catch_warnings():
            warnings.simplefilter("error", IdentifiabilityWarning)
            fit = fit_power_law(x, costs, sigma)
        assert fit.delta.covers(0.5, z=3.0)
        assert fit.y.covers(0.8, z=3.0)
        assert fit.delta.std_error < 0.1
        law = fit.to_law()
        assert isinstance(law, SquareRootLaw)

    def test_recovers_a_non_square_root_exponent(self) -> None:
        rng = np.random.default_rng(99)
        x, costs, sigma = power_sample(rng, 500, y=1.2, delta=0.7, noise=2e-4)
        fit = fit_power_law(x, costs, sigma)
        assert fit.delta.covers(0.7, z=3.0)

    def test_exponent_standard_error_is_honest(self) -> None:
        rng = np.random.default_rng(17)
        trials = 200
        hits = 0
        for _ in range(trials):
            x, costs, sigma = power_sample(rng, 200)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IdentifiabilityWarning)
                hits += fit_power_law(x, costs, sigma, grid=100).delta.covers(0.5)
        assert 0.89 <= hits / trials <= 0.99

    def test_the_profile_finds_the_global_minimum(self) -> None:
        rng = np.random.default_rng(4)
        x, costs, sigma = power_sample(rng, 300)
        fit = fit_power_law(x, costs, sigma)
        for d in np.linspace(0.05, 1.5, 97):
            z = sigma * x**d
            y = float(z @ costs) / float(z @ z)
            assert fit.rss <= float(np.sum((costs - y * z) ** 2)) + 1e-15

    def test_narrow_size_range_warns(self) -> None:
        rng = np.random.default_rng(8)
        x, costs, sigma = power_sample(rng, 300, low=0.01, high=0.015)
        with pytest.warns(IdentifiabilityWarning) as record:
            fit = fit_power_law(x, costs, sigma)
        messages = [str(w.message) for w in record]
        assert any("range" in m for m in messages)
        # With 300 quiet observations the exponent's own standard error stays
        # moderate, but it is several times wider than over a broad range of
        # sizes, and the prefactor absorbs whatever the exponent gets wrong.
        wide = fit_power_law(*power_sample(np.random.default_rng(8), 300))
        assert fit.delta.std_error > 3 * wide.delta.std_error

    def test_noisy_narrow_data_warns_on_the_exponent_error(self) -> None:
        rng = np.random.default_rng(8)
        x, costs, sigma = power_sample(rng, 40, low=0.01, high=0.015, noise=2e-3)
        with pytest.warns(IdentifiabilityWarning) as record:
            fit = fit_power_law(x, costs, sigma)
        assert fit.delta.std_error > 0.25
        assert any("cannot distinguish" in str(w.message) for w in record)

    def test_optimum_on_the_bound_warns(self) -> None:
        rng = np.random.default_rng(12)
        x, costs, sigma = power_sample(rng, 300, delta=1.4, noise=1e-5)
        with pytest.warns(IdentifiabilityWarning, match="search bound"):
            fit_power_law(x, costs, sigma, bounds=(0.05, 1.0))

    def test_fixed_exponent_fits_only_the_prefactor(self) -> None:
        rng = np.random.default_rng(21)
        x, costs, sigma = power_sample(rng, 300, low=0.01, high=0.015)
        # A narrow range cannot identify the exponent, but it can still pin
        # down the prefactor once the exponent is supplied.
        with warnings.catch_warnings():
            warnings.simplefilter("error", IdentifiabilityWarning)
            fit = fit_power_law(x, costs, sigma, delta=0.5)
        assert fit.delta.value == 0.5
        assert fit.delta.std_error == 0.0
        assert fit.y.covers(0.8, z=3.0)

    @pytest.mark.parametrize(
        ("x", "c", "s", "match"),
        [
            ([0.01, 0.0, 0.02, 0.03], [1.0] * 4, [0.02] * 4, "participation"),
            ([0.01, 0.02, 0.03, 0.04], [1.0] * 4, [0.02, 0.0, 0.02, 0.02], "volatility"),
            ([0.01, 0.02], [1.0, 1.0], [0.02], "equal length"),
        ],
    )
    def test_bad_input(self, x: list[float], c: list[float], s: list[float], match: str) -> None:
        with pytest.raises(CalibrationError, match=match):
            fit_power_law(x, c, s)

    def test_too_few_observations(self) -> None:
        with pytest.raises(InsufficientDataError):
            fit_power_law([0.01, 0.02, 0.03], [1.0, 1.0, 1.0], [0.02, 0.02, 0.02])

    def test_invalid_bounds(self) -> None:
        rng = np.random.default_rng(1)
        x, costs, sigma = power_sample(rng, 50)
        with pytest.raises(CalibrationError, match="bounds"):
            fit_power_law(x, costs, sigma, bounds=(1.0, 0.5))


class TestSamplesFromOrders:
    @staticmethod
    def market(price: float = 100.0) -> BarSeries:
        return BarSeries([make_bar(n, price, spread=0.01, volume=25_000.0) for n in range(390)])

    def test_rate_participation_and_cost(self) -> None:
        bars = self.market()
        order = Order(
            symbol="ACME",
            side=Side.SELL,
            quantity=6_000.0,
            decision_time=minute(5),
            arrival_time=minute(10),
            fills=(
                Fill(timestamp=minute(12), quantity=2_000.0, price=99.95),
                Fill(timestamp=minute(39) + timedelta(seconds=30), quantity=4_000.0, price=99.92),
            ),
        )
        (sample,) = samples_from_orders(
            [order],
            {"ACME": bars},
            {"ACME": 9_750_000.0},
            {"ACME": 0.02},
            time_unit=timedelta(hours=1),
        )
        # Arrival at minute 10 to the end of the bar holding the last fill
        # (minute 40) is half an hour, so 6,000 shares is 12,000 an hour.
        assert sample.rate == pytest.approx(12_000.0)
        assert sample.participation == pytest.approx(6_000.0 / 9_750_000.0)
        average = (2_000 * 99.95 + 4_000 * 99.92) / 6_000
        # A sell below arrival is a positive cost.
        assert sample.cost_per_share == pytest.approx(100.0 - average)
        assert sample.cost_fraction == pytest.approx((100.0 - average) / 100.0)
        assert sample.volatility == 0.02

    def test_unfilled_orders_are_skipped(self) -> None:
        order = Order(
            symbol="ACME",
            side=Side.BUY,
            quantity=100.0,
            decision_time=minute(0),
            arrival_time=minute(1),
        )
        assert (
            samples_from_orders(
                [order],
                {"ACME": self.market()},
                {"ACME": 1e6},
                {"ACME": 0.02},
                time_unit=timedelta(days=1),
            )
            == []
        )

    def test_missing_market_data_is_named(self, simple_order: Order) -> None:
        with pytest.raises(CalibrationError, match="ACME"):
            samples_from_orders([simple_order], {}, {}, {}, time_unit=timedelta(days=1))

    def test_time_unit_must_be_positive(self, simple_order: Order) -> None:
        with pytest.raises(CalibrationError, match="time_unit"):
            samples_from_orders([simple_order], {}, {}, {}, time_unit=timedelta(0))

    def test_end_to_end_recovery_from_orders(self) -> None:
        """Synthesise executions obeying a square-root law and fit it back."""
        rng = np.random.default_rng(314)
        bars = self.market()
        daily_volume = 390 * 25_000.0
        orders, series, volumes, vols = [], {}, {}, {}
        for i in range(400):
            symbol = f"S{i:03d}"
            x = float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-2))))
            sigma = float(rng.uniform(0.01, 0.04))
            quantity = x * daily_volume
            cost = 0.6 * sigma * x**0.5 + float(rng.normal(0.0, 3e-4))
            side = Side.BUY if i % 2 == 0 else Side.SELL
            orders.append(
                Order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    decision_time=minute(0),
                    arrival_time=minute(30),
                    fills=(
                        Fill(
                            timestamp=minute(90),
                            quantity=quantity,
                            price=100.0 * (1.0 + side.sign * cost),
                        ),
                    ),
                )
            )
            series[symbol], volumes[symbol], vols[symbol] = bars, daily_volume, sigma
        samples = samples_from_orders(orders, series, volumes, vols, time_unit=timedelta(days=1))
        fit = fit_power_law(
            [s.participation for s in samples],
            [s.cost_fraction for s in samples],
            [s.volatility for s in samples],
        )
        assert fit.delta.covers(0.5, z=3.0)
        assert fit.y.covers(0.6, z=3.0)
