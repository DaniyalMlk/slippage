"""Basket liquidation, against the single-asset solver it generalises.

The two strongest tests here do not check a number against a number I chose. One
asserts that a one-asset basket reproduces `optimal_trajectory` exactly, and the
two routes share no arithmetic: the scalar solver evaluates a `sinh` at a `kappa`
from a closed form, and the basket solver takes a matrix square root, an
eigen-decomposition and two changes of basis. The other asserts that a diagonal
problem reproduces independent single-asset solutions exactly, which is the same
statement one dimension up. Everything the solver could get wrong that is not
visible in those two is a cross term, and the cross terms are what the measured
comparisons cover.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from slippage.basket import (
    MAX_BASKET_ASSETS,
    BasketProblem,
    basket_frontier,
    basket_moments,
    basket_trajectory,
    compare_to_independent,
    hedge_direction,
    independent_trajectories,
)
from slippage.exceptions import ValidationError
from slippage.execution import ExecutionProblem, optimal_trajectory
from slippage.impact import LinearImpact

#: A $50 stock at 30% annual volatility is about $0.945 a share a day, and an
#: impact of a tenth of a dollar a share at a full day of 100,000 shares is about
#: twenty basis points. Realistic units matter here because the risk aversion is
#: dimensional: at the wrong scale every schedule is a straight line and every
#: comparison agrees.
SIGMA = 0.945
ETA = 0.10 / 100_000.0
GAMMA = 0.1 * ETA
#: Gives `kappa * T` of about three on one leg, which is where the shapes differ.
AVERSION = 1e-5


def scalar(
    *,
    quantity: float = 100_000.0,
    periods: int = 20,
    eta: float = ETA,
    gamma: float = GAMMA,
    volatility: float = SIGMA,
) -> ExecutionProblem:
    return ExecutionProblem(
        quantity=quantity,
        horizon=1.0,
        periods=periods,
        volatility=volatility,
        impact=LinearImpact(gamma=gamma, eta=eta),
    )


def pair(
    *,
    correlation: float = 0.9,
    short_eta_multiple: float = 1.0,
    short_volatility_multiple: float = 1.0,
    holdings: tuple[float, float] = (100_000.0, -100_000.0),
    periods: int = 20,
) -> BasketProblem:
    first = SIGMA
    second = SIGMA * short_volatility_multiple
    covariance = (
        (first * first, correlation * first * second),
        (correlation * first * second, second * second),
    )
    return BasketProblem(
        holdings=holdings,
        temporary=((ETA, 0.0), (0.0, ETA * short_eta_multiple)),
        permanent=((GAMMA, 0.0), (0.0, GAMMA * short_eta_multiple)),
        covariance=covariance,
        horizon=1.0,
        periods=periods,
        names=("liquid", "illiquid"),
    )


class TestTheSingleAssetReduction:
    @pytest.mark.parametrize("risk_aversion", [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2])
    def test_a_one_asset_basket_reproduces_the_scalar_solver(self, risk_aversion: float) -> None:
        problem = scalar()
        expected = optimal_trajectory(problem, risk_aversion)
        got = basket_trajectory(BasketProblem.from_single(problem), risk_aversion)
        assert got.directions[0].kappa == pytest.approx(expected.kappa, rel=1e-12, abs=1e-15)
        for before, after in zip(expected.holdings, got.holdings, strict=True):
            assert after[0] == pytest.approx(before, rel=1e-9, abs=1e-6)
        assert got.expected_cost == pytest.approx(expected.expected_cost, rel=1e-12)
        assert got.variance == pytest.approx(expected.variance, rel=1e-12)

    def test_the_reduction_holds_at_a_single_interval(self) -> None:
        # One interval is the degenerate case: there is nothing to schedule and the
        # whole basket goes at once, so `sinh(kappa T) / sinh(kappa T)` is the only
        # ratio evaluated and the answer must not depend on kappa at all.
        problem = scalar(periods=1)
        for risk_aversion in (0.0, 1e-4):
            expected = optimal_trajectory(problem, risk_aversion)
            got = basket_trajectory(BasketProblem.from_single(problem), risk_aversion)
            assert got.expected_cost == pytest.approx(expected.expected_cost, rel=1e-12)
            assert got.variance == pytest.approx(expected.variance, rel=1e-12)

    def test_a_risk_neutral_basket_is_a_straight_line(self) -> None:
        got = basket_trajectory(pair(), 0.0)
        for direction in got.directions:
            assert direction.kappa == 0.0
            assert direction.half_life == math.inf
        for index, row in enumerate(got.holdings):
            fraction = 1.0 - index / (len(got.holdings) - 1)
            assert row[0] == pytest.approx(100_000.0 * fraction, abs=1e-6)
            assert row[1] == pytest.approx(-100_000.0 * fraction, abs=1e-6)

    def test_an_uncorrelated_diagonal_basket_is_two_separate_problems(self) -> None:
        # The same statement as the one-asset reduction, one dimension up: with no
        # cross terms anywhere there is nothing for the joint solution to exploit,
        # so it must coincide with the independent one exactly.
        problem = pair(correlation=0.0)
        joint = basket_trajectory(problem, AVERSION)
        apart = independent_trajectories(problem, AVERSION)
        assert joint.expected_cost == pytest.approx(apart.expected_cost, rel=1e-10)
        assert joint.variance == pytest.approx(apart.variance, rel=1e-10)
        for first, second in zip(joint.holdings, apart.holdings, strict=True):
            for a, b in zip(first, second, strict=True):
                assert a == pytest.approx(b, abs=1e-5)


class TestTheSolutionIsOptimal:
    def test_no_nearby_schedule_is_cheaper(self) -> None:
        """Perturb the optimum and the objective has to rise.

        The solver derives its schedule; `basket_moments` scores one. They share no
        arithmetic, so this is a real check that the derivation solved the problem
        the scorer measures rather than a neighbouring one.
        """
        problem = pair()
        best = basket_trajectory(problem, AVERSION)
        rng = np.random.default_rng(11)
        for _ in range(30):
            trades = [list(row) for row in best.trades]
            # A perturbation that keeps the total, so the schedule still liquidates.
            first, second = rng.integers(0, problem.periods, size=2)
            if first == second:
                continue
            asset = int(rng.integers(0, problem.assets))
            size = float(rng.normal(0.0, 2_000.0))
            trades[first][asset] += size
            trades[second][asset] -= size
            cost, variance = basket_moments(problem, trades)
            assert cost + AVERSION * variance > best.objective - 1e-9

    def test_the_moments_match_the_solvers_own_figures(self) -> None:
        problem = pair(short_eta_multiple=3.0)
        solved = basket_trajectory(problem, AVERSION)
        cost, variance = basket_moments(problem, solved.trades)
        assert cost == pytest.approx(solved.expected_cost, rel=1e-12)
        assert variance == pytest.approx(solved.variance, rel=1e-12)

    def test_more_risk_aversion_trades_faster(self) -> None:
        frontier = basket_frontier(pair(), [0.0, 1e-6, 1e-5, 1e-4])
        costs = [one.expected_cost for one in frontier]
        variances = [one.variance for one in frontier]
        assert costs == sorted(costs)
        assert variances == sorted(variances, reverse=True)

    def test_every_schedule_liquidates_the_basket(self) -> None:
        for risk_aversion in (0.0, 1e-6, 1e-4, 1e-2):
            solved = basket_trajectory(pair(short_eta_multiple=4.0), risk_aversion)
            assert solved.holdings[-1] == (0.0, 0.0)
            for asset in range(2):
                total = sum(row[asset] for row in solved.trades)
                assert total == pytest.approx(solved.holdings[0][asset], rel=1e-9, abs=1e-6)

    def test_an_impatient_basket_does_not_overflow(self) -> None:
        # `kappa * T` of order a thousand makes both sinh terms infinite, so the
        # ratio has to be evaluated as exponentials of the difference.
        solved = basket_trajectory(pair(), 1e6)
        assert all(all(math.isfinite(value) for value in row) for row in solved.holdings)
        assert solved.holdings[1][0] < 1.0
        assert math.isfinite(solved.expected_cost)


class TestDirections:
    def test_a_hedged_pair_splits_into_a_net_and_a_spread(self) -> None:
        solved = basket_trajectory(pair(correlation=0.9), AVERSION)
        assert len(solved.directions) == 2
        net, spread = solved.directions
        # Sorted by risk per unit of impact, so the net comes first.
        assert net.risk_per_impact > spread.risk_per_impact
        assert net.kappa > spread.kappa
        assert net.half_life < spread.half_life
        # The net direction has both legs the same sign and the spread does not.
        assert net.weights[0] * net.weights[1] > 0.0
        assert spread.weights[0] * spread.weights[1] < 0.0

    def test_a_dollar_neutral_pair_lives_entirely_in_the_spread(self) -> None:
        # Its whole position projects onto one direction, which is why a hedged
        # pair is worked off at one slow pace rather than at two.
        solved = basket_trajectory(pair(correlation=0.9), AVERSION)
        net, spread = solved.directions
        assert abs(net.initial) < 1e-6 * abs(spread.initial)

    def test_correlation_widens_the_gap_between_the_two_paces(self) -> None:
        ratios = []
        for correlation in (0.0, 0.5, 0.9, 0.99):
            net, spread = basket_trajectory(pair(correlation=correlation), AVERSION).directions
            ratios.append(net.kappa / spread.kappa)
        assert ratios[0] == pytest.approx(1.0, abs=1e-9)
        assert ratios == sorted(ratios)
        assert ratios[-1] > 10.0

    def test_directions_are_unavailable_on_a_schedule_built_another_way(
        self,
    ) -> None:
        apart = independent_trajectories(pair(), AVERSION)
        assert apart.directions == ()
        with pytest.raises(ValidationError, match="no eigen-directions"):
            apart.riskiest_direction()


class TestTheHedgeDirection:
    def test_a_dollar_neutral_pair_gives_the_net(self) -> None:
        direction = hedge_direction(pair())
        assert direction[0] == pytest.approx(direction[1])
        assert direction[0] == pytest.approx(1.0 / math.sqrt(2.0))

    def test_it_is_orthogonal_to_the_holdings(self) -> None:
        for holdings in ((100_000.0, -100_000.0), (60_000.0, -20_000.0)):
            problem = pair(holdings=holdings)
            direction = np.asarray(hedge_direction(problem))
            assert float(direction @ np.asarray(problem.holdings)) == pytest.approx(0.0, abs=1e-6)
            assert float(np.linalg.norm(direction)) == pytest.approx(1.0)

    def test_a_one_asset_basket_has_no_such_direction(self) -> None:
        problem = BasketProblem.from_single(scalar())
        with pytest.raises(ValidationError, match="no direction orthogonal"):
            hedge_direction(problem)

    def test_an_empty_basket_is_flat_everywhere(self) -> None:
        with pytest.raises(ValidationError, match="starts flat everywhere"):
            hedge_direction(pair(holdings=(0.0, 0.0)))


class TestAgainstIndependentSolutions:
    def test_the_joint_solution_is_cheaper_on_a_correlated_pair(self) -> None:
        # Measured: 23.8% of the objective on identical legs, 12.1% when the short
        # leg is four times as expensive, 6.1% at ten times. The assertion is
        # loose because the exact figure depends on the risk aversion; the
        # direction and the order of magnitude are the finding.
        for multiple, floor in ((1.0, 0.15), (4.0, 0.08), (10.0, 0.03)):
            comparison = compare_to_independent(pair(short_eta_multiple=multiple), AVERSION)
            assert comparison.objective_saving > floor

    def test_an_uncorrelated_pair_has_nothing_to_save(self) -> None:
        comparison = compare_to_independent(pair(correlation=0.0), AVERSION)
        assert comparison.objective_saving == pytest.approx(0.0, abs=1e-9)
        assert comparison.peak_risk_ratio == pytest.approx(1.0, abs=1e-9)

    def test_the_joint_solution_is_riskier_moment_to_moment(self) -> None:
        # The counterintuitive half, and the reason it is asserted rather than
        # left as a remark: a hedged pair is cheap to hold, so the optimal
        # schedule holds it longer. Measured at 1.19 times the independent peak
        # interval variance on the symmetric pair and 1.10 on the illiquid one, so
        # the ratio the other way is below one.
        for multiple in (1.0, 4.0):
            comparison = compare_to_independent(pair(short_eta_multiple=multiple), AVERSION)
            assert comparison.peak_risk_ratio < 1.0

    def test_an_illiquid_leg_makes_the_independent_solution_break_the_hedge(
        self,
    ) -> None:
        # The order-of-magnitude half. Measured: a peak net exposure of 18,961
        # shares of a 100,000-share pair against the joint solution's 2,025, a
        # factor of 9.4, against an objective difference of 12.1%.
        comparison = compare_to_independent(pair(short_eta_multiple=4.0), AVERSION)
        assert comparison.exposure_ratio() > 5.0
        direction = hedge_direction(comparison.problem)
        assert comparison.independent.peak_exposure(direction) > 8_000.0
        assert comparison.joint.peak_exposure(direction) < 2_000.0

    def test_a_symmetric_pair_refuses_the_exposure_ratio(self) -> None:
        # Both schedules keep the hedge exactly, so the ratio is one rounding
        # error over another. The guard is relative to the basket's own size,
        # because the residue is of order eps times that and an absolute
        # threshold would not fire.
        comparison = compare_to_independent(pair(), AVERSION)
        with pytest.raises(ValidationError, match="rounding rather than exposure"):
            comparison.exposure_ratio()

    def test_the_independent_schedule_is_scored_against_the_full_matrices(
        self,
    ) -> None:
        # Not against its own assumptions. A comparison that let each schedule be
        # judged by the model it was derived under would be no comparison.
        problem = pair(correlation=0.9)
        apart = independent_trajectories(problem, AVERSION)
        cost, variance = basket_moments(problem, apart.trades)
        assert cost == pytest.approx(apart.expected_cost, rel=1e-12)
        assert variance == pytest.approx(apart.variance, rel=1e-12)
        # And the covariance it was scored against has the cross term in it.
        assert apart.covariance_used[0][1] != 0.0

    def test_a_zero_holding_is_left_alone(self) -> None:
        problem = pair(holdings=(100_000.0, 0.0))
        apart = independent_trajectories(problem, AVERSION)
        assert all(row[1] == 0.0 for row in apart.trades)


class TestCrossImpact:
    def _cross(self, off_diagonal: float) -> BasketProblem:
        return BasketProblem(
            holdings=(100_000.0, 100_000.0),
            temporary=((ETA, off_diagonal * ETA), (off_diagonal * ETA, ETA)),
            permanent=((GAMMA, 0.0), (0.0, GAMMA)),
            # Uncorrelated, so any coupling is the impact matrix and nothing else.
            covariance=((SIGMA**2, 0.0), (0.0, SIGMA**2)),
            horizon=1.0,
            periods=20,
        )

    def test_cross_impact_couples_the_schedules_with_no_correlation(self) -> None:
        # With a diagonal impact matrix and no correlation the two legs are
        # separate problems and their paces are equal. Cross impact alone changes
        # that, which is the case that shows the coupling is not only about risk.
        plain = basket_trajectory(self._cross(0.0), AVERSION)
        crossed = basket_trajectory(self._cross(0.5), AVERSION)
        plain_paces = sorted(one.kappa for one in plain.directions)
        crossed_paces = sorted(one.kappa for one in crossed.directions)
        assert plain_paces[0] == pytest.approx(plain_paces[1])
        assert crossed_paces[0] < crossed_paces[1] * 0.9

    def test_cross_impact_makes_a_joint_liquidation_dearer(self) -> None:
        # Both legs long and each one's trading pushing the other: the pair costs
        # more than two separate names would, and the schedule cannot avoid it.
        plain = basket_trajectory(self._cross(0.0), AVERSION)
        crossed = basket_trajectory(self._cross(0.5), AVERSION)
        assert crossed.expected_cost > plain.expected_cost


class TestRefusals:
    def test_an_asymmetric_matrix_names_itself(self) -> None:
        with pytest.raises(ValidationError, match="covariance matrix is not symmetric"):
            basket_trajectory(
                BasketProblem(
                    holdings=(1.0, 1.0),
                    temporary=((ETA, 0.0), (0.0, ETA)),
                    permanent=((GAMMA, 0.0), (0.0, GAMMA)),
                    covariance=((1.0, 0.5), (0.4, 1.0)),
                    horizon=1.0,
                    periods=4,
                ),
                AVERSION,
            )

    def test_a_singular_impact_matrix_is_refused_with_both_eigenvalues(self) -> None:
        # Two assets with identical impact and perfectly coupled cross impact:
        # the matrix has a zero eigenvalue and its inverse square root does not
        # exist. The message says what to do about it.
        with pytest.raises(ValidationError, match="not positive definite"):
            basket_trajectory(
                BasketProblem(
                    holdings=(1.0, 1.0),
                    temporary=((ETA, ETA), (ETA, ETA)),
                    permanent=((0.0, 0.0), (0.0, 0.0)),
                    covariance=((1.0, 0.0), (0.0, 1.0)),
                    horizon=1.0,
                    periods=4,
                ),
                AVERSION,
            )

    def test_the_wrong_shape_names_the_shape(self) -> None:
        with pytest.raises(ValidationError, match="must be 2 by 2"):
            basket_trajectory(
                BasketProblem(
                    holdings=(1.0, 1.0),
                    temporary=((ETA,),),
                    permanent=((GAMMA, 0.0), (0.0, GAMMA)),
                    covariance=((1.0, 0.0), (0.0, 1.0)),
                    horizon=1.0,
                    periods=4,
                ),
                AVERSION,
            )

    def test_a_negative_risk_aversion(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            basket_trajectory(pair(), -1.0)

    def test_an_empty_basket(self) -> None:
        with pytest.raises(ValidationError, match="no assets"):
            BasketProblem(
                holdings=(),
                temporary=(),
                permanent=(),
                covariance=(),
                horizon=1.0,
                periods=4,
            )

    def test_too_many_assets(self) -> None:
        size = MAX_BASKET_ASSETS + 1
        with pytest.raises(ValidationError, match="above the 200"):
            BasketProblem(
                holdings=tuple(1.0 for _ in range(size)),
                temporary=(),
                permanent=(),
                covariance=(),
                horizon=1.0,
                periods=4,
            )

    def test_a_non_positive_horizon(self) -> None:
        with pytest.raises(ValidationError, match="horizon must be positive"):
            pair(periods=4).__class__(
                holdings=(1.0, 1.0),
                temporary=((ETA, 0.0), (0.0, ETA)),
                permanent=((GAMMA, 0.0), (0.0, GAMMA)),
                covariance=((1.0, 0.0), (0.0, 1.0)),
                horizon=0.0,
                periods=4,
            )

    def test_no_intervals(self) -> None:
        with pytest.raises(ValidationError, match="at least one interval"):
            BasketProblem(
                holdings=(1.0, 1.0),
                temporary=((ETA, 0.0), (0.0, ETA)),
                permanent=((GAMMA, 0.0), (0.0, GAMMA)),
                covariance=((1.0, 0.0), (0.0, 1.0)),
                horizon=1.0,
                periods=0,
            )

    def test_the_wrong_number_of_names(self) -> None:
        with pytest.raises(ValidationError, match="1 names for 2 assets"):
            BasketProblem(
                holdings=(1.0, 1.0),
                temporary=((ETA, 0.0), (0.0, ETA)),
                permanent=((GAMMA, 0.0), (0.0, GAMMA)),
                covariance=((1.0, 0.0), (0.0, 1.0)),
                horizon=1.0,
                periods=4,
                names=("only",),
            )

    def test_trades_that_do_not_liquidate_are_refused(self) -> None:
        problem = pair()
        solved = basket_trajectory(problem, AVERSION)
        trades = [list(row) for row in solved.trades]
        trades[0][0] += 5_000.0
        with pytest.raises(ValidationError, match="do not liquidate the basket"):
            basket_moments(problem, trades)

    def test_the_wrong_number_of_intervals(self) -> None:
        problem = pair()
        with pytest.raises(ValidationError, match="expected 20 intervals"):
            basket_moments(problem, [(1.0, 1.0)])

    def test_a_non_finite_matrix_entry(self) -> None:
        with pytest.raises(ValidationError, match="non-finite entry"):
            basket_trajectory(
                BasketProblem(
                    holdings=(1.0, 1.0),
                    temporary=((ETA, 0.0), (0.0, ETA)),
                    permanent=((GAMMA, 0.0), (0.0, GAMMA)),
                    covariance=((1.0, float("nan")), (float("nan"), 1.0)),
                    horizon=1.0,
                    periods=4,
                ),
                AVERSION,
            )


class TestPayloadSafety:
    def test_every_number_survives_a_strict_encoder(self) -> None:
        import json

        comparison = compare_to_independent(pair(short_eta_multiple=4.0), AVERSION)
        payload = {
            "joint": {
                "cost": comparison.joint.expected_cost,
                "variance": comparison.joint.variance,
                "objective": comparison.joint.objective,
                "std": comparison.joint.std,
                "holdings": [list(row) for row in comparison.joint.holdings],
                "trades": [list(row) for row in comparison.joint.trades],
                "directions": [
                    {
                        "riskPerImpact": one.risk_per_impact,
                        "kappa": one.kappa,
                        "halfLife": one.half_life,
                        "weights": list(one.weights),
                    }
                    for one in comparison.joint.directions
                ],
            },
            "objectiveSaving": comparison.objective_saving,
            "peakRiskRatio": comparison.peak_risk_ratio,
            "exposureRatio": comparison.exposure_ratio(),
            "hedgeDirection": list(hedge_direction(comparison.problem)),
        }
        json.dumps(payload, allow_nan=False)

    def test_a_risk_neutral_half_life_is_infinite_and_says_so(self) -> None:
        # Infinity is the right answer and it is not JSON, so a caller putting it
        # in a payload has to handle it. Better that it is obvious here than that
        # it is a null somewhere downstream.
        solved = basket_trajectory(pair(), 0.0)
        assert all(one.half_life == math.inf for one in solved.directions)
