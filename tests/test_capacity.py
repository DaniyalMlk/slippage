"""Capacity: the size at which executing the trade spends the reason for it.

**Every break-even is checked against a root of the library's own cost, not
against the closed form it came from.** The fixed-horizon formula is exact
because the objective is ``epsilon X + A X^2``, and the test asserts that by
bisecting :func:`slippage.execution.linear_trajectory`'s cost and comparing —
1e-15 relative, not a tolerance chosen to pass.

**The quadratic claim is tested where it could fail rather than where it is
obvious.** Evaluating the coefficient at sizes four orders of magnitude apart
has to give the same number, including with a mean-variance penalty, because
the variance is homogeneous of degree two as well.

**The two conventions are separated by a measurement, not an argument.** The
log-log slope of cost per share against size is one at every decade with the
horizon fixed, and climbs from near zero towards one with the participation rate
fixed. With the permanent impact removed the fixed-participation cost per share
is the same eight digits at a million shares and at a trillion, and capacity is
refused as unbounded rather than returned as the ceiling.

**The size-independence of the optimal horizon is the sharpest claim here**, so
it is asserted bit for bit across a fortyfold range of size, and the ``sqrt(3)``
relation to the Almgren-Chriss half-life is asserted alongside, since the two
together are what make the closed form more than a curve fit.

Then the refusals, which in this module carry most of the content: an alpha
below the floor, an alpha below the commission, a capacity past the ceiling, and
an optimal horizon asked of a risk-neutral mandate.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from slippage.capacity import (
    BISECTION_STEPS,
    DEFAULT_CEILING_IN_VOLUME,
    MAX_PERIODS,
    Capacity,
    Horizon,
    Mandate,
    Point,
    UnboundedCapacityError,
    alpha_floor,
    break_even_at_horizon,
    break_even_at_participation,
    break_even_continuum,
    capacity_curve,
    cost_elasticity,
    curvature,
    uniform_optimal_horizon,
)
from slippage.exceptions import ValidationError
from slippage.execution import ExecutionProblem, linear_trajectory
from slippage.impact import LinearImpact

VOLUME = 5_000_000.0
IMPACT = LinearImpact(gamma=2e-8, eta=1e-7, epsilon=0.005)
ALPHA = 0.05


def mandate(**overrides: object) -> Mandate:
    fields: dict[str, object] = {
        "alpha": ALPHA,
        "volume": VOLUME,
        "volatility": 0.30,
        "impact": IMPACT,
    }
    fields.update(overrides)
    return Mandate(**fields)  # type: ignore[arg-type]


BASE = mandate()


def bisect_net_zero(
    book: Mandate,
    *,
    horizon: float | None = None,
    participation: float | None = None,
    ceiling: float = 50.0 * VOLUME,
) -> float:
    """A root of the library's own cost, with nothing from `capacity` in it.

    Deliberately written out here rather than imported: the point is to have a
    second implementation, and one that reaches `linear_trajectory` directly.
    """
    rate = None if participation is None else participation * book.volume

    def net_per_share(quantity: float) -> float:
        at = horizon if rate is None else quantity / rate
        assert at is not None
        periods = max(2, min(MAX_PERIODS, math.ceil(at * book.periods_per_unit_time)))
        problem = ExecutionProblem(
            quantity=quantity,
            horizon=at,
            periods=periods,
            volatility=book.volatility,
            impact=book.impact,
        )
        return book.alpha - linear_trajectory(problem).expected_cost / quantity

    low, high = 1.0, ceiling
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if net_per_share(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# -- the fixed-horizon closed form -------------------------------------------


@pytest.mark.parametrize("horizon", [0.25, 1.0, 5.0, 20.0])
def test_the_fixed_horizon_break_even_matches_a_root_of_the_library_cost(
    horizon: float,
) -> None:
    """Exact, because the objective is quadratic. Not a tolerance."""
    found = break_even_at_horizon(BASE, horizon, uniform=True)
    assert found.quantity == pytest.approx(bisect_net_zero(BASE, horizon=horizon), rel=1e-13)


@pytest.mark.parametrize("horizon", [0.25, 1.0, 5.0, 20.0])
def test_the_break_even_size_costs_exactly_the_alpha(horizon: float) -> None:
    """Which is the definition, and the only check that does not restate it."""
    found = break_even_at_horizon(BASE, horizon, uniform=True)
    assert found.cost_per_share == pytest.approx(BASE.alpha, rel=1e-12)
    assert BASE.net_alpha(found.quantity, horizon, uniform=True) == pytest.approx(0.0, abs=1e-6)
    assert found.volume_days == pytest.approx(found.quantity / VOLUME, rel=1e-15)
    assert found.participation is None
    assert found.curvature is not None


@pytest.mark.parametrize("horizon", [0.25, 1.0, 5.0, 20.0])
def test_the_continuum_expression_is_below_the_exact_one_by_the_interval_count(
    horizon: float,
) -> None:
    """The ``1 - 1/N`` the continuum drops, measured rather than waved at.

    Small — between 0.017% and 0.034% here — and worth knowing, because it is
    the whole difference between computing capacity from this library's cost
    functions and computing it from the textbook formula.
    """
    exact = break_even_at_horizon(BASE, horizon, uniform=True).quantity
    gap = break_even_continuum(BASE, horizon=horizon) / exact - 1.0
    assert -4e-4 < gap < -1e-4


@pytest.mark.parametrize("size", [1e3, 1e5, 1e7, 1e9])
def test_the_curvature_is_the_same_number_at_every_size(size: float) -> None:
    """The quadratic claim, tested where it could fail.

    The coefficient is read at one reference size and used at every other, so
    it has to be independent of the reference. Asserted across six orders of
    magnitude and with a risk penalty, where the variance supplies a second
    quadratic term that could have had a different degree.
    """
    for book in (BASE, mandate(risk_aversion=1e-8)):
        coefficient = curvature(book, 1.0, uniform=True)
        objective = book.objective(size, 1.0, uniform=True)
        implied = (objective - book.impact.epsilon * size) / (size * size)
        assert implied == pytest.approx(coefficient, rel=1e-12)


def test_a_risk_penalty_raises_the_curvature_and_lowers_the_capacity() -> None:
    sizes = []
    for aversion in (0.0, 1e-9, 1e-8, 1e-7):
        book = mandate(risk_aversion=aversion)
        sizes.append(break_even_at_horizon(book, 1.0).quantity)
    for larger, smaller in pairwise(sizes):
        assert smaller < larger


# -- the fixed-participation convention --------------------------------------


@pytest.mark.parametrize("participation", [0.02, 0.05])
def test_the_participation_break_even_matches_a_root_of_the_library_cost(
    participation: float,
) -> None:
    found = break_even_at_participation(BASE, participation)
    assert found.quantity == pytest.approx(
        bisect_net_zero(BASE, participation=participation), rel=1e-9
    )
    assert found.cost_per_share == pytest.approx(BASE.alpha, rel=1e-7)
    assert found.participation == participation
    assert found.curvature is None
    assert found.horizon == pytest.approx(found.quantity / (participation * VOLUME), rel=1e-15)


@pytest.mark.parametrize(
    ("participation", "floor"),
    [(0.02, 0.015), (0.05, 0.030), (0.10, 0.055), (0.25, 0.130)],
)
def test_the_alpha_floor_is_the_commission_plus_the_rates_own_impact(
    participation: float, floor: float
) -> None:
    assert alpha_floor(BASE, participation) == pytest.approx(floor, rel=1e-12)


@pytest.mark.parametrize("participation", [0.10, 0.25])
def test_an_alpha_below_the_floor_leaves_no_viable_size(participation: float) -> None:
    """And says so, rather than returning a very small capacity.

    At these rates the temporary impact of the rate alone exceeds the whole
    edge, and no horizon helps because the rate is what chose the horizon.
    """
    assert alpha_floor(BASE, participation) > BASE.alpha
    with pytest.raises(UnboundedCapacityError, match="at or below the floor"):
        break_even_at_participation(BASE, participation)


def test_the_participation_continuum_gap_is_exactly_one_over_the_interval_cap() -> None:
    """A cleaner number than the fixed-horizon case, and for a reason.

    The horizon at these sizes is long enough that the interval count sits on
    its cap, so the only difference between the exact and continuum
    coefficients is the single ``1 - 1/N`` factor. The gap is then
    ``-1/MAX_PERIODS`` and nothing else, which is a sharper statement than a
    range.
    """
    for participation in (0.02, 0.05):
        exact = break_even_at_participation(BASE, participation).quantity
        gap = break_even_continuum(BASE, participation=participation) / exact - 1.0
        assert gap == pytest.approx(-1.0 / MAX_PERIODS, rel=5e-3)


def test_the_headroom_is_the_alpha_the_permanent_impact_spends() -> None:
    for participation in (0.02, 0.05):
        found = break_even_at_participation(BASE, participation)
        assert found.headroom == pytest.approx(
            BASE.alpha - alpha_floor(BASE, participation), rel=1e-6
        )
        assert found.headroom > 0.0


# -- the measurement that separates the two conventions ----------------------


def test_cost_per_share_is_linear_in_the_size_at_a_fixed_horizon() -> None:
    """Slope one at every decade, to four decimal places."""
    curve = capacity_curve(BASE, [1e4, 1e5, 1e6, 1e7, 1e8], horizon=1.0)
    for slope in cost_elasticity(curve, IMPACT.epsilon):
        assert slope == pytest.approx(1.0, abs=5e-5)


def test_cost_per_share_climbs_from_flat_to_linear_at_a_fixed_rate() -> None:
    """The whole content of the convention difference, in four numbers.

    Near zero where the floor dominates, approaching one only once the
    permanent term has taken over. A capacity figure quoted without saying
    which variable was fixed is quoting one of two different functions.
    """
    curve = capacity_curve(BASE, [1e4, 1e5, 1e6, 1e7, 1e8], participation=0.05)
    slopes = cost_elasticity(curve, IMPACT.epsilon)
    assert slopes == pytest.approx((0.0153, 0.1291, 0.5527, 0.9138), abs=5e-4)
    for flatter, steeper in pairwise(slopes):
        assert steeper > flatter


def test_with_no_permanent_impact_the_rate_convention_imposes_no_limit() -> None:
    """Which is the strongest form of the claim and the easiest to check.

    Cost per share is the floor and nothing else, to eight decimals, across six
    orders of magnitude of size. Capacity is then refused rather than returned
    as the ceiling, because the ceiling is an argument.
    """
    free = mandate(impact=LinearImpact(gamma=0.0, eta=1e-7, epsilon=0.005))
    floor = alpha_floor(free, 0.05)
    for point in capacity_curve(free, [1e6, 1e9, 1e12], participation=0.05):
        assert point.cost_per_share == pytest.approx(floor, abs=1e-9)
    with pytest.raises(UnboundedCapacityError, match="no permanent impact"):
        break_even_at_participation(free, 0.05)
    with pytest.raises(UnboundedCapacityError, match="infinite"):
        break_even_continuum(free, participation=0.05)


def test_a_longer_horizon_buys_capacity_with_diminishing_returns() -> None:
    """Linear in the horizon while temporary impact dominates, then flat.

    Four times the horizon from 0.25 to 1.0 gives 3.7 times the capacity; four
    times again gives 3.7 times less of an improvement, and from 5 to 20 only
    doubles it, because by then the permanent impact is most of the cost.
    """
    sizes = [break_even_at_horizon(BASE, T, uniform=True).quantity for T in (0.25, 1.0, 5.0, 20.0)]
    ratios = [later / earlier for earlier, later in pairwise(sizes)]
    assert ratios[0] == pytest.approx(3.73, abs=0.02)
    assert ratios[1] == pytest.approx(3.67, abs=0.02)
    assert ratios[2] == pytest.approx(2.00, abs=0.02)
    for faster, slower in pairwise(ratios):
        assert slower < faster


def test_the_curve_reports_the_horizon_it_used() -> None:
    fixed = capacity_curve(BASE, [1e5, 1e6], horizon=2.0)
    assert all(point.horizon == 2.0 for point in fixed)
    rate = capacity_curve(BASE, [1e5, 1e6], participation=0.05)
    assert rate[1].horizon == pytest.approx(10.0 * rate[0].horizon, rel=1e-12)
    for point in (*fixed, *rate):
        assert isinstance(point, Point)
        assert point.net_alpha_per_share == pytest.approx(
            BASE.alpha - point.cost_per_share, rel=1e-15
        )


# -- the optimal horizon -----------------------------------------------------


@pytest.mark.parametrize("aversion", [1e-9, 1e-8, 1e-7, 1e-6])
def test_the_optimal_horizon_does_not_depend_on_the_order_size(
    aversion: float,
) -> None:
    """Cost falls as ``1 / T`` and the risk penalty rises as ``T``, and both carry
    the same ``X^2``, so the size cancels out of the first-order condition
    entirely.

    **Not bit for bit, and the reason is worth recording.** The objective scales
    as ``X^2``, so near a flat optimum the golden-section search's comparisons
    are decided by rounding in a number whose magnitude depends on the size, and
    a fortyfold range of size moves the answer in its eighth significant figure.
    The invariance is exact in the algebra and good to better than one part in a
    million in the search -- 1.3e-07 at worst over these four risk aversions --
    which is the honest version of the claim.
    """
    book = mandate(risk_aversion=aversion)
    horizons = [uniform_optimal_horizon(book, size).horizon for size in (5e5, 2e6, 2e7)]
    for first, second in pairwise(horizons):
        assert second == pytest.approx(first, rel=1e-6)
    assert max(horizons) / min(horizons) - 1.0 < 5e-7


@pytest.mark.parametrize("aversion", [1e-9, 1e-8, 1e-7, 1e-6])
def test_the_optimal_horizon_matches_its_closed_form(aversion: float) -> None:
    book = mandate(risk_aversion=aversion)
    found = uniform_optimal_horizon(book, 2e6)
    predicted = math.sqrt(3.0 * IMPACT.eta / (aversion * book.volatility**2))
    assert found.horizon == pytest.approx(predicted, rel=1e-3)
    assert isinstance(found, Horizon)
    assert found.interior


@pytest.mark.parametrize("aversion", [1e-9, 1e-8, 1e-7, 1e-6])
def test_the_optimal_horizon_is_root_three_half_lives(aversion: float) -> None:
    """The relation that ties the two schedules together.

    ``kappa = sqrt(lambda sigma^2 / eta_tilde)`` and the uniform optimum is
    ``sqrt(3 eta_tilde / (lambda sigma^2))``, so the ratio is ``sqrt(3)``
    exactly in the continuum. Measured within a quarter of a per cent, closing
    as the intervals shorten relative to the horizon.
    """
    found = uniform_optimal_horizon(mandate(risk_aversion=aversion), 2e6)
    assert found.half_lives == pytest.approx(math.sqrt(3.0), rel=3e-3)
    assert found.half_lives > math.sqrt(3.0)


def test_the_optimal_horizon_really_is_a_maximum() -> None:
    """Checked by finite difference on the objective, not by trusting a search."""
    book = mandate(risk_aversion=1e-8)
    found = uniform_optimal_horizon(book, 2e6)
    best = book.net_alpha(2e6, found.horizon, uniform=True)
    assert found.net_alpha == pytest.approx(best, rel=1e-12)
    for step in (0.02, 0.2, 2.0):
        assert book.net_alpha(2e6, found.horizon - step, uniform=True) < best
        assert book.net_alpha(2e6, found.horizon + step, uniform=True) < best


def test_a_risk_neutral_mandate_has_no_optimal_horizon() -> None:
    """And is told so, rather than handed the end of a search.

    With no risk penalty every extra minute is free, so the objective falls
    monotonically towards the permanent impact alone and the maximiser is at
    infinity.
    """
    with pytest.raises(UnboundedCapacityError, match="no risk aversion"):
        uniform_optimal_horizon(BASE, 1e6)
    costs = [BASE.objective(1e6, T, uniform=True) for T in (1.0, 10.0, 100.0, 1000.0)]
    for dearer, cheaper in pairwise(costs):
        assert cheaper < dearer
    permanent = 0.5 * IMPACT.gamma * 1e6 * 1e6 + IMPACT.epsilon * 1e6
    # It does not reach the permanent impact, and the residual is not slack in
    # the limit: at a thousand units of time the interval count is on its cap,
    # so tau is half a unit and the temporary term eta_tilde X^2 / T is still
    # 95 of the 15,095. Asserted as that expression rather than as a tolerance,
    # since a tolerance would hide which of the two it was.
    tau = 1000.0 / MAX_PERIODS
    eta_tilde = IMPACT.eta - 0.5 * IMPACT.gamma * tau
    assert costs[-1] == pytest.approx(permanent + eta_tilde * 1e12 / 1000.0, rel=1e-9)
    assert costs[-1] - permanent == pytest.approx(95.0, abs=0.5)


def test_the_optimal_schedule_declines_the_extra_time() -> None:
    """Which is why there is no interior optimum for it, and the module says so.

    Net alpha rises monotonically in the nominal horizon and saturates, while
    the half-life the schedule actually trades over barely moves. The risk
    aversion has already chosen the effective horizon.
    """
    book = mandate(risk_aversion=1e-8)
    nets = []
    lives = []
    for horizon in (20.0, 50.0, 100.0, 200.0):
        nets.append(book.net_alpha(2e6, horizon))
        lives.append(1.0 / book.problem(2e6, horizon).kappa(book.risk_aversion))
    for earlier, later in pairwise(nets):
        assert later > earlier
    # Saturating: the last step is a small fraction of the first.
    assert (nets[3] - nets[2]) < 0.2 * (nets[1] - nets[0])
    # The half-life drifts down as the nominal horizon grows, because the
    # interval count is capped: tau rises, eta_tilde falls, kappa rises. A real
    # effect of the discretisation and small, 10.54 days down to 10.49.
    assert lives == pytest.approx((10.536, 10.528, 10.515, 10.488), abs=5e-3)
    for earlier, later in pairwise(lives):
        assert later < earlier


# -- refusals ----------------------------------------------------------------


def test_an_alpha_below_the_commission_is_refused_at_construction() -> None:
    """Because no horizon and no rate can help, so there is nothing to compute."""
    with pytest.raises(ValidationError, match="does not clear the per-share cost"):
        mandate(alpha=IMPACT.epsilon)
    with pytest.raises(ValidationError, match="does not clear the per-share cost"):
        mandate(alpha=0.004)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("volume", 0.0),
        ("volume", -1.0),
        ("volatility", -0.1),
        ("risk_aversion", -1e-9),
        ("periods_per_unit_time", 0),
        ("alpha", math.inf),
        ("volatility", math.nan),
    ],
)
def test_a_mandate_refuses_inputs_with_no_market_behind_them(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        mandate(**{field: value})


@pytest.mark.parametrize("participation", [0.0, -0.1, 1.5, math.nan])
def test_a_participation_outside_the_unit_interval_is_refused(
    participation: float,
) -> None:
    with pytest.raises(ValidationError, match="fraction of volume"):
        alpha_floor(BASE, participation)
    with pytest.raises(ValidationError, match="fraction of volume"):
        break_even_at_participation(BASE, participation)


@pytest.mark.parametrize("horizon", [0.0, -1.0, math.nan])
def test_a_non_positive_horizon_is_refused(horizon: float) -> None:
    with pytest.raises(ValidationError, match="horizon must be positive"):
        BASE.periods_for(horizon)
    with pytest.raises(ValidationError, match="horizon must be positive"):
        break_even_continuum(BASE, horizon=horizon)


def test_exactly_one_of_horizon_and_participation_is_required() -> None:
    for kwargs in ({}, {"horizon": 1.0, "participation": 0.05}):
        with pytest.raises(ValidationError, match="exactly one"):
            break_even_continuum(BASE, **kwargs)
        with pytest.raises(ValidationError, match="exactly one"):
            capacity_curve(BASE, [1e5], **kwargs)


def test_a_capacity_past_the_ceiling_is_refused_rather_than_clipped() -> None:
    """The ceiling is an argument, so returning it would be returning an input."""
    with pytest.raises(UnboundedCapacityError, match="still"):
        break_even_at_participation(BASE, 0.02, ceiling=1_000_000.0)
    # And the default ceiling, which is fifty periods of volume, finds it.
    found = break_even_at_participation(BASE, 0.02)
    assert 1_000_000.0 < found.quantity < DEFAULT_CEILING_IN_VOLUME * VOLUME
    assert isinstance(found, Capacity)
    with pytest.raises(ValidationError, match="ceiling must be a positive size"):
        break_even_at_participation(BASE, 0.02, ceiling=0.0)


def test_a_curve_refuses_an_empty_or_non_positive_set_of_sizes() -> None:
    with pytest.raises(ValidationError, match="at least one size"):
        capacity_curve(BASE, [], horizon=1.0)
    with pytest.raises(ValidationError, match="every size must be positive"):
        capacity_curve(BASE, [1e5, 0.0], horizon=1.0)
    with pytest.raises(ValidationError, match="at least two points"):
        cost_elasticity(capacity_curve(BASE, [1e5], horizon=1.0), IMPACT.epsilon)


def test_an_elasticity_refuses_a_fixed_cost_that_swallows_the_whole_cost() -> None:
    """Which is what a wrong ``epsilon`` looks like, and it would otherwise
    produce the logarithm of a negative number."""
    curve = capacity_curve(BASE, [1e4, 1e5], horizon=1.0)
    with pytest.raises(ValidationError, match="fixed per-share cost"):
        cost_elasticity(curve, 1.0)


def test_the_interval_count_is_capped_and_at_least_two() -> None:
    assert BASE.periods_for(1e-9) == 2
    assert BASE.periods_for(1.0) == 390
    assert BASE.periods_for(1e9) == MAX_PERIODS


def test_everything_returned_is_finite() -> None:
    found = break_even_at_horizon(BASE, 1.0, uniform=True)
    assert found.curvature is not None
    for value in (
        found.quantity,
        found.volume_days,
        found.horizon,
        found.cost_per_share,
        found.floor,
        found.curvature,
        found.headroom,
    ):
        assert math.isfinite(value)
    horizon = uniform_optimal_horizon(mandate(risk_aversion=1e-8), 1e6)
    for value in (horizon.horizon, horizon.net_alpha, horizon.half_life, horizon.half_lives):
        assert math.isfinite(value)


def test_the_figures_printed_in_the_readme() -> None:
    """The console block in the README, pinned to the digits that are in print."""
    fixed = break_even_at_horizon(BASE, 1.0, uniform=True)
    assert f"{fixed.quantity:,.0f}" == "409,186"
    assert f"{break_even_continuum(BASE, horizon=1.0):,.0f}" == "409,091"
    assert f"{break_even_continuum(BASE, horizon=1.0) / fixed.quantity - 1.0:+.6f}" == "-0.000233"
    assert f"{break_even_at_participation(BASE, 0.02).quantity:,.0f}" == "3,501,751"
    assert f"{break_even_at_participation(BASE, 0.05).quantity:,.0f}" == "2,001,001"
    assert f"{alpha_floor(BASE, 0.10):.6f}" == "0.055000"
    assert f"{alpha_floor(BASE, 0.25):.6f}" == "0.130000"
