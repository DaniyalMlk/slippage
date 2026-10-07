"""VWAP tracking: one exact identity, one invariance, and a floor nothing can cross.

The arithmetic here is simple enough that almost everything can be checked
against something other than itself.

**One exact identity.** A schedule that matches the *realised* volume curve has
zero slippage against the VWAP path by path — not in expectation and not
approximately, because the sum ``sum_k (u_k - w_k) p_k`` is identically zero term
by term. A simulation confirms it at exactly zero rather than to a tolerance.

**One invariance that removes a parameter.** Where in a bucket its price is read
adds a constant to every entry of the price covariance, and both quantities a
tracking error is built from are orthogonal to a constant — the schedule
deviation sums to zero, and the volume covariance's rows sum to zero because the
shares sum to one. So the answer does not depend on the choice at all, which is
better than it depending on it weakly.

**One floor.** The irreducible term does not contain the schedule, so it is the
same for every schedule and no amount of risk aversion crosses it. The frontier's
two ends are therefore known in advance: the impact-minimising schedule at one
end and the expected volume profile at the other.

**An independent check.** The closed-form variance is compared with a Monte Carlo
that draws Dirichlet volume shares and a Brownian price walk and shares no code
with it, in units of that simulation's own standard error.
"""

from __future__ import annotations

import math
from datetime import time, timedelta
from itertools import pairwise

import numpy as np
import pytest

from slippage.exceptions import ValidationError
from slippage.impact import LinearImpact, PowerLawImpact
from slippage.tracking import (
    TrackingProblem,
    VolumeUncertainty,
    compare_objectives,
    fit_volume_covariance,
    price_covariance,
    tracking_frontier,
    tracking_moments,
    tracking_schedule,
)
from slippage.volume import VolumeProfile, twap_schedule, vwap_schedule

BUCKETS = 13
#: A U-shaped session: heavy at the open and the close, thin in the middle.
SHAPE = np.array([1.9, 1.3, 1.0, 0.85, 0.8, 0.75, 0.72, 0.75, 0.8, 0.9, 1.1, 1.4, 2.1])
MU = SHAPE / SHAPE.sum()
CONCENTRATION = 60.0
VOLATILITY = 0.004


def profile(concentration: float = CONCENTRATION) -> VolumeProfile:
    """A U-shaped profile whose dispersion is exactly a Dirichlet's.

    Built so that the fit has a known answer to recover, which is the only way to
    tell a working fit from one that returns a plausible number.
    """
    dispersion = np.sqrt(MU * (1.0 - MU) / (concentration + 1.0))
    return VolumeProfile(
        fractions=tuple(MU),
        bucket=timedelta(minutes=30),
        session_open=time(9, 30),
        dispersion=tuple(float(one) for one in dispersion),
        days=60,
    )


def order(midpoint: bool = True, volatility: float = VOLATILITY) -> TrackingProblem:
    return TrackingProblem(
        shares=1_000_000.0,
        profile=profile(),
        volatility=volatility,
        market_volume=20_000_000.0,
        bucket_hours=0.5,
        midpoint=midpoint,
    )


def walk(rng: np.random.Generator, paths: int, *, midpoint: bool) -> np.ndarray:
    """A Brownian price walk sampled at bucket midpoints or ends.

    Written out here rather than taken from the module, so that the closed form
    has something independent to be wrong against. The first gap is half a bucket
    under the midpoint convention and a whole one under the other, which is what
    makes ``Var(b_k)`` equal ``k - 1/2`` and ``k`` respectively.
    """
    steps = rng.standard_normal((paths, BUCKETS))
    gaps = np.ones(BUCKETS)
    gaps[0] = 0.5 if midpoint else 1.0
    return np.cumsum(steps * np.sqrt(gaps), axis=1) * VOLATILITY


def simulate(
    rng: np.random.Generator,
    weights: np.ndarray,
    paths: int,
    *,
    midpoint: bool = True,
    match_realised: bool = False,
) -> np.ndarray:
    """Relative slippage against the realised VWAP, path by path."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    shares = rng.dirichlet(CONCENTRATION * MU, size=paths)
    prices = walk(rng, paths, midpoint=midpoint)
    ours = shares if match_realised else np.broadcast_to(weights, shares.shape)
    return np.asarray(((ours - shares) * prices).sum(axis=1), dtype=np.float64)


# -- the identity --------------------------------------------------------------


def test_matching_the_realised_curve_is_exactly_zero_path_by_path() -> None:
    """Not to a tolerance. The sum cancels term by term, whatever the prices did.

    This is what makes a VWAP benchmark structurally different from an arrival
    price. No schedule can have zero slippage against a fixed arrival price on
    every path; against a benchmark built from the same prices we trade at, one
    can, and the only thing stopping it is that the realised curve is not known in
    advance.
    """
    rng = np.random.default_rng(11)
    slippage = simulate(rng, MU, 20_000, match_realised=True)
    assert np.max(np.abs(slippage)) == 0.0


def test_a_deterministic_volume_curve_has_no_irreducible_error() -> None:
    """The floor is entirely the volume forecast's error, so removing it removes the floor."""
    tightening = [
        tracking_moments(
            order(), MU, fit_volume_covariance(profile(), concentration=concentration)
        ).irreducible_bps
        for concentration in (10.0, 1e3, 1e6, 1e12)
    ]
    for looser, tighter in pairwise(tightening):
        assert tighter < looser
    assert tightening[-1] < 1e-4
    assert tightening[0] > 1.0


def test_zero_dispersion_is_refused_with_what_it_means() -> None:
    """Rather than returning an infinite concentration and a floor of zero.

    A measured dispersion of zero in every bucket is a statement about the data,
    and the right response is to say what it implies rather than to divide by it.
    """
    flat = VolumeProfile(
        fractions=tuple(MU),
        bucket=timedelta(minutes=30),
        session_open=time(9, 30),
        dispersion=(0.0,) * BUCKETS,
        days=60,
    )
    with pytest.raises(ValidationError, match="irreducible tracking error is"):
        fit_volume_covariance(flat)
    # and giving a concentration explicitly still works
    assert fit_volume_covariance(flat, concentration=50.0).concentration == 50.0


# -- the invariance -----------------------------------------------------------


@pytest.mark.parametrize("name", ["expected", "uniform", "front-loaded", "random"])
def test_where_in_a_bucket_the_price_is_read_changes_nothing(name: str) -> None:
    """To twelve significant figures, which is the point: there is no parameter here.

    The midpoint shift adds ``-sigma**2 / 2`` to every entry of ``C``, and both
    terms are orthogonal to a constant matrix — the schedule deviation sums to
    zero and the volume covariance's rows sum to zero. An earlier version of the
    module's docstring claimed the choice moved the *level* and only left the
    optimum alone. It moves neither.
    """
    rng = np.random.default_rng(3)
    weights = {
        "expected": MU,
        "uniform": np.ones(BUCKETS) / BUCKETS,
        "front-loaded": np.exp(-0.35 * np.arange(BUCKETS)),
        "random": rng.random(BUCKETS) + 0.1,
    }[name]
    uncertainty = fit_volume_covariance(profile())
    at_midpoints = tracking_moments(order(midpoint=True), weights, uncertainty)
    at_ends = tracking_moments(order(midpoint=False), weights, uncertainty)
    assert at_midpoints.schedule_variance == pytest.approx(
        at_ends.schedule_variance, rel=1e-12, abs=1e-20
    )
    assert at_midpoints.irreducible == pytest.approx(at_ends.irreducible, rel=1e-12)


def test_adding_any_constant_to_the_price_covariance_changes_nothing() -> None:
    """The general statement the midpoint convention is one instance of.

    Checked at 1000, which is six orders above the covariance's own entries, so a
    term that depended on it would be impossible to miss. The floor moves by
    1.2e-15 relative.
    """
    uncertainty = fit_volume_covariance(profile())
    base = price_covariance(BUCKETS, VOLATILITY)
    reference = float(np.sum(base * uncertainty.covariance))
    for constant in (1.0, -3.7, 1000.0):
        shifted = float(np.sum((base + constant) * uncertainty.covariance))
        assert shifted == pytest.approx(reference, rel=1e-12)


def test_the_volume_covariance_rows_sum_to_zero() -> None:
    """Which is the simplex constraint, and the reason the invariance above holds."""
    uncertainty = fit_volume_covariance(profile())
    assert uncertainty.rows_sum_to_zero < 1e-15
    assert float(np.sum(uncertainty.covariance)) == pytest.approx(0.0, abs=1e-15)
    # off-diagonals are negative: a heavy open means a lighter rest of the day
    off = uncertainty.covariance - np.diag(np.diag(uncertainty.covariance))
    assert np.all(off <= 0.0)
    assert np.all(np.diag(uncertainty.covariance) > 0.0)


def test_a_diagonal_covariance_overstates_the_floor_by_half_again() -> None:
    """Which is why the simplex structure is modelled rather than the dispersion alone.

    Using only the measured per-bucket dispersions and ignoring the correlations
    the constraint forces puts the floor **56.4% too high**, and does so at every
    concentration — both quadratic forms scale the same way, so the ratio is a
    property of the profile's shape rather than of how uncertain it is. A desk
    told its irreducible error was 12.6 basis points when it was 8.1 would accept
    schedules it should reject.
    """
    overstatements = []
    for concentration in (20.0, 60.0, 200.0):
        uncertainty = fit_volume_covariance(profile(), concentration=concentration)
        diagonal = VolumeUncertainty(
            covariance=np.diag(np.diag(uncertainty.covariance)),
            concentration=uncertainty.concentration,
            dispersion_error=0.0,
        )
        full = tracking_moments(order(), MU, uncertainty).irreducible_bps
        partial = tracking_moments(order(), MU, diagonal).irreducible_bps
        overstatements.append(partial / full - 1.0)
    for one in overstatements:
        assert one == pytest.approx(0.564, abs=0.01)
    assert max(overstatements) - min(overstatements) < 1e-9


# -- the fit ------------------------------------------------------------------


@pytest.mark.parametrize("concentration", [5.0, 60.0, 500.0])
def test_the_fit_recovers_a_concentration_it_was_built_from(concentration: float) -> None:
    """A fit that cannot recover a known answer is a number, not a measurement."""
    uncertainty = fit_volume_covariance(profile(concentration))
    assert uncertainty.concentration == pytest.approx(concentration, rel=1e-9)
    assert uncertainty.dispersion_error < 1e-12


def test_the_fit_reports_how_badly_one_parameter_fits_a_vector() -> None:
    """Honest rather than impressive: real dispersion is not exactly a Dirichlet's.

    Perturbing the dispersions away from the family the fit assumes leaves a
    residual, and the result carries it. Returning a concentration without it
    would let a profile that the model cannot describe look as though it had been
    described.
    """
    rng = np.random.default_rng(5)
    base = np.sqrt(MU * (1.0 - MU) / (CONCENTRATION + 1.0))
    bent = base * (1.0 + rng.uniform(-0.4, 0.4, BUCKETS))
    awkward = VolumeProfile(
        fractions=tuple(MU),
        bucket=timedelta(minutes=30),
        session_open=time(9, 30),
        dispersion=tuple(float(one) for one in bent),
        days=60,
    )
    uncertainty = fit_volume_covariance(awkward)
    assert uncertainty.dispersion_error > 0.1
    assert uncertainty.concentration > 0.0
    assert uncertainty.rows_sum_to_zero < 1e-15


def test_a_profile_with_no_dispersion_says_so() -> None:
    """A uniform profile was not estimated from anything, so there is nothing to fit."""
    bare = VolumeProfile.uniform(BUCKETS, timedelta(minutes=30), time(9, 30))
    with pytest.raises(ValidationError, match="carries no per-bucket dispersion"):
        fit_volume_covariance(bare)
    assert fit_volume_covariance(bare, concentration=40.0).concentration == 40.0


# -- against the simulation ---------------------------------------------------


@pytest.mark.parametrize("midpoint", [True, False])
@pytest.mark.parametrize("name", ["expected", "uniform", "front-loaded"])
def test_the_closed_form_variance_matches_a_simulation(name: str, midpoint: bool) -> None:
    """In units of the simulation's own standard error, not a chosen tolerance.

    The simulation draws Dirichlet shares and builds its own price walk, so it
    shares no code with the formula. It also found a defect in the first version
    of this test rather than in the module: a walk written as
    ``cumsum(eps) - eps / 2`` has ``Var(b_k) = k - 3/4`` rather than ``k - 1/2``,
    which showed up as the closed form sitting 10% high at z = -44. The formula
    was right and the test's walk was wrong.
    """
    rng = np.random.default_rng(29)
    weights = {
        "expected": MU,
        "uniform": np.ones(BUCKETS) / BUCKETS,
        "front-loaded": np.exp(-0.35 * np.arange(BUCKETS)),
    }[name]
    weights = weights / weights.sum()
    uncertainty = fit_volume_covariance(profile())
    closed = tracking_moments(order(midpoint=midpoint), weights, uncertainty)
    paths = 400_000
    slippage = simulate(rng, weights, paths, midpoint=midpoint)
    variance = float(slippage.var(ddof=1))
    # Standard error of a variance estimate from a near-normal sample.
    error = variance * math.sqrt(2.0 / paths)
    assert abs(variance - closed.variance) < 4.0 * error
    # and the mean is zero, because the walk has no drift
    assert abs(float(slippage.mean())) < 4.0 * math.sqrt(variance / paths)


# -- the minimiser ------------------------------------------------------------


@pytest.mark.parametrize("midpoint", [True, False])
def test_no_perturbation_of_the_expected_profile_lowers_the_variance(midpoint: bool) -> None:
    """The claim that the volume curve is optimal, tested rather than asserted.

    It is usually given as a rule of thumb and it is a theorem: the
    schedule-dependent variance is ``(u - mu)' C (u - mu)``, so ``mu`` is the exact
    minimiser for any positive-definite ``C``. Three hundred random perturbations
    on the simplex, none of which improves on it, and the schedule term at ``mu``
    is exactly zero rather than small.
    """
    rng = np.random.default_rng(13)
    uncertainty = fit_volume_covariance(profile())
    problem = order(midpoint=midpoint)
    base = tracking_moments(problem, MU, uncertainty)
    assert base.schedule_variance == 0.0
    assert base.variance == base.irreducible
    assert base.avoidable_share == 0.0
    for _ in range(300):
        perturbed = np.abs(MU + rng.normal(0.0, 0.01, BUCKETS))
        assert tracking_moments(problem, perturbed, uncertainty).variance >= base.variance


def test_the_pure_schedule_is_the_volume_curve_and_the_library_already_slices_it() -> None:
    """So the answer agrees with ``volume.vwap_schedule``, which was already there.

    What was missing was not the schedule but the statement that it is optimal and
    the number saying what it still cannot avoid.
    """
    uncertainty = fit_volume_covariance(profile())
    problem = order()
    solved = tracking_schedule(problem, uncertainty)
    sliced = np.asarray(vwap_schedule(problem.shares, problem.profile), dtype=np.float64)
    assert np.allclose(solved, sliced / sliced.sum(), rtol=1e-14)


def test_asking_for_half_a_trade_off_is_refused() -> None:
    uncertainty = fit_volume_covariance(profile())
    with pytest.raises(ValidationError, match="give both impact and risk_aversion"):
        tracking_schedule(order(), uncertainty, risk_aversion=10.0)
    with pytest.raises(ValidationError, match="give both impact and risk_aversion"):
        tracking_schedule(
            order(), uncertainty, impact=LinearImpact(gamma=1e-7, eta=1e-6, epsilon=0.0)
        )


# -- the frontier -------------------------------------------------------------


def test_the_frontier_runs_between_the_two_schedules_it_has_to() -> None:
    """Both ends are known before the solve, which is what makes them a check.

    With no weight on tracking error the schedule minimises a convex temporary
    cost, which an equal slice per bucket does regardless of where the volume is —
    recovered here as ``twap_schedule`` to a part in ten thousand. With a large
    weight it goes to the expected volume curve and the tracking error reaches its
    floor. Measured: impact rises 82.0 to 91.2 basis points as tracking error
    falls 11.64 to 8.05, the floor.
    """
    uncertainty = fit_volume_covariance(profile())
    problem = order()
    model = LinearImpact(gamma=2e-7, eta=2e-6, epsilon=0.01)
    points = tracking_frontier(problem, uncertainty, model, [0.0, 1e2, 1e3, 1e4, 1e6], price=50.0)
    patient = np.asarray(points[0].schedule)
    twap = np.asarray(twap_schedule(problem.shares, BUCKETS), dtype=np.float64)
    assert np.allclose(patient, twap / twap.sum(), atol=1e-4)
    eager = np.asarray(points[-1].schedule)
    assert np.allclose(eager, MU, atol=2e-3)
    assert points[-1].tracking_error_bps == pytest.approx(points[-1].irreducible_bps, rel=2e-3)
    # the trade-off is a trade-off: one rises as the other falls
    for earlier, later in pairwise(points):
        assert later.tracking_error_bps <= earlier.tracking_error_bps + 1e-9
        assert later.impact_bps >= earlier.impact_bps - 1e-9
    # and the floor is the same at every point, because it has no schedule in it
    assert len({round(one.irreducible_bps, 12) for one in points}) == 1


def test_the_frontier_cannot_be_pushed_below_the_floor() -> None:
    """However much risk aversion is applied. The floor is not a soft limit."""
    uncertainty = fit_volume_covariance(profile())
    problem = order()
    model = LinearImpact(gamma=2e-7, eta=2e-6, epsilon=0.01)
    points = tracking_frontier(problem, uncertainty, model, [1e6, 1e9, 1e12], price=50.0)
    for one in points:
        assert one.tracking_error_bps >= one.irreducible_bps - 1e-9


def test_a_power_law_impact_model_also_solves() -> None:
    """The solve is numerical rather than a closed form, so it is not linear-only."""
    uncertainty = fit_volume_covariance(profile())
    problem = order()
    model = PowerLawImpact(gamma=2e-7, eta=3e-5, beta=0.6, epsilon=0.01)
    weights = tracking_schedule(problem, uncertainty, impact=model, risk_aversion=1e3, price=50.0)
    assert float(weights.sum()) == pytest.approx(1.0, rel=1e-12)
    assert np.all(weights >= 0.0)
    moments = tracking_moments(problem, weights, uncertainty)
    at_curve = tracking_moments(problem, MU, uncertainty)
    assert moments.variance >= at_curve.variance


# -- the wrong objective ------------------------------------------------------


def test_an_arrival_price_schedule_tracks_vwap_badly_and_twap_does_not() -> None:
    """The number a desk needs, and it separates two things usually lumped together.

    A front-loaded schedule is the right answer to the arrival-price question and
    the wrong one here: 47.76 basis points of tracking error against the volume
    curve's 8.05, an excess of **4.93 times the irreducible floor**. Risk aversion
    against a fixed benchmark front-loads; against a benchmark built from the same
    prices it does the opposite, so the two objectives genuinely conflict.

    TWAP is the interesting case. It ignores the volume curve completely and is
    still only 0.45 floors worse — inside what a desk can measure. So "do not
    front-load" carries almost all of the benefit of VWAP tracking, and matching
    the curve exactly carries the rest.
    """
    uncertainty = fit_volume_covariance(profile())
    problem = order()
    heavy = np.exp(-0.35 * np.arange(BUCKETS))
    comparison = compare_objectives(problem, uncertainty, heavy)
    assert comparison.vwap_schedule_error_bps == pytest.approx(8.05, abs=0.05)
    assert comparison.other_schedule_error_bps == pytest.approx(47.8, abs=0.5)
    assert comparison.excess_over_floor == pytest.approx(4.93, abs=0.1)

    flat = compare_objectives(problem, uncertainty, np.ones(BUCKETS))
    assert flat.excess_over_floor == pytest.approx(0.45, abs=0.05)
    assert flat.excess_over_floor < comparison.excess_over_floor / 5.0


def test_the_volume_curve_compares_against_itself_at_zero_excess() -> None:
    uncertainty = fit_volume_covariance(profile())
    comparison = compare_objectives(order(), uncertainty, MU)
    assert comparison.excess_bps == pytest.approx(0.0, abs=1e-12)
    assert comparison.excess_over_floor == pytest.approx(0.0, abs=1e-12)


def test_the_excess_over_the_floor_is_infinite_when_there_is_no_floor() -> None:
    """Rather than a division by zero, and reached only at an exactly zero floor.

    A very large concentration is not enough: at 1e18 the floor is 6.3e-08 basis
    points rather than zero, so the ratio comes out at 1.3e8 — large, finite, and
    not what the branch is for. The branch exists for a covariance that is
    identically zero, which is a deterministic volume curve rather than a nearly
    deterministic one, so that is what is passed.
    """
    nearly = fit_volume_covariance(profile(), concentration=1e18)
    almost = compare_objectives(order(), nearly, np.ones(BUCKETS))
    assert almost.irreducible_bps == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(almost.excess_over_floor)
    assert almost.excess_over_floor > 1e6

    exactly = VolumeUncertainty(
        covariance=np.zeros((BUCKETS, BUCKETS)),
        concentration=1.0,
        dispersion_error=0.0,
    )
    comparison = compare_objectives(order(), exactly, np.ones(BUCKETS))
    assert comparison.irreducible_bps == 0.0
    assert math.isinf(comparison.excess_over_floor)
    # and matching the curve is then exactly on the benchmark
    assert tracking_moments(order(), MU, exactly).variance == 0.0


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "shares,volatility,volume,hours",
    [
        (0.0, 0.004, 1e7, 0.5),
        (-1.0, 0.004, 1e7, 0.5),
        (1e6, -0.1, 1e7, 0.5),
        (1e6, math.nan, 1e7, 0.5),
        (1e6, 0.004, -1.0, 0.5),
        (1e6, 0.004, 1e7, 0.0),
    ],
)
def test_the_problem_refuses_what_it_cannot_describe(
    shares: float, volatility: float, volume: float, hours: float
) -> None:
    with pytest.raises(ValidationError):
        TrackingProblem(
            shares=shares,
            profile=profile(),
            volatility=volatility,
            market_volume=volume,
            bucket_hours=hours,
        )


def test_a_schedule_of_the_wrong_length_is_refused_with_both_lengths() -> None:
    uncertainty = fit_volume_covariance(profile())
    with pytest.raises(ValidationError, match="has 5 buckets and the profile has 13"):
        tracking_moments(order(), np.ones(5), uncertainty)


def test_a_covariance_of_the_wrong_length_is_refused() -> None:
    small = fit_volume_covariance(
        VolumeProfile(
            fractions=(0.5, 0.5),
            bucket=timedelta(minutes=30),
            session_open=time(9, 30),
            dispersion=(0.05, 0.05),
            days=10,
        )
    )
    with pytest.raises(ValidationError, match="covers 2 buckets"):
        tracking_moments(order(), MU, small)


@pytest.mark.parametrize("bad", [[0.0, 0.0, 0.0], [-1.0, 2.0], [math.nan, 1.0], []])
def test_a_schedule_that_is_not_a_schedule_is_refused(bad: list[float]) -> None:
    uncertainty = fit_volume_covariance(profile())
    with pytest.raises(ValidationError):
        tracking_moments(order(), bad, uncertainty)


@pytest.mark.parametrize("buckets,volatility", [(0, 0.1), (-1, 0.1), (5, -0.1)])
def test_the_price_covariance_refuses_a_shape_it_cannot_build(
    buckets: int, volatility: float
) -> None:
    with pytest.raises(ValidationError):
        price_covariance(buckets, volatility)


def test_a_bad_concentration_or_price_is_refused() -> None:
    with pytest.raises(ValidationError, match="concentration must be positive"):
        fit_volume_covariance(profile(), concentration=0.0)
    uncertainty = fit_volume_covariance(profile())
    model = LinearImpact(gamma=1e-7, eta=1e-6, epsilon=0.0)
    with pytest.raises(ValidationError, match="price must be positive"):
        tracking_schedule(order(), uncertainty, impact=model, risk_aversion=1.0, price=0.0)
    with pytest.raises(ValidationError, match="risk aversion must be"):
        tracking_schedule(order(), uncertainty, impact=model, risk_aversion=-1.0, price=50.0)


def test_every_reported_number_is_finite() -> None:
    uncertainty = fit_volume_covariance(profile())
    problem = order()
    model = LinearImpact(gamma=2e-7, eta=2e-6, epsilon=0.01)
    for point in tracking_frontier(problem, uncertainty, model, [0.0, 1e3, 1e6], price=50.0):
        assert math.isfinite(point.impact_bps)
        assert math.isfinite(point.tracking_error_bps)
        assert math.isfinite(point.irreducible_bps)
        assert all(math.isfinite(one) for one in point.schedule)
        assert point.tracking_error_bps >= 0.0
