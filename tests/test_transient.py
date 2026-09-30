"""Transient impact under a decay kernel.

The quadratic form is checked against walking the price path trade by trade,
which is a separate computation of the same quantity rather than the same
algebra twice. The closed-form schedule is checked against a numerical optimiser
on the same problem. And the positive-definiteness guard is checked with kernels
that break it, including one that is strictly decreasing, because "decreasing"
is the property everybody assumes is sufficient and it is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise

import numpy as np
import pytest
from scipy.optimize import minimize

from slippage.exceptions import ValidationError
from slippage.reversion import Decay
from slippage.transient import (
    DEFINITENESS_TOLERANCE,
    ExponentialDecay,
    PowerLawDecay,
    cost_matrix,
    impact_path,
    manipulation_round_trip,
    optimal_transient_schedule,
    residual_impact,
    transient_cost,
)

PERMANENT = 1e-6
TRANSIENT = 2e-5
QUANTITY = 100_000.0


def kernel(resilience: float = 0.25, *, permanent: float = PERMANENT) -> ExponentialDecay:
    return ExponentialDecay(permanent=permanent, transient=TRANSIENT, resilience=resilience)


@dataclass(frozen=True)
class Shoulder:
    """A strictly decreasing kernel that is not positive definite.

    It falls almost flat for a while and then drops off a shoulder. Every
    sensible-sounding property holds -- non-negative, decreasing, bounded, with a
    finite permanent floor of zero -- and it still pays a trader to round trip.
    Complete monotonicity is what is actually required, and this is not that.
    """

    width: float = 1.5
    slope: float = 0.02
    floor: float = 0.20
    tail: float = 0.2

    @property
    def permanent(self) -> float:
        return 0.0

    def value(self, lag: float) -> float:
        if lag < self.width:
            return 1.0 - self.slope * lag
        return self.floor * math.exp(-self.tail * (lag - self.width))


@dataclass(frozen=True)
class BuildingImpact:
    """A kernel whose impact grows before it decays.

    Plausible as a story -- the market notices the trade a little after it
    happens -- and inadmissible, because impact that arrives late can be
    front-run by the trader who caused it.
    """

    @property
    def permanent(self) -> float:
        return 0.0

    def value(self, lag: float) -> float:
        return math.exp(-0.25 * lag) * (1.0 - 0.55 * math.exp(-lag))


# -- the kernels themselves --------------------------------------------------


def test_the_exponential_kernel_decays_from_its_instantaneous_value_to_its_floor() -> None:
    one = kernel(resilience=0.25)
    assert one.value(0.0) == pytest.approx(PERMANENT + TRANSIENT)
    assert one.value(one.half_life) == pytest.approx(PERMANENT + TRANSIENT / 2.0)
    assert one.value(1e6) == pytest.approx(PERMANENT)
    assert one.half_life == pytest.approx(math.log(2.0) / 0.25)


def test_the_power_law_kernel_decays_more_slowly_than_any_exponential() -> None:
    """Which is the reason to have it: the memory of a trade is never gone.

    Matched so that both halve at the same lag, the power law is above the
    exponential at every longer lag and the *ratio* between them grows without
    bound. The absolute gap does not: both are heading to zero, so the gap peaks
    and comes back -- 2.31e-6 at twice the crossing, 3.41e-6 at eight times,
    1.36e-6 at forty. A first version of this test asserted the gap was
    decreasing throughout and was measuring the wrong quantity.
    """
    power = PowerLawDecay(permanent=0.0, transient=TRANSIENT, exponent=0.6, scale=1.0)
    matched = ExponentialDecay(
        permanent=0.0,
        transient=TRANSIENT,
        resilience=math.log(2.0) / (2.0 ** (1.0 / 0.6) - 1.0),
    )
    crossing = 2.0 ** (1.0 / 0.6) - 1.0
    assert power.value(crossing) == pytest.approx(TRANSIENT / 2.0, rel=1e-12)
    assert matched.value(crossing) == pytest.approx(TRANSIENT / 2.0, rel=1e-12)
    lags = [crossing * multiple for multiple in (2, 8, 40, 200)]
    assert all(power.value(lag) > matched.value(lag) for lag in lags)
    ratios = [power.value(lag) / matched.value(lag) for lag in lags]
    assert ratios == sorted(ratios)
    assert ratios[-1] > 1e30
    # The gap itself peaks rather than growing, because both are going to zero.
    gaps = [power.value(lag) - matched.value(lag) for lag in lags[:3]]
    assert gaps[1] > gaps[0] > gaps[2]
    assert power.value(1e4) > 0.0
    assert matched.value(1e4) == pytest.approx(0.0, abs=1e-300)


def test_a_kernel_that_says_trading_is_free_is_refused() -> None:
    with pytest.raises(ValidationError, match="not a model of anything"):
        ExponentialDecay(permanent=0.0, transient=0.0, resilience=1.0)
    with pytest.raises(ValidationError, match="not a model of anything"):
        PowerLawDecay(permanent=0.0, transient=0.0, exponent=0.6)
    with pytest.raises(ValidationError, match="resilience must be a positive"):
        ExponentialDecay(permanent=0.0, transient=1.0, resilience=0.0)
    with pytest.raises(ValidationError, match="lag must be non-negative"):
        kernel().value(-1.0)
    with pytest.raises(ValidationError, match="lag must be non-negative"):
        PowerLawDecay(permanent=0.0, transient=1.0, exponent=0.6).value(-1.0)


# -- the cost --------------------------------------------------------------


def test_the_cost_matrix_is_symmetric_and_holds_the_kernel_at_each_lag() -> None:
    one = kernel()
    matrix = cost_matrix(one, 5, 2.0)
    assert matrix.shape == (5, 5)
    assert np.allclose(matrix, matrix.T, atol=0.0)
    assert matrix[0, 0] == pytest.approx(one.value(0.0))
    assert matrix[0, 3] == pytest.approx(one.value(6.0))
    assert matrix[4, 1] == pytest.approx(one.value(6.0))


@pytest.mark.parametrize("resilience", [0.05, 0.25, 1.0, 4.0])
def test_the_quadratic_form_is_the_price_path_walked_trade_by_trade(
    resilience: float,
) -> None:
    """Two computations of the same quantity, not one written twice.

    The quadratic form is a matrix product. The path version loops over slices,
    accumulating what earlier ones left in the price and charging each slice half
    of its own instantaneous impact. Agreement means the rearrangement from one
    to the other is right, which is the only thing that could be wrong here.
    """
    one = kernel(resilience)
    trades = [12_000.0, 9_000.0, 20_000.0, 5_000.0, 30_000.0, 24_000.0]
    path = impact_path(one, trades, 1.0)
    by_hand = math.fsum(
        size * (already + 0.5 * size * one.value(0.0))
        for size, already in zip(trades, path, strict=True)
    )
    assert transient_cost(one, trades, 1.0) == pytest.approx(by_hand, rel=1e-12)
    assert path[0] == 0.0
    assert len(path) == len(trades)


def test_a_permanent_only_kernel_costs_the_same_whatever_the_schedule() -> None:
    """The Almgren-Chriss identity, recovered from the quadratic form.

    With no decay the matrix is ``gamma`` everywhere, so the cost is
    ``gamma X^2 / 2`` and every schedule of the same total is identical. This is
    also the ``resilience -> 0`` limit, which is why the scheduling gain has to
    vanish there.
    """
    flat = ExponentialDecay(permanent=PERMANENT, transient=0.0, resilience=1.0)
    for trades in (
        [QUANTITY / 6] * 6,
        [QUANTITY, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, QUANTITY],
        [10_000.0, 40_000.0, 5_000.0, 25_000.0, 15_000.0, 5_000.0],
    ):
        assert transient_cost(flat, trades, 1.0) == pytest.approx(
            0.5 * PERMANENT * QUANTITY**2, rel=1e-12
        )


def test_an_infinitely_resilient_kernel_charges_for_each_slice_alone() -> None:
    """The other limit, and where the rate models live.

    Decay fast enough and the matrix is diagonal, so the cost is
    ``eta sum n_k^2 / 2`` -- quadratic in each slice separately, which is the
    shape ``LinearImpact`` charges in, and minimised by trading at a constant
    rate.
    """
    fast = ExponentialDecay(permanent=0.0, transient=TRANSIENT, resilience=400.0)
    trades = [10_000.0, 40_000.0, 5_000.0, 45_000.0]
    assert transient_cost(fast, trades, 1.0) == pytest.approx(
        0.5 * TRANSIENT * sum(one**2 for one in trades), rel=1e-6
    )
    best = optimal_transient_schedule(fast, QUANTITY, 8, 1.0)
    assert best.trades == pytest.approx([QUANTITY / 8] * 8, rel=1e-4)
    assert best.saving == pytest.approx(0.0, abs=1e-6)


def test_the_cost_refuses_inputs_that_are_not_a_schedule() -> None:
    one = kernel()
    with pytest.raises(ValidationError, match="non-empty one-dimensional"):
        transient_cost(one, [], 1.0)
    with pytest.raises(ValidationError, match="every trade must be finite"):
        transient_cost(one, [1.0, math.nan], 1.0)
    with pytest.raises(ValidationError, match="tau must be a positive"):
        transient_cost(one, [1.0, 2.0], 0.0)
    with pytest.raises(ValidationError, match="periods must be at least 1"):
        cost_matrix(one, 0, 1.0)


# -- the mark-out curve the model predicts -----------------------------------


def test_the_residual_impact_is_an_exact_exponential_whatever_the_schedule() -> None:
    """Which is why a mark-out recovers the resilience and not the amplitude.

    The horizon appears in every term as the same factor
    ``exp(-resilience * h)``, so it comes outside the sum. The decay *rate* is
    therefore a property of the kernel alone; the level it decays from depends on
    how the order was worked.
    """
    one = kernel(resilience=0.25)
    for trades in (
        [QUANTITY / 8] * 8,
        [QUANTITY],
        [40_000.0, 5_000.0, 5_000.0, 50_000.0],
    ):
        floor = one.permanent * sum(trades)
        horizons = [0.0, 0.5, 1.0, 2.0, 5.0, 11.0]
        curve = residual_impact(one, trades, 1.0, horizons)
        amplitude = curve[0] - floor
        assert amplitude > 0.0
        for value, horizon in zip(curve, horizons, strict=True):
            assert value - floor == pytest.approx(
                amplitude * math.exp(-one.resilience * horizon), rel=1e-12
            )


def test_a_worked_order_understates_the_transient_amplitude_by_a_known_factor() -> None:
    """The measurement behind the caveat on ``from_measured_decay``.

    For a uniform schedule of ``N`` slices over a horizon ``T``, the amplitude a
    mark-out from completion sees is the kernel's transient impact times
    ``(1/N)(1 - e^-rhoT)/(1 - e^-rhoT/N)``, because every slice but the last has
    been decaying already. Over two half-lives that is 0.589 at eight slices,
    falling to 0.541 as the trading becomes continuous -- so reading a kernel off
    a worked order's mark-out understates its transient part by around 40%.
    """
    horizon = 8.0
    resilience = 2.0 * math.log(2.0) / horizon  # two half-lives over the horizon
    one = ExponentialDecay(permanent=PERMANENT, transient=TRANSIENT, resilience=resilience)

    def predicted(slices: int) -> float:
        rho_t = resilience * horizon
        return (1.0 / slices) * (1.0 - math.exp(-rho_t)) / (1.0 - math.exp(-rho_t / slices))

    for slices, expected in ((4, 0.640165), (8, 0.589239), (16, 0.564787), (64, 0.546891)):
        tau = horizon / slices
        trades = [QUANTITY / slices] * slices
        at_completion = residual_impact(one, trades, tau, [0.0])[0]
        amplitude = at_completion - one.permanent * QUANTITY
        assert amplitude / (TRANSIENT * QUANTITY) == pytest.approx(predicted(slices), rel=1e-12)
        assert predicted(slices) == pytest.approx(expected, abs=5e-6)
    # The continuous limit, for the number quoted in the docstring.
    rho_t = resilience * horizon
    assert (1.0 - math.exp(-rho_t)) / rho_t == pytest.approx(0.541011, abs=5e-6)
    # A single block loses nothing, which is the case the inversion is exact for.
    assert predicted(1) == pytest.approx(1.0, rel=1e-12)


def test_a_kernel_can_be_read_back_off_a_measured_decay() -> None:
    one = ExponentialDecay.from_measured_decay(
        Decay(
            half_life=timedelta(minutes=11),
            asymptote_bps=4.0,
            amplitude_bps=6.0,
            r_squared=0.97,
            points=6,
        ),
        quantity=50_000.0,
        price=40.0,
        unit=timedelta(minutes=1),
    )
    assert one.half_life == pytest.approx(11.0, rel=1e-12)
    assert one.resilience == pytest.approx(math.log(2.0) / 11.0, rel=1e-12)
    # 6bp of a 40 price over 50,000 shares.
    assert one.transient == pytest.approx(6.0 / 1e4 * 40.0 / 50_000.0, rel=1e-12)
    assert one.permanent == pytest.approx(4.0 / 1e4 * 40.0 / 50_000.0, rel=1e-12)
    # A block of that size then shows the impact the curve was measured at.
    assert residual_impact(one, [50_000.0], 1.0, [0.0])[0] == pytest.approx(
        10.0 / 1e4 * 40.0, rel=1e-12
    )


def test_reading_back_a_decay_refuses_what_it_cannot_invert() -> None:
    fine = Decay(
        half_life=timedelta(minutes=11),
        asymptote_bps=4.0,
        amplitude_bps=6.0,
        r_squared=0.9,
        points=5,
    )
    with pytest.raises(ValidationError, match="quantity must be a positive"):
        ExponentialDecay.from_measured_decay(
            fine, quantity=0.0, price=40.0, unit=timedelta(minutes=1)
        )
    with pytest.raises(ValidationError, match="unit must be a positive duration"):
        ExponentialDecay.from_measured_decay(fine, quantity=1.0, price=40.0, unit=timedelta(0))
    # A negative asymptote is a measurement, not a kernel: impact that comes back
    # more than it went in. It floors at zero rather than raising, because the
    # half-life beside it is still usable.
    overshoot = Decay(
        half_life=timedelta(minutes=5),
        asymptote_bps=-1.5,
        amplitude_bps=8.0,
        r_squared=0.8,
        points=5,
    )
    read = ExponentialDecay.from_measured_decay(
        overshoot, quantity=1_000.0, price=10.0, unit=timedelta(minutes=1)
    )
    assert read.permanent == 0.0
    assert read.transient > 0.0


def test_a_horizon_before_completion_is_refused() -> None:
    with pytest.raises(ValidationError, match="before the order finished"):
        residual_impact(kernel(), [1.0, 2.0], 1.0, [-1.0])


# -- no free lunch ----------------------------------------------------------


@pytest.mark.parametrize(
    "admissible",
    [
        ExponentialDecay(permanent=0.0, transient=TRANSIENT, resilience=0.3),
        ExponentialDecay(permanent=PERMANENT, transient=TRANSIENT, resilience=0.3),
        ExponentialDecay(permanent=PERMANENT, transient=0.0, resilience=1.0),
        PowerLawDecay(permanent=0.0, transient=TRANSIENT, exponent=0.6),
        PowerLawDecay(permanent=PERMANENT, transient=TRANSIENT, exponent=1.4, scale=3.0),
    ],
)
def test_an_admissible_kernel_admits_no_profitable_round_trip(
    admissible: ExponentialDecay | PowerLawDecay,
) -> None:
    """Completely monotone kernels are mixtures of exponentials, hence positive
    definite, hence safe. Checked at several grids because the matrix, and so
    the eigenvalue that matters, depends on the spacing.
    """
    for periods in (2, 5, 12, 30):
        for tau in (0.1, 1.0, 7.0):
            assert manipulation_round_trip(admissible, periods, tau) is None


def test_a_kernel_whose_impact_builds_pays_you_to_round_trip() -> None:
    """The trip is returned rather than a boolean, because the numbers are the
    argument. Buy and sell in that pattern and the impact cost is negative.
    """
    trip = manipulation_round_trip(BuildingImpact(), 10, 1.0)
    assert trip is not None
    assert trip.cost < 0.0
    assert sum(trip.trades) == pytest.approx(0.0, abs=1e-12)
    assert max(abs(one) for one in trip.trades) == pytest.approx(1.0)
    # And the cost really is what running that trip costs, not a residue of the
    # eigen-decomposition it was found with.
    assert transient_cost(BuildingImpact(), trip.trades, 1.0) == pytest.approx(trip.cost, rel=1e-10)
    # Scaling the trip scales its profit quadratically, so there is no size at
    # which it stops being free money.
    bigger = [10.0 * one for one in trip.trades]
    assert transient_cost(BuildingImpact(), bigger, 1.0) == pytest.approx(
        100.0 * trip.cost, rel=1e-10
    )


def test_a_strictly_decreasing_kernel_is_not_enough() -> None:
    """The property everybody assumes is sufficient, and is not.

    The shoulder kernel is non-negative, bounded, and strictly decreasing at
    every lag -- checked here rather than asserted -- and it still admits a round
    trip costing -2.35 over twelve slices whose largest is one share, in a kernel
    whose instantaneous impact is one. The figure is at the normalisation
    ``manipulation_round_trip`` reports at, largest slice one; the same trip
    scaled to unit norm costs -0.358, and quoting one number for the other is how
    this assertion was wrong first time.
    """
    shoulder = Shoulder()
    lags = [index * 0.05 for index in range(400)]
    values = [shoulder.value(lag) for lag in lags]
    assert all(later < earlier for earlier, later in pairwise(values))
    assert all(one >= 0.0 for one in values)

    trip = manipulation_round_trip(shoulder, 12, 1.0)
    assert trip is not None
    assert trip.cost == pytest.approx(-2.355, abs=0.005)
    unit = np.asarray(trip.trades) / float(np.linalg.norm(trip.trades))
    assert transient_cost(shoulder, unit.tolist(), 1.0) == pytest.approx(-0.358, abs=0.002)
    assert sum(trip.trades) == pytest.approx(0.0, abs=1e-12)


def test_the_definiteness_threshold_is_relative_and_has_to_be() -> None:
    """A test against zero would reject a perfectly good power-law kernel.

    The cost matrix restricted to zero-sum directions is singular by
    construction, so its smallest eigenvalue is zero up to rounding -- and on
    this kernel that rounding comes out negative. The raw number is checked
    here, so that a future change to an absolute threshold fails rather than
    silently reporting a free lunch of 1e-16.
    """
    power = PowerLawDecay(permanent=0.0, transient=TRANSIENT, exponent=0.6)
    matrix = cost_matrix(power, 12, 1.0)
    projector = np.eye(12) - np.ones((12, 12)) / 12
    eigenvalues = np.linalg.eigvalsh(projector @ matrix @ projector)
    scale = float(np.max(np.abs(np.linalg.eigvalsh(matrix))))
    assert eigenvalues[0] < 0.0
    assert abs(eigenvalues[0]) < DEFINITENESS_TOLERANCE * scale
    assert manipulation_round_trip(power, 12, 1.0) is None


def test_a_single_slice_cannot_round_trip() -> None:
    assert manipulation_round_trip(BuildingImpact(), 1, 1.0) is None


def test_an_inadmissible_kernel_is_refused_a_schedule_with_the_trip_named() -> None:
    with pytest.raises(ValidationError, match="admits a round trip"):
        optimal_transient_schedule(BuildingImpact(), QUANTITY, 10, 1.0)


# -- the optimal schedule ---------------------------------------------------


@pytest.mark.parametrize("resilience", [0.05, 0.2, 1.0, 3.0])
def test_the_closed_form_schedule_matches_a_numerical_optimiser(
    resilience: float,
) -> None:
    """And beats it where the optimiser has not converged.

    SLSQP on the same quadratic, from a uniform start. At low resilience the
    closed form is 0.27% *cheaper*, which is the optimiser stopping early rather
    than a disagreement about the answer -- so the assertion is one-sided.
    """
    one = kernel(resilience)
    periods = 8
    best = optimal_transient_schedule(one, QUANTITY, periods, 1.0)
    matrix = cost_matrix(one, periods, 1.0)

    def cost(weights: np.typing.NDArray[np.float64]) -> float:
        return float(0.5 * weights @ matrix @ weights)

    start = np.full(periods, QUANTITY / periods)
    found = minimize(
        cost,
        start,
        constraints=[{"type": "eq", "fun": lambda z: float(np.sum(z)) - QUANTITY}],
        method="SLSQP",
        options={"maxiter": 900, "ftol": 1e-16},
    )
    assert found.success
    assert best.cost <= found.fun * (1.0 + 1e-9)
    assert best.cost == pytest.approx(found.fun, rel=5e-3)
    assert sum(best.trades) == pytest.approx(QUANTITY, rel=1e-12)


def test_the_schedule_is_a_block_a_constant_rate_and_a_block() -> None:
    """Obizhaeva and Wang's shape, and nothing here imposes it.

    The solve is a linear system. That its answer has two equal end blocks and a
    flat middle is a property of the exponential kernel, and it is the most
    convincing form the result can take -- an optimiser tuned until it produced
    the expected picture would not be evidence of anything.
    """
    best = optimal_transient_schedule(kernel(0.25), QUANTITY, 8, 1.0)
    trades = best.trades
    assert trades[0] == pytest.approx(trades[-1], rel=1e-9)
    middle = trades[1:-1]
    assert middle == pytest.approx([middle[0]] * len(middle), rel=1e-9)
    assert trades[0] > 4.0 * middle[0]
    assert best.front_load == pytest.approx(trades[0] / (QUANTITY / 8), rel=1e-12)
    assert best.front_load > 2.0


def test_the_saving_over_a_constant_rate_peaks_at_an_intermediate_resilience() -> None:
    """Both limits give nothing, so the gain is not monotone.

    Infinitely resilient, the matrix is diagonal and the uniform schedule is
    already optimal. Not resilient at all, the matrix is constant and every
    schedule ties. The peak is in between, and on a pure-transient kernel over
    eight slices it is worth 7.04%.
    """
    savings = {
        resilience: optimal_transient_schedule(
            ExponentialDecay(permanent=0.0, transient=TRANSIENT, resilience=resilience),
            QUANTITY,
            8,
            1.0,
        ).saving
        for resilience in (1e-3, 0.05, 0.25, 1.0, 5.0, 200.0)
    }
    assert savings[1e-3] < 0.002
    assert savings[200.0] < 1e-6
    assert savings[0.25] == pytest.approx(0.0704, abs=0.0005)
    assert savings[0.25] > savings[0.05] > savings[1e-3]
    assert savings[0.25] > savings[1.0] > savings[5.0]


def test_the_peak_depends_on_resilience_times_horizon_and_not_on_the_spacing() -> None:
    """A dimensionless statement, and the check that it is one.

    Eight slices over a horizon of eight, and eight over thirty, give identical
    savings once ``resilience * horizon`` matches, so the spacing is not a second
    parameter.

    The peak itself drifts with how finely the horizon is cut, and in both
    coordinates: from 3.57% at ``rho T = 1.62`` on four slices to 11.46% at
    ``rho T = 2.57`` on sixty-four. The solution wants two instantaneous blocks,
    and a finer grid approximates them better. So there is no single peak to
    quote without saying how many slices it is over -- a first version of this
    test read every grid's saving at *one* grid's peak and called the result the
    peak, which understated four slices by 0.08 of a point.
    """

    def saving(periods: int, horizon: float, rho_t: float) -> float:
        return optimal_transient_schedule(
            ExponentialDecay(permanent=0.0, transient=TRANSIENT, resilience=rho_t / horizon),
            QUANTITY,
            periods,
            horizon / periods,
        ).saving

    for periods in (4, 8, 16):
        assert saving(periods, 8.0, 2.0) == pytest.approx(saving(periods, 30.0, 2.0), rel=1e-9)

    # Each grid at its own peak, found by a search rather than assumed.
    own_peak: dict[int, tuple[float, float]] = {}
    for periods in (4, 8, 16, 32, 64):
        best = max(
            ((saving(periods, 8.0, rho_t), rho_t) for rho_t in np.linspace(1.0, 4.0, 121)),
        )
        own_peak[periods] = best
    assert own_peak[4][0] == pytest.approx(0.0357, abs=0.0005)
    assert own_peak[8][0] == pytest.approx(0.0704, abs=0.0005)
    assert own_peak[64][0] == pytest.approx(0.1146, abs=0.0010)
    assert [one for one, _ in own_peak.values()] == sorted(one for one, _ in own_peak.values())
    # And the peak moves to a faster decay as the grid gets finer.
    assert own_peak[4][1] < own_peak[8][1] < own_peak[64][1]

    # At one common decay rate the ordering in the number of slices is the same,
    # which is worth separating from the statement about the peaks.
    common = [saving(periods, 8.0, 2.0049) for periods in (4, 8, 16, 32, 64)]
    assert common == sorted(common)
    assert common[0] == pytest.approx(0.0349, abs=0.0005)


def test_a_single_slice_has_nothing_to_schedule() -> None:
    only = optimal_transient_schedule(kernel(), QUANTITY, 1, 1.0)
    assert only.trades == (QUANTITY,)
    assert only.cost == pytest.approx(0.5 * (PERMANENT + TRANSIENT) * QUANTITY**2)
    assert only.saving == pytest.approx(0.0, abs=1e-15)
    assert only.front_load == pytest.approx(1.0)


def test_the_schedule_refuses_a_quantity_or_a_grid_it_cannot_use() -> None:
    with pytest.raises(ValidationError, match="quantity must be a positive"):
        optimal_transient_schedule(kernel(), 0.0, 4, 1.0)
    with pytest.raises(ValidationError, match="periods must be at least 1"):
        optimal_transient_schedule(kernel(), QUANTITY, 0, 1.0)
    with pytest.raises(ValidationError, match="tau must be a positive"):
        optimal_transient_schedule(kernel(), QUANTITY, 4, -1.0)


def test_the_permanent_floor_does_not_change_the_schedule() -> None:
    """Because linear permanent impact costs the same however it is worked.

    It shifts the cost by ``gamma X^2 / 2`` and leaves the argmin alone, which
    is the Almgren-Chriss result arriving here through a different door. It does
    shrink the reported saving, since the fixed part is in both numerator and
    denominator.
    """
    without = optimal_transient_schedule(
        ExponentialDecay(permanent=0.0, transient=TRANSIENT, resilience=0.25),
        QUANTITY,
        8,
        1.0,
    )
    with_floor = optimal_transient_schedule(kernel(0.25), QUANTITY, 8, 1.0)
    assert with_floor.trades == pytest.approx(without.trades, rel=1e-9)
    assert with_floor.cost - without.cost == pytest.approx(0.5 * PERMANENT * QUANTITY**2, rel=1e-9)
    assert with_floor.saving < without.saving
