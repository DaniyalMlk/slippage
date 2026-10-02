"""Adapting a schedule to the liquidity it finds.

Two kinds of test here, and the first kind is what makes the second worth
reading.

The pinning test is the degenerate one. A chain with a single regime is an
Almgren-Chriss problem, so the backward induction has to reproduce
``optimal_trajectory`` exactly — holdings and objective both. If it does not,
every gain measured against a static schedule is a gain over the wrong thing,
and nothing downstream would show it. There are three more degenerate cases
with answers known in advance: a frozen chain, a strictly alternating chain and
a problem with fewer than three periods all have to give a gain of exactly
nothing, each for a different reason.

The measuring tests pin numbers that were guessed wrong first. The gain does not
peak at the middle of the persistence range, it peaks at 0.182; it is largest at
*zero* risk aversion rather than at high; it comes almost entirely from
liquidity rather than volatility; and the policy's first period trades less than
the static schedule's rather than more. Each of those is asserted here so that a
change which moves them fails rather than quietly rewriting the documentation.

The independent check is the simulation. The recursion computes a value; the
simulation draws regime paths, applies the policy's own fractions, and adds up
what those trades actually cost. The two share no arithmetic, so an error in the
quadratic algebra shows up as a disagreement rather than as a consistent wrong
answer.
"""

from __future__ import annotations

import math

import pytest

from slippage.adaptive import (
    AdaptiveProblem,
    LiquidityRegime,
    RegimeChain,
    adaptivity_gain,
    simulate_policy,
    solve_adaptive,
    static_schedule,
)
from slippage.exceptions import ValidationError
from slippage.execution import optimal_trajectory
from slippage.impact import LinearImpact

QUANTITY = 1.0e6
HORIZON = 1.0
PERIODS = 20
RISK_AVERSION = 2.0e-6

LIQUID = LiquidityRegime(
    impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6), volatility=0.3, label="liquid"
)
ILLIQUID = LiquidityRegime(
    impact=LinearImpact(gamma=2.5e-7, eta=1.25e-5), volatility=0.3, label="illiquid"
)


def two_regime(persistence: float, *, ratio: float = 5.0) -> RegimeChain:
    """Liquid and illiquid, symmetric, with the illiquid impact ``ratio`` times."""
    illiquid = LiquidityRegime(
        impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6 * ratio),
        volatility=0.3,
        label="illiquid",
    )
    leave = 1.0 - persistence
    return RegimeChain(
        regimes=(LIQUID, illiquid),
        transitions=((persistence, leave), (leave, persistence)),
    )


def one_regime(regime: LiquidityRegime = LIQUID) -> RegimeChain:
    return RegimeChain(regimes=(regime,), transitions=((1.0,),))


def problem(chain: RegimeChain, *, periods: int = PERIODS) -> AdaptiveProblem:
    return AdaptiveProblem(quantity=QUANTITY, horizon=HORIZON, periods=periods, chain=chain)


# -- the chain ----------------------------------------------------------------


def test_a_chain_needs_a_regime() -> None:
    with pytest.raises(ValidationError, match="at least one regime"):
        RegimeChain(regimes=(), transitions=())


def test_a_chain_refuses_a_matrix_of_the_wrong_shape() -> None:
    with pytest.raises(ValidationError, match="rows for"):
        RegimeChain(regimes=(LIQUID, ILLIQUID), transitions=((1.0, 0.0),))
    with pytest.raises(ValidationError, match="entries for"):
        RegimeChain(regimes=(LIQUID, ILLIQUID), transitions=((1.0,), (0.0, 1.0)))


def test_a_chain_refuses_a_row_that_does_not_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="leaks probability"):
        RegimeChain(regimes=(LIQUID, ILLIQUID), transitions=((0.7, 0.2), (0.3, 0.7)))


def test_a_chain_refuses_an_entry_that_is_not_a_probability() -> None:
    with pytest.raises(ValidationError, match="not a probability"):
        RegimeChain(regimes=(LIQUID, ILLIQUID), transitions=((1.5, -0.5), (0.3, 0.7)))


def test_a_row_of_thirds_is_accepted_rather_than_argued_with() -> None:
    third = 1.0 / 3.0
    chain = RegimeChain(
        regimes=(LIQUID, ILLIQUID, LIQUID),
        transitions=((third, third, third),) * 3,
    )
    assert len(chain) == 3


def test_the_distribution_mixes_towards_the_stationary_one() -> None:
    chain = two_regime(0.8)
    assert chain.distribution_after(0, 0).tolist() == [1.0, 0.0]
    one = chain.distribution_after(0, 1)
    assert one[0] == pytest.approx(0.8)
    far = chain.distribution_after(0, 200)
    assert far[0] == pytest.approx(0.5, abs=1e-12)


def test_the_distribution_refuses_a_regime_that_is_not_there() -> None:
    with pytest.raises(ValidationError, match="not one of the 2"):
        two_regime(0.8).distribution_after(7, 1)
    with pytest.raises(ValidationError, match="non-negative"):
        two_regime(0.8).distribution_after(0, -1)


def test_a_negative_volatility_is_refused() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        LiquidityRegime(impact=LinearImpact(gamma=0.0, eta=1e-6), volatility=-0.1)


# -- the problem --------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("quantity", 0.0, "quantity must be positive"),
        ("horizon", -1.0, "horizon must be positive"),
    ],
)
def test_the_problem_refuses_nonsense(field: str, value: float, message: str) -> None:
    arguments = {
        "quantity": QUANTITY,
        "horizon": HORIZON,
        "periods": PERIODS,
        "chain": one_regime(),
    }
    arguments[field] = value
    with pytest.raises(ValidationError, match=message):
        AdaptiveProblem(**arguments)  # type: ignore[arg-type]


def test_the_problem_refuses_too_few_periods() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        problem(one_regime(), periods=0)


def test_the_problem_refuses_a_regime_whose_permanent_impact_dominates() -> None:
    """One period of permanent impact outrunning the temporary concession.

    The same guard :class:`~slippage.execution.ExecutionProblem` carries, and it
    has to be per regime: a chain is admissible only if every state in it is,
    because the policy can be routed through any of them.
    """
    heavy = LiquidityRegime(
        impact=LinearImpact(gamma=1e-3, eta=2.5e-6), volatility=0.3, label="heavy"
    )
    with pytest.raises(ValidationError, match="regime heavy has eta"):
        problem(RegimeChain(regimes=(LIQUID, heavy), transitions=((0.5, 0.5),) * 2))


def test_the_problem_hands_back_the_static_problem_of_each_regime() -> None:
    chain = two_regime(0.8)
    one = problem(chain)
    assert one.in_regime(1).impact.eta == pytest.approx(1.25e-5)
    assert one.in_regime(0).volatility == pytest.approx(0.3)
    assert one.in_regime(0).tau == pytest.approx(one.tau)
    with pytest.raises(ValidationError, match="not one of the 2"):
        one.in_regime(5)


def test_the_fixed_cost_is_the_part_no_schedule_can_move() -> None:
    charged = LiquidityRegime(
        impact=LinearImpact(gamma=2.5e-7, eta=2.5e-6, epsilon=0.01), volatility=0.3
    )
    one = problem(one_regime(charged))
    assert one.fixed_cost(0) == pytest.approx(0.01 * QUANTITY + 0.5 * 2.5e-7 * QUANTITY**2)


# -- the degenerate cases, which are what pin everything else -----------------


def test_one_regime_reproduces_the_closed_form_trajectory() -> None:
    """The test the rest of the module rests on.

    A chain with one state is an Almgren-Chriss problem. If the recursion does
    not land on the closed form here, every gain it reports later is a gain over
    the wrong baseline and nothing downstream would reveal it.
    """
    one = problem(one_regime())
    policy = solve_adaptive(one, RISK_AVERSION)
    closed = optimal_trajectory(one.in_regime(0), RISK_AVERSION)

    remaining = QUANTITY
    holdings = [remaining]
    for size in policy.along([0] * PERIODS):
        remaining -= size
        holdings.append(remaining)

    for mine, theirs in zip(holdings, closed.holdings, strict=True):
        assert abs(mine - theirs) / QUANTITY < 1e-15
    assert policy.value(0) == pytest.approx(closed.objective(RISK_AVERSION), rel=1e-14)


@pytest.mark.parametrize("risk_aversion", [0.0, 1e-7, 2e-6, 1e-4])
def test_one_regime_agrees_with_the_closed_form_at_every_urgency(
    risk_aversion: float,
) -> None:
    one = problem(one_regime())
    policy = solve_adaptive(one, risk_aversion)
    closed = optimal_trajectory(one.in_regime(0), risk_aversion)
    assert policy.value(0) == pytest.approx(closed.objective(risk_aversion), rel=1e-13)


def test_one_regime_leaves_nothing_for_adapting_to_find() -> None:
    gain = adaptivity_gain(problem(one_regime()), RISK_AVERSION, 0)
    assert abs(gain.fraction) < 1e-14


@pytest.mark.parametrize("persistence", [0.0, 1.0])
def test_a_predictable_chain_is_worth_nothing_to_adapt_to(persistence: float) -> None:
    """Both ends of the range, and they are zero for different reasons.

    At a persistence of one the chain never moves, so the starting regime is the
    regime forever and the static schedule can use it. At zero the chain
    alternates strictly, which is just as predictable. Variability is not
    uncertainty, and only uncertainty is worth reacting to.
    """
    gain = adaptivity_gain(problem(two_regime(persistence)), RISK_AVERSION, 0)
    assert abs(gain.saved) / gain.static < 1e-14


@pytest.mark.parametrize("periods", [1, 2])
def test_fewer_than_three_periods_cannot_gain_anything(periods: int) -> None:
    """Provably zero rather than numerically small, and the reason is structural.

    The first period's regime is known to the static schedule as well, and the
    last period has no decision in it — whatever is left must be traded. A
    two-period problem therefore reveals nothing before its only choice, and the
    saving comes back as exactly ``0.0``.
    """
    gain = adaptivity_gain(problem(two_regime(0.8), periods=periods), RISK_AVERSION, 0)
    assert gain.saved == 0.0


def test_three_periods_is_where_adapting_starts_to_pay() -> None:
    gain = adaptivity_gain(problem(two_regime(0.8), periods=3), RISK_AVERSION, 0)
    assert gain.fraction == pytest.approx(0.0173, abs=5e-4)


def test_two_identical_regimes_are_worth_nothing_however_they_switch() -> None:
    chain = RegimeChain(regimes=(LIQUID, LIQUID), transitions=((0.7, 0.3), (0.3, 0.7)))
    gain = adaptivity_gain(problem(chain), RISK_AVERSION, 0)
    assert abs(gain.fraction) < 1e-14


# -- what adapting is actually worth ------------------------------------------


def test_the_gain_at_a_persistence_of_four_fifths() -> None:
    one = problem(two_regime(0.8))
    assert adaptivity_gain(one, RISK_AVERSION, 0).fraction == pytest.approx(0.2403, abs=5e-4)
    assert adaptivity_gain(one, RISK_AVERSION, 1).fraction == pytest.approx(0.2464, abs=5e-4)


def test_the_gain_peaks_well_onto_the_mean_reverting_side() -> None:
    """Not in the middle, which is where it was expected.

    A persistence of 0.182: informative about the next period and nearly
    uninformative about the one after. The grid here is coarse enough to run in
    the suite and fine enough to locate the peak away from 0.5, which is the
    claim being pinned.
    """
    best = (0.0, -1.0)
    for step in range(1, 40):
        persistence = step / 40.0
        fraction = adaptivity_gain(problem(two_regime(persistence)), RISK_AVERSION, 0).fraction
        if fraction > best[1]:
            best = (persistence, fraction)
    assert best[0] == pytest.approx(0.175, abs=0.03)
    assert best[1] == pytest.approx(0.373, abs=0.005)
    # And the middle of the range is well below it.
    middle = adaptivity_gain(problem(two_regime(0.5)), RISK_AVERSION, 0).fraction
    assert middle == pytest.approx(0.3423, abs=5e-4)
    assert middle < best[1]


def test_adapting_is_worth_most_to_a_trader_who_does_not_care_about_risk() -> None:
    """Monotone decreasing in risk aversion, which is the opposite of the guess.

    A risk penalty is charged on inventory whatever the regime, so the more of
    the objective it accounts for the less of it the regime can move.
    """
    one = problem(two_regime(0.182))
    gains = [
        adaptivity_gain(one, risk_aversion, 0).fraction for risk_aversion in (0.0, 2e-6, 1e-4, 1e-3)
    ]
    assert gains == sorted(gains, reverse=True)
    assert gains[0] == pytest.approx(0.3764, abs=5e-4)
    assert gains[-1] == pytest.approx(0.1291, abs=5e-4)


def test_it_is_liquidity_worth_adapting_to_and_not_volatility() -> None:
    """The measurement that contradicts the usual description.

    Two regimes differing only in volatility by a factor of five are worth
    0.115%. Two differing only in impact by the same factor are worth 24.03%.
    """
    wild = LiquidityRegime(impact=LIQUID.impact, volatility=LIQUID.volatility * 5.0, label="wild")
    volatility_only = RegimeChain(regimes=(LIQUID, wild), transitions=((0.8, 0.2), (0.2, 0.8)))
    from_volatility = adaptivity_gain(problem(volatility_only), RISK_AVERSION, 0).fraction
    from_liquidity = adaptivity_gain(problem(two_regime(0.8, ratio=5.0)), RISK_AVERSION, 0).fraction

    assert from_volatility == pytest.approx(0.00115, abs=5e-5)
    assert from_liquidity == pytest.approx(0.2403, abs=5e-4)
    assert from_liquidity / from_volatility > 150.0


def test_the_gain_grows_with_how_different_the_regimes_are() -> None:
    fractions = [
        adaptivity_gain(problem(two_regime(0.8, ratio=ratio)), RISK_AVERSION, 0).fraction
        for ratio in (1.0, 2.0, 3.0, 5.0)
    ]
    assert fractions == sorted(fractions)
    assert abs(fractions[0]) < 1e-14
    assert fractions[1] == pytest.approx(0.0535, abs=5e-4)


# -- the policy's shape -------------------------------------------------------


def static_fractions(one: AdaptiveProblem, start: int) -> list[float]:
    schedule = static_schedule(one, RISK_AVERSION, start)
    remaining = one.quantity
    fractions = []
    for size in schedule.trades:
        fractions.append(size / remaining)
        remaining -= size
    return fractions


def test_the_policy_trades_more_when_liquid_and_less_when_not() -> None:
    one = problem(two_regime(0.8))
    policy = solve_adaptive(one, RISK_AVERSION)
    reference = static_fractions(one, 0)
    liquid, illiquid = policy.fractions[10]
    assert liquid / reference[10] == pytest.approx(1.906, abs=0.01)
    assert illiquid / reference[10] == pytest.approx(0.589, abs=0.01)


def test_the_first_period_trades_less_than_the_static_schedule_even_when_liquid() -> None:
    """Where the saving actually comes from.

    A static schedule starting liquid front-loads into the cheap trading it
    expects. The adaptive one does not have to, because it can wait for
    liquidity it will observe rather than liquidity it is forecasting — so in
    the first period it trades *less*, at 0.755 of the static fraction, despite
    being in the better state.
    """
    one = problem(two_regime(0.8))
    policy = solve_adaptive(one, RISK_AVERSION)
    reference = static_fractions(one, 0)
    assert policy.fractions[0][0] / reference[0] == pytest.approx(0.755, abs=0.01)
    assert policy.fractions[0][0] < reference[0]


def test_the_last_period_trades_everything_left() -> None:
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    assert policy.fractions[-1] == (1.0, 1.0)
    assert len(policy.fractions) == PERIODS
    assert len(policy.coefficients) == PERIODS


def test_the_policy_liquidates_along_any_path() -> None:
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    for path in ([0] * PERIODS, [1] * PERIODS, [0, 1] * (PERIODS // 2)):
        trades = policy.along(path)
        assert len(trades) == PERIODS
        assert all(size >= 0.0 for size in trades)
        assert math.fsum(trades) == pytest.approx(QUANTITY, rel=1e-12)


def test_the_policy_refuses_a_path_of_the_wrong_length() -> None:
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    with pytest.raises(ValidationError, match="expected 20 regimes"):
        policy.along([0, 1, 0])


def test_the_policy_refuses_a_period_or_a_regime_it_does_not_have() -> None:
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    with pytest.raises(ValidationError, match="not one of the 20"):
        policy.trade(20, 0, 100.0)
    with pytest.raises(ValidationError, match="not one of the 2"):
        policy.trade(0, 9, 100.0)
    with pytest.raises(ValidationError, match="non-negative"):
        policy.trade(0, 0, -1.0)


def test_the_policy_is_linear_in_what_is_left() -> None:
    """The fractions not depending on inventory is the reason there is no grid."""
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    assert policy.trade(3, 1, 2.0e5) == pytest.approx(2.0 * policy.trade(3, 1, 1.0e5))


def test_the_value_is_quadratic_in_the_order_size() -> None:
    """Up to the fixed costs, which are not.

    ``epsilon`` is linear in the order and permanent impact is quadratic, so the
    schedule-dependent part scaling with the square is checked after the fixed
    part is taken out.
    """
    small = problem(two_regime(0.8))
    big = AdaptiveProblem(
        quantity=2.0 * QUANTITY, horizon=HORIZON, periods=PERIODS, chain=two_regime(0.8)
    )
    one = solve_adaptive(small, RISK_AVERSION).value(0) - small.fixed_cost(0)
    other = solve_adaptive(big, RISK_AVERSION).value(0) - big.fixed_cost(0)
    assert other == pytest.approx(4.0 * one, rel=1e-12)


# -- the static baseline ------------------------------------------------------


def test_the_static_schedule_liquidates_and_splits_its_objective() -> None:
    one = problem(two_regime(0.8))
    schedule = static_schedule(one, RISK_AVERSION, 0)
    assert math.fsum(schedule.trades) == pytest.approx(QUANTITY, rel=1e-12)
    assert schedule.holdings[0] == QUANTITY
    assert schedule.holdings[-1] == 0.0
    assert schedule.expected_impact > 0.0
    assert schedule.expected_risk > 0.0
    assert schedule.objective == pytest.approx(
        schedule.expected_impact + schedule.expected_risk + one.fixed_cost(0)
    )


def test_the_static_schedule_is_the_closed_form_in_one_regime() -> None:
    one = problem(one_regime())
    schedule = static_schedule(one, RISK_AVERSION, 0)
    closed = optimal_trajectory(one.in_regime(0), RISK_AVERSION)
    for mine, theirs in zip(schedule.trades, closed.trades, strict=True):
        assert abs(mine - theirs) / QUANTITY < 1e-15
    assert schedule.objective == pytest.approx(closed.objective(RISK_AVERSION), rel=1e-13)


def test_the_static_schedule_is_not_between_the_two_regimes_own_schedules() -> None:
    """It is outside both of them, which is not what was expected.

    The guess was that a schedule facing a mixture would sit between the two
    pure schedules. It does not, and the reason is that the two pure schedules
    have *constant* coefficients and this one does not. Starting liquid, the
    expected cost of trading rises as the chain mixes away from the cheap state,
    so the schedule front-loads hard: 129,346 shares in the first of twenty
    periods against about 51,000 for either pure schedule, both of which are
    nearly uniform. Starting illiquid it is the mirror image, 32,167 in the
    first period and rising.

    Which is the whole point of the baseline. Comparing the adaptive policy
    against a pure schedule would credit it with a gain that any static trader
    who knew the chain's law could have taken.
    """
    one = problem(two_regime(0.8))
    from_liquid = static_schedule(one, RISK_AVERSION, 0).trades
    from_illiquid = static_schedule(one, RISK_AVERSION, 1).trades
    if_liquid = optimal_trajectory(one.in_regime(0), RISK_AVERSION).trades
    if_illiquid = optimal_trajectory(one.in_regime(1), RISK_AVERSION).trades

    assert from_liquid[0] == pytest.approx(129346.0, abs=5.0)
    assert from_illiquid[0] == pytest.approx(32167.0, abs=5.0)
    assert if_liquid[0] == pytest.approx(51109.0, abs=5.0)
    assert if_illiquid[0] == pytest.approx(50222.0, abs=5.0)
    # Outside the pure schedules on both sides, not between them.
    assert from_liquid[0] > max(if_liquid[0], if_illiquid[0])
    assert from_illiquid[0] < min(if_liquid[0], if_illiquid[0])
    # Front-loaded from the cheap start, back-loaded from the dear one.
    assert from_liquid[0] > from_liquid[1]
    assert from_illiquid[0] < from_illiquid[1]


def test_the_static_schedules_trajectory_view_carries_its_own_trades() -> None:
    one = problem(two_regime(0.8))
    schedule = static_schedule(one, RISK_AVERSION, 0)
    view = schedule.trajectory()
    for mine, theirs in zip(view.trades, schedule.trades, strict=True):
        assert mine == pytest.approx(theirs)


def test_the_static_schedule_refuses_a_regime_it_does_not_have() -> None:
    with pytest.raises(ValidationError, match="not one of the 2"):
        static_schedule(problem(two_regime(0.8)), RISK_AVERSION, 4)


@pytest.mark.parametrize("risk_aversion", [-1.0, math.nan, math.inf])
def test_a_risk_aversion_that_is_not_a_number_is_refused(risk_aversion: float) -> None:
    with pytest.raises(ValidationError, match="risk aversion"):
        solve_adaptive(problem(two_regime(0.8)), risk_aversion)
    with pytest.raises(ValidationError, match="risk aversion"):
        static_schedule(problem(two_regime(0.8)), risk_aversion, 0)


@pytest.mark.parametrize("persistence", [0.05, 0.3, 0.5, 0.75, 0.9, 0.99])
def test_the_optimum_is_never_worse_than_the_schedule_it_contains(
    persistence: float,
) -> None:
    """The sharpest available check on both solvers at once.

    A deterministic schedule is one of the policies the dynamic program ranges
    over, so its value cannot be lower than the program's optimum. An error in
    either solver breaks this on some chain, and it costs one comparison.
    """
    for start in (0, 1):
        gain = adaptivity_gain(problem(two_regime(persistence)), RISK_AVERSION, start)
        assert gain.saved >= -1e-9 * gain.static


def test_the_static_objective_is_positive_for_any_admissible_problem() -> None:
    """Which is why the ratio needs no guard on arithmetic grounds.

    Every regime has ``eta_tilde > 0`` or the problem is refused, so any mixture
    of them does, and the trades sum to a positive quantity — so some trade is
    positive and the impact term alone is positive. Checked at the extremes
    rather than argued: a near-zero impact, no risk aversion, one period.
    """
    faint = LiquidityRegime(
        impact=LinearImpact(gamma=0.0, eta=1e-30), volatility=0.0, label="faint"
    )
    for periods in (1, 2, 20):
        for risk_aversion in (0.0, 1e-9):
            schedule = static_schedule(
                problem(one_regime(faint), periods=periods), risk_aversion, 0
            )
            assert schedule.objective > 0.0


def test_an_objective_that_underflows_to_zero_reports_no_saving() -> None:
    """The one route to a zero denominator, and it is underflow rather than zero.

    A quantity of 1e-150 squares to zero, so the whole objective comes back as
    ``0.0`` and the ratio is undefined rather than small. At 1e-100 it is
    1e-230 and divides perfectly well, which is the other side of the boundary.
    """
    faint = LiquidityRegime(
        impact=LinearImpact(gamma=0.0, eta=1e-30), volatility=0.0, label="faint"
    )
    chain = one_regime(faint)
    underflowed = AdaptiveProblem(quantity=1e-150, horizon=HORIZON, periods=2, chain=chain)
    assert static_schedule(underflowed, 0.0, 0).objective == 0.0
    assert adaptivity_gain(underflowed, 0.0, 0).fraction == 0.0

    representable = AdaptiveProblem(quantity=1e-100, horizon=HORIZON, periods=2, chain=chain)
    assert static_schedule(representable, 0.0, 0).objective > 0.0


# -- the simulation, which shares no arithmetic with the recursion ------------


@pytest.mark.parametrize("start", [0, 1])
def test_the_simulated_objective_matches_the_recursion(start: int) -> None:
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    simulated = simulate_policy(policy, start, draws=4000, seed=20260101 + start)
    assert simulated.draws == 4000
    assert simulated.standard_error > 0.0
    assert simulated.covers(policy.value(start))


def test_the_simulation_of_a_single_regime_has_no_spread_at_all() -> None:
    """One regime means one path, so every draw is the same number.

    Which makes this the strictest form of the comparison: the simulated mean
    has to equal the recursion's value, not merely sit within its error.
    """
    one = problem(one_regime())
    policy = solve_adaptive(one, RISK_AVERSION)
    simulated = simulate_policy(policy, 0, draws=25, seed=7)
    assert simulated.standard_error == pytest.approx(0.0, abs=1e-6)
    assert simulated.mean == pytest.approx(policy.value(0), rel=1e-12)


def test_a_single_draw_covers_nothing_rather_than_everything() -> None:
    """An interval of infinite width contains the answer and says nothing.

    ``covers`` returned True here at first, which is the wrong way round: one
    draw is the weakest evidence available and it was being read as agreement.
    It now reports no agreement when the error is not finite.
    """
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    simulated = simulate_policy(policy, 0, draws=1, seed=3)
    assert simulated.standard_error == math.inf
    assert not simulated.covers(policy.value(0))
    assert not simulated.covers(simulated.mean)


def test_the_simulation_refuses_no_draws_or_an_absent_regime() -> None:
    policy = solve_adaptive(problem(two_regime(0.8)), RISK_AVERSION)
    with pytest.raises(ValidationError, match="at least one"):
        simulate_policy(policy, 0, draws=0)
    with pytest.raises(ValidationError, match="not one of the 2"):
        simulate_policy(policy, 3, draws=10)


def test_the_realised_objective_along_a_path_is_the_cost_of_its_own_trades() -> None:
    """Checked against the library's own cost and variance functions.

    ``realised`` adds up per-period terms. The same path's trades priced through
    ``schedule_cost`` and the variance sum have to agree, which is what ties the
    adaptive arithmetic back to the conventions the rest of the library uses.
    """
    from slippage.impact import schedule_cost

    one = problem(one_regime())
    policy = solve_adaptive(one, RISK_AVERSION)
    path = [0] * PERIODS
    trades = policy.along(path)

    cost = schedule_cost(LIQUID.impact, trades, one.tau).total
    held = QUANTITY
    variance = 0.0
    for size in trades:
        held -= size
        variance += held * held
    variance *= LIQUID.volatility**2 * one.tau

    assert policy.realised(path) == pytest.approx(cost + RISK_AVERSION * variance, rel=1e-12)
