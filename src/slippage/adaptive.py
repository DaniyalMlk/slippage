"""Adapting a schedule to the liquidity it finds, and what that is worth.

Every other schedule in this library is decided before the first share trades.
That is not a shortcut. Under the assumptions Almgren and Chriss make — impact
and volatility constant and known — the optimal strategy really is
deterministic, and a trader who waits to see what the market does cannot beat
one who commits. Nothing is revealed that the plan should depend on, so there is
nothing to react to.

What makes adapting worth something is liquidity that moves. Here it moves as a
Markov chain over :class:`LiquidityRegime` states, each carrying its own impact
coefficient and volatility, and the trader observes the current regime *before*
trading into it. That ordering is the whole model: a trader who had to commit to
a period's size before seeing its liquidity would be back to a deterministic
schedule with averaged coefficients.

**The value function stays quadratic, so there is no grid.** The per-period cost
is quadratic in the trade and the risk penalty is quadratic in the remaining
inventory, and both are homogeneous of degree two, so ``V_k(x, s) = a_k(s) x**2``
for one number per regime per period. Writing ``A = eta_tilde(s) / tau`` for the
cost of trading and ``B = lambda sigma(s)**2 tau + E_s[a_{k+1}]`` for the cost of
still holding, the step is

    minimise  A n**2 + B (x - n)**2   =>   n* = x B / (A + B),  value = x**2 A B / (A + B)

and the terminal constraint — liquidate, whatever it costs — is ``a_N = inf``,
which enters as a zero in the reciprocal form ``1 / a_k(s) = tau / eta_tilde(s) +
1 / B`` rather than as a special case in the algebra. The trade fraction
``B / (A + B)`` depends on the period and the regime but not on the inventory,
which is why the policy is a small table rather than a function of state.

With one regime it has to reproduce :func:`~slippage.execution.optimal_trajectory`,
and it does: holdings agree to **2.3e-16** of the order size and the objective to
**1.7e-16** relative. That is the only check that pins this machinery to the
closed form it generalises, and everything below is built on it.

**The objective is the running penalty, and it has to be.** What is minimised
here is ``E[cost] + lambda E[sum sigma**2 tau x**2]``, not ``E[cost] + lambda
Var[cost]``. For a deterministic schedule those are the same number — the
variance of the total cost *is* that sum — which is why the single-regime
agreement above is exact. For an adaptive schedule they are not, because the
inventory becomes random and the variance of the total picks up a term the sum
of conditional variances does not have. A variance of a total is not a sum of
per-period pieces, so dynamic programming does not apply to it; the running
penalty is the standard time-consistent substitute, and saying so is better than
quietly optimising a different objective from the one the rest of the library
reports.

**What adapting is worth.** The comparison is against the best *deterministic*
schedule facing the same chain, which is not either regime's own schedule: a
static trader who knows the chain's law should use the expected coefficients
period by period, and those drift as the chain mixes away from where it started.
:func:`static_schedule` solves exactly that, through the same recursion with one
state and time-varying coefficients.

On a two-regime chain whose illiquid state has five times the liquid state's
impact, twenty periods, starting liquid, the gain over that schedule is **24.03%**
at a persistence of 0.8. The shape of it is the part worth knowing, and three
features of it were guessed wrong before being measured.

*Both ends are worth nothing, and for the same reason.* At a persistence of 1
the chain never moves, so the starting regime is the regime forever and the
static schedule can use it. At a persistence of 0 the chain alternates strictly,
which is just as predictable, and the static schedule can use that too. Both
come out at zero to 4e-16 of the objective. A chain has to be uncertain, not
merely variable, before reacting to it pays.

*The peak is not in the middle.* Scanning persistence at 0.001 the maximum is
**37.30% at 0.182** — well onto the mean-reverting side, where a regime is
informative about the next period but not about the one after. The location is a
real feature rather than an artefact of one parameter set: it sits between 0.171
and 0.187 across ten, twenty and fifty periods and across impact ratios of two,
five and ten, drifting to about 0.25 only once risk aversion is large enough to
dominate the impact term.

*Adapting is worth most to a trader who does not care about risk.* The gain is
**37.64%** at zero risk aversion and falls monotonically — 37.30% at 2e-06, 26.92%
at 1e-04, 12.91% at 1e-03 — because a risk penalty is paid on inventory whatever
the regime, so the more of the objective it accounts for the less of it the
regime can move.

**Fewer than three periods cannot gain anything.** At one or two periods the
saving is exactly ``0.0``, and that is provable rather than numerical: the first
period's regime is known to the static schedule too, and the last period has no
decision in it, so a two-period problem reveals nothing before its only choice.
The first non-zero gain is 1.73% at three periods.

**It is liquidity that is worth adapting to, not volatility.** This is the
measurement that contradicts how execution adaptivity is usually described. Two
regimes differing only in volatility, by a factor of five, at a persistence of
0.8, are worth **0.115%**. Two differing only in impact, by the same factor of
five, are worth **24.03%** — two hundred and nine times as much. Reacting to a
volatility spike is close to worthless here; reacting to a liquidity one is the
whole effect.

**And the saving comes from waiting, not from hurrying.** "Trade more when
liquidity is good" describes the policy's middle periods: at the tenth of twenty
it trades **1.906** times the static fraction when liquid and **0.589** times it
when illiquid. But in the *first* period it trades 0.755 times the static
fraction even in the liquid state, because a static schedule starting liquid
front-loads into the cheap trading it expects, and the adaptive one does not
need to — it can wait for liquidity it will actually observe. Which is also why
the gain is larger from the illiquid start at a persistence of 0.8, 24.64%
against 24.03%, and the ordering reverses by 0.9, where the liquid start gains
15.06% against 11.99%.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError
from .execution import ExecutionProblem, Trajectory, trajectory_from_trades
from .impact import LinearImpact

__all__ = [
    "AdaptivePolicy",
    "AdaptiveProblem",
    "AdaptivityGain",
    "LiquidityRegime",
    "RegimeChain",
    "SimulatedValue",
    "StaticSchedule",
    "adaptivity_gain",
    "simulate_policy",
    "solve_adaptive",
    "static_schedule",
]

FloatArray = NDArray[np.float64]

#: Rows of a transition matrix are allowed to miss one by this much before the
#: matrix is refused. Tighter than the tolerance on a sum of a few dozen
#: floats needs to be, and loose enough that a hand-written row of thirds is
#: accepted rather than argued with.
_ROW_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LiquidityRegime:
    """One state of the market: an impact model and a volatility.

    The volatility is in price units per square root of the time unit, the same
    convention :class:`~slippage.execution.ExecutionProblem` uses, because the
    two are compared against each other throughout.
    """

    impact: LinearImpact
    volatility: float
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValidationError(
                f"volatility must be non-negative and finite, got {self.volatility!r}"
            )

    @property
    def name(self) -> str:
        return self.label or f"eta={self.impact.eta:g}"

    def eta_tilde(self, tau: float) -> float:
        """Temporary impact net of the permanent impact taken within a period."""
        return self.impact.eta - 0.5 * self.impact.gamma * tau


@dataclass(frozen=True)
class RegimeChain:
    """Regimes and the transition matrix between them.

    ``transitions[i][j]`` is the probability of being in regime ``j`` next
    period given regime ``i`` now. The regime for a period is observed before
    that period is traded, so the chain is a filtration the policy can use and
    not merely a description of what happens.
    """

    regimes: tuple[LiquidityRegime, ...]
    transitions: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.regimes:
            raise ValidationError("a chain needs at least one regime")
        size = len(self.regimes)
        if len(self.transitions) != size:
            raise ValidationError(
                f"the transition matrix has {len(self.transitions)} rows for {size} regimes"
            )
        for index, row in enumerate(self.transitions):
            if len(row) != size:
                raise ValidationError(
                    f"row {index} of the transition matrix has {len(row)} entries "
                    f"for {size} regimes"
                )
            for column, value in enumerate(row):
                if not math.isfinite(value) or value < 0.0 or value > 1.0:
                    raise ValidationError(
                        f"transition {index}->{column} is {value!r}, which is not a probability"
                    )
            total = math.fsum(row)
            if abs(total - 1.0) > _ROW_TOLERANCE:
                raise ValidationError(
                    f"row {index} of the transition matrix sums to {total!r} rather "
                    "than one, so the chain leaks probability: from regime "
                    f"{self.regimes[index].name} the next period is "
                    f"{'under' if total < 1.0 else 'over'}-determined"
                )

    def __len__(self) -> int:
        return len(self.regimes)

    @property
    def matrix(self) -> FloatArray:
        return np.array(self.transitions, dtype=np.float64)

    def distribution_after(self, start: int, steps: int) -> FloatArray:
        """Where the chain is ``steps`` periods after starting in ``start``."""
        self.check_regime(start)
        if steps < 0:
            raise ValidationError(f"steps must be non-negative, got {steps!r}")
        state: FloatArray = np.zeros(len(self), dtype=np.float64)
        state[start] = 1.0
        matrix = self.matrix
        for _ in range(steps):
            state = np.asarray(state @ matrix, dtype=np.float64)
        return state

    def check_regime(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise ValidationError(f"regime {index!r} is not one of the {len(self)} in this chain")
        return index


@dataclass(frozen=True)
class AdaptiveProblem:
    """A liquidation to be worked through a chain of liquidity regimes."""

    quantity: float
    horizon: float
    periods: int
    chain: RegimeChain

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantity) or self.quantity <= 0.0:
            raise ValidationError(f"quantity must be positive, got {self.quantity!r}")
        if not math.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValidationError(f"horizon must be positive, got {self.horizon!r}")
        if self.periods < 1:
            raise ValidationError(f"periods must be at least 1, got {self.periods!r}")
        for regime in self.chain.regimes:
            if regime.eta_tilde(self.tau) <= 0.0:
                raise ValidationError(
                    f"regime {regime.name} has eta - gamma * tau / 2 = "
                    f"{regime.eta_tilde(self.tau):.4g}, which is not positive: its "
                    "permanent impact outweighs its temporary impact over one "
                    "period, so trading the whole order in a single period would "
                    "look cheapest. Use more periods."
                )

    @property
    def tau(self) -> float:
        return self.horizon / self.periods

    def times(self) -> list[float]:
        return [index * self.tau for index in range(self.periods + 1)]

    def in_regime(self, index: int) -> ExecutionProblem:
        """The static problem this becomes if ``index`` held for the whole horizon.

        The bridge back to :func:`~slippage.execution.optimal_trajectory`, and
        the thing a one-regime chain has to agree with exactly.
        """
        regime = self.chain.regimes[self.chain.check_regime(index)]
        return ExecutionProblem(
            quantity=self.quantity,
            horizon=self.horizon,
            periods=self.periods,
            volatility=regime.volatility,
            impact=regime.impact,
        )

    def fixed_cost(self, index: int) -> float:
        """The part of the cost no schedule can change, in regime ``index``.

        The per-share charge on the whole order and half the permanent impact of
        moving the price by ``gamma * X``. Both are quadratic constants rather
        than functions of the schedule, so they sit outside the optimisation —
        and they are reported rather than dropped, because a cost that excludes
        them is not the cost anybody pays.
        """
        regime = self.chain.regimes[self.chain.check_regime(index)]
        return regime.impact.epsilon * self.quantity + 0.5 * regime.impact.gamma * self.quantity**2


@dataclass(frozen=True)
class AdaptivePolicy:
    """The optimal adaptive policy, as a table of trade fractions.

    ``fractions[k][s]`` is the share of whatever is left that should be traded
    in period ``k`` when the regime is ``s``. The last row is all ones: the
    position has to be flat at the horizon.

    ``coefficients[k][s]`` is ``a_k(s)``, so the schedule-dependent part of the
    objective from period ``k`` onwards, holding ``x``, is ``a_k(s) x**2``.
    """

    problem: AdaptiveProblem
    risk_aversion: float
    coefficients: tuple[tuple[float, ...], ...]
    fractions: tuple[tuple[float, ...], ...]

    def value(self, start: int) -> float:
        """The objective from the start, including the costs no schedule moves."""
        self.problem.chain.check_regime(start)
        return self.coefficients[0][start] * self.problem.quantity**2 + self.problem.fixed_cost(
            start
        )

    def trade(self, period: int, regime: int, remaining: float) -> float:
        """How much to trade now, given the period, the regime and what is left."""
        if not 0 <= period < self.problem.periods:
            raise ValidationError(
                f"period {period!r} is not one of the {self.problem.periods} in this problem"
            )
        self.problem.chain.check_regime(regime)
        if not math.isfinite(remaining) or remaining < 0.0:
            raise ValidationError(f"remaining inventory must be non-negative, got {remaining!r}")
        return self.fractions[period][regime] * remaining

    def along(self, path: Sequence[int]) -> list[float]:
        """The trades the policy makes along one realised sequence of regimes."""
        if len(path) != self.problem.periods:
            raise ValidationError(f"expected {self.problem.periods} regimes, got {len(path)}")
        remaining = self.problem.quantity
        trades = []
        for period, regime in enumerate(path):
            size = self.trade(period, regime, remaining)
            trades.append(size)
            remaining -= size
        # The last fraction is exactly one, so what is left is rounding only.
        if trades:
            trades[-1] += remaining
        return trades

    def realised(self, path: Sequence[int]) -> float:
        """The objective actually incurred along one regime path."""
        trades = self.along(path)
        return _path_objective(self.problem, self.risk_aversion, path, trades)


def _path_objective(
    problem: AdaptiveProblem,
    risk_aversion: float,
    path: Sequence[int],
    trades: Sequence[float],
) -> float:
    """``sum eta_tilde(s_k) n_k**2 / tau + lambda sum sigma(s_k)**2 tau x_{k+1}**2``.

    Plus the costs no schedule moves, charged in the regime the order started
    in, so this is comparable with :meth:`AdaptivePolicy.value`.
    """
    tau = problem.tau
    held = problem.quantity
    total = 0.0
    for regime_index, size in zip(path, trades, strict=True):
        regime = problem.chain.regimes[regime_index]
        held -= size
        total += regime.eta_tilde(tau) * size * size / tau
        total += risk_aversion * regime.volatility**2 * tau * held * held
    return total + problem.fixed_cost(path[0])


def _check_risk_aversion(risk_aversion: float) -> float:
    if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
        raise ValidationError(
            f"risk aversion must be non-negative and finite, got {risk_aversion!r}"
        )
    return risk_aversion


def solve_adaptive(problem: AdaptiveProblem, risk_aversion: float) -> AdaptivePolicy:
    """Backward induction on one coefficient per regime per period.

    Exact rather than approximate: the value function is quadratic in the
    remaining inventory at every step, so nothing is discretised and no
    inventory grid appears. The recursion runs on ``1 / a``, which is what lets
    the terminal constraint — liquidate, whatever it costs — enter as a zero
    rather than as a special case.
    """
    risk_aversion = _check_risk_aversion(risk_aversion)
    tau = problem.tau
    chain = problem.chain
    size = len(chain)
    matrix = chain.matrix

    coefficients: list[tuple[float, ...]] = []
    fractions: list[tuple[float, ...]] = []

    # The last period has no choice in it: whatever is left is traded, at this
    # regime's own impact. a_{N-1}(s) = eta_tilde(s) / tau.
    last_a = [chain.regimes[index].eta_tilde(tau) / tau for index in range(size)]
    coefficients.append(tuple(last_a))
    fractions.append(tuple(1.0 for _ in range(size)))

    for _ in range(problem.periods - 1):
        ahead = np.array(coefficients[0], dtype=np.float64)
        expected = matrix @ ahead
        row_a: list[float] = []
        row_f: list[float] = []
        for index in range(size):
            regime = chain.regimes[index]
            trading = regime.eta_tilde(tau) / tau
            holding = risk_aversion * regime.volatility**2 * tau + float(expected[index])
            if holding == 0.0:
                # A risk-neutral trader facing a last period that costs nothing
                # is indifferent; trading nothing now is the convention, and it
                # is reachable only with zero volatility and zero impact ahead.
                row_a.append(0.0)
                row_f.append(0.0)
                continue
            row_a.append(trading * holding / (trading + holding))
            row_f.append(holding / (trading + holding))
        coefficients.insert(0, tuple(row_a))
        fractions.insert(0, tuple(row_f))

    return AdaptivePolicy(
        problem=problem,
        risk_aversion=risk_aversion,
        coefficients=tuple(coefficients),
        fractions=tuple(fractions),
    )


@dataclass(frozen=True)
class StaticSchedule:
    """The best schedule that has to be fixed in advance, and its objective.

    ``expected_impact`` and ``expected_risk`` are the two halves of the
    objective under the chain's law, so a caller can see which one the schedule
    is trading off rather than only the total.
    """

    problem: AdaptiveProblem
    risk_aversion: float
    start: int
    trades: tuple[float, ...]
    holdings: tuple[float, ...]
    expected_impact: float
    expected_risk: float

    @property
    def objective(self) -> float:
        return self.expected_impact + self.expected_risk + self.problem.fixed_cost(self.start)

    def trajectory(self) -> Trajectory:
        """The schedule as a :class:`~slippage.execution.Trajectory`.

        Reported in the regime the order started in, because a trajectory
        carries one impact model and one volatility and this schedule faces a
        distribution of them. The *trades* are the schedule's own; only the
        moments attached to them are the starting regime's, which is why this is
        a view for plotting and comparing shapes rather than a cost.
        """
        return trajectory_from_trades(self.problem.in_regime(self.start), self.trades)


def static_schedule(problem: AdaptiveProblem, risk_aversion: float, start: int) -> StaticSchedule:
    """The optimal deterministic schedule facing the chain from ``start``.

    Not the single-regime schedule. A trader who must commit in advance but
    knows the chain's law should use the *expected* coefficients period by
    period, and those drift as the chain mixes away from where it started: a
    liquid start facing a persistent chain expects cheap trading soon and
    average trading later, which tilts the schedule earlier than either regime's
    own schedule would be.

    Because the schedule is deterministic, the expectation passes straight
    through the quadratic: ``E[sum eta_tilde(s_k) n_k**2] = sum E[eta_tilde(s_k)]
    n_k**2``. So this is the same recursion as :func:`solve_adaptive` with one
    state and time-varying coefficients, which is also why the two agree exactly
    when there is nothing to adapt to.
    """
    risk_aversion = _check_risk_aversion(risk_aversion)
    problem.chain.check_regime(start)
    tau = problem.tau
    regimes = problem.chain.regimes

    trading: list[float] = []
    holding: list[float] = []
    for period in range(problem.periods):
        weights = problem.chain.distribution_after(start, period)
        eta = float(
            sum(
                weight * regime.eta_tilde(tau)
                for weight, regime in zip(weights, regimes, strict=True)
            )
        )
        variance = float(
            sum(
                weight * regime.volatility**2
                for weight, regime in zip(weights, regimes, strict=True)
            )
        )
        trading.append(eta / tau)
        holding.append(risk_aversion * variance * tau)

    # Backward recursion over one state: a_k = A_k (B_k + a_{k+1}) / (A_k + B_k
    # + a_{k+1}), with a_N infinite so the last period trades everything.
    ahead = 0.0
    forward_fractions: list[float] = []
    for period in reversed(range(problem.periods)):
        if period == problem.periods - 1:
            forward_fractions.append(1.0)
            ahead = trading[period]
            continue
        rest = holding[period] + ahead
        if rest == 0.0:
            forward_fractions.append(0.0)
            ahead = 0.0
            continue
        forward_fractions.append(rest / (trading[period] + rest))
        ahead = trading[period] * rest / (trading[period] + rest)
    forward_fractions.reverse()

    remaining = problem.quantity
    trades: list[float] = []
    holdings: list[float] = [remaining]
    for fraction in forward_fractions:
        size = fraction * remaining
        trades.append(size)
        remaining -= size
        holdings.append(remaining)
    if trades:
        trades[-1] += remaining
        holdings[-1] = 0.0

    impact = sum(trading[period] * trades[period] ** 2 for period in range(problem.periods))
    risk = sum(holding[period] * holdings[period + 1] ** 2 for period in range(problem.periods))
    return StaticSchedule(
        problem=problem,
        risk_aversion=risk_aversion,
        start=start,
        trades=tuple(trades),
        holdings=tuple(holdings),
        expected_impact=impact,
        expected_risk=risk,
    )


@dataclass(frozen=True)
class AdaptivityGain:
    """What adapting is worth, from one starting regime."""

    start: int
    adaptive: float
    static: float

    @property
    def saved(self) -> float:
        return self.static - self.adaptive

    @property
    def fraction(self) -> float:
        """The saving as a share of the static objective, zero if there is none."""
        if self.static == 0.0:
            return 0.0
        return self.saved / self.static


def adaptivity_gain(problem: AdaptiveProblem, risk_aversion: float, start: int) -> AdaptivityGain:
    """The adaptive policy's objective against the best static schedule's.

    Never negative, to rounding: the static schedule is one of the policies the
    dynamic program ranges over, so the optimum cannot be worse than it. That
    inequality is the sharpest available check on the two solvers at once, since
    an error in either breaks it on some chain.
    """
    policy = solve_adaptive(problem, risk_aversion)
    fixed = static_schedule(problem, risk_aversion, start)
    return AdaptivityGain(start=start, adaptive=policy.value(start), static=fixed.objective)


@dataclass(frozen=True)
class SimulatedValue:
    """A Monte Carlo average of the policy's realised objective."""

    mean: float
    standard_error: float
    draws: int

    def covers(self, value: float, *, errors: float = 3.0) -> bool:
        """Whether ``value`` sits within ``errors`` standard errors of the mean."""
        return abs(self.mean - value) <= errors * self.standard_error


def simulate_policy(
    policy: AdaptivePolicy,
    start: int,
    *,
    draws: int = 20_000,
    seed: int | np.random.Generator | None = None,
) -> SimulatedValue:
    """Run the policy down simulated regime paths and average what it costs.

    The point of this is that it shares no arithmetic with the recursion. The
    recursion computes a value; this draws paths, applies the policy's
    fractions, and adds up the costs those trades actually incur. If the
    quadratic algebra is wrong the two disagree, and no amount of checking the
    recursion against itself would show it.
    """
    problem = policy.problem
    problem.chain.check_regime(start)
    if draws < 1:
        raise ValidationError(f"draws must be at least one, got {draws!r}")
    generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    matrix = problem.chain.matrix
    size = len(problem.chain)

    total = 0.0
    squares = 0.0
    for _ in range(draws):
        path = [start]
        for _ in range(problem.periods - 1):
            path.append(int(generator.choice(size, p=matrix[path[-1]])))
        realised = policy.realised(path)
        total += realised
        squares += realised * realised
    mean = total / draws
    if draws == 1:
        return SimulatedValue(mean=mean, standard_error=math.inf, draws=draws)
    variance = max(squares / draws - mean * mean, 0.0) * draws / (draws - 1)
    return SimulatedValue(mean=mean, standard_error=math.sqrt(variance / draws), draws=draws)
