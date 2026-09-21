"""Constrained execution schedules by dynamic programming.

The Almgren-Chriss closed form needs linear impact and no constraints. Real
orders carry participation caps, minimum working rates and lot sizes, and a
calibrated impact model may well be a power law. This module solves the same
mean-variance problem numerically.

The state is ``(period, remaining quantity)`` on a grid of whole lots. Trading
``n`` shares in period ``k`` with ``x`` shares still to do costs

* temporary impact ``n * h(n / tau)``,
* permanent impact ``gamma * n * (X - x)`` — the price has already been moved
  by every share done before this one, and
* risk ``lambda * sigma**2 * tau * (x - n)**2`` on what is still held through
  the interval,

and the backward recursion ``V_k(x) = min_n [cost + V_{k+1}(x - n)]`` with
``V_N(0) = 0`` and ``V_N(x > 0) = inf`` finds the optimum. Each period is a
single vectorised minimisation over a ``(states x trades)`` matrix, so the cost
is ``O(N M**2)`` for ``M`` lots — fine into the low thousands of lots.

Because the solution is a *policy* over every state, not a single path,
re-optimising after fills go off plan is a table lookup rather than a new solve.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError
from .impact import ImpactModel, schedule_cost

__all__ = [
    "SchedulePlan",
    "participation_caps",
    "schedule_objective",
    "solve_schedule",
]

# Snap a quantity to the lot grid when it is within this many lots of a whole
# number, so 1e6 / 1e3 counts as 1000 lots despite binary rounding.
_LOT_TOL = 1e-9


def _per_period(
    name: str, value: float | Sequence[float] | None, periods: int
) -> list[float | None]:
    if value is None:
        return [None] * periods
    if isinstance(value, (int, float)):
        values = [float(value)] * periods
    else:
        values = [float(v) for v in value]
        if len(values) != periods:
            raise ValidationError(f"{name} has {len(values)} entries for {periods} periods")
    for v in values:
        if not math.isfinite(v) or v < 0.0:
            raise ValidationError(f"{name} entries must be non-negative, got {v!r}")
    return list(values)


def participation_caps(expected_volume: Sequence[float], rate: float) -> list[float]:
    """Per-period trade limits from an expected volume profile.

    ``rate`` is the maximum fraction of market volume the order may take, so a
    10% cap in a period expected to print 50,000 shares allows 5,000.
    """
    if not 0.0 < rate <= 1.0:
        raise ValidationError(f"participation rate must be in (0, 1], got {rate!r}")
    caps = []
    for v in expected_volume:
        if not math.isfinite(v) or v < 0.0:
            raise ValidationError(f"expected volume must be non-negative, got {v!r}")
        caps.append(rate * v)
    return caps


def schedule_objective(
    impact: ImpactModel,
    trades: Sequence[float],
    tau: float,
    volatility: float,
    risk_aversion: float,
    *,
    already_done: float = 0.0,
) -> float:
    """``E[cost] + lambda Var[cost]`` for a schedule, computed independently of the solver.

    ``already_done`` shares executed before this schedule starts add the
    permanent impact they left behind to every remaining share.
    """
    remaining = sum(trades)
    cost = schedule_cost(impact, trades, tau).total + impact.gamma * already_done * remaining
    held = remaining
    variance = 0.0
    for n in trades:
        held -= n
        variance += held * held
    return cost + risk_aversion * volatility**2 * tau * variance


@dataclass(frozen=True)
class SchedulePlan:
    """The solved programme: value function and optimal action for every state."""

    quantity: float
    periods: int
    tau: float
    lot_size: float
    risk_aversion: float
    values: NDArray[np.float64] = field(repr=False)
    """``values[k, i]``: optimal cost-to-go from period ``k`` with ``i`` lots left."""
    policy: NDArray[np.int64] = field(repr=False)
    """``policy[k, i]``: lots to trade in period ``k`` with ``i`` lots left; -1 if infeasible."""

    @property
    def lots(self) -> int:
        return int(self.values.shape[1]) - 1

    @property
    def objective(self) -> float:
        """Optimal ``E + lambda V`` from the start."""
        return float(self.values[0, self.lots])

    def _lots(self, remaining: float) -> int:
        count = remaining / self.lot_size
        rounded = round(count)
        if abs(count - rounded) > _LOT_TOL * max(1.0, count) or not 0 <= rounded <= self.lots:
            raise ValidationError(
                f"remaining quantity {remaining} is not a whole number of lots "
                f"between 0 and {self.quantity}"
            )
        return int(rounded)

    def is_feasible(self, period: int, remaining: float) -> bool:
        if period == self.periods:
            return self._lots(remaining) == 0
        return bool(np.isfinite(self.values[period, self._lots(remaining)]))

    def next_trade(self, period: int, remaining: float) -> float:
        """The optimal trade in ``period`` with ``remaining`` shares still to do."""
        if not 0 <= period < self.periods:
            raise ValidationError(f"period must be in [0, {self.periods}), got {period}")
        i = self._lots(remaining)
        action = int(self.policy[period, i])
        if action < 0:
            raise ValidationError(
                f"no schedule from period {period} with {remaining} shares left "
                "satisfies the constraints"
            )
        return action * self.lot_size

    def trades_from(self, period: int, remaining: float) -> list[float]:
        """The optimal schedule for the rest of the horizon from any state.

        This is the re-optimisation after fills go off plan: the policy already
        covers every state, so no new solve is needed.
        """
        trades = []
        left = remaining
        for k in range(period, self.periods):
            n = self.next_trade(k, left)
            trades.append(n)
            left = (self._lots(left) - round(n / self.lot_size)) * self.lot_size
        return trades

    @property
    def trades(self) -> list[float]:
        """The optimal schedule from the start."""
        return self.trades_from(0, self.quantity)


def solve_schedule(
    *,
    quantity: float,
    horizon: float,
    periods: int,
    volatility: float,
    impact: ImpactModel,
    risk_aversion: float,
    lot_size: float,
    max_trade: float | Sequence[float] | None = None,
    min_trade: float | Sequence[float] | None = None,
) -> SchedulePlan:
    """Solve the constrained mean-variance execution problem.

    Parameters
    ----------
    quantity, horizon, periods, volatility, risk_aversion
        As for :class:`~slippage.execution.ExecutionProblem`.
    impact
        Any :class:`~slippage.impact.ImpactModel`; power-law impact is fine.
    lot_size
        Grid resolution and the smallest tradable unit. ``quantity`` must be a
        whole number of lots.
    max_trade
        Largest trade allowed in each period, as one number or one per period;
        see :func:`participation_caps`. Rounded *down* to whole lots.
    min_trade
        Smallest trade in each period while shares remain, rounded *up* to
        whole lots. A floor larger than what is left requires only what is left.
    """
    if not math.isfinite(quantity) or quantity <= 0.0:
        raise ValidationError(f"quantity must be positive, got {quantity!r}")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValidationError(f"horizon must be positive, got {horizon!r}")
    if periods < 1:
        raise ValidationError(f"periods must be at least 1, got {periods}")
    if not math.isfinite(volatility) or volatility < 0.0:
        raise ValidationError(f"volatility must be non-negative, got {volatility!r}")
    if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
        raise ValidationError(f"risk aversion must be non-negative, got {risk_aversion!r}")
    if not math.isfinite(lot_size) or lot_size <= 0.0:
        raise ValidationError(f"lot size must be positive, got {lot_size!r}")
    count = quantity / lot_size
    lots = round(count)
    if lots < 1 or abs(count - lots) > _LOT_TOL * max(1.0, count):
        raise ValidationError(f"quantity {quantity} is not a whole number of {lot_size}-share lots")

    caps = _per_period("max_trade", max_trade, periods)
    floors = _per_period("min_trade", min_trade, periods)
    tau = horizon / periods

    sizes = np.arange(lots + 1, dtype=np.float64) * lot_size
    temporary = np.array([n * impact.temporary(n / tau) for n in sizes])
    state = np.arange(lots + 1)[:, None]
    action = np.arange(lots + 1)[None, :]
    after = state - action
    valid = after >= 0
    after_clipped = np.where(valid, after, 0)
    done_before = quantity - state * lot_size
    permanent = impact.gamma * (action * lot_size) * done_before
    risk = risk_aversion * volatility**2 * tau * (after_clipped * lot_size) ** 2
    step_cost = temporary[None, :] + permanent + risk

    values = np.full((periods + 1, lots + 1), np.inf)
    values[periods, 0] = 0.0
    policy = np.full((periods, lots + 1), -1, dtype=np.int64)

    for k in range(periods - 1, -1, -1):
        allowed = valid.copy()
        cap = caps[k]
        if cap is not None:
            allowed &= action <= math.floor(cap / lot_size + _LOT_TOL)
        floor = floors[k]
        if floor is not None:
            floor_lots = math.ceil(floor / lot_size - _LOT_TOL)
            allowed &= action >= np.minimum(floor_lots, state)
        total = np.where(allowed, step_cost + values[k + 1][after_clipped], np.inf)
        best = np.argmin(total, axis=1)
        best_value = total[np.arange(lots + 1), best]
        values[k] = best_value
        policy[k] = np.where(np.isfinite(best_value), best, -1)

    if not np.isfinite(values[0, lots]):
        detail = ""
        if all(c is not None for c in caps):
            cap_total = sum(c for c in caps if c is not None)
            detail = f"; the caps allow at most {cap_total:g} shares"
        raise ValidationError(
            f"no schedule completes {quantity:g} shares under the constraints{detail}"
        )

    return SchedulePlan(
        quantity=quantity,
        periods=periods,
        tau=tau,
        lot_size=lot_size,
        risk_aversion=risk_aversion,
        values=values,
        policy=policy,
    )
