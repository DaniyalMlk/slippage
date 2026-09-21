"""Intraday volume profiles and the benchmark schedules built on them.

Volume is not spread evenly through a session. For most equities it is heavy
at the open, thins out through the middle of the day and rises into the close.
A VWAP schedule needs an estimate of that shape: trading in proportion to
*expected* volume is what keeps an execution close to the interval VWAP.

Profiles are estimated as the average of each day's volume *fractions*, not as
the fraction of summed volume. Summing first lets one heavy day, such as an
index rebalance, dominate the shape; averaging fractions gives every day an
equal vote on how volume is distributed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import numpy as np
from numpy.typing import NDArray

from .exceptions import InsufficientDataError, ValidationError
from .series import BarSeries

__all__ = [
    "PovSchedule",
    "VolumeProfile",
    "estimate_profile",
    "pov_schedule",
    "round_to_lots",
    "twap_schedule",
    "vwap_schedule",
]


@dataclass(frozen=True)
class VolumeProfile:
    """Expected share of a session's volume in each of its buckets."""

    fractions: tuple[float, ...]
    bucket: timedelta
    session_open: time
    dispersion: tuple[float, ...] = ()
    """Cross-day standard deviation of each bucket's fraction, when estimated."""
    days: int = 0

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValidationError("a profile needs at least one bucket")
        if any(not math.isfinite(f) or f < 0.0 for f in self.fractions):
            raise ValidationError("profile fractions must be non-negative")
        total = sum(self.fractions)
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValidationError(f"profile fractions sum to {total}, not 1")
        if self.bucket <= timedelta(0):
            raise ValidationError("bucket length must be positive")

    @classmethod
    def uniform(cls, buckets: int, bucket: timedelta, session_open: time) -> VolumeProfile:
        if buckets < 1:
            raise ValidationError(f"buckets must be at least 1, got {buckets}")
        return cls(fractions=(1.0 / buckets,) * buckets, bucket=bucket, session_open=session_open)

    def __len__(self) -> int:
        return len(self.fractions)

    def expected_volume(self, daily_volume: float) -> list[float]:
        """Expected shares printed in each bucket on a day of ``daily_volume``."""
        if not math.isfinite(daily_volume) or daily_volume < 0.0:
            raise ValidationError(f"daily volume must be non-negative, got {daily_volume!r}")
        return [f * daily_volume for f in self.fractions]

    def cumulative(self) -> list[float]:
        """Fraction of the session's volume expected by the end of each bucket."""
        return list(np.cumsum(self.fractions))


def _bucket_index(moment: datetime, session_open: time, bucket: timedelta) -> int:
    start = datetime.combine(moment.date(), session_open, tzinfo=moment.tzinfo)
    return math.floor((moment - start) / bucket)


def estimate_profile(
    days: Iterable[BarSeries],
    *,
    session_open: time,
    bucket: timedelta,
    buckets: int,
) -> VolumeProfile:
    """Estimate a volume profile from historical sessions, one ``BarSeries`` per day.

    Each bar is assigned to the bucket containing its start. Bars before the
    open or after the last bucket are ignored rather than folded into the edge
    buckets, since pre-market and post-close prints are not volume a working
    order can reach. A day with no volume in the session is skipped.
    """
    if buckets < 1:
        raise ValidationError(f"buckets must be at least 1, got {buckets}")
    if bucket <= timedelta(0):
        raise ValidationError("bucket length must be positive")
    rows: list[NDArray[np.float64]] = []
    for series in days:
        counts = np.zeros(buckets)
        for bar in series:
            index = _bucket_index(bar.timestamp, session_open, bucket)
            if 0 <= index < buckets:
                counts[index] += bar.volume
        total = counts.sum()
        if total > 0.0:
            rows.append(counts / total)
    if not rows:
        raise InsufficientDataError("no session with any volume to estimate a profile from")
    matrix = np.vstack(rows)
    mean = matrix.mean(axis=0)
    mean = mean / mean.sum()
    dispersion = matrix.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(buckets)
    return VolumeProfile(
        fractions=tuple(float(f) for f in mean),
        bucket=bucket,
        session_open=session_open,
        dispersion=tuple(float(d) for d in dispersion),
        days=len(rows),
    )


# -- schedules --------------------------------------------------------------


def round_to_lots(amounts: Sequence[float], lot_size: float) -> list[float]:
    """Round a schedule to whole lots while preserving its total exactly.

    Largest-remainder rounding: floor every entry, then hand the leftover lots
    to the entries that lost the most. Rounding each entry independently can
    add or drop whole lots from the order, which a schedule must never do.
    """
    if not math.isfinite(lot_size) or lot_size <= 0.0:
        raise ValidationError(f"lot size must be positive, got {lot_size!r}")
    units = [a / lot_size for a in amounts]
    if any(u < 0.0 or not math.isfinite(u) for u in units):
        raise ValidationError("schedule amounts must be non-negative")
    total = sum(units)
    whole = round(total)
    if abs(total - whole) > 1e-6 * max(1.0, total):
        raise ValidationError(f"the schedule totals {total:g} lots, not a whole number")
    floors = [math.floor(u + 1e-9) for u in units]
    leftover = whole - sum(floors)
    order = sorted(range(len(units)), key=lambda i: units[i] - floors[i], reverse=True)
    for i in order[:leftover]:
        floors[i] += 1
    return [f * lot_size for f in floors]


def twap_schedule(quantity: float, periods: int, *, lot_size: float | None = None) -> list[float]:
    """Equal slices in every period."""
    if not math.isfinite(quantity) or quantity <= 0.0:
        raise ValidationError(f"quantity must be positive, got {quantity!r}")
    if periods < 1:
        raise ValidationError(f"periods must be at least 1, got {periods}")
    schedule = [quantity / periods] * periods
    return schedule if lot_size is None else round_to_lots(schedule, lot_size)


def vwap_schedule(
    quantity: float,
    profile: VolumeProfile | Sequence[float],
    *,
    lot_size: float | None = None,
) -> list[float]:
    """Slices in proportion to expected volume in each period.

    ``profile`` is a :class:`VolumeProfile` or any sequence of non-negative
    weights, which are normalised.
    """
    if not math.isfinite(quantity) or quantity <= 0.0:
        raise ValidationError(f"quantity must be positive, got {quantity!r}")
    weights = list(profile.fractions if isinstance(profile, VolumeProfile) else profile)
    if not weights or any(not math.isfinite(w) or w < 0.0 for w in weights):
        raise ValidationError("profile weights must be non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValidationError("profile weights sum to zero")
    schedule = [quantity * w / total for w in weights]
    return schedule if lot_size is None else round_to_lots(schedule, lot_size)


@dataclass(frozen=True)
class PovSchedule:
    """A percentage-of-volume schedule and whatever it failed to complete."""

    trades: tuple[float, ...]
    unfilled: float

    @property
    def completed(self) -> bool:
        return self.unfilled == 0.0


def pov_schedule(
    quantity: float,
    expected_volume: Sequence[float],
    rate: float,
    *,
    lot_size: float | None = None,
) -> PovSchedule:
    """Take ``rate`` of each period's volume until the order is done.

    Unlike TWAP and VWAP, a percentage-of-volume strategy fixes the pace, not
    the finish: on a quiet day it simply does not complete. The remainder is
    reported rather than forced into the last period, since forcing it would
    break the participation limit the strategy exists to respect.
    """
    if not math.isfinite(quantity) or quantity <= 0.0:
        raise ValidationError(f"quantity must be positive, got {quantity!r}")
    if not 0.0 < rate <= 1.0:
        raise ValidationError(f"rate must be in (0, 1], got {rate!r}")
    remaining = quantity
    trades = []
    for v in expected_volume:
        if not math.isfinite(v) or v < 0.0:
            raise ValidationError(f"expected volume must be non-negative, got {v!r}")
        take = min(rate * v, remaining)
        if lot_size is not None:
            take = math.floor(take / lot_size + 1e-9) * lot_size
        trades.append(take)
        remaining -= take
    if remaining < 1e-9 * quantity:
        remaining = 0.0
    return PovSchedule(trades=tuple(trades), unfilled=remaining)
