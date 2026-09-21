"""A sorted series of OHLCV bars with windowed aggregates.

Windowing convention
--------------------

Every window is half-open on bar *start* times: a bar belongs to
``[start, end)`` when ``start <= bar.timestamp < end``. Bars are not split, so
a window that begins mid-bar excludes that bar entirely rather than
interpolating a fraction of its volume. Interpolating would invent a price path
inside the bar that the data does not contain; excluding it is wrong in a way
the caller can see and correct by passing a coarser window.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta

from .exceptions import InsufficientDataError, NoVolumeError, ValidationError
from .types import Bar

__all__ = ["BarSeries"]


class BarSeries(Sequence[Bar]):
    """An immutable, strictly time-ordered collection of bars."""

    __slots__ = ("_bars", "_starts", "_durations")

    def __init__(self, bars: Iterable[Bar]) -> None:
        ordered = tuple(sorted(bars, key=lambda b: b.timestamp))
        if not ordered:
            raise InsufficientDataError("a bar series needs at least one bar")
        starts = [b.timestamp for b in ordered]
        for previous, current in zip(starts, starts[1:], strict=False):
            if previous == current:
                raise ValidationError(f"duplicate bar timestamp {current!r}")
        self._bars = ordered
        self._starts = starts
        self._durations = self._infer_durations(starts)

    @staticmethod
    def _infer_durations(starts: list[datetime]) -> list[timedelta]:
        """Each bar spans until the next one starts.

        The final bar has no successor, so it takes the median of the observed
        gaps. A single-bar series gets a nominal one-minute span, which never
        affects a result because every weight in a one-bar average cancels.
        """
        if len(starts) == 1:
            return [timedelta(minutes=1)]
        gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
        ordered = sorted(gaps)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2 == 1
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return [*gaps, median]

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def __getitem__(self, index: int) -> Bar:  # type: ignore[override]
        return self._bars[index]

    def __repr__(self) -> str:
        return (
            f"BarSeries({len(self._bars)} bars, "
            f"{self._bars[0].timestamp.isoformat()} to {self.end.isoformat()})"
        )

    # -- span ---------------------------------------------------------------

    @property
    def start(self) -> datetime:
        """Start of the first bar."""
        return self._bars[0].timestamp

    @property
    def end(self) -> datetime:
        """End of the last bar, using its inferred duration."""
        return self._bars[-1].timestamp + self._durations[-1]

    @property
    def bar_duration(self) -> timedelta:
        """The median bar span, used wherever a nominal granularity is needed."""
        return self._durations[-1]

    # -- windowing ----------------------------------------------------------

    def _index_range(self, start: datetime | None, end: datetime | None) -> tuple[int, int]:
        lo = 0 if start is None else bisect.bisect_left(self._starts, start)
        hi = len(self._bars) if end is None else bisect.bisect_left(self._starts, end)
        return lo, max(lo, hi)

    def window(self, start: datetime | None = None, end: datetime | None = None) -> BarSeries:
        """Bars whose start lies in ``[start, end)``."""
        lo, hi = self._index_range(start, end)
        if lo == hi:
            raise InsufficientDataError(
                f"no bars in window [{start!r}, {end!r}); series spans "
                f"[{self.start.isoformat()}, {self.end.isoformat()})"
            )
        return BarSeries(self._bars[lo:hi])

    def bar_containing(self, moment: datetime) -> Bar:
        """The bar in progress at ``moment``.

        Clamped to the ends of the series: a moment before the first bar
        returns the first bar, and a moment after the last returns the last.
        """
        index = bisect.bisect_right(self._starts, moment) - 1
        return self._bars[max(index, 0)]

    def price_at(self, moment: datetime) -> float:
        """The price prevailing at ``moment``, to bar resolution.

        The open of the bar in progress. Sub-bar precision is not available, so
        this is the most recent price the data actually asserts. After the end
        of the series it is the final close.
        """
        if moment >= self.end:
            return self._bars[-1].close
        if moment <= self.start:
            return self._bars[0].open
        return self.bar_containing(moment).open

    # -- aggregates ---------------------------------------------------------

    def total_volume(self, start: datetime | None = None, end: datetime | None = None) -> float:
        lo, hi = self._index_range(start, end)
        return sum(b.volume for b in self._bars[lo:hi])

    def vwap(self, start: datetime | None = None, end: datetime | None = None) -> float:
        """Volume-weighted average of bar typical prices over the window."""
        lo, hi = self._index_range(start, end)
        if lo == hi:
            raise InsufficientDataError(f"no bars in window [{start!r}, {end!r})")
        volume = 0.0
        notional = 0.0
        for bar in self._bars[lo:hi]:
            volume += bar.volume
            notional += bar.notional
        if volume <= 0.0:
            raise NoVolumeError(
                f"window [{start!r}, {end!r}) traded no volume, so VWAP is undefined"
            )
        return notional / volume

    def twap(self, start: datetime | None = None, end: datetime | None = None) -> float:
        """Time-weighted average of bar typical prices over the window.

        Weighted by each bar's span, so a series with a lunchtime gap or a
        mixture of granularities does not over-weight the sparse stretch. With
        uniform bars the weights cancel and this is the plain mean.
        """
        lo, hi = self._index_range(start, end)
        if lo == hi:
            raise InsufficientDataError(f"no bars in window [{start!r}, {end!r})")
        weight = 0.0
        total = 0.0
        for index in range(lo, hi):
            seconds = self._durations[index].total_seconds()
            weight += seconds
            total += self._bars[index].typical_price * seconds
        return total / weight

    def close_price(self, start: datetime | None = None, end: datetime | None = None) -> float:
        """Close of the last bar in the window."""
        lo, hi = self._index_range(start, end)
        if lo == hi:
            raise InsufficientDataError(f"no bars in window [{start!r}, {end!r})")
        return self._bars[hi - 1].close

    def open_price(self, start: datetime | None = None, end: datetime | None = None) -> float:
        """Open of the first bar in the window."""
        lo, hi = self._index_range(start, end)
        if lo == hi:
            raise InsufficientDataError(f"no bars in window [{start!r}, {end!r})")
        return self._bars[lo].open
