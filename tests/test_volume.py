from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from slippage.exceptions import InsufficientDataError, ValidationError
from slippage.series import BarSeries
from slippage.types import Bar
from slippage.volume import (
    VolumeProfile,
    estimate_profile,
    pov_schedule,
    round_to_lots,
    twap_schedule,
    vwap_schedule,
)

OPEN = time(9, 30)
HALF_HOUR = timedelta(minutes=30)

# A U-shaped session in thirteen half-hour buckets: heavy open and close.
U_SHAPE = np.array([12, 8, 6, 5, 4.5, 4, 4, 4, 4.5, 5, 6, 9, 14], dtype=float)
U_SHAPE = U_SHAPE / U_SHAPE.sum()


def session(
    day: date, fractions: NDArray[np.float64], total: float, *, minutes: int = 5
) -> BarSeries:
    """One day of ``minutes``-minute bars whose bucket totals follow ``fractions``."""
    bars = []
    per_bucket = 30 // minutes
    start = datetime.combine(day, OPEN)
    for b, f in enumerate(fractions):
        for i in range(per_bucket):
            ts = start + HALF_HOUR * b + timedelta(minutes=minutes * i)
            bars.append(
                Bar(
                    ts, open=100.0, high=100.1, low=99.9, close=100.0, volume=total * f / per_bucket
                )
            )
    return BarSeries(bars)


class TestProfile:
    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="sum to"):
            VolumeProfile(fractions=(0.5, 0.4), bucket=HALF_HOUR, session_open=OPEN)

    def test_negative_fraction_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VolumeProfile(fractions=(1.5, -0.5), bucket=HALF_HOUR, session_open=OPEN)

    def test_uniform_and_expected_volume(self) -> None:
        profile = VolumeProfile.uniform(4, HALF_HOUR, OPEN)
        assert profile.expected_volume(1e6) == [2.5e5] * 4
        assert profile.cumulative() == pytest.approx([0.25, 0.5, 0.75, 1.0])
        assert len(profile) == 4


class TestEstimation:
    def test_recovers_a_noisy_u_shape(self) -> None:
        rng = np.random.default_rng(1)
        days = []
        for d in range(60):
            noisy = U_SHAPE * rng.lognormal(0.0, 0.2, U_SHAPE.size)
            noisy /= noisy.sum()
            days.append(session(date(2026, 1, 5) + timedelta(days=d), noisy, 5e6))
        profile = estimate_profile(days, session_open=OPEN, bucket=HALF_HOUR, buckets=13)
        assert profile.days == 60
        assert sum(profile.fractions) == pytest.approx(1.0)
        # Each bucket within four standard errors of the truth.
        for estimate, truth, spread in zip(
            profile.fractions, U_SHAPE, profile.dispersion, strict=True
        ):
            assert abs(estimate - truth) < 4 * spread / math.sqrt(60)
        assert profile.fractions[0] > profile.fractions[6] < profile.fractions[-1]

    def test_a_heavy_day_does_not_dominate_the_shape(self) -> None:
        flat = np.full(13, 1 / 13)
        spike = np.zeros(13)
        spike[-1] = 1.0
        days = [session(date(2026, 1, 5) + timedelta(days=d), flat, 1e6) for d in range(9)]
        # One rebalance day with fifty times normal volume, all at the close.
        days.append(session(date(2026, 1, 20), spike, 5e7))
        profile = estimate_profile(days, session_open=OPEN, bucket=HALF_HOUR, buckets=13)
        # Averaging fractions gives the spike day a tenth of the vote. Summing
        # volume first would have put 85% of the profile in the last bucket.
        assert profile.fractions[-1] == pytest.approx(0.9 / 13 + 0.1)
        summed_share = (5e7 + 9 * 1e6 / 13) / (5e7 + 9e6)
        assert summed_share > 0.85

    def test_out_of_session_bars_are_ignored(self) -> None:
        day = session(date(2026, 1, 5), U_SHAPE, 1e6)
        early = Bar(datetime(2026, 1, 5, 8, 0), 100.0, 100.0, 100.0, 100.0, volume=9e9)
        late = Bar(datetime(2026, 1, 5, 17, 0), 100.0, 100.0, 100.0, 100.0, volume=9e9)
        noisy = BarSeries([early, *day, late])
        profile = estimate_profile([noisy], session_open=OPEN, bucket=HALF_HOUR, buckets=13)
        assert profile.fractions == pytest.approx(tuple(U_SHAPE))
        assert profile.dispersion == (0.0,) * 13

    def test_no_volume_raises(self) -> None:
        empty = session(date(2026, 1, 5), np.full(13, 1 / 13), 0.0)
        with pytest.raises(InsufficientDataError):
            estimate_profile([empty], session_open=OPEN, bucket=HALF_HOUR, buckets=13)

    def test_invalid_buckets(self) -> None:
        with pytest.raises(ValidationError):
            estimate_profile([], session_open=OPEN, bucket=HALF_HOUR, buckets=0)
        with pytest.raises(ValidationError):
            estimate_profile([], session_open=OPEN, bucket=timedelta(0), buckets=3)


class TestRounding:
    @given(
        weights=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=30),
        lots=st.integers(min_value=1, max_value=10_000),
    )
    def test_preserves_the_total_and_stays_within_a_lot(
        self, weights: list[float], lots: int
    ) -> None:
        total = sum(weights)
        if total <= 0.0:
            return
        raw = [lots * 100.0 * w / total for w in weights]
        rounded = round_to_lots(raw, 100.0)
        assert sum(rounded) == pytest.approx(lots * 100.0)
        assert all(abs(a - b) < 100.0 for a, b in zip(raw, rounded, strict=True))
        assert all(r / 100.0 == round(r / 100.0) for r in rounded)

    def test_independent_rounding_would_lose_a_lot(self) -> None:
        # Three thirds of 100 lots: rounding each gives 33 + 33 + 33 = 99.
        raw = [100 * 100.0 / 3] * 3
        assert sum(round(r / 100) * 100 for r in raw) == 9_900
        assert sum(round_to_lots(raw, 100.0)) == 10_000.0

    def test_non_whole_total_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="whole number"):
            round_to_lots([150.0], 100.0)


class TestSchedules:
    def test_twap(self) -> None:
        assert twap_schedule(1_000.0, 4) == [250.0] * 4
        assert twap_schedule(1_000.0, 3, lot_size=100.0) == [400.0, 300.0, 300.0]

    def test_vwap_follows_the_profile(self) -> None:
        profile = VolumeProfile(fractions=tuple(U_SHAPE), bucket=HALF_HOUR, session_open=OPEN)
        schedule = vwap_schedule(1e6, profile)
        assert schedule == pytest.approx(list(U_SHAPE * 1e6))
        lots = vwap_schedule(1e6, profile, lot_size=100.0)
        assert sum(lots) == 1e6

    def test_vwap_normalises_raw_weights(self) -> None:
        assert vwap_schedule(90.0, [1.0, 2.0, 6.0]) == pytest.approx([10.0, 20.0, 60.0])

    def test_vwap_rejects_bad_weights(self) -> None:
        with pytest.raises(ValidationError):
            vwap_schedule(1.0, [0.0, 0.0])
        with pytest.raises(ValidationError):
            vwap_schedule(1.0, [1.0, -1.0])

    def test_pov_completes_when_volume_allows(self) -> None:
        result = pov_schedule(30_000.0, [100_000.0, 100_000.0, 100_000.0], 0.2)
        assert result.trades == (20_000.0, 10_000.0, 0.0)
        assert result.completed

    def test_pov_reports_what_it_could_not_do(self) -> None:
        result = pov_schedule(100_000.0, [100_000.0, 50_000.0], 0.1, lot_size=1_000.0)
        assert result.trades == (10_000.0, 5_000.0)
        assert result.unfilled == pytest.approx(85_000.0)
        assert not result.completed

    @pytest.mark.parametrize("rate", [0.0, 1.2])
    def test_pov_rate_bounds(self, rate: float) -> None:
        with pytest.raises(ValidationError):
            pov_schedule(1.0, [1.0], rate)
