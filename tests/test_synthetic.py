from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from slippage.synthetic import SyntheticBook, synthetic_book


@pytest.fixture(scope="module")
def book() -> SyntheticBook:
    return synthetic_book(np.random.default_rng(11), symbols=6, orders=80)


class TestSyntheticBook:
    def test_book_is_well_formed(self, book: SyntheticBook) -> None:
        for order in book.orders.values():
            series = book.bars[order.symbol]
            assert series.start <= order.decision_time <= order.arrival_time
            assert order.filled_quantity <= order.quantity
            for fill in order.fills:
                assert order.arrival_time <= fill.timestamp < series.end
        cut_short = sum(1 for o in book.orders.values() if not o.is_complete)
        assert 0 < cut_short < len(book.orders) // 3

    def test_is_reproducible(self) -> None:
        a = synthetic_book(np.random.default_rng(3), orders=5)
        b = synthetic_book(np.random.default_rng(3), orders=5)
        assert a.orders == b.orders

    def test_fills_follow_the_arrival_order(self) -> None:
        book = synthetic_book(np.random.default_rng(9), orders=20)
        for order in book.orders.values():
            times = [f.timestamp for f in order.fills]
            assert times == sorted(times)
            if times:
                assert times[-1] - order.arrival_time <= timedelta(minutes=245)
