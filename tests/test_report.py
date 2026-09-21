from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from slippage.calibration import fit_power_law
from slippage.exceptions import IdentifiabilityWarning, ValidationError
from slippage.impact import SquareRootLaw
from slippage.report import build_report, compare_to_model, modified_z_scores
from slippage.synthetic import SyntheticBook, synthetic_book
from slippage.types import Fill, Order


@pytest.fixture(scope="module")
def book() -> SyntheticBook:
    return synthetic_book(np.random.default_rng(11), symbols=6, orders=80)


def component_sum(values: dict[str, float]) -> float:
    return sum(values.values())


class TestAggregation:
    def test_every_order_has_a_row(self, book: SyntheticBook) -> None:
        report = build_report(book.orders, book.bars)
        assert len(report) == 80
        assert {row.order_id for row in report.orders} == set(book.orders)

    def test_bps_add_up_at_every_level(self, book: SyntheticBook) -> None:
        report = build_report(book.orders, book.bars, fees_per_share=0.001)
        for row in report.orders:
            assert component_sum(row.shortfall.components_bps()) == pytest.approx(row.total_bps)
        total = report.total()
        assert component_sum(total.bps()) == pytest.approx(total.total_bps)
        for group in report.group_by(lambda r: r.symbol).values():
            assert component_sum(group.bps()) == pytest.approx(group.total_bps)

    def test_groups_partition_the_book(self, book: SyntheticBook) -> None:
        report = build_report(book.orders, book.bars)
        total = report.total()
        groups = report.group_by(lambda r: r.side.value)
        assert sum(g.orders for g in groups.values()) == total.orders
        assert sum(g.total for g in groups.values()) == pytest.approx(total.total)
        assert sum(g.paper_notional for g in groups.values()) == pytest.approx(total.paper_notional)

    def test_book_bps_is_a_ratio_of_sums_not_an_average(self, book: SyntheticBook) -> None:
        report = build_report(book.orders, book.bars)
        total = report.total()
        weighted = sum(r.total_bps * r.shortfall.paper_notional for r in report.orders) / sum(
            r.shortfall.paper_notional for r in report.orders
        )
        naive = sum(r.total_bps for r in report.orders) / len(report)
        assert total.total_bps == pytest.approx(weighted)
        assert not math.isclose(total.total_bps, naive, rel_tol=1e-3)

    def test_fees_are_charged_per_executed_share(self, book: SyntheticBook) -> None:
        report = build_report(book.orders, book.bars, fees_per_share=0.01)
        for row in report.orders:
            assert row.shortfall.fees == pytest.approx(0.01 * row.shortfall.filled_quantity)

    def test_report_serialises_to_json(self, book: SyntheticBook) -> None:
        payload = json.loads(json.dumps(build_report(book.orders, book.bars).to_dict()))
        assert payload["total"]["orders"] == 80
        assert set(payload["by_side"]) <= {"buy", "sell"}
        assert len(payload["orders"]) == 80

    def test_missing_bars_are_named(self, book: SyntheticBook) -> None:
        with pytest.raises(ValidationError, match="no bars for"):
            build_report(book.orders, {})

    def test_empty_book_and_bad_fees(self, book: SyntheticBook) -> None:
        with pytest.raises(ValidationError, match="no orders"):
            build_report({}, book.bars)
        with pytest.raises(ValidationError, match="fees"):
            build_report(book.orders, book.bars, fees_per_share=-1.0)


class TestOutliers:
    def test_modified_z_scores(self) -> None:
        scores = modified_z_scores([1.0, 2.0, 3.0, 4.0, 100.0])
        # median 3, MAD 1: 0.6745 * 97 / 1.
        assert scores[-1] == pytest.approx(0.6745 * 97)
        assert scores[2] == 0.0

    def test_zero_mad_falls_back_to_mean_deviation(self) -> None:
        scores = modified_z_scores([5.0, 5.0, 5.0, 5.0, 9.0])
        assert scores[-1] == pytest.approx(4.0 / (1.253314 * 0.8))
        assert modified_z_scores([2.0, 2.0]) == [0.0, 0.0]
        assert modified_z_scores([]) == []

    def test_an_injected_bad_execution_is_the_one_flagged(self, book: SyntheticBook) -> None:
        orders = dict(book.orders)
        victim_id = next(i for i, o in orders.items() if o.is_complete and len(o.fills) >= 2)
        victim = orders[victim_id]
        # Fill every share 3% worse than it actually traded.
        worse = tuple(
            Fill(
                timestamp=f.timestamp,
                quantity=f.quantity,
                price=f.price * (1 + 0.03 * victim.side.sign),
                commission=f.commission,
            )
            for f in victim.fills
        )
        orders[victim_id] = victim.with_fills(worse)
        report = build_report(orders, book.bars)
        flagged = [row.order_id for row, _ in report.outliers()]
        assert flagged[0] == victim_id

    def test_mean_based_z_would_let_outliers_hide(self) -> None:
        # Two huge outliers inflate the standard deviation enough that neither
        # clears 3 on a classical z-score; the modified score catches both.
        values = [0.0, 1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.8, -0.8, 50.0, 55.0]
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        assert all(abs(v - mean) / sd < 3.0 for v in values)
        modified = modified_z_scores(values)
        assert modified[-1] > 3.5 and modified[-2] > 3.5


class TestModelComparison:
    def test_realised_cost_tracks_the_generating_law(self, book: SyntheticBook) -> None:
        report = build_report(book.orders, book.bars)
        law = SquareRootLaw(y=book.impact_y)
        rows = compare_to_model(report, law, book.daily_volume, book.daily_volatility)
        assert len(rows) == sum(1 for o in book.orders.values() if o.fills)
        # Realised cost includes a 2 bp half spread and market noise on top of
        # the law, so the median excess should sit near +2 bps.
        excess = np.median([r.excess_bps for r in rows])
        assert 0.0 < excess < 6.0

    def test_calibration_recovers_the_law_from_the_report(self) -> None:
        big = synthetic_book(np.random.default_rng(5), symbols=10, orders=600)
        report = build_report(big.orders, big.bars)
        rows = compare_to_model(report, SquareRootLaw(), big.daily_volume, big.daily_volatility)
        by_id = {r.order_id: r for r in rows}
        x, c, s = [], [], []
        for row in report.orders:
            if row.order_id not in by_id:
                continue
            order = big.orders[row.order_id]
            done = np.cumsum([f.quantity for f in order.fills])
            # Each fill paid impact on the size done so far, so an order's
            # average impact is its peak times this path factor. Folding it
            # into the regressor makes the fitted prefactor the generator's.
            path = float(np.sum(np.diff(done, prepend=0.0) * np.sqrt(done / done[-1])) / done[-1])
            x.append(row.shortfall.filled_quantity / big.daily_volume[row.symbol])
            # Remove the known half spread so only impact is fitted.
            c.append(by_id[row.order_id].realised_bps / 1e4 - 0.0002)
            s.append(big.daily_volatility[row.symbol] * path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IdentifiabilityWarning)
            fit = fit_power_law(x, c, s, delta=0.5)
        assert fit.y.covers(big.impact_y, z=3.0)

    def test_missing_market_data(self, book: SyntheticBook) -> None:
        report = build_report(book.orders, book.bars)
        with pytest.raises(ValidationError, match="no daily volume"):
            compare_to_model(report, SquareRootLaw(), {}, {})


def test_unfilled_orders_are_reported_without_vwap() -> None:
    book = synthetic_book(np.random.default_rng(2), orders=3)
    order = next(iter(book.orders.values()))
    empty: Order = order.with_fills(())
    report = build_report({"X": empty}, book.bars)
    row = report.orders[0]
    assert row.vwap_bps is None
    assert row.participation == 0.0
    assert row.shortfall.trading == 0.0
