from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from slippage.io import InputError, load_bars, load_orders
from slippage.types import Side

ORDERS = """order_id,symbol,side,quantity,decision_time,arrival_time,decision_price
A1,ACME,buy,1000,2026-03-02T09:30:00,2026-03-02T09:31:00,50.00
A2,ACME,S,500,2026-03-02T10:00:00,2026-03-02T10:00:00,
"""

FILLS = """order_id,timestamp,quantity,price,commission
A1,2026-03-02T09:32:00,400,50.05,4
A1,2026-03-02T09:35:00,600,50.08,
A2,2026-03-02T10:05:00,200,49.90,2
"""

BARS = """symbol,timestamp,open,high,low,close,volume
ACME,2026-03-02T09:31:00,50.02,50.06,50.00,50.04,12000
ACME,2026-03-02T09:30:00,50.00,50.03,49.98,50.02,15000
ZETA,2026-03-02T09:30:00,10.00,10.01,9.99,10.00,500
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def load(tmp_path: Path, orders: str = ORDERS, fills: str = FILLS) -> dict[str, object]:
    return dict(load_orders(write(tmp_path, "o.csv", orders), write(tmp_path, "f.csv", fills)))


class TestOrders:
    def test_orders_carry_their_fills(self, tmp_path: Path) -> None:
        orders = load_orders(write(tmp_path, "o.csv", ORDERS), write(tmp_path, "f.csv", FILLS))
        a1, a2 = orders["A1"], orders["A2"]
        assert a1.side is Side.BUY
        assert a1.decision_price == 50.0
        assert a1.filled_quantity == 1_000.0
        assert a1.total_commission == 4.0
        assert a1.arrival_time == datetime(2026, 3, 2, 9, 31)
        assert a2.side is Side.SELL
        assert a2.decision_price is None
        assert a2.unfilled_quantity == 300.0

    def test_missing_column_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match=r"missing column.*arrival_time"):
            load(tmp_path, orders="order_id,symbol,side,quantity,decision_time\n")

    def test_bad_number_names_file_and_line(self, tmp_path: Path) -> None:
        bad = FILLS.replace("50.08", "fifty")
        with pytest.raises(InputError, match=r"f\.csv:3: bad price 'fifty'"):
            load(tmp_path, fills=bad)

    def test_bad_timestamp(self, tmp_path: Path) -> None:
        bad = ORDERS.replace("2026-03-02T09:31:00", "yesterday")
        with pytest.raises(InputError, match=r"o\.csv:2: bad arrival_time"):
            load(tmp_path, orders=bad)

    def test_bad_side(self, tmp_path: Path) -> None:
        bad = ORDERS.replace(",buy,", ",hold,")
        with pytest.raises(InputError, match=r"o\.csv:2: bad side 'hold'"):
            load(tmp_path, orders=bad)

    def test_overfilled_order_is_reported_with_its_id(self, tmp_path: Path) -> None:
        bad = FILLS.replace("A1,2026-03-02T09:35:00,600", "A1,2026-03-02T09:35:00,900")
        with pytest.raises(InputError, match=r"order 'A1'.*exceeds"):
            load(tmp_path, fills=bad)

    def test_negative_fill_is_reported(self, tmp_path: Path) -> None:
        bad = FILLS.replace(",400,", ",-400,")
        with pytest.raises(InputError, match=r"f\.csv:2"):
            load(tmp_path, fills=bad)

    def test_duplicate_order_id(self, tmp_path: Path) -> None:
        dup = ORDERS + "A1,ACME,buy,1,2026-03-02T09:30:00,2026-03-02T09:30:00,50\n"
        with pytest.raises(InputError, match="duplicate order_id 'A1'"):
            load(tmp_path, orders=dup)

    def test_orphan_fills_are_reported(self, tmp_path: Path) -> None:
        orphan = FILLS + "ZZ9,2026-03-02T10:05:00,1,1,0\n"
        with pytest.raises(InputError, match=r"unknown order.*ZZ9"):
            load(tmp_path, fills=orphan)

    def test_whitespace_in_header_and_cells_is_tolerated(self, tmp_path: Path) -> None:
        spaced = ORDERS.replace("order_id,symbol", "order_id , symbol").replace(
            "A2,ACME", " A2 , ACME"
        )
        orders = load(tmp_path, orders=spaced)
        assert set(orders) == {"A1", "A2"}


class TestBars:
    def test_bars_are_grouped_and_sorted(self, tmp_path: Path) -> None:
        series = load_bars(write(tmp_path, "b.csv", BARS))
        assert set(series) == {"ACME", "ZETA"}
        acme = series["ACME"]
        assert [b.timestamp.minute for b in acme] == [30, 31]
        assert acme.total_volume() == 27_000.0

    def test_inconsistent_bar_is_located(self, tmp_path: Path) -> None:
        bad = BARS.replace("50.02,50.06,50.00,50.04", "50.02,49.00,50.00,50.04")
        with pytest.raises(InputError, match=r"b\.csv:2"):
            load_bars(write(tmp_path, "b.csv", bad))

    def test_duplicate_bar_is_reported_by_symbol(self, tmp_path: Path) -> None:
        dup = BARS + "ACME,2026-03-02T09:30:00,50.00,50.03,49.98,50.02,1\n"
        with pytest.raises(InputError, match="ACME: duplicate"):
            load_bars(write(tmp_path, "b.csv", dup))
