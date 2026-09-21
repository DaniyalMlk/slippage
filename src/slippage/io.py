"""Loading orders, fills and bars from CSV.

The formats are deliberately plain, one row per record with a header line:

``orders.csv``
    ``order_id, symbol, side, quantity, decision_time, arrival_time`` and an
    optional ``decision_price``.
``fills.csv``
    ``order_id, timestamp, quantity, price`` and an optional ``commission``.
``bars.csv``
    ``symbol, timestamp, open, high, low, close, volume``.

Timestamps are ISO 8601. Every error names the file and line it came from: a
TCA run over ten thousand fills that fails with "could not convert string to
float" and no location is a run nobody can fix.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from .exceptions import SlippageError, ValidationError
from .series import BarSeries
from .types import Bar, Fill, Order, Side

__all__ = ["InputError", "load_bars", "load_orders", "read_rows"]

T = TypeVar("T")

ORDER_COLUMNS = ("order_id", "symbol", "side", "quantity", "decision_time", "arrival_time")
FILL_COLUMNS = ("order_id", "timestamp", "quantity", "price")
BAR_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")


class InputError(SlippageError):
    """A row in an input file could not be read, with its location."""


def read_rows(path: str | Path, required: tuple[str, ...]) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield ``(line_number, row)`` pairs, checking the header first."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise InputError(f"{source}: missing column(s) {', '.join(missing)}")
        reader.fieldnames = header
        for row in reader:
            yield reader.line_num, {k: (v or "").strip() for k, v in row.items() if k}


def _field(source: Path, line: int, row: dict[str, str], name: str, parse: Callable[[str], T]) -> T:
    raw = row.get(name, "")
    try:
        return parse(raw)
    except (ValueError, ValidationError) as error:
        raise InputError(f"{source}:{line}: bad {name} {raw!r}: {error}") from None


def _optional_float(raw: str) -> float | None:
    return float(raw) if raw else None


def load_orders(orders_path: str | Path, fills_path: str | Path) -> dict[str, Order]:
    """Load orders and attach their fills, keyed by ``order_id``."""
    orders_file = Path(orders_path)
    fills_file = Path(fills_path)

    fills: defaultdict[str, list[Fill]] = defaultdict(list)
    for line, row in read_rows(fills_file, FILL_COLUMNS):
        try:
            fill = Fill(
                timestamp=_field(fills_file, line, row, "timestamp", datetime.fromisoformat),
                quantity=_field(fills_file, line, row, "quantity", float),
                price=_field(fills_file, line, row, "price", float),
                commission=_field(
                    fills_file, line, row, "commission", lambda s: float(s) if s else 0.0
                ),
                venue=row.get("venue") or None,
            )
        except ValidationError as error:
            raise InputError(f"{fills_file}:{line}: {error}") from None
        fills[row["order_id"]].append(fill)

    orders: dict[str, Order] = {}
    for line, row in read_rows(orders_file, ORDER_COLUMNS):
        order_id = row["order_id"]
        if not order_id:
            raise InputError(f"{orders_file}:{line}: empty order_id")
        if order_id in orders:
            raise InputError(f"{orders_file}:{line}: duplicate order_id {order_id!r}")
        try:
            orders[order_id] = Order(
                symbol=row["symbol"],
                side=_field(orders_file, line, row, "side", Side.parse),
                quantity=_field(orders_file, line, row, "quantity", float),
                decision_time=_field(
                    orders_file, line, row, "decision_time", datetime.fromisoformat
                ),
                arrival_time=_field(orders_file, line, row, "arrival_time", datetime.fromisoformat),
                fills=tuple(fills.pop(order_id, [])),
                decision_price=_field(orders_file, line, row, "decision_price", _optional_float),
            )
        except ValidationError as error:
            raise InputError(f"{orders_file}:{line}: order {order_id!r}: {error}") from None

    if fills:
        orphans = ", ".join(sorted(fills)[:5])
        raise InputError(f"{fills_file}: fills reference unknown order(s): {orphans}")
    return orders


def load_bars(path: str | Path) -> dict[str, BarSeries]:
    """Load bars grouped into one series per symbol."""
    source = Path(path)
    grouped: defaultdict[str, list[Bar]] = defaultdict(list)
    for line, row in read_rows(source, BAR_COLUMNS):
        try:
            bar = Bar(
                timestamp=_field(source, line, row, "timestamp", datetime.fromisoformat),
                open=_field(source, line, row, "open", float),
                high=_field(source, line, row, "high", float),
                low=_field(source, line, row, "low", float),
                close=_field(source, line, row, "close", float),
                volume=_field(source, line, row, "volume", float),
            )
        except ValidationError as error:
            raise InputError(f"{source}:{line}: {error}") from None
        grouped[row["symbol"]].append(bar)
    series: dict[str, BarSeries] = {}
    for symbol, bars in grouped.items():
        try:
            series[symbol] = BarSeries(bars)
        except ValidationError as error:
            raise InputError(f"{source}: {symbol}: {error}") from None
    return series
