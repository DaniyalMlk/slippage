"""Transaction cost analysis across a book of orders.

Aggregation rule
----------------

Every aggregate is a ratio of sums, never an average of ratios: a group's cost in
basis points is its total currency cost divided by its total paper notional.
Averaging per-order basis points instead would give a 100-share order the same
vote as a million-share one, and — the practical problem — the averaged
components would no longer add up to the averaged total once orders of
different size are mixed. With ratios of sums they add up at every level:
order, group and book.

Outliers
--------

Outliers are flagged by the modified z-score of Iglewicz and Hoaglin (1993),
``0.6745 (x - median) / MAD``, with their suggested threshold of 3.5. A
z-score built on the mean and standard deviation is a poor way to find
outliers, because the outliers inflate the standard deviation and hide
themselves; the median and MAD barely move.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .benchmarks import Benchmark, score_order
from .costs import BPS_PER_UNIT
from .exceptions import NoVolumeError, ValidationError
from .impact import SquareRootLaw
from .series import BarSeries
from .shortfall import DelayBasis, ShortfallBreakdown, participation_rate, shortfall_from_market
from .types import Order, Side

__all__ = [
    "Aggregate",
    "ModelComparison",
    "OrderReport",
    "TcaReport",
    "build_report",
    "compare_to_model",
    "modified_z_scores",
]

OUTLIER_THRESHOLD = 3.5
"""Iglewicz and Hoaglin's recommended cut-off for the modified z-score."""

COMPONENTS = ("delay", "trading", "opportunity", "commission", "fees")


@dataclass(frozen=True)
class OrderReport:
    """One order's shortfall and how it traded."""

    order_id: str
    symbol: str
    side: Side
    shortfall: ShortfallBreakdown
    participation: float
    vwap_bps: float | None
    """Cost of the filled part against interval VWAP; ``None`` when unfilled."""

    @property
    def total_bps(self) -> float:
        return self.shortfall.total_bps

    def to_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.shortfall.target_quantity,
            "filled": self.shortfall.filled_quantity,
            "paper_notional": self.shortfall.paper_notional,
            "components": self.shortfall.components(),
            "components_bps": self.shortfall.components_bps(),
            "total": self.shortfall.total,
            "total_bps": self.total_bps,
            "participation": self.participation,
            "vwap_bps": self.vwap_bps,
        }


@dataclass(frozen=True)
class Aggregate:
    """Summed costs over a set of orders."""

    orders: int
    paper_notional: float
    components: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.components.values())

    def bps(self) -> dict[str, float]:
        """Each component as basis points of the group's paper notional."""
        return {k: BPS_PER_UNIT * v / self.paper_notional for k, v in self.components.items()}

    @property
    def total_bps(self) -> float:
        return BPS_PER_UNIT * self.total / self.paper_notional

    def to_dict(self) -> dict[str, object]:
        return {
            "orders": self.orders,
            "paper_notional": self.paper_notional,
            "components": self.components,
            "components_bps": self.bps(),
            "total": self.total,
            "total_bps": self.total_bps,
        }


def _aggregate(rows: Iterable[OrderReport]) -> Aggregate:
    rows = list(rows)
    if not rows:
        raise ValidationError("cannot aggregate an empty set of orders")
    components = dict.fromkeys(COMPONENTS, 0.0)
    notional = 0.0
    for row in rows:
        notional += row.shortfall.paper_notional
        for name, value in row.shortfall.components().items():
            components[name] += value
    return Aggregate(orders=len(rows), paper_notional=notional, components=components)


def modified_z_scores(values: Iterable[float]) -> list[float]:
    """Iglewicz-Hoaglin modified z-scores.

    Falls back to the mean absolute deviation, scaled to match, when more than
    half the values are identical and the MAD is zero; returns zeros when every
    value is the same.
    """
    data = list(values)
    if not data:
        return []
    median = statistics.median(data)
    deviations = [abs(x - median) for x in data]
    mad = statistics.median(deviations)
    if mad > 0.0:
        return [0.6745 * (x - median) / mad for x in data]
    mean_ad = statistics.fmean(deviations)
    if mean_ad > 0.0:
        return [(x - median) / (1.253314 * mean_ad) for x in data]
    return [0.0] * len(data)


@dataclass(frozen=True)
class TcaReport:
    """Shortfall for every order, with aggregation and outlier screening."""

    orders: tuple[OrderReport, ...]

    def __len__(self) -> int:
        return len(self.orders)

    def total(self) -> Aggregate:
        return _aggregate(self.orders)

    def group_by(self, key: Callable[[OrderReport], str]) -> dict[str, Aggregate]:
        groups: dict[str, list[OrderReport]] = {}
        for row in self.orders:
            groups.setdefault(key(row), []).append(row)
        return {name: _aggregate(rows) for name, rows in sorted(groups.items())}

    def outliers(self, threshold: float = OUTLIER_THRESHOLD) -> list[tuple[OrderReport, float]]:
        """Orders whose total cost is anomalous, worst first, with their scores."""
        scores = modified_z_scores(row.total_bps for row in self.orders)
        flagged = [
            (row, z) for row, z in zip(self.orders, scores, strict=True) if abs(z) > threshold
        ]
        return sorted(flagged, key=lambda pair: -abs(pair[1]))

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total().to_dict(),
            "by_symbol": {k: v.to_dict() for k, v in self.group_by(lambda r: r.symbol).items()},
            "by_side": {k: v.to_dict() for k, v in self.group_by(lambda r: r.side.value).items()},
            "orders": [row.to_dict() for row in self.orders],
            "outliers": [{"order_id": row.order_id, "modified_z": z} for row, z in self.outliers()],
        }


def build_report(
    orders: Mapping[str, Order],
    bars: Mapping[str, BarSeries],
    *,
    fees_per_share: float = 0.0,
    delay_basis: DelayBasis = DelayBasis.ORDER,
) -> TcaReport:
    """Decompose every order's shortfall against its symbol's bars.

    ``fees_per_share`` is charged on executed shares as an explicit cost on top
    of each fill's commission. Unfilled remainders are marked at the last close
    in the symbol's series.
    """
    if not math.isfinite(fees_per_share) or fees_per_share < 0.0:
        raise ValidationError(f"fees per share must be non-negative, got {fees_per_share!r}")
    if not orders:
        raise ValidationError("no orders to report on")
    rows = []
    for order_id, order in orders.items():
        series = bars.get(order.symbol)
        if series is None:
            raise ValidationError(f"order {order_id!r}: no bars for {order.symbol!r}")
        shortfall = shortfall_from_market(
            order,
            series,
            fees=fees_per_share * order.filled_quantity,
            delay_basis=delay_basis,
        )
        vwap_bps: float | None = None
        if order.fills:
            try:
                vwap_bps = score_order(order, series, Benchmark.INTERVAL_VWAP).cost_bps
            except NoVolumeError:
                vwap_bps = None
        try:
            participation = participation_rate(order, series)
        except ValidationError:
            participation = math.nan
        rows.append(
            OrderReport(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                shortfall=shortfall,
                participation=participation,
                vwap_bps=vwap_bps,
            )
        )
    return TcaReport(orders=tuple(rows))


@dataclass(frozen=True)
class ModelComparison:
    """Realised trading cost against what an impact model expected."""

    order_id: str
    realised_bps: float
    expected_bps: float

    @property
    def excess_bps(self) -> float:
        return self.realised_bps - self.expected_bps


def compare_to_model(
    report: TcaReport,
    law: SquareRootLaw,
    daily_volume: Mapping[str, float],
    daily_volatility: Mapping[str, float],
) -> list[ModelComparison]:
    """Compare each filled order's trading cost with the square-root law.

    Both sides are per executed share relative to arrival, so an order that
    was cut short is judged on what it did trade, not penalised for size it
    never took on. Unfilled orders are skipped.
    """
    out = []
    for row in report.orders:
        s = row.shortfall
        if s.filled_quantity == 0.0:
            continue
        try:
            volume = daily_volume[row.symbol]
            vol = daily_volatility[row.symbol]
        except KeyError as missing:
            raise ValidationError(
                f"no daily volume or volatility for {missing.args[0]!r}"
            ) from None
        realised = BPS_PER_UNIT * s.trading / (s.filled_quantity * s.arrival_price)
        expected = law.expected_cost_bps(s.filled_quantity, volume, vol)
        out.append(ModelComparison(row.order_id, realised, expected))
    return out
