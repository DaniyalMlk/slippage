"""The ``slippage`` command.

Four subcommands:

``tca``
    Implementation shortfall report over a book of orders read from CSV.
``schedule``
    An execution schedule: the Almgren-Chriss closed form for linear impact
    without constraints, the dynamic programme otherwise.
``frontier``
    Expected cost against risk across a range of risk aversions.
``placement``
    Rest a limit order or cross the spread, priced.
``sample-data``
    Write a synthetic book to CSV, to try ``tca`` without real data.

Every subcommand takes ``--json`` for machine-readable output. Errors in the
inputs exit with status 2 and a one-line message naming the problem, rather
than a traceback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Sequence
from datetime import time as dt_time
from datetime import timedelta
from pathlib import Path
from typing import TextIO

import numpy as np

from . import __version__
from .adaptive import (
    AdaptiveProblem,
    LiquidityRegime,
    RegimeChain,
    adaptivity_gain,
    simulate_policy,
    solve_adaptive,
    static_schedule,
)
from .basket import (
    BasketProblem,
    basket_trajectory,
    compare_to_independent,
    hedge_direction,
)
from .exceptions import SlippageError, ValidationError
from .execution import ExecutionProblem, efficient_frontier, optimal_trajectory
from .impact import ImpactModel, LinearImpact, PowerLawImpact
from .io import load_bars, load_orders, write_bars, write_orders
from .placement import (
    Outcome,
    Placement,
    evaluate,
    monitoring_shift,
    simulate_placement,
)
from .report import COMPONENTS, TcaReport, build_report
from .reversion import DEFAULT_HORIZONS, reversion_profile
from .scheduling import schedule_objective, solve_schedule
from .series import BarSeries
from .shortfall import DelayBasis
from .synthetic import synthetic_book
from .tracking import (
    TrackingFrontierPoint,
    TrackingProblem,
    compare_objectives,
    fit_volume_covariance,
    tracking_frontier,
    tracking_moments,
)
from .transient import (
    ExponentialDecay,
    PowerLawDecay,
    optimal_transient_schedule,
    residual_impact,
)
from .volume import VolumeProfile

__all__ = ["main"]


def _float_list(text: str) -> list[float]:
    try:
        return [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}"
        ) from None


def _add_problem_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("problem")
    group.add_argument("--quantity", type=float, required=True, help="shares to execute")
    group.add_argument("--horizon", type=float, required=True, help="time available")
    group.add_argument("--periods", type=int, required=True, help="number of intervals")
    group.add_argument(
        "--volatility",
        type=float,
        required=True,
        help="price volatility per square root of the horizon's time unit, in price units",
    )
    impact = parser.add_argument_group("impact, in the same time unit as the horizon")
    impact.add_argument("--gamma", type=float, default=0.0, help="permanent impact per share")
    impact.add_argument("--eta", type=float, required=True, help="temporary impact coefficient")
    impact.add_argument("--epsilon", type=float, default=0.0, help="fixed cost per share")
    impact.add_argument(
        "--beta", type=float, default=None, help="temporary impact exponent (default: linear)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slippage", description="Transaction cost analysis and optimal execution."
    )
    # The version is read from the package rather than from installed metadata,
    # so that the answer is the same whether the copy in reach was installed
    # from a wheel, from an sdist, editable, or not installed at all.
    parser.add_argument("--version", action="version", version=f"slippage {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    tca = sub.add_parser("tca", help="implementation shortfall report over a book of orders")
    tca.add_argument("--orders", type=Path, required=True)
    tca.add_argument("--fills", type=Path, required=True)
    tca.add_argument("--bars", type=Path, required=True)
    tca.add_argument("--fees-per-share", type=float, default=0.0)
    tca.add_argument(
        "--delay-basis", choices=[b.value for b in DelayBasis], default=DelayBasis.ORDER.value
    )
    tca.add_argument("--group-by", choices=["symbol", "side"], default="symbol")
    tca.add_argument("--outlier-threshold", type=float, default=3.5)
    tca.add_argument("--json", action="store_true")

    schedule = sub.add_parser("schedule", help="compute an execution schedule")
    _add_problem_arguments(schedule)
    schedule.add_argument("--risk-aversion", type=float, required=True)
    schedule.add_argument("--lot-size", type=float, default=None)
    schedule.add_argument("--max-trade", type=_float_list, default=None)
    schedule.add_argument("--min-trade", type=_float_list, default=None)
    schedule.add_argument("--json", action="store_true")

    frontier = sub.add_parser("frontier", help="expected cost against risk")
    _add_problem_arguments(frontier)
    frontier.add_argument(
        "--risk-aversions", type=_float_list, default=[0.0, 1e-8, 1e-7, 1e-6, 1e-5]
    )
    frontier.add_argument("--json", action="store_true")

    marks = sub.add_parser(
        "markouts",
        help="post-trade mark-outs: which part of the impact came back",
        description=(
            "Splits the price move over each order's window into the part that "
            "persisted and the part that reverted. Pass --benchmark: at these "
            "horizons the market's move is larger than the impact being measured."
        ),
    )
    marks.add_argument("--orders", type=Path, required=True)
    marks.add_argument("--fills", type=Path, required=True)
    marks.add_argument("--bars", type=Path, required=True)
    marks.add_argument(
        "--horizons",
        type=_horizon_list,
        default=list(DEFAULT_HORIZONS),
        metavar="MINUTES",
        help="comma-separated minutes after completion. Default 1,5,15,30,60.",
    )
    marks.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="bars for an index, whose move is subtracted from every mark-out",
    )
    marks.add_argument(
        "--benchmark-symbol",
        default=None,
        help="which symbol in --benchmark to use, if it holds more than one",
    )
    marks.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="how much of the benchmark's move to subtract. Defaults to 1.",
    )
    marks.add_argument("--json", action="store_true")

    liquidate = sub.add_parser(
        "basket",
        help="liquidate a basket, with the legs solved together",
        description=(
            "Reads a CSV of holdings and a covariance matrix and solves the "
            "liquidation jointly, which is not the same as solving each leg on its "
            "own: a hedged pair carries almost no risk while both legs are on, so "
            "the joint schedule keeps them on together. Reports the eigen-"
            "directions the basket is really liquidated along, and the same "
            "problem solved leg by leg for comparison."
        ),
    )
    liquidate.add_argument(
        "--holdings",
        type=Path,
        required=True,
        help="CSV with columns symbol,quantity,eta,gamma. Quantities are signed, "
        "positive long. eta and gamma are that leg's own impact coefficients.",
    )
    liquidate.add_argument(
        "--covariance",
        type=Path,
        required=True,
        help="CSV of the covariance matrix, one row per asset in the order the "
        "holdings file gives them, no header. In currency per share squared per "
        "unit of the horizon's clock.",
    )
    liquidate.add_argument(
        "--cross-impact",
        type=Path,
        default=None,
        help="CSV of the full temporary impact matrix, replacing the diagonal "
        "built from the holdings file. Trading one name moves the others, and with "
        "this the schedules couple even when the returns do not.",
    )
    liquidate.add_argument("--horizon", type=float, default=1.0)
    liquidate.add_argument("--periods", type=int, default=20)
    liquidate.add_argument("--risk-aversion", type=float, required=True)
    liquidate.add_argument(
        "--compare",
        action="store_true",
        help="also solve it leg by leg and report what the joint solution saves",
    )
    liquidate.add_argument("--json", action="store_true")

    transient = sub.add_parser(
        "transient",
        help="schedule against a decay kernel rather than a trading rate",
        description=(
            "The rate models charge for the current trading speed, so their "
            "temporary impact is gone the instant the speed is zero. A decay "
            "kernel lets it come back at a rate instead, which is what "
            "`slippage markouts` measures. The cost-minimising schedule is then a "
            "linear solve, and for an exponential kernel it comes out as a block, "
            "a constant rate, and a block."
        ),
    )
    transient.add_argument("--quantity", type=float, required=True)
    transient.add_argument("--horizon", type=float, required=True)
    transient.add_argument("--periods", type=int, required=True)
    transient.add_argument(
        "--gamma", type=float, default=0.0, help="permanent impact, price per share"
    )
    transient.add_argument(
        "--eta", type=float, required=True, help="transient impact, price per share"
    )
    transient.add_argument(
        "--half-life",
        dest="half_life",
        type=float,
        default=None,
        help="decay half-life in the same units as --horizon",
    )
    transient.add_argument(
        "--resilience",
        type=float,
        default=None,
        help="decay rate per unit time; the same parameter as --half-life",
    )
    transient.add_argument(
        "--exponent",
        type=float,
        default=None,
        help="use a power-law kernel with this exponent, scaled by --half-life",
    )
    transient.add_argument("--json", action="store_true")

    adaptive = sub.add_parser(
        "adaptive",
        help="adapt the schedule to the liquidity regime, and price the option",
        description=(
            "Every other schedule here is fixed before the first share trades, "
            "which is right when impact and volatility are constant: nothing is "
            "revealed that the plan should depend on. This one lets liquidity "
            "switch between two regimes on a Markov chain the trader observes "
            "before trading into, and reports what reacting to it is worth "
            "against the best schedule that still has to be fixed in advance. "
            "The comparison is not against either regime's own schedule: a "
            "static trader who knows the chain's law uses the expected "
            "coefficients period by period, and beating that is the only gain "
            "adapting can claim."
        ),
    )
    adaptive.add_argument("--quantity", type=float, required=True)
    adaptive.add_argument("--horizon", type=float, required=True)
    adaptive.add_argument("--periods", type=int, required=True)
    adaptive.add_argument(
        "--gamma", type=float, default=0.0, help="permanent impact, price per share"
    )
    adaptive.add_argument(
        "--eta",
        type=float,
        required=True,
        help="temporary impact of the liquid regime, price per share per unit time",
    )
    adaptive.add_argument(
        "--illiquid-eta",
        dest="illiquid_eta",
        type=float,
        required=True,
        help="temporary impact of the illiquid regime, in the same units",
    )
    adaptive.add_argument("--volatility", type=float, required=True, help="of the liquid regime")
    adaptive.add_argument(
        "--illiquid-volatility",
        dest="illiquid_volatility",
        type=float,
        default=None,
        help="of the illiquid regime; defaults to the liquid one's",
    )
    adaptive.add_argument(
        "--persistence",
        type=float,
        required=True,
        help=(
            "probability a regime repeats next period. Worth nothing to adapt to "
            "at either one or zero: both are perfectly predictable chains"
        ),
    )
    adaptive.add_argument(
        "--risk-aversion",
        dest="risk_aversion",
        type=float,
        default=0.0,
        help="lambda on the running variance penalty",
    )
    adaptive.add_argument(
        "--start",
        type=int,
        default=0,
        choices=(0, 1),
        help="regime the order starts in: 0 liquid, 1 illiquid",
    )
    adaptive.add_argument(
        "--draws",
        type=int,
        default=0,
        help=(
            "simulate this many regime paths and report the realised objective, "
            "which shares no arithmetic with the recursion"
        ),
    )
    adaptive.add_argument("--seed", type=int, default=0)
    adaptive.add_argument("--json", action="store_true")

    placement = sub.add_parser(
        "placement",
        help="rest a limit order or cross the spread, priced",
        description=(
            "Prices the decision to rest a buy limit order below the mid "
            "against crossing immediately, under a Brownian mid with a drift. "
            "Three results are worth reading before the price. The fill "
            "probability is a running-minimum probability, which is exactly "
            "twice the chance of merely ending below the limit. Conditional on "
            "filling, the expected mid at the horizon is exactly the limit "
            "price, so the adverse selection cancels the whole apparent "
            "saving. And the expected cost therefore depends on the distance "
            "only through the fill probability, with the standard deviation "
            "rising in the same direction -- so there is no frontier here, and "
            "the table printed by --sweep is there to show that rather than to "
            "be optimised over."
        ),
    )
    placement.add_argument(
        "--distance",
        type=float,
        required=True,
        help="how far below the arrival mid the order rests, in price units",
    )
    placement.add_argument(
        "--horizon",
        type=float,
        required=True,
        help="how long it rests before being crossed, in the volatility's time unit",
    )
    placement.add_argument(
        "--volatility",
        type=float,
        required=True,
        help="of the mid, in price units per root time unit",
    )
    placement.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="of the mid, per time unit. Positive runs away from a buyer",
    )
    placement.add_argument(
        "--half-spread",
        dest="half_spread",
        type=float,
        default=0.0,
        help="paid when the order has to cross",
    )
    placement.add_argument(
        "--taker-fee", dest="taker_fee", type=float, default=0.0, help="paid on crossing"
    )
    placement.add_argument(
        "--maker-rebate",
        dest="maker_rebate",
        type=float,
        default=0.0,
        help="earned when the resting order fills",
    )
    placement.add_argument(
        "--sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="SD",
        help="also evaluate these distances, in standard deviations of the mid",
    )
    placement.add_argument(
        "--draws",
        type=int,
        default=0,
        help=(
            "simulate this many paths and report the fill frequency, which is "
            "biased low by discrete monitoring and is reported beside the "
            "continuity-corrected formula for that reason"
        ),
    )
    placement.add_argument(
        "--steps", type=int, default=1000, help="observations per simulated path"
    )
    placement.add_argument("--seed", type=int, default=0)
    placement.add_argument("--json", action="store_true")

    vwap = sub.add_parser(
        "vwap",
        help="tracking error against a volume-weighted benchmark, and its floor",
        description=(
            "Every other schedule here minimises cost against the arrival price, "
            "which stands still. A VWAP benchmark is built from the same prices "
            "the order trades at, so what matters is the difference between our "
            "participation and the market's -- and a schedule matching the "
            "realised volume curve would have zero tracking error on every path. "
            "Volume uncertainty is the only thing stopping it, so the report "
            "leads with the irreducible floor that uncertainty sets, which no "
            "schedule crosses."
        ),
    )
    vwap.add_argument("--quantity", type=float, required=True, help="order size in shares")
    vwap.add_argument(
        "--profile",
        type=float,
        nargs="+",
        required=True,
        metavar="W",
        help="expected share of volume in each bucket; normalised",
    )
    vwap.add_argument(
        "--dispersion",
        type=float,
        nargs="+",
        default=None,
        metavar="SD",
        help="cross-day standard deviation of each bucket's share",
    )
    vwap.add_argument(
        "--concentration",
        type=float,
        default=None,
        help="Dirichlet concentration, instead of fitting one to --dispersion",
    )
    vwap.add_argument(
        "--volatility",
        type=float,
        required=True,
        help="fractional price standard deviation over one bucket",
    )
    vwap.add_argument("--price", type=float, default=1.0, help="arrival price")
    vwap.add_argument(
        "--bucket-hours",
        dest="bucket_hours",
        type=float,
        default=1.0,
        help="bucket length in the units --eta is quoted in",
    )
    vwap.add_argument(
        "--eta", type=float, default=None, help="temporary impact, price per share per unit time"
    )
    vwap.add_argument("--gamma", type=float, default=0.0, help="permanent impact, price per share")
    vwap.add_argument(
        "--risk-aversion",
        dest="risk_aversion",
        type=float,
        nargs="+",
        default=None,
        metavar="L",
        help="weights on tracking variance; traces the frontier against --eta",
    )
    vwap.add_argument(
        "--compare",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help="score another schedule, such as an arrival-price trajectory, against VWAP",
    )
    vwap.add_argument("--json", action="store_true")

    sample = sub.add_parser("sample-data", help="write a synthetic book to CSV")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--orders", type=int, default=60)
    sample.add_argument("--symbols", type=int, default=8)
    return parser


# -- tca --------------------------------------------------------------------


def _print_report(report: TcaReport, group_by: str, threshold: float, out: TextIO) -> None:
    total = report.total()
    print(
        f"{total.orders} orders, paper notional {total.paper_notional:,.0f}, "
        f"shortfall {total.total:,.0f} ({total.total_bps:+.2f} bps)",
        file=out,
    )
    print(file=out)
    header = f"{'':<12}{'orders':>7}" + "".join(f"{c:>13}" for c in COMPONENTS) + f"{'total':>11}"
    print(header, file=out)
    print("-" * len(header), file=out)

    def line(name: str, orders: int, bps: dict[str, float], total_bps: float) -> None:
        cells = "".join(f"{bps[c]:>13.2f}" for c in COMPONENTS)
        print(f"{name:<12}{orders:>7}{cells}{total_bps:>11.2f}", file=out)

    groups = report.group_by(lambda r: r.symbol if group_by == "symbol" else r.side.value)
    for name, agg in groups.items():
        line(name, agg.orders, agg.bps(), agg.total_bps)
    print("-" * len(header), file=out)
    line("all", total.orders, total.bps(), total.total_bps)
    print("(basis points of paper notional)", file=out)

    flagged = report.outliers(threshold)
    print(file=out)
    if not flagged:
        print(f"no outliers at a modified z-score of {threshold}", file=out)
        return
    print(f"outliers at a modified z-score of {threshold}:", file=out)
    for row, z in flagged:
        print(
            f"  {row.order_id:<10} {row.symbol:<8} {row.side.value:<4} "
            f"{row.total_bps:+9.2f} bps   z = {z:+.1f}",
            file=out,
        )


def _run_tca(args: argparse.Namespace, out: TextIO) -> None:
    orders = load_orders(args.orders, args.fills)
    bars = load_bars(args.bars)
    report = build_report(
        orders, bars, fees_per_share=args.fees_per_share, delay_basis=DelayBasis(args.delay_basis)
    )
    if args.json:
        json.dump(report.to_dict(), out, indent=2)
        print(file=out)
    else:
        _print_report(report, args.group_by, args.outlier_threshold, out)


# -- mark-outs ---------------------------------------------------------------


def _horizon_list(text: str) -> list[timedelta]:
    minutes = _float_list(text)
    for value in minutes:
        if value <= 0.0:
            raise argparse.ArgumentTypeError(
                f"a mark-out horizon is a positive number of minutes, got {value!r}"
            )
    return [timedelta(minutes=value) for value in minutes]


def _run_markouts(args: argparse.Namespace, out: TextIO) -> None:
    orders = load_orders(args.orders, args.fills)
    bars = load_bars(args.bars)
    benchmark: BarSeries | None = None
    if args.benchmark is not None:
        series = load_bars(args.benchmark)
        if args.benchmark_symbol is not None:
            try:
                benchmark = series[args.benchmark_symbol]
            except KeyError:
                raise ValidationError(
                    f"{args.benchmark} has no symbol {args.benchmark_symbol!r}; it has "
                    f"{', '.join(sorted(series))}"
                ) from None
        elif len(series) == 1:
            benchmark = next(iter(series.values()))
        else:
            raise ValidationError(
                f"{args.benchmark} holds {len(series)} symbols; name one with --benchmark-symbol"
            )
    profile = reversion_profile(
        orders.values(),
        bars,
        horizons=args.horizons,
        benchmark=benchmark,
        beta=args.beta,
    )
    payload: dict[str, object] = {
        "orders": profile.orders,
        "benchmark_adjusted": profile.benchmark_adjusted,
        "beta": args.beta if profile.benchmark_adjusted else None,
        "mean_realised_bps": profile.mean_realised_bps,
        "mean_impact_bps": profile.mean_impact_bps,
        "impact_standard_error": profile.impact_standard_error,
        "reverted_fraction": profile.reverted_fraction,
        "horizons": [
            {
                "minutes": point.horizon.total_seconds() / 60.0,
                "orders": point.orders,
                "mean_permanent_bps": point.mean_permanent_bps if point.orders else None,
                "standard_error": point.standard_error if point.orders else None,
                "mean_reverted_bps": point.mean_reverted_bps if point.orders else None,
                "reverted_fraction": point.reverted_fraction(profile.mean_impact_bps)
                if point.orders
                else None,
            }
            for point in profile.points
        ],
    }
    try:
        decay = profile.decay()
        payload["decay"] = {
            "half_life_seconds": decay.half_life.total_seconds(),
            "asymptote_bps": decay.asymptote_bps,
            "amplitude_bps": decay.amplitude_bps,
            "permanent_fraction": decay.permanent_fraction,
            "r_squared": decay.r_squared,
        }
    except SlippageError as refused:
        # Reported rather than raised: the curve is still worth printing when no
        # half-life can be read off it, and *why* it could not is the finding.
        payload["decay"] = None
        payload["decay_refused"] = str(refused)
    if args.json:
        json.dump(payload, out, indent=2, allow_nan=False)
        print(file=out)
        return

    adjustment = (
        f"market-adjusted at beta {args.beta:g}"
        if profile.benchmark_adjusted
        else "NOT market-adjusted"
    )
    print(
        f"Mark-outs over {profile.orders} orders, {adjustment}\n",
        file=out,
    )
    print(
        f"  realised cost against arrival   {profile.mean_realised_bps:8.2f} bps",
        file=out,
    )
    print(
        f"  impact at completion            {profile.mean_impact_bps:8.2f} bps "
        f"+/- {profile.impact_standard_error:.2f}\n",
        file=out,
    )
    print(
        f"  {'horizon':>9}  {'orders':>6}  {'persisted':>12}  {'reverted':>9}  {'of impact':>9}",
        file=out,
    )
    for point in profile.points:
        if not point.orders:
            print(
                f"  {point.horizon.total_seconds() / 60.0:7.0f}m  {0:6d}  {'no data':>12}",
                file=out,
            )
            continue
        fraction = point.reverted_fraction(profile.mean_impact_bps)
        print(
            f"  {point.horizon.total_seconds() / 60.0:7.0f}m  {point.orders:6d}  "
            f"{point.mean_permanent_bps:7.2f} +/-{point.standard_error:4.2f}  "
            f"{point.mean_reverted_bps:9.2f}  "
            f"{'--' if fraction is None else format(fraction, '9.1%')}",
            file=out,
        )
    decayed = payload["decay"]
    if isinstance(decayed, dict):
        print(
            f"\n  half-life {decayed['half_life_seconds']:.0f}s, heading for "
            f"{decayed['asymptote_bps']:.2f} bps of permanent impact "
            f"(r-squared {decayed['r_squared']:.3f})",
            file=out,
        )
    else:
        print(f"\n  no half-life: {payload['decay_refused']}", file=out)
    if not profile.benchmark_adjusted:
        print(
            "\n  No benchmark was given. Over an hour a stock's own move is tens of "
            "basis points\n  against single figures of impact, so most of the numbers "
            "above are the market's.\n  Pass --benchmark with an index series.",
            file=out,
        )


# -- schedule and frontier --------------------------------------------------


def _impact(args: argparse.Namespace) -> ImpactModel:
    if args.beta is None or args.beta == 1.0:
        return LinearImpact(gamma=args.gamma, eta=args.eta, epsilon=args.epsilon)
    return PowerLawImpact(gamma=args.gamma, eta=args.eta, beta=args.beta, epsilon=args.epsilon)


def _one_or_many(values: list[float] | None) -> float | list[float] | None:
    if values is None:
        return None
    return values[0] if len(values) == 1 else values


def _run_schedule(args: argparse.Namespace, out: TextIO) -> None:
    impact = _impact(args)
    constrained = args.max_trade is not None or args.min_trade is not None
    tau = args.horizon / args.periods
    if isinstance(impact, LinearImpact) and not constrained and args.lot_size is None:
        problem = ExecutionProblem(
            quantity=args.quantity,
            horizon=args.horizon,
            periods=args.periods,
            volatility=args.volatility,
            impact=impact,
        )
        trajectory = optimal_trajectory(problem, args.risk_aversion)
        method = "closed form"
        trades = list(trajectory.trades)
        expected, variance = trajectory.expected_cost, trajectory.variance
    else:
        lot = args.lot_size if args.lot_size is not None else args.quantity / 1000
        plan = solve_schedule(
            quantity=args.quantity,
            horizon=args.horizon,
            periods=args.periods,
            volatility=args.volatility,
            impact=impact,
            risk_aversion=args.risk_aversion,
            lot_size=lot,
            max_trade=_one_or_many(args.max_trade),
            min_trade=_one_or_many(args.min_trade),
        )
        method = f"dynamic programme, lot {lot:g}"
        trades = plan.trades
        expected = schedule_objective(impact, trades, tau, args.volatility, 0.0)
        held = args.quantity
        variance = 0.0
        for n in trades:
            held -= n
            variance += held * held
        variance *= args.volatility**2 * tau

    if args.json:
        payload = {
            "method": method,
            "trades": trades,
            "expected_cost": expected,
            "std": math.sqrt(variance),
            "objective": expected + args.risk_aversion * variance,
        }
        json.dump(payload, out, indent=2)
        print(file=out)
        return
    print(f"method: {method}", file=out)
    print(f"{'period':>6}{'start':>10}{'trade':>14}{'remaining':>14}", file=out)
    remaining = args.quantity
    for k, n in enumerate(trades):
        remaining -= n
        print(f"{k:>6}{k * tau:>10.4g}{n:>14,.0f}{max(remaining, 0.0):>14,.0f}", file=out)
    print(
        f"\nexpected cost {expected:,.0f}, standard deviation {math.sqrt(variance):,.0f}", file=out
    )


def _run_transient(args: argparse.Namespace, out: TextIO) -> None:
    """Schedule against a decay kernel rather than against a trading rate.

    The interesting column is the saving against a constant rate, because it is
    zero at both extremes of resilience and the reason is different at each end:
    decay fast enough and the constant rate is already optimal, decay slowly
    enough and every schedule ties.
    """
    if args.half_life is not None and args.resilience is not None:
        raise SlippageError(
            "give either --half-life or --resilience, not both; they are the same "
            "parameter written two ways"
        )
    if args.half_life is None and args.resilience is None:
        raise SlippageError("the kernel needs a decay rate: pass --half-life or --resilience")
    tau = args.horizon / args.periods
    kernel: ExponentialDecay | PowerLawDecay
    if args.exponent is not None:
        if args.half_life is None:
            raise SlippageError("a power-law kernel is scaled by --half-life, not --resilience")
        kernel = PowerLawDecay(
            permanent=args.gamma,
            transient=args.eta,
            exponent=args.exponent,
            scale=args.half_life,
        )
    else:
        resilience = (
            args.resilience if args.resilience is not None else math.log(2.0) / args.half_life
        )
        kernel = ExponentialDecay(permanent=args.gamma, transient=args.eta, resilience=resilience)
    plan = optimal_transient_schedule(kernel, args.quantity, args.periods, tau)
    horizons = [multiple * tau for multiple in (0.0, 1.0, 2.0, 5.0, 10.0)]
    residual = residual_impact(kernel, list(plan.trades), tau, horizons)
    if args.json:
        payload = {
            "kernel": type(kernel).__name__,
            "instantaneous_impact_per_share": kernel.value(0.0),
            "permanent_impact_per_share": kernel.permanent,
            "trades": list(plan.trades),
            "cost": plan.cost,
            "uniform_cost": plan.uniform_cost,
            "saving": plan.saving,
            "front_load": plan.front_load,
            "residual_impact": [
                {"periods_after": multiple, "impact": value}
                for multiple, value in zip((0.0, 1.0, 2.0, 5.0, 10.0), residual, strict=True)
            ],
        }
        json.dump(payload, out, indent=2)
        print(file=out)
        return
    print(f"kernel: {type(kernel).__name__}", file=out)
    print(
        f"impact per share: {kernel.value(0.0):.6g} now, {kernel.permanent:.6g} permanent",
        file=out,
    )
    print(f"{'period':>6}{'start':>10}{'trade':>14}{'remaining':>14}", file=out)
    remaining = args.quantity
    for index, size in enumerate(plan.trades):
        remaining -= size
        print(
            f"{index:>6}{index * tau:>10.4g}{size:>14,.0f}{remaining:>14,.0f}",
            file=out,
        )
    print(
        f"\ncost {plan.cost:,.0f} against {plan.uniform_cost:,.0f} at a constant rate: "
        f"{100.0 * plan.saving:.3f}% saved, first slice {plan.front_load:.3g}x uniform",
        file=out,
    )
    print("\nimpact left after the order, in periods:", file=out)
    for multiple, value in zip((0.0, 1.0, 2.0, 5.0, 10.0), residual, strict=True):
        print(f"{multiple:>10.0f}{value:>16.6g}", file=out)


def _run_frontier(args: argparse.Namespace, out: TextIO) -> None:
    impact = _impact(args)
    if not isinstance(impact, LinearImpact):
        raise SlippageError("the frontier uses the closed form, which needs linear impact")
    problem = ExecutionProblem(
        quantity=args.quantity,
        horizon=args.horizon,
        periods=args.periods,
        volatility=args.volatility,
        impact=impact,
    )
    points = efficient_frontier(problem, args.risk_aversions)
    rows = [
        {
            "risk_aversion": lam,
            "expected_cost": t.expected_cost,
            "std": t.std,
            "half_life": None if math.isinf(t.half_life) else t.half_life,
            "first_trade": t.trades[0],
        }
        for lam, t in zip(args.risk_aversions, points, strict=True)
    ]
    if args.json:
        json.dump(rows, out, indent=2)
        print(file=out)
        return
    print(f"{'lambda':>10}{'E[cost]':>14}{'sd':>14}{'half-life':>11}{'first trade':>14}", file=out)
    for row in rows:
        half = "inf" if row["half_life"] is None else f"{row['half_life']:.3g}"
        print(
            f"{row['risk_aversion']:>10g}{row['expected_cost']:>14,.0f}{row['std']:>14,.0f}"
            f"{half:>11}{row['first_trade']:>14,.0f}",
            file=out,
        )


def _run_vwap(args: argparse.Namespace, out: TextIO) -> None:
    """Tracking error against a moving benchmark, and the part of it nobody can remove.

    The floor is the first number because it is the one that decides whether any
    of the rest is worth acting on. A schedule 2 basis points worse than optimal
    against a 20 basis point floor is inside what a desk can measure; the same 2
    against a 1 basis point floor is not.

    ``--compare`` is the number that separates two things usually lumped together.
    An arrival-price trajectory front-loads, which is the right answer to a
    different question, and scoring it here says what that answer costs when the
    benchmark is VWAP.
    """
    weights = list(args.profile)
    total = sum(weights)
    if total <= 0.0:
        raise SlippageError("the profile weights sum to zero")
    fractions = tuple(one / total for one in weights)
    dispersion: tuple[float, ...] = ()
    if args.dispersion is not None:
        if len(args.dispersion) != len(fractions):
            raise SlippageError(
                f"--dispersion has {len(args.dispersion)} entries and --profile has "
                f"{len(fractions)}"
            )
        dispersion = tuple(float(one) for one in args.dispersion)
    if args.dispersion is None and args.concentration is None:
        raise SlippageError(
            "volume uncertainty is what makes a VWAP benchmark unreachable, so it "
            "has to come from somewhere: pass --dispersion, which "
            "volume.estimate_profile measures, or --concentration directly"
        )
    profile = VolumeProfile(
        fractions=fractions,
        bucket=timedelta(hours=args.bucket_hours),
        session_open=dt_time(9, 30),
        dispersion=dispersion,
        days=0,
    )
    uncertainty = fit_volume_covariance(profile, concentration=args.concentration)
    problem = TrackingProblem(
        shares=args.quantity,
        profile=profile,
        volatility=args.volatility,
        bucket_hours=args.bucket_hours,
    )
    matched = tracking_moments(problem, problem.expected, uncertainty)

    payload: dict[str, object] = {
        "buckets": problem.buckets,
        "irreducible_bps": matched.irreducible_bps,
        "tracking_error_bps": matched.tracking_error_bps,
        "concentration": uncertainty.concentration,
        "dispersion_fit_error": uncertainty.dispersion_error,
        "schedule": [float(one) for one in problem.expected],
    }
    points: tuple[TrackingFrontierPoint, ...] = ()
    if args.risk_aversion is not None:
        if args.eta is None:
            raise SlippageError("a frontier against tracking error needs --eta")
        model = LinearImpact(gamma=args.gamma, eta=args.eta, epsilon=0.0)
        points = tracking_frontier(
            problem, uncertainty, model, args.risk_aversion, price=args.price
        )
        payload["frontier"] = [
            {
                "risk_aversion": one.risk_aversion,
                "impact_bps": one.impact_bps,
                "tracking_error_bps": one.tracking_error_bps,
                "schedule": list(one.schedule),
            }
            for one in points
        ]
    if args.compare is not None:
        if len(args.compare) != problem.buckets:
            raise SlippageError(
                f"--compare has {len(args.compare)} entries and --profile has {problem.buckets}"
            )
        comparison = compare_objectives(problem, uncertainty, args.compare)
        payload["comparison"] = {
            "vwap_schedule_error_bps": comparison.vwap_schedule_error_bps,
            "other_schedule_error_bps": comparison.other_schedule_error_bps,
            "excess_bps": comparison.excess_bps,
            "excess_over_floor": comparison.excess_over_floor,
        }

    if args.json:
        print(json.dumps(payload, indent=2), file=out)
        return

    print(f"buckets                    {problem.buckets}", file=out)
    print(f"irreducible floor          {matched.irreducible_bps:.4f} bp", file=out)
    print(f"volume-matching schedule   {matched.tracking_error_bps:.4f} bp", file=out)
    print(
        f"Dirichlet concentration    {uncertainty.concentration:.4f}"
        f"  (worst dispersion miss {uncertainty.dispersion_error:.1%})",
        file=out,
    )
    if points:
        print(file=out)
        print(f"{'lambda':>14}{'impact bp':>12}{'tracking bp':>14}", file=out)
        for point in points:
            print(
                f"{point.risk_aversion:>14.4g}{point.impact_bps:>12.4f}"
                f"{point.tracking_error_bps:>14.4f}",
                file=out,
            )
    if args.compare is not None:
        comparison = compare_objectives(problem, uncertainty, args.compare)
        print(file=out)
        print(
            f"the schedule given tracks at {comparison.other_schedule_error_bps:.4f} bp "
            f"against the volume curve's {comparison.vwap_schedule_error_bps:.4f} bp: "
            f"{comparison.excess_bps:+.4f} bp, which is "
            f"{comparison.excess_over_floor:+.2f} times the floor",
            file=out,
        )


def _run_sample(args: argparse.Namespace, out: TextIO) -> None:
    book = synthetic_book(
        np.random.default_rng(args.seed), symbols=args.symbols, orders=args.orders
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_orders(book.orders, args.out / "orders.csv", args.out / "fills.csv")
    write_bars(book.bars, args.out / "bars.csv")
    fills = sum(len(o.fills) for o in book.orders.values())
    print(
        f"wrote {len(book.orders)} orders, {fills} fills and {len(book.bars)} symbols' bars "
        f"to {args.out}",
        file=out,
    )


def _read_matrix(path: Path, size: int, name: str) -> list[list[float]]:
    """Read a square matrix of numbers, naming the line that is wrong.

    No header, one row per asset, in the order the holdings file gives them.
    Errors name the line for the same reason the rest of this interface does: a
    matrix is pasted together by hand and "row 3 has 2 entries" is different
    information from a traceback out of a dataclass.
    """
    rows: list[list[float]] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field for field in line.replace(",", " ").split() if field]
        try:
            values = [float(field) for field in fields]
        except ValueError:
            raise SlippageError(
                f"{name}, line {number}: {line!r} is not a row of numbers"
            ) from None
        if len(values) != size:
            raise SlippageError(
                f"{name}, line {number} has {len(values)} entries for {size} assets"
            )
        rows.append(values)
    if len(rows) != size:
        raise SlippageError(f"{name} has {len(rows)} rows for {size} assets")
    return rows


def _run_basket(args: argparse.Namespace, out: TextIO) -> None:
    names: list[str] = []
    quantities: list[float] = []
    etas: list[float] = []
    gammas: list[float] = []
    with args.holdings.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "quantity", "eta", "gamma"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SlippageError(f"the holdings file is missing {', '.join(sorted(missing))}")
        for number, row in enumerate(reader, start=2):
            try:
                quantities.append(float(row["quantity"]))
                etas.append(float(row["eta"]))
                gammas.append(float(row["gamma"]))
            except (TypeError, ValueError):
                raise SlippageError(
                    f"the holdings file, line {number}: a number is missing or unreadable"
                ) from None
            names.append(str(row["symbol"]))
    if not names:
        raise SlippageError("the holdings file has no rows in it")
    size = len(names)
    covariance = _read_matrix(args.covariance, size, "the covariance file")
    if args.cross_impact is not None:
        temporary = _read_matrix(args.cross_impact, size, "the cross-impact file")
    else:
        temporary = [
            [etas[row] if row == column else 0.0 for column in range(size)] for row in range(size)
        ]
    permanent = [
        [gammas[row] if row == column else 0.0 for column in range(size)] for row in range(size)
    ]
    problem = BasketProblem(
        holdings=tuple(quantities),
        temporary=tuple(tuple(row) for row in temporary),
        permanent=tuple(tuple(row) for row in permanent),
        covariance=tuple(tuple(row) for row in covariance),
        horizon=args.horizon,
        periods=args.periods,
        names=tuple(names),
    )
    solved = basket_trajectory(problem, args.risk_aversion)
    payload: dict[str, object] = {
        "assets": list(solved.names),
        "risk_aversion": args.risk_aversion,
        "expected_cost": solved.expected_cost,
        "std": solved.std,
        "objective": solved.objective,
        "directions": [
            {
                "risk_per_impact": one.risk_per_impact,
                "kappa": one.kappa,
                "half_life": None if math.isinf(one.half_life) else one.half_life,
                "weights": list(one.weights),
            }
            for one in solved.directions
        ],
        "holdings": [list(row) for row in solved.holdings],
    }
    comparison = None
    if args.compare:
        comparison = compare_to_independent(problem, args.risk_aversion)
        payload["independent"] = {
            "expected_cost": comparison.independent.expected_cost,
            "std": comparison.independent.std,
            "objective": comparison.independent.objective,
        }
        payload["objective_saving"] = comparison.objective_saving
        payload["peak_risk_ratio"] = comparison.peak_risk_ratio
        try:
            payload["exposure_ratio"] = comparison.exposure_ratio()
            payload["hedge_direction"] = list(hedge_direction(problem))
        except ValidationError as refusal:
            # Not an error: on a symmetric basket both schedules keep the hedge
            # exactly and the ratio is one rounding error over another. Saying so
            # is more useful than printing a number that means nothing.
            payload["exposure_ratio"] = None
            payload["exposure_note"] = str(refusal)
    if args.json:
        json.dump(payload, out, indent=2)
        print(file=out)
        return
    print(
        f"{size} assets over {args.periods} intervals, risk aversion {args.risk_aversion:g}\n",
        file=out,
    )
    print(f"expected cost {solved.expected_cost:,.0f}   sd {solved.std:,.0f}", file=out)
    print(f"{'direction':>32}{'risk/impact':>14}{'half-life':>12}", file=out)
    for one in solved.directions:
        weights = " ".join(f"{value:+.3f}" for value in one.weights)
        half = "inf" if math.isinf(one.half_life) else f"{one.half_life:.3g}"
        print(f"{weights:>32}{one.risk_per_impact:>14.4g}{half:>12}", file=out)
    print(
        "\nA basket is liquidated along these directions, each at its own pace. "
        "The riskiest per unit of impact is worked off first; a direction with "
        "little risk in it is cheap to hold and is left until the end, which is "
        "why a hedged pair comes off as a pair.",
        file=out,
    )
    if comparison is not None:
        print(
            f"\nSolved leg by leg the cost is "
            f"{comparison.independent.expected_cost:,.0f} against "
            f"{solved.expected_cost:,.0f}, and the joint solution improves on the "
            f"whole objective by {comparison.objective_saving:.2%}.",
            file=out,
        )
        try:
            print(
                f"At its worst the leg-by-leg schedule leaves "
                f"{comparison.exposure_ratio():.1f} times as much exposure along "
                f"the direction the basket starts flat in, which is the part the "
                f"totals do not show.",
                file=out,
            )
        except ValidationError as refusal:
            print(str(refusal), file=out)
        print(
            f"Its peak one-interval variance is {comparison.peak_risk_ratio:.2f} "
            f"times the joint solution's: below one means the joint schedule is "
            f"deliberately the riskier one moment to moment, because a hedged "
            f"basket is cheap to hold and the optimal schedule holds it longer.",
            file=out,
        )


def _run_adaptive(args: argparse.Namespace, out: TextIO) -> None:
    """Price the option to react, against the best schedule that cannot.

    The number to read is the saving, and it is zero at both ends of the
    persistence range for the same underlying reason: a chain that never moves
    and a chain that strictly alternates are both perfectly predictable, and a
    static schedule can use either. Only uncertainty is worth reacting to.
    """
    if not 0.0 <= args.persistence <= 1.0:
        raise SlippageError(f"--persistence is a probability, got {args.persistence!r}")
    leave = 1.0 - args.persistence
    liquid = LiquidityRegime(
        impact=LinearImpact(gamma=args.gamma, eta=args.eta),
        volatility=args.volatility,
        label="liquid",
    )
    illiquid = LiquidityRegime(
        impact=LinearImpact(gamma=args.gamma, eta=args.illiquid_eta),
        volatility=(
            args.volatility if args.illiquid_volatility is None else args.illiquid_volatility
        ),
        label="illiquid",
    )
    chain = RegimeChain(
        regimes=(liquid, illiquid),
        transitions=((args.persistence, leave), (leave, args.persistence)),
    )
    problem = AdaptiveProblem(
        quantity=args.quantity,
        horizon=args.horizon,
        periods=args.periods,
        chain=chain,
    )
    policy = solve_adaptive(problem, args.risk_aversion)
    fixed = static_schedule(problem, args.risk_aversion, args.start)
    gain = adaptivity_gain(problem, args.risk_aversion, args.start)

    remaining = args.quantity
    static_fractions = []
    for size in fixed.trades:
        static_fractions.append(size / remaining if remaining > 0.0 else 0.0)
        remaining -= size

    simulated = (
        simulate_policy(policy, args.start, draws=args.draws, seed=args.seed)
        if args.draws > 0
        else None
    )

    if args.json:
        payload: dict[str, object] = {
            "start": chain.regimes[args.start].name,
            "adaptive_objective": gain.adaptive,
            "static_objective": gain.static,
            "saved": gain.saved,
            "saved_fraction": gain.fraction,
            "static_expected_impact": fixed.expected_impact,
            "static_expected_risk": fixed.expected_risk,
            "periods": [
                {
                    "period": index,
                    "static_fraction": static_fractions[index],
                    "liquid_fraction": policy.fractions[index][0],
                    "illiquid_fraction": policy.fractions[index][1],
                }
                for index in range(problem.periods)
            ],
        }
        if simulated is not None:
            payload["simulated"] = {
                "mean": simulated.mean,
                "standard_error": simulated.standard_error,
                "draws": simulated.draws,
                "covers_the_recursion": simulated.covers(gain.adaptive),
            }
        json.dump(payload, out, indent=2)
        print(file=out)
        return

    print(f"starting regime: {chain.regimes[args.start].name}", file=out)
    print(
        f"objective: {gain.adaptive:.6g} adapting against {gain.static:.6g} fixed in advance",
        file=out,
    )
    print(
        f"saving: {gain.saved:.6g} ({100.0 * gain.fraction:.3f}% of the static objective)",
        file=out,
    )
    print(
        f"{'period':>6}{'static':>12}{'liquid':>12}{'illiquid':>12}{'liquid/static':>16}",
        file=out,
    )
    for index in range(problem.periods):
        reference = static_fractions[index]
        liquid_fraction, illiquid_fraction = policy.fractions[index]
        ratio = liquid_fraction / reference if reference > 0.0 else float("nan")
        print(
            f"{index:>6}{reference:>12.6f}{liquid_fraction:>12.6f}"
            f"{illiquid_fraction:>12.6f}{ratio:>16.3f}",
            file=out,
        )
    if simulated is not None:
        errors = (
            abs(simulated.mean - gain.adaptive) / simulated.standard_error
            if simulated.standard_error > 0.0
            else 0.0
        )
        print(
            f"simulated over {simulated.draws} paths: {simulated.mean:.6g} "
            f"+/- {simulated.standard_error:.4g}, {errors:.2f} standard errors "
            f"from the recursion",
            file=out,
        )
    print(
        "A persistence of one or of zero is worth nothing to adapt to: the first "
        "never moves and the second strictly alternates, so a schedule fixed in "
        "advance can use either.",
        file=out,
    )


def _run_placement(args: argparse.Namespace, out: TextIO) -> None:
    """Price resting against crossing, and show why there is nothing to optimise.

    The table under ``--sweep`` is the point of the command. Both the expected
    cost and its standard deviation rise with the distance, so a reader looking
    for the distance that trades one off against the other will not find it:
    the answer is always the tightest price the book allows. What changes that
    is the drift, and the same table at a non-zero ``--drift`` shows by how
    much.
    """
    resting = Placement(
        distance=args.distance,
        horizon=args.horizon,
        volatility=args.volatility,
        drift=args.drift,
        half_spread=args.half_spread,
        taker_fee=args.taker_fee,
        maker_rebate=args.maker_rebate,
    )
    outcome = evaluate(resting)
    deviation = resting.deviation
    terminal = 0.5 * math.erfc(resting.standardised_distance / math.sqrt(2.0))

    swept: list[tuple[float, Outcome]] = []
    if args.sweep:
        for multiple in args.sweep:
            if multiple <= 0.0:
                raise SlippageError(
                    f"--sweep takes positive multiples of a standard deviation, got {multiple!r}"
                )
        swept = [(multiple, evaluate(resting.at(multiple * deviation))) for multiple in args.sweep]

    drawn = None
    corrected = None
    if args.draws > 0:
        drawn = simulate_placement(
            resting,
            paths=args.draws,
            steps=args.steps,
            rng=np.random.default_rng(args.seed),
        )
        shift = monitoring_shift(args.volatility, args.horizon, args.steps)
        corrected = evaluate(resting.at(args.distance + shift)).fill_probability

    if args.json:
        payload: dict[str, object] = {
            "deviation": deviation,
            "standardised_distance": resting.standardised_distance,
            "fill_probability": outcome.fill_probability,
            "terminal_probability": terminal,
            "running_over_terminal": outcome.fill_probability / terminal,
            "expected_cost": outcome.expected_cost,
            "cost_deviation": outcome.cost_deviation,
            "cost_if_filled": outcome.cost_if_filled,
            "mid_if_filled": outcome.mid_if_filled,
            "mid_if_unfilled": outcome.mid_if_unfilled,
            "chase_cost": outcome.chase_cost,
            "crossing_cost": outcome.crossing_cost,
            "advantage": outcome.advantage,
        }
        if swept:
            payload["sweep"] = [
                {
                    "deviations": multiple,
                    "distance": multiple * deviation,
                    "fill_probability": each.fill_probability,
                    "expected_cost": each.expected_cost,
                    "cost_deviation": each.cost_deviation,
                }
                for multiple, each in swept
            ]
        if drawn is not None and corrected is not None:
            payload["simulated"] = {
                "fill_probability": drawn.fill_probability,
                "standard_error": drawn.fill_error,
                "expected_cost": drawn.expected_cost,
                "cost_standard_error": drawn.cost_error,
                "cost_deviation": drawn.cost_deviation,
                "paths": drawn.paths,
                "steps": drawn.steps,
                "monitoring_bias": outcome.fill_probability / drawn.fill_probability - 1.0,
                "corrected_formula": corrected,
                "corrected_gap": corrected / drawn.fill_probability - 1.0,
            }
        json.dump(payload, out, indent=2)
        print(file=out)
        return

    print(f"one standard deviation of the mid over the horizon: {deviation:.6g}", file=out)
    print(f"resting {resting.standardised_distance:.4f} standard deviations out", file=out)
    print(
        f"fill probability: {outcome.fill_probability:.6f}, against "
        f"{terminal:.6f} for merely ending below the limit "
        f"(ratio {outcome.fill_probability / terminal:.6f})",
        file=out,
    )
    print(
        f"mid conditional on a fill: {outcome.mid_if_filled:.6g}, against a limit "
        f"price of {-resting.distance:.6g}",
        file=out,
    )
    print(f"mid conditional on a miss: {outcome.mid_if_unfilled:.6g}", file=out)
    print(
        f"expected cost: {outcome.expected_cost:.6g} +/- {outcome.cost_deviation:.6g}, "
        f"against {outcome.crossing_cost:.6g} to cross now "
        f"(advantage {outcome.advantage:.6g})",
        file=out,
    )
    if swept:
        print(f"{'sd out':>8}{'fill':>12}{'mean cost':>14}{'deviation':>14}", file=out)
        for multiple, each in swept:
            print(
                f"{multiple:>8.3f}{each.fill_probability:>12.6f}"
                f"{each.expected_cost:>14.6g}{each.cost_deviation:>14.6g}",
                file=out,
            )
        print(
            "Both columns move the same way, so there is no distance that trades "
            "one against the other: rest at the tightest price the book allows.",
            file=out,
        )
    if drawn is not None and corrected is not None:
        print(
            f"simulated over {drawn.paths} paths of {drawn.steps} steps: "
            f"{drawn.fill_probability:.6f} +/- {drawn.fill_error:.6f}",
            file=out,
        )
        bias = outcome.fill_probability / drawn.fill_probability - 1.0
        gap = corrected / drawn.fill_probability - 1.0
        print(
            f"discrete monitoring reads {100.0 * bias:.4f}% low; the "
            f"continuity-corrected formula gives {corrected:.6f}, "
            f"{100.0 * gap:+.4f}% from the simulation",
            file=out,
        )


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    stream = sys.stdout if out is None else out
    args = build_parser().parse_args(argv)
    runners = {
        "tca": _run_tca,
        "markouts": _run_markouts,
        "schedule": _run_schedule,
        "frontier": _run_frontier,
        "basket": _run_basket,
        "transient": _run_transient,
        "adaptive": _run_adaptive,
        "placement": _run_placement,
        "vwap": _run_vwap,
        "sample-data": _run_sample,
    }
    try:
        runners[args.command](args, stream)
    except BrokenPipeError:
        # The reader went away (``slippage tca ... | head``). Point stdout at
        # devnull so the interpreter's final flush does not raise again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1
    except (SlippageError, OSError) as error:
        print(f"slippage: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
