"""The ``slippage`` command.

Four subcommands:

``tca``
    Implementation shortfall report over a book of orders read from CSV.
``schedule``
    An execution schedule: the Almgren-Chriss closed form for linear impact
    without constraints, the dynamic programme otherwise.
``frontier``
    Expected cost against risk across a range of risk aversions.
``sample-data``
    Write a synthetic book to CSV, to try ``tca`` without real data.

Every subcommand takes ``--json`` for machine-readable output. Errors in the
inputs exit with status 2 and a one-line message naming the problem, rather
than a traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import numpy as np

from . import __version__
from .exceptions import SlippageError
from .execution import ExecutionProblem, efficient_frontier, optimal_trajectory
from .impact import ImpactModel, LinearImpact, PowerLawImpact
from .io import load_bars, load_orders, write_bars, write_orders
from .report import COMPONENTS, TcaReport, build_report
from .scheduling import schedule_objective, solve_schedule
from .shortfall import DelayBasis
from .synthetic import synthetic_book

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


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    stream = sys.stdout if out is None else out
    args = build_parser().parse_args(argv)
    runners = {
        "tca": _run_tca,
        "schedule": _run_schedule,
        "frontier": _run_frontier,
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
