# slippage

An optimal execution and transaction cost analysis engine.

The library answers two questions that sit either side of a trade:

- **After the fact** — what did this order actually cost, and where did the cost
  come from: the delay before it reached the market, the impact of trading it,
  or the part that never got done?
- **Before the fact** — given a market impact model and a tolerance for risk,
  how should the remaining quantity be spread over the trading horizon?

See [ROADMAP.md](ROADMAP.md) for the build order.

## Install

```bash
pip install -e ".[dev]"
python -m pytest
```

Python 3.10 or later. The only runtime dependency is NumPy.

## Scoring an execution

```python
from datetime import datetime, timedelta
from slippage import Bar, BarSeries, Benchmark, Fill, Order, Side, score_order

t0 = datetime(2026, 3, 2, 9, 30)
bars = BarSeries(
    Bar(
        t0 + timedelta(minutes=i),
        open=50 + 0.02 * i,
        high=50.05 + 0.02 * i,
        low=49.97 + 0.02 * i,
        close=50.02 + 0.02 * i,
        volume=20_000,
    )
    for i in range(30)
)
order = Order(
    symbol="ACME",
    side=Side.BUY,
    quantity=5_000,
    decision_time=t0,
    arrival_time=t0 + timedelta(minutes=2),
    fills=(
        Fill(t0 + timedelta(minutes=3), 2_000, 50.09),
        Fill(t0 + timedelta(minutes=7), 3_000, 50.16),
    ),
)
for b in (Benchmark.ARRIVAL, Benchmark.INTERVAL_VWAP, Benchmark.CLOSE):
    s = score_order(order, bars, b)
    print(f"{b.value:<14} {s.benchmark_price:8.4f}  {s.cost_bps:+6.2f} bps")
```

```
arrival         50.0400  +18.39 bps
interval_vwap   50.1033   +5.72 bps
close           50.1600   -5.58 bps
```

The same fills cost eighteen basis points against arrival and *saved* five and a
half against the close. Neither number is wrong; they answer different
questions. Arrival measures what trading cost relative to doing nothing;
interval VWAP measures whether the order kept pace with the market while it
traded; the close rewards any buy that finished before a rally. Choosing the
benchmark is choosing the question.

## Implementation shortfall

A single benchmark says whether an execution was good. Implementation shortfall
says where the cost came from, by comparing the real portfolio with a paper one
that bought everything instantly at the decision price.

[`examples/shortfall_worked_example.py`](examples/shortfall_worked_example.py)
buys 10,000 shares decided at 50.00, arriving at 50.10, filling 7,000 and
cancelling the rest at 50.50:

```
component          by hand     library      bps
delay             1,000.00    1,000.00    20.00
trading           1,100.00    1,100.00    22.00
opportunity       1,200.00    1,200.00    24.00
commission           70.00       70.00     1.40
fees                  5.00        5.00     0.10
total             3,375.00    3,375.00    67.50
```

- **Delay** is the market moving before the order reached it. It is a process
  cost, owned by whoever routes orders to the desk rather than by the trader.
- **Trading** is execution against arrival. Given the half-spread it splits into
  the price of immediacy and the remainder, which is impact and timing.
- **Opportunity** is the move on shares that never traded. A trader who cuts an
  order short to save trading cost moves cost here rather than removing it.
- **Explicit** costs are kept apart so that a cheap commission cannot mask an
  expensive execution.

Every figure is in basis points of the paper notional, so the components add up
in bps exactly as they do in currency. On every call the components are
checked against the direct Perold formula computed by a separate route; a
mismatch beyond rounding raises rather than returning a number.

Where the unexecuted shares' delay belongs is a convention. The default *order*
basis charges delay on the whole target, since all of it sat idle; the
*executed* basis charges it only on shares that traded. Both split the same
total, which the test suite checks on randomly generated orders.

## Conventions

**One sign, carried by the side.** `Side.BUY.sign` is `+1` and `Side.SELL.sign`
is `-1`. Every cost is `sign * (execution - reference)`, so a positive number
always means worse than the reference, for either side, and no cost function
branches on buy versus sell. The same sign turns an unsigned quantity into
signed order flow for the impact models, because impact moves the price against
the trader in exactly the direction that makes cost positive.

**Decision and arrival are separate timestamps.** The gap between a portfolio
manager deciding to trade and the order reaching the market is where delay cost
accrues. Collapsing them into one field makes that cost unmeasurable.

**Interval benchmarks use the order's own window** — from arrival to the end of
the bar containing the last fill — rather than the whole session. A full-day
VWAP scores a twenty-minute order on hours it had no part in.

**Windows are half-open on bar starts and bars are never split.** Splitting a
bar would require inventing a price path inside it that the data does not
contain.

**VWAP over an interval with no volume raises** rather than quietly falling back
to TWAP. A caller who asked for VWAP and silently received something else would
be comparing against the wrong benchmark without knowing it.

## Licence

MIT
