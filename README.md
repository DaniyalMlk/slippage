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

## Market impact

Two levels of model, because they answer different questions.

**Rate models** price trading at a given speed, which is what a scheduler
needs. `LinearImpact` is the Almgren–Chriss parameterisation — permanent
`g(v) = γv`, temporary `h(v) = ε + ηv` — and `PowerLawImpact` replaces the
temporary term with `ηv^β`. `schedule_cost` prices any discrete schedule under
either, split into temporary and permanent cost; the tests check it against the
hand-integrated cost of uniform schedules and against the Almgren–Chriss
identity `γ(X² − Σnₖ²)/2` for arbitrary ones.

Permanent impact is linear in both, and there is no way to make it otherwise.
Huberman and Stanzl (2004) show that nonlinear permanent impact admits
round-trip manipulation, so an optimiser given such a model would find profit
in its own footprint.

**The square-root law** `I = Yσ(Q/V)^δ` prices a whole order by its size
relative to daily volume. It gives the *peak* impact; an order worked at a
constant rate pays the average of the impact path, `I/(1+δ)` — two thirds of
the peak under the square root. Quoting the peak as the cost overstates it by
half.

### Calibration

Every fit returns estimates with standard errors and a list of identifiability
warnings, also raised as `IdentifiabilityWarning`. Five hundred synthetic orders
generated from `Y = 0.8`, `δ = 0.5`:

```python
fit = fit_power_law(participation, cost, volatility)
fit.y  # 0.814 +/- 0.024
fit.delta  # 0.503 +/- 0.009
fit.to_law().expected_cost_bps(1e5, 1e7, 0.02)  # 10.7 bps for 1% of ADV at 2% vol
```

Forty noisy orders that all sit between 1% and 1.5% of daily volume:

```
delta = 1.50 +/- 0.84
 - order sizes span only a 1.47x range; the exponent is not identified below a 4x range — fix delta instead
 - the exponent 1.500 sits on the search bound (0.05, 1.5); the true optimum may lie outside it
 - exponent standard error 0.84 exceeds 0.25; the data cannot distinguish a square-root law from a linear one
```

The exponent is pinned down by how much order sizes *vary*, not by how many
orders there are, and a fit that returned 1.50 without comment would hand a
scheduler a number the data never supported. The thresholds behind each
warning are module constants in `slippage.calibration`, stated so they can be
argued with. The tests also check that the reported standard errors are honest:
over repeated synthetic samples the 95% interval contains the true exponent
between 89% and 99% of the time.

`samples_from_orders` turns executed `Order`s into calibration inputs, measuring
cost against arrival so that delay does not leak into the impact coefficients.
Its time unit is a required argument: a rate coefficient fitted per hour and
used per day is wrong by a factor of the trading day's length, silently.

## Optimal execution

`optimal_trajectory` solves the Almgren–Chriss problem: execute `X` shares over
`N` intervals minimising `E[cost] + λ·Var[cost]` under linear impact. The
holdings are `x_j = X sinh(κ(T − t_j)) / sinh(κT)`, with the urgency `κ` solved
exactly from the discrete relation `2/τ²·(cosh κτ − 1) = λσ²/η̃`.

[`examples/almgren_chriss.py`](examples/almgren_chriss.py) runs the paper's
own example — a million shares over five days — across risk aversions:

```
  lambda   kappa  half-life    E[cost]         sd    day 1
       0   0.000       infd    662,500  1,040,673  200,000
   1e-07   0.195      5.14d    670,057    961,623  242,118
   1e-06   0.607      1.65d    911,227    603,431  458,044
   1e-05   1.727      0.58d  1,845,211    171,712  822,132
```

The paper reports `κ ≈ 0.6/day, so κT ≈ 3` at `λ = 10⁻⁶`, which the test suite
checks. The risk-neutral row is the straight line and its cost is the hand
figure `½γX² + εX + η̃X²/T`. Moving from `λ = 10⁻⁷` to `10⁻⁶` cuts the standard
deviation of cost by 37% for a 36% rise in its expectation; that exchange rate is
the efficient frontier, and `efficient_frontier` traces it.

Two computational choices:

- **Moments are summed over the schedule, not taken from the closed form.** The
  published expressions for `E` and `V` contain `sinh(2κT)` and overflow for an
  impatient trader long before the trajectory does. They are kept as
  `closed_form_moments`, and the tests require the two to agree to nine
  significant figures wherever both are finite.
- **Holdings switch to a ratio of exponentials above `κT = 20`**, so a
  schedule with `κT` in the thousands is still finite. The tests check the two
  forms against the textbook ratio on both sides of the switch.

`half_life_sensitivity` gives the elasticity of the half-life `1/κ` to each
input by differentiating the discrete relation. In continuous time they are
exactly `−½` for risk aversion, `−1` for volatility and `+½` for temporary
impact; with one-day intervals the example gives −0.485, −0.970 and +0.511,
and the tests check both the analytic values against finite differences and
their convergence to the limits as the interval shrinks.

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
