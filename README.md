# slippage

An optimal execution and transaction cost analysis engine.

The library answers two questions that sit either side of a trade:

- **After the fact** — what did this order actually cost, and where did the cost
  come from: the delay before it reached the market, the impact of trading it,
  or the part that never got done?
- **Before the fact** — given a market impact model and a tolerance for risk,
  how should the remaining quantity be spread over the trading horizon?

Under both sits a validated numerical core: implementation shortfall that is
checked against its own direct formula on every call, impact models fitted with
honest standard errors, the Almgren–Chriss trajectory checked against the
paper's published figures, and a constrained scheduler checked against
exhaustive enumeration. Each section below states what was measured and how.

See [ROADMAP.md](ROADMAP.md) for the build order.

## Install

```bash
pip install -e ".[dev]"
python -m pytest
```

Python 3.10 or later. The only runtime dependency is NumPy.

## Command line

Installing the package provides a `slippage` command. `sample-data` writes a
synthetic book, so every step below runs without real data:

```bash
slippage sample-data --out book
slippage tca --orders book/orders.csv --fills book/fills.csv --bars book/bars.csv \
    --fees-per-share 0.0005 --group-by side
```

```
60 orders, paper notional 205,208,433, shortfall -62,784 (-3.06 bps)

             orders        delay      trading  opportunity   commission         fees      total
-----------------------------------------------------------------------------------------------
buy              32         1.38       -12.29         0.14         0.19         0.05     -10.53
sell             28         5.71         7.51        -0.96         0.26         0.06      12.59
-----------------------------------------------------------------------------------------------
all              60         2.78        -5.89        -0.21         0.21         0.05      -3.06
(basis points of paper notional)

outliers at a modified z-score of 3.5:
  ORD0046    SYM06    sell   -147.34 bps   z = -5.1
  ...
```

```bash
slippage schedule --quantity 1000000 --horizon 5 --periods 5 --volatility 0.95 \
    --gamma 2.5e-7 --eta 2.5e-6 --epsilon 0.0625 --risk-aversion 1e-6 \
    --max-trade 300000 --lot-size 1000
```

```
method: dynamic programme, lot 1000
period     start         trade     remaining
     0         0       300,000       700,000
     1         1       300,000       400,000
     2         2       196,000       204,000
     3         3       118,000        86,000
     4         4        86,000             0

expected cost 756,873, standard deviation 794,266
```

`schedule` uses the closed form when impact is linear and nothing constrains
it, and the dynamic programme otherwise. Here the cap slows the first two days
and so *lowers* expected cost (from 911,227 unconstrained) while raising
risk. The objective the trader asked to minimise, `E + λV`, rises from
1,275,356 to 1,387,732, which is what a binding constraint must do. `frontier`
prints the efficient frontier. Every subcommand takes `--json`, and bad input
exits with status 2 and a message naming the file and line.

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

## Reporting across a book

`build_report` decomposes every order and aggregates by any key. Every
aggregate is a ratio of sums: a group's cost in basis points is its total cost
over its total paper notional. An average of per-order basis points would give
a hundred-share order the same weight as a million-share one, and its
components would stop adding up to its total once orders of different sizes
mix. With ratios of sums the components add up at the order, group and book
level, and the tests check all three.

Outliers are screened by the modified z-score, `0.6745 (x − median) / MAD`,
with Iglewicz and Hoaglin's threshold of 3.5. An ordinary z-score lets large
outliers hide: they inflate the standard deviation they are measured against.
In the test suite, two values fifty times the typical size score under 3 on a
classical z-score and over 3.5 on the modified one. An order whose fills are
made 3% worse is the first one the report flags.

`compare_to_model` sets each order's realised trading cost beside what a fitted
square-root law expected. The comparison is per executed share, so an order
that was cut short is judged on what it traded.

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

## Constrained schedules

The closed form needs linear impact and no constraints. `solve_schedule` solves
the same mean-variance problem by backward dynamic programming over
`(period, remaining lots)`, for any impact model and with per-period
participation caps, minimum trade floors and lot sizes.
[`examples/constrained_schedule.py`](examples/constrained_schedule.py) works a
million shares under square-root impact in trading rate, with and without a 5%
participation cap:

```
period     volume       cap      free    capped
     0  3,000,000   150,000   461,000   150,000
     1  2,000,000   100,000   215,000   100,000
     2  3,000,000   150,000   115,000   150,000
     ...
     9  2,000,000   100,000    13,000    42,000

objective E + lambda V: free 1,203,249, capped 1,694,069
```

The cap binds for the first six periods and the programme then eases off, rather
than spreading the order evenly, because holding risk still argues for
finishing early.

The solution is a **policy** over every state, not one path, so re-optimising
after fills go off plan is a lookup: `plan.trades_from(period, remaining)`
returns the best schedule for the rest of the horizon from wherever the order
actually is. Following the policy from the planned state reproduces the rest of
the plan exactly, which the tests check as time consistency.

Correctness is checked two ways that do not share code with the solver. On
small problems the objective is compared with exhaustive enumeration of every
possible schedule — for linear, square-root and convex impact, at three risk
aversions, and on Hypothesis-generated combinations of caps and floors,
including infeasible ones. On a 2,000-lot grid with slack constraints and
linear impact, every trade lands within one lot of the Almgren–Chriss closed
form, and the objective is never below it: the grid is a subset of the
continuum, so the programme can match the closed form but not beat it. Each
period is one vectorised minimisation over a states-by-trades matrix; the
2,000-lot, ten-period case solves in about a third of a second.

## Volume profiles, benchmark schedules and simulation

`estimate_profile` builds an intraday volume curve from historical sessions.
It averages each day's volume *fractions* instead of summing volume first: a
single rebalance day with fifty times normal volume, all at the close, would
otherwise put 85% of the profile in the last bucket. `twap_schedule`,
`vwap_schedule` and `pov_schedule` generate the standard benchmarks. Lot
rounding hands out leftover lots by largest remainder, so a schedule never
gains or drops a lot. A POV schedule that runs out of volume reports the
unfilled remainder instead of breaking its participation limit.

`simulate_costs` draws cost distributions under any impact model, on the
Almgren–Chriss price process. [`examples/strategy_comparison.py`](examples/strategy_comparison.py)
sells 500,000 shares in one session under four schedules on the same price paths:

```
strategy       mean        sd   95th pct  unfilled
TWAP        685,096   256,910  1,106,680         0
VWAP        817,545   249,927  1,227,047         0
POV 12%     842,096   217,690  1,198,577         0
optimal     686,781   250,213  1,097,370         0
```

The optimal schedule costs almost the same as TWAP on average and has the lower
95th percentile, which is the trade the risk aversion asks for. VWAP looks
worst on average, and the reason is the model, not VWAP. Almgren–Chriss assumes
liquidity is constant through the day, so trading heavily at the open and close
counts as pure cost. That concentration is the point of VWAP, because those are
the hours when the market can absorb it. The row shows what the assumption
costs.

The simulated cost is linear in the Gaussian price shocks. That has three
consequences worth stating:

- Simulated mean, variance, 95th percentile and expected shortfall all match
  their closed forms within Monte Carlo error, which is the simulator's main
  test.
- **Antithetic sampling makes the mean exact**, since each pair of paths
  averages to the deterministic cost, but it does little for the tail. Over 400
  repetitions of 10,000 paths it cut the standard error of the 95th percentile
  by 7% and of the expected shortfall by 10%. The examples turn it on because it costs nothing, not because it
  transforms tail estimates.
- `simulate_prices` produces the unaffected price path and every fill price.
  The tests rebuild each path's cost from those fills and check it against the
  direct computation.

## From fills to a schedule

[`examples/end_to_end.py`](examples/end_to_end.py) runs the whole pipeline. It
writes a 400-order synthetic book to CSV and reads it back, reports the book's
shortfall, fits the square-root law to the executions, and turns the fit into a
rate model for a new order worked under a 10% participation cap:

```
square-root law: Y = 0.512 +/- 0.088

recommended schedule for 993,000 SYM00 at a 10% participation cap:
  bucket  expected vol       cap     trade
    9:30     2,507,904   250,790   250,000  cap binds
   10:00     1,960,822   196,082   196,000  cap binds
   10:30     1,628,738   162,874   162,000  cap binds
   11:00     1,370,646   137,065   118,000
   ...
   15:30     2,439,236   243,924    12,000
```

The synthetic fills pay impact on the size done *so far*, so an order's
realised cost is its peak impact times a path factor. Across this book the path
factor averages about 0.92, so the 0.7 used to generate the book implies a
fitted prefactor near 0.64; the fit lands within 1.5 standard errors of it. The
test suite folds the path factor into the regression and recovers the
generating 0.7 directly.

## Layout

| module | contents |
|---|---|
| `types`, `series`, `benchmarks`, `costs` | orders, fills, bars, benchmark prices and the sign convention |
| `shortfall`, `report` | implementation shortfall, fill attribution, book-level TCA and outliers |
| `impact`, `calibration` | impact models, schedule costs and fitting them to executions |
| `execution` | Almgren–Chriss trajectories, the efficient frontier, half-life sensitivities |
| `scheduling` | the constrained dynamic programme and its re-optimisation policy |
| `volume`, `simulate` | volume profiles, TWAP/VWAP/POV schedules and Monte Carlo costs |
| `io`, `cli`, `synthetic` | CSV input and output, the `slippage` command, synthetic books |


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
