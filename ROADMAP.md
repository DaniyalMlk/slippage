# Roadmap

An execution-analytics library in two halves: measuring what a completed order
actually cost, and deciding how a pending order should be scheduled. The phases
below build the measurement side first, because a scheduler you cannot score is
untestable.

## Phase 1 — Order and execution data model

- [x] `Side` convention fixed once, in one place, with a signed multiplier
- [x] `Order`, `Fill` and `Bar` value types with validation at construction
- [x] Benchmark prices: arrival, interval VWAP, interval TWAP, close
- [x] Signed cost in currency and basis points against an arbitrary benchmark
- [x] Packaging, type checking, linting and continuous integration

## Phase 2 — Implementation shortfall

- [x] Perold decomposition: delay, trading and opportunity cost
- [x] Explicit costs (commission, fees) kept separate from implicit, with the half-spread split out of trading cost
- [x] Attribution that sums exactly to the total, asserted as an invariant
- [x] Per-fill attribution and participation statistics
- [x] Worked example reproducing a decomposition by hand

## Phase 3 — Market impact models

- [x] Linear temporary and permanent impact (Almgren–Chriss parameterisation)
- [x] Power-law impact with the square-root special case
- [x] Calibration of impact coefficients from realised executions
- [x] Goodness-of-fit diagnostics and identifiability warnings
- [x] Table-driven tests against analytically integrable cases

## Phase 4 — Optimal execution trajectories

- [x] Closed-form Almgren–Chriss trajectory for linear impact
- [x] Limiting cases verified: risk neutrality collapses to a straight line
- [x] Expected cost and variance of a schedule in closed form
- [x] Efficient frontier of execution over risk aversion
- [x] Half-life of the trade and its sensitivity to the model parameters

## Phase 5 — Constrained scheduling

- [x] Discrete dynamic programme over remaining quantity and time
- [x] Participation caps, minimum trade floors and lot sizes
- [x] Agreement with the closed form when constraints are slack
- [x] Adaptive re-optimisation from a partially executed state

## Phase 6 — Volume curves and simulation

- [x] Intraday volume profile estimation from historical bars
- [x] TWAP, VWAP and percentage-of-volume schedule generators
- [x] Fill simulator combining impact, drift and volatility
- [x] Monte Carlo cost distributions with variance-reduction where it helps
- [x] Simulated costs checked against the closed-form mean and variance

## Phase 7 — Reporting and interface

- [x] Cost attribution report across a set of orders
- [x] Outlier detection and comparison against a fitted impact model
- [x] Command line interface over the whole pipeline
- [x] End-to-end worked example from raw fills to a scheduling recommendation
- [x] README covering usage, conventions and the design decisions that mattered

## Phase 8 — Installable from a package index

- [x] Distribution name distinct from the taken one, import name unchanged
- [x] SPDX licence expression, with the licence file inside both artefacts
- [x] `--version` on the command line, agreeing with the packaged metadata
- [x] Metadata tests: version agreement, typing marker, entry points, licence
- [x] Tag-driven release with a version guard and no stored credential
- [ ] First release on the index

## Phase 9 — Decomposition without a tape

- [x] Shortfall from order totals, with the identity and its cross-check there
- [x] `implementation_shortfall` reduced to a wrapper over it
- [x] Property test driving both routes over randomly generated orders
- [x] The timestamp-independence the totals path relies on, asserted not assumed

## Phase 10 — Which part of the impact came back

`fit_permanent` documented its second argument as the moves that persisted after
each order completed and nothing in the library measured them. The report had no
post-trade measurement at all, so a total cost against arrival could not be split
into the part a slower schedule avoids and the part it does not — which is the
only thing the impact model is for.

- [x] Mark-outs at a set of horizons after completion, signed for the side
- [x] The move over the order window split into persisted and reverted, with the
      identity between them asserted rather than assumed
- [x] A benchmark subtracted with a beta, and the result recording whether it was
- [x] Aggregation across orders with standard errors per horizon
- [x] An `observed` flag per mark-out, because `price_at` clamps to the final
      close and would otherwise report no data as no movement
- [x] An exponential fitted to the curve by separable least squares, refusing a
      half-life where the curve does not decay
- [x] `permanent_moves_from_orders`, the pair `fit_permanent` always wanted
- [x] A `markouts` command that says in its own output when no benchmark was given
- [x] `synthetic.decaying_book`, because the existing generator keeps impact out
      of the prints on purpose and a mark-out reads impact off the prints
- [x] The comparison measured over 25 books in a worked example that runs in CI

The measurement that matters is not the mark-out, it is what removing the market
does to it. Twenty-five books of 120 orders, 4bps of permanent impact and 6 of
temporary decaying with a 208-second half-life:

| figure | truth | market in | market out |
|---|---|---|---|
| impact at completion, bps | 10.00 | 9.99 (off 0.80) | 10.02 (off 0.35) |
| reverted fraction by an hour | 0.600 | 0.600 (off 0.098) | 0.609 (off 0.046) |
| fitted half-life, seconds | 208 | 244 (off 90) | 222 (off 34) |
| fitted permanent impact, bps | 4.00 | 3.68 (off 0.88) | 3.90 (off 0.48) |
| standard error at an hour, bps | — | 2.35 | 0.82 |
| `fit_permanent` t-statistic | — | 1.74, below 2 in 16 of 25 | 4.55, below 2 in 0 of 25 |

Both are unbiased, which is why leaving the market in is dangerous rather than
obviously wrong: the means are right and only the spread gives it away. The
t-statistic row is the one that settles it — with the market in, the permanent
coefficient cannot be established in most books built with permanent impact in
them by construction.

Two assertions written from intuition were false when measured, and both are
recorded rather than loosened. The benchmark-adjusted point estimate is *not* the
closer one on a single seed — 9.51 against 9.91 for a truth of 10 on the seed the
tests use, which is what standard errors of 0.40 and 1.69 do. And the reverted move
is *not* measured more precisely than the persisted one despite covering a shorter
interval: 1.4230 against 1.4228 at fifteen minutes, because every order in a book
shares one market path so cross-order errors are not independent draws.

The benchmark subtraction is on simple returns: exact at a beta of one, where the
stock's price is the benchmark's times a constant, and first-order elsewhere — 0.04
basis points against a market that wandered thirty. Using the wrong beta leaves 8.

One defect the tests found. The headline reverted fraction was taken from the last
horizon asked for, which is routinely past the end of the bars, so it was a NaN.
`json.dumps` writes that bare and `json.loads` reads it back without complaint, so
a round trip does not catch it and a strict parser at the far end rejects the whole
document.

## Phase 11 — Liquidating a basket

The solver handled one name. Almost nothing is executed one name at a time, and a
basket is not a collection of single-asset problems: risk on the remaining book is
a quadratic form in the covariance matrix, and impact is a matrix too, so the
schedules couple even where the returns do not.

- [x] A basket problem with signed holdings, temporary and permanent impact as
      matrices, and a covariance matrix
- [x] The exact eigen-solution — square root of the effective temporary impact
      matrix, diagonalise, one scalar problem per direction — with each direction's
      risk per unit of impact, urgency and half-life reported
- [x] Cost and variance of any schedule by direct summation, sharing no arithmetic
      with the solver, so optimality is checkable rather than assumed
- [x] The same basket solved leg by leg, scored against the full matrices rather
      than against its own assumptions
- [x] An efficient frontier, and the riskiest direction the basket starts flat in
- [x] A `basket` command taking a holdings file and a covariance matrix, with the
      matrix parser naming the line
- [x] Refusals for an asymmetric matrix, a singular impact matrix, a ragged row and
      a schedule that does not liquidate, each naming what is wrong

The one-asset reduction is the test that carries the weight. A one-asset basket
reproduces `optimal_trajectory` to machine precision at every risk aversion from
zero to 1e-2, and the two routes share no arithmetic: one evaluates a `sinh` at a
closed-form `kappa`, the other takes a matrix square root, an eigen-decomposition
and two changes of basis. An uncorrelated diagonal basket reproduces two
independent single-asset solutions, which is the same statement a dimension up.
Optimality is checked separately by thirty random perturbations that preserve the
total, none of which is cheaper.

That reduction found a real error rather than confirming the code. The permanent
impact of a complete liquidation is not `X' Gamma X / 2`: each trade is not charged
for its own impact, so it is that less `sum_k n_k' Gamma n_k / 2`, and the matrix
the schedule is charged against is the effective one — the same correction as the
scalar `eta_tilde`. Computing the cost with the raw temporary matrix overstated it
by two parts in a thousand, small enough to read as a rounding difference against
the scalar solver and large enough to mean the two were not the same model.

Measured, on a long/short pair of 100,000 shares each correlated at 0.9 over a day
in 20 intervals, against solving each leg separately and scoring both against the
full matrices: the joint solution saves 23.8% of the objective with identical legs,
12.1% when the short leg is four times as expensive to trade, 6.1% at ten times,
and exactly 0.0% when the returns are uncorrelated and the legs identical. The last
figure is the check that the saving is the correlation rather than the solver.

Two results reverse the natural assumption and are in the command's output rather
than only in the documentation.

The joint solution is riskier moment to moment, not safer: its peak one-interval
variance is 1.19 times the independent solution's on the symmetric pair and 1.10 on
the illiquid one. A hedged pair is cheap to hold, so the optimal schedule holds it
longer and pays less impact. It is the cheaper schedule at the same risk aversion,
not the calmer one.

And where the leg-by-leg solution is actually wrong is the shape, by an order of
magnitude rather than by the 12.1%. It works each leg at the pace that leg's own
liquidity justifies, so the liquid leg finishes first and the book is left outright
mid-trade: a net exposure of 18,961 shares of a 100,000-share pair against the
joint solution's 2,025, a factor of 9.4.

Two metrics were wrong on the first attempt and both failed in the same way, by
including a quantity every schedule of the same basket shares. A peak interval
variance that included the first interval read exactly 1.000 for every comparison,
which looks like agreement rather than like a broken metric; a peak exposure that
included the starting holdings read exactly 1.00 for the same reason. Both now
exclude it. And the exposure ratio on a symmetric pair is one rounding error over
another — 0.43, which reads as a finding — so it is refused, with the guard
relative to the basket's own size, because the residue is of order eps times that.
