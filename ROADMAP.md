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

## Phase 12 — Impact that decays at a rate

- [x] A decay kernel with the permanent floor inside it: exponential
      (Obizhaeva-Wang) and power law (Bouchaud), with both existing descriptions
      of impact as its limits
- [x] The cost of a schedule as the quadratic form it is, checked against walking
      the price path trade by trade
- [x] No-price-manipulation enforced rather than documented, with the offending
      round trip returned instead of a boolean
- [x] The cost-minimising schedule as a linear solve, checked against a numerical
      optimiser on the same problem
- [x] The mark-out curve the model predicts, so `markouts` output is an input here
- [x] A kernel read back off a measured decay, with the size of what that loses
      derived rather than waved at
- [x] A command-line entry point

The result worth having is that nothing here needs an optimiser. Writing the cost
as `n' K n / 2` makes the cost-minimising schedule a linear solve, and for an
exponential kernel its answer is a block, a constant rate and a block — the shape
Obizhaeva and Wang derive in continuous time, arriving here without anything in
the code imposing it. Against SLSQP on the same problem the closed form is 0.27%
cheaper at low resilience, because the optimiser stops early.

Two measurements that the natural guess gets wrong.

The gain from scheduling well is not monotone in resilience. Both limits give
nothing: decay fast enough and the cost matrix is diagonal, so a constant rate is
already optimal; decay slowly enough and it is constant, so every schedule ties.
The peak is in between, it depends only on resilience times horizon, and it drifts
with how finely the horizon is cut — 3.57% at four slices to 11.46% at
sixty-four, because the solution wants two instantaneous blocks.

And a mark-out recovers the resilience exactly while barely recovering the
amplitude. The horizon factors out of every term of an exponential kernel at once,
so the curve after the order is a clean exponential at the right rate whatever
schedule produced it. The level it decays from is not: for a uniform schedule of N
slices over horizon T the amplitude is the kernel's transient impact times
`(1/N)(1 - e^-rhoT)/(1 - e^-rhoT/N)`, which is 0.589 over two half-lives at eight
slices and 0.541 in the continuous limit. Reading a kernel off a worked order's
mark-out therefore understates its transient part by about 40%.

Being a decreasing kernel is not sufficient for admissibility, which is the trap
here. A kernel that falls almost flat and then drops off a shoulder is
non-negative, bounded and strictly decreasing at every lag, and admits a round
trip costing -2.35 over twelve slices whose largest is one share. Complete
monotonicity is the property that works. The guard's threshold is relative,
because the cost matrix restricted to zero-sum directions is singular by
construction and on a power-law kernel its smallest eigenvalue comes out negative
at 1e-16 of the scale — a test against zero would reject an admissible model.

Five defects, four of them in the tests. The gap between a power law and a matched
exponential was asserted to shrink throughout and in fact peaks, because both are
heading to zero; the scheduling peak was read off every grid at one grid's peak;
the round trip's cost was quoted at unit norm while the function reports it at
largest-slice-one; and a zip against a shifted copy was written without accounting
for the shift. The one in the code: `impact_path` returned an entry past the last
slice, called it the impact at completion, and had it one interval late —
understating completion impact by exactly one period of decay, 16% on the grid it
was checked on, which is too small to look like a bug and too large to ignore.

## Phase 13 — Is a schedule worth adapting, and to what?

- [x] A liquidity regime carrying its own impact model and volatility, with a
      Markov chain over regimes that the trader observes before trading into
- [x] The optimal adaptive policy by backward induction, exact rather than
      gridded, because the value function stays quadratic in the remaining
      inventory — one coefficient per regime per period
- [x] The terminal liquidation constraint entering as a zero in the reciprocal
      recursion rather than as a special case in the algebra
- [x] One regime reproducing `optimal_trajectory`: holdings to 2.3e-16 of the
      order size, objective to 1.7e-16 relative, at every risk aversion tested
- [x] The best *deterministic* schedule facing the same chain, as the only
      honest thing to measure the policy against
- [x] A Monte Carlo over drawn regime paths, sharing no arithmetic with the
      recursion, agreeing with it inside its standard error
- [x] A command-line entry point reporting the saving and the policy table
      beside the static one

The objective had to change, and that is worth stating rather than hiding. What
is minimised here is `E[cost] + lambda E[sum sigma^2 tau x^2]`, not `E[cost] +
lambda Var[cost]`. For a deterministic schedule those are the same number — the
variance of the total cost *is* that sum — which is why the single-regime
agreement with the closed form is exact. For an adaptive schedule they are not,
because the inventory becomes random and the variance of the total picks up a
term the sum of conditional variances does not have. A variance of a total is
not a sum of per-period pieces, so no dynamic program optimises it, and the
running penalty is the time-consistent substitute.

Four measurements, and three of them came out against the guess.

**Both ends of the persistence range are worth nothing, for the same reason.** At
a persistence of one the chain never moves and the static schedule can use the
starting regime. At zero it strictly alternates, which is just as predictable.
Both come out at zero to 4e-16 of the objective. Variability is not uncertainty,
and only uncertainty is worth reacting to. Nor is the peak in the middle: scanning
at 0.001 it is 37.30% at **0.182** — well onto the mean-reverting side, where a
regime says something about the next period and almost nothing about the one
after. That location holds between 0.171 and 0.187 across ten, twenty and fifty
periods and across impact ratios of two, five and ten.

**Adapting is worth most to a trader who does not care about risk**, which is the
reverse of the intuition that adaptivity is a risk-management device. The gain
runs 37.64% at zero risk aversion, 37.30% at 2e-06, 26.92% at 1e-04 and 12.91% at
1e-03, monotonically down, because a risk penalty is charged on inventory
whatever the regime and the more of the objective it accounts for the less of it
the regime can move.

**It is liquidity worth adapting to and not volatility.** Two regimes differing
only in volatility, by a factor of five, at a persistence of 0.8, are worth
0.115%. Two differing only in impact by that same factor are worth 24.03% — two
hundred and nine times as much. Reacting to a volatility spike is close to
worthless here; reacting to a liquidity one is the whole effect.

**And the saving comes from waiting rather than from hurrying.** At the tenth of
twenty periods the policy trades 1.906 times the static fraction when liquid and
0.589 times it when illiquid, which is the expected shape. But in the *first*
period it trades 0.755 times the static fraction even in the liquid state,
because a static schedule starting liquid front-loads into the cheap trading it
forecasts and an adaptive one does not have to. Which is also why the gain is
larger from the illiquid start at a persistence of 0.8 — 24.64% against 24.03% —
with the ordering reversing by 0.9, where the liquid start gains 15.06% against
11.99%.

Fewer than three periods cannot gain anything, and that is provable rather than
small: the first period's regime is known to the static schedule too, and the
last period has no decision in it, so a two-period problem reveals nothing before
its only choice. The saving comes back as exactly `0.0`; the first non-zero one is
1.73% at three periods.

Three defects, all in what was asserted. The static schedule facing the chain
was asserted to lie between the two pure schedules and lies outside both —
129,346 shares in the first of twenty periods against about 51,000 for either
pure schedule — because theirs have constant coefficients and are nearly uniform
while this one is steeply front-loaded from a liquid start and back-loaded from
an illiquid one. The gain's denominator was guarded against being zero and tested
at a quantity of 1e-100, where it is 1e-230 and divides perfectly well; the only
route to zero is underflow at around 1e-150, which is what the test now uses. And
`covers` read an infinite standard error as agreement, so a single draw — the
weakest evidence available — was reported as the strongest.

## Phase 14 — Rest or cross, and what the choice is actually worth

- [x] The fill probability of a resting order, which is a running-minimum
      probability and not a terminal one
- [x] The mid conditional on the fill and on the miss, in closed form, because
      that is the adverse selection
- [x] The expected cost against the arrival mid, with the half spread, the
      taker fee and the maker rebate in it
- [x] Its variance, so the choice is not made on the mean alone
- [x] The drift case, which is what "picked off" means, with a number on it
- [x] Simulation of the same quantities, and an honest account of the bias
      discrete monitoring puts in
- [x] A command-line entry point

`benchmarks` scores a finished order and `reversion` measures what happened
after it. Neither addressed the decision taken first, which is whether to cross
now or rest and hope. It has a closed form under a Brownian mid, and three of
the results are worth more than the pricing.

**The factor of two.** The fill probability is a statement about the running
minimum, and with no drift it is `2 Phi(-delta / sigma sqrt(T))` — exactly
twice the chance of merely *ending* below the limit, measured as 2.000000 at
four distances. Reading the terminal distribution instead halves the answer and
does so quietly, because the result is still a plausible probability.

**Conditional on filling, the expected mid at the horizon is exactly the limit
price.** `E[X_T 1{touch}]` is `2 b Phi(b/s)` and `P(touch)` is `2 Phi(b/s)`, so
the ratio is `b` — the barrier itself, to twelve digits at four distances. The
adverse selection therefore cancels the whole of the apparent saving: against
the *arrival* mid a filled order saved `delta`, and against the *terminal* mid
it saved nothing at all. Which benchmark is being used is the entire content of
"a passive fill was worth something", and that is a result about `benchmarks`
as much as about this module.

**So there is no frontier.** The whole driftless expected cost is
`(h + f)(1 - p) - r p`, with the distance appearing nowhere else — checked
against the general formula to thirteen digits at six distances. Every term
that referenced the distance cancelled against the adverse selection. And the
standard deviation rises with the distance too: over a day at 20% volatility
with a five basis point half spread, the mean goes from -0.94 to +6.00 basis
points while the deviation goes from 16.4 to 126.0. Both monotone, both the
same way, so resting deeper is worse on both counts and the answer is always
the tightest price the book allows. `frontier()` is named for what a caller
expects and will not find.

**What being picked off costs.** The one thing that changes the answer is a
drift. At half a standard deviation of drift per horizon, resting one standard
deviation out costs 62.3 basis points against 0.45 at the touch; at one
standard deviation of drift, 125.6 against 2.69 — a factor of 47. Out at one
standard deviation the cost is linear in the drift (117, 127 and 130 basis
points per standard deviation of it across three intervals) and at the touch it
grows faster than linearly from a base of -0.94.

**Simulating a barrier is biased, and the correction only fixes half of it.**
A path checked at its own time steps misses the excursions between them, so the
fill frequency comes out low: 3.60% at 250 steps, 1.89% at 1,000, 0.91% at
4,000 and 0.45% at 16,000, with successive ratios of 1.91, 2.08 and 2.00 — the
square-root rate. Moving the barrier by `0.5826 sigma sqrt(T/steps)`, the
Broadie-Glasserman-Kou continuity correction, brings the formula to within
-0.06%, +0.08%, +0.01% and +0.007% of the discrete simulation: between
twenty-three and seventy-eight times better, and at the noise floor of 200,000
paths.

It does *not* fix the mean cost, which was worth finding out rather than
assuming, since the mean is built out of the same barrier. The shifted
formula's mean is still 21.7, 11.9, 6.1 and 1.9 standard errors from the
simulated one at those four step counts: the bias in the mean lives in
`E[X_T 1{fill}]` rather than in the probability, and moving the barrier changes
that term the wrong way. A test on the probability can use a modest step count
and the correction; a test on the mean needs the steps.

Two defects, both in what was asserted.

Resting far out was asserted to lose to crossing immediately. It does not:
delay is free under a martingale, so `(h+f)(1-p) - r p` is below `h+f` for any
positive fill probability, and four standard deviations out the advantage is
4.4e-08 — vanishing and still positive. What makes resting lose is a drift, and
the loss then converges to the drift over the horizon, which is what the test
asserts now.

And the simulation function was first called `simulate`, which shadowed
`slippage.simulate` at package level, where both are re-exported. The suffix in
`simulate_placement` is not decoration.

## Phase 15 — A benchmark that moves, and the floor it sets

- [x] The tracking error of a participation schedule against the realised VWAP,
      in closed form
- [x] Split into the part a schedule controls and the part it cannot
- [x] The variance-minimising schedule derived rather than taken as a rule of
      thumb
- [x] A volume-uncertainty model fitted to the dispersion already measured, with
      the fit's own error reported
- [x] The trade-off against impact cost, so a schedule is chosen rather than
      asserted
- [x] What an arrival-price schedule costs when the benchmark is VWAP, in both
      directions
- [x] A command-line entry point that leads with the floor

Five schedulers in this library — `execution`, `scheduling`, `basket`,
`transient` and `adaptive` — answer one question in different settings: how to
trade so the average price beats the **arrival price**. That benchmark is a number
fixed before the first share trades, so the risk is in the position still held,
and risk aversion front-loads.

Most institutional orders are scored against the interval VWAP, which is a
weighted average of the *same prices the order trades at*. The benchmark moves
with the market, and the exposure is not the position but the difference between
our participation and the market's. `benchmarks.score_order` measured the result
after the fact and `volume.vwap_schedule` sliced a profile; nothing optimised
tracking error and nothing said what part of it is unavoidable.

**The algebra is two lines and both consequences are structural.** With ``u`` our
share of the order in each bucket, ``w`` the market's realised share and both
summing to one, a driftless walk gives
``slippage = sigma sum_k (u_k - w_k) b_k`` — the arrival price cancels. So a
schedule matching the realised curve has **zero** slippage path by path, which no
arrival-price schedule can manage against its own benchmark; and volume
uncertainty is the only obstacle, because ``u`` is chosen before ``w`` is known.

Taking variances splits it exactly into ``(u - mu)' C (u - mu) + trace(C S)``. The
second term has no schedule in it, so it is a floor. The first is a positive
quadratic form minimised at ``u = mu``, which makes the expected volume profile the
variance-minimising schedule — a theorem rather than the rule of thumb it is
usually quoted as, and one that holds for any positive-definite ``C``, so it does
not depend on the price model beyond the walk having no drift. Three hundred
random perturbations confirm it, and the schedule term at ``mu`` is exactly 0.0.

**Where in a bucket its price is read makes no difference at all**, which is one
fewer arbitrary parameter than the model appeared to need. The choice adds a
constant to every entry of ``C``, and both terms are orthogonal to a constant: the
schedule deviation sums to zero and the covariance's rows sum to zero because the
shares sum to one. The two conventions agree to twelve significant figures, and
adding 1000 to every entry of ``C`` — six orders above its own entries — moves the
floor by 1.2e-15. The first draft of the docstring claimed the choice moved the
level and only left the optimum alone; it moves neither.

**`VolumeProfile.dispersion` has been estimated since volume profiles were added
and read by nothing.** It is the input here. Shares live on a simplex, so their
covariance cannot be diagonal, and a Dirichlet is the one-parameter family with
the structure the constraint forces. Using the dispersions alone and dropping the
negative correlations puts the floor **56.4% too high**, at every concentration —
both quadratic forms scale alike, so the overstatement is a property of the
profile's shape rather than of how uncertain it is. A desk told its irreducible
error was 12.6 basis points when it was 8.1 would accept schedules it should
refuse. The fit also reports its own worst per-bucket miss, because one parameter
cannot in general match a vector of them.

**The frontier's ends are known before the solve, which is what makes them a
check.** No weight on tracking error recovers `twap_schedule` to a part in ten
thousand — an equal slice minimises a convex temporary cost regardless of where
the volume is. A large weight recovers the volume curve with the tracking error at
its floor. Between them impact rises from 82.0 to 91.2 basis points as tracking
error falls from 11.64 to 8.05.

**The number a desk needs separates two things usually lumped together.** A
front-loaded arrival-price schedule tracks VWAP at **47.8** basis points against
the volume curve's 8.05 — an excess of **4.93 times the floor**. Risk aversion
front-loads against a fixed benchmark and does the opposite against this one, so
the two objectives genuinely conflict. But TWAP, which ignores the volume curve
entirely, is only **0.45 floors** worse, which is inside what a desk can measure.
So almost all of the benefit is in not front-loading, and matching the curve
exactly is the remainder — worth knowing before building a volume forecast.

A defect in the test rather than the module, recorded because the symptom pointed
the wrong way. The validating simulation wrote its walk as
``cumsum(eps) - eps / 2``, which has ``Var(b_k) = k - 3/4`` rather than
``k - 1/2``, and the closed form read 10% high at z = -44 on two of three
schedules. The formula was right.

## Phase 16 — A schedule that can hold a view

- [x] A per-period price forecast, validated, in price units per share
- [x] The objective written in remaining holdings, so the order constraint
      holds by construction and the forecast term telescopes to a diagonal one
- [x] The optimum by one pass of the Thomas algorithm on a tridiagonal system
- [x] Reduction to the Almgren-Chriss closed form at a zero forecast
- [x] The first-order conditions checked to vanish
- [x] The response kernel in closed form, as the Green's function of the same
      operator, with the same urgency
- [x] The value of a forecast in closed form, and the two consequences of its
      being a quadratic form
- [x] Round trips detected and reported rather than priced
- [x] The fixed per-share cost's schedule-independence, and where it fails
- [x] The proportional-tilt heuristic measured against the optimum
- [x] A command-line entry point leading with the saving and what a tilt
      recovers

`execution.optimal_trajectory` assumes the price is a martingale, so the only
reason to trade early is risk and the schedule can depend on the order and the
stock but never on a view. This phase adds the view.

The objective stays quadratic, which is why this is a solve and not a search.
Writing the schedule as the remaining holdings `x_1 .. x_{N-1}` with `x_0 = X`
and `x_N = 0` makes the "everything must trade" constraint hold by construction,
and the forecast term **telescopes**:

```
sum_k n_k D_{k-1}  =  sum_{k=1}^{N-1} x_k mu_k
```

The reading is exact rather than a convenience. The shares traded in period
`k + 1` or later are precisely the `x_k` still outstanding after period `k`, and
every one of them pays period `k`'s drift. So the forecast enters diagonally,
the Hessian stays tridiagonal, and the first-order conditions are a linear
system.

### Three things that fall out of the algebra

**The response kernel is the Green's function of the Almgren-Chriss operator.**
The conditions are `a (2I - S) x + c x = -mu`, whose homogeneous solutions are
the `sinh` profiles that problem already solves for, so the inverse is known in
closed form and is built from the *same* urgency `kappa`. The solver agrees with
it to between 3e-15 and 2e-14 relative, at risk aversions from zero up to
`kappa T = 5.7`. The smoothing a forecast receives is not a new parameter.

It is also not an exponential decay, which is the shape to expect. At a zero
risk aversion the `sinh` degenerate to their arguments and the kernel is exactly
**triangular** — a tent, the Green's function of a discrete Laplacian. A spike in
the middle of twenty periods moves the holdings by 46, 93, 139, 186 ... 476
shares on the way in and symmetrically out, where a geometric decay would have
given 476, 306, 196, 126.

**The value of a forecast is also closed form**, `mu' G mu / 2`, agreeing with
the solved objectives to 1.4e-12. Being a quadratic form gives two consequences
that need no measurement. The value is **quadratic** in the forecast, confirmed
to twelve figures. And it is therefore **identical for a forecast and its
negative**: a signal saying "hurry" and one saying "wait" are worth exactly the
same, to twelve figures, though the schedules they produce move in opposite
directions. That is not what anybody expects of a trading signal.

**The final period's drift cannot move anything**, because by then there is
nothing left to trade, and `mu_N` genuinely never appears in the system.

### Three measurements that came out against the guess behind them

A million shares over one day in twenty periods, volatility 0.9 dollars a share
per root day, `eta = 2.5e-6`, `gamma = 2.5e-7`, risk aversion 2e-6 — where one
period's volatility is 20.1 cents.

**The optimum is far more reluctant to round-trip than expected.** With a
forecast reversing halfway, the unconstrained optimum first asks for a negative
trade at a drift of **97.7 cents a share a period**, which is 4.86 times a
period's volatility. Impact is quadratic and the forecast is linear, so buying
shares back has to overcome a cost rising faster than the reason to. The guess
before measuring was a fraction of one volatility. When it does happen it is
reported, because `LinearImpact.temporary` refuses a negative rate — correctly,
since a negative rate is a different trade and not a smaller cost.

**The forecast is worth very little and a tuned heuristic captures nearly all of
it.** Against 2 cents a period decaying linearly to zero, the optimum beats the
forecast-blind schedule by 0.0801 basis points of the order's value at fifty
dollars a share, and trading in proportion to the signal recovers **99.29%** of
that.

**Except on a sharp signal.** On a single-period spike the same tuned heuristic
recovers only **63.3%**, because the optimum spreads the response and a
proportional tilt cannot. On an alternating forecast it recovers 99.17% and on
the reversing one 90.6%. So the case for solving this exactly is sharp isolated
signals and it is weak otherwise, which is worth knowing before building the
solver into anything.

### A constant, and the cost that stops being one

The round-tripping branch computes its objective from the quadratic form
directly, since `schedule_cost` will not price a negative rate, and the first
version dropped that function's own constant — `epsilon X + gamma X^2 / 2`. On
this problem that is exactly 125,000, which made the objective incomparable with
the blind one while looking entirely plausible.

The `epsilon` half of it then deserved its own field. A fixed per-share cost is
`epsilon X` for every schedule that trades one way, so it drops out of the
optimisation entirely — which is what makes the quadratic solve the right problem
rather than an approximation to it. A round trip breaks that, because shares
traded backwards and then forwards again each pay it, so the objective
understates the real cost and the optimum is no longer the real optimum.
Reported, rather than quietly wrong.

## Phase 17 — The size at which the cost consumes the reason

Every phase above answers "given this size, how should it be traded?". None
answers the question before it: how large can the order be at all? A strategy's
edge is a number of cents a share, executing spends some of it, and past a size
executing spends all of it. That size is the strategy's capacity and it is the
number an allocation decision turns on.

- [x] Break-even size with the horizon held fixed, in closed form, because the
      objective is exactly quadratic in the size
- [x] Break-even size with the participation rate held fixed, where the horizon
      moves with the size and the solve is a bisection on the library's own cost
- [x] The alpha floor: the per-share cost no size can reduce at a given rate
- [x] An optimal horizon for a uniform schedule, and the statement that the
      optimal schedule has none
- [x] A capacity curve and the elasticity of cost per share to size
- [x] A command-line route that tabulates both conventions together
- [ ] A first release on the package index

### The two conventions disagree about what limits capacity

A model fixes the horizon. A trader quotes participation, which fixes the rate
and lets the horizon grow with the size. Those are different functions of size
and the difference is not a detail. At a fixed horizon the temporary impact per
share is `eta X / T` and cost per share is linear in the size. At a fixed
participation rate it is `eta rho V`, **constant in the size**, so it does not
bound capacity at all: it sets a floor on the alpha required, and the permanent
impact sets the size limit.

Measured as the log-log slope of cost per share against size: 1.0000 at every
decade with the horizon fixed, and 0.015, 0.129, 0.553, 0.914 across four
decades at 5% participation — zero where the floor dominates, approaching one
only once the permanent term has taken over. With the permanent impact removed
the fixed-participation cost per share is 0.03000000 at a million shares, at a
billion and at a trillion: the same eight digits, and capacity genuinely
unbounded. That is raised rather than returned as the ceiling, because the
ceiling is an argument.

### The alpha floor is the number to check first

At the worked parameters a 10% participation rate needs 5.50 cents a share
before any size is viable and a 25% rate needs 13.00, against an alpha of 5.00.
At those rates the strategy has no capacity whatsoever and no amount of patience
about the horizon changes it, because patience is what the rate has already
spent. The fixed per-share cost enters the floor too, so a half-cent commission
that looks negligible against a five-cent edge is a third of the floor at 2%
participation.

### The horizon that maximises net alpha does not depend on the order size

For a uniform schedule the objective is
`(alpha - epsilon) X - eta X^2 / T - gamma X^2 / 2 - lambda sigma^2 X^2 T / 3`,
in which every term carrying `T` carries the same `X^2`, so the optimum is
`sqrt(3 eta_tilde / (lambda sigma^2))` with no size in it. Measured at 500,000,
two million and twenty million shares across four orders of magnitude of risk
aversion: agreement to 1.3e-07 at worst. Not bit for bit, and the residual is
not slack in the invariance — the objective's magnitude goes as `X^2`, so near a
flat optimum the search's comparisons are decided by rounding and a fortyfold
range of size moves the eighth significant figure. A desk that lengthens its
horizon because the order is larger is responding to the wrong variable.

That optimum is `sqrt(3)` times the Almgren-Chriss half-life, since
`kappa = sqrt(lambda sigma^2 / eta_tilde)` over the same ratio: measured at
1.7352, 1.7335, 1.7330 and 1.7322 half-lives as the risk aversion rises through
four decades, against `sqrt(3) = 1.7321`.

### The optimal schedule has no optimal horizon, because it declines the time

Net alpha under the Almgren-Chriss schedule rises monotonically in the nominal
horizon and saturates — 10,348 at twenty, 12,139 at fifty, 12,238 at a hundred,
12,422 at two hundred — while the half-life it actually trades over barely moves,
10.536 down to 10.488. The risk aversion has already chosen the effective
horizon. So the module reports an interior optimum for the uniform schedule and
says plainly that the optimal schedule has none, rather than returning whichever
point a search happened to stop at.

### Two things the discretisation did rather than the model

The interval count is capped, which has consequences worth stating rather than
hiding. The continuum break-even sits 0.017% to 0.033% below the exact one at a
fixed horizon, and **exactly** `-1/N` below it at a fixed participation rate,
where the horizon is long enough that the count is on its cap and the single
`1 - 1/N` factor is the whole discrepancy. And the risk-neutral cost does not
reach the permanent impact as the horizon grows: at a thousand units of time
`tau` is half a unit and the temporary term is still 95 of the 15,095.

## Phase 18 — What a schedule built on the wrong parameters costs

- [x] The urgency ratio `lambda sigma^2 / eta_tilde`, and the invariance that
      makes it the only thing a schedule sees
- [x] The regret of a schedule built on believed parameters and paid for under
      true ones, exact for the discrete problem
- [x] The cost forecast's error reported beside it, because the two are of
      different orders
- [x] The continuous-time closed form, as a limit to check the discrete figure
      against
- [x] The ceiling on the regret's coefficient, and the quartic decay in the
      horizon measured in half-lives
- [x] A minimax urgency over a stated range of possible truths, beside what a
      point estimate would have guaranteed
- [x] A `robust` command, and the whole lot in JSON

### A three-dimensional question that is one-dimensional

The mean-variance objective of an arbitrary schedule splits into a term no
schedule can change, `epsilon X + gamma X^2 / 2`, and `eta_tilde` times
`sum n^2 / tau + r tau sum x^2` with `r = lambda sigma^2 / eta_tilde`. So the
trades depend on the three parameters and the risk aversion only through `r`:
two sets of beliefs with the same ratio produce identical trades to the last bit,
and the fixed per-share cost does not enter at all. An error in the impact
coefficient and an error in the variance rate are the same error, and the tests
assert that rather than the prose claiming it.

### The number that surprises, and the number that should not

The schedule sits at a minimum, so a first-order error in it costs a
second-order amount; the cost forecast has no such protection. A 10% error in the
ratio makes the forecast 3.4% wrong and the trades 0.038% wrong, and the gap
widens without bound as the estimate improves — 16 times at a 50% error, 88 at
10%, 876 at 1%. Both exponents are measured by a log-log slope: 2.00 and 1.00.

### How much it can ever cost

In the urgent limit the relative regret is `(sqrt(k) - 1)^2 / (2 sqrt(k))` for a
ratio believed `k` times its true value, so the quadratic coefficient is exactly
`1/8` and a factor-of-two error costs 6.07%. That is a ceiling: expanding the
objective in both arguments gives `(a^2 - y^2)^2 / 45`, so the coefficient falls
like `y^4 / 45` for a patient trader — 2.2e-6 at a tenth of a half-life against
0.125 above eight of them. The derivation and the measurement were done
separately and the first attempt at the small-urgency coefficient, `1/48`, was
wrong; the expansion settled it.

### The asymmetry, read twice

The urgent-limit form is symmetric under `k -> 1/k`, and at finite urgency it is
not. The coefficient rises with urgency, so the costly configuration is always
the more urgent one. For a fixed truth, believing the ratio 1.5 times too large
costs 1.70 times believing it 1.5 times too small. For a fixed belief, a truth
1.5 times higher costs 1.47 times a truth 1.5 times lower. The second reading is
the one the minimax responds to, which is why the robust urgency sits 11.8%
*above* the geometric midpoint of its range — the first attempt at this phase
had the direction backwards in prose while the search had it right, and the test
that measured the ratio is what caught it.

### What the robust schedule is worth, and what it is not

Over a factor of two either way it guarantees 1.92% against 2.56% for the
geometric midpoint and 2.71% for the arithmetic mean of the range; over a factor
of ten either way, 23.3% against 47.4%. So it is worth a real amount, and it is
worth more the less is known. But the headline of the phase is the other way
round: with the parameters known to within a factor of two, the worst any
schedule can do is under two per cent of the schedule-dependent cost. The place
to spend effort on a bad impact estimate is the quote, not the schedule.

## Phase 19 — Where to put the order, when the liquidity is random

- [x] A discrete outflow distribution per venue on a lot lattice, with tails
      summed from the top and `E[min(L, A)]` from the layer-cake identity
- [x] Empirical, uniform, point and truncated-geometric constructors, the last
      solving for the ratio that matches a requested mean
- [x] The exact distribution of the total passive fill by convolution, checked
      against brute-force enumeration over the product of the supports
- [x] The expected cost of an allocation in four components: the cross, the
      rebates, the sweep of what is unfilled and the unwind of what is not
- [x] The marginal value of one more posted or crossed lot, in closed form
- [x] A router by marginal placement and local search, and an exhaustive
      reference to measure it against
- [x] The average-liquidity split as the baseline, and what it costs
- [x] A `route` command and a worked example

### A posted lot is worth the probability that it is both reached and needed

The marginal is `reached * (-rebate - sweep_cost * needed + unwind_cost *
(1 - needed))`, where `reached` is the venue's own tail at the next lot and
`needed` is a property of the *leave-one-out* distribution — the chance the
rest of the order has not already covered the target. The venue index enters
nowhere else. So the rebate is earned on the first condition alone and the
sweep is saved only on both, and the closed form agrees with the difference of
two exactly computed costs to 1e-12 over four hundred instances.

### Which makes average liquidity the wrong ranking, and measurably so

Two venues: one always has exactly three lots, the other has eight lots seven
times in ten and nothing otherwise. The second has the higher mean, 5.6 against
3.0, and the first is where the first three lots belong because a lot there is
certain to be reached. Filling the higher-mean venue first costs 0.600 per
share against 0.375 for the routed answer — 60% more, on two venues, from one
ranking mistake. The generic version of the same mistake, splitting the target
in proportion to average liquidity, costs a median 29.9% more per share over
3,000 random instances, with an interquartile range of 18.9% to 40.3%; it is
already optimal on 2.5% of them.

### The local optima are as wide as the routing table

Placing lots one at a time by that marginal lands on a strict local optimum
often enough to matter, and the way out is usually to cross a little less and
post a little more at *every* venue at once. The L1 norm of that move is the
size of the table, so no fixed small neighbourhood contains it and widening
the ball is not an option at `(2 radius + 1) ** width` vectors. Adding that one
direction family by name took a two-lot neighbourhood from 96.9% of the optima
to 99.1%. Against the exhaustive optimum over 9,000 random instances in three
families, a one-lot neighbourhood is right 96.1% to 99.1% of the time and a
three-lot one every time; the worst a one-lot neighbourhood cost was 23.4% of
the cost of not trading at all.

### How far past the target to post is a question about the unwind alone

With over-execution free the optimum commits 2.33 times the target across three
venues; at an unwind cost a quarter of the sweep, twice; at equal to the sweep,
1.67 times; at twice, 1.33; at four times, 1.11. Monotone throughout. A router
that posts the target and no more has not chosen a conservative answer, it has
answered a different question.

### Fragmentation is worth something, with the liquidity held fixed

Six venues each expecting one lot are cheaper to trade against than one venue
expecting six: 0.649 per share against 0.813, a saving of 20.2%, with two and
three venues at 0.761 and 0.717 in between. Independent venues diversify the
fill, so the same completion probability needs less over-posting. That is the
opposite of the usual complaint about fragmented markets, and it is a statement
about one order rather than about a market.

## Phase 20 — Where the half-spread comes from

- [x] Roll's serial-covariance estimator, with its undefined case reported
      rather than silently clipped
- [x] The Corwin-Schultz high-low estimator, with the share of windows that
      come out negative on the result
- [x] The Abdi-Ranaldo close-high-low estimator
- [x] The value before negatives were floored, kept on every estimate
- [x] A simulation of an efficient random walk with an independent bounce and
      a finite number of trades a day, so the spread is known
- [x] Bias, dispersion, standard error and refusal rate for each estimator
      across spread-to-volatility ratios and across tick counts
- [x] A `spread` subcommand, in both modes

### What the phase is for

`placement` has taken a `half_spread` argument since Phase 13 and `shortfall`
has charged a spread since Phase 2. Neither says where the number comes from,
and over bars it is not observable. Three published estimators infer it from
prices. What was missing is not any of them but a measurement of when any of
them can be believed.

### The flooring is a one-sided operation on a two-sided error

All three can produce a negative answer: Roll's sample covariance can be
positive, Corwin-Schultz's windows can be negative, Abdi-Ranaldo's mean
product can be. The published treatment sets those to zero, and that cannot be
neutral. Over 200 runs of 252 days at 78 trades a day against a 2% daily
volatility:

| spread / volatility | estimator | mean | bias | before flooring |
|---|---|---|---|---|
| 0.05 | Roll | 45.1bp | +351% | 7.9bp |
| 0.05 | Corwin-Schultz | 72.7bp | +627% | **-4.3bp** |
| 0.05 | Abdi-Ranaldo | 22.0bp | +120% | **-1.1bp** |
| 0.25 | Roll | 57.2bp | +14% | 29.7bp |
| 0.25 | Corwin-Schultz | 90.1bp | +80% | 24.3bp |
| 0.25 | Abdi-Ranaldo | 43.3bp | -13% | 34.4bp |
| 1.00 | Roll | 200.7bp | +0.3% | 200.7bp |
| 1.00 | Corwin-Schultz | 192.7bp | -3.7% | 167.3bp |
| 1.00 | Abdi-Ranaldo | 199.4bp | -0.3% | 199.4bp |

A 10bp spread against a 2% daily volatility is an ordinary large-cap stock,
and the first block says that at that ratio two of the three estimators have
an unfloored mean *below zero*: they contain no information about the spread
and every basis point they report is the flooring. They are not imprecise
there. They are empty, and they return a positive number anyway.

### Unbiased and wide, or tight and wrong

At a ratio of one they all work and they are not interchangeable. Roll's bias
is +0.3% against a standard error of 1.4%, so it is unbiased as far as 200
runs can resolve, and its dispersion is 39.4bp against Abdi-Ranaldo's 13.2 and
Corwin-Schultz's 11.4. Corwin-Schultz is the tightest and its -3.7% bias is
nine standard errors from zero. Roll also stops refusing entirely at that
ratio, where it refuses 47.5% of the time at a ratio of 0.05.

### Corwin-Schultz is calibrated to a trading frequency

The range of a path sampled at finitely many points is not the range of the
path, and the high-low estimators are derived from the latter. Holding
everything else and varying only how many trades a day the high and low are
the extremes of:

| trades a day | Roll | Corwin-Schultz | Abdi-Ranaldo |
|---|---|---|---|
| 8 | -0.0% | **-45.5%** | -5.8% |
| 20 | -1.4% | -22.8% | -2.5% |
| 39 | -3.9% | -11.6% | -1.3% |
| 78 | -4.3% | -3.9% | -1.8% |
| 195 | +1.6% | +3.7% | -0.6% |
| 390 | -1.5% | **+7.0%** | -1.4% |

Nothing in Roll's column exceeds twice its standard error of 2.2%, so it does
not depend on the tick count in any way 100 runs can see. Corwin-Schultz moves
monotonically across fifty-two points and crosses zero near a hundred trades a
day, so it is right at one sampling frequency and which one has nothing to do
with the market. Abdi-Ranaldo has the same dependence with a tenth of the
range.

### Two checks that are not simulations

With the efficient price held still and only the bounce moving, Corwin-Schultz
and Abdi-Ranaldo are exact in closed form: a still price makes `beta` exactly
twice `gamma`, which collapses Corwin-Schultz's `alpha` to the log spread, and
Abdi-Ranaldo's two deviations become the same close against two identical
midranges, so their product is the squared half-spread every time.

And Roll's independence assumption is load-bearing in a way that does not
announce itself. Make the bounce strictly alternate instead of being drawn
independently and the covariance is the whole squared spread rather than a
quarter of it, so the estimator returns **exactly twice** the truth — a wrong
number rather than a refusal.

## Phase 21 — The command line explains itself

- [x] Help text on every option, the required ones included — a required
      argument with no description is the worst case, since it has to be
      supplied and cannot be left alone
- [x] The conventions described rather than listed: the delay basis moves the
      boundary between delay and opportunity cost without changing the total,
      and the outlier threshold is a modified z-score for the reason a z-score
      cannot do the job
- [x] A parametrised sweep that renders `--help` for the root parser and every
      subcommand, since a help string is expanded only when it is displayed
- [x] Commands enumerated by walking the parser, recursively, rather than
      listed by hand
- [x] A completeness check that every argument carries help at all
- [x] The sweep verified against the regression it exists for
- [x] The same sweep in CI against the built wheel

## Phase 22 — The evidence goes above the install line

- [x] One script recomputing every figure the front page quotes, with the two
      checks against enumeration first: a solver checked against a closed form
      is checked against a rearrangement of the same mathematics, and a sign
      error made in both places agrees with itself
- [x] The schedule solver against all 1,820 ways to split twelve lots over five
      periods — two impact models, three risk aversions, capped and uncapped,
      with the capped cases mattering most since an unconstrained problem is one
      a closed form could have answered anyway
- [x] The router against every allocation on randomly generated problems rather
      than hand-picked ones, a hand-picked problem being one whose answer the
      person picking it could see
- [x] The lot grid's objective checked for never being *below* the continuum's,
      not for agreeing with it: the grid is a subset, so a lower objective is
      wrong rather than more accurate, and a tolerance on the signed difference
      would accept it
- [x] The response kernel measured as a tent at zero urgency *and* as not a
      tent at a positive one, since either figure alone cannot distinguish a
      shape the problem produces from one something linearised into it
- [x] Tentness stated parameter-free — a row's second difference vanishing away
      from its own spike — so no scale is fitted and no reference implemented
- [x] The enumeration sizes pinned, because "checked against enumeration" is
      only as strong as the set enumerated and a quiet reduction would weaken
      the front page without changing a figure
- [x] Three directions of the pin, including the rendered claim against the
      bound the check enforces, which nothing else covers

## Phase 23 — Whose volume is in the benchmark

`benchmarks.py` argued about one convention in its module docstring — the
window being the life of the order rather than the whole day — and was silent
on the other. Exchange volume includes the order's own prints, so a VWAP built
from those bars is a benchmark the order helped set, and `slippage tca` has
printed the participation rate and the VWAP cost in adjacent columns since it
was written without connecting them.

- [x] The decomposition `vwap_market = p * own_average + (1 - p) * vwap_ex_own`,
      so the cost against the recorded benchmark is `(1 - p)` times the cost
      against one the order had no part in
- [x] `Benchmark.INTERVAL_VWAP_EX_OWN`, so the corrected benchmark goes through
      `benchmark_price` and `score_order` like every other
- [x] `own_share`, reporting the decomposition over the benchmark's own window,
      which is not the window the participation rate uses
- [x] The identity exact in currency, where the quantity is the same both
      times, and `(1 - p)` times the ratio of the two references in basis
      points, where `cost_bps` normalises by the reference — measured, since
      that is the one place it is not clean
- [x] The degenerate limit asserted before it is refused: an order that was the
      whole of the recorded volume scores exactly zero whatever it paid
- [x] A separate refusal for own volume *exceeding* the recorded volume, which
      is a data problem rather than a correction and would otherwise subtract
      volume nobody added
- [x] Both figures per order in the report and in its JSON, and the book-level
      pair in the text, since the uncorrected number is the one that is read
- [x] An evidence row whose reference is a construction rather than a formula:
      two hundred books built from explicit prints, half the order's and half
      not, where the other traders' VWAP is known before the library is asked
- [x] Seven ways of breaking the correction confirmed to fail first

What it came to. On the shipped sample book the VWAP cost is +8.91 bps against
the bars as recorded and +11.67 against a benchmark the orders had no part in
setting, at a median participation of 9.7%. Over two hundred constructed books
the uncorrected benchmark understates by between 8% and 52%, and the correction
recovers the known answer to 6e-16.

Three things the measuring turned up that the arithmetic did not.

The book-level ratio is not one minus the mean participation rate. It is 0.76
against the 0.86 the mean implies, because participation and cost correlate at
0.54 — the orders that take the most volume are the ones that cost the most. A
report that printed the rates and left the reader to multiply would have been
inviting a wrong number, so it prints the factor and the reason.

The window the correction needs is not the window `participation_rate` uses.
That function starts at the bar containing arrival by a deliberate choice;
`order_window` starts at arrival. With a mid-bar arrival they give 0.200 and
0.278, and only the second satisfies the decomposition. Which one a VWAP
benchmark *should* use is a convention and changing it would move every
existing number, so it is reported rather than resolved.

And the synthetic book cannot show any of that: every order in it arrives on a
bar boundary, so the two rates coincide there. The suite asserts that they do,
because it is the reason the discrepancy was never visible, and a suite that
only ran against the generated fixture would have concluded the two fields were
one quantity.

## Phase 24 — The interval length, not the interval count

`Mandate.periods_for` capped the interval count at `MAX_PERIODS` however long
the horizon was, so the interval length `tau = horizon / periods` grew with the
horizon. The discrete Almgren-Chriss problem is admissible only while
`eta_tilde = eta - gamma tau / 2` is positive, so past `tau = 2 eta / gamma` a
performance knob was deciding a question about the market: on the module's own
test mandate every participation rate at or below 0.2% was refused, which is the
ordinary case for a large order, and the refusal advised using more periods,
which no caller could do because the cap was the thing in the way.

- [x] `Mandate.longest_interval`, so the bound is a property of the impact model
      rather than a number buried in a search
- [x] The count raised with the horizon whenever it has to be, and the declared
      granularity still honoured below the cap
- [x] `MAX_INTERVAL_FRACTION` chosen by measurement: the `sqrt(3)` half-life
      identity is the instrument, and its worst relative error over seven risk
      aversions is 2.02% at a fraction of 0.05, 0.86% at 0.02 and 0.50% at 0.01
- [x] The worst case located rather than sampled — it sits at the crossover,
      where the horizon is just long enough for the bound to start raising the
      count, and the error at any one risk aversion says nothing about the next
- [x] A horizon past what can be discretised affordably refused by name, saying
      that `eta / gamma` is the obstruction, instead of raising the
      admissibility error with advice the caller cannot take
- [x] The break-even test's reference the discretisation's own term rather than
      a tolerance: the discrete answer sits above the continuum one by
      `1 / (N - 1)`, which is also why the gap narrows as the rate falls
- [x] One unanswerable rate no longer taking `slippage capacity` down with it,
      with the reason carried in the table and in the JSON
- [x] The rate column printed to four significant figures, since one decimal
      place read 0.1% and 0.05% as the same rate and patient rates are exactly
      the ones that now have answers
- [x] An evidence row whose reference is another module: the horizon from
      `capacity`'s golden-section search against `execution`'s `kappa`, neither
      of which can produce `sqrt(3)` alone
- [x] Five ways of breaking the bound confirmed to fail first, and three ways of
      breaking the evidence row

What the measuring turned up. The identity's error tracks the interval length
almost linearly — close to `45 f` — and not the interval count, which is the
reason a cap on the count could not control it. An interval count is an integer,
so between one count and the next the solver's own search varies the length in
proportion to the horizon and `eta_tilde` with it; making the count larger
without making it rise with the horizon moves the error but does not remove it.

The first sweep of horizons written to check the bound could not fail. Every
horizon in it was a round multiple of the bound, so rounding the count down
instead of up gave the same count, and the test passed against a deliberately
broken version. The sweep now uses horizons the bound does not divide and
asserts that at least one of them does not.

Break-even capacity, meanwhile, barely notices any of this: 3.8 basis points
over a tenfold change in the bound. The loosest-looking claim in the module had
the sharpest instrument behind it, and the number a reader cares about had
almost none.
