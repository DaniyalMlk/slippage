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
