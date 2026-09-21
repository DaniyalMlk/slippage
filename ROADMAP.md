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

- [ ] Closed-form Almgren–Chriss trajectory for linear impact
- [ ] Limiting cases verified: risk neutrality collapses to a straight line
- [ ] Expected cost and variance of a schedule in closed form
- [ ] Efficient frontier of execution over risk aversion
- [ ] Half-life of the trade and its sensitivity to the model parameters

## Phase 5 — Constrained scheduling

- [ ] Discrete dynamic programme over remaining quantity and time
- [ ] Participation caps, lot sizes and no-crossing constraints
- [ ] Agreement with the closed form when constraints are slack
- [ ] Adaptive re-optimisation from a partially executed state

## Phase 6 — Volume curves and simulation

- [ ] Intraday volume profile estimation from historical bars
- [ ] TWAP, VWAP and percentage-of-volume schedule generators
- [ ] Fill simulator combining impact, drift and volatility
- [ ] Monte Carlo cost distributions with variance-reduction where it helps
- [ ] Simulated costs checked against the closed-form mean and variance

## Phase 7 — Reporting and interface

- [ ] Cost attribution report across a set of orders
- [ ] Outlier detection and peer comparison statistics
- [ ] Command line interface over the whole pipeline
- [ ] End-to-end worked example from raw fills to a scheduling recommendation
- [ ] README covering usage, conventions and the design decisions that mattered
