"""Fitting impact models to realised executions.

Every fit returns point estimates *with standard errors* and a list of
identifiability warnings. An impact coefficient is only as useful as its
precision: a scheduler handed an exponent of 0.5 that is really 0.5 +/- 0.4
will produce a confident schedule from a number the data never supported.

Identifiability is the recurring trap. The exponent of a power law is pinned
down by how much order sizes *vary*, not by how many orders there are: a
thousand orders all at 1% of daily volume say nothing about how cost scales
with size. The checks here look at the spread of the regressors as well as the
fitted standard errors, and warn — through :mod:`warnings` and on the result —
rather than silently returning a precise-looking number.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from numpy.typing import NDArray

from .exceptions import CalibrationError, IdentifiabilityWarning, InsufficientDataError
from .impact import LinearImpact, SquareRootLaw
from .series import BarSeries
from .types import Order

__all__ = [
    "Estimate",
    "ExecutionSample",
    "LinearTemporaryFit",
    "PowerLawFit",
    "fit_linear_temporary",
    "fit_permanent",
    "fit_power_law",
    "samples_from_orders",
]

FloatArray = NDArray[np.float64]

# Thresholds for the identifiability checks. Each is a judgement about when a
# fit stops being informative, stated here so it can be argued with.
MIN_OBSERVATIONS = 10
"""Below this, standard errors from asymptotic formulae are themselves unreliable."""

MIN_RATE_DISPERSION = 0.10
"""Coefficient of variation of trading rates below which intercept and slope blur."""

MIN_SIZE_RANGE = 4.0
"""Ratio of largest to smallest order size below which an exponent is unidentified."""

MAX_EXPONENT_STD_ERROR = 0.25
"""An exponent standard error above this spans both the square-root and linear laws."""


@dataclass(frozen=True)
class Estimate:
    """A fitted parameter and its standard error."""

    value: float
    std_error: float

    @property
    def t_stat(self) -> float:
        if self.std_error == 0.0:
            return math.inf if self.value != 0.0 else 0.0
        return self.value / self.std_error

    def interval(self, z: float = 1.959963984540054) -> tuple[float, float]:
        """Symmetric confidence interval, 95% by default."""
        return self.value - z * self.std_error, self.value + z * self.std_error

    def covers(self, truth: float, z: float = 1.959963984540054) -> bool:
        low, high = self.interval(z)
        return low <= truth <= high


def _as_array(name: str, values: Sequence[float] | FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise CalibrationError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} contains non-finite values")
    return array


def _emit(messages: list[str]) -> tuple[str, ...]:
    for message in messages:
        warnings.warn(message, IdentifiabilityWarning, stacklevel=3)
    return tuple(messages)


def _r_squared(y: FloatArray, residuals: FloatArray) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    if total == 0.0:
        return math.nan
    return 1.0 - float(np.sum(residuals**2)) / total


# -- linear temporary impact ------------------------------------------------


@dataclass(frozen=True)
class LinearTemporaryFit:
    """Least-squares fit of ``cost_per_share = epsilon + eta * rate``."""

    epsilon: Estimate
    eta: Estimate
    r_squared: float
    n: int
    residuals: FloatArray = field(repr=False)
    warnings: tuple[str, ...] = ()

    def to_model(self, gamma: float = 0.0) -> LinearImpact:
        """The fitted model, refusing coefficients no impact model can have."""
        if self.eta.value < 0.0 or self.epsilon.value < 0.0:
            raise CalibrationError(
                f"fitted eta={self.eta.value:.4g}, epsilon={self.epsilon.value:.4g}: "
                "a negative coefficient would reward trading faster"
            )
        return LinearImpact(gamma=gamma, eta=self.eta.value, epsilon=self.epsilon.value)


def fit_linear_temporary(
    rates: Sequence[float] | FloatArray, costs_per_share: Sequence[float] | FloatArray
) -> LinearTemporaryFit:
    """Fit the Almgren-Chriss temporary impact line by ordinary least squares.

    ``rates`` are average trading rates (shares per unit time, in the unit the
    model will be used with) and ``costs_per_share`` the realised temporary
    cost per share, in price units, measured against arrival.
    """
    v = _as_array("rates", rates)
    c = _as_array("costs_per_share", costs_per_share)
    if v.shape != c.shape:
        raise CalibrationError(f"got {v.size} rates but {c.size} costs")
    n = v.size
    if n < 3:
        raise InsufficientDataError(f"need at least 3 observations to fit two parameters, got {n}")

    messages: list[str] = []
    if n < MIN_OBSERVATIONS:
        messages.append(f"only {n} observations; standard errors are unreliable")
    mean_rate = float(v.mean())
    dispersion = float(v.std()) / abs(mean_rate) if mean_rate != 0.0 else 0.0
    if dispersion < MIN_RATE_DISPERSION:
        messages.append(
            f"trading rates vary by only {dispersion:.1%} of their mean; the fixed cost "
            "and the rate coefficient cannot be told apart"
        )

    design = np.column_stack([np.ones(n), v])
    coef, *_ = np.linalg.lstsq(design, c, rcond=None)
    residuals = c - design @ coef
    dof = n - 2
    s2 = float(residuals @ residuals) / dof
    cov = s2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return LinearTemporaryFit(
        epsilon=Estimate(float(coef[0]), float(se[0])),
        eta=Estimate(float(coef[1]), float(se[1])),
        r_squared=_r_squared(c, residuals),
        n=n,
        residuals=residuals,
        warnings=_emit(messages),
    )


# -- permanent impact -------------------------------------------------------


def fit_permanent(
    quantities: Sequence[float] | FloatArray, permanent_moves: Sequence[float] | FloatArray
) -> Estimate:
    """Fit ``move = gamma * quantity`` through the origin.

    ``permanent_moves`` are the price changes that *persisted* after each order
    completed — typically measured well after the last fill, once temporary
    impact has decayed. Through the origin because an order of zero shares
    cannot move the price permanently; an intercept would absorb drift that
    belongs in the residual.
    """
    q = _as_array("quantities", quantities)
    m = _as_array("permanent_moves", permanent_moves)
    if q.shape != m.shape:
        raise CalibrationError(f"got {q.size} quantities but {m.size} moves")
    if q.size < 2:
        raise InsufficientDataError(f"need at least 2 observations, got {q.size}")
    sxx = float(q @ q)
    if sxx == 0.0:
        raise CalibrationError("every quantity is zero; gamma is undefined")
    gamma = float(q @ m) / sxx
    residuals = m - gamma * q
    s2 = float(residuals @ residuals) / (q.size - 1)
    return Estimate(gamma, math.sqrt(s2 / sxx))


# -- power law --------------------------------------------------------------


@dataclass(frozen=True)
class PowerLawFit:
    """Fit of ``cost = Y * sigma * x ** delta`` where ``x`` is size over daily volume."""

    y: Estimate
    delta: Estimate
    rss: float
    r_squared: float
    n: int
    residuals: FloatArray = field(repr=False)
    warnings: tuple[str, ...] = ()

    def to_law(self) -> SquareRootLaw:
        return SquareRootLaw(y=self.y.value, delta=self.delta.value)


def _profile(x: FloatArray, sigma: FloatArray, c: FloatArray, delta: float) -> tuple[float, float]:
    """For fixed ``delta``, the least-squares ``Y`` and the residual sum of squares."""
    z = sigma * x**delta
    zz = float(z @ z)
    y = float(z @ c) / zz
    r = c - y * z
    return y, float(r @ r)


def _golden_section(
    f: Callable[[float], float], low: float, high: float, tol: float = 1e-10, max_iter: int = 200
) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def fit_power_law(
    participation: Sequence[float] | FloatArray,
    costs: Sequence[float] | FloatArray,
    volatility: Sequence[float] | FloatArray,
    *,
    delta: float | None = None,
    bounds: tuple[float, float] = (0.05, 1.5),
    grid: int = 400,
) -> PowerLawFit:
    """Fit the metaorder impact law ``cost = Y * sigma * x ** delta``.

    Parameters
    ----------
    participation
        Order size as a fraction of daily volume, ``Q / V``.
    costs
        Realised impact cost of each order as a fraction of price.
    volatility
        Daily volatility of each order's stock, as a fraction.
    delta
        Fix the exponent (0.5 for the square-root law) and fit only ``Y``.
    bounds
        Search interval for the exponent when it is free.

    The exponent is found by profiling: for each candidate ``delta`` the best
    ``Y`` has a closed form, so the two-parameter problem reduces to a
    one-dimensional search over the residual sum of squares, done on a grid and
    refined by golden section. Standard errors come from the Gauss-Newton
    approximation at the optimum.
    """
    x = _as_array("participation", participation)
    c = _as_array("costs", costs)
    sigma = _as_array("volatility", volatility)
    if not (x.shape == c.shape == sigma.shape):
        raise CalibrationError("participation, costs and volatility must have equal length")
    if np.any(x <= 0.0):
        raise CalibrationError("participation must be strictly positive")
    if np.any(sigma <= 0.0):
        raise CalibrationError("volatility must be strictly positive")
    n = x.size
    free = delta is None
    params = 2 if free else 1
    if n < params + 2:
        raise InsufficientDataError(f"need at least {params + 2} observations, got {n}")

    messages: list[str] = []
    if n < MIN_OBSERVATIONS:
        messages.append(f"only {n} observations; standard errors are unreliable")

    if delta is None:
        low, high = bounds
        if not 0.0 < low < high:
            raise CalibrationError(f"invalid exponent bounds {bounds}")
        size_range = float(x.max() / x.min())
        if size_range < MIN_SIZE_RANGE:
            messages.append(
                f"order sizes span only a {size_range:.2f}x range; the exponent is not "
                f"identified below a {MIN_SIZE_RANGE:.0f}x range — fix delta instead"
            )
        candidates = np.linspace(low, high, grid)
        rss = np.array([_profile(x, sigma, c, d)[1] for d in candidates])
        best = int(np.argmin(rss))
        lo = candidates[max(best - 1, 0)]
        hi = candidates[min(best + 1, grid - 1)]
        fitted_delta = _golden_section(lambda d: _profile(x, sigma, c, d)[1], lo, hi)
        step = (high - low) / (grid - 1)
        if fitted_delta - low < step or high - fitted_delta < step:
            messages.append(
                f"the exponent {fitted_delta:.3f} sits on the search bound {bounds}; "
                "the true optimum may lie outside it"
            )
    else:
        fitted_delta = float(delta)

    y_hat, rss_hat = _profile(x, sigma, c, fitted_delta)
    z = sigma * x**fitted_delta
    residuals = c - y_hat * z
    dof = n - params
    s2 = rss_hat / dof

    if free:
        jacobian = np.column_stack([z, y_hat * z * np.log(x)])
        cov = s2 * np.linalg.pinv(jacobian.T @ jacobian)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        y_est = Estimate(y_hat, float(se[0]))
        delta_est = Estimate(fitted_delta, float(se[1]))
        if delta_est.std_error > MAX_EXPONENT_STD_ERROR:
            messages.append(
                f"exponent standard error {delta_est.std_error:.2f} exceeds "
                f"{MAX_EXPONENT_STD_ERROR}; the data cannot distinguish a square-root "
                "law from a linear one"
            )
    else:
        y_est = Estimate(y_hat, math.sqrt(s2 / float(z @ z)))
        delta_est = Estimate(fitted_delta, 0.0)

    return PowerLawFit(
        y=y_est,
        delta=delta_est,
        rss=rss_hat,
        r_squared=_r_squared(c, residuals),
        n=n,
        residuals=residuals,
        warnings=_emit(messages),
    )


# -- from executions --------------------------------------------------------


@dataclass(frozen=True)
class ExecutionSample:
    """One completed order reduced to the regressors the fits need."""

    symbol: str
    quantity: float
    participation: float
    """Executed quantity as a fraction of daily volume."""
    rate: float
    """Executed quantity per ``time_unit`` over the order's trading window."""
    cost_per_share: float
    """Signed cost against arrival, in price units; positive is a cost."""
    cost_fraction: float
    """``cost_per_share`` as a fraction of the arrival price."""
    volatility: float
    """Daily volatility of the stock, as a fraction."""


def samples_from_orders(
    orders: Iterable[Order],
    series: Mapping[str, BarSeries],
    daily_volume: Mapping[str, float],
    daily_volatility: Mapping[str, float],
    *,
    time_unit: timedelta,
) -> list[ExecutionSample]:
    """Turn executed orders into calibration samples.

    Cost is measured against the arrival price, so it is the *trading*
    component of implementation shortfall: delay is not impact, and including
    it would bias every coefficient by whatever the market did before the
    order arrived. Orders with no fills contribute nothing and are skipped.

    ``time_unit`` has no default on purpose. A rate coefficient fitted per
    hour and used per day is wrong by the ratio of the two, silently.
    """
    if time_unit <= timedelta(0):
        raise CalibrationError(f"time_unit must be positive, got {time_unit!r}")
    samples: list[ExecutionSample] = []
    for order in orders:
        if not order.fills:
            continue
        try:
            bars = series[order.symbol]
            volume = daily_volume[order.symbol]
            vol = daily_volatility[order.symbol]
        except KeyError as missing:
            raise CalibrationError(f"no market data for {missing.args[0]!r}") from None
        if volume <= 0.0 or vol <= 0.0:
            raise CalibrationError(
                f"daily volume and volatility for {order.symbol!r} must be positive"
            )
        arrival = bars.price_at(order.arrival_time)
        last = order.last_fill_time
        assert last is not None
        span = bars.end_of_bar_containing(last) - order.arrival_time
        rate = order.filled_quantity / (span / time_unit)
        cost = order.side.sign * (order.average_price - arrival)
        samples.append(
            ExecutionSample(
                symbol=order.symbol,
                quantity=order.filled_quantity,
                participation=order.filled_quantity / volume,
                rate=rate,
                cost_per_share=cost,
                cost_fraction=cost / arrival,
                volatility=vol,
            )
        )
    return samples
