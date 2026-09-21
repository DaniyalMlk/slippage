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
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .exceptions import CalibrationError, IdentifiabilityWarning, InsufficientDataError
from .impact import LinearImpact

__all__ = [
    "Estimate",
    "LinearTemporaryFit",
    "fit_linear_temporary",
    "fit_permanent",
]

FloatArray = NDArray[np.float64]

# Thresholds for the identifiability checks. Each is a judgement about when a
# fit stops being informative, stated here so it can be argued with.
MIN_OBSERVATIONS = 10
"""Below this, standard errors from asymptotic formulae are themselves unreliable."""

MIN_RATE_DISPERSION = 0.10
"""Coefficient of variation of trading rates below which intercept and slope blur."""


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
