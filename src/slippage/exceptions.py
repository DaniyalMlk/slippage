"""Exception hierarchy.

Every error raised by the library derives from :class:`SlippageError`, so a
caller can distinguish a modelling problem from a bug in their own code with a
single ``except`` clause.
"""

from __future__ import annotations

__all__ = [
    "SlippageError",
    "ValidationError",
    "NoVolumeError",
    "InsufficientDataError",
    "CalibrationError",
    "ConvergenceError",
]


class SlippageError(Exception):
    """Base class for every error raised by this library."""


class ValidationError(SlippageError, ValueError):
    """A value type was constructed from inputs that cannot describe a trade."""


class NoVolumeError(SlippageError):
    """A volume-weighted quantity was requested over an interval with no volume.

    Deliberately an error rather than a silent fallback to an unweighted mean:
    a caller who asked for VWAP and received TWAP without being told would
    compare their execution against the wrong benchmark and never know.
    """


class InsufficientDataError(SlippageError):
    """Not enough observations to compute the requested quantity."""


class CalibrationError(SlippageError):
    """A model could not be fitted to the supplied executions."""


class ConvergenceError(SlippageError):
    """An iterative routine failed to reach its tolerance."""
