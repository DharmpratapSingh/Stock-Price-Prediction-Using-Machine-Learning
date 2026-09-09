"""
Naive return predictors to measure the real models against.

On daily equity returns these are hard to beat, which is the point: a model
that cannot clear "always long" has not learned anything worth trading.
"""

from __future__ import annotations

import numpy as np

# Small enough to be economically meaningless, large enough that ``> 0`` and a
# zero-threshold long/flat rule read the sign unambiguously.
TINY = 1e-6

def zero_return(n: int) -> np.ndarray:
    """Predict no move at all: the honest "I don't know" forecast."""
    return np.zeros(int(n), dtype=float)


def _training_returns(y_train) -> np.ndarray:
    """Coerce the training targets to a non-empty, finite 1-D float array.

    Without this an empty history slides through as a NaN mean, or -- worse --
    as a confident short from :func:`majority_direction`, since ``(empty >
    0).mean()`` is NaN and NaN fails the ``>= 0.5`` test.
    """
    returns = np.asarray(y_train, dtype=float).ravel()

    if returns.size == 0:
        raise ValueError("y_train must be non-empty to fit a baseline.")
    if not np.all(np.isfinite(returns)):
        bad = int(np.count_nonzero(~np.isfinite(returns)))
        raise ValueError(f"y_train contains {bad} non-finite value(s) (NaN or inf).")

    return returns


def train_mean(y_train, n: int) -> np.ndarray:
    """Predict the average training return on every day (drift only)."""
    mean = float(_training_returns(y_train).mean())
    return np.full(int(n), mean, dtype=float)


def always_long(n: int) -> np.ndarray:
    """Predict a tiny up move every day: buy and hold, in prediction form."""
    return np.full(int(n), TINY, dtype=float)


def majority_direction(y_train, n: int) -> np.ndarray:
    """Predict whichever direction was more common in training (ties go long)."""
    up_fraction = float((_training_returns(y_train) > 0).mean())
    sign = TINY if up_fraction >= 0.5 else -TINY
    return np.full(int(n), sign, dtype=float)


# Uniform ``(y_train, n)`` signature so the registry -- and therefore
# BASELINE_NAMES -- has exactly one definition and cannot drift out of step.
_BASELINES = {
    "zero_return": lambda y_train, n: zero_return(n),
    "train_mean": train_mean,
    "always_long": lambda y_train, n: always_long(n),
    "majority_direction": majority_direction,
}

BASELINE_NAMES = list(_BASELINES)


def baseline_predictions(y_train, n: int) -> dict[str, np.ndarray]:
    """Every baseline's prediction for ``n`` days, keyed by name."""
    return {name: baseline(y_train, n) for name, baseline in _BASELINES.items()}
