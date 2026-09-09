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

BASELINE_NAMES = ["zero_return", "train_mean", "always_long", "majority_direction"]


def zero_return(n: int) -> np.ndarray:
    """Predict no move at all: the honest "I don't know" forecast."""
    return np.zeros(int(n), dtype=float)


def train_mean(y_train, n: int) -> np.ndarray:
    """Predict the average training return on every day (drift only)."""
    mean = float(np.asarray(y_train, dtype=float).mean())
    return np.full(int(n), mean, dtype=float)


def always_long(n: int) -> np.ndarray:
    """Predict a tiny up move every day: buy and hold, in prediction form."""
    return np.full(int(n), TINY, dtype=float)


def majority_direction(y_train, n: int) -> np.ndarray:
    """Predict whichever direction was more common in training (ties go long)."""
    up_fraction = float((np.asarray(y_train, dtype=float) > 0).mean())
    sign = TINY if up_fraction >= 0.5 else -TINY
    return np.full(int(n), sign, dtype=float)


def baseline_predictions(y_train, n: int) -> dict[str, np.ndarray]:
    """Every baseline's prediction for ``n`` days, keyed by name."""
    return {
        "zero_return": zero_return(n),
        "train_mean": train_mean(y_train, n),
        "always_long": always_long(n),
        "majority_direction": majority_direction(y_train, n),
    }
