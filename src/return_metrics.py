"""
Scoring metrics for next-day *return* predictions.

Price-level scores flatter a model badly: predicting "tomorrow's price is
today's price" gets an R^2 near one and is worth nothing. Returns strip that
free ride out, so the metrics here are the ones that actually separate a
signal from a coin flip -- error size, sign agreement, and rank correlation.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

METRIC_NAMES = ["rmse", "directional_accuracy", "ic", "r2", "n"]


def _as_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Coerce both inputs to 1-D float arrays of the same non-zero length."""
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()

    if true.size != pred.size:
        raise ValueError(
            f"y_true and y_pred must have the same length, got {true.size} and "
            f"{pred.size}."
        )
    if true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    return true, pred


def rmse(y_true, y_pred) -> float:
    """Root mean squared error between realized and predicted returns."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def directional_accuracy(y_true, y_pred) -> float:
    """Fraction of days where the predicted sign matches the realized sign.

    A flat prediction of exactly zero counts as "down", which is what the
    long/flat backtest does with it too.
    """
    true, pred = _as_pair(y_true, y_pred)
    return float(np.mean((pred > 0) == (true > 0)))


def information_coefficient(y_true, y_pred) -> float:
    """Spearman rank correlation of the predictions against the outcomes.

    Rank correlation ignores the scale of the forecast, so a model that gets
    the ordering of good and bad days right still scores well even when its
    magnitudes are shrunk toward zero.

    Returns:
        The correlation, or NaN if either series is constant (an undefined
        correlation rather than a zero one).
    """
    true, pred = _as_pair(y_true, y_pred)

    # spearmanr returns NaN *and* warns on constant input; the NaN is the
    # answer we want, the warning is noise in an experiment loop.
    if true.size < 2 or np.all(true == true[0]) or np.all(pred == pred[0]):
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation = spearmanr(pred, true).statistic

    return float(correlation)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination against the mean-return benchmark."""
    true, pred = _as_pair(y_true, y_pred)
    return float(r2_score(true, pred))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """All return metrics for one prediction, keyed by :data:`METRIC_NAMES`."""
    true, pred = _as_pair(y_true, y_pred)
    return {
        "rmse": rmse(true, pred),
        "directional_accuracy": directional_accuracy(true, pred),
        "ic": information_coefficient(true, pred),
        "r2": r2(true, pred),
        "n": int(true.size),
    }
