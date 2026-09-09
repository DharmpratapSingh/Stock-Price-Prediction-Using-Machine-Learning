"""
Unit tests for the return-oriented scoring metrics.

The metrics score *returns*, not prices, so the interesting properties are
sign agreement and rank correlation rather than raw error alone.
"""

import warnings

import numpy as np
import pytest

from src.return_metrics import (
    directional_accuracy,
    evaluate,
    information_coefficient,
    r2,
    rmse,
)


@pytest.fixture
def returns():
    """A seeded, non-degenerate return series with both signs present."""
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 0.02, 250)


# --------------------------------------------------------------------------
# rmse
# --------------------------------------------------------------------------


def test_rmse_of_zero_prediction_is_root_mean_square(returns):
    expected = np.sqrt(np.mean(returns**2))
    assert rmse(returns, np.zeros_like(returns)) == pytest.approx(expected)


def test_rmse_of_perfect_prediction_is_zero(returns):
    assert rmse(returns, returns) == pytest.approx(0.0)


def test_rmse_returns_a_python_float(returns):
    assert isinstance(rmse(returns, np.zeros_like(returns)), float)


# --------------------------------------------------------------------------
# directional accuracy
# --------------------------------------------------------------------------


def test_directional_accuracy_same_signs_is_one(returns):
    assert directional_accuracy(returns, np.abs(returns) * np.sign(returns)) == 1.0


def test_directional_accuracy_flipped_signs_is_zero(returns):
    assert directional_accuracy(returns, -returns) == 0.0


def test_directional_accuracy_counts_zero_prediction_as_down():
    # (y_pred > 0) is False at zero, so it only matches the down days.
    y_true = np.array([1.0, -1.0, 2.0, -2.0])
    assert directional_accuracy(y_true, np.zeros(4)) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# information coefficient
# --------------------------------------------------------------------------


def test_ic_of_identical_prediction_is_one(returns):
    assert information_coefficient(returns, returns) == pytest.approx(1.0)


def test_ic_of_negated_prediction_is_minus_one(returns):
    assert information_coefficient(returns, -returns) == pytest.approx(-1.0)


def test_ic_is_rank_based_not_scale_based(returns):
    # A monotone transform of the prediction must not change the IC.
    assert information_coefficient(returns, returns) == pytest.approx(
        information_coefficient(returns, 3.0 * returns + 0.5)
    )


def test_ic_of_constant_prediction_is_nan_without_warning(returns):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = information_coefficient(returns, np.full_like(returns, 0.01))
    assert np.isnan(value)
    assert caught == [], f"unexpected warning(s): {[str(w.message) for w in caught]}"


def test_ic_of_constant_truth_is_nan_without_warning(returns):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = information_coefficient(np.zeros_like(returns), returns)
    assert np.isnan(value)
    assert caught == [], f"unexpected warning(s): {[str(w.message) for w in caught]}"


# --------------------------------------------------------------------------
# r2
# --------------------------------------------------------------------------


def test_r2_of_mean_prediction_is_zero(returns):
    assert np.allclose(r2(returns, np.full_like(returns, returns.mean())), 0.0)


def test_r2_of_perfect_prediction_is_one(returns):
    assert r2(returns, returns) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def test_evaluate_returns_exactly_the_expected_keys(returns):
    result = evaluate(returns, np.zeros_like(returns))
    assert set(result) == {"rmse", "directional_accuracy", "ic", "r2", "n"}


def test_evaluate_reports_the_sample_size(returns):
    assert evaluate(returns, np.zeros_like(returns))["n"] == len(returns)


def test_evaluate_matches_the_individual_metrics(returns):
    rng = np.random.default_rng(1)
    y_pred = 0.3 * returns + rng.normal(0.0, 0.01, returns.size)
    result = evaluate(returns, y_pred)
    assert result["rmse"] == pytest.approx(rmse(returns, y_pred))
    assert result["directional_accuracy"] == pytest.approx(
        directional_accuracy(returns, y_pred)
    )
    assert result["ic"] == pytest.approx(information_coefficient(returns, y_pred))
    assert result["r2"] == pytest.approx(r2(returns, y_pred))


def test_evaluate_accepts_python_lists():
    result = evaluate([0.01, -0.02, 0.03], [0.02, -0.01, 0.04])
    assert result["n"] == 3
    assert result["directional_accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric", [rmse, directional_accuracy, information_coefficient, r2, evaluate]
)
def test_mismatched_lengths_raise_value_error(metric):
    with pytest.raises(ValueError):
        metric(np.zeros(5), np.zeros(4))
