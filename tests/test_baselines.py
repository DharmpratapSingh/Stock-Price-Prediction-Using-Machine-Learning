"""
Unit tests for the naive return baselines.

A model is only interesting if it beats these, so the baselines themselves
have to be exactly what they claim to be.
"""

import numpy as np
import pytest

from src.baselines import (
    BASELINE_NAMES,
    TINY,
    always_long,
    baseline_predictions,
    majority_direction,
    train_mean,
    zero_return,
)
from src.return_metrics import directional_accuracy


@pytest.fixture
def y_train():
    rng = np.random.default_rng(7)
    return rng.normal(0.001, 0.02, 300)


# --------------------------------------------------------------------------
# shapes and dtypes
# --------------------------------------------------------------------------


def test_every_baseline_returns_a_float_array_of_length_n(y_train):
    n = 42
    for prediction in (
        zero_return(n),
        train_mean(y_train, n),
        always_long(n),
        majority_direction(y_train, n),
    ):
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (n,)
        assert prediction.dtype == np.float64


def test_zero_return_is_all_zeros():
    assert np.all(zero_return(10) == 0.0)


# --------------------------------------------------------------------------
# train_mean
# --------------------------------------------------------------------------


def test_train_mean_predicts_the_training_mean(y_train):
    prediction = train_mean(y_train, 5)
    assert np.allclose(prediction, y_train.mean())


def test_train_mean_accepts_a_python_list():
    assert np.allclose(train_mean([0.0, 0.02], 3), 0.01)


# --------------------------------------------------------------------------
# always_long
# --------------------------------------------------------------------------


def test_always_long_is_tiny_and_positive():
    prediction = always_long(4)
    assert np.all(prediction > 0)
    assert np.allclose(prediction, TINY)


def test_always_long_directional_accuracy_is_the_up_day_fraction():
    rng = np.random.default_rng(11)
    y_true = rng.normal(0.0, 0.02, 500)
    expected = float((y_true > 0).mean())
    assert directional_accuracy(y_true, always_long(y_true.size)) == pytest.approx(
        expected
    )


# --------------------------------------------------------------------------
# majority_direction
# --------------------------------------------------------------------------


def test_majority_direction_goes_long_when_training_majority_is_up():
    up_heavy = np.array([0.01, 0.02, 0.03, -0.01])
    assert np.all(majority_direction(up_heavy, 3) == TINY)


def test_majority_direction_goes_short_when_training_majority_is_down():
    down_heavy = np.array([-0.01, -0.02, -0.03, 0.01])
    assert np.all(majority_direction(down_heavy, 3) == -TINY)


def test_majority_direction_breaks_a_tie_long():
    tied = np.array([0.01, -0.01])
    assert np.all(majority_direction(tied, 2) == TINY)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def test_baseline_predictions_keys_match_baseline_names(y_train):
    predictions = baseline_predictions(y_train, 20)
    assert list(predictions) == BASELINE_NAMES


def test_baseline_predictions_have_the_requested_length(y_train):
    for prediction in baseline_predictions(y_train, 20).values():
        assert prediction.shape == (20,)


def test_baseline_predictions_match_the_individual_functions(y_train):
    predictions = baseline_predictions(y_train, 8)
    assert np.array_equal(predictions["zero_return"], zero_return(8))
    assert np.array_equal(predictions["train_mean"], train_mean(y_train, 8))
    assert np.array_equal(predictions["always_long"], always_long(8))
    assert np.array_equal(
        predictions["majority_direction"], majority_direction(y_train, 8)
    )


# --------------------------------------------------------------------------
# training-data validation
# --------------------------------------------------------------------------

TRAINED_BASELINES = [train_mean, majority_direction]


@pytest.mark.parametrize("baseline", TRAINED_BASELINES)
def test_empty_training_data_raises_value_error(baseline):
    """An empty history is no information, not a confident short."""
    with pytest.raises(ValueError):
        baseline(np.array([]), 5)


@pytest.mark.parametrize("baseline", TRAINED_BASELINES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_training_data_raises_value_error(baseline, bad):
    with pytest.raises(ValueError):
        baseline(np.array([0.01, bad, -0.02]), 5)


@pytest.mark.parametrize("bad_train", [np.array([]), np.array([0.01, np.nan])])
def test_baseline_predictions_rejects_bad_training_data(bad_train):
    with pytest.raises(ValueError):
        baseline_predictions(bad_train, 5)
