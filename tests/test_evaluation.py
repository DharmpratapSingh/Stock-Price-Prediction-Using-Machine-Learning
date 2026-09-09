"""
Unit tests for evaluation metrics
"""

import numpy as np
import pytest

from src.evaluation import (
    ModelEvaluator,
    compare_models,
    direction_metrics,
    return_regression_metrics,
    wilson_interval,
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions and actuals"""
    np.random.seed(42)
    n = 100
    actual = np.random.uniform(100, 200, n)
    predicted = actual + np.random.normal(0, 5, n)  # Add some noise
    return actual, predicted


def test_model_evaluator_init(sample_predictions):
    """Test ModelEvaluator initialization"""
    actual, predicted = sample_predictions
    evaluator = ModelEvaluator(actual, predicted)
    
    assert len(evaluator.y_true) == len(actual)
    assert len(evaluator.y_pred) == len(predicted)


def test_mse_rmse_mae(sample_predictions):
    """Test MSE, RMSE, MAE calculations"""
    actual, predicted = sample_predictions
    evaluator = ModelEvaluator(actual, predicted)
    
    mse = evaluator.mse()
    rmse = evaluator.rmse()
    mae = evaluator.mae()
    
    assert mse >= 0
    assert rmse >= 0
    assert mae >= 0
    assert rmse == np.sqrt(mse)


def test_r2_score(sample_predictions):
    """Test R² score"""
    actual, predicted = sample_predictions
    evaluator = ModelEvaluator(actual, predicted)
    
    r2 = evaluator.r2()
    assert -np.inf < r2 <= 1.0


def test_directional_accuracy(sample_predictions):
    """Test directional accuracy"""
    actual, predicted = sample_predictions
    evaluator = ModelEvaluator(actual, predicted)
    
    acc = evaluator.directional_accuracy()
    assert 0 <= acc <= 1.0


def test_calculate_all_metrics(sample_predictions):
    """Test calculate_all_metrics"""
    actual, predicted = sample_predictions
    evaluator = ModelEvaluator(actual, predicted)
    
    metrics = evaluator.calculate_all_metrics()
    
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'R2' in metrics
    assert 'Directional_Accuracy' in metrics


def test_compare_models(sample_predictions):
    """Test model comparison"""
    actual, predicted = sample_predictions
    
    # Create multiple model results
    models_results = {
        'model1': (actual, predicted),
        'model2': (actual, predicted + np.random.normal(0, 2, len(actual)))
    }
    
    comparison_df = compare_models(models_results)
    
    assert len(comparison_df) == 2
    assert 'R2' in comparison_df.columns
    assert 'RMSE' in comparison_df.columns


def test_financial_metrics(sample_predictions):
    """Test financial metrics calculation"""
    actual, predicted = sample_predictions
    evaluator = ModelEvaluator(actual, predicted, prices=actual)
    
    financial_metrics = evaluator.calculate_returns_based_metrics()
    
    # Should have some financial metrics if prices provided
    assert isinstance(financial_metrics, dict)


# ----------------------------------------------------------------------
# Return-space metrics
# ----------------------------------------------------------------------

def test_return_metrics_perfect_forecast():
    y = np.array([0.01, -0.02, 0.003, -0.001])
    metrics = return_regression_metrics(y, y)

    assert metrics['RMSE'] == pytest.approx(0.0)
    assert metrics['R2'] == pytest.approx(1.0)
    assert metrics['R2_vs_zero'] == pytest.approx(1.0)
    assert metrics['n'] == 4


def test_zero_forecast_scores_zero_against_the_zero_baseline():
    """R2_vs_zero is defined so the zero forecast scores exactly 0."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.02, 500)
    metrics = return_regression_metrics(y, np.zeros(500))
    assert metrics['R2_vs_zero'] == pytest.approx(0.0, abs=1e-12)


def test_return_metrics_penalise_a_worse_than_useless_forecast():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.02, 500)
    metrics = return_regression_metrics(y, -y)  # sign-flipped forecast
    assert metrics['R2'] < 0
    assert metrics['R2_vs_zero'] < 0


def test_return_metrics_ignore_nan_pairs():
    y = np.array([0.01, np.nan, 0.02])
    metrics = return_regression_metrics(y, np.array([0.01, 0.5, 0.02]))
    assert metrics['n'] == 2
    assert metrics['RMSE'] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Directional accuracy, intervals and significance
# ----------------------------------------------------------------------

def test_wilson_interval_brackets_the_estimate():
    lower, upper = wilson_interval(55, 100)
    assert lower < 0.55 < upper
    assert 0.0 <= lower and upper <= 1.0


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(55, 100)
    large = wilson_interval(5500, 10000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_stays_in_bounds_at_the_extremes():
    assert wilson_interval(0, 50)[0] >= 0.0
    assert wilson_interval(50, 50)[1] <= 1.0


def test_direction_metrics_on_a_perfect_classifier():
    truth = np.array([1, 0, 1, 1, 0, 1])
    metrics = direction_metrics(truth, truth)

    assert metrics['accuracy'] == pytest.approx(1.0)
    assert metrics['precision_up'] == pytest.approx(1.0)
    assert metrics['recall_up'] == pytest.approx(1.0)


def test_coin_flip_is_not_significant():
    """A 50.4% result over 500 days must not read as skill."""
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 2, 500)
    guess = rng.integers(0, 2, 500)

    metrics = direction_metrics(truth, guess)
    assert metrics['p_vs_0.5'] > 0.05
    assert metrics['ci_lower'] < 0.5 < metrics['ci_upper']


def test_direction_metrics_tests_against_the_reference_rate():
    """
    An always-up predictor scores the up-rate but adds nothing over the baseline.

    Testing only against 0.5 would call this a significant result.
    """
    rng = np.random.default_rng(6)
    truth = (rng.random(2000) < 0.54).astype(int)
    always_up = np.ones(2000, dtype=int)

    metrics = direction_metrics(truth, always_up, reference_rate=0.54)
    assert metrics['p_vs_0.5'] < 0.01        # beats a coin flip
    assert metrics['p_vs_reference'] > 0.05  # but not the always-up baseline
    assert metrics['recall_up'] == pytest.approx(1.0)


def test_direction_metrics_reports_prediction_mix():
    truth = np.array([1, 1, 0, 0])
    predicted = np.array([1, 1, 1, 0])
    metrics = direction_metrics(truth, predicted)

    assert metrics['pred_up_rate'] == pytest.approx(0.75)
    assert metrics['actual_up_rate'] == pytest.approx(0.5)
    assert metrics['precision_up'] == pytest.approx(2 / 3)


def test_direction_metrics_handles_empty_input():
    assert direction_metrics(np.array([]), np.array([]))['n'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

