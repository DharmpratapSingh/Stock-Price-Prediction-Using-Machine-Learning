"""
Unit tests for evaluation metrics
"""

import pytest
import numpy as np
from src.evaluation import ModelEvaluator, compare_models


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

