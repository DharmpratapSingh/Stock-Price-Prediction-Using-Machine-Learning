"""
Unit tests for model implementations
"""

import pytest
import pandas as pd
import numpy as np
from src.models import (
    get_model, LinearRegressionModel, RandomForestModel,
    XGBoostModel, ModelTuner
)
from src.feature_engineering import FeatureEngineer, create_target_variable


@pytest.fixture
def sample_data_with_features():
    """Create sample data with features"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.uniform(1000000, 5000000, len(dates))
    }, index=dates)

    # Ensure High >= Low
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    # Add features
    fe = FeatureEngineer(data)
    fe.create_lag_features(columns=['Close'], lags=[1, 2, 5])
    fe.create_returns(periods=[1, 5])
    fe.create_moving_averages(sma_windows=[10, 20], ema_windows=[12])

    featured_data = pd.concat([data, fe.features], axis=1)
    final_data = create_target_variable(featured_data, target_type='price', horizon=1)

    return final_data


def test_get_model():
    """Test model factory function"""
    model = get_model('linear_regression')
    assert model is not None
    assert model.model_name == "Linear Regression"

    model = get_model('random_forest')
    assert model is not None

    with pytest.raises(ValueError):
        get_model('unknown_model')


def test_linear_regression_model(sample_data_with_features):
    """Test Linear Regression model"""
    data = sample_data_with_features.dropna()
    
    feature_cols = [col for col in data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low']]
    X = data[feature_cols]
    y = data['target']

    # Split
    split_idx = int(len(data) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model = LinearRegressionModel()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert not np.isnan(predictions).any()


def test_random_forest_model(sample_data_with_features):
    """Test Random Forest model"""
    data = sample_data_with_features.dropna()
    
    feature_cols = [col for col in data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low']]
    X = data[feature_cols]
    y = data['target']

    split_idx = int(len(data) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    model = RandomForestModel({'n_estimators': 10, 'max_depth': 5})
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)

    # Test feature importance
    importance_df = model.get_feature_importance(feature_cols)
    assert len(importance_df) == len(feature_cols)


def test_xgboost_model(sample_data_with_features):
    """Test XGBoost model"""
    data = sample_data_with_features.dropna()
    
    feature_cols = [col for col in data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low']]
    X = data[feature_cols]
    y = data['target']

    split_idx = int(len(data) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    model = XGBoostModel({'n_estimators': 10, 'max_depth': 3})
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_model_save_load(sample_data_with_features, tmp_path):
    """Test model save and load"""
    data = sample_data_with_features.dropna()
    
    feature_cols = [col for col in data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low']]
    X = data[feature_cols]
    y = data['target']

    split_idx = int(len(data) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    model = LinearRegressionModel()
    model.fit(X_train, y_train)

    # Save
    save_path = tmp_path / "test_model.joblib"
    model.save(str(save_path))

    # Load
    model_loaded = LinearRegressionModel()
    model_loaded.load(str(save_path))

    # Compare predictions
    pred_original = model.predict(X_test)
    pred_loaded = model_loaded.predict(X_test)

    np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

