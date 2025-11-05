"""
Unit tests for feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer, create_target_variable


@pytest.fixture
def sample_data():
    """Create sample stock data for testing"""
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

    return data


def test_feature_engineer_init(sample_data):
    """Test FeatureEngineer initialization"""
    fe = FeatureEngineer(sample_data)
    assert fe.data is not None
    assert len(fe.data) == len(sample_data)


def test_create_lag_features(sample_data):
    """Test lag feature creation"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_lag_features(columns=['Close'], lags=[1, 2, 5])

    assert 'Close_lag_1' in features.columns
    assert 'Close_lag_2' in features.columns
    assert 'Close_lag_5' in features.columns

    # Check that lag values are correct
    assert pd.isna(features['Close_lag_1'].iloc[0])
    assert features['Close_lag_1'].iloc[1] == sample_data['Close'].iloc[0]


def test_create_returns(sample_data):
    """Test return feature creation"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_returns(periods=[1, 5])

    assert 'return_1d' in features.columns
    assert 'return_5d' in features.columns
    assert 'log_return_1d' in features.columns
    assert 'log_return_5d' in features.columns


def test_create_moving_averages(sample_data):
    """Test moving average creation"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_moving_averages(sma_windows=[10, 20], ema_windows=[12])

    assert 'sma_10' in features.columns
    assert 'sma_20' in features.columns
    assert 'ema_12' in features.columns
    assert 'dist_from_sma_10' in features.columns


def test_create_rsi(sample_data):
    """Test RSI creation"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_rsi(period=14)

    assert 'rsi_14' in features.columns
    assert 'rsi_oversold' in features.columns
    assert 'rsi_overbought' in features.columns

    # RSI should be between 0 and 100
    rsi_values = features['rsi_14'].dropna()
    assert (rsi_values >= 0).all() and (rsi_values <= 100).all()


def test_create_macd(sample_data):
    """Test MACD creation"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_macd(fast=12, slow=26, signal=9)

    assert 'macd_line' in features.columns
    assert 'macd_signal' in features.columns
    assert 'macd_histogram' in features.columns


def test_create_bollinger_bands(sample_data):
    """Test Bollinger Bands creation"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_bollinger_bands(window=20, num_std=2)

    assert 'bb_upper' in features.columns
    assert 'bb_middle' in features.columns
    assert 'bb_lower' in features.columns
    assert 'bb_width' in features.columns


def test_create_volume_features(sample_data):
    """Test volume feature creation"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_volume_features(window=20)

    assert 'volume_sma' in features.columns
    assert 'volume_ratio' in features.columns
    assert 'obv' in features.columns


def test_create_target_variable(sample_data):
    """Test target variable creation"""
    # Add some features first
    fe = FeatureEngineer(sample_data)
    fe.create_lag_features(columns=['Close'], lags=[1])

    data_with_features = pd.concat([sample_data, fe.features], axis=1)

    # Create target
    result = create_target_variable(data_with_features, target_type='price', horizon=1)

    assert 'target' in result.columns
    assert len(result) < len(data_with_features)  # Some rows dropped due to NaN


def test_no_data_leakage(sample_data):
    """Test that there's no data leakage in features"""
    fe = FeatureEngineer(sample_data)
    features = fe.create_lag_features(columns=['Close'], lags=[1])

    # The first lag should be NaN (no future information)
    assert pd.isna(features['Close_lag_1'].iloc[0])

    # Lag 1 at index i should equal Close at index i-1
    for i in range(1, min(10, len(features))):
        assert features['Close_lag_1'].iloc[i] == sample_data['Close'].iloc[i-1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
