"""
Unit tests for feature selection
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_selection import FeatureSelector, analyze_feature_correlation


@pytest.fixture
def sample_data_with_target():
    """Create sample data with target"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Create features with different correlations to target
    np.random.seed(42)
    target = np.random.uniform(100, 200, 100)
    
    # Feature highly correlated with target
    feature1 = target + np.random.normal(0, 5, 100)
    
    # Feature moderately correlated
    feature2 = target * 0.5 + np.random.normal(0, 20, 100)
    
    # Feature with low correlation
    feature3 = np.random.uniform(0, 100, 100)
    
    # Feature highly correlated with feature1 (should be removed)
    feature4 = feature1 + np.random.normal(0, 1, 100)
    
    data = pd.DataFrame({
        'target': target,
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'Close': target  # Price column
    }, index=dates)
    
    return data


def test_feature_selector_init():
    """Test FeatureSelector initialization"""
    selector = FeatureSelector(method='correlation', threshold=0.95)
    assert selector.method == 'correlation'
    assert selector.threshold == 0.95


def test_remove_correlated_features(sample_data_with_target):
    """Test removing correlated features"""
    selector = FeatureSelector(method='correlation', threshold=0.95)
    
    selected = selector.remove_correlated_features(
        sample_data_with_target,
        target_col='target',
        threshold=0.95
    )
    
    assert len(selected) > 0
    assert isinstance(selected, list)


def test_select_by_importance(sample_data_with_target):
    """Test feature selection by importance"""
    selector = FeatureSelector(method='importance')
    
    selected = selector.select_by_importance(
        sample_data_with_target,
        target_col='target',
        top_k=3
    )
    
    assert len(selected) <= 3
    assert 'feature1' in selected or 'feature2' in selected  # Should select correlated features


def test_select_by_mutual_info(sample_data_with_target):
    """Test feature selection by mutual information"""
    selector = FeatureSelector(method='mutual_info')
    
    selected = selector.select_by_mutual_info(
        sample_data_with_target,
        target_col='target',
        top_k=2
    )
    
    assert len(selected) <= 2


def test_select_features(sample_data_with_target):
    """Test main select_features method"""
    selector = FeatureSelector(method='correlation', threshold=0.95)
    
    selected = selector.select_features(
        sample_data_with_target,
        target_col='target',
        top_k=3
    )
    
    assert len(selected) > 0
    assert len(selector.selected_features) > 0


def test_analyze_feature_correlation(sample_data_with_target):
    """Test feature correlation analysis"""
    result = analyze_feature_correlation(
        sample_data_with_target,
        target_col='target',
        top_n=3
    )
    
    assert len(result) <= 3
    assert 'feature' in result.columns
    assert 'correlation' in result.columns
    assert result['correlation'].max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

