"""
Unit tests for backtesting framework
"""

import pytest
import numpy as np
import pandas as pd
from src.backtesting import Backtester, print_backtest_results


@pytest.fixture
def sample_prices():
    """Create sample price data"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    # Create trending prices
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, 100))
    return dates, prices


@pytest.fixture
def sample_predictions(sample_prices):
    """Create sample predictions"""
    dates, prices = sample_prices
    # Predictions slightly ahead of actual prices (simulating good predictions)
    predictions = prices + np.random.normal(0, 1, len(prices))
    return dates, prices, predictions


def test_backtester_init():
    """Test Backtester initialization"""
    backtester = Backtester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    assert backtester.initial_capital == 100000
    assert backtester.commission == 0.001
    assert backtester.slippage == 0.0005


def test_simple_strategy(sample_predictions):
    """Test simple trading strategy"""
    dates, actuals, predictions = sample_predictions
    
    backtester = Backtester(initial_capital=100000)
    
    results = backtester.simple_strategy(
        predictions=predictions,
        actuals=actuals,
        dates=dates,
        threshold=0.01
    )
    
    assert 'equity_curve' in results
    assert 'final_equity' in results
    assert 'total_return' in results
    assert 'metrics' in results
    assert len(results['equity_curve']) == len(actuals) + 1  # +1 for initial capital


def test_buy_and_hold_strategy(sample_prices):
    """Test buy and hold strategy"""
    dates, prices = sample_prices
    
    backtester = Backtester(initial_capital=100000)
    
    results = backtester.buy_and_hold_strategy(
        prices=prices,
        dates=dates
    )
    
    assert 'equity_curve' in results
    assert 'final_equity' in results
    assert 'total_return' in results
    assert len(results['equity_curve']) == len(prices)


def test_backtest_metrics(sample_predictions):
    """Test backtest metrics calculation"""
    dates, actuals, predictions = sample_predictions
    
    backtester = Backtester(initial_capital=100000)
    
    results = backtester.simple_strategy(
        predictions=predictions,
        actuals=actuals,
        dates=dates,
        threshold=0.01
    )
    
    metrics = results['metrics']
    
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics


def test_compare_strategies(sample_predictions):
    """Test strategy comparison"""
    dates, actuals, predictions = sample_predictions
    
    backtester = Backtester(initial_capital=100000)
    
    strategy_results = backtester.simple_strategy(
        predictions=predictions,
        actuals=actuals,
        dates=dates
    )
    
    buy_hold_results = backtester.buy_and_hold_strategy(
        prices=actuals,
        dates=dates
    )
    
    comparison = backtester.compare_strategies(
        {'ML Strategy': strategy_results},
        buy_hold_results
    )
    
    assert len(comparison) >= 2  # At least ML Strategy and Buy & Hold
    assert 'Buy & Hold' in comparison.index


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

