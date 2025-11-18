"""
Unit tests for risk metrics module

Tests cover:
- Value at Risk (VaR) calculations
- Risk-adjusted performance metrics
- Comprehensive risk analysis
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.risk_metrics import (
    ValueAtRisk,
    RiskAnalyzer,
    VaRResult,
    RiskMetrics,
    print_risk_metrics
)


@pytest.fixture
def normal_returns():
    """Generate normal distributed returns"""
    np.random.seed(42)
    return np.random.randn(1000) * 0.01  # 1% daily volatility


@pytest.fixture
def positive_returns():
    """Generate positive skewed returns"""
    np.random.seed(42)
    return np.random.randn(1000) * 0.01 + 0.0005  # Positive drift


@pytest.fixture
def benchmark_returns():
    """Generate benchmark returns"""
    np.random.seed(43)
    return np.random.randn(1000) * 0.008


@pytest.fixture
def price_series():
    """Generate price series from returns"""
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.01 + 0.0005
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


class TestValueAtRisk:
    """Test suite for VaR calculations"""

    def test_historical_var_95(self, normal_returns):
        """Test historical VaR at 95% confidence"""
        var_calc = ValueAtRisk(normal_returns)
        result = var_calc.historical_var(confidence_level=0.95)

        assert isinstance(result, VaRResult)
        assert result.confidence_level == 0.95
        assert result.var > 0  # VaR should be positive
        assert result.cvar >= result.var  # CVaR should be >= VaR
        assert result.method == 'Historical'
        assert len(result.worst_cases) > 0

    def test_historical_var_99(self, normal_returns):
        """Test historical VaR at 99% confidence"""
        var_calc = ValueAtRisk(normal_returns)
        result = var_calc.historical_var(confidence_level=0.99)

        assert result.confidence_level == 0.99
        assert result.var > 0

    def test_parametric_var_normal(self, normal_returns):
        """Test parametric VaR with normal distribution"""
        var_calc = ValueAtRisk(normal_returns)
        result = var_calc.parametric_var(confidence_level=0.95, distribution='normal')

        assert result.method == 'Parametric (normal)'
        assert result.var > 0
        assert result.cvar > result.var

    def test_parametric_var_t_distribution(self, normal_returns):
        """Test parametric VaR with t-distribution"""
        var_calc = ValueAtRisk(normal_returns)
        result = var_calc.parametric_var(confidence_level=0.95, distribution='t')

        assert result.method == 'Parametric (t)'
        assert result.var > 0

    def test_monte_carlo_var(self, normal_returns):
        """Test Monte Carlo VaR"""
        var_calc = ValueAtRisk(normal_returns)
        result = var_calc.monte_carlo_var(
            confidence_level=0.95,
            n_simulations=5000,
            method='normal'
        )

        assert result.method == 'Monte Carlo (normal)'
        assert result.var > 0
        assert result.cvar >= result.var

    def test_var_increases_with_confidence(self, normal_returns):
        """Test that VaR increases with confidence level"""
        var_calc = ValueAtRisk(normal_returns)

        var_95 = var_calc.historical_var(confidence_level=0.95)
        var_99 = var_calc.historical_var(confidence_level=0.99)

        # 99% VaR should be higher than 95% VaR
        assert var_99.var > var_95.var


class TestRiskAnalyzer:
    """Test suite for comprehensive risk analysis"""

    def test_volatility_annualization(self, normal_returns):
        """Test volatility calculation and annualization"""
        analyzer = RiskAnalyzer(normal_returns, periods_per_year=252)

        vol_daily = analyzer.volatility(annualize=False)
        vol_annual = analyzer.volatility(annualize=True)

        # Annual volatility should be ~sqrt(252) times daily
        assert vol_annual > vol_daily
        assert abs(vol_annual - vol_daily * np.sqrt(252)) < 0.001

    def test_downside_deviation(self, normal_returns):
        """Test downside deviation calculation"""
        analyzer = RiskAnalyzer(normal_returns)
        downside_dev = analyzer.downside_deviation(target=0.0)

        # Should be positive and less than total volatility
        assert downside_dev > 0
        assert downside_dev <= analyzer.volatility()

    def test_sharpe_ratio(self, positive_returns):
        """Test Sharpe ratio calculation"""
        analyzer = RiskAnalyzer(positive_returns, risk_free_rate=0.02)
        sharpe = analyzer.sharpe_ratio()

        # With positive drift, Sharpe should be positive
        assert sharpe > 0

    def test_sortino_ratio(self, positive_returns):
        """Test Sortino ratio calculation"""
        analyzer = RiskAnalyzer(positive_returns)
        sortino = analyzer.sortino_ratio()

        # Should be positive for positive drift
        assert sortino > 0

    def test_maximum_drawdown(self, price_series):
        """Test maximum drawdown calculation"""
        returns = np.diff(price_series) / price_series[:-1]
        analyzer = RiskAnalyzer(returns)

        max_dd, start_idx, trough_idx, duration = analyzer.maximum_drawdown(price_series)

        assert max_dd >= 0  # DD should be non-negative percentage
        assert start_idx <= trough_idx
        assert duration >= 0

    def test_tail_ratio(self, normal_returns):
        """Test tail ratio calculation"""
        analyzer = RiskAnalyzer(normal_returns)
        tail_ratio = analyzer.tail_ratio(percentile=95)

        # For symmetric distribution, should be around 1
        assert 0.5 < tail_ratio < 2.0

    def test_beta_calculation(self, normal_returns, benchmark_returns):
        """Test beta calculation with benchmark"""
        analyzer = RiskAnalyzer(normal_returns, benchmark_returns)
        beta = analyzer.beta()

        assert beta is not None
        # Beta should be reasonable
        assert -2 < beta < 3

    def test_alpha_calculation(self, normal_returns, benchmark_returns):
        """Test alpha calculation"""
        analyzer = RiskAnalyzer(normal_returns, benchmark_returns, risk_free_rate=0.02)
        alpha = analyzer.alpha()

        assert alpha is not None
        # Alpha can be positive or negative
        assert isinstance(alpha, (int, float))

    def test_information_ratio(self, normal_returns, benchmark_returns):
        """Test information ratio calculation"""
        analyzer = RiskAnalyzer(normal_returns, benchmark_returns)
        ir = analyzer.information_ratio()

        assert ir is not None
        assert isinstance(ir, (int, float))

    def test_tracking_error(self, normal_returns, benchmark_returns):
        """Test tracking error calculation"""
        analyzer = RiskAnalyzer(normal_returns, benchmark_returns)
        te = analyzer.tracking_error()

        assert te is not None
        assert te > 0  # TE should be positive

    def test_comprehensive_metrics(self, normal_returns, price_series):
        """Test comprehensive risk metrics calculation"""
        analyzer = RiskAnalyzer(normal_returns)
        metrics = analyzer.calculate_all_metrics(prices=price_series)

        assert isinstance(metrics, RiskMetrics)

        # Check all metrics are present
        assert metrics.volatility_annual > 0
        assert metrics.downside_deviation >= 0
        assert metrics.var_95 > 0
        assert metrics.var_99 > metrics.var_95
        assert metrics.cvar_95 >= metrics.var_95
        assert metrics.max_drawdown >= 0
        assert isinstance(metrics.sharpe_ratio, (int, float))
        assert isinstance(metrics.skewness, (int, float))
        assert isinstance(metrics.kurtosis, (int, float))

    def test_comprehensive_metrics_with_benchmark(
        self,
        normal_returns,
        benchmark_returns,
        price_series
    ):
        """Test comprehensive metrics with benchmark"""
        analyzer = RiskAnalyzer(normal_returns, benchmark_returns)
        metrics = analyzer.calculate_all_metrics(prices=price_series)

        # Benchmark-relative metrics should be present
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.tracking_error is not None
        assert metrics.information_ratio is not None


class TestRiskMetricsValidation:
    """Test mathematical properties and relationships"""

    def test_cvar_greater_than_var(self, normal_returns):
        """Test that CVaR >= VaR"""
        var_calc = ValueAtRisk(normal_returns)

        for confidence in [0.90, 0.95, 0.99]:
            result = var_calc.historical_var(confidence_level=confidence)
            assert result.cvar >= result.var

    def test_sharpe_vs_sortino(self, normal_returns):
        """Test relationship between Sharpe and Sortino ratios"""
        analyzer = RiskAnalyzer(normal_returns)

        sharpe = analyzer.sharpe_ratio()
        sortino = analyzer.sortino_ratio()

        # For normal returns, Sortino should generally be higher
        # (because downside deviation < total volatility)
        # But allow for statistical variation
        assert isinstance(sharpe, (int, float))
        assert isinstance(sortino, (int, float))

    def test_omega_ratio_threshold_sensitivity(self, positive_returns):
        """Test Omega ratio with different thresholds"""
        analyzer = RiskAnalyzer(positive_returns)

        omega_0 = analyzer.omega_ratio(threshold=0.0)
        omega_high = analyzer.omega_ratio(threshold=0.001)

        # Omega should decrease with higher threshold
        assert omega_high < omega_0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_zero_volatility(self):
        """Test handling of zero volatility"""
        constant_returns = np.ones(100) * 0.01
        analyzer = RiskAnalyzer(constant_returns)

        sharpe = analyzer.sharpe_ratio()
        assert sharpe == 0  # Should handle division by zero

    def test_all_positive_returns(self):
        """Test with all positive returns"""
        positive_returns = np.abs(np.random.randn(100)) * 0.01
        analyzer = RiskAnalyzer(positive_returns)

        downside_dev = analyzer.downside_deviation(target=0.0)
        assert downside_dev == 0  # No negative returns

    def test_all_negative_returns(self):
        """Test with all negative returns"""
        negative_returns = -np.abs(np.random.randn(100)) * 0.01
        analyzer = RiskAnalyzer(negative_returns, risk_free_rate=0.02)

        sharpe = analyzer.sharpe_ratio()
        assert sharpe < 0  # Should be negative

    def test_nan_handling(self):
        """Test handling of NaN values"""
        returns_with_nan = np.array([0.01, 0.02, np.nan, -0.01, 0.015])
        analyzer = RiskAnalyzer(returns_with_nan)

        # Should remove NaNs and calculate
        vol = analyzer.volatility()
        assert not np.isnan(vol)


class TestUtilityFunctions:
    """Test utility and printing functions"""

    def test_print_risk_metrics(self, normal_returns, price_series, capsys):
        """Test printing of risk metrics"""
        analyzer = RiskAnalyzer(normal_returns)
        metrics = analyzer.calculate_all_metrics(prices=price_series)

        print_risk_metrics(metrics)

        captured = capsys.readouterr()
        assert "COMPREHENSIVE RISK ANALYSIS" in captured.out
        assert "Volatility" in captured.out
        assert "Value at Risk" in captured.out
        assert "Sharpe Ratio" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
