"""
Unit tests for statistical tests module

Tests cover:
- Stationarity tests (ADF, KPSS)
- Cointegration analysis
- Residual diagnostics
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats


try:
    from src.statistical_tests import (
        StationarityTests,
        CointegrationAnalysis,
        ResidualAnalysis,
        print_stationarity_results,
        print_residual_diagnostics
    )
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@pytest.fixture
def stationary_series():
    """Generate stationary time series (white noise)"""
    np.random.seed(42)
    return pd.Series(np.random.randn(1000))


@pytest.fixture
def non_stationary_series():
    """Generate non-stationary time series (random walk)"""
    np.random.seed(42)
    return pd.Series(np.cumsum(np.random.randn(1000)))


@pytest.fixture
def residuals_normal():
    """Generate normal residuals"""
    np.random.seed(42)
    return np.random.randn(500)


@pytest.fixture
def residuals_autocorrelated():
    """Generate autocorrelated residuals"""
    np.random.seed(42)
    n = 500
    residuals = np.zeros(n)
    residuals[0] = np.random.randn()
    for i in range(1, n):
        residuals[i] = 0.7 * residuals[i-1] + np.random.randn()
    return residuals


@pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
class TestStationarityTests:
    """Test suite for stationarity testing"""

    def test_adf_stationary(self, stationary_series):
        """Test ADF on stationary series"""
        tester = StationarityTests(significance_level=0.05)
        result = tester.augmented_dickey_fuller(stationary_series)

        assert result.test_name == "Augmented Dickey-Fuller"
        assert result.is_stationary is True
        assert result.p_value < 0.05
        assert isinstance(result.critical_values, dict)

    def test_adf_non_stationary(self, non_stationary_series):
        """Test ADF on non-stationary series"""
        tester = StationarityTests(significance_level=0.05)
        result = tester.augmented_dickey_fuller(non_stationary_series)

        assert result.is_stationary is False
        assert result.p_value >= 0.05

    def test_kpss_stationary(self, stationary_series):
        """Test KPSS on stationary series"""
        tester = StationarityTests(significance_level=0.05)
        result = tester.kpss_test(stationary_series)

        assert result.test_name == "KPSS"
        # Note: KPSS has opposite hypotheses
        assert result.is_stationary is True

    def test_comprehensive_test(self, stationary_series):
        """Test comprehensive stationarity analysis"""
        tester = StationarityTests()
        results = tester.comprehensive_stationarity_test(stationary_series, "TestSeries")

        assert 'adf_constant' in results
        assert 'adf_trend' in results
        assert 'kpss_level' in results
        assert 'kpss_trend' in results

        # All tests should agree on stationarity
        assert results['adf_constant'].is_stationary is True
        assert results['kpss_level'].is_stationary is True

    def test_differencing_recommendation(self, non_stationary_series):
        """Test differencing order recommendation"""
        tester = StationarityTests()
        d, differenced_series = tester.differencing_recommendation(non_stationary_series, max_d=2)

        # Should recommend at least 1 differencing
        assert d >= 1
        assert len(differenced_series) > 0


@pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
class TestCointegrationAnalysis:
    """Test suite for cointegration testing"""

    def test_engle_granger_cointegrated(self):
        """Test Engle-Granger on cointegrated series"""
        np.random.seed(42)

        # Generate cointegrated series
        z = np.cumsum(np.random.randn(500))  # Common stochastic trend
        x = z + np.random.randn(500) * 0.1
        y = 2 * z + np.random.randn(500) * 0.1

        x_series = pd.Series(x)
        y_series = pd.Series(y)

        coint = CointegrationAnalysis()
        result = coint.engle_granger_test(y_series, x_series)

        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_cointegrated' in result
        assert isinstance(result['is_cointegrated'], bool)

    def test_engle_granger_not_cointegrated(self):
        """Test Engle-Granger on non-cointegrated series"""
        np.random.seed(42)

        # Generate independent random walks
        x = pd.Series(np.cumsum(np.random.randn(500)))
        y = pd.Series(np.cumsum(np.random.randn(500)))

        coint = CointegrationAnalysis()
        result = coint.engle_granger_test(y, x)

        # Should typically not be cointegrated
        assert result['is_cointegrated'] is False or result['p_value'] > 0.05


@pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
class TestResidualAnalysis:
    """Test suite for residual diagnostics"""

    def test_normality_normal_residuals(self, residuals_normal):
        """Test normality tests on normal residuals"""
        analyzer = ResidualAnalysis(residuals_normal)
        results = analyzer.test_normality()

        assert 'jarque_bera' in results
        assert 'shapiro_wilk' in results
        assert 'kolmogorov_smirnov' in results

        # Should pass normality tests (with some tolerance)
        assert results['jarque_bera']['is_normal'] is True
        assert results['shapiro_wilk']['is_normal'] is True

    def test_autocorrelation_white_noise(self, residuals_normal):
        """Test autocorrelation on white noise"""
        analyzer = ResidualAnalysis(residuals_normal)
        results = analyzer.test_autocorrelation(lags=20)

        assert 'ljung_box' in results
        assert 'durbin_watson' in results

        # Durbin-Watson should be around 2 for no autocorrelation
        dw = results['durbin_watson']['statistic']
        assert 1.5 < dw < 2.5

    def test_autocorrelation_ar_process(self, residuals_autocorrelated):
        """Test autocorrelation on AR(1) process"""
        analyzer = ResidualAnalysis(residuals_autocorrelated)
        results = analyzer.test_autocorrelation(lags=20)

        # Should detect autocorrelation
        assert results['ljung_box']['has_autocorrelation'] is True

        # Durbin-Watson should indicate positive autocorrelation
        dw = results['durbin_watson']['statistic']
        assert dw < 1.5

    def test_comprehensive_diagnostics(self, residuals_normal):
        """Test comprehensive residual diagnostics"""
        analyzer = ResidualAnalysis(residuals_normal)
        diagnostics = analyzer.comprehensive_diagnostics()

        assert hasattr(diagnostics, 'normality')
        assert hasattr(diagnostics, 'autocorrelation')
        assert hasattr(diagnostics, 'heteroskedasticity')
        assert hasattr(diagnostics, 'summary')
        assert isinstance(diagnostics.summary, str)


class TestUtilityFunctions:
    """Test utility and printing functions"""

    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
    def test_print_stationarity_results(self, stationary_series, capsys):
        """Test printing of stationarity results"""
        tester = StationarityTests()
        results = tester.comprehensive_stationarity_test(stationary_series)

        print_stationarity_results(results)

        captured = capsys.readouterr()
        assert "STATIONARITY TEST RESULTS" in captured.out
        assert "Augmented Dickey-Fuller" in captured.out

    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
    def test_print_residual_diagnostics(self, residuals_normal, capsys):
        """Test printing of residual diagnostics"""
        analyzer = ResidualAnalysis(residuals_normal)
        diagnostics = analyzer.comprehensive_diagnostics()

        print_residual_diagnostics(diagnostics)

        captured = capsys.readouterr()
        assert "RESIDUAL DIAGNOSTICS" in captured.out
        assert "Normality Tests" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
    def test_empty_series(self):
        """Test handling of empty series"""
        empty_series = pd.Series([])
        tester = StationarityTests()

        with pytest.raises(Exception):
            tester.augmented_dickey_fuller(empty_series)

    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
    def test_series_with_nans(self):
        """Test handling of series with NaN values"""
        series_with_nans = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8])
        tester = StationarityTests()

        # Should handle NaNs by dropping them
        result = tester.augmented_dickey_fuller(series_with_nans)
        assert result is not None

    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
    def test_very_short_series(self):
        """Test handling of very short series"""
        short_series = pd.Series([1, 2, 3, 4, 5])
        tester = StationarityTests()

        # Should handle short series gracefully
        result = tester.augmented_dickey_fuller(short_series)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
