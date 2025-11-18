"""
Statistical Tests Module for Quantitative Finance
Implements rigorous statistical tests for time series analysis and model validation

This module provides essential statistical tests used in quantitative finance:
- Stationarity tests (ADF, KPSS, Phillips-Perron)
- Cointegration analysis
- Residual diagnostics
- Distribution tests
- Autocorrelation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

# Statistical test imports
try:
    from statsmodels.tsa.stattools import (
        adfuller, kpss, acf, pacf,
        grangercausalitytests, coint
    )
    from statsmodels.stats.diagnostic import (
        acorr_ljungbox, het_white, het_breuschpagan
    )
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Install with: pip install statsmodels")


@dataclass
class StationarityTestResult:
    """Results from stationarity tests"""
    test_name: str
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool
    conclusion: str


@dataclass
class ResidualDiagnostics:
    """Comprehensive residual diagnostics"""
    normality: Dict[str, Union[float, bool]]
    autocorrelation: Dict[str, Union[float, bool, pd.DataFrame]]
    heteroskedasticity: Dict[str, Union[float, bool]]
    summary: str


class StationarityTests:
    """
    Comprehensive stationarity testing for time series

    Stationarity is a critical assumption for many time series models.
    This class implements multiple tests to rigorously assess stationarity.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize stationarity tests

        Args:
            significance_level: Significance level for hypothesis tests (default 0.05)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for statistical tests")

        self.significance_level = significance_level

    def augmented_dickey_fuller(
        self,
        series: pd.Series,
        regression: str = 'c',
        maxlag: Optional[int] = None
    ) -> StationarityTestResult:
        """
        Augmented Dickey-Fuller test for unit root (stationarity)

        H0: Series has a unit root (non-stationary)
        H1: Series is stationary

        Args:
            series: Time series to test
            regression: Regression type ('c'=constant, 'ct'=constant+trend, 'ctt'=constant+trend+trend^2, 'n'=none)
            maxlag: Maximum lag to use (None=automatic)

        Returns:
            StationarityTestResult with test details
        """
        logger.info(f"Running Augmented Dickey-Fuller test (regression={regression})")

        # Remove NaN values
        series_clean = series.dropna()

        # Run test
        result = adfuller(series_clean, regression=regression, maxlag=maxlag)

        test_stat = result[0]
        p_value = result[1]
        critical_values = result[4]

        # Determine stationarity
        is_stationary = p_value < self.significance_level

        conclusion = (
            f"Reject H0: Series is stationary (p={p_value:.4f} < {self.significance_level})"
            if is_stationary
            else f"Fail to reject H0: Series has unit root/non-stationary (p={p_value:.4f} >= {self.significance_level})"
        )

        return StationarityTestResult(
            test_name="Augmented Dickey-Fuller",
            test_statistic=test_stat,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            conclusion=conclusion
        )

    def kpss_test(
        self,
        series: pd.Series,
        regression: str = 'c',
        nlags: str = 'auto'
    ) -> StationarityTestResult:
        """
        KPSS test for stationarity

        H0: Series is stationary
        H1: Series has a unit root (non-stationary)

        Note: This test has opposite hypotheses compared to ADF test

        Args:
            series: Time series to test
            regression: Regression type ('c'=level stationary, 'ct'=trend stationary)
            nlags: Number of lags ('auto' or integer)

        Returns:
            StationarityTestResult with test details
        """
        logger.info(f"Running KPSS test (regression={regression})")

        # Remove NaN values
        series_clean = series.dropna()

        # Run test
        result = kpss(series_clean, regression=regression, nlags=nlags)

        test_stat = result[0]
        p_value = result[1]
        critical_values = result[3]

        # Determine stationarity (note: opposite of ADF)
        is_stationary = p_value >= self.significance_level

        conclusion = (
            f"Fail to reject H0: Series is stationary (p={p_value:.4f} >= {self.significance_level})"
            if is_stationary
            else f"Reject H0: Series is non-stationary (p={p_value:.4f} < {self.significance_level})"
        )

        return StationarityTestResult(
            test_name="KPSS",
            test_statistic=test_stat,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            conclusion=conclusion
        )

    def comprehensive_stationarity_test(
        self,
        series: pd.Series,
        name: str = "Series"
    ) -> Dict[str, StationarityTestResult]:
        """
        Run multiple stationarity tests for robust conclusion

        Best practice: Use both ADF and KPSS tests
        - If ADF indicates stationary AND KPSS indicates stationary: Series is stationary
        - If ADF indicates non-stationary AND KPSS indicates non-stationary: Series is non-stationary
        - If they disagree: Inconclusive, may need differencing

        Args:
            series: Time series to test
            name: Name of the series for logging

        Returns:
            Dictionary of test results
        """
        logger.info(f"Running comprehensive stationarity tests for {name}")

        results = {}

        # ADF test with constant
        results['adf_constant'] = self.augmented_dickey_fuller(series, regression='c')

        # ADF test with constant and trend
        results['adf_trend'] = self.augmented_dickey_fuller(series, regression='ct')

        # KPSS test level stationary
        results['kpss_level'] = self.kpss_test(series, regression='c')

        # KPSS test trend stationary
        results['kpss_trend'] = self.kpss_test(series, regression='ct')

        return results

    def differencing_recommendation(
        self,
        series: pd.Series,
        max_d: int = 3
    ) -> Tuple[int, pd.Series]:
        """
        Determine optimal differencing order to achieve stationarity

        Args:
            series: Time series
            max_d: Maximum differencing order to test

        Returns:
            Tuple of (optimal_d, differenced_series)
        """
        logger.info("Determining optimal differencing order")

        current_series = series.copy()

        for d in range(max_d + 1):
            # Test current series
            adf_result = self.augmented_dickey_fuller(current_series)
            kpss_result = self.kpss_test(current_series)

            # Check if both tests agree on stationarity
            if adf_result.is_stationary and kpss_result.is_stationary:
                logger.info(f"Series is stationary after {d} differencing(s)")
                return d, current_series

            # Apply differencing
            if d < max_d:
                current_series = current_series.diff().dropna()

        logger.warning(f"Series not stationary after {max_d} differencing(s)")
        return max_d, current_series


class CointegrationAnalysis:
    """
    Cointegration testing for pairs trading and mean reversion strategies

    Cointegration is crucial for:
    - Pairs trading strategies
    - Statistical arbitrage
    - Determining if two non-stationary series have a stationary linear combination
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize cointegration analysis

        Args:
            significance_level: Significance level for tests
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for cointegration tests")

        self.significance_level = significance_level

    def engle_granger_test(
        self,
        y: pd.Series,
        x: Union[pd.Series, pd.DataFrame],
        trend: str = 'c'
    ) -> Dict[str, any]:
        """
        Engle-Granger two-step cointegration test

        H0: No cointegration
        H1: Series are cointegrated

        Args:
            y: Dependent variable
            x: Independent variable(s)
            trend: Trend parameter ('c'=constant, 'ct'=constant+trend, 'ctt'=constant+trend+quadratic)

        Returns:
            Dictionary with test results
        """
        logger.info("Running Engle-Granger cointegration test")

        # Ensure no NaN values
        if isinstance(x, pd.Series):
            combined = pd.concat([y, x], axis=1).dropna()
            y_clean = combined.iloc[:, 0]
            x_clean = combined.iloc[:, 1]
        else:
            combined = pd.concat([y, x], axis=1).dropna()
            y_clean = combined.iloc[:, 0]
            x_clean = combined.iloc[:, 1:]

        # Run test
        test_stat, p_value, critical_values = coint(y_clean, x_clean, trend=trend)

        is_cointegrated = p_value < self.significance_level

        result = {
            'test_statistic': test_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_cointegrated': is_cointegrated,
            'conclusion': (
                f"Reject H0: Series are cointegrated (p={p_value:.4f})"
                if is_cointegrated
                else f"Fail to reject H0: No cointegration (p={p_value:.4f})"
            )
        }

        return result

    def johansen_test(
        self,
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> Dict[str, any]:
        """
        Johansen cointegration test for multiple time series

        Useful for analyzing cointegration in portfolios with multiple assets

        Args:
            data: DataFrame with multiple time series
            det_order: Deterministic order (-1=no deterministic, 0=constant, 1=linear trend)
            k_ar_diff: Number of lagged differences in VAR

        Returns:
            Dictionary with test results
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        logger.info(f"Running Johansen cointegration test on {data.shape[1]} series")

        # Remove NaN
        data_clean = data.dropna()

        # Run test
        result = coint_johansen(data_clean, det_order=det_order, k_ar_diff=k_ar_diff)

        output = {
            'trace_statistic': result.lr1,
            'max_eigen_statistic': result.lr2,
            'critical_values_trace': result.cvt,
            'critical_values_max_eigen': result.cvm,
            'eigenvalues': result.eig,
            'n_cointegrating_vectors': 0
        }

        # Determine number of cointegrating vectors
        for i in range(len(result.lr1)):
            if result.lr1[i] > result.cvt[i, 1]:  # 95% critical value
                output['n_cointegrating_vectors'] += 1

        return output


class ResidualAnalysis:
    """
    Comprehensive residual diagnostics for model validation

    Critical for assessing model assumptions and quality in quantitative research
    """

    def __init__(self, residuals: np.ndarray, significance_level: float = 0.05):
        """
        Initialize residual analysis

        Args:
            residuals: Model residuals
            significance_level: Significance level for tests
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for residual diagnostics")

        self.residuals = np.array(residuals)
        self.residuals = self.residuals[~np.isnan(self.residuals)]
        self.significance_level = significance_level

    def test_normality(self) -> Dict[str, Union[float, bool]]:
        """
        Test residuals for normality using multiple tests

        Returns:
            Dictionary with normality test results
        """
        logger.info("Testing residuals for normality")

        results = {}

        # Jarque-Bera test
        jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(self.residuals)
        results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'skewness': jb_skew,
            'kurtosis': jb_kurtosis,
            'is_normal': jb_pvalue >= self.significance_level
        }

        # Shapiro-Wilk test (for smaller samples)
        if len(self.residuals) < 5000:
            sw_stat, sw_pvalue = stats.shapiro(self.residuals)
            results['shapiro_wilk'] = {
                'statistic': sw_stat,
                'p_value': sw_pvalue,
                'is_normal': sw_pvalue >= self.significance_level
            }

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(
            self.residuals,
            'norm',
            args=(np.mean(self.residuals), np.std(self.residuals))
        )
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'is_normal': ks_pvalue >= self.significance_level
        }

        # Anderson-Darling test
        ad_result = stats.anderson(self.residuals, dist='norm')
        results['anderson_darling'] = {
            'statistic': ad_result.statistic,
            'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], ad_result.critical_values)),
            'is_normal': ad_result.statistic < ad_result.critical_values[2]  # 5% level
        }

        return results

    def test_autocorrelation(
        self,
        lags: int = 40
    ) -> Dict[str, Union[float, bool, pd.DataFrame]]:
        """
        Test residuals for autocorrelation

        Args:
            lags: Number of lags to test

        Returns:
            Dictionary with autocorrelation test results
        """
        logger.info("Testing residuals for autocorrelation")

        results = {}

        # Ljung-Box test
        lb_result = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
        results['ljung_box'] = {
            'test_results': lb_result,
            'has_autocorrelation': (lb_result['lb_pvalue'] < self.significance_level).any()
        }

        # Durbin-Watson statistic
        dw_stat = durbin_watson(self.residuals)
        results['durbin_watson'] = {
            'statistic': dw_stat,
            'interpretation': (
                'Positive autocorrelation' if dw_stat < 1.5
                else 'No autocorrelation' if 1.5 <= dw_stat <= 2.5
                else 'Negative autocorrelation'
            )
        }

        # ACF and PACF
        acf_values = acf(self.residuals, nlags=lags, fft=True)
        pacf_values = pacf(self.residuals, nlags=lags)

        results['acf'] = acf_values
        results['pacf'] = pacf_values

        return results

    def test_heteroskedasticity(
        self,
        exog: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, bool]]:
        """
        Test residuals for heteroskedasticity (non-constant variance)

        Args:
            exog: Exogenous variables (if None, uses residuals squared)

        Returns:
            Dictionary with heteroskedasticity test results
        """
        logger.info("Testing residuals for heteroskedasticity")

        results = {}

        if exog is None:
            # Create simple exog from residuals
            exog = np.column_stack([
                np.ones(len(self.residuals)),
                self.residuals ** 2
            ])

        # White's test
        try:
            white_stat, white_pvalue, white_fstat, white_fpvalue = het_white(
                self.residuals, exog
            )
            results['white'] = {
                'lm_statistic': white_stat,
                'lm_pvalue': white_pvalue,
                'f_statistic': white_fstat,
                'f_pvalue': white_fpvalue,
                'has_heteroskedasticity': white_pvalue < self.significance_level
            }
        except Exception as e:
            logger.warning(f"White's test failed: {e}")
            results['white'] = {'error': str(e)}

        # Breusch-Pagan test
        try:
            bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = het_breuschpagan(
                self.residuals, exog
            )
            results['breusch_pagan'] = {
                'lm_statistic': bp_stat,
                'lm_pvalue': bp_pvalue,
                'f_statistic': bp_fstat,
                'f_pvalue': bp_fpvalue,
                'has_heteroskedasticity': bp_pvalue < self.significance_level
            }
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            results['breusch_pagan'] = {'error': str(e)}

        return results

    def comprehensive_diagnostics(
        self,
        exog: Optional[np.ndarray] = None
    ) -> ResidualDiagnostics:
        """
        Run comprehensive residual diagnostics

        Args:
            exog: Exogenous variables for heteroskedasticity tests

        Returns:
            ResidualDiagnostics object with all results
        """
        logger.info("Running comprehensive residual diagnostics")

        # Run all tests
        normality = self.test_normality()
        autocorrelation = self.test_autocorrelation()
        heteroskedasticity = self.test_heteroskedasticity(exog)

        # Create summary
        summary_parts = []

        # Normality summary
        jb_normal = normality['jarque_bera']['is_normal']
        summary_parts.append(
            f"Normality: {'PASS' if jb_normal else 'FAIL'} (Jarque-Bera p={normality['jarque_bera']['p_value']:.4f})"
        )

        # Autocorrelation summary
        has_autocorr = autocorrelation['ljung_box']['has_autocorrelation']
        summary_parts.append(
            f"Autocorrelation: {'DETECTED' if has_autocorr else 'NONE'} (Ljung-Box test)"
        )

        # Heteroskedasticity summary
        if 'white' in heteroskedasticity and 'has_heteroskedasticity' in heteroskedasticity['white']:
            has_hetero = heteroskedasticity['white']['has_heteroskedasticity']
            summary_parts.append(
                f"Heteroskedasticity: {'DETECTED' if has_hetero else 'NONE'} (White's test)"
            )

        summary = " | ".join(summary_parts)

        return ResidualDiagnostics(
            normality=normality,
            autocorrelation=autocorrelation,
            heteroskedasticity=heteroskedasticity,
            summary=summary
        )


def print_stationarity_results(results: Dict[str, StationarityTestResult]):
    """
    Print stationarity test results in formatted table

    Args:
        results: Dictionary of stationarity test results
    """
    print("\n" + "=" * 80)
    print("STATIONARITY TEST RESULTS")
    print("=" * 80)

    for test_name, result in results.items():
        print(f"\n{result.test_name} Test ({test_name}):")
        print(f"  Test Statistic: {result.test_statistic:.6f}")
        print(f"  P-value: {result.p_value:.6f}")
        print(f"  Critical Values: {result.critical_values}")
        print(f"  Conclusion: {result.conclusion}")

    print("=" * 80 + "\n")


def print_residual_diagnostics(diagnostics: ResidualDiagnostics):
    """
    Print residual diagnostics in formatted table

    Args:
        diagnostics: ResidualDiagnostics object
    """
    print("\n" + "=" * 80)
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 80)

    print(f"\nSummary: {diagnostics.summary}\n")

    print("Normality Tests:")
    for test_name, results in diagnostics.normality.items():
        if isinstance(results, dict):
            print(f"  {test_name}:")
            for key, value in results.items():
                if key != 'critical_values':
                    print(f"    {key}: {value}")

    print("\nAutocorrelation Tests:")
    print(f"  Durbin-Watson: {diagnostics.autocorrelation['durbin_watson']['statistic']:.4f}")
    print(f"  Interpretation: {diagnostics.autocorrelation['durbin_watson']['interpretation']}")

    if 'ljung_box' in diagnostics.autocorrelation:
        has_autocorr = diagnostics.autocorrelation['ljung_box']['has_autocorrelation']
        print(f"  Ljung-Box: {'Autocorrelation detected' if has_autocorr else 'No significant autocorrelation'}")

    print("\nHeteroskedasticity Tests:")
    for test_name, results in diagnostics.heteroskedasticity.items():
        if isinstance(results, dict) and 'error' not in results:
            print(f"  {test_name}:")
            for key, value in results.items():
                print(f"    {key}: {value}")

    print("=" * 80 + "\n")
