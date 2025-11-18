"""
Factor Models and Regime Detection for Quantitative Finance
Implements factor-based asset pricing models and market regime identification

This module provides:
- CAPM (Capital Asset Pricing Model)
- Fama-French 3-Factor and 5-Factor models
- Carhart 4-Factor model (momentum)
- Rolling factor exposures
- Hidden Markov Models for regime detection
- Regime-switching strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Try to import HMM libraries
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Install with: pip install hmmlearn")


@dataclass
class FactorModelResult:
    """Results from factor model regression"""
    model_name: str
    alpha: float
    alpha_pvalue: float
    betas: Dict[str, float]
    beta_pvalues: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    residuals: np.ndarray
    fitted_values: np.ndarray


@dataclass
class RegimeDetectionResult:
    """Results from regime detection"""
    n_regimes: int
    regime_labels: np.ndarray
    regime_probabilities: np.ndarray
    regime_means: np.ndarray
    regime_volatilities: np.ndarray
    transition_matrix: np.ndarray
    most_likely_regime: int


class CAPM:
    """
    Capital Asset Pricing Model

    E[R_i] = R_f + β_i(E[R_m] - R_f)

    Where:
    - R_i: Return on asset i
    - R_f: Risk-free rate
    - R_m: Market return
    - β_i: Beta (systematic risk)
    """

    def __init__(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize CAPM model

        Args:
            asset_returns: Asset returns
            market_returns: Market/benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        # Align data
        self.data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()

        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        # Calculate excess returns
        self.asset_excess = self.data['asset'] - self.daily_rf
        self.market_excess = self.data['market'] - self.daily_rf

        logger.info("Initialized CAPM model")

    def fit(self) -> FactorModelResult:
        """
        Fit CAPM model using OLS regression

        Returns:
            FactorModelResult with alpha, beta, and statistics
        """
        logger.info("Fitting CAPM model")

        # Prepare data for regression
        X = self.market_excess.values.reshape(-1, 1)
        y = self.asset_excess.values

        # OLS regression
        model = LinearRegression()
        model.fit(X, y)

        # Get parameters
        beta = model.coef_[0]
        alpha = model.intercept_

        # Calculate fitted values and residuals
        fitted_values = model.predict(X)
        residuals = y - fitted_values

        # Calculate statistics
        n = len(y)
        k = 1  # number of predictors

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Standard errors
        mse = ss_res / (n - k - 1)
        var_beta = mse / np.sum((X.flatten() - np.mean(X)) ** 2)
        se_beta = np.sqrt(var_beta)

        var_alpha = mse * (1/n + np.mean(X)**2 / np.sum((X.flatten() - np.mean(X)) ** 2))
        se_alpha = np.sqrt(var_alpha)

        # t-statistics and p-values
        t_alpha = alpha / se_alpha
        t_beta = beta / se_beta

        p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - k - 1))
        p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - k - 1))

        # Annualize alpha
        alpha_annual = alpha * self.periods_per_year

        return FactorModelResult(
            model_name='CAPM',
            alpha=alpha_annual,
            alpha_pvalue=p_alpha,
            betas={'market_beta': beta},
            beta_pvalues={'market_beta': p_beta},
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            residuals=residuals,
            fitted_values=fitted_values
        )

    def rolling_beta(
        self,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling beta

        Args:
            window: Rolling window size

        Returns:
            Series of rolling betas
        """
        logger.info(f"Calculating rolling beta (window={window})")

        def calc_beta(y, x):
            if len(y) < 2 or len(x) < 2:
                return np.nan
            covariance = np.cov(y, x)[0, 1]
            variance = np.var(x)
            return covariance / variance if variance != 0 else np.nan

        rolling_betas = []
        for i in range(len(self.asset_excess)):
            if i < window:
                rolling_betas.append(np.nan)
            else:
                y = self.asset_excess.iloc[i-window:i].values
                x = self.market_excess.iloc[i-window:i].values
                beta = calc_beta(y, x)
                rolling_betas.append(beta)

        return pd.Series(rolling_betas, index=self.asset_excess.index)


class FamaFrenchModel:
    """
    Fama-French Factor Models

    3-Factor Model:
    R_i - R_f = α + β_1(R_m - R_f) + β_2(SMB) + β_3(HML) + ε

    5-Factor Model adds:
    + β_4(RMW) + β_5(CMA)

    Where:
    - SMB: Small Minus Big (size factor)
    - HML: High Minus Low (value factor)
    - RMW: Robust Minus Weak (profitability factor)
    - CMA: Conservative Minus Aggressive (investment factor)
    """

    def __init__(
        self,
        asset_returns: pd.Series,
        factor_data: pd.DataFrame,
        periods_per_year: int = 252
    ):
        """
        Initialize Fama-French model

        Args:
            asset_returns: Asset returns
            factor_data: DataFrame with factor returns (columns: 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF')
            periods_per_year: Trading periods per year
        """
        # Align data
        self.data = pd.concat([asset_returns, factor_data], axis=1).dropna()
        self.asset_returns = asset_returns.name if asset_returns.name else 'asset'
        self.periods_per_year = periods_per_year

        logger.info("Initialized Fama-French model")

    def fit_three_factor(self) -> FactorModelResult:
        """
        Fit Fama-French 3-factor model

        Returns:
            FactorModelResult with factor loadings
        """
        logger.info("Fitting Fama-French 3-Factor model")

        required_factors = ['Mkt-RF', 'SMB', 'HML', 'RF']
        if not all(f in self.data.columns for f in required_factors):
            raise ValueError(f"Missing required factors. Need: {required_factors}")

        # Calculate excess returns
        y = (self.data[self.asset_returns] - self.data['RF']).values

        # Prepare factors
        X = self.data[['Mkt-RF', 'SMB', 'HML']].values

        # OLS regression
        model = LinearRegression()
        model.fit(X, y)

        # Get parameters
        alpha = model.intercept_
        betas = model.coef_

        # Calculate fitted values and residuals
        fitted_values = model.predict(X)
        residuals = y - fitted_values

        # Calculate statistics
        n = len(y)
        k = 3  # number of factors

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Standard errors (simplified)
        mse = ss_res / (n - k - 1)

        # For simplicity, using approximate standard errors
        se_alpha = np.sqrt(mse / n)
        t_alpha = alpha / se_alpha
        p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - k - 1))

        # Beta standard errors and p-values
        beta_dict = {
            'market_beta': betas[0],
            'smb_beta': betas[1],
            'hml_beta': betas[2]
        }

        beta_pvalues = {}
        for i, name in enumerate(['market_beta', 'smb_beta', 'hml_beta']):
            se_beta = np.sqrt(mse / n)  # Simplified
            t_beta = betas[i] / se_beta
            p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - k - 1))
            beta_pvalues[name] = p_beta

        # Annualize alpha
        alpha_annual = alpha * self.periods_per_year

        return FactorModelResult(
            model_name='Fama-French 3-Factor',
            alpha=alpha_annual,
            alpha_pvalue=p_alpha,
            betas=beta_dict,
            beta_pvalues=beta_pvalues,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            residuals=residuals,
            fitted_values=fitted_values
        )

    def fit_five_factor(self) -> FactorModelResult:
        """
        Fit Fama-French 5-factor model

        Returns:
            FactorModelResult with factor loadings
        """
        logger.info("Fitting Fama-French 5-Factor model")

        required_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        if not all(f in self.data.columns for f in required_factors):
            raise ValueError(f"Missing required factors. Need: {required_factors}")

        # Calculate excess returns
        y = (self.data[self.asset_returns] - self.data['RF']).values

        # Prepare factors
        X = self.data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values

        # OLS regression
        model = LinearRegression()
        model.fit(X, y)

        # Get parameters
        alpha = model.intercept_
        betas = model.coef_

        # Calculate fitted values and residuals
        fitted_values = model.predict(X)
        residuals = y - fitted_values

        # Calculate statistics
        n = len(y)
        k = 5  # number of factors

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Standard errors
        mse = ss_res / (n - k - 1)
        se_alpha = np.sqrt(mse / n)
        t_alpha = alpha / se_alpha
        p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - k - 1))

        # Beta p-values
        beta_dict = {
            'market_beta': betas[0],
            'smb_beta': betas[1],
            'hml_beta': betas[2],
            'rmw_beta': betas[3],
            'cma_beta': betas[4]
        }

        beta_pvalues = {}
        factor_names = ['market_beta', 'smb_beta', 'hml_beta', 'rmw_beta', 'cma_beta']
        for i, name in enumerate(factor_names):
            se_beta = np.sqrt(mse / n)  # Simplified
            t_beta = betas[i] / se_beta
            p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - k - 1))
            beta_pvalues[name] = p_beta

        # Annualize alpha
        alpha_annual = alpha * self.periods_per_year

        return FactorModelResult(
            model_name='Fama-French 5-Factor',
            alpha=alpha_annual,
            alpha_pvalue=p_alpha,
            betas=beta_dict,
            beta_pvalues=beta_pvalues,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            residuals=residuals,
            fitted_values=fitted_values
        )


class RegimeDetection:
    """
    Market Regime Detection using Hidden Markov Models

    Identifies different market regimes (e.g., bull, bear, high volatility)
    which is crucial for:
    - Adaptive trading strategies
    - Risk management
    - Portfolio allocation
    """

    def __init__(self, returns: Union[pd.Series, np.ndarray], n_regimes: int = 2):
        """
        Initialize regime detection

        Args:
            returns: Return series
            n_regimes: Number of hidden regimes to detect
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for regime detection. Install with: pip install hmmlearn")

        self.returns = np.array(returns).reshape(-1, 1) if isinstance(returns, pd.Series) else returns.reshape(-1, 1)
        self.returns = self.returns[~np.isnan(self.returns).any(axis=1)]
        self.n_regimes = n_regimes
        self.model = None

        logger.info(f"Initialized regime detection with {n_regimes} regimes")

    def fit(
        self,
        n_iter: int = 100,
        covariance_type: str = 'full'
    ) -> RegimeDetectionResult:
        """
        Fit Hidden Markov Model to detect regimes

        Args:
            n_iter: Number of EM iterations
            covariance_type: Type of covariance parameters ('full', 'diag', 'spherical')

        Returns:
            RegimeDetectionResult with detected regimes
        """
        logger.info("Fitting HMM for regime detection")

        # Create and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42
        )

        self.model.fit(self.returns)

        # Predict regimes
        regime_labels = self.model.predict(self.returns)
        regime_probabilities = self.model.predict_proba(self.returns)

        # Calculate regime statistics
        regime_means = self.model.means_.flatten()
        regime_volatilities = np.sqrt(self.model.covars_.flatten())

        # Get transition matrix
        transition_matrix = self.model.transmat_

        # Determine most likely current regime
        most_likely_regime = regime_labels[-1]

        logger.info(f"Detected regimes - Means: {regime_means}, Volatilities: {regime_volatilities}")

        return RegimeDetectionResult(
            n_regimes=self.n_regimes,
            regime_labels=regime_labels,
            regime_probabilities=regime_probabilities,
            regime_means=regime_means,
            regime_volatilities=regime_volatilities,
            transition_matrix=transition_matrix,
            most_likely_regime=most_likely_regime
        )

    def predict_regime(self, new_returns: np.ndarray) -> int:
        """
        Predict regime for new data

        Args:
            new_returns: New return observations

        Returns:
            Predicted regime label
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        new_returns = new_returns.reshape(-1, 1)
        regime = self.model.predict(new_returns)
        return regime[-1]


def print_factor_model_results(result: FactorModelResult):
    """
    Print factor model results in formatted table

    Args:
        result: FactorModelResult object
    """
    print("\n" + "=" * 80)
    print(f"{result.model_name.upper()} RESULTS")
    print("=" * 80)

    print(f"\nAlpha: {result.alpha*100:.4f}% (annualized)")
    print(f"  p-value: {result.alpha_pvalue:.4f} {'***' if result.alpha_pvalue < 0.01 else '**' if result.alpha_pvalue < 0.05 else '*' if result.alpha_pvalue < 0.1 else ''}")

    print("\nFactor Loadings (Betas):")
    for factor, beta in result.betas.items():
        p_value = result.beta_pvalues[factor]
        significance = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
        print(f"  {factor:20s}: {beta:7.4f}  (p={p_value:.4f}) {significance}")

    print(f"\nModel Fit:")
    print(f"  R-squared:     {result.r_squared:.4f}")
    print(f"  Adj R-squared: {result.adj_r_squared:.4f}")

    print("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1")
    print("=" * 80 + "\n")


def print_regime_results(result: RegimeDetectionResult):
    """
    Print regime detection results

    Args:
        result: RegimeDetectionResult object
    """
    print("\n" + "=" * 80)
    print(f"REGIME DETECTION - {result.n_regimes} REGIMES")
    print("=" * 80)

    print("\nRegime Characteristics:")
    for i in range(result.n_regimes):
        print(f"\nRegime {i}:")
        print(f"  Mean Return:   {result.regime_means[i]*100:6.2f}%")
        print(f"  Volatility:    {result.regime_volatilities[i]*100:6.2f}%")

    print("\nTransition Matrix:")
    print("(Probability of transitioning from regime i to regime j)")
    print(pd.DataFrame(
        result.transition_matrix,
        index=[f"Regime {i}" for i in range(result.n_regimes)],
        columns=[f"→ Regime {i}" for i in range(result.n_regimes)]
    ))

    print(f"\nCurrent Most Likely Regime: Regime {result.most_likely_regime}")

    # Calculate regime persistence
    print("\nRegime Persistence (probability of staying in same regime):")
    for i in range(result.n_regimes):
        persistence = result.transition_matrix[i, i]
        print(f"  Regime {i}: {persistence*100:.2f}%")

    print("=" * 80 + "\n")


def calculate_rolling_factor_exposures(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling factor exposures over time

    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        DataFrame with rolling alpha and beta
    """
    logger.info(f"Calculating rolling factor exposures (window={window})")

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # Align data
    data = pd.DataFrame({
        'asset': asset_returns,
        'market': market_returns
    }).dropna()

    asset_excess = data['asset'] - daily_rf
    market_excess = data['market'] - daily_rf

    rolling_alpha = []
    rolling_beta = []
    rolling_r2 = []

    for i in range(len(data)):
        if i < window:
            rolling_alpha.append(np.nan)
            rolling_beta.append(np.nan)
            rolling_r2.append(np.nan)
        else:
            y = asset_excess.iloc[i-window:i].values
            X = market_excess.iloc[i-window:i].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            alpha = model.intercept_ * periods_per_year  # Annualize
            beta = model.coef_[0]

            # R-squared
            predictions = model.predict(X)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            rolling_alpha.append(alpha)
            rolling_beta.append(beta)
            rolling_r2.append(r2)

    results = pd.DataFrame({
        'alpha': rolling_alpha,
        'beta': rolling_beta,
        'r_squared': rolling_r2
    }, index=data.index)

    return results
