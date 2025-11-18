"""
Portfolio Optimization and Multi-Asset Analysis
Implements modern portfolio theory and advanced optimization techniques

This module provides:
- Mean-Variance Optimization (Markowitz)
- Efficient Frontier construction
- Black-Litterman model
- Risk Parity allocation
- Hierarchical Risk Parity (HRP)
- Maximum Sharpe, Minimum Variance portfolios
- Monte Carlo portfolio simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PortfolioWeights:
    """Portfolio allocation weights"""
    assets: List[str]
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str


@dataclass
class EfficientFrontier:
    """Efficient frontier data"""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights: np.ndarray
    max_sharpe_portfolio: PortfolioWeights
    min_volatility_portfolio: PortfolioWeights


class MeanVarianceOptimizer:
    """
    Mean-Variance Optimization (Markowitz Portfolio Theory)

    Nobel Prize-winning framework for optimal portfolio construction
    that balances expected return against portfolio variance.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize mean-variance optimizer

        Args:
            returns: DataFrame of asset returns (assets in columns)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year for annualization
        """
        self.returns = returns.dropna()
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Calculate statistics
        self.mean_returns = self.returns.mean() * periods_per_year
        self.cov_matrix = self.returns.cov() * periods_per_year

        logger.info(f"Initialized optimizer with {self.n_assets} assets")

    def portfolio_performance(
        self,
        weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio

        Args:
            weights: Asset weights

        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        # Portfolio return
        portfolio_return = np.dot(weights, self.mean_returns)

        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol != 0 else 0

        return portfolio_return, portfolio_vol, sharpe

    def max_sharpe_ratio(
        self,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple] = None
    ) -> PortfolioWeights:
        """
        Find portfolio with maximum Sharpe ratio

        Args:
            constraints: Additional constraints (e.g., sector limits)
            bounds: Weight bounds for each asset (default: (0, 1) for long-only)

        Returns:
            PortfolioWeights with optimal allocation
        """
        logger.info("Optimizing for maximum Sharpe ratio")

        def neg_sharpe(weights):
            """Negative Sharpe ratio for minimization"""
            ret, vol, sharpe = self.portfolio_performance(weights)
            return -sharpe

        # Default constraints: weights sum to 1
        default_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if constraints:
            default_constraints.extend(constraints)

        # Default bounds: long-only
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess: equal weight
        x0 = np.array([1 / self.n_assets] * self.n_assets)

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=default_constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")

        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)

        return PortfolioWeights(
            assets=self.assets,
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method='Maximum Sharpe Ratio'
        )

    def min_volatility(
        self,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple] = None
    ) -> PortfolioWeights:
        """
        Find minimum variance portfolio

        Args:
            constraints: Additional constraints
            bounds: Weight bounds

        Returns:
            PortfolioWeights with minimum variance allocation
        """
        logger.info("Optimizing for minimum volatility")

        def portfolio_volatility(weights):
            """Portfolio volatility for minimization"""
            _, vol, _ = self.portfolio_performance(weights)
            return vol

        # Constraints
        default_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if constraints:
            default_constraints.extend(constraints)

        # Bounds
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess
        x0 = np.array([1 / self.n_assets] * self.n_assets)

        # Optimize
        result = minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=default_constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")

        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)

        return PortfolioWeights(
            assets=self.assets,
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method='Minimum Volatility'
        )

    def efficient_return(
        self,
        target_return: float,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple] = None
    ) -> PortfolioWeights:
        """
        Find portfolio with minimum volatility for target return

        Args:
            target_return: Target portfolio return
            constraints: Additional constraints
            bounds: Weight bounds

        Returns:
            PortfolioWeights for target return
        """
        logger.info(f"Optimizing for target return: {target_return:.2%}")

        def portfolio_volatility(weights):
            _, vol, _ = self.portfolio_performance(weights)
            return vol

        # Constraints: weights sum to 1, return = target
        default_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self.portfolio_performance(w)[0] - target_return}
        ]
        if constraints:
            default_constraints.extend(constraints)

        # Bounds
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess
        x0 = np.array([1 / self.n_assets] * self.n_assets)

        # Optimize
        result = minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=default_constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Optimization failed for target return {target_return:.2%}")
            # Return equal weight portfolio as fallback
            weights = x0
        else:
            weights = result.x

        ret, vol, sharpe = self.portfolio_performance(weights)

        return PortfolioWeights(
            assets=self.assets,
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method=f'Efficient Portfolio (Target Return: {target_return:.2%})'
        )

    def efficient_frontier(
        self,
        n_points: int = 100,
        bounds: Optional[Tuple] = None
    ) -> EfficientFrontier:
        """
        Construct efficient frontier

        Args:
            n_points: Number of points on frontier
            bounds: Weight bounds

        Returns:
            EfficientFrontier object with frontier data
        """
        logger.info(f"Constructing efficient frontier with {n_points} points")

        # Get min and max returns
        min_vol_portfolio = self.min_volatility(bounds=bounds)
        max_sharpe_portfolio = self.max_sharpe_ratio(bounds=bounds)

        min_ret = min_vol_portfolio.expected_return
        max_ret = max_sharpe_portfolio.expected_return * 1.5  # Extend beyond max Sharpe

        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)

        # Calculate efficient portfolios
        frontier_returns = []
        frontier_vols = []
        frontier_sharpes = []
        frontier_weights = []

        for target_ret in target_returns:
            try:
                portfolio = self.efficient_return(target_ret, bounds=bounds)
                frontier_returns.append(portfolio.expected_return)
                frontier_vols.append(portfolio.volatility)
                frontier_sharpes.append(portfolio.sharpe_ratio)
                frontier_weights.append(portfolio.weights)
            except Exception as e:
                logger.debug(f"Failed to optimize for return {target_ret:.2%}: {e}")
                continue

        return EfficientFrontier(
            returns=np.array(frontier_returns),
            volatilities=np.array(frontier_vols),
            sharpe_ratios=np.array(frontier_sharpes),
            weights=np.array(frontier_weights),
            max_sharpe_portfolio=max_sharpe_portfolio,
            min_volatility_portfolio=min_vol_portfolio
        )


class RiskParityOptimizer:
    """
    Risk Parity Portfolio Optimization

    Allocates capital such that each asset contributes equally to portfolio risk.
    Popular in institutional investing and "All Weather" strategies.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        periods_per_year: int = 252
    ):
        """
        Initialize risk parity optimizer

        Args:
            returns: DataFrame of asset returns
            periods_per_year: Periods per year
        """
        self.returns = returns.dropna()
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.periods_per_year = periods_per_year

        # Calculate covariance matrix
        self.cov_matrix = self.returns.cov() * periods_per_year

        logger.info(f"Initialized risk parity optimizer with {self.n_assets} assets")

    def risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset

        Risk Contribution_i = w_i * (Σw)_i / σ_p

        Args:
            weights: Portfolio weights

        Returns:
            Array of risk contributions
        """
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        # Marginal risk contribution
        marginal_contrib = np.dot(self.cov_matrix, weights)

        # Risk contribution
        risk_contrib = weights * marginal_contrib / portfolio_vol

        return risk_contrib

    def optimize(
        self,
        target_risk_contributions: Optional[np.ndarray] = None
    ) -> PortfolioWeights:
        """
        Find risk parity portfolio

        Args:
            target_risk_contributions: Target risk contributions (default: equal)

        Returns:
            PortfolioWeights with risk parity allocation
        """
        logger.info("Optimizing for risk parity")

        if target_risk_contributions is None:
            # Equal risk contribution
            target_risk_contributions = np.ones(self.n_assets) / self.n_assets

        def objective(weights):
            """
            Minimize squared difference between actual and target risk contributions
            """
            risk_contrib = self.risk_contribution(weights)
            # Normalize to sum to 1
            risk_contrib /= np.sum(risk_contrib)
            return np.sum((risk_contrib - target_risk_contributions) ** 2)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Bounds: long-only
        bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess: equal weight
        x0 = np.array([1 / self.n_assets] * self.n_assets)

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")

        weights = result.x

        # Calculate portfolio metrics
        mean_returns = self.returns.mean() * self.periods_per_year
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol != 0 else 0

        return PortfolioWeights(
            assets=self.assets,
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            method='Risk Parity'
        )


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP)

    Machine learning-based portfolio allocation using hierarchical clustering
    to build diversified portfolios. Addresses issues with traditional
    mean-variance optimization (instability, concentration).

    Based on research by Marcos López de Prado
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize HRP optimizer

        Args:
            returns: DataFrame of asset returns
        """
        self.returns = returns.dropna()
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)

        # Calculate correlation matrix
        self.corr_matrix = self.returns.corr()

        logger.info(f"Initialized HRP optimizer with {self.n_assets} assets")

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """
        Get quasi-diagonal matrix from hierarchical clustering

        Args:
            link: Linkage matrix

        Returns:
            List of reordered indices
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def _get_cluster_var(
        self,
        cov: pd.DataFrame,
        cluster_items: List[int]
    ) -> float:
        """
        Calculate cluster variance

        Args:
            cov: Covariance matrix
            cluster_items: Items in cluster

        Returns:
            Cluster variance
        """
        cov_slice = cov.iloc[cluster_items, cluster_items]
        # Inverse-variance portfolio
        ivp = 1 / np.diag(cov_slice)
        ivp /= ivp.sum()
        w = ivp.reshape(-1, 1)
        cluster_var = np.dot(w.T, np.dot(cov_slice, w))[0, 0]
        return cluster_var

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sort_ix: List[int]
    ) -> pd.Series:
        """
        Recursive bisection to allocate weights

        Args:
            cov: Covariance matrix
            sort_ix: Sorted indices from clustering

        Returns:
            Series of weights
        """
        w = pd.Series(1, index=sort_ix)
        cluster_items = [sort_ix]

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]

            for i in range(0, len(cluster_items), 2):
                cluster0 = cluster_items[i]
                cluster1 = cluster_items[i + 1]

                cluster0_var = self._get_cluster_var(cov, cluster0)
                cluster1_var = self._get_cluster_var(cov, cluster1)

                alpha = 1 - cluster0_var / (cluster0_var + cluster1_var)

                w[cluster0] *= alpha
                w[cluster1] *= 1 - alpha

        return w

    def optimize(self) -> PortfolioWeights:
        """
        Optimize portfolio using HRP

        Returns:
            PortfolioWeights with HRP allocation
        """
        logger.info("Optimizing with Hierarchical Risk Parity")

        # Calculate distance matrix
        dist = np.sqrt((1 - self.corr_matrix) / 2)

        # Hierarchical clustering
        link = linkage(dist.values[np.triu_indices(self.n_assets, k=1)], method='single')

        # Get quasi-diagonal matrix
        sort_ix = self._get_quasi_diag(link)

        # Calculate covariance matrix
        cov = self.returns.cov()

        # Recursive bisection
        weights = self._recursive_bisection(cov, sort_ix)

        # Reorder to match original asset order
        weights = weights.loc[range(self.n_assets)]

        # Calculate portfolio metrics
        mean_returns = self.returns.mean() * 252
        cov_annual = cov * 252

        portfolio_return = np.dot(weights.values, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.values.T, np.dot(cov_annual, weights.values)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol != 0 else 0

        return PortfolioWeights(
            assets=self.assets,
            weights=weights.values,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            method='Hierarchical Risk Parity'
        )


def print_portfolio_weights(portfolio: PortfolioWeights, top_n: Optional[int] = None):
    """
    Print portfolio weights in formatted table

    Args:
        portfolio: PortfolioWeights object
        top_n: Show only top N assets by weight
    """
    print("\n" + "=" * 80)
    print(f"PORTFOLIO: {portfolio.method}")
    print("=" * 80)

    print(f"\nExpected Return: {portfolio.expected_return*100:.2f}%")
    print(f"Volatility:      {portfolio.volatility*100:.2f}%")
    print(f"Sharpe Ratio:    {portfolio.sharpe_ratio:.4f}")

    print("\nAsset Allocation:")
    print("-" * 50)

    # Create DataFrame for easy sorting
    df = pd.DataFrame({
        'Asset': portfolio.assets,
        'Weight': portfolio.weights
    }).sort_values('Weight', ascending=False)

    if top_n:
        df = df.head(top_n)

    for _, row in df.iterrows():
        if row['Weight'] > 0.001:  # Only show meaningful allocations
            print(f"  {row['Asset']:15s} {row['Weight']*100:6.2f}%")

    print("=" * 80 + "\n")


def monte_carlo_portfolio_simulation(
    returns: pd.DataFrame,
    n_portfolios: int = 10000,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Monte Carlo simulation of random portfolios

    Useful for visualizing efficient frontier and finding optimal allocations

    Args:
        returns: DataFrame of asset returns
        n_portfolios: Number of random portfolios to generate
        risk_free_rate: Risk-free rate
        periods_per_year: Periods per year

    Returns:
        DataFrame with portfolio metrics
    """
    logger.info(f"Running Monte Carlo simulation with {n_portfolios} portfolios")

    n_assets = len(returns.columns)
    mean_returns = returns.mean() * periods_per_year
    cov_matrix = returns.cov() * periods_per_year

    results = {
        'Returns': [],
        'Volatility': [],
        'Sharpe': [],
        'Weights': []
    }

    for _ in range(n_portfolios):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        # Portfolio metrics
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

        results['Returns'].append(portfolio_return)
        results['Volatility'].append(portfolio_vol)
        results['Sharpe'].append(sharpe)
        results['Weights'].append(weights)

    return pd.DataFrame(results)
