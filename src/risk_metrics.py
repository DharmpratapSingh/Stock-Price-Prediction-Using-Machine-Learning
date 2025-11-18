"""
Advanced Risk Metrics for Quantitative Finance
Implements industry-standard risk measures used by hedge funds and asset managers

This module provides comprehensive risk analysis including:
- Value at Risk (VaR): Historical, Parametric, Monte Carlo
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Information Ratio, Tracking Error
- Downside risk metrics
- Tail risk measures
- Risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Value at Risk calculation results"""
    confidence_level: float
    var: float
    cvar: float  # Conditional VaR (Expected Shortfall)
    method: str
    worst_cases: np.ndarray


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a strategy"""
    # Volatility measures
    volatility_annual: float
    downside_deviation: float
    semi_variance: float

    # VaR measures
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Drawdown measures
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Tail risk
    skewness: float
    kurtosis: float
    tail_ratio: float

    # Benchmark-relative
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    treynor_ratio: Optional[float] = None


class ValueAtRisk:
    """
    Value at Risk (VaR) calculations using multiple methods

    VaR estimates the potential loss in portfolio value over a defined period
    for a given confidence interval.
    """

    def __init__(self, returns: np.ndarray):
        """
        Initialize VaR calculator

        Args:
            returns: Array of returns (not prices)
        """
        self.returns = np.array(returns)
        self.returns = self.returns[~np.isnan(self.returns)]

    def historical_var(
        self,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> VaRResult:
        """
        Historical VaR using empirical distribution

        Non-parametric method - makes no assumptions about return distribution

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            horizon: Time horizon in days

        Returns:
            VaRResult with VaR and CVaR
        """
        logger.info(f"Calculating historical VaR at {confidence_level*100}% confidence")

        # Scale returns for horizon
        if horizon > 1:
            scaled_returns = self.returns * np.sqrt(horizon)
        else:
            scaled_returns = self.returns

        # Calculate VaR as percentile
        var = -np.percentile(scaled_returns, (1 - confidence_level) * 100)

        # Calculate CVaR (expected value of losses exceeding VaR)
        losses_beyond_var = scaled_returns[scaled_returns <= -var]
        cvar = -np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var

        # Get worst cases
        n_worst = max(int(len(scaled_returns) * (1 - confidence_level)), 5)
        worst_cases = np.sort(scaled_returns)[:n_worst]

        return VaRResult(
            confidence_level=confidence_level,
            var=var,
            cvar=cvar,
            method='Historical',
            worst_cases=worst_cases
        )

    def parametric_var(
        self,
        confidence_level: float = 0.95,
        horizon: int = 1,
        distribution: str = 'normal'
    ) -> VaRResult:
        """
        Parametric VaR assuming a distribution

        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            distribution: Distribution assumption ('normal' or 't')

        Returns:
            VaRResult with VaR and CVaR
        """
        logger.info(f"Calculating parametric VaR ({distribution}) at {confidence_level*100}% confidence")

        mu = np.mean(self.returns)
        sigma = np.std(self.returns)

        # Scale for horizon
        mu_scaled = mu * horizon
        sigma_scaled = sigma * np.sqrt(horizon)

        if distribution == 'normal':
            # Normal distribution
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mu_scaled + z_score * sigma_scaled)

            # CVaR for normal distribution (analytical formula)
            cvar = -(mu_scaled - sigma_scaled * stats.norm.pdf(z_score) / (1 - confidence_level))

        elif distribution == 't':
            # Student's t-distribution (captures fat tails)
            df = 6  # degrees of freedom (can be estimated)
            t_score = stats.t.ppf(1 - confidence_level, df)
            var = -(mu_scaled + t_score * sigma_scaled * np.sqrt((df - 2) / df))

            # CVaR for t-distribution (analytical formula)
            cvar = -(mu_scaled - sigma_scaled * np.sqrt((df - 2) / df) *
                    stats.t.pdf(t_score, df) * (df + t_score**2) / ((df - 1) * (1 - confidence_level)))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Get worst cases from empirical data
        n_worst = max(int(len(self.returns) * (1 - confidence_level)), 5)
        worst_cases = np.sort(self.returns)[:n_worst]

        return VaRResult(
            confidence_level=confidence_level,
            var=var,
            cvar=cvar,
            method=f'Parametric ({distribution})',
            worst_cases=worst_cases
        )

    def monte_carlo_var(
        self,
        confidence_level: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 10000,
        method: str = 'normal'
    ) -> VaRResult:
        """
        Monte Carlo VaR using simulated paths

        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            n_simulations: Number of Monte Carlo simulations
            method: Simulation method ('normal', 'bootstrap', 'garch')

        Returns:
            VaRResult with VaR and CVaR
        """
        logger.info(f"Calculating Monte Carlo VaR ({method}) at {confidence_level*100}% confidence")

        if method == 'normal':
            # Simulate from normal distribution
            mu = np.mean(self.returns)
            sigma = np.std(self.returns)
            simulated_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_simulations)

        elif method == 'bootstrap':
            # Bootstrap resampling
            simulated_returns = np.random.choice(self.returns, size=(n_simulations, horizon))
            simulated_returns = np.sum(simulated_returns, axis=1)

        elif method == 'garch':
            # GARCH simulation (simplified)
            # In practice, would use arch package
            mu = np.mean(self.returns)
            sigma = np.std(self.returns)
            simulated_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_simulations)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate VaR and CVaR from simulations
        var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)

        losses_beyond_var = simulated_returns[simulated_returns <= -var]
        cvar = -np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var

        # Get worst cases
        n_worst = max(int(n_simulations * (1 - confidence_level)), 5)
        worst_cases = np.sort(simulated_returns)[:n_worst]

        return VaRResult(
            confidence_level=confidence_level,
            var=var,
            cvar=cvar,
            method=f'Monte Carlo ({method})',
            worst_cases=worst_cases
        )


class RiskAnalyzer:
    """
    Comprehensive risk analysis for trading strategies

    Implements risk metrics used by professional quantitative traders and
    institutional investors.
    """

    def __init__(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize risk analyzer

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns for relative metrics
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (252 for daily, 52 for weekly)
        """
        self.returns = np.array(returns)
        self.returns = self.returns[~np.isnan(self.returns)]
        self.benchmark_returns = (
            np.array(benchmark_returns)[~np.isnan(benchmark_returns)]
            if benchmark_returns is not None else None
        )
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    def volatility(self, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)

        Args:
            annualize: Whether to annualize the volatility

        Returns:
            Volatility value
        """
        vol = np.std(self.returns)
        if annualize:
            vol *= np.sqrt(self.periods_per_year)
        return vol

    def downside_deviation(
        self,
        target: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        Calculate downside deviation (volatility of negative returns)

        Uses only returns below target (typically 0 or MAR)

        Args:
            target: Target return threshold
            annualize: Whether to annualize

        Returns:
            Downside deviation
        """
        downside_returns = self.returns[self.returns < target]
        if len(downside_returns) == 0:
            return 0.0

        downside_dev = np.sqrt(np.mean((downside_returns - target) ** 2))
        if annualize:
            downside_dev *= np.sqrt(self.periods_per_year)
        return downside_dev

    def semi_variance(self, target: float = 0.0) -> float:
        """
        Calculate semi-variance (variance of returns below target)

        Args:
            target: Target return threshold

        Returns:
            Semi-variance
        """
        downside_returns = self.returns[self.returns < target]
        if len(downside_returns) == 0:
            return 0.0
        return np.var(downside_returns - target)

    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return)

        Sharpe = (Return - RiskFree) / Volatility

        Returns:
            Sharpe ratio (annualized)
        """
        excess_returns = self.returns - self.daily_rf
        if np.std(self.returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(self.returns) * np.sqrt(self.periods_per_year)

    def sortino_ratio(self, target: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (Sharpe with downside deviation)

        Sortino = (Return - Target) / DownsideDeviation

        Args:
            target: Minimum acceptable return

        Returns:
            Sortino ratio (annualized)
        """
        excess_returns = self.returns - target
        downside_dev = self.downside_deviation(target, annualize=False)

        if downside_dev == 0:
            return 0.0

        return np.mean(excess_returns) / downside_dev * np.sqrt(self.periods_per_year)

    def calmar_ratio(self, prices: np.ndarray) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown)

        Args:
            prices: Price series (not returns)

        Returns:
            Calmar ratio
        """
        annual_return = np.mean(self.returns) * self.periods_per_year
        max_dd = self.maximum_drawdown(prices)[0]

        if max_dd == 0:
            return 0.0
        return annual_return / max_dd

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio (probability-weighted gains vs losses)

        Omega = Σ(returns above threshold) / Σ(returns below threshold)

        Args:
            threshold: Return threshold

        Returns:
            Omega ratio
        """
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns < threshold]

        if len(losses) == 0 or np.sum(losses) == 0:
            return np.inf

        return np.sum(gains) / np.sum(losses)

    def maximum_drawdown(
        self,
        prices: np.ndarray
    ) -> Tuple[float, int, int, int]:
        """
        Calculate maximum drawdown and related metrics

        Args:
            prices: Price series (not returns)

        Returns:
            Tuple of (max_drawdown %, start_idx, trough_idx, duration_days)
        """
        prices = np.array(prices)

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown series
        drawdown = (prices - running_max) / running_max

        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = abs(drawdown[max_dd_idx])

        # Find start of drawdown period
        start_idx = np.argmax(prices[:max_dd_idx]) if max_dd_idx > 0 else 0

        # Calculate duration
        duration = max_dd_idx - start_idx

        return max_dd * 100, start_idx, max_dd_idx, duration

    def average_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate average drawdown

        Args:
            prices: Price series

        Returns:
            Average drawdown (%)
        """
        prices = np.array(prices)
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max

        # Only consider drawdown periods (negative values)
        drawdowns = drawdown[drawdown < 0]

        if len(drawdowns) == 0:
            return 0.0

        return abs(np.mean(drawdowns)) * 100

    def tail_ratio(self, percentile: float = 95) -> float:
        """
        Calculate tail ratio (right tail / left tail)

        Measures asymmetry in extreme returns

        Args:
            percentile: Percentile for tail definition

        Returns:
            Tail ratio
        """
        right_tail = np.percentile(self.returns, percentile)
        left_tail = abs(np.percentile(self.returns, 100 - percentile))

        if left_tail == 0:
            return np.inf

        return right_tail / left_tail

    def beta(self) -> Optional[float]:
        """
        Calculate beta (systematic risk relative to benchmark)

        Beta = Cov(Strategy, Benchmark) / Var(Benchmark)

        Returns:
            Beta coefficient (None if no benchmark)
        """
        if self.benchmark_returns is None:
            return None

        # Align lengths
        min_len = min(len(self.returns), len(self.benchmark_returns))
        strategy_returns = self.returns[:min_len]
        bench_returns = self.benchmark_returns[:min_len]

        covariance = np.cov(strategy_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)

        if benchmark_variance == 0:
            return None

        return covariance / benchmark_variance

    def alpha(self) -> Optional[float]:
        """
        Calculate alpha (excess return over benchmark, adjusted for beta)

        Alpha = Strategy Return - (RiskFree + Beta * (Benchmark Return - RiskFree))

        Returns:
            Annualized alpha (None if no benchmark)
        """
        if self.benchmark_returns is None:
            return None

        beta = self.beta()
        if beta is None:
            return None

        # Align lengths
        min_len = min(len(self.returns), len(self.benchmark_returns))
        strategy_returns = self.returns[:min_len]
        bench_returns = self.benchmark_returns[:min_len]

        strategy_return = np.mean(strategy_returns) * self.periods_per_year
        benchmark_return = np.mean(bench_returns) * self.periods_per_year

        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))

        return alpha

    def tracking_error(self) -> Optional[float]:
        """
        Calculate tracking error (volatility of excess returns)

        Returns:
            Annualized tracking error (None if no benchmark)
        """
        if self.benchmark_returns is None:
            return None

        # Align lengths
        min_len = min(len(self.returns), len(self.benchmark_returns))
        strategy_returns = self.returns[:min_len]
        bench_returns = self.benchmark_returns[:min_len]

        excess_returns = strategy_returns - bench_returns
        te = np.std(excess_returns) * np.sqrt(self.periods_per_year)

        return te

    def information_ratio(self) -> Optional[float]:
        """
        Calculate Information Ratio (alpha / tracking error)

        Measures risk-adjusted excess return relative to benchmark

        Returns:
            Information ratio (None if no benchmark)
        """
        if self.benchmark_returns is None:
            return None

        # Align lengths
        min_len = min(len(self.returns), len(self.benchmark_returns))
        strategy_returns = self.returns[:min_len]
        bench_returns = self.benchmark_returns[:min_len]

        excess_returns = strategy_returns - bench_returns

        if np.std(excess_returns) == 0:
            return 0.0

        ir = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.periods_per_year)

        return ir

    def treynor_ratio(self) -> Optional[float]:
        """
        Calculate Treynor Ratio (excess return / beta)

        Returns:
            Annualized Treynor ratio (None if no benchmark)
        """
        if self.benchmark_returns is None:
            return None

        beta = self.beta()
        if beta is None or beta == 0:
            return None

        excess_return = np.mean(self.returns - self.daily_rf) * self.periods_per_year

        return excess_return / beta

    def calculate_all_metrics(
        self,
        prices: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """
        Calculate all available risk metrics

        Args:
            prices: Price series for drawdown calculations (optional)

        Returns:
            RiskMetrics object with all metrics
        """
        logger.info("Calculating comprehensive risk metrics")

        # Calculate VaR
        var_calculator = ValueAtRisk(self.returns)
        var_95_result = var_calculator.historical_var(0.95)
        var_99_result = var_calculator.historical_var(0.99)

        # Calculate drawdown metrics
        if prices is not None:
            max_dd, _, _, max_dd_duration = self.maximum_drawdown(prices)
            avg_dd = self.average_drawdown(prices)
            calmar = self.calmar_ratio(prices)
        else:
            max_dd, max_dd_duration, avg_dd, calmar = 0.0, 0, 0.0, 0.0

        metrics = RiskMetrics(
            # Volatility
            volatility_annual=self.volatility(annualize=True),
            downside_deviation=self.downside_deviation(annualize=True),
            semi_variance=self.semi_variance(),

            # VaR
            var_95=var_95_result.var,
            var_99=var_99_result.var,
            cvar_95=var_95_result.cvar,
            cvar_99=var_99_result.cvar,

            # Drawdown
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,

            # Risk-adjusted returns
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            calmar_ratio=calmar,
            omega_ratio=self.omega_ratio(),

            # Tail risk
            skewness=float(pd.Series(self.returns).skew()),
            kurtosis=float(pd.Series(self.returns).kurtosis()),
            tail_ratio=self.tail_ratio(),

            # Benchmark-relative
            beta=self.beta(),
            alpha=self.alpha(),
            tracking_error=self.tracking_error(),
            information_ratio=self.information_ratio(),
            treynor_ratio=self.treynor_ratio()
        )

        return metrics


def print_risk_metrics(metrics: RiskMetrics):
    """
    Print risk metrics in formatted table

    Args:
        metrics: RiskMetrics object
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RISK ANALYSIS")
    print("=" * 80)

    print("\n📊 Volatility Measures:")
    print(f"  Annual Volatility:       {metrics.volatility_annual*100:.2f}%")
    print(f"  Downside Deviation:      {metrics.downside_deviation*100:.2f}%")
    print(f"  Semi-Variance:           {metrics.semi_variance:.6f}")

    print("\n⚠️  Value at Risk (VaR):")
    print(f"  95% VaR:                 {metrics.var_95*100:.2f}%")
    print(f"  99% VaR:                 {metrics.var_99*100:.2f}%")
    print(f"  95% CVaR (ES):           {metrics.cvar_95*100:.2f}%")
    print(f"  99% CVaR (ES):           {metrics.cvar_99*100:.2f}%")

    print("\n📉 Drawdown Measures:")
    print(f"  Maximum Drawdown:        {metrics.max_drawdown:.2f}%")
    print(f"  Average Drawdown:        {metrics.avg_drawdown:.2f}%")
    print(f"  Max DD Duration:         {metrics.max_drawdown_duration} days")

    print("\n💎 Risk-Adjusted Returns:")
    print(f"  Sharpe Ratio:            {metrics.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio:           {metrics.sortino_ratio:.4f}")
    print(f"  Calmar Ratio:            {metrics.calmar_ratio:.4f}")
    print(f"  Omega Ratio:             {metrics.omega_ratio:.4f}")

    print("\n🎲 Tail Risk:")
    print(f"  Skewness:                {metrics.skewness:.4f}")
    print(f"  Excess Kurtosis:         {metrics.kurtosis:.4f}")
    print(f"  Tail Ratio:              {metrics.tail_ratio:.4f}")

    if metrics.beta is not None:
        print("\n📈 Benchmark-Relative Metrics:")
        print(f"  Beta:                    {metrics.beta:.4f}")
        print(f"  Alpha (annual):          {metrics.alpha*100:.2f}%")
        print(f"  Tracking Error:          {metrics.tracking_error*100:.2f}%")
        print(f"  Information Ratio:       {metrics.information_ratio:.4f}")
        print(f"  Treynor Ratio:           {metrics.treynor_ratio:.4f}")

    print("=" * 80 + "\n")
