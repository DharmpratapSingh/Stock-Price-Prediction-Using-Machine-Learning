"""
Visualization module for stock price prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StockVisualizer:
    """
    Visualization tools for stock prediction analysis
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 8), dpi: int = 100):
        """
        Initialize visualizer

        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_predictions(
        self,
        dates: pd.DatetimeIndex,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Stock Price Prediction",
        save_path: Optional[str] = None
    ):
        """
        Plot actual vs predicted prices

        Args:
            dates: Date index
            actual: Actual prices
            predicted: Predicted prices
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(dates, actual, label='Actual', linewidth=2, alpha=0.8)
        ax.plot(dates, predicted, label='Predicted', linewidth=2, alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_predictions_with_confidence(
        self,
        dates: pd.DatetimeIndex,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        title: str = "Predictions with Confidence Intervals",
        save_path: Optional[str] = None
    ):
        """
        Plot predictions with confidence intervals

        Args:
            dates: Date index
            actual: Actual prices
            predicted: Predicted prices
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(dates, actual, label='Actual', linewidth=2, alpha=0.8, color='blue')
        ax.plot(dates, predicted, label='Predicted', linewidth=2, alpha=0.8, color='red')
        ax.fill_between(dates, lower_bound, upper_bound, alpha=0.2, color='red',
                        label='95% Confidence Interval')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)), dpi=self.dpi)

        top_features = importance_df.head(top_n)

        ax.barh(top_features['feature'], top_features['importance'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_residuals(
        self,
        residuals: np.ndarray,
        dates: pd.DatetimeIndex = None,
        save_path: Optional[str] = None
    ):
        """
        Plot residual analysis

        Args:
            residuals: Model residuals
            dates: Date index (optional)
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # Residuals over time
        if dates is not None:
            axes[0, 0].plot(dates, residuals, alpha=0.6)
        else:
            axes[0, 0].plot(residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Time' if dates is not None else 'Index')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals vs predicted (for heteroscedasticity check)
        axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Scatter Plot')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['R2', 'RMSE', 'MAE', 'Directional_Accuracy'],
        save_path: Optional[str] = None
    ):
        """
        Plot model comparison across metrics

        Args:
            comparison_df: DataFrame with models as rows and metrics as columns
            metrics: List of metrics to plot
            save_path: Path to save figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6), dpi=self.dpi)

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in comparison_df.columns:
                data = comparison_df[metric].sort_values(ascending=False)
                ax.bar(range(len(data)), data.values)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data.index, rotation=45, ha='right')
                ax.set_ylabel(metric)
                ax.set_title(f'Model Comparison: {metric}')
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_equity_curve(
        self,
        dates: pd.DatetimeIndex,
        equity_curve: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve from backtesting

        Args:
            dates: Date index
            equity_curve: Portfolio equity over time
            benchmark: Benchmark equity curve (optional)
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(dates, equity_curve, label='Strategy', linewidth=2)

        if benchmark is not None:
            ax.plot(dates, benchmark, label='Buy & Hold', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_drawdown(
        self,
        dates: pd.DatetimeIndex,
        equity_curve: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot drawdown over time

        Args:
            dates: Date index
            equity_curve: Portfolio equity over time
            save_path: Path to save figure
        """
        # Calculate drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        ax.plot(dates, drawdown, color='red', linewidth=2)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add horizontal line at max drawdown
        max_dd = np.min(drawdown)
        ax.axhline(y=max_dd, color='darkred', linestyle='--', linewidth=2,
                  label=f'Max Drawdown: {max_dd:.2f}%')
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str] = ['sma_50', 'sma_200', 'bb_upper', 'bb_lower'],
        price_col: str = 'Close',
        save_path: Optional[str] = None
    ):
        """
        Plot price with technical indicators

        Args:
            data: DataFrame with price and indicators
            indicators: List of indicator column names
            price_col: Price column name
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot price
        ax.plot(data.index, data[price_col], label='Price', linewidth=2)

        # Plot indicators
        for indicator in indicators:
            if indicator in data.columns:
                ax.plot(data.index, data[indicator], label=indicator, linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Price with Technical Indicators', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        top_n: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Plot correlation matrix of features

        Args:
            data: DataFrame with features
            top_n: Number of top features to include
            save_path: Path to save figure
        """
        # Calculate correlation with target
        if 'target' in data.columns:
            correlations = data.corr()['target'].abs().sort_values(ascending=False)
            top_features = correlations.head(top_n).index.tolist()
            data_subset = data[top_features]
        else:
            # Use first top_n columns
            data_subset = data.iloc[:, :top_n]

        # Calculate correlation matrix
        corr = data_subset.corr()

        fig, ax = plt.subplots(figsize=(max(10, top_n*0.5), max(8, top_n*0.4)), dpi=self.dpi)

        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_returns_distribution(
        self,
        returns: np.ndarray,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of returns

        Args:
            returns: Array of returns
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # Histogram
        axes[0].hist(returns, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=np.mean(returns), color='r', linestyle='--',
                       label=f'Mean: {np.mean(returns):.4f}')
        axes[0].axvline(x=np.median(returns), color='g', linestyle='--',
                       label=f'Median: {np.median(returns):.4f}')
        axes[0].set_xlabel('Returns')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Returns Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(returns, vert=True)
        axes[1].set_ylabel('Returns')
        axes[1].set_title('Returns Box Plot')
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def create_dashboard(
        self,
        dates: pd.DatetimeIndex,
        actual: np.ndarray,
        predicted: np.ndarray,
        equity_curve: np.ndarray,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive dashboard

        Args:
            dates: Date index
            actual: Actual prices
            predicted: Predicted prices
            equity_curve: Equity curve
            metrics: Dictionary of metrics
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Predictions plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, actual, label='Actual', linewidth=2)
        ax1.plot(dates, predicted, label='Predicted', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Actual vs Predicted Prices', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Equity curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(dates[:len(equity_curve)], equity_curve, linewidth=2, color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Equity Curve', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Metrics table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('tight')
        ax3.axis('off')

        metric_data = [[k, f"{v:.4f}" if isinstance(v, float) else v]
                      for k, v in metrics.items()]

        table = ax3.table(cellText=metric_data, colLabels=['Metric', 'Value'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax3.set_title('Performance Metrics', fontweight='bold')

        # Residuals
        ax4 = fig.add_subplot(gs[2, 0])
        residuals = actual - predicted
        ax4.scatter(range(len(residuals)), residuals, alpha=0.5)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals Plot', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Residuals histogram
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax5.axvline(x=0, color='r', linestyle='--')
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Residuals Distribution', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        fig.suptitle('Stock Price Prediction Dashboard', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()
