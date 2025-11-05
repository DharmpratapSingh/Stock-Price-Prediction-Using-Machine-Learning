"""
Comprehensive evaluation metrics for stock price prediction models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates model performance using various metrics
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray = None):
        """
        Initialize evaluator

        Args:
            y_true: True values
            y_pred: Predicted values
            prices: Original prices (for financial metrics)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.prices = np.array(prices) if prices is not None else None

        # Remove NaN values
        mask = ~(np.isnan(self.y_true) | np.isnan(self.y_pred))
        self.y_true = self.y_true[mask]
        self.y_pred = self.y_pred[mask]

        if self.prices is not None:
            self.prices = self.prices[mask]

    def mse(self) -> float:
        """
        Calculate Mean Squared Error

        Returns:
            MSE value
        """
        return mean_squared_error(self.y_true, self.y_pred)

    def rmse(self) -> float:
        """
        Calculate Root Mean Squared Error

        Returns:
            RMSE value
        """
        return np.sqrt(self.mse())

    def mae(self) -> float:
        """
        Calculate Mean Absolute Error

        Returns:
            MAE value
        """
        return mean_absolute_error(self.y_true, self.y_pred)

    def mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error

        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = self.y_true != 0
        if not mask.any():
            return np.inf

        mape = np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
        return mape

    def r2(self) -> float:
        """
        Calculate R-squared Score

        Returns:
            R² value
        """
        return r2_score(self.y_true, self.y_pred)

    def directional_accuracy(self) -> float:
        """
        Calculate directional accuracy (% of correct direction predictions)

        Returns:
            Directional accuracy (0-1)
        """
        if len(self.y_true) < 2:
            return 0.0

        # Calculate actual and predicted directions
        actual_direction = np.sign(np.diff(self.y_true))
        predicted_direction = np.sign(np.diff(self.y_pred))

        # Calculate accuracy
        correct = np.sum(actual_direction == predicted_direction)
        total = len(actual_direction)

        return correct / total if total > 0 else 0.0

    def max_error(self) -> float:
        """
        Calculate maximum absolute error

        Returns:
            Max error
        """
        return np.max(np.abs(self.y_true - self.y_pred))

    def explained_variance(self) -> float:
        """
        Calculate explained variance score

        Returns:
            Explained variance
        """
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(self.y_true, self.y_pred)

    def mean_directional_error(self) -> float:
        """
        Calculate mean directional error (bias)

        Returns:
            Mean directional error
        """
        return np.mean(self.y_pred - self.y_true)

    def theil_u_statistic(self) -> float:
        """
        Calculate Theil's U statistic (forecast accuracy measure)

        Returns:
            Theil U value
        """
        numerator = np.sqrt(np.mean((self.y_pred - self.y_true) ** 2))
        denominator = np.sqrt(np.mean(self.y_true ** 2)) + np.sqrt(np.mean(self.y_pred ** 2))

        return numerator / denominator if denominator != 0 else np.inf

    def calculate_returns_based_metrics(self) -> Dict[str, float]:
        """
        Calculate returns-based metrics (requires prices)

        Returns:
            Dictionary of financial metrics
        """
        if self.prices is None:
            logger.warning("Prices not provided, cannot calculate returns-based metrics")
            return {}

        # Calculate returns
        actual_returns = np.diff(self.y_true) / self.y_true[:-1]
        predicted_returns = np.diff(self.y_pred) / self.y_pred[:-1]

        metrics = {}

        # Sharpe Ratio (annualized, assuming daily data)
        if len(actual_returns) > 0:
            sharpe_actual = self._calculate_sharpe_ratio(actual_returns)
            sharpe_predicted = self._calculate_sharpe_ratio(predicted_returns)

            metrics['sharpe_ratio_actual'] = sharpe_actual
            metrics['sharpe_ratio_predicted'] = sharpe_predicted

        # Maximum Drawdown
        metrics['max_drawdown_actual'] = self._calculate_max_drawdown(self.y_true)
        metrics['max_drawdown_predicted'] = self._calculate_max_drawdown(self.y_pred)

        # Volatility (annualized)
        metrics['volatility_actual'] = np.std(actual_returns) * np.sqrt(252)
        metrics['volatility_predicted'] = np.std(predicted_returns) * np.sqrt(252)

        return metrics

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        # Annualize
        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown

        Args:
            prices: Array of prices

        Returns:
            Maximum drawdown (as percentage)
        """
        if len(prices) == 0:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Return maximum drawdown (as positive percentage)
        return abs(np.min(drawdown)) * 100

    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all available metrics

        Returns:
            Dictionary of all metrics
        """
        logger.info("Calculating all evaluation metrics")

        metrics = {
            'MSE': self.mse(),
            'RMSE': self.rmse(),
            'MAE': self.mae(),
            'MAPE': self.mape(),
            'R2': self.r2(),
            'Directional_Accuracy': self.directional_accuracy(),
            'Max_Error': self.max_error(),
            'Explained_Variance': self.explained_variance(),
            'Mean_Directional_Error': self.mean_directional_error(),
            'Theil_U': self.theil_u_statistic()
        }

        # Add financial metrics if prices available
        financial_metrics = self.calculate_returns_based_metrics()
        metrics.update(financial_metrics)

        return metrics

    def print_metrics(self):
        """
        Print all metrics in a formatted way
        """
        metrics = self.calculate_all_metrics()

        print("\n" + "=" * 60)
        print("MODEL EVALUATION METRICS")
        print("=" * 60)

        # Statistical metrics
        print("\n📊 Statistical Metrics:")
        print(f"  MSE:                    {metrics['MSE']:.4f}")
        print(f"  RMSE:                   {metrics['RMSE']:.4f}")
        print(f"  MAE:                    {metrics['MAE']:.4f}")
        print(f"  MAPE:                   {metrics['MAPE']:.2f}%")
        print(f"  R²:                     {metrics['R2']:.4f}")
        print(f"  Explained Variance:     {metrics['Explained_Variance']:.4f}")

        # Prediction quality
        print("\n🎯 Prediction Quality:")
        print(f"  Directional Accuracy:   {metrics['Directional_Accuracy']*100:.2f}%")
        print(f"  Max Error:              {metrics['Max_Error']:.4f}")
        print(f"  Mean Directional Error: {metrics['Mean_Directional_Error']:.4f}")
        print(f"  Theil U Statistic:      {metrics['Theil_U']:.4f}")

        # Financial metrics
        if 'sharpe_ratio_actual' in metrics:
            print("\n💰 Financial Metrics:")
            print(f"  Sharpe Ratio (Actual):     {metrics['sharpe_ratio_actual']:.4f}")
            print(f"  Sharpe Ratio (Predicted):  {metrics['sharpe_ratio_predicted']:.4f}")
            print(f"  Max Drawdown (Actual):     {metrics['max_drawdown_actual']:.2f}%")
            print(f"  Max Drawdown (Predicted):  {metrics['max_drawdown_predicted']:.2f}%")
            print(f"  Volatility (Actual):       {metrics['volatility_actual']:.2f}%")
            print(f"  Volatility (Predicted):    {metrics['volatility_predicted']:.2f}%")

        print("=" * 60 + "\n")


def compare_models(
    models_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    prices: np.ndarray = None
) -> pd.DataFrame:
    """
    Compare multiple models

    Args:
        models_results: Dictionary of {model_name: (y_true, y_pred)}
        prices: Original prices (optional)

    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing {len(models_results)} models")

    results = {}

    for model_name, (y_true, y_pred) in models_results.items():
        evaluator = ModelEvaluator(y_true, y_pred, prices)
        metrics = evaluator.calculate_all_metrics()
        results[model_name] = metrics

    # Create DataFrame
    df = pd.DataFrame(results).T

    # Sort by R² score (descending)
    df = df.sort_values('R2', ascending=False)

    return df


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """
    Perform residual analysis

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true - y_pred

    analysis = {
        'residuals': residuals,
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'q25': np.percentile(residuals, 25),
        'q50': np.percentile(residuals, 50),
        'q75': np.percentile(residuals, 75),
        'skewness': pd.Series(residuals).skew(),
        'kurtosis': pd.Series(residuals).kurtosis()
    }

    # Test for normality (Shapiro-Wilk test)
    from scipy import stats
    if len(residuals) < 5000:  # Shapiro-Wilk limited to 5000 samples
        _, p_value = stats.shapiro(residuals)
        analysis['normality_p_value'] = p_value
        analysis['is_normal'] = p_value > 0.05

    # Autocorrelation of residuals
    if len(residuals) > 1:
        analysis['autocorrelation_lag1'] = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

    return analysis


def calculate_confidence_intervals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction confidence intervals

    Args:
        y_pred: Predicted values
        residuals: Model residuals
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats

    # Calculate standard error
    std_error = np.std(residuals)

    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)

    # Calculate intervals
    margin = z_score * std_error
    lower_bound = y_pred - margin
    upper_bound = y_pred + margin

    return lower_bound, upper_bound


def walk_forward_validation(
    data: pd.DataFrame,
    model_class: type,
    feature_cols: List[str],
    target_col: str,
    train_size: int = 252,  # 1 year
    test_size: int = 21,    # 1 month
    step_size: int = 21     # 1 month
) -> Dict[str, any]:
    """
    Perform walk-forward validation

    Args:
        data: DataFrame with features and target
        model_class: Model class to use
        feature_cols: List of feature column names
        target_col: Target column name
        train_size: Training window size
        test_size: Test window size
        step_size: Step size for rolling window

    Returns:
        Dictionary with validation results
    """
    logger.info("Performing walk-forward validation")

    predictions = []
    actuals = []
    dates = []

    n = len(data)
    start_idx = train_size

    while start_idx + test_size <= n:
        # Define train and test windows
        train_start = start_idx - train_size
        train_end = start_idx
        test_end = min(start_idx + test_size, n)

        # Split data
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[start_idx:test_end]

        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Train model
        model = model_class()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Store results
        predictions.extend(y_pred)
        actuals.extend(y_test.values)
        dates.extend(test_data.index)

        # Move window
        start_idx += step_size

    # Calculate metrics
    evaluator = ModelEvaluator(np.array(actuals), np.array(predictions))
    metrics = evaluator.calculate_all_metrics()

    results = {
        'predictions': np.array(predictions),
        'actuals': np.array(actuals),
        'dates': dates,
        'metrics': metrics,
        'n_windows': len(predictions) // test_size
    }

    logger.info(f"Walk-forward validation completed with {results['n_windows']} windows")

    return results
