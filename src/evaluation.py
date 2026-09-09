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


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly for proportions near 0.5 with a few hundred observations --
    exactly the regime a directional-accuracy claim lives in.

    Args:
        successes: Number of correct predictions
        n: Number of predictions
        confidence: Coverage (default 0.95)

    Returns:
        (lower, upper)
    """
    from scipy import stats

    if n == 0:
        return (float('nan'), float('nan'))

    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def return_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Regression metrics for log-return forecasts.

    R2 here is the honest number: the variance of daily returns explained out of
    sample. Values at or slightly below zero mean the forecast is no better than
    predicting the test window's mean, which is what daily equity returns
    normally deliver.

    Args:
        y_true: Realised log returns
        y_pred: Forecast log returns

    Returns:
        Dict with n, RMSE, MAE, R2, and R2 measured against a zero forecast
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    n = len(y_true)
    if n == 0:
        return {'n': 0, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'R2_vs_zero': np.nan}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(y_true ** 2)))

    return {
        'n': n,
        'RMSE': rmse,
        'MAE': float(np.mean(np.abs(y_true - y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        # Skill against the zero forecast rather than against the test-window mean.
        'R2_vs_zero': float(1 - (rmse ** 2) / (zero_rmse ** 2)) if zero_rmse > 0 else np.nan,
    }


def direction_metrics(
    y_true_direction: np.ndarray,
    y_pred_direction: np.ndarray,
    reference_rate: float = None,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Directional accuracy with an interval and significance tests.

    A bare accuracy number is not interpretable: with ~800 test days, the 95%
    interval on a coin flip is roughly +/-3.5 points, so 53% is not distinguishable
    from chance. Two null hypotheses are tested -- 0.5, and the training-window
    up-frequency, which is the rate an always-up baseline achieves for free.

    Args:
        y_true_direction: Realised direction, 1 for up, 0 otherwise
        y_pred_direction: Predicted direction, 1 for up, 0 otherwise
        reference_rate: Training-window up-frequency for the second test
        confidence: Interval coverage

    Returns:
        Dict with accuracy, interval, p-values, precision/recall of 'up'
    """
    from scipy import stats

    y_true = np.asarray(y_true_direction, dtype=int)
    y_pred = np.asarray(y_pred_direction, dtype=int)

    n = len(y_true)
    if n == 0:
        return {'n': 0}

    correct = int(np.sum(y_true == y_pred))
    accuracy = correct / n
    lower, upper = wilson_interval(correct, n, confidence)

    true_up, pred_up = y_true == 1, y_pred == 1
    tp = int(np.sum(true_up & pred_up))
    precision = tp / int(pred_up.sum()) if pred_up.any() else float('nan')
    recall = tp / int(true_up.sum()) if true_up.any() else float('nan')

    metrics = {
        'n': n,
        'accuracy': accuracy,
        'ci_lower': lower,
        'ci_upper': upper,
        'p_vs_0.5': float(stats.binomtest(correct, n, 0.5).pvalue),
        'precision_up': precision,
        'recall_up': recall,
        'pred_up_rate': float(pred_up.mean()),
        'actual_up_rate': float(true_up.mean()),
    }

    if reference_rate is not None and 0 < reference_rate < 1:
        metrics['reference_rate'] = float(reference_rate)
        metrics['p_vs_reference'] = float(
            stats.binomtest(correct, n, reference_rate).pvalue
        )

    return metrics


def run_walk_forward(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    estimator_factory,
    folds: List,
    task: str = 'regression'
) -> pd.DataFrame:
    """
    Run a walk-forward evaluation and collect out-of-sample predictions.

    This is the only path by which the project produces a test-set number. For
    each fold a *fresh* estimator is built by ``estimator_factory`` and fitted on
    that fold's training rows alone; imputation, winsorisation, scaling and
    feature selection all live inside that estimator, so no fitted statistic ever
    sees a test row. Folds run forward in time with an embargo and are never
    shuffled.

    Args:
        dataset: Chronologically ordered features + targets
        feature_cols: Model matrix columns
        target_col: Target column to fit
        estimator_factory: Zero-argument callable returning an unfitted estimator
        folds: Folds from utils.walk_forward_folds
        task: 'regression' or 'classification'

    Returns:
        DataFrame indexed by date with columns: fold, y_true, y_pred, and for
        classification also y_proba; plus train_up_rate per fold
    """
    frames = []

    for fold in folds:
        train = dataset.iloc[fold.train_start:fold.train_end]
        test = dataset.iloc[fold.test_start:fold.test_end]

        X_train, y_train = train[feature_cols], train[target_col]
        X_test = test[feature_cols]

        estimator = estimator_factory()
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        record = pd.DataFrame(
            {
                'fold': fold.index,
                'y_true': test[target_col].to_numpy(),
                'y_pred': np.asarray(y_pred).ravel(),
                'train_up_rate': float((train['target_direction'] > 0).mean())
                if 'target_direction' in train.columns else np.nan,
            },
            index=test.index,
        )

        if task == 'classification' and hasattr(estimator, 'predict_proba'):
            proba = estimator.predict_proba(X_test)
            record['y_proba'] = proba[:, 1] if proba.ndim == 2 else np.asarray(proba).ravel()

        frames.append(record)

    result = pd.concat(frames)
    logger.info(
        "Walk-forward complete: %d folds, %d out-of-sample rows", len(folds), len(result)
    )
    return result
