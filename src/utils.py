"""
Utility functions for the stock prediction project
"""

import os
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for the project

    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)


def save_model(model: Any, model_name: str, models_dir: str = "models") -> str:
    """
    Save trained model to disk

    Args:
        model: Trained model object
        model_name: Name for the saved model
        models_dir: Directory to save models

    Returns:
        Path to saved model
    """
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    return filepath


def load_model(model_path: str) -> Any:
    """
    Load trained model from disk

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model object
    """
    return joblib.load(model_path)


def time_series_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data chronologically

    Args:
        data: DataFrame to split
        test_size: Fraction for test set
        validation_size: Fraction for validation set

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - validation_size))

    train_data = data.iloc[:val_idx]
    val_data = data.iloc[val_idx:test_idx]
    test_data = data.iloc[test_idx:]

    return train_data, val_data, test_data


def get_date_range(start_date: str = None, end_date: str = None) -> Tuple[str, str]:
    """
    Get date range for data fetching

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Tuple of (start_date, end_date)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        # Default to 5 years ago
        from dateutil.relativedelta import relativedelta
        start_date = (datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d")

    return start_date, end_date


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate percentage returns

    Args:
        prices: Price series

    Returns:
        Returns series
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns

    Args:
        prices: Price series

    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(1))


def normalize_data(data: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
    """
    Normalize data using specified method

    Args:
        data: Data to normalize
        method: Normalization method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of (normalized_data, scaler)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    scaler = scalers.get(method, StandardScaler())
    normalized = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

    return normalized, scaler


def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Print metrics in a formatted table

    Args:
        metrics_dict: Dictionary of model names and their metrics
    """
    df = pd.DataFrame(metrics_dict).T
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")


def save_results(results: Dict[str, Any], filename: str, results_dir: str = "results") -> str:
    """
    Save results to disk

    Args:
        results: Results dictionary
        filename: Name for results file
        results_dir: Directory to save results

    Returns:
        Path to saved results
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"{filename}_{timestamp}.joblib")
    joblib.dump(results, filepath)
    return filepath
