"""
Data loading and validation module
Handles fetching stock data with proper error handling and validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StockDataLoader:
    """
    Handles loading and validating stock market data
    """

    def __init__(self, symbol: str, start_date: str, end_date: str):
        """
        Initialize data loader

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance

        Args:
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with stock data
        """
        logger.info(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")

        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=interval
            )

            if data.empty:
                raise ValueError(f"No data fetched for {self.symbol}")

            logger.info(f"Successfully fetched {len(data)} data points")
            self.data = data

            return data

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def validate_data(self, data: Optional[pd.DataFrame] = None) -> Tuple[bool, list]:
        """
        Validate data quality

        Args:
            data: DataFrame to validate (uses self.data if None)

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if data is None:
            data = self.data

        if data is None:
            return False, ["No data loaded"]

        issues = []

        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            issues.append(f"Missing values detected: {missing[missing > 0].to_dict()}")

        # Check for duplicate indices
        if data.index.duplicated().any():
            issues.append("Duplicate dates detected")

        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns and (data[col] <= 0).any():
                issues.append(f"Negative or zero values in {col}")

        # Check for data consistency (High >= Low, etc.)
        if 'High' in data.columns and 'Low' in data.columns:
            if (data['High'] < data['Low']).any():
                issues.append("High price less than Low price detected")

        # Check for outliers (prices changed by more than 50% in one day)
        if 'Close' in data.columns:
            pct_change = data['Close'].pct_change().abs()
            outliers = pct_change > 0.5
            if outliers.any():
                issues.append(f"Extreme price changes detected: {outliers.sum()} occurrences")

        # Check data continuity (no large gaps)
        date_diff = data.index.to_series().diff()
        expected_diff = pd.Timedelta(days=1)
        large_gaps = date_diff > pd.Timedelta(days=7)
        if large_gaps.any():
            issues.append(f"Large time gaps detected: {large_gaps.sum()} gaps")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        return is_valid, issues

    def handle_missing_data(self, method: str = "ffill") -> pd.DataFrame:
        """
        Handle missing data

        Args:
            method: Method to handle missing data (ffill, bfill, interpolate, drop)

        Returns:
            DataFrame with missing data handled
        """
        if self.data is None:
            raise ValueError("No data loaded")

        data = self.data.copy()

        if method == "ffill":
            data = data.fillna(method='ffill')
        elif method == "bfill":
            data = data.fillna(method='bfill')
        elif method == "interpolate":
            data = data.interpolate(method='linear')
        elif method == "drop":
            data = data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Missing data handled using {method}")
        self.data = data

        return data

    def handle_outliers(self, method: str = "clip", threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in the data

        Args:
            method: Method to handle outliers (clip, remove, winsorize)
            threshold: Z-score threshold for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        if self.data is None:
            raise ValueError("No data loaded")

        data = self.data.copy()

        # Calculate returns to detect outliers
        returns = data['Close'].pct_change()

        if method == "clip":
            # Clip extreme returns
            mean = returns.mean()
            std = returns.std()
            lower = mean - threshold * std
            upper = mean + threshold * std
            returns_clipped = returns.clip(lower, upper)

            # Reconstruct prices
            data['Close'] = data['Close'].iloc[0] * (1 + returns_clipped).cumprod()

        elif method == "remove":
            # Remove outlier rows
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            data = data[z_scores < threshold]

        elif method == "winsorize":
            from scipy.stats import mstats
            returns_win = mstats.winsorize(returns.dropna(), limits=[0.05, 0.05])
            data.loc[returns.notna(), 'Returns'] = returns_win

        logger.info(f"Outliers handled using {method}")
        self.data = data

        return data

    def adjust_for_splits_and_dividends(self) -> pd.DataFrame:
        """
        Ensure data is adjusted for stock splits and dividends

        Returns:
            Adjusted DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # yfinance already provides adjusted close
        # We'll use adjusted close if available
        if 'Adj Close' in self.data.columns:
            logger.info("Using adjusted close prices")
            # Replace Close with Adj Close
            self.data['Close'] = self.data['Adj Close']

        return self.data

    def get_clean_data(
        self,
        handle_missing: str = "ffill",
        handle_outliers: bool = True,
        outlier_method: str = "clip"
    ) -> pd.DataFrame:
        """
        Get clean, validated data ready for feature engineering

        Args:
            handle_missing: Method to handle missing data
            handle_outliers: Whether to handle outliers
            outlier_method: Method to handle outliers

        Returns:
            Clean DataFrame
        """
        if self.data is None:
            self.fetch_data()

        # Validate
        is_valid, issues = self.validate_data()

        # Handle missing data
        if not is_valid and any("Missing" in issue for issue in issues):
            self.handle_missing_data(method=handle_missing)

        # Adjust for splits and dividends
        self.adjust_for_splits_and_dividends()

        # Handle outliers
        if handle_outliers:
            self.handle_outliers(method=outlier_method)

        # Final validation
        is_valid, issues = self.validate_data()

        if not is_valid:
            logger.warning("Data still has issues after cleaning:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        logger.info(f"Final clean data shape: {self.data.shape}")

        return self.data

    def get_multiple_stocks(
        self,
        symbols: list,
        column: str = 'Close'
    ) -> pd.DataFrame:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock symbols
            column: Which column to extract

        Returns:
            DataFrame with multiple stock prices
        """
        logger.info(f"Fetching data for {len(symbols)} stocks")

        data_dict = {}

        for symbol in symbols:
            try:
                loader = StockDataLoader(symbol, self.start_date, self.end_date)
                stock_data = loader.get_clean_data()
                data_dict[symbol] = stock_data[column]
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {str(e)}")

        combined = pd.DataFrame(data_dict)
        logger.info(f"Combined data shape: {combined.shape}")

        return combined


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    validate: bool = True,
    clean: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load stock data

    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        validate: Whether to validate data
        clean: Whether to clean data

    Returns:
        DataFrame with stock data
    """
    loader = StockDataLoader(symbol, start_date, end_date)

    if clean:
        data = loader.get_clean_data()
    else:
        data = loader.fetch_data()

    if validate:
        is_valid, issues = loader.validate_data(data)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")

    return data
