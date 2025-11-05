"""
Feature Engineering Module
Implements comprehensive technical indicators and features for stock prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create technical indicators and features for stock prediction
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer

        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)

    def create_lag_features(self, columns: List[str] = ['Close'], lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features (CRITICAL: prevents data leakage)

        Args:
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features for {columns} with lags {lags}")

        for col in columns:
            if col in self.data.columns:
                for lag in lags:
                    self.features[f'{col}_lag_{lag}'] = self.data[col].shift(lag)

        return self.features

    def create_returns(self, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create return features

        Args:
            periods: Periods for return calculation

        Returns:
            DataFrame with return features
        """
        logger.info(f"Creating return features for periods {periods}")

        close = self.data['Close']

        for period in periods:
            # Percentage returns
            self.features[f'return_{period}d'] = close.pct_change(period)

            # Log returns
            self.features[f'log_return_{period}d'] = np.log(close / close.shift(period))

        return self.features

    def create_moving_averages(
        self,
        sma_windows: List[int] = [10, 20, 50, 100, 200],
        ema_windows: List[int] = [12, 26, 50]
    ) -> pd.DataFrame:
        """
        Create moving average features

        Args:
            sma_windows: Simple moving average windows
            ema_windows: Exponential moving average windows

        Returns:
            DataFrame with MA features
        """
        logger.info(f"Creating moving averages: SMA {sma_windows}, EMA {ema_windows}")

        close = self.data['Close']

        # Simple Moving Averages
        for window in sma_windows:
            self.features[f'sma_{window}'] = close.rolling(window=window).mean()
            # Distance from MA (normalized)
            self.features[f'dist_from_sma_{window}'] = (close - self.features[f'sma_{window}']) / self.features[f'sma_{window}']

        # Exponential Moving Averages
        for window in ema_windows:
            self.features[f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()
            self.features[f'dist_from_ema_{window}'] = (close - self.features[f'ema_{window}']) / self.features[f'ema_{window}']

        return self.features

    def create_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Create Relative Strength Index

        Args:
            period: RSI period

        Returns:
            DataFrame with RSI feature
        """
        logger.info(f"Creating RSI with period {period}")

        close = self.data['Close']
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        self.features[f'rsi_{period}'] = rsi

        # RSI-based signals
        self.features[f'rsi_oversold'] = (rsi < 30).astype(int)
        self.features[f'rsi_overbought'] = (rsi > 70).astype(int)

        return self.features

    def create_macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Create MACD (Moving Average Convergence Divergence)

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD features
        """
        logger.info(f"Creating MACD ({fast}, {slow}, {signal})")

        close = self.data['Close']

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        self.features['macd_line'] = macd_line
        self.features['macd_signal'] = signal_line
        self.features['macd_histogram'] = macd_histogram

        # MACD cross signals
        self.features['macd_bullish_cross'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
        self.features['macd_bearish_cross'] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)

        return self.features

    def create_bollinger_bands(
        self,
        window: int = 20,
        num_std: float = 2
    ) -> pd.DataFrame:
        """
        Create Bollinger Bands

        Args:
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            DataFrame with Bollinger Band features
        """
        logger.info(f"Creating Bollinger Bands ({window}, {num_std})")

        close = self.data['Close']

        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        self.features['bb_upper'] = upper_band
        self.features['bb_middle'] = sma
        self.features['bb_lower'] = lower_band

        # Bollinger Band Width
        self.features['bb_width'] = (upper_band - lower_band) / sma

        # %B (position within bands)
        self.features['bb_percent'] = (close - lower_band) / (upper_band - lower_band)

        # Signals
        self.features['bb_squeeze'] = (self.features['bb_width'] < self.features['bb_width'].rolling(window=50).mean() * 0.8).astype(int)

        return self.features

    def create_atr(self, period: int = 14) -> pd.DataFrame:
        """
        Create Average True Range (volatility measure)

        Args:
            period: ATR period

        Returns:
            DataFrame with ATR feature
        """
        logger.info(f"Creating ATR with period {period}")

        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        self.features[f'atr_{period}'] = atr

        # Normalized ATR
        self.features[f'atr_{period}_pct'] = atr / close

        return self.features

    def create_stochastic_oscillator(
        self,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Create Stochastic Oscillator

        Args:
            k_period: %K period
            d_period: %D period

        Returns:
            DataFrame with Stochastic features
        """
        logger.info(f"Creating Stochastic Oscillator ({k_period}, {d_period})")

        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        self.features['stoch_k'] = stoch_k
        self.features['stoch_d'] = stoch_d

        # Stochastic signals
        self.features['stoch_oversold'] = (stoch_k < 20).astype(int)
        self.features['stoch_overbought'] = (stoch_k > 80).astype(int)

        return self.features

    def create_volume_features(self, window: int = 20) -> pd.DataFrame:
        """
        Create volume-based features

        Args:
            window: Rolling window for volume calculations

        Returns:
            DataFrame with volume features
        """
        logger.info(f"Creating volume features with window {window}")

        if 'Volume' not in self.data.columns:
            logger.warning("Volume data not available")
            return self.features

        volume = self.data['Volume']
        close = self.data['Close']

        # Volume moving average
        self.features['volume_sma'] = volume.rolling(window=window).mean()

        # Volume ratio
        self.features['volume_ratio'] = volume / self.features['volume_sma']

        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        self.features['obv'] = obv

        # Volume-Price Trend
        vpt = volume * (close.pct_change()).fillna(0)
        self.features['vpt'] = vpt.cumsum()

        # Volume Rate of Change
        self.features['volume_roc'] = volume.pct_change(window)

        return self.features

    def create_volatility_features(
        self,
        windows: List[int] = [10, 20, 30]
    ) -> pd.DataFrame:
        """
        Create volatility features

        Args:
            windows: Rolling windows for volatility calculation

        Returns:
            DataFrame with volatility features
        """
        logger.info(f"Creating volatility features for windows {windows}")

        close = self.data['Close']
        returns = close.pct_change()

        for window in windows:
            # Historical volatility (standard deviation of returns)
            self.features[f'volatility_{window}d'] = returns.rolling(window=window).std()

            # Parkinson volatility (uses high-low range)
            if 'High' in self.data.columns and 'Low' in self.data.columns:
                high = self.data['High']
                low = self.data['Low']
                parkinson = np.sqrt(1 / (4 * np.log(2)) * ((np.log(high / low)) ** 2))
                self.features[f'parkinson_vol_{window}d'] = parkinson.rolling(window=window).mean()

        return self.features

    def create_price_patterns(self) -> pd.DataFrame:
        """
        Create price pattern features

        Returns:
            DataFrame with price pattern features
        """
        logger.info("Creating price pattern features")

        open_price = self.data['Open']
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        # Daily range
        self.features['daily_range'] = (high - low) / close

        # Body (close - open)
        self.features['body'] = (close - open_price) / close

        # Upper shadow
        self.features['upper_shadow'] = (high - np.maximum(open_price, close)) / close

        # Lower shadow
        self.features['lower_shadow'] = (np.minimum(open_price, close) - low) / close

        # Gap (today's open vs yesterday's close)
        self.features['gap'] = (open_price - close.shift(1)) / close.shift(1)

        return self.features

    def create_momentum_features(
        self,
        periods: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create momentum indicators

        Args:
            periods: Periods for momentum calculation

        Returns:
            DataFrame with momentum features
        """
        logger.info(f"Creating momentum features for periods {periods}")

        close = self.data['Close']

        for period in periods:
            # Rate of Change
            self.features[f'roc_{period}'] = close.pct_change(period)

            # Momentum (absolute price change)
            self.features[f'momentum_{period}'] = close - close.shift(period)

        return self.features

    def create_trend_features(self) -> pd.DataFrame:
        """
        Create trend identification features

        Returns:
            DataFrame with trend features
        """
        logger.info("Creating trend features")

        close = self.data['Close']

        # Linear regression slope over different windows
        for window in [10, 20, 50]:
            slopes = []
            for i in range(len(close)):
                if i < window:
                    slopes.append(np.nan)
                else:
                    y = close.iloc[i-window:i].values
                    x = np.arange(window)
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)

            self.features[f'trend_slope_{window}'] = slopes

        # ADX (Average Directional Index) - simplified version
        if all(col in self.data.columns for col in ['High', 'Low', 'Close']):
            high = self.data['High']
            low = self.data['Low']

            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            atr = self.features.get('atr_14', close.rolling(14).std())

            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

            self.features['plus_di'] = plus_di
            self.features['minus_di'] = minus_di
            self.features['adx'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        return self.features

    def create_all_features(self, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all features based on configuration

        Args:
            config: Configuration dictionary

        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features")

        if config is None:
            config = {}

        # Extract configuration
        lag_periods = config.get('lag_periods', [1, 2, 3, 5, 10, 20])
        sma_windows = config.get('sma_windows', [10, 20, 50, 100, 200])
        ema_windows = config.get('ema_windows', [12, 26, 50])
        rsi_period = config.get('rsi_period', 14)
        macd_config = config.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        bb_config = config.get('bollinger', {'window': 20, 'std': 2})
        atr_period = config.get('atr_period', 14)
        stoch_config = config.get('stochastic', {'k': 14, 'd': 3})

        # Create all feature types
        self.create_lag_features(columns=['Close', 'Open', 'High', 'Low'], lags=lag_periods)
        self.create_returns(periods=[1, 5, 10, 20])
        self.create_moving_averages(sma_windows=sma_windows, ema_windows=ema_windows)
        self.create_rsi(period=rsi_period)
        self.create_macd(**macd_config)
        self.create_bollinger_bands(**bb_config)
        self.create_atr(period=atr_period)
        self.create_stochastic_oscillator(**stoch_config)
        self.create_volume_features()
        self.create_volatility_features()
        self.create_price_patterns()
        self.create_momentum_features()
        self.create_trend_features()

        # Combine original data with features
        result = pd.concat([self.data, self.features], axis=1)

        # Drop rows with NaN (from rolling calculations)
        result = result.dropna()

        logger.info(f"Created {len(self.features.columns)} features")
        logger.info(f"Final dataset shape: {result.shape}")

        return result


def create_target_variable(
    data: pd.DataFrame,
    target_type: str = 'price',
    horizon: int = 1
) -> pd.DataFrame:
    """
    Create target variable for prediction

    Args:
        data: DataFrame with features
        target_type: Type of target ('price', 'return', 'direction')
        horizon: Prediction horizon (days ahead)

    Returns:
        DataFrame with target variable
    """
    logger.info(f"Creating target variable: {target_type}, horizon: {horizon}")

    result = data.copy()

    if target_type == 'price':
        # Predict future price
        result['target'] = result['Close'].shift(-horizon)

    elif target_type == 'return':
        # Predict future return
        result['target'] = result['Close'].pct_change(horizon).shift(-horizon)

    elif target_type == 'direction':
        # Predict direction (binary classification)
        future_return = result['Close'].pct_change(horizon).shift(-horizon)
        result['target'] = (future_return > 0).astype(int)

    else:
        raise ValueError(f"Unknown target type: {target_type}")

    # Drop rows with NaN target
    result = result.dropna(subset=['target'])

    logger.info(f"Target variable created. Shape: {result.shape}")

    return result
