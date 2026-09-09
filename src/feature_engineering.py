"""
Feature Engineering Module
Implements comprehensive technical indicators and features for stock prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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
        # Fallback period for the simplified ADX when no ATR column exists yet.
        self.adx_period = 14

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

    def create_volume_features(self, window: int = 20, include_obv: bool = True) -> pd.DataFrame:
        """
        Create volume-based features

        Args:
            window: Rolling window for volume calculations
            include_obv: Whether to emit the cumulative OBV / VPT series

        Returns:
            DataFrame with volume features
        """
        logger.info(f"Creating volume features with window {window} (obv={include_obv})")

        if 'Volume' not in self.data.columns:
            logger.warning("Volume data not available")
            return self.features

        volume = self.data['Volume']
        close = self.data['Close']

        # Volume moving average
        self.features['volume_sma'] = volume.rolling(window=window).mean()

        # Volume ratio
        self.features['volume_ratio'] = volume / self.features['volume_sma']

        if include_obv:
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

    def create_trend_features(self, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Create trend identification features

        Args:
            windows: Windows for the rolling linear-regression slope

        Returns:
            DataFrame with trend features
        """
        logger.info(f"Creating trend features for windows {windows}")

        close = self.data['Close']

        # Linear regression slope over different windows
        for window in windows:
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

            # Reuse the configured ATR if it was already built, else fall back to
            # a rolling standard deviation over the same period.
            atr_col = next(
                (c for c in self.features.columns
                 if c.startswith('atr_') and not c.endswith('_pct')),
                None
            )
            if atr_col is not None:
                atr = self.features[atr_col]
                atr_window = int(atr_col.split('_')[1])
            else:
                atr_window = self.adx_period
                atr = close.rolling(atr_window).std()

            plus_di = 100 * (plus_dm.rolling(atr_window).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(atr_window).mean() / atr)

            self.features['plus_di'] = plus_di
            self.features['minus_di'] = minus_di
            self.features['adx'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        return self.features

    def create_all_features(
        self,
        config: Optional[Dict] = None,
        dropna: bool = True
    ) -> pd.DataFrame:
        """
        Create all features based on configuration.

        Every configurable period in ``config['features']`` is threaded through to
        the method that uses it -- nothing is silently hard-coded. Every feature
        at time t is a function of data at times <= t only.

        Args:
            config: The ``features`` block of the project config
            dropna: Drop the leading warm-up rows that rolling windows leave NaN.
                Pass False when you need the un-trimmed frame (e.g. to measure
                the warm-up length for the walk-forward embargo).

        Returns:
            DataFrame with the original OHLCV columns plus all engineered features
        """
        logger.info("Creating all features")

        if config is None:
            config = {}

        # Extract configuration -- keys match config/config.yaml exactly.
        lag_periods = config.get('lag_periods', [1, 2, 3, 5, 10, 20])
        sma_windows = config.get('sma_windows', [10, 20, 50, 100, 200])
        ema_windows = config.get('ema_windows', [12, 26, 50])
        rsi_period = config.get('rsi_period', 14)
        macd_config = {
            'fast': config.get('macd_fast', 12),
            'slow': config.get('macd_slow', 26),
            'signal': config.get('macd_signal', 9)
        }
        bb_config = {
            'window': config.get('bollinger_window', 20),
            'num_std': config.get('bollinger_std', 2)
        }
        atr_period = config.get('atr_period', 14)
        stoch_config = {
            'k_period': config.get('stoch_k', 14),
            'd_period': config.get('stoch_d', 3)
        }
        volume_window = config.get('volume_sma', 20)
        include_obv = config.get('obv', True)
        volatility_windows = config.get('rolling_std_windows', [10, 20, 30])
        change_periods = config.get('price_changes', [1, 5, 10, 20])
        momentum_periods = config.get('momentum_periods', [5, 10, 20])
        trend_windows = config.get('trend_windows', [10, 20, 50])

        self.adx_period = atr_period

        # Create all feature types
        self.create_lag_features(columns=['Close', 'Open', 'High', 'Low'], lags=lag_periods)
        self.create_returns(periods=change_periods)
        self.create_moving_averages(sma_windows=sma_windows, ema_windows=ema_windows)
        self.create_rsi(period=rsi_period)
        self.create_macd(**macd_config)
        self.create_bollinger_bands(**bb_config)
        self.create_atr(period=atr_period)
        self.create_stochastic_oscillator(**stoch_config)
        self.create_volume_features(window=volume_window, include_obv=include_obv)
        self.create_volatility_features(windows=volatility_windows)
        self.create_price_patterns()
        self.create_momentum_features(periods=momentum_periods)
        self.create_trend_features(windows=trend_windows)

        # Combine original data with features
        result = pd.concat([self.data, self.features], axis=1)

        if dropna:
            # Drop the leading warm-up rows left NaN by rolling calculations
            result = result.dropna()

        logger.info(f"Created {len(self.features.columns)} features")
        logger.info(f"Final dataset shape: {result.shape}")

        return result


# Raw OHLCV passthrough columns. They are kept on the dataset (the backtest needs
# the spot close) but are never handed to a model as features.
PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# Target columns produced by create_target_variable.
TARGET_COLUMNS = ['target', 'target_logret', 'target_direction', 'target_price']

# Feature-name prefixes that carry a raw price/volume *level*. A level feature is
# non-stationary: a model fitted on 2018 price levels sees inputs in 2024 that lie
# entirely outside its training range, so the fit does not transfer. The scale-free
# transforms of the same indicators (dist_from_sma_*, bb_percent, rsi, ...) are kept.
_LEVEL_PREFIXES = (
    'Close_lag_', 'Open_lag_', 'High_lag_', 'Low_lag_',
    'sma_', 'ema_', 'bb_upper', 'bb_middle', 'bb_lower',
    'obv', 'vpt', 'volume_sma', 'macd_line', 'macd_signal', 'macd_histogram',
    'momentum_', 'trend_slope_', 'atr_',
)


def _is_level_feature(name: str) -> bool:
    """True when a feature is denominated in price or share units."""
    if name.startswith('atr_') and name.endswith('_pct'):
        return False  # ATR normalised by price is scale-free
    return name.startswith(_LEVEL_PREFIXES)


def feature_columns(data: pd.DataFrame, stationary_only: bool = True) -> List[str]:
    """
    Choose the model matrix columns from an engineered dataset.

    Args:
        data: Output of ``FeatureEngineer.create_all_features``, optionally with
            target columns attached
        stationary_only: Drop level-valued features (see ``_LEVEL_PREFIXES``).
            This is a modelling decision, not a leakage fix: levels are legal
            inputs, they simply do not generalise across a 10-year price range.

    Returns:
        Ordered list of feature column names
    """
    excluded = set(PRICE_COLUMNS) | set(TARGET_COLUMNS)
    cols = [c for c in data.columns if c not in excluded]
    if stationary_only:
        cols = [c for c in cols if not _is_level_feature(c)]
    return cols


def feature_warmup_length(raw: pd.DataFrame, config: Optional[Dict] = None) -> int:
    """
    Measure how many past rows a single feature row depends on.

    This is the number of leading rows that rolling windows leave NaN. It equals
    the deepest lookback across the whole feature set, so it is exactly the
    embargo needed between a training window and the next test window: with a gap
    of this size, no test-fold feature row touches any row used for training.

    Args:
        raw: Raw OHLCV data
        config: The ``features`` config block

    Returns:
        Warm-up length in rows
    """
    engineered = FeatureEngineer(raw).create_all_features(config, dropna=False)
    valid = engineered.notna().all(axis=1)
    if not valid.any():
        raise ValueError("Feature matrix has no fully-valid row")
    return int(np.argmax(valid.to_numpy()))


def create_target_variable(
    data: pd.DataFrame,
    target_type: str = 'log_return',
    horizon: int = 1
) -> pd.DataFrame:
    """
    Attach forward-looking target columns.

    The primary target is the next-day log return, ``log(C(t+h) / C(t))``, plus
    its sign for the direction task. The price level ``C(t+h)`` is also attached
    but only so the level-R2 trap demonstration can be reproduced -- it is not a
    modelling target (see docs/PIPELINE.md).

    Args:
        data: DataFrame with a ``Close`` column
        target_type: 'log_return' (default), 'direction', or 'price'
        horizon: Prediction horizon in trading days

    Returns:
        DataFrame with ``target`` plus the named target columns, rows with an
        undefined target dropped
    """
    logger.info(f"Creating target variable: {target_type}, horizon: {horizon}")

    result = data.copy()
    close = result['Close']

    future_log_return = np.log(close.shift(-horizon) / close)

    result['target_logret'] = future_log_return
    result['target_direction'] = (future_log_return > 0).astype('int64')
    result['target_price'] = close.shift(-horizon)

    if target_type == 'log_return':
        result['target'] = result['target_logret']
    elif target_type == 'direction':
        result['target'] = result['target_direction']
    elif target_type == 'price':
        result['target'] = result['target_price']
    else:
        raise ValueError(
            f"Unknown target type: {target_type!r}. "
            f"Use 'log_return', 'direction' or 'price'."
        )

    result = result.dropna(subset=['target_logret', 'target'])

    logger.info(f"Target variable created. Shape: {result.shape}")

    return result


def build_dataset(
    raw: pd.DataFrame,
    config: Optional[Dict] = None,
    horizon: int = 1,
    stationary_only: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the modelling dataset for one ticker in one call.

    train.py and predict.py both go through this function, so inference builds
    exactly the columns training used.

    Args:
        raw: Raw OHLCV data
        config: The ``features`` config block
        horizon: Prediction horizon in trading days
        stationary_only: Restrict the model matrix to scale-free features

    Returns:
        (dataset with features + targets + OHLCV, ordered feature column names)
    """
    engineered = FeatureEngineer(raw).create_all_features(config, dropna=True)
    dataset = create_target_variable(engineered, target_type='log_return', horizon=horizon)
    cols = feature_columns(dataset, stationary_only=stationary_only)
    return dataset, cols
