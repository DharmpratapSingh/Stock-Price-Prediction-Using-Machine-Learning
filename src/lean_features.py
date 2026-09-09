"""
A small, strictly backward-looking feature set for next-day return prediction.

Every value on row ``t`` is computed from data at or before ``t``: no
``center=True`` windows, no negative shifts. The only forward-looking quantity
in this module is the target itself, which is what we are trying to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_lag0",
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag4",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "sma_ratio_10",
    "sma_ratio_20",
    "sma_ratio_50",
    "sma_10_50",
    "rsi_14",
    "atr_14_pct",
    "vol_21",
    "volume_ratio_20",
]

TARGET_COLUMN = "target"
RSI_PERIOD = 14
ATR_PERIOD = 14


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the lean feature matrix from an OHLCV frame.

    Args:
        df: Frame with ``Open, High, Low, Close, Volume`` columns.

    Returns:
        DataFrame on the same index whose columns are exactly
        :data:`FEATURE_COLUMNS`. Warmup rows are NaN and are dropped later by
        :func:`build_dataset`.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    returns = close.pct_change()

    features = pd.DataFrame(index=df.index)

    # Recent daily returns (momentum at the shortest horizon).
    features["ret_lag0"] = returns
    for lag in range(1, 5):
        features[f"ret_lag{lag}"] = returns.shift(lag)

    # Multi-day momentum.
    for window in (5, 21, 63):
        features[f"ret_{window}d"] = close / close.shift(window) - 1

    # Distance from moving averages, and the fast/slow MA relationship.
    sma = {n: close.rolling(n).mean() for n in (10, 20, 50)}
    for n in (10, 20, 50):
        features[f"sma_ratio_{n}"] = close / sma[n] - 1
    features["sma_10_50"] = sma[10] / sma[50] - 1

    features["rsi_14"] = _wilder_rsi(close, RSI_PERIOD)
    features["atr_14_pct"] = _atr(high, low, close, ATR_PERIOD) / close
    features["vol_21"] = returns.rolling(21).std()
    features["volume_ratio_20"] = volume / volume.rolling(20).mean()

    return features.loc[:, FEATURE_COLUMNS]


def build_target(df: pd.DataFrame) -> pd.Series:
    """Next-day simple return: ``Close[t+1] / Close[t] - 1``."""
    close = df["Close"].astype(float)
    target = close.shift(-1) / close - 1
    target.name = TARGET_COLUMN
    return target


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Features joined to the target with warmup and trailing rows dropped.

    Returns:
        DataFrame with columns ``FEATURE_COLUMNS + ["target"]`` and no NaNs.
        The last row of ``df`` is always dropped because its target is unknown.
    """
    dataset = build_features(df).join(build_target(df))
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    return dataset.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]]


def _wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI using an exponential average with ``alpha = 1 / period``.

    Edge cases are resolved without producing ``inf``: a window with no losses
    is RSI 100, a window with no gains is RSI 0, and a completely flat window
    (no gains *and* no losses) is the neutral 50.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # Guard the division; the degenerate cases are filled in explicitly below.
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100 - 100 / (1 + rs)

    valid = avg_gain.notna() & avg_loss.notna()
    flat = valid & (avg_loss <= 0) & (avg_gain <= 0)
    only_gains = valid & (avg_loss <= 0) & (avg_gain > 0)

    rsi = rsi.mask(only_gains, 100.0)
    rsi = rsi.mask(flat, 50.0)
    return rsi


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD
) -> pd.Series:
    """Average true range (Wilder smoothing) in price units."""
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()
