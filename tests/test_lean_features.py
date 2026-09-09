"""
Unit tests for the lean, leakage-free feature set.

Every feature at row t must use only information available at or before t.
"""

import numpy as np
import pandas as pd
import pytest

from src.lean_features import (
    FEATURE_COLUMNS,
    build_dataset,
    build_features,
    build_target,
)


@pytest.fixture
def ohlcv():
    """400 business days of synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    n = 400
    idx = pd.bdate_range(start="2019-01-02", periods=n, name="Date")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    close = pd.Series(close, index=idx)
    noise = np.abs(rng.normal(0.0, 0.01, n)) + 1e-4
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * (1 + noise),
            "Low": close * (1 - noise),
            "Close": close,
            "Volume": rng.lognormal(15.0, 0.3, n),
        },
        index=idx,
    )


# --------------------------------------------------------------------------
# (a) + (b) leakage
# --------------------------------------------------------------------------


def test_features_do_not_depend_on_future_rows(ohlcv):
    truncated = build_features(ohlcv.iloc[:300])
    full_prefix = build_features(ohlcv).iloc[:300]
    pd.testing.assert_frame_equal(truncated, full_prefix)


def test_appending_a_shock_row_does_not_change_history(ohlcv):
    extra = ohlcv.iloc[[-1]].copy()
    extra.index = pd.DatetimeIndex(
        [ohlcv.index[-1] + pd.tseries.offsets.BDay(1)], name="Date"
    )
    new_close = ohlcv["Close"].iloc[-1] * 1.5
    extra["Close"] = new_close
    extra["Open"] = ohlcv["Close"].iloc[-1]
    extra["High"] = new_close * 1.01
    extra["Low"] = ohlcv["Close"].iloc[-1] * 0.99
    extra["Volume"] = ohlcv["Volume"].iloc[-1]

    extended = pd.concat([ohlcv, extra])

    pd.testing.assert_frame_equal(
        build_features(ohlcv), build_features(extended).iloc[:-1], check_freq=False
    )


# --------------------------------------------------------------------------
# (c) target
# --------------------------------------------------------------------------


def test_target_is_next_day_simple_return(ohlcv):
    target = build_target(ohlcv)
    expected = ohlcv["Close"].shift(-1) / ohlcv["Close"] - 1
    assert target.name == "target"
    pd.testing.assert_series_equal(target, expected, check_names=False)
    assert np.isnan(target.iloc[-1])
    assert target.notna().sum() == len(ohlcv) - 1


# --------------------------------------------------------------------------
# (d) shape / cleanliness
# --------------------------------------------------------------------------


def test_feature_columns_contract(ohlcv):
    assert 15 <= len(FEATURE_COLUMNS) <= 20
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert build_features(ohlcv).columns.tolist() == FEATURE_COLUMNS


def test_build_dataset_columns_and_cleanliness(ohlcv):
    data = build_dataset(ohlcv)

    assert data.columns.tolist() == FEATURE_COLUMNS + ["target"]
    assert not data.isna().any().any()
    assert np.isfinite(data.to_numpy(dtype=float)).all()

    dropped = len(ohlcv) - len(data)
    assert dropped >= 63 + 1
    assert data.index.is_monotonic_increasing
    assert data.index[-1] == ohlcv.index[-2]


# --------------------------------------------------------------------------
# (e) indicator sanity
# --------------------------------------------------------------------------


def test_rsi_and_atr_ranges(ohlcv):
    feats = build_features(ohlcv)

    rsi = feats["rsi_14"].dropna()
    assert len(rsi) > 300
    assert rsi.between(0, 100).all()

    atr = feats["atr_14_pct"].dropna()
    assert (atr > 0).all()

    assert np.isfinite(feats["rsi_14"].dropna()).all()
    assert np.isfinite(feats["atr_14_pct"].dropna()).all()


def test_zero_volume_stretch_yields_nan_not_inf(ohlcv):
    """A dead-volume stretch must not leak inf into volume_ratio_20."""
    dead = ohlcv.copy()
    dead.iloc[100:130, dead.columns.get_loc("Volume")] = 0.0

    feats = build_features(dead)
    ratio = feats["volume_ratio_20"]

    # Deep inside the stretch the 20-day average volume is itself zero.
    assert ratio.iloc[120:130].isna().all()
    assert not np.isinf(feats.to_numpy(dtype=float)).any()
    assert np.isfinite(feats.dropna().to_numpy(dtype=float)).all()


def test_constant_price_series_gives_finite_rsi(ohlcv):
    flat = ohlcv.copy()
    flat["Close"] = 100.0
    flat["Open"] = 100.0
    flat["High"] = 100.5
    flat["Low"] = 99.5

    rsi = build_features(flat)["rsi_14"].iloc[14:]
    assert rsi.notna().all()
    assert np.isfinite(rsi).all()


def test_rsi_saturates_on_monotonic_advance(ohlcv):
    up = ohlcv.copy()
    up["Close"] = 100.0 * (1.01 ** np.arange(len(up)))
    up["High"] = up["Close"] * 1.001
    up["Low"] = up["Close"] * 0.999
    up["Open"] = up["Close"].shift(1).fillna(up["Close"].iloc[0])

    rsi = build_features(up)["rsi_14"].iloc[20:]
    assert np.isfinite(rsi).all()
    assert (rsi > 99.9).all()


def test_known_feature_values(ohlcv):
    feats = build_features(ohlcv)
    close = ohlcv["Close"]

    pd.testing.assert_series_equal(
        feats["ret_lag0"], close.pct_change(), check_names=False
    )
    pd.testing.assert_series_equal(
        feats["ret_lag3"], close.pct_change().shift(3), check_names=False
    )
    pd.testing.assert_series_equal(
        feats["ret_21d"], close / close.shift(21) - 1, check_names=False
    )
    pd.testing.assert_series_equal(
        feats["sma_ratio_20"], close / close.rolling(20).mean() - 1, check_names=False
    )
    pd.testing.assert_series_equal(
        feats["vol_21"], close.pct_change().rolling(21).std(), check_names=False
    )
    pd.testing.assert_series_equal(
        feats["volume_ratio_20"],
        ohlcv["Volume"] / ohlcv["Volume"].rolling(20).mean(),
        check_names=False,
    )
