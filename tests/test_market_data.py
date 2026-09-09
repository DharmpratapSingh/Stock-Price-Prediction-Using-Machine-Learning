"""
Unit tests for adjusted price loading and split-artifact detection.

No network access: yfinance.download is always monkeypatched.
"""

import numpy as np
import pandas as pd
import pytest

from src.market_data import (
    _flatten_columns,
    _normalize,
    check_split_artifacts,
    load_prices,
)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _random_walk(n=60, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n, name="Date")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    return pd.Series(close, index=idx, name="Close")


def _ohlcv_frame(close):
    rng = np.random.default_rng(1)
    n = len(close)
    noise = np.abs(rng.normal(0.0, 0.01, n)) + 1e-4
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]).to_numpy(),
            "High": close.to_numpy() * (1 + noise),
            "Low": close.to_numpy() * (1 - noise),
            "Close": close.to_numpy(),
            "Volume": rng.lognormal(15.0, 0.2, n),
        },
        index=close.index,
    )


def _day_after(timestamp) -> str:
    """The exclusive `end` that keeps `timestamp` inside the requested range."""
    return (pd.Timestamp(timestamp) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _no_network(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("yfinance.download must not be called")

    monkeypatch.setattr("yfinance.download", boom)


def _record_download(monkeypatch, frame_factory):
    calls = []

    def fake_download(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return frame_factory()

    monkeypatch.setattr("yfinance.download", fake_download)
    return calls


# --------------------------------------------------------------------------
# check_split_artifacts
# --------------------------------------------------------------------------


def test_check_split_artifacts_flags_unadjusted_split():
    close = _random_walk()
    split_date = close.index[10]
    # Simulate an unadjusted 4:1 split: price drops to 25% of the prior close.
    close.loc[split_date:] = close.loc[split_date:] * 0.25

    with pytest.raises(ValueError) as excinfo:
        check_split_artifacts(close)

    message = str(excinfo.value)
    assert split_date.strftime("%Y-%m-%d") in message
    assert "-0.7" in message  # the reported return is ~ -75%


def test_check_split_artifacts_passes_on_normal_series():
    check_split_artifacts(_random_walk(seed=7))  # does not raise


def test_check_split_artifacts_respects_threshold():
    close = _random_walk(seed=3)
    close.iloc[5:] = close.iloc[5:] * 0.45  # a single ~ -55% move
    with pytest.raises(ValueError):
        check_split_artifacts(close, max_abs_return=0.5)
    check_split_artifacts(close, max_abs_return=0.9)  # does not raise


# --------------------------------------------------------------------------
# load_prices: cache path
# --------------------------------------------------------------------------


def test_load_prices_uses_cache_without_network(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    frame = _ohlcv_frame(_random_walk(seed=2))
    cache_path = tmp_path / "nvda.csv"
    frame.to_csv(cache_path)

    out = load_prices("NVDA", "2020-01-01", _day_after(frame.index[-1]), cache_path)

    assert out.columns.tolist() == OHLCV
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"
    assert out.index.tz is None
    pd.testing.assert_series_equal(
        out["Close"], frame["Close"], check_freq=False, rtol=1e-9
    )


def test_load_prices_slices_cache_to_requested_range(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    frame = _ohlcv_frame(_random_walk(n=500, seed=8, start="2019-01-02"))
    cache_path = tmp_path / "wide.csv"
    frame.to_csv(cache_path)

    start, end = "2020-01-01", "2020-07-01"
    out = load_prices("NVDA", start, end, cache_path)

    assert len(out) < len(frame)
    assert out.index.min() >= pd.Timestamp(start)
    # yfinance treats `end` as exclusive; the cache slice must match.
    assert out.index.max() < pd.Timestamp(end)
    expected = frame.loc[
        (frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))
    ]
    assert len(out) == len(expected)


def test_load_prices_redownloads_when_cache_ends_too_early(tmp_path, monkeypatch):
    frame = _ohlcv_frame(_random_walk(seed=2))  # ends in Q1 2020
    cache_path = tmp_path / "short.csv"
    frame.to_csv(cache_path)

    calls = _record_download(monkeypatch, lambda: _fake_yf_frame(seed=12))

    load_prices("NVDA", "2020-01-01", "2021-01-01", cache_path)

    assert len(calls) == 1
    assert calls[0]["kwargs"].get("auto_adjust") is True


def test_load_prices_redownloads_when_cache_starts_too_late(tmp_path, monkeypatch):
    frame = _ohlcv_frame(_random_walk(seed=2, start="2020-01-01"))
    cache_path = tmp_path / "late.csv"
    frame.to_csv(cache_path)

    calls = _record_download(monkeypatch, lambda: _fake_yf_frame(seed=13))

    load_prices("NVDA", "2018-01-01", _day_after(frame.index[-1]), cache_path)

    assert len(calls) == 1


def test_load_prices_cache_with_split_artifact_raises(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    close = _random_walk(seed=4)
    close.iloc[20] = close.iloc[19] * 0.1
    cache_path = tmp_path / "bad.csv"
    _ohlcv_frame(close).to_csv(cache_path)

    with pytest.raises(ValueError):
        load_prices("NVDA", "2020-01-01", _day_after(close.index[-1]), cache_path)


def test_load_prices_max_abs_return_is_configurable(tmp_path, monkeypatch):
    _no_network(monkeypatch)

    close = _random_walk(seed=3)
    close.iloc[5:] = close.iloc[5:] * 0.45  # a single ~ -55% move
    cache_path = tmp_path / "halved.csv"
    _ohlcv_frame(close).to_csv(cache_path)

    end = _day_after(close.index[-1])
    with pytest.raises(ValueError):
        load_prices("NVDA", "2020-01-01", end, cache_path)

    out = load_prices("NVDA", "2020-01-01", end, cache_path, max_abs_return=0.9)
    assert len(out) == len(close)


# --------------------------------------------------------------------------
# load_prices: download path
# --------------------------------------------------------------------------


def _fake_yf_frame(n=60, seed=5, close=None):
    """Mimic yfinance 1.x: MultiIndex (field, ticker) columns, tz-aware index."""
    if close is None:
        close = _random_walk(n=n, seed=seed)
    flat = _ohlcv_frame(close)
    flat.index = pd.DatetimeIndex(flat.index, name="Date").tz_localize(
        "America/New_York"
    )
    # yfinance orders columns alphabetically and adds a ticker level.
    flat = flat[["Close", "High", "Low", "Open", "Volume"]]
    flat.columns = pd.MultiIndex.from_product(
        [flat.columns, ["NVDA"]], names=["Price", "Ticker"]
    )
    return flat


def test_load_prices_downloads_with_auto_adjust(tmp_path, monkeypatch):
    calls = _record_download(monkeypatch, _fake_yf_frame)

    cache_path = tmp_path / "nested" / "dir" / "nvda.csv"
    out = load_prices("NVDA", "2019-01-01", "2020-04-01", cache_path)

    assert len(calls) == 1
    assert calls[0]["kwargs"].get("auto_adjust") is True
    assert calls[0]["kwargs"].get("progress") is False

    assert out.columns.tolist() == OHLCV
    assert not isinstance(out.columns, pd.MultiIndex)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    assert not out.isna().any().any()

    assert cache_path.exists()
    cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    assert cached.columns.tolist() == OHLCV
    assert len(cached) == len(out)


def test_load_prices_refresh_bypasses_cache(tmp_path, monkeypatch):
    calls = _record_download(monkeypatch, lambda: _fake_yf_frame(seed=9))

    cache_path = tmp_path / "nvda.csv"
    _ohlcv_frame(_random_walk(seed=2)).to_csv(cache_path)

    load_prices("NVDA", "2019-01-01", "2020-04-01", cache_path, refresh=True)

    assert len(calls) == 1
    assert calls[0]["kwargs"].get("auto_adjust") is True


def test_load_prices_download_dropna_and_sort(tmp_path, monkeypatch):
    frame = _fake_yf_frame(n=30, seed=11)
    frame.iloc[3, :] = np.nan
    frame = frame.iloc[::-1]  # unsorted on purpose

    monkeypatch.setattr("yfinance.download", lambda *a, **k: frame)

    out = load_prices("NVDA", "2019-01-01", "2020-04-01", tmp_path / "x.csv")

    assert len(out) == 29
    assert out.index.is_monotonic_increasing


def test_split_artifact_in_download_is_not_cached(tmp_path, monkeypatch):
    close = _random_walk(seed=6)
    close.iloc[30:] = close.iloc[30:] * 0.25  # unadjusted 4:1 split

    monkeypatch.setattr(
        "yfinance.download", lambda *a, **k: _fake_yf_frame(close=close)
    )

    cache_path = tmp_path / "poison.csv"
    with pytest.raises(ValueError):
        load_prices("NVDA", "2019-01-01", "2020-04-01", cache_path)

    assert not cache_path.exists()


def test_empty_download_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame())

    cache_path = tmp_path / "empty.csv"
    with pytest.raises(ValueError):
        load_prices("NVDA", "2019-01-01", "2020-01-01", cache_path)

    assert not cache_path.exists()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_normalize_names_the_missing_column():
    frame = _ohlcv_frame(_random_walk(n=10)).drop(columns=["Volume"])
    with pytest.raises(ValueError) as excinfo:
        _normalize(frame)
    assert "Volume" in str(excinfo.value)


def test_flatten_columns_handles_ticker_first_orientation():
    frame = _ohlcv_frame(_random_walk(n=10))
    frame.columns = pd.MultiIndex.from_product(
        [["NVDA"], frame.columns], names=["Ticker", "Price"]
    )

    flat = _flatten_columns(frame)

    assert not isinstance(flat.columns, pd.MultiIndex)
    assert flat.columns.tolist() == OHLCV


def test_flatten_columns_passes_through_flat_frames():
    frame = _ohlcv_frame(_random_walk(n=10))
    pd.testing.assert_frame_equal(_flatten_columns(frame), frame)
