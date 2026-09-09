"""
Unit tests for adjusted price loading and split-artifact detection.

No network access: yfinance.download is always monkeypatched.
"""

import numpy as np
import pandas as pd
import pytest

from src.market_data import check_split_artifacts, load_prices

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
    assert check_split_artifacts(_random_walk(seed=7)) is None


def test_check_split_artifacts_respects_threshold():
    close = _random_walk(seed=3)
    close.iloc[5:] = close.iloc[5:] * 0.45  # a single ~ -55% move
    with pytest.raises(ValueError):
        check_split_artifacts(close, max_abs_return=0.5)
    # A looser threshold tolerates it.
    assert check_split_artifacts(close, max_abs_return=0.9) is None


# --------------------------------------------------------------------------
# load_prices: cache path
# --------------------------------------------------------------------------


def test_load_prices_uses_cache_without_network(tmp_path, monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("yfinance.download must not be called when cache exists")

    monkeypatch.setattr("yfinance.download", boom)

    frame = _ohlcv_frame(_random_walk(seed=2))
    cache_path = tmp_path / "nvda.csv"
    frame.to_csv(cache_path)

    out = load_prices("NVDA", "2020-01-01", "2020-04-01", cache_path)

    assert out.columns.tolist() == OHLCV
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "Date"
    assert out.index.tz is None
    pd.testing.assert_series_equal(
        out["Close"], frame["Close"], check_freq=False, rtol=1e-9
    )


def test_load_prices_cache_with_split_artifact_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yfinance.download",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")),
    )

    close = _random_walk(seed=4)
    close.iloc[20] = close.iloc[19] * 0.1
    cache_path = tmp_path / "bad.csv"
    _ohlcv_frame(close).to_csv(cache_path)

    with pytest.raises(ValueError):
        load_prices("NVDA", "2020-01-01", "2020-04-01", cache_path)


# --------------------------------------------------------------------------
# load_prices: download path
# --------------------------------------------------------------------------


def _fake_yf_frame(n=60, seed=5):
    """Mimic yfinance 1.x: MultiIndex (field, ticker) columns, tz-aware index."""
    close = _random_walk(n=n, seed=seed)
    flat = _ohlcv_frame(close)
    flat.index = pd.DatetimeIndex(flat.index, name="Date").tz_localize("America/New_York")
    # yfinance orders columns alphabetically and adds a ticker level.
    flat = flat[["Close", "High", "Low", "Open", "Volume"]]
    flat.columns = pd.MultiIndex.from_product(
        [flat.columns, ["NVDA"]], names=["Price", "Ticker"]
    )
    return flat


def test_load_prices_downloads_with_auto_adjust(tmp_path, monkeypatch):
    calls = []

    def fake_download(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return _fake_yf_frame()

    monkeypatch.setattr("yfinance.download", fake_download)

    cache_path = tmp_path / "nested" / "dir" / "nvda.csv"
    out = load_prices("NVDA", "2019-01-01", "2020-01-01", cache_path)

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
    calls = []

    def fake_download(*args, **kwargs):
        calls.append(kwargs)
        return _fake_yf_frame(seed=9)

    monkeypatch.setattr("yfinance.download", fake_download)

    cache_path = tmp_path / "nvda.csv"
    _ohlcv_frame(_random_walk(seed=2)).to_csv(cache_path)

    load_prices("NVDA", "2019-01-01", "2020-01-01", cache_path, refresh=True)

    assert len(calls) == 1
    assert calls[0].get("auto_adjust") is True


def test_load_prices_download_dropna_and_sort(tmp_path, monkeypatch):
    frame = _fake_yf_frame(n=30, seed=11)
    frame.iloc[3, :] = np.nan
    frame = frame.iloc[::-1]  # unsorted on purpose

    monkeypatch.setattr("yfinance.download", lambda *a, **k: frame)

    out = load_prices("NVDA", "2019-01-01", "2020-01-01", tmp_path / "x.csv")

    assert len(out) == 29
    assert out.index.is_monotonic_increasing
