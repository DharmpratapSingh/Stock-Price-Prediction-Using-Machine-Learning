"""
Split-adjusted daily price loading.

The single most important detail in this module is ``auto_adjust=True``: NVDA
split 4:1 in July 2021 and 10:1 in June 2024, so unadjusted closes contain fake
one-day drops of roughly -75% and -90%. Training on those artifacts teaches a
model to predict crashes that never happened. Every load is therefore checked
by :func:`check_split_artifacts` before it is handed back to the caller.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def check_split_artifacts(close: pd.Series, max_abs_return: float = 0.5) -> None:
    """Raise if any daily simple return exceeds ``max_abs_return`` in magnitude.

    A genuine daily move larger than 50% is essentially unheard of for a large
    cap; a move that large in adjusted data means the series is *not* actually
    adjusted (or is corrupt).

    Args:
        close: Series of closing prices indexed by date.
        max_abs_return: Tolerated absolute one-day simple return.

    Raises:
        ValueError: Listing every offending date and return.
    """
    returns = pd.Series(close).astype(float).pct_change()
    offenders = returns[returns.abs() > max_abs_return].dropna()

    if offenders.empty:
        return None

    details = ", ".join(
        f"{_format_date(date)}: {value:+.4f}" for date, value in offenders.items()
    )
    raise ValueError(
        f"Detected {len(offenders)} daily return(s) beyond "
        f"+/-{max_abs_return:.0%}, which usually means the prices are not "
        f"split-adjusted (use auto_adjust=True). Offending rows -> {details}"
    )


def _format_date(date) -> str:
    """Render an index label as ``YYYY-MM-DD`` when it is a timestamp."""
    try:
        return pd.Timestamp(date).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(date)


def load_prices(
    symbol: str,
    start: str,
    end: str,
    cache_path: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load split-adjusted OHLCV data, from cache when available.

    Args:
        symbol: Ticker to download, e.g. ``"NVDA"``.
        start: Inclusive start date (``YYYY-MM-DD``).
        end: Exclusive end date (``YYYY-MM-DD``).
        cache_path: CSV used as the local cache; written on download.
        refresh: Ignore an existing cache file and re-download.

    Returns:
        DataFrame with columns ``Open, High, Low, Close, Volume`` and an
        ascending, tz-naive ``DatetimeIndex`` named ``Date``.

    Raises:
        ValueError: If the resulting close series contains split artifacts.
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = _normalize(df)
    else:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"No price data returned for {symbol} between {start} and {end}."
            )
        df = _normalize(_flatten_columns(raw))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

    check_split_artifacts(df["Close"])
    return df


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance 1.x ``(field, ticker)`` MultiIndex columns to fields."""
    df = df.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # The field level is whichever one carries the OHLCV names.
    wanted = set(OHLCV_COLUMNS)
    field_level = 0
    for level in range(df.columns.nlevels):
        if wanted & set(df.columns.get_level_values(level)):
            field_level = level
            break

    df.columns = df.columns.get_level_values(field_level)
    df.columns.name = None
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Keep OHLCV only, with a sorted, tz-naive ``Date`` index and no NaNs."""
    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Price data is missing required column(s): {missing}")

    df = df.loc[:, OHLCV_COLUMNS].copy()
    df = df.astype(float)

    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    index.name = "Date"
    df.index = index

    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index().dropna()
