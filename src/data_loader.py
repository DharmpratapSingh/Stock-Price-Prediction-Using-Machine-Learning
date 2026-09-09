"""
Data loading and validation module.

Daily OHLCV data is read from committed CSV snapshots under ``data/raw`` so that
every number in ``results/`` reproduces offline and without a network call. The
network is only touched when a caller explicitly opts in via
``allow_download=True`` (or ``refresh_snapshots.py``-style tooling).

Design note on cleaning: this module deliberately does NOT clip returns and
rebuild ``Close``. Doing so with full-sample mean/standard deviation lets
test-period statistics touch the training window, and a rebuilt ``Close`` no
longer reconciles with ``Open``/``High``/``Low``. Extreme moves are *reported*
here; any clipping that a model needs is a train-window-only transform fitted
inside the modelling pipeline (see ``src.models.WindowedWinsorizer``).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_SNAPSHOT_DIR = "data/raw"


def snapshot_path(symbol: str, snapshot_dir: str = DEFAULT_SNAPSHOT_DIR) -> str:
    """Return the CSV snapshot path for a ticker."""
    return os.path.join(snapshot_dir, f"{symbol.upper()}.csv")


def read_snapshot(symbol: str, snapshot_dir: str = DEFAULT_SNAPSHOT_DIR) -> pd.DataFrame:
    """
    Read a committed OHLCV snapshot from disk.

    Args:
        symbol: Ticker symbol
        snapshot_dir: Directory holding ``<TICKER>.csv`` files

    Returns:
        DataFrame indexed by tz-naive ``Date`` with OHLCV columns
    """
    path = snapshot_path(symbol, snapshot_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No snapshot for {symbol} at {path}. "
            f"Run with allow_download=True to fetch it from Yahoo Finance."
        )

    data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    data.index = pd.DatetimeIndex(data.index).tz_localize(None)
    data.index.name = "Date"
    data = data[[c for c in OHLCV_COLUMNS if c in data.columns]]
    return data.sort_index()


def download_snapshot(
    symbol: str,
    start_date: str,
    end_date: str,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    write: bool = True,
) -> pd.DataFrame:
    """
    Fetch split/dividend-adjusted daily OHLCV from Yahoo Finance and cache it.

    This is the only function in the project that reaches the network.

    Args:
        symbol: Ticker symbol
        start_date: Start date (YYYY-MM-DD), inclusive
        end_date: End date (YYYY-MM-DD), exclusive per yfinance semantics
        snapshot_dir: Where to write ``<TICKER>.csv``
        write: Whether to persist the snapshot

    Returns:
        DataFrame indexed by tz-naive ``Date`` with OHLCV columns
    """
    import yfinance as yf  # imported lazily: offline runs never need it

    logger.info("Downloading %s from %s to %s", symbol, start_date, end_date)
    data = yf.Ticker(symbol).history(
        start=start_date, end=end_date, interval="1d", auto_adjust=True
    )
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")

    data = data[OHLCV_COLUMNS].copy()
    data.index = pd.DatetimeIndex(data.index).tz_localize(None)
    data.index.name = "Date"

    if write:
        os.makedirs(snapshot_dir, exist_ok=True)
        data.to_csv(snapshot_path(symbol, snapshot_dir), float_format="%.6f")
        logger.info("Wrote snapshot for %s (%d rows)", symbol, len(data))

    return data


class StockDataLoader:
    """Loads and validates daily OHLCV data for one ticker."""

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.snapshot_dir = snapshot_dir
        self.data: Optional[pd.DataFrame] = None

    def load(self, allow_download: bool = False) -> pd.DataFrame:
        """
        Load OHLCV data, preferring the committed snapshot.

        Args:
            allow_download: If True, fall back to the network when no snapshot
                exists. Defaults to False so runs stay reproducible and offline.

        Returns:
            DataFrame restricted to ``[start_date, end_date]``
        """
        try:
            data = read_snapshot(self.symbol, self.snapshot_dir)
            logger.info("Loaded %s from snapshot (%d rows)", self.symbol, len(data))
        except FileNotFoundError:
            if not allow_download:
                raise
            data = download_snapshot(
                self.symbol, self.start_date, self.end_date, self.snapshot_dir
            )

        data = data.loc[
            (data.index >= pd.Timestamp(self.start_date))
            & (data.index <= pd.Timestamp(self.end_date))
        ]
        if data.empty:
            raise ValueError(
                f"Snapshot for {self.symbol} has no rows in "
                f"[{self.start_date}, {self.end_date}]"
            )

        self.data = data
        return data

    def validate_data(self, data: Optional[pd.DataFrame] = None) -> Tuple[bool, List[str]]:
        """
        Check data quality. Reports problems; never silently rewrites prices.

        Returns:
            (is_valid, list of human-readable issues)
        """
        if data is None:
            data = self.data
        if data is None:
            return False, ["No data loaded"]

        issues: List[str] = []

        missing = data.isnull().sum()
        if missing.any():
            issues.append(f"Missing values: {missing[missing > 0].to_dict()}")

        if data.index.duplicated().any():
            issues.append("Duplicate dates detected")

        if not data.index.is_monotonic_increasing:
            issues.append("Index is not sorted ascending")

        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns and (data[col] <= 0).any():
                issues.append(f"Non-positive values in {col}")

        if {"High", "Low"} <= set(data.columns) and (data["High"] < data["Low"]).any():
            issues.append("High below Low detected")

        n_extreme = int(self.flag_extreme_moves(data).sum())
        if n_extreme:
            issues.append(f"Extreme daily moves (>50%): {n_extreme}")

        is_valid = not issues
        if is_valid:
            logger.info("Data validation passed for %s", self.symbol)
        else:
            for issue in issues:
                logger.warning("%s: %s", self.symbol, issue)

        return is_valid, issues

    @staticmethod
    def flag_extreme_moves(data: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Flag days whose close-to-close move exceeds ``threshold``.

        Reporting only. Prices are never modified, so ``Close`` always stays
        consistent with ``Open``/``High``/``Low``.
        """
        if "Close" not in data.columns:
            return pd.Series(False, index=data.index)
        return data["Close"].pct_change().abs() > threshold

    def handle_missing_data(self, method: str = "ffill") -> pd.DataFrame:
        """
        Fill gaps in an otherwise-valid series.

        Only forward fill and dropping are offered: back-filling a price series
        copies future prices into the past, which is lookahead.
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if method == "ffill":
            self.data = self.data.ffill()
        elif method == "drop":
            self.data = self.data.dropna()
        else:
            raise ValueError(
                f"Unsupported method: {method!r}. Use 'ffill' or 'drop'; "
                f"back-filling prices would leak future information."
            )

        return self.data


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    allow_download: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load one ticker's daily OHLCV, snapshot-first.

    Args:
        symbol: Ticker symbol
        start_date: Inclusive start date (YYYY-MM-DD)
        end_date: Inclusive end date (YYYY-MM-DD)
        snapshot_dir: Directory of committed CSV snapshots
        allow_download: Permit a network fetch when no snapshot exists
        validate: Log data-quality issues after loading

    Returns:
        DataFrame indexed by date with OHLCV columns
    """
    loader = StockDataLoader(symbol, start_date, end_date, snapshot_dir)
    data = loader.load(allow_download=allow_download)

    if data.isnull().values.any():
        loader.handle_missing_data("ffill")
        data = loader.data

    if validate:
        loader.validate_data(data)

    return data


def load_basket(
    symbols: List[str],
    start_date: str,
    end_date: str,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    allow_download: bool = False,
) -> dict:
    """
    Load several tickers into a ``{symbol: DataFrame}`` mapping.

    Each ticker is modelled independently; nothing is pooled at load time.
    """
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = load_stock_data(
                symbol, start_date, end_date, snapshot_dir, allow_download
            )
        except Exception as exc:  # pragma: no cover - surfaced to the caller
            logger.error("Failed to load %s: %s", symbol, exc)
            raise
    return out
