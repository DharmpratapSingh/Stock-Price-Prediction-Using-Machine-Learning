"""
Chronological train / validation / test splitting.

Random splits leak the future into the past on time series, so the split is
fixed by calendar: everything up to ``train_end`` trains, one whole year
validates, and the following year is the untouched test set.

A calendar boundary is not by itself enough. The target is the *next* day's
return, so the last row on each side of a boundary carries a target computed from
the first close of the following split: training on 2022-12-30 would use the
2023-01-03 close, the first validation day. ``split_by_year`` therefore drops the
final row of train and of validation, which is a one-bar embargo at each
boundary and the smallest gap that makes the splits genuinely disjoint.
"""

from __future__ import annotations

import pandas as pd


def split_by_year(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_year: int = 2023,
    test_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a date-indexed frame into train, validation and test slices.

    Rows outside the three windows (early warmup years, or years after the test
    year) are simply excluded. The final row of train and of validation is dropped
    so that neither carries a next-day target taken from the following split --
    see the module docstring.

    Args:
        df: Frame with a ``DatetimeIndex``.
        train_end: Last date considered for training (inclusive, before the
            one-bar embargo is applied).
        val_year: Calendar year used for validation.
        test_year: Calendar year used for the final test.

    Returns:
        ``(train, val, test)``.

    Raises:
        ValueError: If the index is not a ``DatetimeIndex``, if the windows are
            not strictly ordered, or if any split would be empty.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"split_by_year requires a DatetimeIndex, got {type(df.index).__name__}."
        )

    val_year = int(val_year)
    test_year = int(test_year)
    train_end_ts = pd.Timestamp(train_end)
    val_start = pd.Timestamp(year=val_year, month=1, day=1)

    if train_end_ts >= val_start:
        raise ValueError(
            f"train_end ({train_end_ts.date()}) must fall strictly before the "
            f"start of the validation year ({val_year})."
        )
    if val_year >= test_year:
        raise ValueError(
            f"val_year ({val_year}) must be strictly before test_year ({test_year})."
        )

    index = df.index
    train = df.loc[index <= train_end_ts]
    val = df.loc[index.year == val_year]
    test = df.loc[index.year == test_year]

    # One-bar embargo at each boundary: the dropped row's target is the return
    # into the first day of the next split, so keeping it would put a validation
    # close into training and a test close into threshold selection.
    train = train.iloc[:-1]
    val = val.iloc[:-1]

    empty = [
        name
        for name, split in (("train", train), ("val", val), ("test", test))
        if len(split) == 0
    ]
    if empty:
        span = (
            f"{index.min().date()} to {index.max().date()}" if len(index) else "empty"
        )
        raise ValueError(
            f"Empty split(s): {', '.join(empty)}. Data spans {span}; requested "
            f"train_end={train_end_ts.date()}, val_year={val_year}, "
            f"test_year={test_year}."
        )

    return train, val, test
