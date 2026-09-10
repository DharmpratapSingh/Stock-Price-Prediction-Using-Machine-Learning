"""
Unit tests for the chronological train/val/test split.
"""

import numpy as np
import pandas as pd
import pytest

from src.splits import split_by_year


@pytest.fixture
def frame():
    idx = pd.date_range(start="2017-10-01", end="2025-01-31", freq="D", name="Date")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=len(idx))}, index=idx)


def test_split_is_chronological_and_disjoint(frame):
    train, val, test = split_by_year(frame)

    assert len(train) and len(val) and len(test)

    assert train.index.max() < val.index.min()
    assert val.index.min() < test.index.min()
    assert val.index.max() < test.index.min()

    assert (train.index <= pd.Timestamp("2022-12-31")).all()
    assert (val.index.year == 2023).all()
    assert (test.index.year == 2024).all()

    train_set, val_set, test_set = set(train.index), set(val.index), set(test.index)
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set


def test_next_day_target_never_reaches_into_the_following_split(frame):
    """
    The one-bar embargo: no row's target may be taken from the next split.

    The target is ``Close[t+1] / Close[t] - 1``, so the row immediately before a
    boundary resolves using the first close of the following split. Without the
    embargo, training on the last 2022 bar would consume the first 2023 close and
    threshold selection on the last 2023 bar would consume the first 2024 close.
    The comparison is on the *target* date -- one bar after each row -- not on the
    row's own date.
    """
    train, val, test = split_by_year(frame)
    positions = {ts: i for i, ts in enumerate(frame.index)}

    def target_date(split):
        return frame.index[positions[split.index.max()] + 1]

    assert target_date(train) < val.index.min(), (
        "last training row's target lands inside validation"
    )
    assert target_date(val) < test.index.min(), (
        "last validation row's target lands inside test"
    )


def test_embargo_drops_exactly_one_bar_per_boundary(frame):
    """The embargo costs one row of train and one of validation, and no more."""
    train, val, test = split_by_year(frame)
    index = frame.index

    full_train = index[index <= pd.Timestamp("2022-12-31")]
    full_val = index[index.year == 2023]
    full_test = index[index.year == 2024]

    assert len(train) == len(full_train) - 1
    assert len(val) == len(full_val) - 1
    assert len(test) == len(full_test)  # the test split keeps every row


def test_rows_outside_the_windows_are_excluded(frame):
    train, val, test = split_by_year(frame)
    kept = set(train.index) | set(val.index) | set(test.index)

    outside_2025 = frame.index[frame.index.year == 2025]
    assert len(outside_2025) > 0
    assert not kept & set(outside_2025)

    # 2017 rows are warmup for the model but still belong to train.
    assert (frame.index[frame.index.year == 2017]).isin(train.index).all()


def test_custom_windows(frame):
    train, val, test = split_by_year(
        frame, train_end="2021-12-31", val_year=2022, test_year=2023
    )
    # train_end is the last date *considered*; the one-bar embargo then drops it,
    # so training ends on the preceding bar and its target stops at train_end.
    assert train.index.max() < pd.Timestamp("2021-12-31")
    assert train.index.max() == frame.index[frame.index <= pd.Timestamp("2021-12-31")][-2]
    assert (val.index.year == 2022).all()
    assert (test.index.year == 2023).all()


def test_empty_split_raises(frame):
    with pytest.raises(ValueError):
        split_by_year(frame, train_end="2022-12-31", val_year=2023, test_year=2030)

    only_2023 = frame[frame.index.year == 2023]
    with pytest.raises(ValueError):
        split_by_year(only_2023)  # empty train


def test_invalid_window_ordering_raises(frame):
    with pytest.raises(ValueError):
        split_by_year(frame, train_end="2023-06-30", val_year=2023, test_year=2024)

    with pytest.raises(ValueError):
        split_by_year(frame, train_end="2022-12-31", val_year=2024, test_year=2024)

    with pytest.raises(ValueError):
        split_by_year(frame, train_end="2022-12-31", val_year=2024, test_year=2023)


def test_requires_datetime_index():
    bad = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        split_by_year(bad)
