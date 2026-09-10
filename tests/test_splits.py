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
    assert train.index.max() == pd.Timestamp("2021-12-31")
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
