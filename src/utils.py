"""
Utility functions for the stock prediction project
"""

import os
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def time_series_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data chronologically

    Args:
        data: DataFrame to split
        test_size: Fraction for test set
        validation_size: Fraction for validation set

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - validation_size))

    train_data = data.iloc[:val_idx]
    val_data = data.iloc[val_idx:test_idx]
    test_data = data.iloc[test_idx:]

    return train_data, val_data, test_data


@dataclass(frozen=True)
class Fold:
    """
    One walk-forward fold, as positional slices into a chronological dataset.

    Slices are half-open: train is ``[train_start, train_end)`` and test is
    ``[test_start, test_end)``. ``test_start - train_end`` is the embargo.
    """

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start

    @property
    def embargo(self) -> int:
        return self.test_start - self.train_end

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def walk_forward_folds(
    n_samples: int,
    initial_train: int = 504,
    test_size: int = 63,
    step_size: int = 63,
    embargo: int = 0,
    expanding: bool = True
) -> List[Fold]:
    """
    Build ordered walk-forward folds with an embargo between train and test.

    Folds run strictly forward in time and never shuffle. The embargo is the
    number of rows skipped between the end of the training window and the start
    of the test window. Setting it to the feature warm-up length (see
    ``feature_engineering.feature_warmup_length``) guarantees that no feature row
    in the test fold is computed from any row used for training -- without it,
    the deepest rolling window would straddle the boundary.

    Args:
        n_samples: Number of chronologically ordered rows
        initial_train: Rows in the first training window
        test_size: Rows per test fold
        step_size: Rows the window advances between folds
        embargo: Rows skipped between train end and test start
        expanding: True for an expanding training window (train always starts at
            row 0), False for a rolling window of fixed length ``initial_train``

    Returns:
        List of Fold objects, in time order

    Raises:
        ValueError: if the arguments cannot produce a single fold
    """
    if min(initial_train, test_size, step_size) < 1 or embargo < 0:
        raise ValueError("initial_train, test_size and step_size must be >= 1; embargo >= 0")

    folds: List[Fold] = []
    train_end = initial_train
    idx = 0

    while train_end + embargo + test_size <= n_samples:
        test_start = train_end + embargo
        train_start = 0 if expanding else max(0, train_end - initial_train)
        folds.append(
            Fold(
                index=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_start + test_size,
            )
        )
        idx += 1
        train_end += step_size

    if not folds:
        raise ValueError(
            f"No folds fit: n_samples={n_samples} needs at least "
            f"{initial_train + embargo + test_size} rows for "
            f"initial_train={initial_train}, embargo={embargo}, test_size={test_size}"
        )

    return folds


def folds_to_frame(folds: List[Fold], index: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Tabulate folds, optionally translating positions into dates.

    Args:
        folds: Folds from walk_forward_folds
        index: The dataset's DatetimeIndex, to add date boundaries

    Returns:
        DataFrame with one row per fold
    """
    frame = pd.DataFrame([f.as_dict() for f in folds])
    frame['n_train'] = frame['train_end'] - frame['train_start']
    frame['n_test'] = frame['test_end'] - frame['test_start']
    frame['embargo'] = frame['test_start'] - frame['train_end']

    if index is not None:
        frame['train_start_date'] = [index[f.train_start].date() for f in folds]
        frame['train_end_date'] = [index[f.train_end - 1].date() for f in folds]
        frame['test_start_date'] = [index[f.test_start].date() for f in folds]
        frame['test_end_date'] = [index[f.test_end - 1].date() for f in folds]

    return frame


def save_artifact(artifact: Dict[str, Any], path: str) -> str:
    """
    Persist the single artifact that train.py produces and predict.py consumes.

    Bundling the fitted pipeline with the exact feature list and the config that
    built it is what keeps inference honest: predict.py cannot accidentally
    assemble a different feature set from the one the model was fitted on.

    Args:
        artifact: Must contain 'pipeline', 'feature_columns' and 'config'
        path: Destination .joblib path

    Returns:
        The path written
    """
    required = {'pipeline', 'feature_columns', 'config'}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artifact is missing required keys: {sorted(missing)}")

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: str) -> Dict[str, Any]:
    """
    Load a training artifact and check it carries what inference needs.

    Args:
        path: Path to a .joblib written by save_artifact

    Returns:
        The artifact dictionary
    """
    # joblib.load unpickles, so it executes code from the file. Only ever point
    # this at artifacts this project's own train.py wrote under models/; treat a
    # .joblib from anywhere else as untrusted and do not load it.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or 'pipeline' not in artifact:
        raise ValueError(
            f"{path} is not a training artifact. Retrain with train.py to "
            f"produce a pipeline + feature list bundle."
        )
    return artifact


