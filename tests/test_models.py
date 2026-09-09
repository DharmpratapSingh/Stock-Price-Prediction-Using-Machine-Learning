"""
Unit tests for models, baselines and the fit-in-window pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_dataset
from src.models import (
    DEFAULT_PARAMS,
    MODEL_FAMILIES,
    AlwaysUpClassifier,
    PersistenceReturnRegressor,
    TrainMeanRegressor,
    WindowedWinsorizer,
    ZeroReturnRegressor,
    build_pipeline,
    get_baseline,
    get_estimator,
    get_feature_importance,
)


@pytest.fixture
def synthetic_ohlcv():
    """Deterministic random-walk OHLCV, long enough for the deepest window."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-01", periods=700)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
    spread = close * 0.01
    data = pd.DataFrame(
        {
            "Open": close + rng.normal(0, spread),
            "High": close + np.abs(rng.normal(0, spread)),
            "Low": close - np.abs(rng.normal(0, spread)),
            "Close": close,
            "Volume": rng.uniform(1e6, 5e6, len(dates)),
        },
        index=dates,
    )
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)
    return data


@pytest.fixture
def dataset(synthetic_ohlcv):
    return build_dataset(synthetic_ohlcv, config=None, horizon=1)


def test_get_estimator_covers_all_families():
    for family in MODEL_FAMILIES:
        assert get_estimator(family, "regression") is not None
        assert get_estimator(family, "classification") is not None


def test_get_estimator_rejects_unknown():
    with pytest.raises(ValueError):
        get_estimator("lstm", "regression")
    with pytest.raises(ValueError):
        get_estimator("linear", "ranking")


def test_default_params_documented_for_every_family():
    """Hyperparameters are fixed and explicit, not tuned on a test fold."""
    for family in MODEL_FAMILIES:
        assert {"regressor", "classifier"} == set(DEFAULT_PARAMS[family])


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_regression_pipeline_fits_and_predicts(dataset, family):
    data, features = dataset
    pipeline = build_pipeline(family, "regression", k_features=20)
    split = int(len(data) * 0.7)

    pipeline.fit(data[features].iloc[:split], data["target_logret"].iloc[:split])
    predictions = pipeline.predict(data[features].iloc[split:])

    assert len(predictions) == len(data) - split
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_classification_pipeline_fits_and_predicts(dataset, family):
    data, features = dataset
    pipeline = build_pipeline(family, "classification", k_features=20)
    split = int(len(data) * 0.7)

    pipeline.fit(data[features].iloc[:split], data["target_direction"].iloc[:split])
    predictions = pipeline.predict(data[features].iloc[split:])

    assert set(np.unique(predictions)) <= {0, 1}
    assert pipeline.predict_proba(data[features].iloc[split:]).shape[1] == 2


def test_pipeline_statistics_are_fitted_on_training_rows_only(dataset):
    """
    Refitting on a superset must change the fitted scaler.

    This is the property that makes the pipeline leakage-free: every statistic
    comes from fit(), so adding test rows to the training data is the only way to
    influence it. If scaling had been done up front on the full dataset, these
    two fits would be identical.
    """
    data, features = dataset
    split = int(len(data) * 0.7)

    short = build_pipeline("linear", "regression", k_features=None)
    short.fit(data[features].iloc[:split], data["target_logret"].iloc[:split])

    full = build_pipeline("linear", "regression", k_features=None)
    full.fit(data[features], data["target_logret"])

    assert not np.allclose(
        short.named_steps["scale"].mean_, full.named_steps["scale"].mean_
    )


def test_windowed_winsorizer_learns_bounds_from_fit_only():
    train = np.array([[1.0], [2.0], [3.0], [4.0]])
    clipper = WindowedWinsorizer(lower=0.0, upper=1.0).fit(train)

    # A test value far outside the training range is clipped to the training max,
    # and does not move the learned bound.
    out = clipper.transform(np.array([[1000.0]]))
    assert out[0, 0] == pytest.approx(4.0)
    assert clipper.upper_bounds_[0] == pytest.approx(4.0)


def test_zero_baseline_predicts_zero(dataset):
    data, features = dataset
    model = ZeroReturnRegressor().fit(data[features], data["target_logret"])
    assert np.all(model.predict(data[features]) == 0)


def test_train_mean_baseline_uses_training_mean_only(dataset):
    data, features = dataset
    split = int(len(data) * 0.7)
    y_train = data["target_logret"].iloc[:split]

    model = TrainMeanRegressor().fit(data[features].iloc[:split], y_train)
    predictions = model.predict(data[features].iloc[split:])

    assert np.allclose(predictions, y_train.mean())


def test_persistence_baseline_returns_yesterdays_return(dataset):
    data, features = dataset
    model = PersistenceReturnRegressor().fit(data[features], data["target_logret"])
    predictions = model.predict(data[features])
    np.testing.assert_allclose(predictions, data["log_return_1d"].to_numpy())


def test_persistence_baseline_requires_its_feature():
    frame = pd.DataFrame({"something_else": [1.0, 2.0]})
    with pytest.raises(ValueError, match="log_return_1d"):
        PersistenceReturnRegressor().fit(frame, pd.Series([0.0, 0.0]))


def test_always_up_baseline(dataset):
    data, features = dataset
    model = AlwaysUpClassifier().fit(data[features], data["target_direction"])
    assert np.all(model.predict(data[features]) == 1)
    assert np.all(model.predict_proba(data[features])[:, 1] == 1)


def test_get_baseline_rejects_unknown():
    with pytest.raises(ValueError):
        get_baseline("magic", "regression")


def test_feature_importance_maps_through_selection(dataset):
    """Importances must line up with original names, not post-selection indices."""
    data, features = dataset
    pipeline = build_pipeline("forest", "regression", k_features=10)
    pipeline.fit(data[features], data["target_logret"])

    importance = get_feature_importance(pipeline, features)
    assert len(importance) == 10
    assert set(importance["feature"]) <= set(features)
    assert importance["importance"].is_monotonic_decreasing


def test_no_tensorflow_or_lstm_paths_remain():
    """The LSTM / TensorFlow path was removed, not left dormant."""
    import src.models as models

    source = open(models.__file__).read()
    for banned in ("tensorflow", "keras", "LSTM", "Sequential", "stacking"):
        assert banned.lower() not in source.lower(), f"{banned} still referenced"
