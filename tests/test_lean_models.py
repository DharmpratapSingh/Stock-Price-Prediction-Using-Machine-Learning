"""
Unit tests for the lean model factory.

The factory deliberately does no tuning and no feature selection: every model
sees the same columns, is fitted once on the training split, and is then only
asked to predict.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.lean_models import MODEL_NAMES, fit_predict, make_models

# Small forests / boosters keep the suite fast; the shapes are what matter.
PARAMS = {
    "ridge": {"alpha": 10.0},
    "random_forest": {
        "n_estimators": 20,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
}

COLUMNS = [f"f{i}" for i in range(5)]


def make_frame(n, seed, scale=1.0, offset=0.0):
    """A seeded n x 5 feature frame plus a linearly related target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(offset, scale, (n, len(COLUMNS))),
        columns=COLUMNS,
        index=pd.RangeIndex(n),
    )
    y = 0.01 * X["f0"] - 0.005 * X["f1"] + rng.normal(0.0, 0.001, n)
    return X, y.to_numpy()


@pytest.fixture
def train_data():
    return make_frame(200, seed=0)


# --------------------------------------------------------------------------
# make_models
# --------------------------------------------------------------------------


def test_make_models_returns_the_three_names_in_order():
    assert list(make_models(PARAMS)) == MODEL_NAMES == [
        "ridge",
        "random_forest",
        "xgboost",
    ]


def test_ridge_is_a_pipeline_with_a_standard_scaler():
    ridge = make_models(PARAMS)["ridge"]
    assert isinstance(ridge, Pipeline)
    assert isinstance(ridge.named_steps["scale"], StandardScaler)
    assert ridge.named_steps["model"].alpha == 10.0


def test_model_params_are_passed_through():
    models = make_models(PARAMS)
    assert models["random_forest"].n_estimators == 20
    assert models["random_forest"].max_depth == 4
    assert models["xgboost"].n_estimators == 20
    assert models["xgboost"].learning_rate == pytest.approx(0.03)


def test_random_forest_gets_n_jobs_by_default():
    assert make_models(PARAMS)["random_forest"].n_jobs == -1


def test_explicit_n_jobs_is_not_overridden():
    params = {**PARAMS, "random_forest": {**PARAMS["random_forest"], "n_jobs": 1}}
    assert make_models(params)["random_forest"].n_jobs == 1


@pytest.mark.parametrize("missing", MODEL_NAMES)
def test_missing_model_params_raise_key_error(missing):
    params = {name: dict(cfg) for name, cfg in PARAMS.items() if name != missing}
    with pytest.raises(KeyError, match=missing):
        make_models(params)


def test_every_model_fits_and_predicts(train_data):
    X_train, y_train = train_data
    for model in make_models(PARAMS).values():
        model.fit(X_train, y_train)
        prediction = model.predict(X_train)
        assert prediction.shape == (len(X_train),)


# --------------------------------------------------------------------------
# fit_predict
# --------------------------------------------------------------------------


def test_fit_predict_returns_predictions_for_every_model_and_set(train_data):
    X_train, y_train = train_data
    X_val, _ = make_frame(60, seed=1)
    X_test, _ = make_frame(40, seed=2)

    result = fit_predict(
        make_models(PARAMS), X_train, y_train, {"val": X_val, "test": X_test}
    )

    assert list(result) == MODEL_NAMES
    for name in MODEL_NAMES:
        assert list(result[name]) == ["val", "test"]
        assert result[name]["val"].shape == (60,)
        assert result[name]["test"].shape == (40,)
        assert result[name]["val"].dtype == np.float64


def test_fit_predict_handles_an_empty_set_of_eval_sets(train_data):
    X_train, y_train = train_data
    result = fit_predict(make_models(PARAMS), X_train, y_train, {})
    assert result == {name: {} for name in MODEL_NAMES}


def test_fit_predict_is_deterministic(train_data):
    """Two identical calls agree, so nothing in the eval set changed the fit."""
    X_train, y_train = train_data
    # A deliberately different distribution: if the models were refit on it the
    # predictions would move between runs of the shifted set.
    X_shifted, _ = make_frame(50, seed=3, scale=5.0, offset=2.0)

    first = fit_predict(make_models(PARAMS), X_train, y_train, {"shifted": X_shifted})
    second = fit_predict(make_models(PARAMS), X_train, y_train, {"shifted": X_shifted})

    for name in MODEL_NAMES:
        # The seeded forest is bit-reproducible; the tolerance is prudence
        # about xgboost's threaded reductions. Anything above float noise
        # would mean the fit itself moved.
        np.testing.assert_allclose(
            first[name]["shifted"], second[name]["shifted"], rtol=1e-9, atol=1e-15
        )


def test_fit_predict_predictions_depend_only_on_the_training_data(train_data):
    """Predicting a set alone or alongside another set gives the same answer."""
    X_train, y_train = train_data
    X_val, _ = make_frame(60, seed=1)
    X_other, _ = make_frame(30, seed=4, scale=5.0)

    alone = fit_predict(make_models(PARAMS), X_train, y_train, {"val": X_val})
    together = fit_predict(
        make_models(PARAMS), X_train, y_train, {"val": X_val, "other": X_other}
    )

    for name in MODEL_NAMES:
        np.testing.assert_allclose(alone[name]["val"], together[name]["val"])
