"""
The model zoo for the lean pipeline: build, fit once, predict.

Deliberately boring. There is no hyperparameter search and no feature
selection here -- both would need the evaluation years to choose anything, and
that is exactly the leak this pipeline exists to avoid. Every model is fitted
on the training split alone and then only ever asked to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["ridge", "random_forest", "xgboost"]


def make_models(params: dict) -> dict[str, object]:
    """Construct the three regressors from a config block.

    Args:
        params: Mapping with a ``ridge``, ``random_forest`` and ``xgboost``
            entry, each a dict of constructor keyword arguments.

    Returns:
        Estimators keyed by :data:`MODEL_NAMES`, in that order.

    Raises:
        KeyError: If any of the three model keys is missing.
    """
    missing = [name for name in MODEL_NAMES if name not in params]
    if missing:
        raise KeyError(
            f"Missing model parameters for: {', '.join(missing)}. "
            f"make_models requires an entry for each of {MODEL_NAMES}."
        )

    # Ridge is scale-sensitive; the trees are not, so only it gets a scaler,
    # and the scaler lives inside the pipeline so it is fitted on train only.
    ridge = Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(**params["ridge"]))]
    )

    forest_params = dict(params["random_forest"])
    forest_params.setdefault("n_jobs", -1)

    return {
        "ridge": ridge,
        "random_forest": RandomForestRegressor(**forest_params),
        "xgboost": xgb.XGBRegressor(**params["xgboost"]),
    }


def fit_predict(
    models: dict,
    X_train: pd.DataFrame,
    y_train,
    X_eval_sets: dict[str, pd.DataFrame],
) -> dict[str, dict[str, np.ndarray]]:
    """Fit every model on the training split, then predict each evaluation set.

    Args:
        models: Estimators keyed by name, as returned by :func:`make_models`.
        X_train: Training features.
        y_train: Training targets (next-day returns).
        X_eval_sets: Feature frames to predict, keyed by split name.

    Returns:
        ``result[model_name][set_name]`` -> float predictions for that split.
    """
    y = np.asarray(y_train, dtype=float).ravel()

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(X_train, y)
        predictions[name] = {
            set_name: np.asarray(model.predict(X_eval), dtype=float).ravel()
            for set_name, X_eval in X_eval_sets.items()
        }

    return predictions
