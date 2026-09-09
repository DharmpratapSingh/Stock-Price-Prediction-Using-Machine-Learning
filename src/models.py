"""
Models and baselines for next-day return prediction.

Three model families, each in a regression flavour (predict the next-day log
return) and a classification flavour (predict its sign):

    linear    Ridge / LogisticRegression
    forest    RandomForestRegressor / RandomForestClassifier
    boosting  XGBRegressor / XGBClassifier

Hyperparameters are fixed, documented defaults (see DEFAULT_PARAMS). Nothing is
tuned against a test fold. The settings are deliberately conservative -- shallow
trees, strong regularisation, large leaves -- because the signal-to-noise ratio in
daily returns is low enough that an unconstrained learner memorises the training
window and reports it as skill.

Every model is used through ``build_pipeline``, which puts imputation, optional
winsorisation, scaling and feature selection *inside* an sklearn Pipeline. The
pipeline is fitted on a training window only, so no test row ever contributes to
a fitted statistic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

MODEL_FAMILIES = ['linear', 'forest', 'boosting']

# Fixed hyperparameters. Chosen once, on priors about daily-return noise, and
# never adjusted against a test fold.
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    'linear': {
        # Ridge over ~50 collinear technical features; alpha=10 shrinks hard.
        'regressor': {'alpha': 10.0, 'random_state': RANDOM_SEED},
        'classifier': {'C': 0.1, 'max_iter': 2000, 'random_state': RANDOM_SEED},
    },
    'forest': {
        'regressor': {
            'n_estimators': 300, 'max_depth': 5, 'min_samples_leaf': 50,
            'max_features': 'sqrt', 'random_state': RANDOM_SEED, 'n_jobs': -1,
        },
        'classifier': {
            'n_estimators': 300, 'max_depth': 5, 'min_samples_leaf': 50,
            'max_features': 'sqrt', 'random_state': RANDOM_SEED, 'n_jobs': -1,
        },
    },
    'boosting': {
        'regressor': {
            'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 5.0,
            'min_child_weight': 20, 'random_state': RANDOM_SEED, 'n_jobs': -1,
            'verbosity': 0,
        },
        'classifier': {
            'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 5.0,
            'min_child_weight': 20, 'random_state': RANDOM_SEED, 'n_jobs': -1,
            'verbosity': 0,
        },
    },
}


class WindowedWinsorizer(BaseEstimator, TransformerMixin):
    """
    Clip features to quantiles learned from the training window only.

    This replaces the old full-sample outlier clipping in the data loader. The
    difference is the whole point: quantiles are fitted in ``fit`` (training rows)
    and merely applied in ``transform`` (test rows), so test-period statistics
    never reach the training data, and prices are never rebuilt from clipped
    returns.
    """

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


class ZeroReturnRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline: always forecast a zero return.

    The reference every return model must beat. Because R2 is measured against
    the mean of the test window, a model with R2 < 0 is doing worse than this.
    """

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X))


class TrainMeanRegressor(BaseEstimator, RegressorMixin):
    """Baseline: forecast the mean return of the training window (drift only)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class PersistenceReturnRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline: forecast today's return for tomorrow.

    The return-space analogue of the persistence forecast that makes level-R2
    look like 0.99. Here it is worth roughly nothing, which is the lesson.
    """

    def __init__(self, feature: str = 'log_return_1d'):
        self.feature = feature

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if self.feature not in X.columns:
                raise ValueError(
                    f"PersistenceReturnRegressor needs the {self.feature!r} column"
                )
            self.feature_index_ = X.columns.get_loc(self.feature)
        else:
            self.feature_index_ = 0
        return self

    def predict(self, X):
        values = X[self.feature].to_numpy() if isinstance(X, pd.DataFrame) \
            else np.asarray(X)[:, self.feature_index_]
        return np.asarray(values, dtype=float)


class AlwaysUpClassifier(BaseEstimator):
    """
    Baseline: always predict 'up'.

    Equities drift upward, so this scores the training-window up-frequency --
    typically 52-54%. Beating 50% is not evidence of skill; beating this is the
    weaker claim a direction model actually has to support.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def get_estimator(family: str, task: str = 'regression', params: Optional[Dict] = None):
    """
    Build a bare estimator for one model family.

    Args:
        family: 'linear', 'forest' or 'boosting'
        task: 'regression' (next-day log return) or 'classification' (its sign)
        params: Overrides merged onto DEFAULT_PARAMS

    Returns:
        An unfitted sklearn-compatible estimator
    """
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family!r}. Use one of {MODEL_FAMILIES}.")
    if task not in ('regression', 'classification'):
        raise ValueError(f"Unknown task: {task!r}")

    key = 'regressor' if task == 'regression' else 'classifier'
    settings = {**DEFAULT_PARAMS[family][key], **(params or {})}

    if family == 'linear':
        return Ridge(**settings) if task == 'regression' else LogisticRegression(**settings)
    if family == 'forest':
        cls = RandomForestRegressor if task == 'regression' else RandomForestClassifier
        return cls(**settings)
    cls = XGBRegressor if task == 'regression' else XGBClassifier
    return cls(**settings)


def build_pipeline(
    family: str,
    task: str = 'regression',
    k_features: Optional[int] = 40,
    winsorize: bool = True,
    params: Optional[Dict] = None
) -> Pipeline:
    """
    Assemble the full fit-in-window pipeline for a model family.

    Steps, all fitted on training rows only:
        impute   median imputation of any residual NaN
        clip     winsorise to training-window quantiles (optional)
        scale    standardise
        select   univariate top-k feature selection (optional)
        model    the estimator

    Feature selection lives inside the pipeline on purpose. Selecting features on
    the full dataset before splitting -- the original bug -- leaks the test period
    into the choice of which columns survive.

    Args:
        family: 'linear', 'forest' or 'boosting'
        task: 'regression' or 'classification'
        k_features: Keep the top k features, or None to keep all
        winsorize: Apply train-window winsorisation
        params: Estimator parameter overrides

    Returns:
        An unfitted sklearn Pipeline
    """
    steps: List = [('impute', SimpleImputer(strategy='median'))]

    if winsorize:
        steps.append(('clip', WindowedWinsorizer()))

    steps.append(('scale', StandardScaler()))

    if k_features is not None:
        score_func = f_regression if task == 'regression' else f_classif
        steps.append(('select', SelectKBest(score_func=score_func, k=k_features)))

    steps.append(('model', get_estimator(family, task, params)))

    return Pipeline(steps)


def get_baseline(name: str, task: str = 'regression'):
    """
    Build a baseline predictor by name.

    Args:
        name: 'zero', 'train_mean', 'persistence' (regression) or
            'always_up' (classification)
        task: 'regression' or 'classification'

    Returns:
        An unfitted baseline estimator
    """
    regression_baselines = {
        'zero': ZeroReturnRegressor,
        'train_mean': TrainMeanRegressor,
        'persistence': PersistenceReturnRegressor,
    }
    classification_baselines = {'always_up': AlwaysUpClassifier}

    table = regression_baselines if task == 'regression' else classification_baselines
    if name not in table:
        raise ValueError(
            f"Unknown {task} baseline: {name!r}. Available: {sorted(table)}"
        )
    return table[name]()


def get_feature_importance(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from a fitted pipeline, mapped back to names.

    Handles the SelectKBest step so importances line up with the original column
    names rather than the post-selection positions.

    Args:
        pipeline: A fitted pipeline from build_pipeline
        feature_names: The pipeline's input column names, in order

    Returns:
        DataFrame of feature/importance sorted descending, or empty if the
        estimator exposes neither importances nor coefficients
    """
    names = list(feature_names)
    if 'select' in pipeline.named_steps:
        support = pipeline.named_steps['select'].get_support()
        names = [n for n, keep in zip(names, support) if keep]

    model = pipeline.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, 'coef_'):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        return pd.DataFrame(columns=['feature', 'importance'])

    return (
        pd.DataFrame({'feature': names, 'importance': values})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )
