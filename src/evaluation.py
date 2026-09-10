"""
Evaluation metrics for next-day return forecasts.

Three groups of functions:

*Return regression* -- ``return_regression_metrics`` scores log-return forecasts
and reports R2 against a zero forecast alongside the usual R2.

*Direction* -- ``direction_metrics`` attaches a Wilson interval and significance
tests. Comparisons against a rival predictor on the same rows use McNemar's test
(paired), not a one-sample binomial (unpaired).

*Panel dependence* -- ``design_effect``, ``adjust_proportion_for_clustering`` and
``clustered_mcnemar_pvalue`` correct pooled inference for the fact that several
tickers are scored on identical calendar dates. Quoting an iid interval over a
correlated panel understates the uncertainty, sometimes by a factor of 1.6 on the
interval width. The correction applies to the paired test too: the signed
discordance between two predictors is itself cross-ticker correlated.

*Walk-forward* -- ``run_walk_forward`` is the engine that produces every
out-of-sample number in the project.

Deliberately absent: a level-based directional-accuracy helper. A metric that
takes ``np.sign(np.diff(y_true))`` silently returns the wrong answer when handed a
return target, which is the only target this project models. Use
``direction_metrics`` on return signs instead.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Return regression
# ----------------------------------------------------------------------

def return_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Regression metrics for log-return forecasts.

    R2 here is the honest number: the variance of daily returns explained out of
    sample. Values at or slightly below zero mean the forecast is no better than
    predicting the test window's mean, which is what daily equity returns
    normally deliver.

    Args:
        y_true: Realised log returns
        y_pred: Forecast log returns

    Returns:
        Dict with n, RMSE, MAE, R2, and R2 measured against a zero forecast
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    n = len(y_true)
    if n == 0:
        return {'n': 0, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'R2_vs_zero': np.nan}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(y_true ** 2)))

    return {
        'n': n,
        'RMSE': rmse,
        'MAE': float(np.mean(np.abs(y_true - y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        # Skill against the zero forecast rather than against the test-window mean.
        'R2_vs_zero': float(1 - (rmse ** 2) / (zero_rmse ** 2)) if zero_rmse > 0 else np.nan,
    }


# ----------------------------------------------------------------------
# Proportions: intervals and paired tests
# ----------------------------------------------------------------------

def wilson_interval(
    successes: float,
    n: float,
    confidence: float = 0.95
) -> tuple:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly for proportions near 0.5 with a few hundred observations --
    exactly the regime a directional-accuracy claim lives in.

    Accepts non-integer ``successes``/``n`` so an effective sample size from
    ``design_effect`` can be passed straight in.

    Args:
        successes: Number of correct predictions
        n: Number of predictions
        confidence: Coverage (default 0.95)

    Returns:
        (lower, upper)
    """
    from scipy import stats

    if n <= 0:
        return (float('nan'), float('nan'))

    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> Dict[str, float]:
    """
    Exact McNemar test comparing two predictors on the same rows.

    The right test when two predictors are scored on an identical set of
    observations. A one-sample binomial test against the rival's *rate* treats the
    two as independent samples, which they are not -- they agree on most rows, and
    ignoring that pairing mis-states the evidence in either direction depending on
    how correlated the errors are.

    Only discordant pairs carry information: b is where A is right and B is wrong,
    c the reverse. Under the null the count b is Binomial(b + c, 0.5).

    Args:
        y_true: Realised labels
        y_pred_a: First predictor's labels
        y_pred_b: Second predictor's labels (the reference)

    Returns:
        Dict with the discordant counts, the p-value, and each predictor's hits
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=int)
    correct_a = np.asarray(y_pred_a, dtype=int) == y_true
    correct_b = np.asarray(y_pred_b, dtype=int) == y_true

    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right

    pvalue = 1.0 if (b + c) == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)

    return {
        'mcnemar_b': b,
        'mcnemar_c': c,
        'mcnemar_n_discordant': b + c,
        'p_vs_reference': pvalue,
    }


def clustered_mcnemar_pvalue(b: int, c: int, design_effect: float) -> float:
    """
    McNemar p-value corrected for clustered discordant pairs.

    Plain McNemar assumes the discordant pairs are independent. They are not when
    several tickers are scored on the same dates: the signed discordance
    ``D = correct_model - correct_baseline`` is itself cross-ticker correlated, so
    two predictors can disagree on the same day across the whole basket for one
    market-wide reason. Treating those as independent pairs overstates the evidence.

    The correction divides the discordant counts by the design effect of the D
    panel -- the same ``design_effect`` machinery the pooled accuracy intervals use
    -- and runs the exact binomial at that effective size. The split between b and c
    is preserved, so only the strength of the evidence moves, not its direction.

    Args:
        b: Discordant pairs where the model is right and the reference wrong
        c: Discordant pairs where the reference is right and the model wrong
        design_effect: Design effect of the signed-discordance panel

    Returns:
        Two-sided p-value at the effective number of discordant pairs
    """
    from scipy import stats

    total = b + c
    if total == 0:
        return 1.0

    deff = max(1.0, float(design_effect))
    n_eff = int(round(total / deff))
    if n_eff < 1:
        return 1.0

    b_eff = int(round(b / deff))
    b_eff = min(max(b_eff, 0), n_eff)

    return float(stats.binomtest(b_eff, n_eff, 0.5).pvalue)


def direction_metrics(
    y_true_direction: np.ndarray,
    y_pred_direction: np.ndarray,
    reference_rate: float = None,
    reference_prediction: np.ndarray = None,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Directional accuracy with an interval and significance tests.

    A bare accuracy number is not interpretable: with ~1,500 test days the 95%
    interval on a coin flip is roughly +/-2.5 points, so 53% is not distinguishable
    from chance. Two nulls are tested -- 0.5, and the reference predictor, which is
    the always-up baseline the market's upward drift hands over for free.

    The comparison against the reference is paired (McNemar) when
    ``reference_prediction`` is supplied, because both predictors are scored on the
    same rows.

    Args:
        y_true_direction: Realised direction, 1 for up, 0 otherwise
        y_pred_direction: Predicted direction, 1 for up, 0 otherwise
        reference_rate: The reference predictor's accuracy, for reporting
        reference_prediction: The reference predictor's labels on the same rows,
            enabling McNemar's test
        confidence: Interval coverage

    Returns:
        Dict with accuracy, interval, p-values, precision/recall of 'up'
    """
    from scipy import stats

    y_true = np.asarray(y_true_direction, dtype=int)
    y_pred = np.asarray(y_pred_direction, dtype=int)

    n = len(y_true)
    if n == 0:
        return {'n': 0}

    correct = int(np.sum(y_true == y_pred))
    accuracy = correct / n
    lower, upper = wilson_interval(correct, n, confidence)

    true_up, pred_up = y_true == 1, y_pred == 1
    tp = int(np.sum(true_up & pred_up))
    precision = tp / int(pred_up.sum()) if pred_up.any() else float('nan')
    recall = tp / int(true_up.sum()) if true_up.any() else float('nan')

    metrics = {
        'n': n,
        'accuracy': accuracy,
        'ci_lower': lower,
        'ci_upper': upper,
        'p_vs_0.5': float(stats.binomtest(correct, n, 0.5).pvalue),
        'precision_up': precision,
        'recall_up': recall,
        'pred_up_rate': float(pred_up.mean()),
        'actual_up_rate': float(true_up.mean()),
    }

    if reference_rate is not None:
        metrics['reference_rate'] = float(reference_rate)

    if reference_prediction is not None:
        metrics.update(mcnemar_test(y_true, y_pred, reference_prediction))

    return metrics


# ----------------------------------------------------------------------
# Panel dependence
# ----------------------------------------------------------------------

def design_effect(panel: pd.DataFrame) -> Dict[str, float]:
    """
    Design effect for a mean taken over a panel clustered by date.

    Pooling several tickers scored on identical calendar dates does not multiply
    the information by the number of tickers. Market-wide moves make the tickers'
    outcomes correlate on any given day -- and when the basket contains an index
    alongside its own constituents, the overlap is mechanical, not incidental.

    With m series and mean pairwise correlation rho, the variance of the pooled
    mean is inflated by ``deff = 1 + (m - 1) * rho``, so the effective sample size
    is ``n_total / deff``. Intervals computed from n_total rather than n_effective
    are too narrow by ``sqrt(deff)``.

    Args:
        panel: DataFrame indexed by date, one column per ticker, holding the
            quantity being averaged (e.g. a 0/1 correctness indicator)

    Returns:
        Dict with m, rho, deff, n_total and n_effective
    """
    aligned = panel.dropna(how='any')
    m = aligned.shape[1]
    n_total = int(aligned.size)

    if m < 2 or len(aligned) < 2:
        return {
            'n_series': m, 'mean_pairwise_corr': 0.0, 'design_effect': 1.0,
            'n_total': n_total, 'n_effective': float(n_total),
        }

    correlation = aligned.corr().to_numpy()
    off_diagonal = correlation[~np.eye(m, dtype=bool)]
    rho = float(np.nanmean(off_diagonal))

    deff = max(1.0, 1 + (m - 1) * rho)

    return {
        'n_series': m,
        'mean_pairwise_corr': rho,
        'design_effect': deff,
        'n_total': n_total,
        'n_effective': float(n_total / deff),
    }


def adjust_proportion_for_clustering(
    accuracy: float,
    n_effective: float,
    reference_rate: float = None,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Recompute a proportion's interval and p-value at an effective sample size.

    The point estimate is unchanged -- clustering does not bias the mean, it only
    inflates its variance. Only the interval and the p-values move.

    Args:
        accuracy: The observed proportion
        n_effective: Effective sample size from ``design_effect``
        reference_rate: Optional second null hypothesis
        confidence: Interval coverage

    Returns:
        Dict of design-effect-adjusted interval and p-values
    """
    from scipy import stats

    # The interval and the p-values are computed at the SAME rounded effective
    # size, so a reader cannot find a CI and a p-value that disagree about how
    # much data they were derived from.
    n_round = int(round(n_effective))
    successes = int(round(accuracy * n_round))
    lower, upper = wilson_interval(successes, n_round, confidence)

    adjusted = {
        'n_effective': float(n_effective),
        'adj_ci_lower': lower,
        'adj_ci_upper': upper,
        'adj_p_vs_0.5': float(stats.binomtest(successes, n_round, 0.5).pvalue)
        if n_round > 0 else np.nan,
    }

    if reference_rate is not None and 0 < reference_rate < 1:
        adjusted['adj_p_vs_reference_rate'] = float(
            stats.binomtest(successes, n_round, reference_rate).pvalue
        ) if n_round > 0 else np.nan

    return adjusted


# ----------------------------------------------------------------------
# Walk-forward engine
# ----------------------------------------------------------------------

def run_walk_forward(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    estimator_factory,
    folds: List,
    task: str = 'regression'
) -> pd.DataFrame:
    """
    Run a walk-forward evaluation and collect out-of-sample predictions.

    This is the only path by which the project produces a test-set number. For
    each fold a *fresh* estimator is built by ``estimator_factory`` and fitted on
    that fold's training rows alone; imputation, winsorisation, scaling and
    feature selection all live inside that estimator, so no fitted statistic ever
    sees a test row. Folds run forward in time with an embargo and are never
    shuffled.

    Args:
        dataset: Chronologically ordered features + targets
        feature_cols: Model matrix columns
        target_col: Target column to fit
        estimator_factory: Zero-argument callable returning an unfitted estimator
        folds: Folds from utils.walk_forward_folds
        task: 'regression' or 'classification'

    Returns:
        DataFrame indexed by date with columns: fold, y_true, y_pred, and for
        classification also y_proba; plus train_up_rate per fold
    """
    frames = []

    for fold in folds:
        train = dataset.iloc[fold.train_start:fold.train_end]
        test = dataset.iloc[fold.test_start:fold.test_end]

        X_train, y_train = train[feature_cols], train[target_col]
        X_test = test[feature_cols]

        estimator = estimator_factory()
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        record = pd.DataFrame(
            {
                'fold': fold.index,
                'y_true': test[target_col].to_numpy(),
                'y_pred': np.asarray(y_pred).ravel(),
                'train_up_rate': float((train['target_direction'] > 0).mean())
                if 'target_direction' in train.columns else np.nan,
            },
            index=test.index,
        )

        if task == 'classification' and hasattr(estimator, 'predict_proba'):
            proba = estimator.predict_proba(X_test)
            record['y_proba'] = proba[:, 1] if proba.ndim == 2 else np.asarray(proba).ravel()

        frames.append(record)

    result = pd.concat(frames)
    logger.info(
        "Walk-forward complete: %d folds, %d out-of-sample rows", len(folds), len(result)
    )
    return result
