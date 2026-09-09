"""
Unit tests for evaluation metrics.

Covers return-space regression scoring, directional accuracy with intervals and
paired significance tests, and the panel-dependence correction applied to pooled
results.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    adjust_proportion_for_clustering,
    design_effect,
    direction_metrics,
    mcnemar_test,
    return_regression_metrics,
    wilson_interval,
)


# ----------------------------------------------------------------------
# Return-space metrics
# ----------------------------------------------------------------------

def test_return_metrics_perfect_forecast():
    y = np.array([0.01, -0.02, 0.003, -0.001])
    metrics = return_regression_metrics(y, y)

    assert metrics['RMSE'] == pytest.approx(0.0)
    assert metrics['R2'] == pytest.approx(1.0)
    assert metrics['R2_vs_zero'] == pytest.approx(1.0)
    assert metrics['n'] == 4


def test_zero_forecast_scores_zero_against_the_zero_baseline():
    """R2_vs_zero is defined so the zero forecast scores exactly 0."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.02, 500)
    metrics = return_regression_metrics(y, np.zeros(500))
    assert metrics['R2_vs_zero'] == pytest.approx(0.0, abs=1e-12)


def test_return_metrics_penalise_a_worse_than_useless_forecast():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.02, 500)
    metrics = return_regression_metrics(y, -y)  # sign-flipped forecast
    assert metrics['R2'] < 0
    assert metrics['R2_vs_zero'] < 0


def test_return_metrics_ignore_nan_pairs():
    y = np.array([0.01, np.nan, 0.02])
    metrics = return_regression_metrics(y, np.array([0.01, 0.5, 0.02]))
    assert metrics['n'] == 2
    assert metrics['RMSE'] == pytest.approx(0.0)


def test_return_metrics_handle_empty_input():
    assert return_regression_metrics(np.array([]), np.array([]))['n'] == 0


# ----------------------------------------------------------------------
# Wilson intervals
# ----------------------------------------------------------------------

def test_wilson_interval_brackets_the_estimate():
    lower, upper = wilson_interval(55, 100)
    assert lower < 0.55 < upper
    assert 0.0 <= lower and upper <= 1.0


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(55, 100)
    large = wilson_interval(5500, 10000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_stays_in_bounds_at_the_extremes():
    assert wilson_interval(0, 50)[0] >= 0.0
    assert wilson_interval(50, 50)[1] <= 1.0


def test_wilson_interval_accepts_a_fractional_effective_n():
    """An effective sample size from design_effect is not an integer."""
    lower, upper = wilson_interval(1602.4, 3004.7)
    assert lower < 0.533 < upper


# ----------------------------------------------------------------------
# Directional accuracy
# ----------------------------------------------------------------------

def test_direction_metrics_on_a_perfect_classifier():
    truth = np.array([1, 0, 1, 1, 0, 1])
    metrics = direction_metrics(truth, truth)

    assert metrics['accuracy'] == pytest.approx(1.0)
    assert metrics['precision_up'] == pytest.approx(1.0)
    assert metrics['recall_up'] == pytest.approx(1.0)


def test_coin_flip_is_not_significant():
    """A near-50% result over 500 days must not read as skill."""
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 2, 500)
    guess = rng.integers(0, 2, 500)

    metrics = direction_metrics(truth, guess)
    assert metrics['p_vs_0.5'] > 0.05
    assert metrics['ci_lower'] < 0.5 < metrics['ci_upper']


def test_direction_metrics_reports_prediction_mix():
    truth = np.array([1, 1, 0, 0])
    predicted = np.array([1, 1, 1, 0])
    metrics = direction_metrics(truth, predicted)

    assert metrics['pred_up_rate'] == pytest.approx(0.75)
    assert metrics['actual_up_rate'] == pytest.approx(0.5)
    assert metrics['precision_up'] == pytest.approx(2 / 3)


def test_direction_metrics_handles_empty_input():
    assert direction_metrics(np.array([]), np.array([]))['n'] == 0


def test_direction_metrics_confidence_level_widens_the_interval():
    truth = np.array([1, 0] * 250)
    guess = np.array([1] * 500)
    narrow = direction_metrics(truth, guess, confidence=0.80)
    wide = direction_metrics(truth, guess, confidence=0.99)
    assert (wide['ci_upper'] - wide['ci_lower']) > (narrow['ci_upper'] - narrow['ci_lower'])


# ----------------------------------------------------------------------
# McNemar: paired comparison against a reference predictor
# ----------------------------------------------------------------------

def test_mcnemar_counts_only_discordant_pairs():
    truth = np.array([1, 1, 1, 1, 0, 0])
    #            A:   right on the 0s, B always says up
    pred_a = np.array([1, 1, 1, 1, 0, 0])
    pred_b = np.array([1, 1, 1, 1, 1, 1])

    result = mcnemar_test(truth, pred_a, pred_b)
    assert result['mcnemar_b'] == 2   # A right, B wrong
    assert result['mcnemar_c'] == 0   # never the reverse
    assert result['mcnemar_n_discordant'] == 2


def test_mcnemar_returns_unity_when_predictors_agree_everywhere():
    truth = np.array([1, 0, 1, 0])
    same = np.array([1, 1, 1, 1])
    result = mcnemar_test(truth, same, same)
    assert result['mcnemar_n_discordant'] == 0
    assert result['p_vs_reference'] == pytest.approx(1.0)


def test_mcnemar_detects_a_consistently_better_predictor():
    truth = np.array([1, 0] * 100)
    perfect = truth.copy()
    always_up = np.ones(200, dtype=int)

    result = mcnemar_test(truth, perfect, always_up)
    assert result['mcnemar_b'] == 100
    assert result['mcnemar_c'] == 0
    assert result['p_vs_reference'] < 1e-20


def test_direction_metrics_uses_mcnemar_when_given_a_reference():
    """
    An always-up model compared with the always-up baseline is indistinguishable.

    The unpaired binomial this replaced could not see that the two predictors make
    identical calls on every row.
    """
    rng = np.random.default_rng(6)
    truth = (rng.random(2000) < 0.54).astype(int)
    always_up = np.ones(2000, dtype=int)

    metrics = direction_metrics(
        truth, always_up, reference_rate=0.54, reference_prediction=always_up
    )
    assert metrics['p_vs_0.5'] < 0.01          # beats a coin flip
    assert metrics['mcnemar_n_discordant'] == 0
    assert metrics['p_vs_reference'] == pytest.approx(1.0)


def test_direction_metrics_omits_mcnemar_without_a_reference():
    truth = np.array([1, 0, 1, 0])
    metrics = direction_metrics(truth, np.ones(4, dtype=int))
    assert 'p_vs_reference' not in metrics


# ----------------------------------------------------------------------
# Panel dependence
# ----------------------------------------------------------------------

def _panel(columns: dict) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.date_range("2020-01-01", periods=len(
        next(iter(columns.values()))), freq="B"))


def test_design_effect_is_one_for_a_single_series():
    effect = design_effect(_panel({"A": np.r_[np.ones(50), np.zeros(50)]}))
    assert effect['design_effect'] == pytest.approx(1.0)
    assert effect['n_effective'] == pytest.approx(100.0)


def test_design_effect_is_one_for_independent_series():
    rng = np.random.default_rng(2)
    panel = _panel({name: rng.integers(0, 2, 2000).astype(float)
                    for name in ("A", "B", "C")})
    effect = design_effect(panel)
    assert effect['mean_pairwise_corr'] == pytest.approx(0.0, abs=0.05)
    assert effect['design_effect'] == pytest.approx(1.0, abs=0.15)


def test_design_effect_grows_with_correlation():
    """Perfectly correlated series carry the information of one series."""
    column = np.r_[np.ones(500), np.zeros(500)]
    panel = _panel({"A": column, "B": column.copy(), "C": column.copy()})
    effect = design_effect(panel)

    assert effect['mean_pairwise_corr'] == pytest.approx(1.0)
    assert effect['design_effect'] == pytest.approx(3.0)
    assert effect['n_effective'] == pytest.approx(effect['n_total'] / 3)


def test_design_effect_matches_the_closed_form():
    rng = np.random.default_rng(8)
    shared = rng.normal(size=1000)
    panel = _panel({
        name: shared + rng.normal(scale=1.0, size=1000) for name in ("A", "B", "C", "D", "E")
    })
    effect = design_effect(panel)
    expected = 1 + (5 - 1) * effect['mean_pairwise_corr']
    assert effect['design_effect'] == pytest.approx(expected)
    assert effect['n_effective'] == pytest.approx(effect['n_total'] / expected)


def test_clustering_adjustment_widens_the_interval_and_weakens_the_pvalue():
    """
    The point estimate is unchanged; only the uncertainty moves.

    This is the correction the pooled results need: five tickers scored on the
    same calendar dates do not supply five times the information.
    """
    accuracy, n_total, deff = 0.5335, 7875, 2.62
    iid = direction_metrics(
        np.r_[np.ones(int(accuracy * n_total)), np.zeros(n_total - int(accuracy * n_total))
              ].astype(int),
        np.ones(n_total, dtype=int),
    )
    adjusted = adjust_proportion_for_clustering(accuracy, n_total / deff)

    assert adjusted['n_effective'] == pytest.approx(n_total / deff)
    # Wider interval, larger p-value, same point estimate.
    assert (adjusted['adj_ci_upper'] - adjusted['adj_ci_lower']) > (
        iid['ci_upper'] - iid['ci_lower']
    )
    assert adjusted['adj_p_vs_0.5'] > iid['p_vs_0.5']


def test_clustering_adjustment_is_a_noop_at_design_effect_one():
    adjusted = adjust_proportion_for_clustering(0.53, 1000)
    iid_lower, iid_upper = wilson_interval(530, 1000)
    assert adjusted['adj_ci_lower'] == pytest.approx(iid_lower, abs=1e-9)
    assert adjusted['adj_ci_upper'] == pytest.approx(iid_upper, abs=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
