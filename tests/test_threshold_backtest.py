"""
Unit tests for the long/flat threshold backtest.

The interesting properties are all about the cost convention and the
alignment: a position taken at the close of day ``t`` earns ``realized[t]``,
pays one side going in and one side coming out, and never sees a return it
could not have known about.
"""

import numpy as np
import pandas as pd
import pytest

from src import return_metrics, threshold_backtest
from src._validation import as_1d
from src.threshold_backtest import (
    SUMMARY_KEYS,
    backtest_long_flat,
    buy_and_hold,
    cost_drag_arithmetic,
    select_threshold,
)

TABLE_COLUMNS = [
    "threshold",
    "gross_return",
    "net_return",
    "sharpe",
    "max_drawdown",
    "n_trades",
    "days_in_market",
    "eligible",
]

# Predictions alternate low/high; the low days lose money and the high days
# make it, so a mid threshold is the only profitable rule.
PRED_VAL = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
REALIZED_VAL = np.array([-0.03, 0.02, -0.04, 0.01, -0.03, 0.02])


# --------------------------------------------------------------------------
# shape of the result
# --------------------------------------------------------------------------


def test_result_has_exactly_the_documented_keys():
    result = backtest_long_flat(np.zeros(5), np.zeros(5), 0.0, 0.0)
    assert set(result) == set(SUMMARY_KEYS) | {"equity_curve", "daily_net"}


def test_equity_curve_is_one_longer_than_the_return_series():
    rng = np.random.default_rng(42)
    realized = rng.normal(0.0, 0.02, 33)
    result = backtest_long_flat(np.ones(33), realized, 0.0, 0.0005)
    assert result["equity_curve"].shape == (34,)
    assert result["daily_net"].shape == (33,)
    assert result["equity_curve"][0] == 1.0


# --------------------------------------------------------------------------
# never long
# --------------------------------------------------------------------------


def test_never_long_trades_nothing_and_earns_nothing():
    rng = np.random.default_rng(0)
    realized = rng.normal(0.0, 0.02, 50)
    pred = np.full(50, -1.0)

    result = backtest_long_flat(pred, realized, threshold=0.0, cost_per_side=0.0005)

    assert result["gross_return"] == 0.0
    assert result["net_return"] == 0.0
    assert result["n_trades"] == 0
    assert result["n_round_trips"] == 0.0
    assert result["total_cost"] == 0.0
    assert np.isnan(result["sharpe"])
    assert result["days_in_market"] == 0.0
    assert len(result["equity_curve"]) == len(realized) + 1
    assert np.allclose(result["equity_curve"], 1.0)


# --------------------------------------------------------------------------
# always long == buy and hold
# --------------------------------------------------------------------------


def test_always_long_is_exactly_buy_and_hold():
    rng = np.random.default_rng(1)
    realized = rng.normal(0.0005, 0.02, 60)
    cost = 0.0005

    always = backtest_long_flat(np.ones(60), realized, threshold=0.0, cost_per_side=cost)
    held = buy_and_hold(realized, cost_per_side=cost)

    for key in SUMMARY_KEYS:
        assert always[key] == pytest.approx(held[key], nan_ok=True), key
    assert np.allclose(always["equity_curve"], held["equity_curve"])
    assert np.allclose(always["daily_net"], held["daily_net"])


def test_buy_and_hold_pays_one_entry_and_one_exit():
    rng = np.random.default_rng(2)
    realized = rng.normal(0.0005, 0.02, 60)
    cost = 0.0005

    held = buy_and_hold(realized, cost_per_side=cost)

    # Entry charged on day 0, exit charged on the last day, nothing in between.
    expected = (
        (1 + realized[0] - cost)
        * np.prod(1 + realized[1:-1])
        * (1 + realized[-1] - cost)
        - 1
    )
    assert held["n_trades"] == 2
    assert held["n_round_trips"] == 1.0
    assert held["days_in_market"] == 1.0
    assert held["total_cost"] == pytest.approx(2 * cost)
    assert np.isclose(held["net_return"], expected)
    assert np.isclose(held["gross_return"], np.prod(1 + realized) - 1)


# --------------------------------------------------------------------------
# turnover and cost
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [20, 21])
def test_alternating_positions_pay_only_costs(n):
    cost = 0.001
    realized = np.zeros(n)
    pred = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

    result = backtest_long_flat(pred, realized, threshold=0.0, cost_per_side=cost)

    # Recompute the sides traded independently of the implementation.
    position = (pred > 0.0).astype(float)
    prev = np.concatenate([[0.0], position[:-1]])
    sides = np.abs(position - prev)
    sides[-1] += position[-1]  # the final exit, charged on the last day

    assert result["gross_return"] == 0.0
    assert result["n_trades"] == int(sides.sum())
    assert result["n_round_trips"] == result["n_trades"] / 2
    assert result["total_cost"] == pytest.approx(result["n_trades"] * cost)
    assert np.isclose(result["net_return"], np.prod(1 - cost * sides) - 1)
    # ...which is the linear -(n_trades * cost) to first order.
    assert result["net_return"] == pytest.approx(-result["n_trades"] * cost, rel=0.05)
    assert result["cost_drag"] == pytest.approx(-result["net_return"])


def test_cost_drag_arithmetic_is_the_linear_estimate():
    assert cost_drag_arithmetic(0.0015, 250) == 0.375


def test_zero_cost_leaves_the_gross_returns_untouched():
    rng = np.random.default_rng(3)
    pred = rng.normal(0.0, 0.01, 80)
    realized = rng.normal(0.0, 0.02, 80)

    result = backtest_long_flat(pred, realized, threshold=0.0, cost_per_side=0.0)

    assert result["net_return"] == pytest.approx(result["gross_return"])
    assert result["cost_drag"] == pytest.approx(0.0)
    assert result["total_cost"] == 0.0
    assert np.allclose(result["daily_net"], (pred > 0.0) * realized)


# --------------------------------------------------------------------------
# drawdown
# --------------------------------------------------------------------------


def test_max_drawdown_of_a_monotone_rising_curve_is_zero():
    realized = np.full(30, 0.01)
    result = backtest_long_flat(np.ones(30), realized, threshold=0.0, cost_per_side=0.0)
    assert result["max_drawdown"] == 0.0


def test_max_drawdown_of_a_known_sequence_is_the_deepest_dip():
    # 1.0 -> 1.1 -> 0.55 -> 1.1: the worst peak-to-trough is a halving.
    realized = np.array([0.10, -0.50, 1.00])
    result = backtest_long_flat(np.ones(3), realized, threshold=0.0, cost_per_side=0.0)
    assert result["max_drawdown"] == pytest.approx(-0.5)
    assert np.allclose(result["equity_curve"], [1.0, 1.1, 0.55, 1.1])


# --------------------------------------------------------------------------
# alignment: pred[t] decides the position that earns realized[t]
# --------------------------------------------------------------------------


def test_perfect_foresight_is_long_on_exactly_the_up_days():
    rng = np.random.default_rng(5)
    realized = rng.normal(0.0, 0.02, 120)

    result = backtest_long_flat(realized, realized, threshold=0.0, cost_per_side=0.0)

    up = realized > 0.0
    assert result["days_in_market"] == pytest.approx(up.mean())
    assert result["gross_return"] == pytest.approx(np.prod(1 + realized[up]) - 1)
    assert np.allclose(result["daily_net"], np.where(up, realized, 0.0))


def test_positions_depend_only_on_the_predictions():
    rng = np.random.default_rng(11)
    pred = rng.normal(0.0, 0.01, 40)
    first = rng.normal(0.0, 0.02, 40)
    second = rng.normal(0.0, 0.02, 40)

    a = backtest_long_flat(pred, first, threshold=0.0, cost_per_side=0.001)
    b = backtest_long_flat(pred, second, threshold=0.0, cost_per_side=0.001)

    assert a["n_trades"] == b["n_trades"]
    assert a["days_in_market"] == b["days_in_market"]


# --------------------------------------------------------------------------
# threshold selection
# --------------------------------------------------------------------------


def test_select_threshold_picks_the_only_profitable_rule():
    thresholds = [0.0, 1.0, 2.0]

    best, table = select_threshold(
        PRED_VAL, REALIZED_VAL, thresholds, cost_per_side=0.0, min_days_in_market=0
    )

    assert best == 1.0
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == TABLE_COLUMNS
    assert list(table["threshold"]) == thresholds
    assert table.loc[0, "sharpe"] < 0
    assert table.loc[1, "sharpe"] > 0
    assert np.isnan(table.loc[2, "sharpe"])


def test_select_threshold_keeps_the_given_row_order():
    thresholds = [2.0, 0.0, 1.0]

    best, table = select_threshold(
        PRED_VAL, REALIZED_VAL, thresholds, cost_per_side=0.0, min_days_in_market=0
    )

    assert list(table["threshold"]) == thresholds
    assert best == 1.0


def test_select_threshold_never_picks_a_nan_sharpe():
    # 2.0 sits above every prediction, so that rule is never long and its
    # Sharpe is undefined -- it must not win by being "not worse".
    best, table = select_threshold(
        PRED_VAL, REALIZED_VAL, [1.0, 2.0], cost_per_side=0.0, min_days_in_market=0
    )
    assert best == 1.0
    assert int(table.loc[1, "n_trades"]) == 0


def test_select_threshold_breaks_ties_on_the_lowest_threshold():
    # Both thresholds sit below every prediction, so both are always long.
    best, _ = select_threshold(
        PRED_VAL, REALIZED_VAL, [0.2, 0.0], cost_per_side=0.0, min_days_in_market=0
    )
    assert best == 0.0


def test_select_threshold_rejects_an_empty_grid():
    with pytest.raises(ValueError):
        select_threshold(PRED_VAL, REALIZED_VAL, [], cost_per_side=0.0, min_days_in_market=0)


def test_select_threshold_rejects_an_all_nan_grid():
    with pytest.raises(ValueError, match="undefined Sharpe"):
        select_threshold(
            PRED_VAL, REALIZED_VAL, [2.0, 3.0], cost_per_side=0.0, min_days_in_market=0
        )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pred, realized",
    [
        (np.zeros(5), np.zeros(4)),
        (np.zeros(0), np.zeros(0)),
        (np.array([np.nan, 1.0]), np.array([0.01, 0.01])),
        (np.array([1.0, 1.0]), np.array([0.01, np.inf])),
        (np.zeros((2, 2)), np.zeros(2)),
    ],
)
def test_backtest_rejects_bad_inputs(pred, realized):
    with pytest.raises(ValueError):
        backtest_long_flat(pred, realized, threshold=0.0, cost_per_side=0.0)


@pytest.mark.parametrize("realized", [np.zeros(0), np.array([0.01, np.nan])])
def test_buy_and_hold_rejects_bad_inputs(realized):
    with pytest.raises(ValueError):
        buy_and_hold(realized, cost_per_side=0.0)


# --------------------------------------------------------------------------
# participation guard
# --------------------------------------------------------------------------


@pytest.fixture
def lucky_day_sweep():
    """A sweep where the sparsest rule has the best Sharpe on almost no days.

    Fifty "broad" days carry a small genuine edge through real noise. Five
    "lucky" days are clean 5% winners and nothing else. The lucky-only rule
    scores the higher Sharpe, but it is five days of evidence.
    """
    n = 250
    rng = np.random.default_rng(20)
    realized = rng.normal(0.0, 0.025, n)
    pred = np.zeros(n)

    broad = np.arange(0, n, 5)  # 50 days
    pred[broad] = 1.0
    realized[broad] += 0.002

    lucky = np.array([7, 57, 107, 157, 207])  # 5 days, disjoint from broad
    pred[lucky] = 2.0
    realized[lucky] = 0.05

    return pred, realized


def test_sparse_rule_wins_on_raw_sharpe_but_is_not_eligible(lucky_day_sweep):
    pred, realized = lucky_day_sweep

    best, table = select_threshold(
        pred, realized, [0.5, 1.5], cost_per_side=0.0, min_days_in_market=20
    )

    # The premise: the five-day rule really does score better in isolation.
    assert table.loc[1, "sharpe"] > table.loc[0, "sharpe"] > 0
    assert table.loc[1, "days_in_market"] * len(realized) == pytest.approx(5)
    # ...and it is still reported, just barred from winning.
    assert list(table["eligible"]) == [True, False]
    assert best == 0.5


def test_min_days_in_market_zero_restores_the_raw_best(lucky_day_sweep):
    pred, realized = lucky_day_sweep

    best, table = select_threshold(
        pred, realized, [0.5, 1.5], cost_per_side=0.0, min_days_in_market=0
    )

    assert list(table["eligible"]) == [True, True]
    assert best == 1.5


def test_select_threshold_says_when_nothing_is_eligible(lucky_day_sweep):
    pred, realized = lucky_day_sweep

    # The busiest rule trades 55 days, so a 100-day floor leaves nothing.
    with pytest.raises(ValueError, match="at least 100"):
        select_threshold(
            pred, realized, [0.5, 1.5], cost_per_side=0.0, min_days_in_market=100
        )


# --------------------------------------------------------------------------
# an undefined Sharpe stays undefined
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [7, 30, 100])
def test_a_constant_net_series_has_no_sharpe(n):
    # Floating-point noise makes the standard deviation ~1e-18 rather than a
    # clean zero, which used to report a Sharpe around 9e16.
    result = backtest_long_flat(np.ones(n), np.full(n, 0.01), 0.0, 0.0)
    assert np.isnan(result["sharpe"])


# --------------------------------------------------------------------------
# scalar validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cost", [np.nan, np.inf, -0.001])
def test_rejects_a_cost_that_is_not_a_finite_charge(cost):
    with pytest.raises(ValueError):
        backtest_long_flat(np.ones(5), np.zeros(5), threshold=0.0, cost_per_side=cost)
    with pytest.raises(ValueError):
        buy_and_hold(np.zeros(5), cost_per_side=cost)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_rejects_a_non_finite_threshold(threshold):
    with pytest.raises(ValueError):
        backtest_long_flat(np.ones(5), np.zeros(5), threshold, cost_per_side=0.0)


# --------------------------------------------------------------------------
# input shapes
# --------------------------------------------------------------------------


def test_accepts_a_series_and_a_column_vector():
    pred = np.array([1.0, -1.0, 1.0, 1.0])
    realized = np.array([0.01, -0.02, 0.03, 0.005])
    expected = backtest_long_flat(pred, realized, 0.0, 0.0005)

    shapes = [
        (pd.Series(pred), pd.Series(realized)),
        (pred.reshape(-1, 1), realized.reshape(-1, 1)),
    ]
    for other_pred, other_realized in shapes:
        result = backtest_long_flat(other_pred, other_realized, 0.0, 0.0005)
        for key in SUMMARY_KEYS:
            assert result[key] == pytest.approx(expected[key], nan_ok=True), key


# --------------------------------------------------------------------------
# shared validation
# --------------------------------------------------------------------------


def test_both_modules_use_the_one_shared_validator():
    assert threshold_backtest.as_1d is as_1d
    assert return_metrics.as_1d is as_1d
    assert not hasattr(threshold_backtest, "_as_1d")
    assert not hasattr(return_metrics, "_as_1d")


def test_as_1d_allows_an_empty_array_only_when_asked():
    assert as_1d("x", [], allow_empty=True).shape == (0,)
    with pytest.raises(ValueError, match="non-empty"):
        as_1d("x", [])


def test_as_1d_keeps_its_error_messages():
    with pytest.raises(ValueError, match="must be 1-D"):
        as_1d("x", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        as_1d("x", [1.0, np.nan])
