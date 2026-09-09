"""
Unit tests for the backtesting framework.

Besides the happy paths, these pin the cost and accounting conventions that make
the ML strategy and the benchmark comparable: both may trade on bar 0, both
liquidate on the final bar, both are scored over the same number of bars, and each
reports a single total return net of exit costs.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import Backtester, print_backtest_results


@pytest.fixture
def sample_prices():
    """Trending price series with a fixed seed."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
    prices = 100 + np.cumsum(rng.normal(0.1, 2, 100))
    return dates, prices


@pytest.fixture
def sample_predictions(sample_prices):
    """Forecast of tomorrow's price, close to today's spot."""
    dates, prices = sample_prices
    rng = np.random.default_rng(1)
    predictions = prices * (1 + rng.normal(0, 0.01, len(prices)))
    return dates, prices, predictions


def test_backtester_init():
    backtester = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    assert backtester.initial_capital == 100000
    assert backtester.commission == 0.001
    assert backtester.slippage == 0.0005
    assert backtester.cost_per_side == pytest.approx(0.0015)


def test_simple_strategy(sample_predictions):
    dates, actuals, predictions = sample_predictions
    backtester = Backtester(initial_capital=100000)

    results = backtester.simple_strategy(predictions, actuals, dates, threshold=0.01)

    assert {"equity_curve", "final_equity", "total_return", "metrics"} <= set(results)
    # n+1 equity-curve contract: one seed point plus one point per bar.
    assert len(results["equity_curve"]) == len(actuals) + 1
    assert len(results["positions"]) == len(actuals) + 1


def test_buy_and_hold_strategy(sample_prices):
    dates, prices = sample_prices
    backtester = Backtester(initial_capital=100000)

    results = backtester.buy_and_hold_strategy(prices, dates)

    assert {"equity_curve", "final_equity", "total_return"} <= set(results)
    assert len(results["equity_curve"]) == len(prices) + 1


def test_strategy_with_a_clear_signal_actually_trades():
    """
    A persistent +5% forecast must open a position at any price level.

    Regression test: sizing on price alone made cost-inclusive total_cost exceed
    capital at position_size=1.0, so the affordability guard rejected every trade
    and the backtest was a flat line that looked like a legitimate "no signal"
    result.
    """
    backtester = Backtester(initial_capital=100000)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")

    for price_level in (12.5, 50.0, 100.0, 250.0, 1000.0):
        prices = np.full(50, price_level)
        results = backtester.simple_strategy(prices * 1.05, prices, dates, threshold=0.01)
        assert len(results["trades"]) > 0, f"no trade opened at price {price_level}"
        assert results["trades"][0]["action"] == "BUY"


def test_trade_cost_never_exceeds_available_capital():
    backtester = Backtester(initial_capital=10_000, commission=0.01, slippage=0.01)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = np.full(10, 97.3)

    results = backtester.simple_strategy(prices * 1.1, prices, dates, threshold=0.01)
    buy = results["trades"][0]
    assert buy["value"] + buy["commission"] + buy["slippage"] <= 10_000 + 1e-9


def test_first_bar_is_not_a_free_bar(sample_predictions):
    """
    The strategy must be able to act on bar 0, like the benchmark.

    Previously the signal came from differencing consecutive predictions, so bar 0
    was structurally forced to no-trade. The benchmark paid to enter on bar 0 while
    the strategy sat out for free, biasing the comparison in the strategy's favour
    by roughly 1/n.
    """
    dates, actuals, _ = sample_predictions
    backtester = Backtester(initial_capital=100000)

    # An unambiguous buy signal on every bar, including the first.
    results = backtester.simple_strategy(actuals * 1.05, actuals, dates, threshold=0.01)
    assert results["equity_curve"][1] != results["equity_curve"][0]
    assert results["trades"][0]["date"] == dates[0]


def test_both_strategies_are_scored_over_the_same_bars(sample_predictions):
    dates, actuals, predictions = sample_predictions
    backtester = Backtester(initial_capital=100000)

    strategy = backtester.simple_strategy(predictions, actuals, dates, threshold=0.01)
    benchmark = backtester.buy_and_hold_strategy(actuals, dates)

    assert strategy["metrics"]["n_periods"] == benchmark["metrics"]["n_periods"]
    assert len(strategy["equity_curve"]) == len(benchmark["equity_curve"])


def test_compare_strategies_rejects_mismatched_horizons(sample_predictions):
    """Guard so a future change cannot quietly compare unequal windows."""
    dates, actuals, predictions = sample_predictions
    backtester = Backtester(initial_capital=100000)

    strategy = backtester.simple_strategy(predictions, actuals, dates, threshold=0.01)
    shorter = backtester.buy_and_hold_strategy(actuals[:50], dates[:50])

    with pytest.raises(ValueError, match="different numbers of bars"):
        backtester.compare_strategies({"ML Strategy": strategy}, shorter)


@pytest.mark.parametrize("strategy_name", ["simple", "buy_and_hold"])
def test_one_total_return_net_of_exit_costs(sample_predictions, strategy_name):
    """
    results['total_return'] and metrics['total_return'] must be the same number.

    buy_and_hold used to net exit costs in one and not the other, so
    print_backtest_results and compare_strategies disagreed about the same run.
    """
    dates, actuals, predictions = sample_predictions
    backtester = Backtester(initial_capital=100000)

    if strategy_name == "simple":
        results = backtester.simple_strategy(actuals * 1.05, actuals, dates, 0.01)
    else:
        results = backtester.buy_and_hold_strategy(actuals, dates)

    assert results["total_return"] == pytest.approx(results["metrics"]["total_return"])
    # The final curve point is the post-liquidation value.
    assert results["equity_curve"][-1] == pytest.approx(results["final_equity"])


def test_buy_and_hold_pays_both_entry_and_exit_costs(sample_prices):
    dates, prices = sample_prices

    free = Backtester(initial_capital=100000, commission=0.0, slippage=0.0)
    costly = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)

    assert (
        costly.buy_and_hold_strategy(prices, dates)["final_equity"]
        < free.buy_and_hold_strategy(prices, dates)["final_equity"]
    )
    assert len(costly.buy_and_hold_strategy(prices, dates)["trades"]) == 2


def test_annualization_uses_bars_not_the_seed_point():
    """n+1 curve points describe n periods; counting the seed skews annualisation."""
    backtester = Backtester(initial_capital=100.0)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    actual = np.log1p(np.full(252, 0.001))

    run = backtester.long_flat_backtest(
        np.ones(252), actual, dates, cost_per_side=0.0
    )
    assert run["metrics"]["n_periods"] == 252
    expected = (1.001 ** 252 - 1) * 100
    assert run["metrics"]["annualized_return"] == pytest.approx(expected, rel=1e-6)


def test_backtest_metrics(sample_predictions):
    dates, actuals, predictions = sample_predictions
    backtester = Backtester(initial_capital=100000)

    metrics = backtester.simple_strategy(predictions, actuals, dates, 0.01)["metrics"]

    for key in ("total_return", "sharpe_ratio", "max_drawdown", "win_rate", "n_periods"):
        assert key in metrics


def test_compare_strategies(sample_predictions):
    dates, actuals, predictions = sample_predictions
    backtester = Backtester(initial_capital=100000)

    strategy = backtester.simple_strategy(predictions, actuals, dates)
    benchmark = backtester.buy_and_hold_strategy(actuals, dates)
    comparison = backtester.compare_strategies({"ML Strategy": strategy}, benchmark)

    assert len(comparison) >= 2
    assert "Buy & Hold" in comparison.index


def test_mismatched_input_lengths_are_rejected():
    backtester = Backtester()
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    with pytest.raises(ValueError, match="same length"):
        backtester.simple_strategy(np.ones(5), np.ones(4), dates)
    with pytest.raises(ValueError, match="same length"):
        backtester.long_flat_backtest(np.ones(5), np.ones(4), dates)


def test_print_backtest_results_runs(sample_predictions, capsys):
    dates, actuals, predictions = sample_predictions
    results = Backtester().simple_strategy(predictions, actuals, dates)
    print_backtest_results(results)
    assert "BACKTEST RESULTS" in capsys.readouterr().out
