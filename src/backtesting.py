"""
Cost-aware backtesting for stock prediction models.

Two entry points:

``Backtester.long_flat_backtest``
    The return-space backtest the walk-forward pipeline uses. Positions are
    long/flat, decided from a forecast log return, and charged commission plus
    slippage on every position change.

``Backtester.simple_strategy`` / ``Backtester.buy_and_hold_strategy``
    The share-based simulation, kept for the price-level path.

Conventions, applied identically to every strategy so results are comparable:

*Equity-curve contract.* Every strategy returns an equity curve of length
``n + 1`` for ``n`` bars. ``equity_curve[0]`` is the initial capital before any
trade; ``equity_curve[i]`` for ``i >= 1`` is the mark-to-market equity after bar
``i - 1`` has been acted on. So ``np.diff(equity_curve)`` gives exactly ``n``
period returns.

*Symmetric costs.* Every strategy may transact on bar 0 and every strategy
liquidates any open position on the final bar, paying commission and slippage
both times. The previous code let the ML strategy sit out bar 0 for free while
buy-and-hold paid to enter, which flattered the ML strategy by roughly ``1/n``.

*One total return.* ``results['total_return']`` and
``results['metrics']['total_return']`` are the same number, both net of exit
costs, because the final equity-curve point is the post-liquidation value.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class Backtester:
    """Backtesting engine for stock prediction models."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,   # 10 bps per side
        slippage: float = 0.0005,    # 5 bps per side
        position_size: float = 1.0,  # Fraction of capital per trade
        risk_free_rate: float = 0.0  # Annual, used for Sharpe/Sortino
    ):
        """
        Args:
            initial_capital: Starting capital
            commission: Commission per side, as a fraction of trade value
            slippage: Slippage per side, as a fraction of trade value
            position_size: Fraction of capital deployed per trade
            risk_free_rate: Annual risk-free rate for risk-adjusted ratios.
                Defaults to 0, so Sharpe is the raw risk-adjusted return rather
                than an excess-over-cash claim.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate

    @property
    def cost_per_side(self) -> float:
        """Round-trip cost is charged per side: commission + slippage."""
        return self.commission + self.slippage

    # ------------------------------------------------------------------
    # Return-space backtest (used by the walk-forward pipeline)
    # ------------------------------------------------------------------

    def long_flat_backtest(
        self,
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray,
        dates: pd.DatetimeIndex,
        threshold: float = 0.0,
        cost_per_side: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Long/flat backtest driven by a forecast of the next period's return.

        Signal: go long when the forecast next-day log return exceeds
        ``threshold``, otherwise hold cash. This is a forecast-versus-spot signal
        -- the forecast is already expressed relative to today's price -- not
        momentum in the forecast series.

        Timing: the position for bar ``i`` is chosen from the forecast made at
        the close of bar ``i`` and earns ``actual_returns[i]``, the realised
        return from bar ``i`` to bar ``i + 1``. Costs are charged whenever the
        position changes, including the entry on bar 0 and the liquidation after
        the final bar.

        Args:
            predicted_returns: Forecast log returns, one per bar
            actual_returns: Realised log returns, one per bar
            dates: Date index, one per bar
            threshold: Minimum forecast return required to hold the position
            cost_per_side: Override the instance commission + slippage, as a
                fraction (used for the cost-sensitivity sweep)

        Returns:
            Dict with equity_curve (length n+1), positions, metrics, turnover
        """
        predicted = np.asarray(predicted_returns, dtype=float)
        actual = np.asarray(actual_returns, dtype=float)

        if len(predicted) != len(actual):
            raise ValueError(
                f"predicted_returns ({len(predicted)}) and actual_returns "
                f"({len(actual)}) must be the same length"
            )

        cost = self.cost_per_side if cost_per_side is None else cost_per_side
        n = len(actual)

        positions = (predicted > threshold).astype(float)
        simple_returns = np.expm1(actual)  # log return -> simple return

        # Position changes, including entry from flat and final liquidation.
        prior = np.concatenate([[0.0], positions])
        trades = np.abs(np.diff(np.concatenate([prior, [0.0]])))
        entry_costs, exit_cost = trades[:n], trades[n]

        equity = np.empty(n + 1)
        equity[0] = self.initial_capital
        for i in range(n):
            gross = 1 + positions[i] * simple_returns[i]
            equity[i + 1] = equity[i] * (gross - entry_costs[i] * cost)

        # Liquidate whatever is still open on the last bar.
        equity[n] *= 1 - exit_cost * cost

        metrics = self._calculate_backtest_metrics(equity, n_trades=int(trades.sum()))
        metrics['turnover'] = float(trades.sum())
        metrics['turnover_per_year'] = float(trades.sum() / (n / TRADING_DAYS)) if n else 0.0
        metrics['time_in_market'] = float(positions.mean()) if n else 0.0
        metrics['cost_per_side_bps'] = cost * 10000

        return {
            'equity_curve': equity,
            'positions': positions,
            'dates': dates,
            'final_equity': float(equity[-1]),
            'total_return': metrics['total_return'],
            'metrics': metrics,
        }

    def buy_and_hold_returns(
        self,
        actual_returns: np.ndarray,
        dates: pd.DatetimeIndex,
        cost_per_side: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Buy-and-hold benchmark in return space, on the same bars and cost model.

        Implemented as a long/flat backtest whose forecast is always positive, so
        it goes through exactly the same code path, pays the same entry and exit
        costs, and is scored over the same number of bars.

        Args:
            actual_returns: Realised log returns, one per bar
            dates: Date index
            cost_per_side: Override commission + slippage

        Returns:
            Same structure as long_flat_backtest
        """
        always_long = np.ones(len(actual_returns))
        return self.long_flat_backtest(
            predicted_returns=always_long,
            actual_returns=actual_returns,
            dates=dates,
            threshold=0.0,
            cost_per_side=cost_per_side,
        )

    def cost_sensitivity(
        self,
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray,
        dates: pd.DatetimeIndex,
        cost_grid_bps: List[float] = (0, 5, 10, 20),
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Re-run the strategy and the benchmark across a grid of per-side costs.

        A strategy that only wins at zero cost has not found anything tradeable.

        Args:
            predicted_returns: Forecast log returns
            actual_returns: Realised log returns
            dates: Date index
            cost_grid_bps: Per-side costs in basis points
            threshold: Signal threshold

        Returns:
            DataFrame, one row per cost level
        """
        rows = []
        for bps in cost_grid_bps:
            cost = bps / 10000.0
            strat = self.long_flat_backtest(
                predicted_returns, actual_returns, dates, threshold, cost
            )
            bench = self.buy_and_hold_returns(actual_returns, dates, cost)
            rows.append({
                'cost_bps_per_side': bps,
                'strategy_total_return_pct': strat['metrics']['total_return'],
                'strategy_annualized_pct': strat['metrics']['annualized_return'],
                'strategy_sharpe': strat['metrics']['sharpe_ratio'],
                'strategy_max_drawdown_pct': strat['metrics']['max_drawdown'],
                'buy_hold_total_return_pct': bench['metrics']['total_return'],
                'buy_hold_sharpe': bench['metrics']['sharpe_ratio'],
                'excess_vs_buy_hold_pct': (
                    strat['metrics']['total_return'] - bench['metrics']['total_return']
                ),
                'turnover': strat['metrics']['turnover'],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Share-based backtest (price-level path)
    # ------------------------------------------------------------------

    def simple_strategy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex,
        threshold: float = 0.01
    ) -> Dict[str, any]:
        """
        Threshold strategy on a forecast of tomorrow's price.

        Signal, matching this docstring exactly: compare the forecast for
        tomorrow against *today's spot price*,
        ``pred_return = (predictions[i] - actuals[i]) / actuals[i]``, then

        - buy when ``pred_return > threshold`` and flat
        - sell when ``pred_return < -threshold`` and long
        - hold otherwise

        The earlier implementation differenced consecutive predictions, which
        measures momentum in the forecast series rather than the forecast against
        spot, so the signal did not match either the docstring or the intent.

        Sizing is cost-aware: the trade is sized so that value plus commission
        and slippage fits inside available capital. Sizing on price alone made
        ``total_cost > capital`` at ``position_size=1.0``, so the affordability
        guard silently rejected every trade.

        Args:
            predictions: Forecast price for bar i+1, one per bar
            actuals: Spot price at bar i, one per bar
            dates: Date index, one per bar
            threshold: Minimum absolute forecast return to act on

        Returns:
            Dict with equity_curve (length n+1 -- see module docstring), positions
            (also n+1), trades, final_equity, total_return, metrics
        """
        predictions = np.asarray(predictions, dtype=float)
        actuals = np.asarray(actuals, dtype=float)

        if len(predictions) != len(actuals):
            raise ValueError(
                f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
                f"must be the same length"
            )

        logger.info("Running simple strategy with threshold %s", threshold)

        capital = float(self.initial_capital)
        position = 0

        equity_curve = [capital]
        positions = [0]
        trades: List[Dict] = []

        # Forecast versus spot, available from bar 0 onwards.
        pred_returns = (predictions - actuals) / actuals

        for i in range(len(predictions)):
            price = actuals[i]
            signal = pred_returns[i]

            if signal > threshold and position == 0:
                # Size so that value + costs fits the capital available.
                budget = capital * self.position_size
                shares = int(budget / (price * (1 + self.cost_per_side)))

                if shares > 0:
                    trade_value = shares * price
                    commission_cost = trade_value * self.commission
                    slippage_cost = trade_value * self.slippage
                    capital -= trade_value + commission_cost + slippage_cost
                    position = shares
                    trades.append(self._trade(dates[i], 'BUY', price, shares,
                                              commission_cost, slippage_cost))

            elif signal < -threshold and position > 0:
                capital, position = self._liquidate(
                    capital, position, price, dates[i], trades
                )

            equity_curve.append(capital + position * price)
            positions.append(position)

        # Symmetric exit: liquidate any open position on the final bar.
        if position > 0:
            capital, position = self._liquidate(
                capital, position, actuals[-1], dates[-1], trades
            )
            equity_curve[-1] = capital
            positions[-1] = 0

        equity_curve = np.array(equity_curve)
        metrics = self._calculate_backtest_metrics(equity_curve, trades=trades)
        final_equity = float(equity_curve[-1])

        logger.info(
            "Backtest complete: final equity $%.2f, return %.2f%%, %d trades",
            final_equity, metrics['total_return'], len(trades)
        )

        return {
            'equity_curve': equity_curve,
            'positions': np.array(positions),
            'trades': trades,
            'final_equity': final_equity,
            'total_return': metrics['total_return'],
            'metrics': metrics,
        }

    def buy_and_hold_strategy(
        self,
        prices: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[str, any]:
        """
        Buy on the first bar, liquidate on the last, for comparison.

        Uses the same cost-aware sizing, the same n+1 equity-curve contract and
        the same exit convention as ``simple_strategy``, so the two are scored
        over an identical number of bars and both pay entry and exit costs.

        Args:
            prices: Spot prices, one per bar
            dates: Date index, one per bar

        Returns:
            Dict with equity_curve (length n+1), positions, trades, final_equity,
            total_return, metrics
        """
        prices = np.asarray(prices, dtype=float)
        logger.info("Running buy and hold strategy")

        capital = float(self.initial_capital)
        entry_price = prices[0]
        shares = int((capital * self.position_size) / (entry_price * (1 + self.cost_per_side)))

        trades: List[Dict] = []
        if shares > 0:
            trade_value = shares * entry_price
            commission_cost = trade_value * self.commission
            slippage_cost = trade_value * self.slippage
            capital -= trade_value + commission_cost + slippage_cost
            trades.append(self._trade(dates[0], 'BUY', entry_price, shares,
                                      commission_cost, slippage_cost))

        equity_curve = np.concatenate([[self.initial_capital], capital + shares * prices])

        # Symmetric exit on the final bar, so the last curve point is the
        # post-liquidation value and there is only one total-return number.
        if shares > 0:
            capital, _ = self._liquidate(capital, shares, prices[-1], dates[-1], trades)
            equity_curve[-1] = capital

        metrics = self._calculate_backtest_metrics(equity_curve, trades=trades)
        final_equity = float(equity_curve[-1])

        logger.info(
            "Buy and hold: final equity $%.2f, return %.2f%%", final_equity,
            metrics['total_return']
        )

        return {
            'equity_curve': equity_curve,
            'positions': np.concatenate([[0], np.full(len(prices), shares)]),
            'trades': trades,
            'final_equity': final_equity,
            'total_return': metrics['total_return'],
            'metrics': metrics,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _trade(date, action, price, shares, commission_cost, slippage_cost) -> Dict:
        return {
            'date': date,
            'action': action,
            'price': price,
            'shares': shares,
            'value': shares * price,
            'commission': commission_cost,
            'slippage': slippage_cost,
        }

    def _liquidate(self, capital, position, price, date, trades):
        """Sell the open position at ``price``, paying commission and slippage."""
        trade_value = position * price
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        capital += trade_value - commission_cost - slippage_cost
        trades.append(self._trade(date, 'SELL', price, position,
                                  commission_cost, slippage_cost))
        return capital, 0

    def _calculate_backtest_metrics(
        self,
        equity_curve: np.ndarray,
        trades: Optional[List[Dict]] = None,
        n_trades: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Performance metrics from an n+1 equity curve.

        The curve holds ``n + 1`` points for ``n`` bars, so annualisation uses
        ``n`` periods, not ``n + 1``. Counting the seed point as a trading day
        biased every annualised figure downward.

        Two Sharpe definitions are reported, because they answer different
        questions and disagree whenever the return path is not smooth:

        ``sharpe_ratio``
            ``(CAGR - rf) / annualised volatility``. Geometric in the numerator,
            arithmetic in the denominator. This is the growth an investor actually
            realised per unit of volatility, and it is the column quoted in
            docs/PIPELINE.md.
        ``sharpe_arithmetic``
            ``mean(excess period return) / std(period return) * sqrt(252)``, the
            textbook definition. Always the higher of the two for a volatile path,
            because compounding penalises variance.

        Args:
            equity_curve: Length n+1, starting at initial capital
            trades: Trade list, for win rate and profit factor
            n_trades: Trade count when there is no trade list (return-space path)

        Returns:
            Dict of metrics
        """
        equity_curve = np.asarray(equity_curve, dtype=float)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        n_periods = len(equity_curve) - 1
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100

        years = n_periods / TRADING_DAYS
        annualized_return = (
            ((equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1) * 100
            if years > 0 and equity_curve[-1] > 0 else 0.0
        )

        volatility = float(np.std(returns) * np.sqrt(TRADING_DAYS) * 100) if len(returns) else 0.0
        excess_return = annualized_return / 100 - self.risk_free_rate
        sharpe_ratio = excess_return / (volatility / 100) if volatility else 0.0

        # Textbook arithmetic Sharpe, reported alongside for comparability.
        period_rf = self.risk_free_rate / TRADING_DAYS
        period_std = float(np.std(returns)) if len(returns) else 0.0
        sharpe_arithmetic = (
            (float(np.mean(returns)) - period_rf) / period_std * np.sqrt(TRADING_DAYS)
            if period_std else 0.0
        )

        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = float(abs(np.min(drawdown)) * 100)

        downside = returns[returns < 0]
        downside_vol = float(np.std(downside) * np.sqrt(TRADING_DAYS) * 100) if len(downside) else 0.0
        sortino_ratio = excess_return / (downside_vol / 100) if downside_vol else 0.0

        win_rate, profit_factor, trade_count = 0.0, 0.0, 0
        if trades:
            buys = [t for t in trades if t['action'] == 'BUY']
            sells = [t for t in trades if t['action'] == 'SELL']
            trade_count = min(len(buys), len(sells))

            wins, gross_profit, gross_loss = 0, 0.0, 0.0
            for buy, sell in zip(buys, sells):
                profit = (
                    (sell['price'] - buy['price']) * buy['shares']
                    - buy['commission'] - buy['slippage']
                    - sell['commission'] - sell['slippage']
                )
                if profit > 0:
                    wins += 1
                    gross_profit += profit
                else:
                    gross_loss += abs(profit)

            win_rate = wins / trade_count * 100 if trade_count else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        elif n_trades is not None:
            trade_count = n_trades

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': volatility,
            'sharpe_ratio': float(sharpe_ratio),
            'sharpe_arithmetic': float(sharpe_arithmetic),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': max_drawdown,
            'calmar_ratio': float(annualized_return / max_drawdown) if max_drawdown else 0.0,
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'n_trades': int(trade_count),
            'n_periods': int(n_periods),
        }

    def compare_strategies(
        self,
        strategy_results: Dict[str, Dict],
        buy_hold_results: Dict
    ) -> pd.DataFrame:
        """
        Put strategies and the benchmark side by side.

        Safe to compare directly: every strategy here is scored over the same
        bars under the same entry/exit cost convention.

        Args:
            strategy_results: {strategy_name: results}
            buy_hold_results: Benchmark results

        Returns:
            Comparison DataFrame
        """
        periods = {
            name: res['metrics'].get('n_periods')
            for name, res in {**strategy_results, 'Buy & Hold': buy_hold_results}.items()
        }
        distinct = {p for p in periods.values() if p is not None}
        if len(distinct) > 1:
            raise ValueError(
                f"Strategies cover different numbers of bars and cannot be "
                f"compared: {periods}"
            )

        all_results = {
            'Buy & Hold': buy_hold_results['metrics'],
            **{name: res['metrics'] for name, res in strategy_results.items()},
        }
        return pd.DataFrame(all_results).T


def print_backtest_results(results: Dict[str, any]):
    """
    Print backtest results in a readable block.

    Args:
        results: Backtest results dictionary
    """
    metrics = results['metrics']

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print("\nReturns:")
    print(f"  Total Return:           {metrics['total_return']:.2f}%")
    print(f"  Annualized Return:      {metrics['annualized_return']:.2f}%")

    print("\nRisk:")
    print(f"  Volatility:             {metrics['volatility']:.2f}%")
    print(f"  Maximum Drawdown:       {metrics['max_drawdown']:.2f}%")

    print("\nRisk-Adjusted:")
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:          {metrics['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio:           {metrics['calmar_ratio']:.4f}")

    print("\nTrading:")
    print(f"  Number of Trades:       {int(metrics['n_trades'])}")
    if 'turnover' in metrics:
        print(f"  Turnover (position chg):{metrics['turnover']:.0f}")
        print(f"  Time in Market:         {metrics['time_in_market']*100:.1f}%")
    else:
        print(f"  Win Rate:               {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor:          {metrics['profit_factor']:.4f}")

    print("\nFinal:")
    print(f"  Final Equity:           ${results['final_equity']:,.2f}")
    print("=" * 60 + "\n")
