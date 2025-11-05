"""
Backtesting framework for stock price prediction models
Tests models with realistic trading scenarios including costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine for stock prediction models
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        position_size: float = 1.0   # Fraction of capital to use per trade
    ):
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            slippage: Slippage per trade (as fraction)
            position_size: Position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size

    def simple_strategy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex,
        threshold: float = 0.01
    ) -> Dict[str, any]:
        """
        Simple trading strategy based on predictions

        Strategy:
        - Buy if predicted return > threshold
        - Sell if predicted return < -threshold
        - Hold otherwise

        Args:
            predictions: Predicted prices
            actuals: Actual prices
            dates: Date index
            threshold: Threshold for trading signal (as fraction)

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running simple strategy with threshold {threshold}")

        # Initialize
        capital = self.initial_capital
        position = 0  # Number of shares
        position_value = 0

        # Track history
        equity_curve = [capital]
        positions = [0]
        trades = []

        # Calculate predicted returns
        pred_returns = np.diff(predictions) / predictions[:-1]
        pred_returns = np.concatenate([[0], pred_returns])

        for i in range(1, len(predictions)):
            current_price = actuals[i]
            pred_return = pred_returns[i]

            # Generate signal
            if pred_return > threshold and position == 0:
                # Buy signal
                shares_to_buy = int((capital * self.position_size) / current_price)

                if shares_to_buy > 0:
                    # Calculate costs
                    trade_value = shares_to_buy * current_price
                    commission_cost = trade_value * self.commission
                    slippage_cost = trade_value * self.slippage
                    total_cost = trade_value + commission_cost + slippage_cost

                    if total_cost <= capital:
                        position = shares_to_buy
                        capital -= total_cost
                        position_value = position * current_price

                        trades.append({
                            'date': dates[i],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': trade_value,
                            'commission': commission_cost,
                            'slippage': slippage_cost
                        })

            elif pred_return < -threshold and position > 0:
                # Sell signal
                trade_value = position * current_price
                commission_cost = trade_value * self.commission
                slippage_cost = trade_value * self.slippage
                net_proceeds = trade_value - commission_cost - slippage_cost

                capital += net_proceeds

                trades.append({
                    'date': dates[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': trade_value,
                    'commission': commission_cost,
                    'slippage': slippage_cost
                })

                position = 0
                position_value = 0

            # Update position value
            if position > 0:
                position_value = position * current_price

            # Calculate total equity
            total_equity = capital + position_value

            equity_curve.append(total_equity)
            positions.append(position)

        # Calculate final metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Calculate metrics
        metrics = self._calculate_backtest_metrics(
            equity_curve,
            dates,
            trades
        )

        results = {
            'equity_curve': np.array(equity_curve),
            'positions': np.array(positions),
            'trades': trades,
            'final_equity': final_equity,
            'total_return': total_return,
            'metrics': metrics
        }

        logger.info(f"Backtest completed: Final equity ${final_equity:.2f}, Return {total_return:.2f}%")

        return results

    def buy_and_hold_strategy(
        self,
        prices: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[str, any]:
        """
        Buy and hold strategy for comparison

        Args:
            prices: Actual prices
            dates: Date index

        Returns:
            Dictionary with backtest results
        """
        logger.info("Running buy and hold strategy")

        # Buy at first price
        initial_price = prices[0]
        shares = int(self.initial_capital / initial_price)

        # Calculate costs
        trade_value = shares * initial_price
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        total_cost = trade_value + commission_cost + slippage_cost

        remaining_cash = self.initial_capital - total_cost

        # Calculate equity curve
        equity_curve = remaining_cash + shares * prices

        # Final sell
        final_price = prices[-1]
        final_value = shares * final_price
        final_commission = final_value * self.commission
        final_slippage = final_value * self.slippage
        final_equity = remaining_cash + final_value - final_commission - final_slippage

        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Calculate metrics
        trades = [
            {
                'date': dates[0],
                'action': 'BUY',
                'price': initial_price,
                'shares': shares,
                'value': trade_value,
                'commission': commission_cost,
                'slippage': slippage_cost
            },
            {
                'date': dates[-1],
                'action': 'SELL',
                'price': final_price,
                'shares': shares,
                'value': final_value,
                'commission': final_commission,
                'slippage': final_slippage
            }
        ]

        metrics = self._calculate_backtest_metrics(
            equity_curve,
            dates,
            trades
        )

        results = {
            'equity_curve': equity_curve,
            'positions': np.full(len(prices), shares),
            'trades': trades,
            'final_equity': final_equity,
            'total_return': total_return,
            'metrics': metrics
        }

        logger.info(f"Buy and hold: Final equity ${final_equity:.2f}, Return {total_return:.2f}%")

        return results

    def _calculate_backtest_metrics(
        self,
        equity_curve: np.ndarray,
        dates: pd.DatetimeIndex,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate backtest performance metrics

        Args:
            equity_curve: Equity over time
            dates: Date index
            trades: List of trades

        Returns:
            Dictionary of metrics
        """
        equity_curve = np.array(equity_curve)

        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100

        # Annualized return
        n_days = len(equity_curve)
        years = n_days / 252
        annualized_return = ((equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) * 100

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = annualized_return / 100 - risk_free_rate
        sharpe_ratio = excess_return / (volatility / 100) if volatility != 0 else 0

        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) * 100

        # Win rate
        n_trades = len(trades) // 2  # Buy and sell pairs
        if n_trades > 0:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']

            wins = 0
            total_profit = 0
            total_loss = 0

            for buy, sell in zip(buy_trades, sell_trades):
                profit = (sell['price'] - buy['price']) * buy['shares'] - sell['commission'] - sell['slippage']
                if profit > 0:
                    wins += 1
                    total_profit += profit
                else:
                    total_loss += abs(profit)

            win_rate = wins / n_trades * 100 if n_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            n_trades = 0

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        downside_vol = downside_std * np.sqrt(252) * 100
        sortino_ratio = excess_return / (downside_vol / 100) if downside_vol != 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': n_trades
        }

        return metrics

    def compare_strategies(
        self,
        strategy_results: Dict[str, Dict],
        buy_hold_results: Dict
    ) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            strategy_results: Dictionary of {strategy_name: results}
            buy_hold_results: Buy and hold results for comparison

        Returns:
            Comparison DataFrame
        """
        all_results = {
            'Buy & Hold': buy_hold_results['metrics'],
            **{name: results['metrics'] for name, results in strategy_results.items()}
        }

        df = pd.DataFrame(all_results).T

        return df


class WalkForwardBacktester:
    """
    Walk-forward backtesting with retraining
    """

    def __init__(
        self,
        train_size: int = 252,  # 1 year
        test_size: int = 21,    # 1 month
        retrain_frequency: int = 21,  # Retrain monthly
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize walk-forward backtester

        Args:
            train_size: Training window size
            test_size: Test window size
            retrain_frequency: How often to retrain
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage per trade
        """
        self.train_size = train_size
        self.test_size = test_size
        self.retrain_frequency = retrain_frequency
        self.backtester = Backtester(initial_capital, commission, slippage)

    def run(
        self,
        data: pd.DataFrame,
        model_class: type,
        feature_cols: List[str],
        target_col: str
    ) -> Dict[str, any]:
        """
        Run walk-forward backtest

        Args:
            data: DataFrame with features and target
            model_class: Model class to use
            feature_cols: Feature column names
            target_col: Target column name

        Returns:
            Backtest results
        """
        logger.info("Running walk-forward backtest")

        predictions = []
        actuals = []
        dates = []

        n = len(data)
        start_idx = self.train_size

        while start_idx + self.test_size <= n:
            # Define windows
            train_start = start_idx - self.train_size
            train_end = start_idx
            test_end = min(start_idx + self.test_size, n)

            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[start_idx:test_end]

            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]

            # Train model
            model = model_class()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Store results
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            dates.extend(test_data.index)

            # Move window
            start_idx += self.retrain_frequency

        # Run backtesting on predictions
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        dates = pd.DatetimeIndex(dates)

        # Strategy backtest
        strategy_results = self.backtester.simple_strategy(
            predictions, actuals, dates
        )

        # Buy and hold backtest
        buy_hold_results = self.backtester.buy_and_hold_strategy(
            actuals, dates
        )

        results = {
            'strategy': strategy_results,
            'buy_hold': buy_hold_results,
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates
        }

        logger.info("Walk-forward backtest completed")

        return results


def print_backtest_results(results: Dict[str, any]):
    """
    Print backtest results in a formatted way

    Args:
        results: Backtest results dictionary
    """
    metrics = results['metrics']

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n💰 Returns:")
    print(f"  Total Return:           {metrics['total_return']:.2f}%")
    print(f"  Annualized Return:      {metrics['annualized_return']:.2f}%")

    print(f"\n📊 Risk Metrics:")
    print(f"  Volatility:             {metrics['volatility']:.2f}%")
    print(f"  Maximum Drawdown:       {metrics['max_drawdown']:.2f}%")

    print(f"\n📈 Risk-Adjusted Returns:")
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:          {metrics['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio:           {metrics['calmar_ratio']:.4f}")

    print(f"\n🎯 Trading Statistics:")
    print(f"  Number of Trades:       {int(metrics['n_trades'])}")
    print(f"  Win Rate:               {metrics['win_rate']:.2f}%")
    print(f"  Profit Factor:          {metrics['profit_factor']:.4f}")

    print(f"\n💵 Final Results:")
    print(f"  Final Equity:           ${results['final_equity']:,.2f}")

    print("=" * 60 + "\n")
