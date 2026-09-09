"""
Long/flat threshold backtest with transaction costs and turnover.

A return forecast only matters if it survives the trading it implies, so the
rule here is deliberately the simplest one that can lose money honestly: go
long whenever the forecast clears a threshold, sit in cash otherwise, and pay
a fixed cost every time a side is traded.

Alignment convention -- the whole point of the module:

* ``pred[t]`` is the forecast made with data through the close of day ``t``.
* ``realized[t]`` is the simple return from close ``t`` to close ``t + 1``
  (exactly the ``target`` column produced by the feature builder).
* A position taken at the close of day ``t`` therefore earns ``realized[t]``.

Because the position on day ``t`` reads only ``pred[t]``, there is no
look-ahead by construction.

Cost convention: ``cost_per_side`` is charged once per side traded. Entering
costs one side on the day of entry, exiting costs one side on the day of
exit, and a position still open on the final day is closed out with one extra
side charged on that last day. So a single round trip costs two sides. Costs
are subtracted additively from the day's return before compounding, which
drops the second-order term ``cost_per_side * changes * gross`` -- at a few
basis points a side that term is smaller than the rounding on the prices the
returns came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src._validation import as_1d

TRADING_DAYS_PER_YEAR = 252

# Below this, a "standard deviation" is floating-point noise around a constant
# series rather than risk, and dividing by it invents a Sharpe of ~1e17.
_MIN_VOLATILITY = 1e-12

# A rule that trades a handful of days can post a spectacular Sharpe on one
# lucky move. This is the default number of in-market days a threshold must
# have before its score is allowed to win a sweep -- roughly a trading month.
DEFAULT_MIN_DAYS_IN_MARKET = 20

# The scalar fields of a backtest result, in report order. Task 5's summary
# tables are built from this, so it stays the single definition of "what we
# report" -- the two array fields are deliberately excluded.
SUMMARY_KEYS = [
    "gross_return",
    "net_return",
    "cost_drag",
    "total_cost",
    "sharpe",
    "max_drawdown",
    "n_trades",
    "n_round_trips",
    "days_in_market",
]

# The per-threshold sweep reports a readable subset: the compounding fields
# are the same story twice, and cost_drag is just gross minus net.
_SWEEP_METRICS = [
    "gross_return",
    "net_return",
    "sharpe",
    "max_drawdown",
    "n_trades",
    "days_in_market",
]

TABLE_COLUMNS = ["threshold", *_SWEEP_METRICS, "eligible"]


def _as_cost(cost_per_side) -> float:
    """Coerce a per-side cost to a finite, non-negative fraction."""
    cost = float(cost_per_side)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(
            "cost_per_side must be a finite, non-negative fraction, got "
            f"{cost_per_side!r}."
        )
    return cost


def _as_threshold(threshold) -> float:
    """Coerce a threshold to a finite float.

    A NaN threshold would make ``pred > threshold`` False everywhere, so the
    strategy would silently sit in cash and report a flat, blameless year.
    """
    value = float(threshold)
    if not np.isfinite(value):
        raise ValueError(f"threshold must be finite, got {threshold!r}.")
    return value


def _validated_pair(pred, realized) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a forecast and a return series to aligned 1-D float arrays."""
    predictions = as_1d("pred", pred)
    returns = as_1d("realized", realized)
    if predictions.size != returns.size:
        raise ValueError(
            f"pred and realized must have the same length, got "
            f"{predictions.size} and {returns.size}."
        )
    return predictions, returns


def _annualized_sharpe(net: np.ndarray) -> float:
    """Annualized Sharpe of a daily net-return series, NaN where undefined.

    A flat -- or merely constant -- strategy has no volatility, and a return
    over no volatility is not a good Sharpe but no Sharpe at all. The
    comparison is against a small floor rather than exact zero because
    summing identical floats leaves a standard deviation around 1e-18, which
    is enough to manufacture a Sharpe of 9e16.
    """
    if net.size < 2:
        return float("nan")

    volatility = float(net.std(ddof=1))
    if not volatility > _MIN_VOLATILITY:
        return float("nan")

    return float(net.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR))


def _run(position: np.ndarray, realized: np.ndarray, cost_per_side: float) -> dict:
    """Score an already-decided position path against the realized returns."""
    # One entry per day: 1 when a side is traded that day, 0 when the position
    # is simply carried. Starting flat makes the first long day an entry.
    prev = np.concatenate([[0.0], position[:-1]])
    changes = np.abs(position - prev)

    gross = position * realized
    net = gross - cost_per_side * changes
    # A position still open after the last day has to be closed; charge that
    # final side on the last day so the equity curve ends flat and honest.
    net[-1] -= cost_per_side * position[-1]

    n_trades = int(round(float(changes.sum() + position[-1])))

    equity_curve = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    gross_return = float(np.prod(1.0 + gross) - 1.0)
    net_return = float(equity_curve[-1] - 1.0)

    return {
        "gross_return": gross_return,
        "net_return": net_return,
        "cost_drag": gross_return - net_return,
        # The linear sum actually paid, as opposed to the compounded drag.
        "total_cost": float(n_trades * cost_per_side),
        "sharpe": _annualized_sharpe(net),
        "max_drawdown": float(
            (equity_curve / np.maximum.accumulate(equity_curve) - 1.0).min()
        ),
        "n_trades": n_trades,
        # Every entry is matched by an exit, including the forced final one,
        # so n_trades is even by construction and this division is exact.
        "n_round_trips": n_trades // 2,
        "days_in_market": float(position.mean()),
        "equity_curve": equity_curve,
        "daily_net": net,
    }


def backtest_long_flat(
    pred, realized, threshold: float, cost_per_side: float
) -> dict:
    """Backtest "long when ``pred > threshold``, flat otherwise".

    Args:
        pred: Forecast for each day, made through that day's close.
        realized: Simple return from each day's close to the next.
        threshold: Forecast level a day must clear to be traded long. Finite.
        cost_per_side: Cost charged once per side traded, as a fraction of
            position value (0.0005 is five basis points). Non-negative.

    Returns:
        A dict with the :data:`SUMMARY_KEYS` scalars plus ``equity_curve``
        (length ``n + 1``, starting at 1.0) and ``daily_net`` (length ``n``).
        ``days_in_market`` is the *fraction* of days holding the position, 0
        to 1, not a count; ``n_trades`` counts sides, ``n_round_trips`` pairs
        of them. All returns are fractions, so 0.12 is a 12% gain and
        ``max_drawdown`` of -0.3 is a 30% peak-to-trough loss.

    Raises:
        ValueError: If the two series differ in length, are empty, are not
            1-D, or contain non-finite values; if ``threshold`` is not
            finite; or if ``cost_per_side`` is negative or not finite.
    """
    predictions, returns = _validated_pair(pred, realized)
    position = (predictions > _as_threshold(threshold)).astype(float)
    return _run(position, returns, _as_cost(cost_per_side))


def buy_and_hold(realized, cost_per_side: float) -> dict:
    """The benchmark: long every day, one entry and one exit, nothing else.

    Same keys as :func:`backtest_long_flat`, so the two drop into the same
    summary table.
    """
    returns = as_1d("realized", realized)
    return _run(np.ones(returns.size), returns, _as_cost(cost_per_side))


def cost_drag_arithmetic(round_trip_cost: float, round_trips_per_year: float) -> float:
    """Back-of-the-envelope annual cost of a given trading frequency.

    The linear estimate, not the compounded one: 0.0015 per round trip at 250
    round trips a year is 0.375, i.e. 37.5% of capital handed to the broker
    before the strategy has made a cent. It exists to make that number
    unmissable next to the backtested drag.
    """
    return float(round_trip_cost) * float(round_trips_per_year)


def select_threshold(
    pred_val,
    realized_val,
    thresholds,
    cost_per_side: float,
    min_days_in_market: int = DEFAULT_MIN_DAYS_IN_MARKET,
) -> tuple[float, pd.DataFrame]:
    """Sweep a threshold grid on validation data and pick the best net Sharpe.

    A high threshold can trade two days a year, land on one +30% session and
    post a Sharpe no honest rule can touch. ``min_days_in_market`` is the
    guard against that: rules with less participation than the floor are
    still scored and still reported, they just cannot win.

    Args:
        pred_val: Validation-period forecasts.
        realized_val: Validation-period realized next-day returns.
        thresholds: Threshold grid to try, reported in the order given.
        cost_per_side: Cost charged once per side traded.
        min_days_in_market: Minimum number of in-market *days* (a count, not
            a fraction) a threshold needs to be eligible to win. 0 disables
            the guard.

    Returns:
        ``(best_threshold, table)`` where ``table`` has one row per threshold
        with the :data:`TABLE_COLUMNS` columns, including a boolean
        ``eligible`` flag.

    Raises:
        ValueError: If the grid is empty, if no threshold meets the
            participation floor, or if every eligible threshold has an
            undefined (NaN) Sharpe.
    """
    predictions, returns = _validated_pair(pred_val, realized_val)
    cost = _as_cost(cost_per_side)
    grid = [_as_threshold(threshold) for threshold in thresholds]
    if not grid:
        raise ValueError("thresholds must be non-empty.")
    floor = int(min_days_in_market)

    rows = []
    in_market_days = []
    for threshold in grid:
        position = (predictions > threshold).astype(float)
        result = _run(position, returns, cost)
        days = int(round(float(position.sum())))
        in_market_days.append(days)
        rows.append(
            {
                "threshold": threshold,
                **{key: result[key] for key in _SWEEP_METRICS},
                "eligible": days >= floor,
            }
        )

    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)

    eligible = table["eligible"].to_numpy(dtype=bool)
    sharpes = table["sharpe"].to_numpy(dtype=float)

    # Two different failures, two different fixes: too strict a floor is not
    # the same problem as a grid nobody could trade.
    if not eligible.any():
        raise ValueError(
            f"No threshold was in the market at least {floor} day(s); the "
            f"busiest rule traded {max(in_market_days)} of {returns.size} "
            "day(s). Lower min_days_in_market or widen the grid."
        )

    usable = eligible & ~np.isnan(sharpes)
    if not usable.any():
        raise ValueError(
            "Every eligible threshold produced an undefined Sharpe; none of "
            "them took a position with any variation in its daily returns."
        )

    # Among equally good rules the lowest threshold wins: it trades the most
    # days, so its score rests on the most evidence.
    best_sharpe = sharpes[usable].max()
    tied = [
        threshold
        for threshold, ok, sharpe in zip(grid, usable, sharpes)
        if ok and sharpe == best_sharpe
    ]
    return min(tied), table
