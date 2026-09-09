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
side charged on that last day. So a single round trip costs two sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

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
# are the same story twice, and cost_drag is gross minus net.
TABLE_COLUMNS = [
    "threshold",
    "gross_return",
    "net_return",
    "sharpe",
    "max_drawdown",
    "n_trades",
    "days_in_market",
]


def _as_1d(name: str, values) -> np.ndarray:
    """Coerce one input to a non-empty, finite 1-D float array."""
    array = np.asarray(values, dtype=float)

    # A column vector is the usual shape accident (a DataFrame slice, an
    # sklearn output) and means the same thing; anything else is a real bug.
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (or a single-column 2-D array), got shape "
            f"{np.shape(values)}."
        )
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # A NaN prediction would silently read as "flat" and a NaN return would
    # poison the whole equity curve from that day on. Either is a bug upstream.
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(f"{name} contains {bad} non-finite value(s) (NaN or inf).")

    return array


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
        "n_round_trips": n_trades / 2,
        "days_in_market": float(position.mean()),
        "equity_curve": equity_curve,
        "daily_net": net,
    }


def _annualized_sharpe(net: np.ndarray) -> float:
    """Annualized Sharpe of a daily net-return series, NaN where undefined.

    A flat strategy has zero volatility, and zero over zero is not a good
    Sharpe -- it is no Sharpe at all, so say NaN rather than invent one.
    """
    if net.size < 2:
        return float("nan")

    volatility = float(net.std(ddof=1))
    if volatility == 0.0:
        return float("nan")

    return float(net.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR))


def backtest_long_flat(
    pred, realized, threshold: float, cost_per_side: float
) -> dict:
    """Backtest "long when ``pred > threshold``, flat otherwise".

    Args:
        pred: Forecast for each day, made through that day's close.
        realized: Simple return from each day's close to the next.
        threshold: Forecast level a day must clear to be traded long.
        cost_per_side: Cost charged once per side traded, as a fraction.

    Returns:
        A dict with the :data:`SUMMARY_KEYS` scalars plus ``equity_curve``
        (length ``n + 1``, starting at 1.0) and ``daily_net`` (length ``n``).

    Raises:
        ValueError: If the two series differ in length, are empty, are not
            1-D, or contain non-finite values.
    """
    predictions = _as_1d("pred", pred)
    returns = _as_1d("realized", realized)
    if predictions.size != returns.size:
        raise ValueError(
            f"pred and realized must have the same length, got "
            f"{predictions.size} and {returns.size}."
        )

    position = (predictions > float(threshold)).astype(float)
    return _run(position, returns, float(cost_per_side))


def buy_and_hold(realized, cost_per_side: float) -> dict:
    """The benchmark: long every day, one entry and one exit, nothing else.

    Same keys as :func:`backtest_long_flat`, so the two drop into the same
    summary table.
    """
    returns = _as_1d("realized", realized)
    return _run(np.ones(returns.size), returns, float(cost_per_side))


def cost_drag_arithmetic(round_trip_cost: float, round_trips_per_year: float) -> float:
    """Back-of-the-envelope annual cost of a given trading frequency.

    The linear estimate, not the compounded one: 0.0015 per round trip at 250
    round trips a year is 0.375, i.e. 37.5% of capital handed to the broker
    before the strategy has made a cent. It exists to make that number
    unmissable next to the backtested drag.
    """
    return float(round_trip_cost) * float(round_trips_per_year)


def select_threshold(
    pred_val, realized_val, thresholds, cost_per_side: float
) -> tuple[float, pd.DataFrame]:
    """Sweep a threshold grid on validation data and pick the best net Sharpe.

    Args:
        pred_val: Validation-period forecasts.
        realized_val: Validation-period realized next-day returns.
        thresholds: Threshold grid to try, reported in the order given.
        cost_per_side: Cost charged once per side traded.

    Returns:
        ``(best_threshold, table)`` where ``table`` has one row per threshold
        with the :data:`TABLE_COLUMNS` columns.

    Raises:
        ValueError: If the grid is empty or every rule has an undefined
            (NaN) Sharpe, which means no rule ever traded.
    """
    grid = [float(threshold) for threshold in thresholds]
    if not grid:
        raise ValueError("thresholds must be non-empty.")

    rows = []
    for threshold in grid:
        result = backtest_long_flat(pred_val, realized_val, threshold, cost_per_side)
        rows.append(
            {"threshold": threshold, **{key: result[key] for key in TABLE_COLUMNS[1:]}}
        )

    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)

    sharpes = table["sharpe"].to_numpy(dtype=float)
    if np.all(np.isnan(sharpes)):
        raise ValueError(
            "Every threshold produced an undefined Sharpe; none of them ever "
            "took a position."
        )

    # NaN never wins, and among equally good rules the lowest threshold does:
    # it trades the most days, so its Sharpe rests on the most evidence.
    best_sharpe = np.nanmax(sharpes)
    tied = [
        threshold
        for threshold, sharpe in zip(grid, sharpes)
        if not np.isnan(sharpe) and sharpe == best_sharpe
    ]
    return min(tied), table
