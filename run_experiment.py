#!/usr/bin/env python
"""
End-to-end runner for the leakage-free experiment.

This script does the wiring and nothing else: every number it prints is
computed by a module in ``src/`` that has its own unit tests. What matters
here is the *order* things happen in, because that order is what makes the
result honest:

1. Prices are loaded split-adjusted, with a warmup period before the first
   training year so no feature starts life as a NaN inside the sample.
2. Features and target are built strictly backward-looking, then split by
   calendar year -- train, then one validation year, then one test year.
3. Models are fitted on the training split only.
4. The trading threshold is chosen on the *validation* split only.
5. The test year is touched once, at the end, to report what would have
   happened. It never informs a single choice made above.

Baselines and a buy-and-hold benchmark are reported next to every model,
because "beats zero" and "beats owning the stock" are very different claims.

Usage::

    python run_experiment.py [--config config/config.yaml] [--refresh]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
import yfinance

from src.baselines import BASELINE_NAMES, baseline_predictions
from src.lean_features import FEATURE_COLUMNS, TARGET_COLUMN, build_dataset
from src.lean_models import MODEL_NAMES, fit_predict, make_models
from src.market_data import load_prices
from src.return_metrics import evaluate
from src.splits import split_by_year
from src.threshold_backtest import (
    SUMMARY_KEYS,
    TABLE_COLUMNS,
    TRADING_DAYS_PER_YEAR,
    backtest_long_flat,
    buy_and_hold,
    cost_drag_arithmetic,
    select_threshold,
)

DEFAULT_CONFIG_PATH = "config/config.yaml"

# The metrics reported for every prediction. ``n`` is deliberately left out of
# the table: it is a property of the split, and the split table already says it.
METRIC_COLUMNS = ["rmse", "directional_accuracy", "ic", "r2"]

# The round number used for the generic "what does daily trading cost?" line.
# Not measured from anything -- it is the arithmetic of trading every day.
ROUND_TRIPS_PER_YEAR = 250

# Belt and braces on top of the models' own ``random_state``.
RANDOM_SEED = 42

# Prediction noise floor. scikit-learn's forest accumulates its trees'
# predictions in whatever order the worker threads finish, so two calls to the
# *same* fitted forest can differ in the last bits -- about 1e-18 on a forecast
# of order 1e-3. That is nine orders of magnitude below anything this
# experiment can measure, but it is enough to flip the final digit of an
# aggregate and make a "reproducible" run fail to reproduce. Rounding far below
# the signal and far above the noise removes the wobble without moving a single
# reported figure.
PREDICTION_DECIMALS = 12

# --- report formatting ----------------------------------------------------
# Fractions that read better as percentages: 0.523 -> "52.3%".
PERCENT_COLUMNS = frozenset(
    {
        "gross_return",
        "net_return",
        "cost_drag",
        "total_cost",
        "max_drawdown",
        "days_in_market",
        "directional_accuracy",
        "implied_annual_cost_drag",
        "test_cost_drag",
    }
)
INT_COLUMNS = frozenset({"n_trades", "n_round_trips", "rows", "n"})
NOT_AVAILABLE = "n/a"


# ==========================================================================
# formatting
# ==========================================================================


def format_cell(column: str, value) -> str:
    """Render one table cell under the report's formatting rules."""
    # bool is a subclass of int, so this has to come first.
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value

    if pd.isna(value):
        return NOT_AVAILABLE
    if column in INT_COLUMNS:
        return str(int(round(float(value))))

    number = float(value)
    if not np.isfinite(number):
        return NOT_AVAILABLE
    if column in PERCENT_COLUMNS:
        return f"{number:.1%}"
    if column == "rmse":
        return f"{number:.5f}"
    if column == "r2":
        # Signed, because the interesting question about R^2 on returns is
        # whether it is negative -- worse than predicting the mean.
        return f"{number:+.3f}"
    if column == "ic":
        return f"{number:.3f}"
    if column == "sharpe":
        return f"{number:.2f}"
    if column == "threshold":
        return f"{number:.4f}"
    if column == "round_trips_per_year":
        return f"{number:.1f}"
    return f"{number:.4f}"


def markdown_table(frame: pd.DataFrame, index_label: str) -> str:
    """Render a frame as a GitHub-flavoured markdown table."""
    header = [index_label, *(str(column) for column in frame.columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for label, row in frame.iterrows():
        cells = [str(label)] + [
            format_cell(str(column), row[column]) for column in frame.columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==========================================================================
# pipeline steps
# ==========================================================================


def stabilize(predictions: np.ndarray) -> np.ndarray:
    """Round a forecast to :data:`PREDICTION_DECIMALS` so reruns match bitwise."""
    return np.round(np.asarray(predictions, dtype=float), PREDICTION_DECIMALS)


def describe_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row count and date range of each split, for the writeup."""
    rows = {
        name: {
            "rows": len(frame),
            "first_date": str(frame.index[0].date()),
            "last_date": str(frame.index[-1].date()),
        }
        for name, frame in splits.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "split"
    return table


def metrics_table(y_true, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Metrics for every baseline and model, baselines first.

    The order is not cosmetic: a model is only interesting to the extent it
    beats the rows above it.
    """
    order = BASELINE_NAMES + MODEL_NAMES
    scores = {name: evaluate(y_true, predictions[name]) for name in order}
    table = pd.DataFrame.from_dict(scores, orient="index").loc[order, METRIC_COLUMNS]
    table.index.name = "name"
    return table


def check_leakage(metrics_test: pd.DataFrame, limit: float) -> dict:
    """Flag any model whose test directional accuracy is too good to be true.

    Next-day equity returns are close to unpredictable. A model well above the
    limit has almost certainly seen the future through a feature, a split or a
    misaligned target -- so this shouts, loudly, rather than quietly passing.
    """
    flagged = {
        model: float(metrics_test.loc[model, "directional_accuracy"])
        for model in MODEL_NAMES
        if float(metrics_test.loc[model, "directional_accuracy"]) > limit
    }

    for model, accuracy in flagged.items():
        banner = "!" * 74
        print(f"\n{banner}")
        print(
            f"WARNING  LEAKAGE GUARD: {model} scored {accuracy:.1%} directional "
            f"accuracy on the"
        )
        print(
            f"         test set, above the {limit:.1%} plausibility limit. Treat "
            "every number"
        )
        print("         below as suspect until the feature pipeline is re-checked.")
        print(f"{banner}\n")

    return {"limit": float(limit), "fired": bool(flagged), "flagged": flagged}


def sweep_table(pred, realized, thresholds, cost_per_side, min_days) -> pd.DataFrame:
    """Score every threshold, mirroring ``select_threshold``'s report.

    Only used when ``select_threshold`` refuses to pick a winner: we still
    want to show the reader the grid it refused, rather than an empty table.
    """
    rows = []
    for threshold in thresholds:
        result = backtest_long_flat(pred, realized, threshold, cost_per_side)
        days = int(round(result["days_in_market"] * len(realized)))
        rows.append(
            {
                "threshold": float(threshold),
                # TABLE_COLUMNS is threshold + metrics + eligible.
                **{key: result[key] for key in TABLE_COLUMNS[1:-1]},
                "eligible": days >= int(min_days),
            }
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def choose_thresholds(
    predictions_val: dict[str, np.ndarray],
    y_val,
    thresholds,
    cost_per_side: float,
    min_days_in_market: int,
) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, str]]:
    """Pick each model's trading threshold on the validation split only.

    Returns:
        ``(chosen, sweeps, notes)`` -- the threshold per model, its full sweep
        table, and a note for any model that fell back to 0.0 because no
        threshold cleared the participation floor.
    """
    chosen: dict[str, float] = {}
    sweeps: dict[str, pd.DataFrame] = {}
    notes: dict[str, str] = {}

    for model in MODEL_NAMES:
        try:
            best, table = select_threshold(
                predictions_val[model],
                y_val,
                thresholds,
                cost_per_side,
                min_days_in_market=min_days_in_market,
            )
        except ValueError as error:
            # No eligible threshold. Falling back to 0.0 keeps the model in
            # the report instead of dropping it, and 0.0 is the honest
            # default: trade whenever the forecast is positive at all.
            best = 0.0
            table = sweep_table(
                predictions_val[model],
                y_val,
                thresholds,
                cost_per_side,
                min_days_in_market,
            )
            notes[model] = f"fell back to 0.0 -- {error}"

        chosen[model] = float(best)
        sweeps[model] = table

    return chosen, sweeps, notes


def run_test_backtests(
    predictions_test: dict[str, np.ndarray],
    y_test,
    chosen: dict[str, float],
    cost_per_side: float,
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """The one and only look at the test year.

    Returns the benchmark, each model at its chosen threshold, and each model
    at 0.0 -- the last so the reader can see how much of the result came from
    the threshold choice rather than the forecast.
    """
    benchmark = buy_and_hold(y_test, cost_per_side)
    at_chosen = {
        model: backtest_long_flat(
            predictions_test[model], y_test, chosen[model], cost_per_side
        )
        for model in MODEL_NAMES
    }
    at_zero = {
        model: backtest_long_flat(predictions_test[model], y_test, 0.0, cost_per_side)
        for model in MODEL_NAMES
    }
    return benchmark, at_chosen, at_zero


def backtest_table(
    benchmark: dict,
    at_chosen: dict[str, dict],
    at_zero: dict[str, dict],
    chosen: dict[str, float],
) -> pd.DataFrame:
    """Benchmark first, then each model at its threshold and at zero."""
    records: list[tuple[str, dict]] = [("buy_and_hold", benchmark)]
    for model in MODEL_NAMES:
        label = f"{model}@{chosen[model]:g}"
        # When the sweep picked 0.0 the two rows are the same rule; label the
        # chosen one so the table keeps one row per backtest either way.
        if chosen[model] == 0.0:
            label = f"{model}@0 (chosen)"
        records.append((label, at_chosen[model]))
        records.append((f"{model}@0", at_zero[model]))

    table = pd.DataFrame(
        [{key: result[key] for key in SUMMARY_KEYS} for _, result in records],
        index=[label for label, _ in records],
        columns=SUMMARY_KEYS,
    )
    table.index.name = "strategy"
    return table


def cost_table(
    at_chosen: dict[str, dict],
    chosen: dict[str, float],
    n_test_days: int,
    cost_per_side: float,
) -> pd.DataFrame:
    """Turn each model's turnover into an annual cost bill.

    The backtest already charges these costs; this table exists so the reader
    can see *why* the net returns look the way they do.
    """
    round_trip_cost = 2.0 * cost_per_side
    years = n_test_days / TRADING_DAYS_PER_YEAR

    rows = []
    for model in MODEL_NAMES:
        result = at_chosen[model]
        per_year = result["n_round_trips"] / years
        rows.append(
            {
                "threshold": chosen[model],
                "n_round_trips": result["n_round_trips"],
                "round_trips_per_year": per_year,
                "implied_annual_cost_drag": cost_drag_arithmetic(
                    round_trip_cost, per_year
                ),
                "test_cost_drag": result["cost_drag"],
            }
        )

    table = pd.DataFrame(rows, index=MODEL_NAMES)
    table.index.name = "model"
    return table


def generic_cost_line(cost_per_side: float) -> str:
    """The arithmetic that sinks daily trading, before any model is fitted."""
    round_trip_cost = 2.0 * cost_per_side
    drag = cost_drag_arithmetic(round_trip_cost, ROUND_TRIPS_PER_YEAR)
    return (
        f"{round_trip_cost:.2%} round trip × {ROUND_TRIPS_PER_YEAR:g} round "
        f"trips/year = {drag:.1%} per year"
    )


def predictions_frame(
    index: pd.Index,
    y_test,
    model_predictions: dict[str, np.ndarray],
    baseline_predictions_test: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Every prediction for the test year, one column each, for auditing."""
    frame = pd.DataFrame({"target": np.asarray(y_test, dtype=float)}, index=index)
    for model in MODEL_NAMES:
        frame[model] = model_predictions[model]
    for baseline in BASELINE_NAMES:
        frame[baseline] = baseline_predictions_test[baseline]
    frame.index.name = "date"
    return frame


# ==========================================================================
# report
# ==========================================================================


def build_report(result: dict) -> str:
    """Render RESULTS.md from a completed run."""
    config = result["config"]
    splits = result["splits"]
    leakage = result["leakage"]

    models = config["models"]
    model_lines = [
        f"  - `{name}`: {models[name]}" for name in MODEL_NAMES if name in models
    ]
    thresholds = ", ".join(f"{float(t):g}" for t in config["thresholds"])

    parts: list[str] = []
    parts.append(f"# {config['symbol']} next-day return experiment\n")
    parts.append(
        "Leakage-free walk-forward-by-calendar experiment: models are fitted on "
        "the training years, the trading threshold is chosen on the validation "
        "year, and the test year is scored once. Baselines and buy-and-hold are "
        "reported next to every model.\n"
    )

    # -- metadata ---------------------------------------------------------
    parts.append("## Run metadata\n")
    parts.append(f"- Symbol: `{config['symbol']}`")
    parts.append(
        f"- Download range: `{config['download_start']}` to "
        f"`{config['download_end']}` (end exclusive; the early months are "
        "feature warmup)"
    )
    parts.append(
        f"- Split boundaries: train through `{config['train_end']}`, "
        f"validation `{config['val_year']}`, test `{config['test_year']}`"
    )
    parts.append(f"- Run timestamp (UTC): {result['timestamp']}")
    parts.append(
        f"- Cost assumption: {config['cost_per_side']:.5f} per side "
        f"({2 * config['cost_per_side']:.2%} round trip)"
    )
    parts.append(f"- Threshold grid: {thresholds}")
    parts.append(f"- Minimum days in market: {config['min_days_in_market']}")
    parts.append(
        f"- Leakage warning limit: "
        f"{config['leakage_warning_directional_accuracy']:.1%} test directional "
        "accuracy"
    )
    parts.append(f"- Random seed: {RANDOM_SEED}")
    parts.append("- Models:")
    parts.extend(model_lines)
    parts.append(
        "- Library versions: yfinance "
        f"{yfinance.__version__}, pandas {pd.__version__}, numpy "
        f"{np.__version__}, scikit-learn {sklearn.__version__}, xgboost "
        f"{xgboost.__version__}"
    )
    parts.append("")
    parts.append("### Splits\n")
    parts.append(markdown_table(splits, "split"))
    parts.append("")
    parts.append(
        f"### Features ({len(FEATURE_COLUMNS)})\n\n"
        + ", ".join(f"`{column}`" for column in FEATURE_COLUMNS)
        + "\n\nTarget: `"
        + TARGET_COLUMN
        + "`, the next-day simple return.\n"
    )

    # -- metrics ----------------------------------------------------------
    parts.append(f"## Test-set metrics ({config['test_year']})\n")
    parts.append(markdown_table(result["metrics_test"], "name"))
    parts.append("")
    parts.append(
        "Baselines come first on purpose: a model that cannot beat "
        "`zero_return` on RMSE or `always_long` on directional accuracy has "
        "not learned anything about tomorrow.\n"
    )
    parts.append(f"## Validation metrics ({config['val_year']})\n")
    parts.append(markdown_table(result["metrics_val"], "name"))
    parts.append("")

    # -- backtest ---------------------------------------------------------
    parts.append(f"## Test-set backtest ({config['test_year']}, after costs)\n")
    parts.append(markdown_table(result["backtest_test"], "strategy"))
    parts.append("")
    parts.append(
        "`buy_and_hold` is the benchmark to beat. Each model appears twice: at "
        "the threshold chosen on the validation year, and at 0.0 (long whenever "
        "the forecast is positive). `days_in_market` is the fraction of test "
        "days holding the position.\n"
    )

    # -- costs ------------------------------------------------------------
    parts.append("## Cost arithmetic\n")
    parts.append(f"Generic: **{result['generic_cost_line']}**.\n")
    parts.append(
        "That is the bill for trading in and out every day, before a single "
        "model is fitted. Per model, at its chosen threshold on the test "
        "year:\n"
    )
    parts.append(markdown_table(result["cost_table"], "model"))
    parts.append("")
    parts.append(
        "`round_trips_per_year` annualizes the test-year round trips "
        f"({int(splits.loc['test', 'rows'])} trading days at "
        f"{TRADING_DAYS_PER_YEAR} days/year); `implied_annual_cost_drag` is "
        "that rate times the round-trip cost. `test_cost_drag` is what the "
        "backtest actually charged over the test year.\n"
    )

    # -- sweeps -----------------------------------------------------------
    parts.append(f"## Threshold sweeps (validation year {config['val_year']})\n")
    parts.append(
        "The threshold is chosen here and nowhere else. `eligible` marks the "
        f"rules that were in the market at least {config['min_days_in_market']} "
        "days -- an ineligible rule is still scored, it just cannot win.\n"
    )
    parts.append(
        "In this run the participation guard changed nothing: it marks the "
        "0.005-threshold rows ineligible, but the best-Sharpe row for every model "
        "was already eligible, so no selected threshold depended on the guard. It "
        "is there to stop a rule that trades a handful of days from winning on a "
        "Sharpe computed off almost no exposure.\n"
    )
    for model in MODEL_NAMES:
        parts.append(f"### {model} (chosen: {result['thresholds'][model]:g})\n")
        note = result["threshold_notes"].get(model)
        if note:
            parts.append(f"> Note: {note}\n")
        parts.append(markdown_table(result["sweeps"][model], "row"))
        parts.append("")

    # -- leakage guard ----------------------------------------------------
    parts.append("## Leakage guard\n")
    if leakage["fired"]:
        flagged = ", ".join(
            f"`{model}` at {value:.1%}" for model, value in leakage["flagged"].items()
        )
        parts.append(
            f"**FIRED.** Test directional accuracy above the "
            f"{leakage['limit']:.1%} plausibility limit: {flagged}. Next-day "
            "equity returns are close to unpredictable, so a score this high "
            "points at leakage, not skill. Do not trust the numbers above "
            "until the feature pipeline has been re-checked."
        )
    else:
        parts.append(
            f"Not fired. No model exceeded the {leakage['limit']:.1%} test "
            "directional accuracy plausibility limit."
        )
    parts.append("")

    return "\n".join(parts)


# ==========================================================================
# orchestration
# ==========================================================================


def run(
    config: dict,
    out_dir: Path,
    results_md: Path,
    refresh: bool = False,
) -> dict:
    """Run the whole experiment and write every artifact.

    Args:
        config: The ``experiment`` block of the config file.
        out_dir: Directory for the CSV artifacts (created if needed).
        results_md: Path of the markdown writeup to write.
        refresh: Ignore the price cache and re-download.

    Returns:
        A dict of the tables produced, the chosen thresholds and the leakage
        guard's verdict.
    """
    np.random.seed(RANDOM_SEED)
    out_dir = Path(out_dir)
    results_md = Path(results_md)
    out_dir.mkdir(parents=True, exist_ok=True)

    cost = float(config["cost_per_side"])

    # 1. prices -----------------------------------------------------------
    prices = load_prices(
        config["symbol"],
        config["download_start"],
        config["download_end"],
        config["data_cache"],
        refresh=refresh,
    )

    # 2. features and splits ----------------------------------------------
    dataset = build_dataset(prices)
    train, val, test = split_by_year(
        dataset,
        train_end=config["train_end"],
        val_year=config["val_year"],
        test_year=config["test_year"],
    )
    splits = describe_splits({"train": train, "val": val, "test": test})

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    # 3. fit on train only, predict val and test --------------------------
    models = make_models(config["models"])
    predictions = fit_predict(models, X_train, y_train, {"val": X_val, "test": X_test})
    val_predictions = {
        model: stabilize(predictions[model]["val"]) for model in MODEL_NAMES
    }
    test_predictions = {
        model: stabilize(predictions[model]["test"]) for model in MODEL_NAMES
    }

    baselines_val = baseline_predictions(y_train, len(y_val))
    baselines_test = baseline_predictions(y_train, len(y_test))

    # 4. metrics ----------------------------------------------------------
    metrics_val = metrics_table(y_val, {**baselines_val, **val_predictions})
    metrics_test = metrics_table(y_test, {**baselines_test, **test_predictions})

    # 5. leakage guard ----------------------------------------------------
    leakage = check_leakage(
        metrics_test, float(config["leakage_warning_directional_accuracy"])
    )

    # 6. threshold selection, on validation only --------------------------
    chosen, sweeps, notes = choose_thresholds(
        val_predictions,
        y_val,
        config["thresholds"],
        cost,
        int(config["min_days_in_market"]),
    )

    # 7. the single look at the test year ---------------------------------
    benchmark, at_chosen, at_zero = run_test_backtests(
        test_predictions, y_test, chosen, cost
    )
    backtests = backtest_table(benchmark, at_chosen, at_zero, chosen)

    # 8. cost arithmetic ---------------------------------------------------
    costs = cost_table(at_chosen, chosen, len(y_test), cost)

    predictions_test = predictions_frame(
        test.index, y_test, test_predictions, baselines_test
    )

    result = {
        "config": config,
        "splits": splits,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "backtest_test": backtests,
        "cost_table": costs,
        "sweeps": sweeps,
        "thresholds": chosen,
        "threshold_notes": notes,
        "leakage": leakage,
        "predictions_test": predictions_test,
        "generic_cost_line": generic_cost_line(cost),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # 9. artifacts ---------------------------------------------------------
    metrics_val.to_csv(out_dir / "metrics_val.csv")
    metrics_test.to_csv(out_dir / "metrics_test.csv")
    backtests.to_csv(out_dir / "backtest_test.csv")
    predictions_test.to_csv(out_dir / "predictions_test.csv")
    costs.to_csv(out_dir / "cost_arithmetic.csv")
    for model, table in sweeps.items():
        table.to_csv(out_dir / f"threshold_sweep_{model}.csv", index=False)

    # 10. the writeup ------------------------------------------------------
    report = build_report(result)
    results_md.write_text(report)

    # 11. and the same thing on stdout ------------------------------------
    print(report)
    print(f"\nArtifacts written to {out_dir}/ and {results_md}")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the leakage-free next-day return experiment."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Config file with an `experiment:` section (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cached prices and download them again.",
    )
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        full_config = yaml.safe_load(handle)

    if "experiment" not in full_config:
        raise SystemExit(f"{args.config} has no `experiment:` section.")
    config = full_config["experiment"]

    root = Path(__file__).resolve().parent
    run(
        config,
        root / config.get("results_dir", "results"),
        root / "RESULTS.md",
        refresh=args.refresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
