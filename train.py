"""
Walk-forward training and evaluation pipeline.

Runs the full honest evaluation for one ticker or the whole basket:

  1. Load daily OHLCV from the committed snapshots.
  2. Engineer features (every value at time t uses data <= t).
  3. Build expanding walk-forward folds with an embargo equal to the feature
     warm-up length.
  4. For every fold, fit each model and baseline on that fold's training rows
     only -- imputation, winsorisation, scaling and feature selection all inside
     the pipeline -- and predict the held-out rows.
  5. Score return regression, direction, and a cost-aware long/flat backtest
     against buy-and-hold, plus a cost-sensitivity sweep.
  6. Reproduce the level-R2 trap on the same folds.
  7. Write every table and figure under results/, and save one artifact
     (fitted pipeline + feature list + config) that predict.py consumes.

Usage:
    python train.py                      # NVDA, 2018-2024
    python train.py --ticker SPY
    python train.py --basket             # SPY, AAPL, MSFT, NVDA, JPM
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.backtesting import Backtester
from src.data_loader import load_stock_data
from src.evaluation import (
    adjust_proportion_for_clustering,
    design_effect,
    direction_metrics,
    return_regression_metrics,
    run_walk_forward,
)
from src.feature_engineering import (
    build_dataset,
    feature_columns,
    feature_warmup_length,
)
from src.feature_selection import analyze_feature_correlation
from src.models import build_pipeline, get_baseline, get_feature_importance
from src.utils import (
    folds_to_frame,
    load_config,
    save_artifact,
    setup_logging,
    walk_forward_folds,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "linear": "Ridge / Logistic",
    "forest": "Random Forest",
    "boosting": "XGBoost",
}
REGRESSION_BASELINES = {
    "zero": "Baseline: zero return",
    "train_mean": "Baseline: train-window mean",
    "persistence": "Baseline: yesterday's return",
}
DIRECTION_BASELINE = {"always_up": "Baseline: always up"}


# ----------------------------------------------------------------------
# Per-ticker evaluation
# ----------------------------------------------------------------------

def evaluate_ticker(symbol: str, config: Dict, start_date: str = None) -> Dict:
    """
    Run the whole walk-forward evaluation for one ticker.

    Args:
        symbol: Ticker symbol
        config: Full project config
        start_date: Override the config start date. The basket run passes
            ``basket_start_date`` so every ticker covers an identical window and
            the per-ticker rows stay comparable.

    Returns:
        Dict with per-model out-of-sample predictions, metrics and fold table
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    fs_cfg = config.get("feature_selection", {})

    raw = load_stock_data(
        symbol=symbol,
        start_date=start_date or data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        snapshot_dir=data_cfg.get("snapshot_dir", "data/raw"),
        allow_download=data_cfg.get("allow_download", False),
    )
    logger.info("%s: %d rows, %s to %s", symbol, len(raw),
                raw.index[0].date(), raw.index[-1].date())

    horizon = data_cfg.get("horizon", 1)
    stationary_only = config["features"].get("stationary_only", True)

    dataset, model_features = build_dataset(
        raw, config["features"], horizon=horizon, stationary_only=stationary_only
    )
    level_features = feature_columns(dataset, stationary_only=False)

    # Embargo = deepest feature lookback + the forecast horizon. The lookback term
    # keeps any test feature row from being computed out of a training row; the
    # horizon term accounts for the last training row's target reaching forward
    # h bars, so the raw bars behind train and test are strictly disjoint.
    embargo = train_cfg.get("embargo", "auto")
    if embargo == "auto":
        embargo = feature_warmup_length(raw, config["features"]) + horizon

    folds = walk_forward_folds(
        n_samples=len(dataset),
        initial_train=train_cfg.get("initial_train", 504),
        test_size=train_cfg.get("test_size", 63),
        step_size=train_cfg.get("step_size", 63),
        embargo=int(embargo),
        expanding=train_cfg.get("expanding", True),
    )
    logger.info(
        "%s: %d features, %d rows, %d folds, embargo %d",
        symbol, len(model_features), len(dataset), len(folds), embargo
    )

    k = fs_cfg.get("top_k", 40) if fs_cfg.get("enabled", True) else None
    k = min(k, len(model_features)) if k else None
    winsorize = fs_cfg.get("winsorize", True)

    predictions: Dict[str, pd.DataFrame] = {}

    # --- Return regression: three model families -----------------------
    for family in config["models"]["families"]:
        predictions[f"reg::{family}"] = run_walk_forward(
            dataset, model_features, "target_logret",
            lambda f=family: build_pipeline(f, "regression", k, winsorize),
            folds, task="regression",
        )

    # --- Return regression: baselines ----------------------------------
    for name in REGRESSION_BASELINES:
        predictions[f"reg::{name}"] = run_walk_forward(
            dataset, model_features, "target_logret",
            lambda n=name: get_baseline(n, "regression"),
            folds, task="regression",
        )

    # --- Direction classification --------------------------------------
    for family in config["models"]["families"]:
        predictions[f"clf::{family}"] = run_walk_forward(
            dataset, model_features, "target_direction",
            lambda f=family: build_pipeline(f, "classification", k, winsorize),
            folds, task="classification",
        )
    predictions["clf::always_up"] = run_walk_forward(
        dataset, model_features, "target_direction",
        lambda: get_baseline("always_up", "classification"),
        folds, task="classification",
    )

    # --- Level-R2 trap: same folds, price levels instead of returns ----
    level_predictions = {}
    for family in ("linear", "forest"):
        level_predictions[family] = run_walk_forward(
            dataset, level_features, "target_price",
            lambda f=family: build_pipeline(f, "regression", None, False),
            folds, task="regression",
        )
    # Persistence in level space: forecast tomorrow's close as today's close.
    test_index = level_predictions["linear"].index
    level_predictions["persistence"] = pd.DataFrame(
        {
            "y_true": dataset.loc[test_index, "target_price"].to_numpy(),
            "y_pred": dataset.loc[test_index, "Close"].to_numpy(),
        },
        index=test_index,
    )

    return {
        "symbol": symbol,
        "dataset": dataset,
        "model_features": model_features,
        "folds": folds,
        "fold_table": folds_to_frame(folds, dataset.index),
        "embargo": int(embargo),
        "predictions": predictions,
        "level_predictions": level_predictions,
    }


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def score_returns(result: Dict) -> pd.DataFrame:
    """Return-regression metrics for every model and baseline of one ticker."""
    rows = []
    labels = {**{f: MODEL_LABELS[f] for f in MODEL_LABELS}, **REGRESSION_BASELINES}

    for key, frame in result["predictions"].items():
        kind, name = key.split("::")
        if kind != "reg":
            continue
        metrics = return_regression_metrics(frame["y_true"], frame["y_pred"])
        rows.append({
            "ticker": result["symbol"],
            "model": labels.get(name, name),
            "is_baseline": name in REGRESSION_BASELINES,
            **metrics,
        })
    return pd.DataFrame(rows)


def score_direction(result: Dict, confidence: float = 0.95) -> pd.DataFrame:
    """
    Directional metrics for every classifier and the always-up baseline.

    The reference comparison is paired: the always-up baseline's labels on the
    same rows are handed to direction_metrics so it can run McNemar's test rather
    than an unpaired binomial against the baseline's rate.
    """
    rows = []
    labels = {**{f: MODEL_LABELS[f] for f in MODEL_LABELS}, **DIRECTION_BASELINE}

    baseline_frame = result["predictions"]["clf::always_up"]
    baseline_truth = baseline_frame["y_true"].to_numpy()
    baseline_accuracy = float(
        (baseline_frame["y_pred"].to_numpy() == baseline_truth).mean()
    )

    for key, frame in result["predictions"].items():
        kind, name = key.split("::")
        if kind != "clf":
            continue

        is_baseline = name in DIRECTION_BASELINE
        metrics = direction_metrics(
            frame["y_true"].to_numpy(),
            frame["y_pred"].to_numpy(),
            reference_rate=baseline_accuracy,
            # Comparing the baseline with itself carries no information.
            reference_prediction=None if is_baseline
            else baseline_frame["y_pred"].to_numpy(),
            confidence=confidence,
        )
        rows.append({
            "ticker": result["symbol"],
            "model": labels.get(name, name),
            "is_baseline": is_baseline,
            **metrics,
        })
    return pd.DataFrame(rows)


def score_backtest(result: Dict, config: Dict) -> pd.DataFrame:
    """Long/flat backtest of each return model against buy-and-hold."""
    bt_cfg = config["backtesting"]
    backtester = Backtester(
        initial_capital=bt_cfg["initial_capital"],
        commission=bt_cfg["commission"],
        slippage=bt_cfg["slippage"],
    )
    threshold = bt_cfg.get("threshold", 0.0)
    labels = {**{f: MODEL_LABELS[f] for f in MODEL_LABELS}, **REGRESSION_BASELINES}

    rows = []
    reference = next(iter(result["predictions"].values()))
    actual = reference["y_true"].to_numpy()
    dates = reference.index

    benchmark = backtester.buy_and_hold_returns(actual, dates)
    rows.append({
        "ticker": result["symbol"], "model": "Buy & Hold", "is_baseline": True,
        **_backtest_row(benchmark),
    })

    for key, frame in result["predictions"].items():
        kind, name = key.split("::")
        if kind != "reg":
            continue
        run = backtester.long_flat_backtest(
            frame["y_pred"].to_numpy(), frame["y_true"].to_numpy(),
            frame.index, threshold,
        )
        rows.append({
            "ticker": result["symbol"],
            "model": labels.get(name, name),
            "is_baseline": name in REGRESSION_BASELINES,
            **_backtest_row(run),
            "excess_vs_buy_hold_pct":
                run["metrics"]["total_return"] - benchmark["metrics"]["total_return"],
        })

    return pd.DataFrame(rows)


def _backtest_row(run: Dict) -> Dict:
    metrics = run["metrics"]
    return {
        "total_return_pct": metrics["total_return"],
        "annualized_return_pct": metrics["annualized_return"],
        # sharpe = (CAGR - rf) / annualised vol; sharpe_arithmetic is the textbook
        # mean-excess-return form. Both reported; docs quote `sharpe`.
        "sharpe": metrics["sharpe_ratio"],
        "sharpe_arithmetic": metrics["sharpe_arithmetic"],
        "max_drawdown_pct": metrics["max_drawdown"],
        "turnover": metrics.get("turnover", np.nan),
        "time_in_market": metrics.get("time_in_market", np.nan),
        "n_periods": metrics["n_periods"],
    }


def score_level_trap(result: Dict) -> pd.DataFrame:
    """
    The level-R2 trap: the same models scored on price levels and on returns.

    Level R2 near 0.99 is the autocorrelation of the price series, not skill --
    the persistence forecast gets it for free.
    """
    rows = []
    label = {"linear": "Ridge", "forest": "Random Forest",
             "persistence": "Persistence: yhat(t+1) = Close(t)"}

    return_r2 = {}
    for key, frame in result["predictions"].items():
        kind, name = key.split("::")
        if kind == "reg" and name in ("linear", "forest", "persistence"):
            return_r2[name] = return_regression_metrics(
                frame["y_true"], frame["y_pred"]
            )["R2"]

    for name, frame in result["level_predictions"].items():
        level = return_regression_metrics(frame["y_true"], frame["y_pred"])
        rows.append({
            "ticker": result["symbol"],
            "model": label.get(name, name),
            "level_R2": level["R2"],
            "level_RMSE_usd": level["RMSE"],
            "return_R2": return_r2.get(name, np.nan),
            "n": level["n"],
        })
    return pd.DataFrame(rows)


def pool_predictions(results: List[Dict], prefix: str) -> Dict[str, pd.DataFrame]:
    """Concatenate out-of-sample predictions across tickers, per model."""
    pooled: Dict[str, List[pd.DataFrame]] = {}
    for result in results:
        for key, frame in result["predictions"].items():
            if key.startswith(prefix):
                pooled.setdefault(key, []).append(frame)
    return {k: pd.concat(v) for k, v in pooled.items()}


def correctness_panel(results: List[Dict], model_key: str) -> pd.DataFrame:
    """
    Build a date x ticker panel of 0/1 correctness for one direction model.

    Each column is one ticker's out-of-sample hit/miss series. Because every
    ticker is scored on the same calendar dates, the columns align and their
    correlation is what inflates the pooled interval.
    """
    columns = {}
    for result in results:
        frame = result["predictions"][model_key]
        columns[result["symbol"]] = pd.Series(
            (frame["y_pred"].to_numpy() == frame["y_true"].to_numpy()).astype(float),
            index=frame.index,
        )
    return pd.DataFrame(columns)


def score_pooled_dependence(results: List[Dict]) -> pd.DataFrame:
    """
    Quantify how much less information the pooled panel carries than n suggests.

    Reports the design effect for each direction model's correctness panel, plus
    two context rows: the realised up/down indicator and the realised log return,
    whose cross-ticker correlation is the underlying cause.
    """
    rows = []

    for key in sorted(k for k in results[0]["predictions"] if k.startswith("clf::")):
        name = key.split("::")[1]
        panel = correctness_panel(results, key)
        rows.append({
            "quantity": f"correctness indicator: {name}",
            **design_effect(panel),
        })

    # Context: the shared market moves that drive the correlation above.
    reference = {r["symbol"]: r["predictions"]["clf::always_up"] for r in results}
    direction_panel = pd.DataFrame(
        {s: pd.Series(f["y_true"].to_numpy(), index=f.index) for s, f in reference.items()}
    )
    rows.append({"quantity": "realised up/down indicator", **design_effect(direction_panel)})

    return_panel = pd.DataFrame({
        r["symbol"]: pd.Series(
            r["predictions"]["reg::zero"]["y_true"].to_numpy(),
            index=r["predictions"]["reg::zero"].index,
        )
        for r in results
    })
    rows.append({"quantity": "realised log return", **design_effect(return_panel)})

    return pd.DataFrame(rows)


def score_pooled(
    results: List[Dict],
    config: Dict,
    confidence: float = 0.95
) -> Dict[str, pd.DataFrame]:
    """
    Score every model on all tickers' out-of-sample rows pooled together.

    Pooled direction rows carry design-effect-adjusted intervals and p-values
    alongside the iid ones. The tickers share calendar dates -- and SPY contains
    the other four outright -- so an iid interval over the pooled panel is too
    narrow. The point estimates are unaffected; only the uncertainty changes.
    """
    labels = {**{f: MODEL_LABELS[f] for f in MODEL_LABELS},
              **REGRESSION_BASELINES, **DIRECTION_BASELINE}

    reg_rows = []
    for key, frame in pool_predictions(results, "reg::").items():
        name = key.split("::")[1]
        reg_rows.append({
            "ticker": "POOLED", "model": labels.get(name, name),
            "is_baseline": name in REGRESSION_BASELINES,
            **return_regression_metrics(frame["y_true"], frame["y_pred"]),
        })

    pooled_baseline = pool_predictions(results, "clf::")["clf::always_up"]
    baseline_truth = pooled_baseline["y_true"].to_numpy()
    baseline_accuracy = float((pooled_baseline["y_pred"].to_numpy() == baseline_truth).mean())

    dir_rows = []
    for key, frame in pool_predictions(results, "clf::").items():
        name = key.split("::")[1]
        is_baseline = name in DIRECTION_BASELINE

        metrics = direction_metrics(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(),
            reference_rate=baseline_accuracy,
            reference_prediction=None if is_baseline
            else pooled_baseline["y_pred"].to_numpy(),
            confidence=confidence,
        )

        if len(results) > 1:
            effect = design_effect(correctness_panel(results, key))
            metrics.update({
                "design_effect": effect["design_effect"],
                "mean_pairwise_corr": effect["mean_pairwise_corr"],
                **adjust_proportion_for_clustering(
                    metrics["accuracy"], effect["n_effective"],
                    reference_rate=baseline_accuracy, confidence=confidence,
                ),
            })

        dir_rows.append({
            "ticker": "POOLED", "model": labels.get(name, name),
            "is_baseline": is_baseline, **metrics,
        })

    return {
        "returns": pd.DataFrame(reg_rows),
        "direction": pd.DataFrame(dir_rows),
    }


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def to_markdown(frame: pd.DataFrame, precision: int = 4) -> str:
    """
    Render a DataFrame as a GitHub-flavoured Markdown table.

    Written by hand rather than via ``DataFrame.to_markdown`` so the results
    files do not depend on the optional ``tabulate`` package.
    """
    def cell(value) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else ""
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            if value != 0 and abs(value) < 10 ** -precision:
                return f"{value:.2e}"
            return f"{value:.{precision}f}"
        return str(value)

    header = list(frame.columns)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, name: str, results_dir: Path) -> None:
    """Write one results table as both CSV and Markdown."""
    frame.to_csv(results_dir / f"{name}.csv", index=False)
    (results_dir / f"{name}.md").write_text(to_markdown(frame))


def make_figures(results: List[Dict], config: Dict, results_dir: Path) -> None:
    """Equity curves versus buy-and-hold, and the level-R2 trap illustration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt_cfg = config["backtesting"]
    backtester = Backtester(
        initial_capital=bt_cfg["initial_capital"],
        commission=bt_cfg["commission"],
        slippage=bt_cfg["slippage"],
    )
    threshold = bt_cfg.get("threshold", 0.0)

    # --- Figure 1: walk-forward equity curves --------------------------
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    for ax, result in zip(axes.ravel(), results):
        reference = result["predictions"]["reg::linear"]
        dates, actual = reference.index, reference["y_true"].to_numpy()
        benchmark = backtester.buy_and_hold_returns(actual, dates)
        ax.plot(dates, benchmark["equity_curve"][1:], label="Buy & hold",
                color="#444444", linewidth=1.8)

        for family, colour in zip(config["models"]["families"],
                                  ["#1f77b4", "#2ca02c", "#d62728"]):
            frame = result["predictions"][f"reg::{family}"]
            run = backtester.long_flat_backtest(
                frame["y_pred"].to_numpy(), frame["y_true"].to_numpy(),
                frame.index, threshold,
            )
            ax.plot(dates, run["equity_curve"][1:], label=MODEL_LABELS[family],
                    linewidth=1.2, alpha=0.85, color=colour)

        ax.set_title(f"{result['symbol']} - walk-forward equity, net of 15 bps per side")
        ax.set_ylabel("Equity ($)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(results_dir / "equity_curves.png", dpi=120)
    plt.close(fig)

    # --- Figure 2: the level-R2 trap -----------------------------------
    trap = pd.concat([score_level_trap(r) for r in results])
    summary = trap.groupby("model")[["level_R2", "return_R2"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    summary["level_R2"].plot.barh(ax=axes[0], color="#c44e52")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title("R² predicting the price LEVEL C(t+1)")
    axes[0].set_xlabel("R²")
    axes[0].axvline(0.99, ls="--", lw=1, color="#333333")
    for i, v in enumerate(summary["level_R2"]):
        axes[0].text(min(v, 1.0) - 0.02, i, f"{v:.4f}", va="center",
                     ha="right", color="white", fontsize=9)

    summary["return_R2"].plot.barh(ax=axes[1], color="#4c72b0")
    axes[1].set_xlim(-0.1, 0.1)
    axes[1].set_title("R² predicting the next-day RETURN")
    axes[1].set_xlabel("R²")
    axes[1].axvline(0, ls="-", lw=1, color="#333333")
    for i, v in enumerate(summary["return_R2"]):
        if not np.isnan(v):
            axes[1].text(v, i, f" {v:+.4f}", va="center", fontsize=9)

    fig.suptitle("Same models, same folds: the level score is the price "
                 "autocorrelation, not skill", fontsize=11)
    fig.tight_layout()
    fig.savefig(results_dir / "level_r2_trap.png", dpi=120)
    plt.close(fig)

    # --- Figure 3: directional accuracy with confidence intervals ------
    combined = pd.concat([score_direction(r) for r in results])
    if len(results) > 1:
        combined = pd.concat([combined, score_pooled(results, config)["direction"]])

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(combined) + 2))
    labels = [f"{r.ticker} - {r.model}" for r in combined.itertuples()]
    y = np.arange(len(combined))
    ax.errorbar(
        combined["accuracy"] * 100, y,
        xerr=[(combined["accuracy"] - combined["ci_lower"]) * 100,
              (combined["ci_upper"] - combined["accuracy"]) * 100],
        fmt="o", markersize=4, capsize=3, linewidth=1,
        color="#1f77b4", ecolor="#999999",
    )
    ax.axvline(50, color="#c44e52", ls="--", lw=1.2, label="Coin flip (50%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Directional accuracy (%), 95% Wilson interval")
    ax.set_title("Every interval that straddles 50% is a result "
                 "indistinguishable from chance")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "directional_accuracy.png", dpi=120)
    plt.close(fig)


def fit_final_artifact(result: Dict, config: Dict, models_dir: Path) -> str:
    """
    Fit one pipeline on all available rows and save the inference artifact.

    predict.py loads exactly this: fitted pipeline plus the feature list and the
    config that produced it, so inference cannot build a different feature set
    from the one the model was fitted on.
    """
    fs_cfg = config.get("feature_selection", {})
    features = result["model_features"]
    k = fs_cfg.get("top_k", 40) if fs_cfg.get("enabled", True) else None
    k = min(k, len(features)) if k else None

    family = config["models"]["families"][0]
    dataset = result["dataset"]

    pipeline = build_pipeline(family, "regression", k, fs_cfg.get("winsorize", True))
    pipeline.fit(dataset[features], dataset["target_logret"])

    direction_pipeline = build_pipeline(
        family, "classification", k, fs_cfg.get("winsorize", True)
    )
    direction_pipeline.fit(dataset[features], dataset["target_direction"])

    models_dir.mkdir(parents=True, exist_ok=True)
    path = str(models_dir / f"{result['symbol']}_{family}.joblib")

    save_artifact(
        {
            "pipeline": pipeline,
            "direction_pipeline": direction_pipeline,
            "feature_columns": features,
            "config": config,
            "family": family,
            "symbol": result["symbol"],
            "target": "target_logret",
            "horizon": config["data"].get("horizon", 1),
            "trained_through": str(dataset.index[-1].date()),
            "n_train_rows": len(dataset),
        },
        path,
    )
    logger.info("Saved inference artifact to %s", path)
    return path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(
    config_path: str = "config/config.yaml",
    ticker: str = None,
    basket: bool = False,
    download: bool = False,
    figures: bool = True,
    results_dir: str = None,
) -> None:
    config = load_config(config_path)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    if download:
        config["data"]["allow_download"] = True

    results_dir = Path(results_dir or config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if basket:
        symbols = config["data"]["tickers"]
        start_date = config["data"].get("basket_start_date", config["data"]["start_date"])
    elif ticker:
        symbols = [ticker.upper()]
        start_date = config["data"]["start_date"]
    else:
        symbols = [config["data"]["symbol"]]
        start_date = config["data"]["start_date"]

    logger.info("Evaluating %s from %s", ", ".join(symbols), start_date)

    results = [evaluate_ticker(symbol, config, start_date) for symbol in symbols]

    # --- Assemble tables ----------------------------------------------
    confidence = config.get("evaluation", {}).get("confidence", 0.95)

    # A pooled row only says something new when more than one ticker ran.
    per_ticker_returns = pd.concat([score_returns(r) for r in results])
    per_ticker_direction = pd.concat([score_direction(r, confidence) for r in results])

    dependence_table = None
    if len(results) > 1:
        pooled = score_pooled(results, config, confidence)
        returns_table = pd.concat([per_ticker_returns, pooled["returns"]])
        direction_table = pd.concat([per_ticker_direction, pooled["direction"]])
        dependence_table = score_pooled_dependence(results)
    else:
        returns_table, direction_table = per_ticker_returns, per_ticker_direction

    returns_table = returns_table.reset_index(drop=True)
    direction_table = direction_table.reset_index(drop=True)
    backtest_table = pd.concat(
        [score_backtest(r, config) for r in results]
    ).reset_index(drop=True)
    trap_table = pd.concat([score_level_trap(r) for r in results]).reset_index(drop=True)
    fold_table = pd.concat(
        [r["fold_table"].assign(ticker=r["symbol"]) for r in results]
    ).reset_index(drop=True)

    # Cost sensitivity, pooled across tickers for the best-Sharpe model family.
    bt_cfg = config["backtesting"]
    backtester = Backtester(
        initial_capital=bt_cfg["initial_capital"],
        commission=bt_cfg["commission"],
        slippage=bt_cfg["slippage"],
    )
    cost_rows = []
    for result in results:
        for family in config["models"]["families"]:
            frame = result["predictions"][f"reg::{family}"]
            sweep = backtester.cost_sensitivity(
                frame["y_pred"].to_numpy(), frame["y_true"].to_numpy(),
                frame.index, bt_cfg.get("cost_grid_bps", [0, 5, 10, 20]),
                bt_cfg.get("threshold", 0.0),
            )
            sweep.insert(0, "model", MODEL_LABELS[family])
            sweep.insert(0, "ticker", result["symbol"])
            cost_rows.append(sweep)
    cost_table = pd.concat(cost_rows).reset_index(drop=True)

    tables = [
        (returns_table, "return_regression"),
        (direction_table, "direction"),
        (backtest_table, "backtest"),
        (cost_table, "cost_sensitivity"),
        (trap_table, "level_r2_trap"),
        (fold_table, "walk_forward_folds"),
    ]
    if dependence_table is not None:
        tables.append((dependence_table, "pooled_dependence"))

    for frame, name in tables:
        write_table(frame, name, results_dir)

    # --- Feature importance, fitted on training rows only -----------------
    # Fitting this on the full sample would mix out-of-sample rows into the
    # reported importances, so it is pinned to the final fold's training window.
    last = results[-1]
    fs_cfg = config.get("feature_selection", {})
    k = fs_cfg.get("top_k", 40) if fs_cfg.get("enabled", True) else None
    k = min(k, len(last["model_features"])) if k else None

    final_fold = last["folds"][-1]
    importance_window = last["dataset"].iloc[final_fold.train_start:final_fold.train_end]
    pipeline = build_pipeline("boosting", "regression", k, fs_cfg.get("winsorize", True))
    pipeline.fit(importance_window[last["model_features"]], importance_window["target_logret"])

    importance = get_feature_importance(pipeline, last["model_features"])
    importance.insert(0, "ticker", last["symbol"])
    importance.insert(
        1, "window",
        f"fold {final_fold.index} train rows only (n={len(importance_window)})",
    )
    write_table(importance.head(25), "feature_importance", results_dir)

    # Feature/target correlation, computed on the FIRST fold's training rows only.
    # Running this on the full dataset would be the leak the rebuild removed, so
    # the window is pinned explicitly.
    first_fold = last["folds"][0]
    train_window = last["dataset"].iloc[first_fold.train_start:first_fold.train_end]
    correlation = analyze_feature_correlation(
        train_window[last["model_features"] + ["target_logret"]],
        target_col="target_logret",
        top_n=25,
    )
    correlation.insert(0, "ticker", last["symbol"])
    correlation.insert(1, "window", f"fold 0 train rows only (n={len(train_window)})")
    write_table(correlation, "feature_target_correlation", results_dir)

    if figures:
        logger.info("Rendering figures")
        make_figures(results, config, results_dir)

    artifact_path = fit_final_artifact(results[-1], config, Path(config["paths"]["models_dir"]))

    # --- Console summary ------------------------------------------------
    print("\n" + "=" * 78)
    print("RETURN REGRESSION (out-of-sample log returns)")
    print("=" * 78)
    print(returns_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n" + "=" * 78)
    print("DIRECTION (p_vs_reference = McNemar against the always-up baseline)")
    print("=" * 78)
    columns = ["ticker", "model", "n", "accuracy", "ci_lower", "ci_upper",
               "p_vs_0.5", "p_vs_reference", "precision_up", "recall_up"]
    print(direction_table[[c for c in columns if c in direction_table]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if dependence_table is not None:
        print("\n" + "=" * 78)
        print("POOLED PANEL DEPENDENCE (iid intervals would be too narrow)")
        print("=" * 78)
        print(dependence_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nDesign-effect-adjusted pooled direction:")
        adjusted = direction_table[direction_table["ticker"] == "POOLED"]
        adj_cols = ["model", "accuracy", "n", "n_effective", "design_effect",
                    "adj_ci_lower", "adj_ci_upper", "adj_p_vs_0.5"]
        print(adjusted[[c for c in adj_cols if c in adjusted]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print("BACKTEST vs BUY & HOLD (net of 15 bps per side)")
    print("=" * 78)
    print(backtest_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("LEVEL-R2 TRAP")
    print("=" * 78)
    print(trap_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print(f"\nTables and figures written to {results_dir}/")
    print(f"Inference artifact: {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-forward training and evaluation for next-day return prediction"
    )
    parser.add_argument("--config", default="config/config.yaml", help="Config file")
    parser.add_argument("--ticker", default=None, help="Single ticker to evaluate")
    parser.add_argument("--basket", action="store_true",
                        help="Evaluate the full basket from config data.tickers")
    parser.add_argument("--download", action="store_true",
                        help="Allow refreshing snapshots from Yahoo Finance")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    parser.add_argument("--results-dir", default=None,
                        help="Override the output directory (default: paths.results_dir)")

    args = parser.parse_args()
    main(
        config_path=args.config,
        ticker=args.ticker,
        basket=args.basket,
        download=args.download,
        figures=not args.no_figures,
        results_dir=args.results_dir,
    )
