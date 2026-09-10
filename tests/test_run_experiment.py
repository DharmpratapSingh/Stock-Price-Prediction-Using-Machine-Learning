"""
Smoke test for the end-to-end experiment runner.

The runner's job is wiring, not arithmetic -- every number it reports is
computed by a module that already has its own unit tests. So this test asserts
the wiring holds: the artifacts appear, the baselines are reported next to the
models, the buy-and-hold benchmark and the cost arithmetic reach the writeup,
and running twice on the same inputs produces the same numbers.

No network: ``run_experiment.load_prices`` is monkeypatched with a synthetic
random walk covering the same calendar span as the real run.
"""

import numpy as np
import pandas as pd
import pytest

import run_experiment
from src.baselines import BASELINE_NAMES
from src.lean_models import MODEL_NAMES

# Small forest / booster: this test is about plumbing, and a 300-tree forest
# would make it slow without making it stricter.
TINY_CONFIG = {
    "symbol": "TEST",
    "download_start": "2017-10-01",
    "download_end": "2025-01-01",
    "train_end": "2022-12-31",
    "val_year": 2023,
    "test_year": 2024,
    # Kept at the real value so the generic cost line reads "37.5% per year".
    "cost_per_side": 0.00075,
    "thresholds": [0.0, 0.001],
    "min_days_in_market": 5,
    "leakage_warning_directional_accuracy": 0.56,
    "models": {
        "ridge": {"alpha": 10.0},
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 4,
            "min_samples_leaf": 20,
            "random_state": 42,
        },
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
}


def synthetic_prices() -> pd.DataFrame:
    """A seeded business-day OHLCV random walk over the real date span."""
    dates = pd.bdate_range("2017-10-01", "2024-12-31", name="Date")
    rng = np.random.default_rng(20240101)

    returns = rng.normal(0.0005, 0.02, len(dates))
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0.0, 0.01, len(dates))) * close

    return pd.DataFrame(
        {
            "Open": close - rng.normal(0.0, 0.005, len(dates)) * close,
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": rng.lognormal(16.0, 0.4, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def offline_prices(monkeypatch):
    """Replace the loader so the runner never touches the network."""
    prices = synthetic_prices()

    def fake_load_prices(symbol, start, end, cache_path, refresh=False, **kwargs):
        return prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(end))]

    monkeypatch.setattr(run_experiment, "load_prices", fake_load_prices)
    return prices


def config_for(tmp_path) -> dict:
    config = dict(TINY_CONFIG)
    config["data_cache"] = str(tmp_path / "prices.csv")
    config["results_dir"] = str(tmp_path / "results")
    return config


def test_run_writes_every_artifact_and_is_reproducible(tmp_path, offline_prices):
    out_dir = tmp_path / "results"
    results_md = tmp_path / "RESULTS.md"

    result = run_experiment.run(config_for(tmp_path), out_dir, results_md)

    # -- the writeup ------------------------------------------------------
    assert results_md.exists()
    report = results_md.read_text()
    for name in BASELINE_NAMES + MODEL_NAMES:
        assert name in report, f"{name} missing from RESULTS.md"
    assert "buy_and_hold" in report
    # The generic cost arithmetic: 0.15% round trip x 250 round trips a year.
    assert "37.5%" in report

    # -- the metrics table ------------------------------------------------
    metrics_test = pd.read_csv(out_dir / "metrics_test.csv", index_col=0)
    assert list(metrics_test.index) == BASELINE_NAMES + MODEL_NAMES

    # -- the backtest table: benchmark, each model at its own threshold,
    #    and each model again at zero -------------------------------------
    backtest = pd.read_csv(out_dir / "backtest_test.csv", index_col=0)
    assert len(backtest) == 1 + 2 * len(MODEL_NAMES)

    # -- the return value -------------------------------------------------
    assert set(result["thresholds"]) == set(MODEL_NAMES)
    assert all(np.isfinite(value) for value in result["thresholds"].values())

    # -- determinism: same inputs, same numbers ---------------------------
    first = (out_dir / "metrics_test.csv").read_text()
    rerun_dir = tmp_path / "results_again"
    run_experiment.run(config_for(tmp_path), rerun_dir, tmp_path / "RESULTS_again.md")
    assert (rerun_dir / "metrics_test.csv").read_text() == first


def test_other_artifacts_are_written(tmp_path, offline_prices):
    out_dir = tmp_path / "results"
    run_experiment.run(config_for(tmp_path), out_dir, tmp_path / "RESULTS.md")

    assert (out_dir / "metrics_val.csv").exists()
    assert (out_dir / "predictions_test.csv").exists()
    for model in MODEL_NAMES:
        assert (out_dir / f"threshold_sweep_{model}.csv").exists()

    predictions = pd.read_csv(out_dir / "predictions_test.csv", index_col=0)
    assert list(predictions.columns) == ["target"] + MODEL_NAMES + BASELINE_NAMES
