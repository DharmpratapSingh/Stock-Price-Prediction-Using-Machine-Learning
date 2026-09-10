"""
Property tests for the leakage-critical parts of the pipeline.

These are the tests that matter most in this repository. Each one pins down a
property that, if it silently broke, would turn an honest near-zero result into a
fake good one:

  * features at time t use only data <= t (no lookahead)
  * walk-forward folds run forward in time with a real embargo
  * train.py and predict.py build an identical feature matrix
  * the backtest trades on forecast-versus-spot, as documented
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import Backtester
from src.evaluation import run_walk_forward
from src.feature_engineering import (
    FeatureEngineer,
    build_dataset,
    feature_warmup_length,
)
from src.models import build_pipeline, get_baseline
from src.utils import Fold, save_artifact, load_artifact, walk_forward_folds


@pytest.fixture
def ohlcv():
    """Deterministic random-walk OHLCV covering the deepest rolling window."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2017-01-02", periods=900)
    close = 80 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, len(dates))))
    jitter = close * 0.008
    data = pd.DataFrame(
        {
            "Open": close + rng.normal(0, jitter),
            "High": close + np.abs(rng.normal(0, jitter)),
            "Low": close - np.abs(rng.normal(0, jitter)),
            "Close": close,
            "Volume": rng.uniform(1e6, 4e6, len(dates)),
        },
        index=dates,
    )
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)
    # The real loader always names the index 'Date'; match it so round-tripping
    # the fixture through a CSV snapshot reproduces it exactly.
    data.index.name = "Date"
    return data


# ----------------------------------------------------------------------
# (i) No lookahead in features
# ----------------------------------------------------------------------

# Cut points for the truncation test. Truncating at C perturbs only row C-1, so a
# leak that fills forward over a gap of period p is visible only when C = 1 (mod p).
# These three are strictly inside the 900-row fixture and cover different residues:
#   301 -> 1 mod 5, 0 mod 7      596 -> 1 mod 5, 1 mod 7      851 -> 1 mod 5, 4 mod 7
# 596 is the one that catches period-7 gaps; all three catch period-5 gaps and any
# plain shift(-1). test_lookahead_check_catches_planted_leaks enforces that claim.
LOOKAHEAD_CUTS = [301, 596, 851]


def _assert_no_lookahead(raw: pd.DataFrame, cut: int) -> None:
    """Recompute features on raw[:cut] and require the overlap to be identical."""
    assert 0 < cut < len(raw), "cut must fall strictly inside the series"

    full = FeatureEngineer(raw).create_all_features(dropna=False)
    truncated = FeatureEngineer(raw.iloc[:cut]).create_all_features(dropna=False)

    assert len(truncated) == cut
    pd.testing.assert_frame_equal(
        full.loc[truncated.index], truncated,
        check_exact=False, rtol=1e-10, atol=1e-12,
    )


@pytest.mark.parametrize("cut", LOOKAHEAD_CUTS)
def test_features_use_no_future_data(ohlcv, cut):
    """
    Truncating the price series must not change any earlier feature value.

    If a feature peeked ahead -- a centred window, a full-sample statistic, a
    shift with the wrong sign -- then removing the tail would change rows before
    the cut. Recomputing on a truncated series and comparing the overlap is a
    direct test of the property the whole project depends on.
    """
    _assert_no_lookahead(ohlcv, cut)


# Leaks planted into feature engineering to prove the check above can fail. Each
# takes the Close series and returns a column that uses information from the future.
PLANTED_LEAKS = {
    # The obvious case: tomorrow's close, straight in.
    "shift_minus_1": lambda close: close.shift(-1),
    # Interior gaps filled backwards from the next observation -- the leak a
    # single cut point can miss, at two different periods.
    "interior_bfill_p5": lambda close: close.mask(
        np.arange(len(close)) % 5 == 0).bfill(),
    "interior_bfill_p7": lambda close: close.mask(
        np.arange(len(close)) % 7 == 0).bfill(),
    # Linear interpolation across a one-row gap averages past and future.
    "interior_interpolate_p5": lambda close: close.mask(
        np.arange(len(close)) % 5 == 0).interpolate(),
    "interior_interpolate_p7": lambda close: close.mask(
        np.arange(len(close)) % 7 == 0).interpolate(),
}


@pytest.mark.parametrize("leak_name", sorted(PLANTED_LEAKS))
def test_lookahead_check_catches_planted_leaks(ohlcv, monkeypatch, leak_name):
    """
    Mutation probe: plant a leak in feature engineering, require detection.

    A passing test proves nothing unless it can fail. This patches
    ``create_price_patterns`` so the leaky column flows through
    ``create_all_features`` exactly like a real feature, then asserts that at
    least one of the configured cut points rejects it. That makes the probe
    pipeline-level -- it exercises the same code path the real test does, rather
    than checking that ``assert_frame_equal`` works on a hand-built frame.
    """
    leak = PLANTED_LEAKS[leak_name]
    original = FeatureEngineer.create_price_patterns

    def leaky_create_price_patterns(self):
        features = original(self)
        features['planted_leak'] = leak(self.data['Close'])
        return features

    monkeypatch.setattr(
        FeatureEngineer, 'create_price_patterns', leaky_create_price_patterns
    )

    caught = []
    for cut in LOOKAHEAD_CUTS:
        try:
            _assert_no_lookahead(ohlcv, cut)
        except AssertionError:
            caught.append(cut)

    assert caught, (
        f"{leak_name} escaped every cut in {LOOKAHEAD_CUTS}; the cut points no "
        f"longer cover the residues needed to see this class of leak"
    )


def test_planted_leak_probe_is_not_self_fulfilling(ohlcv):
    """Without a planted leak, the same cut points must all pass."""
    for cut in LOOKAHEAD_CUTS:
        _assert_no_lookahead(ohlcv, cut)


def test_every_engineered_column_is_covered_by_the_lookahead_check(ohlcv):
    """Guard against the test above silently covering only a handful of columns."""
    engineered = FeatureEngineer(ohlcv).create_all_features(dropna=False)
    assert len(engineered.columns) > 80


def test_target_is_strictly_forward_looking(ohlcv):
    """target_logret at t must equal log(C(t+1)/C(t)), never use C(t-1)."""
    dataset, _ = build_dataset(ohlcv, config=None, horizon=1)
    close = ohlcv["Close"]

    for timestamp in dataset.index[:25]:
        position = close.index.get_loc(timestamp)
        expected = np.log(close.iloc[position + 1] / close.iloc[position])
        assert dataset.loc[timestamp, "target_logret"] == pytest.approx(expected)
        assert dataset.loc[timestamp, "target_direction"] == int(expected > 0)


def test_warmup_length_matches_first_valid_row(ohlcv):
    """The embargo length is the real dependency depth of a feature row."""
    warmup = feature_warmup_length(ohlcv, config=None)
    engineered = FeatureEngineer(ohlcv).create_all_features(dropna=False)

    assert engineered.iloc[:warmup].isna().any(axis=1).all()
    assert engineered.iloc[warmup].notna().all()


def test_all_configured_periods_are_wired_through(ohlcv):
    """
    Every configurable period in config['features'] must reach the feature set.

    The original bug was that configured MACD / Bollinger / stochastic periods
    were silently ignored; volume_sma, obv, rolling_std_windows and price_changes
    were hard-coded too. This pins all of them by using unmistakable values.
    """
    config = {
        "lag_periods": [4],
        "sma_windows": [7],
        "ema_windows": [9],
        "rsi_period": 11,
        "macd_fast": 6,
        "macd_slow": 13,
        "macd_signal": 4,
        "bollinger_window": 17,
        "bollinger_std": 3,
        "atr_period": 19,
        "stoch_k": 23,
        "stoch_d": 5,
        "volume_sma": 29,
        "obv": False,
        "rolling_std_windows": [31],
        "price_changes": [3],
        "momentum_periods": [37],
        "trend_windows": [41],
    }
    features = FeatureEngineer(ohlcv).create_all_features(config, dropna=False)

    for expected in [
        "Close_lag_4", "sma_7", "ema_9", "rsi_11", "atr_19",
        "volatility_31d", "return_3d", "momentum_37", "trend_slope_41",
    ]:
        assert expected in features.columns, f"{expected} missing"

    # Periods that do not appear in a column name are verified numerically.
    close = ohlcv["Close"]
    expected_bb_middle = close.rolling(17).mean()
    pd.testing.assert_series_equal(
        features["bb_middle"], expected_bb_middle, check_names=False
    )

    expected_macd = (
        close.ewm(span=6, adjust=False).mean() - close.ewm(span=13, adjust=False).mean()
    )
    pd.testing.assert_series_equal(
        features["macd_line"], expected_macd, check_names=False
    )

    expected_volume_sma = ohlcv["Volume"].rolling(29).mean()
    pd.testing.assert_series_equal(
        features["volume_sma"], expected_volume_sma, check_names=False
    )

    expected_stoch_d = features["stoch_k"].rolling(5).mean()
    pd.testing.assert_series_equal(
        features["stoch_d"], expected_stoch_d, check_names=False
    )

    # obv: False must actually suppress the cumulative volume columns.
    assert "obv" not in features.columns
    assert "vpt" not in features.columns


def test_bollinger_and_stochastic_accept_config_without_raising(ohlcv):
    """
    Regression test for the TypeError that stopped train.py running.

    The fallback dicts used to pass 'std', 'k' and 'd', which the methods do not
    accept, so create_all_features raised before producing anything.
    """
    config = {"bollinger_window": 20, "bollinger_std": 2, "stoch_k": 14, "stoch_d": 3}
    features = FeatureEngineer(ohlcv).create_all_features(config, dropna=False)
    assert {"bb_upper", "bb_lower", "stoch_k", "stoch_d"} <= set(features.columns)


# ----------------------------------------------------------------------
# (ii) Walk-forward folds: ordered, embargoed, non-overlapping
# ----------------------------------------------------------------------

def test_folds_are_ordered_with_embargo_and_no_overlap():
    embargo = 50
    folds = walk_forward_folds(
        n_samples=1000, initial_train=300, test_size=60, step_size=60, embargo=embargo
    )
    assert len(folds) > 1

    for fold in folds:
        # Train strictly precedes test, separated by at least the embargo.
        assert fold.train_end <= fold.test_start
        assert fold.test_start - fold.train_end == embargo
        assert fold.embargo == embargo
        assert fold.train_start < fold.train_end
        assert fold.test_start < fold.test_end
        assert fold.test_end <= 1000

    for earlier, later in zip(folds, folds[1:]):
        # Folds move forward and test windows never overlap.
        assert later.train_end > earlier.train_end
        assert later.test_start >= earlier.test_end


def test_expanding_window_always_starts_at_zero():
    folds = walk_forward_folds(1000, 300, 60, 60, embargo=10, expanding=True)
    assert all(f.train_start == 0 for f in folds)
    assert folds[-1].n_train > folds[0].n_train


def test_rolling_window_keeps_train_size_fixed():
    folds = walk_forward_folds(1000, 300, 60, 60, embargo=10, expanding=False)
    assert all(f.n_train == 300 for f in folds[1:])


def test_embargo_removes_rows_a_zero_embargo_would_have_used():
    without = walk_forward_folds(1000, 300, 60, 60, embargo=0)
    with_embargo = walk_forward_folds(1000, 300, 60, 60, embargo=200)
    assert with_embargo[0].test_start - without[0].test_start == 200
    assert len(with_embargo) < len(without)


def test_walk_forward_raises_when_no_fold_fits():
    with pytest.raises(ValueError, match="No folds fit"):
        walk_forward_folds(n_samples=100, initial_train=90, test_size=60, embargo=50)


def test_run_walk_forward_predictions_are_out_of_sample(ohlcv):
    """
    Every prediction must come from a model that never saw that row.

    Checked structurally: the returned index equals exactly the union of the test
    slices, with no duplicates and in ascending date order.
    """
    dataset, features = build_dataset(ohlcv, config=None, horizon=1)
    folds = walk_forward_folds(len(dataset), 300, 60, 60, embargo=199)

    predictions = run_walk_forward(
        dataset, features, "target_logret",
        lambda: get_baseline("zero", "regression"), folds,
    )

    expected = np.concatenate([
        np.arange(f.test_start, f.test_end) for f in folds
    ])
    assert list(predictions.index) == list(dataset.index[expected])
    assert not predictions.index.duplicated().any()
    assert predictions.index.is_monotonic_increasing


def test_run_walk_forward_refits_per_fold(ohlcv):
    """
    The train-mean baseline must report a different constant per fold.

    A single fit reused across folds would emit one constant everywhere, which is
    exactly the bug an "expanding window" can hide.
    """
    dataset, features = build_dataset(ohlcv, config=None, horizon=1)
    folds = walk_forward_folds(len(dataset), 300, 60, 60, embargo=199)

    predictions = run_walk_forward(
        dataset, features, "target_logret",
        lambda: get_baseline("train_mean", "regression"), folds,
    )
    per_fold = predictions.groupby("fold")["y_pred"].nunique()
    assert (per_fold == 1).all()                      # constant within a fold
    assert predictions["y_pred"].nunique() == len(folds)  # different across folds


# ----------------------------------------------------------------------
# (iii) train / predict feature parity
# ----------------------------------------------------------------------

def test_train_and_predict_build_identical_features(ohlcv, tmp_path):
    """
    predict.py must reproduce the training matrix exactly.

    The original failure mode: train.py dropped Open/High/Low then applied
    selection, while predict.py kept them and selected nothing, so the model was
    handed the wrong columns at inference. Both paths now go through
    build_dataset with the config carried inside the artifact.
    """
    import predict as predict_module

    config = {
        "data": {
            "symbol": "TEST", "start_date": "2017-01-01", "end_date": "2025-01-01",
            "snapshot_dir": str(tmp_path), "allow_download": False,
        },
        "features": {"stationary_only": True},
        "paths": {"models_dir": str(tmp_path)},
    }

    ohlcv.to_csv(tmp_path / "TEST.csv", index_label="Date")

    dataset, features = build_dataset(ohlcv, config["features"], horizon=1)
    pipeline = build_pipeline("linear", "regression", k_features=None)
    pipeline.fit(dataset[features], dataset["target_logret"])

    artifact_path = str(tmp_path / "artifact.joblib")
    save_artifact(
        {
            "pipeline": pipeline, "feature_columns": features, "config": config,
            "symbol": "TEST", "family": "linear", "horizon": 1,
        },
        artifact_path,
    )

    artifact = load_artifact(artifact_path)
    inference = predict_module.build_inference_features("TEST", artifact)

    # Same columns, same order, same values. check_freq is relaxed only because a
    # CSV round-trip drops the index's inferred frequency; the dates are identical.
    assert list(inference[features].columns) == list(features)
    pd.testing.assert_frame_equal(
        inference[features], dataset[features], check_freq=False
    )
    assert list(inference.index) == list(dataset.index)

    # And the model consumes them without a shape error.
    out = predict_module.predict(artifact_path, "TEST")
    assert len(out) == 1
    assert np.isfinite(out["predicted_log_return"]).all()


def test_parity_failure_is_raised_not_silently_tolerated(ohlcv, tmp_path):
    """A missing training feature must fail loudly at inference."""
    import predict as predict_module

    ohlcv.to_csv(tmp_path / "TEST.csv", index_label="Date")
    artifact = {
        "pipeline": None,
        "feature_columns": ["rsi_14", "a_feature_that_does_not_exist"],
        "config": {
            "data": {
                "symbol": "TEST", "start_date": "2017-01-01",
                "end_date": "2025-01-01", "snapshot_dir": str(tmp_path),
            },
            "features": {"stationary_only": True},
        },
        "horizon": 1,
    }
    with pytest.raises(ValueError, match="parity check failed"):
        predict_module.build_inference_features("TEST", artifact)


def test_save_artifact_requires_feature_list(tmp_path):
    with pytest.raises(ValueError, match="missing required keys"):
        save_artifact({"pipeline": object()}, str(tmp_path / "bad.joblib"))


def test_load_artifact_rejects_a_bare_model(tmp_path):
    import joblib

    path = tmp_path / "bare.joblib"
    joblib.dump(build_pipeline("linear", "regression"), path)
    with pytest.raises(ValueError, match="not a training artifact"):
        load_artifact(str(path))


# ----------------------------------------------------------------------
# (iv) Backtest signal semantics
# ----------------------------------------------------------------------

def test_long_flat_signal_is_forecast_vs_spot():
    """
    Position is long exactly when the forecast return clears the threshold.

    Not when the forecast is rising relative to the previous forecast -- that was
    the original bug, and it makes a monotonically increasing forecast look like a
    permanent buy signal regardless of its level.
    """
    backtester = Backtester(initial_capital=100_000)
    dates = pd.date_range("2021-01-01", periods=6, freq="B")

    # Forecasts rise every bar but are negative for the first three.
    predicted = np.array([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03])
    actual = np.zeros(6)

    run = backtester.long_flat_backtest(predicted, actual, dates, threshold=0.0)
    np.testing.assert_array_equal(run["positions"], [0, 0, 0, 1, 1, 1])


def test_long_flat_respects_threshold():
    backtester = Backtester(initial_capital=100_000)
    dates = pd.date_range("2021-01-01", periods=4, freq="B")
    predicted = np.array([0.0005, 0.002, 0.0005, 0.002])

    run = backtester.long_flat_backtest(
        predicted, np.zeros(4), dates, threshold=0.001
    )
    np.testing.assert_array_equal(run["positions"], [0, 1, 0, 1])


def test_long_flat_earns_the_return_it_is_positioned_for():
    """A fully-invested, cost-free run must compound the realised returns."""
    backtester = Backtester(initial_capital=100.0)
    dates = pd.date_range("2021-01-01", periods=3, freq="B")
    actual = np.log1p(np.array([0.10, -0.05, 0.20]))

    run = backtester.long_flat_backtest(
        np.ones(3), actual, dates, threshold=0.0, cost_per_side=0.0
    )
    assert run["final_equity"] == pytest.approx(100 * 1.10 * 0.95 * 1.20)


def test_costs_reduce_return_and_scale_with_turnover():
    backtester = Backtester(initial_capital=100_000)
    dates = pd.date_range("2021-01-01", periods=40, freq="B")
    rng = np.random.default_rng(3)
    actual = rng.normal(0, 0.01, 40)
    alternating = np.where(np.arange(40) % 2 == 0, 0.01, -0.01)

    free = backtester.long_flat_backtest(alternating, actual, dates, cost_per_side=0.0)
    costly = backtester.long_flat_backtest(alternating, actual, dates, cost_per_side=0.002)

    assert costly["final_equity"] < free["final_equity"]
    assert costly["metrics"]["turnover"] > 10


def test_cost_sensitivity_is_monotonic_in_cost():
    backtester = Backtester(initial_capital=100_000)
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    rng = np.random.default_rng(5)
    actual = rng.normal(0.0005, 0.01, 60)
    predicted = rng.normal(0.0005, 0.01, 60)

    sweep = backtester.cost_sensitivity(
        predicted, actual, dates, cost_grid_bps=[0, 5, 10, 20]
    )
    assert list(sweep["cost_bps_per_side"]) == [0, 5, 10, 20]
    assert sweep["strategy_total_return_pct"].is_monotonic_decreasing


def test_buy_and_hold_returns_matches_holding_throughout():
    backtester = Backtester(initial_capital=100.0)
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    actual = np.log1p(np.full(5, 0.01))

    bench = backtester.buy_and_hold_returns(actual, dates, cost_per_side=0.0)
    assert np.all(bench["positions"] == 1)
    assert bench["final_equity"] == pytest.approx(100 * 1.01 ** 5)
    # Enters once and exits once: turnover of 2.
    assert bench["metrics"]["turnover"] == pytest.approx(2.0)
