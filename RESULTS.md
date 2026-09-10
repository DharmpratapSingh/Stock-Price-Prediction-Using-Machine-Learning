# NVDA next-day return experiment

Leakage-free walk-forward-by-calendar experiment: models are fitted on the training years, the trading threshold is chosen on the validation year, and the test year is scored once. Baselines and buy-and-hold are reported next to every model.

## Run metadata

- Symbol: `NVDA`
- Download range: `2017-10-01` to `2025-01-01` (end exclusive; the early months are feature warmup)
- Split boundaries: train through `2022-12-31`, validation `2023`, test `2024`
- Run timestamp (UTC): 2026-09-10T00:19:55Z
- Cost assumption: 0.00075 per side (0.15% round trip)
- Threshold grid: 0, 0.0005, 0.001, 0.002, 0.005
- Minimum days in market: 20
- Leakage warning limit: 56.0% test directional accuracy
- Random seed: 42
- Models:
  - `ridge`: {'alpha': 10.0}
  - `random_forest`: {'n_estimators': 300, 'max_depth': 4, 'min_samples_leaf': 20, 'random_state': 42}
  - `xgboost`: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42}
- Library versions: yfinance 1.7.0, pandas 3.0.2, numpy 2.3.5, scikit-learn 1.8.0, xgboost 3.2.0

### Splits

| split | rows | first_date | last_date |
| --- | --- | --- | --- |
| train | 1259 | 2018-01-02 | 2022-12-30 |
| val | 250 | 2023-01-03 | 2023-12-29 |
| test | 251 | 2024-01-02 | 2024-12-30 |

### Features (16)

`ret_lag0`, `ret_lag1`, `ret_lag2`, `ret_lag3`, `ret_lag4`, `ret_5d`, `ret_21d`, `ret_63d`, `sma_ratio_10`, `sma_ratio_20`, `sma_ratio_50`, `sma_10_50`, `rsi_14`, `atr_14_pct`, `vol_21`, `volume_ratio_20`

Target: `target`, the next-day simple return.

## Test-set metrics (2024)

| name | rmse | directional_accuracy | ic | r2 |
| --- | --- | --- | --- | --- |
| zero_return | 0.03334 | 44.2% | n/a | -0.020 |
| train_mean | 0.03317 | 55.8% | n/a | -0.010 |
| always_long | 0.03334 | 55.8% | n/a | -0.020 |
| majority_direction | 0.03334 | 55.8% | n/a | -0.020 |
| ridge | 0.03309 | 50.6% | 0.072 | -0.004 |
| random_forest | 0.03291 | 53.8% | 0.091 | +0.006 |
| xgboost | 0.03355 | 52.6% | -0.019 | -0.033 |

Baselines come first on purpose: a model that cannot beat `zero_return` on RMSE or `always_long` on directional accuracy has not learned anything about tomorrow.

## Validation metrics (2023)

| name | rmse | directional_accuracy | ic | r2 |
| --- | --- | --- | --- | --- |
| zero_return | 0.03093 | 43.2% | n/a | -0.030 |
| train_mean | 0.03072 | 56.8% | n/a | -0.017 |
| always_long | 0.03093 | 56.8% | n/a | -0.030 |
| majority_direction | 0.03093 | 56.8% | n/a | -0.030 |
| ridge | 0.03152 | 46.8% | -0.118 | -0.070 |
| random_forest | 0.03086 | 53.2% | -0.069 | -0.025 |
| xgboost | 0.03130 | 54.0% | -0.056 | -0.055 |

## Test-set backtest (2024, after costs)

| strategy | gross_return | net_return | cost_drag | total_cost | sharpe | max_drawdown | n_trades | n_round_trips | days_in_market |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | 178.9% | 178.4% | 0.4% | 0.1% | 2.22 | -27.0% | 2 | 1 | 100.0% |
| ridge@0 (chosen) | 154.8% | 133.9% | 20.9% | 8.6% | 2.20 | -13.3% | 114 | 57 | 61.4% |
| ridge@0 | 154.8% | 133.9% | 20.9% | 8.6% | 2.20 | -13.3% | 114 | 57 | 61.4% |
| random_forest@0 (chosen) | 132.7% | 123.5% | 9.2% | 4.0% | 2.01 | -22.8% | 54 | 27 | 72.5% |
| random_forest@0 | 132.7% | 123.5% | 9.2% | 4.0% | 2.01 | -22.8% | 54 | 27 | 72.5% |
| xgboost@0.0005 | 54.8% | 46.3% | 8.6% | 5.7% | 1.15 | -33.1% | 76 | 38 | 60.6% |
| xgboost@0 | 64.5% | 55.4% | 9.1% | 5.7% | 1.28 | -31.4% | 76 | 38 | 63.3% |

`buy_and_hold` is the benchmark to beat. Each model appears twice: at the threshold chosen on the validation year, and at 0.0 (long whenever the forecast is positive). `days_in_market` is the fraction of test days holding the position.

## Cost arithmetic

Generic: **0.15% round trip × 250 round trips/year = 37.5% per year**.

That is the bill for trading in and out every day, before a single model is fitted. Per model, at its chosen threshold on the test year:

| model | threshold | n_round_trips | round_trips_per_year | implied_annual_cost_drag | test_cost_drag |
| --- | --- | --- | --- | --- | --- |
| ridge | 0.0000 | 57 | 57.2 | 8.6% | 20.9% |
| random_forest | 0.0000 | 27 | 27.1 | 4.1% | 9.2% |
| xgboost | 0.0005 | 38 | 38.2 | 5.7% | 8.6% |

`round_trips_per_year` annualizes the test-year round trips (251 trading days at 252 days/year); `implied_annual_cost_drag` is that rate times the round-trip cost. `test_cost_drag` is what the backtest actually charged over the test year.

## Threshold sweeps (validation year 2023)

The threshold is chosen here and nowhere else. `eligible` marks the rules that were in the market at least 20 days -- an ineligible rule is still scored, it just cannot win.

### ridge (chosen: 0)

| row | threshold | gross_return | net_return | sharpe | max_drawdown | n_trades | days_in_market | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 40.7% | 31.0% | 0.89 | -17.8% | 96 | 55.6% | yes |
| 1 | 0.0005 | 20.4% | 11.7% | 0.47 | -18.7% | 100 | 50.4% | yes |
| 2 | 0.0010 | 12.9% | 4.8% | 0.30 | -24.1% | 100 | 42.0% | yes |
| 3 | 0.0020 | 12.0% | 4.5% | 0.29 | -19.3% | 92 | 32.0% | yes |
| 4 | 0.0050 | -3.4% | -4.2% | -0.54 | -10.1% | 12 | 2.4% | no |

### random_forest (chosen: 0)

| row | threshold | gross_return | net_return | sharpe | max_drawdown | n_trades | days_in_market | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 155.3% | 148.5% | 2.29 | -19.7% | 36 | 78.8% | yes |
| 1 | 0.0005 | 114.7% | 108.0% | 1.91 | -21.0% | 42 | 71.6% | yes |
| 2 | 0.0010 | 101.3% | 94.2% | 1.80 | -20.5% | 48 | 66.8% | yes |
| 3 | 0.0020 | 32.5% | 25.7% | 1.01 | -17.6% | 70 | 46.4% | yes |
| 4 | 0.0050 | 2.0% | 1.8% | 0.96 | -0.1% | 2 | 0.4% | no |

### xgboost (chosen: 0.0005)

| row | threshold | gross_return | net_return | sharpe | max_drawdown | n_trades | days_in_market | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 77.3% | 69.5% | 1.80 | -26.4% | 60 | 69.2% | yes |
| 1 | 0.0005 | 77.4% | 67.9% | 1.83 | -24.5% | 74 | 63.6% | yes |
| 2 | 0.0010 | 60.2% | 51.6% | 1.60 | -21.0% | 74 | 56.0% | yes |
| 3 | 0.0020 | 15.0% | 8.6% | 0.46 | -21.9% | 76 | 42.0% | yes |
| 4 | 0.0050 | 8.9% | 6.4% | 0.62 | -7.6% | 32 | 6.8% | no |

## Leakage guard

Not fired. No model exceeded the 56.0% test directional accuracy plausibility limit.
