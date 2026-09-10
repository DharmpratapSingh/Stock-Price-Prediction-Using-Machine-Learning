# NVDA next-day return experiment

Leakage-free walk-forward-by-calendar experiment: models are fitted on the training years, the trading threshold is chosen on the validation year, and the test year is scored once. Baselines and buy-and-hold are reported next to every model.

## Run metadata

- Symbol: `NVDA`
- Download range: `2017-10-01` to `2025-01-01` (end exclusive; the early months are feature warmup)
- Split boundaries: train through `2022-12-31`, validation `2023`, test `2024`
- Run timestamp (UTC): 2026-09-10T02:18:50Z
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
| train | 1258 | 2018-01-02 | 2022-12-29 |
| val | 249 | 2023-01-03 | 2023-12-28 |
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
| ridge | 0.03309 | 50.2% | 0.069 | -0.004 |
| random_forest | 0.03294 | 53.8% | 0.084 | +0.005 |
| xgboost | 0.03356 | 51.4% | -0.021 | -0.033 |

Baselines come first on purpose: a model that cannot beat `zero_return` on RMSE or `always_long` on directional accuracy has not learned anything about tomorrow.

## Validation metrics (2023)

| name | rmse | directional_accuracy | ic | r2 |
| --- | --- | --- | --- | --- |
| zero_return | 0.03095 | 43.0% | n/a | -0.032 |
| train_mean | 0.03073 | 57.0% | n/a | -0.018 |
| always_long | 0.03095 | 57.0% | n/a | -0.032 |
| majority_direction | 0.03095 | 57.0% | n/a | -0.032 |
| ridge | 0.03153 | 47.4% | -0.115 | -0.071 |
| random_forest | 0.03088 | 53.4% | -0.084 | -0.027 |
| xgboost | 0.03135 | 51.0% | -0.067 | -0.059 |

## Test-set backtest (2024, after costs)

| strategy | gross_return | net_return | cost_drag | total_cost | sharpe | max_drawdown | n_trades | n_round_trips | days_in_market |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | 178.9% | 178.4% | 0.4% | 0.1% | 2.22 | -27.0% | 2 | 1 | 100.0% |
| ridge@0 (chosen) | 150.6% | 130.1% | 20.5% | 8.6% | 2.16 | -14.7% | 114 | 57 | 61.8% |
| ridge@0 | 150.6% | 130.1% | 20.5% | 8.6% | 2.16 | -14.7% | 114 | 57 | 61.8% |
| random_forest@0 (chosen) | 133.6% | 123.6% | 9.9% | 4.4% | 2.01 | -22.8% | 58 | 29 | 71.7% |
| random_forest@0 | 133.6% | 123.6% | 9.9% | 4.4% | 2.01 | -22.8% | 58 | 29 | 71.7% |
| xgboost@0.0005 | 64.2% | 54.5% | 9.8% | 6.2% | 1.28 | -21.7% | 82 | 41 | 59.4% |
| xgboost@0 | 71.9% | 62.9% | 9.0% | 5.4% | 1.41 | -23.4% | 72 | 36 | 62.9% |

`buy_and_hold` is the benchmark to beat. Each model appears twice: at the threshold chosen on the validation year, and at 0.0 (long whenever the forecast is positive). `days_in_market` is the fraction of test days holding the position.

## Cost arithmetic

Generic: **0.15% round trip × 250 round trips/year = 37.5% per year**.

That is the bill for trading in and out every day, before a single model is fitted. Per model, at its chosen threshold on the test year:

| model | threshold | n_round_trips | round_trips_per_year | implied_annual_cost_drag | test_cost_drag |
| --- | --- | --- | --- | --- | --- |
| ridge | 0.0000 | 57 | 57.2 | 8.6% | 20.5% |
| random_forest | 0.0000 | 29 | 29.1 | 4.4% | 9.9% |
| xgboost | 0.0005 | 41 | 41.2 | 6.2% | 9.8% |

`round_trips_per_year` annualizes the test-year round trips (251 trading days at 252 days/year); `implied_annual_cost_drag` is that rate times the round-trip cost. `test_cost_drag` is what the backtest actually charged over the test year.

## Threshold sweeps (validation year 2023)

The threshold is chosen here and nowhere else. `eligible` marks the rules that were in the market at least 20 days -- an ineligible rule is still scored, it just cannot win.

In this run the participation guard changed nothing: it marks the 0.005-threshold rows ineligible, but the best-Sharpe row for every model was already eligible, so no selected threshold depended on the guard. It is there to stop a rule that trades a handful of days from winning on a Sharpe computed off almost no exposure.

### ridge (chosen: 0)

| row | threshold | gross_return | net_return | sharpe | max_drawdown | n_trades | days_in_market | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 45.8% | 35.5% | 0.99 | -17.6% | 98 | 56.6% | yes |
| 1 | 0.0005 | 27.2% | 18.0% | 0.62 | -18.7% | 100 | 51.0% | yes |
| 2 | 0.0010 | 12.9% | 4.6% | 0.29 | -22.6% | 102 | 43.4% | yes |
| 3 | 0.0020 | 10.1% | 2.6% | 0.23 | -18.2% | 94 | 32.5% | yes |
| 4 | 0.0050 | -4.1% | -5.2% | -0.68 | -11.0% | 16 | 3.2% | no |

### random_forest (chosen: 0)

| row | threshold | gross_return | net_return | sharpe | max_drawdown | n_trades | days_in_market | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 166.6% | 159.0% | 2.40 | -19.7% | 38 | 77.9% | yes |
| 1 | 0.0005 | 97.7% | 91.9% | 1.75 | -19.7% | 40 | 71.9% | yes |
| 2 | 0.0010 | 106.6% | 99.6% | 1.88 | -21.0% | 46 | 67.5% | yes |
| 3 | 0.0020 | 42.2% | 35.3% | 1.30 | -18.0% | 66 | 47.0% | yes |
| 4 | 0.0050 | 2.0% | 1.8% | 0.96 | -0.1% | 2 | 0.4% | no |

### xgboost (chosen: 0.0005)

| row | threshold | gross_return | net_return | sharpe | max_drawdown | n_trades | days_in_market | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 131.9% | 123.4% | 2.10 | -21.5% | 50 | 67.5% | yes |
| 1 | 0.0005 | 136.4% | 126.1% | 2.16 | -21.5% | 60 | 63.1% | yes |
| 2 | 0.0010 | 61.1% | 53.3% | 1.62 | -22.1% | 66 | 55.4% | yes |
| 3 | 0.0020 | 17.6% | 10.4% | 0.53 | -26.6% | 84 | 42.2% | yes |
| 4 | 0.0050 | 18.3% | 16.3% | 1.40 | -4.4% | 22 | 5.2% | no |

## Leakage guard

Not fired. No model exceeded the 56.0% test directional accuracy plausibility limit.
