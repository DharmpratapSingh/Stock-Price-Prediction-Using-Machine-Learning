| ticker | model | is_baseline | total_return_pct | annualized_return_pct | sharpe | sharpe_arithmetic | max_drawdown_pct | turnover | time_in_market | n_periods | excess_vs_buy_hold_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | Buy & Hold | yes | 589.3478 | 81.1261 | 1.4697 | 1.3493 | 66.3351 | 2.0000 | 1.0000 | 819 |  |
| NVDA | Ridge / Logistic |  | 76.1065 | 19.0209 | 0.4722 | 0.6319 | 56.6188 | 326.0000 | 0.6300 | 819 | -513.2413 |
| NVDA | Random Forest |  | 293.3151 | 52.4043 | 1.1710 | 1.1609 | 59.7475 | 280.0000 | 0.6716 | 819 | -296.0327 |
| NVDA | XGBoost |  | 137.0061 | 30.4099 | 0.7552 | 0.8555 | 48.3303 | 264.0000 | 0.5812 | 819 | -452.3417 |
| NVDA | Baseline: zero return | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 819 | -589.3478 |
| NVDA | Baseline: train-window mean | yes | 589.3478 | 81.1261 | 1.4697 | 1.3493 | 66.3351 | 2.0000 | 1.0000 | 819 | 0.0000 |
| NVDA | Baseline: yesterday's return | yes | 96.7465 | 23.1496 | 0.6161 | 0.7417 | 46.7410 | 382.0000 | 0.5360 | 819 | -492.6013 |
