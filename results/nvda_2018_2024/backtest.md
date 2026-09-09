| ticker | model | is_baseline | total_return_pct | annualized_return_pct | sharpe | max_drawdown_pct | turnover | time_in_market | n_periods | excess_vs_buy_hold_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | Buy & Hold | yes | 585.8623 | 80.8438 | 1.4647 | 66.3351 | 2.0000 | 1.0000 | 819 |  |
| NVDA | Ridge / Logistic |  | 86.4626 | 21.1320 | 0.5246 | 55.1312 | 328.0000 | 0.6300 | 819 | -499.3997 |
| NVDA | Random Forest |  | 310.5981 | 54.4343 | 1.2158 | 58.3673 | 284.0000 | 0.6716 | 819 | -275.2642 |
| NVDA | XGBoost |  | 130.4673 | 29.2921 | 0.7279 | 48.3303 | 264.0000 | 0.5800 | 819 | -455.3950 |
| NVDA | Baseline: zero return | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 819 | -585.8623 |
| NVDA | Baseline: train-window mean | yes | 585.8623 | 80.8438 | 1.4647 | 66.3351 | 2.0000 | 1.0000 | 819 | 0.0000 |
| NVDA | Baseline: yesterday's return | yes | 95.7517 | 22.9577 | 0.6111 | 46.7410 | 382.0000 | 0.5360 | 819 | -490.1106 |
