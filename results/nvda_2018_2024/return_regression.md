| ticker | model | is_baseline | n | RMSE | MAE | R2 | R2_vs_zero |
|---|---|---|---|---|---|---|---|
| NVDA | Ridge / Logistic |  | 819 | 0.0354 | 0.0269 | -0.0632 | -0.0582 |
| NVDA | Random Forest |  | 819 | 0.0345 | 0.0260 | -0.0096 | -0.0049 |
| NVDA | XGBoost |  | 819 | 0.0356 | 0.0267 | -0.0720 | -0.0670 |
| NVDA | Baseline: zero return | yes | 819 | 0.0345 | 0.0259 | -0.0047 | 0.0000 |
| NVDA | Baseline: train-window mean | yes | 819 | 0.0344 | 0.0258 | -0.0012 | 0.0035 |
| NVDA | Baseline: yesterday's return | yes | 819 | 0.0492 | 0.0371 | -1.0468 | -1.0372 |
