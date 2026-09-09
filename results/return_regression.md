| ticker | model | is_baseline | n | RMSE | MAE | R2 | R2_vs_zero |
|---|---|---|---|---|---|---|---|
| SPY | Ridge / Logistic |  | 1575 | 0.0129 | 0.0087 | -0.0544 | -0.0526 |
| SPY | Random Forest |  | 1575 | 0.0126 | 0.0083 | -0.0064 | -0.0047 |
| SPY | XGBoost |  | 1575 | 0.0129 | 0.0085 | -0.0439 | -0.0421 |
| SPY | Baseline: zero return | yes | 1575 | 0.0126 | 0.0083 | -0.0017 | 0.0000 |
| SPY | Baseline: train-window mean | yes | 1575 | 0.0126 | 0.0083 | -0.0004 | 0.0013 |
| SPY | Baseline: yesterday's return | yes | 1575 | 0.0190 | 0.0122 | -1.2868 | -1.2830 |
| AAPL | Ridge / Logistic |  | 1575 | 0.0199 | 0.0142 | -0.0172 | -0.0148 |
| AAPL | Random Forest |  | 1575 | 0.0197 | 0.0139 | 0.0069 | 0.0093 |
| AAPL | XGBoost |  | 1575 | 0.0198 | 0.0141 | -0.0055 | -0.0031 |
| AAPL | Baseline: zero return | yes | 1575 | 0.0198 | 0.0140 | -0.0024 | 0.0000 |
| AAPL | Baseline: train-window mean | yes | 1575 | 0.0197 | 0.0139 | -0.0005 | 0.0019 |
| AAPL | Baseline: yesterday's return | yes | 1575 | 0.0293 | 0.0205 | -1.2016 | -1.1964 |
| MSFT | Ridge / Logistic |  | 1575 | 0.0189 | 0.0134 | -0.0404 | -0.0380 |
| MSFT | Random Forest |  | 1575 | 0.0185 | 0.0130 | 0.0004 | 0.0027 |
| MSFT | XGBoost |  | 1575 | 0.0189 | 0.0134 | -0.0423 | -0.0399 |
| MSFT | Baseline: zero return | yes | 1575 | 0.0185 | 0.0130 | -0.0023 | 0.0000 |
| MSFT | Baseline: train-window mean | yes | 1575 | 0.0185 | 0.0129 | -0.0004 | 0.0019 |
| MSFT | Baseline: yesterday's return | yes | 1575 | 0.0283 | 0.0195 | -1.3480 | -1.3425 |
| NVDA | Ridge / Logistic |  | 1575 | 0.0343 | 0.0249 | -0.0636 | -0.0599 |
| NVDA | Random Forest |  | 1575 | 0.0334 | 0.0242 | -0.0074 | -0.0039 |
| NVDA | XGBoost |  | 1575 | 0.0342 | 0.0247 | -0.0574 | -0.0537 |
| NVDA | Baseline: zero return | yes | 1575 | 0.0333 | 0.0243 | -0.0035 | 0.0000 |
| NVDA | Baseline: train-window mean | yes | 1575 | 0.0333 | 0.0242 | -0.0023 | 0.0012 |
| NVDA | Baseline: yesterday's return | yes | 1575 | 0.0488 | 0.0356 | -1.1479 | -1.1403 |
| JPM | Ridge / Logistic |  | 1575 | 0.0192 | 0.0128 | -0.0356 | -0.0348 |
| JPM | Random Forest |  | 1575 | 0.0190 | 0.0125 | -0.0088 | -0.0080 |
| JPM | XGBoost |  | 1575 | 0.0194 | 0.0129 | -0.0575 | -0.0567 |
| JPM | Baseline: zero return | yes | 1575 | 0.0189 | 0.0125 | -0.0008 | 0.0000 |
| JPM | Baseline: train-window mean | yes | 1575 | 0.0189 | 0.0125 | -0.0006 | 0.0002 |
| JPM | Baseline: yesterday's return | yes | 1575 | 0.0282 | 0.0178 | -1.2315 | -1.2298 |
| POOLED | Ridge / Logistic |  | 7875 | 0.0222 | 0.0148 | -0.0471 | -0.0450 |
| POOLED | Random Forest |  | 7875 | 0.0217 | 0.0144 | -0.0034 | -0.0014 |
| POOLED | XGBoost |  | 7875 | 0.0222 | 0.0147 | -0.0451 | -0.0430 |
| POOLED | Baseline: zero return | yes | 7875 | 0.0217 | 0.0144 | -0.0020 | 0.0000 |
| POOLED | Baseline: train-window mean | yes | 7875 | 0.0217 | 0.0144 | -0.0007 | 0.0013 |
| POOLED | Baseline: yesterday's return | yes | 7875 | 0.0322 | 0.0211 | -1.2065 | -1.2021 |
