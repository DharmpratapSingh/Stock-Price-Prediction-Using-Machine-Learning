| ticker | model | is_baseline | n | accuracy | ci_lower | ci_upper | p_vs_0.5 | precision_up | recall_up | pred_up_rate | actual_up_rate | reference_rate | p_vs_reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | Ridge / Logistic |  | 819 | 0.5031 | 0.4689 | 0.5372 | 0.8889 | 0.5307 | 0.6310 | 0.6374 | 0.5360 | 0.5433 | 0.0226 |
| NVDA | Random Forest |  | 819 | 0.5201 | 0.4859 | 0.5542 | 0.2635 | 0.5395 | 0.7153 | 0.7106 | 0.5360 | 0.5433 | 0.1943 |
| NVDA | XGBoost |  | 819 | 0.4994 | 0.4652 | 0.5336 | 1.0000 | 0.5279 | 0.6241 | 0.6337 | 0.5360 | 0.5433 | 0.0127 |
| NVDA | Baseline: always up | yes | 819 | 0.5360 | 0.5018 | 0.5699 | 0.0426 | 0.5360 | 1.0000 | 1.0000 | 0.5360 | 0.5433 | 0.6997 |
