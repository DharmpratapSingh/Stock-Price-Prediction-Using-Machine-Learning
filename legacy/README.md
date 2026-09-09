# Legacy

The superseded naive version, kept for history: a shuffled `train_test_split` on a
time series, predicting the next day's price *level*, reporting R² ≈ 0.997.

That R² is the autocorrelation of the price series, not forecasting skill — the
`results/level_r2_trap.*` tables reproduce it on proper walk-forward folds and
show a persistence forecast scoring the same. The current pipeline predicts
next-day log returns instead, validated walk-forward with an embargo.
