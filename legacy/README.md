# Legacy artifacts

The original notebook and its writeup PDF, kept for history only: a shuffled
`train_test_split` on a time series, predicting next-day *price levels*, reporting
R² ≈ 0.997.

That R² is the autocorrelation of the price series, not forecasting skill — it is
the trap both current code paths exist to replace. `results/pipeline/level_r2_trap.*`
reproduces it on proper walk-forward folds and shows a persistence forecast scoring
the same.

For the numbers the main README reports, run `python run_experiment.py` from the
repo root. For the five-ticker walk-forward pipeline, run `python train.py --basket`
and see [`../docs/PIPELINE.md`](../docs/PIPELINE.md).
