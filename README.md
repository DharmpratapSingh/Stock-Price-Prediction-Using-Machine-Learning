# NVDA next-day return prediction

A leakage-free daily ML experiment on NVDA, with realistic trading costs.
The point is not that the model wins. It does not.

Train 2018–2022, validate 2023, test 2024. Target is the next-day simple return.
Features are a lean set of 16 backward-looking signals. Thresholds are chosen on
validation only. The test year is scored once. Full tables live in [`RESULTS.md`](RESULTS.md).

## Results (test year 2024)

| name | RMSE | dir. acc. | IC | R² |
| --- | --- | --- | --- | --- |
| zero_return | 0.03334 | 44.2% | n/a | −0.020 |
| train_mean | 0.03317 | 55.8% | n/a | −0.010 |
| always_long | 0.03334 | 55.8% | n/a | −0.020 |
| ridge | 0.03309 | 50.6% | 0.072 | −0.004 |
| random_forest | 0.03291 | 53.8% | 0.091 | +0.006 |
| xgboost | 0.03355 | 52.6% | −0.019 | −0.033 |

| strategy | net return | Sharpe | max DD | round trips | cost drag |
| --- | --- | --- | --- | --- | --- |
| buy_and_hold | 178.4% | 2.22 | −27.0% | 1 | 0.4% |
| ridge (chosen) | 133.9% | 2.20 | −13.3% | 57 | 20.9% |
| random_forest (chosen) | 123.5% | 2.01 | −22.8% | 27 | 9.2% |
| xgboost (chosen) | 46.3% | 1.15 | −33.1% | 38 | 8.6% |

Cost assumption: 0.15% round trip. Arithmetic for trading every day:
**0.15% × 250 ≈ 37.5% per year** before any model is fitted. That alone kills
most daily strategies.

## Findings

Directional accuracy landed in the low fifties. That is the honest range for
next-day equity returns. Anything north of about 56% on this problem is a
leakage warning, not skill — and the guard did not fire.

R² on returns is roughly zero. A slightly negative out-of-sample R² is the
expected result. The random forest's +0.006 is noise, not an edge.

Every long/flat model lost to buy-and-hold after costs. NVDA rose hard in 2024.
Sitting in cash on some days is expensive when the asset itself is the trade.

## What it does

1. Download split-adjusted OHLCV via `yfinance` (`auto_adjust=True`).
2. Build 16 features from data at or before `t` (lagged returns, MA ratios, RSI, ATR, volume ratio).
3. Split by calendar year. No shuffling.
4. Fit ridge, random forest, and XGBoost on train only.
5. Pick a long/flat threshold on validation. Score the test year once.
6. Report RMSE, directional accuracy, Spearman IC, and a costed backtest against buy-and-hold.

Baselines are mandatory table rows: predict zero, predict the training-mean
return, always-long / majority direction.

## Setup

```bash
git clone https://github.com/DharmpratapSingh/Stock-Price-Prediction-Using-Machine-Learning.git
cd Stock-Price-Prediction-Using-Machine-Learning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
```

Config for this experiment lives under `experiment:` in `config/config.yaml`.
Re-download prices with `--refresh`. The original notebook and PDF are under
`legacy/`; they are not the experiment this README reports.

## Limitations

Single ticker. Daily frequency. Long/flat only — no shorts, no sizing.
No walk-forward across many windows; one fixed year split.
Transaction costs are a flat 7.5 bps per side, not a live broker schedule.
2024 NVDA is a strong upward regime; results will not generalize by assertion.

## License

MIT. Not investment advice.
