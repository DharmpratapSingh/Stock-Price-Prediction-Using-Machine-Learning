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
| ridge | 0.03309 | 50.2% | 0.069 | −0.004 |
| random_forest | 0.03294 | 53.8% | 0.084 | +0.005 |
| xgboost | 0.03356 | 51.4% | −0.021 | −0.033 |

| strategy | net return | Sharpe | max DD | round trips | cost drag |
| --- | --- | --- | --- | --- | --- |
| buy_and_hold | 178.4% | 2.22 | −27.0% | 1 | 0.4% |
| ridge (chosen) | 130.1% | 2.16 | −14.7% | 57 | 20.5% |
| random_forest (chosen) | 123.6% | 2.01 | −22.8% | 29 | 9.9% |
| xgboost (chosen) | 54.5% | 1.28 | −21.7% | 41 | 9.8% |

Every Sharpe in that table is high because NVDA itself returned ~178% in 2024, so
the absolute figures say more about the year than about any model. The only
comparison that means anything is model versus the `buy_and_hold` row — and no
model beats it after costs.

Cost assumption: 0.15% round trip. Arithmetic for trading every day:
**0.15% × 250 ≈ 37.5% per year** before any model is fitted. That alone kills
most daily strategies.

## Findings

Directional accuracy landed in the low fifties. That is the honest range for
next-day equity returns. Anything north of about 56% on this problem is a
leakage warning, not skill — and the guard did not fire.

R² on returns is roughly zero. A slightly negative out-of-sample R² is the
expected result. The random forest's +0.005 is noise, not an edge.

Every long/flat model lost to buy-and-hold after costs. NVDA rose hard in 2024.
Sitting in cash on some days is expensive when the asset itself is the trade.

Ridge's information coefficient is −0.115 on validation and +0.069 on test. A
predictor whose correlation with the target flips sign between two adjacent years
has not found a stable relationship; that is what noise looks like.

## Also here: the five-ticker walk-forward pipeline

The experiment above is one ticker and one fixed year split, chosen to be easy to
audit. The repository also contains the general version: `train.py` runs
expanding-window **walk-forward** validation over **SPY, AAPL, MSFT, NVDA and JPM**
from 2015–2024 — 25 folds per ticker, a 200-row embargo between each training
window and its test fold, three model families against zero-return,
train-mean, persistence and always-up baselines, and a cost-aware long/flat
backtest with a 0–20 bps cost sweep. It reaches the same conclusion over 7,875
out-of-sample days: pooled return R² of −0.003, directional accuracy that beats a
coin flip but not the always-up baseline, and 1 of 15 ticker×model backtests
beating buy-and-hold after costs.

Full write-up in [`docs/PIPELINE.md`](docs/PIPELINE.md); committed tables and
figures in [`results/pipeline/`](results/pipeline).

```bash
python train.py --basket                          # five tickers -> results/pipeline/
python predict.py --model models/JPM_linear.joblib # artifact the basket run saved
```

`--basket` saves one inference artifact, for the last ticker it evaluates (JPM).
`models/` is gitignored, so a fresh clone must run `train.py` before `predict.py`.
To get a different ticker's artifact, run that ticker on its own — but send it to
its own output directory so it does not overwrite the basket tables:
`python train.py --ticker NVDA --results-dir results/pipeline/nvda_2018_2024`.

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

Config for this experiment lives under `experiment:` in `config/config.yaml`; the
sections above it configure the walk-forward pipeline. Re-download prices with
`--refresh`. Tests for both paths: `python -m pytest -q`. The original notebook and
PDF are under `legacy/`; they are not the experiment this README reports.

## Repository layout

```
├── run_experiment.py       # this README's NVDA experiment -> RESULTS.md
├── train.py                # five-ticker walk-forward pipeline -> results/pipeline/
├── predict.py              # inference from a train.py artifact
├── RESULTS.md              # experiment write-up
├── docs/PIPELINE.md        # pipeline write-up
├── config/config.yaml      # pipeline sections + experiment: section
├── src/                    # experiment modules (lean_*, splits, baselines,
│                           #   return_metrics, threshold_backtest, market_data)
│                           #   and pipeline modules (feature_engineering,
│                           #   models, evaluation, backtesting, data_loader, utils)
├── data/
│   ├── NVDA_adjusted.csv   # experiment price cache
│   └── raw/                # committed OHLCV snapshots for the five tickers
├── results/                # experiment tables; pipeline/ holds the pipeline's
│                           #   tables and figures
├── tests/                  # 291 tests across both paths
└── legacy/                 # superseded notebook + PDF
```

## Limitations

Single ticker. Daily frequency. Long/flat only — no shorts, no sizing.
No walk-forward across many windows in this experiment; one fixed year
split. (The walk-forward study is the pipeline section above.)
Transaction costs are a flat 7.5 bps per side, not a live broker schedule.
2024 NVDA is a strong upward regime; results will not generalize by assertion.

## License

MIT. Not investment advice.
