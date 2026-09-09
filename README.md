# Stock Price Prediction Using Machine Learning

Next-day closing-price prediction for a single ticker (NVDA by default), built twice: once as a short
notebook that gets the naive version working end to end, and once as a `src/` pipeline that redoes it
with chronological splits, ~96 engineered features, hyperparameter search over `TimeSeriesSplit` folds,
and a backtest that charges commission and slippage.

The most useful thing in this repository is not the R². It is the section below explaining why that R²
is close to meaningless, and what should replace it.

## The headline number is a trap

These are the results actually recorded in the repo — the executed outputs of
`Stock Price Prediction using Machine Learning.ipynb`, written up in
`Stock Price Prediction Using Machine Learning.pdf`. NVDA, 2018-01-01 to 2024-01-01, 80/20 split,
target = next day's closing price.

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 0.4033 | 0.9975 |
| Random Forest | 0.4954 | 0.9970 |
| XGBoost | 0.6101 | 0.9963 |

R² ≈ 0.997 looks like a solved problem. It is not, and two things in the repo's own artifacts give it away:

1. **Linear regression wins.** Over `[Close, Returns, 10-day MA, 50-day MA]`, a linear model can do
   little more than emit a scaled copy of today's close. That it beats both tree ensembles means the
   score is not rewarding learned structure.
2. **`Close` carries roughly 0.88 of the Random Forest's feature importance** (the feature-importance
   plot in the notebook and the PDF). The model's dominant input is today's price, and its output is
   approximately today's price.

Daily equity prices are close to a random walk: tomorrow's *level* is today's level plus a small
increment. So the persistence forecast — "predict today's close for tomorrow" — already explains
almost all the variance in price levels on its own. **R² computed on price levels is therefore
measuring the autocorrelation of the price series, not forecasting skill.** A model can score 0.99 and
be worth nothing, because the sliver it misses is the entire tradeable signal.

The honest comparison is cheap and this repo does not yet run it: score `ŷ(t+1) = Close(t)` with the
same `r2_score` on the same test split, then report how much each model beats that baseline. If the
gap is not material, the model is not adding anything. That delta, not the raw R², is the number this
project should be judged on.

**What would actually be meaningful here:**

- **Directional accuracy** — did the model get the sign of the move right? Only a durable margin over
  50%, net of costs, is real. `ModelEvaluator.directional_accuracy` in `src/evaluation.py` computes it,
  but no run in this repo has committed a value, so no directional number is claimed here.
- **Predicting returns instead of levels.** `create_target_variable(..., target_type='return')` already
  exists in `src/feature_engineering.py` and removes the persistence floor entirely. The pipeline
  currently calls it with `target_type='price'`.
- **Backtested return net of commission and slippage, against buy-and-hold.** `src/backtesting.py`
  implements this. Again, no run is committed.

No backtest or directional figures appear in this README because there are no committed results to
quote — `results/`, `models/` and `logs/` are gitignored.

## What the pipeline does

- **Data** — daily OHLCV from Yahoo Finance via `yfinance` (`src/data_loader.py`), with schema
  validation, missing-value handling and outlier treatment.
- **Features** — 96 engineered columns from `src/feature_engineering.py`: lags of O/H/L/C, simple and
  log returns, SMA/EMA plus distance-from-MA, RSI, MACD, Bollinger Bands, ATR, stochastic oscillator,
  OBV / VPT / volume ratio, realized and Parkinson volatility, candle body / shadows / gap, momentum
  and ROC, rolling linear-regression trend slopes, and a simplified ADX. All hand-written in
  pandas/numpy — TA-Lib is listed in `requirements.txt` but never imported.
- **Target** — `Close.shift(-1)`, i.e. the next day's price *level*. See the section above.
- **Split** — chronological only (`time_series_split` in `src/utils.py`): train, then validation, then
  test, in time order, never shuffled.
- **Models** — linear regression, random forest and XGBoost are trained by default; LightGBM and an
  LSTM are also implemented in `src/models.py`. Tuning is `RandomizedSearchCV` over `TimeSeriesSplit`
  folds (`ModelTuner`).
- **Backtest** — `src/backtesting.py` runs a threshold strategy against buy-and-hold with 0.1%
  commission and 0.05% slippage, and reports Sharpe, Sortino, Calmar, max drawdown, win rate and
  profit factor.

## Setup

```bash
git clone https://github.com/DharmpratapSingh/Stock-Price-Prediction-Using-Machine-Learning.git
cd Stock-Price-Prediction-Using-Machine-Learning
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is heavier than the code needs. TA-Lib (platform-specific and awkward to install),
FastAPI and Streamlit are listed but never imported; TensorFlow is only needed for the LSTM. pandas,
numpy, scikit-learn, xgboost, yfinance, pyyaml, joblib and matplotlib cover the default path.

## Usage

Everything is driven by `config/config.yaml` — ticker, date range, split fractions, feature
parameters, hyperparameter grids, feature-selection method, and backtest costs.

```bash
python train.py                                  # linear regression, random forest, xgboost
python train.py --model random_forest            # one model
python train.py --config config/custom.yaml      # alternative config
```

Models are written to `models/<name>_<timestamp>.joblib`:

```bash
python predict.py --interactive
python predict.py --model models/random_forest_20240101_120000.joblib --symbol NVDA
python predict.py --model models/random_forest_20240101_120000.joblib --batch --symbols NVDA AMD TSM INTC
```

`predict.py` also accepts `--days N`, `--config`, and `--no-plot`. Tests: `pytest tests/ -v`.

## Known issues

Written down rather than papered over. Roughly the order I would fix them in.

1. **`train.py` does not currently run end to end.** `FeatureEngineer.create_all_features` looks for
   `config['bollinger']` and `config['stochastic']`, but `config/config.yaml` spells those as
   `bollinger_window` / `bollinger_std` / `stoch_k` / `stoch_d`. The fallback dicts then use keyword
   names the methods do not accept (`std`, `k`, `d` instead of `num_std`, `k_period`, `d_period`), so
   feature engineering raises `TypeError: create_bollinger_bands() got an unexpected keyword argument
   'std'`. The same mis-plumbing means the configured MACD, Bollinger and stochastic periods are
   silently ignored.
2. **`predict.py` builds a different feature set than `train.py`.** Training drops `Open`/`High`/`Low`
   and then applies feature selection; prediction keeps them and selects nothing, so a model saved by
   `train.py` is handed the wrong number of columns at inference time.
3. **Feature selection sees the test set.** `train.py` selects features on the full dataset *before*
   the chronological split, leaking test-period information into which features are kept. It should be
   fit on the training window only.
4. **Outlier cleaning uses full-sample statistics.** `handle_outliers` clips returns at ±3σ using the
   mean and standard deviation of the entire series, then rebuilds `Close` from the clipped returns —
   so test-period statistics touch the training data, and `Close` no longer reconciles with
   `Open`/`High`/`Low`.
5. **The backtest signal is not what its docstring describes.** `Backtester.simple_strategy` derives
   its signal from the change between *consecutive predictions*, not from the prediction relative to
   today's price — closer to momentum in the forecast than to a forecast-versus-spot signal.
6. **Walk-forward validation is implemented but not wired in.** `walk_forward_validation` and
   `WalkForwardBacktester` exist; setting `training.use_walk_forward: true` logs a note and proceeds
   with the standard split anyway.
7. Single ticker, single one-day horizon, no regime awareness, no position sizing or risk limits, and
   no sensitivity analysis on transaction costs.

## Layout

```
├── src/
│   ├── data_loader.py          # yfinance fetch, validation, cleaning
│   ├── feature_engineering.py  # technical indicators + target construction
│   ├── feature_selection.py    # correlation / importance / mutual-info / RFE
│   ├── models.py               # model wrappers + RandomizedSearchCV tuner
│   ├── ensemble.py             # average / weighted / stacking
│   ├── cache.py                # on-disk cache for fetched data
│   ├── evaluation.py           # statistical, directional and financial metrics
│   ├── backtesting.py          # cost-aware trading simulation
│   ├── visualize.py            # plots
│   └── utils.py                # config, logging, chronological split, persistence
├── config/config.yaml
├── tests/                      # pytest suites for features, models, evaluation,
│                               # backtesting, feature selection
├── train.py                    # training pipeline
├── predict.py                  # prediction CLI
├── Stock Price Prediction using Machine Learning.ipynb   # original baseline notebook
└── Stock Price Prediction Using Machine Learning.pdf     # write-up of that notebook
```

`data/`, `models/`, `results/`, `logs/` and `cache/` are created at runtime and gitignored.

## Disclaimer

Educational project. Not investment advice, and not something to trade with. Historical results —
including the ones above — do not predict future returns.

## License

MIT.
