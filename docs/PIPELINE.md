# Full pipeline: walk-forward next-day return prediction

Predicts the **next-day log return** (and its sign) for a basket of liquid US
equities — SPY, AAPL, MSFT, NVDA, JPM — using daily OHLCV from 2015-01-01 to
2024-12-31 and 96 engineered technical features (48 after dropping price levels;
see §3). Validation is walk-forward
only, with an embargo; every transform is fitted inside the training window; every
model is scored against baselines that cost nothing to compute.

**The headline finding is that next-day returns are close to unpredictable, and the
project's value is that it establishes this rigorously rather than hiding it.**
Pooled over 7,875 out-of-sample days, the best model's return R² is **-0.003** —
worse than forecasting zero. Directional accuracy tops out at **53.4% [52.2, 54.5]**,
which beats a coin flip but *not* the always-up baseline (53.8%), because equities
drift upward. Net of 15 bps per side, **1 of 15** ticker×model backtests beat
buy-and-hold, and that one is not statistically significant.

Every number below comes from a committed file under [`results/`](../results).

---

## 1. Results

### 1.1 Return regression — pooled across all five tickers

Out-of-sample log returns, 7,875 days, 25 walk-forward folds per ticker.
Full per-ticker breakdown: [`results/return_regression.md`](../results/return_regression.md).

| Model | RMSE | R² | R² vs zero-forecast |
|---|---|---|---|
| Random Forest | 0.0217 | **-0.0034** | -0.0014 |
| XGBoost | 0.0222 | -0.0451 | -0.0430 |
| Ridge | 0.0222 | -0.0471 | -0.0450 |
| *Baseline: train-window mean* | 0.0217 | -0.0007 | +0.0013 |
| *Baseline: zero return* | 0.0217 | -0.0020 | 0.0000 |
| *Baseline: yesterday's return* | 0.0322 | -1.2065 | -1.2021 |

Every model lands at or below zero. The best of them (Random Forest, R² = -0.0034)
is beaten by the two-line baseline that always predicts the training window's mean
return. The persistence baseline — "tomorrow's return will look like today's" — is
catastrophically bad (R² = -1.21), which is the direct measurement that daily
returns carry essentially no first-order autocorrelation.

Per-ticker R² ranges from -0.064 (NVDA, Ridge) to +0.007 (AAPL, Random Forest).
That single positive value is the largest return R² anywhere in the study.

### 1.2 Direction — pooled

Full table incl. per-ticker: [`results/direction.md`](../results/direction.md).
`p vs 0.5` is a binomial test against a coin flip; `p vs base` is against the
training-window up-frequency, i.e. what the always-up baseline gets for free.

| Model | n | Accuracy | 95% CI (Wilson) | p vs 0.5 | p vs base | Precision (up) | Recall (up) |
|---|---|---|---|---|---|---|---|
| Random Forest | 7875 | 53.35% | [52.24, 54.45] | 3.0e-09 | 0.477 | 54.68% | 77.23% |
| Ridge / Logistic | 7875 | 52.01% | [50.91, 53.11] | 0.0004 | 0.0020 | 54.52% | 64.81% |
| XGBoost | 7875 | 51.71% | [50.60, 52.81] | 0.0025 | 0.0003 | 54.35% | 63.56% |
| *Baseline: always up* | 7875 | **53.77%** | [52.66, 54.86] | 2.5e-11 | 0.982 | 53.77% | 100% |

This table is the reason the direction task needs two null hypotheses. Against a
coin flip, all three models are "significant" (p < 0.01) — a result that would look
publishable. Against the always-up baseline, **none of them wins**: Random Forest
ties it (p = 0.48, and its point estimate is *lower*), while Ridge and XGBoost are
significantly **worse** (p = 0.002 and p = 0.0003). The models are rediscovering
market drift, not direction.

Best single cell in the whole study: AAPL / Random Forest at 55.11% [52.65, 57.55],
p = 0.13 against that ticker's baseline. One promising cell out of 15 tested
combinations, at p = 0.13, is what noise looks like.

### 1.3 Backtest vs buy-and-hold

Long/flat, one position per day, sized at full capital, net of 10 bps commission +
5 bps slippage **per side**, charged on entry, on every position change, and on the
final liquidation. Mean across the five tickers; per-ticker rows in
[`results/backtest.md`](../results/backtest.md).

| Strategy | Total return | Annualised | Sharpe | Max drawdown | Turnover | Time in market |
|---|---|---|---|---|---|---|
| **Buy & Hold** | **612.2%** | **29.01%** | **0.824** | 43.9% | 2 | 100% |
| Random Forest | 536.2% | 23.27% | 0.723 | 39.9% | 322 | 78.0% |
| XGBoost | 177.2% | 14.71% | 0.522 | 46.7% | 452 | 70.2% |
| Ridge | 118.3% | 9.36% | 0.310 | 37.9% | 576 | 55.2% |
| *Baseline: train-window mean* | 612.2% | 29.01% | 0.824 | 43.9% | 2 | 100% |
| *Baseline: yesterday's return* | -52.8% | -12.19% | -0.661 | 67.4% | 788 | 53.8% |

No model beats buy-and-hold on average, on return or on Sharpe. Two details worth
reading carefully:

- The **train-window-mean baseline reproduces buy-and-hold exactly**. Its forecast
  is a positive constant, so it is always long. Any strategy whose "edge" is being
  long most of the time has found drift, not signal.
- Random Forest achieves a **lower max drawdown** (39.9% vs 43.9%) while holding
  the asset 78% of the time. That is the honest version of what these models do:
  mild de-risking, not forecasting.

Exactly one of fifteen ticker×model runs beat buy-and-hold at the headline cost
level — AAPL / Random Forest, +255.6pp, Sharpe 1.60 vs 0.87. Its *return* R² is
+0.007, i.e. essentially zero, so the outperformance comes from timing luck, not
forecast accuracy.

### 1.4 Cost sensitivity

Mean across tickers, per-side cost swept from 0 to 20 bps.
Full grid: [`results/cost_sensitivity.md`](../results/cost_sensitivity.md); the
15 bps row is the headline configuration and comes from
[`results/backtest.md`](../results/backtest.md).

| Per-side cost | Excess vs buy-and-hold: RF | Ridge | XGBoost | Runs beating buy-and-hold |
|---|---|---|---|---|
| 0 bps | **+279.9pp** | -195.1pp | -156.0pp | 7 of 15 |
| 5 bps | +143.1pp | -324.6pp | -271.5pp | 4 of 15 |
| 10 bps | +25.3pp | -421.4pp | -362.8pp | 4 of 15 |
| 15 bps (headline) | -76.0pp | -493.8pp | -435.0pp | 1 of 15 |
| 20 bps | -163.1pp | -547.9pp | -492.0pp | 1 of 15 |

This is the single most informative table here. At zero cost, Random Forest looks
like a real strategy (+280pp over buy-and-hold, 7 of 15 runs winning). The entire
apparent edge is consumed between 0 and 15 bps — a realistic retail cost — because
the strategy turns over ~322 times where buy-and-hold turns over twice. A backtest
quoted without a cost sweep is not a result.

---

## 2. Why not R² on prices

The same models, the same folds, scored two ways. Full table:
[`results/level_r2_trap.md`](../results/level_r2_trap.md); figure:
[`results/level_r2_trap.png`](../results/level_r2_trap.png).

| Ticker | Persistence: ŷ(t+1) = C(t) | Ridge on levels | Ridge, same model in return space |
|---|---|---|---|
| SPY | 0.9978 | 0.9971 | -0.0544 |
| AAPL | 0.9981 | 0.9978 | -0.0172 |
| MSFT | 0.9981 | 0.9978 | -0.0404 |
| NVDA | 0.9981 | 0.9979 | -0.0636 |
| JPM | 0.9964 | 0.9959 | -0.0356 |

Predicting the next day's price *level* scores R² ≈ 0.998 on every ticker — but so
does the persistence forecast "tomorrow's close equals today's close", which
contains no model at all and in fact **beats every trained model on every ticker**.
R² on levels measures the autocorrelation of the price series, not forecasting
skill: prices move by well under 1% a day, so today's close already explains ~99.8%
of tomorrow's variance before anyone fits anything. Move the identical models to
the quantity a trader actually needs — the return — and the same fits score
approximately zero.

One extra finding worth keeping: Random Forest scores only 0.32–0.64 on levels,
far *worse* than the linear model. Trees cannot extrapolate beyond their training
range, so when NVDA's price leaves the training window the forest's predictions
saturate. A model can score 0.998 or 0.32 on the same task purely from how it
handles a trend — more evidence that the level score describes the target's
statistics, not the model's skill.

---

## 3. Methodology — the design decisions

**Returns, not levels.** The target is `log(C(t+1)/C(t))`. Price levels are
non-stationary and trivially autocorrelated, so a level target hands a model a
0.998 R² for free (§2). Returns remove that floor: R² near zero is then a real
statement about predictability.

**Walk-forward only, with an embargo.** 25 expanding-window folds per ticker: a
504-day (~2y) initial training window, 63-day (~1 quarter) test folds, advancing
one quarter at a time. Between each training window and its test fold sits a
**199-row embargo**, computed as the feature warm-up length — the deepest rolling
lookback in the feature set. Without it, a test-fold feature row computed from a
200-day window would straddle the boundary and be partly built from training rows.
Nothing is ever shuffled. Fold boundaries with dates:
[`results/walk_forward_folds.md`](../results/walk_forward_folds.md).

**Everything fitted in-window.** Imputation, winsorisation, scaling and feature
selection (`SelectKBest`, top 40) all live inside an sklearn `Pipeline` that is
constructed fresh and refitted for every fold. Selecting features on the full
dataset before splitting — the original implementation — leaks the test period into
which columns survive. Outlier handling is a train-window quantile clip inside the
pipeline, never a full-sample statistic, and prices are never rebuilt from clipped
returns.

**Baselines first.** Zero-return, train-window-mean and persistence for regression;
always-up for direction; buy-and-hold for the backtest. These are rows in every
results table, not an afterthought. Two of the study's three most important
conclusions (§1.1, §1.2) are visible only because the baselines are there.

**Scale-free features only.** The model matrix keeps the 48 features that are
invariant to price level — returns, `dist_from_sma_*`, RSI, `bb_percent`, ATR/price,
stochastics, volume ratios, realised volatility, candle geometry — and drops raw
levels like `sma_50` and `obv`. A model fitted on 2015 price levels sees 2024 inputs
outside its entire training range.

**Cost-aware backtest.** Long/flat, 10 bps commission + 5 bps slippage per side,
charged on entry, on every position change and on the final liquidation. The
strategy and the benchmark run through the same code path and are scored over the
same bars, so neither gets a free bar. Reported with a cost sweep (§1.4) and
turnover, because a strategy that only wins at zero cost has not found anything.

**Fixed hyperparameters, no tuning on test.** Three families — Ridge/Logistic,
Random Forest, XGBoost — with documented defaults in `src/models.py`
(`DEFAULT_PARAMS`), chosen once from priors about return noise: shallow trees
(`max_depth=3–5`), large leaves (`min_samples_leaf=50`), strong regularisation
(`alpha=10`, `reg_lambda=5`). Nothing was adjusted after seeing a test fold.

**Snapshot data.** OHLCV is committed as CSV under `data/raw/`. The loader reads
snapshots by default and only touches the network behind an explicit flag, so every
number above reproduces offline and cannot drift when a data vendor restates history.

---

## 4. How to run

Verified end to end on Python 3.11.15 (numpy 2.3.5, pandas 3.0.2, scikit-learn
1.8.0, xgboost 3.2.0).

```bash
pip install -r requirements.txt

# Full basket -> results/           (~2 min)
python train.py --basket

# Default single ticker, NVDA 2018-2024 -> results/nvda_2018_2024/   (~15 s)
python train.py --results-dir results/nvda_2018_2024

# One ticker, no figures
python train.py --ticker SPY --no-figures

# Forecast from the saved artifact
python predict.py --model models/NVDA_linear.joblib --recent 3

# Refresh the committed snapshots from Yahoo Finance (the only networked path)
python train.py --basket --download

# Tests
python -m pytest -q          # 93 passed
```

`train.py` writes every table as both `.csv` and `.md`, plus three figures
(`equity_curves.png`, `level_r2_trap.png`, `directional_accuracy.png`), and saves
one artifact to `models/` holding the fitted pipeline, the exact ordered feature
list and the config that built it. `predict.py` consumes that artifact and rebuilds
features through the same `build_dataset` call, so training and inference cannot
drift apart — a mismatch raises rather than silently producing a wrong number.

---

## 5. Layout

```
├── src/
│   ├── data_loader.py          # snapshot-first OHLCV loading + validation
│   ├── feature_engineering.py  # indicators, targets, warm-up measurement
│   ├── feature_selection.py    # correlation reporting (in-window only)
│   ├── models.py               # 3 families, baselines, fit-in-window pipeline
│   ├── evaluation.py           # return/direction metrics, walk-forward engine
│   ├── backtesting.py          # cost-aware long/flat backtest + benchmark
│   └── utils.py                # config, folds with embargo, artifact I/O
├── config/config.yaml
├── data/raw/                   # committed OHLCV snapshots (5 tickers)
├── results/                    # committed tables + figures
│   └── nvda_2018_2024/         # single-ticker default run
├── legacy/                     # superseded notebook + PDF
├── tests/                      # 93 tests
├── train.py                    # walk-forward pipeline
└── predict.py                  # inference CLI
```

---

## 6. Limitations

Stated plainly, because they bound every number above.

- **Daily frequency only.** Any genuine short-horizon signal is likely intraday;
  this study cannot see it.
- **Five tickers, one regime-blind model.** Large-cap US equities over a decade
  that includes a historic bull run. No regime detection, so a single model is
  asked to describe 2018, the 2020 crash and 2023–24 alike.
- **No position sizing or risk management.** Long/flat at full capital. No
  volatility targeting, stops, or portfolio construction — all of which matter more
  to realised performance than the forecast does.
- **Multiple comparisons.** 15 ticker×model combinations were evaluated. The one
  that beat buy-and-hold (AAPL / Random Forest) should be read with that in mind;
  no correction was applied because no cell survives it anyway.
- **Costs are modelled, not measured.** 10 bps + 5 bps per side is a reasonable
  retail assumption, not a fill-level simulation. No market impact, no borrow
  costs, no slippage that scales with size.
- **Survivorship.** The basket was chosen in 2024 from tickers that are liquid
  today.
- **Not investment advice.** This is a methodology demonstration. Historical
  results, including these, do not predict future returns.
