# Full pipeline: walk-forward next-day return prediction

Predicts the **next-day log return** (and its sign) for a basket of liquid US
equities — SPY, AAPL, MSFT, NVDA, JPM — using daily OHLCV from 2015-01-01 to
2024-12-31 and 96 engineered technical features (48 after dropping price levels;
see §3). Validation is walk-forward only, with an embargo; every transform is
fitted inside the training window; every model is scored against baselines that
cost nothing to compute.

**The headline finding is that next-day returns are close to unpredictable, and the
project's value is that it establishes this rigorously rather than hiding it.**
Pooled over 7,875 out-of-sample days, the best model's return R² is **-0.003** —
worse than forecasting zero. Directional accuracy tops out at **53.3%**, which
beats a coin flip but *not* the always-up baseline (53.8%), because equities drift
upward. Net of 15 bps per side, **1 of 15** ticker×model backtests beat
buy-and-hold, and that one is not statistically significant.

Every number below comes from a committed file under [`results/pipeline/`](../results/pipeline).

---

## 1. Results

### 1.1 Return regression — pooled across all five tickers

Out-of-sample log returns, 7,875 days, 25 walk-forward folds per ticker.
Full per-ticker breakdown: [`results/pipeline/return_regression.md`](../results/pipeline/return_regression.md).

| Model | RMSE | R² | R² vs zero-forecast |
|---|---|---|---|
| Random Forest | 0.0218 | **-0.0034** | -0.0013 |
| XGBoost | 0.0222 | -0.0455 | -0.0433 |
| Ridge | 0.0222 | -0.0467 | -0.0445 |
| *Baseline: train-window mean* | 0.0218 | -0.0007 | +0.0013 |
| *Baseline: zero return* | 0.0218 | -0.0021 | 0.0000 |
| *Baseline: yesterday's return* | 0.0323 | -1.2009 | -1.1963 |

Every model lands at or below zero. The best of them (Random Forest, R² = -0.0034)
is beaten by the two-line baseline that always predicts the training window's mean
return. The persistence baseline — "tomorrow's return will look like today's" — is
catastrophically bad (R² = -1.20), which is the direct measurement that daily
returns carry essentially no first-order autocorrelation.

Across tickers, **model** R² ranges from -0.063 (NVDA, Ridge) to +0.007 (AAPL,
Random Forest); that single positive value is the largest model R² anywhere in the
study. The persistence baseline runs far lower still, reaching -1.35 (MSFT).

### 1.2 Direction — pooled

Full table incl. per-ticker: [`results/pipeline/direction.md`](../results/pipeline/direction.md).
`p vs 0.5` is a binomial test against a coin flip. `p vs base` is **McNemar's exact
test** against the always-up baseline — paired, because both predictors are scored
on the same rows; an unpaired test against the baseline's *rate* would ignore that
they agree on most days.

The five tickers are scored on identical calendar dates, and SPY mechanically
contains the other four, so the pooled panel carries less information than n =
7,875 suggests. The realised up/down indicator has a mean cross-ticker correlation
of **0.405**, giving a design effect of 1 + (m−1)ρ = **2.62** and an effective n of
**3,004**; the realised log returns correlate even more strongly (ρ = 0.624, deff =
3.50). Each model's interval below is widened by the design effect of *its own*
correctness indicator, which is the quantity being averaged — computed in
[`results/pipeline/pooled_dependence.md`](../results/pipeline/pooled_dependence.md).

| Model | Accuracy | iid 95% CI | deff | eff. n | **adjusted 95% CI** | adj. p vs 0.5 | McNemar p vs base (nominal → clustered) |
|---|---|---|---|---|---|---|---|
| Random Forest | 53.32% | [52.22, 54.42] | 1.47 | 5,342 | **[51.97, 54.65]** | 1.36e-06 | 0.448 → **0.528** |
| Ridge / Logistic | 52.08% | [50.97, 53.18] | 1.30 | 6,067 | **[50.81, 53.32]** | 0.0013 | 0.0140 → **0.0523** |
| XGBoost | 51.67% | [50.57, 52.77] | 1.25 | 6,281 | **[50.43, 52.90]** | 0.0087 | 0.0026 → **0.0179** |
| *Baseline: always up* | **53.75%** | [52.65, 54.85] | 2.62 | 3,004 | **[51.98, 55.54]** | 4.0e-05 | — |

Note the design effects differ by design. The always-up baseline's correctness
indicator *is* the market's up/down indicator, so it absorbs the full 2.62 and its
interval widens most. The models' errors decorrelate across tickers (ρ = 0.06–0.12),
so they retain more effective sample — but their accuracy is lower, and that is what
decides the question.

The clustering correction applies to the paired test as well. The signed
discordance `D = correct_model − correct_baseline` is itself cross-ticker
correlated (ρ_D = 0.11–0.16, deff 1.43–1.63), so the discordant pairs McNemar
counts are not independent either; the clustered column divides them by that panel's
own design effect before running the exact binomial.

This table is the reason the direction task needs two null hypotheses. Against a
coin flip, all three models remain significant even after the clustering
adjustment. Against the always-up baseline, **none of them wins**: Random Forest
ties it on either version of the test (p = 0.45 nominal, 0.53 clustered, and its
point estimate is *lower* — 929 days where it wins against 963 where the baseline
does), while **Ridge is borderline and XGBoost significantly worse (p = 0.014 →
0.052 and 0.0026 → 0.018 once the discordance panel's own design effect is
applied)**. The models are rediscovering market drift, not direction.

Best single cell in the study: AAPL / Random Forest at 54.98% [52.52, 57.43],
McNemar p = 0.26 against that ticker's baseline. One promising cell out of 15 tested
combinations, at p = 0.26, is what noise looks like.

### 1.3 Backtest vs buy-and-hold

Long/flat, one position per day, sized at full capital, net of 10 bps commission +
5 bps slippage **per side**, charged on entry, on every position change, and on the
final liquidation. Mean across the five tickers; per-ticker rows in
[`results/pipeline/backtest.md`](../results/pipeline/backtest.md). `Sharpe` is (CAGR − rf) / annualised
volatility with rf = 0; the textbook arithmetic Sharpe is shown beside it (see §3).

| Strategy | Total return | Annualised | Sharpe | Sharpe (arith.) | Max drawdown | Turnover | Time in market |
|---|---|---|---|---|---|---|---|
| **Buy & Hold** | **634.3%** | **29.68%** | **0.843** | **0.894** | 43.9% | 2 | 100% |
| Random Forest | 556.4% | 24.02% | 0.750 | 0.776 | 39.4% | 321 | 78.0% |
| XGBoost | 179.7% | 15.09% | 0.538 | 0.617 | 47.1% | 451 | 70.2% |
| Ridge | 121.6% | 9.68% | 0.322 | 0.413 | 38.1% | 577 | 55.1% |
| *Baseline: train-window mean* | 634.3% | 29.68% | 0.843 | 0.894 | 43.9% | 2 | 100% |
| *Baseline: yesterday's return* | -51.0% | -11.73% | -0.636 | -0.576 | 67.4% | 790 | 53.8% |

No model beats buy-and-hold on average, on return or on either Sharpe. Two details
worth reading carefully:

- The **train-window-mean baseline reproduces buy-and-hold exactly**. Its forecast
  is a positive constant, so it is always long. Any strategy whose "edge" is being
  long most of the time has found drift, not signal.
- Random Forest achieves a **lower max drawdown** (39.4% vs 43.9%) while holding
  the asset 78% of the time. That is the honest version of what these models do:
  mild de-risking, not forecasting.

Mean excess versus buy-and-hold at the headline cost: Random Forest **-78.0pp**,
XGBoost -454.7pp, Ridge -512.7pp. Exactly one of fifteen ticker×model runs beat
buy-and-hold — AAPL / Random Forest, +263.8pp, Sharpe 1.61 vs 0.87. Its *return* R²
is +0.007, i.e. essentially zero, so the outperformance comes from timing luck, not
forecast accuracy.

### 1.4 Cost sensitivity

Mean excess return versus buy-and-hold, per-side cost swept from 0 to 20 bps.
Full grid: [`results/pipeline/cost_sensitivity.md`](../results/pipeline/cost_sensitivity.md); the
15 bps row is the headline configuration and comes from
[`results/pipeline/backtest.md`](../results/pipeline/backtest.md).

| Per-side cost | Random Forest | Ridge | XGBoost | Runs beating buy-and-hold |
|---|---|---|---|---|
| 0 bps | **+289.7pp** | -208.9pp | -175.1pp | 8 of 15 |
| 5 bps | +148.3pp | -340.6pp | -290.7pp | 4 of 15 |
| 10 bps | +26.6pp | -439.1pp | -382.3pp | 4 of 15 |
| 15 bps (headline) | -78.0pp | -512.7pp | -454.7pp | 1 of 15 |
| 20 bps | -167.9pp | -567.6pp | -511.9pp | 1 of 15 |

This is the single most informative table here. At zero cost, Random Forest looks
like a real strategy (+290pp over buy-and-hold, 8 of 15 runs winning). The entire
apparent edge is consumed between 0 and 15 bps — a realistic retail cost — because
the strategy turns over ~321 times where buy-and-hold turns over twice. A backtest
quoted without a cost sweep is not a result.

---

## 2. Why not R² on prices

The same models, the same folds, scored two ways. Full table:
[`results/pipeline/level_r2_trap.md`](../results/pipeline/level_r2_trap.md); figure:
[`results/pipeline/level_r2_trap.png`](../results/pipeline/level_r2_trap.png).

| Ticker | Persistence: ŷ(t+1) = C(t) | Ridge on levels | Ridge, same model in return space |
|---|---|---|---|
| SPY | 0.9978 | 0.9971 | -0.0544 |
| AAPL | 0.9981 | 0.9978 | -0.0173 |
| MSFT | 0.9981 | 0.9978 | -0.0402 |
| NVDA | 0.9981 | 0.9979 | -0.0634 |
| JPM | 0.9961 | 0.9956 | -0.0344 |

Predicting the next day's price *level* scores R² ≈ 0.998 on every ticker — but so
does the persistence forecast "tomorrow's close equals today's close", which
contains no model at all and in fact **beats every trained model on every ticker**.
R² on levels measures the autocorrelation of the price series, not forecasting
skill: prices move by well under 1% a day, so today's close already explains ~99.8%
of tomorrow's variance before anyone fits anything. Move the identical models to
the quantity a trader actually needs — the return — and the same fits score
approximately zero.

One extra finding worth keeping: Random Forest scores only 0.32–0.64 on levels, far
*worse* than the linear model. Trees cannot extrapolate beyond their training range,
so when NVDA's price leaves the training window the forest's predictions saturate. A
model can score 0.998 or 0.32 on the same task purely from how it handles a trend —
more evidence that the level score describes the target's statistics, not the
model's skill.

---

## 3. Methodology — the design decisions

**Returns, not levels.** The target is `log(C(t+1)/C(t))`. Price levels are
non-stationary and trivially autocorrelated, so a level target hands a model a
0.998 R² for free (§2). Returns remove that floor: R² near zero is then a real
statement about predictability.

**Walk-forward only, with an embargo.** 25 expanding-window folds per ticker: a
504-day (~2y) initial training window, 63-day (~1 quarter) test folds, advancing one
quarter at a time. Between each training window and its test fold sits a **200-row
embargo**, computed as the feature warm-up length (199 — the deepest rolling lookback
in the feature set) **plus the one-day forecast horizon**. The lookback term stops a
test-fold feature row from being built out of training rows; the horizon term
accounts for the last training row's target reaching one bar forward, so the raw
bars behind train and test are strictly disjoint. Nothing is ever shuffled. Fold
boundaries with dates: [`results/pipeline/walk_forward_folds.md`](../results/pipeline/walk_forward_folds.md).

**Everything fitted in-window.** Imputation, winsorisation, scaling and feature
selection (`SelectKBest`, top 40) all live inside an sklearn `Pipeline` that is
constructed fresh and refitted for every fold. Selecting features on the full
dataset before splitting — the original implementation — leaks the test period into
which columns survive. Outlier handling is a train-window quantile clip inside the
pipeline, never a full-sample statistic, and prices are never rebuilt from clipped
returns. The reported feature importances and feature/target correlations are also
fitted on a single training window, and each table names the window it used.

**Baselines first.** Zero-return, train-window-mean and persistence for regression;
always-up for direction; buy-and-hold for the backtest. These are rows in every
results table, not an afterthought. Two of the study's three most important
conclusions (§1.1, §1.2) are visible only because the baselines are there.

**Paired tests and clustered intervals.** Comparing a model with a baseline on the
same rows uses McNemar's exact test, not a one-sample binomial against the
baseline's rate. Pooled intervals are widened by the panel's design effect, because
five tickers scored on the same dates — one of which contains the other four — do
not supply five times the information. The same correction is carried through to
McNemar itself, since the signed discordance between two predictors is correlated
across tickers too; both the nominal and the clustered p-value are reported (§1.2).
Intervals and p-values at an effective sample size are computed from the same
rounded effective n, so a reader never sees a CI and a p-value derived from
different amounts of data.

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
**Sharpe definition:** the `Sharpe` column is (CAGR − rf) / annualised volatility
with rf = 0 — geometric numerator, arithmetic denominator, i.e. realised growth per
unit of volatility. The textbook arithmetic form, mean excess period return / period
σ × √252, is reported alongside as `sharpe_arithmetic`; it is the higher of the two
for a volatile path because compounding penalises variance. §1.3 quotes both.

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

# Default single ticker, NVDA 2018-2024 -> results/pipeline/nvda_2018_2024/   (~15 s)
python train.py --results-dir results/pipeline/nvda_2018_2024

# One ticker, no figures
python train.py --ticker SPY --no-figures

# Forecast from the saved artifact
python predict.py --model models/NVDA_linear.joblib --recent 3

# Refresh the committed snapshots from Yahoo Finance (the only networked path)
python train.py --basket --download

# Tests
python -m pytest -q          # 107 passed
```

`train.py` writes every table as both `.csv` and `.md`, plus three figures
(`equity_curves.png`, `level_r2_trap.png`, `directional_accuracy.png`), and saves
one artifact to `models/` holding the fitted pipeline, the exact ordered feature
list and the config that built it. `predict.py` consumes that artifact and rebuilds
features through the same `build_dataset` call, so training and inference cannot
drift apart — a mismatch raises rather than silently producing a wrong number.

**Artifacts.** `--basket` saves one inference artifact, for the last ticker it
evaluates (JPM). `models/` is gitignored, so a fresh clone must run `train.py`
before `predict.py`. To get a different ticker's artifact, run that ticker on its
own — and send it to its own output directory so it does not overwrite the basket
tables: `python train.py --ticker NVDA --results-dir results/pipeline/nvda_2018_2024`.

---

## 5. Layout

The repository holds two independent code paths over the same idea. The single-ticker
NVDA experiment (`run_experiment.py`, results in `RESULTS.md`) is the README's
headline; this document covers the five-ticker walk-forward pipeline (`train.py`).
They share no modules, so neither can silently change the other's numbers.

```
├── train.py                    # walk-forward pipeline (this document)
├── predict.py                  # inference CLI for the pipeline artifact
├── run_experiment.py           # single-ticker NVDA experiment -> RESULTS.md
├── src/
│   │                           # -- walk-forward pipeline --
│   ├── data_loader.py          # snapshot-first OHLCV loading + validation
│   ├── feature_engineering.py  # indicators, targets, warm-up measurement
│   ├── feature_selection.py    # correlation reporting (in-window only)
│   ├── models.py               # 3 families, baselines, fit-in-window pipeline
│   ├── evaluation.py           # return/direction metrics, McNemar, design
│   │                           #   effect, walk-forward engine
│   ├── backtesting.py          # cost-aware long/flat backtest + benchmark
│   ├── utils.py                # config, folds with embargo, artifact I/O
│   │                           # -- NVDA experiment --
│   ├── market_data.py          # price download + local cache
│   ├── lean_features.py        # experiment feature set
│   ├── lean_models.py          # experiment models
│   ├── splits.py               # year-based train/val/test split
│   ├── baselines.py            # experiment baselines
│   ├── return_metrics.py       # experiment metrics
│   ├── threshold_backtest.py   # threshold sweep + cost-aware backtest
│   └── _validation.py          # shared 1-D input validation
├── config/config.yaml          # pipeline sections + experiment: section
├── data/
│   ├── NVDA_adjusted.csv       # experiment price cache
│   └── raw/                    # committed OHLCV snapshots (5 tickers)
├── results/                    # experiment tables at top level
│   └── pipeline/               # walk-forward pipeline tables + figures
│       └── nvda_2018_2024/     # single-ticker default run of the pipeline
├── docs/PIPELINE.md            # this document
├── RESULTS.md                  # experiment write-up
├── legacy/                     # superseded notebook + PDF
└── tests/                      # 107 tests across both paths
```

---

## 6. Limitations

Stated plainly, because they bound every number above.

- **Daily frequency only.** Any genuine short-horizon signal is likely intraday;
  this study cannot see it.
- **Survivorship and selection in the basket.** The five tickers were chosen in
  2024 from names that are liquid *today*, and three of them — AAPL, MSFT and
  especially NVDA — are among the biggest equity winners of the decade. SPY then
  overlaps the other four outright. This is not a neutral sample: it is why
  buy-and-hold compounds to ~634% and why that is such a punishing bar for any
  strategy to clear. A basket including the decade's losers and delisted names
  would lower the benchmark and could easily flatter the models by comparison.
- **Pooled statistics are not iid.** Because the tickers share calendar dates and
  SPY contains the rest, pooled intervals are widened by a measured design effect
  (§1.2). Even so, the effective n is an approximation from a mean pairwise
  correlation, not a full panel model with time-varying dependence.
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
- **Not investment advice.** This is a methodology demonstration. Historical
  results, including these, do not predict future returns.
