# Module 07 — Time Series Forecasting Foundations

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 34 hours (12 lecture, 12 lab, 10 project)
> **Track fit:** Common core
> **Prerequisites:** Module 01A (metrics, units, time buckets); Module 06 (splits, leakage, evaluation); basic regression
> **Tools:** Python 3.12, pandas, NumPy, statsmodels, statsforecast, matplotlib, DuckDB; optional sktime and GluonTS
> **Industry question:** "What is the best forecast we can defend on Monday morning, at this grain and horizon, given that a naive benchmark is free?"

---

## 1. Why industry cares

Almost every planning decision consumes a number that somebody produced by extrapolating a history. Replenishment, production scheduling, labour, transportation capacity, procurement, and financial planning all inherit the error of that number, amplified by lead time and by the policy that converts a forecast into an order.

The business problem is not "which algorithm". It is that the forecast is produced under a deadline, at a grain the planner can act on, and it will be compared with a naive baseline that costs nothing to compute. A method that cannot beat seasonal naive at the decision grain has no business being in production, however sophisticated it looks in a paper.

Four facts shape the 2026 landscape:

1. **Benchmarks are strong.** Naive and seasonal-naive forecasts are hard to beat on short horizons and low-volume series. Most published gains come from cross-learning across many series, exogenous variables, and combination — not from a single better model.
2. **Machine learning won the large retail competitions.** In the M5 Accuracy competition, the winning entry was an equal-weighted combination of 220 LightGBM models with a Tweedie objective, achieving a weighted root mean squared scaled error (WRMSSE) of 0.520 — 22.4% better than the exponential smoothing bottom-up benchmark. All top-50 methods were machine learning and beat the statistical benchmark by more than 14%.
3. **Foundation models are candidates, not defaults.** TSFM-Bench (KDD 2025) found no single dominant time-series foundation model, with zero-shot models beating task-specific models on 10 of 21 datasets and scaling laws not holding strictly. A 2025 audit found that in the datasets it examined only about 7% had never been used in pretraining or fine-tuning, which makes contamination a first-order concern.
4. **Forecast value is created or destroyed after the model runs.** Manual overrides, rounding, and the policy that consumes the forecast determine the realised benefit more than a 1-point error improvement does.

The cost of getting this wrong is concrete. A 3-point WAPE degradation at the SKU-store grain propagates into safety stock, expedite spend, and markdown; a biased forecast at the item level becomes an inventory imbalance that takes a quarter to unwind. And a forecast that planners do not trust is overridden into uselessness, which is a failure of communication as much as of method.

## 2. Learning outcomes

By the end of this module, you can:

- Assess forecastability for a series or a portfolio and state the realistic accuracy ceiling.
- Decompose a series into level, trend, seasonality, holiday, event, and noise components and choose a method accordingly.
- Implement and defend naive, seasonal-naive, moving-average, and drift benchmarks before any other model.
- Fit and interpret simple exponential smoothing (SES), Holt, and Holt-Winters, and explain what each parameter controls.
- Specify and diagnose ARIMA, SARIMA, and SARIMAX models, including stationarity, differencing, and residual checks.
- Build a regression forecaster with lags, Fourier terms, and exogenous drivers, and avoid the leakage traps that come with lags.
- Reconcile a hierarchy by bottom-up, top-down, middle-out, or optimal combination, and explain aggregation effects.
- Recognise intermittent demand and choose a starting method appropriate to it.
- Combine forecasts and demonstrate the conditions under which combination beats selection.
- Design and run a rolling-origin backtest and report per-window and pooled metrics.
- Compute MAE, RMSE, WAPE, MASE, bias, and pinball loss, and choose the one that matches the decision.
- Audit manual overrides with forecast value added (FVA) and quantify the cost of a bad override policy.
- Compute coefficient of variation (CV), average demand interval (ADI), and CV-squared (CV²) and classify demand patterns.
- Decide when a statistical model, an ML model, or a foundation model is the right candidate, and justify it with evidence.

## 3. Mental model

### 3.1 The forecast is a decision input, not an answer

```
History ──► features ──► model ──► forecast distribution ──► plan ──► order ──► service / cost
                            ▲              ▲                  ▲
                       benchmark      prediction interval   planner override
```

Each arrow carries error. The model contributes one term; the policy, the override, and the execution contribute the rest. Optimising only the model is optimising the smallest lever.

### 3.2 Signal and noise

```
y_t = level + trend + seasonality(m) + holiday_effect + event_effect + noise
```

| Component | Typical period | Method that captures it |
|---|---|---|
| Level | — | SES, mean |
| Trend | — | Holt, drift, ARIMA(d) |
| Weekly seasonality | 7 days | Seasonal naive, Holt-Winters, Fourier |
| Annual seasonality | 52 weeks / 12 months | Holt-Winters, Fourier, SARIMA |
| Holiday | Moving dates | Explicit event dummies with lead/lag |
| Event | Promotions, strikes, weather | Explicit regressors or intervention dummies |
| Noise | — | Everything else; sets the accuracy floor |

The accuracy floor is the noise. If the residual standard deviation is 15% of the mean, no method will deliver 5% WAPE, and a project promising it is measuring something else.

### 3.3 Forecastability

A practical definition for planning purposes:

```
Forecastability F = 1 − MSE_model / MSE_naive
```

on a proper backtest. F near 0 means the model adds nothing over the benchmark; F of 0.2 means it removes 20% of the naive error. Portfolio-level F is usually much higher than SKU-level F, because aggregation averages out noise — which is why aggregate accuracy claims are so often misleading.

Three diagnostics describe how hard a series is:

| Diagnostic | Formula | Reading |
|---|---|---|
| CV | sd(y) / mean(y) | Overall relative volatility |
| ADI | periods / periods with nonzero demand | Average interval between demand occurrences |
| CV² | (sd of nonzero demand / mean of nonzero demand)² | Variability of demand size when it occurs |

| ADI | CV² | Class | Starting method |
|---|---|---|---|
| < 1.32 | < 0.49 | Smooth | SES, Holt-Winters, ARIMA |
| ≥ 1.32 | < 0.49 | Intermittent | Croston, SBA, TSB |
| < 1.32 | ≥ 0.49 | Erratic | Holt-Winters, robust methods |
| ≥ 1.32 | ≥ 0.49 | Lumpy | Aggregate-then-disaggregate; probabilistic |

The 1.32 and 0.49 cut-offs come from the Syntetos-Boylan classification and are conventions, not laws. They are useful for triage, not for excluding a series from modelling.

## 4. Core concepts

### 4.1 Benchmarks first

| Benchmark | Formula | Beats what | Fails when |
|---|---|---|---|
| Naive | ŷ_{t+h} = y_t | Slow-moving series | Trend or seasonality present |
| Seasonal naive | ŷ_{t+h} = y_{t+h−m} | Strongly seasonal series | Level shifts |
| Moving average | mean of last k observations | Noisy, stable series | Trend and seasonality |
| Drift | y_t + h × (y_t − y_1)/(t−1) | Steady linear trend | Trend changes |
| Mean | ȳ | Pure noise | Anything structured |

Publish the benchmark in every forecast report. It anchors the conversation and prevents the team from claiming credit for seasonality the naive model already captures.

### 4.2 Exponential smoothing

**SES.** `ℓ_t = α y_t + (1−α) ℓ_{t−1}`, forecast `ŷ_{t+h} = ℓ_t`. α near 1 tracks; α near 0 smooths. Optimal α minimises one-step squared error.

**Holt.** Adds a trend: `ℓ_t = α y_t + (1−α)(ℓ_{t−1} + b_{t−1})`, `b_t = β(ℓ_t − ℓ_{t−1}) + (1−β) b_{t−1}`, forecast `ℓ_t + h b_t`.

**Holt-Winters.** Adds seasonality, additive or multiplicative:

```
Additive:       ŷ_{t+h} = ℓ_t + h b_t + s_{t+h−m(k+1)}
Multiplicative: ŷ_{t+h} = (ℓ_t + h b_t) × s_{t+h−m(k+1)}
```

| Variant | Use when | Warning |
|---|---|---|
| Additive seasonality | Seasonal amplitude roughly constant | Understates peaks when amplitude grows with level |
| Multiplicative seasonality | Amplitude scales with level | Breaks near zero; produces zeros and negatives |
| Damped trend | Trend likely to flatten | Often more accurate on long horizons than linear trend |

Multiplicative seasonality on low-volume series produces fractional and zero forecasts, which then break the replenishment policy. Check the forecast for non-negativity before it reaches the planner.

### 4.3 ARIMA, SARIMA, SARIMAX

ARIMA(p,d,q): autoregressive order p, differencing d, moving average order q.

```
(1 − Σ φᵢ Bⁱ)(1 − B)^d y_t = (1 + Σ θⱼ Bʲ) ε_t
```

SARIMA adds seasonal terms (P,D,Q,m). SARIMAX adds exogenous regressors.

| Step | Action | Diagnostic |
|---|---|---|
| Stationarity | Difference until the augmented Dickey-Fuller test rejects a unit root | ADF p-value, differenced plot |
| Order selection | Akaike information criterion (AIC) or auto-ARIMA | AIC, residual ACF |
| Seasonality | Seasonal differencing at m, or seasonal AR/MA terms | Seasonal ACF spikes |
| Residual check | Residuals should be white noise | Ljung-Box test, ACF of residuals |
| Exogenous | Add drivers with future values known | Coefficient signs, residual improvement |

A residual ACF with a significant spike at lag 7 means the weekly seasonality is not captured, regardless of what the information criterion says. Always plot residuals.

### 4.4 Regression with lags, Fourier terms, and exogenous variables

A regression forecaster is often the most practical option because it accommodates promotion, price, weather, and holidays directly.

```
y_t = β₀ + Σ_k [a_k sin(2πkt/52) + b_k cos(2πkt/52)]
      + Σ_j γ_j holiday_{j,t}
      + Σ_p δ_p y_{t−p}
      + Σ_q ζ_q x_{q,t} + ε_t
```

| Element | Purpose | Leakage risk |
|---|---|---|
| Fourier terms | Smooth annual and weekly seasonality | None if generated from the calendar |
| Holiday dummies | Moving-date effects | None if the calendar is known in advance |
| Lagged y | Short-term autocorrelation | Recursive forecasting compounds error |
| Exogenous x | Promotion, price, weather | Must be known at forecast time, or itself forecast |
| Lagged exogenous | Distributed effects | Easy to misalign by one period |

**The lag trap.** A model with `y_{t−1}` as a feature must be forecast recursively for h > 1, and errors compound. A direct multi-horizon model, one model per horizon, avoids compounding but multiplies the models. For supply chain horizons of 1–13 weeks, direct models are usually the safer choice.

**The alignment trap.** Weekly data keyed on ISO week but joined to a promotion calendar keyed on fiscal week will silently misalign by one week in some years. Validate the join by checking that known promotion weeks land where expected.

### 4.5 Hierarchy and aggregation

Forecasts must exist at several levels: SKU, SKU-store, category, region, total.

| Approach | How | Strength | Weakness |
|---|---|---|---|
| Bottom-up | Forecast the lowest level, sum | Consistent; captures local detail | Noisy at low volume |
| Top-down | Forecast the total, split by proportions | Stable at the top | Ignores local dynamics |
| Middle-out | Forecast a middle level, aggregate up and disaggregate down | Balances both | Requires a good middle level |
| Optimal reconciliation (MinT) | Forecast all levels, reconcile with a covariance-weighted projection | Usually most accurate | Needs covariance estimation; more compute |

Aggregation reduces relative error because independent noise cancels. This is why a category-level WAPE of 6% can coexist with a SKU-store WAPE of 34%, and why quoting the aggregate figure to a planner who orders at SKU-store is misleading. Module 08 develops reconciliation and the weighted metrics (WRMSSE, scaled pinball) that score a hierarchy coherently.

### 4.6 Intermittent demand

Many SKUs sell on a minority of days. Standard methods fail because the series is mostly zeros and the error metrics divide by small numbers.

| Method | Idea | Strength | Weakness |
|---|---|---|---|
| Croston | Separate interval and size exponential smoothing | Handles zeros by construction | Biased upward; lags turning points |
| SBA | Croston with a 0.95 bias correction | Reduces Croston's bias | Still lags |
| TSB | Updates probability of occurrence and size separately | Responds to obsolescence | More parameters |
| Aggregate-disaggregate | Forecast the group, split by share | Stable | Depends on share stability |

MASE and WAPE are the usable metrics here; mean absolute percentage error (MAPE) is undefined on zero actuals and explodes on small ones.

### 4.7 Forecast combination

Combination beats selection more reliably than any single method beats its rivals, because errors across methods are imperfectly correlated.

| Combination | How | When |
|---|---|---|
| Simple mean | Average the forecasts | Default; hard to beat |
| Median | Robust to one wild model | Noisy portfolio |
| Weighted by inverse error | Weight ∝ 1/MSE | Stable error estimates |
| Cross-validation weights | Fit weights on rolling-origin folds | Enough folds |
| Stacking | Learn a meta-model on out-of-fold forecasts | Large data; risk of overfitting |

Bates and Granger established the result in 1969, and the M-competitions have reinforced it: a simple average of a few decent methods usually beats the best single method selected in hindsight. The practical rule is to combine three to five diverse methods — one seasonal, one trend, one regression — rather than ten near-identical variants.

### 4.8 Rolling-origin backtesting

A single holdout answers one question once. A rolling origin answers it repeatedly and exposes stability.

```
Origin 1: train [1..T]        test [T+1..T+h]
Origin 2: train [1..T+s]      test [T+s+1..T+s+h]
Origin 3: train [1..T+2s]     test [T+2s+1..T+2s+h]
...
```

| Design choice | Rule | Consequence if wrong |
|---|---|---|
| Horizon h | Match the decision horizon | A 1-step backtest says nothing about a 13-week plan |
| Step s | Usually equal to h, so windows do not overlap | Overlapping windows inflate apparent stability |
| Number of windows | At least 4; 8–12 preferred | One window is a holdout, not a backtest |
| Training window | Expanding or sliding | Expanding suits stable series; sliding suits regime change |
| Preprocessing | Fit inside each origin | Leakage of future scaling parameters |
| Aggregation | Report per-window and pooled | Pooled alone hides a bad window |

### 4.9 Metrics

| Metric | Formula | Decision it matches | Failure |
|---|---|---|---|
| MAE | mean(\|y − ŷ\|) | Symmetric unit cost | Scale-dependent across series |
| RMSE | √mean((y − ŷ)²) | Squared-error cost | Outlier-dominated |
| WAPE | Σ\|y − ŷ\| / Σ\|y\| | Volume-weighted operational accuracy | Low-volume series dominate the ratio |
| MASE | MAE / MAE_naive(in-sample) | Comparing across series and scales | Benchmark choice changes the value |
| Bias | mean(ŷ − y) | Detecting systematic over/under-forecast | Cancels within the window |
| Pinball loss | Quantile-specific asymmetric loss | Probabilistic and safety-stock decisions | Needs quantile forecasts |

**Worked example.** Five weeks of actuals and a point forecast:

| Week | Actual | Forecast | Error (f − a) | Absolute | Squared |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 105 | +5 | 5 | 25 |
| 2 | 120 | 115 | −5 | 5 | 25 |
| 3 | 90 | 100 | +10 | 10 | 100 |
| 4 | 140 | 130 | −10 | 10 | 100 |
| 5 | 110 | 120 | +10 | 10 | 100 |
| Total | 560 | 570 | +10 | 40 | 350 |

```
MAE   = 40 / 5 = 8.00 units
RMSE  = √(350 / 5) = √70 = 8.37 units
Bias  = 10 / 5 = +2.00 units (over-forecast)
WAPE  = 40 / 560 = 0.0714 = 7.14%
```

**MASE.** The in-sample naive forecast is the previous actual, with errors at weeks 2–5: +20, −30, +50, −30.

```
MAE_naive = (20 + 30 + 50 + 30) / 4 = 130 / 4 = 32.50
MASE = 8.00 / 32.50 = 0.246
```

A MASE below 1 means the forecast beats the naive benchmark; 0.246 means it removes about 75% of the naive error. For seasonal data, use the seasonal naive in the denominator and state the season length.

**Pinball loss** at the 0.9 quantile, treating the forecast as a 0.9 quantile forecast:

```
L_q(y, ŷ) = q(y − ŷ)      if y ≥ ŷ
          = (1 − q)(ŷ − y) if y < ŷ
```

| Week | y | ŷ | Case | Loss |
|---|---:|---:|---|---:|
| 1 | 100 | 105 | y < ŷ | 0.1 × 5 = 0.5 |
| 2 | 120 | 115 | y ≥ ŷ | 0.9 × 5 = 4.5 |
| 3 | 90 | 100 | y < ŷ | 0.1 × 10 = 1.0 |
| 4 | 140 | 130 | y ≥ ŷ | 0.9 × 10 = 9.0 |
| 5 | 110 | 120 | y < ŷ | 0.1 × 10 = 1.0 |
| Mean | | | | 3.20 |

Pinball is the correct scoring rule for a quantile forecast because it is minimised in expectation by the true quantile. It is also the metric that connects the forecast to safety stock, which is set from an upper quantile of lead-time demand.

**Which metric to report.** Report WAPE at the decision grain, MASE for cross-series comparison, bias separately (never netted into accuracy), and pinball if a distribution is produced. Report MAPE only with a stated exclusion rule for near-zero actuals, or not at all.

### 4.10 Forecast value added and the cost of overrides

FVA compares each step in the forecasting process with the step before it, using the same data and horizon.

```
FVA(step) = (WAPE_previous − WAPE_step) / WAPE_previous
```

**Worked illustration.** 400 SKUs, weekly, 13-week horizon, one backtest window.

| Process step | WAPE | FVA vs previous |
|---|---:|---:|
| Naive benchmark | 12.0% | — |
| Statistical model | 9.5% | +20.8% |
| After planner override | 10.2% | −7.4% |

The override made the forecast worse overall. The naive conclusion — "ban overrides" — is wrong, because overrides are not applied at random. Planners override the SKUs they believe are wrong, which are disproportionately the hard, high-variance ones. The correct comparison is **within SKU**: for the 100 overridden SKUs, model WAPE 14.0% versus post-override 16.8%, a degradation of 2.8 points; for the 300 untouched SKUs, 8.0%.

| Group | n | Model WAPE | Post-override WAPE | Change |
|---|---:|---:|---:|---:|
| Overridden | 100 | 14.0% | 16.8% | +2.8 pts worse |
| Not overridden | 300 | 8.0% | 8.0% | 0.0 |
| All | 400 | 9.5% | 10.2% | +0.7 pts worse |

Two conclusions follow. The override policy destroys value on the cases it touches, and the planners' attention is concentrated exactly where their judgment is least reliable — a well-documented pattern, because judgment adds most where there is contextual information the model lacks (a known store closure, a lost contract) and least where the series is simply noisy.

The organisational fixes are structural, not statistical: require a reason code, cap the override magnitude, review override performance quarterly by planner and reason, and route the genuinely new information into the model as a regressor instead of a manual adjustment.

### 4.11 Statistical, ML, and foundation models

| Family | Strength | Weakness | Evidence |
|---|---|---|---|
| Statistical (ETS, ARIMA, Theta) | Strong on short series, interpretable, fast, no training data needed | Limited cross-learning; manual feature work | M4: hybrid and combination methods led; pure statistical methods remained competitive |
| ML (LightGBM, XGBoost, NN) | Cross-learning across thousands of series; handles exogenous and hierarchy | Needs many series, careful validation, feature engineering | M5: all top-50 were ML, >14% better than the statistical benchmark |
| Foundation models (Chronos, TimesFM, Moirai, TTM) | Zero-shot, no training, handles cold start | Contamination risk, mixed benchmark evidence, cost | TSFM-Bench 2025: no dominant model; zero-shot won on 10 of 21 datasets |

**The M5 result, stated precisely.** The M5 Accuracy competition used Walmart daily data: 42,840 hierarchical series, 1,913 training days, a 28-day horizon, scored by WRMSSE, with 5,507 teams. The winner combined 220 LightGBM models with a Tweedie objective, built per store, per store-category, and per store-department, with equal weights, reaching WRMSSE 0.520 — 22.4% better than the ES-bu benchmark. The drivers were cross-learning, exogenous variables, combination, and time-based validation, not a novel architecture.

**What M5 does not tell you.** It is one retailer, one horizon, one metric, one product mix. WRMSSE rewards performance on high-revenue series and on aggregate levels. A 22.4% improvement over a statistical benchmark on that task does not transfer automatically to a 4-week-ahead forecast for 900 pharmaceutical SKUs with short histories.

**Foundation models: candidates, not defaults.** TSFM-Bench (KDD 2025, arXiv:2410.11802) evaluated Chronos, TimesFM, Moirai, TTM, ROSE, Timer, and MOMENT across 21 datasets. It found no single dominant model, that zero-shot models beat task-specific models on 10 of 21 datasets, that scaling laws did not hold strictly, and that large language model (LLM)-based forecasters generally underperformed. Two 2025 domain studies point the same way: bi-seasonal multiple seasonal-trend decomposition using LOESS (MSTL) was never statistically beaten by foundation models in a five-market electricity-price study (arXiv:2506.08113), and zero-shot foundation models were beaten by linear and seasonal-naive baselines on real cloud workloads (arXiv:2502.12944).

**Contamination.** A 2025 preprint audit (arXiv:2510.13654) found that only about 7% of the benchmark datasets it examined had never been used for pretraining or fine-tuning. A foundation model's zero-shot score on M4 or M5 may reflect memorisation of the test period rather than forecasting skill. Any foundation-model evaluation must therefore include a genuinely post-cutoff window and a contamination statement.

### 4.12 A repeatable forecasting pipeline

```
1. Define grain, horizon, and refresh cadence.
2. Build a date spine per series; insert explicit zeros; flag stockouts.
3. Compute diagnostics: CV, ADI, CV²; segment the portfolio.
4. Fit benchmarks; record WAPE, MASE, bias.
5. Fit candidates per segment; combine.
6. Backtest with rolling origin at the decision horizon.
7. Select on the decision metric, not on average error across all series.
8. Produce a distribution, not just a point.
9. Reconcile the hierarchy (Module 08).
10. Publish, monitor, and re-run on a fixed cadence with a champion-challenger.

### 4.13 Reference implementation

Two functions do most of the work: a benchmark runner over `statsforecast` and a backtest harness you can point at any model.

```python
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive, SeasonalNaive, HistoricAverage,
    SimpleExponentialSmoothing, HoltWinters, AutoETS, AutoARIMA, AutoTheta,
)

def benchmark_models(season_length: int = 7) -> list:
    """Eight candidates spanning the accuracy ladder. Pin the library version:
    class names and constructor arguments change between releases."""
    return [
        Naive(),
        SeasonalNaive(season_length=season_length),
        HistoricAverage(),
        SimpleExponentialSmoothing(alpha=0.2),
        HoltWinters(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoARIMA(season_length=season_length),
        AutoTheta(season_length=season_length),
    ]

def run_benchmark(panel: pd.DataFrame, h: int = 28, n_windows: int = 4,
                  season_length: int = 7, freq: str = "D") -> pd.DataFrame:
    """panel: columns [unique_id, ds, y], gapless per series, sorted.
    Returns one row per model with WAPE, MAE, and bias on identical folds."""
    sf = StatsForecast(models=benchmark_models(season_length), freq=freq, n_jobs=-1)
    cv = sf.cross_validation(df=panel, h=h, n_windows=n_windows,
                             step_size=h, refit=True)
    model_cols = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    rows = []
    for col in model_cols:
        err = cv[col] - cv["y"]
        rows.append({
            "model": col,
            "wape": float(err.abs().sum() / cv["y"].abs().sum()),
            "mae": float(err.abs().mean()),
            "bias": float(err.mean()),
        })
    return pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)

def rolling_origin_backtest(panel: pd.DataFrame, model_fn, h: int = 28,
                            n_windows: int = 6, step: int = 28) -> pd.DataFrame:
    """model_fn(train_df, h) -> DataFrame [unique_id, ds, yhat].
    Expands the training window from a fixed start; non-overlapping test blocks."""
    last = panel["ds"].max()
    origins = sorted(last - pd.Timedelta(days=h + step * k) for k in range(n_windows))
    out = []
    for k, origin in enumerate(origins):
        train = panel[panel["ds"] <= origin]
        test = panel[(panel["ds"] > origin) &
                     (panel["ds"] <= origin + pd.Timedelta(days=h))]
        yhat = model_fn(train, h)
        m = test.merge(yhat, on=["unique_id", "ds"], how="inner")
        assert len(m) > 0, f"window {k}: no overlapping forecast rows"
        assert m["ds"].min() > origin, f"window {k}: leakage across the origin"
        err = m["yhat"] - m["y"]
        out.append({"window": k, "origin": origin, "n": len(m),
                    "wape": float(err.abs().sum() / m["y"].abs().sum()),
                    "mae": float(err.abs().mean()),
                    "bias": float(err.mean())})
    return pd.DataFrame(out)

def mase(y_true, y_pred, y_train, seasonality: int = 1) -> float:
    scale = np.abs(np.diff(y_train, n=seasonality)).mean()
    return float(np.abs(y_true - y_pred).mean() / scale)
```

Expected output on a 400-series daily panel with `h = 28` and 4 windows: a table with AutoETS and SeasonalNaive near the top, Naive at the bottom, `wape` around 0.11–0.18 depending on the panel, and `bias` that should be small but is rarely exactly zero. The absolute values are not the point; the ranking and the spread are. Re-run `run_benchmark` and `rolling_origin_backtest` on the same panel and confirm the per-window WAPE from the harness matches the library's cross-validation to within numerical tolerance.
```

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Sales history | SKU × location × day | POS / OMS | Stockout flags; returns netted consistently |
| Shipments | SKU × location × week | ERP / TMS | Distinguish sell-in from sell-out |
| Calendar | Day | Planning system | Fiscal calendar, holidays, 4-5-4 weeks |
| Promotion plan | SKU × store × week | TPM | Future promotions known at forecast time |
| Price | SKU × store × week | Pricing | Regular versus paid price |
| Master data | Item, location | PIM / ERP | Hierarchy versioned over time |
| External | Region × day | Weather, macro | Forecast vintage, not realised values |

**Time standards.** ISO 8601 for dates; ISO 8601 week numbering or the retail 4-5-4 calendar, chosen once and documented; explicit handling of the 53-week year. A forecast pipeline that mixes fiscal and ISO weeks will misalign holiday effects in roughly one year in five.

**Metric standards.** ASCM SCOR Digital Standard (SCOR DS) for the service metrics the forecast ultimately drives. M5's WRMSSE is a competition metric, not a production metric; production scoring should use WAPE at the decision grain plus a bias monitor.

**Workflow checks.** Verify the date spine has no gaps; verify zeros are true zeros and not missing rows; verify stockout days are flagged before any model sees the series; verify promotion weeks align across systems; verify the forecast horizon is at least the replenishment lead time plus the review period.

## 6. Industry case studies

### 6.1 The M5 Accuracy competition (2020–2021)

Walmart daily sales data, 42,840 hierarchical series across 3,049 products and 10 stores in three states, 1,913 training days, a 28-day horizon, scored by WRMSSE, with 5,507 teams. The winning entry was an equal-weighted combination of 220 LightGBM models with a Tweedie objective, organised per store, per store-category, and per store-department, reaching WRMSSE 0.520 and 22.4% better than the ES-bu benchmark. All top-50 methods were ML and beat the statistical benchmark by more than 14%.

What transferred: cross-learning across series, exogenous variables (price, promotion, calendar), combination, and time-based validation. What did not: the absolute score, which is specific to Walmart's series mix and to WRMSSE's weighting toward high-revenue items and aggregate levels, and the assumption that 220 models are operationally manageable — the winning ensemble is a research artifact, not a production design.

### 6.2 The M4 competition (2018–2020)

M4 used 100,000 series across multiple frequencies, including micro, industry, and macro data, with a mixed-frequency design and multiple horizons. The winning entry was a hybrid combining exponential smoothing with a recurrent neural network, and the broader finding was that combination and hybrid approaches outperformed single methods.

What transferred: hybrids and combinations are the reliable path to accuracy, and simple statistical methods remain competitive on short and sparse series. What did not: the result does not support a claim that neural networks are necessary — the hybrid's statistical component carried much of the performance, and the competition's series mix differs sharply from a single firm's portfolio.

### 6.3 Rossmann store sales (2015)

A Kaggle competition on 1,115 German drugstore stores with 942 days of daily sales, a 6-week horizon, and strong calendar and promotion effects, including a store-closure pattern that creates legitimate zero-sales days.

What transferred: the leading solutions were gradient-boosted trees with calendar, promotion, and store metadata features rather than classical univariate time series. The competition also made explicit that zero sales on closure days are not zero demand and must be handled as a separate state. What did not: the dataset has no stockout information, so the sales-equals-demand assumption is untested — a limitation that would be fatal in a replenishment setting.

### 6.4 Corporación Favorita grocery sales (2018)

A Kaggle competition on grocery sales across stores and item families in Ecuador, with promotion flags, a holiday calendar, and a large, sparse item mix.

What transferred: the strong solutions used LightGBM and neural networks with promotion and transfer features, and the competition demonstrated the value of a clean calendar model and of handling item cold start explicitly. What did not: the metric rewarded aggregate accuracy, and the winning approaches were ensembles tuned to that metric; the transfer to a per-SKU replenishment decision, where tail error matters, is not established by the competition result.

### 6.5 TSFM-Bench and the contamination audit (2025)

TSFM-Bench (KDD 2025, arXiv:2410.11802) evaluated seven time-series foundation models across 21 datasets and found no dominant model, zero-shot superiority on 10 of 21 datasets, non-strict scaling, and weak LLM-based forecasters. A separate 2025 preprint (arXiv:2510.13654) audited benchmark contamination and found that only about 7% of the datasets it examined had never been used in pretraining or fine-tuning.

What transferred: a disciplined default — evaluate a foundation model against seasonal naive and a statistical benchmark on your own post-cutoff data, and report the contamination status. What did not: any general claim that foundation models are or are not better. The evidence supports "candidate, evaluate locally", and nothing stronger.

## 7. Labs

### Lab 7.1 — Benchmark twelve methods on a retail panel

**Objective.** Establish the accuracy ladder from naive to ML on a real panel.

**Data.** M5 or Favorita, aggregated to store-department or store-category weekly to keep runtime manageable; at least 200 series and 3 years of history.

**Steps.** (1) Build a complete date spine and flag missing days. (2) Compute CV, ADI, and CV² and segment the portfolio. (3) Fit naive, seasonal naive, moving average, drift, SES, Holt, Holt-Winters (additive and multiplicative), AutoETS, AutoARIMA, Theta, and a LightGBM with calendar and price features. (4) Backtest with a rolling origin at a 4-week horizon, 6 windows. (5) Report WAPE, MASE, and bias per method and per segment. (6) Test a simple mean combination of the best three.

**Deliverable.** A benchmark report with a method × segment table, a per-window stability chart, and a recommendation.

**Acceptance criteria.** All twelve methods run on identical folds; WAPE and MASE reported with the seasonal-naive denominator stated; bias reported separately; combination tested; a clear statement of which method wins at which segment.

**Stretch.** Add a hierarchical reconciliation and report whether it improves the aggregate level without degrading the SKU level.

### Lab 7.2 — Write and validate a rolling-origin backtest

**Objective.** Implement the backtest harness that every later module will reuse.

**Data.** A synthetic panel with a known injected effect, plus the Lab 7.1 panel.

**Steps.** (1) Implement `rolling_origin_backtest` with configurable horizon, step, number of windows, and expanding or sliding training. (2) Assert that no test row precedes its origin. (3) Assert that preprocessing is fitted inside each origin. (4) Verify against `statsforecast.cross_validation` on the same folds. (5) Inject a level shift and confirm the backtest detects the degradation.

**Deliverable.** A tested module with docstrings, plus a short verification note showing agreement with the library implementation.

**Acceptance criteria.** No leakage assertion passes; results match the library to within numerical tolerance; the injected shift is detected; the harness accepts a pluggable model function.

**Stretch.** Add parallel execution across windows and confirm identical results with a fixed seed.

### Lab 7.3 — Diagnostics and intermittent demand

**Objective.** Classify a portfolio and show where standard methods fail.

**Data.** 2,000 SKUs × 104 weeks from UCI Online Retail II or a synthetic intermittent generator.

**Steps.** (1) Compute CV, ADI, and CV² per SKU-week. (2) Assign the four Syntetos-Boylan classes. (3) Backtest SES, Croston, SBA, and TSB with MASE. (4) Report accuracy by class and identify the class where each method fails. (5) Compute the share of the portfolio that is intermittent and lumpy, by SKU count and by revenue.

**Deliverable.** Classification table, accuracy-by-class table, and a one-page note on portfolio composition.

**Acceptance criteria.** Diagnostics computed correctly on nonzero demand; MASE denominator stated; results reported by class; the revenue-weighted share computed, not only the SKU count.

**Stretch.** Add an aggregate-disaggregate method and compare on the lumpy segment.

### Lab 7.4 — Override audit and forecast value added

**Objective.** Quantify whether planner overrides add or destroy value, and design a better policy.

**Data.** Course override log: 12 months of statistical forecasts, planner-adjusted forecasts, and actuals for 600 SKUs, with reason codes.

**Steps.** (1) Compute WAPE for the statistical forecast, the adjusted forecast, and the naive benchmark. (2) Compute FVA for each step. (3) Compare within-SKU for overridden versus non-overridden series. (4) Break the result down by reason code and by planner. (5) Estimate the planner hours spent and the margin effect of the degradation. (6) Propose a revised override policy.

**Deliverable.** An override audit report with tables by reason code and a policy proposal.

**Acceptance criteria.** Within-SKU comparison, not cross-SKU; FVA arithmetic correct; reason-code breakdown; the policy proposal names a cap, a required reason code, and a review cadence.

**Stretch.** Build a model that incorporates the override reason as a regressor and test whether it recovers the value the manual adjustment intended.

## 8. Assignment — Forecast benchmark report

Produce a **benchmark report** for one real or realistic portfolio that answers: which method should run in production, at what grain and horizon, and what accuracy is honestly achievable.

| Section | Content |
|---|---|
| 1. Scope | Decision, grain, horizon, cadence, and the policy the forecast feeds |
| 2. Data | Source, period, stockout treatment, calendar, known defects |
| 3. Diagnostics | CV, ADI, CV² distribution and segment sizes |
| 4. Benchmarks | Naive, seasonal naive, drift, with WAPE and MASE |
| 5. Candidates | Statistical, regression, ML, and combination, with settings |
| 6. Backtest design | Horizon, step, windows, expanding versus sliding, seeds |
| 7. Results | Method × segment table, per-window stability, bias |
| 8. Recommendation | Chosen method, expected accuracy range, and why not the runner-up |
| 9. Production plan | Cadence, runtime, monitoring, champion-challenger, retraining trigger |
| 10. Reproduction | Lockfile, data snapshot hash, one-command rerun |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem and data framing | 15% | Forecast without a decision or horizon | Decision and horizon stated | Grain, horizon, cadence, policy link, and stockout treatment all explicit |
| Benchmark discipline | 20% | No baseline | Naive and seasonal naive included | Benchmarks per segment; improvement over benchmark quantified with intervals |
| Method breadth and justification | 20% | One method | Three or more families | Families chosen from diagnostics; combination tested; failure cases named |
| Backtest design | 20% | Single holdout | Rolling origin implemented | Origin count justified; leakage assertions; per-window stability reported |
| Metrics and interpretation | 15% | MAPE only | WAPE and MASE | WAPE, MASE, bias, and pinball where relevant, with the decision link |
| Production plan and communication | 10% | Notebook | Runtime and cadence noted | Monitoring, champion-challenger, retraining trigger, and a planner-facing summary |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Forecast versus target | A forecast presented as a target distorts planner behaviour | Label every artifact; separate the statistical forecast from the commercial target |
| Supplier commitments | Forecasts shared with suppliers can become contractual | Publish a forecast with a stated accuracy range and an explicit non-binding clause |
| Consumer demand signalling | Sharing forecasts can facilitate coordination in concentrated markets | Legal review before sharing forward-looking volume with competitors |
| EU AI Act (Regulation 2024/1689) | Automated planning affecting access to goods or services may be high-risk | Document method, validation, and human oversight |
| Data minimisation | Store-level forecasts can reveal sensitive commercial information about partners | Aggregate where the decision does not need granularity |
| Fairness of allocation | Forecast error concentrates in low-volume or rural locations | Report error by store format and region; set a service floor |
| Commercial | Accuracy claims are used in business cases | Report improvement over benchmark with a confidence interval, not a single number |

Two professional obligations follow. **Never present a forecast as a commitment**, and **never report aggregate accuracy to justify a SKU-level decision**. Both are common, both are misleading, and both are avoidable with one sentence of labelling.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No benchmark | Report shows only the new method | Improvement unmeasured | Value claim unsupported | Always publish naive and seasonal naive |
| Single holdout | One split, one number | Period-specific luck | Overstated or understated accuracy | Rolling origin with 6+ windows |
| Horizon mismatch | Backtest at 1 week, plan at 13 | Error grows with horizon | Plan error far above reported | Backtest at the decision horizon |
| Sales treated as demand | Zero-sales days included as zeros | Stockout censoring | Systematic under-forecast of popular items | Stockout flags; censored model |
| MAPE on intermittent series | Huge or undefined errors | Division by near-zero | Metric unusable; series excluded arbitrarily | WAPE or MASE; explicit exclusion rule |
| Multiplicative seasonality near zero | Negative or zero forecasts | Model structure | Broken replenishment | Additive seasonality; floor at zero |
| Lag misalignment | Promotion effect appears one week early | Calendar join error | Wrong lift, wrong stock | Validate joins against known events |
| Override without audit | Adjustment rate above 30% | Judgment applied where it adds least | Value destroyed silently | FVA by reason code; cap; review |
| Foundation model as default | No local benchmark | Contamination and domain mismatch | Cost with no accuracy gain | Local post-cutoff evaluation first |
| Aggregate accuracy quoted | 6% at category, 34% at SKU-store | Noise cancellation on aggregation | Planner misled | Report at the decision grain |
| Rigid retraining | Model unchanged for 18 months | Regime change | Silent decay | Scheduled retraining; drift monitor |
| Over-engineering | 220-model ensemble for 400 SKUs | Complexity without need | Maintenance burden | Simplest method within noise of the best |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What is the first model you build for a new forecasting problem? | Seasonal naive, plus naive and drift; establishes the floor and the seasonal structure. |
| 2 | Why does MASE exist? | Scale-free comparison across series using a naive benchmark in the denominator; state the season length. |
| 3 | When is WAPE better than MAPE? | Low or zero actuals; MAPE explodes and is undefined; WAPE weights by volume. |
| 4 | Your backtest uses one 12-week holdout. What is wrong? | No stability evidence; a single period can be atypical; use rolling origin with multiple windows. |
| 5 | How do you decide between statistical and ML forecasting? | Series count, history length, exogenous drivers, need for cross-learning, validation capacity. |
| 6 | What did M5 show, and what did it not show? | Cross-learning and combination win on that retail task; it does not transfer automatically to other portfolios or horizons. |
| 7 | Why treat foundation-model benchmarks with caution? | Contamination, domain mismatch, non-strict scaling, mixed evidence in TSFM-Bench; evaluate on post-cutoff local data. |
| 8 | How would you detect that a forecast has gone stale? | Rolling WAPE by horizon, bias monitor, input drift, champion-challenger gap. |
| 9 | A planner overrides 40% of forecasts. What do you do? | Audit FVA within SKU by reason code; quantify hours and margin; cap and require reasons; move real information into regressors. |
| 10 | Why is a category-level WAPE of 6% compatible with a SKU-store WAPE of 34%? | Aggregation cancels independent noise; report at the decision grain. |
| 11 | How do you set the number of backtest windows? | At least 4; 8–12 for stability; enough to cover a full seasonal cycle where possible. |
| 12 | What is the difference between a forecast and a plan? | Forecast is the expected outcome; plan is the feasible response; target is desired; commitment is promised. |

## 13. Key terms

| Term | Definition |
|---|---|
| Forecast | Best estimate of what will happen, at a stated grain and horizon |
| Plan | Feasible response given constraints |
| Target | Commercially desired outcome |
| Commitment | What has been promised to a customer or supplier |
| Benchmark | Naive or seasonal-naive forecast used as the accuracy floor |
| Seasonal naive | Forecast equal to the value one season earlier |
| Drift | Forecast extrapolating the average historical change |
| SES | Simple exponential smoothing; level only |
| Holt | Exponential smoothing with level and trend |
| Holt-Winters | Exponential smoothing with level, trend, and seasonality |
| ARIMA | Autoregressive integrated moving average |
| SARIMA | Seasonal ARIMA |
| SARIMAX | SARIMA with exogenous regressors |
| Stationarity | Constant mean, variance, and autocovariance over time |
| Fourier terms | Sine and cosine pairs representing smooth seasonality |
| Hierarchy | Nested levels of product, location, and time |
| Reconciliation | Making forecasts coherent across hierarchy levels |
| Intermittent demand | Demand occurring in a minority of periods |
| ADI | Average demand interval; periods per demand occurrence |
| CV² | Squared coefficient of variation of nonzero demand sizes |
| Croston | Method separating demand interval and size |
| TSB | Teunter-Syntetos-Babai method for obsolescence |
| Combination | Averaging or weighting multiple forecasts |
| Rolling origin | Repeated backtests advancing the forecast origin |
| Backtest | Out-of-sample evaluation over historical origins |
| WAPE | Sum of absolute errors over sum of actuals |
| MASE | Mean absolute scaled error, scaled by a naive benchmark |
| Pinball loss | Asymmetric loss for a quantile forecast |
| FVA | Forecast value added; improvement over the prior process step |
| WRMSSE | Weighted root mean squared scaled error, the M5 metric |
| Foundation model | Pretrained time-series model used zero-shot or fine-tuned |

## 14. Sources and further reading

**Standards and regulation**

- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). Accessed 27 October 2026.
- ISO 8601, *Date and time — Representations for information interchange*, including week numbering.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act), consolidated text, EUR-Lex. Accessed 27 October 2026.

**Peer-reviewed and methodological**

- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. Open access.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. "The M4 Competition: 100,000 Time Series and 61 Forecasting Methods." *International Journal of Forecasting*, 2020.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. "M5 Accuracy Competition: Results, Findings, and Conclusions." *International Journal of Forecasting*, 2022.
- In, Y., and Jung, J. "Simple Averaging of LightGBM Models for the M5 Accuracy Competition." *International Journal of Forecasting*, 2022.
- Bates, J. M., and Granger, C. W. J. "The Combination of Forecasts." *Operational Research Quarterly*, 1969.
- Syntetos, A. A., and Boylan, J. E. "The Accuracy of Intermittent Demand Estimates." *International Journal of Forecasting*, 2005.
- Croston, J. D. "Forecasting and Stock Control for Intermittent Demands." *Operational Research Quarterly*, 1972.
- Teunter, R. H., Syntetos, A. A., and Babai, M. Z. "Intermittent Demand: Linking Forecasting to Inventory Obsolescence." *European Journal of Operational Research*, 2011.
- Hyndman, R. J., and Khandakar, Y. "Automatic Time Series Forecasting: The forecast Package for R." *Journal of Statistical Software*, 2008.
- Kolassa, S., and Schütz, W. "Advantages of the MAD/Mean Ratio over the MAPE." *Foresight*, 2007.
- Tashman, L. J. "Out-of-Sample Tests of Forecasting Accuracy: An Analysis and Review." *International Journal of Forecasting*, 2000.
- Petropoulos, F., et al. "Forecasting: Theory and Practice." *International Journal of Forecasting*, 2022.
- Gilliland, M. *The Business Forecasting Deal*. Wiley, 2010. Source for forecast value added.

**Benchmarks, foundation models, and audits**

- TSFM-Bench: "TSFM-Bench: A Comprehensive and Unified Benchmark of Foundation Models for Time Series Forecasting." KDD 2025, arXiv:2410.11802.
- Contamination audit of time-series benchmarks, arXiv:2510.13654, 2025 preprint. [VERIFY: preprint status and final venue at time of use.]
- "Classical bi-seasonal MSTL was never statistically beaten by foundation models in a five-market electricity-price study," arXiv:2506.08113, 2025. [VERIFY: author list and venue before citing.]
- "Zero-shot foundation models beaten by linear and seasonal-naive baselines on real cloud workloads," arXiv:2502.12944, 2025. [VERIFY: author list and venue before citing.]
- M5 competition repository and data, https://github.com/Mcompetitions/M5-methods — accessed 27 October 2026.
- Rossmann Store Sales and Corporación Favorita Grocery Sales, Kaggle. Accessed 27 October 2026.

**Datasets and tools**

- `statsforecast` documentation: `StatsForecast`, `Naive`, `SeasonalNaive`, `HistoricAverage`, `AutoETS`, `AutoARIMA`, `AutoTheta`, `cross_validation`. https://nixtlaverse.nixtla.io/statsforecast — accessed 27 October 2026.
- `statsmodels` `tsa` module: `ExponentialSmoothing`, `SARIMAX`, `STL`, `adfuller`. https://www.statsmodels.org — accessed 27 October 2026.
- `sktime`, `darts`, and `GluonTS` documentation for alternative harnesses. Accessed 27 October 2026.
- UCI Online Retail II dataset for intermittent-demand practice. Accessed 27 October 2026.

> **Note:** WRMSSE is a competition metric with revenue and aggregate weighting. Do not use it as a production KPI without re-weighting to the decision; report WAPE at the decision grain for operational monitoring.

---

> **Next:** `08_hierarchical_intermittent_probabilistic_forecasting.md` — reconciliation, intermittent-demand methods, predictive distributions, and the scoring rules that make a hierarchy coherent.
