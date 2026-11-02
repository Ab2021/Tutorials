# ARIMA, SARIMA, and SARIMAX — Differencing, Orders, Seasonality, and Exogenous Regressors

> Module 07 — Time Series Forecasting Foundations
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 01A topics 09–11; Module 04 topic 04; Module 05 topic 06; Module 07 topics 01–02
> Version 2026.5

---

## 1. Why this topic exists

Autoregressive integrated moving average (ARIMA) models are the most taught family in time series
forecasting and, in operational planning, among the least examined. Almost every planning suite ships an
automatic ARIMA engine. Almost none of them show the planner which orders were selected, why the seasonal
period was set to twelve, or what the model assumes about the next twelve months. The planner sees a
number. The four decisions that produced it are invisible.

That invisibility is expensive, and it fails in ways that are specific and predictable.

**The over-differenced series.** An analyst sees a curve and differences twice, because the second
difference looks flatter than the first. The differenced series now carries a moving average root close to
the unit circle, the forecast becomes a straight line with an enormous interval, and the seasonal shape —
which was never removed — has been smeared across the differenced series. The model fits. The forecast is
indefensible. Section 5.1 gives the arithmetic.

**The correlogram read on the wrong series.** A partial autocorrelation function (PACF) that cuts off
after lag two on a series with a unit root does not mean "an autoregressive model of order two". It means
the series has not been made stationary and the cut-off is an artefact of the trend.

**The information criterion compared across differences.** "We tried `d = 0, 1, 2` and took the lowest
AIC." The Akaike information criterion (AIC) is computed on the data actually modelled, and each value of
`d` changes that data by one observation per difference. The three numbers are not comparable. Step 12
states the rule.

**The exogenous driver that arrives after the decision.** A SARIMAX model (seasonal ARIMA with exogenous
regressors) is fitted with promotional spend, competitor price, and a fill-rate series as drivers.
In-sample fit is excellent. At forecast time, next quarter's promotional spend is a plan, next quarter's
competitor price is unknown, and next quarter's fill rate is an outcome of the very forecast the model is
producing. The model is not wrong; the input assumption was never stated.

**The model that fits and cannot forecast.** An ARIMA model with clean residuals and an AIC lower than
every rival produces a twelve-month forecast with a mean error of a third of the level. Nothing in the fit
statistics warned anyone, because fit statistics describe one-step-ahead performance inside the estimation
window.

The common thread is that the *structure* — what was differenced, which orders were chosen, what the
seasonal term assumes, which drivers are known when — was never examined. This topic is about examining
it.

**Why it sits here.** Module 07 topic 01 established that a forecast means nothing without a benchmark.
Topic 02 built exponential smoothing, first as recursion and then as a state-space model with a
likelihood. This topic builds the other classical route, and the contrast is the point: both families are
linear models of the same series, both can be estimated by maximum likelihood, and on most supply chain
data they are close enough that the choice is decided by robustness and cost rather than by accuracy.
Topic 04 will do the same job again with regression and Fourier terms; topic 08 decides which of the three
wins on a rolling origin.

**The claim of this topic.** ARIMA is a filter plus a forecast. Differencing removes the parts of the
series you cannot forecast — a level, a straight-line trend, a repeating seasonal shape. The
autoregressive and moving average terms forecast what is left. Re-integrating turns that forecast of
differences back into a forecast of the level. Every failure in this topic is a failure to keep track of
which of those three steps went wrong, and the diagnostics are designed to tell you.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain**, without notation, what one order of differencing removes from a series, what it
   leaves behind, and what it costs in observations, variance, and interpretability.
2. **Diagnose** over-differencing and under-differencing from a correlogram, a variance table, and
   a fitted moving average coefficient, and **state** the repair for each.
3. **Read** an autocorrelation function (ACF) and a partial autocorrelation function (PACF) and
   **name** the candidate orders they imply, including at seasonal lags, and **state** the
   conditions under which the reading is unreliable.
4. **Decide** an order with information criteria, and **explain** why a criterion is only
   comparable across models fitted to the same differenced data.
5. **Write** the multiplicative seasonal structure of a SARIMA model as a product of two filters
   and **explain** what the airline model assumes about a supply chain series.
6. **Specify** a SARIMAX model with an exogenous driver and **identify** whether the driver is
   known at the forecast origin, forecastable, or an outcome of the forecast itself.
7. **Diagnose** a fitted ARIMA model from residuals, parameter boundaries, and forecast behaviour,
   and **distinguish** a model that fits from a model that forecasts.
8. **Compare** an ARIMA forecast against the benchmark from topic 01 and a damped smoothing model,
   and **defend** or reject the added complexity with evidence.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a series is passed through two filters that remove what cannot be forecast, a small
linear model forecasts what survives, and the forecast is then passed back through the filters in reverse.
Nothing else is happening.

```text
   WHAT DIFFERENCING REMOVES, AND WHAT IT COSTS

   y_t   drifting level                    d1_t   the change, not the level
   700 |                          /         12 |
   600 |                     .  /            2 |  .  . .  . .  .  . . . .
   500 |              . .  /                -8 |    .      .       .
   400 |  . . . . . /                          +--------------------------> t
       +--------------------> t                  the slope became a constant

   one difference  removes a constant level and a straight-line trend
   two differences removes a straight line and leaves nothing of it
   one seasonal difference removes a fixed repeating shape at period m
   every difference costs one observation and adds noise to what remains
```

The first diagram is the whole of `d` and `D`. Differencing does not make a series "better"; it removes a
specific, nameable feature and charges a specific, computable price. If you cannot say which feature you
are removing, you are not specifying a model, you are pressing a button.

```text
   TWO WAYS A SERIES REMEMBERS ITS PAST

   AR: momentum                        MA: error correction
   today = phi * yesterday             today = -theta * yesterday's shock
         + new shock                         + new shock

   ACF        |\                       ACF   |\
              | \                             | \
              |  \                            |  \___
              |   \____                       |      __________
              +--------> lag                 +--------------> lag
                tails off                     cuts off after q

   PACF       |\                       PACF  |\
              | \                             | \
              |  \___ (zero after p)          |  \___
              |      ____                     |      \___
              +--------> lag                 +--------------> lag
                cuts off after p               tails off
```

Both terms are ways of writing "today depends on the past". An autoregressive (AR) term says the level
carries momentum: a high month tends to be followed by a high month. A moving average (MA) term says the
*shock* carries: an unusually high month leaves a residue that decays over the next few months rather than
disappearing at once. They are not different phenomena. They are two finite approximations of the same
infinite memory, and a model with five MA terms and a model with two AR terms can produce almost identical
forecasts on the same data. Section 5.2 develops this, because it explains why order selection is a
parsimony exercise rather than a search for truth.

```text
   THE PIPELINE: FILTER, MODEL, INTEGRATE

   y_t
    |  (1 - B^12)      seasonal difference, D = 1     removes the fixed seasonal shape
    v
   z_t
    |  (1 - B)         first difference, d = 1        removes the drifting level
    v
   w_t  ---------- ARMA(p,q)(P,Q) ----------> e_t  white noise
    |
    |  forecast w at horizons 1..h
    v
   z_hat = z_T + cumsum(w_hat)                       integrate once
    |
   y_hat = y_{T+h-12} + z_hat                        integrate the seasonal difference
```

Read the diagram from the bottom up, because that is the direction that matters operationally. A forecast
of `y` is built by accumulating forecasts of differences. The level forecast inherits every error in the
accumulation, which is why the interval widens with the horizon far faster than the ARMA part alone would
suggest.

Four properties follow, and they explain most of what is difficult about the family.

**Property 1 — The model forecasts differences, not levels.** After differencing, the object being
modelled is the monthly change (or the change from the same month last year). When an ARIMA forecast is
wrong by a constant amount in every period of the horizon, the level was mis-anchored and the shape was
right; when it is wrong by a growing amount, the shape was wrong. Section 6.5 uses exactly this signature.

**Property 2 — Integration makes errors cumulative.** A small bias in the forecast of the change is paid
once per period for the whole horizon. That is why a model can look excellent on one-step error and be
useless at twelve steps, and why the number to report is the error at the horizon you will actually use.

**Property 3 — The orders are statements about correlation, not about the business.** `p` and `q` say how
many lags of the series and of the shock are needed to describe the correlation structure; they carry no
operational meaning. A planner who asks "what does the second moving average term mean for my
replenishment?" is asking a question the model cannot answer. The seasonal period `m`, by contrast, is a
business fact, and getting it wrong is a data error rather than a modelling preference.

**Property 4 — The model has no opinion beyond the pattern it has seen.** There is no mechanism in ARIMA
for a product launch, a capacity constraint, a competitor exit, or a price step. Those enter only as an
intervention dummy, an exogenous regressor, or a structural break that the model will average into its
parameters for as long as its memory lasts.

If you remember one thing from this section, remember the vertical order of the pipeline: filter, model,
integrate. Nearly every diagnostic in section 5 asks "which stage produced this symptom?".

## 4. Definitions and vocabulary

The fourth column names the unit or coding case. A definition of a time series quantity without one is the
defect this module exists to prevent: `d` is a count of differences, `theta` is dimensionless, `m` is
periods per cycle, and the forecast is in units per period. Confusing any two of them produces a number
that is wrong by a factor of the bucket length.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Stationarity | A property of a series whose statistical behaviour does not change with time | Treated as "looks flat on a chart" | Property, no unit |
| Unit root | A characteristic root of the process equal to one, so shocks persist forever | Used as a synonym for "trend" | Root on the unit circle |
| Integration order `d` | The number of times a series must be differenced to become stationary | Read as "the trend is of order d" | Count of differences |
| Differencing | Replacing `y_t` by `y_t - y_(t-1)` | Applied twice "to be safe" | Units per period |
| Seasonal differencing | Replacing `y_t` by `y_t - y_(t-m)` | Confused with a seasonal dummy | Units |
| Over-differencing | Differencing a series that was already stationary | Invisible in the fit; visible in the forecast | Diagnostic label |
| Backshift operator `B` | The operator with `B y_t = y_(t-1)`, used to write filters compactly | Read as an algebraic variable that can be cancelled carelessly | Operator |
| Autoregressive term AR(p) | A regression of the series on its own previous `p` values | Described as "the trend component" | Order `p` |
| Moving average term MA(q) | A regression of the series on the previous `q` forecast errors | Confused with a moving average of the data | Order `q` |
| ARMA(p,q) | A stationary series modelled with `p` AR and `q` MA terms | Applied to a non-stationary series without differencing | Two orders |
| ARIMA(p,d,q) | ARMA applied to a series differenced `d` times | Written without stating `d` | Three orders |
| SARIMA | ARIMA with seasonal AR and MA terms at lag `m` | Called "seasonal ARIMA" without stating `m` | Orders plus `m` |
| SARIMAX | SARIMA with exogenous regressors | Described as a causal model | Orders, `m`, plus `k` regressors |
| Autocorrelation function (ACF) | The correlation between `y_t` and `y_(t-k)` for each lag `k` | Read on an undifferenced series to choose `p` | Dimensionless, −1 to 1 |
| Partial autocorrelation (PACF) | The correlation at lag `k` after removing the effect of lags 1 to `k-1` | Assumed to equal the ACF at lag 1 | Dimensionless, −1 to 1 |
| Correlogram | A plot of the ACF or PACF against lag, with a confidence band | Read without the band, or with the wrong band | Plot |
| White noise | A sequence with zero mean, constant variance, and no autocorrelation | Confused with "random-looking" | Series property |
| Innovation | The unpredictable part of the current observation given the past | Used as a synonym for residual | Units |
| Residual | The observed one-step error from a fitted model | Assumed to have the same variance as the innovation | Units |
| Invertibility | The property that lets a model be written as an infinite AR process | Assumed automatically when a fit converges | Property of the roots |
| AIC | Akaike information criterion: `-2 lnL + 2k`, fit penalised by parameter count | Compared across different `d` or different software | Dimensionless |
| AICc | AIC with a small-sample correction, `AIC + 2k(k+1)/(n-k-1)` | Omitted on series under about 60 observations | Dimensionless |
| BIC | Bayesian information criterion: `-2 lnL + k ln(n)` | Treated as strictly better than AIC for forecasting | Dimensionless |
| Log-likelihood | The log of the probability of the observed data under the fitted model | Compared across models fitted to different data | Dimensionless |
| Ljung-Box statistic | A portmanteau test for residual autocorrelation up to lag `K` | Run with degrees of freedom equal to `K` | Chi-square statistic |
| Exogenous regressor | A variable determined outside the model, supplied as an input | Assumed to be known at the forecast origin | Units of the driver |
| Leakage | Using, at forecast time, information that would not have been available | Called "using all available data" | Defect |
| Seasonal period `m` | Observations per seasonal cycle | Set to 12 on a 4-4-5 retail calendar | Periods per cycle |
| Multiplicative seasonal structure | A seasonal filter written as a product with the non-seasonal filter | Assumed to be additive because the model is linear | Product of polynomials |
| Airline model | ARIMA(0,1,1)(0,1,1)`m`, the canonical seasonal model | Treated as a default rather than a hypothesis | Model label |
| Drift | A constant added per period, equivalent to a deterministic linear trend | Confused with a fitted AR term | Units per period |

## 5. The mechanics, step by step

The procedure has six phases, which are the Box-Jenkins loop with the parts that practice has changed
since 1970 made explicit. Each step names what you do, what you check, and what goes wrong if you skip the
check.

### Phase A — Decide what the series is

**Step 1. Fix the bucket, the unit, the horizon, and the seasonal period.** Write one sentence: "`y_t` is
monthly demand for the SK-14 seal kit at one regional DC, in cases, for `t = 1` to 48, forecast to
`h = 12` , seasonal period `m = 12` ." Check that the bucket and the unit do not change mid-series, and
that `m` matches the calendar the series is actually on. Failure: a model with `m = 12` on a
thirteen-period retail year, which section 8.3 shows cannot be repaired by tuning.

**Step 2. Decide what the observation is.** Demand, shipment, orders, or consumption. A shipment series is
censored at available stock, so a stock-out truncates the observation exactly where demand was highest; an
order series contains batching and forward buying. Neither is demand. Failure: the model learns the
fulfilment constraint and forecasts it confidently.

**Step 3. Plot it and name the components you can see.** Level? Trend? A repeating shape at a period you
can name? A variance that grows with the level? A structural break? Do not fit yet. The plot decides which
of the next four steps you even need. Failure: applying a seasonal difference to a series with no
seasonality because the software's default said so.

### Phase B — Stationarise, and know what you removed

**Step 4. Decide `D`, the seasonal differencing order.** Ask whether the seasonal shape is fixed in units
and repeats. If it is, `D = 1` removes it: `z_t = y_t - y_(t-m)` is the change from the same month last
year, and a stable seasonal pattern cancels exactly. Look for a correlogram that decays very slowly with
spikes at `m`, `2m`, `3m` before differencing, and spikes that collapse after. `D = 2` is almost never
justified and is usually a symptom of a mis-specified trend.

**Step 5. Decide `d`, the non-seasonal differencing order.** On the seasonally differenced series, ask
whether the level still drifts. If `z_t` still rises or falls across the window, `d = 1`; if the change is
roughly centred on zero with no persistent sign, `d = 0`. Use a unit-root test as evidence, not as an
oracle: the augmented Dickey-Fuller (ADF) test has a null of a unit root, the
Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test has a null of stationarity, and the two disagree often
enough that the disagreement is itself informative. Failure: choosing `d` by whichever value makes the AIC
smallest, which is circular, as step 12 explains.

**Step 6. Check the cost of what you removed.** Compute the standard deviation at each stage of
differencing and tabulate it. Differencing a stationary series *raises* the variance — the variance of the
difference of two independent observations is twice the variance of either — so a second difference that
increases the standard deviation removed nothing and added noise. Section 5.1 gives the algebra and
section 6.2 gives the table.

**Step 7. Stop differencing as soon as the series is stationary.** The temptation is to keep differencing
until the plot looks flat. The plot always looks flatter after another difference, because differencing
removes low-frequency variation whether or not that variation was a unit root. Over-differencing produces
a non-invertible MA term, which is diagnosable but not harmless.

### Phase C — Identify candidate orders

**Step 8. Read the ACF and PACF on the differenced series, and only on the differenced series.** The
classic signatures are in the table below. They are a starting point: on fewer than about 50 effective
observations, one spike inside or outside the band is not evidence.

| Model | ACF | PACF | What it means in words |
|---|---|---|---|
| AR(p) | Tails off, exponentially or sinusoidally | Cuts off after lag `p` | Momentum of order `p` |
| MA(q) | Cuts off after lag `q` | Tails off | Shocks that decay over `q` periods |
| ARMA(p,q) | Tails off | Tails off | Both; the plots do not identify `p` and `q` |
| Seasonal AR(P) at lag `m` | Spikes at `m, 2m, ...` decaying | Cuts off after lag `m` | A year-ago effect with momentum |
| Seasonal MA(Q) at lag `m` | Cuts off after lag `m` | Spikes at `m, 2m, ...` decaying | A shock at the same month last year |
| Non-stationary | Decays very slowly; still above 0.8 at lag 10 | One large spike at lag 1 | A unit root, not a long memory |

**Step 9. State the candidate set before fitting anything.** Write down two to five models, for example
`(0,1,1)(0,1,1)12`, `(1,1,0)(0,1,1)12`, `(0,1,2)(0,1,1)12`, and `(0,1,1)(1,1,0)12`. A candidate set is a
hypothesis about the correlation structure, and it is the thing an automatic procedure cannot produce.
Failure: accepting whatever the engine returns without stating what alternatives were plausible.

**Step 10. Fit each candidate by maximum likelihood and record the parameters.** Record every parameter
with its value, whether it sits at a bound, and whether the roots of the AR and MA polynomials lie outside
the unit circle. A parameter reported at 0.9999 is not an estimate; it is the optimizer at the edge of its
box, and the model has collapsed to a simpler one.

### 5.1 What differencing actually removes, and what it costs

Take the operator seriously and the whole of `d` and `D` becomes arithmetic.

```text
   y_t = a                      constant            -> (1-B) y_t = 0
   y_t = a + b t                linear trend        -> (1-B) y_t = b        (a constant)
   y_t = a + b t                linear trend        -> (1-B)^2 y_t = 0
   y_t = s_t,  s_t = s_(t-m)    fixed seasonal      -> (1-B^m) y_t = 0
   y_t = y_(t-1) + e_t          random walk         -> (1-B) y_t = e_t      (white noise)
   y_t = e_t                    white noise         -> (1-B) y_t = e_t - e_(t-1)
```

The last line is the important one. Differencing a series that was already stationary produces a
first-order moving average with a coefficient of exactly one in the `(1 - theta B)` convention, or minus
one in the `(1 + theta B)` convention used by most software. A unit MA root is not invertible: the model
cannot be written as an infinite autoregression, the parameters are weakly identified, and the forecast
behaves like a straight line with an interval that grows without bound. That is the signature of
over-differencing, and it is visible in the fitted parameter table as `theta` close to plus or minus one.

**The observation cost.** Each difference loses one observation, and each seasonal difference loses `m`. A
48-month series with `d = 1` and `D = 1` at `m = 12` leaves 35 usable observations for estimation. That is
thin for two parameters, and thinner for six. The effective sample size is the number you should quote
when someone asks how much data the model had.

**The variance cost.** For a stationary series with variance `sigma^2` and lag-one correlation `rho`, the
variance of the first difference is `2 sigma^2 (1 - rho)`. At `rho = 0.9` that is `0.2 sigma^2`, so
differencing genuinely reduces variance; at `rho = 0` it is `2 sigma^2`, so differencing doubles it. A
differenced series whose standard deviation is higher than the original's is a series that did not need
differencing.

**The interpretability cost.** After differencing, the model is about changes. A coefficient of 0.4 on the
first lag of the differenced series says that a one-case increase this month predicts a 0.4-case increase
next month — not a 0.4-case level. Stakeholders read level statements. Section 6.6 shows how to translate
back.

### 5.2 Why AR and MA are two approximations of the same thing

Any stationary process with no deterministic component can be written as an infinite moving average of its
shocks — the Wold decomposition, and the reason the family works at all. If the process is invertible it
can equally be written as an infinite autoregression. ARIMA models are finite truncations of one or the
other, which explains three things practitioners find confusing.

**Why a fitted AR(2) and a fitted MA(5) can forecast almost identically.** Both approximate the same
infinite representation and converge to the same forecast as the orders grow. A difference in AIC of two
points between them is not evidence about the world.

**Why parsimony matters more than the correlogram.** Box and Jenkins insisted on the smallest model that
leaves white-noise residuals, because every parameter costs a factor in the forecast variance and on 35
observations the cost is large. A six-parameter model that fits marginally better than a two-parameter one
is usually worse out of sample — which is why AICc, penalising small samples harder, is the default in
serious software.

**Why the correlogram rules in step 8 fail so often in practice.** The rules assume the true orders are
small and the sample is long. On a 40-observation seasonal series the confidence band is plus or minus
`1.96/sqrt(n)` — about 0.31 — so a single autocorrelation of 0.34 is "significant" and means nothing. The
rules generate candidates; they do not choose between them.

### 5.3 The multiplicative seasonal structure, and the airline model

A seasonal ARIMA model writes the seasonal filter as a separate polynomial in `B^m`:

```text
   (1 - phi B)(1 - Phi B^m) (1 - B)^d (1 - B^m)^D y_t  =  (1 - theta B)(1 - Theta B^m) e_t
   \_____/    \___________/   \____/   \___________/      \_______/    \____________/
    AR(p)      seasonal AR(P)  diffs    seasonal diffs     MA(q)        seasonal MA(Q)
```

The word "multiplicative" describes how the two filters combine, not how they affect the data. A
non-seasonal difference applied to a seasonally differenced series is the same operation as the seasonal
difference applied to a first difference, because the operators commute. What the product buys is
parsimony: a single seasonal MA parameter describes a decay that repeats at every seasonal lag, instead of
needing one parameter per lag. Expanding the operators shows what the model does to the data:
`(1 - B)(1 - B^12) = 1 - B - B^12 + B^13` , so each value is the value one month ago subtracted, the value
twelve months ago subtracted, and the value thirteen months ago added back.

That is the **airline model**, ARIMA(0,1,1)(0,1,1)12, named for the Box-Jenkins analysis of international
airline passenger totals. It has two parameters and it fits a large share of monthly seasonal series
reasonably well, which is exactly why it is dangerous: a model that always looks acceptable discourages
the question of whether it is right.

What the airline model assumes about a supply chain series is specific and worth stating:

| Assumption | Consequence if it holds | Consequence if it fails |
|---|---|---|
| The seasonal shape is fixed in units and repeats exactly | One seasonal MA parameter captures it | Every peak is under-forecast and every trough over-forecast |
| The annual increment is a random walk | Next year's growth resembles this year's | A single unusual year is extrapolated for the whole horizon |
| Shocks persist at the seasonal lag | A shock this June lowers next June | A one-off event contaminates the forecast a year later |
| No drift beyond what is in the last annual increment | The forecast tracks recent growth | A genuine trend change is absorbed slowly |

The second row is the one that bites. Because there is no drift parameter, the model carries the most
recent annual increment forward as if it were the expected increment. In the worked example that produced
a forecast biased high by 27 cases a month, on a level near 750, from a model whose residuals passed every
test. Section 6.5 walks through it.

### Phase D — Select the orders, with a criterion you can defend

**Step 11. Compare candidates with AICc, on identical differenced data.** The Akaike information criterion
is `AIC = -2 lnL + 2k`, where `lnL` is the maximised log-likelihood and `k` the number of estimated
parameters. The small-sample correction `AICc = AIC + 2k(k+1)/(n-k-1)` exists because AIC is biased toward
complexity when `n/k` is small, which is the normal condition in demand planning. The Bayesian information
criterion (BIC) penalises with `k ln(n)` instead of `2k`, and so selects smaller models. AIC is
asymptotically equivalent to one-step-ahead out-of-sample mean squared error, which makes it a defensible
default for forecasting rather than merely a fit statistic; it is not, however, a forecast accuracy
measure, because it says nothing about multi-step performance. BIC is consistent — it selects the true
model with probability approaching one if a true model of finite order is in the candidate set — but in
demand forecasting that premise is almost never satisfied. A large study on roughly 91,000 series from the
M4 competition found BIC over-penalised and selected models that were too simple, particularly on seasonal
data, while AICc was the best of the criteria. Use AICc among models on the same data, prefer
cross-validation when you can afford it, and never treat a two-point difference as decisive: below about 2
is not evidence, above 10 is worth acting on.

**Step 12. Enforce the comparability rule.** AIC, AICc, and BIC are comparable only across models fitted
to the *same observations of the same transformed series*. Changing `d` or `D` changes the data, so a
comparison across differencing orders is meaningless. Changing the estimation window changes the data.
Changing software often changes the treatment of initial values and constants, which shifts the likelihood
by an additive term and therefore the criterion. State the rule in the model documentation, because it is
the one most often broken.

> **Note:** The rule has a corollary that surprises people: you cannot use AIC to choose `d`. Use
> unit-root tests and the variance table for `d` and `D`, then use AICc for `p, q, P, Q` conditional
> on the differencing you chose. This is exactly why the standard automatic procedure picks `d` by
> repeated unit-root tests before it ever evaluates a likelihood.

**Step 13. Run an automatic selection as a check, not as an answer.** The standard procedure — Hyndman and
Khandakar's algorithm, implemented as `auto.arima` in R and mirrored in most suites — chooses `d` by
repeated KPSS tests, chooses `D` from a seasonal-strength measure or a seasonal unit-root test, then
searches the `(p,q,P,Q)` lattice by stepwise AICc from four seed models, varying each order by one at a
time. It is a good procedure. It is also bounded: by default the search allows `p` and `q` up to 5, `P`
and `Q` up to 2, `d` up to 2, `D` up to 1, and a stepwise path that never explores combinations more than
one step from a seed.

Automatic selection is a starting point rather than an answer for four reasons, and each is a real
limitation rather than a complaint about software:

1. **It minimises an in-sample criterion.** AICc is asymptotically related to one-step-ahead
   out-of-sample error, not to twelve-step error, which is what the planner uses.
2. **It cannot know about interventions.** A promotion, a recall, a plant closure, or a price step
   is absorbed into the parameters unless you supply a dummy or truncate the window.
3. **The search space is the analyst's choice.** A maximum order of 5 encodes a belief, and a
   stepwise path that moves one order at a time can stall on a shoulder.
4. **It is selected on the same data it is evaluated on.** The reported AICc of the winner is
   optimistic, because the winner was chosen for that number. Only a backtest on data the selection
   never saw gives an honest comparison.

**Step 14. Test the chosen model against the benchmark from topic 01 before anything else.** If the
selected model cannot beat the seasonal-naive forecast on a rolling origin, the orders do not matter. This
check costs ten minutes and prevents most of the failures in section 10.

### 5.4 Exogenous regressors, SARIMAX, and the leakage trap

A SARIMAX model adds a regression on external variables to the seasonal ARIMA structure:

```text
   y_t  =  beta_1 x_1t + beta_2 x_2t + ... + u_t
   u_t  follows a SARIMA(p,d,q)(P,D,Q)m process
```

Read the equation carefully: the regression is on the *contemporaneous* value of each driver, and the
ARIMA structure applies to the residual `u_t`, not to the driver's effect. That has two consequences that
are easy to miss.

**The driver's effect is instantaneous.** Unlike a transfer function model, in which an input at time `t`
affects the output at `t`, `t+1`, and beyond through a lag polynomial, the standard SARIMAX form gives the
driver a single coefficient at lag zero. If a price change takes two months to reach demand, the model as
written will attribute the effect to whichever month the price series happens to move in, and the
coefficient will be attenuated. The fix is to lag the driver explicitly and include the lags you believe
in.

**The forecast is conditional on future driver values you must supply.** The model has no mechanism for
forecasting its own inputs. If you ask for twelve steps ahead, you must hand it twelve future values of
every driver, or it will refuse. That single mechanical fact is the entire leakage discipline.

Sort every candidate driver into one of four buckets before you fit anything:

| Bucket | Test | Example | Legitimate? |
|---|---|---|---|
| Known in advance | The value is fixed by contract or calendar before the origin | Day of week, holiday dates, contracted price step, planned promotion calendar | Yes |
| Forecastable | A separate model produces a defensible forecast with its own error | Category-level market demand, fuel index, freight rate | Yes, with the input error propagated |
| Planned but uncertain | A number exists but is a plan, not a commitment | Production plan, promotional spend, assortment change | Yes, with a scenario, not a point value |
| Known only after the fact | The value is measured after demand is realised | Actual fill rate, actual markdown, actual competitor stock-out, actual returns | No — this is leakage |

The fourth bucket is where supply chain models fail, because the tempting drivers are exactly the ones
that are measured after the fact. A fill-rate series is an outcome of demand and supply; using it to
forecast demand is forecasting a number with itself. A markdown series is decided after seeing
sell-through. A competitor stock-out series is usually reconstructed retrospectively. Each produces a
beautiful in-sample fit and an unusable forecast.

> **Note:** The practical test is mechanical, not philosophical: write down the date on which each
> driver's value for period `T+6` becomes known. If that date is after the forecast origin `T`, the
> driver cannot be used at that horizon. Doing this for every driver, in writing, converts an
> argument about causality into a table anyone can check.

### Phase E — Diagnose the residuals

**Step 15. Plot the residuals before testing them.** Four plots: residual over time, residual
autocorrelation, histogram, and normal probability plot. Look for a trend (a missing drift term), a funnel
(heteroscedasticity), a step (a structural break), and a spike (an outlier dominating the fit). A single
large outlier can drive every parameter in a 35-observation model.

**Step 16. Run a portmanteau test with the right degrees of freedom.** The Ljung-Box statistic is

```text
   Q = n (n + 2) * sum over k = 1..K of  r_k^2 / (n - k)
```

where `r_k` is the residual autocorrelation at lag `k` . Under the null of no residual autocorrelation,
`Q` is approximately chi-square with `K - p - q` degrees of freedom for a non-seasonal model, and
`K - p - q - P - Q` for a seasonal one. The degrees of freedom must subtract the estimated parameters;
using `K` instead makes the test too conservative and hides misfit. Remember also that the residuals of an
estimated model are not the innovations — their variance at small lags is less than `1/n` and they are
correlated across lags — so use the test as a screen, not a verdict. The Box-Pierce statistic is the same
sum without the `n(n+2)/(n-k)` correction, and it always under-detects misfit. Use Ljung-Box.

**Step 17. Interpret a borderline result honestly.** A p-value of 0.04 on 22 degrees of freedom means one
or two modest autocorrelations somewhere in the first two years of lags. It does not mean the model is
wrong, and it does not mean the forecast is bad. Look at which lag produced it: a spike at the seasonal
lag means the seasonal structure is wrong; a slow decay across many lags means the differencing is wrong;
a single spike at an arbitrary lag is usually noise.

### Phase F — Check that it forecasts, not just that it fits

**Step 18. Backtest on a rolling origin against the benchmark.** Fit on data up to origin `T`, forecast
`h` steps, record the error, move the origin forward, repeat. Report the error at the horizon the business
uses, not at one step, and report it against seasonal naive. Section 9.3 gives the evidence for why this
step decides the argument more often than the model comparison does.

**Step 19. Compare the in-sample error with the out-of-sample error at the same horizon.** If one-step
in-sample root mean squared error (RMSE) is 20 and twelve-step out-of-sample RMSE is 39, nothing has gone
wrong — that is the normal cost of the horizon. If the twelve-step error is three times the in-sample
one-step error *and* biased in one direction, the level is mis-anchored, and the fix is a trend or drift
term rather than a different order.

**Step 20. Decide whether the model earns its complexity.** Three outcomes are possible. The model beats
the benchmark and the damped smoothing alternative, in which case use it. The model beats the benchmark
but not the smoothing alternative, in which case use the simpler model and say so. The model beats
neither, in which case the honest answer is the benchmark. All three are respectable findings; only the
first is usually reported.

## 6. Worked example

Kestrel Components is a fictional distributor of hydraulic service parts. The SK-14 seal kit is forecast
at one regional distribution centre in cases per month. Sixty months of history are available; months 1 to
48 are used for estimation and months 49 to 60 are held out. The series was constructed to be internally
consistent, and every number below is reproducible from the table.

### 6.1 The data

| Month | Cases | Month | Cases | Month | Cases | Month | Cases |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 462.1 | 16 | 575.1 | 31 | 692.9 | 46 | 703.7 |
| 2 | 470.1 | 17 | 597.9 | 32 | 660.3 | 47 | 670.9 |
| 3 | 502.6 | 18 | 606.2 | 33 | 647.4 | 48 | 721.7 |
| 4 | 565.9 | 19 | 603.1 | 34 | 622.5 | 49 | 711.4 |
| 5 | 546.9 | 20 | 557.5 | 35 | 604.2 | 50 | 739.0 |
| 6 | 592.1 | 21 | 557.4 | 36 | 630.8 | 51 | 758.4 |
| 7 | 589.7 | 22 | 530.0 | 37 | 643.5 | 52 | 787.9 |
| 8 | 537.8 | 23 | 501.6 | 38 | 658.8 | 53 | 795.1 |
| 9 | 523.8 | 24 | 539.0 | 39 | 692.4 | 54 | 831.4 |
| 10 | 482.0 | 25 | 533.5 | 40 | 735.1 | 55 | 794.4 |
| 11 | 498.7 | 26 | 563.2 | 41 | 759.1 | 56 | 792.7 |
| 12 | 523.0 | 27 | 593.7 | 42 | 803.3 | 57 | 708.4 |
| 13 | 513.9 | 28 | 649.1 | 43 | 766.4 | 58 | 771.9 |
| 14 | 529.6 | 29 | 700.8 | 44 | 706.9 | 59 | 718.2 |
| 15 | 559.6 | 30 | 688.1 | 45 | 741.6 | 60 | 764.9 |

The series rises from about 460 to about 760 cases over five years, with a repeating shape that peaks in
late spring and troughs in late autumn, and an irregular component of roughly 17 cases. No unit change, no
zero months, no obvious break.

### 6.2 What each difference removes

The table below traces six consecutive months through each transformation. `d1` is the change on the
previous month, `d12` the change from the same month a year earlier, and `dd` the change in `d12` from one
month to the next.

| Month | `y` | `d1` | `d12` | `dd` |
|---:|---:|---:|---:|---:|
| 14 | 529.6 | +15.7 | +59.5 | +7.7 |
| 15 | 559.6 | +30.0 | +57.0 | −2.5 |
| 16 | 575.1 | +15.5 | +9.2 | −47.8 |
| 17 | 597.9 | +22.8 | +51.0 | +41.8 |
| 18 | 606.2 | +8.3 | +14.1 | −36.9 |
| 19 | 603.1 | −3.1 | +13.4 | −0.7 |

The variance table settles the question better than the eye:

| Series | Standard deviation (cases) | What the number says |
|---|---:|---|
| `y` | 86.78 | Level and seasonality dominate |
| `d1` | 31.78 | Trend removed, seasonality intact |
| `d12` | 33.40 | Seasonality removed, trend intact |
| `dd` | 25.53 | Both removed; close to the noise floor |

`dd` at 25.53 is close to what the two differences of a 17-case noise process would produce on their own,
which is the sign that nothing is left to remove. The reverse pattern — a second difference with a
*higher* standard deviation than the first — would be the sign of over-differencing, and section 5.1 gives
the algebra behind it.

### 6.3 Reading the correlogram

The ACF of the raw series starts at 0.88 and is still 0.35 at lag 12: it decays slowly, which is the
signature of a series that is not stationary. The ACF of `d1` collapses at lag 1 but keeps a spike of 0.44
at lag 12, which is the seasonal pattern. The ACF of `d12` is still 0.70 at lag 1, because the trend
survives the seasonal difference. Only after both differences does the correlogram settle:

| Lag | 1 | 2 | 3 | 4 | 5 | 6 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACF of `dd` | −0.41 | −0.14 | +0.34 | −0.14 | −0.01 | −0.11 | −0.22 |
| PACF of `dd` | −0.41 | −0.37 | +0.15 | +0.07 | +0.08 | −0.26 | −0.37 |

The confidence band at 35 effective observations is plus or minus `1.96/sqrt(35)` = 0.33. The lag-1 value
of −0.41 is outside it and the lag-12 value of −0.22 is inside it; the lag-3 value of +0.34 sits exactly
on the boundary. Two honest conclusions follow. First, a first-order moving average is supported at the
non-seasonal lag. Second, the seasonal moving average is *not* supported by the correlogram alone — the
lag-12 spike is too small on this sample to read. The candidate set is therefore `(0,1,1)(0,1,1)12`,
`(0,1,1)(0,1,0)12`, `(1,1,1)(0,1,1)12`, and `(0,1,2)(0,1,1)12`, and the choice among them has to be made
by fitting. This is the normal condition, not a failure of method: the correlogram proposes, the
likelihood disposes.

### 6.4 The fit

Fitting the four candidates by conditional least squares on the 35 usable observations gives:

| Model | Parameters | SSE (cases²) | AICc | Verdict |
|---|---|---:|---:|---|
| `(0,1,1)(0,1,1)12` | `theta = 0.55`, `Theta = 0.50` | 14,654.8 | 315.00 | Selected |
| `(0,1,1)(0,1,0)12` | `theta = 0.55` | 17,355.5 | 318.67 | 3.7 AICc points worse |
| `(1,1,1)(0,1,1)12` | three parameters | 14,498.1 | 317.20 | No gain for the extra parameter |
| `(0,1,2)(0,1,1)12` | three parameters | 14,402.6 | 316.90 | No gain for the extra parameter |

The selected model is the airline model of section 5.3. Its log-likelihood is −155.31, giving
`AIC = 2(155.31) + 2(2) = 314.63` and `AICc = 314.63 + 2(2)(3)/32 = 315.00` . The one-step residual
standard deviation is `sqrt(14,654.8/35) = 20.46` cases.

> **Note:** The MA parameters are quoted in the `(1 - theta B)(1 - Theta B^m)` convention used
> throughout this topic. R's `arima()` and statsmodels' `SARIMAX` write `(1 + theta B)` and will
> report −0.55 and −0.50 for the same fit. The sign is a convention, not a finding, and it is the
> first thing to check when two tools appear to disagree.

The residuals are clean, and the numbers are worth reading rather than accepting: autocorrelations of
−0.07, −0.10, +0.24, +0.03, +0.04, and −0.21 at lags 1 to 6, and −0.03 at the seasonal lag. The Ljung-Box
statistic is 8.98 at `K = 12` with `12 - 2 = 10` degrees of freedom, and 17.15 at `K = 24` with 22 degrees
of freedom. Neither rejects. By every in-sample criterion the model is adequate.

### 6.5 The forecast, and the error the diagnostics did not see

Forecasting twelve months from the end of month 48 gives the path below, against what actually happened.

| `h` | Month | Forecast | Actual | Error |
|---:|---:|---:|---:|---:|
| 1 | 49 | 717.3 | 711.4 | +5.9 |
| 3 | 51 | 767.5 | 758.4 | +9.1 |
| 6 | 54 | 863.6 | 831.4 | +32.2 |
| 9 | 57 | 807.9 | 771.9 | +36.0 |
| 12 | 60 | 791.4 | 764.9 | +26.5 |

The first forecast can be checked by hand. The seasonal difference at month 48 is
`z_48 = y_48 - y_36 = 721.7 - 630.8 = 90.9` cases per year. The model's forecast of the next change in
that annual increment is

```text
   w_49 = -0.55 * e_48 - 0.50 * e_37 + 0.275 * e_36
        = -0.55 * 12.98 - 0.50 * 15.91 + 0.275 * (-7.44)
        = -7.14 - 7.96 - 2.05 = -17.14 cases

   z_49 = z_48 + w_49 = 90.90 - 17.14 = 73.76
   y_49 = y_37 + z_49 = 643.5 + 73.76 = 717.26
```

The forecast is right about the shape and wrong about the level. Its shape is almost exactly the actual
shape, because the seasonal structure forces next year to repeat this year's pattern; its level is 27
cases too high on average, because the model carries the last annual increment of 90.9 cases forward and
that year happened to be an unusually strong one for a series whose underlying drift is 26.4 cases per
year.

**This is the central lesson of the topic, and it is why the diagnostics passed.** Residuals are
one-step-ahead errors inside the estimation window. They test whether the model describes the correlation
structure of the past, and this one does. They cannot test whether the annual increment is a random walk,
because that is an assumption about the future, not a property of the residuals. A model can fit and still
forecast badly, and the only test that catches it is a backtest.

### 6.6 The comparison that decides whether the model earns its place

| Method | MAE (cases) | MASE | RMSE (cases) | Mean error (cases) |
|---|---:|---:|---:|---:|
| Seasonal naive | 53.06 | 0.827 | 56.48 | −47.52 |
| ARIMA `(0,1,1)(0,1,1)12` | 27.86 | 0.434 | 38.72 | +27.25 |
| Holt-Winters, additive, damped | 18.93 | 0.295 | 25.83 | −1.53 |

The mean absolute scaled error (MASE) divides each method's mean absolute error by 64.13 cases, the
in-sample mean absolute error of the seasonal-naive forecast. MASE of 1.0 means "no better than seasonal
naive"; 0.434 means roughly 57% better.

The ARIMA model beats the benchmark comfortably, which is the minimum bar. It loses to a Holt-Winters
model with additive seasonality and a trend, fitted on the same 48 months with a coarse grid search that
selected `alpha = 0.50`, `beta = 0.02`, `gamma = 0.30`, and `phi = 1.00`. Three caveats must be stated
with that result, and stating them is part of the finding:

1. **One origin, one split.** This is an anecdote, not a benchmark. A single twelve-month holdout
   can favour either method by chance; topic 08 is where this is done properly.
2. **The smoothing parameters came from a coarse grid**, and the damping parameter landed on its
   upper bound, so the winner is an undamped trend model rather than a genuinely damped one.
3. **The comparison is against one specification**, not the best available smoothing model;
   topic 02's full ETS family would be the fair opponent.

What survives those caveats is the pattern that section 9.3 documents at scale: on a seasonal series with
a modest trend, ARIMA's advantage over a well-chosen smoothing model is not automatic, and the smoothing
model is cheaper to specify, cheaper to explain, and more robust to a mis-specified difference. The ARIMA
model here is a defensible choice. It is not an obvious one.

## 7. Code, query, or calculation

One runnable block that reproduces section 6, with the assumptions stated. It requires the 60 monthly
values from section 6.1 in a two-column file, `seal_kit_monthly.csv`, with columns `month` and `cases`.

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

y = pd.read_csv("seal_kit_monthly.csv", index_col="month", parse_dates=True)["cases"]
train, test = y.iloc[:48], y.iloc[48:]          # 48 months fit, 12 held out

model = SARIMAX(train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                trend="n", enforce_stationarity=False, enforce_invertibility=True)
fit = model.fit(disp=False)
print(fit.params[["ma.L1", "ma.S.L12", "sigma2"]].round(3))   # -0.55, -0.50, 418.7
k, n = 2, int(fit.nobs)
print(round(fit.aic + 2 * k * (k + 1) / (n - k - 1), 2))       # ~315 (see note)

fc = fit.get_forecast(steps=12)
print(fc.predicted_mean.round(1).tolist())                     # 717.3 ... 791.4
print(fc.conf_int(alpha=0.05).round(1).head(3))
fit.plot_diagnostics(figsize=(9, 6))                           # residual ACF, histogram, Q-Q
```

**Expected output.** The MA parameters print as −0.55 and −0.50 because statsmodels writes the MA
polynomial as `(1 + theta B)`; these are the same fit as the +0.55 and +0.50 in section 6.4. The forecast
list begins at 717.3 and ends at 791.4. The AICc is close to 315.0 but not identical, because statsmodels
maximises the exact likelihood and section 6.4 uses conditional least squares with the first thirteen
observations treated as fixed.

**Three checks before trusting the output.** Confirm `fit.mle_retvals["converged"]` is true; a
non-converged fit still prints parameters. Inspect `fit.params` for values within 0.001 of the
invertibility boundary — a MA coefficient of ±0.999 means the differencing is wrong. And compare
`fc.predicted_mean` at `h = 1` with the hand computation; a disagreement of more than a few cases means
your reading of the differencing convention does not match the software's.

## 8. Data, standards, and regulatory context

No standard mandates ARIMA, and none specifies an order. That absence is the regulatory context: because
the method is unregulated, documentation is the analyst's entire defence, and the standards that matter
are the ones that fix the clock, the vocabulary, and the audit trail.

### 8.1 What the series must contain before you fit

| Question | Why it matters | What to record |
|---|---|---|
| Is the observation demand, shipment, order, or consumption? | Each has a different variance and a different censoring rule | Source system and field name |
| What is the bucket, and is it constant? | Every order and rate scales with the bucket | Bucket length and boundary convention |
| Is the unit constant across the window? | A unit change reads as a structural break | Unit of measure and any conversion |
| Are zero periods present as rows? | Missing zeros inflate the level and bias `d` | Row count per period and the zero rule |
| How many seasonal cycles are complete? | Fewer than two makes `D` and `P, Q` unidentified | Cycle count |
| Is the series censored by stock-outs? | Sales are truncated at available inventory | Service level across the window |
| Has the history been restated? | Restatements break backtest comparability | Vintage or snapshot date |

The last row is specific to ARIMA and is often missed. Because order selection and backtesting depend on
the exact series, a demand history restated monthly — through returns, reclassifications, or unit
corrections — means a model fitted in March and one fitted in June were fitted to different data. Both may
be correct and neither is comparable. Store the vintage with the model.

### 8.2 Standards that fix the clock and the vocabulary

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 8601-1:2019 and ISO 8601-2:2019 | Date and time representation | What a period is, hence what `m` means | B |
| ISO 3534-1:2016 | Statistical vocabulary and symbols | The distinction between a prediction interval and a confidence interval | B |
| ISO 5725 series | Accuracy, trueness, and precision | Separates measurement error from process variation | B |
| GS1 EPCIS 2.0 | Event data for objects in a supply chain | The occurrence timestamps from which a series is built | B |
| ISO 8000 series | Data quality | Completeness and validity of the input series | B |

**ISO 3534-1** settles the collision that matters most in the forecast pack. The interval from an ARIMA
model is a prediction interval: a statement about a future observation that includes process noise. A
confidence interval for a parameter is a different object and is usually much narrower. Software labels
the first as the second often enough that the error has become conventional.

### 8.3 The calendar decides `m`, and `m` decides the model

The seasonal order is not a free parameter. It is the number of observations per cycle in the calendar the
business actually runs.

| Calendar | Period definition | Correct `m` | Effect of using `m = 12` | Grade |
|---|---|---|---|---|
| Gregorian month | 28–31 days | 12 | Correct, with a small period-length artifact | B |
| 4-4-5 retail | 4, 4, 5 weeks repeating | 13 | Seasonal term is an aliasing artefact | C |
| 13-period | Thirteen four-week periods plus a residual week | 13 | As above, plus a residual-week discontinuity | C |
| Weekly, ISO | Monday-start week | 52 (53 in a 53-week year) | Meaningless; the filter never aligns | B |
| Daily | Calendar day | 7 for day-of-week | Weekly pattern smeared across a 12-day lag | B |

The 4-4-5 row is the one that causes silent damage in retail and consumer goods. A seasonal ARIMA fitted
with `m = 12` to a thirteen-period year is estimating a seasonal relationship between periods that are not
the same period, and no amount of order search repairs it. Either convert the series to Gregorian months
before fitting, or set `m = 13`. The 4-4-5 calendar is an industry convention maintained by the National
Retail Federation rather than a statutory standard, so the grade is C for the calendar itself.

### 8.4 Model risk and documentation

Where a forecast feeds a reported number — inventory provisions, obsolescence reserves, revenue guidance,
service-level commitments — it is a model in the sense used by model risk frameworks. The Federal
Reserve's SR 11-7 is the reference document, and it asks for conceptual soundness, documented development,
ongoing monitoring, and outcomes analysis.

An ARIMA model that reports a forecast without stating `d`, `D`, the seasonal period, the estimation
window, the software, the treatment of initial values, and the parameter bounds would not pass such a
review. The remedy is not more prose. It is a specification sheet with nine fields, which exercise 7 in
section 12 asks you to write. Grade B for the guidance; its application to demand forecasting is an
inference, not a requirement.

### 8.5 What is convention rather than requirement

- The airline model as a default is a convention: a hypothesis about one covariance structure,
  wrong on many series.
- The `±1.96/sqrt(n)` correlogram band assumes independent autocorrelations, which they are not.
- `K = 24` for Ljung-Box on monthly data is a habit; `K = 2m` is the usual starting point, and the
  criterion is that autocorrelations beyond `K` should be negligible.
- "Two seasonal cycles minimum" is a rule of thumb; three is better and four is comfortable. The 5%
  level is a screen inherited from a different literature, not a decision rule.

An analyst who can separate a requirement from a convention can challenge a convention. An analyst who
cannot will treat a software default as a statistical law.

## 9. Industry practice and case evidence

### 9.1 Where the method came from, and why it dominated for thirty years

Box and Jenkins published *Time Series Analysis: Forecasting and Control* in 1970, and the book did
something no predecessor had done: it gave forecasting a loop. Identify a candidate structure from the
correlogram, estimate it by maximum likelihood, diagnose the residuals, and return to identification if
the residuals were not white noise. The loop was teachable, and it produced a likelihood and intervals.

That combination is why ARIMA was, for roughly three decades, the only fully respectable option in
academic time series. Exponential smoothing was widely used in industry and widely regarded as heuristic,
because it had no likelihood and therefore no principled intervals — a judgement that held until Hyndman,
Koehler, Snyder and co-authors put the family into state-space form in the 2000s, which is exactly the
story told in topic 02. That history explains a great deal of the institutional conservatism you will
meet: a planning organisation that adopted ARIMA in 1995 for defensible reasons may still be running it
for reasons that no longer apply.

What practitioners still argue about is narrower and more interesting. Whether correlogram identification
is worth teaching now that search is cheap (the argument for: it teaches what the model is doing; the
argument against: the rules fail on the sample sizes practitioners actually have). Whether AIC or
cross-validation should choose orders — step 11 gives the evidence, which favours AICc among criteria. And
whether `d` should be tested or chosen by judgement, where the tests disagree often enough that judgement
is not optional.

### 9.2 What the forecasting competitions found

The Makridakis competitions are the closest thing the field has to a controlled trial, and they are
unusually unkind to the assumption that ARIMA is the accurate choice.

| Competition | Design | Result relevant to this topic | Grade |
|---|---|---|---|
| M3 (Makridakis and Hibon, 2000) | 3,003 series, 24 methods | Theta best overall at 13.01% sMAPE; automatic Box-Jenkins 14.01%; exponential smoothing methods occupied most of the top of the table | A |
| M3 recalculation (Hyndman, Mcomp) | Same series, R implementations | ETS 13.07% sMAPE; automatic ARIMA 13.57%; the average of the two 12.77% | A |
| M4 (Makridakis, Spiliotis, Assimakopoulos, 2020) | 100,000 series, 61 methods | The benchmark to beat was a combination of three exponential smoothing models; a minority of entrants beat it | A |
| Statistical vs machine learning (Makridakis et al., 2018) | 1,045 monthly M3 series | ARIMA and ETS statistically indistinguishable over the full horizon (11.28% against 11.26% sMAPE in the comparison summarised in Mcomp), with ARIMA far more computationally expensive | A |

Three conclusions follow. **The two classical families are close enough that the choice is usually decided
by something other than accuracy** — robustness, cost, explainability, and the length of the series.
**Combination beats selection**, which is the empirical case for topic 07. And **automatic selection is
competitive, not optimal**: automatic ARIMA performing well in M3 is evidence that the procedure is sound,
not that the selected model is best for a given series.

> **Note:** sMAPE and MASE are computed on the competition's series mix, which is not your series
> mix. The gap between 13.07 and 13.57 is a statement about a heterogeneous collection of 3,003
> series, not a prediction that ETS will beat ARIMA on your SKU. Use it as a prior, not a result.

### 9.3 The honest comparison with damped smoothing

The claim that ARIMA is beaten by a damped exponential smoothing model on many supply chain series is
supported by the competition evidence and by the structure of the two methods. ARIMA's weaknesses on
operational demand are specific. It needs `d` and `D` to be right, and a mis-specified difference costs
more than a mis-specified smoothing constant, because it changes the object being modelled rather than the
weight on the recent past. Its seasonal structure repeats last year's shape mechanically unless the
seasonal MA term is small, and it has no natural damping, so a trend either persists or is removed. It
also needs more data than a smoothing model to estimate the same number of effective parameters.

Smoothing's weaknesses are equally specific: it cannot represent a moving average error structure, so it
responds to a shock with a lag rather than a decay; it handles a seasonal pattern that changes shape
poorly; and it cannot use exogenous information at all.

The practical consequence is the reason topic 01 exists. On a portfolio of thousands of series, the
defensible default is to fit a damped ETS and a seasonal-naive benchmark, and to fit ARIMA where there is
a specific reason to believe the error structure is moving-average, where the series is long, and where
someone will read the diagnostics. The reverse default — ARIMA everywhere because it is the more
sophisticated name — is common and is not supported by the evidence.

### 9.4 What enterprise systems do, and what the public record omits

Planning modules expose ARIMA as an automatic option, usually with a maximum order, a seasonal period
inherited from the item's calendar, and a stepwise search that the planner cannot see. Product
documentation reviewed for this course in 2026 confirms that the orders and the criterion are configurable
and that the selected orders are typically not surfaced in the planner's view. Grade D: vendor
documentation, not evidence of behaviour.

The consequence is a specific organisational failure: two planners, two items from the same family, two
different selected orders, no visible explanation. When the forecast is questioned, nobody can say what
model produced it. The fix is not a better engine but to surface the selected orders and the AICc in the
planner's worksheet, which costs nothing and changes the conversation.

Company-level evidence that isolates the ARIMA specification is thin. Disclosures rarely state the model,
the orders, the differencing, or the estimation window, and vendor cases lack a baseline. There is no
public, audited case of a supply chain failure attributable to a specific order choice, and this topic
does not invent one. The direction of the evidence is clear — ARIMA is competitive, rarely dominant, and
fragile to differencing errors — but the magnitude is setting-dependent. Validate on your own series with
a rolling-origin backtest rather than settling it by citation.

## 10. Failure modes and red flags

Each row names the failure, then a red flag that is visible in a meeting or in a dataset — not a
diagnostic you would have to run, but something you can see across the table.

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Differencing a stationary series | "We differenced twice because the first difference still had a trend" | Second difference adds an MA unit root and doubles the noise | Unstable forecast, enormous intervals, parameters at the invertibility boundary | Check the variance table; stop at the first stationary difference |
| Leaving a unit root in | Forecast converges to the mean of the history from a level far above it | The AR polynomial has a root at one; the model is mean-reverting by construction | Systematic under-forecast on a trending series | Unit-root test; plot the forecast against the last level |
| Comparing AIC across `d` | "We tried `d = 0, 1, 2` and picked the lowest AIC" | Each `d` changes the data and the observation count | A meaningless comparison reported as a result | Choose `d` and `D` first; compare AICc only at fixed `d, D` |
| Reading the correlogram before differencing | The identification table is applied to the raw series | A unit root produces a spurious PACF cut-off at lag 1 or 2 | Model that regresses the level on its own past | Read ACF and PACF only on the differenced series |
| Automatic selection treated as an answer | No candidate set, no backtest, only the engine's output | The search is bounded, local, and selected on the same data | Model that is adequate in sample and unexamined out of sample | State candidates; backtest against seasonal naive |
| Wrong seasonal period | `m = 12` on a 4-4-5 calendar or on weekly data | The seasonal filter never aligns with the true cycle | Seasonal term is an aliasing artefact | Match `m` to the calendar; convert the bucket |
| Exogenous driver known only after the fact | The driver is a fill rate, a markdown, or a realised return | The model is fed an outcome of the variable it forecasts | Excellent in-sample fit, unusable forecast | Classify every driver by the date its value becomes known |
| Future driver held flat at the last value | The twelve-month forecast uses an unchanged promotional spend | The software cannot forecast its own inputs and the analyst filled forward | Forecast conditional on a scenario nobody stated | State the scenario, or forecast the driver separately and propagate |
| Model reported on in-sample error only | One-step RMSE quoted for a twelve-month decision | Fit statistics describe one-step performance in the window | Model that fits and cannot forecast (section 6.5) | Report error at the horizon used, on a rolling origin |
| No benchmark in the comparison | "ARIMA improved accuracy by 18%" with no seasonal-naive number | Improvement is measured against the previous model, not against a baseline | Complexity bought with no evidence | Require MASE against seasonal naive in every pack |
| Structural break inside the estimation window | A level step or a slope change in the residual plot | Parameters are an average of two regimes | Forecast describes a process that no longer exists | Intervention dummy, or truncate the window at the break |

One of these deserves a note.

**Leaving a unit root in is the mirror image of over-differencing, and it is more common in practice**
because software that chooses `d` by unit-root test can fail to reject when the sample is short — the test
has low power against a near-unit root. The symptom is a forecast that converges towards a level well
below the recent data, which looks conservative and is wrong. If the forecast for month 12 sits below
every observation of the last year, check `d` before checking anything else.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The series is stationary after `d` and `D` differences | The ARMA representation is defined only for stationary processes | Parameters are unidentified; forecast reverts to a spurious mean | Unit-root tests; slow ACF decay; forecast path against recent levels |
| The differenced series is linear in its past | The model is a finite linear filter | Nonlinear structure (regime switches, thresholds) is absorbed into noise | Residual plots by regime; compare against a regime-switching alternative |
| Errors are uncorrelated | The likelihood treats them as independent | Standard errors and intervals understate uncertainty | Residual ACF; Ljung-Box with correct df |
| Errors are homoscedastic | One `sigma^2` describes the whole window | Intervals too narrow where variance is high | Residual standard deviation by sub-period; plot the residuals |
| Errors are Gaussian | The interval multiplier and the AIC penalty assume it | Asymmetric or heavy-tailed intervals; AIC comparisons distorted | Histogram and normal probability plot of residuals |
| The seasonal pattern repeats at a fixed period | The seasonal filter assumes a fixed `m` | The seasonal term is misaligned and the forecast shape is wrong | Plot seasonal subseries; check the calendar |
| The seasonal amplitude is stable in units | The airline model is additive in the seasonal difference | Peaks under-forecast and troughs over-forecast, worsening over time | Seasonal amplitude against level; consider a log transform |
| Exogenous regressors are known at the forecast origin | The forecast is conditional on them | Leakage; an unusable forecast with an excellent fit | The "date known" table from section 5.4 |
| The estimation sample is long enough to identify the orders | Each parameter costs forecast variance | Overfitting; unstable orders across windows | Effective sample size; refit on a shorter window |

Three of these are limits rather than assumptions, and they should be stated as limits in any model
documentation.

**ARIMA cannot represent a level shift without help.** A price change, a new customer, or a distribution
change moves the level permanently. The model treats the shift as a large shock and decays it according to
its MA structure, so the forecast is wrong for as long as the decay lasts. The fix is an intervention
dummy or a truncated window, and both are analyst decisions.

**ARIMA's intervals assume the future resembles the past.** They are conditional on the model being
correct, including no structural break in the forecast window, and there is no diagnostic inside the model
that can tell you when that assumption has failed.

## 12. Exercises

### Exercise 1 — What does a difference remove? (calculation, 25 minutes)

Series A is `y_t = 400 + 6t`; series B is `y_t = 400 + e_t` where `e_t` is white noise with standard
deviation 20; series C is a fixed seasonal pattern repeating at `m = 4`.

Compute `(1-B)y_t` for A, for B, and for C, and state what each result is and what it means for choosing
`d`. **Deliverable:** three difference expressions with a one-line interpretation each.
**Acceptance criteria:** A gives a constant, B gives an MA(1) with a unit coefficient and a variance of `2 x 400`, C is
not removed by a first difference and is zero under a seasonal difference. **Time:** 25 minutes.

### Exercise 2 — Variance table (calculation, 30 minutes)

Using the values in section 6.1, reproduce the standard deviations for `y`, `d1`, `d12`, and `dd` over
months 1 to 48, and state whether each difference was justified. **Deliverable:** a four-row table with
units and a sentence of interpretation. **Acceptance criteria:** values within 0.5 cases of section 6.2;
the interpretation distinguishes "removed something" from "removed nothing and added noise"; the effective
sample size after both differences is stated. **Time:** 30 minutes.

### Exercise 3 — Read a correlogram (analysis, 35 minutes)

On a monthly series with 40 observations, the ACF of the differenced series is 0.55 at lag 1, 0.38 at lag
2, 0.21 at lag 3, and 0.05 at lag 4, with a spike of 0.42 at lag 12. The PACF is 0.55 at lag 1, 0.06 at
lag 2, and 0.03 at lag 3. Name three candidate models, state the confidence band, and say which reading
you would refuse to make from this evidence alone. **Deliverable:** three candidates in `(p,d,q)(P,D,Q)m`
notation, the band, and the refused reading. **Acceptance criteria:** the band is computed as
`1.96/sqrt(n)` with `n` stated as effective observations, not raw rows; one candidate is seasonal; the
refused reading names a specific ambiguous pattern. **Time:** 35 minutes.

### Exercise 4 — The comparability rule (critique, 25 minutes)

A colleague reports: "ARIMA(1,0,0) has AIC 412.3, ARIMA(1,1,0) has AIC 388.9, ARIMA(0,2,1) has AIC 379.4,
so the third is best." Write the correction, and compute what would have to be true for the comparison to
be valid. **Deliverable:** a one-paragraph correction and the condition under which the numbers become
comparable. **Acceptance criteria:** the correction names the data change, not just the criterion; the
condition is stated as an operation the analyst could perform; the answer notes that a lower AIC on a
differenced series is expected, not informative. **Time:** 25 minutes.

### Exercise 5 — Airline model arithmetic (calculation, 40 minutes)

For ARIMA(0,1,1)(0,1,1)12 with `theta = 0.55` and `Theta = 0.50` , write out `w_t` as a function of
current and past errors, then compute the forecast of `w` at `T+1` and `T+2` given `e_T = 12.98` ,
`e_(T-1) = -16.54` , `e_(T-11) = 15.91` , `e_(T-12) = -7.44` , and all earlier errors zero.
**Deliverable:** the expansion, two forecast values, and the implied `z` and `y` forecast if `z_T = 90.9`
and `y_(T-11) = 643.5` . **Acceptance criteria:** the expansion has four terms; the `T+2` forecast uses
`e_(T-10)` and `e_(T-11)` correctly; the integration is performed in the right order. **Time:** 40
minutes.

### Exercise 6 — Classify the drivers (design, 30 minutes)

For a monthly forecast of a snack product, classify each candidate driver as known in advance,
forecastable, planned but uncertain, or known only after the fact: Easter date; average category price
index; planned promotional spend; realised promotional lift; competitor stock-out days reconstructed from
panel data; contracted list price; actual fill rate; number of selling days. **Deliverable:** an eight-row
table with the bucket, the date the value becomes known, and whether you would use it.
**Acceptance criteria:** at least two are correctly placed in the leakage bucket; the "date known" column is a date,
not a judgement; the answer distinguishes "forecastable" from "known". **Time:** 30 minutes.

### Exercise 7 — Write the specification (design, 45 minutes)

Write a one-page model specification for the SK-14 series, or for a series you know, in a form a reviewer
could reproduce. **Deliverable:** bucket, unit, `m`, `d`, `D`, `(p,q)(P,Q)`, estimation window, software
and version, initial-value treatment, parameter values with bounds, criterion and its value, benchmark,
backtest design, and residual tests with lags. **Acceptance criteria:** a second analyst could refit from
the document alone; every parameter has a unit or is marked dimensionless; the benchmark and the error
metric are named; the backtest states the origin count and the horizon. **Time:** 45 minutes.

### Exercise 8 — Fit versus forecast (critique, 40 minutes)

A model has one-step in-sample RMSE of 20.5 cases, twelve-step out-of-sample RMSE of 38.7 cases, a mean
error of +27.3 cases, and residuals that pass Ljung-Box at `K = 24`. Write a one-page diagnosis that
explains how all four facts can be true at once, and name the specification change that would fix the
bias. **Deliverable:** a causal account, the arithmetic linking one-step and twelve-step error, and one
specific repair. **Acceptance criteria:** the answer distinguishes fit from forecast; it explains why the
residual test cannot detect the problem; the repair addresses the level, not the orders. **Time:** 40
minutes.

## 13. Self-check answers

**Exercise 1.** Series A: `(1-B)(400 + 6t) = 6` for every `t` . A constant, so one difference removes a
linear trend completely and `d = 1` is sufficient. Series B: `(1-B)e_t = e_t - e_(t-1)` , an MA(1) with a
coefficient of exactly one in the `(1 - theta B)` convention and variance `2 x 400 = 800` . Differencing a
stationary series manufactured a unit MA root and doubled the variance, which is the definition of
over-differencing. Series C: a fixed seasonal pattern is unchanged by a first difference — `(1-B)s_t` is
not zero — and is annihilated by `(1-B^4)` . So the correct treatment is a seasonal difference with
`d = 0` , which is the case most often got wrong.

**Exercise 2.** The values are 86.78, 31.78, 33.40, and 25.53 cases. `d1` reduces the standard deviation
by 63%, which says a drifting level was present. `d12` reduces it by 62% relative to the level but leaves
it slightly *above* `d1`, because the trend survives: a seasonal difference removes the shape and keeps
the slope. `dd` at 25.53 is the lowest of the four, and it is close to what two differences of a 17-case
noise process would produce, so nothing is left to remove. Both differences were justified; a third would
not be. The effective sample size is `48 - 1 - 12 = 35`.

**Exercise 3.** The band is `1.96/sqrt(40) = 0.31`. The ACF tails off from 0.55 and the PACF cuts off
after lag 1, which is the signature of an AR(1) with no differencing. The lag-12 spike of 0.42 is outside
the band, and with an ACF that cuts off rather than tails off at the seasonal lag it suggests a seasonal
MA(1). Candidates: `(1,0,0)(0,0,1)12`, `(1,0,0)(1,0,0)12`, and `(1,0,1)(0,0,1)12`. The reading to refuse
is any claim about `q` from the non-seasonal lags: the ACF tails off smoothly, which is consistent with an
MA component and with a higher-order AR, and the two cannot be separated on 40 observations by eye. Fit
the candidates and compare AICc.

**Exercise 4.** The three AIC values are computed on three different datasets: the level series for
`d = 0` , the first difference for `d = 1` , and the second difference for `d = 2` . Each difference
removes an observation and changes the scale of what is being modelled, and the likelihood is a likelihood
of the data actually fitted. The comparison is not merely unfair; it is undefined. For the numbers to
become comparable, the analyst would have to hold `d` fixed — choose `d` by unit-root test and by the
variance table — and then compare models within that `d` . A falling AIC as `d` increases is expected,
because differencing moves variance from the level into the error term and the criterion rewards a
better-fitting error.

**Exercise 5.** `w_t = e_t - 0.55 e_(t-1) - 0.50 e_(t-12) + 0.275 e_(t-13)` . At `T+1` :
`w = -0.55(12.98) - 0.50(15.91) + 0.275(-7.44) = -17.14` . At `T+2` , taking `e_(T+1) = 0` and using the
known values: `w = -0.50(-7.44) + 0.275(15.91) = 3.72 + 4.38 = 8.10` . Integration gives
`z_(T+1) = 90.90 - 17.14 = 73.76` and `y_(T+1) = 643.5 + 73.76 = 717.26` . For `T+2` ,
`z_(T+2) = 73.76 + 8.10 = 81.86` and `y_(T+2) = y_(T-10) + 81.86 = 658.8 + 81.86 = 740.66` .

**Exercise 6.** Known in advance: Easter date, contracted list price, number of selling days (derivable
from the calendar). Forecastable: the category price index (with its own model and error). Planned but
uncertain: planned promotional spend. Known only after the fact: realised promotional lift, competitor
stock-out days reconstructed from panel data, actual fill rate. The fill rate is the clearest case: it is
an outcome of demand and supply in the same month, so using it to forecast demand is circular. The
realised lift is leakage for the same reason — it is measured from the demand it is supposed to explain.

**Exercise 7 rubric.** A strong specification states the bucket, unit, and `m`; gives `d` and `D` with the
evidence that chose them; lists `(p,q)(P,Q)` and the criterion value; names the software and version and
the initial-value treatment; reports every parameter with its value and whether it is at a bound; names
the benchmark, the error metric, and the backtest origin count and horizon; and lists the residual lags
tested with their degrees of freedom. A weak one names a model label and a few parameter values, omits `d`
and `D`, and validates with a single fit statistic.

**Exercise 8.** The facts are compatible because they describe different quantities. One-step in-sample
RMSE measures how well the model predicts the next observation from the previous one inside the window,
and the residual test asks only whether the leftover correlation is zero; both can be excellent when the
shape is right. Twelve-step error accumulates twelve forecasts, so a small level error is paid twelve
times, which is why the RMSE is 1.9 times larger and the mean error is the same sign in every period. The
residual test cannot detect this because the annual increment being a random walk is an assumption about
the future, not a property of past residuals. The repair is a level anchor — a drift or trend term, or a
model on a series that is not seasonally differenced twice — not a change of `p` or `q`.

## 14. Interview questions

**1. "What does differencing actually do?"** Strong: it removes a specific feature — a constant level, a
linear trend, a repeating seasonal shape — and the answer names which feature each difference removes and
what it costs in observations and noise. Weak: "it makes the series stationary", which is the purpose and
not the mechanism.

**2. "How do you know if you have over-differenced?"** Strong: the variance of the differenced series
rises instead of falls, the fitted MA coefficient approaches ±1, the forecast becomes a straight line with
an unbounded interval, and the model is non-invertible. Weak: "the ACF looks white", which is the symptom
of under-differencing as often as over-differencing.

**3. "ACF tails off and PACF cuts off at lag 2. What model?"** Strong: an AR(2) on the differenced series,
provided the series was already differenced and the sample is long enough for the band to mean something,
and offered as one candidate rather than a conclusion. Weak: "ARIMA(2,0,0)" delivered as a fact, without
asking what series the plots were computed on.

**4. "Can you compare AIC across models with different `d`?"** Strong: no, because changing `d` changes
the data and the observation count, so the likelihoods are for different datasets; `d` is chosen by
unit-root tests and the variance table, and AICc compares orders only at fixed `d` and `D`. Weak: yes, if
you use AICc, which is the correction most often offered by candidates who have memorised the formula and
not the condition.

**5. "What is the airline model and why is it called that?"** Strong: ARIMA(0,1,1)(0,1,1)12, from the
Box-Jenkins analysis of airline passenger series, two parameters, multiplicative in the sense that the
seasonal and non-seasonal filters multiply, and it assumes a fixed seasonal shape plus a random-walk
annual increment. Weak: "a seasonal ARIMA", which does not distinguish it from thousands of others.

**6. "You have a promotion calendar. How do you get it into the model?"** Strong: as an exogenous
regressor if the calendar is fixed before the forecast origin, with the "date known" test applied to every
driver; lagged if the effect is delayed; and with the future values supplied explicitly, because the model
cannot forecast its own inputs. Weak: "add it as a variable", without saying what happens at forecast
time.

**7. "What is leakage in a forecasting model?"** Strong: using at forecast time a value that was not
knowable then — a realised fill rate, an actual markdown, a reconstructed competitor stock-out — with the
mechanical test of the date the value becomes known. Weak: "using future data", which is correct and too
vague to act on.

**8. "Your residuals pass Ljung-Box and the forecast is bad. How?"** Strong: residual tests describe
one-step fit inside the window; they cannot test whether the annual increment is a random walk or whether
a level shift occurred after the window; the backtest at the decision horizon is the only test that
catches it. Weak: "the model must be overfitted", which is one possible cause among several and not the
most likely here.

**9. "Why would you use a damped exponential smoothing model instead of ARIMA?"** Strong: fewer
specification decisions, more robust to a wrong `d`, no invertibility problems, works on shorter series,
and the competition evidence shows the two are close in accuracy with ARIMA considerably more expensive.
Weak: "ARIMA is old-fashioned", which is an assertion about fashion, not evidence.

**10. "What would make you reject an automatically selected ARIMA model?"** Strong: an order at the search
boundary, a parameter at the invertibility boundary, a failure to beat seasonal naive on a rolling origin,
a residual spike at the seasonal lag, or a forecast whose level is outside the range of the recent data
for no stated reason. Weak: "if the AIC is high", which is a number without a scale.

## 15. Key terms

- **Stationarity** — constant mean, variance, and lag-dependent autocovariance.
- **Unit root** — a characteristic root of one; shocks persist and the level has no anchor.
- **Integration order (`d`)** — the number of differences needed for stationarity.
- **Seasonal differencing (`D`)** — differencing at lag `m` to remove a fixed seasonal shape.
- **Over-differencing** — differencing a stationary series; produces a non-invertible MA root.
- **Backshift operator (`B`)** — `B y_t = y_(t-1)`; a compact way to write filters.
- **AR(p)** — regression on the previous `p` values; momentum.
- **MA(q)** — regression on the previous `q` errors; shock decay.
- **ARIMA(p,d,q)** — ARMA on a series differenced `d` times.
- **SARIMA** — ARIMA with seasonal AR and MA terms at lag `m`.
- **SARIMAX** — SARIMA with exogenous regressors.
- **ACF** — autocorrelation function, plotted against lag.
- **PACF** — partial autocorrelation, net of intervening lags.
- **Correlogram** — the plot of an ACF or PACF with a confidence band.
- **White noise** — zero mean, constant variance, no autocorrelation.
- **Innovation** — the unpredictable part of the current observation.
- **Residual** — the observed one-step error of a fitted model.
- **Invertibility** — the model can be written as an infinite autoregression.
- **AIC / AICc / BIC** — fit penalised by parameter count; AICc adds a small-sample term.
- **Ljung-Box statistic** — portmanteau test for residual autocorrelation.
- **Exogenous regressor** — an input determined outside the model.
- **Leakage** — using information at forecast time that was not available then.
- **Airline model** — ARIMA(0,1,1)(0,1,1)`m`.
- **Drift** — a constant increment per period, equivalent to a deterministic trend.
- **MASE** — mean absolute scaled error, scaled by the seasonal-naive in-sample error.

## 16. Sources

**Peer-reviewed and academic**

- Box, G. E. P. and Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*.
  Holden-Day; 5th edition with Reinsel and Ljung, Wiley, 2015. The origin of the
  identification-estimation-diagnosis loop. Grade A.
- Makridakis, S. and Hibon, M. (2000). "The M3-Competition: Results, Conclusions and Implications."
  *International Journal of Forecasting*, 16(4). DOI: 10.1016/S0169-2070(00)00057-1. Grade A.
- Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020). "The M4 Competition: 100,000 Time
  Series and 61 Forecasting Methods." *International Journal of Forecasting*.
  DOI: 10.1016/j.ijforecast.2019.04.014. Grade A.
- Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2018). "Statistical and Machine Learning
  Forecasting Methods: Concerns and Ways Forward." *PLOS ONE*. PMC5870978. Grade A.
- Hyndman, R. J. and Khandakar, Y. (2008). "Automatic Time Series Forecasting: The Forecast Package
  for R." *Journal of Statistical Software*, 27(3). Grade A.
- Billah, B., Hyndman, R. J. and Koehler, A. B. (2005). "Empirical Information Criteria for Time
  Series Forecasting Model Selection." DOI: 10.1080/00949650410001687208. Grade A.
- Spiliotis, E., Petropoulos, F. and Assimakopoulos, V. (2023). "On the Disagreement of Forecasting
  Model Selection Criteria." *Forecasting*, 5(2). https://www.mdpi.com/2571-9394/5/2/27. Grade A.
- Dickey, D. A. and Pantula, S. G. (1987). "Determining the Order of Differencing in Autoregressive
  Processes." *Journal of Business and Economic Statistics*, 5(4).
  DOI: 10.1080/07350015.1987.10509614. Grade A.
- Crato, N. and de Lima, P. J. F. (1997). "On the Power of Underdifferencing and Overdifferencing
  Tests against Nearly Nonstationary Alternatives." *Communications in Statistics*.
  DOI: 10.1080/03610919708813448. Grade A.
- Plosser, C. I. and Schwert, G. W. "Estimation of a Noninvertible MA Process: The Case of
  Overdifferencing." Author-hosted copy: billschwert.com/overdif.htm. Grade A.

**Textbooks and reference**

- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition,
  OTexts, https://otexts.com/fpp3/. Sections on non-seasonal and seasonal ARIMA and on selecting
  predictors. Grade B.
- Hyndman, R. J. "Facts and fallacies of the AIC." https://fpp.robjhyndman.com/hyndsight/aic/.
  An author's blog post; consistent with the textbook, not peer-reviewed. Grade C.

**Standards and official**

- ISO 8601-1:2019 and ISO 8601-2:2019, *Date and time — Representations*. Grade B.
- ISO 3534-1:2016, *Statistics — Vocabulary and symbols — Part 1*. Grade B.
- ISO 5725 series, *Accuracy (trueness and precision) of measurement methods and results*. Grade B.
- ISO 8000 series, *Data quality*. Grade B.
- GS1 EPC Information Services (EPCIS) 2.0. Grade B.
- NIST/SEMATECH e-Handbook, section 6.6.2.4, "Model Validation". Grade B.
- Federal Reserve SR 11-7, *Guidance on Model Risk Management*, 2011. Grade B.

**Software documentation and maintainer statements**

- statsmodels, `SARIMAX` API reference and FAQ; documents that forecasts with `exog` are
  conditional on future input values. Grade D.
- statsmodels issue #9061, maintainer statement that out-of-sample forecasts require future
  `exog` values and provide no method for estimating them. Grade C.
- R `forecast` reference for `auto.arima`, documenting the Hyndman-Khandakar search and bounds.
  Grade D.

**Industry and secondary**

- Hyndman, R. J. Mcomp package, "Forecast comparisons", https://pkg.robjhyndman.com/Mcomp/.
  Recomputes M3 results with open-source implementations. Grade B.

**Note on citations.** Findings are reported as the cited sources state them and were not replicated.
Where a comparison is drawn from a secondary summary rather than the primary paper, the text says so. No
DOI has been supplied that was not present in the source consulted.

## 17. Links

**Inside this module**

- Module map and topic sequence: [`_index.md`](_index.md)
- The benchmark this model must beat: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md)
- The rival family, and the state-space view: [`02_exponential-smoothing.md`](02_exponential-smoothing.md)
- Planned siblings, not yet written: `04_regression-lags-fourier.md` (the same problem with
  calendar features), `05_hierarchy-aggregation.md` (where the series is defined),
  `06_intermittent-demand.md` (where ARIMA models the wrong object), `07_forecast-combination.md`
  (why averaging the two families usually wins), `08_rolling-origin-backtesting.md` (which decides
  the argument), `09_error_metrics-cost.md` (which error to report), `10_overrides-fva.md` (whether
  planner edits add value). All are `planned` in the module index.

**Earlier in the program**

- Time-series regression, the prerequisite for the exogenous section: [`04_regression-time-series.md`](../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md)
- Calendars and time zones, which decide `m`: [`04_calendars-weeks-time-zones.md`](../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Error metrics used throughout: [`09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
- Lead-time distributions the forecast feeds: [`11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Part I map: [`_index.md`](../../01_foundations/_index.md)

**Later and sideways**

- Hierarchical and probabilistic forecasting: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)
- Inventory optimisation, which consumes the forecast and its interval: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Module survey file: [`07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Return to the part map: [`02_methods/_index.md`](../_index.md)
