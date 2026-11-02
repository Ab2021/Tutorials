# Exponential Smoothing — Level, Trend, Seasonality, Damping, and the State-Space View

> Module 07 — Time Series Forecasting Foundations
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 01A topics 09–11; Module 04 topic 04; Module 05 topic 06; Module 07 topic 01
> Version 2026.5

---

## 1. Why this topic exists

Exponential smoothing is the workhorse of operational demand planning, and it is the method most
often used without being understood. Almost every enterprise resource planning (ERP) and advanced
planning system ships it as the default statistical engine. Almost none of them show the planner
the weights, the initialization, or the parameter values that produced the number on the screen.
The planner sees a forecast. The arithmetic that generated it is invisible.

That invisibility is the problem. The same software that produces a sensible flat forecast for a
stable fastener will, on a rising series, produce a forecast that lags the level by a constant
amount, and will keep lagging it for as long as the rise continues. It will, on a series with a
trend, extrapolate that trend linearly to the end of the horizon — through a product launch, past
the point where the market saturates, and into a number that nobody in the room believes. It
will, on a seasonal series, apply an additive seasonal pattern to a level that has tripled, and
under-forecast every peak. None of these failures throws an error. All of them produce a number.

The purpose of this topic is to make the arithmetic visible. By the end you should be able to
look at a fitted smoothing model and say what each parameter is doing, why the forecast has the
shape it has, what the prediction interval is assuming, and which of the assumptions is most
likely to be wrong for this series.

**Four examples of the failure, each invisible in the output.** A distributor forecasts an
industrial adhesive in cases per month on a series that has grown for three years, using simple
exponential smoothing (SES) at `alpha = 0.2`; every forecast lands about 50 units below actual, and
a planner adds a manual uplift of 50 units every month, logged as judgement. It is not judgement
but the algebraic bias of SES on a trending series, computable in advance. A seasonal item is
fitted with additive Holt-Winters while its volume grows 40%, so every peak is under-forecast and
every trough over-forecast, with alternating signs that make the aggregate error look small. A
trend model fitted at 18 units per month extrapolates 216 units of growth into a market with a
documented capacity below the result. An analyst reports plus or minus 40 units at every horizon
for ETS(A,A,N), assuming the horizon-one width holds, when the variance of a linear trend grows
roughly with the cube of the horizon.

The common thread is that the model's structure — which components exist, how they are weighted,
how they extrapolate — was never examined. This topic is about examining it.

**Why it sits here.** Module 07 topic 01 established that a forecast is meaningful only against a
benchmark. This topic shows that the naive method is not a rival to exponential smoothing but a
special case of it at a parameter boundary, and that fact reorganises how you read the family. The
topics that follow — ARIMA (topic 03), regression with calendar effects (topic 04), hierarchy
(topic 05), intermittent demand (topic 06) — all assume you can read a state-space model, because
the state-space form is the shared language.

**The claim of this topic.** Exponential smoothing is one idea — a recursively updated weighted
average — with three additions: trend, seasonality, and damping. The state-space formulation turns
the recursion into a statistical model with a likelihood, and the likelihood is what makes
estimation, selection, and prediction intervals possible. The rest is bookkeeping.

## 2. Learning outcomes

By the end of this topic you can:

1. **Derive** the weights of simple exponential smoothing from the recursion, show that they sum
   to one, and **convert** a smoothing constant into an equivalent moving-average window and a
   half-life.
2. **Explain** what the smoothing constant controls, what happens at each boundary of its range,
   and why the naive forecast is the case `alpha = 1`.
3. **Write** the Holt, damped-trend, and Holt-Winters recursions from memory, naming the unit of
   every term.
4. **Compute** the damped-trend forecast as a geometric sum, and **quantify** the gap between
   damped and undamped extrapolation at a stated horizon.
5. **Choose** between additive and multiplicative seasonality from a time plot and from the
   relationship between seasonal amplitude and level, and **state** the constraint each imposes
   on the data.
6. **Write** the state-space form of an ETS model as a measurement equation plus transition
   equations, **explain** why the error type changes the prediction interval but not the point
   forecast, and **read** an ETS(A,A,N) or ETS(M,N,N) label as a statement of commitments.
7. **Compute** the h-step forecast variance for ETS(A,N,N) and **explain** why the trend model's
   variance grows far faster and the damped model's does not.
8. **Diagnose** a fitted smoothing model from residuals, parameter values at their bounds, and
   forecast behaviour, and **critique** a forecast produced by an unseen model by asking the
   questions listed in section 5.

Each outcome maps to an exercise in section 12 and to a self-check item in section 13.

## 3. The core mental model

Hold one picture: a single weighted average, updated one observation at a time, with three
optional structures bolted onto it. Everything in the family is that picture with a different
subset of the bolts fitted.

```text
   ONE IDEA, THREE ADDITIONS

   level only
     l_t = alpha * y_t + (1 - alpha) * l_(t-1)
     forecast at any horizon = l_T                     ->  a flat line

   + trend  (b_t tracks the slope)
     forecast = l_T + h * b_T                          ->  a straight line, forever

   + damping  (multiply the slope by phi at each step)
     forecast = l_T + b_T * (phi + phi^2 + ... + phi^h) ->  a curve that flattens

   + seasonality  (s_t repeats every m periods)
     additive         forecast = l_T + h*b_T + s_(T+h-mk)
     multiplicative   forecast = (l_T + h*b_T) * s_(T+h-mk)
```

Read the diagram as an inventory of commitments, not a menu. Each addition is a claim about how
the future relates to the past, and each claim can be wrong in a specific and predictable way.

### 3.1 The weighted average, and what alpha controls

Simple exponential smoothing is a weighted average of every observation ever seen, with weights
`alpha * (1 - alpha)^j` for age `j`. They sum to one over an infinite history, which is why the
method is a weighted average rather than a weighted sum. The recursion computes that infinite sum
in constant memory: you never store the history, you carry one number.

**The recursion is exact, not an approximation.** Expanding `l_t = alpha*y_t + (1-alpha)*l_(t-1)`
repeatedly gives `l_t = alpha * sum over j of (1-alpha)^j * y_(t-j)`, plus a term in the initial
value that decays to nothing. The initial value is therefore a real modelling choice, not a
computational detail, and Step 5 treats it as one.

**What alpha controls.** It controls the *rate of forgetting*. A small alpha means a long memory
and a slow response; a large alpha means a short memory and a fast one. The trade-off is bias
against variance, the same one a moving average makes when you choose the window length.

```text
   equivalent moving-average window   N = 2/alpha - 1
   mean age of the data               (1 - alpha) / alpha   periods
   half-life                          h = ln(0.5) / ln(1 - alpha)   periods
```

| `alpha` | Window `N` | Half-life | Behaviour |
|---:|---:|---:|---|
| 0.05 | 39 | 13.5 | Very slow; stable level, low noise |
| 0.10 | 19 | 6.6 | Slow; the common "smooth" default |
| 0.20 | 9 | 3.1 | Moderate; a reasonable monthly starting point |
| 0.30 | 5.7 | 1.9 | Responsive |
| 0.50 | 3 | 1.0 | Fast; roughly a three-point average |
| 0.80 | 1.5 | 0.43 | Very fast; tracks noise as signal |
| 1.00 | 1 | 0 | No memory; the forecast is the last observation |

The last row is the important one. At `alpha = 1` the recursion collapses to `l_t = y_t`, so the
one-step forecast is the last observation, which is exactly the naive method from topic 01. **The
naive benchmark is not an alternative to exponential smoothing; it is exponential smoothing at
the boundary of its parameter space.** A model selected automatically at `alpha` near 1 has told
you the series is best forecast by a random walk, dressed in a smoothing costume.

### 3.2 Trend, and the danger of extrapolating it

A trend component makes the forecast a straight line from the last level, with slope `b_T`, which
says the rate of change observed at the end of the series will persist unchanged for every period
of the horizon. Three things make that fragile in operational demand. Trends are usually bounded: a
product saturates, a market matures, a promotion ends, a competitor enters. The estimated slope is
itself noisy, and the noise compounds linearly — an error of 2 units per month in the slope becomes
24 units at twelve months. And a trend fitted to a short window is often a segment of a longer
cycle, real for the segment and wrong for the horizon. The result is the characteristic failure of
trend extrapolation: forecasts plausible for two or three periods and absurd by period twelve.

### 3.3 Damping, and why it usually wins

Damping keeps the trend but makes it decay. Each step into the future multiplies the slope by a
factor `phi` between 0 and 1, so the increment at horizon `h` is `phi^h * b_T` rather than `b_T`.

```text
   EXTRAPOLATING A TREND: LINEAR (phi = 1.00) VERSUS DAMPED (phi = 0.90)

   forecast
   840 |                                          *  phi = 1.00
   800 |                                      *
   760 |                                  *       |
   720 |                              *      o    |  the gap at h = 12
   680 |                          *      o        |  is 99 units
   640 |                      *      o            |
   600 |  o---o---o---o---o---o---o                |
       +------------------------------------------> h (periods ahead)
          0   2   4   6   8   10  12  14  16  18

   *  undamped:  l_T + h * b_T                (a straight line)
   o  damped:   l_T + b_T * (phi + ... + phi^h)  (a curve with a ceiling)
```

The damped forecast converges: as `h` grows the sum of the multipliers approaches `phi/(1-phi)`,
so the forecast approaches `l_T + b_T * phi / (1 - phi)` and stops. With `phi = 0.90` the total
future trend contribution is at most nine times the current slope, however long the horizon. With
`phi = 1.00` there is no ceiling at all.

**Why this is the right default for operational demand.** Most demand series are not pure
exponentials; they rise, then flatten, and a prior that matches the world forecasts better than one
that does not. Gardner and McKenzie (1985, 1989) report that damping improved accuracy at longer
horizons with no loss of short-range accuracy, and avoided overshooting the seasonal pattern.

> **Note:** Damping is not universally better. Gardner and McKenzie (2010) argue that its success
> comes less from damping as such than from the family's flexibility: with estimated parameters the
> optimum frequently lands on a special case, a random walk at `phi = 0` or a linear trend at
> `phi = 1`. Estimate `phi`; do not impose it.

### 3.4 Seasonality, additive and multiplicative

A seasonal component says the pattern within the year repeats. The two forms differ in what is
assumed constant, and the diagnostic is visual: a constant absolute swing is additive, a swing
that grows with the level is multiplicative.

| Property | Additive | Multiplicative |
|---|---|---|
| Seasonal term is | A number of units, summed | A ratio, multiplied |
| Over a cycle | Deviations sum to zero | Indices average to one |
| Amplitude | Constant in units | Constant as a percentage of level |
| Data requirement | Any values | Strictly positive, level away from zero |

**The cost of getting it wrong is asymmetric.** An additive model on a multiplicative series
under-forecasts every peak and over-forecasts every trough, with the error growing with the level;
a multiplicative model on an additive series does the reverse and is worse at low levels.

### 3.5 The state-space view, and the taxonomy

The recursions above look like recipes. The state-space form turns them into a statistical model
by naming the error.

```text
   SINGLE SOURCE OF ERROR (innovations form)

   measurement:   y_t = l_(t-1) + eps_t          eps_t ~ N(0, sigma^2)
   transition:    l_t = l_(t-1) + alpha * eps_t

   One shock arrives each period. It moves the observation by eps_t and
   the level by alpha * eps_t; there is no second, independent error in
   the state equation. One likelihood follows, hence AIC, maximum
   likelihood estimates, and prediction intervals.
```

The phrase "single source of error" is the load-bearing part. Classical structural models carry a
separate disturbance in each state equation, which makes the likelihood hard to write. The
innovations form asserts that every movement comes from the one shock observed as the forecast
error. That restriction is what makes the family estimable by ordinary maximum likelihood, and it
is the technical reason the family became the default rather than a collection of recipes.

The taxonomy classifies methods by content — trend N, A, or Ad; season N, A, or M, giving nine
methods — and pairs each with an additive or multiplicative error, giving eighteen models.
`ETS(A,A,N)` means additive error, additive trend, no seasonality; section 5 gives the reading
procedure.

| Label | Common name | Parameters | Forecast shape | Interval width |
|---|---|---|---|---|
| ETS(A,N,N) | Simple exponential smoothing | `alpha`, `sigma` | Flat | Grows like `sqrt(h)` |
| ETS(A,A,N) | Holt's linear trend | `alpha`, `beta`, `sigma` | Straight line | Grows like `h^1.5` |
| ETS(A,Ad,N) | Damped trend | `alpha`, `beta`, `phi`, `sigma` | Curve with a ceiling | Converges to a bound |
| ETS(A,A,A) | Holt-Winters additive | `alpha`, `beta`, `gamma`, `sigma` | Line plus fixed seasonal offsets | Grows in steps |
| ETS(M,A,M) | Holt-Winters multiplicative | `alpha`, `beta`, `gamma`, `sigma` | Level times season | Fans with the level |

> **Note:** The three models with additive error and multiplicative season — ETS(A,N,M),
> ETS(A,A,M), ETS(A,Ad,M) — are excluded by most automatic selection routines for numerical
> instability. The exclusion is a convention of the implementation, not a statement that the
> models are meaningless.

**Why the error type changes intervals but not point forecasts.** With additive error
(`y_t = mu_t + eps_t`) the forecast variance depends on `sigma` and the parameters only. With
multiplicative error (`y_t = mu_t * (1 + eps_t)`) it scales with the square of the forecast level.
Both give the same point forecast, because the error has mean zero, and very different intervals.

If you remember one diagram, remember this one: a horizontal line, a straight line, a curve that
flattens, and a wave — the four forecast shapes — under a single recursive average whose memory is
set by `alpha`, driven by one error term that generates both the observation and the state update.

## 4. Definitions and vocabulary

The fourth column names the unit or coding system. A definition of a smoothing quantity without
one is the defect this module exists to prevent: `alpha` is dimensionless, `b_t` is in units per
period, and confusing them produces a forecast that is wrong by a factor of the period length.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Exponential smoothing | A family of forecast methods that update a weighted average recursively, with weights decaying geometrically | Used to mean SES only | Method family name |
| Smoothing constant `alpha` | The weight on the current observation in the level update, between 0 and 1 (0 to 2 in the admissible region) | Read as a percentage, or as "accuracy" | Dimensionless |
| Level `l_t` | The estimated local mean of the series at time `t`, after removing trend and season | Confused with the observed value `y_t` | Units |
| Trend `b_t` | The estimated local rate of change per period | Reported without "per period" | Units per period |
| Seasonal component `s_t` | The estimated repeating deviation or ratio for the season containing `t` | Reported without stating additive or multiplicative | Units (additive) or dimensionless ratio (multiplicative) |
| Seasonal period `m` | The number of observations in one full seasonal cycle | Set to 12 on weekly data | Periods per cycle |
| State vector | The set of quantities the model carries forward: level, trend, and `m` seasonal terms | Treated as unobservable and therefore unexamined | Mixed units |
| Simple exponential smoothing (SES) | The level-only member of the family, ETS(A,N,N) or ETS(M,N,N) | Assumed to handle trend if the data are scaled | Method name |
| Holt's linear trend | The level-plus-trend member, ETS(A,A,N) | Called "double exponential smoothing" without saying which double | Method name |
| Damped trend | A trend model in which the slope is multiplied by `phi` at each step into the future | Confused with a trend that is set to zero | Method name |
| Damping parameter `phi` | The per-step decay applied to the trend, `0 < phi <= 1` | Reported as a percentage per year when it is per period | Dimensionless |
| Holt-Winters | The level-plus-trend-plus-season member of the family | Used as a synonym for "seasonal smoothing" | Method name |
| Additive seasonality | A seasonal effect expressed as a constant number of units added to the level | Applied to data whose seasonal amplitude grows | Units |
| Multiplicative seasonality | A seasonal effect expressed as a ratio multiplied into the level | Applied to data containing zeros or negatives | Dimensionless ratio, centred on 1 |
| Forecast horizon `h` | The number of periods ahead being forecast | Stated without the period length | Periods |
| One-step error | `y_t` minus the forecast made at `t-1` | Called "residual" before the model is fitted | Units |
| Residual | The one-step error of a fitted model, used for diagnosis | Assumed independent when autocorrelated | Units |
| Innovations | The error process driving the state-space model, `eps_t` | Treated as a synonym for residual without checking the model form | Units, or relative units if multiplicative |
| State-space model | A representation with a measurement equation for the observation and transition equations for the state | Assumed to imply a Kalman filter is required | Model form |
| Single source of error | The assumption that one shock per period drives both observation and state | Confused with a single state variable | Model assumption |
| ETS taxonomy | The classification of models by error, trend, and seasonal type | Called a "model" rather than a family | Three-letter code |
| Error type | Whether the error enters additively (A) or multiplicatively (M) | Assumed to follow from the seasonality type | Letter A or M |
| Admissible parameter space | The region of parameter values under which the model is stable or forecastable | Confused with the narrower "usual" box | Inequality set |
| Initialization | The procedure that supplies the state at the start of the series | Treated as a computational detail | Method name |
| Initial state | The values of level, trend, and seasonal terms before the first observation | Omitted from model documentation | Mixed units |
| Heuristic initialization | Setting initial states from classical decomposition or a short regression | Used on short series where it dominates the fit | Method name |
| Estimated initialization | Estimating initial states jointly with the parameters by maximum likelihood | Used on very short series, where it overfits | Method name |
| Likelihood | The probability of the observed series as a function of the parameters | Reported without stating the error distribution | Function of parameters |
| AIC | An information criterion penalising fit by parameter count | Compared across models fitted to different windows | Dimensionless |
| Prediction interval | An interval for a future observation, incorporating process and parameter uncertainty | Called a "confidence interval" | Units |
| Forecast variance | The variance of the h-step forecast error | Assumed constant across horizons | Units squared |
| Equivalent window | The moving-average length with the same average data age, `2/alpha - 1` | Quoted as an exact equivalence of variance | Periods |
| Half-life | The number of periods after which an observation's weight halves | Confused with the equivalent window | Periods |
| Damped trend limit | The ceiling `l_T + b_T * phi/(1-phi)` that a damped forecast approaches | Reported as the forecast at a specific horizon | Units |

Six of these carry most of the operational damage. **`alpha` as a percentage:** it is a
dimensionless weight, not an accuracy setting, and a planner who "turns smoothing up to 90%" has
set the model to nearly ignore history. **`b_t` without "per period":** a monthly trend of 18 and
a weekly trend of 18 are different worlds. **Additive versus multiplicative seasonality stated
without a unit:** a seasonal index of 1.30 and a seasonal deviation of +390 units are the same
phenomenon expressed in forms that cannot be added. **`phi` per period versus per year:** `phi =
0.9` per month is `0.9^12 = 0.28` per year, which is a very different claim. **Residual versus
innovation:** the residual is what you observe; the innovation is what the model assumes, and
diagnostics test the assumption against the observation. **Prediction interval versus confidence
interval:** the first is a statement about a future observation and includes the process noise;
the second is a statement about a parameter and does not.

## 5. The mechanics, step by step

The procedure has four phases. Each step names what you do, what you check, and what goes wrong
if you skip the check.

### Phase A — Decide what the series asks for

**Step 1. Fix the bucket, the unit, and the horizon.** Write one sentence: "y_t is monthly demand
for item I at location L, in cases, for t = 1 to 36, forecast to h = 12." Check that the bucket is
constant, the unit does not change mid-series, and the horizon is in the same periods as the data.
Failure: a monthly model applied to a horizon stated in weeks, or a unit change at month 20 that
reads as a level shift.

**Step 2. Plot the series and classify its components.** Look for a level, a trend, a seasonal
pattern, and a variance that changes with the level. Do not fit yet. Ask three questions in order:
is there a trend that persists across the whole window, or only a segment; is there a seasonal
pattern at a period you can name; is the seasonal amplitude constant in units or in percentage.
Failure: fitting a trend that is one phase of a three-year cycle, which the model extrapolates.

**Step 3. Decide the error type before the method.** Ask whether the noise looks like a constant
band around the level (additive) or a band that widens with the level (multiplicative). This choice
affects intervals, not point forecasts, and it is easy to get wrong because the point forecast
looks fine either way. Failure: an additive-error model on a series whose residual standard
deviation triples over the window, producing intervals too wide early and far too narrow late.

**Step 4. Check the data constraints the model will impose.** Multiplicative seasonality requires
strictly positive data; check the minimum. A seasonal model requires at least two full cycles, and
preferably three, to separate the seasonal pattern from the trend; check the cycle count. Failure:
a twelve-month seasonal model fitted to eighteen months of data, where the seasonal indices and
the trend are not separately identified.

### Phase B — Initialize

**Step 5. Choose an initialization and state it.** *Heuristic* initialization sets the initial
level and trend from a short regression on the first few seasonally adjusted points, and the
initial seasonal terms from a classical decomposition. *Estimated* initialization treats the
initial states as free parameters and estimates them with the rest by maximum likelihood. Heuristic
is more stable on short series; estimated is better when the series is long enough to identify the
states. Failure: a model whose entire fit is determined by its first six observations.

**Step 6. Check what the initialization is worth.** The influence of the initial state decays as
`(1-alpha)^t`: at `alpha = 0.2`, after 12 periods the initial level retains `0.8^12 = 0.069`, about
7%; at `alpha = 0.05` it retains 54%. Short series therefore cannot separate initialization from
estimation, and software defaults differ invisibly. Hyndman, Koehler, Ord and Snyder (2008) estimate
the initial states as parameters, which is principled when the series is long enough; the heuristic
exists because joint estimation is unstable on short series. Failure: two analysts fit the same
model and get different forecasts because their software used different defaults.

**Step 7. Set starting values and note the bounds.** Most implementations start from a heuristic
grid and optimize within a box. Record the box: conventionally `0 < alpha < 1`, `0 < beta < alpha`,
`0 < gamma < 1 - alpha`, with `phi` in `[0.8, 0.98]`. The admissible region is wider — for
ETS(A,N,N), `0 < alpha < 2` — so the box can exclude a better fit. Failure: a parameter reported at
exactly 0.9999, which is the optimizer hitting a bound rather than finding an optimum.

### Phase C — Estimate and extrapolate

**Step 8. Estimate by maximum likelihood and record the converged values.** For additive-error
models, maximizing the likelihood is equivalent to minimizing the sum of squared one-step errors,
so the estimate is familiar. For multiplicative-error models the criterion differs and the
software should say so. Record every parameter with its value and whether it is interior or at a
bound. Failure: a parameter table with no bounds information, so a boundary solution is read as an
estimate.

**Step 9. Compute the point forecast and check its shape against the plot.** With `l_T`, `b_T`,
`s_T`, and `phi` in hand, write the forecast for `h = 1, 3, 6, 12`. Ask whether the shape is the
shape you would defend in a meeting. For a damped model, compute the ceiling `l_T + b_T*phi/(1-phi)`
and ask whether the ceiling is commercially plausible. Failure: a forecast that is defensible at
`h = 1` and indefensible at `h = 12`, with nobody looking at `h = 12`.

**Step 10. Compute the forecast variance and the interval at each horizon.** For ETS(A,N,N),
`sigma_h^2 = sigma^2 [1 + alpha^2(h-1)]`, so the half-width grows like the square root of the
horizon; for ETS(A,A,N) the variance is a cubic in `h` and the width grows roughly like `h^1.5`;
for ETS(A,Ad,N) both converge; for multiplicative-error models the variance carries a factor of the
squared forecast level. Section 6.6 gives the arithmetic. Failure: a constant interval width across
horizons, usually the horizon-one width copied down the column.

### Phase D — Diagnose, do not trust

**Step 11. Test the residuals as you would test any regression's.** Compute the residual
autocorrelation at lags 1 to `2m`. A correctly specified model leaves them approximately
uncorrelated; a spike at lag `m` means the seasonal component is missing or mis-specified; a slow
decay across many lags means the level is under-fitting. A persistent non-zero mean residual is
bias, and in a smoothing model it almost always means a missing component. Failure: a model
selected on AIC and never diagnosed, with a lag-12 autocorrelation of 0.4.

**Step 12. Test the parameters against their boundaries.** A `phi` at 1.00 means damping has
collapsed to a linear trend, an `alpha` at 1.00 means a random walk, a `gamma` at 0 means
seasonality is frozen. None is an error — each is the data saying the richer model was not
supported — but each must be reported, because it changes what the forecast means. Failure: a
"damped trend model" reported as such when `phi` is at its upper bound and damping does nothing.

**Step 13. Compare against the benchmarks from topic 01.** A smoothing model that cannot beat the
seasonal-naive method on a rolling-origin backtest has not earned its complexity, whatever its
AIC. Failure: a model selected by in-sample information criterion and never compared out of
sample.

### 5.1 Parameter boundaries and what happens at `alpha = 1`

The conventional box and the admissible region are different objects, and the difference explains
why two packages give different answers on the same data. Hyndman, Akram and Archibald (2008)
derive the admissible regions from the eigenvalues of the model's transition matrix.

```text
   CONVENTIONAL BOX                  ADMISSIBLE REGION (non-seasonal)
   0 < alpha < 1                      ETS(A,N,N):  0 < alpha < 2
   0 < beta  < alpha                  ETS(A,A,N):  0 < alpha < 2, 0 < beta < 4 - 2*alpha
   0 < gamma < 1 - alpha              ETS(A,Ad,N):  1 - 1/phi < alpha < 1 + 1/phi
   0.8 <= phi <= 0.98                              alpha*(phi-1) < beta < (1+phi)*(2-alpha)

   the box is strictly inside the region
```

**`alpha = 1` is not a degenerate case; it is the naive method.** The level, the one-step forecast
and every interval become the random walk's. If your fitted model has `alpha` at 1.00, the correct
description is "this series is best forecast by a random walk", not "the smoothing constant is
high".

**`alpha = 2` is admissible and pathological.** The level update becomes
`l_t = l_(t-1) + 2*eps_t`, which overshoots and oscillates: stable in the eigenvalue sense, and
forecasts no planner would accept. Stability is a mathematical property, not a quality judgement.

**Seasonal models need more than the box.** Hyndman, Akram and Archibald show that ETS(A,N,A),
ETS(A,A,A) and ETS(A,Ad,A) are never stable in the strict sense, because the transition matrix
always carries a unit eigenvalue; they remain forecastable if the relevant characteristic roots lie
inside the unit circle. In practice the seasonal terms are normalized to sum to zero, removing the
redundancy. The detail matters to an implementer, and as evidence that "the optimizer converged"
is a weaker statement than it looks.

## 6. Worked example

Halden Fasteners is a fictional regional distributor of industrial fasteners. It stocks the
HF-220 anchor bolt at one warehouse. All figures are constructed to be internally consistent and
checkable by hand.

### 6.1 The data

Monthly demand in cases, 24 months:

| Month | Demand | Month | Demand | Month | Demand |
|---|---:|---|---:|---|---:|
| 1 | 420 | 9 | 545 | 17 | 706 |
| 2 | 445 | 10 | 568 | 18 | 748 |
| 3 | 468 | 11 | 590 | 19 | 781 |
| 4 | 455 | 12 | 640 | 20 | 762 |
| 5 | 480 | 13 | 658 | 21 | 803 |
| 6 | 502 | 14 | 631 | 22 | 826 |
| 7 | 530 | 15 | 672 | 23 | 858 |
| 8 | 512 | 16 | 690 | 24 | 902 |

The series rises from about 420 to about 900 over two years. There is no visible seasonal pattern
at a period of 12; the within-year variation is irregular. The variance looks roughly constant in
absolute terms rather than growing with the level, which points to additive error.

**A first classification.** Level: present. Trend: present across the whole window, not a
segment. Seasonality: none visible. Error: additive. So the candidate models are SES, Holt, and
damped trend, and the question is whether the trend is real and whether it should be damped.

### 6.2 SES with `alpha = 0.20` and `alpha = 0.50`

The first twelve months, initialized at the first observation `l_0 = 420`.

| `t` | `y_t` | `l_(t-1)` | Forecast | Error | `l_t` with `alpha = 0.20` | `l_t` with `alpha = 0.50` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 420 | 420.000 | 420.000 | 0.000 | 420.000 | 420.000 |
| 2 | 445 | 420.000 | 420.000 | 25.000 | 425.000 | 432.500 |
| 3 | 468 | 425.000 | 425.000 | 43.000 | 433.600 | 450.250 |
| 4 | 455 | 433.600 | 433.600 | 21.400 | 437.880 | 452.625 |
| 5 | 480 | 437.880 | 437.880 | 42.120 | 446.304 | 466.313 |
| 6 | 502 | 446.304 | 446.304 | 55.696 | 457.443 | 484.156 |
| 7 | 530 | 457.443 | 457.443 | 72.557 | 471.955 | 507.078 |
| 8 | 512 | 471.955 | 471.955 | 40.045 | 479.964 | 509.539 |
| 9 | 545 | 479.964 | 479.964 | 65.036 | 492.971 | 527.270 |
| 10 | 568 | 492.971 | 492.971 | 75.029 | 507.977 | 547.635 |
| 11 | 590 | 507.977 | 507.977 | 82.023 | 524.381 | 568.817 |
| 12 | 640 | 524.381 | 524.381 | 115.619 | 547.505 | 604.409 |

**Results over `t = 2` to `12` (11 one-step forecasts):**

| Quantity | `alpha = 0.20` | `alpha = 0.50` |
|---|---:|---:|
| Sum of squared errors (units squared) | 44,630.7 | 15,834.9 |
| Root mean squared error (units) | 63.7 | 37.9 |
| Mean error (units) | +58.0 | +33.5 |

Two things are visible in the table and both matter.

**Every error is positive.** With `alpha = 0.20` all eleven one-step errors are positive, averaging
58 cases: the forecast is below the actual every period. That is not noise and not a data problem.
It is the algebraic consequence of a level-only model on a rising series — a weighted average of
the past is *always* below the present when the series is rising — and the expected bias is about
`b(1-alpha)/alpha`, which for `b = 18` and `alpha = 0.2` is 72 cases. The observed 58 is in that
region.

**Raising `alpha` halves the bias and cuts the error by 40%.** At `alpha = 0.50`, bias falls to
33.5 and RMSE to 37.9. The model is compensating for a missing component by forgetting faster: a
real improvement in one-step accuracy and entirely the wrong fix, since the forecast still lags a
rising series. **Add the trend component; do not raise the smoothing constant.** A planner who
keeps raising `alpha` to "make the forecast keep up" is correcting a specification error with a
tuning knob, and that is the most common parameter-tuning error in demand planning.

### 6.3 Holt's linear trend

With `alpha = 0.30`, `beta = 0.10`, initialized at `l_0 = 420`, `b_0 = 20` cases per month. The
starting slope is a guess based on the first few months; it will be revised.

| `t` | `y_t` | Forecast `l + b` | Error | `l_t` | `b_t` |
|---:|---:|---:|---:|---:|---:|
| 1 | 420 | 440.000 | -20.000 | 434.000 | 19.400 |
| 2 | 445 | 453.400 | -8.400 | 450.880 | 19.148 |
| 3 | 468 | 470.028 | -2.028 | 469.420 | 19.087 |
| 4 | 455 | 488.507 | -33.507 | 478.455 | 18.082 |
| 5 | 480 | 496.537 | -16.537 | 491.576 | 17.586 |
| 6 | 502 | 509.162 | -7.162 | 507.013 | 17.371 |
| 7 | 530 | 524.384 | 5.616 | 526.069 | 17.539 |
| 8 | 512 | 543.608 | -31.608 | 534.126 | 16.591 |
| 9 | 545 | 550.717 | -5.717 | 549.002 | 16.420 |
| 10 | 568 | 565.422 | 2.578 | 566.195 | 16.497 |
| 11 | 590 | 582.692 | 7.308 | 584.885 | 16.716 |
| 12 | 640 | 601.601 | 38.399 | 613.121 | 17.868 |

**Results over `t = 2` to `12`:**

| Quantity | SES `alpha = 0.20` | SES `alpha = 0.50` | Holt |
|---|---:|---:|---:|
| SSE (units squared) | 44,630.7 | 15,834.9 | 4,074.4 |
| RMSE (units) | 63.7 | 37.9 | 19.2 |
| Mean error (units) | +58.0 | +33.5 | -3.4 |
| Parameter count (incl. `sigma`) | 2 | 2 | 3 |

Holt reduces RMSE from 63.7 to 19.2 and, more importantly, the mean error from +58.0 to −3.4. The
bias is gone — the signature of a correctly specified model, which shows in errors with no
systematic sign, not merely smaller ones. Note the pattern: months 1 to 6 have mostly negative
errors, because the model started with a slope of 20 when the true early slope was nearer 18, and
months 7 to 12 have mostly positive errors, because the slope accelerated. The early-start penalty
is the initialization, and it is doing more damage than the parameters.

### 6.4 Extrapolating the trend: damped and undamped

At `t = 12` the state is `l_12 = 613.121` and `b_12 = 17.868` cases per month. Two forecasts
follow from the same state: `l_12 + h*b_12` for the undamped case, and
`l_12 + b_12*(phi + phi^2 + ... + phi^h)` for the damped case with `phi = 0.90`.

| Horizon `h` | Undamped | Damped `phi = 0.90` | Difference | Difference as % of damped |
|---:|---:|---:|---:|---:|
| 1 | 630.99 | 629.20 | 1.79 | 0.3% |
| 3 | 666.72 | 656.70 | 10.02 | 1.5% |
| 6 | 720.33 | 688.46 | 31.87 | 4.6% |
| 12 | 827.54 | 728.51 | 99.03 | 13.6% |
| limit | unbounded | 773.93 | — | — |

**The gap is negligible at `h = 1` and material at `h = 12`.** This is why damping is nearly free
when the horizon is short and why it is the single most valuable structural choice when the
horizon is long. It also explains why the empirical literature found damping improved long-horizon
accuracy "with no loss of short-range accuracy": at short horizons the two models are numerically
almost identical.

**The ceiling is the number to take to the meeting.** The undamped model has no answer to "what
does this look like in year three?"; the damped model says the series approaches 774 cases and
stops, which is a claim that can be argued with.

### 6.5 Additive and multiplicative seasonality on a second item

The second item, the HF-118 hex cap screw, is seasonal. Quarterly demand in cases, three years:

| Quarter | 1 | 2 | 3 | 4 | Year total | Year mean |
|---|---:|---:|---:|---:|---:|---:|
| Year 1 | 1,200 | 960 | 1,080 | 1,560 | 4,800 | 1,200 |
| Year 2 | 1,300 | 1,040 | 1,170 | 1,690 | 5,200 | 1,300 |
| Year 3 | 1,400 | 1,120 | 1,260 | 1,820 | 5,600 | 1,400 |

**Multiplicative.** Dividing each quarter by its year mean gives 1.000, 0.800, 0.900 and 1.300 in
every year, summing to 4.000. The series is exactly multiplicative, with a 30% peak and a 20%
trough.

**Additive.** The deviations are 0, −240, −120, +360 in year 1 and 0, −280, −140, +420 in year 3.
They are not constant: the Q4 deviation rises 17% over two years, tracking the 17% rise in level.
An additive model estimated over the whole window would use +390, too large for year 1 and too
small for year 3, leaving negative residuals in early peaks and positive ones in late peaks — a
pattern that looks like "the market changed" and is the model averaging away a known feature.

**At the next level.** With the year-4 level at 1,500 cases per quarter, the multiplicative
forecast for Q4 is 1,950 and the additive 1,890: a 60-case gap, 3.1% of the peak, small in
percentage terms and large operationally, because the peak is where capacity and safety stock are
decided.

> **Note:** Multiplicative seasonality requires strictly positive data and a level that never
> approaches zero. On a series with zero-demand periods the multiplicative form is undefined and
> additive is the only option. When the diagnostic is ambiguous, fit both and compare
> out-of-sample error.

### 6.6 Prediction intervals, and why the trend model's are enormous

Take the SES fit at `alpha = 0.50`, whose residual standard deviation is about 38 units; use 40.

| `h` | ETS(A,N,N) `sigma_h` | 95% half-width | Half-width as % of level 604 |
|---:|---:|---:|---:|
| 1 | 40.0 | 78.4 | 13.0% |
| 3 | 49.0 | 96.0 | 15.9% |
| 6 | 60.0 | 117.6 | 19.5% |
| 12 | 77.5 | 151.8 | 25.1% |

The width grows like the square root of the horizon: at `h = 12` it is `sqrt(3.75) = 1.94` times
the `h = 1` width. That is the arithmetic of a level-only model.

For ETS(A,A,N) with `sigma = 20`, `alpha = 0.30` and `beta = 0.05`, the bracket
`1 + (h-1)(alpha^2 + alpha*beta*h + beta^2*h(2h-1)/6)` gives:

| `h` | Bracket | `sigma_h` (units) | 95% half-width |
|---:|---:|---:|---:|
| 1 | 1.000 | 20.0 | 39.2 |
| 6 | 2.038 | 28.6 | 56.0 |
| 12 | 5.235 | 45.8 | 89.7 |
| 24 | 22.16 | 94.2 | 184.5 |

At `h = 24` the trend model's standard deviation is 94 units against the level model's 35 — a
factor of 2.7 from the same order of residual noise. The extra width is the slope's uncertainty
compounding over the horizon. A trend model that looks well fitted will still produce useless
intervals at long horizons; that is a property of the model, not a defect of the data.

With multiplicative error (`sigma = 0.08` relative, level 604) the interval is a percentage of the
forecast: `sigma_h` is 48.3 units at `h = 1`, 72.5 at `h = 6`, and 93.6 at `h = 12`, so it fans
with the level. Additive error says the uncertainty is a fixed number of units; multiplicative
error says it is a fixed percentage.

## 7. Code, query, or calculation

One by-hand recursion, written as a table rather than as library code, because the purpose is to
make the arithmetic checkable. The values reproduce section 6.2 and can be verified line by line
with a calculator.

```text
   SES recursion, alpha = 0.20, l_0 = 420.0 units
   ---------------------------------------------------------------
   t    y_t    l_(t-1)    forecast    error      l_t
   ---------------------------------------------------------------
   1    420    420.000     420.000     0.000    420.000
   2    445    420.000     420.000    25.000    425.000
   3    468    425.000     425.000    43.000    433.600
   4    455    433.600     433.600    21.400    437.880
   5    480    437.880     437.880    42.120    446.304
   6    502    446.304     446.304    55.696    457.443
   7    530    457.443     457.443    72.557    471.955
   8    512    471.955     471.955    40.045    479.964
   9    545    479.964     479.964    65.036    492.971
   10   568    492.971     492.971    75.029    507.977
   11   590    507.977     507.977    82.023    524.381
   12   640    524.381     524.381   115.619    547.505
   ---------------------------------------------------------------
   SSE (t = 2..12) = 44,630.7   RMSE = 63.7 units   mean error = +58.0
```

Three checks before trusting any such recursion, by hand or in software.

- **The level must lie between the previous level and the observation.** `l_t` is a weighted
  average of the two, so it cannot fall outside the interval they define. A level outside that
  interval means `alpha` is outside `[0, 1]` or the update has been written with the weights
  swapped. This single check catches most transcription errors.
- **The errors must sum to a finite number that does not shrink as you add terms.** If the mean
  error tends to zero as the series lengthens on a visibly trending series, you are looking at a
  model that has a trend component. On a level-only model with a trend, the mean error converges
  to the bias, not to zero.
- **At `alpha = 1` the table must collapse to the naive method.** Set `alpha = 1` and confirm that
  every `l_t` equals `y_t` and every forecast equals the previous observation. If it does not,
  the implementation is doing something other than what the documentation says.

## 8. Data, standards, and regulatory context

No standard mandates an exponential smoothing model, and none specifies a smoothing constant. That
absence is itself the regulatory context: because the method is unregulated, the burden of
documentation falls entirely on the analyst, and the standards that matter are the ones that fix
the vocabulary and the time axis.

### 8.1 What the data must contain

Before a smoothing model can be fitted, the dataset must answer six questions.

| Question | Why it matters | What to record |
|---|---|---|
| Is the observation demand, shipment, or order? | Shipments are censored by stock-outs; orders contain batching | Source system and field name |
| What is the time bucket, and is it fixed? | Every rate, trend, and seasonal index scales with the bucket | Bucket length and boundary convention |
| Is the unit constant across the window? | A unit change at month 20 reads as a level shift | Unit of measure and any conversion |
| Are zero periods recorded as rows? | Missing zeros inflate the level and corrupt the seasonal index | Row count per period and the zero rule |
| How many full seasonal cycles are present? | Fewer than two makes seasonality and trend inseparable | Cycle count |
| Is the series censored by stock-outs? | Low service truncates demand and biases the level downward | Service level over the window |

The first and last are the most consequential. A stock-out truncates sales at available inventory,
so the model fits *fulfilled* demand, not demand, and the level is pulled down exactly where demand
was highest. The under-forecast is then worst where service was worst — a self-reinforcing loop
that looks like a demand problem and is a data problem.

### 8.2 Standards that fix the vocabulary and the clock

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 3534-1:2016 | Statistics vocabulary and symbols | Meanings of expectation, variance, prediction, and interval | B |
| ISO 8601-1:2019 and ISO 8601-2:2019 | Date and time representation | What a period is, hence what a seasonal period `m` means | B |
| ISO 5725 series | Accuracy, trueness, and precision of measurement | Separates measurement error from process variation | B |
| JCGM 100:2008 (GUM) | Expression of uncertainty in measurement | Defines the coverage factor and interval language | B |
| GS1 EPCIS 2.0 | Event data for objects in a supply chain | Occurrence timestamps from which the series is built | B |

**ISO 3534-1** settles the vocabulary collision that matters most here: a prediction interval is a
statement about a future observation, and a confidence interval a statement about a parameter.
Software often labels the first as the second. **JCGM 100:2008** supplies the coverage factor `k`
for a stated level, numerically the same object as the multiplier on `sigma_h`, valid under the
same assumptions and failing in the same ways.

### 8.3 The retail calendar is a modelling constraint, not a formatting detail

Many retailers and consumer goods companies plan on a 4-4-5 calendar: four weeks, four weeks,
five weeks, repeating, with the year divided into thirteen four-week periods. On that calendar,
"month 12" is not December and "period 1" is not January. A seasonal model with `m = 12` fitted to
4-4-5 periods is estimating seasonal indices for periods that drift against the Gregorian calendar
by one or two days a year and by a week every five or six years.

| Calendar | Period definition | Effect on `m = 12` seasonality | Grade |
|---|---|---|---|
| Gregorian month | Calendar month, 28–31 days | Period length varies 10%, injecting noise into the level | B |
| 4-4-5 retail | 4, 4, 5 weeks repeating | Period length varies 25%, and periods drift against the calendar | C |
| 13-period | 13 equal four-week periods plus a residual week | `m = 13`, not 12 | C |
| Weekly | ISO week, Monday start | `m = 52` or `m = 53`; the 53-week year breaks a fixed `m` | B |

The practical consequence: a seasonal smoothing model fitted on a 4-4-5 calendar should use
`m = 13` if the periods are the observation unit, or the series should be converted to a Gregorian
month before fitting with `m = 12`. Fitting `m = 12` to thirteen periods per year produces a
seasonal component that is a mixture of the true pattern and an aliasing artifact, and no amount
of parameter tuning repairs it.

> **Note:** The 4-4-5 calendar is an industry convention maintained by the National Retail
> Federation, not a statutory standard. Grade C. It is near-universal in North American retail
> reporting and largely absent in European manufacturing, so a model transferred between the two
> may be using the wrong seasonal period without anyone noticing.

### 8.4 Model risk governance

In regulated financial institutions, model risk management frameworks — the Federal Reserve's
SR 11-7 is the reference document — require documented conceptual soundness, documented
development, ongoing monitoring, and outcomes analysis. A forecasting model that sizes inventory or
provisions for obsolescence is a model in that sense if it feeds a reported number.

A model with an undocumented initialization, an unstated error type, and parameters reported
without bounds would not pass such a review. The remedy is not more documentation for its own
sake; it is the checks in section 5, written down. Grade B for the regulatory guidance; the
application to demand forecasting is an inference, not a requirement.

## 9. Industry practice and case evidence

### 9.1 What the forecasting competitions found

The Makridakis competitions are the closest thing the field has to a controlled trial, and their
results on exponential smoothing are consistent across three decades.

| Competition | Design | Result relevant to this topic | Grade |
|---|---|---|---|
| M3 (Makridakis and Hibon, 2000) | 3,003 series, 24 methods | Theta best overall; damped exponential smoothing among the strongest classical methods; automatic ETS competitive with commercial software | A |
| M4 (Makridakis, Spiliotis and Assimakopoulos, 2020) | 100,000 series, 61 methods | Winner was a hybrid of exponential smoothing and a recurrent neural network; the baseline `comb` benchmark was SES + Holt + damped, and few methods beat it | A |
| Makridakis et al. (2018), PLOS ONE | 1,045 M3 monthly series | Statistical methods dominated machine learning at all horizons; ETS was the most accurate statistical method | A |

Three conclusions follow, each contradicting a common belief. **The classical family is not
obsolete:** in M4 the benchmark to beat was three exponential smoothing models combined, and most
of the sixty-one entrants did not beat it. **Combination beats selection more often than not:** the
`comb` benchmark is not clever, it is an average of three, and that it is hard to beat is the
empirical case for topic 07. **Automatic selection is competitive, not optimal:** automatic ETS
performed well in M3, which is evidence that the procedure is sound, not that the selected model
is the best possible for a given series — a distinction routinely conflated in vendor material.

### 9.2 The damped-trend evidence, and why it took time to be adopted

Gardner and McKenzie (1985) introduced the damped trend and reported, on Makridakis's 1,001
series, that damping improved long-horizon accuracy with no loss of short-range accuracy. Gardner
and McKenzie (1989) extended it to seasonal models, reporting that damping avoided overshooting the
data and amplifying the seasonal pattern beyond anything observed.

**Why was it resisted?** Three mechanisms, and only the first is documented. Damping adds `phi`,
which must be estimated for every series — a real cost on a portfolio of tens of thousands of
stock-keeping units (SKUs), and more so in the 1980s and 1990s. Near `phi = 1` the likelihood is
nearly flat when the data do not discriminate, so the estimate flips between damped and linear on a
small change of data, and practitioners reasonably concluded the extra parameter was not earning
its place (a statistical observation, not a cited finding). And a planning convention resisted
decay: a trend in a business plan is often a commitment, so a model that makes the trend decay
tells the organisation its growth assumption is temporary. That last mechanism is inference.

Gardner and McKenzie (2010) offer a fourth explanation that is more interesting than any of these:
the method works because it is *flexible* rather than because damping is right. When parameters are
estimated, the optimum frequently lands on a special case — a random walk at one end or a linear
trend at the other. In the M3 data, they report that true damped-trend parameters were selected only
about 43% of the time, the most common special case being a damped drift. **The practical reading is
not "always damp" but "always estimate `phi` and report where it landed".**

### 9.3 What enterprise systems actually do

Planning modules in major ERP and advanced planning suites expose `alpha`, `beta`, and `gamma` as
per-item parameters, usually with defaults in the 0.1 to 0.3 range, and usually with the
initialization, the error type, and the parameter bounds hidden from the planner. Product
documentation reviewed for this course in 2026 confirms the parameters are exposed and the
specification is not. Grade D: vendor documentation, not evidence of behaviour.

The spreadsheet pattern is equally common and more dangerous, because it is invisible to
governance:

```text
   = alpha * actual_t + (1 - alpha) * previous_forecast_t
```

This is a level-only update with no trend and no seasonality, applied to whatever series happens
to be in the column, with `alpha` typed as a constant in a cell. It is the single most common
forecasting model in the world. It is also, on any series with a trend, a model with a known and
computable bias that nobody computes. Grade E as evidence; grade A as a description of practice.

### 9.4 What is missing from the public record

Company-level evidence that isolates the smoothing specification is thin: disclosures rarely state
the model, the parameters, the initialization, or the error type, and vendor cases lack a baseline.
The direction is clear — damped trend usually helps at long horizons, multiplicative seasonality
helps when the amplitude grows, and additive error understates uncertainty on a rising series — but
the magnitude is setting-dependent. Settle it on your own series, not by citation.

## 10. Failure modes and red flags

Each row names the failure, then a red flag that is visible in a meeting or in a dataset — not a
diagnostic you would have to run, but something you can see across the table.

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Level-only model on a trending series | Every one-step error has the same sign, month after month | A weighted average of the past lags a rising present by `b(1-alpha)/alpha` | Persistent under-forecast; planners add manual uplift | Check the sign pattern of residuals; add a trend component |
| Trend extrapolated without damping | "The forecast reaches 1,300 by month 18 and we have capacity for 900" | A linear trend asserts the last slope persists forever | Absurd long-horizon numbers; loss of credibility | Fit `phi`; report the damped ceiling; cap the horizon |
| `alpha` tuned upward to fix a specification error | Planner repeatedly raises smoothing "to make it keep up" | Higher `alpha` compensates for a missing component by forgetting faster | Noise tracked as signal; forecast variance up | Fix the specification; then tune |
| Additive seasonality on a growing series | Seasonal residuals alternate in sign by quarter and grow with the level | A constant seasonal deviation cannot represent a proportional swing | Peak under-forecast, trough over-forecast, worse each year | Plot the seasonal band; switch to multiplicative |
| Multiplicative seasonality on data with zeros | Zeros in the history, or a series that was differenced | Division by a seasonal index near zero is undefined | Fit fails, or produces enormous indices | Use additive, or model occurrence separately |
| Seasonal model with fewer than two cycles | "We have fourteen months, so we set the period to twelve" | Seasonal indices and trend are not separately identified | Arbitrary seasonal terms; forecast depends on initialization | Wait for cycles, or fix the seasonal indices from a related item |
| Parameter at a bound reported as an estimate | `alpha = 0.9999`, `phi = 1.0000`, `gamma = 0.0000` | The optimizer stopped at the edge of the box | The model is a degenerate special case, described as the full model | Report the bound; describe the model it collapsed to |
| Flat prediction interval across horizons | The same plus-or-minus appears at `h = 1` and `h = 12` | `sigma_h` grows with `h` in every ETS model | Interval is too narrow where the decision is made | Compute `sigma_h` per horizon; plot the fan |
| Initialization undocumented and dominant | Two analysts, same data, different forecasts | Initial state weight is `(1-alpha)^n`, large when `alpha` is small | Irreproducible forecasts; disputes about whose model is right | State the initialization; report the residual weight of `l_0` |
| Wrong seasonal period for the calendar | `m = 12` on a 4-4-5 calendar or on weekly data | Period length does not match the seasonal cycle | Seasonal component is an aliasing artifact | Match `m` to the calendar; convert the bucket |
| Demand series censored by stock-outs | Service level below target in the same periods the level estimate dips | Sales are truncated at available inventory | Level biased down; forecast under-states demand; loop reinforces | Use unconstrained demand or lost-sales data; flag the window |
| Damped model reported when `phi = 1` | The write-up says "damped trend"; the parameter table says 1.00 | Damping at the boundary is no damping at all | Stakeholders believe the extrapolation is bounded when it is not | Check `phi` against its bound before describing the model |
| Model selected on in-sample AIC only | No rolling-origin comparison in the pack | AIC rewards fit within the estimation window | Model that fits history and loses to naive out of sample | Require a backtest against seasonal-naive |

Two deserve a note. **The sign pattern of residuals is the cheapest diagnostic in forecasting:**
in the worked example every one-step error under SES is positive, visible in a column of numbers
without a statistic. **The bound-at-an-estimate failure is a communication failure more than a
technical one:** a `phi` of 1.00 is not a mistake, it is the data saying damping is not supported;
the mistake is describing the model as damped in the slide pack, and the organisational fix is
harder than the technical one because "damped" sounds more sophisticated than "linear trend".

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| One error per period drives observation and state | The single-source-of-error form gives the model its likelihood | Likelihood is mis-specified; intervals and AIC are wrong | Compare against a multi-error structural model on a long series |
| The error is homoscedastic (additive) or scales with the level (multiplicative) | Forecast variance depends on the error form | Intervals too wide early or too narrow late | Plot residual standard deviation against the fitted level |
| Errors are uncorrelated | The likelihood treats them as independent | Standard errors and intervals understate uncertainty | Residual autocorrelation at lags 1 to `2m` |
| Errors are normally distributed | The interval multiplier assumes it | Symmetric intervals on a skewed error distribution | Histogram and normal probability plot of residuals |
| The seasonal period is constant | The seasonal index repeats at a fixed `m` | `m` is wrong; seasonal component aliases | Check the calendar and the periodogram |
| The level, trend, and season are identified | The model needs enough cycles and variation | Parameters trade off; estimates unstable | Cycle count; parameter correlation; stability across windows |
| The initialization does not dominate | The fit describes the series, not its first few points | Forecast is a statement about the initial state | Compute `(1-alpha)^n`; refit with a different initialization |
| The trend is either persistent or geometrically damped | The forecast shape follows from this | Neither shape matches; forecast wrong at long horizons | Compare damped and undamped out of sample |
| Parameters are constant over the window | One parameter set describes the whole series | Model describes an average of regimes | Split-window comparison of parameters |
| The series is not censored by stock-outs | The fitted level is demand, not fulfilled demand | Level biased down exactly where demand is highest | Compare against unconstrained orders or lost-sales records |
| The bucket is fixed and `m` matches it | Rates and seasonal indices scale with the bucket | Every parameter is off by the bucket ratio | Recomputed at two buckets; compare |

Three of these are limits rather than assumptions. **Additive-error models cannot produce a
widening fan:** if the residual standard deviation grows with the level, the intervals are too
narrow exactly where the level and the decision are largest, and the fix is the multiplicative
form, not a wider multiplier. **A damped trend is still a trend:** it converges to a ceiling that
may be wrong, so if the market genuinely doubles the model under-forecasts the second half. **No
smoothing model handles a structural break:** at `alpha = 0.2`, 90% of the weight is on the last
eleven periods, so a break takes about a year to work through, and the fix is an intervention
variable or a shorter window, not a different smoothing constant.

## 12. Exercises

### Exercise 1 — Derive the weights (calculation, 25 minutes)

Expand the SES recursion three times and show that the weight on `y_(t-3)` is `alpha(1-alpha)^3`.
Then compute the weights for ages 0 to 6 at `alpha = 0.25`, the cumulative weight on the last four
observations, and the equivalent moving-average window.

**Deliverable:** the expansion, a weight table, and the equivalent window. **Acceptance
criteria:** the weights sum to 1 minus the residual on the initial value; the equivalent window is
computed as `2/alpha - 1`; every weight is labelled dimensionless. **Time:** 25 minutes.

### Exercise 2 — Parameter boundaries (analysis, 30 minutes)

For each fitted model below, say what the parameter value means, whether it is at a bound, what
special case the model has collapsed to, and how you would describe it in a slide.

(a) ETS(A,N,N) with `alpha = 1.00`. (b) ETS(A,Ad,N) with `phi = 1.00`. (c) ETS(A,A,A) with
`gamma = 0.00`. (d) ETS(A,N,N) with `alpha = 0.001`. (e) ETS(A,Ad,N) with `phi = 0.00`.

**Deliverable:** a five-row table with the special case and the one-line description for each.
**Acceptance criteria:** each answer names the method the model has become; at least one answer
identifies a model that is not a special case but a degenerate boundary. **Time:** 30 minutes.

### Exercise 3 — Damped against undamped (calculation, 40 minutes)

A series ends with `l_T = 480` units and `b_T = 25` units per month. Compute the forecast at `h =
1, 3, 6, 12, 24` under `phi = 1.00`, `phi = 0.95`, and `phi = 0.85`. Compute the ceiling for each
damped case. State at which horizon the undamped and damped forecasts differ by more than 10%.

**Deliverable:** a table of forecasts and ceilings, plus the crossover horizons.
**Acceptance criteria:** the geometric sum is shown explicitly for at least one horizon; the
ceiling is computed as `b_T phi/(1-phi)`; the crossover is identified by interpolation or by
inspection. **Time:** 40 minutes.

### Exercise 4 — Additive or multiplicative (analysis, 35 minutes)

Quarterly demand for four years is 800, 640, 720, 1,040; 880, 704, 792, 1,144; 968, 774, 871,
1,259; 1,065, 852, 958, 1,385.

Compute the seasonal indices both ways, state which form the data support, and quantify the error
an additive model would make at the peak in year 4.

**Deliverable:** two seasonal tables, a one-sentence diagnosis, and the peak error with its sign.
**Acceptance criteria:** indices sum to `m` or deviations sum to zero as appropriate; the
diagnosis cites the constancy of one form against the drift of the other; the peak error is in
units and in per cent. **Time:** 35 minutes.

### Exercise 5 — Interval shape (calculation, 40 minutes)

For ETS(A,N,N) with `sigma = 25` units and `alpha = 0.4`, compute `sigma_h` and the 95% half-width
at `h = 1, 3, 6, 12`. Then compute the same for ETS(A,A,N) with `sigma = 25`, `alpha = 0.3`,
`beta = 0.05`, and state the ratio of the two at `h = 12`.

**Deliverable:** two tables and the ratio. **Acceptance criteria:** the level-model formula is
applied correctly; the trend-model bracket term is computed term by term; the ratio is interpreted
in one sentence. **Time:** 40 minutes.

### Exercise 6 — Diagnose the residuals (analysis, 30 minutes)

A fitted ETS(A,A,A) model has residual autocorrelations of 0.42 at lag 1, 0.08 at lag 2, 0.05 at
lag 3, and 0.51 at lag 4, on quarterly data. The mean residual is +12 units on a level of 900.
Propose the three most likely specification problems and the test that would distinguish them.

**Deliverable:** three hypotheses with a discriminating test for each. **Acceptance criteria:**
each hypothesis names a specific component; the lag-4 spike is addressed explicitly; the mean
residual is interpreted as bias rather than noise. **Time:** 30 minutes.

### Exercise 7 — Write the specification (design, 45 minutes)

Write a one-page model specification for an item you know, or for the HF-220 bolt, in the form a
reviewer could reproduce.

**Deliverable:** bucket, unit, horizon, model label, parameters with units, initialization method,
estimation window, parameter bounds, benchmark, backtest design, and diagnosis plan.
**Acceptance criteria:** a second analyst could refit from the document alone; every parameter has
a unit; the benchmark and the error metric are named; the diagnosis plan specifies the lags to
test. **Time:** 45 minutes.

### Exercise 8 — Critique a vendor claim (critique, 30 minutes)

A vendor states: "Our AI engine automatically selects the optimal exponential smoothing model for
every item and improves forecast accuracy by 30%."

Write a one-page critique that separates the claim into testable parts.

**Deliverable:** the decomposition, the missing baseline, the missing specification details, and
four questions for the vendor. **Acceptance criteria:** at least one part is identified as
unfalsifiable as stated; the critique distinguishes selection accuracy from model accuracy; the
questions are specific enough that a vendor could answer them. **Time:** 30 minutes.

## 13. Self-check answers

### Exercise 1 — the weights

Expanding the recursion twice gives `l_t = alpha*y_t + alpha(1-alpha)*y_(t-1) +
alpha(1-alpha)^2*y_(t-2) + (1-alpha)^3*l_(t-3)`, so the weight on `y_(t-3)` is
`alpha(1-alpha)^3`.

At `alpha = 0.25` the weights for ages 0 to 6 are 0.2500, 0.1875, 0.1406, 0.1055, 0.0791,
0.0593, 0.0445; the last four carry 68.4% of the weight. The equivalent window is
`2/0.25 - 1 = 7` periods, the mean age `(1-0.25)/0.25 = 3.0` periods, and the half-life
`ln(0.5)/ln(0.75) = 2.41` periods. The finite sum is `1 - (1-alpha)^n`, which at `n = 7` gives
0.8665, leaving 13.4% of the weight on the initial value.

### Exercise 2 — what each boundary means

| Case | What the model has become | How to describe it |
|---|---|---|
| (a) ETS(A,N,N), `alpha = 1.00` | The naive method; level equals the last observation | "The model selected a random walk" |
| (b) ETS(A,Ad,N), `phi = 1.00` | Holt's linear trend; damping does nothing | "The trend was not damped by the data" |
| (c) ETS(A,A,A), `gamma = 0.00` | Seasonality frozen at its initial estimates | "Seasonality fixed at the initial decomposition" |
| (d) ETS(A,N,N), `alpha = 0.001` | Level frozen; forecast is nearly the initial value | "The level is not being updated" |
| (e) ETS(A,Ad,N), `phi = 0.00` | SES with a redundant trend parameter | "Over-parameterised SES; refit as ETS(A,N,N)" |

(d) retains `0.999^n` of `l_0` after `n` periods, which is 97% after 30 periods. (e) is the only
genuine degeneration in the list: the forecast has no trend, but `b_t` is still estimated and
still enters the state equation as `phi*b_(t-1) = 0`. (c) is legitimate when the seasonal pattern
is contractually fixed and suspect when it is drifting.

### Exercise 3 arithmetic

| `h` | Undamped, `phi = 1.00` | `phi = 0.95`, `S_h = 19(1-0.95^h)` | `phi = 0.85`, `S_h = 5.6667(1-0.85^h)` |
|---:|---:|---:|---:|
| 1 | 505.00 | 503.75 | 501.25 |
| 3 | 555.00 | 547.70 | 534.58 |
| 6 | 630.00 | 601.23 | 571.33 |
| 12 | 780.00 | 695.43 | 601.65 |
| 24 | 1,080.00 | 820.95 | 618.73 |
| ceiling | none | 955.00 | 621.67 |

The undamped and `phi = 0.95` forecasts differ by more than 10% first at `h = 12`: 780.00 against
695.43 is a 10.9% gap. For `phi = 0.85` the gap reaches 9.3% at `h = 6` and 22.9% at `h = 12`.
The `phi = 0.85` column rises only 17 units from `h = 12` to `h = 24`, against 199 in the undamped
column: that is what a ceiling looks like in a forecast table.

### Exercise 4 arithmetic

| Year | Total | Mean | Multiplicative indices | Additive deviations |
|---|---:|---:|---|---|
| 1 | 3,200 | 800 | 1.000 0.800 0.900 1.300 | 0, −160, −80, +240 |
| 2 | 3,520 | 880 | 1.000 0.800 0.900 1.300 | 0, −176, −88, +264 |
| 3 | 3,872 | 968 | 1.000 0.800 0.900 1.300 | 0, −194, −97, +291 |
| 4 | 4,260 | 1,065 | 1.000 0.800 0.900 1.300 | 0, −213, −107, +320 |

The indices are constant; the deviations more than double from year 1 to year 4. The data are
exactly multiplicative.

The average additive peak deviation is `(240+264+291+320)/4 = 278.75` units. Against the year-4
level of 1,065 an additive model would forecast 1,343.75 against an actual of 1,385 — an
under-forecast of 41.25 units, or 3.0%. The error grows each year by the gap between the true
deviation and the average.

### Exercise 5 arithmetic

Level model: `sigma_h^2 = 625[1 + 0.16(h-1)]`. Trend model: `sigma_h^2 = 625 * B_h`, where
`B_h = 1 + (h-1)[0.09 + 0.015h + 0.0004167h(2h-1)]`.

| `h` | Level `sigma_h` | Level half-width | `B_h` | Trend `sigma_h` | Trend half-width |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.00 | 49.0 | 1.000 | 25.00 | 49.0 |
| 3 | 28.72 | 56.3 | 1.2825 | 28.31 | 55.5 |
| 6 | 33.54 | 65.7 | 2.0375 | 35.68 | 69.9 |
| 12 | 41.53 | 81.4 | 5.235 | 57.20 | 112.1 |

The trend model's interval is 38% wider at `h = 12` from the same residual scale, and the gap
grows with the horizon. The extra width is the sampling error of the slope, and it is not
reducible by a better fit — it is the price of asserting a trend.

### Exercise 6 — three hypotheses and their tests

1. **The seasonal period or phase is wrong.** A lag-4 autocorrelation of 0.51 on quarterly data is
   exactly one cycle, which is what a mis-specified seasonal component leaves behind. Test: refit
   with the seasonal terms fixed from a classical decomposition and compare the lag-4 value.
2. **`gamma` is too small and the pattern is drifting.** A nearly frozen component leaves a lag-`m`
   spike that grows across the window. Test: compute the lag-4 autocorrelation on the first and
   second halves separately. Small early and large late means the pattern moved.
3. **The +12 mean residual is a missing level shift.** That is 1.3% bias on a level of 900, small
   in percentage terms and material if concentrated. Test: plot the cumulative sum of residuals. A
   step at a known event — a price change, a new customer — is a structural break.

The lag-1 correlation of 0.42 is a fourth issue, addressed by re-estimating `alpha`. The three
hypotheses above are ranked by the size of the diagnostic they explain; lag 4 is the largest.

### Exercise 7 rubric

A strong specification names the bucket, unit, and horizon; states the ETS label including the
error type; lists every parameter with its value, unit, and whether it is at a bound; states the
initialization method and estimation window; names the benchmark and error metric; specifies a
rolling-origin backtest with origin count and horizon; and lists the residual lags to be tested.
A weak one names a method and a few values, omits the error type and initialization, and validates
with a single fit statistic.

### Exercise 8 rubric

A strong critique separates four claims — that selection is automatic, that it is optimal, that
accuracy improves by 30%, and that the improvement is attributable to the engine — then notes
that "optimal" needs a stated criterion and validation design, that 30% has no baseline and is
unfalsifiable as stated, that in-sample selection differs from out-of-sample accuracy, and that
per-item selection on short histories overfits the selection. It asks for the baseline, the error
metric, the backtest design, and one representative selected model. A weak critique says the
vendor is probably exaggerating without naming what would falsify the claim.

## 14. Interview questions

**1. "What does the smoothing constant actually control?"**

A strong answer says the rate of forgetting — the geometric decay of weight on past observations —
and gives the equivalent window `2/alpha - 1` and a half-life as concrete translations, noting that
`alpha = 1` is the naive method. A weak answer says it weights recent data more heavily and stops.

**2. "Why does SES produce biased forecasts on a trending series?"**

A strong answer derives the bias: the level is a weighted average of the past, so on a rising
series it is always below the present by about `b(1-alpha)/alpha`, and raising `alpha` reduces the
bias while remaining the wrong fix, because the missing component is a trend. A weak answer says
the model lags, without quantifying it or naming the fix.

**3. "What is a damped trend and why would you use one?"**

A strong answer describes the geometric decay of the slope, gives the ceiling `l_T + b_T
phi/(1-phi)`, explains that most demand series flatten rather than grow exponentially, and notes
that damping is nearly free at short horizons and material at long ones. A weak answer says it
stops the forecast running away, which is the symptom and not the mechanism.

**4. "How do you choose between additive and multiplicative seasonality?"**

A strong answer describes the diagnostic — a constant-width band versus a fanning band — states
that multiplicative requires strictly positive data, and notes that when the diagnostic is
ambiguous you fit both and compare out of sample. A weak answer says multiplicative is for
percentages.

**5. "What does ETS(A,A,N) mean?"**

A strong answer decodes it as additive error, additive trend, no seasonality, names it as Holt's
linear trend, and states what each letter commits to: the error type fixes the interval shape, the
trend type the forecast shape, the seasonal type the parameter count. A weak answer says
Holt-Winters, which is wrong on the seasonal letter.

**6. "Why do prediction intervals widen with the horizon?"**

A strong answer gives `sigma_h^2 = sigma^2[1 + alpha^2(h-1)]` for the level model, notes that the
trend model's variance is cubic in `h` and that the extra width is the slope's uncertainty, and
notes that a damped model's variance converges. A weak answer says forecasts get less certain
further out, which is true and explains nothing.

**7. "What happens if your fitted `alpha` is 1.0?"**

A strong answer says the model has collapsed to the naive method, that the level is the last
observation, that the correct description is "a random walk was selected", and that this is
information rather than an error. A weak answer says the model is over-fitting.

**8. "How does initialization affect the forecast?"**

A strong answer gives the decay `(1-alpha)^n`, notes that the effect is large when `alpha` is
small or the series is short, states that different software defaults produce different forecasts
on the same data, and recommends reporting the residual weight of the initial value. A weak
answer says initialization washes out, which holds only for long series with large `alpha`.

**9. "You are handed a forecast from a model you did not build. What do you ask?"**

A strong answer walks the checks: label and error type, parameter values and bounds,
initialization, the forecast shape at three horizons, the damped ceiling, the interval width per
horizon, and residual autocorrelation. A weak answer asks for the accuracy, which is one number
and does not describe the model.

**10. "When would you not use exponential smoothing at all?"**

A strong answer names the cases: intermittent demand, where the level is mostly zero and the
method models the wrong object; strong calendar structure better handled by regression with
Fourier terms; known causal drivers a univariate model cannot use; and histories short enough that
initialization dominates. A weak answer says never, or says machine learning is better, which the
M4 results do not support at this data scale.

## 15. Key terms

- **Exponential smoothing** — a family of recursive weighted-average forecast methods.
- **Smoothing constant (`alpha`)** — the dimensionless weight on the current observation.
- **Level (`l_t`)** — the estimated local mean, in units.
- **Trend (`b_t`)** — the estimated local slope, in units per period.
- **Seasonal component (`s_t`)** — the repeating deviation in units, or ratio, dimensionless.
- **Seasonal period (`m`)** — the number of observations in one cycle.
- **Simple exponential smoothing (SES)** — the level-only member, ETS(A,N,N).
- **Holt's linear trend** — level plus trend, ETS(A,A,N).
- **Damped trend** — a trend whose slope decays by `phi` per step, ETS(A,Ad,N).
- **Damping parameter (`phi`)** — the per-step decay applied to the trend, dimensionless.
- **Holt-Winters** — the level, trend, and seasonal member of the family.
- **Additive seasonality** — a constant seasonal deviation, in units.
- **Multiplicative seasonality** — a constant seasonal ratio, dimensionless.
- **Equivalent window** — the moving-average length with the same data age, `2/alpha - 1`.
- **Half-life** — the periods after which an observation's weight halves.
- **State-space model** — a measurement equation plus transition equations.
- **Single source of error** — one shock per period drives observation and state.
- **Innovations** — the error process driving the state-space model.
- **ETS taxonomy** — the classification by error, trend, and seasonal type.
- **Admissible parameter space** — the region in which the model is stable or forecastable.
- **Heuristic initialization** — initial states from decomposition or a short regression.
- **Estimated initialization** — initial states estimated jointly with the parameters.
- **Prediction interval** — an interval for a future observation, in units.
- **Forecast variance** — the variance of the h-step forecast error, in units squared.

## 16. Sources

**Peer-reviewed and academic**

- Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill. Grade A.
- Holt, C. C. (1957). "Forecasting Trends and Seasonals by Exponentially Weighted Averages."
  Office of Naval Research Memorandum No. 52; reprinted in *International Journal of Forecasting*,
  20(1), 2004. Grade A.
- Winters, P. R. (1960). "Forecasting Sales by Exponentially Weighted Moving Averages."
  *Management Science*, 6(3), 324–342. DOI: 10.1287/mnsc.6.3.324. Grade A.
- Gardner, E. S. and McKenzie, E. (1985). "Forecasting Trends in Time Series." *Management
  Science*, 31(10), 1237–1246. DOI: 10.1287/mnsc.31.10.1237. Grade A.
- Gardner, E. S. and McKenzie, E. (1989). "Seasonal Exponential Smoothing with Damped Trends."
  *Management Science*, 35(3), 372–376. DOI: 10.1287/mnsc.35.3.372. Grade A.
- McKenzie, E. and Gardner, E. S. (2010). "Damped Trend Exponential Smoothing: A Modelling
  Viewpoint." *International Journal of Forecasting*, 26(4), 661–665.
  DOI: 10.1016/j.ijforecast.2009.07.001. Grade A.
- Gardner, E. S. and McKenzie, E. (2010). "Why the Damped Trend Works." *Journal of the
  Operational Research Society*. DOI: 10.1057/jors.2010.37. Grade A.
- Hyndman, R. J., Koehler, A. B., Snyder, R. D. and Grose, S. (2002). "A State Space Framework for
  Automatic Forecasting Using Exponential Smoothing Methods." *International Journal of
  Forecasting*, 18(3), 439–454. Grade A.
- Hyndman, R. J., Koehler, A. B., Ord, J. K. and Snyder, R. D. (2005). "Prediction Intervals for
  Exponential Smoothing Using Two New Classes of State Space Models." *Journal of Forecasting*.
  DOI: 10.1002/for.938. Grade A.
- Hyndman, R. J., Koehler, A. B., Ord, J. K. and Snyder, R. D. (2008). *Forecasting with
  Exponential Smoothing: The State Space Approach*. Springer. Grade A.
- Hyndman, R. J., Akram, M. and Archibald, B. C. (2008). "The Admissible Parameter Space for
  Exponential Smoothing Models." *Annals of the Institute of Statistical Mathematics*, 60, 407–426.
  Grade A.
- Makridakis, S. and Hibon, M. (2000). "The M3-Competition: Results, Conclusions and Implications."
  *International Journal of Forecasting*, 16(4). DOI: 10.1016/S0169-2070(00)00057-1. Grade A.
- Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020). "The M4 Competition: 100,000 Time
  Series and 61 Forecasting Methods." *International Journal of Forecasting*.
  DOI: 10.1016/j.ijforecast.2019.04.014. Grade A.
- Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2018). "Statistical and Machine Learning
  Forecasting Methods: Concerns and Ways Forward." *PLOS ONE*. PMC5870978. Grade A.

**Textbooks and reference**

- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition,
  OTexts, chapter 8. https://otexts.com/fpp3/. Grade B.

**Standards and official**

- ISO 3534-1:2016, *Statistics — Vocabulary and symbols — Part 1*. Grade B.
- ISO 8601-1:2019 and ISO 8601-2:2019, *Date and time — Representations for information
  interchange*. Grade B.
- ISO 5725 series, *Accuracy (trueness and precision) of measurement methods and results*.
  Grade B.
- JCGM 100:2008, *Guide to the expression of uncertainty in measurement*. Grade B.
- GS1 EPC Information Services (EPCIS) 2.0. Grade B.
- Board of Governors of the Federal Reserve System, SR 11-7, *Guidance on Model Risk Management*,
  2011. Grade B.

**Industry and vendor**

- Minitab Support, "Additive and multiplicative models" in time series. Useful for the diagnostic,
  not evidence of accuracy. Grade D.
- Enterprise planning module documentation reviewed in 2026: `alpha`, `beta`, and `gamma` exposed
  per item; initialization, error type, and parameter bounds not exposed. Grade D.
- Vendor case studies claiming accuracy gains from automated model selection, without a stated
  baseline or validation design. Grade D.

**Press and anecdote**

- Stack Exchange discussion of the `alpha = 2/(N+1)` equivalence with a moving average; a useful
  derivation check, not a citation. Grade E.
- Wikipedia, "Exponential smoothing", for the historical attribution of Brown, Holt and Winters;
  consistent with the primary sources above. Grade E.

**Note on citations.** Findings are reported as the cited sources state them and were not
replicated. Unconfirmed DOIs are omitted rather than guessed. The application of SR 11-7 to demand
forecasting is an inference, labelled as such in section 8.4.

## 17. Links

**Inside this module**

- Module map and topic sequence: [`_index.md`](_index.md)
- Previous topic, the benchmarks this family must beat: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md)
- Planned siblings, not yet written: `03_arima-sarimax.md` (the Box-Jenkins route to the same
  series), `04_regression-lags-fourier.md` (calendar effects as an alternative to a seasonal
  component), `05_hierarchy-aggregation.md` (where the level is forecast), `06_intermittent-demand.md`
  (where smoothing models the wrong object), `07_forecast-combination.md` (the case for averaging),
  `08_rolling-origin-backtesting.md` (which decides whether any model earns its place),
  `09_error_metrics-cost.md` (metric choice as a cost decision), `10_overrides-fva.md` (whether
  planner edits help). All are `planned` in the module index.

**Earlier in the program**

- Error metrics used before fitting any model: [`09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
- The lead-time distributions these forecasts feed: [`11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Unit-of-measure discipline every parameter depends on: [`01_unit-of-measure-hierarchy.md`](../../01_foundations/01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md)
- Calendars, weeks, and time zones, which decide `m`: [`04_calendars-weeks-time-zones.md`](../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Time-series regression, the prerequisite for the state-space view: [`04_regression-time-series.md`](../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md)
- Power and minimum detectable effect, the discipline behind a backtest claim: [`06_power-mde.md`](../../01_foundations/05_experimentation_pilots_evidence/06_power-mde.md)
- Part I map: [`_index.md`](../../01_foundations/_index.md)

**Later and sideways**

- Hierarchical and probabilistic forecasting, where intervals become the product: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)
- Inventory optimisation, which consumes the forecast and the interval: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Module survey file: [`07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Part II map, the part this module sits in: [`_index.md`](../_index.md)
- Return to the part map: [`02_methods/_index.md`](../_index.md)
