# Regression with Lags, Fourier Terms, and Exogenous Variables

> Module 07 — Time Series Forecasting Foundations
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 04–05; Module 05 topic 06; Module 07 topics 01–03
> Version 2026.5

---

## 1. Why this topic exists

A demand planner does not ask "what will demand be". The planner asks "what will demand be if we hold
price at 2.49 and run the Easter promotion". The moment the question contains an *if*, a pure
time-series model cannot answer it, because a pure time-series model has no vocabulary for the thing
being varied. A regression does. That is why regression remains the workhorse of commercial
forecasting long after ARIMA and exponential smoothing have been automated: it is the only common
forecaster that accepts a cause.

The cost of that capability is a discipline that no other forecasting method requires. A regression
forecast is not one prediction. It is a prediction of the outcome *conditional on a path for every
regressor*, and the path has to be produced by someone, from something, at some point in time. Most
of the professional difficulty in this topic lives in that sentence and not in the estimation.

Three failures illustrate the shape of the problem.

1. **The model that used the price it was trying to explain.** A dairy forecasts weekly volume for a
   chilled beverage. The design matrix includes average selling price, taken from the sales
   warehouse. Average selling price is a *realised* quantity: it falls when unplanned markdowns are
   triggered by weak sell-through. The regression estimates a strong negative price elasticity and
   fits beautifully. In the following quarter, a cost-driven list-price increase is announced and the
   model predicts a volume collapse that does not happen, because the historical correlation mixed
   the effect of price on demand with the effect of demand on price. The coefficient was not wrong
   about the data. It was wrong about the direction of causation, and the forecast used it as if it
   were a lever.
2. **The backtest that halved its error by reading the future.** A category team backtests a
   promotional model using actual temperature for the holdout weeks. The reported error is excellent.
   In production, the same model runs on a 10-day weather forecast, and the error roughly doubles at
   horizons beyond two weeks. Nothing about the model changed. The backtest was answering "how well
   can I predict demand if I already know the weather", which is not the question the business is
   paying for.
3. **The inventory regressor that contained the outcome.** An analyst adds ending inventory to a
   model of weekly sales, reasoning that low stock suppresses sales. The fit improves sharply. The
   improvement is mechanical: ending inventory is computed as opening inventory plus receipts minus
   sales, so it is a deterministic function of the target. The model has been given the answer in
   disguised form. This is target leakage, and it is the single most common reason a supply chain
   model validates well and fails on contact with a live planning cycle.

None of these is an estimation error. All three are errors in the *information set*: what the model
was allowed to know, and when it was allowed to know it.

**The claim of this topic.** A regression forecast is a claim about three things at once: the shape of
the calendar, the causal window of each driver, and the availability of every input at the forecast
origin. Get the calendar wrong and seasonality absorbs the holidays. Get the window wrong and the
promotion effect is the peak rather than the total. Get availability wrong and the model is a
narrative about the past dressed as a forecast of the future. The first two are estimation problems
and Module 04 topic 04 treats them formally. The third is a production problem, and it is what this
topic adds.

**Where this sits.** Topic 01 gave you the benchmarks that any regression must beat. Topic 02 gave you
the state-space view of trend and seasonality. Topic 03 gave you the ARIMA error structure that a
regression can borrow. This topic builds the design matrix, projects it forward, and states the error
honestly. Topic 08 turns the backtest into a protocol. Topic 09 turns the error into a cost.

> **Note:** The distinction that governs everything below is between a *conditional* forecast and an
> *unconditional* one. A conditional forecast says "if temperature follows this path, volume will be
> 1,840 units". An unconditional forecast says "volume will be 1,840 units, with a distribution that
> includes the chance that temperature does not follow that path". Planners need the second. Most
> regression outputs supply the first and label it the second.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** every candidate regressor into one of four availability types — known exactly, known
   with revision, forecastable, and simultaneous — and **state** for each how it enters a forecast
   and what error it contributes.
2. **Build** a regressor availability table for a real forecasting problem, recording the event date,
   the knowledge date, the revision policy, and the owner of the future path.
3. **Construct** a Fourier seasonal basis for a stated period and **choose** the harmonic order by an
   information criterion, stating the parameter cost and the equivalence point with seasonal dummies.
4. **Distinguish** a fixed holiday, a moving holiday, and a moving *window*, and **build** a proximity
   regressor for a holiday or school term whose dates are known but whose position in the calendar
   bucket varies.
5. **Estimate** a finite distributed lag, report the cumulative effect with its own standard error,
   and **explain** why the peak lag is not the effect.
6. **Project** each regressor to the horizon using one of three declared paths — contracted, external
   forecast, or own model — and **propagate** the path uncertainty into the forecast interval.
7. **Audit** a regression backtest for the six leakage channels and **state** which one each finding
   belongs to.
8. **Report** a forecast with its conditional and unconditional error separated, and **name** the
   three assumptions that would invalidate it.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a regression forecaster is a machine with two inputs, and only one of them is
usually discussed.

```text
   AT THE ORIGIN t0                          AT THE HORIZON t0 + h
   ----------------                          ---------------------
   observed history                          unknown outcome
   y(1..t0), x(1..t0)                        y(t0 + h)

        |                                          ^
        |  estimate                                |  score
        v                                          |
   +----------------+       apply      +---------------------------+
   |   model f(x)   |----------------->|   f( x(t0+h | t0) )       |
   +----------------+                  +---------------------------+
                                               ^
                                               |
                                      regressor path
                                      x(t0+1 .. t0+h | t0)

                                      known?  forecast?  assumed?
                                      -----------------------------
                                      this is the second input,
                                      and it is somebody's job
```

The first input is the model. The second is the path. A forecast regression is only as honest as its
description of the second input, and most forecast documentation describes only the first.

### 3.1 Three clocks, one number

Every data point has three dates, and the whole leakage catalogue is a failure to keep them apart.

```text
   value
     ^
     |                                      *  revised value
     |                        *  published value
     |  *  event
     +--------------------------------------------------> time
        t_event        t_pub        t_rev

   t_event   the period the number describes
   t_pub     the date it first became visible in a system
   t_rev     the date it was restated

   An as-of line at the forecast origin may use only values published
   before it. A backtest that reads today's table uses t_rev, always,
   and therefore uses the future.
```

A forecast made on 20 April may use March's event. It may not use March's revision published in June.
The distinction sounds pedantic until you notice that almost every warehouse table stores only the
latest revision, with no record of what was known when. In that architecture the honest backtest is
not merely difficult. It is impossible without reconstructing vintages, and most teams do not.

### 3.2 The four availability types

The single most useful classification in this topic. Every regressor belongs to exactly one box, and
the box determines the method, the error contribution, and the conversation you have with the owner.

| Type | Examples | How it enters the forecast | Error it contributes |
|---|---|---|---|
| Known exactly | Day of week, Easter date, contracted price, frozen promotion calendar, published school term dates, contractual payment terms | Direct path, no uncertainty | Zero, unless the plan itself changes |
| Known with revision | Weather actuals, competitor price scrapes, macro indices, shipment history, footfall counts | Use the vintage, never the latest value; model or bound the revision | Revision variance, often correlated with the level |
| Forecastable | Temperature, category demand, footfall, exchange rate, housing starts | Plug in an external forecast or your own model, with its error | Forecast error of the driver, amplified by the coefficient |
| Simultaneous | Competitor reaction, own reactive price, service level, availability | Cannot be used at face value; instrument it, or treat it as a scenario | Bias, not variance, if ignored |

**The first box is where the value is.** Calendar and contract are the only inputs you can know
exactly at any horizon, which is why calendar effects are the highest-return feature engineering in
supply chain forecasting and also the most over-claimed: a model that "captures seasonality" has
usually captured the day of week and the public holidays, which is real value and not the same thing
as understanding demand.

**The fourth box is where the arguments are.** A simultaneous regressor is not useless for
forecasting; it is useless for *intervention*. If you never change the price rule, a regression on
realised price can predict acceptably, because it estimates a conditional expectation that is stable
under the historical policy. The moment you ask "what if we raise price by 5%", the coefficient is
answering a different question. Section 5 Phase D step 22 states the test that separates the two uses.

### 3.3 Why the reported error is usually too small

The residual standard deviation from a regression is the error *conditional on the regressor path*.
The business experiences the error *unconditional on it*, because the path is itself uncertain. In
the linear Gaussian case the two are related by a single term:

```text
   sigma^2_unconditional  =  sigma^2_model  +  beta' Sigma_x beta

   sigma^2_model   residual variance from the fit
   Sigma_x         covariance of the regressor path forecast errors
   beta            the coefficient vector

   At h = 1 with a known calendar, Sigma_x is nearly zero and the two agree.
   At h = 13 with a forecast temperature path, the second term can dominate.
```

This is not a technicality: it determines whether an interval is 6% wide or 15% wide at the horizon
the planner cares about, and why an interval calibrated on one-step residuals under-covers at
multi-step horizons.

> **Note:** When the regressors are stochastic, the forecast error distribution is not normal even if
> everything else is Gaussian. The classical reference is Feldstein (1971) on the forecast error of
> econometric models with stochastic forecast-period exogenous variables; McCullough (1996) shows the
> conventional predictor's interval is inconsistent and proposes a bootstrap. For practical work,
> simulate the regressor paths rather than adding variances by hand.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A definition without one is the defect this
module exists to prevent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Forecast origin | The point in time from which a forecast is made and whose information set bounds it | Confused with the first forecast period | Date, with an explicit cut-off time |
| Forecast horizon | Number of periods between the origin and the target period | Reported as a date range instead of a step count | Integer steps `h` |
| Information set | Everything knowable at the origin, including publication lags | Assumed to be "the dataset" | Set of series with as-of timestamps |
| Exogenous variable | A regressor whose value is not determined by the outcome within the model | Called exogenous because it is external to the company | Series with a stated generating process |
| Regressor path | The sequence of regressor values used for `h = 1 ... H` | Left implicit | Matrix of `H` rows by number of regressors |
| Conditional forecast | A forecast that holds the regressor path fixed | Reported as if unconditional | Outcome units, given a stated path |
| Unconditional forecast | A forecast integrating over the distribution of the regressor path | Called "the forecast" without stating the integration | Outcome units with a full distribution |
| Dynamic regression | A regression whose errors follow an ARIMA process | Used to mean any regression with lags | Model with both `x` terms and error dynamics |
| ARIMAX | ARIMA with exogenous regressors, in the sense of regression with ARMA errors | Assumed to deliver causal coefficients | Model; coefficients conditional on the error specification |
| ARDL | Autoregressive distributed lag: the outcome depends on its own lags and on current and lagged regressors | Used interchangeably with ARIMAX | Model with `p` own lags and `q` regressor lags |
| Distributed lag | A specification in which the outcome depends on current and past values of a regressor | Reported as a single contemporaneous coefficient | Coefficients indexed by lag |
| Lag weight | The coefficient on one lag of a regressor | Read as the total effect | Outcome units per regressor unit |
| Cumulative effect | Sum of lag weights from zero to a stated horizon | Computed by adding standard errors | Outcome units per regressor unit |
| Long-run multiplier | In an ARDL, `sum(beta_j) / (1 - sum(phi_i))` | Reported as `sum(beta_j)` | Outcome units per regressor unit |
| Fourier term | A sine or cosine at an integer harmonic of the fundamental frequency | Called a seasonal dummy | Dimensionless, in `[-1, 1]` |
| Harmonic order K | Number of sine-cosine pairs included | Reported without the period it applies to | Integer; `2K` columns |
| Seasonal period m | Observations in one seasonal cycle | Assumed to be 12 regardless of frequency | Observations per cycle |
| Amplitude | Peak deviation of a harmonic from its mean, `sqrt(a^2 + b^2)` | Read as the coefficient on the sine term | Outcome units |
| Phase | Horizontal shift of a harmonic, `atan2(b, a)` | Ignored, so the peak week is misread | Radians; convert to periods |
| Nyquist limit | Highest identifiable frequency, one cycle per two observations | Exceeded by including `K > m/2` | Cycles per observation |
| Seasonal dummy | A 0/1 indicator for one position in the seasonal cycle | Used with `m` levels and an intercept | 0 or 1 |
| Moving holiday | A holiday whose calendar date changes year to year | Treated as a fixed date | Event date per year and jurisdiction |
| Holiday window | Span of days relative to an event over which its effect acts | Set to one week by default | Days before and after the event |
| Proximity regressor | The share of a holiday or term window falling in a period | Confused with a dummy | Proportion in `[0, 1]` |
| Calendar effect | Any variation attributable to the arrangement of days rather than to demand | Used to mean seasonality | Effect per period, in outcome units |
| Trading-day effect | Variation from the number and mix of days in a period | Ignored in weekly data, where it still applies at boundaries | Days, or a day-count contrast |
| Leakage | Use of information not available at the forecast origin | Used to mean "overfitting" | Binary defect; no unit |
| Target leakage | Leakage in which a regressor is a function of the outcome | Missed because the regressor looks like a driver | Binary defect |
| Data vintage | The version of a series as it stood at a particular date | Ignored; the latest value is used everywhere | Series plus as-of date |
| Publication lag | Delay between the event period and first availability | Confused with the revision lag | Days |
| Point-in-time | A data architecture that preserves every vintage | Treated as a nice-to-have | Table keyed by event date and as-of date |
| Scenario forecast | A forecast under a stated, non-probabilistic regressor path | Presented without the scenario label | Outcome units per named scenario |
| Elasticity | Proportional change in the outcome per proportional change in a regressor | Reported as a unit effect | Dimensionless; from a log-log specification |
| Endogeneity | Correlation between a regressor and the error term | Used to mean "the regressor is bad" | Property of the specification |
| Simultaneity | Endogeneity arising because outcome and regressor are determined together | Treated as a forecasting problem rather than an identification one | Property of the system |
| Instrumental variable | A variable that shifts the regressor but not the outcome directly | Chosen for convenience | Series with an exclusion restriction |

Three of these carry most of the interpretive weight.

**Cumulative effect versus peak effect.** The lag weights are a shape; the sum is the effect. A
promotion that lifts sales in week 0 and gives back in weeks 1 to 3 has a cumulative effect that can
be a fraction of its peak, and the peak is what a dashboard shows you. The standard error for the sum
comes from the coefficient covariance matrix as `1'V1`, not from adding the individual standard
errors, because the fit trades one lag against the next and the weights are negatively correlated.

**Long-run multiplier.** A lagged dependent variable makes the distributed lag infinite even when
only a few regressor lags are included: a sustained one-unit change in `x` moves `y` by
`sum(beta_j) / (1 - sum(phi_i))`. Reporting `sum(beta_j)` as "the effect" understates it by the
persistence factor, which at `sum(phi) = 0.6` is a factor of 2.5.

**Exogenous and conditional are claims about the model.** Price is exogenous to a demand equation
only if it is set by a process independent of the demand shock, and almost no commercial price is.
Likewise a conditional forecast suits a scenario meeting while an unconditional one suits a
replenishment buffer; supplying the wrong one is a category error no diagnostic catches.

## 5. The mechanics, step by step

This section builds a regression forecast in nine phases. The order is not arbitrary: Phase B
determines what can enter the model at all, and a defect there cannot be repaired by better estimation
later.

### Phase A — Fix the problem before the model

**Step 1. Name the decision the forecast feeds.** A replenishment quantity, a promotional buy, a
capacity booking, and a financial plan need different horizons, different error metrics, and different
treatments of the regressor path. Write the decision down before writing the equation.

**Step 2. State the origin and the horizon in steps, not dates.** "Thirteen weeks from the Sunday
freeze" is testable. "Q3" is not. Every subsequent availability judgement is relative to this origin.

**Step 3. State the time bucket and its definition.** Weekly, daily, four-weekly, or monthly. The
bucket determines which seasonal frequencies exist, and it must be identical for the outcome, the
promotion calendar, the price series, and the holiday dates; a one-week misalignment looks exactly
like a demand anomaly.

**Step 4. Count what the horizon does to the information set.** At `h = 1` almost everything is known.
At `h = 13` almost nothing is. The same model can be honest at one horizon and leaky at another, which
is why a single backtest number across all horizons hides the problem.

### Phase B — Audit every regressor for availability

**Step 5. List candidate regressors without judging them.** Calendar, price, promotion, weather,
competitor activity, distribution, macro, and any lagged outcome terms. Judgement comes after the
audit, because a variable that is unavailable is unavailable regardless of how well it would fit.

**Step 6. For each, record the event date, the publication date, and the revision policy.** This is
the availability table, and it is the most valuable artefact this topic produces. A worked version
appears in section 6.

**Step 7. Assign each to one of the four boxes.** Known exactly, known with revision, forecastable,
simultaneous. The assignment is a claim about the world and should be signed by someone who owns the
process, not by the analyst alone.

**Step 8. Decide the path policy for each box.** Known exactly: use the contracted or published path.
Known with revision: use the vintage and carry a revision allowance. Forecastable: use an external
forecast where one exists, and record its error. Simultaneous: either instrument it or fix it as a
scenario. Write the policy down; it is part of the model.

**Step 9. Drop or transform what cannot be carried.** A regressor available only at the target date
but not at the origin is not a regressor. It is a post-hoc explanation. Either replace it with a
lagged version, replace it with a forecast of itself, or remove it and say so.

> **Note:** The most common outcome of a genuine availability audit is that the model loses one or
> two of its best-fitting variables. Expect the fit to worsen. That worsening is the removal of
> leakage, not a loss of explanatory power.

### Phase C — Build the calendar basis

**Step 10. Separate the three calendar components.** Seasonality (the smooth annual shape), fixed
holidays (same date or same weekday rule every year), and moving events (Easter, lunar new year,
Ramadan, school terms). They need different regressors because they have different generating
processes.

**Step 11. Choose dummies or harmonics.** They are the same basis at different resolution: at
`K = m/2` the harmonic basis spans the same space as `m - 1` seasonal dummies. The choice is a point
on a bias-variance curve, not between theories. Dummies are local and need one observation per
parameter per year; harmonics are smooth, need `2K` parameters estimated from all observations, and
extrapolate to any future period.

**Step 12. Use harmonics when the period is long.** With `m = 52`, `m - 1 = 51` dummies require
several years before each is estimated from more than three observations; with `m = 365` the dummy
basis is not estimable at all. Harmonics handle both, and handle multiple seasonalities by adding
pairs at different periods.

**Step 13. Choose K by information criterion, starting at one.** Fit `K = 1, 2, 3, ...` and minimise
AICc (corrected Akaike information criterion) or BIC (Bayesian information criterion). For 52 weeks,
`K` between 2 and 4 is usually enough for a smooth shape; for monthly data `K = 6` is the equivalence
point with dummies and is rarely justified. Cap `K` below `m/2`, where the basis becomes singular.

**Step 14. Check the collinearity with the trend.** At `m = 52` the first harmonic is not collinear
with a straight line over one year, but over three years the correlation rises. Report the condition
number; above about 30, trend and seasonality are trading off and both coefficients are unstable.

**Step 15. Build the moving-holiday regressor from a date table, never from a formula.** Easter can be
computed from the computus, but Ramadan depends on observance and one-off or substitute holidays are
announced. Store the event date per year and per jurisdiction. Then build a window in days relative to
the event, with weights that reflect the mechanism.

**Step 16. Distinguish a dummy from a proximity window.** A dummy says "this bucket contains the
event". A proximity regressor says "this fraction of the event's effect falls in this bucket". The
dummy is adequate when the bucket always contains the event whole. The proximity regressor is
necessary when the window straddles a bucket boundary, which for monthly data it usually does. The
standard method builds the regressor from the share of the window falling in the period, and centres
it on its calendar mean so that it is orthogonal to the intercept.

**Step 17. Do not double count a holiday and a seasonal peak.** If the holiday always falls near the
seasonal maximum, the harmonic and the holiday regressor compete and the split is arbitrary. Report
the combined fitted value rather than the two coefficients separately.

### Phase D — Specify the causal terms

**Step 18. Choose the functional form from the mechanism.** A log-log specification gives an
elasticity directly and stabilises variance that grows with volume. A linear specification is easier
to explain and is often adequate for narrow ranges. A promotion depth index in `[0, 1]` interacts
naturally with price. State the form and its implied units before estimation.

**Step 19. Set the lag window from the decision process, not from the fit.** A pantry-loading reversal
is two to four weeks. A durable promotion can pull demand forward by a quarter. A price change may
pass through over two quarters. State the mechanism, set the horizon, then test the endpoint. Do not
extend the lag until the last term becomes insignificant; that is selection by t-statistic and it
invalidates the reported standard errors.

**Step 20. Report the cumulative effect and its own standard error.** With `d` the lag coefficients and
`V` their covariance matrix, the effect is `1'd` and its variance is `1'V1`. Report the shape only as
a shape.

**Step 21. Test whether the regressor is simultaneous.** Ask who sets it and when. If price is set
after observing the week's sell-through, it is simultaneous. If the promotion is planned a quarter
ahead and frozen, it is not. If the answer is "it depends on the week", treat it as simultaneous.

**Step 22. Decide whether the use is prediction or intervention.** For prediction under an unchanged
policy, a simultaneous regressor can be acceptable, because the conditional expectation is stable.
For intervention — "what if we change price" — it is not, and the coefficient must come from a design
that identifies the effect, such as an instrument, a controlled experiment, or a discontinuity. This
is the branch that separates a forecasting model from a causal model, and choosing the wrong one is
how a good forecast becomes a bad decision.

### Phase E — Estimate with an honest error structure

**Step 23. Decide between regression with ARMA errors and OLS with HAC errors.** ARMA errors model
the residual autocorrelation explicitly and improve multi-step forecasts, because the error forecast
is added to the regression forecast. HAC (heteroskedasticity- and autocorrelation-consistent)
standard errors correct the inference and leave the point forecast unchanged. Use the first when the
residual dynamics matter; the second when you only need honest standard errors.

**Step 24. Do not difference the outcome reflexively.** Differencing removes the level information
that a regression on price, weather, and promotion needs. With two trending series the choice is
between a trend term, a cointegration model, and a regression in differences, and the choice must be
stated. Module 04 topic 04 gives the test sequence.

**Step 25. Check the residual autocorrelation before interpreting anything.** Plot the ACF, run
Ljung-Box at the seasonal lag and twice it, and report the statistic. Residual autocorrelation at lag
1 means the dynamics are missing; at lag `m` the seasonal basis is insufficient; at lags just beyond
the last included regressor lag, the window is too short. Module 04 topic 05 covers the formal
treatment.

### Phase F — Project the regressors

**Step 26. Produce one declared path per regressor per horizon.** Not a value, a path. For contracted
price, the path is the contract. For promotion, the frozen plan. For weather, an archived forecast or
a seasonal normal with a stated error. For a competitor index, your own model with its own error.

**Step 27. Record the owner of each path.** The planner owns the promotion path, procurement owns the
price path, the meteorologist or the vendor owns the weather path. A path with no owner is a guess
with extra steps, and it will be revised silently.

**Step 28. State the path's error.** For a contracted path, zero. For an external forecast, the
vendor's published error, or your own backtest of it. For a seasonal-normal profile, the historical
dispersion around the normal at that horizon. Without this number the unconditional interval cannot be
computed.

**Step 29. Choose the scenario structure.** One baseline path plus named alternatives, or a
distribution over paths. Name the scenarios by the decision they test: "price held", "price up 5%",
"promotion cancelled". A scenario without a decision attached is decoration.

### Phase G — Produce the distribution

**Step 30. Simulate rather than add variances.** Draw the regressor paths from their stated
distributions, evaluate the model on each draw, and take quantiles. This handles non-linear
specifications, correlated regressors, and the non-normality that stochastic regressors induce.
Adding `beta' Sigma_x beta` to the residual variance is a linear approximation for a quick check.

**Step 31. Separate the variance components and report them.** Model residual, regressor path,
revision, and parameter uncertainty. The shares say where to invest: if the path dominates, improve
the driver forecast; if the residual dominates, the model is missing structure.

**Step 32. Check the interval against realised errors at each horizon.** A 90% interval should contain
about 90% of realised outcomes at each `h`, not 90% overall. Calibration that holds on average and
fails at the long horizon is the signature of a conditional interval reported as unconditional.

### Phase H — Backtest with the information that existed

**Step 33. Roll the origin and rebuild the regressor path at each origin.** The path used for the
forecast from origin `t0` must be the path that was available at `t0`, not the realised values and not
the path that a later, better-informed model would have produced.

**Step 34. Use vintages for revised series.** If the warehouse holds only the latest revision, say so
and bound the bias: a backtest on revised data is an upper bound on accuracy, and the gap is usually
larger than the improvement the project claims.

**Step 35. Re-fit inside each fold.** Scaling parameters, Box-Cox lambda, the harmonic order `K`, the
holiday window, and the lag length are all estimated quantities. Selecting them on the whole sample
and then backtesting is a form of leakage that is easy to miss because it does not appear in the
design matrix.

**Step 36. Audit the six channels explicitly.** Revised data, actual future regressors, target leakage
through a short lag, centred or forward-looking windows, fold-external preprocessing, and selection
leakage. Section 10 lists the red flag for each.

### Phase I — Report

**Step 37. Report the conditional and the unconditional error separately.** "Conditional on the
seasonal-normal temperature path, 11.2% WAPE. Including the temperature path error, 12.4%." One number
is honest; the other is a claim about a world where the weather is known.

**Step 38. Report against the benchmark, not against zero.** The comparison that matters is the
seasonal naive from Topic 01. An R-squared of 0.93 means nothing if the seasonal naive achieves 0.91.

**Step 39. Name the three assumptions that would invalidate the forecast.** A regressor whose
historical range is exceeded, a structural break in the driver, and a revision policy that changed
mid-sample are the usual three. State them in the note, not in an appendix.

## 6. Worked example

A regional dairy forecasts weekly volume for one chilled beverage SKU (stock-keeping unit) across a
single store cluster. All figures are illustrative but internally consistent, and every calculation
can be reproduced by hand.

### 6.1 The decision and the series

The forecast feeds a weekly replenishment order with a two-week supplier lead time, produced every
Thursday for the following thirteen weeks. The series is 156 weeks of units per week, ranging from
about 1,500 to 2,300 units, with a mean of 1,842 and a standard deviation of 214.

The commercial question is not "what will volume be". It is "what will volume be, given the price
contract we have signed and the promotion calendar we have frozen". That framing already tells you
that price and promotion are in the first availability box, and that weather is in the third.

### 6.2 The regressor availability audit

| Regressor | Source | Event period | Knowledge date | Revision policy | Path used |
|---|---|---|---|---|---|
| Day of week / week index | Calendar | The week itself | Years ahead | None | Known exactly |
| Easter window | Holiday date table | Easter Sunday ± 1 week | Years ahead | None | Known exactly |
| List price | Contract register | The week itself | Signed 12 months ahead | None within contract | Known exactly |
| Promotion depth | Promotion plan | The week itself | Frozen 8 weeks ahead | Amendments logged | Known exactly, with an execution flag |
| Temperature | National meteorological service | The week itself | 10-day forecast; seasonal normal beyond | Forecast revised daily | External forecast, error known |
| Cooling degree days | Derived from temperature | The week itself | As above | Reanalysis revises actuals | Forecastable |
| Competitor price | Weekly store audit | The week itself | Published 5 days after week end | Occasional restatement | Known with revision; lagged two weeks |
| Ending inventory | Warehouse | Week end | Available 1 day after week end | None | **Excluded: target leakage** |

Two entries repay attention. The competitor price is measured in the week but published five days
later, so at a Thursday origin the most recent usable value is from the week before last. Using last
week's audit would leak five days of the future. And ending inventory is excluded because it is
computed as opening stock plus receipts minus sales; including it would hand the model the answer.

### 6.3 The specification

```text
   ln(units_t) = 4.82 + 0.031 t_yr
               + K = 3 Fourier pairs, m = 52
               + 0.19 Easter_window_t
               - 0.84 ln(price_t) - 0.19 ln(price_{t-1})
               + 0.58 promo_t + 0.21 promo_{t-1} - 0.14 promo_{t-2}
               + 0.0031 cdd_t
               + e_t

   t_yr        years since the start of the sample        dimensionless
   Easter_window  weights 0.5, 1.0, 0.5 on weeks -1, 0, +1   dimensionless
   price       contract list price, EUR per unit           EUR per unit
   promo       promotion depth, 0 = none, 1 = full depth   index in [0, 1]
   cdd         cooling degree days, base 18 C              degree-days per week
   e_t         residual                                    log units
```

The price coefficients sum to `-1.03`, a cumulative elasticity just above one in absolute value,
which is plausible for a branded chilled beverage with close substitutes. The promotion coefficients
sum to `0.65`, so a full-depth promotion raises volume by `exp(0.65) - 1 = 92%` over the three weeks,
not by the `exp(0.58) - 1 = 79%` visible in the promotion week.

### 6.4 Choosing K

Fitting the same specification at `K = 1` to `K = 5` gives residual sums of squares and information
criteria:

| K | Parameters | RSS | AICc | BIC |
|---:|---:|---:|---:|---:|
| 1 | 11 | 3.180 | -583.5 | -551.8 |
| 2 | 13 | 2.460 | -618.8 | -581.7 |
| 3 | 15 | 2.243 | -628.3 | -586.0 |
| 4 | 17 | 2.219 | -625.0 | -577.6 |
| 5 | 19 | 2.210 | -620.5 | -568.1 |

AICc selects `K = 3` and BIC selects `K = 3`, which is reassuring on 156 observations. The gain from
`K = 2` to `K = 3` is 0.217 in RSS for two parameters; the gain from `K = 3` to `K = 4` is 0.024 and
is priced as worse by both criteria. Nineteen parameters against 156 observations is already a
parameter-to-observation ratio of about one in eight, so the cap is doing real work.

> **Note:** The seasonal shape has a summer peak and a smaller pre-Christmas peak, and `K = 3`
> represents both because they are smooth. A one-week spike that is not a holiday cannot be
> represented by any `K` below the Nyquist limit, and should not be: it is an event, not a season.

### 6.5 The distributed lag on price and promotion

| Lag | Price weight | Promotion weight | Price cumulative | Promotion cumulative |
|---:|---:|---:|---:|---:|
| 0 | -0.84 | +0.58 | -0.84 | 0.58 |
| 1 | -0.19 | +0.21 | -1.03 | 0.79 |
| 2 | 0.00 | -0.14 | -1.03 | 0.65 |

The price effect is essentially contemporaneous with a small carry-over, which is what a weekly
beverage series should show: households adjust within the week. The promotion effect is
contemporaneous, positive in the following week, and negative in the second week, which is the
pantry-loading give-back. Reporting `+0.58` as "the promotion effect" overstates it by 11% relative to
the cumulative `0.65`; reporting the peak week's units as the lift overstates it by more, because the
peak also contains the seasonal and weather contributions.

The standard error of the cumulative promotion effect comes from the covariance matrix, not from the
individual standard errors. With `1'V1 = 0.0049`, the standard error is `0.070`, so the cumulative
effect is `0.65 ± 1.96 × 0.070 = [0.51, 0.79]` in log units, or `[67%, 120%]` in percentage terms.
Adding the three individual standard errors, which are 0.041, 0.038, and 0.036, would give `0.115` and
an interval wide enough to include no effect at all.

### 6.6 Projecting the regressors

Four paths, four different stories.

| Regressor | Path for h = 1..13 | Path error (log units, h = 13) | Owner |
|---|---|---:|---|
| Price | Contract schedule, flat at 2.49 | 0.000 | Procurement |
| Promotion | Frozen plan: one full-depth week at h = 4 | 0.000, plus 0.020 execution risk | Category |
| Easter | Date table, window at h = 6 | 0.000 | Analytics |
| Temperature | Seasonal normal from a 30-year climatology | 0.052 | Analytics, from meteorology |

The temperature path is the interesting one. At `h = 1` the meteorological service's forecast error is
about 1.8 degree-days, which maps through the coefficient `0.0031` to `0.0056` in log units. At
`h = 13` there is no forecast with skill, so the path is a climatological normal and the error is the
historical dispersion around that normal, about 16.8 degree-days, or `0.052` in log units. The path
error grows by a factor of nine between the short and long horizon, which is exactly the behaviour a
single backtest number across all horizons would hide.

### 6.7 The error budget

At `h = 13`, the variance components are:

```text
   component                 variance   share
   model residual             0.011664   78.6%
   temperature path           0.002704   18.2%
   promotion execution        0.000400    2.7%
   price path                 0.000064    0.4%
   ------------------------------------------------
   total                      0.014832  100.0%
   standard deviation         0.1218  ->  12.2% of the median
```

At `h = 1` the total standard deviation is `sqrt(0.011664 + 0.000031) = 0.1082`, or 10.8%. So the
interval widens from 10.8% to 12.2% across the horizon, and most of the width at every horizon is the
model residual rather than the regressor path.

That is an honest and slightly deflating result, and it is worth stating plainly: **when the model is
weak, the regressor path barely matters, because the model's own error swamps it. The regressor path
matters most when the model is good.** A model with a 4% residual and a 5% weather-path error has an
unconditional error that is 40% larger than its conditional error, and that is the case in which
reporting the conditional number is most misleading.

### 6.8 The leaky backtest and the honest one

Two backtests over the same 26 holdout weeks, using rolling origins and refitting inside each fold.

| Backtest | Price series | Temperature | WAPE |
|---|---|---|---:|
| A — leaky | Realised average selling price | Actual temperature | 8.1% |
| B — honest | Contract list price | Archived forecast, then climatology | 10.9% |
| Production, first quarter | Contract list price | Live forecast | 11.4% |

WAPE (weighted absolute percentage error) is `sum|error| / sum(actual)`.

The gap between A and B is 2.8 percentage points, which is larger than the annual improvement most
forecasting programmes report. It comes from two channels. Realised price contains the markdowns that
were triggered by weak sell-through, so the leaky model is effectively told which weeks were weak.
And actual temperature is a better predictor than any forecast of it, by construction.

Backtest B is the number to quote. It is still slightly optimistic relative to production, because
the archived forecast is the one the meteorological service issued, while the live process sometimes
uses a stale copy, and because promotion execution risk is modelled but not observed.

### 6.9 What the model can and cannot say

It can say: "Given the contract price, the frozen promotion plan, and a seasonal-normal temperature
path, volume over the next thirteen weeks is forecast at 1,910 units per week, with a 90% interval of
1,700 to 2,145."

It cannot say: "A 5% price increase would reduce volume by 5.2%." The price coefficient is estimated
on a period in which price moved within a narrow contractual band, the realised price is partly
simultaneous, and no instrument has been used. The elasticity is a forecasting parameter, not a
causal one, and the difference is the subject of section 5 Phase D step 22.

It also cannot say anything about a price outside the observed range: the contract has never been
above 2.79, and a log-log specification will extrapolate to 3.50 along a line no observation supports.

## 7. Code, query, or calculation

One compact specification in Python using `statsmodels`. The point is the design matrix and the
regressor availability, not the library.

```python
import numpy as np, pandas as pd, statsmodels.api as sm

m, K = 52, 3
d = df.sort_values("week").reset_index(drop=True)
t = np.arange(1, len(d) + 1)

X = pd.DataFrame({"const": 1.0, "trend": t / 52.0})
for k in range(1, K + 1):                        # three harmonic pairs
    X[f"sin{k}"] = np.sin(2 * np.pi * k * t / m)
    X[f"cos{k}"] = np.cos(2 * np.pi * k * t / m)
X["easter"] = d["easter_w"]                      # built from the date table
X["promo"] = d["promo_depth"]                    # frozen promotion plan
X["cdd"] = d["cooling_degree_days"]              # observed history only
for j in range(3):                               # distributed lag on price
    X[f"lprice_l{j}"] = np.log(d["price"]).shift(j)

X, y = X.iloc[2:], np.log(d["units"]).iloc[2:]
fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(fit.params[["lprice_l0", "lprice_l1", "lprice_l2"]].sum())
```

Expected output: `-1.03`, the cumulative price elasticity, in log units per log unit.

Three things to notice. The Easter column is built from the date table and not from observed sales;
built from sales it would contain the outcome. The `shift(j)` uses the contract price, which is known
before the week, so it is not leakage, whereas shifting a realised price would be. And `cov_type`
changes the standard errors and not the coefficients, so it cannot repair a mean equation that
includes a simultaneous regressor.

The forecast step, which is where most implementations go wrong, is the part this block deliberately
omits. It needs a second matrix with `H` rows holding the *projected* regressors, and the projection
must be the same object described in section 6.6, with an owner and an error for each column.

## 8. Data, standards, and regulatory context

Forecast regression sits on reference data that is more governed than most analysts expect. Holiday
dates are not a formula. They are a published calendar with a version, and in several jurisdictions
they are amended by annual decree.

### 8.1 Calendar and time reference

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 8601 | Date and time representation | The week date and ordinal date used as join keys | B |
| IANA time zone database | Offset rules by region | Which local date an event falls on | B |
| National statistical office holiday calendars | Public holidays by jurisdiction | The event dates used to build regressors | B |
| X-13ARIMA-SEATS built-in regressors | `easter[n]`, `labor[n]`, `thank[n]`, `easterstock[n]` | Default window and centering for common holidays | B |
| Genhol | User-defined moving-holiday regressors | Window bounds, proportionality versus count, centering | B |
| JDemetra+ calendar correction | EU statistical practice | Trading-day, leap-year, and Easter contrasts with worked code | B |
| NRF 4-5-4 retail calendar | Retail fiscal weeks | The week index, including the 53rd week | C |

The distinction between a standard and a convention matters here. ISO 8601 is a standard: a week is
defined, and the definition is stable. The 4-5-4 calendar is a convention with accounting
consequences, and two retailers can use different conventions in the same year. A model built on one
and validated on the other has a one-week misalignment that looks like a demand anomaly.

### 8.2 Data vintage and point-in-time architecture

| Practice | What it provides | Status | Grade |
|---|---|---|---|
| Point-in-time tables keyed by event date and as-of date | Every vintage, queryable at any origin | Best practice; uncommon outside finance and official statistics | B |
| Real-time databases maintained by statistical agencies | Archived vintages of published series | Available for major official series | B |
| Warehouse with only the latest revision | Current values, no history of revisions | The default in most companies | C |
| `ASOF JOIN` semantics in a query engine | Nearest prior record at or before the origin | Necessary but not sufficient; requires a stored vintage column | C |

The practical consequence is stark. If the warehouse keeps only the latest revision, an honest
backtest of a model that uses revised regressors cannot be run without reconstructing history, and
that is usually impossible. The correct professional move is to say so: report the backtest as an
upper bound, state the direction of the bias, and quantify it where a vintage does exist.

### 8.3 Weather and external drivers

Weather data is the most commonly used external regressor in retail and the most commonly mishandled.
Three points constrain it.

- **Actuals are revised.** Reanalysis products restate historical values, sometimes months later.
  Using the reanalysis as "what the weather was" is fine for training and wrong for backtesting,
  because the forecast issued at the origin was made from a different product.
- **Forecasts have a horizon-dependent error.** Skill falls with lead time, and the fall is not
  linear. A model that uses a 10-day forecast at `h = 10` and a climatological normal at `h = 30`
  must use both errors, not one.
- **Degree-day definitions vary.** Base temperature, whether cooling and heating degree days are both
  counted, and whether they are computed on daily means or hourly data all change the series. State
  the base and the aggregation.

### 8.4 What is convention rather than requirement

A one-week holiday window is a default, not a finding. `K = 6` for monthly data is the equivalence
point with dummies, not a recommendation. A linear trend is a software default. A 10-day weather
horizon and a one-week competitor-price lag are habits; the correct lag is the audit's publication
lag. An analyst who can separate a requirement from a convention can defend a specification.

## 9. Industry practice and case evidence

### 9.1 What the forecasting literature recommends

Hyndman and Athanasopoulos set out dynamic harmonic regression as the method of choice when the
seasonal period is long or when there are several seasonal periods. Their argument is threefold:
seasonal ARIMA with a period of 52 or 365 needs `m - 1` initial states and becomes impractical;
Fourier terms handle multiple periods by adding pairs at different frequencies; and smoothness is
controlled by `K` rather than by the data. They select `K` by AICc, and they state the main limitation
plainly: the seasonal pattern is assumed fixed over time.

In their worked example on Australian eating-out expenditure, AICc is minimised at `K = 6`, which for
monthly data is the point at which the harmonic basis reproduces the seasonal dummies exactly. The
gain from harmonics is in the long-period and multiple-period cases, not in the monthly case. Grade A
for method.

### 9.2 The Easter-effect literature

The U.S. Census Bureau's work on Easter regressors is the most rigorous public treatment of the
moving-holiday problem, and its findings are sobering for anyone who has added an Easter dummy
casually. Findley, Wills, and Monsell (2005) examine how many years of data are needed to detect an
Easter effect reliably, how often a spurious effect is found, and how often the window length is
misidentified — selecting a longer window when the true effect is concentrated in a short one. Their
conclusion, in general terms, is that short samples produce unreliable and frequently false Easter
effects, and that window selection is a substantial source of error even when the effect is real.

Soukup and Findley (2000) make the prior point: moving holidays are not confined to the seasonal
component, and ignoring them reduces forecasting ability and produces residual lack of fit. The
commercial implication is direct. A weekly model with three years of history does not have enough
independent Easter observations to estimate an Easter effect with any confidence, and the honest
response is a prior-informed window or no Easter term at all, not a significant coefficient.

The Census Genhol utility documentation describes the practical machinery: a window specified in days
before and after the event, a choice between a proportionality regressor and a count regressor, and
centering on the mean or the calendar. McElroy, Monsell, and Hutchinson (2018) extend the treatment to
daily data and report significant holiday effects for Easter, Black Friday, Cyber Monday, and Labor
Day in daily retail series. Grade B for the Census material: official statistical practice, not
peer-reviewed in the journal sense, but the most careful public work available.

### 9.3 Calendar features in retail machine learning

Huber and Stuckenschmidt (2020) evaluate machine-learning methods for daily retail demand with
explicit attention to calendar special days. The paper is relevant here less for a headline result
than for its framing: in daily retail data, calendar effects are not a preprocessing detail. They are
a substantial share of the predictable variation, and how they are encoded — dummies, proximity
windows, or event-specific regressors — changes the result. Grade A for method.

The M5 retail dataset, used in the 2020 competition and distributed with a calendar file of events,
holidays, and SNAP days (the US Supplemental Nutrition Assistance Program), is the standard public
testbed. The data are real and public; claims about which methods won are competition claims. Grade C
for the narrative, A for the data.

### 9.4 Temporal aggregation and the calendar

Calendar effects are better identified at an aggregate level than at item level, because the
signal-to-noise ratio is higher and the calendar variation is common across items. Kourentzes,
Rostami-Tabar, and Barrow (2017) report that multiple temporal aggregation levels are more robust
than a single "optimal" level, with gains of roughly one to three percent on real datasets;
Kourentzes and Petropoulos (2015) extend the idea to promotional modelling and report lower error and
bias than single-level benchmarks on UK cider SKU data. Both are as reported in the abstracts and are
dataset-specific: Grade A for method, with the effect sizes indicative. The message is that a
calendar regressor built at SKU level may be noisier than the same regressor estimated at category
level and reconciled down, which is Module 08 territory.

### 9.5 The leakage evidence

Leakage is documented more thoroughly in machine learning than in supply chain forecasting, and the
lessons transfer directly. Kaufman, Rosset, Perlich, and Stitelman (2012) formalise leakage as a
property of the relationship between the feature and the target relative to the data generation
process, and distinguish it from overfitting: a leaky model is not too complex, it is too well
informed. The practitioner literature on time-series leakage catalogues the same channels this topic
lists — revised data, forward-looking windows, fold-external preprocessing, and covariates aligned to
the target date rather than to their publication date.

The course notes and practitioner articles that document these cases are not peer-reviewed and should
be read as experience reports. Their value is that they show the failure is common and the mechanism
is mundane. The tooling evidence is stronger in one respect: open feature requests on forecasting
libraries ask for "as-of" filtering of exogenous covariates in backtesting, which is direct evidence
that the default implementations do not enforce the availability boundary. Grade D for tool
documentation, E for course notes, A for the formal leakage paper.

One pattern from the practitioner material is worth repeating: in the documented cases, every leaky
feature outperformed the honest model in validation and contributed far less in production. That is
the signature to look for.

## 10. Failure modes and red flags

The first four failures are silent: the output looks healthy, the R-squared is high, and the signs are
expected. The red flag is in the design, not the output.

| # | Failure mode | What it looks like | Why it happens | Red flag visible in a meeting or dataset | Repair |
|---:|---|---|---|---|---|
| 1 | Actual future regressor in the backtest | Holdout error excellent, production error much worse | The backtest used realised values for weeks the model had to forecast | Production WAPE is more than 1.5 times the backtest WAPE, and the gap grows with horizon | Rebuild the path as of each origin; archive the forecasts that existed |
| 2 | Revised data in the backtest | Backtest error falls when the data are refreshed | The warehouse keeps only the latest revision | Nobody can say what the series looked like on the validation date | Use vintages; if absent, report the backtest as an upper bound |
| 3 | Target leakage through a short lag | A driver has an implausibly high coefficient and the model is unbeatable | The regressor is computed from the outcome, or recorded after it | Coefficient sign implies a physical impossibility, such as inventory driving sales with elasticity near one | Trace each regressor to its definition and its recording time |
| 4 | Conditional error reported as unconditional | Prediction intervals do not widen with horizon | Only the residual standard deviation is used | Interval width is constant across a 13-week horizon | Simulate the regressor paths; report both errors |
| 5 | Regressor path with no owner | The forecast changes when someone re-runs it | Each analyst assumes a different price or promotion path | Two forecasts for the same week differ and nobody can name the input that changed | Assign an owner per path; version the path with the forecast |
| 6 | Simultaneous driver used as a lever | A price elasticity is quoted for a decision that changes price | The coefficient is a conditional association, not a causal effect | The elasticity is larger than any experimental estimate, or the sign is wrong | Instrument, experiment, or relabel the output as a forecast rather than an effect |
| 7 | Over-harmonised seasonality | The seasonal fit wobbles and the last pair is barely identified | `K` chosen by maximum R-squared | Condition number above 30; `K` approaching `m/2` | AICc with a stated cap; stability check across origins |
| 8 | Holiday dummy keyed to a calendar slot | The holiday coefficient is small and unstable | A moving holiday is pinned to a week | Residual spikes migrate to a different week each year | Event-keyed window; proximity regressor |
| 9 | Distributed lag truncated at the peak | The promotion looks more effective than it is | The give-back lies beyond the last included lag | All lag weights are positive, and the residual ACF spikes just past the last lag | Extend to the mechanism horizon, then test the endpoint |
| 10 | Regressor extrapolated outside its range | A scenario produces an implausible forecast | A linear or log-linear form has no interior bound | The scenario price is below any observed price by a wide margin | Bound the scenario; use a form with saturation; report the range |
| 11 | Structural break in a driver | Coefficients move across rolling origins | A new competitor, a packaging change, a channel shift | Recursive coefficient moves by more than two standard errors | Add a break term, shorten the sample, or split the model |
| 12 | Residual autocorrelation left in | Standard errors too small; multi-step forecast biased | The error dynamics were ignored | Ljung-Box p below 0.01; Durbin-Watson far from 2 | Model the errors, or move to a state-space formulation |

Failure modes 1, 3, and 5 survive a coefficient review because each produces a plausible number with a
plausible sign; they are detectable only by checking the design against the calendar, the data
dictionary, and the process.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks if it fails | How to test it |
|---:|---|---|---|---|
| 1 | Linearity in the chosen link | The coefficient is a constant effect | A threshold, a saturation, or a multiplicative pattern makes the forecast biased at the extremes | Residuals against fitted values and against each regressor |
| 2 | Exogeneity of the regressor | The coefficient is stable under re-running the process | Price set in response to demand makes the coefficient a mix of both directions | Ask who sets the value and when; compare with an experimental estimate |
| 3 | Stable parameters over the sample | One coefficient describes the whole history | A regime change makes the forecast a blend of two regimes | Recursive estimation; Chow test at a known break |
| 4 | A usable regressor path exists | The forecast needs values at the horizon | No forecast exists for a driver that matters | Check the path owner and the path error before fitting |
| 5 | Correct calendar mapping | Seasonality and holidays are attributed to the right bucket | Mixed week definitions shift effects by one period | Row counts per year; week-boundary spot checks against the source calendar |
| 6 | No material measurement error in regressors | Classical measurement error attenuates the coefficient toward zero | A noisy promotion flag understates the promotion effect and overstates the price effect | Compare two sources; check whether the coefficient rises when a better measure is used |
| 7 | No omitted time-varying confounder | The coefficient is not picking up a third variable | A competitor launch in the same weeks is attributed to your price | Event log; check coefficient stability when the confounder is added |
| 8 | Errors are stationary and not strongly autocorrelated after modelling | Standard errors and intervals are valid | Inference is too confident and multi-step forecasts are biased | ACF, Ljung-Box, and a rolling-origin calibration check |
| 9 | Regressors stay inside their historical range | The functional form is supported by data | Extrapolation is arbitrary, especially in log-log form | Compare the scenario path with the observed range and report the overlap |
| 10 | The revision process is stable | Vintage-adjusted and latest-value data are comparable | A change in reporting practice looks like a level shift | Plot revisions by vintage; check the revision distribution over time |

Three limits follow. **Causality:** the design estimates an association conditional on the controls;
if the promotion ran in already-weak weeks the coefficient is a lower bound, and no diagnostic reveals
it. **Adaptation:** a Fourier basis imposes a fixed seasonal shape, so refit annually or test for a
break. **Power:** Ljung-Box at 52 lags with three years of data has little power, and an Easter
coefficient from five Easters is not an estimate of anything stable.

## 12. Exercises

Each exercise states a deliverable, acceptance criteria, and a time estimate. Section 13 gives full
calculations for exercises 1 to 6 and rubrics for the rest.

### Exercise 1 — Classify the regressors (application, 30 minutes)

List eight candidate regressors for a forecasting problem you know and assign each to one of the four
availability types.

**Deliverable:** a table of regressor, type, path policy, owner, and path error. **Acceptance:** all
four types represented; each policy specific enough that two analysts would produce the same path; at
least one regressor dropped for being uncarryable. **Time:** 30 minutes.

### Exercise 2 — Build the basis and choose K (calculation, 50 minutes)

For `m = 52`, construct the Fourier columns for `K = 1` and `K = 2` at weeks 1, 7, 13, 26, 39, 47, and
52. Then recompute AICc for `K = 1` to `K = 5` from the RSS table in section 6.4.

**Deliverable:** the column table, the recomputed criteria, and the selected `K`. **Acceptance:** sine
and cosine values correct to four decimals; `AICc = n ln(RSS/n) + 2p + 2p(p+1)/(n-p-1)` applied
consistently; the selection justified against the parameter count. **Time:** 50 minutes.

### Exercise 3 — Build a proximity regressor (calculation, 35 minutes)

A school half-term window runs from 9 days before to 5 days after the first Monday of the holiday, and
in a monthly series the first Monday falls on the 3rd of a 31-day month.

**Deliverable:** the two monthly shares, the centering mean, and the regressor values. **Acceptance:**
shares sum to one across the affected months; the centering mean is computed over the full calendar,
not only event years; the distinction between proportionality and count is stated. **Time:** 35
minutes.

### Exercise 4 — Cumulative effect and its standard error (calculation, 30 minutes)

A three-week distributed lag on promotion gives weights `(0.58, 0.21, -0.14)` with individual standard
errors `(0.041, 0.038, 0.036)` and `1'V1 = 0.0049`.

**Deliverable:** the cumulative effect, its standard error, the 95% interval in log units and in
percentage terms, and the interval from summing the standard errors. **Acceptance:** `1'V1` used
directly; the transformation to percentages stated; one sentence on which interval is correct and why.
**Time:** 30 minutes.

### Exercise 5 — The leakage audit (critique, 45 minutes)

A backtest uses rolling origins, 26 folds, refitting inside each fold, and four regressors: realised
price, actual temperature, planned promotion, and a two-week lag of competitor price. The reported
WAPE is 8.1%.

**Deliverable:** a ranked list of leakage channels with the mechanism and the evidence requested for
each. **Acceptance:** at least three channels identified; each ranked by expected effect on the
reported number; the requests are specific artefacts, not general concerns. **Time:** 45 minutes.

### Exercise 6 — Propagate the path error (calculation, 35 minutes)

A model has a residual standard deviation of 0.06 in log units. Two regressors have path errors at
`h = 12` of 0.03 and 0.00, with coefficients 0.8 and -0.5.

**Deliverable:** the variance components, the unconditional standard deviation, and the path share.
**Acceptance:** the coefficient is applied to the path error before squaring; the independence
assumption is stated; the result is interpreted in percentage terms. **Time:** 35 minutes.

### Exercise 7 — Diagnose the wrong sign (diagnosis, 30 minutes)

A regression of log volume on log price returns a coefficient of `+0.31` with `t = 4.2`.

**Deliverable:** three mechanisms, three tests, and one reporting decision. **Acceptance:** at least
one mechanism is simultaneity and one is an omitted variable; the tests are feasible with the data
described; the reporting decision does not overclaim. **Time:** 30 minutes.

### Exercise 8 — Write the forecast note (writing, 40 minutes)

Using the worked example, write the two-paragraph note that accompanies the forecast. The first
paragraph states the conditional forecast and its interval; the second states the unconditional
interval and the assumptions that would invalidate it.

**Deliverable:** two paragraphs with units, intervals, and named assumptions. **Acceptance:**
conditional and unconditional errors separated; no causal verb stronger than the design supports;
three invalidating assumptions named. **Time:** 40 minutes.

## 13. Self-check answers

### Answer 1 — classification

A strong submission has at least one regressor in each box and is specific about the path. For a
grocery category: day of week and Easter (known exactly, zero error); competitor price (known with
revision, error from the restatement dispersion, path lagged by the audit lag); temperature
(forecastable, error from the meteorological service's published skill); own promotional price
(simultaneous if markdowns are reactive, known exactly if the plan is frozen). The dropped regressor
is usually realised average selling price or ending inventory. A weak submission classifies
everything as "external" and gives no path owner.

### Answer 2 — basis and K

With `m = 52`, the angle is `theta_t = 2 pi t / 52 = pi t / 26`.

| `t` | `theta` | `sin` | `cos` | `2 theta` | `sin 2 theta` | `cos 2 theta` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1208 | 0.1205 | 0.9927 | 0.2417 | 0.2393 | 0.9709 |
| 7 | 0.8458 | 0.7485 | 0.6631 | 1.6916 | 0.9927 | -0.1205 |
| 13 | 1.5708 | 1.0000 | 0.0000 | 3.1416 | 0.0000 | -1.0000 |
| 26 | 3.1416 | 0.0000 | -1.0000 | 6.2832 | 0.0000 | 1.0000 |
| 39 | 4.7124 | -1.0000 | 0.0000 | 9.4248 | 0.0000 | -1.0000 |
| 47 | 5.6790 | -0.5681 | 0.8230 | 11.3581 | -0.9350 | 0.3546 |
| 52 | 6.2832 | 0.0000 | 1.0000 | 12.5664 | 0.0000 | 1.0000 |

For the criteria, `n = 156` and `p = 9 + 2K`: intercept, trend, Easter, promotion, cooling degree
days, and two price lags account for nine parameters before the harmonics.

```text
   K = 1: p = 11, ln(3.180/156) = -3.8930, 156 x (-3.8930) + 22 + 264/144
                                = -607.31 + 22 + 1.83 = -583.48
   K = 3: p = 15, ln(2.243/156) = -4.2421, 156 x (-4.2421) + 30 + 480/140
                                = -661.77 + 30 + 3.43 = -628.34
   K = 4: p = 17, ln(2.219/156) = -4.2529, 156 x (-4.2529) + 34 + 612/138
                                = -663.45 + 34 + 4.43 = -625.02
```

`K = 3` minimises and `K = 4` is penalised, matching section 6.4. A submission that omits the
`2p(p+1)/(n-p-1)` correction, or changes the parameter count between rows, does not count: the
correction is 1.8 at `K = 1` and 4.4 at `K = 4`, enough to change a close call.

### Answer 3 — proximity regressor

The window is 15 days: 9 before, the first Monday, and 5 after. With the first Monday on the 3rd, the
window runs from the 25th of the previous month to the 8th of the current month.

```text
   days_previous = 31 - 25 + 1 = 7
   days_current  = 8
   total         = 15
   share_previous = 7 / 15 = 0.4667
   share_current  = 8 / 15 = 0.5333
```

Both shares sum to one. Centre on the mean share over the full calendar, not on 0.5, because the
current-month mean share is well above 0.5 and using 0.5 biases the intercept. State whether the
regressor measures intensity (proportionality) or days (count): the two carry different units.

### Answer 4 — cumulative effect

```text
   sum d = 0.58 + 0.21 - 0.14 = 0.65 log units
   SE    = sqrt(0.0049) = 0.0700
   95% CI = 0.65 +/- 1.96 x 0.0700 = [0.513, 0.787]

   In percentage terms: exp(0.513) - 1 = 67.0%;  exp(0.787) - 1 = 119.7%
   So the effect is between +67% and +120% over the three weeks.

   Summed standard errors: 0.041 + 0.038 + 0.036 = 0.115
   95% CI = 0.65 +/- 0.225 = [0.425, 0.875]  -- too wide, and it includes
            values the data do not support at the lower end.
```

Only the first interval is correct. The lag weights are negatively correlated because the fit trades
one week against the next, so the sum is better determined than any individual weight. The summed
standard error assumes independence and overstates the uncertainty by about 64%.

### Answer 5 — leakage audit

The channels present, in likely order of magnitude:

1. **Actual temperature.** The largest, because temperature is a genuine driver and the realised value
   is a better predictor than any forecast. Expected to account for most of the 2 to 3 point gap.
2. **Realised price.** If the realised average selling price includes markdowns triggered by weak
   sell-through, the model is told which weeks were weak. The magnitude depends on how many weeks had
   unplanned markdowns; ask for the markdown log.
3. **Selection leakage.** If `K`, the lag length, or the holiday window were chosen on the full sample
   before the backtest, the reported error is optimistic. Ask for the selection log and the fold
   boundaries.
4. **Competitor price lag.** Two weeks may be correct, but the audit's publication lag must be
   checked against it: a two-week lag is honest if the audit appears five days after week end, a
   one-week lag is not. Occasional restatements of the audit are a fifth, smaller channel.

The requests are: the archived temperature forecasts by issue date, the markdown log, the model
selection log with fold boundaries, and the competitor audit publication timestamps.

### Answer 6 — path error propagation

```text
   model variance         0.06^2            = 0.003600
   regressor 1            (0.8 x 0.03)^2    = 0.000576
   regressor 2            (-0.5 x 0.00)^2   = 0.000000
   total                                     0.004176
   sd                                       0.0646  -> 6.5% of the median
   path share             0.000576 / 0.004176 = 13.8%
```

The coefficient is applied to the path error before squaring, which is the step most submissions miss.
The independence assumption is doing work: if the path errors are correlated, the cross term
`2 beta_1 beta_2 cov(x_1, x_2)` must be included, and it can be negative. With a 6% residual and a 14%
path share, the unconditional error is 8% larger than the conditional one.

### Answer 7 — the wrong sign

Three mechanisms. **Simultaneity:** price is raised when demand is strong, so the observed
price-volume correlation is positive even though the demand curve slopes down. **Omitted variable:** a
quality improvement or a packaging change coincided with the price increase, and the model attributes
the quality effect to price. **Composition:** the price variable is an average across pack sizes, and
the mix shifted toward larger packs, which have a lower unit price and higher volume.

The tests: for simultaneity, find a cost shifter (a commodity input price, a freight rate) and
estimate by instrumental variables, or find a period in which price changed for an exogenous reason.
For the omitted variable, add the quality or pack variable and see whether the sign flips. For
composition, rebuild the price variable at constant mix.

In the meantime, report the coefficient as unidentified, use the model only for forecasting under the
historical price policy, and say so.

### Answer 8 — the note

Conditional: "Given the contract price of EUR 2.49, the frozen promotion plan with one full-depth week
in week 4, and a seasonal-normal temperature path, volume for weeks 1 to 13 is forecast at 1,910 units
per week, with a 90% interval, conditional on those paths, of 1,735 to 2,100 units (10.8% error)."

Unconditional: "Including the error in the temperature path, the 90% interval widens to 1,700 to 2,145
units per week, a 12.2% error. The forecast assumes the contract price holds, the promotion runs as
planned, and temperature stays within its historical range; a contract renegotiation, a promotion
cancellation, or a heat event outside the range would invalidate it, and it should not be used to
evaluate a price change."

Both paragraphs must carry units and intervals, and the second must name three assumptions. A
submission that reports one interval and calls it "the" forecast fails.

## 14. Interview questions

**1. "What is the difference between a conditional and an unconditional forecast?"**

Strong: the conditioning set is the regressor path; the residual standard deviation is conditional;
the variance decomposition shows where path error enters; planners need the unconditional one, which
most outputs do not supply. Weak: "one includes uncertainty" without saying which.

**2. "A regressor predicts beautifully but is published six weeks after the period. What do you do?"**

Strong: classify it as known-with-revision or unavailable, replace it with a lagged version or a
forecast of itself, and quantify what the model loses. If the publication lag exceeds the horizon, the
variable cannot be used at all. Weak: "use it anyway, it is in the warehouse".

**3. "Why use Fourier terms instead of seasonal dummies?"**

Strong: the same basis at different resolution; equivalence at `K = m/2`; parameter count; long and
multiple periods; smoothness controlled by `K`; extrapolation to any future period. For monthly data
the advantage is small. Weak: Fourier is "more advanced" or "captures seasonality better".

**4. "Easter moves. How do you handle it?"**

Strong: an event-keyed window in days relative to Easter Sunday, built from a date table, centred on
its calendar mean, with the window chosen from the mechanism. Short samples cannot identify the
window, and the Census literature finds frequent window misidentification. Weak: a dummy for "April".

**5. "What is the cumulative effect of a distributed lag and how do you get its standard error?"**

Strong: the sum of the weights, with variance `1'V1` from the coefficient covariance matrix, not the
sum of the individual standard errors, because the weights are negatively correlated. The peak is not
the effect. Weak: reports the lag-zero coefficient, or adds the standard errors.

**6. "A backtest shows 8% error and production shows 12%. What are the candidate explanations?"**

Strong: actual future regressors, revised data, target leakage through a short lag, fold-external
preprocessing, selection leakage, and a genuine distribution shift; ranks them and names the artefact
that would confirm each. Weak: "the model overfit", which is unlikely under a rolling-origin design.

**7. "How would you detect target leakage?"**

Strong: trace every regressor to its definition and recording time; look for regressors computed from
the outcome, such as ending inventory; check for coefficients that imply physical impossibilities; and
lag the suspect by one more period to see whether the fit survives. Weak: "use cross-validation".

**8. "When can you use a simultaneous regressor in a forecast?"**

Strong: for prediction under an unchanged policy, where the conditional expectation is stable; never
for an intervention that changes the regressor rule; and the distinction must be stated. Weak:
"never", which is too strong, or "always, if it fits better", which is the failure.

**9. "How many harmonics would you use for weekly data with three years of history?"**

Strong: `K` between 2 and 4, selected by AICc, capped below 26, and checked for stability across
origins; 156 observations against `2K` columns plus the other terms is a tight budget. Weak: `K = 10`
because "more harmonics capture more detail", or `K = 6` because that is the monthly default.

**10. "How do you report forecast error when the regressor path is uncertain?"**

Strong: report both the conditional and the unconditional error, simulate the path rather than adding
variances, and check calibration by horizon rather than overall; the unconditional error is not normal
when the regressors are stochastic. Weak: reports the regression RMSE.

**11. "What would make you distrust a price elasticity of -1.9 with t = 8?"**

Strong: a realised rather than planned price, simultaneity, an omitted quality or pack variable, a
narrow observed price range, and price movement concentrated in promotions. A large t on 156
observations is easy with a simultaneous regressor. Weak: "t = 8 is significant, so it is fine".

## 15. Key terms

- **Forecast origin** — the point in time whose information set bounds the forecast.
- **Information set** — everything knowable at the origin, including publication lags; the
  **forecast horizon** is the number of steps between origin and target.
- **Regressor path** — the sequence of regressor values used across the horizon.
- **Conditional forecast** — a forecast holding the regressor path fixed.
- **Unconditional forecast** — a forecast integrating over the path distribution.
- **Dynamic regression** — a regression whose errors follow an ARIMA process.
- **ARIMAX** — ARIMA with exogenous regressors, in the regression-with-ARMA-errors sense.
- **ARDL** — autoregressive distributed lag, with own lags and regressor lags.
- **Distributed lag** — a specification in which the outcome depends on current and past regressor
  values.
- **Lag weight** — the coefficient on one lag; **cumulative effect** — their sum.
- **Long-run multiplier** — `sum(beta) / (1 - sum(phi))` in an ARDL.
- **Fourier term** — a sine or cosine at an integer harmonic of the fundamental frequency.
- **Harmonic order K** — the number of sine-cosine pairs, giving `2K` columns.
- **Seasonal period m** — observations in one seasonal cycle.
- **Amplitude** — `sqrt(a^2 + b^2)`, in outcome units; **phase** — `atan2(b, a)`, in radians.
- **Nyquist limit** — the highest identifiable frequency, `K <= m/2`.
- **Moving holiday** — a holiday whose calendar date changes year to year.
- **Holiday window** — the span of days around an event over which its effect acts.
- **Proximity regressor** — the share of a window falling in a period.
- **Leakage** — use of information not available at the origin; **target leakage** is the case where a
  regressor is a function of the outcome.
- **Data vintage** — a series as it stood at a particular date; **point-in-time** is the architecture
  that preserves every vintage.
- **Elasticity** — proportional change in outcome per proportional change in a regressor.
- **Endogeneity** — correlation between a regressor and the error term; **simultaneity** is the joint-determination case.

## 16. Sources

**Peer-reviewed methods — Grade A**

- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed. OTexts,
  sections 7.4 and 10.5. https://otexts.com/fpp3/dhr.html. Grade A for method; a textbook, not a
  primary source for any statistic.
- De Livera, A. M., Hyndman, R. J., and Snyder, R. D. (2010). "Forecasting time series with complex
  seasonal patterns using exponential smoothing." Trigonometric seasonality, and equivalence with
  index seasonality at `k = m/2`. Grade A.
- Feldstein, M. S. (1971). "The error of forecast in econometric models when the forecast-period
  exogenous variables are stochastic." *Econometrica*. Grade A.
- McCullough, B. D. (1996). "Consistent forecast intervals when the forecast-period exogenous
  variables are stochastic." *Journal of Forecasting*. Grade A.
- Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. (2012). "Leakage in data mining:
  formulation, detection, and avoidance." Leakage as distinct from overfitting. Grade A.
- Berry, S. T., and Haile, P. A. (2021). "Foundations of demand estimation." NBER Working Paper
  29305. Simultaneity, and why price instruments must be excluded cost shifters. Grade A.
- Berry, S. T., and Haile, P. A. (2016). "Identification in differentiated products markets."
  *Annual Review of Economics*, 8, 27–52. Grade A.
- Chintagunta, P., Kadiyali, V., and Vilcassim, N. (2006). "Endogeneity and simultaneity in
  competitive pricing and advertising." *Journal of Business*, 79(6), 2761–2788. Grade A.
- Ebbes, P., Papies, D., and van Heerde, H. J. (2017). "Dealing with endogeneity: a nontechnical
  guide for marketing researchers." Springer. Grade A/B.
- Huber, J., and Stuckenschmidt, H. (2020). "Daily retail demand forecasting using machine learning
  with emphasis on calendric special days." *International Journal of Forecasting*. Grade A.
- Kourentzes, N., Rostami-Tabar, B., and Barrow, D. K. (2017). "Demand forecasting by temporal
  aggregation." *Journal of Business Research*. Effect sizes as reported, dataset-specific. Grade A.
- Kourentzes, N., and Petropoulos, F. (2015). "Forecasting with multivariate temporal aggregation:
  the case of promotional modelling." *International Journal of Production Economics*. Grade A.
- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., and Petropoulos, F. (2017). "Forecasting with
  temporal hierarchies." *European Journal of Operational Research*. Grade A.
- Artis, M., Clavel, J. G., Hoffmann, M., and Nachane, D. "Harmonic regression models: a comparative
  review with applications." SSRN working paper. Grade B.

**Official statistical practice — Grade B**

- Findley, D. F., Wills, K. C., and Monsell, B. C. (2005). "Issues in estimating Easter regressors
  using RegARIMA models with X-12-ARIMA." U.S. Census Bureau.
  https://www.census.gov/library/working-papers/2005/adrm/findley-01.html
- Soukup, R. J., and Findley, D. F. (2000). "Modeling and model selection for moving holidays."
  U.S. Census Bureau.
- McElroy, T., Monsell, B., and Hutchinson, R. (2018). "Modeling of holiday effects and seasonality
  in daily time series." U.S. Census Bureau Research Report RRS2018-01.
- U.S. Census Bureau. *Genhol: user-defined moving holiday regressors*.
  https://www2.census.gov/software/x-13arima-seats/win-genhol/
- JDemetra+. "Calendar correction." https://doc.jdemetra.org/a-calendar-correction
- Eurostat. *ESS Guidelines on Seasonal Adjustment*.
- ISO 8601 date and week representation; IANA time zone database.

**Industry and company evidence — Grade C**

- National Retail Federation. Retail 4-5-4 calendar, including the 53rd week. A convention with
  accounting consequences, not a standard.
- M5 retail dataset and calendar file, 2020 competition. Public data; competition narratives are
  claims, not findings.

**Vendor and tool documentation — Grade D**

- `statsmodels` documentation for `OLS`, `cov_type`, and `ardl`.
- `sktime` ARDL forecaster documentation, including automatic lag selection and the `causal` option.
- `skforecast` documentation and issue #1131, requesting as-of filtering for exogenous covariates in
  backtesting. Evidence that the boundary is not enforced by default.
- EViews "Forecast Basics", on decomposing forecast error and on ARIMAX intervals conditioning on
  the regressors.
- GAUSS `arimaForecast`, which states that ARIMAX prediction intervals ignore future regressor
  uncertainty.

**Practitioner commentary and course notes — Grade E**

- DataInterview, "Data leakage in time series forecasting". Experience report, not verified.
- SlackQuant Research, "Look-ahead bias and data leakage". Not peer-reviewed.
- OpenForecast, "smooth package for R: exogenous variables". Vendor-adjacent blog.
- adsp-stats, "Harmonic predictors and multiple seasonality". Course notes.

**Not verified.** No claim in this file depends on a vendor-reported effect size. Numbers taken from
papers are those reported in their abstracts and are labelled dataset-specific.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_arima-sarimax.md`](03_arima-sarimax.md)
- Next topic: [`05_hierarchy-aggregation.md`](05_hierarchy-aggregation.md)
- Backtesting: [`08_rolling-origin-backtesting.md`](08_rolling-origin-backtesting.md)
- Error metrics: [`09_error-metrics-cost.md`](09_error-metrics-cost.md)
- Foundations of the benchmark: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md)
- Error structure: [`02_exponential-smoothing.md`](02_exponential-smoothing.md)
- The formal treatment of seasonality, holidays, and distributed lags:
  [`../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md`](../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md)
- Autocorrelation-robust inference:
  [`../../01_foundations/04_statistics_causal_inference/05_autocorrelation-robust-errors.md`](../../01_foundations/04_statistics_causal_inference/05_autocorrelation-robust-errors.md)
- Calendars, weeks, and time zones:
  [`../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Foundations index: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Part map: [`../_index.md`](../_index.md)
- Demand sensing and promotions: [`../09_demand_sensing_promotions_npi/_index.md`](../09_demand_sensing_promotions_npi/_index.md)
- Module survey: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Related survey: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
