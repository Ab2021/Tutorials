> Module 04 — Statistics and Causal Inference
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.02, 04.03

# Regression with Time Series, Seasonality, and Holidays

---

## 1. Why this topic exists

A weekly demand model for a distribution centre reports that this December will be 9.4% below
last December. The category manager has one question: is demand falling, or did the calendar
move? Three things moved. Easter fell a week later than last year, so the pre-Easter build
landed in a different week. The prior year had 53 retail weeks, so "December" was one week
longer. And a November promotion pulled volume forward from December, which the model recorded
as a December decline because the promotion was not in it.

None of the three is a demand signal. All three are calendar and specification. The model was
estimated correctly and answered the wrong question, which is the characteristic failure of
regression on supply chain time series.

**The assumption that breaks.** Ordinary least squares (OLS) treats observations as
exchangeable draws. The standard error of a coefficient is built from the assumption that the
errors are independent and identically distributed (i.i.d.): each row tells you something new,
and no row tells you something about another. A weekly series violates this in at least three
ways, and each violation has a different repair.

| Break | What it looks like | What it does to OLS | Where handled |
|---|---|---|---|
| Serial correlation | Residuals correlated week to week; a shock persists for several weeks | Standard errors too small, t-statistics too large, confidence intervals too narrow | Phase G; Topic 05 |
| Shared calendar shocks | Every SKU spikes in the same week; a holiday is attributed to whatever covariate is nearby | Errors correlated across units and across time; omitted-variable bias in seasonal coefficients | Phase D; section 6.7 |
| Trending regressor and trending outcome | Both series rise for unrelated reasons; R-squared near 0.9 with no relationship | Spurious regression: high R-squared, large t, meaningless coefficient | Phase F; section 6.8 |

A fourth break is equally common: **overlapping horizons**. Forecast four weeks ahead from a
four-week moving average and consecutive errors share three of their four inputs, so they are
autocorrelated by construction. The error structure is a property of the forecast construction,
and it must be modelled or the intervals are fiction.

**Regression is used for two jobs.** The forecasting job asks what next week will be. The
attribution job asks what the promotion, the holiday, or the price change did. This topic is
about the second job, which is why diagnostics come before interpretation. A forecast can be
useful with a misspecified error process, because the point prediction may still be close. A
coefficient cannot be interpreted with one, because the standard error attached to it is not
the standard error of anything.

**The claim of the topic.** A time-series regression is a claim about three things at once: the
shape of time, the calendar, and the causal window. Get the calendar wrong and the seasonal
coefficients absorb holidays. Get the trend wrong and the coefficients are spurious. Get the
window wrong and the intervention effect is the first lag rather than the sum.

**Where this sits.** Topic 02 gave you intervals that survive non-normal data. Topic 03 gave you
a decision rule and a family. This topic builds the design matrix those tools are applied to.
Topic 05 repairs the standard errors once the specification is right. Topic 06 draws the graph
that decides which covariates belong in the matrix at all.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** why weekly and daily supply chain series violate the independence assumption, and
   **name** the three distinct violations and their different consequences.
2. **Construct** a Fourier (harmonic) seasonal basis for a series of period `m`, by hand for
   `K = 1` and `K = 2`, and **state** what each additional pair adds and what it costs.
3. **Compare** a seasonal dummy specification with a harmonic specification on parameter count,
   data requirement, smoothness, and behaviour under a moving holiday, and **choose** between
   them for a stated series length.
4. **Build** a holiday regressor keyed to the event rather than the calendar date, including a
   window and a centering rule, for Easter, Ramadan, Chinese New Year, and Thanksgiving.
5. **Estimate** and **interpret** a finite distributed lag, reporting the cumulative effect
   with its own standard error rather than the sum of the individual standard errors.
6. **Diagnose** a trending demand series for a unit root using both a unit-root test and a
   stationarity test, and **explain** why the two can disagree and what to do when they do.
7. **Run** the diagnostic loop in the correct order and **state**, before interpreting any
   coefficient, which specification defect would invalidate the interpretation.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a single observed series is the sum of four things you cannot see separately,
and the job is to draw the boundaries between them in the right place.

```text
   y_t   OBSERVED, units per week, 52 weeks
   28000 |                  ^                    ^
         |                 / \                  / \
   26000 |                /   \       ^        /   \
         |               /     \     / \      /     \
   24000 |     ___      /       \   /   \    /       \___
         |    /   \    /         \_/     \__/            \
   22000 |   /     \__/                                    \___
         +-----------------------------------------------------> t
           1    7    13   19   25   31   37   43   49   52

   THE SAME SERIES, DECOMPOSED

   TREND      21,850 + 42 t          a straight line; +2,184 units over the year
   HARMONIC   K = 2 pairs            two smooth waves, period 52 and period 26
                wave 1: amplitude 1,798 units/week, peak week 51
                wave 2: amplitude   358 units/week, peaks weeks 17 and 43
   HOLIDAY    narrow spikes         keyed to the EVENT, not to the calendar slot
                Thanksgiving  +2,400 units/week, week 47 every year
                Christmas     -3,100 units/week, week 52 every year
                Easter        +900 units/week  <-- THE SPIKE MOVES
   RESIDUAL   what is left          should look like noise; sd about 319 units/week
```

The decomposition is not unique. Any two of the four components can absorb part of a third,
and the specification choices in Phase C to Phase F are choices about which absorption you are
willing to accept. That is why the model is a claim, not a measurement.

### 3.1 The moving holiday is the tell

A fixed holiday is easy: week 47 is Thanksgiving, every year. A moving holiday destroys the
correspondence between the calendar slot and the event, and that is what makes it diagnostic.
If your seasonal specification is a set of week-of-year dummies, a moving holiday leaks into
whichever dummy happens to be near it, in a different way each year.

```text
   EASTER SPIKE, THREE YEARS, PLOTTED ON THE SAME CALENDAR AXIS

   year 1    ....|....X....|....|....        Easter week 13
   year 2    ....|....|....|..X.|....        Easter week 16
   year 3    ....|....|X...|....|....        Easter week 14

   A dummy keyed to "week 13" fires in year 1 and misses in years 2 and 3.
   The week-13 coefficient is the average of one true spike and two zeros:
   roughly one third of the real effect.
   A regressor keyed to "the three weeks around Easter Sunday" fires every year.
```

### 3.2 Five principles

**Principle 1 — Time is a variable with a shape, not an index.** `t` enters as a number and its
coefficient is units per week per week. If the shape is wrong, every other coefficient is
contaminated, because trend and seasonality compete for the same low-frequency variation.

**Principle 2 — A seasonal dummy and a harmonic pair are the same basis at different
resolution.** A week-of-year dummy is a local, nonparametric estimate of one week's deviation.
A harmonic pair is a global, parametric estimate of one frequency. At `K = m/2` the harmonic
basis spans the same space as the full dummy set. Choosing `K` is choosing a point on a
bias-variance curve, not choosing between two theories.

**Principle 3 — A moving holiday is a regressor keyed to the event.** The date is data, not a
formula. Store the event date per year and per jurisdiction, and build the regressor from it. A
dummy keyed to the calendar slot is attenuated by the probability that the slot holds the event.

**Principle 4 — A distributed lag is a claim about a causal window.** The lag-zero coefficient
is the immediate effect; the sum is the effect. If the promotion pulls demand forward, some lags
are negative and the sum can be a small fraction of the peak. Reporting the peak as "the lift"
is the most common error in promotional analytics.

**Principle 5 — Trend has three candidate specifications, each with a different
counterfactual.** No trend reverts to a constant mean. A deterministic trend reverts to a moving
line. A unit root says shocks persist and the level has no anchor. The three give different
forecasts, intervals, and causal claims, and the data often cannot separate them.

```text
   THREE TREND SPECIFICATIONS, SAME DATA

   none            y_t = mu + ...              reverts to a constant
                   .......................
                   ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

   deterministic   y_t = mu + beta t + ...    reverts to a line
                   ..................../'
                   ~ ~ ~ ~ ~ ~ ~ ~ ~/

   stochastic      y_t = y_{t-1} + ...         no reversion; shocks persist
                   .....................
                   ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  (wanders, does not return)
```

### 3.3 What "independent" is doing in the standard error

The variance of an OLS coefficient is `sigma^2 (X'X)^-1` only when the errors are spherical.
With serial correlation the correct variance is a sandwich that includes the autocovariances.
For a first-order autoregression with parameter `rho`, the variance inflation on a mean is
roughly `(1 + rho) / (1 - rho)`: 3 at `rho = 0.5`, and 9 at `rho = 0.8`. A t-statistic of 6
computed naively is a t-statistic of 2 correctly, and the difference decides whether the
coefficient is reported as real.

> **Note:** The consequence is asymmetric. Serial correlation usually makes naive standard
> errors too small, so it manufactures significance; it rarely hides a real effect. An analyst
> who ignores it produces false positives, not false negatives.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A definition without one is the defect
this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Time series | A sequence of observations indexed by time, where order carries information | Treated as a sample of independent rows | Observations at a stated frequency |
| Serial correlation | Correlation between a series and its own past values | Used interchangeably with autocorrelation without saying which | Correlation coefficient at a stated lag |
| Autocorrelation function (ACF) | The correlation of a series with its lagged self, as a function of lag | Read without stating the sampling frequency | Dimensionless, lag in periods |
| Partial autocorrelation (PACF) | The correlation at lag k after removing the effect of shorter lags | Interpreted as a causal statement about lag k | Dimensionless, lag in periods |
| White noise | A sequence with zero mean, constant variance, and zero autocorrelation | Confused with "random" or "unpredictable" | Residual series with stated sd |
| Stationarity | A property of the data-generating process, not of the sample: constant mean, variance, and autocovariance | Asserted from a single plot | Property of a process |
| Weak stationarity | Constant mean and variance, autocovariance depending only on lag | Called "stationary" without qualification | Mean, variance, ACF |
| Trend-stationary | Stationary deviations around a deterministic function of time | Confused with difference-stationary | Deterministic trend plus stationary error |
| Difference-stationary | Stationary only after differencing; contains a unit root | Called "non-stationary" and left there | Order of integration `d` |
| Unit root | A root of the autoregressive polynomial on the unit circle; shocks do not decay | Used as a synonym for "trending" | Root at 1, or `d = 1` |
| Order of integration, I(d) | The number of differences needed for stationarity | Reported without the test that produced it | Integer `d` |
| Random walk | `y_t = y_{t-1} + e_t`; the canonical unit-root process | Treated as a model of demand rather than a null | Units per period |
| Spurious regression | A regression between independent non-stationary series that yields high R-squared and significant t-statistics | Diagnosed from R-squared alone | R-squared, t, Durbin-Watson |
| Cointegration | Two I(1) series whose linear combination is stationary | Assumed whenever two series trend together | Residual order of integration |
| Fourier term | A sine or cosine at a harmonic of the fundamental frequency | Called a "seasonal dummy" | Dimensionless, in [-1, 1] |
| Harmonic | A frequency that is an integer multiple of the fundamental | Confused with the fundamental itself | Integer order `k` |
| Fundamental frequency | One cycle per seasonal period, `1/m` cycles per observation | Omitted from the definition of `k` | Cycles per observation |
| Harmonic order, K | The number of sine-cosine pairs included in the model | Reported without the seasonal period it applies to | Integer; 2K columns |
| Seasonal period, m | The number of observations in one seasonal cycle | Assumed to be 12 for all data | Observations per cycle |
| Amplitude | The peak deviation of a harmonic from its mean, `sqrt(a^2 + b^2)` | Read as the coefficient on the sine term | Outcome units |
| Phase | The horizontal shift of a harmonic, `atan2(b, a)` | Ignored, so the peak week is misread | Radians; convert to weeks |
| Nyquist frequency | The highest frequency identifiable at a given sampling rate, one cycle per two observations | Exceeded by including `K > m/2` | Cycles per observation |
| Seasonal dummy | A 0/1 indicator for one position in the seasonal cycle | Used with `m` levels and an intercept | 0 or 1 |
| Seasonal index | A multiplicative or additive factor by position in the cycle | Estimated on inconsistent bucket lengths | Dimensionless |
| Moving holiday | A holiday whose Gregorian date changes from year to year | Treated as a fixed date | Event date per year, per jurisdiction |
| Fixed holiday | A holiday on a constant Gregorian date | Assumed to fall in the same week each year | Month and day |
| Holiday window | The span of days or weeks around an event over which its effect acts | Set to one week by default | Days before and after the event |
| Proximity regressor | A regressor whose value is the share of a holiday window falling in the period | Confused with a dummy | Proportion in [0, 1] |
| Trading day effect | Variation caused by the number and weekday composition of days in a period | Ignored in monthly data | Count of weekdays per period |
| Distributed lag | A model in which the outcome depends on current and past values of a regressor | Reported as a single contemporaneous coefficient | Coefficients by lag |
| Lag weight | The coefficient on one lag of the regressor | Interpreted as the total effect | Outcome units per regressor unit |
| Cumulative effect | The sum of lag weights from zero to the chosen horizon | Computed by adding standard errors | Outcome units per regressor unit |
| Impulse response | The path of the outcome after a one-unit shock to the regressor | Confused with the lag weights when the regressor is persistent | Outcome units by horizon |
| Koyck lag | An infinite lag with geometrically declining weights, `delta_j = delta_0 lambda^j` | Estimated by OLS with a lagged dependent variable without noting the bias | Two parameters, `delta_0` and `lambda` |
| Almon lag | A finite lag whose weights lie on a low-order polynomial | Used without checking the endpoint restrictions | Polynomial degree plus horizon |
| Lag order | The maximum lag included in the model | Chosen by inspecting t-statistics | Integer, in periods |
| Akaike information criterion (AIC) | `n ln(RSS/n) + 2p`; a penalised fit measure | Compared across models with different samples | Dimensionless |
| Bayesian information criterion (BIC) | `n ln(RSS/n) + p ln(n)`; a consistent penalised fit measure | Treated as interchangeable with AIC | Dimensionless |
| Durbin-Watson statistic | A test statistic for first-order autocorrelation, near 2 when absent | Reported without its inconclusive zone | Statistic in [0, 4] |
| Ljung-Box statistic | A portmanteau test for autocorrelation up to a stated lag | Run at a single lag chosen after seeing the ACF | Chi-squared statistic with degrees of freedom |
| Augmented Dickey-Fuller (ADF) test | A test whose null is a unit root | Read as proof of a unit root when not rejected | Test statistic against critical values |
| KPSS test | A test whose null is stationarity | Read as a unit-root test in the same direction as ADF | Lagrange multiplier statistic |
| HAC standard error | A heteroskedasticity- and autocorrelation-consistent standard error | Assumed to fix a misspecified mean equation | Standard error with a bandwidth |

Five of these deserve expansion, because they carry most of the interpretive weight.

**Stationarity is a property of the process, not the sample.** You never observe it. You observe
a finite path and you test a hypothesis about the process that could have generated it. A test
that fails to reject a unit root has not found one; it has failed to rule one out, and the power
of the test depends on the sample length, the trend specification, and the amount of
low-frequency variation the seasonal component has left behind.

**Amplitude versus coefficient.** A harmonic pair is two coefficients, `a` and `b`, and neither
is the amplitude. The amplitude is `sqrt(a^2 + b^2)`, and the peak occurs where the phase puts
it. An analyst who reports "the seasonal coefficient is 1,785" has reported a number that
depends on where the series starts, because shifting the time origin rotates `a` and `b` into
each other. The amplitude and the peak week are invariant. Report those.

**Cumulative effect versus sum of standard errors.** The variance of a sum is not the sum of
the variances unless the terms are uncorrelated. Lag weights from the same regression are
almost always negatively correlated, because the fit can trade one lag against the next. The
correct standard error comes from the full covariance matrix, and it is usually smaller than
the naive sum. Section 6.6 works this through.

**AIC versus BIC.** Both penalise fit by parameter count. AIC penalises each parameter by 2;
BIC by `ln(n)`, which is 3.95 at `n = 52` and 4.61 at `n = 101`. BIC is consistent: it selects
the true model with probability approaching one as the sample grows. AIC is not consistent and
tends to over-select, but it is asymptotically efficient for one-step prediction. For seasonal
order selection on 52 observations, the two usually agree; when they disagree, BIC is choosing
the more parsimonious model and AIC is hedging against missing a small real signal.

**Moving holiday versus fixed holiday.** A fixed holiday has two sources of movement even when
the date is constant: the weekday it falls on, and the retail week it lands in. A moving
holiday adds a third: the date itself. The modelling consequence is that a fixed holiday can be
handled with a dummy on the week containing it, provided the week assignment is computed from
the actual date each year, while a moving holiday needs the event date in the data and usually
a window.

## 5. The mechanics, step by step

This section builds a specification in eight phases. The order matters: each phase assumes the
previous one is settled, and a defect in an early phase cannot be repaired by a later one.

### Phase A — Fix the time index

**Step 1. State the bucket and its length.** Weekly, daily, four-weekly, or monthly. The bucket
determines which seasonal frequencies exist and which are averaged away. A weekly series has no
day-of-week pattern; a daily series has both day-of-week and annual patterns, plus moving
holidays that cannot be represented at weekly resolution without loss.

**Step 2. State the week definition.** ISO week, retail 4-5-4 week, or week-ending Sunday. These
are different indices, and joining them without a mapping produces a one-week misalignment that
looks like a demand anomaly. The calendar topic in 01A treats the mapping; here the requirement
is only that the same definition is used for the outcome, the promotion calendar, and the
holiday dates.

**Step 3. Handle the 53-week year explicitly.** A 53-week year is 371 days against 364, which
is 1.923% longer. Any level comparison across that boundary carries the difference. Either
exclude the extra week, restate it, or include a 53-week indicator. Do not annualise through it
silently.

**Step 4. Confirm the series has no gaps.** A missing week is not a zero. Interpolating it
creates a value that the model treats as observed, and if the gap coincides with a holiday the
model attributes the interpolation to the holiday. Carry an explicit missing indicator or drop
the week and record the drop.

**Step 5. Count the seasonal cycles.** Three full cycles is the practical minimum for estimating
a stable seasonal pattern with a parametric basis. With two cycles, the seasonal and trend
components are nearly confounded, and the estimate is a decomposition assumption rather than a
measurement.

### Phase B — Write the model as a unit-bearing claim

**Step 6. Write the equation before fitting anything.** A specification you cannot write on one
line with units is a specification you do not understand.

```text
   y_t   = b0 + b1 t
         + SUM_{k=1..K} [ a_k sin(2 pi k t / m) + c_k cos(2 pi k t / m) ]
         + SUM_{h} g_h H_{h,t}
         + SUM_{j=0..L} d_j P_{t-j}
         + e_t

   y_t     units per week
   b0      units per week
   b1      units per week per week
   a_k, c_k  units per week
   g_h     units per week per unit of holiday regressor
   d_j     units per week per unit of promotion intensity (index 0 to 1)
   e_t     units per week
```

**Step 7. State the estimand.** Which coefficient is the answer to the question? If the
question is "what did the promotion do", the estimand is `SUM d_j`, not `d_0`. If the question
is "how seasonal is this item", the estimand is the amplitude at `k = 1`. If the question is
"what is the trend", the estimand is `b1` in units per week per week. Three questions, three
coefficients, one regression.

### Phase C — Build the seasonal basis

**Step 8. Choose between dummies and harmonics.** The choice is a point on a bias-variance
curve.

| Property | Week-of-year dummies | Fourier pairs, order K |
|---|---|---|
| Parameters | `m - 1` (with intercept) | `2K` |
| Estimated from | One observation per year per dummy | All observations, at each frequency |
| Shape | Arbitrary; each week free | Smooth; sums of sine waves |
| Data requirement | At least 3 observations per dummy, ideally more | Depends on K, not on m |
| Behaviour at a moving holiday | Absorbed into whichever dummy is near it, differently each year | Absorbed into the smooth wave, also wrongly, but smoothly |
| Extrapolation | Undefined outside the observed weeks | Defined for any t |
| When to use | Long history, stable calendar, need for week-specific effects | Short history, long period, multiple seasonalities, forecasting |

**Step 9. Construct the columns.** For period `m` and order `k`, the pair is
`sin(2 pi k t / m)` and `cos(2 pi k t / m)`. Include `k = 1 ... K`. The pair at `k = 1` is one
cycle per year. The pair at `k = 2` is two cycles per year, which captures a semi-annual
pattern such as a spring and an autumn peak, not a "second harmonic of the same peak".

**Step 10. Choose K.** Start at `K = 1` and increase while the residual autocorrelation at the
seasonal lag falls and the information criterion improves. The textbook guidance is to minimise
AICc over a range, with the caveat that AICc over-selects for long seasonal periods, so a
subjective cap is often applied: `K = 5` for weekly data and `K = 10` for daily data in the
worked bank-calls example in Hyndman and Athanasopoulos. For a 52-week series, `K` between 2 and
4 is almost always sufficient for a smooth seasonal shape.

**Step 11. Check the Nyquist limit.** At weekly frequency the highest identifiable seasonal
frequency is 26 cycles per year, so `K <= 26`. At `K = 26` the basis has 52 columns and spans
the same space as 52 week dummies. Beyond that, the sine and cosine columns alias onto
frequencies already in the model and the design matrix becomes singular.

**Step 12. Check collinearity with the trend.** A harmonic at a very low frequency is nearly
collinear with a linear trend. At `m = 52`, `k = 1` is a one-cycle-per-year wave and is not
collinear with a straight line over 52 weeks, but across three years the correlation between
`t` and `sin(2 pi t / 52)` rises. Report the condition number; if it exceeds about 30, the
trend and the seasonal terms are trading off and both coefficients are unstable.

### Phase D — Add the calendar

**Step 13. Separate fixed from moving holidays.** Fixed-date holidays still move in weekday and
in retail-week assignment. Moving holidays move in date as well. Build the date table first,
then the regressor.

| Holiday | Date rule | Range | Typical window | Modelling note |
|---|---|---|---|---|
| Easter, Western | First Sunday after the first full moon on or after 21 March | Late March to late April | 8 days before to 2 days after | X-13ARIMA-SEATS provides `easter[n]`, default `n = 8` |
| Ramadan and Eid al-Fitr | Islamic lunar calendar, subject to crescent sighting | About 10 to 11 days earlier each Gregorian year | 1 to 2 weeks before Eid | Moves through all months; centre on the mean, not the calendar |
| Chinese New Year | Lunar new year, second new moon after the winter solstice | 21 January to 20 February | 3 to 4 weeks before, 1 to 2 weeks after | Production halt before, restock after; two windows |
| US Thanksgiving | Fourth Thursday of November | 22 to 28 November | Week containing it | Fixed weekday, moving date |
| Canadian Thanksgiving | Second Monday of October | 8 to 14 October | Week containing it | Different date from the US; do not share a regressor |

**Step 14. Key the regressor to the event, not the slot.** For a weekly series, find the week
containing the event date for each year and set the dummy there. For a monthly series, use a
proximity regressor: the share of the holiday window falling in the month. The share version is
what statistical agencies use, because it avoids the arbitrary assignment of a window that
straddles a period boundary.

**Step 15. Choose the window from the mechanism.** Easter affects shipping and production in the
weeks before, because the holiday closes plants and docks. Chinese New Year affects production
for weeks before, because migrant workers travel home, and restocking for weeks after.
Thanksgiving affects a single week in US retail. The window is a claim about the operational
mechanism, and it should be stated in days relative to the event, not chosen to maximise fit.

**Step 16. Centre the regressor.** A dummy that is 1 in one week and 0 elsewhere has mean
`1/m`, so it is correlated with the intercept. Statistical agencies centre holiday regressors by
subtracting the calendar mean before estimation, so the coefficient is the deviation of the
holiday week from the average week. If you do not centre, part of the holiday effect is absorbed
into the intercept and the seasonal terms, and the coefficient is not the holiday effect.

**Step 17. Do not double count.** If Christmas week is also the seasonal peak, a Christmas
dummy and a harmonic peaking in the same week compete. The model will split the effect between
them arbitrarily. Either exclude the holiday weeks from the seasonal fit, or accept that the
split is not identified and report the combined fitted value rather than the two coefficients
separately.

### Phase E — Add the intervention with a lag

**Step 18. Write the finite distributed lag.** `y_t = ... + SUM_{j=0..L} d_j x_{t-j} + e_t`.
The coefficient `d_j` is the effect of a one-unit change in `x` at time `t - j` on `y` at time
`t`, in outcome units per regressor unit. The cumulative effect to horizon `L` is `SUM d_j`.

**Step 19. Choose L from the decision process, not from the fit.** How long does a promotion
plausibly act? For a consumer staple, the pantry-loading reversal is 2 to 4 weeks. For a
durable, an promotion may pull demand forward by a quarter. For a price change, the pass-through
may take two quarters. State the mechanism, set `L` from it, and then test whether the last lag
is distinguishable from zero. Do not extend `L` until the last lag is insignificant; that is
lag-length selection by t-test, and it over-fits.

**Step 20. Consider a restricted lag if `L` is large.** The Koyck scheme sets
`d_j = d_0 lambda^j` with `0 < lambda < 1`, giving two parameters and a long-run effect of
`d_0 / (1 - lambda)`. It is estimable by including the lagged dependent variable, but that
estimator is biased and inconsistent when the error is autocorrelated, which it usually is.
The Almon scheme puts `d_j` on a polynomial of low degree, reducing parameter count and
imposing smoothness; the endpoint restrictions (weights zero before `j = 0` and after `j = L`)
must be stated, because they are assumptions, not conveniences.

**Step 21. Compute the cumulative effect and its own standard error.** With `d` the vector of
lag coefficients and `V` their covariance matrix, the cumulative effect is `1'd` and its
variance is `1'V1`. The square root is the standard error. This is one line of matrix algebra
and it is the number the business needs.

> **Note:** The individual lag coefficients are usually poorly identified even when the
> cumulative effect is well identified. A promotion that lifts sales by 1,200 in week 0 and
> gives back 550 over the next three weeks has a cumulative effect of 650, and the cumulative
> standard error can be a third of the sum of the four individual standard errors. Report the
> cumulative effect, and report the individual lags only as a shape.

### Phase F — Choose the trend

**Step 22. Plot the series at full length and at the seasonal scale.** A trend that looks
obvious over ten years may be a level shift. A trend that looks absent over one year may be
clear over five. Plot both.

**Step 23. Test, and test both ways.** The augmented Dickey-Fuller (ADF) test has a unit root as
its null. The KPSS test has stationarity as its null. Running both gives four outcomes, and the
two-by-two table is more informative than either test alone.

| ADF (H0: unit root) | KPSS (H0: stationarity) | Reading | Action |
|---|---|---|---|
| Not rejected | Rejected | Unit root | Difference, or model as I(1) with an error-correction term |
| Rejected | Not rejected | Stationary | Level regression with the seasonal and holiday terms |
| Rejected | Rejected | Trend-stationary, or a structural break | Inspect the plot; add a deterministic trend or a break |
| Not rejected | Not rejected | Uninformative: low power in both directions | Longer sample, or an economic judgement stated as an assumption |

**Step 24. Deseasonalise before testing when seasonality is strong.** A large unmodelled
seasonal component inflates the estimated persistence of a series and makes ADF fail to reject
a unit root that is not there. Fit the seasonal terms first, test the residual, and report both
the raw and the deseasonalised result. This is the single most common source of a false unit
root in demand data.

**Step 25. State the counterfactual the trend implies.** A deterministic trend says that in the
absence of the intervention, demand would have continued on the line. A unit root says the
best forecast of the future level is the current level, so the intervention's effect is
measured against no growth. These are different claims about the world, and the choice must be
defended on operational grounds, not only on a test statistic.

### Phase G — Diagnose in a fixed order

**Step 26. Plot the residuals against time.** Look for a trend, a seasonal wave, a level shift,
and a variance change. A residual plot that looks like a funnel invalidates the standard errors
before any test does.

**Step 27. Plot the ACF and PACF to at least twice the seasonal lag.** A spike at lag `m` means
the seasonal terms are insufficient. A spike at lag 1 means the dynamics are missing. A spike
at lags 3 to 5 after a promotion means the lag window is too short.

**Step 28. Run the Ljung-Box test at the seasonal lag and at twice the seasonal lag.** Report
the statistic, the degrees of freedom, and the p-value. Do not run it at a lag chosen after
looking at the ACF; that is a selected test.

**Step 29. Check coefficient stability recursively.** Estimate the model on the first 60% of the
sample, then on expanding windows, and plot each coefficient. A coefficient that moves by more
than its own standard error across windows is not stable enough to interpret.

**Step 30. Hold out a period that contains a moving holiday.** Random cross-validation is wrong
for time series: it trains on the future. Use a rolling or expanding origin, and make sure at
least one holdout fold contains an Easter or a Chinese New Year. A model that has never been
tested on a moving holiday has not been tested.

**Step 31. Compare against a seasonal naive benchmark.** The benchmark is the same week last
year. A model that cannot beat it has no business being interpreted. Report the ratio of mean
absolute error to the benchmark's, not the R-squared.

### Phase H — Interpret, then state what would change it

**Step 32. Interpret only the estimand named in Step 7.** If the estimand is the cumulative
promotion effect, do not lead with `d_0`. If the estimand is the seasonal amplitude, do not lead
with `a_1`.

**Step 33. Attach units to every number in the sentence.** "The promotion raised volume by 650
units per week of promotion, with a 95% interval of 24 to 1,276" is interpretable. "The
promotion was significant" is not.

**Step 34. State the three specification defects that would overturn the interpretation.** For
each coefficient you report, name the defect that would invalidate it: an unmodelled moving
holiday, a lag window that is too short, a trend that should have been a difference. This is
the sentence that separates an analyst from a regression output.

### 5.1 The specification ladder

Most applied work proceeds by adding terms until the diagnostics pass. That is acceptable if
each step is recorded and if the final model is compared with the simpler ones. It is not
acceptable if the ladder is climbed silently, because the reported standard errors then
conditional on a search that the reader cannot see.

| Rung | Specification | What it buys | What it costs |
|---|---|---|---|
| 1 | `y` on `x` | Nothing defensible | Omitted trend, seasonality, and autocorrelation |
| 2 | Add a linear trend | Removes the shared drift | Imposes a functional form; absorbs a level shift |
| 3 | Add week-of-year dummies | Arbitrary seasonal shape | `m - 1` parameters; unstable with short history |
| 4 | Replace dummies with Fourier terms | Parsimony; extrapolation | Smoothness imposed; moving holidays still absorbed |
| 5 | Add holiday regressors | Event-keyed calendar effects | Windows and centering are judgement calls |
| 6 | Add distributed lags | Correct causal window | More parameters; collinearity across lags |
| 7 | Model the errors, or use HAC | Valid inference | Does not repair a wrong mean equation |

Rung 7 is the subject of Topic 05. The point of the ladder is that rungs 1 to 6 are about the
mean, and no amount of rung 7 repairs a mean that is wrong.

## 6. Worked example

A regional distribution centre ships a single fast-moving consumer good to retail stores. The
series is weekly outbound volume in units per week, 52 weeks, one retail year with no 53rd
week. All figures are illustrative but internally consistent, and every calculation below can
be reproduced by hand.

### 6.1 The series and the question

The commercial question is: "Did the week-30 promotion work, and how much of the December
decline is real?" Two estimands follow. The first is the cumulative effect of the promotion on
volume. The second is the December level after removing trend, seasonality, and the calendar.

### 6.2 The estimated specification

```text
   y_t = 21,850 + 42.0 t
       - 215.0 sin(2 pi t / 52) + 1,785.0 cos(2 pi t / 52)
       + 310.0 sin(4 pi t / 52) -   180.0 cos(4 pi t / 52)
       + 2,400 Thanksgiving_t - 3,100 Christmas_t + 900 Easter_window_t
       + 1,200 P_t + 480 P_{t-1} - 620 P_{t-2} - 410 P_{t-3}
       + e_t
```

| Term | Estimate | HAC standard error | t | Units |
|---|---:|---:|---:|---|
| Intercept | 21,850 | 410 | 53.29 | units per week |
| Trend | 42.0 | 6.8 | 6.18 | units per week per week |
| `a_1` | -215.0 | 78.0 | -2.76 | units per week |
| `b_1` | 1,785.0 | 82.0 | 21.77 | units per week |
| `a_2` | 310.0 | 76.0 | 4.08 | units per week |
| `b_2` | -180.0 | 74.0 | -2.43 | units per week |
| Thanksgiving | 2,400 | 190 | 12.63 | units per week |
| Christmas | -3,100 | 210 | -14.76 | units per week |
| Easter window | 900 | 240 | 3.75 | units per week at peak weight |
| `d_0` | 1,200 | 260 | 4.62 | units per week per unit promotion |
| `d_1` | 480 | 240 | 2.00 | units per week per unit promotion |
| `d_2` | -620 | 235 | -2.64 | units per week per unit promotion |
| `d_3` | -410 | 230 | -1.78 | units per week per unit promotion |

`n = 52`, parameters `p = 13`, residual degrees of freedom 39, residual standard deviation 319
units per week, R-squared 0.918, adjusted R-squared 0.893. The Easter regressor is a
three-week window with weights 0.5, 1.0, 0.5 centred on the week containing Easter Sunday, so
the coefficient is the effect in the peak week.

### 6.3 Hand evaluation at six weeks

Take the fitted equation and evaluate it. This is the check that catches sign errors, phase
errors, and double counting.

| Week `t` | Trend `21,850 + 42t` | Harmonic 1 | Harmonic 2 | Holiday | Fitted | Observed | Residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 22,396 | -35.0 | 0.0 | +900 | 23,261 | 23,510 | +249 |
| 26 | 22,942 | -1,785.0 | -180.0 | 0 | 20,977 | 20,810 | -167 |
| 39 | 23,488 | +215.0 | +180.0 | 0 | 23,883 | 24,120 | +237 |
| 47 | 23,824 | +1,591.2 | -353.7 | +2,400 | 27,461 | 27,900 | +439 |
| 51 | 23,992 | +1,797.9 | -249.0 | 0 | 25,541 | 25,300 | -241 |
| 52 | 24,034 | +1,785.0 | -180.0 | -3,100 | 22,539 | 22,600 | +61 |

Read two rows closely. At week 51 the harmonic seasonal peaks at +1,797.9 units per week, and
there is no holiday term. At week 52 the harmonic is almost as high at +1,605.0, but the
Christmas regressor subtracts 3,100, so the fitted value falls by 3,002 units per week in one
week. The observed series falls by 2,700. That is the shape a distribution centre shows when
it ships hard into the last full week before Christmas and then closes.

At week 13 the harmonic contribution is -35.0, which is close to the annual average, so almost
all of the fitted elevation is the Easter window. If you had used a week-of-year dummy instead,
that dummy would carry both the Easter effect and the local seasonal level, in a proportion
that changes every year.

### 6.4 What the Fourier coefficients mean

The pair `(a_k, b_k)` is not two effects. It is one wave, described by an amplitude and a
phase.

```text
   amplitude      R_k = sqrt(a_k^2 + b_k^2)
   phase          phi_k = atan2(b_k, a_k)
   peak position  t_peak = (m / 2 pi) (pi/2 - phi_k)  modulo m/k
```

For the first harmonic:

```text
R_1 = sqrt((-215.0)^2 + 1,785.0^2)
    = sqrt(46,225 + 3,186,225)
    = sqrt(3,232,450)
    = 1,797.90 units per week

phi_1 = atan2(1,785.0, -215.0)
      = pi - atan(1,785.0 / 215.0)
      = 3.14159 - 1.45087
      = 1.69072 radians

t_peak = (52 / 6.28319) x (1.57080 - 1.69072)
       = 8.2765 x (-0.11992)
       = -0.992  ->  week 51.0
```

For the second harmonic:

```text
R_2 = sqrt(310.0^2 + (-180.0)^2) = sqrt(96,100 + 32,400) = sqrt(128,500)
    = 358.47 units per week

phi_2 = atan2(-180.0, 310.0) = -0.52625 radians
t_peak = (26 / 6.28319) x (1.57080 + 0.52625)
       = 4.1382 x 2.09705
       = 8.68  ->  weeks 17.4 and 43.4
```

The correct sentence is "seasonal volume peaks in week 51 at 1,798 units per week above the
trend, with a secondary peak in weeks 17 and 43 of 358 units per week". The incorrect sentence
is "the sine coefficient is -215". The second number is not invariant to where the year is
deemed to start.

### 6.5 Choosing K

Four models were estimated, differing only in `K`. The parameter count includes the intercept,
the trend, three holiday regressors, and four lag weights, so `p = 9 + 2K`.

| K | Parameters `p` | RSS | `ln(RSS/n)` | AIC | BIC |
|---:|---:|---:|---:|---:|---:|
| 1 | 11 | 4,820,000 | 11.4370 | 616.72 | 638.19 |
| 2 | 13 | 3,960,000 | 11.2407 | 610.52 | 635.88 |
| 3 | 15 | 3,910,000 | 11.2278 | 613.85 | 643.12 |
| 4 | 17 | 3,895,000 | 11.2240 | 617.65 | 650.82 |

Worked for `K = 2`:

```text
RSS / n        = 3,960,000 / 52 = 76,153.85
ln(76,153.85)  = 11.2407
n x ln(RSS/n)  = 52 x 11.2407 = 584.52
AIC = 584.52 + 2 x 13 = 584.52 + 26.00 = 610.52
BIC = 584.52 + 13 x ln(52) = 584.52 + 13 x 3.9512
    = 584.52 + 51.37 = 635.89
```

Both criteria choose `K = 2`. The gain from `K = 1` to `K = 2` is 22.6 RSS points per parameter,
which is large. The gain from `K = 2` to `K = 3` is 50,000 RSS for two parameters, which AIC
prices at 3.33 and BIC at 7.31, and neither is worth it. `K = 4` adds 15,000 RSS for two more
parameters and is worse on both criteria.

> **Note:** The `K = 3` improvement is not zero, and on a different year it might cross the
> AIC threshold. This is the ordinary fragility of order selection on 52 observations. The
> defence is not a better criterion but a stability check: refit on the first 39 weeks and
> confirm that the chosen `K` does not change.

### 6.6 The promotion and its hangover

The promotion ran in week 30 at full depth, coded `P_30 = 1.0`. Four lag weights were
estimated.

| Lag `j` | Week | `d_j` | SE | t | Cumulative |
|---:|---:|---:|---:|---:|---:|
| 0 | 30 | +1,200 | 260 | 4.62 | 1,200 |
| 1 | 31 | +480 | 240 | 2.00 | 1,680 |
| 2 | 32 | -620 | 235 | -2.64 | 1,060 |
| 3 | 33 | -410 | 230 | -1.78 | 650 |

The shape is the mechanism. Week 30 and week 31 gain, because the price cut brings forward
purchases and some of them are incremental. Weeks 32 and 33 lose, because households work
through the stock they bought. The cumulative effect is 650 units per week of promotion.

The standard error of the cumulative effect is not the sum of the four standard errors, which
would be `260 + 240 + 235 + 230 = 965`. It is computed from the covariance matrix:

```text
Var(sum d_j) = 1' V 1
             = 310^2
             = 96,100
SE(sum d_j)  = 310 units per week
```

The 95% interval with 39 degrees of freedom uses `t = 2.022`:

```text
650 +/- 2.022 x 310  =  650 +/- 627  =  [23, 1,277] units per week
```

Three sentences follow, and only the first is safe.

1. "The promotion raised net volume by 650 units per week of promotion, with a 95% interval of
   23 to 1,277." Correct, and it says the effect is probably positive but imprecisely measured.
2. "The promotion raised volume by 1,200 units per week." Wrong. That is the peak, and 550
   units of it were given back.
3. "The promotion had no effect, because the last lag is not significant." Wrong in a different
   way. The last lag is insignificant, and the cumulative effect is significant.

The naive interval using the sum of standard errors would be `650 +/- 1,951`, which contains
large negative values and would support the false conclusion that the promotion destroyed
volume. The covariance between lag weights is negative, because the fit trades one week
against the next, and ignoring it inflates the interval by a factor of three.

### 6.7 The moving holiday, quantified

Suppose the Easter regressor had been a dummy on week 13, and the event fell as follows.

| Year | Week containing Easter Sunday | Week-13 dummy fires | Contribution recorded at week 13 | Where the true effect goes |
|---|---:|---:|---:|---|
| Y1 | 13 | Yes | 900 | Recorded correctly |
| Y2 | 14 | No | 0 | Week 14, absorbed into residual |
| Y3 | 14 | No | 0 | Week 14, absorbed into residual |
| Y4 | 15 | No | 0 | Week 15, absorbed into residual |
| Y5 | 13 | Yes | 900 | Recorded correctly |

The dummy is 1 in two years out of five, so its expected coefficient is:

```text
E[gamma_hat] = 900 x (2 / 5) = 360 units per week
```

The estimate is attenuated to 40% of the true effect, and the standard error is computed as if
the regressor were correct. The two years of Easter effect in week 14 do not disappear; they
sit in the residual, inflate the residual variance, and make every other coefficient less
precise. This is why a moving holiday is not a nuisance to be absorbed but a specification
error that contaminates the whole fit.

The repair is the three-week window keyed to the event date. With weights 0.5, 1.0, 0.5, every
year contributes, the coefficient is the peak-week effect, and the adjacent weeks are modelled
rather than left in the residual.

> **Note:** A window does not make the regressor orthogonal to the seasonal terms. If Easter
> always falls near a seasonal trough, the window and the harmonic will share the effect. Check
> the correlation between the holiday regressor and the seasonal columns, and report it.

### 6.8 Trend and stationarity

The series rises from about 21,900 to about 24,000 units per week over the year, which is a 9.6%
increase. Three tests were run.

| Test | Series | Statistic | 5% critical value | Decision |
|---|---|---:|---:|---|
| ADF, intercept, 4 lags | Levels | -2.31 | -2.93 | Fail to reject unit root |
| KPSS, intercept, 4 lags | Levels | 0.61 | 0.463 | Reject stationarity |
| ADF, intercept, 4 lags | After removing the fitted seasonal | -4.12 | -2.93 | Reject unit root |
| KPSS, intercept, 4 lags | After removing the fitted seasonal | 0.21 | 0.463 | Fail to reject stationarity |

The raw tests agree on a unit root. The deseasonalised tests agree on stationarity. The
resolution is that the seasonal component, which has an amplitude of 1,798 units per week
against a trend of 2,184 units per week over the whole year, dominates the low-frequency
variation that the unit-root tests are trying to read. ADF interprets a slow seasonal swing as
persistence and fails to reject. This is a known low-power problem, and it is the reason the
KPSS paper argues that failure to reject a unit root may reflect the test rather than the data.

The specification decision: treat the series as trend-stationary, include the linear trend,
and state the counterfactual explicitly. If the business believes the category is in secular
decline, the trend should be re-estimated on a longer window and the counterfactual changed.

### 6.9 Diagnostics before interpretation

| Check | Result | Reading |
|---|---|---|
| Residual plot | No trend, no funnel, one outlier at week 30 | Acceptable; the outlier is the promotion week and is modelled |
| ACF at lag 1 | 0.11 | Within the 95% band of `1.96 / sqrt(52) = 0.272` |
| ACF at lag 52 | -0.06 | No annual residue |
| Ljung-Box, 8 lags | Q = 9.4, df = 8, p = 0.31 | No evidence of remaining autocorrelation |
| Ljung-Box, 52 lags | Q = 14.8, df = 52, p = 0.79 | No evidence of annual residue; note the low power at this lag |
| Recursive `d_0` | 1,200 +/- 170 across the last 26 origins | Stable within one standard error |
| Holdout, last 13 weeks | MAE 2.9% of mean, seasonal naive 6.1% | Beats the benchmark by a factor of 2.1 |

The holdout contains Christmas and New Year, which is the point. A holdout that avoids the
moving parts of the calendar tests nothing that matters.

### 6.10 The sentence you are allowed to write

"The week-30 promotion raised net volume by 650 units per week of promotion (95% CI 23 to
1,277), with a peak effect of 1,200 in the promotion week and a give-back of 550 over the
following three weeks. The December level is 3,100 units per week below the seasonal norm,
which is the Christmas closure and not a demand decline."

Three things would overturn the second sentence: a Christmas regressor that should have been a
two-week window; a 53-week prior year that was not restated; and a December promotion that was
not in the model. The first is a specification defect, the second is a calendar defect, and the
third is an omitted variable. All three are checkable, and none is checkable from the
regression output alone.

## 7. Code, query, or calculation

One compact specification in Python, using `statsmodels`. The point is the design matrix, not
the library.

```python
import numpy as np, pandas as pd, statsmodels.api as sm

m = 52
d = df.sort_values("week").reset_index(drop=True)
d["t"] = np.arange(1, len(d) + 1)

cols = {"const": np.ones(len(d)), "trend": d["t"].astype(float)}
for k in (1, 2):                                   # K = 2 harmonic pairs
    cols[f"sin{k}"] = np.sin(2 * np.pi * k * d["t"] / m)
    cols[f"cos{k}"] = np.cos(2 * np.pi * k * d["t"] / m)
cols["thanks"] = d["week"].eq(47).astype(float)
cols["xmas"] = d["week"].eq(52).astype(float)
cols["easter"] = 0.5 * d["easter_m1"] + d["easter_0"] + 0.5 * d["easter_p1"]
for j in range(4):                                 # lags 0..3 weeks
    cols[f"promo_l{j}"] = d["promo"].shift(j).fillna(0.0)

X = pd.DataFrame(cols)
fit = sm.OLS(d["units"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
lags = ["promo_l0", "promo_l1", "promo_l2", "promo_l3"]
print(fit.params[lags].sum(), fit.cov_params().loc[lags, lags].values.sum() ** 0.5)
```

Three things to notice. The `easter_*` columns are built from the holiday date table, not from
the observed sales pattern; if they were built from sales, the regressor would contain the
outcome. The `shift(j)` uses the promotion calendar, which is known before the week, so it is
not leakage. And `cov_type="HAC"` is a placeholder for the fuller treatment in Topic 05; it
changes the standard errors and not the coefficients, so it cannot repair a wrong mean
equation.

## 8. Data, standards, and regulatory context

Time-series regression sits on top of reference data that is more governed than most analysts
expect. Holiday dates are not a formula. They are a published calendar with a version, and in
several jurisdictions they are amended by annual decree.

### 8.1 Calendar and holiday reference data

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 8601 | Date and time representation | The week date and ordinal date used as join keys | B |
| IANA time zone database | Offset rules by region | Which local date an event falls on | B |
| National statistical office holiday calendars | Public holidays by jurisdiction | The event dates used to build regressors | B |
| NRF retail calendar | 4-5-4 fiscal weeks | The week index for retail series, including the 53rd week | C |
| X-13ARIMA-SEATS built-in regressors | `easter[n]`, `labor[n]`, `thank[n]`, `easterstock[n]` | The default window and centering for common holidays | B |
| Genhol utility | User-defined moving-holiday regressors | Window bounds, proportionality versus count, centering | B |

The practical consequence: a holiday table is a versioned dataset, not a function. Easter can
be computed from the computus, but Ramadan depends on observance, and one-off national holidays,
moved holidays, and substitute holidays are announced. A model that derives holidays from a
formula will be wrong in the years that matter, which are the unusual ones.

### 8.2 Seasonal adjustment practice

Official statistical agencies treat moving holidays as a regression-ARIMA pre-adjustment
problem. The Census Bureau's X-13ARIMA-SEATS includes a default Easter regressor with an
eight-day window before Easter, a `thank[n]` regressor for Thanksgiving, and a `labor[n]`
regressor for Labor Day. For holidays the built-in set does not cover, the Genhol utility
generates a user-defined regressor from a file of dates, with a window defined in days before
and after the event and a choice between a proportionality regressor and a count regressor.

Two details from that documentation transfer directly to commercial work. First, holiday
regressors are **centered** before estimation, either on the calendar mean or on the overall
mean, because an uncentered dummy is correlated with the intercept. Second, the window is
specified in days relative to the event, so the regressor follows the event when the date
moves. Both are conventions that a commercial regression should copy.

### 8.3 What is convention rather than requirement

- A one-week holiday window is a default, not a finding. Easter and Chinese New Year usually
  need longer windows in one direction.
- `K = 6` for monthly data is the point at which the harmonic basis equals the dummy basis, not
  a recommended value.
- A linear trend is the default in most software and the least defensible default in most
  demand series.
- `K = 5` for weekly and `K = 10` for daily are textbook rules of thumb from one applied
  example, not derived optima.
- The 4-5-4 retail calendar is an industry convention with real accounting consequences, not a
  statistical standard.

An analyst who can separate a requirement from a convention can defend a specification. An
analyst who cannot will treat every software default as a physical constant.

## 9. Industry practice and case evidence

### 9.1 What statistical agencies do, and why it transfers

The X-13ARIMA-SEATS documentation is the most rigorous public statement of how moving holidays
are handled in production. The regressor is built from a date file, the window is specified in
days relative to the event, the regressor is centered, and the choice between proportionality
and count is made explicit. The Australian Bureau of Statistics has used a two-part Easter
regressor, with separate before and after windows, because the effect of Easter on retail and
on production does not have the same shape on both sides of the holiday.

The transfer to commercial demand is direct. A distribution centre shipping into retail has a
pre-Easter build and a post-Easter lull, and the two are not symmetric. A single dummy forces
them to be.

### 9.2 What the forecasting literature recommends

Hyndman and Athanasopoulos recommend dynamic harmonic regression when the seasonal period is
long: Fourier terms for the seasonality, plus an ARIMA error process for the short-term
dynamics. They give three reasons. First, seasonal ARIMA with a period of 52 or 365 requires
`m - 1` initial seasonal states and becomes impractical. Second, Fourier terms handle multiple
seasonal periods by adding pairs at different frequencies. Third, the smoothness of the
seasonal pattern is controlled by `K` rather than by the data.

They also state the main limitation plainly: the seasonality is assumed fixed over time. A
harmonic basis cannot let the seasonal shape evolve, which is a real constraint for products
in a growth or decline phase, and a reason to check the stability of the seasonal coefficients
across years.

### 9.3 What the evidence says about lag length

Almon's 1965 application is still the clearest worked example of choosing a lag from a
mechanism rather than from a fit statistic. Estimating the distributed lag between capital
appropriations and capital expenditures on quarterly US manufacturing data, she found weights
centred on eight to nine quarters, with roughly 45% of the effect realised within a year and
about 92% within two years. The polynomial restriction was what made the estimation possible;
the economic question was what set the horizon.

The choice among lag orders is a model-selection problem with a known trade-off. AIC is not
consistent and tends to over-select; BIC is consistent and selects the true order with
probability approaching one as the sample grows, at the cost of under-selecting in small
samples. Koehler and Murphree compared the two criteria for order selection in 1988 and found
the expected pattern: BIC more parsimonious, AIC better at short-horizon prediction. For a
52-week demand series with four to eight candidate lags, the two usually agree, and disagreement
is a signal that the sample is too short to distinguish the orders.

The deeper problem is the one the criteria do not address. Selecting the lag order by
inspecting t-statistics and then reporting the final model's standard errors as if the order
were known in advance understates the uncertainty. The reported interval is conditional on a
search, and the search is invisible. The practical remedy is to fix the horizon from the
mechanism before estimation and to report the sensitivity of the cumulative effect to `L`.

### 9.4 The Chinese New Year problem in practice

The Reserve Bank of Australia's work on seasonal adjustment of Chinese economic statistics is a
useful documented case. Chinese New Year falls between 21 January and 20 February on the
Gregorian calendar, a drift of about four weeks, and the RBA notes that year-on-year growth
rates are distorted by it. The same problem appears in any global demand series with a Chinese
supply base: January and February shipments depend on where the holiday fell, and a
year-on-year comparison across a two-week shift is comparing different production calendars.

The standard remedy is the interval regressor: the value in a period is the share of the
holiday window falling in that period, with separate before, during, and after windows. This is
the approach the Census documentation attributes to Bell and Hillmer, and it is implemented in
Genhol. The commercial version is usually simpler and should not be: a single dummy on "the
week containing Chinese New Year" captures the closure but not the pre-holiday build or the
post-holiday restock, and those are the parts that matter for a supply plan.

### 9.5 Vendor and practitioner claims

Vendor case studies in this area are common and should be read as marketing until replicated.
One planning vendor reports that a machine-learning uplift model for Chinese New Year improved
forecast accuracy by 3.5 percentage points and reduced bias by 44.6 percentage points in a
client engagement. The figures are not independently verified, the baseline is not described in
enough detail to reproduce, and the sample is one client. Treat it as an existence claim: it is
possible to improve on a naive holiday dummy. Do not treat the effect size as transferable.
Grade D.

The same caution applies to any claim that a harmonic specification "captures" seasonality. A
harmonic basis is a smoother; whether it captures the seasonality that matters is an
out-of-sample question, and the answer differs by item.

### 9.6 What practitioners do versus what the guidance says

| Recommendation | Observed practice | Why the gap persists |
|---|---|---|
| Key holidays to event dates | Dummy on a calendar week or month | The date table is maintained by a different team, or not at all |
| Use a window for moving holidays | One-week dummy | Window length requires a mechanism argument nobody owns |
| Centre holiday regressors | Uncentered dummy | Centering changes the intercept, which somebody has already reported |
| Choose `K` by information criterion and stability | `K` fixed by template | Re-estimating per series is expensive at scale |
| Test for a unit root both ways | ADF only, or neither | KPSS is not in the default software path |
| Report the cumulative lag effect | Report the contemporaneous coefficient | The contemporaneous coefficient is the one the dashboard exposes |
| Hold out a period containing a moving holiday | Random cross-validation | The tooling defaults to random splits |

Each row is an organisational fact before it is a technical one. The analyst who knows why the
convention exists can argue for changing it; the analyst who does not will inherit the default
and call it a specification.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Regressing two trending series in levels | R-squared above 0.9 with a Durbin-Watson below 0.5 | Both series inherit a common stochastic trend; the t-statistic diverges | A significant coefficient between unrelated series | Difference, detrend, or test for cointegration before interpreting |
| Holiday dummy keyed to a calendar slot | Holiday coefficient smaller than the operational effect implies | The dummy fires only in years when the event lands in that slot | Attenuated estimate and contaminated residuals | Build from the event date table, with a window |
| One-week window for a multi-week effect | Residual spike in the week after the holiday | The unmodelled part of the effect sits in the residual | Inflated residual variance; other coefficients less precise | Set the window from the mechanism; test adjacent weeks |
| Uncentered holiday regressor | Intercept shifts when a holiday is added | The dummy has a nonzero mean and competes with the intercept | Holiday effect partly absorbed into the level | Centre on the calendar mean, as statistical agencies do |
| Seasonal dummies on a 53-week year | Week 53 shows an implausible level | The fiscal year has 371 days against 364 | A 1.9% level bias that looks like demand | Restate, exclude, or include a 53-week indicator |
| `K` chosen by residual ACF alone | `K` rises until the ACF is white | Each added pair fits noise as well as signal | Over-fit seasonality; poor out-of-sample behaviour | AIC or BIC plus a stability check across origins |
| Lag length chosen by t-statistics | The last lag is just insignificant | Sequential testing selects on noise | Over-fit dynamics; understated uncertainty | Fix the horizon from the mechanism; report sensitivity |
| Reporting `d_0` as the promotion effect | Peak lift much larger than the volume change in the following month | Pull-forward and stockpiling give back part of the peak | A promotion appears to work and does not | Report the cumulative effect with its own standard error |
| Summing standard errors across lags | Interval implausibly wide | Lag weights are negatively correlated | A real effect is declared inconclusive | Compute `sqrt(1'V1)` from the covariance matrix |
| Naive standard errors with autocorrelated errors | t-statistics larger than the effect size warrants | The variance formula assumes spherical errors | Manufactured significance | HAC or an explicit error model; Topic 05 |
| Linear trend imposed on a unit-root series | Forecasts revert to a line the series has left | The trend is estimated from the sample and extrapolated | Systematically biased forecasts and counterfactuals | Test both ways; state the counterfactual |
| Random cross-validation | Holdout accuracy much better than live accuracy | Training on future observations | An optimistic error estimate that does not survive deployment | Rolling or expanding origin with a calendar-aware split |

Two of these deserve a note.

**Uncentered holiday regressors** is the quietest. Adding a Thanksgiving dummy to a model with
an intercept changes the intercept, and if the intercept is a reported number, somebody will
notice and ask why. The usual fix is to drop the centering, which makes the holiday coefficient
depend on the sample share of holiday weeks and makes the intercept no longer the average week.
Neither number is then interpretable in the way the reader assumes.

**Summing standard errors across lags** is the most consequential in commercial use, because it
runs in the conservative direction and therefore escapes review. An interval that is three
times too wide makes a real promotion effect look inconclusive, and the organisation concludes
that promotions cannot be measured. The measurement failure is arithmetic.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Errors are uncorrelated, or the correlation is modelled | The standard error formula depends on it | Persistent shocks, overlapping horizons, omitted dynamics | ACF and Ljung-Box on residuals; HAC as a partial repair |
| Seasonality is fixed over the sample | A harmonic basis has constant coefficients | Product life cycle, channel shift, a new competitor | Recursive seasonal coefficients; compare first and second half |
| Holiday effects are constant across years | One coefficient per holiday | A holiday that grows with e-commerce, or a one-off closure | Estimate by year where the sample allows; plot the effect |
| The trend specification is correct | Every counterfactual depends on it | A level shift read as a trend, or a trend read as a unit root | ADF and KPSS together; plot at two time scales |
| No structural break in the sample | One equation describes the whole period | New plant, new customer, a pandemic, a tariff change | Recursive residuals; Chow-type test at a known date |
| Regressors are exogenous to the error | OLS is unbiased only under this | Price set in response to demand; promotion timed to a shortfall | Topic 06's causal graph; instrument or design where not |
| The lag window covers the true response | Truncating the lag biases the sum | Long pantry-loading, durable replacement cycles | Extend `L` and check the cumulative effect, not the last t |
| Calendar alignment is correct across all inputs | The design matrix joins on the week index | Different week definitions in sales, promotion, and holiday data | Row-count and week-key reconciliation before estimation |
| At least three seasonal cycles are observed | The seasonal and trend terms must be separable | New product, short history | Count cycles, not rows |
| Promotion intensity is measured, not inferred | The regressor must not contain the outcome | Intensity coded from realised sales rather than the plan | Build `x` from the promotion calendar, never from `y` |

Two limits are worth stating plainly.

**A regression cannot tell you the calendar is wrong.** Every diagnostic in Phase G is computed
conditional on the design matrix. A moving holiday keyed to the wrong week produces residuals
that look like noise if the window is wide enough to absorb the misalignment, and the
coefficient is then a weighted average of effects in different weeks. The only reliable check is
external: compare the holiday dates in the model against the published calendar, row by row.

**This topic does not solve identification.** A well-specified time-series regression with
correct standard errors can still estimate a coefficient that is not causal, because the
regressor was chosen in response to the outcome or because a confounder moves with both. That
is Topic 06's problem, and no amount of seasonal adjustment substitutes for it.

## 12. Exercises

### Exercise 1 — Construct a Fourier basis for a monthly series (calculation, 30 minutes)

For `m = 12`, construct the design columns for `K = 1` and `K = 2` for `t = 1 ... 12`. Tabulate
the sine and cosine values to four decimal places.

**Deliverable:** a 12-row table with columns `t`, `sin1`, `cos1`, `sin2`, `cos2`.
**Acceptance criteria:** values correct to four decimals; the `k = 1` columns complete exactly
one cycle; the `k = 2` columns complete exactly two; a one-sentence statement of why
`sin(2 pi t / 12)` at `t = 1` equals `sin(30 degrees) = 0.5`.
**Time:** 30 minutes.

### Exercise 2 — Evaluate the seasonal component by hand (calculation, 45 minutes)

Using the coefficients `a_1 = -215`, `b_1 = 1,785`, `a_2 = 310`, `b_2 = -180` and `m = 52`,
compute the seasonal contribution at `t = 1, 13, 26, 39, 47, 52`.

**Deliverable:** a six-row table with the harmonic-1 value, the harmonic-2 value, and the total.
**Acceptance criteria:** arithmetic shown for at least two rows; the `t = 13` and `t = 39` results
match `-35` and `+395`; a statement of which week the fitted seasonal peaks.
**Time:** 45 minutes.

### Exercise 3 — Dummy versus harmonic, counted (analysis, 40 minutes)

For `m = 12` and `m = 52`, state the parameter count of a full set of seasonal dummies with an
intercept, and the value of `K` at which the Fourier basis spans the same space. Explain what
each basis assumes about the shape of the seasonal pattern.

**Deliverable:** a short table plus two paragraphs.
**Acceptance criteria:** the `K = m/2` equivalence stated and justified by column count; the
data requirement per parameter stated for each basis; one example of a seasonal pattern that a
low-order harmonic basis cannot represent.
**Time:** 40 minutes.

### Exercise 4 — Build an Easter alignment table (design, 45 minutes)

Take five consecutive years and, for each, record the week containing Easter Sunday in your
organisation's week convention. Build both a week-13 dummy and a three-week window regressor.
Compute the attenuation of the dummy coefficient if the true effect is 900 units per week.

**Deliverable:** a five-row alignment table and the attenuation arithmetic.
**Acceptance criteria:** the week assignment uses a stated convention; the attenuation is
computed as the share of years in which the dummy fires; the window regressor is shown to fire
in every year.
**Time:** 45 minutes.

### Exercise 5 — Cumulative effect and its standard error (calculation, 50 minutes)

A promotion model estimates `d = (800, 350, -400, -250)` with individual standard errors
`(150, 140, 135, 130)`. The covariance matrix implies `Var(sum d) = 175^2`. Compute the
cumulative effect, its 95% interval, and the interval you would get by summing the standard
errors. State which is correct and why.

**Deliverable:** the arithmetic and a two-sentence interpretation.
**Acceptance criteria:** cumulative effect 500; correct interval using the covariance-based
standard error; the naive interval shown for contrast; the sign of the covariance discussed.
**Time:** 50 minutes.

### Exercise 6 — Choose `K` by information criterion (calculation, 45 minutes)

With `n = 104`, `p = 8 + 2K`, and residual sums of squares 2,100,000 (`K = 1`), 1,720,000
(`K = 2`), and 1,690,000 (`K = 3`), compute AIC and BIC for each and state the choice.

**Deliverable:** a table with `ln(RSS/n)`, AIC, and BIC for the three models.
**Acceptance criteria:** correct arithmetic to two decimals; the chosen `K` stated for each
criterion; a sentence on what to do if the two criteria disagree.
**Time:** 45 minutes.

### Exercise 7 — Spurious regression by simulation (application, 60 minutes)

Generate two independent random walks of 52 observations. Regress one on the other and record
R-squared, the t-statistic on the slope, and the Durbin-Watson statistic. Difference both series
and repeat. Run 200 replications and record the share of level regressions with a t-statistic
above 2 in absolute value.

**Deliverable:** a short script, a table of the two specifications, and the rejection share.
**Acceptance criteria:** the data-generating process is stated; the seed is recorded; the
level-regression rejection share is far above 5%; the differenced rejection share is near 5%;
the Durbin-Watson statistic is reported for both.
**Time:** 60 minutes.

### Exercise 8 — ADF and KPSS together (analysis, 40 minutes)

You run ADF and KPSS on four series and obtain the four possible combinations. For each, state
the reading and the next action, and give one supply chain example that would produce it.

**Deliverable:** the completed two-by-two table with examples.
**Acceptance criteria:** the null of each test stated correctly; the "both reject" cell
explained by trend-stationarity or a break; the "neither rejects" cell described as
uninformative rather than as evidence of stationarity.
**Time:** 40 minutes.

### Exercise 9 — Read a residual ACF (diagnosis, 40 minutes)

A model with `K = 1` on a 52-week series leaves residual autocorrelations of 0.45 at lag 1,
0.38 at lag 2, 0.30 at lag 3, 0.12 at lag 26, and 0.41 at lag 52. Diagnose the specification and
propose two changes, ordered by which you would try first.

**Deliverable:** a one-page diagnosis with the two changes and the test that would confirm each.
**Acceptance criteria:** the lag-1 to lag-3 pattern attributed to error dynamics rather than
seasonality; the lag-52 spike attributed to the seasonal basis; the proposed order justified by
cost and by the diagnostic each change addresses.
**Time:** 40 minutes.

### Exercise 10 — Specification memo (design, 60 minutes)

Write a one-page memo specifying a regression for a weekly series of your choice, naming the
estimand, the seasonal basis and `K`, the holiday regressors and their windows, the lag horizon
and its mechanism, the trend specification, and the three defects that would invalidate the
interpretation.

**Deliverable:** a one-page memo with an equation, a design-matrix column list, and a diagnostics
plan.
**Acceptance criteria:** every coefficient carries units; the estimand is one named coefficient
or sum; the holiday windows are justified by mechanism; the trend choice is defended on both
statistical and operational grounds; the diagnostics are listed in order.
**Time:** 60 minutes.

## 13. Self-check answers

**Exercise 1 — `m = 12`, `K = 1` and `K = 2`.**

Angles are `2 pi t / 12 = 30t` degrees for `k = 1` and `60t` degrees for `k = 2`.

| `t` | `sin1` | `cos1` | `sin2` | `cos2` |
|---:|---:|---:|---:|---:|
| 1 | 0.5000 | 0.8660 | 0.8660 | 0.5000 |
| 2 | 0.8660 | 0.5000 | 0.8660 | -0.5000 |
| 3 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| 4 | 0.8660 | -0.5000 | -0.8660 | -0.5000 |
| 5 | 0.5000 | -0.8660 | -0.8660 | 0.5000 |
| 6 | 0.0000 | -1.0000 | 0.0000 | 1.0000 |
| 7 | -0.5000 | -0.8660 | 0.8660 | 0.5000 |
| 8 | -0.8660 | -0.5000 | 0.8660 | -0.5000 |
| 9 | -1.0000 | 0.0000 | 0.0000 | -1.0000 |
| 10 | -0.8660 | 0.5000 | -0.8660 | -0.5000 |
| 11 | -0.5000 | 0.8660 | -0.8660 | 0.5000 |
| 12 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |

At `t = 1`, `k = 1` gives `sin(30 degrees) = 0.5` and `cos(30 degrees) = 0.8660`. The `k = 1`
columns return to their starting values at `t = 13`, so exactly one cycle is completed in 12
observations. The `k = 2` columns repeat every 6 observations, so exactly two cycles are
completed. Note that the `k = 2` column at `t = 6` is `(0, 1)`, not `(0, -1)`: the second
harmonic is not the first harmonic shifted by half a period.

**Exercise 2 — the 52-week construction by hand.**

The two bases are:

```text
K = 1:  S1(t) = a1 sin(2 pi t / 52) + b1 cos(2 pi t / 52)
K = 2:  S2(t) = S1(t) + a2 sin(4 pi t / 52) + b2 cos(4 pi t / 52)
```

Angles at the chosen weeks, computed from `theta = 2 pi t / 52`:

| `t` | `theta` (rad) | `sin1` | `cos1` | `2 theta` (rad) | `sin2` | `cos2` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1208 | 0.1205 | 0.9927 | 0.2417 | 0.2393 | 0.9709 |
| 7 | 0.8458 | 0.7485 | 0.6631 | 1.6916 | 0.9927 | -0.1205 |
| 13 | 1.5708 | 1.0000 | 0.0000 | 3.1416 | 0.0000 | -1.0000 |
| 26 | 3.1416 | 0.0000 | -1.0000 | 6.2832 | 0.0000 | 1.0000 |
| 39 | 4.7124 | -1.0000 | 0.0000 | 9.4248 | 0.0000 | -1.0000 |
| 47 | 5.6790 | -0.5681 | 0.8230 | 11.3581 | -0.9351 | 0.3546 |
| 52 | 6.2832 | 0.0000 | 1.0000 | 12.5664 | 0.0000 | 1.0000 |

Three of these are exact and can be read off the circle. At `t = 13`, `theta = pi/2`, so
`(sin, cos) = (1, 0)` and `2 theta = pi`, so `(0, -1)`. At `t = 26`, `theta = pi`, so `(0, -1)`
and `2 theta = 2 pi`, so `(0, 1)`. At `t = 39`, `theta = 3 pi/2`, so `(-1, 0)` and
`2 theta = 3 pi`, so `(0, -1)`. The other rows require a calculator or a table.

Now evaluate:

| `t` | `S1` | `S2` | Note |
|---:|---:|---:|---|
| 1 | 1,746.1 | 1,645.5 | The second harmonic pulls the January value down |
| 7 | 1,022.7 | 1,352.1 | And pushes the February value up |
| 13 | -215.0 | -35.0 | Near the annual mean |
| 26 | -1,785.0 | -1,965.0 | The summer trough, deepened |
| 39 | 215.0 | 395.0 | Rising into autumn |
| 47 | 1,591.2 | 1,237.5 | Thanksgiving week, before the peak |
| 52 | 1,785.0 | 1,605.0 | Christmas week |

Worked longhand for `t = 47`:

```text
theta  = 2 pi x 47 / 52 = 5.6790 rad
sin1   = -0.5681,  cos1 = 0.8230
S1     = (-215.0)(-0.5681) + (1,785.0)(0.8230)
       = 122.14 + 1,469.06
       = 1,591.20

2 theta = 11.3581 rad, which is 11.3581 - 6.2832 = 5.0749 rad after one full turn
sin2    = -0.9351,  cos2 = 0.3546
S2      = 1,591.20 + (310.0)(-0.9351) + (-180.0)(0.3546)
        = 1,591.20 - 289.88 - 63.83
        = 1,237.49
```

The comparison at `t = 7` and `t = 26` is the point of the exercise. Adding the second harmonic
does not simply raise the peak. It reshapes the wave: up in early February, down in midsummer.
That is what a second frequency does, and it is why `K` cannot be chosen by asking whether the
fit "looks more seasonal".

The fitted seasonal peaks at week 51 with amplitude 1,797.9 units per week, computed as
`sqrt(215^2 + 1,785^2)`. The second harmonic peaks at weeks 17.4 and 43.4 with amplitude 358.5
units per week. Reporting the coefficients `a_1 = -215` and `b_1 = 1,785` without the amplitude
and phase hides both facts.

**Exercise 3 — counting parameters.**

| Period `m` | Dummies with intercept | Equivalent `K` | Fourier columns | Parameters saved at `K = 2` |
|---:|---:|---:|---:|---:|
| 12 | 11 + 1 = 12 | 6 | 12 | 12 - 4 = 8 |
| 52 | 51 + 1 = 52 | 26 | 52 | 52 - 4 = 48 |

The equivalence is a column-count argument. The Fourier basis at `K = m/2` has `2K = m` columns,
and the `m` week indicators also have `m` columns and are linearly independent. Two bases of the
same dimension spanning the same space represent the same fitted values, so the models are
observationally equivalent even though the coefficients mean different things.

The dummy basis assumes nothing about shape and estimates each week from the weeks observed in
that position, which is one observation per year. The harmonic basis assumes the seasonal
pattern is a sum of `K` sine waves, which is a strong smoothness assumption, and estimates it
from all observations. A seasonal pattern a low-order harmonic cannot represent: a single
one-week spike that does not repeat smoothly, such as a national holiday week that is 40% above
every neighbouring week.

**Exercise 4 — Easter alignment, model answer.**

A correct table records, for each year, the week containing Easter Sunday under a stated
convention, and marks whether a week-13 dummy fires. With two firing years out of five and a
true effect of 900 units per week:

```text
E[gamma_hat] = 900 x (2 / 5) = 360 units per week
attenuation  = 1 - 360 / 900 = 0.60, so 60% of the effect is lost
```

The three-week window fires in all five years, so no attenuation from misalignment remains.
A submission that reports only the attenuation and not where the missing effect went should be
marked down: the answer is that it sits in weeks 14 and 15, in the residual, and inflates the
residual variance.

**Exercise 5 — cumulative effect.**

```text
cumulative = 800 + 350 - 400 - 250 = 500 units per week per unit promotion

correct SE = 175
95% CI     = 500 +/- 2.0 x 175 = 500 +/- 350 = [150, 850]

naive SE   = 150 + 140 + 135 + 130 = 555
naive CI   = 500 +/- 2.0 x 555 = 500 +/- 1,110 = [-610, 1,610]
```

The correct interval is the first. The individual weights are negatively correlated: the fit
can shift effect from week 1 to week 2 without changing the fit much, so the sum is much better
determined than its components. The naive interval is 3.2 times too wide and crosses zero,
which would support the false conclusion that the promotion had no effect.

**Exercise 6 — `K` selection.**

`ln(104) = 4.6444`.

| `K` | `p` | RSS | `RSS/n` | `ln(RSS/n)` | AIC | BIC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 2,100,000 | 20,192.31 | 9.9131 | 1,050.96 | 1,077.40 |
| 2 | 12 | 1,720,000 | 16,538.46 | 9.7134 | 1,034.20 | 1,065.93 |
| 3 | 14 | 1,690,000 | 16,250.00 | 9.6959 | 1,036.37 | 1,073.39 |

Worked for `K = 2`:

```text
n ln(RSS/n) = 104 x 9.7134 = 1,010.20
AIC = 1,010.20 + 2 x 12 = 1,034.20
BIC = 1,010.20 + 12 x 4.6444 = 1,010.20 + 55.73 = 1,065.93
```

Both choose `K = 2`. If they disagreed, BIC's choice is the more parsimonious and AIC's the
better short-horizon predictor; on 104 observations the difference between them is small, and
the stability check across origins matters more than the criterion.

**Exercise 7 — what the simulation should show.**

With independent random walks, the level regression will produce an absolute t-statistic above
2 far more often than 5% of the time, typically in the range of 40% to 70% of replications
depending on the sample length and the drift. The R-squared is high and the Durbin-Watson
statistic is low, often below 0.5. After differencing, the rejection share falls to about 5%
and the Durbin-Watson statistic moves toward 2. The exact numbers vary by seed, which is why the
deliverable requires the seed.

The marking criterion is not the exact rejection share but whether the submission reports the
Durbin-Watson statistic alongside the R-squared. Granger and Newbold's original point was that
the combination of high R-squared and low Durbin-Watson is the signature of the problem, and a
submission that reports only R-squared has not demonstrated understanding.

**Exercise 8 — the two-by-two.**

| ADF | KPSS | Reading | Example |
|---|---|---|---|
| Not rejected | Rejected | Unit root | Category-level demand drifting with population; a commodity price series |
| Rejected | Not rejected | Stationary | Item-level units around a stable seasonal mean with no long-run drift |
| Rejected | Rejected | Trend-stationary or break | A series with a genuine linear growth path, or a plant opening mid-sample |
| Not rejected | Not rejected | Uninformative | A short series with a large seasonal component and 40 observations |

The fourth cell is the one that is most often misread. Failing to reject both nulls does not mean
the series is stationary: it means the sample cannot distinguish the hypotheses. The correct
response is to state an assumption, justify it operationally, and test the sensitivity of the
conclusion to it.

**Exercise 9 — reading the ACF.**

The slow decay from lag 1 to lag 3 (0.45, 0.38, 0.30) is the signature of an error process with
persistence, not of seasonality. The spike at lag 52 (0.41) is the signature of an unfitted
annual pattern, and the small value at lag 26 (0.12) says the problem is at the annual
frequency, not at the semi-annual one.

Order of repair: first raise `K` from 1 to 2 and re-check lag 52, because it is one parameter
pair and it addresses the largest single autocorrelation. Second, if lag 1 to lag 3 persist,
model the error as AR(1) or use HAC standard errors, which is Topic 05's territory. Do not add
lagged dependent variables to fix the ACF and then interpret the other coefficients as if the
model were static.

**Exercise 10 — memo rubric.**

A strong memo names one estimand and computes it as a sum where the estimand is cumulative. It
gives every coefficient a unit. It states `K`, the holiday windows in days relative to the
event, the lag horizon in weeks with a mechanism, and the trend specification with a
counterfactual. Its diagnostics are ordered: residual plot, ACF to twice the seasonal lag,
Ljung-Box at `m` and `2m`, recursive coefficient stability, and a rolling-origin holdout
containing a moving holiday. Its three invalidation conditions are specific to the design, not
generic.

A weak memo lists covariates without an estimand, reports `K` as a software default, gives no
units, and treats "check the residuals" as a diagnostics plan.

## 14. Interview questions

**1. "Why can't you just regress weekly demand on price and promotion?"**

A strong answer names the three violations: serial correlation in the errors, shared calendar
shocks that are omitted, and the possibility that both series trend. It says which one biases
the coefficient and which one only biases the standard error. A weak answer says "because time
series are different" without naming a mechanism.

**2. "What does K control in a harmonic regression?"**

A strong answer says `K` is the number of sine-cosine pairs, that each pair is one frequency,
that `K` controls smoothness, and that `K = m/2` reproduces the full dummy basis. It mentions
the bias-variance trade and the Nyquist limit. A weak answer says "the number of seasons" or
"how seasonal the model is".

**3. "When would you use seasonal dummies instead of Fourier terms?"**

A strong answer says dummies when history is long, the calendar is stable, and week-specific
effects matter; harmonics when the period is long, history is short, there are multiple
seasonalities, or you need to forecast beyond the observed weeks. It notes the parameter count.
A weak answer says "dummies are more accurate" without a condition.

**4. "How do you handle Easter in a weekly model?"**

A strong answer says: get the date from a calendar table, find the week containing the event
each year, use a window of several weeks with weights because the effect builds and decays, and
centre the regressor. It mentions that a dummy on a fixed week is attenuated. A weak answer says
"add a dummy for Easter week" and stops.

**5. "Your company sources from China and sells in Europe. How does Chinese New Year enter the model?"**

A strong answer distinguishes the supply-side effect (production halt, pre-holiday build) from
the demand-side effect, notes the date moves between 21 January and 20 February, and proposes
separate before and after windows, possibly on different series (shipments versus orders). It
mentions that a single dummy will not capture a four-week drift. A weak answer treats Chinese
New Year as a fixed holiday.

**6. "A promotion shows a 1,200-unit lift in the promotion week, but the following month is flat. What is the effect?"**

A strong answer says the effect is the cumulative sum of the lag weights, that the peak is not
the effect, and that the sum requires its own standard error from the covariance matrix. It
notes that the give-back may be stockpiling or pantry loading. A weak answer reports 1,200 or
says the promotion "did not work" because the following month was flat.

**7. "How do you choose the lag length?"**

A strong answer fixes the horizon from the decision process or the mechanism, then tests
sensitivity, and mentions AIC or BIC as a check rather than as the primary rule. It notes that
selecting by t-statistics invalidates the reported standard errors. A weak answer says "AIC
picks it".

**8. "Two series both grow over ten years and your regression gives R-squared 0.94 with t = 18. What do you check?"**

A strong answer says the Durbin-Watson statistic first, then a unit-root test on both series,
then cointegration if both are I(1). It says the high t may be an artifact of non-stationarity
and that the coefficient has no causal meaning until that is resolved. A weak answer says "the
sample is large, so it is probably real".

**9. "ADF fails to reject a unit root but KPSS rejects stationarity. What now?"**

A strong answer reads the pair as evidence of a unit root, then checks whether an unmodelled
seasonal component is inflating persistence, and re-tests after deseasonalising. It states that
the two tests have opposite nulls and that agreement in this direction is the informative case.
A weak answer picks whichever test supports the desired conclusion.

**10. "Your residual ACF has a spike at lag 52. What does that mean and what do you do?"**

A strong answer says the seasonal basis is insufficient, or a moving holiday is leaking into the
annual frequency, and proposes raising `K` or adding the holiday window before touching the
error model. It notes that fixing the ACF with an AR term while leaving the mean misspecified
changes the interpretation of every coefficient. A weak answer says "add an AR(1) term".

**11. "Why is in-sample R-squared useless for judging this model?"**

A strong answer says the model has many parameters relative to 52 observations, that R-squared
rises mechanically with parameter count, and that the relevant comparison is out-of-sample
against a seasonal naive benchmark on a calendar-aware split. A weak answer says "use adjusted
R-squared" and stops.

**12. "When would you not include a trend?"**

A strong answer says when the series is stationary around a constant or seasonal mean, when a
trend would absorb a level shift, and when the operational counterfactual is no growth. It notes
that a trend estimated on a short window is a forecast in disguise. A weak answer says "always
detrend" or "never detrend".

## 15. Key terms

- **Serial correlation** — correlation of a series with its own past; the reason naive standard errors fail.
- **Stationarity** — constant mean, variance, and lag-dependent autocovariance; a property of the process.
- **Unit root** — an autoregressive root at one; shocks do not decay and the level has no anchor.
- **Trend-stationary** — stationary deviations around a deterministic function of time.
- **Difference-stationary** — stationary only after differencing; contains a unit root.
- **Spurious regression** — high R-squared and significant t between independent non-stationary series.
- **Cointegration** — two I(1) series whose linear combination is stationary.
- **Fourier term** — a sine or cosine at a harmonic of one cycle per seasonal period.
- **Harmonic order K** — the number of sine-cosine pairs; controls seasonal smoothness.
- **Seasonal period m** — observations per seasonal cycle; 52 for weekly, 365.25 for daily.
- **Amplitude** — `sqrt(a^2 + b^2)`; the peak deviation of a harmonic, in outcome units.
- **Phase** — `atan2(b, a)`; fixes the week in which the harmonic peaks.
- **Seasonal dummy** — a 0/1 indicator for one position in the seasonal cycle.
- **Nyquist limit** — the highest identifiable frequency, one cycle per two observations.
- **Moving holiday** — a holiday whose Gregorian date changes year to year.
- **Holiday window** — the span in days around an event over which its effect acts.
- **Proximity regressor** — a regressor holding the share of a holiday window in a period.
- **Centering** — subtracting the calendar or overall mean from a holiday regressor.
- **Trading day effect** — variation from the count and weekday mix of days in a period.
- **Distributed lag** — a model with current and lagged values of a regressor.
- **Lag weight** — the coefficient on one lag; not the total effect.
- **Cumulative effect** — the sum of lag weights; the estimand for an intervention.
- **Koyck lag** — geometric decline, `delta_j = delta_0 lambda^j`, two parameters.
- **Almon lag** — weights constrained to a low-order polynomial.
- **AIC** — `n ln(RSS/n) + 2p`; efficient but not consistent.
- **BIC** — `n ln(RSS/n) + p ln(n)`; consistent order selection.
- **Ljung-Box** — portmanteau test for residual autocorrelation to a stated lag.
- **ADF test** — null of a unit root; low power against trend-stationary alternatives.
- **KPSS test** — null of stationarity; complements ADF because the nulls are opposite.

## 16. Sources

**Peer-reviewed and academic**

- Granger, C. W. J., and Newbold, P. (1974). "Spurious Regressions in Econometrics."
  *Journal of Econometrics*, 2(2), 111-120. DOI: 10.1016/0304-4076(74)90034-7. Grade A.
- Dickey, D. A., and Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive
  Time Series With a Unit Root." *Journal of the American Statistical Association*, 74(366),
  427-431. DOI: 10.1080/01621459.1979.10482531. Grade A.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., and Shin, Y. (1992). "Testing the Null
  Hypothesis of Stationarity Against the Alternative of a Unit Root: How Sure Are We That
  Economic Time Series Have a Unit Root?" *Journal of Econometrics*, 54(1-3), 159-178.
  DOI: 10.1016/0304-4076(92)90104-Y. Grade A.
- Almon, S. (1965). "The Distributed Lag Between Capital Appropriations and Expenditures."
  *Econometrica*, 33(1), 178-196. DOI: 10.2307/1911894. Grade A.
- Koyck, L. M. (1954). *Distributed Lags and Investment Analysis*. Amsterdam: North-Holland.
  Grade A.
- Franses, P. H. (2004). "Fifty Years Since Koyck (1954)." *Statistica Neerlandica*. DOI:
  10.1111/j.1467-9574.2004.00266.x. Grade A.
- Akaike, H. (1974). "A New Look at the Statistical Model Identification." *IEEE Transactions
  on Automatic Control*, AC-19(6), 716-723. DOI: 10.1109/TAC.1974.1100705. Grade A.
- Schwarz, G. (1978). "Estimating the Dimension of a Model." *The Annals of Statistics*, 6(2),
  461-464. DOI: 10.1214/aos/1176344136. Grade A.
- Koehler, A. B., and Murphree, E. S. (1988). "A Comparison of the Akaike and Schwarz Criteria
  for Selecting Model Order." *Journal of the Royal Statistical Society, Series C*, 37(2),
  187-195. DOI: 10.2307/2347338. Grade A.

**Official statistics, standards, and government**

- U.S. Census Bureau, X-13ARIMA-SEATS Reference Manual, regression chapter, covering the
  built-in `easter[n]`, `easterstock[n]`, `labor[n]`, and `thank[n]` regressors. Grade B.
- U.S. Census Bureau, "Genhol - A Utility That Generates User-Defined Moving Holiday
  Regressors," documentation, 28 August 2020. Grade B.
- U.S. Census Bureau, "Notes on Centering Holiday Regressors." Grade B.
- U.S. Census Bureau, "Modeling of Holiday Effects and Seasonality in Daily Time Series,"
  Research Report RRS2018-01. Grade B.
- Roberts, I., and White, G. (2015). "Seasonal Adjustment of Chinese Economic Statistics."
  Reserve Bank of Australia Research Discussion Paper 2015-13. Grade B.
- Lin, J.-L., and Liu, T.-S. (2002). "Modeling Lunar Calendar Holiday Effects in Taiwan."
  U.S. Census Bureau working paper. Grade B.
- Congressional Research Service, "Lunar New Year: Fact Sheet," R46674, updated January 2026.
  Grade B.
- ISO 8601, date and time representation. Grade B.
- IANA time zone database. Grade B.
- National Retail Federation retail calendar, 4-5-4 convention. Grade C: industry convention
  with accounting consequences, not a statistical standard.

**Textbooks and technical references**

- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition,
  section 10.5 "Dynamic harmonic regression" and section 12.1 "Complex seasonality."
  https://otexts.com/fpp3/dhr.html and https://otexts.com/fpp3/complexseasonality.html.
  Grade A for the method; the `K = 5` weekly and `K = 10` daily guidance is one applied example,
  not a derived optimum.
- Hyndman, R. J., "Forecasting with Daily Data," blog post,
  https://robjhyndman.com/hyndsight/dailydata/. Grade C: practitioner guidance from a
  statistician, not peer reviewed.
- `forecast::fourier` reference documentation,
  https://pkg.robjhyndman.com/forecast/reference/fourier.html. Grade B.
- Bell, W. R., and Hillmer, S. C. (1983), on modelling time series with calendar variation. The
  interval-regressor approach used in X-13 is attributed to this work in the Census Genhol
  documentation; consult that documentation for the implementation. Grade A for the method.

**Industry and vendor**

- EyeOn, "Improve Forecasting Seasonality for Chinese New Year Using Machine Learning," vendor
  blog. Reports +3.5 percentage points forecast accuracy and 44.6 percentage points bias
  reduction in one client engagement. Grade D: vendor marketing, single case, baseline not
  reproducible.

**Note on citations.** Bibliographic details were checked against the sources returned by
search. Figures attributed to vendor case studies are reported as the vendor states them and
are not independently verified. Almon's weight estimates are reported as the paper describes
them and are one application, not a general law of lag lengths. Where a number in this topic is
illustrative, the text says so.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite: [`02_probability-bootstrap.md`](02_probability-bootstrap.md)
- Prerequisite: [`03_hypothesis-tests.md`](03_hypothesis-tests.md)
- Next topic: `05_autocorrelation-robust-errors.md` (planned)
- Related design: `08_difference-in-differences.md` (planned)
- Calendar and week conventions: [`../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Deeper forecasting treatment: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Module root: [`../../../modules/01_foundations/_index.md`](../../../modules/01_foundations/_index.md)

