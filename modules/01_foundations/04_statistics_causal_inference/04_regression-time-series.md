> Module 04 — Statistics and Causal Inference
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.02, 04.03

# Regression with Time Series, Seasonality, and Holidays

---

## 1. Why this topic exists

A weekly demand model for a distribution centre reports that this December will be 9.4% below last
December. The category manager has one question: is demand falling, or did the calendar move? Three things
moved. Easter fell a week later, so the pre-Easter build landed in a different week. The prior year had 53
retail weeks, so "December" was one week longer. And a November promotion pulled volume forward, which the
model recorded as a December decline because the promotion was not in it.

None of the three is a demand signal. All three are calendar and specification. The model was estimated
correctly and answered the wrong question, which is the characteristic failure of regression on supply
chain time series.

**The assumption that breaks.** Ordinary least squares (OLS) treats observations as exchangeable draws.
The standard error of a coefficient is built on errors being independent and identically distributed
(i.i.d.): each row tells you something new, and no row tells you about another. A weekly series violates
this in at least three ways, and each has a different repair.

| Break | What it looks like | What it does to OLS | Where handled |
|---|---|---|---|
| Serial correlation | Residuals correlated week to week; a shock persists | Standard errors too small, t too large, intervals too narrow | Phase G; Topic 05 |
| Shared calendar shocks | Every SKU spikes in the same week | Errors correlated across units; holidays attributed to nearby covariates | Phase D; section 6.7 |
| Trending regressor and outcome | Both rise for unrelated reasons; R-squared near 0.9 | Spurious regression: high R-squared, large t, meaningless coefficient | Phase F; section 6.8 |

A fourth break is equally common: **overlapping horizons**. Forecast four weeks ahead from a four-week
moving average and consecutive errors share three of four inputs, so they are autocorrelated by
construction. The error structure is a property of the forecast method, and it must be modelled or the
intervals are fiction.

**Regression is used for two jobs.** The forecasting job asks what next week will be. The attribution job
asks what the promotion, the holiday, or the price change did. This topic is about the second, which is
why diagnostics come before interpretation. A forecast can be useful with a misspecified error process. A
coefficient cannot be interpreted with one, because the standard error attached to it is not the standard
error of anything.

**The claim of the topic.** A time-series regression is a claim about three things at once: the shape of
time, the calendar, and the causal window. Get the calendar wrong and the seasonal coefficients absorb
holidays. Get the trend wrong and the coefficients are spurious. Get the window wrong and the intervention
effect is the first lag rather than the sum.

**Where this sits.** Topic 02 gave you intervals that survive non-normal data. Topic 03 gave you a
decision rule and a family. This topic builds the design matrix those tools are applied to. Topic 05
repairs the standard errors once the specification is right. Topic 06 draws the graph that decides which
covariates belong in the matrix at all.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** why weekly and daily supply chain series violate the independence assumption,
and **name** the three violations and their different consequences.
2. **Construct** a Fourier seasonal basis for period `m`, by hand for `K = 1` and `K = 2`, and
**state** what each pair adds and what it costs.
3. **Compare** seasonal dummies with harmonics on parameter count, data requirement, smoothness,
and behaviour under a moving holiday, and **choose** for a stated series length.
4. **Build** a holiday regressor keyed to the event, with a window and a centering rule, for
Easter, Ramadan, Chinese New Year, and Thanksgiving.
5. **Estimate** a finite distributed lag and report the cumulative effect with its own standard
error rather than the sum of the individual standard errors.
6. **Diagnose** a trending series with both a unit-root test and a stationarity test, and
**explain** why they can disagree.
7. **Run** the diagnostic loop in order and **state**, before interpreting a coefficient, which
specification defect would invalidate the interpretation.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: an observed series is the sum of four things you cannot see separately, and the job is
to draw the boundaries in the right place.

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

   TREND      21,850 + 42 t       a straight line; +2,184 units over the year
   HARMONIC   K = 2 pairs         two smooth waves, period 52 and period 26
                wave 1: amplitude 1,798 units/week, peak week 51
                wave 2: amplitude   358 units/week, peaks weeks 9 and 35
   HOLIDAY    narrow spikes      keyed to the EVENT, not the calendar slot
                Thanksgiving  +2,400 units/week, week 47 every year
                Christmas     -3,100 units/week, week 52 every year
                Easter        +900 units/week  <-- THE SPIKE MOVES
   RESIDUAL   what is left       should look like noise; sd about 319 units/week
```

The decomposition is not unique. Any two components can absorb part of a third, and the specification
choices in Phases C to F decide which absorption you accept. That is why the model is a claim, not a
measurement.

### 3.1 The moving holiday is the tell

A fixed holiday is easy: week 47 is Thanksgiving every year. A moving holiday destroys the correspondence
between the calendar slot and the event, which makes it diagnostic. If the seasonal specification is a set
of week-of-year dummies, a moving holiday leaks into whichever dummy is near it, differently each year.

```text
   EASTER SPIKE, THREE YEARS, ON THE SAME CALENDAR AXIS

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
coefficient is units per week per week. If the shape is wrong, every other coefficient is contaminated,
because trend and seasonality compete for the same low-frequency variation.

**Principle 2 — A seasonal dummy and a harmonic pair are the same basis at different resolution.** A
week-of-year dummy is a local, nonparametric estimate of one week's deviation. A harmonic pair is a
global, parametric estimate of one frequency. At `K = m/2` the harmonic basis spans the same space as the
full dummy set. Choosing `K` is choosing a point on a bias-variance curve, not choosing between two
theories.

**Principle 3 — A moving holiday is a regressor keyed to the event.** The date is data, not a formula.
Store the event date per year and per jurisdiction, and build the regressor from it. A dummy keyed to the
calendar slot is attenuated by the probability that the slot holds the event.

**Principle 4 — A distributed lag is a claim about a causal window.** The lag-zero coefficient is the
immediate effect; the sum is the effect. If the promotion pulls demand forward, some lags are negative and
the sum can be a small fraction of the peak. Reporting the peak as "the lift" is the most common error in
promotional analytics.

**Principle 5 — Trend has three candidate specifications, each with a different counterfactual.** No trend
reverts to a constant mean. A deterministic trend reverts to a moving line. A unit root says shocks
persist and the level has no anchor. The three give different forecasts, intervals, and causal claims, and
the data often cannot separate them.

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

The variance of an OLS coefficient is `sigma^2 (X'X)^-1` only when errors are spherical. With serial
correlation the correct variance is a sandwich that includes the autocovariances. For a first-order
autoregression with parameter `rho`, the variance inflation on a mean is roughly `(1 + rho) / (1 - rho)`:
3 at `rho = 0.5`, and 9 at `rho = 0.8`. A t-statistic of 6 computed naively is a t-statistic of 2
correctly, and the difference decides whether the coefficient is reported as real.

> **Note:** The consequence is asymmetric. Serial correlation usually makes naive standard
> errors too small, so it manufactures significance; it rarely hides a real effect. An analyst
> who ignores it produces false positives, not false negatives.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A definition without one is the defect this module
exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Time series | A sequence indexed by time, where order carries information | Treated as a sample of independent rows | Observations at a stated frequency |
| Serial correlation | Correlation between a series and its own past values | Used interchangeably with autocorrelation | Coefficient at a stated lag |
| Autocorrelation function (ACF) | Correlation of a series with its lagged self, as a function of lag | Read without stating the sampling frequency | Dimensionless, lag in periods |
| Partial autocorrelation (PACF) | Correlation at lag k after removing shorter lags | Interpreted as a causal statement about lag k | Dimensionless, lag in periods |
| Stationarity | Constant mean, variance, and autocovariance depending only on lag | Asserted from a single plot | Property of a process |
| Trend-stationary | Stationary deviations around a deterministic function of time | Confused with difference-stationary | Deterministic trend plus stationary error |
| Difference-stationary | Stationary only after differencing; contains a unit root | Called "non-stationary" and left there | Order of integration `d` |
| Unit root | An autoregressive root at one; shocks do not decay | Used as a synonym for "trending" | Root at 1, or `d = 1` |
| Spurious regression | A regression between independent non-stationary series yielding high R-squared and significant t | Diagnosed from R-squared alone | R-squared, t, Durbin-Watson |
| Cointegration | Two I(1) series whose linear combination is stationary | Assumed whenever two series trend together | Residual order of integration |
| Fourier term | A sine or cosine at a harmonic of the fundamental frequency | Called a "seasonal dummy" | Dimensionless, in [-1, 1] |
| Harmonic | A frequency that is an integer multiple of the fundamental | Confused with the fundamental itself | Integer order `k` |
| Harmonic order, K | Number of sine-cosine pairs included | Reported without the period it applies to | Integer; 2K columns |
| Seasonal period, m | Observations in one seasonal cycle | Assumed to be 12 for all data | Observations per cycle |
| Amplitude | Peak deviation of a harmonic from its mean, `sqrt(a^2 + b^2)` | Read as the coefficient on the sine term | Outcome units |
| Phase | Horizontal shift of a harmonic, `atan2(b, a)` | Ignored, so the peak week is misread | Radians; convert to weeks |
| Nyquist frequency | Highest identifiable frequency, one cycle per two observations | Exceeded by including `K > m/2` | Cycles per observation |
| Seasonal dummy | A 0/1 indicator for one position in the seasonal cycle | Used with `m` levels and an intercept | 0 or 1 |
| Moving holiday | A holiday whose Gregorian date changes year to year | Treated as a fixed date | Event date per year and jurisdiction |
| Fixed holiday | A holiday on a constant Gregorian date | Assumed to fall in the same week each year | Month and day |
| Holiday window | Span of days or weeks around an event over which its effect acts | Set to one week by default | Days before and after the event |
| Proximity regressor | A regressor holding the share of a holiday window falling in the period | Confused with a dummy | Proportion in [0, 1] |
| Distributed lag | A model where the outcome depends on current and past regressor values | Reported as a single contemporaneous coefficient | Coefficients by lag |
| Lag weight | The coefficient on one lag of the regressor | Interpreted as the total effect | Outcome units per regressor unit |
| Cumulative effect | Sum of lag weights from zero to the chosen horizon | Computed by adding standard errors | Outcome units per regressor unit |
| Koyck lag | Infinite lag with geometrically declining weights, `d_j = d_0 lambda^j` | Estimated by OLS with a lagged dependent variable, ignoring the bias | Two parameters |
| Almon lag | Finite lag whose weights lie on a low-order polynomial | Used without checking the endpoint restrictions | Polynomial degree plus horizon |
| Akaike information criterion (AIC) | `n ln(RSS/n) + 2p`, a penalised fit measure | Compared across models with different samples | Dimensionless |
| Bayesian information criterion (BIC) | `n ln(RSS/n) + p ln(n)`, a consistent penalised fit measure | Treated as interchangeable with AIC | Dimensionless |
| Durbin-Watson statistic | Test statistic for first-order autocorrelation, near 2 when absent | Reported without its inconclusive zone | Statistic in [0, 4] |
| Ljung-Box statistic | Portmanteau test for autocorrelation up to a stated lag | Run at a lag chosen after seeing the ACF | Chi-squared with degrees of freedom |
| Augmented Dickey-Fuller (ADF) test | A test whose null is a unit root | Read as proof of a unit root when not rejected | Statistic against critical values |
| KPSS test | A test whose null is stationarity | Read in the same direction as ADF | Lagrange multiplier statistic |
| HAC standard error | Heteroskedasticity- and autocorrelation-consistent standard error | Assumed to repair a misspecified mean equation | Standard error with a bandwidth |

Four of these carry most of the interpretive weight.

**Stationarity is a property of the process, not the sample.** You observe a finite path and test a
hypothesis about the process that could have generated it. Failing to reject a unit root has not found
one; it has failed to rule one out, and the power depends on sample length, trend specification, and how
much low-frequency variation the seasonal component left behind.

**Amplitude versus coefficient.** A harmonic pair is two coefficients, and neither is the amplitude. The
amplitude is `sqrt(a^2 + b^2)`, and the peak sits where the phase puts it. Reporting "the seasonal
coefficient is 1,785" reports a number that depends on where the series starts, because shifting the time
origin rotates `a` and `b` into each other.

**Cumulative effect versus sum of standard errors.** The variance of a sum is not the sum of the variances
unless the terms are uncorrelated, and lag weights from one regression are almost always negatively
correlated because the fit trades one lag against the next. The correct standard error comes from the
covariance matrix and is usually smaller.

**AIC versus BIC.** AIC penalises each parameter by 2, BIC by `ln(n)`, which is 3.95 at `n = 52`. BIC is
consistent and selects the true model with probability approaching one; AIC tends to over-select but is
asymptotically efficient for one-step prediction. On 52 observations they usually agree, and disagreement
is a sign the sample is too short to separate the orders.

## 5. The mechanics, step by step

This section builds a specification in eight phases. The order matters: each phase assumes the previous
one is settled, and a defect in an early phase cannot be repaired by a later one.

### Phase A — Fix the time index

**Step 1. State the bucket and its length.** Weekly, daily, four-weekly, or monthly. The bucket determines
which seasonal frequencies exist and which are averaged away. A weekly series has no day-of-week pattern;
a daily series has both day-of-week and annual patterns, plus moving holidays that weekly resolution
cannot represent without loss.

**Step 2. State the week definition.** ISO week, retail 4-5-4 week, or week-ending Sunday. Joining them
without a mapping produces a one-week misalignment that looks like a demand anomaly. The same definition
must be used for the outcome, the promotion calendar, and the holiday dates.

**Step 3. Handle the 53-week year explicitly.** A 53-week year is 371 days against 364, or 1.923% longer.
Exclude the extra week, restate it, or include an indicator. Do not annualise through it silently.

**Step 4. Confirm there are no gaps.** A missing week is not a zero. Interpolating it creates a value the
model treats as observed, and if the gap coincides with a holiday the model attributes the interpolation
to the holiday. Carry a missing indicator or drop the week and record it.

**Step 5. Count the seasonal cycles.** Three is the practical minimum for a stable parametric seasonal
estimate. With two, seasonality and trend are nearly confounded and the estimate is a decomposition
assumption rather than a measurement.

### Phase B — Write the model as a unit-bearing claim

**Step 6. Write the equation before fitting anything.** A specification you cannot write on one line with
units is a specification you do not understand.

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

**Step 7. State the estimand.** Which coefficient answers the question? If the question is "what did the
promotion do", the estimand is `SUM d_j`, not `d_0`. If it is "how seasonal is this item", it is the
amplitude at `k = 1`. If it is "what is the trend", it is `b1` in units per week per week. Three
questions, three coefficients, one regression.

### Phase C — Build the seasonal basis

**Step 8. Choose between dummies and harmonics.** The choice is a point on a bias-variance curve.

| Property | Week-of-year dummies | Fourier pairs, order K |
|---|---|---|
| Parameters | `m - 1` with intercept | `2K` |
| Estimated from | One observation per year per dummy | All observations, at each frequency |
| Shape | Arbitrary; each week free | Smooth; sums of sine waves |
| Data requirement | Three or more observations per dummy | Depends on K, not on m |
| Moving holiday | Absorbed into the nearest dummy, differently each year | Absorbed into the smooth wave, also wrongly but smoothly |
| Extrapolation | Undefined outside observed weeks | Defined for any t |
| When to use | Long history, stable calendar, week-specific effects | Short history, long period, multiple seasonalities, forecasting |

**Step 9. Construct the columns.** For period `m` and order `k`, the pair is `sin(2 pi k t / m)` and
`cos(2 pi k t / m)`, for `k = 1 ... K`. The pair at `k = 1` is one cycle per year; at `k = 2` it is two
cycles per year, capturing a spring and an autumn peak, not a "second harmonic of the same peak".

**Step 10. Choose K.** Start at `K = 1` and increase while the information criterion improves and the
seasonal residual autocorrelation falls. The textbook guidance is to minimise AICc over a range, with the
caveat that AICc over-selects for long periods, so a cap is often applied: `K = 5` for weekly and `K = 10`
for daily in the bank-calls example. For 52 weeks, `K` between 2 and 4 is usually enough for a smooth
shape.

**Step 11. Check the Nyquist limit.** At weekly frequency the highest identifiable seasonal frequency is
26 cycles per year, so `K <= 26`. At `K = 26` the basis has 52 columns and spans the same space as 52 week
dummies; beyond it, sine and cosine alias onto existing frequencies and the design matrix becomes
singular.

**Step 12. Check collinearity with the trend.** At `m = 52`, `k = 1` is not collinear with a straight line
over one year, but across three years the correlation between `t` and `sin(2 pi t / 52)` rises. Report the
condition number; above about 30, trend and seasonality are trading off and both coefficients are
unstable.

### Phase D — Add the calendar

**Step 13. Separate fixed from moving holidays.** Fixed-date holidays still move in weekday and
retail-week assignment. Moving holidays move in date as well. Build the date table first, then the
regressor.

| Holiday | Date rule | Range | Typical window | Modelling note |
|---|---|---|---|---|
| Easter, Western | First Sunday after the first full moon on or after 21 March | Late March to late April | 8 days before to 2 after | X-13ARIMA-SEATS provides `easter[n]`, default `n = 8` |
| Ramadan and Eid al-Fitr | Islamic lunar calendar, subject to crescent sighting | About 10 to 11 days earlier each Gregorian year | 1 to 2 weeks before Eid | Moves through all months; centre on the mean |
| Chinese New Year | Lunar new year, second new moon after the winter solstice | 21 January to 20 February | 3 to 4 weeks before, 1 to 2 after | Production halt before, restock after; two windows |
| US Thanksgiving | Fourth Thursday of November | 22 to 28 November | Week containing it | Fixed weekday, moving date |
| Canadian Thanksgiving | Second Monday of October | 8 to 14 October | Week containing it | Different date from the US; do not share a regressor |

**Step 14. Key the regressor to the event.** For a weekly series, find the week containing the event date
each year and set the dummy there. For a monthly series, use a proximity regressor: the share of the
holiday window falling in the month. Statistical agencies use the share version because it avoids
arbitrary assignment when a window straddles a boundary.

**Step 15. Choose the window from the mechanism.** Easter affects shipping and production in the weeks
before, because the holiday closes plants and docks. Chinese New Year affects production for weeks before,
because migrant workers travel home, and restocking for weeks after. Thanksgiving affects one week in US
retail. State the window in days relative to the event, not as the value that maximises fit.

**Step 16. Centre the regressor.** A dummy that is 1 in one week and 0 elsewhere has mean `1/m`, so it is
correlated with the intercept. Statistical agencies centre holiday regressors by subtracting the calendar
mean, so the coefficient is the deviation of the holiday week from the average week. Without centering,
part of the effect is absorbed into the intercept and the seasonal terms.

**Step 17. Do not double count.** If Christmas week is also the seasonal peak, a Christmas dummy and a
harmonic peaking in the same week compete, and the model splits the effect arbitrarily. Either exclude
holiday weeks from the seasonal fit or report the combined fitted value rather than the two coefficients
separately.

### Phase E — Add the intervention with a lag

**Step 18. Write the finite distributed lag.** `y_t = ... + SUM_{j=0..L} d_j x_{t-j} + e_t`. The
coefficient `d_j` is the effect of a one-unit change in `x` at `t - j` on `y` at `t`, in outcome units per
regressor unit. The cumulative effect to horizon `L` is `SUM d_j`.

**Step 19. Choose L from the decision process, not from the fit.** A consumer staple's pantry-loading
reversal is 2 to 4 weeks. A durable promotion may pull demand forward by a quarter. A price change may
pass through over two quarters. State the mechanism, set `L` from it, then test whether the last lag is
distinguishable from zero. Do not extend `L` until the last lag is insignificant; that is selection by
t-test and it over-fits.

**Step 20. Consider a restricted lag if L is large.** Koyck sets `d_j = d_0 lambda^j` with `0 < lambda <
1`: two parameters and a long-run effect of `d_0 / (1 - lambda)`. It is estimable by including the lagged
dependent variable, but that estimator is biased and inconsistent when the error is autocorrelated, which
it usually is. Almon puts `d_j` on a low-degree polynomial, reducing parameters and imposing smoothness;
the endpoint restrictions must be stated, because they are assumptions, not conveniences.

**Step 21. Compute the cumulative effect and its own standard error.** With `d` the vector of lag
coefficients and `V` their covariance matrix, the cumulative effect is `1'd` and its variance is `1'V1`.
That square root is the number the business needs.

> **Note:** Individual lag coefficients are usually poorly identified even when the cumulative
> effect is well identified. A promotion that lifts sales by 1,200 in week 0 and gives back 550
> over three weeks has a cumulative effect of 650, and its standard error can be a third of the
> sum of the four individual standard errors. Report the sum, and the lags only as a shape.

### Phase F — Choose the trend

**Step 22. Plot the series at full length and at the seasonal scale.** A trend that looks obvious over ten
years may be a level shift. A trend that looks absent over one year may be clear over five.

**Step 23. Test both ways.** The augmented Dickey-Fuller (ADF) test has a unit root as its null. The KPSS
test has stationarity as its null. Running both gives four outcomes, and the two-by-two table is more
informative than either test.

| ADF (H0: unit root) | KPSS (H0: stationarity) | Reading | Action |
|---|---|---|---|
| Not rejected | Rejected | Unit root | Difference, or model as I(1) with an error-correction term |
| Rejected | Not rejected | Stationary | Level regression with seasonal and holiday terms |
| Rejected | Rejected | Trend-stationary, or a break | Inspect the plot; add a trend or a break |
| Not rejected | Not rejected | Uninformative: low power both ways | Longer sample, or a stated judgement |

**Step 24. Deseasonalise before testing when seasonality is strong.** A large unmodelled seasonal
component inflates estimated persistence and makes ADF fail to reject a unit root that is not there. Fit
the seasonal terms first, test the residual, and report both raw and deseasonalised results. This is the
most common source of a false unit root in demand data.

**Step 25. State the counterfactual the trend implies.** A deterministic trend says that absent the
intervention, demand would have continued on the line. A unit root says the best forecast of the future
level is the current level. These are different claims, and the choice needs an operational defence, not
only a test statistic.

### Phase G — Diagnose in a fixed order

**Step 26. Plot the residuals against time.** Look for a trend, a seasonal wave, a level shift, and a
variance change. A funnel shape invalidates the standard errors before any test does.

**Step 27. Plot the ACF and PACF to at least twice the seasonal lag.** A spike at lag `m` means the
seasonal terms are insufficient; at lag 1, the dynamics are missing; at lags 3 to 5 after a promotion, the
lag window is too short.

**Step 28. Run Ljung-Box at the seasonal lag and at twice it.** Report the statistic, degrees of freedom,
and p-value. Do not choose the lag after looking at the ACF; that is a selected test.

**Step 29. Check coefficient stability recursively.** Estimate on the first 60% of the sample, then on
expanding windows, and plot each coefficient. One that moves by more than its own standard error across
windows is not stable enough to interpret.

**Step 30. Hold out a period that contains a moving holiday.** Random cross-validation trains on the
future. Use a rolling or expanding origin, and ensure at least one fold contains an Easter or a Chinese
New Year. A model never tested on a moving holiday has not been tested.

**Step 31. Compare against a seasonal naive benchmark.** The benchmark is the same week last year. Report
the ratio of mean absolute error to the benchmark's, not the R-squared.

### Phase H — Interpret, then state what would change it

**Step 32. Interpret only the estimand named in Step 7.** If the estimand is the cumulative promotion
effect, do not lead with `d_0`. If it is the seasonal amplitude, do not lead with `a_1`.

**Step 33. Attach units to every number in the sentence.** "The promotion raised volume by 650 units per
week of promotion, 95% CI 23 to 1,277" is interpretable. "The promotion was significant" is not.

**Step 34. State the three specification defects that would overturn the interpretation.** For each
coefficient, name the defect that would invalidate it: an unmodelled moving holiday, a lag window that is
too short, a trend that should have been a difference.

## 6. Worked example

A regional distribution centre ships one fast-moving consumer good to retail stores. The series is weekly
outbound volume in units per week, 52 weeks, one retail year with no 53rd week. All figures are
illustrative but internally consistent, and every calculation can be reproduced by hand.

### 6.1 The series and the question

The commercial question is: "Did the week-30 promotion work, and how much of the December decline is
real?" Two estimands follow: the cumulative effect of the promotion, and the December level after removing
trend, seasonality, and the calendar.

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

`n = 52`, parameters `p = 13`, residual degrees of freedom 39, residual standard deviation 319 units per
week, R-squared 0.918, adjusted R-squared 0.893. The Easter regressor is a three-week window with weights
0.5, 1.0, 0.5 centred on the week containing Easter Sunday, so the coefficient is the effect in the peak
week.

### 6.3 Hand evaluation at six weeks

| Week `t` | Trend `21,850 + 42t` | Harmonic 1 | Harmonic 2 | Holiday | Fitted | Observed | Residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 22,396 | -215.0 | +180.0 | +900 | 23,261 | 23,510 | +249 |
| 26 | 22,942 | -1,785.0 | -180.0 | 0 | 20,977 | 20,810 | -167 |
| 39 | 23,488 | +215.0 | +180.0 | 0 | 23,883 | 24,120 | +237 |
| 47 | 23,824 | +1,591.2 | -353.7 | +2,400 | 27,461 | 27,900 | +439 |
| 51 | 23,992 | +1,797.9 | -249.0 | 0 | 25,541 | 25,300 | -241 |
| 52 | 24,034 | +1,785.0 | -180.0 | -3,100 | 22,539 | 22,600 | +61 |

Read two rows closely. At week 51 the harmonic peaks at +1,797.9 units per week with no holiday term. At
week 52 the harmonic is almost as high at +1,605.0, but Christmas subtracts 3,100, so the fitted value
falls 3,002 units per week in one week against an observed fall of 2,700. That is the shape of a
distribution centre shipping hard into the last full week and then closing.

At week 13 the harmonic contribution is -35.0, close to the annual average, so almost all of the fitted
elevation is the Easter window. A week-of-year dummy would instead carry both the Easter effect and the
local seasonal level, in a proportion that changes every year.

### 6.4 What the Fourier coefficients mean

The pair `(a_k, b_k)` is not two effects. It is one wave, described by an amplitude and a phase.

```text
   amplitude      R_k = sqrt(a_k^2 + b_k^2)
   phase          phi_k = atan2(b_k, a_k)
   peak position  t_peak = (m / 2 pi) (pi/2 - phi_k)  modulo m/k
```

For the first harmonic:

```text
R_1 = sqrt((-215.0)^2 + 1,785.0^2) = sqrt(3,232,450) = 1,797.90 units per week

phi_1 = atan2(1,785.0, -215.0) = pi - atan(1,785.0 / 215.0)
      = 3.14159 - 1.45087 = 1.69072 radians

t_peak = (52 / 6.28319) x (1.57080 - 1.69072)
       = 8.2765 x (-0.11992) = -0.992  ->  week 51.0
```

For the second harmonic, `R_2 = sqrt(310^2 + 180^2) = 358.47` units per week, `phi_2` is `-0.52625`
radians, and the peaks fall at weeks 8.7 and 34.7.

The correct sentence is "seasonal volume peaks in week 51 at 1,798 units per week above the trend, with a
secondary peak in weeks 9 and 35 of 358 units per week". The incorrect sentence is "the sine coefficient
is -215". The second number is not invariant to where the year starts.

### 6.5 Choosing K

Estimating the same model at `K = 1` to `K = 4` gives residual sums of squares of 4,820,000, 3,960,000,
3,910,000, and 3,895,000. With `p = 9 + 2K` and `n = 52`, AIC is minimised at `K = 2` (610.52) and BIC
also at `K = 2` (635.89). The step from `K = 1` improves AIC by 6.2; the step to `K = 3` buys 50,000 RSS
for two parameters and is priced as worse by both criteria.

That is the bias-variance trade in one line, and the fragility is ordinary on 52 observations. The `K = 3`
improvement is not zero, and another year might cross the AIC threshold. The defence is a stability check:
refit on the first 39 weeks and confirm `K` does not change.

### 6.6 The promotion and its hangover

The promotion ran in week 30 at full depth, coded `P_30 = 1.0`.

| Lag `j` | Week | `d_j` | SE | t | Cumulative |
|---:|---:|---:|---:|---:|---:|
| 0 | 30 | +1,200 | 260 | 4.62 | 1,200 |
| 1 | 31 | +480 | 240 | 2.00 | 1,680 |
| 2 | 32 | -620 | 235 | -2.64 | 1,060 |
| 3 | 33 | -410 | 230 | -1.78 | 650 |

The shape is the mechanism. Weeks 30 and 31 gain because the price cut brings purchases forward and some
are incremental. Weeks 32 and 33 lose because households work through the stock they bought. The
cumulative effect is 650 units per week of promotion.

The standard error of the cumulative effect is not the sum of the four standard errors, which would be
`260 + 240 + 235 + 230 = 965`. It comes from the covariance matrix:

```text
Var(sum d_j) = 1' V 1 = 310^2 = 96,100
SE(sum d_j)  = 310 units per week

95% CI with 39 df, t = 2.022:
650 +/- 2.022 x 310 = 650 +/- 627 = [23, 1,277] units per week
```

Only the first of three candidate sentences is safe: "The promotion raised net volume by 650 units per
week of promotion, 95% CI 23 to 1,277." Saying 1,200 reports the peak and ignores the 550 given back;
saying there was no effect because the last lag is insignificant ignores the significant sum.

The naive interval, `650 +/- 1,951`, contains large negative values and would support the false conclusion
that the promotion destroyed volume. Lag weights are negatively correlated because the fit trades one week
against the next, and ignoring that inflates the interval threefold.

### 6.7 The moving holiday, quantified

Suppose the Easter regressor had been a dummy on week 13, and the event fell as follows.

| Year | Week containing Easter | Week-13 dummy fires | Recorded at week 13 | Where the true effect goes |
|---|---:|---:|---:|---|
| Y1 | 13 | Yes | 900 | Recorded correctly |
| Y2 | 14 | No | 0 | Week 14, into the residual |
| Y3 | 14 | No | 0 | Week 14, into the residual |
| Y4 | 15 | No | 0 | Week 15, into the residual |
| Y5 | 13 | Yes | 900 | Recorded correctly |

```text
E[gamma_hat] = 900 x (2 / 5) = 360 units per week
```

The estimate is attenuated to 40% of the true effect, and the standard error is computed as if the
regressor were correct. The two years of Easter effect in week 14 do not disappear; they sit in the
residual, inflate the residual variance, and make every other coefficient less precise. This is why a
moving holiday is not a nuisance to be absorbed but a specification error that contaminates the whole fit.

The repair is the three-week window keyed to the event date. With weights 0.5, 1.0, 0.5, every year
contributes, the coefficient is the peak-week effect, and adjacent weeks are modelled rather than left in
the residual.

> **Note:** A window does not make the regressor orthogonal to the seasonal terms. If Easter
> always falls near a seasonal trough, the window and the harmonic share the effect. Check the
> correlation between the holiday regressor and the seasonal columns, and report it.

### 6.8 Trend and stationarity

The series rises from about 21,900 to about 24,000 units per week over the year, a 9.6% increase.

| Test | Series | Statistic | 5% critical value | Decision |
|---|---|---:|---:|---|
| ADF, intercept, 4 lags | Levels | -2.31 | -2.93 | Fail to reject unit root |
| KPSS, intercept, 4 lags | Levels | 0.61 | 0.463 | Reject stationarity |
| ADF, intercept, 4 lags | After removing the fitted seasonal | -4.12 | -2.93 | Reject unit root |
| KPSS, intercept, 4 lags | After removing the fitted seasonal | 0.21 | 0.463 | Fail to reject stationarity |

The raw tests agree on a unit root. The deseasonalised tests agree on stationarity. The resolution is that
the seasonal component, with an amplitude of 1,798 units per week against a trend of 2,184 over the whole
year, dominates the low-frequency variation the tests read. ADF interprets a slow seasonal swing as
persistence. This is a known low-power problem, and it is why the KPSS paper argues that failing to reject
a unit root may reflect the test rather than the data.

The decision: treat the series as trend-stationary, include the linear trend, and state the counterfactual
explicitly. If the business believes the category is in secular decline, the trend should be re-estimated
on a longer window.

### 6.9 Diagnostics before interpretation

| Check | Result | Reading |
|---|---|---|
| Residual plot | No trend, no funnel, one outlier at week 30 | Acceptable; the outlier is the promotion week and is modelled |
| ACF at lag 1 | 0.11 | Within the band `1.96 / sqrt(52) = 0.272` |
| ACF at lag 52 | -0.06 | No annual residue |
| Ljung-Box, 8 lags | Q = 9.4, df = 8, p = 0.31 | No evidence of remaining autocorrelation |
| Ljung-Box, 52 lags | Q = 14.8, df = 52, p = 0.79 | No annual residue; note low power at this lag |
| Recursive `d_0` | 1,200 +/- 170 across the last 26 origins | Stable within one standard error |
| Holdout, last 13 weeks | MAE 2.9% of mean; seasonal naive 6.1% | Beats the benchmark by 2.1 times |

The holdout contains Christmas and New Year, which is the point. A holdout that avoids the moving parts of
the calendar tests nothing that matters.

### 6.10 The sentence you are allowed to write

"The week-30 promotion raised net volume by 650 units per week of promotion (95% CI 23 to 1,277), with a
peak effect of 1,200 in the promotion week and a give-back of 550 over the following three weeks. The
December level is 3,100 units per week below the seasonal norm, which is the Christmas closure and not a
demand decline."

Three things would overturn the second sentence: a Christmas regressor that should have been a two-week
window; a 53-week prior year that was not restated; and a December promotion that was not in the model.
The first is a specification defect, the second a calendar defect, the third an omitted variable. All
three are checkable, and none is checkable from the regression output alone.

## 7. Code, query, or calculation

One compact specification in Python, using `statsmodels`. The point is the design matrix, not the library.

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

Three things to notice. The `easter_*` columns are built from the holiday date table, not from observed
sales; built from sales, the regressor would contain the outcome. The `shift(j)` uses the promotion
calendar, known before the week, so it is not leakage. And `cov_type="HAC"` is a placeholder for Topic 05:
it changes the standard errors and not the coefficients, so it cannot repair a wrong mean equation.

## 8. Data, standards, and regulatory context

Time-series regression sits on reference data that is more governed than most analysts expect. Holiday
dates are not a formula. They are a published calendar with a version, and in several jurisdictions they
are amended by annual decree.

### 8.1 Calendar and holiday reference data

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 8601 | Date and time representation | The week date and ordinal date used as join keys | B |
| IANA time zone database | Offset rules by region | Which local date an event falls on | B |
| National statistical office holiday calendars | Public holidays by jurisdiction | The event dates used to build regressors | B |
| NRF retail calendar | 4-5-4 fiscal weeks | The week index for retail series, including the 53rd week | C |
| X-13ARIMA-SEATS built-in regressors | `easter[n]`, `labor[n]`, `thank[n]`, `easterstock[n]` | Default window and centering for common holidays | B |
| Genhol utility | User-defined moving-holiday regressors | Window bounds, proportionality versus count, centering | B |

A holiday table is a versioned dataset, not a function. Easter can be computed from the computus, but
Ramadan depends on observance, and one-off, moved, and substitute holidays are announced. A model that
derives holidays from a formula will be wrong in the years that matter.

### 8.2 Seasonal adjustment practice

Official agencies treat moving holidays as a regression-ARIMA pre-adjustment problem. X-13ARIMA-SEATS
includes a default Easter regressor with an eight-day window before Easter, a `thank[n]` regressor for
Thanksgiving, and a `labor[n]` regressor for Labor Day. For holidays the built-in set does not cover,
Genhol generates a user-defined regressor from a file of dates, with a window in days before and after the
event and a choice between a proportionality regressor and a count regressor.

Two details transfer directly to commercial work. Holiday regressors are **centered** before estimation,
on the calendar mean or the overall mean, because an uncentered dummy is correlated with the intercept.
And the window is specified in days relative to the event, so the regressor follows the event when the
date moves. Both are conventions a commercial regression should copy.

### 8.3 What is convention rather than requirement

- A one-week holiday window is a default, not a finding.
- `K = 6` for monthly data is where the harmonic basis equals the dummy basis, not a
recommendation.
- A linear trend is a software default and the least defensible one in most demand series.
- `K = 5` weekly and `K = 10` daily are rules of thumb from one applied example, not optima.
- The 4-5-4 calendar is an industry convention with accounting consequences, not a standard.

An analyst who can separate a requirement from a convention can defend a specification.

## 9. Industry practice and case evidence

### 9.1 What statistical agencies do, and why it transfers

The X-13ARIMA-SEATS documentation is the most rigorous public statement of how moving holidays are handled
in production. The regressor is built from a date file, the window is specified in days relative to the
event, the regressor is centered, and the choice between proportionality and count is explicit. The
Australian Bureau of Statistics has used a two-part Easter regressor, with separate before and after
windows, because Easter affects retail and production differently on the two sides of the holiday.

The transfer to commercial demand is direct. A distribution centre shipping into retail has a pre-Easter
build and a post-Easter lull, and the two are not symmetric. A single dummy forces them to be.

### 9.2 What the forecasting literature recommends

Hyndman and Athanasopoulos recommend dynamic harmonic regression when the seasonal period is long: Fourier
terms for seasonality plus an ARIMA error process for short-term dynamics. They give three reasons.
Seasonal ARIMA with a period of 52 or 365 requires `m - 1` initial seasonal states and becomes
impractical. Fourier terms handle multiple seasonal periods by adding pairs at different frequencies. And
smoothness is controlled by `K` rather than by the data.

They also state the main limitation plainly: seasonality is assumed fixed over time. A harmonic basis
cannot let the seasonal shape evolve, which is a real constraint for products in a growth or decline
phase, and a reason to check the stability of seasonal coefficients across years.

### 9.3 What the evidence says about lag length

Almon's 1965 application is still the clearest worked example of choosing a lag from a mechanism rather
than a fit statistic. Estimating the distributed lag between capital appropriations and capital
expenditures on quarterly US manufacturing data, she found weights centred on eight to nine quarters, with
roughly 45% of the effect realised within a year and about 92% within two. The polynomial restriction made
the estimation possible; the economic question set the horizon.

The choice among lag orders has a known trade-off. AIC is not consistent and tends to over-select; BIC is
consistent at the cost of under-selecting in small samples. Koehler and Murphree compared the two for
order selection in 1988 and found the expected pattern: BIC more parsimonious, AIC better at short-horizon
prediction. On a 52-week demand series with four to eight candidate lags the two usually agree, and
disagreement signals that the sample is too short.

The deeper problem is the one criteria do not address. Selecting the order by inspecting t-statistics and
then reporting the final standard errors as if the order were known in advance understates the
uncertainty. The remedy is to fix the horizon from the mechanism before estimation and report the
sensitivity of the cumulative effect to `L`.

### 9.4 The Chinese New Year problem in practice

The Reserve Bank of Australia's work on seasonal adjustment of Chinese economic statistics is a useful
documented case. Chinese New Year falls between 21 January and 20 February on the Gregorian calendar, a
drift of about four weeks, and the RBA notes that year-on-year growth rates are distorted by it. The same
problem appears in any global demand series with a Chinese supply base: January and February shipments
depend on where the holiday fell, and a year-on-year comparison across a two-week shift compares different
production calendars.

The standard remedy is the interval regressor: the value in a period is the share of the holiday window
falling in that period, with separate before, during, and after windows. This is the approach the Census
documentation attributes to Bell and Hillmer, and it is implemented in Genhol. The commercial version is
usually simpler and should not be: a single dummy on "the week containing Chinese New Year" captures the
closure but not the pre-holiday build or the restock, and those are the parts that matter for a supply
plan.

### 9.5 Vendor and practitioner claims

Vendor case studies here are common and should be read as marketing until replicated. One planning vendor
reports that a machine-learning uplift model for Chinese New Year improved forecast accuracy by 3.5
percentage points and reduced bias by 44.6 percentage points in a client engagement. The figures are not
independently verified, the baseline is not described in enough detail to reproduce, and the sample is one
client. Treat it as an existence claim: it is possible to improve on a naive holiday dummy. Do not treat
the effect size as transferable. Grade D.

The same caution applies to any claim that a harmonic specification "captures" seasonality. A harmonic
basis is a smoother; whether it captures what matters is an out-of-sample question, and the answer differs
by item.

The gap between guidance and practice is organisational before it is technical. Holidays are keyed to a
calendar week because another team maintains the date table, and `K` is fixed by template because
per-series re-estimation is expensive at scale.

## 10. Failure modes and red flags

The first three failure modes are silent: the regression output looks healthy, the R-squared is high, and
the coefficient has the expected sign. The red flag is in the design, not the output.

| # | Failure mode | What it looks like | Why it happens | Red flag | Repair |
|---:|---|---|---|---|---|
| 1 | Spurious regression | High R-squared, large t, tiny Durbin-Watson | Two I(1) series regressed in levels | R-squared above 0.95 with DW below 0.5 | Difference, or test for cointegration |
| 2 | Moving holiday absorbed by a dummy | Seasonal coefficient changes sign between years | Dummy keyed to calendar slot, event moves | Residual spike at a different week each year | Event-keyed window regressor |
| 3 | Uncentered holiday dummy | Intercept shifts when a holiday is added | Dummy mean is `1/m`, not zero | Adding one holiday moves the intercept by the coefficient divided by `m` | Center on the calendar mean |
| 4 | Lag window too short | Cumulative effect too small; residual autocorrelation at lags 2 to 4 | Horizon chosen by software default | ACF spike just after the last included lag | Extend to the mechanism horizon, then test the endpoint |
| 5 | Lag length chosen by t-statistics | Jagged weights, alternating signs at long lags | Adding lags until the last is insignificant | Standard errors reported as if the order were pre-specified | Fix `L` before estimation; report sensitivity |
| 6 | Sum of standard errors | Interval three times too wide; includes zero | `SE(sum) = sum(SE)` instead of `1'V1` | Interval width grows linearly in the number of lags | Use the covariance matrix |
| 7 | Over-harmonised | Seasonal fit wobbles; last harmonic barely identified | `K` chosen by maximum R-squared | Condition number above 30; `K` approaching `m/2` | AICc or BIC with a stated cap |
| 8 | Unit root test on raw seasonal data | ADF fails to reject for a plainly stationary series | Seasonality read as persistence | ADF and KPSS disagree in the "both reject" cell | Deseasonalise, then test |
| 9 | Random-split validation | Holdout error far better than production error | Future weeks used to predict past weeks | Adjacent weeks split across train and test | Rolling or expanding origin |
| 10 | 53-week year annualised through | Prior-year comparison off by about 2% | 371 days treated as 364 | Every fourth or fifth year shows a spurious dip | Exclude, restate, or add an indicator |
| 11 | Holiday window straddling a boundary | Holiday coefficient unstable across years | Whole window assigned to one period | Coefficient jumps when the event date crosses a boundary | Proximity regressor with shares |
| 12 | Double counting a holiday and a seasonal peak | Holiday coefficient and seasonal peak both shrink | Both regressors target the same week | Sum of the two is stable while each is unstable | Report the combined fitted value |
| 13 | Trend fitted across a level shift | Trend steep and significant; residuals show a step | A new customer or a lost account treated as growth | Residual plot has a flat segment then a jump | Add a level-shift regressor or split the sample |

Failure modes 2, 3, and 5 are the ones that survive a coefficient review. Each produces a plausible number
with a plausible sign, and each is detectable only by checking the design against the calendar and the
mechanism.

## 11. Assumptions and limits

| # | Assumption | What it says | What breaks it | How to check |
|---:|---|---|---|---|
| 1 | Correct functional form | The outcome is linear in the included terms | A threshold, a saturation, or a multiplicative seasonal pattern | Residual plot against fitted values and against each regressor |
| 2 | Exogeneity | Regressors are uncorrelated with the error | Price set in response to demand; promotion timed to weak weeks | Timing of the decision relative to the outcome; Topic 09 |
| 3 | No omitted time-varying confounder | Nothing else moves with both regressor and outcome | A competitor launch in the same weeks | Event log; Topic 06 |
| 4 | Independent errors | No serial correlation | Weekly data, overlapping horizons | ACF, Ljung-Box, Durbin-Watson |
| 5 | Homoskedastic errors | Constant error variance | Volume-dependent noise; a level shift | Residuals against fitted; White or Breusch-Pagan test |
| 6 | Stable parameters | Coefficients constant over the sample | A regime change, a new customer, a price reset | Recursive estimates; Chow test at a known break |
| 7 | Correct calendar mapping | Every observation maps to the intended period | Mixed week definitions; the 53rd week | Row count per year; week-boundary spot checks |
| 8 | Sufficient seasonal cycles | At least three cycles for a parametric seasonal estimate | Two years of weekly data | Count cycles; compare with a dummy basis |
| 9 | Correct trend specification | The chosen trend matches the data-generating process | A unit root modelled as a deterministic trend, or the reverse | ADF and KPSS both ways, with and without seasonality |
| 10 | Consistent outcome definition | The same thing is measured throughout | A unit change, a channel reclassification, a returns policy change | Data dictionary version against the observation dates |

Three limits follow from the table and are worth stating separately.

**This design cannot establish causality on its own.** A regression with a full calendar and a well-chosen
lag window estimates an association conditional on those controls. If the promotion was scheduled in weeks
that were already weak, the coefficient is a lower bound on the effect and nothing in the diagnostics
reveals it. The design that fixes this is Topic 08.

**The seasonal basis cannot adapt.** Fourier terms impose a fixed shape. A product moving from
introduction to maturity changes its seasonal profile, and a fixed basis will average the two. Refit
annually, or test for a break in the seasonal coefficients.

**The diagnostics have low power on 52 observations.** Ljung-Box at 52 lags with `n = 52` has almost no
power, and ADF with one year of data cannot distinguish a unit root from a trend-stationary process with
any confidence. Report the diagnostics, and report their limitations in the same sentence.

## 12. Exercises

Each exercise states a deliverable, the criteria that make it acceptable, and a time estimate. The
self-check answers in section 13 give full calculations for exercises 1, 3, 5, 6, 7 and 10 and design
rubrics for the rest.

### Exercise 1 — Build the Fourier basis by hand (calculation, 60 minutes)
Take `m = 52` and `t = 1 ... 52`. Construct the design columns for `K = 1` and `K = 2`. Evaluate the
seasonal contribution at weeks 1, 7, 13, 26, 39, 47 and 52 using `a_1 = -215`, `b_1 = 1,785`, `a_2 = 310`,
`b_2 = -180`. Compute the amplitude and the peak week for both harmonics.

**Deliverable:** a table of the four columns at the seven weeks, a second table of contributions, and the
amplitude and peak week for each harmonic.
**Acceptance criteria:** every sine and cosine value correct to four decimals; the week-47 total
reproduced longhand; amplitudes stated in units per week; peaks stated as weeks, not radians. **Time:** 60 minutes.

### Exercise 2 — Dummy versus harmonic (design, 25 minutes)
For `m = 52`, tabulate parameter counts, data requirements, extrapolation behaviour, and behaviour under a
moving holiday for the two bases. State the value of `K` at which the harmonic basis spans the same space
as the dummy basis.

**Deliverable:** a comparison table and a one-paragraph recommendation for a series with 30 months of
weekly history.
**Acceptance criteria:** parameter counts correct; the equivalence stated with its proof sketch; the
recommendation names the trade-off, not only the answer. **Time:** 25 minutes.

### Exercise 3 — Attenuation from a moving holiday (calculation, 20 minutes)
A dummy is keyed to week 13. Over five years, Easter falls in week 13 twice, week 14 twice, and week 15
once. The true peak-week effect is 900 units per week. Compute the expected estimated coefficient, the
attenuation as a percentage, and state where the missing effect goes.

**Deliverable:** the expected coefficient, the attenuation, and one sentence on the residual.
**Acceptance criteria:** expected value computed as a probability-weighted average; attenuation expressed
against the true effect; the residual consequence stated in units. **Time:** 20 minutes.

### Exercise 4 — Build a proximity regressor (design, 35 minutes)
A holiday window runs from 8 days before to 2 days after the event. For a monthly series in which the
event falls on the 3rd of a 31-day month, compute the share of the window in that month and in the
previous month. State the centering rule you would apply.

**Deliverable:** the two shares, the centering rule, and the resulting regressor values for both months.
**Acceptance criteria:** shares sum to one across the affected months; the centering mean is computed over
the full calendar; the distinction between proportionality and count is stated. **Time:** 35 minutes.

### Exercise 5 — Cumulative effect and its standard error (calculation, 25 minutes)
A four-week distributed lag gives `d = (900, 400, -500, -300)` with individual standard errors `(220, 200,
190, 185)` and `1'V1 = 30,625`. Compute the cumulative effect, its standard error, the 95% interval, the
interval you would get by assuming independence, and the interval from summing the standard errors.

**Deliverable:** the point estimate and three intervals, with the arithmetic shown.
**Acceptance criteria:** `1'V1` used directly rather than reconstructed; the degrees of freedom stated; a
one-sentence statement of which interval is correct and why. **Time:** 25 minutes.

### Exercise 6 — Choose K by information criterion (calculation, 30 minutes)
With `n = 52`, `p = 9 + 2K`, and residual sums of squares of 4,820,000, 3,960,000, 3,910,000 and 3,895,000
for `K = 1` to `4`, compute AIC and BIC for each and select.

**Deliverable:** a table of `ln(RSS/n)`, AIC, and BIC by `K`, with the selected order.
**Acceptance criteria:** the log term computed once and reused; both criteria applied; a statement of what
to do if they disagree. **Time:** 30 minutes.

### Exercise 7 — Read the two-by-two (interpretation, 20 minutes)
A series gives ADF `-2.31` against a 5% critical value of `-2.93`, and KPSS `0.61` against `0.463`. After
removing a fitted seasonal component, ADF gives `-4.12` and KPSS gives `0.21`. State the reading at each
stage, the most likely explanation, and the specification you would estimate.

**Deliverable:** two readings, one explanation, one specification with its counterfactual.
**Acceptance criteria:** null hypotheses stated correctly for both tests; the seasonal explanation given
in units per week; the counterfactual stated in words. **Time:** 20 minutes.

### Exercise 8 — Diagnose the spurious regression (diagnosis, 25 minutes)
Two unit-root series, both rising, are regressed in levels. R-squared is 0.96, the trend coefficient has
`t = 14`, and Durbin-Watson is 0.31. State the diagnosis, the test that confirms it, and two repairs with
their costs.

**Deliverable:** a diagnosis, a confirming test, and two repairs.
**Acceptance criteria:** Durbin-Watson interpreted against its bounds, not as a p-value; the repairs
distinguished by the assumption each requires. **Time:** 25 minutes.

### Exercise 9 — Order the diagnostic loop (design, 20 minutes)
You are given six outputs: a residual plot with a funnel, an ACF spike at lag 52, a Ljung-Box p-value of
0.03 at 8 lags, a recursive coefficient that moves by three standard errors, a holdout that beats the
seasonal naive by 1.2 times, and a condition number of 41. State the order in which you would act and
which single defect you would fix first.

**Deliverable:** a ranked list with a one-line justification per item and the first fix.
**Acceptance criteria:** variance defects before mean defects before inference; the first fix justified by
consequence, not by ease. **Time:** 20 minutes.

### Exercise 10 — Write the sentence (interpretation, 20 minutes)
Using the worked example, write the interpretation sentence for the promotion and for the December level.
Then state three specification defects that would overturn each.

**Deliverable:** two sentences with units and intervals, and three defects per sentence.
**Acceptance criteria:** units present; interval reported; no causal verb stronger than the design
supports; each defect checkable. **Time:** 20 minutes.

## 13. Self-check answers

### Answer 1 — the Fourier basis by hand

With `m = 52`, the angle is `theta_t = 2 pi t / 52 = pi t / 26`. The `K = 1` columns are `sin(theta_t)`
and `cos(theta_t)`; `K = 2` adds `sin(2 theta_t)` and `cos(2 theta_t)`.

| `t` | `theta` | `sin theta` | `cos theta` | `2 theta` | `sin 2 theta` | `cos 2 theta` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1208 | 0.1205 | 0.9927 | 0.2417 | 0.2393 | 0.9709 |
| 7 | 0.8458 | 0.7485 | 0.6631 | 1.6916 | 0.9927 | -0.1205 |
| 13 | 1.5708 | 1.0000 | 0.0000 | 3.1416 | 0.0000 | -1.0000 |
| 26 | 3.1416 | 0.0000 | -1.0000 | 6.2832 | 0.0000 | 1.0000 |
| 39 | 4.7124 | -1.0000 | 0.0000 | 9.4248 | 0.0000 | -1.0000 |
| 47 | 5.6790 | -0.5681 | 0.8230 | 11.3581 | -0.9350 | 0.3546 |
| 52 | 6.2832 | 0.0000 | 1.0000 | 12.5664 | 0.0000 | 1.0000 |

The `K = 1` contribution is `S1 = -215 sin theta + 1,785 cos theta`; the `K = 2` addition is `S2 = 310 sin
2 theta - 180 cos 2 theta`.

| `t` | `S1` | `S2` | `S1 + S2` |
|---:|---:|---:|---:|
| 1 | 1,746.07 | -100.58 | 1,645.49 |
| 7 | 1,022.75 | 329.49 | 1,352.24 |
| 13 | -215.00 | 180.00 | -35.00 |
| 26 | -1,785.00 | -180.00 | -1,965.00 |
| 39 | 215.00 | 180.00 | 395.00 |
| 47 | 1,591.15 | -353.69 | 1,237.46 |
| 52 | 1,785.00 | -180.00 | 1,605.00 |

Longhand for `t = 47`: `theta = 47 pi / 26 = 5.67903` radians (325.38 degrees), so `sin theta = -0.56806`
and `cos theta = 0.82298`.

```text
S1 = -215 x (-0.56806) + 1,785 x 0.82298 = 122.13 + 1,469.02 = 1,591.15 units per week
2 theta = 11.35806 rad; subtract 2 pi = 6.28319 to get 5.07488 rad
sin = -0.93502, cos = 0.35460
S2 = 310 x (-0.93502) - 180 x 0.35460 = -289.86 - 63.83 = -353.69 units per week
S1 + S2 = 1,591.15 - 353.69 = 1,237.46 units per week
```

Amplitudes and peaks use `R_k = sqrt(a_k^2 + b_k^2)` and `t_peak = (m / 2 pi k)(pi/2 - phi_k)` with `phi_k
= atan2(b_k, a_k)`:

```text
R_1 = sqrt(215^2 + 1,785^2) = sqrt(3,232,450) = 1,797.90 units per week
phi_1 = atan2(1,785, -215) = 1.69072 rad
t_peak = (52 / 6.28319)(1.57080 - 1.69072) = -0.992  ->  week 51.0

R_2 = sqrt(310^2 + 180^2) = sqrt(128,500) = 358.47 units per week
phi_2 = atan2(-180, 310) = -0.52625 rad
t_peak = (52 / 12.56637)(1.57080 + 0.52625) = 8.678  ->  weeks 8.7 and 34.7
```

Check: at `t = 13`, `S1 = -215` and `S2 = +180`, so the combined contribution is `-35.00`, which is why
week 13 sits near the trend line in the worked example. The columns are orthogonal over a complete cycle,
so adding `K = 2` leaves the `K = 1` coefficients essentially unchanged.

### Answer 2 — dummy versus harmonic

Dummies need 51 parameters at `m = 52`, one observation per parameter per year. Harmonics need `2K`,
estimated from all 52 observations. The two bases span the same space at `K = 26`, where the 52 columns
reproduce 52 week effects exactly. For 30 months, about 130 observations and 2.5 cycles, the
recommendation is `K = 2` or `K = 3`: the dummy basis is not estimable in any useful sense, and the
harmonic cost is that a genuine one-week spike that is not a holiday is smoothed into its neighbours.

### Answer 3 — attenuation

```text
E[gamma_hat] = 900 x (2/5) + 0 x (3/5) = 360 units per week
attenuation  = 1 - 360/900 = 0.60, or 60%
```

The three misfiring years do not lose the effect; the Easter volume appears in week 14 or 15 as an
unexplained residual. That inflates the residual variance and enlarges every other standard error, so the
coefficient is biased down while precision is overstated.

### Answer 4 — proximity regressor

The window is 11 days: 8 before, the event day, and 2 after. With the event on the 3rd it runs from the
26th of the previous month to the 5th of the current month.

```text
days_previous = 31 - 26 + 1 = 6;  days_current = 5;  total = 11
share_previous = 6/11 = 0.5455;   share_current = 5/11 = 0.4545
```

Centre each regressor by its mean over the full calendar, not by 0.5. If the event falls in the current
month in 7 of 20 years, the current-month mean share is about 0.35 and the regressor is `share - 0.35`.
Rubric: shares sum to one across affected periods; the centering mean is computed over the whole calendar,
not only event years; and the answer states whether the regressor measures intensity (proportionality) or
days (count), because the two coefficients carry different units.

### Answer 5 — cumulative effect

```text
sum d = 900 + 400 - 500 - 300 = 500 units per week
SE(sum) = sqrt(1'V1) = sqrt(30,625) = 175 units per week;  t(39, 0.975) = 2.022
95% CI = 500 +/- 2.022 x 175 = 500 +/- 353.9 = [146, 854]

Independent assumption: sqrt(220^2 + 200^2 + 190^2 + 185^2) = sqrt(158,725) = 398.4
95% CI = 500 +/- 805.6 = [-306, 1,306]

Summed standard errors: 220 + 200 + 190 + 185 = 795
95% CI = 500 +/- 1,607.7 = [-1,108, 2,108]
```

Only the first is correct. Both alternatives include zero and would support the conclusion that the
promotion did nothing. The true standard error is smaller because the lag weights are strongly negatively
correlated: the fit can move volume between adjacent weeks without changing the total, so the sum is far
better determined than any single weight.

### Answer 6 — information criteria

`ln(RSS/n)` is computed once per row; `p = 9 + 2K`.

| `K` | RSS | `ln(RSS/n)` | `p` | AIC | BIC |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,820,000 | 11.4370 | 11 | 616.72 | 638.18 |
| 2 | 3,960,000 | 11.2405 | 13 | 610.51 | 635.88 |
| 3 | 3,910,000 | 11.2278 | 15 | 613.85 | 643.12 |
| 4 | 3,895,000 | 11.2239 | 17 | 617.64 | 650.81 |

Worked for `K = 2`: `52 x 11.2405 = 584.51`; AIC `= 584.51 + 26 = 610.51`; BIC `= 584.51 + 13 x 3.9512 =
635.88`. Both select `K = 2`. If they disagreed, the choice would depend on purpose, AIC for one-step
forecasting and BIC when the order itself is the claim; on 52 observations a disagreement means the sample
cannot separate the orders, and the choice should be reported as a sensitivity.

### Answer 7 — the two-by-two

Raw: ADF does not reject a unit root (`-2.31 > -2.93`) and KPSS rejects stationarity (`0.61 > 0.463`), so
both point to a unit root. After removing the fitted seasonal: ADF rejects (`-4.12 < -2.93`) and KPSS does
not (`0.21 < 0.463`), so both point to stationarity.

The seasonal amplitude of 1,798 units per week is large against a trend of 2,184 units over the whole
year, so the raw series has a slow, large-amplitude swing that ADF reads as persistence. The specification
is a level regression with a deterministic linear trend, `K = 2` harmonics, the holiday regressors, and
the distributed lag. The counterfactual is that absent the intervention, volume would have continued on
the fitted line; if the category is in structural decline, the trend should be re-estimated on a longer
window.

### Answer 8 — spurious regression

Two independent I(1) series give a high R-squared, a large t, and a Durbin-Watson far below 2, because the
residual is itself strongly autocorrelated. Confirm by differencing both series and re-estimating: if the
coefficient collapses toward zero and loses significance, the level relationship was not real. Two
repairs: model the relationship in differences, which answers "how does a change in `x` relate to a change
in `y`" and discards level information; or test for cointegration and estimate an error-correction model,
which retains the level relationship but requires the cointegration assumption to be defended.

### Answer 9 — the diagnostic loop

Order: the funnel in the residual plot first, because the wrong error variance makes every standard error
unreadable; then the lag-52 ACF spike and the condition number of 41, which are mean-equation defects;
then the Ljung-Box result, recomputed on the repaired model; then recursive stability; then the holdout,
which answers a different question. The condition number of 41 says trend and seasonality are trading off,
so any seasonal fix must be re-checked against collinearity. First fix: the variance defect, since a model
with the wrong error variance cannot be improved by adding seasonal terms, and a funnel often means a log
scale or a level shift.

Rubric for the ordering: variance defects before mean defects before inference defects, because each stage
invalidates the diagnostics after it. Fixing the Ljung-Box result first scores zero, since the test was run
on a model whose error variance is wrong.

### Answer 10 — the sentence

Promotion: "The week-30 promotion raised net volume by 650 units per week of promotion (95% CI 23 to
1,277), with a peak effect of 1,200 in the promotion week and 550 given back over the following three
weeks." December: "December volume is 3,100 units per week below the seasonal norm after adjusting for
trend and seasonality, which is consistent with the Christmas closure rather than a demand decline."

Defects that would overturn the first: a lag window extending beyond three weeks, so the give-back is
understated; a competitor stock-out in the same weeks, so the lift is not attributable; and an unmodelled
price change. Defects that would overturn the second: a Christmas regressor keyed to week 52 when the
closure spans two weeks; an unrestated 53-week prior year; and an omitted December promotion.
## 14. Interview questions

**Q1. Why does ordinary least squares fail on a weekly demand series?** Strong: names serial correlation
and its effect on the variance of the coefficient, not only on the point estimate; notes that the
coefficient stays consistent, so the failure is inferential; gives the variance inflation `(1 + rho)/(1 -
rho)`. Weak: says "the data are not independent" and stops, or claims the coefficients are biased.

**Q2. You have three years of weekly data. How many Fourier pairs?** Strong: a range tied to `2K`
parameters against 156 observations, the Nyquist limit of 26, a proposal of `K = 2` to `3`, AICc or BIC as
the selector, and a stability check. Weak: says "six" because that is the monthly default, or picks `K` by
maximising R-squared.

**Q3. What is the difference between a seasonal dummy and a Fourier term?** Strong: the same basis at
different resolution; `K = m/2` is the equivalence point; the dummy is local and nonparametric, the
harmonic global and smooth; `m - 1` parameters against `2K`. Weak: says they are different methods for
different data, or that Fourier is "more advanced".

**Q4. Easter moves. How do you handle it?** Strong: key the regressor to the event date, use a window in
days relative to the event, centre it, and check its correlation with the seasonal terms; notes the
attenuation from a misfiring calendar-slot dummy. Weak: says "add a dummy for Easter" without saying which
week, or assumes the computus matches the retail calendar.

**Q5. A promotion shows +1,200 in week 0 and -550 across the next three weeks. What is the effect?**
Strong: 650, with a standard error from the covariance matrix rather than the sum of the individual
standard errors; states that the peak is not the effect. Weak: reports 1,200, or reports 650 with an
interval that includes zero because the standard errors were summed.

**Q6. How do you choose the lag length?** Strong: from the mechanism first, then test the endpoint, then
report the sensitivity of the cumulative effect to `L`; notes that selecting by t-statistics invalidates
the reported standard errors. Weak: "add lags until the last one is insignificant", or "use AIC" with no
mechanism.

**Q7. ADF says unit root, KPSS says stationary. What now?** Strong: the nulls are opposite, so the
combination is informative; check whether seasonality was removed first; consider a break or a
trend-stationary process with low ADF power; state the counterfactual each specification implies. Weak:
picks the test that supports the desired conclusion, or averages the two.

**Q8. What is spurious regression and how do you detect it?** Strong: Granger and Newbold; high R-squared
with a Durbin-Watson far below 2 on trending series; detect by differencing and re-estimating or by a
cointegration test; explains that the residual is non-stationary, so the t statistics are invalid. Weak:
says "correlation is not causation", which is a different problem.

**Q9. Why is a random train-test split wrong here?** Strong: adjacent weeks are correlated, so a random
split puts near-duplicates on both sides and the holdout error is optimistic; the fix is a rolling or
expanding origin; the holdout should contain a moving holiday. Weak: says "use cross-validation" without
changing the split, or says random splits are fine with enough data.

**Q10. How do you know the seasonal specification is adequate?** Strong: residual ACF at lags `m` and
`2m`, Ljung-Box at the seasonal lag, comparison against a seasonal naive benchmark on a holdout containing
a holiday, and stability of the seasonal coefficients across years. Weak: cites R-squared, or says "the
seasonal terms are significant".

**Q11. What would make you distrust a holiday coefficient of +2,400 with t = 12?** Strong: a window that
is too narrow or too wide, an uncentered regressor, collinearity with the seasonal peak, a holiday that
moved within the sample, and a level shift in the same weeks; a large t on 52 observations is easy with a
trending regressor. Weak: says "t = 12 is significant, so it is fine".

## 15. Key terms

- **Autocorrelation** — correlation between a series and its own lagged values, at a stated lag.
- **Amplitude** — peak deviation of a harmonic from its mean, `sqrt(a^2 + b^2)`, in outcome units.
- **Centering** — subtracting a regressor's calendar mean so it is orthogonal to the intercept.
- **Cointegration** — two I(1) series whose linear combination is stationary.
- **Cumulative effect** — sum of lag weights to a stated horizon, in outcome units per regressor unit.
- **Distributed lag** — a model in which the outcome depends on current and past regressor values.
- **Durbin-Watson** — a first-order autocorrelation statistic, near 2 when absent.
- **Fourier term** — a sine or cosine at an integer harmonic of the fundamental frequency.
- **Harmonic order (K)** — number of sine-cosine pairs, giving `2K` columns.
- **Holiday window** — the span of days around an event over which its effect acts.
- **Impulse response** — the path of the outcome after a one-unit shock.
- **Koyck lag** — infinite geometric lag with weights `d_0 lambda^j`.
- **Lag weight** — the coefficient on one lag of the regressor.
- **Ljung-Box statistic** — portmanteau test for autocorrelation up to a stated lag.
- **Moving holiday** — a holiday whose Gregorian date changes year to year.
- **Nyquist limit** — the highest identifiable frequency, `K <= m/2`.
- **Phase** — horizontal shift of a harmonic, `atan2(b, a)`, in radians.
- **Proximity regressor** — the share of a holiday window falling in a period.
- **Seasonal dummy** — a 0/1 indicator for one position in the seasonal cycle.
- **Seasonal period (m)** — observations in one seasonal cycle.
- **Serial correlation** — see autocorrelation; the two are used interchangeably here.
- **Spurious regression** — a high-R-squared regression between independent non-stationary series.
- **Stationarity** — constant mean, variance, and autocovariance depending only on lag.
- **Unit root** — an autoregressive root at one; shocks do not decay.

## 16. Sources

**Grade A — peer-reviewed methods.**

- Granger, C. W. J., and Newbold, P. (1974). "Spurious regressions in econometrics." *Journal of Econometrics*, 2(2), 111–120. doi:10.1016/0304-4076(74)90034-7. The spurious regression result and the Durbin-Watson diagnostic.
- Dickey, D. A., and Fuller, W. A. (1979). "Distribution of the estimators for autoregressive time series with a unit root." *JASA*, 74(366), 427–431. doi:10.1080/01621459.1979.10482531. The ADF test.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., and Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root." *Journal of Econometrics*, 54(1–3), 159–178. doi:10.1016/0304-4076(92)90104-Y. The KPSS test.
- Almon, S. (1965). "The distributed lag between capital appropriations and expenditures." *Econometrica*, 33(1), 178–196. doi:10.2307/1911806. The polynomial lag restriction and the application in section 9.3.
- Akaike, H. (1974). "A new look at the statistical model identification." *IEEE Transactions on Automatic Control*, 19(6), 716–723. doi:10.1109/TAC.1974.1100705.
- Schwarz, G. (1978). "Estimating the dimension of a model." *Annals of Statistics*, 6(2), 461–464. doi:10.1214/aos/1176344136. The BIC.
- Koehler, A. B., and Murphree, E. S. (1988). "A comparison of the Akaike and Schwarz criteria for selecting model order." *JRSS Series C*, 37(2), 187–195. doi:10.2307/2347339. AIC versus BIC in order selection.
- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition. OTexts. Dynamic harmonic regression, Fourier terms, and the `K = 5` weekly and `K = 10` daily guidance. Textbook; grade A for method, not a primary source for any statistic.

**Grade B — official statistical practice.**

- U.S. Census Bureau. *X-13ARIMA-SEATS Reference Manual*: `easter[n]`, `labor[n]`, `thank[n]`, `easterstock[n]` and their default windows. census.gov.
- U.S. Census Bureau. *Genhol*, a holiday regressor generator; proportionality versus count, and centering. census.gov.
- Australian Bureau of Statistics. Seasonal adjustment methodology and the two-part Easter regressor. abs.gov.au.
- Reserve Bank of Australia. Seasonal adjustment of Chinese economic statistics, and the Chinese New Year timing distortion. rba.gov.au.
- Eurostat. *ESS Guidelines on Seasonal Adjustment*, on holiday regressors and calendar effects.

**Grade C — industry and company evidence.**

- National Retail Federation. Retail 4-5-4 calendar documentation, including the 53rd week.
- Company demand-planning case studies adding holiday regressors to a statistical baseline; outcomes are company-specific.

**Grade D — vendor marketing.**

- EyeOn. Vendor case study reporting a 3.5 percentage point accuracy gain and a 44.6 percentage
point bias reduction from a Chinese New Year uplift model in one client engagement. Not independently
verified; baseline not described sufficiently to reproduce. Accessed 2026.

**Grade E — press and anecdote.**

- Trade press coverage of holiday forecasting failures. Used for illustration only, not as
evidence for any effect size.

**Not verified, and therefore stated qualitatively.** The bell-and-hillmer attribution for the interval
regressor is made in the Census Genhol documentation without a retrievable primary reference in this
search; the method is described here without a volume or page citation. No standalone `holiday` package
for R was found; the `forecast` package's `easter()` helper is the only documented function cited.

## 17. Links

- [Module 04 index](_index.md) — topic list and prerequisites.
- [04.02 Probability, Sampling, Bootstrap, and Intervals](02_probability-bootstrap.md) — the
interval machinery used in sections 6.6 and 13.
- [04.03 Hypothesis Tests and Multiple Comparisons](03_hypothesis-tests.md) — the decision rule
applied to the diagnostic tests.
- `05_autocorrelation-robust-errors.md` (planned) — the HAC standard errors referenced in section
7 and Phase G.
- `08_difference-in-differences.md` (planned) — the design that establishes the counterfactual
this topic can only assume.
- [01A.04 Calendars, Weeks, and Time Zones](../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
— week definitions, the 4-5-4 calendar, and the 53rd week.
- [Course survey: Statistics and Causal Inference](../../../course/04_statistics_causal_inference.md)
— the module narrative.
- [Course survey: Time Series Forecasting](../../../course/07_time_series_forecasting.md) — the
forecasting counterpart to this topic.
- [Foundations module index](../_index.md) — the wider module map.

