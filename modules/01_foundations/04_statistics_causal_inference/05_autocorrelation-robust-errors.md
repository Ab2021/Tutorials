> Module 04 — Statistics and Causal Inference
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.03, 04.04

# Autocorrelation, Heteroskedasticity, and Robust Errors

---

## 1. Why this topic exists

A category team runs a 26-week display test in 120 stores. The analyst regresses weekly units on
a treatment indicator with store and week fixed effects. The coefficient is 42.0 units per
store-week, the conventional standard error is 11.2, and the t-statistic is 3.75. The rollout is
approved.

Six weeks later a reviewer reruns the same regression with standard errors clustered by store.
The coefficient is still 42.0. The standard error is 33.4, the t-statistic is 1.26, and the 95%
interval runs from −24.1 to +108.1. Nothing about the estimated effect changed. Everything about
the confidence changed, and the confidence was the input to the decision.

That is the subject of this topic. In supply chain data the coefficient is often approximately
right while the uncertainty around it is badly wrong. The two failures have different causes,
different diagnostics, and different repairs, and conflating them is the most common statistical
error in commercial analytics.

**What the textbook standard error assumes.** The ordinary least squares (OLS) variance formula
is `sigma^2 (X'X)^-1`. It is derived under errors that are independent and identically
distributed (i.i.d.) with constant variance `sigma^2`. The word *independent* does the heavy
lifting: it says that each row of the design matrix contributes a new piece of information, and
that no row tells you anything about another. Every observation in a supply chain dataset
violates this in at least one direction.

| Assumption | What it means | What supply chain data does | Consequence if violated |
|---|---|---|---|
| Errors independent across rows | Each observation is a fresh draw | Store-weeks share a store; lanes share a carrier; weeks share a promotion | Standard errors too small; t too large |
| Constant error variance | One `sigma^2` for every row | Variance grows with volume, season, and item velocity | Standard errors wrong, usually too small in aggregate |
| No shared shocks | Units fail independently | Regional weather, port congestion, a national promotion wave | Standard errors too small; false significance |
| Correct mean equation | No omitted dynamics | Trend, seasonality, holidays, lagged response | The coefficient itself is wrong |

The first three rows are inference problems. The fourth is an identification problem, and no
correction in this topic repairs it. Topic 04 built the mean equation; this topic assumes it is
right and asks whether the interval around it is honest.

**Why the distinction matters commercially.** A wrong coefficient sends a programme in the wrong
direction. A wrong standard error sends a programme forward with false confidence, or kills a
real effect that looks noisy. The second is more common and harder to see, because the point
estimate is plausible and the software prints a number with three decimal places.

**Provenance in one paragraph.** Heteroskedasticity-consistent (HC) errors were stated by Eicker
in the 1960s and made standard by White (1980). Heteroskedasticity- and autocorrelation-consistent
(HAC) errors were made usable by Newey and West (1987), whose Bartlett weighting guarantees a
positive semi-definite covariance matrix. Cluster-robust errors came from survey statistics, where
Kish formalised the design effect and Moulton applied it to regression. The few-cluster problem was
documented by Bertrand, Duflo, and Mullainathan (2004) and Cameron, Gelbach, and Miller (2008).
Each repair answers a specific failure; none is a general "robust" switch.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** what the OLS standard error formula assumes, and **name** the three assumptions that
   supply chain panels violate most often.
2. **Distinguish** a wrong coefficient from a wrong standard error, and **say** which problems in
   a regression table imply which.
3. **Compute** the design effect `1 + (m - 1) rho` and the effective sample size `n / deff` by
   hand, and **explain** the result to a planner in units of stores rather than rows.
4. **Choose** among HC, HAC, one-way cluster, two-way cluster, and Driscoll–Kraay errors for a
   stated data structure, and **justify** the choice from the dependence, not from the software
   default.
5. **Select** a HAC truncation lag from the sample length and the seasonal cycle, and **show**
   that the conclusion is insensitive to a plausible range.
6. **Diagnose** serial correlation with the Durbin–Watson statistic, the Breusch–Godfrey test, and
   the residual autocorrelation function, and **explain** when Durbin–Watson is invalid.
7. **Report** an estimate with the error type, the cluster definition, the lag or bandwidth, and
   the number of clusters, in a form a reviewer can reproduce.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: two datasets with the same fitted line and different residual behaviour. The
coefficient is identical. The interval is not.

```text
   PANEL A — INDEPENDENT ERRORS
   units
    60 |                    *
       |          *    *
    50 |     *         *        *
       |  *    *   *        *
    40 |*           *   *
       |
    30 |
       +----------------------------> week
   fitted:     units = 34.0 + 2.0 t
   residuals:  +  -  +  +  -  -  +  -  +  -     signs flip most steps

   PANEL B — POSITIVELY AUTOCORRELATED ERRORS
   units
    60 |  *
       |     *
    50 |        *   *
       |              *
    40 |                 *   *
       |                       *
    30 |
       +----------------------------> week
   fitted:     units = 34.0 + 2.0 t           SAME SLOPE, SAME FIT
   residuals:  +  +  +  +  -  -  -  -  +  +   long runs of one sign

   b = 2.0 in both panels.
   naive SE(b) = 0.32 in A, 0.95 in B, because B has rho = 0.70.
   95% CI in A: [1.36, 2.64]   excludes zero
   95% CI in B: [0.10, 3.90]   includes zero
```

The picture has three properties worth memorising.

**Property 1 — The unit of independence is not the row.** In a store-week panel the row is a
store-week, but the independent unit is closer to the store. In a lane-month panel it is the lane.
In a promotion evaluation it is the promotion event or the store, depending on how the treatment
was assigned. The design matrix has no column that records this. The analyst supplies it.

**Property 2 — Correlation shrinks the effective sample size, not the row count.** A thousand
observations with a within-store correlation of 0.35 do not contain a thousand observations'
worth of information. They contain roughly a tenth of it for a store-level question. The
software still counts rows, which is why the naive standard error is optimistic.

```text
   ROWS -> INFORMATION

   3,120 store-weeks          120 stores             about 12
   +---------------+          +-----------+          independent
   | x x x x x x x |   -->    |     O     |   -->    store-
   | x x x x x x x |          |     O     |          equivalents
   | x x x x x x x |          |     O     |          (rho = 0.35,
   +---------------+          +-----------+           m = 26)
     rows counted              clusters              information counted
```

**Property 3 — The coefficient and the standard error fail separately.**

| Failure | What moves | Typical cause | Repair |
|---|---|---|---|
| Wrong coefficient | Point estimate | Omitted variable, wrong functional form, simultaneity, measurement error | Design and specification, Topics 06 to 10 |
| Wrong standard error | Interval width | Serial correlation, heteroskedasticity, clustering | This topic |

A regression can have both failures at once, and the robust standard error does not fix the
coefficient. It can make a wrong coefficient look precisely estimated, which is worse than
leaving it noisy.

**The sandwich, stated once.** The OLS coefficient is `b = (X'X)^-1 X'y`. Its variance is
`(X'X)^-1 [Var(X'e)] (X'X)^-1`. Under i.i.d. errors the middle term is `sigma^2 X'X`, and the
whole expression collapses to `sigma^2 (X'X)^-1`. Robust estimators differ only in how they
estimate that middle term from the residuals.

```text
   NAIVE      V(b) = sigma^2 (X'X)^-1
   SANDWICH   V(b) = (X'X)^-1  Omega  (X'X)^-1

              Omega = sum over rows of w_it  e_it^2  x_it x_it'
              w_it  = 1            HC0
                    = n/(n-k)      HC1
                    = 1/(1-h_it)   HC3   (h_it = leverage)

   The point estimate is untouched. Only Omega changes.
```

> **Note:** The sandwich corrects the variance, not the mean. If the residuals are autocorrelated
> because a variable was omitted, the robust standard error is a precise statement about an
> imprecise model. Always ask whether the residual pattern is a nuisance or a specification
> error. The distinction is not visible in the standard error.

### 3.1 Why supply chain data is unusually exposed

Four structural features make this problem worse here than in most applied fields. Panels are
short and wide: four hundred stores and fifty-two weeks is normal, so the cluster count looks
safe while the within-cluster correlation makes the effective count small. Treatments are
assigned in groups: a promotion runs in a wave of stores, a supplier programme covers a region,
and a carrier change affects a lane, so the cluster must follow the assignment rather than the
row. Series are seasonal and trending, which produces positive autocorrelation by construction
and makes the variance grow with the mean. And the outcome is a count or a value with a floor:
units per store-week are non-negative and often intermittent, so homoskedasticity is untenable
before any test is run.

None of these is exotic. They are the default state of the data, which is why the default
standard error is the wrong one.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A definition without one is the defect
this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Standard error | Estimated standard deviation of an estimator across repeated samples | Read as the spread of the data | Outcome units, or a ratio |
| Sampling variance | Variance of an estimator over repeated samples | Confused with residual variance | Outcome units squared |
| Spherical errors | Errors with constant variance and zero covariance across observations | Assumed whenever OLS is run | Variance-covariance matrix `sigma^2 I` |
| Independent and identically distributed (i.i.d.) | Each observation drawn independently from the same distribution | Asserted from a random-looking table | Distributional property |
| Homoskedasticity | Constant error variance across all observations | Checked by eyeballing one residual plot | Constant `sigma^2` |
| Heteroskedasticity | Error variance that changes with regressors or fitted values | Treated as a bias in the coefficient | Function of `x` or of the mean |
| Serial correlation | Correlation between errors at different times | Used interchangeably with autocorrelation | Coefficient at a stated lag |
| Autocovariance | Covariance of a series with its own lag | Reported without the lag | Outcome units squared at lag `j` |
| Autocorrelation function (ACF) | Autocovariance divided by variance, as a function of lag | Read without stating the sampling frequency | Dimensionless, lag in periods |
| Partial autocorrelation function (PACF) | Correlation at lag `k` after removing shorter lags | Interpreted as a structural statement | Dimensionless, lag in periods |
| First-order autoregression, AR(1) | `e_t = rho e_{t-1} + u_t` with `u` white noise | Applied to a series with a weekly cycle | `rho` in [−1, 1] |
| Persistence | How slowly a shock decays | Confused with a trend | Half-life in periods |
| Effective sample size | Number of independent observations with the same information content | Reported as the row count | Independent observations |
| Design effect (deff) | Ratio of the true variance to the i.i.d. variance | Computed after the model is chosen | Dimensionless, `>= 1` |
| Intra-cluster correlation (ICC) | Average correlation between two observations in the same cluster | Assumed zero whenever a cluster identifier exists | `rho` in [−1, 1] |
| Cluster | A group within which observations may be correlated arbitrarily | Set to the row identifier | Identifier with a stated hierarchy |
| Cluster-robust variance estimator (CRVE) | Sandwich estimator allowing arbitrary within-cluster correlation | Called "robust" without naming the cluster | Covariance matrix plus cluster count |
| Sandwich estimator | `(X'X)^-1 Omega (X'X)^-1` | Used as a synonym for any robust error | Covariance matrix |
| HC0 | Heteroskedasticity-consistent sandwich with no small-sample correction | Reported as "White errors" without the variant | Covariance matrix |
| HC1 | HC0 with the `n/(n-k)` degrees-of-freedom correction | Compared across software without checking the default | Covariance matrix |
| HC3 | HC0 with a leverage correction, `1/(1-h_it)` | Assumed necessary at large `n` | Covariance matrix |
| Heteroskedasticity- and autocorrelation-consistent (HAC) | Sandwich using weighted autocovariances of the score | Applied to a panel as if it were one time series | Covariance matrix plus bandwidth |
| Newey–West estimator | HAC estimator with Bartlett weights | Treated as the only HAC option | Bandwidth `m` |
| Bartlett kernel | Weight `1 - j/(m+1)` on lag `j` | Chosen without checking sensitivity | Kernel weights |
| Bandwidth | Number of lags over which autocovariances are summed | Set by software default | Integer lags |
| Truncation lag | The largest lag included in a HAC estimator | Confused with the model's lag order | Integer periods |
| Two-way clustering | Clustering on two dimensions simultaneously, such as store and week | Assumed to equal the sum of two one-way corrections | Two identifiers |
| Driscoll–Kraay estimator | HAC applied to period-by-period cross-sectional averages | Used with a short time dimension | Bandwidth plus `T` |
| Few-cluster problem | Downward bias in CRVE when the number of clusters is small | Diagnosed from the row count | Number of clusters `G` |
| Wild cluster bootstrap | Bootstrap that resamples residual signs within clusters | Used with one treated cluster | Bootstrap replications |
| Degrees-of-freedom correction | Small-sample scaling of a robust variance | Omitted when comparing software outputs | Scalar multiplier |
| Durbin–Watson statistic | Test statistic for first-order autocorrelation, near 2 when absent | Reported with a lagged dependent variable | Statistic in [0, 4] |
| Breusch–Godfrey test | Lagrange multiplier test for serial correlation to a stated order | Run at an order chosen after seeing the ACF | Chi-squared with `p` degrees of freedom |
| White test | Test for heteroskedasticity based on squared residuals | Confused with White standard errors | Chi-squared statistic |
| Wooldridge panel test | Test of no first-order serial correlation in a panel | Applied to a time series rather than a panel | F statistic |
| Ljung–Box statistic | Portmanteau test for autocorrelation up to a stated lag | Run at a lag chosen post hoc | Chi-squared statistic |
| Model-based standard error | Standard error from an assumed error distribution | Compared with design-based errors as if equivalent | Covariance matrix |
| Design-based standard error | Standard error from the sampling or assignment design | Assumed to require survey weights | Covariance matrix |
| Coverage | Fraction of intervals that contain the true parameter across repeated samples | Replaced by "the interval looks reasonable" | Proportion, nominal 0.95 |
| Over-rejection | Rejecting a true null more often than the nominal rate | Inferred from a single significant result | Proportion at a stated nominal level |
| Moulton factor | Variance inflation from clustering, `1 + (m-1) rho` for equal clusters | Applied when clusters are very unequal | Dimensionless |

Four of these carry most of the interpretive weight.

**Design effect versus effective sample size.** The design effect is a variance ratio; the
effective sample size is `n / deff`. They contain the same information, and the second is easier
to explain. "Your 3,120 rows carry about 320 rows of information" lands in a meeting. "The design
effect is 9.75" does not.

**HAC versus clustered.** HAC allows correlation over time within one series. Clustering allows
arbitrary correlation within a group, in any direction. A panel of stores observed over weeks has
both time dependence and cross-sectional dependence; HAC on the pooled panel addresses the first
and not the second, and often does the first badly because the kernel crosses store boundaries.
Cluster by store, and add a second dimension if weeks share shocks.

**Model-based versus design-based.** A model-based standard error comes from an assumed
likelihood. A design-based standard error comes from how the data were generated or assigned.
The second is the right frame for a promotion rollout, because the randomness that matters is
which stores got the display and which weeks got the promotion, not a hypothetical error
distribution.

**Few clusters is not a sample-size problem.** Adding rows within the same clusters does not help.
Adding clusters does. A dataset with two million rows and twelve clusters has the same
few-cluster problem as one with two thousand rows and twelve clusters.

## 5. The mechanics, step by step

This section is a procedure in six phases. The order matters: a variance estimator chosen before
the dependence is diagnosed is a guess with a decimal point.

### Phase A — Write down what the inference assumes

**Step 1. State the unit of observation and the unit of independence.** "The row is a store-week.
The treatment is assigned at the store. Therefore the cluster is the store." Write both
sentences. If they differ, the second one governs every standard error in the analysis.

**Step 2. Write the model and the estimand in one line each.** The variance estimator depends on
the regressor of interest. A store-level treatment has a larger design effect than a
within-store, week-varying regressor, because the former is constant within the cluster.

**Step 3. Count the clusters before fitting anything.** `G` stores, `L` lanes, `E` promotion
events. If `G` is below about 30, the asymptotic cluster-robust interval is optimistic and you
will need a small-sample correction or a bootstrap. If `G` is below about 10, asymptotic cluster
inference is not credible and the design should be reconsidered.

**Step 4. Count the effective clusters.** `G / deff` with `deff = 1 + (m - 1) rho`. The number is
usually smaller than expected and is the number to quote when someone asks how much evidence
there is.

### Phase B — Diagnose the error structure

**Step 5. Plot residuals against time, within cluster.** Look for a trend, a seasonal wave, a
variance funnel, and runs of one sign. A funnel means heteroskedasticity. Runs mean positive
autocorrelation. Both mean the naive standard error is wrong.

**Step 6. Plot residuals against fitted values and against volume.** In supply chain data the
variance usually grows with the mean. Slow-moving items have small counts and Poisson-like
variance; fast movers have large counts and proportional variance.

**Step 7. Compute the ACF and PACF to at least twice the seasonal period.** A spike at lag 1
means short-run dependence. A spike at lag 7 for daily data or lag 52 for weekly data means the
mean equation left seasonality in the residual.

**Step 8. Test, using the test that matches the structure.** Durbin–Watson for first-order
autocorrelation in a time series with no lagged dependent variable. Breusch–Godfrey for higher
orders and for models with lagged dependent variables. Wooldridge's panel test for a short panel.
White's test for heteroskedasticity. State the null and the order.

**Step 9. Estimate `rho` and the cluster sizes.** The average within-cluster correlation and the
distribution of `m` are the two inputs to the design effect. Unequal cluster sizes raise the
design effect above the equal-cluster formula.

### Phase C — Choose the variance estimator

**Step 10. Match the estimator to the dependence, not to the software default.**

| Data structure | Dependence present | Estimator | Cluster or bandwidth unit |
|---|---|---|---|
| Single time series, no groups | Serial correlation | HAC (Newey–West) | Truncation lag `L` |
| Cross-section, heteroskedastic | Variance varies with size | HC1 or HC3 | None |
| Panel, treatment at group level | Within-group correlation | One-way cluster | Assignment group |
| Panel, group and time shocks | Both dimensions | Two-way cluster | Group and period |
| Long panel, common shocks | Cross-sectional and serial | Driscoll–Kraay | Bandwidth plus `T` |
| Few clusters | Unknown within-cluster dependence | Wild cluster bootstrap | Cluster |
| Overlapping forecast horizons | Induced moving-average errors | HAC with `L >= horizon - 1` | Horizon |

**Step 11. Prefer the coarser cluster when in doubt.** Clustering at a level above the assignment
is conservative; clustering below it is anti-conservative. A store-clustered error when the
treatment is at region level still understates, because stores in a region share the treatment.
Cluster at the level at which the treatment varies, or one level above.

**Step 12. Do not cluster on the row identifier.** That produces the naive standard error with
extra steps. The cluster variable must group rows that could plausibly share an unobserved
shock.

**Step 13. Use two-way clustering when both dimensions carry shocks.** Store and week is the
standard pair for retail panels: stores differ persistently, and weeks contain national
promotions, weather, and holidays. Two-way clustering is not the sum of the two one-way
estimates; it is the sum minus the intersection, and it can be larger or smaller than either.

**Step 14. Consider Driscoll–Kraay when the cross-section is large and the time dimension is
long enough.** The estimator averages the score across units in each period and applies a HAC
kernel to that average. It requires a long `T`; with fifty-two weeks the asymptotics are thin,
and two-way clustering is usually the safer choice.

**Step 15. Decide whether the problem is the variance or the mean.** If the residual ACF decays
slowly and a lagged dependent variable is plausible, the mean equation is incomplete. Adding
lags changes the coefficient; changing the standard error does not. Run both and compare.

**Step 16. Check whether a lagged dependent variable is present.** With one, OLS is biased and
inconsistent in a short panel (the Nickell bias), and Durbin–Watson is invalid. The repair is an
instrument or a bias-corrected estimator, not a robust standard error.

### Phase D — Choose the lag or bandwidth

**Step 17. Start from the rule of thumb, then adjust for seasonality.** A common default is
`L = floor(4 (T/100)^(2/9))`. It gives 4 at `T = 100`, 4 at 200, 5 at 500, and 6 at 1,000. The
rule knows nothing about seasonality: for daily data with a weekly cycle, `L` should be at least
7, and for weekly data with an annual cycle it should be at least 52 if annual residual
dependence remains.

**Step 18. Check sensitivity across a plausible range.** Refit at `L`, `L + 2`, and `2L`. If the
significance of the coefficient changes across that range, the result is not robust and the
range must be reported.

**Step 19. Do not choose the lag by maximising significance.** That is selection on the test
statistic and it invalidates the reported interval. Choose from the sample length, the seasonal
period, and the forecast horizon.

**Step 20. Remember the truncation bias.** Too small a bandwidth over-rejects; too large a
bandwidth has low power. The first error is worse, because it produces false positives.

### Phase E — Handle few clusters

**Step 21. Use the `G - 1` degrees of freedom, not `n - k`.** Most software does this when asked.
The interval widens and the test becomes conservative, which is the right direction.

**Step 22. Apply a small-sample correction when `G` is below about 30.** The CR2 or CR3
corrections, or a wild cluster bootstrap, perform better than the asymptotic CRVE. The bootstrap
resamples residual signs within clusters and preserves the within-cluster dependence.

**Step 23. Check the distribution of cluster sizes.** A few very large clusters dominate the
variance and make the effective `G` smaller than the nominal `G`. Report the largest cluster's
share of rows.

**Step 24. Check the number of treated clusters.** Fifty clusters with two treated is not fifty
clusters of information. The wild bootstrap's performance depends on the number of treated
clusters, not only the total.

**Step 25. Do not add rows within clusters to fix precision.** It does not increase the effective
sample size for a cluster-level question. Add clusters, or change the design.

### Phase F — Report

**Step 26. Name the estimator.** "Store-clustered standard errors, 120 clusters" is a report.
"Robust standard errors" is not.

**Step 27. Report the cluster definition and the lag or bandwidth.** A reader must be able to
reproduce the interval from the description.

**Step 28. Report the naive and the corrected interval side by side when the difference is
material.** The comparison is the evidence that the correction matters, and it prevents a later
reviewer from "discovering" the naive number.

**Step 29. State what the interval does not cover.** Clustering handles dependence within the
named groups. It does not handle dependence across groups, a wrong mean equation, or selection
into the sample.

### 5.1 The decision ladder

| Rung | What you do | What it buys | What it costs |
|---|---|---|---|
| 1 | OLS with default errors | A number | Assumes spherical errors |
| 2 | Plot residuals and ACF | Diagnosis | No correction |
| 3 | HC1 or HC3 | Correct variance if independent | Ignores dependence |
| 4 | HAC with a stated lag | Correct variance for one series | Wrong for panels; bandwidth choice |
| 5 | One-way cluster | Arbitrary within-group dependence | Cluster choice; few-cluster bias |
| 6 | Two-way cluster or Driscoll–Kraay | Two sources of dependence | Few-cluster bias in two dimensions |
| 7 | Wild bootstrap or CR2 | Better few-cluster coverage | Computation; less familiar |
| 8 | Fix the mean equation | Correct coefficient | Requires design, not software |

Rungs 1 to 7 concern the variance. Rung 8 concerns the estimate. Most analyses stop at rung 5
and call it robust.

## 6. Worked example

A home goods retailer tests a new in-store display in 60 of its 120 stores for 26 weeks. The
outcome is units sold per store per week. All figures are illustrative but internally consistent,
and every calculation can be reproduced by hand.

### 6.1 The test and the model

```text
   units_it = a + b (treat_i x post_t) + store_i + week_t + e_it

   i = 1 ... 120 stores
   t = 1 ... 26 weeks
   n = 120 x 26 = 3,120 store-weeks
   treat_i = 1 for 60 stores, 0 for 60
   post_t  = 1 for weeks 14 to 26, 0 for weeks 1 to 13
```

The coefficient `b` is the difference-in-differences estimate: the change in treated stores
relative to the change in control stores. The estimand is the average effect of the display on
weekly units per store.

### 6.2 The naive table

| Term | Estimate | Naive SE | t | 95% CI | Units |
|---|---:|---:|---:|---|---|
| Display effect `b` | 42.0 | 11.2 | 3.75 | [20.0, 64.0] | units per store-week |
| Store fixed effects | 120 levels | — | — | — | units per store-week |
| Week fixed effects | 26 levels | — | — | — | units per store-week |
| Residual sd | 118.4 | — | — | — | units per store-week |
| R-squared | 0.71 | — | — | — | dimensionless |

The naive standard error is 11.2. With 3,120 rows and 145 parameters, the residual degrees of
freedom are `3,120 - 145 = 2,975`, so the critical value is 1.96 and the interval is
`42.0 +/- 1.96 x 11.2 = [20.0, 64.0]`. The effect is significant and the rollout looks justified.

### 6.3 The design effect, by hand

The residuals are correlated within store. The average within-store correlation across the
26 weeks is `rho = 0.35`, estimated as the mean pairwise correlation of residuals in the same
store. Each store contributes `m = 26` weeks.

```text
   n          = 120 x 26                     = 3,120 store-weeks
   m          = 26 weeks per store
   rho        = 0.35 average within-store correlation

   deff       = 1 + (m - 1) rho
              = 1 + 25 x 0.35
              = 1 + 8.75                     = 9.75

   n_eff      = n / deff
              = 3,120 / 9.75                 = 320.0 independent store-weeks

   stores_eff = 120 / 9.75                   = 12.3 independent store-equivalents

   SE multiplier = sqrt(deff) = sqrt(9.75)   = 3.122

   approximate cluster SE = 11.2 x 3.122     = 34.97 units per store-week
```

The arithmetic says that 3,120 rows carry about 320 rows of information, and that 120 stores
carry about 12 stores of information for a question that varies at store level. That is the
sentence to say out loud. It is also why adding a second year of weekly data to the same 120
stores does not fix the problem: `m` rises, `deff` rises, and `n_eff` barely moves.

The same arithmetic across three plausible designs:

| Design | `n` rows | `m` | `rho` | `deff` | `n_eff` | SE multiplier |
|---|---:|---:|---:|---:|---:|---:|
| 120 stores x 26 weeks | 3,120 | 26 | 0.35 | 9.75 | 320 | 3.12 |
| 400 stores x 52 weeks | 20,800 | 52 | 0.22 | 12.22 | 1,702 | 3.50 |
| 40 DCs x 60 weeks | 2,400 | 60 | 0.10 | 6.90 | 348 | 2.63 |

Read the second row: eight times the rows of the first, and the effective sample size is five
times larger, not eight. The design effect consumed the rest.

### 6.4 The four standard errors

| Estimator | SE | t | df | 95% CI | Verdict at 5% |
|---|---:|---:|---:|---|---|
| Naive OLS | 11.2 | 3.75 | 2,975 | [20.0, 64.0] | Significant |
| HC3 | 12.1 | 3.47 | 2,975 | [18.3, 65.7] | Significant |
| HAC, lag 4, pooled | 21.6 | 1.94 | 2,975 | [-0.3, 84.3] | Marginal |
| Cluster by store | 33.4 | 1.26 | 119 | [-24.1, 108.1] | Not significant |
| Cluster by store and week | 37.8 | 1.11 | 118 | [-32.8, 116.8] | Not significant |

The sequence tells a story.

- **HC3 barely moves.** Heteroskedasticity is present but small. The problem is dependence, not
  variance.
- **HAC doubles the standard error.** It captures some of the within-store serial correlation,
  but it truncates at four lags and it treats the pooled panel as one long series. It does not
  capture the common weekly shocks, and the kernel crosses store boundaries.
- **Clustering by store roughly triples it.** This is the estimator that matches the assignment
  level. The computed value, 33.4, is 4.5% below the design-effect approximation of 35.0, which
  is the expected accuracy when the regressor is not perfectly constant within cluster.
- **Two-way clustering widens it further.** Weeks share national promotions, weather, and
  holidays, so store-clustered errors still understate.

The interval that should be reported is the store-clustered one, with the two-way result as a
sensitivity check: `42.0 +/- 1.98 x 33.4 = [-24.1, 108.1]`.

> **Note:** The coefficient did not move across any of these estimators. That is the signature of
> a dependence problem rather than an identification problem. If the coefficient had moved, the
> robust standard error would have been a distraction from the real defect.

### 6.5 A daily series and the effective sample size

Now take a distribution centre's daily outbound units, 728 days (104 weeks), with a linear trend.
The residual autocorrelation is well described by an AR(1) with `rho = 0.72`.

```text
   T          = 728 daily observations
   rho        = 0.72

   variance inflation = (1 + rho) / (1 - rho)
                      = 1.72 / 0.28            = 6.143

   SE multiplier      = sqrt(6.143)           = 2.478

   T_eff      = T x (1 - rho) / (1 + rho)
              = 728 x 0.28 / 1.72
              = 728 x 0.16279                 = 118.5 independent days

   naive t    = 5.60
   corrected t = 5.60 / 2.478                 = 2.26
```

The trend coefficient is `b = 1.20` units per day per day. The naive standard error is
`1.20 / 5.60 = 0.214`, giving `95% CI [0.78, 1.62]`. The corrected standard error is
`0.214 x 2.478 = 0.531`, giving `95% CI [0.16, 2.24]`. Same coefficient, an interval 2.5 times
wider, and a claim that moves from strong to marginal.

The HAC lag rule gives `floor(4 x (728/100)^(2/9)) = floor(6.22) = 6`. The weekly cycle argues for
at least 7; a check at 14 gives 0.58 and does not change the conclusion. Report the range.

### 6.6 Wrong coefficient versus wrong standard error

| Problem | Effect on `b` | Effect on SE | Diagnostic | Repair |
|---|---|---|---|---|
| Serial correlation, exogenous regressors | None; unbiased and consistent | Too small | Residual ACF; Breusch–Godfrey | HAC or cluster |
| Heteroskedasticity, independent errors | None; unbiased | Wrong, usually too small | White test; residual funnel | HC1 or HC3 |
| Clustering | None if regressors exogenous | Too small | Design effect; ICC | Cluster at assignment level |
| Lagged dependent variable in a short panel | Biased and inconsistent | Too small | Nickell bias; Wooldridge test | IV, GMM, bias correction |
| Omitted variable | Biased | May look precise | Coefficient moves when a control is added | DAG, design |
| Measurement error in the regressor | Attenuated toward zero | — | Validation subsample | Instrument, correction |
| Simultaneity | Biased in an unknown direction | — | Theory, timing | Instrument, experiment |

The table is the most useful diagnostic in the topic. If the coefficient is stable across
specifications and the standard error is not, the problem is inference. If the coefficient moves,
the problem is identification, and no robust estimator will save it.

### 6.7 The sentence you are allowed to write

"The display test is associated with +42.0 units per store-week. With standard errors clustered
by store (120 clusters), the 95% interval is −24.1 to +108.1, so the data are consistent with
effects ranging from a meaningful loss to a large gain. With naive standard errors the interval
would be +20.0 to +64.0, about three times too narrow; the estimated design effect is 9.75, and
the 3,120 store-weeks are worth about 320 independent observations."

Three things would change the conclusion: a smaller within-store correlation (a different
outcome, such as a category-level total, often has one); a treatment assigned within store rather
than across stores (which would justify a smaller cluster effect); and a longer pre-period that
allows store-specific trends. All three are design facts, not software settings.

## 7. Code, query, or calculation

One compact specification in Python. The point is that only the covariance estimator changes.

```python
import numpy as np, pandas as pd, statsmodels.formula.api as smf

d = df.sort_values(["store", "week"]).reset_index(drop=True)
d["post"] = (d["week"] >= 14).astype(int)

base = smf.ols("units ~ treat:post + C(store) + C(week)", data=d).fit()

fits = {
    "naive":  base,
    "HC3":    base.get_robustcov_results(cov_type="HC3"),
    "HAC(4)": base.get_robustcov_results(cov_type="HAC", maxlags=4),
    "store":  base.get_robustcov_results(
                  cov_type="cluster", groups=d["store"]),
}

for name, m in fits.items():
    b  = m.params["treat:post"]
    se = m.bse["treat:post"]
    lo, hi = b - 1.98 * se, b + 1.98 * se
    print(f"{name:7s} b={b:6.2f} se={se:6.2f} t={b/se:5.2f} CI=[{lo:7.1f},{hi:7.1f}]")
```

Two details matter. The `get_robustcov_results` call changes only `m.bse` and the tests; `m.params`
is identical across all four fits. And `groups=d["store"]` must be the assignment unit, not the
row index. A two-way cluster is computed by summing the one-way score contributions and
subtracting the intersection, which most packages expose as a separate option rather than a
nested call.

## 8. Data, standards, and regulatory context

No regulator prescribes Newey–West errors. The governance that matters is the governance of the
identifiers and the honesty of the reporting, and both are weaker than they look.

### 8.1 Identifiers that define clusters

| Identifier | What it groups | What breaks it | Grade |
|---|---|---|---|
| Store ID | Repeated weekly observations of one store | Renumbering, closures, acquisitions | C |
| Item–location key | The same product at the same place | Re-keying after a range reset | C |
| Lane ID | Origin–destination–mode flows | Carrier change, route redefinition | C |
| Promotion event ID | Stores and weeks touched by one promotion | Overlapping promotions, waves | C |
| Supplier ID | Orders and deliveries from one vendor | Legal-entity changes, mergers | C |
| Region or sales district | Shared weather, labour market, policy | Boundary redraws | C |

These are master-data objects, not analysis choices. If a store is renumbered mid-panel, the
software sees two clusters where the business sees one, and the standard error is computed on a
false partition. Before clustering, check that the identifier is stable over the sample window
and that the hierarchy is a tree rather than a graph.

### 8.2 Calendars and buckets

The time bucket determines the lag structure and the seasonal period, and therefore the HAC
bandwidth and the autocorrelation test order.

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 8601 | Date, week, and ordinal representation | The week key used in every lag and cluster | B |
| IANA time zone database | Local offsets and daylight-saving rules | Which calendar day an event falls on | B |
| Retail 4-5-4 calendar | Fiscal weeks and the 53-week year | The period index; a 53rd week changes `T` | C |
| Company promotion calendar | Event windows and waves | The treatment timing and the cluster for shared shocks | C |

A 53-week year adds one period. With 52 weeks the design effect is `1 + 51 rho`; with 53 it is
`1 + 52 rho`. The difference is small, but the lag structure and the seasonal period change, and
a model that assumes 52 periods will misalign every subsequent year.

### 8.3 Precision and reporting conventions

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| Clinical-trial reporting convention | Unit of randomisation and clustering statement | Requires the unit, the cluster count, the intra-cluster correlation, and the adjustment method | C |
| Survey design literature (Kish) | Design effects in complex samples | The `1 + (m-1) rho` convention and its interpretation | A |
| Journal reporting guidelines | Regression tables | Increasingly require the error type and cluster count | C |
| Company measurement standard | Precision of counts, weights, and currency | Rounding and unit conventions that change the estimated variance | C |

The transferable convention from clinical trials is not the statistics; it is the reporting
discipline. A cluster-randomised trial must state the unit of randomisation, the number of
clusters, the intra-cluster correlation, and the method used to account for it. A promotion
evaluation should state the same four things. Very few do.

### 8.4 What is convention rather than requirement

- `L = floor(4 (T/100)^(2/9))` is a rule of thumb from Newey–West practice, not a theorem.
- Clustering at the assignment level is a design principle, not a legal requirement.
- The `G - 1` degrees-of-freedom correction is a convention, and some software omits it.
- "At least 42 clusters" is folklore, and the literature that produced it treats it as such.
- HC1 versus HC3 is a small-sample correction choice, not a statement about the data.
- A 95% interval is a convention; under clustering its true coverage may be 90% or 85%.

An analyst who can separate a requirement from a convention can defend an estimator choice. An
analyst who cannot will report whatever the default produced.

## 9. Industry practice and case evidence

### 9.1 Serial correlation in difference-in-differences

Bertrand, Duflo, and Mullainathan (2004) is the canonical demonstration. They surveyed 92
difference-in-differences papers in six journals (1990–2000), found an average of 16.5 periods per
unit, and found that almost none corrected for serial correlation. Using placebo laws in
state-level wage data, they showed that conventional standard errors severely understate the true
dispersion: a spurious effect significant at the 5% level appeared in up to 45% of placebo
interventions. The residual autocorrelations were 0.51, 0.44, and 0.33 at lags 1 to 3, a slowly
decaying correlogram rather than an AR(1).

Two lessons transfer. First, the treatment dummy is itself serially correlated (zero before the
rollout, one after), which compounds the problem. Second, parametric AR(1) corrections performed
poorly, largely because the estimated `rho` is downward biased in short panels. The repairs that
worked were clustering, block bootstrap, and collapsing to pre/post periods. Grade A for the
method and simulation; the rejection rates are from their design, not supply chain data.

### 9.2 The few-cluster problem

Cameron, Gelbach, and Miller (2008) showed that asymptotic cluster-robust tests over-reject when
the number of clusters is small. In their Monte Carlo design, a nominal 5% test rejected 6.8% of
the time with 30 clusters, 8.1% with 20, 11.8% with 10, and 20.8% with 5; their wild cluster
bootstrap reduced the distortion substantially.

Cameron and Miller's 2015 practitioner's guide is explicit that there is no clear-cut definition
of "few": depending on the design it may mean fewer than 20 clusters or fewer than 50, and the
threshold rises when clusters are unbalanced. They cite Bertrand–Duflo–Mullainathan rejection
rates of 0.063, 0.058, 0.080, and 0.115 at 50, 20, 10, and 6 clusters. Grade A.

The widely repeated "rule of 42" is folklore. MacKinnon and Webb (2017) trace it to Angrist and
Pischke's *Mostly Harmless Econometrics*, who suggested it "not entirely seriously," and show
that it fails for unbalanced clusters. It also fails when few clusters are treated: a design
with 100 clusters, 4 of them treated, is not a 100-cluster design for the treatment effect.
Grade A for the correction; the rule itself should be treated as a mnemonic, not a threshold.

### 9.3 HAC bandwidth and over-rejection

Lazarus, Lewis, Stock, and Watson (2018) evaluate HAC inference in practice and find that the
textbook Newey–West approach with conventional critical values can over-reject severely. In their
simulations, nominal 5% tests rejected 11% to 18% of the time under moderate or high persistence.
They recommend a much larger truncation, on the order of `1.3 sqrt(T)`, with fixed-`b` critical
values, which at `T = 200` means about 19 lags against 5 for the textbook rule.

That recommendation is not yet standard in commercial software, and it is not what most analysts
will produce. The practical response is not to adopt the largest bandwidth blindly, because power
falls; it is to report sensitivity across a range and to avoid claiming significance when the
result depends on a small bandwidth. Grade A for the simulation; transfer to a 26-week panel is
limited because the time dimension is short.

### 9.4 What practitioners do, and why

| Observed practice | Why it persists | Consequence |
|---|---|---|
| "Robust standard errors" with no type named | Software menu label | Reader cannot reproduce |
| HAC applied to a pooled panel | It is the default for time series | Cross-sectional dependence ignored |
| Clustering on the row identifier | The column exists and looks like an ID | Naive standard error, relabelled |
| Durbin–Watson reported with a lagged dependent variable | It is in the default output | Invalid test, often unnoticed |
| No cluster count reported | The software prints `G` only in some packages | Reader cannot judge the few-cluster risk |
| One specification, one standard error | The analysis was run once | No sensitivity, no evidence of robustness |

The gap is organisational before it is technical. The cluster variable is owned by a master-data
team, the promotion calendar by commercial, and the regression by an analyst who inherits both.
Nobody owns the sentence that says which error was used and why.

### 9.5 Vendor and analyst claims

Demand-planning and revenue-management vendors routinely report accuracy improvements from
machine learning without stating the error model, the validation split, or the clustering of the
evaluation sample. A claim that a model "reduced forecast error by 40%" is usually a point-accuracy
comparison on a holdout that treats store-weeks as independent, which is the same defect this
topic describes, applied to the evaluation rather than the regression.

The claim should be read as an existence claim at best: it is possible to improve point accuracy.
It does not establish that the intervals are calibrated, that the evaluation respected the panel
structure, or that the improvement survives a clustered standard error. Grade D, dated 2026, based
on the general form of vendor material rather than a specific audited study.

The same caution applies to "AI-powered" anomaly detection on event streams: a threshold
calibrated on independent rows will fire far too often on serially correlated data.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating store-weeks as independent | Standard error falls as rows are added within stores | Within-store correlation ignored | False significance; overconfident rollout | Cluster at store; compute design effect |
| Clustering below the assignment level | Treatment assigned to stores, errors clustered on orders | Assignment-level shocks not grouped | Standard error still too small | Cluster at the level the treatment varies |
| HAC on a pooled panel | One long series sorted by store then week | Kernel crosses store boundaries; no cross-sectional term | Understated standard error | Cluster by store; two-way if weeks share shocks |
| HAC lag too short | Result flips when `L` is doubled | Truncation omits autocovariance | Over-rejection; false positive | Sensitivity across `L`, `2L`; seasonal minimum |
| HAC lag too long | Interval very wide; unstable across samples | Few effective terms in the variance | Low power; real effect missed | Choose from `T` and seasonality, not from fit |
| Using robust errors to fix a wrong mean | Coefficient moves when a control is added | Variance correction does not change `b` | Precise estimate of the wrong quantity | Specification first, variance second |
| Durbin–Watson with a lagged dependent variable | DW printed next to an autoregressive model | Null distribution invalid | Wrong test, wrong conclusion | Breusch–Godfrey or Durbin's h |
| DW inconclusive read as "no autocorrelation" | Statistic between the bounds | Bounds test has an indeterminate region | Dependence missed | Breusch–Godfrey at a stated order |
| Assuming autocorrelation biases the coefficient | "We must fix the bias" | OLS is unbiased under strict exogeneity | Effort spent on the wrong repair | Check exogeneity; use HAC for inference |
| Too few clusters | `G` below 20; unstable intervals across specs | CRVE downward biased | False significance | `G - 1` df; CR2; wild bootstrap |
| Cluster count inflated by singletons | Many clusters with one row | Singletons contribute little and distort | Nominal `G` overstates information | Report cluster-size distribution |
| Ignoring common weekly shocks | Every store moves in promotion weeks | Cross-sectional dependence | Store clustering still too optimistic | Two-way cluster; Driscoll–Kraay |
| Logging to "fix" heteroskedasticity | Variance still grows with fitted value | Log changes the model, not only the variance | Coefficient now on a different scale | Check the residual plot after logging |
| Unnamed robust errors | "Robust SE" in a table note | Ambiguous estimator | Irreproducible | Name estimator, cluster, lag, `G` |
| Overlapping forecast horizons ignored | Consecutive errors share inputs | Induced moving-average errors | Intervals too narrow | HAC with `L >= horizon - 1`; block bootstrap |
| Panel fixed effects with lagged outcome | Short `T`, dynamic model | Nickell bias | Coefficient biased toward zero | IV or GMM; do not rely on robust SE |

Two deserve expansion.

**HAC on a pooled panel** is the most common technical error in supply chain analytics, because
the software makes it easy. A panel sorted by store and week is treated as one long series; the
kernel sums residuals from different stores at the boundary, and the estimate depends on the sort
order. The result is neither a valid cluster variance nor a valid HAC variance. If the data have
groups, cluster.

**Unnamed robust errors** is the most common reporting error. "Robust standard errors" could mean
HC0 through HC3, HAC with any bandwidth, cluster with any grouping, or a bootstrap. The
differences are often larger than the effect being reported. Name the estimator, the cluster, the
number of clusters, and the lag.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Strict exogeneity of the regressors | OLS unbiasedness; robust errors do not repair endogeneity | Omitted variables, simultaneity, measurement error | Coefficient stability across controls; design |
| Correct mean specification | The residual is the object being modelled | Missing trend, seasonality, holiday, lag | Residual ACF; added-variable plots |
| The cluster identifies the dependence | CRVE groups correlated rows | Wrong level, unstable identifier, network dependence | Compare one-way and two-way; ICC by group |
| Clusters are independent of each other | The sandwich sums cluster scores | Spillovers across stores or regions | Spatial correlation; placebo clusters |
| Enough clusters | Asymptotic CRVE accuracy | `G` below about 30, unbalanced sizes, few treated | `G`, size distribution, wild bootstrap |
| Bandwidth captures the dependence | HAC consistency | Too short truncation; seasonality beyond `L` | Sensitivity across `L` and `2L` |
| Errors are stationary | HAC sums a stable autocovariance | Level shifts, regime change, trending variance | Rolling estimates; structural-break tests |
| No cross-sectional dependence | One-way cluster assumes it away | Common shocks, shared promotion waves | Two-way cluster; Driscoll–Kraay |
| Cluster assignments are correctly measured | Grouping must match the data-generating process | Renumbered stores, merged lanes | Audit the identifier over time |
| The sample of clusters is representative | Inference generalises to the cluster population | Convenience sample of stores or lanes | Compare cluster characteristics |
| The treatment is measured at the right level | Design effect depends on it | Store-level treatment, order-level rows | Reconcile assignment and observation levels |
| The nominal level is the actual level | Reporting depends on coverage | Few clusters, model selection, multiple testing | Simulation; bootstrap coverage check |

Two limits of the whole approach are worth stating plainly.

**Robust errors are not identification.** Every estimator here assumes the regressors are
exogenous or that the design supplies the counterfactual. If the display went to stores that were
already growing faster, no standard error repairs the comparison; the interval is honest about a
biased estimate, which is not the same as being useful.

**The design effect is an approximation.** `1 + (m - 1) rho` assumes equal cluster sizes and a
cluster-constant regressor. Real panels have unequal `m`, and most regressors vary within cluster,
which reduces the inflation. Use the formula to build intuition and to sanity-check the computed
CRVE, not to replace it.

## 12. Exercises

### Exercise 1 — Design effect by hand (calculation, 30 minutes)

A retailer has 200 stores observed for 24 weeks. The average within-store residual correlation is
0.30, and the naive standard error of a store-level treatment effect is 8.0 units per store-week.

**Deliverable:** the design effect, the effective sample size, the effective number of stores, the
standard error multiplier, and the approximate cluster-robust standard error, each with units.
**Acceptance criteria:** every step shown; `deff` and `n_eff` labelled; the sentence you would say
to a planner written in plain language.
**Time:** 30 minutes.

### Exercise 2 — Effective sample size for a daily series (calculation, 30 minutes)

A distribution centre has 520 daily observations of outbound units. The residual autocorrelation
is AR(1) with `rho = 0.65`. The naive t-statistic on a trend is 4.80.

**Deliverable:** the variance inflation factor, the standard error multiplier, the effective sample
size, and the corrected t-statistic.
**Acceptance criteria:** the formula stated; both the inflation factor and its square root shown;
the corrected t compared with the appropriate critical value.
**Time:** 30 minutes.

### Exercise 3 — Four standard errors (analysis, 40 minutes)

Use the table in section 6.4. The coefficient is 42.0, the naive SE is 11.2, HC3 is 12.1, HAC(4)
is 21.6, store-clustered is 33.4, and two-way clustered is 37.8. The store-clustered degrees of
freedom are 119 and the two-way degrees of freedom are 118.

**Deliverable:** a table with t-statistics and 95% intervals for all five estimators, plus a
three-sentence recommendation naming the estimator you would report and why.
**Acceptance criteria:** correct critical values; the recommendation names the cluster and the
number of clusters; the difference between the naive and clustered widths is quantified.
**Time:** 40 minutes.

### Exercise 4 — Choose the cluster level (design, 35 minutes)

For each scenario, name the unit of observation, the unit of independence, and the cluster you
would use.

1. Orders from 4,000 stores, with a display assigned to 60 stores.
2. Daily shipments on 300 lanes, with a carrier change applied to 12 lanes.
3. Weekly item-location sales for 500 stores and 2,000 items.
4. A promotion applied to all stores in 8 of 40 regions.
5. A supplier scorecard with monthly observations for 150 suppliers over 3 years.

**Deliverable:** a five-row table with the three columns and one sentence of justification each.
**Acceptance criteria:** at least one scenario where the observation unit is finer than the
assignment unit; at least one where two-way clustering is warranted.
**Time:** 35 minutes.

### Exercise 5 — Lag sensitivity (calculation, 45 minutes)

Compute the rule-of-thumb HAC lag `floor(4 (T/100)^(2/9))` for `T = 104`, 260, 728, and 1,560. Then
state the minimum lag you would use for (a) daily data with a weekly cycle and (b) weekly data
with an annual cycle.

**Deliverable:** a four-row table of `T`, the computed lag, the seasonal minimum, and the lag you
would use; plus a paragraph on why the rule does not know about seasonality.
**Acceptance criteria:** arithmetic correct to the integer; the seasonal argument stated in terms
of the cycle length, not in terms of fit.
**Time:** 45 minutes.

### Exercise 6 — Durbin–Watson versus Breusch–Godfrey (diagnosis, 40 minutes)

A weekly regression has `n = 104`, five regressors including the intercept, and no lagged
dependent variable. Durbin–Watson is 1.42. A Breusch–Godfrey test at order 4 gives a statistic of
18.6 with a p-value of 0.001.

**Deliverable:** the DW decision using approximate 5% bounds, the BG decision, a reconciliation of
the two, and a statement of what you would do if DW were 1.65.
**Acceptance criteria:** the inconclusive region named; the null and order of each test stated;
the recommendation distinguishes first-order from higher-order dependence.
**Time:** 40 minutes.

### Exercise 7 — Simulate over-rejection (computation, 60 minutes)

Simulate a regression with `n = 100`, one regressor, and AR(1) errors with `rho = 0.8`. Run 1,000
replications. For each, compute the naive standard error and a HAC standard error with lag 4, and
record whether the 5% test rejects.

**Deliverable:** the rejection rate for each estimator, a histogram of the two t-statistics, and
two sentences of interpretation.
**Acceptance criteria:** the simulation seeded and the seed reported; the nominal level marked on
the histogram; the result compared with the qualitative prediction in section 9.
**Time:** 60 minutes.

### Exercise 8 — Few clusters and the wild bootstrap (analysis, 60 minutes)

Take a panel with 12 clusters and a treatment assigned at cluster level. Estimate the effect with
the asymptotic CRVE and with a wild cluster bootstrap using Rademacher weights and 999
replications.

**Deliverable:** the two intervals, the bootstrap p-value, and a paragraph on why they differ.
**Acceptance criteria:** the number of treated clusters stated; the bootstrap procedure described
in enough detail to reproduce; the recommendation for a 12-cluster design.
**Time:** 60 minutes.

### Exercise 9 — Reporting paragraph (writing, 30 minutes)

Write a 150-word methods paragraph for the display test in section 6, stating the model, the
estimand, the error type, the cluster, the number of clusters, the lag or bandwidth, and the
sensitivity checks.

**Deliverable:** the paragraph, plus a table note suitable for a slide.
**Acceptance criteria:** a reader could reproduce the interval; no unexplained acronym; the naive
result is mentioned as a comparison, not as the headline.
**Time:** 30 minutes.

### Exercise 10 — Audit a regression (critique, 45 minutes)

Find a published or internal regression on supply chain data. Identify the unit of observation,
the claimed unit of independence, the error type, and whether the standard error is credible.

**Deliverable:** a one-page audit with a verdict and one recommended re-estimation.
**Acceptance criteria:** the cluster level is named; at least one red flag from section 10 is
identified with evidence; the recommendation is specific enough to run.
**Time:** 45 minutes.

## 13. Self-check answers

**Exercise 1 arithmetic.**

```text
n        = 200 x 24                        = 4,800 store-weeks
m        = 24 weeks per store
rho      = 0.30

deff     = 1 + (m - 1) rho
         = 1 + 23 x 0.30
         = 1 + 6.90                       = 7.90

n_eff    = 4,800 / 7.90                   = 607.6 independent store-weeks

stores_eff = 200 / 7.90                   = 25.3 independent store-equivalents

SE multiplier = sqrt(7.90)                = 2.811

cluster SE = 8.0 x 2.811                  = 22.5 units per store-week
```

The planner's sentence: "We have 4,800 store-weeks, but for a question that varies at store level
they carry about 600 weeks of information, or about 25 stores. The honest standard error is 22.5,
not 8.0."

The common wrong answer is `8.0 x 7.90 = 63.2`, which multiplies by the variance ratio instead of
its square root. The standard error scales with the square root of the design effect. Mark that
wrong regardless of the rest of the arithmetic.

**Exercise 2 arithmetic.**

```text
T          = 520
rho        = 0.65

variance inflation = (1 + rho) / (1 - rho)
                   = 1.65 / 0.35           = 4.714

SE multiplier      = sqrt(4.714)           = 2.171

T_eff      = 520 x (1 - 0.65) / (1 + 0.65)
           = 520 x 0.35 / 1.65
           = 520 x 0.21212                 = 110.3 independent days

corrected t = 4.80 / 2.171                 = 2.21
```

At 5% with a large sample the critical value is 1.96, so the corrected trend is still significant
but marginal. If the analysis had used a 1% threshold, the conclusion would reverse. The honest
report is the interval, not the verdict.

**Exercise 3 arithmetic.**

| Estimator | SE | t | Critical value | 95% CI |
|---|---:|---:|---:|---|
| Naive | 11.2 | 3.75 | 1.96 | [20.0, 64.0] |
| HC3 | 12.1 | 3.47 | 1.96 | [18.3, 65.7] |
| HAC(4) | 21.6 | 1.94 | 1.96 | [-0.3, 84.3] |
| Store cluster | 33.4 | 1.26 | 1.98 | [-24.1, 108.1] |
| Two-way cluster | 37.8 | 1.11 | 1.98 | [-32.8, 116.8] |

Widths: naive 43.9, HC3 47.4, HAC 84.6, store-clustered 132.3, two-way 149.6. The store-clustered
interval is 3.0 times the naive width, matching the standard-error ratio of 33.4/11.2 = 2.98.

The recommendation: report the store-clustered interval, because the treatment is assigned at
store level and the design effect is 9.75. Report the two-way interval as a sensitivity check,
because promotion weeks are shared shocks. Do not report the naive interval as the headline.

**Exercise 4 model answers.**

| Scenario | Observation unit | Independence unit | Cluster | Reason |
|---|---|---|---|---|
| Store display | Order or store-week | Store | Store | Treatment assigned to stores |
| Carrier change | Lane-day | Lane | Lane | Treatment assigned to lanes |
| Item-location sales | Item-store-week | Store, or item within store | Store (two-way with item if items are shared) | Items are common across stores |
| Regional promotion | Store-week | Region | Region | Treatment assigned to regions |
| Supplier scorecard | Supplier-month | Supplier | Supplier | Repeated observations per supplier |

The trap is scenario 3. The observation is item-store-week and the temptation is to cluster on
item-store, which is defensible. But if the same item appears in many stores and a national
promotion affects it everywhere, the item dimension carries a common shock and two-way clustering
by store and item, or by store and week, is warranted.

**Exercise 5 arithmetic.**

| `T` | `4 (T/100)^(2/9)` | Rule lag | Daily, weekly cycle | Weekly, annual cycle |
|---:|---:|---:|---:|---:|
| 104 | 4.04 | 4 | 7 | 52 |
| 260 | 4.95 | 4 | 7 | 52 |
| 728 | 6.22 | 6 | 7 | 52 |
| 1,560 | 7.36 | 7 | 7 | 52 |

The rule gives a lag that grows very slowly with `T`, because the optimal bandwidth for a HAC
estimator grows more slowly than the sample. It knows nothing about the seasonal cycle, which is
a property of the data rather than the sample size. For daily data the weekly cycle sets a floor
of 7; for weekly data with annual seasonality the floor is 52, which is often too large to be
useful and is a sign that the mean equation should absorb the seasonality rather than the
variance estimator.

**Exercise 6 model answer.**

For `n = 104` and five regressors, the approximate 5% Durbin–Watson bounds are `d_L = 1.57` and
`d_U = 1.78`. The statistic 1.42 is below `d_L`, so the test rejects no autocorrelation in favour
of positive first-order autocorrelation.

The Breusch–Godfrey statistic of 18.6 on 4 degrees of freedom has a p-value of 0.001, so it
rejects no serial correlation up to order 4. The two tests agree on the presence of dependence;
BG says it extends beyond the first order.

If DW were 1.65, it would fall in the inconclusive region between `d_L` and `d_U`. The correct
response is not to call it "no autocorrelation" but to run Breusch–Godfrey, which does not have
an inconclusive region and which can test higher orders. This is exactly the case that motivated
the LM tests.

**Exercise 7 expected result.**

At `rho = 0.8` and `n = 100`, the naive 5% test typically rejects in 30% to 50% of replications,
depending on the regressor's persistence. HAC with lag 4 reduces this substantially but usually
not to 5%; residual over-rejection of 8% to 15% is common. Report the student's own simulation.
The qualitative point is that the naive test is not approximately valid and HAC is an improvement,
not a cure.

**Exercise 8 model answer.**

With 12 clusters the asymptotic CRVE is downward biased, and the interval is too narrow. The wild
cluster bootstrap, which resamples residual signs within clusters, produces a wider interval and
a larger p-value. If only 2 of the 12 clusters are treated, the bootstrap's performance is itself
fragile, and the honest conclusion may be that the design cannot support a precise claim. The
recommendation is to report the bootstrap interval, state the number of treated clusters, and
avoid a definitive significance verdict.

**Exercise 9 model paragraph.**

"The display effect is estimated by OLS on 3,120 store-weeks with store and week fixed effects.
The coefficient is 42.0 units per store-week. Standard errors are clustered by store, the level
at which the display was assigned; there are 120 clusters. The 95% confidence interval is −24.1
to +108.1. A HAC estimator with four lags gives 21.6 and a two-way cluster by store and week gives
37.8; both are reported in the appendix. The naive i.i.d. standard error of 11.2 is shown for
comparison only. The estimated design effect is 9.75, so the 3,120 rows are worth about 320
independent observations."

A strong paragraph names the estimator and the cluster; a weak one says "robust standard errors
were used."

**Exercise 10 rubric.**

| Criterion | Weight | What a strong audit does |
|---|---:|---|
| Unit of observation stated | 15% | Distinguishes row from independent unit |
| Error type identified | 20% | Names HC, HAC, cluster, or unknown |
| Cluster level assessed | 20% | Compares to assignment level |
| Few-cluster risk assessed | 15% | Reports `G`, not `n` |
| Red flag evidenced | 15% | Quotes the table or methods text |
| Re-estimation specified | 15% | Names estimator, cluster, lag, and software |

**General design rubric for this topic.**

| Criterion | Weight | Passing standard |
|---|---:|---|
| Estimand stated with units | 15% | A number with a unit and a population |
| Dependence diagnosed, not assumed | 20% | Residual ACF and cluster ICC reported |
| Estimator matched to dependence | 20% | Cluster level equals assignment level |
| Lag or bandwidth justified | 15% | Rule plus seasonal floor plus sensitivity |
| Few-cluster handling | 10% | `G - 1` df, correction, or bootstrap |
| Reporting completeness | 15% | Estimator, cluster, `G`, lag all named |
| Coefficient-versus-SE distinction | 5% | States which problem the repair addresses |

## 14. Interview questions

**1. "Why can autocorrelation leave OLS unbiased but make the t-statistic wrong?"**

A strong answer separates the point estimate from its variance: OLS is unbiased under strict
exogeneity regardless of the error covariance, but the textbook variance formula assumes zero
off-diagonal covariance, so it omits terms that are positive under positive autocorrelation. A
weak answer says autocorrelation "biases the regression," which conflates the two failures.

**2. "What does the standard error formula assume, and what does supply chain data do to it?"**

A strong answer names independence, constant variance, and a correct mean equation, then gives a
concrete violation such as store-weeks sharing a store or variance growing with volume. A weak
answer says "the data is messy," which is not an assumption.

**3. "When would you use HAC and when would you cluster?"**

A strong answer says HAC for a single time series with serial correlation, clustering for grouped
data with arbitrary within-group dependence, and notes that HAC on a pooled panel is a category
error because it ignores cross-sectional dependence and crosses group boundaries. A weak answer
treats them as interchangeable "robust" options.

**4. "How many clusters are enough?"**

A strong answer says there is no sharp threshold, that the literature ranges from about 20 to 50
depending on balance and the number of treated clusters, and that below about 20 you should use a
small-sample correction or a wild bootstrap. A weak answer quotes 42 as a rule.

**5. "Explain the design effect to a planner."**

A strong answer gives `1 + (m-1) rho`, computes an example, and translates it into effective
stores rather than rows: "120 stores carry about 12 stores of information." A weak answer gives
the formula and stops, or confuses the variance ratio with the standard-error ratio.

**6. "What is the difference between Durbin–Watson and Breusch–Godfrey?"**

A strong answer says DW tests first-order autocorrelation only, has an inconclusive region, and is
invalid with a lagged dependent variable; BG tests a stated higher order, has no inconclusive
region, and remains valid with lagged outcomes. A weak answer says both test autocorrelation and
picks whichever the software prints.

**7. "Does heteroskedasticity bias OLS?"**

A strong answer says no, under exogeneity the coefficient is unbiased and consistent; the
efficiency loss and the invalid standard errors are the problem, and HC errors fix the second. A
weak answer says yes, which is a common and consequential confusion.

**8. "When do you need two-way clustering?"**

A strong answer says when both dimensions carry shocks, such as stores and weeks in a retail
panel where national promotions and weather affect all stores in a week, and notes that two-way
is not the sum of the two one-way estimates. A weak answer says "always cluster by firm and
year" without a mechanism.

**9. "A vendor reports 'robust standard errors.' What do you ask?"**

A strong answer asks which estimator, which cluster, how many clusters, and what lag or
bandwidth, then asks whether the evaluation sample respected the panel structure. A weak answer
accepts the phrase as sufficient.

**10. "How do you choose the HAC lag?"**

A strong answer starts from `floor(4 (T/100)^(2/9))`, raises it to at least the seasonal period,
checks sensitivity at `L` and `2L`, and refuses to choose it by maximising significance. A weak
answer says "the software default."

**11. "What does effective sample size mean, and why does it matter for a dashboard?"**

A strong answer says it is the number of independent observations with the same information
content, that it can be an order of magnitude below the row count, and that a dashboard showing
row counts as sample size overstates precision. A weak answer equates it with the number of rows
after cleaning.

**12. "Give an example of a wrong coefficient and an example of a wrong standard error."**

A strong answer gives a wrong coefficient from an omitted store-specific trend or from selecting
stores that were already growing, and a wrong standard error from ignoring store clustering, and
notes that the second is invisible in the point estimate while the first is not. A weak answer
gives two variants of the same failure.

## 15. Key terms

- **Standard error** — estimated standard deviation of an estimator across repeated samples.
- **Spherical errors** — constant variance and zero covariance across observations.
- **Heteroskedasticity** — error variance that varies with regressors or fitted values.
- **Serial correlation** — correlation between errors at different times.
- **Autocorrelation function (ACF)** — correlation of a series with its lagged self, by lag.
- **AR(1)** — first-order autoregression, `e_t = rho e_{t-1} + u_t`.
- **Persistence** — how slowly a shock decays, often summarised by a half-life.
- **Effective sample size** — independent observations with the same information content.
- **Design effect** — variance ratio from clustering, `1 + (m-1) rho` for equal clusters.
- **Intra-cluster correlation (ICC)** — average correlation between two observations in a cluster.
- **Cluster** — a group within which observations may be correlated arbitrarily.
- **Cluster-robust variance estimator (CRVE)** — sandwich estimator using cluster score sums.
- **Sandwich estimator** — `(X'X)^-1 Omega (X'X)^-1`, with `Omega` estimated from residuals.
- **HC0, HC1, HC3** — heteroskedasticity-consistent variants with different small-sample weights.
- **HAC** — heteroskedasticity- and autocorrelation-consistent covariance estimator.
- **Newey–West estimator** — HAC with Bartlett kernel weights `1 - j/(m+1)`.
- **Bartlett kernel** — triangular lag weights that guarantee a positive semi-definite variance.
- **Bandwidth** — number of lags over which autocovariances are summed in a HAC estimator.
- **Two-way clustering** — clustering on two dimensions, such as store and week.
- **Driscoll–Kraay** — HAC applied to period-by-period cross-sectional averages of the score.
- **Few-cluster problem** — downward bias in CRVE when the number of clusters is small.
- **Wild cluster bootstrap** — bootstrap that resamples residual signs within clusters.
- **Durbin–Watson statistic** — first-order autocorrelation test with an inconclusive region.
- **Breusch–Godfrey test** — Lagrange multiplier test for serial correlation to a stated order.
- **White test** — heteroskedasticity test based on squared residuals and their cross-products.
- **Wooldridge panel test** — test of no first-order serial correlation in a linear panel.
- **Model-based standard error** — standard error from an assumed error distribution.
- **Design-based standard error** — standard error from the sampling or assignment design.
- **Coverage** — fraction of intervals containing the true parameter across repeated samples.
- **Over-rejection** — rejecting a true null more often than the nominal rate.

## 16. Sources

**Peer-reviewed and methodological (Grade A)**

- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct
  Test for Heteroskedasticity." *Econometrica*, 48(4), 817–838. DOI: 10.2307/1912934.
- Newey, W. K., and West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708. DOI: 10.2307/1913610.
- Bertrand, M., Duflo, E., and Mullainathan, S. (2004). "How Much Should We Trust
  Differences-in-Differences Estimates?" *Quarterly Journal of Economics*, 119(1), 249–275.
  DOI: 10.1162/003355304772839588.
- Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). "Bootstrap-Based Improvements for
  Inference with Clustered Errors." *Review of Economics and Statistics*, 90(3), 414–427.
  DOI: 10.1162/rest.90.3.414.
- Cameron, A. C., and Miller, D. L. (2015). "A Practitioner's Guide to Cluster-Robust Inference."
  *Journal of Human Resources*, 50(2).
- MacKinnon, J. G., and Webb, M. D. (2017). "Wild Bootstrap Inference for Wildly Different
  Cluster Sizes." *Journal of Applied Econometrics*. DOI: 10.1002/jae.2508.
- Driscoll, J. C., and Kraay, A. C. (1998). "Consistent Covariance Matrix Estimation with
  Spatially Dependent Panel Data." *REStat*, 80(4), 549–560. DOI: 10.1162/003465398557825.
- Hoechle, D. (2007). "Robust Standard Errors for Panel Regressions with Cross-Sectional
  Dependence." *Stata Journal*, 7(3), 281–312. DOI: 10.1177/1536867X0700700301.
- Godfrey, L. G. (1978). "Testing for Higher Order Serial Correlation in Regression Equations
  when the Regressors Include Lagged Dependent Variables." *Econometrica*, 46(6).
- Durbin, J., and Watson, G. S. (1950, 1951). "Testing for Serial Correlation in Least Squares
  Regression," Parts I and II. *Biometrika*, 37 and 38.
- Lazarus, E., Lewis, D. J., Stock, J. H., and Watson, M. W. (2018). "HAR Inference:
  Recommendations for Practice." *J. Bus. Econ. Stat.*, 36(4), 541–559. DOI: 10.1080/07350015.2018.1506926.
- Drukker, D. M. (2003). "Testing for Serial Correlation in Linear Panel-data Models." *Stata
  Journal*, 3(2), 168–177. DOI: 10.1177/1536867x0300300206.
- Moulton, B. R. (1990). "An Illustration of a Pitfall in Estimating the Effects of Aggregate
  Variables on Micro Units." *Review of Economics and Statistics*.

**Books and reference texts (Grade A/B)**

- Angrist, J. D., and Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University
  Press, Chapter 8. Source of the "42 clusters" mnemonic, offered as a rule of thumb, not a threshold.
- Wooldridge, J. M. *Introductory Econometrics*, Chapter 12, for the HAC lag rule and serial correlation.
- Kish, L. (1965). *Survey Sampling*. Wiley. Origin of the design effect in complex samples.

**Standards and official (Grade B)**

- ISO 8601, date and time representation, including week dates.
- IANA time zone database, for local offsets and daylight-saving rules.

**Software documentation (Grade B/C)**

- Stata `xtserial` and `xtscc` documentation; Stata FAQ on panel-level heteroskedasticity and
  autocorrelation.
- `statsmodels` documentation for `cov_type="HC0"`, `"HC1"`, `"HC3"`, `"HAC"`, and `"cluster"`.
- R `sandwich::NeweyWest` and `plm::vcovSCC`.

**Industry and analyst (Grade D)**

- Public demand-planning and revenue-management vendor material, 2026. Claims about accuracy
  improvements generally do not state the error model, the validation split, or the clustering of
  the evaluation sample. Treated as existence claims, not evidence of calibrated inference.

**Note.** Method citations are to the published papers returned by the searches. Simulation
rejection rates are as the cited papers state them, not independently replicated here.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_regression-time-series.md`](04_regression-time-series.md)
- Survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Calendar and bucket conventions: [`../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Next topic: `06_dags.md` (planned)
- Later topic: `08_difference-in-differences.md` (planned)
- Later topic: `09_elasticity-endogeneity.md` (planned)
