### Step 4 — Choose the decay on a development fold, not on the whole sample

The decay is a parameter, and the only defensible way to pick it is to score candidates on data that
the final claim will not use. The development fold here is the first 899 scored origins.

| Decay | Half-life (sessions) | ESS (sessions) | Dev-fold QLIKE | Test-fold QLIKE |
|---|---|---|---|---|
| 0.85 | 4.27 | 12.33 | 0.05187 | 0.07561 |
| 0.90 | 6.58 | 19.00 | 0.03262 | 0.03888 |
| 0.92 | 8.31 | 24.00 | 0.02856 | 0.02789 |
| 0.94 | 11.20 | 32.33 | 0.02831 | 0.02005 |
| 0.96 | 16.98 | 49.00 | 0.03471 | 0.01729 |
| 0.97 | 22.76 | 65.67 | 0.04205 | 0.01908 |
| 0.98 | 34.31 | 99.00 | 0.05373 | 0.02400 |
| 0.99 | 68.97 | 199.00 | 0.07243 | 0.03267 |

Every number in that table is `ILLUSTRATIVE` and comes from the single seeded series. The
development column is the only one a practitioner is allowed to read when choosing, and it selects
0.94, whose development loss of 0.02831 is the minimum of the row. The test column is shown to make
a different point: 0.96 would have scored better on the test fold, 0.01729 against 0.02005, and a
team that tuned on the test fold would have shipped 0.96 and reported the better number. That is
precisely the selection bias the fold structure exists to prevent, and the size of it here, about
14% of the test-fold loss, is a useful reminder that "we tuned the parameter and it got better"
is not evidence of anything.

The shape of the development column is convex and informative. Loss falls steeply from 0.85 to 0.94
and then rises gently, so the optimum is not knife-edge on the fast side but is on the slow side:
moving from 0.94 to 0.99 more than doubles the loss, while moving from 0.94 to 0.90 only raises it by
15%. That asymmetry is a practical guide to how much the choice matters, and it says that if the
selected value must be rounded for documentation, rounding down is the safer error.

### Step 5 — Choose the window length too, and notice it matters more

The baseline's window is also a parameter, and treating it as a fixed feature of the environment is
how the comparison gets rigged. The same development fold, applied to the window length:

| Window | Dev-fold QLIKE | Test-fold QLIKE | Mean forecast over mean true variance |
|---|---|---|---|
| 10 | 0.08964 | 0.15437 | 1.0488 |
| 20 | 0.04547 | 0.05580 | 1.0490 |
| 30 | 0.04925 | 0.04604 | 1.0476 |
| 60 | 0.06901 | 0.04864 | 1.0431 |
| 120 | 0.09333 | 0.05049 | 1.0317 |

The development fold prefers 20, which is convenient because 20 is the window already in production;
a proposal that had to argue for changing the baseline's window as well as its weighting would face
a much harder governance conversation. But the test column shows a 30-session window beating the
20-session window by 17.5%, and the final comparison in Section 9 therefore uses the 30-session
window as the honest comparator. That decision was taken after seeing the development fold's ranking
of windows and before scoring the candidate against it, and it is recorded in the ledger as such.

The last column is the diagnostic that prevents a false conclusion. All five windows have a mean
forecast within 5% of the mean true variance, so no window is systematically biased in level. Their
losses differ anyway, by a factor of two between 20 and 10, which means the loss is being driven by
*when* the estimate is high or low rather than by *how high* it is on average. A window that tracks
the level but lags the timing gets a good bias and a bad loss; that distinction is invisible to a
bias check and is one of the reasons mean-error diagnostics are insufficient for variance forecasts.

### Step 6 — Use a loss that is proper for variance

Mean squared error on squared returns is a popular choice and it is the wrong one for this target,
for two reasons that can both be shown with two numbers.

```text
FORMULA [D6]: Q(f, y) = y/f - ln(y/f) - 1
WHERE: y is the realised squared return (or any proxy) in squared decimal returns;
       f is the forecast variance, strictly positive, same units;
       Q is dimensionless and non-negative, with Q = 0 exactly when y = f.
CONSTRAINT: f must be strictly positive; y must be strictly positive for ln(y/f)
            to exist, so a zero return has to be handled by a documented floor.
NUMERICAL EXAMPLE: y = 0.0004, f = 0.0001 gives y/f = 4, ln(4) = 1.386294,
            Q = 4 - 1.386294 - 1 = 1.6137.
            The mirrored case y = 0.0001, f = 0.0004 gives y/f = 0.25,
            ln(0.25) = -1.386294, Q = 0.25 + 1.386294 - 1 = 0.6363.
INTERPRETATION: the same ratio error costs 2.54 times more when the forecast is too
            low than when it is too high, which matches the desk's asymmetry.
```

The first reason Q is right is that asymmetry. A fourfold understatement costs 1.6137 and a fourfold
overstatement costs 0.6363, and their ratio is 2.5361. For a risk application that asymmetry is not
a mathematical convenience but the actual loss structure: understating volatility understates the
limit utilisation the desk reports and understates the capital it holds.

The second reason is that squared error on a noisy proxy does not rank variance forecasts correctly,
and the failure can be large. Consider two sessions, with true-or-proxy squared returns 0.0001 and
0.0009, and two competing forecasts. Forecast A prints 0.0001 for both sessions. Forecast B prints
0.00015 and 0.0017. Forecast A's mean squared error is (0 + 6.4e-7)/2 = 3.2e-7; forecast B's is
(2.5e-9 + 6.4e-7)/2 = 3.2125e-7, slightly worse. By squared error, A wins. But A's quasi-likelihood
is (0 + 5.80278)/2 = 2.9014 while B's is (0.072132 + 0.165401)/2 = 0.1188, so by Q, B wins by a
factor of 24. The reason is mechanical: squared error is dominated by the single large observation,
and A's zero error on the small observation counts for nothing, whereas Q is a relative measure and
B's roughly proportionate forecasts of both sessions are rewarded.

That example is not a curiosity; it is the reason the case reports Q as the primary metric and
reports squared error only as a secondary diagnostic. A team that selects on squared error will
systematically prefer forecasts that are too smooth, because smooth forecasts avoid large absolute
misses on the rare extreme observation at the cost of being wrong most of the time.

### Step 7 — Score the candidate against the baseline on the frozen protocol

With the decay frozen at 0.94 and the comparator frozen at a 30-session window, the full sample and
the test fold give the following, quoted against both proxies because no single proxy is decisive.

| Method | QLIKE vs latent variance | QLIKE vs squared return | MSE vs squared return | Mean ratio to latent |
|---|---|---|---|---|
| Candidate, decay 0.94 | 0.02500 | 1.34898 | 4.1720e-08 | 1.0466 |
| Baseline, 20-session | 0.04961 | 1.35603 | 4.2607e-08 | 1.0490 |
| Full-sample constant | 0.08549 | 1.42137 | 4.4795e-08 | 0.9554 |

The three methods are ordered identically on all four columns, which is reassuring but not the
interesting part. The interesting part is the *magnitude*, which depends entirely on which proxy is
used. Against the latent variance, which is available only because the series is synthetic, the
candidate improves the loss by 49.59%. Against the observable squared return, the improvement is
0.52%. Against squared error on the same observable, it is 2.08%.

That spread is not a bug in the arithmetic; it is the statistical content of the case. The latent
comparison isolates the quality of the variance forecast because it removes proxy noise. The
observable comparison is what a real desk can compute and it is dominated by the noise in a single
squared return, whose sampling variance is so large that a genuine 50% improvement in forecast
quality registers as half a percent of measured loss. A practitioner who reports only the second
number will conclude that the method is worthless; one who reports only the first will be accused,
correctly, of evaluating against something nobody can observe.

The formal test makes the same point more sharply. A Diebold-Mariano statistic on the loss
differential against the latent variance gives a mean difference of -0.02460 with a standard
deviation of 0.0672 over 1,499 origins, hence a statistic of -14.176. Against the observable proxy
the mean difference is -0.00705 with a standard deviation of 0.3161, giving a statistic of -0.863.
One of those is overwhelming evidence and the other is indistinguishable from zero, on the same
data, from the same forecasts, differing only in the proxy.

The Mincer-Zarnowitz diagnostics add a dimension the loss does not. Regressing the latent variance on
the forecast through the origin gives a slope of 0.8723 for the candidate with an R-squared of
0.7209, and a slope of 0.8185 for the baseline with an R-squared of 0.5665. A slope below one means
the forecast over-reacts relative to the truth: high forecasts are too high and low forecasts are
too low, a shrinkage failure. The candidate's slope is closer to one and its R-squared is 27% higher,
so it is both better calibrated and more informative, which is a stronger statement than the loss
comparison alone. The full-sample constant, by contrast, has a slope of 1.0467 and an R-squared of
essentially zero, because a constant has no information at all and the regression has nothing to fit.

### Step 8 — Convert the forecast into the currency number the desk actually sees

A variance forecast is not a decision input until it becomes a limit comparison. The pack uses a
Gaussian one-day 99% quantile on the constant notional.

```text
FORMULA [D7]: VaR = z_0.99 * sigma * E
WHERE: z_0.99 = 2.32635 is the standard normal 99% quantile, dimensionless;
       sigma is the forecast daily volatility, decimal, strictly positive;
       E is the exposure notional, 250 million currency units;
       VaR is a positive loss magnitude in currency units.
CONSTRAINT: the formula assumes zero conditional mean, Gaussian innovations,
            and a one-session holding period with no intra-session path.
NUMERICAL EXAMPLE: sigma = 0.016, E = 250.
            z * sigma = 2.32635 * 0.016 = 0.0372216.
            VaR = 0.0372216 * 250 = 9.3054 million.
INTERPRETATION: at 1.6% daily volatility the pack prints 9.3054 million against
            a 10 million limit, leaving 0.6946 million, or 6.95%, of headroom.
```

The headroom arithmetic is where a forecast error becomes an operational fact. Holding the limit
fixed, the volatility at which the printed VaR exactly equals 10 million solves
sigma-star = 10 / (2.32635 * 250) = 0.017194, that is 1.7194% daily. So the pack can tolerate a
forecast volatility up to 7.46% higher than its current level before the printed number breaches the
limit, and any forecast that is more than 7.46% too low will understate the reported figure. In the
simulation the candidate's mean ratio to the latent variance is 1.0466, which corresponds to a mean
volatility ratio of the square root of 1.0466, that is 1.0230, so on average it is 2.3% too high
rather than too low. That is the benign direction for a limit, and it is not the direction that the
tail statistics care about.

Switching the tail measure from VaR to expected shortfall changes the number and the headroom. For a
Gaussian, expected shortfall at 99% is the density at the quantile divided by the tail probability,
which is 0.026652 / 0.01 = 2.66521, so ES is 1.1457 times VaR. At 1.6% volatility that is 10.6609
million, which is already above the 10 million limit. The volatility at which ES equals the limit is
10 / (2.66521 * 250) = 0.015008, or 1.5008% daily. A desk that measures risk with ES therefore has
6.2% less volatility headroom than one that measures with VaR at the same nominal confidence, and
that difference is a property of the tail model, not of the forecast.

The multi-session conversion is the last place the arithmetic can mislead. Scaling a one-day VaR by
the square root of ten gives 9.3054 * 3.1623 = 29.4262 million. That scaling is only exact when
returns are serially uncorrelated and variance is constant over the horizon. If returns have a
first-order autocorrelation of 0.05, the correct two-day factor is the square root of 2 * (1 + 0.05)
= 1.4491 rather than 1.4142, a 2.47% understatement; at 0.20 the factor is 1.5492, a 9.55%
understatement. Negative autocorrelation corrects in the other direction, 1.3784 at -0.05. The
constructed series has a lag-one return autocorrelation of -0.0123, which is negligible, but the
absolute return autocorrelation is 0.152, so the variance-scaling assumption is more defensible here
than the independence assumption would be for a series with volatility clustering at a longer lag.

### Step 9 — Backtest the tail, not just the average

The average-loss comparison says nothing about whether the printed tail number is realised at the
stated frequency. Two standard tests answer that, and they answer different questions, which is why
both are reported.

```text
FORMULA [D8]: LR_uc = 2 * [ N*ln(N/(n*p0)) + (n-N)*ln((n-N)/(n*(1-p0))) ]
WHERE: N is the observed number of breaches, n the number of scored sessions;
       p0 is the nominal breach probability, 0.01 at 99% confidence;
       LR_uc is asymptotically chi-square with one degree of freedom.
CONSTRAINT: breaches must be counted against the same tail and horizon as p0,
            and overlapping windows invalidate the independence assumption.
NUMERICAL EXAMPLE: N = 26, n = 1,499, p0 = 0.01.
            N/(n*p0) = 26/14.99 = 1.73489, ln = 0.55117, contribution 14.3304.
            (n-N)/(n*0.99) = 1473/1484.01 = 0.99258, ln = -0.00745,
            contribution 1473 * -0.00745 = -10.9730.
            LR_uc = 2 * (14.3304 - 10.9730) = 6.699.
INTERPRETATION: the 95% critical value for one degree of freedom is 3.8415, so the
            candidate's breach rate is significantly above the nominal 1%.
```

The breach counts for the three methods at three confidence levels are worth reading as a single
object, because the pattern is not "the candidate is right and the others are wrong".

| Confidence | Method | Breaches | Expected | Rate | LR_uc | p-value |
|---|---|---|---|---|---|---|
| 99% | Candidate | 26 | 15.0 | 0.0173 | 6.699 | 0.0096 |
| 99% | Baseline 20 | 24 | 15.0 | 0.0160 | 4.627 | 0.0315 |
| 99% | Full-sample constant | 20 | 15.0 | 0.0133 | 1.531 | 0.2160 |
| 97.5% | Candidate | 47 | 37.5 | 0.0314 | 2.301 | 0.1293 |
| 97.5% | Baseline 20 | 48 | 37.5 | 0.0320 | 2.789 | 0.0949 |
| 97.5% | Full-sample constant | 38 | 37.5 | 0.0254 | 0.008 | 0.9309 |
| 95% | Candidate | 86 | 75.0 | 0.0574 | 1.640 | 0.2003 |
| 95% | Baseline 20 | 99 | 75.0 | 0.0660 | 7.412 | 0.0065 |
| 95% | Full-sample constant | 77 | 75.0 | 0.0514 | 0.059 | 0.8089 |

At 99% both the candidate and the baseline fail coverage, and the full-sample constant passes. That
is not a defence of the constant: it passes because it is the *widest* of the three on average, with
a mean ratio to the latent variance of 0.9554 in variance terms, which still leaves it too high
relative to a true 1% tail once the Gaussian shape is assumed. The lesson is that a coverage test
conflates scale error with shape error, and a method can pass coverage by being unnecessarily wide.

The independence test separates the two. Christoffersen's statistic decomposes into the coverage
term plus a term for the clustering of breaches, and in this sample the candidate's clustering term
is 3.051 with a combined statistic of 9.750, p-value 0.0076, while the baseline's clustering term is
0.713 with a combined 5.340, p-value 0.0692. The candidate clusters its breaches more, which is
consistent with it being sharper: a sharper forecast has more days on which the realised return
exceeds a tight bound, and those days arrive in runs. The full-sample constant has the worst
clustering of all at 9.848, because a constant bound is breached exactly when the regime is turbulent
and those are consecutive days by construction.

That combination, better average loss with worse clustering, is the central trade-off of the case
and it is why the guardrail in Section 9 is not simply "do not breach more often". A model that
clusters its breaches is dangerous even when its average rate is acceptable, because clustered
breaches mean the limit is breached on consecutive days, which is when the desk is least able to
absorb it.

### Step 10 — Locate the failure in time

The aggregate numbers hide where the candidate actually struggles. Splitting the scored window at
the constructed regime gives a much more specific diagnosis.

```text
FORMULA [D9]: regime ratio R = mean(f) / mean(true variance) over a sub-window
WHERE: f is the candidate variance forecast; the denominator is the unobservable
       true conditional variance; R is dimensionless and R = 1 is perfect scale.
CONSTRAINT: the sub-window must be defined by information available at the time,
            not by looking at where the forecast failed.
NUMERICAL EXAMPLE: inside sessions 1500 to 1559 the mean candidate variance is
            0.7243 times the mean true variance; outside that window it is 1.0731.
INTERPRETATION: the candidate understates variance by 27.6% in the regime and
            overstates by 7.3% elsewhere, so the average of 1.0466 hides both.
```

The average ratio of 1.0466 looks acceptable and is an artefact of two errors cancelling. Inside the
sixty-session regime the candidate's QLIKE is 0.07923 against the baseline's 0.05527, so the baseline
wins by 30% exactly where it matters most. Outside the regime the candidate's QLIKE is 0.02274
against the baseline's 0.04937, so it wins by 54%. The overall conclusion is therefore conditional:
the candidate is better in calm and transitional conditions and worse in the first weeks of a
regime change, and a proposal that quotes only the aggregate is concealing the case where the number
is used to decide whether to escalate.

The mechanism is arithmetic rather than mysterious. At the regime onset the true variance jumps by
the multiplier of 1.5, but the candidate only learns about the new level through realised squared
returns, and it learns at the rate of the decay. Its weight on the newest observation is 6%, so a
single session in the new regime moves the estimate by at most 6% of the gap. The measured path
shows this directly: in variance units scaled by 10,000, the candidate reads 1.111 at session 1499,
1.045 at 1500, 1.210 at 1510 and 2.258 at 1560, while the true variance reads 1.174, 1.645, 1.834
and 1.730. It took until session 1536 for the candidate's volatility estimate to exceed the
pre-shock level by 50%, and there were 55 consecutive sessions, from 1500 through 1554, in which the candidate understated the true variance by more than 20%.

The baseline fails differently, which is the useful contrast. Its ratio inside the regime is 0.7973,
better than the candidate's 0.7243, because a 20-session window reacts at 5% per session with no
decay and no dilution from a long history. So the regime window inverts the ranking, and any claim
that the candidate dominates has to be restricted to the conditions where it was tested.

### Step 11 — Quantify what the forecast is worth at the limit

The last step connects the statistical comparison to the decision, and it does so with a number the
risk manager can argue about: how much of the limit headroom is consumed by the forecast's typical
error.

At a forecast volatility of 1.6% the pack prints 9.3054 million, leaving 0.6946 million of headroom.
A forecast that is 7.46% too low moves the printed figure to 10.0 million and consumes the entire
buffer; the scale multiplier that brings the observed breach rate exactly to 1% is 1.0642, and at that
multiplier the printed VaR is 9.9028 million, which consumes 99.03% of the headroom. An estimator
that is 6.03% too low, which is the reciprocal of that multiplier, therefore does not merely produce
a worse loss; it produces a number that leaves no room for the next day's error.

That is the argument for the headroom rule in Section 8. The rule is not "forecast well", which is
unactionable, but "print a number whose error band fits inside the buffer", which is checkable. With
a buffer of 0.6946 million against a base of 9.3054 million, the tolerable relative error is 7.46%,
and a review of the candidate's own error distribution shows it exceeded that band on 2.27% of
sessions in the low direction and 1.8% of sessions in the high direction. Those frequencies are the
ones that belong in the model documentation, because they state the probability that the printed
figure is materially wrong rather than the probability that it is wrong on average.
