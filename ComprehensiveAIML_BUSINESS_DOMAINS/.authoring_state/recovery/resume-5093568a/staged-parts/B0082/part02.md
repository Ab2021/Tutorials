## 6. Analysis Walkthrough

### 6.1 The parameter set, stated once

Everything downstream is computed from these constants. They are restated here so that a reader can recompute any figure without reading back.

| Symbol | Meaning | Value | Unit | Source |
|---|---|---|---|---|
| V | accepted orders per hour in the slice | 900 | orders/hour | constructed slice |
| K | contacts placeable per hour | 30 | contacts/hour | staffing plan |
| f | queue fraction, K/V | 0.033333 | dimensionless | derived |
| pi_0 | base late rate | 0.09 | dimensionless | prior quarter |
| P_0 | reference precision at f | 0.375 | dimensionless | offline replay |
| c | loaded cost per contact | 5.80 | currency/contact | payroll |
| e | rescue rate of a successful contact | 0.18 | dimensionless | assumed |
| L | cost of one late delivery | 180 | currency/late order | service credit |
| w | corrupted share of orders | 0.22 | dimensionless | incident |
| pi_A | late rate in the corrupted cohort | 0.23 | dimensionless | incident |
| beta | precision-curve concentration | 0.469949 | dimensionless | fitted |

Two derived constants are used constantly and are worth isolating before anything else.

The value of a prevented late delivery is `e * L = 0.18 * 180 = 32.4` currency units per contact that lands on an order which would otherwise be late.
The break-even precision is `c / (e * L) = 5.80 / 32.4 = 0.179012`, the precision at which one contact's expected rescue value exactly equals its loaded cost.

Independent check on the break-even: at precision 0.179012, one hundred contacts contain 17.9012 late orders, prevent `17.9012 * 0.18 = 3.2222` of them, and save `3.2222 * 180 = 580.00` currency units while costing `100 * 5.80 = 580.00`. The two sides agree.
The same check in per-contact form: `0.179012 * 32.4 = 5.8000`.

Reference value per hour at the starting point: `30 * (0.375 * 32.4 - 5.80) = 30 * 6.35 = 190.50` currency units per hour, or 4,571.99 per day and 1,668,778 per year at the same rate. Those annualised figures are stated only to show scale; the case never claims the rate persists for a year.

### 6.2 Derivation 1: is the drift statistic large, given the sample size?

The first derivation exists because the alert threshold in force is not a probability statement. A PSI of 0.15 means nothing until it is placed against the distribution it would have under no change.

```text
FORMULA [F-01]: PSI = sum over bins b of (q_b - p_b) * ln(q_b / p_b)
WHERE: p_b reference share of bin b [dimensionless, sum 1];
       q_b observed share of bin b [dimensionless, sum 1];
       ln is the natural logarithm; PSI is dimensionless and non-negative.
CONSTRAINT: q_b > 0 in every bin, and the bin edges are fixed in advance and identical
            across windows. If the two share vectors do not each sum to 1, the number is
            not PSI and no threshold applies to it.
NUMERICAL EXAMPLE: p = (0.50, 0.30, 0.20), q = (0.39, 0.234, 0.376) at 900 orders.
       bin 1: (0.39  - 0.50)  * ln(0.78) = -0.110 * -0.248461 = 0.027331
       bin 2: (0.234 - 0.30)  * ln(0.78) = -0.066 * -0.248461 = 0.016398
       bin 3: (0.376 - 0.20)  * ln(1.88) =  0.176 *  0.631272 = 0.111104
       total = 0.027331 + 0.016398 + 0.111104 = 0.154833
INTERPRETATION: 71.76 percent of the statistic comes from the single tail bin, so the alert
       is driven by mass entering "over 5 km", not by a diffuse reshaping of all three bins.
```

Four checks before the number is used for anything.

Each bin contribution is non-negative because `(q - p)` and `ln(q/p)` always share a sign; a negative contribution means an arithmetic error rather than a subtlety.
The shares sum to one: `0.39 + 0.234 + 0.376 = 1.000`, and the reference sums to 1.000.
Converting to counts at 900 orders gives 351, 211 and 338 observed against 450, 270 and 180 expected, both sides summing to 900.
The statistic is invariant to which bin carries the loss of mass: bins 1 and 2 both move by a factor of 0.78, so their log-ratios are identical and only the differences differ.

Now the part that a fixed threshold hides. Under the null hypothesis that the population is unchanged, `N * PSI` is asymptotically chi-square with `k - 1` degrees of freedom, where `N` is the count in the window and `k` the number of bins.

```text
FORMULA [F-02]: N * PSI ~ chi-square(k - 1) under the null of no distribution change
WHERE: N orders in the window [orders]; k number of fixed bins; k - 1 degrees of freedom
       with mean k - 1 and variance 2(k - 1).
CONSTRAINT: fixed bins; expected count N * p_b >= 5 in every bin; windows treated as
            independent. Overlapping windows violate independence and shrink the effective N.
NUMERICAL EXAMPLE: k = 3, so df = 2. The 0.999 quantile is -2 * ln(0.001) = 13.8155.
       one 15-minute window, N = 225: 13.8155 / 225     = 0.061402
       one hour,             N = 900: 13.8155 / 900     = 0.015351
       seven days,     N = 151,200:  13.8155 / 151,200  = 0.0000914
INTERPRETATION: the same PSI of 0.0614 is decisive at 225 orders and unremarkable at
       151,200. A single fixed threshold cannot be a calibrated test at both sample sizes,
       which is why the published "0.2 is severe" folklore is a scale error.
```

The observed 0.154833 in a 900-order hour gives `900 * 0.154833 = 139.35` against a 0.999 critical value of 13.8155, or 10.09 times it. Under the null, a chi-square variable of 2 degrees of freedom exceeds 139.35 with probability `exp(-139.35/2) = 4.8e-31`. "This is noise" is not a live hypothesis at 09:17.

**Executed check (simulation).** Seeded at 20261110, drawing 225 and 900 multinomial samples from the reference distribution, 3,000 trials at each size, computing PSI on the same fixed bins:

| N | simulated median | chi-square median / N | simulated 0.99 | chi-square 0.99 / N | simulated 0.999 | chi-square 0.999 / N |
|---:|---:|---:|---:|---:|---:|---:|
| 225 | 0.006273 | 0.006161 | 0.043968 | 0.040935 | 0.058916 | 0.061402 |
| 900 | 0.001569 | 0.001540 | 0.010814 | 0.010234 | 0.015642 | 0.015351 |

The simulated quantiles track the chi-square prediction to within about 7 percent at 225 orders and about 6 percent at 900, which is as good as an asymptotic approximation should be at these expected counts. The calibration in F-02 is therefore usable, and the correct statement of the alarm threshold is a sample-size-indexed quantile rather than a constant. This check is the reason the runbook in Section 8 stores thresholds as `chi-square quantile / expected window count` and recomputes them when volume changes.

**Power, not just size.** Size asks how often the test fires when nothing happened; power asks whether it fires when something did. Using the required-N relation inverted from F-02:

| Corrupted share w | PSI at that share | N required at alpha 0.001 | Orders at 900/hour |
|---:|---:|---:|---:|
| 0.010 | 0.000394 | 35,050 | 39.0 hours |
| 0.020 | 0.001555 | 8,884 | 9.9 hours |
| 0.033 | 0.004242 | 3,257 | 3.6 hours |
| 0.050 | 0.009345 | 1,478 | 1.6 hours |
| 0.100 | 0.035347 | 391 | 26.1 minutes |
| 0.220 | 0.154833 | 89 | 5.9 minutes |

The table reads as a detection-latency budget, and it contains the case's central irony: at the realised 22 percent corruption the statistic needs only 89 orders — under six minutes of traffic — to cross its own critical value. The alarm arrived twenty-five minutes late for reasons that have nothing to do with statistics and everything to do with the rule layered on top of it (Section 6.8).

**Boundary where the formula breaks.** If a bin's expected count falls below five, the chi-square tail becomes optimistic and the calibrated threshold under-fires. With a smallest reference share of 0.20 this needs `N < 25` orders, so the approximation is safe here but would fail immediately for a ten-bin histogram on a low-traffic channel. In that regime the correct move is to pool adjacent bins until every expected count exceeds five, and to report the pooling rule, because pooling changes the degrees of freedom and therefore the threshold.

### 6.3 Derivation 2: the corrupted share is the wrong exposure variable

The instinctive reading of 22 percent is "under a quarter of orders are affected, so under a quarter of the value is at risk". That is wrong by a factor of about six in this system, and the derivation shows exactly where the error enters.

```text
FORMULA [F-03]: s(w) = min(1, V*w / K), the share of queue slots consumed by the cohort
WHERE: V*w affected orders per hour [orders/hour]; K slots per hour [contacts/hour];
       s dimensionless in [0,1]; s = 1 means the cohort fills the entire queue.
CONSTRAINT: requires the corrupted cohort to outrank clean orders, which holds when the
            unit error inflates the feature into the high-distance bins. If the corruption
            only reorders within the cohort, s is the measured share of cohort orders in
            the top K, not this closed form.
NUMERICAL EXAMPLE: V = 900, K = 30, w = 0.22
       900 * 0.22 = 198 affected orders per hour
       198 / 30 = 6.6, capped at 1, so s = 1.00
       saturation begins at V*w = K, i.e. w* = 30/900 = 0.033333
INTERPRETATION: any corruption above 3.33 percent of orders monopolises the queue. Above
       that share harm is flat in w, so "only 22 percent affected" understates the problem
       by treating a saturated queue as a proportional one.
```

The consequence is stark as a table, and this table is the reason the incident is expensive rather than annoying.

| Corrupted share w | Affected orders/hour | Slots used | Queue precision | Value/hour | Harm/hour |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.0 | 0.0 | 0.375 | 190.50 | 0.00 |
| 0.005 | 4.5 | 4.5 | 0.370 | 185.69 | 4.81 |
| 0.010 | 9.0 | 9.0 | 0.362 | 178.12 | 12.38 |
| 0.020 | 18.0 | 18.0 | 0.334 | 151.05 | 39.45 |
| 0.033 | 30.0 | 30.0 | 0.230 | 49.56 | 140.94 |
| 0.220 | 198.0 | 30.0 | 0.230 | 49.56 | 140.94 |
| 1.000 | 900.0 | 30.0 | 0.230 | 49.56 | 140.94 |

The precision column is not a modelling choice at saturation. When the cohort fills the queue every slot contains a cohort order, so queue precision equals cohort precision, `pi_A = 0.23`. Between zero and saturation the mix is linear in slots, with the clean remainder falling back to the reference curve:

```text
FORMULA [F-04]: P(w) = [A * pi_A + (K - A) * P(f_c)] / K   for A = V*w < K
                P(w) = pi_A                                for A >= K
WHERE: A cohort orders per hour that outrank clean orders; f_c = (K - A) / (V*(1-w)) is
       the queue fraction applied to the clean remainder; P(.) is the curve of F-05.
CONSTRAINT: exact only if cohort orders occupy the top A ranks. If the cohort is shifted
       upward without dominating, use the measured A rather than V*w.
NUMERICAL EXAMPLE: w = 0.02 -> A = 18, f_c = 12 / (900 * 0.98) = 0.013605
       t = 0.013605^0.469949 = 0.132796
       P(f_c) = 0.09*0.132796 / (0.09*0.132796 + 0.91*0.013605)
              = 0.011952 / (0.011952 + 0.012381) = 0.491175
       P(0.02) = (18*0.23 + 12*0.491175) / 30 = (4.1400 + 5.8941) / 30 = 0.334470
INTERPRETATION: harm is convex in w up to saturation and flat after it. Doubling an
       already-saturated corruption changes nothing, while moving from 2 to 3.33 percent
       nearly quadruples hourly harm.
```

Endpoint checks pin the formula down. At `w = 0` it must return `P_0 = 0.375`, and substituting `A = 0`, `f_c = 30/900` returns 0.3749997, the rounding of the curve fit rather than an error. At `w = 1` the cohort is the population and the cap returns `pi_A = 0.23`.

The reference curve inside F-04 needs its own statement, because every currency figure in this case depends on it.

```text
FORMULA [F-05]: P(f) = pi_0 * f^beta / (pi_0 * f^beta + (1 - pi_0) * f), beta = 0.469949
WHERE: f queue fraction in (0,1]; beta a concentration parameter fitted so that
       P(0.033333) = 0.375 and P(1) = pi_0 = 0.09; all quantities dimensionless.
CONSTRAINT: monotone decreasing and bounded in (0.09, 1), exact at both endpoints by
       construction. Invalid outside (0,1]; its curvature is a fit, not a law, so it must
       be replaced by a measured lift curve before any budget decision is made on it.
NUMERICAL EXAMPLE: f = 0.05 -> t = 0.05^0.469949 = 0.242787
       P = 0.09*0.242787 / (0.09*0.242787 + 0.91*0.05)
         = 0.021851 / (0.021851 + 0.045500) = 0.324444
       reported curve value 0.326129; the 0.17-point gap is the fit's tolerance at 0.05.
INTERPRETATION: what the curve is used for here is the ordering and rough magnitude of
       precision as capacity grows, not its third digit. Every claim in this case survives
       a two-point error in any curve value.
```

### 6.4 Derivation 3: from precision to currency, and where the decision flips

Precision is not money. The conversion is one multiplication, and it is the step most incident reports skip on the way to a recommendation.

```text
FORMULA [F-06]: Value/hour = K * (P * e * L - c)
WHERE: K contacts per hour; P realised queue precision; e rescue rate [dimensionless];
       L cost per late delivery [currency/late order]; c loaded cost per contact
       [currency/contact]. Result in currency per hour.
CONSTRAINT: e must be constant across the ranked population. If the rescue rate declines
       with rank — late orders found by a weaker ranker are harder to rescue — the linear
       form overstates the value of the low-precision slots, and the error grows exactly
       when the queue is degraded, which is when the number is being used.
NUMERICAL EXAMPLE: reference P = 0.375
       0.375 * 32.4 = 12.1500 expected rescue value per contact
       12.1500 - 5.80 = 6.3500 net per contact
       6.3500 * 30 = 190.50 currency per hour
    observed P = 0.23
       0.23 * 32.4 = 7.4520; 7.4520 - 5.80 = 1.6520; 1.6520 * 30 = 49.56 currency per hour
INTERPRETATION: the degraded queue is still positive-value, because 0.23 exceeds the
       0.179012 break-even. Harm is 140.94 currency per hour, not a total loss, so the
       correct recommendation is containment rather than shutdown.
```

The break-even deserves to be solved rather than read off, because it decides whether a degraded queue should be paused at all.

Setting `P * e * L = c` and rearranging gives `P* = c / (e * L) = 5.80 / 32.4 = 0.179012`.
In the curve's terms the break-even queue fraction solves `P(f) = 0.179012`, which is `f = 0.225029`, or 202.5 orders per hour at 900 per hour of volume.
Interpretation: the queue stops paying for itself once it is deep enough to include roughly 22.5 percent of all accepted orders.

Two capacity comparisons follow, and both are counterintuitive enough to be worth spelling out.

At half the capacity, 15 contacts per hour, precision is `P(0.016667) = 0.464205` and saturated harm from a 22 percent corruption is `15 * 32.4 * (0.464205 - 0.23) = 113.82` per hour.
At double the capacity, 60 contacts per hour, precision is `P(0.066667) = 0.293545` and saturated harm is `60 * 32.4 * (0.293545 - 0.23) = 123.53` per hour.

Both are lower than the 140.94 at the actual 30 contacts. Harm is therefore not monotone in capacity: it peaks in the middle, where the queue is deep enough to reach the mixed population but not deep enough to be dominated by the clean top of the ranking. That result matters operationally because it denies the team an easy answer — "staff up and the problem goes away" is false, and at 30 contacts the team is sitting near the worst exposure point for this failure mode.

### 6.5 Derivation 4: the dashboard precision is biased low by label maturity

At 09:30 the quality dashboard reports precision 0.238 for the 08:00 to 09:00 hour, whose true precision was 0.375. The gap is not noise and not a join bug; it is what an immature label set does to a conditional probability.

The mechanism is differential maturity. A late delivery takes longer to confirm than an on-time one, so at any fixed age fewer late outcomes are recorded. Conditioning on "recorded" therefore over-samples on-time orders and deflates precision, and the deflation is larger where lateness is rarer.

```text
FORMULA [F-07]: P_obs(t) = P * m_L / (P * m_L + (1 - P) * m_O)
WHERE: m_L = 1 - exp(-t/tau_L) fraction of late outcomes recorded at age t;
       m_O = 1 - exp(-t/tau_O) fraction of on-time outcomes recorded at age t;
       tau_L = 3.2 hours, tau_O = 0.9 hours, t age in hours; P true precision.
CONSTRAINT: requires recording to be independent of the score and of the contact decision.
       If contacted orders are followed up faster than uncontacted ones, m_L becomes a
       function of treatment and the inverse recovers a policy effect rather than truth.
NUMERICAL EXAMPLE: t = 2 hours
       m_L = 1 - exp(-2/3.2) = 1 - 0.535261 = 0.464739
       m_O = 1 - exp(-2/0.9) = 1 - 0.108368 = 0.891632
       numerator   = 0.375 * 0.464739 = 0.174277
       denominator = 0.174277 + 0.625 * 0.891632 = 0.174277 + 0.557270 = 0.731547
       P_obs = 0.174277 / 0.731547 = 0.238231
INTERPRETATION: a stable system reports 0.238 where the truth is 0.375, a 36.5 percent
       understatement created entirely by differential maturity. The number is correct for
       the subset it describes and wrong for the question the incident asked of it.
```

The inverse map is exact and worth keeping: `P = P_obs * m_O / (P_obs * m_O + (1 - P_obs) * m_L)`, which returns 0.375000 from 0.238231 at two hours.

The deflation is not a constant factor, and that is the part that misleads triage.

| True precision P | Observed at 2 hours | Understatement | Observed late rate read from dashboard |
|---:|---:|---:|---:|
| 0.375 | 0.238231 | 36.5% | 0.2382 |
| 0.345 | 0.215159 | 37.6% | 0.2152 |
| 0.318 | 0.195516 | 38.5% | 0.1955 |
| 0.230 | 0.134716 | 41.4% | 0.1347 |
| 0.179 | 0.102052 | 43.0% | 0.1021 |
| 0.090 | 0.049022 | 45.5% | 0.0490 |

Because the deflation is stronger at low precision, a reviewer comparing the two dashboard readings would compute a drop of `0.238231 / 0.134716 = 1.768`, while the true drop is `0.375 / 0.23 = 1.630`. The direction is right, the magnitude is inflated by 8.5 percent, and a report that says "precision fell 43 percent" when it fell 39 percent has mispriced the incident by a visible amount.

Two corrections are wrong often enough to name.

Dividing by a single pooled completeness rate — 0.8532 for this window — returns `0.375 * 0.8532 = 0.3200` instead of 0.375, because it applies the on-time recording curve to late orders.
Treating unrecorded orders as on-time returns `0.375 * 0.464739 = 0.174277`, which confuses "the fraction of late orders that are recorded" with "the fraction of recorded orders that are late". Both errors look disciplined and both would understate harm.

**Boundary where the formula breaks.** If the contact team prioritises follow-up on contacted orders, `m_L` depends on treatment and the inverse recovery is no longer causal. In that regime the defensible comparison is contacted versus uncontacted orders within the same score band, which is exactly the holdout design specified in Section 9. The maturity correction then applies equally to both arms and cancels in the difference — which is why the holdout survives the flaw that breaks the dashboard.
