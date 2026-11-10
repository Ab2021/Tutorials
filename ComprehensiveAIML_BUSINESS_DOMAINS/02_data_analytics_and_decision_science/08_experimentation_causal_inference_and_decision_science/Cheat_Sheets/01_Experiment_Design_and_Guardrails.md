---

## 5. Worked Numerical Example

### 5.1 The scenario, stated as a contract

A checkout team proposes replacing a three-step address form with a single-page form. The decision owner is the checkout product lead; the deadline is the next release train, 22 calendar days out. The pre-registration, committed before any exposure, contains the following.

| Field | Value | Why this value |
|---|---|---|
| Decision | Ship the single-page form to 100% of mobile web sessions, or keep the current form | Binary, reversible, deadline-bound |
| Estimand | Difference in session-level purchase rate, new minus current, over the 14 days after first exposure | The metric the release decision is actually about |
| Randomization unit | Session | The change is rendered per session; users would notice a mix, but a session-level split is technically feasible and is what the team already operates |
| Analysis unit | Session | Matches the randomization unit, so the design effect is 1 |
| Population | Eligible mobile web sessions with a non-empty cart | Pre-treatment filter; excludes sessions that cannot check out |
| Primary metric | Purchases per eligible session, deduplicated on order ID | Prevents double counting a retried confirmation |
| Baseline | $p_C=0.120$ from the trailing 28 days, same population | Historical, same window length |
| Effect of interest | $\delta=0.012$ absolute (10% relative) | Smallest lift whose monthly margin exceeds build and run cost |
| $\alpha$ | 0.05 two-sided | Team convention; guardrail tests get their own budget |
| Power | 0.80 | Team convention |
| Guardrails | Crash rate per session, refund rate per purchase, p95 time-to-confirmation | Each with a tolerance below |
| Stopping rule | Fixed horizon at 13,300 sessions per arm, no interim analysis | Chosen to keep the analysis simple; Section 5.5 quantifies what peeking would have cost |

Two design decisions deserve explicit defence. Analysing at session level matches randomization, so no design effect is applied. But the estimand is then a *session-weighted* effect, which can differ from a user-weighted effect if heavy users behave differently — a limitation recorded in Section 8, not repaired by reweighting after the fact.

### 5.2 Sample size, computed both ways

Using [E-02] with $\sigma_T^2=\sigma_C^2=p_C(1-p_C)=0.12\times0.88=0.1056$:

- variance sum: $0.1056+0.1056=0.2112$
- quantile sum: $z_{0.975}+z_{0.80}=1.95996+0.84162=2.80158$
- squared: $7.84887$
- numerator: $7.84887\times0.2112=1.65768$
- divide by $\delta^2=0.012^2=0.000144$: $1.65768/0.000144=11511.7$… wait, this must be checked step by step.

Recomputing carefully: $7.84887\times0.2112=1.657681$. Then $1.657681/0.000144=11511.67$. That disagrees with the 12,001 quoted earlier, so the earlier figure used the *observed* post-hoc rates (0.13203 and 0.12, variance sum 0.22018) rather than the planning rates. Both numbers are correct for different inputs, and the discrepancy is itself the lesson: using the treatment-arm variance in a pre-launch calculation double-counts information you do not have.

The correct planning figure is $\sigma^2=p_C(1-p_C)$ for both arms, giving **11,512 per arm**. The team rounds up to a 14-day window that, at forecast traffic of 1,900 eligible sessions per day, delivers $1900\times14=26{,}600$ total, i.e. 13,300 per arm — 15.5% more than required, which is prudent headroom for the possibility that $p_C$ drifts.

The realized variance, however, differs: the treated arm's rate is 0.132 and its variance is $0.132\times0.868=0.114576$, so the *achieved* variance sum is $0.1056+0.114576=0.220176$. Plugging that into [E-02] with the same $\delta$ gives $7.84887\times0.220176/0.000144=12000.9$, i.e. 12,001 per arm for 80% power — which is why the realized power at 13,300 per arm is slightly above the target (Section 5.4).

Three checks on this calculation:

1. **Dimension check.** $\delta^2$ has units of (rate)$^2$; the numerator has (rate)$^2$; the quotient is dimensionless. Correct.
2. **Order-of-magnitude check.** With $p=0.12$ and 13,300 per arm, $\mathrm{SE}=0.00407$ and the MDE is $2.80\times0.00407=0.0114$, close to the target 0.012. Correct.
3. **Reverse check.** Substituting 12,001 into the MDE formula returns exactly 0.012000. Correct.

### 5.3 Assignment audit before any outcome is read

The split is 50/50 on a hashed session identifier with a salt committed to the repository. Two checks run on day one.

**Count check.** Realized assignments: 13,306 treatment, 13,294 control, total 26,600. Under the design, each arm has expected count 13,300. The chi-square statistic is
$$\chi^2=\frac{(13306-13300)^2}{13300}+\frac{(13294-13300)^2}{13300}=\frac{36}{13300}+\frac{36}{13300}=0.005414.$$
With one degree of freedom, $p\approx0.942$. No mismatch.

**Sensitivity of the check.** The check is powerful enough to catch material breakage but not microscopic drift. At 26,600 units, a chi-square of 3.84 (the 0.05 critical value) corresponds to a split of about 13,351/13,249, i.e. 50.19/49.81. So the test flags departures of roughly 0.2 percentage points or more. For comparison, a 13,300/12,700 split gives $\chi^2=13.846$, $p\approx0.0002$, which would be a hard stop, and a 13,120/12,880 split gives $\chi^2=2.215$, $p\approx0.137$, which is not.

**Exposure check.** The treatment-rendered log records 13,241 of the 13,306 assigned sessions as having rendered the single-page form, a 99.5% render rate. The 65 missing sessions are treated as non-compliance and remain in the ITT denominator. Dropping them would compare the new form against a control that includes sessions where the form failed, which is a different — and unrequested — estimand.

### 5.4 The primary readout, arithmetic shown

Fourteen days after the last session entered, the analysis script produces:

| Quantity | Treatment | Control |
|---|---:|---:|
| Eligible sessions | 13,300 | 13,300 |
| Purchases (deduplicated on order ID) | 1,756 | 1,596 |
| Purchase rate | 0.1320301 | 0.1200000 |

Step 1 — the difference: $\hat\tau=0.1320301-0.1200000=0.0120301$, i.e. +1.203 percentage points.

Step 2 — the pooled null rate: $\bar p=3352/26600=0.1260150$.

Step 3 — the null standard error from [E-03]:
$$\mathrm{SE}_0=\sqrt{0.1260150\times0.8739850\times\left(\tfrac{1}{13300}+\tfrac{1}{13300}\right)}.$$
The product $0.1260150\times0.8739850=0.110136$; the reciprocal-sum is $2/13300=0.000150376$; their product is $1.6560\times10^{-5}$; the square root is $0.0040696$.

Step 4 — the test statistic: $z=0.0120301/0.0040696=2.9561$.

Step 5 — the two-sided p-value: $2(1-\Phi(2.9561))=0.00312$.

Step 6 — the confidence interval, using the unpooled standard error:
$$\mathrm{SE}_1=\sqrt{\frac{0.1320301\times0.8679699}{13300}+\frac{0.12\times0.88}{13300}}=\sqrt{8.618\times10^{-6}+7.940\times10^{-6}}=\sqrt{1.6558\times10^{-5}}=0.0040690.$$
Margin: $1.95996\times0.0040690=0.0079757$. Interval in absolute terms: $[0.0040544,\,0.0200058]$. In relative terms: $[0.0338,\,0.1667]$.

Step 7 — realized power at these rates. Using $z=\delta/\mathrm{SE}_0$ with $\delta=0.0120301$ and the achieved variance sum, power is $\Phi(2.9561-1.95996)+\Phi(-2.9561-1.95996)$. The first term is $\Phi(0.9961)=0.8404$; the second is $\Phi(-4.916)\approx1.0\times10^{-7}$, which is the leakage from the wrong tail. So realized power is **0.840**, a little above the 0.80 target because traffic overshot.

**What the interval means for the decision.** The whole interval lies above zero, so the change is not neutral. It also lies below the effect that would justify a full rewrite of the form's validation logic — but the pre-registered action threshold was not "exceed the rewrite cost", it was "margin exceeds build and run cost", and the lower bound does clear that. Had the interval been $[-0.002,\,0.016]$, the correct readout would have been state 3 (undersized inconclusive), not "no effect".

### 5.5 What peeking would have done

Traffic arrived evenly, so the planned 26,600 total corresponds to information fractions of 0.25, 0.50, 0.75, and 1.00 at roughly days 3.5, 7, 10.5, and 14. Suppose the team had looked at each of these and had stopped at the first $p<0.05$.

**At n = 6,650 per arm** (quarter of the traffic), the same rates give
$$\mathrm{SE}=\sqrt{0.1260150\times0.8739850\times\tfrac{2}{6650}}=0.005755,\quad z=0.0120301/0.005755=2.0903,\quad p=0.0366.$$
So the test would have "passed" at the first look and the team would have shipped at day 3.5 with an estimate whose interval at that sample size spans $[0.00075, 0.0233]$ — a range from "barely worth it" to "nearly twice the planning assumption."

**At n = 9,500 per arm** the same exercise gives $\mathrm{SE}=0.004815$, $z=2.4983$, $p=0.0125$.

The critical point is not that the effect was real — it was, in this construction — but that the *procedure* of stopping at the first crossing is what Section 4.3.7 quantifies: with five naive looks at $c=1.96$, the simulation estimate of the false-positive rate is 0.143 rather than 0.050. The team would have paid a 2.9× inflation of its error rate in exchange for 10.5 days saved. Sometimes that trade is right; it must be priced, not stumbled into. If the team wanted weekly looks, the honest boundary would have been roughly $c=2.41$ for five looks, and at $n=6{,}650$ per arm the observed $z=2.09$ would not have crossed it.

### 5.6 The guardrail that nearly inverted the decision

The crash guardrail counts sessions where the checkout page raised an unhandled error. Counts: 56 in treatment (rate 0.0042105), 41 in control (rate 0.0030827). The absolute difference is $+0.0011278$, i.e. +1.13 crashes per thousand sessions.

Pooled and unpooled standard errors:
$$\bar p_g=97/26600=0.0036466,\quad \mathrm{SE}_{0,g}=\sqrt{0.0036466\times0.9963534\times\tfrac{2}{13300}}=0.0007407.$$
$$z_g=0.0011278/0.0007407=1.523,\quad p_{\text{two-sided}}=0.128,\quad p_{\text{one-sided}}=0.0635.$$
The one-sided upper 95% bound is $0.0011278+1.644854\times0.0007391=0.0011278+0.0012157=0.0023436$, i.e. up to +76.0% relative to the control crash rate.

**The tolerance decided the outcome.** The guardrail was registered with the rule "breach if the one-sided 95% upper bound on the relative increase exceeds 50%." The upper bound is 76%, so the guardrail **breached** even though the point estimate was not significant. Under the pre-registered rule the experiment therefore could not be shipped to 100% without an engineering fix, and the decision became "ship behind a kill switch at 25% while the form's validation path is hardened."

This is the most instructive number in the sheet. A team that had defined the guardrail as "$p<0.05$ on the crash rate" would have concluded "no significant harm" and shipped a change that plausibly increased crashes by up to 76%. A team that had defined it as "any increase in crash count" would have stopped on noise. The tolerance is what converts a metric into a decision, and it must be chosen before the counts arrive.

**Power of that guardrail.** At 13,300 per arm and $p\approx0.0031$, the MDE at 80% power is
$$\mathrm{MDE}_g=(1.95996+0.84162)\times\sqrt{2\times0.0030827\times0.9969173/13300}=2.80158\times0.0006803=0.001906,$$
which is a **61.8% relative increase**. The observed 37% relative increase was, a priori, undetectable. The guardrail's breach was therefore driven entirely by the bound-based tolerance rule, not by adequate power — and that is a legitimate design only if the team accepts that the guardrail will frequently fire on imprecision. A team that wants a powered guardrail needs roughly
$$n=\frac{(2.80158)^2\times2\times0.0030827\times0.9969173}{(0.25\times0.0030827)^2}=81{,}378$$
sessions per arm to detect a 25% relative increase. At 1,900 eligible sessions per day that is 86 days, which is why guardrail power and calendar time are in direct conflict and must be negotiated explicitly.

### 5.7 The novelty check as an interaction test

Split the 14 days into the first 3 days (2,850 sessions per arm) and the remaining 11 days (10,450 per arm). Observed counts:

| Window | Treatment | Control | Treatment rate | Control rate | Difference |
|---|---:|---:|---:|---:|---:|
| Days 1–3 | 405 | 353 | 0.142105 | 0.123860 | +0.018246 |
| Days 4–14 | 1,351 | 1,243 | 0.129282 | 0.118947 | +0.010335 |

The early effect is 1.82pp and the late effect is 1.03pp, so the decline is real in this data. The question is whether the decline is distinguishable from sampling noise. Compute the difference-of-differences and its standard error:
$$\hat\Delta=0.018246-0.010335=0.007911.$$
$$\mathrm{SE}(\hat\Delta)=\sqrt{\frac{0.1421\times0.8579}{2850}+\frac{0.1239\times0.8761}{2850}+\frac{0.1293\times0.8707}{10450}+\frac{0.1189\times0.8811}{10450}}=0.010082.$$
$$z=0.007911/0.010082=0.7846,\quad p=0.433.$$
The interaction is not significant, and the design's power to detect an interaction of this size is poor: the MDE at 80% for the interaction is $2.80158\times0.010082=0.02825$, nearly eight times the observed decline. So the correct statement is: "the effect is smaller in the later window, but this design cannot distinguish a genuine novelty decay of 0.8pp from zero; a dedicated duration test with a longer horizon would be required." Reporting "novelty effect confirmed" from this table would be a 433-in-1000 kind of claim.

### 5.8 The revenue metric the team also wanted

The stakeholder asked for revenue per eligible session as a secondary metric. Measured values: 59.2 currency units per 1,000 sessions in control ($0.0592$ per session) and 60.6 per 1,000 in treatment ($0.0606$ per session). The difference is 0.1053 units per 1,000 sessions.

This metric is far noisier per unit than the rate. With an illustrative per-session standard deviation of 18 currency units (revenue is heavy-tailed: most sessions spend nothing), the sample size to detect a 10% lift on the treatment mean of 0.0606 — that is, $\delta=0.00606$ — would be
$$n=\frac{7.84887\times2\times324}{0.00606^2}=\frac{5086.1}{3.6724\times10^{-5}}=138{,}497$$
per arm. That is ten times the planned traffic. Even a generous CUPED adjustment at $\rho=0.55$ only brings it to $138{,}497\times0.6975=96{,}601$, still seven times over.

The conclusion the team drew was correct and is the general lesson: **do not add a noisy secondary metric to a test sized for a rate metric and then interpret its sign.** Either pre-register it as a directional diagnostic with no decision authority, or run a separate, longer test. Reporting "revenue also trended up" from a test with 15% of the required sample is the multiplicity problem from Section 4.3.8 wearing a business costume.

### 5.9 The final decision record

| Item | Recorded value |
|---|---|
| Readout state | Decisive win on primary (interval above zero), guardrail breach on crash bound |
| Primary estimate | +1.203pp, 95% CI [0.405pp, 2.001pp], $z=2.956$, $p=0.0031$ |
| Relative estimate | +10.03%, 95% CI [+3.38%, +16.67%] |
| Guardrail | Crash bound +76% vs 50% tolerance → breach |
| Decision | Ship at 25% with kill switch; harden validation; re-read crash rate after 7 days |
| Follow-up | Duration test at 50/50 for 28 days to separate novelty from durable effect |
| Recorded limitations | Session-level estimand, not user-level; 14-day horizon cannot speak to annual effects; guardrail underpowered by design |

The decision was not "ship" or "kill." It was a conditional shipment with a named repair, a re-read date, and an acknowledged power limitation on the guardrail. That is what a complete readout looks like.

> **Takeaway:** in a real readout the primary test and the guardrail test can point in opposite directions, and the pre-registered tolerance — not the p-value — is what resolves the conflict.

## 6. Visual / Geometric Intuition

### 6.1 Two sampling distributions and the two error regions

Picture two bell curves on one axis, both with standard error $\mathrm{SE}$: one centred at 0 (the null) and one centred at the true effect $\delta$. The decision rule draws vertical critical lines at $\pm z_{1-\alpha/2}\mathrm{SE}$ around zero.

```text
                      decision axis (effect estimate)
        |                        |                        |
   -z*SE                       0.0                     +z*SE
        \______________________/  \_____________________/
         reject (negative)          reject (positive)
              ^                          ^
              |  alpha/2 = 0.025         |  alpha/2 = 0.025
                                       
   null curve:      .        /\        .
                            /  \                 <- area outside lines = alpha = 0.05
                           /    \
   alt curve:                  /\                  <- shifted right by delta
                              /  \
                             /    \
                     area right of +z*SE = power = 1-beta
```

Three geometric facts follow, and each is a common interview answer.

**Fact 1: the two areas trade off along one axis.** Making $\alpha$ smaller moves the critical lines outward, which moves the alt curve's mass inside them, lowering power. There is no setting that improves both without changing $\mathrm{SE}$.

**Fact 2: power is the area under the alt curve beyond the line, not the distance between centres.** With $\delta/\mathrm{SE}=2.80$ and $z_{1-\alpha/2}=1.96$, power is $\Phi(2.80-1.96)=\Phi(0.84)=0.80$. The relevant distance is $\delta/\mathrm{SE}-z$, not $\delta/\mathrm{SE}$.

**Fact 3: the same curve picture explains the interval.** The confidence interval is the set of null values that would not be rejected; it is the reflection of the decision region. A 95% interval that excludes zero is exactly a test that rejects at 0.05 — they are the same geometry viewed from two directions, which is why reporting both a p-value and an interval from different variance assumptions (pooled versus unpooled) is an inconsistency.

### 6.2 Why the sample size curve is a square-root curve

Power against $n$ follows $\Phi(\delta\sqrt{n}/\sigma\sqrt2-z)$. Since $\delta\sqrt{n}$ grows with the square root of traffic, doubling power from 0.5 to 0.8 is not doubling traffic. Worked: at 50% power, $z_{1-\beta}=0$, so the requirement is $n=(z_\alpha)^22\sigma^2/\delta^2$; at 80% power the factor is $(z_\alpha+0.8416)^2$. The ratio is $(2.8016/1.9600)^2=2.043$. **Buying the second 30 points of power costs about twice the traffic of the first 50.** This is the single most useful shape fact in experiment sizing, because it explains why "just make it 99% powered" requests are rejected and why 80% is a convention rather than a law.

### 6.3 The interval-width funnel

As $n$ grows, the interval width shrinks as $1/\sqrt n$. In the worked example the half-widths are:

| Total n per arm | $\mathrm{SE}$ | 95% half-width | MDE at 80% |
|---:|---:|---:|---:|
| 6,650 | 0.005755 | 0.01128 | 0.01612 |
| 9,500 | 0.004815 | 0.00944 | 0.01349 |
| 13,300 | 0.004070 | 0.00798 | 0.01140 |
| 20,000 | 0.003319 | 0.00651 | 0.00930 |

The funnel shape matters for planning: going from 13,300 to 20,000 per arm (50% more traffic) buys only 18% narrower intervals. A team that "adds a week" to fix a marginal result is usually buying much less precision than it imagines.

### 6.4 The SRM picture: what a real imbalance looks like

Draw the chi-square distribution with one degree of freedom: it is heavily right-skewed, with 95% of its mass below 3.84. Observed statistics from Section 5.3 and its sensitivity cases:

- 0.0054 (13,306/13,294) — deep in the bulk, unremarkable
- 2.215 (13,120/12,880) — inside, at $p=0.137$
- 8.424 (13,234/12,766) — outside, $p=0.0037$, suspicious
- 13.846 (13,300/12,700) — far outside, $p=0.0002$, broken

The visual lesson is that the difference between "fine" and "broken" is small in raw counts (0.0% versus 2.3% imbalance) but enormous on the chi-square scale, because the statistic is quadratic in the deviation while its null distribution's tail is exponential. This is why the check must be automated with a threshold rather than eyeballed.

### 6.5 The peeking picture: a random walk with a boundary

Under the null, the sequence of standardized test statistics over information fractions behaves like a random walk with independent increments. A fixed-horizon test checks the walk's position at $t=1$ only. A peeking process checks whether the walk ever crosses $\pm c$ before $t=1$, which is a first-passage problem, and first-passage probabilities are much larger than endpoint probabilities.

```text
   Z
 3 |                    *  <- endpoint check only: |Z|=1.2, no reject
   |            *   *
 2 |- - - - - - - - - - - - - boundary +/-1.96
   |     *  /  \  *
 1 |    /  *    \/  *
   |  /              \
 0 *------------------*--------> information fraction t
   0    .25   .5   .75   1.0
        ^
        | path touched ~2.1 at t=0.4 then returned
        naive peeking rejects; fixed-horizon test does not
```

The simulation in Section 4.3.7 makes this concrete: five looks at 1.96 give a 14.3% family-wise error rate. Geometrically, the boundary must be raised to about 2.41 so that the total first-passage probability is back to 5%. The path drawn above is exactly the pathology: an excursion that would have been ignored at a fixed horizon becomes a "result" if someone was watching mid-flight.

### 6.6 Guardrail geometry: the tolerance is a line on the interval scale

The crash guardrail is easiest to see as an interval on a relative scale with a vertical tolerance line.

```text
relative change in crash rate
   -50%      0%       +50%      +100%
     |        |         |         |
     |========|=========|         |      <- 95% CI of relative change: [+? , +76%]
                       ^
                   tolerance line
     0 < observed point estimate (+37%) < +50%
     but the upper bound (+76%) crosses the line -> breach
```

The point estimate sits comfortably inside tolerance; the interval does not. This geometry is the whole argument for bound-based guardrails: a point estimate inside tolerance with an interval reaching 76% is a statement that the true harm could be large, and the decision is about the true harm.

> **Takeaway:** three pictures — two shifted curves, a first-passage walk, and an interval against a tolerance line — explain power, peeking, and guardrail breach without any probability algebra.

## 7. Comparison / Selection Table

### 7.1 Choosing the randomization unit

| Unit | Use when | Precision cost | Failure it invites | Fallback if wrong |
|---|---|---|---|---|
| Session | Change renders per session; user-level mixing is acceptable; high traffic | None (design effect 1) | Estimand is session-weighted, not user-weighted; repeat users counted many times | Re-analyse with user-clustered standard errors and report the estimand change |
| User | Change is durable (onboarding, pricing, recommendations); carryover likely | Low, unless outcomes are per-session | Mixed experience if a user has multiple devices or is logged out | Device-level split with a login-based merge caveat |
| Device | Anonymous traffic dominates; identity is unreliable | Low | Cross-device users get both arms; undermines exchangeability | User split for logged-in traffic, device for the rest, reported separately |
| Store / region | Operational change where units interfere (staffing, delivery supply) | High: design effect from Section 4.3.4 | Few clusters make precision cluster-count-bound | Switchback in time, or a matched-pair design with a paired analysis |
| Time-based switchback | Supply-side or marketplace changes with strong interference | Moderate; needs many periods | Period effects confound treatment if the schedule is not randomized | Randomize the schedule and include period fixed effects |

Selection rule: choose the coarsest unit at which the treatment is well defined and interference is absent. Then compute the design effect, and if the resulting sample size is unaffordable, do not quietly revert to a finer unit — that reintroduces the interference and produces a confidently wrong answer.

### 7.2 Choosing the inference approach for the primary metric

| Situation | Method | Why | Disqualifier |
|---|---|---|---|
| Binary rate, large counts | Two-proportion z with pooled null SE [E-03] | Exact null variance, closed form | Any expected cell under about 5 |
| Binary rate, small counts | Fisher exact or score/Wilson interval | Wald degenerates; score intervals retain coverage | Large data (compute) but fine to 10^5 with a score test |
| Continuous, roughly symmetric | Welch t / difference in means [E-01] | Robust to unequal variances | Heavy tails with small n; use bootstrap |
| Continuous, heavy-tailed | Bootstrap or winsorized mean with pre-declared winsorization | Mean is unstable; tail dominates | Winsorization chosen after seeing data |
| Ratio of random sums | Delta method [E-04] or bootstrap | Denominator randomness must be priced | Denominator materially moved by treatment |
| Clustered assignment | Cluster-robust SEs or cluster-level aggregation [E-06] | Unit-level SEs are too small | Fewer than ~40 clusters; use randomization inference |
| Sequential design | Alpha-spending boundary [E-09] | Controls first-passage error | Boundary chosen post hoc |
| Many metrics | Bonferroni or BH [E-10], or a pre-declared gatekeeping order | Controls FWER or FDR honestly | Family declared after seeing results |

### 7.3 Choosing how to handle non-compliance

| Goal | Analysis | Assumption | Report alongside |
|---|---|---|---|
| Effect of *offering* the change (rollout decision) | ITT: compare assigned arms | Randomization only | Compliance rate, so the reader can scale |
| Effect of *using* the change for compliers | IV/LATE with assignment as instrument | Exclusion restriction, monotonicity, relevance | First-stage strength, complier share |
| Effect on everyone if all complied | ATE via scaling ITT by compliance | Homogeneous effect among compliers or an explicit model | A statement that this is an extrapolation |
| Diagnostic only | As-treated or per-protocol | None — not causal | An explicit "not causal" label |

The selection rule is boring and correct: **report ITT as the headline, IV as a secondary if the first stage is strong, and never present per-protocol as the effect of the feature.** Per-protocol comparisons reintroduce exactly the selection that randomization removed.

### 7.4 Choosing the guardrail set and its tolerance

| Guardrail type | Typical metric | Tolerance style | Why that style |
|---|---|---|---|
| Reliability | Crash/error rate per session | Bound-based: upper 95% relative increase ≤ T | Rare events make point estimates unstable; the bound captures plausible harm |
| Economics | Refund rate, return rate, contribution margin per order | Bound-based on margin, point estimate on rate | Margin has a hard business floor |
| Experience | p95 latency, time-to-interactive | Absolute threshold, not relative | User-perceived thresholds are absolute (e.g. p95 ≤ 2.5 s) |
| Trust | Support contacts per 1,000 orders | Bound-based | Rare, lagged, and expensive to measure well |
| Long-run health | 28-day retention, repeat purchase | Directional only, no gate | Cannot be measured inside the test window; needs a holdback |

The governing principle: **a guardrail without a tolerance is a metric, and a tolerance without a bound is a point-estimate rule.** Bound-based tolerances are usually right for rare harms, because the decision is about the plausible worst case rather than the observed average.

### 7.5 Choosing between experiment, quasi-experiment, and decision without measurement

| Option | Cost | Identification strength | When it is the right choice |
|---|---|---|---|
| Randomized experiment | Traffic + engineering + calendar | Strongest under stated assumptions | Change is reversible, unit-level assignment is feasible, harm is bounded |
| Switchback / cluster-randomized | Same plus design-effect multiplication | Strong if schedule is randomized | Interference is structural (marketplace, shared supply) |
| Difference-in-differences | Historical data only | Requires parallel trends; vulnerable to concurrent shocks | Rollout already happened at a level you cannot randomize |
| Regression discontinuity | Needs a threshold rule | Strong locally at the cutoff | Eligibility is rule-based (score cutoffs, size thresholds) |
| Synthetic control | Aggregate data | Requires a credible donor pool | One or few treated units, long pre-period |
| Ship and monitor | Nearly free | Weak; confounded by seasonality and concurrent changes | Reversal is cheap and harm is bounded, or the effect is large enough to see in aggregate |
| Decide without measurement | Free | None | The decision is dominated by constraint (legal, contractual, irreversible harm) |

The honest entry is "ship and monitor." It is a legitimate choice when the change is cheap to reverse and the expected effect is much larger than plausible drift; it becomes malpractice when it is used to avoid the cost of a test for a change that is expensive to unwind.

### 7.6 Choosing the primary metric form

| Form | Definition | Strength | Weakness |
|---|---|---|---|
| Rate (conversions / eligible) | Binary per unit | Simple variance, closed form | Insensitive to magnitude per converter |
| Mean per unit (revenue / session) | Continuous | Captures magnitude | Heavy tails inflate variance drastically |
| Ratio of sums (revenue / sessions when sessions vary per user) | Ratio metric | Matches business reporting | Needs delta method [E-04]; denominator can move |
| Quantile (p95 latency) | Order statistic | Matches perception | Slower convergence; requires larger n or bootstrapping |
| Trimmed/winsorized mean | Tail-limited mean | Robust to outliers | Result depends on the trimming rule; must be pre-declared |
| Composite/guardrail index | Weighted combination | Single decision number | Weights become the argument; hides component regressions |

### 7.7 Selection summary: the decision tree in words

1. Is the change reversible and is unit-level assignment technically feasible? If no, go to Section 7.5 for a quasi-experimental design; do not run a broken experiment.
2. Do units interfere through a shared resource? If yes, choose a cluster or switchback design and apply [E-06]; if no, choose the coarsest unit at which the treatment is well defined.
3. Is the outcome binary with adequate counts? Use [E-03]. Is it a ratio? Use [E-04]. Is it heavy-tailed? Pre-declare a winsorization or use a bootstrap.
4. Compute $n$ from [E-02] with the *control* variance, add headroom for rate drift, and convert to calendar days at forecast traffic.
5. If the calendar exceeds the decision deadline, either raise $\delta$ (accept that smaller effects will not be resolved), coarsen nothing, or switch to a cheaper design from Section 7.5.
6. Define guardrails with bound-based tolerances, compute their MDE, and state honestly whether they are powered or merely directional.
7. Freeze the analysis script, hash it, and name the person who will run it.

> **Takeaway:** unit choice, inference method, non-compliance handling, guardrail tolerance, and design type are five separable decisions, and each has a disqualifier that must be checked before the others are optimized.
