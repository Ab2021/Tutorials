# Power, MDE, and Sample Size — what the pilot can detect and what it must not pretend to detect

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 02–05; Module 05 topics 01–05
> Version 2026.5

---
## 1. Why this topic exists
A steering committee approves a pilot in eight stores over six weeks. The team runs it, sees no
significant change in conversion, and writes in the readout that "the pilot showed the intervention
does not work". Nobody computed what the pilot could have shown. With eight stores per arm and a
store-level intra-cluster correlation (ICC, the share of outcome variance that lies between stores)
of 0.02, the design could detect a change of roughly 25% relative, and the business cared about a
change of 4%. The null was guaranteed. The sentence is false.

This is one of the most common statistical failures in commercial supply chain work, and it is not
a modelling failure. It is a failure to do arithmetic that takes twenty minutes and could have been
done before the first store was briefed.

Four shapes recur.

**One. The pilot sized by politics.** The number of stores, the number of weeks, and the number of
lanes are chosen because they are available and uncontroversial, not because they can answer the
question. The power calculation is then written backwards to justify the choice. A calculation
written backwards is not a calculation. It is a rationalisation with a formula attached.

**Two. The guardrail read as a safety certificate.** A pilot reports "no significant increase in
damage rate" from a baseline of 0.4%. With 1,200 orders per arm, the design could not distinguish
0.4% from 0.9%. The absence of a significant difference is reported as evidence of safety, when the
data are compatible with more than a doubling of the defect rate. The guardrail was never powered;
it was hoped at.

**Three. The unit that was not the unit.** A distribution centre (DC) pilot with 12 treated and 12
control sites is analysed as 400,000 orders. The analysis treats 400,000 correlated orders as
400,000 independent facts. The standard error is too small by the square root of the design effect,
often a factor of five to fifteen, and the p-value is meaningless. The design was a cluster design;
the analysis was not.

**Four. The ratio metric that looked precise.** Cost per shipment is computed as total cost divided
by total shipments. The analyst powers the test using the standard deviation of per-shipment cost
divided by the square root of the number of shipments. The denominator is random, the numerator and
denominator are correlated, and the variance of a ratio is not the variance of a mean. The pilot is
declared adequately powered on a number that is wrong by a factor of two or more.

> **Note:** Each of these failures produces a decision, not just a bad number. A pilot that could
> not detect the effect teaches the organisation that the intervention failed. That lesson is
> expensive and it is false.

**Why the topic sits here.** Topics 01 to 04 chose the claim, the unit, the design family, and the
geography. This topic is the arithmetic that decides whether the chosen design is worth running.
Topic 07 treats interference, which changes the variance the arithmetic assumes. Topic 08 treats
peeking, which changes the error rate the arithmetic assumes. Topic 09 treats observational
alternatives when no design is available. Everything after this point either repairs an assumption
in the power calculation or replaces the design that needed it.

**The claim of this topic.** A power calculation is not a validation step appended to a pilot. It
is the design. It states what the pilot can detect, what it cannot, and therefore what a null result
will and will not mean. The minimum detectable effect (MDE) is the honest sentence that converts a
pilot from theatre into evidence. An analyst who cannot state the MDE before launch is not running
an experiment. They are running an activity with a dashboard.
## 2. Learning outcomes
By the end of this topic you can:

1. **State** the four quantities — significance level, power, effect size, and sample size — and
   **solve** for any one of them given the other three and the variance of the metric.
2. **Derive** the MDE for a proportion, a mean, and a ratio metric, and **explain** why the ratio
   metric requires a covariance term that the other two do not.
3. **Compute** the design effect, the effective sample size, and the cluster-level MDE for a stated
   ICC, cluster size, and number of clusters per arm.
4. **Adjust** a power calculation for unequal cluster sizes, non-compliance, attrition, and a small
   number of clusters, and **state** the direction and approximate size of each adjustment.
5. **Distinguish** the MDE from the break-even effect and the smallest effect size of interest, and
   **judge** a pilot by whether its MDE sits below the effect that would change the decision.
6. **Power** a guardrail metric as a non-inferiority test with an explicit margin, or **label** it
   as monitored rather than tested.
7. **Write** an MDE paragraph containing every field a reviewer needs, and **state** in one sentence
   what the design could not detect.
8. **Refuse** an impossible pilot honestly: name the constraint, show the arithmetic, propose the
   design that could answer, and specify what will be reported instead.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.
## 3. The core mental model
Hold four pictures. The first is the identity that links the four quantities. The second is the
sampling distribution that gives the identity its meaning. The third is the square-root law that
makes precision expensive. The fourth is the budget of penalties that turns an individual-level
variance into a planning variance.
### 3.1 The four quantities and the one identity
```text
                    THE FOUR QUANTITIES

      alpha  ---------\
      power  ----------\
      effect -----------+---->  fix any three  ---->  the fourth follows
      n      ----------/

      MDE  =  ( z(1 - alpha/2)  +  z(power) )  x  SE(effect)
               \_____________________________/
                    the multiplier

                    80% power, two-sided 5%   ->  2.8016
                    90% power, two-sided 5%   ->  3.2415
                    80% power, one-sided 5%   ->  2.4865

      n    =  ( multiplier )^2  x  variance of the metric  /  MDE^2
```

The multiplier is the only part most people remember, and it is the least interesting part. The
variance is where the design lives. A proportion, a mean, and a ratio metric have three different
variance expressions, and the choice between a customer-level and a store-level assignment changes
the variance by a factor that can exceed a hundred.

**The identity is not an equation you solve once.** It is a constraint you move around. If the
business fixes the available sample, the identity returns the MDE. If the business fixes the effect
it cares about, the identity returns the sample. If both are fixed, the identity returns the power,
which is the probability that the pilot will be informative. The third case is the one that should
be presented to the steering committee, because it is the one that says whether the pilot is worth
running at all.
### 3.2 The sampling distribution behind the identity
```text
   H0: true effect = 0                  H1: true effect = MDE
        |                                     |
        v                                     v
   ....|....                             ....|....
  .    |    .                           .    |    .
 .     |     .                         .     |     .
.      |      .                       .      |      .
-------+---------                     -------+---------
        ^                                    ^
   critical value                        observed mean
   (1.96 SE above 0)                     (2.80 SE above 0)

   alpha = area under H0 beyond the critical value        = 5%
   beta  = area under H1 short of the critical value      = 20%
   power = 1 - beta                                       = 80%
```

The picture explains three facts that prose usually obscures.

**Fact one. Power is a property of a hypothetical effect, not of a study.** "The study had 80%
power" is shorthand for "the study had 80% power if the true effect were exactly the value used in
the calculation". At any other true effect, the power is different. A completed study does not have
power. It has a result.

**Fact two. The MDE is the effect at which the study has 80% power, not the smallest effect that
could reach significance.** A true effect half the size of the MDE still reaches significance about
29% of the time. A true effect of zero reaches it 5% of the time. The MDE is a planning benchmark,
not a detection threshold.

**Fact three. The critical value is a choice, and the power quantile is a choice.** Change alpha
from 0.05 to 0.01 and the multiplier rises from 2.80 to 3.29, so the MDE rises by 17% and the
required sample by 38%. Change power from 80% to 90% and the multiplier rises to 3.24, so the MDE
rises by 16% and the sample by 34%. These are business choices about how much evidence is enough,
and they should be made explicitly rather than inherited.
### 3.3 The square-root law
The MDE falls with the square root of the sample: to halve it you need four times the units, and to
cut it to one third you need nine times. If a pilot can detect a 3% relative change and the business
cares about 1%, the pilot needs nine times the sample, not 50% more. In a cluster design that usually
means nine times the clusters, not nine times the rows, and the clusters may not exist.
### 3.4 The power budget
```text
   START: variance of an individual observation
     |
     |  x DEFF = 1 + (m - 1) x ICC        clustering
     v
   variance of a cluster mean
     |
     |  / compliance^2                    dilution of the delivered effect
     v
   variance per unit of delivered effect
     |
     |  x (t / z)^2                       few clusters, estimated variance
     v
   PLANNING VARIANCE   --->   MDE = multiplier x sqrt(planning variance / units)
```

Each arrow is a penalty that the naive calculation omits. The design effect is usually the largest.
Compliance dilution is usually the most forgotten. The small-cluster correction is usually the most
denied. A power calculation that omits all three can understate the MDE by a factor of ten and still
look rigorous, because it contains a formula.
### 3.5 Three effects that are not the same
| Quantity | Question it answers | Who supplies it | Typical units |
|---|---|---|---|
| MDE | What can this design detect at stated alpha and power? | The design and the variance | Outcome units and per cent of baseline |
| Break-even effect | What effect would pay for the intervention's cost? | Finance and operations | Currency per unit, or per cent |
| Smallest effect size of interest (SESOI) | What is the smallest effect the business would act on? | The decision-maker | Outcome units, stated before the test |

A pilot is worth running as a confirmatory test only when the MDE sits below the break-even effect,
with enough headroom that a plausible true effect is likely to be detected. If the MDE is above the
break-even, the pilot can produce a significant result only by luck, and a null result is
uninformative about the decision. If the MDE is far below the break-even, the pilot may be
over-powered: it will detect changes too small to matter and consume sample that another question
needs more.

> **Note:** The three quantities are often reported as one. "We need to detect a 2% lift" can mean
> any of them. Ask which, and ask who chose it. The answer usually reveals that nobody did.
## 4. Definitions and vocabulary
The fourth column names the unit or coding convention. A definition without a unit is the defect
this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Statistical power | Probability that a test rejects the null when the true effect equals the value used in the calculation | Reported as a property of a completed study | Proportion, 0 to 1; conventionally 0.80 |
| Significance level (alpha) | Probability of rejecting a true null hypothesis | Confused with the p-value of one result | Proportion, usually 0.05, two-sided |
| Type I error | Rejecting a true null | Called a false positive without stating the null | Indicator, 0 or 1 |
| Type II error (beta) | Failing to reject a false null | Omitted from reporting | Proportion; power equals 1 minus beta |
| Effect size | The quantity the decision turns on, in outcome units | Used to mean a standardised coefficient | Outcome units, absolute; or per cent of baseline |
| Minimum detectable effect (MDE) | Smallest true effect a design detects with stated alpha and power | Treated as the smallest effect that exists | Outcome units and per cent of baseline |
| Minimum worthwhile effect | Smallest effect that would justify the intervention's cost | Confused with the MDE | Currency per unit, or per cent |
| Break-even effect | Effect at which incremental contribution equals incremental cost | Computed without stating the cost inputs | Outcome units |
| Smallest effect size of interest (SESOI) | Smallest effect the business would act on, stated before the test | Chosen after seeing the result | Outcome units |
| Sample size | Number of independent units in the analysis | Equated with row count under clustering | Count of assignment units |
| Standard error (SE) | Standard deviation of an estimator's sampling distribution | Called the standard deviation of the data | Outcome units |
| Standard deviation (SD) | Dispersion of individual observations | Used interchangeably with SE | Outcome units |
| Variance | Square of the standard deviation | Reported without saying of what | Outcome units squared |
| z-multiplier | Sum of the critical value and the power quantile | Used without stating alpha and power | Dimensionless; 2.80 at 80% power, two-sided 5% |
| Confidence interval (CI) | Interval with a stated long-run coverage probability | Interpreted as the probability that the parameter lies inside | Outcome units, with coverage |
| Half-width | Half the width of a confidence interval | Compared with the MDE as if identical | Outcome units; about 0.70 times the MDE at 95% |
| Design effect (DEFF) | Variance under clustering divided by variance under individual assignment | Reported as a count | Dimensionless, at least 1 |
| Intra-cluster correlation (ICC) | Share of total variance that lies between clusters | Assumed zero when unmeasured | Proportion, 0 to 1 |
| Effective sample size | Row count divided by the design effect | Reported as the row count | Count of independent units |
| Cluster size (m) | Number of analysis units inside one cluster | Treated as constant | Count per cluster |
| Coefficient of variation of cluster size | SD of cluster sizes divided by the mean | Ignored in the design effect | Dimensionless |
| Degrees of freedom (between-within) | df for the cluster comparison, often clusters minus two | Set to the row count | Count |
| Randomisation inference | Inference from the distribution of estimates over reassignments | Replaced by a z-test when clusters are few | p-value with a floor of 1 divided by the number of assignments |
| Ratio metric | A metric whose value is a numerator divided by a random denominator | Treated as a mean of per-unit ratios | Currency per count, with both sums |
| Delta method | First-order Taylor approximation to the variance of a ratio | Applied without the covariance term | Variance in metric units squared |
| Covariance term | The 2R times Cov(Y, X) term in the ratio variance | Omitted, which biases the SE | Metric units squared |
| Post-treatment denominator | A denominator that is itself affected by the treatment | Used as if fixed | Count, with the treatment's effect on it stated |
| Guardrail metric | Metric that must not degrade, used to veto a winner | Tested for superiority after the fact | Definition plus a non-inferiority margin |
| Non-inferiority margin | Largest degradation the business will tolerate | Set to zero, which requires infinite sample | Outcome units |
| Attrition | Loss of units between assignment and analysis | Ignored in the power calculation | Proportion |
| Compliance | Share of assigned units that received the treatment as designed | Assumed to be 100% | Proportion |
| Dilution | Attenuation of the intention-to-treat effect by non-compliance | Reported as if it were the delivered effect | Multiplicative factor |
| Post-hoc power | Power recomputed from the observed effect | Presented as evidence about the study | Proportion; not informative |
| Conditional exaggeration | Upward bias in the estimated effect among results that reach significance | Called winner's curse without the mechanism | Ratio of expected estimate to true effect |

Five of these carry most of the practical damage.

**MDE versus break-even versus SESOI.** The MDE is a property of the design. The break-even is a
property of the economics. The SESOI is a property of the decision. A pilot can have an MDE of 1.2
percentage points, a break-even of 0.11 points, and a SESOI of 0.3 points, and those three numbers
imply three different verdicts. Reporting only the MDE hides the comparison that matters.

**Power versus precision.** Power is the probability of crossing a threshold. Precision is the width
of the interval. They move together for a fixed effect, but they answer different questions. A
business audience understands "the interval is plus or minus 0.8 points" more readily than "the
study had 80% power to detect 1.2 points". Report both, and report the half-width.

**Design effect versus effective sample size.** The design effect is a variance ratio. The effective
sample size is a row count divided by it. The effective sample size has a ceiling of clusters divided
by the ICC, no matter how many rows sit inside each cluster. Reporting 1.7 million sessions when the
effective sample is 8,000 is not a rounding difference; it is a false claim about evidence.

**Ratio metric versus mean of ratios.** A ratio of totals weights each unit by its denominator. A
mean of per-unit ratios weights each unit equally. Cost per shipment computed as total cost divided
by total shipments is the first. The average lane's cost per shipment is the second. They answer
different questions, and the variance formulas differ.

**Post-hoc power versus prospective power.** Prospective power is computed from a variance and an
effect chosen before the data. Post-hoc power is computed from the observed effect, which is the
quantity the test was trying to estimate. It is a monotone transformation of the p-value and adds no
information. It is also, in most presentations, an attempt to explain away a null.
## 5. The mechanics, step by step
Twenty-four steps in six phases. Each step names the check that fails if it is skipped. The order
matters: the variance structure is fixed before the sample size is solved, because solving first
produces a number that the design cannot deliver.
### Phase A — Frame the decision before touching a formula
**Step 1. Write the decision rule as a sentence with a role, an action, a threshold, and a date.**
"The trading director will switch the click-and-collect promise on for all stores from the start of
the fourth quarter if the pilot shows a conversion gain above the break-even with a 95% interval
excluding zero, and no guardrail degradation beyond the stated margin." That sentence contains the
primary metric, the comparison, the decision rule, and the deadline. *Fails if skipped:* the power
calculation has no target, so any sample size can be defended.

**Step 2. State the break-even effect and show its inputs.**
Break-even is the effect at which incremental contribution equals incremental cost. For a
conversion intervention with cost `c` per order, contribution `v` per order, and baseline conversion
`p`, the break-even lift in conversion is:

```text
   delta*  =  c x p / (v - c)
```

With `c = 0.90`, `v = 31.00`, and `p = 0.036`, `delta* = 0.001076`, or 0.108 percentage points, or
2.99% relative. State every input and its source. *Fails if skipped:* the MDE is compared with zero,
and a significant but unprofitable effect is rolled out.

**Step 3. State the SESOI separately from the break-even.**
The break-even is arithmetic. The SESOI is a judgement: the smallest change the decision-maker would
act on, given competition, capacity, and the cost of switching. They are often close and sometimes
not. If the SESOI is larger than the break-even, the design should be powered for the SESOI, because
a smaller significant effect will not change the decision. *Fails if skipped:* the pilot is powered
for a number nobody cares about.
### Phase B — Fix the variance structure
**Step 4. Choose the analysis unit, and make it the assignment unit unless you can model the
clustering.**

The analysis unit is the object whose independent variation the standard error assumes. If stores
were assigned, stores are the units. Analysing sessions inside stores requires a variance component
for stores, which is a different calculation and usually a different conclusion. *Fails if skipped:*
the standard error is too small by the square root of the design effect.

**Step 5. Write the variance expression for the metric type.**
| Metric type | Estimator | Variance of the estimator, one arm | What the symbols mean |
|---|---|---|---|
| Proportion | `p_hat` | `p(1 - p) / n` | `n` independent units; `p` baseline rate |
| Mean | `y_bar` | `sigma^2 / n` | `sigma` SD at the analysis unit |
| Ratio | `R = sum(Y) / sum(X)` | `[sigma_Y^2 + R^2 sigma_X^2 - 2 R sigma_XY] / (k X_bar^2)` | `k` units; `Y` and `X` are unit totals; `sigma_XY` their covariance |
| Cluster mean, any metric | mean of `m` rows | `sigma^2 [rho + (1 - rho)/m]` | `rho` ICC, `m` rows per cluster |

The first three rows are the anatomy the topic exists to teach. The fourth is the correction that
converts any of them to a cluster design. *Fails if skipped:* the wrong variance is used, and every
later number inherits the error.

**Step 6. Estimate the variance components from pre-period data, not from a vendor default.**
For a proportion, use the observed baseline rate in the same population and season. For a mean, use
the SD at the analysis unit, not at the row level. For a ratio, you need the SD of the numerator, the
SD of the denominator, and their covariance at the unit level. For a cluster design, you need the
ICC and the distribution of cluster sizes. *Fails if skipped:* the calculation is an assumption
dressed as a measurement.

**Step 7. Compute the design effect and the effective sample size.**
```text
   DEFF            =  1 + (m - 1) x rho
   effective n     =  n / DEFF
   ceiling         =  k / rho          as m grows without limit
```

With `m = 18,000`, `rho = 0.012`, and `k = 48` stores per arm, `DEFF = 216.99` and the effective
sample is 3,982 per arm against 864,000 rows. The ceiling is `48 / 0.012 = 4,000`, which the
effective sample is already approaching. Adding rows inside stores buys almost nothing. *Fails if
skipped:* the row count is mistaken for evidence.

**Step 8. Decide the analysis route, because it changes the multiplier.**
| Route | Units of inference | When it is right | Cost |
|---|---|---|---|
| Cluster-level summary | `k` clusters per arm | Few clusters, unequal sizes | Loses within-cluster precision |
| Mixed model | Rows, with a cluster random effect | Many clusters, cluster-specific estimand wanted | Distributional assumption |
| Cluster-robust SE | Rows, corrected at cluster level | Many clusters, few parameters | Downward bias below about 20 to 40 clusters |
| Randomisation inference | Assignment mechanism | Very few clusters, any statistic | Computation; p-value floor |

Below about 20 clusters per arm, the honest routes are cluster-level summaries or randomisation
inference. A cluster-robust standard error with 12 clusters and no correction is not conservative;
it is wrong. *Fails if skipped:* the design is analysed as something it is not.
### Phase C — Solve the identity
**Step 9. Fix alpha, sidedness, and power, and write down why.**
Two-sided 5% is a convention, not a law. One-sided is defensible only when a degradation in the other
direction would not change the decision, which is rare for a guardrail and common for a
non-inferiority test. Eighty percent power is a convention; ninety percent is common when a false
null is expensive. *Fails if skipped:* the multiplier is chosen by habit, and nobody can defend it.

**Step 10. Solve for the number of units at the target effect.**
For equal arms and a two-sided test:

```text
   n per arm  =  (z(1 - alpha/2) + z(power))^2  x  variance per unit  /  delta^2
```

For a proportion, the variance per unit is `p1(1 - p1) + p0(1 - p0)`; for a mean it is `2 sigma^2`;
for a ratio it is twice the delta-method variance. The formula is the same identity in three
costumes. *Fails if skipped:* the sample is computed for the wrong metric type.

**Step 11. Invert to the MDE at the sample the design can actually deliver.**
The available sample is usually fixed by traffic, store count, or calendar. Invert the identity and
report the MDE, not the sample the design would have needed. *Fails if skipped:* the pilot is
described by its ambition rather than its capability.

**Step 12. Convert units to calendar time and check the season.**
If 18,000 sessions per store arrive over 12 weeks, the design needs those 12 weeks and no less. If
the required sample implies 30 weeks, the window crosses at least one seasonal peak and the baseline
is not stable. State the weeks and the season. *Fails if skipped:* the pilot runs long enough to
include a peak and the effect is confounded with seasonality.

**Step 13. Adjust for compliance, attrition, and measurement error.**
| Adjustment | Effect on the calculation | Factor |
|---|---|---|
| Compliance `c` | ITT effect is diluted by `c`; inflate n by `1/c^2` | 80% compliance: 1.56 times the sample |
| Attrition `a` | Analysable units are `1 - a` of assigned; inflate by `1/(1 - a)^2` | 15% attrition: 1.38 times |
| Measurement error | Attenuates the observed effect; inflate by `1/reliability^2` | Reliability 0.9: 1.23 times |

These multiply. A design with 80% compliance and 15% attrition needs about 2.15 times the sample
computed on the clean effect. *Fails if skipped:* the pilot is powered on a delivered effect that
will not be delivered.
### Phase D — Guardrails and multiple metrics
**Step 14. Separate primary, secondary, and guardrail metrics, and say what each is for.**
The primary metric drives the launch decision. Secondaries explain the mechanism and are exploratory
unless pre-specified. Guardrails veto: a degradation beyond the margin blocks a launch even if the
primary wins. A metric cannot play two roles. *Fails if skipped:* the primary becomes whichever
metric moved.

**Step 15. Power the guardrail as a non-inferiority test with an explicit margin.**
The question is not "did the guardrail improve?" but "can we rule out a degradation larger than the
margin?". The multiplier for a one-sided 5% non-inferiority test at 80% power is 2.4865, not 2.8016.
The margin is a business input: the largest degradation operations will tolerate. *Fails if
skipped:* "no significant difference" is reported as safety.

**Step 16. Put multiplicity into the power calculation, not the readout.**
If there are `m` primary comparisons at alpha 0.05, the family-wise error rate is
`1 - 0.95^m`: 9.75% for two, 14.26% for three, 40.13% for ten. Either split alpha and inflate the
sample, designate one primary and label the rest, or use a hierarchical gate. *Fails if skipped:*
the nominal power is correct for each test and wrong for the family.
### Phase E — Stress the calculation
**Step 17. Produce a sensitivity table over the inputs you are least sure of.**
Vary the ICC, the baseline rate, the cluster size, and the compliance rate across plausible ranges.
Report the MDE at each corner. A single-point MDE implies a precision the inputs do not have.
*Fails if skipped:* a reviewer who doubts the ICC has no way to see whether the conclusion survives.

**Step 18. Simulate when the analytic formula is strained.**
Ratio metrics, few clusters, unequal cluster sizes, and heavy-tailed outcomes are all cases where the
analytic formula is an approximation. Simulate the design under a plausible data-generating process
and check that the analytic power is within Monte Carlo error. *Fails if skipped:* the formula is
treated as exact where it is not.

**Step 19. Compare at least two designs on MDE, calendar time, and cost.**
The cluster design, the switchback, the matched-pair design, and the stepped rollout have different
variance structures and different costs. Present them side by side. The comparison is the analysis;
the recommendation follows from it. *Fails if skipped:* the first design considered becomes the
design defended.
### Phase F — Write it down and govern it
**Step 20. Write the MDE paragraph with every field.**
```text
  DESIGN      assignment unit, analysis unit, arms, clusters per arm, rows per cluster
  METRIC      definition, numerator, denominator, unit, baseline value
  VARIANCE    SD or rate, ICC, cluster-size CV, source of each
  ALPHA       value, sidedness, and why
  POWER       value, and the effect at which it applies
  MDE         absolute and relative, with the arithmetic shown
  BREAK-EVEN  absolute and relative, with cost inputs
  ADJUSTMENTS compliance, attrition, measurement error
  LIMIT       what the design cannot detect, in one sentence
```

**Step 21. Pre-register the calculation and have the decision-maker sign it.**
The signature is not bureaucracy. It converts a statistical objection into a governance artefact: the
decision-maker has agreed, before the data, what effect the pilot can and cannot see. *Fails if
skipped:* the MDE is discovered after a null and treated as an excuse.

**Step 22. At readout, report the interval against the MDE and the break-even, not against zero.**
A 95% interval of `[-0.4, +0.9]` percentage points against an MDE of 1.17 says the data are compatible
with a loss of 0.4 and a gain of 0.9, and that the design could not have distinguished a gain of 0.9
from zero reliably. That is the honest sentence. *Fails if skipped:* the p-value is the only number
reported, and non-significance is read as no effect.

**Step 23. State the limit in one sentence, in the language of the decision.**
"The pilot cannot detect a conversion change smaller than 1.17 percentage points, and the break-even
is 0.11 points; a null result is consistent with an effect ten times the break-even." That sentence
does more work than a page of caveats. *Fails if skipped:* the reader supplies their own, usually
wrong, interpretation.

**Step 24. Archive the calculation with its inputs, code, and date.**
The power calculation is a design artefact. Store it beside the pre-registration, with the data
extract that produced the variance inputs. A calculation that cannot be reproduced cannot be
audited, and one that cannot be audited will be rewritten after the result. *Fails if skipped:* the
organisation cannot learn from its own designs.
## 6. Worked example
Ravensbourne Retail operates 96 stores with an e-commerce channel fulfilled from store stock. The
trading director proposes a same-day click-and-collect promise: order by 14:00, collect from the
store after 17:00. The promise requires staff training and a small reserve stock in every
participating store, so it cannot be delivered to half the sessions inside a store. The store is the
assignment unit. All figures are constructed to be internally consistent and checkable by hand.
### 6.1 The decision and the arithmetic base
The decision: switch the promise on for all stores from the start of the fourth quarter if the pilot
shows a conversion gain whose interval clears the break-even, with no guardrail degradation beyond
0.5 percentage points. The pilot pool is 96 stores over 12 weeks, 48 per arm, with about 18,000
eligible sessions per store over the window. That is 864,000 sessions per arm and 1,728,000 in
total.

| Input | Value | Source or note |
|---|---:|---|
| Baseline conversion `p` | 3.60% | Same 12 weeks last year, same stores |
| Sessions per store | 18,000 | Four-quarter average, seasonally adjusted |
| Orders per store | 1,200 | 18,000 times 3.6% plus rounding |
| Average order value | GBP 118.50 | Median of the last four quarters |
| SD of order value | GBP 54.00 | Order-level, winsorised at the 99.5th percentile |
| Revenue per session | GBP 41.20 | Total revenue divided by total sessions |
| SD of store revenue | GBP 96,000 | Across stores, 12-week totals |
| SD of store sessions | 2,400 | Across stores, 12-week totals |
| Correlation of store revenue and sessions | 0.85 | Pre-period, across the 96 stores |
| On-time pick rate | 97.8% | Store-level, 12-week average |
| ICC, conversion | 0.012 | Pre-period store-level variance decomposition |
| ICC, on-time rate | 0.030 | Same method, different metric |
| Promise cost per order | GBP 0.90 | Labour and reserve-stock carrying cost |
| Contribution per order | GBP 31.00 | Revenue minus product and delivery cost |

The break-even lift in conversion follows from Step 2:

```text
   delta* = c x p / (v - c)
          = 0.90 x 0.036 / (31.00 - 0.90)
          = 0.0324 / 30.10
          = 0.0010764
          = 0.10764 percentage points
          = 2.99% relative to the 3.60% baseline
```

So the business needs a conversion gain of about 0.108 percentage points to pay for the promise. The
question for the pilot is whether it can detect a change of that size, or at least a change
comfortably above it. The answer, computed below, is no.

> **Note:** The break-even assumes the promise affects only conversion and cost. If it also raises
> basket size, or shifts existing orders from home delivery to collection at a lower delivery cost,
> the true break-even is smaller. A smaller break-even makes the pilot's shortfall worse, not better.
### 6.2 The proportion: conversion
Conversion is a proportion. At the session level, the variance of one session's outcome is
`p(1 - p) = 0.036 x 0.964 = 0.034704`. If sessions were independent and assigned individually,
864,000 sessions per arm would give a standard error of the difference of 0.0283 percentage points
and an MDE of 0.0794 points, which is below the break-even. That is the calculation a team produces
when it forgets that the treatment is assigned to stores.

At the store level, the variance of a store's conversion rate is:

```text
   var(store mean) = p(1 - p) x [rho + (1 - rho)/m]
                   = 0.034704 x [0.012 + 0.988/18,000]
                   = 0.034704 x 0.012054889
                   = 0.000418353
```

The standard error of the difference across 48 stores per arm is `sqrt(2 x 0.000418353 / 48) =
0.0041751`, or 0.4175 percentage points. The MDE at 80% power and a two-sided 5% test is
`2.8016 x 0.0041751 = 0.011697`, or 1.1697 percentage points.

The design effect is `1 + 17,999 x 0.012 = 216.99`, and `sqrt(216.99) = 14.73`. The cluster MDE is
14.73 times the naive MDE, exactly as the design effect predicts. The 1.7 million sessions are
worth about 8,000 independent sessions, which is 4,000 per arm at the ceiling.

**The first verdict.** The pilot can detect a 1.17 percentage point conversion change. The break-even
is 0.108 points. The MDE is 10.9 times the effect that would pay for the promise. If the true lift
were 5% relative, or 0.18 points, the power would be:

```text
   power = Phi(0.0018 / 0.0041751 - 1.96)
         = Phi(0.4311 - 1.96)
         = Phi(-1.5289)
         = 0.063
```

About 6%. A null result is the overwhelmingly likely outcome even if the promise works and is
profitable. The pilot is not a test of the promise. It is a test of whether the promise produces an
implausibly large effect.
### 6.3 The mean: average order value
Average order value is a mean over orders. The order-level SD is GBP 54.00. Ignoring clustering,
5,086 orders per arm would detect a GBP 3.00 change at 80% power:

```text
   n per arm = 2 x 7.84896 x 54.00^2 / 3.00^2
             = 15.69792 x 2,916 / 9
             = 5,086
```

The pilot has 57,600 orders per arm, so the order-level MDE is much smaller than GBP 3.00:
`2.8016 x sqrt(2 x 2,916 / 57,600) = GBP 0.8915`, or 0.75% relative.

But orders are clustered within stores. With `rho = 0.02` and `m = 1,200` orders per store:

```text
   var(store mean) = 2,916 x [0.02 + 0.98/1,200]
                   = 2,916 x 0.020816667
                   = 60.7014
   SE(diff)        = sqrt(2 x 60.7014 / 48)
                   = 1.5904
   MDE             = 2.8016 x 1.5904
                   = GBP 4.4556
                   = 3.76% relative
```

The design effect is `1 + 1,199 x 0.02 = 24.98`, and the effective sample is 2,306 orders per arm.
To detect a GBP 3.00 change at the store level would need 106 stores per arm; the pilot has 48. At
the achieved 48 stores, a GBP 3.00 change has power `Phi(3.00/1.5904 - 1.96) = Phi(-0.074) = 0.47`.
About 47%, which is a coin flip.
### 6.4 The ratio: revenue per session
Revenue per session is a ratio of two store-level totals: `R = sum(Y) / sum(X)`, where `Y` is store
revenue and `X` is store sessions. The denominator is random, and revenue and sessions are strongly
correlated across stores. The delta-method variance for one arm is:

```text
   var(R) = [sigma_Y^2 + R^2 sigma_X^2 - 2 R sigma_XY] / (k X_bar^2)

   sigma_Y      = 96,000          sigma_Y^2 = 9,216,000,000
   sigma_X      = 2,400           sigma_X^2 = 5,760,000
   R            = 41.20           R^2       = 1,697.44
   sigma_XY     = 0.85 x 96,000 x 2,400 = 195,840,000
   X_bar        = 18,000          k         = 48
```

The numerator is `9,216,000,000 + 9,777,254,400 - 16,137,216,000 = 2,856,038,400`. The denominator is
`48 x 324,000,000 = 15,552,000,000`. So `var(R) = 0.183644`, and the standard error of the difference
is `sqrt(2 x 0.183644) = 0.6060`. The MDE is `2.8016 x 0.6060 = GBP 1.6979`, or 4.12% of GBP 41.20.

| Variance treatment | SE of the difference | MDE | MDE relative |
|---|---:|---:|---:|
| Session level, clustering ignored | GBP 0.1461 | GBP 0.409 | 0.99% |
| Store level, covariance set to zero | GBP 1.5629 | GBP 4.378 | 10.63% |
| Store level, delta method with covariance | GBP 0.6060 | GBP 1.698 | 4.12% |

Two errors sit in this table, and they point in opposite directions. Ignoring clustering understates
the MDE by a factor of 4.15. Ignoring the covariance overstates it by a factor of 2.58. The correct
answer is between them. A team that computes either of the first two rows and reports it as "the"
MDE has not made a rounding error; it has described a different design.

The covariance is doing real work: it removes about 85% of the naive numerator. Busy stores have
both more sessions and more revenue, so the ratio is more stable than either component. This is not
a universal property of ratio metrics. For cost per shipment, cost and shipments may be negatively
correlated in some periods, and omitting the covariance would then understate the MDE. The sign of
the covariance is an empirical question, and it must be estimated, not assumed.

> **Note:** The delta method assumes the denominator is bounded away from zero and the ratio is
> approximately normal. With 48 stores and a skewed revenue distribution, check the approximation by
> simulation before relying on the third row.
### 6.5 The guardrail: on-time pick rate
The guardrail is a proportion with a high baseline, 97.8%, and a low absolute variance. That does not
make it easy to power. With `rho = 0.03` and `m = 1,200`:

```text
   var(store mean) = 0.978 x 0.022 x [0.03 + 0.97/1,200]
                   = 0.021516 x 0.030808333
                   = 0.000662884
   SE(diff)        = sqrt(2 x 0.000662884 / 48)
                   = 0.0052555
   MDE, two-sided 80% = 2.8016 x 0.0052555
                      = 0.014724
                      = 1.4724 percentage points
```

The pilot can rule out a degradation of about 1.47 points at 80% power. But the guardrail question is
one-sided and non-inferiority: can we rule out a degradation larger than the 0.5-point margin? The
one-sided 5%, 80% multiplier is 2.4865, giving a detectable margin of
`2.4865 x 0.0052555 = 1.307` points.

```text
   stores per arm for a 0.5 pp margin
     = 2 x 2.4865^2 x 0.021516 x 0.030808333 / 0.005^2
     = 328
```

The pilot has 48 stores per arm. It can rule out a 1.31-point degradation, not the 0.5-point
degradation the business says it tolerates. The guardrail is not powered for its own margin. The
honest report says: "The pilot can exclude a degradation larger than 1.31 points. It cannot exclude a
degradation of 0.5 points. We will report the interval and treat the guardrail as monitored, not
tested."
### 6.6 What the pilot can and cannot say
| Metric | Baseline | MDE at 48 stores per arm | Relative MDE | Break-even or margin | Verdict |
|---|---:|---:|---:|---:|---|
| Conversion | 3.60% | 1.1697 pp | 32.49% | 0.1076 pp | Cannot detect the decision effect |
| Average order value | GBP 118.50 | GBP 4.4556 | 3.76% | GBP 3.00 target | Underpowered by about 2.2 times |
| Revenue per session | GBP 41.20 | GBP 1.6979 | 4.12% | Not stated | Detects only large shifts |
| On-time pick rate | 97.80% | 1.307 pp (one-sided margin) | 1.34% | 0.5 pp margin | Cannot rule out the tolerated loss |

Read the table as limits, not results. The pilot can establish operational feasibility, measure the
direction and rough size of a large conversion change, and rule out effects above 1.17 points. It
cannot detect the 0.108-point break-even effect or a plausible 0.18-point effect, rule out the
0.5-point guardrail loss, estimate a segment effect, speak to long-run effects, or support a claim
that the promise "does not work" if the result is null.
### 6.7 What would answer the question
Four routes, none free. **More stores:** the break-even needs 5,670 stores per arm
(`0.0065673 / 0.0010764^2`); Ravensbourne has 96, so this route is closed. **Session-level
assignment:** the break-even needs 470,193 sessions per arm, which the traffic can supply, but the
treatment is a store property; the route is statistically open and operationally closed.
**Switchback in time:** alternating the promise within each store, with washout, makes store-weeks
the independent units and absorbs between-store variance, but the calculation must be redone under
carryover and autocorrelation (see [`05_switchback-pilots.md`](05_switchback-pilots.md)). **Matched pairs or stepped
rollout:** these reduce between-store variance without changing the unit count by a factor of ten,
so they help at the margin, not as a rescue.

If the operational case is strong and the expected lift is above break-even, a staged rollout with a
measurement plan may be rational. That is a commercial decision made with the MDE in hand, not a
statistical failure. The failure is running the 48-store pilot and calling the null a verdict.

The memo for this design is the model paragraph in the Exercise 8 answer, section 13.
## 7. Code, query, or calculation
One block reproduces sections 6.2 to 6.4 so the parameters can be changed. It uses the variance of a
cluster mean for the proportion and the mean, and the delta method for the ratio.

```python
import math
z = 1.959964 + 0.841621                 # 80% power, two-sided 5%

def mde_prop(p, units, rho=0.0, m=1):
    v = p * (1 - p) * (rho + (1 - rho) / m)
    return z * math.sqrt(2 * v / units)

def mde_mean(sd, units, rho=0.0, m=1):
    v = sd ** 2 * (rho + (1 - rho) / m)
    return z * math.sqrt(2 * v / units)

def mde_ratio(sy, sx, r, cov, k, xbar):
    v = (sy**2 + r**2 * sx**2 - 2 * r * cov) / (k * xbar**2)
    return z * math.sqrt(2 * v)

print("prop naive  ", mde_prop(0.036, 864000) * 100)
print("prop cluster", mde_prop(0.036, 48, 0.012, 18000) * 100)
print("mean cluster", mde_mean(54.0, 48, 0.02, 1200))
print("ratio       ", mde_ratio(96000, 2400, 41.2, 0.85*96000*2400, 48, 18000))
print("ratio no cov", mde_ratio(96000, 2400, 41.2, 0.0, 48, 18000))
```

Expected output: `prop naive` 0.07940; `prop cluster` 1.16969; `mean cluster` 4.4556; `ratio` 1.6979;
`ratio no cov` 4.3780. The first two are in percentage points; the last three are in pounds.

Three checks before trusting any such block. First, the cluster-to-naive ratio must equal
`sqrt(DEFF)`: 1.1697 / 0.0794 = 14.73 = `sqrt(216.99)`. Second, the 95% interval half-width must be
about 70% of the MDE (`1.96 / 2.8016 = 0.6996`). Third, the covariance must move the ratio MDE in the
direction the data imply, the MDE must fall as one over the square root of units (doubling stores
divides it by `sqrt(2)`), and every MDE must be converted to a relative effect before comparison.

> **Note:** The block is a calculator, not a design. It takes the variance structure as given. If
> the ICC, the cluster-size distribution, or the ratio covariance is wrong, the block returns a
> precise answer to the wrong question.
## 8. Data, standards, and regulatory context
### 8.1 What the data must contain before the calculation can be made
A power calculation is a data product. It consumes a small number of quantities, and each has a
provenance requirement.

| Field | Why the calculation depends on it | What goes wrong without it |
|---|---|---|
| Baseline rate or mean | Sets the variance and the relative scale | MDE wrong by the ratio of the true SD to the assumed one |
| Variance at the analysis unit | The numerator of the standard error | Precision overstated, often by the design effect |
| ICC with provenance | The design effect | MDE wrong by the square root of a factor that can exceed 100 |
| Cluster size mean and CV | The design effect and the effective-sample ceiling | Understates variance when cluster sizes are unequal |
| Clusters per arm | The unit count that matters | Row count used instead, and the p-value is meaningless |
| Allocation ratio | Precision at fixed total | MDE computed for 50/50 when the design is 70/30 |
| Analysis unit and degrees of freedom | The multiplier | z used where t is required, and the MDE is too small |
| Compliance and attrition | Dilution of the delivered effect | Power overstated by `1/c^2` and `1/(1-a)^2` |
| Metric numerator and denominator | Ratio variance, and whether the denominator is post-treatment | Standard error wrong, or the metric measures selection |
| Break-even inputs | The comparison that gives the MDE meaning | MDE compared with zero, and a loss is rolled out |
| Guardrail base rate and margin | Guardrail power | A safety claim the design cannot support |
| Calendar volume and seasonality | Conversion of units to weeks | The window crosses a peak and the baseline moves |

The ICC and the cluster-size distribution are the two fields most often missing. A team that has the
row count but not the cluster count has the wrong number in the most important place.
### 8.2 Standards and reporting guidance
| Instrument | What it requires | Status for a commercial pilot | Grade |
|---|---|---|---|
| CONSORT 2010, item 7a | Primary outcome, all quantities used, expected control result, the difference not to be missed, and allowance for attrition | Reporting standard for randomised trials; not binding on a retailer, but the minimum field list | A |
| CONSORT 2025, item 16a | All assumptions supporting the sample size, with the DELTA² items and a sensitivity analysis | Current version replacing item 7a; explicitly discourages post-hoc power | A |
| DELTA² guidance (Cook et al., 2018) | How to specify the target difference by outcome type, and how to report the calculation | Guidance, not regulation; closest thing to a standard for the target effect | A |
| Company pre-registration template | Design, metric, MDE, decision rule, analysis | Internal governance artefact, not an external standard | None |

The standards are written for clinical trials and none of them binds a supply chain pilot. Their
value is that they encode the fields a reviewer needs. A pilot that adopted the CONSORT 2025 item 16a
checklist would eliminate most of the failures in section 10, and the exercise would take an hour.

> **Note:** CONSORT does not use the phrase "minimum detectable effect". It frames the same idea as
> the target difference, which must be justified prospectively rather than reverse-engineered from
> the sample. That framing is stricter than MDE reporting, and it is the better habit.
### 8.3 What is a convention rather than a requirement
- **Eighty percent power and a two-sided 5% test.** Conventions with no theoretical privilege.
  Ninety percent is common in high-stakes decisions, 70% is sometimes defensible for screening, and
  one-sided is defensible for a non-inferiority guardrail but suspect for a superiority claim.
- **Fifty-fifty allocation.** Optimal for precision at a fixed total; 70/30 can be right when
  exposure must be limited, at a computable cost.
- **Reporting an MDE at all.** Not required by any regulator for a commercial pilot, but required by
  anyone who wants the pilot to be interpretable after a null. "The pilot was powered to detect X"
  is often a claim about a calculation that was never done; ask for the document and the date.

An analyst who can separate a requirement from a convention can challenge the convention. An analyst
who cannot will treat 80% power as a physical constant and 5% as a law of nature.
## 9. Industry practice and case evidence
### 9.1 Where the MDE idea comes from
| Contribution | Source | What it established | Grade |
|---|---|---|---|
| Minimum detectable effects as a reporting device | Bloom (1995), "Minimum Detectable Effects," *Evaluation Review* | MDE as the smallest true impact an experiment has a good chance of detecting, computed for continuous and binary outcomes | A |
| Power and MDE for causal inference, including cluster designs | Djimeu and Houndolo (2016), *Journal of Development Effectiveness* | Explicit formulas for sample size and MDE, with rules of thumb and a calculator | A |
| Absence of evidence is not evidence of absence | Altman and Bland (1995), *BMJ*; Alderson (2004), *BMJ* | Non-significance does not establish the null, especially at low power | A |
| Power analysis as a prerequisite for interpretable effects | Brysbaert (2019), tutorial with reference tables | Power depends on effect, alpha, and N; underpowered studies produce inflated effect estimates and poor replicability | A |
| Borderline p-values usually reflect underpowering | Hackshaw and Kirkwood (2011), *BMJ* | Confidence intervals are more informative than the p-value, and "no effect" is usually the wrong conclusion | A |

The idea is old and settled. What is not settled is practice. The gap between the arithmetic in these
sources and the arithmetic in a typical pilot readout is the subject of the rest of this section.
### 9.2 Design effects and ICC: what the clinical literature established
| Finding | Source | Detail | Grade |
|---|---|---|---|
| Inflate an individually randomised sample by the design effect | Rutterford, Copas and Eldridge (2015), *International Journal of Epidemiology* | `DE = 1 + (n - 1) rho`; formulas for continuous, binary, and other outcomes; extensions for variable cluster sizes, attrition, and non-compliance | A |
| Cluster-size variation matters above a modest threshold | Same | A coefficient of variation of cluster size above about 0.23 warrants an adjustment to the design effect | A |
| The ICC is a variance ratio, not a nuisance parameter | Kerry and Bland, *BMJ* | `rho = s_between^2 / (s_between^2 + s_within^2)`; worked example with `rho = 0.019` and `m = 50` giving `DE` of about 1.93 | A |
| ICCs are metric- and setting-specific | Gao, Earnest, Matchar, Campbell and Machin (2015), *Contemporary Clinical Trials* | Primary-care ICCs typically 0.01 to 0.05, median around 0.01; sensitivity analysis across a range is advised | A |
| Optimal cluster size depends on the ICC and the cost ratio | van Breukelen and Candel (2012), *Journal of Clinical Epidemiology* | Optimal cluster sizes typically in the range 7 to 70; cluster-size variation compensated by about 10% more clusters | A |
| Sample-size methods for cluster trials are mature | Leyrat, Eldridge, Taljaard and Hemming (2024), *Journal of Epidemiology and Population Health* | Practical guidance extending the design effect to cross-over, baseline-measurement, and stepped-wedge designs | A |

The transferable lesson for supply chain work is not the clinical setting. It is that **the ICC is a
measured property of a metric in a population, and a borrowed ICC is an assumption**. The same
conversion metric can have an ICC of 0.005 across stores in one market and 0.04 across stores in
another, because the between-store variation depends on assortment, demographics, and competition.
The design effect moves by a factor of eight, and the MDE by a factor of 2.8, on a quantity that is
often copied from a blog post.
### 9.3 Few clusters: the power cost of small k
| Finding | Source | Detail | Grade |
|---|---|---|---|
| Standard corrections lose substantial power below about 20 clusters | Leyrat, Morgan, Leurent and Kahan (2018), *International Journal of Epidemiology* | Across 12 analysis approaches with 40 or fewer clusters, small-sample corrections maintained Type I error but power was under 50% with fewer than 20 clusters, and no method reached 80% | A |
| Likelihood-ratio tests are anti-conservative with few clusters | Nugent and Kleinman (2021), *BMC Medical Research Methodology* | With 10 to 100 clusters, LRT was anti-conservative when the ICC was large and clusters were few; between-within or Satterthwaite degrees of freedom maintained nominal error | A |
| Denominator degrees of freedom decide the error rate | Li and Redden (2015), *BMC Medical Research Methodology* | Between-within degrees of freedom preserved nominal Type I error at 10 clusters and were robust to unequal sizes; residual and containment defaults inflated error below 30 clusters | A |
| Covariate adjustment does not rescue very few clusters | Qiu, Cook and Bobb (2023), simulation study | With 10 to 20 clusters and at least two covariates, between-within degrees of freedom performed near nominally; no method worked uniformly with five or more covariates | A |

The practical implication is stark. A cluster design with 8 to 12 clusters per arm is not a
small version of a well-powered design. It is a design whose variance is estimated from a handful of
values, whose test statistic does not follow its nominal distribution, and whose correction for that
fact removes much of the power that remained. The MDE computed with a z multiplier is optimistic; the
MDE computed with a t multiplier and a cluster-robust variance is closer, and still optimistic
because the variance estimate itself is noisy.
### 9.4 Ratio metrics: the delta method
| Finding | Source | Detail | Grade |
|---|---|---|---|
| Ratio metrics require a first-order Taylor approximation | Deng, Knoblich and Lu (2018), KDD | Delta-method intervals for ratio and percent-change metrics, with a bias correction and an extension to cluster randomisation | C |
| The covariance term is the difference between a ratio and a mean | Same | The variance of a ratio depends on the covariance of numerator and denominator at the randomisation unit | C |
| Sizing for ratio metrics is a distinct calculation | Expedia Group Tech, "How to Size for Online Experiments with Ratio Metrics," Medium | Ratio variance formula and a sample-size function; a simulation reported about 83.4% empirical power against 80% nominal | D, undated, retrieved 2026 |

The vendor and engineering-blog sources are graded D because they are not peer-reviewed and carry no
date in the retrieved version. Their formulas agree with the peer-reviewed delta-method literature,
which is why they are usable as a cross-check rather than as authority. The important point is not
the formula's provenance. It is that **a ratio metric is not a mean, and a team that sizes it as a
mean will be wrong in a direction it cannot predict without the covariance**.
### 9.5 What audits of practice show
| Finding | Source | Detail | Grade |
|---|---|---|---|
| Sample-size reporting is frequently incomplete | Charles, Giraudeau, Dechartres, Baron and Ravaud (2009), *BMJ* | Review of 215 trials: 5% reported no sample-size calculation, 43% omitted required parameters, 30% of recalculations differed by more than 10%, and only 34% were fully adequate | A |
| Non-significant trials are rarely interpreted against their intervals | Gewandter et al. (2017), *BMJ Open* | Of 247 trials, 85 of 99 with non-significant primary results reported intervals but only 17 interpreted them; of 22 compatible with a clinically meaningful effect, only 4 acknowledged it | A |
| Most online experiments do not improve the primary metric | Kohavi and Chen (2024), KDD, "False Positives in A/B Tests" | Success rates of about 10% to 20% are most common; the paper argues that a 5% threshold at scale produces many false positives and proposes sequential testing | C |

The Kohavi and Chen figure is a company-scale observation about online product experiments, not a
supply chain measurement, and it is graded C for that reason. It is still the best public calibration
of the base rate: if most experiments fail to move the metric, a pilot that reports a null is
reporting the common case, and the question is whether the design could have distinguished the null
from a useful effect. Usually it could not.
### 9.6 What the public record does not contain
We found no public audit of power calculations in commercial supply chain pilots. Retailers,
carriers, and distributors do not publish their MDEs, their ICCs, or their pre-registrations. The
evidence in this section comes from clinical trials, online experimentation, and methodological
simulation, and it transfers because the arithmetic is the same. What does not transfer is the
governance: clinical trials have registries, reporting checklists, and regulators. A store pilot has
a steering committee.

That asymmetry is the practical problem. The methods are settled and the enforcement is absent. An
analyst who wants the pilot to be interpretable has to supply the governance themselves: the written
calculation, the signed decision rule, the MDE paragraph, and the limit sentence.
## 10. Failure modes and red flags
| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Post-hoc power reported | "Observed power was 0.34, so the study was underpowered" | Power is recomputed from the observed effect, which is a monotone function of the p-value | A null is explained away with a number that adds no information | Report the prospective MDE and the interval instead |
| MDE never stated | The readout has a p-value and no detectable-effect sentence | The design was never documented | A null is read as no effect | Require the MDE paragraph in the pre-registration |
| Relative MDE compared with absolute break-even | "We can detect 2%, and the break-even is 0.1 percentage points" | The numbers are in different units, so their ratio is undefined | A unit-mismatch error can reverse the verdict | Convert both to percentage points at the stated baseline |
| Power computed at row level for a cluster treatment | Degrees of freedom equal the row count; the arm count is not mentioned | Within-cluster correlation ignored | Interval too narrow by the square root of the design effect | Compute the design effect before the sample size |
| ICC assumed zero | "We have 1.7 million sessions, so precision is not a problem" | The row count is treated as independent evidence | Overconfidence by a factor of 10 to 15 | Estimate the ICC from pre-period data |
| ICC borrowed with no provenance | "Industry standard ICC is 0.05" on a slide with no citation | ICC is metric-, category-, and geography-specific | Power wrong by a factor of two or more | Estimate locally, or report a range |
| Cluster-size variation ignored | Mean cluster size reported, distribution absent | The design effect understates variance when sizes are skewed | MDE too small; power overstated | Report the CV and adjust when it exceeds about 0.23 |
| Guardrail asserted without a margin | "No significant degradation" from a 0.4% baseline | A superiority test at 0.05 cannot rule out a meaningful increase | Safety claim the data cannot support | Pre-specify a non-inferiority margin and power it |
| Four guardrails tested at 0.05 | Four p-values, no adjusted threshold | Family-wise error is 18.5% under the null | A spurious degradation blocks a good launch, or a real one is lost in noise | Pre-specify one guardrail family and correct, or label as monitored |
| Ratio metric sized as a mean | The variance uses the SD of the numerator only | The denominator is random and correlated with the numerator | MDE wrong in an unpredictable direction | Use the delta method with the covariance |
| Ratio denominator is post-treatment | Cost per order falls because orders rose | The denominator is an outcome, not a fixed exposure | The metric measures selection, not efficiency | Use a fixed denominator, or report volume and unit cost separately |
| Few clusters with a z multiplier | 10 clusters per arm, MDE computed with 1.96 and 0.84 | The variance is estimated from few values and the test is not normal | Type I error inflated, power overstated | Use a t multiplier with between-within df, or randomisation inference |
| Compliance and attrition omitted | The power calculation assumes every assigned store is analysed | ITT dilution and missing data reduce the delivered effect | Power overstated by `1/c^2` and `1/(1-a)^2` | Inflate the sample, and report the delivery rate |
| MDE compared with zero | "The pilot was powered to detect a 0.5-point lift" with no break-even | Any detectable effect is treated as decision-relevant | A significant but unprofitable change is launched | Compute the break-even and compare |

Two of these survive review most easily.

**The relative-versus-absolute comparison.** It looks like a units detail and it can reverse the
decision. A team reports "we can detect a 2% change" and finance reports "the break-even is 0.1
percentage points". On a 3.6% baseline, 2% relative is 0.072 points, which is below the break-even,
yet the relative number looks twenty times larger. The habit that prevents the error is simple:
always write the baseline next to the relative number.

**The post-hoc power sentence.** It is offered as a kindness, to explain why the pilot failed. It
does the opposite: it tells the room that the team knows the study was underpowered and ran it
anyway. The prospective MDE says the same thing before the data, when it can still change the
design.
## 11. Assumptions and limits
| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The baseline is stable over the test window | The MDE is computed at a fixed baseline rate or mean | Seasonality, a promotion, a competitor entry, a price change | Plot the metric over the same weeks in prior years; check for a trend |
| The variance at the analysis unit is known | The standard error is the square root of variance over units | A new metric, a new population, a truncated or winsorised series | Estimate from pre-period data; compare with a comparable period |
| The ICC is stable and correctly estimated | The design effect is a function of it | Different metric, category, region, or season | Estimate in pre- and test periods; report a range and a sensitivity table |
| Clusters are exchangeable within arm after blocking | The cluster means must be comparable | Voluntary participation, selection on the outcome, different formats | Balance table on pre-period covariates and outcomes |
| Cluster sizes are equal, or the CV is small | `1 + (m-1) rho` assumes a common `m` | A few very large stores or lanes | Compute the CV; adjust when it exceeds about 0.23 |
| There is no interference between units | The variance assumes independent units | Shared inventory, shared fleet, shared staff, spillover across geos | Shared-resource map; control-trend check; exposure mapping |
| The effect is constant over the window | The MDE is a single effect, not an average of a decay | Novelty effects, stock-up, learning, competitor response | Plot the effect by week; pre-specify a washout |
| The break-even inputs are stable | The MDE is judged against it | Cost changes, price changes, a new contract during the test | Recompute at readout; state the inputs and their dates |
| The pilot population represents the rollout population | The effect must generalise | Pilot stores are volunteered, better staffed, better located | Compare pilot and non-pilot stores on pre-period metrics |
| The ratio denominator is not affected by treatment | The metric must measure unit cost, not selection | A treatment that changes orders changes cost per order mechanically | Use a fixed denominator, or report volume and unit cost separately |
| The analysis uses one look at a fixed sample | The alpha and power are calibrated for one test | Daily peeking, early stopping, adaptive allocation | Pre-register the analysis; see Topic 08 for valid sequential methods |

**What the arithmetic cannot do.** It cannot tell you whether the mechanism is right, whether the
pilot population generalises, whether the decision will actually be made on the result, or whether
the metric measures the behaviour you care about. Those are design and governance questions. The
arithmetic only tells you what the design can detect, and therefore what a null will mean.

**Three hard limits.** The number of independent units is the first. A network with 96 stores
supports at most 96 independent observations of a store-level treatment, however many sessions flow
through them. The cost of precision is the second: MDE falls with the square root of the sample, so
each halving costs four times the units, and the units may not exist. The third is the decision: a
design that can detect an effect nobody will act on is as useless as one that cannot detect the
effect that matters.

**One limit that is often denied.** The power calculation is only as good as its variance inputs,
and the variance inputs are estimates. A single-point MDE presented without a sensitivity range
claims a precision that the inputs do not support. The correct presentation is a small table: MDE at
low, central, and high ICC; at the observed and a 20% higher baseline; at full and 80% compliance.
## 12. Exercises
### Exercise 1 — The four quantities (calculation, 25 minutes)
A two-arm test has alpha 0.05 two-sided, 80% power, a baseline conversion of 10%, and a target
absolute effect of 2 percentage points. Compute the sample per arm. Then hold the sample at that
value and compute the MDE if the true baseline were 5% instead of 10%.

**Deliverable:** both calculations with the formula, the substitution, and the units.
**Acceptance criteria:** the variance `p(1-p)` is computed at both baselines; the MDE is reported in
percentage points and relative terms; the direction of the change is explained.
**Estimated time:** 25 minutes.
### Exercise 2 — Proportion at a small effect (calculation, 30 minutes)
A pilot has 864,000 sessions per arm, a baseline conversion of 3.6%, and a target effect of 0.15
percentage points. Compute the required sample per arm at 80% power, two-sided 5%, ignoring
clustering. State the multiple of the available sample.

**Deliverable:** the required n, the available n, and the ratio.
**Acceptance criteria:** the unpooled and pooled variance forms are both shown and the difference is
explained; the answer is rounded up to a whole session.
**Estimated time:** 30 minutes.
### Exercise 3 — Continuous outcome (calculation, 25 minutes)
A delivery-cost pilot measures cost per drop. The SD is GBP 18.00 and the business needs to detect a
GBP 1.50 reduction. Compute the drops per arm at 80% power. Then compute the MDE at 2,258 drops per
arm.

**Deliverable:** the sample size and the MDE, both with arithmetic.
**Acceptance criteria:** the formula `2 (z sum)^2 sigma^2 / delta^2` is used; the MDE is reported in
pounds and as a percentage of a stated baseline.
**Estimated time:** 25 minutes.
### Exercise 4 — Ratio metric (calculation, 40 minutes)
A lane-level pilot measures cost per shipment. Across lanes, the SD of total cost is GBP 40,000, the
SD of shipments is 900, the mean shipments per lane is 6,000, the ratio is GBP 7.20, the correlation
between cost and shipments is 0.70, and there are 60 lanes per arm. Compute the delta-method SE of
the difference and the MDE. Then recompute with the covariance set to zero and state the direction of
the error.

**Deliverable:** both SEs, both MDEs, and a one-paragraph explanation.
**Acceptance criteria:** the covariance is computed from the correlation and the two SDs; the
denominator is `k` times the squared mean of shipments; the direction of the bias is stated.
**Estimated time:** 40 minutes.
### Exercise 5 — Design effect and clusters (calculation, 35 minutes)
A store-level test has 48 stores per arm, 18,000 sessions per store, a baseline conversion of 3.6%,
and an ICC of 0.012. Compute the design effect, the effective sample per arm, the SE of the
difference, and the MDE. Then compute the stores per arm needed to detect a 0.5 percentage point
change.

**Deliverable:** five quantities with arithmetic.
**Acceptance criteria:** the design effect and the effective sample agree with the direct cluster
variance; the required store count is rounded up; the answer states that the ceiling is `k/rho`.
**Estimated time:** 35 minutes.
### Exercise 6 — Few clusters (application, 35 minutes)
A pilot has 12 stores per arm, a baseline of 92%, 5,000 orders per store, and an ICC of 0.04. Compute
the MDE with a z multiplier and with a t multiplier using 10 degrees of freedom. State the ratio.
Then compute the smallest attainable two-sided p-value from a randomisation test with 12 clusters
per arm.

**Deliverable:** the two MDEs, the ratio, and the p-value floor.
**Acceptance criteria:** the t values are stated with their degrees of freedom; the randomisation
floor uses the number of distinct assignments; the conclusion names the design that would be needed.
**Estimated time:** 35 minutes.
### Exercise 7 — Guardrail as non-inferiority (design and calculation, 40 minutes)
A guardrail has a baseline of 97.8%, an ICC of 0.03, 1,200 orders per store, 48 stores per arm, and a
business tolerance of 0.5 percentage points. Compute the detectable margin at 48 stores. Compute the
stores per arm needed to rule out the 0.5-point margin at one-sided 5% and 80% power.

**Deliverable:** the detectable margin, the required stores, and a reporting sentence.
**Acceptance criteria:** the one-sided multiplier 2.4865 is used and justified; the margin is
distinguished from a superiority effect; the sentence says what the pilot can and cannot exclude.
**Estimated time:** 40 minutes.
### Exercise 8 — Write the MDE paragraph (design, 30 minutes)
Using the Ravensbourne numbers in section 6, write the nine-field MDE paragraph for a steering
committee. Keep it under 200 words.

**Deliverable:** the paragraph with every field present.
**Acceptance criteria:** design, metric, variance, alpha, power, MDE, break-even, adjustments, and
limit all appear; the MDE is in both absolute and relative units; the limit is one sentence.
**Estimated time:** 30 minutes.
### Exercise 9 — Critique a power claim (critique, 40 minutes)
A team writes: "The pilot is powered at 80% to detect a 2% lift. We randomised 30 stores per arm and
will analyse 900,000 orders. The guardrails showed no significant change." Identify every missing or
wrong element and rewrite the claim.

**Deliverable:** a numbered list of defects and a corrected three-sentence claim.
**Acceptance criteria:** at least six defects; each names the mechanism, not just the missing field;
the rewrite states the unit, the MDE, the break-even condition, and the guardrail limit.
**Estimated time:** 40 minutes.
### Exercise 10 — The impossible pilot (design and critique, 50 minutes)
A commercial director asks for a six-week pilot in 10 stores to prove that a new packing process
raises throughput by 3%. Each store handles about 4,000 orders per week at a mean throughput of
1,000 units per store per week; the throughput SD is 12% of the mean, and the store-level ICC is
0.05. Write the memo.

**Deliverable:** a one-page memo with the arithmetic, the verdict, two alternative designs, and the
decision you recommend if no design is feasible.
**Acceptance criteria:** the MDE is computed at the requested design; the required stores are
computed; at least one alternative is a time-based design; the recommendation separates the
feasibility question from the effect question.
**Estimated time:** 50 minutes.
## 13. Self-check answers
### Exercise 1
```text
   p1(1-p1) + p2(1-p2) = 0.0900 + 0.1056 = 0.1956
   n per arm = 7.84896 x 0.1956 / 0.02^2 = 3,839
   at p = 0.05: MDE = 2.8016 x sqrt(0.1126 / 3,839) = 1.517 pp
```

The variance of a proportion is smaller near 5%, so the same sample detects 1.52 points instead of
2.00. Relative to baseline the MDE worsens from 20.0% to 30.3%, which is why both framings are
reported.
### Exercise 2
```text
   p1(1-p1) + p2(1-p2) = 0.034704 + 0.036094 = 0.070798
   n per arm = 7.84896 x 0.070798 / 0.0015^2 = 246,973
```

The pooled form, `2 x 0.03675 x 0.96325 = 0.070798`, returns the same number. The available 864,000
per arm is 3.50 times the requirement, so the session-level design is powered for 0.15 points. The
problem is not volume; the treatment cannot be delivered at the session level.
### Exercise 3
```text
   n per arm = 2 x 7.84896 x 18.00^2 / 1.50^2 = 5,080.3 -> 5,081
   at 2,258: MDE = 2.8016 x sqrt(2 x 324 / 2,258) = GBP 1.5009
```

The MDE reproduces the design target, as inverting the same identity must. Against a baseline cost
per drop of GBP 200, the relative MDE is 0.75%.
### Exercise 4
```text
   sigma_XY = 0.70 x 40,000 x 900 = 25,200,000;  k x X_bar^2 = 2,160,000,000
   numerator = 1,600,000,000 + 41,990,400 - 362,880,000 = 1,279,110,400
   var(R) = 0.592181;  SE(diff) = sqrt(2 x 0.592181) = 1.08828
   MDE = 2.8016 x 1.08828 = GBP 3.049   (42.3% of GBP 7.20)
```

With the covariance set to zero the MDE is GBP 3.455 (48.0%), 13.3% too high. The direction is not
general: a negative covariance would make the naive MDE too small. Estimate the covariance.
### Exercise 5
```text
   var(store mean) = 0.034704 x [0.012 + 0.988/18,000] = 0.000418353
   SE(diff) = sqrt(2 x 0.000418353 / 48) = 0.0041751
   MDE = 2.8016 x 0.0041751 = 1.1697 pp
   DEFF = 216.99;  effective n = 3,982;  ceiling = 48/0.012 = 4,000
   stores for 0.5 pp = 0.0065673 / 0.005^2 = 262.7 -> 263 per arm
```

The effective sample is within 0.5% of the ceiling, so adding sessions to existing stores barely
moves the MDE. Only more stores do.
### Exercise 6
```text
   var(store mean) = 0.92 x 0.08 x [0.04 + 0.96/5,000] = 0.00295813
   SE(diff) = sqrt(2 x 0.00295813 / 12) = 0.022204
   z MDE = 2.8016 x 0.022204 = 6.22 pp
   t MDE = (2.228 + 0.879) x 0.022204 = 6.90 pp   (10.9% wider)
```

The randomisation floor is `1/C(24,12) = 3.7 x 10^-7`, not the binding constraint. The variance
estimate from 12 clusters is. A 12-cluster arm is a case series with a test attached.
### Exercise 7
```text
   var(store mean) = 0.978 x 0.022 x [0.03 + 0.97/1,200] = 0.000662884
   SE(diff) = sqrt(2 x 0.000662884 / 48) = 0.0052555
   multiplier (one-sided 5%, 80%) = 2.4865
   detectable margin = 2.4865 x 0.0052555 = 1.307 pp
   stores for 0.5 pp = 2 x 2.4865^2 x 0.021516 x 0.030808333 / 0.005^2 = 328
```

Report: "The pilot can exclude a degradation larger than 1.31 points; it cannot exclude the 0.5-point
degradation the business tolerates, so the guardrail is monitored, not tested."
### Exercise 8
> The design assigns 96 stores, 48 per arm, over 12 weeks, with about 18,000 sessions per store.
> The primary metric is conversion, baseline 3.60%, store-level ICC 0.012. At two-sided 5% alpha and
> 80% power the MDE is 1.17 percentage points, or 32.5% relative. Break-even is 0.108 points, so the
> MDE is 10.9 times the effect that pays for the promise. Adjusting for 90% compliance would raise
> the sample by 23%, but the pilot is not powered before that adjustment. The pilot can exclude a
> gain larger than 1.17 points; it cannot detect the effect that matters, and a null is not evidence
> that the promise does not work.

A weak answer reports "80% power to detect 1.17 points" without the break-even, leaving the reader
unable to judge whether 1.17 is good or bad.
### Exercise 9
Defects, in the order a reviewer should raise them:

1. "2% lift" is relative with no baseline, so it cannot be converted to points.
2. Assignment and analysis units are not stated; 900,000 orders is a row count, not independent n.
3. No ICC, cluster size, or design effect, so the power claim cannot be reproduced.
4. The MDE is asserted without alpha, power, variance, or arithmetic.
5. Guardrails are significance-tested with no margin and no power statement.
6. "No significant change" is reported as safety.
7. No break-even, so the MDE is never compared with the decision.
8. No compliance or attrition adjustment, and no analysis-unit degrees of freedom.
9. The claim conflates a pilot result with a rollout effect.

Corrected claim:

> The design assigns 30 stores per arm, about 30,000 orders per store, store-level ICC 0.02. At 80%
> power and a two-sided 5% test the MDE is 1.4 points, or 39% relative to the 3.6% baseline. The
> break-even is 0.11 points. The pilot cannot detect the effect that matters, and the guardrail
> excludes only a degradation larger than 1.5 points against a 0.5-point tolerance. It is a
> feasibility pilot, not a confirmatory test.
### Exercise 10
```text
   mean 1,000 units/store/week;  SD 120;  target 30 units;  m = 24,000;  5 stores/arm
   var(store mean) = 14,400 x [0.05 + 0.95/24,000] = 720.57
   SE(diff) = sqrt(2 x 720.57 / 5) = 16.977
   MDE = 2.8016 x 16.977 = 47.56 units = 4.76%
   stores for 30 units = 11,311.5 / 900 = 12.57 -> 13 per arm (26 total)
```

Five stores per arm detect 4.76%, not 3%: the effect is 1.59 times too coarse and the store count
2.6 times too small. The memo should say 26 stores would be powered, and that otherwise the options
are a switchback in the existing 10 stores, a matched-pair design with more sites, or a feasibility
pilot. Separate "can this be operated?" from "does it raise throughput by 3%?". The first is
answerable in 10 stores; the second is not.
## 14. Interview questions
**1. "What is statistical power?"**
Strong: the probability that a test rejects the null when the true effect equals the value used in the
calculation; it is a property of a design at an assumed effect, not of a completed study. It names
alpha, the effect, and the variance. Weak: "the chance the experiment works", which names none of the
three and implies a study either has power or does not.

**2. "What is the minimum detectable effect and why would you report it?"**
Strong: the smallest true effect the design detects at stated alpha and power; reporting it before
launch tells the reader what a null will mean and prevents the null from being read as no effect.
Weak: "the smallest effect we can measure", which confuses the MDE with the resolution of the
measurement and omits the power condition.

**3. "Why is the MDE usually larger than the effect the business cares about?"**
Strong: because the business cares about small relative changes, the variance is driven by
between-unit differences that do not shrink with more rows, and the available units are fixed by
operations; the square-root law makes each halving of the MDE cost four times the units. It gives a
concrete example where the ratio is ten or more. Weak: "because we don't have enough data", which
does not explain why more data would not fix it.

**4. "How do you power a cluster design?"**
Strong: compute the individual-level variance, apply `DEFF = 1 + (m - 1) rho`, convert to the
variance of a cluster mean, and solve for clusters per arm; report the ICC's provenance, the
cluster-size distribution, and the effective sample. Weak: multiplies the individual sample by the
design effect without stating the ICC, which is the same arithmetic with the important assumption
hidden.

**5. "What is the difference between the design effect and the effective sample size?"**
Strong: the design effect is a variance ratio; the effective sample is the row count divided by it,
with a ceiling of clusters divided by the ICC. A million rows in 48 stores are worth about 4,000
independent observations. Weak: uses the two terms interchangeably, or reports the row count as the
sample.

**6. "Why do 12 clusters per arm change the calculation?"**
Strong: the variance is estimated from 12 values, the test statistic does not follow its nominal
distribution, and corrections such as between-within degrees of freedom or randomisation inference
are required; power is typically below 50% even when Type I error is controlled. Weak: says "small
samples are noisy", which does not name the degrees of freedom, the correction, or the power cost.

**7. "How do you power a ratio metric such as revenue per session?"**
Strong: use the delta method with the variance of the numerator, the variance of the denominator, and
their covariance at the randomisation unit; state whether the denominator is fixed or
post-treatment. Weak: divides the SD of revenue by the square root of sessions, which treats the
ratio as a mean and ignores both the random denominator and the clustering.

**8. "What is wrong with post-hoc power?"**
Strong: it is computed from the observed effect, which is the quantity the test was estimating, so it
is a monotone transformation of the p-value and adds no information; the prospective MDE and the
confidence interval are the right reports. Weak: says "it is bad practice" without the mechanism, or
confuses post-hoc power with a prospective power calculation.

**9. "Why can't you power on the primary metric and hope on the guardrails?"**
Strong: a guardrail is a non-inferiority question with its own margin, base rate, and variance; low
base rates make it expensive, and several guardrails multiply the error rate. If it is not powered
for the margin, it is monitored, not tested. Weak: says "we will check the guardrails", which is a
monitoring plan presented as a safety guarantee.

**10. "What do you say when asked for a pilot that cannot answer the question?"**
Strong: show the MDE against the break-even, state the design that would answer and its cost, offer a
feasibility pilot that answers a narrower question, and put the limit in writing before launch. It
does not refuse to run anything; it refuses to overclaim. Weak: runs the pilot and reports the null,
or refuses without offering an alternative.
## 15. Key terms
- **Power** — probability of rejecting the null at a stated true effect.
- **Alpha** — probability of rejecting a true null; the significance level.
- **Beta** — probability of failing to reject a false null; power is 1 minus beta.
- **MDE** — minimum detectable effect at stated alpha and power.
- **Break-even effect** — effect at which incremental contribution equals incremental cost.
- **SESOI** — smallest effect size of interest, a decision input.
- **z-multiplier** — critical value plus power quantile; 2.80 at 80% power, two-sided 5%.
- **Half-width** — half the width of a confidence interval; about 0.70 times the MDE at 95%.
- **Design effect (DEFF)** — variance inflation from clustering, `1 + (m - 1) rho`.
- **ICC** — intra-cluster correlation; the share of variance between clusters.
- **Effective sample size** — row count divided by the design effect; ceiling `k / rho`.
- **Cluster size (m)** — analysis units per cluster.
- **Cluster-size CV** — SD of cluster sizes divided by the mean.
- **Between-within degrees of freedom** — `k - 2` correction for few clusters.
- **Randomisation inference** — inference from the assignment mechanism; p-value floor of one over
  the number of assignments.
- **Ratio metric** — a numerator divided by a random denominator.
- **Delta method** — first-order variance approximation for a ratio.
- **Covariance term** — the `-2 R Cov(Y, X)` term in the ratio variance.
- **Post-treatment denominator** — a denominator affected by the treatment.
- **Guardrail metric** — a metric that must not degrade beyond a margin.
- **Non-inferiority margin** — the largest tolerated degradation.
- **Compliance** — share of assigned units receiving the treatment as designed.
- **Dilution** — attenuation of the ITT effect by non-compliance.
- **Post-hoc power** — power recomputed from the observed effect; uninformative.
- **Conditional exaggeration** — upward bias among results that reach significance.
- **Simulation-based power** — power estimated by simulating the design.
## 16. Sources
**Grading convention.** A peer-reviewed or audited; B regulatory, official, or registered; C company
case, investor claim, preprint, or conference paper; D vendor or consultancy marketing; E press or
anecdote. Vendor material is dated where a date is visible; undated pages are marked as retrieved in
2026.

**Peer-reviewed methodology**

- Bloom, H. S. (1995). "Minimum Detectable Effects." *Evaluation Review*. doi:10.1177/0193841X9501900504. Grade A.
- Djimeu, E. W., and Houndolo, D.-G. (2016). "Power calculation for causal inference in social
  science: sample size and minimum detectable effect determination." *Journal of Development
  Effectiveness*. doi:10.1080/19439342.2016.1244555. Grade A.
- Rutterford, C., Copas, A., and Eldridge, S. (2015). "Methods for sample size determination in
  cluster randomized trials." *International Journal of Epidemiology*. PMC4521133. Grade A.
- Kerry, S., and Bland, J. M. "The intracluster correlation coefficient in cluster randomisation."
  *BMJ*. PMC1113123. Grade A. Year not stated in the retrieved record.
- Gao, F., Earnest, A., Matchar, D. B., Campbell, M. J., and Machin, D. (2015). "Sample size
  calculations for the design of cluster randomized trials: A summary of methodology."
  *Contemporary Clinical Trials*. Grade A.
- van Breukelen, G. J. P., and Candel, M. J. J. M. (2012). "Calculating sample sizes for cluster
  randomized trials: We can keep it simple and efficient!" *Journal of Clinical Epidemiology*.
  Grade A.
- Leyrat, C., Eldridge, S., Taljaard, M., and Hemming, K. (2024). "Practical considerations for
  sample size calculation for cluster randomized trials." *Journal of Epidemiology and Population
  Health*. doi:10.1016/j.jeph.2024.202198. Grade A.
- Leyrat, C., Morgan, K. E., Leurent, B., and Kahan, B. C. (2018). "Cluster randomized trials with a
  small number of clusters: which analyses should be used?" *International Journal of Epidemiology*,
  47(1), 321. Grade A.
- Nugent, J., and Kleinman, K. (2021). "Type I error control for cluster randomized trials under
  varying small sample structures." *BMC Medical Research Methodology*. PMC8019504. Grade A.
- Li, P., and Redden, D. T. (2015). "Comparing denominator degrees of freedom approximations for the
  generalized linear mixed model in analyzing binary outcome in small sample cluster-randomized
  trials." *BMC Medical Research Methodology*. PMC4458010. Grade A.
- Qiu, Cook, and Bobb (2023). Simulation study of tests for cluster-randomised trials with few
  clusters under covariate-adjusted mixed models. PMC10872819. Grade A. Journal detail not verified
  for this file.
- Gewandter, J. S., et al. (2017). "Interpretation of CIs in clinical trials with non-significant
  results: systematic review and recommendations." *BMJ Open*.
  doi:10.1136/bmjopen-2017-017288. Grade A.
- Hackshaw, A., and Kirkwood, A. (2011). "Interpreting and reporting clinical trials with results of
  borderline significance." *BMJ*. doi:10.1136/bmj.d3340. Grade A.
- Altman, D. G., and Bland, J. M. (1995). "Absence of evidence is not evidence of absence." *BMJ*,
  311, 485. Grade A.
- Charles, P., Giraudeau, B., Dechartres, A., Baron, G., and Ravaud, P. (2009). "Reporting of sample
  size calculation in randomised controlled trials: review." *BMJ*, 338, b1732. Grade A.
- Cook, J. A., et al. (2018). "DELTA2 guidance on choosing the target difference and undertaking and
  reporting the sample size calculation for a randomised controlled trial." *BMJ*, 363, k3750.
  Grade A.
- Brysbaert, M. (2019). "How Many Participants Do We Have to Include in Properly Powered
  Experiments? A Tutorial of Power Analysis with Reference Tables." PMC6640316. Grade A.

**Standards and official guidance**

- CONSORT 2010 Explanation and Elaboration, item 7a. PMC2844943. Grade A.
- Hopewell, S., et al. (2025). "CONSORT 2025 explanation and elaboration: updated guideline for
  reporting randomised trials." *BMJ*, 389, e081124. doi:10.1136/bmj-2024-081124. Grade A.
- Evidence for Action, "Guidance for Sample Size, Statistical Power, and Smallest Detectable Effect
  Size Calculations" (Matthay, 2020). Foundation guidance, not a regulator. Grade C.

**Company, conference, and unrefereed**

- Kohavi, R., and Chen, N. (2024). "False Positives in A/B Tests." KDD.
  doi:10.1145/3637528.3671631. Grade C.
- Deng, A., Knoblich, U., and Lu, Y. (2018). "Applying the Delta Method in Metric Analytics: A
  Practical Guide with Novel Ideas." KDD. Grade C.
- Expedia Group Tech, "How to Size for Online Experiments with Ratio Metrics," Medium. Grade D,
  undated, retrieved 2026.
- Optimizely, "Ratio metric," product documentation. Grade D, undated, retrieved 2026.
- Statsig, "Delta Method," product documentation. Grade D, undated, retrieved 2026.

**Note on citations.** Findings are reported as the cited works state them. The worked example is
constructed for teaching, and its arithmetic was checked against the code in section 7. Where a
bibliographic detail could not be verified, the entry says so rather than supplying a plausible year,
journal, or page range.
## 17. Links
- Module map and topic sequence: [`_index.md`](_index.md)
- The claim and estimand this topic powers: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- The unit whose independence the variance assumes: [`02_randomisation-units.md`](02_randomisation-units.md)
- Design effects and cluster randomisation, the design this topic sizes: [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md)
- Between-geo variance and pair-level MDEs: [`04_geo-experiments.md`](04_geo-experiments.md)
- Time as the unit when the cluster count is too small: [`05_switchback-pilots.md`](05_switchback-pilots.md)
- Interference, which changes the variance this topic assumes: [`07_interference-spillover.md`](07_interference-spillover.md)
- Peeking and early stopping, which change the error rate: [`08_sequential-testing.md`](08_sequential-testing.md)
- Observational alternatives when no design is available: [`09_mmm-incrementality.md`](09_mmm-incrementality.md)
- Appraising the evidence a vendor supplies: [`10_vendor-evidence-appraisal.md`](10_vendor-evidence-appraisal.md)
- Hypothesis tests and the variance an F statistic divides:
  [`../04_statistics_causal_inference/03_hypothesis-tests.md`](../04_statistics_causal_inference/03_hypothesis-tests.md)
- Resampling and simulation checks for the analytic formula:
  [`../04_statistics_causal_inference/02_probability-bootstrap.md`](../04_statistics_causal_inference/02_probability-bootstrap.md)
- Autocorrelation, which matters when a switchback replaces a cluster design:
  [`../04_statistics_causal_inference/05_autocorrelation-robust-errors.md`](../04_statistics_causal_inference/05_autocorrelation-robust-errors.md)
- The three-flow model that tells you which resources are shared:
  [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)
- Module survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Promotion measurement, a common source of ratio metrics and guardrails:
  [`../../../course/16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
- Inventory and service metrics, whose base rates decide guardrail power:
  [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
- Return to the part map: [`../_index.md`](../_index.md)
