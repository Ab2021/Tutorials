> Module 04 — Statistics and Causal Inference
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.01, 04.02

# Hypothesis Tests and Multiple Comparisons

---

## 1. Why this topic exists

A replenishment dashboard has 240 tiles, each a supplier–lane–metric comparison across lead time,
fill rate, forecast bias, expedite spend, and days of cover. The tiles are coloured by a p-value;
twelve are red this week. A category manager asks which three to act on, and nobody can answer,
because the dashboard has produced a number that looks like a verdict and is not one.

Under a global null — no real change anywhere — 240 independent tests at alpha 0.05 produce about
12 red tiles. That is not a malfunction. It is the design. The dashboard is a machine for
manufacturing apparent findings at a known rate, and it reports the findings without reporting the
rate.

Three failures recur, and this topic addresses each.

**One. The p-value is read as something it is not.** The three sentences heard most often are "the
p-value is the probability the null hypothesis is true", "it is the probability the result happened
by chance", and "it is the probability the result will replicate". None is the definition. The
first inverts a conditional probability. The second has no referent, because every result happened
by chance in some sense and none did in another. The third describes a different experiment. The
ASA statement on p-values lists these misreadings explicitly and is the closest thing the field has
to an agreed correction (Wasserstein and Lazar, 2016).

**Two. Statistical significance is confused with operational significance.** A promotion test
reduces expedite spend by EUR 18 per order with p = 0.004. The operational threshold, agreed with
finance, is EUR 25 per order. The result is statistically significant and operationally trivial.
Meanwhile inventory days of cover rise by 1.8 days with p = 0.049. The threshold is 2.0 days, so the
effect is below the line, but the interval may still contain a costly true value. A dashboard that
sorts on p sorts on the wrong quantity. The decision quantity is the effect, in its unit, with its
interval, against a threshold that someone with budget authority has agreed to.

**Three. Multiplicity is ignored, or corrected in the wrong place.** The dashboard runs 240
comparisons and reports the one that passed. A category team slices a national promotion by 40
stores, finds three winners, and rolls out to all 40. An analyst tests a one-sided hypothesis after
seeing which direction the data went. Each is the same error in a different costume: the reported
p-value is not calibrated for the number of looks that were taken, and the selection that produced
it is invisible in the output.

**What this topic does about it.** It defines the p-value precisely, separates the statistical
question from the operational one, and gives the arithmetic for the three standard corrections:
Bonferroni, Holm, and Benjamini–Hochberg. It then gives the design fix that no correction can
substitute for — a pre-specified family and analysis plan — and the different test you need when
the question is "is it no worse?" rather than "is it different?".

**Why the theory matters.** The conventions were built for confirmatory science and regulated
trials, not operational dashboards. Alpha 0.05 is a convention with a history, not a law. Bonferroni
is conservative by construction, and FDR answers a question about the reported set rather than about
any single tile. An analyst who knows why each convention exists can defend a choice.

**The claim.** A p-value is a calibrated filter, not a verdict. The unit of a decision is an effect
with an interval and a threshold. The unit of error control is a family declared before the data are
seen. Topic 01 fixed the estimand, Topic 02 attached an interval, and this topic turns the interval
into a decision rule. Topic 04 moves the logic into regression, Topic 05 repairs standard errors
under dependence, and Topic 10 asks how fragile a finding is.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** a p-value correctly, including the null model and the "at least as extreme" clause,
   and **name** the three things it is routinely said to be but is not.
2. **Distinguish** statistical from operational significance, and **convert** an effect and interval
   into a decision against a stated threshold in units, euros, or days of cover.
3. **Explain** why a one-sided test chosen after inspecting the data is not a one-sided test, and
   **compute** the effect of that choice on a reported p-value.
4. **Define** a comparison family for a given dashboard, and **compute** the family-wise error rate
   (FWER) and false discovery rate (FDR) consequences of testing that family uncorrected.
5. **Apply** Bonferroni, Holm, and Benjamini–Hochberg by hand to a set of p-values, and **state**
   which error rate each controls and under what dependence assumptions.
6. **Write** a pre-analysis plan for a metric change or pilot that fixes the primary metric, the
   family, the tails, and the decision threshold before the data arrive.
7. **Design and interpret** an equivalence or non-inferiority test, including the margin, and
   **explain** why a non-significant difference is not evidence of no difference.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold four pictures: a null world that manufactures winners at a fixed rate, a p-value as a position
in that world, a decision on two axes rather than one, and a family as the real unit of error control.

### 3.1 The null world is a machine that manufactures winners

Suppose you run 100 independent tests and there is no true effect anywhere. Set alpha to 0.05. Each
test has a 5% chance of crossing the line by construction. The 100 tests are not a search for truth;
they are 100 draws from a machine calibrated to produce five false winners.

```text
   ONE HUNDRED TESTS, NO TRUE EFFECT ANYWHERE, alpha = 0.05

   Each cell is one test.  . = p >= 0.05      X = p < 0.05
   Columns are dashboards; rows are metrics within a dashboard.

   dashboard:   1    2    3    4    5    6    7    8    9   10
              +----+----+----+----+----+----+----+----+----+----+
   metric   1 |  . |  . |  . |  . |  . |  . |  . |  . |  . |  . |
            2 |  . |  . |  . |  X |  . |  . |  . |  . |  . |  . |
            3 |  . |  . |  . |  . |  . |  . |  . |  . |  . |  . |
            4 |  . |  . |  . |  . |  . |  . |  . |  . |  X |  . |
            5 |  . |  . |  . |  . |  . |  . |  . |  . |  . |  . |
            6 |  . |  . |  . |  . |  . |  . |  . |  . |  . |  . |
            7 |  . |  . |  . |  . |  . |  . |  X |  . |  . |  . |
            8 |  . |  . |  . |  . |  . |  . |  . |  . |  . |  . |
            9 |  . |  . |  . |  . |  . |  . |  . |  . |  . |  . |
           10 |  . |  . |  . |  . |  X |  . |  . |  . |  . |  . |
              +----+----+----+----+----+----+----+----+----+----+
                            five X's in one hundred cells

   Expected number of X's          = m x alpha = 100 x 0.05 = 5
   Probability of at least one X   = 1 - (1 - alpha)^m
                                   = 1 - 0.95^100
                                   = 0.9941
```

The arithmetic is the argument: with 12 comparisons the probability of at least one false positive is
`0.4596`, with 20 it is `0.6415`, and with 100 it is `0.9941`. Reporting "the significant one" is reporting a draw from this machine.

### 3.2 A p-value is a position, not a probability about hypotheses

A p-value answers one question: if the null model and all the test's assumptions were true, what is
the probability of a test statistic at least as extreme as the one observed? It is a tail area of a
distribution that exists only under the null.

The p-value is the tail area to the right of the observed statistic, computed assuming the null is
true: not `P(null is true)`, but `P(data this extreme | null)`. Three consequences follow, and they are the three misreadings.

- **It is not the probability the null is true.** That would require a prior over hypotheses and a
  likelihood for the alternative. A p-value contains neither. `P(data | null)` and `P(null | data)`
  are different quantities, and confusing them is the prosecutor's fallacy applied to metrics.
- **It is not the probability the result occurred by chance.** The null model *is* a chance model.
  The observed result is one draw. The p-value is the probability of a draw at least this extreme,
  not the probability that chance produced this particular draw.
- **It is not the probability of replication.** Replication depends on power, on whether the effect
  is real, and on whether the next sample is exchangeable with this one. A p of 0.05 is compatible
  with a replication probability well below 0.5.

A fourth sentence is also common and also wrong: "p = 0.06 means there is a 6% chance the result is
false." The p-value is computed under the null, not under the observed result, and says nothing
about the base rate of true effects among the tests you run.

> **Note:** The p-value is valid only relative to the model that generated it. If the test assumes
> independent observations and the data are 40 weekly readings from one lane, the p-value is
> computed against the wrong null distribution. It is not "conservative" or "anti-conservative" in
> general; it is a number for a different experiment. Topic 05 treats this failure in detail.

### 3.3 Two axes, one decision

Statistical significance is a statement about compatibility with a null. Operational significance
is a statement about consequence. They are independent, and the four combinations call for
different actions.

| | Operationally trivial | Operationally material |
|---|---|---|
| **Statistically significant** | Do not act. Large samples make small effects significant. Report the effect and the threshold it failed. | Act, subject to cost and reversibility. This is the case the test was built for. |
| **Not statistically significant** | No action. The interval excludes material effects, or the question does not matter. | Inconclusive, not negative. The interval may still contain a costly effect. Collect more data, or decide on cost of error rather than on p. |

The bottom-right cell is the one that costs money. "Not significant" is routinely reported as "no
effect", and the report suppresses the interval that would show the effect is still plausible.
Equivalence testing, section 5.7, is the formal remedy when the decision is "no worse".

### 3.4 The family is the real unit of error control

A test does not have an error rate in isolation. It has an error rate relative to a set of
comparisons that someone might have reported. That set is the **family**.

Three families, three different corrections:

| Family definition | Example | Consequence of ignoring it |
|---|---|---|
| All tiles on the dashboard | 240 supplier–lane–metric comparisons | About 12 false reds per refresh under a global null |
| All pre-specified scorecard metrics | 12 metrics agreed at the start of the quarter | About 0.6 false positives expected; 46% chance of at least one |
| All pairwise comparisons within a group | 8 suppliers, 28 pairs | The best-looking pair is selected on noise |
| All subgroups examined | 40 stores, 4 regions, 3 channels | The winning slice is a selection artifact |

The practical rule: **declare the family before looking, and report the family size next to every
p-value.** A p-value without a family is a number without a denominator.

### 3.5 The garden of forking paths

Multiplicity is usually taught as the problem of running many tests. Gelman and Loken (2014) show
that it arises even when you run one test, because the analysis contains many choices that could
have gone the other way: which weeks to exclude, which outliers to trim, which covariate to adjust
for, which subgroup to report, how to define the metric. If those choices are made after seeing the
data, the reported p-value is selected from a set of analyses you did not enumerate. The
multiplicity is invisible because only one analysis appears in the output.

The reported p is one leaf of a tree with dozens of leaves, and if the path was chosen after seeing
the data, the p is not calibrated. The remedy is not "do not explore". Exploration is how hypotheses are generated. The remedy is to
separate exploration from confirmation: explore freely, then declare a new family and collect new
data, or label the finding as exploratory and do not attach a decision to it.

### 3.6 Equivalence flips the null

The standard test asks whether the data are compatible with no effect. The equivalence test asks
whether the data are compatible with an effect no larger than a margin. The null and alternative
swap places.

```text
   SUPERIORITY                  EQUIVALENCE
   H0: theta = 0                H0: |theta| >= delta
   H1: theta != 0               H1: |theta| < delta

   "is it different?"           "is it close enough to zero to call it no worse?"
   rejection = evidence of      rejection = evidence of absence within the margin
   a difference

   NON-INFERIORITY
   H0: theta <= -delta          H1: theta > -delta
   "is it not worse than the comparator by more than delta?"
```

This is the only statistical machinery in the topic that can support the sentence "no worse". A
non-significant superiority test cannot, because failure to reject is compatible with a large
effect and a small sample. Section 5.7 gives the mechanics; section 6.9 works an example.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A definition without one is the defect this
module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Null hypothesis (H0) | The precise statement tested, usually no difference or no effect, against which the data are compared | Stated loosely as "nothing is happening" | A statement about a parameter, fixed before analysis |
| Alternative hypothesis (H1) | The set of parameter values considered if H0 is false | Treated as "the effect is large" | A statement about a parameter; may be one- or two-sided |
| Test statistic | A function of the data whose distribution under H0 is known or approximated | Chosen after looking at which statistic gives the smallest p | Dimensionless or in the units of the estimate |
| p-value | Probability, under H0 and the test's assumptions, of a statistic at least as extreme as observed | Read as P(H0 is true), P(result by chance), or P(replication) | Probability in [0, 1], conditional on H0 |
| Significance level (alpha) | The pre-specified Type I error rate at which H0 is rejected | Treated as a property of the data rather than a decision rule | Probability in [0, 1], fixed before analysis |
| Type I error | Rejecting H0 when it is true | Called a "false positive" without saying false about what | Indicator, 0 or 1, per test |
| Type II error | Failing to reject H0 when H1 is true | Ignored in dashboards, where it is usually the larger problem | Indicator, 0 or 1, per test |
| Power | Probability of rejecting H0 when a specified H1 is true | Reported without stating the effect size at which it applies | Probability in [0, 1], at a stated effect |
| One-sided test | A test whose rejection region lies in one tail, with direction fixed before analysis | Chosen after seeing the sign of the effect | Direction stated in the analysis plan |
| Two-sided test | A test whose rejection region lies in both tails | Used by default without asking whether the opposite direction is possible | Two tails, alpha split across them |
| Effect size | The magnitude of the estimated quantity, in its natural unit | Reported as a p-value or as "significant" | Units of the outcome: days, EUR, units, pp |
| Minimum detectable effect (MDE) | The smallest true effect a design can detect with stated power at a stated alpha | Confused with the observed effect or with the operational threshold | Outcome units, at stated alpha and power |
| Statistical significance | A p-value below the pre-specified alpha | Reported as "important" or "proven" | Boolean at a stated alpha |
| Operational significance | An effect large enough to change a decision against a pre-agreed threshold | Asserted after the result is known | Boolean at a stated threshold in outcome units |
| Confidence interval (CI) | A procedure that captures the parameter at a stated rate over repeated samples | Read as a probability statement about this interval | Endpoints in outcome units, with coverage stated |
| Multiplicity | The inflation of error rates when many hypotheses are tested and some are selected for reporting | Treated as a problem only when tests are formally counted | Count m of hypotheses in the family |
| Family | The set of comparisons over which an error rate is controlled | Left undefined next to a reported p-value | Declared set of m hypotheses |
| Family-wise error rate (FWER) | Probability of at least one false rejection anywhere in the family | Confused with the per-test error rate | Probability in [0, 1] |
| False discovery rate (FDR) | Expected proportion of false rejections among the rejections made | Read as the probability that any particular rejection is false | Expected proportion in [0, 1] |
| False discovery proportion (FDP) | Realised proportion of false rejections among rejections, V/max(R, 1) | Confused with its expectation, the FDR | Proportion in [0, 1] |
| Bonferroni correction | Reject any hypothesis with p <= alpha/m | Applied to correlated tests without noting the loss of power | Threshold alpha/m |
| Holm procedure | Step-down correction: compare the k-th smallest p to alpha/(m - k + 1), stop at the first failure | Described as identical to Bonferroni | Threshold sequence alpha/(m - k + 1) |
| Benjamini–Hochberg (BH) procedure | Step-up correction: find the largest k with p(k) <= k alpha/m and reject the first k | Applied under arbitrary dependence without the Benjamini–Yekutieli correction | Threshold sequence k alpha/m |
| Step-up procedure | Starts at the largest p-value and moves toward the smallest, rejecting a block once a threshold is met | Confused with step-down | Order statistic index k |
| Step-down procedure | Starts at the smallest p-value and stops at the first failure | Confused with step-up | Order statistic index k |
| Adjusted p-value | The smallest family-wise or FDR level at which a given hypothesis would be rejected | Reported without naming which error rate it adjusts | Probability in [0, 1] |
| q-value | A BH-type adjusted p-value interpretable as an estimated FDR at the threshold | Called a p-value | Probability in [0, 1] |
| p-hacking | Trying analyses until a p-value crosses a threshold and reporting only the successful one | Denied because the analyst did not consciously "fish" | Number of unreported analyses |
| HARKing | Hypothesising after the results are known, then presenting the hypothesis as prior | Described as "framing" | Analysis-plan version identifier |
| Garden of forking paths | Invisible multiplicity created by data-dependent analysis choices, even with one reported test | Reduced to "multiple comparisons" and dismissed | Count of plausible analysis paths |
| Researcher degrees of freedom (RDF) | The set of choices in data collection and analysis that can affect the result | Treated as dishonesty rather than as a design property | Enumerated choices in the plan |
| Preregistration | Public or archived registration of hypotheses, design, and analysis before data collection or inspection | Confused with a grant application or a project charter | Timestamped registry record |
| Analysis plan (SAP, PAP) | A document fixing the primary estimand, family, tests, tails, thresholds, and handling of deviations | Written after the data arrive and backdated in spirit | Versioned document with a date |
| Selective reporting | Reporting a subset of analyses, outcomes, or subgroups, usually the favourable ones | Justified as "focus" | Reported analyses divided by conducted analyses |
| Winner's curse | The tendency of the selected best estimate to overstate the true effect | Applied only to auctions and not to metric rankings | Difference between selected and replicated effect |
| Look-elsewhere effect | Inflated significance when a signal could have appeared anywhere in a search space | Confused with a correction for a fixed family | Search-space volume |
| Equivalence margin (delta) | The largest difference regarded as operationally equivalent | Chosen after seeing the confidence interval | Outcome units, fixed before analysis |
| Non-inferiority margin | The largest acceptable worsening relative to a comparator | Set to whatever the data can demonstrate | Outcome units, fixed and justified before analysis |
| Two one-sided tests (TOST) | Two one-sided tests against -delta and +delta; equivalence is declared if both reject | Run as a single two-sided test at alpha | Two one-sided p-values; report the maximum |
| Superiority test | A test of H0: no difference against a two-sided or one-sided alternative | Used to support a "no worse" claim | p-value and interval |
| Alpha spending | Allocation of the total Type I error across interim looks | Ignored when a dashboard is refreshed weekly | Cumulative alpha by look |
| Gatekeeping | A testing procedure in which later hypotheses are tested only if earlier ones reject | Confused with a correction for all tests | Ordered hypothesis sequence |
| Primary metric | The single outcome that carries the decision and the error budget | One of twelve metrics reported with equal weight | Named in the analysis plan |
| Secondary or exploratory metric | An outcome reported for context, not used for the confirmatory claim | Used to rescue a failed primary result | Labelled as exploratory |
| Guardrail metric | A metric that must not worsen, tested for non-inferiority rather than superiority | Corrected as if it were a success metric | Non-inferiority margin in outcome units |

Three of these deserve expansion.

**FWER versus FDR.** FWER is the probability of one false rejection anywhere. FDR is the expected
share of false rejections among those you report. FWER is the right target when a single false
claim triggers an irreversible action: a supplier termination, a product recall, a safety
declaration. FDR is the right target when you are screening many candidates and can tolerate a
known fraction of duds because follow-up is cheap. The choice is a statement about the cost of
being wrong, not a statistical preference.

**Adjusted p-value versus q-value.** An adjusted p-value is comparable to alpha: reject if it is at
or below your chosen rate. A q-value is a BH-adjusted p-value and carries an FDR interpretation at
the threshold. Both are probabilities, neither is a raw p, and a report that mixes them without
labels is unreadable.

**Operational significance is a business object.** It requires a threshold, an owner, and a unit.
"Material" without those three is an opinion. The threshold should be set before the result, by the
person who controls the budget or the service promise, and recorded next to the metric definition.
If no such threshold exists, the honest report is an interval and a question, not a verdict.

## 5. The mechanics, step by step

The sequence below is the order in which the choices must be made. Most bad dashboard statistics
come from making them in a different order: test first, family later, threshold after the result.

### 5.1 Phase A — Fix the estimand and the family

State what quantity is being compared, in what population, over what window, and in what unit. That
is the estimand from Topic 01. Then state the family: the complete set of comparisons over which
you will control an error rate.

A family definition has three parts:

- **The unit of comparison.** Supplier, lane, SKU, store, week, or supplier–lane pair.
- **The metric list.** Named before the refresh, with definitions.
- **The direction and the test.** Two-sided or one-sided, and which test.

Write the family size `m` next to the analysis. If the dashboard displays 240 tiles but the team
will only ever act on 12, the family is still 240 unless the other 228 are removed from the display
or explicitly labelled as context and never used for a claim. Selecting 12 after seeing 240 is the
error, not the correction.

> **Note:** A family is not a physical object; it is a claim about what would have been reported.
> If a different analyst, given the same data, could have reported a different tile, that tile is in
> the family. The test is counterfactual: what is the set of results that could have appeared in
> this report?

### 5.2 Phase B — State H0, H1, and the tail before seeing data

The null must be a statement about a parameter, not about the data. "The promotion had no effect on
fill rate" is a start; "the difference in mean fill rate between test and control stores over the
eight-week window is zero" is testable.

The tail is part of the hypothesis, not part of the output. A two-sided test asks whether the
effect differs in either direction. A one-sided test asks whether it differs in a named direction,
and it requires a justification that the other direction is impossible, already known, or
irrelevant to the decision.

| Question | Tail | Example |
|---|---|---|
| Did the metric change at all? | Two-sided | Fill rate before versus after a process change |
| Did the intervention improve the metric? | One-sided, pre-specified | A training programme expected to reduce pick errors |
| Is the supplier no worse than the incumbent? | One-sided non-inferiority | A second source qualified against a margin |
| Is the effect within a tolerance? | Two-sided equivalence | A new carrier whose transit time must stay within 12 hours |

Two-sided is the right default when either direction would change a decision, which is most of the
time in operations. A one-sided test chosen because the result went the expected way is not a
one-sided test; it is a two-sided test with the p-value halved after the fact. Section 6.8 computes
the difference.

### 5.3 Phase C — Choose the test and its null distribution

The test must match the design: the unit of randomisation, the dependence structure, and the scale
of the outcome.

| Design | Usual test | Null distribution | What must hold |
|---|---|---|---|
| Two independent groups, continuous outcome | Welch t-test | t approximation | Approximately normal means, or large n; unequal variances allowed |
| Two independent groups, proportion | Two-proportion z-test or Fisher exact | Normal or exact | Independent observations; counts, not rates with varying denominators |
| Paired or before–after, one group | Paired t-test or sign test | t or exact | Pairs are meaningful; differences approximately symmetric for t |
| Several groups | Analysis of variance (ANOVA) then contrasts | F, then t | Independent errors, constant variance; correction still needed for contrasts |
| Counts per period | Poisson or negative binomial | Poisson or NB | Mean–variance relationship; overdispersion handled |
| Clustered or repeated measures | Cluster-robust or mixed model | Robust or model-based | Clusters independent; within-cluster dependence modelled |
| Rank or outlier-heavy outcome | Mann–Whitney or permutation | Exact or permutation | Exchangeability under the null |

Two recurring mismatches in supply chain work. First, the unit of analysis is often not the unit of
assignment: a pilot randomises 20 stores but analyses 4,000 orders, and the standard error is
computed as if there were 4,000 independent units. Topic 05 and Module 05 treat the repair. Second,
the outcome is a ratio with a varying denominator: fill rate computed over orders of very different
size is not a proportion with a constant denominator, and a naive z-test overstates precision.

> **Note:** The test's assumptions are part of the null model. If they fail, the p-value is not
> merely imprecise; it is computed against the wrong distribution. A p-value from a test whose
> assumptions are violated is not a conservative estimate of anything in particular.

### 5.4 Phase D — Compute the p-value and the interval together

Report both. The p-value answers "is there evidence of any difference"; the interval answers "how
large could the difference be". The interval is the input to the operational decision, because the
decision threshold is in outcome units.

For a two-sided test at alpha, the interval and the test agree: a 95% interval excludes zero exactly
when p < 0.05. That correspondence is exact for the same test and the same assumptions, and it is
the reason a report can lead with the interval and treat the p-value as a footnote.

The interval also exposes the failure that the p-value hides. An interval of `[0.1, 9.8]` days
excludes zero and is statistically significant, and it is compatible with an operationally trivial
effect and with a very costly one. The p-value cannot distinguish those readings. The interval can.

### 5.5 Phase E — Choose the error rate: FWER or FDR

The choice is about the consequence of a false claim, not about statistical fashion.

| Error rate | Definition | Right when | Cost of choosing it wrongly |
|---|---|---|---|
| FWER | P(at least one false rejection in the family) | A single false claim triggers an irreversible action: contract termination, safety notice, public ranking | Low power; real effects missed |
| FDR | E[V / max(R, 1)], the expected share of false rejections among rejections | Many candidates are screened and follow-up is cheap; a known fraction of duds is tolerable | Some reported findings are false by design |

Useful arithmetic for the choice. Under a global null with `m` independent tests, the expected
number of false positives is `m x alpha`, and the probability of at least one is
`1 - (1 - alpha)^m`.

For 12 comparisons the expected count is 0.60 and the probability of at least one is 0.4596; for
40 it is 2.00 and 0.8715; for 100 it is 5.00 and 0.9941; for 240 it is 12.0 and effectively 1. At 240
comparisons, "at least one false positive" is effectively certain. The only question is how
many are reported and whether the report says so.

### 5.6 Phase F — Apply the correction

Let `p(1) <= p(2) <= ... <= p(m)` be the ordered p-values.

**Bonferroni.** Reject any hypothesis with `p_i <= alpha/m`. Controls FWER under any dependence
structure. It is the simplest and the most conservative; with many correlated metrics it gives away
a great deal of power.

**Holm.** Step down from the smallest p-value. At step `k`, compare `p(k)` to `alpha/(m - k + 1)`.
If it passes, reject and continue; if it fails, stop and reject nothing further. Controls FWER
under any dependence, and is uniformly at least as powerful as Bonferroni.

**Benjamini–Hochberg.** Step up from the largest p-value. Find the largest `k` such that
`p(k) <= k alpha/m`, and reject all hypotheses `1..k`. Controls FDR when the tests are independent
or positively dependent. Under arbitrary dependence, replace `alpha` with `alpha / H_m`, where
`H_m = 1 + 1/2 + ... + 1/m` (Benjamini–Yekutieli).

In the section 6 example with `m = 12`, the rejection counts are 6 uncorrected, 1 for Bonferroni, 2
for Holm, and 3 for BH.

Adjusted p-values make the procedures comparable across metrics. For Holm, the adjusted p-value of
the k-th ordered hypothesis is the running maximum of `(m - j + 1) p(j)` for `j <= k`. For BH, it is
the running minimum of `(m/j) p(j)` for `j >= k`, truncated at 1. Section 13 computes both by hand.

> **Note:** Correction does not repair a badly defined family. Applying BH to 240 dashboard tiles
> after selecting the 12 most interesting produces adjusted p-values for a family that was never
> tested. The family size must be the number of comparisons that could have been reported, not the
> number that were.

### 5.7 Phase G — Equivalence and non-inferiority

A superiority test cannot establish "no worse". Failure to reject is compatible with a large effect
and an underpowered design. Equivalence and non-inferiority tests invert the burden of proof.

**Equivalence.** Hypotheses: `H0: |theta| >= delta` against `H1: |theta| < delta`, where `delta` is
the equivalence margin in outcome units. The two one-sided tests (TOST) procedure runs one test
against `-delta` and one against `+delta`, each at level alpha, and takes the larger p-value.
Equivalently, declare equivalence at level alpha when the `(1 - 2 alpha)` confidence interval lies
entirely inside `(-delta, +delta)`. At alpha 0.05 that is a 90% interval.

**Non-inferiority.** Hypotheses: `H0: theta <= -delta` against `H1: theta > -delta`, where a
positive `theta` favours the new option. Reject H0 when the lower bound of the `(1 - 2 alpha)`
interval exceeds `-delta`. The test is one-sided by construction.

Read the test as an interval rule: for equivalence the whole interval must sit inside the margin;
for non-inferiority its lower bound must clear `-delta`.

Three rules govern the margin: fix it before seeing the data, justify it in outcome units rather
than standard deviations, and never read non-significance as equivalence, since a wide interval can
contain both zero and a harmful effect.

A report that survives audit states the estimand, the family and `m`, the test and tail, alpha, the
effect with its interval, the threshold and its owner, and the decision. Archive the plan and every
deviation: the deviations are the record that makes the claim interpretable.

## 6. Worked example

### 6.1 The setting

A grocery distributor reviews twelve supplier and distribution metrics each quarter. The metrics
were agreed at the start of the year, so the family is fixed at `m = 12`. Each comparison is a
two-sided Welch test at alpha 0.05, using 26 weeks per arm except where noted. The p-values below are
illustrative but internally consistent with the stated effects and sample sizes.

The operational thresholds were set by the operations director and the finance controller before
the quarter began. They are the smallest changes that would trigger a review, a contract
conversation, or a reallocation of volume.

### 6.2 The twelve comparisons

| # | Metric | Estimate | 95% CI | p | Operational threshold | Material? |
|---:|---|---:|---|---:|---:|---|
| 1 | DC-7 pick error rate | -0.9 pp | [-1.5, -0.3] | 0.0008 | 0.5 pp | Yes |
| 2 | Expedite spend per order | +EUR 18 | [+6, +30] | 0.0044 | EUR 25 | No |
| 3 | Supplier A mean lead time | +3.2 d | [+0.7, +5.7] | 0.0105 | 2.0 d | Yes |
| 4 | Fill rate | -1.4 pp | [-2.7, -0.1] | 0.0289 | 1.0 pp | Yes |
| 5 | Forecast bias | +2.1 pp | [+0.1, +4.1] | 0.0410 | 3.0 pp | No |
| 6 | Inventory days of cover | +1.8 d | [+0.02, +3.6] | 0.0487 | 2.0 d | No |
| 7 | On-time in-full (OTIF) | -0.7 pp | [-1.5, +0.1] | 0.0712 | 1.5 pp | No |
| 8 | Perfect order rate | -1.1 pp | [-2.5, +0.3] | 0.1204 | 1.0 pp | Yes, if real |
| 9 | Stockout events per month | +2.0 | [-1.1, +5.1] | 0.2010 | 4.0 | No |
| 10 | Return rate | +0.3 pp | [-0.3, +0.9] | 0.3305 | 1.0 pp | No |
| 11 | Freight cost per unit | +EUR 0.40 | [-0.70, +1.50] | 0.4780 | EUR 0.75 | No |
| 12 | Cycle count accuracy | -0.2 pp | [-1.2, +0.8] | 0.6902 | 0.5 pp | No |

The table is already more informative than a dashboard, because the effect and the threshold sit
next to the p-value. The multiplicity problem is still present: twelve tests were run.

### 6.3 Raw significance

Six of the twelve have `p < 0.05`. Under a global null the expected number is `12 x 0.05 = 0.6` and
the probability of at least one is `1 - 0.95^12 = 0.4596`. Six is far above that, so some signal is
likely, but not that each of the six is real or that the six are the right ones to act on.

### 6.4 Bonferroni, by hand

The Bonferroni threshold is

```text
alpha / m = 0.05 / 12 = 0.0041667
```

Only comparison 1 has `p = 0.0008 <= 0.0041667`. Comparison 2 at `0.0044` misses by `0.00023`.
Bonferroni rejects one hypothesis out of twelve. It controls FWER at 0.05 under any dependence, and
it has thrown away five findings to do so.

### 6.5 Holm, by hand

Holm compares the k-th smallest p-value to `alpha/(m - k + 1)` and stops at the first failure.

| Step k | Comparison | p(k) | Threshold alpha/(13 - k) | Pass? | Action |
|---:|---|---:|---:|---|---|
| 1 | DC-7 pick error | 0.0008 | 0.05/12 = 0.004167 | Yes | Reject H0 |
| 2 | Expedite spend | 0.0044 | 0.05/11 = 0.004545 | Yes | Reject H0 |
| 3 | Supplier A lead time | 0.0105 | 0.05/10 = 0.005000 | No | Stop; retain this and all later |

Holm rejects two. The second rejection is the gain over Bonferroni: `0.0044` lies between
`0.004167` and `0.004545`. Holm still does not reject comparison 3, because once step 3 fails the
procedure stops, even though `0.0105` would pass a later, looser threshold.

### 6.6 Benjamini–Hochberg, by hand

BH compares `p(k)` to `k alpha/m` and finds the largest k that passes.

| k | Comparison | p(k) | k x 0.05/12 | Pass? |
|---:|---|---:|---:|---|
| 1 | DC-7 pick error | 0.0008 | 0.004167 | Yes |
| 2 | Expedite spend | 0.0044 | 0.008333 | Yes |
| 3 | Supplier A lead time | 0.0105 | 0.012500 | Yes |
| 4 | Fill rate | 0.0289 | 0.016667 | No |
| 5 | Forecast bias | 0.0410 | 0.020833 | No |
| 6 | Inventory days | 0.0487 | 0.025000 | No |
| 7 | OTIF | 0.0712 | 0.029167 | No |
| 8 | Perfect order | 0.1204 | 0.033333 | No |
| 9 | Stockouts | 0.2010 | 0.037500 | No |
| 10 | Return rate | 0.3305 | 0.041667 | No |
| 11 | Freight cost | 0.4780 | 0.045833 | No |
| 12 | Cycle count accuracy | 0.6902 | 0.050000 | No |

The largest passing k is 3, so BH rejects comparisons 1, 2, and 3. The estimated FDR among those
three rejections is at most 5%: in repeated studies of this kind, about one in twenty such
rejection sets would contain a false discovery.

BH recovers comparison 3, which Holm lost, because it allows a later, larger p-value to justify
rejecting earlier ones. That is the step-up logic, and it is why BH has more power when several
effects are real.

### 6.7 Statistical versus operational significance

Now apply the thresholds. Among the three BH rejections:

| Comparison | p | BH rejects? | Effect vs threshold | Act? |
|---|---:|---|---|---|
| DC-7 pick error | 0.0008 | Yes | 0.9 pp vs 0.5 pp, better | Yes: investigate the cause and lock it in |
| Expedite spend | 0.0044 | Yes | EUR 18 vs EUR 25 | No: real but below the line |
| Supplier A lead time | 0.0105 | Yes | 3.2 d vs 2.0 d, worse | Yes: open a corrective-action review |

And among the non-rejections:

| Comparison | p | BH rejects? | Effect vs threshold | Act? |
|---|---:|---|---|---|
| Fill rate | 0.0289 | No | -1.4 pp vs 1.0 pp, worse | Watch: the interval excludes zero and crosses the threshold |
| Perfect order | 0.1204 | No | -1.1 pp vs 1.0 pp | Inconclusive: the interval spans trivial and material |
| Inventory days | 0.0487 | No | +1.8 d vs 2.0 d | No immediate action; the interval includes 3.6 d |

Two lessons. First, the BH rejection set and the action set are not the same: expedite spend is real
and below the line, while fill rate is outside the BH set and operationally material, so the right
response is targeted follow-up rather than dismissal. Second, sorting matters: a dashboard sorted by
p puts expedite spend first, and one sorted by `effect / threshold` puts fill rate first. The second
sort matches the decision.

### 6.8 The one-sided temptation

Comparison 6, inventory days, has a two-sided p of 0.0487. Suppose the analyst, seeing that the
change was an increase, recomputes a one-sided p in that direction and reports 0.0244. The number
halves and crosses 0.05 comfortably.

If the direction had been pre-specified — "we expect the new replenishment rule to increase days of
cover" — the one-sided test would have been legitimate, and the analysis plan would say so. It was
not. The direction was chosen after seeing the sign, which means the reported p is selected from
two possible tests. The correct statement is the two-sided one: `p = 0.0487`, interval
`[0.02, 3.6]` days, below the 2.0-day threshold at the point estimate and crossing it in the
interval.

> **Note:** Halving a two-sided p is valid only when the one-sided test was planned and its
> direction justified in advance. Done after inspection, it is a one-line p-hack that is almost
> invisible in a dashboard tooltip.

### 6.9 Equivalence for the supplier change

The same firm qualifies a second source for a component. The question is not whether the new
supplier differs from the incumbent, but whether it is no worse on fill rate by more than 2
percentage points. The margin `delta = 2.0 pp` was agreed with the operations director before the
pilot, on the grounds that a 2 pp drop would breach the service promise for two key accounts.

The observed difference (new minus incumbent) is `-0.65 pp`, with standard error `0.55 pp`.

```text
Lower test:  H0: theta <= -2.0
             z = (-0.65 - (-2.0)) / 0.55 = 1.35 / 0.55 = 2.4545
             one-sided p = 0.0071

Upper test:  H0: theta >= +2.0
             z = (-0.65 - 2.0) / 0.55 = -2.65 / 0.55 = -4.8182
             one-sided p = 0.0000007

TOST p = max(0.0071, 0.0000007) = 0.0071
90% CI = -0.65 +/- 1.645 x 0.55 = [-1.55, +0.25] pp
```

The interval lies entirely inside `(-2.0, +2.0)`. Equivalence is established at the 5% level: the data are
compatible with the new supplier being between 1.55 pp worse and 0.25 pp better, and incompatible
with a worsening of 2 pp or more.

A non-inferiority claim would use the same interval and a one-sided rule: the lower bound `-1.55`
exceeds `-2.0`, so the new supplier is non-inferior. The non-inferiority statement is weaker than
equivalence — it permits any amount of improvement and bounds only the worsening — and is usually
the right claim for a second source.

### 6.10 The base rate, and why a small p is not a small error probability

The correction machinery controls error rates within the family. It does not tell you the
probability that a particular rejection is false, which depends on how many tested hypotheses were
true before you looked. With a 10% base rate, power 0.80, and alpha 0.05:
`P(significant) = 0.10 x 0.80 + 0.90 x 0.05 = 0.125`, so
`P(H0 | significant) = 0.045 / 0.125 = 0.36`. At a 2% base rate the same arithmetic gives 0.75.
More than a third, and then three quarters, of significant results are false. This is why
exploratory dashboards produce findings that do not replicate, and why FDR is the more honest
target for screening than the per-test alpha.

## 7. Code, query, or calculation

The block below reproduces the section 6 arithmetic in Python with no dependencies beyond NumPy. It
is a check, not a production workflow; a production implementation belongs in the metric layer,
where the family and the thresholds are stored with the metric definitions.

```python
import numpy as np

p = np.array([0.0008, 0.0044, 0.0105, 0.0289, 0.0410, 0.0487,
              0.0712, 0.1204, 0.2010, 0.3305, 0.4780, 0.6902])
m, alpha = len(p), 0.05
s = np.sort(p)
bonf = s <= alpha / m
holm = np.zeros(m, dtype=bool)
for i in range(m):
    if s[i] <= alpha / (m - i):
        holm[i] = True
    else:
        break

bh_cut = alpha * np.arange(1, m + 1) / m
passing = s <= bh_cut
k = np.max(np.where(passing)[0]) + 1 if passing.any() else 0
bh = np.arange(m) < k

print("raw        ", int((p < alpha).sum()))
print("Bonferroni ", int(bonf.sum()))
print("Holm       ", int(holm.sum()))
print("BH         ", int(bh.sum()))
```

Expected output: `raw 6`, `Bonferroni 1`, `Holm 2`, `BH 3`. Two checks before trusting any such
routine: the Holm loop must break at the first failure rather than continue, and the BH rejection
must be the block `1..k`, not the set of individual hypotheses that pass their own threshold. A
routine that rejects only the passing indices produces a different, wrong answer whenever a later
p-value passes after an earlier one fails.

## 8. Data, standards, and regulatory context

### 8.1 Why there is a regulatory layer at all

Hypothesis testing acquired its conventions where a wrong claim has direct consequences: a drug
approved on a false positive, a safety signal dismissed, a trial reported selectively. Regulators
responded by fixing the error rate, the primary endpoint, and the analysis plan in advance. A
distribution centre needs no such regulator, but the logic applies wherever a claim triggers an
irreversible action: a supplier terminated, a safety stock cut, a penalty invoked.

### 8.2 Confirmatory clinical trials

| Requirement | Where | What it constrains | Grade |
|---|---|---|---|
| ICH E9, Statistical Principles for Clinical Trials | International Council for Harmonisation (ICH), adopted 1998 | Pre-specification of primary endpoint, analysis population, and methods; control of Type I error in confirmatory trials | B |
| ICH E9(R1) addendum on estimands and sensitivity analysis | ICH, Step 4, 20 November 2019; EMA effective 30 July 2020; FDA guidance May 2021 | The estimand framework: treatment, population, variable, intercurrent events, population-level summary; sensitivity analyses must target the same estimand | B |
| Multiplicity control in confirmatory trials | ICH E9 and regional guidance | The family of primary and key secondary endpoints must be pre-specified, with a stated procedure controlling FWER | B |
| Alpha spending for interim analyses | Trial guidance and Data Monitoring Committee practice | Total Type I error is allocated across interim looks; unplanned looks inflate error | B |
| Reporting of all pre-specified analyses | ICH E9, CONSORT | Selective reporting of endpoints or subgroups is a protocol deviation | B |

The E9(R1) framework is worth reading even if you never run a trial, because it is the most precise
statement available of something this module keeps repeating: the estimand is fixed before the
analysis, and sensitivity analyses are tests of robustness to assumptions, not a search for a
favourable specification. The addendum's five attributes — treatment, population, variable,
handling of intercurrent events, and population-level summary — map cleanly onto a supply chain
metric definition.

### 8.3 Vocabulary and measurement standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 3534-1:2016 | Statistics vocabulary and symbols | Definitions of probability, hypothesis, significance level, and error types | B |
| ISO 5725 series | Accuracy and precision of measurement methods | Separates bias from random error; relevant when the metric itself is measured with error | B |
| JCGM 100:2008 | Expression of uncertainty in measurement | The uncertainty budget behind a measured quantity | B |
| ISO 8601 | Date and time representation | The window over which a metric is computed | B |

These standards constrain vocabulary and measurement, not inference. They will not tell you which
correction to use, only whether two reports mean the same object by the same word.

### 8.4 Professional standards rather than regulation

The ASA statement (Wasserstein and Lazar, 2016) is a professional consensus, not a regulation. Its
six principles say that a p-value indicates incompatibility with a specified model; that it is not
the probability the hypothesis is true or that the data arose by chance; that decisions should not
rest on a threshold alone; that full reporting includes the number of hypotheses tested; that it
does not measure effect size or importance; and that by itself it is weak evidence.

The 2019 follow-up (Wasserstein, Schirm, and Lazar) goes further and recommends abandoning the term
"statistically significant" altogether, on the grounds that the label itself drives dichotomous
reporting. That recommendation is contested, including among the contributors to the same special
issue. For operational work, the practical version is: report the effect, the interval, the family
size, and the threshold; use the word "significant" only with an explicit rate and family attached.

### 8.5 What regulation does not supply

No regulator specifies how a commercial dashboard should handle multiplicity. There is no
equivalent of a statistical analysis plan for a quarterly supplier scorecard, and no requirement to
register one. That absence is the gap this topic fills, and it has three consequences.

- **The family is whatever the tool defines.** A business intelligence platform may compute a
  p-value per tile with no family at all. The analyst has to impose one.
- **The threshold is a management decision.** Nothing in a standard says whether 1 pp of fill rate
  matters. That number comes from the service promise, the contract, or the cost of a stockout.
- **The audit trail is internal.** Without a written plan, the only record of what was tested is the
  dashboard's query log, which is usually neither complete nor retained.

> **Note:** The absence of external regulation is not permission to skip the discipline. It means
> the discipline has to be created inside the organisation, and that the person who creates it is
> usually the analyst who would otherwise be blamed for the noise.

## 9. Industry practice and case evidence

### 9.1 What the clinical model established

Regulated trials are the strongest evidence for multiplicity control: the plan, the family, and the
alpha are public before the data exist. ICH E9 requires pre-specification and Type I error control,
and E9(R1) fixes the estimand and its sensitivity analyses in advance. The lesson is procedural:
**the plan is written before the data, and deviations are reported.**

### 9.2 Online controlled experiments

A/B testing platforms are the closest commercial analogue, and their documentation is explicit
about the multiplicity problem.

| Source | Claim | Grade | Date |
|---|---|---|---|
| GrowthBook documentation, multiple corrections | Two variations and ten metrics produce 100 simultaneous tests; Holm is offered as an FWER default, BH for FDR | D | Accessed 2026 |
| Optimizely documentation, false discovery rate control | Tiers metrics into primary, secondary, and monitoring, and applies FDR control within tiers | D | Accessed 2026 |
| Spotify Confidence documentation, number of success metrics | More success metrics imply a smaller per-metric alpha and a larger required sample | D | Accessed 2026 |
| Kohavi and Chen, "False Positives in A/B Tests", KDD 2024 | False positive risk depends strongly on the base rate of true effects; reports that a substantial share of significant results can be false under plausible low base rates | B | 2024 |

The vendor documentation is useful as evidence of what practitioners are offered, not as evidence
of what works. It is graded D and dated. The KDD paper is peer-reviewed and graded B; its central
point — that the probability a significant result is false depends on the prior odds that the tested
idea works — is the same base-rate argument computed in section 6.10.

The practice gap in commercial experimentation is well documented in the platforms' own
documentation: correction is available, often off by default, and almost always applied to a family
defined by the tool rather than by the decision. A guardrail metric tested for non-inferiority and a
success metric tested for superiority do not belong in the same correction family, yet many tools
place them there.

### 9.3 Preregistration and analysis plans

The empirical record on preregistration is more nuanced than the advocacy.

- **Brodeur, Cook, Hartley, and Heyes (2024)** analyse 15,992 test statistics from randomised
  controlled trials in 15 top economics journals. Preregistration alone shows no evidence of
  reducing p-hacking or publication bias; preregistration with a substantive pre-analysis plan is
  associated with reductions in both. Specificity matters more than the act of registering.
- **Claesen et al. (2021)** examine the first generation of preregistered studies in
  *Psychological Science*. Of 27 studies, 2 had no deviations from their plan, and 89% had at least
  one undisclosed deviation. Deviations were most common for sample size, exclusion criteria, and
  statistical analysis.
- **Bakker et al. (2020)** compare structured and unstructured preregistration formats across 29
  researcher degrees of freedom. Structured formats restrict choices better, but neither format
  eliminates them; inter-coder agreement on the number of hypotheses was only 14%.
- **Lakens et al. (2024)** review the evidence for preregistration and Registered Reports, and
  report that Registered Reports publish far more null results than standard journals, citing
  Scheel et al. (2021) for a 44% versus 96% comparison. The figure is as reported in that review,
  not independently verified here.

The implication for supply chain analytics is direct. A one-page plan that names the primary
metric, the family, the tails, the threshold, and the handling of missing weeks is worth more than
a registration with no content. A plan that is written and then silently abandoned is worth
nothing, and the deviation record is what distinguishes the two.

### 9.4 What the public record does not show

There is no published, audited measurement of false discovery rates in commercial supply chain
dashboards. Vendor case studies report improvements, not the number of metrics tested. Companies do
not publish their scorecard families. Any claim of the form "dashboard X reduced false positives by
Y%" should be treated as a vendor claim, grade D, until the family, the base rate, and the
correction are disclosed. The absence is itself a finding: the methods are mature and the tools are
available, but the reporting convention that would make the error rate observable is missing.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reading p as P(null true) | "There is a 5% chance this is noise" | Inverts the conditional probability | Decisions calibrated to the wrong quantity | Report the definition and the interval; train the readers |
| "Not significant" reported as "no effect" | The interval is omitted from the slide | Failure to reject is confused with evidence of absence | A real and costly effect is dismissed | Report the interval; use equivalence when the claim is "no worse" |
| One-sided test chosen after the sign is known | p is exactly half the two-sided value | Data-dependent choice of tail | Inflated Type I error, invisible in output | Fix the tail in the analysis plan; flag any post-hoc halving |
| Family defined after selection | m equals the number of metrics that looked interesting | The family is the reported set, not the tested set | Correction applied to the wrong denominator | Declare the family before the refresh; record m |
| Correction applied to the wrong family | Bonferroni on 12 while 240 were scanned | Under-correction by a factor of 20 | False discoveries presented as FWER-controlled | Count every comparison that could have been reported |
| Effect size ignored | A significant result with a tiny estimate leads the slide | Large n makes trivial effects significant | Management attention spent on noise | Report effect and threshold beside every p |
| Threshold invented after the result | The "material" line moves between meetings | Operational significance is asserted, not agreed | Post-hoc rationalisation | Fix the threshold with its owner before the analysis |
| HARKing | The hypothesis matches the result exactly | Hypothesis generated from the data, presented as prior | Confirmatory claim with exploratory evidence | Date-stamp the plan; label exploratory findings |
| Winner's curse in rankings | The top supplier looks much better than the second | Selection on noisy estimates | Replication regresses to the mean | Shrink estimates; correct for selection; replicate |
| Peeking at a running test | The dashboard refreshes daily and someone stops when it turns red | Optional stopping without alpha spending | Inflated error rate | Sequential methods or a fixed analysis date |
| Guardrails corrected as success metrics | A guardrail fails and is dropped from the family | Wrong error rate for the decision | Safety or service degradation ignored | Non-inferiority tests with their own margins |
| Equivalence claimed from a non-significant test | Wide interval, p > 0.05, conclusion "no difference" | Absence of evidence read as evidence of absence | A harmful change is accepted | TOST or a non-inferiority test with a justified margin |

Two deserve expansion.

**The family defined after selection** is the hardest to see: a team runs 240 tiles, notices 12 red
ones, and corrects those 12. The arithmetic is valid for a family of 12 and meaningless, because the
12 were chosen for being red. **The winner's curse in rankings** is the operational version. A
supplier league table ranks 40 suppliers on a noisy metric; the top supplier is top partly because
the noise was favourable, and the next measurement will be lower even if nothing changes. Shrinkage
and replication are the remedies, and neither is common in scorecards.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The null model is correctly specified | The p-value is a tail area under that model | Wrong distribution, omitted covariates, wrong functional form | Compare with a permutation or bootstrap null |
| Observations are independent, or dependence is modelled | Standard errors and null distributions assume it | Repeated measures, clustering, autocorrelation | Inspect the design; use cluster-robust or block methods |
| The test statistic's distribution is known or well approximated | The p-value is computed from it | Small samples, skewed data, boundary parameters | Simulation; exact or permutation tests |
| The family is fixed before the data are seen | Error rates are properties of a declared set | Post-hoc selection of metrics, stores, or subgroups | Compare the plan with the reported set; record deviations |
| Alpha is fixed before the analysis | It is a decision rule, not a summary | Changing alpha after seeing p | Date-stamp the plan; audit the analysis log |
| The tail is fixed before the analysis | One-sided tests are legitimate only with a pre-specified direction | Choosing direction after seeing the sign | Plan review; flag half-p-values |
| The operational threshold is fixed and owned | The decision rule depends on it | Moving the line to fit the result | Record the threshold and its owner with the metric definition |
| The metric definition is stable over the window | A change in definition is a change in estimand | Redefinitions, system migrations, scope changes | Version the metric dictionary; test for breaks |
| The sample is exchangeable with the decision population | Inference generalises to the next period | Regime change, supplier change, seasonality | Compare periods; check for structural breaks |
| The margin in an equivalence test is justified independently | Otherwise the test is calibrated to the answer | Margin chosen from the observed interval | Require a clinical or operational rationale in outcome units |
| No unplanned peeking or optional stopping | Repeated looks inflate the error rate | Daily dashboard refreshes, interim reads | Sequential design or fixed analysis dates |

Four limits are worth stating plainly.

**Correction does not create signal.** A family of 240 tests with no true effects will produce false
discoveries at the declared rate no matter which procedure is used. The correction changes which
findings are reported, not whether any are real.

**FDR is not a per-comparison error probability.** A 5% FDR means that about 5% of the rejections in
a long-run series are false, not that a given rejection has a 5% chance of being false. For a single
decision with irreversible consequences, FWER or a fully specified decision-theoretic calculation
is the more appropriate object.

**The procedures assume the p-values are valid.** If the underlying test is wrong — clustered data
analysed as independent, a skewed outcome analysed with a normal test — the correction is applied
to invalid inputs. Multiplicity control is the last step, not the first.

**Equivalence needs a margin that means something.** A margin of 10 pp on fill rate will almost
always be met and will almost never be useful. A margin of 0.1 pp will almost never be met and will
block a beneficial change. The margin is a business judgement, and the statistics cannot supply it.

## 12. Exercises

### Exercise 1 — Run the three corrections by hand (calculation, 45 minutes)

Use this set of ten p-values from a pre-specified supplier scorecard:
`0.002, 0.0052, 0.011, 0.028, 0.039, 0.048, 0.072, 0.130, 0.270, 0.640`.

**Deliverable:** a table with the ordered p-values, the Bonferroni threshold, the Holm sequence, and
the BH sequence, with the rejection count for each at alpha 0.05 and at q = 0.10.
**Acceptance criteria:** correct ordering; Holm stops at the first failure; BH rejects the block
`1..k`; the three counts differ; adjusted p-values reported for at least the first five hypotheses.
**Time:** 45 minutes.

### Exercise 2 — Define the family for a dashboard you know (design, 50 minutes)

Take a dashboard or scorecard you have used. Enumerate every comparison it displays or could
display, and write a one-paragraph family definition.

**Deliverable:** the family definition, the count `m`, and a list of comparisons that would be
excluded and why.
**Acceptance criteria:** `m` is justified against the counterfactual "what could have been
reported"; at least one comparison is excluded with a reason that is not "we do not act on it"; the
expected number of false positives at alpha 0.05 is computed.
**Time:** 50 minutes.

### Exercise 3 — Audit a one-sided claim (critique, 35 minutes)

Find or construct a report that uses a one-sided test. Determine whether the direction was fixed
before the data.

**Deliverable:** a short memo stating the reported p, the implied two-sided p, the evidence that the
direction was or was not pre-specified, and the corrected conclusion.
**Acceptance criteria:** the arithmetic is shown; the memo distinguishes a legitimate one-sided
design from a post-hoc choice; the corrected conclusion states what the data support.
**Time:** 35 minutes.

### Exercise 4 — Design an equivalence test for a supplier switch (design, 50 minutes)

A second source must be qualified. The service promise allows a fill-rate worsening of no more than
1.5 pp. The pilot has 30 weeks per supplier and an expected standard error of the difference of
0.6 pp.

**Deliverable:** the hypotheses, the margin and its justification, the TOST calculation, the 90%
interval, and the decision.
**Acceptance criteria:** the margin is stated in pp and tied to the service promise; the TOST
p-value is the maximum of the two one-sided p-values; the interval rule agrees with the test; the
power at a true difference of 0.5 pp is computed and compared with 0.80.
**Time:** 50 minutes.

### Exercise 5 — Compare the MDE with the operational threshold (calculation, 45 minutes)

For the design in Exercise 4, compute the minimum detectable effect at alpha 0.05 and power 0.80.
Compare it with the 1.5 pp margin and with a 1.0 pp operational threshold.

**Deliverable:** the MDE calculation, the comparison table, and a recommendation on sample size or
margin.
**Acceptance criteria:** the formula `MDE = (1.96 + 0.842) x SE` is used and the inputs are labelled;
the conclusion states whether the design can detect a breach of the threshold; the recommendation
is specific about what to change.
**Time:** 45 minutes.

### Exercise 6 — Simulate the null world (computation, 60 minutes)

Simulate 100 independent two-sample tests with no true effect, 30 observations per arm, normal data
with equal variance. Record the number of p-values below 0.05 over 1,000 replications.

**Deliverable:** the mean and distribution of the false-positive count, the observed count in one
run, and a histogram.
**Acceptance criteria:** the mean is close to 5; the code fixes a seed; the write-up distinguishes the
expected count from the probability of at least one; the result is compared with `1 - 0.95^100`.
**Time:** 60 minutes.

### Exercise 7 — Write a one-page pre-analysis plan (design, 50 minutes)

Choose a metric change, pilot, or process intervention you could plausibly evaluate. Write the plan
before looking at the data.

**Deliverable:** one page containing the estimand, the primary metric, the family and `m`, the test
and tail, the alpha, the operational threshold and its owner, the decision rule, and the handling of
missing or excluded periods.
**Acceptance criteria:** every field is filled with a specific value or name; the threshold predates
the data; at least one plausible deviation is anticipated with a rule for handling it; the plan is
dated and versioned.
**Time:** 50 minutes.

### Exercise 8 — Choose FWER or FDR (design, 40 minutes)

For each decision, state the error rate you would control and why: (a) terminating a supplier for
safety non-compliance; (b) screening 200 store–SKU pairs for a replenishment pilot; (c) publishing a
public supplier ranking; (d) qualifying a second source on a guardrail metric.

**Deliverable:** four short answers, each with the error rate, the family, and the consequence of a
false claim.
**Acceptance criteria:** the answers differ where the consequences differ; guardrails are treated as
non-inferiority tests; at least one answer uses FWER and at least one uses FDR.
**Time:** 40 minutes.

### Exercise 9 — Write the reporting standard (design, 60 minutes)

Draft a one-page standard for how your team reports hypothesis tests in dashboards.

**Deliverable:** the standard, with a required-fields table and two worked examples, one good and
one bad.
**Acceptance criteria:** the standard requires the estimand, family size, test and tail, effect and
interval in units, threshold and owner, and decision; it prohibits "significant" without a stated
rate and family; the bad example is corrected line by line.
**Time:** 60 minutes.

## 13. Self-check answers

### Exercise 1 arithmetic, in full

The ordered p-values for `m = 10` are:

| k | p(k) | Bonferroni alpha/m = 0.005 | Holm alpha/(11 - k) | Pass? | BH k x 0.05/10 | Pass? |
|---:|---:|---|---:|---|---:|---|
| 1 | 0.0020 | 0.0020 <= 0.0050 | 0.05/10 = 0.005000 | Yes | 0.005 | Yes |
| 2 | 0.0052 | 0.0052 > 0.0050 | 0.05/9 = 0.005556 | Yes | 0.010 | Yes |
| 3 | 0.0110 | 0.0110 > 0.0050 | 0.05/8 = 0.006250 | No | 0.015 | Yes |
| 4 | 0.0280 | 0.0280 > 0.0050 | 0.05/7 = 0.007143 | Stopped | 0.020 | No |
| 5 | 0.0390 | 0.0390 > 0.0050 | 0.05/6 = 0.008333 | Stopped | 0.025 | No |
| 6 | 0.0480 | 0.0480 > 0.0050 | 0.05/5 = 0.010000 | Stopped | 0.030 | No |
| 7 | 0.0720 | 0.0720 > 0.0050 | 0.05/4 = 0.012500 | Stopped | 0.035 | No |
| 8 | 0.1300 | 0.1300 > 0.0050 | 0.05/3 = 0.016667 | Stopped | 0.040 | No |
| 9 | 0.2700 | 0.2700 > 0.0050 | 0.05/2 = 0.025000 | Stopped | 0.045 | No |
| 10 | 0.6400 | 0.6400 > 0.0050 | 0.05/1 = 0.050000 | Stopped | 0.050 | No |

**Rejection counts at alpha 0.05:** Bonferroni 1, Holm 2, BH 3.

- Bonferroni rejects only `0.0020`, because `0.0052` exceeds `0.0050` by `0.0002`.
- Holm rejects `0.0020` and `0.0052`, then stops at `0.0110`, which exceeds `0.006250`.
- BH finds the largest passing k. Step 3 passes (`0.0110 <= 0.0150`); step 4 fails
  (`0.0280 > 0.0200`). The largest passing k is 3, so BH rejects the first three.

**At q = 0.10**, the BH thresholds double: `0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
0.080, 0.090, 0.100`. Steps 1 through 6 pass; step 7 fails (`0.0720 > 0.0700`). BH rejects six.
The Bonferroni and Holm counts do not change, because those procedures control FWER at the
chosen alpha, which was not doubled.

**BH adjusted p-values** (running minimum of `(10/j) p(j)` from the right) are `0.0200, 0.0260,
0.0367, 0.0700, 0.0780, 0.0800, 0.1029, 0.1625, 0.3000, 0.6400`. The third is below 0.05 and the
fourth above, confirming k = 3.

**Holm adjusted p-values** (running maximum of `(10 - j + 1) p(j)` from the left) are `0.0200,
0.0468, 0.0880, 0.1960, 0.2340, 0.2400, 0.2880, 0.3900, 0.5400, 0.6400`; the second is below 0.05
and the third above, confirming the count of 2.

A common error is to reject every row that passes its own BH threshold. That coincides with the
block rule only when the passing set is a prefix; when a later row passes after an earlier failure,
the block rule is the correct one and the per-row rule is not.

### Exercise 4 arithmetic

With `delta = 1.5 pp`, `SE = 0.6 pp`, and an observed difference of `-0.4 pp`:

```text
Lower test:  H0: theta <= -1.5
             z = (-0.4 + 1.5) / 0.6 = 1.8333
             one-sided p = 0.0334

Upper test:  H0: theta >= +1.5
             z = (-0.4 - 1.5) / 0.6 = -3.1667
             one-sided p = 0.0008

TOST p = max(0.0334, 0.0008) = 0.0334 < 0.05
90% CI = -0.4 +/- 1.645 x 0.6 = [-1.39, +0.59] pp
```

The interval lies inside `(-1.5, +1.5)`, so equivalence is established. The lower test binds: the
data are only just compatible with a worsening as large as 1.5 pp, and the decision is sensitive to
the margin. If the margin were 1.2 pp, the lower z would be `(-0.4 + 1.2)/0.6 = 1.333`, p = 0.091,
and equivalence would not be established.

**Power at a true difference of 0.5 pp.** For the lower test, the rejection probability is
`Phi((theta + delta)/SE - 1.645)`. With `theta = -0.5`:

```text
Phi((1.0 / 0.6) - 1.645) = Phi(1.667 - 1.645) = Phi(0.022) = 0.509
```

The upper test has power `Phi((1.5 + 0.5)/0.6 - 1.645) = Phi(1.688) = 0.954`. The TOST power is
approximately the smaller, about 0.51, well below 0.80. At a true difference of zero the power is
about 0.80. The design is therefore adequate only if the suppliers are nearly identical, which is
the case the test is supposed to distinguish from a meaningful difference.

### Exercise 5 arithmetic

```text
MDE = (z_0.975 + z_0.80) x SE
    = (1.960 + 0.842) x 0.6
    = 2.802 x 0.6
    = 1.68 pp
```

The MDE of 1.68 pp exceeds the 1.5 pp margin, so a superiority test at 80% power cannot reliably
detect a difference at the margin. The design can detect a 1.68 pp difference, which is larger than
the largest difference the decision tolerates. Two remedies:

- **Increase the sample.** To reach a TOST power of 0.80 at a true 0.5 pp difference, the lower
  test needs `(1.0/SE) - 1.645 = 0.842`, so `SE = 1.0/2.487 = 0.402 pp`. With `SE` proportional to
  `1/sqrt(n)`, the required sample is `(0.6/0.402)^2 = 2.23` times the current size, or about 67
  weeks per supplier instead of 30.
- **Widen the margin**, with an explicit operational justification. A 2.0 pp margin would give
  power `Phi((1.5/0.6) - 1.645) = Phi(0.855) = 0.804` at a true 0.5 pp difference. Widening the
  margin weakens the claim, so it is a business decision, not a statistical one.

The MDE and the TOST power answer different questions. The MDE is the smallest difference a
superiority test can detect. The TOST power is the probability of demonstrating that the difference
is inside the margin. Reporting one when the decision needs the other is a common design error.

### Exercise 6 arithmetic

Under the global null, the count of false positives in 100 independent tests is binomial with
`n = 100` and `p = 0.05`.

```text
mean = 100 x 0.05 = 5
standard deviation = sqrt(100 x 0.05 x 0.95) = sqrt(4.75) = 2.18
P(at least one) = 1 - 0.95^100 = 0.9941
P(exactly five) = C(100, 5) x 0.05^5 x 0.95^95 = 0.180
```

A single simulated run will typically produce between 3 and 7 false positives, and will almost
never produce zero. A run that produces zero is a useful reminder that the expected value is not a
guarantee, and a run that produces ten is not evidence of a broken simulation.

### Rubrics for the design exercises

**Exercise 7 — pre-analysis plan (100 points).**

| Criterion | Weight | What earns full marks |
|---|---:|---|
| Estimand | 20 | Population, unit, outcome, window, and comparison all stated |
| Family and primary metric | 20 | `m` is a number; one primary metric is named; secondaries are labelled |
| Test and tail | 15 | Test named, tail justified, alpha stated |
| Threshold and owner | 20 | Threshold in outcome units, with the role that owns it |
| Decision rule | 15 | States what result leads to act, wait, or stop |
| Deviations | 10 | At least one plausible deviation anticipated with a handling rule |

**Exercise 9 — reporting standard (100 points).**

| Criterion | Weight | What earns full marks |
|---|---:|---|
| Required fields | 30 | Estimand, family, test, tail, effect, interval, threshold, owner, decision |
| Prohibitions | 20 | "Significant" without a rate and family is prohibited; p alone is insufficient |
| Worked examples | 30 | The bad example is corrected line by line, not just criticised |
| Usability | 20 | A analyst could apply the standard without asking a follow-up question |

## 14. Interview questions

**1. "What is a p-value?"**

Strong: "The probability, assuming the null model and the test's assumptions are true, of a test
statistic at least as extreme as the one observed. It is a tail area under the null, not the
probability the null is true, and not the probability of replication." Weak: "The probability that
the result is due to chance", which has no clear referent and is the misreading the topic exists to
correct.

**2. "A metric moved from 94.1% to 94.6% with p = 0.03. Do we act?"**

Strong: asks for the interval, the operational threshold, the family size, and the cost of the
action; notes that 0.5 pp may be inside noise or may be material depending on the service promise.
Weak: "Yes, p is below 0.05", which answers the statistical question and skips the decision.

**3. "You test 12 metrics and one is significant at p = 0.04. What do you report?"**

Strong: reports the family size, the uncorrected p, the adjusted p or the rejection decision under a
named procedure, the effect and interval, and the threshold; states that a single uncorrected result
in a family of 12 has a 46% chance of at least one false positive under the global null. Weak:
reports p = 0.04 as a finding, with no family and no correction.

**4. "Explain the difference between FWER and FDR, and when you would use each."**

Strong: FWER is the probability of at least one false rejection in the family; FDR is the expected
share of false rejections among rejections. FWER for irreversible single claims such as supplier
termination or a safety notice; FDR for screening many candidates when follow-up is cheap. Weak:
describes FDR as "a less strict Bonferroni" without the decision consequence.

**5. "Walk me through Benjamini–Hochberg."**

Strong: order the p-values; find the largest k with `p(k) <= k alpha/m`; reject the block `1..k`;
state that it controls FDR under independence or positive dependence, and that the block rule
matters rather than per-row passing. Weak: says "it is like Bonferroni but divided by rank" and
rejects only the rows that individually pass.

**6. "Why is a one-sided test chosen after seeing the data a problem?"**

Strong: the direction becomes a data-dependent choice, so the reported p is selected from two
possible tests; the nominal alpha is no longer the actual error rate. A pre-specified one-sided test
is legitimate and more powerful. Weak: says one-sided tests are "cheating" without distinguishing
the pre-specified case.

**7. "What is the garden of forking paths?"**

Strong: Gelman and Loken's argument that multiplicity arises from data-dependent analysis choices —
exclusions, transformations, subgroups, metric definitions — even when only one test is reported;
the reported p is one leaf of a tree the analyst did not enumerate. Weak: reduces it to "running
many tests", which misses the invisible multiplicity.

**8. "How would you show that a new supplier is no worse than the incumbent?"**

Strong: an equivalence or non-inferiority design with a pre-specified margin in outcome units, a
TOST or a 90% interval inside the margin, and a power calculation at a relevant difference. Weak:
runs a superiority test, finds p > 0.05, and concludes "no difference", which is absence of evidence
read as evidence of absence.

**9. "How do you decide the equivalence margin?"**

Strong: from the operational consequence — the service promise, the contract penalty, the cost of a
stockout — in outcome units, fixed before the data, with the owner named. Notes that a margin chosen
to fit the observed interval is not a margin. Weak: picks a conventional value such as 0.2 standard
deviations, or picks the margin that the pilot can demonstrate.

**10. "A vendor says its tool 'eliminates false positives'. What do you ask?"**

Strong: asks for the family definition, the base rate of true effects in the validation, the
correction used, the power, and whether the claim is a product feature or a measured error rate.
Grades the claim D until those are disclosed. Weak: accepts the claim because the tool reports a
"confidence score".

## 15. Key terms

- **p-value** — the probability, under the null model, of a statistic at least as extreme as observed.
- **Null hypothesis** — the precise statement tested, usually no difference, fixed before analysis.
- **Alternative hypothesis** — the parameter values considered if the null is false.
- **Significance level (alpha)** — the pre-specified Type I error rate at which the null is rejected.
- **Type I error** — rejecting a true null.
- **Type II error** — failing to reject a false null.
- **Power** — the probability of rejecting the null when a stated alternative is true.
- **Effect size** — the magnitude of the estimate, in outcome units such as days, EUR, or pp.
- **Operational significance** — an effect large enough to change a decision against an agreed threshold.
- **Minimum detectable effect** — the smallest true effect a design can detect at stated alpha and power.
- **One-sided test** — a test with the rejection region in one pre-specified tail.
- **Two-sided test** — a test with the rejection region split across both tails.
- **Multiplicity** — inflation of error rates when many hypotheses are tested and some are selected.
- **Family** — the declared set of comparisons over which an error rate is controlled.
- **Family-wise error rate** — the probability of at least one false rejection in the family.
- **False discovery rate** — the expected share of false rejections among the rejections made.
- **Bonferroni correction** — reject hypotheses with p at or below alpha/m; controls FWER under any dependence.
- **Holm procedure** — step-down FWER control comparing the k-th smallest p to alpha/(m - k + 1).
- **Benjamini–Hochberg procedure** — step-up FDR control rejecting the block 1..k for the largest k with p(k) at or below k alpha/m.
- **Adjusted p-value** — the smallest error rate at which a hypothesis would be rejected, for a named procedure.
- **p-hacking** — trying analyses until p crosses a threshold and reporting only the successful one.
- **HARKing** — hypothesising after the results are known and presenting the hypothesis as prior.
- **Garden of forking paths** — invisible multiplicity from data-dependent analysis choices, even with one reported test.
- **Preregistration** — timestamped registration of hypotheses, design, and analysis before data inspection.
- **Analysis plan** — a versioned document fixing the estimand, family, tests, tails, thresholds, and deviation rules.
- **Winner's curse** — the tendency of a selected best estimate to overstate the true effect.
- **Equivalence margin** — the largest difference regarded as operationally equivalent, fixed before analysis.
- **Non-inferiority margin** — the largest acceptable worsening relative to a comparator.
- **Two one-sided tests (TOST)** — the procedure that declares equivalence when both one-sided tests against the margin reject.
- **Guardrail metric** — a metric that must not worsen, tested for non-inferiority rather than superiority.

## 16. Sources

**Peer-reviewed and methodological (Grade A unless noted)**

- Wasserstein, R. L. and Lazar, N. A. (2016). "The ASA Statement on p-Values: Context, Process, and
  Purpose." *The American Statistician*, 70(2), 129–133.
  DOI: 10.1080/00031305.2016.1154108.
- Wasserstein, R. L., Schirm, A. L. and Lazar, N. A. (2019). "Moving to a World Beyond 'p < 0.05'."
  *The American Statistician*, 73(sup1), 1–19. DOI: 10.1080/00031305.2019.1583913.
- Benjamini, Y. and Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B*,
  57(1), 289–300. DOI: 10.1111/j.2517-6161.1995.tb02031.x.
- Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian
  Journal of Statistics*, 6(2), 65–70. DOI: 10.2307/4615733.
- Gelman, A. and Loken, E. (2014). "The Statistical Crisis in Science." *American Scientist*,
  102(6), 460–465. The garden-of-forking-paths argument; an earlier working paper is dated 2013.
- Schuirmann, D. J. (1987). "A Comparison of the Two One-Sided Tests Procedure and the Power
  Approach for Assessing the Equivalence of Average Bioavailability." *Journal of
  Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680. DOI: 10.1007/BF01068419.
- Walker, E. and Nowacki, A. S. (2011). "Understanding Equivalence and Noninferiority Testing."
  *Journal of General Internal Medicine*, 26(2), 192–196. PMC3019319.
- Lakens, D. (2017). "Equivalence Tests: A Practical Primer for t Tests, Correlations, and
  Meta-Analyses." *Social Psychological and Personality Science*. PMC5502906.
- Brodeur, A., Cook, N., Hartley, J. and Heyes, A. (2024). "Do Preregistration and Preanalysis Plans
  Reduce p-Hacking and Publication Bias? Evidence from 15,992 Test Statistics." *Journal of
  Political Economy Microeconomics*. DOI: 10.1086/730455.
- Claesen, A., Gomes, S., Tuerlinckx, F. and Vanpaemel, W. (2021). "Comparing Dream to Reality: An
  Assessment of Adherence of the First Generation of Preregistered Studies." *Royal Society Open
  Science*. PMC8548785.
- Bakker, M. et al. (2020). "Ensuring the Quality and Specificity of Preregistrations." *PLOS
  Biology*. DOI: 10.1371/journal.pbio.3000937.
- Lakens, D. et al. (2024). "The Benefits of Preregistration and Registered Reports."
  *Evidence-Based Toxicology*. DOI: 10.1080/2833373X.2024.2376046. Reports the 44% versus 96%
  significant-result comparison, citing Scheel et al. (2021); not independently verified here.
- Kohavi, R. and Chen, X. (2024). "False Positives in A/B Tests." *Proceedings of the ACM SIGKDD
  Conference on Knowledge Discovery and Data Mining*. DOI: 10.1145/3637528.3671631.
- Cochrane Handbook for Systematic Reviews of Interventions, Chapter 15, "Interpreting Results and
  Drawing Conclusions." Current edition. Guidance on effect estimates, intervals, and the
  arbitrary 0.05 threshold.

**Regulatory and standards (Grade B)**

- ICH E9, *Statistical Principles for Clinical Trials*, adopted 1998.
- ICH E9(R1), *Addendum on Estimands and Sensitivity Analysis in Clinical Trials*, Step 4,
  20 November 2019. EMA effective 30 July 2020; FDA guidance May 2021.
- ISO 3534-1:2016, *Statistics — Vocabulary and symbols — Part 1*.
- ISO 5725 series, *Accuracy (trueness and precision) of measurement methods and results*.
- JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in
  measurement*.
- ISO 8601-1:2019 and ISO 8601-2:2019, *Date and time — Representations for information
  interchange*.

**Practitioner and vendor documentation (Grade D, dated 2026)**

- GrowthBook documentation, "Multiple Testing Corrections." Accessed 2026. Describes 100
  simultaneous tests from two variations and ten metrics; offers Holm and BH.
- Optimizely documentation, "False Discovery Rate Control." Accessed 2026. Describes metric tiers
  and FDR control.
- Spotify Confidence documentation, "Number of Success Metrics." Accessed 2026. Describes the
  effect of metric count on per-metric alpha and sample size.

**Note on citations.** Only sources returned by research for this topic are cited. Vendor
documentation is graded D and dated at access. No audited measurement of false discovery rates in
commercial supply chain dashboards was found; section 9.4 says so. Figures attributed to reviews are
reported as those reviews state them and were not independently replicated.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- Previous topic, which fixes the estimand this topic tests: [`01_estimands-question-types.md`](01_estimands-question-types.md)
- The interval machinery this topic converts into decisions: [`02_probability-bootstrap.md`](02_probability-bootstrap.md)
- Next topic, which carries the same logic into regression: `04_regression-time-series.md` (planned)
- Dependent-data standard errors, the assumption this topic relies on: `05_autocorrelation-robust-errors.md` (planned)
- Fragility of a finding to omitted variables: `10_sensitivity-omitted-bias.md` (planned)
- Experimental design, where multiplicity is reduced at source: [`../05_experimentation_pilots_evidence/_index.md`](../05_experimentation_pilots_evidence/_index.md)
- Distribution background for the test statistics used here: [`../01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Module survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Return to the part map: [`../_index.md`](../_index.md)

