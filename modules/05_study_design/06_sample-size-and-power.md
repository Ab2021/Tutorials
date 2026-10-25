# Sample Size and Power

> Module 05 — Study Design and Evidence Generation
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 02 and 05
> Version 2026.3

## 1. Why this topic exists

Sample size is where the scientific question meets the budget. A study that is too small cannot answer the question it was designed for, and a study that is too large wastes money and exposes participants to unnecessary risk. The calculation is not a formality at the end of protocol writing; it is the numerical expression of the design.

The commercial consequences are asymmetric. An underpowered study that fails to show a real effect can kill a programme that would have succeeded with more participants. An overpowered study spends money that could have funded a second indication or a better endpoint. A study powered for the wrong outcome can succeed statistically and fail commercially. The calculation is therefore a strategic decision, not a technical one.

The statistical consequences are equally direct. Power depends on the effect size, the variance, the significance level, the power target, the allocation ratio, the dropout rate, the clustering, and the multiplicity. Change any one of these and the required sample size changes, sometimes by a factor of two or three. A sample-size section that reports a single number without the assumptions is not reproducible and cannot be defended.

Regulators expect the calculation to be explicit. ICH E9, the Statistical Principles for Clinical Trials, requires the number of participants to be justified by the expected effect, the variability, the significance level, and the power. The FDA's non-inferiority guidance of November 2016 requires the margin to be justified from historical evidence. The CONSORT 2025 statement requires the sample-size calculation to be reported with the assumptions. These are guidance and convention, not statute, but a protocol that omits them will not survive review.

This topic teaches the arithmetic and the judgement. The arithmetic is straightforward once the inputs are fixed. The judgement is in choosing the inputs: which effect is clinically meaningful, which variance is plausible, which dropout rate is realistic, and which design effect applies.

### The three ways to size a study

There are three legitimate reasons to choose a sample size.

**Power for a specified effect.** The study is sized to detect a difference of a specified magnitude with a specified probability. This is the standard approach for a superiority claim.

**Precision for an estimate.** The study is sized to estimate a quantity with a specified confidence interval width. This is appropriate for descriptive studies, for non-inferiority trials, and for studies where the decision depends on the bound rather than on significance.

**Feasibility and decision value.** The study is sized to the largest feasible sample, and the analysis reports the minimum detectable effect. This is common in rare diseases and in pragmatic trials with fixed infrastructure. It is legitimate only if the minimum detectable effect is clinically useful.

A protocol should state which approach it uses. A study sized by feasibility that reports a non-significant result cannot claim that the treatment is ineffective; it can only claim that the study could not detect the effect it was powered for.

### Why sample-size calculations fail

Four failure patterns recur.

**The borrowed effect.** The effect size is copied from a similar trial without checking whether the population, endpoint, or comparator is the same. The borrowed effect is usually larger than the true effect, because published effects are inflated by selection and because the new population is often less responsive.

**The optimistic variance.** The variance is taken from a pilot with a small sample, where the standard deviation is underestimated. The result is an underpowered trial.

**The ignored design effect.** A cluster trial is powered as if individuals were independent. The result is a trial with a fraction of the intended power.

**The forgotten multiplicity.** The trial has two primary endpoints or three arms, and the sample size is computed for one comparison at alpha 0.05. The result is an inflated false-positive rate or an underpowered claim.

Each failure is avoidable with a sensitivity table and a design review. The table shows how the sample size changes when each assumption is varied, and it makes the fragility visible.

### What the calculation is not

- It is not a guarantee of success. A powered trial can fail because the effect is smaller than assumed, the population is different, or the implementation is poor.
- It is not a substitute for a feasible recruitment plan. A sample size that cannot be recruited is a wish.
- It is not a single number. It is a set of assumptions, a formula, and a sensitivity analysis.
- It is not a regulatory formality. A protocol that reports a number without the inputs is not reviewable.
- It is not independent of the estimand. A change in the estimand changes the sample size.
- It is not independent of the design. A cluster, crossover, or adaptive design changes the information per participant.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** sample size for continuous, binary, and time-to-event outcomes, with and without unequal allocation. (Exercises 1 and 2, self-check 2)
2. **Adjust** a sample size for dropout, non-adherence, and clustering, and explain the design effect. (Exercise 3, self-check 3)
3. **Derive** a non-inferiority margin from historical evidence and compute the corresponding sample size. (Exercise 4, self-check 4)
4. **Account** for multiplicity across multiple endpoints or arms, and choose a testing strategy. (Exercise 5, interview 5)
5. **Size** a study for precision rather than power, and explain when that is appropriate. (Exercise 6, self-check 6)
6. **Build** a sensitivity table that shows how the sample size changes with the effect, variance, dropout, and ICC. (Exercise 7, interview 7)
7. **Critique** a published sample-size section and identify the assumptions that are missing or optimistic. (Exercise 8, interview 8)

### How the outcomes connect

- Outcomes 1 and 2 provide the arithmetic for the common outcome types.
- Outcomes 3 and 4 provide the design adjustments that most often change the answer.
- Outcomes 5 and 6 provide the precision and multiplicity perspective.
- Outcomes 7 and 8 provide the critique and communication skills.
- A learner who can perform outcomes 1 to 4 can compute a sample size.
- A learner who can also perform outcomes 5 to 8 can defend it in review.

## 3. The core mental model

Sample size is a function of information. The information in a study depends on the number of independent observations and the variability of the outcome. The design determines how many independent observations a given number of participants provides.

```text
   QUESTION AND ESTIMAND
            |
            v
   OUTCOME TYPE ---------> continuous | binary | survival | count
            |
            v
   EFFECT SIZE ----------> MCID | pilot | literature | decision threshold
            |
            v
   VARIANCE OR EVENT RATE -> SD | p0 | hazard | overdispersion
            |
            v
   ERROR RATES -----------> alpha | power | one- or two-sided
            |
            v
   BASE SAMPLE SIZE
            |
            +--> allocation ratio
            +--> dropout and non-adherence
            +--> clustering (design effect)
            +--> multiplicity
            +--> interim analyses
            |
            v
   FEASIBILITY CHECK
            |
            +--> recruitment rate
            +--> number of clusters
            +--> event count
            |
            v
   SENSITIVITY TABLE AND DECISION
```

The order matters. The outcome type determines the formula. The effect size and variance determine the base size. The design features inflate it. The feasibility check determines whether the design can be delivered. The sensitivity table shows how fragile the result is.

### The anatomy of a sample-size formula

Every formula has the same structure:

```text
n proportional to (z_alpha + z_beta)^2 * variance / effect^2
```

The numerator is the information required for the error rates. The denominator is the signal. A smaller effect or a larger variance increases the sample size. A stricter alpha or a higher power increases the numerator.

For a two-arm comparison, the formula also includes a factor for the allocation ratio. Equal allocation minimizes the total sample for a given power. Unequal allocation increases the total but may be justified by ethics or recruitment.

### Information and the design effect

Clustering reduces the amount of independent information in a sample. The design effect is the factor by which the sample size must be multiplied to recover the information lost to clustering:

```text
DE = 1 + (m - 1) * ICC
```

where m is the average cluster size and ICC is the intracluster correlation coefficient. When cluster sizes vary, the design effect is larger:

```text
DE = 1 + ((1 + CV^2) * m - 1) * ICC
```

where CV is the coefficient of variation of cluster size. A CV above about 0.23 can materially change the sample size.

The design effect is not a penalty for bad design. It is a description of the information structure. A cluster trial with a high ICC and large clusters provides less independent information per participant than an individually randomised trial, because participants within a cluster are more similar.

### Multiplicity and the alpha budget

When a study tests more than one hypothesis, the family-wise error rate is the probability of at least one false positive across the family. If each test is conducted at alpha 0.05 and there are three independent tests, the family-wise error rate is 1 - 0.95^3 = 0.143, not 0.05.

The sample size must reflect the alpha allocated to each test. A trial with two primary endpoints, each tested at alpha 0.025, needs a larger sample than a trial with one primary endpoint at alpha 0.05. A trial with a hierarchical testing strategy can allocate alpha sequentially, but the sample size must be adequate for the first test in the hierarchy.

### Precision-based sizing

Not every study is a hypothesis test. A study may be sized to estimate a prevalence, a mean, or a risk difference with a specified confidence interval width. The formula is:

```text
n proportional to z^2 * variance / width^2
```

Precision-based sizing is appropriate when the decision depends on the estimate rather than on significance. A payer may accept a coverage decision based on a cost estimate with a confidence interval, not on a p-value. A descriptive study of disease burden may need a prevalence estimate with a margin of error, not a hypothesis test.

### Information as the unifying concept

Every sample-size formula can be read as a statement about information. The information in a study is the inverse of the variance of the estimate. The variance depends on the number of independent observations and the variability of the outcome.

| Design feature | Effect on information | Sample-size consequence |
|---|---|---|
| Larger sample | More information | Smaller variance |
| Lower outcome variability | More information per participant | Smaller sample |
| Equal allocation | Maximum information for a total sample | Smallest total |
| Clustering | Less information per participant | Design effect |
| Measurement error | Less information per participant | Larger sample |
| Multiplicity | Less alpha per test | Larger sample |
| Interim looks | Alpha spent early | Larger maximum sample |
| Non-adherence | Diluted effect | Larger sample |

The table is the conceptual core of the topic. The formulas are bookkeeping; the information argument is the reason they work.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Power | The probability of rejecting the null hypothesis when a specified alternative is true | Reported without the assumed effect | Probability, 0 to 1 |
| Alpha | The prespecified type I error rate | Adjusted informally after multiple looks | Probability |
| Effect size | The magnitude of the difference or association of interest | Chosen to make the sample affordable | Outcome units |
| Minimum clinically important difference (MCID) | The smallest effect that would change practice | Confused with the observed effect | Outcome units |
| Standard deviation (SD) | The variability of a continuous outcome | Underestimated from small pilots | Outcome units |
| Event rate | The proportion of participants with a binary outcome | Assumed constant across arms | Proportion |
| Hazard ratio (HR) | The ratio of hazards between groups | Interpreted as a risk ratio | Ratio |
| Information fraction | The proportion of planned events observed at an interim | Confused with the proportion of participants | Proportion |
| Design effect (DE) | The sample-size multiplier for clustering | Ignored in cluster trials | Multiplier |
| Intracluster correlation coefficient (ICC) | The proportion of variance attributable to between-cluster variation | Assumed to be zero | Proportion |
| Coefficient of variation (CV) | The SD of cluster size divided by the mean cluster size | Ignored in sample-size calculations | Ratio |
| Allocation ratio | The ratio of participants assigned to each arm | Chosen without considering power | Ratio |
| Dropout | Loss of participants before the primary outcome | Confused with non-adherence | Proportion |
| Non-adherence | Failure to take or deliver the intervention as specified | Confused with dropout | Proportion |
| Attrition inflation | Dividing the sample size by (1 - dropout) | Applied without justification | Multiplier |
| Multiplicity | The inflation of type I error from multiple tests | Ignored in multi-arm trials | Probability |
| Family-wise error rate (FWER) | The probability of at least one false positive across a family | Confused with per-test alpha | Probability |
| False discovery rate (FDR) | The expected proportion of false positives among rejected hypotheses | Confused with FWER | Proportion |
| Alpha spending | Allocating alpha across interim analyses | Confused with alpha adjustment for endpoints | Function |
| Non-inferiority margin | The largest acceptable loss of effect relative to an active comparator | Chosen for convenience | Effect units |
| Assay sensitivity | The ability of a trial to distinguish an effective treatment from an ineffective one | Assumed rather than demonstrated | Trial property |
| Superiority margin | The smallest effect that would be considered clinically meaningful | Confused with the NI margin | Effect units |
| Precision | The width of a confidence interval | Confused with power | Effect units |
| Minimum detectable effect (MDE) | The smallest effect a study can detect with a given power | Reported without the power | Effect units |
| Events per variable | The number of outcome events per predictor in a model | Confused with participants per variable | Ratio |
| Schoenfeld formula | The formula for the number of events required in a survival trial | Applied without the event probability | Formula |
| Overdispersion | Variance greater than the mean in count data | Ignored in Poisson models | Parameter |
| Cluster size variation | The distribution of cluster sizes around the mean | Assumed to be zero | Distribution |
| Recruitment rate | The number of participants enrolled per site per month | Assumed rather than measured | Count per time |
| Screening-to-enrolment ratio | The number of patients screened per participant enrolled | Ignored in feasibility planning | Ratio |
| Feasibility | Whether the study can be delivered with the available resources | Confused with power | Judgement |
| Power curve | A plot of power against effect size | Confused with a sensitivity table | Curve |
| Assurance | The probability of a successful trial averaged over the prior | Confused with power | Probability |
| Predictive power | The probability of success given the interim data | Confused with conditional power | Probability |
| Conditional power | The probability of success given the current data and an assumed effect | Confused with predictive power | Probability |
| Information-based design | Sizing by the amount of statistical information | Confused with event count | Design method |

Two distinctions deserve emphasis.

**Power versus precision.** Power is the probability of rejecting a null hypothesis. Precision is the width of an estimate. A study can be powered for a hypothesis test and still produce an estimate too imprecise for the decision. A study can be precise and not powered for a formal test. The protocol should state which it needs.

**Dropout versus non-adherence.** Dropout is missing data; non-adherence is failure to receive the intervention. Both reduce the observed effect, but they require different adjustments. Dropout inflates the sample size by 1/(1 - dropout). Non-adherence dilutes the intention-to-treat effect and may require a larger sample or a per-protocol analysis.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 5 fix the inputs. Steps 6 to 10 compute and adjust. Steps 11 to 14 check feasibility, test sensitivity, and document.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Claim and decision | Estimand and threshold | No target for the calculation |
| 2 | Estimand | Outcome type | Wrong formula |
| 3 | MCID or pilot | Effect size and range | Underpowered or overpowered |
| 4 | Pilot or registry | Variance or event rate | Wrong precision |
| 5 | Decision and regulation | Alpha, power, allocation | Wrong error rates |
| 6 | Inputs | Base sample size | No reference point |
| 7 | Retention data | Dropout inflation | Too few evaluable participants |
| 8 | ICC and cluster size | Design effect and cluster count | Underpowered cluster trial |
| 9 | Endpoints and arms | Alpha allocation | Inflated false-positive rate |
| 10 | Interim plan | Maximum and expected sample | Type I error or power loss |
| 11 | Sites and rates | Feasibility assessment | Unrecruitable trial |
| 12 | Ranges | Sensitivity table | Hidden fragility |
| 13 | Decision | Precision calculation | Imprecise estimate |
| 14 | All of the above | Documented sample-size section | Unreproducible protocol |

### Formula reference

| Outcome | Formula | Inputs | Notes |
|---|---|---|---|
| Continuous, equal | (z_a/2 + z_b)^2 * 2 * SD^2 / delta^2 | SD, delta | Normal approximation |
| Continuous, unequal | (z_a/2 + z_b)^2 * SD^2 * (1 + 1/r) / delta^2 | SD, delta, r | r = n2/n1 |
| Binary | (z_a/2 + z_b)^2 * [p0(1-p0) + p1(1-p1)] / (p1-p0)^2 | p0, p1 | Use exact method if small |
| Survival | (z_a/2 + z_b)^2 / (log HR)^2 / (p1*p2) | HR, allocation | Events, not participants |
| Cluster | Individual n * [1 + ((1+CV^2)m - 1) ICC] | ICC, m, CV | Convert to clusters |
| Non-inferiority | 2 * (z_a + z_b)^2 * SD^2 / (margin - delta)^2 | Margin, SD | One-sided alpha |
| Precision | z^2 * variance / half-width^2 | Variance, width | No alternative hypothesis |

### Step 1: State the estimand and the decision

**Input:** the claim, the estimand, and the decision rule.

**Output:** a statement of the quantity to be estimated and the threshold that determines the decision.

**Rationale:** The sample size depends on the quantity and the decision. A trial powered for a mean difference is different from a trial powered for a risk difference or a hazard ratio.

The decision rule should state whether the study is a superiority test, a non-inferiority test, or a precision study. It should also state the threshold: the smallest effect that would change the decision.

### Step 2: Identify the outcome type

**Input:** the estimand.

**Output:** an outcome type: continuous, binary, time-to-event, count, or composite.

**Rationale:** The outcome type determines the formula and the metric. A continuous outcome is powered on the mean difference and the SD. A binary outcome is powered on the risk difference and the event rates. A time-to-event outcome is powered on the number of events and the hazard ratio.

| Outcome type | Effect measure | Variance input | Sample-size driver |
|---|---|---|---|
| Continuous | Mean difference | SD | Participants per arm |
| Binary | Risk difference or ratio | Event rates | Participants and events |
| Time-to-event | Hazard ratio | Event probability | Events |
| Count | Rate ratio | Mean and dispersion | Person-time |
| Composite | Effect on the composite | Component rates | Events and components |

### Step 3: Choose the effect size

**Input:** the MCID, pilot data, literature, or a decision threshold.

**Output:** a target effect with a justification and a range.

**Rationale:** The effect size is the most consequential and least certain input. A smaller effect requires a larger sample, often disproportionately so.

The effect should be justified by one of three sources:

- the MCID, derived from clinical judgement or a stated preference study,
- a pilot or Phase 2 estimate, with a confidence interval,
- a published effect, adjusted for the differences between the published and planned populations.

The protocol should state the source and the uncertainty. A target effect with no source is a guess, and a guess that is too large produces an underpowered trial.

Sources of an effect size, in order of preference:

- a prespecified MCID from clinical guidelines or a preference study;
- a pilot or Phase 2 estimate with a confidence interval;
- a published effect from a trial in a similar population and setting;
- a decision threshold derived from the commercial or coverage model;
- an expert consensus, labelled as such and treated as uncertain.

If the only available effect is from a published trial, the protocol should state how the planned population differs and whether the effect is likely to be larger or smaller. Published effects are often inflated by publication bias and by the selection of the most responsive population.

### Step 4: Choose the variance or event rate

**Input:** pilot data, literature, or registry data.

**Output:** a variance or event rate with a justification and a range.

**Rationale:** The variance determines the precision of a continuous estimate; the event rate determines the number of events in a binary or survival study.

The variance should be estimated from the most relevant population. A pilot with 30 participants gives an imprecise SD; the upper bound of its confidence interval may be a safer input than the point estimate. For event rates, the control-arm rate should be based on the current standard of care, not on a historical trial.

### Step 5: Set alpha, power, and allocation

**Input:** the decision rule, the regulatory context, and the ethics.

**Output:** a one- or two-sided alpha, a power target, and an allocation ratio.

**Rationale:** Alpha and power determine the information required. The conventional choices are alpha 0.05 two-sided and power 80 or 90 percent. A confirmatory trial often uses 90 percent power; a exploratory study may use 80 percent.

The allocation ratio affects the total sample. Equal allocation is most efficient. Unequal allocation, such as 2:1, may be justified by participant preference or by the need to gather more safety data, but it increases the total sample for the same power.

### Step 6: Compute the base sample size

**Input:** the effect, the variance, alpha, power, and the allocation ratio.

**Output:** a base sample size with the formula and arithmetic.

**Rationale:** The base calculation is the reference point. The adjustments in the following steps are applied to it.

The formulas for the common outcome types are:

```text
Continuous, equal allocation:
n per arm = (z_alpha/2 + z_beta)^2 * 2 * SD^2 / delta^2

Binary, equal allocation:
n per arm = (z_alpha/2 + z_beta)^2 * [p0(1-p0) + p1(1-p1)] / (p1 - p0)^2

Survival, equal allocation:
events = (z_alpha/2 + z_beta)^2 / (log(HR))^2 / (0.5 * 0.5)
participants = events / event probability

Continuous, unequal allocation (ratio r):
n1 = (z_alpha/2 + z_beta)^2 * SD^2 * (1 + 1/r) / delta^2
n2 = r * n1
```

The formulas assume independent observations, a known variance, and a normal approximation. They should be checked against a validated calculator or a simulation for small samples or extreme proportions.

### Step 7: Adjust for dropout and non-adherence

**Input:** the expected dropout and adherence rates.

**Output:** an inflated sample size with the adjustment stated.

**Rationale:** Dropout reduces the number of participants with an observed outcome. The sample size should be divided by (1 - dropout) to retain the planned number of evaluable participants.

Non-adherence reduces the observed effect under an intention-to-treat analysis. If the expected adherence is 70 percent and the per-protocol effect is the target, the intention-to-treat effect may be diluted by a factor of 0.7. The sample size should be increased accordingly, or the estimand should be changed to a per-protocol effect with an appropriate analysis.

The adjustment arithmetic is simple but easy to get wrong:

- Inflate for dropout by dividing by (1 - dropout), not by multiplying by (1 + dropout).
- Inflate for non-adherence by dividing the target effect by the expected adherence before computing the sample, if the per-protocol effect is the target.
- Apply the dropout inflation after the clustering adjustment if the dropout is at the participant level, or before if it is at the cluster level.
- Round up at each step, not only at the end.
- State whether the reported number is randomised, treated, or evaluable.

A sample-size section that reports "1,200 patients" without saying whether that is randomised or evaluable cannot be checked against the protocol's flow diagram.

### Step 8: Adjust for clustering

**Input:** the ICC, the average cluster size, and the cluster-size variation.

**Output:** a design effect and the number of clusters required.

**Rationale:** Clustering reduces the effective sample size. The design effect must be applied to the individual sample size, and the result must be converted into a number of clusters.

The steps are:

1. Compute the individual sample size per arm.
2. Compute the design effect.
3. Multiply the individual sample size by the design effect.
4. Divide by the average cluster size to obtain the number of clusters per arm.
5. Round up and check feasibility.

If the number of clusters is fixed, the design may be infeasible. The feasibility check is that the number of clusters per arm must exceed the individual sample size per arm times the ICC. If it does not, increasing the cluster size cannot achieve the target power.

### Step 9: Adjust for multiplicity

**Input:** the number of primary endpoints, arms, or subgroups.

**Output:** an alpha allocation and a revised sample size.

**Rationale:** Multiple tests inflate the family-wise error rate. The sample size must reflect the alpha allocated to each test, or the testing strategy must be hierarchical so that only one test is conducted at the full alpha.

The common strategies are:

| Strategy | How it works | Sample-size implication |
|---|---|---|
| Single primary endpoint | One test at alpha | Base size |
| Hierarchical | Test in a fixed order, each at alpha | Power the first test; later tests conditional |
| Bonferroni | Split alpha across tests | Larger sample for each test |
| Holm | Step-down adjustment | Slightly larger than Bonferroni |
| Hochberg | Step-up adjustment | Slightly larger than Bonferroni |
| FDR | Control the expected proportion of false positives | Larger for the same power |
| Gatekeeping | Test a primary endpoint before secondaries | Power the gate |

A trial with two primary endpoints and no hierarchy needs a larger sample than a trial with one. A trial that reports a subgroup claim needs to be powered for the interaction, which is usually much larger than the main effect.

### Step 10: Adjust for interim analyses

**Input:** the number and timing of interim looks, and the alpha-spending function.

**Output:** a maximum sample size and an expected sample size.

**Rationale:** Interim analyses that allow early efficacy stopping inflate the type I error unless alpha is spent formally. The maximum sample size must be increased to preserve power, and the expected sample size may be lower because of early stopping.

Futility analyses generally do not inflate the type I error, but they can reduce power if the boundary is aggressive. The operating characteristics should be simulated, not assumed.

### Step 11: Check feasibility

**Input:** the recruitment rate, the number of sites, the number of clusters, and the event count.

**Output:** a feasibility assessment with the recruitment timeline and the event timeline.

**Rationale:** A sample size that cannot be recruited is not a sample size. The feasibility check should include the screening-to-enrolment ratio, the site activation timeline, and the event accrual rate.

For a survival trial, the event timeline is often the binding constraint. A trial with 500 events and a 2 percent annual event rate may need many more participants or a longer follow-up than planned.

A feasibility assessment should include:

- the number of eligible sites and the activation timeline;
- the screening-to-enrolment ratio;
- the expected recruitment rate per site per month;
- the event rate and the expected time to the target number of events;
- the retention rate and the expected evaluable sample;
- the number of clusters and the expected cluster size, if applicable;
- the budget and the cost per participant;
- the contingency plan if recruitment or events fall short.

### Step 12: Build a sensitivity table

**Input:** the base assumptions and plausible ranges.

**Output:** a table of sample sizes across scenarios.

**Rationale:** The sensitivity table shows how fragile the calculation is. A sample size that doubles when the effect is 20 percent smaller is a design that depends on an optimistic assumption.

The table should vary the effect, the variance or event rate, the dropout, and the ICC. It should also show the minimum detectable effect at the planned sample size.

### Step 13: Consider precision-based sizing

**Input:** the decision and the acceptable confidence interval width.

**Output:** a sample size based on precision, or a justification for power-based sizing.

**Rationale:** Some decisions depend on the bound of an estimate rather than on statistical significance. A payer may need a cost estimate with a confidence interval narrow enough to support a budget decision. A prevalence study may need a margin of error.

The precision formula is:

```text
n = z^2 * variance / width^2
```

For a proportion, the variance is p(1-p), and the width is the half-width of the confidence interval.

### Step 14: Document and review

**Input:** all of the above.

**Output:** a sample-size section with the inputs, the formula, the arithmetic, the adjustments, the sensitivity table, and the feasibility assessment.

**Rationale:** The section must be reproducible. A reviewer should be able to recompute the number from the stated inputs. A section that reports only the final number is not reproducible.

The section should state the software or method used, the version, and any simulation code. It should also state the assumptions that are uncertain and the sensitivity analyses that address them.

A complete sample-size section contains:

- the primary estimand and the decision rule;
- the outcome type and the effect measure;
- the target effect and its source;
- the variance or event rate and its source;
- alpha, power, sidedness, and allocation ratio;
- the formula and the arithmetic;
- the dropout and adherence adjustments;
- the clustering adjustment and ICC, if applicable;
- the multiplicity strategy;
- the interim analysis plan and alpha spending;
- the feasibility assessment and recruitment plan;
- the sensitivity table;
- the software and version.

A section missing any of these is incomplete, even if the final number is correct.

### Edge cases

**Non-inferiority margins.** A non-inferiority trial is powered on the margin, not on the expected effect. The margin must be justified from historical evidence of the comparator's effect, and it must be smaller than that effect. A margin chosen for convenience produces a trial that cannot distinguish a non-inferior product from an ineffective one.

**Multi-arm trials.** A trial with several experimental arms and a shared control has correlated comparisons. The sample size must account for the multiplicity and the shared control. A trial powered for one comparison and analysed for five is not powered for any of them.

**Cluster size variation.** The design effect assumes equal cluster sizes. When cluster sizes vary, the effective sample size is lower. The coefficient of variation should be estimated from prior data, and the modified design effect should be used when the CV is above about 0.23.

**Adaptive interim timing.** The information fraction at the interim determines the value of the adaptation. An interim at 25 percent of the events has unstable estimates; an interim at 75 percent has little opportunity to save resources. The timing should be justified and simulated.

**External control drift.** When the control is external, the sample size must account for the uncertainty in the external estimate. The effective sample size of the external control may be much smaller than its nominal size, and the drift in the standard of care adds a bias that the sample size cannot fix.

**Missing endpoint data.** Missing data reduce the evaluable sample and can bias the estimate. The sample size should be inflated for the expected missingness, and the analysis should prespecify the handling method and sensitivity analyses.

### Assumptions that drive the calculation

| Assumption | Typical source | Typical range | Consequence of error |
|---|---|---|---|
| Effect size | MCID, pilot, literature | 0.5 to 1.5 times the target | Largest effect on sample |
| SD | Pilot, registry | 0.8 to 1.3 times the pilot | Moderate effect |
| Control event rate | Registry, prior trial | 0.7 to 1.3 times the prior | Moderate to large effect |
| ICC | Pilot, literature | 0 to 0.15 | Large effect in cluster trials |
| Cluster size | Site data | 0.5 to 1.5 times the plan | Moderate effect |
| Dropout | Retention data | 0.1 to 0.3 | Moderate effect |
| Adherence | Pilot | 0.5 to 0.9 | Moderate effect |
| Multiplicity | Testing strategy | 1 to 4 tests | Moderate effect |
| Interim looks | Adaptation plan | 1 to 5 | Small to moderate effect |

The table is a reminder that the effect size and the ICC are the assumptions most likely to change the answer by a factor of two or more. A sensitivity table should always vary them.

### A multiplicity worked example

Suppose a trial has two co-primary endpoints, each tested at alpha 0.025, with 90 percent power. The sample size for one endpoint at alpha 0.05 would be 273 per arm. For alpha 0.025, the z quantile rises from 1.96 to 2.24, and the sample size becomes:

```text
n = (2.241 + 1.282)^2 * 2 * 18^2 / 5^2
  = 12.428 * 648 / 25
  = 322.1
```

or 323 per arm for each endpoint. If the endpoints are independent, the trial needs 323 per arm to have 90 percent power for both. If the endpoints are correlated, the required sample may be smaller because the tests share information, but the calculation must account for the correlation.

### Decision table: sample-size approach by claim

| Claim | Approach | Key input | Typical pitfall |
|---|---|---|---|
| Superiority, continuous | Power for a mean difference | SD and MCID | SD underestimated |
| Superiority, binary | Power for a risk difference | Control rate and target rate | Control rate outdated |
| Superiority, survival | Power for a hazard ratio | Event probability and HR | Event accrual too slow |
| Non-inferiority | Power for the margin | Historical effect and margin | Margin not justified |
| Equivalence | Power for the equivalence margin | Margin and variance | Margin too wide |
| Precision | Width of the confidence interval | Variance and target width | Width not decision-relevant |
| Subgroup claim | Power for the interaction | Subgroup effect and prevalence | Subgroup underpowered |
| Multi-arm | Power for the testing strategy | Alpha allocation | No multiplicity control |
| Cluster | Power with the design effect | ICC and cluster size | ICC assumed zero |
| Adaptive | Simulation of operating characteristics | Adaptation rule | No simulation report |

## 6. Worked example

### The setting

A regional health authority is evaluating a community hypertension programme delivered by primary care clinics. The programme includes nurse-led medication titration, home blood pressure monitoring, and monthly follow-up. The authority wants to know whether the programme reduces systolic blood pressure at 12 months compared with usual care.

The intervention is delivered at the clinic level, so the unit of allocation is the clinic. The primary outcome is the change in systolic blood pressure at 12 months. The analysis will use a mixed-effects model with a clinic random effect and adjustment for baseline blood pressure.

### Step 1: Estimand and decision

The estimand is:

> In adults with uncontrolled hypertension attending participating primary care clinics, the difference in mean change in systolic blood pressure at 12 months between the programme and usual care, regardless of whether participants remained in the programme.

The intercurrent-event strategy is treatment policy. The decision rule is that the programme will be adopted if the 95 percent confidence interval for the difference excludes zero and the point estimate is at least 3 mmHg, the minimum clinically important difference.

### Step 2: Outcome type

The outcome is continuous: change in systolic blood pressure in mmHg. The effect measure is the mean difference. The variance input is the SD of the change.

### Step 3: Effect size

The target effect is 5 mmHg, based on a pilot in two clinics and on the published effects of similar nurse-led programmes. The pilot estimated a difference of 5.4 mmHg with a 95 percent confidence interval of 2.1 to 8.7, so 5 mmHg is within the plausible range but not the most optimistic estimate.

The MCID is 3 mmHg. A target of 5 mmHg gives the trial a margin above the MCID, so that a true effect of 3 to 4 mmHg would still be detectable with reasonable power.

### Step 4: Variance

The SD of the change in systolic blood pressure is assumed to be 18 mmHg, based on the pilot and on published trials. The pilot's SD was 17.2 mmHg, and the upper bound of its confidence interval was about 21 mmHg. The calculation uses 18 mmHg and the sensitivity table includes 21 mmHg.

### Step 5: Alpha, power, and allocation

Alpha is 0.05 two-sided. Power is 90 percent. Allocation is 1:1 at the clinic level.

### Step 6: Base sample size

The individual sample size per arm is:

```text
n = (z_alpha/2 + z_beta)^2 * 2 * SD^2 / delta^2
  = (1.959964 + 1.281552)^2 * 2 * 18^2 / 5^2
  = 10.507 * 2 * 324 / 25
  = 10.507 * 648 / 25
  = 272.3
```

Rounding up gives 273 participants per arm, or 546 individually randomised participants.

### Step 7: Dropout

The expected dropout is 15 percent, based on the pilot. The inflated individual sample size is:

```text
273 / 0.85 = 321.2
```

so 322 participants per arm, or 644 in total, if the design were individually randomised.

### Step 8: Clustering

The average clinic size is 120 participants. The ICC is 0.04, based on the pilot and on published values for blood pressure outcomes in primary care. The coefficient of variation of clinic size is 0.35.

The design effect is:

```text
DE = 1 + ((1 + 0.35^2) * 120 - 1) * 0.04
   = 1 + (1.1225 * 120 - 1) * 0.04
   = 1 + (134.7 - 1) * 0.04
   = 1 + 5.348
   = 6.348
```

The required individual sample size per arm is 273 * 6.348 = 1,733.0, so 1,733. Dividing by the average clinic size gives 1,733 / 120 = 14.44, so 15 clinics per arm, or 30 clinics in total.

The total number of participants is 30 * 120 = 3,600. With 15 percent dropout, the evaluable sample is 3,060, or 1,530 per arm.

The feasibility check is that the number of clinics per arm must exceed the individual sample size per arm times the ICC:

```text
273 * 0.04 = 10.92
15 > 10.92
```

The check passes. If the ICC were 0.08, the design effect would be 11.70, the required individual sample size would be 3,193 per arm, and the number of clinics per arm would be 27, which would exceed the available clinics.

### Step 9: Multiplicity

There is one primary endpoint. The secondary endpoints (diastolic blood pressure, control rate, and cost per patient) are reported as supportive, with confidence intervals and no multiplicity-adjusted claims. The sample size does not need an adjustment for the primary analysis.

If the authority wanted a claim on the control rate as a co-primary endpoint, the alpha would be split, and the sample size would increase. The protocol states that no such claim will be made.

### Step 10: Interim analyses

There is no interim analysis. The programme is low-risk, and the authority wants the full 12-month data before deciding. A futility interim was considered and rejected because the expected sample-size saving was small relative to the operational complexity.

### Step 11: Feasibility

The region has 46 eligible clinics. The trial needs 30, leaving a margin for clinics that withdraw. The pilot recruited about 8 participants per clinic per month, so 120 participants per clinic would take about 15 months. The 12-month follow-up means the total trial duration is about 27 months plus analysis.

The recruitment rate is the binding constraint. If the rate is 6 participants per clinic per month, the recruitment period extends to 20 months. The protocol includes a recruitment monitoring plan and a contingency to add clinics.

### Step 12: Sensitivity table

| Scenario | Effect (mmHg) | SD | ICC | Dropout | Clinics per arm | Participants |
|---|---:|---:|---:|---:|---:|---:|
| Base case | 5 | 18 | 0.04 | 15% | 15 | 3,600 |
| Smaller effect | 4 | 18 | 0.04 | 15% | 23 | 5,520 |
| Larger SD | 5 | 21 | 0.04 | 15% | 20 | 4,800 |
| Higher ICC | 5 | 18 | 0.08 | 15% | 27 | 6,480 |
| Higher dropout | 5 | 18 | 0.04 | 25% | 17 | 4,080 |
| Lower ICC | 5 | 18 | 0.02 | 15% | 9 | 2,160 |

The table shows that the sample size is most sensitive to the effect size and the ICC. A 20 percent smaller effect increases the number of clinics from 15 to 23 per arm. A doubling of the ICC increases it from 15 to 24 per arm. The base case is feasible; the smaller-effect scenario is at the edge of the available clinics.

### Step 13: Precision

The authority's decision depends on the lower bound of the confidence interval, not only on significance. With 30 clinics and an assumed ICC of 0.04, the effective sample size per arm is 1,530 / 6.348 = 241. The standard error of the mean difference is approximately:

```text
SE = sqrt(2 * SD^2 / n_eff)
   = sqrt(2 * 324 / 241)
   = sqrt(2.689)
   = 1.64 mmHg
```

The 95 percent confidence interval half-width is 1.96 * 1.64 = 3.21 mmHg. With a point estimate of 5 mmHg, the interval is 1.8 to 8.2 mmHg. The lower bound is below the MCID of 3 mmHg, so the trial may not be able to demonstrate that the effect exceeds the MCID even if the point estimate does.

This is an important limitation. The trial is powered for superiority, not for a margin above the MCID. If the authority requires evidence that the effect exceeds 3 mmHg, the sample size must be larger. The protocol states that the primary claim is superiority, and the MCID is reported as a secondary interpretation.

### Power versus precision in this trial

| Quantity | Value | Interpretation |
|---|---|---|
| Effective sample per arm | 241 | After the design effect |
| Standard error of the difference | 1.64 mmHg | Precision of the estimate |
| 95% CI half-width | 3.21 mmHg | Width of the interval |
| Power for a 5 mmHg effect | 90% | Probability of significance |
| Power for a 3 mmHg effect | About 45% | Probability of detecting the MCID |
| Probability the CI excludes 3 mmHg | Low | The trial cannot establish the MCID |

The table shows the difference between power and precision. The trial is powered to detect a 5 mmHg effect, but it cannot establish that the effect exceeds 3 mmHg. If the decision requires that latter claim, the trial is underpowered for the decision, even though it is powered for the test.

### Step 14: Documentation

The protocol reports the inputs, the formulas, the arithmetic, the design effect, the feasibility assessment, and the sensitivity table. It also reports the software used and the random seed for the simulation check. The sample-size section is reviewed by an independent statistician before the protocol is finalized.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the ICC check.** The design assumes an ICC of 0.04. If the true ICC is 0.08, the design effect doubles and the required number of clinics rises from 15 to 24 per arm. A trial with 15 clinics per arm would be substantially underpowered. The diagnostic is to estimate the ICC from the pilot with a confidence interval and to include the upper bound in the sensitivity table. A pilot with only two clinics cannot estimate the ICC reliably.

**Diagnostic 2: the cluster-size check.** The design assumes an average of 120 participants per clinic. If clinics recruit only 60 participants, the design effect falls to 3.37, but the total sample falls to 1,800, and the effective sample per arm is 15 * 60 / 3.37 = 267, marginally below the required 273. The trial would be slightly underpowered. The diagnostic is to monitor recruitment per clinic and to extend recruitment if the average falls below the plan.

### Vary one assumption: the effect size

If the true effect is 4 mmHg rather than 5 mmHg, the individual sample size per arm is:

```text
n = 10.507 * 2 * 324 / 16
  = 425.6
```

so 426 per arm. With the design effect of 6.348, the required individual sample is 426 * 6.348 = 2,704.2, or 2,705. Dividing by 120 gives 22.5, so 23 clinics per arm and 46 clinics in total. The region has 46 eligible clinics, so the trial would use every clinic with no margin for withdrawal. The design is at the edge of feasibility.

### Vary one assumption: the ICC

If the ICC is 0.08, the design effect is 11.70, the required individual sample is 273 * 11.70 = 3,193, and the number of clinics per arm is 27. The total is 54 clinics, which exceeds the available 46. The trial would need to reduce the cluster size, add clinics from another region, or change the design to an individually randomised trial with clinic-level adjustment.

### Vary one assumption: the dropout

If dropout is 25 percent rather than 15 percent, the individual sample per arm becomes 273 / 0.75 = 364. With the design effect, the required individual sample is 364 * 6.348 = 2,311, and the number of clinics per arm is 19.3, so 20 clinics per arm and 40 in total. The trial remains feasible, but the cost rises by about a third.

### The decision

The authority approves the trial with 30 clinics and 3,600 participants, with a recruitment monitoring plan and a contingency to add clinics if the average recruitment falls below 100 per clinic. The protocol states that the primary claim is superiority and that the MCID interpretation is secondary. The sensitivity table is included in the protocol and the funding request.

The example shows the central lesson of sample size: the calculation is not a single number but a set of assumptions and a feasibility judgement. The base case is feasible; the plausible variations are not all feasible. The protocol must state which variation would trigger a redesign.

## 7. Code, query, or calculation

The code below implements sample-size calculations for continuous, binary, and survival outcomes, the design effect for clustering, and the non-inferiority calculation. It uses only the Python standard library.

```python
import math
from statistics import NormalDist

Z = NormalDist().inv_cdf


def n_mean(delta, sd, alpha=0.05, power=0.90, ratio=1.0):
    """Participants in arm 1 for a two-arm mean difference."""
    za = Z(1 - alpha / 2)
    zb = Z(power)
    return (za + zb) ** 2 * sd ** 2 * (1 + 1 / ratio) / delta ** 2


def n_prop(p0, p1, alpha=0.05, power=0.90, ratio=1.0):
    """Participants in arm 1 for a two-arm risk difference."""
    za = Z(1 - alpha / 2)
    zb = Z(power)
    return (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1) / ratio) / (p1 - p0) ** 2


def n_events_survival(hr, alpha=0.05, power=0.90, allocation=0.5):
    """Number of events for a two-arm survival trial (Schoenfeld)."""
    za = Z(1 - alpha / 2)
    zb = Z(power)
    return (za + zb) ** 2 / (math.log(hr) ** 2) / (allocation * (1 - allocation))


def design_effect(m, icc, cv=0.0):
    """Design effect for a cluster trial with equal or variable cluster sizes."""
    return 1 + ((1 + cv ** 2) * m - 1) * icc


def n_ni(delta, sd, margin, alpha=0.025, power=0.90):
    """Participants per arm for a non-inferiority trial with expected delta."""
    za = Z(1 - alpha)
    zb = Z(power)
    return 2 * (za + zb) ** 2 * sd ** 2 / (margin - delta) ** 2


def inflate(n, dropout):
    return math.ceil(n / (1 - dropout))


if __name__ == "__main__":
    n = n_mean(5, 18, power=0.90)
    print(f"continuous 5 mmHg, SD 18: n/arm={math.ceil(n)}")
    print(f"  80% power: n/arm={math.ceil(n_mean(5, 18, power=0.80))}")
    print(f"  dropout 15%: n/arm={inflate(math.ceil(n), 0.15)}")

    print(f"binary 30% vs 42%: n/arm={math.ceil(n_prop(0.30, 0.42))}")
    print(f"survival HR 0.75, 90% power: events={math.ceil(n_events_survival(0.75))}")
    print(f"survival HR 0.75, 80% power: events={math.ceil(n_events_survival(0.75, power=0.80))}")

    for icc in (0.02, 0.04, 0.08):
        de = design_effect(120, icc, cv=0.35)
        required = math.ceil(n) * de
        print(f"ICC={icc}: DE={de:.3f} individuals/arm={required:.0f} "
              f"clusters/arm={math.ceil(required/120)}")

    print(f"NI margin 2 mmHg: n/arm={math.ceil(n_ni(0, 18, 2))}")
    print(f"NI margin 3 mmHg: n/arm={math.ceil(n_ni(0, 18, 3))}")
```

Expected output:

```text
continuous 5 mmHg, SD 18: n/arm=273
  80% power: n/arm=204
  dropout 15%: n/arm=322
binary 30% vs 42%: n/arm=331
survival HR 0.75, 90% power: events=508
survival HR 0.75, 80% power: events=380
ICC=0.02: DE=3.674 individuals/arm=1003 clusters/arm=9
ICC=0.04: DE=6.348 individuals/arm=1733 clusters/arm=15
ICC=0.08: DE=11.696 individuals/arm=3193 clusters/arm=27
```

The output illustrates four points.

- The 90 percent power requirement adds 34 percent to the sample compared with 80 percent power (273 versus 204 per arm). Power is expensive.
- The dropout inflation adds 18 percent (273 to 322).
- The ICC has a larger effect than either: moving from 0.02 to 0.08 more than triples the required sample. The ICC is the most consequential assumption in a cluster trial.
- The non-inferiority sample size depends on the margin. A 3 mmHg margin requires 757 per arm; a 2 mmHg margin requires 1,703. The margin is the expensive parameter.

The design-effect values in the output differ slightly from the worked example because the worked example uses a CV of 0.35 and the code uses the same CV; the difference is rounding in the intermediate steps. The conclusion is unchanged: the ICC dominates.

### Verifying the calculation by simulation

An analytical formula is an approximation. A short simulation can confirm that the planned sample has the intended power. The code below simulates the continuous outcome case with the worked example's inputs.

```python
import math
import random


def simulate_power(n_per_arm, delta, sd, reps=2000, seed=1):
    """Simulate power for a two-arm mean difference with a z-test."""
    random.seed(seed)
    significant = 0
    for _ in range(reps):
        control = [random.gauss(0, sd) for _ in range(n_per_arm)]
        treated = [random.gauss(delta, sd) for _ in range(n_per_arm)]
        se = math.sqrt(2 * sd * sd / n_per_arm)
        z = (sum(treated) / n_per_arm - sum(control) / n_per_arm) / se
        if abs(z) > 1.959964:
            significant += 1
    return significant / reps


if __name__ == "__main__":
    print("n=273, delta=5, SD=18:", simulate_power(273, 5, 18))
    print("n=204, delta=5, SD=18:", simulate_power(204, 5, 18))
    print("n=273, delta=4, SD=18:", simulate_power(273, 4, 18))
```

Expected output (Monte Carlo values may differ slightly with a different seed):

```text
n=273, delta=5, SD=18: 0.907
n=204, delta=5, SD=18: 0.797
n=273, delta=4, SD=18: 0.7305
```

The simulation confirms the analytical calculation: 273 per arm gives about 90 percent power for a 5 mmHg difference, 204 per arm gives about 80 percent, and 273 per arm gives about 73 percent power for a 4 mmHg difference. The last result is the most useful: it shows how quickly power falls when the true effect is smaller than planned.

A simulation is not a substitute for the analytical calculation in a protocol, but it is a useful check. It is also the only way to verify the operating characteristics of a complex design.

### A precision calculation

For a continuous outcome, the half-width of the 95 percent confidence interval is:

```text
half_width = 1.96 * sqrt(2 * SD^2 / n_eff)
```

For the worked example, with an effective sample of 241 per arm and an SD of 18:

```text
half_width = 1.96 * sqrt(2 * 324 / 241)
           = 1.96 * 1.640
           = 3.21 mmHg
```

The calculation shows why a trial can be powered for superiority and still not demonstrate that the effect exceeds the MCID. The precision is adequate to distinguish the effect from zero, but not to establish a margin above 3 mmHg.

## 8. Data, standards, and regulatory context

### Binding requirements

**ICH E9.** The Statistical Principles for Clinical Trials (E9, 1998) is the foundational guideline. It requires the number of participants to be justified by the expected effect, the variability, the significance level, and the power. It is guidance, not statute, but it is the standard of statistical review in the three major jurisdictions.

**ICH E9(R1).** The estimand addendum (Step 4, 20 November 2019; EMA Step 5 effective 30 July 2020; FDA guidance May 2021) requires the estimand to be defined before the sample size. The sample size depends on the estimand: a treatment-policy estimand may require a larger sample than a hypothetical estimand because of dilution by non-adherence.

**EU Clinical Trials Regulation.** Regulation (EU) No 536/2014 requires the protocol to include a justification of the sample size. It is binding.

**US investigational regulations.** 21 CFR Part 312 (drugs) and Part 812 (devices) require the protocol to include the number of participants and the rationale. The regulations are binding, but they do not prescribe a formula.

### Formal guidance

**FDA non-inferiority guidance.** "Non-Inferiority Clinical Trials to Establish Effectiveness" (November 2016) describes the M1 and M2 margins, assay sensitivity, and the constancy assumption. It requires the margin to be justified from historical evidence. It is guidance, not regulation.

**FDA adaptive designs guidance.** "Adaptive Designs for Clinical Trials of Drugs and Biologics" (December 2019) requires prespecification of adaptation rules, a data access plan, and simulation reports. It does not prescribe a sample-size formula, but it requires the operating characteristics to be documented.

**ICH E10.** The Choice of Control Group guideline (2000) describes assay sensitivity and the conditions under which a non-inferiority or external-control design can support a claim. It is guidance.

**CONSORT 2025.** The reporting guideline requires the sample-size calculation to be reported with the assumptions, the effect size, the power, and the method. It is a convention enforced by journals.

**SPIRIT 2025.** The protocol guideline requires the sample-size calculation and the recruitment plan. It is a convention.

### What the standards constrain and what they leave open

| Instrument | Constrains | Leaves open |
|---|---|---|
| ICH E9 | The justification of the sample size | The specific effect and variance |
| ICH E9(R1) | The estimand before the sample size | The strategy and the dilution |
| FDA NI guidance | The margin justification | The historical data and the constancy assumption |
| FDA adaptive guidance | The operating characteristics | The adaptation rule |
| CONSORT 2025 | The reporting of the calculation | The assumptions |
| SPIRIT 2025 | The protocol content | The design |

The standards require the calculation to be explicit, not to be correct. The correctness depends on the assumptions, and the assumptions depend on the clinical and commercial context.

### Questions a statistical reviewer will ask

- What is the primary estimand, and how does it relate to the sample size?
- What is the target effect, and what is the evidence for it?
- What is the variance or event rate, and what is the evidence for it?
- Is the alpha one- or two-sided, and why?
- Is the power 80 or 90 percent, and why?
- How is the allocation ratio justified?
- How is dropout handled, and what is the evidence for the rate?
- How is clustering handled, and what is the ICC?
- How is multiplicity controlled?
- Is the sample feasible, and what is the recruitment plan?
- What is the sensitivity of the result to the assumptions?
- What is the minimum detectable effect at the planned sample?

A protocol that answers these questions is reviewable. A protocol that reports only the final number is not.

### What a protocol must contain

- The estimand and the decision rule.
- The primary outcome and the effect measure.
- The target effect and its justification.
- The variance or event rate and its justification.
- The alpha, power, and allocation ratio.
- The sample-size formula and arithmetic.
- The dropout and clustering adjustments.
- The multiplicity and interim plans.
- The feasibility and recruitment plan.
- The sensitivity table.

## 9. Industry practice and case evidence

### Sample size in practice

Industry sample-size calculations are often driven by the smallest effect that will support the commercial claim, not by the MCID. A drug that is expected to capture a large market may be developed on a small effect; a drug in a crowded market may need a larger effect to differentiate. The sample size is therefore a commercial decision as well as a statistical one.

The documented practice is that sample sizes in confirmatory trials have increased over time, particularly in oncology and cardiology, because the effects of new treatments are smaller relative to the standard of care. The inference is that the marginal cost of evidence has risen, and the design must be more efficient to remain viable.

**Lesson.** The effect size should be justified by the claim and the market, not by the sample size the company can afford.

### The RECOVERY trial

The RECOVERY trial randomised 2,104 participants to dexamethasone and 4,321 to usual care. The trial was large, pragmatic, and designed to detect a clinically important mortality difference. The overall 28-day mortality was 22.9 percent versus 25.7 percent (age-adjusted rate ratio 0.83, 95 percent confidence interval 0.75 to 0.93). Among participants receiving invasive mechanical ventilation, mortality was 29.3 percent versus 41.4 percent (rate ratio 0.64, 95 percent confidence interval 0.51 to 0.81).

The documented fact is the sample size and the effect. The inference is that the trial was powered for a modest overall effect but was large enough to detect a large effect in the ventilated subgroup. The subgroup analysis was prespecified, but the trial was not powered for the subgroup; the result is credible because the effect was large and biologically plausible.

**Lesson.** A trial powered for the overall population may detect a large subgroup effect, but the subgroup claim should be labelled as supportive unless the trial was powered for the interaction.

### The Apple Heart Study

The Apple Heart Study enrolled 419,297 participants. It was not powered for a clinical outcome; it was a single-arm study of detection performance. The large sample provided precise estimates of the notification rate and the confirmation rate, but it could not estimate a mortality or stroke effect.

The documented fact is the sample size and the descriptive estimates. The inference is that a large sample can provide precise descriptive estimates without supporting a causal claim. The sample size must match the claim, not only the precision.

**Lesson.** A large sample does not substitute for a comparison. Precision without a counterfactual is not causal evidence.

### Cluster trials in practice

Cluster randomised trials often have sample-size problems. Reviews have found that a minority account for clustering in their calculations, and that the ICC is often assumed rather than estimated. The consequence is underpowered trials and inflated type I error.

The documented fact is the prevalence of the problem in the methodological literature. The inference is that cluster designs are often chosen for feasibility without a corresponding investment in the statistical design. The mitigation is to estimate the ICC from prior data, to include the upper bound in the sensitivity analysis, and to report the ICC in the results.

**Lesson.** The ICC is the most consequential assumption in a cluster trial. It should be estimated, not assumed.

### Practice comparison

| Practice | Mature teams | Less mature teams | Consequence |
|---|---|---|---|
| Effect source | MCID, pilot, literature | Round number | Under- or over-powered |
| Variance | Upper bound of pilot CI | Pilot point estimate | Underpowered |
| ICC | Estimated and varied | Assumed zero | Severe underpowering |
| Dropout | Retention data | Optimistic 5% | Too few evaluable |
| Multiplicity | Prespecified strategy | Ignored | Inflated false positives |
| Feasibility | Recruitment model | Site assumption | Trial extension |
| Sensitivity | Scenario table | Single number | Hidden fragility |
| Review | Independent statistician | Self-review | Missed errors |

The table is an inference from public protocols and methodological reviews, not a systematic survey. It should be read as a description of common practice, not as a ranking of companies.

### What companies actually do

The following practices are inferred from public protocols and guidance.

1. **Sample size is reviewed at phase gates.** The calculation is revisited when the effect, variance, or recruitment assumptions change.
2. **Sensitivity tables are standard in mature protocols.** A single number is no longer acceptable for a confirmatory trial.
3. **Blinded sample-size reviews are used.** The pooled event rate is reviewed at an interim to detect a wrong assumption.
4. **Simulation is used for adaptive and cluster designs.** Analytical formulas are insufficient for complex designs.
5. **Feasibility is assessed before commitment.** Recruitment rates and site activation are modelled, not assumed.

### Where practice falls short

- **Borrowed effects.** The effect is copied from a trial in a different population.
- **Optimistic variance.** The SD is taken from a small pilot without an upper bound.
- **Ignored clustering.** The design effect is omitted.
- **No multiplicity control.** Multiple endpoints are tested at alpha 0.05.
- **No feasibility check.** The sample size is computed without a recruitment plan.
- **No sensitivity table.** The protocol reports a single number.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Underpowered trial | Effect borrowed from a different population | Optimistic effect | False negative | Sensitivity table; feasibility review |
| Overpowered trial | Effect much larger than MCID | Inefficient design | Wasted resources | Reassess the effect |
| Wrong outcome | Power calculated for a surrogate | Surrogate has a larger effect | Underpowered for the clinical outcome | Power the clinical outcome |
| Ignored clustering | Sample size equals an individual trial | Design effect omitted | Severe underpowering | ICC and design effect |
| Too few clusters | Cluster count fixed by sites | Cluster size cannot compensate | Unstable variance | Feasibility check |
| Multiplicity ignored | Several endpoints or arms | Family-wise error inflation | False positive | Alpha allocation |
| Dropout underestimated | Pilot retention higher than real world | Attrition | Underpowered analysis | Inflate for realistic dropout |
| Event rate wrong | Control rate from an old trial | Secular trend | Too few events | Blinded review; sensitivity |
| Non-inferiority margin unjustified | Margin chosen for convenience | No historical anchor | Uninterpretable result | Historical effect; assay sensitivity |
| Interim not simulated | No operating characteristics | Adaptive rule untested | Type I error or power loss | Simulation report |
| Subgroup underpowered | Subgroup claim without interaction power | Multiplicity and prevalence | False subgroup claim | Power the interaction |
| Precision ignored | Only power reported | Decision needs a bound | Estimate too imprecise | Precision calculation |
| Recruitment infeasible | No screening-to-enrolment ratio | Overoptimistic sites | Trial extends or fails | Feasibility study |
| No sensitivity table | Single number | Fragile assumptions | Unanticipated failure | Scenario table |
| No independent review | Statistician reviews own work | Undetected arithmetic error | Protocol amendment | Independent check |
| Unreported software | Number cannot be reproduced | Opaque calculation | Review query | Report software and version |

### Red flags in a sample-size section

- The effect size has no source.
- The variance is a round number with no justification.
- The dropout rate is lower than the pilot.
- The ICC is assumed to be zero in a cluster trial.
- The number of clusters is not reported.
- The multiplicity strategy is missing.
- The non-inferiority margin is not justified.
- The sample size is not linked to the decision rule.
- The feasibility assessment is absent.
- The sensitivity table is absent.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The effect size is plausible | Power depends on it | Optimistic borrowing | Sensitivity table; pilot CI |
| The variance is correct | Precision depends on it | Small pilot; different population | Upper bound; blinded review |
| The event rate is correct | Events drive survival power | Secular trend; population shift | Blinded review; registry data |
| The ICC is correct | Design effect depends on it | Few clusters in the pilot | Sensitivity over a range |
| The cluster size is achievable | Number of clusters depends on it | Recruitment shortfall | Feasibility monitoring |
| Dropout is accurately estimated | Evaluable sample depends on it | Differential attrition | Retention data; sensitivity |
| Adherence is sufficient | ITT effect depends on it | Side effects; complexity | Pilot adherence |
| Alpha is correctly specified | Type I error depends on it | Multiplicity; interim looks | Alpha budget |
| Power is correctly specified | Sample size depends on it | Underpowered design | Power review |
| The design effect captures clustering | Variance depends on it | Contamination; network effects | Simulation; sensitivity |
| Recruitment is feasible | The trial can be delivered | Site activation delay | Feasibility study |
| The decision rule is stable | The sample size is linked to it | Changing claim | Protocol lock |
| The analysis method matches the design | The variance estimate depends on it | Wrong model | Pre-specify the analysis |
| The multiplicity strategy is fixed | Alpha allocation depends on it | Post hoc testing | SAP lock |
| The interim plan is fixed | Alpha spending depends on it | Unplanned look | DMC charter |
| The variance model is correct | The information calculation depends on it | Wrong distribution | Simulation check |
| The effect is constant over time | The hazard ratio assumption | Non-proportional hazards | Time-varying analysis |
| The population is stable | The event rate depends on it | Case-mix drift | Interim review |
| The analysis model is prespecified | The variance estimate depends on it | Model uncertainty | SAP lock |

### Limits of this topic

This topic covers the arithmetic and the main design adjustments. It does not cover the full theory of adaptive designs, the details of Bayesian sample-size determination, or the design of complex platform trials. Those are specialised topics.

The formulas are approximations. For small samples, extreme proportions, or survival outcomes with few events, exact or simulation-based methods are preferable. The worked example is designed to be reproducible by hand, not to be copied into a protocol without a statistician's review.

The case evidence is drawn from public sources and is not a systematic review. The inferences about company practice are labelled as inferences. The regulatory statements are educational summaries, not legal advice.

## 12. Exercises

### Exercise 1 — Continuous outcome (calculation, 25 minutes)

**Deliverable:** sample size per arm for a mean difference of 0.4 SD, alpha 0.05 two-sided, power 80 percent, equal allocation, 10 percent dropout.

**Acceptance criteria:** formula correct; arithmetic correct; dropout applied; assumptions stated.

**Time:** 25 minutes.

### Exercise 2 — Binary and survival outcomes (calculation, 35 minutes)

**Deliverable:** sample size for (a) a risk difference from 20 percent to 15 percent at 90 percent power and (b) a hazard ratio of 0.80 at 80 percent power with a 30 percent event probability.

**Acceptance criteria:** both formulas correct; event count and participant count distinguished; assumptions stated.

**Time:** 35 minutes.

### Exercise 3 — Cluster adjustment (calculation, 30 minutes)

**Deliverable:** the number of clusters per arm for an individual sample of 400 per arm, an average cluster size of 100, an ICC of 0.05, and a CV of 0.3.

**Acceptance criteria:** design effect correct; cluster count correct; feasibility check stated.

**Time:** 30 minutes.

### Exercise 4 — Non-inferiority margin (design, 40 minutes)

**Deliverable:** a justification for a non-inferiority margin of 2 percentage points for a new antibiotic, using a historical trial that showed a 10 percentage point advantage over placebo, and the corresponding sample size.

**Acceptance criteria:** M1 and M2 described; margin justified; sample size computed; assay sensitivity discussed.

**Time:** 40 minutes.

### Exercise 5 — Multiplicity (design, 30 minutes)

**Deliverable:** an alpha allocation for a trial with two primary endpoints and three doses, with the resulting sample-size implications.

**Acceptance criteria:** strategy named; alpha allocated; sample-size effect described; assumptions stated.

**Time:** 30 minutes.

### Exercise 6 — Precision (calculation, 25 minutes)

**Deliverable:** the sample size needed to estimate a prevalence of 20 percent with a 95 percent confidence interval half-width of 2 percentage points.

**Acceptance criteria:** formula correct; arithmetic correct; the difference from a power calculation explained.

**Time:** 25 minutes.

### Exercise 7 — Sensitivity table (application, 35 minutes)

**Deliverable:** a sensitivity table for a trial with a base effect of 10 mmHg, SD 20, ICC 0.03, dropout 15 percent, varying the effect by 20 percent and the ICC by 50 percent.

**Acceptance criteria:** table complete; direction of change correct; the most influential assumption identified.

**Time:** 35 minutes.

### Exercise 8 — Critique a sample-size section (critique, 40 minutes)

**Deliverable:** a critique of a published sample-size section, identifying at least five missing or optimistic assumptions and their consequences.

**Acceptance criteria:** specific to the paper; each assumption linked to a consequence; a remediation proposed.

**Time:** 40 minutes.

## 13. Self-check answers

**Answer 1 (continuous).** A difference of 0.4 SD is delta = 0.4. n per arm = (1.96 + 0.8416)^2 * 2 / 0.4^2 = 7.849 * 2 / 0.16 = 98.1, so 99 per arm. With 10 percent dropout, 99 / 0.9 = 110 per arm.

**Answer 2 (binary and survival).** (a) p0 = 0.20, p1 = 0.15. n = (1.96 + 1.2816)^2 * [0.16 + 0.1275] / 0.0025 = 10.507 * 0.2875 / 0.0025 = 1,208.3, so 1,209 per arm. (b) Events = (1.96 + 0.8416)^2 / (log 0.8)^2 / 0.25 = 7.849 / 0.0498 / 0.25 = 630.4, so 631 events. With a 30 percent event probability, participants = 631 / 0.30 = 2,103.

**Answer 3 (cluster).** DE = 1 + ((1 + 0.09) * 100 - 1) * 0.05 = 1 + (109 - 1) * 0.05 = 1 + 5.4 = 6.4. Individuals per arm = 400 * 6.4 = 2,560. Clusters per arm = 2,560 / 100 = 25.6, so 26 clusters per arm.

**Answer 4 (non-inferiority).** The historical trial showed a 10 percentage point advantage over placebo. M1 is the entire effect, 10 points. M2 is the largest loss that is clinically acceptable, often set at 50 percent of M1, so 5 points. A margin of 2 points is more conservative than M2 and is justified if the new antibiotic has other benefits. The sample size for a 2-point margin with event rates near 20 percent is approximately (1.96 + 1.28)^2 * [0.16 + 0.16] / 0.0004 = 10.507 * 0.32 / 0.0004 = 8,406 per arm. Assay sensitivity requires that the trial could have detected the historical effect; the constancy assumption requires that the standard of care and population are similar.

**Answer 5 (multiplicity).** A reasonable strategy is a hierarchical test: the first primary endpoint at alpha 0.05, and the second only if the first is significant. The doses could be tested with a gatekeeping procedure or a Dunnett adjustment. If the endpoints are independent and both must be tested at alpha 0.025, the sample size increases by about 15 percent relative to a single endpoint at 0.05. If the doses are compared with a shared control, the sample size must account for the correlation and the multiplicity.

**Answer 6 (precision).** For a prevalence of 0.20 and a half-width of 0.02, n = (1.96^2 * 0.20 * 0.80) / 0.02^2 = (3.8416 * 0.16) / 0.0004 = 1,536.6, so 1,537 participants. The calculation is a precision calculation, not a power calculation; it does not involve an alternative hypothesis.

**Answer 7 (sensitivity).** The base case with a 10 mmHg effect, SD 20, ICC 0.03, and dropout 15 percent gives an individual sample of (1.96 + 1.28)^2 * 2 * 400 / 100 = 10.507 * 8 = 84.1, so 85 per arm. With a 20 percent smaller effect (8 mmHg), the sample is 10.507 * 800 / 64 = 131.3, so 132 per arm. With a 50 percent higher ICC (0.045), the design effect rises, and the sample rises proportionally. The effect size is usually the most influential assumption; the ICC is the most influential in a cluster trial.

**Answer 8 (critique rubric).** A strong critique identifies the missing source of the effect, the optimistic variance, the ignored clustering, the absent multiplicity control, and the unrealistic dropout. It links each to a consequence for power or interpretation. A weak critique says the sample size is "too small" without the assumptions.

## 14. Interview questions

**Question 1. What are the inputs to a sample-size calculation?**

A strong answer: the effect size, the variance or event rate, alpha, power, the allocation ratio, the dropout, the clustering, and the multiplicity. A weak answer names only the effect and the sample size.

**Question 2. Why does a smaller effect require a disproportionately larger sample?**

A strong answer: because the sample size is proportional to 1/effect^2; halving the effect quadruples the sample. A weak answer says "smaller effects are harder to detect" without the squared relationship.

**Question 3. What is the design effect, and when does it apply?**

A strong answer: it is 1 + (m - 1) * ICC, applied when the unit of allocation is a cluster; it inflates the sample to recover the information lost to within-cluster correlation. A weak answer defines it without the application.

**Question 4. How do you justify a non-inferiority margin?**

A strong answer: from the historical effect of the active control, with M1 as the entire effect and M2 as the largest acceptable loss, plus assay sensitivity and the constancy assumption. A weak answer says "a margin of 10 percent is standard."

**Question 5. How do you handle multiple endpoints?**

A strong answer: prespecify a hierarchy, a gatekeeping procedure, or an alpha allocation; power the trial for the strategy; report all endpoints. A weak answer says "Bonferroni" without the consequences.

**Question 6. What is the difference between power and precision?**

A strong answer: power is the probability of rejecting a null hypothesis; precision is the width of an estimate. A study can be powered but imprecise, or precise but not powered. A weak answer treats them as synonyms.

**Question 7. What would you do if the sample size is not feasible?**

A strong answer: revisit the effect, the endpoint, the design, the population, or the claim; consider a different design or a consortium; state the minimum detectable effect and whether it is decision-relevant. A weak answer says "enrol more participants" or "accept the risk."

**Question 8. Why is the ICC the most important assumption in a cluster trial?**

A strong answer: because the design effect is 1 + (m - 1) * ICC, and the ICC can vary widely; a doubling of the ICC can double the sample, and with a fixed number of clusters it can make the trial infeasible. A weak answer says "clustering matters."

**Question 9. How do you size a study for a subgroup claim?**

A strong answer: power the interaction, not the subgroup main effect; account for the subgroup prevalence and the multiplicity; prespecify the subgroup. A weak answer says "enrol enough subgroup patients."

**Question 10. What is a blinded sample-size review?**

A strong answer: a review of the pooled event rate or variance without unblinding the treatment effect, used to detect a wrong assumption and re-estimate the sample size. A weak answer confuses it with an unblinded interim analysis.

## 15. Key terms

- **Alpha:** the prespecified type I error rate.
- **Alpha spending:** allocating alpha across interim analyses.
- **Allocation ratio:** the ratio of participants assigned to each arm.
- **Assay sensitivity:** the ability of a trial to distinguish an effective treatment from an ineffective one.
- **Attrition inflation:** dividing the sample size by (1 - dropout).
- **Blinded sample-size review:** a review of pooled data without unblinding.
- **Cluster size variation:** the distribution of cluster sizes around the mean.
- **Coefficient of variation (CV):** the SD of cluster size divided by the mean.
- **Design effect (DE):** the sample-size multiplier for clustering.
- **Dropout:** loss of participants before the primary outcome.
- **Effect size:** the magnitude of the difference or association of interest.
- **Event rate:** the proportion of participants with a binary outcome.
- **Events per variable:** the number of outcome events per predictor.
- **Family-wise error rate (FWER):** the probability of at least one false positive across a family.
- **False discovery rate (FDR):** the expected proportion of false positives among rejected hypotheses.
- **Feasibility:** whether the study can be delivered with the available resources.
- **Hazard ratio (HR):** the ratio of hazards between groups.
- **Information fraction:** the proportion of planned events observed at an interim.
- **Intracluster correlation coefficient (ICC):** the proportion of variance attributable to between-cluster variation.
- **Minimum clinically important difference (MCID):** the smallest effect that would change practice.
- **Minimum detectable effect (MDE):** the smallest effect a study can detect with a given power.
- **Multiplicity:** the inflation of type I error from multiple tests.
- **Non-adherence:** failure to take or deliver the intervention as specified.
- **Non-inferiority margin:** the largest acceptable loss of effect relative to an active comparator.
- **Overdispersion:** variance greater than the mean in count data.
- **Power:** the probability of rejecting the null hypothesis when a specified alternative is true.
- **Precision:** the width of a confidence interval.
- **Recruitment rate:** the number of participants enrolled per site per month.
- **Schoenfeld formula:** the formula for the number of events required in a survival trial.
- **Screening-to-enrolment ratio:** the number of patients screened per participant enrolled.
- **Standard deviation (SD):** the variability of a continuous outcome.
- **Superiority margin:** the smallest effect that would be considered clinically meaningful.

## 16. Sources

All URLs accessed 25 October 2026.

### Guidelines

- ICH, E9 Statistical Principles for Clinical Trials, Step 4, 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
- ICH, E9(R1) Addendum on Estimands and Sensitivity Analysis, Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
- ICH, E10 Choice of Control Group and Related Issues in Clinical Trials, Step 4, 20 July 2000. https://database.ich.org/sites/default/files/E10_Guideline.pdf
- FDA, "Non-Inferiority Clinical Trials to Establish Effectiveness," final guidance, November 2016. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/non-inferiority-clinical-trials
- FDA, "Adaptive Designs for Clinical Trials of Drugs and Biologics," final guidance, December 2019. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry
- Hopewell S, Chan A-W, Collins GS, et al. CONSORT 2025 statement. BMJ 2025;389:e081123. https://doi.org/10.1136/bmj-2024-081123
- Chan A-W, Boutron I, Hopewell S, et al. SPIRIT 2025 statement. BMJ 2025;389:e081477. https://doi.org/10.1136/bmj-2024-081477

### Methods

- Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. Biometrics 1983;39:499-503. https://doi.org/10.2307/2531021
- Rutterford C, Copas A, Eldridge S. Methods for sample size determination in cluster randomized trials. Int J Epidemiol 2015;44:1051-1067. https://doi.org/10.1093/ije/dyv113
- Hemming K, Girling AJ, Sitch AJ, et al. Sample size calculations for cluster randomised controlled trials with a fixed number of clusters. BMC Med Res Methodol 2011;11:102. https://doi.org/10.1186/1471-2288-11-102
- Campbell MK, Elbourne DR, Altman DG. CONSORT statement: extension to cluster randomised trials. BMJ 2004;328:702-708. https://doi.org/10.1136/bmj.328.7441.702
- Friedman LM, Furberg CD, DeMets DL, et al. Fundamentals of Clinical Trials. 5th edition. Springer, 2015.

### Case evidence

- RECOVERY Collaborative Group. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med 2021;384:693-704. https://doi.org/10.1056/NEJMoa2021436
- Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. N Engl J Med 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183

### Further reading

- Schulz KF, Grimes DA. Sample size calculations in randomised trials: mandatory and mystical. Lancet 2005;365:1348-1353. https://doi.org/10.1016/S0140-6736(05)61034-3
- Bland JM. The tyranny of power: is there a better way to calculate sample size? BMJ 2009;339:b3985. https://doi.org/10.1136/bmj.b3985
- Eldridge SM, Ashby D, Kerry S. Sample size for cluster randomized trials: effect of coefficient of variation of cluster size and analysis method. Int J Epidemiol 2006;35:1292-1300. https://doi.org/10.1093/ije/dyl129
- Julious SA. Sample Sizes for Clinical Trials. Chapman and Hall/CRC, 2009.
- Stallard N. Optimal adaptive designs for early-phase clinical trials. In: Adaptive Design Methods in Clinical Trials. Chapman and Hall/CRC, 2016.
- Whitehead J. The Design and Analysis of Sequential Clinical Trials. Wiley, 1997.
- Jennison C, Turnbull BW. Group Sequential Methods with Applications to Clinical Trials. Chapman and Hall/CRC, 2000.
- Chow SC, Shao J, Wang H, Lokhnygina Y. Sample Size Calculations in Clinical Research. 3rd edition. Chapman and Hall/CRC, 2017.
- Spiegelhalter DJ, Abrams KR, Myles JP. Bayesian Approaches to Clinical Trials and Health-Care Evaluation. Wiley, 2004.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_bias-taxonomy.md`](05_bias-taxonomy.md)
- Next topic: [`07_estimands.md`](07_estimands.md)
- Related topic: [`02_trial-designs.md`](02_trial-designs.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
