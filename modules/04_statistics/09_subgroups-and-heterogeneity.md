# Subgroups and Heterogeneity

> Module 04 — Statistics and Causal Inference
> Topic 09 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 02, 05
> Version 2026.3

---

## 1. Why this topic exists

Every trial report and every observational analysis eventually faces the same question: does the treatment work differently for some patients? A drug that reduces stroke overall may be more effective in patients with diabetes. A program that reduces readmissions overall may have no effect in small hospitals. A device that improves outcomes on average may be harmful in one anatomical subgroup.

The question is legitimate and important. Averages can hide clinically meaningful variation, and a decision that applies an average effect to everyone can harm the patients for whom the effect is absent or reversed. But the search for subgroups is also the most reliable way to manufacture a false finding. Test enough subgroups and one will be "significant" by chance. Split a continuous variable at enough thresholds and one will produce a dramatic contrast. Report only the subgroup that worked and the analysis becomes a selection of noise.

The statistical issues are distinct:

- **Multiplicity.** Ten independent subgroup tests at alpha 0.05 produce at least one false positive with probability 1 - 0.95^10 = 0.40, if the null is true everywhere. Twenty tests raise it to 0.64.
- **Power.** A trial powered for the overall effect has much less power within a subgroup, because the subgroup is smaller. A non-significant subgroup result is usually a statement about sample size, not about absence of effect.
- **Scale.** An interaction can be present on the additive scale and absent on the multiplicative scale, or the reverse. A risk difference that differs between subgroups implies an interaction on the additive scale; a ratio that differs implies one on the multiplicative scale. Reporting one without the other can reverse the interpretation.
- **Selection.** Subgroups defined after seeing the data, or cutpoints chosen to maximize the contrast, produce estimates that are biased and intervals that are too narrow.
- **Credibility.** Even a statistically significant interaction can be a chance finding if it was not pre-specified, is not replicated, has no biological rationale, and is one of many comparisons.

The industry cost is concrete. A pharmaceutical sponsor uses a subgroup result to support a narrower label; a regulator asks for the interaction test and the pre-specification, and the claim fails. A device company markets a product for a subgroup that showed a favorable point estimate in a post hoc analysis; a confirmatory trial in that subgroup finds no effect. A payer carves out an exception for a subgroup based on a forest plot with eight comparisons and no multiplicity control.

The reporting standards address the issue directly. Consolidated Standards of Reporting Trials (CONSORT) 2010 item 12b requires the methods for subgroup analyses, and item 18 requires the results, distinguishing pre-specified from exploratory. The International Council for Harmonisation (ICH) E9 guideline requires subgroup analyses to be pre-specified and interpreted with caution. The credibility criteria popularized by Sun, Ioannidis, Agoritsas, Alba, and Guyatt in JAMA in 2014 provide a structured way to judge a subgroup claim.

This topic builds the mechanics: defining the question, pre-specifying subgroups, choosing the scale, testing interaction, controlling multiplicity, assessing credibility, and reporting a forest plot that does not mislead.

> **Note:** A subgroup analysis answers one of two different questions. "What is the effect in this subgroup?" is a description. "Does the effect differ between subgroups?" is an interaction test. A significant effect within a subgroup does not answer the second question, and a non-significant effect within a subgroup does not answer it either.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** effect modification from confounding and **state** why the distinction matters for adjustment.
2. **Pre-specify** a small set of subgroups with a direction and a rationale.
3. **Test** interaction on the additive and multiplicative scales, and **explain** why the two can disagree.
4. **Interpret** a within-subgroup p-value correctly and **explain** why it is not an interaction test.
5. **Control** multiplicity across subgroup tests with a pre-specified family and a Holm or gatekeeping procedure.
6. **Apply** the five credibility criteria to a published subgroup claim.
7. **Report** subgroup effects with intervals, a forest plot, and a clear statement of pre-specification.
8. **Recognize** data-driven cutpoints, sparse subgroups, and qualitative interactions.

Outcome 3 maps to Exercise 2 and the worked example. Outcome 6 maps to Exercise 5. Outcome 7 maps to Exercise 8.

## 3. The core mental model

Heterogeneity is a property of the treatment effect, not of the sample. The analysis asks whether the effect differs across a pre-specified characteristic, and the answer has two parts: the estimate within each subgroup, and the test of the difference between them.

```text
  QUESTION
  Does the effect differ by subgroup S?
        |
        |  pre-specify S, the direction, and the scale
        v
  ESTIMAND
  Effect in each level of S; difference in effects (interaction)
        |
        |  fit the model with treatment x S
        v
  ESTIMATES
  Subgroup-specific effects with intervals
  Interaction estimate with interval and p-value
        |
        |  multiplicity and credibility
        v
  JUDGMENT
  Pre-specified? Replicated? Biologically plausible?
  One of how many tests?
        |
        v
  REPORT: forest plot + interaction test + caveat
```

### 3.1 Effect modification is not confounding

A confounder is a common cause of treatment and outcome; it must be adjusted to estimate the effect. An effect modifier is a variable that changes the size of the treatment effect; it must not be adjusted away, because the variation is the finding.

| Structure | Relationship to treatment | Action | Example |
|---|---|---|---|
| Confounder | Causes treatment and outcome | Adjust | Prior severity affects treatment and outcome |
| Effect modifier | Changes the effect of treatment | Report by stratum; include interaction | Age changes the absolute benefit |
| Mediator | Caused by treatment | Do not adjust for the total effect | Adherence |
| Collider | Caused by treatment and outcome | Do not adjust | Hospitalization |

A variable can be both a confounder and an effect modifier. Age may affect who receives treatment and may also change the treatment effect. The analysis must adjust for the confounding and report the modification.

### 3.2 Additive and multiplicative scales

An interaction is scale-dependent. Suppose the risk difference is -10 percentage points in younger patients and 0 in older patients. On the additive scale, the difference between these effects is 10 points, which is an interaction. On the multiplicative scale, the risk ratio is 0.50 in the younger group and 1.00 in the older group, which is also an interaction. The two scales agree in direction here, but they need not.

A classic pattern is a treatment that halves the risk in every subgroup. The risk ratio is constant (no multiplicative interaction), but the absolute benefit is larger in the subgroup with the higher baseline risk (an additive interaction). A decision about population impact should use the additive scale; a decision about biological mechanism may use the multiplicative scale.

| Scale | Measure | Interaction means | Decision use |
|---|---|---|---|
| Additive | Risk difference | The absolute benefit differs | Budget, NNT, population impact |
| Multiplicative | Risk ratio, odds ratio, hazard ratio | The relative effect differs | Etiology, mechanism |
| Either | Statistical model | The interaction term is non-zero on that scale | Report both when they disagree |

### 3.3 The interaction test is not a subgroup p-value

A common error is to compare the p-value in one subgroup with the p-value in another, or to declare an interaction because one subgroup is significant and the other is not. The correct test is a single model with a treatment-by-subgroup interaction term. The interaction coefficient estimates the difference in effects, and its p-value tests whether that difference is distinguishable from zero.

The difference between "significant in one subgroup, not in the other" and "significant interaction" is not subtle. A trial can have a significant effect in men and a non-significant effect in women, with no evidence that the effects differ, because the women's subgroup is smaller. The interaction test accounts for the precision of both estimates.

### 3.4 The credibility criteria

Sun and colleagues proposed five criteria for judging a subgroup claim:

1. **Chance:** Is the interaction test significant, and is it robust to multiplicity?
2. **Consistency:** Is the effect replicated in other studies or datasets?
3. **A priori hypothesis:** Was the subgroup pre-specified, with a direction?
4. **Biological rationale:** Is there a plausible mechanism?
5. **Within-study versus between-study:** For meta-analyses, within-study comparisons are stronger.

A claim that satisfies one or two criteria is hypothesis-generating. A claim that satisfies four or five is credible enough to influence a decision, but it is still not proof.

### 3.5 Pre-specification is the main protection

The strongest protection against a false subgroup claim is to specify the subgroups, the direction, and the analysis before seeing the data. A pre-specified subgroup with a significant interaction is credible; a post hoc subgroup with the same p-value is a hypothesis.

The number of pre-specified subgroups should be small. A protocol with 30 subgroup hypotheses has not pre-specified anything meaningful; it has moved the multiplicity problem from the analysis to the protocol.

### 3.6 What heterogeneity can and cannot tell you

- It can tell you that the effect differs across a characteristic in this sample.
- It cannot tell you why, unless the characteristic is a plausible mechanism.
- It cannot tell you the effect in a subgroup that was not studied.
- It cannot rescue a trial that was underpowered overall.
- It cannot be established by a single study without replication.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Subgroup | A subset defined by a baseline characteristic | Defined after seeing the outcome | Group |
| Effect modification | A change in the effect across a characteristic | Confused with confounding | Effect |
| Interaction | A departure from additivity or multiplicativity | Tested with within-subgroup p-values | Model term |
| Additive interaction | A difference in risk differences | Ignored when ratios are reported | Risk difference |
| Multiplicative interaction | A difference in ratios | Reported as the only scale | Ratio |
| Qualitative interaction | The effect has opposite signs across subgroups | Confused with quantitative | Direction |
| Quantitative interaction | The effect differs in magnitude but not sign | Called qualitative | Magnitude |
| Within-subgroup p-value | A test of the effect within one subgroup | Used as an interaction test | Probability |
| Interaction p-value | A test of the difference between subgroup effects | Interpreted without multiplicity | Probability |
| Multiplicity | The inflation of error when many tests are run | Ignored in subgroup analyses | Probability |
| Family-wise error rate | The probability of at least one false positive in a family | Confused with per-test alpha | Probability |
| Holm procedure | A step-down multiplicity adjustment | Applied without a pre-specified family | Method |
| Gatekeeping | A hierarchical testing procedure | Used without a logical order | Method |
| Pre-specification | Defining the hypothesis before seeing the data | Claimed for a post hoc analysis | Design |
| Post hoc | Defined after seeing the data | Presented as confirmatory | Design |
| Credibility criteria | Five criteria for judging a subgroup claim | Treated as a score | Framework |
| Forest plot | A plot of subgroup estimates with intervals | Used without the interaction test | Figure |
| Cutpoint | A threshold used to dichotomize a continuous variable | Chosen to maximize the contrast | Value |
| Data-driven subgroup | A subgroup defined by the data | Presented as pre-specified | Design |
| Sparse subgroup | A subgroup with few events | Interpreted as no effect | Count |
| Power | The probability of detecting an effect | Assumed equal across subgroups | Probability |
| Baseline characteristic | A variable measured before randomization | A post-treatment variable | Variable |
| Treatment-by-subgroup interaction | The product term in the model | Omitted when subgroup effects are compared | Model term |
| Relative effect | A ratio measure | Compared across subgroups with different baseline risk | Ratio |
| Absolute effect | A difference measure | Omitted from the forest plot | Difference |
| Ecological fallacy | Inferring individual effects from group averages | Ignored in subgroup interpretation | Bias |
| Confirmatory subgroup | A pre-specified subgroup for a regulatory claim | Confused with exploratory | Design |
| Exploratory subgroup | A hypothesis-generating analysis | Presented as confirmatory | Design |
| Multiplicity adjustment | A procedure to control error rates | Applied only to primary endpoints | Method |
| Heterogeneity | Variation in effect across studies or subgroups | Confused with inconsistency | Variation |
| Credible subgroup | A subgroup claim that meets the criteria | Any significant subgroup | Judgment |

Two conventions in this topic:

- A subgroup effect is reported with its interval and the interaction p-value, not with the within-subgroup p-value alone.
- A forest plot includes the overall estimate, the subgroup estimates, the interaction p-value, and the number of events per subgroup.

> **Note:** "The effect was significant in women but not in men" is not a statement about effect modification. It is a statement about two p-values, and it can occur when the effects are identical but the sample sizes differ.

## 5. The mechanics, step by step

The procedure has twelve steps. Steps 1 to 3 define the question and the scale. Steps 4 to 7 estimate and test. Steps 8 to 12 judge and report.

### Step 1. Decide which question you are asking

**Input:** The decision and the clinical hypothesis.

**Output:** A statement of whether the analysis estimates the effect within a subgroup or tests whether the effect differs between subgroups.

**Why it exists:** The two questions have different estimands and different error rates. A report that conflates them cannot be interpreted.

| Question | Estimand | Method | Error control |
|---|---|---|---|
| What is the effect in patients over 75? | Effect in that subgroup | Subgroup-specific estimate with interval | Interval coverage |
| Does the effect differ by age? | Difference in effects | Treatment-by-age interaction | Multiplicity across tests |
| Should the label be restricted to patients over 75? | Effect in the subgroup, plus credibility | Subgroup estimate plus interaction plus replication | Regulatory pre-specification |

### Step 2. Pre-specify the subgroups

**Input:** Clinical knowledge, the mechanism, and the decision.

**Output:** A short list of baseline subgroups, each with a direction and a rationale.

**Why it exists:** Pre-specification is the main protection against a false claim. A post hoc subgroup is a hypothesis, not evidence.

| Criterion for a subgroup | Good practice | Bad practice |
|---|---|---|
| Timing | Measured at baseline | Measured after treatment |
| Number | A few, clinically motivated | Dozens, data-driven |
| Direction | Stated in advance | Discovered after the result |
| Rationale | Mechanism or prior evidence | "It looked interesting" |
| Analysis | Interaction test | Separate within-subgroup tests |

### Step 3. Choose the scale

**Input:** The effect measure and the decision.

**Output:** An additive scale, a multiplicative scale, or both, with a justification.

**Why it exists:** An interaction can be present on one scale and absent on the other. The choice determines the conclusion.

| Decision | Scale | Measure | Example |
|---|---|---|---|
| Budget impact | Additive | Risk difference, NNT | The absolute benefit is larger in high-risk patients |
| Biological mechanism | Multiplicative | Risk ratio, odds ratio, hazard ratio | The relative effect is constant across risk levels |
| Regulatory label | Either, pre-specified | The measure in the primary analysis | The interaction must be pre-specified |
| Clinical communication | Both | Absolute and relative | The patient needs both |

### Step 4. Fit the interaction model

**Input:** The outcome, the treatment, the subgroup variable, and the adjustment set.

**Output:** A model with a treatment-by-subgroup term and the interaction estimate.

**Why it exists:** The interaction term is the test of effect modification. The within-subgroup analyses are descriptive.

For a binary outcome with a risk difference:

```text
Risk difference in subgroup s = RD_s
Interaction = RD_1 - RD_0
SE(interaction) = sqrt(SE_1^2 + SE_0^2)
z = Interaction / SE(interaction)
```

For a ratio measure, fit the model on the log scale and test the ratio of ratios.

**Edge case:** In a randomized trial, the subgroup variable is balanced by randomization, so the interaction estimate is unconfounded. In an observational study, the subgroup variable may be a confounder, and the interaction estimate requires adjustment.

### Interaction models for different outcomes

| Outcome | Model | Interaction term | Scale |
|---|---|---|---|
| Binary, risk difference wanted | Binomial with identity link, or logistic plus standardization | Treatment × subgroup | Additive |
| Binary, ratio wanted | Logistic | Treatment × subgroup | Multiplicative (OR) |
| Continuous | Linear | Treatment × subgroup | Additive (mean difference) |
| Count | Poisson or negative binomial | Treatment × subgroup | Multiplicative (rate ratio) |
| Time-to-event | Cox | Treatment × subgroup | Multiplicative (HR) |
| Time-to-event, absolute wanted | Standardized survival or RMST | Treatment × subgroup | Additive |

A model with a treatment-by-subgroup term estimates the interaction directly. A model fitted separately in each subgroup does not, and comparing the two coefficients informally is not a test.

**The interaction estimate on the ratio scale.** For a logistic or Cox model, the interaction coefficient is the log ratio of the subgroup-specific ratios. Exponentiating it gives the ratio of odds ratios or hazard ratios. Its interval is on the same scale.

### Step 5. Estimate the subgroup effects

**Input:** The fitted model.

**Output:** An estimate and interval for each subgroup, on the decision scale.

**Why it exists:** The interaction test says whether the effects differ; the subgroup estimates say how large they are.

- Report the absolute effect in each subgroup, not only the relative effect.
- Report the number of patients and events in each subgroup.
- Do not report a subgroup estimate without its interval.
- Do not compare the significance of two subgroup estimates; compare the interaction.

### Power inside a subgroup

A trial powered at 80% for the overall effect has less power in a subgroup. If the subgroup is 30% of the sample, the variance of the subgroup estimate is roughly 1/0.30 = 3.3 times larger, and the power for an effect of the same size falls well below 80%. A non-significant subgroup result is therefore expected even when the treatment works equally well in the subgroup.

The practical consequence is that a subgroup analysis is usually underpowered unless the trial was designed for it. A negative subgroup result should be reported with its interval and described as inconclusive, not as evidence of no effect.

### Forest plot template

| Row | Estimate | 95% CI | Events / n | Interaction p |
|---|---|---|---|---|
| Overall | -4.0 pp | (-8.4, +0.4) | 150 / 1,000 | — |
| Under 65 | -10.0 pp | (-16.9, -3.1) | 60 / 400 | 0.029 |
| 65 and older | 0.0 pp | (-5.7, +5.7) | 90 / 600 | 0.029 |
| Holm-adjusted | — | — | — | 0.291 |

The table is the forest plot in text form. It includes the overall estimate, the subgroups, the events, the unadjusted interaction, and the adjusted interaction. A figure without the adjusted p-value invites the reader to over-interpret the nominal one.

### Common fallacies in subgroup reporting

- **The significance fallacy:** comparing within-subgroup p-values instead of testing the interaction.
- **The multiplicity omission:** reporting the one significant subgroup out of many without an adjustment.
- **The post hoc promotion:** describing an exploratory subgroup as if it were pre-specified.
- **The scale switch:** reporting the additive interaction when it is significant and the multiplicative when it is not, without saying so.
- **The cutpoint search:** trying thresholds until the contrast is largest.
- **The power misread:** interpreting a wide subgroup interval as no effect.
- **The ecological leap:** applying a group average to every individual in the group.
- **The replication skip:** treating a single-study subgroup as established.

### Heterogeneity in meta-analysis

The subgroup logic extends to meta-analysis, where heterogeneity is measured across studies rather than across patient groups.

| Statistic | What it measures | Range | Interpretation |
|---|---|---|---|
| Cochran's Q | Weighted sum of squared deviations | 0 to infinity | A test of heterogeneity; low power with few studies |
| I-squared | Proportion of variance due to heterogeneity | 0% to 100% | 25% low, 50% moderate, 75% high (conventions) |
| Tau-squared | Between-study variance | 0 to infinity | The variance of the true effects |
| Prediction interval | The range of effects in a new study | Effect units | More informative than I-squared for a decision |

A subgroup analysis in a meta-analysis compares within-study effects, which is stronger than comparing study-level averages. The same credibility criteria apply: pre-specification, replication, mechanism, and chance.

### Regulatory use of subgroups

A regulator will consider a subgroup claim for a label only when the subgroup was pre-specified, the interaction is significant after controlling the family, the effect is clinically important, and the finding is supported by a plausible mechanism or an independent study. The bar is higher than in an exploratory publication because the label determines who receives the product.

| Claim type | Evidence standard | Example |
|---|---|---|
| Exploratory | An interaction test with a caveat | A hypothesis for a future trial |
| Supportive | Pre-specified, significant, consistent | A secondary analysis in a submission |
| Confirmatory | Pre-specified, adjusted, replicated | A label restriction |

### Branch points at a glance

| If the situation is | Then the default is | Because |
|---|---|---|
| One pre-specified subgroup | Interaction test, no adjustment | A single test |
| Many subgroups | Pre-specified family; Holm or gatekeeping | Multiplicity |
| Continuous covariate | Continuous interaction or pre-specified cutpoint | Avoids cutpoint bias |
| Sparse subgroup | Report the interval; no interpretation | Precision |
| Qualitative interaction claimed | Replicate before acting | Rare and consequential |
| Observational data | Adjust the interaction for confounders | The subgroup may be confounded |
| Meta-analysis | Within-study subgroup comparison | Avoids ecological bias |
| Regulatory claim | Pre-specified and replicated | The label determines access |

### Reporting checklist for a subgroup analysis

1. The question: effect within subgroups or modification across them.
2. The subgroups, with pre-specification status and direction.
3. The scale: additive, multiplicative, or both.
4. The interaction estimate and interval.
5. The subgroup estimates and intervals.
6. The number of patients and events per subgroup.
7. The multiplicity procedure and the adjusted p-value.
8. The credibility assessment, criterion by criterion.
9. The replication status.
10. The decision implication.

A report that omits items 2, 7, and 8 has omitted the information that determines whether the subgroup claim is credible. The estimates alone are not enough.

- Items 1 to 3 are design decisions and belong in the protocol.
- Items 4 to 7 are results and belong in the results section.
- Items 8 to 10 are interpretation and belong in the discussion.

### Step 6. Control multiplicity

**Input:** The family of subgroup tests.

**Output:** Adjusted p-values or a pre-specified testing hierarchy.

**Why it exists:** The family-wise error rate grows with the number of tests. A nominal 0.05 in one of ten tests is not evidence at the 0.05 level.

| Procedure | When | How | Note |
|---|---|---|---|
| Bonferroni | Few tests, any dependence | alpha / m | Conservative |
| Holm | Few tests, any dependence | Step-down | More powerful than Bonferroni |
| Gatekeeping | Ordered hypotheses | Test in sequence | Requires a logical order |
| Hierarchical | Primary then secondary | Only test secondary if primary passes | Common in trials |
| None | A single pre-specified test | No adjustment | Requires one test only |

**The family must be pre-specified.** Adjusting for all possible subgroups after the fact is not a solution; it is a post hoc choice of family.

### Step 7. Assess credibility

**Input:** The interaction result, the pre-specification, the replication, and the mechanism.

**Output:** A credibility judgment with a rationale.

**Why it exists:** Statistical significance is one of five criteria, not the only one.

| Criterion | Question | Strong evidence | Weak evidence |
|---|---|---|---|
| Chance | Is the interaction significant after multiplicity? | Adjusted p below alpha | Nominal p only |
| Consistency | Is it replicated? | Independent study or dataset | Single study |
| A priori | Was it pre-specified with a direction? | Protocol | Post hoc |
| Biology | Is there a mechanism? | Plausible pathway | No rationale |
| Within-study | Is the comparison within the same study? | Yes | Between-study only |

### Step 8. Distinguish qualitative from quantitative interaction

**Input:** The subgroup estimates.

**Output:** A statement of whether the effect changes magnitude or sign.

**Why it exists:** A qualitative interaction changes the decision for the subgroup; a quantitative interaction changes the size of the benefit.

| Type | Pattern | Decision implication | Example |
|---|---|---|---|
| Quantitative | Same direction, different magnitude | The treatment is beneficial everywhere, more in one group | Larger absolute benefit in high-risk patients |
| Qualitative | Opposite directions | The treatment is harmful in one subgroup | Benefit in one genotype, harm in another |
| None | Same magnitude | One effect applies to all | No interaction |

**Qualitative interactions are rare and require stronger evidence.** A claimed qualitative interaction from a single post hoc analysis is almost always a chance finding.

### Step 9. Check precision and power

**Input:** The subgroup sizes and event counts.

**Output:** A statement of the precision in each subgroup.

**Why it exists:** A non-significant subgroup result is often a power statement. The absence of evidence is not evidence of absence.

- Report the events per subgroup.
- Report the width of the subgroup interval.
- Do not interpret a wide interval as no effect.
- Consider a pre-specified pooled analysis or a larger trial if the subgroup question is important.

### Step 10. Avoid data-driven cutpoints

**Input:** A continuous variable proposed as a subgroup.

**Output:** A pre-specified cutpoint or a continuous interaction analysis.

**Why it exists:** Choosing the cutpoint that maximizes the contrast inflates the Type I error and produces an unreplicable finding.

| Approach | When | Risk |
|---|---|---|
| Pre-specified clinical cutpoint | A guideline threshold exists | None if pre-specified |
| Median split | No clinical threshold | Loses information; arbitrary |
| Data-driven cutpoint | Never for confirmatory claims | Inflated error; unreplicable |
| Continuous interaction | The effect may vary smoothly | Requires a functional form |

### Step 11. Report with a forest plot

**Input:** The subgroup estimates and the interaction test.

**Output:** A figure and a table that show the overall estimate, the subgroup estimates, the intervals, and the interaction p-value.

**Why it exists:** The forest plot is the standard way to show heterogeneity without hiding the uncertainty.

A complete forest plot has:

- The overall estimate at the top.
- One row per subgroup, with the estimate and interval.
- The number of events and patients per subgroup.
- The interaction p-value, unadjusted and adjusted.
- A note on pre-specification.

### Step 12. Plan replication and sensitivity

**Input:** The subgroup claim.

**Output:** A statement of what would confirm it.

**Why it exists:** A subgroup finding from one dataset is a hypothesis. Replication is the test.

- Pre-specify the replication in an independent cohort or trial.
- Test the interaction in the replication with the same scale.
- Report the result whether or not it confirms the original.
- Do not pool the original and replication unless both are pre-specified.

### Edge cases that change the conclusion

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| Sparse subgroup | Wide interval; unstable estimate | Few events | Report; do not interpret |
| Continuous variable dichotomized | Loss of power; cutpoint bias | The cutpoint is data-driven | Continuous interaction |
| Many subgroups | Multiplicity | More than a few tests | Pre-specify; adjust |
| Post-treatment subgroup | The subgroup is a mediator or collider | Measured after baseline | Exclude |
| Qualitative interaction | Opposite signs | The estimates cross the null | Replicate before acting |
| Ecological fallacy | Group average applied to individuals | The subgroup is geographic or aggregate | Individual-level analysis |
| Confounded subgroup in an observational study | The interaction is confounded | The subgroup is a confounder | Adjust |
| Subgroup with no overlap | One treatment is absent | Positivity violation | Do not estimate |

## 6. Worked example

### 6.1 The question and the subgroups

A program to reduce 12-month hospitalization is being considered for expansion. The overall effect is modest, and the clinical team believes it may be larger in younger patients, who are more likely to engage with the program. The team also examined nine other subgroups: sex, diabetes, prior hospitalization, depression, living alone, insurance type, rural residence, baseline cost, and number of medications.

The analysis asks two questions:

1. What is the effect in patients under 65 and in patients 65 or older?
2. Does the effect differ between the two age groups?

The second question is the interaction test. The first is descriptive.

### 6.2 The data

| Subgroup | Program: events / n | Risk | Usual care: events / n | Risk |
|---|---|---:|---|---:|
| Under 65 | 20 / 200 | 10.0% | 40 / 200 | 20.0% |
| 65 and older | 45 / 300 | 15.0% | 45 / 300 | 15.0% |
| Overall | 65 / 500 | 13.0% | 85 / 500 | 17.0% |

### 6.3 Subgroup risk differences

```text
Under 65:
RD = 0.10 - 0.20 = -0.10 = -10.0 percentage points
SE = sqrt(0.10×0.90/200 + 0.20×0.80/200)
   = sqrt(0.00045 + 0.00080) = sqrt(0.00125) = 0.03536
95% CI = -0.10 +/- 1.96 × 0.03536 = (-0.1693, -0.0307)

65 and older:
RD = 0.15 - 0.15 = 0.00 = 0.0 percentage points
SE = sqrt(0.15×0.85/300 + 0.15×0.85/300)
   = sqrt(0.000425 + 0.000425) = 0.02915
95% CI = 0.00 +/- 1.96 × 0.02915 = (-0.0571, 0.0571)

Overall:
RD = 0.13 - 0.17 = -0.04 = -4.0 percentage points
SE = sqrt(0.13×0.87/500 + 0.17×0.83/500)
   = sqrt(0.0002262 + 0.0002822) = 0.02255
95% CI = -0.0842 to +0.0042
```

The overall interval includes zero. The younger subgroup interval excludes zero. The older subgroup interval is centered on zero. A naive report would say "the program works in younger patients but not in older patients" and stop. That report is wrong for two reasons: it confuses a within-subgroup p-value with an interaction, and it ignores multiplicity.

### 6.4 The additive interaction

```text
Interaction = RD_young - RD_old = -0.10 - 0.00 = -0.10
SE = sqrt(0.03536^2 + 0.02915^2)
   = sqrt(0.00125 + 0.00085)
   = sqrt(0.00210) = 0.04583
z = -0.10 / 0.04583 = -2.182
two-sided p = 0.0291
95% CI = -0.10 +/- 1.96 × 0.04583 = (-0.1898, -0.0102)
```

The additive interaction is -10 percentage points, with an interval that excludes zero. On this scale, the evidence for effect modification is nominally significant.

### 6.5 The multiplicative interaction

```text
RR_young = 0.10 / 0.20 = 0.50
RR_old = 0.15 / 0.15 = 1.00
Ratio of RRs = 0.50 / 1.00 = 0.50
SE(log ratio) = sqrt(1/20 - 1/200 + 1/40 - 1/200
                     + 1/45 - 1/300 + 1/45 - 1/300)
              = sqrt(0.05 - 0.005 + 0.025 - 0.005
                     + 0.02222 - 0.00333 + 0.02222 - 0.00333)
              = sqrt(0.10278) = 0.32059
95% CI = exp(log(0.50) +/- 1.96 × 0.32059)
       = exp(-0.6931 +/- 0.6284)
       = (0.2667, 0.9373)
z = log(0.50) / 0.32059 = -2.162
p = 0.0306
```

The multiplicative interaction is also nominally significant, with a similar p-value. In this example the two scales agree. In other examples they can disagree, which is why both should be reported when the decision depends on the scale.

### 6.5b What the two scales say

| Scale | Younger | Older | Difference | 95% CI | p |
|---|---:|---:|---:|---|---:|
| Additive RD | -10.0 pp | 0.0 pp | -10.0 pp | (-19.0, -1.0) | 0.029 |
| Multiplicative RR | 0.50 | 1.00 | 0.50 | (0.27, 0.94) | 0.031 |

The two scales agree in this example, which is not guaranteed. A report that shows only one scale has not shown that the conclusion is scale-independent.

### 6.5c Precision

The younger subgroup has 60 events across 400 patients; the older subgroup has 90 events across 600 patients. The interaction interval is 18 percentage points wide. Even the nominally significant result is compatible with a difference as small as 1 point and as large as 19 points. A larger sample would be needed to estimate the modification precisely, and a replication would be needed to establish it.

### 6.6 Multiplicity: ten subgroup tests

The team examined ten subgroups. The age interaction p-value is 0.0291. Under a Holm procedure across the ten tests, with the other nine p-values assumed to be 0.60:

```text
Order the p-values: 0.0291 is smallest, then nine values at 0.60.
Step 1: adjusted p = 10 × 0.0291 = 0.291
Step 2: adjusted p = 9 × 0.60 = 5.4, truncated at 1.0
All adjusted p-values are at least 0.291.
```

The age interaction is no longer significant after multiplicity. The nominal p-value was one of ten, and one of ten tests at 0.05 is expected to be below 0.05 about 40% of the time when no effect exists.

### 6.7 Credibility assessment

| Criterion | Assessment | Verdict |
|---|---|---|
| Chance | Nominal p = 0.029; Holm-adjusted p = 0.291 | Fails |
| Consistency | No replication dataset | Unknown |
| A priori | Age was one of ten subgroups; the direction was stated | Partial |
| Biology | Younger patients may engage more, but the mechanism is speculative | Weak |
| Within-study | The comparison is within the same study | Passes |

Two of five criteria are met. The claim is hypothesis-generating, not confirmatory. A report that presents it as evidence of a subgroup effect is overstating the data.

### 6.8 Vary one assumption: a single pre-specified subgroup

Suppose age was the only pre-specified subgroup, with a stated direction and a biological rationale, and the analysis was planned before the data were seen. Then the nominal p-value of 0.0291 is the only test, the Holm adjustment is unnecessary, and the credibility assessment changes:

| Criterion | Assessment | Verdict |
|---|---|---|
| Chance | Nominal p = 0.029, no multiplicity | Passes |
| Consistency | No replication | Unknown |
| A priori | Pre-specified with direction | Passes |
| Biology | Plausible but not established | Partial |
| Within-study | Yes | Passes |

Three and a half of five criteria are met. The claim is stronger, but still not confirmatory, because it has not been replicated. The material change is the multiplicity: the same p-value supports a different conclusion depending on how many tests were run and whether the subgroup was pre-specified.

### 6.9 What to report

Report the overall effect, the two subgroup estimates with intervals, the additive and multiplicative interaction estimates, the unadjusted and adjusted interaction p-values, and the credibility assessment.

The primary estimate for the decision is the overall effect: **-4.0 percentage points (95% CI -8.4 to +0.4)**. The subgroup analysis is secondary and exploratory. The age interaction is **-10.0 percentage points (95% CI -19.0 to -1.0; nominal p = 0.029; Holm-adjusted p = 0.291)**.

The honest conclusion: "The overall effect was a 4.0-percentage-point reduction in 12-month hospitalization, with an interval that includes no effect. The effect was larger in patients under 65 (-10.0 points) than in older patients (0.0 points), but the interaction was one of ten subgroup tests and does not survive multiplicity adjustment. The finding is hypothesis-generating and should be tested in a pre-specified analysis or an independent cohort before it influences coverage."

A decision-maker who wants to act on the subgroup should require:

- The subgroup was pre-specified with a direction.
- The interaction survives the multiplicity adjustment.
- The effect is replicated in an independent dataset.
- The absolute effect is large enough to matter.
- The mechanism is plausible.
- The subgroup definition is stable and measurable in practice.

None of these is satisfied here except the mechanism, and the mechanism is speculative. The correct action is to plan a confirmatory analysis, not to carve out an exception.

The worked example also shows why the overall estimate remains primary. The overall effect is imprecise and includes no effect, but it is the estimate that applies to the population the program would serve. The subgroup analysis does not rescue the overall result; it identifies a question for the next study.

| Audience | What they need | What to show |
|---|---|---|
| Clinician | The absolute benefit in the patient in front of them | Subgroup RD with interval, and the caveat |
| Payer | The population impact | Overall RD, budget arithmetic |
| Regulator | The pre-specified interaction | Adjusted p-value and replication |
| Researcher | The hypothesis | The interaction estimate and the design for confirmation |

The same data support four different conversations. The reporting must give each audience the quantity it needs without implying more certainty than the data support.

## 7. Code, query, or calculation

The script computes the subgroup risk differences, the additive and multiplicative interactions, and a Holm adjustment across ten tests. It uses only the standard library.

**Assumptions:**

- The subgroup risks are estimated from the counts in Section 6.
- The interaction is tested on the additive scale with a normal approximation.
- The multiplicative interaction uses the log ratio of risk ratios.
- The family for the Holm adjustment is the ten subgroup tests, with nine assumed null p-values of 0.60.

```python
import math
from statistics import NormalDist

z = NormalDist().inv_cdf(0.975)

def rd(e1, n1, e0, n0):
    p1, p0 = e1 / n1, e0 / n0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return p1, p0, p1 - p0, se

young = rd(20, 200, 40, 200)
old = rd(45, 300, 45, 300)
overall = rd(65, 500, 85, 500)

for name, r in (("under 65", young), ("65 and older", old), ("overall", overall)):
    print(f"{name}: p1={r[0]:.4f} p0={r[1]:.3f} RD={r[2]:.4f} "
          f"CI=({r[2]-z*r[3]:.4f}, {r[2]+z*r[3]:.4f})")

diff = young[2] - old[2]
se_diff = math.sqrt(young[3] ** 2 + old[3] ** 2)
p_add = 2 * (1 - NormalDist().cdf(abs(diff / se_diff)))
print(f"additive interaction={diff:.4f} SE={se_diff:.4f} "
      f"CI=({diff-z*se_diff:.4f}, {diff+z*se_diff:.4f}) p={p_add:.4f}")

rr_y, rr_o = young[0] / young[1], old[0] / old[1]
ratio = rr_y / rr_o
se_log = math.sqrt(1/20 - 1/200 + 1/40 - 1/200 + 1/45 - 1/300 + 1/45 - 1/300)
p_mul = 2 * (1 - NormalDist().cdf(abs(math.log(ratio) / se_log)))
print(f"multiplicative: RR_y={rr_y:.4f} RR_o={rr_o:.4f} ratio={ratio:.4f} "
      f"CI=({math.exp(math.log(ratio)-z*se_log):.4f}, {math.exp(math.log(ratio)+z*se_log):.4f}) "
      f"p={p_mul:.4f}")

# Holm adjustment over 10 tests
pvals = [p_add] + [0.60] * 9
order = sorted(range(len(pvals)), key=lambda i: pvals[i])
adjusted = [0.0] * len(pvals)
running = 0.0
for k, i in enumerate(order):
    val = min(1.0, (len(pvals) - k) * pvals[i])
    running = max(running, val)
    adjusted[i] = running
print(f"Holm adjusted p for age interaction={adjusted[0]:.4f}")
print(f"nominal p={p_add:.4f}; significant at 0.05 before adjustment: {p_add < 0.05}")
```

**Expected output:**

```text
under 65: p1=0.1000 p0=0.200 RD=-0.1000 CI=(-0.1693, -0.0307)
65 and older: p1=0.1500 p0=0.150 RD=0.0000 CI=(-0.0571, 0.0571)
overall: p1=0.1300 p0=0.170 RD=-0.0400 CI=(-0.0842, 0.0042)
additive interaction=-0.1000 SE=0.0458 CI=(-0.1898, -0.0102) p=0.0291
multiplicative: RR_y=0.5000 RR_o=1.0000 ratio=0.5000 CI=(0.2667, 0.9373) p=0.0306
Holm adjusted p for age interaction=0.2910
nominal p=0.0291; significant at 0.05 before adjustment: True
```

**What the output shows:**

- The overall risk difference is -4.0 points with an interval that includes zero. The younger subgroup is -10.0 points with an interval that excludes zero; the older subgroup is 0.0 points.
- The additive and multiplicative interactions are both -0.50 in ratio terms and -10 points in absolute terms, with p-values of 0.029 and 0.031. The two scales agree here, which is a check but not a guarantee.
- The Holm-adjusted p-value is 0.291. The nominal significance does not survive the multiplicity correction.
- The script assumes the other nine p-values are 0.60. In a real analysis they would be the observed values, and the adjusted p-value would depend on them.

> **Note:** The additive interaction standard error assumes independence between the subgroup estimates. In a randomized trial the subgroups are independent samples, so the assumption holds. In a matched or weighted sample, the estimates may be correlated, and a bootstrap or a model-based interaction variance is preferable.

The output does not show:

- Whether the subgroup was pre-specified.
- Whether the subgroup is measured reliably.
- Whether the finding replicates.
- Whether the interaction is confounded in an observational design.
- Whether the absolute effect is clinically important.
- Whether the result is sensitive to the cutpoint.
- Whether the model is correctly specified.
- Whether other subgroups show a similar pattern.

Those questions are answered by the design, the credibility assessment, and the replication, not by the script.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| ICH E9 (1998) | Binding through regional adoption | Pre-specified subgroup analyses; caution in interpretation | The number of subgroups |
| ICH E9(R1) (2019) | Binding through regional adoption | The estimand and the population for a subgroup claim | The interaction method |
| FDA Multiple Endpoints guidance (October 2022) | Final guidance | Multiplicity across endpoints and subgroups | The credibility criteria |
| CONSORT 2010 item 12b | Journal editorial policy | Methods for subgroup and adjusted analyses | The analysis |
| CONSORT 2010 item 18 | Journal editorial policy | Results of subgroup analyses, pre-specified versus exploratory | The interpretation |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) | Journal editorial policy | Reporting of subgroup analyses in observational studies | The method |
| Sun et al. credibility criteria (2014) | Methodological reference | A structured judgment of a subgroup claim | A regulatory standard |

### 8.2 What CONSORT requires

CONSORT item 12b asks for the methods for additional analyses, including subgroup analyses and adjusted analyses. Item 18 asks for the results of any other analyses, distinguishing pre-specified from exploratory. Item 20 asks for the limitations, including multiplicity.

The requirement is modest and frequently unmet. A paper that reports a forest plot without saying which subgroups were pre-specified has not satisfied item 18. A paper that reports a subgroup p-value without the interaction test has not satisfied the spirit of item 12b.

### 8.3 What ICH E9 requires

ICH E9 states that subgroup analyses should be pre-specified, that the number of subgroups should be limited, and that the results should be interpreted with caution. It distinguishes confirmatory subgroup analyses, which can support a regulatory claim, from exploratory analyses, which generate hypotheses.

The guideline does not prescribe a multiplicity procedure or a credibility score. It requires the sponsor to state the subgroups, the analysis, and the interpretation in advance.

### 8.4 What the FDA multiple-endpoints guidance adds

The October 2022 guidance addresses multiplicity across endpoints, including co-primary endpoints, composite endpoints, and multi-component endpoints. It provides a framework for controlling the family-wise error rate with Bonferroni, Holm, Hochberg, fixed-sequence, gatekeeping, and graphical procedures.

A subgroup claim is not an endpoint, but the same logic applies: if several subgroups are tested, the family-wise error rate must be controlled, and the procedure must be pre-specified.

### 8.5 What a protocol should say about subgroups

1. The subgroup variables, limited to a small number.
2. The rationale and the expected direction.
3. The scale (additive, multiplicative, or both).
4. The interaction test and the model.
5. The multiplicity procedure and the family.
6. The credibility criteria that will be applied.
7. The replication plan, if the claim is important.
8. The reporting format, including the forest plot.

A protocol that lists subgroups without items 3 to 7 has not specified the analysis. It has listed variables.

### 8.6 What a reviewer will ask

| Question | Strong answer | Weak answer |
|---|---|---|
| How many subgroups? | Three, pre-specified | "We explored several" |
| Was the interaction tested? | Estimate, interval, p-value | "It was significant in the subgroup" |
| Was multiplicity controlled? | Holm across the pre-specified family | "We used 0.05" |
| Is it replicated? | An independent cohort | "It is biologically plausible" |
| What is the absolute effect? | RD with interval | "The HR was 0.65" |

The questions are predictable because the failure modes are common. A team that prepares the first column can defend the analysis.

### 8.7 Documentation for a subgroup claim

- The protocol section that pre-specified the subgroup.
- The interaction model and the scale.
- The subgroup estimates and intervals.
- The events and sample size per subgroup.
- The multiplicity procedure and the adjusted p-value.
- The credibility assessment.
- The replication plan or result.

A claim without the first item is exploratory. A claim without the third is unquantified. A claim without the fifth is unadjusted.

### 8.8 What the standards do not cover

- They do not specify how many subgroups are acceptable.
- They do not require a credibility assessment.
- They do not define a pre-specified family for subgroup tests.
- They do not resolve the scale question.
- They do not validate a subgroup claim; replication does.

## 9. Industry practice and case evidence

### 9.1 Documented case: the ISIS-2 zodiac sign

The ISIS-2 trial of aspirin and streptokinase in myocardial infarction is famous for a subgroup analysis by zodiac sign. The analysis found that aspirin appeared to be harmful in one sign and beneficial in another, with a nominally significant interaction. The finding was a demonstration that enough subgroup tests will produce a striking pattern by chance.

The example is used in Sun et al. (2014) to illustrate the chance criterion. It is not a criticism of the trial, which was a landmark; it is a demonstration of why a subgroup claim needs pre-specification and replication.

**Inference:** the exact p-values and effect sizes are not reproduced here because the point is the pattern, not the numbers. The example is documented in the methodological literature.

### 9.2 Documented case: hormone replacement therapy subgroups

The Women's Health Initiative trial reported subgroup analyses by age, years since menopause, and other characteristics. Some subgroups showed different relative effects, and the differences were widely discussed. The overall trial result, an increased risk of coronary heart disease in the estrogen-plus-progestin arm, remained the primary basis for the guideline change.

The case illustrates the tension: a subgroup pattern can be clinically plausible and still not override the overall result, especially when the subgroup analyses are exploratory and the number of tests is large.

### 9.3 Documented practice: subgroup analyses in submissions

Pharmaceutical sponsors pre-specify a small number of subgroups in the statistical analysis plan, often age, sex, region, and a disease-severity variable. The forest plot is a standard submission figure. Confirmatory subgroup claims are rare, because the evidence standard is high: pre-specification, a significant interaction after multiplicity control, a clinically important effect, and a plausible mechanism or independent support.

**Inference:** the number and choice of subgroups vary by therapeutic area and are not standardized. The requirement for pre-specification is documented in ICH E9 and CONSORT.

### 9.4 Practice pattern: the forest plot without the interaction test

A common industry pattern is a forest plot with eight subgroup rows, each with a point estimate and a confidence interval, and no interaction p-value. The reader is invited to compare the intervals and draw a conclusion. That is the significance fallacy in graphical form.

The remedy is to add the interaction estimate, its interval, and the adjusted p-value to the plot. The plot then shows both the descriptive estimates and the test of whether they differ.

### 9.5 Documented evidence: credibility of subgroup claims

Sun and colleagues, in a systematic review published in the BMJ in 2012, examined 464 claims of subgroup effects in randomized trials. Only about 15% of the strong claims met more than five of the ten credibility criteria they applied. The finding is evidence that many subgroup claims in the literature are weaker than their presentation suggests.

### 9.6 Documented fact versus inference

| Claim | Status | Basis |
|---|---|---|
| Many subgroup claims are weak | Documented | Sun et al. 2012 systematic review |
| Enough subgroup tests produce false positives | Documented | The zodiac-sign example; probability arithmetic |
| Interaction and within-subgroup tests answer different questions | Documented | Statistical theory; CONSORT E&E |
| A significant within-subgroup effect proves modification | Not supported | It ignores the difference between groups |
| A forest plot is sufficient reporting | Not supported | It omits the interaction and multiplicity |
| A post hoc subgroup can support a label | Not supported | Regulatory standards require pre-specification |

The first three rows are the evidence and theory. The last three are the practices the evidence does not support.

### 9.7 The cost of a false subgroup claim

A false subgroup claim can:

- Narrow a label and deny effective treatment to patients outside the subgroup.
- Expand a label and expose patients in a subgroup where the treatment is ineffective or harmful.
- Trigger a confirmatory trial that fails, delaying access for everyone.
- Create a coverage exception that increases cost without improving outcomes.
- Damage the credibility of the next subgroup analysis from the same team.

**Inference:** the frequency of these outcomes is not captured in a single statistic. The regulatory emphasis on pre-specification and the credibility literature exist because the outcomes were common enough to require a standard.

### 9.8 What mature teams do differently

1. They pre-specify a small number of subgroups with a direction.
2. They report the interaction test, not only the within-subgroup p-values.
3. They report both additive and multiplicative scales when they disagree.
4. They adjust for multiplicity across the pre-specified family.
5. They apply the credibility criteria explicitly.
6. They plan a replication before acting on a subgroup.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Within-subgroup p-value as interaction | "Significant in women, not in men" | Two p-values are compared | False modification claim | Test the interaction |
| Multiplicity ignored | One of many subgroups is significant | Family-wise error inflates | False positive | Pre-specify; Holm |
| Post hoc subgroup presented as pre-specified | The protocol is not cited | Selection on the result | Overstated credibility | Report pre-specification status |
| Data-driven cutpoint | The threshold is not clinical | The cutpoint maximizes the contrast | Inflated Type I error | Pre-specify or continuous |
| Scale switch | Additive significant, multiplicative not, or reverse | The scale is chosen after the result | Misleading conclusion | Report both |
| Sparse subgroup interpreted | Wide interval called "no effect" | Low power | Missed effect | Report the interval; no claim |
| Qualitative interaction claimed post hoc | Opposite signs in one small subgroup | Chance | Harmful decision | Replicate |
| Ecological fallacy | Group average applied to individuals | Aggregate exposure | Wrong individual decision | Individual-level analysis |
| Post-treatment subgroup | The subgroup is a mediator | Conditioning on treatment effects | Biased | Use baseline variables |
| No replication | A single study | Chance or context | Unstable claim | Independent cohort |
| Subgroup not adjusted in observational data | The subgroup is a confounder | The interaction is confounded | Biased | Adjust the interaction |
| Forest plot without interaction | Rows of intervals, no test | Visual comparison | Significance fallacy | Add the interaction |
| Too many subgroups | Dozens of tests | Multiplicity | False positive | Limit and pre-specify |
| Subgroup definition unstable | The variable is measured inconsistently | Misclassification | Unreplicable | Validate the definition |
| Interaction confounded in observational data | The subgroup affects treatment | The interaction is not unconfounded | Biased modification | Adjust; use a DAG |
| Multiple scales reported selectively | One scale is shown | The scale is chosen after the result | Misleading | Report both |
| Subgroup claim without absolute effect | Only the ratio is reported | The baseline risk is hidden | Wrong decision | Report the RD |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The subgroup is measured at baseline | Pre-treatment status is required | A post-treatment variable | Timing review |
| The subgroup is correctly measured | Classification error biases the interaction | Misclassification | Validation sample |
| The interaction model is correctly specified | The test depends on the model | Nonlinearity or omitted interaction | Alternative models |
| The subgroups are independent | The variance of the difference assumes independence | Matched or clustered data | Bootstrap or model-based variance |
| Multiplicity is controlled | The family-wise error is preserved | Many tests, no adjustment | Pre-specified procedure |
| The scale is pre-specified | The conclusion depends on the scale | Scale chosen post hoc | Report both |
| The subgroup has adequate power | The estimate is interpretable | Sparse events | Report the interval |
| The subgroup is not a confounder in observational data | The interaction is unconfounded | The subgroup affects treatment | Adjust |
| Replication is independent | The claim is confirmed | The same data | Pre-specified replication |
| The credibility criteria are applied | The judgment is structured | Significance alone | Criterion-by-criterion review |
| The subgroup is stable and measurable | The claim can be applied | The variable is unreliable | Validation study |
| The absolute effect is reported | The decision uses the right scale | Only the ratio is given | Report RD and RR |

**Limits of this topic.** A subgroup analysis can describe variation but cannot establish its mechanism. The interaction test is scale-dependent, and the choice of scale is a judgment. The credibility criteria are a framework, not a score, and reasonable analysts can weight them differently. The worked example is a small teaching dataset with wide intervals; the estimates are illustrative. The multiplicity procedure assumes a pre-specified family; a post hoc family is not a valid basis for adjustment.

## 12. Exercises

### Exercise 1. Interaction or within-subgroup? (recall, 30 minutes)

For each statement, say whether it is a within-subgroup claim, an interaction claim, or neither:

1. "The drug reduced stroke in patients with diabetes (p = 0.02)."
2. "The effect was larger in patients with diabetes than in those without (p = 0.04)."
3. "The drug was effective overall."
4. "The effect was significant in women but not in men."
5. "The interaction between treatment and age was not significant."

**Deliverable.** A five-row table with the claim type and the correct test.

**Acceptance criteria.**

- The within-subgroup claims are identified as descriptive.
- The interaction claims are identified as tests of difference.
- Statement 4 is identified as a comparison of p-values, not an interaction.
- The correct test is named for each.

### Exercise 2. Compute an interaction (application, 50 minutes)

**Data.** Subgroup A: 30 events / 150 treated, 50 / 150 control. Subgroup B: 40 / 250 treated, 45 / 250 control.

**Deliverable.** The subgroup risk differences, the additive interaction, its standard error, the 95% interval, and the p-value. Then compute the multiplicative interaction.

**Acceptance criteria.**

- Each subgroup RD and SE is shown.
- The interaction is the difference of the RDs.
- The SE uses the sum of the variances.
- The multiplicative interaction is computed on the log scale.
- The two scales are compared.

### Exercise 3. Multiplicity (application, 35 minutes)

**Data.** Six subgroup tests with p-values 0.01, 0.03, 0.04, 0.20, 0.40, and 0.80.

**Deliverable.** The Holm-adjusted p-values and the number of tests significant at 0.05 before and after adjustment.

**Acceptance criteria.**

- The p-values are ordered.
- The step-down multipliers are applied correctly.
- The monotonicity correction is applied.
- The conclusion is stated.

### Exercise 4. Credibility assessment (critique, 45 minutes)

**Claim.** "In a post hoc analysis of one of 12 subgroups, the treatment reduced mortality in patients under 60 (p = 0.03). The mechanism is plausible because younger patients have more reserve."

**Deliverable.** A five-criterion assessment with a verdict.

**Acceptance criteria.**

- Each criterion is assessed with a reason.
- Chance is assessed with the multiplicity adjustment.
- Consistency is assessed as unknown if no replication exists.
- The verdict distinguishes hypothesis-generating from confirmatory.
- The answer states what evidence would raise the credibility.

### Exercise 5. Critique a forest plot (critique, 50 minutes)

**Scenario.** A forest plot has eight subgroup rows, each with a point estimate and a 95% interval. There is no overall estimate, no interaction p-value, and no events column.

**Deliverable.** A list of at least seven deficiencies and a corrected figure specification.

**Acceptance criteria.**

- The missing overall estimate and interaction are identified.
- The missing events column is identified.
- The scale is identified as absent.
- The corrected specification includes the interaction and the multiplicity.
- The critique distinguishes presentation from analysis.

### Exercise 6. Choose the scale (application, 40 minutes)

**Scenario.** A treatment halves the risk in a low-risk group (2% to 1%) and in a high-risk group (20% to 10%).

**Deliverable.** The additive and multiplicative effects in each group, the interaction on each scale, and a recommendation for a payer.

**Acceptance criteria.**

- The risk differences are computed (1 and 10 points).
- The risk ratios are both 0.5.
- The additive interaction is present; the multiplicative is absent.
- The payer recommendation uses the additive scale.
- The answer explains why the scales disagree.

### Exercise 7. Write a subgroup analysis plan (design, 90 minutes)

**Scenario.** A trial of a heart-failure drug with 4,000 patients and a primary composite endpoint.

**Deliverable.** A one-page plan with the subgroups, the rationale, the scale, the interaction model, the multiplicity procedure, the credibility criteria, and the reporting format.

**Acceptance criteria.**

- The number of subgroups is small and justified.
- The direction is stated for each.
- The scale is specified.
- The multiplicity family is defined.
- The forest plot template is included.

### Exercise 8. Audit a label claim (critique, 60 minutes)

**Claim.** "The drug is indicated for patients with elevated biomarkers, based on a subgroup analysis of 220 patients (HR 0.65, p = 0.03)."

**Deliverable.** An audit with the interaction test, the pre-specification status, the multiplicity, the replication, and the regulatory standard.

**Acceptance criteria.**

- The audit asks for the interaction, not the within-subgroup p-value.
- It asks whether the subgroup was pre-specified.
- It asks how many subgroups were tested.
- It asks for replication.
- The verdict is conditional on the answers.
- The audit reports the interaction estimate and interval, not only the p-value.
- The audit states the number of subgroups tested.
- The audit distinguishes a label claim from an exploratory finding.
- The audit recommends a pre-specified replication if the claim is important.
- The audit names the regulatory standard.
- The audit states what evidence would change the verdict.
- The audit is written for a decision-maker, not only for a statistician.
- The audit separates a weak claim from a false one.
- The audit ends with a recommendation.

## 13. Self-check answers

**Answer 1 (Exercise 2 interaction).**

```text
Subgroup A:
p1 = 30/150 = 0.2000; p0 = 50/150 = 0.3333
RD = -0.1333; SE = sqrt(0.2×0.8/150 + 0.3333×0.6667/150)
   = sqrt(0.001067 + 0.001481) = 0.05048

Subgroup B:
p1 = 40/250 = 0.1600; p0 = 45/250 = 0.1800
RD = -0.0200; SE = sqrt(0.16×0.84/250 + 0.18×0.82/250)
   = sqrt(0.000538 + 0.000590) = 0.03359

Interaction = -0.1333 - (-0.0200) = -0.1133
SE = sqrt(0.05048^2 + 0.03359^2) = sqrt(0.002548 + 0.001128) = 0.06063
95% CI = -0.1133 +/- 1.96 × 0.06063 = (-0.2321, +0.0055)
z = -1.869; p = 0.0616
```

The additive interaction is not significant at 0.05. The multiplicative interaction:

```text
RR_A = 0.2000/0.3333 = 0.6000
RR_B = 0.1600/0.1800 = 0.8889
Ratio = 0.6750
```

The ratio of risk ratios is 0.675, and its interval includes 1. Neither scale shows a significant interaction, so the subgroup difference is compatible with chance.

**Answer 2 (Exercise 3 Holm).**

```text
Ordered p: 0.01, 0.03, 0.04, 0.20, 0.40, 0.80
Step 1: 6 × 0.01 = 0.06
Step 2: 5 × 0.03 = 0.15
Step 3: 4 × 0.04 = 0.16
Step 4: 3 × 0.20 = 0.60
Step 5: 2 × 0.40 = 0.80
Step 6: 1 × 0.80 = 0.80
Monotonicity: 0.06, 0.15, 0.16, 0.60, 0.80, 0.80
```

No test is significant at 0.05 after adjustment; the smallest adjusted p-value is 0.06. Before adjustment, three tests were below 0.05. The example shows how quickly the family-wise error correction removes nominal significance.

**Answer 3 (Exercise 6 scales).**

```text
Low risk:  2% -> 1%; RD = -1 point; RR = 0.50
High risk: 20% -> 10%; RD = -10 points; RR = 0.50
Additive interaction = -1 - (-10) = +9 points (the benefit is larger in high risk)
Multiplicative interaction = 0.50/0.50 = 1.00 (none)
```

The treatment halves the risk everywhere, so there is no multiplicative interaction. The absolute benefit is 10 times larger in the high-risk group, so there is a strong additive interaction. A payer deciding where to target the treatment should use the additive scale, because the budget impact is proportional to the absolute benefit.

**Answer 4 (Exercise 4 credibility).**

| Criterion | Assessment | Verdict |
|---|---|---|
| Chance | Nominal p = 0.03, one of 12 tests; Bonferroni-adjusted p = 0.36 | Fails |
| Consistency | No replication | Unknown |
| A priori | Post hoc | Fails |
| Biology | Plausible reserve mechanism | Partial |
| Within-study | Yes | Passes |

Two of five. The claim is hypothesis-generating. The evidence that would raise it: a pre-specified analysis in an independent trial with a significant interaction after multiplicity control.

**Answer 5 (Exercise 5 deficiencies).**

- No overall estimate.
- No interaction p-value.
- No events or sample size per subgroup.
- No statement of the scale.
- No pre-specification status.
- No multiplicity adjustment.
- No confidence interval for the interaction.
- No caveat about the exploratory nature.

The corrected figure adds the overall row, the interaction row, the events column, the scale note, and the pre-specification note.

**Answer 6 (Exercise 1 classification).**

| Statement | Type | Test |
|---|---|---|
| 1 | Within-subgroup effect | Subgroup estimate with interval |
| 2 | Interaction | Treatment-by-subgroup interaction |
| 3 | Overall effect | Primary analysis |
| 4 | Comparison of p-values | Not a valid test; use interaction |
| 5 | Interaction (null) | Interaction test |

**Answer 7 (Exercise 7 rubric).**

A strong plan:

- Specifies three subgroups: age, ejection fraction, and baseline NYHA class.
- States the direction and rationale for each.
- Uses the additive scale for the absolute benefit and the multiplicative scale for the relative effect.
- Pre-specifies a Holm procedure across the three tests.
- Applies the credibility criteria.
- Includes a forest plot template with the overall estimate and the interaction.
- States that a subgroup claim will be exploratory unless replicated.

**Answer 8 (Exercise 8 audit).**

The audit asks:

1. Was the biomarker subgroup pre-specified?
2. What is the interaction estimate and p-value?
3. How many subgroups were tested, and what is the adjusted p-value?
4. Is there an independent replication?
5. Is the effect clinically important in absolute terms?
6. Does the label claim meet the regulatory standard for a confirmatory subgroup?

A claim based on a within-subgroup p-value of 0.03 in 220 patients, with no interaction and no replication, does not meet the standard. The audit should say so conditionally: if the interaction is significant after adjustment and replicated, the claim is stronger; if not, it is exploratory.

## 14. Interview questions

**1. What is effect modification?**

A strong answer defines it as a change in the effect across a characteristic, distinguishes it from confounding, and notes that it is scale-dependent. A weak answer says it is "when the treatment works differently."

**2. Why is a within-subgroup p-value not an interaction test?**

A strong answer explains that it tests the effect within one group, not the difference between groups, and that it depends on the subgroup's sample size. A weak answer says they are "basically the same."

**3. What is the difference between additive and multiplicative interaction?**

A strong answer gives an example where the risk ratio is constant but the risk difference varies, and ties the choice to the decision. A weak answer says one is "more statistical."

**4. How do you control multiplicity across subgroups?**

A strong answer pre-specifies the family and uses Holm, Bonferroni, or gatekeeping. A weak answer says "use a stricter alpha" without a procedure.

**5. What are the credibility criteria for a subgroup claim?**

A strong answer lists chance, consistency, a priori hypothesis, biology, and within-study comparison. A weak answer says "pre-specification."

**6. Why are subgroup analyses usually underpowered?**

A strong answer explains that the subgroup is smaller and the variance larger, so a non-significant result is expected. A weak answer says "small samples."

**7. What is a qualitative interaction, and why is it important?**

A strong answer defines opposite signs and says it changes the decision for the subgroup, and that it requires stronger evidence. A weak answer confuses it with a quantitative difference.

**8. Why are data-driven cutpoints a problem?**

A strong answer explains that choosing the cutpoint that maximizes the contrast inflates the Type I error and produces unreplicable findings. A weak answer says "they are arbitrary."

**9. How would you report a subgroup analysis?**

A strong answer describes a forest plot with the overall estimate, subgroup estimates, events, interaction, multiplicity, and pre-specification. A weak answer says "a table of subgroup p-values."

A follow-up is "where does the interaction p-value go?" The strong answer puts it in the figure and in the text, with the scale and the adjustment stated. The subgroup rows alone are descriptive; the interaction row is the inferential statement.

**10. Can a subgroup analysis change a label?**

A strong answer says only with pre-specification, a significant adjusted interaction, a clinically important effect, and replication or strong mechanistic support. A weak answer says "if it is significant."

**11. What is the ecological fallacy in subgroup analysis?**

A strong answer explains that a group-level association may not hold at the individual level, and gives an example. A weak answer has not heard of it.

**12. What would make you distrust a subgroup claim?**

A strong answer lists post hoc definition, one of many tests, no interaction, no replication, a data-driven cutpoint, and a scale chosen after the result. A weak answer says "a small subgroup."

## 15. Key terms

- **Additive interaction:** a difference in risk differences across subgroups.
- **Baseline characteristic:** a variable measured before treatment.
- **Confirmatory subgroup:** a pre-specified subgroup for a regulatory claim.
- **Credibility criteria:** five criteria for judging a subgroup claim.
- **Cutpoint:** a threshold used to dichotomize a continuous variable.
- **Data-driven subgroup:** a subgroup defined by the data.
- **Effect modification:** a change in the effect across a characteristic.
- **Ecological fallacy:** inferring individual effects from group averages.
- **Exploratory subgroup:** a hypothesis-generating analysis.
- **Family-wise error rate:** the probability of at least one false positive in a family.
- **Forest plot:** a plot of subgroup estimates with intervals.
- **Gatekeeping:** a hierarchical testing procedure.
- **Heterogeneity:** variation in effect across studies or subgroups.
- **Holm procedure:** a step-down multiplicity adjustment.
- **Interaction:** a departure from additivity or multiplicativity.
- **Interaction p-value:** a test of the difference between subgroup effects.
- **Multiplicative interaction:** a difference in ratios across subgroups.
- **Multiplicity:** the inflation of error when many tests are run.
- **Post hoc:** defined after seeing the data.
- **Power:** the probability of detecting an effect.
- **Pre-specification:** defining the hypothesis before seeing the data.
- **Qualitative interaction:** opposite signs across subgroups.
- **Quantitative interaction:** a difference in magnitude without a sign change.
- **Sparse subgroup:** a subgroup with few events.
- **Subgroup:** a subset defined by a baseline characteristic.
- **Treatment-by-subgroup interaction:** the product term in the model.
- **Within-subgroup p-value:** a test of the effect within one subgroup.
- **Relative effect:** a ratio measure.
- **Absolute effect:** a difference measure.
- **Credible subgroup:** a subgroup claim that meets the criteria.
- **Replication:** confirmation in an independent dataset.
- **Scale dependence:** the property that interaction depends on the measure.
- **Subgroup estimate:** the effect within one level of a characteristic.
- **Interaction estimate:** the difference between subgroup effects.
- **Hypothesis-generating:** an analysis that suggests a question for further study.
- **Confirmatory analysis:** an analysis designed to test a pre-specified hypothesis.
- **Subgroup variable:** the characteristic that defines the subsets.
- **Direction of effect:** whether the treatment is beneficial or harmful.
- **Baseline risk:** the outcome risk without treatment in a subgroup.
- **Absolute benefit:** the risk difference in a subgroup.
- **Relative benefit:** the risk ratio in a subgroup.
- **Multiple testing:** the conduct of several hypothesis tests in one analysis.
- **Interaction variance:** the variance of the difference between subgroup effects.
- **Power for interaction:** the probability of detecting a true difference between subgroups.
- **Subgroup-specific estimate:** the effect within a single subgroup.
- **Pooled estimate:** the overall effect across subgroups.
- **Consistency of effect:** whether the direction is the same across subgroups.
- **Pre-specified direction:** the expected sign of the subgroup effect.
- **Subgroup credibility:** the judgment that a subgroup claim is trustworthy.
- **Interaction scale:** the measure on which the interaction is tested.
- **Effect measure modification:** another term for effect modification.
- **Heterogeneity test:** a test of variation across studies or subgroups.
- **Subgroup analysis plan:** the pre-specified document defining the subgroups and the tests.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. Sun, X., Ioannidis, J. P. A., Agoritsas, T., Alba, A. C., Guyatt, G. "How to Use a Subgroup Analysis: Users' Guide to the Medical Literature." *JAMA* 2014;311(4):405-411. https://doi.org/10.1001/jama.2013.285063
2. Sun, X., Briel, M., Busse, J. W., et al. "Credibility of claims of subgroup effects in randomised controlled trials: systematic review." *BMJ* 2012;344:e1553. https://doi.org/10.1136/bmj.e1553
3. Rothwell, P. M. "Subgroup analysis in randomised controlled trials: importance, indications, and interpretation." *Lancet* 2005;365(9454):176-186. https://doi.org/10.1016/S0140-6736(05)17709-5
4. Schulz, K. F., Altman, D. G., Moher, D., for the CONSORT Group. "CONSORT 2010 Statement." *PLoS Medicine* 2010;7(3):e1000251. https://doi.org/10.1371/journal.pmed.1000251
5. Moher, D., Hopewell, S., Schulz, K. F., et al. "CONSORT 2010 Explanation and Elaboration." *BMJ* 2010;340:c869. https://doi.org/10.1136/bmj.c869
6. International Council for Harmonisation. *ICH E9: Statistical Principles for Clinical Trials*. Step 4, 5 February 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
7. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
8. United States Food and Drug Administration. *Multiple Endpoints in Clinical Trials: Guidance for Industry*. October 2022. https://www.fda.gov/media/162416/download
9. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
10. Rothman, K. J., Greenland, S., Lash, T. L. *Modern Epidemiology*. 3rd edition. Lippincott Williams and Wilkins, 2008.
11. Wang, R., Lagakos, S. W., Ware, J. H., Hunter, D. J., Drazen, J. M. "Statistics in medicine — reporting of subgroup analyses in clinical trials." *New England Journal of Medicine* 2007;357(21):2189-2194. https://doi.org/10.1056/NEJMsr077003
12. Assmann, S. F., Pocock, S. J., Enos, L. E., Kasten, L. E. "Subgroup analysis and other (mis)uses of baseline data in clinical trials." *Lancet* 2000;355(9209):1064-1069. https://doi.org/10.1016/S0140-6736(00)02039-0
13. Yusuf, S., Wittes, J., Probstfield, J., Tyroler, H. A. "Analysis and interpretation of treatment effects in subgroups of patients in randomized clinical trials." *JAMA* 1991;266(1):93-98.
14. Pocock, S. J., Assmann, S. E., Enos, L. E., Kasten, L. E. "Subgroup analysis, covariate adjustment and baseline comparisons in clinical trial reporting: current practice and problems." *Statistics in Medicine* 2002;21(19):2917-2930.
15. Brookes, S. T., Whitely, E., Egger, M., et al. "Subgroup analyses in randomized trials: risks of subgroup-specific analyses; power and sample size for the interaction test." *Journal of Clinical Epidemiology* 2004;57(3):229-236.

> **Note:** The Sun 2014 paper is the source for the five credibility criteria and the subgroup framework. The Sun 2012 BMJ review is the source for the finding that only about 15% of strong subgroup claims met more than five of ten criteria. The zodiac-sign example is discussed in Sun 2014; the exact trial numbers are not reproduced.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) — robustness and heterogeneity.
- Causal graphs: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — effect modification and adjustment.
- Regression: [`02_regression-models.md`](02_regression-models.md) — interaction terms.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — forest plots and effect scales.
- Survey file: [`../../04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.2 compares subgroup estimates.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md) — assumptions ledger.
- Estimation: [`01_estimation-and-inference.md`](01_estimation-and-inference.md) — interaction intervals.
- Target trial emulation: [`07_target-trial-emulation.md`](07_target-trial-emulation.md) — pre-specified subgroup analyses.
- Module 13: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md) — label claims and trials.
