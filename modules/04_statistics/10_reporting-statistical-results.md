# Reporting Statistical Results

> Module 04 — Statistics and Causal Inference
> Topic 10 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 02, 09
> Version 2026.3

---

## 1. Why this topic exists

An analysis that is correct and unreported is worthless. An analysis that is reported in a way that overstates its certainty is worse than worthless, because it transfers a wrong belief to a decision-maker who cannot see the data. Reporting is the last mile of the statistical workflow, and it is where many of the earlier topics fail in practice.

The common failures are not exotic:

- **The p-value replaces the estimate.** A report says "p = 0.03" and does not say how large the effect is. The reader learns that the result is unlikely under the null and nothing about whether it matters.
- **The relative effect replaces the absolute effect.** "A 33% reduction" sounds large. If the baseline risk is 18% and the reduction is to 12%, the absolute reduction is 6 percentage points, and the number needed to treat is 17. The relative number is true and incomplete.
- **The interval is omitted.** A point estimate without an interval hides the precision and invites overconfidence. A confidence interval is not an optional decoration; it is the report of the uncertainty.
- **The harms are missing.** A report that describes the benefit and omits the adverse events is not a benefit-risk assessment. The omission can be unintentional or strategic; the effect on the decision is the same.
- **The analysis population is unclear.** A reader cannot tell whether the estimate is intention-to-treat or per-protocol, or how many patients were excluded. The estimate is not interpretable without the population.
- **The abstract overstates the result.** Boutron and colleagues, in a 2010 study in JAMA, found that among 72 randomized trials with a non-significant primary outcome, 58.3% of abstract conclusions contained spin: reporting that highlighted the treatment as beneficial despite the non-significant primary result. The abstract is what most readers see.
- **The uncertainty is not propagated.** A cost-effectiveness model built on a hazard ratio with a wide interval is reported with a single incremental cost-effectiveness ratio. The interval is lost.

The regulatory and editorial standards address these failures. Consolidated Standards of Reporting Trials (CONSORT) 2010 item 17a requires effect sizes with precision, and item 17b requires both absolute and relative effects for binary outcomes. The Statistical Analyses and Methods in the Published Literature (SAMPL) guidelines specify the minimum statistical reporting for biomedical journals. The International Council for Harmonisation (ICH) E9(R1) addendum requires the estimand to be reported with the estimate. None of these standards is enforced by law, but they are enforced by reviewers, editors, and, increasingly, by payers who read the same papers.

This topic builds the reporting workflow: the estimand, the effect on both scales, the interval, the absolute risk and number needed to treat, the harms, the analysis population, the multiplicity, the subgroups, the sensitivity analyses, and the language that keeps the conclusion within the data.

> **Note:** A results section is not a summary of the analysis. It is a set of quantities that a reader can use to reconstruct the decision. If a reader cannot compute the absolute benefit, the number needed to treat, and the range of plausible effects from the report, the report is incomplete.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a results paragraph that contains the estimand, the effect on the absolute and relative scales, the interval, and the analysis population.
2. **Convert** between risk differences, risk ratios, odds ratios, and numbers needed to treat, with intervals.
3. **Report** the number needed to treat with its horizon and its interval, including the NNTB/NNTH convention when the interval crosses zero.
4. **Report** harms alongside benefits, on the same scale and with the same precision.
5. **State** the analysis population and the handling of missing data in one sentence.
6. **Report** multiplicity and subgroup analyses without overstating the confirmatory content.
7. **Recognize** spin in an abstract and **rewrite** it to match the data.
8. **Design** a results table and a figure that a decision-maker can use.

Outcome 2 maps to Exercise 2 and the worked example. Outcome 7 maps to Exercise 6. Outcome 8 maps to Exercise 8.

## 3. The core mental model

Reporting is a chain from the estimand to the decision. Each link must be present, and each must be on the scale the next link needs.

```text
  ESTIMAND
  population, contrast, outcome, horizon, summary
        |
        v
  EFFECT
  absolute (RD, NNT) + relative (RR, OR, HR)
        |
        v
  UNCERTAINTY
  95% interval, on both scales
        |
        v
  POPULATION AND DATA
  analysis set, missingness, exclusions
        |
        v
  BENEFITS AND HARMS
  same scale, same horizon, same precision
        |
        v
  CONTEXT
  multiplicity, subgroups, sensitivity
        |
        v
  DECISION SENTENCE
  what the data support, and what they do not
```

### 3.1 The absolute scale is the decision scale

A relative effect answers "how much more likely?" An absolute effect answers "how many patients are affected?" The second is what a budget, a clinic schedule, and a patient conversation require.

| Measure | Formula | Example | Use |
|---|---|---|---|
| Risk difference | p1 - p0 | 12% - 18% = -6 pp | Population impact |
| Risk ratio | p1 / p0 | 0.12 / 0.18 = 0.67 | Relative effect |
| Odds ratio | (a/b)/(c/d) | 0.62 | Logistic model output |
| NNT | 1 / absolute RD | 1 / 0.06 = 17 | Communication |
| NNH | 1 / |risk increase| | 1 / 0.01 = 100 | Harms |

The NNT is the reciprocal of the absolute risk reduction. It is not a separate estimate; it is a transformation of the risk difference. Its interval is obtained by taking the reciprocals of the risk difference interval and reversing the order.

### 3.2 The interval is the report of uncertainty

A 95% confidence interval states the range of effects compatible with the data under the model. It is more informative than a p-value because it shows both the direction and the precision. A p-value can be derived from a Wald interval, but the reverse is not true: a p-value does not tell you the width.

The interval should be reported on the scale of the estimate. For a ratio, report the ratio and its interval. For a difference, report the difference and its interval. For an NNT, report the NNT and its interval, with the NNTB/NNTH convention if the interval crosses zero.

### 3.3 The NNT interval is not symmetric

Because the NNT is a reciprocal, its interval is asymmetric and can be disconnected when the risk difference interval includes zero. If the risk difference is -6 pp with a 95% CI of -9.5 to -2.5, the NNT is 17 with a 95% CI of 11 to 40. If the risk difference interval were -2 to +4, the NNT interval would run from NNTB 50 through infinity to NNTH 25. Reporting a single "NNT 17 (95% CI 11 to 40)" when the interval crosses zero is wrong.

### 3.4 The harms belong in the same table

A benefit-risk report puts the benefit and the harm on the same scale, with the same horizon and the same precision. A table that reports the primary benefit with an interval and the harms as counts without denominators is not a benefit-risk assessment.

| Outcome | Treatment | Control | RD (95% CI) | NNT/NNH |
|---|---:|---:|---|---|
| Primary benefit | 12.0% | 18.0% | -6.0 pp (-9.5, -2.5) | NNT 17 (11, 40) |
| Serious adverse event | 5.0% | 4.0% | +1.0 pp (-1.0, +3.0) | NNTH 100 (NNTB 100 to NNTH 33) |

The second row is what makes the first row interpretable. A 6-point benefit with a 1-point harm may be worth it; a 6-point benefit with a 6-point harm may not.

### 3.5 The analysis population is part of the estimate

An effect estimate is conditional on the population that produced it. A report that says "the treatment reduced the outcome" without saying whether the analysis was intention-to-treat, per-protocol, or complete-case has omitted a condition that can change the estimate.

The minimum statement is one sentence: "The primary analysis was intention-to-treat, including all 1,600 randomized patients; 4.2% had missing primary outcomes, which were imputed under a missing-at-random assumption with 20 imputations."

### 3.6 Spin is a reporting strategy, not a statistical error

Spin is the use of language or selective reporting to make a result appear more favorable than the data support. Boutron and colleagues defined it as specific reporting strategies that highlight the treatment as beneficial despite a non-significant primary outcome, or that distract the reader from the non-significant result.

Common forms:

- Reporting a secondary outcome as if it were primary.
- Reporting a within-subgroup effect without the interaction.
- Reporting the relative effect and omitting the absolute effect.
- Describing a non-significant result as "a trend toward benefit."
- Focusing the conclusion on a favorable secondary outcome.
- Omitting the harms.

The remedy is not a different adjective. It is a structured report that includes the primary outcome, the absolute and relative effects, the interval, the harms, and the population, in that order.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Effect size | The magnitude of the difference between groups | Used as a synonym for p-value | Effect units |
| Absolute effect | A difference in risks or means | Omitted in favor of the relative effect | Percentage points or units |
| Relative effect | A ratio of risks or rates | Reported alone | Unitless ratio |
| Risk difference | The difference in outcome proportions | Confused with the risk ratio | Percentage points |
| Risk ratio | The ratio of outcome proportions | Called "relative risk reduction" when the complement is meant | Unitless |
| Odds ratio | The ratio of odds | Interpreted as a risk ratio | Unitless |
| Hazard ratio | The ratio of hazards | Interpreted as a ratio of cumulative risks | Unitless |
| NNT | The number needed to treat for one additional benefit | Reported without a horizon | Patients |
| NNH | The number needed to harm for one additional harm | Reported without a horizon | Patients |
| NNTB | Number needed to treat for benefit | Confused with NNT when the interval crosses zero | Patients |
| NNTH | Number needed to treat for harm | Omitted when the interval crosses zero | Patients |
| Confidence interval | The range of effects compatible with the data | Replaced by a p-value | Effect units |
| Precision | The width of the interval | Confused with accuracy | Effect units |
| Analysis population | The set of patients included in the analysis | Left implicit | Population |
| Intention-to-treat | Analysis by assigned group | Called "as-treated" | Population |
| Per-protocol | Analysis by treatment received | Called "intention-to-treat" | Population |
| Missingness | The proportion of missing outcome data | Omitted from the results | Percentage |
| Multiplicity | The number of hypotheses tested | Ignored in the report | Count |
| Pre-specified | Defined before the analysis | Claimed for post hoc analyses | Design |
| Exploratory | Hypothesis-generating | Presented as confirmatory | Design |
| Spin | Reporting that overstates the result | Treated as a writing issue | Reporting |
| Statistical significance | A p-value below a threshold | Equated with clinical importance | Decision |
| Clinical importance | An effect large enough to matter | Equated with significance | Judgment |
| Minimal clinically important difference | The smallest important effect | Confused with the observed effect | Effect units |
| Baseline risk | The outcome risk in the control group | Omitted when the relative effect is reported | Probability |
| Horizon | The time over which the effect is measured | Left implicit | Time |
| Benefit-risk | The balance of benefits and harms | Reported as benefit only | Judgment |
| Number needed to treat | The reciprocal of the absolute risk reduction | Reported without an interval | Patients |
| Reporting guideline | A standard for the content of a report | Treated as a quality score | Document |
| CONSORT | The reporting guideline for randomized trials | Applied to observational studies | Checklist |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) | The reporting guideline for observational studies | Applied to trials | Checklist |
| SAMPL | The statistical reporting guideline | Confused with CONSORT | Checklist |
| Effect measure | The scale on which the effect is reported | Chosen after the result | Scale |

Two conventions in this topic:

- A risk difference is reported in percentage points, not percent: "6 percentage points," not "6%."
- An NNT is reported with a horizon and an interval: "NNT 17 (95% CI 11 to 40) over 12 months."

> **Note:** A p-value and a confidence interval are not interchangeable. The interval contains the effect size and the precision; the p-value contains neither. A report that gives only the p-value has removed the information a decision requires and kept the information that is easiest to misread.

## 5. The mechanics, step by step

The procedure has twelve steps. Steps 1 to 6 build the result. Steps 7 to 11 add the context. Step 12 checks the language.

### Step 1. Identify the audience and the decision

**Input:** The decision, the readers, and the format.

**Output:** A statement of what the reader must be able to do after reading the report.

**Why it exists:** A regulatory table, a payer submission, a clinical guideline, and a patient leaflet need different quantities. The report should give each audience the quantity it needs without omitting the others.

| Audience | Needs | Report first |
|---|---|---|
| Clinician | Absolute benefit and harm for a patient like theirs | RD, NNT, harms |
| Payer | Population impact and budget | RD, eligible population, cost |
| Regulator | Estimand, pre-specified analysis, precision | Primary estimate, interval, multiplicity |
| Journal reader | The result and its uncertainty | Estimate, interval, population |
| Patient | The chance of benefit and harm | Natural frequencies |

### Step 2. State the estimand

**Input:** The protocol and the analysis plan.

**Output:** One sentence with the population, contrast, outcome, horizon, and summary.

**Why it exists:** The estimate is meaningless without the quantity it estimates. The estimand is the anchor for every number that follows.

Example: "Among adults discharged after heart failure, the effect of the transition program versus usual care on the 12-month risk of unplanned readmission, summarized as a risk difference."

### Step 3. Report the absolute and relative effects

**Input:** The analysis output.

**Output:** A risk difference and a risk ratio (or mean difference and standardized effect), both with units.

**Why it exists:** The relative effect communicates the strength of the association; the absolute effect communicates the population impact. Neither is sufficient alone.

| Outcome type | Absolute | Relative |
|---|---|---|
| Binary | Risk difference in percentage points | Risk ratio or odds ratio |
| Continuous | Mean difference in units | Standardized mean difference |
| Count | Rate difference per person-time | Rate ratio |
| Time-to-event | Risk difference at a horizon, or RMST difference | Hazard ratio |

### Reporting phrases for common measures

| Measure | Report as | Avoid |
|---|---|---|
| Risk difference | "6.0 percentage points lower (95% CI 2.5 to 9.5)" | "6% lower" |
| Risk ratio | "RR 0.67 (95% CI 0.52 to 0.85)" | "33% reduction" alone |
| Odds ratio | "OR 0.62 (95% CI 0.47 to 0.82)" | "62% as likely" |
| Hazard ratio | "HR 0.75 (95% CI 0.60 to 0.94)" | "25% lower risk" |
| Mean difference | "2.4 points lower (95% CI 1.1 to 3.7)" | "a 2.4-point benefit" |
| NNT | "NNT 17 (95% CI 11 to 40) over 12 months" | "NNT 17" |
| NNH | "NNTH 100 (95% CI NNTH 33 to NNTB 97)" | "NNH 100 (95% CI 33 to 97)" |

The phrase carries the scale, the direction, the interval, and, for the NNT, the horizon. A phrase that omits any of these is incomplete.

### Step 4. Report the interval

**Input:** The standard error or the model output.

**Output:** A 95% confidence interval on the same scale as the estimate.

**Why it exists:** The interval is the report of uncertainty. A point estimate without it invites overconfidence.

- Use the same number of decimal places as the estimate, not more.
- State the confidence level.
- Do not report an interval that is asymmetric on a ratio scale as if it were symmetric.
- Do not report a p-value instead of an interval.

### Interval versus p-value

| Reporting choice | What the reader learns | What is lost |
|---|---|---|
| Estimate + 95% CI | Size, direction, precision | Nothing essential |
| Estimate + p-value | Size, compatibility with null | Precision, the range of plausible effects |
| p-value only | Compatibility with null | Size, direction, precision |
| "Significant" only | A threshold was crossed | Everything quantitative |
| Estimate + CI + p-value | All of the above | Nothing, if the interval is primary |

The p-value can be added after the interval. It cannot substitute for it.

### Step 5. Report the NNT with its horizon and interval

**Input:** The risk difference and its interval.

**Output:** An NNT and a 95% interval, with the horizon.

**Why it exists:** The NNT is the most communicable form of the absolute effect, and the most easily misreported.

```text
NNT = 1 / |RD|
NNT interval = (1 / |upper RD|, 1 / |lower RD|), with the order reversed
```

**When the interval crosses zero**, the NNT interval is disconnected. Use the NNTB/NNTH notation:

```text
"NNTH 33 to infinity to NNTB 97"
```

The phrase means that the data are compatible with a harm requiring 33 patients to harm one, or a benefit requiring 97 patients to help one, or no effect.

### NNT examples and their traps

| Baseline risk | Relative effect | Absolute effect | NNT | Trap |
|---|---:|---:|---:|---|
| 18% | RR 0.67 | -6.0 pp | 17 | None |
| 2% | RR 0.67 | -0.67 pp | 150 | The relative effect looks the same; the NNT is nine times larger |
| 50% | RR 0.67 | -16.5 pp | 6 | A small relative effect is a large absolute effect |
| 18% | RR 0.90 | -1.8 pp | 56 | A "10% reduction" is modest in absolute terms |
| 18% | RR 1.10 | +1.8 pp | NNTH 56 | A harmful relative effect |

The table is the argument for reporting both scales. The same relative effect can correspond to an NNT of 6 or 150, depending on the baseline risk.

### Step 6. Report harms on the same scale

**Input:** The adverse event data.

**Output:** A table with the same denominators, the same horizon, and the same precision as the benefit.

**Why it exists:** A benefit without a harm is not a benefit-risk assessment.

- Use the same analysis population for benefits and harms.
- Report the absolute risk and the interval.
- Do not report a harm as a count without a denominator.
- Report serious and common adverse events separately.
- State the ascertainment method.

### Step 7. State the analysis population and missingness

**Input:** The cohort flow and the missing-data analysis.

**Output:** One or two sentences that a reader can use to interpret the estimate.

**Why it exists:** The estimate is conditional on the population. A reader who does not know the population cannot apply the estimate.

The minimum statement:

- The number randomized or eligible.
- The number analyzed.
- The analysis set (intention-to-treat, per-protocol, complete-case).
- The proportion missing and the method.
- The number excluded and the reasons.

### Population and missingness checklist

- Number screened, eligible, randomized, and analyzed.
- The analysis set, named explicitly.
- The number and reasons for exclusions after randomization.
- The proportion missing for the primary outcome, by arm.
- The missing-data method and the number of imputations.
- The sensitivity analysis for missingness.
- The follow-up time, median and range.

A results section that omits the third and fourth items cannot be assessed for selection or attrition bias.

### Step 8. Report multiplicity

**Input:** The family of hypotheses and the adjustment.

**Output:** A statement of the family, the procedure, and the adjusted result.

**Why it exists:** A significant result in a family of many tests is not a significant result at the nominal level.

- State the primary endpoint and the family.
- State the procedure (Holm, gatekeeping, fixed sequence).
- Report the adjusted p-value or the adjusted interval.
- Label exploratory analyses as exploratory.

### Multiplicity reporting table

| Endpoint | Family | Nominal p | Adjusted p | Procedure | Status |
|---|---|---|---:|---|---|
| Primary readmission | Primary | 0.001 | 0.001 | Not adjusted | Confirmatory |
| Cardiovascular death | Secondary | 0.03 | 0.09 | Holm (3 tests) | Not significant |
| Quality of life | Secondary | 0.04 | 0.12 | Holm (3 tests) | Not significant |
| Subgroup under 65 | Exploratory | 0.03 | 0.29 | Holm (10 tests) | Exploratory |

The table makes the multiplicity visible. A report that shows only the nominal p-values invites the reader to treat every 0.04 as a finding.

### Step 9. Report subgroup analyses as exploratory unless pre-specified

**Input:** The subgroup analysis.

**Output:** A forest plot or table with the interaction and the multiplicity.

**Why it exists:** A subgroup claim can change a label or a coverage decision. It should be reported with the evidence that supports or undermines it.

- State which subgroups were pre-specified.
- Report the interaction, not only the within-subgroup p-values.
- Report the adjusted p-value.
- Label the analysis exploratory if it was not pre-specified.

### Step 10. Report the sensitivity analyses

**Input:** The primary and sensitivity results.

**Output:** A short table with the assumption, the method, and the result.

**Why it exists:** The primary result is one estimate under one set of assumptions. The sensitivity analyses show how much the conclusion depends on them.

- Report the sensitivity analyses that were pre-specified.
- Report the ones that changed the conclusion, even if they were not pre-specified.
- Do not report only the sensitivity analyses that agree with the primary.

### Step 11. Design the tables and figures

**Input:** The results.

**Output:** A table and a figure that a reader can use without the text.

**Why it exists:** Readers go to the tables and figures first. A table that requires the text to interpret is a failed table.

Table 1: baseline characteristics by group, with the standardized difference rather than a p-value.
Table 2: primary and secondary outcomes, with absolute and relative effects, intervals, and NNT.
Figure 1: the flow of participants.
Figure 2: the primary result, with an interval and a horizon.

### Tables and figures that work

| Artifact | Must contain | Common failure |
|---|---|---|
| Table 1 | Baseline characteristics by group, standardized differences | p-values instead of balance measures |
| Table 2 | Primary and secondary outcomes, absolute and relative effects, intervals, NNT | Odds ratios only |
| Table 3 | Harms, with denominators and intervals | Counts without denominators |
| Figure 1 | Participant flow | No reasons for exclusion |
| Figure 2 | Primary result with interval and horizon | A bar chart with no interval |
| Figure 3 | Kaplan-Meier with number at risk | No number-at-risk table |
| Appendix | Sensitivity analyses and model specification | Only the analyses that agree |

The table and figure list is also a checklist. A report that lacks Table 2 and Figure 2 has not reported the primary result.

### Branch points at a glance

| If the result is | Then report | Because |
|---|---|---|
| Binary, common outcome | RD and RR, NNT | The OR overstates the ratio |
| Binary, rare outcome | RR and RD, NNT | The OR approximates the RR |
| Time-to-event, proportional | HR, risk at horizon, NNT | The HR is not a risk |
| Time-to-event, non-proportional | RMST or milestone risk | The HR is time-averaged |
| Continuous | Mean difference, MCID comparison | Standardized effects are hard to interpret |
| Non-significant | Estimate and interval, no "trend" | Absence of evidence is not evidence of absence |
| Significant but small | Absolute effect and NNT | Significance is not importance |
| Harmful | NNTH, with the interval | The direction matters |

### Reporting checklist

1. The estimand, in one sentence.
2. The analysis population.
3. The absolute effect with an interval.
4. The relative effect with an interval.
5. The NNT with a horizon and an interval.
6. The harms with denominators and intervals.
7. The missingness and the method.
8. The multiplicity and the subgroup status.
9. The sensitivity analyses.
10. The conclusion, within the data.

A report with all ten is complete. A report with items 3 to 5 is usable. A report with only a p-value is neither.

- Items 1 and 2 belong at the start of the results.
- Items 3 to 6 belong in the primary results paragraph.
- Items 7 to 9 belong in a short methods or sensitivity paragraph.
- Item 10 belongs in the conclusion, and it should be one sentence.
- The checklist can be completed before the analysis, as a report template.
- A completed template is the fastest way to catch a missing absolute effect.

### Step 12. Check the language for spin

**Input:** The draft abstract and conclusions.

**Output:** A revised text that matches the data.

**Why it exists:** Spin is common and it changes decisions. The check is a structured comparison of the claim with the result.

| Spin pattern | Example | Revision |
|---|---|---|
| Secondary as primary | "The drug reduced hospitalizations" when the primary was symptoms | Name the primary outcome |
| Relative only | "A 33% reduction" | Add the absolute reduction and NNT |
| Trend language | "A trend toward benefit (p = 0.08)" | "The result was not statistically significant" |
| Subgroup promotion | "Effective in younger patients" | "The interaction was exploratory" |
| Harm omission | No adverse events reported | Add the harms table |
| Certainty language | "The drug prevents stroke" | "The drug was associated with a lower risk" |

### Edge cases that change the report

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| Interval crosses zero | The NNT interval is disconnected | RD interval includes 0 | Use NNTB/NNTH |
| Common outcome with an OR | The OR overstates the RR | Control risk above 15% | Report the RR and RD |
| Non-proportional hazards | The HR is time-averaged | Schoenfeld test | Report RMST or milestone risk |
| Missing outcome | The estimate depends on the assumption | Missingness above 5% | Report the method and sensitivity |
| Multiplicity | The nominal p-value is not adjusted | Many endpoints | Report the adjusted result |
| Subgroup claim | The interaction is not significant | Within-subgroup p only | Report the interaction |
| Harms rare | The NNH is large and uncertain | Few events | Report the counts and the interval |
| Composite endpoint | The components may move in opposite directions | Composite significant | Report the components |

## 6. Worked example

### 6.1 The study and the estimand

A randomized trial compares a transition program with usual care after heart failure discharge. The primary outcome is unplanned readmission within 12 months. The analysis is intention-to-treat.

The estimand: "Among adults discharged alive after heart failure, the effect of the transition program versus usual care on the 12-month risk of unplanned readmission, summarized as a risk difference and a risk ratio, with the number needed to treat over 12 months."

### 6.2 The data

| Group | Readmitted | Not readmitted | Total | 12-month risk |
|---|---:|---:|---:|---:|
| Program | 96 | 704 | 800 | 12.0% |
| Usual care | 144 | 656 | 800 | 18.0% |
| Total | 240 | 1,360 | 1,600 | 15.0% |

### 6.3 The absolute effect

```text
RD = 0.1200 - 0.1800 = -0.0600 = -6.00 percentage points
SE = sqrt(0.12×0.88/800 + 0.18×0.82/800)
   = sqrt(0.000132 + 0.0001845)
   = sqrt(0.0003165) = 0.017790
95% CI = -0.0600 +/- 1.96 × 0.017790
       = (-0.09487, -0.02513) = (-9.49, -2.51) percentage points
```

The absolute reduction is 6.00 percentage points, with an interval from 2.51 to 9.49 points. The interval excludes zero.

### 6.4 The relative effects

```text
RR = 0.1200 / 0.1800 = 0.6667
SE(log RR) = sqrt(1/96 - 1/800 + 1/144 - 1/800)
           = sqrt(0.010417 - 0.001250 + 0.006944 - 0.001250)
           = sqrt(0.014861) = 0.12191
95% CI = exp(log(0.6667) +/- 1.96 × 0.12191)
       = exp(-0.4055 +/- 0.2389)
       = (0.5250, 0.8466)

OR = (96/704) / (144/656) = 0.13636 / 0.21951 = 0.6212
SE(log OR) = sqrt(1/96 + 1/704 + 1/144 + 1/656)
           = sqrt(0.010417 + 0.001420 + 0.006944 + 0.001524)
           = sqrt(0.020305) = 0.14250
95% CI = exp(-0.4759 +/- 0.2793) = (0.4698, 0.8214)
```

The risk ratio is 0.67, corresponding to a 33.3% relative reduction. The odds ratio is 0.62, which is farther from 1 because the outcome is common (18% in the control group). Reporting the odds ratio as "a 38% reduction" would overstate the relative effect by about 5 percentage points.

### 6.5 The NNT

```text
NNT = 1 / 0.0600 = 16.67, reported as 17
95% CI = (1 / 0.09487, 1 / 0.02513)
       = (10.54, 39.79), reported as (11, 40)
```

The correct statement is "NNT 17 (95% CI 11 to 40) over 12 months." The NNT is 17 patients treated for one additional patient to avoid a readmission within 12 months.

The interval is asymmetric because the NNT is a reciprocal. A symmetric "17 +/- 15" would be wrong.

### 6.6 The harms

Serious adverse events occurred in 40 of 800 program patients (5.0%) and 32 of 800 usual-care patients (4.0%):

```text
RD = 0.0500 - 0.0400 = +0.0100 = +1.00 percentage point
SE = sqrt(0.05×0.95/800 + 0.04×0.96/800)
   = sqrt(0.00005938 + 0.00004800)
   = sqrt(0.00010738) = 0.010362
95% CI = 0.0100 +/- 1.96 × 0.010362
       = (-0.01031, +0.03031) = (-1.03, +3.03) percentage points
```

The interval crosses zero, so the NNH interval is disconnected:

```text
NNH point estimate = 1 / 0.0100 = 100
Interval: NNTH 33 to infinity to NNTB 97
```

The correct statement is "NNTH 100 (95% CI NNTH 33 to NNTB 97) over 12 months." The data are compatible with a harm requiring 33 patients to harm one, a benefit requiring 97 patients to help one, and no effect.

### 6.7 The population and missingness

The trial randomized 1,600 patients: 800 to the program and 800 to usual care. The primary analysis was intention-to-treat. Primary outcome data were missing for 38 patients (2.4%): 15 in the program arm and 23 in usual care. Missing outcomes were imputed under a missing-at-random assumption with 20 imputations, and a delta-adjusted sensitivity analysis shifted the imputed outcomes by -5 to +5 percentage points. The conclusion was unchanged across the range.

### 6.8 The results paragraph

"The program was associated with a lower 12-month readmission risk: 12.0% versus 18.0%, a risk difference of -6.0 percentage points (95% CI -9.5 to -2.5) and a risk ratio of 0.67 (95% CI 0.52 to 0.85). The number needed to treat was 17 (95% CI 11 to 40) over 12 months. Serious adverse events occurred in 5.0% versus 4.0% (risk difference +1.0 percentage point, 95% CI -1.0 to +3.0; NNTH 100, 95% CI NNTH 33 to NNTB 97). The analysis was intention-to-treat with 2.4% missing primary outcomes, imputed under a missing-at-random assumption; a delta-adjusted sensitivity analysis did not change the conclusion."

### 6.9 Diagnostic checks

**Check 1. Counts reconcile.** 96 + 704 = 800; 144 + 656 = 800; 240 + 1,360 = 1,600. The overall risk of 15.0% lies between the group risks.

**Check 2. The RD and RR agree in direction and magnitude.** A 6-point reduction from an 18% baseline is a 33.3% relative reduction: 0.06/0.18 = 0.3333, and 1 - 0.3333 = 0.6667. The RR of 0.67 matches.

**Check 3. The OR is farther from 1 than the RR.** 0.62 versus 0.67, as expected for a common outcome. If the OR were closer to 1 than the RR, the coding would be suspect.

**Check 4. The NNT is the reciprocal of the RD.** 1/0.06 = 16.67. If the NNT did not match, the RD or the NNT would be wrong.

**Check 5. The NNT interval is the reciprocal of the RD interval, reversed.** 1/0.09487 = 10.54; 1/0.02513 = 39.79. The order is correct: the smaller NNT corresponds to the larger absolute reduction.

**Check 6. The harms interval crosses zero and the NNH notation reflects it.** A report of "NNH 100 (95% CI 33 to 97)" would be wrong, because it would imply a benefit at the upper limit. The NNTB/NNTH notation is required.

### 6.10 Vary one assumption: the baseline risk is lower

Suppose the same risk ratio of 0.67 applies in a population with a 2% baseline risk. Then:

```text
Control risk = 2.00%
Treatment risk = 0.02 × 0.6667 = 1.3333%
RD = -0.6667 percentage points
NNT = 1 / 0.006667 = 150
```

The relative effect is unchanged, and the absolute effect falls from 6 points to 0.67 points. The NNT rises from 17 to 150. A report that gives only the relative effect would suggest the same benefit in both populations; the absolute effect shows that the program prevents one readmission for every 150 patients treated in the low-risk population, versus every 17 in the trial population.

This is the material variation that makes the absolute scale necessary. The relative effect is transportable; the absolute effect is not, and the decision depends on the absolute effect.

| Population | Control risk | Treatment risk | RD | NNT |
|---|---:|---:|---:|---:|
| Trial | 18.0% | 12.0% | -6.0 pp | 17 |
| Low risk | 2.0% | 1.33% | -0.67 pp | 150 |

A report that gives only the relative effect implies that the program is equally valuable in both populations. The table shows that it is nine times less efficient in the low-risk population, which changes the budget calculation.

### 6.11 The spin check

A draft abstract read: "The program reduced readmissions by 33% (p < 0.05), with a favorable safety profile."

The problems:

- "Reduced" implies causation; the trial supports a causal claim under randomization, so this is acceptable, but the magnitude is relative only.
- "33%" omits the absolute reduction and the NNT.
- "p < 0.05" replaces the interval.
- "Favorable safety profile" is not supported by the data: the serious adverse event rate was numerically higher, with an interval that includes both harm and benefit.

A revised abstract conclusion: "The program lowered the 12-month readmission risk by 6.0 percentage points (95% CI 2.5 to 9.5), corresponding to an NNT of 17 (95% CI 11 to 40). Serious adverse events were 1.0 percentage point more frequent (95% CI -1.0 to +3.0), a difference compatible with harm or no effect. The benefit-risk balance favors the program for readmission but requires monitoring for serious adverse events."

### 6.12 What to report

Report the absolute effect first, the relative effect second, the NNT third, and the harms fourth. Report the population and missingness in a sentence. Report the sensitivity analysis. Do not report a p-value without an interval.

The estimate to lead with is the risk difference of -6.0 percentage points (95% CI -9.5 to -2.5), because it is the quantity that maps to the budget and the clinical conversation. The risk ratio of 0.67 (0.52 to 0.85) supports the relative claim, and the NNT of 17 (11 to 40) makes the absolute effect concrete.

### 6.13 A benefit-risk table

| Outcome | Program | Usual care | RD (95% CI) | NNT/NNH (95% CI) |
|---|---:|---:|---|---|
| Readmission | 12.0% | 18.0% | -6.0 pp (-9.5, -2.5) | NNT 17 (11, 40) |
| Serious adverse event | 5.0% | 4.0% | +1.0 pp (-1.0, +3.0) | NNTH 100 (NNTH 33 to NNTB 97) |

The table is the report a decision-maker needs. It puts the benefit and the harm on the same scale, with the same horizon and the same precision. A narrative that mentions only the first row is incomplete; a narrative that mentions only the second is misleading.

The benefit-risk judgment is not in the table. It depends on how the decision-maker weighs a readmission against a serious adverse event, and that weight is a value judgment, not a statistical one. The table provides the quantities; the judgment belongs to the decision.

The report should therefore:

- State the benefit and the harm with equal precision.
- Avoid adjectives that prejudge the balance.
- State the horizon for both.
- Invite the decision-maker to apply their own weights.

## 7. Code, query, or calculation

The script computes the absolute and relative effects, the confidence intervals, the NNT, and the NNTB/NNTH notation, and it shows how the absolute effect changes with the baseline risk. It uses only the standard library.

**Assumptions:**

- The outcome is binary and the groups are independent.
- The confidence intervals use the normal approximation.
- The NNT interval is the reciprocal of the risk difference interval.
- The baseline-risk examples apply the trial risk ratio to a different control risk.

```python
import math
from statistics import NormalDist

z = NormalDist().inv_cdf(0.975)

def report(e1, n1, e0, n0, label, harmful=False):
    p1, p0 = e1 / n1, e0 / n0
    rd = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    lo, hi = rd - z * se, rd + z * se
    rr = p1 / p0
    se_log = math.sqrt(1/e1 - 1/n1 + 1/e0 - 1/n0)
    orr = (e1 / (n1 - e1)) / (e0 / (n0 - e0))
    se_or = math.sqrt(1/e1 + 1/(n1-e1) + 1/e0 + 1/(n0-e0))
    print(f"{label}:")
    print(f"  risks {p1:.4f} vs {p0:.4f}; RD={rd:.4f} "
          f"CI=({lo:.4f}, {hi:.4f})")
    print(f"  RR={rr:.4f} CI=({math.exp(math.log(rr)-z*se_log):.4f}, "
          f"{math.exp(math.log(rr)+z*se_log):.4f})")
    print(f"  OR={orr:.4f} CI=({math.exp(math.log(orr)-z*se_or):.4f}, "
          f"{math.exp(math.log(orr)+z*se_or):.4f})")
    if lo < 0 < hi:
        print(f"  interval crosses zero: NNTB {1/abs(lo):.0f} to infinity "
              f"to NNTH {1/abs(hi):.0f}")
    else:
        print(f"  NNT={1/abs(rd):.1f} CI=({1/abs(lo):.1f}, {1/abs(hi):.1f})")

report(96, 800, 144, 800, "readmission")
report(40, 800, 32, 800, "serious adverse event")

# Absolute effect under a different baseline risk
rr = 0.12 / 0.18
for p0 in (0.02, 0.18, 0.50):
    p1 = p0 * rr
    rd = p1 - p0
    print(f"baseline {p0:.2f}: treatment {p1:.4f} RD={rd:.4f} NNT={1/abs(rd):.1f}")
```

**Expected output:**

```text
readmission:
  risks 0.1200 vs 0.1800; RD=-0.0600 CI=(-0.0949, -0.0251)
  RR=0.6667 CI=(0.5250, 0.8466)
  OR=0.6212 CI=(0.4698, 0.8214)
  NNT=16.7 CI=(10.5, 39.8)
serious adverse event:
  risks 0.0500 vs 0.0400; RD=0.0100 CI=(-0.0103, 0.0303)
  RR=1.2500 CI=(0.7935, 1.9690)
  OR=1.2632 CI=(0.7851, 2.0323)
  interval crosses zero: NNTB 97 to infinity to NNTH 33
baseline 0.02: treatment 0.0133 RD=-0.0067 NNT=150.0
baseline 0.18: treatment 0.1200 RD=-0.0600 NNT=16.7
baseline 0.50: treatment 0.3333 RD=-0.1667 NNT=6.0
```

**What the output shows:**

- The risk difference is -6.0 points and the NNT is 16.7, which rounds to 17. The interval is (10.5, 39.8), which rounds to (11, 40).
- The risk ratio is 0.6667 with an interval (0.5250, 0.8466), and the odds ratio is 0.6212 with an interval (0.4698, 0.8214). The OR is farther from 1 because the outcome is common.
- The serious adverse event interval crosses zero, and the script reports the disconnected interval as NNTB 97 to infinity to NNTH 33. A report that writes "NNH 100 (33 to 97)" would be wrong.
- The baseline-risk lines show the absolute effect and NNT at three baseline risks. The relative effect is held constant at 0.6667, and the NNT ranges from 6 to 150.

> **Note:** The script reports the NNT interval by taking reciprocals of the risk difference limits and reversing the order. When the risk difference interval crosses zero, the reciprocal transformation produces a disconnected interval, and the script labels both sides. A production report should use a method with better coverage, such as the Newcombe hybrid score interval for the risk difference, before transforming.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| CONSORT 2010 item 17a | Journal editorial policy | Effect size and precision for each outcome | The choice of effect measure |
| CONSORT 2010 item 17b | Journal editorial policy | Both absolute and relative effects for binary outcomes | The method of calculation |
| CONSORT 2010 item 18 | Journal editorial policy | Subgroup and adjusted analyses, pre-specified versus exploratory | The subgroup method |
| SAMPL | Journal editorial policy | Basic statistical reporting: numbers, tests, intervals | The analysis |
| STROBE | Journal editorial policy | Reporting of observational studies | The estimator |
| ICH E9 (1998) | Binding through regional adoption | Precision, confidence intervals, multiplicity | The reporting format |
| ICH E9(R1) (2019) | Binding through regional adoption | The estimand reported with the estimate | The table design |
| FDA Multiple Endpoints guidance (2022) | Final guidance | Multiplicity reporting for endpoints | The subgroup presentation |

### 8.2 What CONSORT requires

CONSORT item 17a requires "for each primary and secondary outcome, results for each group, and the estimated effect size and its precision (such as 95% confidence interval)." Item 17b requires "for binary outcomes, presentation of both absolute and relative effect sizes."

The two items together define the minimum for a binary primary outcome: the group risks, the absolute effect, the relative effect, and the interval. A report that gives only the relative effect has not satisfied item 17b. A report that gives only a p-value has not satisfied item 17a.

### 8.3 What SAMPL requires

The SAMPL guidelines specify the minimum statistical reporting for biomedical journals. They cover:

- The numbers analyzed and the descriptive statistics.
- The effect sizes and confidence intervals.
- The tests used and the assumptions.
- The handling of missing data and outliers.
- The reporting of survival analyses and regression models.

SAMPL is a checklist for authors and editors, not a regulatory requirement. Its value is that it names the items that are often omitted, including the interval and the analysis population.

### 8.4 What ICH E9 and E9(R1) require

ICH E9 requires the primary analysis to be pre-specified, the effect to be reported with a confidence interval, and the multiplicity to be controlled. ICH E9(R1) requires the estimand to be reported with the estimate, so that the reader knows what quantity the number refers to.

A report that gives an estimate without the estimand, the population, or the interval is incomplete under both guidelines, even if the arithmetic is correct.

### 8.5 What a results section must contain

| Element | Required by | Example |
|---|---|---|
| Group risks or means | CONSORT 17a | 12.0% vs 18.0% |
| Absolute effect | CONSORT 17b | -6.0 percentage points |
| Relative effect | CONSORT 17b | RR 0.67 |
| Interval | CONSORT 17a; SAMPL | 95% CI -9.5 to -2.5 |
| NNT | Convention | 17 (11 to 40) over 12 months |
| Harms | CONSORT 19 | 5.0% vs 4.0% |
| Analysis population | CONSORT 16 | Intention-to-treat, 1,600 |
| Missingness | SAMPL; STROBE | 2.4%, MI with 20 imputations |
| Multiplicity | ICH E9 | Holm across 3 secondary endpoints |
| Subgroup status | CONSORT 18 | Exploratory |

The table is a crosswalk between the standards and the quantities. A results section that contains all ten is complete under the guidelines cited.

### 8.6 Reporting by study type

| Study type | Guideline | Additional items |
|---|---|---|
| Randomized trial | CONSORT | Flow diagram, randomization, blinding, harms |
| Observational cohort | STROBE | Confounding control, selection, follow-up |
| Routinely collected data | REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) | Code lists, linkage, validation |
| Diagnostic accuracy | STARD | Sensitivity, specificity, spectrum |
| Prediction model | TRIPOD+AI | Calibration, validation, subgroups |
| Systematic review | PRISMA | Search strategy, risk of bias, synthesis |
| Real-world evidence | TARGET plus STROBE/RECORD | Target trial protocol, emulation gaps |

The study type determines the guideline, but the statistical core is the same: an estimate, an interval, a population, and a conclusion that matches the design.

### 8.7 What the standards do not cover

- They do not require the NNT.
- They do not require the harms to be on the same scale as the benefits.
- They do not require a benefit-risk table.
- They do not prohibit spin; they require the content that makes spin detectable.
- They do not validate the analysis; they standardize the report.

## 9. Industry practice and case evidence

### 9.1 Documented evidence: spin in abstracts

Boutron, Dutton, Ravaud, and Altman, in a study published in JAMA in 2010, examined 72 randomized trials with a statistically non-significant primary outcome. They found spin in 18.0% of titles, 37.5% of abstract results, and 58.3% of abstract conclusions. More than 40% of the reports had spin in at least two main-text sections.

The finding is the empirical case for structured reporting. The abstract conclusion is what most readers see, and it is where the overstatement is most common.

### 9.2 Documented practice: CONSORT and the reporting of absolute effects

CONSORT item 17b was added because relative effects were routinely reported without absolute effects. The requirement is now widely adopted by journals, but compliance is incomplete. A review of trial reports would still find abstracts that report a relative reduction without a baseline risk.

**Inference:** the frequency of non-compliance is not documented in a single current statistic. The requirement's existence is evidence that the omission was common enough to warrant a standard.

### 9.3 Documented practice: the NNT in clinical communication

The NNT was popularized as a communication tool because it translates a risk difference into a number of patients. Altman, in a 1998 BMJ paper, showed how to compute its confidence interval and warned about the disconnected interval when the risk difference crosses zero. The NNTB/NNTH notation is the standard response.

The NNT is not a substitute for the risk difference. It is a transformation, and it inherits the risk difference's dependence on the baseline risk and the horizon.

### 9.4 Documented case: the harm that was not reported

A trial report that emphasizes a statistically significant benefit and reports harms only as a list of adverse events without denominators makes the benefit-risk comparison impossible. The omission is a reporting failure even when the data are complete.

The remedy is a benefit-risk table with the same population, the same horizon, and the same precision for both. The table does not make the decision; it makes the decision possible.

### 9.5 Practice pattern: the results slide

An industry results slide often shows a relative effect, a p-value, and a logo. The absolute effect, the interval, the population, and the harms are in the appendix. The slide is the artifact that the decision-maker remembers.

The reporting standard should apply to the slide as well as the paper. A slide that cannot be interpreted without the appendix is a failed slide.

### 9.6 Documented fact versus inference

| Claim | Status | Basis |
|---|---|---|
| Spin is common in non-significant trials | Documented | Boutron et al. 2010 |
| Relative effects are reported more often than absolute effects | Documented in the reporting literature | CONSORT item 17b rationale |
| The NNT interval is disconnected when the RD crosses zero | Documented | Altman 1998 |
| A report that follows CONSORT is a good study | Not supported | Reporting is not quality |
| A p-value is sufficient for a decision | Not supported | The interval carries the effect and precision |
| An NNT is comparable across baseline risks | Not supported | The NNT depends on the baseline risk |

The first three rows are the evidence. The last three are the inferences the standards do not support.

### 9.7 The cost of poor reporting

Poor reporting costs:

- A clinician cannot compute the NNT and defaults to the relative effect.
- A payer cannot model the budget and defaults to the list price.
- A reviewer cannot reproduce the analysis and rejects the paper.
- A regulator cannot verify the estimand and asks for a reanalysis.
- A patient receives a treatment whose benefit was overstated.

**Inference:** the costs are not measured in a single study. The existence of CONSORT, SAMPL, and the spin literature is evidence that the costs were recurring.

The costs fall disproportionately on readers who cannot access the data:

- Clinicians who rely on the abstract.
- Payers who rely on the published estimate.
- Patients who rely on the clinician.
- Reviewers who cannot reproduce the analysis.

### 9.8 What mature teams do differently

1. They write the results table before the analysis.
2. They report the absolute effect first.
3. They report the interval on every estimate.
4. They report the harms on the same scale.
5. They state the population and missingness in a sentence.
6. They run a spin check on the abstract.
7. They keep the conclusion within the data.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| p-value without effect size | "p = 0.03" only | The estimate is omitted | The reader cannot judge importance | Report the effect and interval |
| Relative effect only | "33% reduction" | The baseline risk is omitted | Benefit overstated | Report the RD and NNT |
| No interval | A point estimate alone | Precision is hidden | Overconfidence | Report the 95% CI |
| NNT without horizon | "NNT 17" | The time frame is omitted | Uninterpretable | State the horizon |
| NNT interval wrong | "NNT 17 (11 to 40)" when the RD crosses zero | The reciprocal is disconnected | Misleading | Use NNTB/NNTH |
| Harms omitted | Only benefits reported | Incomplete benefit-risk | Wrong decision | Report harms with denominators |
| Population unclear | No analysis set | The estimate is conditional | Misapplication | State ITT or per-protocol |
| Missingness omitted | No missing-data statement | The assumption is hidden | Overconfidence | Report the method |
| Multiplicity ignored | Many nominal p-values | The family is not controlled | False findings | Report adjusted p-values |
| Subgroup promoted | "Effective in subgroup" | The interaction is omitted | False claim | Report the interaction |
| Spin | "Trend toward benefit" | Language overstates the data | Wrong belief | Structured spin check |
| Decimal overprecision | "0.42371" | False precision | Distracts from uncertainty | Round to the estimate's precision |
| Composite hiding components | A significant composite | Components move in opposite directions | Misleading | Report the components |
| No baseline table | The groups cannot be compared | Confounding or imbalance is hidden | Uninterpretable | Report Table 1 |
| Conclusion beyond data | "The drug prevents stroke" | Causal language without support | Overclaim | Match the claim to the design |
| No table 1 | The groups cannot be compared | Baseline imbalance is hidden | Uninterpretable | Report baseline characteristics |
| Time horizon omitted | "NNT 17" without a period | The effect is time-dependent | Misapplied | State the horizon |
| Confidence level omitted | "CI 0.52 to 0.85" | The level is ambiguous | Misread | State 95% |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The estimate is correctly computed | The report depends on it | Arithmetic or model error | Recompute; check |
| The interval has nominal coverage | The uncertainty statement depends on it | Model misspecification | Alternative methods |
| The population is representative | The estimate is applied to a population | Selection or exclusion | Compare with the target |
| The missing-data assumption is stated | The estimate depends on it | Unstated MNAR | Sensitivity analysis |
| The effect measure is appropriate | The interpretation depends on it | OR read as RR | Report both scales |
| The horizon is relevant | The decision depends on it | A different horizon | Report the horizon |
| The harms are completely ascertained | The benefit-risk balance depends on it | Underreporting | Active surveillance |
| The multiplicity is controlled | The significance claim depends on it | Many tests | Adjusted p-values |
| The subgroup is pre-specified | The claim depends on it | Post hoc analysis | Protocol check |
| The conclusion matches the design | The causal claim depends on it | Observational data | Match language to design |
| The confidence level is stated | The interval is interpretable | Ambiguous level | State 95% |
| The horizon is stated | The effect is time-dependent | Omitted horizon | Report the period |

**Limits of this topic.** A report cannot make a biased analysis valid; it can only make the bias visible. The reporting standards are checklists, not quality scores, and a report that satisfies every item can still be wrong. The worked example is a single trial with a binary outcome; other designs require different tables and different quantities. The NNT is a communication tool, not a decision rule, and it depends on the baseline risk and the horizon.

## 12. Exercises

### Exercise 1. Find the missing elements (recall, 30 minutes)

**Paragraph.** "The treatment was effective (p = 0.02). There were fewer events in the treatment group, and the relative reduction was 28%."

**Deliverable.** A list of at least six missing elements and a revised paragraph.

**Acceptance criteria.**

- The estimand, population, absolute effect, interval, NNT, and harms are identified as missing.
- The revised paragraph includes the absolute effect and the interval.
- The p-value is retained only as a secondary statement.
- The claim is limited to what the design supports.

### Exercise 2. Compute the reporting quantities (application, 45 minutes)

**Data.** Treatment: 45 events / 500. Control: 70 / 500.

**Deliverable.** The group risks, risk difference, risk ratio, odds ratio, NNT, and all intervals.

**Acceptance criteria.**

- Each quantity is computed with the formula.
- The intervals are on the correct scale.
- The NNT interval is the reciprocal of the RD interval.
- The OR is labelled as an OR and not interpreted as an RR.

### Exercise 3. NNT when the interval crosses zero (application, 35 minutes)

**Data.** Risk difference +1.0 percentage point, 95% CI -1.0 to +3.0.

**Deliverable.** The NNH and the NNTB/NNTH notation for the interval.

**Acceptance criteria.**

- The point estimate is 100.
- The interval is reported as NNTH 33 to infinity to NNTB 100.
- The report does not write "NNH 100 (33 to 100)".
- The interpretation includes harm, benefit, and no effect.

### Exercise 4. Report harms (application, 40 minutes)

**Data.** Serious adverse events: 25/400 treatment, 18/400 control. Headache: 60/400 treatment, 30/400 control.

**Deliverable.** A harms table with risks, risk differences, intervals, and NNT/NNH.

**Acceptance criteria.**

- Both harms are reported with denominators.
- The intervals are computed.
- The table distinguishes serious from common events.
- The benefit-risk implication is stated without prejudging the weights.

### Exercise 5. Rewrite a spun abstract (critique, 50 minutes)

**Claim.** "The drug demonstrated a trend toward reducing mortality (p = 0.08) and significantly improved quality of life (p = 0.04). These findings support a favorable risk-benefit profile."

**Deliverable.** A rewritten abstract conclusion and a list of the spin patterns.

**Acceptance criteria.**

- The "trend" language is removed.
- The primary outcome is named and reported with its estimate and interval.
- The secondary outcome is labelled secondary.
- The risk-benefit claim is supported by a harms table or removed.
- The revised conclusion is no longer than the original.

### Exercise 6. Design Table 2 (design, 60 minutes)

**Scenario.** A trial with one primary binary outcome, three secondary outcomes, and a harms section.

**Deliverable.** A table shell with columns for group risks, absolute effect, relative effect, interval, and NNT.

**Acceptance criteria.**

- The primary outcome is first and labelled.
- The absolute and relative effects are both present.
- The intervals are present for every estimate.
- The harms are in a separate section with denominators.
- The table can be read without the text.

### Exercise 7. Write a decision memo (design, 60 minutes)

**Scenario.** A payer is deciding whether to cover the program from Section 6.

**Deliverable.** A one-page memo with the estimate, the interval, the NNT, the harms, the budget arithmetic, and the recommendation.

**Acceptance criteria.**

- The absolute effect is the lead quantity.
- The budget arithmetic uses the eligible population.
- The harms are included.
- The recommendation is conditional on the evidence.
- The memo states what would change the decision.

### Exercise 8. Write a full results section (design, 120 minutes)

**Scenario.** A completed analysis of a program with a binary primary outcome, two secondary outcomes, and a harms table.

**Deliverable.** A results section with a flow paragraph, Table 1, Table 2, a harms table, and a conclusion.

**Acceptance criteria.**

- The estimand is stated.
- The population and missingness are stated.
- The primary result has absolute and relative effects and an interval.
- The harms are on the same scale.
- The conclusion does not exceed the data.
- The section passes the ten-item reporting checklist.
- The flow paragraph gives the number screened, eligible, randomized, and analyzed.
- The primary outcome is named before the secondary outcomes.
- The NNT has a horizon.
- The harms table has denominators.
- The conclusion is one sentence.
- The sensitivity analysis is summarized in a sentence.
- The table and figure references are in order.
- The abstract conclusion matches the results.
- The section avoids "trend" and "significant" without a referent.
- The section states the confidence level.
- The section rounds to a sensible precision.

## 13. Self-check answers

**Answer 1 (Exercise 2 quantities).**

```text
p1 = 45/500 = 0.0900; p0 = 70/500 = 0.1400
RD = -0.0500 = -5.00 percentage points
SE = sqrt(0.09×0.91/500 + 0.14×0.86/500)
   = sqrt(0.0001638 + 0.0002408) = 0.02011
95% CI = (-0.0894, -0.0106) = (-8.94, -1.06) percentage points

RR = 0.09/0.14 = 0.6429
SE(log RR) = sqrt(1/45 - 1/500 + 1/70 - 1/500) = 0.17687
95% CI = exp(-0.4418 +/- 0.3467) = (0.4541, 0.9100)

OR = (45/455)/(70/430) = 0.09890/0.16279 = 0.6075
SE(log OR) = sqrt(1/45 + 1/455 + 1/70 + 1/430) = 0.19802
95% CI = exp(-0.4983 +/- 0.3881) = (0.4132, 0.8933)

NNT = 1/0.05 = 20
95% CI = (1/0.0894, 1/0.0106) = (11.2, 94.3), reported as (12, 95)
```

The NNT is 20, with an interval from 12 to 95. The interval is wide because the absolute effect is small relative to its standard error.

**Answer 2 (Exercise 3 NNT).**

```text
Point estimate: 1/0.01 = 100 (NNH)
Lower limit: -1.0 pp, a benefit of 1.0 pp -> NNTB 100
Upper limit: +3.0 pp, a harm of 3.0 pp -> NNTH 33
```

The correct statement is "NNTH 100 (95% CI NNTH 33 to NNTB 100)." The interval runs from a harm of 1 in 33 to a benefit of 1 in 100, through infinity. Writing "33 to 100" implies that both limits are harms, which is wrong.

**Answer 3 (Exercise 4 harms).**

```text
Serious adverse event:
  6.25% vs 4.50%; RD = +1.75 pp
  SE = sqrt(0.0625×0.9375/400 + 0.045×0.955/400) = 0.01575
  95% CI = (-0.0012, +0.0362)
  NNTH 57 (NNTH 28 to NNTB 833)

Headache:
  15.0% vs 7.5%; RD = +7.5 pp
  SE = sqrt(0.15×0.85/400 + 0.075×0.925/400) = 0.01984
  95% CI = (+3.61, +11.39) pp
  NNTH 13 (9, 28)
```

The serious adverse event interval includes zero, so the NNTB/NNTH notation is required. The headache interval excludes zero, so the NNTH is reported directly. The two harms are on different scales of clinical importance, which is why they are reported separately.

**Answer 4 (Exercise 5 spin).**

The spin patterns:

- "Trend toward" for p = 0.08.
- "Significantly improved" for a secondary outcome, presented without the primary result.
- "Favorable risk-benefit profile" without a harms table.

A revised conclusion: "The primary outcome did not differ significantly between groups (estimate and 95% CI). Quality of life improved by X points (95% CI Y to Z) as a secondary outcome. Serious adverse events occurred in A% versus B% (95% CI). The risk-benefit balance is not established by these data."

**Answer 5 (Exercise 1 missing elements).**

- The estimand and population.
- The group risks.
- The absolute effect.
- The confidence interval.
- The NNT.
- The harms.
- The analysis population and missingness.
- The multiplicity and subgroup status.

The revised paragraph reports the risk difference and interval, the risk ratio, the NNT, and the harms, and labels the p-value as secondary.

**Answer 6 (Exercise 7 budget arithmetic).**

For 3,000 eligible patients and a 6.0-percentage-point reduction:

```text
Events avoided = 0.06 × 3,000 = 180
Saving at USD 12,000 per event = 180 × 12,000 = USD 2,160,000
Program cost = USD 2,400,000
Net = -USD 240,000
```

At the lower confidence limit (2.5 points), the saving is 0.025 × 3,000 × 12,000 = USD 900,000. At the upper limit (9.5 points), the saving is USD 3,420,000. The decision is sensitive to the interval, and the memo should say so.

**Answer 7 (Exercise 6 table shell).**

| Outcome | Program | Control | RD (95% CI) | RR (95% CI) | NNT (95% CI) |
|---|---:|---:|---|---|---|
| Primary: readmission | 12.0% | 18.0% | -6.0 (-9.5, -2.5) | 0.67 (0.52, 0.85) | 17 (11, 40) |
| Secondary: death | ... | ... | ... | ... | ... |
| Secondary: quality of life | ... | ... | ... | ... | ... |
| Harms: serious adverse event | 5.0% | 4.0% | +1.0 (-1.0, +3.0) | 1.25 (0.79, 1.97) | NNTH 100 (NNTH 33 to NNTB 97) |

The shell shows the required columns. A table without the absolute column, or without the interval, would fail the checklist.

**Answer 8 (Exercise 8 rubric).**

A strong results section:

- Opens with the flow of participants and the analysis population.
- States the estimand.
- Reports the primary outcome with group risks, RD, RR, intervals, and NNT.
- Reports the secondary outcomes with the multiplicity adjustment.
- Reports the harms with denominators and intervals.
- Reports the missingness and the sensitivity analysis.
- Concludes with a sentence that matches the data and the design.

A weak section reports p-values, omits the absolute effect, and ends with a causal claim the design cannot support.

The rubric can be applied to a published paper in about ten minutes:

- Is the estimand stated in the first two paragraphs?
- Is the primary outcome distinguishable from the secondary outcomes?
- Is there an absolute effect?
- Is there an interval?
- Is the NNT present, with a horizon?
- Are the harms present, with denominators?
- Is the analysis population stated?
- Is the missingness stated?
- Is the multiplicity stated?
- Does the conclusion match the data?

## 14. Interview questions

**1. What must a results paragraph contain?**

A strong answer lists the estimand, population, absolute and relative effects, interval, NNT, and harms. A weak answer says "the result and the p-value."

**2. Why report both absolute and relative effects?**

A strong answer explains that the relative effect shows strength and the absolute effect shows population impact, and gives an example where the relative effect is constant but the NNT varies. A weak answer says "because guidelines require it."

**3. How do you compute the NNT interval?**

A strong answer says take the reciprocals of the risk difference limits and reverse the order, and notes the disconnected interval when the RD crosses zero. A weak answer gives a symmetric interval.

**4. Why is the NNT interval disconnected when the RD crosses zero?**

A strong answer explains the reciprocal transformation and the NNTB/NNTH notation. A weak answer says "because of the math."

**5. What is spin, and how do you detect it?**

A strong answer defines it and lists the common patterns, including trend language, secondary-as-primary, and relative-only reporting. A weak answer says "exaggeration."

**6. How do you report harms?**

A strong answer says on the same scale, with the same population and horizon, with denominators and intervals. A weak answer says "list the adverse events."

**7. What is the difference between statistical and clinical significance?**

A strong answer distinguishes the p-value from the effect size and the MCID, and notes that a significant result can be trivial. A weak answer says they are "different."

**8. Why is the analysis population part of the estimate?**

A strong answer explains that ITT, per-protocol, and complete-case answer different questions and can give different estimates. A weak answer says "it is a methods detail."

**9. How do you report multiplicity?**

A strong answer states the family, the procedure, and the adjusted p-value, and labels exploratory analyses. A weak answer says "we adjusted for multiple comparisons."

**10. What is a benefit-risk table?**

A strong answer describes a table with benefits and harms on the same scale, with intervals. A weak answer says "a table of side effects."

**11. Why is a p-value not enough for a decision?**

A strong answer says it lacks the effect size, the precision, and the absolute scale, and it cannot be converted into a budget or a patient conversation. A weak answer says "p-values are misleading."

**12. What would make you distrust a results section?**

A strong answer lists p-only reporting, relative-only reporting, no interval, no harms, unclear population, and spin. A weak answer says "a small sample."

## 15. Key terms

- **Absolute effect:** a difference in risks or means.
- **Adjusted p-value:** a p-value after multiplicity control.
- **Analysis population:** the set of patients included in the analysis.
- **Benefit-risk:** the balance of benefits and harms.
- **Clinical importance:** an effect large enough to matter.
- **CONSORT:** the reporting guideline for randomized trials.
- **Effect size:** the magnitude of the difference between groups.
- **Exploratory:** hypothesis-generating.
- **Horizon:** the time over which the effect is measured.
- **Intention-to-treat:** analysis by assigned group.
- **Minimal clinically important difference:** the smallest important effect.
- **Missingness:** the proportion of missing outcome data.
- **Multiplicity:** the number of hypotheses tested.
- **NNT:** the number needed to treat for one additional benefit.
- **NNH:** the number needed to harm for one additional harm.
- **NNTB:** number needed to treat for benefit.
- **NNTH:** number needed to treat for harm.
- **Odds ratio:** the ratio of odds.
- **Per-protocol:** analysis by treatment received.
- **Precision:** the width of the interval.
- **Pre-specified:** defined before the analysis.
- **Relative effect:** a ratio of risks or rates.
- **Reporting guideline:** a standard for the content of a report.
- **Risk difference:** the difference in outcome proportions.
- **Risk ratio:** the ratio of outcome proportions.
- **SAMPL:** the statistical reporting guideline.
- **Spin:** reporting that overstates the result.
- **STROBE:** the reporting guideline for observational studies.
- **Statistical significance:** a p-value below a threshold.
- **Hazard ratio:** the ratio of hazards.
- **Confidence interval:** the range of effects compatible with the data.
- **Baseline risk:** the outcome risk in the control group.
- **Benefit-risk table:** a table of benefits and harms on the same scale.
- **Disconnected interval:** an interval that runs through infinity.
- **Natural frequency:** a count per fixed number of patients.
- **Overprecision:** reporting more decimal places than the data support.
- **Transportability:** whether an effect applies to another population.
- **Reporting bias:** the selective reporting of results.
- **Selective outcome reporting:** reporting only the favorable outcomes.
- **Outcome switching:** changing the primary outcome after the analysis.
- **Publication bias:** the selective publication of positive results.
- **Results table:** a table of estimates, intervals, and populations.
- **Effect measure scale:** the units in which the effect is reported.
- **Decision sentence:** the one-sentence conclusion supported by the data.
- **Report template:** a pre-specified structure for the results.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. Schulz, K. F., Altman, D. G., Moher, D., for the CONSORT Group. "CONSORT 2010 Statement: Updated Guidelines for Reporting Parallel Group Randomised Trials." *PLoS Medicine* 2010;7(3):e1000251. https://doi.org/10.1371/journal.pmed.1000251
2. Moher, D., Hopewell, S., Schulz, K. F., et al. "CONSORT 2010 Explanation and Elaboration." *BMJ* 2010;340:c869. https://doi.org/10.1136/bmj.c869
3. Lang, T. A., Altman, D. G. "Basic Statistical Reporting for Articles Published in Biomedical Journals: The SAMPL Guidelines." EQUATOR Network, 2013. https://www.equator-network.org/2013/03/20/basic-statistical-reporting-for-articles-published-in-biomedical-journals-the-statistical-analyses-and-methods-in-the-published-literature-or-the-sampl-guidelines/
4. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
5. International Council for Harmonisation. *ICH E9: Statistical Principles for Clinical Trials*. Step 4, 5 February 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
6. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
7. United States Food and Drug Administration. *Multiple Endpoints in Clinical Trials: Guidance for Industry*. October 2022. https://www.fda.gov/media/162416/download
8. Boutron, I., Dutton, S., Ravaud, P., Altman, D. G. "Reporting and interpretation of randomized controlled trials with statistically nonsignificant results for primary outcomes." *JAMA* 2010;303(20):2058-2064. https://doi.org/10.1001/jama.2010.651
9. Altman, D. G. "Confidence intervals for the number needed to treat." *BMJ* 1998;317(7168):1309-1312. https://doi.org/10.1136/bmj.317.7168.1309
10. Chiu, K., Grundy, Q., Bero, L. "'Spin' in published biomedical literature: A methodological systematic review." *PLoS Biology* 2017;15(9):e2002173.
11. Cochrane Collaboration. *Cochrane Handbook for Systematic Reviews of Interventions*, Chapter 15: Interpreting results and drawing conclusions. https://training.cochrane.org/handbook
12. Rothman, K. J., Greenland, S., Lash, T. L. *Modern Epidemiology*. 3rd edition. Lippincott Williams and Wilkins, 2008.
13. Altman, D. G., Andersen, P. K. "Calculating the number needed to treat for trials where the outcome is time to an event." *BMJ* 1999;319(7223):1492-1495.
14. Schulz, K. F., Altman, D. G., Moher, D. "CONSORT 2010 statement: updated guidelines for reporting parallel group randomised trials." *BMJ* 2010;340:c332. https://doi.org/10.1136/bmj.c332
15. Ioannidis, J. P. A. "The proposal to lower P value thresholds to .005." *JAMA* 2018;319(14):1429-1430.
16. Wasserstein, R. L., Lazar, N. A. "The ASA Statement on p-Values: Context, Process, and Purpose." *The American Statistician* 2016;70(2):129-133. https://doi.org/10.1080/00031305.2016.1154108
17. Benchimol, E. I., Smeeth, L., Guttmann, A., et al. "The RECORD Statement." *PLoS Medicine* 2015;12(10):e1001885. https://doi.org/10.1371/journal.pmed.1001885

> **Note:** The CONSORT item numbers (17a, 17b, 18, 19) are from the 2010 statement and its explanation and elaboration. The Boutron 2010 figures (18.0% of titles, 58.3% of abstract conclusions) are quoted from the published abstract. The NNT interval method and the NNTB/NNTH convention are from Altman 1998.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_subgroups-and-heterogeneity.md`](09_subgroups-and-heterogeneity.md) — reporting subgroup claims.
- Estimation: [`01_estimation-and-inference.md`](01_estimation-and-inference.md) — intervals and effect scales.
- Sensitivity analysis: [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) — reporting robustness.
- Regression: [`02_regression-models.md`](02_regression-models.md) — reporting adjusted estimates.
- Survey file: [`../../04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — the reporting exercises in Lab 4.6.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md) — units and notation.
- Module 18: [`../../18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md) — quality reporting and evaluation.
- Target trial emulation: [`07_target-trial-emulation.md`](07_target-trial-emulation.md) — the TARGET reporting items.
- Missing data: [`04_missing-data.md`](04_missing-data.md) — reporting missingness and the FMI.
- Propensity scores: [`06_propensity-scores-and-weighting.md`](06_propensity-scores-and-weighting.md) — reporting balance.
- Study design: [`../../05_study_design_evidence.md`](../../course/05_study_design_evidence.md) — design and reporting alignment.
