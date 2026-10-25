# Epidemiology Basics

> Module 01A — Foundations, Assumptions, and Units
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 05
> Version 2026.3

## 1. Why this topic exists

Epidemiology is the study of who gets sick, how often, and why. Its core quantities are simple:
counts, denominators, and time. Its failures are almost always failures of the denominator, the case
definition, or the comparison group.

- A hospital reports that its sepsis mortality "fell from 9.1% to 7.8%." The denominator changed
  because a new coding policy admitted more low-severity patients. Mortality did not improve; the case
  mix changed.
- A quality team compares two hospitals on crude mortality. Hospital A has 6.8% and hospital B has
  8.9%, so A is declared better. After stratification, A has higher mortality in every severity
  stratum, and its lower crude rate is entirely explained by a younger, less severe case mix.
- A central line infection rate is reported as "12 infections this year" without a denominator. The
  unit expanded from 20 to 60 beds, so the count rose while the rate fell.
- A claims analysis finds that patients who see a specialist within 30 days of diagnosis have lower
  mortality. The specialist visit is a marker of access and of less severe disease, and the "effect"
  is healthy-user bias.
- A screening programme finds more early-stage cancers and concludes that it saves lives. The
  five-year survival from diagnosis rises because diagnosis was moved earlier, not because death was
  delayed. This is lead-time bias, and it can make an ineffective programme look transformative.
- An incidence rate is computed with the number of patients as the denominator rather than person-time.
  Patients with different follow-up durations are weighted equally, and the rate is wrong.

These errors share a structure: the numerator, the denominator, and the time window are not defined
consistently, or the comparison group is not comparable. The arithmetic is trivial; the definitions
are not.

The topic matters to data scientists because most real-world evidence is observational. Claims
databases, registries, electronic health records, and wearables all produce data in which the
denominator is ambiguous and the comparison group is self-selected. A model that ignores this produces
estimates that are precise, plausible, and biased.

The commercial consequence is direct. Payer negotiations, quality bonuses, regulatory submissions, and
product claims all rest on epidemiological quantities. An analyst who can compute an age-standardised
rate and explain why it differs from the crude rate is the difference between a defensible claim and
a retraction.

> **Note:** This topic covers descriptive and basic analytical epidemiology. Causal identification is
> Module 07's later treatment and Module 10's ledger; here the focus is on getting the measure right
> before asking whether it is causal.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** prevalence, incidence proportion, and incidence rate, and choose the measure that
   matches a stated question. Assessed in Exercise 1.
2. **Compute** person-time from individual follow-up data and use it as a rate denominator. Assessed
   in Exercise 2.
3. **Compute** crude and stratum-specific rates and explain why they differ. Assessed in Exercise 3.
4. **Standardise** rates directly and indirectly, compute an SMR with a confidence interval, and
   explain when the two methods disagree. Assessed in Exercise 4.
5. **Compute** risk ratios, odds ratios, risk differences, NNT, and attributable fractions from a 2×2
   table, and state which measure fits which design. Assessed in Exercise 5.
6. **Identify** the major biases in an observational comparison and state the direction of the likely
   effect. Assessed in Exercise 6.
7. **Critique** a surveillance rate or a quality metric for denominator, case definition, and
   case-mix problems. Assessed in Exercise 7.
8. **Design** a descriptive analysis plan for a registry, including the case definition, the
   denominator, the standard population, and the sensitivity analyses. Assessed in Exercise 8.

## 3. The core mental model

Every epidemiological measure is a ratio with three components: a count of events, a denominator of
people or person-time, and a time window.

```text
                    POPULATION AT RISK
                    (who is eligible, when)
                              |
              +---------------+---------------+
              |                               |
        CASES (numerator)               NON-CASES
              |                               |
              +---------------+---------------+
                              |
                     DENOMINATOR
              people (prevalence, risk)  or  person-time (rate)
                              |
                              v
                     MEASURE + TIME WINDOW
```

A measure without all three components is not an epidemiological quantity. "Twelve infections" is a
count. "Twelve infections per 1,000 line-days in the first quarter" is a rate. "Four percent of
inpatients had sepsis on admission day" is a prevalence.

### 3.1 The three core measures

| Measure | Numerator | Denominator | Question answered |
|---|---|---|---|
| Prevalence | Existing cases at a time | Population at that time | How much disease is there now? |
| Incidence proportion (risk) | New cases over a period | Population at risk at the start | What is the probability of developing disease? |
| Incidence rate | New cases over a period | Person-time at risk | How fast are new cases arising? |

The distinction between the two incidence measures is the treatment of time. A risk assumes a fixed
follow-up period and a closed cohort; a rate accommodates variable follow-up and open cohorts. When
follow-up differs across individuals, the rate is the appropriate measure, and the risk can be
approximated as 1 − e^(−rate × time) only if the rate is constant.

### 3.2 The 2×2 table

```text
                     Outcome +      Outcome -      Total
   Exposed               a              b          a + b
   Unexposed             c              d          c + d
   Total               a + c          b + d          N
```

From this table:

```text
Risk in exposed      = a/(a + b)
Risk in unexposed    = c/(c + d)
Risk ratio (RR)      = [a/(a+b)] / [c/(c+d)]
Odds ratio (OR)      = (a/b) / (c/d)
Risk difference (RD) = a/(a+b) - c/(c+d)
NNT                  = 1/RD  (when RD > 0)
```

The odds ratio approximates the risk ratio only when the outcome is rare. When the outcome is common,
the OR is further from 1 than the RR, and reporting an OR as a risk ratio overstates the effect.

### 3.3 Standardisation

Crude rates differ between populations for two reasons: the underlying rates differ, and the
populations differ in their composition. Standardisation separates the two.

```text
   Direct:    apply the study population's stratum-specific rates
              to a standard population's structure

   Indirect:  apply a standard population's stratum-specific rates
              to the study population's structure, giving expected events;
              SMR = observed / expected
```

Direct standardisation answers "what would the rate be if this population had the standard age
structure?" Indirect standardisation answers "how many events would we expect if this population
experienced the standard rates, and how does the observed count compare?" The SMR is the ratio of
observed to expected.

The two methods can disagree when the stratum-specific rate ratios are not constant across strata.
This is not a paradox; it is a consequence of weighting. Direct standardisation weights by the
standard population; indirect standardisation weights implicitly by the study population's own
structure.

### 3.4 The causal question is separate

A measure of association is not a measure of effect. The association between exposure and outcome can
arise from causation, confounding, selection, or measurement. Epidemiology supplies the measures and
the study designs that make causal interpretation more or less plausible; it does not supply causation
from a table alone.

The practical rule: report the measure, name the design, list the biases, and only then discuss what
the association might mean.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Epidemiology | The study of the distribution and determinants of health states | Using it as a synonym for biostatistics | none |
| Population at risk | People who could develop the outcome | Including people who already have it | count |
| Denominator | The population or person-time base | Using the wrong catchment population | count or person-time |
| Numerator | The count of events meeting the case definition | Counting events without a definition | count |
| Prevalence | Existing cases divided by the population at a time | Confusing with incidence | proportion |
| Point prevalence | Prevalence at one instant | Reporting a period as a point | proportion |
| Period prevalence | Prevalence over an interval | Including cases that resolved | proportion |
| Incidence proportion (risk) | New cases divided by the population at risk at the start | Calling it a rate | proportion |
| Incidence rate | New cases divided by person-time | Dividing by the number of people | 1/time |
| Person-time | The sum of time at risk across individuals | Counting people once | person-time |
| Hazard | Instantaneous event rate among those still at risk | Confusing with cumulative risk | 1/time |
| Cumulative incidence | Risk over a defined period, accounting for competing risks | Equating with 1 − survival | proportion |
| Competing risk | An event that prevents the outcome of interest | Treating it as censoring | none |
| Censoring | End of observation without the outcome | Confusing with a competing event | none |
| Attack rate | Incidence proportion in an outbreak | Using it for a rate | proportion |
| Case definition | The criteria that classify a person as a case | Using a code without validation | none |
| Surveillance definition | A case definition designed for monitoring, not diagnosis | Treating it as a clinical diagnosis | none |
| Crude rate | A rate calculated without standardisation | Comparing across populations without adjustment | 1/time |
| Specific rate | A rate within a stratum | Reporting it without the stratum | 1/time |
| Standardised rate | A rate adjusted to a standard population structure | Calling it the true rate | 1/time |
| Direct standardisation | Applying study rates to a standard structure | Using it with unstable stratum rates | 1/time |
| Indirect standardisation | Applying standard rates to the study structure | Using it to compare studies with different structures | ratio |
| SMR | Observed deaths divided by expected deaths | Interpreting it as a rate | dimensionless |
| SIR | Observed infections divided by expected infections | Confusing with SMR | dimensionless |
| Standard population | The population whose structure is used for adjustment | Choosing it after seeing results | count |
| Confounding | A third variable associated with exposure and outcome | Calling any covariate a confounder | none |
| Effect modification | A difference in effect across strata | Confusing with confounding | none |
| Selection bias | A non-representative sample or follow-up | Calling it confounding | none |
| Information bias | Measurement error differing by exposure or outcome | Treating it as random noise | none |
| Recall bias | Differential recall of exposure by outcome | Assuming it is random | none |
| Healthy-user bias | Healthier people use preventive services | Treating use as an effect | none |
| Lead-time bias | Earlier diagnosis lengthens survival without delaying death | Crediting screening with survival gains | none |
| Length-time bias | Screening detects slower-growing disease | Overestimating benefit | none |
| Immortal time bias | Follow-up time during which the outcome cannot occur | Crediting treatment with survival | none |
| Survivorship bias | Analysing only those who survived to be sampled | Generalising from survivors | none |
| Ecological fallacy | Inferring individual effects from group data | Using group rates for individual risk | none |
| Risk ratio (RR) | Ratio of risks | Calling an OR an RR | dimensionless |
| Odds ratio (OR) | Ratio of odds | Interpreting it as a risk ratio | dimensionless |
| Risk difference (RD) | Absolute difference in risk | Reporting only the RR | proportion |
| Attributable risk | Risk in exposed minus risk in unexposed | Confusing with attributable fraction | proportion |
| Attributable fraction (exposed) | (RR − 1)/RR | Using it without the exposure prevalence | proportion |
| Population attributable fraction (PAF) | Proportion of cases attributable to exposure | Confusing with the exposed attributable fraction | proportion |
| NNT | 1/risk difference | Applying it across baselines | count |
| Standardised infection ratio (SIR) | Observed infections divided by predicted | Confusing with the standardised incidence ratio | dimensionless |
| Age standardisation | Adjustment for age structure | Assuming it adjusts for everything | none |
| Case-mix adjustment | Adjustment for patient severity and comorbidity | Assuming it is complete | none |
| Registry | A systematic collection of cases for a defined population | Treating it as complete | none |
| Notifiable disease | A disease required to be reported | Assuming reporting is complete | none |
| Under-reporting | Cases missing from surveillance | Treating counts as complete | proportion |
| Capture-recapture | Estimating the unobserved population from overlapping sources | Assuming source independence | count |

Two distinctions deserve emphasis.

**Surveillance definition versus clinical diagnosis.** A central line-associated bloodstream infection
(CLABSI) is a surveillance definition: it is designed to be applied consistently across hospitals for
comparison, not to diagnose an individual patient. A patient can meet the CLABSI definition without a
clinical bloodstream infection, and vice versa. Treating a surveillance rate as a clinical quality
measure of diagnosis is a category error, and it is the basis of much of the debate about hospital
comparison.

**Attributable fraction in the exposed versus the population.** The attributable fraction in the
exposed is (RR − 1)/RR, the proportion of cases among the exposed that would not have occurred without
the exposure. The population attributable fraction includes the exposure prevalence:
PAF = Pe(RR − 1)/(1 + Pe(RR − 1)). The two differ substantially when the exposure is uncommon, and
confusing them overstates the population impact of a rare exposure.

## 5. The mechanics, step by step

The procedure takes an epidemiological question from the population definition to a reported measure.

### Step 1 — Define the population and the eligibility window

**Input.** A question such as "what is the sepsis mortality in our network?"

**Output.** A population definition: who is eligible, in which facilities, over which period.

**Reasoning.** The population is the denominator's foundation. A network-wide rate and a single-hospital
rate are different quantities. An open cohort admits new patients over time; a closed cohort follows a
fixed group. The choice determines the measure and the analysis.

**Branch point.** If the population cannot be defined consistently across sites, the comparison is not
valid. Restrict to sites with a common definition or report site-specific rates.

### Step 2 — Define the case

**Input.** The outcome concept.

**Output.** A case definition with source, criteria, and validation status.

**Reasoning.** A case can be defined by a laboratory result, a diagnosis code, a surveillance algorithm,
or a chart review. Each has a different sensitivity and positive predictive value. The definition must
be stated, and its validity in the population must be known or estimated.

**Branch point.** If the case definition has not been validated, the measure is a statement about the
definition, not about the disease. Label it accordingly.

### Step 3 — Choose the measure

**Input.** The question and the data structure.

**Output.** Prevalence, incidence proportion, or incidence rate.

**Reasoning.** Prevalence answers "how much disease is there"; incidence answers "how fast are new cases
arising." If follow-up varies across individuals, a rate is appropriate; if the cohort is closed and
follow-up is fixed, a risk is appropriate. The measure must match the question, not the data
availability.

**Branch point.** If the data cannot support person-time (no follow-up dates), a risk or a prevalence
is the only defensible measure. State the limitation.

### Step 4 — Define time zero and follow-up

**Input.** The measure and the cohort.

**Output.** A time origin and a follow-up rule for each person.

**Reasoning.** Time zero is the point at which a person becomes eligible to experience the outcome.
Getting it wrong produces immortal time bias: if treatment is defined after time zero and follow-up
starts before it, the period before treatment is credited to the treated group even though the outcome
could not have occurred then.

**Branch point.** If time zero differs across groups, the comparison is biased. Rebuild the cohort so
that time zero has the same meaning for everyone.

### Step 5 — Count person-time

**Input.** Individual follow-up intervals.

**Output.** Total person-time, with the time unit.

**Reasoning.** Person-time is the sum over individuals of the time at risk. A patient followed for
10 days contributes 10 person-days; a patient followed for 2 days contributes 2. The sum is the
denominator of an incidence rate. Using the number of people instead of person-time is the most common
denominator error in clinical analytics.

**Branch point.** If follow-up is interrupted (transfer, discharge, death), the person-time must stop
or be censored at that point. Do not count time after the outcome.

### Step 6 — Handle competing risks

**Input.** The outcome and the competing events.

**Output.** A decision about how to treat competing events.

**Reasoning.** Death prevents a readmission; discharge prevents a hospital-acquired infection. Treating
a competing event as censoring assumes that the person could still experience the outcome later, which
is false. Cumulative incidence functions, subdistribution hazards, or a composite outcome are the
alternatives.

**Branch point.** If the competing event is common, the naive Kaplan-Meier estimate overstates the
cumulative incidence of the outcome. Use a competing-risk method.

### Step 7 — Compute the crude measure

**Input.** Numerator and denominator.

**Output.** A crude rate or proportion, with its unit and time window.

**Reasoning.** The crude measure is the starting point, not the answer. It is what the population
experienced, unadjusted. It is useful for planning and for describing burden, and it is misleading for
comparison when populations differ.

**Branch point.** If the crude measure is being used to compare groups, go to Step 8 before drawing a
conclusion.

### Step 8 — Examine the strata

**Input.** The crude measure and candidate stratifiers (age, sex, severity, site).

**Output.** Stratum-specific rates and a judgement about heterogeneity.

**Reasoning.** If the stratum-specific rates differ substantially across strata, the crude comparison
is confounded by the distribution of the strata. The classic example is age: a younger population has
lower crude mortality even with identical age-specific rates.

**Branch point.** If the stratum-specific rates are similar across groups but the crude rates differ,
the difference is compositional. If the stratum-specific rates differ in the same direction as the
crude rates, the association is not explained by the stratifier.

### Step 9 — Choose a standard population

**Input.** The strata and the purpose.

**Output.** A standard population structure.

**Reasoning.** The standard can be the study population, a national population, a WHO standard, or a
registry standard. The choice changes the standardised rate but not the comparison if the same standard
is used for both groups. The standard must be stated because a standardised rate is interpretable only
relative to it.

**Branch point.** If two reports use different standards, their standardised rates are not comparable.
Recompute on a common standard before comparing.

### Step 10 — Direct standardisation

**Input.** Study stratum-specific rates and the standard structure.

**Output.** A directly standardised rate.

**Reasoning.** Multiply each stratum rate by the standard population's proportion in that stratum and
sum. The result is the rate the study population would have if it had the standard structure. Direct
standardisation requires stable stratum-specific rates, which requires adequate numbers in each
stratum.

**Branch point.** If a stratum has very few events, the stratum-specific rate is unstable and the
directly standardised rate is unstable. Use indirect standardisation or a model-based approach.

### Step 11 — Indirect standardisation and the SMR

**Input.** Study observed events and the standard stratum-specific rates.

**Output.** Expected events and the SMR, with a confidence interval.

**Reasoning.** Expected events = Σ(standard rate × study population in stratum). SMR = observed/expected.
An SMR above 1 means more events than expected given the standard rates. The confidence interval is
usually based on the Poisson distribution, and it is wide when the observed count is small.

**Branch point.** If the SMR is being compared across studies with different population structures, the
comparison can mislead because the indirect weights differ. Use direct standardisation for
between-study comparisons when the data allow.

### Step 12 — Compare the direct and indirect results

**Input.** Both standardised measures.

**Output.** A consistency check.

**Reasoning.** The two methods answer different questions and can disagree when the rate ratios vary
across strata. A large disagreement signals effect modification by the stratifier and should be
reported, not hidden by choosing the more favourable method.

**Branch point.** If the methods disagree qualitatively, the summary measure is not robust. Report the
stratum-specific rates instead of a single standardised number.

### Step 13 — Compute measures of association

**Input.** The 2×2 table or the rates.

**Output.** RR, OR, RD, and their intervals.

**Reasoning.** The RR is appropriate for cohort and trial designs; the OR is appropriate for
case-control designs and for logistic models. The RD is the absolute scale and is usually the most
decision-relevant. Report all three where possible, with the baseline risk.

**Branch point.** If the outcome is common, do not present the OR as an approximation of the RR.
Convert it using the baseline risk, or report the RD.

### Step 14 — Match the measure to the design

**Input.** The study design.

**Output.** The valid measure.

**Reasoning.** A case-control study samples on outcome, so it cannot estimate incidence or risk
directly; it estimates the OR (and, with a known sampling fraction, the RR). A cross-sectional study
estimates prevalence and prevalence ratios, not incidence. A cohort estimates risk and rate. Using a
measure that the design cannot support is a structural error.

**Branch point.** If the design is cross-sectional and the question is about incidence, the data cannot
answer it. State the limitation.

### Step 15 — Compute attributable measures

**Input.** RR and the exposure prevalence.

**Output.** Attributable fraction in the exposed and the population attributable fraction.

**Reasoning.** AF_exposed = (RR − 1)/RR. PAF = Pe(RR − 1)/(1 + Pe(RR − 1)). The PAF depends on the
prevalence of the exposure and assumes that the association is causal; it is a policy quantity, not a
description of the data.

**Branch point.** If the association is not plausibly causal, do not compute a PAF. A PAF computed on a
confounded association is a number without a meaning.

### Step 16 — Attach confidence intervals

**Input.** The measure and the counts.

**Output.** An interval on the appropriate scale.

**Reasoning.** Rates based on small counts have wide Poisson intervals. Proportions near 0 or 1 have
asymmetric intervals. Ratio measures are usually modelled on the log scale, where the interval is
symmetric. The interval must reflect the design, including clustering if present.

**Branch point.** If the count is very small (below about 20), use an exact or Byar approximation rather
than a normal approximation.

### Step 17 — Assess confounding

**Input.** The association and candidate confounders.

**Output.** A judgement about whether the association is explained by a third variable.

**Reasoning.** A confounder must be associated with the exposure, associated with the outcome
conditional on the exposure, and not lie on the causal path. Stratification, regression, matching, or
weighting can adjust for measured confounders. Unmeasured confounding cannot be removed by analysis,
only bounded or discussed.

**Branch point.** If a strong unmeasured confounder is plausible, the association is not causal. Use a
negative control or an E-value sensitivity analysis.

### Step 18 — Assess effect modification

**Input.** Stratum-specific effects.

**Output.** A statement about heterogeneity.

**Reasoning.** Effect modification is a difference in the effect across strata. It is a finding, not a
nuisance. It should be tested with an interaction term, not by comparing within-stratum p-values, and
it should be pre-specified or labelled exploratory.

**Branch point.** If the effect differs qualitatively across strata, a single summary effect is
misleading. Report the strata.

### Step 19 — Assess the major biases

**Input.** The design and the data sources.

**Output.** A bias table with the direction of the likely effect.

**Reasoning.** Selection bias, information bias, lead-time bias, length-time bias, immortal time bias,
and healthy-user bias each have a characteristic direction. Naming the direction is more useful than
naming the bias, because it tells the reader whether the true effect is likely larger or smaller than
the estimate.

**Branch point.** If a bias could reverse the direction of the association, the conclusion is not
robust. Report the range consistent with the bias.

### Step 20 — Assess temporality

**Input.** The exposure and outcome timing.

**Output.** A statement about whether the exposure preceded the outcome.

**Reasoning.** Temporality is the one criterion that distinguishes a causal hypothesis from a
correlation. If the exposure is measured after the outcome, or if the timing is unknown, the
association cannot be causal in the stated direction. This is the most common defect in claims-based
analyses.

**Branch point.** If temporality cannot be established, the analysis is descriptive. Do not use causal
language.

### Step 21 — Check for surveillance artefacts

**Input.** The surveillance data.

**Output.** A list of artefacts and their likely effect.

**Reasoning.** Surveillance definitions change, reporting requirements change, testing practices
change, and coding practices change. A rise in cases can be a rise in testing. A fall in infections
can be a rise in the denominator. These artefacts are not random and do not average out.

**Branch point.** If a policy or definition changed mid-period, do not compare across the change
without adjustment or a clear annotation.

### Step 22 — Validate the case definition in a sample

**Input.** A sample of records.

**Output.** Sensitivity, specificity, and PPV of the definition.

**Reasoning.** A case definition's performance determines the measure's validity. A definition with a
PPV of 60% means that 40% of the counted cases are not cases, and the rate is overstated by a factor
that depends on prevalence. Validation against chart review is the standard method.

**Branch point.** If the PPV is low and differs across sites, the comparison is invalid. Report the
validation and adjust or restrict the claim.

### Step 23 — Report with a recognised guideline

**Input.** The analysis.

**Output.** A report that follows STROBE, RECORD, or the relevant surveillance protocol.

**Reasoning.** Reporting guidelines require the case definition, the denominator, the time window, the
handling of missing data, and the confounders considered. They make the analysis reproducible and
comparable.

**Branch point.** If the report cannot state the denominator and the case definition, it is not
complete.

### Step 24 — Run sensitivity analyses on the fragile assumptions

**Input.** The primary analysis.

**Output.** A table of alternative estimates.

**Reasoning.** The fragile assumptions are usually the case definition, the denominator, the
standardisation, and the handling of missing data. Varying them shows how much the conclusion depends
on choices that cannot be tested.

**Branch point.** If the conclusion changes across plausible assumptions, report the range and the
conditions.

### Step 25 — Communicate absolute and relative measures together

**Input.** The results.

**Output.** A report that gives the baseline risk, the absolute difference, and the relative measure.

**Reasoning.** A relative measure without the baseline is uninterpretable, and an absolute measure
without the relative is incomplete. The pair is the minimum for a decision.

**Branch point.** If the audience is a patient or a policymaker, lead with the absolute measure. If the
audience is a researcher, give both.

### Step 26 — Write the assumptions into the ledger

**Input.** All of the above.

**Output.** Ledger rows for the case definition, the denominator, the standardisation, the confounders,
and the biases.

**Reasoning.** Each assumption becomes a testable statement with an owner and a test. The ledger is
what makes the analysis auditable after the fact.

**Branch point.** If an assumption cannot be tested, mark it open and state what would change the
decision.

### 5.1 The branch map

```text
Epidemiological question
  |
  +-- existing or new cases? --------> prevalence vs incidence
  |
  +-- variable follow-up? -----------> incidence rate with person-time
  |
  +-- competing event common? -------> cumulative incidence function
  |
  +-- populations differ in age? ----> direct standardisation
  |
  +-- small stratum counts? ---------> indirect standardisation / SMR
  |
  +-- case-control design? ----------> odds ratio
  |
  +-- common outcome? ---------------> risk difference, not OR as RR
  |
  +-- exposure after outcome? -------> descriptive only
  |
  +-- policy changed mid-period? ----> do not compare across the change
```

### 5.2 The four checks that catch most epidemiological errors

1. **State the denominator.** A count is not a rate, and a rate with the wrong denominator is worse
   than a count.
2. **State the case definition and its validation.** A code is not a diagnosis.
3. **Stratify before comparing.** A crude comparison across populations with different structures is a
   composition statement.
4. **Check the timing.** If the exposure follows the outcome, there is no causal question to answer.

## 6. Worked example

Two hospitals in the same network report 30-day mortality for patients admitted with infection. Each
admitted 1,000 patients. Hospital A reports 68 deaths and hospital B reports 89, so the crude mortality
is 6.8% and 8.9%. The quality committee concludes that A is safer.

The case stratifies by severity, standardises directly and indirectly, and shows that the conclusion
reverses. It then computes a central line infection rate with a Poisson interval and a set of
association measures from a 2×2 table.

### 6.1 Crude mortality

```text
Hospital A: 68/1,000 = 0.0680 = 6.80%
Hospital B: 89/1,000 = 0.0890 = 8.90%

Crude difference = 6.80 - 8.90 = -2.10 percentage points
Crude ratio = 6.80/8.90 = 0.764
```

On the crude comparison, hospital A has 24% lower mortality. The number is arithmetically correct and
analytically incomplete.

### 6.2 Stratify by severity

Severity is defined as low or high by a pre-specified clinical criterion.

```text
Hospital A
  Low:  800 patients, 24 deaths -> 24/800  = 3.00%
  High: 200 patients, 44 deaths -> 44/200  = 22.00%
  Total: 68/1,000 = 6.80%

Hospital B
  Low:  500 patients, 30 deaths -> 30/500  = 6.00%
  High: 500 patients, 59 deaths -> 59/500  = 11.80%
  Total: 89/1,000 = 8.90%
```

Hospital A has a lower mortality in the low-severity stratum (3.00% versus 6.00%) and a higher
mortality in the high-severity stratum (22.00% versus 11.80%). The crude comparison is a mixture of
these two effects weighted by case mix.

Hospital A's population is 80% low severity; hospital B's is 50% low severity. The high-severity
patients, who have much higher mortality, are concentrated in hospital B, which raises B's crude rate.

### 6.3 Direct standardisation

Standardise both hospitals to a common structure of 50% low severity and 50% high severity.

```text
Hospital A
  = 0.50 x 3.00% + 0.50 x 22.00%
  = 1.50% + 11.00%
  = 12.50%

Hospital B
  = 0.50 x 6.00% + 0.50 x 11.80%
  = 3.00% + 5.90%
  = 8.90%
```

After direct standardisation, hospital A has a mortality of 12.50% and hospital B of 8.90%. The
conclusion reverses: A is worse, not better. The crude comparison favoured A only because A treated a
younger, less severe population.

The standardised rates are not predictions for either hospital. They are the rates each hospital would
have if its case mix matched the standard. A different standard would give different numbers but the
same ordering, because A's rate is higher in one stratum and lower in the other; the ordering depends
on the weight given to each stratum, which is why the standard must be stated.

### 6.4 Indirect standardisation and the SMR

Use the network's standard stratum-specific mortality as the reference: 4.0% for low severity and
15.0% for high severity.

```text
Hospital A
  Expected = 800 x 0.040 + 200 x 0.150
           = 32.0 + 30.0
           = 62.0

  SMR = observed/expected = 68/62.0 = 1.0968

Hospital B
  Expected = 500 x 0.040 + 500 x 0.150
           = 20.0 + 75.0
           = 95.0

  SMR = 89/95.0 = 0.9368
```

Hospital A has about 10% more deaths than expected given its case mix; hospital B has about 6% fewer.
The direction agrees with the direct standardisation.

The 95% Poisson interval for each SMR, using the Byar approximation:

```text
For A: O = 68, E = 62.0
  lower = 68 x (1 - 1/(9x68) - 1.96/(3xsqrt(68)))^3 / 62.0
        = 68 x (1 - 0.001634 - 0.079230)^3 / 62.0
        = 68 x 0.919136^3 / 62.0
        = 68 x 0.7765 / 62.0
        = 52.80 / 62.0
        = 0.852

  upper = 69 x (1 - 1/(9x69) + 1.96/(3xsqrt(69)))^3 / 62.0
        = 69 x (1 - 0.001610 + 0.078655)^3 / 62.0
        = 69 x 1.077045^3 / 62.0
        = 69 x 1.2491 / 62.0
        = 86.19 / 62.0
        = 1.390

For B: O = 89, E = 95.0
  lower = 0.752
  upper = 1.153
```

Hospital A: SMR 1.10 (95% CI 0.85 to 1.39). Hospital B: SMR 0.94 (95% CI 0.75 to 1.15).

Both intervals include 1. The data are compatible with A having up to 39% more deaths than expected or
15% fewer, and with B having up to 15% more or 25% fewer. The point estimates suggest a difference; the
intervals do not establish one. A committee that acts on the point estimates alone is over-reading the
data.

### 6.5 Why the direct and indirect results differ in magnitude

The direct standardisation gives A 12.50% and B 8.90%, a difference of 3.6 percentage points. The SMR
gives A 1.10 and B 0.94, a ratio of 1.17. The two are not on the same scale: the direct method produces
a rate, and the indirect method produces a ratio. The apparent difference in magnitude is a difference
in measure, not a contradiction.

The ordering agrees because both methods weight the strata in a way that reflects A's higher
high-severity mortality. If the standard population were heavily weighted toward low severity, the
direct standardised rates would converge, and if it were heavily weighted toward high severity, the gap
would widen. The SMR's implicit weighting is the study populations themselves, which is why indirect
standardisation is preferred for small populations and direct standardisation for comparisons across
populations.

### 6.6 A central line infection rate

Hospital A reports 12 central line-associated bloodstream infections over 4,800 central line-days.
Hospital B reports 18 over 12,000 central line-days.

```text
Hospital A
  Rate = 12/4,800 x 1,000 = 2.50 per 1,000 line-days

Hospital B
  Rate = 18/12,000 x 1,000 = 1.50 per 1,000 line-days

Rate ratio = 2.50/1.50 = 1.667
```

The count is higher at B (18 versus 12), but the rate is higher at A (2.50 versus 1.50). The unit's
size and line utilization determine the denominator, and comparing counts across units of different
size is the denominator error in its purest form.

The 95% Poisson interval for A, using the Byar approximation with E = 4.8 thousand line-days:

```text
  lower = 12 x (1 - 1/108 - 1.96/(3 x sqrt(12)))^3 / 4.8
        = 12 x (1 - 0.009259 - 0.188612)^3 / 4.8
        = 12 x 0.802129^3 / 4.8
        = 12 x 0.5161 / 4.8
        = 6.193/4.8
        = 1.29 per 1,000 line-days

  upper = 13 x (1 - 1/117 + 1.96/(3 x sqrt(13)))^3 / 4.8
        = 13 x (1 - 0.008547 + 0.181172)^3 / 4.8
        = 13 x 1.172625^3 / 4.8
        = 13 x 1.6124 / 4.8
        = 20.96/4.8
        = 4.37 per 1,000 line-days
```

Hospital A: 2.50 per 1,000 line-days (95% CI 1.29 to 4.37). The interval is wide because the count is
small. A rate based on 12 events cannot distinguish a true rate of 1.3 from a true rate of 4.4, which
is why small-unit comparisons require pooling over time or a model-based approach.

### 6.7 Association measures from a 2×2 table

A cohort study follows 400 exposed and 600 unexposed patients. Forty exposed and 30 unexposed develop
the outcome.

```text
                     Outcome +   Outcome -   Total
  Exposed                40         360        400
  Unexposed              30         570        600

Risk exposed   = 40/400  = 0.1000
Risk unexposed = 30/600  = 0.0500

RR = 0.1000/0.0500 = 2.000
OR = (40/360)/(30/570) = 0.11111/0.052632 = 2.1111
RD = 0.1000 - 0.0500 = 0.0500 = 5.0 percentage points
NNT = 1/0.0500 = 20.0
```

The OR is 2.11 and the RR is 2.00. The OR is further from 1 because the outcome is common (10% in the
exposed). Reporting the OR as if it were the RR would overstate the relative effect by 5.6%.

The population attributable fraction, with exposure prevalence 0.40:

```text
PAF = Pe(RR - 1)/(1 + Pe(RR - 1))
    = 0.40 x 1.00/(1 + 0.40 x 1.00)
    = 0.40/1.40
    = 0.2857 = 28.6%
```

If the association were causal, about 29% of cases in this population would be attributable to the
exposure. The PAF is a policy quantity, and it is only meaningful under a causal assumption that this
observational comparison cannot establish.

### 6.8 What the example demonstrates

| Step | Result | Lesson |
|---|---|---|
| Crude mortality | A 6.80%, B 8.90% | A crude comparison favours A |
| Stratified | A 3.00%/22.00%, B 6.00%/11.80% | A is worse in the high-severity stratum |
| Direct standardisation | A 12.50%, B 8.90% | The conclusion reverses |
| SMR | A 1.10, B 0.94 | The direction agrees, the scale differs |
| SMR interval | A 0.85–1.39 | The difference is not established |
| CLABSI rate | A 2.50, B 1.50 per 1,000 line-days | Counts across units mislead |
| Poisson interval | A 1.29–4.37 | Small counts give wide intervals |
| RR and OR | 2.00 and 2.11 | The OR overstates a common outcome |
| PAF | 28.6% | A policy quantity under a causal assumption |

## 7. Code, query, or calculation

The script reproduces every number in section 6, including the Byar Poisson intervals.

```python
"""Topic 07 verification: rates, direct and indirect standardisation, association."""
from math import sqrt

def byar_ci(o, e, z=1.96):
    lo = o * (1 - 1/(9*o) - z/(3*sqrt(o)))**3 / e
    hi = (o + 1) * (1 - 1/(9*(o+1)) + z/(3*sqrt(o+1)))**3 / e
    return lo, hi

print("7.1 Crude mortality")
A = {"low": (800, 24), "high": (200, 44)}
B = {"low": (500, 30), "high": (500, 59)}
for name, d in (("A", A), ("B", B)):
    n = sum(v[0] for v in d.values())
    deaths = sum(v[1] for v in d.values())
    print(f"  hospital {name}: {deaths}/{n} = {deaths/n*100:.2f}%")

print("\n7.2 Stratum-specific rates")
for name, d in (("A", A), ("B", B)):
    for stratum, (n, deaths) in d.items():
        print(f"  {name} {stratum:4s}: {deaths}/{n} = {deaths/n*100:.2f}%")

print("\n7.3 Direct standardisation to a 50/50 standard")
std = {"low": 0.5, "high": 0.5}
for name, d in (("A", A), ("B", B)):
    s = sum(std[k] * (d[k][1]/d[k][0]) for k in std)
    print(f"  hospital {name}: {s*100:.2f}%")

print("\n7.4 Indirect standardisation (SMR)")
ref = {"low": 0.04, "high": 0.15}
for name, d in (("A", A), ("B", B)):
    obs = sum(v[1] for v in d.values())
    exp = sum(d[k][0] * ref[k] for k in d)
    smr = obs/exp
    lo, hi = byar_ci(obs, exp)
    print(f"  {name}: observed {obs}, expected {exp:.1f}, SMR {smr:.4f} "
          f"({lo:.3f} to {hi:.3f})")

print("\n7.5 CLABSI rates")
for label, cases, days in (("A", 12, 4800), ("B", 18, 12000)):
    rate = cases/days*1000
    lo, hi = byar_ci(cases, days/1000)
    print(f"  hospital {label}: {cases}/{days} = {rate:.2f} per 1000 line-days "
          f"({lo:.2f} to {hi:.2f})")

print("\n7.6 Association measures")
a, b, c, d_ = 40, 360, 30, 570
r1 = a/(a+b)
r0 = c/(c+d_)
rr = r1/r0
orr = (a/b)/(c/d_)
arr = r1-r0
nnt = 1/arr
pe = (a+b)/(a+b+c+d_)
paf = pe*(rr-1)/(1+pe*(rr-1))
print(f"  risk exposed = {r1:.4f}, unexposed = {r0:.4f}")
print(f"  RR = {rr:.4f}")
print(f"  OR = {orr:.4f}")
print(f"  ARR = {arr:.4f}, NNT = {nnt:.1f}")
print(f"  exposure prevalence = {pe:.4f}")
print(f"  PAF = {paf:.4f}")

print("\n7.7 Prevalence and incidence")
print(f"  prevalence = 480/12000 = {480/12000*100:.2f}%")
print(f"  incidence rate = 12/4800 x 1000 = {12/4800*1000:.2f} per 1000 line-days")
```

**Expected output.**

```text
7.1 Crude mortality
  hospital A: 68/1000 = 6.80%
  hospital B: 89/1000 = 8.90%

7.2 Stratum-specific rates
  A low : 24/800 = 3.00%
  A high: 44/200 = 22.00%
  B low : 30/500 = 6.00%
  B high: 59/500 = 11.80%

7.3 Direct standardisation to a 50/50 standard
  hospital A: 12.50%
  hospital B: 8.90%

7.4 Indirect standardisation (SMR)
  A: observed 68, expected 62.0, SMR 1.0968 (0.852 to 1.390)
  B: observed 89, expected 95.0, SMR 0.9368 (0.752 to 1.153)

7.5 CLABSI rates
  hospital A: 12/4800 = 2.50 per 1000 line-days (1.29 to 4.37)
  hospital B: 18/12000 = 1.50 per 1000 line-days (0.89 to 2.37)

7.6 Association measures
  risk exposed = 0.1000, unexposed = 0.0500
  RR = 2.0000
  OR = 2.1111
  ARR = 0.0500, NNT = 20.0
  exposure prevalence = 0.4000
  PAF = 0.2857

7.7 Prevalence and incidence
  prevalence = 480/12000 = 4.00%
  incidence rate = 12/4800 x 1000 = 2.50 per 1000 line-days
```

**Assumptions stated by the script.**

1. Severity is measured without error and is the only relevant stratifier. Residual confounding by age,
   comorbidity, and transfer status remains.
2. The standard population is a 50/50 mix. A different standard would change the standardised rates
   and, if the stratum-specific ordering is inconsistent, could change the ordering.
3. The reference rates for the SMR are the network's rates, not a national standard. An SMR is
   interpretable only against its reference.
4. The Poisson interval assumes that the counts are Poisson and independent. Clustering within units or
   within patients would widen the interval.
5. The 2×2 table is from a cohort with complete follow-up. A case-control design would not permit the
   risk calculations.
6. The PAF assumes a causal effect of the exposure. The study design does not establish causation.

## 8. Data, standards, and regulatory context

### 8.1 Case definitions and classifications

| Standard | What it defines | Binding status |
|---|---|---|
| ICD-11 | Disease and cause-of-death classification | Binding for mortality and morbidity coding in adopting countries |
| ICD-O-3 | Tumour morphology and topography | Required for cancer registration |
| SNOMED CT | Clinical concepts used in EHR data | Required in several national programmes |
| LOINC | Observation codes for laboratory and vital signs | Required by many interoperability profiles |
| CDC/NHSN surveillance definitions | Healthcare-associated infection case definitions | Required for CMS quality reporting in the US |
| WHO case definitions | Outbreak and notifiable disease definitions | Binding for International Health Regulations reporting |
| AJCC TNM | Cancer stage | Required for cancer registration |
| STROBE | Reporting of observational studies | Journal requirement |
| RECORD | Reporting of observational studies using routinely collected data | Journal requirement |
| GATHER | Reporting of global health estimates | Journal requirement |

**ICD-11** is the eleventh revision of the International Classification of Diseases, adopted by the
World Health Assembly in 2019 and in use in adopting countries from 2022 onward. It provides a
classification of diseases, not a case definition: a code assigned in a record is an assertion by a
coder, and its validity as a case definition depends on coding practice.

**CDC/NHSN** publishes the surveillance definitions used for healthcare-associated infections in the
United States. The CLABSI definition is a laboratory-confirmed bloodstream infection with an eligible
central line in place on the date of the event or the day before, with the line in place for more than
two consecutive calendar days. The rate is expressed per 1,000 central line-days. Since 2011, US
hospitals have reported CLABSI data to NHSN for CMS programmes, and the current comparison metric is
the standardised infection ratio, observed divided by predicted.

The CLABSI case illustrates the difference between a surveillance definition and a clinical diagnosis.
The definition is designed for consistent comparison across hospitals. It is not a clinical diagnosis,
and a hospital's rate is a statement about the definition's application as much as about infection
prevention.

### 8.2 Reporting guidelines

**STROBE** requires observational studies to report the study design, the setting, the eligibility
criteria, the case definition, the denominator, the handling of missing data, the confounders
considered, and the effect estimates with intervals. **RECORD** extends STROBE for studies using
routinely collected health data, requiring the codes and algorithms used to define the population,
exposure, and outcome.

The guidelines are journal requirements, not law, but a study that omits the case definition and the
denominator will not survive peer review in a serious journal. The requirement is the formal
expression of the checks in section 5.

### 8.3 Regulatory and payment context

**CMS quality programmes** use standardised ratios (observed/predicted) rather than crude rates for
hospital comparison, precisely because crude rates are confounded by case mix. The Hospital
Readmissions Reduction Program, the Hospital-Acquired Condition Reduction Program, and the Value-Based
Purchasing Program all use risk-adjusted measures.

**Marketing and reimbursement claims** for diagnostics and devices often rest on epidemiological
quantities: prevalence, incidence, positive predictive value, and number needed to screen. A claim
that a test "detects disease earlier" requires evidence that the earlier detection changes outcomes,
not only that it moves the diagnosis earlier in time.

**Public health surveillance** under the International Health Regulations requires notification of
specified diseases and events. The case definitions are set by WHO, and the reporting obligation is
binding for signatory states.

### 8.4 What the standards do not constrain

- They do not validate a case definition. A code can be applied consistently and still be wrong.
- They do not choose the standard population. That is an analyst decision that must be stated.
- They do not resolve confounding. A risk-adjusted measure adjusts for the variables in the model, not
  for unmeasured ones.
- They do not make a surveillance rate a clinical quality measure. The definition serves comparison,
  not diagnosis.
- They do not establish causation. No reporting guideline can.

> **Note:** The regulatory content is educational. Surveillance and reimbursement calculations require
> review against the current programme specifications.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Hospitals** report healthcare-associated infection rates to NHSN and receive standardised infection
ratios relative to a national baseline. Internal quality teams stratify by unit type and device
utilization because crude hospital-wide rates mix very different populations.

**Public health agencies** publish age-standardised rates so that comparisons across regions and years
are not driven by demographic change. The WHO World Standard Population and national standard
populations are used for this purpose, and the choice of standard is stated in the methods.

**Insurers and health services researchers** use claims data for population-level analyses. Claims have
the advantage of scale and the disadvantage of coding-based case definitions, incomplete capture of
severity, and churn in enrolment. RECORD-compliant studies report the codes and algorithms used and
validate a sample against records.

**Screening programmes** are evaluated with mortality endpoints rather than survival from diagnosis,
because survival from diagnosis is susceptible to lead-time and length-time bias. The distinction is
the reason screening trials are large and long.

### 9.2 Case evidence

**The standardised infection ratio.** CDC's NHSN moved from publishing stratified rates to a
standardised infection ratio, observed divided by predicted, with predictions from a risk model and a
specified baseline period. The change is a documented case of a crude measure being replaced by a
risk-adjusted one because crude comparisons across hospitals were confounded by case mix and device
use.

**Lead-time bias in screening.** A screening test that advances diagnosis by two years without delaying
death increases five-year survival from diagnosis by two years, even if the date of death is unchanged.
The phenomenon is standard in cancer screening evaluation and is the reason trials use
disease-specific mortality as the endpoint. The arithmetic is simple: survival is measured from
diagnosis, and diagnosis moved.

**Healthy-user bias in observational prevention studies.** Observational studies of preventive
services consistently find larger effects than randomised trials, in part because people who use
preventive services are healthier, wealthier, and more adherent. The bias is documented across
screening, vaccination, and supplementation studies, and it is a selection effect, not a failure of
the statistical method.

**Immortal time bias in treatment comparisons.** A cohort study that classifies patients as treated if
they received a drug at any point during follow-up, and starts follow-up at diagnosis, credits the
treated group with the period before treatment, during which death could not have occurred in the
treated group by definition. The bias produces an apparent survival benefit for almost any treatment
and is corrected by aligning time zero with treatment initiation or by a landmark analysis.

> **Note:** The NHSN metric change and the bias mechanisms are documented in the cited standards and
> methods literature. The specific institutional consequences described are inference, not audited case
> reports.

### 9.3 The practical pattern

Teams that avoid epidemiological errors do four things:

1. They write the case definition and the denominator before computing any rate.
2. They stratify before comparing populations.
3. They report absolute and relative measures with the baseline.
4. They name the bias and its likely direction rather than listing bias categories.

Each practice is a control on a named failure mode, and each is checkable by a reviewer without access
to the raw data.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Count reported without a denominator | "Twelve infections this year" | No population or exposure base | Size changes look like rate changes | Report the denominator and the time window |
| 2 | Persons used instead of person-time | Variable follow-up, rate computed as cases/people | Unequal exposure time | Rate biased toward the long-followed | Compute person-time; state the unit |
| 3 | Crude comparison across different case mixes | Two hospitals compared on crude mortality | Compositional difference | Reversed or overstated conclusion | Stratify and standardise |
| 4 | Standard population unstated | "Age-standardised rate" with no standard named | The rate is relative to a structure | Rates from different reports are incomparable | Name the standard; recompute on a common one |
| 5 | Direct standardisation on unstable strata | A stratum has 2 events | Small-number instability | Unstable standardised rate | Use indirect standardisation or a model |
| 6 | SMR interpreted as a rate | "SMR 1.2 means 1.2% mortality" | It is a ratio of observed to expected | Magnitude misread | Report observed, expected, and the ratio |
| 7 | Odds ratio reported as a risk ratio | Common outcome and OR far from RR | Odds exceed risk | Effect overstated | Convert or report the RD |
| 8 | Case definition unvalidated | A code-based definition with no PPV | Codes reflect documentation | Rate measures coding, not disease | Validate a sample against chart review |
| 9 | Surveillance definition treated as diagnosis | A CLABSI rate called a clinical infection rate | The definition is for comparison | Misleading quality claim | Label the measure as surveillance |
| 10 | Lead-time bias | Survival from diagnosis improves without mortality change | Diagnosis moved earlier | Ineffective screening looks effective | Use mortality as the endpoint |
| 11 | Length-time bias | Screening finds more slow-growing disease | Slower disease is over-represented | Benefit overstated | Randomised evaluation |
| 12 | Immortal time bias | Treated group classified after follow-up starts | Time before treatment is credited | Almost any treatment looks beneficial | Align time zero; landmark analysis |
| 13 | Healthy-user bias | Preventive service users are healthier | Self-selection | Effect overstated | Randomisation or adjustment for health status |
| 14 | Competing risk ignored | Kaplan-Meier cumulative incidence exceeds the naive proportion | Competing events treated as censoring | Incidence overstated | Cumulative incidence function |
| 15 | Policy change mid-period | A coding or testing change coincides with a rate change | Surveillance artefact | Trend attributed to care | Annotate the change; do not compare across it |
| 16 | Ecological inference | A regional rate used to infer individual risk | Group-level association | Ecological fallacy | Use individual-level data or state the limitation |
| 17 | PAF computed on a non-causal association | A PAF reported without a causal argument | The formula assumes causation | Policy number without meaning | State the causal assumption; do not compute otherwise |

Three rows deserve emphasis.

**Row 3, case mix.** The worked example in section 6 shows a complete reversal from a 2.1-point
advantage to a 3.6-point disadvantage after standardisation. The reversal is not rare; it happens
whenever the exposure groups differ substantially in a strong prognostic factor. Any comparison of
hospitals, regions, or time periods must check the case mix before the headline is written.

**Row 12, immortal time.** The bias is subtle because the data look complete and the analysis looks
standard. The tell is a treatment definition that depends on event-free survival to treatment. If a
patient must survive to receive the drug, the treated group is guaranteed to have survived at least
until treatment, and that guaranteed time is credited to the drug. The fix is to start follow-up at
treatment initiation, or to use a landmark analysis that conditions on being alive at a fixed time.

**Row 15, policy change.** A rate change that coincides with a coding change, a testing change, or a
surveillance definition change is not interpretable as a care change. The tell is a step change rather
than a gradual trend, and it usually appears at a fiscal year boundary. Annotate the change on every
plot and do not compare across it without adjustment.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The population is completely enumerated | The denominator is the population at risk | Under-reporting, out-of-area patients | Capture-recapture; compare with an independent source |
| 2 | The case definition identifies cases consistently | The numerator is comparable over time and place | Coding changes, testing changes | Validation sample; definition audit |
| 3 | Follow-up is complete | Person-time is accurate | Loss to follow-up, transfers | Compare observed with expected follow-up |
| 4 | Time zero is the same for all groups | Comparability of follow-up | Treatment defined after time zero | Timeline audit; landmark analysis |
| 5 | The strata capture the important confounders | Standardisation removes composition effects | Unmeasured severity, comorbidity | Residual confounding analysis |
| 6 | The standard population is appropriate and stated | Standardised rates are comparable | Different standards across reports | Recompute on a common standard |
| 7 | The stratum-specific rates are stable | Direct standardisation is valid | Small numbers | Use indirect standardisation |
| 8 | The reference rates are applicable | SMR interpretation | Different era or population | Compare with a current reference |
| 9 | The outcome is measured without differential error | Association is unbiased | Differential detection by exposure | Blinded assessment; validation substudy |
| 10 | The exposure precedes the outcome | Causal interpretation | Reverse causation | Timing audit |
| 11 | Competing events are handled correctly | Cumulative incidence is valid | Death or discharge | Competing-risk analysis |
| 12 | The association is not explained by selection | Generalisability | Healthy-user, referral, survivor selection | Compare with a randomised or representative sample |
| 13 | The PAF is computed only under a causal assumption | Policy interpretation | Confounded association | State the assumption; sensitivity analysis |
| 14 | The reporting is complete | Reproducibility | Selective outcome reporting | Pre-specification; STROBE |

The limits of this topic:

- It does not cover causal identification methods (instrumental variables, difference-in-differences,
  target trial emulation) in depth. Those belong to a dedicated causal inference module.
- It does not cover survival analysis beyond the concept of person-time and competing risks.
- It does not cover outbreak investigation or molecular epidemiology.
- It does not validate any specific registry or surveillance system.
- It does not make clinical or policy recommendations.

## 12. Exercises

### Exercise 1 — Choose the measure (application)

For each question, state whether prevalence, incidence proportion, or incidence rate is appropriate,
and why: how many adults currently have diabetes in a region; the probability that a patient develops
a surgical site infection within 30 days of operation; the speed at which new infections arise per
catheter-day; the proportion of emergency admissions that are readmitted within 30 days.

**Deliverable.** A four-row table with the measure and the reason.

**Acceptance criteria.** Diabetes: prevalence; surgical site infection: incidence proportion (fixed
30-day window, closed cohort); catheter infections: incidence rate (variable dwell time); readmission:
incidence proportion (fixed window). Each reason names the denominator and the time structure.

**Time.** 20 minutes.

### Exercise 2 — Person-time (calculation)

Three patients are followed after discharge: patient 1 for 10 days, patient 2 for 5 days, patient 3 for
2 days. Six events occur, one in each of the first two patients and four in the third.

Compute the total person-time and the incidence rate per 100 person-days. Then compute the rate if the
denominator were the number of patients, and explain the difference.

**Deliverable.** Two rates and an explanation.

**Acceptance criteria.** Person-time 17 days; rate 35.3 per 100 person-days; patient-denominator rate
2.0 events per patient; the explanation states that the patient denominator weights a 2-day patient
equally with a 10-day patient and therefore overstates exposure per unit time.

**Time.** 25 minutes.

### Exercise 3 — Crude and specific rates (calculation)

A programme reports 60 deaths in 2,000 patients (3.0%). Stratified: 1,600 low-risk patients with 16
deaths, and 400 high-risk patients with 44 deaths. Compute the stratum-specific rates, verify that they
reproduce the crude rate, and state which rate should be reported to a patient considering enrolment.

**Deliverable.** Three rates and a recommendation.

**Acceptance criteria.** Low 1.0%, high 11.0%, crude 3.0%; the weighted sum 0.8 × 1.0 + 0.2 × 11.0 =
3.0% reproduces the crude; the recommendation gives the stratum-specific rate for the patient's risk
group, not the crude.

**Time.** 25 minutes.

### Exercise 4 — Standardise (calculation)

Hospital X has 600 low-risk patients with 12 deaths and 400 high-risk with 60 deaths. Hospital Y has
400 low-risk with 12 deaths and 600 high-risk with 54 deaths. Compute the crude rates, the
stratum-specific rates, and the directly standardised rates to a 50/50 standard. State which hospital
performs better and on what basis.

**Deliverable.** A table of rates and a conclusion.

**Acceptance criteria.** X crude 7.2%, Y crude 6.6%; X low 2.0%, high 15.0%; Y low 3.0%, high 9.0%;
standardised X 8.5%, Y 6.0%; Y performs better after standardisation, but X has a lower low-risk rate;
the conclusion states that the summary depends on the standard and reports both strata.

**Time.** 35 minutes.

### Exercise 5 — Association measures (calculation)

A cohort has 120 events among 1,000 exposed and 80 among 1,000 unexposed. Compute the risks, RR, OR,
RD, NNT, and the PAF assuming the exposure prevalence is 50%.

**Deliverable.** Six values with units.

**Acceptance criteria.** Risks 12% and 8%; RR 1.50; OR 1.568; RD 4.0 percentage points; NNT 25; PAF
20.0%; the answer notes that the OR exceeds the RR because the outcome is common.

**Time.** 30 minutes.

### Exercise 6 — Name the bias (critique)

For each scenario, name the bias and state the direction of its effect on the estimated association:
(a) a study of statins and mortality that defines exposure as "any statin during follow-up"; (b) a
screening study reporting five-year survival from diagnosis; (c) a survey of wearable users finding
that they have better health outcomes; (d) a comparison of two hospitals on crude mortality.

**Deliverable.** Four labelled scenarios with the direction.

**Acceptance criteria.** (a) immortal time, overstates benefit; (b) lead-time, overstates survival
benefit; (c) healthy-user, overstates benefit; (d) confounding by case mix, direction depends on the
case mix and can reverse the conclusion.

**Time.** 30 minutes.

### Exercise 7 — Audit a quality metric (critique)

A network reports a "sepsis mortality rate" defined as deaths among patients with an ICD code for
sepsis divided by all patients with the code. The rate fell from 9.1% to 7.8% after a coding
educational programme. Critique the metric and propose a validation and a sensitivity analysis.

**Deliverable.** A one-page audit with three defects, a validation plan, and a sensitivity analysis.

**Acceptance criteria.** Defects include the code-based case definition, the possibility that the
denominator expanded with milder cases, and the absence of a severity adjustment; the validation plan
uses chart review with a PPV estimate; the sensitivity analysis varies the case definition and
stratifies by severity.

**Time.** 45 minutes.

### Exercise 8 — Design a registry analysis plan (design)

Design a one-page analysis plan for a regional registry of patients with a chronic disease. Specify:
the population, the case definition and its validation, the denominator, the prevalence and incidence
measures, the standard population, the stratifiers, the biases, and the sensitivity analyses.

**Deliverable.** A one-page plan with seven labelled sections.

**Acceptance criteria.** Case definition has a source and a validation plan; denominator and
person-time are defined; prevalence and incidence are both specified with time windows; the standard
population is named; at least three biases are named with direction; at least two sensitivity analyses
are specified.

**Time.** 50 minutes.

## 13. Self-check answers

### Answer to Exercise 1

| Question | Measure | Reason |
|---|---|---|
| Adults currently with diabetes | Prevalence | Existing cases at a point in time |
| Surgical site infection within 30 days | Incidence proportion | Fixed window, closed cohort at risk |
| New infections per catheter-day | Incidence rate | Variable dwell time, person-time denominator |
| Readmission within 30 days | Incidence proportion | Fixed window, defined cohort of discharges |

The catheter-day question is the one that requires a rate: catheters stay in for different durations, so
a per-patient denominator would weight a one-day catheter the same as a ten-day catheter.

### Answer to Exercise 2

```text
Person-time = 10 + 5 + 2 = 17 person-days
Rate = 6/17 x 100 = 35.29 per 100 person-days

Patient denominator
  Rate = 6/3 = 2.0 events per patient
```

The patient denominator says nothing about exposure time. Patient 3 contributed 2 days and four events,
so the rate per day is much higher than the per-patient count suggests. The person-time rate is the
correct measure when follow-up varies.

### Answer to Exercise 3

```text
Low-risk:  16/1,600 = 1.00%
High-risk: 44/400   = 11.00%

Weighted check:
  (1,600/2,000) x 1.00% + (400/2,000) x 11.00%
  = 0.80 x 1.00 + 0.20 x 11.00
  = 0.80 + 2.20
  = 3.00%
```

The crude rate is a weighted average of the stratum-specific rates, with weights equal to the
population proportions. A patient considering enrolment should be told the rate for their own risk
group: 1.0% if low risk, 11.0% if high risk. The crude 3.0% describes the programme's population, not
the individual's prognosis.

### Answer to Exercise 4

```text
Hospital X
  Crude: (12 + 60)/1,000 = 72/1,000 = 7.20%
  Low:   12/600 = 2.00%
  High:  60/400 = 15.00%

Hospital Y
  Crude: (12 + 54)/1,000 = 66/1,000 = 6.60%
  Low:   12/400 = 3.00%
  High:  54/600 = 9.00%

Direct standardisation, 50/50:
  X: 0.50 x 2.00 + 0.50 x 15.00 = 1.00 + 7.50 = 8.50%
  Y: 0.50 x 3.00 + 0.50 x 9.00  = 1.50 + 4.50 = 6.00%
```

Hospital Y performs better after standardisation (6.0% versus 8.5%), and the crude comparison agrees
in direction (6.6% versus 7.2%). Hospital X has the lower low-risk rate (2.0% versus 3.0%) but the
higher high-risk rate (15.0% versus 9.0%). The summary depends on the standard: a standard weighted
entirely to low-risk patients would favour X. Both strata should be reported.

### Answer to Exercise 5

```text
Risk exposed    = 120/1,000 = 0.120
Risk unexposed  = 80/1,000  = 0.080

RR = 0.120/0.080 = 1.500
OR = (120/880)/(80/920) = 0.136364/0.086957 = 1.5682
RD = 0.120 - 0.080 = 0.040 = 4.0 percentage points
NNT = 1/0.040 = 25

PAF = Pe(RR-1)/(1 + Pe(RR-1)) with Pe = 0.50
    = 0.50 x 0.50/(1 + 0.50 x 0.50)
    = 0.25/1.25
    = 0.20 = 20.0%
```

The OR exceeds the RR because the outcome is common in both groups. Reporting the OR as a risk ratio
would overstate the relative effect by 4.5%.

### Answer to Exercise 6

| Scenario | Bias | Direction |
|---|---|---|
| (a) Statin exposure defined during follow-up | Immortal time | Overstates benefit |
| (b) Five-year survival from diagnosis | Lead-time | Overstates survival benefit |
| (c) Wearable users have better outcomes | Healthy-user selection | Overstates benefit |
| (d) Crude hospital mortality comparison | Confounding by case mix | Direction depends on case mix; can reverse |

In (a), a patient must survive to receive the statin, so the treated group is guaranteed survival until
treatment. In (b), diagnosis is moved earlier without necessarily delaying death. In (c), the users
differ from non-users in health, wealth, and adherence. In (d), the direction is not fixed; the worked
example in section 6 shows a reversal.

### Answer to Exercise 7 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Case definition | 25% | Code-based, unvalidated, sensitive to coding practice |
| Denominator | 25% | The denominator may have expanded with milder cases; the rate is not severity-adjusted |
| Confounding | 20% | The educational programme may have changed coding, not care |
| Validation | 15% | Chart review with PPV and sensitivity estimates, stratified by period |
| Sensitivity analysis | 15% | Vary the case definition; stratify by severity; compare with a clinical registry |

A weak audit says the rate "may be biased" without naming the mechanism or proposing a validation.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Population and case definition | 20% | Eligible population, source, criteria, and validation plan |
| Denominator and person-time | 20% | Both defined, with time zero and follow-up rules |
| Measures | 20% | Prevalence and incidence with time windows |
| Standardisation | 15% | Named standard population and stratifiers |
| Bias assessment | 15% | At least three biases with direction |
| Sensitivity analyses | 10% | At least two, each tied to a fragile assumption |

A weak plan lists the data fields but does not define the population, the case, or the denominator,
which are the three components that determine whether any measure is interpretable.

## 14. Interview questions

### Question 1 — "What is the difference between incidence and prevalence?"

**Strong answer.** "Prevalence is existing cases divided by the population at a time; incidence is new
cases over a period, either as a proportion of those at risk or as a rate per person-time. Prevalence
depends on both incidence and duration, so a chronic disease with low incidence can have high
prevalence."

**Weak answer.** "Incidence is new cases, prevalence is all cases." That is the definition but not the
relationship, which is what the question tests.

### Question 2 — "Why use person-time instead of the number of patients?"

**Strong answer.** "Because patients have different follow-up durations. A rate per person-time weights
each patient by the time they were actually at risk. Using the number of patients weights a two-day
patient the same as a ten-day patient and biases the rate toward the long-followed."

**Weak answer.** "Because it is more accurate." It is the correct denominator when exposure time varies;
that is the reason.

### Question 3 — "How do you compare mortality across two hospitals with different case mixes?"

**Strong answer.** "Stratify by the strong prognostic factors, then either standardise directly to a
common structure or compute an SMR against a common reference. Report the stratum-specific rates as
well, because a summary can hide a reversal. State the standard population and the risk-adjustment
model."

**Weak answer.** "Use the crude rate; it is what happened." The crude rate is a mixture of care and case
mix, and the mixture can reverse the comparison.

### Question 4 — "What is an SMR and what does it not tell you?"

**Strong answer.** "The SMR is observed events divided by expected events, where expected events come
from applying reference rates to the study population's structure. It does not tell you the absolute
rate, and it does not adjust for variables that are not in the reference rates. It is also sensitive to
the choice of reference."

**Weak answer.** "It is the mortality rate adjusted for age." It is a ratio, not a rate, and the
adjustment depends on the reference rates.

### Question 5 — "When can you report an odds ratio as a risk ratio?"

**Strong answer.** "When the outcome is rare, roughly below 10%. Above that, the OR is further from 1
than the RR, and reporting it as an RR overstates the effect. For a common outcome, report the RR if
the design allows, or convert the OR using the baseline risk, or report the risk difference."

**Weak answer.** "Always, because they are close." They are close only when the outcome is rare.

### Question 6 — "What is immortal time bias?"

**Strong answer.** "It is the bias that arises when the treated group is defined by an event that
occurs after follow-up starts, so the period before that event is credited to the treated group even
though the outcome could not have occurred. It produces an apparent benefit for almost any treatment.
The fix is to start follow-up at treatment initiation or use a landmark analysis."

**Weak answer.** "It is when patients are immortal." It is a follow-up alignment problem, not a
biological property.

### Question 7 — "Why can five-year survival from diagnosis mislead about a screening programme?"

**Strong answer.** "Because survival is measured from diagnosis, and screening moves diagnosis earlier
without necessarily delaying death. Five-year survival rises even if the date of death is unchanged.
The correct endpoint is disease-specific mortality in a randomised comparison."

**Weak answer.** "Because survival is not mortality." That is true but does not explain the mechanism,
which is lead time.

### Question 8 — "What is a surveillance case definition?"

**Strong answer.** "It is a definition designed for consistent monitoring and comparison across
populations, not for diagnosing individuals. CLABSI is the standard example: it is applied uniformly
across hospitals to produce a comparable rate, and a patient can meet it without a clinical
bloodstream infection."

**Weak answer.** "It is the definition used by public health." That does not capture the distinction
between surveillance and diagnosis, which is the point.

### Question 9 — "How would you validate a code-based case definition?"

**Strong answer.** "Against a chart review or a clinical adjudication sample, estimating sensitivity,
specificity, and positive predictive value. I would stratify by site and period, because coding
practices vary. Then I would report the validation and, if the PPV differs across sites, restrict or
adjust the comparison."

**Weak answer.** "Check that the code exists in the data dictionary." That checks availability, not
validity.

### Question 10 — "What is the population attributable fraction and when is it meaningful?"

**Strong answer.** "It is the proportion of cases in the population that would not occur if the
exposure were eliminated, assuming a causal effect. It depends on the exposure prevalence and the
relative risk. It is meaningful only under a causal assumption, which an observational association does
not establish on its own."

**Weak answer.** "It is the fraction of cases caused by the exposure." That is the definition under
causation; the question is whether causation has been established.

### Question 11 — "What is the difference between confounding and effect modification?"

**Strong answer.** "Confounding is a third variable that distorts the association and should be adjusted
for. Effect modification is a genuine difference in the effect across strata and should be reported, not
adjusted away. The distinction is whether the stratum-specific effects differ because of the
stratifier's causal role or because the population composition differs."

**Weak answer.** "They are both about subgroups." They require opposite responses: adjustment versus
stratified reporting.

### Question 12 — "How do you handle a policy change in the middle of a surveillance period?"

**Strong answer.** "Annotate it and do not compare across it without adjustment. A coding change, a
testing change, or a definition change produces a step change that is not a care change. I would split
the series at the change, compare within periods, and use an interrupted time series if a trend
comparison is needed."

**Weak answer.** "Exclude the change period." Excluding it hides the artefact and may bias the trend;
the change must be modelled or the comparison restricted.

## 15. Key terms

- **Attack rate** — incidence proportion in an outbreak.
- **Attributable fraction** — proportion of cases among the exposed attributable to the exposure.
- **Case definition** — the criteria that classify a person as a case.
- **Case-mix adjustment** — adjustment for patient severity and comorbidity.
- **Censoring** — end of observation without the outcome.
- **Competing risk** — an event that prevents the outcome of interest.
- **Confounding** — a third variable associated with exposure and outcome.
- **Crude rate** — a rate calculated without standardisation.
- **Cumulative incidence** — risk over a period, accounting for competing risks.
- **Denominator** — the population or person-time base of a measure.
- **Direct standardisation** — applying study rates to a standard structure.
- **Ecological fallacy** — inferring individual effects from group data.
- **Effect modification** — a difference in effect across strata.
- **Epidemiology** — the study of the distribution and determinants of health states.
- **Hazard** — instantaneous event rate among those still at risk.
- **Healthy-user bias** — healthier people use preventive services.
- **Immortal time bias** — follow-up time during which the outcome cannot occur.
- **Incidence proportion** — new cases divided by the population at risk at the start.
- **Incidence rate** — new cases divided by person-time.
- **Indirect standardisation** — applying standard rates to the study structure.
- **Information bias** — measurement error differing by exposure or outcome.
- **Lead-time bias** — earlier diagnosis lengthens survival without delaying death.
- **Length-time bias** — screening detects slower-growing disease.
- **NNT** — 1/risk difference.
- **Notifiable disease** — a disease required to be reported.
- **Odds ratio** — ratio of odds.
- **PAF** — population attributable fraction.
- **Person-time** — the sum of time at risk across individuals.
- **Point prevalence** — prevalence at one instant.
- **Population at risk** — people who could develop the outcome.
- **Prevalence** — existing cases divided by the population at a time.
- **RECORD** — reporting guideline for studies using routinely collected data.
- **Registry** — a systematic collection of cases for a defined population.
- **Risk difference** — absolute difference in risk.
- **Risk ratio** — ratio of risks.
- **Selection bias** — a non-representative sample or follow-up.
- **SIR** — standardised infection ratio, observed divided by predicted.
- **SMR** — standardised mortality ratio, observed divided by expected.
- **Standard population** — the population whose structure is used for adjustment.
- **Standardised rate** — a rate adjusted to a standard population structure.
- **STROBE** — reporting guideline for observational studies.
- **Surveillance definition** — a case definition designed for monitoring, not diagnosis.
- **Survivorship bias** — analysing only those who survived to be sampled.
- **Under-reporting** — cases missing from surveillance.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Methods and reporting

- Rothman, K. J., Greenland, S., Lash, T. L., *Modern Epidemiology*, 3rd edition, Lippincott Williams
  & Wilkins, 2008.
- von Elm, E. et al., "The Strengthening the Reporting of Observational Studies in Epidemiology
  (STROBE) statement: guidelines for reporting observational studies", *The Lancet* 370(9596),
  1453–1457, 2007. https://doi.org/10.1016/S0140-6736(07)61602-X
- Benchimol, E. I. et al., "The REporting of studies Conducted using Observational Routinely-collected
  health Data (RECORD) statement", *PLoS Medicine* 12(10), e1001885, 2015.
  https://doi.org/10.1371/journal.pmed.1001885
- Suissa, S., "Immortal time bias in pharmacoepidemiology", *American Journal of Epidemiology* 167(4),
  492–499, 2008. https://doi.org/10.1093/aje/kwm324
- Tripepi, G., Jager, K. J., Dekker, F. W., Zoccali, C., "Stratification for confounding — Part 2:
  Direct and indirect standardization", *Nephron Clinical Practice* 116(4), c322–c325, 2010.
  https://doi.org/10.1159/000319591

### Classifications and surveillance

- World Health Organization, *International Classification of Diseases, 11th Revision (ICD-11)*.
  https://icd.who.int/
- World Health Organization, *International Health Regulations (2005)*, 3rd edition, 2016.
  https://www.who.int/publications/i/item/9789241580496
- Centers for Disease Control and Prevention, *National Healthcare Safety Network (NHSN) Patient Safety
  Component Manual*, Chapter 4: Bloodstream Infection Event.
  https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf
- Centers for Disease Control and Prevention, *NHSN Key Terms*.
  https://www.cdc.gov/nhsn/pdfs/pscmanual/16psckeyterms_current.pdf
- Centers for Disease Control and Prevention, *NHSN FAQs: Analysis*.
  https://www.cdc.gov/nhsn/faqs/faq-analysis.html
- Centers for Medicare & Medicaid Services, *Hospital Readmissions Reduction Program*.
  https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp

### Global health estimates

- Stevens, G. A. et al., "Guidelines for Accurate and Transparent Health Estimates Reporting: the
  GATHER statement", *The Lancet* 388(10062), e19–e23, 2016.
  https://doi.org/10.1016/S0140-6736(16)30388-9

> **Note:** The NHSN manual is updated regularly; the chapter cited is the current version at the access
> date. Rates and definitions should be checked against the version in force for the reporting period.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`06_pharmacokinetics.md`](06_pharmacokinetics.md)
- Next topic: [`08_engineering-and-device-basics.md`](08_engineering-and-device-basics.md)
- Probability and intervals: [`02_probability-and-statistics-basics.md`](02_probability-and-statistics-basics.md)
- Biological basis of the variables: [`05_biological-basics.md`](05_biological-basics.md)
- Assumption recording: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — `03_cohort_pipeline.ipynb` builds a cohort and is the natural
  place to apply the denominator and case-definition checks in section 5.
