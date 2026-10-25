# Target Trial Emulation

> Module 04 — Statistics and Causal Inference
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: 02, 05, 06
> Version 2026.3

---

## 1. Why this topic exists

Most industry questions about treatment effects cannot be answered by a randomized trial. The trial would be too large, too slow, too expensive, or unethical. The alternative is an observational analysis, and the observational analysis is usually conducted as if the data were a trial with missing pieces: a cohort is assembled, an exposure is defined, a regression is fitted, and an effect is reported. The result is often biased by design decisions that were never made explicitly.

Target trial emulation is a discipline for making those decisions explicit. The analyst first specifies the hypothetical randomized trial that would answer the question, including eligibility, treatment strategies, assignment, time zero, outcomes, follow-up, and the causal contrast. Then the analyst asks whether each element can be emulated with the available data. The protocol is the benchmark; the observational analysis is judged by how closely it reproduces the benchmark and where it cannot.

The approach was formalized by Hernán and Robins in the American Journal of Epidemiology in 2016 and extended in a 2016 paper in the Journal of Clinical Epidemiology by Hernán, Sauer, Hernández-Díaz, Platt, and Shrier, which catalogues four ways that misaligning time zero, eligibility, and treatment assignment produces bias. The framework is now used in regulatory submissions, payer analyses, and clinical research, and the TARGET statement (Transparent Reporting of Observational Studies Emulating a Target trial), published in JAMA in September 2025, provides a 21-item reporting checklist for studies that explicitly aim to emulate a trial.

The biases the framework prevents are concrete:

- **Immortal time bias.** A treated group defined by an event that occurs after time zero is guaranteed to survive until that event. The period before treatment is immortal, and counting it as treated time makes the treatment look protective.
- **Prevalent user bias.** Patients who have already survived on treatment are selected for the cohort. They are healthier than incident users, and the comparison is with a different population.
- **Misaligned time zero.** Treatment and control groups start follow-up at different points, so the risk sets are not comparable.
- **Inconsistent eligibility.** Eligibility is assessed at different times for different patients, often using information that was not available at the start of follow-up.
- **Informative censoring.** Patients who stop treatment are censored in a way that is related to prognosis.

Each of these can be described as a failure to specify the target trial. The protocol forces the analyst to say when follow-up starts, who is eligible at that moment, and what treatment strategy is being compared.

This topic builds the protocol, the mapping to data, the analysis, and the diagnostics. It assumes the causal graph and propensity methods from Topics 05 and 06.

> **Note:** A target trial emulation is not a claim that the observational estimate equals the randomized estimate. It is a claim that the design decisions are aligned with a hypothetical trial, and that the remaining differences are stated. The estimate can still be biased by unmeasured confounding.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a seven-component target trial protocol for a comparative question.
2. **Map** each protocol component to a data source and **identify** where the mapping fails.
3. **Align** time zero across treatment strategies and **detect** immortal time and prevalent-user bias.
4. **Distinguish** intention-to-treat, per-protocol, and as-treated contrasts, and **choose** the one that matches the decision.
5. **Specify** a treatment strategy, including a grace period or a dynamic rule, without using future information.
6. **Design** a cloning-censoring-weighting analysis for a per-protocol contrast.
7. **Diagnose** an emulation with a time-zero audit, a person-time check, and a positivity assessment.
8. **Report** the protocol, the emulation, and the discrepancies using the TARGET checklist.

Outcome 3 maps to Exercise 3 and the worked example. Outcome 4 maps to Exercise 4. Outcome 8 maps to Exercise 8.

## 3. The core mental model

The target trial is a template. The emulation maps each element of the template to a data operation, and the analysis estimates the effect in the emulated trial.

```text
  TARGET TRIAL PROTOCOL
  1. Eligibility        who would be enrolled, assessed when?
  2. Treatment strategies   what is compared, with what grace period?
  3. Assignment         how would treatment be assigned?
  4. Time zero          when does follow-up start for each arm?
  5. Outcomes           what is measured, how, over what horizon?
  6. Follow-up          how long, and what censors?
  7. Causal contrast    ITT, per-protocol, or as-treated?
  8. Analysis plan      how is confounding controlled?
        |
        |  map each element to the data
        v
  EMULATION
  eligibility algorithm, treatment algorithm, time-zero rule,
  outcome definition, censoring rule, estimator
        |
        |  diagnose the mapping
        v
  TIME-ZERO AUDIT / PERSON-TIME CHECK / POSITIVITY
        |
        v
  ESTIMATE + INTERVAL + SENSITIVITY
```

### 3.1 The seven protocol components

| Component | Question | Example |
|---|---|---|
| Eligibility | Who would be enrolled, and at what moment? | Adults discharged alive after heart failure, at discharge |
| Treatment strategies | What strategies are compared, including initiation and adherence? | Program enrollment within 7 days vs no enrollment |
| Assignment | How would treatment be assigned? | Randomization, if the trial were run |
| Time zero | When does follow-up start? | Day of discharge for both strategies |
| Outcomes | What is measured, how, and over what horizon? | First unplanned readmission within 30 days |
| Follow-up | How long, and what ends follow-up? | 30 days, death, disenrollment |
| Causal contrast | ITT, per-protocol, or as-treated? | ITT for the offer; per-protocol for the strategy |

The eighth element, the analysis plan, describes how confounding is controlled. It is not part of the trial protocol in the same sense, but it is part of the emulation.

### 3.2 Time zero is the load-bearing element

The four emulation failures in Hernán et al. (2016) are all misalignments of time zero (T0), eligibility (E), and assignment (A):

1. **T0 after E and A:** left truncation or prevalent users. The cohort starts after treatment has already begun.
2. **T0 at E but after A:** selection bias. Treatment is assigned using information that occurs after T0.
3. **T0 before E and A:** immortal time. No one can die between assignment and eligibility, so the treated group is guaranteed to survive.
4. **T0 at E but strategy assigned using post-T0 information:** classical immortal time bias, such as requiring three prescriptions before classifying a patient as treated.

The remedy in every case is the same: define T0 first, assess eligibility at T0, and assign the treatment strategy using only information available at T0.

### 3.3 Intention-to-treat, per-protocol, and as-treated

The causal contrast is a choice, and the choice changes the estimate:

- **Intention-to-treat (ITT):** compare the strategies as assigned at time zero, regardless of adherence. It answers "what happens if we offer the strategy?" It is the natural contrast for a policy or coverage decision.
- **Per-protocol:** compare the strategies as if they were followed. It answers "what happens if the strategy is followed?" It requires censoring at deviation and inverse probability of censoring weights, or a parametric g-formula.
- **As-treated:** compare the treatment actually received at each time. It is vulnerable to time-varying confounding and is rarely the right contrast for a decision.

The ITT emulation is usually easier: assign the strategy at time zero, follow everyone, and compare. The per-protocol emulation is harder: it requires a model for adherence and censoring, and the assumptions are stronger.

### 3.4 Cloning, censoring, and weighting

For a per-protocol contrast with time-varying treatment, the cloning-censoring-weighting approach makes the target trial explicit:

1. **Clone** each patient into one copy for each treatment strategy at time zero.
2. **Censor** a clone when it deviates from its assigned strategy.
3. **Weight** the remaining clones by the inverse probability of not being censored, using the time-varying covariates that predict deviation.

The result is a pseudo-population in which the strategy is followed, and the contrast is the per-protocol effect. The method is conceptually clean and computationally demanding; it also makes the assumptions about censoring visible.

### 3.5 What the emulation cannot fix

- It cannot fix unmeasured confounding. Randomization is replaced by an assumption.
- It cannot fix a data source that does not measure the outcome or the confounders.
- It cannot fix a treatment that is not well defined.
- It cannot create overlap where the strategies are never both plausible.
- It cannot make a biased estimate unbiased by labelling it an emulation.

The framework's value is that it separates the design decisions from the data limitations, so each can be evaluated.

### 3.6 The reporting standard

The TARGET statement, published in JAMA in September 2025, provides a 21-item checklist. The items cover the title and abstract, the causal question, the target trial specification, the emulation, the results, and the discussion. The core requirement is that the target trial protocol and its emulation are reported side by side, so the reader can see which elements were emulated exactly and which were approximated.

A report that describes an observational cohort without the target trial protocol cannot show which design decisions were made. The TARGET checklist exists to close that gap.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Target trial | The hypothetical randomized trial that would answer the question | Called the observational study | Protocol |
| Emulation | The observational analysis designed to approximate the target trial | Called a validation of the trial | Analysis |
| Eligibility | The criteria for entry, assessed at time zero | Assessed using future information | Criteria |
| Treatment strategy | The intervention being compared, including initiation and adherence | Defined as "ever exposed" | Strategy |
| Grace period | A window after time zero during which treatment initiation is allowed | Used without a time zero rule | Time |
| Time zero | The start of follow-up for each strategy | Different across arms | Date |
| Assignment | How treatment would be assigned in the trial | Confused with actual treatment | Mechanism |
| ITT | Compare strategies as assigned, regardless of adherence | Called "as-treated" | Contrast |
| Per-protocol | Compare strategies as if followed | Implemented without censoring weights | Contrast |
| As-treated | Compare treatment actually received | Vulnerable to time-varying confounding | Contrast |
| Immortal time | Follow-up during which the outcome cannot occur | Counted as exposed time | Time |
| Prevalent user | A patient already on treatment at cohort entry | Included without a washout | Patient |
| New-user design | A cohort of patients initiating treatment at time zero | Confused with prevalent users | Design |
| Left truncation | Entry after the start of follow-up | Ignored in the risk set | Time |
| Cloning | Creating one copy of each patient per strategy | Confused with matching | Design |
| Censoring at deviation | Removing a clone that does not follow its strategy | Treated as loss to follow-up | Design |
| Inverse probability of censoring weighting | Weights that adjust for informative censoring | Confused with IPTW | Weight |
| Positivity | Both strategies are possible for every patient type | Assumed from the data | Assumption |
| Dynamic strategy | A strategy that depends on time-varying information | Specified using future data | Rule |
| Washout period | A pre-enrollment window used to identify new users | Omitted, allowing prevalent users | Time |
| Eligible person-time | The person-time during which a patient could be enrolled | Ignored in cohort construction | Person-time |
| Disenrollment | Loss of insurance or data coverage | Treated as random censoring | Event |
| Outcome horizon | The time over which the outcome is measured | Left implicit | Time |
| Estimand | The quantity the emulation targets | Confused with the estimator | Specification |
| TARGET statement | A reporting checklist for emulation studies | Treated as a design requirement | Checklist |
| RCT-DUPLICATE | An initiative that emulated 32 trials | Treated as proof of validity | Study |
| Data availability | Whether the data can support the protocol | Assumed from the sample size | Constraint |
| Measurement error | The difference between the construct and the recorded variable | Ignored in the protocol | Error |
| Confounding control | The method used to adjust for measured confounders | Confused with randomization | Analysis |
| Sensitivity analysis | An analysis under a different assumption | Repeating the main analysis | Analysis |

Two conventions in this topic:

- Time zero is stated as a rule, not a variable: "the date of discharge, for both strategies."
- A grace period is stated with its length and its justification, not chosen to maximize the effect.

> **Note:** The target trial framework does not require that the trial be feasible. It requires that the trial be specified precisely enough that the emulation can be compared with it. A trial that could never be run is still a useful benchmark if its protocol is clear.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 7 write the protocol. Steps 8 to 11 map it to the data and estimate. Steps 12 to 14 diagnose and report.

### Step 1. Write the causal question and the protocol

**Input:** The decision, the population, the treatment options, and the outcome.

**Output:** A one-page protocol with the seven components and the analysis plan.

**Why it exists:** The protocol is the benchmark. Without it, every design decision is ad hoc and cannot be reviewed.

| Component | Must be specific enough to | Example |
|---|---|---|
| Eligibility | Be implemented as an algorithm at time zero | Age at least 18, discharged alive, primary heart-failure diagnosis |
| Treatment strategies | Distinguish initiation from adherence | Enrollment within 7 days vs no enrollment in that window |
| Assignment | State what randomization would have done | Randomization at discharge, concealed allocation |
| Time zero | Be the same rule for both strategies | Date of discharge |
| Outcomes | Be measured identically in both arms | First unplanned inpatient readmission within 30 days |
| Follow-up | State the horizon and the censoring events | 30 days, death, disenrollment |
| Causal contrast | Name ITT, per-protocol, or as-treated | ITT for the offer of the program |

### Step 2. Implement eligibility at time zero

**Input:** The protocol criteria and the data.

**Output:** An eligibility algorithm that uses only information available at or before time zero.

**Why it exists:** Eligibility assessed with future information selects patients who survived or who later met a criterion. That is a form of selection on the future, and it biases the comparison.

Checks:

- Every criterion uses a measurement date at or before time zero.
- The algorithm does not exclude patients because of an event after time zero.
- The number excluded at each step is reported.
- The eligible population is compared with the protocol population.

### Step 3. Define the treatment strategies and the grace period

**Input:** The clinical options and the data on initiation.

**Output:** A strategy definition with an initiation window and an adherence rule.

**Why it exists:** "Ever treated" is not a strategy. The strategy must specify when treatment starts and what happens if it does not.

| Strategy type | Definition | Example | Note |
|---|---|---|---|
| Point initiation | Start on the day of time zero | Drug administered in hospital | Rare in observational data |
| Grace period | Start within a defined window after time zero | Program enrollment within 7 days | The window must be pre-specified |
| Dynamic | Start when a time-varying criterion is met | Initiate when eGFR falls below 30 | Requires g-methods |
| Discontinuation | Stop after a defined event | Stop after remission | Different estimand |
| Switching | Change from one treatment to another | Switch after intolerance | Different estimand |

**Grace period caution.** A grace period is a design choice. It should be justified by the clinical process, not by the size of the effect. A 7-day window for a program that enrolls patients at discharge is plausible. A 365-day window for a drug that should start immediately is a way of classifying almost everyone as treated.

### Step 4. Align time zero across strategies

**Input:** The strategy definitions and the data.

**Output:** One time-zero rule applied to every patient, with the same follow-up start for both strategies.

**Why it exists:** The risk sets must start at the same moment. If treated patients start at treatment and controls start at diagnosis, the comparison is between different periods of disease.

| Alignment problem | Symptom | Remedy |
|---|---|---|
| Treated start at treatment, controls at diagnosis | Treated group has shorter follow-up | Set time zero at diagnosis for both |
| Treated defined by surviving to treatment | Immortal time | Time-varying exposure or cloning |
| Eligibility uses future labs | Selected population | Assess at time zero |
| Grace period differs by arm | Different windows | Same window for both |

### Step 5. Define the outcome and its ascertainment

**Input:** The protocol outcome and the data source.

**Output:** An outcome algorithm with a validation result.

**Why it exists:** The emulation is only as good as the outcome measurement. A code-based outcome that captures transfers as readmissions will bias both arms, and the bias can differ by exposure if coding practice differs.

- Use the same outcome definition in both arms.
- Validate the code set in a sample.
- Decide whether the outcome is the first event or a recurrent event.
- Define the competing events and how they are handled.
- State the horizon and the ascertainment window.

### Step 6. Define follow-up and censoring

**Input:** The protocol follow-up and the data on disenrollment, death, and loss.

**Output:** A censoring rule for each reason, with the assumption required.

**Why it exists:** Informative censoring can mimic or mask an effect. The emulation must state which censoring events are administrative and which are informative.

| Censoring event | Assumption | Method |
|---|---|---|
| End of study period | Independent | Administrative censoring |
| Death | Competing event or censoring depending on the estimand | Competing-risk analysis |
| Disenrollment | Possibly informative | IPCW; sensitivity analysis |
| Treatment deviation | Informative for per-protocol | Censor and weight |
| Outcome ascertainment gap | Possibly informative | Validation; sensitivity analysis |

### Step 7. Choose the causal contrast

**Input:** The decision and the strategy definitions.

**Output:** ITT, per-protocol, or as-treated, with a justification.

**Why it exists:** The contrast determines the estimand and the analysis. A per-protocol estimate requires a model for adherence; an ITT estimate does not.

| Decision | Contrast | Why |
|---|---|---|
| Should the program be offered? | ITT | The offer is the decision |
| Should the program be continued once started? | Per-protocol | Adherence is part of the strategy |
| What is the effect of the drug as taken? | Per-protocol or as-treated | Requires adherence modeling |
| What is the effect of the policy? | ITT | Policy is an offer |

### The two contrasts in arithmetic

The ITT and per-protocol contrasts differ in what they do with patients who deviate. A simple table makes the difference concrete:

| Patient | Assigned strategy | Actual treatment | ITT contribution | Per-protocol contribution |
|---|---|---|---|---|
| 1 | Program | Enrolled, stayed | Program | Program |
| 2 | Program | Never enrolled | Program | Censored at deviation |
| 3 | Program | Enrolled, stopped at day 30 | Program | Censored at day 30 |
| 4 | No program | Enrolled privately | No program | Censored at deviation |
| 5 | No program | No program | No program | No program |

The ITT estimate uses everyone as assigned. The per-protocol estimate uses only the person-time during which the assigned strategy was followed, weighted by the inverse probability of remaining uncensored. The per-protocol estimate answers a different question, and it requires a model for the censoring.

**Which one to report.** Report the ITT estimate as primary for an offer or coverage decision, and the per-protocol estimate as a secondary analysis when adherence is part of the strategy. Report both when the decision could change depending on whether the effect is due to the offer or to the treatment itself.

### Step 8. Map the protocol to the data and state the gaps

**Input:** The protocol and the data dictionary.

**Output:** A side-by-side table of the protocol element, the data implementation, and the gap.

**Why it exists:** The gaps are where the emulation differs from the trial. They must be reported, not hidden.

| Protocol element | Data implementation | Gap |
|---|---|---|
| Eligibility | Diagnosis codes plus age | Codes may miss cases |
| Treatment strategy | Enrollment record within 7 days | Enrollment may be underrecorded |
| Time zero | Discharge date | Discharge date may be missing |
| Outcome | Readmission code | Transfers may be counted |
| Follow-up | Insurance enrollment | Disenrollment is informative |

### Step 9. Assess positivity for the strategies

**Input:** The eligible population and the strategy definitions.

**Output:** A description of who could plausibly receive each strategy.

**Why it exists:** A strategy that is never used in a subgroup cannot be compared there. The emulation must state the population in which both strategies are possible.

- Cross-tabulate strategy by site, year, and key covariates.
- Identify subgroups with zero or near-zero probability of one strategy.
- Restrict the target population or report the limitation.
- Do not extrapolate the effect to a subgroup with no data.

### Step 10. Estimate the effect

**Input:** The aligned cohort, the contrast, and the confounder adjustment.

**Output:** An estimate and an interval.

**Why it exists:** The estimate is conditional on the contrast and the alignment; the interval reflects the design.

| Contrast | Estimator | Note |
|---|---|---|
| ITT | IPTW or standardization at time zero | Weights from baseline covariates |
| Per-protocol, point treatment | IPTW with censoring at deviation | Censoring weights |
| Per-protocol, time-varying | Cloning-censoring-weighting or g-formula | Time-varying weights |
| As-treated | Time-varying Cox or MSM | Vulnerable to time-varying confounding |

The confounder adjustment follows Topic 06: a treatment model from the DAG, balance diagnostics, and a robust variance.

### Cloning, censoring, and weighting in more detail

The cloning-censoring-weighting procedure is the most explicit implementation of a per-protocol target trial:

1. **Clone** every patient at time zero into one copy per strategy. A patient is now in both arms.
2. **Censor** each clone when its observed treatment deviates from the strategy it represents. The clone that represents "program" is censored if the patient never enrolls; the clone that represents "no program" is censored if the patient enrolls.
3. **Weight** each remaining clone by the inverse probability of remaining uncensored, estimated from time-varying covariates. The weights recreate the population in which everyone followed the assigned strategy.

The weighted risk difference compares the strategies as if they were followed. The method makes the censoring assumption explicit: the probability of deviation depends on the measured time-varying covariates, not on the unmeasured prognosis.

**What can go wrong.** The weights can be unstable when deviation is common, and the censoring model can be misspecified. Report the weight distribution, the effective sample size, and a sensitivity analysis with a different censoring model. The method does not fix unmeasured confounding; it fixes the handling of treatment deviation.

### Step 11. Check the time-zero audit

**Input:** The cohort and the time-zero rule.

**Output:** A table of follow-up start, pre-time-zero person-time, and post-time-zero person-time by strategy.

**Why it exists:** The audit is the direct test for immortal time. If the treated group has person-time before treatment that is counted as treated, the audit shows it.

| Audit item | Expected | Red flag |
|---|---|---|
| Time zero distribution by arm | Similar | Treated starts later |
| Pre-time-zero person-time counted as exposed | Zero | Positive |
| Number at risk at time zero | Equals cohort | Fewer in one arm |
| Follow-up time by arm | Similar | Treated much shorter |
| Events before time zero | Zero | Any event |

### Step 12. Run diagnostics for prevalent users and selection

**Input:** The cohort and the treatment history.

**Output:** A description of prior treatment and a new-user restriction if needed.

**Why it exists:** Prevalent users are selected on prior survival; the diagnostic makes the selection visible.

- Identify patients already on treatment at time zero.
- Apply a washout period if the data allow.
- Compare incident and prevalent users on baseline characteristics.
- Report the proportion of prevalent users excluded.

### Step 13. Run sensitivity analyses

**Input:** The primary emulation.

**Output:** Estimates under alternative design choices.

**Why it exists:** The emulation depends on design choices; the sensitivity analyses show which ones matter.

| Assumption | Sensitivity analysis | What a change means |
|---|---|---|
| Grace period length | 3, 7, 14 days | The estimate depends on the window |
| Time zero | Discharge vs first follow-up | Misalignment matters |
| Censoring | IPCW vs complete follow-up | Censoring is informative |
| Confounding | E-value; negative control | Unmeasured confounding |
| Outcome definition | Alternative code set | Misclassification |
| Eligibility | Broader or narrower | Transportability |

### Step 14. Report against the TARGET checklist

**Input:** The protocol, the emulation, and the results.

**Output:** A report that a reviewer can map to the checklist.

**Why it exists:** The protocol and the emulation must be reported side by side for the gaps to be visible.

The core items:

1. Identify the study as a target trial emulation in the title or abstract.
2. State the causal question and the estimand.
3. Report the target trial protocol, component by component.
4. Report the emulation of each component and the discrepancies.
5. Report the confounder adjustment and the positivity assessment.
6. Report the effect with precision.
7. Report the sensitivity analyses.
8. Discuss the assumptions and the limits.

### The target trial and the DAG are complements

The target trial protocol defines the design: who is eligible, what is compared, when follow-up starts, and what is measured. The DAG defines the adjustment set: which variables must be blocked to identify the effect. Neither substitutes for the other.

A common failure is to draw a DAG but never align time zero, which produces immortal time even with a perfect adjustment set. The reverse failure is to align time zero but adjust for a collider, which produces a clean design with a biased estimate. The two tools answer different questions and must be used together.

A practical sequence is to write the protocol first, then draw the DAG at the specified time zero, then choose the adjustment set, then choose the estimator. Writing the DAG before the protocol often produces arrows for variables that are measured after time zero, which is a sign that the design has not been aligned.

**Choosing the grace period.** The grace period should be the time within which the treatment is expected to start under the strategy. It should come from the clinical protocol or the delivery process, not from the data. A sensitivity analysis over a range of windows (for example, 3, 7, and 14 days) shows whether the estimate depends on the choice. If it does, the estimand is not well defined, and the analysis should report the range rather than a single number.

A grace period is not a substitute for a time-varying analysis when treatment can start at any time. It is a simplification that makes the strategy definable at time zero. The simplification is acceptable when the window is short relative to the outcome horizon, and questionable when it is long.

The window also affects the population: a longer window classifies more patients as treated, including those who start late because they were initially too sick. That selection can reintroduce confounding by indication, which is why the window should be clinically justified and tested.

Both the window and the alignment must be written before the outcome is examined.

### Common protocol errors

| Error | What it looks like | Why it matters | Fix |
|---|---|---|---|
| Eligibility assessed after time zero | Patients who survive to a lab are included | Selection on survival | Assess at time zero |
| Treatment defined by a future prescription | "At least two fills in 90 days" | Immortal time | Define initiation at time zero |
| Time zero differs by arm | Treated start at first fill, controls at diagnosis | Risk sets not comparable | Same rule for both |
| No grace period for a slow-start treatment | Patients who start on day 2 are controls | Misclassification | Define a window |
| Grace period chosen post hoc | The window maximizes the effect | Design depends on the result | Pre-specify |
| Outcome measured differently by arm | Treated patients have more follow-up visits | Detection bias | Same ascertainment |
| Follow-up ends at disenrollment | Disenrollment differs by arm | Informative censoring | IPCW; sensitivity |

### Branch points at a glance

| If the situation is | Then the default is | Because |
|---|---|---|
| Treatment starts after time zero | Grace period plus landmark or cloning | Aligns the risk sets |
| Treatment changes over time | Clone-censor-weight or MSM | Per-protocol contrast |
| Patients already on treatment | Washout and new-user design | Avoids prevalent-user bias |
| Disenrollment is common | IPCW and a sensitivity analysis | Censoring may be informative |
| Outcome is fatal | Competing-risk or composite estimand | Death prevents the outcome |
| Both strategies are rare in a subgroup | Restrict the population | Positivity |
| Calendar practice changed | Stratify by period | Confounding by time |
| The decision is about an offer | ITT | The offer is the intervention |

### Edge cases that break the emulation

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| Treatment switching | The strategy is not followed | Switching rates by arm | Per-protocol with censoring |
| Grace period chosen post hoc | The estimate depends on the window | Sensitivity to the window | Pre-specify; report the range |
| Prevalent users | The cohort is selected on survival | Prior prescriptions at time zero | Washout; new-user design |
| Time-varying confounding | Baseline adjustment is insufficient | Treatment and confounders evolve | MSM; g-formula |
| Informative disenrollment | The risk set is selected | Disenrollment differs by arm | IPCW; sensitivity |
| Outcome misclassification | The endpoint is measured differently | Validation sample | Chart review; alternative codes |
| No overlap | One strategy is never used in a subgroup | Cross-tabulation | Restrict; report |
| Calendar trends | The standard of care changes | Enrollment spans a change | Stratify by period |

## 6. Worked example

### 6.1 The target trial protocol

A hospital wants to know whether a transitional-care program reduces 100-day mortality after heart-failure discharge. The program can only be started after discharge, and patients must survive to enroll. The naive analysis classifies patients as treated if they enroll within 10 days and follows everyone from discharge.

| Component | Specification |
|---|---|
| Eligibility | Adults discharged alive after a primary heart-failure admission |
| Treatment strategies | Program enrollment within 10 days of discharge vs no enrollment in that window |
| Assignment | Randomization at discharge, if the trial were run |
| Time zero | Date of discharge, for both strategies |
| Outcome | All-cause death within 100 days of discharge |
| Follow-up | 100 days, or until death |
| Causal contrast | ITT for the offer of the program |
| Analysis plan | Risk difference and risk ratio at 100 days, standardized for age and comorbidity |

The protocol has a built-in tension: the program cannot start before discharge, and enrollment takes time. The target trial would randomize at discharge and offer the program immediately. The emulation must decide how to handle the gap between discharge and enrollment.

### 6.2 The cohort

One hundred patients are eligible. Fifty enroll in the program on day 10. Fifty do not enroll. Ten of the non-enrollees die before day 10; no enrollee dies before day 10, because death before enrollment makes enrollment impossible.

After day 10, the 100-day mortality is 20% in both groups:

- Program: 10 deaths among 50 patients.
- No program: 8 deaths among the 40 patients still alive at day 10.

The true effect of the program is zero after day 10. The only difference is the early mortality among patients who could not enroll.

### 6.3 The naive analysis

The naive analysis classifies patients by enrollment within 10 days and follows everyone from discharge:

| Group | Deaths | n | Risk |
|---|---:|---:|---:|
| Program (enrolled by day 10) | 10 | 50 | 20.0% |
| No program | 18 | 50 | 36.0% |

```text
RD = 0.20 - 0.36 = -0.16 = -16.0 percentage points
RR = 0.20 / 0.36 = 0.5556
OR = (10/40) / (18/32) = 0.2500 / 0.5625 = 0.4444
SE(RD) = sqrt(0.20×0.80/50 + 0.36×0.64/50) = sqrt(0.0032 + 0.004608) = 0.08836
95% CI = -0.16 +/- 1.96 × 0.08836 = (-0.3332, 0.0132)
```

The naive analysis suggests a 16-percentage-point mortality benefit, with a confidence interval that includes zero. The point estimate is biased even though the interval is inconclusive.

### 6.3b Mapping the protocol to the data

| Protocol element | Data implementation | Emulation gap |
|---|---|---|
| Eligibility | Heart-failure diagnosis code plus discharge status | Codes may miss cases; no clinical validation in this example |
| Strategies | Enrollment record within 10 days vs none | Enrollment date is assumed exact |
| Time zero | Discharge date | Discharge date is assumed complete |
| Outcome | Death within 100 days | Death ascertainment is assumed complete |
| Follow-up | 100 days or death | No disenrollment in this example |
| Contrast | ITT for the offer | The naive analysis does not implement ITT |

The table is the core of the TARGET reporting requirement: each element is stated, and each gap is named. An emulation with no gaps is implausible; an emulation with unstated gaps is uninterpretable.

### 6.4 The time-zero audit

The audit counts the person-time before enrollment that the naive analysis attributes to the program:

```text
Program patients contribute 50 × 10 = 500 person-days before enrollment.
Those 500 days are counted as program time in the naive analysis.
No program patient can die during those days, because death prevents enrollment.
```

This is immortal time. The naive program group is guaranteed to survive 500 person-days, and the naive no-program group is not. The 10 early deaths are counted against the no-program group, which is why the naive estimate shows a benefit.

### 6.4b A cloning view of the same problem

The cloning-censoring-weighting approach makes the alignment explicit. Each of the 100 patients is cloned into two copies at discharge:

| Original patient | Program clone | No-program clone |
|---|---|---|
| Enrolls on day 10 | Followed from day 0 as program | Censored on day 10 (deviation) |
| Never enrolls, dies day 5 | Censored on day 5 (deviation or death) | Event on day 5 |
| Never enrolls, alive day 10 | Censored on day 10 (deviation) | Followed as no program |

The program clone of a non-enrollee is censored when the patient fails to enroll. The no-program clone of an enrollee is censored when the patient enrolls. The weighted analysis compares the clones that remain, with weights that adjust for the censoring.

In this example, the censoring is entirely determined by enrollment, which is observed, so the weights are 1 for the remaining clones and the estimate is the same as the landmark estimate. In a real analysis with time-varying predictors of enrollment, the weights would differ from 1 and would need to be reported with their distribution and effective sample size.

### 6.5 The landmark analysis

The landmark analysis starts follow-up at day 10 for both groups, when enrollment status is known:

| Group | Deaths after day 10 | At risk at day 10 | Risk |
|---|---:|---:|---:|
| Program | 10 | 50 | 20.0% |
| No program | 8 | 40 | 20.0% |

```text
RD = 0.20 - 0.20 = 0.00 = 0.0 percentage points
RR = 1.00
SE(RD) = sqrt(0.20×0.80/50 + 0.20×0.80/40) = sqrt(0.0032 + 0.0040) = 0.08485
95% CI = 0.00 +/- 1.96 × 0.08485 = (-0.1663, 0.1663)
```

The landmark estimate is zero, which matches the data-generating process. The difference between the naive and landmark estimates is the immortal time.

### 6.6 Diagnostic checks

**Check 1. Person-time by arm.** The naive program arm has 500 person-days before enrollment. The correct analysis has zero. If the audit shows positive pre-enrollment person-time counted as exposed, the analysis has immortal time.

**Check 2. Number at risk at time zero.** The naive cohort has 50 in each arm at day 0. The landmark cohort has 50 and 40 at day 10. The difference of 10 is the early mortality that the naive analysis attributes to the no-program arm.

**Check 3. The arithmetic of the reversal.** The naive no-program risk is 18/50 = 36%. The landmark no-program risk is 8/40 = 20%. The numerator falls by 10 (the early deaths) and the denominator falls by 10 (the same patients). The entire 16-point difference comes from moving those 10 patients out of the risk set. This is a useful check: if the difference between the naive and landmark estimates does not equal the early-event contribution, the cohort or the time zero is wrong.

**Check 4. The direction of the bias.** The naive estimate favours the program. Immortal time bias always favours the exposed group when the exposure requires survival, because the exposed group's pre-exposure time is guaranteed event-free. A naive estimate that favours the treated group in a new-user design with a grace period should be checked for this bias before it is interpreted.

**Check 5. The result is unchanged when the early event rate changes.** The true post-day-10 effect is zero in both scenarios, and the landmark estimate is zero in both. The naive estimate moves from -16 to -8 points when the early deaths fall from 10 to 5. A result that moves with the early event rate is measuring the design, not the treatment.

**Check 6. The protocol and the emulation agree on the contrast.** The protocol specifies ITT for the offer. The naive analysis is not ITT, because it classifies patients by what they did rather than by the strategy assigned at time zero. The landmark analysis is closer to ITT: it assigns the strategy at day 10 and follows both groups from that point. A reviewer can see the difference from the protocol table.

**Check 7. The landmark population is described.** The landmark analysis excludes the 10 patients who died before day 10. That exclusion is legitimate because those patients could not be classified at the landmark, but it changes the population: the estimate applies to patients who survive to day 10, not to all discharged patients. The report must state this. A target trial that randomized at discharge would include those deaths, and the ITT estimate from such a trial would not be zero if the program affected early mortality. The landmark estimate answers a different question from the trial, which is why the protocol mapping matters.

### 6.6b Summary of the three analyses

| Analysis | Population at start | Deaths included | RD (95% CI) | What it estimates |
|---|---|---:|---|---|
| Naive | 100 at discharge | 28 | -16.0 (-33.3, +1.3) | A mixture of program effect and immortal time |
| Landmark | 90 at day 10 | 18 | 0.0 (-16.6, +16.6) | Effect among survivors to day 10 |
| Crude rate | Person-time | 28 | RR 0.45 | A rate ratio confounded by the early hazard |

The table is the report in miniature. The naive and landmark rows differ in the population and the deaths included, and the difference is the bias.

```text
Immortal person-time check:
  treated person-days before enrollment = 50 × 10 = 500
  deaths during those days = 0 (by construction)
  untreated deaths during the same 10 days = 10
  risk difference attributable to the early period = 10/50 = 20 points
  observed naive difference = 16 points (the remaining 4 points are the
  difference between 18/50 and 8/40 in the post-period)
```

The decomposition shows that the naive estimate is mostly the early period, not the treatment.

### 6.7 Vary one assumption: five early deaths instead of ten

If only five non-enrollees die before day 10, the naive and landmark analyses become:

```text
Naive: program 10/50 = 20.0%; no program (5 + 9)/50 = 28.0%; RD = -8.0 points
Landmark: program 10/50 = 20.0%; no program 9/45 = 20.0%; RD = 0.0 points
```

The bias falls from 16 points to 8 points, exactly in proportion to the number of early deaths. The naive estimate is still biased, and the landmark estimate is still zero. The magnitude of the bias is a function of the early event rate and the length of the grace period, not of the treatment effect.

This is the material variation: changing the early mortality changes the naive estimate and leaves the correct estimate unchanged. A result that moves when the grace period or the early event rate changes is a result driven by the design, not by the treatment.

### 6.8 What to report

Report the landmark estimate as the primary result: **0.0 percentage points (95% CI -16.6 to +16.6)**, with the protocol and the discrepancy between the protocol and the emulation. Report the naive estimate (-16.0 points) and explain that it is an artifact of immortal time. Report the time-zero audit, including the 500 pre-enrollment person-days.

The estimate to report is the landmark estimate because it aligns time zero with the moment enrollment status is known. The naive estimate is not a version of the treatment effect; it is a comparison that includes a period during which the program group could not die.

The honest conclusion: "In this emulation, the program was not associated with a difference in 100-day mortality after aligning time zero at day 10 (RD 0.0 percentage points, 95% CI -16.6 to +16.6). A naive analysis that classified patients by enrollment and followed them from discharge suggested a 16-point benefit, but that estimate counts 500 pre-enrollment person-days as program time and attributes 10 early deaths to the no-program group. The confidence interval is wide, and the study is underpowered for the decision."

Three limitations belong with that conclusion. First, the cohort is small: 100 patients and 18 deaths cannot rule out an effect of 10 percentage points in either direction. Second, the post-day-10 mortality of 20% is assumed equal in both groups; in a real dataset it would be estimated, and the comparison would need adjustment for age, comorbidity, and ejection fraction. Third, the emulation assumes that enrollment is not confounded by unmeasured prognosis. Patients who enroll may be more engaged, and engagement is associated with survival. The landmark estimate removes the immortal time but not that confounding.

The decision implication is not "the program does not work." It is "this analysis cannot distinguish a meaningful benefit from no effect, and the naive estimate that suggested a benefit is an artifact." A larger cohort with a pre-specified protocol, a time-varying exposure, and adjustment for engagement would be required to answer the question.

## 7. Code, query, or calculation

The script reproduces the naive, landmark, and person-time analyses for the worked example, and repeats them for the five-early-death scenario. It uses only the standard library.

**Assumptions:**

- The cohort is 100 patients, 50 treated at day 10 and 50 untreated.
- Post-day-10 mortality is 20% in both groups.
- Early deaths occur at a mean of day 5 for person-time purposes.
- The landmark analysis starts at day 10 for both groups.
- The person-time rates are crude and are not adjusted for covariates.

```python
import math
from statistics import NormalDist

z = NormalDist().inv_cdf(0.975)

def rd_ci(e1, n1, e0, n0):
    p1, p0 = e1 / n1, e0 / n0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return p1, p0, p1 - p0, se, (p1 - p0) - z * se, (p1 - p0) + z * se

def scenario(early_deaths, horizon=90, n_t=50, n_c=50, treat_day=10):
    post_c = round(0.2 * (n_c - early_deaths))
    print(f"early deaths={early_deaths}")
    p1, p0, d, se, lo, hi = rd_ci(10, n_t, early_deaths + post_c, n_c)
    print(f"  naive: treated {p1:.4f} control {p0:.4f} RD={d:.4f} "
          f"95% CI=({lo:.4f}, {hi:.4f})")
    p1l, p0l, dl, sel, lol, hil = rd_ci(10, n_t, post_c, n_c - early_deaths)
    print(f"  landmark day {treat_day}: treated {p1l:.4f} control {p0l:.4f} RD={dl:.4f} "
          f"95% CI=({lol:.4f}, {hil:.4f})")
    # person-time rates per 1,000 person-days
    pt_t = n_t * horizon
    pt_c = (n_c - early_deaths) * horizon + early_deaths * 5
    rate_t = 1000 * 10 / pt_t
    rate_c = 1000 * (early_deaths + post_c) / pt_c
    print(f"  rates per 1,000 person-days: treated {rate_t:.4f} control {rate_c:.4f} "
          f"RR={rate_t/rate_c:.4f}")
    print(f"  immortal person-days in treated arm: {n_t * treat_day}")

scenario(10)
scenario(5)
```

**Expected output:**

```text
early deaths=10
  naive: treated 0.2000 control 0.3600 RD=-0.1600 95% CI=(-0.3332, 0.0132)
  landmark day 10: treated 0.2000 control 0.2000 RD=0.0000 95% CI=(-0.1663, 0.1663)
  rates per 1,000 person-days: treated 2.2222 control 4.9315 RR=0.4506
  immortal person-days in treated arm: 500
early deaths=5
  naive: treated 0.2000 control 0.2800 RD=-0.0800 95% CI=(-0.2467, 0.0867)
  landmark day 10: treated 0.2000 control 0.2000 RD=0.0000 95% CI=(-0.1611, 0.1611)
  rates per 1,000 person-days: treated 2.2222 control 3.4356 RR=0.6468
  immortal person-days in treated arm: 500
```

**What the output shows:**

- The naive risk difference is -16.0 points with 10 early deaths and -8.0 points with 5. The bias is proportional to the number of early deaths, which is the immortal-time mechanism in one line.
- The landmark risk difference is zero in both scenarios. The correct analysis is invariant to the early event rate because it starts after the period in which enrollment is determined.
- The crude person-time rate ratio is 0.45 and 0.65, both suggesting benefit. This is not evidence of a treatment effect: the unexposed person-time includes the early high-hazard period, when patients were recently discharged and at higher risk. A time-varying Cox model with a baseline hazard, or a landmark analysis, is needed to separate the early hazard from the treatment effect.
- The immortal person-days line is the direct audit: 500 person-days are attributed to the treated arm before any treatment could start.

> **Note:** The crude rate ratio is included to show that person-time alone does not fix immortal time. The fix is the alignment of time zero, not the choice of a rate denominator.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| International Council for Harmonisation (ICH) E9 (1998) | Binding through regional adoption | Pre-specified analysis and analysis populations | Target trial emulation |
| ICH E9(R1) (2019) | Binding through regional adoption | The estimand, intercurrent events, sensitivity analyses | The design of an observational emulation |
| FDA RWE guidance (August 2023) | FDA guidance | Data reliability, pre-specified protocol, transparency | The specific emulation method |
| TARGET statement (JAMA, September 2025) | Reporting guideline, not binding | 21 items across 6 sections for emulation studies | The analysis itself |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) and REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) | Journal editorial policy | Reporting of observational studies and routinely collected data | Emulation-specific items |
| Consolidated Standards of Reporting Trials (CONSORT) 2010 | Journal editorial policy | Reporting of the randomized trial that would be the benchmark | Emulation studies |

### 8.2 What the TARGET statement requires

The TARGET statement was published in JAMA in September 2025, with a simultaneous publication in the BMJ. It provides a 21-item checklist across six sections: abstract, introduction, methods, results, discussion, and other information. The methods section has paired items for the target trial specification (6a to 6h) and its emulation (7a to 7h), covering eligibility, treatment strategies, assignment, time zero, outcomes, follow-up, causal contrast, and analysis plan.

The checklist also asks for the identifying assumptions, the statistical methods, the effect estimate with precision, and the sensitivity analyses. It is designed for observational studies that explicitly aim to emulate a parallel-group, individually randomized target trial with baseline confounding adjustment. It is not intended for instrumental-variable designs or other advanced identification strategies.

The checklist is a reporting standard, not a regulatory requirement. Its value is that it forces the protocol and the emulation to be reported side by side, which is the only way a reader can see where the emulation departs from the benchmark.

### 8.3 What the FDA RWE guidance adds

The August 2023 final guidance emphasizes the reliability and relevance of the data source, the pre-specification of the protocol and statistical analysis plan, and transparency about the study conduct. A target trial emulation is a natural way to satisfy the pre-specification requirement, because the protocol is written before the analysis and the data mapping is explicit.

The guidance does not require a target trial emulation, and it does not endorse any specific estimator. It requires that the study be designed and reported in a way that a reviewer can evaluate. The TARGET checklist is one way to meet that expectation.

### 8.4 The TARGET methods items in practice

| Item | What it asks | What a weak report does |
|---|---|---|
| 6a / 7a | Eligibility criteria and their emulation | Lists codes without stating the assessment date |
| 6b / 7b | Treatment strategies and their emulation | Defines exposure as "ever treated" |
| 6c / 7c | Assignment procedure and its emulation | Skips the assignment element |
| 6d / 7d | Time zero and its emulation | Leaves time zero implicit |
| 6e / 7e | Outcomes and their ascertainment | Uses a code set without validation |
| 6f / 7f | Follow-up and censoring | Does not report disenrollment |
| 6g / 7g | Causal contrast | Reports a hazard ratio without naming the contrast |
| 6h / 7h | Analysis plan | Reports the estimator but not the estimand |

The paired structure is the point. Each protocol element has an emulation element, and the reader can compare them.

### 8.5 An example of the paired reporting format

| Target trial | Emulation | Discrepancy |
|---|---|---|
| Randomization at discharge | None; assignment is observed | Confounding by indication remains |
| Program offered immediately | Enrollment within 10 days | Grace period of 10 days |
| Follow-up from discharge | Follow-up from discharge | None |
| Death within 100 days | Death within 100 days | Ascertainment assumed complete |
| ITT | ITT approximation with a landmark | Early deaths excluded at the landmark |

A table like this does more for credibility than a paragraph of methods. It shows the reviewer exactly where the emulation departs from the benchmark.

### 8.6 Data-source constraints on the protocol

| Constraint | Effect on the protocol | What to report | Workaround |
|---|---|---|---|
| No exact discharge date | Time zero must be approximated | The approximation rule | Use the first post-discharge claim |
| Enrollment underrecorded | Strategy misclassified | Validation rate | Combine with billing codes |
| No death linkage | Outcome incomplete | Linkage coverage | Use a registry or vital records |
| Disenrollment common | Informative censoring | Disenrollment rate by arm | IPCW; sensitivity |
| No clinical covariates | Confounding control limited | The missing variables | E-value; negative control |
| Claims lag | Outcome ascertainment delayed | The lag distribution | Restrict to a mature period |
| Coding changes | Outcome definition inconsistent | The code set by year | Validate by period |

Each row is a place where the protocol and the emulation diverge. The TARGET checklist asks for these to be reported, not resolved.

### 8.7 The regulatory pathway for an emulation

An emulation submitted as real-world evidence typically follows four steps:

1. A pre-submission meeting to discuss the question, the data source, and the design.
2. A protocol and statistical analysis plan, written before the analysis.
3. A study report with the protocol-to-data mapping and the sensitivity analyses.
4. A review that focuses on the data reliability, the confounding control, and the estimand.

The FDA RWE guidance describes the expectations for each step without prescribing the method. The TARGET checklist supplies the reporting structure.

A pre-submission meeting is the place to resolve three questions:

- Is the data source acceptable for the outcome and the confounders?
- Is the proposed time zero implementable?
- What sensitivity analyses would the review division expect?

The answers shape the protocol before the analysis begins, which is the point of the process.

### 8.8 What the standards do not cover

- They do not require a target trial protocol for every observational study.
- They do not specify the grace period or the time-zero rule.
- They do not resolve the choice between ITT and per-protocol.
- They do not validate the no-unmeasured-confounding assumption.
- They do not require the negative control or the E-value, although reviewers increasingly ask for both.

## 9. Industry practice and case evidence

### 9.1 Documented case: RCT-DUPLICATE

The RCT-DUPLICATE initiative emulated 32 randomized trials using claims data, with protocols registered on ClinicalTrials.gov before analysis. The reported correlation between emulated and randomized effect estimates was 0.82 overall, and 0.93 in the 16 trials with close emulation. The emulations used new-user cohorts, propensity score matching, and Cox models.

The design elements that carried the result were the eligibility criteria, the time zero, the treatment strategies, and the outcome definitions. The estimator mattered less than the alignment. This is the empirical case for the target trial framework: the emulations that matched the trial protocol closely reproduced the trial result, and the ones that did not did not.

**Inference:** the correlation is an average over trials and databases. It does not establish that any particular emulation is valid, and it does not remove the need for a sensitivity analysis.

### 9.2 Documented evidence: reporting quality

Hansford and colleagues, in a systematic review published in JAMA Network Open in 2023, examined 200 observational studies that explicitly aimed to emulate a target trial. They reported that 58% completely reported how the target trial protocol was emulated, 44% described both the protocol and the emulation, 22% cited any reporting guideline, 10% were prospectively registered, and 37% reported potential unmeasured confounders.

The review is the evidence that the TARGET statement was needed. The gap is not in the methods, which are widely described in the literature; it is in the reporting, which often omits the protocol and the assumptions.

### 9.3 Documented case: immortal time in pharmacoepidemiology

Suissa's 2008 paper in the American Journal of Epidemiology documents five cohort designs that produce immortal time bias, with a worked example in a lung cancer cohort. The exposure-based design changed a hazard ratio of 0.73 to 1.05 after correction; the multiple-event design changed 0.48 to 0.91. The paper is the empirical demonstration that the bias can be large enough to reverse a conclusion.

The target trial framework generalizes the lesson: the bias is a design defect, and the remedy is to align time zero, not to add covariates.

### 9.4 Practice pattern: the protocol as a checklist

Mature analytics teams write a target trial protocol before the analysis, even when no regulator requires it. The protocol is one page, and it is attached to the analysis plan. The review meeting uses the protocol to ask whether the data can support each element, and the results are reported against the protocol.

**Inference:** the practice is recommended in the methodological literature and visible in some regulatory submissions, but it is not universal. The TARGET review's 44% figure for complete protocol-and-emulation reporting suggests that many studies still skip the step.

### 9.5 Documented fact versus inference

| Claim | Status | Basis |
|---|---|---|
| Misaligning time zero can produce large bias | Documented | Hernán et al. 2016; Suissa 2008 |
| Closely emulated trials can reproduce randomized results | Documented with variation | RCT-DUPLICATE correlation |
| Emulation reporting is incomplete | Documented | Hansford et al. 2023 review |
| A target trial protocol is sufficient for identification | Not supported | The no-unmeasured-confounding assumption remains |
| An emulation validates the observational data source | Not supported | It validates a design, not the data |

The first three rows are the evidence base. The last two are the claims that the framework is often misread to support. The protocol improves the design; it does not make the data randomized.

### 9.6 The cost of skipping the protocol

An emulation without a protocol fails in predictable ways:

- The time zero is chosen to maximize the sample, not to match the trial.
- The treatment definition is chosen to maximize the number of treated patients.
- The outcome definition is chosen to match the available codes.
- The sensitivity analyses are chosen after the primary result.

Each choice is defensible in isolation and indefensible in combination, because the reader cannot tell which choices were driven by the question and which by the result. The cost is a study that cannot be replicated or reviewed.

**Inference:** the specific cost depends on the decision. The reporting review in Section 9.2 documents the frequency of the gaps.

### 9.7 What to ask a vendor who claims an emulation

1. Can you share the target trial protocol?
2. What is the time-zero rule, and is it the same for both strategies?
3. How did you handle patients who started treatment after time zero?
4. What is the causal contrast, and why?
5. What are the positivity and balance diagnostics?
6. What sensitivity analyses did you run?

A vendor who can answer these has a study. A vendor who cannot has a dataset.

### 9.8 Practice patterns across sectors

| Sector | Typical emulation question | Common design | Characteristic weakness |
|---|---|---|---|
| Pharma | Comparative safety of two drugs | New-user cohort, PS matching | Outcome validation |
| Payer | Coverage of a program | ITT with a grace period | Disenrollment |
| Health system | Quality program effect | Difference-in-differences or emulation | Calendar trends |
| Device | Long-term revision risk | Registry cohort, competing risks | Loss to follow-up |
| Regulatory | Confirmatory RWE | Pre-specified protocol, IPTW | Unmeasured confounding |

The table is an inference from documented practice, not a census. The common thread is that each sector's weakness is a property of its data source, and the target trial protocol is the tool that makes the weakness visible.

### 9.9 What mature teams do differently

1. They write the protocol before the data analysis.
2. They align time zero for both strategies.
3. They use a new-user design with a washout period.
4. They pre-specify the grace period and test its sensitivity.
5. They report the protocol-to-data mapping and the gaps.
6. They report the ITT and per-protocol contrasts when both are relevant.
7. They run a time-zero audit and a negative control.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Immortal time | Treated group must survive to be classified | Pre-treatment time counted as exposed | Spurious benefit | Align time zero; landmark; cloning |
| Prevalent users | Patients already on treatment at entry | Selection on prior survival | Biased comparison | Washout; new-user design |
| Time zero differs by arm | Treated start at first fill, controls at diagnosis | Risk sets not comparable | Bias | Same rule for both |
| Eligibility uses future data | A lab after time zero determines entry | Selection on the future | Biased population | Assess at time zero |
| Grace period chosen post hoc | The window maximizes the effect | Design depends on the result | Overstated effect | Pre-specify; sensitivity |
| Informative censoring | Disenrollment differs by arm | The risk set is selected | Bias | IPCW; sensitivity |
| Outcome misclassification | Treated patients have more follow-up | Detection bias | Biased effect | Same ascertainment; validation |
| No protocol | The analysis is described only by methods | The design is not specified | Unreviewable | Write the protocol |
| Protocol and emulation conflated | The report does not distinguish them | The reader cannot see the gaps | Overconfidence | Report side by side |
| As-treated contrast presented as ITT | The label does not match the analysis | Time-varying confounding | Biased effect | Label the contrast correctly |
| No positivity assessment | Some subgroups never receive one strategy | Extrapolation | Unsupported estimate | Restrict; report |
| Calendar trend ignored | Enrollment spans a practice change | Confounding by time | Biased effect | Stratify by period |
| No sensitivity analysis | Only one estimate is reported | The assumptions are untested | Overconfidence | Grace period, censoring, confounding |
| Landmark without stating the population | The estimate is applied to everyone | Survivors to the landmark are selected | Wrong target | State the landmark population |
| Cloning weights unreported | The per-protocol estimate looks precise | Censoring weights are ignored | Too narrow interval | Report weight distribution and ESS |
| Protocol written after analysis | The protocol matches the result | Reverse engineering | Untrustworthy | Lock the protocol |
| Outcome horizon chosen post hoc | The horizon with the largest effect | Selection on the result | Overstated effect | Pre-specify the horizon |
| Strategy defined by adherence | Only adherers are "treated" | Selection on post-baseline behaviour | Bias | Define the strategy at time zero |
| No report of the target population | The estimate is generalized to everyone | The emulation population differs from the trial | Wrong decision | State the population |
| No report of the data lag | The outcome is incomplete at the end | Ascertainment bias | Underestimated events | Restrict to mature data |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Time zero aligned | The risk sets must start together | Different rules by arm | Time-zero audit |
| No immortal time | Pre-treatment time must not count as exposed | Classification requires survival | Person-time audit |
| No prevalent users | The cohort is incident | Prior treatment at entry | Washout; prior prescription check |
| Consistency | The strategy corresponds to a defined intervention | Multiple versions of the program | Strategy definition review |
| Positivity | Both strategies are possible for every patient type | Structural zeros | Cross-tabulation |
| No unmeasured confounding | The adjustment set identifies the effect | An unmeasured common cause | E-value; negative control |
| Correct outcome measurement | The endpoint is the same in both arms | Differential ascertainment | Validation sample |
| Non-informative censoring | The risk set remains representative | Disenrollment related to prognosis | IPCW; sensitivity |
| Correct grace period | The strategy is defined at time zero | The window is arbitrary | Sensitivity over windows |
| Correct model specification | The estimator targets the effect | Nonlinearity or interaction | Alternative specifications |
| Stable calendar period | The standard of care is common | Practice change during enrollment | Stratify by period |
| Correct landmark population | The estimate applies to a defined group | Survivors to the landmark differ | Compare with the full cohort |
| Correct cloning weights | The per-protocol estimate is unbiased | The censoring model is wrong | Weight diagnostics; sensitivity |
| Complete outcome ascertainment | The events are counted | Claims lag | Restrict to mature data |
| Correct protocol-to-data mapping | The emulation matches the benchmark | A gap is unreported | Review the mapping table |
| No selection on adherence | The strategy is defined at time zero | Adherers only | Define at time zero |
| Reported population matches the estimate | The decision uses the right population | Generalizing from a landmark | State the population |

**Limits of this topic.** A target trial emulation does not randomize. The no-unmeasured-confounding assumption remains, and the E-value or a negative control is the only check. The protocol is a benchmark, not a guarantee: a poorly specified protocol produces a well-aligned analysis of the wrong question. The worked example is a 100-patient teaching dataset; the confidence intervals are wide and the estimates are illustrative. The cloning-censoring-weighting method requires time-varying data and a correct censoring model, and it is not implemented in the script.

## 12. Exercises

### Exercise 1. Write a target trial protocol (design, 60 minutes)

**Scenario.** A payer wants to compare two diabetes drugs for cardiovascular hospitalization over 3 years. The data are claims, and the drugs are initiated at different times after diagnosis.

**Deliverable.** A seven-component protocol plus the analysis plan.

**Acceptance criteria.**

- Every component is specific enough to implement.
- Time zero is a single rule applied to both strategies.
- Treatment strategies specify initiation and adherence.
- The causal contrast is named and justified.
- The analysis plan names the confounder adjustment.

### Exercise 2. Align time zero (application, 50 minutes)

**Scenario.** A cohort defines treatment as "statin initiated within 90 days of discharge." Patients who die before day 90 cannot be treated. The analysis follows everyone from discharge.

**Deliverable.** A timeline, a description of the immortal time, a corrected design with a landmark at day 90 or a time-varying exposure, and the population each design estimates.

**Acceptance criteria.**

- The immortal period is quantified.
- The corrected design has the same time zero for both strategies.
- The population change is stated.
- The answer explains why adjusting for baseline covariates does not fix the bias.
- The corrected design is compared with the naive design in a table.
- The population change is quantified by the number of patients excluded.
- The answer states which estimate would be reported and why.

### Exercise 3. Compute the naive and landmark estimates (application, 45 minutes)

**Data.** 200 patients; 100 enroll on day 14; 100 do not. Twelve non-enrollees die before day 14. After day 14, mortality is 15% in both groups.

**Deliverable.** The naive risks, the landmark risks, the risk differences, and the number of immortal person-days.

**Acceptance criteria.**

- The naive analysis includes the early deaths in the non-enrolled arm.
- The landmark analysis excludes them and changes the denominator.
- The immortal person-days are 100 × 14.
- The answer explains the direction of the bias.

### Exercise 4. Choose the contrast (design, 40 minutes)

**Scenario.** A drug is offered to patients at discharge, but 30% stop within 6 months because of side effects. The decision is whether to add the drug to a formulary.

**Deliverable.** A recommendation between ITT and per-protocol, with the estimand, the analysis, and the interpretation.

**Acceptance criteria.**

- The ITT is linked to the offer decision.
- The per-protocol is linked to the adherence question.
- The recommended primary contrast is justified.
- The answer states what the per-protocol analysis requires.

### Exercise 5. Design a cloning-censoring-weighting analysis (design, 70 minutes)

**Scenario.** A time-varying treatment: patients can start a drug at any time after diagnosis, and the decision to start depends on a time-varying biomarker.

**Deliverable.** A design with the cloning rule, the censoring rule, the weight model, the diagnostics, and the estimand.

**Acceptance criteria.**

- Clones are created at time zero.
- Censoring is at deviation, not at the end of follow-up.
- The weight model includes the time-varying predictors.
- The weight distribution and ESS are pre-specified.
- The estimand is per-protocol.

### Exercise 6. Critique an emulation report (critique, 60 minutes)

**Claim.** "We emulated a target trial. Treatment was defined as at least two prescriptions in the first year. Follow-up started at diagnosis. The hazard ratio was 0.72 (p = 0.01)."

**Deliverable.** A red-flag list with at least six items and a corrected protocol.

**Acceptance criteria.**

- The "at least two prescriptions" definition is identified as immortal time.
- The time-zero misalignment is identified.
- The missing protocol elements are listed.
- The corrected protocol aligns time zero and defines initiation.
- The revised claim reports the contrast and the interval.

### Exercise 7. Audit against TARGET (critique, 50 minutes)

Use a published emulation study or a constructed report.

**Deliverable.** A table with the TARGET items, the reported content, the missing content, and the consequence.

**Acceptance criteria.**

- At least 12 items are assessed.
- The protocol and emulation items are paired.
- The consequences are stated in decision terms.
- The audit distinguishes missing reporting from a flawed design.

### Exercise 8. Design a full emulation (design, 120 minutes)

**Scenario.** A health system wants to compare two care models for chronic kidney disease using 5 years of EHR data.

**Deliverable.** A protocol, a data mapping table, an analysis plan, a time-zero audit, and a sensitivity analysis plan.

**Acceptance criteria.**

- The protocol has all seven components.
- The mapping table names the gaps.
- The estimator matches the contrast.
- The audit includes person-time and the number at risk.
- At least three sensitivity analyses are specified.
- The report identifies the contrast and the target population.
- The time-zero audit is reproducible from the protocol.
- The gaps between protocol and data are stated in a table.
- The recommendation states what evidence would change the decision.

## 13. Self-check answers

**Answer 1 (Exercise 3 arithmetic).**

```text
Naive:
  Program: 15 deaths / 100 = 15.0%
  No program: 12 early + 0.15 × 88 = 12 + 13.2 ≈ 25 deaths / 100 = 25.0%
  RD = -10.0 percentage points

Landmark at day 14:
  Program: 15 / 100 = 15.0%
  No program: 13.2 / 88 = 15.0%
  RD = 0.0 percentage points

Immortal person-days = 100 × 14 = 1,400
```

The naive estimate shows a 10-point benefit that is entirely the 12 early deaths. The landmark estimate is zero. The immortal person-days are the audit quantity: 1,400 days counted as program time before any program could start.

**Answer 2 (Exercise 2 correction).**

The immortal period is the time between discharge and day 90 for patients who eventually start. A patient who dies on day 40 cannot be classified as treated, so the treated group is guaranteed to survive 90 days. The corrected design uses a landmark at day 90: classify treatment status at day 90 and start follow-up then for both groups.

| Design | Population | Time zero | Contrast | Bias |
|---|---|---|---|---|
| Naive | Everyone discharged | Discharge | Ever treated vs not | Immortal time |
| Landmark | Survivors to day 90 | Day 90 | Treated by day 90 vs not | None from immortal time; population restricted |
| Time-varying | Everyone discharged | Discharge | Exposed vs unexposed person-time | Requires a time-varying model |
 The population becomes patients who survive to day 90. Alternatively, a time-varying exposure counts pre-initiation time as unexposed and post-initiation time as exposed, preserving the full cohort. The landmark is simpler and changes the population; the time-varying analysis preserves the population but requires a model for the time-varying hazard. Neither is fixed by adjusting for baseline covariates, because the bias is in the person-time, not in the covariate distribution.

**Answer 3 (Exercise 4 contrast).**

The formulary decision is about the offer, so the ITT is the primary contrast: it estimates the effect of making the drug available, including the effect of discontinuation. The per-protocol contrast estimates the effect if everyone adhered, which is relevant to the question "how much of the benefit is lost to side effects?" but not to the coverage decision. Report the ITT as primary and the per-protocol as secondary, with the censoring weights and their diagnostics. If the per-protocol effect is much larger, the difference is the cost of discontinuation, which is a finding in itself.

**Answer 4 (Exercise 6 red flags).**

A strong list:

- "At least two prescriptions in the first year" requires survival to the second prescription, which is immortal time.
- Follow-up starts at diagnosis but treatment is defined over the first year, so the treated group's pre-treatment time is counted as exposed.
- The target trial protocol is not reported.
- Time zero is not aligned across strategies.
- The hazard ratio is not accompanied by the cumulative incidence.
- The p-value is reported without the interval.
- The contrast (ITT or per-protocol) is not named.
- Unmeasured confounding is not discussed.

The corrected protocol defines initiation at time zero (or uses a grace period), aligns follow-up, and reports the contrast.

**Answer 5 (Exercise 7 rubric).**

A strong audit:

- Checks whether the study is identified as an emulation in the title or abstract.
- Checks whether the protocol is reported component by component.
- Checks whether each protocol element has a paired emulation element.
- Checks the time zero and the treatment definition for immortal time.
- Checks the confounder adjustment and the positivity assessment.
- Checks the effect estimate and the sensitivity analyses.
- States the consequence of each missing item.

**Answer 6 (Exercise 5 design).**

A strong design:

- Clones each patient at diagnosis into "start now" and "do not start" strategies.
- Censors a clone at the first deviation from its strategy.
- Fits a time-varying censoring model with the biomarker, age, comorbidity, and prior treatment.
- Weights the remaining clones by the inverse probability of remaining uncensored.
- Reports the weight distribution, the ESS, and a sensitivity analysis with a different censoring model.
- States the estimand as the per-protocol effect of starting versus not starting.

**Answer 7 (Exercise 1 rubric).**

A strong protocol:

- Defines eligibility at initiation, not at diagnosis.
- Defines the strategies as "initiate drug A within 30 days" vs "initiate drug B within 30 days."
- Sets time zero at the first eligible prescription for both strategies.
- Defines the outcome as cardiovascular hospitalization within 3 years, with death as a competing event.
- Names the ITT contrast for the formulary decision.
- Specifies IPTW with a DAG-based adjustment set and a robust variance.

**Answer 8 (Exercise 8 rubric).**

A strong emulation:

- Writes the protocol before the analysis.
- Maps each element to EHR data and names the gaps.
- Aligns time zero at the first eligible encounter.
- Uses a new-user design with a washout period.
- Reports the time-zero audit and the positivity assessment.
- Reports the ITT and a per-protocol sensitivity analysis.
- Includes an E-value and a negative control outcome.

## 14. Interview questions

**1. What is a target trial emulation?**

A strong answer describes the hypothetical randomized trial and the mapping to observational data, and says the protocol is the benchmark. A weak answer calls it "an observational study that mimics a trial" without the components.

**2. Why is time zero the most important design choice?**

A strong answer explains that the risk sets must start together, and that misalignment produces immortal time or selection bias. A weak answer says it is "when follow-up starts" without the consequence.

**3. What is immortal time bias?**

A strong answer defines the period during which the outcome cannot occur and explains that counting it as exposed time produces a spurious benefit. A weak answer says it is "a bias from time."

**4. How do you choose between ITT and per-protocol?**

A strong answer ties the contrast to the decision: ITT for an offer, per-protocol for adherence. It notes the additional assumptions of per-protocol. A weak answer says per-protocol is "more accurate."

**5. What is a grace period, and how do you choose it?**

A strong answer defines the window for initiation and says it should be clinically justified and tested for sensitivity. A weak answer says "the time allowed to start treatment."

**6. What is cloning-censoring-weighting?**

A strong answer describes cloning at time zero, censoring at deviation, and weighting by the inverse probability of censoring. A weak answer confuses it with matching.

**7. What is a prevalent user, and why is the design problematic?**

A strong answer explains that prevalent users have already survived on treatment and are selected for health, and that a new-user design with a washout is the remedy. A weak answer says "patients already on the drug."

**8. How do you know the emulation is valid?**

A strong answer says you do not; you assess the alignment, the positivity, the confounding, and the sensitivity analyses, and you compare with a trial if one exists. A weak answer says "if the estimate is plausible."

**9. What does the TARGET statement require?**

A strong answer describes the paired protocol and emulation items across the checklist. A weak answer says "reporting guidelines."

**10. What can an emulation not fix?**

A strong answer lists unmeasured confounding, poor measurement, no overlap, and an ill-defined treatment. A weak answer says "nothing."

**11. How would you handle treatment switching?**

A strong answer uses a per-protocol analysis with censoring at switch and IPCW, or a clone-censor-weight design. A weak answer ignores the switch.

A follow-up is "what if switching is informative?" The strong answer says the censoring weights model the probability of remaining on the strategy given time-varying covariates, and a sensitivity analysis varies that model. If switching depends on unmeasured prognosis, the per-protocol estimate is biased and the ITT estimate is the safer primary.

**12. What would make you distrust an emulation?**

A strong answer lists no protocol, misaligned time zero, prevalent users, no positivity, no sensitivity, and an unnamed contrast. A weak answer says "a small sample."

## 15. Key terms

- **As-treated contrast:** comparison by treatment actually received.
- **Assignment:** how treatment would be assigned in the trial.
- **Causal contrast:** the comparison the emulation targets.
- **Cloning:** creating one copy of each patient per strategy.
- **Censoring at deviation:** removing a clone that does not follow its strategy.
- **Eligibility:** the criteria for entry, assessed at time zero.
- **Emulation:** the observational analysis designed to approximate a target trial.
- **Grace period:** the window after time zero during which initiation is allowed.
- **Immortal time:** follow-up during which the outcome cannot occur.
- **Informative censoring:** censoring related to prognosis.
- **Intention-to-treat:** comparison by assigned strategy.
- **Inverse probability of censoring weighting:** weights that adjust for informative censoring.
- **Landmark analysis:** an analysis starting at a fixed time after baseline.
- **Left truncation:** entry after the start of follow-up.
- **New-user design:** a cohort of patients initiating treatment at time zero.
- **Per-protocol:** comparison as if the strategy were followed.
- **Positivity:** both strategies are possible for every patient type.
- **Prevalent user:** a patient already on treatment at cohort entry.
- **Protocol:** the specification of the target trial.
- **Target trial:** the hypothetical randomized trial that would answer the question.
- **TARGET statement:** a reporting checklist for emulation studies.
- **Time zero:** the start of follow-up for each strategy.
- **Treatment strategy:** the intervention being compared, including initiation and adherence.
- **Washout period:** a pre-enrollment window used to identify new users.
- **Dynamic strategy:** a strategy that depends on time-varying information.
- **Clone-censor-weight:** the three-step implementation of a per-protocol contrast.
- **Data mapping:** the translation of protocol elements into data operations.
- **Protocol gap:** a difference between the trial and the emulation.
- **Negative control:** an outcome or exposure used to test for residual bias.
- **Estimand:** the quantity the emulation targets.
- **Protocol-to-data mapping:** the table that translates each trial element into a data operation.
- **Eligible person-time:** the time during which a patient could have been enrolled.
- **Outcome horizon:** the period over which the outcome is measured.
- **Data lag:** the delay between an event and its appearance in the data.
- **Strategy deviation:** a departure from the assigned treatment strategy.
- **Reference trial:** the randomized trial used as a benchmark for the emulation.
- **Causal question:** the population, strategies, outcome, and horizon the study addresses.
- **Design element:** one of the seven components of the target trial protocol.
- **Emulation gap:** a difference between the protocol element and the data implementation.
- **Pre-submission meeting:** a regulatory discussion before the protocol is finalized.
- **Incident user:** a patient starting treatment at time zero.
- **Exposure definition:** the rule that classifies patients as treated or untreated.
- **Benchmark trial:** the trial against which an emulation is compared.
- **Design alignment:** the correspondence between protocol and data operations.
- **Follow-up window:** the period over which events are counted.
- **Causal diagram:** the graph of assumptions used to choose confounders.
- **Sensitivity analysis:** an analysis under an alternative design assumption.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. Hernán, M. A., Robins, J. M. "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology* 2016;183(8):758-764. https://doi.org/10.1093/aje/kwv254
2. Hernán, M. A., Sauer, B. C., Hernández-Díaz, S., Platt, R., Shrier, I. "Specifying a target trial prevents immortal time bias and other self-inflicted injuries in observational analyses." *Journal of Clinical Epidemiology* 2016;79:70-75. https://doi.org/10.1016/j.jclinepi.2016.04.014
3. Hernán, M. A., Wang, W., Leaf, D. E. "Target Trial Emulation: A Framework for Causal Inference From Observational Data." *JAMA*. Published online 12 December 2022. https://doi.org/10.1001/jama.2022.21383
4. Swanson, S. A., Lee, H., Jones, M. D., et al. "Transparent Reporting of Observational Studies Emulating a Target Trial—The TARGET Statement." *JAMA*. Published online 3 September 2025. https://doi.org/10.1001/jama.2025.13350
5. Hansford, H. J., Cashin, A. G., Jones, M. D., et al. "Reporting of Observational Studies Explicitly Aiming to Emulate Randomized Trials: A Systematic Review." *JAMA Network Open* 2023;6(9):e2336023. https://doi.org/10.1001/jamanetworkopen.2023.36023
6. Wang, S. V., Schneeweiss, S., and the RCT-DUPLICATE Initiative. "Emulation of Randomized Clinical Trials With Nonrandomized Database Analyses: Results of 32 Clinical Trials." *JAMA* 2023;329(16):1376-1385. https://doi.org/10.1001/jama.2023.4221
7. Suissa, S. "Immortal Time Bias in Pharmacoepidemiology." *American Journal of Epidemiology* 2008;167(4):492-499. https://doi.org/10.1093/aje/kwm324
8. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
9. United States Food and Drug Administration. *Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products*. August 2023. https://www.fda.gov/media/171667/download
10. Benchimol, E. I., Smeeth, L., Guttmann, A., et al. "The RECORD Statement." *PLoS Medicine* 2015;12(10):e1001885. https://doi.org/10.1371/journal.pmed.1001885
11. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
12. Hernán, M. A., Robins, J. M. *Causal Inference: What If*. Chapman and Hall/CRC, 2020.
13. Matthews, A. A., Danaei, G., Islam, N., Kurth, T. "Target trial emulation: applying principles of randomised trials to observational studies." *BMJ* 2022;378:e071108.
14. Hernán, M. A., Dahabreh, I. J., Dickerman, B. A., Swanson, S. A. "The Target Trial Framework for Causal Inference From Observational Data: Why and When Is It Helpful?" *Annals of Internal Medicine* 2025. https://doi.org/10.7326/ANNALS-24-01871
15. Danaei, G., Rodríguez, L. A. G., Cantero, O. F., Logan, R., Hernán, M. A. "Observational data for comparative effectiveness research: an emulation of randomised trials of statins and primary prevention of coronary heart disease." *Statistical Methods in Medical Research* 2013;22(1):70-96.
16. Rosenbaum, P. R., Rubin, D. B. "The central role of the propensity score in observational studies for causal effects." *Biometrika* 1983;70(1):41-55. https://doi.org/10.1093/biomet/70.1.41
17. Robins, J. M., Hernán, M. A., Brumback, B. "Marginal Structural Models and Causal Inference in Epidemiology." *Epidemiology* 2000;11(5):550-560.

> **Note:** The TARGET statement is the 2025 JAMA publication; the earlier BMJ Open paper is the development protocol, not the final checklist. The RCT-DUPLICATE correlation and the Hansford review figures are quoted from their published abstracts and main results.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_propensity-scores-and-weighting.md`](06_propensity-scores-and-weighting.md) — the estimator used inside the emulation.
- Causal graphs: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — the adjustment set.
- Survival analysis: [`03_survival-analysis.md`](03_survival-analysis.md) — time-zero and competing risks.
- Sensitivity analysis: [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) — unmeasured confounding.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — the results table.
- Survey file: [`../../04_statistics_causal_inference.md`](../../04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.3 writes a target trial protocol.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md) — assumptions ledger.
- Missing data: [`04_missing-data.md`](04_missing-data.md) — loss to follow-up and censoring.
- Module 13: [`../../13_biopharma_clinical_development.md`](../../13_biopharma_clinical_development.md) — trial design and real-world evidence.
- Module 25: [`../../25_trial_operations_hta_global_regulatory.md`](../../25_trial_operations_hta_global_regulatory.md) — HTA and regulatory use.
