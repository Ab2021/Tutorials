# Post-Authorisation Safety Studies and Active Safety Surveillance

> Module 13 — Biopharma and Clinical Development
> Topic 13 of 13
> Estimated study time: 5 hours
> Prerequisites: Topics 02, 06, 07, and 12
> Version 2026.3

## 1. Why this topic exists

Spontaneous reporting, covered in Topic 12, has no denominator. It can tell a safety team that an
event is reported disproportionately often for a drug, and it cannot tell them how often the event
occurs among treated patients. That distinction decides what can be done with the finding. A
disproportionality signal justifies a review. It does not justify a label change, because nobody can
say whether the absolute risk is one in a thousand or one in a million.

Active surveillance supplies the denominator. A network of hospitals, insurers, and primary care
systems holds longitudinal records on defined populations: who was dispensed which drug, when, and
what happened afterwards. From those records a study can compute person-time at risk, count events,
and estimate an incidence rate per thousand patient-years. The same records can support a cohort
study, a case-control study, or a self-controlled design, each with its own biases and its own
requirements.

Two regulatory instruments make this work routine rather than optional. In the European Union, a
post-authorisation safety study (PASS) can be imposed as a condition of marketing authorisation, and
must be registered before it starts. In the United States, the Food and Drug Administration Amendments
Act of 2007 created the Sentinel Initiative, a national active surveillance system, and gave its
statutory hook in section 505(k)(3) of the Federal Food, Drug, and Cosmetic Act. A sponsor that once
argued about whether a safety question was worth a study now receives the question as a protocol
milestone with a date.

### 1.1 The distinction that organises the topic

The word "signal" and the word "risk" are used interchangeably in commercial material and are not
interchangeable. A signal is information that suggests a new association and requires assessment. A
confirmed risk is an association supported by adequate evidence, usually including a study with a
denominator, and it is what a label describes. The work in this topic is the bridge between them, and
most of the methodological content exists because the bridge is easier to describe than to cross.

| State | Evidence | What can be said | What cannot |
|---|---|---|---|
| Signal | Disproportionate reporting, case series, trial finding | "Reports suggest a possible association" | "The drug causes the event" |
| Validated signal | Case review, temporality, plausibility, class comparison | "The association warrants assessment" | "The risk is X per thousand" |
| Confirmed risk | Cohort or case-control study with a denominator and bias analysis | "The incidence is X per thousand patient-years, with an excess of Y" | "This patient's event was caused by the drug" |

### 1.2 The failure pattern

1. A signal appears in spontaneous reports for a new anticoagulant and intracranial haemorrhage.
2. The sponsor runs a PASS in a claims database, comparing users of the new drug with users of
   warfarin.
3. The crude incidence rate ratio is 1.20 and is reported as a 20 percent increased risk.
4. The comparator group is younger and has fewer prior bleeds, and the analysis does not adjust.
5. The adjusted estimate is 1.08 with an interval from 0.91 to 1.28.
6. The label already carries a warning written at step 3, and withdrawing it is harder than never
   adding it.

Step 4 is the analytical failure. The crude comparison is confounded by the indication and by the
prescriber's judgement about who can tolerate which drug. The corrected estimate is not "no
difference"; it is "no difference detectable in this population, with an interval that excludes a
large increase and does not exclude a modest one". The difference between those two statements is the
subject of Section 6.

### 1.3 Why the design choice matters more than the analysis

In a randomised trial, the design is fixed and the analysis is the hard part. In an observational
safety study, the design is the hard part and the analysis follows from it. Four design decisions
determine whether the estimate means anything:

- **Who is in the cohort?** New users of the drug, or anyone who ever took it. New-user designs avoid
  the bias that comes from selecting survivors, and they cost sample size.
- **When does follow-up start?** At the first prescription, at the diagnosis, or at the cohort entry
  date. A follow-up that starts before exposure creates immortal time, which biases the estimate
  downward for the exposed.
- **Who is the comparator?** An active comparator with the same indication is the default for a
  serious question. An untreated comparator compares treated patients with people who are different
  in every way that led to treatment.
- **What is the outcome definition?** A diagnosis code, a validated algorithm, or a chart review. The
  positive predictive value of the definition, and whether it differs between arms, can move the
  estimate more than the adjustment does.

Each decision is a place where a study can be made to produce a preferred answer. The protocol is the
mechanism that prevents this, and registration is the mechanism that makes the protocol binding.

### 1.4 Who is blocked when the study is weak

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Safety physician | A risk statement with a number | Depends on data the sponsor does not own |
| Regulator | Confidence in the benefit-risk decision | Receives an estimate without its bias analysis |
| Patient | Accurate information about absolute risk | Has no access to the underlying data |
| Sponsor | A label that survives review | Cannot rerun the study after the protocol is fixed |
| Payer | A basis for coverage decisions | Sees the label, not the study |
| Data partner | Trust in the network | Bears liability for data it cannot audit centrally |

### 1.5 The industry question

> Given a signal and a distributed network of observational data, what study would turn the signal
> into an estimate a regulator would accept, and what would have to be true for that estimate to be
> wrong?

The second half of the question is the quantitative bias analysis, and it is the part most often
omitted. An estimate without a bias analysis is a point on a page. An estimate with a bias analysis is
a claim with boundaries.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** passive reporting from active surveillance on the dimensions of denominator,
   provenance, and bias.
2. **Write** a PASS protocol outline that satisfies the registration and reporting requirements of the
   European Medicines Agency and the postmarketing requirements of the US Food and Drug Administration.
3. **Choose** between a cohort, a case-control, and a self-controlled case series design for a given
   safety question, and justify the choice.
4. **Compute** an incidence rate with person-time, an incidence rate ratio with a confidence interval,
   and an excess risk per thousand patient-years.
5. **Apply** propensity score matching or stratification and explain what it does and does not remove.
6. **Detect** immortal time bias, outcome misclassification, and informative censoring from the study
   design and the data.
7. **Quantify** the robustness of an estimate with an E-value and a differential misclassification
   analysis.
8. **Interpret** an active surveillance result against the original signal, and classify the pair as
   signal, validated signal, or confirmed risk.

## 3. The core mental model

### 3.1 The distributed data architecture

```text
   Data partners                        Coordinating centre
   ----------------                     -------------------
   Hospital A  ---\
   Insurer B   ----\
   Primary care C ---->  common data model  ---->  query package
   Registry D   ----/     (local, versioned)         (code, no patient data)
   Hospital E  ---/                                        |
                                                            v
                                              each partner runs the query
                                              on its own data, behind its
                                              own firewall
                                                            |
                                                            v
                                              aggregated counts returned
                                              (cells below a minimum suppressed)
                                                            |
                                                            v
                                              meta-analysis at the centre
```

The architecture exists because the data cannot be pooled. Privacy law, contracts, and institutional
policy keep patient records with the data partner. What travels is code and aggregate results, not
records. The common data model is what makes one query runnable in five different source systems.

Two consequences follow. First, the analyst never sees individual records, so data-quality problems
are discovered through aggregate diagnostics rather than by inspection. Second, the query must be
written before the data are seen, which is a feature: it forces the analysis to be specified in
advance.

### 3.2 The target trial framing

The cleanest way to design an observational safety study is to describe the randomised trial it is
imitating, and then to identify where the imitation fails.

| Target trial component | Randomised version | Observational version |
|---|---|---|
| Eligibility | Protocol criteria | Cohort entry criteria applied to records |
| Treatment strategies | Randomised arms | New user of drug A versus new user of drug B |
| Assignment | Random | Prescriber and patient choice |
| Start of follow-up | Randomisation date | First dispensing date |
| Outcome | Adjudicated endpoint | Diagnosis code or validated algorithm |
| Follow-up end | Trial end or event | Disenrolment, death, event, or study end |
| Causal contrast | Intention to treat or per protocol | Analogous contrasts, explicitly defined |
| Analysis | Pre-specified | Pre-specified and registered |

The table is a diagnostic tool. Every row where the observational version differs from the target is
a place where bias can enter, and the protocol should say how each is addressed. The largest gaps are
usually assignment, outcome definition, and follow-up end.

### 3.3 The three designs

```text
   COHORT
   Exposed   [============ person-time ============]  events
   Comparator[============ person-time ============]  events
             ^ start                    end ^
   Estimate: incidence rate ratio

   CASE-CONTROL
   Cases (with event)      <- exposure history before event
   Controls (sampled from risk set) <- exposure history
   Estimate: odds ratio approximating the rate ratio

   SELF-CONTROLLED CASE SERIES
   Case 1  [-- unexposed --][== risk window ==][-- unexposed --]
   Case 2  [-- unexposed --][== risk window ==][-- unexposed --]
   Estimate: relative incidence, within person
```

The cohort is the default because it produces incidence and person-time directly. The case-control is
efficient when the outcome is rare and the cohort is large. The self-controlled case series removes
all time-invariant confounding by comparing each person with themselves, and it fails when the event
changes the probability of subsequent exposure, which is exactly what happens with a fatal event.

### 3.4 The bias ladder

```text
   Confounding        -> adjust, stratify, or match; report the residual
   Selection          -> new-user design, active comparator, restriction
   Immortal time      -> align follow-up start with exposure
   Misclassification  -> validate the outcome, quantify the PPV
   Informative        -> account for disenrolment and death
     censoring
   Random error       -> confidence interval, and a power calculation
```

The ladder is ordered by how often each item changes the answer. Confounding and immortal time
account for most of the distance between a crude and a defensible estimate. Random error is the one
that gets the most attention in the write-up.

### 3.5 What active surveillance still cannot do

It cannot answer a question that the data do not contain. A claims database records diagnoses and
dispensings, not blood pressure, smoking, or alcohol use. A confounder that is not measured cannot be
adjusted for, and the E-value exists to say how strong such a confounder would have to be. It cannot
establish causality in an individual patient. And it cannot substitute for a trial when the question
is efficacy, because the patients who receive a drug are not the patients who would have been
randomised.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Passive surveillance | Safety monitoring based on voluntary reports | Treated as a measure of incidence | System |
| Active surveillance | Monitoring using longitudinal records on a defined population | Treated as complete ascertainment | System |
| Post-authorisation safety study (PASS) | A study of an authorised medicine to identify, characterise, or quantify a safety hazard, confirm its safety profile, or measure risk-management effectiveness | Used for any post-marketing study | Study |
| Post-authorisation efficacy study (PAES) | A study required to address efficacy questions after authorisation | Confused with PASS | Study |
| Postmarketing requirement (PMR) | A study or trial required by the FDA under section 505(o)(3) | Confused with a commitment | Requirement |
| Postmarketing commitment (PMC) | A study agreed with the FDA but not statutorily required | Reported as a requirement | Commitment |
| Risk management plan (RMP) | The document describing how risks will be characterised and minimised | Treated as a communication plan | Document |
| Sentinel Initiative | The FDA's national active surveillance system | Used as a synonym for its database | System |
| Sentinel Distributed Database | The pooled, de-identified data accessible through Sentinel | Described as a single dataset | Database |
| Sentinel Common Data Model (SCDM) | The structure to which data partners map their data | Confused with OMOP | Specification |
| OMOP CDM | The Observational Medical Outcomes Partnership common data model, used by DARWIN EU and OHDSI | Confused with SCDM | Specification |
| DARWIN EU | The European Medicines Agency's data and analytics network | Treated as a single database | Network |
| Distributed network | A set of data partners that run common queries locally | Treated as pooled data | Architecture |
| Federated query | A query executed at each site with only aggregates returned | Assumed to be equivalent to pooled analysis | Method |
| Common data model | A standard structure and vocabulary to which source data are mapped | Assumed to preserve all clinical detail | Specification |
| Incident user | A patient starting the drug during the observation window, with no prior use | Confused with prevalent user | Design |
| New-user design | A cohort restricted to incident users, with follow-up from first exposure | Omitted without comment | Design |
| Active comparator | A comparator drug used for the same indication | Replaced by untreated controls | Design |
| Prevalent user bias | Selection of survivors who tolerated the drug | Missed in claims analyses | Bias |
| Immortal time bias | Follow-up time during which the outcome cannot occur, credited to an exposure | Created by defining exposure after follow-up starts | Bias |
| Propensity score | The probability of treatment given covariates | Treated as a causal adjustment | Probability |
| Standardised mean difference | A balance metric between groups | Confused with a significance test | Dimensionless |
| Positivity | Every patient has a non-zero probability of each treatment | Ignored in heavily restricted cohorts | Assumption |
| Person-time | The total time at risk contributed by cohort members | Replaced by patient counts | Patient-years |
| Incidence rate | Events divided by person-time | Called a risk | Per 1,000 patient-years |
| Incidence rate ratio | The ratio of two incidence rates | Called a relative risk without comment | Dimensionless |
| Excess risk | The absolute difference in incidence rates | Omitted from the label | Per 1,000 patient-years |
| Number needed to harm | The number of patients treated for one additional event | Computed from a rate ratio | Patients or patient-years |
| Self-controlled case series (SCCS) | A design comparing event rates within person across exposed and unexposed periods | Used when the event affects exposure | Design |
| Risk window | The period after exposure during which the event is attributed to the drug | Chosen without a mechanism | Time |
| Washout period | A period excluded or treated as unexposed between exposures | Omitted | Time |
| Event-dependent exposure | The event changes the probability of later exposure | Ignored in SCCS | Assumption |
| Outcome validation | Confirmation of a coded outcome against a reference standard | Assumed unnecessary in claims | Process |
| Positive predictive value (PPV) | The proportion of coded cases that are true cases | Confused with sensitivity | Proportion |
| Negative control outcome | An outcome not expected to be affected by the drug | Used without a plausible null | Design |
| Quantitative bias analysis | A calculation of how much bias would change the estimate | Replaced by a paragraph of caveats | Analysis |
| E-value | The minimum strength of association a confounder would need with both exposure and outcome to explain the estimate | Interpreted as a probability | Ratio |
| Protocol registration | Public registration of the study protocol before data collection | Treated as optional | Process |
| EU PAS number | The identifier assigned to a registered post-authorisation study | Confused with a protocol number | Identifier |
| Feasibility count | A pre-study count of eligible patients and events | Treated as a result | Count |
| Attributable risk | The difference between exposed and unexposed incidence | Confused with the rate ratio | Per 1,000 patient-years |

## 5. The mechanics, step by step

This procedure runs from a safety question to a registered, analysed, and interpreted study. It
assumes access to a distributed network or a licensed claims or electronic health record database,
and it assumes the study will be registered and reported. The sequence is the one a regulator
expects: protocol first, data second, analysis third.

### Step 1. Frame the safety question and the decision it informs

**Input:** a signal, a risk management plan obligation, or a regulator's request.

**Output:** a one-paragraph question naming the population, the exposure contrast, the outcome, the
time horizon, and the decision the study will inform.

**Rationale:** A PASS is commissioned to answer a question, and the question determines the design.
"Is there an excess of intracranial haemorrhage among new users of drug A compared with new users of
drug B, over 12 months, sufficient to change the label?" is a different study from "what is the
incidence of intracranial haemorrhage among all users of drug A?". The first requires a comparator
and a contrast; the second requires only a denominator.

**Example:** "Among adults with non-valvular atrial fibrillation initiating anticoagulation in
2022-2025, what is the incidence rate of intracranial haemorrhage within 12 months of initiation with
drug A compared with warfarin, and is any excess large enough to require a label change?"

**Edge case:** the regulator's request names an outcome but not a comparator. Propose one in the
protocol and justify it; do not let the absence of a specified comparator become an untreated
comparison by default.

### Step 2. Write the protocol synopsis before touching the data

**Input:** the question and the available data sources.

**Output:** a protocol synopsis with eligibility, exposure, outcome, follow-up, comparator, estimand,
analysis, and sensitivity analyses.

**Rationale:** The protocol is the instrument that distinguishes a study from an analysis. It fixes
the choices before the results are known, which is what makes the result interpretable. Writing it
before a feasibility count is uncomfortable and necessary, because a count that shows the study is
underpowered creates pressure to change the question.

The synopsis should state the estimand in the ICH E9(R1) sense: the population, the treatment
conditions being compared, the outcome variable, the handling of intercurrent events such as
discontinuation and death, and the population-level summary. A study that compares "ever exposed" with
"never exposed" and reports a hazard ratio has not stated an estimand that corresponds to a clinical
decision.

**Edge case:** the data source cannot support the protocol as written. Change the protocol and record
why, before registration. Changing it after seeing the results is a different act.

### Step 3. Choose the design

**Input:** the question, the outcome frequency, the available data, and the expected exposure
prevalence.

**Output:** one primary design, with a written justification and a pre-specified secondary design for
sensitivity.

**Rationale:** The design determines which biases are controllable and which are not. A cohort gives
incidence and person-time directly. A case-control is efficient for rare outcomes but requires
sampling from the risk set. A self-controlled case series eliminates time-invariant confounding but
fails when the outcome affects subsequent exposure.

| Design | Best when | What it estimates | Main threat | Typical use |
|---|---|---|---|---|
| New-user cohort | Exposure is common; incidence is needed; comparator exists | Incidence rate ratio, with absolute rates | Confounding by indication | Default for a comparative safety question |
| Case-control | Outcome is rare; cohort would be enormous | Odds ratio approximating the rate ratio | Control sampling and recall | Rare serious events in large databases |
| Self-controlled case series | Event is acute and non-fatal; exposure is intermittent | Relative incidence within person | Event-dependent exposure | Vaccine and acute drug reactions |
| Case-crossover | Exposure is transient; event is acute | Odds ratio within person | Time-varying confounding by indication | Trigger studies, e.g. acute events after a dose |
| Descriptive incidence | Only a denominator is needed | Incidence rate | No comparator, so no contrast | Background rates for a risk management plan |

The table has five rows because the choice is not binary. A study that reports a cohort as primary
and an SCCS as sensitivity is stronger than either alone, provided both are pre-specified.

### Step 4. Select data sources and assess fitness for purpose

**Input:** candidate databases and the protocol.

**Output:** a data source assessment covering coverage, enrolment, coding, outcome validation, and
the availability of confounders.

**Rationale:** A database that covers a million patients is useless if it does not record the
confounders that matter or if enrolment is discontinuous. Fitness for purpose is specific to the
question, not a general property of the database.

The assessment should answer:

1. **Coverage.** What proportion of the target population is represented, and is enrolment
   continuous or episodic?
2. **Exposure capture.** Are dispensings recorded with dose and days supplied, or only prescriptions?
3. **Outcome capture.** Are diagnoses recorded in hospital, outpatient, or both? Is there a validated
   algorithm, and what is its positive predictive value?
4. **Confounders.** Which variables in the protocol are available, and at what granularity?
5. **Linkage.** Can patients be followed across providers, and is death recorded?
6. **Latency.** How long after an event does the record appear?

A negative answer to any of these is not disqualifying. It is a limitation to be stated and, where
possible, quantified.

### Step 5. Define exposure, outcome, and the risk window

**Input:** the protocol and the data dictionary.

**Output:** computable definitions with codes, dates, and grace periods.

**Rationale:** Most disagreements about an observational safety result are disagreements about
definitions. A "new user" who received a prescription 400 days earlier is not new. An outcome defined
by a hospital discharge code excludes events in the emergency department. A risk window of 30 days
imputes a mechanism, and a 365-day window imputes a different one.

The definitions should specify:

- **Exposure:** the drug, the route, the dose, the days supplied, and the grace period that bridges
  gaps between dispensings.
- **Outcome:** the code set, the care setting, and whether prevalent cases are excluded by a
  look-back window.
- **Risk window:** the period after initiation attributed to the drug, with the mechanism or
  precedent that justifies it.
- **Washout:** the period after discontinuation during which the effect is assumed to persist.

**Edge case:** the drug is used intermittently. A simple "on-treatment" definition then leaves large
unexposed gaps that are not comparable with continuous use. Consider an as-treated versus
intention-to-treat contrast, and pre-specify both.

### Step 6. Build the cohort: new user, eligibility, and follow-up

**Input:** the definitions and the source data.

**Output:** a cohort with one row per patient, an index date, and follow-up time.

**Rationale:** The index date anchors everything. Follow-up must start at the first dispensing, not
at the diagnosis that led to it and not at the cohort entry date. Starting before exposure creates
immortal time: the period between cohort entry and first prescription is time during which the
patient cannot have the outcome while unexposed, and crediting it to the exposed group biases the
estimate.

The build should apply, in order:

1. Eligibility at the index date, including age and the indication.
2. A look-back period with no prior use of the drug, typically 6 to 12 months.
3. Exclusion of patients with the outcome before the index date, if the question is incident events.
4. Follow-up from the index date to the earliest of outcome, disenrolment, death, or study end.
5. A record of the reason for each censoring event, so that informative censoring can be assessed.

**Edge case:** a patient switches from the comparator to the target drug. The primary analysis may
censor at switch, which estimates the effect of the initial treatment, or continue, which estimates
the effect of a treatment strategy. Both are legitimate; they answer different questions and must be
labelled.

### Step 7. Choose the comparator and check balance

**Input:** the cohort and the candidate comparators.

**Output:** a primary comparator, a balance table, and a decision on whether the comparison is
supportable.

**Rationale:** An active comparator with the same indication is the default because it makes the
treated groups similar in the reason for treatment. An untreated comparator compares patients who
received a drug with patients who did not, and the difference between those groups includes
everything that led to treatment.

Balance is assessed with standardised mean differences, not p-values. A p-value depends on sample
size and will flag trivial differences in a large cohort. A standardised difference above 0.1 is
conventionally treated as material imbalance.

| Comparator | What it controls | What it does not | When to use |
|---|---|---|---|
| Active drug, same indication | Confounding by indication; health-seeking behaviour | Confounding by contraindication if the drugs are used in different patients | Default for a comparative question |
| Active drug, different indication | Some health-system effects | The indication itself | When no same-indication comparator exists |
| Untreated, same indication | The indication | Severity, comorbidity, and everything else that drives prescribing | Only with a strong design and a bias analysis |
| Historical control | Secular trends in the source | Changes in practice and coding | When contemporaneous data are unavailable |

### Step 8. Control confounding

**Input:** the cohort, the confounder list, and the balance table.

**Output:** a primary adjusted estimate and at least one sensitivity analysis with a different
method.

**Rationale:** No single method removes confounding. Propensity score matching balances observed
covariates and makes the groups comparable on the distribution of the score; it does not remove
unmeasured confounding, and it can hide positivity violations when the groups do not overlap.
Stratification is transparent but struggles with many confounders. Weighting uses all patients but
is sensitive to extreme weights.

| Method | What it does | Strength | Limitation |
|---|---|---|---|
| Propensity score matching | Matches each exposed patient to similar comparators | Transparent balance; interpretable | Discards patients; residual confounding |
| Propensity score weighting | Weights to a pseudo-population | Uses all patients; estimates a marginal effect | Extreme weights; unstable with rare exposure |
| Stratification | Compares within strata | Simple and auditable | Sparse with many confounders |
| Regression adjustment | Models the outcome | Efficient | Model-dependent; extrapolation |
| Negative control outcome | Tests for residual confounding | Detects bias that adjustment missed | Requires a plausible null outcome |

The negative control is the most underused. An outcome that the drug cannot plausibly affect, but
that shares the confounders, should show no association. If it does, the primary estimate is suspect.

### Step 9. Estimate incidence and effect with person-time

**Input:** the cohort with follow-up time and events.

**Output:** incidence rates per 1,000 patient-years, the rate ratio with a confidence interval, and
the excess rate.

**Rationale:** The rate is the quantity that a label can state and a clinician can use. A rate ratio
alone does not tell a patient whether the absolute difference is one event per hundred thousand
patient-years or one per hundred.

```text
   IR      = events / person-years
   IRR     = IR_exposed / IR_comparator
   SE(ln IRR) = sqrt(1/events_exposed + 1/events_comparator)
   95% CI  = exp(ln IRR +/- 1.96 x SE)
   Excess  = IR_exposed - IR_comparator
   NNH     = 1 / Excess        (in patient-years)
```

**Edge case:** the events are not independent because patients can have more than one, or because
sites cluster. Use a robust variance or a frailty model, and state it. The Poisson interval above
assumes independence and is anti-conservative when it fails.

### Step 10. Quantify the bias that remains

**Input:** the primary estimate and the study design.

**Output:** a quantitative bias analysis covering at least misclassification, immortal time, and
unmeasured confounding.

**Rationale:** A caveat paragraph is not a bias analysis. Each bias can be expressed as a
calculation, and the calculation says whether the conclusion survives.

For unmeasured confounding, the E-value is the minimum strength of association that a confounder
would need with both the exposure and the outcome to explain away the observed rate ratio:

```text
   For RR >= 1:  E = RR + sqrt(RR x (RR - 1))
   For RR < 1:   use 1/RR in the formula
```

For differential outcome misclassification, when the positive predictive value differs between arms
and specificity is high:

```text
   observed RR = true RR x (PPV_exposed / PPV_comparator)
```

For immortal time, compare the person-time accumulated before the index date under the flawed and
corrected definitions. A material difference means the flawed analysis is biased toward the
exposed.

### Step 11. Register, run, and check provenance

**Input:** the final protocol and the query package.

**Output:** a registration record, a query package, and a provenance log.

**Rationale:** Registration fixes the protocol and makes deviations visible. In the European Union,
imposed non-interventional PASS must be registered in the HMA-EMA catalogue of real-world data
studies, which replaced the EU PAS Register in February 2024, and the study keeps an EU PAS number.
The protocol is uploaded before data collection starts, and the final report abstract within one
month of finalisation.

The provenance log should record the data cut date, the common data model version, the query version,
the code list version, and the date each partner ran the query. Without it, the result cannot be
reproduced, and a regulator cannot tell whether a difference between two runs reflects a change in
the data or a change in the code.

### Step 12. Interpret and classify

**Input:** the primary estimate, the sensitivity analyses, and the bias analysis.

**Output:** a classification and a recommendation, with the boundaries of the evidence stated.

**Rationale:** The study produces an estimate. The classification decides what can be said about it,
and the categories are evidentiary, not rhetorical.

| Result pattern | Classification | What can be said | Next step |
|---|---|---|---|
| No association; interval excludes a large effect | No confirmed risk | No excess detected in this population | Continue routine monitoring |
| Modest association; interval includes 1; bias analysis cannot exclude it | Unresolved | An excess cannot be excluded | Further study or extended follow-up |
| Association robust to bias analysis; absolute excess material | Confirmed risk | An excess of X per 1,000 patient-years | Label, risk management, communication |
| Association in the opposite direction to the signal | Signal not confirmed | The signal is not supported by this study | Reassess the signal; check the design |
| Negative control outcome also associated | Estimate unreliable | Residual confounding is present | Redesign before interpreting |

The last row is the one teams resist. If the negative control shows an association, the study has
failed a test it set for itself, and the primary estimate should not be reported as a finding until
the reason is understood.

### 5.13 What a feasibility count must establish

Before registration, a count of eligible patients and expected events should establish three things,
and a study that skips them risks being uninterpretable whatever it finds.

1. **Cohort size after exclusions.** Count patients at each eligibility step, not only at the end. A
   cohort of 40,000 that began as 400,000 has a selection process worth describing.
2. **Expected events, not only patients.** Person-time multiplied by the background incidence gives
   the expected event count. A cohort of 100,000 with a one-year follow-up and an outcome incidence
   of 1 per 1,000 patient-years yields about 100 events.
3. **The minimum detectable rate ratio.** With E events in the exposed and C in the comparator, the
   smallest rate ratio detectable with 80 percent power at a two-sided alpha of 0.05 is approximately:

```text
   MDHR = exp(2.802 x sqrt(1/E + 1/C))
```

A study with 216 and 400 events has a standard error of 0.0844 on the log scale and a minimum
detectable rate ratio of exp(2.802 x 0.0844) = 1.27. It can detect a 27 percent increase. It cannot
detect a 10 percent increase, and a null result from it means "no increase larger than about a
quarter", not "no increase". Stating this before the study prevents the null from being over-read
afterwards.

The count is also the moment to check positivity. If the two drugs are used in almost disjoint
populations, no adjustment will make them comparable, and the right conclusion is that the data
cannot answer the question.

## 6. Worked example

The example follows a post-authorisation safety study from protocol to a regulatory decision. The
product is a direct oral anticoagulant, called drug A here, and the outcome is intracranial
haemorrhage. The numbers are constructed for teaching and are internally consistent; they are not
from a real product or a real study.

### 6.1 The setting

A signal from spontaneous reports suggested an excess of intracranial haemorrhage with drug A
relative to warfarin. The sponsor is required to run a PASS. The protocol specifies a new-user cohort
of adults with non-valvular atrial fibrillation, a primary comparison of drug A against warfarin, a
12-month follow-up, and an outcome defined by hospital discharge codes for nontraumatic intracranial
haemorrhage, with a 12-month look-back to exclude prevalent cases.

| Design element | Specification |
|---|---|
| Population | Adults with non-valvular atrial fibrillation, newly initiating anticoagulation, 2022-2025 |
| Exposure | First dispensing of drug A or warfarin, with no use in the prior 12 months |
| Comparator | Warfarin, same indication, active comparator |
| Outcome | Hospital discharge diagnosis of nontraumatic intracranial haemorrhage |
| Follow-up | From first dispensing to outcome, disenrolment, death, or 12 months |
| Estimand | Rate ratio of incident intracranial haemorrhage, treatment-policy, over 12 months |
| Primary analysis | Propensity score matching on 42 baseline covariates, Cox model |

### 6.2 The crude rates

| | Patients | Person-years | Events | Rate per 1,000 patient-years |
|---|---:|---:|---:|---:|
| Drug A | 48,000 | 36,000 | 216 | 6.00 |
| Warfarin | 96,000 | 80,000 | 400 | 5.00 |

```text
IRR = 6.00 / 5.00 = 1.20

SE(ln IRR) = sqrt(1/216 + 1/400)
           = sqrt(0.004630 + 0.002500)
           = 0.08444

95% CI = exp(ln 1.20 +/- 1.96 x 0.08444)
       = exp(0.1823 +/- 0.1655)
       = 1.02 to 1.42

Excess rate = 6.00 - 5.00 = 1.00 per 1,000 patient-years
NNH         = 1 / 0.001 = 1,000 patient-years
```

The crude analysis says the rate is 20 percent higher with drug A, the interval excludes 1, and one
additional event would be expected per 1,000 patient-years of treatment. On its own, that would
support a label change.

### 6.3 The adjusted estimate

The propensity score model includes 42 covariates: age, sex, prior bleeding, renal function, liver
disease, antiplatelet use, alcohol-related diagnoses, falls, dementia, prior stroke, and the
components of the CHA2DS2-VASc and HAS-BLED scores. Matching is 1:1 with a caliper of 0.2 times the
standard deviation of the logit of the score.

After matching, 38,200 pairs remain. All covariates have standardised mean differences below 0.08.
The Cox model gives a hazard ratio of 1.08 with a 95 percent interval from 0.91 to 1.28.

```text
ln(HR) = 0.0770
SE     = (ln 1.28 - ln 0.91) / (2 x 1.96)
       = (0.2469 - (-0.0943)) / 3.92
       = 0.0870
Check: exp(0.0770 +/- 1.96 x 0.0870) = exp(-0.0936) to exp(0.2475) = 0.91 to 1.28
```

The adjustment moved the estimate from 1.20 to 1.08 and widened the interval so that it includes 1.
Most of the crude excess was confounding: drug A was prescribed to older patients with more prior
bleeding, which is exactly the group at higher risk of intracranial haemorrhage.

The E-value for the adjusted estimate is:

```text
E = 1.08 + sqrt(1.08 x (1.08 - 1))
  = 1.08 + sqrt(0.0864)
  = 1.08 + 0.294
  = 1.37
```

An unmeasured confounder would need a rate ratio of at least 1.37 with both drug A and the outcome to
explain away the estimate. That is a weak confounder by the standards of this literature: frailty,
falls, and alcohol use could plausibly be that strong and are imperfectly measured in claims. The
result is therefore fragile in a specific and statable way.

### 6.4 Diagnostic 1: immortal time in the original build

The first version of the cohort defined the index date as the date of the first atrial fibrillation
diagnosis, not the date of the first anticoagulant dispensing. The median gap between the two was 21
days, and during that gap the patient was unexposed but counted in the exposed group's follow-up.

```text
Pre-index time credited to drug A = 48,000 patients x 21/365 years = 2,762 person-years
Corrected person-time            = 36,000 - 2,762 = 33,238 person-years
Corrected rate                   = 216 / 33,238 x 1,000 = 6.50 per 1,000 patient-years
Corrected crude IRR              = 6.50 / 5.00 = 1.30
```

Removing the immortal time raises the crude estimate from 1.20 to 1.30. The flawed version made the
drug look safer, which is the usual direction: the immortal period is time during which the outcome
cannot occur, and counting it as exposed time dilutes the exposed rate.

The diagnostic is simple. For every exposed patient, compute the difference between the cohort entry
date and the first dispensing date. If the median is more than a few days, or if the total pre-index
person-time exceeds a small fraction of the exposed person-time, the build has immortal time.

### 6.5 Diagnostic 2: differential outcome validation

The outcome definition is a diagnosis code, and its positive predictive value is not the same in both
arms. A chart review of 400 sampled cases finds that 92 percent of the warfarin-coded events are true
intracranial haemorrhages, against 71 percent of the drug A-coded events. The difference is plausible:
warfarin patients have routine international normalised ratio monitoring and more imaging, and a
coded bleed in that group is more likely to have been confirmed.

Assuming high specificity, the observed rate ratio is the true rate ratio multiplied by the ratio of
the positive predictive values:

```text
observed RR = true RR x (PPV_drug A / PPV_warfarin)
1.08        = true RR x (0.71 / 0.92)
1.08        = true RR x 0.772
true RR     = 1.08 / 0.772 = 1.40
```

Correcting for the differential validation moves the estimate from 1.08 to about 1.40, and the
conclusion from "no detectable increase" to "an increase of about 40 percent". The direction is not
arbitrary: if the exposed arm's outcomes are less often true events, the observed rate is too low,
and correcting upward is required.

This is the most consequential diagnostic in the example, and it is invisible in the primary
analysis. It requires a validation sample, which requires access to records or a linked registry, and
it is the first thing a good reviewer asks about.

### 6.6 The varied assumption: the comparator

Suppose the team had used untreated patients with atrial fibrillation as the comparator instead of
warfarin. That group contributes 60,000 person-years and 192 events, a rate of 3.20 per 1,000
patient-years.

```text
IRR against untreated = 6.00 / 3.20 = 1.88
```

The estimate nearly doubles. The reason is not that drug A is more dangerous than the adjusted
analysis suggested; it is that untreated patients are a selected group. Some have a contraindication
to anticoagulation, some have paroxysmal disease judged low risk, and some are not coded as treated
because their prescriptions fall outside the database. Comparing treated patients with untreated
patients compares the drug with the reasons for not taking it.

| Comparator | Rate in comparator | IRR | What it answers |
|---|---:|---:|---|
| Warfarin, crude | 5.00 | 1.20 | Unadjusted comparison of two treated groups |
| Warfarin, adjusted | 5.00 | 1.08 | Confounding-adjusted comparison of two treated groups |
| Untreated | 3.20 | 1.88 | Comparison of treated with untreated, which is not a treatment contrast |
| Warfarin, PPV-corrected | 5.00 | 1.40 | Adjusted and corrected for differential validation |

The table is the argument for reporting more than one comparator. The same events produce 1.08, 1.40,
and 1.88 depending on choices that a reader cannot see unless they are shown.

### 6.7 The decision

The study team takes three findings to the safety committee: the adjusted hazard ratio of 1.08 with
its interval, the bias analysis showing that differential outcome validation could move the estimate
to 1.40, and the power calculation showing that the study could detect a rate ratio of about 1.27 but
not 1.10.

The committee concludes that the study does not confirm the signal and does not exclude it. The label
is updated to describe the uncertainty and the absolute rates, and a second study with a validated
outcome and a larger sample is commissioned. The signal remains open.

That outcome is unsatisfying and correct. A study that cannot distinguish 1.08 from 1.40 has not
answered the question, and saying so is more useful than choosing the estimate that suits the
commercial position.

### 6.8 What the example shows

First, adjustment changed the estimate by 10 percent and the conclusion by a great deal, because the
crude interval excluded 1 and the adjusted interval did not. The width of the interval matters as much
as the point.

Second, the two diagnostics that mattered were not in the primary analysis. Immortal time and
differential validation are properties of the design and the measurement, and neither appears in a
model output.

Third, the comparator determined whether the answer was 1.08 or 1.88. An untreated comparison is not a
neutral default; it is a different question with a different bias.

Fourth, the honest conclusion was "unresolved". The pressure to produce a number that supports a
decision is the main reason studies overstate what they found.

## 7. Code, query, or calculation

The script reproduces the worked example's arithmetic and adds the two calculations that a reviewer
will ask for: the minimum detectable rate ratio, and a sensitivity grid for differential outcome
validation. It uses only the standard library.

Assumptions: events are independent and Poisson-distributed; follow-up is complete within the study
window; specificity of the outcome definition is near 1, so the positive predictive value ratio
approximates the misclassification ratio; and the E-value formula applies to a rate ratio interpreted
as a risk ratio, which is conventional for rare outcomes.

```python
"""PASS arithmetic: rates, rate ratios, E-value, misclassification, and power."""
import math

Z975 = 1.959963984540054
Z80 = 0.8416212335729143


def rate(events, person_years):
    return 1000.0 * events / person_years


def rate_ratio(e1, py1, e2, py2):
    r1, r2 = rate(e1, py1), rate(e2, py2)
    irr = r1 / r2
    se = math.sqrt(1 / e1 + 1 / e2)
    return irr, math.exp(math.log(irr) - Z975 * se), math.exp(math.log(irr) + Z975 * se)


def e_value(rr):
    """Minimum confounder association with both exposure and outcome to explain away rr."""
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * (rr - 1))


def ppv_correct(observed, ppv_exposed, ppv_comparator):
    """True RR when positive predictive value differs and specificity is near 1."""
    return observed / (ppv_exposed / ppv_comparator)


def mdhr(e1, e2, power_z=Z80, alpha_z=Z975):
    """Smallest rate ratio detectable at the given power, two-sided alpha."""
    return math.exp((power_z + alpha_z) * math.sqrt(1 / e1 + 1 / e2))


def main():
    py_a, ev_a = 36000.0, 216
    py_w, ev_w = 80000.0, 400
    print("CRUDE RATES")
    print(f"  drug A     {ev_a} events / {py_a:,.0f} PY = {rate(ev_a, py_a):.2f} per 1,000 PY")
    print(f"  warfarin   {ev_w} events / {py_w:,.0f} PY = {rate(ev_w, py_w):.2f} per 1,000 PY")
    irr, lo, hi = rate_ratio(ev_a, py_a, ev_w, py_w)
    print(f"  IRR {irr:.3f}  95 percent CI {lo:.3f} to {hi:.3f}")
    print(f"  excess {rate(ev_a, py_a) - rate(ev_w, py_w):.2f} per 1,000 PY"
          f"  NNH {1000 / (rate(ev_a, py_a) - rate(ev_w, py_w)):.0f} PY")

    print("\nADJUSTED ESTIMATE AND E-VALUE")
    adj = 1.08
    print(f"  adjusted HR {adj:.2f}  95 percent CI 0.91 to 1.28")
    print(f"  E-value for the adjusted estimate {e_value(adj):.2f}")
    print(f"  E-value for the crude estimate    {e_value(irr):.2f}")

    print("\nIMMORTAL TIME")
    pre = 48000 * 21 / 365
    corrected_py = py_a - pre
    print(f"  pre-index person-time credited to drug A {pre:,.0f} PY")
    print(f"  corrected person-time {corrected_py:,.0f} PY")
    print(f"  corrected rate {rate(ev_a, corrected_py):.2f} per 1,000 PY")
    print(f"  corrected crude IRR {rate(ev_a, corrected_py) / rate(ev_w, py_w):.2f}")

    print("\nDIFFERENTIAL OUTCOME VALIDATION (observed RR 1.08)")
    print("  PPV comparator = 0.92, PPV exposed varies")
    for ppv in (0.90, 0.80, 0.71, 0.60, 0.50):
        print(f"    PPV {ppv:.2f}  true RR {ppv_correct(adj, ppv, 0.92):.2f}")
    print("  PPV equal at 0.90 in both arms: true RR "
          f"{ppv_correct(adj, 0.90, 0.90):.2f}")

    print("\nCOMPARATOR SENSITIVITY")
    for label, py, ev in (("warfarin", py_w, ev_w), ("untreated", 60000.0, 192)):
        r, l, h = rate_ratio(ev_a, py_a, ev, py)
        print(f"  against {label:<10} comparator rate {rate(ev, py):.2f}"
              f"  IRR {r:.2f} ({l:.2f} to {h:.2f})")

    print("\nMINIMUM DETECTABLE RATE RATIO (80 percent power, two-sided 0.05)")
    print(f"  with {ev_a} and {ev_w} events: {mdhr(ev_a, ev_w):.2f}")
    print(f"  with a 50 percent larger cohort ({ev_a * 3 // 2} and {ev_w * 3 // 2} events): "
          f"{mdhr(ev_a * 3 // 2, ev_w * 3 // 2):.2f}")


if __name__ == "__main__":
    main()
```

Output on the authoring machine, Python 3.12:

```text
CRUDE RATES
  drug A     216 events / 36,000 PY = 6.00 per 1,000 PY
  warfarin   400 events / 80,000 PY = 5.00 per 1,000 PY
  IRR 1.200  95 percent CI 1.017 to 1.416
  excess 1.00 per 1,000 PY  NNH 1000 PY

ADJUSTED ESTIMATE AND E-VALUE
  adjusted HR 1.08  95 percent CI 0.91 to 1.28
  E-value for the adjusted estimate 1.37
  E-value for the crude estimate    1.69

IMMORTAL TIME
  pre-index person-time credited to drug A 2,762 PY
  corrected person-time 33,238 PY
  corrected rate 6.50 per 1,000 PY
  corrected crude IRR 1.30

DIFFERENTIAL OUTCOME VALIDATION (observed RR 1.08)
  PPV comparator = 0.92, PPV exposed varies
    PPV 0.90  true RR 1.10
    PPV 0.80  true RR 1.24
    PPV 0.71  true RR 1.40
    PPV 0.60  true RR 1.66
    PPV 0.50  true RR 1.99
  PPV equal at 0.90 in both arms: true RR 1.08

COMPARATOR SENSITIVITY
  against warfarin   comparator rate 5.00  IRR 1.20 (1.02 to 1.42)
  against untreated  comparator rate 3.20  IRR 1.88 (1.54 to 2.28)

MINIMUM DETECTABLE RATE RATIO (80 percent power, two-sided 0.05)
  with 216 and 400 events: 1.27
  with a 50 percent larger cohort (324 and 600 events): 1.21
```

### 7.1 What the output shows

The crude interval excludes 1 and the adjusted interval does not. That single change, from 1.02 to
1.42 down to 0.91 to 1.28, is the difference between a label change and a research question, and it
comes from adjusting for variables that were recorded.

The E-value of 1.37 is the most useful single number for a reviewer. It says that a confounder with a
rate ratio of 1.37 against both drug A and intracranial haemorrhage would explain the result. Falls,
frailty, and alcohol use are plausibly at least that strong and are poorly measured in claims data,
so the estimate is not robust.

The misclassification grid is the argument for validating outcomes. At a positive predictive value of
0.71 in the exposed arm against 0.92 in the comparator, the true rate ratio is 1.40. The observed 1.08
is not wrong arithmetic; it is the wrong quantity.

The minimum detectable rate ratio shows the study's real resolution. At 216 and 400 events it can
detect 1.27. It cannot detect 1.10, which is the size of effect that would matter for a label
decision in a population with a baseline rate of 5 per 1,000 patient-years.

### 7.2 Limits of this calculation

The Poisson interval assumes independent events and complete follow-up. A real analysis with
recurrent events, competing risks, or clustering would use a robust variance or a frailty model. The
positive predictive value correction assumes high specificity; if specificity is low, the correction
is more complex and the observed rate ratio can be biased in either direction. The E-value addresses
only unmeasured confounding, not selection or misclassification, and it is a sensitivity device
rather than a probability.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and describes published frameworks at the access date, 27
> October 2026. It is not legal or regulatory advice.

### 8.1 The two legal routes

| Jurisdiction | Instrument | What it provides | Binding or guidance |
|---|---|---|---|
| European Union | Directive 2001/83/EC, Article 21a; Regulation (EC) No 726/2004, Article 10a | Authority to require a non-interventional PASS | Binding |
| European Union | Commission Implementing Regulation (EU) No 520/2012, Article 36(5) and Annex III | Registration of imposed PASS and milestones in the final report | Binding |
| European Union | GVP Module VIII, Rev 3 | Detailed requirements for PASS conduct, reporting, and registration | Binding through the GVP structure |
| United States | Federal Food, Drug, and Cosmetic Act, section 505(o)(3), created by FDAAA 2007 | Authority to require postmarketing studies and clinical trials | Binding |
| United States | FDCA section 505(k)(3), created by FDAAA 2007 Title IX section 905 | The postmarket risk identification and analysis system that became Sentinel | Binding |
| United States | 21 CFR 314.80 | Postmarketing adverse event reporting | Binding |

The European and American routes differ in emphasis. The EU route is built around the PASS as a study
with a protocol, a registration, and a final report. The US route is built around postmarketing
requirements and a national surveillance system that the agency operates itself. A sponsor with a
global product may face both, and the protocols should be harmonised where possible without
pretending the requirements are identical.

### 8.2 Registration and reporting

In the EU, an imposed non-interventional PASS must be registered before data collection starts. The
protocol is uploaded to the register, and the final study report abstract is submitted within one
month of finalisation. Since February 2024 the register is the HMA-EMA Catalogue of real-world data
studies, which replaced the EU PAS Register; migrated records kept their EU PAS numbers, and new
studies continue to receive them. The ENCePP Code of Conduct and the ENCePP Guide on Methodological
Standards remain the methodological reference points even though the register moved.

| Artefact | Timing | Content | Consequence of omission |
|---|---|---|---|
| Protocol registration | Before data collection | Objectives, design, population, outcomes, analysis | Study is not compliant; results are hard to defend |
| Protocol amendment | Before the change is implemented | Reason, date, effect on the analysis | Apparent post-hoc change |
| Progress report | Annual, per the RMP or the requirement | Recruitment, data cuts, emerging issues | Missed milestone |
| Final study report | Within the required period after the end | Results, limitations, interpretation | Incomplete regulatory record |
| Abstract in the register | Within one month of finalisation | Summary of results | Public record incomplete |

### 8.3 Data standards

| Standard | Role | What it constrains | What it does not |
|---|---|---|---|
| Sentinel Common Data Model | Structure for Sentinel data partners | Table names, variable definitions, enrolment and dispensing structure | The clinical content of source records |
| OMOP CDM | Structure used by DARWIN EU and OHDSI | Concept identifiers, domain structure, vocabulary mapping | Outcome validation |
| ICD-10-CM and ICD-9-CM | Diagnosis coding in US claims | The code set from which outcome definitions are built | Whether a coded diagnosis is correct |
| SNOMED CT | Clinical terminology, often mapped to OMOP concepts | Concept definitions and hierarchy | Local coding practice |
| MedDRA | Adverse event coding for regulatory reporting | Event terms for spontaneous reports | Diagnosis codes in claims |
| ICH E2E | Pharmacovigilance planning | The risk management plan structure | Study design choices |

Two practical points follow. First, the common data model standardises structure, not meaning. Two
partners can both map to the Sentinel model and still differ in how they define an inpatient stay or
a dispensing. The query must be tested in each partner before it is trusted. Second, the outcome
definition is a code list, and code lists are versioned. A change from ICD-9-CM to ICD-10-CM, or a
change in a code's meaning, breaks the time series and must be handled explicitly rather than
interpolated.

### 8.4 What the frameworks do not settle

They do not specify a minimum detectable effect, a required follow-up length, a preferred design, or
a threshold for action. They require a protocol, a registration, a report, and a justification. The
scientific choices remain with the sponsor and the reviewers, which is why the protocol is the most
important document in the study.

## 9. Industry practice and case evidence

### 9.1 The scale of the networks

| Network | Data partners | Population | Access model |
|---|---|---|---|
| FDA Sentinel Distributed Database | Multiple partners, data from 2000 onward | About 541 million unique patient identifiers and 405 million patients with at least one day of drug and medical coverage, with 1.5 billion person-years, as published in October 2025 | Distributed; queries run by partners; aggregate results returned |
| DARWIN EU | About 30 data partners across 16 European countries in 2025, with a target of 40 in early 2026 | Reported at roughly 180 to 188 million patients in 2025 | Distributed, mapped to OMOP |
| HMA-EMA catalogue of real-world data studies | More than 3,000 registered studies by late 2025 | Not applicable | Public register |

The numbers move, and a document that quotes them should quote the date. The Sentinel figures in
particular are counts of patient identifiers, not of distinct people, because a person who changes
health plan can appear more than once. The "405 million patients with at least one day of drug and
medical coverage" is the more conservative denominator and the one to use for planning.

### 9.2 Documented practice and inference

| Claim | Status | Basis |
|---|---|---|
| Sentinel runs distributed queries against a common data model | Documented | Sentinel methods and data documentation |
| Sentinel uses TreeScan and other signal-identification methods | Documented | Sentinel publications and tool documentation |
| DARWIN EU maps partner data to OMOP and runs federated analyses | Documented | EMA network reports |
| Imposed EU PASS must be registered before data collection | Documented | Implementing Regulation 520/2012 and GVP Module VIII |
| Most sponsors register voluntary PASS as well | Inference | Consistent with guidance and practice; not auditable |
| A large share of PASS protocols are amended after registration | Inference | Common in complex studies; no public denominator |
| Active surveillance has replaced spontaneous reporting for signal detection | Not supported | Both remain in use, for different purposes |
| Sentinel results are publicly reproducible | Not supported | Aggregate results are published; the underlying data are not accessible |

### 9.3 What the networks changed

Before Sentinel, a safety question about a marketed drug was answered by commissioning a database
study, which took months to design, procure data for, and analyse. Sentinel changed the unit of work:
a query can be specified, reviewed, and run against a standing network in weeks. That speed is the
point, and it has three consequences.

First, the bottleneck moved from data access to protocol design. A query that is wrong is now fast and
wrong, and the review that catches it is methodological rather than logistical.

Second, the networks made routine surveillance possible. A product can be monitored continuously
against a set of pre-specified outcomes rather than reviewed once a year, which is closer to how the
risk management plan describes the obligation.

Third, the networks made the limits visible. A distributed query returns counts from each partner, and
the heterogeneity between partners is often larger than the effect being studied. A pooled estimate
that hides that heterogeneity is less informative than a table that shows it.

The pattern across sponsors is a shift from one-off studies to standing capability: a protocol
template, a pre-specified outcome library, a query package tested in each partner, and a review
process that can act on the output. The organisations that have this capability treat the protocol as
code and version it. The ones that do not run each study as a project and relearn the same lessons.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Immortal time | Follow-up starts before first dispensing | Exposure defined after the index date | Exposed rate diluted; drug looks safer | Index date equals first dispensing; audit the gap |
| Prevalent user bias | Cohort includes patients with prior use | Survivors who tolerated the drug are selected | Understates early risk | 6 to 12 month look-back with no prior use |
| Untreated comparator | Methods call it "controls" | Treated and untreated differ in everything that led to treatment | Inflated or deflated estimate, uninterpretable | Active comparator with the same indication |
| Confounding by contraindication | One drug is avoided in patients at high risk | Prescriber choice depends on the outcome risk | Estimate biased away from the null | Restriction, propensity score, negative control |
| Differential outcome validation | PPV differs by arm | One arm is monitored more intensively | Observed rate biased toward the better-monitored arm | Validation study; quantitative correction |
| Informative censoring | Disenrolment differs by arm | Sick patients leave the database | Estimated rate depends on follow-up | Report censoring reasons; sensitivity analysis |
| Unmeasured confounding | E-value close to plausible confounder strength | Claims lack clinical variables | Residual confounding | E-value and negative control outcomes |
| No overlap in the propensity score | Many exposed patients unmatched | The drugs are used in disjoint populations | Estimate applies to a small subset or none | Check overlap; report the matched population |
| Protocol changed after results | Amendment dated after the data cut | Pressure to fit the finding | Result no longer pre-specified | Register before data collection; log amendments |
| Multiple outcomes unadjusted | One of twelve outcomes is "significant" | Testing at a nominal alpha | False finding | Pre-specify a primary outcome; adjust secondaries |
| Partner heterogeneity hidden | Pooled estimate with no site table | Different coding and practice across partners | Average that describes no one | Report partner-specific estimates |
| Query version drift | Results change between runs with the same data | Code updated without a version record | Non-reproducible finding | Version the query; log the run date |
| Registration after data collection | Registration date follows the data cut | Study conducted then registered | Non-compliant; not credible | Register first, without exception |
| Healthy adherer bias | Adherent patients have better outcomes | Adherence correlates with health behaviour | Protective effect for any drug | Compare with an active comparator, not non-use |

The first three rows account for most of the distance between a crude and a defensible estimate. The
middle rows require data that claims often lack, and the E-value exists to say how much they could
matter. The last rows are process failures that make the study non-reproducible regardless of its
statistical quality.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The index date equals the start of exposure | Person-time must reflect exposed time | Cohort entry defined at diagnosis | Compare dates; compute pre-index person-time |
| New users are genuinely new | Prevalent users are survivors | Prior use outside the look-back window | Extend the look-back; check the prior-use distribution |
| The comparator is exchangeable after adjustment | Causal interpretation of the contrast | Unmeasured confounding; contraindication | Negative control outcome; E-value |
| The outcome code identifies the event | The rate depends on it | Coding variation; differential monitoring | Validation study with PPV by arm |
| Follow-up is complete | Person-time and events must be captured | Disenrolment, out-of-network care, death | Report censoring; link to death data |
| Events are independent | Poisson intervals assume it | Recurrent events; clustering by site | Robust variance; frailty model |
| Positivity holds | Every patient could receive either drug | Disjoint prescribing populations | Overlap plots; matched sample description |
| The risk window reflects the mechanism | The contrast is defined by it | Window chosen for convenience | Pre-specify with a pharmacological rationale |
| Data partners are comparable | Pooled estimates require it | Different coding, coverage, and practice | Partner-specific estimates; heterogeneity statistics |
| The query is reproducible | A result must be checkable | Code changes without versioning | Version control; run log |
| The protocol predates the data | Pre-specification is what makes the result interpretable | Registration after the data cut | Registration date versus data cut date |

The most fragile assumption is exchangeability, and no observational study can test it directly. The
negative control outcome is the closest available test: an outcome the drug cannot plausibly affect
should show no association after adjustment, and if it does, the adjustment has failed.

The overall limit is that a PASS answers the question it was designed to answer, in the population it
studied, with the outcome definition it used. Extending the result to a different population, a
different dose, or a different outcome is an extrapolation, and it should be labelled as one.

## 12. Exercises

**Exercise 1 — Design choice (application, 25 minutes).**
Deliverable: a design recommendation for a PASS on a vaccine and an acute neurological event, with
the cohort, case-control, and self-controlled options compared.
Acceptance criteria: three designs compared on bias and efficiency; one recommended with a
justification; the event-dependent exposure problem addressed.
Time: 25 minutes.

**Exercise 2 — Rates and rate ratio (application, 30 minutes).**
Deliverable: incidence rates, the rate ratio, the 95 percent interval, and the excess rate for 84
events in 24,000 patient-years against 100 events in 50,000 patient-years.
Acceptance criteria: rates per 1,000 patient-years; interval on the log scale; excess and number
needed to harm stated with their units.
Time: 30 minutes.

**Exercise 3 — Confounding (application, 35 minutes).**
Deliverable: an adjustment plan for a comparison in which the exposed group is older and has more
prior events.
Acceptance criteria: at least three confounders named with their source variables; a propensity score
or stratification approach specified; balance assessed with standardised differences; a negative
control outcome proposed.
Time: 35 minutes.

**Exercise 4 — Immortal time audit (critique, 30 minutes).**
Deliverable: an audit of a cohort build in which follow-up starts at diagnosis, with the pre-index
person-time computed and the corrected estimate.
Acceptance criteria: the gap distribution described; the corrected person-time and rate shown; the
direction of the bias stated.
Time: 30 minutes.

**Exercise 5 — Outcome validation (design, 40 minutes).**
Deliverable: a validation plan for an outcome defined by diagnosis codes, including the sample size,
the reference standard, and the analysis by arm.
Acceptance criteria: PPV estimated separately by arm; the correction method named; the assumption of
high specificity stated.
Time: 40 minutes.

**Exercise 6 — Bias analysis (design, 40 minutes).**
Deliverable: a quantitative bias analysis for an adjusted hazard ratio of 1.12 with a 95 percent
interval from 0.95 to 1.32.
Acceptance criteria: E-value computed; at least one misclassification scenario computed; the
conclusion under each scenario stated; the analysis labelled as a sensitivity analysis.
Time: 40 minutes.

**Exercise 7 — Protocol and registration (application, 30 minutes).**
Deliverable: a one-page protocol synopsis and a registration checklist for a PASS.
Acceptance criteria: estimand stated; eligibility, exposure, outcome, and follow-up defined; the
registration timing stated; the amendment rule stated.
Time: 30 minutes.

**Exercise 8 — Interpretation memo (design, 50 minutes).**
Deliverable: a two-page memo to a safety committee interpreting the worked example, with a
recommendation.
Acceptance criteria: the estimate, its interval, and its bias analysis presented; the study's minimum
detectable effect stated; a classification of signal, validated signal, or confirmed risk; a
recommendation proportionate to the evidence.
Time: 50 minutes.

## 13. Self-check answers

**Exercise 1.** A cohort is inefficient for a rare acute event and vulnerable to confounding by
indication. A case-control is efficient but needs careful control sampling. A self-controlled case
series removes time-invariant confounding and is well suited to a vaccine and an acute event, provided
the event does not change the probability of subsequent vaccination; a fatal event violates that
assumption, and a pre-exposure washout period may be needed. The recommendation should name the
assumption and the sensitivity analysis.

**Exercise 2.** Exposed rate = 84/24,000 x 1,000 = 3.50 per 1,000 patient-years. Comparator rate =
100/50,000 x 1,000 = 2.00 per 1,000 patient-years. IRR = 1.75. SE = sqrt(1/84 + 1/100) = 0.1480. The
interval is exp(0.5596 +/- 0.2901) = 1.31 to 2.34. Excess = 1.50 per 1,000 patient-years, so the
number needed to harm is 1/0.0015 = 667 patient-years.

**Exercise 3.** Age, prior bleeding, renal function, antiplatelet use, and prior stroke are typical.
A propensity score with 1:1 matching and a caliper of 0.2 times the standard deviation of the logit is
a standard approach. Balance is assessed with standardised mean differences below 0.1. A negative
control outcome, such as an event the drug cannot cause but which shares the confounders, tests
residual confounding.

**Exercise 4.** If the median gap between diagnosis and first dispensing is 21 days and 48,000
patients are exposed, the pre-index person-time is 48,000 x 21/365 = 2,762 patient-years. Subtracting
it raises the exposed rate and the crude rate ratio. The bias is toward the null for the exposed
group, because the immortal period is time during which the outcome cannot occur.

**Exercise 5.** Sample several hundred cases per arm, use chart review or a linked registry as the
reference standard, and estimate the positive predictive value separately by arm with a confidence
interval. Correct the observed rate ratio by the ratio of predictive values, assuming high
specificity, and report both the observed and corrected estimates. State that the correction assumes
that the misclassification is non-differential with respect to outcome status within each arm.

**Exercise 6.** E-value = 1.12 + sqrt(1.12 x 0.12) = 1.12 + 0.3666 = 1.49. If the exposed arm's
positive predictive value is 0.75 against 0.90 in the comparator, the corrected estimate is
1.12/0.833 = 1.34. The conclusion changes from "no detectable increase" to "a possible increase of
about a third", and the memo should say so.

**Exercise 7.** The synopsis states the population, exposure, comparator, outcome and horizon, the
estimand with its intercurrent-event strategy, the primary analysis, and the sensitivity analyses.
Registration happens before data collection, the protocol is uploaded, the EU PAS number is recorded,
and any amendment is dated and justified before implementation.

**Exercise 8 rubric.** Full marks require the estimate and interval, the bias analysis with a
corrected estimate, the minimum detectable effect, a classification consistent with the evidence, and
a recommendation that states what would change the conclusion. Deduct marks for reporting the crude
estimate as the finding, for omitting the validation correction, or for a recommendation that implies
more certainty than the interval supports.

## 14. Interview questions

**1. What can active surveillance do that spontaneous reporting cannot?** Strong: compute incidence
and person-time in a defined population, and therefore absolute risk and excess risk. Weak: says it
gives "real-world evidence" without the denominator argument.

**2. What is a PASS, and when is it required?** Strong: a study of an authorised medicine to
characterise or quantify a safety hazard, required by a regulator as a condition of authorisation or
after new safety information, with a protocol and a registration. Weak: describes it as any
post-marketing study.

**3. Why is an active comparator preferred to an untreated group?** Strong: it controls confounding by
indication and health-seeking behaviour, because both groups were judged to need treatment. Weak:
says it is "more rigorous" without the confounding argument.

**4. What is immortal time bias, and how do you detect it?** Strong: follow-up time before exposure is
credited to the exposed group, during which the outcome cannot occur; detect it by comparing cohort
entry and first dispensing dates and computing the pre-index person-time. Weak: calls it "lead-time
bias" without a mechanism.

**5. How do you handle an outcome defined by diagnosis codes?** Strong: validate a sample against a
reference standard, estimate the positive predictive value by arm, and correct or bound the estimate.
Weak: assumes the code is accurate because it is widely used.

**6. A PASS finds a hazard ratio of 1.08 with an interval of 0.91 to 1.28. What do you conclude?**
Strong: no increase detected, but the study's minimum detectable effect matters; state it and the
bias analysis before calling it a null. Weak: says the drug is safe.

**7. What does an E-value of 1.37 tell you?** Strong: an unmeasured confounder would need a rate ratio
of at least 1.37 with both exposure and outcome to explain the estimate; whether that is plausible
depends on the subject matter. Weak: treats it as a p-value or as proof of robustness.

**8. How would you run a study across five data partners without pooling records?** Strong: map to a
common data model, distribute a versioned query, return aggregate counts with small-cell suppression,
and check heterogeneity across partners. Weak: proposes to send record-level data to a central server.

**9. What would make you distrust an active surveillance result?** Strong: no registered protocol, no
validation of the outcome, no balance table, no negative control, and a comparator that differs from
the treated group in the reason for treatment. Weak: distrusts it because it is observational.

## 15. Key terms

- **Active comparator** — a drug used for the same indication as the study drug.
- **Active surveillance** — monitoring using longitudinal records on a defined population.
- **Attributable risk** — the difference between exposed and unexposed incidence.
- **Common data model** — a standard structure to which source data are mapped.
- **DARWIN EU** — the European Medicines Agency's real-world data network.
- **Distributed network** — data partners running common queries locally.
- **E-value** — the confounder strength needed to explain away an estimate.
- **EU PAS number** — the identifier for a registered European post-authorisation study.
- **Excess risk** — the absolute difference in rates, per 1,000 patient-years.
- **Federated query** — a query run at each site with aggregates returned.
- **Immortal time bias** — follow-up before exposure credited to the exposed group.
- **Incidence rate** — events per unit of person-time.
- **Informative censoring** — censoring related to the outcome risk.
- **New-user design** — a cohort restricted to patients starting the drug.
- **OMOP CDM** — a common data model used by DARWIN EU and OHDSI.
- **PASS** — post-authorisation safety study.
- **Person-time** — total time at risk contributed by cohort members.
- **Positive predictive value** — the proportion of coded cases that are true cases.
- **Postmarketing requirement** — a study or trial required by the FDA.
- **Prevalent user bias** — selection of patients who already tolerated the drug.
- **Propensity score** — the probability of treatment given covariates.
- **Protocol registration** — public registration of the protocol before data collection.
- **SCDM** — the Sentinel Common Data Model.
- **Self-controlled case series** — a within-person design comparing exposed and unexposed periods.
- **Sentinel Initiative** — the FDA's national active surveillance system.
- **Standardised mean difference** — a balance metric between groups.
- **Target trial** — the randomised trial an observational study imitates.

## 16. Sources

- Directive 2001/83/EC of the European Parliament and of the Council, Article 21a.
- Regulation (EC) No 726/2004, Article 10a.
- Commission Implementing Regulation (EU) No 520/2012, Article 36(5) and Annex III.
- European Medicines Agency. *Guideline on Good Pharmacovigilance Practices, Module VIII:
  Post-Authorisation Safety Studies*, Rev 3. EMA/813938/2011.
- US Food and Drug Administration Amendments Act of 2007, Public Law 110-85, Title IX.
- US Code of Federal Regulations, Title 21, Part 314.80.
- Sentinel Initiative. *Key Database Statistics*, published October 2025.
- Sentinel Initiative. *Sentinel Common Data Model*, versions 8.1.0 and 8.2.0.
- Platt R, Brown JS, Robb M, et al. The FDA Sentinel Initiative: an evolving national resource.
  *New England Journal of Medicine*, 2018;379(22):2091-2093.
- Whitaker HJ, Farrington CP, Spiessens B, Musonda P. Tutorial in biostatistics: the self-controlled
  case series method. *Statistics in Medicine*, 2006;25(10):1768-1797.
- Farrington CP. Relative incidence estimation from case series for vaccine safety evaluation.
  *Biometrics*, 1995;51(1):228-235.
- Kulldorff M, Fang Z, Walsh SJ. A tree-based scan statistic for database disease surveillance.
  *Biometrics*, 2003;59(2):323-331.
- VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value.
  *Annals of Internal Medicine*, 2017;167(4):268-274.
- HMA-EMA Catalogues of real-world data sources and studies, launched February 2024.
- European Medicines Agency. *Network Data Steering Group Annual Report 2025*.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`12_statistical-pharmacovigilance.md`](12_statistical-pharmacovigilance.md)
- Related topic: [`11_digital-twins-in-clinical-trials.md`](11_digital-twins-in-clinical-trials.md)
- Related topic: [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md)
- Related topic: [`06_safety-and-pharmacovigilance.md`](06_safety-and-pharmacovigilance.md)
- Survey file: [`../../course/13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
