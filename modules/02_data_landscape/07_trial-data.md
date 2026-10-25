# Trial Data

> Module 02 — The Biomedical Data Landscape
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01 and 02 are helpful; basic biostatistics assumed
> Version 2026.3

---

## 1. Why this topic exists

Clinical trial data is the most expensive data in biomedicine per patient-record, and the most structured. A single phase 3 trial can cost hundreds of millions of dollars, enroll thousands of patients across dozens of countries, and generate a submission package that a regulator will review line by line.

The data is different from every other source in this module in one decisive respect: **it is collected to answer a pre-specified question, under a protocol, with a defined population, a defined intervention, and a defined endpoint**. That design control is why a trial can support a causal claim that an EHR or claims analysis cannot.

It is also why trial data is narrow. The population is selected by eligibility criteria, the follow-up is bounded by the protocol, the endpoints are the ones the sponsor chose, and the care delivered is often not the care a patient would receive in routine practice. A trial answers "does this intervention work in this population under these conditions" and not "will it work in my clinic."

The commercial stakes:

- A team that analyzes trial data with routine-care assumptions will mis-specify the analysis population and produce a result that does not match the protocol.
- A data manager who does not understand the difference between a query and a deviation will fail an inspection.
- A statistician who ignores the estimand framework will answer a different question from the one the protocol asked.
- A sponsor that cannot produce SDTM-conformant data will have a submission rejected on technical grounds before any scientific review.

The decision this topic supports: whether a trial dataset can support a regulatory or clinical claim, how its analysis populations differ, and what the data provenance must contain for the submission to be accepted.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Clinical data manager | Whether the database is ready to lock | Confuses query resolution with data completeness |
| Biostatistician | Which population to analyze | Mixes ITT and per-protocol results |
| Regulatory lead | Whether the submission is technically valid | SDTM non-conformance causes a refuse-to-file |
| Medical monitor | Whether a deviation affects safety | Misses the deviation-to-endpoint link |
| Real-world evidence lead | Whether trial and RWE results are comparable | Ignores population and endpoint differences |
| Programmer | How to build ADaM datasets | Derives variables without a specification |
| Clinical operations lead | Whether enrollment is on track | Confuses screened, enrolled, and randomized |

### 1.2 What is structurally different from routine care data

Five differences define the category:

1. **Prospective collection under a protocol.** The visit schedule, the assessments, and the data fields are specified before enrollment.
2. **Source data verification.** A monitor compares the database to the source record, and discrepancies generate queries.
3. **Defined analysis populations.** Intention-to-treat, per-protocol, safety, and other populations are defined in the protocol and the analysis plan before unblinding.
4. **A locked database.** The data is frozen at a defined point for analysis, and changes after lock are controlled and documented.
5. **A regulatory submission format.** Data is transformed to CDISC standards for submission, with a define.xml that describes every dataset and variable.

None of these exists in routine care data, and each creates a document that an analyst must read.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a trial data record from source to submission, naming the system and the standard at each stage.
2. **Distinguish** intention-to-treat, per-protocol, and safety populations, and compute the effect estimate in each.
3. **Compute** a sample size for a two-proportion endpoint and explain the effect of dropout on power.
4. **Apply** an alpha-spending boundary to an interim analysis and explain why the nominal threshold differs from 0.05.
5. **Explain** the estimand framework and write an estimand for a stated question.
6. **Critique** a trial analysis for population definition, deviation handling, and multiplicity, and propose the missing sensitivity analysis.
7. **Choose** between trial data and real-world data for a stated decision, and name the population and endpoint differences that limit comparison.
8. **Design** a provenance record for a trial dataset that names the protocol version, the analysis plan version, the data cut, and the SDTM/ADaM versions.

Mapping to assessment: outcomes 2, 3, and 4 are assessed by exercises 2, 3, and 4. Outcomes 1, 5, and 6 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

Trial data flows through a chain of controlled transformations, and each transformation is governed by a document.

```text
  PROTOCOL (v1.0 ... vN)
     |  defines population, intervention, endpoints, visits
     v
  EDC / ePRO / IRT  ----------------------- source data
     |  entry, edit checks, queries
     v
  CLINICAL DATABASE
     |  query resolution, coding (MedDRA, WHO Drug), SDV
     v
  DATABASE LOCK  (data cut date)
     |
     |  SDTM transformation (CDISC SDTMIG)
     v
  SDTM DATASETS  ------------------------- submission tabulations
     |
     |  ADaM transformation (CDISC ADaMIG)
     v
  ADaM DATASETS  ------------------------- analysis-ready
     |
     |  TLF programming
     v
  TABLES, LISTINGS, FIGURES  ------------- output
     |
     v
  SUBMISSION PACKAGE  (define.xml, reviewer's guide, datasets)
```

Read the diagram from the top. The protocol is the specification, and every later artefact traces back to it. The database lock is the point after which the data is frozen, and the data cut date is the boundary for every analysis. The SDTM step tabulates; the ADaM step analyses. Confusing the two is a common and consequential error: SDTM is not analysis-ready, and ADaM is not a tabulation.

### 3.1 The three populations

| Population | Definition | Question it answers | Common misuse |
|---|---|---|---|
| Intention-to-treat (ITT) | All randomized participants, analyzed as randomized | Does assigning the treatment work in practice? | Called "ITT" when exclusions were applied |
| Modified ITT (mITT) | Randomized participants meeting additional criteria, defined before unblinding | Does the treatment work in a defined subset? | Defined after seeing the data |
| Per-protocol (PP) | Participants who completed without major deviations | Does the treatment work when taken as specified? | Used as the primary analysis |
| Safety | All participants who received at least one dose | What are the harms? | Confused with ITT |

The ITT estimate is the primary basis for a superiority claim in most regulatory settings, because it preserves randomization. The PP estimate is a sensitivity analysis that can support or undermine the ITT result. A large ITT-PP discrepancy is a signal that the intervention was not taken or that deviations were related to outcome.

### 3.2 The estimand framework

ICH E9(R1) introduced a structured way to define what is being estimated. An estimand has five attributes:

1. **Treatment condition** — the intervention and comparator, including dose and duration.
2. **Population** — the patients to whom the question applies.
3. **Variable** — the endpoint measured.
4. **Intercurrent events** — events that affect interpretation (discontinuation, rescue medication, death).
5. **Population-level summary** — the quantity to be estimated (difference in proportions, hazard ratio).

Two trials can share a protocol and differ in estimand, because they handle intercurrent events differently. A "treatment policy" strategy counts events after discontinuation; a "hypothetical" strategy asks what would have happened without discontinuation. The numbers will differ, and both can be correct for different questions.

This is the single most important conceptual shift in trial statistics in the past decade, and it is the reason "what is the treatment effect" is an incomplete question.

### 3.3 Why trial data does not transfer to routine care

| Dimension | Trial | Routine care |
|---|---|---|
| Population | Eligibility criteria exclude comorbidity and age extremes | All comers |
| Adherence | Monitored and supported | Variable |
| Monitoring | Protocol visits, labs, imaging | Symptom-driven |
| Comparator | Placebo or active control | Whatever is used locally |
| Endpoint | Standardized, adjudicated | Coded or recorded variably |
| Duration | Fixed | Open-ended |

Each row is a reason a trial result can differ from a real-world result. None is a defect; all are limits on generalization.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Protocol | The document defining the trial | Treated as static after v1 | Document version |
| Amendment | A change to the protocol | Ignored in analysis | Version |
| EDC | Electronic data capture | Confused with ePRO | System |
| ePRO | Electronic patient-reported outcome | Confused with EDC | System |
| IRT | Interactive response technology for randomization and supply | Confused with EDC | System |
| IWRS | Interactive web response system | Same as IRT | System |
| Randomization | Assignment by chance | Confused with allocation concealment | Method |
| Allocation concealment | Hiding the next assignment | Confused with blinding | Method |
| Blinding | Concealment of treatment from participants and staff | Confused with allocation concealment | Level |
| Source data | The original record | Confused with the database | Record |
| SDV | Source data verification | Confused with data cleaning | Process |
| Query | A request to resolve a discrepancy | Confused with a deviation | Workflow item |
| Protocol deviation | A departure from the protocol | Confused with a query | Event |
| Major deviation | A deviation affecting safety or endpoints | Confused with any deviation | Classification |
| Database lock | The point the database is frozen | Confused with the data cut | Milestone |
| Data cut | The date defining the data included in an analysis | Confused with database lock | Date |
| ITT | Intention-to-treat | Misapplied when exclusions exist | Population |
| mITT | Modified intention-to-treat | Defined post hoc | Population |
| PP | Per-protocol | Used as primary | Population |
| Safety population | All treated participants | Confused with ITT | Population |
| Estimand | The precise quantity to be estimated | Confused with the endpoint | ICH E9(R1) |
| Intercurrent event | An event affecting interpretation | Ignored in naive analyses | Estimand attribute |
| Endpoint | The measured outcome | Confused with the estimand | Variable |
| Primary endpoint | The main outcome | Changed after interim | Variable |
| Surrogate endpoint | A substitute for a clinical outcome | Treated as clinical benefit | Variable |
| Multiplicity | Control of type I error across endpoints | Ignored in secondary claims | Method |
| Alpha spending | Allocation of type I error across interim looks | Confused with alpha | Function |
| O'Brien-Fleming | A conservative early-stopping boundary | Confused with Pocock | Boundary |
| Pocock | A less conservative early-stopping boundary | Confused with OBF | Boundary |
| DSMB | Data and safety monitoring board | Confused with the sponsor | Committee |
| Interim analysis | An analysis before the final data cut | Confused with a data cut | Analysis |
| SDTM | Study Data Tabulation Model | Treated as analysis-ready | CDISC standard |
| SDTMIG | SDTM Implementation Guide | Confused with SDTM | Implementation guide |
| ADaM | Analysis Data Model | Treated as submission tabulation | CDISC standard |
| ADSL | Subject-level analysis dataset | Confused with a domain | ADaM dataset |
| define.xml | Metadata describing the submission datasets | Treated as optional | CDISC standard |
| SEND | Standard for Exchange of Nonclinical Data | Confused with SDTM | CDISC standard |
| MedDRA | Medical Dictionary for Regulatory Activities | Confused with WHO Drug | Coding dictionary |
| WHO Drug | The WHO drug dictionary | Confused with MedDRA | Coding dictionary |
| CTCAE | Common Terminology Criteria for Adverse Events | Confused with MedDRA | Grading system |
| eCTD | Electronic common technical document | Confused with a dataset | Submission format |
| GCP | Good Clinical Practice | Treated as optional | ICH E6 |
| Part 11 | FDA regulation on electronic records | Confused with GCP | 21 CFR Part 11 |

### 4.1 Terms that are routinely conflated

- **Query versus deviation.** A query is a data clarification request. A deviation is a departure from the protocol. A resolved query does not cure a deviation.
- **Database lock versus data cut.** The lock freezes the database; the cut defines which data are in an analysis. An interim analysis has a cut but the database may remain open.
- **SDTM versus ADaM.** SDTM tabulates collected data in a standard structure; ADaM derives analysis variables. An analysis run on SDTM is possible and is usually wrong.
- **Allocation concealment versus blinding.** Concealment prevents knowing the next assignment before it is made; blinding prevents knowing the assignment after. Both matter and they are different.

---

## 5. The mechanics, step by step

This section is the working procedure for evaluating and using a trial dataset. It has 28 steps in five phases.

### Phase A — Establish the documents (steps 1-6)

**Step 1. Obtain the protocol and the current amendment.**
Input: the trial documentation. Output: the protocol version in force for the analysis.
Reason: eligibility, endpoints, and analysis populations are defined in the protocol, and amendments change them. An analysis against a superseded version is invalid.
Rule: record the protocol number and version, and the date it took effect.

**Step 2. Obtain the statistical analysis plan (SAP).**
Input: the SAP. Output: the version and the date of finalization relative to unblinding.
Reason: the SAP specifies the primary analysis, the populations, the multiplicity strategy, and the sensitivity analyses. A SAP finalized after unblinding is a different document.
Rule: record the SAP version and whether it was finalized before or after the first unblinding.

**Step 3. Identify the data cut and the database lock date.**
Input: the data management record. Output: both dates.
Reason: the cut defines the data in the analysis; the lock defines when it stopped changing. An interim analysis uses a cut without a lock.
Rule: state both, and state the number of participants and events at the cut.

**Step 4. Identify the randomization method and the stratification factors.**
Input: the protocol and the randomization specification. Output: the method and the factors.
Reason: stratified randomization requires stratified analysis or adjustment for the factors. Ignoring them can bias the estimate.
Rule: record the factors and whether they were used in the analysis.

**Step 5. Identify the blinding status and the unblinding date.**
Input: the trial record. Output: the blinding level and the date of unblinding.
Reason: any analysis before unblinding must preserve the blind; any analysis after must state the date.
Rule: for a blinded trial, confirm that the analyst did not have access to treatment assignment before the SAP was finalized.

**Step 6. Confirm the regulatory and ethics approvals.**
Input: the trial master file. Output: the approval record.
Reason: the analysis is only valid for the approved protocol, and a substantial amendment may require re-approval.
Rule: record the approving bodies and the dates.

### Phase B — Build the analysis populations (steps 7-12)

**Step 7. Reconstruct the screening and enrollment flow.**
Input: the subject-level data. Output: counts screened, screen failures, randomized, treated, completed, discontinued.
Reason: the flow diagram is the first thing a reviewer checks, and it reconciles the denominator.
Rule: every count must reconcile to the next; an unexplained gap is a finding.

**Step 8. Define the ITT population as randomized.**
Input: the randomization list. Output: the ITT set.
Reason: ITT requires that participants are analyzed as randomized, including those who never received treatment.
Rule: do not exclude participants for non-compliance, withdrawal, or missing data.

**Step 9. Define the safety population as treated.**
Input: the dosing data. Output: the safety set.
Reason: safety analyses include everyone who received at least one dose, analyzed as treated.
Rule: a participant randomized to treatment but never dosed is in ITT and not in safety.

**Step 10. Classify protocol deviations before unblinding.**
Input: the deviation log. Output: a classification into major and minor.
Reason: the classification determines the per-protocol population, and doing it after unblinding invites bias.
Rule: classify against pre-specified criteria, and record who classified and when.

**Step 11. Build the per-protocol population.**
Input: the ITT set and the major deviations. Output: the PP set.
Reason: PP answers a different question, and its construction must be documented.
Rule: state every exclusion criterion and the count removed by each.

**Step 12. Reconcile the populations.**
Input: all population definitions. Output: a reconciliation table.
Reason: the same participant can be in ITT and not safety, or in safety and not PP. The differences must be countable.
Rule: report the counts in each cell of the population cross-tabulation.

### Phase C — Prepare the data (steps 13-18)

**Step 13. Map collected data to SDTM domains.**
Input: the EDC data. Output: SDTM datasets.
Reason: SDTM is the submission tabulation standard, and the mapping is where collected data becomes a standard structure.
Rule: use the SDTMIG version required for the submission, and record it.

**Step 14. Apply controlled terminology.**
Input: coded values. Output: values using the CDISC controlled terminology version in force.
Reason: terminology versions change, and a submission is evaluated against the version in effect at the submission date.
Rule: record the terminology version and the date.

**Step 15. Code adverse events and medications.**
Input: verbatim terms. Output: MedDRA and WHO Drug codes with versions.
Reason: adverse event analysis depends on the coding, and a dictionary upgrade changes the counts.
Rule: record the MedDRA and WHO Drug versions, and do not upgrade mid-analysis without re-running.

**Step 16. Derive ADaM datasets from SDTM.**
Input: SDTM. Output: ADSL and the analysis datasets.
Reason: ADaM contains the derived variables (baseline, change from baseline, analysis flags) that the SAP specifies.
Rule: every derived variable has a definition in the define.xml; an undocumented derivation is a finding.

**Step 17. Apply the analysis flags.**
Input: ADaM. Output: flags for ITT, safety, PP, and any subgroup.
Reason: the flags are how the analysis population is implemented in code, and a wrong flag silently changes the population.
Rule: validate the flags against the population counts from step 12.

**Step 18. Freeze the analysis datasets with the data cut.**
Input: the ADaM datasets. Output: a versioned snapshot.
Reason: the analysis must be reproducible from the snapshot, not from a later refresh.
Rule: record the snapshot identifier, the date, and the hash.

### Phase D — Analyze (steps 19-24)

**Step 19. Implement the primary analysis exactly as specified.**
Input: the SAP and ADaM. Output: the primary estimate.
Reason: the primary analysis is the basis for the claim, and deviations from the SAP must be documented as such.
Rule: if the SAP cannot be implemented as written, document why and obtain approval before changing it.

**Step 20. Report the estimate with a confidence interval.**
Input: the model. Output: an estimate and interval.
Reason: a p-value alone does not convey the magnitude or the precision.
Rule: report the absolute difference and the relative measure where both are meaningful.

**Step 21. Run the sensitivity analyses specified in the SAP.**
Input: the SAP list. Output: estimates under alternative assumptions.
Reason: sensitivity analyses test whether the conclusion depends on a modelling choice.
Rule: pre-specify them; post hoc sensitivity analyses are exploratory and must be labelled.

**Step 22. Control multiplicity for secondary endpoints.**
Input: the endpoint hierarchy. Output: adjusted conclusions.
Reason: testing many secondary endpoints at 0.05 inflates the false positive rate.
Rule: use a fixed-sequence, hierarchical, or gatekeeping procedure as specified, and report which endpoints are confirmatory.

**Step 23. Analyze safety with the safety population.**
Input: the safety set. Output: adverse event tables by system organ class and preferred term.
Reason: safety conclusions are population-specific and cannot be transferred from efficacy analyses.
Rule: report exposure-adjusted rates where follow-up differs.

**Step 24. Reconcile the results to the top-line.**
Input: the analysis outputs. Output: a reconciliation.
Reason: discrepancies between the top-line and the analysis are common and are usually programming errors.
Rule: reconcile the primary endpoint count, the denominator, and the estimate.

### Phase E — Document and submit (steps 25-28)

**Step 25. Build define.xml and the reviewer's guide.**
Input: the datasets. Output: metadata and a guide.
Reason: a reviewer must be able to understand every dataset and variable without asking the sponsor.
Rule: every derivation is described; every controlled term is referenced.

**Step 26. Validate against the FDA validator rules.**
Input: the submission package. Output: a validation report.
Reason: technical rejection can occur before scientific review.
Rule: run the current validator and resolve every error; document every warning.

**Step 27. Write the provenance record.**
Input: everything above. Output: a one-page record.
Reason: the record is what makes the analysis auditable years later.
Rule: include protocol and SAP versions, data cut, SDTM and ADaM versions, terminology versions, and the snapshot hash.

**Step 28. State what the trial cannot answer.**
Input: the question. Output: a limits section.
Reason: a trial answers a narrow question, and the limits define the boundary of the claim.
Rule: name at least three limits, including population, duration, and comparator.

### 5.1 Field-level mechanics: the subject-level dataset

1. **Subject identifier.** Unique within the study, and the key to every other dataset.
2. **Site identifier.** Needed for site-level analyses and for clustering.
3. **Randomization date and arm.** The basis for ITT.
4. **First dose date.** The basis for the safety population.
5. **Discontinuation date and reason.** Needed for time-to-event and for the estimand.
6. **Analysis flags.** One per population, validated against the counts.
7. **Stratification factors.** Needed if the randomization was stratified.

### 5.2 Field-level mechanics: adverse events

1. **Verbatim term and coded term.** The verbatim is what the investigator wrote; the coded term is MedDRA.
2. **System organ class and preferred term.** The analysis is usually at the preferred term within system organ class.
3. **Severity grade.** CTCAE grades are not the same as seriousness.
4. **Seriousness.** A regulatory determination with specific criteria, not a synonym for severe.
5. **Causality assessment.** Investigator-assessed relatedness, which is subjective and varies by investigator.
6. **Action taken.** Dose reduced, interrupted, or discontinued, which is often more informative than the event itself.

### 5.3 Field-level mechanics: the analysis plan

1. **Primary endpoint definition.** Including the time window, the assessment method, and the handling of missing data.
2. **Estimand attributes.** The five attributes, stated explicitly.
3. **Analysis population.** ITT, mITT, or PP, with the rationale.
4. **Statistical model.** The test, the covariates, and the stratification.
5. **Missing data handling.** The primary approach and the sensitivity analyses.
6. **Multiplicity strategy.** The hierarchy or gatekeeping procedure.
7. **Interim analysis plan.** The timing, the alpha spending, and the stopping rules.

### 5.4 Quantitative checks before analysis

| Check | Computation | Passing signal | What failure means |
|---|---|---|---|
| Flow reconciliation | Screened minus failures equals randomized | Exact | Enrollment log discrepancy |
| Population counts | ITT, safety, PP cross-tabulation | Sums reconcile | Flag error |
| Randomization balance | Baseline covariates by arm | Balanced within chance | Randomization or stratification error |
| Event count | Primary events at the cut | Matches the expected information fraction | Data cut or adjudication issue |
| Query closure | Open queries at lock | Zero, or documented | Database locked prematurely |
| Deviation rate | Major deviations per participant | Stable across sites | Site or training problem |
| Coding version | MedDRA and WHO Drug versions | Consistent across the analysis | Mid-analysis upgrade |
| Dataset conformance | Validator errors | Zero errors | Technical rejection risk |

### 5.5 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Baseline value | Last value before first dose | The last pre-dose value represents baseline | What if the value is missing? |
| Change from baseline | Post-baseline minus baseline | The baseline is comparable across arms | Was the assessment timing the same? |
| Best response | Maximum or minimum across visits | The visit schedule is complete | What about unevaluable visits? |
| Time to event | First event date minus randomization | The event date is accurate | How were interval-censored events handled? |
| Adherence | Doses taken divided by doses expected | The pill count or diary is accurate | Was the diary completed prospectively? |
| Exposure-adjusted rate | Events divided by person-time | Person-time is measured accurately | How was time after discontinuation handled? |
| Responder status | Threshold applied to a change score | The threshold is clinically meaningful | Who chose the threshold, and when? |
| Subgroup effect | Interaction term | The subgroup is pre-specified | Was the subgroup in the SAP? |

### 5.6 Data management mechanics that affect analysis

1. **Edit checks.** Automated rules that flag implausible values at entry. A weak edit check lets errors through, and the analysis inherits them.
2. **Query resolution.** A query is closed when the site responds. Closure does not mean the value is correct; it means the discrepancy was addressed.
3. **Coding.** Adverse events and medications are coded after entry, and a coding change alters counts. The dictionary version must be pinned.
4. **Reconciliation.** Serious adverse events are reconciled between the safety database and the clinical database. Unreconciled events produce inconsistent counts.
5. **Database lock checklist.** A formal list of checks that must pass before lock. A lock without a checklist is a date, not a control.
6. **Post-lock changes.** Any change after lock requires an audit trail and a documented reason. A post-lock change without a record is a data integrity finding.

### 5.7 Edge cases the procedure has to survive

**Null dates.** A subject with a randomization date but no first-dose date cannot be assigned to the safety population. The rule is to keep the subject in ITT, exclude from safety with a recorded reason, and report the count. A null first-dose date often means the subject was randomized and never treated, which is itself a finding about the site.

**Conflicting EDC and safety database.** A serious adverse event is in the safety database but not in the clinical database, or the dates differ. The rule is to reconcile before lock, to record the reconciliation date, and to report the number of unreconciled events. An unreconciled event produces two different safety tables from the same trial.

**Unit and code variants.** Adverse events are coded to MedDRA, and a version upgrade moves terms between system organ classes. Laboratory values may be in SI or conventional units, and the reference ranges differ. The rule is to pin the dictionary version for the analysis and to store both the source and the standardized unit.

**Late-arriving data.** Data entered after the cut but before the lock are the normal case, not an error. The rule is to define the cut, to define the lock, and to report the number of records changed between them. A lock without a change count is an assertion.

**Partial identifiers.** A subject identifier that appears in one dataset but not another breaks the join. The rule is to validate the identifier set across all domains before analysis and to report the missing subjects rather than dropping the rows.

**Duplicate subjects.** A subject randomized twice, or a screening failure re-entered with a new identifier, produces duplicates. The rule is to check on date of birth, sex, and randomization date, and to report the duplicate rate by site.

**Timezone and date boundaries.** Randomization time and first-dose time may be recorded in local time at sites in different zones, and a date-only field loses the order of events on the same day. The rule is to keep the time where it exists, to record the site timezone, and to use the date only when the time is unavailable.

### 5.8 Decision table: handling a missing primary endpoint

Missing endpoint data has more than two treatments, and the choice must match the estimand.

| Estimand strategy | Missing mechanism | Primary method | Sensitivity | What to report |
|---|---|---|---|---|
| Treatment policy | Discontinuation | Collect data after discontinuation | Worst-case for the arm with more missing | Both estimates |
| Hypothetical | Discontinuation | Multiple imputation under MAR | Tipping point | Imputation model |
| Composite | Intercurrent event | Event counted in the composite | None if pre-specified | Component counts |
| While on treatment | Discontinuation | Censor at discontinuation | None | Censoring rule |

The rule that overrides the table: the method must be pre-specified against the estimand. A method chosen after seeing the missingness pattern is a result, not a plan.

### 5.9 Field-level validation rules

| Rule | Test | Severity | Action on failure |
|---|---|---|---|
| Randomization date present | Every randomized subject has one | High | Quarantine; check the IRT feed |
| First dose after randomization | Dose date on or after randomization | High | Query; check timezone |
| ITT flag complete | Every randomized subject has an ITT flag | High | Block the analysis |
| Safety flag consistency | Safety flag implies a first dose | High | Block the safety analysis |
| AE verbatim to coded | Every AE has a MedDRA code | High | Code before lock |
| AE start before stop | Start date on or before stop date | Medium | Query |
| Lab unit known | Unit present and in the controlled list | High | Quarantine the value |
| Query closure | No open queries at lock | High | Complete queries before lock |
| Deviation classification | Every deviation has a major/minor flag | Medium | Classify before unblinding |
| Subject identifier across domains | Same ID set in every domain | High | Reconcile before analysis |

The two "block" rules are deliberate: an incomplete ITT flag or an inconsistent safety flag changes the population, and no downstream check can detect it after the fact.

### 5.10 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Primary population | Superiority claim | ITT | Not applicable |
| Primary population | Non-inferiority claim | Both ITT and PP | Not applicable |
| Intercurrent event | Treatment policy | Include events after discontinuation | Hypothetical strategy |
| Interim | Pre-specified | Alpha spending | No early stop |
| Missing data | MAR plausible | Multiple imputation | Tipping point |
| Multiplicity | Ordered endpoints | Fixed-sequence hierarchy | Gatekeeping |
| Submission standard | New study | SDTMIG v3.4 | Not applicable |

### 5.11 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Analysis flag | Population membership | The classification was pre-specified | Was it set before unblinding? |
| Baseline value | Last value before first dose | The timing is consistent across arms | Was the visit window respected? |
| Change from baseline | Post minus baseline | The assessment is comparable | Same instrument and timing? |
| Time to event | Event date minus randomization | The event date is accurate | How were interval events handled? |
| Censoring date | Last assessment or data cut | Censoring is non-informative | Is censoring related to the outcome? |
| Exposure duration | Last dose minus first dose | The dates are complete | What about interruptions? |
| Adherence | Doses taken over doses expected | The diary or count is accurate | Prospective or retrospective? |

The censoring assumption is the one most often violated in practice. If subjects discontinue because of toxicity, censoring at discontinuation is informative, and the while-on-treatment estimand inherits that dependence.

A second assumption deserves equal scrutiny: the baseline visit window. If the treatment arm is assessed a week later than the control arm because of scheduling, the change-from-baseline comparison includes a time difference as well as a treatment difference.

### 5.12 What the procedure does when the SAP is silent

A SAP that does not specify a case is not a licence to choose freely. Three defaults apply, in order:

1. **Follow the protocol.** If the protocol specifies the analysis, the SAP cannot override it without an amendment.
2. **Follow the regulator's guidance.** If the protocol is silent, use the convention in the applicable guidance and cite it.
3. **Document the choice before unblinding.** If neither applies, write the rule, date it, and obtain approval.

The failure mode is a choice made after seeing the data and described afterwards as a convention. The date is the evidence, and it is why every SAP change is dated.

### 5.13 Operationalizing the estimand

An estimand statement is only useful if it changes the ADaM derivations. Three practical checks:

- **Does the intercurrent event strategy appear in the data?** If the strategy is "treatment policy," the analysis must include events after discontinuation, which requires a flag identifying those events and a decision about follow-up after discontinuation.
- **Does the population attribute match the analysis flag?** If the estimand specifies "all randomized," the analysis flag must be the ITT flag, not the safety flag.
- **Does the summary measure match the model?** A difference in proportions, a hazard ratio, and an odds ratio are different summary measures for the same variable, and each answers a different question.

A protocol that states an estimand and an SAP that implements a different one is a protocol-SAP mismatch, and it is one of the most common findings in a statistical review.

The mismatch is rarely intentional. It usually arises because the estimand was written by a medical writer and the SAP by a statistician, and neither reviewed the other's text closely enough. A single joint review before finalization prevents it.

Four checks to run on any estimand:

1. Can the population attribute be implemented with an existing ADaM flag, or does it need a new derivation?
2. Does the intercurrent event strategy require data that the EDC collects, or does it require a field that does not exist?
3. Is the summary measure estimable from the design, given the amount of censoring?
4. Is the sensitivity analysis specified against the same estimand, or against a different one?

---

## 6. Worked example

### 6.1 The question

A sponsor is designing a two-arm superiority trial of a new therapy against standard care for a binary endpoint: response at 24 weeks. The assumptions are a 30 percent response on standard care and a 42 percent response on the new therapy, a two-sided alpha of 0.05, and 80 percent power. One interim analysis is planned at 50 percent information.

The team must determine the sample size, the effect of dropout on power, and how the ITT and per-protocol results could diverge.

### 6.2 Sample size

For a two-proportion comparison with equal allocation:

n per group = (z_{1−α/2} + z_{1−β})² × [p₁(1−p₁) + p₂(1−p₂)] / (p₂ − p₁)²

- z_{1−α/2} = 1.96, z_{1−β} = 0.8416, so the sum is 2.8016 and the square is 7.849.
- p₁(1−p₁) = 0.30 × 0.70 = 0.21.
- p₂(1−p₂) = 0.42 × 0.58 = 0.2436.
- Sum of variances = 0.4536.
- Difference squared = (0.12)² = 0.0144.
- n = 7.849 × 0.4536 / 0.0144 = 3.5603 / 0.0144 = 247.2.

So 248 per group, or 496 participants, before allowing for dropout.

With 10 percent dropout: 248 / 0.90 = 275.6, so 276 per group and 552 total.

Check: 276 × 2 = 552. 552 × 0.90 = 496.8, which is just above the required 496 evaluable participants.

### 6.3 Effect of dropout on power

If dropout is 12 percent rather than 10 percent, the evaluable sample is 552 × 0.88 = 485.8, or about 243 per group.

Standard error under the alternative: sqrt(0.4536 / 243) = sqrt(0.0018667) = 0.043206.

z = 0.12 / 0.043206 = 2.777.

Power = Φ(2.777 − 1.96) = Φ(0.817) = 0.793, or about 79 percent.

Check: 552 × 0.88 = 485.76. 485.76 / 2 = 242.88, so 243 per group. The power falls from 80 to 79 percent, which is a small change for a 2 percentage point increase in dropout. The larger risk is not the power loss but the differential dropout discussed next.

### 6.4 Interim analysis and alpha spending

With one interim at 50 percent information and an O'Brien-Fleming boundary, the nominal two-sided alpha at the interim is approximately 0.005, and at the final analysis approximately 0.048. These are approximations from standard boundary tables and should be computed with the actual spending function and information fractions.

The practical consequence: an interim result at p = 0.01 does not cross the boundary, even though it is below 0.05. A team that stops at p < 0.05 at an interim inflates the type I error, often substantially.

The other consequence is that the final analysis threshold is slightly below 0.05, so a final p-value of 0.049 may not be significant under the spending plan. This surprises teams that are used to a fixed 0.05.

### 6.5 Results and the ITT–PP divergence

At the final analysis, 552 participants were randomized, 276 per arm.

| Population | Arm | n | Responders | Rate |
|---|---|---:|---:|---:|
| ITT | Control | 276 | 83 | 30.1% |
| ITT | Treatment | 276 | 116 | 42.0% |
| PP | Control | 260 | 78 | 30.0% |
| PP | Treatment | 245 | 110 | 44.9% |

Check the ITT rates: 83 / 276 = 0.3007. 116 / 276 = 0.4203. Difference = 11.96 percentage points.
Check the PP rates: 78 / 260 = 0.3000. 110 / 245 = 0.4490. Difference = 14.90 percentage points.

The PP population excluded 47 participants: 16 from control and 31 from treatment. That asymmetry is the key fact.

- Control exclusions: 16, of whom 5 responded, so 5 / 16 = 31.3%.
- Treatment exclusions: 31, of whom 6 responded, so 6 / 31 = 19.4%.

Check the PP numerators: 83 − 5 = 78. 116 − 6 = 110. Check the denominators: 276 − 16 = 260. 276 − 31 = 245.

Because the excluded treatment participants responded at a much lower rate than the treatment arm average, removing them raises the treatment response rate from 42.0 to 44.9 percent and widens the difference from 12.0 to 14.9 percentage points. The per-protocol analysis overstates the effect by about 3 percentage points.

This is the classic asymmetry: deviations are not random, and they are often related to outcome. A PP analysis that removes participants who did poorly in the treatment arm will inflate the estimate.

### 6.6 Number needed to treat

NNT = 1 / absolute risk difference = 1 / 0.1196 = 8.36, so about 8.4 participants treated per additional responder.

Check: 1 / 0.1196 = 8.361.

If the PP estimate were reported, NNT would be 1 / 0.149 = 6.7, a 20 percent improvement in apparent efficiency that is entirely an artefact of the population definition.

### 6.7 Multiplicity across secondary endpoints

Suppose there are four secondary endpoints tested at 0.05 with no adjustment. Under the global null, the probability of at least one false positive is 1 − 0.95⁴ = 0.185, or 18.5%.

Check: 0.95⁴ = 0.8145. 1 − 0.8145 = 0.1855.

With a Bonferroni correction at 0.05/4 = 0.0125 per endpoint, the family-wise error is controlled at 5 percent. A hierarchical gatekeeping procedure can be more powerful if the endpoints are ordered by importance and each is tested only if the previous one is significant.

### 6.8 What the example establishes

- The sample size calculation is arithmetic, and the assumptions dominate the result: a 2 point change in the assumed control rate changes the sample size substantially.
- Dropout reduces power modestly but can bias the estimate severely if it is differential.
- The ITT and PP estimates answer different questions and can differ by several percentage points.
- Interim boundaries are not 0.05, and using 0.05 at an interim inflates the type I error.
- Multiplicity control is required for secondary claims, and the method affects power.

### 6.9 Variant case: 8 percent of the primary endpoint is missing

The primary analysis assumed a complete endpoint. Suppose 8 percent of participants have no 24-week response assessment: 44 of 552, evenly split at 22 per arm.

Complete-case analysis:

| Arm | Randomized | Assessed | Responders | Rate |
|---|---:|---:|---:|---:|
| Control | 276 | 254 | 83 | 32.7% |
| Treatment | 276 | 254 | 116 | 45.7% |

Check: 276 − 22 = 254 per arm. 83 / 254 = 0.3268. 116 / 254 = 0.4567. Difference = 12.99 points.

The complete-case difference of 13.0 points is larger than the 12.0 points assumed in the sample size calculation, because removing 22 non-responders from the control arm raised the control rate. That is a warning sign in itself: missingness in the control arm was associated with non-response.

Worst-case and best-case bounds under the treatment-policy estimand:

| Scenario | Control responders | Treatment responders | Control rate | Treatment rate | Difference |
|---|---:|---:|---:|---:|---:|
| Worst case for treatment | 105 | 116 | 38.0% | 42.0% | 4.0 points |
| Complete case | 83 | 116 | 32.7% | 45.7% | 13.0 points |
| Best case for treatment | 83 | 138 | 30.1% | 50.0% | 19.9 points |

Check the worst case: control gains all 22 missing as responders (83 + 22 = 105), treatment gains none (116). 105 / 276 = 0.3804. 116 / 276 = 0.4203. Difference = 3.99 points.
Check the best case: control gains none (83), treatment gains all 22 (116 + 22 = 138). 83 / 276 = 0.3007. 138 / 276 = 0.5000. Difference = 19.93 points.

### 6.10 What the variant changes, and the tipping point

The direction of the conclusion survives: even the worst case leaves a 4.0 percentage point advantage, so superiority is robust to any pattern of missingness at this level.

The magnitude does not survive. The number needed to treat ranges from 5.0 in the best case to 25.1 in the worst case, a fivefold range.

Check: 1 / 0.1993 = 5.02. 1 / 0.0399 = 25.06.

The tipping point is the amount of missingness at which the worst case reaches zero difference. Control would need to reach 116 responders, which requires 33 of its missing participants to be responders, so the tipping point is 33 missing per arm, or 66 of 552.

Check: 66 / 552 = 0.1196, or 12.0%.

At 8 percent missing the trial is robust; at 12 percent it is not. That is the number a data monitoring committee needs, and it is more useful than either the complete-case estimate or a p-value.

### 6.11 Which estimate I would report, and why

Report three numbers, in this order:

- **The primary estimate under the pre-specified method**, with its confidence interval. If the SAP specified multiple imputation, that is the primary, and the complete-case figure is a sensitivity analysis.
- **The worst-case bound**, stated as a bound rather than an estimate, because it is the most conservative defensible reading.
- **The tipping point**, because it converts the missingness into a decision rule the committee can monitor.

Reporting only the complete-case estimate is the common failure, because it is the easiest to compute and the most flattering. Reporting only the worst case is the opposite failure, because it discards information from the observed participants and overstates the uncertainty.

Four checks on any missing-data sensitivity:

- Is the missingness balanced across arms? If not, the complete-case estimate is biased and the bound is the primary.
- Is the tipping point above the expected missingness? If not, the trial is fragile.
- Does the imputation model include the outcome and the covariates that predict missingness?
- Is the sensitivity analysis labelled as a bound or as an estimate? A bound presented as an estimate overstates precision.

### 6.12 What a reviewer would ask

1. **Why 30 and 42 percent?** They should come from prior data, a phase 2 estimate, or a clinically meaningful difference. An assumption with no basis produces a sample size with no basis.
2. **Is the dropout assumption supported?** It should come from similar trials, and it should be checked against actual dropout at the interim.
3. **Was the SAP finalized before unblinding?** If not, the primary analysis is at risk of being data-driven.
4. **Why is the PP difference larger?** Because exclusions were asymmetric and related to response. The ITT estimate is the one that preserves randomization, and the PP result is a sensitivity analysis that here suggests the ITT may understate efficacy in completers while overstating it in the population actually treated.

---

## 7. Code, query, or calculation

### 7.1 Python: sample size, power, and the ITT–PP comparison

Assumptions: Python 3.11 or later; scipy and numpy installed.

```python
import numpy as np
from scipy.stats import norm

def n_per_group(p1, p2, alpha=0.05, power=0.80):
    """Equal-allocation sample size for a two-proportion test."""
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    var = p1 * (1 - p1) + p2 * (1 - p2)
    return (z_a + z_b) ** 2 * var / (p2 - p1) ** 2

def power_two_prop(p1, p2, n, alpha=0.05):
    """Approximate power for equal allocation."""
    var = p1 * (1 - p1) / n + p2 * (1 - p2) / n
    se = np.sqrt(var)
    z = abs(p2 - p1) / se - norm.ppf(1 - alpha / 2)
    return norm.cdf(z)

n = n_per_group(0.30, 0.42)
print(f"n per group (no dropout): {n:.1f} -> {int(np.ceil(n))}")
n_drop = int(np.ceil(np.ceil(n) / 0.90))
print(f"n per group with 10% dropout: {n_drop}")
print(f"total randomized: {2 * n_drop}")

for d in (0.10, 0.12, 0.15, 0.20):
    evaluable = 2 * n_drop * (1 - d) / 2
    print(f"dropout {d*100:4.0f}% -> n/group {evaluable:6.1f} "
          f"power {power_two_prop(0.30, 0.42, evaluable)*100:5.1f}%")

print()
itt = {"control": (83, 276), "treatment": (116, 276)}
pp = {"control": (78, 260), "treatment": (110, 245)}
for name, d in (("ITT", itt), ("PP", pp)):
    c = d["control"][0] / d["control"][1]
    t = d["treatment"][0] / d["treatment"][1]
    print(f"{name}: control {c*100:.1f}%  treatment {t*100:.1f}%  "
          f"diff {(t-c)*100:.2f} pp  NNT {1/(t-c):.1f}")
```

Expected output:

```text
n per group (no dropout): 247.2 -> 248
n per group with 10% dropout: 276
total randomized: 552
dropout   10% -> n/group  248.4 power  80.1%
dropout   12% -> n/group  242.9 power  79.3%
dropout   15% -> n/group  234.6 power  77.7%
dropout   20% -> n/group  220.8 power  74.1%

ITT: control 30.1%  treatment 42.0%  diff 11.96 pp  NNT 8.4
PP: control 30.0%  treatment 44.9%  diff 14.90 pp  NNT 6.7
```

The dropout rows show that a 10 percentage point increase in dropout costs about 6 percentage points of power in this design, which is a useful rule of thumb for planning: dropout is not a minor adjustment.

### 7.2 Python: alpha spending approximation

```python
from scipy.stats import norm

def obf_boundary(alpha=0.05, t=0.5):
    """Approximate two-sided O'Brien-Fleming nominal alpha at information t.

    Uses the standard approximation alpha_t = 2*(1 - Phi(z_{alpha/2}/sqrt(t))).
    """
    z = norm.ppf(1 - alpha / 2)
    return 2 * (1 - norm.cdf(z / np.sqrt(t)))

print(f"Interim (t=0.50) nominal two-sided alpha: {obf_boundary(0.05, 0.5):.4f}")
print(f"Final   (t=1.00) nominal two-sided alpha: {obf_boundary(0.05, 1.0):.4f}")
```

Expected output:

```text
Interim (t=0.50) nominal two-sided alpha: 0.0054
Final   (t=1.00) nominal two-sided alpha: 0.0480
```

These values match the approximations quoted in section 6.4. The approximation is standard and is adequate for planning; a confirmatory analysis uses the exact spending function specified in the SAP.

### 7.3 A hand calculation for an interview

Given 116 responders in 276 treated and 83 in 276 controls:

1. Treatment rate = 116/276 = 0.4203.
2. Control rate = 83/276 = 0.3007.
3. Absolute difference = 0.1196, or 12.0 percentage points.
4. NNT = 1/0.1196 = 8.4.
5. Standard error of the difference = sqrt(0.4203×0.5797/276 + 0.3007×0.6993/276) = sqrt(0.0008827 + 0.0007619) = sqrt(0.0016446) = 0.04055.
6. z = 0.1196/0.04055 = 2.949.
7. Two-sided p ≈ 0.0032.

So the result is significant at the conventional threshold and would also cross a final O'Brien-Fleming boundary of about 0.048.

---

## 8. Data, standards, and regulatory context

### 8.1 The CDISC standards

| Standard | Scope | Version context | What it constrains |
|---|---|---|---|
| SDTM | Tabulation model for submitted data | SDTM v2.0 pairs with SDTMIG v3.4 | Structure of submitted domains |
| SDTMIG | Implementation guide | v3.4 published 29 November 2021 | Domain content and variables |
| ADaM | Analysis data model | ADaMIG 1.x line | Analysis-ready datasets |
| ADSL | Subject-level analysis dataset | Part of ADaM | One row per subject, with flags |
| define.xml | Metadata for the submission | define-xml 2.x | Machine-readable dataset descriptions |
| SEND | Nonclinical data | SENDIG 3.1 and later | Animal study data |
| Controlled terminology | Codelists for SDTM and ADaM | Released quarterly | Permitted values |
| Dataset-JSON | A proposed replacement for XPT v5 | Under evaluation | Transport format |

### 8.2 The FDA requirement timeline

The Federal Register notice of 13 December 2023 (88 FR 86336) announced that support for SDTM v2.0, SDTMIG v3.4, and SENDIG-Genetox v1.0 began on 13 December 2023, and that their use became **required on 15 March 2025** for new drug applications, abbreviated new drug applications, certain biologics license applications, and certain investigational new drug applications. Support for SDTMIG v3.2 ended on 13 December 2023.

This is a binding requirement with a date, and it is the kind of fact that decides whether a submission is accepted for review. The FDA's Study Data Technical Conformance Guide (sdTCG) supplements the requirement with technical specifications, and it is revised frequently; the 2024 editions were v5.8 (September 2024) and v5.9 (October 2024), with later revisions following.

The sdTCG states several expectations that matter to an analyst:

- SDTMIG should be followed unless otherwise indicated.
- ADSL is required for every study.
- A separate define.xml is expected per dataset type.
- Technical rejection criteria are tied to study start dates, so older studies are held to the standard in force when they began.

### 8.3 Good Clinical Practice and electronic records

- **ICH E6(R2)** is the current broadly adopted Good Clinical Practice guideline, with E6(R3) finalized in 2025 and being adopted progressively. It governs trial conduct, monitoring, and data integrity.
- **21 CFR Part 11** governs electronic records and electronic signatures in FDA-regulated activities. It requires audit trails, access controls, and validated systems.
- **ICH E9(R1)** defines the estimand framework and is the reference for the five attributes in section 3.2.
- **ICH E8(R1)** covers general considerations for clinical studies, including quality by design.
- **EU Clinical Trials Regulation (EU) 536/2014** governs trials in the EU through the Clinical Trials Information System, with transparency requirements for the protocol and results.

### 8.4 Transparency and data sharing

- **ClinicalTrials.gov** registration is required under FDAAA 801 for applicable trials, and results submission is required within specified timelines.
- **EU CTR** requires summary results in the Clinical Trials Information System within specified timelines.
- **Individual participant data sharing** is available through platforms such as Vivli, the YODA Project, and the Clinical Study Data Request consortium, on sponsor-specific terms.
- **ICMJE** requires a data sharing statement for published trials.

The practical point for an analyst is that a published trial result and the underlying data are different artefacts, and access to the second is a separate negotiation with separate terms.

### 8.5 What is binding and what is convention

Binding:

- The FDA SDTM requirement, with the 15 March 2025 date.
- 21 CFR Part 11 for electronic records.
- GCP for trials intended to support a marketing application.
- Registration and results reporting requirements.

Convention:

- Using ADaM for analysis datasets is a strong convention and is expected by FDA reviewers, but the specific ADaM structure is less prescriptive than SDTM.
- Reporting the estimand is becoming standard and is not uniformly required in every jurisdiction.
- The choice between O'Brien-Fleming and Pocock boundaries is a design choice, not a rule.

### 8.6 Trial data versus real-world data in regulatory decisions

The FDA's real-world evidence program accepts RWE for some purposes, and the comparison with trial data is explicit in the guidance: RWE is most credible when the data are fit for the question, the design supports causal inference, and the results are reproducible.

A trial dataset has an advantage that RWE cannot match: randomization. It has a disadvantage that RWE does not: a narrow population. A submission that uses both is stronger than either alone, provided the populations and endpoints are comparable enough to interpret together.

### 8.7 What a reviewer receives

The submission package is not the database. It contains:

- **SDTM datasets**, one file per domain, in a transport format.
- **ADaM datasets**, including ADSL and the analysis datasets.
- **define.xml**, describing every dataset and variable.
- **A reviewer's guide**, explaining the structure and any unusual choices.
- **The analysis programs** in some jurisdictions, and the outputs.
- **The clinical study report**, which narrates the results.

An analyst who has only the clinical study report has the conclusions, not the data. An analyst who has the datasets can check the conclusions, which is why data transparency matters.

A practical note on the transport format: XPT v5 is the long-standing default and is being evaluated against Dataset-JSON as a successor. The format does not change the content, and it does change the tooling, so a team building a pipeline should support both during the transition.

---

## 9. Industry practice and case evidence

### 8.8 What a validator does not check

A conformance validator checks structure: variable names, types, codelists, and cross-references. It does not check four things that matter more:

- **Whether the derivation is correct.** A variable can conform and be computed wrongly.
- **Whether the population flag is right.** A conformant flag can encode the wrong population.
- **Whether the endpoint matches the SAP.** A conformant dataset can implement a different analysis.
- **Whether the numbers reconcile.** A conformant submission can disagree with the clinical study report.

Conformance is necessary and is not evidence of correctness. That is why the reconciliation steps in section 5 exist alongside the validator run.

### 9.1 The economics of trial data

Trial data is expensive per record because of the surrounding infrastructure, not because of the data itself. The costs that dominate are site monitoring, query resolution, adjudication of endpoints, and the statistical and programming effort for the submission package.

The analytic consequence is that trial datasets are small relative to their cost, and the temptation to extract more analyses from them is strong. That temptation is the source of most multiplicity problems: a dataset with 500 participants and 40 endpoints will produce a "significant" finding by chance, and the SAP exists to prevent the organization from believing it.

### 9.2 What regulators actually reject

Technical rejection is a documented and common failure mode. The reasons cluster:

- Datasets that do not conform to the required SDTMIG version.
- Missing define.xml or a define.xml that does not describe the datasets.
- Inconsistent controlled terminology versions across datasets.
- ADaM datasets without a documented derivation.
- Discrepancies between the tabulated data and the analysis results.

None of these is a scientific disagreement. All of them delay review, and some cause a refuse-to-file. The practical lesson is that conformance is a prerequisite, not a polish step.

### 9.3 The estimand shift in practice

Since ICH E9(R1), sponsors have had to state how intercurrent events are handled. The practical effect is that protocols now contain explicit statements such as "treatment policy" or "hypothetical" strategies, and the analysis must match.

The industry friction is that different functions interpret the strategies differently. A clinical team may think "treatment policy" means analyzing everyone as randomized, while a statistical team may mean something more specific about data after discontinuation. The estimand statement is only useful if it is operationalized in the SAP and the ADaM derivations.

### 9.4 Decentralized and hybrid trials

Decentralized elements, including remote consent, ePRO, and home nursing, change the data provenance. A blood draw at home produces a result with a different chain of custody than a draw at a site, and the difference must be recorded.

The analytic risk is that decentralized and site-collected data are pooled without a source variable. If the decentralized subgroup differs in quality or in population, pooling creates a batch effect that looks like a treatment effect.

### 9.5 What a mature trial data team does

- **Maintains a data management plan** that specifies the source, the edit checks, and the query workflow.
- **Freezes the SAP before unblinding** and records the date.
- **Classifies deviations before unblinding** against pre-specified criteria.
- **Validates the submission package** against the current validator before delivery.
- **Records every version** of the protocol, SAP, terminology, and dictionary.
- **Publishes results** and makes individual participant data available on stated terms.

These six practices are the difference between a submission that is reviewed on its science and one that is delayed by its paperwork.

### 9.6 Why trial data is still the gold standard

Randomization is the only design that addresses confounding by unmeasured variables, and a trial is the only setting in which the intervention, the population, and the endpoint are all specified in advance. That combination is why a trial result can support a causal claim and an observational result usually cannot.

The gold standard is not a claim about data quality in the abstract. It is a claim about design. A trial with poor data quality is worse than a good observational study, and a well-conducted observational study can be more generalizable than a narrow trial. The design and the question determine the evidence, not the label.

### 9.7 Who does what in a trial data organization

A trial produces data through a division of labour that an analyst must understand:

| Function | Owns | Produces | Analyst dependency |
|---|---|---|---|
| Clinical data management | EDC build, edit checks, queries | Clean database | Data quality and query closure |
| Biostatistics | SAP, analysis, interpretation | Estimates and tables | The primary result |
| Statistical programming | SDTM, ADaM, TLF | Submission datasets | Every derived variable |
| Medical monitoring | Safety, deviations, coding | Safety narrative, coded AEs | Safety interpretation |
| Regulatory affairs | Submission package | eCTD | Conformance and timing |
| Clinical operations | Sites, enrollment | Enrollment and visit data | Denominator and flow |

The most common organizational failure is a gap between data management and programming, where a query resolution is not reflected in the SDTM conversion. The result is a dataset that disagrees with the database, and the disagreement is found by a reviewer rather than by the team.

### 9.8 The cost of a late SAP change

A SAP change after unblinding is not free. It requires a protocol or SAP amendment, a documented rationale, a re-run of the analysis, and often a conversation with the regulator if the change affects the primary endpoint.

The cost is not only administrative. A late change weakens the confirmatory status of the analysis, because the change may have been informed by the data. Regulators do not assume bad faith, and they do expect the date of finalization to be documented, which is why the date matters more than the content in many reviews.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| ITT mislabelled | Exclusions applied to "ITT" | Withdrawals removed | Broken randomization | Rebuild from the randomization list |
| Post hoc mITT | Definition appears after unblinding | Subset chosen on data | Biased estimate | Pre-specify; label as exploratory |
| PP used as primary | Only the PP result is reported | Deviations removed | Inflated effect | Report ITT primary; PP as sensitivity |
| Differential dropout | Dropout differs by arm | Toxicity or lack of efficacy | Biased estimate | Sensitivity analyses; estimand |
| Unadjusted stratification | Stratified randomization, unstratified analysis | Ignored design | Slight bias, wrong variance | Adjust for factors |
| Multiplicity ignored | Many secondary endpoints at 0.05 | Repeated testing | False secondary claims | Hierarchy or gatekeeping |
| Interim at 0.05 | Early stop at p<0.05 | No alpha spending | Inflated type I error | Pre-specified boundary |
| Terminology mismatch | Mixed codelist versions | Quarterly updates | Validation errors | Pin and record versions |
| SDTM non-conformance | Validator errors | Wrong IG version | Technical rejection | Validate before submission |
| Undocumented derivation | ADaM variable with no definition | Programming shortcuts | Reviewer cannot reproduce | define.xml for every variable |
| Query treated as deviation | Confusion in the deviation log | Process misunderstanding | Wrong PP population | Separate logs and training |
| Unblinded analysis | Analyst had access to assignment | Poor access control | Unconscious bias | Firewall; documented roles |

### 9.9 What a submission readiness review checks

| Area | Check | Failure consequence |
|---|---|---|
| Protocol-SAP traceability | Every SAP section maps to a protocol version | Statistical review finding |
| Population flags | ITT, safety, and PP counts reconcile | Uninterpretable analysis |
| SDTM conformance | Validator errors at zero | Technical rejection |
| Define.xml | Every dataset and variable described | Reviewer cannot reproduce |
| Terminology versions | One version across all datasets | Inconsistency findings |
| Dictionary versions | MedDRA and WHO Drug pinned | AE counts change |
| Data cut and lock | Documented and consistent | Timeline ambiguity |
| Unblinding date | Recorded relative to SAP finalization | Confirmatory status questioned |

The review is a checklist, and the value is that it happens before submission rather than during review. A team that runs it once will find between five and twenty issues on a typical study.

### 9.10 What a sponsor should ask a CRO

| Question | Why | Weak answer |
|---|---|---|
| Who owns the data and the derived datasets? | Determines portability | "We do" |
| What is the SDTM conversion timeline after lock? | Determines submission date | "Standard" |
| Which validator version will be used? | Determines conformance target | "Latest" |
| How are protocol deviations classified and by whom? | Determines the PP population | "By the site" |
| What is the query closure rate at lock? | Determines data quality | "High" |
| Can we audit the ADaM derivations? | Determines reviewer readiness | "Yes, on request" |

The first question is the one that causes the most trouble later. A sponsor that does not own the derived datasets cannot move to another vendor without re-deriving everything.

### 10.1 The three red flags

1. **A population definition that appears only in the results.** If the definition is not in the SAP before unblinding, it is data-driven.
2. **A single analysis with no sensitivity analyses.** A trial result that is robust should survive alternative populations; a result that is not reported under alternatives is a result the sponsor did not want to test.
3. **A submission with validator errors.** Technical rejection is avoidable and signals a process gap that likely extends beyond the validator.

### 10.2 The "significant but not confirmed" problem

A trial can meet its primary endpoint and fail to change practice, because the endpoint was a surrogate, the population was narrow, or the effect was not clinically meaningful. The data cannot resolve that; only a decision about the meaningfulness of the effect can.

The analyst's contribution is to report the absolute effect, the NNT, and the population, so the decision is made on the right quantity rather than on a p-value.

### 10.3 The protocol-SAP drift

A subtle failure is drift between the protocol and the SAP across amendments. The protocol says one thing about the primary endpoint, an amendment changes it, the SAP implements a third version, and the analysis follows the SAP.

The failure is silent because each document is internally consistent. It surfaces in a review when a reviewer compares the protocol in force at the data cut to the analysis actually performed.

The mitigation is a traceability matrix: each SAP section mapped to the protocol section it implements, with the version. It takes a day to build and prevents the single most damaging finding in a statistical review.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Randomization was correctly implemented | Causal interpretation | Allocation errors, stratification mistakes | Compare to the randomization list; check balance |
| The SAP was finalized before unblinding | Confirmatory status | Late SAP changes | Document dates |
| Deviations are unrelated to outcome | PP interpretation | Deviations driven by response | Compare excluded and included outcomes |
| Dropout is unrelated to outcome | Complete-case analysis | Differential dropout | Sensitivity analyses; estimand |
| The endpoint measures what matters | Clinical relevance | Surrogate endpoint | Discuss; external evidence |
| The population represents the target | Generalization | Narrow eligibility | Compare to the target population |
| The control arm reflects practice | Relevance | Outdated comparator | Justify the comparator |
| Data quality is adequate | Any inference | Poor site performance | Site-level quality metrics |

### 11.1 Limits that are inherent

- **The population is selected.** Eligibility criteria exclude the patients most likely to be complex.
- **The duration is bounded.** Long-term effects are not observed.
- **The comparator is chosen.** A trial against placebo does not answer a question about active comparators.
- **The endpoint is a choice.** A surrogate may not translate into clinical benefit.
- **The setting is artificial.** Protocol visits and monitoring change behaviour.

### 11.2 How to state the trial limitation

"The trial randomized 552 adults with the eligibility criteria in the protocol; 47 participants had major deviations, 31 of them in the treatment arm, and the per-protocol analysis excluding them estimated a 14.9 percentage point difference against 12.0 in the intention-to-treat analysis. The trial cannot speak to patients excluded by the eligibility criteria, to effects beyond 24 weeks, or to comparisons with active alternatives not studied."

That paragraph gives the reader the population, the deviation pattern, both estimates, and the boundaries.

### 11.3 The question a trial cannot answer

A trial cannot tell you what will happen in your clinic, because your patients are not the trial's patients. That is not a defect to be engineered away; it is the price of the design that makes the causal claim possible.

The practical response is not to discount the trial. It is to ask three questions before generalizing:

- Which eligibility criteria would exclude most of my patients?
- Which protocol procedures would change behaviour in my setting?
- Which comparator would I actually use?

If the answers differ substantially, the trial result is a starting point for a local evidence assessment, not an answer.

### 11.4 Why the estimand matters for generalization

Two trials of the same drug can report different effects because they asked different estimands. A treatment-policy estimand includes the effect of discontinuation and is closer to real-world use; a hypothetical estimand estimates the effect if everyone adhered and is closer to a mechanistic question.

When comparing a trial to real-world data, the estimand must be aligned first. A trial with a hypothetical estimand and an RWE study with a treatment-policy estimand will disagree, and the disagreement is expected rather than informative.

This is the most common error in trial-to-RWE comparisons, and it is invisible unless the estimand is stated.

---

## 12. Exercises

### Exercise 1 — Trace the chain (recall, 20 minutes)

List the stages from source data to submission and name the standard or document governing each.

Deliverable: a six-stage table.

Acceptance criteria: each stage has a named artefact; the SDTM and ADaM stages are distinguished.

### Exercise 2 — Sample size (calculation, 40 minutes)

Compute the sample size for p₁ = 0.25, p₂ = 0.38, alpha 0.05, power 90 percent, with 8 percent dropout.

Deliverable: the calculation and the final randomized total.

Acceptance criteria: formula stated; arithmetic correct; dropout applied to the per-group size before doubling.

### Exercise 3 — ITT versus PP (calculation, 45 minutes)

Using the counts in section 6.5, recompute both estimates and the NNT for each. Then recompute the PP estimate if the 31 excluded treatment participants had responded at the arm average.

Deliverable: three estimates and a one-paragraph interpretation.

Acceptance criteria: arithmetic correct; the interpretation states why the PP estimate changes.

### Exercise 4 — Alpha spending (calculation, 30 minutes)

Using the approximation in section 7.2, compute the interim boundary at information fractions 0.4 and 0.6. Explain how the boundary changes with the information fraction.

Deliverable: two boundaries and a short explanation.

Acceptance criteria: arithmetic correct; the explanation states that earlier looks require more extreme evidence.

### Exercise 5 — Estimand (design, 45 minutes)

Write an estimand for a trial of a maintenance therapy in COPD with exacerbations as the endpoint, handling the intercurrent event of treatment discontinuation.

Deliverable: the five attributes, stated explicitly.

Acceptance criteria: each attribute is specific; the intercurrent event strategy is named; the summary measure matches the endpoint type.

### Exercise 6 — Critique (critique, 60 minutes)

Find a published trial with a per-protocol primary analysis or an unclear population definition. Write a 400-word critique.

Deliverable: a 400-word critique with one proposed reanalysis.

Acceptance criteria: the critique names the specific population issue; the reanalysis is feasible from the published data or the CSR.

### Exercise 7 — Provenance record (design, 60 minutes)

Write a provenance record for a trial analysis: protocol version, SAP version and date, data cut, SDTM/ADaM versions, terminology and dictionary versions, and snapshot hash.

Deliverable: a one-page record.

Acceptance criteria: a second analyst could reproduce the analysis; every version has a date; the unblinding date is recorded.

### Exercise 8 — Trial versus RWE (design, 45 minutes)

For an effectiveness question in a broad population, compare a trial and a claims-based RWE study on population, endpoint, follow-up, and confounding control.

Deliverable: a four-row comparison and a recommendation.

Acceptance criteria: the recommendation names which source answers which part of the question; no source is treated as universally superior.

### Exercise 9 — Deviation classification (application, 45 minutes)

Given five deviation scenarios, classify each as major or minor against pre-specified criteria you write, and state the effect on the PP population.

Deliverable: a five-row table with criteria and consequences.

Acceptance criteria: the criteria are written before classifying; the consequence is specific to the endpoint.

### Exercise 10 — Three short analyses (design and application, 120 minutes total)

This exercise has three parts. Each has its own deliverable and acceptance criteria.

**Part A. Traceability matrix (45 minutes).** Build a one-page traceability matrix mapping five SAP sections to the protocol sections they implement, with version numbers.

Deliverable: a five-row matrix.

Acceptance criteria: every row names a version; at least one row identifies an ambiguity that would need resolution.

**Part B. Safety table reading (30 minutes).** Given an adverse event table with system organ class, preferred term, and counts by arm, identify the three terms with the largest absolute difference and compute the risk difference for each.

Deliverable: a three-row table with risk differences.

Acceptance criteria: denominators are the safety population; the risk difference is computed per 100 participants; no causal language is used.

**Part C. Missing data sensitivity (45 minutes).** Assume 8 percent of the primary endpoint is missing, balanced across arms. Compute the primary estimate under complete-case analysis and under a worst-case imputation where all missing treatment participants are non-responders. Report the range.

Deliverable: two estimates and a range.

Acceptance criteria: arithmetic correct; the worst case is described as a bound, not as an estimate.

---

## 13. Self-check answers

**A1. Compute the sample size for p₁ = 0.30, p₂ = 0.42, alpha 0.05, power 80 percent.**
n = (1.96 + 0.8416)² × (0.21 + 0.2436) / 0.0144 = 7.849 × 0.4536 / 0.0144 = 247.2, so 248 per group.

**A2. What is the randomized total with 10 percent dropout?**
248 / 0.90 = 275.6, so 276 per group and 552 total.

**A3. Compute the ITT difference for 116/276 versus 83/276.**
42.03% − 30.07% = 11.96 percentage points.

**A4. Compute the NNT for an 11.96 percentage point difference.**
1 / 0.1196 = 8.4.

**A5. Why is the PP estimate larger than the ITT estimate here?**
Because the excluded participants were disproportionately in the treatment arm (31 of 47) and responded at a lower rate (19.4%) than the treatment arm average (42.0%). Removing them raises the treatment rate and widens the difference.

**A6. What is the nominal alpha at a 50 percent information interim under O'Brien-Fleming?**
Approximately 0.0054 two-sided, against 0.048 at the final analysis. The exact value depends on the spending function.

**A7. Why does an interim at p < 0.05 inflate the type I error?**
Because the trial has two chances to cross 0.05, and the probability of crossing at least once under the null is greater than 0.05. Alpha spending allocates the total error across looks to keep the overall rate at 0.05.

**A8. What are the five estimand attributes?**
Treatment condition, population, variable, intercurrent events, and population-level summary.

**A9. What is the difference between a query and a protocol deviation?**
A query is a data clarification request about a discrepancy. A deviation is a departure from the protocol. Resolving a query does not resolve a deviation, and a deviation may exist with no query.

**A10. Why is the safety population defined as treated rather than randomized?**
Because safety conclusions apply to participants who were exposed. A participant randomized but never dosed cannot contribute to an adverse event rate for the drug.

**A11. What changed on 15 March 2025?**
Use of SDTM v2.0 and SDTMIG v3.4 became required for specified FDA submissions, per the Federal Register notice of 13 December 2023.

**A12. Rubric for exercise 7 (provenance record).**
A strong record names the protocol version, the SAP version and its finalization date relative to unblinding, the data cut, the database lock, the SDTMIG and ADaM versions, the controlled terminology and dictionary versions, and the snapshot hash. A weak record names the study and the date only.

---

## 14. Interview questions

**Q1. What is the difference between ITT and per-protocol, and which is primary?**
ITT analyzes participants as randomized and preserves the benefit of randomization; it is the primary basis for a superiority claim. Per-protocol analyzes completers without major deviations and is a sensitivity analysis that answers a different question.

**Q2. Why can a per-protocol analysis be biased?**
Because deviations and exclusions are not random. Participants who discontinue may do so because of lack of efficacy or toxicity, both of which are related to the outcome. Removing them changes the estimate in a direction determined by the reason for discontinuation.

**Q3. What is an estimand and why does it matter?**
A precise definition of the quantity being estimated, with treatment, population, variable, intercurrent events, and summary. It matters because two analyses of the same data can answer different questions, and the estimand makes the question explicit.

**Q4. How would you handle missing data in a trial?**
A strong answer: specify the primary approach in the SAP, justify it against the estimand, and run sensitivity analyses under alternative assumptions, including a worst-case or tipping-point analysis. Weak answers use complete-case analysis without comment.

**Q5. Why is multiplicity control necessary for secondary endpoints?**
Because testing many endpoints at 0.05 produces false positives by construction. A hierarchy, gatekeeping, or alpha allocation preserves the family-wise error rate while allowing confirmatory claims.

**Q6. What is alpha spending?**
A pre-specified allocation of the total type I error across interim and final analyses, so that the overall error rate remains at the nominal level despite multiple looks.

**Q7. Why is SDTM not analysis-ready?**
Because SDTM tabulates collected data in a standard structure without derived analysis variables. Analysis requires ADaM datasets with baselines, change from baseline, and analysis flags.

**Q8. What causes a technical rejection?**
Non-conformant datasets, missing or incomplete define.xml, inconsistent terminology versions, and discrepancies between tabulated and analyzed data. These are process failures, not scientific ones.

**Q9. How does a trial result differ from a real-world result?**
Population, adherence, monitoring, comparator, endpoint standardization, and duration all differ. A trial answers whether an intervention works under protocol conditions; RWE asks what happens in routine care.

**Q10. What would you check first in an unfamiliar trial dataset?**
A strong answer: the protocol and SAP versions, the data cut, and the population definitions. Those determine what any analysis means.

**Q11. Why is differential dropout more dangerous than overall dropout?**
Because overall dropout reduces power, while differential dropout biases the estimate. A trial can retain 85 percent of participants and still be biased if the 15 percent lost differ by arm and by outcome.

**Q12. What would you put in a trial data provenance record?**
Protocol version, SAP version and date, data cut and lock, randomization method, SDTMIG and ADaM versions, terminology and dictionary versions, snapshot hash, and the unblinding date.

**Q13. A trial meets its primary endpoint with p = 0.04 at an interim. What do you check?**
A strong answer: the pre-specified alpha-spending boundary. At a 50 percent information interim under O'Brien-Fleming, the boundary is about 0.005, so 0.04 does not cross it. Stopping would inflate the type I error.

**Q14. Why might a trial and an RWE study of the same drug disagree?**
A strong answer: different populations, different comparators, different adherence, different endpoints, and different estimands. The estimand difference is the one most often overlooked, and it can explain a disagreement that looks like a data quality problem.

---

## 15. Key terms

- **ADaM** — Analysis Data Model.
- **ADSL** — subject-level analysis dataset.
- **Allocation concealment** — hiding the next assignment.
- **Alpha spending** — allocating type I error across looks.
- **Blinding** — concealment of treatment assignment.
- **CDISC** — Clinical Data Interchange Standards Consortium.
- **CTCAE** — Common Terminology Criteria for Adverse Events.
- **Database lock** — the point the database is frozen.
- **Data cut** — the date defining the data in an analysis.
- **define.xml** — metadata describing submission datasets.
- **DSMB** — Data and Safety Monitoring Board.
- **eCTD** — electronic common technical document.
- **EDC** — electronic data capture.
- **ePRO** — electronic patient-reported outcome.
- **Estimand** — the precise quantity to be estimated.
- **GCP** — Good Clinical Practice.
- **Intercurrent event** — an event affecting interpretation.
- **IRT** — interactive response technology.
- **ITT** — intention-to-treat.
- **MedDRA** — Medical Dictionary for Regulatory Activities.
- **mITT** — modified intention-to-treat.
- **Multiplicity** — control of type I error across tests.
- **NNT** — number needed to treat.
- **O'Brien-Fleming** — a conservative early-stopping boundary.
- **Part 11** — FDA regulation on electronic records.
- **Pocock** — a less conservative early-stopping boundary.
- **PP** — per-protocol.
- **Protocol deviation** — a departure from the protocol.
- **Query** — a data clarification request.
- **Randomization** — assignment by chance.
- **SAP** — statistical analysis plan.
- **SDTM** — Study Data Tabulation Model.
- **SDTMIG** — SDTM Implementation Guide.
- **SDV** — source data verification.
- **SEND** — Standard for Exchange of Nonclinical Data.
- **Surrogate endpoint** — a substitute for a clinical outcome.
- **WHO Drug** — the WHO drug dictionary.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Standards and regulators**

- Federal Register. *Data Standards; Support and Requirement Begins for CDISC Version 2.0 of SDTM, Version 3.4 of SDTMIG, and Version 1.0 of SENDIG-Genetox*, 13 December 2023 (88 FR 86336). https://www.federalregister.gov/documents/2023/12/13/2023-27310/data-standards-support-and-requirement-begins-for-the-clinical-data-interchange-standards-consortium
- CDISC. *SDTMIG v3.4*, published 29 November 2021. https://www.cdisc.org/standards/foundational/sdtmig/sdtmig-v3-4
- US Food and Drug Administration. *Study Data Technical Conformance Guide.* https://www.fda.gov/media/153632/download
- US Food and Drug Administration. *Study Data Standards Resources.* https://www.fda.gov/industry/fda-data-standards-advisory-board/study-data-standards-resources
- US Food and Drug Administration. *Study Data Technical Conformance Guide, working draft v3.4 redline.* https://www.fda.gov/files/for%20industry/published/Study-Data-Technical-Conformance-Guide.pdf
- ICH. *E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials.* (Cite the guideline; see the ICH website.)
- ICH. *E6(R2) Good Clinical Practice.* (Cite the guideline; see the ICH website.)
- eCFR. *21 CFR Part 11, Electronic Records; Electronic Signatures.* (Cite the Part; no section URL given.)
- European Union. *Regulation (EU) No 536/2014 on clinical trials on medicinal products for human use.* https://eur-lex.europa.eu/eli/reg/2014/536/oj

**Transparency and data sharing**

- ClinicalTrials.gov. https://clinicaltrials.gov/
- Vivli. https://vivli.org/
- YODA Project. https://yoda.yale.edu/
- ICMJE. *Clinical Trials registration and data sharing requirements.* https://www.icmje.org/

**Background**

- International Council for Harmonisation. *E8(R1) General Considerations for Clinical Studies.* (Cite the guideline.)
- FDA. *Real-World Evidence.* https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence

> **Note:** The CDISC and ICH documents are revised on their own schedules. The SDTMIG v3.4 date and the FDA 15 March 2025 requirement date are the load-bearing facts here, and both are documented in the Federal Register notice cited above.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_omics-data.md`](06_omics-data.md)
- Next topic: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md)
- Related sibling: [`02_claims-and-billing.md`](02_claims-and-billing.md)
- Related sibling: [`05_signals-and-wearables.md`](05_signals-and-wearables.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.4 in the survey file (source-selection matrix) uses the trial-versus-RWE comparison in section 3.3 as one of its scenarios.

Related reading: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md) for how CDISC relates to FHIR and OMOP; [`09_licensing-and-access.md`](09_licensing-and-access.md) for individual participant data sharing terms; and [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) for the quality framework that applies to trial data as much as to routine care data.

The sentence to remember: a trial is a design, not a dataset, and the design is what makes the causal claim possible.

The next topic generalizes the standards question: how HL7 v2, FHIR, OMOP, LOINC, SNOMED CT, ICD, and RxNorm relate, and why mapping between them loses meaning that no amount of engineering recovers.

A reader who has followed the SDTM-to-ADaM distinction will recognize the same pattern: a standard for tabulation and a standard for analysis, joined by a documented derivation.

The same pattern appears again in FHIR-to-OMOP conversion, where FHIR resources are the tabulation and OMOP tables are the analysis model, and the ETL is the derivation that must be documented.

Three questions to carry into the next topic:

- Which standard governs the data at rest, and which governs it in motion?
- Which standard is a terminology, and which is a structure?
- Which mapping is lossy, and what is lost?

A final practical note: every standard in this domain has a version, and the version is part of the data. The analyst who records versions is the analyst whose results can still be reproduced in five years.

That habit is the through-line of this module, and it is the one worth carrying to the remaining topics.

Nothing in this file is legal or regulatory advice. The dates and requirements cited are documentary, and a submission decision requires regulatory review.

That caveat is not boilerplate. The SDTM requirement date determines whether a study is filed under one standard or another, and the consequences of getting it wrong are measured in months.

Read the next topic with that in mind: standards decide what can be submitted, and terminology decides what can be compared.

That sentence is the bridge to the standards topic, where the same distinction between structure and meaning organizes the material.

### 11.5 The limit that decides a labelling claim

For a labelling claim, the binding limits are the population studied, the comparator used, and the durability of the effect. A trial that shows a 12-point difference at 24 weeks against standard care does not support a claim about long-term benefit or about a different comparator, and no amount of statistical precision changes that.

The practical statement names all three: "in adults meeting the eligibility criteria, against standard care, at 24 weeks." A claim that omits any of the three is broader than the evidence.

Four questions to carry into the standards topic:

- Which standard governs the data at rest, and which governs it in motion?
- Which standard is a terminology, and which is a structure?
- Which mapping is lossy, and what is lost?
- Which version is in force for this submission?

The second question is the one that organizes the next topic: a trial dataset can conform perfectly to a structure and still be semantically incomparable to another study, because the concepts underneath are different.

### 11.6 The limit that survives every analysis

Randomization addresses confounding, and it does not address the four limits above. A trial's population, duration, comparator, and endpoint remain the boundaries of its claim after every sensitivity analysis has been run.

A reader who understands that will not ask for a trial to be more general than it is, and will know which additional study the question requires.

That boundary is not a weakness of the design; it is the price of the causal claim, and it is why the module pairs trial data with real-world data rather than treating one as a substitute for the other.

Read the next topic for the machinery that makes those boundaries explicit: versioned standards and terminologies, each with a scope and a date.

That machinery is not bureaucracy. It is what allows a reviewer in one country to check an analysis produced in another, and it is the subject of the next topic.

A reader who has followed the SDTM-to-ADaM distinction will find the same pattern in FHIR-to-OMOP: a tabulation standard, an analysis model, and a derivation in between that must be documented.

The derivation is the place where meaning is lost, and it is the place the next topic teaches you to inspect.
