# Pharma and Safety Datasets

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 10 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02, Topic 03
> Version 2026.3

## 1. Why this topic exists

Pharmacovigilance data are the clearest example of a numerator without a denominator. A spontaneous reporting system counts reports, not cases, and it cannot count the people who took the drug and did not report. A raw count of 150 reports for an event can look alarming and be uninterpretable, because the number depends on how many people were exposed, how many were monitored, how many knew to report, and how much attention the event received.

The failures are specific:

- A report count is presented as an incidence rate, with no exposure denominator.
- A disproportionality signal is presented as a causal finding, when it is a hypothesis.
- A media story increases reporting, and the increase is interpreted as a change in risk.
- Duplicate reports inflate the count, and the duplicates are not removed.
- A trial registry entry is treated as a complete results record, when results reporting is incomplete.
- An individual-patient-data request is treated as a download, when it is a reviewed application.
- A safety signal from one database is confirmed in the same database, which is not independent.

This topic covers the major resources: the FDA Adverse Event Reporting System, now the Adverse Event Monitoring System (AEMS), Sentinel, ClinicalTrials.gov and the Aggregate Analysis of ClinicalTrials.gov (AACT), EudraVigilance, VigiBase, VAERS, MAUDE, and the individual-patient-data platforms Vivli and YODA. It treats reporting bias, stimulated reporting, duplicate reports, and the denominator problem as the core mechanics.

The central arithmetic is a two-by-two table. A disproportionality measure compares the proportion of reports with the event for the drug of interest with the proportion for all other drugs. It produces a ratio and a confidence interval, and it does not produce a risk. A team that understands the difference can use the signal responsibly; a team that does not will overclaim.

The practical consequence is that safety analysis is a discipline of boundaries. The boundary between a report and a case, between a signal and a risk, and between an association and a cause must be stated in every memo. A team that states them can act on weak evidence without overclaiming; a team that does not will confuse a database artefact with a patient harm.

A second consequence is that the denominator is the most valuable missing piece. Most safety questions cannot be answered from reports alone, and the programme that can supply a denominator determines the timeline.

> **Note:** The FDA renamed FAERS to the Adverse Event Monitoring System (AEMS) in the module survey. The underlying reporting system and data structure are continuous, and older documentation still uses FAERS. Verify the current name and extract at the FDA page before citing.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a report, a case, an event, and an incidence rate, and state the denominator each requires. Assessed by Exercise 1 and self-check item 1.
2. **Compute** a proportional reporting ratio and a reporting odds ratio with confidence intervals from a two-by-two table. Assessed by Exercise 2.
3. **Explain** why a disproportionality signal does not establish causation, and what evidence would. Assessed by Exercise 3.
4. **Detect** duplicate reports and describe the limits of exact and probabilistic matching. Assessed by Exercise 4.
5. **Evaluate** stimulated reporting and notoriety bias in a time series. Assessed by Exercise 5.
6. **Map** a safety question to FAERS/AEMS, Sentinel, trial registries, and individual-patient-data platforms. Assessed by Exercise 6.
7. **Design** a signal evaluation that combines spontaneous reports, real-world data, and trial data without double counting. Assessed by Exercise 7.
8. **Write** a safety data memo that states the numerator, the denominator, the bias, and the prohibited claims. Assessed by Exercise 8.

## 3. The core mental model

A safety dataset is a reporting system with a numerator, a partial denominator, and a set of reporting incentives. The analysis must separate the signal from the report-generating process.

```text
   EXPOSED POPULATION
   unknown size, unknown monitoring
        |
        v
   EVENT OCCURS
   true cases, some diagnosed
        |
        v
   REPORTING DECISION
   clinician | patient | manufacturer | literature
        |
        v
   REPORT
   drug, event, date, reporter, narrative
        |
        v
   DATABASE
   deduplication | coding | case series
        |
        v
   SIGNAL METRIC
   count | PRR | ROR | EBGM
        |
        v
   INTERPRETATION
   hypothesis | signal | risk | causation
```

**The numerator rule.** A report is not a case, and a case is not an event in a defined population. The report count is a function of the event, the exposure, the monitoring, and the reporting culture.

**The denominator rule.** Without an exposure denominator, a count cannot become a rate. With a partial denominator, the rate is a lower bound or an estimate with unknown coverage.

**The reporting-bias rule.** Reporting is stimulated by media, litigation, label changes, and professional attention. A rise in reports can reflect a rise in attention, not a rise in risk.

**The duplicate rule.** One case can generate multiple reports through different reporters, follow-ups, and subsidiaries. Duplicates inflate counts and distort disproportionality.

**The signal rule.** Disproportionality measures whether an event is reported more often than expected relative to other events. It generates a hypothesis. Causation requires temporality, plausibility, dose-response, dechallenge, rechallenge, and control of confounding.

| Measure | Numerator | Denominator | What it supports |
|---|---|---|---|
| Report count | Reports | None | Hypothesis generation |
| Proportional reporting ratio | Drug-event proportion | Other-drug proportion | Disproportionality |
| Reporting odds ratio | Odds of event for drug | Odds for other drugs | Disproportionality |
| Empirical Bayes geometric mean | Observed/expected | Database expectation | Disproportionality |
| Reporting rate | Reports | Exposure (if known) | Crude rate with coverage caveat |
| Incidence | Cases | Person-time | Risk |

## 4. Definitions and vocabulary

### 4.1 Reporting and case terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Spontaneous report | A report submitted voluntarily. | Treated as a case | Report |
| Case | A patient episode described by one or more reports. | Counted per report | Case |
| Serious adverse event | An event causing death, hospitalization, disability, or similar. | Treated as caused by the drug | Classification |
| Adverse event of special interest | An event selected for enhanced monitoring. | Treated as confirmed | List |
| Reporter | The person or organisation submitting. | Ignored in bias analysis | Category |
| Narrative | The free-text description. | Treated as structured data | Text |
| Follow-up report | An update to an existing case. | Counted as a new case | Report |
| Duplicate | Two reports of the same case. | Ignored | Pair |
| MedDRA | The standard event terminology. | Treated as a clinical ontology | Code |
| Preferred term | The specific MedDRA term. | Confused with the system organ class | Code |

### 4.2 Signal and statistics terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Disproportionality | Event reported more than expected relative to other events. | Treated as causation | Ratio |
| PRR | Proportional reporting ratio. | Reported without a CI | Ratio |
| ROR | Reporting odds ratio. | Reported without a CI | Ratio |
| EBGM | Empirical Bayes geometric mean. | Treated as a probability | Ratio |
| Confidence interval | Uncertainty around the ratio. | Omitted | Interval |
| Signal | A hypothesis requiring evaluation. | Treated as a finding | Status |
| Causality assessment | Structured judgement of the drug-event relationship. | Replaced by disproportionality | Category |
| Notoriety bias | Increased reporting after publicity. | Ignored in trends | Bias |
| Stimulated reporting | Increased reporting after an event. | Treated as increased risk | Bias |
| Channeling | Non-random prescribing to risk groups. | Ignored in comparisons | Bias |

### 4.3 Data source terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| FAERS | FDA Adverse Event Reporting System. | Treated as a complete case series | Database |
| AEMS | Adverse Event Monitoring System, the current FDA name. | Confused with a new database | System |
| Sentinel | FDA distributed claims and EHR network. | Treated as open data | Programme |
| ClinicalTrials.gov | Trial registry and results database. | Treated as complete results | Registry |
| AACT | Relational mirror of ClinicalTrials.gov. | Treated as real-time | Database |
| EudraVigilance | EU adverse reaction reporting system. | Treated as global | Database |
| VigiBase | WHO global case safety reports. | Treated as open | Database |
| VAERS | Vaccine adverse event reporting system. | Treated as causal | Database |
| MAUDE | Device adverse event reports. | Treated as complete | Database |
| Vivli | Individual-patient-data sharing platform. | Treated as a download | Platform |
| YODA | Yale Open Data Access project. | Treated as open data | Project |

### 4.4 Data quality and access terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Data cut | The date the extract was fixed. | Not recorded | Date |
| Deduplication | Removing repeated reports. | Assumed complete | Process |
| Case consolidation | Merging follow-ups into one case. | Dropped or duplicated | Process |
| Coding audit | Checking event coding. | Omitted | Process |
| Reporting completeness | Share of cases reported. | Assumed 100 percent | Proportion |
| Access tier | Open, registered, or restricted. | Assumed open | Tier |
| Data use agreement | Contract for controlled data. | Read as a publication licence | Contract |
| Programme access | Access through a collaborative programme. | Treated as a download | Route |
| Data lock point | The fixed analysis dataset date. | Confused with the data cut | Date |
| Extract hash | A fingerprint of the extract. | Not recorded | Hash |

### 4.5 Epidemiological terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Incidence rate | New cases per person-time. | Replaced by report count | Rate |
| Prevalence | Proportion with a condition. | Confused with incidence | Proportion |
| Relative risk | Risk ratio between exposed and unexposed. | Confused with odds ratio | Ratio |
| Odds ratio | Odds ratio between groups. | Treated as a risk ratio | Ratio |
| Confounding by indication | The drug is given for a condition that causes the event. | Ignored | Bias |
| Immortal time bias | Exposure classification uses future information. | Introduced by cohort design | Bias |
| Healthy-user bias | Users are healthier than non-users. | Ignored | Bias |
| Depletion of susceptibles | High-risk people are removed early. | Ignored | Bias |
| Channeling bias | Prescribing differs by prognosis. | Ignored | Bias |
| Protopathic bias | The early disease causes the prescription. | Ignored | Bias |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the question and the data source, audit the reports and the denominator, compute and interpret the signal, and confirm and report.

### Phase A — Define the question and the source

**Step 1. Write the safety question.**
Input: the drug, the event, the population, and the decision.
Output: a question labelled as signal detection, signal evaluation, or risk quantification.
Rationale: the question determines whether a spontaneous reporting database is adequate.
Branch: if the question is risk quantification, a denominator-based source is required.
Edge case: a regulatory question may require both a signal assessment and a cumulative review.

**Step 2. Choose the data source.**
Input: the question and the available sources.
Output: a primary source and a confirmatory source.
Rationale: spontaneous reports generate hypotheses; claims, EHR, and trials test them.
Branch: use the source decision table below.
Edge case: a source can be adequate for one event and not another because of reporting culture.

**Step 3. Define the event and the drug.**
Input: the terminology and the product labels.
Output: MedDRA terms, drug names, and the coding strategy.
Rationale: the event definition determines the numerator.
Branch: if the event is a syndrome, combine preferred terms and state the grouping.
Edge case: a drug can have multiple names and formulations; include all.

**Step 4. Define the comparison.**
Input: the database structure.
Output: the comparator set: all other drugs, a therapeutic class, or a specific drug.
Rationale: the comparator determines the expected proportion.
Branch: a class comparator is more interpretable and smaller.
Edge case: a comparator can share the same confounding, which biases toward the null.

### Source profiles

| Source | Custodian | Unit | Access and licence | Strength | What it cannot answer |
|---|---|---|---|---|---|
| FAERS/AEMS | FDA | Report | Open extracts and API | Broad spontaneous reporting | Incidence or risk |
| Sentinel | FDA | Enrollee | Programme access | Denominator and confounding control | Open research |
| ClinicalTrials.gov | NIH | Trial | Open | Landscape and eligibility | Complete results |
| AACT | CTTI | Trial | Open | SQL analysis | Real-time updates |
| EudraVigilance | EMA | Report | Public and registered tiers | EU reports | Global coverage |
| VigiBase | WHO-UMC | Report | Restricted to members | Global reports | Open access |
| VAERS | CDC and FDA | Report | Open | Vaccine reports | Causation |
| MAUDE | FDA | Report | Open | Device reports | Completeness |
| Vivli | Vivli | Trial participant | Request and review | Individual patient data | All trials |
| YODA | Yale | Trial participant | Request and review | Independent reanalysis | All sponsors |

The access column is the first filter. A source that cannot be obtained before the decision is not a candidate, however strong the science.

### The two-by-two table

| Cell | Definition | Example |
|---|---|---|
| a | Reports with the drug and the event | 150 |
| b | Reports with the drug and other events | 850 |
| c | Reports with other drugs and the event | 300 |
| d | Reports with other drugs and other events | 98,700 |

The table is built from reports, not people. A patient with two events appears in more than one table, and a patient with two drugs appears in more than one drug row.

### Disproportionality measures

| Measure | Formula | Interpretation | Caveat |
|---|---|---|---|
| PRR | (a/(a+b)) / (c/(c+d)) | Proportion ratio | Not a risk ratio |
| ROR | (a*d) / (b*c) | Odds ratio | Not a risk ratio |
| EBGM | Observed/expected with shrinkage | Stabilised ratio | Not a probability |
| IC | Information component | Bayesian signal | Prior-dependent |
| Chi-square | Test of independence | Statistical association | Large n inflates |

All of the measures answer the same question: is the drug-event combination reported more often than expected relative to the rest of the database? None answers whether the drug caused the event.

### Reporting bias

| Bias | Mechanism | Detection | Mitigation |
|---|---|---|---|
| Notoriety bias | Publicity increases reporting | Spike after media | Pre/post analysis |
| Stimulated reporting | A publication or label change | Time-series break | Stratify by period |
| Under-reporting | Most events are not reported | Literature estimates | Do not treat as complete |
| Duplicate reporting | One case, many reports | Matching | Deduplicate |
| Channeling | Drug given to high-risk patients | Prescribing data | Confounding control |
| Web-based reporting | New channel increases reports | Mode mix | Reporter stratification |
| Litigation-driven | Legal action increases reports | Legal timeline | Sensitivity analysis |

The biases are not independent. A media story can increase reporting, which increases duplicates, which increases the count and the disproportionality.

### Duplicate detection

| Method | What it matches | Strength | Limit |
|---|---|---|---|
| Exact ID | Report identifier | Certain | Misses re-reports |
| Exact fields | Drug, event, date, site | Simple | Misses variations |
| Fuzzy name | Patient or reporter | Flexible | False positives |
| Temporal clustering | Same event near same date | Finds clusters | Not case-level |
| Narrative similarity | Text overlap | Finds re-descriptions | Manual review |
| Manufacturer consolidation | Company case number | Authoritative | Not always available |

Deduplication is a judgement. Report the rule, the number removed, and the effect on the signal.

### Causality assessment

| Method | Input | Output | Limit |
|---|---|---|---|
| WHO-UMC | Temporal and clinical criteria | Certain, probable, possible, unlikely | Depends on report quality |
| Naranjo | Questionnaire | Score and category | Weighted items |
| RUCAM | Liver-specific criteria | Score and category | Liver only |
| Clinical judgement | Full case | Narrative judgement | Not reproducible |
| Disproportionality | Database counts | Signal | Not individual causality |

A causality assessment for an individual case is different from a population signal. The two should not be substituted for each other.

### Denominator sources

| Source | Denominator | Strength | Limit |
|---|---|---|---|
| Claims | Enrolment months | Large and longitudinal | Diagnosis coding |
| EHR | Patient panel | Clinical detail | Site coverage |
| Registry | Condition cohort | Deep phenotyping | Selection |
| Trial | Randomised exposure | Causal inference | Size and generalizability |
| Spontaneous reports | None | Broad | No denominator |

A report count becomes a rate only with a denominator. The denominator source determines whether the rate is credible.

### ClinicalTrials.gov fields

| Field | Use | Limitation |
|---|---|---|
| Registration | Trial existence | Not results |
| Status | Recruitment state | Can be stale |
| Eligibility | Population | Not enforcement |
| Interventions | Drug and dose | Not outcomes |
| Primary outcome | Endpoint | Not results |
| Results | Posted summary | Incomplete |
| Adverse events | Posted events | Selective |
| Sponsor | Accountability | Not quality |

A registry entry is a plan and a status. It is not a results report, and a missing result is not evidence of no effect.

### Individual-patient-data platforms

| Platform | Scope | Access | Strength | Limit |
|---|---|---|---|---|
| Vivli | Multiple sponsors | Request and review | IPD reanalysis | Approval and scope |
| YODA | Selected sponsors | Request and review | Independent analysis | Limited trials |
| Project Data Sphere | Oncology trials | Registration | Comparator arms | Oncology only |
| CSDR | Multiple sponsors | Request | Broad | Review time |

An IPD request is a research collaboration with a review, not a download. Plan the review time and the analysis plan.

The source profile is also a statement about what the analysis can conclude. A spontaneous database supports signal detection; a claims network supports rate estimation; a trial supports causal inference. Mixing the conclusions of one source into the claim of another is the central error this topic exists to prevent.

A final point about the source profile: the same database can be a 0 on population fit and a 2 on decision fit for the same question, depending on whether the claim is "there is a signal" or "the risk is X." Write the claim before scoring the gate.

The denominator is not a technical detail; it is the boundary between a signal and a risk. A team that states the boundary clearly can make a decision; a team that blurs it will overclaim in one direction or the other.

A signal evaluation without a denominator is a hypothesis with a well-documented numerator. That is a legitimate output when it is labelled as such.

### Phase B — Audit reports and denominator

**Step 5. Extract the reports.**
Input: the database and the query.
Output: a case-level extract with report ID, drug, event, date, reporter, and outcome.
Rationale: the extract is the numerator.
Branch: if the extract is report-level, deduplicate to case level.
Edge case: a case can have multiple drugs and events; the unit must be defined.

**Step 6. Deduplicate.**
Input: the report identifiers and the case fields.
Output: a deduplicated case set with a documented rule.
Rationale: duplicates inflate counts and distort the two-by-two table.
Branch: exact duplicates by ID and date are removed first; probabilistic matching follows.
Edge case: follow-up reports can add information and should be merged, not dropped.

**Step 7. Code the events.**
Input: the narratives and the coded terms.
Output: a mapped event set with a coding audit.
Rationale: coding determines which reports enter the numerator.
Branch: if the coding is inconsistent, audit a sample and report agreement.
Edge case: a narrative can describe an event that the coded term misses.

**Step 8. Audit the denominator.**
Input: the exposure data, if any.
Output: a statement of whether a denominator exists and its coverage.
Rationale: a count without a denominator is not a rate.
Branch: if no denominator exists, the analysis is limited to disproportionality.
Edge case: a partial denominator produces a lower-bound rate with unknown coverage.

**Step 9. Audit reporting bias.**
Input: the report dates and the external events.
Output: a timeline of reports and publicity, label changes, and publications.
Rationale: a spike can reflect attention, not risk.
Branch: if a spike follows publicity, analyse pre- and post-publicity periods separately.
Edge case: a label change can increase reporting of the labelled event and decrease other reports.

### Phase C — Compute and interpret

**Step 10. Build the two-by-two table.**
Input: the deduplicated cases and the event coding.
Output: a, b, c, d with definitions.
Rationale: every disproportionality measure is a function of the table.
Branch: define whether the unit is the report, the case, or the drug-event pair.
Edge case: a case with multiple events contributes to multiple tables.

**Step 11. Compute PRR and ROR.**
Input: the two-by-two table.
Output: point estimates and confidence intervals.
Rationale: the interval shows whether the signal is stable.
Branch: use the appropriate variance formula for the measure.
Edge case: small cells produce unstable estimates; use a shrinkage method.

**Step 12. Apply a signal threshold.**
Input: the estimates and the pre-specified rule.
Output: a signal classification.
Rationale: a threshold makes the decision repeatable.
Branch: a common rule requires a minimum count, a ratio above a threshold, and a lower confidence bound above one.
Edge case: a threshold is a screening rule, not a causal criterion.

**Step 13. Assess causality with a structured method.**
Input: the case narratives and the temporal and clinical information.
Output: a causality category per case and an aggregate judgement.
Rationale: disproportionality does not assess individual causality.
Branch: use a published algorithm and report the categories.
Edge case: the algorithm depends on the information in the report, which is often incomplete.

### Phase D — Confirm and report

**Step 14. Confirm in a denominator-based source.**
Input: claims, EHR, or registry data.
Output: an incidence rate or a relative risk with confounding control.
Rationale: a signal needs a population-based test.
Branch: if no denominator source exists, the signal remains a hypothesis.
Edge case: confounding by indication can produce a spurious association.

**Step 15. Check consistency across sources.**
Input: spontaneous reports, real-world data, and trials.
Output: a triangulation table.
Rationale: agreement across independent sources strengthens the hypothesis.
Branch: if the sources disagree, investigate the difference rather than averaging.
Edge case: the sources can measure different populations and severities.

**Step 16. Quantify uncertainty and bias.**
Input: the estimates and the bias assessment.
Output: a range and a bias statement.
Rationale: a point estimate without bias is misleading.
Branch: conduct sensitivity analyses for duplicates, reporting lag, and comparator choice.
Edge case: the bias can be larger than the effect.

**Step 17. Write the safety memo.**
Input: the analyses and the evidence.
Output: a memo with numerator, denominator, bias, and prohibited claims.
Rationale: the memo is the decision artefact.
Branch: if the evidence is insufficient, recommend a study rather than a conclusion.
Edge case: a regulatory submission has specific format requirements.

**Step 18. Re-run on data cut, label, or publicity change.**
Input: the data cut and the external events.
Output: an updated memo with a new date.
Rationale: safety data change with reports and labels.
Branch: re-run when the data cut, the label, or a major publication changes.
Edge case: a data cut can add reports for prior periods through backfill.

### Signal interpretation ladder

| Evidence | What it supports | What it does not |
|---|---|---|
| Case report | A hypothesis | Frequency or cause |
| Case series | A cluster | Incidence |
| Disproportionality | Reporting excess | Risk |
| Denominator study | Association with control | Causation without design |
| Trial | Causal effect in the trial population | Real-world effectiveness |

The ladder is cumulative. A conclusion should not be placed higher than the evidence.

### Source decision table

| Question | Source | Strength | Limit |
|---|---|---|---|
| Is there a signal? | FAERS/AEMS | Broad, spontaneous | No denominator |
| Is the signal confirmed? | Sentinel, claims, EHR | Denominator and confounding control | Access and lag |
| What trials exist? | ClinicalTrials.gov | Landscape and eligibility | Incomplete results |
| What did the trials find? | Vivli, YODA, publications | Individual patient data | Approval and scope |
| Is the event vaccine-related? | VAERS | Vaccine-specific | Reporting bias |
| Is the device-related? | MAUDE | Device-specific | Under-reporting |
| Is the event global? | VigiBase, EudraVigilance | International | Access tiers |

### Signal threshold table

| Criterion | Common value | Meaning | Caveat |
|---|---|---|---|
| Minimum reports | 3 or more | Avoids one-case noise | Arbitrary |
| PRR | Above 2 | Disproportionality | Not causation |
| ROR lower bound | Above 1 | Statistical stability | Not risk |
| EBGM lower bound | Above 2 | Shrinkage-based signal | Not probability |
| Temporal plausibility | After exposure | Supportive | Report date is not event date |
| Dechallenge | Improvement on stopping | Supportive | Confounded by disease course |

### Edge cases that change the procedure

**Licence forbids commercial use.** FAERS/AEMS extracts are open, Sentinel is not open, and Vivli requests have use terms. A commercial analysis must check each source.

**Dataset withdrawn.** A data extract can be revised or removed. Keep the query, the cut date, and the extract hash.

**Credentialing takes months.** Sentinel access is through FDA or collaborators; Vivli requests are reviewed. A timeline shorter than the access path fails Gate 3.

**Version drift.** A data cut adds reports and can change the signal. Record the cut date and re-run.

**Label errors discovered.** A coded event can be wrong. Audit a sample and report agreement.

**Benchmark contamination.** A signal confirmed in the same database is not independent confirmation. Use a different source.

## 6. Worked example

### The case

A pharmaceutical company markets an antiemetic approved for chemotherapy-induced nausea. A published case series suggests an association with QT prolongation. The safety team must decide whether to add a warning, conduct a study, or take no action. The decision is due in six months.

**Decision statement.** "Decide whether the evidence supports a label warning for QT prolongation; a warning reduces prescribing in high-risk patients and may reduce the market, and a missed signal exposes patients to arrhythmia."

**Question type.** Signal evaluation, not risk quantification. The team needs to know whether the signal is stable, whether it is explained by bias, and whether a denominator-based study is feasible.

### Candidate long list

| Candidate | Custodian | Unit | Access and licence | Strength | Why it is on the list |
|---|---|---|---|---|---|
| FAERS/AEMS | FDA | Report | Open | Broad spontaneous reports | Signal detection |
| Sentinel | FDA | Enrollee | Programme access | Denominator and confounding control | Signal confirmation |
| ClinicalTrials.gov | NIH | Trial | Open | Landscape | Existing trials |
| Vivli | Vivli | Participant | Request and review | Individual patient data | QT data from trials |
| Internal safety database | Company | Case | Internal | Company cases and follow-up | Case-level detail |

### Gate scores for the label decision

| Gate | FAERS/AEMS | Sentinel | ClinicalTrials.gov | Vivli | Internal |
|---|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 2 | 1 | 2 | 2 |
| 2 Population fit | 0 | 2 | 1 | 1 | 2 |
| 3 Access fit | 2 | 0 | 2 | 1 | 2 |
| 4 Licence fit | 2 | 1 | 2 | 1 | 2 |
| 5 Benchmark fit | 1 | 2 | 1 | 1 | 2 |
| Veto | population | access | none | none | none |

**FAERS/AEMS, Gate 2.** The database has no exposure denominator and no control group. It cannot estimate risk or incidence. It can detect disproportionality. Score 0 on population fit for a risk claim, 2 for a signal claim.

**Sentinel, Gate 3.** Access is through the FDA programme or an approved collaborator, not an open download. The company is not a participant. Score 0, veto for this timeline.

**ClinicalTrials.gov, Gate 1.** The registry describes trials that may have collected QT data. It is a landscape and feasibility source, not a results source. Score 1.

**Vivli, Gate 3.** An individual-patient-data request is reviewed and can take months. It is feasible within six months only if the request is submitted immediately and the sponsor agrees. Score 1.

**Internal, Gate 1.** The company has its own cases, follow-up, and trial data. It is the strongest source for case-level causality and the weakest for population risk. Score 2.

### The two diagnostics

**Diagnostic 1: the count has no denominator.** The database contains 150 reports of QT prolongation for the drug. The company's exposure estimate is about 2 million patient-courses. The crude reporting rate is 7.5 per 100,000 exposed, and it is a lower bound. If reporting completeness is 10 percent, the true rate could be 75 per 100,000; if it is 5 percent, 150 per 100,000. The database does not measure completeness. A raw count of 150 is not a risk, and a reporting rate is not an incidence rate without a completeness assumption.

**Diagnostic 2: the signal is stable but not causal.** The two-by-two table gives a PRR of 49.5 (95 percent CI 41.1 to 59.6) and an ROR of 58.1 (95 percent CI 47.2 to 71.4). The lower bound is far above 1. The signal is statistically stable and is not evidence of causation. The comparison uses all other drugs and events in the database, and the reporting of QT prolongation is itself stimulated by the publication.

A third diagnostic is notoriety bias. Report counts before and after the publication can differ by a factor that reflects attention. The analysis should split the periods and report both.

### The selected route

The admissible set is FAERS/AEMS, ClinicalTrials.gov, Vivli, and the internal database. The plan uses the internal and spontaneous data for signal evaluation, Vivli for trial-level QT data, and a claims partner for a denominator-based study if the signal persists.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Signal analysis | Deduplicate, code, PRR/ROR, sensitivity | Safety analytics | 6 weeks | Staff time |
| Case review | Causality assessment of 50 cases | Medical review | 6 weeks | Medical time |
| Trial data request | Vivli application and analysis | Biostatistics | 3 to 6 months | Review and staff |
| Claims feasibility | Partner or vendor | Epidemiology | 8 weeks | Contract |
| Label decision | Benefit-risk review | Safety committee | 4 weeks | Internal |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Spontaneous analysis | Extract, dedup, coding, ratios | 6 weeks | Staff |
| Case review | 50 narratives | 6 weeks | Medical |
| Vivli request | Application and review | 3 to 6 months | Staff and fees |
| Claims study | Denominator and confounding | 6 months | Six figures |
| Total to a decision | Parallel tracks | About 6 months | Dominated by data access |

### Substitute plan if Vivli and Sentinel are unavailable

If neither individual-patient data nor Sentinel is available, the plan changes in three ways:

1. The evidence is limited to spontaneous reports, internal cases, and published trials.
2. The decision is a precautionary label update or a post-marketing study commitment, not a quantified risk assessment.
3. The uncertainty is stated explicitly, and the label language reflects the strength of the evidence.

The fallback is a weaker evidence base and a defensible precautionary decision. It is not equivalent to a denominator-based study.

### Vary one assumption: the event is not stimulated by publicity

Suppose the report counts are stable before and after the publication. The notoriety-bias explanation weakens, and the signal is more likely to reflect reporting of a real effect. The plan still requires a denominator-based confirmation, because disproportionality is not causation.

If the variation is the comparator, a class comparator changes the ratio and can reduce confounding by indication. If the variation is the data cut, a later cut adds reports and can change the interval.

### Conditions under which the case would fail

- **The signal is explained by duplicates.** The ratio falls after deduplication, and the hypothesis weakens.
- **The signal is explained by publicity.** Pre-publicity reports are stable, and the post-publicity spike is reporting.
- **No denominator source is available.** The decision is precautionary and cannot be quantified.
- **The internal cases show no temporal plausibility.** The causality assessment weakens the signal.
- **A trial shows no QT effect.** The signal is not confirmed, and the label decision changes.
- **The data cut adds a cluster of reports.** The estimate must be re-run before a decision.

The decision can be a warning, a study commitment, or no action. The evidence grade determines which is defensible.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | FAERS/AEMS | Reports and event coding | 2 | No denominator |
| 1 | Vivli | Trial QT data | 2 | Request scope |
| 1 | Internal | Cases and follow-up | 2 | Company-only |
| 2 | FAERS/AEMS | No control group | 0 | Cannot estimate risk |
| 2 | Sentinel | Enrollee denominator | 2 | Access |
| 3 | FAERS/AEMS | Open extract | 2 | None |
| 3 | Sentinel | Programme access | 0 | Not a participant |
| 3 | Vivli | Review process | 1 | Timeline |
| 4 | FAERS/AEMS | Open data | 2 | None |
| 4 | Vivli | Use terms | 1 | Sponsor approval |
| 5 | FAERS/AEMS | Disproportionality possible | 1 | No causal benchmark |
| 5 | Internal | Case-level causality | 2 | Small numbers |

### Decision timeline

| Month | Signal track | Data track | Clinical track | Deliverable |
|---:|---|---|---|---|
| 1 | Deduplicate and code | Vivli request | Case selection | Clean case set |
| 2 | PRR/ROR and sensitivity | Claims partner enquiry | Case review | Signal report |
| 3 | Notoriety analysis | Vivli review | Causality summary | Bias assessment |
| 4 | Triangulation | Trial data analysis | Benefit-risk | Evidence table |
| 5 | Label options | Claims feasibility | Committee review | Recommendation |
| 6 | Submission | Final data cut | Sign-off | Decision |

### What the case demonstrates

- The report count was not a rate, and the reporting rate was not an incidence rate. The denominator assumption changed the implied risk by a factor of 20.
- The disproportionality signal was stable and not causal. The lower bound above 1 supported a hypothesis, not a warning.
- The best confirmation source was inaccessible, and the fallback was a precautionary decision rather than a quantified risk.
- Duplicates and notoriety bias could inflate the signal, and both required explicit sensitivity analyses.
- The internal cases were the best source for causality and the worst for population risk.
- The decision was due in six months, and the data access, not the analysis, set the schedule.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| Sentinel access granted | 3 | Yes | Add a denominator-based study |
| Vivli request approved | 3 | Yes | Add trial QT analysis |
| Report counts stable after publicity | 2 | No | Notoriety explanation weakens |
| A class comparator is used | 5 | No | Recompute the ratio |
| A later data cut adds reports | 1 | No | Re-run with the new cut |
| A trial shows no QT signal | 2 | No | Weaken the signal, review the label |

The case ends with an evidence grade and a decision rule: if the signal survives deduplication and bias analysis and is supported by trial data, the label changes; if not, the signal remains under monitoring. That is a defensible process even when the data are incomplete.

A final point: the same data can support a signal report and not a risk report. The decision artefact should say which one it is, so that a reader does not infer a risk from a signal.

The memo should be readable by a safety committee that will not compute the ratio. It should say what the data show, what they do not show, and what decision follows.

The decision rule is pre-specified so that the committee cannot choose the threshold after seeing the data. The rule is part of the method.

The evidence grade, not the ratio, determines the action.

A safety memo that says "signal, not risk" is more useful than one that says "risk" and cannot defend it.

The boundary between the two is the whole argument of the memo.

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement the two-by-two disproportionality calculation, exact duplicate detection, and a reporting-rate lower bound.

### 7.1 PRR and ROR with confidence intervals

```python
import math

a, b, c, d = 150, 850, 300, 98_700
prr = (a/(a+b)) / (c/(c+d))
ror = (a*d) / (b*c)

se_ln_prr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
se_ln_ror = math.sqrt(1/a + 1/b + 1/c + 1/d)

prr_lo, prr_hi = prr*math.exp(-1.96*se_ln_prr), prr*math.exp(1.96*se_ln_prr)
ror_lo, ror_hi = ror*math.exp(-1.96*se_ln_ror), ror*math.exp(1.96*se_ln_ror)

print(f"2x2: a={a}, b={b}, c={c}, d={d:,}")
print(f"PRR: {prr:.2f}  95% CI {prr_lo:.2f} to {prr_hi:.2f}")
print(f"ROR: {ror:.2f}  95% CI {ror_lo:.2f} to {ror_hi:.2f}")
print()
print("A lower bound above 1 is a signal, not a causal finding.")
print("The comparison uses all other drugs and events in the database, not a control group.")
```

Expected output:

```text
2x2: a=150, b=850, c=300, d=98,700
PRR: 49.50  95% CI 41.11 to 59.61
ROR: 58.06  95% CI 47.19 to 71.43

A lower bound above 1 is a signal, not a causal finding.
The comparison uses all other drugs and events in the database, not a control group.
```

The interval is narrow because the cells are large. A small database with the same ratio would have a wider interval and a weaker signal.

### 7.2 Duplicate detection

```python
reports = [
    ("R001", "drugA", "nausea", "2024-01-10", "site1"),
    ("R002", "drugA", "nausea", "2024-01-10", "site1"),
    ("R003", "drugA", "nausea", "2024-02-01", "site2"),
    ("R004", "drugB", "nausea", "2024-02-02", "site2"),
]
seen = {}
duplicates = []
for rid, drug, event, date, site in reports:
    key = (drug, event, date, site)
    if key in seen:
        duplicates.append((rid, seen[key]))
    else:
        seen[key] = rid
print(f"Reports: {len(reports)}")
print(f"Duplicate pairs: {duplicates}")
print(f"Unique reports after exact dedup: {len(seen)}")
print()
print("Exact duplicates are easy. Case-level duplicates with different dates or spellings")
print("require probabilistic matching and manual review.")
```

Expected output:

```text
Reports: 4
Duplicate pairs: [('R002', 'R001')]
Unique reports after exact dedup: 3

Exact duplicates are easy. Case-level duplicates with different dates or spellings
require probabilistic matching and manual review.
```

The exact rule removes one of four reports. A real database needs a documented rule, a manual review sample, and a sensitivity analysis with and without the removed reports.

### 7.3 Reporting rate as a lower bound

```python
exposed = 2_000_000
reports = 150
rate_per_100k = reports / exposed * 100_000
print(f"Exposed: {exposed:,}")
print(f"Reports: {reports}")
print(f"Crude reporting rate: {rate_per_100k:.2f} per 100,000 exposed")
print()
for under in (0.5, 0.8, 0.9, 0.95):
    true_rate = rate_per_100k / (1 - under)
    print(f"Under-reporting {under:.0%}: estimated true rate {true_rate:.1f} per 100,000")
print()
print("The reporting rate is a lower bound whose distance from the true rate")
print("depends on reporting completeness, which the database does not measure.")
```

Expected output:

```text
Exposed: 2,000,000
Reports: 150
Crude reporting rate: 7.50 per 100,000 exposed

Under-reporting 50%: estimated true rate 15.0 per 100,000
Under-reporting 80%: estimated true rate 37.5 per 100,000
Under-reporting 90%: estimated true rate 75.0 per 100,000
Under-reporting 95%: estimated true rate 150.0 per 100,000

The reporting rate is a lower bound whose distance from the true rate
depends on reporting completeness, which the database does not measure.
```

The calculation shows why a reporting rate is not an incidence rate. The same 150 reports imply rates from 7.5 to 150 per 100,000 depending on an assumption the database cannot supply.

## 8. Data, standards, and regulatory context

Pharmacovigilance is one of the most standardised areas of biomedical data. The standards define the report format, the terminology, the coding, and the reporting obligations.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| ICH E2B | Binding in many jurisdictions | Individual case safety report structure | Establish causality |
| ICH E2D | Guidance | Post-approval safety data management | Define a signal threshold |
| MedDRA | Required by many regulators | Event terminology and coding | Capture every narrative detail |
| WHO-UMC causality | Guidance | Causality categories | Validate a population signal |
| FDA 21 CFR 314.80 | Binding in the US | Post-marketing reporting | Require a denominator |
| Sentinel programme | FDA programme | Distributed data analysis | Open access |
| GVP modules | Binding in the EU | Pharmacovigilance processes | Global harmonisation |
| GDPR | Binding in the EU | Personal data in reports | Define a case |
| HIPAA | Binding in the US | PHI in reports | Define reporting completeness |
| CIOMS | Guidance | Benefit-risk and signal management | Enforce a method |

**A report is not a case series.** The same case can generate several reports, and a case series requires deduplication, follow-up, and a denominator discussion.

**A signal is not a risk.** The standards define how to report and code; they do not convert disproportionality into incidence.

### Report structure and coding

| Element | Standard | Use | Limit |
|---|---|---|---|
| Patient | ICH E2B | Demographics | Often incomplete |
| Drug | WHO Drug Dictionary | Product identity | Name variations |
| Event | MedDRA | Coding | Coding variability |
| Temporal | ICH E2B | Dates | Report date not event date |
| Outcome | ICH E2B | Seriousness | Reporter judgement |
| Causality | WHO-UMC | Category | Report quality |
| Narrative | Free text | Detail | Unstructured |
| Follow-up | Case ID | Linkage | Inconsistent |

### Data quality checks

| Check | Method | Frequency |
|---|---|---|
| Duplicate rate | Matching and manual review | Each extract |
| Coding agreement | Sample re-code | Each extract |
| Report completeness | Field completion rates | Each extract |
| Reporter mix | Distribution by source | Each quarter |
| Event date availability | Field completion | Each extract |
| Seriousness distribution | Compare with expectation | Each quarter |
| Country mix | Distribution | Each release |
| Follow-up linkage | Case ID match | Each extract |

The checks are simple and they catch the errors that change a signal. A duplicate rate or a coding shift can move a ratio more than a new analysis method.

### Signal management process

| Stage | Activity | Output | Owner |
|---|---|---|---|
| Detection | Disproportionality and case review | Signal list | Safety analytics |
| Validation | Deduplication, coding, bias | Validated signal | Safety team |
| Prioritisation | Seriousness and plausibility | Priority | Safety committee |
| Evaluation | Causality and denominator | Evidence report | Epidemiology |
| Action | Label, study, communication | Decision | Regulatory |
| Monitoring | Follow-up and re-analysis | Updated signal | Safety team |

The process is cyclical. A signal that is not confirmed remains under monitoring rather than being closed.

### Reporting sources and their biases

| Source | Who reports | Main bias | Consequence |
|---|---|---|---|
| Clinician | Health professionals | Under-reporting | Numerator incomplete |
| Consumer | Patients and families | Media sensitivity | Stimulated reporting |
| Manufacturer | Companies | Legal obligations | Duplicate consolidation |
| Literature | Authors | Publication bias | Selective cases |
| Registry | Sites | Coverage | Denominator unknown |
| Claims | Payers | Coding | Outcome misclassification |

The source of the report is a variable. A change in the reporter mix can change the signal without a change in risk.

### Regulatory artefacts

- **Signal evaluation report** states the data cut, the method, the threshold, and the cases.
- **Causality assessment** states the algorithm and the categories.
- **Benefit-risk assessment** integrates efficacy and safety.
- **Risk management plan** states the monitoring and the studies.
- **Label negotiation** states the evidence and the proposed language.

## 9. Industry practice and case evidence

### What organisations actually do

- **Separate signal detection from risk quantification.** The first uses spontaneous data; the second uses denominators.
- **Deduplicate before counting.** The rule is documented and the effect is reported.
- **Stratify by reporting period.** Publicity and label changes create breaks.
- **Use a pre-specified threshold.** The rule is fixed before the analysis.
- **Triangulate across sources.** Spontaneous, claims, and trial data answer different questions.
- **Report the denominator assumption.** A reporting rate is a lower bound.
- **Plan data access early.** Sentinel and Vivli are not downloads.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| FAERS/AEMS | The FDA maintains a spontaneous reporting system with open extracts and a public dashboard; the module survey notes the rename to AEMS. | The data support signal detection and not incidence. |
| Sentinel | The programme uses distributed claims and EHR data for active surveillance and is not an open dataset. | Denominator-based confirmation requires programme access or a partner. |
| ClinicalTrials.gov | The registry contains trial registrations, status, and results for many studies, with incomplete results reporting. | A missing result is not evidence of no effect. |
| Vivli and YODA | Both provide individual-patient-data access through a request and review process. | IPD analysis is a collaboration with a timeline, not a download. |
| VigiBase | The WHO global database is restricted to member countries and centres. | Global coverage does not mean open access. |

The first two rows are stable. Programme names and access routes can change; verify at the source.

### What good looks like

- The question is labelled as signal detection, evaluation, or risk quantification.
- The denominator is stated or the analysis is limited to disproportionality.
- Duplicates are removed with a documented rule.
- Reporting periods are split around publicity and label changes.
- The interval and the threshold are reported.
- Causality is assessed with a structured method.
- Confirmation uses an independent source.
- The memo states the prohibited claims.

### Documented fact versus inference

- **Documented fact:** a regulatory guidance, a database documentation, a label, or a published study.
- **Inference:** a planning consequence for a different drug or population.
- **Unverified summary:** a secondary account of a signal or a programme.

A case table should carry a source and a status. Safety programmes change, and a stale access route can misstate both the data and the timeline.

### Programme access by organisation

| Organisation | FAERS/AEMS | Sentinel | Vivli | VigiBase |
|---|---|---|---|---|
| Academic | Open | Collaborator | Request | Member centre |
| Health system | Open | Partner | Request | Member centre |
| Manufacturer | Open | Limited | Sponsor or request | Member country |
| Regulator | Full | Full | Request | Member |
| Startup | Open | Not available | Request | Not available |

Access is a function of the organisation, not only the question. A plan that assumes access without a route is a plan with a hidden dependency.

### Why safety data mislead product teams

A product team sees a report count and a ratio and concludes that the drug is dangerous or safe. The count reflects reporting, and the ratio reflects the database composition. The correct conclusion is a hypothesis with a stated evidence grade. The teams that understand this plan a study; the teams that do not either overreact or underreact.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Count treated as rate | No denominator | Reports are not cases | False risk | Denominator source |
| Disproportionality as causation | Ratio reported alone | Comparison is internal | False causal claim | Structured causality |
| Duplicates inflated | Counts exceed case counts | Re-reports | Inflated signal | Deduplicate |
| Notoriety bias | Spike after publicity | Attention | False trend | Pre/post analysis |
| Under-reporting ignored | Rate called incidence | Reporting completeness unknown | Understated risk | Report as lower bound |
| Comparator contamination | Comparator shares the event | Class effect | Null bias | Active comparator |
| Confounding by indication | Drug given for the event's cause | Channeling | Spurious association | Confounding control |
| Immortal time | Exposure defined after outcome | Cohort design | Biased effect | Point-in-time design |
| Registry results assumed complete | Missing results | Selective reporting | False reassurance | Use IPD or publications |
| Data cut mismatch | Different cut dates | Backfill | Non-comparable analyses | Pin the cut |
| Label change ignored | Reporting shifts | Stimulated reporting | False trend | Stratify by label period |
| Small cells overinterpreted | Wide interval ignored | Sparse data | False signal | Shrinkage and interval |
| Source dependence | Same database confirms itself | No independence | False confirmation | Independent source |
| Access assumed | Sentinel treated as open | Programme access | Timeline slip | Access plan |
| Report date as event date | Temporal analysis uses report date | Delay and clustering | False temporality | Use event date |
| Seriousness misclassified | Reporter judgement varies | Classification | False severity | Adjudicate a sample |
| Class effect ignored | Comparator shares mechanism | Confounding | Null or spurious signal | Active comparator |
| Follow-up merged incorrectly | New information not linked | Case fragmentation | Double counting | Case ID merge |
| Database version change | Counts shift without a label | Extract update | False trend | Pin the cut |

Red flags that should stop a safety claim:

- The denominator is missing.
- The ratio is reported without an interval.
- The duplicates are not removed.
- The reporting period spans a publicity event.
- The confirmation uses the same database.
- The registry result is assumed complete.
- The data cut is not stated.
- The causality assessment is absent.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Reports are deduplicated | Counts depend on it | Re-reports | Matching audit |
| The event coding is consistent | The numerator depends on it | Coding variability | Sample audit |
| The comparator is appropriate | The ratio depends on it | Class effect | Active comparator |
| The reporting period is homogeneous | Trends depend on it | Publicity or label change | Stratify |
| The denominator is known | Rates depend on it | No exposure data | State as lower bound |
| The data cut is pinned | Reproducibility depends on it | Backfill | Record the cut |
| The causality method is applied | Case judgement depends on it | No algorithm | Structured review |
| The confirmation is independent | Triangulation depends on it | Same source | Different database |
| The registry is not complete | Results depend on it | Selective reporting | IPD or publication |
| The small cells are stable | Signals depend on it | Sparse data | Shrinkage |
| The label period is known | Reporting depends on it | Label change | Label timeline |
| The population is comparable | Rates depend on it | Channeling | Confounding control |
| The event date is available | Temporality depends on it | Report date only | Use report date with a caveat |
| Seriousness is adjudicated | Severity depends on it | Reporter judgement | Sample adjudication |
| The class effect is considered | The comparator depends on it | Shared mechanism | Active comparator |
| Follow-up is merged | Counting depends on it | Case fragmentation | Case ID merge |
| The database version is pinned | Reproducibility depends on it | Extract update | Record the cut |

The deepest limit is that spontaneous reporting cannot quantify risk. It can generate a hypothesis, describe a case series, and support a precautionary decision. It cannot tell you the incidence.

## 12. Exercises

**Exercise 1. Classify safety quantities.**
Deliverable: a table of six quantities labelled report, case, signal, rate, risk, or incidence.
Acceptance criteria: six quantities; denominator named; one mislabel corrected.
Time: 30 minutes.

**Exercise 2. Compute PRR and ROR.**
Deliverable: a two-by-two table, both ratios, and 95 percent intervals.
Acceptance criteria: arithmetic correct; cells defined; interpretation as a signal.
Time: 40 minutes.

**Exercise 3. Explain the causal gap.**
Deliverable: a one-page memo on what disproportionality does not establish and what would.
Acceptance criteria: names temporality, plausibility, dechallenge, confounding; no causal claim.
Time: 35 minutes.

**Exercise 4. Deduplicate a case series.**
Deliverable: a rule, a duplicate list, and a sensitivity analysis with and without duplicates.
Acceptance criteria: rule documented; counts before and after; effect on the ratio.
Time: 45 minutes.

**Exercise 5. Detect stimulated reporting.**
Deliverable: a time series with a publicity marker and pre/post counts.
Acceptance criteria: break identified; alternative explanation stated; a sensitivity analysis.
Time: 40 minutes.

**Exercise 6. Map sources to questions.**
Deliverable: a table of four safety questions and the source for each, with the limit.
Acceptance criteria: four questions; one source rejected; access route named.
Time: 40 minutes.

**Exercise 7. Design a triangulation.**
Deliverable: a plan combining spontaneous, claims, and trial data without double counting.
Acceptance criteria: independent sources; unit alignment; a decision rule.
Time: 50 minutes.

**Assessment note.** Exercises 2, 4, 5, and 6 produce checkable artefacts: a two-by-two table, a deduplication report, a time series, and a source map. Exercises 1, 3, 7, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write a safety memo.**
Deliverable: a two-page memo with numerator, denominator, bias, and prohibited claims.
Acceptance criteria: every element present; one quantified limitation; a recommendation.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. Why is a report not a case?**
A case can generate multiple reports through follow-up, different reporters, and subsidiaries. A report is a submission; a case is the underlying patient episode.

**Item 2. What is the denominator problem?**
Spontaneous reports have no exposure denominator, so a count cannot become a rate. A partial denominator produces a lower bound with unknown coverage.

**Item 3. What does a PRR of 49.5 mean?**
The drug-event proportion is 49.5 times the proportion for other drugs in the database. It is a disproportionality signal, not a risk ratio.

**Item 4. Why is the lower confidence bound important?**
It shows whether the signal is stable. A lower bound above 1 supports a hypothesis; an interval that includes 1 does not.

**Item 5. What is notoriety bias?**
Increased reporting after publicity, litigation, or a label change. It can create a spike that looks like increased risk.

**Item 6. Why do duplicates matter?**
They inflate the numerator and distort the two-by-two table. The effect on the ratio depends on where the duplicates fall.

**Item 7. What does Sentinel add?**
A denominator and confounding control through distributed claims and EHR data. It supports rate estimation and active surveillance.

**Item 8. Why is a missing trial result not evidence of no effect?**
Results reporting is incomplete and selective. A missing result means the result is unknown, not that it was null.

**Item 9. What is confounding by indication?**
The drug is prescribed for a condition that itself causes the event. The association reflects the indication, not the drug.

**Item 10. When is a precautionary label update justified?**
When the signal is stable, plausible, and not explained by bias, and when a denominator-based study is unavailable within the decision window. The label should reflect the uncertainty.

**Item 11. Why is a class comparator sometimes better than all other drugs?**
A class comparator shares the indication and often the mechanism, which reduces confounding by indication and isolates the drug-specific effect. It also reduces the sample and can widen the interval.

**Item 12. What is the difference between a report date and an event date?**
The report date is when the report was submitted; the event date is when the event occurred. Reporting delays and clusters can make the report date misleading for temporality.

**Item 13. Why is seriousness not the same as severity?**
Seriousness is a regulatory category based on outcomes such as death or hospitalization; severity is the clinical intensity. A severe event can be non-serious by the regulatory definition.

**Item 14. What is the difference between a signal and a risk?**
A signal is a statistical disproportionality requiring evaluation; a risk is a probability of harm in a defined population. The first is a hypothesis, the second is a quantity with a denominator.

**Item 15. Why can a label change increase reports of the labelled event?**
The label raises awareness and reporting. The increase is stimulated reporting, not necessarily a change in incidence. A pre/post label analysis is needed.

**Item 16. What is the role of individual patient data in a safety evaluation?**
It allows reanalysis of outcomes, time-to-event, and subgroups with full control. It is particularly useful when published results are incomplete or aggregated.

**Item 17. Why is a disproportionality result not a relative risk?**
The denominator is other reports in the database, not an unexposed population. The ratio compares reporting proportions, not incidence. A strong answer names the denominator. A weak answer calls it a risk ratio.

**Item 18. What is the difference between a case series and a cohort study?**
A case series describes cases without a comparison group; a cohort study follows exposed and unexposed people and estimates risk. A strong answer names the design. A weak answer treats a case series as a cohort.

**Item 19. Why is a denominator study not automatically causal?**
It controls some confounding and remains observational. Channeling, unmeasured confounding, and reverse causation can persist. A strong answer proposes a target trial or a negative control. A weak answer treats the association as causal.

**Item 20. What is the first question to ask about a new safety database?**
What is the unit and is there a denominator. A strong answer then asks about duplicates and the data cut. A weak answer starts with the count.

**Rubric for the design exercises (2, 4, 6, 8).** Full marks require a denominator statement, a bias assessment, an interval, and a prohibited-claim statement. Partial marks for a correct ratio without the causal caveat. No marks for a report count presented as incidence.

## 14. Interview questions

**1. How would you evaluate a safety signal from FAERS?**
Deduplicate, code the event, compute PRR/ROR with intervals, assess bias, review cases, and confirm in a denominator source. A strong answer separates signal from risk. A weak answer reports the count.

**2. Why can a raw report count mislead?**
It has no denominator and depends on reporting. A count can rise because exposure rose, monitoring rose, or publicity rose. A strong answer asks for the denominator. A weak answer treats the count as cases.

**3. What is the difference between PRR and ROR?**
Both compare the drug-event proportion with the rest of the database; PRR is a proportion ratio and ROR is an odds ratio. A strong answer notes that neither is a risk ratio. A weak answer calls them relative risks.

**4. What does a signal threshold establish?**
That the combination is reported more than expected. It does not establish causation. A strong answer lists the causal criteria. A weak answer treats the threshold as proof.

**5. How do you handle stimulated reporting?**
Plot reports over time, mark publicity and label changes, and analyse pre- and post-periods. A strong answer treats the spike as a reporting phenomenon until shown otherwise. A weak answer interprets the spike as risk.

**6. Why is Sentinel not a download?**
It is a distributed programme with governed access. A strong answer names the access route. A weak answer assumes open data.

**7. What can ClinicalTrials.gov tell you?**
What trials exist, their status, eligibility, and some results. It cannot tell you the complete results or the individual patient data. A strong answer notes selective reporting. A weak answer treats it as a results database.

**8. Why request individual patient data?**
To reanalyse outcomes, subgroups, and time-to-event with full control. A strong answer notes the review and the analysis plan. A weak answer treats it as a faster publication.

**9. What is channeling bias?**
Prescribing a drug to patients with a different prognosis. It confounds comparisons. A strong answer proposes confounding control. A weak answer compares crude rates.

**10. When would you recommend no label change?**
When the signal is explained by bias, duplicates, or confounding, and the evidence does not support a warning. A strong answer documents the negative assessment. A weak answer ignores the signal.

**11. How do you report a safety analysis to a regulator?**
Data cut, method, cases, denominator, bias, causality, and benefit-risk. A strong answer follows the format. A weak answer reports a ratio.

**12. What belongs in a safety data memo?**
Numerator, denominator, bias, interval, source, cut date, and prohibited claims. A strong answer states the evidence grade. A weak answer summarizes the count.

**13. How do you handle a report with multiple drugs?**
Attribute the event to each drug in separate two-by-two tables, or use a case-level causality assessment. A strong answer avoids double counting the case. A weak answer counts the case once for every drug.

**14. What would you ask before using a disproportionality result in a label?**
The data cut, the comparator, the duplicates, the bias analysis, the interval, and the causality assessment. A strong answer asks for the full signal evaluation. A weak answer asks for the ratio.

**15. How do you evaluate a vaccine safety signal?**
Use VAERS for hypothesis generation, a denominator-based system for confirmation, and a causal framework for the assessment. A strong answer notes the reporting biases. A weak answer treats VAERS counts as risk.

**16. What is the difference between a data cut and a database version?**
A data cut is the date of the extract; a database version is the release of the database structure or content. Both must be recorded for reproducibility.

**17. Why is under-reporting a structural problem rather than a data-cleaning problem?**
It is a property of the reporting system, not of the records. No cleaning can recover the unreported cases. The analysis must state the limitation.

**18. When is a safety signal strong enough to act on?**
When it is stable after deduplication and bias analysis, plausible, supported by an independent source, and the benefit-risk balance favours action. A strong answer states the evidence grade. A weak answer acts on the ratio alone.

**19. How do you combine spontaneous and claims data?**
Use spontaneous data for hypothesis generation and claims for rate estimation, with a clear mapping of the event and the exposure. A strong answer avoids pooling counts across sources. A weak answer adds the counts together.

**20. What is the first check on a new safety extract?**
The unit and the denominator: is a row a report or a case, and is there an exposure denominator. Then duplicates and the data cut. A strong answer checks these before computing a ratio. A weak answer starts with the ratio.

## 15. Key terms

- **AEMS** — FDA Adverse Event Monitoring System.
- **AACT** — Aggregate Analysis of ClinicalTrials.gov.
- **Adverse event** — an unwanted medical occurrence.
- **Causality assessment** — structured judgement of a drug-event relationship.
- **Channeling bias** — prescribing by prognosis.
- **CIOMS** — Council for International Organizations of Medical Sciences.
- **ClinicalTrials.gov** — US trial registry.
- **Confounding by indication** — the indication causes the event.
- **Data cut** — the date of the database extract.
- **Disproportionality** — event reported more than expected.
- **Duplicate report** — two reports of one case.
- **EBGM** — empirical Bayes geometric mean.
- **EudraVigilance** — EU adverse reaction system.
- **FAERS** — FDA Adverse Event Reporting System.
- **Follow-up report** — an update to a case.
- **GVP** — Good Pharmacovigilance Practices.
- **ICH E2B** — individual case safety report standard.
- **Incidence rate** — new cases per person-time.
- **Individual patient data** — participant-level trial data.
- **MAUDE** — device adverse event database.
- **MedDRA** — event terminology.
- **Notoriety bias** — increased reporting after publicity.
- **PRR** — proportional reporting ratio.
- **Report** — a submitted safety notification.
- **Reporting odds ratio** — ROR.
- **Sentinel** — FDA active surveillance programme.
- **Signal** — a hypothesis requiring evaluation.
- **Spontaneous report** — a voluntary report.
- **Stimulated reporting** — reporting increased by attention.
- **Under-reporting** — cases not reported.
- **VAERS** — vaccine adverse event reporting system.
- **VigiBase** — WHO global case safety database.
- **Vivli** — individual-patient-data platform.
- **Adverse reaction** — an adverse event with a suspected drug relationship.
- **Benefit-risk** — the balance of efficacy and harm.
- **Case narrative** — the free-text case description.
- **Case series** — a collection of cases.
- **Causality category** — certain, probable, possible, unlikely.
- **Coding** — assigning a terminology term.
- **Data lock point** — the date the analysis dataset is fixed.
- **Dechallenge** — stopping the drug.
- **Disproportionality analysis** — comparing report proportions.
- **Event date** — when the event occurred.
- **Exposure** — the population that received the drug.
- **Follow-up** — additional information on a case.
- **Incidence** — new cases per person-time.
- **Individual case safety report** — a structured case report.
- **Label** — the approved product information.
- **Marketing authorisation holder** — the company responsible.
- **Medical review** — clinical assessment of cases.
- **Naranjo scale** — a causality algorithm.
- **Periodic safety report** — a scheduled safety report.
- **Pharmacovigilance** — the science of drug safety.
- **PRR threshold** — the signal rule.
- **Rechallenge** — restarting the drug.
- **Report date** — when the report was submitted.
- **Reporting completeness** — the share of cases reported.
- **Risk management plan** — the safety monitoring plan.
- **RUCAM** — a liver injury causality scale.
- **Serious adverse event** — an event with a serious outcome.
- **Signal detection** — finding disproportionality.
- **Signal evaluation** — assessing a signal.
- **Spontaneous reporting** — voluntary reporting.
- **Temporal plausibility** — timing consistent with causation.
- **Under-reporting** — cases not reported.
- **Vaccine safety** — safety of vaccines.
- **Volume 9A** — an older EU pharmacovigilance guidance.
- **Web report** — a report submitted online.
- **Withdrawal** — removing a product from the market.
- **Yellow Card** — the UK reporting scheme.
- **EudraVigilance access** — the EU access tiers.
- **FAERS public dashboard** — the FDA query tool.
- **MedDRA hierarchy** — system organ class to preferred term.
- **Case validity** — whether the case meets the definition.
- **Duplicate probability** — the chance two reports are the same case.
- **Reporting rate** — reports per exposure.
- **Exposure denominator** — the number exposed.
- **Absolute risk** — incidence in a population.
- **Attributable risk** — excess risk from exposure.
- **Background incidence** — the expected rate without the drug.
- **Case-control study** — a design comparing cases and controls.
- **Cohort study** — a design following exposed and unexposed people.
- **Confidence interval** — uncertainty around an estimate.
- **Counterfactual** — what would have happened without exposure.
- **Drug utilization study** — a study of prescribing patterns.
- **Epidemiological signal** — a signal from population data.
- **Exposure window** — the period during which the drug acts.
- **Healthy-user bias** — users are healthier.
- **Immortal time bias** — future information in exposure.
- **Incidence density** — incidence per person-time.
- **Lag time** — delay between exposure and event.
- **Latency** — the biologically plausible delay.
- **Meta-analysis** — a statistical synthesis of studies.
- **Misclassification** — wrong exposure or outcome assignment.
- **Observational study** — a study without randomisation.
- **Odds ratio** — a ratio of odds.
- **Person-time** — time at risk.
- **Pharmacoepidemiology** — the study of drug effects in populations.
- **Protopathic bias** — the early disease causes the prescription.
- **Real-world data** — data from routine care.
- **Relative risk** — a ratio of risks.
- **Risk difference** — an absolute difference in risk.
- **Sensitivity analysis** — testing assumptions.
- **Standardised incidence ratio** — observed/expected cases.
- **Target trial** — an emulated randomised trial.
- **Unmeasured confounding** — confounding not controlled.
- **Validated outcome** — an outcome with measured accuracy.
- **Adverse event report** — a submitted report.
- **Case definition** — criteria for counting a case.
- **Causality** — the relationship between drug and event.
- **Data quality** — fitness of the data for the analysis.
- **Denominator** — the population at risk.
- **Drug-event pair** — the unit of a two-by-two table.
- **Event coding** — assigning a terminology code.
- **Exposure** — the population that received the drug.
- **Follow-up completeness** — how much is known about a case.
- **Incidence proportion** — cases among exposed people.
- **Indication** — the reason for prescribing.
- **Medication error** — an error in prescribing or use.
- **Narrative review** — reading the case text.
- **Off-label use** — use outside the approved indication.
- **Pharmacodynamic** — what the drug does to the body.
- **Pharmacokinetic** — what the body does to the drug.
- **Prescription event monitoring** — a cohort of new users.
- **Product label** — approved information.
- **Reporting rate** — reports per exposure.
- **Risk minimisation** — measures to reduce harm.
- **Safety communication** — a regulatory alert.
- **Signal validation** — confirming a signal.
- **Spontaneous reporting rate** — reports per exposure.
- **Standardised mortality ratio** — observed/expected deaths.
- **Therapeutic class** — drugs with a shared mechanism.
- **Time to onset** — interval from exposure to event.
- **Under-reporting factor** — the ratio of true to reported cases.
- **Withdrawal study** — a study after market withdrawal.
- **Case ascertainment** — identifying cases in a population.
- **Exposure ascertainment** — identifying who was exposed.

## 16. Sources

All URLs accessed 26 October 2026. Programme names and access routes change; verify at the source.

1. US Food and Drug Administration. FAERS/AEMS. https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers
2. US Food and Drug Administration. Sentinel Initiative. https://www.sentinelinitiative.org/
3. ClinicalTrials.gov. https://clinicaltrials.gov/
4. Clinical Trials Transformation Initiative. AACT. https://aact.ctti-clinicaltrials.org/
5. European Medicines Agency. EudraVigilance. https://www.ema.europa.eu/en/human-regulatory-overview/research-development/pharmacovigilance-research-development/eudravigilance
6. Uppsala Monitoring Centre. VigiBase. https://www.who-umc.org/vigibase/
7. US Department of Health and Human Services. VAERS. https://vaers.hhs.gov/
8. US Food and Drug Administration. MAUDE. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfMAUDE/search.CFM
9. Vivli. https://vivli.org/
10. Yale Open Data Access. https://yoda.yale.edu/
11. ICH. E2B guideline. https://www.ich.org/page/efficacy-guidelines
12. MedDRA. https://www.meddra.org/
13. Council for International Organizations of Medical Sciences. https://cioms.ch/
14. Module 21 survey file, section 12. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, public health datasets: [`09_public-health-datasets.md`](09_public-health-datasets.md)
- Text and LLM benchmarks: [`08_text-and-llm-benchmarks.md`](08_text-and-llm-benchmarks.md)
- Access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the benchmark validity audit: [`../../labs/`](../../labs/)

