# Safety and Pharmacovigilance: From Case Report to Signal Decision

> Module 13 — Biopharma and Clinical Development
> Topic 06 of 10
> Estimated study time: 5 hours
> Prerequisites: Module 04 (statistics); topics 02 and 05
> Version 2026.3

## 1. Why this topic exists

Every drug that reaches the market has passed a safety assessment that was, by construction,
incomplete. Phase III trials enrol hundreds to a few thousand patients, exclude the very old, the
very sick, and those on interacting medications, and follow patients for months rather than years.
They can detect common adverse events with reasonable precision and cannot detect events that occur
once in ten thousand patients. The post-marketing system exists to find those events, and it operates
on data that are biased in ways that are easy to forget.

The core problem is denominators. A spontaneous reporting system counts reports, not patients, and
the number of reports depends on the number of people exposed, the severity of the event, the
attention of the reporter, media coverage, and litigation. A report count cannot be divided by an
exposure count to produce an incidence, because the numerator and the denominator come from different
populations. Disproportionality metrics such as the proportional reporting ratio (PRR) and the
reporting odds ratio (ROR) address this by comparing the proportion of a given event among reports for
one drug with the proportion among reports for other drugs. They are screening tools, not measures of
risk, and the distinction is the difference between a hypothesis and a conclusion.

The commercial and human stakes are high in both directions. A missed signal can harm patients and
destroy a product. A false signal can withdraw a useful drug, trigger litigation, and consume
regulatory capacity. The discipline of pharmacovigilance is the management of that trade-off: detect
early enough to act, and validate thoroughly enough not to act wrongly.

This topic covers the case-reporting pipeline, the coding and causality judgements that determine
what a case means, the arithmetic of disproportionality and its limits, the process of signal
validation and assessment, and the aggregate reporting obligations. It follows the data standards
topic, because safety data are among the most standardised in the industry, and precedes the
real-world evidence topic, because active surveillance is the complement to spontaneous reporting.

### 1.1 The failure pattern

1. A new drug launches with a safety database of 3,400 patients from Phase III.
2. Eighteen months later, a cluster of reports of a rare hepatic event appears in the spontaneous
   reporting system.
3. The disproportionality metric is elevated: PRR 2.98 against all other drugs.
4. The safety team treats the elevation as evidence of a causal association and proposes a label
   restriction.
5. No one compares the drug with other drugs in the same class, which have a similar event rate.
6. The restriction is negotiated, the label narrows, and subsequent epidemiological study finds no
   excess risk relative to the class.

Step 5 is the analytical failure. The comparison group determines the answer, and "all other drugs"
is almost never the right comparison for a class effect. The epidemiological study in step 6 was
needed before the label change, and the disproportionality signal alone could not distinguish a drug
effect from a class effect, an indication effect, or a reporting artefact.

### 1.2 Who is blocked when safety data are mishandled

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Safety physician | A defensible causality assessment | Receives coded cases without clinical context |
| Patient | Timely warning or continued access | Has no view of the evidence |
| Regulatory authority | Capacity to review real signals | Receives noise and under-reporting together |
| Commercial team | A product's label and market | Cannot influence the safety evidence |
| Clinical team | A trial that continues safely | Depends on the safety database's review |
| Legal | Exposure to litigation | Sees the decisions after they are made |

### 1.3 The industry question

> Is this pattern of reports a signal of a drug effect, an artefact of reporting, or a class or
> indication effect that the drug shares with its comparators?

The question has three parts because a disproportionality metric cannot distinguish them. Answering
it requires the metric, the case narratives, the comparator choice, and the epidemiological context.

### 1.4 Three questions that keep the analysis honest

1. **What is the denominator?** If it is reports, the result is a reporting proportion. If it is
   patients or person-time, the result can be a rate.
2. **What is the comparator?** The answer changes with the comparator, so the comparator is part of
   the finding, not a technical detail.
3. **What would refute the signal?** A case series that does not replicate, a class comparison that
   shows the same pattern, or an epidemiological study with no association.

A safety assessment that cannot answer these three questions is a narrative, not an analysis. The
questions also explain why the same database can support opposite conclusions: two analysts can choose
different comparators and both be internally consistent.

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a case from the initial report through triage, coding, seriousness assessment,
   expectedness, causality, and submission.
2. **Compute** an adverse event incidence with a confidence interval and state the population and
   denominator.
3. **Compute** PRR, ROR, and their confidence intervals from a two-by-two table, and interpret them
   as screening metrics rather than risks.
4. **Choose** a comparator group and explain how the choice changes the conclusion.
5. **Distinguish** a signal, a validated signal, and a confirmed risk, and name the evidence each
   requires.
6. **Critique** a signal assessment that ignores duplicate reports, notoriety, or confounding by
   indication.
7. **Map** the aggregate reporting obligations (DSUR, PSUR/PBRER) and their data requirements.

## 3. The core mental model

### 3.1 The case pipeline

```text
 Reporter                Intake                 Processing                Decision
 --------                ------                 ----------                --------
 Patient / HCP   ---->   Case intake      ---->  Triage              ---->  Serious?  ---+
 / literature /          (minimum criteria)      (valid case?)             Expected?     |
 regulator                                                                  Related?      |
                              |                     |                                    |
                              v                     v                                    v
                        Duplicate check        Coding (MedDRA)                     Expedited report
                        (worldwide)            Causality                         or periodic report
                              |                     |                                    |
                              +---------------------+------------------------------------+
                                                    |
                                                    v
                                          Safety database (case series)
                                                    |
                                    +---------------+---------------+
                                    |                               |
                                    v                               v
                          Disproportionality                 Clinical review
                          (PRR, ROR, BCPNN)                  (narratives, mechanism)
                                    |                               |
                                    +---------------+---------------+
                                                    v
                                          Signal validation and assessment
                                                    |
                                                    v
                                     Label / RMP / communication / study
```

The pipeline has two properties that matter for analysis. First, each step loses cases: invalid
reports, duplicates, and unassessable cases reduce the denominator before any metric is computed.
Second, the coding step converts free text into a dictionary term, and the chosen term determines
whether the case enters a given analysis.

### 3.2 Reports are not patients

| Quantity | What it counts | Can it be divided by exposure? |
|---|---|---|
| Report | A submission to a safety database | No, unless exposure is known |
| Case | A unique patient-event combination after deduplication | No, unless exposure is known |
| Event | A coded term within a case | No |
| Patient | An exposed individual | Yes, if the cohort is known |
| Person-time | Time at risk across exposed patients | Yes, and it is the preferred denominator |

The distinction is why spontaneous reporting cannot produce incidence. A report rate can rise because
reporting improved, not because risk increased. The industry response is disproportionality, which
compares reporting patterns within the database, and active surveillance, which measures rates in a
defined population.

### 3.3 The two-by-two table

```text
                     Event of interest        All other events
 Drug of interest            a                       b
 All other drugs             c                       d
```

```text
PRR = [a / (a + b)] / [c / (c + d)]
ROR = (a / b) / (c / d)
```

The PRR compares the share of the event among the drug's reports with the share among comparator
reports. The ROR compares the odds. Both are measures of disproportionality, not of risk. Neither
has a denominator of exposed patients, and neither can be interpreted as a relative risk.

### 3.4 Signal, validated signal, confirmed risk

| Stage | Definition | Evidence | Typical action |
|---|---|---|---|
| Signal | Information suggesting a new potentially causal association or a new aspect of a known association | One or more case reports, a disproportionality finding, a trial finding | Validate |
| Validated signal | A signal for which the available documentation supports further assessment | Case series review, temporality, plausibility, consistency | Assess |
| Confirmed risk | An association supported by adequate evidence, often epidemiological | Controlled study, dose-response, causality assessment | Label, risk management, communication |
| Refuted | Evidence against the association | Comparator data, epidemiological study, mechanism | Document and close |

The progression is not automatic. Most signals are validated and then assessed as not requiring
action. A programme that acts at the first stage will over-react; one that waits for the third will
under-react.

### 3.5 What changes the answer

| Factor | Effect on a disproportionality result | How to check |
|---|---|---|
| Comparator choice | Can move PRR from 3.0 to 1.2 | Compare with the class, not all drugs |
| Duplicate reports | Inflates the numerator if selective | Deduplicate by patient and event |
| Notoriety and media | Increases reporting after publicity | Examine the time trend |
| Confounding by indication | The disease, not the drug, causes the event | Compare with untreated or other-treated patients |
| Coding changes | Moves cases between terms | Check dictionary version and term mapping |
| Under-reporting | Reduces the numerator, usually non-differentially | Cannot be corrected within the database |
| Healthy-user bias | Treated patients may be healthier | Compare with the indication population |

The first factor is the one that most often determines the conclusion, and it is a design choice made
before any arithmetic.

### 3.6 What a screening system actually sees

A screening system does not see patients. It sees a matrix of drug-event combinations, most of which
have very few reports. For a product with 12,480 reports and 8,000 possible preferred terms, the
median combination has one or two reports, and the statistical behaviour of a ratio computed from two
reports is dominated by noise.

This is why screening systems impose a minimum case count. It is also why the output is a ranked
list, not a set of findings: the system's job is to order the queue so that a clinician reviews the
most promising combinations first. A programme that treats the ranking as a set of conclusions has
misunderstood the tool.

The practical consequence is that recall matters more than precision at the screening stage. A missed
signal is more costly than a reviewed false positive, because the false positive is closed in an hour
and the missed signal may take years to resurface.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Adverse event (AE) | Any untoward medical occurrence in a patient administered a product | Used as a synonym for adverse reaction | Event |
| Adverse reaction (AR) | An AE for which a causal relationship is considered at least reasonable | Used for any AE | Event |
| Serious adverse event (SAE) | An AE that results in death, is life-threatening, requires hospitalisation, causes disability, or is a congenital anomaly | Confused with severity | Event classification |
| Severity | The intensity of an event, e.g. grade 1-5 | Confused with seriousness | Grade |
| Expectedness | Whether an event is listed in the reference safety information | Confused with causality | Classification |
| Causality | The assessment of whether the product caused the event | Treated as a binary | Category or probability |
| Individual case safety report (ICSR) | A report of one or more AEs in one patient for one product | Confused with an event | Report |
| Minimum criteria | The four elements required for a valid case: reporter, patient, product, event | Assumed to be sufficient | Criteria |
| MedDRA | Medical Dictionary for Regulatory Activities | Used for non-regulatory coding | Dictionary, versioned |
| Preferred term (PT) | The MedDRA term for a specific event | Confused with the verbatim term | Term |
| System organ class (SOC) | The MedDRA hierarchy grouping | Used as a clinical category | Hierarchy |
| Disproportionality | Over-representation of an event among reports for a drug | Interpreted as risk | Metric |
| PRR | Proportional reporting ratio | Interpreted as relative risk | Ratio |
| ROR | Reporting odds ratio | Interpreted as odds of disease | Ratio |
| BCPNN | Bayesian confidence propagation neural network, an empirical Bayes method | Treated as a frequentist test | Metric |
| Signal | Information suggesting a new causal association | Confused with a confirmed risk | Stage |
| Signal validation | The process of checking whether a signal is supported by documentation | Confused with assessment | Process |
| Signal assessment | The evaluation of a validated signal for action | Confused with validation | Process |
| Periodic safety update report (PSUR) | Periodic aggregate report in the EU and ICH regions | Confused with the DSUR | Report |
| PBRER | Periodic benefit-risk evaluation report, the ICH format for PSURs | Used as a synonym for PSUR without the format distinction | Report |
| Development safety update report (DSUR) | Annual aggregate report during development | Confused with the PSUR | Report |
| Risk management plan (RMP) | The plan for pharmacovigilance and risk minimisation activities | Treated as a label annex | Document |
| EudraVigilance | The EU database for adverse reaction reports | Confused with AEMS | Database |
| AEMS | FDA Adverse Event Monitoring System, formerly FAERS | Called FAERS after the rename | Database |
| E2B(R3) | ICH standard for electronic transmission of ICSRs | Confused with the report content standard | Standard |
| Notoriety bias | Increased reporting after publicity | Ignored in trend analysis | Bias |
| Confounding by indication | The indication, not the drug, explains the association | Ignored in spontaneous data | Bias |
| Duplicate | The same case reported more than once | Counted as separate cases | Case status |
| Follow-up report | An update to an existing case | Counted as a new case | Case status |
| Person-time | Time at risk across exposed patients | Absent from spontaneous data | Unit |
| Active surveillance | Systematic collection of safety data in a defined population | Confused with spontaneous reporting | Method |

### 4.1 Seriousness, severity, and expectedness are three different judgements

- **Seriousness** is a regulatory classification based on the outcome (death, hospitalisation,
  disability, life-threatening, congenital anomaly, or medically important). It determines reporting
  timelines.
- **Severity** is the intensity of the event (mild, moderate, severe, or grade 1-5). A severe
  headache may not be serious; a mild hospitalisation is serious by definition.
- **Expectedness** compares the event with the reference safety information. An expected event may
  still be serious, and an unexpected event may be non-serious.

A case assessment that conflates the three produces either an unnecessary expedited report or a
missed one, and both are regulatory findings.

## 5. The mechanics, step by step

This procedure describes the path from a report to a signal decision. It is written for a sponsor's
pharmacovigilance function but applies to any organisation that receives safety reports.

### Step 1. Intake and validity

**Input:** a report from a health professional, patient, literature, or regulator.

**Output:** a valid case with the four minimum elements (identifiable reporter, identifiable patient,
suspect product, adverse event), or a record of why it is invalid.

**Rationale:** An invalid case cannot be submitted and cannot be analysed. The minimum criteria are a
regulatory convention, and a case missing one element is followed up rather than discarded, because
the information often arrives later.

**Edge case:** a report with multiple products and multiple events. It becomes multiple
product-event combinations, and the causality assessment must be made per combination, not per
report.

### Step 2. Duplicate detection

**Input:** the valid case and the existing database.

**Output:** a link to an existing case (follow-up) or a new case.

**Rationale:** Duplicates inflate counts and distort disproportionality, particularly when
duplication is selective. Detection uses patient identifiers, event terms, dates, and narrative
similarity. The check must be worldwide, because the same case can arrive through a partner, a
regulator, and a literature report.

**Edge case:** the same patient with two different events. This is one case with two events, not two
cases, and the distinction changes the denominator of every metric.

### Step 3. Coding

**Input:** the verbatim event terms and medication names.

**Output:** MedDRA preferred terms and WHO Drug terms, with the dictionary versions recorded.

**Rationale:** Coding determines which cases enter an analysis. A term coded to a broad PT may be
diluted; a term coded to a narrow PT may be missed by a search on the broader concept. The coding
convention should be consistent across the programme, and the version must be declared.

### Step 4. Seriousness and expectedness

**Input:** the case narrative and the reference safety information.

**Output:** a seriousness classification with criteria, and an expectedness classification.

**Rationale:** These determine the reporting timeline and the aggregate analysis. A serious,
unexpected, suspected adverse reaction requires expedited reporting; a non-serious expected event
does not. The reference safety information must be the version in force at the time of the report,
not the current version.

### Step 5. Causality assessment

**Input:** the case, the temporal relationship, the mechanism, and the alternative explanations.

**Output:** a causality category with a rationale, using a defined method (for example WHO-UMC or a
sponsor scale).

**Rationale:** Causality is a judgement, and a structured method makes it reviewable. The method
should separate temporality, plausibility, dechallenge, rechallenge, and alternative causes. A
category without a rationale cannot be reviewed and cannot be defended.

### Step 6. Expedited reporting

**Input:** the assessed case.

**Output:** a submitted ICSR within the regulatory timeline, or a documented decision not to submit.

**Rationale:** Timelines are legal requirements. A fatal or life-threatening unexpected suspected
adverse reaction has a shorter timeline than other serious unexpected reactions, and the clock starts
at the first receipt of the minimum criteria, not at the completion of the assessment.

### Step 7. Database maintenance and reconciliation

**Input:** submitted cases.

**Output:** a clean, deduplicated case series with versioned coding.

**Rationale:** The database is the denominator for every signal analysis. Reconciliation against
clinical trial data, partner databases, and literature is necessary because cases arrive through
multiple channels and the same event can be reported twice.

### Step 8. Signal detection

**Input:** the case series and the comparator data.

**Output:** a list of drug-event combinations that meet the screening criteria, with their metrics.

**Rationale:** Screening is a hypothesis-generation step. The criteria are conventions (for example,
a lower confidence bound above one and a minimum case count), not tests of causality. A combination
that fails the criteria may still be a signal on clinical grounds, and one that passes may be noise.

### Step 9. Signal validation

**Input:** the screening output and the case narratives.

**Output:** a validated or refuted signal, with a documented rationale.

**Rationale:** Validation asks whether the cases are real, whether they describe the same event, and
whether the association is plausible. It is a clinical review, not a statistical one, and it is where
most screening hits are closed.

### Step 10. Signal assessment

**Input:** the validated signal and the available evidence.

**Output:** an assessment of strength, frequency, seriousness, and preventability, with a
recommendation.

**Rationale:** Assessment weighs the signal against the benefit and against alternative explanations.
It may require an epidemiological study, a class comparison, or a mechanistic study. The output is a
recommendation, not a decision; the decision belongs to the governance body.

### Step 11. Action and communication

**Input:** the assessment and the regulatory obligations.

**Output:** a label update, a risk management plan update, a communication, a study, or no action.

**Rationale:** The action must be proportionate to the evidence. A label update based on
disproportionality alone is not defensible; a decision to take no action without documenting the
rationale is not auditable.

### Step 12. Aggregate reporting and review

**Input:** the cumulative case series and the benefit-risk assessment.

**Output:** a DSUR, PSUR/PBRER, or equivalent, with a summary of new information and an
interpretation.

**Rationale:** Aggregate reports are the mechanism by which a regulator sees the cumulative picture.
They require the same reconciliation discipline as the trial data, because the counts in the report
must match the database and the label.

### 5.3 The reporting clock and its conventions

| Case type | Typical timeline | Clock starts | Source of the rule |
|---|---|---|---|
| Fatal or life-threatening, unexpected, suspected | 7 calendar days, with follow-up within 8 more | First receipt of minimum criteria | ICH E2A, implemented in regional law |
| Other serious, unexpected, suspected | 15 calendar days | First receipt of minimum criteria | ICH E2A |
| Serious, expected | Periodic report | Data lock point of the period | Regional periodic reporting rules |
| Non-serious | Periodic report | Data lock point | Regional periodic reporting rules |

Two practical points follow. First, the clock starts at receipt, not at medical review, so an
organisation that routes cases slowly will fail timeliness even with a fast assessment. Second, the
timeline is a legal requirement in the jurisdiction, while the choice of causality scale and the
internal escalation path are conventions.

### 5.4 Causality methods and what they produce

| Method | Output | Strength | Limitation |
|---|---|---|---|
| WHO-UMC scale | Certain, probable, possible, unlikely, conditional, unassessable | Widely used, structured | Categories overlap; inter-rater variability |
| Naranjo algorithm | Numeric score with a category | Reproducible | Weighting is arbitrary; poor for biologics |
| Bayesian methods | Posterior probability | Explicit priors | Requires data and expertise |
| Sponsor-specific scale | Company categories | Fits internal governance | Not comparable across companies |
| Clinical judgement with rationale | Narrative assessment | Captures context | Not reproducible |

The method matters less than the documentation. A causality category without the reasoning cannot be
reviewed by a regulator, and a programme that uses different methods across regions cannot aggregate
its own data.

### 5.5 Screening criteria in practice

Screening criteria are conventions set by the organisation and, in the EU, by the EMA for
EudraVigilance screening. A typical set combines:

- a minimum number of cases, commonly three for products under additional monitoring and five for
  established products;
- a lower bound of the 95% confidence interval of the ROR above one;
- restriction to important medical events or to events of special interest;
- exclusion of litigation cases and of cases with insufficient information.

The thresholds are empirical and have no gold standard. Raising the case threshold reduces false
positives at the cost of sensitivity, which is why the choice should be stated and justified rather
than inherited.

### 5.6 What an aggregate report must reconcile

An aggregate report (DSUR, PSUR/PBRER) is a reconciliation exercise as much as a narrative. The
counts in the report must match the safety database at the data lock point, the exposure estimates
must match the clinical or marketing data, and the label must match the reference safety information
in force.

| Section | Data required | Common defect |
|---|---|---|
| Exposure | Patients or patient-time by indication and region | Estimates not reconcilable to sales or trial data |
| Case overview | Counts by seriousness, age, sex, region | Duplicates and follow-ups counted as new |
| Signals | Validated signals and their status | Closed signals not documented |
| Benefit-risk | Integrated assessment | Narrative disconnected from the data |
| Risk minimisation | Activities and their evaluation | Activities listed without evidence of effect |

### 5.7 Governance of signal management

Signal management fails organisationally in three ways: nobody owns the queue, the queue is owned by
a function without authority to act, or the decision is made without a record.

A workable structure separates four roles:

| Role | Responsibility | Independence |
|---|---|---|
| Signal detection | Runs screening and triage | Operational |
| Safety physician | Validates and assesses | Clinical |
| Safety governance board | Decides on action | Cross-functional |
| Independent reviewer | Challenges the assessment | Not accountable to the programme |

The independent reviewer is the equivalent of the gate challenger in topic 01. Without that role,
the assessment reflects the programme's preference, and the preference is usually to avoid a label
restriction.

### 5.8 Trial safety and post-marketing safety are different systems

| Property | Trial safety | Spontaneous reporting |
|---|---|---|
| Denominator | Known: randomised or dosed patients | Unknown |
| Comparison | Concurrent control | Historical or external |
| Event ascertainment | Active, protocol-driven | Passive, reporter-driven |
| Reporting timeline | Expedited plus annual report | Expedited plus periodic |
| Confounding | Randomisation | Indication, age, comorbidity |
| Strength | Causal inference for common events | Detection of rare events |

A programme that applies trial methods to spontaneous data, or spontaneous methods to trial data,
will misread both. The trial answers "does the drug cause this event in this population at this
frequency"; the spontaneous system answers "is there a pattern worth investigating".

### 5.9 Data sources for surveillance

| Source | Coverage | Strength | Limitation |
|---|---|---|---|
| AEMS (formerly FAERS) | US reports, public | Large, accessible | Under-reporting, duplicates |
| EudraVigilance | EU reports | Large, regulatory | Access restrictions for full data |
| VAERS | US vaccine reports | Vaccine-specific | Stimulated reporting |
| Sentinel | US claims and EHR | Defined population, active | Selected products and outcomes |
| National registries | Disease or product specific | Complete for the disease | Selection, size |
| EHR networks | Provider-based | Denominators available | Coding, coverage |
| Literature | Published cases | Detailed | Publication bias |

The choice of source determines what can be estimated. Only the sources with a defined population can
produce incidence; the spontaneous sources can produce disproportionality.

### 5.10 Coding traps that change counts

| Trap | Example | Effect |
|---|---|---|
| Broad term captures the syndrome | "Hepatotoxicity" coded to a single PT | Cases diluted across narrower terms |
| Narrow term misses the concept | Searching one PT only | Cases coded to related terms are missed |
| Verbatim retained as coded | "Liver failure" used as the PT | Invalid term; excluded from the analysis |
| Multiple events in one case | Hepatic failure and encephalopathy | Counted once or twice depending on the query |
| MedDRA version drift | Term split or merged between versions | Count changes without a clinical change |
| WHO Drug ingredient versus product | Combination product coded to one ingredient | Exposure miscounted |

A signal analysis should state its search strategy: the PTs, the SOC, the version, and whether it
searched the narrow or broad scope. A result without a search strategy cannot be reproduced.

The same discipline applies to the comparator: state which drugs were included, why, and how many
reports they contributed. A comparator that is described only as "all other drugs" is not a
definition.

### 5.1 Decision table for a screening hit

| Screening result | Clinical review | Recommended outcome | Condition |
|---|---|---|---|
| PRR elevated, ≥3 cases, plausible, consistent | Cases describe a coherent syndrome | Validate and assess | Comparator analysis and literature review |
| PRR elevated, cases heterogeneous | Different clinical events coded to one term | Refute as a single signal; consider separate terms | Coding review |
| PRR elevated, explained by duplicates | Duplicate reports concentrated in the event | Recompute after deduplication | Deduplication before any metric |
| PRR elevated, explained by notoriety | Reporting spike after publicity | Assess with a time-series adjustment | Compare pre- and post-publicity |
| PRR not elevated, but cases are serious and plausible | Rare serious event below screening threshold | Validate on clinical grounds | Case-by-case review |
| PRR elevated only against all drugs, not the class | Class effect or indication confounding | Do not restrict on this evidence alone | Class comparator and epidemiological study |
| PRR elevated in one country only | Reporting or regulatory difference | Investigate before concluding | Country-level review |

### 5.2 Edge cases that change the arithmetic

**Duplicates and follow-ups.** A follow-up report updates an existing case. Counting it as new
inflates the numerator, and if the event is serious it is more likely to be followed up, so the
inflation is selective.

**Stimulated reporting.** Media coverage or a regulator's communication increases reporting. The
increase is real in the database and not necessarily real in the population, which is why time-trend
analysis is part of validation.

**Confounding by indication.** A drug used for a disease that itself causes the event will show
elevated reporting without a drug effect. The comparator should be a drug used in the same
indication, or the analysis should adjust for the indication.

**Healthy-user bias.** Patients who receive a new drug may be healthier than the general population,
which biases spontaneous comparisons in the opposite direction.

**Coding version changes.** A MedDRA migration can move cases between terms, changing counts without
any change in patients.

**Multiple events per case.** The unit of analysis is the case-event combination, but the case is the
unit for duplicates. Mixing the two produces inconsistent denominators.

**Under-reporting.** Spontaneous systems capture a small and unknown fraction of events. The fraction
varies by event severity, time since launch, and country, which is why the absence of reports is not
evidence of safety.

**Litigation and legal cases.** Some programmes exclude litigation cases from screening because the
reporting is driven by legal rather than clinical considerations. The exclusion must be stated.

## 6. Worked example

> **Note:** The drug, trial, and database counts are fictional and constructed for teaching. The
> regulatory criteria and databases cited are real and referenced in Section 16.

### 6.1 The trial safety data

A Phase III programme randomised 1,204 patients; 1,198 received at least one dose and form the safety
population. During the study, 96 patients experienced at least one serious adverse event (SAE).

```text
SAE incidence (safety population) = 96 / 1,198 = 8.01%
Wilson 95% CI                     = 6.61% to 9.69%
```

The historical SAE rate for the comparator regimen in this indication is 6.5%. The confidence
interval's lower bound (6.61%) is just above 6.5%, so the difference is nominally statistically
detectable, but the absolute difference is 1.5 percentage points and the interval is wide. A safety
review that reports "a statistically significant increase" without the absolute difference and the
interval is technically correct and clinically misleading.

### 6.2 The spontaneous reports

Eighteen months after launch, the safety database contains 12,480 reports for the drug. Of these,
186 mention hepatic failure. The comparator database of all other drugs contains 4,920,000 reports,
of which 24,600 mention hepatic failure.

```text
PRR = (186 / 12,480) / (24,600 / 4,920,000)
    = 0.014904 / 0.005000
    = 2.98

ROR = (186 / 12,294) / (24,600 / 4,895,400)
    = 0.015129 / 0.005025
    = 3.01
95% CI for ROR = 2.60 to 3.48
Chi-square (Yates) = 242.2
```

The screening criteria used by the programme are: at least three cases, lower bound of the ROR
confidence interval above one, and a clinically coherent case series. The combination passes the
statistical criteria comfortably.

### 6.3 The comparator choice

The drug belongs to a class of four agents used in the same indication. Reports for the class
comparator (a marketed drug in the same class) total 300,000, with 3,600 hepatic failure reports.

```text
PRR (class comparator) = (186 / 12,480) / (3,600 / 300,000)
                       = 0.014904 / 0.012000
                       = 1.24

ROR = 1.25 (95% CI 1.07 to 1.45)
Chi-square (Yates) = 8.2
```

The signal shrinks from a PRR of 2.98 to 1.24 when the comparator is the class rather than all other
drugs. The class comparator has a hepatic failure reporting proportion six times higher than the
all-drug background, which is consistent with a class effect or with the indication itself.

### 6.4 What the two analyses mean together

The trial data show an SAE rate modestly above the historical comparator, with a wide interval. The
spontaneous data show an elevated reporting proportion against all drugs and a much smaller elevation
against the class. Neither analysis establishes causation, and together they suggest three
possibilities:

1. A class effect that the drug shares with its comparators.
2. An indication effect that affects all drugs used in this disease.
3. A drug-specific effect that is smaller than the all-drug comparison suggests.

Distinguishing them requires an epidemiological study in a defined population, with a comparator
cohort and adjustment for indication severity. The disproportionality analysis has done its job: it
generated a hypothesis and narrowed the question. It has not answered it.

### 6.5 Two diagnostics that reveal a broken analysis

**Diagnostic 1 — the denominator is reports, not patients.** If the assessment states an "incidence"
of 186/12,480 = 1.49%, it has divided reports by reports. The number is the proportion of reports
mentioning the event, not the proportion of patients experiencing it. The diagnostic is to ask what
the denominator counts; if the answer is "reports", the metric is a reporting proportion and must be
labelled as such.

**Diagnostic 2 — the comparator includes the drug's own class.** The all-other-drugs comparator
contains the three other class members, whose higher reporting proportion inflates the contrast. The
diagnostic is to decompose the comparator: compute the metric against the class, against the
indication, and against all drugs, and report all three. If the conclusion changes across
comparators, the all-drug result is not the finding; the pattern across comparators is.

A third check is the duplicate rate. If 40% of the hepatic failure reports are follow-ups to existing
cases while the overall duplicate rate is 5%, the numerator is selectively inflated. Deduplicating
selectively reduces the PRR from 2.98 to about 2.39 in this example, which still passes the
screening criteria but changes the strength of the signal.

### 6.6 Vary one assumption: selective duplication

The programme's deduplication audit finds that hepatic failure reports are more likely to be
followed up than other reports. After removing duplicate follow-ups, the drug has 9,360 unique
reports, of which 112 are unique hepatic failure cases.

```text
PRR = (112 / 9,360) / (24,600 / 4,920,000)
    = 0.011966 / 0.005000
    = 2.39
```

The signal remains above the screening threshold but is 20% weaker. The decision does not change:
the signal is validated and referred for assessment. What changes is the urgency and the framing of
the communication. A programme that had reported 2.98 without the deduplication would have overstated
the strength of a signal that it was about to escalate.

### 6.7 What the example establishes

1. Trial incidence and spontaneous disproportionality answer different questions and cannot be
   substituted for each other.
2. The comparator choice can change the conclusion more than the metric choice.
3. Duplicates and follow-ups are a selective bias, not a rounding error.
4. A screening hit is a hypothesis; the assessment requires a defined population and a comparator.
5. A statistically detectable difference in a trial can be clinically small, and the interval carries
   the information that the p-value hides.

### 6.8 The study that would settle the question

The signal can be assessed properly only in a defined population with a comparator. The design would
be a cohort study using claims or EHR data:

- **Exposed cohort:** patients initiating the drug after launch, with a new-user design.
- **Comparator cohort:** patients initiating a class comparator, matched or weighted on indication
  severity, age, and comorbidity.
- **Outcome:** hospitalisation with hepatic failure, validated against records in a sample.
- **Time zero:** the date of initiation, aligned across cohorts.
- **Analysis:** hazard ratio with adjustment, and a sensitivity analysis using a negative control
  outcome.

The study's feasibility depends on the number of exposed patients and the outcome frequency. With
12,000 exposed patients and an outcome rate of 0.3% per year, the expected number of events is 36,
which is enough to detect a doubling but not a 20% increase.

The power arithmetic is worth doing before commissioning the study. At 36 expected events, the
study's confidence interval for a hazard ratio will span roughly 0.7 to 1.4 around a null result,
which cannot exclude a clinically important increase. A sponsor that needs to exclude a 1.5-fold
increase needs several hundred events, and therefore either a larger cohort or a longer follow-up.
Commissioning a study that cannot answer the question is a way of deferring the decision, not making
it.

### 6.9 The decision record

The assessment should record: the screening metric and comparator; the deduplicated case count; the
case-series review conclusion; the class and indication comparisons; the planned epidemiological
study; the interim risk-minimisation action; and the trigger for re-review. The record is what allows
a later reviewer to see why the programme acted as it did, which is the difference between a
defensible decision and a retrospective rationalisation.

A final element is the re-review trigger. A signal assessment with no trigger becomes a document that
is filed and forgotten. The trigger should be a date, a case count, or the completion of the
epidemiological study, and it should be owned.

The assessment should also state what would change the conclusion: a class comparator result below
1.1, an epidemiological hazard ratio below 1.2, or a case series that does not replicate. A signal
assessment without a falsification condition is a position, not an analysis.

## 7. Code, query, or calculation

The script computes the trial incidence with a Wilson interval and the disproportionality metrics
with confidence intervals and a chi-square test. It uses SciPy for the chi-square and the standard
library for the rest.

```python
import math
from scipy.stats import chi2_contingency

def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

def disproportionality(a, n_drug, c, n_other, z=1.959964):
    """PRR, ROR, ROR CI, and Yates chi-square from a 2x2 table."""
    b = n_drug - a
    d = n_other - c
    prr = (a / n_drug) / (c / n_other)
    ror = (a / b) / (c / d)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    lo = math.exp(math.log(ror) - z * se)
    hi = math.exp(math.log(ror) + z * se)
    chi2, _, _, _ = chi2_contingency([[a, b], [c, d]], correction=True)
    return prr, ror, lo, hi, chi2

lo, hi = wilson(96, 1198)
print(f"SAE 96/1198 = {96/1198:.2%} (95% CI {lo:.2%} to {hi:.2%})")

for label, args in [
    ("all other drugs", (186, 12480, 24600, 4920000)),
    ("class comparator", (186, 12480, 3600, 300000)),
    ("after dedup", (112, 9360, 24600, 4920000)),
]:
    prr, ror, lo, hi, chi2 = disproportionality(*args)
    print(f"{label:<16} PRR {prr:.2f}  ROR {ror:.2f} ({lo:.2f}-{hi:.2f})  chi2 {chi2:.1f}")
```

Expected output (Python 3.12, SciPy 1.17, executed):

```text
SAE 96/1198 = 8.01% (95% CI 6.61% to 9.69%)
all other drugs  PRR 2.98  ROR 3.01 (2.60-3.48)  chi2 242.2
class comparator PRR 1.24  ROR 1.25 (1.07-1.45)  chi2 8.2
after dedup      PRR 2.39  ROR 2.41 (2.00-2.90)  chi2 89.5
```

Reading the output:

- The trial incidence is 8.01% with an interval of 6.61% to 9.69%. The interval, not the point
  estimate, is the finding.
- The all-drug comparison gives a PRR of 2.98 and an ROR of 3.01 with a narrow interval. The
  statistical evidence for disproportionality is strong.
- The class comparison gives a PRR of 1.24 and an ROR of 1.25 with an interval that excludes one.
  The disproportionality is weak and consistent with a class reporting pattern.
- Deduplication reduces the all-drug PRR to 2.39, which still passes screening. The direction is
  unchanged; the strength is lower.

Assumptions: the two-by-two table treats each report as one unit; the comparator counts are taken as
fixed; the chi-square uses Yates correction; the Wilson interval assumes independent observations.
The first assumption is the most consequential, because reports are not independent patients.

The script deliberately does not compute a confidence interval for the PRR. The PRR's sampling
distribution is less well behaved than the ROR's, and the ROR interval is the conventional choice for
screening. A programme that reports a PRR without an interval is reporting a point estimate from a
screening tool.

A stratified version answers the comparator question directly. Instead of comparing the drug with all
other drugs, compare it with a named class:

```python
# same function, different comparator column
class_cases, class_total = 3_600, 300_000
prr, ror, lo, hi, chi2 = disproportionality(186, 12_480, class_cases, class_total)
print(f"vs class: PRR {prr:.2f}, ROR {ror:.2f} ({lo:.2f}-{hi:.2f})")
```

```text
vs class: PRR 1.24, ROR 1.25 (1.07-1.45)
```

The two lines of output are the whole argument of Section 6.3. The metric did not change; the
comparator did, and the conclusion moved from a strong signal to a weak one.

## 8. Data, standards, and regulatory context

### 8.1 The instruments

| Instrument | Scope | Bindingness | What it does not cover |
|---|---|---|---|
| ICH E2A | Clinical safety data management, definitions, expedited reporting | ICH guideline, implemented regionally | Signal detection methods |
| ICH E2B(R3) | Electronic transmission of individual case safety reports | ICH standard; mandated by FDA through ESG NextGen | Case quality |
| ICH E2C(R2) | Periodic benefit-risk evaluation reports | ICH guideline | National periodic reporting formats |
| ICH E2D | Post-approval safety data management | ICH guideline | Active surveillance design |
| ICH E2E | Pharmacovigilance planning | ICH guideline | Signal assessment methods |
| EU GVP Modules | Good pharmacovigilance practice, including Module IX on signal management | EU implementing rules | US practice |
| FDA 21 CFR 314.80 and 314.81 | Post-marketing reporting of adverse drug experiences; periodic reports | US regulation | Signal detection methodology |
| FDA AEMS | The FDA adverse event database, formerly FAERS | Operational system | Causality |
| EudraVigilance | The EU adverse reaction database | EU system | US reports |
| MedDRA | Coding dictionary | Standard required in most jurisdictions | Case assessment |

### 8.2 AEMS and the 2026 transition

The FDA has replaced the FDA Adverse Event Reporting System (FAERS) with the Adverse Event Monitoring
System (AEMS), described by the agency as a unified platform covering drugs, biologics, vaccines,
devices, tobacco, food, cosmetics, and veterinary products. The agency launched a public look-up tool
in March 2026 and has stated that electronic submissions will require E2B(R3) through ESG NextGen
from 1 October 2026. Historical data are being migrated and legacy systems decommissioned on a
staggered schedule.

The practical consequences for a programme are:

- **Submission format.** E2B(R3) becomes the required electronic format for FDA submissions, so
  legacy E2B(R2) pipelines need a migration plan.
- **Data access.** Public dashboards are being consolidated, and the query interfaces may change.
  Analyses that depend on a specific download should record the extraction date and the source
  version.
- **Naming.** Documents that refer to FAERS may be referring to the same underlying reports under the
  new system name. A data provenance note should state which system and date were used.

A programme should verify the current submission requirements with the FDA rather than rely on a
summary, because the transition dates are recent and operationally significant.

### 8.3 Screening criteria are conventions, not requirements

The EMA's 2016 guidance on screening in EudraVigilance moved the method of choice from PRR to ROR and
set criteria that include a lower 95% confidence bound above one, a minimum of three cases for
products under additional monitoring and five for others, and restriction to important medical
events. These are EU screening conventions. The FDA does not prescribe a statistical screening rule
in the same way, and a sponsor's internal criteria may differ.

The distinction matters because a programme can comply with the EU convention and still miss a signal
that clinical review would catch, or flag a signal that the convention would not. The criteria are a
triage mechanism, and the clinical review is the substantive step.

### 8.4 What the regulations require and what they do not

- **Require:** expedited reporting of serious unexpected suspected adverse reactions within defined
  timelines; periodic reporting; a pharmacovigilance system with a qualified person; signal
  management as part of good pharmacovigilance practice in the EU.
- **Do not require:** a specific disproportionality metric, a specific causality scale, or a specific
  threshold for action. Those are scientific and organisational choices.
- **Do not establish:** that a report implies causation, or that a disproportionality finding is a
  risk. The regulations govern process, not inference.

### 8.5 The signal management process in the EU

GVP Module IX describes a cycle rather than a single step: signal detection, validation,
confirmation, analysis and prioritisation, assessment, and recommendation for action. Each stage has
a documented output, and the process must be described in the pharmacovigilance system master file.

The element that surprises teams outside the EU is the requirement to document the outcome of
validation even when the signal is closed. A closed signal with no record is indistinguishable from a
signal that was never reviewed, and inspections treat the two the same way.

### 8.6 Vaccine and device safety differ from drug safety

| Property | Drugs | Vaccines | Devices |
|---|---|---|---|
| Exposure | Patients in treatment | Large healthy populations | Patients and users |
| Event baseline | Disease-related | Very low in healthy people | Procedure and device related |
| Reporting system | AEMS, EudraVigilance | VAERS (US), EudraVigilance | MAUDE, EUDAMED |
| Key confounder | Indication | Temporal association with vaccination | Learning curve, operator |
| Signal tools | Disproportionality | Disproportionality plus observed-to-expected | Failure analysis, registries |

A method developed for drugs does not transfer unmodified. In particular, observed-to-expected
analysis is central for vaccines because the background rate in a healthy population can be
estimated, which is rarely true for a drug used in a diseased population.

### 8.7 What makes a case usable

Not every report can support an assessment. A usable case has an identifiable patient, a suspect
product with dose and dates, an event with onset and outcome, relevant medical history, concomitant
medications, and a reporter who can be contacted. A case missing these elements may still be valid
for reporting, but it cannot support a causality judgement.

The practical consequence is that case series shrink as they are reviewed. A signal validated on 40
reports may rest on 12 usable narratives. The assessment should report both numbers, because the
difference is the uncertainty.

## 9. Industry practice and case evidence

### 9.1 How pharmacovigilance is organised

A marketed product typically has a case-processing team, a medical review team, a signal management
team, and a qualified person responsible for pharmacovigilance in the EU. The teams may be in-house,
outsourced, or mixed. Signal detection is increasingly supported by statistical tools that screen the
whole database monthly and rank combinations for review.

The maturity gap between organisations is large. Some run automated screening with a documented
triage and a governance board; others rely on medical review of individual cases and periodic
literature searches. Both can detect signals; the first is more consistent, and the second depends on
the attention of individuals.

### 9.2 Case: rofecoxib and the limits of spontaneous reporting

Rofecoxib, a cyclo-oxygenase-2 inhibitor, was withdrawn worldwide by its manufacturer on 30 September
2004 after a randomised trial showed an increased risk of cardiovascular events compared with
placebo. The case is a standard reference for several reasons: the signal concerned a common event
(myocardial infarction) rather than a rare one; the relevant comparison was against placebo and
against other anti-inflammatory drugs; and the spontaneous reporting system had not produced a clear
signal before the trial.

**Documented:** the withdrawal date, the trial basis, and the subsequent regulatory and legal
consequences. **Inference:** that spontaneous reporting is poorly suited to detecting an increase in
a common event against a high background, because the event is common in the population and reporters
do not perceive it as drug-related. The inference is consistent with the methodological literature on
disproportionality and common events.

The lesson for this topic is that the absence of a disproportionality signal is not evidence of
safety, particularly for events that are common in the treated population.

### 9.5 Why common events are hard

Disproportionality works best when the event is rare in the comparator population. For a common event,
the comparator proportion is high, the ratio is compressed toward one, and the signal is hard to see.
The worked example's hepatic failure is uncommon; a myocardial infarction in an elderly population is
not.

The methods that address this are the ones that measure rates in a defined population: cohort studies,
self-controlled designs, and active surveillance. They are more expensive and slower, and they are the
only way to answer a question about a common event.

### 9.6 Benefit-risk is not a ratio

A benefit-risk assessment is often presented as a number, and the number conceals the judgements.
There is no common unit for a prevented hospitalisation and an avoided stroke, and the weights depend
on the patient, the disease, and the alternative treatments.

A defensible assessment presents the benefits and the risks separately, in the same population, over
the same time horizon, with the uncertainty stated. A ratio can be computed afterwards if a decision
framework requires one, but the ratio is a summary of judgements, not a substitute for them.

The discipline is to ask what the alternative is. A risk is only interpretable against the risk of
the next best treatment or of no treatment, and that comparison is specific to the indication and
the patient group.

### 9.3 Case: a class effect detected through comparison

When a signal appears for one member of a class, the first analytical question is whether the other
members show it. A class comparison can be computed from the same spontaneous database by partitioning
the comparator, as in the worked example. If the class members show similar reporting proportions,
the signal is likely to be a class or indication effect, and a drug-specific label change may be
inappropriate.

This is a methodological inference rather than a documented case. The practice is standard in signal
assessment, but the outcome depends on the class and the data.

### 9.4 What is documented and what is inferred

| Claim | Status |
|---|---|
| AEMS replaced FAERS and consolidates FDA adverse event reporting | Documented by FDA announcements and pages |
| E2B(R3) becomes mandatory for FDA electronic submissions from 1 October 2026 | Documented in FDA transition communications |
| Disproportionality metrics cannot estimate incidence | Documented in the pharmacoepidemiology literature |
| Most screening hits are not confirmed risks | Inference from the purpose of screening; the exact proportion is not publicly established |
| Class comparison changes the interpretation of a signal | Documented method; the frequency of its use is not measured |
| Spontaneous reporting under-detects common-event risks | Supported by the rofecoxib case and methodological literature |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reports treated as patients | An "incidence" computed from report counts | Denominator error | Inflated or meaningless risk | State the denominator and the population |
| Comparator includes the class | PRR against all drugs only | Contrast contaminated by class members | Overstated signal | Decompose the comparator |
| Duplicates counted as new | Duplicate rate differs by event | Selective inflation | Overstated signal and case counts | Deduplicate before analysis |
| Follow-ups counted as cases | Case counts exceed patient counts | Case status error | Inflated numerator | Link follow-ups to the parent case |
| Notoriety ignored | Reporting spike after publicity | Stimulated reporting | False signal | Time-trend analysis |
| Confounding by indication ignored | Drug used for a disease that causes the event | Indication drives the event | False attribution | Indication comparator or adjustment |
| Coding version drift | Counts change after a migration | Term split or merge | Apparent signal change | Version audit; bridge |
| Seriousness confused with severity | "Severe" used for "serious" | Classification error | Wrong reporting timeline | Train and audit |
| Expectedness assessed against the current label | Cases assessed against a newer reference | Wrong reference version | Misclassified expedited reports | Use the version in force |
| Signal closed without rationale | No record of the decision | Process gap | Repeat investigation; audit finding | Document validation outcomes |
| Aggregate report not reconciled | Counts differ from the database | Separate analyses | Inconsistent regulatory record | Reconcile at lock |
| Causality asserted from temporality | "Post hoc ergo propter hoc" | Single-case reasoning | Over-attribution | Structured causality method |
| Active surveillance absent | Only spontaneous data used | Passive system limits | Rare and common-event signals missed | Complementary cohort studies |

The red flag that is easiest to detect and most consequential is an "incidence" computed from a
spontaneous database. If the denominator is not an exposed population, the number is a reporting
proportion, and calling it an incidence misstates what the data can support.

### 10.1 The cost of a false signal

A false signal is not free. A label restriction based on a spurious association can reduce access for
patients who benefit, shift prescribing to less well studied alternatives, and consume regulatory and
company resources. The cost is asymmetric and hard to reverse, because labels rarely widen after a
restriction.

The discipline that prevents it is the same as the discipline that detects real signals: a comparator
analysis, a case-series review, and an epidemiological study where the question is important enough.
The temptation to act on a single elevated metric is strongest when the event is serious and the
public attention is high, which is exactly when the cost of over-reaction is greatest.

### 10.2 Which metric belongs to which data source

| Data source | Denominator | Appropriate metric | Inappropriate metric |
|---|---|---|---|
| Spontaneous reports (AEMS, EudraVigilance) | Unknown | PRR, ROR, BCPNN | Incidence, relative risk |
| Trial safety population | Randomised or dosed patients | Incidence, rate difference, hazard ratio | Disproportionality |
| Claims or EHR cohort | Defined population and person-time | Incidence rate, hazard ratio | Report counts |
| Registry | Defined disease population | Incidence, standardised ratio | Disproportionality |
| Literature | Published cases | Case series description | Any rate |

The table is a guard against the most common category error in safety analytics: applying a metric to
a data source that cannot support its denominator.

It is also a guard against a subtler error: reporting a metric from a source that can support it but
with a comparator that cannot be interpreted. A trial incidence compared with a historical rate from a
different era has a denominator but not a valid comparison.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Reports are independent | The two-by-two table treats them as units | Duplicates and follow-ups | Deduplication audit |
| The comparator represents the background | Disproportionality is relative to it | Class or indication patterns | Decompose the comparator |
| Coding is consistent | Term-level analysis | Version drift, coding variation | Version audit; term mapping |
| Reporting is stable over time | Trend analysis | Notoriety, regulatory action | Time-series review |
| Under-reporting is non-differential | Comparisons are biased only if reporting differs by drug | Event-specific reporting differences | Cannot be tested within the database |
| Seriousness is classified consistently | Reporting timelines | Site interpretation | Audit by site |
| Causality categories are comparable | Aggregation across cases | Different methods or assessors | Inter-rater review |
| The database is complete for the period | Aggregate reporting | Late-arriving cases | Reconciliation at lock |
| Exposure can be estimated for the population | Incidence and rate | No exposure data | Use sales or cohort data with caveats |
| Screening thresholds are appropriate | Triage | Thresholds set by convention | Sensitivity analysis at alternative thresholds || The event is rare enough for disproportionality to work | The metric's discrimination | Common events compress the ratio | Use a cohort design instead |
| Case narratives are available for validation | Clinical review | Missing narratives | Follow up with reporters |

The deepest limit is that spontaneous reporting measures reporting, not occurrence. Every metric in
this topic inherits that limit, and no analytical refinement removes it. The complement is active
surveillance in a defined population, which measures occurrence and costs far more.

A practical response to the limit is to triangulate: a signal that appears in spontaneous data, in a
trial safety analysis, and in an epidemiological study is a different object from a signal that
appears in one source. The triangulation does not remove the bias in any single source; it makes the
conclusion robust to the bias in each.

## 12. Exercises


The exercises use the worked example's data where possible so that the arithmetic can be checked
against Section 6.

**Exercise 1 — Case pipeline (recall, 20 minutes).**
Deliverable: a diagram of the case pipeline from reporter to aggregate report, with the decision at
each step and the artefact produced.
Acceptance criteria: at least eight steps; seriousness, expectedness, and causality shown as separate
decisions; the reporting clock indicated.
Time: 20 minutes.

**Exercise 2 — Incidence with interval (application, 25 minutes).**
Deliverable: an SAE incidence with a 95% confidence interval for 42 events in 860 dosed patients,
with the population and denominator stated.
Acceptance criteria: interval method named; absolute difference from a 3.5% historical rate stated;
interpretation limited to the trial population.
Time: 25 minutes.

**Exercise 3 — Disproportionality (application, 35 minutes).**
Deliverable: PRR, ROR, ROR confidence interval, and chi-square for a two-by-two table of your choice,
with the comparator stated.
Acceptance criteria: arithmetic shown; interpretation as a screening metric; no causal language.
Time: 35 minutes.

**Exercise 4 — Comparator sensitivity (critique, 40 minutes).**
Deliverable: a 400-word analysis of how the conclusion changes when the comparator is all drugs, the
class, and the indication, using the worked example's numbers.
Acceptance criteria: all three comparators computed; the change in conclusion stated; a recommendation
for which comparator to report.
Time: 40 minutes.

**Exercise 5 — Duplicate audit (application, 30 minutes).**
Deliverable: a plan for a deduplication audit, including the matching variables, the tolerance, and
the effect on the metrics.
Acceptance criteria: matching rule precise; selective duplication addressed; expected effect
quantified for at least one event.
Time: 30 minutes.

**Exercise 6 — Signal assessment memo (design, 50 minutes).**
Deliverable: a two-page signal assessment for a validated signal, including evidence, comparator
analysis, alternative explanations, and a recommendation.
Acceptance criteria: distinguishes signal, validated signal, and confirmed risk; states what would
change the conclusion; assigns an owner and a re-review date.
Time: 50 minutes.

**Exercise 7 — Causality method (application, 35 minutes).**
Deliverable: a causality assessment for three cases using a structured method, with the rationale for
each category.
Acceptance criteria: temporality, dechallenge, alternatives, and plausibility addressed; categories
consistent across cases; no category without a rationale.
Time: 35 minutes.

**Exercise 8 — Aggregate report reconciliation (application, 40 minutes).**
Deliverable: a reconciliation plan for a PSUR, covering case counts, exposure, and label version.
Acceptance criteria: each count tied to a database extract and a date; differences explained;
label version stated.
Time: 40 minutes.

## 13. Self-check answers

**Exercise 1.** The pipeline runs reporter, intake, validity, duplicate check, coding, seriousness,
expectedness, causality, expedited reporting, database, screening, validation, assessment, action,
aggregate report. Seriousness, expectedness, and causality are separate decisions because they answer
different questions and drive different actions. The clock starts at receipt of the minimum criteria.

**Exercise 2.** 42/860 = 4.88%; Wilson 95% CI approximately 3.62% to 6.55%. The historical rate of
3.5% lies below the lower bound, so the increase is statistically detectable, but the absolute
difference is 1.4 percentage points and the population is the trial's, not the label's.

**Exercise 3.** For a = 50, n_drug = 5,000, c = 10,000, n_other = 2,000,000: PRR = (0.01)/(0.005) =
2.00; ROR = (50/4,950)/(10,000/1,990,000) = 0.010101/0.005025 = 2.01; the 95% CI is approximately
1.52 to 2.66; chi-square is large. The interpretation is that reporting of the event is
disproportionately frequent for the drug, not that the drug doubles the risk.**Exercise 3.** For a = 50, n_drug = 5,000, c = 10,000, n_other = 2,000,000: PRR = (0.01)/(0.005) =
2.00; ROR = (50/4,950)/(10,000/1,990,000) = 0.010101/0.005025 = 2.01; the 95% CI is approximately
1.52 to 2.66; chi-square is large. The interpretation is that reporting of the event is
disproportionately frequent for the drug, not that the drug doubles the risk.

The screening criteria would be met: more than three cases, and a lower confidence bound above one.
The conclusion is nevertheless limited, because the comparator is all other drugs and the event may be
a class or indication effect. The next step is validation, not action.

**Exercise 4.** Against all drugs, PRR 2.98; against the class, 1.24; against the indication, a value
between the two depending on the indication's baseline. The conclusion changes from "elevated
reporting consistent with a drug effect" to "reporting similar to the class, consistent with a class
or indication effect". The recommendation is to report all three and to base the assessment on the
class comparator, with the all-drug result as context.

**Exercise 5.** Match on patient identifiers (age, sex, initials where available), event term, event
date, and reporter. Tolerance of one day on dates. The audit should stratify by event seriousness.
If duplicates are concentrated in serious events, the effect on the numerator is selective; the
worked example shows a reduction from PRR 2.98 to 2.39.

**Exercise 6 rubric.** Full marks require a clear evidence table, a comparator analysis, at least
three alternative explanations, a recommendation proportionate to the evidence, and a falsification
condition. Deduct marks for causal language based on disproportionality alone, or for a
recommendation without an owner and date.

**Exercise 7.** A WHO-UMC assessment for a case with a plausible temporal relationship, a positive
dechallenge, no alternative cause, and a known mechanism would be "probable". A case with a plausible
temporal relationship but multiple alternative causes would be "possible". A case with no temporal
relationship would be "unlikely". The rationale must be recorded for each.

**Exercise 8.** The reconciliation should tie case counts to a database extract with a date and
version, exposure to the clinical or sales source with its date, and the label version to the
reference safety information in force. Differences should be explained by duplicates, follow-ups, or
late-arriving cases, and the report should state the data lock point.

## 14. Interview questions

**1. Why can't you compute incidence from FAERS or AEMS?**
Strong: the denominator is unknown, reports are not patients, and reporting depends on severity,
attention, and publicity. Weak: says the data are "dirty" without the denominator argument.

**2. What does a PRR of 3 mean?**
Strong: reporting of the event is three times more frequent among the drug's reports than among
comparator reports; it is a screening signal, not a risk. Weak: says the drug triples the risk.

**3. How would you choose a comparator?**
Strong: the class, the indication, and all drugs, reported together, with the class as the primary
comparison for a suspected class effect. Weak: uses all other drugs by default.

**4. What is the difference between seriousness and severity?**
Strong: seriousness is an outcome-based regulatory classification; severity is intensity. Weak: uses
them interchangeably.

**5. What would make you close a signal?**
Strong: heterogeneous cases, explained duplicates, a class comparison showing no excess, or an
epidemiological study showing no association. Weak: closes because the count is small.

**6. How do you handle duplicate reports?**
Strong: worldwide deduplication by patient, event, and date, with follow-ups linked to the parent
case, and a selective-duplication audit. Weak: removes exact duplicates only.

**7. What is notoriety bias and how do you detect it?**
Strong: increased reporting after publicity; detect with a time-series analysis and comparison with
pre-publicity trends. Weak: says it is "media attention" without a detection method.

**8. What are the obligations for an unexpected serious reaction?**
Strong: expedited reporting within the jurisdiction's timeline, with the clock starting at receipt of
the minimum criteria. Weak: says "report quickly".

**9. How does active surveillance complement spontaneous reporting?**
Strong: defined population and denominator, enabling incidence and rate ratios, at higher cost and
with different biases. Weak: says it is "better data".

**10. What should a signal assessment contain?**
Strong: evidence, comparator analysis, alternative explanations, strength and consistency, a
recommendation, and a falsification condition. Weak: a summary of the cases.

**11. What is the difference between a signal and a risk?**
Strong: a signal is information suggesting a causal association and requires validation and
assessment; a risk is an established association with a characterised frequency and severity,
usually supported by controlled or epidemiological data. A signal can be closed without becoming a
risk. Weak: treats the two as synonyms, which leads either to premature label changes or to
under-reaction.

## 15. Key terms

- **AEMS** — FDA Adverse Event Monitoring System, formerly FAERS.
- **Adverse event (AE)** — any untoward medical occurrence after exposure.
- **Adverse reaction (AR)** — an AE with a reasonable causal relationship.
- **Causality** — assessment of whether the product caused the event.
- **Confounding by indication** — the indication explains the association.
- **Disproportionality** — over-representation of an event among a drug's reports.
- **DSUR** — Development Safety Update Report.
- **E2B(R3)** — ICH standard for electronic case report transmission.
- **EudraVigilance** — the EU adverse reaction database.
- **Expectedness** — whether an event is listed in the reference safety information.
- **ICSR** — individual case safety report.
- **MedDRA** — Medical Dictionary for Regulatory Activities.
- **Minimum criteria** — reporter, patient, product, and event.
- **Notoriety bias** — increased reporting after publicity.
- **PBRER** — Periodic Benefit-Risk Evaluation Report.
- **Person-time** — time at risk across exposed patients.
- **PRR** — proportional reporting ratio.
- **PSUR** — Periodic Safety Update Report.
- **ROR** — reporting odds ratio.
- **SAE** — serious adverse event.
- **Signal** — information suggesting a new causal association.
- **Signal validation** — checking whether a signal is supported by documentation.
- **Stimulated reporting** — reporting driven by external attention.
- **Under-reporting** — the unknown fraction of events not reported.

## 16. Sources

- US Food and Drug Administration. *FDA Adverse Event Monitoring System (AEMS)*.
  https://www.fda.gov/drugs/surveillance-post-drug-approval-activities/fda-adverse-event-monitoring-system-aems
  (accessed 25 October 2026).
- US Food and Drug Administration. *FDA Launches New Adverse Event Look-Up Tool*. March 2026.
  https://www.fda.gov/news-events/press-announcements/fda-launches-new-adverse-event-look-tool
  (accessed 25 October 2026).
- US Food and Drug Administration. *AEMS Electronic Submissions*.
  https://www.fda.gov/drugs/fda-adverse-event-monitoring-system-aems/fda-adverse-event-monitoring-system-aems-electronic-submissions
  (accessed 25 October 2026).
- European Medicines Agency. *Guideline on Good Pharmacovigilance Practices (GVP), Module IX: Signal
  Management*, Rev 1, 2017.
  https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-good-pharmacovigilance-practices-gvp-module-ix-signal-management-rev-1_en.pdf
  (accessed 25 October 2026).
- European Medicines Agency. *Screening for Adverse Reactions in EudraVigilance*, EMA/849944/2016.
  https://www.ema.europa.eu/en/documents/other/screening-adverse-reactions-eudravigilance_en.pdf
  (accessed 25 October 2026).
- European Medicines Agency. *Guideline on the Use of Statistical Signal Detection Methods in the
  EudraVigilance Data Analysis System*, EMEA/106464/2006 rev. 1.
  https://www.ema.europa.eu/en/documents/regulatory-procedural-guideline/guideline-use-statistical-signal-detection-methods-eudravigilance-data-analysis-system_en.pdf
  (accessed 25 October 2026).
- ICH. *E2A Clinical Safety Data Management: Definitions and Standards for Expedited Reporting*.
  https://www.ich.org/page/efficacy-guidelines (accessed 25 October 2026).
- ICH. *E2B(R3) Electronic Transmission of Individual Case Safety Reports*.
  https://www.ich.org/page/efficacy-guidelines (accessed 25 October 2026).
- ICH. *E2C(R2) Periodic Benefit-Risk Evaluation Report*. https://www.ich.org/page/efficacy-guidelines
  (accessed 25 October 2026).
- US Food and Drug Administration. *21 CFR 314.80 Postmarketing Reporting of Adverse Drug
  Experiences*. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-314/subpart-B
  (accessed 25 October 2026).
- MedDRA. *MedDRA and the MSSO*. https://www.meddra.org/ (accessed 25 October 2026).
- Uppsala Monitoring Centre. *WHO-UMC Causality Assessment Scale*.
  https://who-umc.org/ (accessed 25 October 2026).

> **Note:** The trial, database counts, and drug in Section 6 are fictional. The rofecoxib withdrawal
> date and the regulatory documents cited are real; the interpretation is the course's own.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_cdisc-and-trial-standards.md`](05_cdisc-and-trial-standards.md)
- Next topic: [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md)
- Related: [`02_data-in-pharma.md`](02_data-in-pharma.md)
- Related: [`10_submission-data.md`](10_submission-data.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
