# Post-Market Surveillance

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 10 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 04, 05, 06, 09
> Version 2026.3

## 1. Why this topic exists

A medical device is not proven at clearance. It is proven in use. The premarket evidence describes a version, a
population, a setting, and a set of assumptions. The post-market period tests whether those assumptions hold when the
device meets real patients, real workflows, real data, and real adversaries. For software and artificial intelligence
(AI), the test is continuous because the world changes: coding practices shift, scanners are replaced, referral pathways
change, and the model itself may be updated.

Post-market surveillance (PMS) is the system that turns those changes into evidence and action. It collects complaints,
adverse events, performance data, service records, literature, and security intelligence. It analyzes whether the device
is performing as intended, whether new risks have appeared, whether the labeling is adequate, and whether a corrective
action is needed. It feeds the risk management file, the quality management system (QMS), the regulatory reporting
process, and the next product version.

The business problem is that PMS is often under-resourced because it does not look like product development. A team
ships a model, declares victory, and moves to the next feature. The complaints trickle in. The customer success team
handles them as support tickets. The regulatory team files reports without a feedback loop. The data science team never
sees the field data. The result is a device that is nominally monitored but practically unobserved. The first serious
signal may arrive as a recall, a lawsuit, or a newspaper story.

The regulatory problem is that PMS is mandatory, not optional. In the United States, manufacturers must report certain
deaths, serious injuries, and malfunctions under 21 CFR Part 803, and certain corrections and removals under 21 CFR Part
806. In the European Union, the Medical Device Regulation (MDR) requires a PMS system, a PMS plan, periodic safety
update reports for higher-risk devices, post-market clinical follow-up, and vigilance reporting within 2, 10, or 15 days
depending on the incident. A device that is marketed without these systems is non-compliant even if it performs well.

The clinical problem is that post-market signals are often subtle. A model's average accuracy may be stable while its
calibration drifts. An alert may still fire, but clinicians may stop responding to it. A subgroup may experience more
false negatives than the overall population. A data feed may change its units, and the model may interpret a value as a
different scale. These problems do not appear in a standard complaint. They require deliberate monitoring, denominators,
and subgroup analysis.

The commercial problem is that post-market evidence is increasingly a market requirement. Hospitals ask for performance
reports, security advisories, and real-world validation. Payers ask for outcomes. Procurement asks for references. A
manufacturer that can show a credible PMS system has an advantage over one that cannot. A manufacturer that cannot
produce a signal investigation may lose a contract before it loses a regulator.

This topic gives you the vocabulary and procedure to design a PMS system for software and AI devices, to distinguish
complaint handling from surveillance, to calculate and interpret post-market metrics, and to decide when a signal
requires a report, a corrective action, a labeling change, or a new submission. It closes the module by connecting the
premarket evidence to the post-market evidence. The treatment is educational and does not replace the regulations,
guidance, or professional review.

## 2. Learning outcomes

By the end of this topic, you can:

1. Distinguish complaint handling, adverse-event reporting, post-market surveillance, and post-market clinical follow-up.
2. Design a PMS plan with data sources, metrics, thresholds, responsibilities, and review cadence.
3. Apply US reporting requirements under 21 CFR Parts 803 and 806 to a software-related event.
4. Apply EU MDR vigilance timelines and PSUR requirements to a device incident.
5. Build a performance-monitoring dashboard for a deployed model, including drift, calibration, subgroup, and alert-burden metrics.
6. Investigate a signal, distinguish a false alarm from a true safety issue, and recommend an action.
7. Integrate PMS outputs into the risk management file, CAPA, labeling, and change control.
8. Critique a PMS plan that collects data but has no decision rules.

The outcomes map to the exercises and self-check items.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1 | Question 1 |
| 2 | Exercise 2 | Question 2 |
| 3 | Exercise 3 | Question 3 |
| 4 | Exercise 4 | Question 4 |
| 5 | Exercise 5 | Question 5 |
| 6 | Exercise 6 | Question 6 |
| 7 | Exercise 7 | Question 7 |
| 8 | Exercise 8 | Questions 8, 9, 10 |

## 3. The core mental model

PMS is a feedback loop, not a report. The loop starts with a question about the device in use, collects data that can
answer it, analyzes the data against a pre-specified threshold, decides on an action, and feeds the action back into the
device, the labeling, the risk file, and the next version.

```text
Device in use
    |
    v
[1] Data sources: complaints, adverse events, service, performance, security, literature
    |
    v
[2] Signal detection: metric exceeds threshold or pattern emerges
    |
    v
[3] Signal assessment: is it real, device-related, and clinically significant?
    |
    +-------------------+-------------------+-------------------+
    | no action         | labeling/QMS      | field action       |
    v                   v                   v
[4] Document why      [5] Correct/update   [6] Report/recall
    |                   |                   |
    +-------------------+-------------------+
                        |
                        v
[7] Risk file, CAPA, PMS plan, next version
```

The first distinction is between a complaint and a signal. A complaint is an individual communication. A signal is a
pattern or a single event that suggests a new risk or a change in performance. A complaint can be a signal, but most
complaints are not. A signal can exist without a complaint, such as a gradual calibration drift or a security
vulnerability. A PMS system that only reads complaints will miss the quiet failures.

The second distinction is between reporting and surveillance. Reporting is the mandatory notification of a specific
event within a specific time. Surveillance is the broader system of data collection and analysis. A manufacturer can
file every required report and still have a weak surveillance system. A strong surveillance system may generate fewer
reports because it detects and fixes issues earlier.

The third distinction is between pre-market and post-market evidence. Pre-market evidence answers whether the device can
work under controlled conditions. Post-market evidence answers whether it does work in the intended population and
workflow. The same metric can mean different things in the two contexts. A sensitivity of 0.90 in a curated test set may
fall to 0.78 in a real hospital with a different case mix. A calibration slope of 1.0 in a development set may drift to
0.85 after a change in coding practice.

The fourth distinction is between performance drift and concept drift. Performance drift is a change in measured
performance. Concept drift is a change in the relationship between inputs and outcomes. Data drift is a change in the
input distribution. A model can experience data drift without performance drift, or performance drift without a clear
data drift. The monitoring plan should measure all three where possible, not only the output metric.

The fifth distinction is between a signal and a cause. A signal says that something changed. A cause says why. A good
investigation separates the two. A drop in sensitivity may be caused by a change in the case mix, a change in the
reference standard, a data pipeline error, a model update, or a real change in disease presentation. The response
depends on the cause. A labeling change is not the right response to a data pipeline bug.

The sixth distinction is between a correction and a removal. A correction fixes a device or its labeling. A removal
takes a device off the market. Both can be voluntary or mandatory. In the US, certain corrections and removals must be
reported under 21 CFR Part 806. In the EU, a field safety corrective action triggers a field safety notice and vigilance
reporting. The classification of the action determines the reporting route, not the manufacturer's preference.

A useful shorthand is: collect, detect, assess, act, report, feed back. If any step is missing, the loop is open. A
system that collects but does not detect is a data lake. A system that detects but does not assess is an alert factory.
A system that assesses but does not act is a report. A system that acts but does not feed back will repeat the problem.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Source |
|---|---|---|---|
| Post-market surveillance (PMS) | Systematic collection and analysis of data after launch to verify safety and performance | Treating it as complaint handling only | MDR Article 83; FDA QMSR |
| PMS plan | Document defining data sources, methods, metrics, and responsibilities | A plan with no thresholds or owners | MDR Article 84 |
| PMSR | Periodic safety update report for Class I devices under the MDR | Confusing it with a PSUR | MDR Article 85 |
| PSUR | Periodic safety update report for Class IIa, IIb, and III devices | Assuming one frequency for all classes | MDR Article 86; MDCG 2022-21 |
| PMCF | Post-market clinical follow-up | Treating it as optional for all devices | MDR Annex XIV |
| Complaint | Written or oral communication alleging a deficiency | Treating every complaint as a reportable event | 21 CFR 820.3; ISO 13485 |
| Adverse event | Harm or potential harm associated with a device | Assuming all adverse events are reportable | 21 CFR 803; MDR Article 87 |
| Serious incident | Incident that led or could have led to death or serious deterioration | Using the US definition unchanged in the EU | MDR Article 2(65) |
| MDR (reporting) | Medical Device Reporting under 21 CFR Part 803 | Confusing with the EU MDR regulation | 21 CFR Part 803 |
| Field safety corrective action (FSCA) | Action to reduce or eliminate a risk after launch | Treating a software patch as automatically non-reportable | MDR Article 2(68) |
| Field safety notice (FSN) | Communication to users about an FSCA | Sending a notice without regulatory review | MDR Article 89 |
| Correction | Repair, modification, or labeling change to address a deficiency | Assuming a correction is always reportable | 21 CFR Part 806 |
| Removal | Taking a device off the market | Confusing with a market withdrawal | 21 CFR Part 806 |
| Recall | Removal or correction of a marketed device that violates the law | Assuming all recalls are mandatory | 21 CFR Part 7 |
| Signal | Information suggesting a new risk or changed performance | Treating a single complaint as a confirmed signal | FDA/IMDRF pharmacovigilance concepts |
| Data drift | Change in input distribution | Calling it model drift | FDA AI lifecycle draft, 2025 |
| Concept drift | Change in the input-outcome relationship | Confusing with data drift | FDA AI lifecycle draft, 2025 |
| Performance drift | Change in measured model performance | Assuming it is always caused by the model | FDA AI lifecycle draft, 2025 |
| Calibration | Agreement between predicted probabilities and observed outcomes | Monitoring only AUC | ISO 14971 / clinical prediction literature |
| Alert burden | Number of alerts per unit of clinical activity | Ignoring it because accuracy is stable | Human factors literature |
| Denominator | The population or exposure used to calculate a rate | Reporting counts without denominators | Epidemiology |
| CAPA | Corrective and preventive action | Closing a CAPA without effectiveness check | ISO 13485 |
| Root cause | The underlying reason a problem occurred | Fixing the symptom | ISO 13485 |
| Trend report | Analysis of increases in non-serious incidents | Reporting individual events only | MDR Article 88 |
| UDI | Unique Device Identification | Treating it as a SKU | MDR Article 27; 21 CFR Part 830 |
| EUDAMED | European database on medical devices | Assuming it is fully populated | MDR Article 33 |
| AEMS | FDA Adverse Event Monitoring System, launched 11 March 2026 | Assuming it is only a rename of FAERS | FDA AEMS page |
| MAUDE | Manufacturer and User Facility Device Experience database | Assuming it remains the only device adverse-event source | FDA database |
| Real-world performance | Performance observed in routine care | Assuming it equals clinical trial performance | FDA AI monitoring draft, 2025 |

Two definitions need more detail. The first is "serious incident" under the MDR. It is not every adverse event. It is an
incident that led, could have led, or might lead to death or a serious deterioration in health. That includes a device
malfunction that could cause harm if it recurred, even if no patient was harmed in the specific event. The manufacturer
must assess the potential, not only the actual outcome.

The second is "signal." A signal is not proof. It is a hypothesis that something may be wrong. The investigation tests
the hypothesis against the data, the device history, the clinical context, and the alternative explanations. A signal
that is dismissed without an investigation is a missed opportunity. A signal that is escalated without an investigation
is an alarm that erodes trust.

A third term, "post-market clinical follow-up," is often confused with PMS. PMCF is a specific activity: the collection
of clinical data after launch to confirm safety and performance, detect emerging risks, and identify misuse. PMS is the
broader system that includes PMCF, complaints, literature, and performance data. A PMS plan without PMCF may be
acceptable for some low-risk devices, but not for all.

## 5. The mechanics, step by step

A PMS system is designed before launch and operated after launch. The design should be proportionate to the risk, the
device class, the novelty, and the data available. The steps below produce a PMS plan, a monitoring dashboard, a
signal-investigation procedure, and a reporting decision.

### Step 1 — Define the surveillance scope

**Input.** Intended use, device description, risk analysis, market geography, and launch plan.

**Output.** A scope statement naming the device, versions, populations, settings, and jurisdictions.

**Rationale.** A PMS plan that covers everything covers nothing. The scope should name the device, the versions in the
field, the intended population, the care settings, and the countries where it is marketed. It should also name the
versions that are no longer supported and the plan for them. A model update can create multiple deployed versions; the
surveillance system must know which version produced which output.

**Procedure.** (1) Name the device and software versions. (2) Name the intended population and settings. (3) Name the
markets and regulatory regimes. (4) Name the post-market period. (5) Name the versions in scope and the end-of-support
plan.

**Branch points.** A single-market device has one reporting regime. A multi-market device needs a matrix of
requirements. A device with a PCCP needs version-level surveillance. A legacy device needs a retrospective risk
assessment.

**Edge cases.** A device sold through distributors may have limited visibility into the end user. A cloud device may
have a single deployed version across all customers. A device with a local adaptation may have site-specific
performance. A device that is no longer sold may still have patients using it.

### Step 2 — Identify data sources

**Input.** The scope, the device architecture, and the customer relationships.

**Output.** A data-source inventory with owner, access method, frequency, and limitations.

**Rationale.** PMS needs data that can answer the surveillance questions. Complaints are one source, but not the only
one. Performance data, service records, security advisories, literature, registries, and customer feedback may be more
informative. The inventory should state what each source can and cannot show. A complaint database cannot tell you the
denominator. A performance dashboard cannot tell you whether the clinician acted on the output.

**Procedure.** (1) List internal sources: complaints, CAPA, service, returns, audits, security. (2) List external
sources: registries, literature, regulators, customers, distributors. (3) List product telemetry: usage, alerts,
overrides, uptime. (4) Define access and data-use rights. (5) Define frequency and owner. (6) Record limitations.

**Branch points.** A device with telemetry can measure performance continuously. A device without telemetry must rely on
site data and periodic studies. A device with a small customer base may need proactive outreach. A device with a
vulnerable population may need additional sources.

**Edge cases.** Customer data may be governed by contracts that limit use. Registry data may lag. Literature may be
biased toward positive studies. Social media may provide early signals but unreliable denominators. A data source that
is not accessible in practice is not a source.

### Step 3 — Define performance and safety metrics

**Input.** The intended use, the risk analysis, the premarket evidence, and the clinical claim.

**Output.** A metric set with definitions, numerators, denominators, stratifications, and data sources.

**Rationale.** A metric without a definition is not measurable. "Model accuracy" is not a metric until you define the
outcome, the reference standard, the population, the time window, and the handling of missing data. The metric set
should cover performance, safety, use, security, and operations. It should include subgroup stratifications where the
premarket evidence identified a risk.

**Procedure.** (1) Define the clinical outcome and reference standard. (2) Define sensitivity, specificity, predictive
values, and calibration where relevant. (3) Define alert burden and override rate. (4) Define data quality and
missingness. (5) Define uptime, latency, and security incidents. (6) Define subgroups. (7) Define the denominator.

**Branch points.** A diagnostic model needs sensitivity, specificity, and predictive values. A prognostic model needs
calibration and discrimination. A triage model needs time-to-action and workflow outcomes. A monitoring model needs
alert burden and false-alarm rate. A cybersecurity metric needs incident severity and time to patch.

**Edge cases.** A metric that depends on a clinician's action may be confounded by workflow. A metric that uses a proxy
outcome may not reflect clinical benefit. A metric that is measured only at one site may not generalize. A metric that
is not stratified may hide a subgroup harm.

### Step 4 — Set thresholds and decision rules

**Input.** The metrics, the premarket performance, the clinical risk, and the operational context.

**Output.** A threshold for each metric and a decision rule for each threshold.

**Rationale.** A dashboard without thresholds is a wall of numbers. A threshold converts a metric into a decision:
green, amber, or red; investigate, report, or act. The threshold should be justified by the premarket evidence, the
clinical risk, and the variability of the metric. It should be set before the data are seen, not after.

**Procedure.** (1) Set a baseline from premarket or early post-market data. (2) Set a green range. (3) Set an amber
range that triggers investigation. (4) Set a red range that triggers action. (5) Define the statistical method for
detecting a change. (6) Define the review cadence. (7) Record the rationale.

**Branch points.** A safety metric may use a zero-tolerance threshold for a severe event. A performance metric may use a
statistical process control rule. A workflow metric may use a relative change threshold. A security metric may use a
severity-based threshold.

**Edge cases.** A threshold that is too tight will generate noise. A threshold that is too loose will miss a real
signal. A threshold that is based on a small sample will fluctuate. A threshold that ignores seasonality will
false-alarm every year. A threshold that is never revisited will become obsolete.

### Step 5 — Establish data governance and quality

**Input.** The data sources, the metric definitions, and the privacy requirements.

**Output.** A data governance plan covering access, quality, privacy, retention, and audit.

**Rationale.** Post-market data are messy. They come from different systems, with different definitions, different
completeness, and different incentives. A PMS system that does not govern the data will produce unreliable conclusions.
Governance includes who can access the data, how it is de-identified, how quality is checked, how long it is retained,
and how the analysis is reproduced.

**Procedure.** (1) Define data ownership and access. (2) Define de-identification and privacy controls. (3) Define
quality checks and missing-data rules. (4) Define versioning and audit trails. (5) Define retention. (6) Define the
analysis approval.

**Branch points.** A multi-site study needs a common data model. A cloud device may collect data centrally. A device
with limited connectivity may need manual abstraction. A global device needs jurisdiction-specific privacy controls.

**Edge cases.** A data source that changes its coding practice can create a false signal. A data source that is
incomplete can hide a safety event. A data source that is not representative can bias the conclusion. A data source that
is not auditable cannot support a regulatory decision.

### Step 6 — Detect signals

**Input.** The metrics, thresholds, and data.

**Output.** A signal log with date, metric, deviation, affected population, and initial evidence.

**Rationale.** Signal detection is the transition from monitoring to action. It can be rule-based, statistical, or
expert-review-based. The method should be pre-specified so that the team does not cherry-pick. A signal can come from a
single serious event, a cluster of complaints, a statistical trend, a literature report, or a security advisory.

**Procedure.** (1) Run the pre-specified detection rules. (2) Review the output. (3) Log the signal. (4) Assign an
owner. (5) Set an investigation deadline. (6) Notify the safety committee if the signal is severe.

**Branch points.** A single death or serious injury is a signal that requires immediate assessment. A statistical trend
requires investigation but may not be a reportable event. A complaint cluster requires a pattern analysis. A security
advisory requires a vulnerability assessment.

**Edge cases.** A signal that appears only in a subgroup may be missed by an aggregate metric. A signal that appears
only at one site may be a local data issue. A signal that appears after a model update may be a change effect. A signal
that appears in a new market may be a population difference.

### Step 7 — Triage for reportability

**Input.** The signal, the event details, the device history, and the regulatory criteria.

**Output.** A reportability decision for each jurisdiction, with rationale.

**Rationale.** Reportability is a legal decision with strict timelines. In the US, a manufacturer must report a death,
serious injury, or malfunction that meets the criteria in 21 CFR 803.50 within 30 calendar days, and certain events
within 5 days. In the EU, a serious incident must be reported within 15 days, a death or unanticipated serious
deterioration within 10 days, and a serious public health threat within 2 days. The clock starts when the manufacturer
becomes aware, not when the investigation is complete.

**Procedure.** (1) Determine whether the event meets the definition of a reportable event. (2) Determine whether the
device caused or contributed. (3) Determine the jurisdiction. (4) Determine the timeline. (5) Prepare the initial
report. (6) Plan follow-up.

**Branch points.** A death or serious injury is usually reportable. A malfunction is reportable if it could cause or
contribute to death or serious injury. A use error may be reportable if the device contributed. A security incident may
be reportable if it affects safety.

**Edge cases.** A near miss may be reportable in the EU if it could have led to serious harm. A complaint from a patient
may be reportable even if the clinician disagrees. A software bug that is fixed before any harm may still be reportable
depending on the potential. A foreign event may be reportable in the US if it involves a device marketed in the US.

### Step 8 — Investigate the signal

**Input.** The signal log, device records, data, and clinical input.

**Output.** An investigation report with root cause, contributing factors, and conclusion.

**Rationale.** An investigation separates a real signal from a false alarm and identifies the cause. It should consider
the device, the user, the environment, the data, and the process. It should test alternative explanations. It should
document the evidence, not only the conclusion.

**Procedure.** (1) Review the event and the device history. (2) Review the data and the denominator. (3) Interview users
and service staff. (4) Inspect the device and logs. (5) Test the hypothesis. (6) Consider alternatives. (7) Conclude and
document.

**Branch points.** A device defect leads to a correction. A use error leads to labeling or training. A data issue leads
to a pipeline fix. A population difference leads to a labeling or performance update. An unknown cause leads to further
monitoring.

**Edge cases.** A root cause that is convenient but unproven is not a root cause. A conclusion that blames the user
without examining the interface is incomplete. A data analysis that ignores the denominator is misleading. An
investigation that takes too long can miss a reporting deadline.

### Step 9 — Reassess benefit-risk

**Input.** The investigation, the risk file, and the clinical context.

**Output.** An updated benefit-risk assessment and a recommendation.

**Rationale.** A signal can change the benefit-risk balance even if the device still performs as labeled. A new risk, a
higher frequency, a more severe harm, or a loss of benefit can shift the balance. The assessment should consider the
alternatives available to patients and the consequences of removing the device.

**Procedure.** (1) Update the hazard and risk estimates. (2) Assess the new benefit-risk balance. (3) Consider the
alternatives. (4) Identify the affected populations. (5) Recommend an action. (6) Document the rationale.

**Branch points.** A favorable balance may support continued marketing with monitoring. An unfavorable balance may
require a field action. An uncertain balance may require a restriction or a further study.

**Edge cases.** A small benefit for a large population may outweigh a rare serious risk. A large benefit for a small
population may justify a higher risk. A risk that is concentrated in a vulnerable subgroup may be unacceptable even if
the average is acceptable. A benefit that is not clinically meaningful may not justify any new risk.

### Step 10 — Choose and implement the action

**Input.** The benefit-risk assessment and the root cause.

**Output.** An action plan with owner, timeline, verification, and communication.

**Rationale.** The action should address the cause, not only the symptom. It can be a design change, a labeling change,
a training update, a software patch, a field correction, or a removal. The action should be proportionate to the risk
and should be verified for effectiveness.

**Procedure.** (1) Choose the action. (2) Define the scope. (3) Assign the owner. (4) Define the timeline. (5)
Implement. (6) Verify. (7) Communicate. (8) Monitor.

**Branch points.** A design change requires verification and possibly a new submission. A labeling change requires
regulatory review. A software patch requires change control and cybersecurity assessment. A field correction requires
reporting and customer notification.

**Edge cases.** A training-only action may not be sufficient for a use error caused by a confusing interface. A software
patch may not reach all customers. A labeling change may not be read. A field correction may create a new risk.

### Step 11 — Update the risk file, CAPA, and labeling

**Input.** The investigation and action.

**Output.** Updated risk management file, CAPA record, and labeling.

**Rationale.** PMS is not complete until the organization learns. The risk file must reflect the new hazard, the new
probability, and the new control. The CAPA must address the root cause and verify effectiveness. The labeling must
reflect the new limitations or instructions.

**Procedure.** (1) Update the hazard analysis. (2) Update the risk estimate. (3) Update the control. (4) Open or update
the CAPA. (5) Update the labeling. (6) Train the relevant staff. (7) Verify effectiveness.

**Branch points.** A new hazard requires a new risk analysis. A changed probability requires a re-evaluation. A new
control requires verification. A labeling change requires regulatory review.

**Edge cases.** A CAPA that closes without an effectiveness check is incomplete. A risk file that is updated after the
audit is not a control. A labeling change that is not distributed is not effective. A training record that is not linked
to the change is not evidence.

### Step 12 — Report to regulators and communicate with users

**Input.** The reportability decision, the action, and the audience.

**Output.** Regulatory reports, field safety notices, customer letters, and public disclosures.

**Rationale.** Reporting is a legal obligation with timelines and formats. Communication is a safety obligation. The two
are related but not identical. A report to a regulator may not reach the user. A user notice may not satisfy the
regulator. The manufacturer must do both.

**Procedure.** (1) Prepare the regulatory report. (2) Submit within the deadline. (3) Prepare the user notice. (4)
Coordinate with the regulator if required. (5) Distribute. (6) Record acknowledgments. (7) Follow up.

**Branch points.** A US MDR report goes to FDA. An EU serious incident report goes to the competent authority. A field
safety corrective action requires a field safety notice. A recall requires a recall strategy and status reports.

**Edge cases.** A report that is late is a violation even if the investigation is ongoing. A user notice that is unclear
can cause confusion. A public disclosure that is premature can harm trust. A communication that is not translated can
fail in a multilingual market.

### Step 13 — Conduct periodic review

**Input.** The signal log, the metrics, the CAPA, and the literature.

**Output.** A periodic report, a management review, and an updated PMS plan.

**Rationale.** Periodic review looks at the pattern across signals, not only individual events. It asks whether the PMS
system is working, whether the metrics are still relevant, and whether the risk profile has changed. It is also a
regulatory requirement: the EU requires a PSUR for Class IIa, IIb, and III devices, with different frequencies, and a
PMSR for Class I devices.

**Procedure.** (1) Aggregate the data. (2) Analyze trends. (3) Review the signal investigations. (4) Review the CAPA
effectiveness. (5) Review the literature. (6) Update the benefit-risk assessment. (7) Update the PMS plan. (8) Report to
management.

**Branch points.** A Class III or implantable device needs a more frequent and detailed review. A Class I device may use
a PMSR. A device with a PCCP needs a review of the authorized changes. A device with a security risk needs a
vulnerability review.

**Edge cases.** A periodic report that only lists events is not an analysis. A review that ignores subgroups is
incomplete. A plan that is not updated after a signal is stale. A management review that has no decisions is a meeting,
not a review.

### Step 14 — Feed back into the lifecycle

**Input.** The PMS outputs and the lifecycle records.

**Output.** Updated requirements, design, verification, training, and supplier controls.

**Rationale.** The point of PMS is to improve the device and the process. The feedback should reach the requirements,
the design, the risk file, the verification plan, the labeling, and the supplier controls. A PMS system that does not
change anything is a reporting obligation, not a learning system.

**Procedure.** (1) Translate the finding into a requirement or control. (2) Update the design or process. (3) Verify the
change. (4) Update the training. (5) Update the supplier controls. (6) Monitor the effect.

**Branch points.** A design change goes through change control. A process change goes through the QMS. A supplier change
goes through procurement and quality. A labeling change goes through regulatory.

**Edge cases.** A finding that is not translated into a requirement will recur. A change that is not verified may
introduce a new risk. A supplier change that is not assessed may affect safety. A training change that is not measured
may not change behavior.

### Decision table: reportability

| Event | US | EU | Action |
|---|---|---|---|
| Death or serious injury caused by device | 30-day MDR; 5-day if remedial action | 10 days | Report, investigate, risk update |
| Malfunction that could cause serious harm | 30-day MDR | 15 days if serious incident | Report if criteria met |
| Near miss with potential serious harm | Case-by-case | 15 days if serious incident | Assess and document |
| Use error with device contribution | 30-day MDR if criteria met | 15 days if serious incident | Investigate interface and labeling |
| Security vulnerability without harm | Report if required by law | Assess under vigilance | Patch and communicate |
| Complaint without harm | No report unless malfunction criteria | Trend report | Log and monitor |

### Decision table: PMS frequency by EU class

| Class | PMS report | PSUR frequency | PMCF |
|---|---|---|---|
| I | PMSR | Not required | Based on risk |
| IIa | PSUR | Every 2 years | Based on risk |
| IIb | PSUR | Annually | Usually required |
| III | PSUR | Annually | Required |

### Decision table: signal response

| Signal strength | Evidence | Response |
|---|---|---|
| Weak | Single complaint, no pattern | Monitor and log |
| Moderate | Cluster or statistical trend | Investigate |
| Strong | Reproducible device defect | Correct and report |
| Severe | Death or serious harm | Immediate action and report |

The procedure is a system, not a sequence of isolated tasks. The next section applies it to a deployed model.

## 6. Worked example

The product, sponsor, sites, and numbers in this case are hypothetical. They show how a post-market signal moves from a
dashboard to a regulatory decision. They are not a real recall, a real PSUR, or a real FDA action.

### 6.1 Product and surveillance scope

RetinaTrace is the clinician-facing diabetic retinopathy triage software introduced in Topic 01. It analyzes color
fundus photographs and returns "no referable diabetic retinopathy detected," "referable diabetic retinopathy suspected,"
or "image ungradable." It is a Class II SaMD in the US with a 510(k), and a Class IIb device in the EU under the MDR. It
is deployed at 14 primary-care sites and three reading centers. The current version is 3.4. A planned model update is
version 4.0.

The PMS plan covers three years. It includes complaints, service desk tickets, site-reported performance, a quarterly
random sample of 400 cases per reading center with adjudicated reference labels, alert burden, override rate, ungradable
rate, uptime, and security incidents. The plan stratifies by site, camera model, age, sex, race, ethnicity, and skin
tone when available. The thresholds are:

| Metric | Green | Amber | Red |
|---|---|---|---|
| Sensitivity, overall | ≥ 0.88 | 0.82–0.88 | < 0.82 |
| Sensitivity, darkly pigmented skin | ≥ 0.85 | 0.78–0.85 | < 0.78 |
| Specificity, overall | ≥ 0.85 | 0.80–0.85 | < 0.80 |
| Calibration slope | 0.90–1.10 | 0.85–0.90 or 1.10–1.15 | < 0.85 or > 1.15 |
| Alert burden | ≤ 18 per 100 patients | 18–25 | > 25 |
| Override rate | ≤ 25% | 25–35% | > 35% |
| Ungradable rate | ≤ 8% | 8–12% | > 12% |

The thresholds were set before launch from the premarket study and the pilot data. The plan names a safety committee
that meets monthly, a medical monitor, a data engineer, and a regulatory lead. The committee reviews the dashboard and
the signal log.

### 6.2 The first six months

The first two quarters are green on the aggregate metrics. Sensitivity is 0.90, specificity is 0.88, calibration slope
is 0.98, and alert burden is 14 per 100 patients. The ungradable rate is 6 percent. The override rate is 19 percent.

In month seven, a new site joins with a different camera model. The site's images have a different color profile and a
higher proportion of undilated pupils. The site's ungradable rate is 19 percent, above the red threshold. The aggregate
ungradable rate rises to 8.5 percent, which is amber. The site reports that clinicians are ignoring the "ungradable"
output and referring patients without a readable image.

The signal log records the site, the camera, the rate, and the date. The committee classifies the signal as moderate: a
site-specific operational issue with a potential safety consequence, because a patient with an unreadable image may not
receive a timely exam. The committee opens an investigation.

### 6.3 The investigation

The investigation finds three contributing factors. First, the new camera's default white balance differs from the
validated cameras. Second, the site's photographers are less experienced, and the image capture protocol is not followed
for 22 percent of images. Third, the site's workflow does not route ungradable images to a repeat-capture queue; the
clinician sees the result and refers the patient to a specialist, which increases waiting time.

The root cause is not the model. It is a combination of an input domain shift and a workflow gap. The model's ungradable
classifier is working as designed: it flags images that are outside the validated domain. The site's workflow does not
act on the flag.

The corrective action has two parts. First, the sponsor adds the camera model to the validation set and updates the
labeling to list validated camera models. Second, the site adds a repeat-capture queue and a training module for
photographers. The sponsor also adds a site-level ungradable metric to the dashboard and a rule that a site above 12
percent for two consecutive months triggers a site review.

The action is a labeling and workflow correction, not a model change. It is reported through the quality system. The EU
assessment concludes that the action is not a field safety corrective action because the device performs as labeled and
the issue is a site workflow gap; the site training and labeling update are documented in the technical file. The US
assessment concludes that no MDR report is required because there was no death, serious injury, or malfunction, but the
complaint and correction are logged.

### 6.4 The second signal

In month ten, the quarterly audit finds that sensitivity in patients with darkly pigmented skin is 0.79. That is above
the red threshold of 0.78 but below the green threshold of 0.85, so the metric is amber. The overall sensitivity remains
0.89. The subgroup sample is 92 cases, which is small; the 95 percent confidence interval is wide.

The committee investigates. The data engineer finds that the site with the new camera contributes 38 of the 92 cases.
The camera's color profile is not the only difference: the site also serves a population with a higher prevalence of
advanced disease, and the reference standard at that site is a single grader rather than a consensus panel. The apparent
sensitivity drop may be a case-mix and reference-standard effect, not a model effect.

The committee does not dismiss the signal. It increases the subgroup sample to 200 cases, adds a second grader for the
site, and stratifies by camera and disease severity. The next quarter's estimate is 0.84 with a 95 percent confidence
interval from 0.78 to 0.89. The point estimate is still below the overall sensitivity but within the green range for the
subgroup. The committee keeps the amber status and continues monitoring.

### 6.5 Two diagnostics that would sink the surveillance system

**Diagnostic 1: the denominator changes without notice.** The quarterly sensitivity calculation uses the number of
adjudicated cases as the denominator. In month nine, the reading center changes its case-selection process and starts
including more mild cases. The prevalence of referable disease in the sample falls from 22 percent to 14 percent.
Sensitivity appears to fall from 0.90 to 0.84, but the model has not changed. The dashboard flags a red signal. If the
committee investigates the model without checking the denominator and the case mix, it may order an unnecessary model
change. The surveillance system must track the sampling frame and the prevalence alongside the metric.

**Diagnostic 2: duplicate patient records inflate the event count.** The complaint system counts each report as a
separate event. A single patient with a delayed diagnosis is reported by the primary-care site, the reading center, and
the patient's advocate. The signal log shows three events in one week and the committee treats it as a cluster. The
investigation finds one patient and three reports. The event count is wrong, and the cluster is an artifact. The
surveillance system must deduplicate by patient, device, and event, and must distinguish reports from events.

Either diagnostic can lead to a wrong action: an unnecessary model change in the first case, and a missed or
mischaracterized signal in the second. The system needs data quality checks, not only statistical thresholds.

### 6.6 The EU PSUR decision

RetinaTrace is Class IIb in the EU. Under MDR Article 86, the manufacturer must prepare a PSUR annually for a Class IIb
device. The PSUR must include the benefit-risk determination, the main PMCF findings, sales volume and population
estimates, vigilance data, and CAPA information. The first PSUR is due within one year of CE marking, and the notified
body assesses it.

The sponsor's PSUR includes the site-specific ungradable signal, the subgroup sensitivity signal, the corrective
actions, the updated labeling, and the PMCF plan. The benefit-risk conclusion is that the device remains favorable, with
a limitation on camera models and a requirement for site-level monitoring. The notified body asks for the subgroup
confidence intervals, the camera validation, and the site training records. The sponsor provides them.

The US and EU decisions differ in form but not in substance. The US files a 510(k) labeling update if the camera list
change affects the indications or the performance claims. The EU updates the technical documentation and the PSUR. The
sponsor does not assume that a US labeling change satisfies the EU requirement, or vice versa.

### 6.7 Vary one assumption

The base case is a Class IIb EU device with an annual PSUR. Change the assumption: the device is Class IIa in the EU.
The PSUR frequency becomes every two years, and the PMS plan may rely more on a PMSR-like summary, although the MDR
still requires a PSUR for Class IIa. The PMCF requirement may be lighter if the risk analysis supports it. The signal
investigation and the corrective action do not change. The reporting timeline for a serious incident remains 2, 10, or
15 days; the class affects the periodic report, not the vigilance clock.

Alternatively, change the US assumption: the device is Class III and has a PMA. The PMS requirements become a condition
of approval, and the sponsor may need a post-approval study. The reporting obligations under Part 803 remain. The
surveillance system must integrate the PMA conditions, the annual report, and the post-approval study data. The class
changes the governance and the evidence burden, not the basic loop.

| Assumption changed | PMS consequence | Reporting consequence | Evidence consequence |
|---|---|---|---|
| Base: EU Class IIb | Annual PSUR | 2/10/15-day vigilance | PMCF and PSUR data |
| EU Class IIa | PSUR every 2 years | Same vigilance clock | Lighter PMCF may be justified |
| US Class III/PMA | PMA conditions and annual report | Part 803 plus conditions | Post-approval study may be required |
| Predicate-based 510(k) | PMS plan and complaints | Part 803 and Part 806 | Labeling update may need a new 510(k) |
| New model version | Version-level monitoring | Report if new risk | PCCP or new submission check |

The lesson is that PMS is not a single report. It is a system that adapts to the class, the market, the version, and the
signal. The same event can require a US report, an EU vigilance report, a PSUR update, a CAPA, and a labeling change.
The manufacturer must run the system, not wait for a regulator to run it for them.

### 6.8 What the case does not show

The case does not show the full complaint log, the CAPA records, the supplier audits, or the security incident process.
It also does not show the statistical process control method, the sample-size calculation for the quarterly audit, or
the adjudication manual. Those artifacts are part of the PMS system, and their absence would be a finding in an audit.

The case also does not show a confirmed device-related harm. That is deliberate. Most post-market work is the
investigation of signals that turn out to be data artifacts, workflow gaps, or expected performance in a new population.
A system that only responds to confirmed harm is a system that waits too long. A system that investigates every signal
with the same rigor, and documents why the benign ones are benign, is a system that learns.

The case does not show the cost of the surveillance system. A quarterly audit of 400 cases per reading center, with
adjudication, is expensive. The sponsor must decide whether the evidence is worth the cost, and whether a smaller sample
with a different design would answer the question. The answer depends on the risk, the prevalence, and the variability
of the metric. A PMS plan that is not resourced is a plan that will be abandoned when the first budget cut arrives. A
plan that is resourced but not authorized to act is equally fragile.

## 7. Code, query, or calculation

The script below implements the predictive values from Section 6, a three-level signal classifier, and a simplified US
reportability triage. It is a teaching aid, not a legal determination.

```python
"""Post-market metric and triage helpers for Module 14, Topic 10."""

from __future__ import annotations


def predictive_values(sensitivity: float, specificity: float, prevalence: float) -> dict[str, float]:
    tp = sensitivity * prevalence
    fn = (1.0 - sensitivity) * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    tn = specificity * (1.0 - prevalence)
    return {"ppv": tp / (tp + fp), "npv": tn / (tn + fn)}


def signal_level(value: float, green: float, amber: float, red: float, higher_is_worse: bool) -> str:
    """Return green, amber, or red for a metric with ordered thresholds."""
    if higher_is_worse:
        if value > amber:
            return "red"
        if value > green:
            return "amber"
        return "green"
    if value < amber:
        return "red"
    if value < green:
        return "amber"
    return "green"


def us_report(severity: str, malfunction: bool, remedial: bool) -> str:
    """Simplified reportability triage; not a legal determination."""
    severity = severity.strip().lower()
    if severity in {"death", "serious_injury"}:
        return "5-day report if remedial action needed, otherwise 30-day MDR"
    if severity == "malfunction" and malfunction:
        return "30-day MDR if the malfunction could cause serious harm"
    if remedial:
        return "assess Part 806 correction/removal reporting"
    return "no MDR report on these facts; document rationale"


if __name__ == "__main__":
    values = predictive_values(0.90, 0.88, 0.12)
    print(f"PPV={values['ppv']:.3f} NPV={values['npv']:.3f}")

    print("sensitivity 0.79:", signal_level(0.79, green=0.85, amber=0.78, red=0.78, higher_is_worse=False))
    print("alert burden 26:", signal_level(26, green=18, amber=25, red=25, higher_is_worse=True))
    print("ungradable 19:", signal_level(19, green=8, amber=12, red=12, higher_is_worse=True))

    print(us_report("death", malfunction=False, remedial=True))
    print(us_report("malfunction", malfunction=True, remedial=False))
```

Expected output:

```text
PPV=0.506 NPV=0.985
sensitivity 0.79: amber
alert burden 26: red
ungradable 19: red
5-day report if remedial action needed, otherwise 30-day MDR
30-day MDR if the malfunction could cause serious harm
```

The `signal_level` function uses two boundaries for three levels. For a lower-is-better metric such as sensitivity, a
value below the amber boundary is red, a value between amber and green is amber, and a value at or above green is green.
For a higher-is-worse metric such as alert burden, the logic is reversed. The function does not account for sample size,
seasonality, or multiplicity. A real dashboard should use confidence intervals or a control chart, not a single point
estimate.

The `us_report` function is deliberately conservative and incomplete. It does not cover every reportable event, every
exemption, or every timeline. It does not address EU vigilance at all. It is a prompt for a conversation with a
regulatory professional, not a substitute for one.

The script uses point estimates. A dashboard that alerts on a single quarter's sensitivity can fire on noise, especially
when the subgroup sample is small. A better design uses a confidence interval, a sequential test, or a control chart
with limits derived from the baseline. The choice depends on the metric, the sample size, and the cost of a false alarm.
A safety metric with severe consequences may use a simple threshold and accept false alarms; a performance metric with a
stable baseline may use a statistical process control rule.

The script does not deduplicate events. A real PMS system must define the unit of analysis: a report, an event, a
patient, or a device. The same patient can generate multiple reports; the same device can generate multiple events; the
same event can be reported by multiple sources. The unit of analysis determines the numerator, and the denominator must
match it.

The script does not adjust for case mix. A sensitivity estimate from a sample with a different prevalence or severity
distribution is not comparable to the baseline. A real analysis should use standardization, stratification, or a
regression adjustment. The choice should be pre-specified in the PMS plan.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Reporting decisions are fact-specific and time-sensitive. Confirm the current regulations, guidance, and reporting systems with qualified regulatory counsel.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| 21 CFR Part 803 | Binding US regulation | Medical Device Reporting for manufacturers, importers, and user facilities | Does not cover every complaint |
| 21 CFR Part 806 | Binding US regulation | Reports of corrections and removals | Does not cover routine servicing |
| 21 CFR Part 7 | Binding US regulation | Recall policy and procedures | Does not replace Part 806 |
| FD&C Act §519(g) | Binding US statute | Authority for correction and removal reports | Does not define every reportable action |
| MDR Article 83 | Binding EU regulation | PMS system obligations | Does not prescribe every metric |
| MDR Article 84 | Binding EU regulation | PMS plan | Does not set thresholds |
| MDR Article 85 | Binding EU regulation | PMSR for Class I | Does not apply to Class IIa+ |
| MDR Article 86 | Binding EU regulation | PSUR for Class IIa, IIb, III | Does not set the template |
| MDR Article 87 | Binding EU regulation | Vigilance reporting timelines | Does not cover every incident type |
| MDR Article 88 | Binding EU regulation | Trend reporting | Does not replace individual reports |
| MDCG 2022-21 | EU guidance | PSUR content and frequency | Is not binding |
| MDCG 2023-3 | EU guidance | Vigilance terms and concepts | Is not binding |
| FDA AI lifecycle draft, 7 January 2025 | Draft guidance | Performance monitoring and drift | Is not final |
| ISO 13485:2016 | Voluntary standard | Complaint handling and CAPA | Does not set reporting timelines |

The US reporting framework has three main tracks. Part 803 requires manufacturers to report deaths, serious injuries,
and certain malfunctions. The general manufacturer timeline is 30 calendar days, with a 5-day report when a remedial
action is needed to prevent an unreasonable risk of substantial harm. User facilities report deaths and serious injuries
within 10 work days, and importers report within 30 calendar days. Part 806 requires reports of corrections and removals
that reduce a risk to health or remedy a violation, generally within 10 working days. Part 7 governs recalls, including
recall classification and status reports.

The EU framework is broader. The MDR requires a PMS system for every device, a PMS plan, and a PMSR or PSUR depending on
class. Class IIa devices require a PSUR every two years; Class IIb and III devices require one annually. The PSUR must
include the benefit-risk determination, the main PMCF findings, sales volume, vigilance data, and CAPA information.
Serious incidents must be reported within 15 days, death or unanticipated serious deterioration within 10 days, and a
serious public health threat within 2 days. Trend reporting covers increases in non-serious incidents that could become
serious.

The FDA's adverse-event infrastructure changed in 2026. On 11 March 2026, FDA launched the Adverse Event Monitoring
System (AEMS), a unified platform that replaced several legacy systems, including the FDA Adverse Event Reporting
System. Device reports from MAUDE were scheduled to move into AEMS in May 2026. The practical implication is that
post-market analytics teams should verify the current data source and field definitions before building a pipeline. A
schema that was correct in 2025 may not be correct in 2026.

The FDA's draft lifecycle guidance for AI-enabled device software functions, issued 7 January 2025, describes a
post-market performance monitoring plan. It names performance change and degradation, data drift, changes in patient
demographics or disease prevalence, data-pipeline integrity, and user behavior as things to monitor. It notes that such
a plan may be a special control for a De Novo device or a condition of approval for a PMA. The draft is not final, but
it is the most direct statement of FDA's expectations for AI monitoring.

The standards landscape supports the regulations. ISO 13485:2016 requires complaint handling, CAPA, and feedback. ISO
14971:2019 requires production and post-production monitoring. IEC 62304:2006+A1:2015 requires a maintenance process.
IEC 62366-1:2015+A1:2020 requires post-production usability information. None of these standards sets a reporting
timeline, but they require the system that produces the evidence for the timeline.

### Reporting comparison

| Question | US | EU |
|---|---|---|
| Death or serious injury | 30-day manufacturer MDR; 5-day in some remedial cases | 10 days for death or unanticipated serious deterioration |
| Serious public health threat | Report as required; 5-day may apply | 2 days |
| Other serious incident | 30-day MDR if criteria met | 15 days |
| Malfunction | 30-day MDR if it could cause serious harm | 15 days if serious incident |
| Correction or removal | Part 806, generally 10 working days | Field safety corrective action and field safety notice |
| Periodic report | PMA annual report or as required | PMSR Class I; PSUR IIa every 2 years, IIb/III annually |
| Trend reporting | Not a separate general requirement | Article 88 trend report |

The comparison is a simplification. The US and EU definitions of reportable event, serious injury, and malfunction
differ. A single event can be reportable in one jurisdiction and not the other. The manufacturer should maintain a
jurisdiction matrix and a single event record that records each decision separately.

### PSUR and PMSR content

| Section | What it contains | Common weakness |
|---|---|---|
| Device and scope | Device, versions, markets, period | Version list incomplete |
| Sales and exposure | Units sold, patients exposed, population estimates | No denominator |
| Vigilance | Serious incidents, FSCAs, trend reports | No analysis of patterns |
| Complaints | Complaint rates and categories | Counts without rates |
| PMCF | Study status and findings | Study not started |
| CAPA | Corrective and preventive actions | No effectiveness check |
| Benefit-risk | Updated conclusion | Repeats the premarket conclusion |
| Actions | Labeling, design, training, field actions | No owner or timeline |

A PSUR that is a data dump is not a periodic safety update. The report should answer whether the benefit-risk balance
has changed, what the main findings are, and what the manufacturer did about them. The notified body will look for the
analysis, not only the tables.

### AI performance monitoring

A model monitoring plan should cover four layers. The input layer checks data quality, completeness, units, and
distribution. The output layer checks score distribution, calibration, and decision rates. The outcome layer checks
sensitivity, specificity, predictive values, and clinical outcomes against a reference standard. The process layer
checks alert burden, override rate, time to action, and user feedback. A plan that monitors only the output layer can
miss an input change that has not yet affected the output, or a workflow change that has affected the outcome.

The monitoring frequency should match the risk and the data volume. A high-volume, high-risk device may need daily
checks and monthly adjudication. A low-volume device may need quarterly or annual review. A model with a PCCP may need
version-level monitoring at each authorized change. A model with no post-market data access may need a prospective
study.

### Data sources and what they can answer

| Source | Can answer | Cannot answer |
|---|---|---|
| Complaints | What users report, failure themes | Denominator, silent failures |
| Adverse-event reports | Serious harms and malfunctions | Rates without exposure data |
| Product telemetry | Usage, alerts, overrides, uptime | Clinical outcome without linkage |
| Site audits | Adjudicated performance | Population-level incidence |
| Registries | Outcomes in a defined population | Device-specific causality |
| Literature | Emerging risks and comparable devices | Your device's performance |
| Service records | Hardware and installation issues | Clinical impact |
| Security advisories | Vulnerabilities and patches | Patient exposure |
| Customer meetings | Workflow and usability issues | Statistical representativeness |

The table is a reminder that no single source is sufficient. A complaint gives a narrative but no denominator. Telemetry
gives a denominator but often no outcome. A registry gives outcomes but may not identify the device version. The PMS
plan should combine sources and state the limitations of each.

A common mistake is to treat the absence of complaints as evidence of safety. Complaints are affected by reporting
culture, user expectations, and the effort required to report. A quiet site may be a safe site or a site that does not
report. The PMS plan should include proactive data collection, not only passive receipt.

A PMS system also needs a documented method for handling conflicting evidence. A complaint may suggest a device defect,
while the performance data show no change. A site may report a problem that the central data cannot reproduce. The
method should define how the team weighs the sources, what triggers an on-site investigation, and how the conclusion is
documented. A conflict that is resolved by choosing the convenient source is not an investigation.

A PMS system should also define what it will do with a signal that is real but not device-related. A workflow change, a
coding change, or a new clinical guideline can alter performance without any device defect. The response may be a
labeling clarification, a customer education campaign, or a product change. The important point is that the
investigation does not stop at "not our fault." It asks whether the device can be made safer or more useful in the
changed environment.

The PMS plan should be written so that a new team member can run it. That means named roles, concrete data queries,
definitions, thresholds, and escalation paths. A plan that depends on the founder's memory is not a system. A plan that
cannot survive a personnel change is a single point of failure.

## 9. Industry practice and case evidence

**Epic Sepsis Model.** Wong et al. published an external validation of a widely deployed proprietary sepsis prediction
model in JAMA Internal Medicine in 2021. The study reported lower discrimination in the external health system than the
vendor-reported values, and a high alert burden at the deployed threshold. The documented lesson is that a model can be
commercially available and still underperform in a new setting. The inference is that local validation and monitoring
are necessary even when the product has a regulatory clearance or a vendor performance claim.

**IDx-DR.** FDA's De Novo authorization DEN180001 for autonomous diabetic retinopathy screening included labeling that
defined the intended use, the patient population, and the follow-up pathway. The public record does not establish a
specific post-market performance rate. The documented lesson is that a narrow intended use makes post-market
surveillance tractable: the manufacturer knows which population to monitor and which outcome to measure. The inference
is that a broad claim makes a precise PMS plan harder.

**Apple Irregular Rhythm Notification Feature.** FDA's 510(k) clearance K231173, dated 21 July 2023, included a PCCP for
bounded algorithm changes. A PCCP creates a specific post-market obligation: the manufacturer must monitor the
performance of the authorized changes and ensure they remain within the predetermined bounds. The documented lesson is
that a PCCP is not only a premarket instrument; it is a surveillance commitment. The inference is that a manufacturer
without a version-level monitoring system cannot execute the PCCP.

**FDA AEMS.** The launch of AEMS in March 2026 is a public example of a regulator consolidating adverse-event data. The
documented change is the platform and the data source, not the underlying reporting obligation. The inference is that
analytics teams should treat the data source as a controlled dependency, with schema validation and versioned queries,
just like a model dependency.

**Industry practice: safety committees with authority.** Mature manufacturers have a safety committee that can stop a
release, order a field action, and escalate to executive management. The committee includes clinical, quality,
regulatory, engineering, and legal representatives. It meets on a fixed cadence and on an ad hoc basis for severe
signals. A committee without authority is a discussion group.

**Industry practice: closed-loop monitoring.** Companies that operate AI devices well connect the monitoring dashboard
to the model lifecycle. A drift signal creates a ticket. A ticket creates an investigation. An investigation updates the
risk file. A risk file update triggers a change. The loop is documented, not improvised. The same companies tend to have
better customer trust because they can explain what changed and why.

**Industry practice: customer transparency.** Hospitals increasingly ask for performance reports, security advisories,
and change notifications. A manufacturer that shares a clear, honest performance summary is easier to trust than one
that shares only marketing claims. The summary should include the population, the denominator, the confidence intervals,
the limitations, and the actions taken. A dashboard screenshot is not a transparency report.

**What the cases do not prove.** The Epic validation does not prove that every sepsis model fails externally. IDx-DR
does not prove that autonomous screening is safe in every population. Apple's PCCP does not prove that every model
update is manageable. AEMS does not prove that reporting is easier. Each case is a specific data point. The transferable
method is the loop: collect, detect, assess, act, report, and feed back.

**Industry practice: separate safety from performance.** A performance signal and a safety signal travel different
paths. A drop in accuracy may be a quality issue that requires a model update but not a regulatory report. A false
negative that leads to delayed care is a safety issue that may require a report. The same metric can generate both. The
team should classify each signal by its potential harm, not only by its statistical significance.

**Industry practice: the medical monitor's role.** A medical monitor reviews signals for clinical significance, advises
on reportability, and helps design the investigation. The role is not a rubber stamp. A medical monitor who sees only
the dashboard and not the complaints cannot assess the context. A medical monitor who sees only the complaints and not
the performance data cannot see the quiet failures.

**Industry practice: post-market data access.** A manufacturer that cannot access site data cannot monitor performance.
Contracts should define data access, use, de-identification, and reporting. A hospital may be willing to share aggregate
performance but not patient-level data. The PMS plan should state what data are available and what the manufacturer will
do if access is limited. A plan that depends on data the manufacturer does not have is not executable.

**Industry practice: regulatory intelligence.** Reporting systems, forms, and guidance change. The FDA's AEMS launch in
March 2026 is one example. The EU's EUDAMED rollout and the AI Act timeline are others. A manufacturer should assign an
owner to monitor changes, update the PMS procedures, and train the team. A procedure that was correct at launch may be
wrong a year later.

**Industry practice: learning from complaints that are not reportable.** Most complaints are not reportable, but they
can reveal usability problems, labeling confusion, or workflow friction. A complaint about a confusing alert may not
meet the reporting criteria, but it may predict a future use error. The PMS system should analyze complaint themes, not
only individual reportability.

**Industry practice: the board-level metric.** Some manufacturers report a small set of post-market metrics to the
board: serious incident rate, complaint rate, CAPA closure, and top safety signals. The board does not need a model
dashboard, but it does need to know whether the device is safe and whether the organization is responding. A board
metric that is never acted on is decoration; a board metric that triggers questions is governance.

**Industry practice: transparency after a signal.** When a signal is confirmed, the manufacturer faces a choice between
a narrow disclosure and a clear one. A narrow disclosure that omits the population, the denominator, or the limitation
can damage trust when the full story emerges. A clear disclosure that states what is known, what is not known, and what
the manufacturer is doing is more defensible. The regulatory report and the customer communication should be consistent.

**Industry practice: post-market evidence in procurement.** Hospitals increasingly ask vendors for real-world
performance, not only premarket accuracy. A vendor that can present a stratified performance report, a complaint
summary, and a security posture is easier to trust than one that presents a clearance letter. The same evidence that
satisfies a regulator can support a sales conversation, provided it is accurate and appropriately limited.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Complaint-only surveillance | No performance dashboard | Quiet failures missed | Late detection | Add telemetry and audits |
| No denominator | Counts reported without rates | Rates cannot be compared | False alarm or missed signal | Define numerator and denominator |
| Threshold-free dashboard | Lots of charts, no decisions | No trigger for action | Analysis paralysis | Set green/amber/red rules |
| Denominator drift | Case mix changes unnoticed | Metric shifts without device change | Wrong action | Track sampling frame and prevalence |
| Duplicate reports | One event counted many times | Cluster artifact | Overreaction | Deduplicate by patient/device/event |
| Subgroup blindness | Only aggregate metrics | inequity hidden | Harm to subgroup | Stratify and power subgroups |
| Late reporting | Investigation outruns deadline | Clock starts at awareness | Regulatory violation | Triage within days |
| CAPA without effectiveness check | Ticket closed, problem returns | Symptom fix | Recurrence | Verify effectiveness |
| PMS outside QMS | No link to CAPA or risk | Learning lost | Repeat failure | Integrate with QMS |
| Model update unmonitored | Version not recorded | Performance change invisible | Safety signal missed | Version-level metrics |
| Security ignored | No vulnerability feed | Exploit becomes safety event | Harm and privacy loss | Integrate security monitoring |
| Overreaction to noise | Every amber triggers a project | False signals consume resources | Real signals missed | Statistical process control |
| Underreaction to weak signals | "Only one complaint" | Pattern not aggregated | Late field action | Trend analysis |
| No customer communication | Users learn from press | Trust loss | Reputational harm | Proactive notices |

The most dangerous failure is not a missing dashboard. It is a dashboard that creates false confidence. A team that sees
green metrics may stop looking for the complaints, the workflow workarounds, and the subgroup signals that the metrics
do not capture. A good PMS review starts with the question "what could this dashboard be missing?" before it reviews the
colors.

A mature PMS organization also tracks the signals it did not investigate. A backlog of unassessed amber metrics is a
risk in itself. The review should ask whether the team has the capacity to investigate what it detects, and whether the
thresholds are generating more work than the team can absorb. A system that detects everything and investigates nothing
is worse than a system that detects less and acts on what it finds.

The system should also be tested. A tabletop exercise with a simulated signal, a reporting clock, and a cross-functional
team will expose gaps that a document review cannot. Run one before launch, not after the first real event.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The intended population is known | Metrics must be stratified | Off-label use or new site | Usage and site review |
| The denominator is stable | Rates must be comparable | Case-mix or sampling change | Track sampling frame |
| The reference standard is valid | Performance estimates depend on it | Grader variability or drift | Adjudication and audits |
| The data source is complete | Events must be captured | Underreporting or interface gap | Compare sources |
| The reporting clock is understood | Late reports are violations | Awareness date disputed | Training and triage log |
| The thresholds are meaningful | Decisions depend on them | Arbitrary or stale thresholds | Review with clinical and statistical input |
| The QMS will act on signals | Surveillance must lead to action | No authority or budget | CAPA and management review |
| The version is identifiable | Performance must map to software | Multiple deployed versions | Version telemetry |
| The market requirements are current | Timelines and forms change | New regulation or system | Regulatory intelligence |

The limits of this topic are explicit. It does not provide a legal opinion or a reporting decision for a specific event.
It does not cover every FDA or EU requirement, every device type, or every market. It does not replace a PMS plan, a
CAPA procedure, or a quality-system audit. It gives a structured way to design and operate the loop.

The most important limit is organizational. A PMS plan can specify metrics, thresholds, and timelines, but it cannot
create the authority to act. If the safety committee cannot stop a release, order a field action, or escalate to
executive management, the plan is advisory. The governance model should be documented before launch, not invented during
the first crisis.

The final limit is that post-market surveillance cannot prove a negative. It can show that no signal was detected in the
data available, but it cannot show that no harm occurred. The plan should state its detection sensitivity and its blind
spots. A surveillance system that claims certainty is overstating what the data can support.

## 12. Exercises

### Exercise 1 — Distinguish the processes (20 minutes)

Write a one-page table comparing complaint handling, adverse-event reporting, PMS, and PMCF. For each, state the
purpose, the trigger, the output, and the owner. Deliverable: the table plus one example event for each. Acceptance
criteria: the examples are distinct; reporting timelines are not confused with surveillance cadence.

### Exercise 2 — PMS plan (40 minutes)

Draft a PMS plan for a deployed AI device. Include scope, data sources, metrics, thresholds, review cadence, roles, and
escalation. Deliverable: a two-page plan. Acceptance criteria: every metric has a definition and denominator; every
threshold has a decision rule; every role has an owner.

### Exercise 3 — US reportability (30 minutes)

Given three hypothetical events, decide whether each is reportable under 21 CFR Part 803 or Part 806. State the timeline
and the rationale. Deliverable: a decision table. Acceptance criteria: the analysis distinguishes death, serious injury,
malfunction, and correction; the clock start is stated.

### Exercise 4 — EU vigilance (30 minutes)

Given three hypothetical incidents, classify each under MDR Article 87 and state the 2-, 10-, or 15-day timeline.
Deliverable: a decision table. Acceptance criteria: serious public health threat, death/unanticipated serious
deterioration, and other serious incidents are distinguished.

### Exercise 5 — Dashboard design (45 minutes)

Design a one-page dashboard for a deployed model. Include performance, calibration, subgroup, alert burden, override,
ungradable, uptime, and security metrics. Deliverable: a mock dashboard with definitions. Acceptance criteria: each
metric has a denominator and threshold; the dashboard shows trend, not only a snapshot.

### Exercise 6 — Signal investigation (45 minutes)

Write an investigation report for a hypothetical signal. Include the event, the data, the denominator, alternative
explanations, root cause, and conclusion. Deliverable: a two-page report. Acceptance criteria: at least two alternatives
are tested; the conclusion is supported by evidence; the report distinguishes signal from cause.

### Exercise 7 — CAPA and risk update (35 minutes)

Translate the investigation into a CAPA and a risk-file update. Include the correction, the preventive action, the
effectiveness check, and the risk re-evaluation. Deliverable: a CAPA form and a risk update. Acceptance criteria: the
root cause is addressed; the effectiveness check is measurable; the residual risk is stated.

### Exercise 8 — Periodic report (40 minutes)

Draft an outline for an EU PSUR or a US annual PMS report. Include the required sections, the data sources, and the
benefit-risk conclusion. Deliverable: an annotated outline. Acceptance criteria: the outline covers sales volume,
vigilance, PMCF, CAPA, and benefit-risk; the frequency matches the device class.

### Assessment rubric

| Criterion | Weight | Competent | Expert |
|---|---:|---|---|
| Scope and data sources | 15% | Sources listed | Source limitations and access stated |
| Metrics and denominators | 25% | Metrics defined | Denominators, stratifications, and case mix handled |
| Thresholds and decisions | 20% | Thresholds set | Statistical rule and action defined |
| Reporting | 20% | Timelines identified | Jurisdiction matrix and clock start documented |
| Investigation | 10% | Root cause stated | Alternatives tested and evidence linked |
| Feedback | 10% | CAPA opened | Effectiveness verified and risk file updated |

A completed exercise should be usable in a real PMS review, not only in a classroom. The metric definitions should
survive a change in the data source, the thresholds should be defensible to a clinician, and the investigation should
distinguish a signal from a cause. If the exercise produces a dashboard that looks good but cannot answer "compared with
what?" it is incomplete.

Each exercise should produce a record that could be dropped into a PMS file with minimal editing. That means a title, a
date, an author, a data source, and a decision. A classroom answer that cannot be traced is not a rehearsal for
regulated work.

## 13. Self-check answers

**Question 1. What is the difference between a complaint and a signal?** A complaint is an individual communication. A
signal is a pattern or event that suggests a new risk or changed performance. A complaint can be a signal, but most
complaints are not; a signal can exist without a complaint.

**Question 2. What belongs in a PMS plan?** Scope, data sources, metrics and definitions, thresholds, review cadence,
roles, escalation, reporting, and feedback into the QMS. A plan without thresholds or owners is incomplete.

**Question 3. When is a US MDR report due?** Generally within 30 calendar days of becoming aware of a reportable death,
serious injury, or malfunction. A 5-day report is required in certain cases when a remedial action is needed. User
facilities and importers have different timelines.

**Question 4. What are the EU vigilance timelines?** 2 days for a serious public health threat, 10 days for death or
unanticipated serious deterioration, and 15 days for other serious incidents. The clock starts at awareness.

**Question 5. Why is calibration important?** A model can rank patients well but assign probabilities that are too high
or too low. Calibration affects the threshold, the alert burden, and the clinical interpretation. AUC alone does not
capture it.

**Question 6. What makes a signal investigation credible?** A clear event definition, a denominator, a test of
alternative explanations, a root cause, and evidence. A conclusion that blames the user without examining the interface
is not credible.

**Question 7. How does PMS feed the QMS?** Through complaints, CAPA, risk-file updates, design changes, labeling
changes, training, and management review. A PMS output that does not change a controlled record is not a learning loop.

**Question 8. What is the difference between a correction and a removal?** A correction fixes a device or labeling; a
removal takes it off the market. Both can trigger reporting. A market withdrawal for business reasons is not the same as
a removal for safety or violation.

**Question 9. Why can a denominator change create a false signal?** If the case mix or sampling frame changes, the
measured performance can change without any change in the device. The dashboard must track the denominator and the
prevalence alongside the metric.

**Question 10. What is the first thing you do with a severe signal?** Triage it for reportability and patient safety,
assign an owner, start the investigation, and notify the safety committee. Do not wait for a complete root cause to make
the initial report if the timeline requires it.

**Question 11. What is the unit of analysis in a PMS event count?** It can be a report, an event, a patient, or a
device. The choice determines the numerator and must match the denominator. A system that counts reports rather than
events can overstate a cluster when the same event is reported by multiple sources.

**Question 12. Why is a quiet complaint log not proof of safety?** Reporting is affected by culture, effort, and
awareness. A site may not report, or may report through a different channel. A PMS system needs proactive data
collection, not only passive receipt.

**Question 13. What is the difference between a field safety corrective action and a recall?** A field safety corrective
action is an EU concept for an action taken to reduce or eliminate a risk after a device is placed on the market. A
recall is a US concept for the removal or correction of a violative device. The terms overlap but are not identical, and
the reporting routes differ.

## 14. Interview questions

**1. What is post-market surveillance?** A strong answer describes the systematic collection and analysis of data after
launch, the link to risk management and CAPA, and the difference from complaint handling. A weak answer says "tracking
complaints."

**2. How do you design a PMS plan?** A strong answer covers scope, sources, metrics, denominators, thresholds, cadence,
roles, and feedback. A weak answer lists data sources without decisions.

**3. When do you report an adverse event?** A strong answer distinguishes US and EU timelines and the criteria for
reportability. A weak answer says "within 30 days" for everything.

**4. What is a serious incident under the MDR?** A strong answer includes death, serious deterioration, and the
potential for harm from a malfunction. A weak answer limits it to actual patient harm.

**5. How do you monitor a model after deployment?** A strong answer covers performance, calibration, subgroups, drift,
alert burden, override, and version. A weak answer says "we check accuracy."

**6. What is the difference between data drift and concept drift?** A strong answer defines input-distribution change
versus input-outcome relationship change. A weak answer uses the terms interchangeably.

**7. How do you investigate a signal?** A strong answer tests alternative explanations, checks the denominator, reviews
the device and workflow, and documents the root cause. A weak answer assumes the model is at fault.

**8. What is a CAPA effectiveness check?** A strong answer defines a measurable check that the corrective action
prevented recurrence. A weak answer says "we closed the ticket."

**9. How does a PCCP affect PMS?** A strong answer says the PCCP creates version-specific monitoring and acceptance
criteria. A weak answer treats the PCCP as a one-time submission.

**10. What would you do if a hospital reports a problem you cannot reproduce?** A strong answer collects the site data,
checks the configuration and workflow, looks for a subgroup or input difference, and documents the investigation. A weak
answer dismisses it as user error.

**11. How do you know your PMS system is working?** A strong answer points to signals detected, investigations
completed, CAPAs effective, reports filed on time, and changes made. A weak answer points to the number of dashboards.

**12. What would you do if a regulator asked for the denominator?** A strong answer shows the sampling frame, the
inclusion criteria, the data source, and the calculation. A weak answer says the vendor reports the numbers.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| PMS | Post-market surveillance |
| PMS plan | Document defining data, metrics, thresholds, and responsibilities |
| PMSR | Periodic safety update report for Class I under the MDR |
| PSUR | Periodic safety update report for Class IIa, IIb, III |
| PMCF | Post-market clinical follow-up |
| Complaint | Communication alleging a device deficiency |
| Adverse event | Harm or potential harm associated with a device |
| Serious incident | Incident that led or could lead to death or serious deterioration |
| MDR (reporting) | US Medical Device Reporting under 21 CFR Part 803 |
| FSCA | Field safety corrective action |
| FSN | Field safety notice |
| Correction | Repair, modification, or labeling change |
| Removal | Taking a device off the market |
| Recall | Removal or correction of a violative device |
| Signal | Information suggesting a new risk or changed performance |
| Data drift | Change in input distribution |
| Concept drift | Change in input-outcome relationship |
| Performance drift | Change in measured performance |
| Calibration | Agreement between predicted and observed risk |
| Alert burden | Alerts per unit of clinical activity |
| Denominator | Population or exposure used for a rate |
| CAPA | Corrective and preventive action |
| Root cause | Underlying reason for a problem |
| Trend report | Analysis of increases in non-serious incidents |
| UDI | Unique Device Identification |
| EUDAMED | European database on medical devices |
| AEMS | FDA Adverse Event Monitoring System |
| MAUDE | Manufacturer and User Facility Device Experience database |
| Real-world performance | Performance observed in routine care |
| Safety committee | Cross-functional group with authority over safety actions |
| Exposure | Number of patients or devices at risk in a period |
| Standardization | Adjustment for differences in case mix |
| Control chart | Time-series tool for distinguishing signal from noise |
| Field action | Correction or removal affecting devices in the field |
| Benefit-risk determination | Conclusion on whether benefits outweigh risks |

## 16. Sources

All URLs were accessed on 25 October 2026. Verify current versions and reporting systems.

1. US Food and Drug Administration. "Medical Device Reporting (MDR)." https://www.fda.gov/medical-devices/medical-device-safety/medical-device-reporting-mdr-how-report-medical-device-problems
2. Electronic Code of Federal Regulations. 21 CFR Part 803. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-803
3. Electronic Code of Federal Regulations. 21 CFR Part 806. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-806
4. US Food and Drug Administration. "Recalls, Corrections and Removals (Devices)." https://www.fda.gov/medical-devices/postmarket-requirements-devices/recalls-corrections-and-removals-devices
5. US Food and Drug Administration. "FDA Adverse Event Monitoring System (AEMS)." https://www.fda.gov/drugs/surveillance-post-drug-approval-activities/fda-adverse-event-monitoring-system-aems
6. US Food and Drug Administration. "AI-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations." Draft, 7 January 2025. https://www.fda.gov/media/184856/download
7. European Parliament and Council. Regulation (EU) 2017/745 (MDR), Articles 83–89. https://eur-lex.europa.eu/eli/reg/2017/745/oj
8. Medical Device Coordination Group. MDCG 2022-21, "Guidance on Periodic Safety Update Report (PSUR)." https://health.ec.europa.eu/system/files/2023-01/mdcg_2022-21_en.pdf
9. Medical Device Coordination Group. MDCG 2023-3, "Questions and Answers on Vigilance Terms and Concepts." https://health.ec.europa.eu/system/files/2023-02/mdcg_2023-3_en_0.pdf
10. Medical Device Coordination Group. MDCG 2025-10, "Guidance on post-market surveillance." https://health.ec.europa.eu/document/download/a9ad86b7-1b8e-4bae-beb4-48b2b3ed2f05_en
11. International Organization for Standardization. ISO 13485:2016. https://www.iso.org/standard/59752.html
12. International Organization for Standardization. ISO 14971:2019. https://www.iso.org/standard/72704.html
13. International Electrotechnical Commission. IEC 62304:2006+A1:2015. https://webstore.iec.ch/en/publication/22794
14. Wong A, et al. "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients." JAMA Internal Medicine, 2021. https://doi.org/10.1001/jamainternmed.2021.2626
15. US Food and Drug Administration. "De Novo Classification Request DEN180001." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
16. US Food and Drug Administration. "Irregular Rhythm Notification Feature, K231173." https://www.accessdata.fda.gov/cdrh_docs/pdf23/K231173.pdf
17. US Food and Drug Administration. "MAUDE Database." https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfMAUDE/search.CFM
18. US Food and Drug Administration. "Postmarket Requirements (Devices)." https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/postmarket-requirements-devices
19. US Food and Drug Administration. "FDA Launches New Adverse Event Look-Up Tool." Press announcement, 11 March 2026. https://www.fda.gov/news-events/press-announcements/fda-launches-new-adverse-event-look-tool
20. European Parliament and Council. Regulation (EU) 2024/1689 (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
21. European Parliament and Council. Regulation (EU) 2026/1744 amending the AI Act. https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng
22. Medical Device Coordination Group. MDCG 2025-6, "Guidance on the AI Act and medical devices." https://health.ec.europa.eu/system/files/2025-06/mdcg_2025-6_en.pdf

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_the-evidence-package.md`](09_the-evidence-package.md)
- Related topic: [`04_gmlp-and-pccp.md`](04_gmlp-and-pccp.md)
- Related topic: [`06_iso-14971-risk-management.md`](06_iso-14971-risk-management.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
