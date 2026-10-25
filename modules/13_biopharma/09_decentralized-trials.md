# Decentralised Trials: Remote Consent, Digital Endpoints, and the Data Integrity Question

> Module 13 — Biopharma and Clinical Development
> Topic 09 of 10
> Estimated study time: 5 hours
> Prerequisites: Module 03 (data engineering); topics 02, 05, and 08
> Version 2026.3

## 1. Why this topic exists

A decentralised trial moves some or all trial activities away from the investigator's site: consent,
visits, assessments, and data collection happen at home, at a local clinic, or through a device. The
motivation is access and retention. Patients who live far from a specialist centre, who cannot travel
repeatedly, or who would otherwise decline participation can be enrolled. The cost per visit falls,
and the burden on participants falls with it.

The evidence problem is that decentralisation changes the measurement. A six-minute walk test
performed in a corridor by a trained coordinator is not the same measurement as a step count from a
wrist-worn device. A blood pressure reading taken by a nurse after five minutes of rest is not the
same as one taken by a participant at home. A telehealth visit is not the same as a physical
examination. Each substitution must be justified, and the justification is not that the new
measurement is more convenient.

The data integrity problem is that the systems are heterogeneous. A participant's phone, a sponsor's
device, a local clinic's records, and a central platform all produce data, and the chain from the
participant to the durable repository must be documented and controlled. Regulatory expectations do
not relax for decentralised trials; the FDA's September 2024 guidance states that the requirements
are the same as for traditional trials, and the EMA's computerised systems guideline applies the same
data integrity principles.

This topic covers the design of a decentralised trial, the selection and validation of digital
measurement tools, the consent and oversight arrangements, the data flow and integrity controls, and
the missing-data problem that decentralisation amplifies. It follows the AI topic, because many
decentralised measurements are model-derived, and precedes the submission topic, because the data
from these systems must be submission-ready.

### 1.1 The failure pattern

1. A Phase III trial uses a wrist-worn device to measure physical activity as a secondary endpoint.
2. Overall device compliance is 78%, which the protocol considers acceptable.
3. Compliance is 87% in participants under 65 and 62% in participants aged 65 and over.
4. Older participants are also more likely to discontinue the device when they feel worse.
5. The complete-case analysis shows a 12% improvement in activity in the treatment arm.
6. A sensitivity analysis that accounts for the differential missingness reduces the effect to 3%,
   with a confidence interval that includes no effect.
7. The endpoint is reported as supportive only, and the label does not include it.

Step 3 is visible in the data before the analysis. The failure is not the device; it is the decision
to treat a differential compliance pattern as random missingness.

### 1.2 Who is blocked when decentralisation is poorly designed

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Participant | A trial they could have joined | Depends on the design and the site |
| Site coordinator | Control of the measurement | Receives a device and a protocol |
| Data manager | A traceable data flow | Depends on vendors and participants |
| Biostatistician | An unbiased endpoint | Receives data with a missingness pattern |
| Regulatory reviewer | Confidence in the endpoint | Receives a derived measure without validation |
| Sponsor | A label claim | Cannot add evidence after the analysis |

### 1.3 What remote does not mean

Remote does not mean unmonitored, and it does not mean lower quality. It means the monitoring is
central and the quality controls are built into the systems. A trial that replaces site visits with
an app but does not replace site oversight with central oversight has removed a control without
adding one.

### 1.4 The three substitutions

Every decentralised element makes one or more of three substitutions, and each has a different
evidence requirement.

| Substitution | Example | Evidence required |
|---|---|---|
| Mode | Paper PRO to ePRO | Mode equivalence |
| Setting | Site walk test to home walk test | Measurement equivalence and variance |
| Sensor | Clinician count to wearable count | Analytical and clinical validation |

A mode substitution is usually the easiest to justify, a setting substitution changes the conditions,
and a sensor substitution changes the measurement itself. A programme that treats all three as
"digital" will under-invest in the third.

### 1.5 The industry question

> Does moving this activity away from the site change what is measured, who is measured, or how the
> measurement is verified — and if so, is the change acceptable for this endpoint?

The question must be answered per activity, not for the trial as a whole. A trial can decentralise
consent and visit scheduling without changing the endpoint; it cannot decentralise an imaging
assessment without addressing the measurement.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** trial activities to decentralisation options and identify which substitutions change the
   measurement.
2. **Specify** a digital endpoint with its device, sampling, derivation, and validation evidence.
3. **Design** an electronic consent process that satisfies the regulatory requirements and is usable
   by the intended population.
4. **Document** a data flow from device or participant to the durable repository, with attribution and
   audit trail.
5. **Quantify** compliance and missingness by subgroup, and test whether the missingness is
   differential.
6. **Analyse** a decentralised endpoint under complete-case, MAR, and delta-adjusted assumptions, and
   explain how the conclusion changes.
7. **Assess** a decentralised design against the FDA and EMA expectations and identify the gaps.

## 3. The core mental model

### 3.1 The activity map

```text
 Activity                 Site-based                    Decentralised option        Measurement change
 --------                 ----------                    --------------------        ------------------
 Consent                  In person, paper              Electronic, remote          Process change; comprehension risk
 Eligibility screening    Site tests                    Local lab, home sample      Assay and handling change
 Dosing                   Site infusion                 Home nursing, self-inject   Adherence and observation change
 Efficacy assessment      Site visit                    Telehealth, local clinic    Examination component lost
 Patient-reported outcome Site questionnaire            ePRO app                    Mode effect, compliance
 Physical function        Walk test in corridor         Wearable, home test         Construct change
 Safety lab               Site draw                     Home phlebotomy             Handling and timing change
 Imaging                  Site radiology                Local imaging               Protocol and quality change
 AE reporting             Site query                    App, call centre            Ascertainment change
```

The right-hand column is the one that determines whether the decentralised option is acceptable. A
process change is a logistics problem; a measurement change is an evidence problem.

### 3.2 The endpoint validation ladder for digital measures

```text
 1. Analytical validation   Does the sensor measure the quantity accurately?
 2. Technical validation    Does the algorithm derive the endpoint correctly?
 3. Clinical validation     Is the derived endpoint associated with the clinical outcome?
 4. Utility                 Does using it change a decision or improve an outcome?
 5. Qualification           Has a regulator accepted it for a context of use?
```

A device that passes rung 1 and 2 is a measurement instrument. It becomes an endpoint only at rung 3,
and a label claim only at rung 4 or 5. Most decentralised programmes stop at rung 2 and describe the
output as an endpoint.

### 3.3 The data flow

```text
 Participant / device
        |  (sensor, app, ePRO, telehealth)
        v
 Capture layer (sponsor app, vendor platform)
        |  timestamp, attribution, encryption
        v
 Transfer (API, batch, manual upload)
        |  validation, reconciliation
        v
 First durable electronic data repository  <-- source data location per FDA
        |  audit trail, access control, versioning
        v
 Analysis dataset (SDTM, ADaM)
```

The FDA's electronic systems Q&A states that the source data location is the durable electronic data
repository, not the individual device. That means the repository must capture the data as received,
with metadata, before any transformation. A pipeline that writes only the derived value has lost the
source.

### 3.4 Missingness is the central statistical problem

Decentralised measurement produces three kinds of missingness:

| Type | Example | Mechanism | Consequence |
|---|---|---|---|
| Device non-use | Participant does not wear the device | Burden, comfort, forgetfulness | Missing days, biased averages |
| Visit non-completion | Remote visit missed or converted | Travel, connectivity, illness | Missing endpoint |
| Data loss | Transmission failure, app crash | Technical | Missing records |

The mechanism matters. If missingness is unrelated to the outcome, complete-case analysis is
unbiased. If it is related to the outcome, complete-case analysis is biased, and the direction
depends on who is missing. Decentralisation makes the second more likely, because the participants
who disengage from the device are often those whose health is changing.

### 3.5 The equity dimension

Decentralisation can widen access or widen disparity, depending on the design. A bring-your-own-device
(BYOD) model excludes participants without a suitable phone; a sponsor-provided device model reduces
that exclusion but adds logistics. A telehealth-only model excludes participants without broadband or
privacy at home. A home-nursing model excludes participants in regions without coverage.

None of these exclusions is inevitable; each is a design choice with a cost and with a
distributional consequence. The programme should
state which exclusions it accepts and what it would cost to remove them. The design should measure who is excluded, not assume that remote equals accessible.

### 3.6 The three populations of a decentralised trial

| Population | Definition | Why it matters |
|---|---|---|
| Randomised | All participants allocated | The ITT denominator |
| Measured | Participants with a valid endpoint value | The complete-case denominator |
| Effectively measured | Participants whose data reflect their true state | The population the estimate describes |

The third population is the one that missingness creates, and it is never observed. A trial can
randomise 480, measure 431, and effectively measure a younger, healthier subset. The estimate
describes that subset, not the randomised population, unless the missing-data analysis corrects it.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Decentralised trial (DCT) | Trial with some or all activities conducted away from the site | Treated as a fully remote trial | Design |
| Decentralised element | A specific activity conducted remotely or locally | Used for the whole design | Element |
| Telehealth | Remote clinical care or assessment by video or phone | Treated as equivalent to a visit | Mode |
| Electronic consent (eConsent) | Consent obtained through an electronic system | Treated as a signature only | Process |
| Electronic clinical outcome assessment (eCOA) | Clinical outcome captured electronically | Confused with ePRO | Category |
| Electronic patient-reported outcome (ePRO) | PRO captured on a device | Assumed equivalent to paper | Instrument |
| Bring your own device (BYOD) | Participants use their own phone or device | Assumed to be inclusive | Model |
| Digital health technology (DHT) | A device or software used to collect health data | Used for any app | Category |
| Digital endpoint | An endpoint derived from a DHT | Confused with a raw sensor signal | Endpoint |
| Fit for purpose | Evidence that a tool is suitable for the context of use | Treated as a vendor claim | Assessment |
| Analytical validation | Accuracy and precision of the sensor | Confused with clinical validation | Study |
| Technical validation | Correctness of the algorithm that derives the endpoint | Omitted | Study |
| Clinical validation | Association of the endpoint with the outcome | Assumed from plausibility | Study |
| Qualification | Regulatory acceptance for a context of use | Confused with device clearance | Regulatory status |
| Compliance | Proportion of expected measurements that are valid | Confused with adherence to drug | Proportion |
| Valid day | A day meeting the minimum wear or data criteria | Undefined in protocols | Definition |
| Missingness mechanism | The process that produces missing data | Assumed random | MCAR, MAR, MNAR |
| MCAR | Missing completely at random | Assumed without testing | Assumption |
| MAR | Missing at random, given observed data | Confused with MCAR | Assumption |
| MNAR | Missing not at random | Ignored | Assumption |
| Delta adjustment | A sensitivity analysis that shifts imputed values by a stated amount | Treated as the primary analysis | Sensitivity method |
| Multiple imputation | Replacing missing values with draws from a model | Used without stating the model | Method |
| Durable electronic data repository | The controlled system that holds the source data | Confused with the device | System |
| Data originator | The person or system that created the data | Omitted from metadata | Attribution |
| Audit trail | A time-stamped record of changes | Treated as a log file | Record |
| Part 11 | US regulation for electronic records and signatures | Treated as a vendor certification | Regulation |
| Local health care professional (HCP) | A clinician who performs protocol activities locally | Confused with a subinvestigator | Role |
| Direct-to-participant shipment | Investigational product delivered to the participant | Treated as a pharmacy activity | Logistics |
| Risk-based monitoring | Monitoring proportional to risk | Confused with reduced monitoring | Approach |

### 4.1 What "the same requirements" means

The FDA's guidance states that the regulatory requirements for trials with decentralised elements are
the same as for traditional trials. This does not mean the design is the same; it means the
obligations are. Informed consent must still meet 21 CFR Part 50, electronic records must meet Part
11, adverse events must be assessed and reported, and the investigator remains responsible for
conduct.

The practical consequence is that decentralisation adds obligations rather than removing them: a new
consent process, a new data flow, a new oversight model, and a new set of vendor agreements.

## 5. The mechanics, step by step

This procedure designs and operates a trial with decentralised elements. It assumes a protocol, a
target population, and a set of candidate technologies.

### Step 1. Decide which activities to decentralise

**Input:** the protocol's schedule of activities and the target population.

**Output:** an activity map with the proposed mode for each activity and the measurement consequence.

**Rationale:** Decentralisation is not a property of a trial; it is a set of choices. Consent and
scheduling can be remote with little effect on the endpoint; an imaging assessment cannot. The map
makes the choices explicit and separates logistics from measurement.

**Branch point:** if the activity produces a primary endpoint, the decentralised option requires a
validation package. If it produces a secondary or exploratory endpoint, a lighter package may be
acceptable, but the limitation must be stated.

### Step 2. Specify the digital endpoint

**Input:** the clinical concept and the candidate device.

**Output:** an endpoint specification: sensor, sampling frequency, wear criteria, derivation
algorithm, and analysis window.

**Rationale:** "Physical activity" is not an endpoint. "Mean daily step count over the 14 days before
each visit, with at least 10 valid hours on at least 4 days" is. The specification determines the
compliance definition, the missing-data rule, and the validation evidence required.

### Step 3. Establish fit-for-purpose evidence

**Input:** the endpoint specification and the device documentation.

**Output:** a validation summary covering analytical, technical, and clinical evidence, with the gaps
stated.

**Rationale:** A consumer device may be accurate for step counting in a laboratory and inaccurate in
a frail older population using a walker. The evidence must match the intended population and the
intended use. A vendor's validation in healthy adults is not evidence for a disease population.

### Step 4. Design the consent process

**Input:** the protocol, the population, and the regulatory requirements.

**Output:** an eConsent process with the content, the comprehension checks, the signature, the copy
provided to the participant, and the re-consent procedure.

**Rationale:** Consent is an investigator responsibility and cannot be delegated to a local clinician
who is not trial personnel. The electronic process must satisfy Part 50, provide a copy, allow
questions, and be usable by participants with limited digital literacy.

**Edge case:** participants who cannot use the electronic system. A paper or assisted option must be
available, and the choice must not correlate with the endpoint.

### Step 5. Design the data flow and the repository

**Input:** the devices, apps, and vendors.

**Output:** a data flow diagram from capture to durable repository, with the file formats, transfer
methods, validation checks, and retention.

**Rationale:** The durable repository is the source data location. It must hold the data as received,
with metadata, before transformation. A flow that writes only derived values cannot support an
inspection or a re-analysis.

**Branch point — transfer modes:**

| Mode | Use | Risk | Control |
|---|---|---|---|
| Direct API | Sponsor app to repository | Authentication, retry | Encryption, idempotency, monitoring |
| Vendor platform export | Device vendor to sponsor | Format change, delay | Contract, validation, reconciliation |
| Manual upload | Local clinic or site | Transcription error | Dual entry, audit trail |
| Participant-entered | ePRO app | Fabrication, misunderstanding | Timestamps, range checks, reminders |

### Step 6. Train participants and staff

**Input:** the protocol and the device instructions.

**Output:** a training package, a competency record, and a help desk.

**Rationale:** Most device non-use is explained by unclear instructions and lack of support. Training
is an intervention that affects compliance, and compliance affects the endpoint. The training should
be tested with representative participants, not only with staff.

### Step 7. Operate risk-based monitoring

**Input:** the data flow and the risk assessment.

**Output:** a monitoring plan with central checks, triggers, and site or participant follow-up.

**Rationale:** In a decentralised trial, monitoring cannot rely on frequent on-site visits. Central
statistical monitoring of compliance, missingness, and outlier patterns replaces some on-site review,
and it must be designed to detect the failures that matter: a device that is not worn, a site that
is not following the protocol, and a participant who is fabricating entries.

### Step 8. Manage safety remotely

**Input:** the protocol's safety reporting plan.

**Output:** a process for identifying, assessing, and reporting adverse events that are detected
remotely.

**Rationale:** An adverse event reported through an app or a telehealth visit must be assessed by the
investigator, and urgent care must be arranged when needed. The protocol must state how remote events
are evaluated and who is responsible for the assessment.

### Step 9. Quantify compliance and missingness

**Input:** the device and visit data.

**Output:** compliance by participant, by subgroup, and over time; missingness by endpoint and visit.

**Rationale:** Compliance is an intermediate variable on the path to the endpoint. A trial with 78%
compliance has 22% of expected data missing, and the pattern of missingness determines the analysis.

### Step 10. Test whether missingness is differential

**Input:** the compliance and outcome data.

**Output:** a comparison of missingness by treatment arm, by subgroup, and by outcome trajectory.

**Rationale:** Differential missingness is the mechanism that biases complete-case analysis. A
difference between arms, or a correlation between missingness and change in the outcome, is evidence
against MCAR and requires a sensitivity analysis.

### Step 11. Analyse under multiple assumptions

**Input:** the endpoint data and the missingness assessment.

**Output:** complete-case, MAR (multiple imputation), and delta-adjusted estimates, with the
assumptions stated.

**Rationale:** The primary analysis follows the estimand and the pre-specified missing-data strategy.
The sensitivity analyses show how much the conclusion depends on assumptions that cannot be tested.

### Step 12. Document and submit

**Input:** the trial data and the validation evidence.

**Output:** a submission package with the device validation, the data flow, the audit trail, the
missing-data analysis, and the limitations.

**Rationale:** A reviewer must be able to see what was measured, how it reached the repository, and
how the missing data were handled. The documentation is part of the evidence, not an appendix.

### 5.1 Decision table for a proposed decentralised element

| Element | Measurement change | Validation available | Recommendation |
|---|---|---|---|
| eConsent | Process only | Usability evidence | Adopt with a paper fallback |
| Telehealth visit for history | Examination component lost | None required | Adopt for history; retain in-person examination |
| Home phlebotomy | Handling and timing change | Assay stability evidence | Adopt if stability is demonstrated |
| Wearable step count as secondary | Construct change | Technical and clinical evidence | Adopt as secondary with pre-specified missing-data plan |
| Wearable step count as primary | Construct change | Qualification or strong clinical validation | Do not adopt without regulatory agreement |
| Local clinic imaging | Protocol and quality change | Site qualification | Adopt with central review and site qualification |
| ePRO for a validated instrument | Mode effect | Equivalence study | Adopt with mode-equivalence evidence |

### 5.2 Edge cases that change the design

**Connectivity.** A participant without reliable broadband cannot complete video visits. The protocol
needs a phone option, a local clinic option, or both.

**Device loss or upgrade.** A participant's phone is replaced mid-study, and the app's data are not
migrated. The protocol needs a device-change procedure and a bridge in the analysis.

**Time zone and clock changes.** Device timestamps may use the device's local time or UTC. The
repository must record an unambiguous time and the source of the clock.

**Multiple participants per household.** A shared device can mix data between participants unless the
app enforces a login.

**Language and literacy.** Consent and instructions must be available in the participants' languages,
and comprehension must be checked, not assumed.

**Local clinician scope.** A local clinician may perform only activities within their qualification
and not requiring detailed protocol knowledge. The protocol must define the boundary.

**Direct-to-participant shipment.** Investigational product shipped to a home needs temperature
control, accountability, and return procedures.

**Inspection access.** The FDA expects to inspect trial records at a physical location. A fully
remote trial still needs a location where records can be accessed.

### 5.3 Consent in practice

An electronic consent process has five components, and each can fail independently.

| Component | Requirement | Failure mode |
|---|---|---|
| Content | All Part 50 elements, in plain language | Missing risks, unclear alternatives |
| Presentation | Readable on a phone, accessible | Small text, no screen-reader support |
| Comprehension | Participant can explain the study | Signature without understanding |
| Signature | Attributable, dated, linked to the version | Shared login, wrong version |
| Copy | Provided and retained | No record of what the participant received |

The comprehension check is the component most often omitted. A consent that is signed but not
understood is not valid consent, and the risk is higher in a remote process where no coordinator is
present to notice confusion.

### 5.4 Device selection criteria

| Criterion | Question | Why it matters |
|---|---|---|
| Fit for purpose | Does it measure the construct in this population? | The endpoint depends on it |
| Usability | Can participants wear or use it for the study duration? | Compliance |
| Connectivity | Does it work in participants' homes? | Data completeness |
| Battery life | Does it last between charges? | Missing days |
| Data access | Can the sponsor obtain the raw data? | Analysis and traceability |
| Vendor stability | Will the vendor support it for the trial duration? | Continuity |
| Regulatory status | Is it a regulated device, and for what use? | Claim support |
| Cost | Device, data plan, replacement, support | Feasibility |

A device that is accurate but uncomfortable will have poor compliance, and poor compliance is a
measurement problem even when the sensor is perfect.

### 5.5 What the durable repository must hold

The repository is the source of record. At minimum it must hold:

- the raw data as received from the device or app, without transformation;
- the metadata: device identifier, participant identifier, timestamp and time source, app version,
  and data originator;
- the audit trail of any correction or transformation;
- the version of the transfer specification;
- the access log.

A repository that holds only the derived daily step count cannot answer a question about a specific
hour, cannot support an inspection, and cannot be re-analysed if the derivation changes.

### 5.6 Safety reporting when the site is not present

A remote safety process must answer four questions:

1. **How does a participant report an event?** App, phone line, or telehealth visit.
2. **Who assesses it?** The investigator or a delegated clinician, within a stated time.
3. **How is urgent care arranged?** A local provider, an emergency department, or a home visit.
4. **How is it documented?** In the trial record, with the source and the assessment.

A process that relies on the participant to decide whether an event is serious will under-report.
The protocol should give examples and a contact route that does not depend on the participant's
clinical knowledge.

### 5.7 The protocol sections that decentralisation changes

A protocol written for a site-based trial will not support a decentralised one. Six sections need
substantive revision:

| Section | What must be added |
|---|---|
| Schedule of activities | Which visits are remote, in-person, or participant choice |
| Eligibility | Device ownership, connectivity, and digital literacy requirements |
| Informed consent | Electronic process, fallback, re-consent |
| Data management | Data flow, repository, device versions, audit trail |
| Safety reporting | Remote detection, assessment, and urgent care |
| Statistical analysis | Endpoint derivation, valid-day rule, missing-data strategy |

A protocol that adds a device without revising these sections leaves the operational team to invent
the rules, and the invented rules will differ by site.

### 5.8 The missing-data section of the analysis plan

The plan should state, before the analysis:

1. The estimand and how missing data relate to it.
2. The primary missing-data strategy and its assumption.
3. The sensitivity analyses and the departures they represent.
4. The compliance definition and the handling of low-compliance participants.
5. The subgroup analyses for differential missingness.

A plan that says only "multiple imputation will be used" has not stated the assumption, the model,
or the sensitivity analyses. The reviewer cannot tell whether the choice was made before or after the
result.

### 5.9 The central monitoring metrics

A decentralised trial is monitored through a small set of metrics, each with a threshold.

| Metric | Definition | Threshold example | Action |
|---|---|---|---|
| Device compliance | Valid days / expected days | Below 70% at any site | Site support and participant outreach |
| Visit completion | Completed visits / scheduled | Below 85% | Review scheduling and connectivity |
| Data latency | Days from capture to repository | Above 3 days | Vendor escalation |
| Query rate | Queries / data points | Above 5% | Training and form review |
| Adverse event rate | Events / participant-months | Outside the expected range | Safety review |
| Protocol deviation rate | Deviations / participant | Above the plan | Site review |

The thresholds are conventions, but the metrics are not. A trial without them is monitored by
anecdote, and anecdote in a remote trial is what the site coordinator happens to mention.

### 5.10 Training and support as a compliance intervention

In a site-based trial, the coordinator notices when a participant is struggling. In a decentralised
trial, the help desk is the only observer, and its data are a compliance signal.

| Support element | Purpose | Metric |
|---|---|---|
| Onboarding call | Confirm device setup and understanding | Setup completion rate |
| Written guide | Reference for common problems | Support ticket volume |
| Help desk | Resolve problems quickly | Time to resolution |
| Reminders | Prompt data entry and wear | Compliance rate |
| Escalation | Contact participants who stop | Re-engagement rate |

A help desk that logs calls by problem type produces the evidence for improving the design. A help
desk that only resolves tickets produces no learning.

## 6. Worked example

> **Note:** The trial, device, and data are fictional. The regulatory documents cited are real and
> referenced in Section 16.

### 6.1 The trial

A Phase III trial in a chronic respiratory disease randomises 480 participants (240 per arm) to an
investigational therapy or placebo. The primary endpoint is change in six-minute walk distance
(6MWD) at week 24, measured at the site. A secondary endpoint is mean daily step count, measured by a
wrist-worn device provided to all participants.

Participants may complete visits remotely or in person. Consent is electronic, with a paper option.
The device uploads through a sponsor app to a vendor platform, which transfers data nightly to the
sponsor's durable repository.

### 6.2 The operational data

| Metric | Value | Denominator |
|---|---:|---|
| Randomised | 480 | — |
| Week 24 visit completed remotely | 432 | 480 scheduled (90.0%) |
| Week 24 6MWD available | 431 | 480 randomised (89.8%) |
| Expected device days | 80,640 | 480 × 168 days |
| Valid device days | 62,899 | 80,640 (78.0%) |
| Participants with at least 4 valid days per month | 401 | 480 (83.5%) |

Compliance by age:

```text
Under 65: 300 participants, 87% valid days
65 and over: 180 participants, 62% valid days
Weighted: (300 x 0.87 + 180 x 0.62) / 480 = (261 + 111.6) / 480 = 77.6%
```

The weighted rate matches the observed 78.0% within rounding. The gap between the age groups is 25
percentage points, and it is visible before any analysis.

### 6.3 The primary endpoint analysis

Complete-case analysis of 6MWD change at week 24:

```text
Treatment: n = 209, mean change +16.9 m
Control:   n = 222, mean change -1.3 m
Difference: 18.2 m (95% CI 4.1 to 32.3)
```

The treatment effect is statistically detectable and at the lower end of clinical importance for this
disease. The complete-case population excludes 49 participants: 31 in the treatment arm and 18 in the
control arm.

### 6.4 The missingness diagnostic

The difference in missingness between arms (31 versus 18) is 13 participants, or 5.4% of the
randomised population. A chi-square test of 31/240 versus 18/240 gives a p-value of about 0.07, which
is suggestive but not conclusive. The stronger evidence against MCAR is the association with clinical
worsening, not the arm difference.

The reason is visible in the visit data: participants who discontinued the device or missed the remote
visit were more likely to have worsened on the site-reported clinical global impression at week 12.
The missingness is associated with the outcome, which is the definition of MNAR.

### 6.5 The sensitivity analyses

| Analysis | Assumption | Difference (95% CI) |
|---|---|---:|
| Complete case | Missing unrelated to outcome | 18.2 (4.1 to 32.3) |
| Multiple imputation | MAR, given observed covariates and intermediate outcomes | 14.6 (1.2 to 28.0) |
| Delta adjustment, -15 m in the treatment arm | Missing treatment participants did 15 m worse than MAR predicts | 2.4 (-11.8 to 16.6) |

The delta adjustment assumes that the 31 missing treatment participants had a change 15 m lower than
the imputation model predicts. That is a plausible magnitude given the observed association between
worsening and missingness.

Under the delta adjustment, the confidence interval includes zero and the lower bound is negative.
The conclusion changes from "a statistically detectable benefit" to "a benefit that cannot be
distinguished from no effect under a plausible missing-data assumption".

### 6.6 Two diagnostics that reveal a broken analysis

**Diagnostic 1 — compliance is reported only in aggregate.** The overall 78% hides the 25-point gap
between age groups. The diagnostic is to report compliance by age, sex, site, and treatment arm, and
to compare the analysed population with the randomised population on baseline characteristics. In
this example, the analysed population is younger than the randomised population, which explains part
of the apparent benefit.

**Diagnostic 2 — the missing-data strategy is not pre-specified.** If the protocol says only that
"missing data will be handled by multiple imputation" without stating the imputation model, the
assumption, or the sensitivity analyses, the analysis is not reproducible and the choice can be made
after seeing the result. The diagnostic is to compare the protocol's missing-data section with the
analysis plan and the code.

A third check is the source data trail. Pick one participant's week 24 step count and trace it from
the device to the vendor platform to the repository to the analysis dataset. If any step is missing,
the data are not traceable.

### 6.7 Vary one assumption: the delta

If the delta is -5 m rather than -15 m, the adjusted difference is approximately 8.6 m with a
confidence interval that includes zero at the lower bound but is centred above it. The conclusion
becomes "a small benefit that is sensitive to the missing-data assumption".

The decision changes from "do not pursue the endpoint" to "report the endpoint as supportive, with
the sensitivity analysis disclosed". Neither delta is known to be correct; the point of the analysis
is to show how much the conclusion depends on an untestable assumption.

### 6.8 What the example establishes

1. Decentralisation changes who is measured, and the change is visible in compliance by subgroup.
2. Aggregate compliance hides differential missingness.
3. Complete-case analysis assumes MCAR, which is rarely true when the outcome affects engagement.
4. The delta-adjusted result is not the "true" answer; it is a bound on how fragile the conclusion is.
5. Traceability from device to repository is part of the evidence, not an IT concern.

### 6.9 The traceability walk

Pick one participant and walk the data:

1. The device recorded 7,412 steps on 14 March at 23:59 device time.
2. The app uploaded the record on 15 March at 02:14 with the device identifier and the app version.
3. The vendor platform stored the record and transferred it in the nightly batch.
4. The repository holds the raw record with the receipt timestamp and the transfer version.
5. The analysis dataset holds the derived daily step count and the valid-day flag.

If any step is missing, the endpoint cannot be defended. The most common gap is step 4: the
repository holds the derived value and not the raw record, so the derivation cannot be checked.

### 6.10 What would change the decision

The endpoint would become reportable if the missing-data sensitivity analysis were pre-specified and
the delta-adjusted estimate remained above the clinical importance threshold. It would become
non-reportable if the delta-adjusted estimate fell below the threshold, as in this example.

The programme could also change the endpoint definition to reduce sensitivity to missingness, for
example by using the proportion of participants with a clinically meaningful improvement rather than
a mean change. That definition is less affected by a few missing values but loses information and
changes the estimand.

### 6.11 How to read the three estimates together

The three estimates are not three attempts at the same number. They answer three questions:

- **Complete case:** what is the effect in the participants we measured?
- **MAR:** what is the effect if missingness is explained by the observed data?
- **Delta:** what is the effect if the missing participants did worse by a stated amount?- **Delta:** what is the effect if the missing participants did worse by a stated amount?

The three are ordered by the strength of their assumption, from MCAR to MAR to a specific departure.

A report that presents only the complete-case estimate has chosen the most optimistic assumption
without saying so. A report that presents all three lets the reviewer see the range and the
assumption that drives it.

The primary analysis should follow the estimand and the pre-specified strategy. The sensitivity
analyses are not optional; they are the evidence that the conclusion is robust or fragile.

### 6.12 The estimand question

The estimand determines which missing data matter. If the estimand is the effect of treatment
regardless of discontinuation (a treatment-policy estimand), participants who discontinue remain in
the analysis and their outcomes are collected where possible. If the estimand is the effect of
treatment while adhering (a hypothetical estimand), the analysis addresses what would have happened
if everyone adhered, which requires modelling.

A decentralised trial often has more discontinuation and more missing data, so the estimand choice has
a larger effect than in a site-based trial. The choice should be made with the clinical team and
stated in the protocol, not left to the analysis.

The ICH E9(R1) framework requires the estimand to define the population, the treatment condition, the
endpoint, the handling of intercurrent events, and the population-level summary. A missing-data plan
that does not refer to the estimand is a statistical technique in search of a question.

## 7. Code, query, or calculation

The script computes compliance by subgroup, tests the difference in missingness between arms, and
applies a delta adjustment to an imputed difference. It uses the standard library and SciPy.

```python
import math
from scipy.stats import chi2_contingency

def compliance(valid_days, expected_days):
    return valid_days / expected_days

print(f"overall compliance: {compliance(62899, 80640):.1%}")
print(f"under 65: {compliance(0.87*300*168, 300*168):.1%}")
print(f"65 and over: {compliance(0.62*180*168, 180*168):.1%}")

# differential missingness between arms
table = [[31, 240 - 31], [18, 240 - 18]]
chi2, p, dof, _ = chi2_contingency(table, correction=True)
print(f"missingness 31/240 vs 18/240: chi2 {chi2:.2f}, p {p:.4f}")

# delta adjustment: shift the imputed mean in the treatment arm by delta
def adjusted_difference(mar_diff, n_treat, n_control, delta):
    """Approximate shift of the treatment-arm mean by delta, weighted by missingness."""
    missing_share = 31 / 240
    return mar_diff + delta * missing_share

for delta in (-5, -10, -15):
    d = adjusted_difference(14.6, 209, 222, delta)
    print(f"delta {delta:>3} m: adjusted difference {d:5.1f} m")
```

Expected output (Python 3.12, SciPy 1.17, executed):

```text
overall compliance: 78.0%
under 65: 87.0%
65 and over: 62.0%
missingness 31/240 vs 18/240: chi2 3.27, p 0.0704
delta -5 m: adjusted difference 14.0 m
delta -10 m: adjusted difference 13.3 m
delta -15 m: adjusted difference 12.7 m
```

The last block uses a simplified shift, and its output does not reproduce the table in Section 6.5.
That is deliberate: the worked example's delta-adjusted estimate of 2.4 m comes from a full
imputation model applied at the participant level, not from shifting the arm mean. The simplified
calculation shows why the arithmetic must be done at the participant level: shifting a mean by the
missingness share understates the effect of a large delta.

A correct implementation fits the imputation model, applies the delta to the imputed values in the
treatment arm, and re-estimates the difference across multiple imputations. The simplified version is
presented to make the logic visible and to warn against the shortcut.

The chi-square test is itself a simplification: it treats participants as independent and ignores the
covariates that predict missingness. A production analysis would use a logistic model for missingness
with arm, age, and week-12 change as predictors.

## 8. Data, standards, and regulatory context

### 8.1 The FDA guidance

The FDA's final guidance *Conducting Clinical Trials With Decentralized Elements* was issued in
September 2024 (89 FR 76481). Its central statement is that the regulatory requirements are the same
as for traditional trials. The operational expectations include:

- The protocol specifies which visits are in-person, which are remote, and which are participant
  choice.
- The investigator ensures privacy for in-home and telehealth visits.
- Local health care professionals may perform only activities within their qualifications that do not
  require detailed protocol knowledge; they are not subinvestigators.
- Records capture the visit type, location, date, and data originator.
- Informed consent may be electronic, provided Part 50 is satisfied; consent is an investigator
  responsibility and is not delegated to local clinicians.
- Electronic systems and records are subject to Part 11; live video and audio are not themselves
  electronic records, but the visit documentation is.
- Inspections require a physical location where records can be accessed.

### 8.2 The digital health technology guidance

The FDA's *Digital Health Technologies for Remote Data Acquisition in Clinical Investigations*
(December 2023) sets expectations for device selection and data handling:

- The device must be fit for purpose for the context of use.
- The sponsor describes the data flow from the device to the first durable electronic data repository.
- Operational specifications should minimise missing data, with alerts and reminders.
- The statistical analysis plan pre-specifies how missing data and data quality issues are handled,
  including sensitivity analyses.
- For BYOD, a sponsor-provided option must be available so that participants are not excluded.

### 8.3 The electronic records guidance

The FDA's *Electronic Systems, Electronic Records, and Electronic Signatures in Clinical
Investigations: Questions and Answers* (October 2024) addresses attribution, access control, audit
trails, and the location of source data. Its practical points are that the durable repository is the
source data location, that the data originator must be identifiable, and that the metadata must be
retained.

### 8.4 The EMA position

The EMA's *Guideline on Computerised Systems and Electronic Data in Clinical Trials* (2023) applies
ALCOA+ principles to eCOA, ePRO, and BYOD. Its Annex 5 addresses device heterogeneity, app-level
access control, an external time source, data minimisation, and uninstallation without residue. The
guideline also notes that ePRO and BYOD can reduce missing data through reminders and scheduling.

### 8.5 Requirement versus expectation

- **Requirement.** Part 11 for electronic records and signatures; Part 50 for informed consent; the
  protocol and GCP obligations for trial conduct. These apply regardless of decentralisation.
- **Expectation.** Documenting the visit type and data originator, providing a sponsor device option
  for BYOD, and pre-specifying the missing-data handling are expectations expressed in guidance.
- **Convention.** The choice of device, the definition of a valid day, and the compliance threshold
  are design conventions. They should be justified by the endpoint and the population.

### 8.6 Inspection readiness in a remote trial

An inspection of a decentralised trial asks the same questions as any inspection, with additional
focus on the electronic records. The preparation should include:

- a physical location where records can be accessed;
- a data flow diagram and a repository specification;
- the audit trail and access logs for the trial systems;
- the device validation and version history;
- the consent records and the comprehension evidence;
- the monitoring records, including central statistical monitoring.

A programme that cannot produce these on request will lose time during the inspection, and the
findings will focus on the systems rather than on the clinical data.

### 8.7 Data protection in a decentralised trial

A decentralised trial collects data in participants' homes, often on their own devices, and often
including location or activity patterns that could identify them. The data protection assessment
should cover:

- the legal basis for processing, including any special-category data;
- minimisation, so that the app does not collect data the protocol does not need;
- retention on the participant's device and deletion at the end of the trial;
- transfers between vendors and countries;
- the participant's access and deletion rights, subject to trial retention obligations.

A BYOD app that retains data after the trial, or that collects continuous location, creates a risk
that no consent wording fully removes.

### 8.8 Consent version control

A decentralised trial may amend the consent several times. Each version must be linked to the
participants who signed it, and participants must be re-consented when the change affects their
rights, their obligations, or the risks. A system that stores only the latest version cannot
demonstrate which participants agreed to what.

The record should include the version identifier, the date and time of signature, the participant
identifier, and the method of delivery. An electronic signature without the version is incomplete.

## 9. Industry practice and case evidence

### 9.1 What decentralisation is used for

| Element | Adoption | Evidence | Main concern |
|---|---|---|---|
| eConsent | Broad | Usability studies | Comprehension and equity |
| Telehealth visits | Broad | Operational data | Loss of examination |
| ePRO | Broad | Mode-equivalence studies | Compliance and mode effects |
| Home nursing and phlebotomy | Moderate | Operational data | Sample handling |
| Wearables | Growing | Validation studies | Fit for purpose, missing data |
| Direct-to-participant shipment | Growing | Operational data | Accountability, temperature |
| Fully remote trials | Limited | Case reports | Oversight, inspection access |

The pattern is that process elements are widely adopted and measurement elements are adopted more
slowly, because the evidence burden is higher.

### 9.6 What the evidence does not show

| Claim | Status |
|---|---|
| Decentralised trials increase enrolment | Mixed; depends on the elements and the population |
| Decentralised trials reduce cost per participant | Plausible for visits, not for device and support costs |
| ePRO improves data quality | Documented for some instruments; mode effects vary |
| Wearables improve endpoint sensitivity | Depends on the construct and the population |
| Decentralisation increases diversity | Not established; BYOD can reduce it |

The last row is the one most often assumed and least often tested, and it is the row that
determines whether the design widens or narrows access. A remote design can widen access
for people with transport barriers and narrow it for people without devices or connectivity.

### 9.7 Adoption patterns by region

Decentralised elements are adopted unevenly. Telehealth and eConsent are common in North America and
Europe; home nursing is more developed in some European markets; direct-to-participant shipment is
limited by national medicines law in many countries. A global trial therefore needs a country-by-
country assessment of which elements are permissible.

The assessment should be done at protocol design, not at site initiation. A country that cannot
support direct-to-participant shipment may still participate if the site dispenses, but the design
must say so.

### 9.5 The vendor landscape and its risks

A decentralised trial typically involves more vendors than a site-based trial: a device vendor, an
app vendor, a platform vendor, a home-nursing provider, and a logistics provider. Each introduces a
contract, a data flow, and a validation obligation.

| Vendor type | What they provide | Key contractual point |
|---|---|---|
| Device | Sensor and firmware | Data access, version notification |
| App | Participant interface | Validation, update control |
| Platform | Data aggregation | Transfer specification, retention |
| Home nursing | Visits and sample collection | Training, GCP obligations |
| Logistics | Direct-to-participant shipment | Temperature, accountability |

The most common failure is a vendor update that changes the data format or the algorithm without the
sponsor's knowledge. The contract should require notification and a validation period before any
change that affects the data.

### 9.2 What the evidence shows

The FDA and EMA guidance is consistent on two points: decentralisation does not lower the evidence
standard, and missing data must be pre-specified and analysed. The methodological literature on
wearables adds a third: compliance is socially patterned, so a decentralised endpoint can change the
population that is effectively measured.

**Documented:** the guidance positions and the requirement for fit-for-purpose devices.
**Inference:** that most decentralised programmes under-invest in the missing-data analysis relative
to the device validation. The inference is consistent with the guidance's emphasis, but the frequency
is not measured.

### 9.3 Case: a remote visit that changed the endpoint

A programme replaced a site-based physical function test with a home-based test performed by the
participant using a smartphone. The home test correlated well with the site test in a validation
substudy (r = 0.81), but the variance was larger, and the treatment effect size in the trial was
smaller relative to the noise.

**Documented:** the general phenomenon that home-based tests have larger variance than supervised
tests is well described. **Inference:** the specific programme outcome is constructed here to
illustrate that a correlation of 0.81 is not sufficient evidence for an endpoint substitution; the
relevant question is whether the measurement's variance and responsiveness support the trial's power.

### 9.4 What is documented and what is inferred

| Claim | Status |
|---|---|
| The FDA issued final DCT guidance in September 2024 | Documented |
| The FDA requires a sponsor-provided device option for BYOD | Documented in the DHT guidance |
| ePRO can reduce missing data through reminders | Documented in the EMA guideline |
| Compliance is lower in older participants | Documented in multiple device studies; magnitude varies |
| Decentralised trials enrol more diverse populations | Mixed evidence; depends on the elements and the design |
| Most decentralised endpoints lack qualification | Inference; qualification is rare but the denominator is unknown |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Endpoint substituted without validation | No technical or clinical validation | Measurement change | Wrong effect estimate | Fit-for-purpose evidence |
| Aggregate compliance only | No subgroup breakdown | Differential missingness hidden | Biased analysis | Report by subgroup |
| Complete-case primary analysis | No missing-data plan | MCAR assumed | Bias | Pre-specified MAR and delta analyses |
| BYOD without a sponsor option | Device requirement in eligibility | Exclusion of less affluent participants | Equity gap, selection bias | Provide devices |
| eConsent without comprehension check | Signature rate high, queries high | Misunderstanding | Consent validity | Teach-back, usability testing |
| Local clinician overreach | Local staff performing protocol assessments | Role confusion | GCP deviation | Define scope in the protocol |
| Data flow not documented | No diagram, no repository spec | Untraceable source | Inspection finding | Data flow and validation |
| Device clock not controlled | Timestamps from the device | Ambiguous time | Misclassified events | External time source |
| Shared device | Multiple participants, one login | Data mixing | Wrong attribution | Per-participant authentication |
| No inspection location | Fully remote operation | Records inaccessible | Inspection delay | Designate a location |
| Safety reporting gap | Remote events not assessed | Ascertainment loss | Under-reported safety | Remote safety process |
| Retention of raw data | Only derived values stored | Source lost | Re-analysis impossible | Durable repository with raw data |

The most consequential red flag is a primary endpoint measured by a device with no clinical
validation and a missing-data plan that assumes MCAR.

### 10.1 Red flags by element

| Element | Red flag | What it usually means |
|---|---|---|
| Consent | No comprehension check | Signature without understanding |
| Visits | No record of visit type | Cannot distinguish remote from in-person |
| Device | No version history | Changes are untracked |
| Data flow | No repository specification | Source data may be lost |
| Safety | No remote assessment route | Events under-reported |
| Analysis | No missing-data plan | MCAR assumed |
| Oversight | No central monitoring | Problems found late |
| Inspection | No physical location | Records inaccessible |

A programme that checks these eight before start-up avoids most of the failures that appear at
inspection.

### 10.2 The red flag that is a design choice

A trial that decentralises every activity, including the primary endpoint, is not necessarily wrong,
but it is a design choice with a high evidence burden. The red flag is not decentralisation itself;
it is decentralisation without a corresponding increase in validation and monitoring.

The test is simple: for each decentralised element, can the team state the measurement change, the
validation evidence, and the missing-data plan? If any answer is "we will handle it in analysis", the
element is not ready.

### 10.3 The failure that looks like an operational success

A trial can meet its enrolment target and still fail its endpoint because the participants who
enrolled were not the participants who were measured. High enrolment and low compliance can coexist,
and the enrolment dashboard will look healthy.

The defence is to report enrolment and measurement together: the number randomised, the number with a
valid endpoint, and the characteristics of both. A programme that reports only enrolment has reported
the input, not the output. The combination produces a precise-looking
estimate that cannot be defended.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The device measures the intended construct | The endpoint depends on it | Construct drift, population differences | Clinical validation |
| Compliance is unrelated to outcome | Complete-case validity | Symptoms affect engagement | Missingness analysis |
| The data flow is complete | Traceability | Gaps in transfer | Reconciliation |
| Participants can use the technology | The design depends on it | Digital literacy, connectivity | Usability testing |
| The local environment is private | Consent and assessment validity | Shared housing | Participant report |
| The time source is reliable | Event timing | Device clock | External time source |
| The device does not change during the trial | Measurement stability | OS or firmware update | Change control |
| The analysis population is representative | Generalisation | Differential enrolment | Compare with the randomised set |
| The safety process covers remote events | Participant safety | Events not reported | Audit of remote events |
| Inspection access is available | Regulatory compliance | No physical location | Designate a location || Device versions are controlled | Measurement stability | Firmware updates | Change notification and validation || The analysis population is described | Generalisation | Differential missingness | Compare measured with randomised |
| Site oversight is replaced by central oversight | Quality | No monitoring | Central statistical monitoring |
| The app is usable by the population | Compliance and equity | Digital literacy | Usability testing with representative users |

The deepest limit is that decentralisation changes the measurement and the population at the same
time. A trial can have a valid device and still produce a biased estimate because the people who use
it differ from the people who were randomised. Only the subgroup and missingness analyses can reveal
that.

A second limit is that the technology changes faster than the protocol. A device firmware update, an
operating system upgrade, or a vendor acquisition can change the measurement mid-trial, and the
protocol's version-control provisions must cover those changes.

A third limit is that the regulatory guidance is recent. The DCT guidance is final, but the DHT and
electronic records guidance continues to evolve, and the EMA guideline is being applied through
inspections. A programme should check the current texts at design and at submission.

A fifth limit is that decentralisation shifts work from the site to the participant, and the
participant is not compensated for it in the trial budget. The burden shows up as non-compliance, and
non-compliance shows up as missing data.

The practical response is to treat the decision to decentralise as a series of element-level decisions,
each with its own evidence and its own monitoring, rather than as a trial-level identity. A trial can be
80% decentralised and still have a site-measured primary endpoint, and that combination is often the
most defensible. The identity of the trial matters less than the validity of each measurement.

A fourth limit is that the trial's operational data are also evidence. Device compliance, help-desk
logs, and visit records are part of the trial record and may be inspected. A programme that treats
them as operational exhaust rather than as data will not have them when they are needed.

## 12. Exercises

The time estimates assume access to the worked example data.


The exercises assume a trial protocol and a device specification; where these are unavailable, use the
worked example's structure. Each exercise should produce a document that could be added to a trial
master file, and the documents should be consistent with each other.

**Exercise 1 — Activity map (application, 30 minutes).**
Deliverable: an activity map for a trial in a chronic disease, with the decentralised option and the
measurement consequence for each activity.
Acceptance criteria: at least eight activities; measurement change identified for each; a
recommendation for each.
Time: 30 minutes.

**Exercise 2 — Endpoint specification (design, 30 minutes).**
Deliverable: a specification for a wearable-derived endpoint, including sensor, sampling, valid-day
definition, derivation, and analysis window.
Acceptance criteria: valid day defined numerically; derivation reproducible; missing-data rule stated.
Time: 30 minutes.

**Exercise 3 — Compliance analysis (application, 30 minutes).**
Deliverable: compliance by age and sex for a supplied dataset, with a test of differential
missingness.
Acceptance criteria: denominators stated; test named; interpretation limited to association.
Time: 30 minutes.

**Exercise 4 — Delta adjustment (application, 30 minutes).**
Deliverable: complete-case, MAR, and delta-adjusted estimates for a supplied endpoint, with the
assumptions stated.
Acceptance criteria: each estimate labelled by assumption; the delta justified; conclusion stated for
each.
Time: 30 minutes.

**Exercise 5 — eConsent design (design, 40 minutes).**
Deliverable: an eConsent process for a population with mixed digital literacy, including content,
comprehension checks, signature, copy, and re-consent.
Acceptance criteria: Part 50 elements covered; fallback described; usability testing planned.
Time: 40 minutes.

**Exercise 6 — Data flow (design, 35 minutes).**
Deliverable: a data flow diagram from device to durable repository, with formats, transfer methods,
validation checks, and retention.
Acceptance criteria: source location identified; attribution and audit trail described; failure modes
addressed.
Time: 35 minutes.

**Exercise 7 — Regulatory gap assessment (critique, 45 minutes).**
Deliverable: a 450-word assessment of a decentralised design against the FDA DCT and DHT guidance,
with the gaps and remedies.
Acceptance criteria: at least five expectations checked; gaps prioritised; remedies specific.
Time: 45 minutes.

**Exercise 8 — Equity analysis (application, 35 minutes).**
Deliverable: an analysis of who is excluded by a BYOD design, with the enrolment implications and a
mitigation.
Acceptance criteria: exclusion mechanism described; magnitude estimated; mitigation costed.
Time: 35 minutes.

## 13. Self-check answers

**Exercise 1.** Consent, screening, dosing, efficacy assessment, PRO, physical function, safety labs,
imaging, and AE reporting. Consent and scheduling are process changes; efficacy assessment by
telehealth loses the examination; a wearable changes the construct; imaging at a local site changes
quality control. Each activity should have a recommendation and a validation requirement.

**Exercise 2.** A specification might define a valid day as at least 10 hours of wear, require at
least 4 valid days in the 14 days before a visit, derive mean daily steps from the raw acceleration
with a stated algorithm, and state that days with fewer than 10 hours are excluded. The missing-data
rule should state what happens when the minimum is not met.

**Exercise 3.** Report valid days divided by expected days, by age and sex, with the number of
participants in each cell. Test differential missingness between arms with a chi-square or a logistic
model. The interpretation is that missingness is associated with the subgroup, not that the device
causes the difference.

**Exercise 4.** Complete case: the observed difference. MAR: the imputed difference under a stated
model. Delta: the imputed difference after shifting the missing treatment values by a stated amount.
The conclusion should state which estimate is primary and how the others bound it.

**Exercise 5.** The eConsent must include all Part 50 elements, a comprehension check (for example,
teach-back questions), an electronic signature with attribution, a copy for the participant, and a
re-consent procedure for protocol changes. A paper or assisted option must be available, and the
choice should be recorded.

**Exercise 6.** The diagram should show the device, the app, the vendor platform, the transfer, the
durable repository, and the analysis dataset, with the format and the validation at each step. The
repository is the source data location, and it must retain the raw data and metadata.

**Exercise 7 rubric.** Full marks require checks of visit-type documentation, consent, local clinician
scope, data originator, Part 11, inspection access, and missing-data handling, with gaps prioritised
and remedies specific. Deduct marks for a generic statement that the guidance is followed.

**Exercise 8.** A BYOD design excludes participants without a compatible device, which is correlated
with age, income, and digital literacy.**Exercise 8.** A BYOD design excludes participants without a compatible device, which is correlated
with age, income, and digital literacy. The excluded fraction can be estimated from device ownership
in the target population, and the effect on the trial depends on whether device ownership is related
to the outcome.**Exercise 8.** A BYOD design excludes participants without a compatible device, which is correlated
with age, income, and digital literacy. The excluded fraction can be estimated from device ownership
in the target population, and the effect on the trial depends on whether device ownership is related
to the outcome.

A complete answer quantifies the exclusion (for example, 12% of the target population lacks a
compatible device), estimates the effect on the enrolled sample if those participants differ on
baseline severity, and costs the sponsor-provided alternative. The mitigation should be budgeted, not
mentioned.The mitigation should be budgeted, not
mentioned.

A strong answer also distinguishes exclusion at eligibility (the participant cannot join) from
exclusion at analysis (the participant joined but has no valid data). The first changes the
population; the second changes the estimate. The magnitude can be estimated from device ownership in the
target population. The mitigation is a sponsor-provided device, with the cost of provisioning and
support included in the budget.

## 14. Interview questions

**1. What makes a trial decentralised?**
Strong: specific activities moved away from the site, each with a measurement consequence; not a
binary property. Weak: says remote visits.

**2. How do you decide whether to decentralise an assessment?**
Strong: identify whether the measurement changes, require fit-for-purpose evidence, and assess the
missing-data pattern. Weak: decentralises because participants prefer it.

**3. What is a valid day, and why does it matter?**
Strong: a day meeting stated wear and data criteria; it defines compliance and therefore the
denominator. Weak: says any day with data.

**4. Why is complete-case analysis risky in a decentralised trial?**
Strong: missingness is related to outcome and subgroup, so MCAR fails and the estimate is biased.
Weak: says it loses power.

**5. What does a delta adjustment show?**
Strong: how the conclusion changes under a plausible departure from MAR; a sensitivity analysis, not
an estimate. Weak: calls it the true effect.

**6. What are the consent requirements for a remote trial?**
Strong: Part 50 elements, electronic signature, copy, comprehension, investigator responsibility, no
delegation to local clinicians. Weak: says an electronic signature is enough.

**7. Where is the source data in a decentralised trial?**
Strong: the durable electronic data repository, with raw data and metadata, not the device. Weak: says
the device.

**8. How do you monitor a decentralised trial?**
Strong: central statistical monitoring of compliance, missingness, and outliers, with triggers and
follow-up. Weak: says remote monitoring visits.

**9. What are the equity risks of BYOD?**
Strong: exclusion by device ownership, digital literacy, and connectivity; mitigation by providing
devices. Weak: says BYOD is convenient.

**10. What would make you reject a decentralised design?**
Strong: an unvalidated primary endpoint, uncontrollable missingness, or no inspection access.
Weak: cost.

**11. How does the estimand change the missing-data plan?**
Strong: a treatment-policy estimand retains discontinued participants, while a hypothetical estimand
requires modelling adherence; the missing-data strategy follows from the estimand. Weak: says
imputation is always used.

**12. What is the first question you ask about a wearable endpoint?**
Strong: whether the device has been validated for the construct in this population, and whether the
missing-data pattern has been characterised by subgroup. Weak: asks about the brand of the device.

## 15. Key terms

- **Bring your own device (BYOD)** — participants use their own device.
- **Compliance** — proportion of expected measurements that are valid.
- **Decentralised element** — an activity conducted away from the site.
- **Decentralised trial (DCT)** — a trial with decentralised elements.
- **Delta adjustment** — a sensitivity analysis shifting imputed values.
- **Digital endpoint** — an endpoint derived from a digital health technology.
- **Digital health technology (DHT)** — a device or software collecting health data.
- **Durable electronic data repository** — the controlled system holding source data.
- **eCOA** — electronic clinical outcome assessment.
- **eConsent** — electronic informed consent.
- **ePRO** — electronic patient-reported outcome.
- **Fit for purpose** — evidence that a tool suits the context of use.
- **Local health care professional** — a clinician performing local protocol activities.
- **MAR** — missing at random.
- **MCAR** — missing completely at random.
- **MNAR** — missing not at random.
- **Multiple imputation** — replacing missing values with draws from a model.
- **Risk-based monitoring** — monitoring proportional to risk.
- **Telehealth** — remote care or assessment by video or phone.
- **Valid day** — a day meeting stated wear or data criteria.

## 16. Sources

- US Food and Drug Administration. *Conducting Clinical Trials With Decentralized Elements: Guidance
  for Industry, Investigators, and Other Interested Parties*. September 2024.
  https://www.fda.gov/media/167696/download (accessed 25 October 2026).
- Federal Register. Conducting Clinical Trials With Decentralized Elements; Guidance; Availability.
  89 FR 76481, 18 September 2024.
  https://www.federalregister.gov/documents/2024/09/18/2024-21078/conducting-clinical-trials-with-decentralized-elements-guidance-for-industry-investigators-and-other
  (accessed 25 October 2026).
- US Food and Drug Administration. *Digital Health Technologies for Remote Data Acquisition in
  Clinical Investigations*. December 2023. https://www.fda.gov/media/155022/download
  (accessed 25 October 2026).
- US Food and Drug Administration. *Electronic Systems, Electronic Records, and Electronic Signatures
  in Clinical Investigations: Questions and Answers*. October 2024.
  https://www.fda.gov/media/166215/download (accessed 25 October 2026).
- European Medicines Agency. *Guideline on Computerised Systems and Electronic Data in Clinical
  Trials*. 2023.
  https://www.ema.europa.eu/en/documents/regulatory-procedural-guideline/guideline-computerised-systems-and-electronic-data-clinical-trials_en.pdf
  (accessed 25 October 2026).
- ICH. *E6(R3) Good Clinical Practice*. 2025. https://www.ich.org/page/efficacy-guidelines
  (accessed 25 October 2026).
- Demanuele C, et al. Considerations for conducting bring your own "device" (BYOD) clinical studies.
  *Digital Biomarkers*. 2022. https://doi.org/10.1159/000525080 (accessed 25 October 2026).

> **Note:** The trial and data in Section 6 are fictional. The regulatory guidance and the BYOD
> literature cited are real; the interpretation is the course's own.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_ai-in-discovery-and-development.md`](08_ai-in-discovery-and-development.md)
- Next topic: [`10_submission-data.md`](10_submission-data.md)
- Related: [`04_feasibility-and-recruitment.md`](04_feasibility-and-recruitment.md)
- Related: [`05_cdisc-and-trial-standards.md`](05_cdisc-and-trial-standards.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
