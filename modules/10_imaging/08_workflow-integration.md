# Workflow Integration

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 05, 06, and 07
> Version 2026.3

## 1. Why this topic exists

A model that is accurate in a notebook and absent from the reading room has no clinical value. The
gap between the two is workflow integration: the data connection, the latency, the result format,
the display, the worklist, the alert policy, the failure path, and the trust of the people who use
it. Most imaging AI products that fail commercially do not fail because the model is inaccurate.
They fail because the model does not fit the way the department works.

The industry problem is that workflow is treated as an engineering afterthought. The model is
built, validated, and then handed to an integration team that discovers the PACS does not send the
series in the expected order, the inference takes longer than the read, the result cannot be
displayed in the viewer, and the alert fires on studies that the radiologist has already signed.
Each discovery costs a release cycle.

Workflow also changes the regulatory category. The same model can be a triage device, a
concurrent-read device, a second-read device, or an autonomous device, and each role carries a
different evidence burden and a different risk. The role is not a marketing choice; it determines
what the product must prove.

This topic teaches the imaging pathway, the integration standards, the latency budget, the result
objects, the alert policy, and the evaluation of workflow impact. It uses the published evidence
on stroke triage, where the outcome is time to treatment rather than detection accuracy.

### 1.1 The decisions this topic unblocks

| Decision | What workflow knowledge provides |
|---|---|
| Workflow role | Triage, concurrent, second read, autonomous, or quantitative |
| Integration standard | DICOMweb, DIMSE, HL7, FHIR, or a vendor API |
| Latency budget | Whether the result arrives before the read |
| Result format | DICOM SR, SEG, SC, or a worklist flag |
| Alert policy | Threshold, volume, and escalation |
| Failure path | What happens when the service is down |
| Evaluation endpoint | Time to review, time to treatment, alert burden |
| Monitoring | What to log and what to alarm on |

### 1.2 The same model, four products

A haemorrhage detector can be:

- **Triage**: move the study to the top of the worklist. The radiologist still reads every study.
- **Concurrent**: show a mark while the radiologist reads. The radiologist can accept or ignore.
- **Second read**: run after the preliminary read and flag disagreements.
- **Autonomous**: issue a report without radiologist review.

The model is identical. The risk, the evidence, and the integration are not. A triage product can
be cleared with a retrospective study and a notification pathway; an autonomous product needs
prospective evidence and a much narrower intended use.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** the imaging pathway and identify the insertion point of a model.
2. **Choose** a workflow role and state its regulatory and evidence implications.
3. **Design** the data flow from acquisition to result, including the failure path.
4. **Compute** an end-to-end latency budget and compare it to the reading workflow.
5. **Select** an integration standard and explain what it does and does not carry.
6. **Design** a result object that the PACS and the viewer can display.
7. **Estimate** alert burden and its effect on trust and reading time.
8. **Evaluate** workflow impact with time-to-review, time-to-treatment, and missed-finding
   endpoints.

## 3. The core mental model

The imaging pathway is a sequence of handoffs. A model is a component at one handoff, not an
oracle outside the pathway.

```text
Order → Protocol → Acquisition → Reconstruction → Archive → Worklist → Read → Report → Action
  │         │            │              │            │          │        │        │
  │         │            │              │            │          │        │        └─ Communication
  │         │            │              │            │          │        └─ AI mark or second read
  │         │            │              │            │          └─ AI priority
  │         │            │              │            └─ DICOMweb or DIMSE
  │         │            │              └─ Series, metadata, geometry
  │         │            └─ Scanner, dose, contrast, protocol
  │         └─ Body part, indication, contrast
  └─ Clinical question
```

### 3.1 Insertion points

| Insertion point | What the model sees | What it can change | Latency requirement |
|---|---|---|---|
| Order | Order metadata | Protocol suggestion | Before acquisition |
| Acquisition | Scout or live images | Technique, coverage | Seconds |
| Reconstruction | Raw or reconstructed data | Image quality | Seconds to minutes |
| Archive | Completed series | Routing, priors | Minutes |
| Worklist | Series plus metadata | Priority | Minutes |
| Read | Series plus priors | Mark, measurement, report | During read |
| Post-read | Final report | Second read, disagreement | After read |
| Follow-up | Report plus outcome | Surveillance | Days |

A model placed at the wrong point cannot affect the decision. A triage model that runs after the
report is signed is a retrospective audit, not triage.

### 3.2 The latency budget

The end-to-end latency is the sum of the components from image availability to result display:

```text
L_total = L_transfer + L_preprocess + L_inference + L_postprocess + L_display
```

The budget is set by the workflow. For a concurrent-read alert, the result must arrive before the
radiologist reaches the relevant slice, often within 2-5 minutes of series availability. For a
triage alert, the result must arrive before the study would otherwise be read, often within
10-20 minutes. For a batch audit, hours are acceptable.

A model that takes 90 seconds per volume is compatible with a 5-minute read and not with a
30-second triage window. The compute profile from Topic 05 is a workflow input.

### 3.3 The alert burden equation

An alert is useful only if the radiologist acts on it. The burden is the number of alerts per
shift, and the yield is the proportion that are true positives:

```text
alerts per day = studies per day × (sensitivity × prevalence + (1 - specificity) × (1 - prevalence))
PPV = true alerts / total alerts
```

At a prevalence of 10%, sensitivity 0.90, and false positives 0.10 per scan, a department with 120
studies per day gets:

```text
true alerts = 120 × 0.10 × 0.90 = 10.8
false alerts = 120 × 0.10 = 12
total = 22.8 alerts per day
PPV = 10.8 / 22.8 = 0.474
```

Twenty-three alerts per day, half of them false, is a trust problem. The department may accept it
for a critical finding and reject it for a nuisance finding. The threshold is a workflow decision,
not only a statistical one.

### 3.4 Trust is built by calibration and failure behavior

Radiologists trust a system that is transparent about uncertainty, consistent in its output, and
graceful in failure. A model that is silent when it is unsure, that explains its mark with a
region and a score, and that fails open (no alert rather than a wrong alert) is easier to adopt
than a model that is confidently wrong.

Trust is destroyed by alert fatigue, by marks that do not correspond to the image, by latency that
varies unpredictably, and by a support process that cannot explain a failure. The integration
design must address all four.

### 3.5 The sociotechnical system

A workflow is not only software. It is the people, the roles, the incentives, the physical space,
and the culture of the department. A model that is accurate and ignored is a sociotechnical
failure, not a modelling failure.

The dimensions that matter most are workload (does the model add or remove work?), authority (who
is responsible for the final decision?), feedback (does the radiologist see whether the model was
right?), and exception handling (what happens when the model is wrong?). A product that ignores
these dimensions will be evaluated on accuracy and rejected on experience.

### 3.6 The four workflow questions

Before any integration, answer four questions. Who acts on the output? What information do they
have at that moment? What action can they take? What happens if the output is wrong? If any answer
is unclear, the integration is not specified. A model without a clear actor is a dashboard, not a
device.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| PACS | Picture archiving and communication system | Treating as a database | System |
| VNA | Vendor-neutral archive | Treating as a PACS replacement | System |
| Worklist | List of studies to be read | Confusing with the order list | Workflow |
| DIMSE | Traditional DICOM network services | Assuming firewall-friendly | Protocol |
| DICOMweb | RESTful DICOM services | Confusing with a viewer | PS3.18 |
| QIDO-RS | DICOM query service | Using for bulk export | Service |
| WADO-RS | DICOM retrieve service | Assuming rendered output | Service |
| STOW-RS | DICOM store service | Sending non-DICOM | Service |
| UPS-RS | Unified procedure step service | Confusing with a worklist | Service |
| HL7 v2 | Messaging standard for orders and reports | Confusing with FHIR | Standard |
| FHIR | Fast Healthcare Interoperability Resources | Assuming it carries pixels | Standard |
| ImagingStudy | FHIR resource describing a study | Assuming it carries geometry | Resource |
| IHE | Integrating the Healthcare Enterprise | Confusing with a standards body | Initiative |
| AIW-I | AI Workflow for Imaging profile | Confusing with a product | IHE profile |
| CADt | Computer-assisted triage and notification | Confusing with CADe | Regulatory category |
| CADe | Computer-assisted detection | Confusing with CADx | Regulatory category |
| CADx | Computer-assisted diagnosis | Confusing with CADe | Regulatory category |
| Concurrent read | AI output shown during reading | Confusing with second read | Workflow role |
| Second read | AI output after the human read | Confusing with concurrent | Workflow role |
| Autonomous | AI without clinician review | Confusing with triage | Workflow role |
| Triage | Priority assignment | Confusing with diagnosis | Workflow role |
| Turnaround time | Time from order to report | Confusing with read time | Minutes |
| Door-to-groin | Time from arrival to arterial access | Confusing with door-to-needle | Minutes |
| Alert burden | Alerts per unit time | Ignoring false alerts | Count |
| Alert yield | True alerts / total alerts | Confusing with sensitivity | Proportion |
| Silent mode | Running without display | Calling it a trial | Deployment mode |
| Fail open | No output on failure | Confusing with fail safe | Behavior |
| Fail safe | Safe output on failure | Confusing with fail open | Behavior |
| Result object | DICOM SR, SEG, SC, or flag | Assuming a JSON is enough | Object |
| DICOM SR | Structured report | Assuming free text | Object |
| DICOM SEG | Segmentation object | Assuming a viewer supports it | Object |
| DICOM SC | Secondary capture | Losing geometry | Object |
| Presentation state | Display settings | Confusing with image | Object |
| Read time | Time to interpret a study | Confusing with turnaround | Minutes |
| Throughput | Studies per unit time | Confusing with latency | Count/hour |
| Escalation | Notification path for critical findings | Ignoring | Workflow |
| Audit log | Record of inputs and outputs | Ignoring | Log |

### 4.1 Workflow roles and regulatory implications

| Role | Output | Human review | Typical evidence | Risk |
|---|---|---|---|---|
| Triage | Priority | All studies reviewed | Retrospective plus workflow | Delay for others |
| Concurrent | Mark or score | During read | Standalone plus reader study | Alert fatigue |
| Second read | Independent finding | After read | Reader study | Disagreement handling |
| Autonomous | Report | None | Prospective, narrow use | Missed diagnosis |
| Quantitative | Measurement | Used in report | Accuracy and repeatability | Measurement error |

### 4.2 Integration standards

| Standard | Carries | Does not carry | Use |
|---|---|---|---|
| DICOM DIMSE | Images, query, retrieve, store | Modern web semantics | Legacy PACS |
| DICOMweb | Images, metadata, store | Workflow state | Cloud and web |
| HL7 v2 | Orders, reports, ADT | Images | Hospital interfaces |
| FHIR | Resources, references | Pixels, geometry | App integration |
| IHE AIW-I | AI task orchestration | Model accuracy | Workflow profile |
| Vendor API | Product-specific | Portability | Single-vendor integration |

## 5. The mechanics, step by step

This procedure integrates a validated model into a clinical workflow. It assumes the model has
passed the evaluation in Topic 07. Each step states input, output, rationale, branches, and edge
cases.

### Step 1. Choose the workflow role

**Input:** The clinical claim and the department's current pathway.

**Output:** One role: triage, concurrent, second read, autonomous, or quantitative.

**Rationale:** The role determines the evidence, the integration, the display, and the liability.
A model cannot be "just an AI" without a role; the role is what the hospital buys.

**Branch points:**
- Triage: prioritize, do not diagnose.
- Concurrent: mark during read.
- Second read: check after read.
- Autonomous: report without review.
- Quantitative: measure for the report.

**Edge case:** A product may support several roles with different configurations. Each
configuration is a separate claim and needs separate evidence.

### Step 2. Map the current pathway and the insertion point

**Input:** The role and the department's workflow.

**Output:** A process map with the current steps, times, actors, and systems, and the proposed
insertion point.

**Rationale:** The insertion point determines what data are available and what action is possible.
A model inserted after report sign-off cannot change the report.

**Branch points:**
- Before read: worklist priority.
- During read: viewer overlay.
- After read: second-read queue.
- After discharge: audit.

**Edge case:** A department may have several workflows (emergency, inpatient, outpatient). The
model may fit one and not the others.

### Step 3. Define the data flow and the trigger

**Input:** The insertion point and the PACS capabilities.

**Output:** A trigger event, the data retrieved, the preprocessing, the inference, and the result
destination.

**Rationale:** The trigger is usually "series available" or "study completed". The trigger must be
reliable; a missed event is a missed study. The data retrieval must include the correct series,
not the first series in the study.

**Branch points:**
- Series-available trigger: low latency, but the series may still be receiving.
- Study-completed trigger: complete data, higher latency.
- Polling: simple, but delayed and load-generating.
- Event-driven: low latency, but requires integration.

**Edge case:** A study may be reopened or appended. The trigger must handle a new series without
duplicating the result.

### Step 4. Select the integration standard

**Input:** The PACS, the viewer, the network, and the vendor.

**Output:** A standard and a conformance statement for each interface.

**Rationale:** The standard determines portability and support. DICOMweb is the modern choice for
cloud and web; DIMSE is common in legacy PACS. HL7 and FHIR carry orders and reports, not pixels.
A vendor API may be faster to integrate and harder to replace.

**Branch points:**
- Cloud: DICOMweb with WADO-RS, STOW-RS, QIDO-RS.
- On-premises legacy: DIMSE C-FIND, C-MOVE, C-STORE.
- Workflow orchestration: IHE AIW-I with UPS-RS.
- Reporting: HL7 v2 or FHIR DiagnosticReport.

**Edge case:** A PACS may support DICOMweb for query but not for retrieve. The conformance
statement must be tested, not assumed.

### Step 5. Build the inference service

**Input:** The model artefact and the preprocessing specification.

**Output:** A service with a versioned container, a health check, a queue, and a timeout.

**Rationale:** The service must be reproducible and observable. The container pins the library
versions from Topic 03. The queue absorbs bursts. The timeout prevents a stuck study from blocking
the queue.

**Branch points:**
- Synchronous: simple, but latency-sensitive.
- Asynchronous queue: resilient, but adds delay.
- Batch: cheap, but not concurrent.
- Edge: low latency, but limited compute.

**Edge case:** A service that loads the model on every request adds seconds of latency. Load once
and keep warm.

### Step 6. Retrieve and validate the series

**Input:** The trigger and the study identifiers.

**Output:** The selected series, validated against the intended use.

**Rationale:** The model was validated on a protocol envelope. A series outside the envelope
should be rejected or flagged, not silently processed. The validation is the same as Topic 02:
SOP class, geometry, rescale, thickness, contrast.

**Branch points:**
- In envelope: process.
- Out of envelope: skip with a logged reason.
- Missing required series: skip.
- Corrupt: skip and alert.

**Edge case:** A study may contain a series that looks correct but is a derived reconstruction.
Check the SOP class and the derivation description.

### Step 7. Run inference and post-process

**Input:** The validated series.

**Output:** A score, a mask, boxes, or a measurement, with the model version and the preprocessing
version.

**Rationale:** The output must be traceable to the model and the input. The post-processing
(threshold, NMS, connected components) is part of the product and must be versioned.

**Branch points:**
- Score: calibrate and threshold.
- Mask: threshold and clean.
- Boxes: NMS and merge.
- Measurement: compute and validate units.

**Edge case:** A model that outputs a probability must have a calibration map for the deployment
prevalence. A raw sigmoid is not a probability unless it was calibrated.

### Step 8. Write the result object

**Input:** The output and the source identifiers.

**Output:** A DICOM SR, SEG, SC, or a worklist flag, with references to the source series.

**Rationale:** The result must be displayable and auditable. A JSON in a database is not a
radiology result. A DICOM object with the correct references can be opened in the viewer and
linked to the study.

**Branch points:**
- Measurement or category: DICOM SR.
- Mask: DICOM SEG.
- Image overlay: DICOM SC or presentation state.
- Priority: worklist flag plus HL7 message.

**Edge case:** A viewer may not support DICOM SEG. Provide a rendered overlay as a fallback, and
document which object is authoritative.

### Step 9. Update the worklist and display

**Input:** The result object and the worklist.

**Output:** A priority change, an overlay, or a notification, with the time stamp.

**Rationale:** The result must reach the radiologist at the moment of the read. A result that
requires opening a separate application is less likely to be used.

**Branch points:**
- Priority: worklist reorder.
- Overlay: viewer integration.
- Notification: mobile or pager.
- Report: structured report insertion.

**Edge case:** A priority change may be overridden by a radiologist. The override must be logged.

### Step 10. Define the failure path

**Input:** The service, the network, and the workflow.

**Output:** A documented behavior for timeout, error, missing data, and service outage.

**Rationale:** A failed AI service must not block the clinical workflow. The default should be to
continue without the AI result and log the failure, not to hold the study.

**Branch points:**
- Fail open: no result, workflow continues.
- Fail safe: a conservative result, workflow continues.
- Retry: transient errors.
- Fallback: a simpler model or a human review.

**Edge case:** A silent failure that looks like a negative result is dangerous. The absence of a
result must be distinguishable from a negative result.

### Step 11. Instrument the workflow

**Input:** The service and the PACS.

**Output:** Logs of trigger time, retrieval time, inference time, display time, and user action.

**Rationale:** Workflow evaluation requires timestamps. Without them, the team cannot compute
time-to-review or latency. The logs also support incident review.

**Branch points:**
- Per-study log: debugging.
- Aggregate dashboard: monitoring.
- Audit trail: regulatory.
- User feedback: trust and error analysis.

**Edge case:** Logs may contain identifiers. Apply the same de-identification policy as the
images, or store only pseudonymous identifiers.

### Step 12. Evaluate and monitor the workflow

**Input:** The logs and the clinical outcomes.

**Output:** Time-to-review, time-to-treatment, alert burden, override rate, and user feedback.

**Rationale:** The model's accuracy is not the workflow outcome. The outcome is whether the
pathway improved without creating new harm. The evaluation must include the studies that were not
alerted and the alerts that were overridden.

**Branch points:**
- Silent mode: measure distribution and alert rate.
- Pilot: measure workflow with a small group.
- Prospective: measure outcomes.
- Post-market: monitor drift and incidents.

**Edge case:** A decrease in time-to-review can be caused by a change in staffing, not the model.
Use a concurrent comparison or a stepped-wedge design.

### 5.1 Integration testing

Integration testing is not a final step. It is a set of checks that run before the product touches
a clinical study and again after every change.

1. Send a known study through the full path and compare the result to the offline result.
2. Send a study with a missing series and confirm the skip is logged.
3. Send a study with an out-of-envelope thickness and confirm the rejection.
4. Stop the inference service and confirm the workflow continues.
5. Delay the PACS response and confirm the timeout behavior.
6. Send a duplicate study and confirm no duplicate result.
7. Send a study with a burned-in annotation and confirm the preprocessing matches training.
8. Verify the result object opens in the deployment viewer.
9. Verify the audit log has the model, preprocessing, and input identifiers.
10. Verify the priority change is visible in the worklist within the latency budget.

A test that is not run before go-live is a surprise waiting for the first real study.

### 5.2 Incident response and rollback

A deployed model will fail. The question is whether the failure is detected, contained, and
explained. The incident response plan should define:

- The severity levels and the on-call owner.
- The signal that triggers a pause (for example, an alert rate above twice baseline).
- The rollback procedure to the previous model and preprocessing version.
- The communication to the department.
- The post-incident review and the data retention.

A rollback that takes hours is a clinical risk. The model artefact, the preprocessing container,
and the configuration should be versioned so that a rollback is a deployment switch, not a
rebuild.

### 5.3 Training and change management

The radiologists need to know what the output means, what it does not mean, and how to override
it. A one-hour training session is not enough for a concurrent alert; the department needs a
reference guide, a feedback channel, and a review of early cases.

Change management is part of the product. A model that is introduced without it will be evaluated
by rumor rather than by evidence, and the rumor will be shaped by the first false positive.

### 5.4 Multi-site deployment

A product deployed at several sites must handle differences in PACS vendor, viewer, network,
protocol, and reading culture. The integration should be configurable per site without changing
the model. The configuration includes the series selection rule, the threshold, the worklist
mapping, and the escalation path.

A site-specific threshold is acceptable if it is documented and validated at that site. A
site-specific model is a different product and needs its own evidence. The distinction must be
explicit in the deployment record.

### 5.5 Decision table: integration approach

| Constraint | Option | Trade-off | When to use |
|---|---|---|---|
| Cloud PACS | DICOMweb | Needs network and auth | Modern deployment |
| Legacy PACS | DIMSE | Firewall and latency issues | On-premises |
| Multiple viewers | DICOM SR plus SC | Redundant objects | Heterogeneous sites |
| Worklist priority | HL7 or UPS-RS | Integration complexity | Triage |
| Real-time overlay | Vendor API | Portability | Single vendor |
| Batch audit | Nightly export | No latency requirement | Research |

### 5.6 Edge cases that require a written rule

**Mixed slice thickness.** The service must validate thickness and reject or flag out-of-envelope
series. A silent acceptance produces a result the model was not validated for.

**Missing series.** A study without the required series is skipped, not treated as negative. The
skip is logged and visible in the worklist.

**Inconsistent spacing.** The service uses the position-derived geometry. A series with
non-uniform spacing is rejected.

**Rescale absent.** A CT series without rescale is rejected for HU-based models.

**Burned-in annotation.** The service must not rely on the annotation; if the model was trained
with it, the deployment must reproduce the same preprocessing.

**Laterality mismatch.** A side-specific result requires a verified side. A mismatch is escalated,
not silently processed.

**Multi-frame objects.** The service must parse per-frame geometry before inference. A frame-order
assumption is a defect.

**Private tags carrying labels.** The service must not read private tags for features. The
inference input is the validated image and the allowed metadata.

## 6. Worked example

### 6.1 The case

A comprehensive stroke centre receives 120 CT angiography (CTA) studies per day over a 10-hour
window, an average of 12 per hour. Three radiologists read the studies, each taking 18 minutes per
study, giving a capacity of 10 studies per hour. The department is therefore under capacity by 2
studies per hour, and a queue forms.

Large-vessel occlusion (LVO) prevalence in this population is 10%, or 12 studies per day. The
current pathway is first-in, first-out. A triage model flags suspected LVO and moves the study to
the top of the worklist. The model's sensitivity is 0.90 and its false-positive rate is 0.10 per
scan.

### 6.2 Queue arithmetic

The arrival rate is 12 per hour and the service rate is 10 per hour. The utilization is:

```text
ρ = 12 / 10 = 1.2
```

A utilization above 1 means the queue grows without bound during the day. In practice, the
department catches up after hours or with extra reads. The simulation below uses the 120
arrivals, 3 servers, and 18-minute service time to estimate waiting times.

With first-in, first-out, the simulated median wait for an LVO study is 54 minutes and the maximum
is 108 minutes. With priority triage, the median LVO wait falls to 2 minutes and the maximum to 6
minutes. The median wait for non-LVO studies rises slightly, from 58.5 to 59.0 minutes, because
the LVO studies jump the queue.

The 52-minute median reduction is the workflow effect. It is larger than the 11.2-minute
door-to-groin reduction reported in the randomized trial discussed in Section 9, because the
simulation assumes a saturated queue and a perfect priority signal. The simulation is an upper
bound, not a prediction.

### 6.3 Alert burden

At 120 studies per day, prevalence 10%, sensitivity 0.90, and false-positive rate 0.10 per scan:

```text
true alerts = 120 × 0.10 × 0.90 = 10.8
false alerts = 120 × 0.10 = 12.0
total alerts = 22.8 per day
PPV = 10.8 / 22.8 = 0.474
```

The radiologists receive about 23 alerts per day, of which about 11 are true. If each alert takes
30 seconds to review, the added workload is:

```text
22.8 × 0.5 minutes = 11.4 minutes per day
```

That is small relative to three readers × 10 hours, but it is concentrated: an alert that arrives
during a complex case is more disruptive than the average suggests.

### 6.4 Latency budget

The service components are:

| Component | Time |
|---|---:|
| Transfer from PACS | 15 s |
| Preprocessing | 90 s |
| Inference | 20 s |
| Post-processing | 5 s |
| Result write and display | 30 s |
| Total | 160 s |

The result arrives about 2.7 minutes after the series is available. If the radiologist starts the
read 4 minutes after acquisition, the alert is available before the read begins. If the
radiologist starts immediately, the alert arrives during the read, which is acceptable for a
concurrent alert but not for a priority change that requires a worklist refresh.

The latency is dominated by preprocessing. A GPU-accelerated resampling step could reduce it, but
the 90 seconds includes retrieval and conversion, not only interpolation. The team must measure
each component, not estimate the total.

### 6.5 Diagnostic 1: alert fatigue

After two weeks, the override rate is 62%: radiologists dismiss 14 of the 23 daily alerts without
action. The PPV of the alerts that were acted on is higher than the overall PPV, which suggests
that the radiologists are using context the model does not have.

The diagnostic is the override rate by alert type. A high override rate for a specific finding
indicates that the threshold is too low or the finding is not actionable. The remedy is to raise
the threshold for that finding, not to add more alerts.

### 6.6 Diagnostic 2: batch integration latency

The initial integration exports studies to a research server every 30 minutes. The median
time-to-alert is 34 minutes, and 18% of alerts arrive after the study has been signed. The model's
standalone accuracy is unchanged, but the workflow effect is zero or negative.

The diagnostic is the distribution of result time minus report time. If a meaningful fraction is
positive, the integration is a retrospective audit, not a triage product. The remedy is an
event-driven connection with a per-series trigger.

### 6.7 Varying one assumption

Assume the model's sensitivity is 0.75 instead of 0.90, with the same false-positive rate. The true
alerts fall to:

```text
120 × 0.10 × 0.75 = 9.0
```

and the missed LVOs rise to:

```text
12 - 9 = 3 per day
```

The PPV falls to:

```text
9.0 / (9.0 + 12.0) = 0.429
```

The conclusion changes from "the alert burden is acceptable for a critical finding" to "the
missed cases are too many for a triage claim". The sensitivity requirement is a clinical decision,
and it must be set before the model is selected.

### 6.8 The priority that no one sees

The triage model moves an LVO study to the top of the worklist, but the worklist is sorted by
radiologist preference and the priority flag is displayed in a column that the radiologists have
collapsed. The median wait does not change. The model is working; the integration is not.

The diagnostic is the difference between the intended and the observed worklist order. Log the
rank of the study at the time of the read and compare it to the rank the triage policy intended.
If the observed rank is unchanged, the priority is not reaching the user.

### 6.9 Alert timing versus read timing

The queue simulation assumes that a priority change affects the order in which studies are read. In
practice, the radiologist may already have opened the study when the alert arrives. If the read
starts at minute 4 and the alert arrives at minute 2.7, the alert is on time. If the read starts at
minute 1, the alert is late.

The diagnostic is the distribution of `alert time - read start time`. A product with a median of
-1.3 minutes and a 90th percentile of +0.8 minutes is late for one in ten studies. The fix is to
reduce preprocessing latency, not to increase the model's accuracy.

### 6.10 Capacity sensitivity

The queue effect depends on the capacity margin. If the department adds a fourth reader, capacity
rises to 13.3 studies per hour and the utilization falls to 0.9. The queue shrinks, and the LVO
wait under FIFO falls from 54 to about 12 minutes. The triage benefit falls from 52 to about 10
minutes.

The conclusion changes from "triage is the main lever" to "capacity and triage are both levers, and
their relative value depends on staffing". A workflow model that ignores capacity will overstate
the AI effect.

### 6.11 The value of a minute

The clinical value of a minute depends on the finding. For an LVO, every minute of delay is
associated with worse functional outcome, and the trial's 11-minute reduction is meaningful. For a
stable nodule follow-up, a one-day delay is irrelevant. The evaluation should state the time
horizon that matters for the finding, not treat all minutes as equal.

## 7. Code, query, or calculation

The following code simulates the queue with and without priority, computes alert burden and PPV,
and sums the latency budget. It uses only the standard library. The simulation is minute-by-minute
with three servers and a fixed service time.

```python
import statistics

def simulate(priority=False, n=120, service=18, servers=3, interval=5):
    arrivals = []
    for i in range(n):
        arrivals.append((i * interval, i, i % 10 == 0))  # every 10th is LVO

    waiting = []
    next_free = [0] * servers
    waits_lvo, waits_other = [], []
    idx = 0

    for minute in range(0, n * interval + 400):
        while idx < len(arrivals) and arrivals[idx][0] == minute:
            waiting.append(arrivals[idx])
            idx += 1
        for s in range(servers):
            if next_free[s] <= minute and waiting:
                if priority:
                    waiting.sort(key=lambda x: (not x[2], x[0]))
                else:
                    waiting.sort(key=lambda x: x[0])
                scan = waiting.pop(0)
                wait = minute - scan[0]
                (waits_lvo if scan[2] else waits_other).append(wait)
                next_free[s] = minute + service

    return waits_lvo, waits_other

for mode in (False, True):
    lvo, other = simulate(priority=mode)
    all_waits = lvo + other
    label = "priority" if mode else "fifo    "
    print(label,
          "n_lvo", len(lvo),
          "median_lvo", statistics.median(lvo),
          "max_lvo", max(lvo),
          "median_all", statistics.median(all_waits),
          "max_all", max(all_waits))

studies, prevalence = 120, 0.10
sensitivity, fp_per_scan = 0.90, 0.10
true_alerts = studies * prevalence * sensitivity
false_alerts = studies * fp_per_scan
total = true_alerts + false_alerts
print("true alerts  =", round(true_alerts, 1))
print("false alerts =", round(false_alerts, 1))
print("total alerts =", round(total, 1))
print("PPV          = {:.3f}".format(true_alerts / total))

latency = {"transfer": 15, "preprocess": 90, "inference": 20,
           "postprocess": 5, "display": 30}
print("latency (s)  =", sum(latency.values()))
print("latency (min)= {:.1f}".format(sum(latency.values()) / 60))
```

Expected output:

```text
fifo     n_lvo 12 median_lvo 54.0 max_lvo 108 median_all 58.5 max_all 117
priority n_lvo 12 median_lvo 2.0 max_lvo 6 median_all 59.0 max_all 117
true alerts  = 10.8
false alerts = 12.0
total alerts = 22.8
PPV          = 0.474
latency (s)  = 160
latency (min)= 2.7
```

The simulation assumes a saturated queue, a perfect priority signal, and no interruptions. A real
department has breaks, urgent cases, and variable reading times, which widen the distribution. The
numbers are illustrative of the method, not a forecast for a specific site.

## 8. Data, standards, and regulatory context

### 8.1 DICOM and DICOMweb

DICOM PS3.18 defines the RESTful services: QIDO-RS for query, WADO-RS for retrieve, STOW-RS for
store, and UPS-RS for worklist management. These services are the modern integration path for
cloud and web deployments. The traditional DIMSE services (C-FIND, C-MOVE, C-STORE) remain common
in on-premises PACS.

The standard defines the transport, not the workflow. A DICOMweb connection can retrieve a series
without any guarantee about when the series is complete or which series is diagnostic. Those are
workflow semantics, supplied by IHE profiles or by the product.

### 8.2 IHE AI Workflow for Imaging

The IHE Radiology AI Workflow for Imaging (AIW-I) supplement defines how AI tasks are requested,
managed, and performed. It uses DICOM UPS-RS for the worklist and WADO-RS/STOW-RS for data
transfer, with actors for the task requestor, task manager, task performer, image manager, watcher,
and report manager. It supports pull, triggered-pull, and push workflows.

The profile is a integration blueprint, not a regulatory approval. It tells you which transactions
to implement and which actors to assign. It does not tell you whether the model is accurate or
whether the workflow role is appropriate.

### 8.3 HL7 and FHIR

HL7 v2 carries orders, reports, and patient administration messages. FHIR carries resources such
as `ImagingStudy`, `DiagnosticReport`, and `ServiceRequest`. The `ImagingStudy` resource references
a DICOMweb endpoint; the pixels are retrieved with WADO-RS. FHIR does not carry pixel data or the
DICOM geometry.

A integration that uses FHIR for discovery and DICOMweb for content is the current best practice.
A integration that expects FHIR to carry the image is a category error.

### 8.4 FDA and the workflow role

The FDA has separate classifications for computer-assisted triage and notification (CADt) and for
computer-assisted detection and diagnosis (CADe/CADx). The CADt classification, created by the
Viz.AI Contact De Novo in 2018 (DEN170073, 21 CFR 892.2080), covers software that alerts a
clinician to a potential finding but does not replace review. The CADe/CADx classification at
21 CFR 892.2090 covers detection and diagnosis and requires standalone and reader-study evidence.

The workflow role determines the classification. A product that claims triage and a product that
claims diagnosis are different devices, even if the model is the same.

### 8.5 What the standards do not cover

No standard defines the acceptable alert burden, the latency budget, or the escalation policy.
Those are clinical and operational decisions. The standards define the transactions; the
department defines the workflow.

### 8.6 Cybersecurity and privacy

An inference service receives protected health information and returns a clinical result. It is
part of the hospital's attack surface. The minimum controls are authenticated and encrypted
transport, least-privilege access to the PACS, no persistent storage of images beyond the
processing window, and audit logs that record access.

A cloud service adds a data-processing agreement, a region constraint, and a breach notification
path. A model that sends images to a vendor without those controls is a privacy incident waiting
to happen, regardless of its accuracy.

### 8.7 Contracting for uptime and support

A clinical service needs a defined uptime, a support window, and a response time. The contract
should state the service-level objective, the measurement method, the credit or remedy, and the
escalation path. A vendor that offers 99.9% uptime without a measurement method is offering a
slogan.

The clinical risk of downtime depends on the workflow role. A triage service that is down reverts
to the standard queue; an autonomous service that is down may stop reporting. The failure path
must be designed for the role, not for the service.

### 8.8 Data retention and audit

The service should retain the minimum data needed for audit and no more. The retention period, the
storage location, and the access controls should be documented. An audit that requires the original
image should retrieve it from the PACS by reference, not keep a copy in the model service.

## 9. Industry practice and case evidence

### 9.1 Viz.AI and the first CADt clearance

The FDA announced in February 2018 that it permitted marketing of the Viz.AI Contact application,
the first AI-based computer-aided triage and notification software. The De Novo submission
DEN170073 created 21 CFR 892.2080 and product code QAS. The device analyzes CTA for suspected
large-vessel occlusion and sends a notification to a neurovascular specialist; it is not for
diagnosis.

The reported evidence included a 300-patient retrospective study comparing the algorithm with two
neuroradiologists, with AUC 0.91 and about 90% sensitivity and specificity, and a median
scan-to-notification under 6 minutes. The notification is the product; the diagnosis remains with
the clinician.

### 9.2 The randomized trial of LVO triage

Martinez-Gutierrez et al. (JAMA Neurology, 2023) conducted a stepped-wedge cluster randomized trial
across four comprehensive stroke centres in Houston, with 243 patients. AI alerts plus secure
messaging reduced door-to-groin time by 11.2 minutes (95% CI -18.2 to -4.2) and CT-initiation-to-
EVT start by 9.8 minutes (95% CI -16.9 to -2.6). There was no significant difference in IV tPA
times, length of stay, or 90-day functional independence; mortality was lower (13% vs 31%).

The trial is the strongest evidence that triage changes a process outcome. The effect is modest
and does not by itself prove a clinical benefit. The observational studies that report 20-100
minute reductions are subject to secular trends and Hawthorne effects.

### 9.3 Alert fatigue in practice

A concurrent-read product that fires 20-30 alerts per day per reader is likely to be ignored. The
published literature on clinical decision support consistently finds that alert fatigue reduces
response rates and that specificity matters more than sensitivity for adoption. The workflow
design must include a threshold that the department can tolerate, an override mechanism, and a
review of overrides.

### 9.4 Integration latency in practice

A batch integration that exports studies every 30 minutes is common in research deployments and
unacceptable for triage. The difference between a 2.7-minute event-driven latency and a 34-minute
batch latency is the difference between a product and an audit. The integration architecture is a
clinical decision, not an IT detail.

### 9.5 Inference, not fact

The following are inferences: that a specific department will tolerate 23 alerts per day; that a
specific latency budget is achievable on a hospital network; that a specific triage effect will
replicate in another stroke system. Each requires local measurement. The documented facts are the
FDA clearance, the trial result, and the standards.

### 9.6 Procurement and total cost

The purchase price of a model is a small part of the total cost. The larger costs are integration,
validation, training, support, and the radiologist time to review alerts. A product that costs
less but requires a six-month integration is not cheaper.

The procurement questions should include the integration standard, the latency, the failure path,
the update policy, the audit log, the training, and the reference sites. A vendor that cannot
answer them is selling a model, not a product.

### 9.7 What a silent-mode deployment measures

A silent-mode deployment measures the input distribution, the processing coverage, the alert rate
at the deployment threshold, and the agreement between the model and the final report. It does not
measure clinical benefit. The value is that it finds the integration and distribution problems
before anyone relies on the output.

A silent mode that runs for two weeks and processes 95% of studies, with an alert rate close to the
validation estimate, is a green light for a pilot. A silent mode that processes 60% of studies, or
whose alert rate is double the estimate, is a stop signal.

### 9.8 Reference sites and the selection effect

Vendors often cite reference sites that are early adopters with dedicated informatics teams. Those
sites are not a random sample. A product that works at a reference site may fail at a community
hospital with a different PACS, a smaller IT team, and a different case mix. The procurement
process should ask for a site with a similar profile, not only the showcase site.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong workflow role | Claim says triage, output is diagnosis | Role and evidence mismatch | Regulatory and safety risk | Define role first |
| Batch latency | Alerts after sign-off | Periodic export | No workflow effect | Event-driven trigger |
| Alert fatigue | Override rate >50% | Low specificity | Trust loss | Raise threshold |
| Missing trigger | Studies not processed | Event loss | Silent gaps | Reconciliation job |
| Wrong series | Result on localizer | Series selection by description | Invalid output | SOP and geometry checks |
| Viewer incompatibility | Result not displayed | Unsupported object | No use | Fallback overlay |
| Service outage blocks workflow | Study held | Synchronous dependency | Delay | Fail open |
| Silent failure looks negative | No result, no error | Missing status | Missed finding | Explicit status |
| Identifier leak in logs | Names in log files | Poor logging policy | Privacy breach | Pseudonymous logs |
| Priority override not logged | No audit trail | Missing event | Unexplainable behavior | Log overrides |
| Model update without notice | Output changes | Version drift | Trust loss | Versioned deployment |
| Time-to-review improves but harm rises | More alerts, more workup | Threshold too low | Net harm | Outcome monitoring |
| Alert arrives after report | Time distribution positive | Latency | No effect | Measure result time |

### 10.1 Red flags that stop a deployment

1. The result arrives after the report for a meaningful fraction of studies.
2. The override rate is above 50%.
3. The service is synchronous and a failure blocks the worklist.
4. The viewer cannot display the result object.
5. The model version in production is not the validated version.
6. The alert threshold was changed without re-evaluation.
7. The logs contain direct identifiers.

### 10.2 A go-live checklist

1. The workflow role is stated in the labeling.
2. The latency is measured end to end at p95.
3. The alert volume is measured at deployment prevalence.
4. The override path is implemented and logged.
5. The failure path is fail open and visible.
6. The result object opens in the deployment viewer.
7. The model and preprocessing versions are pinned.
8. The reconciliation job compares processed and received studies.
9. The on-call and rollback plan are documented.
10. The department has been trained on the output and its limits.

A go-live that skips any of these is a pilot, not a deployment.

### 10.3 The integration red flag that is easy to miss

The most common integration defect is not a crash. It is a silent gap: studies that were never
triggered, series that were skipped, or results that were written but never displayed. The
reconciliation job that compares received studies to processed studies is the control. Without it,
the team sees only the studies that worked.

### 10.4 The go-live decision

The go-live decision is a judgment, not a metric. The team should be able to say what will be
monitored, who will respond, and what will trigger a pause. If those answers are not ready, the
deployment is not ready, even if the model's AUC is high. The strongest predictor of a successful
go-live is not the model's accuracy but the readiness of the people and the integration.

### 10.5 What success looks like

A successful integration is boring. The study arrives, the result appears before the read, the
alert is right often enough to be trusted, the override is easy, the failure is visible, and the
monthly review shows no drift. The model is not the story; the absence of surprises is.

### 10.6 The first 90 days

The first 90 days are a learning period, not a victory lap. Review every override, every missed
trigger, and every late result. Publish a monthly note to the department with the counts and the
fixes. The note builds trust faster than any accuracy claim.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The trigger fires for every study | Complete coverage | Event loss | Reconciliation |
| The series is complete at trigger | Valid inference | Streaming acquisition | Completeness check |
| The PACS exposes the required metadata | Series selection | Vendor restriction | Conformance test |
| The network latency is stable | Latency budget | Congestion | Measure p95 |
| The viewer supports the result | Display | Legacy viewer | Pilot |
| The department tolerates the alert volume | Adoption | Low specificity | Override audit |
| The workflow role matches the claim | Regulatory | Role creep | Labeling review |
| The failure path is safe | No harm | Silent failure | Failure injection |
| The logs are complete | Evaluation | Missing timestamps | Log audit |
| The model version is pinned | Reproducibility | Auto-update | Version check |

### 11.1 Limits of this topic

This topic covers the integration and workflow evaluation of an imaging model. It does not cover
PACS administration, network security architecture, or the full design of a randomized workflow
trial. It gives the data scientist the vocabulary to specify the integration and measure its
effect, not to run a hospital IT project.

### 11.2 What would change the conclusion

If the department had spare reading capacity, the queue would not form and triage would have little
effect on wait time. If the alert threshold were raised to improve PPV, sensitivity would fall and
the missed LVO count would rise. If the integration were batch rather than event-driven, the
latency would exceed the read time and the product would become an audit.

### 11.3 The limit of simulation

A queue simulation can show the direction and rough size of a workflow effect. It cannot show
whether radiologists will trust the alert, whether the IT team can keep the service running, or
whether the department will change its behavior. Those require a pilot and, ultimately, a
prospective evaluation.

## 12. Exercises

### Exercise 1. Workflow role (recall, 20 minutes)

**Deliverable:** For a pneumothorax detector, state four possible roles and the evidence each
would require.

**Acceptance criteria:** Four roles; evidence type per role; risk per role.

**Time:** 20 minutes.

### Exercise 2. Latency budget (calculation, 25 minutes)

**Deliverable:** For components 20 s transfer, 60 s preprocess, 15 s inference, 5 s postprocess,
and 20 s display, compute the total and compare to a 3-minute budget.

**Acceptance criteria:** Correct total (120 s); margin stated; bottleneck identified.

**Time:** 25 minutes.

### Exercise 3. Alert burden (calculation, 25 minutes)

**Deliverable:** For 200 studies/day, prevalence 8%, sensitivity 0.92, and 0.05 FP/scan, compute
true alerts, false alerts, total, and PPV.

**Acceptance criteria:** Correct values (14.7, 10, 24.7, 0.596); units stated.

**Time:** 25 minutes.

### Exercise 4. Integration standard (application, 35 minutes)

**Deliverable:** A one-page choice of DICOMweb or DIMSE for a given PACS, with the transactions and
the conformance questions.

**Acceptance criteria:** Standard justified; transactions named; three conformance questions.

**Time:** 35 minutes.

### Exercise 5. Failure path (design, 35 minutes)

**Deliverable:** A failure-mode table for service timeout, PACS unavailable, missing series, and
model error, with the clinical behavior.

**Acceptance criteria:** Four modes; fail-open or fail-safe stated; logging and alerting.

**Time:** 35 minutes.

### Exercise 6. Result object (design, 40 minutes)

**Deliverable:** A choice of DICOM SR, SEG, or SC for a segmentation product, with the references
and the fallback.

**Acceptance criteria:** Object type justified; references to source series; viewer fallback;
authoritative object stated.

**Time:** 40 minutes.

### Exercise 7. Workflow evaluation (design, 50 minutes)

**Deliverable:** A silent-mode and pilot plan with the metrics, the comparison group, and the
duration.

**Acceptance criteria:** Metrics include time-to-review and alert burden; comparison design;
sample size rationale.

**Time:** 50 minutes.

### Exercise 8. Trust plan (critique, 40 minutes)

**Deliverable:** A 500-word plan for building radiologist trust, including transparency, override,
feedback, and support.

**Acceptance criteria:** At least four mechanisms; override data used; support escalation.

**Time:** 40 minutes.

## 13. Self-check answers

**A1. What are the four workflow roles and their evidence?**
Triage needs retrospective plus workflow evidence; concurrent needs standalone plus reader study;
second read needs a reader study; autonomous needs prospective evidence and a narrow intended use.

**A2. Latency budget for 20+60+15+5+20 seconds.**
Total 120 seconds, or 2 minutes. It fits a 3-minute budget with a 1-minute margin. The bottleneck
is preprocessing at 60 seconds.

**A3. Alert burden for 200 studies, 8% prevalence, sensitivity 0.92, 0.05 FP/scan.**
True = 200 × 0.08 × 0.92 = 14.72. False = 200 × 0.05 = 10. Total = 24.72. PPV = 14.72/24.72 =
0.595.

**A4. Why does DICOMweb not solve workflow?**
It defines query, retrieve, and store, not when a series is complete, which series is diagnostic,
or how a result is displayed. Those are workflow semantics.

**A5. What is fail open, and when is it appropriate?**
Fail open means the workflow continues without the AI result when the service fails. It is
appropriate when the AI is an adjunct and the standard workflow is safe without it.

**A6. Why is a batch integration incompatible with triage?**
A batch delay of 30 minutes means the alert can arrive after the study is read. Triage requires
the result before the read, so the integration must be event-driven.

**A7. What does a high override rate indicate?**
The alerts are not actionable, often because specificity is low or the finding is not clinically
important. The remedy is to raise the threshold or change the finding, not to add alerts.

**A8. Why log the model and preprocessing versions?**
A result is only reproducible if the model and the input transform are known. A version change can
alter the output without any visible change in the interface.

**A9. Design rubric for Exercise 5.** A strong table states the clinical behavior, the log, the
alert, and the recovery for each failure mode. A weak table says "retry".

**A10. Design rubric for Exercise 7.** A strong plan has a silent phase, a pilot with a comparison
group, pre-specified metrics, and a duration long enough to capture variation. A weak plan measures
only satisfaction.

## 14. Interview questions

**Q1. How do you decide the workflow role?**
A strong answer starts from the clinical decision, the human review, and the risk, then maps to
triage, concurrent, second read, or autonomous. A weak answer starts from the model's accuracy.

**Q2. What is the latency budget for a concurrent alert?**
A strong answer derives it from the read time and the point at which the radiologist needs the
information, typically 2-5 minutes. A weak answer says "fast".

**Q3. Why is alert burden a clinical metric?**
A strong answer explains that alerts consume attention and that a low-specificity alert is ignored,
so the effective sensitivity falls. A weak answer says more alerts are better.

**Q4. How do you integrate with a legacy PACS?**
A strong answer describes DIMSE C-FIND/C-MOVE/C-STORE, the firewall and latency issues, and a
fallback path. A weak answer says "use an API".

**Q5. What is the difference between DICOM SR and SEG?**
A strong answer says SR carries structured findings and measurements, SEG carries voxel masks, and
the choice depends on the output. A weak answer uses them interchangeably.

**Q6. What happens when the AI service is down?**
A strong answer describes fail open, logging, alerting, and reconciliation, so the workflow is not
blocked and the gap is visible. A weak answer says "the study waits".

**Q7. How do you measure workflow impact?**
A strong answer uses time-to-review, time-to-treatment, alert burden, override rate, and a
concurrent or stepped-wedge comparison. A weak answer uses user satisfaction.

**Q8. Why did the LVO trial show a modest effect?**
A strong answer says the workflow, not the detection, is the bottleneck, and that the trial
measured door-to-groin in a real system with confounders. A weak answer says the model was not
accurate enough.

**Q9. How do you prevent alert fatigue?**
A strong answer sets the threshold from the tolerated alert volume, reviews overrides, and retires
low-yield alerts. A weak answer lowers the threshold to catch more cases.

**Q10. What would you monitor after deployment?**
A strong answer monitors latency, coverage, alert rate, override rate, result timing, and a
periodic accuracy audit. A weak answer monitors only uptime.

## 15. Key terms

- **AIW-I** — IHE AI Workflow for Imaging profile.
- **Alert burden** — alerts per unit time.
- **Alert yield** — true alerts / total alerts.
- **Autonomous** — AI without clinician review.
- **CADe** — computer-assisted detection.
- **CADt** — computer-assisted triage and notification.
- **CADx** — computer-assisted diagnosis.
- **Concurrent read** — AI output during reading.
- **DICOM SR** — structured report object.
- **DICOM SEG** — segmentation object.
- **DICOMweb** — RESTful DICOM services.
- **DIMSE** — traditional DICOM services.
- **Door-to-groin** — time to arterial access.
- **Fail open** — continue without AI on failure.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **HL7 v2** — hospital messaging standard.
- **IHE** — Integrating the Healthcare Enterprise.
- **ImagingStudy** — FHIR resource for a study.
- **Latency budget** — allowed time from input to result.
- **PACS** — picture archiving and communication system.
- **QIDO-RS** — DICOM query service.
- **Read time** — time to interpret a study.
- **Second read** — AI check after the human read.
- **Silent mode** — deployment without display.
- **STOW-RS** — DICOM store service.
- **Throughput** — studies per unit time.
- **Triage** — priority assignment.
- **Turnaround time** — order to report.
- **UPS-RS** — unified procedure step service.
- **VNA** — vendor-neutral archive.
- **WADO-RS** — DICOM retrieve service.
- **Worklist** — list of studies to read.

## 16. Sources

- DICOM Standard, PS3 current edition 2026d, PS3.18 Web Services.
  https://www.dicomstandard.org/ (accessed 25 October 2026).
- IHE Radiology, AI Workflow for Imaging (AIW-I) Technical Framework Supplement.
  https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_AIW-I.pdf (accessed
  25 October 2026).
- DICOM Standard, "DICOMweb and other standards."
  https://www.dicomstandard.org/using/dicomweb/dicomweb-and-hl7-fhir (accessed 25 October 2026).
- U.S. Food and Drug Administration, press release, "FDA permits marketing of clinical decision
  support software for alerting providers of a potential stroke in patients," 13 February 2018.
  https://www.fda.gov/news-events/press-announcements/fda-permits-marketing-clinical-decision-support-software-alerting-providers-potential-stroke
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, De Novo DEN170073 (Viz.AI Contact), 2018.
  https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170073.pdf (accessed 25 October 2026).
- 21 CFR 892.2080, Radiological computer aided triage and notification software.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2080
  (accessed 25 October 2026).
- 21 CFR 892.2090, Radiological computer-assisted detection and diagnosis software.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- Martinez-Gutierrez, J. C. et al., "Automated Large Vessel Occlusion Detection Software and
  Thrombectomy Treatment Times: A Cluster Randomized Clinical Trial," JAMA Neurology (2023).
- Tejani, A. S. et al., "Integrating and Adopting AI in the Radiology Workflow: A Primer for
  Standards and Integrating the Healthcare Enterprise (IHE) Profiles," Radiology 2024.
  https://doi.org/10.1148/radiol.232653
- HL7 FHIR, ImagingStudy resource. https://hl7.org/fhir/imagingstudy.html (accessed 25 October
  2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_evaluation.md`](07_evaluation.md)
- Next topic: [`09_regulatory-path.md`](09_regulatory-path.md)
- Related topic: [`02_dicom-and-metadata.md`](02_dicom-and-metadata.md)
- Root survey: [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)

> **Note:** The workflow role is a product decision; the model is only one component of it.
> Measure the pathway, not only the algorithm.

> **Note:** Integration quality is measured by coverage, latency, and override rate, not by uptime alone.
> A service that is up but never displayed is not integrated.
> The pathway is the product; the model is one component of it.
