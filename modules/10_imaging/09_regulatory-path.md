# Regulatory Path

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 05, 07, and 08
> Version 2026.3

## 1. Why this topic exists

A model that is accurate, integrated, and clinically useful cannot be sold in the United States
until it is a legally marketed device, and cannot be sold in the European Union until it carries a
CE mark under the Medical Device Regulation. The pathway determines the evidence, the time, the
cost, and the claims the product can make. It is a product design constraint, not a paperwork
exercise after the model is built.

The industry problem is that regulatory strategy is often deferred until the model is finished.
By then, the intended use is vague, the evidence does not match the claim, the labeling is
incomplete, and the submission is a reconstruction rather than a record. The cost of a late
regulatory decision is measured in months and in re-work.

Imaging AI has a relatively mature regulatory landscape. The FDA has authorized more than 1,600
AI-enabled devices as of September 2026, about three-quarters of them in radiology. The pathways
are known, the special controls are codified, and the guidance on predetermined change control
plans is final. The uncertainty is not the existence of a pathway; it is whether the product's
intended use, evidence, and labeling fit it.

This topic teaches the pathway from intended use to post-market surveillance. It covers the FDA
routes (510(k), De Novo, PMA), the special controls for radiological CADe/CADx and CADt, the
predetermined change control plan, the quality system, cybersecurity, labeling, and the European
framework. It uses cleared devices as case evidence and separates documented facts from inference.

### 1.1 The decisions this topic unblocks

| Decision | What regulatory knowledge provides |
|---|---|
| Intended use | The claim that determines the pathway |
| Device or not | Whether clinical decision support software is exempt |
| Risk class | Class I, II, or III, and the corresponding route |
| Predicate | Whether a substantial-equivalence route exists |
| Evidence | Standalone, reader study, analytical, or prospective |
| Labeling | What must be stated and what must be warned |
| Change control | Whether an update needs a new submission |
| Post-market | What must be monitored and reported |

### 1.2 The claim is the product

A model is a software function. The regulated object is the device, defined by its intended use,
its indications for use, its technology, and its labeling. Two products with the same model but
different claims are different devices. A claim that says "assists detection" and a claim that
says "diagnoses" are not interchangeable, even if the output looks similar.

The discipline is to write the intended use first, in one paragraph, and to test every design and
evidence decision against it. If the evidence does not support the claim, either the evidence or
the claim must change.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** an intended use and indications for use for an imaging AI product.
2. **Determine** whether a software function is a device or clinical decision support.
3. **Classify** an imaging device and select an FDA pathway (510(k), De Novo, or PMA).
4. **Identify** a predicate and assess substantial equivalence, including the limits of the
   comparison.
5. **Design** the evidence package: standalone, reader study, analytical, and clinical.
6. **Draft** a predetermined change control plan for a planned model update.
7. **Map** the quality-system, cybersecurity, and labeling requirements.
8. **Explain** the European MDR and AI Act obligations and their timing.

## 3. The core mental model

Regulation follows risk. The risk determines the class, the class determines the pathway, and the
pathway determines the evidence.

```text
Intended use
   │
   ▼
Is it a device? ── no ──► enforce discretion / not regulated
   │ yes
   ▼
Risk classification
   │
   ├── Class I (low) ──► exempt or 510(k)
   ├── Class II (moderate) ──► 510(k) or De Novo
   └── Class III (high) ──► PMA
                              │
                              ▼
                    Evidence package
                    (standalone, reader, analytical, clinical)
                              │
                              ▼
                    Labeling + QMS + cybersecurity
                              │
                              ▼
                    Marketing + post-market
```

### 3.1 Risk is a function of the claim and the workflow

The same model can be Class II as a triage tool and Class III as an autonomous diagnostic. The
difference is the consequence of an error and the human review in the pathway.

| Workflow role | Human review | Typical risk | Typical pathway |
|---|---|---|---|
| Triage and notification | All studies reviewed | Moderate | De Novo/510(k), CADt |
| Concurrent detection | Reviewed during read | Moderate | 510(k)/De Novo, CADe |
| Second read | Reviewed after read | Moderate | 510(k)/De Novo |
| Autonomous diagnosis | None | High | De Novo or PMA |
| Quantitative measurement | Used in report | Moderate | 510(k) |

### 3.2 Substantial equivalence

A 510(k) relies on substantial equivalence to a legally marketed predicate. The comparison covers
intended use, technology, and performance. A new intended use, a new population, or a new
acquisition can break equivalence.

The practical test is: would the predicate's labeling and evidence support the new claim? If not,
the predicate is not a predicate for this product. A device with a broader claim may need a De Novo
or a PMA.

### 3.3 The evidence pyramid

| Evidence | Question | Typical design |
|---|---|---|
| Analytical | Does the software compute what it claims? | Phantom, reference implementation |
| Standalone | Does the model perform on a fixed dataset? | Retrospective, multi-site |
| Reader study | Does the radiologist perform better with it? | MRMC, blinded |
| Clinical | Does the pathway improve? | Prospective, randomized or stepped-wedge |
| Post-market | Does it keep working? | Registry, complaint, audit |

The pathway determines which levels are required. A CADe 510(k) usually needs standalone plus a
reader study; an autonomous De Novo needs prospective clinical evidence.

### 3.4 Change control

A model that is updated after clearance is a changed device unless the change is within an
approved predetermined change control plan (PCCP). The PCCP describes the modifications, the
modification protocol, and the impact assessment. Without a PCCP, each update may require a new
submission.

The PCCP is a design document. It constrains what the model can learn from and how it can be
retrained, and it requires the data management, retraining, evaluation, and update procedures to
be specified in advance.

### 3.5 The evidence-to-claim matrix

A useful regulatory tool is a matrix with one row per claim element and one column per evidence
source. Every claim element should be supported by at least one evidence source, and every
evidence source should map to a claim element. An empty cell is a gap.

| Claim element | Analytical | Standalone | Reader | Clinical | Labeling |
|---|---|---|---|---|---|
| Detects 4 mm solid nodules | Yes | Yes | Yes | No | Yes |
| Works on 1.0-1.5 mm CT | Yes | Yes | Yes | No | Yes |
| Improves reader detection | No | No | Yes | No | Yes |
| Reduces time to report | No | No | No | Yes | Only if shown |
| Works across scanners | No | Yes | No | No | Yes |

The matrix makes the omission visible. A claim about time reduction with no clinical evidence is
an empty cell, not a marketing statement.

### 3.6 The regulatory question is a design question

The pathway is not chosen after the model. It shapes the data collection, the reference standard,
the reader study, and the labeling. A team that knows it needs a reader study will collect reader
interpretations during the study; a team that does not will discover the gap after the data are
frozen.

The practical consequence is that regulatory affairs should be in the design review, not in the
submission review. The cost of a late change is a new study; the cost of an early question is a
meeting.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Device | Regulated article intended for diagnosis or treatment | Calling software "not a device" | Legal category |
| SaMD | Software as a medical device | Confusing with SiMD | Regulatory category |
| SiMD | Software in a medical device | Confusing with SaMD | Regulatory category |
| Intended use | General purpose of the device | Confusing with indications | Statement |
| Indications for use | Specific population and conditions | Omitting the population | Statement |
| 510(k) | Premarket notification with substantial equivalence | Calling it an approval | Pathway |
| Predicate | Legally marketed device used for comparison | Assuming any cleared device works | Device |
| Substantial equivalence | Similar intended use and technology with comparable performance | Confusing with equivalence of claims | Finding |
| De Novo | Classification route for a novel low/moderate-risk device | Confusing with PMA | Pathway |
| PMA | Premarket approval for high-risk devices | Confusing with 510(k) | Pathway |
| Class I | Low risk | Assuming all Class I are exempt | Class |
| Class II | Moderate risk | Assuming no special controls | Class |
| Class III | High risk | Assuming AI cannot be Class III | Class |
| CADe | Computer-assisted detection | Confusing with CADx | Product code |
| CADx | Computer-assisted diagnosis | Confusing with CADe | Product code |
| CADt | Computer-assisted triage and notification | Confusing with CADe | Product code |
| Special controls | Requirements specific to a device type | Confusing with general controls | Regulation |
| PCCP | Predetermined change control plan | Treating as a post-hoc change | Document |
| GMLP | Good machine learning practice | Treating as a standard | Principles |
| QMS | Quality management system | Treating as documentation only | System |
| ISO 13485 | Medical device QMS standard | Confusing with FDA QSR | Standard |
| IEC 62304 | Medical device software lifecycle | Confusing with ISO 14971 | Standard |
| ISO 14971 | Risk management for medical devices | Confusing with cybersecurity | Standard |
| Cybersecurity | Protection of device and data | Treating as IT only | Requirement |
| UDI | Unique device identification | Confusing with accession | Identifier |
| Labeling | All written and graphical information | Treating as the user manual only | Requirement |
| MDR | EU Medical Device Regulation 2017/745 | Confusing with MDD | Regulation |
| Notified body | EU conformity assessment body | Confusing with FDA | Organization |
| CE mark | EU conformity mark | Treating as FDA approval | Mark |
| AI Act | EU Regulation 2024/1689 | Confusing with MDR | Regulation |
| High-risk AI | AI Act category including medical devices | Assuming all AI is high-risk | Category |
| Clinical evaluation | Evidence of safety and performance under MDR | Confusing with a trial | Process |
| Post-market surveillance | Ongoing monitoring after launch | Confusing with a complaint log | Process |
| Predicate mismatch | Proposed device differs from predicate | Ignoring population | Finding |

### 4.1 FDA pathways compared

| Pathway | Risk | Predicate | Evidence | Typical time |
|---|---|---|---|---|
| 510(k) | Low to moderate | Required | Substantial equivalence, performance | Months |
| De Novo | Low to moderate, novel | None | Classification request, special controls | Months to a year |
| PMA | High | None | Clinical evidence, manufacturing | Years |
| Exempt | Low | None | General controls | Not applicable |

### 4.2 The special controls for CADe/CADx

21 CFR 892.2090, finalized in June 2025, codifies the classification and the special controls. The
controls include:

- A description of the algorithm, inputs, outputs, components, and limitations.
- Pre-specified performance testing protocols and datasets.
- Standalone performance results with diagnostic accuracy measures, and localization accuracy
  where applicable.
- Representative datasets with justified enrichment for important cohorts.
- Reader-study results showing improved assisted read with readers matching the intended users.
- Software documentation: hazard analysis, requirements, design, traceability, testing, and
  cybersecurity.
- Labeling with the patient population, instructions, intended user, compatible hardware and
  protocols, warnings, and a detailed performance summary with stratified sub-analyses.

The regulation is binding. The guidance documents explain how the FDA expects the controls to be
met; they are not law but are the practical standard.

## 5. The mechanics, step by step

This procedure takes a product concept to a regulatory strategy. It assumes the model and the
workflow role are defined. Each step states input, output, rationale, branches, and edge cases.

### Step 1. Write the intended use and indications for use

**Input:** The clinical problem, the workflow, and the model output.

**Output:** A one-paragraph intended use and a specific indications-for-use statement with the
population, anatomy, modality, acquisition, and workflow role.

**Rationale:** The intended use is the anchor for the pathway, the evidence, and the labeling. A
vague statement cannot be tested and cannot be cleared.

**Branch points:**
- Screening: asymptomatic population, prevalence low.
- Diagnosis: symptomatic or referred population.
- Triage: emergency or high-acuity workflow.
- Monitoring: longitudinal comparison.

**Edge case:** A statement that includes "and other findings" is broader than the evidence. The
indications should name the findings the device is validated to detect.

### Step 2. Determine whether the software is a device

**Input:** The intended use and the workflow.

**Output:** A device determination with the rationale.

**Rationale:** Some clinical decision support software is excluded from the device definition
under the 21st Century Cures Act criteria. The criteria include displaying the basis for a
recommendation so that the clinician can independently review it, and not directing the clinician
to a specific course of action for a serious condition.

**Branch points:**
- Device: intended to diagnose, treat, or prevent disease.
- Non-device CDS: supports a clinician who can independently review the basis.
- Exempt: low-risk wellness or administrative software.

**Edge case:** A triage notification that directs the clinician to prioritize a study may still be
a device. The exclusion is narrow, and the analysis should be documented with the specific
criteria.

### Step 3. Classify the device

**Input:** The intended use, the risk, and the existing classifications.

**Output:** A proposed class and product code, with the regulation that applies.

**Rationale:** The class determines the pathway and the controls. Imaging AI is mostly Class II
under 21 CFR 892.2090 or 892.2080, but an autonomous diagnostic claim can be higher risk.

**Branch points:**
- Class I: low risk, general controls.
- Class II: moderate risk, special controls, usually 510(k).
- Class III: high risk, PMA.

**Edge case:** A device can be Class II but still require a De Novo if no predicate exists. The
class and the pathway are related but not identical.

### Step 4. Search for a predicate

**Input:** The proposed classification and the FDA database.

**Output:** A list of candidate predicates with their intended use, technology, and clearance.

**Rationale:** A predicate must have the same intended use and comparable technology. A predicate
with a narrower or different use does not support the claim.

**Branch points:**
- Close predicate: 510(k) with a comparison.
- Distant predicate: 510(k) may fail; consider De Novo.
- No predicate: De Novo.

**Edge case:** A predicate cleared 10 years ago may not reflect current technology, but it is still
a legal predicate. The comparison must address the differences.

### Step 5. Design the evidence package

**Input:** The claim, the predicate, and the special controls.

**Output:** A protocol for standalone, reader, analytical, and (if needed) clinical evidence.

**Rationale:** The evidence must answer the questions the controls ask. A standalone AUC without a
reader study does not support an assisted-read claim; a reader study without a standalone result
does not show the model's intrinsic performance.

**Branch points:**
- CADe: standalone plus reader study.
- CADt: standalone plus workflow evidence.
- CADx: standalone plus reader study plus clinical evidence.
- Autonomous: prospective clinical evidence.

**Edge case:** A model that is accurate but not usable in the reader study may fail the assisted-
read claim even if the standalone result is strong. The reader study tests the combination of model
and interface.

### Step 6. Pre-submission and agency interaction

**Input:** The draft protocol and the questions.

**Output:** A pre-submission package and a record of the agency's feedback.

**Rationale:** The pre-submission is the opportunity to agree on the evidence before it is
collected. It reduces the risk of a deficiency letter and a costly repeat study.

**Branch points:**
- Formal pre-submission: written feedback.
- Informal meeting: less binding.
- No interaction: higher risk.

**Edge case:** The feedback is not a guarantee of clearance. It is advice, and the final decision
rests with the reviewer.

### Step 7. Establish the quality management system

**Input:** The product, the processes, and the suppliers.

**Output:** A QMS conforming to ISO 13485 and the applicable FDA requirements, with design
controls, risk management, and software lifecycle records.

**Rationale:** The QMS is the evidence that the device is manufactured and maintained
reproducibly. For software, it includes the software lifecycle under IEC 62304 and the risk file
under ISO 14971.

**Branch points:**
- Full QMS: required for a Class II/III device.
- Abbreviated QMS: not available for a device; the full system is the baseline.
- Supplier controls: cloud, model, and annotation vendors.

**Edge case:** A model trained outside the QMS is a design-control gap. The training data,
preprocessing, and model selection must be under change control.

### Step 8. Address cybersecurity

**Input:** The device architecture, the network, and the data flows.

**Output:** A cybersecurity risk assessment, threat model, security controls, and a vulnerability
management plan.

**Rationale:** A device that receives images and returns a clinical result is a target. The
requirements include authentication, encryption, access control, logging, and a process for
security patches.

**Branch points:**
- On-premises: hospital network controls.
- Cloud: data-processing agreement, region, and breach notification.
- Hybrid: both.

**Edge case:** A security patch that changes the model or the preprocessing may be a device change.
The change-control process must cover it.

### Step 9. Write the labeling

**Input:** The evidence and the intended use.

**Output:** Labeling with the indications, the intended user, the compatible hardware and
protocols, the performance summary, the warnings, and the limitations.

**Rationale:** The labeling is the contract with the user. It must state what the device does, for
whom, on what inputs, and with what performance. A label that omits the acquisition envelope
invites off-label use.

**Branch points:**
- Full labeling: indications, performance, warnings.
- Physician labeling: technical detail for the user.
- Patient labeling: not typical for imaging AI.

**Edge case:** A performance summary that reports only the pooled AUC, without subgroup results,
does not meet the special controls.

### Step 10. Draft the PCCP

**Input:** The planned updates and the current evidence.

**Output:** A PCCP with the description of modifications, the modification protocol, and the impact
assessment.

**Rationale:** The PCCP allows planned updates without a new submission. It requires the data
management, retraining, evaluation, and update procedures to be specified in advance, so the
agency can review the process rather than each model.

**Branch points:**
- No updates planned: no PCCP.
- Performance updates within a pre-specified envelope: PCCP.
- New intended use: new submission, not a PCCP.

**Edge case:** A PCCP cannot authorize a new population or a new modality. Those are new claims.

### Step 11. Prepare and submit

**Input:** The evidence, the QMS records, the labeling, and the PCCP.

**Output:** A submission with the required sections and a cover letter that states the claim.

**Rationale:** The submission is a structured argument. The reviewer should be able to find the
claim, the evidence, and the labeling without inference.

**Branch points:**
- 510(k): comparison to predicate.
- De Novo: classification request and special controls.
- PMA: clinical and manufacturing evidence.

**Edge case:** A submission that relies on a predicate with a different acquisition protocol will
receive a deficiency. The comparison must address the difference explicitly.

### Step 12. Plan post-market surveillance

**Input:** The approved labeling and the risk file.

**Output:** A post-market plan with complaint handling, periodic reporting, performance
monitoring, and a trigger for re-evaluation.

**Rationale:** The device's performance can change with the population, the scanners, and the
protocols. Post-market surveillance is the evidence that the device continues to perform as
cleared.

**Branch points:**
- Routine reporting: complaints and adverse events.
- Registry: site-level performance.
- Real-world monitoring: input and output drift.

**Edge case:** A drop in alert rate can be a model failure or a change in prevalence. The plan must
distinguish them.

### Step 13. Map the European path

**Input:** The product and the EU market.

**Output:** A MDR classification, a notified-body strategy, a clinical evaluation, and an AI Act
gap analysis.

**Rationale:** The EU requires a CE mark under the MDR, with a clinical evaluation and post-market
surveillance. The AI Act adds obligations for high-risk AI systems, and medical devices are in
scope when a notified body is involved.

**Branch points:**
- Class I: self-certification for some, notified body for others.
- Class IIa/IIb: notified body.
- Class III: notified body with stricter clinical evidence.

**Edge case:** The AI Act timeline has changed. The high-risk obligations for Annex I products,
including medical devices, were deferred to 2 August 2028 by the Digital Omnibus regulation
adopted in 2026. The MDR obligations apply independently.

### 5.1 Analytical validation

Analytical validation shows that the software computes what it claims, independent of the clinical
performance. For an imaging device, it includes:

- A reference implementation of the measurement or segmentation on a phantom or synthetic case.
- A test of the geometry pipeline: spacing, orientation, resampling, and units.
- A test of the output format: the DICOM object, the references, and the display.
- A test of the failure behavior: missing input, corrupt input, and out-of-range input.

Analytical validation is often skipped because it is not a clinical study. It is the evidence that
the model's inputs and outputs are what the labeling says, and it is the first thing a reviewer
will ask about when the clinical result is surprising.

### 5.2 Software documentation and traceability

The special controls require software documentation: hazard analysis, software requirements
specification, software design specification, traceability, system-level test protocol, pass/fail
criteria, results, and cybersecurity. The documentation is not a formality; it is how the reviewer
confirms that the device was built under control.

The traceability matrix links each requirement to a design element, a test, and a result. A
requirement without a test is a gap; a test without a requirement is unexplained. The matrix is
also the fastest way to assess the impact of a change: find the requirement, find the tests, and
re-run them.

### 5.3 The pre-submission questions

A pre-submission is only useful if it asks specific questions. Good questions name the claim, the
proposed evidence, and the alternative. For example: "We propose a standalone study with 300
scans from three sites and a reader study with 8 readers and 200 cases. Is this sufficient to
support the assisted-read claim for 4 mm solid nodules on 1.0-1.5 mm CT?" A vague question such
as "Is our plan acceptable?" invites a vague answer.

The pre-submission package should include the intended use, the proposed protocol, the analysis
plan, and the labeling outline. The agency's feedback should be recorded and mapped to the
submission sections.

### 5.4 Decision table: pathway selection

| Situation | Predicate | Risk | Pathway | Evidence focus |
|---|---|---|---|---|
| Nodule CADe, same use as predicate | Yes | Moderate | 510(k) | Standalone plus reader |
| Novel triage notification | No | Moderate | De Novo | Standalone plus workflow |
| Autonomous diagnosis | No | High | De Novo or PMA | Prospective clinical |
| Measurement tool with predicate | Yes | Moderate | 510(k) | Accuracy and repeatability |
| New modality, no predicate | No | Moderate | De Novo | Analytical plus clinical |
| Update within PCCP | Yes | Moderate | No new submission | Protocol compliance |

### 5.5 Edge cases that require a written rule

**Mixed slice thickness.** The labeling must state the validated thickness range. A model
validated on 1 mm and deployed on 5 mm is outside the claim.

**Missing series.** The labeling must state the required series. A study without the series is not
covered.

**Inconsistent spacing.** The validated geometry must be stated; a series outside it is off-label.

**Rescale absent.** A HU-based claim requires the unit. A series without rescale is outside the
input specification.

**Burned-in annotation.** If the model was trained with markers, the labeling must state whether
markers are allowed. If not, the preprocessing must remove them.

**Laterality mismatch.** A side-specific claim requires a verified side; the labeling must state
the laterality requirement.

**Multi-frame objects.** The validated object types and frame handling must be stated.

**Private tags carrying labels.** A device that reads private tags for features must declare it;
a device that does not must not be affected by them.

## 6. Worked example

### 6.1 The case

A company is developing a CADe device for solid lung nodules of 4 mm or more on low-dose CT. The
intended use is concurrent read: the device marks candidate nodules during the radiologist's
interpretation. The claimed population is adults aged 50-80 with a smoking history, screened with
thin-section CT at 1.0-1.5 mm thickness.

The device is Class II under 21 CFR 892.2090. A predicate exists: a CADe device cleared for solid
nodules of 4 mm or more on thin-section CT, with a concurrent-read workflow. The company plans a
510(k).

### 6.2 Standalone evidence

The standalone study uses 300 scans from three sites, with 180 reference nodules in 220 positive
scans and 80 negative scans. At the operating point of 1.0 false positive per scan, the device
detects 166 of 180 nodules:

```text
sensitivity = 166 / 180 = 0.922
false positives = 1.0 per scan × 300 = 300
```

Wilson 95% confidence interval for 166/180:

```text
p = 0.9222, n = 180
centre ≈ 0.914
half-width ≈ 0.039
CI ≈ 0.874 to 0.953
```

The confidence interval is reported with the unit (lesion), the reference standard (three-reader
consensus with adjudication), and the operating point (1.0 FP/scan).

### 6.3 Reader study

Eight radiologists read 200 scans, half with nodules, twice: unaided and assisted. The mean AUC is
0.81 unaided and 0.86 assisted. The mean difference is 0.05, with a 95% confidence interval of
0.02 to 0.08. The lower bound is above zero, so the study supports an assisted-read superiority
claim.

The reading time falls from 4.1 to 3.8 minutes, but the interval includes zero. The time claim is
not supported. The company removes the time claim from the labeling.

### 6.4 Predicate comparison

The predicate's intended use is "solid nodules of 4 mm or more on CT with slice thickness 1.0-2.5
mm". The new device's claim is 1.0-1.5 mm. The new device is narrower, so the predicate's
population includes the new one. The technology is a convolutional network in both; the new device
uses a 3D network, which is a difference but not a new intended use.

The substantial-equivalence argument is that the new device has the same intended use, a narrower
population, and comparable performance. The narrower population is a limitation, not an expansion,
which supports equivalence but also limits the label.

### 6.5 PCCP

The company plans to update the model every 12 months with new scanner data. The PCCP describes:

- The modifications: retraining with additional scanners, no change in intended use.
- The modification protocol: data management, retraining, evaluation on a locked test set, and
  acceptance criteria.
- The impact assessment: the effect on sensitivity, false positives, and subgroup performance.

The PCCP does not authorize a change in population, modality, or workflow role. Those require a
new submission.

### 6.6 Diagnostic 1: predicate mismatch

The predicate was cleared on 2.5 mm slices, not 1.0-1.5 mm. The new device's claim is thinner
slices, but the predicate's labeling does not cover them. The substantial-equivalence argument
fails on the input specification, and the reviewer issues a deficiency.

The diagnostic is a side-by-side comparison of the predicate's indications and the new device's
indications. A difference in acquisition, population, or workflow is a potential break. The remedy
is either a different predicate or a De Novo.

### 6.7 Diagnostic 2: labeling omission

The draft labeling says "CT" without stating the slice thickness range. The reviewer notes that the
device was validated on 1.0-1.5 mm and that the labeling permits use on 5 mm, which is outside the
evidence. The labeling is revised to state the range and to warn that thicker slices are not
validated.

The diagnostic is a mapping from every inclusion criterion in the evidence to a statement in the
labeling. An inclusion criterion without a labeling statement is an off-label pathway.

### 6.8 Varying one assumption

Assume the company changes the claim from concurrent read to autonomous diagnosis. The workflow
role changes, the risk rises, and the predicate no longer supports the claim because the predicate
is a concurrent-read device. The pathway becomes De Novo (or PMA), and the evidence needs a
prospective study with a clinical endpoint, not a retrospective reader study.

The conclusion changes from "510(k) with standalone and reader evidence" to "De Novo with
prospective evidence and a narrow intended use". The model did not change; the claim did.

### 6.9 Third diagnostic: the subgroup that was not enrolled

The standalone study enrolled adults aged 50-80 with a smoking history, but the labeling says
"adults". A reviewer notices that the evidence does not cover adults outside the age range, and
the labeling is revised to match the enrolled population.

The diagnostic is a comparison of the enrollment criteria with the indications for use. A broader
label than the evidence is a gap, even if the model is likely to work. The remedy is either a
narrower label or additional evidence.

### 6.10 Second variation: the PCCP scope

Assume the company's PCCP includes a plan to expand from solid nodules to part-solid nodules. The
reviewer rejects the PCCP because part-solid nodules are a different finding with a different
clinical meaning, not a performance update. The company must submit a new 510(k) or De Novo for the
expanded claim.

The lesson is that a PCCP covers performance updates within the cleared intended use, not new
claims. The scope boundary is the intended use, not the model architecture.

### 6.11 Cost and timeline arithmetic

A 510(k) with a standalone study and a reader study is not free. A planning estimate:

| Item | Estimate |
|---|---:|
| Regulatory affairs | 150,000 USD |
| Standalone study (300 scans, 3 sites) | 120,000 USD |
| Reader study (8 readers, 200 cases) | 180,000 USD |
| QMS and cybersecurity | 100,000 USD |
| Submission and response | 80,000 USD |
| Total | 630,000 USD |

A De Novo with a prospective study can be several times larger. The estimate is illustrative and
varies by product and region. The point is that the pathway is a budget line, not a footnote.

### 6.12 What the reviewer sees

The reviewer sees the claim, the evidence, and the labeling, in that order. If the claim is
broader than the evidence, the review stops. If the evidence is strong but the labeling is vague,
the review asks for clarification. If the labeling is clear but the evidence is a single-site
study, the review asks for generalizability.

The submission should be written for that reading order. A cover letter that states the claim and
the evidence in one page is more useful than a hundred pages of appendices without a map.


### 6.13 The decision log

Every regulatory decision should be recorded with the date, the decision, the evidence, and the
owner. The log becomes the narrative of the submission and the defence in an audit. A decision
that is not in the log is a decision that will be remade, often differently, by the next person.

### 6.14 The one-page summary

A submission should have a one-page summary that a reviewer can read in five minutes: the claim,
the population, the input, the output, the evidence, the performance, the limitations, and the
post-market plan. If the summary cannot be written, the claim is not clear enough to submit.

### 6.15 The go/no-go question

Before submission, ask whether the evidence would survive a skeptical reviewer who has no
incentive to approve. If the answer is no, the submission is premature. The question is cheaper
than a deficiency letter.

## 7. Code, query, or calculation

The following code computes the standalone confidence interval, checks whether a proposed claim is
within a predicate's envelope, and estimates the sample size for a reader study. It uses only the
standard library and a small rule table.

```python
import math

def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

lo, hi = wilson(166, 180)
print("sensitivity        = {:.3f}".format(166 / 180))
print("Wilson 95% CI      = {:.3f} to {:.3f}".format(lo, hi))

predicate = {
    "modality": "CT",
    "finding": "solid lung nodule",
    "min_size_mm": 4,
    "thickness_min_mm": 1.0,
    "thickness_max_mm": 2.5,
    "role": "concurrent",
    "population": "adults 50-80 with smoking history",
}
proposed = {
    "modality": "CT",
    "finding": "solid lung nodule",
    "min_size_mm": 4,
    "thickness_min_mm": 1.0,
    "thickness_max_mm": 1.5,
    "role": "concurrent",
    "population": "adults 50-80 with smoking history",
}

def envelope_check(pred, prop):
    issues = []
    if prop["modality"] != pred["modality"]:
        issues.append("modality differs")
    if prop["finding"] != pred["finding"]:
        issues.append("finding differs")
    if prop["min_size_mm"] < pred["min_size_mm"]:
        issues.append("size range expands below predicate")
    if prop["thickness_min_mm"] < pred["thickness_min_mm"]:
        issues.append("thickness range expands below predicate")
    if prop["thickness_max_mm"] > pred["thickness_max_mm"]:
        issues.append("thickness range expands above predicate")
    if prop["role"] != pred["role"]:
        issues.append("workflow role differs")
    return issues

issues = envelope_check(predicate, proposed)
print("envelope issues    =", issues if issues else "none")

# Sample size for a proportion at sensitivity 0.90, half-width 0.05
p, e, z = 0.90, 0.05, 1.96
n = z * z * p * (1 - p) / (e * e)
print("required positives = {:.0f}".format(math.ceil(n)))

# Reader study: 8 readers, 200 cases, paired AUC difference
# Standard error approximated from an assumed reader variance
delta, se = 0.05, 0.015
print("reader delta CI    = {:.3f} to {:.3f}".format(delta - 1.96 * se, delta + 1.96 * se))
```

Expected output:

```text
sensitivity        = 0.922
Wilson 95% CI      = 0.874 to 0.953
envelope issues    = none
required positives = 139
reader delta CI    = 0.021 to 0.079
```

The code assumes: a simple predicate envelope, no population expansion, and a normal
approximation for the reader-study difference. A real substantial-equivalence analysis is more
detailed and is reviewed by regulatory affairs, not by a script.

## 8. Data, standards, and regulatory context

### 8.1 United States

The FDA regulates medical devices under the Federal Food, Drug, and Cosmetic Act and 21 CFR. The
routes are 510(k), De Novo, and PMA. Imaging AI is usually Class II under 21 CFR 892.2080 (CADt) or
892.2090 (CADe/CADx). The special controls are binding; the guidance documents explain how to meet
them.

The FDA's list of AI-enabled devices recorded 1,614 authorizations through 29 June 2026, of which
about 1,230 (76%) were radiology devices. The agency's overview page states more than 1,600 as of
September 2026. The counts are updated quarterly and should be cited with the date.

### 8.2 Predetermined change control plans

The final guidance, "Marketing Submission Recommendations for a Predetermined Change Control Plan
for AI-Enabled Device Software Functions," was issued 18 August 2025 (originally 4 December 2024).
It applies to AI-enabled device software functions reviewed via 510(k), De Novo, or PMA. A PCCP
has three components: the description of modifications, the modification protocol, and the impact
assessment. The modification protocol covers data management, retraining, performance evaluation,
and update procedures.

The guidance is not a regulation, but it is the FDA's stated expectation. A submission without a
PCCP can still be cleared, but each model update may require a new submission.

### 8.3 Good machine learning practice

The FDA, Health Canada, and the UK MHRA jointly published good machine learning practice (GMLP)
principles in 2021. They cover the data, the model, the testing, the monitoring, and the
documentation. The principles are not binding, but they are the reference for a credible process.

### 8.4 European Union

The Medical Device Regulation (EU) 2017/745 requires a CE mark, a clinical evaluation, and
post-market surveillance. The classification rules put most imaging AI in Class IIa or IIb,
requiring a notified body.

The AI Act (EU) 2024/1689 adds obligations for high-risk AI systems. Medical devices that require
a notified body are high-risk under Article 6(1). The original high-risk date for Annex I products
was 2 August 2027; the Digital Omnibus regulation adopted in 2026 deferred it to 2 August 2028.
The MDR obligations are unaffected and apply on their own timeline.

### 8.5 Other jurisdictions

Health Canada, the UK MHRA, Japan's PMDA, and Australia's TGA each have their own routes. A
product intended for multiple markets needs a gap analysis, not a single submission copied across
borders. The common core is a quality system, a risk file, and a clinical evaluation; the
differences are in the classification and the evidence format.

### 8.6 What the standards do not cover

No standard defines the clinical threshold for a specific product, the acceptable performance, or
the non-inferiority margin. Those are clinical and regulatory judgments, negotiated with the
agency and supported by the evidence.

### 8.7 Reimbursement is not clearance

A cleared device is not automatically reimbursed. In the United States, payment depends on the
CPT code, the payer policy, and the site's billing pathway. Some imaging AI has dedicated codes;
much does not. A product with a strong clinical case and no payment pathway may still fail
commercially.

The regulatory team should engage the reimbursement team early. The evidence that supports
clearance may not be the evidence a payer wants; payers often ask for clinical utility and cost-
effectiveness, not only diagnostic accuracy.

### 8.8 Predicate lineage and the "predicate problem"

A predicate can be a device cleared years ago with a narrower claim. The new device may be
technologically superior but legally comparable only within the predicate's envelope. The result
is that the new device's label is constrained by an old device's label, a phenomenon sometimes
called the predicate problem.

The strategic response is to decide whether to stay within the predicate envelope or to seek a De
Novo for a broader claim. The first is faster and narrower; the second is slower and potentially
more valuable.

### 8.9 Liability and the standard of care

A cleared device does not transfer responsibility from the clinician to the vendor. The clinician
remains responsible for the interpretation, and the vendor is responsible for the device's
performance and labeling. The split matters in litigation and in hospital contracting.

The labeling should state the intended role and the limitations clearly enough that the clinician
can use the device within the standard of care. A label that implies autonomy without the evidence
creates liability for both parties.

### 8.10 Standards for the quality system

ISO 13485 defines the quality management system; IEC 62304 defines the software lifecycle; ISO
14971 defines risk management. The three are interdependent: a software change is a risk input, a
risk control is a design output, and a design output is a test requirement. A QMS that treats them
as separate documents is not a system.

## 9. Industry practice and case evidence

### 9.1 IDx-DR

IDx-DR received De Novo authorization DEN180001 on 12 January 2018, the first FDA-authorized
autonomous AI diagnostic system. The pivotal study enrolled 900 patients at 10 primary care sites;
819 were analyzable. Observed sensitivity for more-than-mild diabetic retinopathy was 87.4%
(173/198, 95% CI 81.9-92.9%), observed specificity 89.5% (556/621, 95% CI 86.9-93.1%), and
imageability 96.1% (819/852). The reference standard was reading-centre grading of widefield
stereoscopic photographs plus optical coherence tomography.

The case shows that an autonomous claim is possible with a narrow intended use, a fixed capture
device, a pre-specified endpoint, and a workflow that routes ungradable images to a human. It does
not show that the same approach transfers to CT or MRI.

### 9.2 Viz.AI Contact

Viz.AI Contact received De Novo authorization DEN170073 in February 2018, creating 21 CFR 892.2080
for computer-assisted triage and notification. The device analyzes CTA for suspected large-vessel
occlusion and notifies a specialist; it is not for diagnosis. The classification allowed later
devices to use 510(k) with a predicate.

The case shows that the workflow role determines the regulatory category. A notification device is
not a diagnostic device, even when the underlying model detects the finding.

### 9.3 The CADe/CADx special controls

The final rule at 21 CFR 892.2090 was published on 13 June 2025, effective the same date. It
codified a classification that had been created by the OsteoDetect De Novo in 2018 and made the
special controls explicit: algorithm description, pre-specified testing, standalone performance,
localization accuracy, reader study, software documentation, and labeling with subgroup analyses.

The practical effect is that a CADe submission needs a reader study when the claim is assisted
read, and the reader population must match the intended user. A standalone AUC alone is not
enough.

### 9.4 Inference, not fact

The following are inferences: that a specific predicate will support a specific new claim; that a
specific reader-study design will satisfy a reviewer; that a specific PCCP will be accepted. Each
depends on the details and on the agency's judgment. The documented facts are the regulations, the
guidance dates, and the cleared device summaries.

### 9.5 Lessons from cleared devices

Three patterns recur in cleared imaging AI. First, the successful devices have narrow intended
uses and a clear workflow role. Second, the evidence matches the claim: a triage device has
workflow evidence, a detection device has reader evidence. Third, the labeling states the input
envelope, including the acquisition and the population.

The failures are equally patterned: a broad claim with a narrow study, a reader study with the
wrong readers, a label that omits the protocol, and a model update that was treated as a bug fix.

### 10.2 A submission readiness checklist

1. The intended use and indications are specific and consistent.
2. The pathway and predicate are documented.
3. The evidence matrix has no empty cells for claimed elements.
4. The standalone and reader studies are pre-specified.
5. The labeling states the input envelope and the performance.
6. The QMS covers the training data and model selection.
7. The cybersecurity file is complete.
8. The PCCP is in scope or absent.
9. The post-market plan is funded.
10. The EU gap analysis is current.

A submission that fails any of these is not ready, regardless of the model's AUC.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Vague intended use | Claim says "AI for chest CT" | No population or finding | Unclear pathway | Write specific use |
| Predicate mismatch | Different thickness or population | Substantial equivalence breaks | Deficiency | Choose right predicate |
| Missing reader study | Standalone only | Assisted-read claim unsupported | Deficiency | Add MRMC |
| Labeling omission | Inclusion criteria not in label | Off-label use | Safety and liability | Map criteria to label |
| No PCCP | Annual retraining planned | Each update is a new device | Repeated submissions | Draft PCCP |
| QMS gap | Training outside design control | Uncontrolled change | Audit finding | Bring under QMS |
| Cybersecurity gap | No threat model | Network exposure | Vulnerability | Add controls |
| Autonomous claim with retrospective data | No prospective study | Evidence mismatch | Rejection | Prospective design |
| Subgroup missing | Pooled only | Equity and safety risk | Deficiency | Pre-specify subgroups |
| EU timeline error | Plan assumes 2027 | AI Act deferral | Delay or non-compliance | Track 2028 |
| Predicate age | Cleared 10 years ago | Technology difference | Review questions | Address differences |
| Off-label expansion | New modality without submission | Claim creep | Enforcement risk | New submission |

### 10.1 Red flags that stop a submission

1. The indications for use are broader than the evidence.
2. The predicate has a different acquisition or population.
3. The labeling does not state the validated input envelope.
4. The reader study readers do not match the intended users.
5. The PCCP includes a new intended use.
6. The QMS does not cover the training data and model selection.
7. The cybersecurity risk assessment is missing.

### 10.3 The submission narrative

A submission is an argument, not a data dump. The narrative states the claim, the predicate, the
evidence, the labeling, and the residual risk. A reviewer who can follow the narrative can find
the evidence; a reviewer who cannot will ask for it, and the clock restarts.

### 10.4 The most common late discovery

The most common late discovery is that the evidence does not cover the population in the label.
The second is that the reader study used the wrong readers. Both are design errors, not writing
errors, and both are cheaper to fix before the data are collected.

### 10.5 What a good regulatory strategy looks like

A good strategy is written on one page, updated at every design review, and owned by a named
person. It states the claim, the pathway, the predicate or De Novo rationale, the evidence, the
labeling, the change-control plan, and the post-market plan. It is boring, which is the point.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The intended use is specific | Pathway and evidence | Vague claim | Review with regulatory |
| The predicate is comparable | 510(k) route | Different input or use | Side-by-side table |
| The evidence matches the claim | Clearance | Standalone only | Protocol review |
| The labeling states the envelope | Safe use | Omission | Criteria-to-label map |
| The QMS covers the model | Design control | Training outside QMS | Audit |
| The PCCP is within scope | Update without submission | New population | Scope review |
| The EU classification is correct | CE mark | Wrong class | Notified-body consult |
| The AI Act timeline is current | Compliance plan | Deferral | Track official sources |
| The post-market plan is funded | Surveillance | No budget | Plan review |
| The clinical threshold is justified | Claim | No clinical input | Clinical advisory |

### 11.1 Limits of this topic

This topic is educational and is not legal advice. It covers the main FDA and EU pathways for
imaging AI, not the full regulatory law, the state-level requirements, or the reimbursement
pathway. A real submission requires regulatory affairs, quality, clinical, and legal review.

### 11.2 What would change the conclusion

If the predicate's labeling covered 1.0-1.5 mm CT, the substantial-equivalence argument would be
stronger. If the claim were limited to 2.5 mm, a different predicate might apply. If the product
were intended only for research use, the regulatory pathway would be different, but the clinical
claim would be absent.

## 12. Exercises

### Exercise 1. Intended use (recall, 20 minutes)

**Deliverable:** A one-paragraph intended use and indications for use for a fracture detector on
radiographs.

**Acceptance criteria:** Population, anatomy, modality, finding, workflow role, and output.

**Time:** 20 minutes.

### Exercise 2. Pathway selection (application, 30 minutes)

**Deliverable:** For three product concepts, select 510(k), De Novo, or PMA and justify.

**Acceptance criteria:** Correct pathway per concept; predicate analysis; risk rationale.

**Time:** 30 minutes.

### Exercise 3. Predicate comparison (application, 35 minutes)

**Deliverable:** A side-by-side table of a proposed device and a predicate, with the differences
and the equivalence argument.

**Acceptance criteria:** At least eight attributes; differences flagged; conclusion stated.

**Time:** 35 minutes.

### Exercise 4. Sample size (calculation, 25 minutes)

**Deliverable:** For sensitivity 0.90 and a half-width of 0.04, compute the number of positive
lesions; then for 0.03.

**Acceptance criteria:** Correct values (217 and 385); formula shown.

**Time:** 25 minutes.

### Exercise 5. PCCP outline (design, 45 minutes)

**Deliverable:** A PCCP outline for a model that will be retrained annually on new scanners, with
the three components.

**Acceptance criteria:** Description, protocol, impact assessment; scope limits; acceptance
criteria.

**Time:** 45 minutes.

### Exercise 6. Labeling review (critique, 40 minutes)

**Deliverable:** A review of a draft label that omits slice thickness, population, and reader
study results.

**Acceptance criteria:** Identifies three omissions; explains the risk; proposes language.

**Time:** 40 minutes.

### Exercise 7. EU gap analysis (design, 45 minutes)

**Deliverable:** A table mapping MDR and AI Act obligations to evidence and owner.

**Acceptance criteria:** Classification, notified body, clinical evaluation, post-market, AI Act
timeline.

**Time:** 45 minutes.

### Exercise 8. Post-market plan (design, 40 minutes)

**Deliverable:** A plan with complaints, performance monitoring, drift triggers, and reporting.

**Acceptance criteria:** At least five metrics; trigger thresholds; owner and cadence.

**Time:** 40 minutes.

### Exercise 9. Evidence matrix (application, 35 minutes)

**Deliverable:** An evidence-to-claim matrix for a triage device, with each claim element mapped to
a study and a labeling statement.

**Acceptance criteria:** No empty cells for claimed elements; at least five claim elements; gaps
identified.

**Time:** 35 minutes.

### Exercise 10. Timeline (calculation, 25 minutes)

**Deliverable:** A 12-month plan for a 510(k) with standalone and reader studies, including the
dependencies and the critical path.

**Acceptance criteria:** Milestones; dependencies; at least one risk and mitigation.

**Time:** 25 minutes.

## 13. Self-check answers

**A1. What is the difference between intended use and indications for use?**
Intended use is the general purpose; indications for use specify the population, conditions, and
anatomy. The indications are the narrower, testable statement.

**A2. When is a 510(k) appropriate?**
When a legally marketed predicate has the same intended use and comparable technology, and the
differences do not raise new safety or effectiveness questions.

**A3. What is a De Novo?**
A classification route for a novel device of low or moderate risk that has no predicate. It creates
a new classification and can be a predicate for later devices.

**A4. Sample size for sensitivity 0.90, half-width 0.04 and 0.03.**
n = 1.96² × 0.9 × 0.1 / 0.04² = 216.1, so 217. For 0.03, 384.2, so 385.

**A5. What are the three PCCP components?**
Description of modifications, modification protocol (data management, retraining, evaluation,
update procedures), and impact assessment.

**A6. Why does the workflow role change the pathway?**
The role changes the consequence of an error and the human review. A triage device with review is
lower risk than an autonomous diagnostic device.

**A7. What must the labeling state for a CT model?**
The population, the finding, the intended user, the compatible scanner and protocol, the slice
thickness range, the contrast phase, the performance with subgroups, and the warnings.

**A8. What changed in the EU AI Act timeline?**
The high-risk obligations for Annex I products, including medical devices, were deferred from
2 August 2027 to 2 August 2028 by the Digital Omnibus regulation adopted in 2026. The MDR
obligations are separate.

**A9. Design rubric for Exercise 5.** A strong PCCP specifies the data sources, the retraining
trigger, the locked evaluation, the acceptance criteria, and the rollback. A weak PCCP says "we
will retrain when needed".

**A10. Design rubric for Exercise 8.** A strong plan has metrics, thresholds, owners, cadence, and
a reporting path. A weak plan lists complaint handling only.

## 14. Interview questions

**Q1. How do you choose a regulatory pathway?**
A strong answer starts from the intended use, the risk, and the predicate, and maps to 510(k), De
Novo, or PMA. A weak answer says "510(k) because everyone does".

**Q2. What is substantial equivalence?**
A strong answer compares intended use, technology, and performance, and identifies differences
that could raise new questions. A weak answer says "similar to a cleared device".

**Q3. When is a reader study required?**
A strong answer says when the claim is assisted read or diagnosis, and describes the MRMC design
and the reader population. A weak answer says always or never.

**Q4. What is a PCCP?**
A strong answer describes the three components and the scope limit: no new intended use. A weak
answer calls it a post-market plan.

**Q5. How does the workflow role affect risk?**
A strong answer explains human review and consequence, and gives triage vs autonomous examples. A
weak answer says the model is the same.

**Q6. What are the special controls for CADe?**
A strong answer lists algorithm description, pre-specified testing, standalone and reader evidence,
software documentation, and labeling with subgroups. A weak answer says "clinical evidence".

**Q7. What does the labeling need to state?**
A strong answer covers population, inputs, protocol, performance, warnings, and limitations. A weak
answer says "instructions for use".

**Q8. How do you handle a model update after clearance?**
A strong answer uses a PCCP if the change is in scope, otherwise a new submission. A weak answer
says updates are covered by the original clearance.

**Q9. What is the EU AI Act timeline for medical devices?**
A strong answer states the 2 August 2028 date after the 2026 deferral, and distinguishes it from
MDR. A weak answer says 2027 or confuses the two regimes.

**Q10. What is the biggest regulatory risk in an imaging AI project?**
A strong answer says the mismatch between the claim and the evidence, often discovered late. A weak
answer says the FDA is slow.

## 15. Key terms

- **510(k)** — premarket notification with substantial equivalence.
- **AI Act** — EU Regulation 2024/1689.
- **CADe** — computer-assisted detection.
- **CADt** — computer-assisted triage and notification.
- **CADx** — computer-assisted diagnosis.
- **CE mark** — EU conformity mark.
- **Class I/II/III** — FDA risk classes.
- **Clinical evaluation** — MDR evidence of safety and performance.
- **De Novo** — classification route for novel moderate-risk devices.
- **Device** — regulated article for diagnosis or treatment.
- **GMLP** — good machine learning practice.
- **IEC 62304** — medical device software lifecycle standard.
- **Indications for use** — specific population and conditions.
- **Intended use** — general purpose of the device.
- **ISO 13485** — medical device quality management standard.
- **ISO 14971** — risk management standard.
- **MDR** — EU Medical Device Regulation 2017/745.
- **Notified body** — EU conformity assessment organization.
- **PCCP** — predetermined change control plan.
- **PMA** — premarket approval.
- **Predicate** — legally marketed comparison device.
- **QMS** — quality management system.
- **SaMD** — software as a medical device.
- **SiMD** — software in a medical device.
- **Special controls** — device-type-specific requirements.
- **Substantial equivalence** — similarity finding for 510(k).
- **UDI** — unique device identification.
- **Post-market surveillance** — ongoing monitoring after launch.

## 16. Sources

- U.S. Food and Drug Administration, 21 CFR 892.2090, Radiological computer-assisted detection and
  diagnosis software; final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2080, Radiological computer aided triage and
  notification software. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2080
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, "Marketing Submission Recommendations for a Predetermined
  Change Control Plan for Artificial Intelligence-Enabled Device Software Functions," final
  guidance, 18 August 2025 (originally 4 December 2024).
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, "Clinical Performance Assessment: Considerations for
  Computer-Assisted Detection Devices Applied to Radiology Images and Radiology Device Data in
  Premarket Notification (510(k)) Submissions," final guidance, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-performance-assessment-considerations-computer-assisted-detection-devices-applied-radiology
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, List of Artificial Intelligence-Enabled Medical Devices.
  https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
  (accessed 25 October 2026). Counts through 29 June 2026.
- U.S. Food and Drug Administration, De Novo DEN180001 (IDx-DR), decision summary.
  https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf (accessed 25 October 2026).
- U.S. Food and Drug Administration, De Novo DEN170073 (Viz.AI Contact).
  https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170073.pdf (accessed 25 October 2026).
- European Union, Regulation (EU) 2017/745 on medical devices.
  https://eur-lex.europa.eu/eli/reg/2017/745/oj (accessed 25 October 2026).
- European Union, Regulation (EU) 2024/1689 laying down harmonised rules on artificial
  intelligence. https://eur-lex.europa.eu/eli/reg/2024/1689/oj (accessed 25 October 2026).
- European Commission, AI Act timeline and implementation.
  https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
  (accessed 25 October 2026).
- International Medical Device Regulators Forum, Good Machine Learning Practice principles (2021).
  https://www.imdrf.org/ (accessed 25 October 2026).
- ISO 13485:2016, Medical devices — Quality management systems.
- ISO 14971:2019, Medical devices — Application of risk management.
- IEC 62304:2006/AMD1:2015, Medical device software — Software life cycle processes.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_workflow-integration.md`](08_workflow-integration.md)
- Next topic: [`10_foundation-models.md`](10_foundation-models.md)
- Related topic: [`07_evaluation.md`](07_evaluation.md)
- Root survey: [`../../10_medical_imaging.md`](../../course/10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)

> **Note:** This topic is educational and is not legal advice.
> The regulatory facts are dated; verify the current guidance before submission.
> A real submission requires regulatory affairs, quality, clinical, and legal review.
> Verify the current edition of every cited regulation.
