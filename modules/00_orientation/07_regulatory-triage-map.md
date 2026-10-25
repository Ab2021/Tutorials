# Regulatory Triage Map

> Module 00 — Orientation and Industry Map
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02, 05
> Version 2026.3

## 1. Why this topic exists

Regulatory scope is the most expensive thing a biomedical data team can discover late. A model
that was built as an internal quality improvement project becomes a regulated device when the
marketing team describes it as a diagnostic aid. A research prototype becomes a clinical trial
system when it is used to allocate treatment. A de-identified dataset becomes a compliance
problem when someone joins it to a public registry.

None of these transitions is exotic. Each is a normal consequence of a product finding a market.
The cost of retrofitting a quality management system, a clinical evidence package, or a privacy
programme onto a finished product is measured in months and sometimes in the product's survival.

Regulatory triage is the week-one activity that prevents this. It is not a legal opinion. It is a
structured set of questions that determines which regimes may apply, which evidence each requires,
and who inside the organization must own the answer. The output is a one-page memo that a
regulatory affairs professional can act on.

This topic gives you that map. It covers the US Food and Drug Administration (FDA) device and
clinical decision support (CDS) framework, the EU Medical Device Regulation (MDR) and In Vitro
Diagnostic Regulation (IVDR), the EU Artificial Intelligence Act, US privacy law under HIPAA, EU
data protection under GDPR, laboratory regulation under CLIA, and the quality standards that
attach to regulated software. It is educational orientation, not legal advice.

### 1.1 The four questions that determine everything

Most triage reduces to four questions:

1. Does the product make a claim about diagnosis, treatment, prevention, or safety for a specific
   patient?
2. Does the output inform a decision that a clinician or patient acts on, and can the basis be
   independently reviewed?
3. Does the product process identifiable health data, and under whose legal authority?
4. Is the product placed on the market, used in a trial, or used only internally?

The answers determine the regime. A product that makes no medical claim, processes no
identifiable data, and is used only internally is largely unregulated by device law. Change any
one answer and the map changes.

### 1.2 Why teams get it wrong

Three patterns.

**Confusing the technology with the claim.** A large language model is not a device. A large
language model that recommends a diagnosis is. The regime follows the intended use, not the
architecture.

**Confusing clearance with evidence.** A 510(k) clearance is a finding of substantial equivalence
to a predicate device. It is not a finding that the product improves outcomes. Teams sometimes
treat clearance as the end of evidence generation when it is the beginning of post-market
obligations.

**Confusing internal use with no regulation.** An internal tool used in patient care can still be
a device if it meets the definition, even if it is never sold. The FDA's device definition turns
on intended use, not on commerce. Enforcement discretion may apply, but discretion is not
exemption.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Classify a product as research, QI, CDS, SaMD, or IVD using a documented rationale | Exercise 1; self-check 2 |
| 2 | Apply the FDA CDS criteria to a clinical software product | Exercise 2; self-check 3 |
| 3 | Identify the US premarket pathway and the evidence each requires | Exercise 3 |
| 4 | Map an EU product to MDR or IVDR class and the conformity route | Exercise 4; self-check 5 |
| 5 | Determine whether the EU AI Act adds obligations and on what timeline | Exercise 5 |
| 6 | Identify the privacy regime, legal basis, and contract artifacts for a data flow | Exercise 6 |
| 7 | Estimate the time and cost of a regulatory pathway with explicit assumptions | Exercise 7 |
| 8 | Write a one-page triage memo that names owners and open questions | Exercise 8 |

## 3. The core mental model

Triage is a decision tree with branches that can run in parallel. The output is not a single
label but a set of applicable regimes, each with an owner and an evidence requirement.

### 3.1 The triage tree

```text
START: What does the product claim, for whom, in what setting?
   |
   +-- No medical claim, general wellness only?
   |      |
   |      +-- Consumer app, no diagnosis/treatment claim
   |             -> FTC health breach rule if health data
   |             -> state privacy law; GDPR if EU
   |             -> not a device (confirm claim discipline)
   |
   +-- Medical claim?
          |
          +-- Is it software?
                 |
                 +-- Does it meet the device definition (intended use)?
                        |
                        +-- NO -> internal QI / operations
                        |        -> HIPAA if PHI; IRB if research
                        |
                        +-- YES -> device
                              |
                              +-- US: which pathway?
                              |    510(k) if predicate and substantial equivalence
                              |    De Novo if novel, low-moderate risk, no predicate
                              |    PMA if high risk
                              |    CDS exclusion analysis first
                              |
                              +-- EU: MDR or IVDR?
                              |    MDR if device; IVDR if in vitro diagnostic
                              |    Class I / IIa / IIb / III -> conformity route
                              |
                              +-- AI Act?
                              |    High-risk if Annex I product or Annex III use
                              |    Transparency obligations if limited risk
                              |
                              +-- Clinical trial use?
                                    -> IDE (US) / 21 CFR 812
                                    -> ICH E6 GCP; 21 CFR Part 11
```

### 3.2 The two-axis map

```text
                    CLAIM STRENGTH (high)
                             |
   Research / QI             |   Regulated device
   (no product claim)        |   (diagnosis, treatment)
                             |
   INTERNAL ---------------+--------------- MARKETED
                             |
   Operations analytics      |   Consumer wellness
   (scheduling, supply)      |   (no medical claim)
                             |
                    CLAIM STRENGTH (low)
```

The horizontal axis is distribution: internal use versus marketed product. The vertical axis is
claim strength. The upper-right quadrant carries the heaviest obligations. The lower-left carries
the lightest, but privacy law applies in all four quadrants if identifiable data are processed.

### 3.3 Claim determines regime, not algorithm

| Product description | Claim | Likely regime |
|---|---|---|
| Sepsis risk score shown to nurses | Supports a review decision | CDS analysis; possibly device |
| Sepsis risk score that orders antibiotics | Treatment recommendation | Device |
| Documentation summarizer | Administrative summarization | Not a device if no medical claim |
| Documentation summarizer that suggests diagnoses | Diagnosis support | Device |
| Imaging triage that orders studies | Triage and prioritization | Device |
| Imaging quality check | Technical quality only | Usually not a device |
| Consumer AFib notification | Possible medical claim | Device if claim is diagnostic |
| Hospital capacity forecast | Operational | Not a device |

The table is a starting point, not a classification. Each row requires the analysis in section 5.

### 3.4 The obligations stack

A regulated product does not face one regulator. It faces a stack.

```text
   +-------------------------------------------+
   |  Payment and coverage (CMS, payers)       |
   +-------------------------------------------+
   |  AI-specific obligations (EU AI Act)      |
   +-------------------------------------------+
   |  Privacy and security (HIPAA, GDPR)       |
   +-------------------------------------------+
   |  Device regulation (FDA, MDR/IVDR)        |
   +-------------------------------------------+
   |  Quality system (ISO 13485, 21 CFR 820)   |
   +-------------------------------------------+
   |  Software lifecycle (IEC 62304)           |
   +-------------------------------------------+
   |  Risk management (ISO 14971)              |
   +-------------------------------------------+
```

The lower layers are prerequisites for the upper ones. A submission cannot be supported without a
quality system, and a quality system cannot be maintained without risk management. Teams that
start at the top of the stack discover the lower layers during review.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Medical device | Article intended for diagnosis, treatment, prevention, or mitigation of disease | Applied to any health software | Legal category |
| SaMD | Software as a Medical Device; software with a medical purpose not part of a hardware device | Applied to all clinical software | Regulatory category |
| SiMD | Software in a Medical Device; embedded software that is part of a hardware device | Confused with SaMD | Regulatory category |
| Intended use | Regulated statement of what the product does, for whom, in what setting | Written as marketing copy | Regulatory statement |
| Indications for use | Specific clinical circumstances in which a device is intended for use | Used interchangeably with intended use | Regulatory statement |
| 510(k) | Premarket notification demonstrating substantial equivalence to a predicate | Called "approval" | FDA pathway |
| Predicate | Legally marketed device to which equivalence is claimed | Assumed to exist for novel products | Device reference |
| Substantial equivalence | Same intended use and similar technological characteristics, or different characteristics with equivalent safety and effectiveness | Treated as proof of effectiveness | Regulatory finding |
| De Novo | Pathway for novel devices of low to moderate risk with no predicate | Confused with 510(k) | FDA pathway |
| PMA | Premarket approval for high-risk devices | Applied to any device | FDA pathway |
| Class I/II/III | US risk classes; III is highest | Confused with EU classes | Classification |
| CDS | Clinical decision support; software that helps clinicians decide | Assumed exempt | Regulatory category |
| CDS criteria | Four criteria in FDA's 2022 guidance for CDS outside the device definition | Applied without reading the guidance | Guidance criteria |
| PCCP | Predetermined change control plan for authorized model updates | Assumed to allow arbitrary retraining | Regulatory mechanism |
| GMLP | Good machine learning practice principles | Treated as binding law | Guidance |
| MDR | EU Medical Device Regulation 2017/745, applied from May 2021 | Confused with the US FD&C Act | Regulation |
| IVDR | EU In Vitro Diagnostic Regulation 2017/746, applied from May 2022 | Confused with MDR | Regulation |
| Notified body | Independent EU organization assessing conformity | Confused with a regulator | EU entity |
| CE marking | EU conformity mark | Called "EU approval" | Conformity mark |
| Class I/IIa/IIb/III | EU device risk classes | Mapped directly onto US classes | Classification |
| IVD | In vitro diagnostic; a test on a specimen | Confused with a device generally | Product category |
| Companion diagnostic | Test required to select patients for a therapy | Confused with any biomarker test | Regulatory category |
| RUO | Research use only; labeling that limits use to research | Assumed to permit clinical use | Labeling |
| IUO | Investigational use only | Confused with RUO | Labeling |
| CLIA | US Clinical Laboratory Improvement Amendments, 42 CFR 493 | Confused with FDA clearance | Regulation |
| LDT | Laboratory-developed test; designed, manufactured, and used in one laboratory | Assumed to be FDA-regulated | Category |
| CAP | College of American Pathologists accreditation | Confused with CLIA | Accreditation |
| HIPAA | US health privacy and security law | Applied to all health data | Regulation |
| PHI | Protected health information | Equated with all health data | Legal category |
| Covered entity | Provider, plan, or clearinghouse subject to HIPAA | Applied to app developers | Legal category |
| Business associate | Vendor handling PHI for a covered entity | Assumed optional | Legal category |
| BAA | Business associate agreement | Confused with a DPA | Contract |
| Minimum necessary | HIPAA principle limiting PHI use to what is needed | Applied only to humans | Standard |
| De-identification | Removal of identifiers under Safe Harbor or Expert Determination | Assumed to eliminate risk | Legal standard |
| GDPR | EU General Data Protection Regulation 2016/679 | Applied to anonymous data | Regulation |
| Controller | Entity determining purposes and means of processing | Confused with processor | Legal role |
| Processor | Entity processing data on behalf of a controller | Confused with controller | Legal role |
| DPIA | Data protection impact assessment under GDPR Article 35 | Confused with a security review | Assessment |
| AI Act | Regulation (EU) 2024/1689, as amended in 2026 | Assumed to replace MDR | Regulation |
| High-risk AI system | AI system subject to the AI Act's high-risk obligations | Assumed to mean clinically high risk | Regulatory category |
| FRIA | Fundamental rights impact assessment under the AI Act | Confused with a DPIA | Assessment |
| GxP | Good practice regulations for labs, trials, manufacturing | Used for any regulated activity | Regulatory family |
| CSV | Computer system validation | Confused with software testing | Process |
| 21 CFR Part 11 | US regulation on electronic records and signatures | Assumed to apply to all software | Regulation |
| ICH E6 | Good clinical practice guideline | Confused with GxP broadly | Guideline |
| ISO 13485 | Quality management system standard for medical devices | Treated as optional | Standard |
| IEC 62304 | Medical device software lifecycle standard | Confused with ISO 13485 | Standard |
| ISO 14971 | Medical device risk management standard | Confused with a security risk assessment | Standard |
| Post-market surveillance | Ongoing monitoring of a device after market entry | Confused with customer support | Process |
| PMCF | Post-market clinical follow-up under MDR | Confused with a new trial | Process |
| UDI | Unique Device Identification | Confused with a serial number | Identifier |
| HTI-1 | US health IT certification rule with AI transparency provisions | Confused with FDA regulation | Rule |
| Enforcement discretion | Regulator's choice not to enforce a requirement | Treated as legal exemption | Policy |

Three definitions deserve expansion.

**Intended use.** The intended use is the single most important sentence in a regulatory file. It
determines device status, the predicate, the clinical evidence, and the claims that may be made.
A change to intended use can change the pathway entirely.

**Enforcement discretion.** The FDA may choose not to enforce certain requirements for certain
products, for example some CDS or some laboratory tests. Discretion is a policy position, not a
statutory exemption. It can change with guidance or enforcement priorities, and it does not bind
other regulators or private litigants.

**Substantial equivalence.** Substantial equivalence is not a finding that the device works. It
is a finding that the device is as safe and effective as a predicate with the same intended use.
The predicate itself may be old and may never have been evaluated against modern standards.

## 5. The mechanics, step by step

This section is a fourteen-step triage procedure. It produces a memo, not a decision. The memo
names the regimes, the evidence, the owners, and the open questions for regulatory affairs and
counsel.

### Step 1: Write the intended use in one paragraph

**Input:** product description.
**Output:** function, population, setting, user, input, output, and exclusions.

**Reasoning.** Every later step depends on this paragraph. Write it before any classification.
If the team cannot agree on the paragraph, the product has not been defined.

**Branch point.** If the paragraph contains a diagnostic, treatment, or prevention claim, assume
device status until proven otherwise. If it contains only administrative or operational language,
the device question may not arise, but privacy still does.

### Step 2: Separate the claim from the feature list

**Input:** intended use and marketing materials.
**Output:** a table of claims, each marked as medical or non-medical.

**Reasoning.** Marketing materials are part of the evidence of intended use. A feature list that
says "detects pneumonia" creates a claim even if the intended use statement is cautious. The
triage must cover all public statements, including the website, the sales deck, and the app store
description.

**Branch point.** If the marketing claims exceed the intended use, stop and fix the claims before
continuing. A claim that is not supported by the regulatory file is a liability.

### Step 3: Determine whether the product is software with a medical purpose

**Input:** intended use paragraph.
**Output:** a yes/no judgement with rationale.

**Reasoning.** In the US, the device definition turns on intended use. Software that is intended
for diagnosis, treatment, prevention, or mitigation of disease is a device unless an exclusion
applies. In the EU, software that performs a medical purpose is a medical device under MDR
Article 2.

**Branch point.** If the product is administrative, operational, or purely informational, document
the rationale and proceed to privacy. If it is medical, continue.

### Step 4: Apply the FDA CDS exclusion criteria

**Input:** intended use and output design.
**Output:** a criterion-by-criterion analysis with a conclusion.

**Reasoning.** The FDA's 2022 final guidance describes four criteria that must all be met for CDS
to fall outside the device definition. The criteria concern whether the software displays,
analyzes, or prints medical information; whether the basis for the recommendation is independent
of the software; whether the clinician can independently review the basis; and whether the
software provides a specific directive rather than a recommendation. If any criterion fails, the
software is likely a device.

**Branch point.** If the product provides a specific treatment directive or the clinician cannot
review the basis, assume device status. If all criteria are met, document the analysis and
proceed to privacy, but note that state law or institutional policy may impose additional
requirements.

**Practical note.** The criteria are about the basis of the recommendation, not about the
sophistication of the model. A complex model can meet the criteria if the underlying evidence is
accessible to the clinician. A simple rule can fail them if it directs a specific action.

### Step 5: Select the US premarket pathway

**Input:** device status and risk.
**Output:** pathway recommendation with rationale.

**Reasoning.** The pathway determines evidence, timeline, and cost. A 510(k) requires a predicate
and substantial equivalence. De Novo is for novel devices of low to moderate risk without a
predicate. PMA is for high-risk devices and requires the most evidence.

**Branch point.** If no predicate exists and the device is low to moderate risk, De Novo is the
route. If the device is high risk, PMA. If a predicate exists and equivalence can be shown,
510(k).

**Evidence types.** Bench testing, software verification and validation, analytical validation,
clinical validation, human factors, cybersecurity, and labeling. The mix depends on the device
and the pathway.

### Step 6: Map the EU route

**Input:** device status and intended purpose.
**Output:** MDR or IVDR classification and conformity route.

**Reasoning.** The EU class determines whether a notified body is required. Class I devices can
often self-certify, with exceptions for sterile, measuring, and reusable surgical instruments.
Class IIa, IIb, and III require a notified body. IVDs have their own classification under IVDR.

**Branch point.** If the product is an IVD, IVDR applies, not MDR. If the product is a medical
device with software, MDR applies and the software classification rules in Annex VIII determine
the class.

**Timeline reality.** Notified body certification is slow. The European Commission's notified
body surveys report that a majority of notified bodies take 13 to 18 months to issue a new MDR
quality management system certificate, and a substantial minority take 19 to 24 months for a
combined quality management system and product certificate.

### Step 7: Determine AI Act applicability

**Input:** product function and EU market presence.
**Output:** risk tier and obligations.

**Reasoning.** The AI Act applies to AI systems placed on the EU market or used in the EU. A
medical device that is also a high-risk AI system must meet both MDR and the AI Act. The AI Act's
high-risk obligations were deferred by the Digital Omnibus on AI, which entered into force on
27 July 2026: stand-alone Annex III systems from 2 December 2027, and Annex I products, including
medical devices, from 2 August 2028.

**Branch point.** If the product is a medical device under MDR or IVDR, it is likely an Annex I
high-risk AI system. Plan for both regimes, even though the AI Act dates are later.

### Step 8: Determine privacy and data protection obligations

**Input:** data flow.
**Output:** regimes, legal basis, contracts, and assessments.

**Reasoning.** In the US, HIPAA applies to covered entities and business associates. A vendor
handling PHI needs a BAA. In the EU, GDPR applies to personal data, with additional conditions
for health data under Article 9. A DPIA may be required for high-risk processing.

**Branch point.** If the product processes only de-identified data, HIPAA may not apply, but
re-identification risk and state law still matter. If the product is a consumer app not covered
by HIPAA, the FTC health breach notification rule may apply.

**Contract artifacts.** BAA for HIPAA, DPA for GDPR, DUA for research data, and a security
addendum for enterprise customers.

### Step 9: Determine laboratory regulation if applicable

**Input:** product type.
**Output:** CLIA and IVDR analysis.

**Reasoning.** A test performed on a specimen may be regulated as an IVD or as a laboratory-
developed test. In the US, the FDA's 2024 rule phasing out enforcement discretion for LDTs was
vacated by a federal court on 31 March 2025, and the FDA formally implemented the vacatur on
19 September 2025. The pre-2024 enforcement-discretion framework therefore governs.

**Branch point.** If the product is a laboratory test, CLIA certification and CAP accreditation
may apply alongside FDA requirements. If the product is software that analyzes an image but does
not perform a test, the analysis differs.

> **Note:** The LDT landscape has changed repeatedly. Confirm the current status with regulatory
> affairs before relying on any statement in a submission plan.

### Step 10: Determine trial-related obligations

**Input:** intended use of the study.
**Output:** IDE, IRB, GCP, and Part 11 analysis.

**Reasoning.** If the product is used in a clinical trial to evaluate safety or effectiveness,
the trial may require an investigational device exemption (IDE) in the US, IRB approval, and
compliance with ICH E6 good clinical practice. Electronic records used in the trial may be
subject to 21 CFR Part 11.

**Branch point.** If the study is a quality improvement evaluation not intended to support a
marketing claim, the obligations differ, but the determination should be documented.

### Step 11: Identify the quality system obligations

**Input:** device status and pathway.
**Output:** QMS scope, standards, and gap assessment.

**Reasoning.** A device manufacturer needs a quality management system. ISO 13485 is the
international standard; 21 CFR 820 is the US regulation. IEC 62304 governs the software
lifecycle, and ISO 14971 governs risk management. The QMS must exist before design freeze, not
before launch.

**Branch point.** If the organization has no QMS, the first regulatory milestone is a gap
assessment and a remediation plan. This is often the longest item on the critical path.

### Step 12: Build the evidence matrix

**Input:** pathway, claims, and risks.
**Output:** a matrix of claim, evidence type, study, status, and owner.

**Reasoning.** Each claim needs evidence. The matrix prevents the common failure of discovering a
missing study during submission review. It also shows which evidence is reusable across
jurisdictions and which is jurisdiction-specific.

**Branch point.** If a claim has no evidence and no feasible study, remove the claim. The fastest
way to reduce regulatory burden is to narrow the claim.

### Step 13: Estimate timeline and cost

**Input:** evidence matrix and QMS gap.
**Output:** a schedule with dependencies and a cost estimate with assumptions.

**Reasoning.** Regulatory work is the long pole in many product plans. Estimating it forces the
organization to decide whether the market justifies the investment. The estimate should be a
range, not a point.

**Branch point.** If the timeline exceeds the funding runway, the options are to narrow the
claim, delay the market entry, partner with a regulated entity, or stop.

### Step 14: Write the triage memo and assign owners

**Input:** all outputs.
**Output:** one page with regimes, conclusions, owners, open questions, and review date.

**Reasoning.** The memo is the artifact. It must be written so that regulatory affairs can
disagree with it productively. Each conclusion should cite the evidence or guidance it relies on.

**Branch point.** If a conclusion cannot be supported, mark it as an open question rather than
asserting it. An honest open question is more useful than a confident error.

### 5.1 Regime comparison

| Dimension | US FDA | EU MDR | EU IVDR | EU AI Act |
|---|---|---|---|---|
| Trigger | Intended use as a device | Medical purpose | In vitro diagnostic purpose | AI system on EU market |
| Classification | Class I, II, III | Class I, IIa, IIb, III | Class A, B, C, D | Prohibited, high, limited, minimal |
| Conformity | 510(k), De Novo, PMA | Self-certification or notified body | Notified body for most | Self-assessment or notified body |
| Clinical evidence | Varies by pathway | Clinical evaluation | Performance evaluation | Risk management, data governance |
| Post-market | MDR reporting, recalls | PMS, PMCF | PMS, PMPF | Post-market monitoring |
| Change control | PCCP for AI devices | Significant change assessment | Significant change assessment | Substantial modification rules |

### 5.2 What triggers a new submission

In the US, a change that significantly affects safety or effectiveness, or that changes intended
use, generally requires a new submission unless it falls within an authorized PCCP. In the EU, a
significant change requires notified body assessment. The practical rule is that changes to the
model, the population, the input data, or the output require a change assessment before release.

### 5.3 Common misclassifications

| Situation | Common belief | Actual issue |
|---|---|---|
| Internal tool only | "Not marketed, so not a device" | Device definition turns on intended use |
| Clinician in the loop | "CDS, so exempt" | Four criteria must all be met |
| Documentation assistant | "Administrative, so unregulated" | If it suggests diagnoses, it is a claim |
| Research prototype | "RUO, so no requirements" | Clinical use exceeds RUO labeling |
| De-identified data | "HIPAA does not apply" | Re-identification and state law still apply |
| CE marked in EU | "Can sell in US" | Separate US clearance required |
| 510(k) cleared | "Proven effective" | Clearance is substantial equivalence, not efficacy |

### 5.4 The triage memo template

The memo is one page. It is addressed to regulatory affairs and copied to the product owner. It
records conclusions and open questions, not advocacy.

```markdown
# Regulatory triage memo
Product: <name> | Version: <x.y> | Date: <yyyy-mm-dd> | Review due: <yyyy-mm-dd>
Author: <role> | Reviewer: <regulatory affairs>

### Intended use (verbatim)
<paragraph>

### Claims reviewed
<list of public claims and where they appear>

### Conclusions
| Question | Conclusion | Basis | Confidence |
|---|---|---|---|
| US device status | <yes/no/unclear> | <guidance or statute> | <high/medium/low> |
| US pathway | <510(k)/De Novo/PMA/NA> | <predicate or rationale> | |
| EU classification | <class or NA> | <Annex VIII reasoning> | |
| AI Act tier | <high/limited/minimal/NA> | <Annex reference> | |
| Privacy regimes | <HIPAA/GDPR/other> | <data flow> | |
| Trial obligations | <IDE/IRB/GCP/NA> | <study purpose> | |

### Open questions
1. <question> — owner <role> — due <date>

### Assumptions and review trigger
<what would change these conclusions>
```

The confidence column is the most useful part. A conclusion marked low confidence is a signal to
commission a formal assessment rather than to proceed.

## 6. Worked example

This example triages a dermatology triage product and estimates the US and EU pathways. The
product and costs are constructed; the arithmetic is exact and the assumptions are stated.

### 6.1 Product

"DermAssist" analyzes a smartphone photograph of a skin lesion and flags lesions that a primary
care clinician should refer to dermatology. It does not diagnose. It is intended for use by
primary care clinicians in the US and the EU.

### 6.2 Intended use

"DermAssist estimates the likelihood that a photographed skin lesion warrants dermatology
referral in adults aged 18 years and older, using a smartphone image captured in a primary care
setting. It is intended to support, not replace, clinician judgment. It is not intended for
patients under 18 years, for lesions on mucosal surfaces, or for direct-to-consumer use without
a clinician."

### 6.3 US classification

The product makes a claim about referral for a suspected condition. It is software with a medical
purpose. The clinician sees a risk flag and can review the image and the model's heat map. The
product does not direct a specific treatment.

The CDS analysis is mixed. The output is a recommendation rather than a directive, and the
clinician can review the basis if the heat map and the training population are described. But the
claim concerns triage of a suspected malignancy, which is a diagnostic purpose. The conservative
conclusion is device status.

A predicate search identifies several cleared lesion-analysis devices. The product claims a
similar intended use and similar technological characteristics, so a 510(k) is the likely
pathway. If no acceptable predicate exists, De Novo is the fallback.

### 6.4 Evidence and timeline

Assumptions for the schedule, in weeks:

| Workstream | Duration | Dependency |
|---|---:|---|
| Predicate search and regulatory strategy | 4 | None |
| QMS gap assessment and remediation | 20 | Can start week 1 |
| Software lifecycle documentation (IEC 62304) | 12 | Starts after QMS gap |
| Algorithm validation (analytical) | 16 | Starts after data lock |
| Clinical validation study | 24 | Starts after algorithm freeze |
| Submission preparation | 8 | After clinical study |
| FDA 510(k) review | 13 | After submission |

```text
Critical path: predicate 4 + algorithm 16 + clinical 24 + preparation 8 + review 13
             = 65 weeks
QMS path:       QMS 20 + software 12 = 32 weeks, which fits inside the first 40 weeks
Total:          65 weeks = 15.0 months
```

The FDA's 510(k) review goal is 90 FDA days, which is about 13 weeks. The total of 65 weeks
includes the applicant's work, not only the agency's review.

### 6.5 Cost build-up

| Item | Low (USD) | High (USD) | Basis |
|---|---:|---:|---|
| QMS implementation | 120,000 | 240,000 | Consultant and staff time |
| Software lifecycle documentation | 60,000 | 120,000 | Engineering and quality time |
| Analytical validation | 80,000 | 160,000 | Data, compute, statistician |
| Clinical validation study | 250,000 | 550,000 | Site costs, adjudication, monitoring |
| Submission preparation | 40,000 | 90,000 | Regulatory and writing |
| Regulatory counsel | 60,000 | 150,000 | Review and strategy |
| FDA user fee | Fee schedule | Fee schedule | Set annually; check current rate |
| **Total excluding user fee** | **610,000** | **1,310,000** | Sum of ranges |

```text
Midpoint estimate = (610,000 + 1,310,000) / 2 = 960,000 USD
Per year over a 5-year product life = 960,000 / 5 = 192,000 USD per year
Revenue required to recover regulatory cost at 75% gross margin
  = 192,000 / 0.75 = 256,000 USD per year
```

The product must generate at least 256,000 USD per year in gross profit simply to recover the
regulatory investment, before engineering, sales, or support. That is a useful threshold for the
business case.

### 6.6 EU route

Under MDR, software intended for a medical purpose is a medical device. Classification under
Annex VIII depends on the significance of the decision and the potential for harm. A triage tool
that flags lesions for referral is likely Class IIa, which requires a notified body.

The notified body survey data suggest 13 to 18 months for a new MDR quality management system
certificate for most notified bodies, and 19 to 24 months for a combined QMS and product
certificate for a substantial minority. Assume 18 months for planning, with a range of 13 to 24
months.

```text
EU incremental cost estimate = 250,000 to 450,000 USD
EU incremental time = 13 to 24 months after US work is reusable
```

The technical file can reuse much of the US evidence, but the clinical evaluation under MDR has
its own requirements, including a systematic literature review and post-market clinical follow-up
planning.

### 6.7 AI Act

The product is an AI system and a medical device, so it is likely an Annex I high-risk AI system
under the AI Act. The high-risk obligations for Annex I products apply from 2 August 2028,
following the Digital Omnibus amendment that entered into force on 27 July 2026.

Planning implication: the product can launch earlier under MDR, but the AI Act obligations must be
designed in before August 2028. Retrofitting data governance, logging, and human oversight
documentation is expensive. The triage memo should record this as a dated obligation with an
owner.

### 6.8 The triage memo

| Question | Conclusion | Owner | Open issue |
|---|---|---|---|
| Device status (US) | Yes, likely Class II | Regulatory affairs | Confirm CDS analysis with counsel |
| Pathway | 510(k) with De Novo fallback | Regulatory affairs | Predicate search incomplete |
| QMS | ISO 13485 and 21 CFR 820 | Quality | Gap assessment not started |
| Clinical evidence | Prospective study, 24 weeks | Clinical affairs | Site agreements pending |
| Privacy (US) | PHI processed; BAA required | Privacy officer | Cloud vendor BAA |
| EU class | Likely IIa | Regulatory affairs | Annex VIII justification |
| AI Act | Annex I high-risk; obligations from 2 Aug 2028 | Regulatory affairs | Gap assessment in 2027 |
| Trial obligations | Possibly IDE if study supports a claim | Regulatory affairs | Determine with FDA pre-submission |

### 6.9 What would change the conclusion

If the product were positioned as an image-quality check with no referral claim, device status
might not apply. If it were sold only to researchers with RUO labeling, the pathway would differ,
but clinical use would exceed the labeling. If the output were a specific biopsy directive, the
CDS exclusion would fail and the device analysis would be stronger.

The lesson is that the claim, not the code, determines the pathway. A team can change the claim
faster than it can change the model.

## 7. Code, query, or calculation

The block implements a simplified triage rule set and a schedule calculator. It is a teaching
tool, not a regulatory decision system. Standard library only.

```python
"""Simplified regulatory triage and pathway timeline."""

from dataclasses import dataclass, field


@dataclass
class Product:
    name: str
    medical_claim: bool
    software: bool
    directs_action: bool
    clinician_can_review_basis: bool
    processes_phi: bool
    eu_market: bool
    is_ivd: bool = False
    in_trial: bool = False


def triage(p: Product) -> dict:
    regimes = []
    if p.medical_claim and p.software:
        regimes.append("FDA device analysis")
        if p.directs_action or not p.clinician_can_review_basis:
            regimes.append("FDA device: CDS exclusion unlikely")
        else:
            regimes.append("FDA device: CDS exclusion possible, document criteria")
        if p.eu_market:
            regimes.append("EU IVDR" if p.is_ivd else "EU MDR")
            regimes.append("EU AI Act (Annex I high-risk if device)")
    elif p.software and not p.medical_claim:
        regimes.append("Not a device if claim discipline holds")
    if p.processes_phi:
        regimes.append("HIPAA (BAA, minimum necessary)")
    if p.eu_market and p.processes_phi:
        regimes.append("GDPR (controller/processor, DPIA)")
    if p.in_trial:
        regimes.append("IDE / IRB / ICH E6 / 21 CFR Part 11")
    return {"product": p.name, "regimes": regimes}


@dataclass
class Schedule:
    tasks: list = field(default_factory=list)   # (name, weeks, depends_on)

    def critical_path(self) -> tuple:
        best = {}
        for name, weeks, dep in self.tasks:
            start = best[dep][1] if dep in best else 0
            end = start + weeks
            best[name] = (start, end)
        end_name = max(best, key=lambda k: best[k][1])
        return end_name, best[end_name][1]


TASKS = [
    ("predicate", 4, None),
    ("qms", 20, None),
    ("software_docs", 12, "qms"),
    ("analytical", 16, "predicate"),
    ("clinical", 24, "analytical"),
    ("submission", 8, "clinical"),
    ("review", 13, "submission"),
]

s = Schedule(TASKS)
end, weeks = s.critical_path()
print(f"Critical path ends at {end} with {weeks} weeks ({weeks / 52:.1f} years)")

low = 610_000
high = 1_310_000
mid = (low + high) / 2
print(f"Cost midpoint {mid:,.0f} USD")
print(f"Per year over 5 years {mid / 5:,.0f} USD")
print(f"Revenue needed at 75% margin {mid / 5 / 0.75:,.0f} USD per year")

products = [
    Product("DermAssist", True, True, False, True, True, True),
    Product("Discharge summarizer", True, True, True, False, True, True),
    Product("Ward capacity forecast", False, True, False, True, True, False),
]
for prod in products:
    result = triage(prod)
    print(f"\n{result['product']}")
    for r in result["regimes"]:
        print(f"  - {r}")
```

Expected output:

```text
Critical path ends at review with 65 weeks (1.2 years)
Cost midpoint 960,000 USD
Per year over 5 years 192,000 USD
Revenue needed at 75% margin 256,000 USD per year

DermAssist
  - FDA device analysis
  - FDA device: CDS exclusion possible, document criteria
  - EU MDR
  - EU AI Act (Annex I high-risk if device)
  - HIPAA (BAA, minimum necessary)
  - GDPR (controller/processor, DPIA)

Discharge summarizer
  - FDA device analysis
  - FDA device: CDS exclusion unlikely
  - EU MDR
  - EU AI Act (Annex I high-risk if device)
  - HIPAA (BAA, minimum necessary)
  - GDPR (controller/processor, DPIA)

Ward capacity forecast
  - Not a device if claim discipline holds
  - HIPAA (BAA, minimum necessary)
```

Two checks. First, the critical path is 4 + 16 + 24 + 8 + 13 = 65 weeks, not the sum of all
tasks. Second, the revenue threshold of 256,000 USD per year is the regulatory cost only, not the
full cost of doing business.

> **Note:** The rule set is deliberately simplified. It does not model the CDS criteria in detail,
> does not cover device classification, and does not replace regulatory affairs review.

## 8. Data, standards, and regulatory context

This section states what is binding, what is guidance, and what is convention. Statements are
attributed to the issuing body and date where possible. This is educational orientation, not
legal advice.

### 8.1 United States

| Instrument | Status | What it does |
|---|---|---|
| Federal Food, Drug, and Cosmetic Act (FD&C Act) | Statute | Defines devices and prohibits misbranded or adulterated products |
| 21 CFR 820 | Regulation | Quality system regulation for devices |
| 21 CFR 812 | Regulation | Investigational device exemptions |
| 21 CFR Part 11 | Regulation | Electronic records and electronic signatures |
| FDA CDS guidance (final, 2022) | Guidance | Describes CDS criteria for exclusion from device definition |
| FDA PCCP guidance (final December 2024, reissued August 2025) | Guidance | Describes predetermined change control plans for AI-enabled device software functions |
| FDA GMLP principles (2021, with MHRA and Health Canada) | Guidance | Good machine learning practice |
| FDA RWE guidance for devices (final, December 2025) | Guidance | Relevance and reliability of real-world evidence |
| HIPAA Privacy and Security Rules | Regulation | Use, disclosure, and safeguarding of PHI |
| HIPAA Security Rule NPRM (January 2025) | Proposed rule | Would strengthen cybersecurity requirements; not final as of October 2026 |
| CLIA, 42 CFR 493 | Regulation | Laboratory certification and personnel requirements |
| HTI-1 final rule (2024) | Regulation | Health IT certification, including AI transparency provisions |

The HIPAA Security Rule proposed rule was published on 6 January 2025 (90 FR 898, RIN 0945-AA22).
As of 24 October 2026 it has not been finalized, and HHS has moved the projected final action to
2027. The 2013-era Security Rule remains in effect.

### 8.2 European Union

| Instrument | Status | What it does |
|---|---|---|
| MDR, Regulation (EU) 2017/745 | Regulation, applied from May 2021 | Medical device classification, conformity, post-market surveillance |
| IVDR, Regulation (EU) 2017/746 | Regulation, applied from May 2022 | In vitro diagnostic classification and performance |
| AI Act, Regulation (EU) 2024/1689 | Regulation, amended in 2026 | Risk-based obligations for AI systems |
| Digital Omnibus on AI, in force 27 July 2026 | Amending regulation | Defers high-risk dates; other amendments |
| GDPR, Regulation (EU) 2016/679 | Regulation, applied from May 2018 | Personal data protection |

The AI Act's high-risk obligations now apply from 2 December 2027 for stand-alone Annex III
systems and 2 August 2028 for Annex I products, including medical devices. The amendment entered
into force on 27 July 2026.

### 8.3 International standards

| Standard | Status | What it constrains |
|---|---|---|
| ISO 13485 | Voluntary standard, often required | Quality management system |
| ISO 14971 | Voluntary standard, often required | Risk management |
| IEC 62304 | Voluntary standard, often required | Software lifecycle |
| IEC 82304-1 | Voluntary standard | Health software product safety |
| ISO 14155 | Voluntary standard | Clinical investigation of devices |
| IEC 62443 | Voluntary standard | Industrial cybersecurity; used for connected devices |
| ISO/IEC 27001 | Voluntary standard | Information security management |

"Voluntary" means not a statute. In practice, these standards are the accepted means of
demonstrating conformity, and a submission that ignores them will face questions.

### 8.4 What regulation does not decide

Regulation does not decide whether the product is clinically useful, whether the buyer will pay,
or whether the workflow will adopt it. A cleared device can fail commercially. A CE-marked
product can be clinically marginal. The regulatory mark is a gate, not a verdict.

## 9. Industry practice and case evidence

### 9.1 The device authorization landscape

**Documented.** The FDA's public list of AI-enabled medical devices, updated in September 2026
with data through 29 June 2026, contained 1,614 authorizations: 1,553 via 510(k), 40 via De Novo,
and 21 via PMA. Radiology accounted for 1,230 entries, or 76.2% of the total. The FDA notes that
the list is not comprehensive.

**Inference.** The dominance of 510(k) reflects the availability of predicates in imaging. A
product with no predicate faces De Novo, which is slower and requires more evidence. Predicate
availability is a strategic asset, not a detail.

### 9.2 Laboratory-developed tests

**Documented.** The FDA issued a final rule in May 2024 that would have phased out enforcement
discretion for laboratory-developed tests. On 31 March 2025, the US District Court for the Eastern
District of Texas vacated the rule in American Clinical Laboratory Association v. FDA. The FDA did
not appeal and, on 19 September 2025, issued a final rule implementing the vacatur (90 FR 45134,
Docket FDA-2025-N-1730), reverting 21 CFR 809.3(a) to its pre-2024 text.

**Inference.** The episode illustrates regulatory volatility. A product strategy that depended on
the 2024 rule would now be wrong. Triage memos should record the date of the regulatory
assumption and require re-checking before submission.

### 9.3 EU notified body timelines

**Documented.** European Commission notified body surveys report that, as of early 2026, 53
notified bodies were designated (52 under MDR, 19 under IVDR), with 32,898 MDR applications
lodged and 18,010 MDR certificates issued. 62% of notified bodies reported 13 to 18 months to
issue a new MDR quality management system certificate; for combined QMS and product certificates,
51% reported 13 to 18 months and 31% reported 19 to 24 months. For IVDR, 67% reported 13 to 18
months for combined QMS and product certification.

**Inference.** The EU is not a faster route than the US for a new device. A plan that assumes
simultaneous US and EU launch should budget 18 to 24 months for EU certification and should not
assume that US evidence is sufficient for MDR clinical evaluation.

### 9.4 The AI Act timetable change

**Documented.** The Digital Omnibus on AI was proposed on 19 November 2025 and entered into force
on 27 July 2026. It moved the AI Act's high-risk application dates to 2 December 2027 for Annex
III systems and 2 August 2028 for Annex I products.

**Inference.** Companies that had planned for August 2026 gained time but not exemption. The
practical risk is that teams deprioritize AI Act readiness and then face a compressed
implementation window in 2027 and 2028.

### 9.5 Enforcement discretion in practice

**Documented.** The FDA's 2022 CDS guidance describes categories of CDS that the agency does not
intend to enforce against as devices, subject to criteria. The guidance is a statement of
enforcement policy.

**Inference.** Enforcement discretion is not a safe harbour against state attorneys general,
private plaintiffs, or institutional risk committees. A hospital's legal team may impose
requirements that exceed the FDA's enforcement position.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Marketing outruns intended use | Website claims a diagnosis | Claims treated as marketing | Device status, liability | Review all public claims |
| CDS assumed exempt | "Clinician is in the loop" | Criteria not analysed | Unauthorized device | Criterion-by-criterion memo |
| No predicate search | Team assumes De Novo | Pathway choice delayed | Timeline and cost surprise | Predicate search in week one |
| QMS retrofitted | Design freeze before QMS | Quality system absent | Submission delay, audit findings | QMS gap assessment first |
| EU evidence assumed reusable | US study only | MDR clinical evaluation differs | Additional study | Separate EU evidence plan |
| AI Act ignored | "Not applicable until 2028" | Deferred obligations mistaken for none | Compressed remediation | Design obligations in now |
| LDT status assumed | Reliance on 2024 rule | Rule vacated | Non-compliant strategy | Re-check current status |
| PHI flow undocumented | No data flow diagram | Shadow data movement | Privacy breach | Data flow mapping |
| Trial obligations missed | No IDE assessment | Study used for marketing claim | Invalid study | Determine with FDA pre-submission |
| Change control absent | Model retrained monthly | No PCCP or change assessment | Unauthorized change | Change plan before launch |
| Single jurisdiction plan | Only US or only EU | Different regimes | Market entry failure | Two-jurisdiction matrix |
| Regulatory owner unnamed | No one accountable | Diffuse responsibility | Missed deadlines | Named owner in memo |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Intended use is stable | Determines everything | Claim expansion | Re-triage at each claim change |
| Predicate exists | 510(k) route | Novel device | Predicate search |
| QMS gap is closable in 20 weeks | Schedule | No quality function | Gap assessment |
| Clinical study can enroll | Timeline | Site access | Feasibility survey |
| Regulatory dates are current | Planning | Guidance changes | Check at each stage gate |
| Notified body capacity is available | EU timeline | Queue | Early engagement |
| Privacy regime is identifiable | Contracts | Multi-jurisdiction data | Data flow mapping |
| The claim can be narrowed | Reducing burden | Commercial pressure | Claim review with leadership |

### 11.1 Limits

This topic is not legal advice. Device classification, CDS analysis, and privacy obligations are
fact-specific and jurisdiction-specific. A triage memo is a starting point for professional
review, not a substitute for it.

The timelines and costs in section 6 are planning estimates built from stated assumptions. They
are not quotes and should not be used in a budget without local validation.

Regulatory positions change. The LDT rule, the AI Act timetable, and the HIPAA Security Rule have
all changed within the last two years. Every memo should carry an assumptions date and a review
date.

## 12. Exercises

### Exercise 1: Classify five products (application, 30 minutes)

**Deliverable.** Five product descriptions classified as research, QI, CDS, SaMD, or IVD, with a
one-paragraph rationale each.

**Acceptance criteria.** Each rationale cites the intended use, not the technology. At least one
product is borderline and the reasoning is explicit.

**Time.** 30 minutes.

### Exercise 2: Apply the CDS criteria (application, 35 minutes)

**Deliverable.** A criterion-by-criterion analysis of a clinical decision support product, with a
conclusion on device status.

**Acceptance criteria.** All four criteria addressed. The conclusion states which criterion is
closest to failing. No assertion without a rationale.

**Time.** 35 minutes.

### Exercise 3: Choose a US pathway (application, 30 minutes)

**Deliverable.** A pathway recommendation (510(k), De Novo, or PMA) with the evidence each would
require.

**Acceptance criteria.** Predicate availability addressed. Evidence table includes analytical,
clinical, and software verification. Timeline estimate with dependencies.

**Time.** 30 minutes.

### Exercise 4: EU classification (application, 35 minutes)

**Deliverable.** An MDR or IVDR classification with the Annex VIII reasoning and the conformity
route.

**Acceptance criteria.** Class stated with justification. Notified body requirement stated. One
difference from the US pathway identified.

**Time.** 35 minutes.

### Exercise 5: AI Act applicability (application, 25 minutes)

**Deliverable.** A determination of AI Act risk tier and the applicable dates, with the source of
each date.

**Acceptance criteria.** Cites the amended timeline. Distinguishes Annex I from Annex III.
Identifies at least two obligations that must be designed in.

**Time.** 25 minutes.

### Exercise 6: Privacy map (application, 30 minutes)

**Deliverable.** A data flow diagram with legal basis, contracts, and assessments for each flow.

**Acceptance criteria.** Every flow labelled with data type and legal basis. BAA and DPA
distinguished. At least one open question identified.

**Time.** 30 minutes.

### Exercise 7: Timeline and cost (calculation, 45 minutes)

**Deliverable.** A critical path schedule and a cost range for a device pathway, with assumptions.

**Acceptance criteria.** Dependencies shown. Critical path identified. Costs in ranges with a
basis. Revenue threshold computed.

**Time.** 45 minutes.

### Exercise 8: Write the memo (design, 40 minutes)

**Deliverable.** A one-page triage memo with regimes, conclusions, owners, open questions, and a
review date.

**Acceptance criteria.** Every conclusion has a rationale. Every open question has an owner. The
memo carries an assumptions date.

**Time.** 40 minutes.

## 13. Self-check answers

**1. What determines whether software is a medical device?**
Its intended use. If it is intended for diagnosis, treatment, prevention, or mitigation of disease,
it is a device unless an exclusion applies.

**2. What are the four CDS criteria?**
The criteria in the FDA's 2022 guidance concern the display and analysis of medical information,
whether the basis of the recommendation is independent of the software, whether the clinician can
independently review the basis, and whether the software provides a specific directive. All must
be met for the exclusion.

**3. Why is "the clinician is in the loop" not sufficient?**
Because the criteria require more than human involvement. The clinician must be able to
independently review the basis, and the software must not provide a specific directive.

**4. What is the difference between 510(k) and De Novo?**
510(k) relies on a predicate and substantial equivalence. De Novo is for novel devices of low to
moderate risk with no predicate.

**5. What EU class requires a notified body?**
Class IIa, IIb, and III under MDR, and most IVDs under IVDR. Class I devices can often
self-certify, with exceptions.

**6. When do AI Act high-risk obligations apply?**
From 2 December 2027 for stand-alone Annex III systems and 2 August 2028 for Annex I products,
following the 2026 amendment.

**7. What happened to the FDA LDT rule?**
It was vacated by a federal court on 31 March 2025 and formally withdrawn by the FDA on
19 September 2025. The pre-2024 enforcement-discretion framework governs.

**8. What is the critical path in the worked example?**
Predicate search 4 weeks, analytical validation 16, clinical validation 24, submission preparation
8, and FDA review 13, for 65 weeks.

**9. Why is a QMS needed before design freeze?**
Because design controls are part of the quality system, and retrofitting them after the design is
fixed creates documentation gaps and delays.

**10. What is the difference between a BAA and a DPA?**
A BAA implements HIPAA obligations for business associates. A DPA implements GDPR
controller-processor obligations. They can coexist.

### 13.1 Rubric for exercises 2 and 8

Exercise 2 is scored on coverage of all four criteria, quality of the rationale, and whether the
conclusion identifies the weakest criterion.

Exercise 8 is scored on completeness of the regimes, specificity of owners, and honesty about open
questions. A memo with no open questions is usually either trivial or overconfident.

## 14. Interview questions

**1. "Is this product a medical device?"**
Strong: asks about intended use and claims before answering. Weak: answers based on technology.

**2. "What is the CDS exclusion?"**
Strong: describes the four criteria and the need for clinician review of the basis. Weak: "if a
doctor uses it, it is CDS".

**3. "How would you choose between 510(k) and De Novo?"**
Strong: predicate search, intended use comparison, risk, and evidence. Weak: "510(k) is faster".

**4. "What does CE marking allow?"**
Strong: market access in the EU under MDR or IVDR, not US marketing. Weak: "it is global
approval".

**5. "What are the obligations of a high-risk AI system under the AI Act?"**
Strong: risk management, data governance, technical documentation, logging, human oversight,
transparency, post-market monitoring. Weak: "registration".

**6. "When would you involve regulatory affairs?"**
Strong: at intended use definition, before design freeze, and at every claim change. Weak: "before
launch".

**7. "What is a PCCP?"**
Strong: a pre-authorized plan describing modifications, a modification protocol, and an impact
assessment. Weak: "permission to retrain".

**8. "How do you handle a model update after clearance?"**
Strong: assess whether the change is within the PCCP; if not, new submission. Weak: "retrain and
deploy".

**9. "What changed in the EU AI Act timeline?"**
Strong: the 2026 omnibus deferred high-risk dates to December 2027 and August 2028. Weak: "it
starts in 2026".

**10. "What is the difference between RUO and clinical use?"**
Strong: RUO labeling limits use to research; clinical use exceeds it and triggers device
requirements. Weak: "RUO means no regulation".

**11. "How do you estimate regulatory cost?"**
Strong: evidence matrix, QMS gap, clinical study, submission, counsel, with ranges. Weak: "it
depends".

**12. "What would you do if the regulatory timeline exceeded the runway?"**
Strong: narrow the claim, delay, partner, or stop, with a decision framework. Weak: "raise more
money".

## 15. Key terms

- **510(k)** — premarket notification via substantial equivalence.
- **AI Act** — EU regulation 2024/1689, amended 2026.
- **BAA** — HIPAA business associate agreement.
- **CAP** — College of American Pathologists accreditation.
- **CDS** — clinical decision support.
- **CE marking** — EU conformity mark.
- **CLIA** — US laboratory regulation.
- **Companion diagnostic** — test required for therapy selection.
- **Controller** — GDPR role determining processing purposes.
- **CSV** — computer system validation.
- **De Novo** — FDA pathway for novel low-moderate risk devices.
- **DPIA** — data protection impact assessment.
- **DPA** — GDPR data processing agreement.
- **Enforcement discretion** — regulator's policy of not enforcing.
- **FRIA** — fundamental rights impact assessment.
- **GMLP** — good machine learning practice.
- **GxP** — good practice regulations.
- **HIPAA** — US health privacy law.
- **HTI-1** — US health IT certification rule.
- **IDE** — investigational device exemption.
- **Indications for use** — specific clinical circumstances.
- **Intended use** — regulated statement of function and population.
- **IVD** — in vitro diagnostic.
- **IVDR** — EU in vitro diagnostic regulation.
- **LDT** — laboratory-developed test.
- **MDR** — EU medical device regulation.
- **Minimum necessary** — HIPAA limitation principle.
- **Notified body** — EU conformity assessor.
- **PCCP** — predetermined change control plan.
- **PHI** — protected health information.
- **PMCF** — post-market clinical follow-up.
- **PMA** — FDA premarket approval.
- **Post-market surveillance** — ongoing monitoring after market entry.
- **Predicate** — device used for substantial equivalence.
- **Processor** — GDPR role processing on behalf of a controller.
- **RUO** — research use only.
- **SaMD** — software as a medical device.
- **SiMD** — software in a medical device.
- **Substantial equivalence** — regulatory finding relative to a predicate.
- **UDI** — unique device identification.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- US Food and Drug Administration, *Clinical Decision Support Software*, final guidance, 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- US Food and Drug Administration, *Marketing Submission Recommendations for a Predetermined
  Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, final
  guidance, reissued 18 August 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- US Food and Drug Administration, *List of Artificial Intelligence-Enabled Medical Devices*,
  data through 29 June 2026.
  https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
- US Food and Drug Administration, *Use of Real-World Evidence to Support Regulatory
  Decision-Making for Medical Devices*, final guidance, December 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-real-world-evidence-support-regulatory-decision-making-medical-devices
- US Food and Drug Administration, final rule implementing the vacatur of the laboratory developed
  tests rule, 90 FR 45134, 19 September 2025 (Docket FDA-2025-N-1730, RIN 0910-AJ05).
  https://www.federalregister.gov/documents/2025/09/19/2025-18239/regulation-identification-number-0910-aj05-medical-devices-laboratory-developed-tests-implementation
- US Department of Health and Human Services, HIPAA Security Rule Notice of Proposed Rulemaking,
  90 FR 898, 6 January 2025 (RIN 0945-AA22). Not final as of 24 October 2026.
  https://www.hhs.gov/hipaa/for-professionals/security/hipaa-security-rule-nprm/factsheet/index.html
- European Commission, "AI Omnibus enters into force," 27 July 2026.
  https://digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force
- European Parliament and Council, Regulation (EU) 2017/745 (MDR).
- European Parliament and Council, Regulation (EU) 2017/746 (IVDR).
- European Parliament and Council, Regulation (EU) 2024/1689 (AI Act), as amended in 2026.
- European Commission, notified body survey data on certification timelines, 2026.
  https://health.ec.europa.eu/document/download/59b9d90e-be42-4895-9f6f-bec35138bb0a_en
- Team-NB, Medical Device Survey 2025.
  https://www.team-nb.org/wp-content/uploads/2026/05/Survey-2025.pdf
- ISO 13485, Medical devices — Quality management systems.
- ISO 14971, Medical devices — Application of risk management.
- IEC 62304, Medical device software — Software life cycle processes.
- ISO 14155, Clinical investigation of medical devices — Good clinical practice.
- ICH, E6 Good Clinical Practice. https://www.ich.org/
- US Centers for Medicare and Medicaid Services, CLIA, 42 CFR Part 493.

> **Note:** This list mixes binding instruments and guidance. The status column in section 8 is the
> authoritative indicator in this file. Regulatory positions change; verify before submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_evidence-hierarchy-for-industry-claims.md`](06_evidence-hierarchy-for-industry-claims.md)
- Next topic: [`08_ethics-and-professional-conduct.md`](08_ethics-and-professional-conduct.md)
- Related topic: [`02_the-2026-industry-landscape.md`](02_the-2026-industry-landscape.md)
- Related topic: [`05_the-project-charter.md`](05_the-project-charter.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../course/00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
