# Threat Modeling

> **Module 16 - Privacy, Security, and Ethics**
> Topic 04 of 10
> Estimated study time: 3 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

Every clinical system has an attack surface that its builders can describe in detail and its
buyers cannot see. A sepsis alert, a FHIR (Fast Healthcare Interoperability Resources) API, a
connected infusion pump, a model endpoint, and a nightly extract to a cloud bucket each have
entry points, trust boundaries, and failure modes that are invisible in a demo. Threat modeling is
the structured conversation that makes them visible before an attacker finds them.

The cost of not doing it is documented. The Change Healthcare breach in 2024 began with a Citrix
remote-access portal that lacked multi-factor authentication and used compromised credentials.
The attacker moved laterally, exfiltrated about six terabytes of data, and disrupted claims
processing for months. The final breach count was 192.7 million individuals. The WannaCry attack
on the NHS in 2017 affected at least 80 trusts and 603 primary care organisations, and an
estimated 19,494 appointments were cancelled. Neither attack required a novel technique. Both
exploited controls that a threat model would have listed as missing.

Three failure patterns recur in biomedical threat modeling.

**The compliance threat model.** A team fills in a risk register with generic rows: "unauthorised
access", "data breach", "insider threat". Each row has a likelihood and an impact, and none has a
threat actor, a path, or a control that can be tested. The document satisfies a procurement
checklist and does not change the system. The test of a threat model is whether an engineer can
point to a design decision that came from it.

**The perimeter model.** The diagram shows a firewall, a "secure network", and a cloud. Everything
inside is trusted. In practice the threats that matter cross the perimeter with valid credentials,
arrive through a vendor's update channel, or sit inside the clinical workflow as a well-meaning
user exporting a spreadsheet. A threat model that stops at the network boundary misses the
clinical threat surface: shared workstations, group accounts, referral emails, and the integration
engine that can read every record.

**The model-only model.** The team threat-models the machine-learning model and not the system
around it. The model is robust to adversarial examples, and the training pipeline pulls data from
an unauthenticated object store, the inference endpoint logs full request bodies, and the
monitoring dashboard exposes patient identifiers. The model was never the weakest part.

This topic teaches threat modeling as a repeatable procedure that produces three artefacts: a
data-flow diagram with trust boundaries, a STRIDE (Spoofing, Tampering, Repudiation, Information
disclosure, Denial of service, Elevation of privilege) analysis mapped to elements, and a ranked
set of attack trees with controls and verification. It is the security counterpart to the privacy
assessment in Topic 01 and the de-identification determination in Topic 02. The privacy work asks
"what data should not be here"; the threat model asks "who could take it, change it, or stop it".

A fourth pattern is **threat modeling as a one-time gate**. The model is produced for the
architecture review, approved, and filed. The system then gains an interface, a vendor, and a data
type, and none of them is modelled. The document is accurate about a system that no longer exists,
which is worse than no document because it creates false assurance. The control is a change-trigger
list with an owner, not a calendar reminder.

The same shift is visible in procurement. Health systems now ask vendors for a threat model, a
software bill of materials, and a description of the update channel. A vendor that cannot produce
them is treated as a supply-chain risk, because the buyer has learned that the weakest path often
runs through a supplier rather than through the front door. Threat modeling is therefore a
commercial artefact as well as an engineering one.

> **Note:** A threat model is a structured argument about what could go wrong, not a guarantee
> that nothing will. Its value is in the design changes it forces and the tests it defines.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** a data-flow diagram for a clinical system with entities, processes, stores, flows, and
   trust boundaries, and **identify** the boundary crossings that matter.
2. **Apply** STRIDE per element and **explain** why the same threat category produces different
   controls at different elements.
3. **Build** an attack tree for a named threat, with AND and OR nodes, and **compute** the path
   probability and the resulting risk score.
4. **Rank** assets by harm to patients, not by data volume, and **justify** the ranking.
5. **Select** controls that prevent, detect, or respond, and **map** each control to a threat and a
   verification method.
6. **Evaluate** insider risk and **distinguish** it from external attack in likelihood and control.
7. **Integrate** privacy threats (linkage, inference, small cells) into a security threat model.
8. **Write** a threat model that an engineer, a clinician, and an auditor can each use.

## 3. The core mental model

A threat model has four layers: the system, the assets, the adversaries, and the controls. The
system layer is a data-flow diagram. The asset layer is a ranked list of what matters. The
adversary layer is a set of actors with capabilities and goals. The control layer is a set of
mechanisms mapped to threats, each with evidence.

```text
   +-------------+        (1)        +------------------+
   |  Clinician  |------------------>|  EHR / FHIR API  |
   |  workstation|   HTTPS + OAuth   |  (process)       |
   +-------------+                   +---------+--------+
                                             |
                                  (2)        |  (3)
                                             v
                                   +---------------------+
                                   |  Analytics lake     |
                                   |  (store)            |
                                   +----------+----------+
                                              |
                                   (4)        |  (5)
                                              v
                                   +---------------------+
                                   |  Model endpoint     |
                                   |  (process)          |
                                   +----------+----------+
                                              |
                                   (6)        v
                                        External vendor
   - - - - - - - - - - trust boundary - - - - - - - - - -
```

Each numbered flow is a place to ask questions. Flow 1 crosses from a shared workstation into the
API: who authenticates, with what factor, and can the session be hijacked? Flow 2 moves records
into the lake: is the data minimised, encrypted, and logged? Flow 3 is a store: who can read it
directly, and are backups included? Flow 4 feeds the model: is the training data separated from
production? Flow 5 is the endpoint: does it log request bodies, and can a caller enumerate
patients? Flow 6 crosses the organisational boundary: what does the contract say, and what does
the vendor's own attack surface look like?

STRIDE is applied per element, not per system:

| Element | S | T | R | I | D | E |
|---|---|---|---|---|---|---|
| External entity | Impersonation | Message alteration | Denial of action | Credential theft | Flooding | Privilege abuse |
| Process | Spoofed caller | Code or config tampering | Missing audit | Data in memory or logs | Resource exhaustion | Sandbox escape |
| Data store | Shared account | Unauthorised write | No change log | Direct read, backups | Ransomware | Credential escalation |
| Data flow | Session hijack | Man-in-the-middle | No delivery proof | Eavesdropping | Bandwidth saturation | Token scope creep |

The table is a prompt, not a taxonomy to memorise. Its value is that it forces six questions at
every element, including the ones that feel internal and therefore safe.

A second model is the attack tree. The root is the attacker's goal; the children are the ways to
achieve it; AND nodes require all children, OR nodes require any. The tree makes the arithmetic of
risk explicit, because the probability of a path is the product of its edges and the probability of
an OR node is one minus the product of the complements.

```text
                 [ exfiltrate PHI ]
                        OR
          +-------------+-------------+
          |                           |
   [ steal credentials ]      [ exploit gateway ]
          AND                         AND
     +----+----+                 +----+----+
   phish  reuse  API           vuln  exploit
   0.18   0.35   0.90          0.12   0.60
```

The third model is the harm hierarchy. Assets are ranked by what happens to a patient, not by how
many rows they contain. A small table of genetic variants for 200 children can be more harmful to
disclose than a million appointment records. The ranking drives where the modelling effort goes.

The harm hierarchy is not a substitute for the privacy assessment. A threat model can rank an
asset as high-harm and still miss that the harm arises from inference rather than disclosure; that
is the job of Topic 02. The two analyses share a diagram and a vocabulary, and they should share a
review date.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Threat model | A structured analysis of what can go wrong, who could cause it, and what controls reduce it. | A risk register with generic rows. | Process artefact. |
| Threat | A potential cause of harm, composed of an actor, a capability, and a path. | A vulnerability, or a category. | Statement. |
| Vulnerability | A weakness that a threat can exploit. | Confused with threat. | CVE or control gap. |
| Risk | The combination of likelihood and impact for a threat. | Treated as a single score without units. | Score or matrix. |
| Asset | Something of value to the system or its users: data, function, availability, trust. | Limited to data. | Inventory item. |
| Trust boundary | A line where the level of trust changes, such as a network, an organisation, or a user role. | Drawn only at the network edge. | Diagram annotation. |
| Data-flow diagram (DFD) | A diagram of entities, processes, stores, flows, and boundaries. | A network diagram. | Diagram. |
| STRIDE | Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, Elevation of privilege. | Used as a checklist without element mapping. | Threat taxonomy. |
| Attack tree | A tree of goals and sub-goals with AND/OR logic. | A list of attacks. | Diagram with probabilities. |
| AND node | All children must succeed. | Confused with OR. | Logic node. |
| OR node | Any child suffices. | Confused with AND. | Logic node. |
| Likelihood | Probability that a threat path is attempted and succeeds in a period. | Treated as a constant. | Probability per year. |
| Impact | Harm if the threat succeeds, measured in patient, financial, legal, or reputational terms. | Counted as records only. | Ordinal or currency. |
| Risk score | Likelihood times impact on a defined scale. | Compared across scales. | Defined scale. |
| Control | A mechanism that prevents, detects, or responds to a threat. | Listed without an owner or test. | Control ID. |
| Preventive control | Stops the threat path. | Confused with detective. | Control type. |
| Detective control | Detects the threat after it occurs. | Assumed to prevent. | Control type. |
| Responsive control | Limits harm after detection. | Omitted from the model. | Control type. |
| Residual risk | Risk remaining after controls. | Reported as zero. | Score. |
| Insider threat | A threat from someone with legitimate access. | Assumed to be malicious. | Actor category. |
| Negligent insider | An authorised user who causes harm by error. | Ignored in favour of malicious insiders. | Actor category. |
| Privileged access | Access that can change systems or read broadly. | Treated as equivalent to normal access. | Access tier. |
| Attack surface | The set of entry points and exposed functionality. | Counted as a number without context. | Inventory. |
| Lateral movement | Moving from one compromised system to another. | Assumed blocked by the perimeter. | Attack stage. |
| Exfiltration | Removing data from the environment. | Confused with encryption. | Attack stage. |
| Ransomware | Malware that encrypts or steals data for extortion. | Treated only as availability risk. | Threat category. |
| MFA | Multi-factor authentication. | Assumed to stop phishing. | Control. |
| Service account | A non-human account used by a system. | Shared by people. | Identity type. |
| Repudiation | Denying that an action occurred. | Ignored without audit logs. | STRIDE category. |
| Threat actor | A person or group with a goal and capability. | Named only as "attacker". | Actor profile. |
| Kill chain | The stages of an attack from reconnaissance to exfiltration. | Used as a compliance map. | Model. |
| Clinical threat surface | The entry points created by clinical workflow, devices, and integrations. | Reduced to the network. | Scope. |

## 5. The mechanics, step by step

The procedure below produces a threat model in roughly one working day for a system of moderate
complexity. It assumes a cross-functional group: an engineer who knows the architecture, a
clinician who knows the workflow, a security engineer, and a privacy lead. Each step names its
input, its output, and the reason it exists.

### Step 1. Scope the system and the decision it supports

**Input:** System description, intended use, deployment context.
**Output:** A scope statement naming what is in, what is out, and the decision the model will
inform.
**Rationale:** A threat model without a boundary expands until it becomes a model of the
organisation. The scope should name the system, its users, its data, its interfaces, and its
deployment environment. It should also name the decision: approve the architecture, choose between
two designs, certify a vendor, or plan a release.

A useful scope statement is one paragraph: "The insulin dosing decision support service, its
FHIR interface to the EHR, its model endpoint, its vendor update channel, and the analytics lake
that supplies training data. Out of scope: the EHR itself, the hospital network, and the patient
app. The model informs the architecture review scheduled for 14 November."

Branch point: if the system has no clear boundary, the first deliverable is a context diagram, not
a threat list.

### Step 2. Draw the data-flow diagram

**Input:** Architecture, network diagram, API inventory, deployment manifests.
**Output:** A DFD with external entities, processes, data stores, flows, and trust boundaries.
**Rationale:** The DFD is the artefact the rest of the analysis hangs on. It must show where data
enters, where it is transformed, where it rests, and where it leaves. Trust boundaries are drawn
where the level of trust changes: between the clinical network and the cloud, between the
application and the vendor, between the user role and the admin role, and between production and
analytics.

The diagram should be simple enough to redraw from memory. If it has forty boxes, it is an
architecture diagram, not a threat model. Group by function and expand only where a boundary is
crossed.

Branch point: if the vendor's internal architecture is unknown, draw it as a black box with a
boundary and record the unknown as an assumption. A black box is honest; a guessed internal
diagram is not.

### Step 3. Inventory and rank assets

**Input:** Data inventory, clinical workflow, business context.
**Output:** A ranked asset list with a harm statement for each.
**Rationale:** Ranking by record count is a common and misleading shortcut. The relevant question
is what happens to a patient, a clinician, or the organisation if the asset is disclosed, altered,
or unavailable. A model that recommends insulin doses is an availability and integrity asset before
it is a confidentiality asset: a wrong dose harms a patient immediately, while a leaked feature
list may harm them later.

A practical ranking uses four harm dimensions: patient safety, privacy, legal and regulatory,
and operational. Each asset gets a rank per dimension, and the highest rank drives the modelling
effort.

| Asset | Patient safety | Privacy | Legal | Operational | Modelling priority |
|---|---|---|---|---|---|
| Dosing recommendation | High | Medium | High | High | 1 |
| Training data with identifiers | Low | High | High | Medium | 2 |
| Model weights | Low | Medium | Medium | High | 3 |
| Audit logs | Low | Medium | High | Medium | 4 |
| Public model documentation | Low | Low | Low | Low | 5 |

Branch point: if the asset list has no patient-safety column, the model is a data security model,
not a clinical threat model. Add the column before proceeding.

### Step 4. Enumerate threat actors and entry points

**Input:** DFD, asset list, organisational context.
**Output:** An actor list with capability, motivation, and access; an entry-point list.
**Rationale:** Different actors have different capabilities and different likelihoods. A script
kiddie, a ransomware affiliate, a nation-state, a negligent clinician, a malicious insider, and a
vendor with privileged access are not interchangeable. The model should name at least four actors
for a clinical system: an external opportunistic attacker, an organised criminal group, a
negligent insider, and a malicious insider or compromised vendor.

Entry points come from the DFD: APIs, user interfaces, file imports, device interfaces, update
channels, and support tools. Each entry point should be tagged with its authentication method,
its network exposure, and its data scope.

### Step 5. Apply STRIDE per element

**Input:** DFD, actor list, entry points.
**Output:** A threat table with one row per credible threat, mapped to an element and a STRIDE
category.
**Rationale:** STRIDE's value is coverage. Applied per element, it produces threats that a
brainstorm misses, especially repudiation and denial of service, which are usually absent from
privacy-led analyses. The table should record the element, the category, the threat statement, the
actor, and whether the threat is credible in this deployment.

A threat statement should be specific: "an authenticated clinician on a shared workstation can
retrieve any patient's dosing history by changing the patient identifier in the request, because
the endpoint authorises the role but not the patient relationship". That statement implies a
control (object-level authorisation) and a test.

Branch point: if a STRIDE row produces no credible threat, record why. "No credible denial of
service because the endpoint is internal and rate-limited" is a finding, not an omission.

### Step 6. Build attack trees for the top threats

**Input:** The three to five highest-ranked threats.
**Output:** An attack tree per threat with probabilities on the edges.
**Rationale:** The tree converts a narrative into arithmetic. It also exposes single points of
failure: an AND node with one well-controlled child is much less likely than an OR node with two
weak children. The probabilities are estimates, and the model should say so; their value is in
ranking paths and in showing which control changes the result most.

Probability estimates should come from a stated basis: historical incident rates, vendor
disclosure, penetration-test findings, or expert judgement. An estimate with no basis is a guess,
and it should be labelled as such.

### Step 7. Score likelihood and impact

**Input:** Attack trees, asset ranking, actor list.
**Output:** A risk score per threat on a defined scale, with the inputs visible.
**Rationale:** A score without a scale is not comparable. The model should state the scale, the
time period, and the impact dimensions. A simple and defensible scheme is likelihood as an annual
probability and impact as a 1-5 ordinal, with risk = likelihood x impact x 10 on a 0-50 scale.

| Likelihood (annual) | Descriptor | Impact 1-5 | Descriptor |
|---|---:|---|---|
| > 0.50 | Almost certain | 5 | Patient harm or catastrophic disclosure |
| 0.10 - 0.50 | Likely | 4 | Major regulatory or operational harm |
| 0.01 - 0.10 | Possible | 3 | Moderate harm, recoverable |
| 0.001 - 0.01 | Unlikely | 2 | Minor harm |
| < 0.001 | Rare | 1 | Negligible |

The rating decision table turns the score into an action:

| Score | Rating | Required action | Approval level |
|---:|---|---|---|
| >= 15 | Critical | Do not deploy; redesign or accept at executive level | Executive risk owner |
| 5 - 15 | High | Remediate before release; compensating controls allowed | Security and clinical leads |
| 2 - 5 | Medium | Remediate in the next cycle; monitor | Product owner |
| 1 - 2 | Low | Accept and review annually | Team lead |
| < 1 | Minimal | Record and review on change | Team lead |

The thresholds are conventions. Their value is that they are written down before the scores are
known, which prevents the common failure of adjusting the threshold to fit the preferred answer.

### Step 8. Select controls

**Input:** Ranked threats, existing controls, architecture constraints.
**Output:** A control list mapped to threats, with a control type and an owner.
**Rationale:** Controls are chosen to break a path, not to satisfy a category. A control that
prevents one path may leave another open, and a control that detects is not a substitute for one
that prevents when the harm is immediate. Each control should be labelled preventive, detective,
or responsive, and mapped to the specific threat it addresses.

| STRIDE category | Preventive examples | Detective examples | Responsive examples |
|---|---|---|---|
| Spoofing | MFA, client certificates, short-lived tokens | Anomalous login detection | Session revocation, credential reset |
| Tampering | Signed artefacts, immutable logs, input validation | Integrity monitoring, checksum alerts | Rollback, re-deploy from trusted source |
| Repudiation | Per-user identities, signed audit records | Log correlation, time sync | Forensic reconstruction |
| Information disclosure | Encryption, field minimisation, object-level authorisation | DLP, egress monitoring, canary records | Containment, notification |
| Denial of service | Rate limits, quotas, redundancy | Latency and error-rate alerts | Failover, traffic filtering |
| Elevation of privilege | Least privilege, separation of duties, sandboxing | Privilege-change alerts | Revoke, rebuild, audit scope |

Branch point: if a control cannot be tested, it is not a control. Replace it with one that produces
evidence, or record it as an assumption.

### Step 9. Define verification evidence

**Input:** Control list.
**Output:** A test or evidence item per control: configuration check, penetration test, log query,
tabletop exercise, or attestation.
**Rationale:** The model is only as good as the evidence that the controls exist and work. A
control that is configured once and never tested drifts. Verification should be specific and
repeatable: "the API gateway rejects requests without a client certificate, tested weekly by an
automated probe" is verifiable; "access is restricted" is not.

### Step 10. Record residual risk and acceptance

**Input:** Scores, controls, verification results.
**Output:** A residual risk statement per threat and a named acceptor.
**Rationale:** Residual risk is what remains after controls, and it must be accepted by someone
with authority. The acceptance should state the threat, the residual score, the controls relied
on, and the review date. An acceptance without a name is not an acceptance.

### Step 11. Review and maintain

**Input:** Change log, incident reports, new threats.
**Output:** A review cadence and triggers.
**Rationale:** Threat models age. New interfaces, new vendors, new regulations, and new attack
techniques change the picture. Review triggers should include a new external interface, a new data
type, a new vendor, a material architecture change, a relevant incident elsewhere, and the passage
of twelve months.

### Threat modeling for AI components

A clinical AI component adds threats that STRIDE does not name directly. They map onto STRIDE
categories, but the controls are different enough that they deserve their own rows in the threat
table.

| AI threat | STRIDE mapping | Where it enters | Control | Verification |
|---|---|---|---|---|
| Data poisoning | Tampering | Training pipeline, vendor data, public sources | Provenance, source allowlist, outlier review | Canary records; distribution tests |
| Evasion | Tampering | Inference input, device telemetry | Input validation, adversarial training, human review | Adversarial test set; drift monitor |
| Membership inference | Information disclosure | Model endpoint, published model | DP training, output limiting, access control | Attack evaluation |
| Model inversion | Information disclosure | Model endpoint, API | Output minimisation, rate limits, DP | Reconstruction test |
| Model extraction | Information disclosure | Prediction API | Rate limits, watermarking, query monitoring | Query-pattern detection |
| Prompt injection | Tampering, elevation | LLM interface, retrieved documents | Tool allowlists, input isolation, output validation | Red-team prompts |
| Supply-chain model update | Tampering | Vendor update channel | Signing, staged rollout, reproducibility | Signature and provenance check |
| Training-data leakage through logs | Information disclosure | Pipeline logs, experiment tracking | Log redaction, access control | Log sampling and review |

The placement matters. Poisoning is a training-time integrity threat, so the control sits in the
data pipeline and the verification is a data test. Evasion is an inference-time integrity threat,
so the control sits at the endpoint and the verification is an adversarial test. Membership
inference and model inversion are confidentiality threats against the model, and the only
mechanism that bounds them formally is differential privacy (Topic 03); the other controls reduce
exposure but do not bound it.

### Scoring under uncertainty

The probabilities in an attack tree are rarely measured. A defensible model does three things with
that uncertainty. It records the basis for each input (incident data, vendor disclosure, expert
judgement, or illustrative). It runs a sensitivity analysis on the two or three inputs that
dominate the result. And it reports a range rather than a single number when the range changes the
rating.

A simple sensitivity test is to vary each dominant input by a factor of two and recompute. If the
rating does not change, the conclusion is robust to that input. If it changes, the input needs a
better basis before the model is used for a decision. The arithmetic in Section 6 is an example:
the gateway vulnerability dominates, so the credential probability can be halved without changing
the rating, while a change in the vulnerability probability moves the rating across a threshold.

### From threat to control: a worked row

A threat statement is only useful if it implies a control and a test. The template has five parts:

```text
Actor:        an authenticated clinician on a shared workstation
Precondition: the caller has a valid session and knows a patient identifier
Action:       changes the identifier in the request to another patient's record
Element:      FHIR API, Patient resource
Impact:       disclosure of another patient's glucose history
Control:      object-level authorisation bound to the care relationship
Test:         request a record outside the caller's relationship; expect 403 and an audit entry
```

The template forces specificity. "An attacker could access patient data" has no actor, no
precondition, and no test; it cannot be verified or fixed. The five-part statement can be assigned,
implemented, and tested, and it can be closed when the test passes.

### Edge cases

**Small cells.** A published count of one or two patients is a disclosure threat even when no
identifier is present. In a threat model, the release process is an element and the small-cell
inference is a threat path with an impact on privacy. The control is output review and minimum
cell size, and the verification is a test that rejects a table with a cell below the threshold.

**Quasi-identifiers.** Linkage is an attack path: the adversary obtains an auxiliary dataset,
joins on age, sex, and geography, and recovers identity. The threat model should include the
auxiliary data as an external entity and the join as a flow, even though it happens outside the
system. This is the main place where privacy threats and security threats share a diagram.

**Re-identification through linkage.** The impact is privacy harm, and the likelihood depends on
the availability of auxiliary data. The control may be contractual (a DUA), technical
(generalization), or architectural (in-place analysis). The model should record which.

**Free-text PHI.** Clinical notes are an asset with an uncontrolled content channel. The threats
are disclosure through the note field, disclosure through a search index, and disclosure through a
backup. The controls are exclusion, de-identification, and access restriction; the verification is
a sampled review and a detector with a measured false-negative rate.

**Metadata leakage.** File properties, DICOM headers, logs, and error messages are flows that are
often omitted from the DFD. Add a metadata flow to every export and every log sink, and ask what it
contains.

**Consent withdrawal.** Withdrawal is an authorisation event, and a failure to propagate it is a
repudiation and compliance threat. The model should include the withdrawal path from the patient
portal to every derived dataset and model, and the control is a reconciliation job with evidence.

**Minors and deceased patients.** Authorisation rules differ for personal representatives,
adolescents, and decedents. The threat is unauthorised access through a legitimate-looking
account, for example a parent portal that exposes adolescent-sensitive encounters. The control is
age-aware authorisation, and the verification is a test matrix by age and service type.

**Cross-border transfer.** A transfer is a trust-boundary crossing with legal consequences. The
model should mark the boundary, name the destination, and record the transfer mechanism and the
recipient's obligations. The security controls (encryption, access) support the legal analysis but
do not replace it.

**Vendor and supply chain.** The update channel is an entry point. A compromised vendor build, a
malicious dependency, or a support engineer with remote access can bypass every control at the
perimeter. The model should include the vendor as an external entity, the update flow as a path,
and the verification as signed artefacts and provenance checks.

**Identity provider compromise.** The identity provider is a trust boundary that is often drawn
outside the system and therefore outside the model. If it is compromised, every downstream
authorisation decision is made with forged assertions, and object-level checks may still pass
because the caller appears to be a legitimate clinician. The model should include the identity
provider as an external entity, the assertion flow as a data flow, and the signing keys as an
asset. The controls are key protection, assertion validation (audience, issuer, expiry), and
anomaly detection on authentication patterns.

**Clinical workflow workarounds.** A control that slows care will be bypassed. Shared logins,
photographed screens, and spreadsheet exports are threat paths created by usability. The model
should include the workaround as an actor behaviour, because it is often the most likely path.

## 6. Worked example

### Setting

A regional health system is deploying an insulin dosing decision support service across 12
hospitals. The service reads recent glucose values and insulin orders from the EHR through a FHIR
API, computes a dose recommendation, and writes the recommendation back to the ordering screen.
A vendor supplies the model and a monthly update package. An analytics lake supplies training data
and holds 41,280 patient records with identifiers. Forty analysts have lake access; three have
bulk-export rights. The architecture review is in two weeks.

### Step 1: scope

In scope: the FHIR interface, the model endpoint, the vendor update channel, the analytics lake,
and the export path to the vendor's monitoring service. Out of scope: the EHR itself, the hospital
network, and the patient portal. The decision is whether to approve the architecture for pilot
deployment.

### Step 2: the data-flow diagram

The diagram has five elements and six flows:

1. Clinician workstation to FHIR API (HTTPS, OAuth 2.0, shared workstation).
2. FHIR API to analytics lake (nightly extract, service account).
3. Analytics lake store (object storage, 41,280 identified records).
4. Analytics lake to model endpoint (training pipeline, read-only credentials).
5. Model endpoint to vendor monitoring (outbound, API key).
6. Vendor to model endpoint (monthly update package).

Trust boundaries: the clinical network to the cloud (flows 1, 2, 5), the organisation to the
vendor (flows 5, 6), and the analyst role to the production role (flow 2).

### Step 3: assets

| Asset | Patient safety | Privacy | Legal | Operational | Priority |
|---|---|---|---|---|---|
| Dosing recommendation | High | Medium | High | High | 1 |
| Identified training data (41,280 records) | Low | High | High | Medium | 2 |
| Model weights | Low | Medium | Medium | High | 3 |
| Audit logs | Low | Medium | High | Medium | 4 |
| Vendor update package | High | Low | Medium | High | 2 |

The vendor update package is ranked equal to the training data because a malicious or corrupted
update can alter every dose recommendation. This ranking is the first place where a clinical threat
model differs from a data security model.

### Step 4: actors and entry points

- External opportunistic attacker: scans internet-facing services, uses leaked credentials.
- Organised criminal group: ransomware and data extortion, willing to spend weeks in the network.
- Negligent insider: analyst who exports data to a personal drive or emails a spreadsheet.
- Malicious or compromised insider: analyst or vendor engineer with legitimate access.

Entry points: the FHIR API (internet-facing through a gateway), the analyst console, the lake's
object-storage endpoint, the model endpoint, the vendor update channel, and the vendor monitoring
callback.

### Step 5: STRIDE findings

The analysis produces 23 rows. Four are material:

| # | Element | Category | Threat | Actor | Credible |
|---:|---|---|---|---|---|
| 1 | FHIR API | Information disclosure | A caller changes the patient identifier and retrieves another patient's glucose history because authorisation checks role but not patient relationship | External attacker with a valid clinician credential | Yes |
| 2 | Analytics lake | Information disclosure | An analyst with read access copies the identified training table to a personal drive | Negligent insider | Yes |
| 3 | Vendor update | Tampering | A compromised vendor build alters dose logic | Organised criminal or vendor insider | Yes |
| 4 | FHIR API | Repudiation | Fourteen users share one integration service account, so actions cannot be attributed | Any insider | Yes |

Row 1 is the object-level authorisation gap. Row 2 is the insider path. Row 3 is the supply-chain
path. Row 4 is the attribution gap. Each maps to a different control, and none is addressed by
"encrypt the data".

### Step 6: attack tree for row 1

Goal: exfiltrate protected health information (PHI) from the analytics API. Two paths:

- Path A, stolen clinician credentials: phishing succeeds with probability 0.18; the stolen
  credential is reused on the API with probability 0.35 (no MFA, password reuse); the API grants
  access to a patient record with probability 0.90 (role check passes, object-level check absent).
  Path probability = 0.18 x 0.35 x 0.90 = 0.0567.
- Path B, gateway vulnerability: an exploitable vulnerability is present with probability 0.12 in
  a given year; exploitation succeeds with probability 0.60. Path probability = 0.12 x 0.60 =
  0.0720.

The OR node: 1 - (1 - 0.0567) x (1 - 0.0720) = 1 - 0.9433 x 0.9280 = 1 - 0.8754 = 0.1246.

### Step 7: scoring

Impact is 5: a successful exfiltration of the identified lake would affect up to 41,280 patients
and trigger notification, regulatory, and contractual consequences. Risk = 0.1246 x 5 x 10 = 6.23
on the 0-50 scale. Under the rating table, 6.23 is High: remediate before release, with
compensating controls allowed.

The expected number of exposed records per year, if a success exposes the full lake, is
0.1246 x 41,280 = 5,143. At an assumed USD 12 per notified individual for notification and
monitoring, the expected annual notification cost is 5,143 x 12 = USD 61,716. The dollar figure is
a planning assumption, not a benchmark; the useful quantity is the record count, which is driven by
the likelihood and the lake size.

### Diagnostic 1: an internet-reachable endpoint without client authentication

A configuration review finds that the FHIR gateway is reachable from the internet and accepts
bearer tokens without a client certificate. Three of fourteen endpoints lack object-level
authorisation, so 3 / 14 = 21.4 percent of the API surface will return any patient's record to any
authenticated caller who guesses or enumerates an identifier. This is the finding that makes path
A credible at 0.90 and path B credible at all. Without it, the same attack tree would score much
lower, which is exactly why the diagnostic must be a configuration test rather than a workshop
opinion.

### Diagnostic 2: shared service account and bulk-export rights

The integration engine authenticates as a single service account used by 14 people. If an action
is taken with that account, the probability of attributing it to the right person is 1 / 14 = 7.1
percent. Separately, 3 of 40 analysts have bulk-export rights, and each can export 12,000 records
per query: 3 x 12,000 = 36,000 records are reachable through a single account without a per-record
audit trail. At an assumed 2 percent annual probability of misuse by one of the three, the expected
records exposed through this path is 0.02 x 36,000 = 720 per year.

The two diagnostics are different in kind. The first is an external exposure; the second is an
internal attribution and scope problem. A threat model that addresses only the perimeter will miss
the second, and the second is the more likely path in most health systems.

### Vary one assumption: MFA on clinician accounts

Suppose the health system deploys phishing-resistant MFA for all clinician accounts. The
credential-reuse probability falls from 0.35 to 0.03, because a stolen password alone no longer
authenticates. Path A becomes 0.18 x 0.03 x 0.90 = 0.00486. The OR node becomes
1 - (1 - 0.00486) x (1 - 0.0720) = 1 - 0.99514 x 0.9280 = 1 - 0.9235 = 0.0765. Risk = 0.0765 x 5 x
10 = 3.83, which is Medium rather than High.

The conclusion changes: the architecture moves from "remediate before release" to "remediate in
the next cycle". But the risk is still dominated by path B, the gateway vulnerability, because
0.0720 of the 0.0765 total comes from that path. MFA reduced the credential path by 91 percent and
the total risk by only 39 percent. This is the single most useful output of the arithmetic: it
shows which control buys the most risk reduction, and it shows that one control is not enough.

Adding gateway patching, reducing the vulnerability probability from 0.12 to 0.02, gives path B =
0.0120, total likelihood 0.0168, and risk 0.84, which is Minimal. The two controls together move
the decision from "do not deploy" to "deploy with monitoring".

### What the model changed

Three design decisions came out of the exercise, and they are the evidence that the model was not
decorative:

1. Object-level authorisation was added to all fourteen endpoints, with a test that requests a
   record outside the caller's patient relationship and expects a 403.
2. The integration service account was replaced with per-user service identities and short-lived
   tokens, and bulk export was restricted to two named analysts with per-record audit.
3. The vendor update package is now signed and verified, with a staged rollout and a rollback
   procedure.

Each decision maps to a threat row, a control, and a verification. That traceability is the
deliverable a reviewer should ask for.

### Control cost and the stop rule

The arithmetic can also compare controls by cost per unit of risk reduction. MFA reduced
likelihood from 0.1246 to 0.0765, a reduction of 0.0481, at an assumed annual cost of USD 40,000
for 2,000 clinicians. Gateway patching reduced likelihood from 0.0765 to 0.0168, a reduction of
0.0597, at an assumed annual cost of USD 15,000 for engineering time. The second control buys more
risk reduction per dollar in this example, which is the kind of comparison a security investment
decision needs.

The comparison has a limit: the two controls address different paths, and removing one does not
substitute for the other. The cost-per-reduction calculation ranks the order of work; it does not
justify skipping a control that closes a path with patient-safety impact. A stop rule should be
written before the comparison: no deployment while any path with impact 5 is above the High
threshold, regardless of cost.

### What the model does not cover

The model does not cover the EHR itself, the hospital network, or the patient portal, by scope. It
also does not cover the insider path through the model vendor's support engineer, whose remote
access is governed by a separate contract and was assessed only as a black box. It does not
quantify the probability that a clinician will bypass a control under time pressure, which is the
most likely path in many wards. And it does not cover the privacy harm from inference, which is
assessed in Topic 02. These exclusions are recorded so that a reader does not mistake the scope for
the whole risk picture.

## 7. Code, query, or calculation

The script below implements the attack tree, the OR and AND logic, the risk score, and the
sensitivity analysis for the two control scenarios. It uses only the standard library. The
probabilities are illustrative inputs, not measurements.

```python
"""Attack-tree scoring for a synthetic clinical analytics API."""

TREE = {
    "name": "exfiltrate PHI from analytics API",
    "op": "OR",
    "children": [
        {
            "name": "stolen clinician credentials",
            "op": "AND",
            "children": [
                {"name": "phishing success", "p": 0.18},
                {"name": "credential reuse works", "p": 0.35},
                {"name": "API access granted", "p": 0.90},
            ],
        },
        {
            "name": "gateway vulnerability",
            "op": "AND",
            "children": [
                {"name": "exploitable vulnerability present", "p": 0.12},
                {"name": "exploit succeeds", "p": 0.60},
            ],
        },
    ],
}


def node_prob(node, overrides=None):
    overrides = overrides or {}
    if "p" in node:
        return overrides.get(node["name"], node["p"])
    probs = [node_prob(c, overrides) for c in node["children"]]
    if node["op"] == "AND":
        result = 1.0
        for p in probs:
            result *= p
        return result
    result = 1.0
    for p in probs:
        result *= (1 - p)
    return 1 - result


def leaves(node, prefix=()):
    if "p" in node:
        yield prefix + (node["name"],), node["p"]
        return
    for child in node["children"]:
        yield from leaves(child, prefix + (node["name"],))


def paths(node):
    out = []
    for branch in node["children"]:
        names, probs = [], []
        for path, p in leaves(branch):
            names.append(path[-1])
            probs.append(p)
        out.append((branch["name"], names, probs))
    return out


if __name__ == "__main__":
    impact = 5
    for label, overrides in (
        ("base", {}),
        ("MFA on clinician accounts", {"credential reuse works": 0.03}),
        ("MFA + gateway patched", {"credential reuse works": 0.03,
                                   "exploitable vulnerability present": 0.02}),
    ):
        p = node_prob(TREE, overrides)
        print(f"{label}: likelihood={p:.4f}, risk={p * impact * 10:.2f}/50")

    print("paths:")
    for name, names, probs in paths(TREE):
        prod = 1.0
        for x in probs:
            prod *= x
        print(f"  {name}: {' x '.join(f'{n}={p}' for n, p in zip(names, probs))} "
              f"= {prod:.4f}")

    shared_users = 14
    print(f"attribution gap with a shared service account: 1/{shared_users} = "
          f"{1 / shared_users:.1%}")
    print(f"bulk-export exposure: 3 analysts x 12,000 records = {3 * 12_000:,} records")
```

Expected output, verified on Python 3.11:

```text
base: likelihood=0.1246, risk=6.23/50
MFA on clinician accounts: likelihood=0.0765, risk=3.83/50
MFA + gateway patched: likelihood=0.0168, risk=0.84/50
paths:
  stolen clinician credentials: phishing success=0.18 x credential reuse works=0.35 x API access granted=0.9 = 0.0567
  gateway vulnerability: exploitable vulnerability present=0.12 x exploit succeeds=0.6 = 0.0720
attribution gap with a shared service account: 1/14 = 7.1%
bulk-export exposure: 3 analysts x 12,000 records = 36,000 records
```

The output is the arithmetic behind the narrative. The base risk of 6.23 is High on the section's
scale. MFA alone reduces it to 3.83, which is Medium, and the two controls together reduce it to
0.84. The path printout shows why: the gateway path contributes 0.0720 of the base 0.1246, so
removing the credential path cannot bring the total below that floor.

Two cautions about using the script. First, the leaf probabilities are inputs, and the output is
only as good as they are; the model should record the basis for each. Second, the tree assumes
independence between paths, which is false when the same attacker uses both. The OR calculation is
therefore an approximation, and it should be described as one.


## 8. Data, standards, and regulatory context

Threat modeling is required indirectly by several regimes and specified in detail by frameworks.
The distinction matters when a procurement questionnaire asks whether the organisation "does threat
modeling": the answer should name the standard, the artefact, and the review cadence, not the word.

### HIPAA Security Rule

The HIPAA Security Rule requires a risk analysis as the first administrative safeguard. 45 CFR
164.308(a)(1)(ii)(A) requires an accurate and thorough assessment of the potential risks and
vulnerabilities to the confidentiality, integrity, and availability of electronic protected health
information. The risk analysis is not a one-time document; 164.308(a)(1)(ii)(B) requires risk
management measures, and 164.316 requires documentation and periodic review.

The Security Rule's safeguards map onto the threat model:

| Safeguard family | Examples | Threat category addressed |
|---|---|---|
| Administrative (164.308) | Risk analysis, workforce security, access management, incident response, contingency planning | All categories; governance and response |
| Physical (164.310) | Facility access, workstation use, device and media controls | Information disclosure, denial of service |
| Technical (164.312) | Access control, audit controls, integrity, authentication, transmission security | Spoofing, tampering, disclosure, elevation |
| Organisational (164.314) | Business associate contracts | Supply chain, disclosure |
| Documentation (164.316) | Policies, documentation, review | Repudiation, accountability |

NIST SP 800-66 Rev. 2, *Implementing the HIPAA Security Rule*, is the standard implementation
guide. It is guidance, not regulation, but it is the reference an OCR reviewer is likely to use.
HHS OCR also publishes a Security Risk Analysis guidance page and a set of frequently asked
questions that clarify the scope: the analysis must cover all ePHI the entity creates, receives,
maintains, or transmits, regardless of format.

### NIST risk and threat modeling guidance

- NIST SP 800-30 Rev. 1, *Guide for Conducting Risk Assessments* (2012), defines the risk
  assessment process: prepare, conduct, communicate, maintain. It separates threat sources, threat
  events, vulnerabilities, and impact, which is the vocabulary the rest of this topic uses.
- NIST SP 800-154 (Draft), *Guide to Data-Centric System Threat Modeling* (2016), argues that
  threat modeling should start from the data and its flows rather than from the system boundary.
  The draft status matters: it is influential but not a final standard.
- NIST Cybersecurity Framework 2.0 (2024) organizes outcomes into Govern, Identify, Protect,
  Detect, Respond, and Recover, and is voluntary for most organisations.
- MITRE ATT&CK is a catalogue of adversary tactics and techniques. It is useful for mapping a
  threat to observed behaviour and for testing detection coverage, but it is not a risk method; it
  describes how attacks are conducted, not how likely they are in a given system.

### STRIDE and attack trees

STRIDE was introduced by Microsoft in the late 1990s and popularized through the Security
Development Lifecycle and Adam Shostack's *Threat Modeling: Designing for Security* (2014). It is a
convention, not a standard, and it has known limits: it does not cover privacy harms directly, it
does not rank threats, and it can produce long lists without prioritisation. The element-mapping
discipline in Section 5 is the mitigation.

Attack trees were described by Bruce Schneier in 1999. They are a modelling technique, not a
certification. Their probabilities are estimates, and a tree should state whether the values come
from data, expert judgement, or are illustrative.

### Medical devices and AI

- The FDA's premarket cybersecurity guidance for medical devices (final, September 2023) expects a
  security risk assessment, a threat model, and a software bill of materials (SBOM) as part of the
  premarket submission. The threat model is therefore a regulatory artefact for device makers, not
  only an engineering practice.
- IEC 81001-5-1:2021, *Health software and health IT systems safety, effectiveness and security -
  Part 5-1: Security in the lifecycle*, specifies security activities across the lifecycle for
  health software.
- ISO 14971:2019, *Application of risk management to medical devices*, provides the risk management
  process that a device threat model feeds into. The two are complementary: ISO 14971 covers harm
  to the patient from device failure, and the threat model covers deliberate causes.
- In the EU, MDCG 2019-16 provides cybersecurity guidance for medical devices under the Medical
  Device Regulation. It is guidance from the Medical Device Coordination Group, not a regulation
  itself.
- The EU AI Act (Regulation (EU) 2024/1689) requires high-risk AI systems to be resilient to
  errors, faults, and inconsistencies, and to be secure against attempts to exploit vulnerabilities
  (Article 15). A threat model is the natural evidence for that requirement, though the Act does
  not prescribe a method.

### What the standards do not require

None of the standards in this section prescribes a specific threat-modeling method, a notation, or
a scoring scale. The HIPAA risk analysis requires an accurate and thorough assessment; NIST SP
800-30 describes a process; STRIDE and attack trees are conventions. That freedom is useful and
dangerous: a team can choose a method that fits its system, and it can also produce a document that
satisfies the form and not the substance. The test is whether the model contains specific paths,
named controls, and evidence, because those are the elements a reviewer can evaluate regardless of
the method.

> **Note:** Regulatory content in this file is educational and is not legal advice. Whether a
> particular threat model satisfies a regulator depends on the system, the jurisdiction, and the
> evidence produced.

The classification to keep straight: the HIPAA risk analysis is a legal requirement; NIST SP 800-30
and SP 800-66 are guidance that supports compliance; STRIDE and attack trees are modelling
conventions; and a penetration test is evidence about a point in time, not a threat model.

## 9. Industry practice and case evidence

### Change Healthcare, 2024

The Change Healthcare ransomware incident is the most instructive US case for this topic because
the initial access was mundane. According to HHS OCR's incident FAQ and UnitedHealth Group
statements, the attacker entered through a Citrix remote-access portal that did not have
multi-factor authentication, using compromised credentials. The intrusion was detected on
21 February 2024, and the attacker moved laterally, exfiltrated about six terabytes of data, and
deployed ransomware. The breach report was revised several times, reaching 192.7 million
individuals in the filing updated in July 2025. UnitedHealth estimated the total cost at
approximately USD 3.09 billion.

Three threat-model lessons follow. First, the portal was a trust-boundary crossing with a single
authentication factor, which is a Spoofing threat that a DFD would have placed at the top of the
list. Second, the attacker's lateral movement showed that the internal network was treated as
trusted, which is the perimeter failure described in Section 1. Third, the exfiltration of six
terabytes implies that egress was not monitored at a volume that would have raised an alert, which
is a Detective control gap rather than a Preventive one.

### WannaCry and the NHS, 2017

The UK National Audit Office investigation found that at least 80 of 236 NHS trusts were affected,
including 34 that were locked out of devices, and 603 primary care and other organisations were
infected, including 595 general practices. NHS England estimated that 19,494 appointments were
cancelled. The NAO found no evidence that patient data were compromised and no organisation paid
the ransom. It also found that none of the 88 trusts given on-site assessments had passed the
cyber guidance, and that the attack was relatively unsophisticated and could have been prevented
by basic IT security practice.

The threat-model lesson is about availability and about the update channel. A ransomware worm
exploits unpatched systems; the control is patch management, which is a preventive control that
most organisations know about and many fail to verify. The clinical impact was not a data breach
but cancelled care, which is why the asset ranking in Section 5 includes availability and patient
safety, not only confidentiality.

### Anthem, 2015

The Anthem breach, discussed in Topic 01, began with spear-phishing against an Anthem subsidiary
and resulted in the theft of data for 78.8 million people. OCR's findings included failure to
conduct an enterprise-wide risk analysis and insufficient technical access controls. For threat
modeling, the case is a reminder that the human entry point is part of the attack surface and that
an enterprise risk analysis must cover subsidiaries and business units, not only the core system.

### What mature programmes do

- **Model at design, not at audit.** Microsoft's Security Development Lifecycle and the FDA's
  premarket expectations both place threat modeling before implementation, when the cost of a
  design change is low.
- **Maintain a threat library.** Reusable threat patterns for common architectures (FHIR APIs,
  device gateways, model endpoints, data lakes) reduce the time to a first draft and improve
  coverage. The library is a starting point, not a substitute for system-specific analysis.
- **Tie every control to a test.** The strongest programmes maintain a control-to-test matrix and
  report coverage, not a list of controls.
- **Review on change.** New interfaces and new vendors are the triggers that matter most; a
  calendar-only review misses them.

**The 405(d) Health Industry Cybersecurity Practices.** The US HHS 405(d) programme publishes the
Health Industry Cybersecurity Practices (HICP), a set of five threats and ten practices intended
for small, medium, and large organisations. It is not a threat-modeling method, but it is a useful
cross-check: if the model's threat list does not include the threats the sector programme
considers most consequential (social engineering, ransomware, loss or theft of equipment,
insider accidental or intentional data loss, and attacks against connected medical devices), the
model is probably incomplete. The HICP is maintained by a public-private task group and is updated
periodically; the current edition should be checked rather than assumed.

**Penetration tests are not threat models.** A penetration test finds exploitable weaknesses at a
point in time; a threat model asks what paths exist and whether the controls cover them. The two
are complementary, and the order matters: a model tells the tester where to look, and the test
tells the model whether a control works. An organisation that buys only penetration tests tends to
fix the findings and leave the unmodelled paths untouched.

> **Note:** The case facts above are drawn from the cited public reports. Statements about which
> control would have prevented an incident are inference, labelled as such in the text.

### Where practice is weaker than the documents suggest

Three gaps recur in assessments, and they are inference from published incident reviews rather than
from a single study. First, threat models cover the application and not the identity provider, the
integration engine, or the backup system, which are often the paths with the broadest access.
Second, controls are listed without an owner, so remediation stalls. Third, the model is not
updated when a vendor or interface changes, so the document describes a system that no longer
exists. Each gap is detectable: ask for the diagram date, the control owners, and the last change
that triggered a review.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Generic risk register | Rows say "unauthorised access" with no path | No actor, element, or control | No design change results | Threat statements with actor and path |
| Perimeter-only model | Diagram ends at the firewall | Internal traffic trusted | Lateral movement succeeds | Trust boundaries inside the network |
| Missing identity provider | DFD has the app but not the IdP | Identity is the real boundary | Credential compromise bypasses controls | Include IdP and federation |
| Vendor as a black box with no assumptions | Update channel drawn but not analysed | Supply-chain path unmodelled | Malicious update compromises all | Signed builds; provenance; vendor assessment |
| Control without a test | "Access is restricted" | No evidence of configuration | Control drifts or was never enabled | Control-to-test matrix |
| Score without a scale | Risk "high" with no inputs | Thresholds adjusted after the fact | Prioritisation is political | Define scale before scoring |
| Insider omitted | Actors are all external | Legitimate access not modelled | Bulk export or misuse undetected | Include negligent and malicious insiders |
| Availability omitted | Assets ranked by confidentiality | Ransomware and outages unmodelled | Care disruption | Add patient-safety and availability dimensions |
| Model-only analysis | Robustness tested; pipeline not | The weakest link is elsewhere | Data or endpoint compromise | Model the system, not only the model |
| No review trigger | Review date is annual only | New interface or vendor not assessed | Stale model | Change-triggered review |
| Probabilities presented as measurements | No basis for 0.18 or 0.35 | False precision | Wrong control prioritisation | Label basis and run sensitivity |
| Repudiation ignored | Shared accounts; no per-user logs | Actions cannot be attributed | Investigation fails | Per-user identity; audit |
| Privacy threats excluded | No linkage or inference rows | Security model only | Re-identification unaddressed | Add privacy threat rows |
| Acceptance without a name | Residual risk "accepted by the team" | No accountable owner | Risk persists unnoticed | Named acceptor and date |
| Identity provider out of scope | The IdP is not on the diagram | The real trust boundary is unmodelled | Forged assertions bypass authorisation | Include IdP and assertion flows |
| Workaround not modelled | Staff share a login to save time | Usability defeats the control | Unattributable access | Model workarounds; fix the workflow |

### Reading the failure table

The first four rows are model-quality failures: the document exists but does not describe a system
that can be attacked. The middle rows are coverage failures: a real path is not on the diagram. The
last rows are process failures: the model is not owned, tested, or maintained. In an assessment,
model-quality failures are the cheapest to fix and the most common; coverage failures require a
second review with an engineer who knows the integration layer.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The DFD matches the deployed system | All analysis depends on it | Drift after deployment | Compare to infrastructure inventory |
| The actor list covers the plausible attackers | Likelihood estimates depend on it | New actor or insider scenario | Red-team review; incident review |
| Probabilities have a stated basis | Scores are only as good as inputs | Guesswork presented as data | Record basis; sensitivity analysis |
| Paths are independent | OR arithmetic assumes it | Same attacker uses multiple paths | Treat as approximation; scenario analysis |
| Controls work as described | Residual risk depends on them | Misconfiguration or drift | Control-to-test matrix |
| The scope boundary is stable | Review triggers depend on it | New interface or vendor | Change log review |
| Impact ranking reflects patient harm | Prioritisation depends on it | Ranking by record count | Clinical review of asset list |
| Insiders are modelled | They are a frequent path | Model focuses on external actors | Include negligent and malicious roles |
| Privacy threats are in scope | Linkage and inference are real | Security-only framing | Add privacy rows and review |
| Vendors disclose their architecture | Supply-chain analysis depends on it | Black box with no evidence | Contractual security questionnaire |
| The model is reviewed on change | Staleness is the main failure | Calendar-only review | Trigger list and audit |

Two limits are structural. First, a threat model is a point-in-time argument; it cannot prove the
absence of a path, only that the identified paths are controlled. Second, the probabilities are
usually judgemental, and a model that hides that fact is less honest than one that labels it. The
value of the arithmetic is in comparing paths and controls, not in predicting incidents.

## 12. Exercises

The eight exercises are sequenced so that each produces an artefact the next one consumes: the
diagram from Exercise 1 feeds the STRIDE table in Exercise 2, the table feeds the attack tree in
Exercise 3, and the tree feeds the scoring in Exercise 4. Exercises 5 to 8 apply the same
artefacts to controls, insiders, vendors, and maintenance. If you do only one, do Exercise 1; a
correct diagram with labelled boundaries is the part that most often changes a design.

**Exercise 1 - Draw a DFD (application, 40 minutes).**
Draw a data-flow diagram for a remote patient monitoring system with a home device, a mobile app,
a cloud ingestion service, a clinician dashboard, and a vendor analytics service. Mark trust
boundaries. Deliverable: one diagram with a legend. Criteria: five element types present; at least
four boundaries; every flow labelled with protocol and authentication.

**Exercise 2 - STRIDE per element (application, 45 minutes).**
Apply STRIDE to each element of your diagram and record at least twelve credible threats. Deliverable:
threat table. Criteria: each row has element, category, actor, and a specific path; at least one
repudiation and one denial-of-service row.

**Exercise 3 - Build an attack tree (calculation, 45 minutes).**
Build a tree for "attacker alters a dose recommendation" with at least two paths and one AND node.
Assign probabilities with a stated basis and compute the total likelihood. Deliverable: tree diagram
plus arithmetic. Criteria: AND/OR logic correct; OR computed as one minus the product of
complements; basis stated.

**Exercise 4 - Score and rate (application, 30 minutes).**
Using the scale in Section 5, score three threats from Exercise 2 and assign a rating and action.
Deliverable: scoring table. Criteria: scale stated; impact justified by patient harm; action
consistent with the rating table.

**Exercise 5 - Control mapping (design, 45 minutes).**
For five threats, propose one preventive, one detective, and one responsive control, and a test for
each. Deliverable: control matrix. Criteria: controls are specific; tests are repeatable; no control
is listed without an owner.

**Exercise 6 - Insider scenario (critique, 40 minutes).**
A clinician with legitimate access searches for a celebrity's record. Model the threat, the
controls that would detect it, and the privacy harm. Deliverable: one-page analysis. Criteria:
distinguishes access from authorisation; names audit and break-glass controls; addresses
notification.

**Exercise 7 - Review a vendor claim (critique, 45 minutes).**
A vendor says its device is "secure by design" and provides a penetration test from 2023. List the
questions you would ask and the evidence you would require. Deliverable: questionnaire with ten
items. Criteria: covers threat model, SBOM, patch cadence, incident history, and update signing.

**Exercise 8 - Change trigger (design, 30 minutes).**
Write the review trigger list for the system in Section 6 and the first three actions when a
trigger fires. Deliverable: one page. Criteria: at least six triggers; each has an owner and an
action; includes vendor and interface changes.

## 13. Self-check answers

**Answer 1.** A threat is a potential cause of harm described by an actor, a capability, and a
path; a vulnerability is a weakness that a path exploits. A stolen credential is a threat path even
when no software vulnerability exists, and an unpatched server is a vulnerability even when no
actor is currently targeting it. Risk combines the likelihood that a threat path is attempted and
succeeds with the impact if it does. A control can act on any of the three: it can remove the
actor's capability (MFA), close the vulnerability (patching), or reduce the impact (segmentation,
backups). Keeping the terms separate matters because a risk register that lists vulnerabilities as
threats tends to produce controls that are patches rather than designs.

**Answer 3.** A correct tree has a root goal, at least two branches, and at least one AND node. The
arithmetic for an AND path is the product of the edge probabilities; for an OR node it is one minus
the product of the complements. For example, a tree with paths 0.05 and 0.08 gives
1 - (0.95 x 0.92) = 1 - 0.874 = 0.126. A tree that adds the path probabilities (0.13) is wrong,
because it double-counts the case where both paths succeed. The basis for each probability should
be stated as data, expert judgement, or illustrative.

**Answer 4.** The score is likelihood x impact x 10. A threat with annual likelihood 0.08 and
impact 4 scores 3.2, which is Medium: remediate in the next cycle and monitor. A threat with
likelihood 0.3 and impact 5 scores 15, which is Critical: do not deploy without redesign or
executive acceptance. The action must follow the rating table that was fixed before scoring; if the
threshold moves after the score is known, the exercise has no integrity.

**Answer 5.** Rank by patient harm first: a dosing recommendation or a device command can injure a
patient immediately, while a leaked appointment list harms privacy and trust. Then rank privacy,
legal and regulatory, and operational harm. The ranking is not a single number; an asset can be low
on privacy and high on patient safety, and the highest dimension drives the modelling effort.
Record counts are a poor proxy because a small genetic dataset can be more harmful than a large
appointment dataset. The ranking should be reviewed by a clinician, not only by the security team.

**Answer 6.** The clinician has legitimate access, so the threat is not unauthorised entry but
unauthorised purpose. The preventive control is purpose-based or relationship-based authorisation
where feasible, and break-glass access with a reason code where it is not. The detective control is
audit review of access to VIP or high-profile records, with alerts on searches that do not
correspond to an appointment or referral. The responsive control is an investigation, notification
to the privacy office, and, where required, notification to the individual. The privacy harm is
disclosure of sensitive information and the loss of trust; the security harm is reputational and
regulatory. A model that only asks "who can log in" misses this entirely.

**Answer 7.** The evidence is a test with a result and a date: a configuration probe that shows the
gateway rejects unauthenticated requests, a log query that shows object-level authorisation
denials, a penetration-test report with scope and remediation status, or a tabletop exercise with
actions. A policy statement is evidence that a decision was made, not that a control operates. The
distinction matters in an audit: the auditor will ask for the last test result, not the paragraph
in the standard operating procedure.

**Answer 8.** Triggers: a new external interface; a new vendor or subprocessor; a new data type
(genetic, biometric, free text); a material architecture change (new cloud region, new identity
provider); a relevant incident in a peer organisation; a change in the regulatory classification;
and the annual review date. Actions when a trigger fires: update the DFD and the asset list; re-run
STRIDE on the changed elements; re-score the affected threats; and update the controls and tests.
Each action has an owner and a date.

**Rubric for Exercise 1.** Full credit requires all five element types, at least four labelled
boundaries, protocols and authentication on every flow, and a legend. A network diagram without
processes and stores does not pass.

**Rubric for Exercise 2.** Full credit requires twelve credible rows with specific paths, at least
one repudiation and one denial-of-service row, and a note on any category with no credible threat.
A list of categories without paths does not pass.

**Rubric for Exercise 5.** Full credit requires three control types per threat, a repeatable test
for each, and an owner. A control that cannot be tested is marked as an assumption, not a control.

**Rubric for Exercise 7.** Full credit requires questions on the threat model, SBOM, patch cadence,
incident history, update signing, penetration-test scope and date, data flows, subprocessors,
incident notification, and audit rights. A questionnaire that asks only for a certificate does not
pass.

## 14. Interview questions

**1. What is the difference between a threat and a vulnerability?**
Strong answer: a threat is an actor with a capability and a path; a vulnerability is a weakness the
path exploits; a threat can exist without a known vulnerability (credential theft) and a
vulnerability without an actor. Weak answer: uses the terms interchangeably.

**2. Why apply STRIDE per element?**
Strong answer: the same category produces different threats and controls at an entity, a process, a
store, and a flow; per-element application gives coverage and specificity. Weak answer: "to cover
all categories".

**3. How do you compute the probability of an attack tree with OR nodes?**
Strong answer: one minus the product of the complements of the path probabilities, because the
paths may co-occur; AND nodes multiply. Weak answer: sums the paths.

**4. What is a trust boundary and where do you draw them?**
Strong answer: a line where the level of trust changes, including network, organisation, role,
identity provider, and vendor; draw them inside the perimeter too. Weak answer: "the firewall".

**5. How do you rank assets in a clinical system?**
Strong answer: by patient harm first, then privacy, legal, and operational; availability and
integrity matter as much as confidentiality. Weak answer: by record count.

**6. How do you model the insider threat?**
Strong answer: separate negligent and malicious insiders, model legitimate access paths, bulk
export, shared accounts, and audit gaps; controls are least privilege, per-user identity, and
detection. Weak answer: "insiders are trusted".

**7. What evidence shows a control works?**
Strong answer: a repeatable test with a result and a date, such as a configuration probe, a
penetration test, a log query, or a tabletop exercise. Weak answer: "it is in the policy".

**8. How does threat modeling relate to the HIPAA risk analysis?**
Strong answer: the threat model is an input to the risk analysis required by 164.308(a)(1)(ii)(A);
the risk analysis adds likelihood, impact, and management decisions across the enterprise. Weak
answer: they are the same document.

**9. What would you do if the vendor will not share its architecture?**
Strong answer: treat it as a black box, record the assumption, require contractual security terms,
SBOM, and attestations, and test what can be tested from outside. Weak answer: exclude the vendor
from scope.

**10. How do you keep a threat model current?**
Strong answer: change triggers (interface, vendor, data type, architecture, incident, regulation)
plus an annual review, with an owner and a date. Weak answer: annual review only.

**11. How would you threat-model an LLM feature that reads clinical notes?**
Strong answer: models prompt injection through retrieved documents, tool access and allowlists,
data exfiltration through outputs, logging of prompts, and the model provider as a trust boundary;
adds evaluation with adversarial prompts and output validation. Weak answer: focuses only on the
model's accuracy.

**12. What would make you reject a threat model as inadequate?**
Strong answer: no diagram, no trust boundaries, generic threats, controls without tests, no
named acceptor, no review trigger, or probabilities presented without a basis. Weak answer: it is
too short.

## 15. Key terms

- **AND node** - all children must succeed in an attack tree.
- **Attack surface** - the set of entry points and exposed functionality.
- **Attack tree** - a goal tree with AND/OR logic and probabilities.
- **Control** - a mechanism that prevents, detects, or responds to a threat.
- **Data-flow diagram** - entities, processes, stores, flows, and trust boundaries.
- **Denial of service** - making a system unavailable.
- **Detective control** - detects a threat after it occurs.
- **Elevation of privilege** - gaining access beyond one's authorisation.
- **Exfiltration** - removing data from an environment.
- **Impact** - harm if a threat succeeds.
- **Information disclosure** - unauthorised access to data.
- **Insider threat** - a threat from someone with legitimate access.
- **Kill chain** - the stages of an attack.
- **Lateral movement** - moving between systems after compromise.
- **Likelihood** - probability that a path is attempted and succeeds.
- **MFA** - multi-factor authentication.
- **Negligent insider** - an authorised user who causes harm by error.
- **OR node** - any child suffices in an attack tree.
- **Preventive control** - stops a threat path.
- **Repudiation** - denying an action occurred.
- **Residual risk** - risk remaining after controls.
- **Risk** - likelihood combined with impact.
- **Risk score** - a number on a defined scale.
- **Ransomware** - malware that encrypts or steals data for extortion.
- **Service account** - a non-human identity used by a system.
- **Spoofing** - impersonating a user or system.
- **STRIDE** - Spoofing, Tampering, Repudiation, Information disclosure, Denial of service,
  Elevation of privilege.
- **Tampering** - unauthorised modification.
- **Threat actor** - a person or group with a goal and capability.
- **Threat model** - a structured analysis of what can go wrong and what controls reduce it.
- **Trust boundary** - a line where the level of trust changes.
- **Vulnerability** - a weakness a threat can exploit.
- **Data-centric threat modeling** - starting from data and flows rather than the system boundary.
- **Egress monitoring** - detecting unusual outbound data volume.
- **SBOM** - software bill of materials; the inventory of components in a build.
- **Object-level authorisation** - checking access to a specific record, not only a role.
- **Break-glass access** - emergency access with logging and review.
- **Threat library** - reusable threat patterns for common architectures.
- **Control-to-test matrix** - the mapping from each control to the evidence that it works.

## 16. Sources

All URLs accessed 26 October 2026.

**US regulation and guidance**
- 45 CFR 164.308, *Administrative safeguards, risk analysis*.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.308
- HHS Office for Civil Rights, *Security Risk Analysis*.
  https://www.hhs.gov/hipaa/for-professionals/security/guidance/cybersecurity/index.html
- HHS Office for Civil Rights, *Change Healthcare Cybersecurity Incident Frequently Asked
  Questions*. https://www.hhs.gov/hipaa/for-professionals/special-topics/change-healthcare-cybersecurity-incident-frequently-asked-questions/index.html
- FDA, *Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket
  Submissions*, final guidance, September 2023.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-system-considerations-and-content-premarket-submissions

**Frameworks and standards**
- NIST, *SP 800-30 Rev. 1, Guide for Conducting Risk Assessments*, 2012.
  https://csrc.nist.gov/pubs/sp/800/30/r1/final
- NIST, *SP 800-66 Rev. 2, Implementing the HIPAA Security Rule*, 2024.
  https://csrc.nist.gov/pubs/sp/800/66/r2/final
- NIST, *SP 800-154 (Draft), Guide to Data-Centric System Threat Modeling*, 2016.
  https://csrc.nist.gov/pubs/sp/800/154/draft
- NIST, *Cybersecurity Framework 2.0*, 2024. https://www.nist.gov/cyberframework
- MITRE, *ATT&CK*. https://attack.mitre.org/
- ISO 14971:2019, *Application of risk management to medical devices*.
- IEC 81001-5-1:2021, *Health software and health IT systems safety, effectiveness and security -
  Part 5-1: Security in the lifecycle*.
- MDCG, *MDCG 2019-16 Guidance on Cybersecurity for medical devices*, 2019.
  https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en

**Case evidence**
- National Audit Office, *Investigation: WannaCry cyber attack and the NHS*, 27 October 2017.
  https://www.nao.org.uk/reports/investigation-wannacry-cyber-attack-and-the-nhs/
- NHS England, *Lessons learned review of the WannaCry ransomware cyber attack*, February 2018.
  https://www.england.nhs.uk/aac/wp-content/uploads/2018/02/lessons-learned-review-wannacry-ransomware-cyber-attack-cio-review.pdf
- HHS OCR, *Anthem Resolution Agreement*, 2018.
  https://www.hhs.gov/hipaa/for-professionals/compliance-enforcement/agreements/anthem/index.html
- UnitedHealth Group, *Change Healthcare cybersecurity incident updates*, 2024-2025.
  https://www.unitedhealthgroup.com/changehealthcare.html

**Methods**
- Shostack, A. *Threat Modeling: Designing for Security*. Wiley, 2014.
- Schneier, B. *Attack Trees*. Dr. Dobb's Journal, 1999.
  https://www.schneier.com/academic/archives/1999/12/attack_trees.html

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_privacy-enhancing-technologies.md`](03_privacy-enhancing-technologies.md)
- Next topic: [`05_access-control-and-cryptography.md`](05_access-control-and-cryptography.md)
- Related topic: [`09_breach-and-incident.md`](09_breach-and-incident.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
