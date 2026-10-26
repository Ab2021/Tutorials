# Model Governance

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 09 of 10
> Estimated study time: 3 hours
> Prerequisites: 03, 04
> Version 2026.3

## 1. Why this topic exists

Governance is the set of decisions, authorities, and records that make a model accountable. It is
not a committee, a policy document, or a spreadsheet. It is the answer to four questions that every
organization eventually asks: what models do we have, who owns each one, what evidence supports its
use, and who approved the last change.

Most organizations cannot answer the first question. Models are built in notebooks, piloted by
departments, embedded in vendor products, and deployed by teams that have since reorganized. The
inventory is incomplete, the owners are stale, and the evidence is scattered across email, shared
drives, and a business intelligence tool that nobody remembers commissioning.

The gap becomes visible in three situations: an audit, an incident, and a merger. In each, the
organization needs to produce a list, an owner, and a record, quickly. A team that cannot produce
them is not necessarily unsafe; it is unaccountable, which in a regulated setting is treated as
unsafe.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of weak governance |
|---|---|---|
| Which models require the most oversight | Quality, clinical leadership | Uniform process that over-controls low-risk models and under-controls high-risk ones |
| Who can approve a model change | Regulatory, engineering | Changes approved by people without authority |
| What evidence is required before use | Clinical safety, quality | A model goes live with a validation report nobody can find |
| How to handle a vendor model | Procurement, security | Third-party risk is invisible |
| When a model should be retired | Product, finance | Zombie models accumulate cost and risk |
| How to demonstrate control to an auditor | Quality, legal | Audit findings and remediation cost |

### 1.2 Governance is a control plane

The production loop is the data plane: it moves data, produces predictions, and acts on them.
Governance is the control plane: it decides which models may run, under what conditions, with what
evidence, and who is accountable. The control plane is slower and less visible, and it is what
makes the data plane defensible.

A control plane has three registers:

- **Inventory**: what exists, where it runs, and who owns it.
- **Risk file**: what can go wrong, how likely, how severe, and what controls mitigate it.
- **Change log**: what changed, when, why, who approved it, and what evidence supported it.

If any register is missing, the other two cannot be trusted. An inventory without owners is a list.
A risk file without an inventory is a document about nothing. A change log without a risk file
cannot judge whether a change was material.

### 1.3 The regulatory shift toward lifecycle governance

The FDA's Quality Management System Regulation, effective 2 February 2026, incorporates
ISO 13485:2016 by reference and aligns the device quality system with the international standard.
The regulation adds supplemental requirements for records, complaints, and labeling. The practical
effect is that governance is not a separate layer bolted onto engineering; it is the quality system
applied to the model lifecycle.

The FDA's AI-enabled device guidance, still in draft as of October 2026, recommends a lifecycle
approach covering design, validation, monitoring, and change. The predetermined change control plan (PCCP) guidance provides the
mechanism for pre-authorized change. Together they describe a governance system, not a one-time
approval.

### 1.4 The cost of poor governance

The cost is visible in audit findings, incident response time, and duplicated effort. An auditor
who cannot find the owner of a model will write a finding; the remediation consumes months. An
incident responder who cannot find the validation report will investigate from scratch. A merger
that discovers 40 undocumented models will spend a year cataloguing them.

The cost is also strategic. An organization that can produce an inventory, an owner, and a record
can adopt new models faster, because the review is a known process rather than a negotiation. Good
governance is a speed enabler, not a brake, when it is proportionate to risk.

### 1.5 Governance and speed

A common objection is that governance slows delivery. The objection is true when the process is
uniform and manual: every model waits for the same review, regardless of risk. It is false when the
process is tiered and automated: a tier 4 model is approved in a day, and a tier 1 model gets the
scrutiny it needs.

The speed benefit is largest in the second and third deployments. The first model establishes the
evidence set and the approval path. The next model reuses them, and the review becomes a check
against a known standard rather than a negotiation about what evidence is required.

### 1.6 Who owns governance

Governance is owned by the accountable executive for the model portfolio, supported by a quality
function and a model risk function. It is not owned by the data science team, because the team
cannot audit itself, and it is not owned by IT, because IT does not own the clinical decision.

In a small organization, one person may hold several roles. The minimum is that the person who
approves a model is not the person who built it, and the person who audits the system is not the
person who operates it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** a model inventory with the fields needed for ownership, risk, and audit.
2. **Assign** a risk tier using intended use, population, autonomy, and reversibility.
3. **Define** an approval matrix that matches authority to risk.
4. **Specify** the evidence set required before use and after each change.
5. **Design** a change-control process that links to the PCCP and the quality system.
6. **Apply** third-party governance to vendor and open-source models.
7. **Audit** a governance system for coverage, freshness, and segregation of duties.

## 3. The core mental model

Governance is a control plane with three registers and a decision path that runs through them.

```text
                        ┌──────────────────────────┐
                        │      MODEL INVENTORY     │
                        │ id, owner, tier, version │
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌────────────────┐    ┌──────────────────┐    ┌────────────────┐
     │   RISK FILE    │    │   APPROVAL       │    │  CHANGE LOG    │
     │ hazards,       │    │   MATRIX         │    │ what, why,     │
     │ controls,      │    │ who signs what   │    │ who, evidence  │
     │ residual risk  │    │ at which tier    │    │                │
     └────────┬───────┘    └────────┬─────────┘    └────────┬───────┘
              │                     │                       │
              └─────────────────────┼───────────────────────┘
                                    ▼
                        ┌──────────────────────────┐
                        │   EVIDENCE REPOSITORY    │
                        │ validation, monitoring,  │
                        │ incidents, decisions     │
                        └──────────────────────────┘
```

### 3.1 The inventory is the anchor

Every other control depends on the inventory. If a model is not in the inventory, it has no owner,
no tier, no risk file, and no change log. It is invisible to governance and visible only to its
users.

The inventory must be generated and reconciled, not hand-maintained. A spreadsheet that is updated
when someone remembers is stale within a quarter. A better design links the inventory to the
deployment system, the model registry, and the procurement system, and reconciles discrepancies.

The minimum fields are: identifier, name, intended use, owner, accountable executive, risk tier,
model version, deployment pattern, sites, data sources, validation date, last review date, status,
and retirement date. Additional fields support the risk file and the change log.

### 3.2 Risk tiering

A tier determines the controls. A tiering scheme that is too coarse applies the same process to a
scheduling model and a sepsis model. A scheme that is too fine is unusable.

A workable four-tier scheme uses four dimensions:

| Dimension | Low | Medium | High |
|---|---|---|---|
| Intended use | Operational efficiency | Clinical support with human review | Clinical decision or autonomous action |
| Population | Narrow, well characterized | Broad, heterogeneous | Vulnerable or underrepresented groups |
| Autonomy | Advisory, no action | Recommendation with override | Action with no human in the loop |
| Reversibility | Easily reversed | Reversible with effort | Irreversible harm possible |

The tier is the highest level across the dimensions, not the average. A model that is advisory but
affects a vulnerable population is high tier.

### 3.3 Segregation of duties

The person who builds a model should not be the only person who approves it. The person who
approves it should not be the only person who audits it. Segregation of duties is a control, not a
sign of distrust.

In a small organization, perfect segregation is impossible. The minimum is a second reviewer who
is not the author, and an annual audit by someone outside the model team. The reviewer's name and
the evidence they saw must be recorded.

### 3.4 The evidence repository

The evidence repository is where the validation report, the model card, the monitoring record, the
incident reports, and the change approvals live. It must be searchable by model identifier and
retained for the audit period.

A repository that is a shared drive with folders named after people is not a repository. A
repository that is a document management system with version control and access logs is.

### 3.5 Governance proportionality

The controls should be proportional to the tier. A tier 4 model might need an annual review and a
model card. A tier 1 model might need independent validation, a risk file, continuous monitoring,
a PCCP, and a quarterly review by a committee.

Proportionality is what makes the system survivable. A governance system that requires the same
evidence for every model will be bypassed for the low-risk ones and will collapse under its own
weight for the high-risk ones.

### 3.6 The governance calendar

Governance runs on a calendar even when nothing is wrong. The calendar should include: monthly
tier 1 reviews, quarterly tier 2 reviews, annual tier 3 and 4 reviews, an annual inventory
reconciliation, an annual audit, and a quarterly review of open corrective actions.

A calendar is a control because it makes the absence of a review visible. A model that has not been
reviewed on schedule is a finding, even if no incident has occurred.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Governance | The decisions, authorities, and records that make a model accountable | A committee | None |
| Model inventory | A register of models with owner, tier, version, and status | A spreadsheet of projects | Count |
| Model owner | The role accountable for the model's lifecycle | The author | Role |
| Accountable executive | The leader who accepts the residual risk | The model owner | Role |
| Risk tier | A classification that determines the controls | The model's accuracy | Tier 1 to 4 |
| Model card | A structured summary of intended use, data, performance, and limits | A README | Document version |
| Independent validation | Validation by someone who did not build the model | A second look by the author | Report identifier |
| Change control | The process for approving, recording, and reversing changes | A ticket | Record identifier |
| PCCP | A pre-authorized description of permitted modifications | A general permission to update | Submission identifier |
| Corrective and preventive action (CAPA) | A change that reduces recurrence or impact | Any fix | Record identifier |
| Quality management system | The organizational system for quality, including processes and records | A document folder | Standard reference |
| QMSR | FDA Quality Management System Regulation, effective 2026 | The old QSR | 21 CFR Part 820 |
| Design history file | The record of design and development for a device | A project folder | DHF identifier |
| Technical file | The evidence package for a device in some jurisdictions | The DHF | File identifier |
| Post-market surveillance | Systematic monitoring after release | Complaint handling | Plan identifier |
| Audit trail | A record of who did what and when | A log file | Event record |
| Segregation of duties | Separating author, approver, and auditor | A nice-to-have | Control |
| RACI | Responsible, accountable, consulted, informed | A chart nobody reads | Role matrix |
| Third-party risk | Risk from vendors, open-source, and partners | A procurement checkbox | Assessment identifier |
| Model risk management | The discipline of managing model risk, formalized in banking | A synonym for governance | Framework |
| Attestation | A formal statement that a control is in place | A signature | Statement |
| Decommissioning | Planned removal of a model from service | Deleting a file | Record identifier |
| Zombie model | A model still running with no owner or use | An old model | Count |
| Shadow model | A model deployed outside the governance process | A shadow deployment | Count |
| Material change | A change that alters intended use, performance, or risk | Any code change | Determination |
| Evidence pack | The set of documents supporting a decision | A folder | Pack identifier |
| Traceability | The ability to link a requirement to a test and a record | A document property | Link |
| Transparency | Clear information about a model's use and limits | A marketing page | Disclosure |

Three distinctions matter most.

**Governance is not approval.** Approval is one decision. Governance is the system that defines who
approves what, with what evidence, and what happens after. A model can be approved and still be
ungoverned if there is no owner, no monitoring, and no change record.

**A risk tier is not a quality score.** A highly accurate model can be high tier because of its
intended use. A mediocre model can be low tier because it is advisory and easily reversed. The tier
describes the consequence of failure, not the probability.

**An inventory is not a portfolio.** A portfolio is a set of investments. An inventory is a set of
accountable assets. A portfolio review asks whether to invest more; an inventory review asks
whether each asset is owned, evidenced, and still in use.

## 5. The mechanics, step by step

The procedure builds a governance system that is proportionate, auditable, and usable. It starts
with scope, because a governance system that tries to cover everything at once collapses.

### Step 1 — Scope the governance system

**Input.** The organization's model landscape, legal entities, and regulatory status.

**Output.** A written scope: which models, which sites, which entities, and which exclusions.

**Rationale.** Scope determines effort. A hospital that governs every spreadsheet formula will
fail; a hospital that governs only FDA-cleared devices will miss the clinical models that influence
care without being devices. The scope should be defined by the decision the model affects, not by
the technology.

A workable starting scope: any model that influences a clinical, operational, or financial decision
about a patient or a population, including vendor models, regardless of whether it is a regulated
device.

Branch point: if a model is used only for research and never for a decision, it may be out of
scope, provided the exclusion is documented and the model cannot be repurposed without review.

### Step 2 — Define risk tiers

**Input.** The scope and the risk appetite.

**Output.** A tiering scheme with criteria, examples, and a rule for ambiguous cases.

**Rationale.** The tier determines the controls. The scheme should be simple enough to apply
consistently and specific enough to distinguish a scheduling model from a sepsis model.

The four dimensions in Section 3.2 are a starting point: intended use, population, autonomy, and
reversibility. The tier is the highest level across the dimensions.

Branch point: if a model's tier is disputed, assign the higher tier and review. A dispute about
tier is usually a dispute about consequence, and the conservative answer is safer.

### Step 3 — Build and reconcile the inventory

**Input.** The deployment systems, the model registry, the procurement system, and interviews.

**Output.** An inventory with the minimum fields, reconciled against at least two sources.

**Rationale.** The inventory is the anchor for every other control. A hand-maintained list is stale
within a quarter. Reconciliation finds the models that were deployed outside the process, which are
the ones that matter most.

Reconciliation sources: the container registry (what images are running), the electronic health record (EHR) integration
inventory (what models write into the chart), the procurement system (what was purchased), and the
cloud billing records (what compute is being used).

Branch point: if a model appears in a deployment system but not in the inventory, it is a shadow
model. It must be added and tiered before it can be governed, and the reason for the gap must be
recorded.

### Step 4 — Assign owners and accountable executives

**Input.** The inventory and the organizational structure.

**Output.** A named owner and accountable executive for every model.

**Rationale.** Ownership is the difference between a list and a control. The owner is accountable
for the lifecycle: validation, monitoring, changes, and retirement. The accountable executive
accepts the residual risk and can fund the work.

The owner should be a role, not a person, so that the ownership survives a departure. The role must
have the authority and the time to do the work; a nominal owner who is never available is not an
owner.

Branch point: if no one will accept ownership, the model should be retired or disabled. A model
without an owner is an unmanaged risk.

### Step 5 — Define the approval matrix

**Input.** The risk tiers and the organizational roles.

**Output.** A matrix: tier, change type, approver, evidence, and service level.

**Rationale.** The matrix makes approval predictable and prevents both rubber stamps and
bottlenecks. It should specify who approves what, what they see, and how long they have.

A workable matrix:

| Tier | New model | Retrain within protocol | Threshold change | Retirement |
|---|---|---|---|---|
| 1 | Clinical, quality, regulatory | Quality and clinical | Clinical and quality | Clinical and quality |
| 2 | Clinical and quality | Quality | Clinical | Clinical |
| 3 | Model owner and clinical reviewer | Owner | Owner | Owner |
| 4 | Model owner | Owner | Owner | Owner |

The matrix is a template. The roles vary by organization; the principle is that authority matches
consequence.

Branch point: if an approver is also the author, the approval is not independent. The matrix must
name a backup who is not the author.

### Step 6 — Define the evidence set per tier

**Input.** The tiers and the regulatory status.

**Output.** A list of required documents for each tier, before use and after each change.

**Rationale.** The evidence set is what the approver reviews and what the auditor samples. It
should be small enough to be produced and large enough to demonstrate control.

A tier 1 evidence set typically includes: intended use, model card, data provenance, validation
report with subgroup performance, risk analysis, human-factors assessment, monitoring plan,
rollback plan, and change-control record. A tier 4 set might include only the model card and a
validation summary.

Branch point: if the evidence cannot be produced for an existing model, the model is in a
remediation state. It should be reviewed, and either brought into compliance or retired.

### Step 7 — Define change control and materiality

**Input.** The tiers, the PCCP status, and the change history.

**Output.** A materiality rule and a change process for each tier.

**Rationale.** Not every change is material. A log message is not a model change; a threshold
change is. The materiality rule determines whether the change needs revalidation, approval, or
regulatory review.

The rule should be written as a decision tree, not a paragraph. It should cover code, model,
configuration, dependency, data source, and infrastructure changes.

Branch point: if a change is not described in the PCCP or the change protocol, treat it as
material until regulatory affairs says otherwise. The conservative default is the safe one.

### Step 8 — Define independent validation

**Input.** The tiers and the available expertise.

**Output.** A validation standard: who validates, against what criteria, and with what independence.

**Rationale.** Independent validation is the strongest evidence that a model is fit for use. It is
also the control most often weakened by convenience: the author validates, the reviewer signs, and
the independence is nominal.

The validator should not have built the model, should have access to the data and the code, and
should be able to fail the model without career consequences. The validation report should state
the criteria, the results, and the limitations.

Branch point: if the organization has no independent validator, external review or a peer-review
arrangement is needed for tier 1 models. Self-validation is not independent validation.

### Step 9 — Define monitoring and review cadence

**Input.** The tiers and the label latency.

**Output.** A monitoring plan and a review cadence per tier.

**Rationale.** Governance is not a one-time approval. The model must be reviewed on a cadence that
matches its risk and its label latency. A tier 1 model with monthly labels should be reviewed
monthly; a tier 4 model with annual review may be sufficient.

The review should have a fixed agenda: calibration, discrimination, subgroup performance, alert
burden, incidents, changes, and a disposition. A review without a disposition is a meeting.

Branch point: if the review finds a breach, the response follows the incident or drift process, not
the governance calendar. Governance sets the cadence; the monitoring sets the trigger.

### Step 10 — Define third-party and vendor controls

**Input.** The vendor contracts and the procurement process.

**Output.** A vendor assessment and a contract clause set for model updates, audit, and notification.

**Rationale.** A vendor model is still the deploying organization's risk. The vendor may update it
without notice, refuse an audit, or go out of business. The contract is the control.

The clauses that matter: notification of model changes, access to validation evidence, audit
rights, data-use restrictions, incident notification, and exit provisions including model and data
portability.

Branch point: if a vendor will not accept notification and audit clauses, the model is a
higher-risk dependency. The organization must either accept the risk explicitly or choose another
vendor.

### Step 11 — Define records, retention, and audit trail

**Input.** The evidence set and the regulatory retention requirements.

**Output.** A records schedule: what is kept, where, for how long, and who can access it.

**Rationale.** An audit is a sampling exercise. The auditor asks for the record and expects it
within a day. A record that is in a departed employee's folder does not exist for audit purposes.

The schedule should align with the quality system's retention rule, not the engineering team's
storage policy. HIPAA documentation retention is six years; device records have their own
requirements; the longer of the applicable periods governs.

Branch point: if the record contains protected health information, access must be restricted and
the retention must satisfy both the quality and privacy requirements.

### Step 12 — Define retirement and decommissioning

**Input.** The inventory and the usage data.

**Output.** A retirement process: decision, communication, data retention, and access removal.

**Rationale.** A model that is no longer used is still a risk if it is still running. Retirement is
a change with safety, privacy, and cost implications. It must be planned, not discovered.

The process should include: a decision record, a communication to users, a replacement plan, a
data-retention decision, and an access-removal step. The inventory status changes to retired, and
the evidence is retained.

Branch point: if the model is embedded in a vendor product, retirement may require a contract
change or a replacement product. That is a procurement action, not an engineering one.

### Step 13 — Train, audit, and improve

**Input.** The governance system and its records.

**Output.** A training program, an audit schedule, and a set of improvements.

**Rationale.** A governance system that is not audited decays. The audit should sample the
inventory, the evidence, and the approvals, and should report coverage and freshness.

The audit should be independent of the model teams. It can be internal audit, quality assurance, or
an external reviewer. The findings should feed a corrective action process with owners and dates.

Branch point: if the audit finds systemic gaps, the response is a system change, not a training
reminder. Training is appropriate for knowledge gaps; it is not a control for process gaps.

### 5.14 Tier-to-control matrix

| Control | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|
| Inventory entry | Required | Required | Required | Required |
| Named owner | Required | Required | Required | Required |
| Model card | Required | Required | Required | Recommended |
| Independent validation | Required | Required | Peer review | Self-validation |
| Risk analysis | Required | Required | Lightweight | Not required |
| Subgroup performance | Required | Required | Recommended | Not required |
| Monitoring plan | Required | Required | Basic | Basic |
| Review cadence | Monthly | Quarterly | Annual | Annual |
| Change approval | Clinical, quality, regulatory | Clinical, quality | Owner | Owner |
| Rollback test | Quarterly | Semi-annual | Annual | Not required |
| PCCP or equivalent | Required if device | Assess | Assess | Not required |

The matrix is a starting point. The organization should adjust it to its risk appetite and record
the rationale for any deviation.

### 5.15 Edge cases the procedure must handle

**Shadow models.** A model deployed outside the process appears in the registry. Add it, tier it,
assign an owner, and record the gap. Do not delete it without a replacement.

**Vendor updates.** A vendor changes a model without notice. The contract should require
notification; if it does not, the organization must monitor the vendor's release notes and validate
the change.

**Mergers and acquisitions.** A new entity brings its own inventory. Reconcile the two, map
duplicate models, and assign owners before integration.

**Research models.** A research model used for a publication may be out of scope, but a research
model used to prioritize patients is in scope. The distinction is the decision, not the label.

**Open-source models.** An open-source model has no vendor to notify. The organization owns the
validation, the monitoring, and the updates. The licence and the provenance must be recorded.

**Model retirement.** A model that is disabled but not removed still appears in the inventory. Mark
it retired and remove access; do not leave it in a zombie state.

### 5.16 The inventory schema

The inventory is a database, even if it is stored in a spreadsheet. The schema below is the
minimum for governance and audit.

| Field | Type | Required | Source | Purpose |
|---|---|---|---|---|
| Model identifier | String | Yes | Generated | Stable key |
| Name and description | String | Yes | Owner | Human identification |
| Intended use | Text | Yes | Owner | Scope and tiering |
| Owner role | String | Yes | Assignment | Accountability |
| Accountable executive | String | Yes | Assignment | Residual risk acceptance |
| Risk tier | Enum | Yes | Tiering | Control selection |
| Model version | String | Yes | Registry | Current deployed version |
| Deployment pattern | Enum | Yes | Architecture | Batch, application programming interface (API), edge, embedded |
| Sites and populations | List | Yes | Deployment | Envelope |
| Data sources | List | Yes | Data contract | Provenance |
| Validation date | Date | Yes | Validation report | Freshness |
| Last review date | Date | Yes | Review record | Cadence |
| Status | Enum | Yes | Lifecycle | Active, paused, retired |
| Retirement date | Date | No | Lifecycle | Decommissioning |

A schema with these fields supports the questions an auditor will ask. A schema with only name and
owner supports a list, not governance.

### 5.17 The materiality decision tree

Materiality determines whether a change needs revalidation, approval, or regulatory review.

```text
change
  │
  ├── Does it alter the intended use, population, or output meaning?
  │        └── yes ──▶ material; full review and regulatory assessment
  │
  ├── Does it alter model parameters, features, or threshold?
  │        └── yes ──▶ material; revalidation and tier approval
  │
  ├── Does it alter a dependency that can change numeric output?
  │        └── yes ──▶ assess with a parity test; material if outputs differ
  │
  ├── Does it alter infrastructure without changing output?
  │        └── yes ──▶ operational; standard change control
  │
  └── Does it alter documentation or logging only?
           └── yes ──▶ minor; record and notify
```

The tree is a starting point. The key discipline is that the assessment is recorded before the
change, not reconstructed after it.

## 6. Worked example

A regional health system has 38 models in its inventory: 6 in tier 1, 11 in tier 2, 15 in tier 3,
and 6 in tier 4. The governance team is preparing for an accreditation audit and has four weeks to
close gaps.

### 6.1 The cost of governance

The annual cost per model is estimated from the review hours, validation effort, and monitoring
support required at each tier.

| Tier | Models | Cost per model per year | Total |
|---|---:|---:|---:|
| 1 | 6 | 18,000 USD | 108,000 USD |
| 2 | 11 | 9,000 USD | 99,000 USD |
| 3 | 15 | 3,000 USD | 45,000 USD |
| 4 | 6 | 1,000 USD | 6,000 USD |
| **Total** | **38** | | **258,000 USD** |

The average cost is 258,000 / 38 = 6,789 USD per model per year. The average is misleading: a tier
1 model costs eighteen times a tier 4 model. A governance budget that is allocated per model rather
than per tier will underfund the high-risk models.

### 6.2 The review-currency objective

Tier 1 models are reviewed monthly. Six models × 12 reviews = 72 reviews per year. The objective is
that 90 percent of reviews occur within five business days of the scheduled date.

```text
allowed_late = 72 × (1 − 0.90) = 7.2
observed_late = 11
burn_rate = (11 / 72) / 0.10 = 1.53
```

The burn rate of 1.53 means the review process is missing its objective by about 50 percent. The
cause is not the review itself; it is the evidence assembly, which takes the owner two days per
review.

### 6.3 The audit sample

The audit samples 12 of 38 models, or 31.6 percent. The sample is stratified: all 6 tier 1 models,
4 tier 2, and 2 tier 3.

| Finding | Count | Percentage of sample | Affected tiers |
|---|---:|---:|---|
| No named owner | 5 | 41.7 percent | Tier 2 and 3 |
| Validation older than 12 months | 4 | 33.3 percent | Tier 2 and 3 |
| No rollback test in the last year | 3 | 25.0 percent | Tier 2 |
| Model card missing | 2 | 16.7 percent | Tier 3 |

The tier 1 models have owners and current validation, except one whose validation is 13 months old.
The gaps are concentrated in tier 2 and tier 3, which is where the process was least enforced.

### 6.4 The remediation arithmetic

The remediation cost is estimated from the work required to close each finding.

```text
owner assignment   = 5 × 3 h × 120 USD = 1,800 USD
validation         = 4 × 40 h × 95 USD = 15,200 USD
rollback test      = 3 × 8 h × 95 USD = 2,280 USD
model card         = 2 × 6 h × 95 USD = 1,140 USD
total              = 20,420 USD
```

The remediation is 7.9 percent of the annual governance cost. It is affordable, and it is the
minimum needed to pass the audit. The deeper issue is why the gaps appeared: the process was
defined but not enforced for tier 2 and tier 3.

### 6.5 Diagnostic one: the inventory is incomplete

The team reconciles the inventory against the container registry and the EHR integration list. It
finds 14 models that are running but not in the inventory: 9 developed by departments, 3 vendor
models embedded in a purchased application, and 2 prototypes that were never retired.

```text
inventory_before = 38
shadow_models    = 14
inventory_after  = 52
growth           = 14 / 38 = 36.8 percent
```

The tier mix after reconciliation is 8 tier 1, 14 tier 2, 21 tier 3, and 9 tier 4. The annual
governance cost rises to:

```text
8 × 18,000 + 14 × 9,000 + 21 × 3,000 + 9 × 1,000
= 144,000 + 126,000 + 63,000 + 9,000
= 342,000 USD
```

The increase is 84,000 USD, or 32.6 percent. The budget had been built on an inventory that omitted
a third of the models.

### 6.6 Diagnostic two: the change log has gaps

The team audits 23 model changes from the last quarter. Nine have no change record, and four of
those are threshold changes that altered the operating point.

```text
unrecorded_rate = 9 / 23 = 39.1 percent
material_unrecorded = 4
```

The four threshold changes are material under any reasonable rule. If they had been caught, each
would have required a clinical review and a recalibration check, at about 3,800 USD each:

```text
cost_if_caught = 4 × 3,800 = 15,200 USD
```

The organization avoided 15,200 USD of review by not recording the changes. It also created an
unaudited change to a clinical model, which is a larger risk than the cost avoided.

The first diagnostic is the pipeline's promotion gate. The gate allowed a threshold change without
a change record because the threshold lives in a configuration file that is not versioned with the
model. The second is the approval matrix, which did not classify threshold changes as material.

### 6.7 Vary one assumption: the tier thresholds

Suppose the organization lowers the tier 1 threshold so that four tier 2 models are reclassified
as tier 1. The annual cost change is:

```text
4 × (18,000 − 9,000) = 36,000 USD
```

The review burden rises from 72 to 120 reviews per year, and the independent validation requirement
applies to four more models. The cost is real, and so is the benefit: the four models are the ones
with the highest clinical exposure in tier 2.

The tier threshold is a risk-appetite decision, not a technical one. The example shows that it has
a direct budget consequence, which is why it should be made by the accountable executive, not by
the governance team alone.

### 6.8 What the example establishes

The inventory was the anchor. Once it was reconciled, the cost, the tier mix, and the audit sample
all changed. The change log revealed that the materiality rule was not enforced at the pipeline.

The audit findings were concentrated in the tiers where the process was least enforced, not in the
tiers with the most risk. That is a common pattern: governance effort follows attention, and
attention follows the highest tier. The middle tiers are where undocumented risk accumulates.

The example also shows that governance cost is dominated by tier 1. A budget that is spread evenly
across models will underfund the models that matter most and overfund the ones that do not.

### 6.9 The remediated state

After four weeks, the team reports the following.

| Metric | Before | After | Target |
|---|---:|---:|---:|
| Models in inventory | 38 | 52 | Complete |
| Models with a named owner | 33 (87 percent) | 52 (100 percent) | 100 percent |
| Validation current within 12 months | 34 (89 percent) | 50 (96 percent) | 95 percent |
| Rollback tested within 12 months | 35 (92 percent) | 47 (90 percent) | 90 percent |
| Changes with a record | 14 of 23 (61 percent) | 19 of 21 (90 percent) | 95 percent |

The rollback-test percentage fell because the 14 new models had never been tested. The target for
the next quarter is 95 percent, with the 5 remaining models scheduled.

The change-record rate improved but did not reach the target. The pipeline gate is the next
investment: a threshold change should fail the gate if there is no linked change record. That is a
technical control, and it is more reliable than a reminder.

## 7. Code, query, or calculation

The script implements the risk tiering rule, the governance cost model, and the coverage metrics
from Section 6. The tier is the maximum across four dimensions, which is the conservative choice:
a model is high tier if any dimension is high.

```python
"""Risk tiering, inventory coverage, and governance cost. Standard library only."""

LEVELS = {"low": 1, "medium": 2, "high": 3}


def tier(intended_use, population, autonomy, irreversibility):
    """Tier is the highest level across the four dimensions."""
    return max(
        LEVELS[intended_use],
        LEVELS[population],
        LEVELS[autonomy],
        LEVELS[irreversibility],
    )


def governance_cost(counts, unit_costs):
    return {t: counts[t] * unit_costs[t] for t in counts}


def coverage(owned, total):
    return owned / total


def main() -> None:
    cases = [
        ("sepsis autonomous", "high", "high", "high", "high"),
        ("scheduling advisory", "low", "low", "low", "low"),
        ("readmission human review", "medium", "medium", "medium", "medium"),
        ("imaging triage vulnerable group", "medium", "high", "medium", "medium"),
    ]
    for name, i, p, a, r in cases:
        print(f"{name:36s} tier={tier(i, p, a, r)}")

    counts = {1: 6, 2: 11, 3: 15, 4: 6}
    unit = {1: 18_000, 2: 9_000, 3: 3_000, 4: 1_000}
    costs = governance_cost(counts, unit)
    total = sum(costs.values())
    print(f"total_models          {sum(counts.values())}")
    print(f"annual_cost           {total} USD")
    print(f"cost_per_model        {total / sum(counts.values()):.0f} USD")

    counts_after = {1: 8, 2: 14, 3: 21, 4: 9}
    total_after = sum(governance_cost(counts_after, unit).values())
    print(f"annual_cost_after     {total_after} USD")
    print(f"increase_pct          {(total_after - total) / total * 100:.1f}")

    print(f"owner_coverage        {coverage(33, 38):.3f}")
    print(f"validation_coverage   {coverage(34, 38):.3f}")
    print(f"rollback_coverage     {coverage(35, 38):.3f}")

    allowed_late = 72 * 0.10
    burn = (11 / 72) / 0.10
    print(f"allowed_late          {allowed_late:.1f}")
    print(f"review_burn           {burn:.2f}")

    assert tier("high", "high", "high", "low") == 3
    assert tier("low", "low", "low", "low") == 1
    assert total == 258_000
    assert total_after == 342_000
    assert abs(burn - 1.53) < 0.01


if __name__ == "__main__":
    main()
```

Expected output, confirmed on Python 3.11:

```text
sepsis autonomous                    tier=3
scheduling advisory                  tier=1
readmission human review             tier=2
imaging triage vulnerable group      tier=3
total_models          38
annual_cost           258000 USD
cost_per_model        6789 USD
annual_cost_after     342000 USD
increase_pct          32.6
owner_coverage        0.868
validation_coverage   0.895
rollback_coverage     0.921
allowed_late          7.2
review_burn           1.53
```

The tier outputs show the conservative rule. The readmission model is medium on every dimension and
lands at tier 2. The imaging triage model is medium on intended use and autonomy but high on
population because it affects a vulnerable group, so it lands at tier 3. That is the intended
behaviour: a model that is advisory but inequitable is not low risk.

The cost outputs show that the average is not a useful budget unit. A tier 1 model costs eighteen
times a tier 4 model, so a per-model budget underfunds the models that matter.

The coverage metrics are the audit's summary. Owner coverage of 0.868 means five models have no
named owner, which is the finding that drives the remediation.

The reference service's `model_card.md` is a tier 4 artifact: intended use, model description,
output, evaluation, limitations, monitoring, and change control. It has no owner field, no risk
tier, no validation date, and no retirement plan. Those are the fields a governance inventory would
add.

## 8. Data, standards, and regulatory context

Governance is where the quality system and the model lifecycle meet. The instruments below define
the records, the reviews, and the authorities.

### 8.1 The instruments

| Instrument | Status | What it requires | What it does not prescribe |
|---|---|---|---|
| FDA QMSR, 21 CFR Part 820, effective 2 February 2026 | Binding for device manufacturers | A quality management system incorporating ISO 13485:2016, plus supplemental record, complaint, and labeling requirements | A model inventory schema |
| ISO 13485:2016 | Binding when incorporated or invoked | Document control, records, management review, corrective action | AI-specific controls |
| IEC 62304:2006/Amd 1:2015 | Binding when invoked | Software lifecycle, configuration management, problem resolution | Governance roles |
| ISO 14971:2019 | Binding when invoked | Risk management across the lifecycle | Risk tier definitions |
| FDA PCCP guidance, 2024, reissued 2025 | Final guidance | Pre-authorized modification protocol and impact assessment | Internal approval matrix |
| FDA GMLP principles, 2021 | Non-binding | Multidisciplinary expertise, representative data, monitoring | Governance structure |
| FDA AI-enabled device guidance, draft 2025 | Draft | Lifecycle management and postmarket monitoring recommendations | Inventory or tiering |
| ISO/IEC 42001:2023 | Voluntary, certifiable | AI management system, roles, risk and impact assessment, internal audit | Device-specific records |
| NIST AI RMF 1.0, 2023 | Voluntary | Govern, Map, Measure, Manage | A certification |
| EU AI Act, Regulation 2024/1689 | Binding in the EU, phased | Risk classification, quality management, post-market monitoring | US device records |
| HIPAA Security Rule | Binding for covered entities | Access control, audit controls, documentation retention | Model governance |
| 21 CFR Part 11 | Binding where predicate rules apply | Electronic records and signatures, audit trails | Approval workflows |

### 8.2 What the QMSR changes

The Quality Management System Regulation replaces the older quality system regulation and
incorporates ISO 13485:2016 by reference, with supplemental requirements. It became effective on
2 February 2026. For a model team, the practical consequences are:

- Management review and internal audit records are part of the inspectable quality system.
- Record control applies to model validation reports, change records, and monitoring evidence.
- Complaint handling must include model-related complaints, even when the complaint is about an
  alert rather than a device failure.
- Supplier controls apply to vendor models and to cloud providers that process device data.

A governance system that exists outside the quality system will not satisfy an inspection, even if
it is well designed.

### 8.3 What ISO/IEC 42001 adds

ISO/IEC 42001 specifies an AI management system with context, leadership, planning, support,
operation, performance evaluation, and improvement. It adds an AI system impact assessment that is
not present in ISO 27001, and it requires documented roles and authorities.

The standard is certifiable and voluntary. Its value for a device manufacturer is that it provides
a management-system structure for AI that complements the device quality system. Its risk is that
certification can be mistaken for evidence that a specific model is safe.

### 8.4 What the EU AI Act adds

The EU AI Act establishes risk categories and obligations for providers and deployers. Medical
device AI can be high-risk under the product-safety route, with obligations for risk management,
data governance, technical documentation, logging, transparency, human oversight, and post-market
monitoring.

The timetable has been amended since the regulation was adopted, and the current dates should be
verified with the official sources. The governance implication is that a model sold in the EU may
need a technical file and a quality management system that satisfy both the MDR and the AI Act.

### 8.5 Model risk management from banking

The Federal Reserve and the Office of the Comptroller of the Currency issued SR 11-7, guidance on
model risk management, in 2011. It describes model development, implementation, use, validation,
and governance, including a model inventory, a validation function independent of the model owner,
and board-level oversight.

The guidance is binding for the banks it supervises, not for health systems. It is widely used as
a reference because it is specific and practical. The inference for clinical AI is that its
structure, especially the independent validation and the inventory, transfers well; its
quantitative thresholds do not.

### 8.6 What is convention

- A four-tier risk scheme. The number of tiers is a design choice.
- A model card format. The content matters, not the template.
- A monthly or quarterly review cadence. The cadence should follow risk and label latency.
- A governance committee. A committee can approve, but it cannot own a model.
- An annual audit. The frequency should follow the risk and the audit findings.

> **Note:** Certification to ISO/IEC 42001 or ISO 13485 is a system-level attestation. It does not
> certify a specific model, and it does not replace validation, monitoring, or change control for
> that model.

## 9. Industry practice and case evidence

### 9.1 Model risk management in banking

SR 11-7 is the most detailed public model risk management guidance. It requires a model inventory,
a validation function independent of the model developers, ongoing monitoring, and reporting to
senior management and the board. It defines model risk as the potential for adverse consequences
from decisions based on incorrect or misused model outputs.

The documented guidance is from banking supervision. The inference for clinical AI is that the
structure is directly applicable: inventory, independent validation, monitoring, and reporting.
The difference is the consequence of error, which in clinical settings is measured in patients, not
dollars.

### 9.2 Model cards

Mitchell and colleagues proposed model cards in 2019 as short documents reporting intended use,
evaluation conditions, and performance across demographic and phenotypic groups. The proposal
includes intersectional analysis, such as age by race or sex by skin type.

The documented contribution is a reporting format. The inference for governance is that a model
card is a transparency artifact, not a control. It becomes a control only when it is required,
reviewed, and linked to the inventory and the change record.

### 9.3 The Epic Sepsis Model and vendor governance

The Epic Sepsis Model was embedded in the EHR and widely deployed. The external validation by Wong
and colleagues found lower discrimination and a high alert burden at Michigan Medicine. The
deploying organization had relied on the vendor's validation rather than conducting its own.

The documented facts are the validation results. The inference for governance is that accepting a
vendor's validation without local evidence is a governance decision, and it should be recorded as
one. The contract and the review process should require local validation for high-tier vendor
models.

### 9.4 The FDA AI-enabled device list

The FDA maintains a list of AI-enabled medical devices that have been authorized. The list is
public and searchable, and it shows the growth in authorizations over time.

The documented fact is the existence of the list and the authorizations. The inference for
governance is that a manufacturer's inventory should reconcile against the FDA list for devices it
has authorized, and a health system's inventory should reconcile against the vendor's authorization
status. A model that is marketed as AI-enabled but has no authorization may be a general wellness
product or an unregulated tool; the distinction matters.

#A reconciliation also protects the manufacturer: an authorized device that is not in the
inventory may be updated without change control, and an unauthorized product marketed as AI may
create regulatory exposure that the quality system never sees.

This is the same reconciliation logic as the internal inventory, applied to the regulator's
public record rather than the organization's own systems.

### 9.5 What the evidence does not show

There is little public evidence on the effectiveness of specific governance structures, the
frequency of audit findings by control type, or the relationship between governance maturity and
patient outcomes. The case literature describes failures; it does not provide base rates.

A team should track its own inventory coverage, review currency, change-record rate, and audit
findings, and should use those numbers to justify the governance investment. The numbers are more
persuasive to a budget owner than a framework citation.

### 9.6 The base-rate problem

Public governance case studies describe failures and near-failures. They do not report how many
models were governed without incident, so they cannot estimate the effect of a control. A team
considering an expensive control has no benchmark for its expected benefit.

The practical response is to measure leading indicators: inventory coverage, review currency,
change-record rate, and audit findings. These are proxies for control effectiveness, and they are
available locally. They are not proof of safety, but they are better than a framework citation
alone.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Incomplete inventory | Models appear in billing but not in the register | No reconciliation with deployment systems | Shadow models with no owner | Reconcile quarterly against registry and procurement |
| Stale owner | The named owner has left | Ownership assigned to a person, not a role | No one responds to a review or incident | Role-based ownership and annual attestation |
| Tier too low | A clinical model is classified as operational | Tiering by technology or team | Controls are weaker than the risk | Tier by intended use and consequence |
| Approval by the author | The same name appears as author and approver | No segregation of duties | Independent review is nominal | Name a backup approver and audit approvals |
| Vendor model unvalidated | The only evidence is the vendor's report | Procurement accepted the vendor claim | Local performance unknown | Require local validation for high tiers |
| Change log gaps | Threshold changes appear in production without records | Config not versioned with the model | Unaudited changes to a clinical model | Gate promotion on a linked change record |
| Stale validation | The validation date is older than the review cycle | No trigger for revalidation | The model may no longer be fit | Validation expiry and review calendar |
| Missing model card | The model has no intended-use statement | Documentation not required at build | Users do not know the limits | Require a card before deployment |
| No retirement | Disabled models remain in the inventory | No decommissioning process | Zombie models with access | Retirement workflow and access removal |
| Records inaccessible | The auditor cannot find the report | Records in personal folders | Audit finding | Central repository with index |
| Policy without enforcement | The policy exists but is not followed | No gate, no audit | The policy is decorative | Enforce at the pipeline and audit compliance |
| Governance only for devices | Non-device clinical models are excluded | Scope defined by regulatory status | The riskiest models may be ungoverned | Scope by decision, not by device status |
| Repeat findings | The same audit finding appears twice | Corrective action not verified | The control is not working | CAPA with verification and closure |
| Vendor update unnoticed | Behaviour changes after a vendor release | No notification clause or monitoring | An uncontrolled change | Contract clause and release monitoring |

Red flags: an inventory that has not changed in a year; an owner who cannot describe the model's
monitoring; a tier assigned by the build team without review; a change record that lists only code
changes; a vendor contract with no audit rights.

## 11. Assumptions and limits

Each assumption is testable; the test column names the evidence.

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The inventory is complete | Every control depends on it | Shadow deployments | Reconcile against registry and billing |
| Owners have authority and time | Accountability requires capacity | Nominal owners | Review response times and decisions |
| The tier reflects consequence | Controls are proportional | Tiering by technology | Audit a sample against the criteria |
| Approvers are independent | Approval is a control | Author-approver overlap | Compare author and approver records |
| The evidence is accessible | Audit and review depend on it | Records in personal folders | Retrieve a sample within one day |
| The vendor reports changes | Third-party risk depends on it | Contract has no clause | Review the contract and release notes |
| The review cadence matches the risk | Reviews are timely | Calendar-driven cadence | Measure review currency |
| Records are retained | Audit covers a period | Storage policy shorter than the period | Retrieve a two-year-old record |
| The audit is independent | Findings must be credible | The auditor reports to the model team | Review the reporting line |

## 12. Exercises

### Exercise 12.1 — Build the inventory schema

For a health system with five model types (sepsis, readmission, scheduling, billing denial, imaging triage), write the inventory schema and one filled row for each type.

**Deliverable.** A field table and five example rows.

**Acceptance criteria.** Every required field from Section 5.16 is present; owners are roles; tiers are justified; status values are defined.

**Time.** 40 minutes.

### Exercise 12.2 — Assign tiers

Assign a tier to each: (a) an autonomous insulin dosing model; (b) a clinic scheduling optimizer; (c) a readmission score reviewed by a discharge planner; (d) a billing denial prediction used by finance; (e) an imaging triage model for a population with limited access.

**Deliverable.** A five-row table with the four dimensions and the tier.

**Acceptance criteria.** Tier is the maximum across dimensions; the vulnerable-population case is tier 3; the rationale is one sentence per row.

**Time.** 25 minutes.

### Exercise 12.3 — Design an approval matrix

Write an approval matrix for four tiers and four change types: new model, retrain within protocol, threshold change, and retirement.

**Deliverable.** A 4×4 matrix with approvers and evidence.

**Acceptance criteria.** No author approves their own change; every cell has an approver and an evidence item; the matrix is consistent with the tier controls.

**Time.** 35 minutes.

### Exercise 12.4 — Apply the materiality tree

Classify each change as material or operational: (a) upgrade a logging library; (b) change a threshold from 0.18 to 0.20; (c) retrain on six more months of data within the protocol; (d) add a new laboratory feature; (e) change a Kubernetes resource limit.

**Deliverable.** A five-row table with the path through the tree and the required review.

**Acceptance criteria.** Threshold and feature changes are material; the retrain is material but covered by the protocol; the infrastructure change is operational; the rationale is recorded.

**Time.** 30 minutes.

### Exercise 12.5 — Write a model card

Write a model card for the reference service in `labs/mlops/`, including intended use, population, data, performance, subgroup performance, limitations, monitoring, and change control.

**Deliverable.** A one-page model card.

**Acceptance criteria.** States that the model is not a clinical device; includes the evaluation limitations; includes a monitoring section and a change-control section; has a version and date.

**Time.** 40 minutes.

### Exercise 12.6 — Design vendor clauses

Write five contract clauses for a vendor-supplied clinical model covering notification, validation evidence, audit rights, incident reporting, and exit.

**Deliverable.** Five clauses with a one-sentence rationale each.

**Acceptance criteria.** Each clause is testable; notification includes a time window; audit rights include model changes; exit includes data and model portability.

**Time.** 35 minutes.

### Exercise 12.7 — Design an audit sample

A health system has 52 models: 8 tier 1, 14 tier 2, 21 tier 3, 9 tier 4. Design a stratified audit sample of about 20 models. State the strata, the sample per stratum, and the finding categories.

**Deliverable.** A sampling table and a finding taxonomy.

**Acceptance criteria.** All tier 1 models are included; the sample is justified; findings are categorized by control; the audit can be repeated.

**Time.** 35 minutes.

### Exercise 12.8 — Design a retirement

A tier 2 model is being replaced by a new version. Write the retirement plan, including the decision record, communication, data retention, access removal, and inventory update.

**Deliverable.** A one-page plan with a timeline.

**Acceptance criteria.** No coverage gap; users are notified; records are retained; access is removed; the inventory status is updated.

**Time.** 30 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

A filled row for the sepsis model: identifier `MDL-001`; name `Sepsis deterioration v2`; intended
use `identify inpatients at risk of deterioration within 12 hours`; owner role `Clinical AI lead`;
accountable executive `Chief Medical Information Officer`; tier 3; version `2.1.0`; pattern
`event-driven API`; sites `3 hospitals`; data sources `EHR vitals, laboratory, admissions`; validation
date `2026-08-14`; last review `2026-09-30`; status `active`; retirement date blank.

The scheduling model would be tier 1, with a shorter evidence set and an owner in operations rather
than clinical leadership.

### 13.2 Exercise 12.2

(a) Tier 3. High intended use, high autonomy, high irreversibility.
(b) Tier 1. Low on every dimension.
(c) Tier 2. Medium intended use and autonomy, human review.
(d) Tier 2. The clinical consequence is low, but the financial and equity consequences of a denial
prediction are medium.
(e) Tier 3. Medium intended use and autonomy, high population vulnerability.

### 13.3 Exercise 12.3

A strong matrix names a clinical reviewer and a quality reviewer for tier 1 and tier 2 new models,
the model owner and a peer for tier 3, and the owner for tier 4. Threshold changes at tier 1 and 2
require clinical review; retrains within a protocol require quality review. Retirement requires the
clinical owner and, for tier 1, regulatory affairs.

### 13.4 Exercise 12.4

(a) Operational. No output change.
(b) Material. The operating point changes.
(c) Material but protocol-covered, if the protocol describes the data and the evaluation.
(d) Material and likely outside the protocol. New input space.
(e) Operational. No output change, but it needs a capacity check.

### 13.5 Exercise 12.5

A strong card states the intended use as a teaching demonstration, not a clinical device. It reports
the synthetic coefficients, the absence of clinical validation, and the lack of subgroup analysis.
It includes a monitoring section (request rate, error rate, latency, prediction volume, input ranges,
model version) and a change-control section (revalidation, new version, rollback path). It has a
version and a date.

### 13.6 Exercise 12.6

Five clauses: notification of any model change within 30 days before release; provision of
validation evidence on request; audit rights including access to model documentation and change
logs; incident notification within 24 hours; exit with data and model portability. Each clause
should state the remedy for non-compliance.

### 13.7 Exercise 12.7

A stratified sample of 20: all 8 tier 1, 6 of 14 tier 2, 4 of 21 tier 3, 2 of 9 tier 4. The
finding taxonomy: ownership, validation currency, evidence completeness, change records, monitoring
records, rollback test, vendor documentation, retirement status. The audit is repeatable because
the strata and the categories are fixed.

### 13.8 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Completeness | 25% | Missing fields | Required fields present | Reconciled with another source |
| Proportionality | 25% | One process for all | Tiered controls | Tier rationale recorded |
| Independence | 20% | Author approves | Named reviewer | Backup and audit of approvals |
| Evidence | 20% | A folder of links | Indexed repository | Retrievable within a day |
| Lifecycle | 10% | No retirement | Retirement step | Tested decommissioning |

The self-check answers for the design exercises are rubrics, not model answers, because the
right structure depends on the organization. The numeric answers in 13.2 and 13.4 are not.

## 14. Interview questions

**1. What is model governance?**
Strong answers describe decisions, authorities, and records across the lifecycle. Weak answers
describe a committee or a policy.

**2. What belongs in a model inventory?**
Strong answers name owner, tier, version, intended use, validation date, and status. Weak answers
say a list of models.

**3. How do you assign a risk tier?**
Strong answers use intended use, population, autonomy, and reversibility, and take the maximum.
Weak answers use accuracy or model type.

**4. Who should approve a model change?**
Strong answers match authority to tier and require independence from the author. Weak answers say
the manager.

**5. What is a material change?**
Strong answers define it by effect on intended use, output, or risk, and give examples. Weak answers
say any code change.

**6. How do you govern a vendor model?**
Strong answers require notification, validation evidence, audit rights, and exit provisions. Weak
answers rely on the vendor's reputation.

**7. How do you keep the inventory current?**
Strong answers reconcile against deployment, procurement, and billing systems. Weak answers rely on
annual updates.

**8. What is the role of a model card?**
Strong answers say it is a transparency artifact that becomes a control when required and reviewed.
Weak answers say it is documentation.

**9. How do you audit governance?**
Strong answers use a stratified sample, fixed finding categories, and coverage metrics. Weak answers
review documents.

**10. When should a model be retired?**
Strong answers cite no owner, no use, degraded performance, or replacement, with a plan for
communication and access removal. Weak answers say when it is old.

## 15. Key terms

- **Accountable executive** — the leader who accepts residual risk.
- **Attestation** — a formal statement that a control is in place.
- **Audit trail** — a record of who did what and when.
- **CAPA** — corrective and preventive action.
- **Change control** — the process for approving and recording changes.
- **Decommissioning** — planned removal of a model from service.
- **Design history file** — the record of design and development for a device.
- **Evidence pack** — the documents supporting a decision.
- **Governance** — the decisions, authorities, and records that make a model accountable.
- **Independent validation** — validation by someone who did not build the model.
- **Material change** — a change that alters intended use, performance, or risk.
- **Model card** — a structured summary of intended use, data, performance, and limits.
- **Model inventory** — a register of models with owner, tier, version, and status.
- **Model owner** — the role accountable for the model's lifecycle.
- **Model risk management** — the discipline of managing model risk, formalized in banking.
- **PCCP** — a pre-authorized description of permitted modifications.
- **Post-market surveillance** — systematic monitoring after release.
- **QMS** — quality management system.
- **QMSR** — FDA Quality Management System Regulation, effective 2026.
- **RACI** — responsible, accountable, consulted, informed.
- **Risk tier** — a classification that determines the controls.
- **Segregation of duties** — separating author, approver, and auditor.
- **Shadow model** — a model deployed outside the governance process.
- **Technical file** — the evidence package for a device in some jurisdictions.
- **Third-party risk** — risk from vendors, open-source, and partners.
- **Traceability** — the ability to link a requirement to a test and a record.
- **Transparency** — clear information about a model's use and limits.
- **Zombie model** — a model still running with no owner or use.

## 16. Sources

All URLs accessed 26 October 2026.

- US Food and Drug Administration. *Quality Management System Regulation (QMSR)*. https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
- Federal Register. *Medical Devices; Quality System Regulation Amendments*, 89 FR 7496, 2 February 2024. https://www.federalregister.gov/documents/2024/02/02/2024-01709/medical-devices-quality-system-regulation-amendments
- 21 CFR Part 820, Quality Management System Regulation. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820
- ISO 13485:2016. *Medical devices — Quality management systems*. https://www.iso.org/standard/59752.html
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- NIST. *AI Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, 2023. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- US Food and Drug Administration. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, 2024, reissued 2025. https://www.fda.gov/media/166704/download
- US Food and Drug Administration. *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations*, draft guidance, January 2025. https://www.fda.gov/media/184856/download
- Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency. *SR 11-7: Guidance on Model Risk Management*, 2011. https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- Mitchell M, Wu S, Zaldivar A, et al. Model cards for model reporting. *Proceedings of FAT\* 2019*, 220-229. https://doi.org/10.1145/3287560.3287596
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*. 2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- US Food and Drug Administration. *Artificial Intelligence and Machine Learning (AI/ML)-Enabled Medical Devices*. https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices
- Council of the European Union. *AI Act timeline*. https://www.consilium.europa.eu/en/policies/artificial-intelligence-act/timeline-artificial-intelligence/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`08_rollout-and-rollback.md`](08_rollout-and-rollback.md)
- Next topic: [`10_cost-and-capacity.md`](10_cost-and-capacity.md)
- Related topic: [`03_packaging-and-reproducibility.md`](03_packaging-and-reproducibility.md)
- Related topic: [`07_incident-response.md`](07_incident-response.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Model card: [`../../labs/mlops/model_card.md`](../../labs/mlops/model_card.md)
- Runbook: [`../../labs/mlops/RUNBOOK.md`](../../labs/mlops/RUNBOOK.md)
