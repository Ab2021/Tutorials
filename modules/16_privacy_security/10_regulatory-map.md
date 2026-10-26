# Regulatory Map

> **Module 16 - Privacy, Security, and Ethics**
> Topic 10 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 07
> Version 2026.3

## 1. Why this topic exists

A biomedical data product rarely sits under one law. A symptom-triage app sold to employers may be
a consumer health product under the Federal Trade Commission's rules, a business associate under
HIPAA when it integrates with a health plan, a medical device under the Medical Device Regulation
in the EU, a high-risk AI system under the AI Act, and a covered business under several state
privacy statutes. Each regime has its own definitions, deadlines, and enforcement body, and none of
them organises itself around the product's architecture.

The result is predictable. A team reads one law, builds to it, and discovers at procurement or at
launch that another law imposes a different obligation: a 72-hour breach notice, a consent
standard, a conformity assessment, a data protection impact assessment, or a prohibition on sharing
health data for advertising. The cost is a delayed launch, a redesign, or an enforcement action.

Three failure patterns recur.

**Single-regime thinking.** The team treats HIPAA as the definition of health privacy. It applies
to covered entities and business associates, and it leaves consumer apps, employers, and most
wellness products outside its scope. A product that is not a covered entity can still be subject to
the FTC Act, the FTC Health Breach Notification Rule, state consumer health privacy laws, and, in
the EU, the GDPR. The absence of HIPAA is not the absence of regulation.

**Entity misclassification.** A hospital's research spin-out is treated as part of the hospital, or
a vendor is treated as a business associate when it is acting as a controller, or a wellness app is
treated as a medical device. The classification determines the obligations, and it is a legal
question about facts, not a product-name question. Misclassification is the most expensive kind of
regulatory error because it changes the entire obligation set.

**The static map.** The regulatory map is built once, at launch, and never updated. The AI Act
applies in stages through 2028, the European Health Data Space applies from 2029, state legislatures
add new health privacy laws each year, and enforcement priorities shift. A map with no review date
is a map of a jurisdiction that no longer exists.

This topic teaches the regulatory map as an applicability analysis: identify the entity and its
role, classify the data and the activity, list the regimes that apply, resolve conflicts, assess
readiness, and sequence the work. It is the capstone of the module because it connects the privacy,
security, fairness, ethics, and governance topics to the legal obligations that make them
mandatory or voluntary.

The distinction that runs through the topic is between a binding requirement, a conditional
requirement (which depends on facts such as role, data type, or risk classification), and a
voluntary framework. Most regulatory errors come from treating a conditional requirement as
inapplicable without checking the condition, or from treating a voluntary framework as a legal
safe harbour.

The map is also a communication tool for the product team. A designer who knows that Washington
requires separate consent for consumer health data, and that the EU requires a DPIA, can build the
consent flow and the data map once rather than retrofit them after a legal review. The earlier the
map is available, the cheaper the compliance.

The map is also a due-diligence artefact. An acquirer will ask for the obligation set, the evidence,
and the open gaps; a company that can produce them in a data room is worth more than one that
cannot, because the buyer's counsel will otherwise price the uncertainty into the deal.

**The map is a living document, and the review is the control.** A map reviewed on schedule, with a
change log and an owner, is a governance asset; a map produced for a launch and filed is a
snapshot that will be wrong within a year.

> **Note:** This topic is educational and is not legal advice. The applicability of a regime to a
> specific product requires counsel in each jurisdiction.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** an entity and its role under HIPAA, GDPR, the AI Act, and a state consumer health
   law, and **explain** how the classification changes the obligations.
2. **Build** an applicability matrix that lists regimes, obligations, status, and evidence.
3. **Compute** a weighted readiness coverage score and **identify** the regimes with the largest
   gaps.
4. **Resolve** conflicts between regimes on consent, breach timing, data subject rights, and
   cross-border transfer.
5. **Sequence** remediation by legal deadline, market-entry dependency, and risk.
6. **Explain** the staged application of the EU AI Act and the EHDS, and **identify** which
   obligations apply to a medical device in 2026, 2027, and 2028.
7. **Distinguish** a binding requirement, a conditional requirement, and a voluntary framework in a
   vendor or internal claim.
8. **Maintain** the map with a review cadence and a change log.

## 3. The core mental model

The regulatory map has four layers: the entity, the data, the activity, and the regime. Each layer
narrows the set of applicable obligations.

```text
  ENTITY            who are you?  provider | plan | clearinghouse | BA |
                    consumer app | device maker | researcher | employer
        |
  DATA              what data?    PHI | personal data | special category |
                    consumer health data | de-identified | anonymous
        |
  ACTIVITY          what do you do? treatment | payment | research |
                    marketing | model training | secondary use | transfer
        |
  REGIME            which law?    HIPAA | FTC Act/HBNR | GDPR | AI Act |
                    MDR | EHDS | state law | sector law
        |
        v
  OBLIGATION SET -> evidence -> readiness -> remediation
```

The layers interact. The same data can be PHI in one flow and consumer health data in another. The
same entity can be a covered entity for its clinical operations and a business associate for a
payer's analytics. The same activity can be treatment in one jurisdiction and research in another.
The map must be per flow, not per company.

A second model is the obligation taxonomy:

| Type | Example | Consequence of missing it |
|---|---|---|
| Prohibition | Marketing without authorisation; AI practice banned under Article 5 | Enforcement; stop order |
| Conditional requirement | DPIA for high-risk processing; conformity assessment for a medical device | Cannot launch lawfully |
| Documentation | Records of processing; risk analysis; technical file | Burden of proof on the entity |
| Notification | Breach notice; transparency notice | Penalty; loss of trust |
| Rights | Access, erasure, correction, human review | Complaint; enforcement |
| Governance | DPO; privacy officer; quality system | Accountability gap |

The third model is the conflict map. Regimes conflict on consent (HIPAA authorisation versus GDPR
consent), breach timing (60 days versus 72 hours), rights (HIPAA access versus GDPR erasure), and
transfer (HIPAA has no general transfer restriction; GDPR Chapter V does). The resolution is
usually to meet the stricter requirement in the design and document why, rather than to build two
systems.

```text
  HIPAA 60 days --------\
                         >-- design to the shorter/stricter rule
  GDPR 72 hours --------/

  HIPAA authorisation --\
                         >-- separate instruments, mapped to each legal test
  GDPR consent ---------/
```

The final model is the readiness score. It is not a legal conclusion; it is a management tool that
shows where the evidence is missing. A score without a list of specific gaps is a number without a
decision.

The regulatory map is the last topic because it is the one that turns the rest of the module into
obligations. De-identification, access control, fairness, ethics, and governance are good practice
in some settings and legally required in others; the map tells you which. A team that has the
artefacts but not the map does not know which ones a regulator will ask for.

The four layers also interact across time. A product that is a wellness app today can become a
medical device when a feature changes; a company that is a processor for one customer can be a
controller for its own analytics. The map is therefore a living document with a version, an owner,
and a review date, not a one-time deliverable.

The map should be written for the reader who has to act on it: the product manager who must change
the consent flow, the engineer who must add a retention control, the board member who must approve
the budget. A map that is legally precise and operationally unusable will not change any of those
decisions.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Regime | A body of law with its own definitions and enforcement. | A single statute. | Legal system. |
| Applicability | Whether a regime applies to an entity, data, and activity. | Assumed from the product name. | Legal analysis. |
| Covered entity | A health plan, clearinghouse, or provider transmitting health information electronically for a covered transaction. | Any health company. | HIPAA role. |
| Business associate | An entity handling PHI on behalf of a covered entity. | A controller. | HIPAA role. |
| Controller | The party determining the purposes and means of processing. | A processor. | GDPR role. |
| Processor | The party processing on behalf of a controller. | A controller. | GDPR role. |
| Joint controller | Two or more controllers determining purposes jointly. | A processor. | GDPR role. |
| PHI | Protected health information under HIPAA. | Any health data. | US legal category. |
| Consumer health data | Health data collected by a consumer product, defined by state law. | PHI. | State legal category. |
| Special category data | GDPR Article 9 data, including health and genetic data. | All personal data. | EU legal category. |
| Personal data | Information relating to an identified or identifiable person. | PHI. | EU legal category. |
| Medical device | A product intended for a medical purpose under MDR or FDA rules. | Any clinical software. | Regulatory classification. |
| High-risk AI | An AI system meeting the AI Act criteria in Article 6 and the annexes. | Any medical AI. | EU classification. |
| Conformity assessment | The process of demonstrating compliance with a product regime. | A self-declaration. | Procedure. |
| CE marking | The EU mark indicating conformity with applicable legislation. | A quality label. | Mark. |
| Predetermined change control plan | FDA mechanism for pre-authorised model updates. | A free update. | Regulatory pathway. |
| DPIA | Data protection impact assessment. | A privacy policy. | Process. |
| Records of processing | GDPR Article 30 documentation. | A data map. | Document. |
| Breach notification | Legal duty to notify a breach. | A press release. | Obligation. |
| Data subject rights | Access, rectification, erasure, restriction, portability, objection. | A customer service request. | Rights. |
| Transfer mechanism | The legal basis for a cross-border transfer under GDPR Chapter V. | Encryption. | Legal instrument. |
| Adequacy decision | An EU decision that a country provides adequate protection. | A contract. | Legal instrument. |
| Standard contractual clauses | Model clauses for transfers. | A DPA. | Contract. |
| Sectoral law | Regulation specific to a sector, such as health or devices. | A general law. | Legal category. |
| Preemption | The effect of federal law displacing state law. | Automatic. | Legal doctrine. |
| Enforcement discretion | A regulator's choice of what to pursue. | A safe harbour. | Practice. |
| Guidance | A regulator's interpretation, not binding law. | Regulation. | Document. |
| Safe harbour | A legal provision that removes liability if conditions are met. | A best practice. | Legal concept. |
| Voluntary framework | A standard or framework adopted by choice. | A legal requirement. | Framework. |
| Readiness score | A weighted measure of evidence coverage. | A compliance certificate. | Management metric. |
| Regulatory change log | The record of changes and their impact. | A newsletter. | Document. |
| Lead authority | The supervisory authority coordinating a cross-border case. | The first to complain. | GDPR concept. |

## 5. The mechanics, step by step

The procedure below builds a regulatory map for a digital health product or programme. It is
written for a legal, privacy, or regulatory lead working with a product team, and it assumes that
the privacy, security, fairness, ethics, and governance artefacts from the earlier topics exist.
Each step names its input, output, and rationale.

### Step 1. Identify the entity and its role, per data flow

**Input:** Corporate structure, contracts, product architecture, data flows.
**Output:** A role statement per flow: covered entity, business associate, controller, processor,
joint controller, consumer product, device manufacturer, or researcher.
**Rationale:** Role determines the obligation set. The same company can be a covered entity for its
clinics, a business associate for a payer's analytics, and a controller for its consumer app. A role
statement per flow is the only accurate representation.

| Entity situation | HIPAA role | GDPR role | Other regimes | Key consequence |
|---|---|---|---|---|
| Hospital providing care | Covered entity | Controller | State provider laws; MDR if it makes devices | Full HIPAA and GDPR obligations |
| Analytics vendor serving a hospital | Business associate | Processor | State breach laws | BAA; controller instructions; breach notice to the covered entity |
| Consumer symptom app | Not a covered entity | Controller | FTC Act; FTC HBNR; state consumer health laws | No HIPAA, but FTC and state obligations |
| Employer wellness programme | Not a covered entity (unless a health plan) | Controller | ADA; state laws; ERISA if a plan | Consent and discrimination risk |
| Research spin-out using hospital data | May be a covered entity or BA | Controller or joint controller | Common Rule; GDPR research provisions | Data use agreement; ethics review |
| Device maker | Not a covered entity | Controller for patient data | MDR; FDA; AI Act | Conformity assessment; post-market surveillance |

Branch point: if a flow has two possible roles, document both and the facts that distinguish them.
A vendor that determines the purpose of processing is not a processor, whatever the contract says.

### Step 2. Classify the data

**Input:** Data inventory, field-level classification.
**Output:** A data class per field: PHI, personal data, special category, consumer health data,
de-identified, anonymous, or non-personal.
**Rationale:** The data class determines which regimes attach. A field can be PHI in one flow and
consumer health data in another; a de-identified dataset can be anonymous under one regime and
personal data under another. The classification should reference the determinations from Topic 02,
not a label in a catalogue.

The classification should also record the identifiers, the quasi-identifiers, and the sensitive
attributes, because many regimes attach specific obligations to those categories (for example,
genetic data, mental health, substance use, reproductive health).

### Step 3. Classify the activity

**Input:** Purpose statement, workflow, model lifecycle.
**Output:** An activity label per flow: treatment, payment, operations, research, marketing, model
training, secondary use, cross-border transfer.
**Rationale:** Activity determines the legal basis and the exceptions. Treatment has a different
basis from marketing, research has different rules from operations, and model training may be a
secondary use that requires its own analysis. A product that describes itself as "analytics" has
not classified the activity.

### Step 4. List the regimes and their obligations

**Input:** Entity, data, activity, jurisdictions.
**Output:** A regime list with the specific obligations that apply.
**Rationale:** The list should be specific, not a list of statute names. "GDPR" is not an
obligation; "Article 35 DPIA for high-risk processing" is. The obligations should be grouped by
type (prohibition, conditional requirement, documentation, notification, rights, governance) so
that the team can see the shape of the burden.

### Step 5. Check the conditions and thresholds

**Input:** Regime list, product facts.
**Output:** A condition determination per obligation: applies, does not apply, or undetermined.
**Rationale:** Many obligations are conditional. The AI Act applies to high-risk systems, which
depends on the intended purpose and the annex; the MDR applies to products with a medical purpose;
the GDPR DPIA applies to high-risk processing; state consumer health laws have thresholds and
exemptions. The condition should be recorded with the facts that support it, because a regulator
will ask.

Branch point: if a condition is undetermined, treat the obligation as applying for planning
purposes and resolve the condition before launch. An undetermined condition is not an exemption.

### Step 6. Resolve conflicts

**Input:** Obligations across regimes.
**Output:** A conflict resolution table with the chosen design and the rationale.
**Rationale:** Conflicts are common and usually resolvable by meeting the stricter rule. The table
should name the conflict, the regimes, the options, and the decision.

| Conflict | Regimes | Resolution | Cost |
|---|---|---|---|
| Consent standard | HIPAA authorisation; GDPR consent | Separate instruments; GDPR consent for EU, authorisation for US | Two forms; two records |
| Breach timing | HIPAA 60 days; GDPR 72 hours | Design to 72 hours; one incident process | Faster escalation; more initial notices |
| Access rights | HIPAA access; GDPR access and portability | Provide access in both; add portability for EU | Portal and export work |
| Erasure | HIPAA retention; GDPR erasure | Retain where law requires; erase elsewhere; document | Data map and deletion tooling |
| Transfer | HIPAA no general restriction; GDPR Chapter V | Transfer assessment and SCCs for EU data | Legal and contractual work |
| Marketing | HIPAA authorisation; GDPR consent; FTC Act | Explicit opt-in; no sale without consent | Lower conversion |

The resolution should be recorded in the design, not only in a legal memo. A conflict resolved on
paper and not in the product is still a conflict.

### Step 7. Assess evidence and readiness

**Input:** Obligation list, evidence register.
**Output:** A readiness table with status per obligation and a weighted coverage score.
**Rationale:** Readiness is about evidence, not intention. Each obligation should have a status
(met, partial, missing), an evidence item, an owner, and a date. The weighted score is a management
tool: it shows where the largest gaps are relative to the weight of the obligation.

The weights should reflect legal risk and market-entry dependency, not only effort. A missing
conformity assessment that blocks market entry is weighted higher than a documentation gap that can
be closed before an audit.

### Step 8. Sequence remediation

**Input:** Readiness table, deadlines, dependencies.
**Output:** A sequenced plan with dates, owners, and dependencies.
**Rationale:** Not all gaps can be closed at once. The sequence should put legal deadlines first
(for example, an AI Act obligation that applies from a fixed date), then market-entry dependencies
(a CE mark, a BAA), then risk-reduction items, then documentation. The plan should show the
critical path to launch.

### Step 9. Build the evidence register and the map

**Input:** All outputs above.
**Output:** A regulatory map document with the entity and role, the data and activity
classifications, the obligation matrix, the conflict resolutions, the readiness score, and the
evidence register.
**Rationale:** The map is the artefact that a board, an auditor, or a acquirer will ask for. It
should be versioned and owned, and it should be readable by a non-lawyer. The evidence register is
what turns the map from a list of laws into a management system.

### Step 10. Monitor and review

**Input:** Regulatory change sources, enforcement actions, product changes.
**Output:** A change log and a review cadence.
**Rationale:** The map changes when the law changes, when the product changes, and when the
enforcement climate changes. The review should be triggered by a new jurisdiction, a new data type,
a new activity, a new entity structure, a material product change, and a relevant enforcement
action, as well as by the calendar.

### Role analysis in practice

Role analysis is fact-intensive, and the contract is not decisive. A processor that determines the
purposes of processing becomes a controller; a business associate that uses PHI for its own
purposes outside the contract may become a covered entity for that use. The analysis should record
the facts: who decides why the data are processed, who decides how, who benefits, who sets
retention, and who can disclose to third parties.

A useful test is to ask three questions for each flow. Who would a patient call to exercise a
right? Who decides whether to honour a deletion request? Who bears the consequence of a breach?
The answers usually identify the controller and the boundaries between roles better than the
contract labels.

### Conditional obligations and thresholds

Many obligations apply only if a condition is met, and the condition is often the whole analysis.

| Regime | Condition | What to check | If undetermined |
|---|---|---|---|
| AI Act | High-risk under Article 6 and the annexes | Intended purpose; safety component; annex listing | Treat as high-risk for planning |
| MDR | Medical purpose | Claims, intended use, clinical benefit | Obtain a classification opinion |
| GDPR DPIA | High-risk processing | Systematic monitoring, special category data, scale, new technology | Complete a DPIA |
| GDPR transfer | Data leave the EEA | Destination, recipient, mechanism | Use SCCs and a transfer assessment |
| HIPAA | Covered entity or business associate | Role per flow; electronic transactions | Obtain a BAA or restructure the flow |
| FTC HBNR | Vendor of personal health records | Consumer health data; not HIPAA-covered | Assess; the rule reaches apps |
| State consumer health law | Doing business in the state; consumer health data | Thresholds, exemptions, geofencing | Comply for launch states |

The table's last column is the practical rule: an undetermined condition is treated as applying
for planning, because the cost of preparing is lower than the cost of a late discovery.

### The evidence register

The register is a table with one row per obligation: the obligation, the owner, the artefact, the
location, the date, the reviewer, and the review date. The artefact should be the thing an auditor
would ask for, not a link to a policy hub. For example, the DPIA is an artefact; the privacy
notice is an artefact; the BAA is an artefact; the technical file is an artefact.

The register should be reviewed before launch and at each change trigger. An obligation with no
artefact is a gap; an artefact with no owner is a liability; an artefact with no review date is a
snapshot that will age.

### Working with counsel

The map is built by the product and privacy team and reviewed by counsel. The division of labour
matters: the team supplies the facts (flows, data, contracts, claims), and counsel supplies the
legal characterisation. A map that counsel writes without the facts is generic; a map that the team
writes without counsel is a guess.

The engagement should be structured around the decisions, not around a request to "review
compliance". A one-page list of the five decisions, with the facts and the options, produces a
better answer than a 100-page pack. The legal advice should be recorded against the decision, so
that a successor can see why the design took the path it did.

### What a map is not

A map is not a legal opinion, a certification, or a guarantee. It does not resolve a contested
interpretation, and it does not protect against a regulator taking a different view. It is a
structured list of the obligations, the evidence, and the gaps, and its value is that it makes the
gaps visible before they become findings.

The map is also not a substitute for the privacy, security, fairness, ethics, and governance work in
the earlier topics. It tells you which of those artefacts are legally required; it does not produce
them.

### A worked obligation row

Take the GDPR DPIA. The row should say: obligation (Article 35 DPIA for high-risk processing);
condition (systematic monitoring, special category data at scale, or new technology); status
(draft); evidence (DPIA document, version, date, reviewer); owner (privacy lead); dependency
(processing description and data-flow map); review date (before launch and on material change). A
row with those fields can be managed; a row that says "GDPR - partial" cannot.

The same structure applies to a HIPAA risk analysis, an MDR technical file, an AI Act risk
management file, and a state privacy notice. The obligation differs; the management is the same.

### Sequencing by condition and deadline

The sequence should put fixed legal dates first, then market-entry dependencies, then risk
reduction, then documentation. Fixed dates include the AI Act milestones and the EHDS application
dates. Market-entry dependencies include a CE mark, a BAA, and a state registration. Risk reduction
includes encryption, access control, and monitoring. Documentation includes policies, records, and
notices.

The sequence should also account for dependencies between workstreams: the MDR technical file and
the AI Act risk management file share evidence, and building them together saves effort. A
sequence that treats them as separate programmes duplicates work and creates inconsistencies.

### When a regime is unclear

Some questions do not have a settled answer: whether a particular app is a medical device, whether
a model update is a new device, whether a dataset is anonymous, whether a transfer is necessary for
the performance of a contract. The map should record the question, the competing interpretations,
the facts that would resolve it, and the conservative position until it is resolved.

A conservative position is not a permanent answer. It should have a date, an owner, and a plan to
resolve the question with counsel or a regulator. An ambiguity that is never resolved becomes an
assumption that everyone forgets was an assumption.

### The map review

The review should be a standing agenda item for the privacy or regulatory committee, with a change
log and a version. A review that consists of re-reading the same document is not a review; it should
compare the map to the current product, the current contracts, and the current law, and record what
changed.

### A map for a model, not only a product

A model can have its own regulatory life. A model that is a medical device has an MDR pathway; a
model that is a component of a device shares the device's pathway; a model that is a
general-purpose model used in a high-risk system has AI Act obligations for the provider; a model
trained on secondary-use data may be subject to the EHDS permit terms. The map should include the
model as an entity where the regime treats it as one, and it should state who is the provider and
who is the deployer.

The provider/deployer distinction matters: under the AI Act, the provider bears the conformity
obligations, and a deployer that substantially modifies the system can become a provider. A
hospital that fine-tunes a vendor model may become a provider without realising it.

### The map's first page

The first page should state the product, the markets, the entity roles, the top three gaps, the
top three deadlines, and the decisions requested. A reader who stops after the first page should
know whether the launch is on track and what they are being asked to decide.

### The map and the budget

Every obligation has a cost: a DPIA takes weeks, a BAA takes legal time, a technical file takes
months, a conformity assessment takes a notified body. The map should be accompanied by an estimate,
because a board cannot decide between pathways without it. The estimate does not need to be precise;
it needs to be comparable across pathways.

### When the map is wrong

The map will be wrong somewhere. A regime may apply that nobody identified, a condition may resolve
differently, or an enforcement action may change the interpretation. The response should be a
correction process, not a defence of the document: log the error, assess the impact, update the map,
and tell the decision-maker. A map that cannot be corrected is a liability.

### Edge cases

**Small cells.** A regulatory filing or transparency report that includes small counts can identify
patients. The publication should aggregate and suppress, as in Topic 02.

**Quasi-identifiers.** A dataset classified as de-identified may still be personal data under GDPR
if the recipient can link it. The classification should follow the determination, not the label.

**Re-identification through linkage.** A consumer app that collects location and health data can
re-identify users through linkage. The classification should consider the combination, not the
fields in isolation.

**Free-text PHI.** Free-text fields can contain special category data even when the structured
fields do not. The classification should sample and review the text.

**Metadata leakage.** A regulatory submission or a public register can leak metadata. The
publication review should include file properties and author fields.

**Consent withdrawal.** A regime may require withdrawal to be as easy as consent; another may
permit processing without consent. The design should meet the stricter rule and document the
process.

**Minors and deceased patients.** Age thresholds and representative rules differ by regime and
jurisdiction. The map should record the age of digital consent and the representative rules per
market.

**Cross-border transfer.** A transfer can be lawful under one regime and not another. The map
should include the transfer mechanism, the destination, and the transfer impact assessment.

**Changing entity structure.** An acquisition, spin-out, or reorganisation can change the role and
the obligations. The map should be reviewed on any corporate change.

**Enforcement discretion.** A regulator's current priorities are not a safe harbour. The map should
record guidance and enforcement trends as context, not as exemptions.

## 6. Worked example

### Setting

SymptomCheck is a direct-to-consumer symptom-triage app. Users enter symptoms, the app suggests a
likely urgency level, and it recommends whether to seek emergency care, see a clinician, or
self-care. The company sells a clinician-facing dashboard to employers and, through a partnership,
to a health plan. It plans to launch in the United States and the European Union.

The company's regulatory lead builds a map. The company is not a hospital and has assumed it is
"not a HIPAA company". The app's claims include "diagnose" and "treat", which the lead flags as a
possible medical-device claim in both jurisdictions.

### Step 1: entity and role

| Flow | Entity role | Regimes | Note |
|---|---|---|---|
| Consumer app, direct users | Controller (GDPR); consumer product (US) | FTC Act; FTC HBNR; state consumer health laws; GDPR | No HIPAA unless a covered entity is involved |
| Employer dashboard | Controller or processor for the employer | GDPR if EU employees; state laws; ADA | Employer may be a health plan |
| Health plan partnership | Business associate (HIPAA) | HIPAA Security and Privacy Rules; BAA | The company handles PHI for the plan |
| EU launch | Controller; possible device manufacturer | GDPR; MDR; AI Act | Device classification depends on claims |

The health-plan flow is the one the company missed. A vendor that handles PHI for a health plan is a
business associate, even if its consumer app is not a covered entity. The company needs a BAA, the
Security Rule safeguards, and breach notification to the plan.

### Step 2: data and activity

The app collects symptoms, age, sex, medications, location, and device data. Symptoms and
medications are health data; location combined with symptoms can be sensitive. The activity is
triage (a health service), marketing to employers, and model training on user data. Model training
is a secondary use that requires its own basis in the EU and a review under the FTC Act for
deceptive claims.

### Step 3: the applicability matrix

The lead scores 12 obligations, with weight 3 for a market-entry or high-risk obligation, 2 for a
significant obligation, and 1 for a documentation or lower-risk obligation. Status is 2 for met, 1
for partial, and 0 for missing.

| # | Obligation | Regime | Weight | Status | Evidence |
|---:|---|---|---:|---:|---|
| 1 | BAA with the health plan | HIPAA | 3 | 0 | None |
| 2 | Security Rule safeguards for the plan flow | HIPAA | 3 | 0 | None |
| 3 | Health Breach Notification Rule process | FTC | 2 | 0 | None |
| 4 | No deceptive health claims | FTC Act | 2 | 1 | Marketing review in progress |
| 5 | Article 9 basis for health data | GDPR | 3 | 1 | Consent draft |
| 6 | Article 13/14 transparency | GDPR | 2 | 2 | Privacy notice live |
| 7 | DPIA for high-risk processing | GDPR | 2 | 1 | Draft |
| 8 | MDR classification and CE marking | MDR | 3 | 0 | Not started |
| 9 | AI Act high-risk readiness | AI Act | 3 | 0 | Not started |
| 10 | Washington consumer health consent | Washington | 2 | 0 | Not started |
| 11 | CCPA/CPRA disclosures | California | 1 | 1 | Partial |
| 12 | State breach notification | States | 1 | 1 | Partial |

The weighted coverage is:

- Achieved = sum(weight x status) = 0 + 0 + 0 + 2 + 3 + 4 + 2 + 0 + 0 + 0 + 1 + 1 = 13.
- Maximum = sum(weight x 2) = 54.
- Coverage = 13 / 54 = 24.1 percent.

The number is not a compliance percentage. It is a management signal that three quarters of the
weighted evidence is missing, concentrated in the highest-weight obligations.

### Diagnostic 1: the HIPAA role was misclassified

The company's belief that it is "not a HIPAA company" is true for the consumer flow and false for
the health-plan flow. The two HIPAA obligations carry weight 6 of the 27 total weight, and both are
at zero. The consequence is not only a documentation gap: without a BAA, the health plan cannot
lawfully share PHI with the company, and the partnership cannot launch.

The correction is to treat the plan flow as a separate product with its own role, its own BAA, and
its own Security Rule evidence. The consumer flow remains outside HIPAA. The map should show both,
because a single company-level label would be wrong in both directions.

### Diagnostic 2: the device and AI Act path

The app's claims ("diagnose", "treat") support a medical-device classification under the MDR, which
requires a conformity assessment and CE marking before the EU launch. Separately, AI that is a
medical device falls under Annex I of the AI Act, and the high-risk obligations for Annex I systems
apply from 2 August 2028 under the amended timeline. From October 2026, that is 22 months away.

The company has 0 percent readiness on both. The diagnosis is not that the company is late; it is
that the EU launch cannot happen without a device pathway, and the AI Act work needs to start now
to be ready when the Annex I obligations apply. The two are linked: the technical documentation and
risk management for the MDR can be reused for the AI Act, if they are designed together.

### Vary one assumption: wellness positioning

Suppose the company removes the clinical claims and positions the app as a wellness tool that
"helps you decide when to seek care" without diagnosing or treating. The MDR and AI Act
obligations may no longer apply, and the matrix changes:

- Remove obligations 8 and 9 (weight 6, maximum 12).
- Achieved remains 13; maximum becomes 42.
- Coverage becomes 13 / 42 = 31.0 percent.

The conclusion changes from "two market-entry barriers, one of which is a device pathway" to "one
market-entry barrier, with a residual classification risk". The residual risk is real: regulators
look at the claims and the intended purpose, not only the marketing label, and a wellness app that
tells a user not to seek care may still be a device. The repositioning reduces the obligation set
and does not eliminate the need for a classification analysis.

### The sequence

The lead sequences the work by deadline and dependency:

1. **Months 1-3:** HIPAA BAA and Security Rule evidence for the plan flow; FTC HBNR process.
2. **Months 1-6:** Washington consent and privacy notice; CCPA/CPRA completion; breach process.
3. **Months 1-9:** MDR classification, quality management system, and technical documentation if
   the device claims remain.
4. **Months 6-22:** AI Act readiness, reusing the MDR risk management file; conformity assessment
   planning.
5. **Months 3-12:** DPIA completion and Article 9 basis for the EU consumer flow.

The sequence is the deliverable that a board can fund. The coverage score tells the board how much
is missing; the sequence tells it what to do first.

### The plan and the board

The board does not need the matrix; it needs the three or four decisions that the matrix implies:
whether to enter the EU market with a device pathway, whether to accept the HIPAA business
associate role for the plan channel, whether to reposition as wellness, and what budget is
required. The regulatory lead should present the options with the coverage and the sequence, and
the board should record the decision.

A board that approves a launch without seeing the obligation set has approved an assumption, not a
plan. The map is the artefact that converts the assumption into a decision.

### The cost of each pathway

The device pathway is expensive: a quality management system, clinical evaluation, technical
documentation, and a notified body assessment can take a year or more and cost six figures. The
wellness pathway is cheaper and narrower, and it may not support the clinical claims the product
wants to make. The business associate pathway requires a BAA, Security Rule evidence, and a breach
process, which is a smaller but non-trivial programme.

The board's decision is a comparison of market access, cost, and risk. The map supplies the
obligation set; the finance team supplies the cost; the commercial team supplies the revenue. The
decision is not a regulatory decision alone.

### What the score would look like with evidence

If the company closes the HIPAA obligations (status 2 for both, adding 12 to the achieved score) and
the Washington obligation (status 2, adding 4), the achieved score becomes 13 + 12 + 4 = 29, and
the coverage becomes 29 / 54 = 53.7 percent. The score rises without changing the device pathway,
which remains at zero. The example shows why the per-regime view matters: a headline improvement
can leave the market-entry blocker untouched.

### The decision

The lead recommends: do not launch the EU consumer product until the device classification is
resolved; launch the US employer dashboard only after the HIPAA business associate programme and
the Washington consent work are complete; and start the MDR/AI Act work now, because the 2028 date
is fixed and the technical file takes longer than a quarter.

The recommendation is a sequencing decision, not a legal conclusion. It gives the board a path that
respects the obligations without pretending that the gaps can be closed overnight.

### What the board saw

The board saw a one-page summary: the US employer channel can launch after three months of HIPAA
and state work; the EU consumer product waits for the device classification; the AI Act readiness
starts now and is funded over two years. The summary did not include the matrix; it included the
decisions and the dates. That is the right level for a board and the wrong level for a regulator,
which is why both artefacts exist.

## 7. Code, query, or calculation

The script below computes the weighted coverage, the per-regime gaps, and the wellness scenario. It
uses only the standard library. The obligations and weights are the ones in Section 6.

```python
"""Regulatory applicability and readiness coverage for a synthetic product."""

OBLIGATIONS = [
    {"id": "HIPAA-BAA", "regime": "HIPAA", "weight": 3, "status": 0},
    {"id": "HIPAA-SEC", "regime": "HIPAA", "weight": 3, "status": 0},
    {"id": "FTC-HBNR", "regime": "FTC", "weight": 2, "status": 0},
    {"id": "FTC-S5", "regime": "FTC", "weight": 2, "status": 1},
    {"id": "GDPR-ART9", "regime": "GDPR", "weight": 3, "status": 1},
    {"id": "GDPR-ART13", "regime": "GDPR", "weight": 2, "status": 2},
    {"id": "GDPR-DPIA", "regime": "GDPR", "weight": 2, "status": 1},
    {"id": "MDR-CE", "regime": "MDR", "weight": 3, "status": 0},
    {"id": "AIACT-HR", "regime": "AI Act", "weight": 3, "status": 0},
    {"id": "WA-MHMDA", "regime": "Washington", "weight": 2, "status": 0},
    {"id": "CCPA", "regime": "California", "weight": 1, "status": 1},
    {"id": "STATE-BREACH", "regime": "States", "weight": 1, "status": 1},
]


def coverage(obligations):
    achieved = sum(o["weight"] * o["status"] for o in obligations)
    maximum = sum(o["weight"] * 2 for o in obligations)
    return achieved, maximum, achieved / maximum


def drop_regimes(obligations, regimes):
    return [o for o in obligations if o["regime"] not in regimes]


if __name__ == "__main__":
    a, m, pct = coverage(OBLIGATIONS)
    print(f"base: achieved={a}, maximum={m}, coverage={pct:.1%}")

    for regime in sorted({o["regime"] for o in OBLIGATIONS}):
        subset = [o for o in OBLIGATIONS if o["regime"] == regime]
        _, sub_max, sub_pct = coverage(subset)
        print(f"  {regime}: {sub_pct:.1%} of its own maximum "
              f"(weight={sum(o['weight'] for o in subset)})")

    wellness = drop_regimes(OBLIGATIONS, {"MDR", "AI Act"})
    a2, m2, pct2 = coverage(wellness)
    print(f"wellness positioning: achieved={a2}, maximum={m2}, "
          f"coverage={pct2:.1%}")

    hipaa = [o for o in OBLIGATIONS if o["regime"] == "HIPAA"]
    print(f"HIPAA obligations met: {sum(o['status'] for o in hipaa)}/"
          f"{len(hipaa)}")
    print(f"months from 2026-10 to 2028-08: "
          f"{(2028 - 2026) * 12 + (8 - 10)}")
```

Expected output, verified on Python 3.11:

```text
base: achieved=13, maximum=54, coverage=24.1%
  AI Act: 0.0% of its own maximum (weight=3)
  California: 50.0% of its own maximum (weight=1)
  FTC: 25.0% of its own maximum (weight=4)
  GDPR: 64.3% of its own maximum (weight=7)
  HIPAA: 0.0% of its own maximum (weight=6)
  MDR: 0.0% of its own maximum (weight=3)
  States: 50.0% of its own maximum (weight=1)
  Washington: 0.0% of its own maximum (weight=2)
wellness positioning: achieved=13, maximum=42, coverage=31.0%
HIPAA obligations met: 0/2
months from 2026-10 to 2028-08: 22
```

Three readings of the output. The base score shows a large gap, concentrated in HIPAA, MDR, the AI
Act, and Washington. The per-regime lines show that GDPR is the most mature area and the device
pathway the least. The wellness scenario raises the score by removing obligations, not by adding
evidence, which is a reminder that a coverage score can be improved by changing the scope rather
than the work.

The weights are judgements. A company that treats the EU launch as optional would weight MDR and
the AI Act lower; a company that treats the employer channel as strategic would weight HIPAA
higher. The weights should be agreed by the executive who owns the launch decision, and recorded
with the rationale.


### Reading the coverage

The coverage score is a summary; the per-regime lines are the diagnosis. A score of 24.1 percent
with 0 percent in HIPAA and the device pathway is a different problem from the same score spread
evenly. The first requires a role correction and a market-entry decision; the second requires a
programme of documentation.

The weights are the most contestable part of the model. A weight of 3 for the AI Act reflects
market-entry risk in 2028; a company that does not plan to enter the EU would set it lower. The
weights should be reviewed with the executive and recorded in the map, so that a reader can see
why the score is what it is.

The score is a management tool, not a legal measure. It should be presented with the gap list and
the sequence, never as a standalone percentage.

## 8. Data, standards, and regulatory context

This section summarises the regimes a biomedical data product is most likely to encounter. It is a
map, not a legal analysis; each entry names what the regime constrains and what it does not.

### United States

**HIPAA.** The Privacy Rule (45 CFR 164 Subpart E), Security Rule (Subpart C), and Breach
Notification Rule (Subpart D) apply to covered entities and business associates. They constrain the
use and disclosure of PHI, require administrative, physical, and technical safeguards, and set
breach notification deadlines. They do not reach a consumer app that is not acting for a covered
entity, and they do not generally regulate the sale of de-identified data.

**FTC Act and Health Breach Notification Rule.** Section 5 of the FTC Act prohibits unfair or
deceptive acts or practices, which reaches privacy and security claims by consumer health
companies. The Health Breach Notification Rule (16 CFR Part 318) requires vendors of personal
health records and related entities to notify individuals, the FTC, and sometimes the media after
a breach of unsecured identifiable health information. The amended rule took effect in 2024 and
reaches health apps. It is not HIPAA, and a company can be subject to both.

**Medical devices.** The FDA regulates devices through premarket review (510(k), De Novo, PMA) and
post-market requirements. AI-enabled devices may use a predetermined change control plan to
pre-authorise certain updates. The FDA's cybersecurity guidance expects a threat model, a software
bill of materials, and security risk management in the premarket submission.

**Information blocking.** The 21st Century Cures Act information-blocking rule (45 CFR Part 171)
requires sharing of electronic health information with patients and other requesters, subject to
exceptions. It pushes toward access, in tension with minimum necessary and with privacy
restrictions; the exceptions are the place where the two are reconciled.

**State law.** State comprehensive privacy laws, consumer health privacy laws (for example,
Washington's My Health My Data Act), provider confidentiality laws (for example, California's
Confidentiality of Medical Information Act), breach notification laws, and genetic privacy laws add
requirements. Where state law is more protective, it generally governs, subject to HIPAA
preemption analysis.

### European Union

**GDPR.** The GDPR applies to controllers and processors processing personal data of people in the
EU, regardless of where the entity is established, where Article 3(2) applies. Health data are
special category data under Article 9. The GDPR requires a lawful basis, transparency, data
protection by design, a DPIA for high-risk processing, records of processing, and compliance with
data subject rights. Chapter V governs transfers.

**Medical Device Regulation.** Regulation (EU) 2017/745 governs medical devices, including
software with a medical purpose. It requires classification, a quality management system, clinical
evaluation, technical documentation, and CE marking, with post-market surveillance.

**AI Act.** Regulation (EU) 2024/1689 entered into force on 1 August 2024. The timeline has been
amended by the Digital Omnibus on AI, adopted in 2026. The current milestones are: prohibitions
from 2 February 2025; general-purpose AI obligations from 2 August 2025; most remaining rules,
including transparency and enforcement, from 2 August 2026; high-risk obligations for Annex III
systems from 2 December 2027; and high-risk obligations for Annex I products, including medical
devices, from 2 August 2028. A medical device that is an AI system is therefore in the Annex I
cohort, not the 2026 cohort. The Act requires risk management, data governance, technical
documentation, human oversight, accuracy and robustness, and post-market monitoring for high-risk
systems.

**European Health Data Space.** Regulation (EU) 2025/327 entered into force on 26 March 2025. Its
governance provisions apply from 26 March 2027, and most of the secondary-use chapter from 26 March
2029. It creates health data access bodies, data permits, and secure processing environments, and
it prohibits re-identification. For a product that uses secondary data, the EHDS changes the access
route, not only the compliance checklist.

**Data Governance Act.** Regulation (EU) 2022/868 provides a framework for data intermediation and
data altruism, and applies from September 2023. It is relevant to data trusts and data-sharing
intermediaries.

### United Kingdom

UK GDPR and the Data Protection Act 2018 apply after Brexit, with the ICO as regulator. Medical
devices are regulated by the MHRA under the UK MDR, with a transition to the new UKCA regime. The
UK is not bound by the EU AI Act or the EHDS, and its approach to AI regulation is
principles-based and sector-led.

**The map is a board artefact.** A board that understands the obligation set can make a market-entry
decision with its eyes open; a board that sees only a launch plan cannot. The regulatory lead
should present the map in the same meeting as the budget, because the obligations are the reason
the budget is what it is.

**The map should name the enforcement body.** An obligation without a regulator is a slogan. The
map should record who enforces each regime (HHS OCR, FTC, state attorneys general, supervisory
authorities, notified bodies, the AI Office) because the enforcement body determines the
consequence and the relationship.

**The map should include the date of the last change in the law.** A map that is accurate but
undated cannot be trusted. The version, the date, the reviewer, and the next review date belong on
the first page.

**The map's owner should be senior enough to say no.** A regulatory lead who reports to the product
manager cannot block a launch. The owner should have a reporting line to the executive who owns the
risk, and the map should be reviewed by that executive before a launch decision.

A map that is not maintained becomes a source of false assurance. The review date is not a
formality; it is the date on which the organisation checks whether the assumptions in the map are
still true. The version history is the evidence that the check happened.

> **Note:** Regulatory content in this file is educational and is not legal advice. The
> applicability of a regime depends on facts that require counsel, and the timelines in the AI Act
> and EHDS have changed and may change again.

The classification to keep straight: HIPAA, GDPR, the AI Act, the MDR, and the FTC rules are
binding within their scope; NIST frameworks and ISO standards are voluntary; guidance from a
regulator is an interpretation, not a statute; and a safe harbour is a legal provision with
conditions. A vendor that says "HIPAA compliant" has made a claim about a scope that may not
include the buyer's flow.

## 9. Industry practice and case evidence

### GoodRx, 2023

The FTC's first enforcement action under the Health Breach Notification Rule resulted in a USD 1.5
million civil penalty against GoodRx in February 2023. The FTC alleged that GoodRx shared users'
health information with advertising platforms, contrary to its privacy promises, and failed to
notify consumers and the FTC of the breach. The order banned disclosure of health information for
advertising and required affirmative express consent for other disclosures.

The case is the clearest example of the consumer-health gap: GoodRx was not a HIPAA covered entity,
and it was still subject to the FTC Act and the HBNR. A company that builds to HIPAA alone would
not have the consent, notification, or third-party deletion controls the order required.

### BetterHelp, 2023

The FTC's final order against BetterHelp in July 2023 required a USD 7.8 million payment for
consumer refunds and banned sharing health data for advertising. The FTC alleged that BetterHelp
shared email addresses, IP addresses, and health questionnaire answers with advertising platforms
after promising confidentiality. The order required a privacy programme with biennial independent
assessments for 20 years.

The case adds two lessons. First, the data that triggered the action were not clinical records;
they were email addresses, IP addresses, and questionnaire answers, which many teams would classify
as marketing data. Second, the remedy was a governance programme, not only a fine.

### 23andMe, 2023

The genetic testing company confirmed that a credential-stuffing attack gave attackers access to
data belonging to approximately 6.9 million users, including health-related information for a
subset. The incident illustrates the regulatory complexity of consumer genetic data: it is health
data, it is personal data, it is subject to state genetic privacy laws, and it may be subject to
the FTC Act and the HBNR, depending on the entity and the flow.

### Change Healthcare, 2024

The Change Healthcare incident reached 192.7 million individuals, the largest US health breach on
record. It illustrates the business associate chain: a single processor compromise affected
covered entities across the country, each of which had its own notification obligation and its own
regulatory exposure. The incident also shows that a cyber incident can become a continuity incident
for the health system, which is a different regulatory and clinical problem.

### What mature programmes do

- **Maintain a regulatory map per flow.** Not per company, and not per product name. The map is
  reviewed on change and owned by a named role.
- **Separate the legal analysis from the control design.** Counsel determines applicability; the
  team designs controls that satisfy the strictest applicable requirement.
- **Build evidence as they go.** The DPIA, the risk analysis, the technical file, and the BAA are
  produced during design, not reconstructed at audit.
- **Track the timelines.** The AI Act and EHDS dates are fixed and staged; a programme that starts
  in 2027 cannot meet the 2028 Annex I date without a plan.
- **Treat enforcement as a signal.** GoodRx, BetterHelp, and the state AG actions show where
  regulators are looking, even when the guidance has not changed.

**Enforcement is sector-specific.** The FTC has been active on consumer health data and advertising
pixels; HHS OCR on HIPAA risk analysis and breach notification; state attorneys general on consumer
privacy and genetic data; EU supervisory authorities on transfers and legal basis; notified bodies
on device conformity. The map should record the enforcement pattern, because it tells the team
where the first challenge is likely to come from.

**The regime list is not the strategy.** Knowing that twelve obligations apply does not tell a
company what to build. The map becomes useful when it is paired with a sequencing decision, a
budget, and an owner. The strategy is the sequence, not the list.

**Enforcement is also a market signal.** A category of product that receives an enforcement action
often sees a wave of compliance activity from competitors, insurers, and procurement teams. The map
should record the enforcement action and the design change it implies, even when the organisation
was not a party.

**The map should be tested against a real decision.** The test is not whether the document is
complete; it is whether a product manager can use it to decide what to build next. If the map does
not change a roadmap item, a budget line, or a launch date, it is a reference document rather than
a management tool.

> **Note:** The case facts above are drawn from the cited public enforcement releases. Statements
> about what an organisation should have done differently are inference, labelled as such.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Single-regime thinking | "We are not a HIPAA company" | Consumer and state regimes ignored | FTC or state enforcement | Applicability per flow |
| Entity misclassification | Role assumed from company name | Obligation set wrong | Unlawful data flow | Role analysis per flow |
| Device claim ignored | Marketing says "diagnose" | MDR/FDA classification missed | Cannot launch; recall risk | Claims review; classification |
| AI Act timing wrong | Team plans for 2026 high-risk | Amended timeline not tracked | Missed 2028 readiness | Timeline register |
| Static map | No review date | Law and product change | Compliance drift | Change triggers |
| Evidence gap | No DPIA or technical file | Documentation deferred | Burden of proof lost | Evidence register |
| Conflict unresolved | Two consent forms, one process | Regimes not reconciled | One regime breached | Conflict table |
| Transfer assumed | "Encryption is enough" | Chapter V not analysed | Unlawful transfer | Transfer assessment |
| State law ignored | Only federal law mapped | Washington or California missed | State AG action | State matrix |
| Vendor claim accepted | "HIPAA compliant" | Scope mismatch | Buyer's flow unprotected | Contract and audit |
| Enforcement discretion treated as law | "They have not enforced it" | Risk tolerance without basis | Sudden action | Counsel review |
| Timeline missed | No critical path | Dependencies not sequenced | Launch delay | Sequenced plan |
| Documentation without evidence | Policies with no tests | Paper compliance | Audit finding | Evidence per obligation |
| Reorganisation not mapped | Spin-out or acquisition | Role changes | New obligations missed | Corporate change trigger |
| Budget without obligations | Launch plan has no compliance line | Cost discovered late | Delay or non-compliance | Cost per obligation |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The entity structure is stable | Role analysis depends on it | Acquisition or spin-out | Corporate change review |
| The data classification is correct | Regime applicability depends on it | New field or source | Field-level review |
| The activity is correctly described | Legal basis depends on it | Marketing changes the purpose | Purpose statement review |
| The AI Act timeline is current | Planning depends on it | Omnibus or new guidance | Commission timeline check |
| The device classification is right | Market entry depends on it | Claims or features change | Regulatory review |
| State law is tracked | US launch depends on it | New state statute | Legislative monitoring |
| The transfer analysis is current | EU data flows depend on it | New vendor or region | Transfer register |
| The evidence exists | Readiness depends on it | Documentation deferred | Evidence sample |
| The weights reflect risk | Prioritisation depends on them | Business priorities change | Executive sign-off |
| Counsel reviews the map | Legal conclusions depend on it | Team self-assesses | Legal sign-off |
| Enforcement climate is stable | Risk tolerance depends on it | New enforcement action | Regulatory monitoring |

Two limits are structural. First, a map is not a legal opinion; it is a structured list of
questions for counsel. Second, the map is a snapshot of a changing system: the law changes, the
product changes, and the entity changes. A map without a review date and an owner is already out of
date.

## 12. Exercises

The exercises build a map for a real or realistic product, then test the decisions that the map
implies. Exercise 2 is the core artefact; Exercise 7 is the one that turns it into a plan.

The exercises build a map for a real or realistic product, then test the decisions that the map
implies. Exercise 2 is the core artefact; Exercise 7 is the one that turns it into a plan. Each
exercise states a deliverable, criteria, and a time budget, and the outputs can be assembled into a
regulatory pack.

**Exercise 1 - Role classification (application, 40 minutes).**
For a hospital, an analytics vendor, a consumer app, and a device maker, classify the HIPAA and
GDPR roles per flow and name one obligation that changes with the role. Deliverable: table.
Criteria: per-flow analysis; no company-level label.

**Exercise 2 - Applicability matrix (application, 60 minutes).**
Build a 10-obligation matrix for a product you know, with regime, weight, status, and evidence.
Deliverable: matrix with a coverage calculation. Criteria: obligations specific; weights justified;
evidence named.

**Exercise 3 - Conflict resolution (design, 45 minutes).**
Resolve the consent, breach-timing, and erasure conflicts for a US-EU product. Deliverable: table
with options and decisions. Criteria: stricter rule identified; cost stated; design change named.

**Exercise 4 - AI Act timeline (recall, 30 minutes).**
List the AI Act milestones from entry into force to the Annex I high-risk date and state which
applies to a medical device. Deliverable: timeline. Criteria: dates correct per the amended
timeline; Annex I identified.

**Exercise 5 - Evidence register (design, 45 minutes).**
Design an evidence register for the matrix in Exercise 2: owner, artefact, date, review. Deliverable:
register. Criteria: one artefact per obligation; owner named; review date.

**Exercise 6 - Enforcement analysis (critique, 45 minutes).**
Read the GoodRx or BetterHelp order and identify three controls the company lacked. Deliverable:
500 words. Criteria: controls specific; link to the order's findings; relevance to a product.

**Exercise 7 - Sequencing plan (design, 45 minutes).**
Sequence the remediation for the Section 6 product by deadline and dependency. Deliverable:
Gantt-style list. Criteria: critical path; owners; dependencies.

**Exercise 8 - Change trigger (design, 30 minutes).**
Write the review triggers for a regulatory map. Deliverable: one page. Criteria: at least six
triggers; each with an owner and an action.

## 13. Self-check answers

**Answer 1.** A hospital is a covered entity and a controller for its clinical flows; an analytics
vendor is a business associate and a processor for the hospital's data, and a controller for its own
product; a consumer app is not a covered entity but is a controller under GDPR and subject to the
FTC Act; a device maker is a controller for patient data and a manufacturer under MDR/FDA. The
obligation that changes with the role is, for example, breach notification: a business associate
notifies the covered entity, while a controller notifies individuals and the regulator.

**Answer 2.** A correct matrix names specific obligations ("Article 35 DPIA", "BAA", "MDR
conformity assessment"), assigns a weight with a reason, records status with evidence, and computes
the weighted coverage. A matrix that lists statutes without obligations does not pass.

**Answer 3.** Consent: separate instruments, GDPR consent for EU and HIPAA authorisation for US,
with a shared record of scope. Breach timing: design to 72 hours, with a single incident process
and a phased notice. Erasure: erase where the law requires and retain where a legal obligation
requires, document the exception, and give the individual a clear explanation.

**Answer 4.** Entry into force 1 August 2024; prohibitions 2 February 2025; GPAI obligations 2
August 2025; most remaining rules and enforcement 2 August 2026; Annex III high-risk 2 December
2027; Annex I high-risk, including medical devices, 2 August 2028. The dates reflect the amended
timeline adopted in 2026.

**Answer 6.** GoodRx lacked affirmative express consent for sharing health data for advertising, a
third-party deletion mechanism, and a breach notification process under the HBNR. BetterHelp lacked
similar controls and a privacy programme with independent assessments. The exercise should tie each
gap to a specific finding in the order.

**Rubric for Exercise 2.** Full credit requires ten specific obligations, justified weights, status
with evidence, and a correct coverage calculation. A matrix without evidence does not pass.

**Rubric for Exercise 3.** Full credit requires a conflict table, the stricter rule identified, the
cost stated, and a design change. A memo that says "comply with both" without a design does not
pass.

**Rubric for Exercise 7.** Full credit requires a critical path, dates, owners, and dependencies,
with market-entry items before launch. A list of tasks without sequencing does not pass.

**Answer 5.** The EHDS is Regulation (EU) 2025/327, in force on 26 March 2025. Governance provisions
apply from 26 March 2027 and most of the secondary-use chapter from 26 March 2029. It creates
health data access bodies, data permits, and secure processing environments, and it prohibits
re-identification. It changes the access route for secondary use; it does not replace the GDPR.

**Answer 7.** Identify the stricter requirement, design to it, document the rationale, and verify
that the design change is implemented. Where the conflict cannot be resolved by the stricter rule
(for example, a legal retention obligation versus an erasure request), document the exception and
explain it to the individual. A conflict resolved only in a legal memo is not resolved.

**Answer 9.** Ask for the role analysis, the BAA, the risk analysis, the subprocessor list, the
breach notification terms, the security evidence, and the audit rights. A HIPAA compliance claim
describes the vendor's own scope; it does not establish that the buyer's flow is covered. The
buyer's obligation is to verify the scope against its own data flows.

**Rubric for Exercise 5.** Full credit requires one artefact per obligation, an owner, a date, and a
review date. A register that lists policies rather than artefacts does not pass.

**Rubric for Exercise 8.** Full credit requires at least six triggers (new jurisdiction, new data
type, new activity, new entity structure, product change, enforcement action) with an owner and an
action for each. A list of dates does not pass.

**Answer 10.** Change triggers: a new jurisdiction, a new data type, a new activity, a new entity
structure, a material product change, a new vendor or transfer, a relevant enforcement action, and
the annual review. Each trigger should have an owner and an action, and the review should produce a
new version of the map with a date.

**Rubric for Exercise 4.** Full credit requires the entry-into-force date, the prohibitions, the
GPAI date, the 2026 general application, the Annex III date, and the Annex I date, with a statement
that medical devices are Annex I. A list that puts all high-risk systems in 2026 does not pass.

**Rubric for Exercise 6.** Full credit requires three specific controls, a link to the order's
findings, and relevance to a product. A summary of the order without a control analysis does not
pass.

**Rubric for Exercise 1.** Full credit requires a per-flow role analysis for all four entities and
one obligation that changes with the role. A company-level label does not pass.

## 14. Interview questions

**1. Is a consumer health app subject to HIPAA?**
Strong answer: generally not, unless it is acting for a covered entity as a business associate; but
it may be subject to the FTC Act, the HBNR, state consumer health laws, and GDPR. Weak answer:
"yes, because it handles health data".

**2. What makes a vendor a business associate?**
Strong answer: it performs a function or service for a covered entity involving PHI, with a BAA;
the label in the contract does not decide it, the facts do. Weak answer: "if it signs a BAA".

**3. How do HIPAA and GDPR breach deadlines interact?**
Strong answer: HIPAA is 60 days from discovery; GDPR is 72 hours to the regulator and without undue
delay to individuals for high risk; design to the shorter, stricter process. Weak answer: "HIPAA
preempts GDPR".

**4. When does the AI Act apply to a medical device?**
Strong answer: a medical device that is an AI system falls under Annex I high-risk, with the
obligations applying from 2 August 2028 under the amended timeline; MDR applies earlier for market
access. Weak answer: "from 2026".

**5. What is the EHDS and when does it apply?**
Strong answer: Regulation (EU) 2025/327, in force 26 March 2025, with governance from 2027 and most
secondary use from 2029; it creates access bodies, permits, and secure processing environments.
Weak answer: "a data-sharing law".

**6. What is the difference between a controller and a processor?**
Strong answer: the controller determines purposes and means; the processor acts on instructions;
the role determines obligations and liability, and a contract cannot change the facts. Weak answer:
"the controller owns the data".

**7. How do you resolve a conflict between two regimes?**
Strong answer: identify the stricter requirement, design to it, document the rationale, and check
that the design change is implemented, not only written. Weak answer: "follow the local law".

**8. What is a regulatory map?**
Strong answer: a per-flow document with entity role, data and activity classification, obligations,
conditions, conflicts, readiness, evidence, and review date. Weak answer: "a list of laws".

**9. What would you do if a vendor claims HIPAA compliance?**
Strong answer: ask for the role, the BAA, the risk analysis, the subprocessor list, the breach
terms, and the evidence; the claim does not cover the buyer's flow. Weak answer: "accept the
certificate".

**10. How do you keep the map current?**
Strong answer: change triggers (law, product, entity, jurisdiction, enforcement) plus an annual
review, with an owner and a change log. Weak answer: "annual review".

**15. How do you decide whether to be a device or a wellness product?**
Strong answer: the intended purpose and the claims determine the classification; a wellness label
does not override clinical claims, and the decision should be made with regulatory counsel before
the marketing is written. Weak answer: "call it wellness".

**16. What is the provider/deployer distinction in the AI Act?**
Strong answer: the provider places the system on the market and bears the conformity obligations;
a deployer uses it, and can become a provider if it substantially modifies the system or puts it
under its own name. Weak answer: "the vendor is always the provider".

**17. What is the difference between a guidance document and a regulation?**
Strong answer: a regulation is binding law; guidance is a regulator's interpretation, which is
persuasive but not binding, and can change; a compliance programme should follow the law and treat
guidance as evidence of expectations. Weak answer: "guidance is law".

**18. How do you manage a multi-country launch?**
Strong answer: a per-country applicability matrix, a lead jurisdiction for the core design, local
counsel for variations, and a single evidence register that maps the common artefacts to each
country's requirements. Weak answer: "use the strictest law for everything".

**19. What is the first thing you would do for a new market?**
Strong answer: identify the entity role, the data class, and the activity in that market, then list
the regimes and the conditions, then get local counsel for the differences; do not assume the home
market's design transfers. Weak answer: "translate the privacy policy".

**20. How do you present a regulatory gap to a board?**
Strong answer: the decision, the options, the cost, the deadline, and the consequence of inaction;
the matrix is an appendix, not the presentation. Weak answer: "show the matrix".

**Rubric for Exercise 8.** Full credit requires at least six triggers with an owner and an action
for each. A list of dates does not pass.

## 15. Key terms

- **Applicability** - whether a regime applies to an entity, data, and activity.
- **Business associate** - an entity handling PHI for a covered entity.
- **CE marking** - EU conformity mark.
- **Conditional requirement** - an obligation that applies only if a condition is met.
- **Conformity assessment** - the process of demonstrating compliance with a product regime.
- **Consumer health data** - health data collected by a consumer product.
- **Controller** - the party determining purposes and means.
- **Covered entity** - a HIPAA-covered plan, clearinghouse, or provider.
- **DPIA** - data protection impact assessment.
- **EHDS** - European Health Data Space Regulation.
- **Enforcement discretion** - a regulator's choice of priorities.
- **FTC HBNR** - Health Breach Notification Rule.
- **GDPR** - General Data Protection Regulation.
- **High-risk AI** - an AI system meeting the AI Act criteria.
- **HIPAA** - Health Insurance Portability and Accountability Act.
- **Information blocking** - interference with access or exchange of health information.
- **Joint controller** - controllers determining purposes jointly.
- **Lead authority** - the supervisory authority coordinating a case.
- **Medical device** - a product with a medical purpose under MDR or FDA rules.
- **MDR** - EU Medical Device Regulation.
- **Personal data** - information relating to an identifiable person.
- **PHI** - protected health information.
- **Preemption** - federal law displacing state law.
- **Processor** - a party processing on behalf of a controller.
- **Readiness score** - weighted evidence coverage.
- **Records of processing** - GDPR Article 30 documentation.
- **Regime** - a body of law with definitions and enforcement.
- **Safe harbour** - a legal provision removing liability if conditions are met.
- **Special category data** - GDPR Article 9 data.
- **Transfer mechanism** - the legal basis for a cross-border transfer.
- **Voluntary framework** - a standard adopted by choice.
- **Provider** - the party placing an AI system on the market.
- **Deployer** - the party using an AI system under its authority.
- **Notified body** - an organisation assessing conformity under EU product law.
- **Regulatory change log** - the record of legal changes and their impact.
- **Substantial modification** - a change that can create a new regulatory obligation.
- **Lead supervisory authority** - the authority coordinating a cross-border case.
- **Market surveillance authority** - the body enforcing product law after launch.
- **Conformity assessment body** - the body assessing products against a regime.
- **Regulatory exposure** - the range of penalties and remedies for non-compliance.
- **Applicability analysis** - the reasoning that a regime does or does not apply.
- **Regulatory gap** - an obligation with no evidence or control.
- **Compliance artefact** - a document or test that evidences an obligation.
- **Review date** - the date by which the map must be reassessed.
- **Regulatory change trigger** - an event that requires the map to be updated.
- **Local counsel** - qualified advice in the relevant jurisdiction.
- **Regulatory strategy** - the sequencing of obligations, evidence, and market entry.
- **Evidence gap** - an obligation with no artefact or test.
- **Regulatory owner** - the named person accountable for the map.
- **Obligation set** - the specific requirements that apply to a flow.
- **Regulatory roadmap** - the sequenced plan to close the gaps.
- **Compliance owner** - the person accountable for evidence on an obligation.
- **Market-entry dependency** - an obligation that must be met before launch.

## 16. Sources

All URLs accessed 26 October 2026.

**United States**
- HHS Office for Civil Rights, *HIPAA Privacy Rule*.
  https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
- HHS Office for Civil Rights, *Breach Notification Rule*.
  https://www.hhs.gov/hipaa/for-professionals/breach-notification/index.html
- Federal Trade Commission, *Health Breach Notification Rule*, 16 CFR Part 318.
  https://www.ftc.gov/business-guidance/privacy-security/health-privacy
- Federal Trade Commission, *GoodRx enforcement action*, February 2023.
  https://www.ftc.gov/news-events/news/press-releases/2023/02/ftc-enforcement-action-bar-goodrx-sharing-consumers-sensitive-health-info-advertising
- Federal Trade Commission, *BetterHelp final order*, July 2023.
  https://www.ftc.gov/news-events/news/press-releases/2023/07/ftc-gives-final-approval-order-banning-betterhelp-sharing-sensitive-health-data-advertising
- FDA, *Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket
  Submissions*, 2023.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-system-considerations-and-content-premarket-submissions

**European Union**
- Regulation (EU) 2016/679 (GDPR). https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Regulation (EU) 2017/745 (Medical Device Regulation).
  https://eur-lex.europa.eu/eli/reg/2017/745/oj
- Regulation (EU) 2024/1689 (Artificial Intelligence Act).
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2025/327 (European Health Data Space).
  https://eur-lex.europa.eu/eli/reg/2025/327/oj
- Regulation (EU) 2022/868 (Data Governance Act).
  https://eur-lex.europa.eu/eli/reg/2022/868/oj
- European Commission, *AI Act timeline*, AI Act Service Desk.
  https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
- Council of the EU, *AI Act timeline*.
  https://www.consilium.europa.eu/en/policies/artificial-intelligence-act/timeline-artificial-intelligence/

**United Kingdom**
- UK Information Commissioner's Office, *UK GDPR guidance*.
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/
- MHRA, *Medical devices regulation*. https://www.gov.uk/government/organisations/medicines-and-healthcare-products-regulatory-agency

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_breach-and-incident.md`](09_breach-and-incident.md)
- Related topic: [`01_privacy-principles.md`](01_privacy-principles.md)
- Related topic: [`02_deidentification-and-anonymization.md`](02_deidentification-and-anonymization.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
