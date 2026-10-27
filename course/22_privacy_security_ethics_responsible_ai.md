# Module 22 — Privacy, Security, Ethics, and Responsible AI

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (14 lecture, 8 lab, 4 case clinic)
> **Track fit:** Common core; required for all tracks
> **Prerequisites:** Modules 02, 03, 21
> **Tools:** Threat-model template, data-classification scheme, DPIA template, incident runbook
> **Industry question:** "What can go wrong for people, partners, and the business — and what controls make the answer acceptable?"

---

## 1. Why industry cares

Supply chain analytics sits on a unusual concentration of sensitive data and physical consequence:

- **Commercial:** prices, costs, margins, volumes, contract terms, supplier capacity.
- **Personal:** customer addresses, phone numbers, delivery preferences, driver locations, warehouse worker productivity.
- **Operational:** routing, inventory positions, production schedules, facility layouts.
- **Partner:** EDI payloads, forecasts, inventory levels shared under contract.

A leak can reveal a company's cost structure to a competitor. A compromised transportation management system can redirect goods. A manipulated forecast can cause a shortage or a write-off. An allocation model can systematically disadvantage certain regions or small suppliers. An LLM assistant can leak a contract into a third-party service.

Security, privacy, and ethics are not a compliance chapter at the end. They are design constraints on every data product in this program.

## 2. Learning outcomes

By the end of this module, you can:

- Classify supply chain data by sensitivity and contractual constraint.
- Build a threat model for a planning or visibility system.
- Apply privacy principles to customer, worker, and supplier data.
- Assess a vendor or partner against security and privacy requirements.
- Identify ethical harms in allocation, pricing, labor, and routing.
- Evaluate an AI system against the EU AI Act risk classes and NIST AI RMF.
- Design human oversight and appeal paths.
- Write an incident response and breach-notification plan.
- Red-team an LLM or agent workflow.
- Explain trade-offs between visibility, privacy, and security.

## 3. Mental model

```text
   ASSET ──> THREAT ──> VULNERABILITY ──> IMPACT
     │          │             │              │
  data,      actor,        control         harm to
  model,     motive,       gap             people,
  system     capability                    partners,
                                           business
     │
     v
   CONTROL: prevent, detect, respond, recover, govern
```

Security asks "who can do what to which asset?" Privacy asks "whose data is this, and do they expect this use?" Ethics asks "who is helped, who is harmed, and who decides?" Responsible AI asks all three across the lifecycle.

### 3.1 The harm triangle

| Harm | Example | Who bears it |
|---|---|---|
| Physical | Contaminated or diverted goods | Consumers, patients |
| Financial | Price fixing, fraud, ransom | Customers, suppliers, shareholders |
| Privacy | Worker surveillance, address leak | Employees, customers |
| Fairness | Allocation biased to powerful buyers | Small suppliers, regions |
| Autonomy | Algorithmic scheduling without appeal | Workers |
| Dignity | Scoring people as "low value" | Customers, workers |
| Environmental | Routing that externalizes emissions | Communities |

## 4. Core concepts

### 4.1 Data classification

| Class | Examples | Handling |
|---|---|---|
| Public | Published lead times, catalog | Open |
| Internal | Aggregated KPIs | Authenticated access |
| Confidential | Costs, margins, forecasts | Need-to-know, encryption |
| Restricted | Personal data, contracts, credentials | Strict access, logging, DLP |
| Regulated | Health, payment, export-controlled | Legal review, retention rules |

Classify at the **column** level, not only the table. A shipment table is internal; the consignee address column is personal.

### 4.2 Threat modelling

Use a structured method such as STRIDE:

| Threat | Supply chain example | Control |
|---|---|---|
| Spoofing | Fake carrier update | Authentication, signed messages |
| Tampering | Altered EDI 856 | Integrity checks, signatures |
| Repudiation | Disputed delivery | Audit logs, proof of delivery |
| Information disclosure | Forecast leak | Access control, encryption |
| Denial of service | TMS outage | Redundancy, rate limits |
| Elevation of privilege | Vendor admin account | Least privilege, MFA |

Map trust boundaries: browser to API, planning service to warehouse, company to EDI partner, cloud to on-premise OT, model to third-party LLM.

### 4.3 Privacy principles

- **Lawful basis and purpose limitation:** collect for a stated purpose; do not repurpose silently.
- **Minimization:** use the least data needed; aggregate where possible.
- **Retention:** delete when no longer needed.
- **Accuracy:** correct errors that affect people.
- **Transparency:** tell people what is collected and why.
- **Security:** protect what is held.
- **Accountability:** document decisions.

| Data use | Privacy question |
|---|---|
| Delivery route optimization | Is driver location monitoring proportionate? |
| Customer segmentation | Is the segment based on sensitive traits? |
| Worker productivity | Is the measure fair and disclosed? |
| Supplier risk scoring | Is the score shared without context? |
| LLM assistant | Does the prompt contain personal or contract data? |

### 4.4 Security controls for planning systems

| Layer | Control |
|---|---|
| Identity | SSO, MFA, least privilege, joiner-mover-leaver |
| Network | Segmentation, private endpoints, egress control |
| Data | Encryption at rest and in transit, key management, masking |
| Application | Input validation, dependency scanning, SAST/DAST |
| Pipeline | Signed artifacts, secret scanning, branch protection |
| Vendor | Security questionnaire, right to audit, incident notification |
| OT | Network isolation, allow-listing, patching window |
| Monitoring | SIEM, anomaly detection, audit logs |

### 4.5 Third-party and supply chain security

The supply chain's own supply chain is an attack surface:

- **Software dependencies:** Log4j (2021) showed how a library can expose thousands of firms.
- **Managed service providers:** SolarWinds (2020) showed how a trusted vendor update can compromise customers.
- **Facility vendors:** the Target breach (2013) began with credentials stolen from an HVAC vendor.
- **Logistics partners:** a compromised carrier portal can expose shipments.

Require vendors to demonstrate patching, incident notification, and least-privilege access. Do not give a vendor broad network access for a narrow task.

### 4.6 Incident response

| Phase | Actions |
|---|---|
| Prepare | Roles, contacts, runbooks, backups |
| Detect | Monitoring, alerts, reports |
| Contain | Isolate systems, revoke credentials |
| Eradicate | Remove cause, patch |
| Recover | Restore, verify integrity |
| Learn | Post-incident review, control updates |

For personal data breaches, legal notification timelines apply (for example, GDPR's 72-hour supervisory notification where required). Do not guess; have counsel and a decision tree ready.

### 4.7 Ethical allocation

Allocation under scarcity is a moral decision, not only an optimization.

| Rule | Basis | Risk |
|---|---|---|
| Pro rata | Equal treatment | Ignores need |
| Priority by contract | Legal obligation | Entrenches power |
| Need-based | Equity | Hard to measure |
| Historical share | Stability | Freezes advantage |
| Auction | Willingness to pay | Excludes poor buyers |
| Optimization | Stated objective | Hides value choices |

The model can compute any of these. The choice must be made by accountable humans with the criteria stated publicly.

### 4.8 Labor and algorithmic management

Warehouse and delivery analytics can monitor pace, breaks, and route adherence. Risks:

- Surveillance without consent or proportionality.
- Metrics that incentivize unsafe behavior.
- Automation that removes discretion without appeal.
- Scheduling that ignores care responsibilities.

Controls: disclose metrics, avoid impossible targets, provide human review, and bargain with worker representatives where applicable.

### 4.9 Responsible AI frameworks

| Framework | Use |
|---|---|
| NIST AI RMF | Govern, map, measure, manage |
| EU AI Act | Risk classification and obligations |
| ISO/IEC 42001 | AI management system |
| OECD AI Principles | Human-centered values |
| Internal AI policy | Binding rules and approvals |

A practical AI review asks:

1. What decision does it influence?
2. Who is affected and can they appeal?
3. What data was used, and is it lawful?
4. What are the failure modes?
5. What human oversight exists?
6. How is it monitored after launch?
7. What is the exit path?

### 4.10 LLM and agent risks

| Risk | Example | Control |
|---|---|---|
| Prompt injection | A supplier email tells the agent to ignore rules | Treat retrieved text as data, not instructions |
| Data leakage | Contract pasted into a public model | Approved models, DLP, redaction |
| Hallucination | Fabricated lead time | Retrieval with citations, verification |
| Tool misuse | Agent issues a purchase order | Least privilege, approvals, spend limits |
| Excessive agency | Agent changes a forecast silently | Human-in-the-loop, audit log |
| Model inversion | Extracting training data | Access control, vendor terms |

An agent with write access to ERP is a production system with a natural-language interface. Treat it accordingly.

### 4.11 Privacy-enhancing technologies

| Technique | Use | Limit |
|---|---|---|
| Aggregation | Share demand without customer detail | Small cells re-identify |
| Pseudonymization | Replace identifiers | Linkage risk |
| Differential privacy | Add noise to aggregates | Utility loss |
| Secure multi-party computation | Joint analysis without sharing raw data | Cost and complexity |
| Federated learning | Train across partners | Coordination and leakage |
| Synthetic data | Share structure | Fidelity risk |

## 5. Data, standards, and workflow

### 5.1 Standards and regulation

| Instrument | Scope | Note |
|---|---|---|
| GDPR | Personal data in the EU | Extra-territorial reach |
| UK GDPR / DPA 2018 | U.K. personal data | Similar structure |
| CCPA/CPRA | California consumers | Rights and opt-out |
| EU AI Act | AI systems | Phased obligations |
| NIS2 | Critical infrastructure and supply chains | EU cybersecurity |
| DORA | Financial sector digital resilience | Includes ICT third parties |
| ISO 27001 | Information security management | Certification |
| ISO 27701 | Privacy management | Extension |
| ISO 28000 | Supply chain security | Management system |
| SOC 2 | Service organization controls | Vendor assurance |
| PCI DSS | Payment card data | If payments in scope |

### 5.2 Workflow for a new data product

1. Classify data and map flows.
2. Define purpose and lawful basis.
3. Complete a privacy impact assessment if personal data is involved.
4. Threat-model the system.
5. Review vendor and partner controls.
6. Define access, retention, and deletion.
7. Assess AI risk and human oversight.
8. Approve, document, and monitor.
9. Re-review on material change.

### 5.3 Vendor security questionnaire — minimum

- Certifications and recent audit reports.
- Data location and subprocessors.
- Encryption and key management.
- Access control and MFA.
- Incident history and notification terms.
- Vulnerability management and patching.
- Business continuity and exit.
- AI training use of customer data.
- Right to audit.

## 6. Industry case studies

### 6.1 Target — vendor credentials (2013)

**Context:** A retail breach exposed payment card data for tens of millions of customers.
**Intervention:** Attackers obtained credentials from an HVAC vendor and moved into the corporate network.
**Evidence:** Public investigation and reporting described the vendor pathway and network segmentation gaps.
**Outcome:** A landmark case in third-party risk and segmentation.
**What transferred:** vendor access must be scoped, monitored, and segmented.
**What did not:** the specific retail payment scope; the principle applies to any partner integration.

### 6.2 Maersk and NotPetya (2017)

**Context:** A global malware attack disrupted Maersk's IT systems.
**Intervention:** The worm spread through networks and encrypted systems.
**Evidence:** Maersk publicly described a multi-hundred-million-dollar impact and a rapid rebuild.
**Outcome:** A reference case for cyber risk in logistics.
**What transferred:** operational resilience depends on recovery, not only prevention.
**What did not:** the exact cost, which varies by firm and incident.

### 6.3 SolarWinds (2020)

**Context:** A compromised software update distributed malware to customers, including government and enterprises.
**Intervention:** Supply chain attack through a trusted vendor.
**Evidence:** Widely documented in public reporting and government advisories.
**Outcome:** Raised the profile of software supply chain security.
**What transferred:** trust in a vendor is not a control; verify artifacts and monitor behavior.
**What did not:** the assumption that only large vendors are targets.

### 6.4 Algorithmic scheduling and worker harm — composite

**Context:** A warehouse introduced productivity targets from a labor model.
**Intervention:** The model set pick rates by task and shift.
**Evidence:** Workers reported injury pressure and informal workarounds; the company adjusted targets after review.
**Outcome:** A cautionary case about metrics without human context.
**What transferred:** a labor model is a management decision with safety consequences.
**What did not:** the claim that all productivity analytics is harmful.

Use this composite as a teaching case, not a citation.

## 7. Labs

### Lab 22.1 — Data classification and flow map

Classify a demand-planning dataset by column. Map flows from source to model to report. Identify restricted data and cross-border transfers.

**Deliverable:** Classification table and flow diagram.

### Lab 22.2 — Threat model

Threat-model a control tower with EDI, TMS, and carrier APIs. Identify trust boundaries and the top five risks.

**Deliverable:** STRIDE table and mitigation plan.

### Lab 22.3 — Privacy impact assessment

Complete a DPIA for a delivery-route optimization that uses driver location. Identify necessity, proportionality, and safeguards.

**Deliverable:** DPIA with a recommendation.

### Lab 22.4 — AI risk review

Assess an LLM assistant that drafts supplier emails. Map EU AI Act risk, NIST functions, and human oversight.

**Deliverable:** Risk assessment and control plan.

### Lab 22.5 — Red team

Attempt prompt injection against a retrieval assistant using a synthetic supplier document. Record what succeeds and what the guardrails catch.

**Deliverable:** Red-team report and fixes.

### Lab 22.6 — Incident runbook

Write a runbook for a suspected forecast-data breach. Include detection, containment, notification, and recovery.

**Deliverable:** Runbook and tabletop exercise notes.

## 8. Assignment / project

Produce a security, privacy, and responsible-AI assessment for one planning or visibility system.

**Required components:**

1. System description and data flow.
2. Data classification and lawful basis.
3. Threat model with top risks.
4. Vendor and partner assessment.
5. Privacy impact assessment.
6. AI risk classification and human oversight.
7. Ethical harms and mitigation.
8. Incident response plan.
9. Residual risk statement and sign-off.

**Length:** 10–14 pages.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Data classification | 15% | Table-level | Column-level and flows | Cross-border and contractual nuance |
| Threat model | 20% | Generic list | STRIDE with boundaries | Prioritized by impact and likelihood |
| Privacy | 20% | Mentions GDPR | Purpose, minimization, DPIA | Proportionality and safeguards |
| AI and ethics | 20% | Generic principles | Risk class and oversight | Names affected groups and appeal |
| Incident response | 15% | Contacts only | Phased runbook | Tested and role-specific |
| Judgment | 10% | Checklist | Trade-offs stated | Residual risk owned |

## 10. Regulatory, ethical, and commercial lens

- **Contractual confidentiality:** sharing forecasts with a partner may be permitted only under a specific agreement.
- **Competition law:** exchanging competitor data through a platform can be illegal even if intermediated.
- **Export control:** location and technical data may be restricted.
- **Worker consultation:** algorithmic management may require information or bargaining.
- **Cross-border transfer:** mechanism and safeguards are required in many jurisdictions.
- **Insurance:** cyber coverage may exclude certain failures.
- **Reputation:** a privacy incident can cost more than the analytics program saves.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Broad vendor access | One account for everything | Convenience | Lateral movement | Scoped access, MFA |
| Shadow AI | Staff paste data into public tools | No policy | Leak | Approved tools, DLP |
| Data lake without classification | "It's internal" | Unknown sensitivity | Breach | Classify at ingestion |
| Consent theater | Notice nobody reads | Compliance form | Loss of trust | Purpose and minimization |
| Allocation by model alone | No policy | Optimization hides values | Inequity | Human decision and publication |
| Surveillance creep | New metric without review | Scope expansion | Labor harm | Review gate |
| Prompt injection | Agent trusts retrieved text | Confused deputy | Wrong action | Treat as data, least privilege |
| Untested recovery | Backups unverified | Silent failure | Extended outage | Restore drills |
| Vendor questionnaire only | No audit or monitoring | Paper assurance | Undetected risk | Right to audit, monitoring |
| Retention forever | No deletion | Storage default | Legal and breach exposure | Retention schedule |

## 12. Interview and discussion questions

1. How would you classify a shipment dataset? *(Hint: table vs column; consignee is personal.)*
2. What is a trust boundary in a control tower? *(Hint: company to carrier, cloud to OT.)*
3. Why is vendor access a supply chain risk? *(Hint: Target, SolarWinds.)*
4. What does purpose limitation mean for demand data? *(Hint: use only for stated purpose.)*
5. When is a DPIA required? *(Hint: likely high risk to individuals.)*
6. How do you allocate scarce stock ethically? *(Hint: stated policy, accountable humans, appeal.)*
7. What is prompt injection? *(Hint: retrieved text treated as instructions.)*
8. What human oversight is meaningful? *(Hint: information, authority, accountability.)*
9. How do you test incident response? *(Hint: tabletop and restore drill.)*
10. What is residual risk, and who owns it? *(Hint: remaining risk after controls; business owner.)*
11. How do you share data with a partner without losing control? *(Hint: aggregation, contracts, MPC.)*
12. What would make you stop an AI deployment? *(Hint: unmitigated harm, no oversight, unlawful data.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Data classification | Assigning sensitivity and handling rules |
| Threat model | Structured analysis of actors, assets, and controls |
| Trust boundary | Where data or control passes between parties |
| Least privilege | Minimum access needed for a task |
| DPIA | Data protection impact assessment |
| Purpose limitation | Using data only for the stated purpose |
| Minimization | Collecting only what is needed |
| Pseudonymization | Replacing identifiers while allowing linkage |
| Differential privacy | Noise-based protection of aggregates |
| Prompt injection | Malicious text that redirects an LLM |
| Excessive agency | An agent with more authority than needed |
| Human oversight | Meaningful review with authority to override |
| Residual risk | Risk remaining after controls |
| Incident response | Prepare, detect, contain, eradicate, recover, learn |
| Right to audit | Contractual access to vendor evidence |
| Segmentation | Isolating systems to limit lateral movement |
| Algorithmic management | Software that directs or evaluates workers |
| Allocation policy | Rule for distributing scarce supply |
| Model inversion | Recovering training data from a model |
| Data residency | Legal location of stored data |

## 14. Sources and further reading

### Standards and regulation

- GDPR, Regulation (EU) 2016/679. https://eur-lex.europa.eu/
- EU AI Act, Regulation (EU) 2024/1689. https://eur-lex.europa.eu/
- NIST, AI Risk Management Framework. https://www.nist.gov/itl/ai-risk-management-framework
- ISO/IEC 27001. https://www.iso.org/standard/27001
- ISO/IEC 27701. https://www.iso.org/standard/71670.html
- ISO 28000. https://www.iso.org/standard/79615.html
- NIS2 Directive. https://eur-lex.europa.eu/
- PCI DSS. https://www.pcisecuritystandards.org/

### Peer-reviewed and professional

- Shostack, A., *Threat Modeling: Designing for Security*.
- Nissenbaum, H., *Privacy in Context*.
- Floridi, L. et al., "AI4People," *Minds and Machines*, 2018.
- O'Neil, C., *Weapons of Math Destruction*.
- Zuboff, S., *The Age of Surveillance Capitalism*.
- Schneier, B., *Click Here to Kill Everybody*.

### Industry and cases

- Target, 2013 breach public reporting and Senate testimony.
- Maersk, NotPetya impact statements, 2017.
- SolarWinds, public advisories, 2020–2021.
- CISA, supply chain security guidance. https://www.cisa.gov/
- ENISA, threat landscape. https://www.enisa.europa.eu/

### Tools

- STRIDE and attack-tree templates.
- Open Policy Agent for policy as code.
- Great Expectations for data quality.
- DLP and secrets scanning in CI.

> **Next:** [Module 23](23_product_economics_leadership.md) connects risk and controls to the value case and adoption plan.
