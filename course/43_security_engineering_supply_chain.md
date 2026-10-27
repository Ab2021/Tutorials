# Module 43 — Security Engineering for Supply Chains

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Risk, Technology, and Operations tracks
> **Prerequisites:** Modules 05, 16, 20, 24
> **Tools:** Python, attack-tree and risk models, SBOM tooling, network segmentation labs
> **Industry question:** "The shipment is real, the invoice is real, the email is real — which one isn't?"

---

## 1. Why industry cares

A supply chain is an attack surface with many doors. It has:

- **Physical doors:** ports, warehouses, trucks, yards, and loading docks.
- **Digital doors:** EDI links, supplier portals, planning systems, and OT networks.
- **Commercial doors:** invoices, purchase orders, payment instructions, and contracts.
- **Human doors:** drivers, agency staff, contractors, and third-party support.

Modern attacks rarely hit the target directly. They enter through a supplier, a software dependency, a logistics partner, or a payment instruction. That is **supply chain security**, and it has become a board-level concern because a single compromise can stop production, expose data, or move money.

The analytics problem is **modelling the adversarial system**: who can reach what, through which path, at what probability and cost, and which control breaks the path most cheaply.

## 2. Learning outcomes

By the end of this module, you can:

- Map the attack surface of a physical and digital supply chain.
- Apply STRIDE and attack trees to supply chain threats.
- Distinguish IT and OT security and apply the Purdue model.
- Design segmentation and zero-trust controls across a supplier network.
- Assess third-party and supplier security posture.
- Read and use an SBOM and understand software supply chain attacks.
- Model cargo theft, fraud, and insider risk physically.
- Quantify risk with annualised loss expectancy and compare controls.
- Build an incident response and business continuity plan for a supply chain.
- Map controls to ISO 27001, NIST CSF, IEC 62443, TSA, and C-TPAT.

## 3. Mental model

```text
   THREAT ACTOR ──> ATTACK PATH ──> ASSET ──> IMPACT
      │                 │              │          │
   criminal,         supplier,      data,      production
   insider,          software,      money,     stop, safety,
   state,            physical,      product    reputation
   activist          payment

   CONTROLS break the path at some point:
   prevent ──> detect ──> respond ──> recover

   RISK = f(likelihood, impact)     ALE = SLE × ARO
```

Security is not a product; it is a set of **paths and controls**. The job is to find the cheapest control that reliably breaks the most consequential path.

### 3.1 The four security domains in a supply chain

| Domain | Examples | Typical control |
|---|---|---|
| Physical | Cargo theft, tampering, yard access | C-TPAT, seals, guards, CCTV |
| Cyber IT | EDI, portals, ERP, email | IAM, segmentation, patching |
| Cyber OT | Warehouse automation, PLCs, port equipment | IEC 62443 zones and conduits |
| Commercial | Invoice fraud, payment diversion | Verification, dual approval |

## 4. Core concepts

### 4.1 Threat modelling with STRIDE

STRIDE categorises threats:

| Letter | Threat | Supply chain example |
|---|---|---|
| S | Spoofing | Fake supplier email with new bank details |
| T | Tampering | Container seal broken and cargo substituted |
| R | Repudiation | Disputed delivery with no signed proof |
| I | Information disclosure | Pricing leaked through a supplier portal |
| D | Denial of service | Ransomware stops warehouse system |
| E | Elevation of privilege | Contractor account gains planning rights |

STRIDE is a checklist for a **data-flow diagram**: for each flow and store, ask which letters apply.

### 4.2 Attack trees

An attack tree decomposes a goal into sub-goals, with AND/OR nodes:

```text
   Goal: divert payment for a shipment
   ├── OR  compromise supplier email
   │     ├── AND phish accounts payable
   │     └── AND spoof domain
   ├── OR  obtain a genuine invoice early
   └── OR  insider in finance
```

Assigning probabilities to leaves gives a rough path probability. For an OR node with independent paths of 0.10 and 0.05, the combined probability is 1 − (0.9 × 0.95) = 0.145. For an AND node, multiply. The model is only as good as the estimates, but the **structure** reveals where a single control cuts several paths.

### 4.3 IT and OT security, and the Purdue model

Operational technology — PLCs, SCADA, warehouse control systems, port equipment — was not designed for the internet. The Purdue model layers it:

| Level | Function | Example |
|---|---|---|
| 0 | Process | Sensors, actuators |
| 1 | Basic control | PLC, RTU |
| 2 | Supervisory | SCADA, HMI |
| 3 | Operations management | MES, WMS |
| 3.5 | DMZ | Firewall, jump host |
| 4 | Enterprise | ERP, email |
| 5 | External | Cloud, supplier portals |

Zones and conduits (IEC 62443) define what may talk to what. A warehouse management system should not be able to send arbitrary commands to a conveyor PLC.

### 4.4 Zero trust and segmentation

Zero trust assumes the network is hostile:

- Authenticate and authorise every request.
- Least privilege, just-in-time access.
- Micro-segmentation between systems.
- Continuous verification, not one-time login.

In supply chains, the hard part is **third parties**. A supplier, a carrier, or a maintenance vendor may need access. The answer is not to trust their network; it is to give them the smallest access that does the job, through a controlled gateway.

### 4.5 Third-party and supplier security

Third-party risk management asks:

- What data and systems does the supplier touch?
- What is their security posture (certifications, incidents)?
- What contractual security terms apply?
- What is their own supply chain (fourth parties)?
- How is their access monitored and revoked?

A questionnaire is not an assessment. Evidence — audit reports, penetration test summaries, incident history — is.

### 4.6 Software supply chain security

Software is a supply chain of components. Attacks target:

- **Dependencies:** a compromised package or a typosquatted name.
- **Build systems:** injected code that is not in the source.
- **Updates:** a signed update channel abused to deliver malware.
- **Maintainers:** account takeover of a trusted project.

Controls:

- **SBOM** (software bill of materials): the list of components and versions.
- **SLSA** or equivalent build provenance.
- **Dependency pinning and scanning.**
- **Signed artefacts and verified builds.**
- **Vendor security requirements.**

### 4.7 Physical and cargo security

Cargo crime is a structured business:

- **Strategic theft:** impersonating a carrier and collecting a load.
- **Straight theft:** stealing from a yard or in transit.
- **Pilferage:** small, repeated losses.
- **Internal theft:** staff collusion.

Controls:

- Verified carrier onboarding and identity checks.
- Seals, GPS, and geofencing.
- Yard access control and CCTV.
- Dual control for high-value loads.
- Route and stop monitoring.
- C-TPAT and AEO programmes for customs-trusted operators.

### 4.8 Fraud and payment security

Common frauds:

- **Business email compromise:** a supplier email is spoofed and bank details change.
- **Invoice fraud:** a fake invoice from a real supplier.
- **Duplicate invoicing:** the same invoice paid twice.
- **Vendor master manipulation:** an insider changes a bank account.

Controls:

- Callback verification on bank detail changes, using a known number.
- Segregation of duties between vendor setup and payment.
- Duplicate invoice detection.
- Payment anomaly analytics.
- Bank account validation services.

### 4.9 Incident response and continuity

An incident response plan has phases:

1. **Prepare:** roles, playbooks, contacts, backups.
2. **Detect:** monitoring and alerting.
3. **Contain:** isolate the affected system or shipment.
4. **Eradicate:** remove the threat.
5. **Recover:** restore operations, with a tested plan.
6. **Learn:** post-incident review.

For supply chains, recovery time and recovery point matter as much as containment. A warehouse system down for a day is a delivery failure; a cold chain system down for a day is product loss.

### 4.10 Quantifying risk

Annualised loss expectancy:

\[
\text{ALE} = \text{SLE} \times \text{ARO}
\]

where SLE is single loss expectancy and ARO is annualised rate of occurrence.

A cargo theft event costs €2 million, and it occurs 0.5 times a year:

\[
\text{ALE} = 2{,}000{,}000 \times 0.5 = €1{,}000{,}000
\]

A control costing €250,000 per year reduces the rate to 0.1:

\[
\text{New ALE} = 2{,}000{,}000 \times 0.1 = €200{,}000
\]

\[
\text{Net benefit} = 1{,}000{,}000 - 200{,}000 - 250{,}000 = €550{,}000
\]

The arithmetic is simple; the estimates are not. Use ranges, and state the assumption behind each probability.

### 4.11 Worked example — attack path probability

Two independent paths can divert a payment: a phishing campaign that succeeds 10% of the time, and a supplier-portal credential leak exploited 5% of the time.

\[
P(\text{at least one}) = 1 - (1 - 0.10)(1 - 0.05) = 1 - 0.855 = 0.145
\]

Adding multi-factor authentication cuts the phishing path to 2% and the credential path to 1%:

\[
P = 1 - (0.98 \times 0.99) = 0.0298
\]

The probability falls by about 80%. The control does not need to be perfect; it needs to remove the easy path.

### 4.12 Worked example — supplier questionnaire scoring

A supplier assessment scores five domains on 0–4:

| Domain | Score | Weight |
|---|---:|---:|
| Governance | 3 | 0.15 |
| Access control | 2 | 0.25 |
| Data protection | 3 | 0.20 |
| Incident management | 1 | 0.25 |
| Supply chain | 2 | 0.15 |

\[
\text{Weighted score} = 3(0.15) + 2(0.25) + 3(0.20) + 1(0.25) + 2(0.15) = 2.1
\]

On a 0–4 scale, 2.1 is "conditional". The weak domains are incident management and access control — precisely the domains that matter when something goes wrong. A single composite score hides this; a weighted profile exposes it.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Asset inventory | CMDB / OT register | What must be protected |
| Network flows | Firewall / Zeek | Segmentation and paths |
| Identity and access | IAM / PAM | Privilege and third-party access |
| Vulnerability and patch | Scanner / SBOM | Exposure |
| Supplier assessments | TPRM system | Third-party risk |
| Cargo and shipment | TMS / GPS | Physical movements |
| Seals and access logs | Warehouse / port | Physical integrity |
| Fraud and payments | ERP / bank | Anomalies |
| Incidents | SIEM / case management | Detection and response |

### 5.2 Standards and frameworks

- **ISO/IEC 27001** information security management.
- **NIST Cybersecurity Framework** (Identify, Protect, Detect, Respond, Recover) and **NIST SP 800-53**.
- **IEC 62443** industrial automation and control system security.
- **NIST SP 800-161** cyber supply chain risk management.
- **SLSA** and **SPDX/CycloneDX** for software provenance and SBOMs.
- **TSA** security directives for freight rail and pipeline where applicable.
- **C-TPAT** (US) and **AEO** (EU and others) for trade security.
- **ISPS Code** for port facility security.
- **ISO 28000** security management for the supply chain.

### 5.3 Workflow

1. Inventory assets and data flows.
2. Model threats and attack paths.
3. Assess suppliers and third parties.
4. Design controls and segmentation.
5. Deploy detection and monitoring.
6. Test and exercise response.
7. Measure, review, and improve.

## 6. Industry case studies

### 6.1 Software supply chain compromise

**Context:** A widely used software product's update mechanism was compromised.
**Intervention:** Malicious code distributed through a trusted update channel.
**Evidence:** Public incident reporting and government advisories (grade B).
**Outcome:** Thousands of organisations were exposed; response took months.
**What transferred:** trust in an update channel is a supply chain dependency; provenance and segmentation limit blast radius.
**What did not:** specific victim counts.

### 6.2 Ransomware against a logistics operator

**Context:** A logistics provider suffered ransomware affecting warehouse and transport systems.
**Intervention:** Encryption of operational systems; manual fallback.
**Evidence:** Public incident reporting and industry analysis (grade B/C).
**Outcome:** Shipments were delayed; recovery required rebuilds and manual processes.
**What transferred:** continuity plans must cover manual operation, not only backups.
**What did not:** specific recovery times.

### 6.3 Cargo theft and strategic fraud

**Context:** Organised groups impersonated legitimate carriers to collect loads.
**Intervention:** Fake carrier identities and fraudulent pickup documents.
**Evidence:** Industry and law-enforcement reporting (grade B/C).
**Outcome:** High-value loads were lost; controls were tightened at pickup.
**What transferred:** identity verification at the point of handover is a security control, not a formality.
**What did not:** specific loss values.

### 6.4 A composite failure — the invoice that was almost right

**Context:** An attacker registered a domain one character different from a supplier's and sent a convincing invoice.
**Intervention:** No callback verification; payment was scheduled.
**Evidence:** Composite teaching case.
**Outcome:** A finance clerk noticed the bank country differed, and the payment was stopped.
**What transferred:** one verification step, applied consistently, is worth more than awareness training alone.
**What did not:** the specific amount.

## 7. Labs

### 7.1 Lab 43.1 — Threat model

Draw a data-flow diagram for a supplier portal and an EDI link. Apply STRIDE to each flow and produce a threat register.

**Deliverable:** Diagram and threat register.

### 7.2 Lab 43.2 — Attack tree and risk

Build an attack tree for cargo theft at a distribution centre. Assign ranges to leaf probabilities and compute the ALE for three control packages.

**Deliverable:** Attack tree, risk model, recommendation.

### 7.3 Lab 43.3 — Segmentation design

Design zones and conduits for a warehouse with WMS, conveyor PLCs, and a supplier portal. Identify which flows must be blocked.

**Deliverable:** Zone diagram and firewall rule set.

### 7.4 Lab 43.4 — SBOM review

Take an SBOM for a planning application, identify critical and outdated components, and assess the impact of a hypothetical compromise.

**Deliverable:** SBOM analysis and remediation plan.

### 7.5 Lab 43.5 — Incident response exercise

Run a tabletop exercise for a ransomware incident during peak season. Write the first 24-hour playbook.

**Deliverable:** Playbook and lessons learned.

## 8. Assignment / project

Produce a security architecture and risk case for a supply chain.

**Required components:**

1. Asset and data-flow inventory.
2. Threat model and attack paths.
3. Third-party and supplier assessment.
4. Segmentation and access design.
5. Physical and cargo security.
6. Fraud and payment controls.
7. Incident response and continuity.
8. Risk quantification and control business case.
9. Standards mapping and governance.
10. Risks, assumptions, and limits.

**Length:** 12–16 pages plus models.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Threat model | 20% | Lists threats | STRIDE on flows | Attack paths and controls |
| Architecture | 20% | Names controls | Segmentation design | Zero trust and OT integration |
| Third party | 15% | Sends questionnaire | Weighted assessment | Evidence-based assurance |
| Risk quantification | 15% | Qualitative | ALE with ranges | Control comparison and residual risk |
| Physical and fraud | 15% | Mentions theft | Control set | End-to-end path coverage |
| Response | 10% | Has a plan | Tested playbook | Exercised and measured |
| Communication | 5% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Privacy:** security monitoring must respect data protection law.
- **Critical infrastructure:** ports, rail, and utilities face sector regulation.
- **Disclosure:** incident reporting duties vary by jurisdiction and sector.
- **Contract:** security obligations and liability must be in supplier contracts.
- **Ethics:** monitoring employees and drivers requires proportionality and transparency.
- **Dual use:** security testing must be authorised; unauthorised access is a crime.
- **Insurance:** cyber and cargo policies have exclusions that depend on controls.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Inventory gap | Unknown assets | Shadow IT/OT | Unprotected path | Continuous discovery |
| Questionnaire theatre | All suppliers "green" | Self-attestation | Hidden risk | Evidence-based assurance |
| Flat network | Lateral movement | No segmentation | Full compromise | Zones and conduits |
| Third-party over-access | Standing admin | Convenience | Breach path | Least privilege, JIT |
| Unsigned updates | No provenance | Tampered build | Malware | Signing and SBOM |
| Payment change without callback | Urgent request | Social engineering | Fraud loss | Callback policy |
| Untested continuity | Plan only on paper | Assumed recovery | Prolonged outage | Exercises |
| Risk by heat map only | "High/medium" | No quantification | Poor prioritisation | ALE with ranges |

## 12. Interview and discussion questions

1. What does STRIDE stand for, and how do you apply it? *(Hint: six threat categories on a data-flow diagram.)*
2. Why is OT security different from IT security? *(Hint: availability and safety over confidentiality, legacy protocols.)*
3. What is the Purdue model? *(Hint: layered zones from process to enterprise.)*
4. What is zero trust in a supplier network? *(Hint: verify every request, least privilege.)*
5. How do you assess a third party's security? *(Hint: evidence, not questionnaire self-attestation.)*
6. What is an SBOM, and why does it matter? *(Hint: component inventory for vulnerability response.)*
7. How do you prevent invoice fraud? *(Hint: callback verification and segregation of duties.)*
8. What is ALE, and what are its limits? *(Hint: SLE × ARO; estimates are uncertain.)*
9. How do you stop strategic cargo theft? *(Hint: carrier identity verification at pickup.)*
10. What should the first hour of an incident response look like? *(Hint: contain, preserve evidence, activate playbook.)*
11. How do you measure security programme effectiveness? *(Hint: coverage, time to detect, time to recover.)*
12. Why is a heat map alone a poor risk tool? *(Hint: it hides magnitude and probability.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Threat model | Structured analysis of threats and paths |
| STRIDE | Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, Elevation of privilege |
| Attack tree | Decomposition of an attack into sub-goals |
| OT | Operational technology |
| Purdue model | Layered reference model for industrial control |
| Zone and conduit | IEC 62443 segmentation concept |
| Zero trust | Verify every request, least privilege |
| TPRM | Third-party risk management |
| SBOM | Software bill of materials |
| SLSA | Supply chain levels for software artefacts |
| SLE | Single loss expectancy |
| ARO | Annualised rate of occurrence |
| ALE | Annualised loss expectancy |
| C-TPAT | US Customs-Trade Partnership Against Terrorism |
| AEO | Authorised Economic Operator |
| ISPS | Port facility security code |
| BEC | Business email compromise |

## 14. Sources and further reading

### Standards and regulation

- ISO/IEC 27001. https://www.iso.org/
- NIST Cybersecurity Framework. https://www.nist.gov/cyberframework
- NIST SP 800-53 and SP 800-161. https://csrc.nist.gov/
- IEC 62443 series. https://www.iec.ch/
- SLSA. https://slsa.dev/ ; SPDX. https://spdx.dev/ ; CycloneDX. https://cyclonedx.org/
- ISO 28000 supply chain security management. https://www.iso.org/
- CBP C-TPAT. https://www.cbp.gov/ ; EU AEO. https://taxation-customs.ec.europa.eu/
- IMO ISPS Code. https://www.imo.org/
- TSA security directives. https://www.tsa.gov/

### Reports and official sources

- ENISA threat landscape. https://www.enisa.europa.eu/
- CISA advisories and guidance. https://www.cisa.gov/
- National cyber security centre guidance (UK NCSC and others).
- World Economic Forum, supply chain security reports. https://www.weforum.org/

### Peer-reviewed and professional

- NIST, *Guide to Cyber Supply Chain Risk Management*.
- Ross, R., et al., risk management frameworks.
- Research on cargo theft and organised crime in logistics.
- Software supply chain security literature (provenance, dependency risk).

### Industry

- Transported Asset Protection Association (TAPA). https://tapa.org/
- BSI and ISO 28000 certification bodies.
- Industry incident reports and threat intelligence.

> **Next:** [Module 44](44_mlops_production_analytics.md) turns to the engineering discipline that keeps the analytics themselves running in production.
