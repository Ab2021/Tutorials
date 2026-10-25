# Module 25 — Trial Operations, HTA/Market Access, and Global Regulation

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 28 hours (10 lecture, 4 case clinic, 14 lab)
> **Track fit:** Biopharma & Translational core elective; Health Systems/MedTech elective
> **Prerequisites:** Modules 05, 13, 14, 17
> **Tools:** Protocol templates, CDISC concepts, ClinicalTrials.gov, HTA reports, payer dossiers, regulatory databases
> **Industry question:** "Can this product be executed, reimbursed, and approved in the markets that matter?"

---

## 1. Why this module exists

A product can pass a clinical trial and still fail because:

- Sites cannot enroll the protocol.
- The data are not submission-ready.
- The payer will not cover it.
- The price is unacceptable.
- The regulatory strategy does not fit the jurisdiction.
- The evidence does not match the HTA question.

Data scientists often optimize the model and ignore the operating system around it. This module teaches the execution, reimbursement, and global regulatory context that determines whether the work reaches patients.

## 2. Learning outcomes

By the end of this module, you can:

- Map a protocol to sites, systems, data flows, and submission deliverables.
- Design a risk-based monitoring and data-quality plan.
- Evaluate a decentralized trial design for feasibility and integrity.
- Build a budget-impact and cost-effectiveness model.
- Write a payer value dossier outline.
- Compare regulatory pathways across major markets.
- Identify evidence gaps for each market.
- Design a global regulatory and market-access roadmap.
- Communicate trade-offs among evidence, price, and access.

## 3. Part A — Trial operations

### 3.1 Protocol to execution

```
Protocol → Feasibility → Site selection → Activation → Recruitment → Consent
→ Randomization/treatment → Data capture → Monitoring → Cleaning → Database lock
→ Analysis → Submission → Inspection
```

| Stage | Key question | Data artifact |
|---|---|---|
| Feasibility | How many eligible patients, where? | Site questionnaire, EHR/claims counts |
| Site selection | Which sites can deliver? | Historical enrollment, startup time |
| Activation | When can the site start? | Contracts, IRB, training, drug supply |
| Recruitment | Are patients willing and eligible? | Screening log, screen-failure rate |
| Consent | Is consent valid and documented? | eConsent, audit trail |
| Treatment | Is the protocol followed? | eCRF, drug accountability |
| Monitoring | Where are the risks? | SDV, KRIs, queries |
| Cleaning | Are data correct and complete? | Query log, medical coding |
| Lock | Is the database final? | DBL checklist, audit trail |
| Submission | Is the evidence traceable? | SDTM/ADaM, define.xml |

### 3.2 Systems

| System | Function | Data-science relevance |
|---|---|---|
| EDC | Electronic data capture | Query rate, missingness, audit trail |
| ePRO | Patient-reported outcomes | Adherence, timing, device validation |
| eConsent | Electronic consent | Version control, comprehension |
| Randomization | Treatment allocation | Allocation concealment, stratification |
| Drug supply | Investigational product | Accountability, temperature |
| CTMS | Site and trial management | Activation, enrollment, milestones |
| Safety database | Adverse events | Coding, signal detection |
| Central lab | Biomarkers | Sample tracking, reference ranges |
| Wearable/DCT platform | Remote data | Data quality, missingness, privacy |

### 3.3 Risk-based quality management

- **Critical to quality (CtQ):** what must go right for patient safety and reliable results?
- **Key risk indicators (KRIs):** enrollment lag, query rate, protocol deviations, missing visits, AE reporting delay.
- **Risk-based monitoring:** targeted SDV plus central analytics, not 100% SDV by default.
- **Quality tolerance limits:** pre-defined thresholds that trigger action.

**Data-science role:** build KRI dashboards, anomaly detection, site-level risk models, and query prediction. Validate that a “high-risk” flag leads to a useful action.

### 3.4 Decentralized trials

| Element | Benefit | Risk |
|---|---|---|
| eConsent | Remote enrollment | Comprehension, identity, version control |
| Telehealth visits | Access, convenience | Privacy, licensure, quality |
| Home nursing | Hard-to-reach patients | Training, consistency |
| Wearables/ePRO | Frequent data | Adherence, device bias, missingness |
| Direct-to-patient shipment | Access | Chain of custody, temperature |
| Remote monitoring | Safety | Escalation, connectivity |

**Regulatory:** FDA/EMA accept decentralized elements under existing frameworks, but the protocol must justify them, ensure data integrity, and protect participants.

### 3.5 Trial metrics

| Metric | Why it matters |
|---|---|
| Site activation time | Determines start-up delay |
| Enrollment rate | Drives timeline and cost |
| Screen-failure rate | Eligibility feasibility |
| Query rate and aging | Data quality and workload |
| Protocol deviation rate | Execution quality |
| SDV coverage | Monitoring intensity |
| Database-lock time | Submission timeline |
| Cost per patient | Budget and site selection |

### 3.6 AI in trial operations

- Feasibility and recruitment forecasting.
- Protocol optimization and eligibility simplification.
- Auto-coding and data cleaning.
- Risk-based monitoring and anomaly detection.
- Safety signal detection.
- Drafting documents and queries.

**Rule:** every AI output that affects patient safety, eligibility, or regulatory data needs validation, audit, and human review.

## 4. Part B — HTA and market access

### 4.1 The payer question

Regulators ask: “Is it safe and effective?”
Payers ask: “Is it worth the money compared with what we already pay for?”

### 4.2 HTA bodies

| Body | Market | Focus |
|---|---|---|
| NICE | England/Wales | Clinical and cost-effectiveness |
| ICER | US | Cost-effectiveness, budget impact |
| CADTH | Canada | Clinical and economic evidence |
| PBAC | Australia | Cost-effectiveness, price |
| EUnetHTA | EU | Joint clinical assessments |
| IQWiG | Germany | Benefit assessment |
| HAS | France | Clinical benefit, price |
| AIFA | Italy | Price and reimbursement |

### 4.3 Economic evaluation

| Concept | Definition |
|---|---|
| Perspective | Payer, health system, societal |
| Comparator | Standard of care |
| Time horizon | Long enough to capture costs and outcomes |
| Discount rate | Future costs/benefits |
| QALY | Quality-adjusted life year |
| ICER | Incremental cost per QALY |
| Budget impact | Net annual cost to payer |
| Uncertainty | PSA, scenario, one-way sensitivity |

**ICER:**

\[
\text{ICER} = \frac{C_1 - C_0}{E_1 - E_0}
\]

**Budget impact:**

\[
\text{BI} = \text{eligible population} \times \text{uptake} \times (\text{cost}_{\text{new}} - \text{cost}_{\text{old}})
\]

### 4.4 Value dossier

Required sections:

1. Disease burden and unmet need.
2. Product and comparator.
3. Clinical evidence.
4. Cost-effectiveness.
5. Budget impact.
6. Uncertainty.
7. Patient-reported outcomes.
8. Equity and access.
9. Implementation and monitoring.

### 4.5 Reimbursement pathways

- New CPT/HCPCS code.
- CMS coverage (NCD/LCD).
- NTAP for devices.
- DRG payment.
- Value-based agreements.
- Coverage with evidence development.
- Prior authorization.

**Rule:** FDA authorization does not create payment. Reimbursement is a separate evidence and policy process.

## 5. Part C — Global regulation

### 5.1 Major markets

| Market | Regulator | Device route | Drug route | Notes |
|---|---|---|---|---|
| US | FDA | 510(k)/De Novo/PMA | IND/NDA/BLA | CDS, SaMD, PCCP |
| EU | Commission/notified bodies | MDR | EMA/centralized | AI Act in parallel |
| UK | MHRA | UKCA/MDR transition | MHRA | Post-Brexit divergence |
| Canada | Health Canada | Medical Device Regulations | CTA | RELIANCE pathway |
| Australia | TGA | Medical Devices Regulations | CTN | Comparable overseas regulator reliance |
| Japan | PMDA/MHLW | Pharmaceutical and Medical Device Act | PMDA | Local clinical data may be needed |
| China | NMPA | Device registration | NMPA | Local testing/clinical |
| Brazil | ANVISA | RDC | ANVISA | Local representative |
| India | CDSCO | Medical Device Rules | CDSCO | Local testing |

### 5.2 Common principles

- Intended use and claims.
- Quality management.
- Risk management.
- Clinical evidence.
- Labeling and instructions for use.
- Post-market surveillance.
- Cybersecurity.
- Unique device identification.

### 5.3 Reliance and convergence

- IMDRF harmonization.
- MDSAP audits.
- Reliance on trusted regulators.
- Mutual recognition where available.
- Still verify local requirements.

## 6. Labs

### Lab 26.1 — Protocol feasibility

**Objective:** Translate a protocol into computable criteria and estimate eligible patients.

**Steps:** define eligibility, query claims/EHR, estimate screen failure, map sites, compare to ClinicalTrials.gov, write a feasibility memo.

**Deliverable:** feasibility report.

**Acceptance criteria:** criteria transparent; assumptions stated; site variation.

### Lab 26.2 — Risk-based monitoring plan

**Objective:** Design KRIs and a monitoring plan.

**Steps:** define CtQ, KRIs, thresholds, actions, owners; simulate site risk; write a monitoring plan.

**Deliverable:** RBQM plan.

**Acceptance criteria:** risk-based; actionable; audit trail.

### Lab 26.3 — Decentralized trial assessment

**Objective:** Evaluate a DCT design.

**Steps:** map elements, data flows, consent, devices, safety escalation; identify regulatory and operational risks; propose mitigations.

**Deliverable:** DCT assessment.

**Acceptance criteria:** patient safety; data integrity; local law.

### Lab 26.4 — Budget impact

**Objective:** Build a 3-year budget-impact model.

**Steps:** eligible population, uptake, cost, offsets, sensitivity, payer perspective.

**Deliverable:** model and memo.

**Acceptance criteria:** assumptions; uncertainty; net budget.

### Lab 26.5 — Cost-effectiveness

**Objective:** Calculate an ICER and PSA.

**Steps:** comparators, costs, QALYs, discounting, probabilistic sensitivity, cost-effectiveness acceptability curve.

**Deliverable:** model and report.

**Acceptance criteria:** perspective; time horizon; uncertainty.

### Lab 26.6 — Global regulatory map

**Objective:** Compare three markets for one product.

**Steps:** intended use, classification, evidence, local requirements, timeline, cost, post-market.

**Deliverable:** regulatory roadmap.

**Acceptance criteria:** sourced; jurisdiction-specific; gaps.

## 7. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Trial operations | 20% | Protocol only | Feasibility and monitoring | Risk-based, audit-ready |
| HTA | 25% | No model | ICER and budget impact | Uncertainty and payer perspective |
| Global regulation | 25% | One market | Three markets compared | Reliance and evidence gaps |
| Communication | 15% | Technical | Clear | Payer/regulator ready |
| Ethics/equity | 15% | Ignores | Identifies | Access and equity plan |

## 8. Failure modes

- Feasibility based on codes alone.
- Ignoring screen failure.
- 100% SDV as a substitute for risk management.
- Budget impact without uptake assumptions.
- ICER without perspective or time horizon.
- Assuming FDA approval means reimbursement.
- Assuming EU approval works in the UK.
- Ignoring local clinical evidence requirements.
- No post-market plan.

## 9. Interview questions

1. **How do you estimate trial feasibility?**
   Hint: computable eligibility, screen failure, sites, historical performance.
2. **What is risk-based monitoring?**
   Hint: CtQ, KRIs, targeted SDV, central analytics.
3. **What is an ICER?**
   Hint: incremental cost per QALY.
4. **Why can a cost-effective product fail?**
   Hint: budget impact, adoption, price, procurement.
5. **What evidence does a payer need?**
   Hint: clinical utility, cost-effectiveness, budget impact, equity.
6. **How do regulatory pathways differ across markets?**
   Hint: classification, local data, reliance, timelines.
7. **What is a decentralized trial?**
   Hint: remote consent, visits, data capture, safety.
8. **What is coverage with evidence development?**
   Hint: coverage conditional on more evidence.
9. **How do you build a global regulatory roadmap?**
   Hint: claims, classification, evidence, local requirements, sequencing.
10. **What would make you stop a program?**
    Hint: no feasible enrollment, no reimbursement, unacceptable evidence gap.

## 10. Key terms

| Term | Definition |
|---|---|
| EDC | Electronic data capture |
| ePRO | Electronic patient-reported outcome |
| eConsent | Electronic informed consent |
| CTMS | Clinical trial management system |
| RBQM | Risk-based quality management |
| KRI | Key risk indicator |
| SDV | Source data verification |
| DBL | Database lock |
| DCT | Decentralized trial |
| HTA | Health technology assessment |
| QALY | Quality-adjusted life year |
| ICER | Incremental cost-effectiveness ratio |
| PSA | Probabilistic sensitivity analysis |
| NCD/LCD | National/local coverage determination |
| NTAP | New technology add-on payment |
| UKCA | UK Conformity Assessed |
| PMDA | Japan Pharmaceuticals and Medical Devices Agency |
| NMPA | China National Medical Products Administration |
| ANVISA | Brazil health regulator |
| CDSCO | India Central Drugs Standard Control Organisation |
| IMDRF | International Medical Device Regulators Forum |
| MDSAP | Medical Device Single Audit Program |

## 11. Sources

- FDA, *Real-World Evidence*. https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence
- FDA, *CDISC*. https://www.cdisc.org/
- ICH, *E6 GCP*, *E9(R1)*. https://www.ich.org/
- ClinicalTrials.gov. https://clinicaltrials.gov/
- NICE. https://www.nice.org.uk/
- ICER. https://icer.org/
- CADTH. https://www.cadth.ca/
- EUnetHTA. https://www.eunethta.eu/
- IQWiG. https://www.iqwig.de/
- HAS. https://www.has-sante.fr/
- MHRA. https://www.gov.uk/government/organisations/medicines-and-healthcare-products-regulatory-agency
- Health Canada. https://www.canada.ca/en/health-canada.html
- TGA. https://www.tga.gov.au/
- PMDA. https://www.pmda.go.jp/english/
- NMPA. https://www.nmpa.gov.cn/
- ANVISA. https://www.gov.br/anvisa/
- CDSCO. https://cdsco.gov.in/
- IMDRF. https://www.imdrf.org/
- CMS, *Coverage with Evidence Development*. https://www.cms.gov/medicare/coverage/coverage-with-evidence-development

> **Previous:** [`24_hardware_robotics_labs.md`](24_hardware_robotics_labs.md)
> **Next:** [`26_cloud_streaming_production_labs.md`](26_cloud_streaming_production_labs.md)
