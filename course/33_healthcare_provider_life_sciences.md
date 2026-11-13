# Module 33 — Healthcare Provider and Life Sciences Operations

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Healthcare and Logistics tracks
> **Prerequisites:** Modules 07, 10, 13, 29
> **Tools:** Python, DuckDB, OR-Tools, synthetic EHR/ERP extracts
> **Industry question:** "How do you keep a hospital or a trial supplied without holding everything everywhere?"

---

## 1. Why industry cares

Healthcare supply chains differ from commercial chains in one respect that changes every decision: the consequence of a stockout is measured in patient harm, not lost margin. A missing surgical implant, a contaminated batch, or an expired vaccine is a clinical event, a regulatory event, and a reputational event at the same time.

The domain has two distinct supply chains that are often confused:

- **Provider operations:** hospitals, clinics, and surgical centres buying, storing, and issuing supplies to clinical teams.
- **Life sciences operations:** pharmaceutical, biotech, and device manufacturers running clinical trials, commercial distribution, and post-market surveillance.

Both are high-variety, high-regulation, and low-tolerance for error. Both are data-rich but analytically under-served, because the data sits in clinical, ERP, and quality systems that were never designed to talk to each other.

The analytics problem is **item-level traceability and availability under clinical and regulatory constraints**. Cost matters, but it is never the only objective.

## 2. Learning outcomes

By the end of this module, you can:

- Map a provider supply chain from manufacturer to patient and identify the decision at each node.
- Build an item master with UDI and contract data that supports clinical and financial reporting.
- Set par levels and replenishment policies for clinical units and operating rooms.
- Design case-cart and preference-card supply for surgical services.
- Evaluate consignment, vendor-managed inventory, and just-in-time models for provider settings.
- Trace a device or lot from receipt to patient and back.
- Manage shortages and recalls with an explicit allocation and communication rule.
- Model clinical-trial supply, including randomization, blinding, and resupply.
- Measure provider supply performance without creating perverse incentives.
- Assess the regulatory boundary for data use and clinical decision support.

## 3. Mental model

```text
   MANUFACTURER / DISTRIBUTOR
   item, lot, UDI, temperature, contract
              │
              ▼
   PROVIDER DC / CENTRAL STORE
   receive, inspect, store, pick
              │
              ▼
   UNIT PAR / CABINET ──> CLINICAL TEAM
   par level, replenishment   preference card, case cart
              │                     │
              ▼                     ▼
   PATIENT / PROCEDURE <──── IMPLANT / DEVICE
              │
              ▼
   DOCUMENTATION ──> BILLING ──> QUALITY ──> RECALL / REGISTRY
```

The clinical record is the demand signal and the traceability record at the same time. If the item used in a procedure is not documented, the supply chain cannot bill it, trace it, or recall it.

### 3.1 Two clocks, again

| Clock | Provider | Life sciences |
|---|---|---|
| Clinical | Procedure start, patient need | Trial visit, dosing window |
| Product | Expiry, sterility, temperature | Shelf life, stability, kit expiry |
| Regulatory | Recall, UDI, documentation | GxP, serialization, reporting |

A supply decision that meets the clinical clock but violates the product clock is a failure, even if the item arrives on time.

## 4. Core concepts

### 4.1 The provider item master

The item master is the join between clinical, supply, and financial systems. It must carry:

| Field | Why it matters |
|---|---|
| GTIN / UDI-DI | Unique device identification and recall |
| Manufacturer and supplier | Contract and sourcing |
| Unit of measure and conversion | Each, box, case, pallet |
| Contract price and tier | GPO or local contract |
| Storage requirement | Ambient, refrigerated, frozen, controlled |
| Sterility and single-use flag | Handling and reprocessing |
| Shelf life | Expiry and rotation |
| Substitutes and equivalents | Shortage response |
| Charge code | Billing and cost capture |
| Criticality | Clinical consequence of stockout |

Most provider analytics failures start with a broken item master: the same product exists under three descriptions, two units of measure, and no UDI. Fix identity before optimizing inventory.

### 4.2 Par levels and replenishment

A par level is the quantity a clinical unit should hold to cover demand between replenishments plus a buffer. The simple formula:

\[
\text{Par} = \text{average daily use} \times (\text{review period} + \text{lead time}) + \text{safety stock}
\]

Safety stock depends on demand variability, lead-time variability, and the criticality of the item. For a critical item, the target is not a cost-optimal fill rate; it is clinical availability.

**Par cart** and **cabinet** replenishment are the two common models. Cabinets give item-level usage data; par carts give simplicity. The choice depends on item value, variability, and the clinical workflow.

### 4.3 Operating room and case carts

Surgical supply is a kit problem:

- **Preference card:** the surgeon's standard items for a procedure.
- **Case cart:** the picked kit for a scheduled case.
- **Consignment:** high-value implants stored at the hospital but owned by the supplier until use.
- **Loaner sets:** instruments loaned for a specific case, with tight turnaround.

Metrics: case-cart accuracy, first-case on-time start, missing-item rate, instrument turnaround, and implant traceability. A missing item can delay a case by 30–60 minutes; the cost is theatre idle time and staff overtime, not the item's price.

### 4.4 Consignment and vendor-managed inventory

| Model | Inventory owner | Strength | Risk |
|---|---|---|---|
| Owned stock | Provider | Control | Capital and expiry |
| Consignment | Supplier | Low provider capital | Counting, expiry, recall |
| Vendor-managed | Supplier | Fewer stockouts | Dependency and data sharing |
| Just-in-time | Provider / distributor | Low stock | Fragility under disruption |
| Direct-to-case | Supplier | No hospital stock | Lead time and case changes |

Consignment is not free inventory. The provider still has to count it, trace it, and recall it. The contract must define ownership, insurance, expiry, and data obligations.

### 4.5 Traceability and UDI

A device or lot must be traceable from receipt to patient and back. The chain:

```text
PO → receipt (lot, UDI) → storage → issue → procedure → patient record
                                                   │
                                                   └→ recall / registry / billing
```

**UDI** (unique device identification) provides a device identifier and production identifier. The provider must capture it at the point of use, not only at receipt. Barcode scanning at the point of care is the practical control; manual transcription is the failure mode.

### 4.6 Shortages and recalls

**Shortage management** is an allocation problem with a clinical objective:

| Step | Action |
|---|---|
| Detect | Supplier notice, backorder, demand spike |
| Assess | Clinical criticality, alternatives, inventory cover |
| Allocate | Rule by clinical priority, not by who calls loudest |
| Communicate | Clinicians, pharmacy, procurement, leadership |
| Substitute | Approved equivalent, with clinical sign-off |
| Recover | Root cause, contract, dual sourcing |

**Recall management** is a traceability stress test. The questions are: which lots, which patients, which inventory locations, and how fast can we prove it? A recall drill is the only way to know.

### 4.7 Clinical trial supply

A trial supply chain must deliver the right kit to the right patient at the right visit, without unblinding:

- **Randomization** assigns treatment; supply must match the assignment.
- **Blinding** means the depot and site may not know the treatment.
- **Kit allocation** uses randomization numbers and kit lists.
- **Resupply** is triggered by patient visits and dropout.
- **Expiry and stability** constrain kit shelf life.
- **Temperature excursions** may require quarantine and replacement.

The analytics problem is demand forecasting under a patient recruitment curve with high uncertainty, plus a supply plan that protects against stockout at the site without over-supplying expensive kits.

### 4.8 Provider performance metrics

| Metric | Definition | Caution |
|---|---|---|
| Fill rate | Requests met from stock | Does not capture clinical impact |
| Stockout rate | Items unavailable when needed | Needs criticality weighting |
| Inventory turns | Cost of goods / average inventory | Can reward understocking |
| Case-cart accuracy | Carts complete and correct | Best leading indicator for OR |
| Expiry write-off | Value expired | Small in value, large in signal |
| Recall completion time | Time to identify and remove affected lots | Regulatory importance |
| Contract compliance | Spend through contracted items | May conflict with clinical preference |
| Supply cost per case | Supply spend / procedure | Case-mix confounds |

Do not manage a hospital supply chain on turns alone. A turns target can be met by starving clinical units, and the cost appears in delays rather than in the supply budget.

### 4.9 Worked example — par level for a clinical unit

A unit uses an average of 40 units per day with a standard deviation of 8. Replenishment occurs every two days, and the supplier lead time is three days. The review plus lead time is five days.

\[
\text{Cycle demand} = 40 \times 5 = 200
\]

For a 95% service level, safety stock is approximately:

\[
1.645 \times 8 \times \sqrt{5} \approx 29
\]

So the par level is about 229 units. If the item is clinically critical, the target may be 99%, raising the par to roughly 200 + 2.33 × 8 × 2.24 ≈ 242 units.

The decision is not the formula. It is the service target. A 95% target on a critical item means a 1-in-20 chance of a stockout during the review period, which may be unacceptable in a clinical setting.

### 4.10 Worked example — case-cart accuracy

A surgical suite runs 20 cases per day, 250 days per year. The missing-item rate is 4%, and a missing item delays the case by an average of 30 minutes. Theatre idle cost is €1,200 per hour.

\[
20 \times 250 \times 0.04 = 200 \text{ delayed cases per year}
\]

\[
200 \times 0.5 \text{ h} \times €1{,}200 = €120{,}000 \text{ per year}
\]

A case-cart accuracy programme that halves the missing-item rate saves about €60,000 in theatre time, before considering staff overtime and patient experience. The intervention is usually process and data, not more inventory.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Grain | Use |
|---|---|---|
| Item master | Item | Identity, contract, storage, charge code |
| Purchase orders | PO line | Procurement and price |
| Receipts | Receipt line / lot | Traceability and expiry |
| Inventory | Item × location × lot | Par and expiry |
| Cabinet / par usage | Item × unit × time | Demand |
| OR schedule | Case | Case-cart demand |
| Preference cards | Procedure / surgeon | Kit definition |
| EHR / procedure record | Patient / case | Implant and device use |
| Recall notices | Lot / device | Trace and remove |
| Temperature logs | Logger × time | Cold chain |

### 5.2 Standards and regulation

- **FDA UDI** and GUDID for device identification. https://www.fda.gov/medical-devices/unique-device-identification-system-udi-system
- **FDA DSCSA** for pharmaceutical serialization and tracing; verify current requirements.
- **EU MDR** and **IVDR** for device regulation, including UDI and traceability.
- **EU GDP** for pharmaceutical distribution.
- **GS1** GTIN, GLN, SSCC, and EPCIS.
- **HIPAA** (US) and **GDPR** (EU) for patient data; use de-identified or aggregate data where possible.
- **Joint Commission** and equivalent accreditation standards for medication and supply management.
- **ICH GCP** for clinical trial supply and investigational product handling.

### 5.3 Workflow

1. Maintain the item master and UDI.
2. Contract and procure.
3. Receive with lot and expiry capture.
4. Store under the required condition.
5. Replenish par carts, cabinets, and case carts.
6. Capture use at the point of care.
7. Reconcile billing and inventory.
8. Manage shortages, substitutes, and recalls.
9. Review performance by clinical service, not only by purchasing category.

## 6. Industry case studies

### 6.1 NHS Supply Chain — consolidation and clinical engagement

**Context:** A national health system procuring and distributing supplies to many trusts.
**Intervention:** Category management, national contracts, and a distribution network, with clinical engagement on product choice.
**Evidence:** Public reports and programme disclosures (grade B/C); performance figures vary by period.
**Outcome:** Consolidation improved price leverage but required clinical acceptance and local logistics capability.
**What transferred:** in healthcare, procurement savings are real only when clinical workflow and product preference are respected.
**What did not:** the specific governance and contract structure, which is jurisdiction-specific.

### 6.2 FDA drug shortages — allocation under scarcity

**Context:** Recurring shortages of injectable drugs and other critical products.
**Intervention:** Manufacturer, regulator, and provider responses, including allocation, importation, and substitution.
**Evidence:** Regulatory and public health reporting (grade B).
**Outcome:** Shortages exposed the fragility of single-source and low-margin generic supply, and the difficulty of allocating fairly.
**What transferred:** shortage response needs a pre-agreed clinical allocation rule and a communication channel.
**What did not:** the specific products and regulatory actions, which change over time.

### 6.3 UDI adoption — the traceability gap

**Context:** Regulators required unique device identification to improve recall and safety.
**Intervention:** Labeling and database requirements, plus provider scanning at receipt and point of use.
**Evidence:** Regulatory guidance and implementation reports (grade B/C).
**Outcome:** Labeling improved, but point-of-use capture remains uneven, so recall traceability is still incomplete in many settings.
**What transferred:** traceability is only as good as the weakest capture point.
**What did not:** the pace of adoption and the specific device categories.

### 6.4 A composite failure — the trial kit that expired

**Context:** A clinical trial site held kits for a visit window that shifted because recruitment slowed.
**Intervention:** None; resupply was driven by the original recruitment plan.
**Evidence:** Composite teaching case.
**Outcome:** Kits expired before use, forcing replacement and delaying a cohort.
**What transferred:** trial supply must be driven by the actual recruitment curve, not the plan, and expiry must be part of the resupply trigger.
**What did not:** the specific cost and timeline.

## 7. Labs

### Lab 33.1 — Item master and UDI audit

Given a synthetic item master, identify duplicates, unit-of-measure conflicts, missing UDI, and contract mismatches. Propose a remediation order.

**Deliverable:** Audit, remediation plan, and identity standard.

### Lab 33.2 — Par-level optimization

Build a par-level model for 50 items with demand variability, lead time, and criticality. Compare cost and stockout risk with the current par.

**Deliverable:** Par model, criticality table, and recommendation.

### Lab 33.3 — Case-cart accuracy

Simulate a surgical schedule with preference cards, pick errors, and missing items. Measure first-case delays and the effect of a checking step.

**Deliverable:** Simulation, delay estimate, and process recommendation.

### Lab 33.4 — Shortage allocation

Given a shortage of a critical drug across three hospitals and two patient groups, apply three allocation rules and compare clinical and equity outcomes.

**Deliverable:** Allocation analysis and policy memo.

### Lab 33.5 — Recall trace

Trace a synthetic recalled lot from supplier to patients. Measure time, completeness, and the inventory still unaccounted for.

**Deliverable:** Trace report and corrective action.

## 8. Assignment / project

Design a provider supply chain for one clinical service line.

**Required components:**

1. Item master and UDI standard.
2. Demand and par-level model.
3. OR or procedure supply design.
4. Consignment and contract strategy.
5. Traceability and recall process.
6. Shortage and substitution plan.
7. Performance metrics and governance.
8. Data privacy and clinical-risk assessment.
9. Implementation risks and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Item identity | 20% | Describes items | Item master with UDI | Contract, clinical, and financial linkage |
| Inventory policy | 20% | Fixed par | Variability-based par | Criticality and clinical target |
| Clinical workflow | 20% | Generic supply | OR and unit workflow | Preference cards and case carts |
| Traceability | 15% | Lot number | Receipt-to-use trace | Tested recall and patient linkage |
| Shortage and recall | 15% | Describes process | Allocation rule | Clinical and equity trade-offs |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Patient safety:** supply decisions can affect clinical outcomes; clinical governance must own the risk.
- **Traceability:** UDI, DSCSA, MDR, and GDP impose records and timelines; verify the current requirements.
- **Privacy:** patient-level data is sensitive; use the minimum necessary and secure it.
- **Equity:** shortage allocation can disadvantage vulnerable groups; the rule must be explicit and reviewable.
- **Conflict of interest:** clinician preference, vendor relationships, and procurement incentives must be disclosed and managed.
- **Anti-kickback and transparency:** device and pharmaceutical commercial interactions are regulated in many jurisdictions.
- **Environment:** single-use devices, packaging, and sterilization have significant footprints; disposal must follow medical-waste rules.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Item master chaos | Duplicate items | No identity standard | Wrong product, wrong price | UDI-led cleansing |
| Turns-only management | Clinical complaints | Understocking | Delays and harm | Criticality-weighted metrics |
| Par set by habit | Annual review only | Demand changed | Stockouts or expiry | Data-driven par |
| Case cart unchecked | Missing-item rate | Pick errors | OR delays | Checklist and scan |
| Consignment untracked | Count discrepancies | Ownership unclear | Expiry and recall gaps | Contract and cycle count |
| Recall untested | Trace takes days | Records incomplete | Wider harm | Drills |
| Shortage by escalation | Loudest customer wins | No allocation rule | Inequity | Pre-agreed policy |
| Trial supply on plan | Kits expire | Recruitment slower | Cohort delay | Actual recruitment trigger |
| Privacy overreach | Patient data in analytics | Unnecessary identifiers | Legal risk | De-identification |
| Vendor lock-in | Single source | Contract dependency | Shortage exposure | Dual sourcing and exit terms |

## 12. Interview and discussion questions

1. What is the difference between a provider supply chain and a manufacturer supply chain? *(Hint: clinical consequence and item-level traceability.)*
2. Why is the item master the first analytics problem? *(Hint: identity, unit of measure, contract, UDI.)*
3. How do you set a par level for a critical item? *(Hint: demand, lead time, review period, clinical target.)*
4. What is a preference card, and why does it matter? *(Hint: it defines the case cart and the surgeon's standard.)*
5. What is consignment inventory, and who owns the risk? *(Hint: supplier owns until use; provider owns counting and traceability.)*
6. How do you trace an implant to a patient? *(Hint: UDI at point of use, lot, procedure record.)*
7. How would you allocate a drug shortage across hospitals? *(Hint: clinical priority, equity, transparency.)*
8. Why can inventory turns be a bad target in healthcare? *(Hint: it can reward understocking and harm clinical service.)*
9. What makes clinical trial supply different from commercial supply? *(Hint: randomization, blinding, kit allocation, recruitment uncertainty.)*
10. What should happen when a cold-chain excursion affects a vaccine batch? *(Hint: quarantine, assess, consult, release or destroy.)*
11. Which metric would you use to monitor OR supply performance? *(Hint: case-cart accuracy and first-case on-time.)*
12. How do you balance procurement savings with clinical preference? *(Hint: clinical engagement, value analysis, contract compliance.)*

## 13. Key terms

| Term | Definition |
|---|---|
| UDI | Unique device identification |
| GUDID | FDA Global Unique Device Identification Database |
| Par level | Target quantity held at a clinical unit |
| Par cart | Pre-filled cart replenished to a fixed level |
| Case cart | Picked supplies for a scheduled surgical case |
| Preference card | Surgeon's standard item list for a procedure |
| Consignment | Supplier-owned stock held at the provider |
| Vendor-managed inventory | Supplier manages stock levels at the customer |
| SPD | Sterile processing department |
| GPO | Group purchasing organization |
| Formulary | Approved list of medicines for use |
| DSCSA | US Drug Supply Chain Security Act |
| GDP | Good Distribution Practice |
| MDR | EU Medical Device Regulation |
| IVDR | EU In Vitro Diagnostic Regulation |
| Clinical trial supply | Kit production, randomization, and resupply for a trial |
| Blinding | Keeping treatment assignment hidden |
| Recall | Removal of affected lots or devices from use |
| Fill rate | Share of requests met from stock |
| Case-cart accuracy | Share of carts complete and correct |

## 14. Sources and further reading

### Standards and regulation

- FDA, Unique Device Identification System. https://www.fda.gov/medical-devices/unique-device-identification-system-udi-system
- FDA, Drug Supply Chain Security Act. https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa
- EU Medical Device Regulation (EU) 2017/745. https://eur-lex.europa.eu/
- EU In Vitro Diagnostic Regulation (EU) 2017/746. https://eur-lex.europa.eu/
- European Medicines Agency, Good Distribution Practice guidelines.
- ICH, Good Clinical Practice E6.
- GS1, EPCIS 2.0. https://www.gs1.org/standards/epcis

### Peer-reviewed and professional

- Chopra, S., Meindl, P., *Supply Chain Management*.
- Burns, L. R., et al., *Healthcare Supply Chain Management*.
- Jarrett, P. G., "Healthcare supply chain management," research on provider operations.
- Souza, G. C., et al., research on healthcare operations and drug shortages.

### Industry

- NHS Supply Chain public reports. https://www.supplychain.nhs.uk/
- FDA drug shortage database. https://www.accessdata.fda.gov/scripts/drugshortages/
- Joint Commission standards and accreditation guidance. https://www.jointcommission.org/
- GS1 Healthcare guidance. https://www.gs1.org/industries/healthcare

> **Next:** [Module 34](34_energy_utilities_capital_projects.md) moves from clinical supply to asset-intensive operations, where a spare part can stop a grid, a plant, or a construction schedule.
