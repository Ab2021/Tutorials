# Module 37 — Circular Economy and Reverse Logistics Networks

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Retail and Manufacturing tracks
> **Prerequisites:** Modules 10, 11, 12, 21, 23
> **Tools:** Python, OR-Tools, synthetic returns and grading data
> **Industry question:** "The product comes back — is it waste, a part, or a second sale?"

---

## 1. Why industry cares

Every forward supply chain produces a reverse one: returns, repairs, recalls, cores, packaging, and end-of-life products. For most companies the reverse chain is unplanned, measured poorly, and treated as a cost centre. Yet it is where value is recovered or destroyed.

Three pressures are making it strategic:

- **Regulation.** Extended producer responsibility, right-to-repair, battery and e-waste rules, and deposit schemes make take-back mandatory in many markets.
- **Economics.** Recovering a product can be cheaper than making a new one, especially where materials are scarce or expensive.
- **Customers.** Easy returns and trade-in programmes are part of the offer, and the cost of a bad returns experience is lost loyalty.

The analytics problem is **disposition and network design under uncertain, heterogeneous, low-value flows**.

## 2. Learning outcomes

By the end of this module, you can:

- Map reverse flows and distinguish returns, recalls, cores, packaging, and end-of-life.
- Build a disposition model that chooses between resale, refurbish, repair, remanufacture, harvest, recycle, and destroy.
- Estimate recovery yield, grading accuracy, and the value of information.
- Design a reverse network with collection points, consolidation, and processing locations.
- Measure the unit economics of a return and identify the break-even recovery value.
- Detect return fraud and abuse with data.
- Model take-back regulation and producer responsibility obligations.
- Evaluate packaging reuse and deposit-return systems.
- Assess second-life and recycling options for batteries and electronics.
- Build circular metrics that do not overstate recovery.

## 3. Mental model

```text
   CUSTOMER ──> COLLECTION ──> CONSOLIDATION ──> GRADING ──> DISPOSITION
      │             │               │                │             │
   return        drop-off        hub / 3PL        test and       resale
   recall        pickup         sort             assess         refurbish
   trade-in      mail           cross-dock                      repair
   end-of-life                                                  harvest
                                                                recycle
                                                                destroy
                                                                   │
                                                          ┌────────┴────────┐
                                                          ▼                 ▼
                                                    FORWARD CHAIN      MARKET
                                                    (parts, material)  (second user)
```

The reverse chain has one feature the forward chain does not: **the condition of the unit is unknown until it is inspected**. That uncertainty is the core analytics problem.

### 3.1 The reverse clock

| Stage | Uncertainty | Cost driver |
|---|---|---|
| Collection | Will the customer return it? | Logistics and convenience |
| Consolidation | How much, from where? | Transport and handling |
| Grading | What condition is it in? | Labour and test equipment |
| Disposition | What is it worth? | Market, timing, and yield |

Delay in the reverse chain destroys value: a returned phone loses value every week it sits in a warehouse.

## 4. Core concepts

### 4.1 Reverse flow types

| Flow | Trigger | Value | Regulation |
|---|---|---|---|
| Commercial return | Customer change of mind | Resale or refurbish | Consumer law |
| Warranty return | Defect | Repair or replace | Warranty law |
| Recall | Safety | Quarantine and destroy | Product safety law |
| Core return | Trade-in or exchange | Remanufacture | Contract |
| Packaging | Delivery | Reuse or recycle | EPR and deposit rules |
| End-of-life | Disposal | Recycle or recover | WEEE, battery rules |
| Loaner / rental | Contract end | Refurbish | Contract |

Each flow has a different disposition logic. A single "returns process" is a sign of an immature operation.

### 4.2 The disposition hierarchy

From most to least value:

```text
resell as new ──> resell as used ──> refurbish ──> repair ──> remanufacture
      ──> harvest parts ──> recycle material ──> energy recovery ──> dispose
```

The choice depends on condition, demand, cost, warranty, and brand risk. A returned premium product sold as new can damage brand trust if the condition is not verified.

### 4.3 Grading and yield

Grading assigns a condition class. The quality of grading determines the value captured:

\[
\text{Recovered value} = \sum_g \text{units}_g \times \text{price}_g \times \text{yield}_g
\]

Grading error has two costs:

- **Under-grading:** good units sold cheap.
- **Over-grading:** bad units sold as good, causing a second return and brand damage.

A confusion matrix of predicted versus actual grade is the right diagnostic. The business cost of each error type is asymmetric, so accuracy alone is not enough.

### 4.4 Reverse network design

Reverse networks differ from forward networks:

- Flows are **many-to-few**: thousands of customers, few processing sites.
- Volumes are **small and uncertain** per location.
- Transport is often **less than truckload** and expensive per unit.
- Collection can be **outsourced** to retail, lockers, or postal networks.

Design choices: collection density, consolidation hubs, processing location, and whether to centralise grading.

### 4.5 Return unit economics

\[
\text{Net recovery} = \text{resale value} - \text{transport} - \text{handling} - \text{grading} - \text{refurb cost} - \text{reverse logistics overhead}
\]

A return is profitable only if net recovery exceeds the cost of the forward margin lost. Many "free returns" policies are cross-subsidised by the retained sales they enable.

### 4.6 Returns fraud and abuse

Common patterns:

- Wardrobing (use and return).
- Empty-box or swapped-item returns.
- Serial returners.
- Receipt fraud.
- Collusion at a collection point.

Detection uses behaviour features: return rate, timing, product category, value, payment method, and address clustering. False positives harm legitimate customers, so the model needs a review path.

### 4.7 Remanufacturing and closed-loop

Remanufacturing restores a product to like-new condition with a warranty. It requires:

- Core supply (the returned unit).
- Disassembly and inspection.
- Component reuse and replacement.
- Quality assurance to original specification.
- A market for the remanufactured product.

The analytics problem is **core availability and yield**. You cannot remanufacture more than you collect, and you cannot plan production without a yield estimate.

### 4.8 Packaging reuse and deposit return

Reusable packaging and deposit schemes change the network:

- Collection and washing infrastructure.
- Return rate and loss rate.
- Per-trip cost versus single-use.
- Hygiene and food-safety constraints.

The break-even number of trips is the key metric:

\[
\text{Break-even trips} = \frac{\text{reusable cost} - \text{residual}}{\text{single-use cost} - \text{per-trip cost}}
\]

### 4.9 Batteries and electronics

Batteries and electronics have specific reverse chains:

- **Collection** through retail, municipal, or producer schemes.
- **Transport** under dangerous-goods rules for damaged or defective batteries.
- **Testing** for state of health.
- **Second life** in stationary storage, where viable.
- **Recycling** for critical materials.

State-of-health estimation is the key analytics problem for second life. A battery that is unsuitable for a vehicle may be valuable for storage, but only if its health and safety can be verified.

### 4.10 Circular metrics

| Metric | Definition | Caution |
|---|---|---|
| Return rate | Returns / sales | Channel and category differ |
| Recovery rate | Value recovered / value returned | Can hide disposal |
| Recycling rate | Mass recycled / mass collected | Quality of recycling varies |
| Reuse rate | Units reused / units collected | Reuse may defer, not avoid, disposal |
| Circular material use rate | Recycled material / total material | Economy-level, not firm-level |
| Landfill diversion | Waste diverted / total waste | Diversion is not recovery |
| Product lifetime | Average years in use | Hard to measure |

Avoid claiming circularity from a single percentage. State the boundary, the denominator, and what is excluded.

### 4.11 Worked example — return disposition

An e-commerce retailer has 500,000 orders per year with a 20% return rate, so 100,000 returns. Average forward margin is €30. The return processing cost is €8, and transport is €6.

Grading finds:

| Grade | Share | Resale price | Refurb cost | Net |
|---|---:|---:|---:|---:|
| A (like new) | 45% | €60 | €0 | €46 after costs |
| B (used) | 30% | €40 | €5 | €21 |
| C (repair) | 15% | €25 | €15 | −€4 |
| D (scrap) | 10% | €2 | €0 | −€12 |

Weighted net recovery per return:

\[
0.45 \times 46 + 0.30 \times 21 + 0.15 \times (-4) + 0.10 \times (-12) = 20.7 + 6.3 - 0.6 - 1.2 = €25.2
\]

If the average return also loses €30 of forward margin, the return costs about €4.8 net. The question is whether the returns policy drives enough incremental sales to cover it, and whether grading accuracy can move C units to B.

### 4.12 Worked example — reverse network consolidation

A retailer collects 1,000 parcels per day from 200 stores. Direct store-to-processing transport costs €3.50 per parcel. A consolidation hub costs €1.20 per parcel to operate plus €0.80 per parcel in line-haul, but adds one day of delay.

Direct cost: €3,500 per day. Hub cost: €2,000 per day, saving €1,500 per day or about €547,500 per year. The delay costs one day of value depreciation; if the average item loses €1 per day, the cost is €1,000 per day and the hub still wins. If the item is a fast-depreciating phone losing €4 per day, the hub loses.

The trade-off is explicit: **consolidation saves transport but costs time and value**.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Order and return | OMS / e-commerce | Return rate and reason |
| Return reason | Customer / agent | Root cause |
| Condition grade | Inspection system | Disposition |
| Refurb cost | Work order / costing | Unit economics |
| Resale price | Market feed | Recovery value |
| Collection volume | Store / locker | Network design |
| Core supply | Trade-in programme | Remanufacturing |
| Battery health | Test equipment | Second life |
| Recycling mass | Recycler report | Compliance |

### 5.2 Standards and regulation

- **EU WEEE Directive** for electronic waste.
- **EU Batteries Regulation** for collection, recycling, and battery passports.
- **EU Ecodesign for Sustainable Products Regulation** and digital product passport.
- **Right to repair** legislation in the EU and several national jurisdictions.
- **Extended producer responsibility (EPR)** schemes for packaging and products.
- **Basel Convention** on transboundary movement of waste.
- **Dangerous goods rules** (IATA, IMDG, ADR) for damaged batteries.
- **ISO 14001** environmental management; **ISO 59000** circular economy standards family.

### 5.3 Workflow

1. Capture the return request and reason.
2. Collect the unit.
3. Consolidate and transport.
4. Inspect and grade.
5. Decide disposition.
6. Refurbish, repair, or harvest.
7. Resell, recycle, or dispose.
8. Report compliance and recovery.

## 6. Industry case studies

### 6.1 Retail returns and recommerce

**Context:** High return rates in apparel and electronics.
**Intervention:** Paid returns, drop-off networks, and resale marketplaces.
**Evidence:** Company disclosures and industry research (grade C/D); figures vary widely by retailer.
**Outcome:** Some retailers reduced return cost by charging or by steering to drop-off; others built recommerce channels.
**What transferred:** return policy is a demand-shaping lever, not only a logistics cost.
**What did not:** specific rates and margins.

### 6.2 Deposit return schemes for packaging

**Context:** Governments introduced deposits on beverage containers.
**Intervention:** Reverse vending, retailer collection, and producer-funded systems.
**Evidence:** Government evaluations (grade B).
**Outcome:** Collection rates rose sharply where the deposit was convenient and well communicated.
**What transferred:** convenience and deposit level drive return rate more than exhortation.
**What did not:** specific collection rates.

### 6.3 Battery second life

**Context:** Electric vehicle batteries retired with remaining capacity.
**Intervention:** Testing, grading, and repurposing for stationary storage.
**Evidence:** Academic and industry studies (grade B/C).
**Outcome:** Second life is technically feasible but economically dependent on testing cost, transport, and market price.
**What transferred:** state-of-health verification is the gate to value.
**What did not:** specific economics.

### 6.4 A composite failure — the returns warehouse that became a landfill

**Context:** A retailer centralised returns but did not staff grading or resale.
**Intervention:** None; returns accumulated.
**Evidence:** Composite teaching case.
**Outcome:** Value depreciated to zero, storage cost rose, and most units were scrapped.
**What transferred:** a reverse network without disposition capacity is a storage cost, not a recovery system.
**What did not:** the specific volumes.

## 7. Labs

### 7.1 Lab 37.1 — Disposition model

Build a disposition model for 10,000 returned units with grades, refurb costs, and resale prices. Maximise net recovery subject to capacity.

**Deliverable:** Optimisation model and policy rules.

### 7.2 Lab 37.2 — Grading error analysis

Given predicted and actual grades, build a confusion matrix and estimate the cost of each error type. Propose a threshold change.

**Deliverable:** Error-cost analysis and recommendation.

### 7.3 Lab 37.3 — Reverse network design

Design a collection and consolidation network for 200 stores and three processing sites. Compare cost, time, and recovery value.

**Deliverable:** Network design and trade-off analysis.

### 7.4 Lab 37.4 — Returns fraud detection

Build a fraud-risk score on synthetic return data. Evaluate precision and recall and design a review process.

**Deliverable:** Model, evaluation, and operating policy.

### 7.5 Lab 37.5 — Packaging reuse break-even

Model a reusable packaging scheme with return rate, loss, washing cost, and trips. Find the break-even number of trips and the sensitivity.

**Deliverable:** Break-even model and recommendation.

## 8. Assignment / project

Design a reverse supply chain for one product category.

**Required components:**

1. Flow map and volumes.
2. Collection and consolidation network.
3. Grading and disposition model.
4. Unit economics and break-even.
5. Refurbishment or remanufacturing plan.
6. Regulatory compliance.
7. Fraud and abuse controls.
8. Circular metrics and reporting.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Flow understanding | 20% | Lists returns | Maps flow types | Distinguishes value and regulation |
| Disposition | 25% | Simple rule | Cost-based model | Optimisation with constraints |
| Network design | 20% | Describes collection | Cost comparison | Time-value and consolidation trade-off |
| Unit economics | 15% | Cost per return | Net recovery | Break-even and sensitivity |
| Regulation | 10% | Names directive | Compliance plan | Product passport and EPR |
| Metrics | 10% | Reports rate | Boundary-stated metric | Recovery quality and limits |

## 10. Regulatory, ethical, and commercial lens

- **Producer responsibility:** EPR makes the producer liable for collection and recycling.
- **Waste shipment:** transboundary waste rules restrict where goods can go.
- **Right to repair:** access to parts, tools, and manuals is increasingly mandated.
- **Data:** returned devices may contain personal data; wiping must be verifiable.
- **Safety:** damaged batteries are dangerous goods; transport and storage rules apply.
- **Labour:** informal recycling can involve hazardous work; sourcing recyclers responsibly matters.
- **Greenwashing:** recovery claims must be bounded and auditable.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No disposition capacity | Returns pile up | Grading bottleneck | Value loss | Capacity plan |
| Free returns without cost | Return rate rising | Policy not priced | Margin erosion | Policy and segmentation |
| Over-grading | Second returns | Inspection error | Brand damage | Test and audit |
| Under-grading | Low resale price | Conservative grading | Value loss | Grade calibration |
| Centralised everything | Long transport | Low density | Cost per unit high | Hub-and-spoke |
| Reverse network ignored | No data | Not in planning | Poor service | Include in S&OP |
| Fraud undetected | Serial returners | No controls | Loss | Risk scoring |
| Battery unsafe | Damaged units shipped | No dangerous-goods process | Safety incident | Training and packaging |
| Circular claim overstated | Single percentage | Boundary omitted | Greenwashing | Bounded metrics |
| Recycler uncontrolled | No audit | Informal sector | Environmental harm | Supplier standards |

## 12. Interview and discussion questions

1. Why is the reverse supply chain harder to plan than the forward one? *(Hint: condition unknown until inspection.)*
2. What is the disposition hierarchy? *(Hint: resell, refurbish, repair, remanufacture, harvest, recycle.)*
3. How do you decide whether to refurbish or scrap? *(Hint: net recovery versus cost.)*
4. Why is grading error asymmetric? *(Hint: under-grading loses value, over-grading causes returns.)*
5. How do you design a reverse network? *(Hint: many-to-few, low density, consolidation.)*
6. What is the break-even for reusable packaging? *(Hint: trips versus single-use cost.)*
7. How do you detect return fraud without harming good customers? *(Hint: risk score plus review.)*
8. What is needed to remanufacture at scale? *(Hint: core supply, yield, quality, market.)*
9. How do you verify battery state of health? *(Hint: test, data, and safety.)*
10. What does extended producer responsibility require? *(Hint: collection, recycling, reporting.)*
11. How do you avoid overstating circularity? *(Hint: boundary, denominator, exclusions.)*
12. What is the role of the digital product passport? *(Hint: identity and lifecycle data.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Reverse logistics | Movement of goods from customer back to the chain |
| Disposition | Decision on what to do with a returned unit |
| Grading | Assigning a condition class |
| Recovery yield | Value recovered per unit collected |
| Core | Returned unit used for remanufacture |
| Remanufacturing | Restoring to like-new with warranty |
| Refurbishment | Cleaning, testing, and repair for resale |
| Harvesting | Reusing components from a unit |
| Recommerce | Selling used or refurbished goods |
| EPR | Extended producer responsibility |
| WEEE | Waste electrical and electronic equipment |
| Deposit return scheme | Deposit paid and refunded on return |
| Second life | Repurposing a product or component |
| State of health | Remaining capacity and safety of a battery |
| Digital product passport | Lifecycle data record for a product |
| Landfill diversion | Waste kept out of landfill |
| Greenwashing | Overstating environmental benefit |

## 14. Sources and further reading

### Standards and regulation

- EU WEEE Directive 2012/19/EU. https://eur-lex.europa.eu/
- EU Batteries Regulation (EU) 2023/1542. https://eur-lex.europa.eu/
- EU Ecodesign for Sustainable Products Regulation. https://eur-lex.europa.eu/
- Basel Convention. https://www.basel.int/
- ISO 59000 circular economy standards. https://www.iso.org/
- IATA Dangerous Goods Regulations. https://www.iata.org/

### Reports and official sources

- Ellen MacArthur Foundation, circular economy reports. https://ellenmacarthurfoundation.org/
- European Environment Agency, waste and circularity data. https://www.eea.europa.eu/
- OECD, extended producer responsibility guidance. https://www.oecd.org/
- US EPA, recycling and waste data. https://www.epa.gov/

### Peer-reviewed and professional

- Guide, V. D. R., Van Wassenhove, L. N., "The reverse supply chain," *Harvard Business Review*.
- Guide, V. D. R., et al., research on remanufacturing and reverse logistics.
- Govindan, K., et al., review of reverse logistics and closed-loop supply chains.
- Blackburn, J. D., et al., "Reverse supply chains for commercial returns."

### Industry

- Reverse Logistics Association. https://www.reverselogisticsassociation.org/
- Recommerce and refurbishment industry reports (vendor data; treat as grade C/D).

> **Next:** [Module 38](38_process_industries_commodities.md) closes the domain set with process industries, where the product is a continuous flow rather than a discrete unit.
