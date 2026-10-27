# Module 29 — Industry Electives: CPG, Pharma and Cold Chain, Automotive, Semiconductor, Humanitarian, Fashion

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 24 hours (12 lecture, 8 lab, 4 case clinic)
> **Track fit:** Advanced elective; choose two or more primers
> **Prerequisites:** Modules 01, 07, 08, 10, 14A
> **Tools:** The core stack plus sector data where available
> **Industry question:** "What changes when the same analytics method meets a different product, regulation, lead time, and cost of failure?"

---

## 1. Why this module exists

Supply chain analytics is not one market. A grocery promotion, a pharmaceutical batch record, a automotive sequence delivery, a semiconductor allocation, a humanitarian prepositioning decision, and a fashion markdown have different data, constraints, and consequences.

The methods transfer. The **decision, the data grain, the standards, the cost of error, and the regulatory boundary** do not. This module teaches the differences through six primers. Each primer is a compressed industry immersion: the decision and buyer, the data and standards, the methods that matter, real use cases, regulation and equity, a lab, and interview questions.

Choose at least two primers. A single primer is an introduction; two primers reveal how context changes the answer.

## 2. Learning outcomes

By the end of this module, you can:

- Explain the core planning decision and economic buyer in six supply chain sectors.
- Identify the data sources, standards, and constraints that differ by sector.
- Select methods that fit the sector's lead time, shelf life, regulation, and cost structure.
- Evaluate a sector-specific analytics claim against its evidence.
- Recognize the regulatory and ethical boundary in each sector.
- Design a sector-appropriate lab or capstone.
- Interview credibly for a sector-specific analytics role.

## 3. How to read each primer

Each primer follows the same structure:

| Element | Question |
|---|---|
| Decision and buyer | What is decided, by whom, on what cadence? |
| Product and network | What moves, how fast, and through which nodes? |
| Data and standards | Which systems, identifiers, and regulatory records? |
| Methods that matter | Forecasting, optimization, simulation, or causal? |
| Use cases | What real companies or agencies have done |
| Regulation and equity | What binds, and who can be harmed? |
| Lab | A concrete exercise |
| Interview questions | What a strong candidate says |

## 4. Primer 1 — CPG and retail

### 4.1 Decision and buyer

| Decision | Buyer | Cadence |
|---|---|---|
| Demand plan by category and account | VP demand planning | Monthly S&OP, weekly sensing |
| Trade promotion investment | Commercial finance | Event-based |
| Production and co-packer allocation | Supply planning | Weekly |
| Shelf availability and replenishment | Customer supply chain | Daily/weekly |

### 4.2 Product and network

CPG products are often low-margin, high-volume, and shelf-stable or chilled. The network runs from co-packers and plants through retailer DCs to stores. The retailer owns the shelf, so the manufacturer sees shipments and imperfect POS data. Distributor inventory sits between the manufacturer and the store.

### 4.3 Data and standards

- Retailer POS and syndicated data (Circana, NielsenIQ) under contract.
- Distributor inventory and depletion reports.
- EDI 852 (product activity), 850 (purchase order), 856 (ship notice), 810 (invoice).
- GS1 GTIN and GLN.
- Trade promotion calendars and deduction files.
- Panel data for household behavior.

### 4.4 Methods that matter

- Hierarchical demand forecasting with promotion and price effects.
- Baseline decomposition and incrementality.
- Trade promotion optimization under budget and cannibalization constraints.
- Shelf-availability analytics from POS and store audits.
- Assortment and space analytics with the retailer.

### 4.5 Use cases

- Procter & Gamble's continuous replenishment work with retailers in the 1990s is a canonical CPG–retailer collaboration case.
- Walmart's M5 dataset and competition showed cross-learning and exogenous promotion variables at scale.
- Retailer–manufacturer CPFR pilots improved forecast sharing but often stalled on incentives and data trust.

### 4.6 Regulation and equity

- Competition law limits sharing of competitor pricing and future plans.
- Promotion and pricing claims must be substantiated.
- Category captaincy can create conflicts of interest.
- Supplier payment terms and deductions affect small suppliers.

### 4.7 Lab

Build a weekly category forecast with a promotion calendar. Decompose baseline and incremental volume. Compare a manufacturer shipment forecast with a retailer POS forecast and explain the bullwhip gap.

### 4.8 Interview questions

1. Why does manufacturer shipment demand differ from consumer takeaway?
2. How do you measure promotion incrementality without a control store?
3. What is the risk of using retailer POS data for a national forecast?
4. How would you allocate a constrained product across accounts?

## 5. Primer 2 — Pharma and cold chain

### 5.1 Decision and buyer

| Decision | Buyer | Cadence |
|---|---|---|
| Clinical and commercial supply plan | VP supply chain | Monthly, with rolling updates |
| Batch release and allocation | Quality and supply | Batch |
| Cold-chain excursion response | Logistics and quality | Real time |
| Serialization and tracing compliance | Regulatory affairs | Transaction |
| Expiry and write-off | Finance and supply | Monthly |

### 5.2 Product and network

Pharmaceutical supply chains combine long regulatory lead times, batch production, limited shelf life, and strict quality systems. Clinical supply adds randomization, blinding, and country-specific labeling. Commercial supply adds wholesalers, dispensers, and serialization.

### 5.3 Data and standards

- DSCSA in the U.S., EU FMD in Europe.
- GS1 GTIN, SSCC, and EPCIS events.
- Batch records, certificates of analysis, and quality systems.
- Temperature telemetry and excursion logs.
- Good Distribution Practice (GDP) and WHO GDP.
- Clinical trial supply systems and interactive response technology.

### 5.4 Methods that matter

- Demand forecasting with launch uncertainty and patient persistence.
- Inventory and expiry optimization under batch constraints.
- Cold-chain excursion risk and route reliability.
- Serialization data quality and exception reconciliation.
- Scenario planning for regulatory and launch delays.

### 5.5 Use cases

- COVID-19 vaccine distribution exposed cold-chain, allocation, and equity trade-offs at global scale.
- DSCSA enforcement phased through 2025–2026, forcing package-level EPCIS exchange.
- GLP-1 supply constraints in 2023–2025 showed how demand surges collide with batch capacity and payer access.

### 5.6 Regulation and equity

- GxP, GDP, validation, and data integrity.
- DSCSA and EU FMD serialization.
- Patient safety and access equity.
- Allocation ethics when supply is constrained.
- Privacy of patient-level data.

### 5.7 Lab

Model a two-echelon cold-chain network with batch sizes, shelf life, and a temperature excursion probability. Compare cost and waste under central versus local buffering.

### 5.8 Interview questions

1. How does shelf life change the inventory policy?
2. What does an EPCIS event prove, and what does it not prove?
3. How would you forecast a launch with no history?
4. When is a temperature excursion a quality event versus a data event?

## 6. Primer 3 — Automotive

### 6.1 Decision and buyer

| Decision | Buyer | Cadence |
|---|---|---|
| Vehicle production plan | VP manufacturing | Weekly/monthly |
| Sequence and JIT delivery | Plant logistics | Daily/hourly |
| Supplier allocation and risk | Purchasing | Weekly |
| Model changeover | Program management | Program |
| Warranty and recall analytics | Quality | Event |

### 6.2 Product and network

Automotive is a multi-tier, sequence-driven network with high capital intensity, long tooling lead times, and safety-critical quality. JIT and JIS require supplier proximity and reliable logistics. A single missing part can stop a line.

### 6.3 Data and standards

- EDI 830 (planning schedule), 862 (shipping schedule), 856, 850.
- ISA-95/MES integration and OPC UA.
- IATF 16949 quality management.
- APQP, PPAP, and traceability records.
- Warranty claims and field failure data.

### 6.4 Methods that matter

- Demand-supply balancing under mix and option complexity.
- Finite scheduling and sequence optimization.
- Multi-tier supplier risk and single-point-of-failure mapping.
- Predictive quality and warranty analytics.
- Scenario planning for model changeover and recall.

### 6.5 Use cases

- The 2011 Tōhoku earthquake disrupted automotive supplier networks globally, exposing tier-2 and tier-3 opacity.
- The 2021 semiconductor shortage forced allocation, de-contenting, and production cuts across automakers.
- Toyota's production system is the reference for JIT, kanban, and andon, but it depends on supplier proximity and stability.

### 6.6 Regulation and equity

- Safety recalls and traceability.
- Emissions and fuel-economy regulation.
- IATF 16949 and customer-specific requirements.
- Supplier payment practices and tier-3 labor conditions.
- Transition risk for EV and battery supply chains.

### 6.7 Lab

Build a multi-tier supplier map for one vehicle module. Identify single points of failure, estimate time-to-recover, and compare dual-sourcing against inventory buffering.

### 6.8 Interview questions

1. Why is JIT vulnerable to a tier-3 disruption?
2. How would you schedule a plant with sequence-dependent changeovers?
3. What data would you use to predict a warranty claim?
4. How do you value a day of downtime?

## 7. Primer 4 — Semiconductor

### 7.1 Decision and buyer

| Decision | Buyer | Cadence |
|---|---|---|
| Wafer start and capacity allocation | VP operations | Weekly/monthly |
| Customer allocation | Sales and operations | Weekly |
| Demand-supply matching | Planning | Weekly |
| Capital expansion | CFO and board | Annual/multi-year |
| Yield and quality | Fab operations | Daily |

### 7.2 Product and network

Semiconductors have very long lead times, huge capital intensity, yield uncertainty, and a concentrated supply base. A fab takes years and billions to build. Products may be dual-sourced or single-sourced. Demand can be distorted by double ordering and allocation behavior.

### 7.3 Data and standards

- Demand forecasts, backlog, and design wins.
- Wafer starts, cycle time, yield, and binning.
- Capacity by technology node and package.
- Supplier and fab utilization.
- SEMI standards for equipment communication.

### 7.4 Methods that matter

- Demand-supply matching under yield and capacity uncertainty.
- Allocation rules under scarcity and strategic priority.
- Scenario planning for capacity expansion.
- Double-order detection and demand cleansing.
- Lead-time and cycle-time analytics.

### 7.5 Use cases

- The 2020–2023 chip shortage led automakers and governments to re-examine just-in-time and geographic concentration.
- TSMC's advanced-node concentration is a canonical single-point-of-failure case.
- Allocation disputes during shortages illustrated the difference between forecast accuracy and allocation policy.

### 7.6 Regulation and equity

- Export controls and geopolitical restrictions.
- Intellectual property and technology transfer.
- Environmental impact of fabs (water, energy, chemicals).
- Allocation fairness across customers and countries.
- National industrial policy and subsidies.

### 7.7 Lab

Build a capacity allocation model with yield uncertainty and two customer priority classes. Compare pro-rata, priority, and optimization-based allocation on revenue, service, and fairness.

### 7.8 Interview questions

1. How do you detect double ordering?
2. What is the right allocation rule when supply is short?
3. How does yield uncertainty change capacity planning?
4. Why is a semiconductor lead time not a single number?

## 8. Primer 5 — Humanitarian supply chain

### 8.1 Decision and buyer

| Decision | Buyer | Cadence |
|---|---|---|
| Prepositioning and stock levels | Country office and HQ | Annual/seasonal |
| Needs assessment and targeting | Program and M&E | Event |
| Procurement and sourcing | Procurement | Event |
| Last-mile delivery | Logistics cluster | Daily/weekly |
| Donor reporting | Finance and grants | Grant cycle |

### 8.2 Product and network

Humanitarian supply chains serve uncertain, urgent, and often inaccessible demand. Funding is earmarked and volatile. Local procurement can support markets; international procurement can be faster or more compliant. Last-mile access may depend on security and infrastructure.

### 8.3 Data and standards

- Needs assessments and cluster data.
- INGO and UN logistics systems.
- Sphere standards and humanitarian principles.
- Donor reporting requirements.
- Remote sensing and access data.

### 8.4 Methods that matter

- Needs forecasting under deep uncertainty.
- Prepositioning and network design with access risk.
- Procurement and supplier diversification.
- Last-mile routing and delivery prioritization.
- Scenario planning and allocation ethics.

### 8.5 Use cases

- The 2010 Haiti earthquake and 2014–2015 Ebola response exposed data coordination and last-mile constraints.
- COVID-19 vaccine distribution in low-income countries combined cold chain, allocation, and funding constraints.
- Humanitarian logistics research on prepositioning shows the trade-off between speed and holding cost.

### 8.6 Regulation and equity

- Humanitarian principles: humanity, neutrality, impartiality, independence.
- Do-no-harm and protection of affected people.
- Donor conditionality and earmarking.
- Local market effects of in-kind aid.
- Data protection for vulnerable populations.

### 8.7 Lab

Design a prepositioning network for a flood-prone region. Compare centralized, regional, and local stock under access disruption scenarios. Report cost, response time, and unmet need.

### 8.8 Interview questions

1. How do you forecast demand when the event has not happened?
2. When should you procure locally versus internationally?
3. How do you allocate scarce stock across affected areas?
4. What does "do no harm" mean for a routing model?

## 9. Primer 6 — Fashion and apparel

### 9.1 Decision and buyer

| Decision | Buyer | Cadence |
|---|---|---|
| Seasonal buy and assortment | Merchandising | Season |
| Replenishment and allocation | Planning | Weekly |
| Markdown and clearance | Merchandising and finance | Weekly |
| Sourcing and lead time | Sourcing | Season |
| Returns and resale | Operations | Ongoing |

### 9.2 Product and network

Fashion has short lifecycles, high variety, size curves, trend uncertainty, and long offshore lead times. Buy decisions are made months before demand is known. Markdowns recover margin but train customers to wait. Returns are high in e-commerce.

### 9.3 Data and standards

- POS and e-commerce orders.
- Size curves and color/style attributes.
- Supplier lead times and capacity.
- Markdown and promotion history.
- Returns reasons and resale data.
- GS1 and product information management.

### 9.4 Methods that matter

- New product forecasting with analogy and attributes.
- Size-curve and variant forecasting.
- Allocation and replenishment under lead-time uncertainty.
- Markdown optimization and clearance.
- Assortment and trend analytics.
- Returns and resale analytics.

### 9.5 Use cases

- Inditex/Zara's fast fashion model shortens design-to-shelf lead time and uses store feedback to adjust.
- Shein's on-demand small-batch model uses real-time demand signals and a large supplier network.
- Luxury and outdoor brands use scarcity and markdown discipline to protect brand value.

### 9.6 Regulation and equity

- Labor conditions in garment supply chains.
- Chemical and product safety.
- Greenwashing and sustainability claims.
- Returns and textile waste.
- Size inclusivity and pricing equity.

### 9.7 Lab

Forecast a seasonal style with eight weeks of sales and a long lead time. Compare a buy plan based on a point forecast with one based on a quantile forecast and markdown salvage. Report gross margin and terminal inventory.

### 9.8 Interview questions

1. How do you forecast a style with no history?
2. Why is the size curve a separate forecasting problem?
3. How do you decide markdown depth and timing?
4. What is the cost of a stockout versus a markdown?

## 10. Cross-sector comparison

| Dimension | CPG | Pharma | Automotive | Semiconductor | Humanitarian | Fashion |
|---|---|---|---|---|---|---|
| Lead time | Weeks | Months | Days–weeks | Months–years | Days–weeks | Months |
| Shelf life | Medium | Critical | Low | Low | Variable | Seasonal |
| Variety | High | Medium | High | Medium | Low–medium | Very high |
| Demand uncertainty | Medium | High (launch) | Medium | High | Very high | Very high |
| Capacity constraint | Medium | High | High | Very high | High | Medium |
| Regulation | Medium | Very high | High | High | High | Medium |
| Cost of stockout | Medium | High | Very high | Very high | Critical | Medium |
| Data visibility | Retailer-dependent | Batch/serialized | Tiered | Concentrated | Fragmented | Channel-dependent |
| Dominant method | Forecast + promo | Forecast + expiry | Scheduling + risk | Allocation | Prepositioning | NPI + markdown |

## 11. Labs and assessment

### Lab 29.1 — Sector decision map

Choose two primers. For each, map the top five decisions, the buyer, the cadence, the data grain, and the dominant constraint. Compare them in a table.

**Deliverable:** Two-page comparison.

**Acceptance criteria:** Correct vocabulary; explicit trade-offs; at least one method that does not transfer.

### Lab 29.2 — Sector dataset audit

Find a public or permitted dataset for one sector. Write a data card and assess fit for a specific decision. State the license and the limitations.

**Deliverable:** Data card and fit memo.

### Lab 29.3 — Sector capstone brief

Write a one-page capstone brief for a sector decision, including data, method, validation, impact, and regulatory triage.

**Deliverable:** Brief and a five-minute pitch.

### Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Sector vocabulary | 20% | Generic supply chain terms | Correct decisions and constraints | Explains why the sector differs |
| Data and standards | 20% | Lists systems | Correct identifiers and records | Explains regulatory and provenance limits |
| Method fit | 25% | Applies a favorite method | Chooses by constraint and cost | Explains trade-offs and alternatives |
| Regulation and equity | 15% | Generic statement | Names the binding rules | Identifies affected groups and controls |
| Communication | 20% | Descriptive | Clear comparison | Decision-ready and honest |

## 12. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Method transplant | "It worked in retail" | Ignores lead time and shelf life | Wrong policy | Re-derive from constraints |
| Data overreach | Uses POS as demand | Shipments and stockouts distort | Bias | Model censoring and channel |
| Regulation blind spot | No quality or trade review | Sector rules ignored | Compliance risk | Early legal/quality review |
| Allocation by model only | No fairness rule | Optimization favors power | Ethical and legal risk | Explicit allocation policy |
| Humanitarian abstraction | Treats affected people as demand | Dehumanizing model | Harm | Do-no-harm review |
| Fashion point forecast | Single number for a style | Ignores tail | Markdown or stockout | Quantile and salvage |
| Semiconductor double count | Trusts backlog | Double ordering | Overcapacity | Demand cleansing |
| Cold-chain data trust | Assumes sensor is right | Calibration and gaps | False quality signal | Data validation |

## 13. Interview and discussion questions

1. Which sector has the highest cost of a stockout, and why?
2. How does shelf life change the inventory policy?
3. Why is a semiconductor lead time not a single number?
4. What makes humanitarian demand different from commercial demand?
5. How would you allocate a constrained pharmaceutical product?
6. What is the role of standards in pharmaceutical traceability?
7. How does fashion markdown interact with forecasting?
8. Why is automotive tier-2 visibility so hard?
9. Which sector would you choose for a capstone, and what decision would you improve?
10. What method from your current domain would fail in another sector, and why?

## 14. Sources and further reading

- Chopra, S., Meindl, P., *Supply Chain Management*.
- Simchi-Levi, D. et al., *Designing and Managing the Supply Chain*.
- Sheffi, Y., *The Resilient Enterprise*.
- Lee, H. L. et al., "Information Distortion in a Supply Chain," *Management Science*, 1997.
- Ferdows, K., Lewis, M., Machuca, J., "Rapid-Fire Fulfillment," *Harvard Business Review*, 2004.
- Kraljic, P., "Purchasing Must Become Supply Management," *Harvard Business Review*, 1983.
- FDA, DSCSA implementation. https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa
- GS1, EPCIS 2.0. https://www.gs1.org/standards/epcis
- IATF 16949. https://www.iatfglobaloversight.org/
- SEMI standards. https://www.semi.org/
- Sphere Standards. https://spherestandards.org/
- UN Comtrade. https://comtradeplus.un.org/
- World Bank LPI. https://lpi.worldbank.org/

> **Next:** [Module 30](30_simulation_optimization_production_labs.md) turns sector constraints into simulation and production labs.
