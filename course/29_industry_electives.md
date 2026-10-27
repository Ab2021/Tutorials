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
- Design a sector-appropriate lab or domain project.
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

### 4.9 Sector KPI tree and cost model

| Layer | Metric | Why it matters |
|---|---|---|
| Primary | Shelf availability / case fill rate | The retailer's service measure and the manufacturer's penalty |
| Diagnostic | Forecast bias by account; promotion lift error | Explains the primary metric before anyone blames the model |
| Cost | Trade spend ROI; cost per case; deduction rate | Turns volume into profit |
| Cash | DIO at plant and distributor; DSO | Working capital tied up in a low-margin chain |

The cost model has four lines that are often mixed: manufacturing cost, freight to the retailer DC, trade spend, and deductions. A promotion that raises volume but worsens deduction rate can destroy margin while every service metric improves.

### 4.10 Sector-specific failure modes

| Failure | Mechanism | Consequence |
|---|---|---|
| Shipments treated as demand | Retailer inventory absorbs the difference | Bullwhip and overproduction |
| Promoted sales treated as baseline | No counterfactual | Permanent over-forecast |
| Deduction leakage | Invalid retailer deductions never disputed | Silent margin loss |
| Category captaincy conflict | Retailer data used to favour own brands | Legal and relationship risk |

### 4.11 Twelve-week analytics plan

1. Weeks 1–2: reconcile shipment, POS, distributor inventory, and the promotion calendar.
2. Weeks 3–4: build a baseline by category and account; quantify forecast bias.
3. Weeks 5–6: measure promotion incrementality and cannibalization.
4. Weeks 7–8: model constrained allocation across accounts.
5. Weeks 9–10: pilot a shelf-availability alert with one retailer.
6. Weeks 11–12: write the value case and the data-sharing agreement.

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

### 5.9 Sector KPI tree and cost model

| Layer | Metric | Why it matters |
|---|---|---|
| Primary | OTIF to wholesaler or dispenser; patient access | Service failure can be a supply or an access failure |
| Diagnostic | Batch yield; expiry write-off percentage; excursion rate | Quality and shelf-life losses dominate cost |
| Cost | Cost per batch; cold-chain cost per shipment; write-off | Batch economics are stepwise, not marginal |
| Compliance | Serialization exception rate; data-integrity findings | A compliance failure can stop release |

The cost model is dominated by batch size, release time, and expiry. A demand forecast that is accurate on average can still be wrong for the batch you must commit to now.

### 5.10 Sector-specific failure modes

| Failure | Mechanism | Consequence |
|---|---|---|
| Dispensed demand ignored | Shipments include wholesaler stock-building | Wrong long-run demand |
| Batch constraints omitted | Model assumes continuous supply | Infeasible plan |
| Sensor data trusted blindly | Calibration drift and gaps | False excursion clearance |
| Allocation without an ethics rule | Optimization favours the highest-margin market | Access and reputational harm |

### 5.11 Twelve-week analytics plan

1. Weeks 1–2: map batch, release, shelf life, and country labelling constraints.
2. Weeks 3–4: build demand by patient segment and persistence assumption.
3. Weeks 5–6: model expiry and allocation under batch supply.
4. Weeks 7–8: audit cold-chain telemetry and excursion handling.
5. Weeks 9–10: reconcile serialization events and exceptions.
6. Weeks 11–12: stress-test launch delay and demand surge scenarios.

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

### 6.9 Sector KPI tree and cost model

| Layer | Metric | Why it matters |
|---|---|---|
| Primary | Line-side availability; schedule adherence | A missing part stops the line |
| Diagnostic | Sequence compliance; premium freight; supplier OTIF | Explains why the line stopped |
| Cost | Cost per vehicle; downtime cost per hour; warranty cost per unit | Downtime dominates most analytics business cases |
| Risk | Single-point-of-failure count; time-to-recover | The resilience measure the board asks for |

Downtime cost is the number that justifies the project. It must include labour, fixed-cost absorption, and the dealer and customer consequences, not just the lost unit margin.

### 6.10 Sector-specific failure modes

| Failure | Mechanism | Consequence |
|---|---|---|
| JIT without tier visibility | Tier-3 supplier is invisible | Line stop from an unknown dependency |
| Sequence instability | Plan changes after suppliers commit | Expediting and premium freight |
| Changeover underestimated | Sequence-dependent setup ignored | Infeasible schedule |
| Warranty claims analysed without exposure | Claims counted per unit, not per vehicle in service | Wrong quality ranking |

### 6.11 Twelve-week analytics plan

1. Weeks 1–2: build the multi-tier supplier map for one module.
2. Weeks 3–4: quantify downtime cost and single points of failure.
3. Weeks 5–6: model finite scheduling with changeovers.
4. Weeks 7–8: analyse premium freight and expedite triggers.
5. Weeks 9–10: build a warranty and field-failure baseline.
6. Weeks 11–12: compare dual sourcing, buffering, and redesign.

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

### 7.9 Sector KPI tree and cost model

| Layer | Metric | Why it matters |
|---|---|---|
| Primary | Allocation fill rate by customer; revenue at risk | Supply is allocated, not sold |
| Diagnostic | Backlog quality; double-order index; cycle time; yield | Explains whether demand is real and supply is available |
| Cost | Cost per wafer; cost per good die; capex per node | Yield converts wafers into sellable output |
| Strategic | Capacity by node; customer concentration | Determines multi-year position |

Capacity must be measured in the unit that can be sold. A wafer-start plan is not a die-supply plan until yield, binning, and packaging are applied.

### 7.10 Sector-specific failure modes

| Failure | Mechanism | Consequence |
|---|---|---|
| Backlog treated as demand | Double ordering during shortage | Overcapacity and cancellation |
| Wafer starts treated as supply | Yield and binning ignored | Shortfall at the customer |
| Mean yield used in planning | Yield distribution is skewed | Service misses in bad lots |
| Lead time treated as a scalar | Queue time depends on utilization | Broken promise |

### 7.11 Twelve-week analytics plan

1. Weeks 1–2: cleanse backlog and detect double ordering.
2. Weeks 3–4: map capacity by node, package, and test.
3. Weeks 5–6: model yield and binning as a distribution.
4. Weeks 7–8: compare allocation rules under scarcity.
5. Weeks 9–10: scenario-test capacity expansion.
6. Weeks 11–12: write the allocation policy and its fairness criteria.

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

### 8.9 Sector KPI tree and cost model

| Layer | Metric | Why it matters |
|---|---|---|
| Primary | Response time; unmet need; people reached | The purpose of the chain |
| Diagnostic | Needs-assessment coverage; pipeline lead time; access constraints | Explains response performance |
| Cost | Cost per beneficiary; cost per metric tonne delivered | Donor and programme efficiency |
| Stewardship | Local procurement share; market impact; data protection | Do-no-harm and sustainability |

A cost-per-beneficiary number without a coverage and quality statement is not a result. Report the denominator, the targeting method, and what was not delivered.

### 8.10 Sector-specific failure modes

| Failure | Mechanism | Consequence |
|---|---|---|
| Population treated as demand | No needs, access, or preference data | Wrong commodities, waste |
| Access risk omitted | Route assumes road and security | Failed delivery |
| In-kind aid ignores markets | Local suppliers displaced | Longer-term harm |
| Beneficiary data unprotected | Sensitive data shared widely | Protection risk |

### 8.11 Twelve-week analytics plan

1. Weeks 1–2: assemble needs, access, and pipeline data.
2. Weeks 3–4: build a scenario-based demand range, not a point forecast.
3. Weeks 5–6: design prepositioning under access disruption.
4. Weeks 7–8: compare local and international procurement options.
5. Weeks 9–10: model last-mile prioritization and unmet need.
6. Weeks 11–12: run a do-no-harm and data-protection review.

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

### 9.9 Sector KPI tree and cost model

| Layer | Metric | Why it matters |
|---|---|---|
| Primary | Sell-through; full-price sell-through | Measures whether the buy was right |
| Diagnostic | Size-curve error; markdown depth; weeks of supply | Explains the primary metric |
| Cost | Gross margin; markdown cost; terminal inventory | The profit consequence of the buy |
| Customer | Return rate; resale recovery | The cost of a bad fit or a bad description |

The buy is a portfolio decision. A style that sells out at full price and a style that clears at 60% off can have the same average forecast error and very different profit.

### 9.10 Sector-specific failure modes

| Failure | Mechanism | Consequence |
|---|---|---|
| Point forecast for a style | Ignores the demand tail | Stockout or markdown |
| Size curve treated as fixed | Geography and channel change it | Broken size availability |
| Lead time ignored in trend response | Order placed before signal | Wrong buy |
| Returns excluded from unit economics | Free-looking growth | Negative contribution |

### 9.11 Twelve-week analytics plan

1. Weeks 1–2: build style, colour, size, and channel hierarchies.
2. Weeks 3–4: forecast with analogy and attributes; quantify the tail.
3. Weeks 5–6: model the size curve by store cluster.
4. Weeks 7–8: optimize markdown timing and depth with salvage.
5. Weeks 9–10: analyse returns by reason and fit.
6. Weeks 11–12: compare buy scenarios on gross margin and terminal inventory.

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

### 10.1 Three decisions that never transfer cleanly

**Allocation under scarcity.** CPG allocates a short promotion; pharma allocates doses; automotive allocates parts to a line; semiconductors allocate wafers; humanitarian agencies allocate relief; fashion allocates stock to stores. The optimization looks similar, but the legitimate objective differs: revenue, patient access, line uptime, strategic customer position, unmet need, or full-price sell-through. Writing the objective is the ethical and commercial act.

**Traceability.** CPG traces a lot to a retailer DC; pharma traces a package to a patient; automotive traces a part to a VIN; semiconductor traces a lot to a customer; humanitarian traces a consignment to a distribution point; fashion traces a garment to a fibre or factory. The data standard and the required granularity change the entire data pipeline.

**Launch or new-product forecasting.** CPG launches into known distribution; pharma launches into uncertain reimbursement and prescribing; automotive launches a model year into dealer pipelines; semiconductor launches a design win into a long qualification; humanitarian responds to an event with no baseline; fashion launches a style into a season. Analogy works in all six, but the analogue set and the constraint that dominates early demand are different.

### 10.2 Data-model comparison

| Dimension | CPG | Pharma | Automotive | Semiconductor | Humanitarian | Fashion |
|---|---|---|---|---|---|---|
| Product identifier | GTIN | GTIN + batch/serial | Part number + VIN link | Part + lot + wafer | Item + consignment | GTIN + style/colour/size |
| Location identifier | GLN, store, DC | GLN, wholesaler, site | Plant, line, dock | Fab, node, package site | Country, hub, distribution point | Store, DC, channel |
| Time grain | Day/week | Batch/day | Shift/hour | Week/month | Day/week | Day/week |
| Event standard | EPCIS, EDI 852/856 | EPCIS, serialization | EDI 830/862/856 | Internal lot tracking | Cluster 5W | EPCIS, PIM |
| Quality record | Audit, hold | Batch record, CoA | PPAP, traceability | Yield, bin | Needs assessment | Inspection, chemical test |

### 10.3 Choosing a primer pair

Choose a pair that differs on the dimension you need to learn. CPG plus fashion teaches demand-shape and markdown. Pharma plus humanitarian teaches allocation ethics and traceability. Automotive plus semiconductor teaches multi-tier dependency and capacity. Two similar primers teach vocabulary but not transfer.

### 10A. Building a primer for a sector not listed

The six primers are a method, not a closed list. To add a sector — energy, utilities, defence, construction, healthcare providers, chemicals, mining, agriculture, maritime, rail, public sector — follow the same eight steps.

1. **Start from a decision, not a sector.** "How much spare parts inventory should a distribution network hold for grid maintenance?" is workable. "Utilities analytics" is not.
2. **Name the buyer and the cadence.** Who signs the decision, how often, and what happens when it is wrong?
3. **Describe the product physics.** Value density, shelf life, hazard class, variety, weight and volume, and whether the item is repairable.
4. **Map the network.** Nodes, flows, lead times, capacity, and the degree of vertical integration.
5. **List the data standards.** Identifiers, event standards, quality records, and the regulator that requires them.
6. **Identify the binding constraint.** Capacity, lead time, regulation, capital, labour, or demand uncertainty. The primer should say which one dominates and why.
7. **Collect two or three cases with evidence grades.** Include at least one failure or limit.
8. **Write the playbook and test it in an interview.** If a practitioner does not recognise their own problems, the primer is wrong.

Where to look:

| Source | What it gives | Caveat |
|---|---|---|
| Regulator and government statistics | Definitions, volumes, safety rules | Lagged and aggregated |
| Company annual reports and investor days | Network, capex, disclosed metrics | Company framing |
| Trade associations | Process norms and benchmarks | Member bias |
| Standards bodies (ISO, GS1, industry-specific) | Data and process standards | Not performance evidence |
| Academic operations journals | Methods and causal studies | Context may not transfer |
| Procurement and tender portals | Real volumes and contract terms | Incomplete picture |
| Practitioner interviews | The constraint nobody writes down | Anecdote, grade E |

A completed primer is two to four pages, not twenty. Its value is in stating the constraint and the decision clearly enough that a method can be chosen.

### 10B. Sector readiness diagnostic

Before proposing an analytics program in a new sector, score the environment. The score changes the sequencing, not the ambition.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Data granularity | Monthly aggregates | Weekly, category level | Daily, item and location |
| Master data | Conflicting identifiers | Partial standards | Governed and complete |
| Process stability | Ad hoc planning | Documented process | Stable and measured |
| Measurement maturity | No service or cost baseline | Some KPIs | Agreed KPI tree and baselines |
| Analytical staffing | No internal capability | One or two analysts | Team with domain and engineering |
| Executive sponsorship | Interest without ownership | Sponsor with a pilot | Owner with a budget and cadence |

Interpretation:

- **Mostly 1s:** start with data foundations and one measurement baseline. A forecasting model will fail on bad identifiers.
- **Mostly 2–3:** run one decision-focused pilot with a control group and a defined baseline. Do not buy a platform.
- **Mostly 4–5:** scale across decisions and invest in platform, monitoring, and model risk.
- **A single 1 on master data or sponsorship** blocks scaling regardless of the total. Fix it first or choose a decision that does not depend on it.

The diagnostic is deliberately blunt. Its purpose is to stop a team from building a model on a foundation that cannot support it, and to give the sponsor a concrete list of what to fix.

### 10C. The thirty-minute sector interview

A primer written from documents alone is usually wrong about the constraint. One conversation with a practitioner fixes most of it. Ask in this order and write down the exact words.

| Minutes | Question | What you are listening for |
|---:|---|---|
| 0–5 | "Walk me through the last time this decision went badly." | The real failure mode and who felt it |
| 5–10 | "What do you do when the system and your judgment disagree?" | Shadow process and trust level |
| 10–15 | "Which number do you not believe, and why?" | Measurement defect |
| 15–20 | "What changes in a normal week versus a peak week?" | Variability and the true constraint |
| 20–25 | "If you had one extra person, where would they go?" | Labour bottleneck |
| 25–30 | "What would make you ignore a new model?" | Adoption barriers and past failures |

Then convert the answers into the primer:

- The bad-decision story becomes a failure mode with a mechanism.
- The disagreement becomes an override policy and a trust-calibration requirement.
- The distrusted number becomes a data-quality task before any modeling.
- The peak-week difference becomes the scenario the model must survive.
- The extra-person answer becomes the capacity constraint in the cost model.
- The reason to ignore a model becomes the adoption plan's first risk.

Do not ask "what analytics do you want?" The answer will be a dashboard. Ask about decisions, failures, and workarounds.

### 10D. Primer quality checklist

A primer is ready to teach when all ten are true.

- [ ] It opens with a named decision and a named owner.
- [ ] It states the cadence: daily, weekly, monthly, or event-driven.
- [ ] It describes the product physics that constrain the method.
- [ ] It names the binding constraint and shows what happens when it shifts.
- [ ] It lists the identifiers and event standards a data team must implement.
- [ ] It separates what is usually available from what is usually missing.
- [ ] It gives the KPI tree and the cost model, not only a metric list.
- [ ] It includes two cases and one counter-case, each with an evidence grade.
- [ ] It states the regulatory and ethical boundary.
- [ ] A practitioner in the sector would recognise the problems as theirs.

If the last item fails, the primer is a literature summary. Rewrite it around the interview in Section 10C.

## 11. Labs and assessment

### Lab 29.1 — Sector decision map

Choose two primers. For each, map the top five decisions, the buyer, the cadence, the data grain, and the dominant constraint. Compare them in a table.

**Deliverable:** Two-page comparison.

**Acceptance criteria:** Correct vocabulary; explicit trade-offs; at least one method that does not transfer.

### Lab 29.2 — Sector dataset audit

Find a public or permitted dataset for one sector. Write a data card and assess fit for a specific decision. State the license and the limitations.

**Deliverable:** Data card and fit memo.

### Lab 29.3 — Sector domain-project brief

Write a one-page domain-project brief for a sector decision, including data, method, validation, impact, and regulatory triage.

**Deliverable:** Brief and a five-minute pitch.

### Lab 29.4 — Method transfer test

Take one method from your strongest sector and apply it to a second sector. Write a one-page memo stating what survives, what must be re-derived, and what would fail outright.

**Deliverable:** Transfer memo with a constraint comparison table.

### Lab 29.5 — Regulatory triage

For one sector decision, list the binding regulations and standards, the records they require, the date of the requirement, and the function that must sign off. Mark anything you cannot verify as `[VERIFY]`.

**Deliverable:** One-page triage with owners and dates.

### Lab 29.6 — Readiness diagnostic

Score a real or fictional organisation on the six dimensions in Section 10B. Justify each score with one piece of evidence. Recommend the next decision to attempt and the one to avoid.

**Deliverable:** Scored diagnostic and a sequencing recommendation.

**Acceptance criteria:** Every score has evidence; the blocking dimension is identified; the recommendation follows from the score, not from enthusiasm.

### Lab 29.7 — Build a primer for an unlisted sector

Choose a sector not covered by Primers 1–6. Follow the eight steps in Section 10A and produce a two-page primer. Include at least two cases and one counter-case with evidence grades.

**Deliverable:** Two-page primer and source list.

**Acceptance criteria:** Decision-first framing; binding constraint stated; sources graded; one limit or failure included.

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
| Launch analogy overreach | One analogue for a new category | Analogy is not a distribution | Bad first buy | Use a range of analogues |
| Allocation without a stated objective | Optimizer maximizes revenue by default | Objective is a policy choice | Equity and legal harm | Write and approve the objective |
| Traceability scope creep | "We will track everything" | Granularity drives cost | Programme stall | Define the minimum legal record |

## 13. Interview and discussion questions

1. Which sector has the highest cost of a stockout, and why?
2. How does shelf life change the inventory policy?
3. Why is a semiconductor lead time not a single number?
4. What makes humanitarian demand different from commercial demand?
5. How would you allocate a constrained pharmaceutical product?
6. What is the role of standards in pharmaceutical traceability?
7. How does fashion markdown interact with forecasting?
8. Why is automotive tier-2 visibility so hard?
9. Which sector would you choose for a domain project, and what decision would you improve?
10. What method from your current domain would fail in another sector, and why?
11. Which sector has the longest true lead time, and how does that change the planning cadence?
12. When is a forecast error a quality failure rather than a planning failure?
13. How would you value a day of downtime in automotive versus a day of stockout in pharma?
14. What would you refuse to optimize in a humanitarian allocation, and why?
15. Which sector's data would you trust least for a demand forecast, and what would you do about it?
16. What does a sector primer leave out that only an operator knows?
17. When would you refuse to reuse a method from another sector, even if it benchmarks better?
18. How does the buyer's cadence change the model you would build?

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
- European Medicines Agency, Guidelines on Good Distribution Practice of Medicinal Products for Human Use (2013/C 343/01).
- Codex Alimentarius, General Principles of Food Hygiene (CXC 1-1969).
- ISO 22301, Business continuity management systems.
- UN OCHA, Humanitarian Response Plan guidance and the 5W reporting standard.

> **Next:** [Module 30](30_simulation_optimization_production_labs.md) turns sector constraints into simulation and production labs.
