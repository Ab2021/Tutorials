# Module 39 — Mining, Metals, and Forestry Supply Chains

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Manufacturing and Commodities tracks
> **Prerequisites:** Modules 11, 12, 34, 38
> **Tools:** Python, linear programming, mine and rail simulation, GIS
> **Industry question:** "The ore grade moves, the train is late, and the price falls — which one do you plan around?"

---

## 1. Why industry cares

Mining, metals, and forestry are the front end of most physical supply chains. They share a structure that is unlike manufacturing:

- **The resource is fixed and variable in quality.** You cannot choose your orebody or your forest; you can only decide what to extract, when, and how to blend it.
- **Logistics is heavy and capital-intensive.** Pit-to-port rail, conveyors, slurry pipelines, ports, and bulk vessels dominate cost.
- **Price is exogenous and cyclical.** A mine cannot set the price of copper; it can only manage cost, grade, and volume.
- **Social and environmental obligations are central.** Land access, community consent, water, tailings, and reforestation determine whether a project can operate.

The analytics problem is **grade, volume, and logistics planning under geological, market, and social uncertainty**.

## 2. Learning outcomes

By the end of this module, you can:

- Map a mining or forestry supply chain from resource to customer.
- Model ore grade, recovery, and blending to meet a product specification.
- Plan pit-to-port logistics with rail, conveyor, port, and stockpile constraints.
- Evaluate capital and operating trade-offs in a resource project.
- Manage commodity price exposure through offtake and hedging.
- Incorporate social, environmental, and closure obligations into supply decisions.
- Plan forestry harvest, transport, and certification flows.
- Measure the carbon and water intensity of extraction and processing.
- Assess supply risk from grade, weather, labour, and permitting.
- Build a decision model that survives a price cycle.

## 3. Mental model

```text
   RESOURCE ──> EXTRACTION ──> PROCESSING ──> TRANSPORT ──> PORT ──> CUSTOMER
   (orebody,     (pit, mine,     (crusher,      (rail,        (stockpile,
    forest)       harvest)        smelter,       conveyor,     shiploader)
                                  mill, pulp)    pipeline)
        │             │                │              │
     grade and     recovery         product spec    cycle time
     reserves      and yield        and blend       and cost
```

Two constraints dominate every plan: the **resource** (what the ground or forest contains) and the **logistics chain** (what can be moved at what cost). Neither can be changed quickly.

### 3.1 The mining value chain

| Stage | Output | Key metric |
|---|---|---|
| Resource definition | Reserve and grade | Tonnes, grade, confidence |
| Extraction | Run-of-mine ore | Tonnes, dilution, recovery |
| Processing | Concentrate or metal | Recovery, grade, throughput |
| Transport | Delivered product | Cycle time, cost per tonne |
| Port and shipping | Loaded cargo | Demurrage, quality |
| Closure | Rehabilitated land | Compliance and cost |

## 4. Core concepts

### 4.1 Reserves, grade, and recovery

A reserve is an estimate, not a fact. It has confidence classes and a cut-off grade below which material is waste. The planning chain:

\[
\text{Contained metal} = \text{tonnes} \times \text{grade}
\]

\[
\text{Recovered metal} = \text{contained metal} \times \text{recovery}
\]

A small change in cut-off grade changes the mine plan, the waste volume, the mill feed, and the project life. Grade is the most consequential variable in the chain.

### 4.2 Blending and product specification

Customers buy a specification, not an orebody. Blending combines sources to meet grade, impurity, moisture, and hardness limits:

\[
\sum_i g_i x_i \geq G_{\min}, \qquad \sum_i x_i = 1
\]

with additional constraints for impurities (sulphur, silica, alumina) and cost. This is a linear program, often with stockpile and shipment constraints.

### 4.3 Pit-to-port logistics

Bulk logistics is a cycle-time problem:

\[
\text{Cycle time} = \text{loading} + \text{travel loaded} + \text{unloading} + \text{return empty} + \text{waiting}
\]

\[
\text{Capacity per day} = \frac{\text{payload} \times 24}{\text{cycle time}}
\]

For rail, the constraints are train paths, loading capacity, and port discharge. For conveyors, it is throughput and availability. For ports, it is berth, stockyard, and shiploader capacity.

### 4.4 Stockpiles and working capital

Stockpiles serve three purposes:

- **Buffer** between extraction and transport.
- **Blend** to meet specification.
- **Speculation** on price, which is usually a mistake for an operator.

A stockpile has a cost: capital, handling, degradation, and environmental risk. It is inventory, and it should be justified like inventory.

### 4.5 Commodity price and offtake

Miners often sell through **offtake contracts** with smelters or traders. Terms may include:

- Benchmark pricing with a quotation period.
- Treatment and refining charges (TC/RC).
- Price participation.
- Volume commitments and penalties.

The analytics question is **realised price**, not headline price. A quotation period, a treatment charge, and a quality penalty can move realised revenue by more than the forecast.

### 4.6 Forestry supply chains

Forestry adds biological and certification constraints:

- Harvest scheduling by stand, species, and age.
- Transport from forest to mill, often seasonal.
- Sawmill or pulp mill yield by log grade.
- Certification (FSC, PEFC) and chain-of-custody.
- Reforestation and biodiversity obligations.

The planning problem is a combination of harvest scheduling, grade mix, and transport.

### 4.7 Social licence and environment

A resource project needs **social licence**: community consent, land access, benefit sharing, and grievance mechanisms. Analytics supports:

- Stakeholder mapping and impact assessment.
- Water and tailings risk.
- Closure and rehabilitation costing.
- Local employment and procurement.

A technically optimal plan that ignores social licence is not feasible.

### 4.8 Worked example — grade blending

A copper concentrator blends two ore sources. Source A grades 1.2% copper at a processing cost of $38 per tonne; source B grades 0.8% at $30 per tonne. The mill needs a minimum feed grade of 1.0%.

Let x be the share of A:

\[
1.2x + 0.8(1-x) = 1.0 \Rightarrow 0.8 + 0.4x = 1.0 \Rightarrow x = 0.5
\]

The minimum-cost blend is 50/50 at $34 per tonne. If the price of copper falls, the cut-off grade rises and source B may become waste. The blend is not a fixed recipe; it moves with price, cost, and specification.

### 4.9 Worked example — rail cycle capacity

A train carries 100 wagons of 100 tonnes each, so payload is 10,000 tonnes. The route is 240 km, trains average 60 km/h, loading takes 8 hours, unloading 6 hours, and turnaround 4 hours.

\[
\text{Travel} = 2 \times 240 / 60 = 8 \text{ hours}
\]

\[
\text{Cycle} = 8 + 8 + 6 + 4 = 26 \text{ hours}
\]

\[
\text{Capacity} = 10{,}000 \times 24 / 26 \approx 9{,}231 \text{ tonnes per day per train}
\]

If the mine produces 30,000 tonnes per day, it needs at least four trains, plus redundancy for maintenance. A one-hour delay per cycle reduces capacity by about 4%, which is why dwell time at loading and unloading matters as much as line speed.

### 4.10 Worked example — stockpile sizing

A port stockpile must cover the gap between mine production and vessel loading. The mine delivers 30,000 tonnes per day, and a vessel of 150,000 tonnes loads every six days, drawing 25,000 tonnes per day on average.

If the vessel schedule is reliable, the stockpile needs only a few days of cover. If a vessel is delayed by three days, the mine keeps delivering 90,000 tonnes, which must fit somewhere. At a live capacity of 400,000 tonnes, a three-day delay uses 22.5% of the pile; a two-week delay uses more than the capacity and forces the mine to slow down.

The analytics question is the **distribution of vessel delay**, not the average. A stockpile sized to the average will overflow or starve on most days. Size it to a service level, and price the capital and handling cost against the cost of a mine slowdown or a demurrage event.

### 4.11 Reading a resource plan honestly

A resource plan is a chain of estimates, each with a confidence interval: reserve tonnage, grade, recovery, throughput, ramp-up, and price. The common error is to treat each estimate as a point and then multiply them, producing a plan with no uncertainty at all.

A better practice is to carry the uncertainty through:

- Use low, base, and high cases for grade and recovery.
- Model ramp-up separately from steady state.
- Test the plan at two price levels.
- State the reserve confidence class next to every number.

The output is not a single production forecast. It is a range, with the assumptions that drive the range. That is what a board can actually govern.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Drill and assay | Exploration database | Resource and grade model |
| Mine plan | Mine planning software | Extraction schedule |
| Production | SCADA / MES | Tonnes, grade, recovery |
| Stockpile | Survey / weighbridge | Inventory and blend |
| Rail and port | TMS / port system | Cycle time and demurrage |
| Assay and quality | Laboratory | Specification compliance |
| Price | Market feed | Valuation and cut-off |
| Weather | External feed | Access and harvest |
| Community and permits | Stakeholder system | Social licence |

### 5.2 Standards

- **JORC Code** (Australia), **NI 43-101** (Canada), and **SAMREC** for resource and reserve reporting.
- **ISO 14001** environmental management; **ISO 45001** occupational health and safety.
- **ICMM** mining principles and performance expectations.
- **Global Industry Standard on Tailings Management**.
- **FSC** and **PEFC** for forest certification and chain of custody.
- **ISO 55000** asset management for mobile and fixed plant.
- **MARPOL** and **IMSBC** for bulk shipping.

### 5.3 Workflow

1. Define resource and reserve.
2. Plan extraction and cut-off grade.
3. Blend to specification.
4. Process and recover.
5. Move to port by rail, conveyor, or road.
6. Load and ship.
7. Rehabilitate and close.
8. Report grade, volume, safety, and environment.

## 6. Industry case studies

### 6.1 Iron ore grade and blend

**Context:** Large iron ore producers blend ores to meet steelmaker specifications.
**Intervention:** Mine planning and blend optimisation across pits and stockpiles.
**Evidence:** Company disclosures and industry analysis (grade C/D).
**Outcome:** Blend control is a core margin lever, especially when impurities bind.
**What transferred:** the product is a specification, and blending is a supply chain decision.
**What did not:** specific grades and contract terms.

### 6.2 Copper price cycle and cut-off grade

**Context:** Copper prices fell and mines re-optimised.
**Intervention:** Higher cut-off grades, lower throughput, and cost reduction.
**Evidence:** Industry reporting and academic mine planning literature (grade B/C).
**Outcome:** Mines with high costs curtailed or closed; low-cost assets continued.
**What transferred:** the mine plan is price-contingent; scenario planning is mandatory.
**What did not:** specific price levels.

### 6.3 Forest certification and chain of custody

**Context:** Retail and construction customers required certified timber.
**Intervention:** FSC or PEFC certification and segregated supply chains.
**Evidence:** Certification body data and industry reports (grade B/C).
**Outcome:** Certification opened markets but required segregation and audit.
**What transferred:** certification is a data and traceability obligation, not a label.
**What did not:** specific market premiums.

### 6.4 A composite failure — the stockpile that was not ore

**Context:** A mine built a stockpile of low-grade material for later processing.
**Intervention:** None; the stockpile was treated as inventory.
**Evidence:** Composite teaching case.
**Outcome:** Oxidation and handling degraded the material, and the recovery was lower than planned.
**What transferred:** a stockpile is not free storage; it changes the product.
**What did not:** specific recovery losses.

## 7. Labs

### 7.1 Lab 39.1 — Grade blending

Solve a blend problem with grade, impurity, and cost constraints. Test the effect of a price change on the optimal blend.

**Deliverable:** LP model and sensitivity.

### 7.2 Lab 39.2 — Rail cycle capacity

Model a pit-to-port rail system with loading, travel, unloading, and maintenance. Find the fleet size for a production target.

**Deliverable:** Simulation and capacity plan.

### 7.3 Lab 39.3 — Cut-off grade

Given a grade-tonnage curve and cost structure, compute the cut-off grade at three price levels and the resulting mine life.

**Deliverable:** Analysis and scenario comparison.

### 7.4 Lab 39.4 — Forestry harvest schedule

Plan a harvest schedule across stands with age, species, transport distance, and mill demand.

**Deliverable:** Schedule and grade mix.

### 7.5 Lab 39.5 — Social and environmental risk

Build a stakeholder and environmental risk register for a synthetic project. Identify which risks can stop the supply chain.

**Deliverable:** Risk register and mitigation plan.

## 8. Assignment / project

Design the supply chain for a mining or forestry operation.

**Required components:**

1. Resource and reserve statement.
2. Mine or harvest plan.
3. Grade and blend model.
4. Processing and recovery.
5. Pit-to-port logistics.
6. Price and offtake analysis.
7. Social, environmental, and closure plan.
8. Metrics and governance.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Resource and grade | 20% | Describes orebody | Grade and recovery | Cut-off and confidence |
| Blend | 20% | Fixed recipe | LP blend | Multi-spec with price sensitivity |
| Logistics | 20% | Describes transport | Cycle-time model | Fleet and port capacity |
| Market | 15% | Uses headline price | Realised price | Offtake and quotation period |
| Social and environment | 15% | Mentions permits | Risk register | Licence-to-operate analysis |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Resource reporting:** JORC, NI 43-101, and SAMREC govern public statements.
- **Safety:** mining and forestry are high-hazard; safety obligations override schedule.
- **Environment:** water, tailings, biodiversity, and rehabilitation are regulated.
- **Community:** land rights, consent, and benefit sharing are material.
- **Forestry:** certification and chain of custody are market and regulatory requirements.
- **Corruption:** resource-rich jurisdictions carry high corruption risk.
- **Bulk shipping:** IMSBC and MARPOL rules apply to cargo and emissions.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Grade model overconfidence | Reconciliation gap | Resource estimate error | Volume and cost miss | Reconciliation and update |
| Fixed blend | Spec violations | Feed changes | Customer rejection | Dynamic blend |
| Rail bottleneck | Trains wait | Cycle time underestimated | Production backlog | Fleet and path planning |
| Stockpile degradation | Recovery falls | Oxidation or handling | Value loss | Stockpile management |
| Headline price planning | Margin miss | Realised price differs | Budget shortfall | Offtake analysis |
| Social licence ignored | Protests or permits | Community opposition | Stoppage | Engagement and consent |
| Tailings risk | Inspection finding | Design or operation | Safety and licence | Independent review |
| Certification gap | Audit finding | Chain of custody broken | Market loss | Segregation and records |

## 12. Interview and discussion questions

1. Why is grade the most important variable in a mine? *(Hint: it drives volume, cost, and product spec.)*
2. What is cut-off grade, and how does price affect it? *(Hint: below cut-off is waste.)*
3. How do you blend to a specification? *(Hint: LP with grade and impurity constraints.)*
4. What determines pit-to-port capacity? *(Hint: cycle time and bottleneck.)*
5. Why is a stockpile not free inventory? *(Hint: capital, degradation, handling.)*
6. What is realised price versus headline price? *(Hint: quotation period, TC/RC, penalties.)*
7. How does forestry differ from mining? *(Hint: biological growth, harvest scheduling, certification.)*
8. What is social licence, and why does it affect supply? *(Hint: consent and access.)*
9. How do you plan a mine through a price cycle? *(Hint: scenarios and cut-off flexibility.)*
10. What are the main tailings risks? *(Hint: design, monitoring, governance.)*
11. How do you measure carbon in a mine's supply chain? *(Hint: diesel, electricity, transport.)*
12. What would you do if the grade reconciliation is 10% below plan? *(Hint: investigate, re-blend, revise.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Reserve | Economically extractable resource estimate |
| Grade | Concentration of metal in ore |
| Cut-off grade | Minimum grade that is economic |
| Recovery | Share of contained metal extracted |
| Dilution | Waste mixed into ore |
| Run-of-mine | Ore as extracted |
| Concentrate | Processed ore for smelting |
| Tailings | Waste from processing |
| Pit-to-port | Logistics from mine to export port |
| Cycle time | Time for a transport round trip |
| Offtake | Contract to sell future production |
| TC/RC | Treatment and refining charges |
| JORC | Australian resource reporting code |
| Social licence | Community acceptance to operate |
| Chain of custody | Traceable certified material flow |
| FSC / PEFC | Forest certification schemes |

## 14. Sources and further reading

### Standards and regulation

- JORC Code. https://www.jorc.org/
- NI 43-101. https://www.cim.org/
- SAMREC Code. https://www.samcode.co.za/
- ICMM Mining Principles. https://www.icmm.com/
- Global Industry Standard on Tailings Management. https://globaltailingsreview.org/
- FSC. https://fsc.org/ ; PEFC. https://www.pefc.org/
- IMSBC Code, IMO. https://www.imo.org/

### Reports and official sources

- US Geological Survey, Mineral Commodity Summaries. https://www.usgs.gov/
- IEA, critical minerals reports. https://www.iea.org/
- World Bank, commodity markets outlook. https://www.worldbank.org/
- FAO, forestry data. https://www.fao.org/

### Peer-reviewed and professional

- Newman, A. M., et al., mine planning and optimisation research.
- Osanloo, M., cut-off grade and mine design literature.
- Dyson, R. G., linear programming in blending.
- Gundersen, F., forestry supply chain research.

### Industry

- ICMM member reports. https://www.icmm.com/
- World Steel Association, steel and raw material data. https://worldsteel.org/

> **Next:** [Module 40](40_telecom_technology_hardware.md) closes the domain set with high-technology and network supply chains.
