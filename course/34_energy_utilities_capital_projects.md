# Module 34 — Energy, Utilities, and Capital Project Supply Chains

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Manufacturing and Logistics tracks
> **Prerequisites:** Modules 10, 11, 14, 15, 31
> **Tools:** Python, OR-Tools, P6/MSP schedule extracts, synthetic CMMS data
> **Industry question:** "When a transformer takes 80 weeks and an outage window is 14 days, how do you plan?"

---

## 1. Why industry cares

Energy and utilities are the most capital-intensive supply chains in the economy, and capital projects are the most supply-chain-exposed form of construction. Three features dominate:

- **Uptime is the product.** A utility sells availability. A failed transformer, pump, or turbine is lost revenue and, for regulated assets, a reliability penalty.
- **Lead times exceed planning horizons.** High-voltage transformers, switchgear, wind turbine components, and specialised valves can take 12–36 months. You cannot respond to a demand signal; you must anticipate.
- **Projects are one-off networks.** An EPC project buys engineered equipment, bulk materials, and construction services against a schedule, not a repeating demand pattern.

The analytics problem is **planning under long lead times, high criticality, and schedule uncertainty**, with safety and regulatory obligations that override cost.

## 2. Learning outcomes

By the end of this module, you can:

- Map the energy value chain and identify the buying decision at each segment.
- Build a critical-spares strategy for asset-intensive operations using criticality, failure, and lead time.
- Plan materials and kitting for an outage or turnaround against a fixed window.
- Analyse long-lead equipment procurement and its effect on project schedule risk.
- Forecast bulk material demand from engineering takeoffs and manage wastage and reconciliation.
- Read earned value and critical-path data and connect supply delays to schedule float.
- Explain commodity price exposure and the contract structures used to manage it.
- Design a storm-response or resilience stock strategy for a utility.
- Choose between EPC, EPCM, lump-sum, and reimbursable contract models.
- Assess safety, environmental, and security obligations in energy supply chains.

## 3. Mental model

```text
   RESOURCE ──> GENERATION / PROCESSING ──> TRANSMISSION ──> DISTRIBUTION ──> CUSTOMER
   (fuel, wind,   (plant, refinery,         (grid, pipeline)  (network,         (load)
    solar, gas)    wind farm)                                  substation)
        │                │                      │                  │
        └── MRO, spares, outage services, capital equipment, construction ──┘

   CAPITAL PROJECT
   engineering ──> procurement ──> fabrication ──> logistics ──> construction ──> commissioning
        │               │              │              │               │
        └── takeoff ────┴── long lead ─┴── modules ────┴── laydown ─────┘
```

The same physical asset appears twice: as an operating asset requiring MRO, and as a project deliverable requiring procurement and construction. Both must be planned on one schedule.

### 3.1 The two demand types

| Type | Driver | Forecast method | Supply response |
|---|---|---|---|
| Operating MRO | Failure, condition, consumption | Failure models, usage rates | Stock, contracts, lead time |
| Outage / turnaround | Planned scope | Scope list, history | Kitting, staging, contractor |
| Capital project | Engineering takeoff | BOM and schedule | Purchase orders, fabrication |
| Emergency | Weather, failure | Scenario and pre-positioning | Mutual aid, emergency contracts |

Treating MRO demand as a stable time series is the most common modelling error in the sector.

## 4. Core concepts

### 4.1 Asset-criticality and spares

A spare part is not critical because it is expensive. It is critical because its absence stops a critical function. The inputs:

\[
\text{Criticality} = f(\text{consequence of failure},\ \text{failure rate},\ \text{lead time},\ \text{substitutability})
\]

| Consequence | Example | Spare strategy |
|---|---|---|
| Safety / environmental | Wellhead valve, relief valve | Redundancy, certified stock |
| Total plant outage | Main transformer, gas turbine | Hold or contract, long lead |
| Partial derate | Pump, fan | Stock or repair pool |
| Deferrable | Office equipment | Buy on demand |

For long-lead critical items, the decision is often **hold, share, or contract** rather than stock versus no stock. Utilities increasingly share spares across a fleet or with peers.

### 4.2 The energy value chain and what each segment buys

| Segment | Buys | Supply characteristic |
|---|---|---|
| Upstream oil and gas | OCTG, drilling services, valves, chemicals | Remote, safety-critical, project-driven |
| Refining and chemicals | Catalysts, rotating equipment, turnaround services | Continuous process, outage windows |
| Power generation | Fuel, turbines, transformers, MRO | Long asset life, high criticality |
| Transmission and distribution | Transformers, cable, poles, switchgear | Regulated, long lead, storm exposure |
| Renewables | Turbines, blades, inverters, O&M spares | Fast build, long O&M tail |
| Retail and trading | Metering, billing, customer service | Low asset, high data |

Each segment has a different cost driver. A single "energy supply chain" strategy is not meaningful.

### 4.3 Outage and turnaround planning

An outage is a fixed-window project inside an operating plant. The supply chain must deliver every part, tool, and contractor to the right place before the window opens.

| Step | Requirement |
|---|---|
| Scope freeze | Locked work list and material takeoff |
| Material readiness | Every line item received, inspected, kitted |
| Kitting | Bundled by work package, not by PO |
| Staging | Laydown area, sequence, preservation |
| Contractor mobilisation | Access, permits, safety induction |
| Execution | Daily schedule and material shortage list |
| Close-out | Reconciliation, surplus return, warranty |

The metric is **material readiness at window start**, not on-time delivery. A part that arrives on day 3 of a 14-day outage is late in practice.

### 4.4 Long-lead equipment and project procurement

Long-lead items (LLIs) set the project critical path:

- High-voltage transformers and switchgear.
- Gas and steam turbines.
- Specialty valves, compressors, and reactors.
- Wind turbine nacelles, blades, and towers.
- Custom modules and skids.

The analytics questions are:

1. What is the current lead time and its distribution?
2. When must the PO be placed to protect the schedule?
3. What is the cost of float versus the cost of expediting?
4. What is the risk of a supplier delay, and what is the mitigation?

A useful rule: **procurement is on the critical path when lead time plus fabrication exceeds the engineering float available.**

### 4.5 Bulk materials and construction logistics

Bulk materials — pipe, cable, steel, concrete, insulation — are high-volume, low-value, and easy to mismanage. Controls:

- **Takeoff:** engineering bill of materials by area and system.
- **Wastage allowance:** explicit percentage by material, not hidden.
- **Reconciliation:** issued versus installed versus surplus.
- **Laydown and preservation:** space, weather, security, corrosion.
- **Module strategy:** prefabricate offsite to reduce site labour and congestion.

The common failure is a material shortage caused not by supply but by a takeoff error or a lost shipment in the laydown yard.

### 4.6 Schedule analytics

Earned value connects supply to schedule:

\[
\text{SV} = EV - PV, \qquad \text{SPI} = EV / PV
\]

\[
\text{CV} = EV - AC, \qquad \text{CPI} = EV / AC
\]

For supply, the useful measures are:

- **Material float:** days between required-on-site date and scheduled installation.
- **Procurement SPI:** progress of purchase orders against plan.
- **Critical-path exposure:** number of critical activities with negative float.
- **Expediting cost:** premium paid to recover float.

A project can show a healthy SPI while a single transformer threatens the completion date. Aggregate indices hide single-point failures.

### 4.7 Commodity price exposure

Fuel, metals, and chemicals are priced on indices. Contract structures:

| Structure | Who bears price risk | Use |
|---|---|---|
| Fixed price | Supplier | Short projects, stable inputs |
| Index-linked | Buyer | Long projects, volatile inputs |
| Cost-plus | Buyer | Scope uncertainty |
| Hedge | Buyer or supplier | Financial risk transfer |

Analytics must separate **volume risk** from **price risk**. A forecast that is right on volume but wrong on price still produces a bad budget.

### 4.8 Renewables and grid equipment

Renewables add a new supply profile:

- Fast construction, long operational life.
- Turbine and blade lead times measured in quarters.
- O&M spares concentrated in a few critical components.
- Grid connection equipment is the binding constraint in many markets.

The spares question shifts from "how many" to "which components justify a spare at all", because a blade or gearbox may be uneconomical to stock.

### 4.9 Storm response and resilience

A utility's emergency supply plan is a pre-positioning problem:

- Forecast the event footprint and likely damage.
- Pre-stage poles, wire, transformers, and crews.
- Use mutual aid agreements across regions.
- Prioritise restoration by critical customers and safety.

The trade-off is between **pre-positioning cost** and **restoration time**. Both are measurable; the choice is a policy decision.

### 4.10 Contract models

| Model | Scope | Risk allocation | Supply chain implication |
|---|---|---|---|
| EPC / turnkey | Design, procure, construct | Contractor | Fewer interfaces, higher price |
| EPCM | Design and manage | Owner | Owner buys equipment, more control |
| Lump sum | Fixed scope | Contractor | Change control critical |
| Reimbursable | Actual cost | Owner | Cost visibility critical |
| Framework | Repeat work | Shared | Call-off and rate management |

The contract determines who owns long-lead procurement risk, and therefore who needs the analytics.

### 4.11 Worked example — long-lead transformer

A substation project needs a 230 kV transformer with an 80-week lead time and an installation float of 12 weeks. The project schedule has 60 weeks of engineering and civil work before the transformer is needed.

Required-on-site week is week 72. The PO must be placed by week 72 − 80 = week −8, which is before the project starts. If procurement is not authorised until week 4, the transformer arrives at week 84 and the project slips by at least 12 weeks unless the supplier can expedite.

If expediting costs €600,000 and a 12-week delay costs €250,000 per week in lost revenue and overhead, the delay cost is €3,000,000. Expediting is worth it if it recovers at least 3 weeks. The real question is why the PO was not placed before authorisation — usually a governance gap, not a supplier problem.

### 4.12 Worked example — outage material readiness

A 14-day outage has 1,200 material line items. At window start, 1,140 are available; 60 are missing. The missing items affect 18 of 240 work packages, and 4 of those are on the critical path.

If critical-path work packages average 2.5 days of delay each and the outage cannot extend beyond the window, the plant returns to service late or the work is deferred. Deferring a safety-critical item is not acceptable.

The lesson: track readiness by **work package and critical path**, not by line-item percentage. 95% line readiness can still mean 100% critical-path failure.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | System | Use |
|---|---|---|
| Asset register | EAM / ERP | Criticality and hierarchy |
| Work orders | CMMS | Failure and maintenance history |
| Spare parts | ERP / EAM | Stock, lead time, price |
| Condition data | SCADA / IoT | Predictive maintenance |
| Purchase orders | ERP | Procurement and delivery |
| Project schedule | P6 / MSP | Critical path and float |
| Engineering BOM | PLM / engineering | Takeoff and materials |
| Laydown and logistics | WMS / site system | Receipt and issue |
| Weather | External feed | Storm pre-positioning |
| Contract data | CLM / ERP | Price and risk allocation |

### 5.2 Standards

- **ISO 55000** asset management.
- **ISO 14224** reliability data collection for petroleum, petrochemical, and gas.
- **IEC 60076** power transformers; **IEC 61850** substation automation.
- **NERC CIP** critical infrastructure protection (US).
- **API** standards for oil and gas equipment.
- **FIDIC** contract forms for construction.
- **ANSI/EIA-748** earned value management.
- **OSHA** and local safety regulation; **ISO 45001**.
- Environmental permits and impact assessment.

### 5.3 Workflow

1. Register assets and criticality.
2. Forecast MRO and outage demand.
3. Set spares policy by criticality and lead time.
4. Plan outage scope and materials.
5. Procure long-lead equipment against schedule.
6. Manage construction logistics and laydown.
7. Monitor schedule, cost, and material readiness.
8. Respond to emergencies and restore service.
9. Review reliability and feed learning back.

## 6. Industry case studies

### 6.1 Winter Storm Uri and the Texas grid (2021)

**Context:** Extreme cold caused generation outages and widespread power loss in Texas.
**Intervention:** Emergency response, load shedding, and subsequent regulatory review.
**Evidence:** Official incident reports and regulatory filings (grade B).
**Outcome:** The event exposed fuel-supply interdependence, weatherisation gaps, and the limits of emergency planning.
**What transferred:** resilience analysis must model coupled dependencies, not just individual asset failure.
**What did not:** specific market rules and settlement outcomes.

### 6.2 Transformer lead-time inflation

**Context:** Grid investment and renewable connection increased demand for high-voltage transformers.
**Intervention:** Utilities placed earlier orders, accepted longer lead times, and explored standardisation.
**Evidence:** Industry surveys and trade reporting (grade C/D); lead-time figures vary by voltage class and region.
**Outcome:** Lead times extended, prices rose, and some projects were deferred.
**What transferred:** for long-lead equipment, procurement timing is a schedule decision, not a purchasing decision.
**What did not:** the specific lead times, which change.

### 6.3 Wind turbine O&M spares

**Context:** Operators need high availability across distributed wind farms.
**Intervention:** Condition monitoring, central spare depots, and service contracts with OEMs.
**Evidence:** Industry and academic operations research (grade B/C).
**Outcome:** Predictive maintenance and pooled spares reduced downtime where failure modes were well understood.
**What transferred:** distributed assets need pooled, not local, spares strategies.
**What did not:** specific availability gains, which depend on fleet and site.

### 6.4 A composite failure — the bulk material that was never ordered

**Context:** A construction project relied on a takeoff from an early engineering revision.
**Intervention:** No reconciliation between revisions.
**Evidence:** Composite teaching case.
**Outcome:** A cable type was omitted, discovered during installation, causing a 6-week delay.
**What transferred:** bulk material control is a revision-control problem as much as a purchasing problem.
**What did not:** the specific quantities and costs.

## 7. Labs

### Lab 34.1 — Critical spares classification

Given an asset register and spare parts list, classify spares by criticality and lead time, and recommend hold, share, or contract.

**Deliverable:** Criticality matrix and spares policy.

### Lab 34.2 — Outage kitting

Build a kitting plan for an outage with 200 work packages, limited staging space, and a fixed window. Measure readiness by work package.

**Deliverable:** Kitting plan and readiness dashboard.

### Lab 34.3 — Long-lead procurement risk

Model a project schedule with three long-lead items and stochastic lead times. Estimate the probability of schedule slip and the value of expediting.

**Deliverable:** Schedule risk model and recommendation.

### Lab 34.4 — Earned value and material float

Given a project schedule and cost data, compute SPI, CPI, and material float by critical activity. Identify the supply-driven risk.

**Deliverable:** EV analysis and risk register.

### Lab 34.5 — Storm pre-positioning

Design a pre-positioning plan for a hurricane scenario across five depots. Compare restoration time and cost under three strategies.

**Deliverable:** Scenario analysis and policy recommendation.

## 8. Assignment / project

Design the supply chain plan for one of: a grid reinforcement project, a plant turnaround, or a renewable build.

**Required components:**

1. Asset and demand characterisation.
2. Spares and MRO policy.
3. Outage or construction material plan.
4. Long-lead procurement and schedule analysis.
5. Contract and risk allocation.
6. Safety and regulatory assessment.
7. Resilience and emergency plan.
8. Metrics and governance.
9. Model, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Asset and demand | 20% | Describes assets | Criticality and demand | Failure and consequence modelling |
| Spares policy | 20% | ABC classification | Criticality and lead time | Hold/share/contract economics |
| Outage or project plan | 20% | Material list | Kitting and schedule | Critical-path readiness |
| Schedule and cost | 15% | Reports EV | Interprets EV and float | Links supply to schedule risk |
| Contract and risk | 15% | Describes model | Matches model to scope | Risk allocation and incentives |
| Safety and resilience | 10% | Mentions safety | Plans for emergencies | Coupled-dependency analysis |

## 10. Regulatory, ethical, and commercial lens

- **Safety:** energy assets can injure workers and communities; supply decisions cannot bypass safety cases.
- **Environmental:** permits, emissions, and waste rules constrain procurement and disposal.
- **Security:** critical infrastructure protection rules (for example NERC CIP) govern data and access.
- **Export control and sanctions:** specialised equipment may be controlled; screen suppliers and destinations.
- **Regulated economics:** utility rates and capital recovery affect what investment is allowed.
- **Community impact:** siting, construction traffic, and land use require engagement.
- **Corruption:** large projects are high-risk for bribery and conflicts of interest.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Spares by price | Cheap parts stocked | No criticality view | Critical failure unserved | Criticality policy |
| MRO treated as stable demand | Forecast error high | Failure-driven demand | Over/understock | Failure modelling |
| Outage scope creep | Late work orders | Scope not frozen | Material shortage | Scope freeze gate |
| Line-item readiness | 95% reported | Critical path ignored | Outage overrun | Work-package readiness |
| Late PO | Governance delay | Lead time exceeds float | Schedule slip | Early procurement authority |
| Takeoff error | Revision mismatch | No reconciliation | Missing material | Revision control |
| Laydown loss | Cannot find material | Poor site logistics | Delay and reorder | WMS and preservation |
| Aggregate SPI | Healthy index | Single-point failure | Surprise delay | Critical-path monitoring |
| Price/volume confusion | Budget miss | Commodity exposure | Cost overrun | Separate hedges |
| Emergency improvisation | No pre-positioning | Storm damage | Slow restoration | Scenario planning |

## 12. Interview and discussion questions

1. Why is MRO demand different from retail demand? *(Hint: failure-driven, intermittent, criticality.)*
2. How do you decide whether to stock a critical spare? *(Hint: consequence, failure rate, lead time, sharing.)*
3. What does material readiness mean at outage start? *(Hint: by work package and critical path.)*
4. When does procurement become the critical path? *(Hint: lead time plus fabrication versus float.)*
5. How do you manage bulk material wastage? *(Hint: takeoff, allowance, reconciliation.)*
6. Why can a healthy SPI hide a failing project? *(Hint: aggregate versus single-point risk.)*
7. How do you separate volume risk from price risk? *(Hint: index-linked contracts versus demand forecast.)*
8. What is the spares strategy for distributed wind assets? *(Hint: pooled depots and condition monitoring.)*
9. How would you pre-position for a hurricane? *(Hint: footprint, depots, mutual aid, critical customers.)*
10. What changes under EPC versus EPCM? *(Hint: who buys long-lead equipment and owns risk.)*
11. Which standards matter for reliability data? *(Hint: ISO 14224, ISO 55000.)*
12. What is the biggest supply risk in a capital project? *(Hint: long-lead equipment and engineering revisions.)*

## 13. Key terms

| Term | Definition |
|---|---|
| MRO | Maintenance, repair, and operations |
| Outage / turnaround | Planned shutdown for maintenance or modification |
| Long-lead item | Equipment with a lead time that constrains schedule |
| Critical path | Sequence of activities that determines project duration |
| Float | Slack before an activity delays the project |
| Earned value | Budgeted value of work performed |
| SPI / CPI | Schedule / cost performance index |
| Takeoff | Quantity extraction from engineering documents |
| Laydown yard | Site storage area for materials and modules |
| Module | Prefabricated unit assembled offsite |
| EPC | Engineering, procurement, and construction contract |
| EPCM | Engineering, procurement, and construction management |
| Mutual aid | Sharing crews and materials across utilities |
| Condition monitoring | Using sensor data to predict failure |
| ISO 55000 | Asset management standard |
| ISO 14224 | Reliability data standard |
| NERC CIP | Critical infrastructure protection standards |
| FIDIC | International construction contract forms |

## 14. Sources and further reading

### Standards and regulation

- ISO 55000, Asset management. https://www.iso.org/standard/55088.html
- ISO 14224, Petroleum, petrochemical and natural gas industries — Collection and exchange of reliability and maintenance data. https://www.iso.org/standard/64076.html
- NERC Critical Infrastructure Protection. https://www.nerc.com/pa/Stand/Pages/CIPStandards.aspx
- FIDIC contract forms. https://fidic.org/
- ANSI/EIA-748 Earned Value Management Systems.
- IEC 60076, Power transformers. https://www.iec.ch/

### Reports and official sources

- FERC and NERC, February 2021 Cold Weather Outages in Texas and the South Central United States. https://www.ferc.gov/
- US Department of Energy, grid and transformer reports. https://www.energy.gov/
- IEA, World Energy Investment. https://www.iea.org/reports/world-energy-investment

### Peer-reviewed and professional

- Arts, J., et al., maintenance and spares optimisation research.
- Van Horenbeek, A., et al., "Maintenance optimization models and criteria."
- Project Management Institute, *A Guide to the Project Management Body of Knowledge*.
- Fleming, Q., Koppelman, J., *Earned Value Project Management*.

### Industry

- EPRI research on grid assets and maintenance. https://www.epri.com/
- WindEurope and GWEC market reports. https://windeurope.org/ https://gwec.net/

> **Next:** [Module 35](35_public_sector_defence_humanitarian.md) moves to public-sector supply chains, where the objective is mission assurance and the constraints are political and legal.
