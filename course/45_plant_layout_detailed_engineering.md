# Module 45 — Plant Layout, Facility Design, and Detailed Engineering Interfaces

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Operations, Facilities, and Engineering tracks
> **Prerequisites:** Modules 10, 15, 18, 20
> **Tools:** Python, layout and flow models, sketching tools, P&ID and MTO data sets
> **Industry question:** "The forecast and the network say build here — but can the building actually do it?"

---

## 1. Why industry cares

Forecasts and network models decide **how much** and **where**. Facility design decides whether the operation can physically execute at that volume: how many pallets fit, how many trucks can dock, how far a picker walks, how a pipe is routed, and whether the aisle is wide enough for the truck the business plans to buy.

Design errors are expensive because they are **cast in concrete**:

- An aisle is 0.4 metres too narrow for a new reach truck.
- A dock has too few doors for the peak, and trailers queue onto the road.
- A cold store is short of freezer capacity in the second year.
- A tank on a P&ID is tagged in two systems, and the maintenance bill of materials is wrong.
- A fire egress route is blocked by a pick line, and the site fails inspection.

The analytics problem is **translating flow, volume, and service requirements into space, equipment, and engineering data — and testing that translation before construction**.

## 2. Learning outcomes

By the end of this module, you can:

- Classify facility types and the design logic of each.
- Apply systematic layout planning to a warehouse or plant.
- Build from–to charts and relationship diagrams to reduce flow cost.
- Calculate storage, dock, aisle, and staging requirements from demand.
- Select material handling equipment against throughput and space constraints.
- Validate a layout with flow, capacity, and discrete-event analysis.
- Read a P&ID, isometric, and material take-off, and understand revision control.
- Manage the interface between supply chain design and detailed engineering.
- Design for safety, fire, egress, and ergonomics.
- Recognise the lifecycle cost of a facility decision.

## 3. Mental model

```text
   DEMAND ──> FLOW ──> SPACE ──> EQUIPMENT ──> LAYOUT ──> ENGINEERING DATA
     │          │        │           │            │              │
   volume,    from-to   area and   handling     blocks,        P&ID, MTO,
   mix,       and      volume     and storage   aisles,        isometrics,
   growth     frequency  calc      selection    docks          revisions

   VALIDATE: capacity ──> flow ──> simulation ──> safety ──> cost
```

A layout is a **spatial answer to a flow question**. If the flow is not quantified first, the layout is a drawing, not a design.

### 3.1 Facility types

| Type | Primary flow | Design driver |
|---|---|---|
| Distribution centre | Receive, store, pick, ship | Throughput and cube |
| Cross-dock | Receive and ship | Dock doors and staging |
| Cold store | Temperature-controlled storage | Refrigeration and dwell |
| Manufacturing plant | Transform materials | Process sequence and utilities |
| Micro-fulfilment | High-density urban picking | Speed and footprint |
| Terminal / yard | Transhipment | Equipment and dwell |
| Service depot | Parts and returns | SKU breadth and response time |

## 4. Core concepts

### 4.1 Systematic layout planning

Muther's systematic layout planning (SLP) sequence:

1. **Flow of materials** and **relationships** between areas.
2. **Space requirements** and available space.
3. **Activity relationship chart** (A, E, I, O, U, X).
4. **Block layout** from the relationships.
5. **Detailed layout** with equipment and aisles.
6. **Evaluation** against flow, cost, and safety.

The relationship chart is a qualitative input beside the quantitative flow. A receiving area and a quality lab may have low volume but a strong relationship (A).

### 4.2 From–to charts and flow cost

A from–to chart tabulates the movement between areas. Multiplying each movement by distance and cost per unit-distance gives total flow cost:

\[
\text{Flow cost} = \sum_{i,j} f_{ij} \times d_{ij} \times c
\]

A layout that reduces the largest flows has the biggest effect. Reducing a small flow to zero is usually not worth a redesign.

### 4.3 Storage capacity

Storage requirement is positions, not square metres:

\[
\text{Positions} = \frac{\text{peak inventory}}{\text{units per position}}
\]

\[
\text{Floor positions} = \frac{\text{positions}}{\text{levels}}
\]

\[
\text{Area} = \text{floor positions} \times \text{footprint per position} \times \text{aisle factor}
\]

The aisle factor — typically 1.6 to 2.2 for pallet racking — accounts for aisles, cross-aisles, and columns. It is the number most often underestimated.

### 4.4 Dock and staging design

\[
\text{Door-hours required} = \text{trucks per day} \times \text{hours per appointment}
\]

\[
\text{Doors} = \frac{\text{door-hours required}}{\text{operating hours} \times \text{target utilisation}}
\]

Target utilisation below 100% is essential, because arrivals are variable. A dock designed to average demand will queue at peak.

### 4.5 Aisle and equipment geometry

Aisle width is set by the **truck and its lift height**, not by the pallet:

| Equipment | Typical aisle | Trade-off |
|---|---|---|
| Counterbalance | Wide | Flexible, fast, low density |
| Reach truck | Medium | Good balance |
| Very narrow aisle | Narrow | High density, slower, rail-guided |
| Automated shuttle | Very narrow | Highest density, high capex |

Storage density and throughput pull in opposite directions. Denser storage means slower access and more capital.

### 4.6 Material handling and automation

Selection criteria:

- Throughput (moves or lines per hour).
- Cube and weight.
- SKU profile and seasonality.
- Labour availability and cost.
- Uptime and maintenance.
- Flexibility for growth and change.

Automation payback depends on **utilisation**. A system designed for peak and run at half volume rarely pays back.

### 4.7 Detailed engineering interfaces

Once the layout is fixed, engineering data takes over:

| Artefact | Content | Interface risk |
|---|---|---|
| P&ID | Piping, instruments, control loops | Tag mismatch |
| Isometric | Pipe routing and spools | Clash and rework |
| MTO | Material take-off for procurement | Quantity error |
| Equipment list | Tags, duty, materials | Specification gap |
| Line list | Sizes, fluids, conditions | Design basis drift |
| Cause and effect | Interlocks and shutdowns | Safety gap |

The supply chain interfaces at three points: **procurement of engineered items**, **spares and maintenance bills of materials**, and **commissioning and ramp-up**.

### 4.8 Revision control

Engineering data changes constantly:

- A revision letter or number identifies the drawing state.
- Every tag has a revision history.
- Changes propagate to the MTO, procurement, and maintenance systems.
- A change not propagated becomes a wrong part on site.

The analytics problem is **reconciliation**: comparing the current drawing, the MTO, the purchase orders, and the installed equipment, and finding the differences.

### 4.9 Design for supply chain

Design decisions constrain the supply chain for decades:

- **Standardisation:** fewer part numbers, fewer spares.
- **Modularity:** pre-assembled units that ship and install faster.
- **Design for manufacture and assembly:** simpler joins, fewer fasteners.
- **Design for maintenance:** access and replaceability.
- **Design for transport:** module size against road, rail, and vessel limits.

A design that is optimal in the plant but cannot be transported is not optimal.

### 4.10 Safety, fire, and ergonomics

- **Fire:** compartmentation, sprinklers, smoke ventilation, and separation of hazards.
- **Egress:** travel distance, exit width, and unobstructed routes.
- **Dangerous goods:** segregation and bunding.
- **Ergonomics:** pick height, travel, lifting, and repetitive motion.
- **Traffic:** separation of pedestrians, forklifts, and trucks.

Safety requirements are constraints on the layout, not features added after it.

### 4.11 Validation with simulation

A layout should be tested before construction:

1. **Static capacity check:** does it fit, at peak?
2. **Flow check:** do the distances and equipment support the throughput?
3. **Discrete-event simulation:** queues, utilisation, and bottlenecks under variability.
4. **Sensitivity:** growth, seasonality, and SKU change.

Simulation does not prove a design; it exposes the assumptions that break it.

### 4.12 Worked example — storage and area

A distribution centre needs 30,000 pallet positions at peak. Racking is four levels high.

\[
\text{Floor positions} = 30{,}000 / 4 = 7{,}500
\]

Each position, including the rack frame, occupies 2.4 m² of floor. With an aisle factor of 1.8:

\[
\text{Area} = 7{,}500 \times 2.4 \times 1.8 = 32{,}400 \text{ m}^2
\]

At a 2,400 m² footprint per level in a single-storey building, that is about 13.5 building "blocks" of storage, before receiving, shipping, and offices. If the site is 25,000 m², the design does not fit and the options are higher racking, a mezzanine, or a second site.

### 4.13 Worked example — dock doors

A DC handles 120 trucks per day. Each appointment takes two hours including unloading and paperwork. The dock operates 12 hours per day, and the target door utilisation is 80%.

\[
\text{Door-hours} = 120 \times 2 = 240
\]

\[
\text{Doors} = \frac{240}{12 \times 0.80} = 25
\]

Twenty doors would run at 100% utilisation and queue whenever a truck is late. Twenty-five gives headroom, but the cost is dock face, doors, levellers, and lost storage area. The right number depends on the arrival distribution, not the average.

### 4.14 Worked example — revision reconciliation

A P&ID at revision C contains 340 tags. The MTO was issued at revision B, when there were 312 tags. Procurement has 318 line items; 47 of them reference tags that changed duty in revision C.

A naive comparison of counts suggests the MTO is "close enough" — 312 versus 340. A tag-level comparison shows 47 specification changes, of which 12 affect material selection and 3 affect pressure rating. Those 15 are the items that would fail commissioning or create a safety issue.

The analytics task is not counting; it is **matching tags across systems and classifying the differences by consequence**.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Demand and SKU profile | ERP / WMS | Volume and mix |
| Inventory and peak | WMS / planning | Storage positions |
| Order and line profile | OMS | Picking and docks |
| Truck arrivals | Yard / TMS | Dock design |
| Item dimensions | MDM | Cube and slotting |
| Equipment specs | Vendors | Aisle and throughput |
| Site and survey | CAD / GIS | Available space |
| P&ID and isometrics | Engineering system | Tags and routing |
| MTO and BOM | ERP / PLM | Procurement |
| Safety requirements | Codes and insurer | Constraints |

### 5.2 Standards

- **ISO 9001** and **ISO 45001** for quality and safety management.
- **NFPA 13** (sprinklers), **NFPA 101** (life safety), and local building codes.
- **EN 15635** and **FEM** guidance for racking.
- **IEC 61511** and **IEC 61508** for safety instrumented systems.
- **ISO 10628** for P&ID and process diagrams.
- **ISO 15926** and **ISO 10303 (STEP)** for engineering data exchange.
- **ASME B31.3** and equivalent piping codes.
- **ATEX** and hazardous area classification.
- **ISO 14001** environmental management.

### 5.3 Workflow

1. Quantify flow and volume.
2. Derive space and equipment requirements.
3. Produce block layout and relationship chart.
4. Detail aisles, docks, and staging.
5. Validate with capacity, flow, and simulation.
6. Check safety and egress.
7. Hand over to detailed engineering.
8. Reconcile tags, MTO, and procurement.
9. Commission, ramp up, and review.

## 6. Industry case studies

### 6.1 Automated distribution centre

**Context:** A retailer built a highly automated DC for a large, stable volume.
**Intervention:** Automated storage and retrieval, goods-to-person picking.
**Evidence:** Industry and academic case reporting (grade C/D).
**Outcome:** Throughput and accuracy improved at high utilisation; flexibility was reduced.
**What transferred:** automation is a bet on volume stability as much as on technology.
**What did not:** specific throughput figures.

### 6.2 Cross-dock design

**Context:** A parcel and grocery operator designed a cross-dock around timed waves.
**Intervention:** Door allocation, staging lanes, and wave scheduling.
**Evidence:** Industry reporting (grade C).
**Outcome:** Dwell time fell when inbound and outbound waves were coordinated.
**What transferred:** cross-dock capacity is a scheduling problem, not just a door count.
**What did not:** specific volumes.

### 6.3 Engineering revision failure

**Context:** A capital project issued piping isometrics that did not reflect the latest P&ID.
**Intervention:** No formal reconciliation between disciplines.
**Evidence:** Documented project and industry case reporting (grade C).
**Outcome:** Rework on site, schedule slip, and cost increase.
**What transferred:** revision control is a supply chain control; wrong data becomes wrong material.
**What did not:** specific costs.

### 6.4 A composite failure — the aisle that was 40 centimetres too narrow

**Context:** A site was designed with a narrow-aisle racking layout, then procured a larger truck for flexibility.
**Intervention:** Truck procurement decoupled from layout design.
**Evidence:** Composite teaching case.
**Outcome:** The trucks could not operate safely in the aisles; the racking had to be reconfigured.
**What transferred:** layout and equipment procurement must be designed together.
**What did not:** the specific cost.

## 7. Labs

### 7.1 Lab 45.1 — From–to and relationship chart

Given flows between eight areas, build a from–to chart and an activity relationship chart. Propose two block layouts and compare flow cost.

**Deliverable:** Charts, layouts, and cost comparison.

### 7.2 Lab 45.2 — Capacity calculation

For a given SKU profile and peak inventory, calculate positions, floor area, and dock doors at two service levels.

**Deliverable:** Capacity model and sensitivity.

### 7.3 Lab 45.3 — Equipment selection

Compare three handling options against throughput, density, labour, and capex. Produce a recommendation with a payback estimate.

**Deliverable:** Options analysis.

### 7.4 Lab 45.4 — Layout simulation

Build a discrete-event simulation of a DC layout. Identify the bottleneck and test a peak-season scenario.

**Deliverable:** Simulation and diagnosis.

### 7.5 Lab 45.5 — P&ID and MTO reconciliation

Compare two revisions of a tag list and an MTO. Classify differences by consequence and produce a procurement action list.

**Deliverable:** Reconciliation report.

## 8. Assignment / project

Design a facility and its engineering interface.

**Required components:**

1. Demand, SKU, and flow profile.
2. Space and equipment requirements.
3. Block and detailed layout.
4. Dock and staging design.
5. Handling and automation choice.
6. Simulation validation.
7. Safety, fire, and ergonomics.
8. Engineering data and revision control plan.
9. Cost, lifecycle, and flexibility analysis.
10. Risks, assumptions, and limits.

**Length:** 12–16 pages plus drawings and model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Flow analysis | 20% | Lists areas | From–to and relationship | Flow cost optimisation |
| Capacity | 20% | Rule of thumb | Positions, area, doors | Peak and service level |
| Layout | 20% | Draws blocks | Detailed layout | Validated with simulation |
| Equipment | 10% | Names options | Trade-off analysis | Lifecycle and flexibility |
| Engineering interface | 15% | Reads drawings | Tag and MTO comparison | Revision and consequence control |
| Safety | 10% | Mentions codes | Egress and fire design | Integrated into layout |
| Communication | 5% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Building and fire codes:** layout must comply; insurers may impose additional requirements.
- **Health and safety:** ergonomics, traffic, and racking inspection are legal duties.
- **Environment:** permits for emissions, discharge, noise, and hazardous storage.
- **Labour:** automation changes work; consultation and skills transition are required.
- **Procurement integrity:** engineering and construction procurement is corruption-prone.
- **Community:** traffic, noise, and visual impact affect planning consent.
- **Lifecycle:** design decisions lock in energy, maintenance, and retrofit costs.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Average-demand design | Peak queues | Variability ignored | Service failure | Peak and simulation |
| Aisle factor underestimated | Area shortfall | Hidden aisles and columns | Reconfiguration | Detailed layout early |
| Equipment decoupled | Truck does not fit | Separate procurement | Rework | Joint design |
| Dock over-utilisation | Trucks waiting | No headroom | Congestion | Utilisation target |
| Automation overreach | Low utilisation | Fixed cost | Poor payback | Volume stability test |
| Tag mismatch | Commissioning errors | No reconciliation | Rework | Cross-system matching |
| MTO stale | Wrong material | Revision not propagated | Delay and cost | Revision control |
| Egress blocked | Inspection finding | Layout changed late | Safety and closure | Design review |

## 12. Interview and discussion questions

1. How do you decide between a DC and a cross-dock? *(Hint: storage versus flow.)*
2. What is systematic layout planning? *(Hint: flow, relationships, space, blocks, detail.)*
3. How do you calculate storage positions? *(Hint: peak inventory, units per position, levels.)*
4. What is an aisle factor? *(Hint: gross-up for aisles and columns.)*
5. How many dock doors does a site need? *(Hint: door-hours divided by hours times utilisation.)*
6. Why is a dock designed to 100% utilisation a mistake? *(Hint: variability causes queues.)*
7. How does equipment choice constrain layout? *(Hint: turning radius and lift height.)*
8. What is a P&ID? *(Hint: piping, instruments, and control.)*
9. How do you reconcile an MTO with a drawing revision? *(Hint: tag matching and consequence classification.)*
10. What does design for supply chain mean? *(Hint: standardise, modularise, design for transport and maintenance.)*
11. How do you validate a layout? *(Hint: capacity, flow, simulation, sensitivity.)*
12. What safety constraints most affect layout? *(Hint: egress, fire, traffic separation.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Systematic layout planning | Structured method for facility layout |
| From–to chart | Matrix of flows between areas |
| Relationship chart | Qualitative closeness ratings (A–X) |
| Aisle factor | Ratio of gross to net storage area |
| Pallet position | One storage slot for a pallet |
| Dock door | Loading or unloading bay |
| Cross-dock | Flow-through facility with minimal storage |
| VNA | Very narrow aisle racking |
| AS/RS | Automated storage and retrieval system |
| P&ID | Piping and instrumentation diagram |
| Isometric | Three-dimensional piping drawing |
| MTO | Material take-off |
| Tag | Unique identifier for equipment or instrument |
| Design for supply chain | Designing products and plants for flow and service |
| Egress | Means of escape |

## 14. Sources and further reading

### Standards and regulation

- ISO 9001, 45001, 14001. https://www.iso.org/
- NFPA 13 and NFPA 101. https://www.nfpa.org/
- ISO 10628 process diagrams. https://www.iso.org/
- ISO 15926 and ISO 10303 (STEP). https://www.iso.org/
- IEC 61511 functional safety. https://www.iec.ch/
- ASME B31.3 process piping. https://www.asme.org/
- EN 15635 racking guidance. https://www.en-standard.eu/

### Reports and official sources

- Health and Safety Executive, warehouse and racking guidance. https://www.hse.gov.uk/
- National fire and building code authorities.
- Industry benchmarking on warehouse costs and productivity.

### Peer-reviewed and professional

- Muther, R., *Systematic Layout Planning*.
- Tompkins, J. A., et al., *Facilities Planning*.
- Apple, J. M., plant layout and material handling.
- Boysen, N., et al., warehouse design and order picking research.
- Gu, J., Goetschalckx, M., McGinnis, L., warehouse design research.

### Industry

- CILT and SOLE professional bodies. https://ciltinternational.org/
- Warehousing Education and Research Council resources. https://www.werc.org/
- Engineering contractor and vendor design guidance.

> **End of the advanced domain electives.** Return to the [course outline](COURSE_OUTLINE.md) for the full program, the [coverage map](TOPIC_COVERAGE_MAP.md) for topic-level traceability, or [Module 00](00_orientation_and_industry_map.md) to choose a track.
