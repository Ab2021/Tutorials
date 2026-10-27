# Module 31 — Aftermarket, Spare Parts, and Service Supply Chains

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Manufacturing, Logistics, and Healthcare tracks
> **Prerequisites:** Modules 06, 08, 10, 15
> **Tools:** Python, DuckDB, SimPy, OR-Tools
> **Industry question:** "How do you keep an installed base running without owning a warehouse of every part?"

---

## 1. Why industry cares

Most supply chain teaching stops at the sale. In reality, the sale creates a second supply chain that can run for twenty years: spare parts, repairs, field service, warranties, and service contracts. For many manufacturers, the aftermarket earns a disproportionate share of profit because parts carry higher margins, demand is less price-sensitive, and the customer is locked into the installed asset.

The aftermarket is also where supply chain failures are most visible. A missing €40 seal can stop a €2 million turbine, a hospital scanner, or a production line. The cost of a stockout is not the lost margin on the part; it is the downtime, the contractual penalty, and the customer relationship.

Three forces make this a distinct analytics domain:

- **Demand is driven by the installed base, not by the market.** You cannot forecast spare parts from GDP or footfall. You forecast from what is installed, how old it is, how hard it runs, and how it fails.
- **Demand is intermittent and lumpy.** Most part-location combinations have zero demand in most periods. Averages mislead; the tail is the business.
- **The objective is availability, not fill rate.** A 95% fill rate on parts can still produce unacceptable downtime if the missing 5% are the critical ones.

The analytics problem is to connect the **installed base** to **failure demand**, place **spares** across a multi-echelon network, schedule **technicians**, and price **service contracts** so that the promise is both deliverable and profitable.

## 2. Learning outcomes

By the end of this module, you can:

- Build an installed-base model from asset, serial-number, and contract data.
- Estimate failure demand from failure rates, age, usage, and operating conditions.
- Segment parts by criticality, value, and demand pattern, and choose a policy for each segment.
- Design a multi-echelon sparing network with central, regional, and forward stock.
- Compute availability, fill rate, and response-time metrics, and explain why they differ.
- Model the repair loop, including turnaround time, no-fault-found, and cannibalization.
- Build a field-service scheduling model with skills, travel, and service windows.
- Evaluate service-contract economics, including penalty clauses and renewals.
- Quantify the cost of downtime and use it to set spares targets.
- Design a service parts control tower with the exceptions that matter.

## 3. Mental model

```text
   INSTALLED BASE
   assets × age × usage × environment
              │
              ▼
   FAILURE DEMAND ──> CRITICALITY ──> POLICY
   rate, mode,           consequence       stock level
   interval              of failure        location
              │                 │              │
              └────────────┬────┘              │
                           ▼                   ▼
                    SPARES NETWORK ─────> FIELD SERVICE
                    central / regional      skills, travel,
                    / forward stock         windows, repair
                           │                   │
                           ▼                   ▼
                    REPAIR LOOP <────── RETURNS / EXCHANGE
                    test, repair,        failed unit
                    refurbish, scrap
                           │
                           ▼
                    AVAILABILITY / UPTIME
                    the promise the customer bought
```

The customer buys uptime. The supply chain delivers it through parts, people, and repair capacity. A model that optimizes parts cost alone will under-invest in the part that stops the asset, and over-invest in the part that does not.

### 3.1 Two demand streams

| Stream | Driver | Pattern | Consequence of error |
|---|---|---|---|
| Planned maintenance | Maintenance schedule, intervals | Predictable, lumpy | Schedule slips, overtime |
| Unplanned failure | Failure rate, age, condition | Intermittent, random | Downtime, emergency expedite |
| Installed-base growth | New installations | Step changes | Stockout at commissioning |
| End-of-life | Retirement, last-time buy | Decline, then zero | Obsolete stock or stranded assets |

Treat the streams separately. A single forecast that blends planned kits and random failures will be wrong for both.

## 4. Core concepts

### 4.1 The installed base is the demand driver

Start with the asset register, not the sales history:

| Field | Why it matters |
|---|---|
| Asset ID and serial | Links failures, warranty, and service history |
| Product and configuration | Determines the parts BOM |
| Installation date | Age and warranty status |
| Operating hours or cycles | Usage-based failure |
| Environment | Heat, dust, salt, vibration |
| Service contract | Entitlement and response target |
| Location | Depot, site, region, country |
| Criticality | Consequence of downtime |

An installed-base model converts assets into expected failures:

\[
\text{Expected failures}_{p,t} = \sum_{a \in \text{assets}_p} \lambda_a \cdot \text{exposure}_{a,t}
\]

where \(\lambda_a\) is the failure rate for the asset's age, usage, and environment, and exposure is the fraction of the period the asset is operating. For a first pass, use a fleet-average rate by product and age band; then refine with condition data.

### 4.2 Failure and demand

Failure rates are rarely constant. The bathtub curve is a teaching device, not a model:

- **Early life:** installation and manufacturing defects; decreasing hazard.
- **Useful life:** approximately constant hazard for many electronic and mechanical parts; random failures.
- **Wear-out:** increasing hazard with age or cycles; replace before failure.

For repairable systems, model the failure process; for consumable parts, model the demand count. Do not apply a constant failure rate to a wear-out part and call it a forecast.

**Intermittent demand.** Most part-location series are mostly zeros. Use methods designed for intermittency:

| Method | When it fits | Watch |
|---|---|---|
| Croston | Slow-moving, intermittent | Biased for lumpy series; use SBA variant |
| TSB | Obsolescence risk | Requires probability of demand |
| Poisson or negative binomial | Count data with a stable rate | Overdispersion breaks Poisson |
| Compound Poisson | Random count and random size | Needs size distribution |
| Age- and usage-based hazard | Wear-out parts | Requires asset-level data |

The practical test: plot the inter-arrival times. If they are not exponential, a Poisson assumption is a convenience, not a finding.

### 4.3 Criticality and segmentation

Not every part deserves the same service. Segment on two axes: **consequence** and **demand/value**.

| Segment | Example | Policy |
|---|---|---|
| Critical, high value | Turbine blade, MRI tube | Central stock, guaranteed response, condition monitoring |
| Critical, low value | Seal, sensor, fuse | Forward stock, high service level, cheap insurance |
| Non-critical, high value | Cosmetic panel | Order on demand, long lead time accepted |
| Non-critical, low value | Fastener, filter | Vendor-managed, bulk, local |
| Obsolete / end-of-life | Legacy board | Last-time buy, refurbish, substitute |

Criticality methods:

- **Consequence-based:** downtime cost per hour × expected repair time, plus safety and regulatory impact.
- **FMEA:** severity × occurrence × detection; useful for ranking failure modes, not for setting stock levels.
- **VED:** vital, essential, desirable; fast but subjective.
- **Service criticality:** the contractual response time and penalty attached to the asset.

The output is a policy table, not a score. A part is critical because of what happens when it is missing, not because someone marked it "A".

### 4.4 Multi-echelon sparing

Spars are held at several levels because response time and pooling trade off:

| Echelon | Role | Strength | Cost |
|---|---|---|---|
| Central depot | Pool of last resort, repairs | Highest pooling, lowest total stock | Long response |
| Regional DC | Regional coverage | Faster response, some pooling | More stock |
| Forward / site | Immediate availability | Meets response target | Duplicated stock, low turnover |
| Supplier / OEM | Consignment or direct ship | No inventory ownership | Lead time and dependency |

The classic result is that centralizing slow-moving, expensive parts reduces total stock for a given service level, while local stock reduces response time. The design question is which service promise each customer segment has bought.

**METRIC logic.** For recoverable items, expected backorders at a base depend on the demand rate, the repair turnaround, and the stock level. The multi-echelon model allocates stock to minimize total expected backorders subject to a budget, or minimizes cost subject to a backorder target. The intuition:

- Pool expensive, slow-moving items centrally.
- Push cheap, critical, fast-moving items forward.
- Treat repair turnaround as part of the lead time; a fast repair depot can substitute for stock.
- Recalculate when the installed base, failure rate, or network changes.

A full METRIC implementation is not required for every programme. A spreadsheet with demand during lead time, a target fill rate, and a criticality override captures most of the value; the model matters when the parts portfolio is large and the service targets are contractual.

### 4.5 Availability and service metrics

| Metric | Definition | Use |
|---|---|---|
| Fill rate | Share of part requests met from stock | Parts performance |
| Cycle service level | Probability of no stockout in a cycle | Inventory policy |
| Response time | Time from call to technician arrival | Field service |
| Repair turnaround | Time from failed unit receipt to return | Repair loop |
| Operational availability | Uptime / (uptime + downtime) | Customer promise |
| MTBF | Mean time between failures | Reliability |
| MTTR | Mean time to repair, including waiting | Service design |
| First-time fix | Share of visits resolved on the first visit | Field efficiency |
| Contract uptime | Availability against the SLA | Commercial |

\[
A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}
\]

MTTR includes waiting for the part, the technician, and the access window. A spare-part policy that ignores technician and access time will over-promise.

### 4.6 The repair loop

A failed unit is an asset, not waste. The repair loop has its own supply chain:

```text
failed unit → recovery → test → repair / refurbish → return to stock
                    │
                    └→ no-fault-found → return to stock with a test record
                    └→ beyond repair → cannibalize → scrap
```

Key metrics: repair yield, turnaround, no-fault-found rate, cannibalization rate, and the cost of a repaired unit versus a new one. The loop is a capacity constraint: a depot with a four-week backlog is equivalent to a four-week lead time, no matter what the stock level says.

### 4.7 Field service and scheduling

Field service is a routing and scheduling problem with skills, parts, and service windows:

- **Skills:** not every technician can perform every repair.
- **Parts:** the technician must carry or collect the right part.
- **Windows:** customer access hours, safety permits, production schedules.
- **Travel:** travel time is paid time and a large share of cost.
- **Priority:** emergency, next-day, planned maintenance.

The analytics question is not only "what is the optimal route?" It is "which visits should be combined, which parts should be carried, and which jobs should be deferred?" Measure first-time fix, visits per job, travel time share, and overtime.

### 4.8 Service contracts and pricing

A service contract is a supply chain promise with a price. The economics depend on:

| Element | Question |
|---|---|
| Coverage | Which assets, which failures, which exclusions? |
| Response | On-site within 4 hours, next business day, or remote only? |
| Parts | Included, capped, or charged? |
| Labor | Included hours, overtime, travel? |
| Uptime guarantee | Penalty or service credit? |
| Term and renewal | Annual, multi-year, auto-renew? |

Price the contract against the expected cost of delivering the promise, including the tail. A contract that is profitable on average but loses money in the worst 5% of years is a risk, not a product.

### 4.9 Worked example — spares for a critical part

An installed base of 1,200 machines uses a critical seal. The fleet-average failure rate is 0.15 per machine-year, so expected failures are 180 per year. The replenishment lead time is 45 days, so expected demand during lead time is:

\[
\lambda_{LT} = 180 \times \frac{45}{365} \approx 22.2 \text{ units}
\]

Using a Poisson approximation, a 95% fill-rate target requires roughly:

\[
Q \approx \lambda_{LT} + z \sqrt{\lambda_{LT}} = 22.2 + 1.645 \times 4.71 \approx 30 \text{ units}
\]

Now compare stock levels against downtime:

| Stock level | Expected wait per failure | Operational availability | Downtime cost at €1,500/h | Holding cost |
|---:|---:|---:|---:|---:|
| 22 | 4 days | 98.6% | €155k | €2.2k |
| 30 | 1 day | 99.7% | €39k | €3.0k |
| 38 | 0.2 day | 99.9% | €13k | €3.8k |

The numbers are illustrative, but the structure is the decision: the marginal unit of stock buys a reduction in waiting time, which buys availability, which avoids downtime. For a critical, low-value part, the stock is cheap insurance. For a non-critical, high-value part, the same logic would recommend against holding it locally.

**The mistake to avoid:** applying the same 95% fill rate to every part. A 95% target on the critical seal and a 95% target on a cosmetic panel produce the same metric and completely different customer outcomes.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Grain | Use |
|---|---|---|
| Asset register | Asset / serial | Installed base, age, configuration |
| Work orders | Work order | Failure, repair, labor, parts consumed |
| Failure codes | Work order / component | Failure mode and reliability |
| Parts BOM | Asset / product | Which parts fit which asset |
| Inventory | Part × location | On-hand, in-transit, reserved |
| Repair orders | Unit / serial | Turnaround, yield, no-fault-found |
| Contracts | Contract / asset | Entitlement, SLA, penalties |
| Technician schedule | Technician × day | Capacity and skills |
| Sensor / condition data | Asset × time | Predictive maintenance |

### 5.2 Standards and identifiers

- **GS1 GTIN** for parts and **GS1 GRAI** for returnable assets where applicable.
- **Serial numbers and asset IDs** as the join key between installed base, warranty, and service history.
- **ISO 55000** asset management principles for the asset lifecycle.
- **ISO 13374** condition monitoring and diagnostics where sensor data is used.
- **IEC 62402** obsolescence management for long-lived assets.
- **Aerospace and defence standards** (for example ASD/AIA specifications) where applicable; verify the current revision.
- **UDI** for medical devices (Module 33).

### 5.3 Workflow

1. Maintain the asset register and configuration.
2. Capture failures with a coded failure mode, not free text only.
3. Estimate failure rates by product, age, and environment.
4. Segment parts by criticality and demand.
5. Set stock levels by echelon and service target.
6. Schedule preventive maintenance and field service.
7. Run the repair loop and track turnaround.
8. Review contract performance and renewals.
9. Update the installed base after installs, upgrades, and retirements.

The workflow fails when the asset register is incomplete. A spare-parts model built on a sales history rather than an installed base is a demand forecast for the wrong thing.

## 6. Industry case studies

### 6.1 Rolls-Royce — power-by-the-hour

**Context:** Aircraft engines are sold with long-term service agreements in which the customer pays for engine availability rather than for individual parts and repairs.
**Intervention:** The manufacturer retains responsibility for maintenance, parts, and reliability, and manages a global repair and spare-parts network.
**Evidence:** Company disclosures and public case descriptions (grade C); specific contract economics are not public.
**Outcome:** The business model aligns the manufacturer's incentive with uptime and asset reliability.
**What transferred:** when the supplier owns availability, the supply chain objective becomes uptime, and parts policy follows from reliability and turnaround.
**What did not:** the model depends on long asset lives, high downtime cost, and the manufacturer's ability to control the repair network.

### 6.2 Caterpillar — dealer parts network

**Context:** Construction and mining equipment is serviced through independent dealers with local parts inventories.
**Intervention:** A multi-tier parts network with dealer stock, regional distribution centres, and overnight replenishment.
**Evidence:** Company disclosures describe the dealer network and parts availability (grade C).
**Outcome:** Local availability is delivered through a combination of dealer stock and fast replenishment rather than through duplicating every part everywhere.
**What transferred:** the echelon design is a service promise, and dealer incentives must align with parts availability.
**What did not:** dealer independence and local market conditions change the optimal stock allocation.

### 6.3 Elevator and escalator maintenance — Otis and Schindler

**Context:** Elevators are long-lived assets with safety-critical maintenance and a large installed base.
**Intervention:** Predictive and condition-based maintenance using sensor data, plus a field-service network with local spare parts.
**Evidence:** Company disclosures and industry reporting (grade C/D); performance claims are company statements.
**Outcome:** Remote monitoring and dispatch reduce unnecessary visits and improve first-time fix.
**What transferred:** sensor data changes the demand signal from historical failure to current condition, but the parts and skills network still determines response.
**What did not:** the value depends on connectivity, data rights, and the installed base's age mix.

### 6.4 A composite failure — the part nobody stocked

**Context:** A manufacturer consolidated regional spare-parts stocks into one central depot to reduce inventory.
**Intervention:** Centralization with next-day courier delivery.
**Evidence:** Composite teaching case, not a published study.
**Outcome:** Total inventory fell, but downtime rose for customers with four-hour response contracts. The savings were smaller than the penalties and renewals lost.
**What transferred:** centralization is optimal only against the service promise actually sold. Segment the installed base by contract before consolidating stock.
**What did not:** the specific numbers, which depend on contract terms and customer concentration.

## 7. Labs

### Lab 31.1 — Installed base and failure demand

Build an asset register from a synthetic or permitted dataset. Compute expected failures by product, age band, and region. Compare with observed work orders and explain the gap.

**Deliverable:** Installed-base model, expected-versus-observed chart, and assumptions.

### Lab 31.2 — Criticality segmentation

Score 200 parts on downtime cost, safety, regulatory impact, lead time, and value. Assign segments and propose a stock policy for each.

**Deliverable:** Segmentation table and policy memo.

### Lab 31.3 — Multi-echelon sparing

Given demand rates, lead times, repair turnaround, and holding costs, allocate stock across a central depot and three regional nodes. Compare cost and availability with a flat policy.

**Deliverable:** Allocation model, cost-service frontier, and recommendation.

### Lab 31.4 — Field service scheduling

Build a routing and scheduling model with skills, parts, and time windows. Compare with a first-come-first-served baseline on travel time, first-time fix, and overtime.

**Deliverable:** Schedule, comparison table, and assumptions.

### Lab 31.5 — Service contract P&L

Model a service contract with a response SLA and penalty clause. Simulate demand and repair times. Compute margin under the current stock policy and under two alternatives.

**Deliverable:** Contract P&L, risk distribution, and recommendation.

## 8. Assignment / project

Design a service-parts and field-service network for an installed base of 5,000 assets across three regions.

**Required components:**

1. Installed-base profile and failure assumptions.
2. Criticality segmentation.
3. Demand model, including intermittent demand.
4. Multi-echelon stock policy.
5. Repair-loop capacity and turnaround.
6. Field-service capacity and scheduling.
7. Service promise and contract economics.
8. Measurement plan and control tower.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Installed-base logic | 20% | Uses sales history | Asset-based demand | Age, usage, and environment |
| Criticality | 20% | ABC by value | Consequence and demand | Contract-linked segmentation |
| Sparing network | 20% | Flat stock | Echelon allocation | Pooling, turnaround, and service target |
| Field service | 15% | Describes process | Scheduling model | Skills, parts, and first-time fix |
| Contract economics | 15% | Price only | Cost-to-serve | Tail risk and penalty exposure |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready and honest about limits |

## 10. Regulatory, ethical, and commercial lens

- **Safety-critical parts:** counterfeit or untraceable parts are a safety and liability risk; sourcing controls are mandatory.
- **Right to repair:** jurisdictions are expanding access to parts, tools, and repair information; service strategy must comply.
- **Export controls and sanctions:** spare parts and technical data may be controlled; screen customers and destinations.
- **Warranty and consumer law:** response promises and exclusions must be clear and enforceable.
- **Labour:** field technicians work alone, travel, and face safety risks; scheduling must respect working-time rules.
- **Data:** asset and machine data may be personal or commercially sensitive; define ownership in the contract.
- **Environment:** repair and remanufacture can reduce emissions; disposal must follow waste rules.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Sales-history forecasting | Parts forecast from orders | Orders are censored by stockouts | Wrong stock | Installed-base model |
| Uniform service target | 95% everywhere | Ignores criticality | Downtime on critical parts | Consequence-based targets |
| Centralization without contract check | Inventory falls, penalties rise | Service promise ignored | Lost renewals | Segment by SLA |
| Repair backlog invisible | Lead time assumed from supplier | Depot is the constraint | False availability | Measure turnaround |
| No-fault-found dumped | High NFF rate | Test process weak | Phantom shortage | Test and return to stock |
| Technician as courier | High travel share | Parts not staged | Low productivity | Kitting and forward stock |
| Contract priced on average | Margin looks fine | Tail risk excluded | Losses in bad years | Simulate the tail |
| Obsolete stock | Last-time buy missed | Long asset life | Stranded asset | Obsolescence management |
| Cannibalization unrecorded | Good units stripped | Emergency culture | Hidden cost | Track and authorize |
| Installed base stale | Retirements not recorded | Register not maintained | Wrong demand | Reconcile annually |

## 12. Interview and discussion questions

1. Why is the installed base a better demand driver than sales history? *(Hint: sales are censored by stockouts and do not describe the asset.)*
2. How do you forecast a part with 1.5 units of demand per year? *(Hint: intermittent methods, Poisson or compound, and criticality override.)*
3. What is the difference between fill rate and availability? *(Hint: fill rate counts requests; availability counts uptime.)*
4. When should a critical spare be held locally rather than centrally? *(Hint: downtime cost, response promise, part value, lead time.)*
5. How does repair turnaround act as a lead time? *(Hint: a failed unit is out of service until it returns.)*
6. What is METRIC trying to optimize? *(Hint: stock allocation to minimize backorders or cost across echelons.)*
7. How would you price a service contract with an uptime guarantee? *(Hint: expected cost plus tail risk, not average cost.)*
8. What is first-time fix, and why does it matter? *(Hint: avoidable repeat visits and customer effort.)*
9. How do sensor data change spare-parts planning? *(Hint: condition-based demand, but only if the data is trustworthy.)*
10. What would you do about a high no-fault-found rate? *(Hint: test process, diagnostics, training, and return-to-stock.)*
11. How do you handle end-of-life parts? *(Hint: last-time buy, refurbish, substitute, redesign.)*
12. Which service metric would you put in a customer-facing dashboard, and why?

## 13. Key terms

| Term | Definition |
|---|---|
| Installed base | Population of assets in service at customers |
| Aftermarket | Parts, service, and support sold after the original equipment sale |
| Failure rate | Expected failures per unit of exposure |
| MTBF | Mean time between failures |
| MTTR | Mean time to repair, including waiting |
| Operational availability | Uptime divided by uptime plus downtime |
| Fill rate | Share of part requests served from stock |
| Criticality | Consequence of a part being unavailable |
| FMEA | Failure mode and effects analysis |
| VED | Vital, essential, desirable classification |
| Multi-echelon sparing | Stock placement across central, regional, and forward nodes |
| METRIC | Multi-echelon technique for recoverable item control |
| Repair turnaround | Time from failed-unit receipt to return to service |
| No-fault-found | Unit tested and found to have no defect |
| Cannibalization | Removing a part from one asset to repair another |
| First-time fix | Share of service visits that resolve the issue |
| Service-level agreement | Contractual response and availability commitment |
| Power by the hour | Pricing availability rather than parts and labour |
| Last-time buy | Final purchase of a part before obsolescence |
| Condition-based maintenance | Maintenance triggered by asset condition data |

## 14. Sources and further reading

### Standards and guidance

- ISO 55000, Asset management — overview, principles, and terminology.
- ISO 13374, Condition monitoring and diagnostics of machines.
- IEC 62402, Obsolescence management.
- GS1, GTIN and GRAI identification standards. https://www.gs1.org/standards
- ASD/AIA specifications for aerospace and defence service data, current revision.

### Peer-reviewed and professional

- Sherbrooke, C. C., *Optimal Inventory Modeling of Systems: Multi-Echelon Techniques*.
- Muckstadt, J. A., *Analysis and Algorithms for Service Parts Supply Chains*.
- Silver, E. A., Pyke, D. F., Thomas, D. J., *Inventory and Production Management in Supply Chains*.
- Croston, J. D., "Forecasting and Stock Control for Intermittent Demands," *Operational Research Quarterly*, 1972.
- Cohen, M. A., Agrawal, N., Agrawal, V., "Winning in the Aftermarket," *Harvard Business Review*, 2006.

### Industry

- Rolls-Royce annual reports and service-business disclosures. https://www.rolls-royce.com/
- Caterpillar annual reports and dealer-network disclosures. https://www.caterpillar.com/
- Otis and Schindler annual reports and service disclosures.
- Airbus and Boeing maintenance and spares documentation where public.

> **Next:** [Module 32](32_food_agriculture_perishables.md) applies the same availability logic to products that expire, where freshness and waste replace uptime as the binding constraint.
