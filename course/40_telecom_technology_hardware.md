# Module 40 — Telecom, Technology Hardware, and Aerospace Supply Chains

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Manufacturing and Logistics tracks
> **Prerequisites:** Modules 09, 10, 14, 31, 37
> **Tools:** Python, OR-Tools, BOM and allocation models, GIS for site rollout
> **Industry question:** "The chip is 40 weeks out, the launch is in 12, and the network must be live — what do you commit?"

---

## 1. Why industry cares

Telecom, technology hardware, and aerospace are the supply chains with the highest value density, the longest component lead times, and the least tolerance for a missing part. A handset launch without a component, a network rollout without a radio unit, or an aircraft on the ground without a valve are all failures measured in millions.

Four features dominate:

- **Component scarcity and allocation.** Semiconductors, batteries, and specialised parts are allocated, not ordered. Lead times can exceed the product's life.
- **Rapid obsolescence.** A device or radio may be superseded before its inventory clears.
- **Configuration complexity.** A platform or network has thousands of variants, each with its own BOM and approval.
- **Regulated airworthiness and security.** Aerospace parts need traceability and approval; telecom equipment is subject to security and export rules.

The analytics problem is **allocation, ramp, and configuration under long lead times and short product lives**.

## 2. Learning outcomes

By the end of this module, you can:

- Map a technology supply chain from component to network or platform.
- Model component allocation under shortage and choose an allocation rule.
- Plan a new product introduction and ramp with long-lead components.
- Design a network rollout supply chain with site, civil works, and equipment dependencies.
- Manage configuration and variant complexity in a BOM.
- Plan device supply, channel inventory, and trade-in flows.
- Apply aerospace AOG and parts-pooling logic.
- Assess obsolescence and last-time-buy decisions.
- Apply export-control and security obligations.
- Measure availability, fill rate, and program risk in a technology program.

## 3. Mental model

```text
   COMPONENT ──> BOARD / MODULE ──> DEVICE / SYSTEM ──> DEPLOYMENT ──> CUSTOMER
   (chip,        (EMS, contract     (handset, radio,    (network,       (subscriber,
    battery,      manufacturer)       payload, engine)    platform)       airline)
    sensor)            │                   │                  │
        │              │                   │                  │
   allocation      NPI and ramp       configuration      site and install
   and lead time   and yield          and approval       and commissioning
        │
        └──> repair, spares, reverse, end-of-life
```

Two clocks run in parallel: the **component clock** (lead time, allocation, obsolescence) and the **deployment clock** (site readiness, installation, commissioning). A program is late when either clock slips.

### 3.1 The three sub-domains

| Sub-domain | Product | Critical constraint |
|---|---|---|
| Telecom network | Radio, transport, core, fibre | Site acquisition and civil works |
| Technology hardware | Devices, servers, electronics | Component allocation and NPI |
| Aerospace | Aircraft, engines, avionics | Airworthiness and configuration |

They share allocation, configuration, and reverse-logistics logic. They differ in regulation and deployment.

## 4. Core concepts

### 4.1 Value density and obsolescence

Technology products have high value per kilogram, so transport is rarely the constraint; **time and availability** are. Obsolescence is the dominant inventory risk:

\[
\text{Obsolescence cost} = \text{units} \times (\text{cost} - \text{recovery value})
\]

A device that is one generation behind may lose most of its value in a quarter. Inventory policy must include a **life-cycle clock**, not only a demand forecast.

### 4.2 Component allocation

When supply is short, the supplier allocates. The buyer's problem is to allocate the allocation across products, customers, and commitments:

| Rule | Objective | Risk |
|---|---|---|
| Proportional to forecast | Fairness | Rewards inflated forecasts |
| Proportional to history | Stability | Protects incumbents |
| Margin priority | Profit | Harms strategic customers |
| Contract priority | Legal compliance | Ignores opportunity |
| Strategic priority | Long-term position | Subjective |

The right rule is explicit, documented, and reviewed. Allocation by escalation is not a rule.

### 4.3 EMS and contract manufacturing

Electronics manufacturing services (EMS) assemble boards and products for brand owners. The brand owner typically controls design, demand, and component sourcing; the EMS controls capacity and process.

Key analytics: **consignment versus turnkey**, component ownership, yield, capacity reservation, and liability for obsolete inventory.

### 4.4 New product introduction and ramp

NPI is a sequence:

```text
design ──> prototype ──> qualification ──> pilot ──> ramp ──> volume
   │            │              │              │         │
 BOM freeze   test           certification   yield     capacity
```

The supply chain must commit long-lead components before demand is known. The decision is a bet on the ramp curve, with cancellation and expedite options priced in.

### 4.5 Telecom network rollout

A network rollout is a project supply chain:

- **Site acquisition:** landlords, permits, planning.
- **Civil works:** foundations, towers, power, backhaul.
- **Equipment:** radio, antennas, transport, core.
- **Integration:** commissioning, testing, acceptance.

The constraint is rarely the equipment alone; it is the **site readiness** that lets installed equipment earn revenue. A radio in a warehouse is inventory; a radio on a live site is capacity.

### 4.6 Device supply and channel

Device supply chains add:

- Channel inventory and sell-through versus sell-in.
- Operator subsidies and contract terms.
- Trade-in and upgrade flows.
- SIM and eSIM provisioning.
- Warranty and repair.

The analytics trap is **sell-in versus sell-through**: shipping to channel is not demand, and channel inventory can hide a problem for a quarter.

### 4.7 Aerospace AOG and configuration

Aircraft on ground (AOG) is the highest-priority service event. The supply chain needs:

- AOG desks and expedite channels.
- Parts pooling and exchange pools.
- Airworthiness-approved parts and traceability.
- Configuration management by tail number.
- Repair and overhaul loops.

A part that is physically available but not approved for the configuration is not available.

### 4.8 Repair, reverse, and refurbishment

Technology returns flow into repair, refurbishment, harvest, or recycling (Module 37). The additional constraint is **data wiping and security**: a returned device may contain sensitive data, and a returned network unit may contain credentials.

### 4.9 Export control and security

- **ITAR** and **EAR** control defence and dual-use items and technical data.
- **Entity lists** restrict parties.
- **Telecom security rules** restrict certain vendors in some jurisdictions.
- **Country-of-origin** and **component tracing** requirements are increasing.

The analytics obligation is a **screened party and item register**, not a one-time check.

### 4.10 Worked example — component allocation

A device maker has 900,000 units of a constrained component and demand for 1,200,000 units across three products:

| Product | Demand | Margin per unit | Strategic weight |
|---|---:|---:|---:|
| A | 600,000 | €40 | 1.0 |
| B | 400,000 | €25 | 1.2 |
| C | 200,000 | €60 | 0.8 |

A pure margin rule fills A (600k) then C (200k) then B (100k). A strategic rule may protect B's customer relationship. A proportional rule gives each product 75% of demand: A 450k, B 300k, C 150k.

The "best" rule depends on the objective. The analytics task is to make the trade-off visible and to state the rule before the shortage, not during it.

### 4.11 Worked example — network rollout pipeline

A operator plans 6,000 sites over 150 weeks, or 40 sites per week. Equipment lead time is 20 weeks.

\[
\text{Pipeline requirement} = 40 \times 20 = 800 \text{ site kits in the pipeline}
\]

If site readiness runs at 30 sites per week, the equipment pipeline is not the constraint; the civil works are. Building 800 kits of inventory against a 30-site-per-week readiness rate creates 200 kits of excess per week, or 4,000 kits over 20 weeks. The correct plan is to synchronise equipment call-off to site readiness, not to the nominal rollout rate.

### 4.12 Worked example — last-time buy

A network operator uses a line card that the supplier will stop producing in nine months. Annual consumption is 4,000 units, the remaining product life is four years, and the supplier offers a final buy at €1,200 per unit. Storage and obsolescence cost is 18% per year, and the card's replacement requires a design change costing €2.5 million.

\[
\text{Last-time buy quantity} = 4{,}000 \times 4 = 16{,}000 \text{ units}
\]

\[
\text{Purchase cost} = 16{,}000 \times 1{,}200 = €19.2 \text{ million}
\]

Holding cost over the period is substantial, but the design change may cost more, and a stockout would strand installed equipment. The decision is not "buy everything" or "buy nothing"; it is a comparison of last-time buy, bridge buy with a redesign, and dual sourcing, each with a different risk of shortage and write-off.

The model should state the demand assumption, the holding rate, the redesign cost, and the residual value. A last-time buy without a life-cycle demand model is a guess with a large cheque attached.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| BOM | PLM / ERP | Configuration and components |
| Component lead time | Supplier / ERP | Allocation and planning |
| Allocation | Supplier portal | Shortage response |
| Production | MES | Yield and output |
| Site plan | GIS / project system | Rollout |
| Install status | Field system | Commissioning |
| Channel inventory | Distributor / retail | Sell-in versus sell-through |
| AOG and repairs | MRO system | Availability |
| Export screening | Compliance system | Party and item checks |

### 5.2 Standards

- **IPC** standards for electronics assembly (for example IPC-A-610).
- **JEDEC** for semiconductor quality and reliability.
- **SEMI** standards for semiconductor manufacturing.
- **3GPP** and **ITU** for telecom networks.
- **GSMA** for eSIM and mobile standards.
- **FAA** and **EASA** airworthiness rules; **ASA-100** and **AS9120** for aerospace distribution.
- **ITAR** and **EAR** for export control.
- **ISO 27001** and **NIST** frameworks for security.
- **GS1** for identification and traceability.

### 5.3 Workflow

1. Define configuration and BOM.
2. Commit long-lead components.
3. Allocate supply under shortage.
4. Build and test.
5. Deploy to site or channel.
6. Commission and accept.
7. Repair and reverse.
8. Screen and report compliance.

## 6. Industry case studies

### 6.1 The 2020–2022 semiconductor shortage

**Context:** Demand surged and capacity was constrained across chips, substrates, and components.
**Intervention:** Allocation, long-term agreements, design changes, and inventory investment.
**Evidence:** Industry and government reporting (grade B/C).
**Outcome:** Lead times extended dramatically in some categories; automakers and device makers curtailed production.
**What transferred:** allocation and design flexibility are supply chain capabilities, not purchasing tactics.
**What did not:** specific lead times, which varied by node and package.

### 6.2 5G network rollout

**Context:** Operators deployed new radio and transport equipment across thousands of sites.
**Intervention:** Site planning, equipment supply, and integration programs.
**Evidence:** Operator disclosures and regulator reports (grade C).
**Outcome:** Rollout pace was often set by site acquisition and civil works, not equipment supply.
**What transferred:** deployment is a project supply chain; synchronise equipment to site readiness.
**What did not:** specific coverage and timing.

### 6.3 Aerospace AOG support

**Context:** Airlines need parts within hours to return aircraft to service.
**Intervention:** AOG desks, exchange pools, and expedited logistics.
**Evidence:** Industry and MRO reporting (grade C).
**Outcome:** Availability depends on pool design, approved parts, and logistics speed.
**What transferred:** in aerospace, approval and configuration are part of availability.
**What did not:** specific costs and response times.

### 6.4 A composite failure — the launch that shipped without a component

**Context:** A device launch committed to a date before a critical component was qualified.
**Intervention:** None; the program assumed qualification would pass.
**Evidence:** Composite teaching case.
**Outcome:** The component failed qualification, the launch slipped, and channel commitments were missed.
**What transferred:** NPI supply plans need qualification gates and fallback components.
**What did not:** the specific product and loss.

## 7. Labs

### 7.1 Lab 40.1 — Allocation under shortage

Allocate a constrained component across products and customers under three rules. Compare margin, service, and strategic outcomes.

**Deliverable:** Allocation model and policy memo.

### 7.2 Lab 40.2 — NPI ramp

Build a ramp plan with long-lead components, qualification gates, and yield. Estimate the cost of a two-week slip.

**Deliverable:** Ramp model and risk analysis.

### 7.3 Lab 40.3 — Network rollout

Plan equipment call-off against site readiness for a 5,000-site program. Identify the binding constraint.

**Deliverable:** Rollout plan and constraint analysis.

### 7.4 Lab 40.4 — Configuration and obsolescence

Given a BOM with variants and life-cycle dates, identify obsolete parts and last-time-buy quantities.

**Deliverable:** Obsolescence register and buy plan.

### 7.5 Lab 40.5 — AOG and pool design

Design a spare pool for a fleet with AOG events. Compare pool size, exchange, and expedite.

**Deliverable:** Pool model and service-level analysis.

## 8. Assignment / project

Design a supply chain for a technology program: a device launch, a network rollout, or an aerospace support contract.

**Required components:**

1. Configuration and BOM.
2. Component and allocation strategy.
3. NPI or deployment plan.
4. Capacity and yield.
5. Channel or site logistics.
6. Repair and reverse flow.
7. Export control and security.
8. Metrics and governance.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Configuration | 20% | Lists parts | BOM and variants | Life-cycle and approval |
| Allocation | 20% | Simple rule | Cost-based allocation | Multi-objective and documented |
| Ramp or rollout | 20% | Timeline | Dependencies and capacity | Binding constraint and synchronisation |
| Obsolescence | 15% | Mentions risk | Last-time buy | Life-cycle economics |
| Compliance | 15% | Names rules | Screening process | Item, party, and origin controls |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Export control:** ITAR, EAR, and entity lists restrict items, data, and parties.
- **Airworthiness:** parts must be approved and traceable; unapproved parts are a safety risk.
- **Telecom security:** some jurisdictions restrict vendors and require supply-chain security.
- **Data protection:** returned devices and network equipment may contain personal data.
- **Competition:** allocation and information sharing can raise competition issues.
- **Labour:** electronics supply chains carry labour-risk exposure; traceability is expected.
- **Environment:** e-waste and battery rules apply to devices and network equipment.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Allocation by escalation | Customer complaints | No rule | Inequity and lost trust | Pre-agreed rule |
| Double ordering | Inflated forecasts | Shortage panic | Bullwhip and excess | Forecast audit |
| NPI without gate | Qualification slips | No fallback | Launch delay | Gates and alternates |
| Equipment ahead of sites | Warehouse full | Rollout unsynchronised | Excess inventory | Site-driven call-off |
| Sell-in mistaken for demand | Channel inventory rising | Shipments counted as sales | Overproduction | Sell-through tracking |
| Obsolete stock | Life-cycle date passed | No last-time buy | Write-off | Obsolescence register |
| Unapproved part | AOG delay | Configuration mismatch | Safety and cost | Approval check |
| Export violation | Screening gap | Party or item missed | Penalty | Compliance system |
| Data on returned device | Privacy incident | No wiping | Breach | Secure wipe |

## 12. Interview and discussion questions

1. Why is allocation harder than forecasting in technology supply chains? *(Hint: supply is administered, not priced.)*
2. What allocation rule would you choose, and why? *(Hint: margin, strategy, fairness, contracts.)*
3. What is the difference between sell-in and sell-through? *(Hint: channel inventory.)*
4. Why does NPI need qualification gates? *(Hint: long-lead commitments before demand.)*
5. What binds a network rollout: equipment or site readiness? *(Hint: civil works and permits.)*
6. Why is a radio in a warehouse not capacity? *(Hint: it must be installed and commissioned.)*
7. What is AOG, and what does it require? *(Hint: expedite, pools, approved parts.)*
8. Why is configuration part of availability in aerospace? *(Hint: approved part for tail number.)*
9. How do you decide a last-time buy? *(Hint: life-cycle demand, storage, obsolescence.)*
10. What are the main export-control risks? *(Hint: item, party, destination, data.)*
11. How do you protect data on returned devices? *(Hint: verifiable wipe.)*
12. What metric would you use for a rollout program? *(Hint: sites on air versus kits shipped.)*

## 13. Key terms

| Term | Definition |
|---|---|
| BOM | Bill of materials |
| EMS | Electronics manufacturing services |
| NPI | New product introduction |
| Allocation | Supplier assignment of scarce supply |
| Sell-in | Shipments to channel |
| Sell-through | Sales from channel to end customer |
| AOG | Aircraft on ground |
| MRO | Maintenance, repair, and overhaul |
| Configuration | Approved part and variant for a platform |
| Airworthiness | Regulatory fitness to fly |
| Last-time buy | Final purchase before obsolescence |
| ITAR | US International Traffic in Arms Regulations |
| EAR | US Export Administration Regulations |
| eSIM | Embedded SIM |
| 3GPP | Mobile network standards body |

## 14. Sources and further reading

### Standards and regulation

- ITAR, US Directorate of Defense Trade Controls. https://www.pmddtc.state.gov/
- EAR, US Bureau of Industry and Security. https://www.bis.doc.gov/
- FAA airworthiness. https://www.faa.gov/
- EASA. https://www.easa.europa.eu/
- 3GPP. https://www.3gpp.org/
- GSMA eSIM. https://www.gsma.com/
- IPC standards. https://www.ipc.org/
- JEDEC. https://www.jedec.org/

### Reports and official sources

- US Department of Commerce, semiconductor supply chain reports. https://www.commerce.gov/
- European Commission, chips and telecom policy. https://ec.europa.eu/
- GSMA Mobile Economy reports. https://www.gsma.com/
- IATA and airline MRO reports. https://www.iata.org/

### Peer-reviewed and professional

- Sheffi, Y., *The Power of Resilience*.
- Simchi-Levi, D., et al., research on supply chain risk and disruption.
- Lee, H. L., "The triple-A supply chain."
- Aerospace MRO and spares research in operations management journals.

### Industry

- SEMI, semiconductor supply chain data. https://www.semi.org/
- Airlines for America and MRO provider reports.
- Operator annual reports for network rollout disclosures.

> **Next:** [Module 41](41_maritime_rail_air_operations.md) moves from the commercial network to the physical operations of ports, rail corridors, and air cargo.
