# Module 38 — Process Industries, Commodities, and Bulk Supply Chains

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Manufacturing and Commodities tracks
> **Prerequisites:** Modules 11, 15, 16, 20
> **Tools:** Python, linear programming, process simulation, commodity price data
> **Industry question:** "You cannot store a chemical in a box, and the price moves every hour — how do you plan?"

---

## 1. Why industry cares

Process industries — chemicals, refining, cement, steel, paper, food processing, pharmaceuticals, and utilities — make products that flow rather than count. A refinery does not make 400,000 barrels; it makes a slate of products from a barrel. A cement plant does not hold inventory in bins; it holds clinker in silos.

Four features change the planning problem:

- **The product is often a commodity.** Differentiation is limited, so cost, yield, and logistics decide the margin.
- **Capacity is continuous and expensive to stop.** A furnace or cracker runs 24/7; a shutdown costs days and millions.
- **Yield and quality are uncertain.** Feedstock varies, and output must meet a specification.
- **Price is exogenous.** Commodity prices move with markets, not with your forecast.

The analytics problem is **yield, blend, campaign, and logistics optimisation under commodity price and quality uncertainty**.

## 2. Learning outcomes

By the end of this module, you can:

- Distinguish continuous, batch, and discrete manufacturing and the planning logic of each.
- Model a process with yield, by-products, and quality specifications.
- Build a blend optimisation to meet a specification at minimum cost.
- Plan campaigns and changeovers when setup costs are large.
- Manage silo, tank, and bulk storage constraints.
- Explain commodity price exposure, basis risk, and hedging.
- Model bulk transport (vessel, rail, barge, pipeline) and its contract terms.
- Analyse capacity, throughput, and debottlenecking.
- Account for energy, water, and emissions in process decisions.
- Assess process safety and environmental obligations.

## 3. Mental model

```text
   FEEDSTOCK ──> PROCESS ──> PRODUCT / BY-PRODUCT ──> BLEND ──> BULK LOGISTICS ──> CUSTOMER
   (ore, crude,   (reactor,    (chemical, fuel,       (spec)    (vessel, rail,
    crop, gas)     kiln, mill)  clinker, pulp)                    barge, pipeline)
        │             │                │                    │
     quality       yield and        storage               contract
     and price     energy           (silo/tank)           and laytime
```

The physical flow is continuous, but the planning decisions are discrete: which campaign, which blend, which vessel, which contract.

### 3.1 The three production modes

| Mode | Example | Planning unit | Key constraint |
|---|---|---|---|
| Continuous | Refinery, cement kiln | Rate and campaign | Shutdown and changeover |
| Batch | Specialty chemicals, food | Batch and recipe | Cleaning and sequence |
| Discrete | Assembly | Order and line | Routing and stations |

A process business may run all three: continuous upstream, batch blending, discrete packaging.

## 4. Core concepts

### 4.1 Yield and mass balance

\[
\text{Output} = \text{Input} \times \text{yield}
\]

Mass balance must close across products, by-products, and losses:

\[
\text{Input mass} = \text{product} + \text{by-product} + \text{losses}
\]

Yield is not a single number. It varies with feedstock quality, operating conditions, and product mix. A planning model that uses a fixed yield will misstate both volume and cost.

### 4.2 Quality specifications and blending

Products must meet a specification — grade, purity, moisture, protein, sulphur, viscosity. Blending combines streams to meet the spec at minimum cost.

A simple two-stream blend:

\[
q_A x + q_B (1-x) \geq Q_{\min}
\]

\[
\text{cost} = c_A x + c_B (1-x)
\]

With more streams and multiple specs, this becomes a linear program.

### 4.3 Campaign planning and changeover

When setup or cleaning costs are high, production runs in campaigns:

- Decide the sequence and length of campaigns.
- Balance changeover cost against inventory and service.
- Respect storage capacity between stages.

This is the **lot-sizing and sequencing** problem in a continuous setting, often solved with MILP or heuristics.

### 4.4 Storage and inventory

Bulk inventory is constrained by physical capacity, not by units:

- **Tanks:** volume, segregation, temperature, vapour.
- **Silos:** capacity, segregation, moisture.
- **Yards:** bulk solids, weather, quality degradation.

Inventory policy must consider **minimum operational inventory** (tank heels, pipeline fill) and **quality segregation**.

### 4.5 Commodity price and basis risk

Commodity exposure has two parts:

- **Flat price risk:** the absolute market price.
- **Basis risk:** the difference between the local price and the benchmark.

Hedging the benchmark does not remove basis risk. A producer who hedges flat price but sells at a local discount can still lose money when the basis moves.

### 4.6 Bulk logistics and contracts

| Mode | Contract feature | Risk |
|---|---|---|
| Vessel | Laytime, demurrage, freight rate | Port delay and demurrage |
| Rail | Tariff, car supply | Car availability |
| Barge | Rate, draft limits | River level |
| Pipeline | Throughput, nomination | Capacity and quality |
| Truck | Rate, waiting time | Loading delays |

**Laytime** is the allowed loading or discharging time; excess is **demurrage**. A cargo that loads slowly can cost more than the freight.

### 4.7 Capacity and debottlenecking

Throughput is set by the bottleneck. The analytics steps:

1. Measure actual throughput and downtime.
2. Identify the constraint (reactor, kiln, dryer, packaging).
3. Evaluate debottlenecking options against cost and payback.
4. Check whether the next constraint moves.

A capacity increase that creates a new bottleneck elsewhere is not a capacity increase.

### 4.8 Energy, water, and emissions

Process industries are energy- and water-intensive. Decisions must include:

- Energy cost per tonne.
- Emissions per tonne, including scope 1 and 2.
- Water withdrawal and discharge.
- Waste and by-product handling.
- Carbon border rules for exports.

A production plan that minimises cost may not minimise emissions, and the two objectives can conflict.

### 4.9 Worked example — blend to specification

A feed mill blends two protein sources. Source A is 30% protein at €400 per tonne; source B is 45% protein at €520 per tonne. The product must be at least 36% protein.

Let x be the share of A:

\[
30x + 45(1-x) = 36
\]

\[
45 - 15x = 36 \Rightarrow x = 0.6
\]

The minimum-cost blend is 60% A and 40% B:

\[
0.6 \times 400 + 0.4 \times 520 = €448 \text{ per tonne}
\]

If A rises to €470, the blend cost rises to €490. The model shows the cost sensitivity to each input, which is the basis for procurement and hedging decisions.

### 4.10 Worked example — campaign versus changeover

A plant can run product X for a long campaign or switch to Y. A changeover costs €30,000 and two days of lost output. Holding Y in inventory costs €0.40 per unit per week.

If switching weekly avoids holding 50,000 units of Y for a week:

\[
\text{Holding cost} = 50{,}000 \times 0.40 = €20{,}000
\]

The changeover at €30,000 is more expensive than the holding cost, so weekly switching is not justified. If demand for Y is 200,000 units per week, holding cost is €80,000 and switching pays back.

The decision depends on volume, changeover cost, and storage capacity.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Feedstock quality | Lab / assay | Yield and blend |
| Process conditions | Historian / SCADA | Yield and energy |
| Production | MES | Output and downtime |
| Inventory | Tank and silo gauges | Capacity and availability |
| Quality | LIMS | Specification compliance |
| Price | Market feed | Valuation and hedging |
| Freight | Contracts / index | Logistics cost |
| Emissions | Metering / factors | Reporting |

### 5.2 Standards

- **ISO 9001** quality management; **ISO 14001** environmental.
- **ISO 50001** energy management.
- **OSHA Process Safety Management** and **EU Seveso** for major hazard sites.
- **REACH** for chemicals in the EU.
- **ISO 22000** and **HACCP** for food processing.
- **GHG Protocol** for emissions accounting.
- **ASTM** and **ISO** test methods for product specifications.
- Commodity exchange rules for hedging and delivery.

### 5.3 Workflow

1. Procure feedstock with quality and price terms.
2. Plan campaigns and blends.
3. Operate and monitor yield.
4. Test quality and release.
5. Store in tanks or silos.
6. Move by bulk transport.
7. Hedge price exposure.
8. Report emissions and compliance.

## 6. Industry case studies

### 6.1 Refinery planning and crude selection

**Context:** A refinery chooses crude slates and product mixes.
**Intervention:** Linear programming models for crude selection and blending.
**Evidence:** Industry practice and academic literature (grade B/C).
**Outcome:** LP-based planning is standard in refining and delivers large margin improvements.
**What transferred:** blend optimisation is one of the highest-return analytics applications in process industries.
**What did not:** specific margins.

### 6.2 Cement and clinker campaigns

**Context:** A cement plant balances kiln campaigns, clinker storage, and grinding.
**Intervention:** Campaign scheduling and silo management.
**Evidence:** Industry case reporting (grade C/D).
**Outcome:** Campaign length and storage capacity interact; short campaigns raise changeover cost, long campaigns risk quality and storage overflow.
**What transferred:** the campaign–storage trade-off is the core planning decision.
**What did not:** specific costs.

### 6.3 Commodity hedging failure

**Context:** A producer hedged flat price but not basis.
**Intervention:** Financial hedge on the benchmark.
**Evidence:** Documented commodity risk cases (grade C).
**Outcome:** Basis moved against the producer, and the hedge did not protect the realised margin.
**What transferred:** hedge the exposure you actually have, not the index you can trade.
**What did not:** specific prices.

### 6.4 A composite failure — the silo that filled

**Context:** A plant ran a long campaign to reduce changeovers.
**Intervention:** No storage check.
**Evidence:** Composite teaching case.
**Outcome:** Product silos filled, the plant had to stop, and the changeover happened anyway at a worse time.
**What transferred:** campaign plans must respect downstream storage and quality segregation.
**What did not:** specific volumes.

## 7. Labs

### 7.1 Lab 38.1 — Yield and mass balance

Build a mass-balance model for a two-product process with by-products and losses. Test sensitivity to feedstock quality.

**Deliverable:** Model and sensitivity analysis.

### 7.2 Lab 38.2 — Blend optimisation

Solve a multi-stream blend problem with two quality specifications and minimum cost. Compare with a simple rule.

**Deliverable:** LP model and recommendation.

### 7.3 Lab 38.3 — Campaign scheduling

Plan campaigns for three products with changeover costs and storage limits. Compare two policies.

**Deliverable:** Schedule and cost comparison.

### 7.4 Lab 38.4 — Commodity hedge

Given price and basis history, evaluate a hedge strategy and quantify the residual risk.

**Deliverable:** Hedge analysis and risk report.

### 7.5 Lab 38.5 — Debottlenecking

Using throughput and downtime data, identify the bottleneck and evaluate two debottlenecking options.

**Deliverable:** Analysis and business case.

## 8. Assignment / project

Design a planning model for one process supply chain.

**Required components:**

1. Process flow and mass balance.
2. Yield and quality model.
3. Blend or campaign optimisation.
4. Storage and logistics constraints.
5. Commodity price and hedge analysis.
6. Capacity and debottlenecking.
7. Energy and emissions.
8. Safety and environmental compliance.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Process model | 20% | Describes flow | Mass balance with yield | Quality and by-product modelling |
| Optimisation | 25% | Simple rule | LP or MILP | Multi-spec with constraints |
| Campaign and storage | 20% | Schedule | Changeover trade-off | Storage and quality segregation |
| Commodity risk | 15% | Names price | Hedge analysis | Basis and residual risk |
| Capacity | 10% | Reports throughput | Bottleneck analysis | Debottlenecking business case |
| Safety and environment | 10% | Mentions compliance | Emissions accounting | Process safety integration |

## 10. Regulatory, ethical, and commercial lens

- **Process safety:** major hazard sites are regulated; supply decisions cannot compromise safety.
- **Environmental permits:** emissions, discharge, and waste limits constrain operations.
- **Chemicals regulation:** REACH and equivalent regimes require registration and disclosure.
- **Product quality:** food, pharma, and feed have strict specifications and traceability.
- **Market conduct:** commodity trading is regulated; insider dealing and market manipulation are offences.
- **Community impact:** emissions and traffic affect local populations.
- **Resource use:** water and energy scarcity are material risks.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Fixed yield | Plan misses output | Feedstock varies | Volume and cost error | Yield model by quality |
| Blend by rule | Cost drift | No optimisation | Margin loss | LP blending |
| Long campaign | Storage full | Changeover avoidance | Forced stop | Storage-aware scheduling |
| Hedge mismatch | Margin volatility | Basis risk | Loss | Hedge actual exposure |
| Bottleneck blind spot | Throughput flat | Constraint moved | Wasted capex | Constraint analysis |
| Quality segregation ignored | Off-spec product | Mixing streams | Rejection | Segregation rules |
| Laytime ignored | Demurrage | Slow loading | Cost | Port and contract planning |
| Emissions excluded | Compliance gap | Not measured | Penalty | Carbon accounting |
| Safety trade-off | Near miss | Speed over safety | Incident | Safety case |

## 12. Interview and discussion questions

1. How does process manufacturing differ from discrete manufacturing? *(Hint: continuous flow, yield, quality spec.)*
2. What is yield, and why is a fixed yield dangerous? *(Hint: feedstock and condition variation.)*
3. How do you optimise a blend? *(Hint: linear programming with quality constraints.)*
4. When is a long campaign better than frequent changeovers? *(Hint: setup cost versus holding and storage.)*
5. What is basis risk? *(Hint: local price versus benchmark.)*
6. How do you hedge a commodity exposure? *(Hint: hedge the actual exposure, not the index.)*
7. What is laytime and demurrage in bulk shipping? *(Hint: allowed time and excess charge.)*
8. How do you find the bottleneck? *(Hint: throughput, downtime, and constraint analysis.)*
9. Why can a capacity increase fail? *(Hint: the constraint moves.)*
10. How do you include emissions in a production plan? *(Hint: cost and carbon objectives can conflict.)*
11. What are the main process-safety obligations? *(Hint: PSM, Seveso, permits.)*
12. What makes bulk inventory different from unit inventory? *(Hint: tank and silo capacity, segregation, quality.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Continuous production | Non-stop flow process |
| Batch production | Discrete batches with recipes |
| Yield | Output per unit of input |
| Mass balance | Conservation of mass across the process |
| Blend | Mixing streams to meet a specification |
| Campaign | Production run of one product |
| Changeover | Setup between products |
| Basis risk | Local price minus benchmark price |
| Laytime | Allowed time for loading or discharging |
| Demurrage | Charge for exceeding laytime |
| Debottlenecking | Removing a capacity constraint |
| Clinker | Intermediate product in cement |
| Scope 1 / 2 emissions | Direct and energy-indirect emissions |
| Process safety management | Regime for major hazard facilities |
| Specification | Required product quality attributes |

## 14. Sources and further reading

### Standards and regulation

- ISO 9001, 14001, 50001, 22000. https://www.iso.org/
- OSHA Process Safety Management. https://www.osha.gov/
- EU Seveso Directive 2012/18/EU. https://eur-lex.europa.eu/
- EU REACH. https://echa.europa.eu/
- GHG Protocol. https://ghgprotocol.org/
- ASTM and ISO test methods for product specifications.

### Reports and official sources

- IEA, energy and industry data. https://www.iea.org/
- US EIA, energy statistics. https://www.eia.gov/
- World Bank, commodity markets outlook. https://www.worldbank.org/
- FAO, food and agriculture data. https://www.fao.org/

### Peer-reviewed and professional

- Williams, H. P., *Model Building in Mathematical Programming*.
- Pochet, Y., Wolsey, L. A., *Production Planning by Mixed Integer Programming*.
- Shapiro, J. F., *Modeling the Supply Chain*.
- Gary, J. H., Handwerk, G. E., *Petroleum Refining: Technology and Economics*.

### Industry

- American Chemistry Council, industry data. https://www.americanchemistry.com/
- World Cement Association and industry reports.
- Commodity exchange rulebooks (for example CME, ICE) for hedging mechanics.

> **Next:** [Module 39](39_mining_metals_forestry.md) moves upstream to the resource, where grade and logistics set the product that everything else transforms.
