# Module 01 — Supply Chain Foundations for Data Scientists

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 34 hours (14 lecture, 10 lab, 10 project)
> **Track fit:** Common core (all three tracks)
> **Prerequisites:** Module 00; comfort with spreadsheets and basic algebra
> **Tools:** Python, Jupyter, DuckDB, a spreadsheet, a drawing tool for the decision map
> **Industry question:** "Which decision does this analysis change, who owns it, at what horizon, and what does it cost when the answer is wrong?"

---

## 1. Why industry cares

A supply chain is not a diagram of boxes and arrows. It is a set of **decisions made repeatedly under constraints, with money and service consequences that land on specific people**. A data scientist who cannot name the decision, the planning bucket, the unit of measure, and the cost of being wrong will build models that are technically sound and commercially irrelevant. The recurring failure is not bad mathematics; it is **category error** — optimizing the wrong objective at the wrong grain for the wrong horizon.

| Foundation concept | Decision it changes | What happens when it is ignored |
|---|---|---|
| Service metric definition | How much safety stock to hold | Model maximizes unit fill rate while the customer contract measures OTIF at order level |
| Inventory type | Whether a buffer is needed at all | Safety stock added on top of pipeline inventory that already covers lead time |
| Cost taxonomy | Order quantity, mode, supplier choice | Freight savings booked while holding and markdown costs rise off-budget |
| Working capital | Whether to fund inventory or terms | Project "saves cost" but consumes cash the CFO did not approve |
| Lead-time definition | Reorder point, promise date | Forecast timed to order date instead of availability date, so it is always late |
| Incoterm | Landed cost, risk ownership | Duty, insurance, and port charges omitted from the margin calculation |
| Capacity and MOQ | Feasibility of the plan | Recommendation is mathematically optimal and operationally impossible |
| Metric conflict | Which number is the true objective | Transport, warehousing, and finance each optimize a local metric and the total worsens |

The 2026 context sharpens the stakes. AI forecasting is widely piloted but narrowly embedded: vendor and consultancy surveys in 2025–2026 consistently show adoption intent far above measured value, with most recommendations still reviewed by a human before action (see `_research/industry_brief.md` §1). Resilience, tariffs, and carbon reporting have moved from side constraints to board-level objectives. In that environment, the scarce skill is not model building. It is **translating a business decision into a data, method, and measurement specification** — and defending it to a planner, a finance partner, and an operator in the same meeting.

> **Industry interpretation.** Every model you build is a claim about a decision. If you cannot state the decision, the claim is unfalsifiable and the project will be judged on slide quality rather than outcome.

## 2. Learning outcomes

By the end of this module, you can:

- Diagram the physical, information, and financial flows of a supply chain and identify the record each flow leaves behind.
- Classify a supply chain decision by horizon, owner, cadence, and data grain, and write a one-page decision statement.
- Distinguish six inventory types and compute the relevant quantity for each in a worked case.
- Define and compute cycle service level (CSL), unit fill rate, line fill rate, OTIF, perfect order rate, and lead time, and explain why they diverge.
- Build a landed-cost and total-cost-of-ownership (TCO) estimate from Incoterms 2020 terms, freight, duty, insurance, and carrying cost.
- Compute DIO, DSO, DPO, and the cash-to-cash cycle, and translate an inventory reduction into released cash.
- Read a bill of materials (BOM), a routing, and a material requirements planning (MRP) explosion, and check a plan against capacity.
- Explain and measure the bullwhip effect and connect it to a specific information-sharing or policy intervention.
- Diagnose metric conflicts, propose a measurement set that does not reward local optimization, and produce a decision map that later modules can attach models to.

## 3. Mental model

### 3.1 Three flows, one network

```
                       INFORMATION FLOW
        orders, forecasts, ASNs, invoices, events, master data
        ────────────────────────────────────────────────►
  SUPPLIER ──► PLANT ──► DC / CROSS-DOCK ──► STORE / CUSTOMER ──► RETURN
        ◄────────────────────────────────────────────────
                       PHYSICAL FLOW (and reverse flow)

        ◄────────────────────────────────────────────────
              FINANCIAL FLOW: payment terms, invoices, credit
```

Each flow has a different clock. Physical flow is measured in transit days and handling hours. Information flow is measured in message latency and refresh frequency. Financial flow is measured in payment terms and the cash conversion cycle. A model that uses data from one clock to make a decision on another will fail in a way that looks like a modeling error but is a timing error.

### 3.2 The decision stack

| Layer | Horizon | Typical decision | Data grain | Cadence | Owner |
|---|---|---|---|---|---|
| Strategic | 1–10 years | Network, footprint, automation, sourcing country | Annual by lane and site | Annual / event-driven | Executive, supply chain VP |
| Tactical | 1–18 months | S&OP, category plan, capacity, inventory policy, contracts | Monthly by family and site | Monthly | Planning director, finance |
| Operational | 1–13 weeks | Replenishment, production scheduling, deployment, promotions | Weekly or daily by SKU–location | Weekly / daily | Demand and supply planners |
| Execution | Hours–days | Order release, pick, route, dispatch, exception | Event or transaction | Real time | Operations, warehouse, transport |

The decision stack is the single most useful artifact in this module. It forces three questions before any modeling starts: **At what grain? At what frequency? With what consequence?** A demand model at item–store–day grain feeding a monthly family-level S&OP is over-engineered; a monthly family forecast feeding daily replenishment is unusable.

### 3.3 The anchor statement

Every project should be expressible in one sentence of this form:

> "For **[decision owner]**, at **[cadence]**, we will choose **[decision]** using **[data grain]** and **[method]**, to improve **[primary metric]** without degrading **[guardrail metric]**, and we will know it worked when **[evaluation design]**."

If a sentence cannot be completed, the project is not ready for data.

## 4. Core concepts

### 4.1 Physical, information, and financial flows

| Flow | Typical objects | Record left behind | Latency | Common defect |
|---|---|---|---|---|
| Physical | Cases, pallets, containers, returns | Shipment, receipt, inventory snapshot | Hours to weeks | Events recorded at wrong time or location |
| Information | Purchase order (PO), advance ship notice (ASN), forecast, plan | EDI message, API event, flat file | Seconds to days | Version conflict between systems |
| Financial | Invoice, payment, credit note, duty | Accounts payable/receivable ledger | Days to months | Goods receipt not matched to invoice |
| Carbon / compliance | Activity data, emission factors, chain of custody | Scope 1/2/3 inventory, due-diligence record | Monthly to annual | Spend-based estimate presented as measured |

The information flow is usually the weakest. Physical goods are scanned; financial entries are audited; information about intent (forecast, plan, commitment) is frequently overwritten in place, leaving no version history. That is why so many analytics projects cannot reconstruct what the planner believed at the time of the decision.

### 4.2 Network entities

| Entity | Role | Typical system of record | Key decisions | Natural data grain |
|---|---|---|---|---|
| Supplier | Provides material, components, or finished goods | Supplier relationship management (SRM), ERP | Allocation, PO release, expedite | Supplier–item–week |
| Plant | Converts inputs to outputs | ERP, manufacturing execution system (MES) | Schedule, lot size, changeover | Work center–item–shift |
| Distribution center (DC) | Stores and deploys inventory | Warehouse management system (WMS) | Replenish, pick, deploy | DC–item–day |
| Cross-dock | Transfers without storage | WMS / yard system | Dock scheduling, consolidation | Inbound–outbound pair |
| Store | Serves walk-in demand | Point of sale (POS), store inventory | Replenish, assort, price | Store–item–day |
| Customer | Consumes or resells | Order management system (OMS), CRM | Promise, allocate, prioritize | Customer–order line |
| Third-party logistics provider (3PL) | Operates logistics on contract | 3PL WMS/TMS, EDI | Service level, cost per unit | Client–warehouse–day |
| Carrier | Moves freight | Transportation management system (TMS) | Mode, route, tender | Lane–shipment |

Two entities are frequently missing from analytics models and cause most of the surprises: the **3PL**, whose system holds the true inventory position but reports late, and the **contract manufacturer**, whose capacity is not in your ERP at all.

### 4.3 Planning processes

| Process | Question | Horizon | Cadence | Output | Not the same as |
|---|---|---|---|---|---|
| Demand planning | What will customers want? | 1–18 months | Monthly / weekly | Statistical baseline plus consensus forecast | Target, budget, commitment |
| Demand sensing | What is demand doing now? | 0–4 weeks | Daily | Short-horizon adjustment | A replacement for the baseline |
| Supply planning | Can we meet it, and how? | 1–18 months | Monthly / weekly | Feasible supply plan, gaps, constraints | The demand forecast |
| Execution planning | What do we release today? | 0–14 days | Daily / shift | Orders, schedules, routes | The tactical plan |
| Sales and operations planning (S&OP) | Do demand and supply agree, and what do we do about the gap? | 1–18 months | Monthly | One number plan, decisions, assumptions | A forecasting meeting |
| Integrated business planning (IBP) | Does the plan meet financial targets? | 1–3 years | Monthly / quarterly | Financially reconciled plan | S&OP with a different name |

Distinguish four words used interchangeably in bad projects: **forecast** (a probabilistic statement about what is likely), **plan** (what the organization intends to do given constraints), **target** (what someone is accountable for), and **commitment** (what has been promised to a customer or supplier). A forecast above target is not good news; it is a gap. A plan above capacity is not a plan.

### 4.4 Inventory types

| Type | Definition | Formula (constant demand) | Who owns the decision | Common analytics error |
|---|---|---|---|---|
| Cycle stock | Inventory from ordering in batches | Q / 2 | Replenishment planner | Optimizing Q without setup or MOQ |
| Safety stock | Buffer against demand and lead-time uncertainty | z × σ over protection interval | Inventory planner | Using demand σ instead of lead-time-demand σ |
| In-transit / pipeline | Goods shipped but not received | d × transit time | Transport and supply planning | Counting it as available |
| Seasonal build | Production ahead of a peak | Peak demand minus peak capacity, accumulated | S&OP | Treating it as excess |
| Decoupling / work-in-process | Buffer between process steps | Set by line balance and variability | Production | Ignoring it in lead-time calculations |
| Obsolete / excess / slow-moving | Inventory with low or no future demand | Age or forecast coverage thresholds | Category and finance | Discovering it at write-off |
| Consignment | Stock physically present, supplier-owned | Not on your balance sheet | Procurement | Including it in available-to-promise |
| Returns / repair | Customer returns and repairables | Return rate × sales, plus repair cycle | Reverse logistics | Forecasting gross demand only |

Worked example — pipeline inventory. A retailer sells 8,000 units per week of an imported item. Ocean transit is 20 days; air transit is 3 days. Pipeline inventory at sea is 8,000 ÷ 7 × 20 = **22,857 units**. By air it is 8,000 ÷ 7 × 3 = **3,429 units**. At a unit cost of USD 12.00, the ocean pipeline ties up USD 274,286 versus USD 41,143 by air — a difference of **USD 233,143** in inventory, before any safety stock. The air freight premium must be compared with the carrying cost of that difference, not with the freight bill alone.

### 4.5 Service metrics

Definitions matter because they are not interchangeable, and contracts usually name one of them.

| Metric | Definition | Formula | Grain | Typical use |
|---|---|---|---|---|
| Cycle service level (CSL) | Probability of no stockout in a replenishment cycle | 1 − (cycles with stockout / cycles) | Item–location–cycle | Safety-stock setting |
| Unit fill rate | Share of units demanded that ship | Units shipped / units demanded | Item–location–period | Inventory policy, customer service |
| Line fill rate | Share of order lines shipped complete | Complete lines / order lines | Order line | Order management |
| Order fill rate | Share of orders shipped complete | Complete orders / orders | Order | Commercial reporting |
| On-time in-full (OTIF) | Orders delivered on time and complete | OTIF orders / orders | Order | Retail vendor scorecards |
| Perfect order rate | OTIF × correct documentation × damage-free | Product of components | Order | Customer experience |
| Lead time | Elapsed time from trigger to availability | See Module 01A §4.4 | Order / lane | Reorder point, promise |

Worked example. In a period, a DC receives 1,000 units of demand across 500 order lines and 100 orders. It ships 980 units; 480 lines ship complete; 92 orders arrive on time and complete; 98% of orders have correct documents; 99% arrive damage-free.

- Unit fill rate = 980 / 1,000 = **98.0%**
- Line fill rate = 480 / 500 = **96.0%**
- OTIF = 92 / 100 = **92.0%**
- Perfect order = 0.92 × 0.98 × 0.99 = **89.3%**

The gap between 98.0% unit fill and 89.3% perfect order is the gap between an inventory metric and a customer metric. A model tuned to unit fill rate will look excellent while one in nine orders is a service failure.

> **Industry interpretation.** CSL is a property of a replenishment policy; fill rate is a property of the customer experience. They move together but not proportionally, and the difference is largest when order quantities are small relative to the replenishment quantity.

### 4.6 Cost taxonomy

| Cost | Definition | Driver | Behaviour | Often omitted |
|---|---|---|---|---|
| Holding (carrying) | Cost of owning inventory for a period | Unit value, capital rate, storage, insurance, obsolescence, shrink | Proportional to average inventory | Obsolescence and markdown |
| Ordering / setup | Cost per order or changeover | Transactions, admin, machine setup | Per order | Planner time |
| Shortage | Cost of not serving demand | Lost margin, expedite, penalties, goodwill | Per unit short | Long-run customer loss |
| Transportation | Cost of moving goods | Mode, distance, weight, cube, accessorials | Per shipment | Detention, rework, returns |
| Landed cost | All costs to place goods at destination | FOB, freight, duty, insurance, fees | Per unit | Duty drawback, FX |
| Total cost of ownership (TCO) | Landed cost plus lifecycle costs | Quality, reliability, switching, risk, carbon | Per unit over life | Supplier risk, carbon price |

Worked landed cost. A purchase of 1,200 units has a free-on-board (FOB) value of USD 48,000 (USD 40.00 per unit), ocean freight USD 3,600, insurance at 0.5% of FOB plus freight, duty at 6.5% of FOB, port fees USD 450, drayage USD 900, and brokerage USD 150.

| Component | Calculation | Amount (USD) |
|---|---|---:|
| FOB goods | 1,200 × 40.00 | 48,000 |
| Ocean freight | given | 3,600 |
| Insurance | 0.005 × (48,000 + 3,600) | 258 |
| Duty | 0.065 × 48,000 | 3,120 |
| Port fees | given | 450 |
| Drayage | given | 900 |
| Brokerage | given | 150 |
| **Landed cost** | sum | **56,478** |
| **Landed cost per unit** | 56,478 ÷ 1,200 | **47.07** |

Landed cost is 17.7% above FOB (56,478 ÷ 48,000 = 1.1766). Add 45 days of carrying at 25% per year: 56,478 × 0.25 × 45 ÷ 365 = **USD 1,741**. Add 1.2% scrap (14.4 units × 47.07 = **USD 678**). TCO before risk and carbon is **USD 58,897**, or **USD 49.08 per unit**. A supplier quoting USD 40.00 FOB and one quoting USD 41.00 with 10 days shorter transit are not comparable until this arithmetic is done.

### 4.7 Working capital

| Metric | Formula | Meaning | Lever |
|---|---|---|---|
| Days inventory outstanding (DIO) | Average inventory ÷ COGS × 365 | How long cash sits in stock | Inventory policy, assortment, forecasting |
| Days sales outstanding (DSO) | Accounts receivable ÷ revenue × 365 | How long customers take to pay | Terms, collections, channel mix |
| Days payable outstanding (DPO) | Accounts payable ÷ COGS × 365 | How long you take to pay suppliers | Terms, supply-chain finance |
| Cash-to-cash cycle (CCC) | DIO + DSO − DPO | Days between paying for and collecting on goods | All three, but DPO has limits |

Worked example. A retailer has revenue USD 1.2bn, COGS 70% of revenue (USD 840m), average inventory USD 115m, accounts receivable USD 60m, accounts payable USD 95m.

- DIO = 115 ÷ 840 × 365 = 49.97 → **50.0 days**
- DSO = 60 ÷ 1,200 × 365 = 18.25 → **18.3 days**
- DPO = 95 ÷ 840 × 365 = 41.28 → **41.3 days**
- CCC = 50.0 + 18.3 − 41.3 = **27.0 days**

Cash tied up in the cycle ≈ 27.0 × (1,200 ÷ 365) = 27.0 × 3.288 = **USD 88.8m**. If DIO falls from 50.0 to 45.0 days, released cash = 5.0 × (840 ÷ 365) = 5.0 × 2.301 = **USD 11.5m**. That is the number a CFO will fund a project against — not a percentage improvement in forecast error.

Stretching DPO is the cheapest-looking lever and the most dangerous: it transfers cost to suppliers, degrades their reliability, and is increasingly visible to regulators and large buyers' own due-diligence obligations.

### 4.8 Manufacturing: BOM, routing, MRP, capacity

A **bill of materials (BOM)** states what a product is made of. A **routing** states the operations, work centers, and standard times to make it. **Material requirements planning (MRP)** explodes demand through the BOM net of inventory and scheduled receipts, offset by lead time. **Capacity requirements planning (CRP)** checks whether the resulting load fits work-center capacity.

Worked MRP explosion. Product A has demand of 100 units in week 5. BOM: A = 2 × B + 3 × C; B = 1 × D + 2 × E. On hand: A 0, B 20, C 50, D 100, E 200. Scheduled receipt: 30 B in week 2. Lead times: A 1 week, B 2 weeks, C 1 week, D 1 week, E 1 week.

| Item | Gross requirement | On hand + scheduled | Net requirement | Lead time | Planned order release |
|---|---:|---:|---:|---:|---|
| A | 100 (wk 5) | 0 | 100 | 1 wk | wk 4 |
| B | 200 (wk 4) | 20 + 30 = 50 | 150 | 2 wk | wk 2 |
| C | 300 (wk 4) | 50 | 250 | 1 wk | wk 3 |
| D | 150 (wk 2) | 100 | 50 | 1 wk | wk 1 |
| E | 300 (wk 2) | 200 | 100 | 1 wk | wk 1 |

This is a **time-phased** plan; lot sizing, minimum order quantity (MOQ), or scrap would change the release quantities and may make the plan infeasible. Capacity check: if B requires 0.4 hours per unit on a work center with 3 machines × 2 shifts × 7.5 hours × 5 days = 225 hours per week, then 150 B consumes 60 hours in week 2 — feasible in aggregate only if the load does not collide with other products. Aggregate feasibility is not finite feasibility.

### 4.9 Transportation modes and Incoterms 2020

| Mode | Cost driver | Speed | Minimum economic shipment | Best for | Relative carbon intensity per tonne-km |
|---|---|---|---|---|---|
| Full truckload (FTL) | Truck, distance, lanes | 1–5 days domestic | One trailer | High-volume, stable lanes | Medium |
| Less-than-truckload (LTL) | Weight, class, distance | 2–7 days | 50–500 kg | Mid-volume, multi-stop | Medium–high |
| Parcel | Piece, zone, dim weight | 1–3 days | One parcel | E-commerce, small orders | High |
| Intermodal (rail + truck) | Container, lane | 4–8 days | One container | Long-haul, predictable | Low–medium |
| Ocean FCL | Container, route, season | 20–45 days | One container | Imported goods | Low |
| Ocean LCL | Cubic metres, consolidation | 25–50 days | 1 m³ | Small import lots | Low–medium |
| Air | Chargeable weight | 1–5 days | 100 kg | Urgent, high-value, perishable | Very high |
| Rail | Train, distance | 3–10 days | One wagon or block train | Bulk, inland long-haul | Low |

Carbon intensity ordering (air > road > rail > ocean) is robust; the magnitude is route- and load-specific and must be computed from a documented emission factor, not assumed.

**Incoterms 2020** (International Commercial Terms, ICC) define cost and risk transfer between buyer and seller. They do not transfer title and do not replace the contract of sale.

| Term | Mode | Risk transfers at | Seller pays | Common misuse |
|---|---|---|---|---|
| EXW | Any | At seller's premises, not loaded | Almost nothing | Buyer cannot export-clear in seller's country |
| FCA | Any | To buyer's nominated carrier | Export clearance, delivery to carrier | Used with a bill of lading that does not reflect delivery |
| CPT | Any | To first carrier | Carriage to named place | Risk and cost split at different points |
| CIP | Any | To first carrier | Carriage plus insurance (Institute Cargo Clauses A) | Assuming minimum cover, as under CIF |
| DAP | Any | At destination, ready for unloading | Carriage to named place | Buyer surprised by import clearance |
| DPU | Any | At destination, unloaded | Carriage and unloading | Only term requiring seller to unload |
| DDP | Any | At destination, ready for unloading | Everything including import duty | Seller owes duty in a country where it has no entity |
| FAS | Sea | Alongside ship | Pre-carriage to port | Used for containerized cargo |
| FOB | Sea | On board vessel | Export clearance, loading | Used for containers, where risk transfer is ambiguous |
| CFR | Sea | On board vessel | Freight to destination port | No insurance |
| CIF | Sea | On board vessel | Freight plus minimum insurance (Clauses C) | Assuming all-risk cover |

Incoterms 2020 replaced DAT with DPU, upgraded CIP insurance to Clauses A, and added an optional on-board bill of lading mechanism for FCA. Landed-cost models must map the term to the exact cost components the seller pays; a model that assumes FOB for all ocean imports will understate cost for CIF/DDP purchases and double-count freight.

### 4.10 Warehousing processes

| Process | Activity | Key metric | Typical failure |
|---|---|---|---|
| Receiving | Unload, verify against ASN/PO, record | Dock-to-stock hours | Receipt recorded before quality check |
| Putaway | Move to storage location | Putaway accuracy, travel distance | Location master data wrong |
| Storage | Hold in rack, bulk, or bin | Cube utilization, occupancy | Slotting never revisited |
| Replenishment | Move from reserve to pick face | Stockout at pick face | Replenishment triggered by visual check |
| Picking | Retrieve items per order | Lines per hour, pick accuracy | Wave design ignores cut-off times |
| Packing | Cartonize, label, document | Cost per order, damage rate | Cartonization ignores dim weight |
| Shipping | Stage, load, manifest | On-time dispatch, load fill | Trailer loaded before ASN sent |
| Returns | Receive, inspect, disposition | Return cycle time, recovery rate | Returns not visible to demand planning |

### 4.11 Sourcing and the Kraljic portfolio

| Quadrant | Profit impact | Supply risk | Strategy | Analytics focus |
|---|---|---|---|---|
| Non-critical | Low | Low | Simplify, automate, consolidate tail spend | Spend classification, catalogue compliance |
| Leverage | High | Low | Compete, volume leverage, reverse auctions | Should-cost, price variance, TCO |
| Bottleneck | Low | High | Secure supply, qualify alternatives, buffer | Risk monitoring, lead-time variability |
| Strategic | High | High | Partner, joint planning, multi-year contracts | Supplier scorecards, collaboration, dual sourcing |

The **Kraljic matrix** (Peter Kraljic, Harvard Business Review, 1983) is a segmentation tool, not a strategy. Its value in analytics is that it forces different data and different metrics per quadrant: a leverage item should be measured on price variance, a bottleneck item on availability and recovery time.

### 4.12 Sustainability and reverse logistics

| Scope | Definition | Supply chain examples | Data source | Frequent error |
|---|---|---|---|---|
| Scope 1 | Direct emissions owned or controlled | Company vehicles, on-site fuel combustion, refrigerant leakage | Fuel logs, meter readings, fleet telematics | Excluding leased assets |
| Scope 2 | Indirect emissions from purchased energy | Electricity, steam, heat for warehouses and plants | Utility invoices, supplier-specific factors | Using market-based and location-based interchangeably |
| Scope 3 | All other value-chain emissions (15 GHG Protocol categories) | Purchased goods, upstream transport, distribution, waste, business travel, use of sold products | Supplier data, spend-based estimates, transport activity | Spend-based estimates reported as supplier-measured |

Reverse logistics covers returns, repairs, refurbishment, recycling, and disposal. It is a **demand and inventory problem with negative flow**: returns create uncertain supply at uncertain times, with condition grades that determine whether the unit can be resold. Forecasting returns with the sales curve alone fails because return probability depends on product, channel, customer, and reason.

### 4.13 SCOR DS

The ASCM **Supply Chain Operations Reference Digital Standard (SCOR DS)** organizes supply chain activity into processes: Plan, Source, Make, Deliver, Return, Enable, and Orchestrate. SCOR's lasting contribution is not the taxonomy itself but three disciplines: **performance attributes** (reliability, responsiveness, agility, cost, asset management, sustainability), **process decomposition to level 3**, and **benchmarking with defined metrics**. SCOR DS was introduced to reflect digital, omnichannel, and circular operations. Version numbering and the exact process list should be checked against the current ASCM publication before citing it in a client deliverable. [VERIFY: confirm the current SCOR DS version number and the official process list at time of use.]

Use SCOR as a checklist, not a religion: it describes processes, not decisions, and says little about the commercial trade-offs that determine whether a plan will be executed.

### 4.14 The bullwhip effect

Lee, Padmanabhan, and Whang ("Information Distortion in a Supply Chain: The Bullwhip Effect," *Management Science*, 1997) identified four causes: **demand signal processing** (each tier forecasts from its own orders rather than end-customer demand), **order batching** (periodic or large-lot ordering amplifies variance), **price fluctuation** (forward buying and promotions create spikes unrelated to consumption), and **shortage gaming** (when supply is constrained, customers order more than they need).

Measurement: bullwhip ratio = variance of orders ÷ variance of demand at the downstream tier, computed on de-meaned series at the same time bucket. A common error is comparing variances at different aggregation levels, which mechanically reduces variance and flatters the ratio.

Worked example. Weekly demand at retail has standard deviation 40 units. DC orders have SD 90; plant orders SD 150; supplier orders SD 210.

| Tier | SD (units/week) | Variance ratio vs retail | Interpretation |
|---|---:|---:|---|
| Retail demand | 40 | 1.00 | Reference |
| DC orders | 90 | (90/40)² = 5.06 | Batching and forecast gaming |
| Plant orders | 150 | (150/40)² = 14.06 | Multi-tier amplification |
| Supplier orders | 210 | (210/40)² = 27.56 | Severe distortion |

The remedy is rarely "better forecasting." It is usually **sharing end-demand, shortening cycle time, reducing batch size, or stabilizing price** — four operational changes, each of which changes the data a model should use.

### 4.15 Metric conflicts

| Function | Local metric | Action it rewards | Cost it externalizes |
|---|---|---|---|
| Transport | Cost per case, trailer utilization | Full truckloads, fewer deliveries | Store inventory, stockouts, markdowns |
| Warehousing | Units per labour hour | Large batches, fewer picks | Order cycle time, accuracy |
| Manufacturing | Machine utilization, unit cost | Long runs, low changeover | Finished-goods inventory, obsolescence |
| Procurement | Purchase price variance | Cheapest supplier, larger lots | Lead time, quality, freight, risk |
| Sales | Revenue, gross sales | Forward loading, discounting | Margin, returns, inventory |
| Finance | Working capital, DIO | Cut inventory | Service, expedite cost |
| Customer service | OTIF | Expedite everything | Freight cost, margin |

Worked conflict. A transport team moves an 18-pallet order as a full 26-pallet truckload. The LTL rate for 18 pallets is USD 1,150; the FTL rate is USD 1,450. Per pallet, FTL looks cheaper (USD 55.77 versus USD 63.89). But the incremental spend is USD 300 for 8 pallets that were not ordered. Each pallet holds 48 cases, so 384 extra cases are pushed into the store. At a holding cost of USD 9.00 per case-year for 30 days, that costs 384 × 9.00 × 30 ÷ 365 = **USD 284**. Net position: **USD 16 worse**, before obsolescence and before the working-capital charge. The per-pallet metric said "save USD 8.12 per pallet"; the total-cost view says "do not do it." Both are arithmetic; only one is the right objective.

### 4.16 Service–inventory trade-off

Demand per week is normal with mean 500 units and SD 120 units; lead time is 2 weeks; review is continuous; order quantity Q = 500 units; unit cost USD 40; holding rate 25% per year (USD 10 per unit-year). Demand over the lead time has mean 1,000 units and SD 120 × √2 = 169.7 units.

| CSL | z | Safety stock (units) | Reorder point (units) | Expected shortage per cycle (units) | Fill rate | Annual holding cost of safety stock (USD) | Incremental cost per point of fill (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 90% | 1.282 | 217.5 | 1,217.5 | 8.04 | 98.39% | 2,175 | — |
| 95% | 1.645 | 279.1 | 1,279.1 | 3.55 | 99.29% | 2,791 | 684 |
| 98% | 2.054 | 348.5 | 1,348.5 | 1.27 | 99.75% | 3,485 | 1,509 |
| 99% | 2.326 | 394.8 | 1,394.8 | 0.58 | 99.88% | 3,948 | 3,562 |

Fill rate uses expected shortage per cycle = σ × L(z), where L(z) is the standard normal loss function, and fill rate = 1 − shortage ÷ Q. Moving from 95% to 98% CSL buys 0.46 points of fill rate for USD 694 a year; moving from 98% to 99% buys 0.13 points for USD 463. The marginal cost per point rises from USD 684 to USD 3,562 — a factor of five. This table is the single most useful artifact for arguing about service levels, because it converts a values dispute into a cost curve.

## 5. Data, standards, and workflow

| Data object | Minimum fields | Source | Quality check | Use in this module |
|---|---|---|---|---|
| Order | Order ID, line ID, item, location, order date, requested date, quantity, UoM | OMS, ERP | Duplicate lines, unit mismatch, date logic | Demand, fill rate, OTIF |
| Shipment | Shipment ID, origin, destination, ship date, arrival date, quantity, mode | TMS, WMS, carrier EDI | Missing ASN, late status, weight/cube outliers | Lead time, in-transit |
| Receipt | Receipt ID, PO, item, quantity, receipt date, condition | WMS, ERP | Over/under-receipt, date vs put-away | Lead time, availability |
| Inventory snapshot | Item, location, date, on-hand, allocated, in-transit, UoM | WMS, ERP | Negative stock, snapshot gaps, unit drift | DIO, service, safety stock |
| Invoice | Invoice ID, PO, amount, currency, terms, duty | ERP, accounts payable | Price variance, FX, unmatched receipts | Landed cost, DPO |
| Master data | Item, location, supplier, customer, hierarchy, calendar | ERP, PIM, MDM | Orphan records, duplicate items, stale attributes | Every join |

Workflow for the module project: select one product family and lane; extract 12 months of orders, shipments, receipts, and monthly inventory snapshots; reconcile units to a base unit; compute the metric set; write the decision statement; build the decision map; have a planner review it and record every correction. Standards to name explicitly: GS1 GTIN for items, GLN for locations, SSCC for logistics units; UN/CEFACT and ANSI X12 for messages; Incoterms 2020 for delivery terms; ISO 8601 for dates and ISO weeks; GHG Protocol for emissions; SCOR DS for process reference.

## 6. Industry case studies

### 6.1 Procter & Gamble and Walmart continuous replenishment (late 1980s)

**Context and intervention.** P&G and Walmart operated a conventional retailer–supplier relationship in which P&G forecast from Walmart's orders, not from consumer demand; orders were batched and price-promoted, producing the distortion Lee et al. later formalized. Continuous replenishment changed the information flow: Walmart shared daily store-level sales and inventory, and P&G took responsibility for replenishing within agreed service bounds.

**Evidence and outcome.** The arrangement is documented in the bullwhip literature and in operations case teaching; the widely repeated commercial results are company and case-study claims rather than a controlled evaluation. The case became the canonical demonstration that **sharing end-demand reduces amplification** and shifts the supplier from order-taking to replenishment.

**What transferred.** Data sharing plus a decision-rights change (supplier replenishes) is more powerful than a better forecast on the same data. **What did not.** The model depends on trust, a dominant retailer, and a single-category relationship; it does not generalize automatically to a long-tail supplier base. [VERIFY: exact start year of the P&G–Walmart continuous replenishment program; sources differ between the mid-1980s and 1987.]

### 6.2 Toyota and the 2011 Tōhoku earthquake (11 March 2011)

**Context and intervention.** Toyota's just-in-time (JIT) system held low inventory and depended on tightly sequenced suppliers; the earthquake and tsunami disrupted Tier 1 and, more critically, Tier 2 and Tier 3 suppliers that Toyota did not directly manage. Toyota used supplier mapping and multi-tier visibility to identify constrained components, prioritized recovery of critical suppliers, substituted parts where design allowed, and restarted production in phases.

**Evidence and outcome.** Toyota's own statements and the extensive operations literature on the event document the multi-tier problem; specific lost-production figures vary by source and should be quoted only with the source and date. Toyota recovered faster than several competitors, and the event accelerated industry interest in multi-tier supplier mapping and resilience analytics.

**What transferred.** JIT is a **flow** philosophy, not a zero-inventory policy; the vulnerability was information about Tier 2–3, not the absence of buffer per se. **What did not.** A strategy that works for a manufacturer with deep supplier relationships is not directly transferable to a retailer with thousands of small vendors.

### 6.3 Inditex/Zara and postponement (documented through 2004)

**Context and intervention.** Apparel has long design-to-shelf lead times, high forecast error, and heavy markdown exposure. Zara (Inditex) competed by shortening the design-to-store cycle, postponing dyeing and production decisions, using a responsive supply network close to the main market, and feeding store sales rapidly into design and replenishment. The operating model is described in Ferdows, Lewis, and Machuca, "Rapid-Fire Fulfillment," *Harvard Business Review*, November 2004.

**Evidence and outcome.** The HBR case documents the operating model and its economics qualitatively; it is not a controlled study, and Inditex's financial results are confounded by brand, location, and scale. Zara accepted higher unit production cost in exchange for lower forecast error, lower markdowns, and faster response — a deliberate trade of cost for flexibility.

**What transferred.** The correct comparison is **total cost including markdown and obsolescence**, not piece cost. **What did not.** The model requires a specific supply base, vertical integration, and store format; copying the IT system without the network does not reproduce it.

### 6.4 UPS ORION route optimization (rolled out 2012–2016)

**Context and intervention.** UPS needed to reduce miles and fuel in a large parcel network where drivers had discretion over route sequence. On-Road Integrated Optimization and Navigation (ORION) used heuristics and telematics data to propose route sequences, with driver and dispatcher override.

**Evidence and outcome.** UPS publicly described the rollout and claimed substantial annual savings in miles and fuel; those figures are **company claims**, not independently audited results. ORION became a reference case for optimization at scale and for the gap between an algorithmic recommendation and an executed route.

**What transferred.** Adoption design — including override capture and dispatcher workflow — determined value as much as the algorithm. **What did not.** Route optimization on a fixed demand pattern does not address demand variability or network design.

### 6.5 Maersk and IBM TradeLens shutdown (announced November 2022)

**Context and intervention.** TradeLens was a blockchain-based platform intended to share supply-chain events among carriers, shippers, forwarders, ports, and customs authorities, using a permissioned distributed ledger with industry consortium governance.

**Evidence and outcome.** Maersk and IBM announced discontinuation in November 2022; the platform was wound down in the first quarter of 2023. Public reporting cited insufficient industry-wide participation and the difficulty of achieving commercial viability. The technology worked; the **network and governance model did not** — a platform owned by one carrier struggled to attract competing carriers and their customers.

**What transferred.** Interoperability is a commercial and governance problem before it is a technical one; data-sharing products need neutral governance and a clear value split. **What did not.** The failure does not invalidate event standards (EPCIS, GS1) or the value of visibility; it invalidates a particular ownership model.

## 7. Labs

### Lab 1.1 — Build a supply-chain decision map

**Objective.** Convert a raw order-to-cash extract into a decision map with owners, horizons, cadence, grain, and metrics.

**Data.** Twelve months of orders, shipments, and monthly inventory snapshots for one product family (synthetic or provided). At minimum: order date, requested date, ship date, delivery date, item, location, quantity, UoM, unit price, unit cost.

**Steps.**
1. Profile the extract: row counts, date ranges, distinct items and locations, UoM values, missing dates.
2. Identify every decision that touched the data in the period (assortment, replenishment, deployment, transport mode, promise date).
3. For each, fill the decision-stack row: horizon, cadence, owner, grain, input data, output, metric.
4. Draw the network and mark where each decision is made.
5. Identify three decisions with no data owner and three metrics with no decision.
6. Write the anchor statement for the decision you would model first.

**Deliverable.** A two-page decision map (diagram plus table) and a one-page rationale.

**Acceptance criteria.** At least eight decisions mapped; each has an owner role (not a name), a cadence, and a primary and guardrail metric; the anchor statement names a metric and an evaluation design; no decision is described only as "forecast demand."

**Stretch.** Add a data-latency column and mark which decisions are currently made on stale data.

### Lab 1.2 — Working-capital and service diagnostic

**Objective.** Compute DIO, DSO, DPO, CCC, fill rate, OTIF, and perfect order for a business unit, and quantify the cash effect of a service or inventory change.

**Data.** Provided financial summary (revenue, COGS, inventory, AR, AP) and operational extract (orders, lines, units, on-time flags).

**Steps.**
1. Compute the four working-capital metrics and reconcile to the financial statements.
2. Compute unit fill, line fill, OTIF, and perfect order; explain every gap between them.
3. Build a bridge from a 5-day DIO reduction to released cash, using COGS per day.
4. Test the sensitivity of released cash to a service-level change using the §4.16 table.
5. Write three sentences a CFO would accept, each with a unit and a source.

**Deliverable.** A one-page diagnostic with a metric table, a cash bridge, and three findings.

**Acceptance criteria.** All formulas shown; units and currency stated; the cash bridge reconciles to the DIO change; the service and inventory metrics are computed from the same population; at least one finding contradicts the initial business narrative.

**Stretch.** Add a monthly time series and test whether the DIO improvement is seasonal or structural.

### Lab 1.3 — Service–inventory trade-off and metric choice

**Objective.** Show how the choice of service metric changes the recommended inventory and the cost curve.

**Data.** Synthetic weekly demand with a known distribution; lead time and cost parameters from the lab brief.

**Steps.**
1. Simulate 10,000 weeks of normal demand (mean 500, SD 120) and run a continuous-review policy for CSL targets of 90%, 95%, 98%, and 99%.
2. Measure CSL, unit fill rate, and average inventory from the simulation.
4. Compare simulated results with the analytical table in §4.16 and explain the differences.
5. Recompute the marginal cost per point of fill rate and identify the knee of the curve.
6. Repeat with demand following a negative binomial distribution and compare.

**Deliverable.** A notebook and a two-page memo recommending a service target with a cost justification.

**Acceptance criteria.** Simulation seed fixed; 10,000 weeks; analytical and simulated values agree within sampling error for the normal case; the memo states the metric used and why; the recommendation includes the annual cost and the service consequence.

**Stretch.** Add a lost-sales penalty and compute the cost-optimal CSL.

### Lab 1.4 — Landed cost, Incoterms, and a metric-conflict memo

**Objective.** Build a landed-cost and TCO model for two sourcing options and write a memo that resolves a cross-functional metric conflict.

**Data.** Two supplier quotes under different Incoterms, freight rates, duty rates, transit times, payment terms, and quality history.

**Steps.**
1. Map each Incoterm to the cost components the seller pays.
2. Compute landed cost per unit for both options, showing every line.
3. Add carrying cost, scrap, and expedite risk to reach TCO.
4. Quantify the transport metric conflict in §4.15 using the actual quote quantities.
5. Recommend an option and state the guardrail metric that would reverse the decision.

**Deliverable.** A landed-cost model (spreadsheet or notebook) and a 700-word memo.

**Acceptance criteria.** No double-counted freight; duty basis stated; currency and FX date stated; TCO includes at least three non-price costs; the memo names the function whose metric is being overridden and why.

**Stretch.** Add a carbon cost per tonne-km and show whether the ranking changes.

## 8. Assignment / project

**Supply-chain decision map and diagnostic (individual, 10 hours).**

Choose a real or provided product family. Produce:

1. **Decision map** — network diagram plus a table of at least ten decisions with horizon, cadence, owner role, grain, data source, primary metric, guardrail metric, and current failure mode.
2. **Diagnostic** — DIO/DSO/DPO/CCC, service metrics, landed cost for one lane, and one inventory-type breakdown.
3. **Metric-conflict memo** — 800 words identifying one conflict, quantifying it with arithmetic, and proposing a measurement change.
4. **Model-ready specification** — for the single decision you would model first: anchor statement, target variable, grain, horizon, features, constraints, evaluation design, and the three assumptions most likely to be wrong.

The grader plays the role of a planning director and a finance partner. Every number must have a unit, a period, and a source.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision framing | 25% | Lists processes, not decisions; no owner or cadence | Names decisions with horizon and owner; one metric each | Distinguishes forecast, plan, target, commitment; guardrails defined; decision rights explicit |
| Metric literacy | 20% | Uses one service metric loosely | Computes CSL, fill, OTIF correctly | Explains divergence, selects metric from contract and decision, quantifies trade-off |
| Cost and cash | 20% | Lists cost categories | Computes landed cost and DIO correctly | Builds TCO with risk and carbon; reconciles cash release to financial statements |
| Quantitative accuracy | 15% | Arithmetic and unit errors | Correct with minor presentational issues | Correct, reproducible, sensitivity-tested, assumptions stated |
| Metric-conflict analysis | 10% | Identifies a conflict descriptively | Quantifies the conflict | Proposes a measurement system and predicts behavioural response |
| Communication | 10% | Generic prose | Clear memo with units | Executive-ready; defends trade-offs; states what would change the recommendation |

## 10. Regulatory, ethical, and commercial lens

- **Contract and delivery terms.** Incoterms 2020 govern cost and risk transfer, not title. Customs value, duty, and import responsibility follow the term and the contract; misclassifying them is a compliance issue, not a modeling nuance.
- **Customs and trade.** HS classification, rules of origin, and preferential treatment change landed cost. Tariff scenarios are a planning input and must be dated.
- **Competition law.** Sharing forward-looking price or volume signals with competitors, including through a third-party platform, can constitute information exchange. Supplier and carrier data-sharing arrangements need legal review.
- **Emissions reporting.** CSRD/ESRS, CBAM, and customer requirements impose boundary, allocation, and assurance rules on emissions data. Spend-based estimates must be labelled as estimates.
- **Product safety and traceability.** DSCSA (U.S. pharmaceutical), EU FMD, and food safety rules require lot- and serial-level traceability that constrains how inventory can be commingled.
- **Labour, equity, and commercial framing.** Warehouse productivity metrics affect safety and workload, so a pick-rate target that ignores ergonomics is an ethical choice, not merely an operational one. The value case must state whether benefit is cost reduction, cost avoidance, working-capital release, revenue, or risk reduction; these are not interchangeable and finance treats them differently.

## 11. Failure modes and red flags

| Failure mode | Early signal | Consequence | Countermeasure |
|---|---|---|---|
| Optimizing a metric nobody is measured on | "The model improves MAPE" in a service review | No adoption | Start from the contract and the scorecard |
| Wrong service metric | Fill rate reported to a customer who measures OTIF | False confidence | Map every customer metric to its formula |
| Ignoring pipeline inventory | Safety stock added for the full lead time | Excess inventory | Decompose lead time and ownership |
| Landed cost missing duty or freight | Margin differs from finance by a constant | Wrong sourcing decision | Reconcile to invoice and customs entry |
| Treating a target as a forecast | Plan assumes the target will happen | Chronic shortage or excess | Separate forecast, plan, target |
| MRP without capacity check | Plan releases more hours than exist | Missed schedules, expedite | Run CRP and finite check |
| Assuming Incoterms transfer title | Inventory recorded on the wrong balance sheet | Audit finding | Contract review with legal |
| Bullwhip measured across aggregation levels | Ratio looks small | Wrong diagnosis | Same bucket, same de-meaning |
| Local metric optimization | One function celebrates, total cost rises | Cross-functional conflict | Total-cost objective and guardrails |
| Single-point-of-failure supplier invisible | Tier-2 dependency discovered after disruption | Line stoppage | Multi-tier mapping |
| Scope 3 estimate presented as measured | Supplier-specific claim with no supplier data | Greenwashing risk | Label method and boundary |

## 12. Interview and discussion questions

1. **"What is the difference between cycle service level and fill rate, and when would you choose one over the other?"**
   *Hint:* CSL is the probability of no stockout per cycle and drives safety stock; fill rate is the share of demand met and reflects customer experience. Choose CSL for policy setting, fill rate for customer commitments; they diverge most with small order quantities.

2. **"A CFO says inventory is too high. What do you ask first?"**
   *Hint:* Which inventory type, which segment, and against which service level. Separate cycle, safety, pipeline, seasonal, and obsolete; check DIO against COGS and seasonality.

3. **"How would you measure the bullwhip effect without being fooled by aggregation?"**
   *Hint:* Same time bucket, same de-meaning, variance of orders over variance of demand at the adjacent tier; report the series and the window, not a single ratio.

4. **"Why can a plan be feasible in aggregate and infeasible in practice?"**
   *Hint:* Aggregate capacity hides sequence-dependent setup, shift patterns, labour skills, and material arrival timing; finite scheduling is needed.

5. **"What costs go into landed cost that a buyer typically forgets?"**
   *Hint:* Duty, insurance, port and brokerage fees, drayage, detention, FX, and carrying cost in transit; then TCO adds quality, switching, and risk.

6. **"How would you explain cash-to-cash to a data scientist who only knows forecast metrics?"**
   *Hint:* DIO + DSO − DPO; translate a day of DIO into COGS/365 of cash; show that a 1% forecast improvement may be worth less than a 2-day inventory reduction.

7. **"Under Incoterms 2020, who pays for what under CIF versus CIP?"**
   *Hint:* Both transfer risk on board/first carrier; CIF is sea-only with minimum Clauses C insurance; CIP requires Clauses A cover and applies to any mode.

8. **"When is a full truckload the wrong decision even if the rate per pallet is lower?"**
   *Hint:* When it pushes unneeded inventory into the downstream node; compare incremental freight with holding, markdown, and working-capital cost.

9. **"What is the difference between S&OP and a forecast review?"**
   *Hint:* S&OP reconciles demand and supply and makes decisions with financial consequences; a forecast review only reviews numbers.

10. **"How would you quantify the value of a resilience investment?"**
    *Hint:* Expected loss avoided (probability × impact × recovery time), plus option value; state scenario, exposure, and time-to-recover.

11. **"Why did TradeLens fail, and what does that imply for data-sharing projects?"**
    *Hint:* Neutral governance and incentives, not technology; a carrier-owned platform struggles to attract competitors.

12. **"What would make you reject a supply-chain analytics project?"** *Hint:* No decision owner, no baseline, no access to the required grain, or a metric that cannot be measured after deployment.

## 13. Key terms

| Term | Definition |
|---|---|
| Bill of materials (BOM) | Structured list of components and quantities required to make a product. |
| Bullwhip effect | Amplification of demand variance as orders move upstream. |
| Cash-to-cash cycle (CCC) | DIO + DSO − DPO; days cash is tied up in operations. |
| Cross-dock | Flow-through facility where inbound goods are transferred to outbound without storage. |
| Cycle service level (CSL) | Probability of not stocking out during a replenishment cycle. |
| Cycle stock | Inventory held because replenishment arrives in batches. |
| Days inventory outstanding (DIO) | Average inventory ÷ COGS × 365. |
| Days payable outstanding (DPO) | Accounts payable ÷ COGS × 365. |
| Days sales outstanding (DSO) | Accounts receivable ÷ revenue × 365. |
| Fill rate | Share of demanded units that are shipped. |
| Incoterms 2020 | ICC rules defining cost and risk transfer in international sales. |
| Integrated business planning (IBP) | S&OP extended to financial reconciliation and longer horizon. |
| Kraljic matrix | Segmentation of purchases by profit impact and supply risk. |
| Landed cost | Total cost to place goods at the destination, including freight, duty, and fees. |
| Lead time | Elapsed time from a triggering event to availability for use. |
| Material requirements planning (MRP) | Time-phased explosion of demand through the BOM net of inventory. |
| On-time in-full (OTIF) | Order delivered on the requested date and complete. |
| Perfect order | OTIF plus correct documentation and damage-free delivery. |
| Pipeline (in-transit) inventory | Goods shipped but not yet received. |
| Postponement | Delaying a differentiating step until demand is known. |
| Safety stock | Buffer inventory held against demand and lead-time uncertainty. |
| SCOR DS | ASCM's digital supply chain reference model of processes and metrics. |
| S&OP | Cross-functional monthly process to balance demand and supply. |
| Total cost of ownership (TCO) | Landed cost plus lifecycle costs including quality, risk, and carbon. |

## 14. Sources and further reading

**Standards and regulation**
- ASCM, *Supply Chain Operations Reference Digital Standard (SCOR DS)*. ASCM, current edition. [VERIFY version number before citing.]
- International Chamber of Commerce, *Incoterms 2020*. ICC, 2019.
- GS1, *General Specifications*; GTIN, GLN, SSCC. GS1, current edition.
- GHG Protocol, *Corporate Accounting and Reporting Standard* and *Corporate Value Chain (Scope 3) Standard*. WRI/WBCSD.
- Regulation (EU) 2023/956 (CBAM), definitive regime from 1 January 2026; Directive (EU) 2026/470 (CSRD amendments). EUR-Lex.

**Peer-reviewed and foundational**
- Lee, H., Padmanabhan, V., Whang, S. "Information Distortion in a Supply Chain: The Bullwhip Effect." *Management Science* 43(4), 1997.
- Clark, A., Scarf, H. "Optimal Policies for a Multi-Echelon Inventory Problem." *Management Science* 6(4), 1960.
- Kraljic, P. "Purchasing Must Become Supply Management." *Harvard Business Review*, September–October 1983.
- Chopra, S., Meindl, P. *Supply Chain Management: Strategy, Planning, and Operation*. Pearson, current edition.
- Simchi-Levi, D., Kaminsky, P., Simchi-Levi, E. *Designing and Managing the Supply Chain*. McGraw-Hill, current edition.
- Silver, E., Pyke, D., Peterson, R. *Inventory Management and Production Planning and Scheduling*. Wiley, current edition.

**Industry**
- Ferdows, K., Lewis, M., Machuca, J. "Rapid-Fire Fulfillment." *Harvard Business Review*, November 2004.
- Maersk and IBM, announcement on discontinuation of TradeLens, November 2022; wind-down completed Q1 2023.
- UPS, public materials on ORION rollout, 2012–2016. Company claims; not independently audited.
- Toyota Motor Corporation, statements on production recovery after the 11 March 2011 Tōhoku earthquake.

**Datasets and tools**
- M5 (Walmart) retail dataset, Kaggle; M4; Corporación Favorita; Rossmann. See Module 27. DuckDB documentation for local analytical SQL, https://duckdb.org/docs/ (accessed 27 October 2026). Course source pack: `_research/industry_brief.md` (2026 adoption and trade context).

---

> **Next:** `01A_foundations_units_time_assumptions.md` — the quantitative anchor: units, calendars, lead-time definitions, forecast-error metrics, distributions, and the assumptions ledger that every later model depends on.
