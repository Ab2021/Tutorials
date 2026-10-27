# Three Flows, One Network

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: none
> Version 2026.5

---

## 1. Why this topic exists

A supply chain is not a diagram of boxes and arrows. It is a set of simultaneous flows that
must stay coherent with one another, and most supply chain failures are failures of coherence
rather than failures of any single step.

Three flows run at once:

- **Physical flow** — material moving from extraction to customer, and returns moving back.
- **Information flow** — orders, forecasts, inventory positions, shipment events, invoices.
- **Financial flow** — payment terms, credit, working capital, and the timing of cash.

Every operational decision is a bet on how these three will line up later. When a planner
commits to a production run, they are betting that physical capacity, information accuracy,
and cash availability will all still be true when the run completes. When a buyer signs a
twelve-month contract, they are betting that the price, the lead time, and the payment terms
remain affordable under conditions they cannot yet see.

The three flows move at different speeds, and that speed difference is the source of most
professional difficulty:

| Flow | Typical cycle time | Who owns it | What breaks when it is slow |
|---|---|---|---|
| Physical | Days to months | Operations, logistics | Service level, inventory, expiry |
| Information | Minutes to days | Planning, IT | Forecast error, bullwhip, stock-outs |
| Financial | Weeks to quarters | Treasury, finance | Liquidity, supplier viability, cost |

A shipment takes six weeks by ocean. The information about that shipment can take six minutes.
The cash tied up in it takes ninety days to return. A planner holding all three in their head
at once is doing the job. A planner who thinks only in units is doing half of it.

**Why this matters to a data scientist specifically.** The models you will build in Modules 06
to 20 sit almost entirely inside the information flow. They predict, classify, rank, allocate,
and optimise. If you do not understand what the information flow is *about* — physical material
and financial obligation — you will produce models that are technically sound and operationally
ignored. This is the single most common failure mode of analytics in this field, and it is not
a modelling failure. It is a vocabulary failure.

Three recurring examples show the shape of the problem:

1. **A forecast that improves accuracy but worsens service.** A statistical model reduces mean
   absolute percentage error (MAPE) by two points while systematically under-forecasting the
   high-volume tail. Inventory policy is set on the mean, so the improvement in average error
   produces more stock-outs on the items that matter. The model did what it was asked. The
   objective was wrong, because nobody connected the information flow to the physical consequence.
2. **A lead-time model that ignores the financial flow.** An analyst builds a supplier
   delay-prediction model using shipment data only. It performs well. It is then used to
   reallocate orders away from a supplier who is slow but who is also the only source of
   working-capital financing the company has through reverse factoring. The reallocation
   destroys the company's cash position while improving on-time delivery.
3. **An inventory optimisation that ignores information latency.** A safety-stock model assumes
   daily inventory visibility. In three of the company's regions, stock positions are reconciled
   weekly. The model computes a correct number against data that is up to seven days stale,
   producing recommended orders that are consistently late by exactly the amount the model
   was trying to protect against.

Each of these is a coherence failure. The fix is not a better algorithm. It is a correct mental
model of the network and the flows within it.

This topic is first in the module because every later topic is a zoom into one part of the
picture: the decision stack (Topic 02), the entities (Topic 03), the planning processes that
move the information (Topic 04), inventory and service (Topic 05), cost and cash (Topic 06),
manufacturing (Topic 07), transport (Topic 08), warehousing and sourcing (Topic 09), and the
archetypes that explain why two chains facing the same demand should be designed differently
(Topic 10).

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** a three-flow map of a named supply chain from memory, labelling at least eight nodes
   and identifying where each flow changes ownership.
2. **Trace** a single customer order backwards through information, physical, and financial
   flows, naming the decision and the decision-maker at each hop.
3. **Compute** cash-to-cash cycle time from days inventory outstanding, days sales outstanding,
   and days payable outstanding, and **explain** what each component measures operationally.
4. **Diagnose** a described supply chain failure as a physical, informational, or financial
   coherence failure, and **justify** the classification with evidence from the description.
5. **Explain** the bullwhip effect in terms of the three flows, and **identify** which
   information-flow design choices amplify or dampen it.
6. **Distinguish** push from pull, and **locate** the push–pull boundary in a given network
   with a stated justification.
7. **Critique** a proposed analytics use case for whether it addresses a physical, informational,
   or financial decision, and **state** what would have to be true for it to change behaviour.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture in your head: a loop of material, with a parallel loop of information running
in the opposite direction, and a financial loop lagging both.

```text
                        INFORMATION FLOW  (orders, forecasts, events)
                 <----------------------------------------------------
                                                                      |
   +----------+    +----------+    +----------+    +----------+    +----------+
   | SUPPLIER |--->|   PLANT  |--->|    DC    |--->|  STORE / |--->| CUSTOMER |
   |  tier n  |    |  / EMS   |    | / 3PL    |    |  LAST    |    |  / USER  |
   +----------+    +----------+    +----------+    +----------+    +----------+
        ^                                                                |
        |                                                                |
        +----------------------- RETURNS / RECOVERY ---------------------+
                                  (reverse physical flow)

   FINANCIAL FLOW  (payables out to the left, receivables in from the right)

        pay supplier <---- [working capital held as inventory] ----> collect from customer
        DPO days                                                    DSO days
                        DIO days  (cash is trapped here)
```

The picture has four properties worth memorising, because they explain most of what follows.

**Property 1 — Material moves forward, information moves backward, and they are not
synchronised.** A customer order travels upstream in seconds. The goods travel downstream in
days or weeks. That gap is where inventory lives. Every inventory position in a supply chain is
a physical record of an information delay that has not yet been resolved.

**Property 2 — Each node in the picture is a decision point, not a storage point.** A
distribution centre (DC) is not where boxes sit. It is where a replenishment decision is made,
using local information, local incentives, and local constraints. When you model a DC as a
buffer, you model the symptom and miss the cause.

**Property 3 — The financial flow is a delayed, aggregated readout of the physical flow.**
Cash arrives after the goods leave, minus a credit period, plus or minus a payment term. If
the physical flow is 40 days long and the payment terms are net 60, the company finances the
whole journey. If the terms are net 30, it finances part of it and its supplier finances the
rest. The same physical chain with different terms is a different business.

**Property 4 — Control is distributed, but the consequences are not.** No single actor sets
inventory, price, and lead time together. The supplier sets lead time, the buyer sets order
quantity, finance sets payment terms, sales sets promotion dates, and the customer sets demand.
Each optimises locally. The network-level result is emergent, and often worse than any
participant intended.

### 3.1 The push–pull boundary

Two coordination logics coexist in every chain:

- **Push** — production is scheduled against a forecast, before demand is known. Efficient,
  because it batches and levels load. Fragile, because it commits to a guess.
- **Pull** — production or replenishment is triggered by actual consumption. Responsive,
  because it follows demand. Expensive, because it cannot batch or level.

Real chains are hybrids, and the **push–pull boundary** is the position where the chain stops
guessing and starts responding. Classic placements:

| Business | Push–pull boundary | Why there |
|---|---|---|
| Make-to-stock consumer goods | Finished goods at regional DC | Long production runs make postponement uneconomic at item level |
| Configure-to-order electronics | Sub-assembly, before final configuration | Configurations proliferate; components are common |
| Apparel with basic and fashion lines | Greige fabric, before dyeing | Colour and print are the volatile attributes |
| Automotive | Dealer stock, with build-to-order exception | Physical distribution is expensive; customers accept short waits |
| Quick commerce | Dark store, replenished from a central DC | Delivery promise is minutes; the only pull that works is at the edge |
| Process chemicals | Bulk tank at the plant gate | Product is continuous; there is no unit to postpone |

**The boundary is a business decision, not a modelling output.** Moving it right increases
responsiveness and inventory. Moving it left increases efficiency and forecast dependence. An
analytics team can quantify the trade-off. It cannot make the choice, because the choice
encodes the company's promise to its customer.

### 3.2 The bullwhip, stated in flow terms

Demand variability grows as you move upstream. The classical statement is that the variance of
orders placed by a firm exceeds the variance of sales it faces. In flow terms, the mechanism is
fourfold, and each mechanism is an information-flow design choice:

1. **Demand signal processing.** Each tier forecasts from its own orders, which are already a
   distorted echo of demand. Errors compound.
2. **Order batching.** Buyers order in economic batches or on fixed cycle days, converting
   smooth demand into lumpy orders.
3. **Price fluctuation.** Forward buying during promotions creates demand that is pulled
   forward, not created, and reverses afterwards.
4. **Shortage gaming.** When supply is rationed, buyers over-order to protect their allocation,
   then cancel. The rationing rule, not the shortage, produces the distortion.

Two design levers cut across all four: **information sharing** (each tier sees real demand, not
just orders) and **shorter, more frequent ordering cycles** (less time for distortion to
accumulate). Both are organisational commitments before they are technical ones.

```text
   variance
      ^                                        .  tier-2 supplier
      |                                   .
      |                              .
      |                        .          tier-1 supplier
      |                  .
      |            .                       manufacturer
      |      .
      |  .                                 retailer
      +--------------------------------------------> position in chain
         customer        retail      distribution     production
```

A useful diagnostic: if a company's forecast error measured against *its own orders* looks
small while its forecast error against *end-customer demand* looks large, the company is
forecasting a distorted signal and calling it accuracy.

### 3.3 Where analytics enters

| Analytics task | Flow it acts on | Decision it changes | Failure if the flow model is wrong |
|---|---|---|---|
| Demand forecasting | Information | How much to buy or build | Forecasts orders, not demand |
| Inventory optimisation | Physical + financial | How much to hold | Ignores cash cost or shelf life |
| Network design | Physical | Where to locate capacity | Ignores service promise and tax |
| Lead-time prediction | Information + physical | Which supplier to use | Ignores contractual commitments |
| Cost-to-serve | Financial | Which customers to prioritise | Ignores working capital |
| Control tower alerting | Information | What to intervene on | Alerts on noise; trust collapses |
| Dynamic pricing | Information + financial | Price and promotion timing | Pulls demand forward and books it as growth |

Read the last row twice. It is the most common analytical self-deception in commercial supply
chain work: a promotion that shifts demand forward by three weeks looks like a lift in every
report that measures against plan, and looks like nothing at all in a report that measures
against a baseline corrected for the shift.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Supply chain | The network of organisations, people, processes, information, and resources that moves a product or service from supplier to customer | Used to mean "logistics" | None |
| Value chain | The sequence of activities that progressively add value, including activities outside the chain's ownership | Used interchangeably with supply chain | None |
| Node | A location or organisation where material stops, is transformed, or changes custody | Treated as a warehouse only | Location identifier |
| Lane | A directional connection between two nodes with a mode and a lead time | Treated as a route on a map | Origin–destination pair plus mode |
| Echelon | A level of the chain (supplier, plant, DC, store) | Confused with a tier | Ordinal position |
| Tier | Depth of supplier relationship (tier 1 direct, tier 2, tier n) | Equated with echelon | Tier number |
| Upstream | Toward raw material | Used loosely as "earlier in time" | Direction |
| Downstream | Toward the end customer | Used loosely as "later in time" | Direction |
| Lead time | Elapsed time between an order and availability | Used to mean transit time only | Days |
| Cycle time | Time to complete one pass of a process | Confused with lead time | Hours or days |
| Push | Execution against a forecast | Treated as inherently bad | None |
| Pull | Execution against actual consumption signal | Treated as inherently good | None |
| Push–pull boundary | The point where forecast-driven flow gives way to demand-driven flow | Treated as a fixed technical attribute | Node position |
| Postponement | Delaying product differentiation until demand is closer | Confused with delaying shipment | Stage number |
| Bullwhip effect | Amplification of demand variability moving upstream | Used to describe any variability | Variance ratio |
| Demand signal | The observable proxy for true customer demand | Assumed to equal demand | Units per period |
| Order | A commercial commitment to buy | Treated as demand | Units, with a date |
| Shipment | Physical movement of goods under one document | Treated as an order | Units, weight, volume |
| Consignment | Goods held at a customer site but owned by the supplier until use | Confused with a shipment | Units |
| Inventory position | On-hand plus on-order minus backorders | Confused with on-hand | Units |
| Safety stock | Inventory held to cover variability | Treated as a policy constant | Units |
| Cycle stock | Inventory that exists because of batching | Treated as waste | Units |
| In-transit inventory | Goods owned but not yet at a node | Omitted from position calculations | Units |
| Pipeline inventory | Total inventory in motion and in buffers across a lane | Confused with in-transit | Units |
| Obsolete stock | Inventory with no reasonable prospect of sale or use | Discovered late | Units and value |
| Fill rate | Fraction of demand met from stock | Confused with cycle service level | Proportion |
| Cycle service level (CSL) | Probability of not stocking out in a replenishment cycle | Reported as fill rate | Proportion |
| On-time in-full (OTIF) | Fraction of orders delivered complete and on time | Reported without the definition being stated | Proportion |
| Days inventory outstanding (DIO) | Inventory value divided by daily cost of goods sold | Computed on revenue, which inflates it | Days |
| Days sales outstanding (DSO) | Receivables divided by daily revenue | Computed on cost, which deflates it | Days |
| Days payable outstanding (DPO) | Payables divided by daily cost of goods sold | Compared across firms with different terms | Days |
| Cash conversion cycle (CCC) | DIO + DSO − DPO | Treated as a pure finance metric | Days |
| Working capital | Current assets minus current liabilities | Confused with cash | Currency |
| Landed cost | Total cost of a good delivered to a location, including freight, duty, and fees | Reported excluding duty | Currency per unit |
| Incoterms | ICC rules defining buyer and seller responsibilities for delivery | Treated as payment terms | Three-letter code plus place |
| Total cost of ownership (TCO) | Cost over the whole life, including operation and disposal | Used as a synonym for price | Currency |
| Service level agreement (SLA) | Contractual commitment on a measurable service outcome | Confused with a target | Contract |
| Cost-to-serve | Cost attributable to serving a specific customer, channel, or order | Averaged into a single rate | Currency per order |
| Forecast | A probabilistic statement about the future | Treated as a commitment | Units with a distribution |
| Plan | A decision about what the company will do | Treated as a forecast | Units with a schedule |
| Target | An aspiration set by management | Treated as a plan | Units |
| Commitment | A promise made to a customer or supplier | Treated as a forecast | Contract |

Four of these deserve expansion, because they cause the most damage when confused.

**Forecast, plan, target, commitment.** These four words are used interchangeably in most
meetings and they are four different objects. A forecast is what you believe will happen. A
plan is what you intend to do. A target is what you would like to achieve. A commitment is what
you have promised. Only the plan is under your control. Only the commitment carries a penalty.
A supply chain organisation that cannot separate these four will systematically blame the
forecast for a planning decision. Every topic in this program will ask you to state which of
the four you mean.

**Cycle service level versus fill rate.** CSL is the probability that a replenishment cycle
ends without a stock-out. Fill rate is the fraction of demand units served from stock. A
company can have 95% CSL and 80% fill rate: it rarely runs out, but when it does, it runs out
of its highest-volume item. Reporting one while managing to the other is a classic and
expensive error, explored in Topic 05.

**DIO, DSO, DPO.** The three working-capital days are computed on different denominators, and
mixing them is the most common error in supply chain finance conversations. DIO typically uses
cost of goods sold (COGS) because inventory is carried at cost. DSO uses revenue because
receivables are recorded at selling price. DPO uses COGS because payables are at purchase cost.
Computing DIO on revenue understates the days by the gross margin.

**Landed cost versus price.** Price is what the invoice says. Landed cost is what the unit costs
once it is in your warehouse. For an imported item, freight, insurance, duty, brokerage,
demurrage, and inventory carrying cost during transit can add 20–40% to the invoice price, and
the mix varies by lane and by Incoterm. Any model that optimises on price optimises the wrong
number.

## 5. The mechanics, step by step

This section traces a single customer order through the network, one hop at a time, naming what
happens physically, informationally, and financially at each step. The purpose is to build the
reflex of asking "which flow?" for any described event.

### Phase A — Demand becomes information

**Step 1. The customer expresses a need.**

- Physical: nothing happens yet.
- Informational: a signal exists in the world — a shopper picks an item off a shelf, a buyer
  sends a purchase order (PO), a retailer's point-of-sale (POS) system records a scan.
- Financial: nothing yet.
- Analytic artifact: the demand observation. Whether you can see it depends entirely on
  whether the customer's system is connected to yours.
- Key branch: **is the signal visible, or only its consequence?** A consumer goods manufacturer
  usually sees shipments to the retailer, not the shelf scan. Those are different series with
  different variances, and forecasting one while managing the other is the root of a great deal
  of error.

**Step 2. The signal is aggregated and interpreted.**

- Informational: daily scans become weekly orders; orders become a forecast; the forecast
  becomes a plan.
- Physical: none.
- Financial: none.
- Analytic artifact: a demand history table keyed by item, location, and time bucket.
- Key branch: **what is the bucketing rule?** Daily data aggregated to weekly smooths genuine
  intra-week patterns and creates artificial ones. A week that contains a public holiday looks
  like a demand drop. Every downstream model inherits the artifact.

**Step 3. The order is placed.**

- Informational: a purchase order or replenishment order is transmitted, typically by electronic
  data interchange (EDI) or an application programming interface (API).
- Physical: none.
- Financial: a commitment is created. Under the contract, the buyer now owes money contingent
  on delivery.
- Analytic artifact: an order record with a requested date, a quantity, and a price.
- Key branch: **is the order quantity the real need?** Often not. It is the need rounded to a
  pallet, a truckload, a minimum order quantity, or a promotional buy. Treating it as demand
  imports the batching distortion.

### Phase B — Information becomes material

**Step 4. Supply is allocated.**

- Informational: supply is matched to demand across competing orders, often by rule (priority,
  margin, contract, fairness).
- Physical: none yet.
- Financial: none yet, but the allocation rule determines which customers' cash arrives and
  which are deferred.
- Analytic artifact: an allocation table with a rule and its justification.
- Key branch: **is allocation proportional or priority-based?** Proportional allocation
  encourages shortage gaming. Priority allocation discourages it but creates visible winners
  and losers, which is a commercial decision.

**Step 5. Material is released and produced or picked.**

- Physical: raw material is consumed; work in process (WIP) appears; finished goods are made.
  Or, in a DC, units are picked and packed.
- Informational: production confirmation or pick confirmation events are recorded.
- Financial: cost is incurred — material, labour, overhead — and inventory value increases.
- Analytic artifact: an event stream with timestamps at operation level.
- Key branch: **is the confirmation real-time or end-of-shift?** A manufacturing execution
  system (MES) that confirms at shift end destroys the ability to see WIP position during the
  shift, which is exactly when a planner needs it.

**Step 6. Goods move.**

- Physical: the shipment travels by mode, through nodes, across borders.
- Informational: shipment events — departure, arrival, customs clearance, delivery — are
  recorded, often by multiple parties with different systems.
- Financial: title and risk transfer at the point defined by the Incoterm, which determines who
  insures, who pays freight, and when revenue or inventory is recognised.
- Analytic artifact: a shipment record plus an event log.
- Key branch: **who owns the goods in transit?** Under free carrier (FCA) the buyer owns them
  early. Under delivered duty paid (DDP) the seller owns them late. Get this wrong and your
  inventory position is wrong by the entire transit time.

**Step 7. Goods are received and put away.**

- Physical: unloaded, inspected, stored at a location.
- Informational: receipt confirmation; inventory position updated; discrepancies recorded.
- Financial: three-way match — PO, receipt, invoice — is performed before payment is released.
- Analytic artifact: a receipt record with quantity, condition, and timestamp.
- Key branch: **is receipt confirmation automatic or manual?** Manual receipt lag is one of the
  most common causes of phantom stock-outs: the goods are physically on the shelf but invisible
  to the system for a day.

**Step 8. The order is fulfilled.**

- Physical: picked, packed, labelled, loaded, delivered.
- Informational: delivery confirmation, proof of delivery, and often a serialised event under
  a traceability standard.
- Financial: invoice issued; revenue recognised under the applicable accounting standard;
  receivable created.
- Analytic artifact: a delivery event plus an invoice.
- Key branch: **when is revenue recognised?** Under IFRS 15 / ASC 606, control transfer governs
  recognition. For a shipment FOB destination, control transfers later than for FOB origin.
  A revenue metric without this detail is not comparable across lanes.

### Phase C — Material becomes cash

**Step 9. The invoice is collected.**

- Financial: the buyer pays after the agreed credit period. DSO accrues from invoice date.
- Informational: remittance advice and payment status are matched to the invoice.
- Physical: none.
- Analytic artifact: an aged receivables ledger.
- Key branch: **is the receivable disputed?** Disputes are usually caused by a physical or
  informational discrepancy — a short shipment, a price mismatch, a missing proof of delivery.
  The financial symptom is a supply chain defect.

**Step 10. The supplier is paid.**

- Financial: DPO accrues from invoice receipt to payment.
- Informational: payment run, discount capture, and remittance.
- Physical: none.
- Analytic artifact: an aged payables ledger.
- Key branch: **is the payment term being stretched?** A company that extends DPO from 45 to 75
  days improves its own CCC and transfers stress to its suppliers. The metric looks better. The
  network looks worse. This is the central tension of Topic 06 and of Module 42.

### Phase D — The loop closes

**Step 11. Returns and recovery.**

- Physical: goods come back — warranty, recall, unsold, damaged, or end-of-life.
- Informational: a return authorisation and a disposition decision.
- Financial: credit note, refund, or restocking fee; the inventory value may be restored at a
  lower carrying value.
- Analytic artifact: a returns record with reason code and disposition.
- Key branch: **is the return reason code trustworthy?** Free-text or default-coded reasons
  ("customer preference") hide the real causes, which are often fit, damage, or a channel
  promise that could not be met.

**Step 12. Learning.**

- Informational: the deviation between plan and actual becomes input to the next cycle.
- Physical: none.
- Financial: none.
- Analytic artifact: forecast error decomposition, root-cause tags, and a corrective action.
- Key branch: **is the error attributed to a cause or to a person?** An organisation that
  attributes forecast error to the forecaster stops learning. One that attributes it to a
  mechanism — a promotion not communicated, a lead time shift — learns.

### 5.1 Reading a supply chain diagram correctly

Given any diagram with boxes and arrows, ask six questions in order. The answers tell you more
than the diagram does.

1. **Where does the material stop, and why?** Every stop is either a transformation or a decoupling
   point. Decoupling points are where inventory exists to break a dependency.
2. **Where does ownership change?** Ownership changes are where risk, insurance, and revenue
   recognition change. They are often unmarked on diagrams and always material.
3. **Where does the information become an order?** This is the push–pull boundary, and it is the
   most consequential line on the page.
4. **Where is information delayed?** Any point where data arrives after the decision it should
   inform is a structural defect.
5. **Where does cash leave and return?** Draw the financial loop separately. It rarely matches
   the physical loop.
6. **Which node has discretion and which is a pass-through?** Discretionary nodes create
   amplification. Pass-through nodes transmit it.

### 5.2 Why the flow view changes how you read an error metric

Consider a company reporting 92% OTIF and 88% forecast accuracy. Both numbers look acceptable.
Now add the flow detail:

- OTIF is measured against internal promise dates, not customer-requested dates.
- Forecast accuracy is measured at the weekly total across all SKUs, not at item–location level.
- The OTIF measure excludes shipments that were cancelled and re-booked.

Under the flow view, the two numbers are not describing the same network. One describes an
aggregate information claim. The other describes an execution claim with an exclusion. The
network performance could be materially worse than either number, and the two numbers could
both be computed correctly. This is not fraud. It is what happens when metrics are designed by
different functions for different purposes, which is the default state of every large company.

The analytic skill is not to accuse. It is to reconstruct, from the definitions, what the metric
actually measures, and then to say what it fails to measure.

### 5.3 The same three flows in a service setting

Most of this program concerns goods, but the flow model survives when the product is a service,
and it is worth seeing how. Take a hospital's sterile-processing department, an airline's
maintenance operation, or a field-service network.

- **Physical flow** becomes the movement of the *asset* or the *consumable* into position: an
  instrument tray through sterilisation, an aircraft through a maintenance slot, a technician
  to a site. The unit of flow is not a widget but a slot of capacity.
- **Information flow** becomes the booking, the diagnosis, the work order, and the completion
  record. It still travels backwards against the physical flow.
- **Financial flow** becomes the claim, the reimbursement, the invoice, or the contract payment.
  It still lags both.

The structural lesson transfers exactly: **capacity is the inventory of a service chain**.
Where a goods chain stores units to decouple stages, a service chain stores *time* — appointment
slots, bed-days, technician-hours — and it decays if unused. A hospital bed empty at 02:00 is an
expired unit of inventory. An aircraft on the ground is inventory with a maintenance clock
running.

Two consequences follow for analytics. First, service chains cannot buffer with stock, so they
buffer with demand shaping (yield management) or with capacity flexibility (agency staff,
overtime, cross-training). Second, service demand variability is often larger than goods demand
because it is co-produced with the customer — a restaurant's peak is defined by when people
decide to eat, not by when food arrives. Bullwhip appears in service as booking behaviour:
customers over-book to protect against not getting a slot, then no-show, which produces exactly
the same rationing-and-gaming loop described for goods.

## 6. Worked example

A mid-sized consumer goods company — call it Northwind Foods — sells a single high-volume
product through three channels. We follow one month and one product to see how the three flows
line up. All figures are illustrative but internally consistent.

### 6.1 The network

```text
   [ Supplier A ]  -- 21 d ocean -->  [ Plant (Ohio) ]  -- 2 d truck -->  [ DC East ]
        parts            FCA                build              FTL              |
                                                                               | 3 d truck
                                                                               v
                                                                    [ Retail DC ]  -- 1 d --> [ Stores ]
                                                                               |
                                                                               +--2 d parcel--> [ E-commerce ]

   [ Supplier B ]  -- 35 d ocean --+  (second source for two components)
```

### 6.2 One month of flow

**Physical.** Northwind ships 120,000 units per month to retail and 30,000 units to e-commerce.
The plant builds 150,000 units in the month. Two ocean containers of components arrive per week.

**Informational.** Retail sends weekly EDI purchase orders on Tuesdays for delivery two weeks
later. E-commerce demand is visible daily through the warehouse management system (WMS). The
plant plans on a four-week frozen horizon and a twelve-week planning horizon.

**Financial.** Supplier A is paid at 60 days from invoice. Retail customers pay at 45 days from
invoice. E-commerce marketplace payouts arrive at 14 days. Inventory is carried at 22% annual
holding cost, which includes capital, storage, insurance, and obsolescence.

### 6.3 Tracing one unit

A unit sold on day 1 of the month has a history:

| Event | Day | Flow | Cost or value accrued |
|---|---:|---|---|
| Component ordered from Supplier A | −84 | Information | Commitment created |
| Component shipped, FCA origin | −63 | Physical starts | Freight paid by Northwind from here |
| Component arrives at plant | −42 | Physical | Duty and freight paid; inventory value rises |
| Production run | −28 | Physical | Conversion cost added |
| Finished goods to DC East | −25 | Physical | Freight added |
| DC to Retail DC | −8 | Physical | Freight added |
| Retail DC to store | −2 | Physical | Freight added |
| Sold at store | 0 | Physical | Revenue recognisable at delivery |
| Invoice issued to retailer | 0 | Financial | Receivable created |
| Retailer pays | +45 | Financial | Cash received |

The cash is out at day −63 and back at day +45: **108 days** of financing for this lane, of
which 63 days are upstream (covered partly by Supplier A's 60-day terms, meaning the company
pays around the time the goods leave the supplier) and 45 days are downstream receivables.

### 6.4 Working-capital arithmetic

Take the company-level balance sheet for the month. Cost of goods sold is €9.0 million per
month (€300,000 per day at 30 days). Revenue is €12.0 million per month (€400,000 per day).
Inventory value is €4.5 million. Receivables are €6.0 million. Payables are €3.6 million.

```text
DIO = 4,500,000 / 300,000 = 15.0 days
DSO = 6,000,000 / 400,000 = 15.0 days
DPO = 3,600,000 / 300,000 = 12.0 days

CCC = DIO + DSO - DPO = 15.0 + 15.0 - 12.0 = 18.0 days
```

Eighteen days of cash conversion. At €300,000 of COGS per day, that is:

```text
working capital tied up = 18.0 x 300,000 = €5,400,000
```

Now change one parameter and watch the network respond. Suppose procurement negotiates 90-day
terms with Supplier A instead of 60.

```text
payables rise by 30 days of purchases.
Assume purchased material is 60% of COGS = 180,000 per day.
additional payables = 30 x 180,000 = €5,400,000
new DPO = (3,600,000 + 5,400,000) / 300,000 = 30.0 days
new CCC = 15.0 + 15.0 - 30.0 = 0.0 days
```

Cash conversion cycle falls to zero. The company now finances nothing. On paper this is a
triumph, and it is exactly the kind of result that appears in a procurement business case.

Read it through the three-flow lens and the picture changes. Supplier A is now financing 90 days
on a product whose own production cycle is 42 days and whose ocean transit is 21 days. Supplier
A's own cash conversion cycle must absorb 90 days of receivables. Either Supplier A has the
balance sheet to fund it, or Supplier A raises prices, or Supplier A borrows at a rate that
reflects its smaller size and passes the cost through. In practice, all three happen, in
different proportions, and the price increase typically appears 6–18 months later as a
"market-driven" cost increase that nobody connects to the terms change.

This is the single most important lesson of the worked example: **a working-capital metric
improves whenever you move the burden, and does not tell you whether you moved it somewhere
that can carry it.** Module 42 treats the credit and treasury machinery. Here the point is
only that the metric is a boundary object, not an outcome.

### 6.5 Where the bullwhip shows up in these numbers

Retail demand for the product has a coefficient of variation of 0.12 at store level, measured
weekly. The retailer's orders to Northwind have a coefficient of variation of 0.31. Northwind's
production plan has a coefficient of variation of 0.44.

Three amplifications, three causes:

- **Store to retailer order (0.12 → 0.31):** weekly ordering on a fixed day plus minimum order
  quantities. Batching alone roughly doubles observed variability.
- **Retailer order to production plan (0.31 → 0.44):** the plant plans on a four-week frozen
  horizon, so it must absorb week-to-week variation into a monthly commitment, then level
  production across the month.
- **Additional amplification from promotions:** one price promotion in the month pulled forward
  roughly 9% of the following month's volume. Measured against plan, it was a 14% lift. Measured
  against a baseline corrected for the pull-forward, it was a 5% lift. Both are correct
  computations of different questions.

### 6.6 What an analyst should have asked

Given this case, the questions that change decisions are:

1. Is the 15-day DIO computed on inventory owned by Northwind only, or does it include
   consignment stock at the retailer? Consignment would understate DIO and overstate nothing.
2. Is the 45-day DSO the contractual term or the observed collection period? A contractual 45
   with an observed 58 means 13 days of disputes and deductions.
3. Does the 22% holding cost include obsolescence? If it does, then holding inventory of a
   product with a 9-month shelf life is more expensive than the average, and the average rate
   misallocates.
4. Is the coefficient of variation computed on the same time bucket at all three levels? If
   store CV is weekly and production plan CV is monthly, the comparison is meaningless.

Each question is a coherence question. None requires a model.

## 7. Code, query, or calculation

This section deliberately uses a hand calculation and a small, readable query rather than a
software workflow. The point is the arithmetic, not the tooling.

### 7.1 Hand calculation template

Given: inventory value `I`, receivables `R`, payables `P`, annual revenue `Rev`, annual COGS `C`,
and 365 days per year.

```text
daily revenue        = Rev / 365
daily COGS           = C / 365
DIO                  = I / daily COGS
DSO                  = R / daily revenue
DPO                  = P / daily COGS
CCC                  = DIO + DSO - DPO
cash tied up         = CCC x daily COGS
```

Worked with the Northwind month scaled to a year (×12):

```text
Rev      = 144,000,000
C        = 108,000,000
I        =  54,000,000
R        =  72,000,000
P        =  43,200,000

daily revenue = 144,000,000 / 365 = 394,520.55
daily COGS    = 108,000,000 / 365 = 295,890.41

DIO = 54,000,000 / 295,890.41 = 182.5 ... (monthly figures; see note)
```

**Note:** scaling a single month by twelve assumes the month is representative. If the month is
a peak season, DIO will be overstated. Always state the period and check for seasonality before
annualising. Using the monthly figures directly, as in section 6.4, avoids this trap and is the
safer teaching form.

### 7.2 A minimal query

The following structured query language (SQL) fragment computes the three days for a monthly
snapshot. It assumes a conventional schema; adjust names to the warehouse.

```sql
-- Assumptions: inventory and receivables at period end; COGS and revenue for the period;
-- 30 days in the period. Replace with actual day counts.
with base as (
    select
        p.period_id,
        p.revenue,
        p.cogs,
        b.inventory_value,
        b.receivables,
        b.payables,
        30.0 as days_in_period
    from finance.pnl_monthly p
    join finance.balance_sheet_monthly b using (period_id)
)
select
    period_id,
    round(inventory_value / (cogs / days_in_period), 1)      as dio_days,
    round(receivables   / (revenue / days_in_period), 1)     as dso_days,
    round(payables      / (cogs / days_in_period), 1)        as dpo_days,
    round(inventory_value / (cogs / days_in_period)
        + receivables   / (revenue / days_in_period)
        - payables      / (cogs / days_in_period), 1)        as ccc_days
from base
order by period_id;
```

Expected output for the Northwind month: `15.0, 15.0, 12.0, 18.0`.

Two sanity checks before you trust any such query:

- Compute DIO on COGS, not revenue. If the result changes by more than 10% when you switch, you
  have found a denominator error somewhere upstream.
- Confirm that payables include only trade payables. Including accrued payroll or tax payables
  inflates DPO and flatters the cycle.

### 7.3 A variance decomposition you can do on paper

To locate the bullwhip, compute the coefficient of variation of the same demand construct at
each echelon, over the same bucket:

```text
CV = standard deviation / mean
amplification factor = CV(upstream) / CV(downstream)
```

Northwind: `0.31 / 0.12 = 2.58` at the retail-to-order step, then `0.44 / 0.31 = 1.42` at the
planning step. Cumulative amplification `0.44 / 0.12 = 3.67`.

If your bucket differs between levels, fix the bucket first. A CV comparison across buckets is
not a comparison.

## 8. Data, standards, and regulatory context

The flows are governed by standards at almost every handoff. Knowing which standard governs
which handoff tells you what the data will look like before you look at it.

### 8.1 Identification

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Product identification | The unit sold, ordered, and shipped has a globally unique trade item number | B |
| GS1 SSCC | Logistics unit identification | The pallet or case label identity | B |
| GS1 GLN | Location identification | Party and place identity across EDI and events | B |
| GS1 Digital Link | Product identification on packaging via QR | Consumer-facing identification and the route to product data | B |

GS1 standards are voluntary in most jurisdictions but are effectively mandatory in retail and
consumer goods because trading partners require them contractually. A data model that invents
its own product identifiers will not join to anything.

### 8.2 Information exchange

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| EDI (ANSI X12, EDIFACT) | Business document exchange | Order, invoice, advance ship notice (ASN), and inventory message formats | B |
| EPCIS 2.0 (GS1) | Event data | What happened to which object, where, when, and why | B |
| GS1 EDI / EANCOM | Retail messaging | Regional variants of the same documents | B |
| Peppol BIS / EN 16931 | European e-invoicing | The semantic model for electronic invoices | B |
| OpenAPI / REST | Modern partner integration | Application-level interfaces, not a data standard | C |

The important consequence for analytics: **order data and event data are different animals.**
An order says what someone intends. An event says what actually happened, with a timestamp and
a location. EPCIS is the standard that makes the second one shareable. If a company has no
event layer, it has orders and invoices, which are the two most distorted views of the physical
flow available.

### 8.3 Traceability and product regulation

| Requirement | Jurisdiction | Status as of 2026 | Grade |
|---|---|---|---|
| DSCSA serialisation and interoperable exchange | United States, pharmaceuticals | Enforcement phased; interoperable exchange requirement active | B |
| EU Falsified Medicines Directive | European Union, pharmaceuticals | In force; verification at dispensing | B |
| EU Ecodesign for Sustainable Products Regulation (ESPR) | European Union | Adopted 2024; Digital Product Passport requirements phasing in by product group | B |
| EU Deforestation Regulation (EUDR) | European Union | Phased application; geolocation due diligence for covered commodities | B |
| U.S. Food Traceability Rule (FSMA 204) | United States | Compliance dates extended; key data elements on the food traceability list | B |
| EU CBAM definitive regime | European Union | Begins 1 January 2026 for covered goods | B |

**What these constrain and what they do not.** They constrain *what data must exist and be
retained*, not how you model it. A traceability requirement tells you that an event record with
a location and a timestamp must exist for each transformation. It does not tell you how to
aggregate that record into a forecast feature. It does constrain the *quality floor*: a model
built on a dataset that cannot answer "where was this lot on which date" is a model built on
data that is structurally incomplete.

### 8.4 Financial and tax interfaces

| Requirement | Where | Effect on the flows | Grade |
|---|---|---|---|
| Incoterms 2020 | Global contract convention | Defines the point where cost and risk transfer | B |
| IFRS 15 / ASC 606 | Revenue recognition | Defines when control transfers, hence when revenue exists | B |
| VAT in the Digital Age (ViDA) | EU | Digital reporting and e-invoicing across the single market from the 2030s | B |
| Country e-invoicing mandates | Italy, Spain, France, others | Real-time or near-real-time invoice reporting to tax authorities | B |
| Transfer pricing rules | Multinational | Constrains how margin is allocated across entities | B |

The practical consequence: in several jurisdictions the invoice is now a near-real-time data
object reported to a tax authority. That changes what "the financial flow lags the physical
flow" means. The *cash* still lags. The *information* about the cash no longer does.

### 8.5 What is convention rather than requirement

Several things that feel like rules are only habits:

- The SCOR model's five processes (plan, source, make, deliver, return) are a reference
  framework, not a regulation. Useful for consistent language; not binding.
- "Safety stock of two weeks" is a policy, not a law. It appears in almost every company's
  documentation and is derived from almost nothing.
- ABC classification thresholds of 80/15/5 are a convention from a specific historical context.
  They are not optimal for any particular business.
- Weekly planning cycles are a habit inherited from a paper-based era. Some businesses genuinely
  need them; most do not need to be limited by them.

An analyst who can distinguish a requirement from a convention can challenge a convention. An
analyst who cannot will treat every inherited parameter as a physical constant.

## 9. Industry practice and case evidence

### 9.1 The bullwhip: documented observations

The bullwhip effect was named and formalised in Lee, Padmanabhan, and Whang (1997), "Information
Distortion in a Supply Chain: The Bullwhip Effect," *Management Science*. The paper reports the
Procter & Gamble observation on Pampers: retail sales were remarkably stable, while orders to
P&G from distributors showed large swings. This is a documented case study widely cited in the
operations literature (Grade A for the phenomenon; the specific figures vary by retelling and
should not be quoted precisely).

The Barilla case (Hammond, Harvard Business School case 9-694-046, 1994) documents a related
pattern in pasta: demand stability at the consumer level, extreme variability in distributor
orders driven by sales promotions, transportation discounts, and order batching. Barilla's
response — the just-in-time distribution (JITD) programme — was a data-sharing intervention
that met substantial organisational resistance from distributors who feared losing discretion.
Grade A for the case documentation; C for the outcome, since the full rollout is not cleanly
documented in the public record.

The enduring lesson from both is not "share data." It is that **the distortion is produced by
local incentive structures**, and any data-sharing programme that does not address the
incentives will fail. Barilla's distributors resisted because the programme asked them to give
up control over their own ordering, which was a commercial asset.

### 9.2 Physical flow disruptions and their information signatures

| Event | Physical mechanism | Information signature | Financial signature | Grade |
|---|---|---|---|---|
| Ever Given, Suez, March 2021 | Six-day blockage; ~12% of global trade transits | Vessel AIS gaps, container dwell, booking cancellations | Freight rates spiked; demurrage and detention accrued | B |
| Red Sea diversions from late 2023 | Routing around Cape of Good Hope; 10–14 additional days Asia–Europe | Extended transit times, schedule unreliability | Higher freight, longer lead times, more working capital in transit | B |
| COVID-19 PPE shortage, 2020 | Demand spike plus supply concentration | Orders multiplied; allocation rules created gaming | Spot prices rose several-fold depending on item | B |
| Semiconductor shortage, 2021–2023 | Capacity constraint in mature nodes | Allocation, extended lead times, order inflation | Automotive production cuts; revenue deferral | B |
| Cisco 2001 inventory write-down | Demand collapse after forecast-driven build | Orders cancelled; channel inventory invisible | Write-down of approximately $2.2bn | C |

The Cisco case is instructive because the failure was informational as much as physical. Cisco
had limited visibility into inventory held by contract manufacturers and channel partners, and
the company's own forecast-driven commitments did not unwind quickly when demand fell. The
write-down is a matter of public record; the causal attribution is a widely accepted
interpretation rather than an audited finding. Label the first C, the second an inference.

### 9.3 What companies actually do, versus what the literature recommends

| Recommendation | Observed practice | Gap and why it persists |
|---|---|---|
| Share POS data upstream | Partial; often aggregated and delayed | Retailers treat data as leverage |
| Shorten planning cycles | Weekly or monthly remains standard | Meeting rituals and system batch jobs |
| Use probabilistic forecasts | Point forecasts dominate operational use | Planning systems and incentives are point-based |
| Postpone differentiation | Applied where product architecture allows | Requires design change, not process change |
| Measure end-to-end cost-to-serve | Rare; cost centres dominate | Organisational boundaries, not technical limits |
| Integrate financial and operational planning | 8% report fully integrated end-to-end planning (BearingPoint 2026, European sample) | Different cadences, different owners, different systems |

The BearingPoint 2026 survey figure is a useful calibration point: the fully integrated
end-to-end planning state that most textbooks describe as normal is, in a 193-respondent
European sample, the exception. Any analytics design that assumes integration exists will
under-deliver in most organisations.

### 9.4 The technology context in 2026

Gartner's 2026 supply chain technology framing groups trends into autonomy and agency
(polyfunctional robots, physical AI, agentic AI, multi-agent systems), specialisation and
intelligence (intelligent simulation, domain-specific language models), and trust and governance
(product provenance, decision governance). Gartner's 2026 CSCO outlook is more cautious than the
technology framing: it characterises 2026 AI adoption as incremental rather than breakthrough,
with low trust in autonomous decision-making across large portions of cost of goods sold, and
advises applying the right level of technology to well-defined problems rather than hunting for
use cases. (Grade D for Gartner framing — vendor-analyst marketing; the 81%-in-spreadsheets and
8%-integrated figures from independent surveys are Grade C.)

Both readings can be true at once, and the tension is the practical context for this course:
the tooling is advancing faster than the process integration that would make it valuable.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating orders as demand | Forecast accuracy looks good; service does not improve | Order data contains batching, gaming, and promotion pull-forward | Systematic bias upstream | Reconstruct demand from POS or consumption where available; state the proxy |
| Metric definitions unstated | Two reports give different OTIF for the same week | Internal promise date versus customer request date | Decisions made on incomparable numbers | Publish a metric dictionary with the exact numerator and denominator |
| Working capital improved by burden transfer | CCC falls while supplier payment terms lengthen sharply | Payables stretched to fund receivables | Supplier distress, price increases, supply risk | Track supplier DSO alongside own CCC |
| Inventory position missing in-transit goods | Reconciliations show persistent gaps | Ownership during transit not modelled | Phantom shortages, duplicate orders | Model ownership by Incoterm per lane |
| Push–pull boundary undocumented | Nobody can say where forecast-driven flow ends | Boundary set by history, not design | Wrong inventory at wrong node | Draw and publish the boundary; test moving it |
| Bullwhip diagnosed as forecast error | Improving forecast does not reduce variability | Amplification is caused by batching and rationing | Effort spent on the wrong lever | Decompose CV by echelon and by cause |
| Promotion lift measured against plan | Every promotion is a success | Plan already includes the expected lift, or not | Demand pulled forward booked as growth | Baseline-corrected lift measurement |
| Lead time treated as a constant | Service degrades in peak season | Lead times are distributions, not numbers | Safety stock systematically short | Model lead time variance separately |
| Financial flow ignored in network design | Optimised network has worse cash position | Design objective excludes working capital | Liquidity strain | Include CCC in the network objective |
| Data latency not modelled | Recommendations consistently late | Analytics assumes fresher data than exists | Interventions arrive after the decision | Measure and publish data latency per feed |
| Aggregation hides item-level failure | Aggregate service at 97%, key items at 70% | Averaging across a skewed mix | Commercial damage | Report service at the decision level |
| Single-source assumption untested | No documented alternate for a critical input | Cost optimisation removed redundancy | Total stoppage on one failure | Map single points of failure; price the risk |

Two of these deserve a note.

**Metric definitions unstated** is the most common and least discussed. It is not a data
problem. Two competent analysts computing OTIF from the same warehouse will get different
answers if one uses the customer's requested date and the other uses the order confirmation
date. Neither is wrong. The organisation needs one definition and a note about what the other
would have said.

**Promotion lift measured against plan** is the most commercially consequential. Almost every
consumer goods organisation measures promotional performance against a baseline that is itself
a forecast, and the forecast already contains the expected promotional effect. The result is
that the baseline is contaminated and the lift is a residual, not an effect. Topic 09 of
Module 16 treats the correct measurement; here it matters only as an example of the information
flow misrepresenting the physical flow.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Demand is observable at some echelon | Every method starts from a demand series | No POS or consumption data; only shipments | Ask what the shipment series would show if demand were flat |
| The network boundaries are stable over the horizon | Models assume a fixed node and lane set | Network redesign, new 3PL, plant closure | Check for structural breaks in the data |
| Financial periods and operational periods align | Joining COGS to inventory requires matching periods | 4-4-5 calendars, fiscal years, month-end cutoffs | Compare period definitions across systems |
| Ownership transfers at the documented Incoterm | Inventory valuation depends on it | Side agreements, consignment, bill-and-hold | Confirm with finance, not with the contract alone |
| Cost of capital is a single rate | Simplifies holding cost | Different businesses, different risk | Segment the rate where material |
| Lead times are stationary | Safety stock depends on it | Capacity changes, route changes, seasonality | Plot lead time over time by lane |
| Product mix within a category is substitutable | Aggregation assumes it | Customers reject substitutes | Measure substitution in the data |
| Demand history is long enough to estimate variability | Statistical methods need it | New products, discontinued items | Check observation counts, not calendar length |
| All nodes use the same time zone convention | Event sequencing requires it | Cross-border operations | Audit timestamp offsets |
| Returns are a small fraction of flow | Simplifies net flow | High-return channels (fashion, electronics) | Compute return rate by channel |

Two limits of the three-flow framing are worth stating plainly.

**The framing is descriptive, not predictive.** It tells you where to look. It does not tell you
what will happen. A three-flow map of a chain does not forecast anything, and treating it as a
model is a category error.

**The framing assumes the firm is the unit of analysis.** For a marketplace, a 4PL, or a
platform, the "chain" is a set of relationships the firm coordinates but does not own. The three
flows still exist, but the boundary questions (who owns the goods, who bears the risk, who holds
the cash) become much harder and much more contractual. Module 21 treats marketplace structures
specifically.

## 12. Exercises

### Exercise 1 — Draw the map (recall and application, 45 minutes)

Choose a product you bought in the last month. Draw its three-flow map from memory, with at
least six nodes, marking ownership changes and the push–pull boundary.

**Deliverable:** one page, hand-drawn or text, with the three flows distinguished.
**Acceptance criteria:** at least six nodes; at least two ownership changes marked with the
Incoterm or contract type; the push–pull boundary identified with one sentence of justification;
one lane identified where information is likely delayed.

### Exercise 2 — Compute the cycle (calculation, 30 minutes)

A distributor reports: annual revenue €240m, annual COGS €192m, inventory €28m, receivables
€40m, payables €22m.

**Deliverable:** DIO, DSO, DPO, CCC, and cash tied up, each with units and the denominator used.
**Acceptance criteria:** correct arithmetic; correct denominators; a one-sentence statement of
what the CCC means operationally for this business.

### Exercise 3 — Diagnose the failure (critique, 40 minutes)

A company reports: forecast accuracy improved from 82% to 88%; OTIF fell from 94% to 89%;
inventory rose 12%; CCC rose 6 days. Write a one-page diagnosis.

**Deliverable:** a causal hypothesis with evidence, a list of three alternative explanations, and
the measurement you would take next.
**Acceptance criteria:** each hypothesis names which flow it implicates; at least one alternative
is a measurement artifact rather than a real change; the next measurement is specific.

### Exercise 4 — Bullwhip decomposition (application, 60 minutes)

Using any public demand dataset with at least two echelons (or a constructed one), compute the
coefficient of variation at each level over the same bucket and the amplification factors.

**Deliverable:** a table of CV by echelon, the amplification factors, and a paragraph attributing
the amplification to batching, pricing, rationing, or signal processing.
**Acceptance criteria:** identical buckets across levels; attribution supported by evidence in the
data, not by assertion.

### Exercise 5 — Push–pull redesign (design, 60 minutes)

For a product with a 6-week ocean lead time, a 12-week selling season, and high style risk,
propose a push–pull boundary and defend it.

**Deliverable:** a boundary recommendation, the inventory consequence, the cost consequence, and
the condition under which you would move it.
**Acceptance criteria:** the recommendation names the postponement point explicitly; the
consequences are quantified in units and currency; the reversal condition is specific.

### Exercise 6 — Metric audit (critique, 45 minutes)

Obtain two internal reports (or construct two from public filings) that both claim to measure
service level. Compare their definitions.

**Deliverable:** a metric dictionary entry for each, and a statement of how far apart they could
be on the same underlying data.
**Acceptance criteria:** numerator, denominator, and time window stated for each; at least one
plausible exclusion identified; a defensible estimate of the gap.

### Exercise 7 — Finance–operations join (design, 45 minutes)

Design the join between the operational plan and the financial forecast for a monthly cycle.

**Deliverable:** a mapping table from operational quantities to financial line items, with timing
assumptions.
**Acceptance criteria:** every mapping states its lag; at least three mappings are identified
where the lag differs across lanes or products; the mapping is testable against a closed month.

### Exercise 8 — Case critique (critique, 60 minutes)

Read the public record on one disruption (Red Sea diversions, Suez 2021, or the 2021–2023
semiconductor shortage). Identify the physical, informational, and financial signatures.

**Deliverable:** a one-page analysis separating documented fact from inference, with evidence
grades.
**Acceptance criteria:** every factual claim has a source; every inference is labelled; at least
one claim is explicitly marked as unresolved in the public record.

### Exercise 9 — Data latency map (application, 40 minutes)

For a system you know, map the latency from physical event to analytic availability, step by
step.

**Deliverable:** a table of steps with latency in hours, and the end-to-end number.
**Acceptance criteria:** at least five steps; the end-to-end number stated in hours; one step
identified where the latency is organisational rather than technical.

### Exercise 10 — Decision memo (design, 45 minutes)

Write a one-page memo proposing one analytics intervention for a supply chain you know, framed
in the three-flow model.

**Deliverable:** the decision, the owner, the flow it acts on, the baseline, the expected effect,
and the failure mode that would make it harmful.
**Acceptance criteria:** the owner is a role, not a department; the baseline is measurable; the
harmful failure mode is specific and plausible.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
daily COGS    = 192,000,000 / 365 = 526,027
daily revenue = 240,000,000 / 365 = 657,534

DIO = 28,000,000 / 526,027 = 53.2 days
DSO = 40,000,000 / 657,534 = 60.8 days
DPO = 22,000,000 / 526,027 = 41.8 days

CCC = 53.2 + 60.8 - 41.8 = 72.2 days
cash tied up = 72.2 x 526,027 = EUR 37,979,000 (approximately EUR 38.0m)
```

Operationally: the distributor pays its suppliers 42 days after purchase, holds goods for 53
days, and collects from customers 61 days after sale. It must fund 72 days of cost. If the
distributor's borrowing rate is 6%, the annual financing cost of the cycle is approximately
`38.0m x 0.06 = EUR 2.28m`, which is about 1.2% of COGS. That is the number to put in front of
the CFO. The days figure alone is abstract; the currency figure is not.

A common wrong answer uses revenue for DIO, giving `28/657,534 = 42.6 days` and a CCC of 61.6
days. That understates the cash requirement by roughly €5.6m.

**Exercise 3, model answer outline.**

The most likely mechanism: the accuracy improvement came from better prediction of high-volume
items, while low-volume items — which drive OTIF failures because they are intermittent — got
worse. Aggregate accuracy rose because high-volume items dominate the denominator. OTIF fell
because OTIF is sensitive to small items. Inventory rose because safety stock was set on
forecast error rather than demand variability, and the model's improved average masked
worsening tail error.

Alternative explanations to test: (a) OTIF definition changed mid-period; (b) the demand
environment shifted (a promotion or a competitor exit); (c) inventory rose because of a
procurement decision unrelated to forecasting, such as a bulk buy.

The next measurement: forecast error by ABC class and by demand intermittency, plus OTIF
decomposed by the same segments. If the error improvement is concentrated in A items and OTIF
loss in C items, the hypothesis is supported.

**Exercise 4 acceptance notes.**

The correct answer will state the bucket explicitly. A CV computed on weekly data at one
echelon and monthly at another is not comparable, and a submission that does this should be
marked down regardless of the arithmetic. Amplification attribution should cite a mechanism
visible in the data: order day-of-week clustering implies batching; a spike followed by a
trough of similar magnitude implies forward buying.

**Exercise 5 rubric.**

A strong answer places the boundary at fabric (greige) or at undyed blank, holds 8–14 weeks of
greige, and accepts a 3–4 week dye-and-finish lead time in season. It quantifies the inventory
at the postponement point and notes that greige inventory is cheaper and less obsolescent than
finished goods. It states the reversal condition: if style risk falls (e.g., a basic replenishment
programme with verified sell-through), move the boundary right to reduce holding cost. A weak
answer recommends a single boundary without stating the season length or the obsolescence cost.

**Exercise 6 model answer.**

Two internal reports typically differ on three axes: the date used (requested versus confirmed),
completeness (line-level versus order-level), and exclusions (cancelled, rebooked, and
intercompany shipments). On the same underlying data the two can differ by 4–9 percentage points,
which is larger than most annual improvement programmes achieve. The deliverable should state a
single definition, the alternative, and the direction of the bias.

**Exercise 9 model answer.**

A typical latency chain: physical event at dock (0 h) → barcode scan at receiving (0.1 h) →
WMS transaction commit (0.5 h) → nightly extract to warehouse (8 h) → transformation and load
(2 h) → semantic layer refresh (1 h) → dashboard cache (1 h). End-to-end approximately 12.6 h,
which means the "real-time" dashboard is a day behind. The organisational step is usually the
WMS transaction commit, which is often performed at end of shift because the receiving clerk
batch-scans.

## 14. Interview questions

**1. "What is a supply chain?"**

A strong answer names flows, not functions: material, information, and money moving between
nodes with different owners and different cycle times. It mentions that the three are not
synchronised and that inventory is where the mismatch is stored. A weak answer lists
"procurement, manufacturing, logistics" and stops, which describes departments rather than a
system.

**2. "Why does demand variability increase upstream?"**

A strong answer gives the four mechanisms — signal processing, batching, price fluctuation,
shortage gaming — and notes that each is a design choice rather than a law of nature. A weak
answer says "because forecasts are bad," which is neither necessary nor sufficient.

**3. "Explain cash conversion cycle and its components."**

A strong answer gives the formula, states the denominators correctly (COGS for DIO and DPO,
revenue for DSO), and adds the operational meaning: DIO is how long goods sit, DSO is how long
customers take to pay, DPO is how long you take to pay. It also mentions that CCC can be
improved by transferring burden to suppliers. A weak answer gives the formula and stops.

**4. "A company improves DPO from 45 to 90 days and reports a CCC improvement. What do you
ask?"**

A strong answer asks about supplier concentration and financial health, whether prices rose, and
whether the terms are sustainable. It notes that DPO improvement is not a free variable. A weak
answer congratulates the company.

**5. "Where would you put the push–pull boundary for a fashion retailer?"**

A strong answer locates it at fabric or blank and justifies it with the ratio of style risk to
finishing lead time. It quantifies the inventory implication. A weak answer says "at the
warehouse" without engaging with the postponement logic.

**6. "How would you measure whether a promotion worked?"**

A strong answer separates incremental from pulled-forward demand, describes a baseline that
excludes the promotional period, and mentions the counterfactual problem. A weak answer compares
sales during the promotion to sales before it, which conflates level with timing.

**7. "What is the difference between a forecast, a plan, a target, and a commitment?"**

A strong answer defines each and notes that only the plan is controllable and only the
commitment carries penalty. A weak answer treats them as synonyms, which reveals that the
candidate has not worked inside a planning cycle.

**8. "Give an example of an analytics project that would fail because of a flow misunderstanding."**

A strong answer gives a specific, plausible case: a lead-time model that ignores contractual
commitments, or an inventory model on stale data, or a service metric on the wrong denominator.
It explains the mechanism, not just the outcome. A weak answer gives a generic "bad data" story.

**9. "How would you decide whether to hold inventory or expedite?"**

A strong answer frames both as costs: holding cost per unit per day against expedite premium,
plus the service consequence of each. It notes that expediting has a reliability cost. A weak
answer picks one without quantifying.

**10. "What does 'end-to-end visibility' actually mean, and what does it not solve?"**

A strong answer distinguishes visibility of events from visibility of intent, notes the
organisational barriers to sharing, and states that visibility does not create the authority to
act. A weak answer equates visibility with a control tower dashboard.

**11. "What is cost-to-serve and why is it hard?"**

A strong answer explains allocation of shared costs, the difference between average and marginal
cost-to-serve, and why order-level costing exposes unprofitable customers hidden by averaging.
A weak answer describes it as "total logistics cost."

**12. "If you had one number to describe a supply chain's financial efficiency, what would it
be and what would it hide?"**

A strong answer picks CCC or cash-to-cash and immediately names what it hides: service level,
supplier health, and the composition of the cycle. A weak answer picks a number and defends it
without limits.

## 15. Key terms

- **Three flows** — physical, informational, and financial movement through a network.
- **Node** — a location or organisation where material stops or is transformed.
- **Lane** — a directional connection with a mode and a lead time.
- **Echelon** — a level of the chain; **tier** — a depth of supplier relationship.
- **Push** — execution against forecast; **pull** — execution against consumption.
- **Push–pull boundary** — where the chain stops guessing and starts responding.
- **Postponement** — delaying differentiation to reduce forecast exposure.
- **Bullwhip effect** — upstream amplification of downstream demand variability.
- **Demand signal** — the observable proxy for true demand, often a distorted one.
- **Inventory position** — on-hand plus on-order minus backorders.
- **Pipeline inventory** — goods in motion and in buffers across a lane.
- **Safety stock** — buffer against variability; **cycle stock** — buffer against batching.
- **CSL** — cycle service level, probability of no stock-out in a cycle.
- **Fill rate** — fraction of demand units served from stock.
- **OTIF** — on-time in-full; definition-dependent.
- **DIO / DSO / DPO** — days inventory, sales, and payables outstanding.
- **CCC** — cash conversion cycle, `DIO + DSO − DPO`.
- **Landed cost** — delivered cost including freight, duty, and fees.
- **Incoterms** — ICC rules for delivery responsibility and risk transfer.
- **Cost-to-serve** — cost attributable to a specific customer, channel, or order.
- **Forecast / plan / target / commitment** — belief, intention, aspiration, promise.
- **EPCIS** — GS1 standard for event data about objects moving through a chain.
- **GTIN / SSCC / GLN** — product, logistics unit, and location identifiers.
- **ASN** — advance ship notice, the information that precedes physical arrival.
- **Control tower** — an organisational and technical layer for cross-network visibility.
- **Decision governance** — the rules for who may act on an automated recommendation.

## 16. Sources

**Peer-reviewed and academic**

- Lee, H. L., Padmanabhan, V., and Whang, S. (1997). "Information Distortion in a Supply Chain:
  The Bullwhip Effect." *Management Science*, 43(4), 546–558. DOI: 10.1287/mnsc.43.4.546.
- Lee, H. L., Padmanabhan, V., and Whang, S. (1997). "The Bullwhip Effect in Supply Chains."
  *Sloan Management Review*, 38(3), 93–102.
- Forrester, J. W. (1961). *Industrial Dynamics*. MIT Press. The original system-dynamics
  treatment of amplification.
- Chen, F., Drezner, Z., Ryan, J. K., and Simchi-Levi, D. (2000). "Quantifying the Bullwhip
  Effect in a Simple Supply Chain." *Management Science*, 46(3), 436–443.
- Hammond, J. H. (1994). "Barilla SpA (A)." Harvard Business School Case 9-694-046.

**Standards and official**

- GS1 General Specifications (current edition), including GTIN, SSCC, GLN, and EPCIS 2.0.
- Incoterms 2020, International Chamber of Commerce.
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606 (FASB).
- Regulation (EU) 2024/1781 on ecodesign for sustainable products (ESPR).
- Regulation (EU) 2023/956 establishing the carbon border adjustment mechanism (CBAM).
- Council Directive (EU) 2025/… on VAT in the Digital Age — consult the current consolidated text
  before relying on any implementation date.

**Industry and analyst**

- Gartner, "Top Supply Chain Technology Trends for 2026," press release, 30 June 2026.
  Vendor-analyst; treat as framing, not evidence of adoption.
- Gartner, "Predicts 2026: Supply Chain Planning," April 2026.
- BearingPoint, "Next Gen Integrated Planning," 2026 survey of 193 European executives.
- ABC Supply Chain, S&OP/IBP practitioner survey, November 2025 (164 respondents, 54 countries).
- ORI / SCPI / Gatepoint Research, 2026 supply chain planning survey.

**Note on citations.** Figures attributed to surveys are reported as the surveys state them and
are not independently verified. Where a specific number is uncertain, the text says so rather
than quoting precision it cannot support.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_decision-stack.md`](02_decision-stack.md)
- Related: [`05_inventory-and-service.md`](05_inventory-and-service.md)
- Related: [`06_cost-and-working-capital.md`](06_cost-and-working-capital.md)
- Related: [`10_archetypes-and-metric-conflicts.md`](10_archetypes-and-metric-conflicts.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Deeper treatment of working capital: [`../../../course/42_treasury_credit_financial_supply_chain.md`](../../../course/42_treasury_credit_financial_supply_chain.md)
- Deeper treatment of event standards: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Return to the part map: [`../_index.md`](../_index.md)
