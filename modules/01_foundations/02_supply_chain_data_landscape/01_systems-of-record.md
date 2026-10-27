# Systems of Record

> Module 02 — The Supply Chain Data Landscape
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01.01, 01.02, 01.03
> Version 2026.5

---

## 1. Why this topic exists

Ask five people in a distribution business how many units of a fast-moving item are in the
main warehouse and you will get five numbers. The ERP will say 1,240 cases. The warehouse
management system will say 1,186. The order management system will offer 1,102. The transport
management system will say zero, because nothing is on a truck. Finance will say EUR 41,300.
Every one of those numbers is computed correctly. Every one answers a different question. And
in most companies nobody has written down which question each one answers. That is the problem
this topic exists to solve.

A **system of record** is not a server, a database, or a vendor product. It is a statement
about authority: for this fact, in this scope, this system's value is the one the organisation
has agreed to act on and to be audited against. Authority is a social and contractual
arrangement that happens to be implemented in software. When you ask "where does this number
come from," the useful answer is not a system name. It is a sentence of the form: *this system
owns this fact at this grain, for this population, as of this lag, and here is what it
distorts.*

Data scientists are usually handed a data warehouse and told it is the source of truth. It is
not. It is a copy of copies, each copy made at a different time from a different owner, joined
on keys that were themselves maintained somewhere else. A model that treats the warehouse as
ground truth is a model that has silently accepted an unknown number of reconciliation
decisions made by people it will never meet.

**Why this is a 2026 problem specifically.** The record layer is unusually unstable right now.
SAP's mainstream maintenance for its legacy ERP Central Component (ECC) ends on 31 December
2027 for the most widely deployed enhancement packages, with optional extended maintenance to
2030 and a private-cloud option announced in 2025 that stretches some estates further. That
means a large fraction of the world's supply chain transaction data is either mid-migration
or about to be. Order management is moving from monolith to composable, API-first platforms.
Event-layer mandates in food and pharmaceuticals are forcing companies to build a system that
records *what physically happened*, which most of them have never had. When systems move,
authority is renegotiated, and renegotiation is exactly when analytics breaks.

Three failures recur, and all three are authority failures rather than modelling failures.

1. **The forecast trained on the wrong inventory series.** An analyst builds a replenishment
   model on ERP on-hand because that is the table with the cleanest history. ERP on-hand
   includes stock reserved for confirmed orders and excludes stock received but not yet posted.
   The model over-orders committed items and under-orders fast-moving ones, and accuracy looks
   acceptable in aggregate while service degrades on the items that carry margin.

2. **The service metric computed on the promise, not the request.** An order management system
   overwrites the customer's requested date with the date the company promised. On-time delivery
   measured against the promise measures the company's ability to keep its own promises, which
   is useful but is not the customer's experience. Two teams report different service levels
   from the same orders and both are right.

3. **The "real-time" dashboard that is a day old.** A control tower reads a warehouse table
   refreshed nightly. Operators intervene at 09:00 using positions as of 23:00 the previous day.
   The dashboard is accurate, the latency is undocumented, and the intervention arrives after
   the decision it was meant to inform.

Each failure has the same shape: a number was taken from a system without asking what that
system is authoritative for. The fix is a map, not a model.

This is the first topic in Module 02 because every later topic is a refinement of it. Master
data (Topic 02) is the join layer between records. Transaction data (Topic 03) is the grain at
which records are written. Data movement (Topic 06) is how copies travel. Data quality
(Topic 08) is measured against the owner's definition, not against an abstract ideal. Source
selection (Topic 10) is the act of choosing which record to trust for a decision. None of those
topics is intelligible until you can say what a system of record is.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** a system of record as a scoped authority claim, and **distinguish** it from a
   system of engagement, a system of intelligence, and a data warehouse copy.
2. **Name**, for a described supply chain, which system owns each of at least ten facts —
   including inventory quantity, inventory value, order promise, load tender, engineering
   bill of materials, and consumer sale.
3. **State** the native grain and typical latency of the eight systems covered here, and
   **explain** why a join across two of them at the wrong grain produces a plausible and wrong
   answer.
4. **Distinguish** financial truth from physical truth, and **justify** which one a given
   decision requires.
5. **Build** a reconciliation bridge between two inventory numbers, decomposing the gap into
   timing, status, allocation, and shrinkage components, with each component signed and
   sourced.
6. **Trace** one order through five systems and **explain** why the status codes and quantities
   differ without any system being wrong.
7. **Classify** a proposed analytics tool as a record or a system of intelligence, and
   **state** the consequence of confusing them.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one sentence: **one fact, one owner, many copies — and the copy you are reading is not
the owner.**

Everything else follows from four properties.

**Property 1 — Authority is scoped, not global.** No system is authoritative for "inventory."
The ERP is authoritative for inventory *value* and for the *financial consequence* of a
movement. The WMS is authoritative for *position by bin*. The OMS is authoritative for *what
has been promised to a customer*. The TMS is authoritative for *what is on a vehicle*. A
question that does not name its scope cannot have an authoritative answer, which is why "which
inventory number is right" is not a question — it is an argument waiting to happen.

**Property 2 — Every copy carries a lag and a transformation.** A warehouse table is not the
ERP. It is a projection of the ERP made at some time, filtered by some rule, aggregated to some
grain, and joined to dimensions maintained elsewhere. Each of those four operations is a
decision, usually undocumented, usually made years ago by someone who has left. When a model
behaves oddly, the cause is often in the projection and not in the data.

**Property 3 — The record layer is not the decision layer.** A plan, a forecast, a score, a
recommendation, and an allocation proposal are not records. They are derived objects that
should carry a version, a timestamp, and an owner. Most planning systems overwrite them in
place, which destroys the ability to evaluate whether the plan was any good. Confusing a plan
with a record is the most expensive category error in supply chain analytics, because it makes
the past unobservable.

**Property 4 — Physical truth and financial truth diverge by design, and the divergence is
information.** The WMS knows a pallet is in bin B-14. The ERP knows the pallet is worth
EUR 8,200 and belongs to a specific cost centre. They will disagree at any instant, because
one updates on the scan and the other updates on the posting. The gap is not an error to be
eliminated. It is a measurable quantity with causes, and the causes tell you where the process
is leaking.

### 3.1 The layer stack

Draw this once and keep it. It is the whole topic in one picture.

```text
   +-------------------------------------------------------------------+
   |  DECISION LAYER      forecasts, plans, scores, allocations        |
   |  (no authority)      planning tool / models / dashboards          |
   +-------------------------------------------------------------------+
                     ^  reads copies; writes nothing back as truth
                     |
   +-------------------------------------------------------------------+
   |  RECORD LAYER        one fact, one owner, per scope               |
   |                                                                   |
   |   COMMERCIAL          PHYSICAL              FINANCIAL             |
   |   CRM   customer      WMS   bin position    ERP   value, GL       |
   |   OMS   promise       MES   output, scrap   ERP   AP / AR         |
   |   PLM   design, BOM   TMS   load, tender    ERP   cost            |
   |   POS   consumer sale Event layer: what happened, where, when     |
   +-------------------------------------------------------------------+
                     |
                     v  extract, transform, reconcile
   +-------------------------------------------------------------------+
   |  ANALYTIC LAYER      warehouse, lake, semantic model, features    |
   |  (a copy: every row has an as-of time and a source system)        |
   +-------------------------------------------------------------------+
```

Three reading rules for this diagram:

- Arrows point **up** from record to decision, and **down** from record to analytics. There is
  no arrow from the decision layer back into the record layer that carries authority. When a
  planner overrides a forecast, that override is a new record with an owner, not a correction
  of the old one.
- The analytic layer is drawn below the record layer on purpose. It is downstream, therefore
  the most convenient place to work and the least authoritative place to settle a dispute.
- The event layer sits inside the record layer because events are facts: a shipment departure
  either happened or it did not, and it is not derived from an order.

### 3.2 The four inventories

This is the picture to redraw when someone says the inventory number is wrong.

```text
   same SKU (stock-keeping unit), same minute, four numbers, four questions

   ERP     on hand, posted        1,240 cs   owned + valued, after posting
   WMS     in the building        1,186 cs   by bin, minus quality hold
   OMS     available to sell      1,102 cs   minus allocated to open orders
   TMS     on a vehicle               0 cs   nothing loaded yet
   FIN     inventory value    EUR 41,300     standard cost x ERP quantity

   gaps:  54 cs  physical vs posted   (timing + hold + count)
          84 cs  sellable vs physical (allocation)
```

The 54-case gap between ERP and WMS is a **posting bridge**: goods received at the dock but not
yet posted, goods picked but not yet confirmed out, goods on quality hold, and the residue of
the last cycle count. The 84-case gap between WMS and OMS is an **allocation bridge**: stock
that exists and is physically available but has already been committed to someone else.

Neither gap is a defect. A defect is a gap with no explanation.

### 3.3 The authority test

Before you use any number, run five questions. They take a minute and they prevent most of the
errors in this module.

| # | Question | Why it changes the answer |
|---:|---|---|
| 1 | Which system is the owner of this fact? | A copy may be stale, filtered, or re-derived |
| 2 | What is the owner's native grain? | Joining a bin-level fact to an item-day fact loses or duplicates |
| 3 | What is the lag between event and record? | A lagged number answers yesterday's question |
| 4 | What does the owner deliberately exclude? | Every system has a status filter that removes real stock |
| 5 | What decision will this number change? | If no decision changes, the precision is decoration |

Question 4 is the one analysts skip. Every WMS excludes blocked, hold, and in-transit stock
from its "available" figure. Every ERP excludes unposted receipts. Those exclusions are
correct for the owner's purpose and wrong for yours, and they are almost never in the
documentation you were given.

> **Note:** The five questions produce a *provenance statement*, not a proof. "WMS, bin grain,
> 15-minute lag, excludes quality hold and in-transit" is a claim you can test and a claim you
> can put in a data dictionary. "It came from the warehouse" is not.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| System of record (SoR) | The system whose value for a scoped fact the organisation acts on and is audited against | Used to mean "the main database" | Named system plus scope statement |
| System of engagement | The surface where a human or partner interacts with a process | Confused with the record it writes to | Channel or application |
| System of intelligence (SoI) | A layer that reads records, derives decisions, and orchestrates them into workflow | Treated as authoritative because it is newer | Model or decision service |
| System of action | The layer where an automated agent executes a decision | Confused with the record of that execution | Agent or workflow engine |
| Data authority | The organisational right to define a fact's correct value | Assumed to follow from system ownership | Governance assignment |
| Native grain | The finest level at which a system writes a fact | Assumed to be the level it reports at | Key combination |
| Latency | Elapsed time from physical or commercial event to record availability | Confused with refresh frequency | Minutes or hours |
| Freshness | Age of the data at the moment of reading | Used interchangeably with latency | Timestamp delta |
| Replication lag | Delay between a change at the owner and its appearance in a copy | Ignored in dashboards | Seconds to hours |
| Posting | The act of writing a financial consequence into the ledger | Confused with the physical event | Document number |
| Sub-ledger | The detailed book for one class of transaction, feeding the GL | Confused with the GL itself | Account plus document |
| General ledger (GL) | The consolidated accounting record of all postings | Treated as an operational data source | Account, period, currency |
| Enterprise resource planning (ERP) | The system integrating finance — including accounts payable (AP) and accounts receivable (AR) — materials, production orders, and often sales | Treated as the operational truth for everything | Company code, plant, material |
| Warehouse management system (WMS) | The system that directs and records work inside a warehouse | Confused with an inventory ledger | Warehouse, bin, licence plate |
| Warehouse execution system (WES) | The layer that orchestrates automation and work release inside the warehouse | Used as a synonym for WMS | Wave or work queue |
| Warehouse control system (WCS) | The real-time control layer for conveyors, sorters, and robotics | Confused with WES | Equipment controller |
| Transportation management system (TMS) | The system that plans, tenders, and settles freight movements | Assumed to hold delivery proof | Load, shipment, leg |
| Order management system (OMS) | The system that captures orders, allocates inventory, and holds the promise | Confused with the ERP sales module | Order line, promise date |
| Distributed order management (DOM) | The routing function that decides which node fulfils an order | Treated as a separate system from OMS | Sourcing rule, node |
| Point of sale (POS) | The system recording the consumer transaction at the point of purchase | Assumed to be visible upstream | Transaction line, store, day |
| Product information management (PIM) | The system of record for sellable item attributes and content | Confused with PLM | Item, market, channel |
| Product lifecycle management (PLM) | The system of record for design, specification, and engineering change | Assumed to own the manufacturing BOM | Item, revision, effectivity |
| Engineering BOM (eBOM) | The bill of materials as designed, by function | Treated as buildable | Item plus revision |
| Manufacturing BOM (mBOM) | The bill of materials as built, by operation | Treated as the design | Item plus revision plus plant |
| Sales BOM (sBOM) | The bill of materials as sold, by configurable option | Confused with the mBOM | Item plus configuration |
| Engineering change order (ECO) | The controlled document authorising a design change | Confused with the change request | Change number, effectivity date |
| Customer relationship management (CRM) | The system of record for accounts, contacts, opportunities, and service cases | Assumed to hold order history | Account, opportunity |
| Manufacturing execution system (MES) | The system recording production execution, consumption, and quality at operation level | Confused with the ERP production module | Work order, operation, shift |
| ISA-95 level | The reference model separating business (L4) from manufacturing operations (L3) and control (L0–L2) | Treated as a regulation | Level number |
| EPCIS | GS1 standard for sharing event data about objects moving through a chain | Used as a synonym for traceability | Event, object identifier |
| Object event | An EPCIS record that something happened to an object at a place and time | Confused with an order | Event ID, EPC or GTIN plus lot |
| Aggregation event | An EPCIS record that objects were packed into or unpacked from a parent | Confused with a shipment | Parent and child identifiers |
| Transformation event | An EPCIS record that inputs became outputs in a process | Ignored in most food and pharma models | Input and output quantities |
| Advance ship notice (ASN) | The message announcing an inbound shipment before it arrives | Treated as proof of delivery | EDI 856 or DESADV |
| Available to promise (ATP) | Quantity that can be committed to a new order given supply and existing commitments | Confused with on-hand | Units, by date |
| Available to sell (ATS) | Quantity a channel is permitted to sell under its own rules | Confused with ATP | Units, by channel |
| Allocation | The assignment of specific supply to specific demand | Confused with reservation | Units, order line, lot |
| Reservation | A hold placed on stock for a specific demand | Confused with physical pick | Units, bin, document |
| Backflush | The automatic deduction of components on production confirmation | Assumed to be accurate | Units per operation |
| Cycle count | A periodic partial physical count used to maintain accuracy | Confused with full physical inventory | Count document |
| Book balance | The quantity the system believes it holds | Treated as physical truth | Units by location |
| Goods receipt note (GRN) | The document recording physical receipt | Confused with the posted receipt | Receipt document number |
| Three-way match | The control comparing purchase order, receipt, and invoice before payment | Assumed to be automated | Document triplet |
| Golden record | The single agreed best version of a master-data entity | Confused with a system of record | Entity ID plus survivorship rule |

Six of these carry most of the confusion, and each deserves a paragraph.

**System of record versus single source of truth.** "Single source of truth" is a phrase that
sounds like a design goal and functions as an excuse to stop thinking. There is no single
source of truth for a supply chain, because there is no single truth. There is a set of scoped
authorities and a set of bridges between them. A company that pursues one number in one system
will either fail or, worse, succeed by deleting the distinctions that made the numbers useful.
The defensible goal is: *one owner per fact, one documented bridge per pair of owners, one
as-of timestamp per copy.*

**Native grain.** The grain is the key combination at which the system writes a row. A WMS
writes at warehouse–bin–licence plate–lot. An ERP writes inventory at plant–material–
storage location. An OMS writes at order line. A TMS writes at shipment leg. Joining a
bin-grain fact to a plant-grain fact without aggregating first will either duplicate the
plant figure once per bin or silently drop bins. Neither failure raises an error. Both produce
a number that looks reasonable.

**Latency versus freshness.** Latency is a property of the pipeline: how long after the event
does the record exist. Freshness is a property of the row you are holding: how old is it.
A dashboard can have low latency and poor freshness if the pipeline is fast but the job has
not run. Confusing the two produces the classic claim that a system is "real-time" because its
API responds in milliseconds.

**ATP versus ATS.** ATP is a supply question: what can we commit given existing supply and
commitments. ATS is a commercial question: what is this channel allowed to sell, given
allocation rules, contract terms, and channel priority. A marketplace seller and a wholesale
account can face the same physical stock with different ATS. Reporting one as the other
produces systematic over-selling in the channel with the looser rule.

**Posting versus event.** A receipt event happens when the truck is unloaded. A posting happens
when someone confirms the receipt in the ERP. Between those two moments the goods exist, are
insured, and are invisible to finance. That window is where a large share of inventory
discrepancies are born, and it is a process design choice, not a data error.

## 5. The mechanics, step by step

This section explains how a fact acquires an owner, then walks the eight systems, then treats
the overlap problem directly.

### 5.1 How a fact acquires an owner

**Step 1. A physical or commercial event occurs.** A pallet is unloaded. A customer clicks buy.
A machine completes an operation. A supplier signs a contract. The event exists whether or not
anything records it.

**Step 2. The system closest to the event captures it.** Proximity decides candidacy. The WMS
is closest to the unload, so the WMS captures the receipt. The OMS is closest to the click, so
the OMS captures the order. Proximity is why the physical systems win the physical facts and
the commercial systems win the commercial facts, and it is why the ERP — which is close to
nothing — wins only the financial facts.

**Step 3. The capture is transformed into a transaction.** The scan becomes a receipt line; the
click becomes an order line; the operation confirmation becomes a consumption record. This is
where grain is set, and grain is set once. Every later aggregation is a loss.

**Step 4. The financial consequence is posted.** Value moves. Inventory value rises, cost is
absorbed, a liability appears. The posting is the ERP's native act. It is always later than the
event, sometimes by minutes and sometimes by days.

**Step 5. Copies are made and a decision is taken on a copy.** Extracts, APIs, event streams, and
flat files carry the fact to the analytic layer. The planner, the model, the control tower, and
the dashboard all read copies. The decision is real; the data behind it is a replica, carrying a
lag, a filter, and a schema that is not the owner's schema.

The crucial asymmetry: **steps 1 to 3 are cheap to get right at design time and expensive to
fix later.** A grain chosen wrongly in 2011 is still wrong in 2026, and every model built on it
inherits the error.

### 5.2 The eight systems at a glance

| System | Owns (scope) | Native grain | Typical latency | Question it answers well | Question it is wrongly expected to answer | Known distortion |
|---|---|---|---|---|---|---|
| ERP | Inventory value, cost, payables, receivables, material master, production orders | Plant–material–storage location; document line | Minutes to hours (batch posting) | What is this worth, what did it cost, what do we owe | Where is the pallet | Posting lag; reservations netted into on-hand; standard cost drift |
| WMS | Physical position, bin, licence plate, task state | Warehouse–bin–licence plate (LPN)–lot | Seconds to minutes | What is in the building and where | What may we promise a customer | Excludes hold and in-transit; ghost stock from unconfirmed picks |
| TMS | Loads, tenders, rates, freight settlement | Shipment leg; load | Minutes to hours | What is moving, on what, at what rate | Where the goods are right now | Status events from partners arrive late or never; planning vs execution loads differ |
| OMS | Order capture, allocation, promise | Order line | Seconds | What did the customer ask for and what did we promise | Whether the goods shipped | Promise overwrites request; split and merged lines; cancelled lines retained |
| POS | Consumer transaction at the point of purchase | Transaction line, store, trading day | Seconds on-site, days to weeks upstream | What consumers actually bought | What the retailer will reorder | Basket distortion, returns not netted, panel vs census coverage |
| PLM | Design definition, eBOM, specification, change | Item–revision–effectivity | Days | What is the product designed to be | What we are building today | Design intent versus as-built drift; effectivity dates not honoured |
| CRM | Account, contact, opportunity, service case | Account–interaction | Minutes | Who the customer is, what is in the pipeline | What the customer ordered | Pipeline stages are opinions; duplicate accounts; order history is a copy |
| MES | Execution, consumption, scrap, quality, genealogy | Work order–operation–shift | Seconds | What actually happened on the line | What the schedule should be | Backflush error accumulates silently; downtime coded by default |
| Event layer | What happened to which object, where, when, why | Object plus event time | Seconds to hours | Provenance and chain of custody | Business context and commercial intent | Sparse coverage; event identity collisions across partners |

Read the table by column, not by row. The column "question it is wrongly expected to answer" is
the most useful one, because that is where organisations actually get hurt.

### 5.3 ERP: the financial truth

The ERP is the only system that can answer "what is this worth." That is not a technical
accident. It is the system's purpose: it exists to close the books, satisfy the auditor, and
report to the market. Everything else it does — material master, purchase orders, production
orders, sales orders — is in service of that.

Its native grain is the document line at a plant or storage location. Its latency is the
posting latency, which is usually minutes inside a process and hours across batch jobs. Its
characteristic distortion is that **it represents the world after commercial commitments have
been netted out**. On-hand in the ERP is not "what is in the building." It is "what is in the
building, valued, excluding stock reserved for confirmed demand and excluding receipts not yet
posted."

That is precisely right for finance and precisely wrong for a picking decision. A replenishment
model trained on ERP on-hand has already been told the answer to the question it is trying to
predict.

ERP replacement is also the largest source of short-term data discontinuity in the industry: a
migration changes field names, posting logic, and sometimes the definition of a stock category,
so historic series break at the cutover, usually without a flag.

> **Note:** A migration is not only a technology event. It is a renegotiation of authority.
> After a go-live, "which system owns this fact" must be asked again, because the old answer may
> now be wrong.

### 5.4 WMS: the physical truth

The WMS owns position. It knows that 6 pallets of an item sit across 3 bins, that 2 are on
quality hold, that 1 is staged for a wave, and that the picker has 4 minutes left on the task.
That resolution is the point. The WMS exists to direct work, and work happens at a bin.

Its grain is warehouse–bin–licence plate–lot. Its latency is seconds to minutes, because a task
confirmation is a scan. Its distortions are three:

- **It excludes what it cannot act on.** Hold stock, quarantine, and stock in transit are
  typically excluded from the available figure. Correct for picking, wrong for valuation.
- **It accumulates ghost stock.** A pick confirmed but not physically completed leaves a
  quantity that exists in the system and not on the shelf. Ghost stock is invisible until a
  cycle count finds it, and it inflates availability in exactly the period when accuracy
  matters most.
- **Its accuracy depends on discipline, not on software.** A WMS with a 94% location accuracy
  rate produces confident, precise, wrong answers.

The relationship between ERP and WMS is the single most important authority boundary in
warehousing, and the industry has converged on a position: the WMS owns *quantity and
position*; the ERP owns *valuation and the transaction*. Vendor reference architectures state
this explicitly, and note that automated continuous synchronisation is usually counterproductive
because snapshot timing produces false exceptions. The practical pattern is a scheduled
snapshot comparison with a governed exception queue, not a forced sync.

### 5.5 TMS: the movement truth

The TMS owns loads, tenders, rates, and freight settlement. A tender is a commercial act: the
shipper offers a load to a carrier at a rate, and the carrier accepts or declines. That
transaction is the TMS's reason to exist, and freight cost is where it earns its return.

Its grain is the shipment leg or the load. Its latency is minutes for planning and hours to days
for status, because status events frequently come from a carrier's system through an API or an
EDI message that is batched. Its characteristic distortion is that **planning loads and
execution loads are different objects**. The load the TMS planned may have been consolidated,
split, or re-tendered, and the resulting identifiers do not always reconcile to the shipment the
customer sees.

The question the TMS is wrongly expected to answer is "where is my shipment right now." It can
answer that only to the extent carriers report events, and a TMS without a telematics or event
feed is a rate and settlement engine.

### 5.6 OMS: the promise truth

The OMS owns the order and, critically, the promise. In omnichannel retail it has grown from a
capture system into a routing engine: it decides which node fulfils which line, whether to ship
from a store, whether to split the order, and what date to promise. The routing rules are
commercial policy expressed as code.

Its grain is the order line. Its latency is seconds. Its distortions are consequential:

- **The promise overwrites the request.** Most OMS designs store one date field. Once the
  promise is written, the customer's original request is gone, and on-time performance measured
  against the promise flatters the operation.
- **Lines split, merge, and cancel.** A single customer order can become three fulfilment
  requests and two shipments. Counting orders at the order header and counting shipments at the
  parcel level produces two different denominators for the same service metric.
- **Allocation is not reservation and neither is a pick.** The OMS can allocate supply it does
  not physically have, because allocation is a promise and the pick is a fact. This is the
  single largest source of the WMS–OMS gap.

The OMS is also where the "system of intelligence" confusion is most common. Routing rules look
like optimisation; they are usually a deterministic rule set with a priority order, and calling
it AI does not change what it can be audited against.

### 5.7 POS: the consumer truth, conditionally

POS is the only system that records what a consumer actually bought. Everything upstream
records what an intermediary ordered, which is a different and more distorted series. That
makes POS the most valuable demand signal in consumer goods and the hardest to obtain.

Its grain is the transaction line, by store, by trading day. Its latency depends entirely on the
access arrangement: seconds inside the retailer, hours to days through a direct retailer API,
and typically one to two weeks through syndicated providers. Its distortions are coverage and
composition: a panel covers a sample, a census covers the population, returns and voids may not
be netted, and the trading day boundary is set by the retailer's calendar rather than yours.

The question POS is wrongly expected to answer is "what will the retailer order next week."
It answers what sold. The gap between sell-in and sell-out is channel inventory, and channel
inventory is usually the least visible number in the chain.

Data-sharing arrangements are contractual, not technical. Syndicated licences typically restrict
re-identification and resale, and often restrict combining the data with a retailer's loyalty
data without an explicit fusion clause. An analyst who assumes POS access is a pipeline problem
will discover it is a contract problem.

### 5.8 PLM: the design truth

PLM owns the product definition: the engineering bill of materials, specifications,
computer-aided design (CAD) structure, and the change record. It answers "what is this product
designed to be, and what changed, when, and why."

Its grain is item–revision–effectivity. Its latency is days, because a change is a governed
process with review and approval. Its characteristic distortion is the gap between design
intent and as-built reality.

The BOM question is where organisations get into trouble, because there is not one BOM. There
is an engineering BOM organised by function, a manufacturing BOM organised by operation, and a
sales BOM organised by configuration. Each is owned by a different function and each is correct
for its purpose. A single monolithic BOM fails because engineering, manufacturing, procurement,
and after-sales have genuinely conflicting requirements; the workable pattern is separate views
with governed synchronisation and change-impact analysis. PLM typically owns the eBOM and the
change record; ERP typically owns the mBOM as executed; the interface between them is one of the
weakest links in most manufacturing data landscapes.

### 5.9 CRM: the commercial truth, loosely

CRM owns the account, the contact, the opportunity, and the service case. It answers "who is
this customer, what are we trying to sell them, and what have they complained about."

Its grain is the account–interaction. Its latency is minutes. Its distortion is that **pipeline
is an opinion**, not a fact. A stage field is set by a salesperson with an incentive, and the
conversion probabilities attached to stages are usually inherited from a template rather than
estimated from the company's own history. The question CRM is wrongly expected to answer is
"what did this customer order," which lives in the ERP or the OMS; CRM holds a scheduled copy.

### 5.10 MES: the execution truth

The MES owns what happened on the line: which work order ran, on which resource, with which
consumption, which scrap, which quality result, and which genealogy. Under the ISA-95 reference
model it sits at Level 3, between the business systems at Level 4 and the control systems at
Levels 0 to 2, and ISA-95 is explicitly concerned with the interface between Levels 3 and 4.

Its grain is work order–operation–shift, often finer. Its latency is seconds. Its distortions
are backflush error and coded downtime. Backflush deducts components automatically on
confirmation, assuming the standard quantity was consumed. When it was not, the error
accumulates in inventory and appears later as an unexplained shortage. Downtime reason codes
default to the first item in the list, which turns a nuanced reliability problem into a
reporting artefact.

The MES is where the physical and financial views separate most visibly. Trust the MES for
floor execution and the ERP for accounting outcomes; make the difference explainable rather
than trying to make it zero.

### 5.11 The overlap problem

Four systems hold an inventory number. Two hold an order status. Three hold a delivery date.
This is not a design flaw to be eliminated. It is the structure of the problem, and the job is
to reconcile rather than to unify.

**Inventory, four definitions.**

| System | Field | Means | Excludes | Answers |
|---|---|---|---|---|
| ERP | On-hand quantity | Posted, owned, valued stock | Unposted receipts; may net reservations | What do we own and what is it worth |
| WMS | Available quantity | Physically present and pickable | Hold, quarantine, staged, in transit | What can we pick now |
| OMS | Available to sell | Unallocated supply net of commitments | Stock already promised | What may we promise a new customer |
| TMS | In-transit quantity | Loaded and moving | Everything at rest | What is arriving |

A reconciliation is not a comparison. A comparison says "the numbers differ by 54." A
reconciliation says "the numbers differ by 54, of which 22 is staged for open waves, 18 is on
quality hold, 9 is received but not posted, and 5 is cycle-count variance." Each component is
signed, sourced, and owned by a named process.

**Order status, two definitions.** The OMS status is a commercial state: captured, allocated,
released, shipped, delivered. The ERP status is a financial state: open, delivery created,
goods issued, invoiced, paid. They advance at different times and for different reasons, so an
order can be commercially "shipped" and financially "not goods-issued."

### 5.12 Systems of intelligence are not records

A planning tool reads records, applies a model, and produces a plan. The plan is a proposal. It
is not a record of anything that happened, and it is not authoritative for any fact. The
distinction matters for three reasons. **Versioning:** a record is append-only in principle, a
plan is overwritten in practice, and an unversioned plan cannot be evaluated. **Auditability:**
a record answers what happened, a model answers what it thinks should have happened, and the two
must not be filed in the same place. **Override semantics:** when a planner overrides the model,
the override is a new fact about a human decision, and overwriting the recommendation destroys
all future override analysis.

The practical test: **can you reconstruct, for any past date, what the system believed and what
a human did?** If not, you are holding a system of intelligence that has been mistaken for a
record.

> **Note:** Vendors increasingly describe planning and routing tools as "systems of record for
> decisions." That phrase is a category error dressed as a feature. A decision is an event; it
> can be recorded, and the record of it can be authoritative — but the tool that produced it is
> not thereby authoritative for the facts it read.

## 6. Worked example

Northwind Foods, the consumer goods company from Module 01, receives an order from a retail
customer. We follow that single order through five systems and reconcile what each holds. All
figures are illustrative and internally consistent.

### 6.1 The order

The retailer orders 120 cases of a single SKU for delivery to its regional distribution centre.
The order is placed Monday 09:12. The customer requests delivery on the following Tuesday.

### 6.2 What each system holds

| System | Identifier | Status or state | Quantity | Timestamp held | What it believes |
|---|---|---|---:|---|---|
| OMS | SO-4471 line 10 | ALLOCATED | 120 cs | Mon 09:14 | Promised Tue, 120 cases, from DC East |
| WMS | Wave W-2231, task 40 | PICKED | 118 cs | Mon 16:40 | 118 picked; 2 short in bin A-07 |
| TMS | Load L-8891 | TENDERED | 118 cs, 1,240 kg | Mon 18:05 | Carrier has not accepted |
| ERP | Delivery 8077312 | GOODS ISSUED, not invoiced | 120 cs | Tue 07:30 | Goods left; invoice pending |
| CRM | Account 10-4471 | — | — | Tue 07:31 | Service case open: short delivery |
| Event layer | ObjectEvent, SSCC | SHIPPED | 118 cs | Tue 07:32 | 2 pallets departed DC East |

Read the table twice. Six records, six states, no system malfunctioning, and a customer who
will receive 118 cases against a promise of 120. The full trace runs CAPTURED, ALLOCATED,
ORDER CREATED, RELEASED, PICKED, TENDERED, ACCEPTED, GOODS ISSUED, SHIPPED, INVOICED — ten
states across five systems for one order, of which only one carries the physically true
quantity of 118.

### 6.3 Where they disagree, and why

| Disagreement | Systems | Size | Cause | Owner of the cause |
|---|---|---:|---|---|
| Ordered versus picked | OMS vs WMS | 2 cs | Bin short-pick; 2 cases physically absent | Cycle-count process |
| Picked versus invoiced | WMS vs ERP | 2 cs | Invoice raised on ordered quantity, not shipped | Billing rule |
| Tendered versus shipped | TMS vs event layer | 0 cs | Tender quantity is correct; status is stale | Carrier acceptance lag |
| Promise versus request | OMS field | 0 d | Promise date overwrote the requested date | OMS date design |
| Goods issued versus invoiced | ERP fields | — | Goods issue precedes invoice by design | Billing cycle |

The 2-case gap is the one that becomes a commercial event. It generates a credit note, a
service case in CRM, and a deduction on the retailer's remittance. Three systems will record
three consequences of one physical short-pick.

### 6.4 The inventory bridge

Later that week, someone asks why the ERP and the WMS disagree on this SKU by 54 cases. The
reconciliation:

| Component | Cases | Sign | Source | Explanation |
|---|---:|---|---|---|
| ERP on-hand, posted | 1,240 | base | ERP | Posted, valued, owned |
| Less: staged for open waves | −22 | − | WMS task list | Picked but not confirmed out |
| Less: quality hold | −18 | − | WMS status | Awaiting disposition |
| Less: received, not posted | −9 | − | WMS receipt log | Dock receipt without ERP posting |
| Less: cycle-count variance | −5 | − | Count document | Physical less than book |
| WMS available | 1,186 | = | WMS | Physically pickable now |

Check the arithmetic: `1,240 − 22 − 18 − 9 − 5 = 1,186`. The bridge balances, and every
component has an owner and a document.

Now the second bridge, from physical to sellable:

| Component | Cases | Sign | Source | Explanation |
|---|---:|---|---|---|
| WMS available | 1,186 | base | WMS | Physically pickable |
| Less: allocated to open orders | −84 | − | OMS | Committed, not yet picked |
| OMS available to sell | 1,102 | = | OMS | Free to promise |

`1,186 − 84 = 1,102`. Two bridges, six components, no unexplained residue. That is what a
reconciliation looks like when it is finished. A reconciliation with a residual described as
"other" is not finished.

### 6.5 What the two bridges tell you

The first bridge is a **process latency** measure. Twenty-two cases staged, nine unposted, and
eighteen on hold are all reversible states. If the staged figure trends upward over weeks, the
warehouse is releasing waves faster than it can confirm them, which predicts ghost stock. If
the unposted figure trends upward, receiving is falling behind, which predicts phantom
shortages at month end.

The second bridge is a **commercial exposure** measure. Eighty-four cases allocated against
1,186 available is a 7.1% commitment rate. If that rate rises while the physical position
falls, the OMS will keep promising stock the warehouse cannot pick, and the short-pick rate
will rise with a lag of roughly the promise lead time. Neither bridge requires a model; both
require an owner per component and a timestamp.

### 6.6 What an analyst should have asked

1. Is the OMS promise field the original request or the overwritten promise? If overwritten,
   the on-time metric is measuring the company against itself.
2. Does the ERP invoice quantity come from the delivery or from the order? If from the order,
   every short-pick produces a credit note and a temporary revenue overstatement.
3. Is the TMS tender quantity the same object as the WMS pick quantity, or a planning figure
   that has not been refreshed?
4. Does the event layer cover this lane, or is it populated only for the regulated categories?
   Sparse coverage looks like missing events rather than missing instrumentation.
5. Which of these six records would an auditor accept as evidence, and which would they treat
   as an internal representation?

Each question is an authority question. None requires a model.

## 7. Code, query, or calculation

This section is deliberately a field map plus one short calculation. The point is which fields
exist and who owns them, not the syntax.

### 7.1 Field ownership map

| Field | Owning system | Type | Grain | Written when | Commonly copied to |
|---|---|---|---|---|---|
| `material_id` | ERP (material master) | Master | Plant–material | Item creation or extension | Everywhere |
| `gtin` | PIM or PLM (via GDSN) | Master | Trade item | Item publication | POS, EDI, event layer |
| `location_id` | ERP or WMS | Master | Site | Site creation | Everywhere |
| `on_hand_qty` | ERP | Transaction (posted) | Plant–material–storage location | Posting | Warehouse |
| `available_qty` | WMS | Derived | Bin–LPN | Task or count event | OMS, warehouse |
| `allocated_qty` | OMS | Transaction | Order line | Allocation run | Warehouse |
| `promise_date` | OMS | Transaction | Order line | Order capture or re-promise | ERP, customer portal |
| `load_id` | TMS | Transaction | Load | Tender | ERP, event layer |
| `shipment_status` | TMS | Derived | Shipment leg | Carrier event | OMS, control tower |
| `goods_issue_date` | ERP | Transaction | Delivery | Goods issue posting | Finance only |
| `invoice_qty` | ERP | Transaction | Invoice line | Billing | Finance, AR |
| `bom_component_qty` | PLM (eBOM), ERP (mBOM) | Master | Item–revision | Change order | MES |
| `operation_confirm_qty` | MES | Transaction | Work order–operation | Confirmation | ERP |
| `scan_event` | Event layer | Event | Object–timestamp | Scan | Warehouse, traceability |
| `opportunity_stage` | CRM | Transaction | Opportunity | Sales update | Nowhere authoritative |

The column "commonly copied to" is where duplication enters. Every copy is a future
disagreement.

### 7.2 The bridge calculation

One block, because the arithmetic is the point.

```text
given:
  erp_on_hand        = 1240   cases, posted, owned
  staged_not_issued  =   22   cases, picked, not confirmed out
  quality_hold       =   18   cases, blocked
  received_not_posted=    9   cases, on dock
  count_variance     =    5   cases, physical below book

wms_available = erp_on_hand - staged_not_issued - quality_hold
                              - received_not_posted - count_variance
              = 1240 - 22 - 18 - 9 - 5
              = 1186 cases

oms_available_to_sell = wms_available - allocated_open_orders
                      = 1186 - 84
                      = 1102 cases

residual = erp_on_hand - (wms_available + staged + hold + unposted + variance)
         = 0    <- must be zero; a non-zero residual is an unexplained defect
```

Two sanity checks before trusting the output:

- Every component must carry a document number. A component without a document is an
  assertion, and assertions do not reconcile.
- The residual must be exactly zero. If it is not, do not plug it into "other." Find the
  missing component or declare the reconciliation failed.

## 8. Data, standards, and regulatory context

Systems of record are constrained by standards at three levels: how objects are identified, how
records are exchanged, and what must be recorded by law.

### 8.1 Identification

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identity | The unit ordered, shipped, and sold has one global number | B |
| GS1 GLN | Party and location identity | Sites and parties joinable across systems | B |
| GS1 SSCC | Logistics unit identity | The pallet or case has a serialised label | B |
| GS1 GDSN | Master-data synchronisation | Item attributes published once and consumed by trading partners | B |
| DUNS | Legal entity identity | Supplier and customer entity resolution | C |

Identification is the join layer, and it is where authority claims are tested. Two systems can
hold the same fact and fail to reconcile because they identify the object differently. A GTIN
change from a pack-size modification is the most common single cause of a broken sales history,
and Topic 09 treats it in detail.

### 8.2 Exchange and event standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ANSI X12 / EDIFACT | Business document exchange | Order, ASN, invoice, inventory message formats | B |
| EPCIS 2.0 (GS1) | Event data | What happened to which object, where, when, and why | B |
| EPCIS CBV 2.0 | Event vocabulary | The controlled list of business steps and dispositions | B |
| ISA-95 / IEC 62264 | Enterprise-to-control integration | The boundary between business systems and manufacturing operations | B |
| Peppol BIS / EN 16931 | European e-invoicing | The semantic model for electronic invoices | B |
| OpenAPI / REST | Partner integration | Interface shape, not data semantics | C |

EPCIS 2.0 is the most consequential of these for this topic, because it is the standard that
makes an event layer shareable. It records five dimensions — what, when, where, why, and, new
in version 2.0, how (sensor data such as temperature and humidity) — across six event types:
object, aggregation, transaction, transformation, association, and the base event. The
distinction between an object event and an aggregation event is the difference between "this
pallet moved" and "these cases are inside this pallet," and confusing them is a common cause of
double counting in traceability models.

### 8.3 What must be recorded

| Requirement | Jurisdiction | Status as of 2026 | Grade |
|---|---|---|---|
| DSCSA serialisation and interoperable exchange | United States, pharmaceuticals | Interoperable exchange requirement active | B |
| FSMA 204 Food Traceability Rule | United States, food | Compliance date January 2026; key data elements on the traceability list | B |
| EU Falsified Medicines Directive | European Union, pharmaceuticals | In force; verification at dispensing | B |
| EU Ecodesign for Sustainable Products Regulation | European Union | Adopted 2024; Digital Product Passport phasing by product group | B |
| EU CBAM definitive regime | European Union | Begins 1 January 2026 for covered goods | B |

These requirements do something specific to the record layer: they force the creation of an
event system where most companies had only orders and invoices. That is a material change.
An order says what someone intended. An invoice says what someone charged. Neither says where
a lot was on a date, and the regulations increasingly require the third thing.

> **Note:** A compliance-driven event layer is usually built for the regulator's questions and
> not the planner's. It will have the required critical tracking events and may have no
> meaningful coverage of the lanes a forecasting model needs. Check coverage before assuming
> that "we have EPCIS" means "we have event data."

### 8.4 Financial and audit interfaces

| Requirement | Where | Effect on authority | Grade |
|---|---|---|---|
| IFRS 15 / ASC 606 | Global | Revenue exists when control transfers, not when the order is taken | B |
| Sarbanes-Oxley Section 404 | United States, listed companies | Internal control over financial reporting, including system access | B |
| Incoterms 2020 | Global contract convention | Defines when risk and cost transfer, hence who owns goods in transit | B |
| Country e-invoicing mandates | Italy, Spain, France, others | Invoice becomes a near-real-time reported object | B |

The audit consequence is direct: the ERP's value is the value that will be examined, and a
reconciliation between operational and financial records is an internal control, not a
convenience. An analyst who changes a definition used in a reconciliation is changing a
controlled process.

### 8.5 What is convention rather than requirement

- The ISA-95 level numbering is a reference model, not a regulation. It is useful for
  describing boundaries and carries no legal force.
- "The WMS owns quantity, the ERP owns value" is an industry convention that most vendors
  document and most implementations follow. It is not mandated, and some companies invert it.
- The practice of nightly extracts is a scheduling habit from a batch era, not a constraint.
- Calling something a "system of record" in a vendor deck does not make it one. Authority is
  assigned by the organisation and tested by audit.

## 9. Industry practice and case evidence

### 9.1 The ERP landscape in 2026

The ERP market is large, consolidated at the top, and mid-migration. Vendor-level revenue
rankings published in 2026 place Oracle first, SAP second, and Microsoft fifth by ERP
application revenue, with total ERP application revenue around USD 145.6 billion in 2025 and
growth near 11%. Treat the ranking as directional: it counts application revenue only, excludes
adjacent suite revenue, and is compiled by a firm that sells the ranking. (Grade D.)

The migration timeline is firmer than the market share. SAP's published maintenance schedule
ends mainstream maintenance for ECC 6.0 on enhancement packages 0 to 5 at the end of 2025 and
for enhancement packages 6 to 8 at the end of 2027, with optional extended maintenance to 2030
and a private-cloud edition announced in 2025 for large estates. (Grade B for the dates as
published commitments; the dates have moved before and may move again.) An industry user-group
survey in late 2025 reported 56% of respondents live or migrating, with an average transition
of about 1.5 years. (Grade D — self-selected membership sample.) The analytic consequence is
not about which vendor wins: a large share of the world's ERP data will cross a cutover in the
next three years, and every cutover is a discontinuity in the series models are trained on.

### 9.2 The WMS and TMS market structure

Warehouse management is a replacement market. Industry research published in 2025 sizes the
WMS-related market at roughly USD 12 billion including services and hardware, with licences and
cloud subscriptions a minority of the total, and reports that most purchases replace a second or
third system or displace an ERP warehouse module. Vendor concentration is high at the top tier,
and roughly 70% of vendors bundle adjacent products, most often order management and transport
management. (Grade D — vendor-sponsored market research.) Two structural facts matter for
analysts. **WMS boundaries are blurring** as warehouse execution systems absorb WMS function, so
"which system holds the bin quantity" is not always answerable from the product category. And
**TMS and WMS are converging commercially** into unified platforms, which simplifies integration
while concentrating authority in one vendor. Neither fact changes the reconciliation logic; both
change who you ask.

### 9.3 OMS in omnichannel retail

Order management has moved from capture to orchestration. Market research published in 2025
projects retail OMS software spend rising from about USD 2.2 billion in 2024 to USD 4.3 billion
in 2029, with cloud taking nearly all the growth and on-premise declining. (Grade D — the study
was sponsored by a vendor with a product in the category.) The same research finds agentic AI
capability concentrated in a handful of vendors and largely at roadmap stage elsewhere, which
is a useful corrective to vendor messaging.

The authority shift is the real story: as the OMS takes on routing, it now decides which node
fulfils an order, and that decision determines which WMS holds the physical stock and which TMS
holds the load. The OMS has become an authority *allocator* even though it owns no physical
fact.

### 9.4 POS data access

Consumer sales data reaches a manufacturer through three channels with different latency and
different contractual terms: direct retailer APIs, syndicated census or panel providers, and
retailer portals. Provider material claims census coverage of a large majority of consumer
packaged goods sales and thousands of programme partnerships. (Grade D — vendor marketing; the
coverage claim is not independently verifiable from the source.) Practitioner guidance
distinguishes syndicated latency of one to two weeks from direct API latency of one to three
days, and notes that syndicator contracts commonly restrict re-identification, resale, and
combination with retailer loyalty data absent an explicit clause. (Grade E — training material,
not primary research.) POS access is therefore a licensing question before it is an engineering
question, and the licence determines the latency, which determines what the data can be used
for. A two-week-latency feed cannot drive daily replenishment, whatever the dashboard suggests.

### 9.5 PLM and BOM ownership

The BOM question has a documented industry position: a single monolithic BOM fails, and the
workable design separates engineering, manufacturing, and sales views with each function owning
its own view and governed rules controlling synchronisation. Consultancy research published in
2025 identifies weak PLM–ERP integration and unclear governance as root causes of BOM
data-quality decay. (Grade D — whitepaper with a commercial interest in integration projects.)

### 9.6 Documented system-replacement failures

Two cases are well enough documented to teach with.

**Revlon, 2018.** Revlon's SAP rollout disrupted service levels at a North Carolina plant; the
company reported that it was unable to ship product representing roughly USD 64 million in net
sales, and its share price fell. An investor class action followed, which the company described
as without merit. Reported root causes include compressed timelines following an acquisition,
inadequate end-to-end testing, and change-management gaps. (Grade E for the sequence of events
as reported in the trade press; Grade C for the company's own financial disclosure of the
sales impact; the causal attribution is inference, not an audited finding.)

**Lidl, 2018.** Lidl cancelled a multi-year SAP programme after reported spend of roughly
EUR 500 million, with the stated reason that the original strategic goals could not be met
without further cost. The programme had been intended to cover thousands of stores and more
than a hundred logistics hubs. (Grade E — press reporting of a cancellation and a spend figure
that has varied across retellings.)

The lesson in both is not "ERP projects fail." It is that **the failure is usually in the
process and data transition, not in the software**, and the transition is exactly where
authority is renegotiated. An academic review of ERP risk literature reports the widely repeated
60–90% benefit-shortfall range and identifies poor risk management as a leading cause; treat the
range as a literature convention rather than a measured statistic.

### 9.7 The system-of-record versus system-of-intelligence framing

The three-layer framing — record, engagement, intelligence — is now common in analyst and
vendor material, with a fourth "action" layer added in 2026 framings as agents became plausible.
The most detailed vendor treatment positions the system of intelligence as sitting *between*
the record and the engagement layers, unifying data and orchestrating decisions rather than
holding authority. (Grade D.) Attribution of the exact two-term dichotomy to any single analyst
firm is thin; the idea is older than its packaging, and the layered version is more defensible
than the binary. The framing usefully separates "what happened" from "what should we do," and
dangerously invites the inference that the intelligence layer is more trustworthy. It is not:
it is downstream, and downstream is less authoritative by construction.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating the warehouse as the record | Two teams quote different OTIF | Copies used as authority | Decisions on incomparable numbers | Publish an ownership map per fact |
| Wrong grain join | Row counts inflate after a join | Bin-grain joined to plant-grain | Overstated inventory or demand | Aggregate to the coarser grain before joining |
| Undocumented latency | Recommendations consistently late | Analytics assumes fresher data | Interventions after the decision | Publish latency per feed in seconds |
| Promise overwrites request | Service looks better than customers report | One date field retained | Systematic overstatement of on-time | Retain both dates, measure both |
| Allocation treated as stock | Overselling in one channel | OMS allocation is a promise, not a pick | Short-picks, cancellations, penalties | Report ATP and ATS separately |
| Ghost stock | Availability high, short-picks rising | Picks confirmed but not completed | Confident wrong answers | Cycle counts by bin; task ageing |
| Reconciliation with an "other" bucket | Residual never reaches zero | Unexplained components netted | Defects hidden indefinitely | Force residual to zero or declare failure |
| Invoice on ordered quantity | Credit notes lag shipments by weeks | Billing rule trusts order | Temporary revenue overstatement | Invoice on delivery quantity |
| Planning tool assumed to be a record | Nobody can reconstruct last month's plan | Plans overwritten in place | No evaluation, no override analysis | Version plans with timestamp and author |
| Backflush error accumulates | Unexplained component shortages | Automatic consumption at standard quantity | Inventory drift, line stoppages | Periodic backflush audit against physical |

Three of these deserve emphasis.

**Wrong grain join** is the most common silent error. It produces no exception, no null, and no
warning, only a number that is roughly plausible and wrong by a factor that depends on how many
children the parent has. Always check row counts before and after a join, and always know which
side is the many side.

**Reconciliation with an "other" bucket** is a governance failure disguised as a data failure.
The moment a reconciliation is allowed to absorb a residual, it stops being a control and
becomes a ritual. The rule is simple and unpopular: zero or failed.

**Planning tool assumed to be a record** destroys institutional memory. If last month's plan
cannot be reconstructed, every claim about planning improvement is unfalsifiable.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Each fact has one identifiable owner | The whole model rests on scoped authority | Dual ownership after a merger or migration | Ask two functions; compare answers |
| The owner's grain is stable over the analysis window | Joins depend on keys that do not change | Reorganisation, bin renumbering, plant closure | Profile key cardinality over time |
| Posting latency is bounded and known | Timeliness judgements depend on it | Batch failures, month-end freezes | Measure event-to-record lag directly |
| Status vocabularies are stable | Trend analysis depends on comparable states | Status code changes at upgrade | Maintain a code crosswalk by version |
| Copies are refreshed on a documented schedule | Freshness claims depend on it | Manual refreshes, failed jobs | Monitor last-updated per table |
| Master data is consistent across systems | Joins assume shared identifiers | Item duplication, pack-size change | Sample join hit rates monthly |
| Inventory units are comparable | Aggregation assumes one unit of measure (UoM) | Catch-weight, dual UoM, pack conversions | Check UoM conversion factors per item |
| The event layer covers the lanes in scope | Event-based features assume presence | Regulatory-only rollout | Compute event coverage by lane |
| Financial and operational periods align | Period joins assume matching windows | 4-4-5 calendars, fiscal year offsets | Compare period definitions |
| Users act on the record, not on a side spreadsheet | Behavioural claims assume the system | Shadow systems in planning | Ask where the number was read |

Two limits are worth stating. **The framing assumes the organisation can name its owners.** In a
company grown by acquisition, the same fact may be owned by different systems in different
regions, and there may be no single answer; the unit of analysis then becomes the legal entity
or the region. And **the framing is about facts, not meaning**: knowing that the WMS owns bin
position does not tell you whether bin position is the right input for a forecast.

> **Note:** Do not try to resolve dual ownership by declaring one system the winner. Dual
> ownership is usually a symptom of two processes that genuinely differ.

## 12. Exercises

### Exercise 1 — Ownership map (recall and application, 45 minutes)

Choose a company you know or can research. Build a table of ten facts and name the owning
system for each, with the grain and the latency.

**Deliverable:** a ten-row table with columns: fact, owner, native grain, latency, what the
owner excludes.
**Acceptance criteria:** at least ten facts; at least three are financial; every row states an
exclusion; no row uses "the data warehouse" as an owner.

### Exercise 2 — The four inventories (application, 40 minutes)

For a single SKU at a single site, construct four inventory numbers — ERP, WMS, OMS, TMS — with
a stated as-of time, and write the bridge between them.

**Deliverable:** one table of four numbers and two signed bridges with a named component each.
**Acceptance criteria:** residual is exactly zero; every component has a document or process
owner; the as-of time is the same for all four, or the difference is stated.

### Exercise 3 — Status trace (analysis, 50 minutes)

Take one order record from any system you can access, or construct one from a public case, and
trace its status through at least four systems.

**Deliverable:** a timeline table with system, timestamp, state written, and next expected
transition.
**Acceptance criteria:** at least four systems; at least one pair of states that disagree; a
one-paragraph explanation of why the disagreement is not an error.

### Exercise 4 — Reconciliation failure (diagnosis, 40 minutes)

A company reports a 3.8% gap between ERP and WMS inventory that has persisted for six months
and is carried as "other." Write a diagnosis and a plan.

**Deliverable:** five candidate causes ranked by plausibility, the measurement that would
distinguish each, and a stopping rule.
**Acceptance criteria:** each cause maps to a different measurement; the stopping rule is
specific; at least one cause is a process cause rather than a data cause.

### Exercise 5 — Record or intelligence (classification, 35 minutes)

Classify each of the following as a record, a system of intelligence, or a copy, and justify:
a WMS, a demand planning tool, a data warehouse, a control tower, an OMS, a forecast accuracy
report, an EPCIS repository.

**Deliverable:** a seven-row classification table with one sentence of justification each.
**Acceptance criteria:** the justification names authority, not technology; at least one item is
a copy; the control tower classification acknowledges its mixed nature.

### Exercise 6 — Grain audit (application, 45 minutes)

Given two tables — one at bin grain, one at plant grain — write the aggregation that makes them
joinable without duplication, and state what information is lost.

**Deliverable:** the aggregation logic in plain language plus the lost-attribute list.
**Acceptance criteria:** the row-count check is stated before and after; at least two lost
attributes named; the aggregation is reversible in principle or the irreversibility is stated.

### Exercise 7 — Latency budget (design, 50 minutes)

Design the latency budget for a replenishment decision that must run daily at 06:00, from
physical pick to model input.

**Deliverable:** a table of pipeline stages with latency in minutes and an end-to-end total.
**Acceptance criteria:** at least six stages; the total is compared to the decision deadline;
one stage is identified where latency is organisational rather than technical.

### Exercise 8 — Cutover plan (design, 60 minutes)

An ERP migration goes live in eight weeks. Write the data-continuity plan for the analytics
team.

**Deliverable:** a one-page plan covering series breaks, code crosswalks, backfill, and
validation.
**Acceptance criteria:** at least one historic series is explicitly marked as non-comparable
across the cutover; a parallel-run validation is specified; the plan names who signs off on the
new definitions.

### Exercise 9 — Vendor claim audit (critique, 45 minutes)

Take one vendor document that uses the phrase "system of record" or "single source of truth."
Audit the claim.

**Deliverable:** a short memo separating what the product demonstrably does from what the claim
implies, with evidence grades.
**Acceptance criteria:** every claim is graded; at least one claim is identified as a category
error; the memo states what evidence would settle the question.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
ERP on-hand, posted            1,240 cs
less staged for open waves       -22
less quality hold                -18
less received, not posted         -9
less cycle-count variance         -5
                              ------
WMS available                  1,186 cs

WMS available                  1,186 cs
less allocated to open orders    -84
                              ------
OMS available to sell          1,102 cs

residual = 1240 - (1186 + 22 + 18 + 9 + 5) = 0
```

A common wrong answer subtracts allocated quantity from the ERP figure and calls the result
"available," giving 1,156. That mixes a financial base with a commercial deduction and will
not reconcile to either system. The two bridges are sequential, not additive.

**Exercise 4 model answer.**

Ranked causes: (1) unposted receipts at a busy dock, because the gap is persistent and
inventory-heavy sites usually have a receiving backlog; (2) ghost stock from confirmed but
incomplete picks, because a 3.8% gap with rising short-picks is the signature; (3) timing of
the snapshot, because the two systems are read at different times; (4) a genuine cycle-count
drift, which should show as a step change after a count; (5) a unit-of-measure or pack
conversion error on a subset of items, which would show as a gap concentrated in specific SKUs
rather than spread evenly.

The distinguishing measurements: age of unposted receipts in hours; task ageing for open picks;
timestamp delta between the two snapshots; variance by SKU concentration; UoM conversion audit
on the top twenty items by gap value. The stopping rule: if the gap is not explained to zero
within two count cycles, escalate to a physical wall-to-wall count of the affected bins.

**Exercise 5 answers.**

A WMS is a record for physical position and a system of intelligence for work sequencing — both,
because it does two jobs. A demand planning tool is a system of intelligence; its output is a
proposal. A data warehouse is a copy. A control tower is mixed: it is a copy of records plus a
system of intelligence, and it is authoritative for nothing. An OMS is a record for the promise.
A forecast accuracy report is a derived artefact, not a record, and if it is not versioned it
cannot be evaluated. An EPCIS repository is a record for events, provided the events were
captured at the point of occurrence rather than reconstructed.

**Exercise 6 model answer.**

Aggregate the bin table to plant–material by summing quantity, after excluding hold and
quarantine if the plant figure is a sellable figure, and after converting units. Check that the
summed bin count equals the plant row count in the join, and that the number of distinct
material keys is unchanged. Lost attributes: bin, licence plate, lot, and any status
distinction. The aggregation is not reversible: any later question about pick travel or slotting
must go back to the source.

**Exercise 7 model answer.**

A plausible budget: pick scan (0 min, event time) → WMS task commit (2 min) → WMS extract
(15 min) → staging area load (30 min) → transformation (20 min) → semantic layer refresh
(10 min) → model run (45 min) → output publication (5 min). End-to-end approximately 127
minutes, comfortably inside a 06:00 deadline if the pipeline starts at 03:50. The design
question is not the total; it is which stage has an owner who can be asked to change it.

**Exercise 8 rubric.**

A strong plan names the cutover date, lists the fields whose definitions change, provides a
crosswalk for status codes and material types, marks at least one series as non-comparable
across the boundary, specifies a parallel run of at least two close cycles, and names a
sign-off owner for each redefined measure. A weak plan says "we will validate the data" without
naming a validation, or assumes that unchanged field names imply unchanged semantics.

**Exercise 9 rubric.**

A strong audit separates three claims: what the product stores, what it derives, and what it
asserts authority over. It grades each and identifies the category error where a derived object
is described as a record. It states the evidence that would settle the question, typically an
audit trail showing an unbroken chain from event to stored value with a timestamp.

## 14. Interview questions

**1. "What is a system of record?"**

A strong answer defines it as scoped authority rather than a location, gives an example of two
systems owning different aspects of the same fact, and notes that a warehouse is a copy. A weak
answer names a product category or says "the main database."

**2. "The ERP says 1,240 and the WMS says 1,186. Which is right?"**

A strong answer refuses the framing, asks what decision the number serves, and sketches a
bridge with timing, hold, and allocation components. A weak answer picks one and defends it.

**3. "Why would a company deliberately keep four inventory numbers?"**

A strong answer explains that each answers a different question — ownership and value,
physical pickability, promise capacity, in-transit — and that unifying them would destroy
information. A weak answer calls it a data-quality problem.

**4. "What is the difference between a plan and a record?"**

A strong answer says a record states what happened and carries authority; a plan states what
someone intends and is a proposal; and notes that overwriting plans destroys evaluation. A weak
answer says a plan is a future record.

**5. "How would you decide which system to trust for a service metric?"**

A strong answer starts from the definition of the metric — requested versus promised date,
order versus line versus shipment — and then locates the owner of each input. A weak answer
picks the system with the nicest dashboard.

**6. "What can a TMS not tell you?"**

A strong answer says it cannot reliably tell you where goods are unless carriers report events,
and that planning loads and execution loads are different objects. A weak answer says it tracks
shipments.

**7. "Why is POS data hard to get, and why does it matter?"**

A strong answer separates the commercial arrangement from the technical pipeline, notes that
licences restrict use and combination, and explains that POS is the only true demand signal
while shipments are a distorted proxy. A weak answer says retailers do not share data.

**8. "Who owns the bill of materials?"**

A strong answer distinguishes eBOM, mBOM, and sBOM, assigns each to a function, and describes
the governed synchronisation between them. A weak answer names one system and stops.

**9. "Is a planning tool a system of record?"**

A strong answer says no, explains that its output is a proposal, and notes that the record of a
decision is a separate artefact that must be versioned. A weak answer says yes because it is
where planners work.

**10. "How would you detect that a dashboard is built on the wrong grain?"**

A strong answer describes row-count checks before and after joins, cardinality profiling, and a
comparison of totals against the owner's reported figure. A weak answer says check for nulls.

**11. "What does 'single source of truth' mean in practice?"**

A strong answer says it is a slogan that obscures scoped authority, and replaces it with one
owner per fact, one bridge per pair, one as-of timestamp per copy. A weak answer treats it as a
design target to be achieved.

## 15. Key terms

- **System of record** — the scoped authority for a fact.
- **System of intelligence** — a layer that derives decisions from records; never authoritative
  for the facts it reads.
- **System of engagement** — the interaction surface that writes into a record.
- **Native grain** — the key combination at which a system writes.
- **Latency** — event-to-record elapsed time; **freshness** — age of the row you hold.
- **Posting** — writing the financial consequence into the ledger.
- **ERP** — enterprise resource planning; owns value, cost, and finance.
- **WMS** — warehouse management system; owns bin position and task state.
- **WES / WCS** — execution and control layers inside the warehouse.
- **TMS** — transportation management system; owns loads, tenders, and freight settlement.
- **OMS** — order management system; owns capture, allocation, and promise.
- **DOM** — distributed order management; the routing decision inside or beside the OMS.
- **POS** — point of sale; the consumer transaction record.
- **PIM** — product information management; owns sellable attributes.
- **PLM** — product lifecycle management; owns design definition and change.
- **eBOM / mBOM / sBOM** — engineering, manufacturing, and sales bills of materials.
- **ECO** — engineering change order, with an effectivity date.
- **CRM** — customer relationship management; owns account and pipeline.
- **MES** — manufacturing execution system; owns execution events.
- **ISA-95** — the reference model separating business from manufacturing operations.
- **EPCIS** — GS1 standard for object event data.
- **Object, aggregation, transformation event** — three EPCIS event types with different
  meanings.
- **ASN** — advance ship notice, announcing an inbound shipment.
- **ATP / ATS** — available to promise versus available to sell.
- **Allocation / reservation** — a commercial commitment versus a hold on stock.
- **Backflush** — automatic component deduction at confirmation.
- **Cycle count** — partial physical count to maintain book accuracy.
- **Ghost stock** — quantity in the system that is not on the shelf.
- **Reconciliation bridge** — a signed decomposition of the gap between two records.

## 16. Sources

**Standards and official**

- GS1, *EPCIS 2.0.0* normative standard and *EPCIS and CBV Implementation Guideline v2.0*,
  ref.gs1.org. Grade B.
- GS1 US, *EPCIS Recommendations for FSMA 204 Critical Tracking Events*, Release 2.0, May 2025.
  Grade B.
- ANSI/ISA-95.00.01-2025 (IEC 62264-1 modified), *Enterprise-Control System Integration —
  Models and Terminology*; ISA-95 Part 2 and Part 5 for the Level 3/4 interface. Grade B.
- GS1 General Specifications, including GTIN, GLN, SSCC, and GDSN. Grade B.
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606 (FASB); Incoterms 2020. Grade B.
- U.S. FDA, FSMA Section 204 Food Traceability Rule; DSCSA. Grade B.

**Vendor and product documentation**

- Microsoft Learn, "Balance inventory between Dynamics 365 and external WMS," reference
  architecture. Grade D — authoritative about the product's mechanism, not about outcomes.
- Oracle, Warehouse Management Cloud 25C documentation, "Inventory Discrepancy in
  Reconciliation Workbench" and "Introduce SCM Reconciliation Workbench." Grade D, same caveat.
- SAP, published maintenance schedule for SAP ERP 6.0 / ECC and S/4HANA; private-edition
  announcement, 2025. Grade B for the dates as published commitments.

**Peer-reviewed and academic**

- *Journal of Contemporary Management*, "Supply chain risk management capabilities during
  enterprise resource planning implementation: perspectives of ERP providers and their
  clients." DOI: 10.35683/jcman1061.240. Reports the widely cited 60–90% benefit-shortfall
  range; treat the range as literature convention. Grade A for the paper; the range is not a
  measured statistic.

**Analyst and market research**

- Gartner, *Magic Quadrant for Warehouse Management Systems*, 1 May 2025. Grade D.
- Gartner, *Magic Quadrant for Transportation Management Systems*, 2025. Grade D.
- Gartner, *Market Guide for Distributed Order Management Systems*, 30 June 2025. Grade D.
- Gartner, *Market Share: Enterprise Resource Planning, Worldwide, 2025*. Grade D.
- STIQ, *2025 WMS Software Report*. Grade D — vendor-sponsored market research.
- IHL Group, *The Order Management Market — 2025*, September 2025, sponsored by SAP. Grade D.
- Apps Run The World, "Top 10 ERP Software Vendors, Market Size and Forecast 2025–2030."
  Grade D.
- ASUG, *S/4HANA Adoption Trends*, October 2025. Grade D — self-selected membership sample.
- PwC, *Multi-BOM Excellence* whitepaper, 2025. Grade D.
- Circana, *Liquid Data Collaborate* product material. Grade D — coverage claims not
  independently verifiable.
- Teradata, "System of Intelligence: A Decision Intelligence Layer." Grade D.

**Press and trade reporting**

- TechTarget, "Revlon SAP ERP problems result in rare investor lawsuit," May 2019. Grade E.
- The Register, "ERP disaster zone: the most costly failures of the past decade," December
  2019. Grade E.
- TechTarget, "12 ERP implementation failures and how to avoid them," 2025 roundup. Grade E.
- itbrew, "How Oracle ousted SAP as a top ERP provider," May 2025. Grade E.

**Note on citations.** Market-size and adoption figures are reported as their publishers state
them and are not independently verified. Vendor documentation is cited for mechanism, never for
outcome. Where a figure has varied across retellings, the text says so.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_master-data.md`](02_master-data.md)
- Related: [`03_transaction-data.md`](03_transaction-data.md)
- Related: [`06_data-movement.md`](06_data-movement.md)
- Prerequisite: [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
