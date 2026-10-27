# Transaction Data

> Module 02 — The Supply Chain Data Landscape
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: 02.01, 02.02
> Version 2026.5

---

## 1. Why this topic exists

Almost every supply chain dataset is a pile of documents that describe a small number of events
at several levels of detail. A customer order for 100 units appears as one order header, one
order line, three allocation rows, three shipment headers, four shipment lines, one advance ship
notice (ASN), three receipts, one invoice header, and four invoice lines. Each of those records
is correct. Each answers a different question. The most consequential decision you make about
such a dataset is not which model to fit. It is **what one row means**.

That decision has a name: the **grain**. Grain is the business definition of a single record.
"One row per order line" and "one row per shipment line" are different grains, and a table that
mixes them is not a table with extra columns. It is a table whose sums are wrong.

Three failures recur often enough to justify a whole topic:

1. **The inflated shipment report.** An analyst joins an order-line table to a shipment-line
   table to add delivery status, then sums ordered quantity. An order that shipped in three parts
   now contributes its ordered quantity three times. The report is 2.4 times the truth, the
   arithmetic is correct, and nobody notices until a warehouse manager says the number exceeds
   the building's capacity.
2. **The two on-hand numbers.** The enterprise resource planning (ERP) system says 212 units.
   The warehouse management system (WMS) says 215. Both are right at their own cut-off, hours
   apart. A replenishment model reads the older snapshot and assumes three units that have already
   shipped.
3. **The model that saw the future.** A forecast model is trained on the order table as it looks
   today. Three percent of order lines were amended downward after the fact, and the amendments
   are stored by overwriting the original quantity. The model learns from quantities that the
   planner did not know at decision time, scores well offline, and underperforms in production.

None of these is a modelling failure. All three are grain, cut-off, or time failures. This topic
is placed after systems of record and master data because you cannot discuss grain until you know
which system owns which document, and you cannot join documents until the item and location
identifiers agree. It is placed before reference data because external signals only become useful
once the transaction spine they attach to is trustworthy.

The topic is deliberately theory-heavy. Code appears once, in section 7. The skill being built is
the habit of asking, before any query, what one row means.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the grain of any transaction table in one sentence, in business terms, without
   listing foreign keys.
2. **Distinguish** order header, order line, schedule line, shipment header, shipment line,
   delivery, ASN, receipt, invoice header, invoice line, and payment, and **say** which question
   each one can answer.
3. **Detect** a fan-out double count from a query plan or a row-count comparison, and **correct**
   it by aggregating to a common grain before joining.
4. **Reconcile** a snapshot on-hand quantity against an event-derived quantity, and **explain**
   any residual using cut-off, in-transit, reservation, unit-of-measure, or unposted-transaction
   effects.
5. **Design** an idempotent load for an at-least-once event feed, including the deduplication key
   and the handling of reversals and re-posts.
6. **Model** partial shipments and split lines without double counting, using a shipment-line
   bridge at the correct grain.
7. **Explain** valid time versus transaction time, and **demonstrate** how training on
   current-state data leaks information that was unavailable at decision time.
8. **Write** an as-of query that returns the state of an order as it was known on a given date,
   and **state** what the query cannot recover if history was overwritten.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures. The first is the **document chain**: the sequence of records a commercial
event passes through, from intent to cash. The second is the **grain ladder**: the levels at
which that same event can be counted. Most transaction-data errors come from confusing the two.

### 3.1 The document chain

```text
  quote ──► order ──► allocation ──► shipment ──► ASN ──► receipt ──► invoice ──► payment
    │         │            │             │          │        │           │          │
  header     header     1 row per      header    1 message  header      header    1 remit-
  +lines     +lines     order line     +lines    per ship  +lines      +lines    tance
             +sched     + qty          (item)    (HL tree) (item)      (item)    + n
             lines                                                                 applies
```

Read the chain left to right as a sequence of **claims**, each weaker or stronger than the last:

- A quote is a claim about what might be bought.
- An order is a claim about what will be bought.
- An allocation is a claim about which supply will serve it.
- A shipment is a claim that goods left.
- An ASN is a claim that goods are on their way, sent before arrival.
- A receipt is a claim that goods arrived and were accepted.
- An invoice is a claim about what is owed.
- A payment is the settlement of that claim.

The chain is not a pipeline that transforms one record into the next. It is a set of parallel
records about one event, each with its own key, its own timestamp, and its own grain. The order
line does not become a shipment line. Both exist, forever, and they disagree by design whenever
delivery is partial.

### 3.2 The grain ladder

```text
   one purchase order: SO-1042, 150 units, EUR 2,000
   │
   ├── order_header        1 row      total quantity, total value, status
   ├── order_line          2 rows     one item, quantity ordered, unit price
   ├── schedule_line       2 rows     one requested date per line
   ├── allocation          3 rows     one supply promise per line per date
   ├── shipment_header     3 rows     one physical dispatch
   ├── shipment_line       4 rows     one item inside one shipment
   ├── asn                 3 rows     one message per shipment, tree inside
   ├── receipt             3 rows     one acceptance event per shipment
   ├── invoice_header      3 rows     one billing document
   └── invoice_line        4 rows     one item inside one invoice
```

The same 150 units appear at every level. A query that counts rows at two levels at once counts
the units more than once. That is the whole problem, stated in one diagram.

### 3.3 The fan-out picture

```text
   order_line                        shipment_line
   ┌──────────────────┐              ┌──────────────────────┐
   │ L10   qty 100    │─── 1 : N ───►│ S1/L10   qty  40     │
   └──────────────────┘              │ S2/L10   qty  30     │
                                     │ S3/L10   qty  30     │
                                     └──────────────────────┘

   join on order_line_id, then sum order_line.qty:

        100  +  100  +  100  =  300        truth: 100
```

The join did not duplicate data. It repeated a parent value once per child, which is what a join
does. The error is asking a parent-grain measure a child-grain question.

### 3.4 Five properties worth memorising

**Property 1 — One event, many rows.** A single physical dispatch generates a shipment header, at
least one shipment line, an ASN hierarchy, a receipt, and usually an invoice line. Counting any
two of these together without reconciling grain inflates the count.

**Property 2 — Documents record intent; events record fact.** An order says what someone intended.
A shipment says what happened. A forecast trained on orders is trained on intentions, including
the ones that were later cancelled. A service metric computed from orders measures promises, not
deliveries.

**Property 3 — State is an integral; a snapshot is a sample.** Event-derived inventory is the
opening balance plus every movement. A snapshot is one observation at one instant. They agree only
if the snapshot's cut-off, the movement set, and the unit of measure all agree. In practice they
usually differ by a small amount, and the difference is diagnostic.

**Property 4 — Time has two axes.** Valid time is when something was true in the world.
Transaction time is when the system recorded it. An order line amended on 3 June with effect from
1 June has valid time 1 June and transaction time 3 June. A query that ignores the second axis
cannot reconstruct what anyone knew at the time.

**Property 5 — Joins multiply unless grain is controlled.** Relational algebra does not warn you.
A one-to-many join produces a rectangular result in which the parent's measures are repeated. Every
aggregate over that result is wrong unless the parent measure is divided by the child count, which
is never the right fix. Aggregate first, then join.

### 3.5 The three questions to ask of any table

1. **What is one row?** State it in business language, for example "one item on one shipment." If
   the sentence needs an "and sometimes", the table has more than one grain.
2. **What is the key?** Usually a document number plus a line number; a timestamp in the key
   usually means an event log rather than a document table.
3. **Is the measure additive here?** Quantity usually is; a header total is not; a percentage
   almost never is; a balance is semi-additive, additive across items but not across time.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Grain | The business meaning of one row in a table | Described as a list of foreign keys | A sentence |
| Atomic grain | The lowest level at which a business process captures data | Assumed to be the header | Level name |
| Event | A fact that something happened, with a time | Confused with a status change | Timestamped fact |
| Document | A commercial artefact with a number and lines | Treated as an event | Document number |
| Order header | One row per purchase or sales order | Treated as the order's contents | Order number |
| Order line | One row per item on one order | Confused with a delivery | Order number + line number |
| Schedule line | One row per requested date within an order line | Ignored, then dates are wrong | Order + line + date |
| Allocation | A promise of specific supply to specific demand | Confused with reservation | Allocation identifier |
| Reservation | Stock set aside but not yet shipped | Confused with allocation | Reservation identifier |
| Backorder | Demand that could not be met from available supply | Counted as a new order | Order line remainder |
| Shipment header | One physical dispatch under one document | Assumed to equal one order | Shipment identifier |
| Shipment line | One item inside one shipment | Assumed to equal one order line | Shipment + line |
| Delivery | The arrival event at a destination | Used as a synonym for shipment | Delivery number |
| Handling unit | A pallet, case, or tote with its own identity | Ignored in line-level models | SSCC (Serial Shipping Container Code) |
| ASN | Advance ship notice: contents of a shipment sent before arrival | Treated as a receipt | EDI (electronic data interchange) 856 or DESADV |
| Receipt | The acceptance of goods at a location | Treated as proof of payment | Receipt number |
| Invoice header | One billing document | Assumed to match one order | Invoice number |
| Invoice line | One item on one invoice | Assumed to match one shipment line | Invoice + line |
| Credit note | A negative invoice correcting a prior charge | Nettted silently into revenue | Credit note number |
| Payment | A transfer of cash | Treated as settlement of one invoice | Payment identifier |
| Cash application | The allocation of a payment to invoices | Ignored, leaving open items | Application record |
| Inventory snapshot | A recorded quantity at a stated instant | Treated as a flow | Units at a timestamp |
| Perpetual inventory | Inventory maintained by applying every movement | Assumed to be error-free | Units |
| On-hand | Quantity physically present and available | Confused with available-to-promise | Units |
| Cut-off | The instant at which a report's data is frozen | Left undocumented | Timestamp plus time zone |
| Valid time | When a fact was true in the business world | Confused with creation date | Timestamp interval |
| Transaction time | When a fact was recorded in the system | Called "valid time" by mistake | Timestamp interval |
| Bitemporal | Modelling both valid and transaction time | Used to mean "has history" | Two time axes |
| As-of join | A join that returns the version known at a given time | Implemented as a current-state join | Query pattern |
| Idempotency | Repeating an operation without changing the result | Confused with deduplication | Property |
| Natural key | The business identifier of a record | Replaced by a sequence number | Document + line |
| Additive measure | A measure that sums across all dimensions | Assumed for percentages | Numeric |
| Fan-out | Row multiplication caused by a one-to-many join | Blamed on duplicate data | Row count ratio |
| Restatement | Replacing a previously reported value | Done silently | Versioned record |
| Reversal | A compensating entry that negates a prior posting | Deleted instead of reversed | Negative quantity |
| Three-way match | Agreement of order, receipt, and invoice before payment | Treated as an accounting detail | Match result |
| Unit of measure (UOM) | The unit in which a quantity is expressed | Mixed without conversion | Each, case, pallet |
| Lot or batch | A production group with shared attributes | Dropped when aggregating | Lot identifier |

Four of these deserve expansion, because they carry most of the analytical weight.

**Grain versus key.** The grain is a sentence; the key is the enforcement of that sentence. A table
declared as "one row per order line" whose primary key is `(order_id, line_id)` is internally
consistent. A table declared the same way whose key is `(order_id, line_id, shipment_id)` is
actually at shipment-line grain, and every "order line" measure in it will fan out. When the
declaration and the key disagree, the key wins. This is why grain audits start with the primary
key and the row count, not with the documentation.

**Event versus document.** A document is a container: it has a number, a date, a party, and lines.
An event is a fact: it has an object, a location, a time, and a business step. A shipment document
can be corrected by a replacement document, which is a document-level operation. The underlying
physical events — picked, loaded, departed, arrived — cannot be corrected, only compensated. Systems
that model events as mutable document status lose the distinction and cannot answer "what did we
believe on the 14th."

**Additivity.** Quantity is additive across order lines at the same grain. Value is additive if
the currency and the price basis match. A header total is non-additive across lines because it
already contains them. Fill rate is a ratio and must be recomputed from its numerator and
denominator at the target grain, never averaged from line-level ratios. A balance is semi-additive:
you can add balances across items at one instant, but adding June's balance to July's produces
nonsense.

**Valid time versus transaction time.** Valid time answers "when was this true?" Transaction time
answers "when did we believe it?" A receipt posted on 30 June with a document date of 28 June has
valid time 28 June and transaction time 30 June. Both are needed. Accounting closes on one,
operations act on the other, and machine-learning features must be built on the second to avoid
leakage. Section 6.7 works a small example.

## 5. The mechanics, step by step

### 5.1 The chain, link by link

The table states what each link asserts and where it typically breaks. Read it as a checklist for
any new dataset: find which links you have, and which are missing.

| Link | Record and grain | What it asserts | Where it breaks |
|---|---|---|---|
| Quote | Header plus lines, one row per quoted item | A possible purchase at a price | Expiry ignored; converted to order without versioning |
| Order | Header plus lines plus schedule lines | A commitment to buy a quantity by a date | Amendments overwrite; units differ from supplier units |
| Allocation | One row per order line per supply source | Which supply will serve which demand | Reallocated silently; allocation not released on cancel |
| Shipment | Header plus lines, one row per item dispatched | Goods left a location on a date | Split shipments; one order in several shipments |
| ASN | One message per shipment with a hierarchy | Goods are in transit and here is what is in the box | Sent late; hierarchy code mismatched; quantities differ from shipment |
| Receipt | Header plus lines, one row per item accepted | Goods arrived and were accepted | Over-receipt; receipt before inspection; reversal and re-post |
| Invoice | Header plus lines, one row per billed item | Money is owed for these items | Billed before receipt; price mismatch; one invoice covering many orders |
| Payment | One remittance covering many invoices | Cash moved | Unapplied cash; partial payment; deduction taken |

Three structural facts follow from the table.

First, **the record count is not monotonic along the chain**. One order can produce three shipments
and one invoice; one invoice can cover five orders; one payment can cover twenty invoices.

Second, **each link has its own date**. Order, request, promise, ship, ASN, receipt, invoice, due,
and payment dates are nine different columns, and a correct measure on the wrong one is wrong.

Third, **each link can be corrected by a later record at that link's grain**: a line amendment, a
replacement ASN, a credit note, or a reversing receipt posting.

### 5.2 Phase A — Intent becomes a document

**Step 1. A quote is issued.**

- Record: quote header and quote lines.
- Grain: one row per quoted item, often with validity dates and quantity breaks.
- What breaks: quotes are frequently not retained at all. When they are, they are often converted
  to orders by copying, with no link back. The loss is analytical, not operational: you cannot
  measure quote-to-order conversion or price erosion without the link.

**Step 2. An order is created.**

- Record: order header, order lines, and often schedule lines splitting a line across requested
  dates.
- Grain: one row per item per order; schedule lines add a third dimension, the date.
- What breaks: the order quantity is a commercial quantity, not a need. It is rounded to a pallet,
  a truckload, a minimum order quantity, or a promotional buy. Treating it as demand imports the
  rounding into every downstream forecast.
- Question to ask: is there a separate requested-delivery schedule, or is one date stored on the
  header?

**Step 3. Supply is allocated.**

- Record: allocation rows linking order lines to supply, often to a specific lot, location, or
  inbound shipment.
- Grain: one row per order line per supply source, sometimes per date.
- What breaks: allocation is frequently a transient state held in memory by the order management
  system (OMS) and never persisted. The order then appears unallocated in the warehouse, and the
  analyst cannot explain why an order shipped late when stock was available.

### 5.3 Phase B — Intent becomes material

**Step 4. A shipment is created and dispatched.**

- Record: shipment header, shipment lines, and handling units.
- Grain: header is one physical dispatch; lines are one item within it.
- What breaks: the shipment is not the order. A line ordered as 100 units can ship as 40, 30, and
  30. The shipment line carries `quantity_shipped`, which is the only quantity that should ever be
  summed for a delivered-units measure.
- Question to ask: can one order line appear in more than one shipment? If the answer is yes, and
  it almost always is, the order-line-to-shipment relationship is many-to-many through a
  shipment-line bridge.

**Step 5. An ASN is transmitted.**

- Record: one EDI 856 message or EDIFACT DESADV per shipment.
- Grain: the message is one per shipment, but its internal structure is a tree. In ANSI X12 the
  tree is flattened into hierarchical level (HL) segments: shipment (S), order (O), tare or pallet
  (T), pack or carton (P), and item (I). A common structure is S-O-I; a palletised one is S-O-T-P-I.
- What breaks: the beginning segment for ship notice (BSN) declares which hierarchy is in use, and
  a mismatch between the declared structure and the segments that follow is a standard rejection
  cause. The transaction totals segment (CTT) counts HL segments, not items, which is a frequent
  source of reconciliation confusion.

**Step 6. Goods are received.**

- Record: receipt header and receipt lines, often with lot, serial, and condition.
- Grain: one row per item accepted on one receipt.
- What breaks: receipts are posted late, posted before inspection, over-received against tolerance,
  or reversed and re-posted after a quality hold. Each of these creates a temporary or permanent
  difference between the physical count and the system count.
- Question to ask: is the receipt timestamp the dock scan, the put-away, or the end-of-shift
  confirmation?

### 5.4 Phase C — Material becomes money

**Step 7. An invoice is issued.**

- Record: invoice header and invoice lines.
- Grain: one row per billed item on one billing document.
- What breaks: the invoice line is not guaranteed to reference the shipment line, even when the
  quantities match. Some ERPs link invoice to order, some to delivery, some to both. Without the
  link, cost-to-serve and deduction analysis cannot be done at line level.
- Question to ask: does the invoice line carry a shipment-line reference, or only an order-line
  reference?

**Step 8. Payment is applied.**

- Record: payment and cash-application rows.
- Grain: one payment can apply to many invoices; one application is one payment-invoice pair.
- What breaks: unapplied cash and deductions. A customer pays the invoice minus a disputed amount
  and the difference sits as an open item. The financial symptom is an aged receivable; the cause
  is usually a short shipment, a price mismatch, or a missing proof of delivery.
- Question to ask: can a payment be applied at invoice-line level, or only at invoice level?

### 5.5 State versus flow

Two ways to know inventory coexist in almost every company, and they are not substitutes.

**The snapshot.** At a defined instant, the system writes the quantity of every item at every
location. It is a state observation: cheap to query, the basis of most reporting, only as good as
the count behind it, and silent about how the quantity got there.

**Event-derived inventory.** Start from a known opening balance and apply every receipt, shipment,
transfer, return, and adjustment in sequence. It is a flow reconstruction: computable to any
instant, explanatory, and only as good as the completeness of the movement log.

The two are reconciled with a roll-forward:

```text
expected = opening + receipts + returns + transfers_in
         - shipments - transfers_out - consumption +/- adjustments

variance = snapshot(t) - expected(t)
```

A non-zero variance is not automatically an error. It has five common explanations, and the
analyst's job is to assign the residual to one of them rather than to "shrinkage":

1. **Cut-off mismatch.** The snapshot and the movement set stop at different instants.
2. **In-transit or in-process stock.** Goods shipped but not received, or received but not posted.
3. **Reservation and allocation.** The snapshot may report on-hand, available, or unreserved
   quantity, and the label is often wrong.
4. **Unit-of-measure mismatch.** Cases counted in one system and eaches in the other.
5. **Unposted or reversed transactions.** A receipt reversed in the WMS but not in the ERP.

> **Note:** A variance of a few units is normal and should be monitored as a distribution, not
> investigated line by line. A variance that repeats on the same item every period is a process
> defect and will not be fixed by another adjustment.

### 5.6 Corrections: how history is rewritten

Four correction mechanisms appear repeatedly, and each has a different effect on the data.

**Amendment.** The order line quantity or date changes. An overwrite destroys the original; a
versioned row lets you reconstruct both. The analytical question is always "as of when?"

**Cancellation.** A cancelled line with a non-zero shipped quantity is normal and must not be
excluded from delivery measures. A cancelled line with zero shipped quantity belongs out of demand
measures, but only if you are measuring demand and not orders.

**Reversal.** A receipt is reversed and re-posted. The event log shows three rows: +50, −50, +50.
The net is correct. Counting events gives three receipts; summing quantities gives 50. Both are
correct answers to different questions, which is why event counts and quantity sums must be
labelled explicitly in any metric definition.

**Restatement.** A previously reported figure is replaced. This happens in financial reporting by
design and in operational reporting by accident. Any metric that can be restated needs a
transaction-time column, or the organisation will spend meetings arguing about which version of
last month is real.

### 5.7 Two clocks

Every transaction record carries at least two timestamps, and often four:

- **Event time** — when the thing happened in the world.
- **Recording time** — when the system accepted the record.
- **Posting time** — when the record affected a balance or a ledger.
- **Extraction time** — when the record entered the analytical warehouse.

They can differ by minutes or weeks. A receipt keyed from a paper note has an event time on the
dock and a recording time three days later; a nightly warehouse extract adds another delay.

Bitemporal modelling keeps two explicitly: **valid time** (when the fact was true) and
**transaction time** (when the system held that version). It answers both "what was true on 1 June?"
and "what did we believe on 1 June about what was true on 1 June?"

The practical rule for analytics: **build features on transaction time, evaluate on valid time,
and never mix them without saying so.** A model trained on the current state of a restated table
has seen the future. A model trained on transaction time but evaluated on valid time will appear
worse than it is, because it is being scored against information it could not have had. Both
errors are common; the first is more dangerous because it inflates offline results.

### 5.8 Join discipline

The rule is short: **aggregate each fact to the target grain before joining.**

Three implementations, in order of preference:

1. **Drill across.** Compute each measure separately at the target grain, then combine the result
   sets on conformed dimensions. This is the dimensional-modelling term for what is often called a
   multipass query.
2. **Pre-aggregate the child.** Reduce the many-side to one row per parent key, then join. The
   child's measures become attributes of the parent, and the parent's measures are counted once.
3. **Use a bridge at the correct grain.** When the relationship is genuinely many-to-many, as
   between order lines and shipments, model the bridge explicitly with a quantity on it. A partial
   shipment then becomes a smaller number on a bridge row, not a duplicated parent.

What not to do: join two fact tables on a shared key and aggregate. The result is arithmetically
valid and semantically meaningless. Row-count checks before and after a join catch most cases; a
check that compares the summed measure against the source system's own total catches the rest.

## 6. Worked example

A distributor sells two products (stock-keeping units, SKUs) to one customer. The example is small
enough to check by hand and contains every defect discussed above: a split order, a partial
shipment, an invoice per shipment, a duplicate receipt, a cut-off mismatch, and a retroactive
amendment. All figures are illustrative and internally consistent.

### 6.1 The order

Customer C-07 places order SO-1042 on 2 June for delivery on 12 June.

**order_header**

| order_id | customer_id | order_date | requested_date | currency | status |
|---|---|---|---|---|---|
| SO-1042 | C-07 | 2026-06-02 | 2026-06-12 | EUR | open |

**order_line**

| order_id | line_id | sku | qty_ordered | unit_price | line_value |
|---|---|---|---:|---:|---:|
| SO-1042 | 10 | SKU-A | 100 | 10.00 | 1,000.00 |
| SO-1042 | 20 | SKU-B | 50 | 20.00 | 1,000.00 |

Header totals: 150 units, EUR 2,000.00. Note that neither 150 nor 2,000 is stored on the header in
this example; both are derived. That is deliberate, because a stored header total is the most
common source of double counting in reporting layers.

### 6.2 The shipments

The order ships in three parts. SKU-B is available immediately; SKU-A is short and ships in two
lots.

**shipment_header**

| shipment_id | order_id | ship_date | carrier | freight_cost | status |
|---|---|---|---|---|---:|---|
| S1 | SO-1042 | 2026-06-08 | DHL | 120.00 | delivered |
| S2 | SO-1042 | 2026-06-15 | DHL | 120.00 | delivered |
| S3 | SO-1042 | 2026-06-22 | DHL | 120.00 | delivered |

**shipment_line**

| shipment_id | ship_line_id | order_line_id | sku | qty_shipped | unit_price | line_value |
|---|---|---|---:|---:|---:|---:|
| S1 | 1 | 10 | SKU-A | 40 | 10.00 | 400.00 |
| S1 | 2 | 20 | SKU-B | 50 | 20.00 | 1,000.00 |
| S2 | 1 | 10 | SKU-A | 30 | 10.00 | 300.00 |
| S3 | 1 | 10 | SKU-A | 30 | 10.00 | 300.00 |

Four shipment lines, 150 units, EUR 2,000.00 of goods, EUR 360.00 of freight.

### 6.3 The invoices

One invoice per shipment, which is common when freight is billed per dispatch.

**invoice_header**

| invoice_id | customer_id | invoice_date | shipment_id | net_value | tax | total |
|---|---|---|---|---:|---:|---:|
| INV-1 | C-07 | 2026-06-08 | S1 | 1,400.00 | 280.00 | 1,680.00 |
| INV-2 | C-07 | 2026-06-15 | S2 | 300.00 | 60.00 | 360.00 |
| INV-3 | C-07 | 2026-06-22 | S3 | 300.00 | 60.00 | 360.00 |

**invoice_line**

| invoice_id | inv_line_id | shipment_line_ref | order_line_id | qty_invoiced | unit_price | line_value |
|---|---|---|---:|---:|---:|---:|
| INV-1 | 1 | S1/1 | 10 | 40 | 10.00 | 400.00 |
| INV-1 | 2 | S1/2 | 20 | 50 | 20.00 | 1,000.00 |
| INV-2 | 1 | S2/1 | 10 | 30 | 10.00 | 300.00 |
| INV-3 | 1 | S3/1 | 10 | 30 | 10.00 | 300.00 |

### 6.4 The double count

An analyst wants "ordered units and shipped units by order" and writes a join from `order_line` to
`shipment_line` on `order_line_id`.

The join result has four rows, because line 10 matches three shipment lines.

| order_id | line_id | qty_ordered | shipment_id | qty_shipped |
|---|---|---:|---|---:|
| SO-1042 | 10 | 100 | S1 | 40 |
| SO-1042 | 10 | 100 | S2 | 30 |
| SO-1042 | 10 | 100 | S3 | 30 |
| SO-1042 | 20 | 50 | S1 | 50 |

Now sum `qty_ordered`:

```text
line 10:  100 + 100 + 100 = 300
line 20:   50              =  50
total                      = 350
```

Truth: 150 units ordered. Inflation factor 350 / 150 = 2.33.

Sum `line_value` if it is carried on the order line: line 10 contributes 1,000 three times, so
3,000 + 1,000 = 4,000 against a true 2,000. Inflation factor 2.00. The two factors differ because
the inflation is not uniform across lines: line 10 is multiplied by three, line 20 by one.

The same join also corrupts freight. Summing `freight_cost` from the shipment header across the
joined rows gives 4 × 120 = 480, against a true 360, because each shipment header is repeated once
per shipment line.

**The correction.** Aggregate the child to the parent grain before joining.

```text
shipped_by_line:
  line 10 = 40 + 30 + 30 = 100
  line 20 = 50

joined at order-line grain:
  line 10: ordered 100, shipped 100
  line 20: ordered  50, shipped  50
  total  : ordered 150, shipped 150
```

Now every row is one order line, every measure is counted once, and the reconciliation is exact.
Freight stays at shipment grain.

### 6.5 The inventory reconciliation

Take SKU-A at the main warehouse for June; receipts are goods receipt notes (GRNs).

| Movement | Document | Date | Quantity |
|---|---|---|---:|
| Opening balance | — | 2026-06-01 | 200 |
| Receipt | GRN-1 | 2026-06-05 | +70 |
| Receipt | GRN-2 | 2026-06-18 | +50 |
| Shipment | S1 | 2026-06-08 | −40 |
| Cycle-count adjustment | CC-3 | 2026-06-15 | −5 |
| Shipment | S2 | 2026-06-15 | −30 |
| Shipment | S3 | 2026-06-22 | −30 |
| Shipment | S4 | 2026-06-30 | −3 |

Roll-forward to 30 June 23:59:

```text
200 + 70 + 50 - 40 - 5 - 30 - 30 - 3 = 212
```

The ERP month-end report says 212. The WMS snapshot taken at 18:00 on 30 June says 215. The
difference is three units, fully explained by cut-off: S4 was picked and confirmed at 20:00, after
the snapshot. The systems do not disagree about the world; they disagree about the instant.

This is the most common inventory discrepancy in practice, and the one most often escalated as a
data-quality defect. Publish the cut-off with every snapshot; do not adjust either number.

> **Note:** If the difference had been −7 rather than +3, cut-off would not explain it. A negative
> variance — the system holding less than the roll-forward predicts — usually means an unrecorded
> issue, a shipment posted twice, or a receipt that was reversed and never re-posted. Sign matters
> for diagnosis.

### 6.6 The duplicate receipt

GRN-2 for 50 units was posted on 18 June, reversed on 19 June because the pallet was quarantined,
and re-posted on 21 June after inspection.

The receipt event log contains three rows:

| event_id | receipt_id | event_type | qty | posted_at |
|---|---|---|---:|---|
| E-441 | GRN-2 | post | +50 | 2026-06-18 09:12 |
| E-442 | GRN-2 | reverse | −50 | 2026-06-19 14:40 |
| E-443 | GRN-2 | post | +50 | 2026-06-21 08:05 |

Summing quantity gives 50, which is correct. Counting rows gives three receipts, which is also
correct as an event count and wrong as a receipt count. A dashboard labelled "receipts this month"
without a definition will show one of these and be believed for the other.

If the feed were delivered at least once and the re-post were sent twice, the third row would appear
twice with the same `event_id`, the net would become 100, and no report that fails to reconcile
against the source would show it. The fix is a deterministic key and an idempotent write (Exercise 4).

### 6.7 The retroactive amendment

On 3 June, the customer increases line 10 from 80 to 100 units, effective 1 June. The order system
overwrites the row. A versioned alternative keeps both versions.

| line_id | qty | valid_from | valid_to | recorded_at | superseded_at |
|---|---:|---|---|---|---|
| 10 | 80 | 2026-06-01 | 2026-06-03 | 2026-06-02 10:00 | 2026-06-03 16:22 |
| 10 | 100 | 2026-06-03 | — | 2026-06-03 16:22 | — |

Three questions, three different answers:

| Query | Answer | Why |
|---|---|---|
| Quantity as at 2 June, as known on 2 June | 80 | Transaction time on 2 June sees only version 1 |
| Quantity valid on 2 June, as known today | 80 | Valid time 2 June falls in version 1's interval |
| Quantity valid on 3 June, as known today | 100 | Version 2 starts on 3 June |

The overwrite model can answer only the third. The first two are unrecoverable, and they are
exactly the ones a model trained for a 2 June decision needs.

A forecast trained on the overwritten table learns that the customer ordered 100 units on 2 June.
The planner on 2 June saw 80. Every feature derived from order history — average order size,
week-over-week growth, fill rate against ordered quantity — is contaminated. The contamination is
small here and grows with amendment frequency; in fashion or promotional grocery it can be material.

### 6.8 What the analyst should have asked

Does one order line map to one shipment line? Is the invoice linked to the shipment line or only to
the order? What is the snapshot cut-off, in which time zone? Are reversals stored as negative rows?
Does the order table version amendments? Each question is a grain or time question, and none
requires a model.

## 7. Code, query, or calculation

One structured query language (SQL) query, because the arithmetic is the point: the wrong join, then
the corrected pre-aggregation.

```sql
-- WRONG: parent measure repeated once per child row
select o.order_id, sum(o.qty_ordered) as ordered_units
from order_line o
join shipment_line s on s.order_line_id = o.line_id
group by o.order_id;                       -- SO-1042 -> 350, truth 150

-- RIGHT: reduce the child to the parent grain first
with shipped as (
    select order_line_id, sum(qty_shipped) as qty_shipped
    from shipment_line
    group by order_line_id
)
select o.order_id,
       sum(o.qty_ordered)                  as ordered_units,
       sum(coalesce(s.qty_shipped, 0))     as shipped_units
from order_line o
left join shipped s on s.order_line_id = o.line_id
group by o.order_id;                       -- SO-1042 -> 150 and 150
```

Two checks before trusting the corrected query. Compare `sum(qty_ordered)` against the order
system's own total for the period; a mismatch means the extract is incomplete, not that the query
is wrong. Then compare `sum(qty_shipped)` against the shipment system's total for the same period;
a mismatch means some shipments are not linked to an order line, which is a data defect worth
reporting rather than filtering away.

## 8. Data, standards, and regulatory context

Transaction data is more standardised than master data, because documents cross organisational
boundaries and partners enforce formats. The standards constrain structure, not semantics.

### 8.1 Document and event standards

| Standard | Governs | What it constrains | Source status | Grade |
|---|---|---|---|---|
| American National Standards Institute (ANSI) X12 856 | ASN in North America | Hierarchical level structure, item identification, quantities | Verified in search (IBM, SAP, vendor guides) | B |
| EDIFACT DESADV | ASN internationally | The European equivalent message | Named for context | B |
| GS1 SSCC | Handling-unit identity | The pallet or carton label referenced in the ASN | Referenced in ASN guides | B |
| GS1 GTIN (Global Trade Item Number) | Item identity | What the item-level HL segment carries | Named for context | B |
| AICPA Audit Data Standards, Order-to-Cash | O2C subledger extract | Header and line tables at each stage, field definitions, join keys | Verified in search (July 2015 PDF) | B |
| SQL:2011 temporal features | Valid-time and system-versioned tables | How a database stores history natively | Referenced in Teradata whitepaper | B |
| UBL Order, DespatchAdvice, Invoice | Document syntax in web services | XML structure for the same documents | Named for context | B |

The AICPA order-to-cash standard is the most useful single reference for schema design because it
specifies the table set rather than one message: orders, shipments, invoices, receivables, cash,
application, and adjustments, each with header and line tables. That separation is the schema-level
expression of section 5's grain discipline.

### 8.2 What the ASN standard actually implies

The 856 is a tree flattened into a list. Three details matter analytically.

First, the **hierarchy structure code** in the BSN segment declares which levels are present. Common
values distinguish a simple shipment-order-item structure from a palletised one. If the declared
code and the actual HL segments disagree, the receiver rejects the message, and the shipment
arrives without an ASN.

Second, the **HL parent pointer** makes the tree reconstructible. Each HL segment carries its own
identifier, its parent's identifier, its level code, and a flag saying whether children follow. A
parser that ignores the parent pointer produces a flat list and loses which item was in which
carton.

Third, **CTT counts HL segments, not items**. A shipment with two pallets, ten cartons, and forty
items has more HL segments than items, and a control-total check that assumes otherwise fails.

The analytical consequence: an ASN gives you the packaging hierarchy, which the shipment table
usually does not. If you need to know how many units were in one carton — for returns, for
damage analysis, for shelf-ready packaging — the ASN is the only place the answer exists.

### 8.3 Regulatory context

| Requirement | Jurisdiction | What it does to transaction data | Grade |
|---|---|---|---|
| E-invoicing mandates | Italy, Spain, France, and others | Invoice becomes a near-real-time reported object, often at line level | B |
| Value added tax (VAT) in the Digital Age (ViDA) | European Union | Digital reporting across the single market on a phased timetable | B |
| International Financial Reporting Standard (IFRS) 15 / Accounting Standards Codification (ASC) 606 | Global | Ties revenue timing to control transfer, hence to shipment and delivery terms | B |
| Incoterms 2020 | Global contract convention | Defines where risk and cost transfer, hence which entity records the movement | B |
| Lot and serial traceability rules | Sector-specific | Requires lot or serial on receipt and shipment records | B |

None of these tells you how to model a shipment. They tell you which fields must exist and how long
they must be retained. The practical effect on analytics is that invoice data is increasingly
complete, timely, and line-level, while shipment data remains partner-dependent and uneven.

### 8.4 What is convention rather than requirement

- "One invoice per shipment" is a billing convention, not a rule.
- "Header totals are stored" is a schema habit inherited from paper documents.
- "Shipment line equals order line" holds only when nothing is ever partial.
- "Negative inventory is an error" is a policy choice; some systems permit it deliberately to avoid
  blocking transactions, at the cost of reconciliation noise.

An analyst who can separate a convention from a constraint can challenge the convention.

## 9. Industry practice and case evidence

### 9.1 The modelling canon

The dimensional-modelling literature is unusually clear here. Kimball's technique pages state that
declaring the grain is the pivotal step, that the declaration should be in business terms rather
than a list of foreign keys, and that each distinct grain gets its own physical table (Grade C:
practitioner canon). The same work separates transaction, periodic snapshot, and accumulating
snapshot grains, mapping onto a shipment event, an inventory snapshot, and an order-fulfilment
record with milestones.

The companion rule is that fact tables at different grains should not be joined directly, because
cardinality cannot be controlled; query each separately and combine on conformed dimensions
(Kimball Group, multipass SQL, Grade C). These are strong defaults, not controlled findings.

### 9.2 The order-to-cash reference schema

The AICPA Audit Data Standards for order-to-cash (July 2015) specify a table set close to canonical:
orders and order lines; shipments and shipment lines; invoices and invoice lines; open receivables;
cash received; cash application; and adjustments with lines (Grade B). The standard keeps every
stage at its own grain and gives the join keys between them, which is the schema-level expression
of section 5.

A 2025 paper on Oracle ERP and Oracle Cloud WMS reconciliation reports the same issue at system
level: two systems hold the same stock and reconcile periodically because they serve different
purposes (Grade A, small journal; generalisation beyond Oracle is an inference).

### 9.3 What practitioners report about ASN compliance

EDI integration vendors describe hierarchy-structure mismatches, inconsistent SSCC labelling, and
quantity disagreements between ASN and physical receipt as leading causes of ASN rejection and
chargebacks (Grade D: vendor content, accessed 2026). No independent rejection-rate figure was
found. Treat the direction as credible and the magnitude as unverified.

The implication survives the weak evidence: an ASN is a claim made before arrival, and claims fail.
A dataset that treats ASN quantity as received quantity will overstate transit availability.

### 9.4 The two-system inventory problem

Microsoft's guidance for balancing inventory between Dynamics 365 and an external WMS states the
split plainly: the WMS is the system of record for physical quantity and the ERP for valuation, and
the two are expected to drift and be reconciled rather than to match continuously (Grade C). Oracle's
PeopleSoft documentation warns that adjustments made in one system do not propagate automatically
(Grade C).

The practical reconciliation advice from these sources is consistent and worth adopting:

- Snapshot both systems at the same moment, during a low-activity window.
- Automate detection, not correction. Automatic balancing hides systemic defects.
- Reconcile units first, then value.
- Report on-hand, allocated, and in-transit as separate labelled states, never as one number.
- When an item repeats as unbalanced, audit the transactions rather than adjusting again.

### 9.5 Event sourcing and idempotency in practice

The event-sourcing pattern treats state as a fold over an append-only log, with corrections
represented as compensating events rather than overwrites (Microsoft Azure Architecture Center,
Grade C: vendor architecture guidance). The idempotency requirement follows directly: delivery is
typically at least once, so a duplicated event must not produce a duplicated effect. The standard
implementations are a deterministic event identifier, a uniqueness constraint or deduplication
store, and an upsert keyed on that identifier (Microsoft; Confluent idempotent reader pattern;
practitioner change-data-capture (CDC) guide, all Grade C or D).

The most useful formulation found in the practitioner material is that end-to-end exactly-once
delivery is not achievable; what is achievable is exactly-once *processing*, meaning at-least-once
transport plus idempotent sinks (Grade C: practitioner guide, undated). That is the right framing
when source, broker, and warehouse are three independent failure domains.

### 9.6 Bitemporal practice

Martin Fowler's article distinguishes actual history from record history and describes two storage
approaches: start and end date columns on each version, or a bitemporal event log (Grade C:
practitioner article, 2021). Teradata's paper describes native `TRANSACTIONTIME` and `VALIDTIME`
support and the `AS OF` query forms (Grade C). The VLDB Journal published a 2024 comparison of tuple
and attribute timestamping for temporal warehouses, reporting order-of-magnitude performance
differences on a TPC-DS-derived benchmark (Grade A; DOI 10.1007/s00778-024-00889-2).

Point-in-time correctness in feature pipelines is the same idea under another name. Microsoft's
feature-store documentation describes joins that fetch the latest value between a lookback limit and
a source-delay offset, so data that had not arrived at prediction time is excluded (Grade C,
vendor documentation).
Its worked example shows a reading taken at 08:52 leaking into an observation labelled 08:50.

### 9.7 What the evidence does not establish

No source found in this research quantifies the share of reporting defects caused by grain errors,
or the cost of a reconciliation variance. Claims of the form "X% of supply chain data is bad" are
unsourced unless a study is named. The mechanisms are documented; their prevalence is not.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Joining facts at different grains | Report total exceeds a physical bound | Parent measure repeated per child row | Silent overstatement | Aggregate to common grain first; row-count check |
| Summing header totals across lines | Two reports differ by a factor equal to line count | Header value repeated on every line | Revenue and volume inflated | Never store header totals on line tables |
| Treating order quantity as demand | Forecast error looks small against orders, large against sales | Orders contain batching, gaming, and cancellations | Wrong replenishment | Use consumption or POS where available |
| Counting events instead of netting quantities | Receipt counts exceed receipt documents | Reversals and re-posts add rows | Double-counted volume | Label each metric as count or quantity |
| Duplicate event ingestion | Totals drift upward between runs | At-least-once delivery with no dedup key | Persistent, compounding error | Deterministic event ID plus idempotent upsert |
| Reversal without linkage | Negative rows with no matching original | Correction posted as a new document | Unreconcilable balances | Require reversal-to-original reference |
| Mutable status overwrites history | Cannot answer "what did we know on date X" | Update in place | Point-in-time queries impossible | Version rows or use an event log |
| Snapshot cut-off undocumented | Two systems differ by a small constant | Different freeze instants | False discrepancy escalations | Publish cut-off with every snapshot |
| Snapshot used as a flow | No movement explanation available | State observation treated as movement | Cannot diagnose variance | Pair every snapshot with a roll-forward |
| Unit-of-measure mismatch | Quantities differ by a factor of 6, 12, or 24 | Cases versus eaches | Order-of-magnitude errors | Store UOM on every quantity; convert explicitly |
| One order assumed to be one shipment | Shipment count below order count in a period | Split shipments | Understated deliveries | Model the many-to-many bridge |
| Invoice line not linked to shipment line | Deductions cannot be attributed | Invoice references order only | Cost-to-serve analysis fails | Capture shipment reference at invoice |
| Current-state table used for training | Offline metrics beat production sharply | Restatements and amendments leak | Model learned the future | Build features as of transaction time |

Two of these deserve a note.

**The silent overstatement** is the most dangerous because it produces no error: a query returning
350 instead of 150 runs successfully, plots cleanly, and is believed. Detect it by reconciling
against an independent total — the order system's period total, a physical count, or a source
control total — and build the check before the dashboard.

**The two-system discrepancy** is the most time-consuming because it is usually not a defect. A
team that treats every ERP-WMS difference as an incident spends its capacity on cut-off noise and
misses the repeating item-level variances. Segment it: small and random is noise, repeating is signal.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Each table has one grain | Grain statements and joins depend on it | Mixed-grain legacy extracts | Profile key uniqueness and row counts by key |
| Document numbers are stable | Joins across the chain use them | Renumbering, mergers, system migration | Check referential match rates across stages |
| Line numbers are stable across stages | Line-level linkage requires it | Re-line-numbering on split or amendment | Compare line counts and references |
| Quantity units are consistent | Sums require a common unit | Cases, eaches, kilograms mixed | Reconcile one document by hand |
| Event timestamps are recorded close to event time | Lead-time and latency analysis depends on it | Batch posting, paper documents | Compare event and posting timestamps |
| The movement log is complete | Event-derived inventory depends on it | Missing transfers, unposted issues | Roll-forward against two snapshots |
| Corrections are additive | Net quantity is computed by summing | Overwrites with no negative row | Search for gaps in document sequences |
| One currency per analysis | Value sums require it | Multi-currency orders with dated rates | Check currency distribution per period |
| The order-to-shipment link is populated | Delivery measurement requires it | Direct-to-consumer or EDI-only flows | Measure null rate on the link |
| Snapshot and movement share a time zone | Reconciliation requires it | Regional warehouses, cloud defaults | Audit stored offsets |

Three limits of the approach are worth stating plainly.

**Grain discipline does not fix missing data.** If a company never records allocations, no modelling
recovers them; the correct output is a documented gap.

**Bitemporal modelling is expensive.** It multiplies storage and needs a reliable transaction
timestamp from every source. Where only the latest state exists, record the extraction timestamp
and treat it as a lower bound on transaction time.

**The chain is not universal.** A service order may have no shipment, a marketplace order no seller
invoice. The grain questions still apply; the documents do not.

## 12. Exercises

### Exercise 1 — Grain audit (application, 45 minutes)

Take six tables from a schema you can inspect, or from the AICPA order-to-cash table list. For each,
write the grain in one sentence, the natural key, and whether each numeric measure is additive,
semi-additive, or non-additive at that grain.

**Deliverable:** a six-row table with columns: table, grain sentence, natural key, measure
additivity.
**Acceptance criteria:** every grain sentence is in business terms and contains no foreign-key
names; at least one table is shown to have a key that contradicts its documented grain; at least
one measure is correctly classified as non-additive.

### Exercise 2 — Fan-out arithmetic (calculation, 30 minutes)

Order SO-2087 has three lines: L1 200 units at EUR 5.00, L2 100 units at EUR 8.00, L3 60 units at
EUR 12.00. It ships as: S1 (L1 80, L2 100), S2 (L1 70), S3 (L1 50, L3 60).

**Deliverable:** the naive join result, the naive sum of ordered quantity and ordered value, the
true totals, and both inflation factors.
**Acceptance criteria:** arithmetic shown line by line; the explanation states why the quantity and
value inflation factors differ; the corrected query shape is described in words.

### Exercise 3 — Inventory roll-forward (calculation, 40 minutes)

SKU-C at one location in July: opening 480; receipts 180 and 120; shipments 260; customer returns
15; cycle-count adjustment −8. A WMS snapshot at 17:00 on 31 July shows 535; the ERP month-end
shows 527; a further 8 units shipped at 19:00.

**Deliverable:** the roll-forward, the expected on-hand at midnight, the variance against each
snapshot, and the explanation.
**Acceptance criteria:** the equation is written with signs; both snapshots are reconciled to the
same instant; the explanation names the cut-off rather than calling the difference an error.

### Exercise 4 — Idempotent load design (design, 40 minutes)

Design the load for a receipt feed delivered at least once, where receipts can be reversed and
re-posted, and where the source sometimes resends a batch.

**Deliverable:** the event key, the deduplication mechanism, the handling of reversals, and the
reconciliation check that would detect a missed duplicate.
**Acceptance criteria:** the key is deterministic and survives a resend; reversals net correctly;
the check compares a derived total against a source control total; the design states what happens
on a late-arriving event.

### Exercise 5 — Bitemporal schema (design, 50 minutes)

Design the storage for order lines that can be amended, and answer three queries: quantity valid
on a past date as known then; quantity valid on a past date as known now; all amendments made in a
period.

**Deliverable:** a table definition with valid-from, valid-to, recorded-at, and superseded-at
columns, plus the three queries in plain language or pseudo-SQL.
**Acceptance criteria:** the two time axes are distinct and named; each query states which axis it
uses; the design explains what happens to a same-day amendment and re-amendment.

### Exercise 6 — Document chain mapping (application, 40 minutes)

Choose a business you know or can research. Map its quote-to-cash chain, naming the system that
holds each document and the grain of each record.

**Deliverable:** a one-page chain diagram with system names, record names, and grains, plus three
break points.
**Acceptance criteria:** at least six links; each link names a system and a grain; each break point
names the record that is missing or unreliable and the analytical question it blocks.

### Exercise 7 — Three-way match (calculation, 45 minutes)

A purchase order covers 500 units at EUR 4.00. The receipt records 480 units. The invoice bills 500
units at EUR 4.10.

**Deliverable:** the matched payment, the total discrepancy, and its decomposition into quantity
and price components.
**Acceptance criteria:** arithmetic shown; quantity and price effects separated and shown to sum to
the total; a statement of which document must be corrected first.

### Exercise 8 — Reading an ASN (application, 35 minutes)

Obtain a sample EDI 856 (vendor guides publish redacted examples) or use the structure described in
section 8.2. Identify the hierarchy structure code, the HL levels present, and the grain of the item
level.

**Deliverable:** a table of HL levels with the segment that identifies each and the quantity that
each level carries.
**Acceptance criteria:** the declared structure matches the levels found; the CTT count is checked
against the number of HL segments, not items; the item-level grain is stated in one sentence.

### Exercise 9 — Leakage audit (critique, 50 minutes)

Take a feature list for a demand or lead-time model. For each feature, identify its availability
time relative to the prediction timestamp, and flag any that could not have been known.

**Deliverable:** a table of features with availability time, source table, and leak status.
**Acceptance criteria:** at least five features assessed; at least one leak found and explained
mechanically (amendment, restatement, late posting, or aggregation from future data); the fix
names the as-of join or the transaction-time filter required.

### Exercise 10 — Design memo (design, 45 minutes)

Write a one-page memo proposing the transaction-data model for one decision, such as daily
replenishment, stating the grain of every table.

**Deliverable:** the decision, the tables, the grain of each, the key, the as-of rule, and the
reconciliation check.
**Acceptance criteria:** every table has a one-sentence grain; the as-of rule is explicit; the
reconciliation check is specific and runnable; the memo names one failure mode it prevents.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
ordered quantity by line: L1 200, L2 100, L3 60   total 360
ordered value by line:    L1 1,000; L2 800; L3 720   total 2,520

shipment rows per line:   L1 3, L2 1, L3 1

naive sum of qty_ordered:
  L1: 200 x 3 = 600
  L2: 100 x 1 = 100
  L3:  60 x 1 =  60
  total       = 760          inflation 760 / 360 = 2.11

naive sum of line_value:
  L1: 1,000 x 3 = 3,000
  L2:   800 x 1 =   800
  L3:   720 x 1 =   720
  total         = 4,520      inflation 4,520 / 2,520 = 1.79
```

The factors differ because inflation is per line, not global. L1 is tripled; L2 and L3 are not. A
single "inflation factor" is therefore not a constant that can be corrected by division. The right
fix is to aggregate `shipment_line` to `order_line_id` first, then join, which yields 200, 100, 60
and 2,520 exactly.

A common wrong answer divides the naive total by the average number of shipments per line (760 / 1.67
= 455), which recovers nothing. Correction must happen at the grain, not at the total.

**Exercise 3 arithmetic.**

```text
expected = 480 + 180 + 120 - 260 + 15 - 8
         = 480 + 300 - 260 + 15 - 8
         = 527

variance vs ERP (527)      = 527 - 527 = 0
variance vs WMS 17:00 (535) = 535 - 527 = +8
```

The +8 is the eight units that had not yet shipped at 17:00. The WMS snapshot is a correct
observation of a different instant. Restating it to midnight gives 527, and both systems agree.
The sign convention matters: snapshot minus expected is positive when the snapshot is taken before
an outbound movement.

**Exercise 7 arithmetic.**

```text
approved quantity = min(ordered 500, received 480) = 480
approved price    = PO price 4.00
matched payment   = 480 x 4.00 = 1,920.00
invoice total     = 500 x 4.10 = 2,050.00
discrepancy       = 2,050.00 - 1,920.00 = 130.00

quantity component = (500 - 480) x 4.00 = 20 x 4.00 = 80.00
price component    = 500 x (4.10 - 4.00)   = 500 x 0.10 = 50.00
check              = 80.00 + 50.00 = 130.00
```

The receipt must be corrected first if the 20 units were genuinely short-shipped, because the
quantity discrepancy changes what is owed. If the units arrive later, the invoice can be paid in
full once the receipt is updated. The price component is a separate commercial dispute and should
not be netted into the quantity investigation.

**Exercise 1 rubric.**
A strong submission writes grain sentences such as "one item on one purchase order, before any
split by requested date" and identifies a table whose primary key includes a shipment identifier
despite being documented at order-line grain. A weak submission writes "order data" or lists
foreign keys. Additivity is often got wrong on balances and on percentages; a strong answer marks
on-hand as semi-additive and fill rate as non-additive, and says why.

**Exercise 4 rubric.**
A strong design uses a deterministic key composed of source system, document type, document number,
line number, and event sequence, hashed if necessary. It stores a deduplication record with a
uniqueness constraint, applies an upsert on that key, and represents reversals as separate events
linked to the original. It reconciles the loaded net quantity against a source control total and
alerts on a mismatch rather than silently adjusting. A weak design deduplicates on a timestamp,
which fails when two legitimate events share a second, or deletes duplicates after load, which
leaves the totals wrong during the window.

**Exercise 5 rubric.**
A strong answer separates valid time from transaction time explicitly, uses half-open intervals,
and states the convention for same-day amendments. It answers the three queries with different
filters: transaction time for "as known then", valid time for "as things were", and a transaction
time range for "amendments made". A weak answer adds a single `updated_at` column, which records
only the last change and cannot reconstruct any prior state.

**Exercise 8 rubric.**
A strong answer identifies the hierarchy structure code in the BSN segment, lists the levels in
order, and notes that CTT counts HL segments. It states the item-level grain as "one product within
one pack within one shipment" rather than "one product in the shipment". A weak answer flattens the
message and reports item quantities without the packaging context, which is the information the
ASN exists to carry.

**Exercise 9 answer outline.**
Typical leaks: an order-status field updated in place (future amendment visible); an average order
quantity computed over the full history including later orders; a lead-time feature built from
receipt dates that were posted after the prediction date; a stock-out flag derived from a snapshot
taken after the decision; a supplier scorecard restated at period end. The fix in each case is an
as-of join on transaction time, with a source-delay offset so that data still in the pipeline at
prediction time is excluded.

**Exercise 10 rubric.**
A strong memo names one decision, three to six tables, a grain sentence for each, and a single
as-of rule that applies to all of them. It states the reconciliation check as a comparison against
a source total, and it names the failure it prevents in mechanical terms. A weak memo lists tables
without grains and proposes a dashboard rather than a decision.

## 14. Interview questions

**1. "What is data grain, and why does it matter?"**
A strong answer defines grain as the business meaning of one row, gives an example at two levels,
and explains that joins and sums are only valid at a declared grain. A weak answer says "the level
of detail" and stops, which does not distinguish grain from aggregation.

**2. "Why should you avoid joining two fact tables directly?"**
A strong answer explains that the cardinality is uncontrolled, that a one-to-many join repeats the
parent's measures, and that the result is silently wrong rather than erroring. It names
pre-aggregation or drill-across as the alternative. A weak answer says "it is slow" or "it creates
duplicates", which misdiagnoses the mechanism.

**3. "An order says 100 units; the shipments sum to 90. Which is right?"**
A strong answer says both, for different questions: 100 ordered, 90 shipped, 10 open or cancelled.
It asks whether the remaining 10 are backordered, cancelled, or lost, and notes that the answer
changes the demand measure. A weak answer picks one and calls the other wrong.

**4. "How would you deduplicate a receipt feed?"**
A strong answer describes a deterministic event key, a uniqueness constraint or dedup table, an
idempotent upsert, and a reconciliation against a source control total. It mentions that
at-least-once delivery is the realistic default. A weak answer says "use distinct", which does not
work when two legitimate rows differ only in a load timestamp.

**5. "When would you use a snapshot rather than reconstructing from events?"**
A strong answer says a snapshot is right when you need a fast state read or when the movement log
is incomplete, and that it must carry a documented cut-off. It notes that a snapshot cannot explain
itself and should be paired with a roll-forward for diagnosis. A weak answer says snapshots are
always better because they are simpler.

**6. "What does 'as of' mean in a query?"**
A strong answer distinguishes valid time from transaction time and says which one the query uses.
It gives an example of an amendment where the two answers differ. A weak answer says "the state at
that date", which conflates the two axes.

**7. "Why does a model trained on current-state order data leak?"**
A strong answer explains that overwritten amendments and restatements embed information that was
not available at decision time, that offline metrics rise while production performance does not,
and that the fix is an as-of join with a source-delay offset. A weak answer says "overfitting".

**8. "How do you model an order that ships in three parts?"**
A strong answer keeps order lines at their own grain, adds a shipment-line bridge with
`quantity_shipped`, and aggregates the child before joining to the parent. It notes that a partial
shipment is a smaller number on the bridge, not a duplicated parent row. A weak answer adds three
shipment columns to the order line, which fails at four shipments.

**9. "A receipt is reversed and re-posted. What does your fact table show?"**
A strong answer says three events, net quantity correct, event count inflated, and that the metric
definition must state which is being reported. It mentions linking the reversal to the original. A
weak answer says the reversal deletes the original.

**10. "What is the difference between an invoice line and an order line?"**
A strong answer says they are different grains, usually different counts, and may reference each
other through a shipment line. It notes that invoice value is billed value, not ordered value, and
that a credit note is a separate correcting record. A weak answer treats them as the same thing at
different stages.

**11. "How would you test whether a transaction data model is correct?"**
A strong answer lists reconciliation checks: row counts by key, sums against source control totals,
referential match rates across stages, and a roll-forward against a snapshot. It says the checks
should run as pipeline tests, not as one-off analysis. A weak answer says "look at the data".

**12. "What breaks if you use the order header as the grain?"**
A strong answer says you lose item-level analysis, cannot handle multi-item orders without
arrays or repeated columns, cannot join to shipments without fan-out, and cannot compute fill rate
by item. A weak answer says it is "less detailed", which understates the structural loss.

## 15. Key terms

- **Grain** — the business meaning of one row; the contract every query depends on.
- **Event** — a timestamped fact that something happened.
- **Order header / order line** — the order as a whole, and one item within it.
- **Schedule line** — one requested date within an order line.
- **Allocation** — a promise of specific supply to specific demand.
- **Backorder** — the unshipped remainder of demand.
- **Shipment header / shipment line** — one dispatch, and one item within it.
- **Handling unit** — a pallet, case, or tote with its own identity.
- **ASN** — advance ship notice; a claim about contents sent before arrival.
- **HL** — hierarchical level segment in an X12 ASN; the flattened tree.
- **Receipt** — acceptance of goods at a location.
- **Invoice header / invoice line** — one billing document, and one item within it.
- **Credit note** — a correcting negative invoice.
- **Three-way match** — agreement of order, receipt, and invoice before payment.
- **Inventory snapshot** — a quantity observed at a stated instant.
- **Perpetual inventory** — quantity maintained by applying every movement.
- **Roll-forward** — opening balance plus movements, used to test a snapshot.
- **Cut-off** — the instant at which a report's data is frozen.
- **Valid time** — when a fact was true in the world.
- **Transaction time** — when a fact was recorded in the system.
- **Bitemporal** — modelling both time axes together.
- **As-of join** — a join returning the version known at a given time.
- **Idempotency** — repeating an operation without changing the outcome.
- **Fan-out** — row multiplication from a one-to-many join.
- **Additive / semi-additive / non-additive** — whether a measure sums across dimensions, across
  some but not time, or not at all.

## 16. Sources

**Peer-reviewed and academic**

- "Reconciling tuple and attribute timestamping for temporal data warehouses." *The VLDB Journal*,
  34(1), 2024. DOI: 10.1007/s00778-024-00889-2. Grade A. Compared timestamping approaches for
  temporal warehouses on a TPC-DS-derived benchmark.
- "Reconciliation of Inventory Between Oracle ERP Inventory and Oracle Cloud WMS." *Journal of
  Emerging Technologies and Applications (JETA)*, 5(1), 2025. Grade A, small journal. Reports the
  structural two-system reconciliation problem.
- Koncilia, C. "A Bi-Temporal Data Warehouse Model." *CEUR Workshop Proceedings*, Vol. 74. Grade A
  for the model, workshop publication.

**Standards and official**

- AICPA, *Audit Data Standards: Order-to-Cash Subledger*, July 2015. Grade B. Defines the table set
  and fields for the quote-to-cash chain.
- ANSI X12 856 Advance Ship Notice, implementation guides as published by trading partners and
  networks. Grade B. Defines the HL hierarchy and control segments.
- SAP Business Network, *X12 SH856 4010 Inbound* implementation guide. Grade B. Detailed level
  tables, BSN05 rules, and worked examples.
- GS1 General Specifications, including SSCC and GTIN. Grade B. Defines the identifiers used in
  ASN labels and item-level segments.
- SQL:2011 temporal features. Grade B; implementation guidance in Teradata's paper is Grade C.

**Practitioner and vendor documentation**

- Kimball Group, "Grain," "Declaring the Grain" (2003), "Fact Tables" (2008), and "Multipass SQL to
  Avoid Fact-to-Fact Table Joins." Grade C. Practitioner canon on grain and fact-table joins.
- Martin Fowler, "Bitemporal History," 2021. Grade C. Distinguishes actual history from record
  history and describes two storage approaches.
- Microsoft Azure Architecture Center, "Event Sourcing Pattern." Grade C. Append-only stores,
  idempotent handlers, optimistic concurrency.
- Microsoft Learn, "Offline feature retrieval using a point-in-time join," and Azure Databricks,
  "Point-in-time feature joins," both accessed 2026. Grade C. Source-delay and lookback semantics.
- Microsoft Learn, "Balance inventory between Dynamics 365 and an external WMS." Grade C. States
  the physical-versus-financial system-of-record split.
- Oracle, PeopleSoft supply chain integration assumptions, and Oracle WMS 26B reconciliation
  workbench release notes, 2026. Grade C. Reconciliation behaviour and attribute granularity.
- Oracle, "Understanding Counts," and SKU.io, "How stock takes work." Grade C. Snapshot time versus
  count date and the treatment of intervening movements.
- Confluent, "Idempotent Reader" pattern, and "Exactly-Once Semantics" in *CDC: The Missing
  Manual*. Grade C/D. At-least-once transport with idempotent sinks.
- Teradata, *Enabling the Temporal Data Warehouse*. Grade C. `TRANSACTIONTIME`, `VALIDTIME`, and
  `AS OF` query forms.
- InterSystems, "Principles and Recommendations," Supply Chain Orchestrator 2026.1 documentation.
  Grade C. Explains measure duplication when a measure is based on a parent table.
- DataDriven, "The Long Route" data-modelling practice problem. Grade C. Shipment-line bridge at
  order-line-within-shipment grain.

**Vendor and analyst marketing**

- EDI integration vendor guides on 856 hierarchy and ASN rejection, accessed 2026. Grade D. Useful
  for mechanism, not magnitude; no independent rejection-rate figure was found.
- Order-management and BI integration vendor articles on split shipments and reporting grain.
  Grade D. Consistent with the modelling literature; treat specific numbers as unverified.

**Note on citations.** Every source above was returned by the research searches for this topic;
where a standard is named in the body for context, section 8 marks it "named for context". No
statistic or figure here is attributed to a source that did not report it.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_master-data.md`](02_master-data.md)
- Related: [`01_systems-of-record.md`](01_systems-of-record.md)
- Next in sequence: [`08_data-quality-dimensions.md`](08_data-quality-dimensions.md)
- Related: [`../01_supply_chain_foundations/05_inventory-and-service.md`](../01_supply_chain_foundations/05_inventory-and-service.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
