# GS1 Identifiers and EPCIS 2.0

> Module 03 — Data Engineering and Interoperability
> Topic 09 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01.03, 03.08

---

## 1. Why this topic exists

A lid supplier reports a moulding defect at 09:00. The brand owner has to decide, before the
next production shift, which pallets of yoghurt to hold. The enterprise resource planning (ERP)
system knows that lid lot LID-77 was consumed on line 2. It does not know which cases of finished
product contain those lids, which pallets those cases were packed onto, which distribution
centres received the pallets, or which stores received cases after a pallet was broken down at a
distribution centre (DC). Answering the question takes four departments, two spreadsheets, and
three days. The answer is wrong in both directions: some unaffected pallets are held, and some
affected cases reach a shelf.

This is not a data volume problem. The company has the production records, the warehouse
transactions, and the shipment notices. It lacks a shared way to say **which object** an event
happened to, at **which grain**, and **how objects were linked** to one another. GS1 identifiers
supply the nouns. EPCIS supplies the verbs. Neither, alone, is a traceability system.

Three failures recur, and each has a different shape.

1. **The recall that is too wide.** Without a case-to-pallet link, the only defensible scope is
   every pallet produced that day. The company holds 42 pallets instead of the 9 that actually
   contain the affected lid lot. At EUR 604.80 of product per pallet, the over-hold is EUR 19,958
   of inventory, plus warehouse labour, plus the shelf-life cost of product that returns late or
   not at all.
2. **The recall that is too narrow.** The warehouse records pallet receipts but not the
   break-down of a pallet into store orders. A query that starts from the pallet identifier stops
   at the DC, and three store pallets never appear in the result. The missing scope is small in
   units and total in consequence: those are the pallets that reach consumers.
3. **The evidence that cannot be produced.** A regulator requests lot-level records for a
   shipment within 24 hours. The lot code exists, but as free text in a production note, a
   portable document format (PDF) attachment, and a partner portal that no longer accepts the
   company's login. The data exists. The evidence does not.

**Why this matters to a data engineer specifically.** Event data is the least distorted view of
the physical flow available. An order states an intention. An invoice states a financial
consequence. An event states that an identified object was observed at a place and time, in a
business context. That makes event data the right spine for traceability, dwell-time analysis,
network reconstruction, and recall scope. It is also unusually easy to model badly, because an
event looks like a status row and is not one. It is immutable, it forms a graph, and it is
open-world: a missing event usually means a missing capture, not an absent fact.

This topic builds on the identifier stack in Topic 01.03 and the document-exchange conventions in
Topic 03.08. It does not repeat the entity taxonomy. It zooms into the one part of the entity
model that has to survive contact with a trading partner: the identifier that appears on a label,
and the event record that says what happened to the object carrying it.

> **Note:** regulatory material in this topic is educational. Compliance decisions require
> counsel and a current reading of the applicable text. Dates in this area move, and the text
> says where a date is proposed rather than final.

## 2. Learning outcomes

By the end of this topic you can:

1. **Select** the correct GS1 identifier for a stated decision grain — product type, product
   instance, lot, logistics unit, location, or asset — and **justify** the choice against the
   question the data must answer.
2. **Validate** a GTIN check digit by hand, **explain** the indicator digit, and **diagnose** a
   case that is received as a single each because two pack levels share one number.
3. **Read and write** an EPCIS 2.0 event, mapping each field to the what, when, where, and why
   tuple, and **distinguish** `eventTime` from `recordTime`.
4. **Model** aggregation and disaggregation as a relationship with a lifetime, and **explain**
   why a query that reads only the latest event returns a phantom pallet.
5. **Design** a recall-scope query that traverses transformation, aggregation, and shipment
   events, and **state** the point at which the available granularity stops the answer from
   getting finer.
6. **Distinguish** the capture, query, and subscription interfaces of an EPCIS repository, and
   **name** the operational failure each one is prone to.
7. **Map** the data elements required by the United States Food Traceability Rule (FSMA 204), the
   Drug Supply Chain Security Act (DSCSA), and the European Union Falsified Medicines Directive
   (EU FMD) onto EPCIS fields, and **state** where the mapping is incomplete.
8. **Critique** an adoption claim about EPCIS by identifying the population surveyed, the date,
   and whether the claim measures the standard or a vendor's implementation of it.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a graph whose nodes are identifiers and whose edges are events. The graph is
built over time, it is append-only, and every edge is an assertion by one party.

```text
   FROM LOTS TO PALLETS TO PLACES

   INPUT LOTS              OUTPUT LOT              LOGISTICS UNITS          LOCATIONS
   LID-77  --\
   CUP-31  ---+--[TRANSFORM]--> L-0412 --[AGG]--> SSCC ...0001 --[SHIP]--> GLN DC North
   YOG-08  --/                    |                 SSCC ...0002
                                  |                 SSCC ...0003
                                  +--[TRANSFORM]--> L-0413 --[AGG]--> SSCC ...0007

   edge types
     TRANSFORM  derivation: inputs consumed, outputs produced
     AGG        containment: cases packed onto a pallet; has a start and an end
     SHIP       movement: an object observed leaving, with a source and a destination
```

Read the picture left to right and you get provenance. Read it right to left and you get recall
scope. Read it at one instant and you get inventory position. The same graph answers all three,
which is why the identifier grain and the event vocabulary have to be fixed before the first
event is captured.

Five properties explain most of what follows.

**Property 1 — Identifiers are nouns and events are verbs.** An event without an identifier is a
sentence with no subject. A GTIN on a label does not say where the product is, and an event does
not say what the product is. The two are joined by the identifier, and the join fails silently
when the identifier is reused across pack levels.

**Property 2 — An event is an assertion, not a fact.** It records what a party observed or
declared, at a time and place, in a business context. A shipping event says the shipper asserted
that the goods left. It does not prove the goods were on the vehicle, and it does not prove the
receiver agreed. Design your quality rules around assertion, not around truth.

**Property 3 — Aggregation is a relationship with a lifetime, not an attribute.** A case is on a
pallet between the aggregation `ADD` and the aggregation `DELETE`. Before the `ADD` and after the
`DELETE`, the same case exists and is not on that pallet. A schema that stores `pallet_id` on the
case row overwrites history and cannot answer "where was this case on 12 April".

**Property 4 — Granularity is chosen once and inherited by every query.** If the smallest unit
you identify is the production lot, no query can narrow a recall below one lot. If the lot spans
two lid deliveries, the recall includes product made with the second delivery. Section 6.6
prices this limitation.

**Property 5 — Absence of an event is not evidence of absence.** The receiving party may not have
sent a receiving event, or may have sent it to a different repository. A query that treats
missing events as proof that the goods never arrived will under-scope a recall and overstate
inventory accuracy.

### 3.1 The five questions every event answers

EPCIS calls them the what, when, where, why, and — since version 2.0 — how. Each maps to fields,
and each field has a characteristic misuse.

| Question | Fields | Example | Characteristic misuse |
|---|---|---|---|
| What | `epcList`, `quantityList`, `parentID`, `childEPCs`, `inputEPCList`, `outputEPCList` | 60 cases of lot L-0412 | A quantity with no unit of measure |
| When | `eventTime`, `recordTime` | 2026-04-12T14:22:00Z | Local time stored without an offset |
| Where | `readPoint`, `bizLocation` | GLN of the packing hall | Scan point treated as the business location |
| Why | `bizStep`, `disposition`, `bizTransactionList`, `sourceList`, `destinationList` | `packing`, `container_closed` | `in_transit` used as a business step |
| How | `sensorElementList` | Temperature during a cold-chain leg | Sensor data with no calibration or unit |

### 3.2 Four event types plus one

Before version 2.0, EPCIS had four event types. Version 2.0 added a fifth, `AssociationEvent`,
and removed the old `QuantityEvent`. The distinction that matters is not the list. It is the
relationship each event records.

| Event type | Records | Typical use |
|---|---|---|
| `ObjectEvent` | Something observed about one or more objects | Shipping, receiving, inspecting, sampling |
| `AggregationEvent` | Physical containment of children in a parent | Cases packed onto a pallet; a pallet loaded into a container |
| `TransactionEvent` | Association of objects with a business transaction | Linking a shipment to a purchase order or an advance ship notice |
| `TransformationEvent` | Inputs consumed and outputs produced | Milk and cultures become a yoghurt lot |
| `AssociationEvent` | A persistent, non-containment association | A part installed in an assembly; a sensor fixed in a room |

The last one is the one people get wrong. An `AssociationEvent` is not an aggregation. The child
is not inside the parent in a way that packing can undo. It is attached, installed, or otherwise
associated for a period, and the parent may be a physical location rather than a logistics unit.
It is the only event type whose `parentID` may be a location identifier.

### 3.3 The identifier stack at a glance

Topic 01.03 introduced five layers of identity. Here is the same stack expressed as the question
each layer answers, because the choice of identifier is always a choice of question.

```text
   QUESTION                       IDENTIFIER            GRANULARITY
   Which product is this?         GTIN                  trade item type, per pack level
   Which unit is this?            SGTIN (GTIN+serial)   one instance
   Which batch is this?           LGTIN (GTIN+lot)      one lot or batch
   What is it packed in?          SSCC                  one logistics unit
   Where is it?                   GLN / SGLN            party, location, sub-location
   Which equipment is it?         GIAI / GRAI           asset instance, returnable type
```

An analyst who cannot say which row of this table a question lives on will join two tables at
different grains and produce a number that is arithmetically correct and operationally false.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| GTIN | Global Trade Item Number, a globally unique identifier for a trade item at a pack level | Treated as unique across pack levels | 8, 12, 13, or 14 digits |
| Indicator digit | First digit of a GTIN-14, distinguishing packaging levels | Ignored, so case and each collide | 1–8; 9 for variable measure |
| GLN | Global Location Number, identifying a party or a place | Assumed to identify physical places only | 13 digits |
| SSCC | Serial Shipping Container Code, identifying one logistics unit | Confused with a GTIN, or with a pallet type | 18 digits |
| SGTIN | Serialised Global Trade Item Number, a GTIN plus a serial number | Used when lot-level identification is sufficient | EPC URI or 96-bit tag |
| SGLN | Serialised Global Location Number, a GLN plus an extension | Confused with the bare GLN | EPC URI |
| LGTIN | GTIN plus lot, a class-level identifier for a batch | Treated as an instance identifier | EPC URI |
| GIAI | Global Individual Asset Identifier, for one asset instance | Confused with GRAI | EPC URI |
| GRAI | Global Returnable Asset Identifier, for a returnable asset type plus serial | Used for fixed equipment | EPC URI |
| GSRN | Global Service Relation Number, for a service relationship | Confused with a location | 18 digits |
| EPC | Electronic Product Code, an identifier for a physical object | Treated as a synonym for a barcode | URI or tag encoding |
| EPC URI | The text form of an EPC, such as `urn:epc:id:sgtin:...` | Stored without the scheme, so it cannot be resolved | `urn:epc:id:` or `urn:epc:tag:` |
| GS1 Digital Link | A web-resolvable identifier that carries GS1 keys in a URL path | Treated as a replacement for the GTIN | URI, HTTPS |
| Application Identifier (AI) | A GS1 prefix that says what the following data element means | Ignored when parsing a label | Two to four digits |
| Lot / batch | A quantity produced under conditions treated as one for traceability | Used interchangeably with serial number | Manufacturer-assigned code |
| Serial number | A value unique within a GTIN for one instance | Assumed globally unique on its own | Alphanumeric, up to 20 characters in FMD |
| EPCIS | Electronic Product Code Information Services, the GS1 standard for event data | Treated as a database product | Standard, version 2.0 |
| CBV | Core Business Vocabulary, the controlled values for business steps and dispositions | Treated as optional free text | Standard, version 2.0 |
| Event | An assertion that something happened to an object at a time and place | Treated as a current-status row | Record |
| ObjectEvent | An event about one or more objects, without containment semantics | Used for pallet packing | Event type |
| AggregationEvent | An event recording physical containment of children in a parent | Used for commercial bundles | Event type |
| TransactionEvent | An event associating objects with a business transaction | Used to record pallet contents | Event type |
| TransformationEvent | An event recording inputs consumed and outputs produced | Given an `action` value, which it does not have | Event type |
| AssociationEvent | An event recording a persistent non-containment association | Used for packing | Event type, new in 2.0 |
| `action` | The change an event asserts: `ADD`, `OBSERVE`, or `DELETE` | Extended with custom values | Enumerated, three values |
| `bizStep` | The business process step during which the event occurred | Used to record a state | CBV value or URI |
| `disposition` | The business state of the objects after the event | Used to record a location | CBV value or URI |
| `persistentDisposition` | A disposition that remains set across later events until unset | Assumed to expire with the event | CBV value, with set/unset |
| `readPoint` | The place where the event was observed | Treated as the business location of the goods | SGLN |
| `bizLocation` | The business location where the objects were at the event time | Omitted when it differs from the read point | SGLN |
| `bizTransaction` | A business transaction identifier associated with the event | Used as the object identifier | URI plus transaction type |
| `sourceList` / `destinationList` | The parties or locations the objects came from or went to | Assumed to be the same as read point | Party or location, typed |
| ILMD | Instance or lot master data, attached to a lot or instance | Used as a general product master | Key-value attributes |
| `sensorElement` | Sensor metadata and readings attached to an event | Stored without units or calibration | Structured element |
| Aggregation | The act of physically grouping children into a parent | Confused with a quantity total | Relationship |
| Disaggregation | The act of removing children from a parent | Omitted, leaving a stale link | Relationship ending |
| `parentID` | The identifier of the containing or associated parent | Assumed always to be an SSCC | EPC |
| `childEPCs` / `childQuantityList` | The children in an aggregation, by instance or by class and quantity | Mixed without stating units | EPC list or quantity list |
| Capture interface | The interface that accepts events into a repository | Assumed to validate business truth | REST operation |
| Query interface | The interface that retrieves events by filters | Assumed to return a current state | REST operation |
| Subscription | A standing query that pushes new events to a consumer | Assumed to guarantee delivery | REST operation |
| Repository | The service that stores and serves EPCIS events | Confused with the company data warehouse | Role, not a product |
| `eventTime` | When the event occurred in the physical world | Recorded at end of shift | ISO 8601 with offset |
| `recordTime` | When the repository recorded the event | Assumed to equal `eventTime` | ISO 8601 with offset |
| KDE | Key Data Element, a required data field under FSMA 204 | Treated as a document format | Regulatory term |
| CTE | Critical Tracking Event, a point in the food chain where KDEs must be captured | Confused with an EPCIS event type | Regulatory term |
| TLC | Traceability Lot Code, the lot identifier used across a food chain | Assumed to be the manufacturer's internal lot code | Regulatory term |
| Master data | The slow-changing attributes of an entity | Expected to travel inside EPCIS events | Reference data |

Five of these cause disproportionate damage.

**GTIN, SGTIN, and LGTIN.** A GTIN says "this is a 150 g yoghurt cup". An SGTIN says "this is cup
number 000000000001 of that product". An LGTIN says "this is the lot L-2026-0412 of that product".
A serialised system can answer questions a lot-level system cannot, and pays for it in label
cost, scan cost, and data volume. Most food traceability is lot-level; most pharmaceutical
traceability is instance-level. Choosing instance-level for a product that will never be
recalled by serial is an expensive way to store data nobody queries.

**GLN and SGLN.** A GLN identifies a party or a location. An SGLN adds an extension that can
identify a sub-location, such as a specific dock door or a room inside a plant. The extension `0`
conventionally means the GLN itself. A read point of "dock door 7" is an SGLN; the business
location is the site GLN. Recording only the read point and calling it the location loses the
distinction between where the scanner was and where the goods were.

**Aggregation and association.** Aggregation is containment that packing creates and unpacking
ends. Association is a persistent attachment that packing does not describe: a component installed
in a machine, a sensor mounted in a cold room, a reusable crate assigned to a pool. Using an
`AggregationEvent` for installation loses the fact that the relationship is not a logistics unit
and may outlive every shipment.

**Business step and disposition.** A business step is what was being done (`packing`, `shipping`,
`receiving`, `inspecting`). A disposition is the state of the object afterwards (`container_closed`,
`in_transit`, `sellable_accessible`, `recalled`). `in_transit` is not a business step, and
`shipping` is not a disposition. Mixing them produces a vocabulary that cannot be queried
consistently across partners.

**Event time and record time.** `eventTime` is when the thing happened. `recordTime` is when the
repository learned about it. A receiving event recorded at 23:55 for goods that arrived at 08:10
has an eleven-hour gap, and any dwell-time analysis that uses `recordTime` will report a warehouse
that is far faster than it is. The gap is also the first diagnostic for batch capture at end of
shift.

## 5. The mechanics, step by step

This section follows the order a competent implementation takes: decide the identity, capture the
event, build the links, exchange the records, then reconcile. Each step names the decision and
the trap.

### Phase A — Design the identity

**Step 1. Choose the decision unit before the identifier.**

- What it is: naming the finest object the business must be able to act on. For a recall, it is
  the smallest quantity that can be independently held or destroyed. For inventory, it is the
  stock-keeping unit (SKU). For verification, it is the pack.
- Why it matters: every identifier choice is downstream of this one, and the choice is expensive
  to reverse because labels, partner masters, and historical events all carry it.
- Trap: choosing the finest grain "because storage is cheap". Serialising every cup multiplies
  event volume by the pack factor, adds label cost, and does not improve a recall that is decided
  at pallet level.

**Step 2. Assign the identifier at each grain.**

- What it is: a GTIN per trade item per pack level, an SSCC per logistics unit, a GLN per party
  and location, and a lot or serial identifier where the decision unit requires it.
- Why it matters: identifiers are the join keys to partners. A missing GTIN means the product
  cannot appear in an advance ship notice (ASN) or an EPCIS event that a retailer will accept.
- Trap: allocating SSCCs from a range that is not registered to the company prefix. The check
  digit validates, the label scans, and the partner rejects the shipment because the prefix is
  not yours.

**Step 3. Validate the structure.**

- What it is: length, character set, check digit, and storage type. GS1 identifiers are digits;
  store them as text, because leading zeros are meaningful and a fourteen-digit form is a
  different identifier from its thirteen-digit parent.
- Why it matters: a numeric column silently drops the leading zero, and the identifier stops
  matching the partner's master.
- Test: recompute the check digit on every identifier at ingestion. The algorithm is in section
  8.1 and worked in section 13.

**Step 4. Publish master data separately.**

- What it is: the descriptive attributes — description, net content, dimensions, weights,
  classification — held in a product data pool or exchanged at onboarding.
- Why it matters: EPCIS identifies objects. It does not describe them. A repository full of
  events and no master data can tell you that SSCC ...0001 moved, but not what was on it or what
  it weighed.
- Trap: expecting ILMD to be the product master. ILMD carries lot-specific attributes such as
  harvest date or best-before date. It is not a substitute for a GTIN-keyed item master.

### Phase B — Capture the event

**Step 5. Choose the event type from the relationship.**

| Question the event must answer | Event type | Why not another |
|---|---|---|
| Was this object observed, and in what state? | `ObjectEvent` | No containment or transaction semantics needed |
| What is inside this pallet or container? | `AggregationEvent` | `TransactionEvent` does not imply physical containment |
| Which purchase order or ASN does this shipment belong to? | `TransactionEvent` | Association with a document, not with a parent object |
| What did this lot consume, and what did it become? | `TransformationEvent` | The only type with inputs and outputs |
| What is installed in this asset, or attached to this location? | `AssociationEvent` | Persistent, non-containment, parent may be a location |

- Trap: using `TransactionEvent` for pallet contents because the ASN lists the cases. The ASN
  associates cases with a shipment. It does not assert that the cases are inside a specific
  pallet, and a query for pallet contents will return nothing.

**Step 6. Fill the what.**

- What it is: the identifiers of the objects. Instance-level events use `epcList`; class-level
  events use `quantityList` with `epcClass`, `quantity`, and `uom`.
- Why it matters: a quantity without a unit of measure is not a quantity. "60" could be cases,
  eaches, or kilograms.
- Trap: mixing `epcList` and `quantityList` for the same objects and double-counting them in a
  downstream query.

**Step 7. Fill the when.**

- What it is: `eventTime` for occurrence, `recordTime` for recording, both in ISO 8601 with an
  explicit offset.
- Why it matters: dwell time, shelf life, and custody intervals are all differences of
  `eventTime`. A naive local timestamp makes a two-hour border delay look like a one-hour delay
  across a daylight-saving boundary.
- Trap: a warehouse management system (WMS) that timestamps at confirmation rather than at scan.
  The event time becomes the clerk's end-of-shift time, and the data looks precise while being
  hours wrong.

**Step 8. Fill the where.**

- What it is: `readPoint` for where the observation was made, `bizLocation` for where the objects
  were in business terms. Both are SGLNs, usually with extension `0` for the site itself.
- Why it matters: the difference is the difference between "scanned at gate 3" and "in the
  Utrecht DC". The first is a device location; the second is an inventory location.
- Trap: omitting `bizLocation` and letting consumers fall back to `readPoint`. A scan at a
  cross-dock gate then looks like storage at the gate.

**Step 9. Fill the why.**

- What it is: `bizStep` from the CBV list, `disposition` from the CBV list, plus
  `bizTransactionList` for related documents and `sourceList` / `destinationList` for the parties
  or places on either side.
- Why it matters: the why is what makes an event interpretable six months later. An event that
  says "60 cases moved" without a business step cannot be classified as a sale, a transfer, a
  return, or a correction.
- Trap: inventing local business-step strings. A partner's query for `shipping` will not match
  `SHIP_OUT`.

**Step 10. Attach lot attributes and sensor data where they belong.**

- What it is: ILMD on transformation outputs and commissioning events for lot-specific
  attributes; `sensorElementList` for temperature, humidity, or shock readings associated with
  the event.
- Why it matters: FSMA 204 requires commodity-specific key data elements such as growing area or
  harvest date, and those are lot attributes, not product attributes.
- Trap: putting sensor readings on every event in a cold chain. That multiplies event size and
  obscures the one leg where the excursion occurred.

### Phase C — Build the links

**Step 11. Aggregate at packing.**

- What it is: an `AggregationEvent` with `action: ADD`, a `parentID` (normally an SSCC), and the
  children either as `childEPCs` or as a `childQuantityList`.
- Why it matters: this is the event that makes a pallet a queryable object rather than a
  quantity. It is the link a recall traverses from lot to logistics unit.
- Trap: aggregating at the case level only when cases are serialised. If cases carry only a lot
  code, the child list is a class-and-quantity list, and the query must handle that form.

**Step 12. Disaggregate at break-down.**

- What it is: an `AggregationEvent` with `action: DELETE` removing the children, followed by new
  `ADD` events into the new parents.
- Why it matters: without the `DELETE`, the case appears to be on two pallets at once, and a
  recall query returns both. The phantom link is the most common defect in real repositories.
- Trap: deleting the parent without recording where the children went. A `DELETE` says the link
  ended; it does not say what happened next. The subsequent `ADD` events carry that information.

**Step 13. Record transformation.**

- What it is: a `TransformationEvent` with input and output lists and, where several events
  describe one production run, a shared `transformationID`.
- Why it matters: it is the only link from a raw material lot to a finished lot, and therefore
  the only way to answer "which products contain this ingredient lot".
- Trap: expecting the event to carry the bill of materials (BOM) quantities. It may carry input
  and output quantities, but it is not a consumption record. Yield analysis belongs in the
  manufacturing system, joined on lot.

**Step 14. Use association for persistent attachment.**

- What it is: an `AssociationEvent` with `action: ADD` when a component is installed or a device
  is attached, and `DELETE` when it is removed.
- Why it matters: it keeps installation relationships out of the aggregation graph, where they
  would corrupt pallet content queries.
- Trap: using association for a kit or a display shipper that is physically packed. That is
  containment and belongs in an aggregation.

### Phase D — Exchange the records

**Step 15. Capture into a repository.**

- What it is: a synchronous single-event post or an asynchronous bulk capture, with the
  repository assigning or accepting `recordTime`.
- Why it matters: capture is where idempotency lives. A retried post must not create a duplicate
  event.
- Trap: relying on the transport to deduplicate. Use the `eventID` as the idempotency key and
  reject or ignore a repeat.

**Step 16. Query by the dimensions that matter.**

- What it is: filters on object identifier, business step, disposition, location, event type,
  and time range, with pagination.
- Why it matters: the query interface is the only part of EPCIS most analysts ever touch, and its
  semantics are event-shaped, not state-shaped. `GET /events?EQ_bizStep=shipping` returns
  shipping events, not shipments.
- Trap: assuming a query returns the latest state. It returns events; reconstructing state is
  the consumer's job.

**Step 17. Subscribe for change, and plan for gaps.**

- What it is: a standing query whose new results are pushed to a webhook or a WebSocket, with a
  signature so the receiver can verify the sender.
- Why it matters: subscriptions are how a control tower learns about a receiving event without
  polling every partner.
- Trap: treating a subscription as a guaranteed queue. Networks drop, endpoints fail, and
  signatures expire. Keep a watermark and a replay query for the gap.

### Phase E — Reconcile

**Step 18. Reconstruct state as of a time.**

- What it is: replaying `ADD`, `DELETE`, and `OBSERVE` events in `eventTime` order to answer "what
  was on pallet X on date D".
- Why it matters: this is the difference between an event store and a status table. The event
  store can answer historical questions; the status table cannot.
- Trap: using `recordTime` for ordering. Late-arriving events are normal, and ordering by record
  time produces a history that never happened.

**Step 19. Reconcile the graph against physical counts.**

- What it is: comparing the number of open aggregation links against the pallet count in the
  WMS, and the number of shipped SSCCs against the number of receiving events.
- Why it matters: the graph drifts. A missed `DELETE` leaves a case on two pallets; a missed
  receiving event leaves a pallet in transit forever.
- Test: compute the ratio of open links to physical pallets weekly. A ratio above 1.05 usually
  means missing disaggregation.

**Step 20. Quarantine, correct, and never overwrite.**

- What it is: a correction event that asserts the correct state, plus a quarantine of the
  malformed original with its reason.
- Why it matters: EPCIS is append-only in spirit. Overwriting destroys the audit trail that the
  regulation, and the recall investigation, depend on.
- Trap: deleting a duplicate event to "clean up". If the duplicate was already consumed by a
  downstream system, deletion makes the downstream state unexplainable.
