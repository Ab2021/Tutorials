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
   every pallet produced that day. The company holds 42 pallets instead of the 9 that contain the
   affected lid lot. At EUR 604.80 per pallet, the over-hold is EUR 19,958 of inventory, plus
   labour and the shelf-life cost of product that returns late or not at all.
2. **The recall that is too narrow.** The warehouse records pallet receipts but not the
   break-down of a pallet into store orders. A query that starts from the pallet identifier stops
   at the DC, and three store pallets never appear in the result. The missing scope is small in
   units and total in consequence: those are the pallets that reach consumers.
3. **The evidence that cannot be produced.** A regulator requests lot-level records within 24
   hours. The lot code exists as free text in a production note, a portable document format (PDF)
   attachment, and a partner portal that no longer accepts the company's login. The data exists.
   The evidence does not.

**Why this matters to a data engineer specifically.** Event data is the least distorted view of
the physical flow. An order states an intention; an invoice states a financial consequence; an
event states that an identified object was observed at a place and time, in a business context.
That makes event data the right spine for traceability, dwell-time analysis, network
reconstruction, and recall scope. It is also easy to model badly, because an event looks like a
status row and is not one: it is immutable, it forms a graph, and it is open-world.

This topic builds on the identifier stack in Topic 01.03 and the document exchange in Topic
03.08. It zooms into the part of the entity model that must survive contact with a trading
partner: the identifier on a label, and the event record that says what happened to its object.

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
8. **Evaluate** whether an EPCIS implementation can answer a stated traceability question, and
   **separate** the exchange format from the traceability system around it.

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

**Property 5 — Absence of an event is not evidence of absence.** The receiver may not have sent
the event, or sent it elsewhere. A query that treats a missing event as proof of non-arrival will
under-scope a recall and overstate inventory accuracy.

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

**GTIN, SGTIN, and LGTIN.** A GTIN says "this is a 150 g yoghurt cup". An SGTIN adds a serial and
says "this is cup 000000000001". An LGTIN adds a lot and says "this is lot L-2026-0412".
Serialisation answers questions lot-level data cannot, and costs label, scan, and storage. Most
food traceability is lot-level; most pharmaceutical traceability is instance-level.

**GLN and SGLN.** A GLN identifies a party or a location. An SGLN adds an extension for a
sub-location such as a dock door; extension `0` conventionally means the GLN itself. A read point
of "dock door 7" is an SGLN, while the business location is the site GLN. Recording only the read
point loses the distinction between where the scanner was and where the goods were.

**Aggregation and association.** Aggregation is containment that packing creates and unpacking
ends. Association is a persistent attachment that packing does not describe: a component installed
in a machine, a sensor mounted in a cold room. Using an `AggregationEvent` for installation loses
the fact that the relationship is not a logistics unit.

**Business step and disposition.** A business step is what was being done (`packing`, `shipping`,
`receiving`). A disposition is the state afterwards (`container_closed`, `in_transit`, `recalled`).
`in_transit` is not a business step, and `shipping` is not a disposition. Mixing them produces a
vocabulary that cannot be queried across partners.

**Event time and record time.** `eventTime` is when the thing happened; `recordTime` is when the
repository learned about it. A receiving event recorded at 23:55 for goods that arrived at 08:10
has an eleven-hour gap, and any dwell-time analysis using `recordTime` reports a warehouse far
faster than it is.

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

- What it is: length, character set, check digit, and text storage.
- Why it matters: a numeric column drops the leading zero, and the identifier no longer matches
  the partner's master.
- Test: recompute the check digit at ingestion; the algorithm is in section 8.1.

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

**Step 19. Reconcile the graph, then correct without overwriting.**

- What it is: comparing open aggregation links against the physical pallet count, and shipped
  SSCCs against receiving events. A correction is a new event; the malformed original is
  quarantined with its reason.
- Why it matters: the graph drifts, and EPCIS is append-only in spirit. A missed `DELETE` leaves a
  case on two pallets; an overwritten record destroys the audit trail.
- Test: compute the ratio of open links to physical pallets weekly. Above 1.05 usually means
  missing disaggregation.

## 6. Worked example

Meridian Foods makes a 150 g yoghurt cup in Utrecht and distributes through two distribution
centres. A lid supplier reports a moulding defect in lid lot LID-77. This section follows the
recall from the supplier's notice to the store shelf, using only event data. All figures are
illustrative but internally consistent.

### 6.1 The product and the identifiers

| Level | Contents | Identifier | Identifier value (illustrative) |
|---|---|---|---|
| Cup | 150 g yoghurt | GTIN-13 | company prefix plus item reference |
| Case | 24 cups | GTIN-14, indicator 1 | `10614141107343` |
| Pallet | 60 cases = 1,440 cups | SSCC | `...0000001` to `...0000011` |
| Lot | production batch | LGTIN | GTIN plus `L2026-0412` |
| Plant | Utrecht site | GLN | `0614141.00021.0` |
| DC North | distribution centre | GLN | `0614141.00031.0` |
| DC South | distribution centre | GLN | `0614141.00032.0` |

The cup is the consumer unit and is not serialised. The case carries a GTIN-14 whose indicator
digit distinguishes it from the cup's GTIN-13. The pallet carries an SSCC, because two pallets of
the same product and lot are different physical objects and a recall must name one of them.

> **Note:** the case GTIN and the cup GTIN share the same base number. If the indicator digit is
> dropped, the case and the cup become one identifier, and a receiving system records 60 cases as
> 60 cups. Section 13 works the check digit for `10614141107343`.

### 6.2 Two lots and one lid delivery

Lid lot LID-77 arrives at 05:30 and is loaded on line 2. The line runs two production lots.

| Lot | Production window | Output | Cases | Pallets |
|---|---|---|---:|---:|
| L-2026-0412 | 06:00–13:40 | 8,640 cups | 360 | 6 |
| L-2026-0413 | 14:30–21:00 | 7,200 cups | 300 | 5 |

The lid change to LID-78 happens at 15:30, one hour into the second lot. The transformation event
for L-2026-0413 records LID-77 and LID-78 as inputs, because both were consumed during the lot.
It cannot say which output pallets contain which lid delivery. That limitation is the subject of
section 6.6.

### 6.3 The event chain

Eighteen events describe the flow. The table compresses repeated events but keeps the structure.

| # | Event type | Time (UTC) | What | Where | Why | Action |
|---:|---|---|---|---|---|---|
| 1 | TransformationEvent | 04-12 06:00–13:40 | in: LID-77, CUP-31, YOG-08; out: L-2026-0412 | plant GLN | production | none |
| 2 | AggregationEvent | 04-12 14:22 | 60 CS lot 0412 into SSCC ...0001 | packing hall | `packing` | ADD |
| 3–7 | AggregationEvent | 04-12 14:26–16:10 | 60 CS lot 0412 into SSCC ...0002–...0006 | packing hall | `packing` | ADD |
| 8 | ObjectEvent | 04-12 16:05 | SSCC ...0001–...0004 | dock 7 | `shipping` | OBSERVE |
| 9 | ObjectEvent | 04-12 16:40 | SSCC ...0005–...0006 | dock 8 | `shipping` | OBSERVE |
| 10 | TransformationEvent | 04-12 14:30–21:00 | in: LID-77, LID-78, CUP-31, YOG-08; out: L-2026-0413 | plant GLN | production | none |
| 11–15 | AggregationEvent | 04-12 21:20–23:40 | 60 CS lot 0413 into SSCC ...0007–...0011 | packing hall | `packing` | ADD |
| 16 | ObjectEvent | 04-13 07:10 | SSCC ...0007–...0008 | dock 7 | `shipping` | OBSERVE |
| 17 | ObjectEvent | 04-13 07:40 | SSCC ...0009 | dock 8 | `shipping` | OBSERVE |
| 18 | ObjectEvent | 04-13 18:40 | SSCC ...0001–...0004 | DC North inbound | `receiving` | OBSERVE |

Event 8 records `destinationList` with the DC North GLN. Event 9 records DC South. Events 16 and
17 record DC North and DC South respectively. Without `destinationList`, the recall query would
have to infer destination from a separate shipment table, and the join would be by shipment
identifier rather than by object.

### 6.4 The break-down that a naive query misses

Two days later, DC North breaks pallet SSCC ...0001 into three store orders of 20 cases each.

| # | Event type | What | Why | Action |
|---:|---|---|---|---|
| 19 | AggregationEvent | SSCC ...0001, 60 CS lot 0412 | `unpacking` | DELETE |
| 20 | AggregationEvent | 20 CS lot 0412 into SSCC ...0101 | `packing` | ADD |
| 21 | AggregationEvent | 20 CS lot 0412 into SSCC ...0102 | `packing` | ADD |
| 22 | AggregationEvent | 20 CS lot 0412 into SSCC ...0103 | `packing` | ADD |
| 23–25 | ObjectEvent | SSCC ...0101–...0103 | `shipping` | OBSERVE |

A query that stops at the pallet that left the plant returns SSCC ...0001 and reports it as
"at DC North". It does not return the three store pallets, because they did not exist when the
shipment event was captured. The case-level object is still traceable; the logistics-unit-level
object changed. This is why disaggregation is not housekeeping. It is the event that keeps the
graph connected to the physical world.

### 6.5 The recall query, step by step

The query is a traversal, not a lookup. Each step uses one event type.

| Step | Question | Event type | Predicate | Result |
|---:|---|---|---|---|
| 1 | Which output lots consumed LID-77? | TransformationEvent | input list contains the LID-77 lot | L-0412, L-0413 |
| 2 | Which pallets contain those lots? | AggregationEvent, ADD | child `epcClass` is the lot's LGTIN | 11 SSCCs |
| 3 | Which left the plant? | ObjectEvent | `epcList` contains the SSCC, `bizStep=shipping` | 9 SSCCs |
| 4 | Where did they go? | ObjectEvent | `destinationList` | DC North, DC South |
| 5 | Was any pallet broken down? | AggregationEvent, DELETE | `parentID` is the SSCC | 1 SSCC, 3 children |
| 6 | Where are the children? | ObjectEvent | `epcList` contains a child SSCC | 3 store SSCCs |

Step 5 is the step most implementations omit, because it requires a second pass over the graph.
The correct pattern is a recursive traversal: after finding a parent, look for `DELETE` events
with that parent, then look for `ADD` events with the same children, and continue until no new
logistics units appear.

### 6.6 The arithmetic of scope

Value per pallet is 1,440 cups at EUR 0.42 per cup.

```text
value per pallet       = 1,440 x 0.42            = EUR 604.80
pallets in both lots   = 6 + 5                   = 11
pallets shipped        = 4 + 2 + 2 + 1           = 9
pallets still at plant = 11 - 9                  = 2

hold cost, shipped     = 9 x 604.80              = EUR 5,443.20
hold cost, all lots    = 11 x 604.80             = EUR 6,652.80
naive day scope       = 42 pallets
naive hold cost        = 42 x 604.80             = EUR 25,401.60
over-scope factor      = 42 / 9                  = 4.67
```

The event data reduces the hold from 42 pallets to 9 in commerce plus 2 at the plant. The
difference is EUR 19,958.40 of product, before retrieval freight and disposal. That is the
business case for the event layer, and it is a better one than most analytics projects can make.

Now the granularity limitation. The true exposure is lot 0412 in full, plus the portion of lot
0413 produced before the lid change at 15:30. If that portion corresponds to roughly two pallets,
the ideal scope is 8 pallets. The event record supports 11, because the transformation is
recorded at lot level. The cost of the coarse grain is three pallets, or EUR 1,814.40 of
unnecessary hold, on this recall alone.

```text
ideal scope (illustrative) = 6 + 2              = 8 pallets
supported scope            = 11 pallets
granularity cost           = 3 x 604.80          = EUR 1,814.40
```

The fix is not a better query. It is a smaller output lot: one transformation event per
production window, with the lid delivery recorded against each. That decision has a cost too —
more lots, more labels, more events — and it belongs in the design, not in the recall.

### 6.7 What the events do not tell you

Six things are outside the event record, and each one matters to the decision.

1. **Whether the lids are actually defective.** The event says LID-77 was consumed. The defect
   notice says some of it was. The intersection is an investigation, not a query.
2. **How much of LID-77 went into each lot.** The transformation lists inputs, not consumption
   by output unit. Yield analysis needs the manufacturing system.
3. **Whether a receiving event is missing or the goods never arrived.** Absence is ambiguous.
   The recall must assume receipt until a physical count says otherwise.
4. **What the product is.** Description, weight, and classification live in master data keyed by
   GTIN. The event carries the key, not the description.
5. **Whether the pallet still contains what it contained.** The last aggregation event asserts a
   state at a time. A later break-down that was never captured leaves the assertion stale.
6. **Who is liable.** Events record custody and assertion. Liability is a contractual and legal
   question that the event graph can inform but not decide.

## 7. Code, query, or calculation

The example is a single aggregation event in the EPCIS 2.0 JSON binding, followed by a field
decode and the interface map. The point is to see how much meaning sits in one small record.

```json
{
  "type": "AggregationEvent",
  "eventID": "urn:uuid:0f5c1e7a-3b2d-4c8e-9a41-6d2f8b1c7e30",
  "eventTime": "2026-04-12T14:22:00Z",
  "recordTime": "2026-04-12T14:22:04Z",
  "action": "ADD",
  "bizStep": "packing",
  "disposition": "container_closed",
  "readPoint": {"id": "urn:epc:id:sgln:0614141.00021.0"},
  "bizLocation": {"id": "urn:epc:id:sgln:0614141.00021.0"},
  "parentID": "urn:epc:id:sscc:0614141.0000000001",
  "childQuantityList": [
    {
      "epcClass": "urn:epc:class:lgtin:0614141.107343.L2026-0412",
      "quantity": 60,
      "uom": "CS"
    }
  ]
}
```

### 7.1 Decoding the event

| Field | Question | Value here | What to check |
|---|---|---|---|
| `type` | Which event type? | `AggregationEvent` | Containment, not transaction or association |
| `eventID` | Which event is this? | A UUID uniform resource identifier (URI) | Unique; used as the idempotency key |
| `eventTime` | When did it happen? | 14:22:00 UTC | Offset present; not the end of shift |
| `recordTime` | When was it recorded? | 14:22:04 UTC | Small gap; a large gap signals batch capture |
| `action` | What changed? | `ADD` | `ADD`, `OBSERVE`, or `DELETE` only |
| `bizStep` | What was being done? | `packing` | A CBV business step, not a state |
| `disposition` | What state resulted? | `container_closed` | A CBV disposition, not a process step |
| `readPoint` | Where was it observed? | Plant GLN | The scan location |
| `bizLocation` | Where were the goods? | Plant GLN | May differ from the read point |
| `parentID` | What contains the children? | SSCC `...0000001` | An SSCC, not a GTIN |
| `childQuantityList` | What is inside, by class? | 60 cases of lot L-0412 | `uom` must be explicit |

A note on the vocabulary fields. CBV 2.0 defines both a short token form such as `packing` and a
URI form such as `https://ref.gs1.org/cbv/BizStep-packing`. The JSON-LD context maps the token
to the URI. Which form appears on the wire depends on the binding and the partner's profile, so
validate against the agreed profile rather than against your own example.

### 7.2 The three interfaces

| Interface | Representative operation | Purpose | Failure it invites |
|---|---|---|---|
| Capture | `POST /capture` (asynchronous), `POST /events` (synchronous) | Accept events into the repository | Duplicates on retry; partial batches |
| Query | `GET /events`, `POST /queries`, `GET /queries/{name}/events` | Retrieve events by filter | Pagination gaps; state assumed from events |
| Subscription | `POST /queries/{name}/subscriptions` | Push new results to a webhook or stream | Dropped deliveries; unsigned callbacks |
| Discovery | `OPTIONS` | Advertise limits and error behaviour | Limits discovered only at failure |

Two details matter. Asynchronous capture returns a job identifier, and the job can succeed
partially; the caller must read the error report, not only the status code. Subscriptions carry a
client-generated signature token, and the receiver verifies it on every callback. Without that
verification, the endpoint accepts events from anyone.

### 7.3 Three sanity checks on any repository

- Count events with `recordTime` more than one hour after `eventTime`. A high share means capture
  happens at end of shift, and every dwell-time metric is biased.
- Count children that appear in more than one open aggregation. Any result above zero means a
  missing `DELETE`, and the pallet content query is unreliable.
- Count the distinct `bizStep` values. A long tail of local strings means the CBV is not being
  enforced, and cross-partner queries will silently miss events.

## 8. Data, standards, and regulatory context

### 8.1 Identifier structure

| Identifier | Length | Composition | Check digit | Grain |
|---|---|---|---|---|
| GTIN-8/12/13/14 | 8, 12, 13, or 14 digits | company prefix, item reference, check digit | mod-10 | trade item type at a pack level |
| GLN | 13 digits | company prefix, location reference, check digit | mod-10 | party or location |
| SSCC | 18 digits | extension digit, company prefix, serial reference, check digit | mod-10 | one logistics unit |
| SGTIN | EPC URI or 96-bit tag | GTIN plus serial number | inherited from GTIN | one trade item instance |
| SGLN | EPC URI | GLN plus extension | inherited from GLN | location or sub-location |
| GIAI | EPC URI | company prefix plus individual asset reference | inherited | one asset instance |
| GRAI | EPC URI | company prefix, asset type, serial | inherited | one returnable asset |

The check digit algorithm is the same for all of them. Number the digits from the right,
excluding the check digit. Multiply alternately by 3 and 1, sum the products, and take the
difference from the next multiple of ten.

```text
sum      = d1 x 3 + d2 x 1 + d3 x 3 + ...   (from the right)
check    = (10 - (sum mod 10)) mod 10
```

Two structural notes. The SSCC extension digit is allocated by the party and exists to expand
the serial space; it does not encode pallet type, pack count, or plant. The GLN identifies a
party or a location, and the same thirteen-digit space is used for both; the registrant declares
which. Neither fact is obvious from the number, which is why identifier registries matter.

### 8.2 EPCIS 2.0 and CBV 2.0

EPCIS 2.0 was ratified in June 2022, alongside CBV 2.0. The version matters because it changed
the shape of the data and the shape of the interface.

| Change in 2.0 | What it means | Consequence for analytics |
|---|---|---|
| JSON and JSON-LD bindings | Events can be exchanged as JSON, not only XML | Easier ingestion; context handling becomes a design issue |
| REST bindings | Capture, query, and subscription as HTTP operations | One interface family instead of separate SOAP services |
| `AssociationEvent` added | Persistent non-containment relationships are representable | Installation and sensor relationships stop polluting aggregation |
| `QuantityEvent` removed | Class-level quantities are carried in `quantityList` on other events | Legacy feeds need mapping |
| Sensor dimension | `sensorElementList` adds the "how" | Cold-chain conditions can travel with the event |
| `persistentDisposition` | A disposition can be set and later unset | State changes such as `recalled` become representable |
| Removal of `SimpleMasterDataQuery` | Master data is not queried through EPCIS | Master data needs a separate source |

> **Note:** EPCIS 2.0 has no separate capture, query, and subscription specifications. All three
> interfaces are defined in one REST binding document, organised by tag. Products that market
> "EPCIS 2.2" or "EPCIS 3.0" are usually describing a vendor release, not a GS1 version.

### 8.3 The vocabulary is a separate standard

CBV defines the controlled values for `bizStep` and `disposition`. It is a separate standard with
its own version, and using it badly is the most common interoperability defect after identifier
errors.

| Business step (selection) | Meaning | Disposition (selection) | Meaning |
|---|---|---|---|
| `commissioning` | Object enters the supply chain | `active` | In use and not terminated |
| `inspecting` | Quality or condition check | `in_progress` | Being transformed or moved |
| `packing` | Objects placed into a logistics unit | `container_closed` | Logistics unit is sealed |
| `shipping` | Objects leave a location | `in_transit` | Moving between locations |
| `receiving` | Objects arrive at a location | `sellable_accessible` | Available for sale |
| `storing` | Objects placed in storage | `sellable_not_accessible` | Present but not available |
| `picking` | Objects selected for an order | `non_sellable_other` | Not sellable for another reason |
| `dispensing` | Object supplied to a patient or consumer | `recalled` | Subject to a recall |
| `destroying` | Object rendered unusable | `destroyed` | No longer exists |
| `sampling` | Sample taken (new in 2.0) | `damaged` | Physically damaged |
| `sensor_reporting` | Sensor reading captured (new in 2.0) | `expired` | Past its expiry date |

The full lists are longer, and the implementation guideline explains which combinations are
sensible. The rule that matters for a data engineer is simpler: validate `bizStep` and
`disposition` against the CBV release in force, and record the release version. A vocabulary that
drifts across partners is a query that returns three quarters of the answer.

### 8.4 Regulatory hooks

| Requirement | Jurisdiction | What it requires | How EPCIS relates | Status as of late 2026 | Grade |
|---|---|---|---|---|---|
| FSMA 204 Food Traceability Rule | United States | KDEs at critical tracking events, traceability lot codes, records producible within 24 hours, traceability plans | Can carry the KDEs; the rule does not mandate a message format | Original date 20 January 2026; proposed 30-month extension to 20 July 2028, and a 2026 appropriations act directed FDA not to enforce before that date | B |
| DSCSA | United States | Package-level product identifier, interoperable electronic exchange, verification | FDA's FAQ recommends GS1 EPCIS for interoperable exchange | Statutory date 27 November 2023; exemptions expired for manufacturers May 2025, wholesalers August 2025, large dispensers November 2025; small dispensers November 2026 | B |
| EU FMD | European Union | Unique identifier and anti-tampering device; verification and decommissioning against repositories | The repository is a verification system, not an EPCIS event log | In force since 9 February 2019, with Greece and Italy deferred | B |
| EU Digital Product Passport | European Union | A resolvable product identifier with lifecycle data | Identifier persistence matters; EPCIS is not the carrier | Phasing by product group | B |

Three distinctions in that table are worth stating plainly.

**FSMA 204 is technology-neutral.** It requires data elements, events, and records, not EPCIS.
A company can comply with spreadsheets and PDFs if it can produce the records within 24 hours.
EPCIS is a good way to comply because it standardises the event and the query, but a design that
assumes the regulation mandates it will over-engineer the wrong parts.

**DSCSA is the strongest driver.** It requires package-level serialised data and interoperable
exchange, and FDA's guidance names EPCIS as a recommended mechanism. This is why pharmaceutical
EPCIS adoption is years ahead of food adoption.

**EU FMD is not an event repository.** The national repositories verify and decommission unique
identifiers. They do not answer "where has this pack been". Aggregation of packs to cases was
not operationally ready across the system when the directive applied in 2019, and hospitals used
interim data files from suppliers. A company that assumes FMD repositories provide chain
traceability will be surprised by what a query returns.

### 8.5 Mapping regulatory data elements to event fields

| Required element | EPCIS carrier | Gap or caveat |
|---|---|---|
| Traceability lot code | lot identifier inside `epcClass`, or ILMD | Must be the code the chain agreed to use, not an internal alias |
| Product description | master data keyed by GTIN | Not carried in the event |
| Quantity and unit | `quantityList` with `uom` | Unit must be explicit and consistent |
| Location | `bizLocation`, `readPoint`, GLN | GLN or an agreed location identifier |
| Date of event | `eventTime` | Time zone and offset must be stated |
| Business name and contact | `sourceList` / `destinationList`, or master data | Party identity is often outside the event |
| Growing area, harvest date | ILMD | Commodity-specific; not a core EPCIS field |
| Transaction reference | `bizTransactionList` | Type must be stated, not only the value |

The pattern is consistent: EPCIS carries the event and the keys. It does not carry the
descriptive master data, and it does not enforce that the lot code you used is the one your
customer used. Both gaps are organisational, not technical.

### 8.6 What is convention rather than requirement

- **Serialising every case.** Required in some pharmaceutical contexts, optional in most food
  contexts, and expensive at both.
- **One repository per company.** The standard defines interfaces, not topology; regional
  repositories and a partner's repository are both common.
- **A GLN for every location.** Internal site codes remain common, with a hand-maintained mapping.
- **Retention and latency.** The retention period differs by regulation, and many compliant
  implementations capture in batches hours after the event.

## 9. Industry practice and case evidence

Public evidence about EPCIS is uneven, and it is concentrated in pharmaceuticals because DSCSA
created a legal driver. Food evidence is thinner, and much of it is vendor material. The
following table separates what is measured from what is claimed.

| Claim | Source and date | Grade | What it actually measures |
|---|---|---|---|
| 12 of 15 wholesale distributors received complete serialised data from 80% or more of suppliers by March 2025, compared with 1 of 26 in June 2024 | Partnership for DSCSA Governance stabilisation survey, April 2025 | C | A self-selected industry sample; the denominator changed between periods |
| 52% of companies report full adoption of GS1 standards; 53% can trace beyond one-up/one-down | GS1 US, Powering Supply Chain Confidence, March 2025, n=505 | C | GS1 standards generally, not EPCIS specifically |
| 68% of full GS1 standards adopters report effective traceability, against 48% of partial adopters | GS1 US, March 2025, n=505 | C | An association in a cross-sectional survey; causality is not established |
| 88% of manufacturers had transitioned to EPCIS 1.2 by Q3 2022; 51% of planned manufacturer-to-distributor connections were in process or complete | HDA EPCIS Implementation Benchmarking Survey, 2021–2022 baselines | C | Pharmaceutical manufacturers; older baseline |
| EPCIS can make FSMA 204 compliance easier | GS1 US, reported in trade press, May 2025 | D | A vendor and standards-body position, not a measurement |

The pattern is consistent with the regulatory drivers. Pharmaceutical manufacturers and
wholesalers built EPCIS capability because the law required serialised exchange and FDA named
the standard. Food manufacturers have a rule that requires data elements but not a format, so
adoption follows the path of least resistance: lot fields in EDI 856 notices, supplier portals,
and spreadsheets.

That path is not irrational. An EDI 856 with lot numbers satisfies many customer requirements at
a fraction of the cost of an EPCIS programme. The limitation appears one tier deeper: an 856
tells you what was shipped, not what was packed on a pallet. That difference is exactly the
aggregation event.

Two implementation lessons recur. First, aggregation is the hard part. The EU FMD repositories
were built for verification and decommissioning, and pack-to-case-to-pallet aggregation was not
ready across the system when the directive applied in February 2019; the Commission's hospital
expert group described interim supplier data files and extra checks. Verifying one identifier is
a database lookup; maintaining a containment graph is a discipline at every packing station.

Second, standards adoption is not interoperability. The GS1 US survey reports that 53% can trace
beyond one-up/one-down, which means roughly half of a sample already interested in GS1 standards
cannot reconstruct a multi-tier chain. GTINs on every label do not answer a recall question
without the events that link them.

### 9.1 What companies actually run

| Pattern | What it looks like | Where it works | Where it fails |
|---|---|---|---|
| EDI 856 with lot fields | Advance ship notice carries lot and quantity per line | Retail supplier compliance; shipment-level traceability | No case-to-pallet link; no receiving event from the customer |
| Partner portal upload | Supplier uploads a lot report to a customer portal | Customer-specific compliance | Different portal per customer; no common query |
| Central EPCIS repository | Events captured at packing and shipping, queried by the brand owner | Regulated sectors; own-distribution networks | Depends on partner capture at receipt |
| Repository-as-a-service | Third-party repository receives events from many suppliers | Supplier networks with heterogeneous capability | Data quality varies by supplier; query latency |
| Spreadsheet and PDF | Lot records held in files | Very small suppliers; low-risk products | Cannot meet a 24-hour multi-hop request at scale |

The design question is not "should we adopt EPCIS" but "which link must be queryable, and who
else has to capture the event that creates it". A brand owner can capture packing and shipping;
receiving belongs to the customer, and transformation to the manufacturer. A design that assumes
all three sit in one organisation describes a company that owns its whole chain, which is rare.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Pack-level GTIN collision | A case is received as a single each | Indicator digit ignored or reused across levels | Inventory overstated by the pack factor | Enforce GTIN-14 with indicator per level; validate at receipt |
| Missing disaggregation | A case appears on two pallets | `DELETE` not captured at break-down | Recall scope includes withdrawn pallets and misses new ones | Reconcile open links against physical pallets weekly |
| Events overwritten, not corrected | The audit trail has one row per state | Status-table thinking applied to an event store | History cannot be reconstructed; regulator cannot verify | Append-only store; correction events; quarantine the original |
| Naive timestamps | Dwell time is a round number of hours | Local time stored without offset | Cross-border and daylight-saving errors | ISO 8601 with offset; reject timestamps without one |
| Duplicate events on retry | Event counts exceed scan counts | No idempotency key; transport retries | Inflated volumes; double counting in aggregation | Deduplicate on `eventID`; capture idempotency |
| Free-text lot codes | The same lot has four spellings | Lot captured in a comment field | Recall query misses a production run | Controlled lot field; validate against production records |
| Transformation without inputs | Output lot has no provenance | Output captured, inputs omitted | Cannot trace an ingredient forward | Require at least one input and one output |
| Read point used as location | Goods appear at a gate for days | `bizLocation` omitted | Inventory position wrong; dwell time wrong | Capture both; default only with a documented rule |
| Disposition used as status | `in_transit` set and never unset | No later event clears it | Product looks in transit after delivery | Use `persistentDisposition` with explicit unset |
| Repository treated as master data | Descriptions missing from reports | Events carry keys, not attributes | Analyst cannot label a recall list | Join to a GTIN-keyed item master |
| Subscription assumed reliable | A gap appears in the control tower | Dropped webhook, expired signature | Missed receiving events; late alerts | Watermark plus replay query; signature verification |
| EPCIS assumed to equal compliance | A compliance plan lists "deploy EPCIS" | Standard adopted without process change | Records still incomplete at audit | Map each KDE to a field and a capture point |
| Serial numbers reused | Verification returns two products | Serial allocated per line, not per GTIN | Counterfeit signal; verification failure | Central serial allocation per GTIN |
| Check digit not validated | Bad labels discovered at the customer | No validation at print or ingest | Rejected shipments; manual rework | Validate at allocation, print, and ingest |

Two of these deserve expansion.

**Missing disaggregation** makes a traceability system look complete and behave incorrectly. A
clerk breaks a pallet, picks cases for three orders, and the warehouse system records the picks
but not the end of the pallet. Months later the recall query returns the original pallet at the
DC; the three store pallets are invisible. The fix is a `DELETE` at break-down and an `ADD` for
each new unit, at the station that prints the label.

**Events overwritten, not corrected** turns a compliance asset into a liability. A well-meaning
data-quality pass finds a duplicate or a wrong location and updates the row. The current state
becomes correct and the history becomes fiction. A correction is a new event, not an edit. If the
repository cannot accept one, the design is wrong.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Identifiers are unique and stable over the analysis window | Every join and every graph traversal depends on it | Reissued GLNs, renumbered GTINs, reused serials | Check for duplicate active identifiers and for re-use after retirement |
| Trading partners use the same GTIN for the same pack | The chain joins on the identifier | Private-label variants, pack changes, regional numbering | Sample-match partner item masters against your own |
| Events are captured at or near the time they occur | Dwell time and custody intervals are differences of `eventTime` | End-of-shift confirmation, batch uploads | Compare `recordTime` minus `eventTime` distribution |
| Clocks are synchronised across sites | Event ordering depends on it | Device clocks drift; time zones misconfigured | Monitor clock offset per device and per site |
| Aggregation reflects physical containment at the event time | The containment graph is the recall path | Missed break-down, repacking without capture | Reconcile open links to physical counts |
| The CBV vocabulary is used consistently | Cross-partner queries filter on these values | Local variants, deprecated values, profile extensions | Count distinct values; validate against the release |
| The repository retains events for the regulatory period | Recall and audit depend on history | Retention policy shorter than the obligation | Check retention settings against each regulation |
| A query can cross custodial boundaries | End-to-end traceability requires other parties' events | Partners do not share; repositories are siloed | Measure the share of shipments with a receiving event |
| Lot granularity matches the recall decision | The answer cannot be finer than the identifier | One lot spans multiple deliveries or lines | Compare lot size to the smallest holdable quantity |
| EPCIS covers the objects that matter | The graph is complete only for identified objects | Bulk materials, packaging, consumables not identified | List the inputs to each transformation and check coverage |
| An event is an assertion, not a verified fact | Quality rules must tolerate disagreement | Conflicting events from two parties | Count conflicts between shipping and receiving records |

Two limits of the standard are worth stating plainly.

**EPCIS is an exchange format, not a traceability system.** It defines the record and the capture,
query, and subscription interfaces. It does not guarantee truth, provide master data, resolve
identifiers across companies, compute the graph, enforce retention, or settle disagreement. Those
are system and governance responsibilities. Buying an "EPCIS-compliant" product buys a way to
exchange records; answering a recall depends on what you capture and how you query.

**The standard is open-world by design.** It does not require every party to report every event,
and cannot distinguish a missing event from an absent fact. That openness is why it works across
organisations, and why every traceability query must state how it handles absence.

## 12. Exercises

### Exercise 1 — Identifier grain sort (application, 30 minutes)

Take a product you can describe in detail and write its identifier stack: product type, instance
(if any), lot, logistics unit, plant, DC, and customer. For each level, state the identifier
scheme and the decision that the level supports.

**Deliverable:** a seven-row table with level, identifier, example value, and decision supported.
**Acceptance criteria:** at least one level where an instance identifier is unnecessary, with a
reason; at least one level where a lot identifier is insufficient, with a reason; no level where
two identifiers are used interchangeably.

### Exercise 2 — Check digit and pack levels (calculation, 30 minutes)

Compute the check digit for the GTIN-14 body `1061414110734`. Then explain what changes if the
same body is used with indicator digit `0` and with indicator digit `9`.

**Deliverable:** the arithmetic, the complete fourteen-digit identifier, and a short paragraph on
the two indicator values.
**Acceptance criteria:** weights applied from the right; sum and check digit shown; the answer
states that indicator 9 denotes variable measure and that changing the indicator changes the
identifier, not only its interpretation.

### Exercise 3 — Read an event (analysis, 35 minutes)

Take the aggregation event in section 7 and answer six questions: what is contained, in what
quantity and unit, in what parent, at what business location, during what business step, and in
what state. Then state one question the event cannot answer.

**Deliverable:** six answers with the field name used for each, plus the unanswerable question.
**Acceptance criteria:** every answer cites a field; the unanswerable question is genuinely outside
the event (for example, the product description or the physical count), not merely missing from
the example.

### Exercise 4 — Aggregation lifecycle (design, 45 minutes)

Draw the lifecycle of one case from packing to store delivery, marking every `ADD`, `DELETE`, and
`OBSERVE` event, and the interval during which each containment link is valid. Include one
break-down and one repacking.

**Deliverable:** a time-ordered diagram with event types, actions, and link lifetimes.
**Acceptance criteria:** no interval in which the case belongs to two parents; the `DELETE` precedes
the new `ADD`; the diagram distinguishes containment from movement.

### Exercise 5 — Recall scope calculation (calculation, 60 minutes)

Using the data in section 6, compute the hold quantity in pallets, cups, and currency for: the
shipped product, all product from the two lots, and the naive day scope. Then compute the
granularity cost if the true exposure is 8 pallets.

**Deliverable:** a table of the four scopes with arithmetic shown, and a one-paragraph
recommendation.
**Acceptance criteria:** units stated at every step; the difference between "in commerce" and
"must be held" explained; the recommendation names the cost of finer granularity as well as its
benefit.

### Exercise 6 — Event type selection (application, 40 minutes)

For each scenario, choose the event type and justify it in one sentence: a pallet loaded into a
container; a shipment linked to a purchase order; a sensor fixed in a cold room; milk consumed to
make cheese; a case inspected and released; a reusable crate assigned to a pool.

**Deliverable:** a six-row table with scenario, event type, and justification.
**Acceptance criteria:** no use of `AggregationEvent` for a non-containment relationship; no use of
`TransactionEvent` for physical contents; the association cases identified explicitly.

### Exercise 7 — Regulatory mapping (analysis, 50 minutes)

Map five FSMA 204 key data elements and three DSCSA product-identifier components onto EPCIS
fields. Mark each mapping as direct, indirect (via master data), or incomplete.

**Deliverable:** an eight-row mapping table with field, mapping type, and gap.
**Acceptance criteria:** at least one mapping marked incomplete, with the reason; the difference
between a lot code and a traceability lot code addressed; no claim that EPCIS is mandatory under
either rule.

### Exercise 8 — Repository interface design (design, 60 minutes)

Design the capture, query, and subscription flow for a brand owner receiving packing events from
three contract manufacturers. Specify idempotency, error handling, and gap recovery.

**Deliverable:** a one-page interface design with a sequence of operations and three failure
scenarios with responses.
**Acceptance criteria:** `eventID` used as the idempotency key; partial batch failure handled; the
subscription has signature verification and a replay path; the design states who owns master data.

### Exercise 9 — Failure audit (critique, 45 minutes)

Audit a traceability dataset you can obtain, or construct one from the section 6 example with
three deliberate defects. Identify at least five failure modes from section 10 and rank them by
recall risk.

**Deliverable:** a findings table with failure, evidence, consequence, and fix, ordered by risk.
**Acceptance criteria:** each finding cites a specific record or field; at least one finding is
about a missing event rather than a wrong value; the ranking distinguishes over-scope from
under-scope risk.

## 13. Self-check answers

**Exercise 2 arithmetic.**

The body is `1061414110734`, thirteen digits. Number from the right, excluding the check digit,
and weight alternately 3 and 1:

```text
digit        4   3   7   0   1   1   4   1   4   1   6   0   1
weight       3   1   3   1   3   1   3   1   3   1   3   1   3
product     12   3  21   0   3   1  12   1  12   1  18   0   3

sum = 12 + 3 + 21 + 0 + 3 + 1 + 12 + 1 + 12 + 1 + 18 + 0 + 3 = 87
check = (10 - (87 mod 10)) mod 10 = (10 - 7) mod 10 = 3
GTIN-14 = 10614141107343
```

Indicator `0` produces a different number and denotes the base unit in common practice; indicator
`9` denotes a variable-measure item, whose actual weight travels in an Application Identifier on
the label rather than in the item master. The important point is that changing the indicator
changes the identifier. It is not a formatting choice.

**Exercise 3 model answers.**

What: 60 cases of lot L-2026-0412, from `childQuantityList`. Quantity and unit: 60, `CS`. Parent:
SSCC `...0000001`, from `parentID`. Business location: the plant GLN, from `bizLocation`. Business
step: `packing`, from `bizStep`. State: `container_closed`, from `disposition`. The event cannot
answer what the product is called, what it weighs, or whether the pallet is still intact today.
The first two are master data; the third requires later events.

**Exercise 5 arithmetic.**

```text
value per pallet = 1,440 cups x EUR 0.42 = EUR 604.80

shipped scope:
  pallets = 9
  cups    = 9 x 1,440 = 12,960
  value   = 9 x 604.80 = EUR 5,443.20

all lots:
  pallets = 11
  cups    = 11 x 1,440 = 15,840
  value   = 11 x 604.80 = EUR 6,652.80

naive day scope:
  pallets = 42
  cups    = 42 x 1,440 = 60,480
  value   = 42 x 604.80 = EUR 25,401.60

over-scope versus shipped = 42 / 9 = 4.67 times
saving from event data     = 25,401.60 - 5,443.20 = EUR 19,958.40

granularity cost:
  ideal scope = 8 pallets
  supported   = 11 pallets
  difference  = 3 x 604.80 = EUR 1,814.40
```

The distinction between shipped and held matters. A recall holds product wherever it is,
including the two pallets still at the plant. The customer-facing question is what is in
commerce; the operational question is what must be secured. Reporting only the first understates
the work; reporting only the second overstates the public exposure.

**Exercise 6 answers.**

- Pallet into container: `AggregationEvent`. The container is a logistics unit and the pallet is
  physically inside it.
- Shipment linked to a purchase order: `TransactionEvent`. The objects are associated with a
  commercial document, not contained in it.
- Sensor fixed in a cold room: `AssociationEvent`. The relationship is persistent, not
  containment, and the parent may be a location.
- Milk consumed to make cheese: `TransformationEvent`. Inputs are consumed and outputs produced.
- Case inspected and released: `ObjectEvent`, with `bizStep=inspecting` and a disposition such as
  `conformant` or `non_conformant`.
- Reusable crate assigned to a pool: `AssociationEvent` if the pool is an organisation or
  location; `AggregationEvent` only if the crate is physically inside a specific logistics unit.

**Exercise 4 rubric.**

A strong answer draws a time axis with the case as a continuous line and the containment link as
a bracket that opens at `ADD` and closes at `DELETE`, with the break-down as a `DELETE` before
the new `ADD`s. It separates movement from containment. A weak answer draws a process flowchart,
which loses the lifetimes.

**Exercise 7 rubric.**

A strong answer maps the lot code to `epcClass` or ILMD, the description to master data, quantity
to `quantityList` with a unit, location to a GLN, and date to `eventTime`. It marks description
and business contact as indirect, and commodity-specific elements as incomplete without ILMD,
and states that FSMA 204 does not mandate EPCIS. A weak answer marks every mapping direct.

**Exercise 8 rubric.**

A strong design uses `eventID` as the idempotency key, reads the error report after a partial
asynchronous capture and retries only rejected events, and keeps a watermark per supplier. It
verifies subscription signatures and replays the gap after a failure, and names the owner of the
GTIN master. A weak design treats a 202 as success and assumes delivery.

**Exercise 9 rubric.**

A strong audit finds a missing `DELETE`, a duplicate, and a vocabulary defect, and shows the
recall consequence of each. It ranks missing disaggregation above duplication, because the first
changes scope. A weak audit lists defects without connecting them to a decision.

## 14. Interview questions

**1. "What is the difference between a GTIN, an SGTIN, and an LGTIN?"**

A strong answer separates product type, product instance, and lot, and gives a use case for each:
a retail label, a pharmaceutical pack, a food production lot. A weak answer treats them as
versions of the same number and cannot say when serialisation is worth its cost.

**2. "Why is an aggregation a relationship rather than an attribute?"**

A strong answer explains the `ADD` and `DELETE` lifetime, notes that a case can move between
pallets, and says that an attribute overwrites the history a recall depends on. A weak answer
says "because EPCIS models it that way", which is a description of the standard rather than a
reason for it.

**3. "When would you use an AssociationEvent instead of an AggregationEvent?"**

A strong answer gives a persistent, non-containment relationship such as an installed component
or a fixed sensor, and notes that the parent may be a location. A weak answer says they are
interchangeable, or uses association for pallet packing.

**4. "What is the difference between `bizStep` and `disposition`?"**

A strong answer defines the process step and the resulting state, and gives a pair: `shipping`
with `in_transit`. A weak answer treats both as status fields and cannot explain why
`in_transit` is not a business step.

**5. "Why record both `readPoint` and `bizLocation`?"**

A strong answer separates the observation device location from the business location of the
goods, and explains the inventory and dwell-time errors that follow from conflating them. A weak
answer says they are usually the same, which is true and not a reason to omit one.

**6. "What does `recordTime` tell you that `eventTime` does not?"**

A strong answer describes capture latency, and uses the gap to diagnose end-of-shift scanning
and batch uploads. A weak answer says it is when the system saved the event, without connecting
it to a measurement.

**7. "How would you answer a recall query with EPCIS?"**

A strong answer describes a traversal: ingredient lot to output lot, lot to pallet, pallet to
destination, and disaggregation to store pallets, noting the recursion and the missing-event
ambiguity. A weak answer filters one lot field, which returns one hop.

**8. "What is the difference between EPCIS and a traceability system?"**

A strong answer says EPCIS is an exchange format and interface standard, while a traceability
system adds capture discipline, master data, graph computation, retention, and governance. A weak
answer treats "EPCIS-compliant" as "can trace".

**9. "FSMA 204 does not mandate EPCIS. What does that mean for your design?"**

A strong answer says the rule requires data elements, critical tracking events, and records
within 24 hours, and that EPCIS is one choice among several. It justifies EPCIS on query
capability, not compliance. A weak answer scopes the programme around message conformance.

**10. "A receiving event is missing. What do you conclude?"**

A strong answer says nothing conclusive: the partner may not have captured it, may have sent it
elsewhere, or the goods may not have arrived. It states the evidence needed to distinguish the
cases. A weak answer treats the missing event as proof of non-delivery.

**11. "How would you measure data quality in an EPCIS repository?"**

A strong answer names capture latency, duplicate rate, open aggregation links against physical
pallets, vocabulary conformance, and the share of shipments with a receiving event. A weak
answer says "completeness and accuracy" without defining either.

**12. "Why does the EU FMD experience matter for a food traceability programme?"**

A strong answer draws the lesson that verifying one identifier is a lookup while maintaining a
containment graph is an operational discipline, and notes that FMD aggregation was not ready at
launch. A weak answer treats FMD as a ready template.

## 15. Key terms

- **GTIN** — Global Trade Item Number; identifies a trade item at a pack level.
- **Indicator digit** — the first digit of a GTIN-14 that distinguishes packaging levels.
- **GLN** — Global Location Number; identifies a party or a location.
- **SSCC** — Serial Shipping Container Code; identifies one logistics unit.
- **SGTIN** — a GTIN plus a serial number, identifying one instance.
- **LGTIN** — a GTIN plus a lot, identifying one batch.
- **SGLN** — a GLN plus an extension, identifying a sub-location.
- **GIAI** — Global Individual Asset Identifier, for one asset instance.
- **GRAI** — Global Returnable Asset Identifier, for a returnable asset type and serial.
- **EPC** — Electronic Product Code; an identifier for a physical object.
- **EPCIS** — the GS1 standard for event data about objects in a supply chain.
- **CBV** — Core Business Vocabulary; the controlled values for business steps and dispositions.
- **ObjectEvent** — an event about objects, without containment semantics.
- **AggregationEvent** — an event recording physical containment of children in a parent.
- **TransactionEvent** — an event associating objects with a business transaction.
- **TransformationEvent** — an event recording inputs consumed and outputs produced.
- **AssociationEvent** — an event recording a persistent, non-containment association.
- **`action`** — the change asserted by an event: `ADD`, `OBSERVE`, or `DELETE`.
- **`bizStep`** — the business process step during which an event occurred.
- **`disposition`** — the business state of the objects after an event.
- **`readPoint`** — the place where an event was observed.
- **`bizLocation`** — the business location of the objects at the event time.
- **ILMD** — instance or lot master data attached to a lot or instance.
- **Aggregation** — physically grouping children into a parent; **disaggregation** ends the link.
- **Repository** — the service that stores and serves EPCIS events.
- **KDE** — Key Data Element required under FSMA 204.
- **CTE** — Critical Tracking Event at which KDEs must be captured.
- **TLC** — Traceability Lot Code, the lot identifier agreed across a food chain.

## 16. Sources

**Standards and official (Grade B unless noted)**

- GS1, *EPCIS Standard, Release 2.0*, ratified June 2022. `https://ref.gs1.org/standards/epcis/2.0.0/`
- GS1, *Core Business Vocabulary Standard, Release 2.0*, ratified June 2022.
  `https://ref.gs1.org/standards/cbv/2.0.0/`
- GS1, *EPCIS and CBV Implementation Guideline, Release 2.0*, March 2023.
  `https://ref.gs1.org/guidelines/epcis-cbv/2.0.0/`
- GS1, *EPCIS 2.0 REST Bindings*, OpenAPI 3.0.3 specification, `github.com/gs1/EPCIS`,
  `REST Bindings/openapi.yaml`.
- GS1, *GS1 Digital Link Standard: URI Syntax*, Release 1.7.0, August 2026.
  `https://ref.gs1.org/standards/digital-link/uri-syntax/1.7.0/`
- GS1, *GS1 Digital Link* overview. `https://www.gs1.org/standards/gs1-digital-link`
- ISO/IEC 19988:2024, *GS1 Core Business Vocabulary (CBV)*, third edition. The ISO adoption of
  CBV 2.0.
- U.S. Food and Drug Administration, *FSMA Final Rule on Requirements for Additional Traceability
  Records for Certain Foods*, 21 CFR Part 1 Subpart S.
- U.S. Food and Drug Administration, *Frequently Asked Questions: FSMA Food Traceability Rule*,
  including the compliance-date status. `https://www.fda.gov/food/food-safety-modernization-act-fsma/frequently-asked-questions-fsma-food-traceability-rule`
- U.S. Federal Register, *Requirements for Additional Traceability Records for Certain Foods:
  Compliance Date Extension*, proposed rule, 90 FR 38084, 7 August 2025.
  `https://www.federalregister.gov/documents/2025/08/07/2025-14967/`
- U.S. Food and Drug Administration, *DSCSA Product Tracing Requirements: Frequently Asked
  Questions*, which recommends GS1 EPCIS for interoperable exchange.
- U.S. Food and Drug Administration, *DSCSA Exemptions from Section 582(g)(1) and Other
  Requirements of the FD&C Act for Certain Trading Partners*, 9 October 2024.
- U.S. Food and Drug Administration, *Enhanced Drug Distribution Security Requirements Under
  Section 582(g)(1) of the FD&C Act — Compliance Policies*, August 2023.
- European Union, *Commission Delegated Regulation (EU) 2016/161* supplementing Directive
  2001/83/EC, safety features for medicinal products for human use. CELEX 32016R0161.
- European Medicines Agency, "New safety features for medicines sold in the EU," 8 February 2019.
- European Commission expert group on the hospital setting, paper on implementation of the FMD,
  25 September 2018.

**Industry and analyst (Grade C or D as marked)**

- GS1 US, *Powering Supply Chain Confidence Research Report*, March 2025, n=505, fielded August
  2024. Industry survey; Grade C.
- Partnership for DSCSA Governance, *2025 Stabilization Survey Results*, April 2025. Industry
  consortium survey; Grade C.
- Healthcare Distribution Alliance, *EPCIS Implementation Benchmarking Survey*, 2021–2022
  baselines. Industry benchmarking; Grade C.
- GS1 US, position that EPCIS can support FSMA 204 compliance, reported by *Supermarket
  Perimeter*, May 2025. Trade press reporting a standards-body claim; Grade D/E.

**Note on citations.** Survey figures are reported as the surveys state them and are not
independently verified. The PDG comparison between March 2025 and June 2024 uses different
denominators (15 and 26 respondents), so it describes a change in a self-selected panel rather
than a measured population trend. Regulatory dates are stated as of late 2026 and should be
re-checked against the current Federal Register and EUR-Lex texts before use in a compliance
plan.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_edi-x12-edifact.md`](08_edi-x12-edifact.md)
- Next topic: `10_apis-contracts-mdm.md` (planned, not yet written)
- Quality, quarantine, and lineage: `11_quality-quarantine-lineage.md` (planned, not yet written)
- Related in this module: [`04_sql-supply-chain-patterns.md`](04_sql-supply-chain-patterns.md)
- Related in this module: [`07_orchestration.md`](07_orchestration.md)
- Identifier foundations: [`../01_supply_chain_foundations/03_network-entities.md`](../01_supply_chain_foundations/03_network-entities.md)
- Sensor and vision data: [`../02_supply_chain_data_landscape/05_telemetry-iot-vision.md`](../02_supply_chain_data_landscape/05_telemetry-iot-vision.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Return to the part map: [`../_index.md`](../_index.md)
