# Data Movement: EDI, API, Flat File, Event

> Module 02 — The Supply Chain Data Landscape
> Topic 06 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02.01, 02.02, 02.03, 02.04, 02.05

---

## 1. Why this topic exists

An order is not a fact. It is a message that survived a transport.

The same commercial event — a retailer commits to buy 480 cases of a product — can reach your
systems by four structurally different routes. It can arrive as a nightly CSV file, as an ANSI
X12 850 purchase order, as a JSON response to an HTTPS request, or as an `order.created` event on
a message bus. The business meaning is identical. The data you receive is not.

Each transport makes a different bargain. A batch file gives you completeness and a frozen layout,
and charges you in hours of latency. An application programming interface (API) gives you
sub-second freshness at the source, and charges you in partial responses, pagination loops,
rate limits, and schema that changes without asking. Electronic data interchange (EDI) gives you
a standardised, contract-grade document, and charges you in translation maps, trading-partner
variations, and days of batching. An event stream gives you sub-second delivery and per-key
ordering, and charges you in duplicates, gaps, and late arrivals.

An analyst who does not know which bargain was struck will read the transport's artefacts as
business facts. Three examples show the shape of the error:

1. **The overnight feed that everybody calls real-time.** A dashboard refreshes every five
   minutes. The underlying inventory position arrives once per day at 02:00. The refresh rate
   measures the dashboard, not the data. A replenishment model tuned to the refresh rate will
   over-order, because it believes it is reacting to a change it will not see for 22 hours.
2. **The API that returns 100 of 4,000 rows.** An analyst builds a supplier scorecard from a REST
   endpoint that paginates at 100 records. The first page is fetched, the loop is not written, and
   the scorecard covers the 100 most recent purchase orders. Every supplier with low volume
   disappears from the population. The number is correct for the wrong denominator.
3. **The webhook that fires twice.** A receipt event is delivered, the consumer times out before
   acknowledging, and the producer retries. The warehouse receives the same pallet twice in the
   data and once in reality. Inventory is overstated by one pallet per retry, and the error
   compounds silently across every downstream aggregate.

None of these is a modelling failure. Each is a failure to model the transport. Module 02 has so
far treated data as a landscape of sources: systems of record (Topic 01), master data (Topic 02),
transaction data (Topic 03), reference data (Topic 04), and telemetry (Topic 05). This topic adds
the dimension those sources share and rarely document: the path the data travelled, and what that
path guarantees.

The distinction matters because transport properties are not visible in the schema. A column
called `order_date` looks the same whether it arrived nine hours late or nine seconds late. The
table does not tell you that the 997 acknowledgment confirmed syntax and not acceptance. It does
not tell you that page 3 of 40 was never fetched. Transport guarantees live in the integration
layer, in trading-partner agreements, and in operational memory. Your job is to bring them into
the data model explicitly, because the model will be used by people who assume the data is
complete, current, and stable.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** any supply chain data feed into one of the four transports and state the
   completeness, latency, ordering, and schema-stability guarantee it structurally provides.
2. **Construct** a latency budget from business event to dashboard, with each stage expressed in
   hours, and identify which stage dominates.
3. **Trace** an EDI document from the buyer's application through the translator, the network, and
   the partner's translator, and explain what a 997 acknowledgment does and does not certify.
4. **Diagnose** an API integration for truncation, rate-limit collapse, eventual consistency, and
   schema drift, and state the reconciliation control that would detect each.
5. **Specify** a data contract with schema, semantics, freshness service-level agreement (SLA),
   quality thresholds, ownership, and a compatibility policy, and defend why it is a governance
   artefact rather than a document.
6. **Reconcile** the same business event received by multiple transports, identifying which
   transport is authoritative for which field and where double counting arises.
7. **Design** a deduplication and ordering strategy for an at-least-once event feed, including a
   deduplication key, a late-arrival policy, and a gap-detection method.
8. **Evaluate** a proposed transport choice against the decision it must support, and reject a
   choice whose latency or completeness cannot meet that decision's requirement.

## 3. The core mental model

Hold one picture: four lanes converging on one landing zone, each lane with a different clock,
a different completeness rule, and a different failure signature.

```text
   PARTNER SIDE                    TRANSPORT                   YOUR SIDE
   ------------                    ---------                   ---------

   ERP / OMS -- nightly extract --> [ flat file ] -- SFTP ----+
                                                              |
   ERP / OMS -- X12 850 batch ----> [ EDI VAN ] -- AS2 -------+
                                                              |
   OMS / WMS -- HTTPS request ----> [ REST API ] -- HTTPS ----+--> +------------+
                                                              |    |  landing   |
   WMS / TMS -- publish ----------> [ event bus ] -- webhook -+    |  zone      |
                                                                   +-----+------+
                                                                         |
                                            validate -> stage -> warehouse load
                                                                         |
                                                                   semantic layer
                                                                         |
                                                                  dashboard / model
```

The landing zone is deliberately drawn as one box. In most companies it is several: an SFTP
server, a message queue, an API ingestion job, and an EDI translator's output directory. The
analyst usually sees only what comes out of the far end, after all four have been normalised into
one table. That normalisation is exactly where the transport guarantees are lost.

### 3.1 The four bargains

| Transport | Unit of exchange | Completeness guarantee | Latency at source | Ordering guarantee | Schema stability | Characteristic failure |
|---|---|---|---|---|---|---|
| Batch flat file | One file per cycle | High within the file; unknown between files | Hours to 1 day | File order only | High: layout frozen by agreement | Silent partial file, encoding, trailer rows |
| EDI | One transaction set per document | High per document; gaps visible only via control numbers | Hours to days | Per interchange, not global | High but partner-specific | Translation-map drift, 997 mistaken for acceptance |
| API | One response page | Partial until the pagination loop completes | Seconds to minutes | None across requests | Low: continuous deployment | Truncation, rate limiting, field drift |
| Event stream | One event | Complete only if the consumer handles gaps | Sub-second | Per partition key, best-effort | Medium: versioned event types | Duplicates, late events, reordering |

Read the table as four trades, not four technologies:

- **Batch buys completeness with time.** The file is a closed set. Everything the extract saw is
  in it. But you cannot have it until the schedule says so.
- **EDI buys standardisation with time and variation.** The document is contract-grade and
  syntactically governed. But it is batched, and each trading partner's implementation guide
  narrows the standard differently.
- **API buys time with completeness and stability.** The response is fast and current, but it is a
  page, not a population, and the provider may change a field in a deploy you were not told about.
- **Events buy time with completeness.** Delivery is fast and near-continuous, but exactly-once
  delivery is not free, ordering is per key at best, and any consumer that does not track gaps
  will believe a partial stream is a complete one.

### 3.2 The arrival envelope

Every transport can be described by five timestamps. Learn to ask for all five.

```text
  event        app         network       pipeline        human
  occurs       commits     delivers      lands           decides
    |            |            |             |              |
    t0        t0+La       t0+La+Ln      t0+La+Ln+Lp    t0+La+Ln+Lp+Ld
    |            |            |             |              |
    +-- app lag -+-- network -+-- pipeline -+-- decision --+
    |                                                      |
    |<----------------- end-to-end data latency ----------->|
```

- **Event time (t0).** When the business event happened: the order was accepted, the pallet was
  scanned, the invoice was issued.
- **Application lag (La).** How long the source system took to commit and expose it. A warehouse
  management system (WMS) that confirms at end of shift can contribute eight hours here.
- **Network lag (Ln).** Transport time. Almost always the smallest term, and almost always the
  term vendors talk about.
- **Pipeline lag (Lp).** Landing, validation, staging, warehouse load, transformation, semantic
  refresh. In most enterprises this is the dominant term, measured in hours, not seconds.
- **Decision lag (Ld).** How long the human takes to act. Often the largest term of all, and the
  one analytics most often ignores.

The single most useful diagnostic in this topic: **if a vendor quotes you a network number, you
have been told the smallest part of the answer.** A "real-time API" with a two-hour warehouse
refresh is a two-hour feed. A "sub-second event stream" feeding a nightly dbt run is a nightly
feed.

### 3.3 Three properties worth memorising

**Property 1 — Completeness is a property of the consumer, not the producer.** Every transport can
deliver everything and you can still lose it. A file can arrive with a truncated final line. An API
loop can stop at page 1. An event consumer can drop a partition. The producer's guarantee is
necessary but not sufficient; the consumer must have a reconciliation control that proves it got
the whole population. The control differs by transport: record counts and control totals for
files, interchange control numbers and 997s for EDI, cursors and watermarks for APIs, offsets and
event identifiers for streams.

**Property 2 — Every transport has a reconciliation unit.** Batch has the file and the record
count. EDI has the interchange control number, the group control number, and the transaction-set
control number. API has the cursor and the total count, when the provider supplies one. Events
have the partition offset and the event identifier. If you cannot name the reconciliation unit for
a feed, you cannot prove the feed is complete, and you should treat it as a sample.

**Property 3 — Schema stability is a governance outcome.** A batch layout is stable because two
companies signed an implementation guide and changing it requires a project. An API schema drifts
because one company deploys on a Tuesday. An event schema drifts because a producer team added a
field and considered it backward-compatible. None of these is a technical property of the
protocol. Each is a consequence of who bears the cost of change. That is why the data contract in
section 5.5 is the real control, not the format.

### 3.4 Where the EDI path actually runs

EDI is the transport most often described as a black box. It is not. It is a chain of five
translations, and errors concentrate at the seams.

```text
   BUYER                                              SELLER
   -----                                              ------
   order in ERP
       |
       v
   [ translator ] --(X12 850)--> [ VAN / AS2 ] --> [ translator ] --> order in ERP
       ^                                                                    |
       |                                                                    v
   [ 997 received ] <----(X12 997)---- [ VAN / AS2 ] <----(X12 997)--- [ 997 generated ]
```

The buyer's enterprise resource planning (ERP) system produces an internal representation; a
translator maps it to X12; a value-added network (VAN) or AS2 connection carries it; the seller's
translator maps it back. Only the middle steps are standardised, and the maps at each end are where
the failures live.

### 3.5 The latency ladder

A useful simplification for planning conversations. Each rung is a typical order of magnitude, not
a promise.

| Rung | Typical end-to-end latency | Transport that reaches it | Decision it can support |
|---|---|---|---|
| Batch daily | 12–30 hours | Nightly flat file, daily EDI batch | Replenishment planning, weekly S&OP |
| Batch intraday | 2–8 hours | EDI every 15–60 minutes, hourly extracts | Same-day allocation, exception review |
| Near-real-time | 15 minutes–2 hours | API polling, change data capture | Inventory visibility, order promising |
| Real-time | 1–60 seconds | Event stream, webhook | Fraud checks, dynamic routing, alerts |
| Sub-second | < 1 second | Event stream with in-memory serving | Bidding, robotics, high-frequency control |

Most supply chain decisions live on the top two rungs. Very few need the bottom one, and buying it
without staffing it produces a system that is fast and unused.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Transport | The mechanism by which data moves between systems | Used to mean the data format | Protocol name |
| Batch | Processing a closed set of records at a scheduled time | Used to mean "slow" | Cycle frequency, for example 1 per day |
| Stream | Processing records continuously as they arrive | Used to mean "complete" | Events per second |
| EDI | Electronic data interchange: structured business documents exchanged between trading partners | Used to mean any integration | ANSI X12 or UN/EDIFACT |
| Transaction set | One business document within an X12 interchange, such as a purchase order | Confused with a file | Three-digit code, for example 850 |
| ANSI X12 | The North American EDI standard maintained by the Accredited Standards Committee X12 | Treated as global | Version, for example 005010 |
| UN/EDIFACT | The United Nations EDI standard used outside North America | Treated as interchangeable with X12 | Message type, for example ORDERS |
| 850 | Purchase order | Called "the order file" | X12 transaction set 850 |
| 855 | Purchase order acknowledgment | Assumed to be automatic | X12 transaction set 855 |
| 856 | Advance ship notice (ASN), also called ship notice/manifest | Treated as proof of delivery | X12 transaction set 856 |
| 810 | Invoice | Treated as proof of payment | X12 transaction set 810 |
| 846 | Inventory inquiry/advice | Treated as a real-time position | X12 transaction set 846 |
| 997 | Functional acknowledgment: confirms receipt and syntactic processing of an EDI document | Mistaken for business acceptance | X12 transaction set 997 |
| ASN | Advance ship notice; the 856 document describing what is in a shipment | Treated as a tracking number | X12 856 or EDIFACT DESADV |
| AS2 | Applicability Statement 2: signed, encrypted HTTP transport for EDI with a receipt | Confused with the EDI format itself | RFC 4130 |
| VAN | Value-added network: a third-party EDI mailbox and routing service | Assumed to add validation | Provider name |
| Translation map | The partner-specific rules converting internal data to and from an EDI standard | Treated as part of the standard | Map version |
| SFTP | SSH File Transfer Protocol: file transfer over an encrypted channel | Called "FTP" | Protocol version |
| Flat file | A single file of records with a fixed or delimited structure | Assumed to be CSV | Format name |
| Delimiter | The character separating fields or records | Assumed to be a comma | Character or byte |
| Header row | The first row naming the columns | Assumed always present | Row number |
| Encoding | The character set and byte representation of text | Ignored until it breaks | UTF-8, Windows-1252, ISO-8859-1 |
| Byte-order mark (BOM) | An invisible marker at the start of some UTF-8 files | Mistaken for part of the first column name | Bytes EF BB BF |
| API | Application programming interface: a defined contract for one system to call another | Used to mean "real-time" | Endpoint and version |
| Pagination | Splitting a large result into pages | Ignored in first implementations | Page size and cursor |
| Rate limit | A cap on requests per unit time | Discovered in production | Requests per second or day |
| Backfill | Loading historical data that was not captured incrementally | Confused with a refresh | Date range and record count |
| Eventual consistency | A state where a write is visible to readers after an unspecified delay | Treated as a bug | Seconds of lag |
| Idempotency | The property that repeating an operation has the same effect as doing it once | Assumed by default | Boolean property |
| Deduplication key | The field or combination that identifies a unique event or record | Chosen as the arrival timestamp | Field or composite key |
| At-least-once | A delivery guarantee that every message arrives, possibly more than once | Mistaken for exactly-once | Delivery semantic |
| Exactly-once | A delivery guarantee that each message is processed once, usually within one system boundary | Overclaimed across systems | Delivery semantic |
| Webhook | An HTTP callback the producer makes to a consumer URL when an event occurs | Treated as guaranteed delivery | URL plus secret |
| Event time | When the business event occurred | Confused with arrival time | Timestamp with time zone |
| Late-arriving event | An event whose event time is earlier than data already processed | Dropped silently | Lateness in seconds |
| Schema | The structure, types, and constraints of a dataset | Treated as the whole contract | Schema version |
| Schema drift | A change to structure or meaning that consumers did not expect | Detected only when a job fails | Change type |
| Data contract | A versioned agreement covering schema, semantics, freshness, quality, ownership, and change policy | Treated as documentation | Contract version |
| Freshness SLA | A commitment on how old data may be when available | Confused with refresh frequency | Hours, usually as P50 and P90 |
| Latency budget | The allocation of end-to-end delay across stages | Confused with a single number | Hours per stage |
| EPCIS | Electronic Product Code Information Services: the GS1 standard for event data about objects | Treated as an EDI replacement for all documents | GS1 EPCIS 2.0 |
| Peppol | The pan-European eDelivery network and document specifications | Treated as a software product | Network and BIS profile |
| EN 16931 | The European semantic data model for a core electronic invoice | Confused with a syntax | Business terms and rules |
| Snapshot | A full state at a point in time | Confused with a delta | As-of timestamp |
| Delta | The set of changes since a previous point | Confused with a snapshot | Change window |
| Reconciliation | The control that proves two populations or two transports agree | Treated as a monthly finance task | Record count or value |

Four of these cause more damage than the rest.

**997 versus business acceptance.** A 997 confirms that the receiver's translator parsed the
interchange and the transaction set, and it reports syntax errors at the segment and element
level. It does not confirm that the receiver's ERP accepted the order, priced it correctly, or
will ship it. The business acceptance arrives in an 855, and even the 855 is not a shipment. An
organisation that treats 997 as acceptance will believe an order is live when it is sitting in a
partner's error queue.

**At-least-once versus exactly-once.** At-least-once means duplicates are guaranteed, not
possible. Exactly-once is achievable inside a single system that controls its own storage and
offsets, such as a stream processor with transactional writes. It is not achievable end-to-end
across an HTTP callback into a third-party system without idempotency at the destination. When a
vendor claims exactly-once delivery to your application, ask what happens on a network timeout
after the write succeeded.

**Snapshot versus delta.** A snapshot says "here is the state." A delta says "here is what
changed." Loading a snapshot as if it were a delta double counts. Loading a delta as if it were a
snapshot loses history and produces gaps. Both errors are common, and both are invisible in a row
count.

**Event time versus processing time.** Ordering by arrival time is the default in most ingestion
code, and it is wrong whenever a late event arrives. An event that occurred at 09:00 but arrived
at 11:00 will be placed after events that occurred at 10:00. Every windowed aggregate computed on
arrival time is subtly wrong, and no schema will tell you.

## 5. The mechanics, step by step

This section walks each transport from source to warehouse. For each step, note three things: what
happens, where it breaks, and what the analyst sees when it breaks.

### 5.1 The batch flat file path

**Step 1. Extract.** The source system runs a report or export at a scheduled time, usually
overnight and usually against a read replica.
*Breaks when:* the extract runs before the day's final transactions post, so the file is a
consistent snapshot of the wrong moment.
*Analyst sees:* a day that is systematically short by the volume posted after the cut-off.

**Step 2. Format.** The extract writes a delimited or fixed-width file with a chosen encoding,
delimiter, decimal separator, and date format.
*Breaks when:* the format is locale-dependent. A European locale may write `1.234,56` where the
consumer expects `1234.56`, and a semicolon where the consumer expects a comma.
*Analyst sees:* numeric columns that parse as text, or a single-column table.

**Step 3. Land.** The file is written to a directory on an SFTP server, often with a naming
convention containing the date and a sequence number.
*Breaks when:* the file is written directly to the pickup directory and the consumer reads it
mid-write.
*Analyst sees:* a truncated final line, or a row count that is short by an arbitrary amount.

**Step 4. Signal readiness.** A well-built integration writes to a temporary name and renames
atomically, or writes a `.done` sentinel file.
*Breaks when:* the convention is undocumented and differs by trading partner.
*Analyst sees:* intermittent partial loads that nobody can reproduce.

**Step 5. Pickup.** The consumer polls on a schedule, often every 15 to 60 minutes, and moves the
file to an archive.
*Breaks when:* the poll runs before the file lands, and the next poll is an hour later, or the
next day on a weekend.
*Analyst sees:* a missing day, or a day that arrives 24 hours late with no error.

**Step 6. Parse.** The consumer splits records and fields, handles quoting, and decodes the
encoding.
*Breaks when:* a quoted field contains the delimiter or a newline, or the file carries a UTF-8
BOM that becomes part of the first column name.
*Analyst sees:* column names like `﻿order_id`, or rows shifted by one field.

**Step 7. Validate.** Row counts, control totals, mandatory fields, and referential checks run
against the file.
*Breaks when:* validation checks only that the file is non-empty, not that it contains the
expected population.
*Analyst sees:* a load that succeeds with 40% of the expected rows.

### 5.2 The EDI path

**Step 1. Business document created.** The buyer's ERP creates a purchase order with lines,
quantities, units of measure, prices, and requested dates.
*Breaks when:* the order contains a unit of measure or item identifier the partner's map does not
recognise.
*Analyst sees:* an order that exists internally and never appears at the partner.

**Step 2. Translation.** A translator maps internal fields to the X12 or EDIFACT structure using a
partner-specific map.
*Breaks when:* the map is stale relative to a partner's updated implementation guide, or a
qualifier is hard-coded for the wrong context.
*Analyst sees:* a document that passes your validation and fails theirs, usually with an error
code pointing at a segment rather than a business concept.

**Step 3. Enveloping.** The transaction set is wrapped in a functional group and an interchange,
each with control numbers.
*Breaks when:* control numbers repeat after a system restore, so the partner rejects the
interchange as a duplicate.
*Analyst sees:* a day of orders silently absent, with no error on your side.

**Step 4. Transport.** The interchange travels by VAN mailbox, AS2, or SFTP.
*Breaks when:* the connection is up but the partner's mailbox is not being collected, or the
partner's certificate expired.
*Analyst sees:* delivery confirmation on your side and no processing on theirs.

**Step 5. Partner validation.** The partner's translator validates syntax, code lists, and
required segments, then generates a 997 with AK1 through AK9 segments.
*Breaks when:* the partner requires a segment your map omits, and the 997 reports it as a segment
error three levels down.
*Analyst sees:* a 997 with a reject status and a code that requires the partner's guide to
interpret.

**Step 6. Partner mapping.** The partner maps X12 into its own ERP and may generate an 855
purchase order acknowledgment.
*Breaks when:* the map succeeds syntactically but the business rules reject the order, for example
a discontinued item or a price outside tolerance.
*Analyst sees:* a 997 acceptance and no 855, which is the signal that the 997 was never the
answer.

**Step 7. Acknowledgment return.** The 997 travels back on the same network.
*Breaks when:* the 997 is generated but not routed, or is routed to a mailbox nobody monitors.
*Analyst sees:* an order with unknown status, and a team that assumes silence means success.

> **Note:** in retail, the 856 is the document with the highest commercial stakes. Retailer
> vendor-compliance programmes assess chargebacks on ASN timing, pack hierarchy, carton labels,
> and item identifiers. Vendor-published penalty schedules in 2026 commonly sit in the range of
> USD 200 to USD 500 per shipment for ASN defects, with some retailers charging per carton. These
> figures are vendor-compiled and should be treated as indicative, not audited.

### 5.3 The API path

**Step 1. Authentication.** The consumer obtains a credential, usually an OAuth 2.0 client
credentials token or an API key.
*Breaks when:* the token lifetime is shorter than the job and the refresh is not implemented.
*Analyst sees:* a job that fails partway through, with a subset of pages loaded.

**Step 2. Endpoint selection.** The consumer chooses a resource, for example `/orders` or
`/inventory`, and a version.
*Breaks when:* the version is pinned to a deprecated path and the provider sunsets it.
*Analyst sees:* a hard failure on a date announced in a changelog nobody read.

**Step 3. Filtering.** The request carries a time filter such as `updated_since`, a status filter,
and a page size.
*Breaks when:* the filter uses modification time and the provider backdates records during
corrections, so a record is never picked up.
*Analyst sees:* a slowly growing set of records that never reconcile to the source count.

**Step 4. Pagination.** The response returns one page and a cursor or next link.
*Breaks when:* the loop stops after the first page, or the cursor is stored and reused after it
expires.
*Analyst sees:* a population that is exactly the page size, which is the tell.

**Step 5. Rate limiting.** The provider enforces a requests-per-second or requests-per-day cap and
returns HTTP 429 with a `Retry-After` header.
*Breaks when:* the consumer retries immediately, multiplying load, or all workers retry at once.
*Analyst sees:* a backfill that never completes and a provider that throttles the account.

**Step 6. Partial representation.** The default response may omit line items, status history, or
custom fields unless an expansion parameter is supplied.
*Breaks when:* the consumer treats the default representation as complete.
*Analyst sees:* orders with zero lines, which aggregate to zero units and a plausible-looking
empty chart.

**Step 7. Eventual consistency.** A write returns 202 Accepted and the read model catches up
later, sometimes minutes later.
*Breaks when:* the consumer reads immediately after writing and caches the stale result.
*Analyst sees:* a reconciliation gap that closes if you re-run the job an hour later.

**Step 8. Versioning and deprecation.** The provider adds fields, changes enum values, or removes
attributes under a deprecation policy.
*Breaks when:* the consumer parses strictly and rejects unknown fields, or silently maps a new
enum value to null.
*Analyst sees:* a sudden null rate in one column, with no error anywhere.

The delta-versus-snapshot decision deserves its own rule. Use a **snapshot** when the provider
offers no reliable change feed and the population is small enough to re-read within the freshness
budget. Use a **delta** when the provider exposes `updated_since` with a stable, monotonic
timestamp and a unique identifier that breaks ties. Never use a snapshot to emulate a delta by
diffing consecutive snapshots unless you can prove that deletions are visible; a deleted record
simply vanishes, and a diff cannot distinguish deletion from a failed read.

### 5.4 The event path

**Step 1. Event production.** A producer emits an event with an identifier, a type, a version, an
event timestamp, a subject, and a payload.
*Breaks when:* the identifier is generated per delivery rather than per event, making deduplication
impossible.
*Analyst sees:* duplicates that look unique.

**Step 2. Broker persistence.** The broker appends the event to a partition and assigns an offset.
*Breaks when:* the partition key is chosen poorly, so all events for one order land on different
partitions and lose ordering.
*Analyst sees:* status transitions applied out of order, such as a cancellation before a creation.

**Step 3. Consumption.** A consumer group reads events and processes them.
*Breaks when:* the consumer commits its offset before the downstream write succeeds.
*Analyst sees:* lost events after a crash, with no error.

**Step 4. Acknowledgement and retry.** The consumer acknowledges after processing; on failure the
broker redelivers.
*Breaks when:* the downstream write succeeds and the acknowledgement fails, producing a duplicate.
*Analyst sees:* the same event applied twice.

**Step 5. Deduplication.** The consumer checks the event identifier against a store of processed
identifiers before applying the event.
*Breaks when:* the store is in memory, or has a retention shorter than the producer's retry window.
*Analyst sees:* duplicates reappearing after a restart or after the retention period.

**Step 6. Ordering.** Ordering is guaranteed within a partition, not across partitions.
*Breaks when:* the consumer assumes global order and applies a stale status over a newer one.
*Analyst sees:* a shipment that reverts from delivered to in transit.

**Step 7. Late arrival.** An event with an event time earlier than the current watermark arrives.
*Breaks when:* the window has closed and the event is dropped rather than routed to a late path.
*Analyst sees:* a total that is short by exactly the late volume.

**Step 8. Gap detection.** The consumer monitors offsets, sequence numbers, or a heartbeat to
detect missing ranges.
*Breaks when:* there is no gap detection, and a paused partition looks like a quiet period.
*Analyst sees:* a flat line that is indistinguishable from zero demand.

> **Note:** at-least-once is the realistic default for any cross-system event feed. Exactly-once
> is achievable within a stream processor that controls both its state store and its output
> transaction, and it is not achievable across an HTTP callback into a system you do not control
> without idempotent writes at the destination. Design for duplicates. Prove you handle them.

### 5.5 The data contract

A data contract is a versioned agreement between a producer and a consumer. It has six clauses,
and each clause closes a failure mode that the transport cannot close on its own.

| Clause | What it states | Failure it prevents | Enforcement |
|---|---|---|---|
| Schema | Field names, types, nullability, keys, and allowed values | Silent type changes and dropped columns | Schema validation at ingestion and in continuous integration |
| Semantics | What each field means, its unit, its time zone, and its relationship to business events | Correct values read as the wrong concept | Documented definitions plus reference tests |
| Freshness SLA | Maximum age at availability, expressed as P50 and P90, with a breach process | Dashboards that are stale but green | Freshness monitoring with alerting and a "data as of" timestamp |
| Quality thresholds | Acceptable null rates, referential integrity, and volume bounds | Partial loads accepted as complete | Automated checks with quarantine |
| Ownership | The named role accountable for the producer side, and the named role on the consumer side | Nobody to call when it breaks | Registry with on-call rotation |
| Change policy | Notice period, versioning scheme, deprecation window, and compatibility rules | Tuesday deploys that break Thursday reports | Contract tests in the producer's pipeline |

The contract is a governance artefact because it allocates responsibility. A schema document says
what the data looks like. A contract says who must tell whom, how far in advance, and what happens
when the promise is broken. Without an owner and a breach process, a contract is a README with
extra steps.

Three practical tests tell you whether a contract is real: can you name the person who must approve
a breaking change, or is the answer a team alias; does a violation fail the producer's build, or
only a warehouse job; and is the freshness SLA measured from the business event rather than from
file arrival. A contract that fails any of the three is a schema document.

> **Note:** unchanged schema does not mean unchanged semantics. The most damaging changes are not
> column renames. They are enum splits (one status becomes two), unit changes (cases to eaches),
> scope changes (a region is added to a previously national feed), and backfill changes (history
> is restated). None of these breaks a schema validator. All of them break a model.

### 5.6 Assembling the latency budget

The budget is additive, and the largest term is usually not the transport:

```text
L_total = L_app + L_batch + L_network + L_land + L_validate + L_transform + L_serve + L_decide
```

- `L_app` — source application commit lag. Minutes to a shift.
- `L_batch` — waiting for the scheduled cycle. Hours for EDI and files, zero for events.
- `L_network` — transport time. Seconds to minutes in almost every case.
- `L_validate`, `L_land` — arrival, parsing, schema checks, quarantine. Minutes to an hour.
- `L_transform` — staging to warehouse to semantic model. One to four hours in a typical stack.
- `L_serve`, `L_decide` — cache refresh and human review. Minutes to days.

Upgrading the transport removes `L_batch` and part of `L_app`, and leaves `L_transform`, `L_serve`,
and `L_decide` untouched. If those sum to four hours, an event feed delivers a four-hour system.

## 6. Worked example

Harbor Market, a grocery retailer, places purchase order 4500123456 with Calder & Co, a supplier,
on Monday at 09:00 local time. The order covers 12 stock-keeping units (SKUs) and 480 cases in
total, for delivery on Thursday. The same order exists in four forms because Harbor Market and
Calder & Co have four integrations running in parallel during a migration.

### 6.1 The four arrivals

| Transport | Path | Batch or poll rule | Landing time | End-to-end latency |
|---|---|---|---|---|
| EDI 850 | Harbor ERP to translator to VAN to Calder translator | Harbor batches POs at 16:00 daily | Monday 18:15 | 9.25 hours |
| REST API | Calder order API, polled by Harbor | Poll every 15 minutes; header then line expansion | Monday 11:13 | 2.22 hours |
| Flat file | Harbor portal CSV to SFTP, collected by Calder | Export 23:00; pickup 06:00 | Tuesday 08:51 | 23.85 hours |
| Event stream | Calder `order.created` webhook to Harbor | Push with retry | Monday 10:47 | 1.78 hours |

The event stream is a pilot and is not yet used for planning. It is included because it shows what
the latency floor looks like when the batch term is removed.

### 6.2 What the analyst sees at each hour

| Clock | Elapsed | Event stream | API | EDI | Flat file |
|---|---:|---|---|---|---|
| Mon 09:00 | 0.00 | — | — | — | — |
| Mon 09:15 | 0.25 | — | Order header, 0 lines | — | — |
| Mon 09:40 | 0.67 | — | Header plus 12 lines | — | — |
| Mon 10:05 | 1.08 | — | Plus backorder status | — | — |
| Mon 10:47 | 1.78 | Order plus 12 lines, complete | Header plus lines | — | — |
| Mon 11:13 | 2.22 | Complete in warehouse | Complete in warehouse | — | — |
| Mon 12:00 | 3.00 | Yes | Yes | — | — |
| Mon 17:00 | 8.00 | Yes | Yes | — | — |
| Mon 18:15 | 9.25 | Yes | Yes | 850 landed, 12 lines | — |
| Mon 19:00 | 10.00 | Yes | Yes | Yes | — |
| Tue 01:00 | 16.00 | Yes | Yes | Yes | — |
| Tue 08:51 | 23.85 | Yes | Yes | Yes | 12 lines, quantity in eaches |

The table is the point of the topic. At 09:15 an analyst querying the API sees an order with no
lines. A dashboard that counts orders reports one. A dashboard that sums ordered units reports
zero. Both are correct computations against the data that had arrived. Neither is a correct
description of the business.

At 18:15 the EDI document lands, nine hours after the event. The analyst watching the API has had
the order for nine hours; the analyst relying on EDI has had it for none. If both report to the
same planning meeting, they will disagree about whether the order exists, and both can prove it.

### 6.3 The reconciliation at 24 hours

| Measure | EDI 850 | API | Flat file | Event stream |
|---|---:|---:|---:|---:|
| Order records | 1 | 1 | 1 | 3 deliveries, 1 distinct identifier |
| Line records | 12 | 12 | 12 | 12 per delivery |
| Units as reported | 480 cases | 480 cases | 5,760 eaches | 480 cases |
| Units normalised to cases | 480 | 480 | 480 | 480 |
| Naive sum without normalisation | 480 | 480 | 5,760 | 1,440 |
| Requested delivery date present | Yes | Only with expansion | Yes | Yes |
| Backorder flag present | Yes, in the 855 | Yes | No | Yes |
| Business acknowledgment | 855, separate document | Status field | None | None |

Two errors are visible and one is hidden.

The visible errors are the unit-of-measure mismatch in the file (5,760 eaches is 480 cases at 12
eaches per case, so a naive sum overstates demand by 12 times) and the duplicate event deliveries
(three deliveries of one event, so a naive count overstates order volume by three times).

The hidden error is the requested delivery date. The file does not carry it, the API carries it
only under an expansion parameter, and a planner who joins on the file defaults it to the file
date. That defaults every order to a one-day lead time, which is wrong by the entire order-to-
delivery window.

### 6.4 Which transport is authoritative for which field

| Field | Authoritative source | Why |
|---|---|---|
| Order existence and identifier | Event stream or API | Earliest and exact, with the buyer's own identifier |
| Line quantities and units of measure | EDI 850 | Contract-grade document with explicit unit codes |
| Requested delivery date | EDI 850 | Retained in the document; often dropped in files |
| Business acceptance | EDI 855 | The 997 does not carry it |
| Shipment contents and pack hierarchy | EDI 856 | Pack-level detail with carton identifiers |
| Physical scan at a location | EPCIS event | Event-level truth about an object |
| Invoice amount and tax | EDI 810 or a Peppol invoice | Legal document with tax semantics |
| Inventory position | EDI 846 or API | Depends on cadence; neither is a substitute for a scan |

The table is a design output, not a fact about the world. It says that no single transport is
authoritative for the order. The analyst's job is to assign authority per field and to state the
assignment in the data model, so that a join does not silently pick the wrong source.

### 6.5 Sensitivity: what changes the answer

Move the order from 09:00 to 17:00 and the EDI path changes by a full day. The 16:00 cut-off has
passed, so the document waits for Tuesday's batch, lands at Tuesday 18:15, and the end-to-end
latency becomes 25.25 hours. The API path changes by minutes. The file path changes by nothing,
because it was never going to arrive before Tuesday morning.

That asymmetry is the practical meaning of "batch": the latency is not an average, it is a
function of where the event falls relative to the cut-off. A feed with a 16:00 cut-off has a
latency distribution from 2.25 hours to 26.25 hours, and the mean is a poor description of any
individual order.

> **Note:** the same cut-off effect explains why month-end and quarter-end data is often worse
> than mid-month data. Events cluster near the boundary, the batch boundary falls in the middle of
> the cluster, and the last day of the period is systematically split across two loads.

## 7. Code, query, or calculation

The calculation that matters is the latency budget. Build it as a table, one row per stage, then
sum.

| Stage | EDI 850 | REST API | Event stream | Flat file |
|---|---:|---:|---:|---:|
| Application commit (`L_app`) | 0.25 h | 0.25 h | 0.02 h | 0.00 h |
| Batch wait (`L_batch`) | 7.00 h | 0.13 h poll | 0.00 h | 14.00 h |
| Network (`L_network`) | 0.25 h | 0.10 h fetch | 0.01 h | 0.10 h |
| Landing (`L_land`) | 0.25 h | 0.25 h | 0.25 h | 0.25 h |
| Validation (`L_validate`) | 0.50 h | 0.50 h | 0.50 h | 1.00 h |
| Transform (`L_transform`) | 0.75 h | 0.75 h | 0.75 h | 0.75 h |
| Serve (`L_serve`) | 0.25 h | 0.25 h | 0.25 h | 0.25 h |
| **Total to dashboard** | **9.25 h** | **2.23 h** | **1.78 h** | **16.35 h** |
| Decision lag (`L_decide`) | 4.00 h | 4.00 h | 4.00 h | 4.00 h |
| **Total to action** | **13.25 h** | **6.23 h** | **5.78 h** | **20.35 h** |

```text
EDI    = 0.25 + 7.00 + 0.25 + 0.25 + 0.50 + 0.75 + 0.25 = 9.25 h
API    = 0.25 + 0.13 + 0.10 + 0.25 + 0.50 + 0.75 + 0.25 = 2.23 h
Event  = 0.02 + 0.00 + 0.01 + 0.25 + 0.50 + 0.75 + 0.25 = 1.78 h
File   = 0.00 + 14.00 + 0.10 + 0.25 + 1.00 + 0.75 + 0.25 = 16.35 h
```

The file total here is 16.35 hours to the dashboard and 23.85 hours to the 06:00 pickup plus
parsing, as used in section 6. The difference is the pickup dwell, which is an operational
schedule rather than a pipeline stage. Include it when you are describing the end-to-end business
latency, and exclude it when you are describing your own pipeline's contribution. State which one
you mean.

The arithmetic to remember: **removing the transport term removes at most the transport term.**
In the EDI case transport and network are 0.50 hours of 9.25, or 5%; the batch wait is 76%. Moving
to an API helps only if the batch wait goes with it, and that is a scheduling decision.

## 8. Data, standards, and regulatory context

Transport is governed by standards at every layer: the document, the envelope, the network, the
event, and increasingly the contract itself.

### 8.1 Document standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ANSI ASC X12 | North American EDI | Transaction sets, versions such as 004010 and 005010, segments and elements | B |
| UN/EDIFACT | International EDI | Message types such as ORDERS, DESADV, INVOIC, and INVRPT | B |
| GS1 EANCOM | Retail EDI in many markets | An EDIFACT subset with retail-specific code lists | B |
| VICS | North American retail EDI guidelines | The 850, 856, and 810 subsets retailers commonly mandate | B |
| GS1 EPCIS 2.0 | Event data about physical objects | What happened, to which object, where, when, why, and how | B |
| GS1 CBV 2.0 | Event vocabulary | The code lists that make EPCIS events comparable | B |

X12 version numbers matter more than analysts expect. A partner on 004010 and a partner on 005010
will have different segment requirements for the same document, and a translator configured for
one will reject the other. Version is part of the integration contract, not an implementation
detail.

### 8.2 E-invoicing and the European direction

| Instrument | What it does | Status as of 2026 | Grade |
|---|---|---|---|
| EN 16931 | Defines the European semantic model for a core electronic invoice: business terms, groups, and rules | Published 2017 under Directive 2014/55/EU; current | B |
| Peppol BIS Billing 3.0 | A Core Invoice Usage Specification on EN 16931, bound to UBL 2.1 | Current release maintained by OpenPeppol | B |
| Peppol four-corner model | Routes documents from sender to access point to access point to receiver | Operational; AS4 profile in use | B |
| Peppol AS4 profile | One-way push over TLS on port 443, with dynamic discovery via SMP and SML | Version 2.0.3, April 2024 | B |
| VAT in the Digital Age (ViDA) | Digital reporting and e-invoicing for intra-EU business-to-business transactions | Adopted March 2025; core digital reporting requirement from 1 July 2030 | B |
| National e-invoicing mandates | Country-level clearance or reporting regimes | Several live or phasing between 2026 and 2028 | B |

The ViDA timetable is the one to internalise because it changes what "the invoice arrived" means.
Under a digital reporting regime, the invoice is reported to or cleared by an authority at or near
issuance. The cash still arrives late. The information about the invoice does not.

> **Note:** national mandates move faster than the EU timetable and are revised. Belgium, Germany,
> Poland, France, and Spain have all announced or implemented domestic regimes with dates between
> 2026 and 2028. Vendor-maintained trackers are useful for orientation but should not be used as
> legal advice; confirm against the national authority before designing a compliance date into a
> system.

### 8.3 Traceability and serialisation

| Requirement | Jurisdiction | Transport implication | Grade |
|---|---|---|---|
| GS1 EPCIS 2.0 | Global standard | Event exchange in JSON or XML over REST, AS2, or a message bus | B |
| U.S. Food Traceability Rule (FSMA 204) | United States, food | Key data elements at critical tracking events; EPCIS is a common carrier | B |
| DSCSA | United States, pharmaceuticals | Interoperable, serialised transaction information and verification | B |
| EU Falsified Medicines Directive | European Union, pharmaceuticals | Verification at dispensing against a national repository | B |

FSMA 204 deserves a precise statement because its date has moved. The original compliance date was
20 January 2026. In August 2025 the Food and Drug Administration (FDA) published a proposed rule to
extend it by 30 months to 20 July 2028. As of 2026 the extension has not been finalised as a rule,
but congressional appropriations language directs the FDA not to enforce before 20 July 2028. Treat
the 2028 date as the operative planning assumption and the rulemaking status as unsettled.

DSCSA followed a similar pattern of deadlines and exemptions. Enforcement of the enhanced
interoperable exchange requirements was staged through 2025 for manufacturers, wholesalers, and
larger dispensers, with a longer exemption for small dispensers. The practical consequence for a
data team is that a pharmaceutical feed may be EPCIS 1.2 XML in one connection and EPCIS 2.0 JSON
in another, in the same year, for the same product.

### 8.4 API and contract standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| OpenAPI Specification | API description | Endpoints, parameters, schemas, and response shapes | B |
| JSON Schema | JSON structure | Types, required fields, and validation rules | B |
| Apache Avro | Binary record serialisation with schema evolution | Field addition and removal rules in streaming pipelines | B |
| OpenMetadata data contract standard | Contract structure | Schema, semantics, quality, SLA, ownership, and lifecycle states | C |

The first three are format standards. The fourth is a metadata convention that makes the
non-schema clauses explicit. None of them enforces anything; enforcement comes from the
producer's build pipeline and the consumer's ingestion validation.

### 8.5 What the standards do not cover

Standards govern syntax and, increasingly, semantics. They do not govern four things that decide
whether your analysis is right: freshness (no standard says how quickly a document must follow the
business event), completeness proof (a standard defines control numbers but does not require you to
reconcile them), ownership (a standard says who sends what, not who is accountable when the send
fails), and change notice (a standard defines versions, not how much notice precedes one). Those
four gaps are exactly the content of the data contract in section 5.5, which is why the contract,
not the standard, is the load-bearing artefact.

## 9. Industry practice and case evidence

### 9.1 The API-versus-EDI question

The 2025 evidence points consistently in one direction: hybrid integration, not replacement.

| Claim | Source and date | Grade | What it actually supports |
|---|---|---|---|
| Hybrid EDI and API is "very common practice": EDI for bookings, invoicing, and customs; APIs for tracking and live estimates | Hapag-Lloyd, August 2025 | C | An operator's description of its own integration estate; not a market measurement |
| The future is "enhancing, not replacing" EDI, with APIs and AI assisting mapping and exception handling | IBM community blog, July 2025 | D | A vendor's positioning; useful as a signal of product direction, not adoption |
| An API layer on top of existing EDI systems is the realistic pattern | Orderful, October 2025 | D | A vendor selling exactly that layer |
| APIs are not replacing EDI in retail because retailers mandate EDI documents | TrueCommerce, May 2025 | D | A vendor with an EDI business |
| EDI remains tightly coupled to core systems, making migration high-risk | LTM, undated 2025 | D | A consultancy's view |

The honest reading: every source agrees, and most of them sell the thing they agree about. The one
operator statement, from a carrier, is the most credible and the least quantified. There is no
public dataset that measures the share of business documents moving by each transport, and any
figure that claims to should be treated as a vendor estimate.

### 9.2 Volume evidence

The cleanest public number is a payments-system statistic, not a market forecast. Canadian B2B EDI
transaction volume rose from 4.99 million in 2023 to 5.16 million in 2024, an increase of about
3.5%, with a value of C$228.3 billion, as reported in a 2025 CGI analysis of Payments Canada
statistics. Grade B for the statistic; it covers one country and one network, so it is a
lower-bound indicator of legacy EDI persistence, not a global measure.

Market-size estimates disagree by a factor of seven because they define the market differently:

| Estimate | Scope | Value | Grade |
|---|---|---|---|
| Mordor Intelligence | EDI software | USD 2.60 billion in 2025, 11.8% compound annual growth to 2030 | D |
| QY Research | B2B EDI integration platforms | USD 1.15 billion in 2025, 7.3% growth to 2032 | D |
| The Business Research Company | B2B middleware and integration | USD 7.87 billion in 2025, 9.2% growth to 2030 | D |

The disagreement is not a controversy. It is a definition problem. Quoting any one of these as
"the EDI market" without its scope is a category error, and in a business case it is worse,
because the number will be used to justify an investment.

### 9.3 Where EDI errors actually occur

Public evidence on EDI error rates is weak and mostly vendor-produced. The most specific claims
come from retail chargeback schedules, which are contracts rather than research.

| Claim | Source and date | Grade | Caveat |
|---|---|---|---|
| ASN defects commonly cost USD 200 to USD 500 per shipment across 11 retailers | OrderSync chargeback database, 2026 | D | Vendor-compiled; schedules change and are negotiated per supplier |
| One retailer charges USD 0.75 per carton for ASN violations, with a USD 100 minimum, effective May 2025 | OrderSync, citing retailer schedule | D | A single retailer's rate; not generalisable |
| A 997 confirms receipt and syntactic processing, not business acceptance | Cleo documentation; Microsoft X12 reference | C | Consistent across implementers and matches the standard |
| Partners commonly expect a 997 within 48 hours | SPS Commerce support documentation | D | An operational norm, not a standard requirement |

The defensible conclusion is not a number. It is that **EDI errors concentrate at the boundaries
between systems, not inside the standard**: in the translation map, in the identifier that does not
match, in the acknowledgment that was generated but not routed. A team that monitors only
transmission success will miss most of them.

### 9.4 Event standards in practice

GS1 EPCIS 2.0 was ratified in June 2022 and adds JSON and JSON-LD representations, REST and
OpenAPI bindings, a sensor element list, and an association event for aggregating objects. Grade B
for the standard; the specification is public and stable.

Adoption is being pulled by regulation rather than by analytics demand:

- The GS1 US published EPCIS recommendations for FSMA 204 critical tracking events in May 2025,
  mapping key data elements to EPCIS 2.0 fields. Grade B for the guidance; it is official
  supporting material, not a measurement of adoption.
- The Global Dialogue on Seafood Traceability published a paper in August 2025 arguing that EPCIS
  2.0 with JSON and REST should replace EDI-based traceability exchange. Grade C: an industry
  coalition's position, with an interest in simplifying multi-party exchange.
- Pharmaceutical implementations still run substantial EPCIS 1.2 XML estates because DSCSA
  validation was built against that version. Grade C: an observation from implementation guidance,
  not a census.

The pattern is familiar from EDI: a new standard does not replace the old one. It runs alongside
it, and the data team inherits both.

### 9.5 Freshness in practice

The most useful public benchmark is a company documenting its own pipeline, not an analyst
forecast. Stripe's data pipeline documentation describes a three-hour cadence with a median
freshness of six hours, a 90th percentile of seven hours, and a maximum delay of twelve hours for
most datasets. Grade C: a company describing its own system, which is more credible than a vendor
claim and less generalisable than a survey.

A vendor article on real-time supply chain analytics proposes latency tiers by use case: inventory
visibility at 1 to 15 minutes, stock-out alerts at 5 to 15 minutes, shipment estimates at 1 to 5
minutes, and supplier performance at 15 to 60 minutes. Grade D. The tiers are a reasonable
starting point for a requirements conversation and are not evidence of what companies achieve.

### 9.6 What the evidence does not show

Three gaps are worth stating plainly, because they are where students most often overclaim:

1. **There is no rigorous public comparison of error rates by transport.** Claims that EDI is
   "99.9% accurate" or that APIs "eliminate errors" are marketing. The error rate depends on the
   implementation, the partner, and the reconciliation control, not on the protocol.
2. **There is no reliable public measure of transport mix.** EDI volume statistics exist for
   specific networks; API and event volumes are not publicly reported in comparable units.
3. **There is no evidence that lower latency improves supply chain outcomes by itself.** The
   latency tiers in section 9.5 are design targets. Whether a faster feed produces better service
   depends on whether a decision changes, which is a process question.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating a batch feed as real-time | Dashboard refresh is minutes; source freshness is hours | Refresh frequency is confused with data age | Model reacts to stale state; over-ordering | Publish a "data as of" timestamp beside every metric |
| Treating a 997 as business acceptance | Order status stays "sent" with no 855 | 997 certifies syntax only | Orders believed live while rejected downstream | Track 855 separately; alert on 997 accepted and no 855 |
| Pagination truncation | Population size equals the page size exactly | Loop stops after page 1 | Silent undercount, often by 90% or more | Assert record count against a known total; log page count |
| Rate-limit collapse | Backfill never finishes; provider throttles account | Retries without backoff or jitter | Missing history and a strained partner relationship | Exponential backoff with jitter; respect `Retry-After` |
| Snapshot loaded as delta | Volume doubles on re-run | Full state appended as if it were changes | Double counting in every aggregate | Key by business identifier and upsert; state the mode in metadata |
| Delta loaded as snapshot | History disappears; totals drift down | Only changes retained, no accumulating state | Wrong inventory position and wrong trends | Maintain a current-state table plus an event history |
| Deduplication on arrival time | Duplicates appear after a retry window | Retries have new arrival times | Inflated counts that look plausible | Deduplicate on the producer's event identifier |
| Ordering by processing time | Status regresses from delivered to in transit | Late event applied over newer state | Wrong exception queues and wrong lead times | Order by event time; ignore events older than current state |
| Late events dropped at a closed window | Daily total short by a small, variable amount | Window closed before the event arrived | Persistent undercount in one region or partner | Allowed lateness plus a late-event side path |
| Encoding and BOM corruption | First column name has odd characters | UTF-8 BOM read as data; Windows-1252 read as UTF-8 | Failed joins and null keys | Declare encoding in the contract; decode explicitly |
| Schema drift without a contract | A column's null rate jumps to 30% | Provider added an enum value or renamed a field | Silent semantic change, no error | Contract tests; tolerant reader; alert on new enum values |
| Unit-of-measure ambiguity | Quantities differ by a factor of 6, 12, or 24 | Cases, eaches, and pallets mixed without a unit field | Demand overstated or understated by the pack size | Carry an explicit unit code and a conversion table |
| No cross-transport reconciliation | Two systems disagree; both are "correct" | Each transport is validated only against itself | Planning on one source while finance uses another | Reconcile counts and values across transports daily |

Two of these deserve expansion.
**Pagination truncation** is the most common API defect in analytics code, and it is almost always
invisible. The tell is a population size that is suspiciously round: 100, 250, 1,000. If your
supplier count is exactly 100, you do not have 100 suppliers. You have one page.

**Snapshot loaded as delta** is the most expensive batch defect, because it survives validation.
The row count is plausible, the schema is correct, and the totals are wrong by a factor that
depends on how often the snapshot is reloaded. It is usually discovered by finance, months later.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The partner sends every document it creates | Completeness reasoning starts from the producer side | Silent drops, mailbox collection failures, filtering rules | Reconcile control numbers against the partner's document count |
| Clocks are synchronised within tolerance | Event-time ordering and lateness windows depend on it | Server drift, time-zone misconfiguration | Compare timestamps across systems for the same event |
| Identifiers are stable across transports | Joining an order from EDI to an event requires a shared key | Prefixes, truncation, case changes, renumbering | Sample joins and measure match rate |
| One canonical unit of measure exists | Aggregation requires commensurable quantities | Cases, eaches, pallets, and kilograms mixed | Check the unit field's null rate and value distribution |
| The landing zone is append-only | Reprocessing and audit depend on immutability | In-place overwrite, retention policies | Inspect the write pattern; test a replay |
| Event identifiers are unique per producer | Deduplication depends on it | Identifier generated per delivery | Count distinct identifiers versus deliveries |
| The API follows a published deprecation policy | Version planning depends on notice | Silent field removal, undocumented enum changes | Subscribe to changelogs; contract-test in continuous integration |
| File arrival is atomic | Partial reads are prevented by the convention | Direct writes to the pickup directory | Monitor for truncated final lines and .done sentinels |
| The extract population matches the operational population | Analyst and operator reason about the same set | Filters, test records, intercompany exclusions | Compare counts by status and entity |

Three limits apply. The four-transport taxonomy is a simplification, so classify by guarantee rather
than by protocol name. The latency numbers are illustrative and must be replaced with your own
before a business case. The regulatory dates are a snapshot and should be re-verified against the
authority before being designed into a system.

## 12. Exercises

### Exercise 1 — Transport inventory (recall and application, 45 minutes)

Choose a supply chain you know, or one described in a public case. List every data feed that
crosses an organisational boundary. Classify each by transport, and for each state the
completeness guarantee, the latency, and the reconciliation unit.

**Deliverable:** a table with at least eight feeds, one row per feed, and a one-sentence note on
the weakest guarantee in the set.
**Acceptance criteria:** every row names a reconciliation unit; at least one feed is a hybrid and
is classified by guarantee rather than by protocol name; the weakest guarantee is justified, not
asserted.
**Time:** 45 minutes.

### Exercise 2 — Latency budget arithmetic (calculation, 40 minutes)

Using the stage table in section 7, compute the end-to-end latency for a feed with the following
stages: application commit 0.5 hours, batch wait 3.0 hours, network 0.1 hours, landing 0.25 hours,
validation 0.5 hours, transform 1.5 hours, serve 0.25 hours, decision 6.0 hours. Then compute the
effect of removing the batch wait entirely.

**Deliverable:** two totals in hours, the percentage reduction, and a sentence naming the term that
now dominates.
**Acceptance criteria:** correct arithmetic; the dominant term identified after the change; the
distinction between time-to-dashboard and time-to-action stated explicitly.
**Time:** 40 minutes.

### Exercise 3 — Cross-transport reconciliation (analysis, 50 minutes)

An order for 240 cases arrives by EDI with a unit code of "CA", by API with `quantity: 240` and no
unit field, and by file with `qty: 2880` and a column header `UOM` containing `EA`. The event feed
delivered the order three times with the same `event_id`.

**Deliverable:** a reconciliation table showing reported units, normalised units, and the naive sum
for each transport, plus the deduplicated event count.
**Acceptance criteria:** the pack size is inferred and stated; the naive-sum error is quantified
for each transport; the authoritative source for quantity is named with a reason.
**Time:** 50 minutes.

### Exercise 4 — Read an EDI implementation guide (analysis, 60 minutes)

Obtain a public EDI 856 implementation guide, or the 997 guide referenced in section 16. Identify
three fields that are mandatory in the guide but optional in the base X12 standard, and one segment
whose meaning depends on a qualifier.

**Deliverable:** a one-page summary with the three fields, the qualifier dependency, and a sentence
on what a translator must do to satisfy the guide.
**Acceptance criteria:** correct document and version cited; the qualifier example is specific; the
summary distinguishes standard requirements from partner requirements.
**Time:** 60 minutes.

### Exercise 5 — Draft a data contract (design, 60 minutes)

Write a data contract for the EDI 850 feed in section 6. Cover all six clauses: schema, semantics,
freshness SLA, quality thresholds, ownership, and change policy.

**Deliverable:** a contract of no more than two pages, with a named owner role on each side and a
numeric freshness SLA expressed as P50 and P90.
**Acceptance criteria:** every clause is present and specific; the freshness SLA is measured from
the business event, not the file arrival; the change policy states a notice period and a
compatibility rule; a reader can tell what happens when the contract is breached.
**Time:** 60 minutes.

### Exercise 6 — Pagination audit (analysis, 50 minutes)

Take any public API documentation with pagination, or the pattern described in section 5.3. Write
the loop that would fetch the full population, and list the three failure modes your loop must
handle.

**Deliverable:** pseudocode of no more than 20 lines plus a three-item failure list with the check
for each.
**Acceptance criteria:** the loop terminates on a documented condition, not on an empty page alone;
the rate-limit handling includes `Retry-After`; the audit includes a record-count assertion against
a known total.
**Time:** 50 minutes.

### Exercise 7 — Event deduplication design (design, 45 minutes)

Design the deduplication and ordering strategy for an at-least-once shipment event feed with a
15-minute late-arrival tolerance.

**Deliverable:** a deduplication key, a storage choice with a retention period, an ordering rule, a
late-event policy, and a gap-detection method.
**Acceptance criteria:** the key is the producer's event identifier plus a tie-breaker; retention
exceeds the producer's retry window; the ordering rule uses event time; the late path is described;
gap detection names the signal it monitors.
**Time:** 45 minutes.

### Exercise 8 — File robustness test (application, 45 minutes)

Construct or obtain a CSV with a BOM, a semicolon delimiter, a quoted field containing a comma, a
trailer row, and a Windows-1252 encoded character. Write the parsing rules that handle all five.

**Deliverable:** a list of five rules, each with the symptom it prevents, plus the validation check
that would catch a violation.
**Acceptance criteria:** all five defects addressed; encoding declared explicitly; the trailer row
handled by structure, not by row number; each rule names the failure it prevents.
**Time:** 45 minutes.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
base = 0.50 + 3.00 + 0.10 + 0.25 + 0.50 + 1.50 + 0.25 = 6.10 h to dashboard
action = 6.10 + 6.00 = 12.10 h

without batch wait:
dashboard = 0.50 + 0.00 + 0.10 + 0.25 + 0.50 + 1.50 + 0.25 = 3.10 h
action = 3.10 + 6.00 = 9.10 h

reduction to dashboard = (6.10 - 3.10) / 6.10 = 49.2%
reduction to action    = (12.10 - 9.10) / 12.10 = 24.8%
```

After the change, transform at 1.50 hours is the dominant pipeline term, and decision lag at 6.00
hours dominates the time to action. The correct conclusion is that removing the batch wait halves
the dashboard latency and cuts the time to action by a quarter. A proposal that claims a "tenfold
speed-up" from removing batching is wrong by construction, because it ignores the terms that do not
change.

**Exercise 3 reconciliation.**

Assume a pack size of 12 eaches per case, inferred from 2,880 divided by 240.

| Transport | Reported | Unit | Normalised to cases | Naive sum if taken literally |
|---|---:|---|---:|---:|
| EDI | 240 | CA (cases) | 240 | 240 |
| API | 240 | none | 240, if the contract says cases | 240 |
| Flat file | 2,880 | EA (eaches) | 240 | 2,880 |
| Event | 240 per delivery | cases | 240 distinct | 720 across three deliveries |

The authoritative source for quantity is the EDI 850, because it carries an explicit unit code and
is the contract-grade document. The API is authoritative for order existence and identifier. The
file is authoritative for nothing until its unit column is trusted, and the header `UOM` is a red
flag because it names a concept without constraining its values.

The two naive-sum errors are a 12-times overstatement from the file and a 3-times overstatement
from the event duplicates. Both produce plausible numbers.

**Exercise 5 rubric.**

A strong contract has all six clauses and makes each testable:

- Schema: field list with types, nullability, and keys.
- Semantics: unit of measure, time zone, and the business event each timestamp refers to.
- Freshness: P50 and P90 measured from the business event, with a breach process.
- Quality: null-rate ceilings, referential checks, and volume bounds.
- Ownership: a named role on each side, not a team alias.
- Change policy: notice period, versioning scheme, and compatibility rules.

It also states the reconciliation control: interchange control numbers reconciled daily against
the partner's document count. A contract that lists fields but has no owner, no SLA measured from
the event, and no breach process is a schema document and should be marked as such.

**Exercise 6 expected content.**

The loop must terminate on the documented end condition (a null cursor or a `next` link that is
absent), not merely on an empty page, because an empty page can occur mid-stream. It must persist
the cursor only after the destination commit, and it must handle three failures: a 429 with
`Retry-After`, a cursor expiry mid-scan (restart from the last committed watermark), and a
concurrent write that shifts an offset (the reason to prefer keyset pagination).

**Exercise 7 design rubric.**

A strong answer uses the producer's `event_id` as the deduplication key, stores processed
identifiers for longer than the producer's retry window (commonly 7 days for webhook providers,
sometimes 3 days), orders by `occurred_at` with the identifier as a tie-breaker, routes events
older than the current state to a late path rather than dropping them, and detects gaps by
monitoring partition offsets or a producer sequence number. It also states that the downstream
write is an upsert, so that a duplicate is harmless even if deduplication fails.

A weak answer deduplicates on arrival time, keeps identifiers in memory, or assumes the broker
guarantees exactly-once delivery to the application.

**Exercise 8 expected rules.**

1. Decode with the declared encoding; strip a UTF-8 BOM explicitly (`utf-8-sig` in Python).
2. Detect the delimiter from the header or from the contract; do not assume a comma.
3. Use a real parser that honours quoting, not a string split.
4. Validate row structure against the header width; reject or quarantine rows that do not match.
5. Detect and exclude the trailer row by its structure (a known label in the first field, or a row
   with a different field count), not by row number.

Each rule should name the symptom it prevents: a broken first column name, a single-column table,
shifted fields, a truncated final record, and a doubled total.

## 14. Interview questions

**1. "What is the difference between EDI and an API?"**

A strong answer describes the guarantee, not the format: EDI is a standardised, batched,
contract-grade document exchange with partner-specific maps; an API is a request-response interface
that is fast at the source but partial and schema-unstable. It notes that the two are commonly
hybridised. A weak answer says "EDI is old and APIs are modern," which is a timeline, not a
distinction.

**2. "A dashboard refreshes every five minutes. Is it real-time?"**

A strong answer asks for the source freshness and the pipeline stages, and distinguishes refresh
frequency from data age. It notes that a five-minute refresh over a nightly feed is a daily
dashboard with a fast paint. A weak answer says yes.

**3. "What does a 997 acknowledgment tell you?"**

A strong answer says it confirms receipt and syntactic processing at the segment and element level,
and explicitly that it does not confirm business acceptance, pricing, or shipment. It mentions the
855 as the business acknowledgment. A weak answer says it confirms the order was received, full
stop.

**4. "How would you detect that an API extract is incomplete?"**

A strong answer names the controls: a record-count assertion against a known total, a page-count
log, a watermark that advances monotonically, and a periodic reconciliation against a second
source. It mentions that a population exactly equal to the page size is the tell. A weak answer
says "check for nulls."

**5. "What is the difference between at-least-once and exactly-once?"**

A strong answer says at-least-once guarantees arrival with possible duplicates, exactly-once
guarantees single processing within a controlled boundary, and end-to-end exactly-once across
systems requires idempotency at the destination. It gives the timeout-after-success example. A weak
answer treats exactly-once as a configuration flag.

**6. "Why can't you order events by arrival time?"**

A strong answer explains that late events arrive after later events, so arrival order is not
business order; ordering must use event time with a tie-breaker, and windows need an allowed
lateness. A weak answer says arrival time is fine because the broker preserves order.

**7. "What is a data contract and why is it not just documentation?"**

A strong answer lists the six clauses and says the contract allocates responsibility: an owner, a
freshness SLA, a breach process, and a change policy enforced in the producer's build. A weak
answer describes a schema file.

**8. "You have a choice between EDI and API for a new supplier. How do you decide?"**

A strong answer starts from the decision the data must support and its latency and completeness
requirement, then considers the supplier's capability, the document's legal role, the
reconciliation control, and the cost of change. It may recommend both: EDI for the order and
invoice, API for the tracking. A weak answer picks the newer technology.

**9. "How would you build a latency budget?"**

A strong answer enumerates the stages from event time to decision time, assigns hours to each,
identifies the dominant term, and notes that the transport is usually small. A weak answer quotes a
single end-to-end number without decomposition.

**10. "What is the most dangerous schema change you have seen?"**

A strong answer names a semantic change that did not break the schema: an enum split, a unit
change, a scope change, or a restatement of history. It explains why validation passed and the
model still broke. A weak answer names a dropped column, which is the easy case.

## 15. Key terms

- **Transport** — the mechanism that moves data between systems, with its own latency and
  completeness guarantees.
- **Batch** — a closed set of records processed on a schedule.
- **Stream** — records processed continuously as they arrive.
- **EDI** — standardised business document exchange between trading partners.
- **850 / 855 / 856 / 810 / 846 / 852** — purchase order, order acknowledgment, advance ship
  notice, invoice, inventory advice, and product activity data.
- **997 / 999** — functional and implementation acknowledgments; syntax, not business acceptance.
- **ASN** — advance ship notice, the 856 document describing shipment contents.
- **Translation map** — partner-specific rules converting internal data to and from a standard.
- **Delimiter / header row / encoding / BOM** — the four file properties that break most often.
- **API** — a defined interface for one system to call another.
- **Pagination / cursor** — splitting results into pages and marking a position in the result set.
- **Rate limit** — a cap on requests per unit time, signalled by HTTP 429 and `Retry-After`.
- **Backfill / watermark** — loading history and recording how far an incremental load has reached.
- **Eventual consistency** — a write becomes visible to readers after an unspecified delay.
- **Idempotency** — repeating an operation has the same effect as doing it once.
- **Deduplication key** — the field or composite that identifies a unique event.
- **At-least-once / exactly-once** — delivery guarantees with duplicates, and single processing
  within a boundary.
- **Webhook** — an HTTP callback from producer to consumer when an event occurs.
- **Event time / processing time** — when the event occurred, and when the system handled it.
- **Late-arriving event** — an event whose event time precedes data already processed.
- **Schema drift** — an unexpected change to structure or meaning.
- **Data contract** — a versioned agreement on schema, semantics, freshness, quality, ownership,
  and change policy.
- **Freshness SLA** — a commitment on maximum data age at availability.
- **Latency budget** — the allocation of end-to-end delay across stages.
- **EPCIS** — the GS1 standard for event data about physical objects.
- **Peppol / access point / four-corner model** — the European eDelivery network, its certified
  intermediaries, and its routing topology.
- **EN 16931** — the European semantic model for a core electronic invoice.
- **Snapshot / delta** — full state at a point in time, and the changes since a previous point.
- **Reconciliation** — the control that proves two populations agree.

## 16. Sources

**Peer-reviewed and academic**

- Anon., "A Real-Time Analytics Architecture for Enterprise Order Lifecycle Visibility and Backlog
  Management," *International Journal of Research and Analytics in Information Technology*, 2019.
  DOI: 10.15662/ijrai.2019.0206004. Grade A for peer-review status; the venue is not a top-tier
  journal and the latency figures are a single case, so treat them as illustrative.

**Standards and official**

- Accredited Standards Committee X12, X12 EDI standards, including transaction sets 810, 846, 850,
  852, 855, 856, 997, and 999. Grade B.
- UN/CEFACT, UN/EDIFACT directories, including ORDERS, DESADV, INVOIC, and INVRPT. Grade B.
- GS1, "EPCIS and CBV," https://www.gs1.org/standards/epcis. EPCIS 2.0 ratified June 2022. Grade B.
- GS1, EPCIS 2.0 specification, https://ref.gs1.org/standards/epcis/. Grade B.
- GS1 US, "EPCIS Recommendations for FSMA 204 Critical Tracking Events," Release 2.0, 15 May 2025.
  Grade B.
- OpenPeppol, "Peppol BIS Billing 3.0," https://docs.peppol.eu/poacc/billing/3.0/. Grade B.
- OpenPeppol, "Peppol AS4 Profile," version 2.0.3, 22 April 2024,
  https://docs.peppol.eu/edelivery/as4/specification/. Grade B.
- CEN TC434, EN 16931-1, electronic invoicing semantic data model, 2017, under Directive
  2014/55/EU. Grade B.
- European Commission, "VAT in the Digital Age (ViDA),"
  https://taxation-customs.ec.europa.eu/taxation/vat/vat-digital-age-vida_en. Adopted March 2025;
  core digital reporting from 1 July 2030. Grade B.
- European Commission, "ViDA implementation — Work programme for 2026," 22 May 2026. Grade B.
- U.S. Food and Drug Administration, "Requirements for Additional Traceability Records for Certain
  Foods," FSMA 204 rule page and FAQ. Grade B.
- U.S. Federal Register, "Requirements for Additional Traceability Records for Certain Foods:
  Compliance Date Extension," proposed rule, 7 August 2025, 90 FR 38084, Doc. 2025-14967. Grade B.
- U.S. Food and Drug Administration, "DSCSA Waivers, Exceptions, and Exemptions," and the
  October 2024 exemption letter, https://www.fda.gov/media/182584/download. Grade B.
- IETF, RFC 4180, "Common Format and MIME Type for Comma-Separated Values (CSV) Files." Grade B.

**Company documentation and case material**

- Stripe, "Data Pipeline Data Freshness," https://docs.stripe.com/data/data-pipeline/data-freshness.
  Three-hour cadence, P50 six hours, P90 seven hours, maximum twelve hours. Grade C.
- Hapag-Lloyd, "EDI and API in Shipping: What's the Difference, and When to Use What?", August
  2025. Grade C.
- Cloudaeon, "Databricks Supply Chain Intelligence Modernisation" case study. Grade C.
- Ameriflex, "Flex EDI Specifications," version 1.3, 20 August 2025. Documents a once-per-business-
  day SFTP cut-off at 09:30 Eastern. Grade C.
- Microsoft Learn, "X12 997 Acknowledgment," BizTalk Server documentation. Grade C for the
  implementation description; the underlying standard is Grade B.
- Apache Kafka, Kafka Streams and Confluent delivery-semantics documentation, and Apache Flink,
  "Timely Stream Processing." Grade B for the documented semantics.
- OWASP, "Webhook Security Cheat Sheet." Grade B for security guidance; it is a consensus document,
  not a measurement.

**Vendor, analyst, and trade press**

- IBM, "How IBM Is Modernizing EDI with APIs and GenAI for Smarter B2B Integration," 28 July 2025.
  Grade D.
- Orderful, "From EDI to API: Hybrid Integration for B2B," October 2025. Grade D.
- TrueCommerce, "EDI vs. API: Learn Why APIs Aren't Replacing EDI," May 2025. Grade D.
- LTM, "Are APIs on the Path to Eliminate EDI?", 2025. Grade D.
- SBSA Technology, "EDI 856 ASN Troubleshooting Guide: 10 Errors That Trigger Retail Compliance
  Chargebacks," 2025. Grade D.
- OrderSync, "EDI Chargeback Database 2026," https://ordersync.io/edi-chargebacks. Grade D.
- SPS Commerce, "997 Not Posting to the Portal," and Cleo, "EDI 997 Functional Acknowledgment,"
  support documentation. Grade D.
- Mordor Intelligence (2025), QY Research (2026), and The Business Research Company (2026), EDI
  and B2B integration market reports. Grade D; scopes differ by a factor of seven, as section 9.2
  shows.
- CGI, "Transforming Business Payments in Canada," May 2025, analysing Payments Canada ACSS
  statistics. Grade C for the analysis; the underlying volume statistic is Grade B.
- Global Dialogue on Seafood Traceability, "Beyond Transactions: Building Traceable Supply Chains
  with EPCIS," August 2025. Grade C.
- Rubicon, "Real-time Supply Chain Analytics: Architecture, Use Cases and Costs." Grade D.
- Airbyte, "Data Contracts: What Are They and Do You Need Them?", August 2025. Grade D.
- OpenMetadata Standards, "Data Contract," https://openmetadatastandards.org/data-contracts/;
  The Backend Engineering Handbook, "API Evolution at Scale"; and Sainso Technology, "Reliable API
  Integration for Pagination, Rate Limits, and Bulk Sync." Grade C and D respectively.
- FastCSV, "CSV Interpretation," and the Python Software Foundation, `csv` module documentation.
  Grade B for the format and library behaviour.
- Thomson Reuters, "E-invoicing in the EU: Regulatory Updates and Compliance," and peppolo.com
  mandate tracker, 2026. Grade D for the trackers; use only for orientation.

**Note on citations.** Figures from surveys, market reports, and vendor trackers are reported as
those sources state them and are not independently verified. Regulatory dates change; verify
against the authority before relying on any of them.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_telemetry-iot-vision.md`](05_telemetry-iot-vision.md)
- Next topic: [`07_access-and-licensing.md`](07_access-and-licensing.md)
- Related: [`01_systems-of-record.md`](01_systems-of-record.md)
- Foundation: [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
