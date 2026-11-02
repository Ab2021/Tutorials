> Module 03 — Data Engineering and Interoperability
> Topic 08 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 02.06, 03.03

# EDI: X12 and EDIFACT

---

## 1. Why this topic exists

At 04:12 on a Tuesday, an advance ship notice (ASN) arrives at a retailer's electronic data
interchange (EDI) gateway. It declares twelve pallets and 480 cases against purchase order
PO-4471. At 09:40 the trailer is unloaded and the dock counts 456 cases. The receiving system has
already been told what to expect, so it does not count. It reconciles. The reconciliation
produces a shortage of 24 cases, a dispute that takes eleven days to close, and an inventory
position that is wrong in the supplier's vendor-managed inventory (VMI) calculation for the whole
of that period.

Nothing in that sequence is a pipeline failure in the ordinary sense. Every file arrived. Every
control number matched. Every acknowledgement was generated. The failure is semantic: **the
document that triggered receiving described a shipment that did not exist.** The data was
complete, timely, syntactically valid, and wrong.

That is the subject of this topic. EDI is not a file format to be parsed and forgotten. It is the
layer where commercial obligation becomes machine-readable data, and it is still, in 2026, the
production backbone of most business-to-business (B2B) supply chains. An analyst who treats an
EDI feed as a neutral observation of the physical world will build models on a document trail
that describes intent, not fact.

Three failures show the shape of the problem, and each is a different kind of category error:

1. **The ASN that was a revision.** A supplier ships 480 cases, discovers two damaged pallets at
   the loading dock, repacks, and transmits a replacement 856 for 456 cases. The retailer's
   gateway accepts both messages, because both are syntactically valid and both carry the same
   shipment identifier. The receiving application keeps the first. The dock counts 456. The
   system believes 480. The error is invisible for a month because the invoice, generated from
   the order rather than the receipt, also says 480. Nobody is lying, and every number reconciles
   to another number that is also wrong.
2. **The sell-through file that omitted a region.** A retailer sends weekly 852 Product Activity
   Data. One week the file covers 94% of stores because a regional extract job failed. The
   supplier's model reads the missing stores as zero sales, concludes that demand collapsed, and
   cuts the next order. Two weeks later those stores show a spike as their stock is replenished
   from the retailer's own distribution centre. The model learns a seasonality that never happened.
3. **The acknowledgement that was read as consent.** A buyer transmits an 850 purchase order and
   receives a 997 Functional Acknowledgment. The order team marks the order confirmed. The 997
   says only that the seller's translator parsed the file. The seller may reject the order
   commercially the next day with an 855 that nobody is watching for, because the integration was
   built to wait for the 997 and stop.

The common thread is that each failure involves a document whose *scope* was misunderstood. A 997
is scoped to syntax. An 856 is scoped to a declared shipment. An 852 is scoped to the reporting
period and population the sender chose. The document does not carry its own scope; the
implementation guide and the trading-partner agreement carry it, and neither is in your data
warehouse unless somebody put it there.

**What the prerequisite gave you, and what this topic adds.** Topic 02.06, *Data Movement: EDI,
API, Flat File, Event*, established the transport bargain and the distinction between a 997 and
business acceptance. Topic 03.03, *Ingestion Patterns*, established idempotency, watermarking,
and file-drop discipline. EDI is the hardest test of those patterns, because the same business
event arrives more than once by design (original and replacement documents), ordering is
guaranteed only within an interchange, and a duplicate is not always an error. Distinguishing a
replacement from a retransmission is a semantic problem, and semantics live in the document.

This topic goes inside the documents. It asks which decision each transaction set serves, how the
envelope makes a document traceable, what the acknowledgement chain certifies at each level, how
two partners can implement the same standard differently, and why the 856 is the hinge between
information flow and physical flow.

**Why this is an analytics topic and not only an integration topic.** Every downstream model
consumes EDI-derived facts whether or not the analyst knows it. Forecast labels come from
shipments, which are 856 or 810 artefacts. Sell-through features come from 852. Service metrics
come from receipt timestamps created by a 856-triggered receiving transaction. Supplier scorecards
come from 855 response latency. If you cannot say which document produced a column, you cannot say
what the column measures. That is why the mechanics matter less than the judgement about which
document is authoritative for which fact.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** each of the eight core transaction sets (850, 855, 856, 810, 846, 852, 940, 945) and
   their EDIFACT analogues to the business decision it serves, and state which party originates
   it and which party acts on it.
2. **Draw** the X12 envelope hierarchy from interchange to segment, name the three control-number
   pairs that must match, and explain what a mismatch proves.
3. **Distinguish** the four acknowledgement layers (TA1, 997, 999, CONTRL) by scope, and state
   precisely what each one does and does not certify about the business transaction.
4. **Diagnose** an 856-versus-goods discrepancy, trace its consequences through receiving,
   inventory position, invoice matching, and replenishment, and quantify the direct and indirect
   cost.
5. **Read** a short segment excerpt and identify which element, qualifier, or hierarchy pointer is
   missing or misplaced, and what a lenient downstream parser would wrongly infer.
6. **Compare** two implementation guides for the same transaction set and produce a variance
   list that a mapping team could act on.
7. **Evaluate** whether an 852 or an 846 feed is valid evidence of sell-through or of stock
   position, by checking population, period, unit of measure, and qualifier semantics.
8. **Argue** a defensible boundary between EDI and application programming interface (API)
   integration for a named interface, on the basis of obligation, auditability, and latency
   rather than fashion.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three ideas at once. EDI is a **contract layer**, not a data feed. Every document is
**nested inside an envelope** that makes it traceable. Every acknowledgement certifies a
**narrower claim** than the business reader assumes.

### 3.1 A document is an assertion by a party, not a fact about the world

A purchase order asserts an intention to buy. A ship notice asserts an intention to ship, and a
description of what the sender believes is in the trailer. An invoice asserts an amount owed. An
inventory advice asserts a quantity at a moment the sender chose. None of these is a measurement.
Each is a speech act by an organisation with an interest in what you do next.

This gives EDI three properties that distinguish it from an API response, each with a cost:

| Property | What it buys you | What it costs you | Typical analyst error |
|---|---|---|---|
| Document-scoped: one complete document with its own control number and count | Completeness is checkable per document | Batching; a document is only as fresh as its batch | Treating a daily 852 as continuous demand |
| Partner-narrowed: the guide, not the standard, defines the dialect | Precision where it matters commercially | Map proliferation; silent divergence | Assuming the standard is the specification |
| Obligation-bearing: the document is referenced in a contract | Legal weight and audit trail | Errors propagate into money, not just metrics | Treating a chargeback as a data-quality quirk |

### 3.2 The envelope, drawn once

Every X12 interchange and every EDIFACT interchange has the same three-level shape. The names
differ; the logic does not.

```text
   ONE INTERCHANGE ................................ X12: ISA ... IEA
   :                                              EDIFACT: UNB ... UNZ
   :  ONE FUNCTIONAL GROUP ....................... X12: GS ... GE
   :  :                                           EDIFACT: UNG ... UNE
   :  :  ONE TRANSACTION SET / MESSAGE ........... X12: ST ... SE
   :  :  :                                        EDIFACT: UNH ... UNT
   :  :  :  SEGMENTS, one per line
   :  :  :  +------------------------------------------+
   :  :  :  | ST*850*0001~                             |
   :  :  :  | BEG*00*SA*PO-4471**20260214~             |
   :  :  :  | N1*BY*KESTREL RETAIL*92*000123456~       |
   :  :  :  | ...                                      |
   :  :  :  +------------------------------------------+
   :  :  +--- SE*14*0001~        (segment count, control number)
   :  +------ GE*1*4471~         (transaction set count, group control)
   +--------- IEA*1*000000123~   (group count, interchange control)

   SENDER                                                   RECEIVER
     |  ISA13 = 000000123  ------------------------------->  |  compare with IEA02
     |  GS06  = 4471       ------------------------------->  |  compare with GE02
     |  ST02  = 0001       ------------------------------->  |  compare with SE02
     |                                                        |
     |  <------------------- TA1 (interchange received) -----|  if ISA14 = 1
     |  <------------------- 997 (group accepted/rejected) --|  per GS/GE
     |  <------------------- 999 (guide-level errors) ------|  if agreed
     |  <------------------- CONTRL (EDIFACT) ---------------|  one message, both levels
```

Read the diagram in three directions. **Downwards, the envelope widens:** a segment belongs to a
transaction set, which belongs to a group, which belongs to an interchange, and each trailer
repeats a count. **Outwards, the identifiers chain:** ISA13 must equal IEA02, GS06 must equal
GE02, and ST02 must equal SE02, which is how a receiver knows the document it read is the one
that was sent. **Backwards, acknowledgements retrace the path:** a 997 references the group
control number, a TA1 references the interchange, and a CONTRL can address all three levels.
Treating those granularities as interchangeable is the single most common EDI semantic error.

> **Note:** The envelope is also the unit of atomicity. A receiver may accept one functional
> group and reject another inside the same interchange. It may accept one transaction set and
> reject the next inside the same group. "The file arrived" therefore proves almost nothing about
> which documents inside it were processed.

### 3.3 The document chain and the decision at each link

```text
   BUYER (retailer)                                    SELLER (supplier)
   ----------------                                    ----------------
   850 purchase order        ----------------------->
                             <-----------------------  855 acknowledgment
                                                        (accept / change / backorder)
                             <-----------------------  856 ship notice
   goods physically arrive   <=======================   (the only physical truth)
                             <-----------------------  810 invoice
   820 payment order         ----------------------->

   SEPARATELY, THE REPLENISHMENT LOOP
   852 product activity      <-----------------------   (POS + on-hand, retailer to seller)
   846 inventory advice      ----------------------->   (availability, seller to buyer)
   940 warehouse order       ----------------------->   3PL
   945 warehouse advice      <-----------------------   3PL
```

The chain is not a sequence of copies of one fact. Each document is a different decision
artefact, owned by a different party, with a different failure mode.

| Document | EDIFACT analogue | Originated by | Decision it serves | What it is evidence of | What it is not evidence of |
|---|---|---|---|---|---|
| 850 Purchase Order | ORDERS | Buyer | What to buy, how much, by when | Buyer intent at send time | Seller acceptance |
| 855 PO Acknowledgment | ORDRSP | Seller | Whether the order is accepted, changed, or backordered | Seller's commercial response | Shipment |
| 856 Ship Notice/Manifest | DESADV | Seller or shipper | What to expect at the dock, and when | Declared shipment content | Physical arrival or count |
| 810 Invoice | INVOIC | Seller | What is owed and on what terms | A claim for payment | Receipt or acceptance |
| 846 Inventory Inquiry/Advice | INVRPT | Either party | Whether stock is available to promise | Reported position at report time | Continuous stock level |
| 852 Product Activity Data | SLSRPT | Buyer or retailer | What sold, what is on hand, what is in transit | Sender's reported activity for a population | Total market demand |
| 940 Warehouse Shipping Order | (no standard UN code; sector profile) | Depositor | What the warehouse should ship | Instruction to the 3PL | Movement |
| 945 Warehouse Shipping Advice | (no standard UN code; sector profile) | Warehouse | What the warehouse actually shipped | Warehouse confirmation | Carrier delivery |

Two rows deserve a caution. First, the EDIFACT warehouse pair has no UN-sanctioned message code.
Vendors and sector profiles use names such as WHSORD and WHSRSP, and some catalogues map WHSORD
to the 940, but the official UN/EDIFACT directories list warehouse-family messages such as
WHSREC, WHINRP, and WHIADV instead. If a partner sends you a "WHSRSP," verify the specification
against that partner's guide rather than against the standard. Second, the 852 and 846 are often
described as "real-time inventory" in vendor material. They are reports. Their period, population,
and qualifier set are chosen by the sender.

### 3.4 The acknowledgement chain certifies less than you think

There are four acknowledgement mechanisms in common use, and they answer four different
questions. None of them answers "did the other party agree to the commercial content?"

| Layer | Document | Scope | Certifies | Does not certify | Typical latency |
|---|---|---|---|---|---|
| Interchange | TA1 (X12) | ISA/IEA | Envelope received and control number readable | That any group inside parsed | Seconds to minutes |
| Group | 997 (X12) | GS/GE | Group and its transaction sets parsed; accept/reject per set and per segment | Business acceptance; guide conformance | Minutes to hours |
| Guide | 999 (X12) | GS/GE, against a TR3 | Syntactic and relational conformance to an implementation guide | Application-level validity; commercial acceptance | Minutes to hours |
| Both levels | CONTRL (EDIFACT) | UNB/UNZ, UNG/UNE, UNH/UNT | Technical and functional receipt in one message | Agreement; business response | Minutes to hours |

The business answer arrives in a different document: an 855, an ORDRSP, or a 990 (X12 Response
to a Load Tender, in transport). The distinction is not pedantic. A trading partner can accept
every byte you sent and still reject the order. If your integration treats 997 as confirmation,
you have automated the wrong event.

### 3.5 Four truths, and which one you are allowed to use

For any physical fact, there are up to four versions of it, and they disagree in predictable
ways.

| Truth | Source | Lag | Bias | Legitimate use |
|---|---|---|---|---|
| Declared | 856 | Hours before arrival | Sender's belief; may be stale after repack | Receiving preparation, dock scheduling |
| Counted | Receiving scan against SSCC | At arrival | Human and scanner error; only sampled | Inventory posting, discrepancy claims |
| Billed | 810 | Hours to days after ship | Seller's commercial interest | Payment, not quantity |
| Reported | 846 / 852 | Daily or weekly | Sender's population and period choices | Planning, with stated caveats |

The discipline this topic teaches is simple to state and hard to practise: **name the truth you
are using, and never let a downstream model silently substitute one for another.** A receipt
quantity is not a shipped quantity. A shipped quantity is not an ordered quantity. An ordered
quantity is not demand.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| EDI | Electronic data interchange: the exchange of structured business documents between organisations in a standard machine-readable format | Used to mean any integration or any file transfer | X12 or UN/EDIFACT syntax |
| ANSI ASC X12 | The North American EDI standard, maintained by the Accredited Standards Committee X12 under the American National Standards Institute | Treated as the global standard | Version such as 004010 or 005010 |
| UN/EDIFACT | United Nations rules for Electronic Data Interchange for Administration, Commerce and Transport, the international EDI standard maintained by UN/CEFACT | Treated as interchangeable with X12 | Directory such as D96A, D01B, D21A |
| GS1 EANCOM | The GS1 subset of UN/EDIFACT for retail and consumer goods, restricted to GS1 identifiers and a defined message set | Treated as a separate standard rather than a subset | EANCOM edition and syntax version |
| Transaction set | One complete business document inside an X12 functional group | Confused with a file or an interchange | Three-digit code, for example 850 |
| Segment | One line of an EDI document, made of a tag and a sequence of elements | Called a field or a row | Three-character tag, for example BEG |
| Data element | One value inside a segment, identified by position | Called a column | Position number, for example BEG02 |
| Qualifier | A code that tells the receiver how to interpret the value that follows or precedes it | Ignored as boilerplate | Code from a partner or standard list |
| Envelope | The header and trailer segments that wrap and identify a document or group | Confused with the message content | ISA/GS/ST or UNB/UNH |
| Interchange | The outermost X12 or EDIFACT envelope, one transmission between two parties | Used loosely to mean "a file" | ISA ... IEA, or UNB ... UNZ |
| Functional group | A set of transaction sets of the same type inside one interchange | Assumed to be one document | GS ... GE, or UNG ... UNE |
| Control number | A sender-assigned identifier repeated in the matching trailer | Treated as a database key with meaning | ISA13 (9 digits), GS06 (1–9 digits), ST02 |
| Implementation guide | A trading partner's specification of exactly which segments, elements, codes, and lengths it will accept for a transaction set | Assumed to be the standard | Versioned document, often called an IG or TR3 |
| 850 | X12 Purchase Order | Called "the order file" | Transaction set 850 |
| 855 | X12 Purchase Order Acknowledgment | Assumed to be automatic | Transaction set 855 |
| 856 | X12 Ship Notice/Manifest, the advance ship notice (ASN) | Treated as proof of delivery | Transaction set 856 |
| 810 | X12 Invoice | Treated as proof of payment | Transaction set 810 |
| 846 | X12 Inventory Inquiry/Advice | Treated as a live stock position | Transaction set 846 |
| 852 | X12 Product Activity Data | Treated as total demand | Transaction set 852 |
| 940 | X12 Warehouse Shipping Order | Treated as a shipment | Transaction set 940 |
| 945 | X12 Warehouse Shipping Advice | Treated as proof of delivery | Transaction set 945 |
| ORDERS | EDIFACT purchase order message | Mapped one-to-one to 850 without checking the guide | UN/EDIFACT message ORDERS |
| ORDRSP | EDIFACT purchase order response message | Assumed to carry the same codes as 855 | UN/EDIFACT message ORDRSP |
| DESADV | EDIFACT despatch advice, the EDIFACT ASN | Treated as identical in structure to 856 | UN/EDIFACT message DESADV |
| INVOIC | EDIFACT invoice message | Assumed to be tax-compliant by itself | UN/EDIFACT message INVOIC |
| INVRPT | EDIFACT inventory report message | Treated as a real-time balance | UN/EDIFACT message INVRPT |
| SLSRPT | EDIFACT sales data report message | Treated as a POS feed by definition | UN/EDIFACT message SLSRPT |
| 997 | X12 Functional Acknowledgment: confirms receipt and syntactic processing of a functional group | Mistaken for business acceptance | Transaction set 997 |
| 999 | X12 Implementation Acknowledgment: reports conformance to an implementation guide | Assumed to replace the 997 in all sectors | Transaction set 999 |
| TA1 | X12 Interchange Acknowledgment, returned at the ISA/IEA level | Assumed to be generated automatically by every partner | Segment, not a transaction set |
| CONTRL | EDIFACT syntax and service report, the EDIFACT acknowledgement message | Assumed to have the same accept/reject vocabulary as 997 | UN/EDIFACT message CONTRL |
| VAN | Value-added network: a third-party EDI mailbox and routing service | Assumed to validate business content | Provider and mailbox identifier |
| AS2 | Applicability Statement 2: signed, encrypted HTTP transport for EDI with a non-repudiation receipt | Confused with the EDI format | RFC 4130 |
| Translator | Software that converts between internal formats and EDI syntax using a map | Assumed to enforce the partner's guide | Product and map version |
| Map variant | A partner-specific translation map for one transaction set | Assumed to be one map per document type | Map name and version |
| SSCC | Serial Shipping Container Code: the GS1 identifier for a logistics unit, usually a pallet or carton | Confused with a tracking number | 18-digit GS1 identifier |
| GTIN | Global Trade Item Number: the GS1 identifier for a product | Confused with a supplier part number | 8, 12, 13, or 14 digits |
| GLN | Global Location Number: the GS1 identifier for a party or place | Confused with an internal location code | 13-digit GS1 identifier |
| UOM | Unit of measure: the unit in which a quantity is expressed | Assumed to be consistent across partners | Code such as CA (case), EA (each), PLT (pallet) |
| Chargeback | A financial penalty a buyer deducts from a supplier's invoice for non-compliance with a programme requirement | Treated as a data error rather than a commercial event | Currency amount plus reason code |
| OS&D | Overage, shortage, and damage: the receiving discrepancy taxonomy | Used to mean only shortage | Discrepancy category |
| VMI | Vendor-managed inventory: the supplier decides replenishment using buyer-supplied stock and sales data | Confused with consignment | Programme type |
| Sell-through | Units sold from the retailer to the end customer, as distinct from units sold by the supplier to the retailer | Confused with shipments | Units per period per location |
| Three-way match | The control that compares purchase order, receipt, and invoice before payment | Treated as an accounting detail only | Match outcome per line |

Four of these deserve expansion, because they cause the most damage when confused.

**Qualifier.** A qualifier is a pointer, and the value beside it is meaningless without the
pointer. `DTM*002*20260214` means a delivery-requested date; `DTM*010*20260214` means a
requested ship date. The same date with a different qualifier is a different fact. When a
mapping drops or reorders a qualifier, the value survives and the meaning does not. This is the
most common silent corruption in EDI-derived data, and it is invisible to a schema check because
both rows have the same type and length.

**Implementation guide versus standard.** The standard says a segment is optional and may repeat
up to 100 times. The guide says it is mandatory, may not repeat, and must contain a code from a
list of four. Both statements are true, and only the second one governs your integration. An
analyst who tests data against the standard will pass files that the partner will reject, and
will accept files that the partner would never send.

**997 versus 855.** A 997 is generated by a translator and usually automated. An 855 is generated
by a business process and usually is not. The 997 arrives in seconds and says "I parsed it." The
855 arrives in hours or days and says "I will supply 400 of the 480 you asked for, on the 19th
instead of the 14th." An organisation that monitors 997s and not 855s has automated the easier
half of the conversation and left the commercially important half unobserved.

**ASN versus delivery.** The 856 is a declaration before the fact. It is the trigger for
receiving, not the record of receipt. Systems that post inventory on 856 receipt rather than on
physical receipt create stock that does not exist, and they do so with a clean audit trail,
because every document says the goods arrived.

## 5. The mechanics, step by step

This section follows one purchase order from the buyer's planning decision to the seller's cash,
then follows the separate replenishment loop. At each step the question is the same: what does
the artefact prove, and to whom?

### Phase A — the business event becomes a document

**Step 1. A decision becomes an order, is extracted, and is mapped.**

A planner releases a purchase order in an enterprise resource planning (ERP) system. A batch job
flattens it to one row per line with repeated header fields, and a translator applies the
partner's implementation guide, emitting segments with their qualifiers, codes, and lengths.

*What breaks:* three things, in order. The order may already be a negotiated artefact, rounded to
a pallet or inflated by a promotion, and the 850 will carry that artefact as if it were demand.
The extract may omit a field the ERP holds, which no downstream map can recover. And the map may
lag a partner's code-list change, so the translator emits a code that will be rejected. The 997
reports that rejection, but only if somebody reads it.

### Phase B — the envelope and the transport

**Step 4. The envelope is assembled.**

The X12 envelope has three levels, and each level carries a control number that the matching
trailer must repeat.

| Level | Header | Key elements | Trailer | Matching rule |
|---|---|---|---|---|
| Interchange | ISA | ISA05/06 sender qualifier and ID, ISA07/08 receiver, ISA09/10 date and time, ISA11 repetition separator (005010), ISA12 version, ISA13 control number, ISA14 acknowledgment requested, ISA15 test/production, ISA16 component separator | IEA | ISA13 = IEA02; IEA01 = number of groups |
| Functional group | GS | GS01 functional identifier (PO for 850, SH for 856, IN for 810, FA for 997), GS02/03 application sender and receiver, GS04/05 date and time, GS06 control number, GS07 responsible agency, GS08 version | GE | GS06 = GE02; GE01 = number of transaction sets |
| Transaction set | ST | ST01 transaction set code, ST02 control number | SE | ST02 = SE02; SE01 = number of segments including ST and SE |

Three details are worth memorising because they break parsers:

- **ISA is fixed length.** Every element has a defined width and is space-padded or zero-filled.
  Trimming whitespace before parsing shifts every subsequent position. This is the most common
  cause of "the file worked yesterday" failures after a well-meaning cleanup job.
- **ISA15 distinguishes test from production.** A `T` in a production mailbox means the sender
  believes it is testing. Systems that ignore ISA15 post test orders as real demand, and the
  resulting phantom orders are hard to explain because every document is valid.
- **The separators are declared, not assumed.** ISA16 declares the component separator; ISA11
  declares the repetition separator in 005010. A file using `>` as the segment terminator and `*`
  as the element separator is conventional, not mandated.

EDIFACT uses a parallel structure with different names and one extra optional layer:

| Level | Header | Trailer | Notes |
|---|---|---|---|
| Service string advice | UNA | none | Optional; declares the six separator characters. If absent, defaults apply. |
| Interchange | UNB | UNZ | UNB carries sender, recipient, date/time, control reference, and an acknowledgement-request flag |
| Functional group | UNG | UNE | Optional; many implementations omit it and place messages directly in the interchange |
| Message | UNH | UNT | UNH carries the message reference and the message type with version, for example `ORDERS:D:96A:UN` |

**Step 5. The interchange is transported.**

| Transport | What it is | What it proves | What it does not prove | Typical use |
|---|---|---|---|---|
| VAN | A third-party mailbox and routing service | The message reached a mailbox | That the partner's application read it | Legacy retail and automotive networks |
| AS2 | Signed, encrypted HTTP with a message disposition notification, defined in RFC 4130 | Non-repudiable delivery to the partner's gateway | That the document parsed or was accepted | Most direct partner connections |
| SFTP | A file drop on a shared server | A file was written | Who read it, or whether it was complete | Smaller partners, batch windows |
| API-wrapped EDI | An EDI payload inside an HTTPS request | Delivery and an HTTP status | The same things a 997 would tell you, unless you also parse the EDI ack | Modern middleware front ends |

A transport acknowledgement and an EDI acknowledgement are different objects. An AS2 message
disposition notification says the bytes arrived. A 997 says the bytes parsed. Neither says the
order was accepted. Conflating all three is how organisations end up with an integration that is
green at every monitoring layer while the business relationship is failing.

### Phase C — the acknowledgement chain

**Step 6. The receiver validates and acknowledges at up to three levels.**

The X12 acknowledgement segments are worth knowing because they tell you where to look when
something is missing.

| Document | Structure | What each part reports |
|---|---|---|
| TA1 | One segment | Interchange control number, interchange date and time, interchange acknowledgement code, note code |
| 997 | AK1, AK2, AK3/AK4, AK5, AK9 | AK1 identifies the functional group and its control number; AK2 identifies a transaction set; AK3 reports a segment error with its position and loop; AK4 reports a data-element error; AK5 reports the transaction set accept/reject status; AK9 reports counts and the group status |
| 999 | AK1, AK2, IK3/IK4/IK5, AK9, with CTX where situational | The same shape, but the error codes refer to implementation-guide requirements rather than base syntax. IK3 reports a segment error against the guide; IK4 reports an element error; IK5 reports the transaction set status |
| CONTRL | UCI, UCF, UCM, UCS, UCD, UNT | UCI responds at the interchange level, UCF at the group level, UCM at the message level, UCS at the segment level counting from UNH as position 1, and UCD at the data-element level |

The 999 was introduced in the X12 004061 subrelease and became widely known through the
005010X231 guide for health care insurance, published in 2007. It exists because base X12 syntax
cannot express "this segment is valid X12 but not permitted by this partner's guide." The 997
cannot report that class of error; the 999 can. In retail, the 997 remains far more common.

EDIFACT folds both roles into CONTRL, so a single message can report an interchange-level and a
message-level problem at once. One comparison source states that EDIFACT has no "accepted with
errors" status equivalent to the X12 middle state, so a partially accepted interchange may carry
an overall accepted status with an error code alongside. Confirm the semantics against the
partner's CONTRL guide before building logic on the status code alone.

> **Note:** Never acknowledge an acknowledgement. A 997 must not acknowledge another 997 or a
> 999, and a CONTRL must not trigger a CONTRL. Violating this creates an infinite loop that is
> easy to start and expensive to stop, while both parties' monitors report healthy traffic.

**Step 7. The translator writes to an application interface, and the acknowledgement returns.**

The translator hands the document to order management, the warehouse system, or accounts
payable, and the envelope disappears. The application keeps the document number and control
number and drops the rest, which is why nobody can later say which guide version produced a
field, whether the document was a replacement, or whether it was a test. The 997 then travels
back on the same transport, referencing the group control number rather than the interchange.

*What breaks:* the acknowledgement is generated but not routed, or routed to a mailbox nobody
monitors. The sender's integration reports success because it received an AS2 receipt; the
receiver's reports success because it generated a 997. Both are green and the document is rejected.

### Phase D — the commercial response

**Step 9. The seller responds with an 855 or ORDRSP.**

The 855 reports one outcome per line: accepted as ordered, accepted with a change, backordered,
or rejected. It may also propose a substitute item. It is the first document in the chain to
carry a commercial judgement rather than a technical one.

*What breaks:* the buyer's integration waits for the 997 and treats it as terminal. The 855
arrives into a queue with no consumer, orders that were partially accepted are planned as fully
accepted, and the shortfall appears only when the 856 declares a smaller shipment.

A related pair is easy to miss: the 860 Purchase Order Change and the 865 Purchase Order Change
Acknowledgment. A change is not a new order, and systems that treat every 850 as new will
double-count quantity whenever a buyer amends an order rather than reissuing it.

### Phase E — physical execution

**Step 10. The seller builds the shipment and the 856 declares it.**

The 856 is hierarchical. A typical structure has four levels: shipment, order, pack (a pallet or
carton identified by an SSCC), and item (a GTIN with quantity and unit of measure). Each HL
segment carries a hierarchical ID, a parent ID, and a level code.

| Level | Typical segments | What it declares |
|---|---|---|
| Shipment | HL with level S, TD1, TD5, REF | Total handling units, carrier, bill of lading |
| Order | HL with level O, PRF | Which purchase order the shipment fulfils |
| Pack | HL with level P, MAN | The SSCC of a pallet or carton |
| Item | HL with level I, LIN, SN1 | GTIN, quantity, unit of measure |

*What breaks:* the 856 is generated at pick completion but the shipment is repacked afterwards.
The document then describes a configuration that no longer exists. This is the worked example in
section 6.

**Step 11. The warehouse executes a 940 and confirms with a 945.**

The depositor sends a 940 Warehouse Shipping Order; the warehouse returns a 945 Warehouse
Shipping Advice stating what it actually shipped. The 945 is the warehouse's assertion, not the
carrier's, and not the customer's receipt.

*What breaks:* the 940 is treated as a shipment, so stock leaves the ledger when the order is sent
rather than when the 945 confirms it. The gap is invisible because both documents reference the
same order number.

**Step 12. The goods arrive and are received.**

Receiving compares the physical count with the 856. If the ASN is trusted, the count is a
verification; if the ASN is treated as truth, there is no count at all.

*What breaks:* the receiver posts the ASN quantity, and the discrepancy surfaces at the physical
inventory weeks later as shrinkage.

### Phase F — the money

**Step 13. The invoice is issued and matched.**

The 810 states a quantity and an amount. The buyer performs a three-way match against the
purchase order and the receipt, and releases payment, often through an 820 Payment Order.

*What breaks:* the invoice is generated from the order rather than the shipment or the receipt.
Then all three documents agree with each other and none agrees with the physical world, because
the invoice inherited the order's optimism.

### Phase G — the replenishment evidence

**Step 14. The 852 and 846 close the loop.**

The 852 Product Activity Data reports sales and stock activity from the party closer to the
consumer, usually the retailer. The 846 Inventory Inquiry/Advice reports availability from the
party holding stock, usually the supplier or a distribution centre. Together they are the
evidence base for VMI and for most retail replenishment analytics.

| Question | Document | What to check before trusting it |
|---|---|---|
| What sold to the end customer? | 852 | Reporting period, store population, whether the quantity is units or cases, whether returns are netted |
| What is on hand now? | 846 | Report type (inquiry or advice), timestamp, whether the quantity is on-hand or available-to-promise, whether in-transit is included |
| What is about to arrive? | 856 | Whether the document is a replacement, and whether it was generated before or after final loading |
| What did the warehouse actually ship? | 945 | Whether the quantity is shipped or packed, and whether it is the warehouse's count or the carrier's |

### 5.1 Version and implementation-guide variance

Two documents can both be "an X12 850, version 004010" and share almost no structure. The
standard fixes the grammar; the partner's guide fixes the dialect. Two publicly available guides
illustrate the range.

| Element | A retail general-merchandise guide (4010, VICS subset) | An aerospace guide (4010) | Consequence of one map |
|---|---|---|---|
| BEG02, purchase order type | SA, stand-alone order | NE, new order | Order-type logic selects the wrong branch |
| Product ID qualifier in PO1 | IN, VN, or UP | PN or PD | Item resolution fails or matches the wrong cross-reference |
| Unit price location | PO104 | CTP03 | Price reads as zero and the line is rejected or mispriced |
| Requested date | DTM segment | SCH06 | Date is absent and a default is applied |
| Additional loops | Minimal | N9, MSG, TXI, TD5, FOB | Ignored, or rejected as not used |

The lesson is not that one guide is better. It is that **the map is the unit of integration, not
the standard.** A team that says "we support 850" has said almost nothing. The useful statement
is "we support 43 partner-specific 850 variants, of which 6 are maintained by hand."

Public estimates of how far this goes are weak: one analysis cites 47 map variants for a single
850, and another notes that one retailer may run different schemas across divisions. Both are
practitioner claims rather than measurements (Grade D), but the direction is consistent: variance
grows with the number of partners, not the number of document types. Version is a second axis.
X12 004010 remains widespread in retail, 005010 dominates health care, and EDIFACT directories
such as D96A remain in production decades after publication. A version upgrade is therefore a
programme, not a patch, and it is usually triggered by a partner rather than chosen.

> **Note:** A partner can make an optional segment mandatory, forbid a code the standard allows,
> or change a length. A "standards-compliant" file can therefore fail the partner's validation,
> and a file that fails the standard can pass a lenient partner. Test against the guide.

### 5.2 What actually lands in your warehouse

The analyst rarely sees segments. They see the translator's output: flattened tables, often one
per document type, named like `edi_850_header` or `edi_850_line`. Whatever the shape, six
provenance columns are worth demanding, because without them the data cannot be interpreted later:

| Column | Why it is needed | What goes wrong without it |
|---|---|---|
| Document type and version | A field's meaning depends on it | A 4010 field is read with 5010 semantics |
| Control numbers (interchange, group, set) | Ties the row to the original document and its acknowledgement | Duplicates and replacements cannot be distinguished |
| Partner identifier and guide version | The dialect, not the standard, governs the field | A partner-specific code is read as a standard code |
| Received timestamp and business date | Separates arrival time from effective time | Latency and backdating are invisible |
| Test or production flag | Test traffic must be excludable | Phantom orders enter demand history |
| Replacement or original indicator | The same shipment can have several valid documents | Last-write-wins silently picks the wrong one |

Most EDI-to-warehouse pipelines keep the first two and drop the rest. That is a defensible
short-term choice and an indefensible long-term one, because the information cannot be
reconstructed from the payload after the envelope is gone.

## 6. Worked example

Halden Supply makes household consumables and supplies Kestrel Retail under a VMI programme. The
figures are illustrative and internally consistent; the document behaviour is not hypothetical.

### 6.1 The documents as they arrive

| Time | Document | Content | What it proves |
|---|---|---|---|
| 10 Feb 09:00 | 850 PO-4471 | 480 cases, 12 pallets of 40, requested 16 Feb | Kestrel intends to buy |
| 10 Feb 14:20 | 855 | All 480 accepted, delivery 16 Feb | Halden accepted |
| 14 Feb 04:12 | 856 SHIP-88213 | 12 pallets, 480 cases, 12 SSCCs | Halden believes it shipped 480 |
| 14 Feb 04:13 | 997 from Kestrel | Group accepted | Kestrel's translator parsed the 856 |
| 16 Feb 09:40 | Physical receipt | 456 cases: 11 full pallets and 16 loose cases | What actually arrived |
| 16 Feb 18:00 | 810 | 480 cases at EUR 36.00 = EUR 17,280 | Halden claims payment for 480 |
| 17 Feb 10:00 | Three-way match | PO 480, receipt 456, invoice 480 | A discrepancy exists |

Two things are notable. First, every acknowledgement in the chain was positive. The 997 was
accurate: the 856 parsed. Nothing in the syntax layer was wrong. Second, the discrepancy only
becomes visible at the three-way match, which is a financial control, not an operational one.
By then the goods have been put away and the inventory position has been live for a day.

### 6.2 The arithmetic

```text
ordered                 480 cases
received                456 cases
shortfall                24 cases
shortfall rate           24 / 480 = 5.0%

case cost              EUR 36.00
case pack              12 eaches
retail price per each  EUR 4.33 (EUR 52.00 per case)

value at cost          24 x 36.00 = EUR 864.00
value at retail        24 x 52.00 = EUR 1,248.00

invoice as issued     480 x 36.00 = EUR 17,280.00
correct invoice       456 x 36.00 = EUR 16,416.00
difference                            EUR 864.00
```

The EUR 864 is **not a loss**. It is money that was never owed and is now withheld. An analyst
who reports it as the cost of the discrepancy has double-counted a correction. The costs are the
handling, the investigation, the expedite, and the risk.

### 6.3 The cost that is real

| Cost item | Calculation | Amount (EUR) |
|---|---|---:|
| Retailer handling fee for a receipt discrepancy | fixed per discrepancy, per the programme manual | 75.00 |
| Receiving labour: discrepancy handling and re-scan | 65 minutes at EUR 42.00 per hour loaded | 45.50 |
| Investigation: three people, two hours each | 6 hours at EUR 55.00 per hour | 330.00 |
| Expedite of the 24-case shortfall by courier | 96 kg at EUR 4.20 per kg | 403.20 |
| **Total avoidable cost** | | **853.70** |

The expedite is the largest line and the least certain. It is avoidable only if somebody decides
that 24 cases are worth expediting, which depends on the cover position at the time. In this
case the answer was yes, because the next scheduled delivery was five days away.

### 6.4 The physical and planning consequence

Demand for the item runs at about 60 cases per day. Safety stock is 120 cases, two days of cover.
The reorder point is 360 cases at a four-day lead time.

```text
planned cover after receipt   480 / 60 = 8.0 days
actual cover after receipt    456 / 60 = 7.6 days
cover lost                    0.4 days, about 9.6 hours

safety stock                  120 cases
buffer consumed               24 / 120 = 20.0%
```

No stock-out occurred. That is the ordinary case, and it is worth saying plainly: most ASN
discrepancies cost handling and trust, not sales. The ones that matter are the ones that arrive
when the buffer is already thin, and those are exactly the ones a supplier scorecard will not
predict from the discrepancy rate alone.

The more damaging effect is informational. Kestrel's 852 shows sell-through unchanged at 60
cases per day. Halden's 846 shows on-hand 24 cases lower than planned. Halden's replenishment
model sees normal demand and lower stock, and orders earlier. The model is behaving correctly
given its inputs; the inputs describe a supply position that was created by a document error.
The result is a small, permanent oscillation in order timing that no single report explains.

### 6.5 Root cause

The 856 was generated when picking was complete at 03:40. Between picking and sealing at 05:20,
two pallets were found crushed and the load was reworked into eleven pallets plus sixteen loose
cases. The shipping system was not re-run. The ASN therefore described the pre-rework load.

Three design choices would have prevented it:

1. **Generate the 856 at seal, not at pick.** The document should describe the load that left,
   not the load that was planned.
2. **Treat a post-seal change as a replacement document.** The 856 header carries a purpose code
   that distinguishes an original from a replacement. The exact code values belong to the
   partner's guide. The receiver must apply a last-valid-document rule keyed on that code and the
   document timestamp, not on arrival order, because messages can arrive out of order.
3. **Make receiving count.** The ASN should set the expectation and pre-build the receipt lines.
   The scan should set the quantity. A receiving process that posts the ASN quantity is not
   receiving; it is data entry with extra steps.

### 6.6 Lessons

| Lesson | Generalisation |
|---|---|
| The ASN described a load that no longer existed | A document generated before the physical event is a forecast, not a record |
| Every acknowledgement was positive | Syntax acknowledgements cannot detect semantic error |
| The invoice inherited the order quantity | Financial documents inherit whichever operational document they are built from |
| The planning model behaved correctly on wrong inputs | Model quality is bounded by provenance, not by algorithm |
| The cost was mostly administrative | Do not overstate the physical impact of a discrepancy, and do not understate the handling cost |

## 7. Code, query, or calculation

One excerpt, read closely, teaches more than a parser tutorial. Below is the first pack and item
of the 856 from section 6, with the envelope intact. It is data, not code: a real document would
repeat the pack and item loops once per pallet.

```text
ISA*00*          *00*          *ZZ*HALDENSUPPLY   *ZZ*KESTRELRT      *260214*0412*^*00501*000000123*0*P*>~
GS*SH*HALDENSUPPLY*KESTRELRT*20260214*0412*4471*X*005010~
ST*856*0001~
BSN*00*SHIP-88213*20260214*0412*0001~
HL*1**S~
TD1*PLT*12~
TD5**2*ROADFEED~
REF*BM*BOL-4471~
HL*2*1*O~
PRF*PO-4471~
HL*3*2*P~
MAN*GM*001234567890123456~
HL*4*3*I~
LIN**UP*012345678905~
SN1**40*CA~
CTT*4~
SE*15*0001~
GE*1*4471~
IEA*1*000000123~
```

Read element by element, because this is where meaning lives.

| Segment | Element | Value | Meaning | Trap if misread |
|---|---|---|---|---|
| ISA | ISA05 / ISA06 | ZZ / HALDENSUPPLY | Sender ID qualifier is mutually defined | A DUNS qualifier (01) would change the lookup |
| ISA | ISA11 | `^` | Repetition separator in 005010 | In 004010 this position is the standards identifier |
| ISA | ISA13 | 000000123 | Interchange control number, 9 digits | Must match IEA02 |
| ISA | ISA15 | P | Production traffic | A `T` here means test |
| GS | GS01 | SH | Functional identifier for a ship notice | PO, IN, and FA identify other document types |
| GS | GS06 | 4471 | Group control number | Must match GE02 |
| ST | ST01 / ST02 | 856 / 0001 | Transaction set type and control number | ST02 must match SE02 |
| BSN | BSN01 | 00 | Purpose code: original | A replacement uses a different code |
| BSN | BSN02 | SHIP-88213 | Shipment identification | The key for last-valid-document logic |
| BSN | BSN05 | 0001 | Hierarchy: shipment, order, pack, item | A different code declares a different hierarchy |
| TD1 | TD101 / TD102 | PLT / 12 | Twelve pallets | A declared total, not a counted one |
| TD5 | TD503 | ROADFEED | Carrier SCAC | Not a tracking number |
| REF | REF01 / REF02 | BM / BOL-4471 | Bill of lading number | BM is the qualifier that makes the value a BOL |
| HL | HL01 / HL02 / HL03 | 2 / 1 / O | Hierarchy ID 2, parent 1, level order | A missing HL02 orphans the loop |
| PRF | PRF01 | PO-4471 | Purchase order number | Ties the shipment to the order |
| HL | HL01 / HL02 / HL03 | 3 / 2 / P | Pack level, child of the order | The pack is what carries an SSCC |
| MAN | MAN01 / MAN02 | GM / 001234567890123456 | SSCC-18 for the pallet | GM is the qualifier; the value alone is ambiguous |
| LIN | LIN02 / LIN03 | UP / 012345678905 | UPC as the item identifier | Shifting the qualifier one position destroys the key |
| SN1 | SN102 / SN103 | 40 / CA | Forty cases | CA is case, not each |
| CTT | CTT01 | 4 | Four HL loops in this message | Must equal the actual loop count |
| SE | SE01 / SE02 | 15 / 0001 | Fifteen segments including ST and SE | A mismatch proves truncation or insertion |

Three parsing rules follow from the excerpt, and each has a failure attached:

- **The envelope is positional and fixed-width; the body is delimited.** Trim the ISA and every
  following element shifts. Parse the ISA by position and the body by separator, never the other
  way round.
- **The release character escapes separators inside data.** If a value legitimately contains
  `*`, `>`, or `^`, the sender prefixes it with the release character declared in the interchange.
  A naive `split("*")` corrupts that value and often the rest of the segment.
- **Counts are controls, not metadata.** SE01, GE01, IEA01, and CTT01 all assert a count. A parser
  that ignores them will accept a truncated file. A parser that enforces them will reject an
  otherwise usable file whose sender miscounted, which is why many production systems log the
  mismatch rather than reject.

The same item line in EDIFACT would read `LIN+1++012345678905:UP'` followed by `QTY+12:40'`, where
the colon separates the composite's value from its qualifier and `12` is the despatch quantity
qualifier. The positions change; the qualifier logic does not.

## 8. Data, standards, and regulatory context

### 8.1 The standards families

| Standard | Owner | Scope | What it fixes | Grade |
|---|---|---|---|---|
| ASC X12 | Accredited Standards Committee X12, ANSI | North American business documents | Segment grammar, transaction set definitions, envelope | B |
| UN/EDIFACT | UN/CEFACT | International business documents | Message grammar, service segments, directories | B |
| GS1 EANCOM | GS1 | Retail and consumer goods subset of EDIFACT | Message set, GS1 identifiers, restricted code lists | B |
| GS1 EDI XML | GS1 | XML alternative to EANCOM | Same business content in XML | B |
| Peppol BIS | OpenPeppol | European e-procurement documents | Business interoperability specifications on EN 16931 | B |
| EN 16931 | CEN | European e-invoice semantic model | Semantic model, not a syntax | B |
| AS2 | IETF, RFC 4130 | Transport of EDI over HTTP | Signing, encryption, receipt | B |

The two syntax families are not translations of one another. X12 and EDIFACT were developed
independently, and their differences are structural, not cosmetic.

| Dimension | X12 | EDIFACT |
|---|---|---|
| Service segments | ISA/GS/ST and trailers | UNA (optional), UNB/UNG/UNH and trailers |
| Group layer | GS/GE is standard | UNG/UNE is optional |
| Qualifier placement | Often a preceding qualifier element, for example REF01 | Often a composite with an explicit type, for example `value:qualifier` |
| Acknowledgement | TA1, 997, 999 as separate documents | CONTRL as one message covering both levels |
| Character set | Historically uppercase and a restricted set | Broader repertoire, declared in UNB |
| Decimal notation | Implied by the data element definition | Declared in UNA |

### 8.2 Versions and directories

| Family | Common in production | Notes | Grade |
|---|---|---|---|
| X12 004010 | Retail, manufacturing, logistics | Still widespread decades after publication | C |
| X12 005010 | Health care, newer implementations | The base for the 999 guide | B |
| X12 006020 and later | Limited | Adoption is uneven and partner-driven | C |
| EDIFACT D96A | Retail and general trade, often through EANCOM | Long-lived; many guides still target it | B |
| EDIFACT D01B | EANCOM 2002 and later editions | Paired with EANCOM syntax versions | B |
| EDIFACT D21A, D23A | Newer implementations | Current UNECE directories | B |

A version is a property of the interchange, not of the business relationship. Two partners can
run different versions for different document types, and often do: a 4010 850 may coexist with a
5010 856 from the same sender.

### 8.3 Transport and acknowledgement standards

AS2 is defined in RFC 4130 and provides signing, encryption, and a message disposition
notification. It does not define the EDI payload. A VAN is a commercial service, not a standard,
and its guarantees are contractual. SFTP is a file transfer protocol with no EDI semantics at
all.

The acknowledgement layer has no single global standard. X12 defines TA1, 997, and 999; EDIFACT
defines CONTRL. Which acknowledgements are mandatory is a matter of trading-partner agreement,
not of the standard. This is worth repeating because it is the most common source of false
confidence in an integration review: **the presence of a 997 in the specification does not mean
the partner will send one, and its absence does not mean the document failed.**

### 8.4 Regulatory interfaces

| Requirement | Jurisdiction | Relationship to EDI | Grade |
|---|---|---|---|
| HIPAA transactions and code sets | United States, health care | Mandates X12 for named transactions and drove the 999 guide | B |
| DSCSA | United States, pharmaceuticals | Product tracing has used EDI-derived transaction data; interoperable exchange pushes toward event data | B |
| EU e-invoicing and ViDA | European Union | EN 16931 and Peppol BIS define the semantic model; EDI can carry it but is not the model | B |
| Country e-invoicing mandates | Italy, Spain, France, others | Real-time or near-real-time reporting to tax authorities, often outside EDI | B |
| GS1 standards | Global, voluntary | GTIN, GLN, SSCC are contractually required by most retailers | B |

The regulatory trend is toward event-level and near-real-time reporting, which does not replace
EDI for orders and shipments. It adds a second channel with a different latency and a different
legal owner. An analyst building a single "orders" table from both channels is joining two
regimes with different definitions of when a transaction exists.

### 8.5 What is convention rather than requirement

A 997 for every document is a partner convention, not a standard rule. A 24-hour 856 window is a
retailer programme requirement, not an EDI rule. The `*` element separator and `>` segment
terminator are conventional, not mandated. Daily 852 files are a retail habit, not a property of
the message. And "one map per document type" is an organisational simplification: the technical
reality is one map per partner per document per version.

An analyst who can separate the standard from the convention can tell which "requirements" are
negotiable and which are structural. That distinction is worth more in an integration review
than familiarity with any particular translator.

## 9. Industry practice and case evidence

### 9.1 How large EDI actually is

| Claim | Figure | Date | Source type | Grade | How to use it |
|---|---|---:|---|---|---|
| Share of global B2B transactions | about 78% | 2024, recycled 2025 | Single vendor blog, repeated across marketing content | D | Directional only; do not cite as a measurement |
| Annual EDI transactions | 20 billion or more | 2024 | Same vendor origin | D | Treat as an order-of-magnitude claim |
| Value of EDI-mediated trade | about USD 6.4 trillion | 2024 | Same vendor origin | D | Not independently verifiable |
| Transactions on one network | 31 billion per year | 2025 vendor keynote | Vendor | D | Shows scale of one provider, not the market |
| Italian companies exchanging EDI | 8,042 companies, about 55 million messages in 2024 | 2025 monitoring report | Industry association with a university | C | The most methodologically described figure found |
| Italian EDI-mediated trade | over EUR 62 billion in 2024 | 2025 | Same monitoring report | C | Consistent with the message count |
| Italian adoption rate | about 25% of the relevant population | 2025 | Same monitoring report | C | Large enterprises well ahead of small and medium ones |
| Canadian B2B EDI volume | about 5.16 million transactions, about CAD 228 billion, 2024 | 2025 | Consultancy citing payments data | C | High value per transaction; legacy infrastructure |

The 78% figure deserves a warning. It appears in dozens of vendor pages, almost always without a
method, a sample, or a definition of "B2B transaction." The trail leads back to a single vendor
blog. That does not make it false; it makes it unusable as evidence. If you need a number for a
business case, use a figure with a described method, state its population, and date it.

The GS1 Italy monitoring is more useful precisely because it describes its population and
reports a decline: messages fell about 6% year on year even as the number of participating
companies held steady. A market that is flat to declining in message volume while vendor
material describes explosive growth is a market where portals and web forms are absorbing some
small-partner traffic. That nuance is invisible in the headline claims.

### 9.2 Why EDI persists

Five structural reasons, in rough order of importance:

1. **Mandate.** Large buyers require it. Public supplier requirements from major retailers and
   marketplaces specify EDI documents, timelines, and chargebacks. A supplier that prefers APIs
   does not get to choose.
2. **Obligation and audit.** An EDI document is a dated, control-numbered, retained assertion
   that can be produced in a dispute. An API call is a request whose response may not be stored.
3. **Batch alignment.** Planning runs on daily or weekly cycles. A document that arrives once per
   cycle matches the decision cadence, and its completeness is checkable in a way that a stream
   is not.
4. **Cost at volume.** For high-volume, stable, structured documents, EDI transport and
   translation are cheap per transaction and well understood by a large labour pool.
5. **Inertia with a reason.** Replacing a working interface requires both parties to change, coordinate,
   and retest. The expected benefit rarely exceeds the coordination cost for a document that
   already works.

The realistic 2026 pattern is not EDI versus APIs. It is **API at the edge, EDI in the core**:
a modern portal or API for onboarding, visibility, and exceptions, with the obligation-bearing
documents still flowing as EDI underneath. That pattern is why an analyst who cannot read EDI
will keep finding a layer of the business that the API does not expose.

### 9.3 Evidence on ASN and receiving performance

| Finding | Detail | Source type | Grade | Caveat |
|---|---|---|---|---|
| ASN-enabled receiving reduces cost | about 40% reduction cited | Industry traceability initiative guide, 2024 | D | Vendor and programme advocacy; no method shown |
| ASN failures cluster | Late or missing transactions, and master-data inaccuracy | Supplier and retailer survey, 79 suppliers and 20 retailers, 2022 | C | Self-selected respondents |
| Label integrity is the physical bottleneck | Torn or missing labels are the largest single label defect | Same survey | C | Qualitative |
| Carton-level SSCC is mandatory for some channels | Required on every carton since 2014 in one marketplace programme | Public supplier requirements | C | Applies to that channel only |
| ASN timing windows are enforced | Within 30 minutes of departure or 6 hours before appointment, in one programme | Public supplier requirements | C | Chargeback-driven compliance |

The pattern across these sources is consistent with the worked example: the document is necessary
and not sufficient. A clean 856 with an unreadable pallet label becomes a manual receipt. A
perfect label with a late 856 becomes a receiving queue. Automation only removes manual counting
when the document, the label, and the master data all hold at once.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| 997 read as business acceptance | Order marked confirmed before any 855 arrives | Acknowledgment layer conflated with commercial response | Orders planned as accepted, then partially shipped | Track 855 separately; alert on orders with no 855 after an agreed window |
| ASN not revised after rework | 856 quantity matches the pick, not the load | Document generated before the physical event | Receipt reconciles to a shipment that never existed | Generate at seal; support a replacement purpose code |
| ASN quantity posted as receipt | Receiving scan rate near zero per pallet | ASN treated as truth rather than expectation | Inventory overstated or understated until physical count | Scan every SSCC; post counted quantity |
| Several 856s for one shipment | Duplicate receipt lines or a quantity that flips | Last-write-wins by arrival order, not by document timestamp | Double counting or silent loss | Key on shipment ID plus purpose code plus timestamp |
| Missing or unreadable SSCC | Manual keying at the dock; receiving time per pallet rises | Label damaged in transit or not applied at carton level | Automation collapses to manual receipt | Enforce carton SSCC; verify labels; keep a redundant scan path |
| Control-number gaps unmonitored | No reconciliation report exists | Missing interchange is invisible without sequence checking | Documents silently absent from history | Daily gap report per partner per direction |
| Unit-of-measure mismatch | Quantities differ by a factor near 6, 12, or 24 | CA, EA, and PLT confused across guides | Inventory and invoices wrong by the pack factor | Validate UOM against the item master, not the document |
| Qualifier shifted or dropped | Values present, meaning wrong; schema tests pass | Positional parse after a guide change | Dates and identifiers silently reinterpreted | Assert qualifier values explicitly; reject unknown ones |
| Test traffic in production | Small, oddly round, or duplicated orders | ISA15 ignored by the receiver | Phantom demand enters history | Filter on ISA15 and on partner test IDs |
| 852 population or period change | Weekly total drops without a matching sell-through drop | Regional extract failure or a changed reporting window | Zero demand inferred for missing stores | Require store count in the file; compare to the prior period |
| 846 treated as live position | Replenishment reacts to a snapshot as if continuous | Report timestamp ignored | Order timing oscillates | Store the report timestamp; treat as an observation, not a balance |
| Guide change without map update | Sudden rejections after a partner portal announcement | Partner adds a mandatory segment or code | Document rejection, or silent truncation | Subscribe to guide change notices; diff and retest quarterly |
| 940 or 945 treated as delivery | Inventory leaves the ledger before the carrier moves | Warehouse documents conflated with transport events | Inventory understated in transit | Separate warehouse confirmation from delivery confirmation |

Two of these are worth a note. **Test traffic in production** is rare but spectacular: a partner's
test team points at the production mailbox, a few hundred test orders enter the demand history,
and every forecast learns a spike. A filter on ISA15 takes one line and is omitted too often.
**Unit-of-measure mismatch** is the most expensive per incident because it scales. A factor-of-12
error on a case-versus-each confusion does not produce a small discrepancy that somebody
investigates; it produces an inventory position wrong by an order of magnitude, and the first
person to notice is usually a customer.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The partner sends every document the agreement requires | Completeness is inferred from the counterparty's behaviour | Silent omissions, changed programmes | Compare document counts to the agreement; gap-check control numbers |
| Control numbers are sequential and unique per sender | Gap detection depends on it | Non-sequential assignment, resets after system migration | Plot the sequence; look for resets and duplicates |
| The implementation guide in force is the one you hold | Field semantics depend on the guide | Unannounced guide revision | Log guide version per partner; diff on receipt of a new guide |
| The 856 describes the load that left | Receiving pre-builds from it | Post-pick rework, split shipments | Compare declared versus counted by pallet, not by total |
| SSCCs are unique and scannable | Carton-level automation depends on it | Reused labels, damaged labels, missing carton labels | Scan-rate and read-rate monitoring at the dock |
| 852 quantities are end-customer sales, net of returns | It is used as demand | Sender reports shipments, or gross sales, or does not net returns | Read the guide's quantity qualifiers and return handling |
| 846 quantities share a definition across partners | Replenishment compares them | On-hand versus available-to-promise versus in-transit | Compare the report type and qualifiers, not the column name |
| Item identifiers join to your master data | Every analytic join depends on it | Partner uses a vendor part number or a different GTIN form | Measure unmatched-line rate weekly |
| Acknowledgements are monitored by a human or an alert | The chain only works if somebody looks | Green dashboards with no business metric | Reconcile acks to documents; alert on absence |
| Retained documents are retrievable for the audit period | Disputes are settled from the record | Translator purges after load | Confirm retention with the integration owner, in writing |

Two limits are worth stating. **This describes the document layer, not the whole integration:**
master data synchronisation, onboarding, and chargeback management are adjacent disciplines, and a
clean mapping on top of bad item master data still produces unmatched lines. **It is written from
the receiving analyst's side:** suppliers see the same documents with different incentives, since
an 850 is revenue, an 855 is a commitment, and a chargeback is a cost of doing business. The
mechanics are symmetric; the interpretation is not.

## 12. Exercises

### Exercise 1 — Map documents to decisions (application, 45 minutes)

Choose a supply chain you know. For each of the eight core documents, name the decision it
triggers, the role that acts on it, and the failure that occurs if it is missing.

**Deliverable:** a table with eight rows, each naming the decision, the acting role, and the failure if it is missing.
**Acceptance criteria:** decisions name a role acting, not a department; two rows identify a silent default.

### Exercise 2 — Acknowledgement coverage audit (calculation, 40 minutes)

A supplier sends 1,200 documents in a month. It receives 1,180 997s, of which 14 report a
rejection. It sends 1,100 orders and receives 900 855s. Compute the syntax acceptance rate, the
acknowledgement coverage rate, and the business acknowledgement rate. State which number an
integration dashboard would most likely display.

**Deliverable:** three rates with numerators and denominators, plus one sentence on what the dashboard would hide.
**Acceptance criteria:** correct arithmetic; "parsed" versus "accepted" is explicit; the hidden number is named.

### Exercise 3 — Read the excerpt (diagnosis, 40 minutes)

The following 856 excerpt contains three defects. Identify each, name the element or segment
involved, and state what a lenient downstream parser would wrongly infer.

```text
ST*856*0001~
BSN*00*SHIP-88213*20260214*0412*0001~
HL*1**S~
TD1*PLT*12~
HL*2*1*O~
PRF*PO-4471~
HL*3**P~
MAN*GM*001234567890123456~
HL*4*3*I~
LIN*UP*012345678905~
SN1**40*CA~
CTT*4~
SE*14*0001~
```

**Deliverable:** three findings, each with its segment, its defect, and the wrong inference a lenient parser makes.
**Acceptance criteria:** one finding concerns a hierarchy pointer, one a qualifier position; the segment count is checked and stated.

### Exercise 4 — Discrepancy arithmetic (calculation, 45 minutes)

An 856 declares 18 pallets, 720 cases, and 24 SSCCs. The dock counts 17 pallets and 684 cases.
Case cost is EUR 41.50, case pack is 6 eaches, and retail price is EUR 9.95 per each. Daily demand
is 12 cases.

**Deliverable:** case shortfall and percent, value at cost and at retail, days of cover lost, correct invoice amount, and the SSCC-to-pallet inference.
**Acceptance criteria:** every figure carries units; correction versus cost is stated; the SSCC count is interpreted, not ignored.

### Exercise 5 — Control-number reconciliation (design and calculation, 50 minutes)

A receiver logs ISA13 values from 000004100 to 000004199 inclusive. It finds 97 distinct
interchanges, of which 2 are duplicates. Design a daily control and compute the gap rate. Assume
14 documents per interchange, 30% of documents are orders, and average order value is EUR 18,000.

**Deliverable:** gap count, gap rate, duplicate rate, unaccounted document count, and a control design with an alert threshold.
**Acceptance criteria:** the inclusive range is handled correctly; exposure is labelled exposure, not loss; the threshold is justified.

### Exercise 6 — Two-guide variance diff (analysis, 60 minutes)

Obtain two public implementation guides for the same transaction set from different sectors.
Produce a variance table covering segment usage, qualifier values, code lists, and date handling.

**Deliverable:** a variance table of at least eight rows and a one-page note on which differences break a single map.
**Acceptance criteria:** each difference cites a guide and version; one is a mandatory-versus-optional reversal; syntax and semantics are distinguished.

### Exercise 7 — Sell-through evidence audit (critique, 45 minutes)

You are given weekly 852 data and a daily 846 feed for a VMI programme. State what you can and
cannot conclude about demand, and design three checks that would catch a population or period
change.

**Deliverable:** a two-column list of valid and invalid inferences, plus three checks with expected results.
**Acceptance criteria:** one invalid inference concerns store population, one concerns returns; each check has a pass and a fail condition.

### Exercise 8 — Warehouse reconciliation (design, 40 minutes)

Design the control that reconciles 940 orders to 945 advices for a third-party warehouse, and
state what the control cannot prove.

**Deliverable:** a reconciliation specification with keys, tolerated differences, and escalation.
**Acceptance criteria:** the key includes warehouse order number and line; a 945 is stated not to be a delivery; one tolerance is justified.

### Exercise 9 — Receiving policy memo (design, 50 minutes)

Write a one-page policy for how your receiving operation should use an 856, covering trust,
counting, replacements, and discrepancy escalation.

**Deliverable:** a policy with a decision rule for each of the four topics and a named owner.
**Acceptance criteria:** the ASN is an expectation and the scan the record; replacement handling is specified; the escalation threshold is numeric.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
syntax acceptance rate   = (1,180 - 14) / 1,200 = 1,166 / 1,200 = 97.17%
acknowledgement coverage = 1,180 / 1,200 = 98.33%
business acknowledgement = 900 / 1,100 = 81.82%
orders without an 855    = 1,100 - 900 = 200, or 18.18%
```

A typical integration dashboard displays the syntax acceptance rate, because it is the number
the translator emits. It hides two things. The first is the 20 documents with no 997 at all:
they are neither accepted nor rejected, and they disappear from the denominator of a rate
computed only over acknowledgements received. The second, and larger, is the 200 orders with no
business response. The integration is 97% healthy by its own measure and 82% healthy by the
measure that matters to the planner.

**Exercise 4 arithmetic.**

```text
shortfall          = 720 - 684 = 36 cases
shortfall rate     = 36 / 720 = 5.0%
value at cost      = 36 x 41.50 = EUR 1,494.00
value at retail    = 36 x 6 x 9.95 = EUR 2,149.20
days of cover lost = 36 / 12 = 3.0 days

invoice as issued  = 720 x 41.50 = EUR 29,880.00
correct invoice    = 684 x 41.50 = EUR 28,386.00
difference                        = EUR 1,494.00
```

The EUR 1,494 is a correction, not a loss. The cost is the handling, the investigation, and any
expedite. Separately, 24 SSCCs against 18 pallets means 1.33 SSCCs per pallet. Either the load
contains mixed pallets or some cartons carry their own labels. A receiver that equates SSCC with
pallet will expect 24 handling units and will report six phantom pallets, a 33.3% overstatement
of handling effort and a false discrepancy before a single case is counted.

**Exercise 5 arithmetic.**

```text
possible interchanges = 4,199 - 4,100 + 1 = 100
distinct received     = 97
missing               = 100 - 97 = 3
gap rate              = 3 / 100 = 3.0%
duplicate rate        = 2 / 97 = 2.06%

documents unaccounted = 3 x 14 = 42
order exposure        = 42 x 0.30 x 18,000 = EUR 226,800
```

The exposure is not a loss. It is the value of orders that might be sitting in a mailbox, an
unmonitored queue, or a translator error log. A 3% gap rate against a 0.5% threshold fails the
control by a wide margin, and the first action is not to chase the three gaps but to find out
why the sender's sequence is not being reconciled daily. A duplicate rate above zero also
matters: it means the receiver must be idempotent, and the duplicate may carry different content
if it is a resend after a correction.

**Exercise 3 answer.**

1. `HL*3**P` omits HL02, the parent ID. The pack loop is orphaned. A lenient hierarchy builder
   either attaches it to the shipment root or drops it. If it attaches to the root, the 40 cases
   are counted at shipment level and the pack has no order reference; if it drops the loop, the
   cases vanish from the receipt. Either way the order-to-pack link is broken, and the receipt
   cannot be matched to PO-4471 at line level.
2. `LIN*UP*012345678905` shifts the qualifier. LIN01 is the assigned identification, LIN02 is the
   product ID qualifier, and LIN03 is the ID. Here UP occupies LIN01 and the GTIN occupies LIN02,
   leaving LIN03 empty. A parser that reads LIN03 gets a null item and sends the line to
   suspense. A parser that reads LIN02 as the qualifier tries to look up "012345678905" as a code
   list value and fails. In both cases the item does not resolve, and the quantity of 40 attaches
   to nothing.
3. `SE*14*0001` overstates the segment count. The message contains 13 segments including ST and
   SE. A strict parser rejects the transaction set, which is at least visible. A lenient parser
   accepts it and never learns that a segment is missing, so a dropped date, reference, or
   quantity qualifier is invisible.

A fourth, softer inference is worth stating: `TD1*PLT*12` declares twelve pallets while the
message contains one pack loop. A receiver that builds expected receipt lines from the hierarchy
rather than the TD1 total will expect one pallet, and a receiver that builds them from TD1 will
expect twelve packs that the message never describes. The declared total and the evidenced
structure disagree, and the guide, not the parser, decides which one governs.

**Rubrics for the open exercises.** A strong answer in every case names a mechanism rather than an
outcome, and states the condition under which its recommendation would reverse.

| Exercise | Strong answer | Weak answer |
|---|---|---|
| 1, document map | Separates decisions that fail loudly from those that default silently; a missing 855 defaults to "assumed accepted" | Lists document purposes from a catalogue without naming a role or an action |
| 6, guide diff | Includes a mandatory-versus-optional reversal and a qualifier-value difference, and separates syntax from semantics | Lists segment names without saying what would break |
| 7, sell-through audit | Valid: units sold in the reported population and period. Invalid: total market demand, unreported stores, stale weeks | Treats the 852 total as demand and ignores the store population |
| 8, warehouse control | Keys on order and line, tolerates pack rounding, escalates any 940 with no 945, and says a 945 is not a delivery | Reconciles totals only, hiding line-level substitutions |
| 9, receiving policy | ASN sets expectation, scan sets the record, replacement supersedes by timestamp and purpose code, numeric escalation threshold | Says "verify against the ASN" without saying which quantity is posted |

## 14. Interview questions

**1. "What is the difference between a 997 and an 855?"**

A strong answer says the 997 is a translator-level acknowledgement of syntax and the 855 is a
commercial response to an order, and notes that the first arrives in seconds while the second
may never arrive. A weak answer calls both "acknowledgements" and treats them as interchangeable.

**2. "Walk me through an X12 envelope."**

A strong answer names ISA/IEA, GS/GE, and ST/SE, gives the three control-number pairs, and says
what a mismatch proves. A weak answer names the segments without saying what they control.

**3. "Why can the same 850 look different from two retailers?"**

A strong answer distinguishes the standard from the implementation guide, gives a concrete
example such as price in PO104 versus CTP03, and concludes that the map is the unit of
integration. A weak answer says "partners customise things" without naming a mechanism.

**4. "What does an 856 guarantee?"**

A strong answer says it guarantees a declaration by the sender about an intended shipment, and
nothing about physical arrival, count, or condition. A weak answer calls it proof of shipment or
proof of delivery.

**5. "An ASN says 480 and the dock counts 456. What happens in your pipeline?"**

A strong answer posts the counted quantity, raises a discrepancy against the ASN, holds the
invoice difference, and checks whether the 856 was a replacement. A weak answer adjusts the
inventory and stops, without touching the invoice or the replacement logic.

**6. "Where do you get sell-through, and how is it different from shipments?"**

A strong answer names the 852 or SLSRPT, states the population and period caveats, and notes that
shipments to the retailer are not sales to the customer. A weak answer treats the order or
shipment quantity as demand.

**7. "What is a control number for?"**

A strong answer says it binds a document to its trailer and its acknowledgement, and that the
sequence enables gap detection. A weak answer calls it a unique ID and stops.

**8. "How would you detect a missing EDI document?"**

A strong answer reconciles control-number sequences per partner and direction, and also reconciles
document counts against the expected set by type. A weak answer says "check the logs," which
detects errors but not absence.

**9. "What is the difference between an 846 and an 852?"**

A strong answer says 846 reports stock position from the holder and 852 reports sales and
activity from the party nearer the consumer, and that neither is a continuous balance. A weak
answer says both are inventory feeds.

**10. "What is a qualifier, and why does it matter?"**

A strong answer explains that the qualifier supplies the meaning of the adjacent value, and that
a dropped qualifier leaves a valid-looking value with the wrong meaning. A weak answer describes
it as a code field.

## 15. Key terms

- **EDI** — structured business document exchange between organisations in a standard format.
- **X12** — the North American EDI standard maintained by ASC X12.
- **EDIFACT** — the UN/CEFACT international EDI standard.
- **EANCOM** — the GS1 subset of EDIFACT for retail and consumer goods.
- **Transaction set** — one business document inside an X12 functional group.
- **Message** — the EDIFACT equivalent of a transaction set.
- **Segment** — one line of an EDI document, identified by a tag.
- **Element** — one value inside a segment, identified by position.
- **Qualifier** — a code that supplies the meaning of an adjacent value.
- **Loop** — a repeating group of segments describing a related entity.
- **Envelope** — the header and trailer segments that identify and bound a document or group.
- **Interchange** — the outermost envelope, one transmission between two parties.
- **Functional group** — a set of same-type transaction sets inside one interchange.
- **Control number** — a sender-assigned identifier repeated in the matching trailer.
- **Implementation guide** — a partner-specific specification of accepted segments, codes, and lengths.
- **TR3** — the X12 implementation-guide format used for 005010 guides.
- **850** — purchase order; **855** — purchase order acknowledgment.
- **856** — ship notice or ASN; **DESADV** — the EDIFACT despatch advice.
- **810** — invoice; **INVOIC** — the EDIFACT invoice message.
- **846** — inventory inquiry or advice; **INVRPT** — the EDIFACT inventory report.
- **852** — product activity data; **SLSRPT** — the EDIFACT sales data report.
- **940 / 945** — warehouse shipping order and warehouse shipping advice.
- **997** — functional acknowledgment of syntax at the group level.
- **999** — implementation acknowledgment against a guide.
- **TA1** — interchange-level acknowledgment.
- **CONTRL** — the EDIFACT acknowledgment covering technical and functional levels.
- **AS2** — signed, encrypted HTTP transport for EDI, defined in RFC 4130.
- **VAN** — a third-party EDI mailbox and routing service.
- **SSCC / GTIN / GLN** — GS1 identifiers for logistics units, products, and parties or places.
- **Chargeback** — a buyer's financial penalty for supplier non-compliance.

## 16. Sources

**Standards bodies and official specifications (Grade B)**

- ASC X12, "X12 Transaction Sets," https://x12.org/products/transaction-sets. Definitions of
  850, 855, 856, 810, 846, 852, 940, and 945.
- UNECE, UN/EDIFACT D.21A Message Type Directory,
  https://service.unece.org/trade/untdid/d21a/trmd/trmdi1.htm. Message codes and revisions for
  ORDERS, ORDRSP, DESADV, INVOIC, INVRPT, and SLSRPT.
- UNECE, UN/EDIFACT D.23A Message Type Directory,
  https://service.unece.org/trade/untdid/d23a/timd/timdi1.htm. Current directory, used to check
  that WHSORD and WHSRSP are not UN-sanctioned message codes.
- GS1, "GS1 EANCOM," https://www.gs1.org/standards/eancom. The EANCOM subset and its relationship
  to UN/EDIFACT.
- GS1, "GS1 EANCOM 2022 standards edition 2016 S3," https://www.gs1.org/eancom/syntax-3. Message
  directory for the syntax 3 edition.
- GS1, "EANCOM Part I: Specific Rules,"
  https://www.gs1.org/sites/default/files/docs/eancom/ean02s4/part1/part1_04.htm. GTIN, GLN, SSCC,
  date, and qualifier conventions across messages.
- Microsoft Learn, "EDI Headers and Trailers,"
  https://learn.microsoft.com/en-us/biztalk/core/edi-headers-and-trailers. X12 and EDIFACT
  envelope structure and control-number matching.
- Microsoft Learn, "EDIFACT CONTRL acknowledgments and error codes,"
  https://learn.microsoft.com/en-us/azure/logic-apps/logic-apps-enterprise-integration-edifact-contrl-acknowledgment.
  CONTRL structure, action codes, and error codes.
- Aronoff, R. and Hsing, K., *An Analysis of ANSI ASC X12 and UN/EDIFACT Electronic Data
  Interchange (EDI) Standards*, NISTIR 5631, National Institute of Standards and Technology,
  https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir5631.pdf. Comparison of the two families,
  including the functional-group concept and acknowledgement scope.
- X12, *Implementation Acknowledgment for Health Care Insurance (999)*, 005010X231. Public copy:
  https://www.magellanoflouisiana.com/documents/2021/04/999-guide.pdf. Origin and structure of the
  999.

**Partner implementation guides and programme requirements (Grade B for the document, C for interpretation)**

- Tractor Supply Company, *4010 850 EDI Guidelines*, http://www.iconnect-corp.com/specs/vendors/tractorsupply/4010_850_EDI_GUIDELINES.pdf
- Exostar, *X12 850 Implementation Guide*, https://www.myexostar.com/wp-content/uploads/2023/10/Exostar-X12-850-Implementation-Guide.pdf
- Amazon, *North America EDI 856 Business Requirements*, https://cdn2.hubspot.net/hubfs/435785/rg/Amazon_ANA_EDI_ASN_Business_Requirements.pdf

**Industry and practitioner evidence (Grade C)**

- Produce Traceability Initiative, *Why and How to Use EDI 856 Advance Ship Notice/Manifest*, 2024, https://producetraceability.org/wp-content/uploads/2024/05/PTI-Why-and-How-to-Use-ASNs-2024.pdf
- RVCF, "Reliable and Accurate Automated Receiving Using Logistics Labels," 2022, https://www.rvcf.com/blogpost/1149227/463031/Reliable-and-Accurate-Automated-Receiving-Using-Logistics-Labels (survey of 79 suppliers and 20 retailers)
- GS1 Italy, *Monitoraggio dell'uso dell'EDI nel largo consumo in Italia*, 2025, https://gs1it.org/content/public/5a/61/5a61a72b-2b2b-427e-be68-e0a8c276147a/gs1-italy-monitoraggio-edi-2025.pdf
- Infor, "Planning your EDI implementation," https://docs.infor.com/ifsm/2026.10/en-us/ifsmolh/ifsmediug/hcq1466569722884.html
- TIBCO, "Acknowledgements," BusinessWorks Plug-in for EDI documentation, https://docs.tibco.com/pub/bwpluginedi_standard/6.3.0/doc/html/GUID-71562D07-DDB1-4C01-A8C4-CEABD70D46C0.html

**Vendor and analyst material (Grade D)**

- 1 EDI Source, "Maximizing Efficiency with API and EDI Integration," https://www.1edisource.com/blog/edi-and-api-why-your-businesses-probably-needs-both/ (origin of the 78%, 20 billion, and USD 6.4 trillion figures)
- OpenText, "AI Supply Chain Playbook," https://blogs.opentext.com/ai-optimized-supply-chain-community-playbook/ (the 31 billion transaction claim)
- Better EDI, "EDI 846" and "EDI 852," https://betteredi.com/resources/edi-documents/846.html and https://betteredi.com/resources/edi-documents/852.html
- Ordersync, "EDI 852 Product Activity Data," https://ordersync.io/guides/edi/852-product-activity-data
- Interlink Commerce, "EDI Transactions Directory," https://interlinkcommerce.com/edi-transactions/
- ihateedi, "EDI Mapping Guidelines," https://ihateedi.com/edi-mapping-guidelines/

**Note on citations.** Standards and implementation guides were consulted directly. Volume
figures are reported as their sources state them and are not independently verified; the 78%
share, 20 billion transactions, and USD 6.4 trillion value all trace to a single vendor blog and
are graded D accordingly. Secondary sources disagree on the segment structure of the 852: some
describe BCT and PA segments, others BIA and ZA. The implementation guide governs, and the
disagreement is itself an illustration of section 5.1.

## 17. Links

- Module index and topic map: [`_index.md`](_index.md)
- Prerequisite, data movement: [`../02_supply_chain_data_landscape/06_data-movement.md`](../02_supply_chain_data_landscape/06_data-movement.md)
- Prerequisite, ingestion patterns: [`03_ingestion-patterns.md`](03_ingestion-patterns.md)
- Next in sequence, GS1 identifiers and EPCIS: [`09_gs1-epcis.md`](09_gs1-epcis.md)
- Related, APIs, contracts, and master data: `10_apis-contracts-mdm.md` (planned)
- Related, layered architecture and provenance: [`01_layered-architecture.md`](01_layered-architecture.md)
- Related, SQL patterns for snapshots and as-of joins: [`04_sql-supply-chain-patterns.md`](04_sql-supply-chain-patterns.md)
- Related, orchestration and the planning cutoff: [`07_orchestration.md`](07_orchestration.md)
- Physical context, transport and Incoterms: [`../01_supply_chain_foundations/08_transport-and-incoterms.md`](../01_supply_chain_foundations/08_transport-and-incoterms.md)
- Survey file for this module: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Module map at the part level: [`../_index.md`](../_index.md)
