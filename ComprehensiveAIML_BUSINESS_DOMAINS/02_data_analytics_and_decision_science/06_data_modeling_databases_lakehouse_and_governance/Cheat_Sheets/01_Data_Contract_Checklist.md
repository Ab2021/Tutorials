---
collection: "data-modeling-databases-lakehouse-and-governance"
file_id: "01"
topic: "Data contracts: grain, quality, delivery, and change control"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-08"
reviewed: null
review_trigger: "A producer changes schema, semantics, delivery, ownership, or quality objective; or a downstream decision that consumes the dataset changes its tolerance."
notation_style: "plain-text"
prerequisites: ["Module 02 SQL and basic relational keys", "UTC timestamp handling and GROUP BY aggregation"]
pattern_tags: ["data-product", "governance", "reconciliation", "change-control"]
source_status: "OPEN"
owner_role: "Data product owner"
source_index: "INDEX.md"
---

# Data Contracts: Grain, Quality, Delivery, and Change Control

> **Purpose:** Turn an informal producer–consumer understanding into a checklist that can be executed, scored, and enforced, so that a named decision either receives a certified dataset or receives a documented refusal.
> **Prerequisites:** Relational keys and functional dependency intuition; `GROUP BY` and join semantics; the difference between an event timestamp and a load timestamp; ratios and percentage arithmetic.
> **Key outcome:** Given a producer, a consumer, a decision, and a candidate dataset, you can write the contract, compute its quality and freshness signals with correct denominators, classify a proposed change as additive, semantic, or breaking, and decide whether to release, hold, quarantine, or fall back.
> **This file does NOT cover:** Legal or regulatory interpretation of data-sharing agreements; vendor-specific catalogue or table-format APIs; proof that an upstream business process recorded reality correctly.
> **Level:** 1 — Foundations, with production controls attached so the same checklist survives contact with an on-call rotation.
> **Version / reviewed:** 2.0 / not independently reviewed. Arithmetic in Sections 4, 5, and 9 was recomputed during authoring; no production dataset was measured.
> **Evidence:** `DERIVED` invariants from definitions of ratio and set; `VERIFIED` illustrative arithmetic (recomputed, shown in-line); `SOURCED` pointers to public specifications listed in Section 10; everything numeric about a dataset is `ILLUSTRATIVE`. No company, pipeline, or incident is claimed.
> **Retrieval cue:** If a dashboard moved, a backfill doubled a total, or a "compatible" release broke finance, start at Section 4 for the denominator check and Section 8 for the discriminating test.

## 1. File Header

### 1.1 The decision this sheet supports

A data contract is a versioned, explicit agreement between one producer and a named set of consumers about what a dataset means, what it contains, when it is available, how good it is, and how it may change. The checklist in this file exists to answer one question at a release boundary: **may this consumer use this version of this dataset, for this decision, at this moment?** That question has four independent parts — meaning, population, timing, and control — and a "yes" requires all four. A dataset can be perfectly parseable and still be unusable, because a `decimal` column can remain a `decimal` while switching from net to gross of tax, and a table can arrive on schedule while silently omitting one sales channel.

Most organisations already have three weaker artefacts and mistake one of them for a contract. A schema definition says what types are permitted. A service-level agreement says how often a pipeline is expected to succeed. A quality dashboard says what fraction of rows passed some tests. None of the three states the business grain, the population boundary, the decision consequence of failure, or who has authority to stop the release. The contract is the artefact that binds those four facts to a name and a version, which is why it can be tested in a pull request and cited in an incident review.

> **Framing note.** The unit of a contract is not a table; it is a *dataset version consumed by a decision*. Two consumers reading the same physical table often need two different contracts, because one needs latest-version snapshots for an intraday operational decision and the other needs immutable daily partitions for a filed financial statement. Writing one contract for both produces a document that satisfies neither.

### 1.2 What the reader must be able to do after this file

| Capability | Concrete evidence of having it | Section that teaches it |
|---|---|---|
| State a grain unambiguously | Write a sentence of the form "one row per (keys) per (mutation rule), valid as of (time)" and give a duplicate test that would fail if it were wrong | 3, 5 |
| Compute coverage honestly | Produce `R/E`, `V/R`, and `V/E` from a delivery report and explain which one the release rule uses | 4.1 |
| Separate lateness from incompleteness | Distinguish "newest row is 4 minutes old" from "the population is complete through 10:20" | 4.2 |
| Classify a change | Label a proposed diff as additive, semantic, or breaking, and name the consumers who must sign | 7 |
| Diagnose a metric jump | Order the hypotheses (grain, fan-out, population, unit, definition) before attributing the change to the business | 8 |
| Price the control | Estimate the state, scan, and latency cost of each check and decide which checks must be exact | 9 |
| Defend the design | Answer a challenge about denominators, idempotency, and override authority without hedging | 11 |

### 1.3 The checklist posture: gates, not aspirations

A contract that only describes intent degrades into documentation that nobody reads after the second sprint. The checklist posture converts each clause into a **gate** with three properties: a machine-evaluable predicate, a named authority, and a consequence. If any of the three is missing, the clause is a wish. This is the same discipline reliability engineering applies to service levels — a target without an error budget and an owner produces alerts, not decisions.

Three gate strengths are used throughout, and the reader must not blur them:

- **Blocking gate.** The dataset is not published to the consumer at all; the consumer's job fails with a contract violation rather than writing a number that looks plausible.
- **Quarantine gate.** Rows that fail are diverted to a side table with a reason code; the passing rows continue, and the consumer's metric carries an explicit, machine-readable exclusion count.
- **Warning gate.** The signal is recorded and routed to an owner with a response target, but release proceeds. Warn-only is legitimate for exploratory or low-consequence uses; it is indefensible for a filed number.

The choice of gate strength is a business decision informed by the cost asymmetry in Section 9. A gate that is stronger than necessary burns engineering time and creates pressure to bypass it; a gate that is weaker than necessary is the mechanism by which a wrong number reaches an executive dashboard with no audit trail.

### 1.4 Scope boundary and handoffs

This sheet teaches the *content and enforcement* of the agreement. It deliberately stops short of adjacent subjects that each have their own file in this module. Model notation, cardinality, and join safety are the subject of the notation and join-safety sheets; storage format and snapshot mechanics belong to the lakehouse file; lineage and catalogue operations belong to the metadata file; delivery semantics under exactly-once claims belong to the streaming module.

| Adjacent topic | Where it belongs | What this sheet still owes it |
|---|---|---|
| Entity/relationship notation | Model notation sheet | The contract must name keys in the same vocabulary the model uses |
| Join fan-out and loss | Join-safety sheet | The contract must declare the multiplicity a consumer may assume |
| Snapshot and table-format mechanics | Lakehouse sheet | The contract must state which version a consumer reads |
| Lineage and ownership graph | Metadata sheet | The contract must name an owner role and an escalation path |
| Exactly-once delivery claims | Streaming module | The contract must state its duplicate policy rather than assume transport guarantees |

### 1.5 How to use the sheet operationally

Read Sections 2 through 4 once for the model. Then use Sections 5 and 8 as the working pair: Section 5 is the arithmetic you will actually perform at a close, and Section 8 is the hypothesis list you will actually run at 07:00 when the number is wrong. Section 7 is the table to bring to a design review where a producer proposes a change. Section 11 is the rehearsal for the review itself.

A practical sequence for a first contract on an existing dataset, which is the common case, is: inventory the consumers by decision (not by team); write down the grain as it is actually produced, then compare it with what each consumer assumes; instrument the expected-count ledger before tightening any threshold; classify every field as identity, measure, dimension, or control; and only then negotiate thresholds. Doing thresholds first is the classic failure — the producer proposes 99% because it sounds serious, nobody can compute the denominator, and the number becomes decorative within a month.

### 1.6 A clause-by-clause authoring walkthrough

The fastest way to learn the posture is to watch one clause improve. Take the recurring case of an amount field whose currency encoding is ambiguous. Four drafts, each better than the last, illustrate what a clause actually has to carry.

*Draft 1 — the topic.* "The dataset contains order amounts." This is a description of a column, not a clause. It cannot fail, so it cannot protect anyone.

*Draft 2 — the type.* "`extended_amount` is `decimal(18,2)` and non-null for statuses placed, shipped, delivered." Better: it states a type and a nullability rule, so a schema check can enforce it. It still says nothing about currency, so a switch from minor to major units passes every check and changes every total by 100x.

*Draft 3 — the semantics.* "`extended_amount` is the line total in the currency named by `currency_code`, expressed in major units, excluding tax, and equal to `quantity * unit_price` before line-level discount." Now the arithmetic relations are stated and testable: an assertion can verify `abs(extended_amount - quantity*unit_price) <= 0.01` for non-discounted lines, and a second assertion can verify that `currency_code` is in the allow-list. A unit change now fails the relation if the price is in the other unit, and fails the currency-mapping audit if both were changed consistently.

*Draft 4 — the release clause.* The draft 3 text plus: an owner, an evaluation window, a tolerance for the relation assertion, a gate strength (blocking, because this feeds a filed margin number), a fallback (publish prior certified value labelled as-of), and a change class (semantic, dual-publish required, finance sign-off). This is the finished article.

The progression is the whole lesson: a contract clause moves from topic to type to semantics to *gate*, and each move adds a way for the clause to be violated, which is what makes it a control rather than a comment. Most real contracts stall at draft 2, which is why they are discovered to be inadequate during a unit-change incident rather than before one.

### 1.7 Reading order for a reviewer

A reviewer of a proposed contract should read in the opposite order to the author: start at Section 12's card and ask whether every clause is represented; then read the quality objective and check that each threshold has a denominator and a consequence; then read the grain and try to break it with a duplicate and an aggregation test; then read the change classes and check that at least one consumer is named for each class; and only then read the narrative, looking for claims that the checks cannot enforce.

This order is efficient because it fails fast. A contract that cannot pass the card check — a missing grain, a threshold without a denominator, a change class with no approver — will not be rescued by good prose, and the reviewer has spent five minutes rather than an hour to discover that.

**Takeaway:** A contract is a release gate with a decision attached, and its four parts — meaning, population, timing, control — must all pass independently before a consumer may use the version.

## 2. Concept Map / Mental Model

### 2.1 The contract as a boundary object between two owners

The mental model that survives production is a boundary crossing. On one side a producer owns a business process and its recording. On the other side a consumer owns a decision and its consequence. The contract is the only place where both parties' obligations are written in the same units. Everything the producer believes about its own data quality is irrelevant to the consumer unless it is expressed as a property of the delivered population; everything the consumer assumes about meaning is an unverified guess until the producer confirms it in the contract.

```text
   PRODUCER SIDE                         CONTRACT                        CONSUMER SIDE
   ------------------                    --------                        -------------
   business process                      grain statement                 decision to be made
   source system write                   identity + version rule         population implied by metric
   event timestamp                       time semantics (event/ingest)   as-of cutoff used
   retry / backfill behaviour            duplicate + replay policy       aggregation level
   known defects                         quality objective + gate        tolerance for error
   release cadence                       availability clock              deadline for the number
   engineering owner                     change classes + approvers      decision owner + override right
```

Reading the diagram left to right is the authoring direction: the producer states facts about its process, the contract translates them into testable properties, and the consumer states which of those properties its decision depends on. Reading right to left is the review direction: start from the decision, ask which property would change the answer, and check whether the contract states it.

### 2.2 Lifecycle: the five states a contract version passes through

```text
 [proposed] --author + consumer review--> [agreed] --instrumented--> [enforced]
      |                                      |                          |
      |                                     reject                      |
      v                                      v                          v
  [withdrawn]                          [renegotiated]            [deprecated]
                                                                        |
                                                              migration window
                                                                        v
                                                                  [retired]
```

- **Proposed.** A draft with grain, keys, and field classes filled in, thresholds provisional and marked as such. Nothing enforces it yet.
- **Agreed.** Both the producing engineering owner and each consuming decision owner have signed the quality objective and the change classes. The signature is meaningful only if the consumer's fallback behaviour is also written down.
- **Enforced.** Checks run in the delivery path, failures route to a responder, and the release decision is automated or explicitly human with a recorded name.
- **Deprecated.** A successor version exists, existing consumers still read the old one, and a dated migration window with a named exit condition is published.
- **Retired.** No consumer reads the version; the physical artefact may persist for audit but is no longer a live contract.

A contract that jumps from proposed straight to "documented in the wiki" has no enforced state, which is why so many contracts are discovered to be fiction during an incident. The transition that matters most is agreed → enforced, because it is the one that requires someone to write a check, a route, and a consequence.

### 2.3 The four validation planes and what a pass in one plane proves

```text
                 +--------------------------------------------------+
   plane 1       | SCHEMA      types, nullability, enums, widths    |  proves: parseable
                 +--------------------------------------------------+
                 +--------------------------------------------------+
   plane 2       | IDENTITY    key uniqueness, referential closure  |  proves: countable
                 +--------------------------------------------------+
                 +--------------------------------------------------+
   plane 3       | SEMANTICS   units, denominators, grain, as-of    |  proves: interpretable
                 +--------------------------------------------------+
                 +--------------------------------------------------+
   plane 4       | DELIVERY    completeness, freshness, ordering    |  proves: timely
                 +--------------------------------------------------+
                          pass in all four -> release eligible
```

Each plane answers a different question and each can pass while another fails. Schema passing proves only that a reader can deserialise the bytes. Identity passing proves that a row count means something. Semantics passing proves that the arithmetic the consumer performs matches the business concept. Delivery passing proves that the population is present and current enough for the decision deadline. The commonest expensive mistake in practice is treating plane 1 as a proxy for planes 2–4, which is exactly the mistake a schema registry cannot prevent.

### 2.4 The quality funnel: where records are lost

The second mental model is a funnel with labelled leaks. Every leak has a different owner, so collapsing them into one "quality score" destroys the ability to route the repair.

```text
expected population          E   ── producer process + expectation ledger
   │  lost in production / never produced            owner: business process
   │  lost in transport / filtering                   owner: pipeline operator
   ▼
delivered records            A   ── including retries and duplicates
   │  duplicate suppression                          owner: pipeline operator
   ▼
unique received              R   ── the countable population
   │  rule violations (null, range, currency, status) owner: producer data quality
   ▼
valid records                V   ── arithmetic-safe
   │  authorization / purpose / row-level policy     owner: access governance
   ▼
decision-eligible            Q   ── what the metric may use
```

The funnel matters because the ratio a team quotes is usually ambiguous about its base. "98% quality" could mean `V/R`, `V/E`, or a test-pass rate per assertion, and those three numbers can differ by several points on the same data. A contract must name which ratio the gate uses and must publish the raw counts alongside it, because a ratio without its numerator and denominator cannot be audited.

### 2.5 Change as a lattice, not a binary

```text
                       consumers unaffected ── additive ── new optional field, new partition
                              │
                              ▼
                    parse-compatible ── semantic ── unit, denominator, encoding of meaning
                              │
                              ▼
                    parse-breaking ── structural ── rename, drop, type narrowing, key change
```

The vertical axis is "can the consumer's code still run", the horizontal axis is "does the consumer's number still mean the same thing". The dangerous quadrant is parse-compatible but semantically changed, because every automated check except a semantic one will pass. Section 7 turns this lattice into a table with required approvals, and Section 8 gives the discriminating test for each cell.

### 2.6 Feedback: how the contract learns

```text
  [incident or reconciliation delta]
              |
              v
  [root cause: which plane failed?]
              |
     +--------+---------+------------------+
     v                  v                  v
  gap in clause   gap in check      gap in consequence
  (contract wrong) (test missing)   (gate too weak)
     |                  |                  v
     v                  v            raise gate strength
  amend version    add assertion      + rehearse escalation
     |                  |                  |
     +------------------+------------------+
                        v
              [contract version +1, review date set]
```

A contract that never changes is either trivial or abandoned. The useful discipline is that every material incident produces exactly one of three edits — a clause, a check, or a consequence — and the edit is versioned so a later reader can see why the clause exists.

### 2.7 The nine clause families, and what each one breaks if omitted

Every contract, regardless of domain, contains the same nine families of clauses. Naming them makes completeness reviewable: a reviewer can count the families present and ask, for each absent one, whether its absence is deliberate.

| # | Clause family | What it states | If omitted, the typical failure |
|---|---|---|---|
| 1 | Scope and population | which entities, channels, statuses, and time ranges are in and out | `E` and `R` computed on different bases; coverage near 1 with a wrong total |
| 2 | Grain and identity | what one record is, and the key tuple | double counting on revisions or fan-out on joins |
| 3 | Time semantics | event, ingestion, processing, timezone, partition boundary | "today" means three different things to three consumers |
| 4 | Field semantics | units, currency, encoding, null meaning, derivation relations | silent drift across a parse-compatible release |
| 5 | Availability | close time, deadline, watermark, restatement window | freshness assessed from the wrong timestamp |
| 6 | Quality objective | ratio, threshold, window, gate strength, consequence | dashboards instead of decisions; no release authority |
| 7 | Access and purpose | who may read which rows and columns, and for what | totals that differ by principal with no documented cause |
| 8 | Change control | classes, approvers, dual-publish, migration window | breaking changes shipped as "compatible" |
| 9 | Ownership and escalation | producer owner, responder, decision owner, override right | violations detected and never acted upon |

The families are not equally urgent at adoption. Scope, grain, and time semantics must exist before any number is published, because without them the number is undefined. Quality, access, and change control can follow, because their absence produces wrong or unsafe numbers rather than undefined ones. Ownership cannot be deferred at all, because a contract with no named responder is a document.

### 2.8 Three consumer archetypes and the clauses they load

Different consumers stress different clause families, which is why a single contract often needs per-consumer appendices rather than one global table of thresholds.

*The batch reporter* reads a closed partition once, compares against a prior period, and files or presents a number. It cares about scope, grain, field semantics, and a restatement policy. Freshness matters only up to the close deadline; intraday signal is noise for it. Its worst failure is a silent population change, because nobody compares row counts after the number is filed.

*The intraday operator* reads the latest state continuously and acts within minutes. It cares about availability and freshness above all, tolerates a degree of incompleteness if it is visible, and is harmed more by a late number than by a slightly short one. Its worst failure is a green freshness tile built on the newest-record timestamp, because it produces confident action on a stalled population.

*The model pipeline* reads features and labels, and cares about identity over time, reproducibility, and leakage boundaries. Its worst failure is not a wrong number but a wrong *availability* — a feature computed with information that would not have existed at prediction time. That is a semantic clause of a specific kind (an as-of constraint on every feature's inputs), and it belongs in the contract because no schema check can see it.

| Clause family | Batch reporter | Intraday operator | Model pipeline |
|---|---|---|---|
| Scope and population | critical | important | critical |
| Grain and identity | critical | critical | critical |
| Time semantics | critical (as-of close) | critical (watermark) | critical (point-in-time) |
| Field semantics | critical | important | critical |
| Availability | important (close) | critical (deadline) | important (freshness band) |
| Quality objective | critical | important | important |
| Access and purpose | important | important | important |
| Change control | critical | important | critical |
| Ownership | critical | critical | critical |

The practical output of this table is that "critical" cells become blocking gates and "important" cells become warning gates with an owner. A contract that treats all cells as critical will have nine blocking gates and will be bypassed within a month; one that treats all as warning will have none and will be bypassed immediately.

### 2.9 The contract as a build input, not a document

The single design choice with the largest effect on whether a contract survives is whether the pipeline *reads* it. When the contract file supplies the key list, the rule thresholds, the time semantics, and the expected-count query, divergence between document and behaviour becomes a build failure. When the contract lives beside the pipeline as prose, divergence is the default and the document decays at the rate of the release cadence.

```text
   contract.yaml  ---- read by ---->  validation job (assertions, gate)
        |                                    |
        +--- read by ---->  documentation generator (human-readable page)
        |
        +--- read by ---->  consumer SDK (typed accessors, as-of defaults)
        |
        +--- recorded in --> lineage run facets (evaluated version + result)
```

The consequence is that the contract's machine-readable portion must be small and boring: keys, types, rules with thresholds, time fields, expected-count query reference, gate strengths, owners. The prose portion — the rationale, the incident history, the boundary cases — sits beside it and is not parsed. Mixing the two produces a document that is either unparseable or unreadable.

**Takeaway:** Model the contract as a boundary crossing tested in four independent planes, with every quality ratio anchored to a named funnel stage and every change placed on a compatibility lattice.

## 3. Foundational Definitions

### 3.1 Definitions table

| Term | Meaning | Units / type | Domain | Nonexample |
|---|---|---|---|---|
| Grain | The business statement one record represents, including its mutation rule | prose + key tuple | one per dataset version | "one customer" when rows are order lines |
| Business key | Minimal tuple identifying the real-world entity or event the grain names | stable attribute tuple | non-null, unique per grain after dedup | a sequence-generated load id |
| Surrogate key | System-assigned identifier with no business meaning | integer/uuid | unique by construction | used as evidence of business uniqueness |
| Version (of a record) | Declared mutation ordering for one business key | integer or timestamp plus tie rule | monotone per key | ingestion wall-clock alone |
| Event time | When the represented fact occurred in the world | UTC instant | may be late, out of order | warehouse write time |
| Ingestion time | When this platform first observed the record | UTC instant | monotone per record, not per key | business effective date |
| Processing date | Logical partition the pipeline assigned the record to | date | may be revised by backfill | same as event date in general |
| Watermark | Latest cutoff for which the producer asserts the declared population is complete | UTC instant bound | per partition, per population | newest record timestamp |
| Expected count `E` | Independently produced count of records the population should contain in a window | records/window | positive integer, same scope as `R` | a retry of `R` |
| Delivered count `A` | Records physically delivered, including duplicates and retries | records/window | `≥ R` | unique population |
| Unique received `R` | Delivered records after duplicate suppression on the declared key | records/window | `0 ≤ R`, compared to `E` | all rows ever stored |
| Valid count `V` | Unique received records satisfying every declared rule | records/window | `0 ≤ V ≤ R` | rows that merely deserialise |
| Eligible count `Q` | Valid records the consumer is permitted and able to use for its purpose | records/window | `0 ≤ Q ≤ V` | unrestricted table row count |
| Freshness age `a` | Gap between now and the newest complete watermark | time | nonnegative | age of the newest individual row |
| Availability window | Promised deadline by which a partition is complete for consumption | clock time / duration | per partition and timezone | pipeline start time |
| Quality objective | Named ratio, threshold, window, and consequence | structured clause | versioned | "high quality" |
| Contract violation | A record or partition that fails a published clause | event / partition | routable to an owner | any unexpected data point |
| Compatibility | Whether a named consumer can interpret a proposed version without changing its decision logic | per consumer, per version | boolean with evidence | "the DDL succeeded" |
| Quarantine | Isolation of failing records with a reason code, leaving the passing set usable | table + reason enum | reversible on correction | silent row drop |
| Certified value | A published number with its dataset version and as-of cutoff attached | metric record | superseded, never overwritten | a screenshot of a dashboard |

### 3.2 Grain deserves its own paragraph

Grain is the single most valuable clause in the contract and the one most often written as a noun rather than a statement. "Order lines" is a topic. "One row per `(order_id, line_id)` per `snapshot_version`, representing the state of that line as of the snapshot's cutover, with later versions superseding earlier ones for latest-state readers" is a grain. The second version tells a reader four things simultaneously: what the identity is, what the mutation semantics are, what "latest" means, and therefore which aggregations are safe.

Two tests validate a grain statement, and both should exist as assertions rather than review comments. The first is a duplicate test: group by the declared key tuple and count; the maximum count must equal one for an immutable-fact grain, or the pair count must be well defined for a versioned grain. The second is an aggregation test: sum a measure at the declared grain and compare with the same measure summed after collapsing to the parent entity by the declared rule; the two must agree, or the difference must be attributable to the mutation rule. A grain that passes the duplicate test but fails the aggregation test usually has a hidden many-to-many relationship behind the declared key.

### 3.3 Identity versus uniqueness

Uniqueness is a property of a column set in a table; identity is a claim about the world. A surrogate key is unique by construction and therefore proves nothing about whether two rows describe the same business event. Conversely, a business key can be duplicated legitimately when the dataset carries versions. The contract must state which concept each key serves, because the deduplication policy depends on it: deduplicating on a surrogate key does nothing, while deduplicating on a business key without a version rule silently discards either the correction or the original.

The practical checklist question is: *if the source system emitted this business event twice with different technical identifiers, what should the consumer see?* The answer must be one of "one row, latest according to this ordering rule", "two rows, because restatements are meaningful", or "quarantine, because the contract cannot decide". "Whatever the load order produced" is not an answer and is what most pipelines actually implement.

### 3.4 Event time, ingestion time, and the three lies about "today"

Most metric disputes that are not about grain are about time semantics. A record's event time says when the business fact happened; ingestion time says when the platform first saw it; the processing partition says which logical batch claimed it. A contract that does not name all three leaves "today's sales" ambiguous in a way that survives every schema check.

| Phrase in a request | Which time it usually means | Contract clause needed |
|---|---|---|
| "Sales for 3 March" | event date, business timezone | event-time definition + timezone + late-arrival policy |
| "What arrived today" | ingestion date | ingestion partition boundary and revision policy |
| "Yesterday's close" | processing partition, cut at a stated clock | partition close time, watermark, restatement window |
| "Latest numbers" | undefined until the reader says latest by which clock | as-of rule and supersession policy |

The late-arrival policy is the clause teams forget. A fact that arrives after the partition closed is not automatically wrong; it is a question of whether the consumer recomputes. Three policies are defensible — reject late facts, accept them into an immutable history and recompute a restated value, or accept them into history but freeze the certified value with an explicit "as-of" label. Each has a different consequence for a finance close, and the contract must say which one applies rather than letting the pipeline's behaviour decide by accident.

### 3.5 Watermark versus "last row seen"

A watermark is an assertion of *completeness up to a cutoff*, not a timestamp lifted from the newest row. The distinction is the difference between "I have seen a record stamped 10:34" and "I am willing to assert that no record stamped before 10:20 will ever arrive". Only the second supports a freshness claim, because a stream can deliver one very recent record while a whole partition from twenty minutes earlier is still in flight. Contracts that compute freshness as `now - max(event_time)` therefore report excellent freshness during exactly the incident where data is missing.

The contract clause should read like a legal assertion: *for population P, partition D, the producer asserts completeness through 10:20 UTC at 10:22 UTC*. That sentence can be tested by comparing `E` and `R` for the asserted window, and it can be contradicted, which is what makes it useful.

### 3.6 Quality objective, tolerance, and consequence

A quality objective is a four-tuple, and omitting any element makes it unenforceable: named ratio, threshold, evaluation window, consequence. "Usable coverage `V/E ≥ 0.99` per UTC day; on breach, hold the financial metric and publish the prior certified value with its as-of label" is enforceable because a script can evaluate it and an on-call engineer can act on it. "Data quality must be high" is not an objective; it is a sentiment.

Tolerance is the consumer's number, not the producer's. A marketing attribution model may tolerate 5% missing impressions because the estimate is directional; a revenue recognition process tolerates approximately none because a misstatement is a filing problem. The contract exists precisely to record that asymmetry, so the same producer can publish one dataset with two service levels and two named consumers, rather than an average service level that satisfies neither.

### 3.7 Contract versus SLA versus schema registry versus data-quality test

| Artefact | Answers | Does not answer | Failure when mistaken for a contract |
|---|---|---|---|
| Schema registry | What types may appear, and did this version break parsers? | grain, population, timeliness, meaning | passes a unit change because `decimal` stayed `decimal` |
| SLA | How often does the pipeline succeed within the target? | correctness, population, decision consequence | a "99.9% successful runs" pipeline that is 96% complete every day |
| Data-quality test suite | Did this assertion hold on this run? | which assertion the decision depends on, and what happens on failure | hundreds of green tests, no release authority |
| Catalogue entry | Who owns it and what is it called? | the binding obligations | documentation drift with no enforcement |
| Contract | All of the above, versioned, with a named consumer and consequence | whether reality was recorded correctly upstream | — |

### 3.8 "Compatible" is a relationship, not a property

Compatibility is always relative to a consumer and a use. Adding a nullable column is additive for a consumer that selects named columns and breaking for one that does `SELECT *` into a positional writer. Changing a currency field's encoding from minor units to major units is parse-compatible and semantically catastrophic. Widening an integer is safe; narrowing is not. Renaming is breaking for parsers but may be additive if the old name is dual-published for the migration window.

Because compatibility is relational, the contract must carry a consumer inventory. A release process that cannot enumerate consumers cannot classify a change, and the honest classification in that situation is "unknown — treat as breaking", which is why consumer inventories are the least glamorous and most valuable part of the whole apparatus.

### 3.9 Owner, responder, and override authority

Three roles are needed and they are frequently collapsed into one, which is how a contract becomes orphaned. The **producer engineering owner** can change the pipeline and is accountable for the delivered properties. The **responder** is on a rotation and must acknowledge a violation within a stated time; the responder need not be able to fix the cause. The **decision owner** holds the authority to release a number that violates the objective, and that authority is what makes the gate real rather than advisory. Override must be logged with a name, a reason, and an expiry, because an unlogged override is indistinguishable from a missing check.

### 3.10 Classifying every field: four roles and their obligations

A contract can be reviewed field by field if each field carries a role, because the role determines which assertion applies. Four roles cover nearly everything.

*Identity fields* — keys and version attributes — carry the uniqueness and referential obligations. They must be non-null, stable across reloads, and never repurposed. A change to an identity field is always a breaking change, because every historical join depends on it.

*Measure fields* — amounts, counts, durations — carry arithmetic obligations: domain bounds, sign conventions, aggregation validity, and unit declaration. Every measure needs a stated unit and a stated additivity rule, because a ratio is not additive across rows and a currency is not additive across currencies. The additivity rule is the clause that prevents the most common aggregation error: summing a percentage.

*Dimension fields* — categories, statuses, channels, regions — carry allow-list and hierarchy obligations. Enumerations must be versioned, unknown members must have a defined handling (error or bucket), and hierarchy membership must be as-of, not current, when the dimension is used for historical reporting.

*Control fields* — ingestion timestamp, source file identifier, batch identifier, payload hash — carry audit obligations. They are not business data and must not be exposed to consumers as if they were; but they must be retained, because they are the only way to reconstruct what happened during an incident.

| Role | Non-negotiables | Typical assertion | Consequence of a role error |
|---|---|---|---|
| Identity | non-null, stable, never reused | uniqueness on the declared tuple | joins silently mis-associate history |
| Measure | unit, sign convention, additivity | range + relation to a reference field | totals wrong by a factor or a sign |
| Dimension | versioned enumeration, as-of hierarchy | allow-list + unknown handling | silent bucketing or filter drift |
| Control | retained, not consumer-facing | presence and monotonicity | undiagnosable incidents |

An ambiguous field is a contract defect even when it is not currently causing harm. `status` is the canonical example: it may be an order lifecycle state, a payment state, or an operational flag, and the three have different enumerations and different valid transitions. The field's role is not obvious from its name, and the contract must disambiguate it before any consumer writes a predicate against it.

### 3.11 Rule design: two-sided, precedence-ordered, and attributable

A validation rule is a predicate plus three decisions, and skipping any of them makes the rule unenforceable at scale.

*Two-sided bounds.* "Amount must be positive" is wrong for returns and adjustments. The correct rule is a range with a sign convention tied to a status predicate: negative amounts are valid only on records whose status is a return or an adjustment. A one-sided rule produces either silent exclusions or a rule that is relaxed as soon as the first legitimate exception appears.

*Precedence.* A record can fail several rules, and the count of failing records must not be the sum of per-rule counts. Declaring a precedence order (by business severity, typically: identity, then currency/unit, then range, then completeness of required attributes) makes the attribution deterministic and the arithmetic reconcilable.

*Attribution.* Each failure must carry a machine-readable reason code and the contract version that produced it. Without the version, a later threshold change makes historical quarantine counts meaningless; without the reason code, the top defect cannot be found without re-scanning the data.

A fourth property is often neglected: *attributability to an owner*. A rule whose failures cannot be routed to a team is a rule that will be permanently triaged as noise. The contract should list, per rule family, the owning role that receives failures.

### 3.12 The four null kinds

Null handling is where semantic drift most often hides, and the reason is that SQL has one null while data has at least four.

1. **Unknown** — the value exists in reality but was not captured. Legitimate, and a rising rate is a source-coverage signal.
2. **Not applicable** — the field does not apply to this record (tax on an exempt item). Legitimate and stable.
3. **Withheld** — known but not permitted to be stored (personal data restricted by policy). Legitimate and correlated with access policy, so it must be declared alongside `Q`.
4. **Sentinel-in-disguise** — a default such as `0`, `''`, or `1900-01-01` used to satisfy a not-null constraint. Illegitimate for analysis, because it enters arithmetic as a real value.

A contract that says "the field is nullable" has said nothing usable. The clause must state which kinds are permitted, how each is encoded (null versus explicit marker), and the expected rate band for each. A drop in the unknown rate with a rise in the sentinel rate is a net loss of information that every null-rate dashboard reports as improvement — which is the concrete mechanism behind the counterexample in Section 4.4.

### 3.13 Version numbering that a consumer can reason about

Version identifiers are only useful if the numbering rule encodes compatibility. Two schemes are common and only one is safe.

*Monotone integer with a declared compatibility promise* — for example, major versions may break, minor versions may add — lets a consumer write a range constraint and lets a registry enforce it. The promise must be stated in the numbering rule itself, not inferred from history.

*Timestamp-derived identifiers* — for example, the release date — carry no compatibility information. A consumer cannot express "any version from this lineage after this fix" without a mapping table, and the version sort order is confused with the compatibility order whenever a hotfix is published out of sequence.

Whichever scheme is used, the contract must state (a) what constitutes a new version, (b) whether a version is immutable once published, and (c) how a superseded version is marked. Immutability is the property that makes replay possible; a "version" that can be edited in place is a label, not a version.

**Takeaway:** Define grain as a statement with a mutation rule, treat identity and uniqueness as different claims, and never let "latest", "complete", or "compatible" stand without the relation and cutoff they are relative to.

## 4. Core Formulas with Full Derivation

The formulas in this section are all ratios, counts, or differences over a declared population. That simplicity is the point: the difficulty in a contract is never the algebra, it is the population and the denominator. Each block therefore states its population condition before the arithmetic, and each closes with a boundary where the formula stops being informative.

### 4.1 Base: completeness and validity

**Validity before use.** All records must be assignable to exactly one declared window; `E` must be produced independently of `R`; the duplicate policy must be applied before `R` is counted; and the rule set that defines "valid" must be a specific, versioned list rather than a general notion of quality. If `E` is derived from `R` — for example, yesterday's row count, or the count of files the producer happened to send — the formula computes a number that is structurally near 1 and carries no information.

**Derivation.** Completeness asks what fraction of the expected population actually arrived. The expected population is a count of records the business process should have produced in the window; the arrived population is a count of distinct records in the same window. Dividing the second by the first gives a dimensionless ratio in `[0,1]` whenever delivery is at most complete; values above one indicate replay, duplication, or a mismatched window rather than super-completeness. Validity then asks, of the records that did arrive, what fraction can be used without violating a published rule — a conditional ratio whose denominator is the delivered population, not the expected one. Composing the two gives the fraction of the *expected* population that is usable, because the intermediate population cancels.

```text
FORMULA [DC-01]: c = R / E
WHERE: c  completeness ratio [dimensionless];
       R  unique received records in the declared window [records];
       E  independently expected records for the same population and window [records].
CONSTRAINT: E > 0; E and R share population definition, timezone, cutoff and dedup policy;
            E is not computed from R or from the delivered file set.
NUMERICAL EXAMPLE: E = 250,000 record-slots/day; R = 244,800 unique records/day;
                   c = 244,800 / 250,000 = 0.9792.
INTERPRETATION: 5,200 expected records are absent; the 244,800 present records are
                not evidence that the missing ones were unimportant or small.
```

```text
FORMULA [DC-02]: v = V / R
WHERE: v  validity ratio [dimensionless];
       V  unique received records satisfying every declared rule [records];
       R  unique received records [records]; R > 0.
CONSTRAINT: the rule set is enumerated and versioned with the contract;
            duplicates are already removed from both V and R;
            a record either passes every rule or fails at least one (no partial credit).
NUMERICAL EXAMPLE: R = 244,800; V = 241,152; v = 241,152 / 244,800 = 0.985098...
INTERPRETATION: 3,648 delivered records violate at least one published rule and are
                excluded from the certified arithmetic.
```

```text
FORMULA [DC-03]: u = V / E = (V/R)(R/E) = v * c
WHERE: u  usable coverage of the expected population [dimensionless];
       V, R, E as above.
CONSTRAINT: V is a subset of R which is a subset of the expected population;
            the identity fails if V and R were deduplicated on different keys,
            or if E's window differs from R's.
NUMERICAL EXAMPLE: u = 241,152 / 250,000 = 0.964608;
                   cross-check v*c = 0.985098 * 0.9792 = 0.964608.
INTERPRETATION: the release rule applies to u, because a decision that needs 99% of
                the expected population is not satisfied by 99% of what happened to arrive.
```

The cancellation in `DC-03` is the algebra that makes a single release threshold possible: a consumer cares about how much of the population it *should* have seen, and neither `c` alone nor `v` alone answers that. Reporting both components alongside `u` preserves the repair route, because a shortfall caused by missing delivery is fixed by a different team than a shortfall caused by rule violations.

*Boundary and counterexample.* If `R > E` the ratio exceeds one. That is not "more than complete"; it is a signal that the duplicate policy failed, the window mismatched, or the expectation ledger is stale — the same population counted twice. And `c = 1` with `v = 1` still does not establish semantic correctness: a pipeline that emits the right number of rows with a denominator computed over the wrong channel mix is perfectly complete, perfectly valid, and wrong. Ratios are necessary controls, never sufficient ones.

### 4.2 General: freshness against a complete watermark

**Validity before use.** Clocks are comparable (both UTC, monotonic enough for minute-scale reasoning), the watermark refers to the same population and partition the consumer reads, and the producer's assertion is a completeness claim rather than a maximum timestamp. The objective must also be attached to a decision deadline; a freshness number with no deadline cannot be breached.

**Derivation.** Freshness is an age, so it is a difference between two instants. The choice of the second instant determines the meaning. Using the newest record's timestamp measures transport latency for one lucky record; using the watermark measures how old the *guaranteed-complete* population is. Only the second is a property the consumer can act on, because a consumer needs to know how much of the population it is definitely missing. Subtracting the watermark from now gives the age of the most recent cutoff the producer will defend.

```text
FORMULA [DC-04]: a = t_now - w
WHERE: a      freshness age [minutes];
       t_now  evaluation instant [UTC];
       w      latest watermark for which the producer asserts the declared
              population is complete [UTC].
CONSTRAINT: both instants UTC; w asserted by the producer, not inferred from data;
            w is per population and per partition;
            w <= t_now (a negative value indicates a clock or assertion fault).
NUMERICAL EXAMPLE: t_now = 00:35 UTC; w = 00:00 UTC for the prior day's partition;
                   a = 35 minutes.
INTERPRETATION: with a 15-minute availability objective for this partition, the
                contract is breached even though individual rows stamped 00:31 exist.
```

```text
FORMULA [DC-05]: b = t_now - max(event_time)
WHERE: b  last-event lag [minutes]; used only as a transport-latency diagnostic.
CONSTRAINT: never used as a completeness signal; a single record can make b small
            while an entire earlier partition is still missing.
NUMERICAL EXAMPLE: t_now = 00:35; max(event_time) = 00:34; b = 1 minute.
INTERPRETATION: b and a disagree by 34 minutes here, which is precisely the
                evidence that freshness must be computed from DC-04, not DC-05.
```

The pair `DC-04`/`DC-05` is worth internalising as a diagnostic rather than two competing definitions. When the watermark age is small and the last-event lag is small, the pipeline is healthy. When the last-event lag is small but the watermark is old, a partition or a segment is stalled while a trickle continues — the classic "one file per minute keeps arriving from one region" incident. When both are large, delivery has stopped. The contract clause is written on `DC-04`; `DC-05` exists to explain why on-call dashboards built on the wrong one look green.

A second general form is the attainment ratio over a window, which converts per-run pass/fail into a service-level statement. If `n` evaluation windows occurred in a measurement period and `k` of them met the objective (both coverage and freshness), attainment is `k/n`. This is the ratio an error budget is spent from, and it needs the same discipline about `n`: a period with 1,440 daily windows is not comparable to a period with 30 monthly ones, because one breach costs 1/30 of the budget in the second case and 1/1,440 in the first.

**Takeaway:** Compute coverage as a chain of nested ratios anchored to an independently produced expectation, and compute freshness from an asserted completeness watermark rather than from the newest record.

### 4.3 Variants: uniqueness, duplicate excess, reconciliation, and change impact

#### 4.3.1 Duplicate excess on a declared key

**Validity before use.** The key tuple is the one the contract declares, the comparison happens within a single window, and "accepted" means after transport retries have been applied. If the key is chosen after seeing the data, the metric is circular.

**Derivation.** Let `A` be accepted records and `D` the distinct keys among them. Every distinct key should appear once, so the excess is `A - D`. Expressing that excess relative to `A` gives the share of the delivered stream that is repetitive, a ratio in `[0,1)` because `D ≥ 1` when `A ≥ 1`.

```text
FORMULA [DC-06]: d = (A - D) / A
WHERE: d  duplicate excess share [dimensionless];
       A  accepted records including retries [records];
       D  distinct declared-key values among them [keys]; A > 0.
CONSTRAINT: key is the contract's business key, not a surrogate;
            single window; "accepted" defined by the transport layer.
NUMERICAL EXAMPLE: A = 246,000; D = 244,800; d = (246,000 - 244,800)/246,000
                   = 1,200/246,000 = 0.004878 = 0.4878%.
INTERPRETATION: roughly five records per thousand are repeats; they are harmless only
                if the merge is idempotent on the business key.
```

The important subtlety: `d` measures *record repetition*, and a low `d` does not prove the absence of *business duplication*. If the source system created two orders for one customer action and assigned two different business keys, `d = 0` and the population is still double-counted. That failure is invisible to this formula and is caught only by a source-control total or a semantic reconciliation, which is why `DC-06` is a hygiene check rather than a correctness proof.

#### 4.3.2 Conservation / reconciliation delta

**Validity before use.** Both sides are computed on the same population, the same cutoff, and the same unit. Comparing a source total at 23:59 with a consumer total at 23:30 produces a delta that measures the clock difference, not an error.

**Derivation.** If the consumer's transformation is value-preserving, the sum over the consumer's rows equals the sum over the source rows for the same population. Defining the delta as source minus consumer gives a signed quantity whose magnitude is the unexplained value and whose sign hints at direction: positive delta means the consumer lost value, negative means it invented or double-counted value.

```text
FORMULA [DC-07]: delta = S - C
WHERE: delta  reconciliation difference [currency];
       S      independently computed source total for population P at cutoff t [currency];
       C      consumer total for the same P and t [currency].
CONSTRAINT: identical population predicate, cutoff, currency, and rounding;
            a signed result is retained (do not take absolute value before diagnosing).
NUMERICAL EXAMPLE: S = 18,432,500.00; C = 18,429,875.50; delta = 2,624.50.
INTERPRETATION: 2,624.50 of value present at source is absent in the consumer,
                which is 0.01424% of S; small relative error can still be material
                in absolute currency for a filed number.
```

`DC-07` is a *value* control and `DC-06` is a *row* control; they fail independently. A fan-out join that duplicates 40 rows and an inner join that drops 40 rows can net to zero rows and non-zero value, or to zero value and wrong rows. The pair must therefore be run together with a key-level anti-join in both directions, which is the check that actually localises the defect.

#### 4.3.3 Weighted impact of missing records

A missing-record count is not a materiality statement until it is converted into the unit the decision uses. If the missing records have an average measure value `m` and the metric's reported total is `T`, the worst-case relative impact of the gap is the product of the gap and the plausible maximum per-record value, divided by the reported total. Using the average is optimistic; using a high quantile is defensive.

```text
FORMULA [DC-08]: r_hi = (E - R) * m_hi / T
WHERE: r_hi  upper bound on relative distortion of the metric if every missing
             record carried the high-percentile value [dimensionless];
       E - R  missing record count [records];
       m_hi   high-percentile per-record measure (e.g. p99 of line value) [currency/record];
       T      reported metric total for the window [currency].
CONSTRAINT: m_hi from the same population and period; assumes missingness is not
            systematically correlated with an even larger value; bound, not estimate.
NUMERICAL EXAMPLE: (E - R) = 5,200 records; m_hi = 900.00; T = 41,600,000.00;
                   r_hi = 5,200 * 900 / 41,600,000 = 4,680,000/41,600,000 = 0.1125.
INTERPRETATION: the gap could move the total by up to about 11%, which is far above a
                typical 1% materiality threshold — so the honest action is to hold, not
                to publish a footnote.
```

This is the formula that converts a quality ratio into a decision. Two datasets can both have `u = 0.96` and require opposite actions, because one's missing records are long-tail items and the other's are the largest accounts. A contract that specifies only `u ≥ 0.99` encodes no materiality reasoning at all; a contract that pairs the threshold with `DC-08` tells the decision owner what the breach could cost.

#### 4.3.4 Change impact score for release triage

A producer proposing several changes at once needs an ordering, not just a verdict. Defining impact as a sum over consumers of a weight times a per-consumer severity keeps the classification explicit and auditable.

```text
FORMULA [DC-09]: I = SUM over consumers k of (w_k * s_k)
WHERE: I    change impact score [weighted severity points];
       w_k  decision weight of consumer k [points], assigned by the data governance
            function from consequence of error (e.g. filed=5, operational=3, exploratory=1);
       s_k  severity for consumer k [0 additive, 1 semantic, 2 structural].
CONSTRAINT: weights are assigned before the change is proposed, not after;
            severity is assessed by the consumer's owner, not the producer.
NUMERICAL EXAMPLE: three consumers: filed report w=5 s=2 -> 10; operational
                   dashboard w=3 s=1 -> 3; exploration w=1 s=0 -> 0; I = 13.
INTERPRETATION: I = 13 with a rule "I >= 8 requires dual-publish plus written sign-off"
                means the change cannot ship as a single in-place release.
```

The score is a prioritisation device, not a measurement; its value is that it forces the producer to enumerate consumers and forces each consumer to state a severity. A change with `I = 0` genuinely is safe, and that verdict is now evidence rather than a hope.

#### 4.3.5 Sample size for semantic spot checks

Semantic checks — does this amount include tax, is this timestamp local or UTC — cannot be automated cheaply, so they are usually sampled. The sample size needed to detect a deviation of proportion `p` with margin `e` at confidence `z` follows from the normal approximation to a proportion, with a finite-population correction when the population is small enough to matter.

```text
FORMULA [DC-10]: n0 = z^2 * p(1-p) / e^2 ;  n = n0 / (1 + (n0 - 1)/N)
WHERE: n0  uncorrected sample size [records];
       z   standard normal quantile for the confidence level (1.96 at 95%);
       p   anticipated pass proportion [dimensionless];
       e   half-width of the confidence interval [dimensionless];
       N   population size [records] for the finite-population correction.
CONSTRAINT: simple random sample over the population; independent records;
            normal approximation adequate when n*p and n*(1-p) both exceed ~10;
            correction applies when n0 is a non-trivial fraction of N.
NUMERICAL EXAMPLE: p = 0.98, e = 0.005, z = 1.96 ->
                   n0 = 3.8416 * 0.98 * 0.02 / 0.000025 = 3,011.8 -> 3,012.
                   With N = 244,800: n = 3,012 / (1 + 3,011/244,800) = 2,975.2 -> 2,976.
INTERPRETATION: about 2,976 audited records are needed to bound the semantic pass rate
                within +/-0.5 points; at 20 audits/hour that is roughly 149 auditor-hours,
                which is why semantic checks are sampled on a schedule, not per release.
```

The finite-population correction is the arithmetic that stops teams from demanding a 3,000-record audit on a 4,000-record population — the correction makes the sample nearly exhaustive there, and the right conclusion in that regime is a full audit, not a sample.

#### 4.3.6 Deriving a threshold from an error budget instead of taste

A threshold should be the inverse image of a decision tolerance, not a number chosen for rhetorical weight. The derivation runs backwards from the harm: the consumer states how much coverage loss its decision can absorb before the answer changes; that loss is the tolerance `tau`; the threshold is `1 - tau`, optionally reduced by the checks' own error consumption (a probabilistic dedup structure, a sampling plan).

```text
FORMULA [DC-15]: T = 1 - tau - e_check
WHERE: T        coverage threshold used in the release gate [dimensionless];
       tau      decision tolerance: largest coverage loss that does not change the
                decision, from the consumer's own sensitivity analysis;
       e_check  expected coverage loss introduced by the controls themselves
                (e.g. Bloom false positives, quarantine of ambiguous ties).
CONSTRAINT: tau derived for this measure and decision, not borrowed; e_check >= 0;
            T must exceed the realistic repair ceiling or the gate can never pass.
NUMERICAL EXAMPLE: tau = 0.01 (a 1% swing does not change the routing decision);
                   e_check = 0.000459 (16 bits/key Bloom) -> T = 0.01 - 0.000459
                   subtracted from 1: T = 1 - 0.01 - 0.000459 = 0.989541 -> use 0.9895.
INTERPRETATION: the gate is 98.95%, not 99.5%; the lower number is defensible because
                it is tied to a stated tolerance and already pays for control error.
```

The formula's value is argumentative as much as numerical. A producer asked to meet 99.5% will reasonably ask why 99.5%, and if the answer is "it is standard" the requirement will be negotiated down at the first scheduling conflict. A producer asked to meet 98.95% because the decision tolerates 1% and the dedup filter costs 0.046% has been given a derivation, and a derivation survives reorganisation.

The feasibility clause matters too: if `T` is above the realistic repair ceiling — the coverage achievable when every known defect class is fixed — the gate is a permanent failure and will be ignored. The ceiling is estimated from the historical best month corrected for known-open defects, and the threshold must sit below it by a margin that reflects ordinary noise.

### 4.3.7 Backfill overlap and the double-count window

A backfill that overlaps a live window is the most common cause of transient coverage above one. If a backfill reprocesses days `[d0, d1]` while the live pipeline simultaneously writes the same days, the delivered count for the overlap is inflated by the overlap fraction. Naming the overlap explicitly makes it a scheduling constraint rather than an incident.

```text
FORMULA [DC-16]: R_obs = R_true * (1 + f_overlap)
WHERE: R_obs      observed unique count during the overlap [records];
       R_true     true unique population [records];
       f_overlap  fraction of the population written by both paths [dimensionless].
CONSTRAINT: both paths write the same business keys; dedup on business key with the
            version rule collapses the duplicates; the formula bounds the inflation
            only when the backfill window is fully inside the live window.
NUMERICAL EXAMPLE: R_true = 244,800; f_overlap = 1.0 (full overlap) ->
                   R_obs = 489,600 before dedup; apparent c = 1.9584 against E = 250,000.
INTERPRETATION: c near 2 is the signature of a full backfill overlap, not of excess
                production; the gate should evaluate the post-merge state, not the raw count.
```

The practical control is a mutex: the contract declares that a backfill on days `[d0, d1]` suspends the live writer for those days, or that the merge is idempotent so the overlap is invisible after merge. The second is preferable because it does not require scheduling discipline, but it requires the total version order that Section 8.5 demands.

### 4.4 Limits and counterexample

**A null-rate rule is not a quality metric.** A column whose null rate falls from 4% to 0% looks improved and may indicate a coercion-to-default bug that replaced unknowns with a sentinel. The formula is symmetric in a way the business is not: both a rise and a fall can be the defect. Any rule of the form "null rate must decrease" is therefore invalid; the correct clause is a two-sided range with a documented reason for the lower bound.

**A reconciliation of totals cannot localise an error.** Equal `S` and `C` are consistent with many wrong row sets, including compensating fan-out and loss. The formula's boundary is that it can only *refute* correctness, never establish it. Contracts that rely on a total-only check should be understood as having a lower detection rate than their green status suggests.

**Ratios hide small denominators.** `v = 1.0` computed on 12 rows is not evidence of anything. Every ratio in this sheet needs a minimum-denominator clause, and the contract should state what happens below it: typically "report the count, suppress the ratio, and route to a human". A ratio published without its denominator is a claim that cannot be reviewed.

**Thresholds are not transferable across measures.** A 99% completeness threshold appropriate for a row-count metric can be inappropriate for a sum-based metric, because the loss of 1% of rows by count can be 30% of value by currency. The bound in `DC-08` exists to make that asymmetry explicit; choosing a single threshold across all measures is a modelling error, not a simplification.

**Idempotency does not imply correctness.** A merge that is idempotent on the declared key will happily write the same wrong value twice. Idempotency protects against replay, not against a bad mapping or a wrong join. Contracts often cite "we are idempotent" as a complete answer to duplicate risk; it addresses exactly one of the four ways duplicates become errors.

**The expectation ledger is the real dependency.** Every formula above is only as good as `E`. When `E` comes from the same process that produces `R`, all the ratios collapse toward one by construction. The boundary condition for the entire section is therefore organisational as much as mathematical: there must be at least one count of the population that the producing pipeline cannot influence.

**Takeaway:** Duplicate excess, reconciliation delta, weighted impact, and sample size each answer a question the plain coverage ratio cannot, and each has a specific regime — small denominators, total-only comparison, non-transferable thresholds — where it stops informing.

## 5. Worked Numerical Example

**Scenario statement (ILLUSTRATIVE; no company, pipeline, or measurement is implied).** A retail order-line snapshot dataset is consumed by a daily gross-margin report. The contract declares a grain of one row per `(order_id, line_id)` per snapshot version, with later versions superseding earlier ones for latest-state readers. The business process, a separate reconciliation service, and the finance close calendar together assert an expected 250,000 latest-version lines for the day. The producer delivers 246,000 records. Of those, 1,200 are repeats of business keys already delivered in the same window. Of the 244,800 distinct records, 3,648 fail at least one published rule: 1,904 fail a currency-code allow-list, 1,120 have a negative `extended_amount` on a non-return line, 448 have a null `order_status`, and 176 have an `order_date` outside the declared window. The release rule states `u ≥ 0.99`. Finance's materiality threshold is 1% of the reported total, and the reported total for the window is 41,600,000.00. The availability objective is 15 minutes after the partition's nominal close, and the close is at 00:00 UTC; the evaluation runs at 00:22 UTC with a watermark asserted at 23:58 UTC.

### 5.1 Step 1 — Establish the population and the expectation

The first action is not arithmetic; it is confirming that `E` and `R` describe the same population. The check: the expectation covers the same channels, currencies, and order statuses as the delivered set, and its cutoff matches. A quick falsification test is to compare the expectation's implied daily average against a trailing baseline; a deviation of more than a few percent on a stable business is usually a predicate mismatch rather than a real volume change.

```text
expectation ledger row: population = "all channels, all currencies, status in
{placed, shipped, delivered, cancelled}, order_date = D, latest version only"
asserted E (record-slots)                          = 250,000
delivered records A (including retries)            = 246,000
```

If the finance query that produced `E` had filtered out the marketplace channel, `E` would have been, say, 238,000 and `c` would have been computed against an incomplete denominator. The visible symptom would be `c` close to one with a materially smaller `R` than the trailing baseline. This is the single most common way a completeness score lies.

### 5.2 Step 2 — Deduplicate before anything else

```text
A (delivered, incl. repeats)        = 246,000
D (distinct business keys)          = 244,800
duplicate excess d = (246,000 - 244,800)/246,000
                   = 1,200 / 246,000
                   = 0.00487805  (0.487805%)
R (unique received, adopted for coverage) = 244,800
```

Arithmetic check: `246,000 - 1,200 = 244,800`, and `1,200 / 246,000 = 12/2,460 = 1/205 = 0.004878...`, so the fraction is exactly `1/205`. The deduplication must be on the business key with the declared version tie rule. If the pipeline had deduplicated on a surrogate load id, `D` would have equalled `A`, `d` would have been zero, and the 1,200 repeats would have flowed into the metric as double-counted lines.

### 5.3 Step 3 — Apply the rules and count validity

```text
rule failures (mutually exclusive first-failure attribution):
  currency code not in allow-list                1,904
  negative extended_amount on non-return line    1,120
  null order_status                                448
  order_date outside declared window               176
  ----------------------------------------------------
  total failing records                           3,648
V (passing every rule)   = 244,800 - 3,648 = 241,152
v = 241,152 / 244,800 = 0.98509804  (98.5098%)
```

The addition checks: `1,904 + 1,120 = 3,024`; `3,024 + 448 = 3,472`; `3,472 + 176 = 3,648`. Note the attribution convention: a record failing two rules is counted once, under the first rule in the declared precedence order. Without that convention the per-rule counts exceed the failing total and the arithmetic stops reconciling, which is why the contract must state the precedence.

Fraction check: `241,152 / 244,800`. Cancel the common factor of 96: `241,152/96 = 2,512` and `244,800/96 = 2,550`, so `v = 2,512/2,550 = 0.98509803...`. Independent cross-check: `244,800 * 0.985098 = 241,152.0` (to the nearest record).

### 5.4 Step 4 — Compute coverage and usable coverage

```text
c = R / E = 244,800 / 250,000 = 0.9792   (97.92%)
u = V / E = 241,152 / 250,000 = 0.964608 (96.4608%)

cross-check: v * c = 0.98509804 * 0.9792
                   = 0.9646080...
and directly:  241,152 / 250,000 = 0.964608.  Agreement to 6 decimal places.
```

Component arithmetic: `E - R = 250,000 - 244,800 = 5,200` missing record-slots, which is 2.08% of expectation. `R - V = 3,648` rule failures, which is 1.49% of the delivered unique population. `E - V = 8,848` record-slots that the consumer will never see, which is 3.54% of expectation — exactly `1 - u`.

### 5.5 Step 5 — Evaluate the release rule

The rule is `u ≥ 0.99`. The observed `u = 0.964608` is 2.54 percentage points below the threshold, and the shortfall is 0.025392 in ratio terms. The number of additional valid records needed to reach the threshold is `0.99 * 250,000 - 241,152 = 247,500 - 241,152 = 6,348`. Repairing all 3,648 rule failures would still leave `u = 244,800/250,000 = 0.9792 < 0.99`, because the delivery gap alone is 2.08 points. Conversely, delivering all 5,200 missing records and fixing nothing gives `(244,800 + 5,200 - 3,648)/250,000 = 246,352/250,000 = 0.985408`, still below 0.99. **Both** defects must be repaired to pass, which is the useful finding: a single-team fix cannot satisfy this contract.

```text
path A: fix all rule failures only    -> u = 244,800/250,000 = 0.97920  (fail)
path B: close delivery gap only        -> u = 246,352/250,000 = 0.98541  (fail)
path C: fix both                       -> u = 250,000/250,000 = 1.00000  (pass)
```

Path C is the idealised ceiling; in practice a residual 0.3–1% of records are irreparably defective, so the contract's 0.99 threshold is deliberately set below the realistic ceiling rather than at it. That headroom is the error budget, and it is consumed by ordinary operational noise, not only by incidents.

### 5.6 Step 6 — Assess materiality of the gap

The ratio breach is the process control; the materiality question is what the gap could do to the number.

```text
missing records E - R          = 5,200
high-percentile per-line value m_hi = 900.00
reported total T               = 41,600,000.00
r_hi = 5,200 * 900.00 / 41,600,000.00
     = 4,680,000 / 41,600,000
     = 0.1125  (11.25%)

average per-record value = 41,600,000 / 250,000 = 166.40
optimistic gap impact    = 5,200 * 166.40 / 41,600,000 = 865,280/41,600,000 = 0.0208 (2.08%)
```

The optimist's number (2.08%) and the defensive bound (11.25%) differ by a factor of 5.4 because line values are skewed: the mean is 166.40 while the 99th percentile is 900.00. Finance's materiality threshold of 1% is breached under both estimates, so no interpretation of missingness rescues the number. That agreement across the optimistic and defensive bounds is what makes the hold decision defensible in a review — the decision does not depend on an assumption about which records went missing.

If the optimistic bound had been 0.6% and the defensive bound 11.25%, the decision would genuinely be sensitive to the missingness mechanism, and the correct action would be to characterise the missing records (are they concentrated in one channel? one warehouse?) before deciding. The two-bound spread is therefore a triage device: a narrow spread lets you decide now, a wide spread tells you which investigation to run.

### 5.7 Step 7 — Evaluate freshness

```text
nominal close for partition D-1        = 00:00 UTC
availability objective (close + lag)   = 00:15 UTC
evaluation instant t_now               = 00:22 UTC
watermark asserted by the producer w   = 23:58 UTC  (event time)

overage past the deadline  = 00:22 - 00:15              = 7 minutes
watermark age              = 00:22 - 23:58 (previous day) = 24 minutes
unasserted partition tail  = 00:00 - 23:58               = 2 minutes of event time
last-event lag b           = 00:22 - 00:21 (newest row)  = 1 minute
```

Three quantities disagree, and the contract must name which one the objective is on. The **overage** (7 minutes) answers "are we late against the promised clock" and is the quantity the availability clause should use, because it is the one a consumer can plan against. The **watermark age** (24 minutes) answers "how old is the guaranteed-complete boundary" and is the quantity `DC-04` computes; it differs from the overage by exactly the allowed lag plus the unasserted tail. The **unasserted tail** (2 minutes) is the part of the partition the producer has not yet promised, and it is the number a responder needs to know whether the gap is shrinking.

The `b = 1` minute line is the trap. A dashboard computing `now - max(event_time)` reports one minute of lag and appears healthy while the partition is seven minutes past its deadline with a two-minute hole at the end. This is precisely the arrow-C failure drawn in Section 6.3: the newest row is punctual, and the population is not.

*Verdict.* The availability clause is breached on two independent grounds — the deadline was missed by 7 minutes, and the assertion covers only `23:58`, leaving 2 minutes of event time unpromised. Because the objective is measured from the close rather than from `t_now`, the breach does not resolve by waiting: the partition's data is late in a way that persists until the watermark advances past 00:00.

*Arithmetic check for the responder.* If the watermark advances by one minute per minute of wall clock, the tail clears at `00:22 + 2 = 00:24`, which is 9 minutes past the deadline. That estimate is the one to quote in the incident channel, because it converts a ratio failure into an expected resolution time and lets the decision owner choose between waiting and invoking the fallback.

### 5.8 Step 8 — Decide and record

The combination of a coverage breach and a freshness breach on a filed-adjacent metric triggers the contract's hold clause. The recorded decision:

| Field | Value |
|---|---|
| Decision | Hold the gross-margin report; republish the prior certified value |
| Prior certified value | day D-2 close, dataset version as published, labelled as-of D-2 |
| Reason | `u = 0.964608 < 0.99` and availability overage 7 minutes |
| Materiality bound | 2.08% (optimistic) to 11.25% (defensive) of the reported total, both above 1% |
| Owner notified | Producer engineering owner (rule failures) and source operations (delivery gap) |
| Override considered | Rejected: no logged justification can make an 11.25% bound acceptable for a filing-adjacent metric |
| Follow-up | Expectation ledger review, since 2.08% of expected records were never delivered and no single cause explains it |
| Verification of the decision | Replay the cutoff after repair; the recomputed `u` and `delta` must both pass before the release gate opens |

The alternative path — publishing with a footnote — is the failure this checklist exists to prevent. A footnote converts a control failure into a reader's problem, and the reader of a margin report cannot recompute what is missing.

### 5.9 Step 9 — Verify the repair

The repair verification has three parts, and all three are needed because each catches a different class of mistake.

1. **Recompute the ratios on the replay.** `u` must exceed 0.99 and `v` must not have been improved by relaxing rules. A repair that raises `u` while lowering `v` has widened the rule set rather than fixed the data.
2. **Reconcile the value.** `delta` from `DC-07` at the same cutoff must be within the currency rounding tolerance, and the sign must be explained. A zero delta with a changed row count is suspicious, not reassuring.
3. **Check the repaired records specifically.** The 3,648 previously failing records should either now pass or be explicitly quarantined with reason codes; a repair that makes them disappear from the population is a loss, not a fix.

**Sensitivity note.** The decision to hold is robust to the arithmetic: it survives using the mean instead of the p99 for missing-value impact, and it survives a 20% error in `E` (at `E = 200,000`, `c` would exceed one, which itself indicates a denominator fault). It would *not* survive if the missing 5,200 records were provably all zero-value lines — but that provability is exactly what the contract's quarantine reason codes would have supplied and did not.

### 5.10 Counter-scenario — the release that passed every ratio and was still wrong

The first scenario ends in a hold, which is the easy case because the ratios failed. The harder case is a release where every ratio passes. It is included because a checklist that only catches obvious failures is a checklist that will be declared successful and then ignored.

**Setup.** The same dataset, a later week. `E = 250,000`, `A = 250,400`, `D = 250,100`, `V = 248,400`. Then `c = 250,100/250,000 = 1.0004` — slightly above one, which the team reads as "essentially complete" and waves through — `v = 248,400/250,100 = 0.993203`, and `u = 248,400/250,000 = 0.9936`. Against a 0.99 threshold, this passes with 0.36 points of headroom.

**What happened.** A release changed `extended_amount` from minor to major currency units for one payment provider only, and added its 400 previously-missing records in the same release. The added records pushed `c` above one, which was dismissed as replay noise. Because only one provider changed, the global totals did not move by a recognisable factor — the provider is 6% of volume, so the total fell by about 5.94% of that share, which is `0.06 * 0.99 = 5.94%` of the provider's contribution or roughly 0.36% of the whole. That 0.36% is well inside the normal week-to-week variation, so no ratio flagged it.

```text
provider share of records     = 0.06
value share of that provider  = 0.06 (assumed proportional for illustration)
unit change factor            = 1/100
expected total distortion     = 0.06 * (1 - 0.01) = 0.0594  ->  5.94% of that share
as a share of the whole total = 0.06 * 0.99 = 0.0594 * 0.06 ... 
```

The arithmetic worth writing carefully: if the provider contributes fraction `s = 0.06` of total value and its amounts become `1/100` of before, the new total is `(1 - s) + s/100 = 0.94 + 0.0006 = 0.9406`, so the total falls by `5.94%`. That is a large fall and would be noticed. The reason it was *not* noticed is that the provider's amounts were also restated to include tax in the same release, which raised them by a factor of `1.08` on the same subset. Net factor on that subset: `1.08/100 = 0.0108`. Net total: `0.94 + 0.06*0.0108 = 0.940648`.

So the composite would still be a 5.94% fall — unless the minor-to-major change was applied to a field that the reporting layer re-multiplies. Which is exactly what a second, separate mapping did: the reporting layer had a hardcoded `*100` for that provider, dating from before the source switched to major units. With both errors, the subset's contribution is unchanged to within rounding: `1.08 * 100 / 100 = 1.08` versus the tax change's `1.08`. Two errors cancelled.

**The lesson, stated as a number.** The reconciliation delta was `S - C = 18,432,500.00 - 18,431,120.00 = 1,380.00`, which is `0.00749%` of `S` — comfortably inside a 0.05% tolerance. A totals-based check cannot detect a compensating pair, and this scenario is constructed so that the pair cancels almost exactly. The only detector is the stratified relation assertion: for every record, `abs(extended_amount - quantity * unit_price) <= 0.01` in the declared unit, evaluated per provider and per currency, not globally.

**Discriminating check, run.** Stratify `extended_amount / (quantity * unit_price)` by provider. For the affected provider the ratio is `1.08`, because the amount is in major units while the price is still in minor units, and the tax change scaled the amount. For every other provider the ratio is `1.00`. The global ratio is `0.9988` — indistinguishable from one. This is the concrete demonstration of why the compatibility lattice's upper-right cell needs a value-level, stratified check: every aggregate-level test passes by construction.

**Bounded fix.** Revert the provider's mapping to minor units and re-derive the tax adjustment on the reverted basis; dual-publish for one full cycle; add the per-provider relation assertion as a blocking check; and record the `c > 1` observation as the signal it was — a 400-record addition on a population of 250,000 is a 0.16% step, which on a stable business is not noise.

**Takeaway from the counter-scenario.** `c > 1` is always worth a query, thresholds near the top of the distribution are the ones that hide compensating errors, and the check that would have caught this is one line of arithmetic evaluated per stratum rather than globally.

## 6. Visual / Geometric Intuition

### 6.1 The nested-set picture, drawn to scale

The four counts form nested sets, so a proportional bar makes the losses comparable at a glance. On the worked example's numbers:

```text
E = 250,000  |##################################################|  100.00%
R = 244,800  |###############################################   |   97.92%   <- 5,200 absent
V = 241,152  |#############################################     |   96.46%   <- 3,648 invalid
Q <= V       |###########################################       |   <=96.46%  <- access policy
             ^                                                 ^
             E-R = 2.08 pts        R-V = 1.46 pts        V-Q = access loss
```

(Percentages are of `E`, so the two leaks add: `2.08 + 1.46 = 3.54`, matching `1 - u` to rounding; measured against `R`, the rule-failure share is `3,648/244,800 = 1.49%`, which is the figure to quote when talking to the producer about its own output.)

The visual point is that the two leaks have *different widths and different owners*. A team that reports only the total gap (3.54 points) loses the information that 59% of the gap is delivery loss and 41% is rule failure — `5,200/8,848 = 58.8%` and `3,648/8,848 = 41.2%` — and those are repaired by different people on different timescales. Delivery loss is usually an upstream or transport problem measured in hours; rule failure is usually a mapping or validation-gap problem measured in days.

### 6.2 Why ratios near one are dangerous to eyeball

A ratio of 0.9646 looks close to 1, and human perception compresses the last few percent. In currency terms the same ratio is 8,848 records missing from a 250,000-record population, and the defensive materiality bound is 11.25% of the reported total. The geometric intuition to cultivate is that the *ratio* is linear while the *consequence* is weighted, so the last 3% of a ratio can carry a double-digit share of the value.

```text
ratio axis:    0.90 -------- 0.96 -------- 0.99 ---------- 1.00
                         ^                    ^
                    observed u            required u
                        |<-- 2.54 points -->|
consequence:    |<------- 11.25% of reported value at risk ------>|
```

The two axes are not linearly related, and the contract must be written in the axis that the decision cares about. This is the geometric content of `DC-08`: it is a change of axis from record-share to value-share.

### 6.3 The freshness clock, and the two arrows that get confused

```text
 event time --->                                                now
   23:00        23:30        00:00        00:15        00:22
     |            |            |            |            |
     |            |            |            |            +-- t_now
     |            |            |            +-- availability deadline (close + 15m)
     |            |            +-- partition nominal close
     |            +-- watermark asserted here (23:58) .. actually 23:58
     +-- start of partition

 arrow A (correct): deadline -> now                = 7 min overage
 arrow B (correct): watermark -> now                = 24 min age
 arrow C (wrong):   max(event_time) -> now          = 1 min  ("looks fine")
```

Arrow C is the one that appears on dashboards because it is trivial to compute from data alone and needs no producer assertion. It is also the one that is shortest during an incident, which makes it worse than useless: it is anti-correlated with the thing it is supposed to measure. The geometric image to keep is that arrow C measures the *tip* of the delivery, while arrows A and B measure the *body*.

### 6.4 The compatibility lattice as a plane

```text
                        semantic change?
                     no                 yes
                 +----------------+----------------+
   parse         | ADDITIVE       | SILENT DRIFT   |
   still works   | safe, notify   | most dangerous |
                 +----------------+----------------+
   parse fails   | STRUCTURAL     | TOTAL BREAK    |
                 | loud, planned  | loud + semantic|
                 +----------------+----------------+
```

The upper-right cell is the one that costs money, because every loud signal (CI failure, parse error, alert) is silent there. Its signature is a distribution shift without a type change: a currency column whose values all become 100x smaller, a count column whose denominator changed, an enum whose member ordering changed. Section 8's semantic failure entry is aimed precisely at this cell, and the only reliable detector is a value-level comparison across the release boundary rather than a schema diff.

### 6.5 The expectation ledger as a tripod

```text
        [business process volume]        [independent control total]
                    \                          /
                     \                        /
                      \                      /
                       [ expectation E ]
                              |
                    [ tolerance band ]
                              |
                  outside band -> investigate
```

Completeness is only meaningful if `E` stands on at least two legs: a business-process volume estimate and an independent control total. A single leg is a guess with a decimal point. The tripod image is useful in reviews because "where does `E` come from?" is the question that most often exposes a contract as unenforceable, and the answer must name at least two sources that could disagree.

### 6.6 Quarantine as a side channel, not a bin

```text
  delivered --> [rule evaluation] --pass--> certified set --> consumer
                        |
                        +--fail--> quarantine table (reason code, record hash,
                                    contract version, first-seen, attempt count)
                                        |
                                        +--> repair workflow --> re-entry attempt
                                        +--> audit trail (never deleted before retention)
```

The visual distinction that matters: a bin is where data goes to be forgotten, a side channel is where data goes to be *counted and returned*. Every quarantined record must carry the contract version that rejected it, because after a threshold change the same record may be acceptable, and the audit question "how many records did version 1.4 reject that version 1.5 accepts" is unanswerable without that column.

### 6.7 The four-plane picture as a stack of filters

The planes from 2.3 are worth drawing as a physical stack, because the drawing makes the independence obvious and also shows why a failure in a lower plane cannot be compensated by a pass in a higher one.

```text
        incoming records
             |
     [plane 1 SCHEMA]      drop: unparseable, wrong type, unknown enum   -> hard reject
             |  n_1
     [plane 2 IDENTITY]    drop: duplicate key, orphan reference         -> quarantine
             |  n_2
     [plane 3 SEMANTICS]   drop: unit outside band, relation violated    -> quarantine
             |  n_3
     [plane 4 DELIVERY]    drop: nothing (this plane computes, not filters)
             |  n_4
        certified records  = n_4
```

Each filter has a count, and the counts compose: `n_4 = n_0 * (1 - f_1)(1 - f_2)(1 - f_3)` where `f_i` is the drop fraction at plane `i` computed on the survivors of the previous plane. That product is exactly the chain of ratios in `DC-03` generalised to four factors, and it is the reason a single "quality score" can never be decomposed back into causes without keeping the intermediate counts. The plane that does not filter — delivery — is still a plane because it computes the ratios `c` and the age `a`, which no record-level filter can produce.

The practical corollary: store `n_0` through `n_4` per partition. They are five integers, they cost nothing, and they make every later question answerable. Pipelines that store only the final count cannot answer "was the loss at identity or at semantics?" after the fact, and the answer is usually a one-line query away if the counts were kept.

### 6.8 Threshold placement on the observed distribution

A threshold is a point on a distribution, and drawing the distribution next to the threshold is the fastest way to see whether the gate is calibrated or theatrical.

```text
  frequency
     |
     |        ####
     |      ########
     |    ############
     |  ##################          threshold candidates
     |########################      |          |           |
     +--------------------------------------------->  u
        0.980   0.985   0.990   0.995   0.998   0.999
                    ^                  ^
              observed p50        observed p99.9
```

A threshold placed at the observed median fires half the time and will be muted. A threshold placed beyond `p99.9` fires rarely and, when it does, may fire for reasons unrelated to the decision. The defensible placement is between the tolerance-implied floor from `DC-15` and the realistic repair ceiling, and it should sit where the *consequence* changes rather than where the histogram has a convenient gap. The drawing also exposes a common error: a threshold set just above the worst observed value, which is a description of history rather than a control.

### 6.9 Quarantine growth against population growth

The one chart that catches silent narrowing is quarantine share over time, drawn against population.

```text
 share
 8% |                                        ....
 6% |                              ....````
 4% |                    ....`````
 2% |        ....````
 0% +------------------------------------------------> weeks
      W1   W2   W3   W4   W5   W6   W7   W8   W9  W10
```

A rising share means the certified population is shrinking relative to the delivered one, so a metric computed on `V` is increasingly a metric about a subpopulation that happens to pass the rules. That is survivorship bias introduced by the quality apparatus itself, and it is invisible in `v` alone because `v` is computed as a ratio of the survivors. The chart's shape matters more than its level: a step change points at a new rule or a mapping change, a ramp points at upstream drift, and a sawtooth points at a repair job that runs on a fixed cadence.

**Takeaway:** Keep the per-plane counts so losses can be decomposed, place thresholds where consequence changes rather than where history ends, and watch quarantine share as a survivorship signal.

## 7. Comparison / Selection Table

### 7.1 Choosing the enforcement gate strength

| Situation | Choose | Why | Do not choose | Fallback if the preferred gate is infeasible |
|---|---|---|---|---|
| Filed or regulated number | blocking gate on `u`, `v`, `delta`, and freshness | a wrong number is not recoverable by a later correction | warn-only with a dashboard | blocking on `u` only, plus a human signed reconciliation |
| Operational dashboard refreshed intraday | quarantine + freshness blocking | partial data with visible exclusions is more useful than nothing | all-or-nothing blocking | previous complete snapshot, clearly labelled |
| Exploratory analysis | warn-only, no consumer obligation | cost of a wrong conclusion is low and reversible | full blocking apparatus | none needed; document the warning |
| ML feature feeding a live model | blocking on schema, semantics, and drift band | silent drift degrades predictions without an error | warn-only | shadow scoring until the band is satisfied |
| Regulatory report with restatement allowed | quarantine + restatement window | late facts are legitimate and corrections are auditable | last-write-wins overwrite | immutable history + explicit as-of labels |
| Cross-team shared dimension | blocking on identity + referential closure | a broken dimension corrupts every downstream join | per-consumer patching | publish a versioned dimension with a frozen snapshot |

### 7.2 Choosing the duplicate policy

| Delivery behaviour | Policy | Consequence for the consumer | Failure it prevents |
|---|---|---|---|
| At-least-once transport, stable business key | idempotent merge on key + version | one row per key, latest by rule | double-counted measures |
| Legitimate restatements (corrections) | keep all versions, expose latest selector | consumer must state as-of | loss of audit trail |
| Retries with no stable key | quarantine and reconcile against a control total | visible loss, no silent duplication | unmergeable duplicates |
| Late facts after partition close | accept into history; freeze or recompute certified value | documented restatement | revision of a filed number |
| Out-of-order arrival with equal versions | quarantine the tie, escalate | small, bounded gap | arbitrary last-write-wins |

### 7.3 Choosing a grain representation

| Need | Representation | Cost | When it fails |
|---|---|---|---|
| Immutable facts, append-only | one row per event, event time primary | cheapest to repair | corrections must be modelled as new events |
| Current state of mutable entities | one row per key, latest version materialised | cheap reads, expensive rebuilds | audit questions about prior state |
| Full history of mutable entities | one row per key per version, immutable | storage grows with edit rate | consumers forget the selector and double-count |
| Snapshot at a cutoff | one partition per cutoff, full population | storage multiplies by snapshot count | backfills create inconsistent partitions |
| Slowly changing dimension | versioned rows with validity intervals | join must be as-of, not equality | equality join silently fans out |

### 7.4 Change classification and required approvals

| Change | Class | Consumer action required | Approval | Gate |
|---|---|---|---|---|
| New optional column, documented | additive | none for named-column readers | producer owner | schema check only |
| New enum member | additive if consumers tolerate unknown | verify reader default branch | producer + each consumer owner | semantic note in changelog |
| Widening a numeric type | additive | none | producer owner | schema check |
| Currency minor units → major units | semantic | remap or re-derive | all consumers, written | dual-publish + value comparison |
| Net → gross of tax in an amount | semantic | restate or re-baseline | all consumers + finance sign-off | dual-publish + reconciliation |
| Column rename | structural | update references | all consumers | dual-publish with expiry |
| Dropping a column | structural (breaking) | remove dependency first | all consumers | migration window, then removal |
| Business key change | structural + semantic | rebuild joins and history | all consumers + governance | new version, parallel run |
| Partition boundary or timezone change | semantic | adjust cutoffs | all consumers | dual-run one full cycle |

### 7.5 Choosing the expectation source

| Source of `E` | Independence | Cost | Failure mode |
|---|---|---|---|
| Same pipeline's source file count | low | free | blind to upstream loss |
| Source system row count at cutoff | medium | cheap query | cutoff mismatch, predicate drift |
| Business volume forecast | medium | planning process | conflates demand change with data loss |
| Independent reconciliation service | high | duplicate extraction | lag between services |
| Financial ledger / control total | high | manual or batch | only covers value-bearing records |
| Contractual counterparty file | high | external dependency | timing and format dependencies |

### 7.6 Selecting the freshness measure for a decision

| Decision cadence | Measure | Target form | Wrong choice and its symptom |
|---|---|---|---|
| Intraday operational | watermark age vs availability deadline | minutes, per partition | last-event lag: green during a stalled region |
| Daily close | partition closed + restatement window | clock time, per day | row age: cannot distinguish late from missing |
| Weekly/monthly reporting | certification timestamp | date + version | ingestion age: unrelated to when the number was certified |
| Real-time alerting | end-to-end detection latency | seconds, per event class | table freshness: measures the wrong stage |

### 7.7 When to split one contract into two

| Signal | Split | Keep as one |
|---|---|---|
| Two consumers need different cutoffs | separate contracts per consumer group | if one cutoff is a strict superset |
| One consumer needs latest state, another needs history | split, with history as the parent | if the latest view is a cheap projection of history |
| Materiality thresholds differ by an order of magnitude | split | if both are far from the observed values |
| Access policies differ by row or column | split by policy, same producer | if the difference is uniform and simple |
| Producers differ | always split | never merge across producers |

### 7.8 Reading a quality claim: which number is being quoted?

| Claim as stated | Most likely quantity | What to ask | Why it matters |
|---|---|---|---|
| "99.5% complete" | `c` with an unknown `E` source | where does `E` come from? | may be `R/R` |
| "99% of tests pass" | assertion pass rate, not record coverage | how many records per assertion? | 400 assertions can cover 4 columns |
| "Zero data quality incidents" | none detected, not none occurring | what would detection require? | detection coverage is the real metric |
| "98% accuracy" | undefined without a comparator | accuracy against which reference? | likely a stale golden set |
| "Fresh every 5 minutes" | pipeline schedule, not watermark | what is the completeness cutoff? | schedule is a start time, not a guarantee |

**Takeaway:** Select gate strength from consequence asymmetry, select the duplicate policy from the delivery guarantee plus the business mutation semantics, and treat every proposed change as a row in the compatibility lattice with named approvers.

## 8. Failure Modes & Edge Cases

### 8.1 How to read this section

Each entry is a decision tree rather than a symptom list. The four fields answer, in order: what could produce this observation (competing causes), which observation separates them (discriminating check), how much you are allowed to change right now (bounded fix), and what evidence closes the issue (verification). The bounded fix is deliberately narrow — a freeze, a quarantine, a fallback — because the correct action during a breached close is almost never a redesign.

A contract failure that has no bounded fix is a design failure, not an incident. If the only way to restore service is to rewrite the transformation, then the missing artefact is a fallback clause in the contract, and that is what the post-incident review should add. This is the operational reason Section 1 insists on a named fallback per gate.

### 8.2 F-01 — Grain changed while the schema stayed valid

- **Symptom.** An aggregate rises by a plausible but unplanned amount on the first run after a producer release; row counts and schema both look normal; the dashboard's row-count tile is unchanged because the count assertion was written against a looser predicate.
- **Competing causes.** (a) a genuine demand spike; (b) join fan-out introduced by a new relationship; (c) a grain change upstream that adds rows without changing the declared key's format; (d) a duplicate-delivery storm that the dedup policy absorbed differently.
- **Discriminating check.** Compute `count(*)`, `count(distinct business_key)`, and `count(*)/count(distinct business_key)` for the three days before and the day after the release. A demand spike moves all three proportionally; a grain change moves the ratio while leaving distinct keys roughly constant when the new grain duplicates a parent entity; a fan-out moves rows and the ratio while distinct keys stay flat, and additionally shows up as duplicated parent attributes in a `GROUP BY parent`. Then compare against an independent control total, which a demand spike will not violate and a grain change will.
- **Bounded fix.** Freeze the affected aggregate and republish the prior certified value. If a latest-version selector exists, re-point the consumer to it; if the mutation rule is ambiguous, quarantine the post-release partition and mark `u` unknown rather than computing it against a grain you cannot state.
- **Verification.** Replay the pre-release cutoff with the old and new code; the outputs must match exactly. Add a grain assertion (max rows per key tuple equals one, or the pair count equals the declared version count) to the producer's CI so the next release fails before it ships.

### 8.3 F-02 — "Complete" batch lacks a trustworthy denominator

- **Symptom.** Completeness reports 100% (`c = 1.0`) while downstream totals visibly fall; the expectation query and the delivery query agree because they read the same table.
- **Competing causes.** (a) genuine completeness with a demand drop; (b) `E` computed from `R` after the fact; (c) `E`'s predicate filtered a segment that `R` also filtered, so both are consistently incomplete; (d) timezone mismatch that shifts both counts by the same offset.
- **Discriminating check.** Take an independent control total — a source-system count at the same cutoff, a ledger sum, or a contractual counterparty file — and compare *its* predicate with the expectation's predicate line by line. If the two predicates differ on any clause (channel list, status list, date boundary operator `>=` versus `>`), the denominator is wrong. A useful secondary probe: compare `E` today against a trailing 28-day baseline; a 6% step change in a stable business is a predicate change far more often than a volume change.
- **Bounded fix.** Set `c` to `unknown`, not 1.0, and disable the release gate that depends on it. Publish the consumer's prior value. File the predicate reconciliation as the repair.
- **Verification.** Version the expectation definition and store its predicate hash with every evaluated window. On the next run, assert the hash; a change to the hash requires a review, which prevents silent redefinitions of completeness.

### 8.4 F-03 — Schema-compatible release changes meaning (the silent drift cell)

- **Symptom.** All schema checks pass, all row counts are within tolerance, and a downstream ratio shifts by a few percent. No alert fires. The shift is noticed weeks later in a business review.
- **Competing causes.** (a) genuine behavioural change in the underlying process; (b) unit change (minor→major currency, grams→kilograms); (c) tax inclusion changed from net to gross; (d) denominator changed (per-order to per-line, per-active-user to per-account); (e) enum ordering or encoding changed while member names stayed; (f) null semantics changed from "unknown" to "not applicable".
- **Discriminating check.** Run a value-level distribution comparison across the release boundary on the affected field: min, max, mean, p50, p99, count of zeros, count of nulls, and the ratio of the field to a stable reference field (for example amount per line). A unit change shows a clean factor (100x, 1000x) in the ratio to the reference. A denominator change shows a step in the ratio of the metric to its population count. A tax change shows a factor close to `1 + tax_rate` on the affected subset only, which is why the comparison must be stratified by product or region rather than run globally.
- **Bounded fix.** Isolate the new semantic under a separate version, dual-publish both encodings with an expiry date, and recompute only with an approved mapping. Freeze any filed number produced by the changed field. Do not "fix" the historical numbers in place.
- **Verification.** Dual-run both definitions over a bounded period (typically one full business cycle) and reconcile at both row and value level. Retire the old version only when the deltas are explained record by record and every consumer has confirmed its switch.

### 8.5 F-04 — Late or duplicate delivery corrupts a snapshot

- **Symptom.** A latest-version table contains two rows for one business key, or one row whose values mix data from two delivery attempts. Downstream totals double or oscillate.
- **Competing causes.** (a) at-least-once transport with retries; (b) a legitimate backfill overlapping the live window; (c) producer clock skew making an older version appear newer; (d) a missing or non-total version tie rule, so "latest" is decided by write order; (e) two producers writing to the same key space.
- **Discriminating check.** For the affected keys, print business key, version, event time, ingestion time, payload hash, and source file identifier side by side. If payloads are identical, it is a retry and the merge is failing to be idempotent. If payloads differ and versions are equal, the tie rule is missing. If payloads differ and versions differ but the winner looks wrong, the version field is not monotone in business time — which indicates clock skew or a producer that computes version from a local timestamp.
- **Bounded fix.** Quarantine the tie instead of resolving it arbitrarily. Rebuild the latest view from immutable versions with an explicit ordering key (version integer, then event time, then a deterministic tiebreak such as payload hash). Never use last-write-wins on ingestion time for a corrected entity.
- **Verification.** Replay a fixture containing a duplicate, an out-of-order arrival, and a tie. Assert the resulting latest view is byte-identical to the source's own current-state extract for the affected keys. Keep that fixture in CI permanently; it is the only regression test that catches a policy regression in the merge.

### 8.6 F-05 — Ownership exists on paper and nobody responds

- **Symptom.** A violation is detected, the alert fires into a channel, and the decision is released anyway. Post-incident, no one can name who had authority to stop it.
- **Competing causes.** (a) alert routing broken or pointing at a departed engineer; (b) severity undefined, so the alert looks like noise in a busy channel; (c) no decision consequence attached, so nobody's day is affected; (d) responders lack the permission to pause the release, only the responsibility.
- **Discriminating check.** Run an escalation drill: publish a deliberately breaching fixture to a non-production instance of the contract and time the acknowledgement and the hold. If acknowledgement exceeds the stated response target, or if the responder cannot execute the hold because permissions are missing, the contract is not enforced regardless of its text.
- **Bounded fix.** Assign a named role (not a team alias alone), a response target in minutes, and executable hold authority. Route violations to a system that creates a case rather than a chat message, because chat has no acknowledgement accounting.
- **Verification.** Quarterly drill with recorded timestamps for detection, acknowledgement, diagnosis, mitigation, and closure. The artefact to keep is the timing table, not the meeting note.

### 8.7 F-06 — Quarantine becomes a silent landfill

- **Symptom.** Quarantined row counts grow steadily, no one reviews them, and after a few months a material share of the population lives in the side table while the metric reports "passing".
- **Competing causes.** (a) a rule that is too strict for legitimate edge cases; (b) a mapping defect that fails a whole segment; (c) a schema drift upstream that produces one failure reason at high volume; (d) genuinely poor source quality with no owner.
- **Discriminating check.** Break quarantine volume down by reason code and by source segment. A single reason code dominating, concentrated in one segment, is a mapping or rule defect; a long tail spread across segments is source quality. Then check the trend of `Q/V`: if quarantine growth is not matched by population growth, the certified metric is silently narrowing its base, which is a form of coverage loss that the `u` formula does not see unless quarantine is counted as invalid (it must be).
- **Bounded fix.** Treat quarantined records as failing for coverage purposes — they are, by definition, not in `V`. Set an explicit maximum quarantine share with a gate, and add a review cadence for the top reason code.
- **Verification.** Monthly reconciliation between quarantined volume, repaired volume, and expired volume; the three must sum to the previous period's open count plus this period's new arrivals.

### 8.8 F-07 — Threshold theatre

- **Symptom.** The contract states a 99% objective, observed values are 99.6% for eleven months, and the threshold has never been calibrated against a decision consequence. Then one month at 98.9% triggers an escalation whose cost exceeds any plausible harm.
- **Competing causes.** (a) the threshold was chosen for rhetorical weight; (b) it was copied from an unrelated dataset; (c) it was set from the best month rather than the tail; (d) a legitimate process change moved the distribution permanently.
- **Discriminating check.** Plot the observed distribution of `u` over twelve months with the threshold marked; then compute, for each candidate threshold, the expected number of breach events per year and the harm avoided per prevented release. A threshold in the far tail of a stable distribution buys no protection and generates occasional expensive theatre; a threshold above the median of a noisy distribution will fire constantly and be muted.
- **Bounded fix.** Recompute the threshold from the error budget implied by the decision tolerance, publish the reasoning, and attach a review trigger for a distribution change rather than a date.
- **Verification.** After one quarter, compare breach count against the predicted count. Persistent large divergence means the distribution model is wrong, not that the data got better.

### 8.9 F-08 — Access policy quietly truncates the population

- **Symptom.** The dataset passes every quality gate and one consumer's total is persistently 3% lower than another's, with identical code.
- **Competing causes.** (a) different cutoff; (b) different filter in the consumer's transform; (c) row-level access policy applied at read time; (d) column masking that changes a join key; (e) a view that joins to an authorization table with an inner join, dropping rows.
- **Discriminating check.** Compare row counts for the two principals at the same cutoff with `SELECT count(*)` and no predicate. If they differ, the access layer is filtering, which is by design and must be declared — the contract's `Q` term exists for exactly this. Then check whether masked columns are used in keys or predicates, because masking a key collapses distinct values and produces aggregation errors rather than row loss.
- **Bounded fix.** Declare `Q` in the contract with the policy reference, and quote coverage against `Q` where the decision is per-consumer. Publish both `V` and `Q` so the difference is visible rather than mysterious.
- **Verification.** A periodic reconciliation between the unrestricted count and the sum over policy partitions; the difference must equal the documented exclusion count.

### 8.10 F-09 — The expectation ledger drifts out of date

- **Symptom.** `c` exceeds one, oscillates, or requires a manual bump each month. Over time the expectation becomes a copy of yesterday's actual.
- **Competing causes.** (a) business volume genuinely growing; (b) a new channel added to the source but not to the expectation; (c) a channel retired but still in the expectation; (d) seasonality absorbed into a flat daily expectation; (e) an expectation built from a superseded partition.
- **Discriminating check.** Decompose the expectation by segment (channel, region, product family) and compare each segment's expectation to its actual. A single segment with a persistent large deviation is a definitional gap; deviations spread proportionally across all segments are volume or seasonality and belong in the expectation model.
- **Bounded fix.** Rebuild `E` as a sum over declared segments with an explicit seasonal rule, and store the segment decomposition with version. Do not adjust the total without adjusting the decomposition, because a hand-adjusted total loses the ability to localise the next breach.
- **Verification.** Assert that the expectation's segment list matches the source's segment list at every evaluation; a mismatch is a contract violation on the ledger itself.

### 8.11 F-10 — Contract and reality diverge after an unlogged change

- **Symptom.** The contract says one thing, the pipeline does another, and no one can date the divergence. Consumers built against the document break in ways the document forbids.
- **Competing causes.** (a) an emergency hotfix that bypassed review; (b) an infrastructure migration that changed a default (timezone, null handling, decimal precision); (c) a producer-side schema evolution tool with implicit coercion; (d) a consumer's local patch that became the de facto definition.
- **Discriminating check.** Hash the contract's machine-readable portion and the pipeline's effective configuration, and compare both against the hashes recorded at the last certified release. Divergence with no corresponding contract version increment is an unlogged change; the timestamps tell you which came first.
- **Bounded fix.** Re-certify the contract against the actual pipeline, increment the version even if nothing "should" have changed, and record the divergence and its cause in the changelog.
- **Verification.** Make the contract file a build input: the pipeline reads its key list, rule thresholds, and time semantics from it, so a divergence is a build failure rather than a documentation defect.

**Takeaway:** Every entry above ends in a check that distinguishes causes before any repair, and in a verification that would fail if the repair were cosmetic; a contract whose failures cannot be diagnosed this way is under-specified, not unlucky.

## 9. Complexity & Cost Analysis

### 9.1 The cost model for a contract check

Each clause has a cost that must be paid before every release, so the total is a per-run quantity that must fit inside the delivery window. The four components are scan cost, state cost (deduplication and uniqueness), lookup cost (referential checks), and human cost (review, drill, calibration). Only the first three scale with volume; the fourth scales with the number of clauses and their ambiguity.

```text
FORMULA [DC-11]: C_run = n * (c_row) + s * (c_state) + j * (c_lookup) + h
WHERE: n        number of records scanned per run [records];
       c_row    per-record validation cost [seconds/record], dominated by parsing
                and predicate evaluation rather than by I/O;
       s        peak distinct-key state [keys];
       c_state  per-key state cost [bytes/key] (hash entry, version, checksum);
       j        number of referential lookups [lookups];
       c_lookup per-lookup cost [seconds/lookup];
       h        fixed human review cost per run [seconds].
CONSTRAINT: costs are for one evaluation; the deadline requires C_run < window;
            c_row and c_lookup must be measured, not assumed, for the deployed engine.
NUMERICAL EXAMPLE: n = 40,000,000 records, c_row = 2e-7 s/record ->
                   n*c_row = 8.0 s of pure CPU work; s = 40,000,000 keys,
                   c_state = 32 bytes/key -> 1.28 GB of state;
                   j = 40,000 lookups at 5e-5 s -> 2.0 s; h = 60 s of human review.
INTERPRETATION: CPU and state are trivial relative to the 60-second human step, so
                automating the review is worth more than optimising the scan — but the
                1.28 GB state must still fit in memory to avoid a partitioned rewrite.
```

The state term deserves emphasis because it is the term that fails silently. `40,000,000 * 32 bytes = 1,280,000,000 bytes = 1.28 GB` at a decimal gigabyte, or `1,280,000,000 / 1,073,741,824 = 1.1921 GiB` in binary units. A hash table typically needs a load factor below 0.75 and per-entry overhead beyond the key payload, so the practical planning figure is nearer 2.5 GB. When that exceeds the executor's memory, the engine spills or repartitions, and the check's runtime jumps by an order of magnitude — usually without any error, because spilling is a performance event, not a failure.

### 9.2 Worked sizing: hourly uniqueness state

A common contract clause is "the business key is unique within the hour". The state required is the number of distinct keys in the hour, not the number of records.

```text
FORMULA [DC-12]: S = r * 60 * bytes_per_key
WHERE: S            state bytes [bytes];
       r            arrival rate [records/minute];
       bytes_per_key 32 bytes (16-byte key + 8-byte version + 8 bytes bookkeeping).
CONSTRAINT: one hour of retention; no replication, index, or hash-table overhead.
NUMERICAL EXAMPLE: r = 50,000 records/minute;
                   S = 50,000 * 60 * 32 = 96,000,000 bytes = 91.55 MiB
                   (96,000,000 / 1,048,576 = 91.5527).
INTERPRETATION: a single executor comfortably holds the key set; with a 0.75 load
                factor and 2x bookkeeping the planning figure is about 244 MiB.
```

Doubling the retention to two hours doubles the state, which is the arithmetic that makes "keep a rolling 24-hour dedup window" a 2.2 GB planning problem at this rate rather than a configuration detail. When the state cannot fit, the standard alternatives are partitioning the check by a salt on the key (with a reconciliation pass across partitions) or replacing exact uniqueness with a bounded-error structure.

### 9.3 Probabilistic deduplication and its error budget

A Bloom filter trades exactness for a fixed, tiny state. Its false-positive rate for `k` hash functions over `m` bits holding `n` items is `(1 - e^{-kn/m})^k`, minimised at `k = (m/n) ln 2`.

```text
FORMULA [DC-13]: fpr = (1 - exp(-k*n/m))^k ,  k* = round((m/n) * ln 2)
WHERE: fpr  probability a new key is wrongly reported as seen [dimensionless];
       m    bit array size [bits]; n  inserted keys [keys]; k  hash count.
CONSTRAINT: independent uniform hashes; no deletions; n known within a factor;
            false negatives are impossible, only false positives.
NUMERICAL EXAMPLE: 10 bits/key -> k* = round(6.93) = 7;
                   fpr = (1 - exp(-7/10))^7 = (1 - 0.4966)^7 = 0.008194.
INTERPRETATION: about 0.82% of genuinely new keys would be suppressed as duplicates,
                which is a silent coverage loss of 0.82 percentage points — larger than
                a 0.5-point release tolerance, so this structure is not acceptable alone.
```

The table makes the trade-off concrete at four sizes, all computed with the optimal `k`:

| Bits per key `m/n` | Optimal `k` | False-positive rate | State for 40M keys | Acceptable for a filed metric? |
|---:|---:|---:|---:|---|
| 8 | 6 | 2.16% | 40 MB | no — exceeds typical tolerance by 4x |
| 10 | 7 | 0.819% | 50 MB | no — borderline breach of a 0.5% tolerance |
| 12 | 8 | 0.314% | 60 MB | only with a secondary exact pass |
| 16 | 11 | 0.0459% | 80 MB | yes, with the loss counted in `u` |

The correct engineering conclusion is not "Bloom filters are bad" but "a filter's error rate consumes coverage budget, so the contract must reserve that budget explicitly". A 0.0459% false-positive rate at 16 bits per key is acceptable only because the release threshold leaves headroom; at 8 bits per key the filter alone consumes more than a fifth of a 10-point tolerance.

### 9.4 Sampling cost versus exhaustive checks

Exhaustive semantic checking is impossible when the check requires human judgement, so the sample size from `DC-10` sets the cost. At 2,976 records and a 20-audits-per-hour rate, the cost is `2,976 / 20 = 148.8` auditor-hours. If the check runs quarterly, that is 595 hours per year — a headcount-scale commitment. The design response is to stratify: audit exhaustively where the population is small or the value is concentrated, and sample only the long tail, weighting the audit to the segments that carry material value.

```text
FORMULA [DC-14]: H = n / rate
WHERE: H     audit effort [hours]; n  sample size [records]; rate [records/hour/auditor].
CONSTRAINT: rate measured on the actual audit task; excludes training and rework.
NUMERICAL EXAMPLE: n = 2,976; rate = 20 -> H = 148.8 hours.
INTERPRETATION: at 8 hours/day this is 18.6 auditor-days, so a quarterly cadence needs
                a dedicated part-time reviewer, not an ad-hoc request.
```

### 9.5 The cost of the wrong gate strength

Two error costs are asymmetric and must both be priced. A false release publishes a wrong number; the cost is the decision harm plus the trust repair. A false hold blocks a decision; the cost is delay plus the organisational pressure to weaken the gate — and that pressure is itself a cost, because it accumulates and eventually succeeds. This is why gates set far above the observed distribution are actively harmful: they produce false holds whose only visible effect is pressure to disable the check.

| Scenario | Release on breach | Hold on breach | Observed asymmetry |
|---|---|---|---|
| Filed revenue number | misstatement, restatement, audit finding | one day late, escalated | hold dominates by a wide margin |
| Intraday inventory view | planner acts on stale stock | planner waits, workaround call | release often dominates |
| Exploratory segmentation | analyst sees a slightly wrong cut | analysis stalls | release dominates |
| Model training features | degraded model, slow to detect | retrain delayed | hold unless drift is severe |
| Customer-facing personalisation | wrong offer, recoverable | no personalisation | roughly balanced |

### 9.6 Repair cost and the economics of prevention

The cost of prevention has to be compared with the cost of the incident it prevents. A single contract with one grain assertion, one reconciliation, and one watermark costs a few engineering-days to build and a few minutes per run. The incident it prevents has a cost dominated by investigation time, which for a metric dispute is typically measured in person-days because the investigation is a search over hypotheses rather than a known procedure. Prevention is justified when the expected number of disputes per year times the investigation cost exceeds the build cost — a calculation that is usually obvious once written down, and usually never written down.

The cheaper prevention is not more tests but better failure messages. A check that fails with "assertion violated on column amount" costs an hour of investigation; the same check failing with "1,120 rows have negative amount on non-return line, first seen in source file X at 03:14, contract v1.4 rule R-07" costs minutes. Instrumenting the failure message is nearly free relative to building the check, and it is the highest-return investment in the whole apparatus.

**Takeaway:** Price each clause as scan plus state plus lookup plus human time, remember that a probabilistic check consumes coverage budget, and prefer better failure messages over more assertions.

## 10. Key Theorems / Results Reference

### 10.1 Results table

| Result | Statement used here | Essential condition | Boundary — where it stops applying |
|---|---|---|---|
| Ratio composition | `V/E = (V/R)(R/E)` for nested sets `V ⊆ R ⊆ E` | sets are genuinely nested and counted on the same population | fails if V and R are deduplicated on different keys or windows differ |
| Monotone loss | `u ≤ min(c, v)` | same nesting as above | violated when `R > E` (replay), which signals a fault rather than a gain |
| Conservation | a value-preserving transform satisfies `sum(source) = sum(consumer)` for the same population and cutoff | no filtering, no fan-out, no unit change, no rounding drift | equal sums do not imply equal rows; only refutation is valid |
| Key-level completeness | two relations agree on a key set iff both anti-joins are empty | key is declared and stable across both sides | a shared wrong key definition makes both anti-joins empty |
| Idempotent merge | applying the same input twice yields the same state | merge is a pure function of (key, version, payload) with a total ordering | fails with non-idempotent side effects or a non-total order |
| Watermark soundness | freshness computed from an asserted complete cutoff bounds the age of the delivered population | the assertion is honest and the cutoff is per population | a lying or inferred watermark makes the bound meaningless |
| Bloom false-positive bound | `fpr = (1 - e^{-kn/m})^k` for uniform independent hashes | no deletions, independent hashes, known `n` magnitude | degrading with correlated hashes or saturation beyond design `n` |
| Normal-approximation sample size | `n0 = z^2 p(1-p)/e^2` | simple random sample, independent records, `np` and `n(1-p)` both > ~10 | invalid for rare events (`p` near 0) where exact or Poisson methods are needed |
| Finite-population correction | `n = n0/(1 + (n0-1)/N)` | sampling without replacement from a finite population | negligible when `n0 << N`; incorrect if the population is conceptually infinite |
| Error-budget arithmetic | breaches allowed in a period `= n_windows * (1 - SLO)` | windows are comparable in length and consequence | unequal windows (daily vs monthly) make the count non-comparable |

### 10.2 Why composition holds, and the one-line proof

If `V ⊆ R ⊆ E` as sets, then `|V|/|E| = (|V|/|R|) * (|R|/|E|)` by cancelling `|R|`. The proof is one line; the difficulty is entirely in establishing the nesting. Nesting fails in three practical ways: the dedup key differs between the `R` and `V` computations (so a record can be in `V` but not in `R`); the windows differ (a record counted at 23:59 in one and 00:00 in the other); or `V` is computed on a filtered subset while `E` covers the full population. Each of these produces a `u` that is arithmetically valid and semantically meaningless, which is why the derivation in 4.1 states the nesting as a validity condition rather than a formality.

### 10.3 Watermark soundness, stated carefully

The claim is: if the producer asserts completeness for population `P` through cutoff `w`, and the assertion is true, then at time `t` the consumer possesses every record of `P` with event time `≤ w`, and the age of the most recent such record set is bounded below by `t - w`. Note what is and is not claimed. It does not say records with event time `> w` are absent — some may have arrived. It does not say the population will ever be complete beyond `w`. It certainly does not say the values are correct. It says the consumer's knowledge of `P` is complete up to `w`, which is the only fact a freshness objective can be built on.

The soundness depends on the assertion being produced by a party that can know it. A pipeline that sets its watermark to `now - 5 minutes` by configuration has not asserted anything about completeness; it has asserted a hope. This is the most common way a freshness metric becomes fiction while looking precise.

### 10.4 Reconciliation as refutation only

Conservation gives a one-directional inference: if `sum(S) ≠ sum(C)` then the transform is not value-preserving, so something is wrong. The converse does not hold, and constructing counterexamples is easy — a fan-out on `parent_id` that duplicates `n` rows and an inner join on `child_id` that drops `n` rows can leave the total unchanged while the row set is entirely wrong. This asymmetry is why the theorem table lists conservation under "necessary condition, not sufficient", and why the checklist pairs it with bidirectional key anti-joins. A review that accepts "the totals match" as evidence of correctness is misusing the result.

### 10.5 Compatibility results from schema-evolution practice

The published evolution rules of columnar serialisation formats give a defensible basis for the lattice in 2.5. In the Avro specification, resolving a reader schema against a writer schema permits adding fields with defaults, permits removing fields that have defaults or are optional in the reader, and permits type promotion in a defined direction (int → long → float → double, with narrower promotions for string/bytes). It does not preserve semantics: a rename is resolvable only with an explicit alias, and a change of unit is invisible. The practical lesson for a contract is that format-level evolution rules cover the structural row of the lattice and nothing in the semantic column, which is why both a schema check and a value-level check are required.

Kafka-style schema registries add a compatibility mode (backward, forward, full) enforced on registration. Those modes answer "can a reader with schema A parse data written with schema B", which is the parse axis only. A contract that says "we use backward compatibility" has specified one of four planes: it says nothing about identity, semantics, or delivery, and its name invites the reader to over-trust it.

OpenLineage-style event specifications define a run/facet model for lineage metadata: a run has inputs, outputs, and a job, and facets carry extensible attributes. That model is the natural place to attach contract evaluations — the evaluated dataset version, the contract version, and the observed ratios — so that a later reviewer can reconstruct why a release was allowed. It is a transport and schema convention for metadata, not a quality guarantee, and it does not tell you what to assert.

Primary references to consult before asserting product-specific behaviour, all accessed 2026-11-09: the [Apache Avro specification](https://avro.apache.org/docs/) for schema resolution rules; the [OpenLineage specification](https://openlineage.io/docs/) for the run/facet metadata model; and the [scikit-learn common pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html) for training-only preprocessing, which is the analogous leakage boundary when a dataset feeds a model. These are mechanism sources for the claims above; none supplies a measurement for the illustrative numbers in this file.

### 10.6 Result-to-check mapping

| Result | Contract clause it licenses | Assertion to implement |
|---|---|---|
| Ratio composition | single release threshold on `u` with both components published | recompute `u` and `(v*c)`; assert agreement within 1e-9 |
| Conservation | value reconciliation at cutoff | `abs(S - C) <= tolerance` with sign recorded |
| Key-level completeness | referential closure between dataset and dimension | both anti-joins empty |
| Idempotent merge | replay safety clause | apply twice, compare state hashes |
| Watermark soundness | freshness objective | `t_now - w <= objective`, with `w` from producer |
| Bloom bound | duplicate-suppression error budget | `fpr(m/n, k) <= reserved_budget` |
| Sample size | semantic audit plan | `n` computed and recorded before the audit |

**Takeaway:** Use the composition result to justify a single coverage threshold, use conservation only to refute, and never let a format-level compatibility mode stand in for a semantic change review.

## 11. Interview / Exam Questions

Each answer states the reasoning, not just the conclusion, and each carries a red flag: the specific thing a weak answer would say that reveals a missing distinction.

**Q1. A producer reports 99.4% data quality. What do you need before you can use that number?**
Ask which ratio it is, over which population, with what denominator, computed on which window, with which duplicate policy. If it is `V/R` then a 0.6% validity loss can sit on top of a 4% delivery loss, and true usable coverage is 95.4%. Ask for the raw counts `E`, `R`, `V` rather than the ratio.
*Reasoning:* a ratio without its base is not auditable, and the three plausible bases give materially different answers on the same data.
*Red flag:* accepting the percentage and moving on — or asking only "is that good?" without asking "of what?".

**Q2. Given `E = 2,000`, `A = 1,990`, 10 duplicates, and `V = 1,940`, compute the release metrics.**
`R = 1,990 - 10 = 1,980`; `c = 1,980/2,000 = 0.99`; `v = 1,940/1,980 = 0.979798`; `u = 1,940/2,000 = 0.97`. Duplicate excess `d = 10/1,990 = 0.005025`. Check: `0.99 * 0.979798 = 0.97`.
*Reasoning:* dedup before ratio, and the release number is `V/E` not `V/R`, because the decision needs 97% of the population, not 97% of what arrived.
*Red flag:* computing `1,940/1,990` for validity, which mixes the pre-dedup denominator with the post-dedup numerator.

**Q3. Why is `now - max(event_time)` a bad freshness metric? Give a scenario.**
It measures the newest record, not the completeness of the population. Scenario: a stream carries 12 regional feeds; 11 stall and one continues delivering. The last-event lag stays at one minute while 90% of the population is 40 minutes stale. A watermark per region exposes it; a global maximum timestamp does not.
*Reasoning:* freshness is a property of a population boundary, and the maximum over records is a property of a single record.
*Red flag:* "it's simpler and usually fine" — it is usually fine and precisely misleading during incidents.

**Q4. A nullable column was added. Is the release safe?**
Not established. Nullable addition is additive only for consumers that select named columns and tolerate nulls in a way whose semantics they have defined. It is breaking for `SELECT *` into positional writers, for consumers that compute `sum(col)` (nulls propagate differently than zeros), and for consumers that treat null as "not applicable" versus "unknown". Enumerate consumers, check each one's null handling, and state the null semantics in the contract.
*Reasoning:* compatibility is relational to consumers, and null semantics are semantic, not structural.
*Red flag:* "it's nullable, so nothing can break" — the nullability itself is the change in meaning.

**Q5. Revenue jumped 18% after a release. How do you investigate, in order?**
First confirm the population and cutoff are identical (same channels, same date boundary operator). Second check grain and row multiplicity: `count(*)` and `count(distinct business_key)` before and after. Third check joins for fan-out by comparing a parent-level aggregate. Fourth check units and currency conversion. Fifth read the contract diff for semantic changes. Sixth, and only then, consider a real demand change — and validate it against an independent source such as payments or fulfilment volume.
*Reasoning:* a jump that coincides with a release has a prior probability heavily weighted toward the release; the cheapest checks that can eliminate whole classes come first.
*Red flag:* starting with "let's look at the daily trend" — a trend chart cannot distinguish fan-out from demand.

**Q6. What does idempotency actually guarantee, and what does it not?**
Applying the same input twice produces the same state. It does not choose a correct ordering among different inputs, does not detect two records that represent the same business event under different keys, does not roll back external side effects, and does not make a wrong mapping right. It is a property of the merge function with respect to a specified key and ordering rule.
*Reasoning:* idempotency is relative to a key and an equality relation, both of which are contract decisions.
*Red flag:* "we're idempotent so duplicates are handled" — that answers one of five duplicate risks.

**Q7. Your completeness score is 100% but the total is down 8%. Where do you look first?**
At the expectation. `c = 1.0` with a falling total means `E` moved with `R`, so the expectation is derived from the delivery rather than from the business. Compare the expectation's predicate to an independent control total's predicate and to a trailing baseline. The second place to look is the quarantine share: if failing records are excluded from both numerator and denominator of the delivered set, `c` can be 1.0 while 8% of the population sits in the side table.
*Reasoning:* a ratio that is structurally pinned near one carries no information, and quarantine that is not counted as invalid is a coverage loss.
*Red flag:* assuming the total is wrong — the more likely fault is the denominator.

**Q8. When should a consumer be allowed to override a violated contract?**
When the decision owner holds logged authority, the breach's materiality bound is below the decision's tolerance, a fallback value with an explicit as-of label is published, and the override expires. The override must record name, reason, bound, expiry, and the repair ticket. If any of those is missing, the correct answer is hold.
*Reasoning:* override exists because thresholds are approximations and business judgement is real; it must be an accountable act rather than a cultural habit.
*Red flag:* "the business needs it" without a bound — that is pressure, not an override.

**Q9. Estimate the state cost of an exact uniqueness check on 40 million keys held for one hour, and say when you would stop using exactness.**
At 32 bytes per key (16-byte key + 8-byte version + bookkeeping), `40e6 * 32 = 1.28e9` bytes = 1.28 GB decimal, 1.19 GiB binary. With a 0.75 load factor and hash-table overhead, plan for roughly 2.5 GB and verify it fits the executor, otherwise the check spills and misses the deadline. Replace exactness only when the reserved error budget permits it: a Bloom filter at 16 bits per key has a 0.0459% false-positive rate and 80 MB of state, which is acceptable if the release tolerance is 1 point; at 8 bits per key the 2.16% rate consumes more than the whole budget.
*Reasoning:* the state term is the one that fails silently through spilling, and a probabilistic structure's error consumes quality budget rather than being free.
*Red flag:* quoting memory without the load factor, or adopting a Bloom filter because it is "much smaller" without quantifying the loss.

**Q10. What changed constraint would reverse your gate design?**
If the delivery window shrinks below the check's runtime, the blocking gate must become a quarantine gate plus an asynchronous certification, because a check that finishes after the decision deadline is not a control. If the population doubles and the state no longer fits, exact uniqueness becomes a partitioned check with a reconciliation pass. If a regulator requires restatement, last-write-wins becomes immutable history plus an explicit as-of rule.
*Reasoning:* gates are sized against a deadline and a budget; changing either changes the feasible gate, and pretending otherwise produces a check that runs but cannot block.
*Red flag:* "the design is best practice" — best practice is a claim relative to constraints.

**Q11. How do you verify a migration from an old to a new dataset version?**
Dual-run both for at least one full business cycle, compare at three levels — row counts by key, value totals by cutoff, and a stratified sample of field values — and require every delta to be explained record by record. Keep the old path readable until every consumer has signed off and its own reconciliation passes. Then retire with a dated window.
*Reasoning:* a single-level comparison cannot distinguish compensating errors, and retiring early removes the reference that makes investigation possible.
*Red flag:* comparing only the headline total for one day.

**Q12. A contract with 99% coverage has breached twice this quarter. Is the contract working?**
Probably yes, and that is the point: a threshold is a control, not a promise. Two breaches in a quarter is consistent with an error budget of `n * (1 - SLO)` and tells you the distribution is near the threshold, which is where a control is supposed to sit. The failure mode to exclude is that the breaches were detected but not acted on — check the escalation log and the hold record, not the breach count.
*Reasoning:* the useful question about a control is whether breaches are detected and acted on, not whether they are rare.
*Red flag:* treating any breach as a defect in the threshold, which leads to threshold inflation whenever the check fires.

**Q13. What is the most common wrong method in data contract work?**
Treating the schema registry as the contract. It produces a false sense of control because every automated signal is green while grain, unit, population, and timeliness are unstated. The tell is that the contract document cannot answer "how many records should there be, and how do you know?".
*Reasoning:* the four validation planes are independent; a check in one plane provably cannot cover another.
*Red flag:* naming a tool as the answer rather than a property.

**Q14. How would you detect that a quarantine policy silently ate 6% of the population?**
Assert that quarantined records count as invalid for coverage — that is, `V` excludes them and `u` therefore falls. Then monitor the quarantine share by reason code with a gate, and reconcile monthly. The detector is not the quarantine count itself (growth may be expected) but the ratio of quarantine to population and its trend, plus the divergence between certified and unrestricted totals.
*Reasoning:* quarantine is only safe while it is visible in the denominator of the release decision.
*Red flag:* "quarantine is safe because nothing is deleted" — retention without counting is still a coverage loss.

**Q15. Design the smallest contract that would have prevented a metric that double-counted revisions. Does it need thresholds?**
No thresholds are needed. The minimum is a grain clause stating that one row is one version of one `(order_id, line_id)`, an explicit latest-selector rule with a total ordering, and a duplicate assertion on the declared key tuple. That is enough to prevent the double count, because the defect is definitional. Thresholds would only be needed to detect the *residual* duplicate rate after the selector was applied.
*Reasoning:* many contract defects are definitional, and a definition clause is cheap, exact, and testable — much cheaper than a statistical threshold.
*Red flag:* reaching for a quality threshold when the real gap is a missing definition.

**Takeaway:** Strong answers name the population, the denominator, the cutoff, the gate, and the evidence that would falsify the claim; the red flag is always a missing relation — of ratio to base, of compatibility to consumer, of freshness to watermark.

## 12. Quick Reference Card

| Clause | Rule | Evaluate when | Critical assumption | Avoid when | Gate |
|---|---|---|---|---|---|
| Grain | one row per declared key tuple per mutation rule, with an as-of rule | any metric, join, or export | the key tuple is the business identity | key is a surrogate or the mutation rule is unstated | blocking |
| Identity | max rows per key = 1, or the declared version count | before every aggregate | duplicate policy applied on the business key | dedup ran on a load id | blocking |
| Coverage | `u = V/E`, with `c` and `v` published alongside | release decision | `E` independently produced, same population and window | `E` derived from `R` or from the delivered file set | blocking |
| Validity | `v = V/R` with an enumerated, versioned rule list | every delivery | rules are two-sided and first-failure attribution is defined | a one-sided null-rate rule | quarantine |
| Freshness | `t_now - watermark <= objective`, timed from the partition close | any time-sensitive read | watermark is a producer assertion, per population | computed from `max(event_time)` | blocking for intraday |
| Duplicates | idempotent merge on business key with a total version order | replay, backfill, retry | version ordering is total and monotone in business time | last-write-wins on ingestion time | quarantine the tie |
| Reconciliation | signed `S - C` plus bidirectional key anti-joins | filed or material metrics | same population, cutoff, unit, rounding | totals-only comparison | blocking above tolerance |
| Change class | additive / semantic / structural, per consumer | every producer release | consumer inventory is current | "the DDL succeeded" | dual-publish for semantic |
| Override | named authority, logged reason, materiality bound, expiry | a breached gate | the bound is computed, not asserted | "the business needs it" | logged exception |

**Takeaway:** Use this card as the release checklist: state the grain, prove identity, publish `u` with `c` and `v`, take freshness from the watermark, reconcile value and keys, classify the change per consumer, and let only a named, bounded, logged override open a breached gate.
