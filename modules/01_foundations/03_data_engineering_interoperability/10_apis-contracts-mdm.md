# APIs, Data Contracts, and MDM

> Module 03 — Data Engineering and Interoperability
> Topic 10 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 02.02, 03.03

---

## 1. Why this topic exists

A supply chain plan is an assembly of other people's numbers. It consumes prices, lead times,
capacities, inventory positions, open orders, customer addresses, and calendars. Each number
arrives through an interface that someone else owns. When one of those numbers changes meaning
without changing shape, the plan does not fail loudly. It reprices.

That is the specific failure this topic exists to prevent. The failure is not a corrupted file or
a failed job. Every request returns a success status, HTTP (Hypertext Transfer Protocol) 200. Every field
has the expected type. Every row passes validation. The price is simply in a different unit, or
the quantity now excludes allocated stock, or the customer identifier now points at a different
legal entity. The plan consumes the new meaning under the old labels and produces a decision that
is arithmetically correct and commercially wrong.

Three properties of modern integration make this failure more likely than it used to be.

**The interface replaced the file, but the governance did not follow.** A nightly file drop had a
filename, a row count, and a person who could be asked what changed. A REST (representational
state transfer) endpoint has a URL (uniform resource locator), a status code, and a team that may
have shipped a new version without telling anyone. Version numbers describe the endpoint; they do
not describe the population, the units, or the identity model behind it.

**An application programming interface (API) answers a query, not necessarily a population.** A file was usually a full snapshot. An
API returns the pages you ask for, subject to page-size caps, rate limits, cursor expiry, and
permission filters. "I received a 200" and "I received every record the contract covers" are
different claims, and only the second one supports a planning decision. Pagination and rate
limits are therefore not implementation details. They are completeness hazards with a cost.

**Identity has moved downstream.** When systems were joined inside one enterprise resource
planning (ERP) system, the item key was the item. When data arrives from supplier catalogues,
third-party logistics (3PL) portals, marketplaces, and customer master data, the join is between
records that describe the same entity without sharing a key. Entity resolution decides what the
plan is even about. A false merge combines two customers and misallocates a rebate. A false split
splits one item's demand history and makes both halves look new.

The three subjects of this topic answer these three problems:

| Subject | Question it answers | Failure it prevents |
|---|---|---|
| API as interface | What can this interface actually deliver, and what does it silently omit? | Partial data treated as complete |
| Data contract | What has the producer promised, in a version that can be tested? | Silent semantic change |
| Master data management (MDM) | Which real-world entity does this record denote, and which source is authoritative for each field? | Ambiguous nouns in every downstream sentence |

They belong together because they fail together. A contract without entity resolution produces a
well-versioned dataset about the wrong entities. Entity resolution without a contract produces a
correct entity model fed by a changing population. An API treated as a feed defeats both.

**Why this topic sits here.** Topic 02.02 established what master data is and how keys split,
merge, and dangle. Topic 03.03 established how data is ingested, including schema drift at
arrival and the event-schema compatibility modes. This topic does not repeat either. It adds the
governance layer that those mechanics assume: the interface boundary, the promise attached to it,
and the process that keeps entity identity stable across changes. Topic 11 then makes the result
reproducible through quality tests, quarantine, versioning, and lineage.

The order matters. A team that writes tests before it writes a contract tests the wrong things
confidently. A team that resolves entities before it defines the population resolves the wrong
entities carefully. The contract comes first because it defines what "correct" will mean.

Three examples recur through the sections that follow. They are ordinary, not exotic.

1. **The unit change.** A supplier adds a `sell_unit` field and changes the price from per case to
   per each for part of its catalogue. The schema is backward compatible: no field was removed and
   no type changed. The plan reads the old field name and multiplies by a case quantity. Cost of
   goods falls by a factor of 24 on the affected items.
2. **The page that never arrived.** A nightly extraction paginates over a price table while the
   supplier refreshes it. Records shift between pages. The job reports success, the row count is
   plausible, and several hundred prices are one day stale.
3. **The customer that became two.** A regional distributor is acquired and re-registered under a
   new legal name. The customer master creates a new record because the name changed. Rebates,
   credit limits, and service history split across two identifiers, and the sales team negotiates
   against the smaller half.

None is a statistical failure; each is a boundary failure, preventable with a document and a review rather than a better model.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** an API from a data feed in terms of who defines the population, the query
   shape, and the delivery guarantee, and **classify** a described integration into one of the
   two.
2. **Audit** a paginated interface for completeness, naming the pagination style, the consistency
   boundary, the rate-limit budget, and the reconciliation that would prove a full read.
3. **Write** a four-clause data contract — version, schema, service, owner — with units, null
   semantics, key uniqueness, freshness, completeness, and a change protocol stated explicitly.
4. **Classify** a proposed interface change under semantic versioning and **decide** whether it is
   a major, minor, or patch change for a named consumer, including cases where the schema is
   compatible but the meaning is not.
5. **Design** a deprecation window with consumer registration, notice, overlap, and removal
   criteria, and **state** what makes the window enforceable rather than advisory.
6. **Resolve** item, location, and customer records using deterministic rules first, blocking,
   probabilistic scoring, a review band, and a canonical identifier with a dated crosswalk.
7. **Compute** precision, recall, and cost-weighted error at several match thresholds on a labelled
   pair set, and **choose** a threshold from the relative cost of a false merge and a false split.
8. **Assign** a source of record per field across an ERP, a product information management (PIM)
   system, a warehouse management system (WMS), and a supplier interface, and **justify** each
   assignment by the consequence of getting it wrong.
9. **Run** a contract review that converts a proposed upstream change into a tested consumer
   impact statement before the change reaches a planning cutoff.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a producer on the left that owns a population, a consumer on the right that
owns a decision, and a contract in the middle that is thin enough to read and specific enough to
test.

```text
   PRODUCER SIDE                    CONTRACT BOUNDARY                   CONSUMER SIDE
   -------------                    -----------------                   -------------

   +----------------+       +-----------------------------------+       +----------------+
   | ERP / PIM      |       |  DATA CONTRACT   version 2.4.0     |       | ingestion      |
   | supplier API   | --->  |                                   | --->  | conformance    |
   | WMS / 3PL      |       |  [1] VERSION                      |       | entity         |
   | marketplace    |       |      semver + compatibility +     |       | resolution     |
   |                |       |      deprecation window           |       | mart / plan    |
   | owns the       |       |  [2] SCHEMA                       |       |                |
   | population     |       |      fields, types, units, keys,  |       | owns the       |
   |                |       |      null meaning, enum policy    |       | decision       |
   |                |       |  [3] SERVICE                      |       |                |
   |                |       |      freshness + completeness +   |       |                |
   |                |       |      availability + retention     |       |                |
   |                |       |  [4] OWNER                        |       |                |
   |                |       |      named role + escalation      |       |                |
   +----------------+       +-----------------------------------+       +----------------+
                                          |
                                          | contract tests execute here
                                          v
                             +-----------------------------+
                             | schema | unit | key       |
                             | count  | freshness | range|
                             +-----------------------------+
```

The four clauses are deliberately few. A contract with forty clauses is a policy document that
nobody runs. Four clauses map to four independent failure surfaces, and each can be tested
automatically.

| Clause | The question it answers | What fails without it |
|---|---|---|
| Version | Which promise am I consuming, and when will it change? | A silent change arrives under a stable URL |
| Schema | What does each field mean, in what unit, and what is a valid value? | Types agree while semantics diverge |
| Service | How fresh and how complete is the data, by when? | Partial data is treated as a full snapshot |
| Owner | Who is accountable when the promise is broken? | Nobody can approve, explain, or fix a change |

Seven properties of the picture explain most of what follows.

**Property 1 — The producer owns the population; the consumer owns the decision.** The producer
decides which records exist and what they mean. The consumer decides what to do with them. Most
integration disputes are one party making a decision that belongs to the other: a producer
retiring a field because its own reports no longer use it, or a consumer assuming a filter
returns the whole population. The contract is where that boundary is written down.

**Property 2 — An API is a view, not a copy.** It has a query shape (what you can ask), a
permission shape (what you are allowed to see), and a temporal shape (which moment the answer
describes). A file had the same three properties, but they were implicit in how it was produced.
In an API they are explicit in the parameters and easy to get wrong. Asking for `status=active`
and calling the result "the catalogue" is a decision about the population, made by the consumer,
usually without noticing.

**Property 3 — Completeness is a claim about the source population, not about the bytes
received.** A response can be well-formed, fully paginated, and still incomplete because a
record was inserted between pages, because a permission filter removed a region, or because the
source itself had not yet loaded the day's changes. Completeness needs a denominator that the
producer states and the consumer reconciles against. Without a denominator, "row count looks
normal" is the only available test, and it is not a completeness test.

**Property 4 — Identity is a decision under uncertainty, and the two errors are not symmetric.**
A false merge says two entities are one. A false split says one entity is two. A false split is
usually visible, bounded, and repairable: two records can be joined later if history is kept. A
false merge is often invisible, contaminates history, and may be irreversible once transactions
from two entities are commingled. The asymmetry, not a generic accuracy target, sets the
threshold.

**Property 5 — Semantic versioning protects the interface, not the meaning.** The specification
governs the public API (application programming interface). If the unit, the population, or the
null semantics are not declared part of the public API, a change to them can be shipped as a
minor release and still break every consumer. The contract's job is to pull those semantics into
the versioned surface.

**Property 6 — MDM is a process with a source of record per field, not a product with a golden
record per system.** The question is never "which system wins". It is "which system is
authoritative for this field, for this period, and who decided". A program that declares one
system the golden record will overwrite standard costs with marketing copy within a quarter.

**Property 7 — Three clocks govern whether data is usable, and they are not the same clock.**
Event time is when the thing happened. Availability time is when the consumer can see it.
Decision time is the planning cutoff. Freshness measures the first gap; consumer latency
measures the second. A contract that promises freshness but not a decision cutoff can be
satisfied while the plan is still late.

| Clock | Definition | Typical unit | Failure if conflated |
|---|---|---|---|
| Event time | When the physical or commercial event occurred | Timestamp with offset | Late-arriving facts overwrite newer ones |
| Availability time | When the record is queryable by the consumer | Timestamp | Freshness is measured from the wrong origin |
| Decision time | The cutoff after which a plan cannot change | Local time, stated | Data arrives on time but after the decision |

> **Note:** the most consequential sentence in this topic is that a schema-compatible change can
> be a semantic breaking change. Compatibility modes answer "can the old parser read the new
> bytes". They do not answer "does the old code still compute the right number". Section 5.2
> gives the classification table; section 6 shows what happens when a team applies only the first
> test.

## 4. Definitions and vocabulary

The vocabulary below is ordered from interface, to contract, to resolution, to governance. Terms
that Topic 02.02 already defined — master data, golden record, survivorship, entity resolution as
a general idea — are repeated only where this topic uses them in a narrower sense.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Application programming interface (API) | A defined boundary through which one system requests data or actions from another | Used to mean any integration, including a file drop | Interface version |
| REST | An architectural style using resources, HTTP methods, and representations | Used to mean any JSON (JavaScript Object Notation) endpoint | HTTP method plus media type |
| GraphQL | A query language in which the client specifies the response shape | Treated as a drop-in replacement for REST | Query document plus schema |
| Event API | A push of state changes to a subscriber, often by webhook or stream | Treated as guaranteed delivery | Topic plus event identifier |
| Webhook | A producer-initiated HTTP callback to a consumer-owned URL | Treated as reliable because it is immediate | Endpoint URL |
| OpenAPI | A machine-readable description of an HTTP interface's operations and payloads | Treated as a data contract | Specification version, for example 3.2.0 |
| Schema | The named fields, types, and structural constraints of a payload | Treated as the meaning of the payload | Schema version or registry identifier |
| Data contract | A producer's versioned, testable promise about schema, service, and ownership | Used to mean a PDF (Portable Document Format) file, a service-level agreement, or a handshake | Contract version |
| Open Data Contract Standard (ODCS) | A YAML (human-readable data serialisation) structure for expressing data contracts, maintained by the Bitol project | Treated as a legal or de jure standard | Version 3.2.0 at the time of writing |
| Service-level agreement (SLA) | A contractual commitment with a remedy attached | Confused with an internal target | Contract clause |
| Service-level objective (SLO) | A measurable target for one service property, such as freshness | Confused with an SLA | Percentile over a window |
| Freshness | Delay from event time to the data being available to the consumer | Confused with refresh frequency | Minutes or hours, usually p95 |
| Completeness | Fraction of the contract's defined population present and current | Confused with a successful job status | Percent of population |
| Population | The set of records the interface is claimed to represent | Left undefined, so any row count seems plausible | Count plus definition |
| Watermark | The point in event time up to which the consumer believes it has complete data | Confused with the maximum timestamp received | Event timestamp |
| Pagination | Splitting a result set into separately requested pages | Treated as an implementation detail | Page number or cursor |
| Offset pagination | Selecting records by numeric position in an ordered list | Assumed stable while the source is being written | Integer offset and limit |
| Cursor pagination | Selecting records after an opaque position token | Assumed to imply a consistent snapshot | Opaque token |
| Rate limit | A cap on requests per unit of time | Confused with a total quota | Requests per minute |
| HTTP 429 | The status code for too many requests | Treated as a permanent failure | Three-digit status code |
| Retry-After | A response header stating how long to wait before retrying | Ignored because the client has its own backoff | Seconds or HTTP date |
| Link header | An HTTP header carrying typed links, including pagination relations | Assumed to be present on every API | Relation type such as next |
| Idempotency key | A client-supplied key that makes a repeated write safe | Used only for reads | Universally unique identifier |
| Breaking change | A change that invalidates a consumer's assumptions | Narrowed to "a field was removed" | Compatibility class |
| Backward compatible | A new producer version can be read by consumers built for the previous version | Direction reversed with forward compatible | Compatibility mode |
| Semantic versioning | A MAJOR.MINOR.PATCH numbering convention tied to compatibility | Treated as marketing numbering | Three integers |
| Major version | An incompatible change to the public interface | Bumped for any large release | Integer, from 0 upward |
| Minor version | Backward-compatible functionality added to the public interface | Used to hide a semantic change | Integer |
| Patch version | A backward-compatible correction | Used for data corrections that change values | Integer |
| Deprecation window | The period between announcing a removal and completing it | Confused with the support period after removal | Months |
| Sunset date | The date after which an interface or field is withdrawn | Treated as advisory | Calendar date |
| Consumer registry | The list of known consumers of an interface, with contacts and versions | Assumed complete because it exists | Count of registered consumers |
| Contract test | An automated check that a producer's output satisfies a contract clause | Replaced by a one-off validation query | Test identifier |
| Entity resolution | The process of deciding whether records denote the same real-world entity | Used to mean exact matching | Decision per pair or cluster |
| Record linkage | Joining records from different sources that describe the same entity | Used only for person records | Pair classification |
| Deterministic match | A match declared when exact strong identifiers agree | Assumed sufficient on dirty data | Rule identifier |
| Probabilistic match | A match declared from a score or likelihood ratio | Reported as if it were a probability | Score or log-odds |
| Blocking | Partitioning records into candidate groups so not every pair is compared | Omitted, producing quadratic cost | Key function |
| Blocking key | The field or tuple used to partition records | Chosen without analysing its false negatives | Value tuple |
| Candidate pair | Two records selected for comparison | Counted as a match | Pair count |
| Match threshold | The score above which a pair is linked | Set by intuition rather than a labelled sample | Score on a stated scale |
| Precision | True positives divided by all predicted positives | Confused with accuracy | Proportion |
| Recall | True positives divided by all true positives | Confused with precision | Proportion |
| False merge | Two distinct entities treated as one | Called a duplicate | Error count and cost |
| False split | One entity treated as two | Ignored in match metrics | Error count and cost |
| Transitive closure | Grouping linked pairs into connected components | Assumed to preserve entity boundaries | Cluster identifier |
| Canonical identifier | A surrogate identifier assigned to a resolved entity | Reused after retirement | Internal identifier |
| Crosswalk | A table mapping identifiers across systems and across time | Assumed static and complete | From-key, to-key, validity dates |
| Match review queue | Pairs in the uncertain score band awaiting a human decision | Omitted, so every decision is automated | Queue depth and age |
| Source of record | The system authoritative for a named field | Assigned once per system rather than per field | Field-level designation |
| Survivorship | The rule that chooses a value when sources disagree | Implemented as last write wins | Rule per attribute |
| Master data management (MDM) | The process and tooling for creating, governing, resolving, and distributing master data | Used to mean a software purchase | Process maturity level |

Four of these need expansion, because the rest of the topic depends on the distinction.

**API versus feed.** A feed is a projection that the producer defines and the consumer receives
whole. An API is a query surface that the consumer calls and the producer limits. The practical
difference is who owns the definition of "all". In a feed, the producer decides what the snapshot
contains. In an API, the consumer writes a query and then mistakes the result for a snapshot.
This is why API integrations fail in a characteristic way: not with an error, but with a
plausible subset.

**Schema versus semantics.** The schema says `unit_price` is a decimal with four places. The
semantics say it is a currency amount per sell unit, that the sell unit is stated in a sibling
field, and that a missing sell unit means the historical case convention applies. A schema
validator can pass every record while the semantics are wrong. This is the single most important
distinction in the topic, and section 5.2 turns it into a change-classification table.

**False merge versus false split.** These are not symmetric errors with a shared accuracy metric.
A false merge commingles histories and is often irreversible without reconstructing from source
records that may no longer exist. A false split leaves two records that can be joined later if
the raw history is retained. The correct threshold is therefore a function of the cost ratio,
not of a generic F1 score. Section 13 works the arithmetic.

**Source of record versus system of entry.** A system of entry is where a user types a value. A
source of record is where the value is authoritative after all corrections. They are often
different: a sales representative enters a customer address in a customer relationship
management (CRM) system, but the finance system's address is the one used on invoices because it
has been validated against tax records. Naming the system of entry as the source of record is a
common and quiet error.

## 5. The mechanics, step by step

This section follows one interface from first contact to operating control. Each step names the
artifact it produces, the branch that decides the design, and the failure if it is skipped.

### Phase A — Read the interface as an interface

**Step 1. Classify the interface shape.**

- Artifact: an interface record naming the transport, the interaction pattern, and the owner.
- Four shapes behave differently, and the difference determines where completeness is lost:

| Shape | Consumer asks for | Producer controls | Characteristic completeness hazard |
|---|---|---|---|
| Request/response REST | A resource matching filters | Page size, rate, allowed filters | The population cannot be expressed as a query |
| GraphQL | A selected field graph | Depth and cost limits, resolver behaviour | A 200 response contains partial data plus an errors array |
| Event stream | A subscription to changes | Retention, partitioning, ordering | Consumer lag exceeds retention, so early events expire |
| Bulk transfer | A file or export | Snapshot boundary and extract window | The extract runs while the source is being written |
| Webhook | A callback on change | Retry policy and delivery guarantee | A missed callback has no replay path |

- Key branch: **pull or push?** Pull gives the consumer control of timing and the ability to
  reconcile. Push gives lower latency and no natural reconciliation point.
- Failure if skipped: the team builds pull-style reconciliation for a push interface, or
  assumes a webhook is a complete change log.

**Step 2. Enumerate resources, keys, and grains.**

- Artifact: a list of resources, the identifier of each, and the grain of one row.
- Key branch: **is the identifier a business key or a surrogate?** A surrogate is stable and
  opaque. A business key is meaningful and can be reissued. The interface usually exposes both,
  and the consumer usually joins on the wrong one.
- Failure if skipped: one row is assumed to mean one item, when it means one item per price list
  per validity period.

**Step 3. Define the population and its denominator.**

- Artifact: a written population statement and a count the producer will report.
- The statement has four parts: which entities are in scope, which are excluded, as of which
  moment, and under whose permissions.
- Key branch: **is the denominator a count of records or of entities?** A price interface may
  return 48,000 price records for 6,400 items. Completeness against 48,000 is not the same claim
  as completeness against 6,400.
- Failure if skipped: "the row count looks normal" becomes the only available test.

**Step 4. Measure the delivery envelope.**

- Artifact: a table of page size, page count, rate limit, retry policy, window, and retention.
- Feasibility is arithmetic. A full read requires `N_requests` at an effective rate of
  `min(rate_limit, allocated_rate)`. If that exceeds the available window in minutes, the
  extraction is infeasible without a bulk endpoint, a cache, or a negotiated higher rate.
- Key branch: **is the rate limit per client, per tenant, or per endpoint?** A limit shared with
  three other internal consumers is one third of the advertised number in practice.
- Failure if skipped: the pipeline is designed against a rate the team does not actually get.

### Phase B — Write the contract

**Step 5. Name the four clauses and put the contract under version control.**

- Artifact: a contract file in a repository, with a version, a status, and a change history.
- Key branch: **is the contract machine-readable?** A YAML or JSON (JavaScript Object Notation) document can be validated in
  continuous integration. A wiki page cannot. The Open Data Contract Standard is one structure;
  a team's own schema is acceptable if it is enforced.
- Failure if skipped: the contract becomes a document that describes an interface that no longer
  exists.

**Step 6. Bind the schema and the semantics.**

- Artifact: a field table with name, type, unit, null meaning, allowed values, and key role.
- The fields that cause the most damage are not the complex ones. They are unit, currency,
  timezone, precision, null meaning, enum closure, and key uniqueness.
- Key branch: **is a missing value unknown, not applicable, or zero?** These are three different
  facts and one null. A contract that does not distinguish them guarantees that every consumer
  will guess differently.
- Failure if skipped: the schema is valid and the arithmetic is wrong.

**Step 7. Bind the service levels.**

- Artifact: freshness, completeness, availability, and retention, each with a percentile and a
  measurement window.
- Key branch: **freshness from event time or from source commit?** A supplier that commits prices
  at 02:00 and exposes them at 02:05 has five minutes of interface latency and however many hours
  of business latency preceded it. State which clock the promise uses.
- Failure if skipped: the producer meets its promise and the plan is still late.

**Step 8. Bind the version and the change protocol.**

- Artifact: a compatibility statement, a deprecation window, a notice channel, and a consumer
  registry.
- Key branch: **what counts as the public API?** If units and population are declared part of the
  public API, a unit change is a major change. If they are not, it is a minor change with a
  breaking effect. The contract decides which.
- Failure if skipped: semantic changes ship as minor releases and no consumer is notified,
  because no consumer is registered.

**Step 9. Bind ownership and escalation.**

- Artifact: a named role accountable for the contract, a named steward who operates it, and an
  escalation path with response times.
- Key branch: **does the owner have the authority to refuse a change?** An owner without veto is a
  contact address. The review in Step 15 is where the veto is exercised.
- Failure if skipped: when the promise breaks, the consumer's only recourse is a support ticket
  to a team that does not know the consumer exists.

### Phase C — Resolve entities

**Step 10. Define the entity grain and the decision it supports.**

- Artifact: an entity definition per domain, with the grain stated and the decision named.
- Item grain is the trade item at a packaging level. Location grain is a GLN (Global Location Number) plus a role.
  Customer grain is a legal entity for credit and a ship-to for delivery, and those are different
  resolutions of the same commercial relationship.
- Key branch: **which error is worse here?** For a customer used in credit and rebates, a false
  merge is worse. For a marketing segment, a false split may be cheaper than an expensive review.
- Failure if skipped: the team resolves to the wrong grain and every downstream aggregate inherits
  the error.

**Step 11. Apply deterministic rules first.**

- Artifact: an ordered rule list with a coverage count for each rule.
- Strong identifiers: GTIN (Global Trade Item Number) at a stated packaging level for items, GLN
  (Global Location Number) for locations, a legal entity identifier or tax identifier plus country
  for parties. Each rule should record which
  records it resolved and which it could not.
- Key branch: **does the identifier need a qualifier?** A GTIN without a packaging level collides
  case and each. A GLN without a role collides sold-to and ship-to.
- Failure if skipped: exact matching on a dirty key produces false splits at scale, and the team
  concludes that matching is impossible.

**Step 12. Block, score, threshold, and review.**

- Artifact: a candidate-pair table, a score, a threshold policy, and a review queue.
- Blocking reduces the comparison space. It also creates false negatives by construction: a true
  duplicate whose blocking key differs will never be compared. Choose block keys that survive the
  errors you expect, and run a second pass with a different key if recall matters.
- Key branch: **is there a review band?** Auto-link above a high threshold, auto-reject below a
  low one, and send the middle to a person. A single threshold automates every uncertain decision.
- Failure if skipped: the threshold is set by intuition, and its precision is discovered in
  production.

**Step 13. Cluster, assign a canonical identifier, and publish the crosswalk.**

- Artifact: a cluster table, a canonical identifier per cluster, and a dated crosswalk.
- Pairwise matches do not automatically form valid clusters. If A matches B and B matches C, a
  transitive closure puts A and C together even when their direct score is low. Review the
  chained edges, not just the pairs.
- Key branch: **is the canonical identifier ever reused?** It must not be. Reuse converts a
  retired entity into a live one and silently reassigns history.
- Failure if skipped: history attaches to the wrong entity, and the error is indistinguishable
  from a real change.

### Phase D — Operate the boundary

**Step 14. Register consumers and run contract tests.**

- Artifact: a consumer registry and a test suite that runs on every producer release and every
  consumer deployment.
- Key branch: **does the producer run the tests, or the consumer?** Both. The producer tests that
  the contract is met. The consumer tests that its assumptions still hold against the published
  contract.
- Failure if skipped: the contract is a statement of intent with no runtime evidence.

**Step 15. Run the change review before release.**

- Artifact: a signed impact statement listing the diff, the affected consumers, the test results,
  the migration path, and the rollback.
- Key branch: **is the review triggered by a version bump or by a diff?** A semantic change can be
  shipped without a version bump if nobody inspects the diff. Trigger on the diff.
- Failure if skipped: the change reaches production and the first evidence is a number in a plan
  that nobody can explain.

**Step 16. Reconcile to the plan and monitor drift.**

- Artifact: a mapping from contract fields to plan parameters, and a control chart per field.
- Key branch: **which plan parameter consumes this field?** A price feeds standard cost, margin,
  order quantity, and supplier scorecards. A lead time feeds safety stock and promise dates.
  Naming the parameter turns a data defect into a decision impact.
- Failure if skipped: the data team fixes the pipeline and nobody knows whether the plan changed.

### 5.1 The contract test suite

A contract that is not tested is a description. The following tests are cheap, and each maps to a
clause.

| Test | Clause | What it catches | What it cannot catch |
|---|---|---|---|
| Schema and type validation | Schema | Renames, retypes, missing required fields | A correct type with a new meaning |
| Unit and currency assertion | Schema | A unit change, a currency change | A wrong value in the right unit |
| Key uniqueness | Schema | Duplicate records, grain changes | A duplicated entity under two keys |
| Null-rate and null-meaning check | Schema | A field that silently became optional | A null whose meaning changed while the rate held |
| Accepted-value and enum check | Schema | New enum values, code list drift | A new value that is semantically incompatible |
| Freshness percentile | Service | Stale data, stalled feeds | Data that is fresh but incomplete |
| Completeness reconciliation | Service | Missing pages, filtered records, missed deletes | Records that exist in both but describe different entities |
| Distribution drift | Schema and service | Unit changes, population shifts, mix changes | Small changes below the drift threshold |
| Referential integrity | Schema | Orphan keys after a master data change | A valid key pointing at the wrong entity |

The last row is where contract testing ends and entity resolution begins. A foreign key can be
valid and still denote the wrong customer.

### 5.2 Change classification

Semantic versioning gives the grammar. The table gives the judgement that the grammar does not.

| Change | Schema compatible? | Semantic class for a typical consumer | Contract action |
|---|---|---|---|
| Add an optional field | Yes | Minor, unless consumers reject unknown fields | Minor release, announce |
| Remove a field | No | Major | Major release, deprecation window |
| Rename with an alias period | Yes during overlap | Minor, then major at removal | Minor with deprecation date |
| Widen a decimal or integer | Usually | Minor | Minor release, test rounding |
| Narrow a type or precision | No | Major | Major release |
| Add an enum value | Yes | Minor syntactically; may be major for exhaustive consumers | Minor release, consumer test |
| Change a unit | Yes | Major | Major release, conversion guidance |
| Change null from unknown to not applicable | Yes | Major | Major release, backfill rule |
| Change a timestamp from local time to Coordinated Universal Time (UTC) | Yes | Major | Major release, dual-write period |
| Change the default filter or population | Yes | Major | Major release, count reconciliation |
| Change key uniqueness or deduplication | Yes | Major | Major release, crosswalk |
| Change refresh cadence | Yes | Service change | Minor release with notice |

The middle column is the trap. Six of the twelve changes are schema-compatible and semantically
breaking. A team that classifies changes by running a schema validator will ship all six as minor
releases.

> **Note:** semantic versioning is a convention for the public API, not a law. If a producer
> declares that units and population are outside the public API, the producer is technically
> correct to ship a unit change as a minor release. The contract exists to prevent that
> declaration from being made silently. The negotiation is about what belongs in the public API,
> and it belongs in the contract, not in the changelog.

### 5.3 The change review agenda

The review is short if the contract is good and long if it is not. Eight questions, in order.

| # | Question | Evidence required | If the answer is missing |
|---:|---|---|---|
| 1 | What exactly changes, field by field? | A diff of the contract and a sample payload before and after | Postpone the release |
| 2 | Which population does the change affect? | Counts by segment, not a percentage alone | Postpone |
| 3 | Does any unit, currency, timezone, or null meaning change? | Explicit statement per field | Treat as major |
| 4 | Which registered consumers read the affected fields? | Registry query with owner names | Block until the registry is checked |
| 5 | What did the contract tests return? | Test report against the new version | Block |
| 6 | What happens to history and open transactions? | Restatement or conversion plan with control totals | Block |
| 7 | How is the change rolled back? | Rehearsed rollback with a time estimate | Accept the risk in writing |
| 8 | Who signs for the producer and the consumer? | Named roles, not teams | No release |

Question 3 is the one that catches the expensive failures. A release that changes no schema but
changes a unit is not a technical change. It is a repricing event, and it needs the same approval
as a price change.

## 6. Worked example

Calder Foods is a chilled-ingredient distributor with four distribution centres (DCs), 6,400
active packaging components, and an annual packaging spend of EUR 41.0 million. Its nightly plan
runs at 02:00 and must be complete by 05:30. One supplier, Norvik Packaging, provides prices and
catalogue data through a REST interface. All figures are illustrative and internally consistent.

### 6.1 The interface as it was consumed

| Property | Value | Why it matters |
|---|---|---|
| Endpoint | `GET /v2/price-records` | The version is in the path, not in the payload |
| Page size | 500 records, fixed by the producer | 6,400 records require 13 pages |
| Pagination | Numeric offset and limit | The page window is not stable while the source is written |
| Rate limit | 120 requests per minute per client | Shared with two other internal consumers in practice |
| Retry signal | HTTP 429 with `Retry-After` | The client ignores it and uses a fixed 1-second backoff |
| Freshness | Not stated | The consumer assumes overnight |
| Completeness | Not stated | The consumer assumes a snapshot |
| Owner | Not stated | The consumer has a support queue |

The original contract had a schema clause and nothing else. It described the fields correctly
and said nothing about units, population, service, or ownership. That is the normal starting
point, not an unusual one.

### 6.2 A normal night

Thirteen page requests plus one metadata request return 6,400 price records in about four
minutes. The loader writes them to a staging table, deduplicates on `price_record_id`, and
publishes to the planning mart. The job status is success, the row count is 6,400, and the plan
reads the table at 02:20.

This is the state that makes the later failure possible: the pipeline has no denominator check,
because in the normal case the row count never changes. A control that never fails is a coincidence
with a log line.

### 6.3 The unit change

On a Tuesday, Norvik releases interface version 2.4.0. The changelog says two things: price
precision increases from two decimal places to four, and a new field `sell_unit` is added with
values `CASE` or `EACH`.

Under semantic versioning this is a minor release: functionality is added and nothing is removed.
Under the contract as written, it is also compliant, because the contract never said that the
unit was part of the interface.

The consumer's loader maps `unit_price` to `cost_per_case` and ignores unknown fields. That
ignore-unknown-fields behaviour is usually correct, and it is exactly what makes the failure
silent.

| Quantity | Before 2.4.0 | After 2.4.0, affected records |
|---|---:|---:|
| Components with `sell_unit = EACH` | 0 | 1,180 |
| Case pack | 24 eaches | 24 eaches |
| Price in source | EUR 36.00 per case | EUR 1.50 per each |
| Value stored by the loader | EUR 36.00 per case | EUR 1.50 per case |
| Correct value per case | EUR 36.00 | EUR 1.50 x 24 = EUR 36.00 |
| Error per case | 0 | 24 times understated |

The source is right. The loader is right about the field it read. The meaning changed in the
space between them.

### 6.4 What the plan did with it

The affected components carry EUR 7.56 million of annual spend:

```text
affected spend = 41,000,000 x (1,180 / 6,400) = 7,559,375
reported spend = 7,559,375 / 24             =   314,974
understatement = 7,559,375 - 314,974        = 7,244,401
```

The EUR 7.24 million is a measurement error, not cash that left the company. The cash
consequences come from what the plan did next.

| Consequence | Arithmetic | Annual or one-off effect |
|---|---|---:|
| Product-line margin restated | Revenue EUR 60m, true cost of goods sold (COGS) EUR 48m (20.0%); reported COGS EUR 40.76m (32.1%) | 12.1 points overstated |
| Capacity reallocated to the line | 6% of weekly capacity moved from a line with 24% margin | EUR 144,000 per year |
| Forward buy on apparent saving | 10 weeks of affected spend = 10/52 x EUR 7.56m | EUR 1.45m cash early |
| Carrying cost on the forward buy | EUR 1.45m x 18% x 5/52 average weeks held | EUR 25,200 |
| Supplier scorecard | Reported price deflation of about 96% on affected items | A 20% negotiation target is set |

The capacity arithmetic is the one to read twice. The line's true contribution margin is 20%.
The displaced line's is 24%. Moving EUR 3.6 million of annual revenue (6% of EUR 60 million) from
the second to the first loses four points on that revenue, or EUR 144,000 a year, even though the
first line now looks like the better business.

The scorecard effect is harder to price and longer-lived. A procurement team that believes its
packaging costs fell by 96% will set a should-cost target that cannot be met, reject a fair
quotation as uncompetitive, and spend a quarter discovering that the baseline was wrong.

### 6.5 The pagination gap

In the same week, the price table is refreshed at midnight while the extraction is running. The
interface uses offset pagination, so an insert or delete before the current offset shifts every
later record by one position.

| Event | Count | Effect |
|---|---:|---|
| Population at the start of the run | 6,400 | The denominator the consumer should have used |
| Records inserted during the run | 180 | Shift later pages forward, creating duplicates |
| Records deleted during the run | 60 | Shift later pages backward, creating omissions |
| Distinct records returned | 6,220 | Includes 140 duplicates and misses 320 |
| Completeness | 6,220 / 6,400 = 97.2% | Below the 99.5% the contract should have required |

The 320 missing records carry about EUR 2.05 million of annual spend
(`41,000,000 x 320 / 6,400`). If their prices rose by an average of 3.2% and the consumer kept the
previous values, the plan understates cost by about EUR 65,600 a year. That is smaller than the
unit error and more instructive, because nothing about it is dramatic. It is a permanent, quiet
staleness on 5% of the catalogue.

The rate limit did not cause this. Fourteen requests against 120 per minute is not a constraint.
The cause was the interaction between offset pagination and a mutable source. Rate limits become
the binding constraint when the request count is large relative to the window, which section 13
works through.

### 6.6 The identity problem

Norvik also reissues supplier item codes when it reorganises its catalogue. In the same month,
214 internal items fail to match on `supplier_item_code`. The loader's fallback is to create a new
internal item for each unmatched record.

| Category | Count | What it is | Consequence |
|---|---:|---|---|
| Genuinely new components | 61 | New products with no history | Correct to create |
| Duplicates of existing components | 57 | Same component, new supplier code | Demand history splits |
| Renames of existing components | 96 | Same component, description and code changed | History restarts at zero |

The deterministic pass resolves 132 of the 214 on GTIN plus packaging level. The remaining 82 go
to a probabilistic pass and a review queue. With a threshold that auto-links aggressively, the 57
duplicates and 96 renames are found, but some genuinely new components are also merged into
existing ones. With a threshold that is too conservative, all 153 duplicates and renames become
new items.

The cost of the conservative error is visible and bounded. Using a two-week new-item default
order quantity and an average annual spend of EUR 6,406 per component:

```text
excess initial stock = 153 x 6,406 x 2/52 = EUR 37,698
```

The cost of the aggressive error is not bounded in the same way. A new component merged into an
existing one inherits that component's history, supplier, and quality record. If the two have
different temperature requirements or shelf lives, the error can reach a customer.

### 6.7 The review that did not happen

A contract review before the 2.4.0 release would have taken about ninety minutes: six people,
including the supplier's integration owner, the procurement data owner, the planning systems
lead, and two consumers.

| Review question | What it would have found |
|---|---|
| What changes field by field? | `sell_unit` added; price precision changed; unit semantics changed for 1,180 records |
| Which population is affected? | 18.4% of components, 18.4% of spend |
| Does a unit change? | Yes, for a subset, and the contract does not declare the unit |
| Which consumers read the field? | Three: planning, procurement analytics, and the supplier scorecard |
| What did contract tests return? | No unit test existed; schema tests passed |
| What happens to history? | Nothing was planned; the price series would show a one-day 96% fall |
| How is it rolled back? | Not rehearsed |
| Who signs? | Nobody had the authority to block it |

The cost of the review is easy to state: `6 people x 1.5 hours x EUR 90 per hour = EUR 810`. The
benefit is the difference between a classified major change and an unclassified minor one. The
review does not need to be expensive. It needs to exist before the release, and it needs the
producer and the consumer in the same conversation.

> **Note:** the review is not a substitute for testing. It is the mechanism that decides which
> tests are required. A team that runs every test on every change will still miss a semantic
> change that nobody thought to assert. The review's job is to ask "what could this mean
> differently" before the test list is fixed.

### 6.8 What the analyst should have asked

1. Is `unit_price` defined per case or per sell unit in the contract, and if the answer is "it
   depends", which field carries the dependency?
2. Does the population denominator count price records or components, and which one does the
   completeness promise use?
3. Is the pagination window stable while the source is being written, and if not, what snapshot
   parameter does the interface offer?
4. Which registered consumers read the affected fields, and did any of them test version 2.4.0
   before release?
5. For each unmatched supplier code, is the correct action create, link, or review, and who
   decides?
6. What is the plan's decision impact if this field is wrong by a factor of 24 for one week, and
   who is told?

## 7. Code, query, or calculation

The artifact that carries most of this topic is a document, not a program. The block below is a
small contract for the Norvik interface, written in the spirit of the Open Data Contract
Standard. It is illustrative and not a conforming ODCS document; it is short enough to review in
a meeting.

```yaml
# Illustrative contract; ODCS-inspired, not a conforming ODCS document.
contract: norvik-packaging-prices
version: 2.4.0
status: active
owner:
  role: Supplier Data Owner, Procurement Systems
  escalation: procurement-data@example.com
schema:
  keys: [supplier_item_code, price_list_id, valid_from]
  fields:
    - name: unit_price
      type: decimal(12,4)
      unit: currency per sell_unit
    - name: sell_unit
      type: enum[CASE, EACH]
      required: true
service:
  freshness: p95 <= 6 h after price change
  completeness: >= 99.5% of active catalogue, reconciled nightly
  change_notice: 90 d for MAJOR, 30 d for MINOR with semantic change
  deprecation: 12 months after successor is generally available (GA)
```

Read the block clause by clause.

| Clause | What it makes explicit | What it would have prevented in section 6 |
|---|---|---|
| `version` | The promise has a version, and semantic changes need a major bump | A unit change shipped as 2.4.0 |
| `schema.fields.unit` | The price is per `sell_unit`, not per case | The 24-times error |
| `schema.fields.sell_unit` | The unit is mandatory and enumerated | The ignored new field |
| `service.completeness` | Completeness has a denominator and a nightly reconciliation | The 97.2% silent shortfall |
| `service.change_notice` | Semantic minor changes need 30 days' notice | The unannounced Tuesday release |
| `owner` | A named role can be asked, and can refuse | The support queue with no authority |

Two design choices in the block are deliberate and worth defending.

**The unit is inside the schema clause, not in a comment.** A comment is not testable. A field
named `unit` with an enumerated value can be asserted by a contract test on every row. The test
does not prove the supplier's arithmetic, but it proves that the consumer's interpretation is
declared rather than assumed.

**The change-notice rule distinguishes major from semantic minor.** A purely additive field is a
minor change and needs a short notice. A change that alters the meaning of an existing field is a
major change even if the schema is compatible. The contract states both, because a reader who
only knows semantic versioning will classify the second one wrongly.

The block is not a substitute for the test suite in section 5.1. It is the specification that
tells the suite what to assert. A test suite without a contract tests whatever the last engineer
thought of.

## 8. Data, standards, and regulatory context

The standards relevant here are young, partly vendor-originated, and unevenly adopted. That is
not a reason to ignore them. It is a reason to distinguish what is normative from what is a
convention that happens to be common.

### 8.1 Interface and contract standards

| Standard or specification | Governs | What it constrains | Grade |
|---|---|---|---|
| OpenAPI Specification 3.2.0, published 19 September 2025; 3.2.1 is a patch release | Description of HTTP interfaces | Operations, parameters, payload schemas, and response codes; it does not describe units, population, or service levels | B |
| Semantic Versioning 2.0.0 | Version numbering and compatibility intent | MAJOR for incompatible public API changes, MINOR for backward-compatible functionality, PATCH for backward-compatible fixes | B |
| RFC (Request for Comments) 6585, April 2012 | Additional HTTP status codes | Defines 429 Too Many Requests and the optional Retry-After header; does not define how requests are counted | B |
| RFC 8288, October 2017 | Web linking | Defines the Link header and the relation types used for pagination, including next and prev | B |
| Open Data Contract Standard 3.2.0, Bitol project | Structure of a data contract | Fundamentals, schema, quality, service levels, team, roles, and support channels in YAML | D |
| Schema Registry compatibility modes, as documented by Confluent | Event schema evolution | BACKWARD, FORWARD, FULL, and their transitive variants; product semantics, not a standard | D |
| GS1 GTIN and GLN | Entity identifiers | Product identity at a packaging level; party and location identity | B |

The first four are the ones an analyst can rely on as normative references. OpenAPI and
Semantic Versioning are specifications with defined meanings, and the two RFCs are IETF
(Internet Engineering Task Force) Proposed Standards. The ODCS is a community specification under
the LF AI and Data Foundation; it is useful and well documented, and it is not a legal standard.
Grade D reflects its origins, not its quality.

> **Note:** OpenAPI describes an interface, not a data product. It can tell you that a field is a
> string with a maximum length. It cannot tell you that the string is a currency code, that the
> amount beside it is per case, or that the endpoint returns only active records. Teams that
> treat an OpenAPI document as the contract have a machine-readable description of the syntax and
> no description of the meaning.

### 8.2 What the compatibility modes do and do not answer

Topic 03.03 introduced the event-schema compatibility modes. Their relevance here is narrower:
they answer one question, and it is not the question that reprices a plan.

| Mode | Question answered | Question left open |
|---|---|---|
| BACKWARD | Can a consumer built for the previous schema read data written with the new one? | Does the old consumer compute the right number? |
| FORWARD | Can a consumer built for the new schema read data written with the previous one? | What does the old data mean under the new semantics? |
| FULL | Can both directions be read? | Has any semantic field changed? |
| NONE | Nothing is checked | Everything |

A schema registry will accept a unit change without complaint, because the bytes still parse.
The registry is a syntax gate. The contract review is the semantics gate.

### 8.3 Regulatory and legal interfaces

Three areas of law touch this topic, and each constrains a different clause.

**Data protection.** Customer master data contains personal data in most jurisdictions. Entity
resolution across customer records increases the risk of merging two people or two households,
and a false merge can disclose one party's data to another. The schema clause should classify
personal fields, and the resolution process should treat a false merge as a potential incident,
not only a data-quality defect. Retention rules also constrain how long crosswalks and review
queues may be kept.

**Sector traceability.** Where a regulator requires event data, as with pharmaceutical
serialisation or food traceability, the event exchange has defined data elements. The interface
that carries those events is not itself the regulated standard; the content is. A contract for
such an interface should reference the regulatory data elements rather than inventing a parallel
vocabulary. Topic 09 covers the event standards; this topic covers the promise around the
interface that carries them.

**Financial and tax reporting.** Invoice and e-invoicing interfaces have statutory deadlines in
several jurisdictions. A freshness promise measured from the supplier's commit time is not the
same as a statutory filing deadline measured from the transaction date. State both clocks.

None of these regimes prescribes a data contract format. They constrain what the contract must
cover: personal data classification, regulatory data elements, and statutory timing.

### 8.4 What is convention rather than requirement

Several practices in this area feel like rules and are not:

- REST resource naming, JSON field casing, and page-size defaults are conventions. No standard
  requires `snake_case`, and no standard requires a page size of 100.
- Using 429 rather than 503 for rate limiting is conventional. RFC 6585 defines 429; it does not
  require every rate limiter to use it.
- A twelve-month deprecation window is a policy choice. Google's published stability commitment
  for generally available enterprise APIs is a minimum of one year's notice for breaking changes,
  which is a company policy and a useful benchmark, not a universal rule. Grade C for the policy
  itself.
- Applying semantic versioning to data products is a convention. Many producers version the
  endpoint and leave the payload unversioned.
- "One item per GTIN" is an internal rule, not a GS1 requirement, as Topic 02.02 explains.

An analyst who can separate the normative from the conventional can negotiate the conventional.
An analyst who cannot will treat a vendor's default as a law and pay for it in a review meeting.

## 9. Industry practice and case evidence

The evidence base for this topic is split. Entity resolution has a mature statistical literature
and decades of operational use. Data contracts are a young practice with mostly self-reported
adoption and vendor case studies. Interface completeness is well understood in computer science
and poorly measured in supply chain practice.

| Claim | Source and date | Grade | What it supports |
|---|---|---|---|
| Probabilistic record linkage can be framed as a likelihood-ratio decision with three regions: link, possible link, and non-link | Fellegi and Sunter, *JASA*, 1969 | A | The threshold model in section 13 |
| Blocking reduces quadratic pair comparison and introduces false negatives by construction | Papadakis et al., *ACM (Association for Computing Machinery) Computing Surveys*, 2020 | A | The blocking design in Step 12 |
| Probabilistic linkage remains the reference framework after fifty years, with Bayesian extensions | Tuuto, *The Survey Statistician*, 2025 | A | The continuing relevance of the method |
| 94% of surveyed organisations validate schemas, 76% define semantics, 65% state operational guarantees, and 57% define evolution rules; 62% start with document-based contracts | Atluri, *WJARR*, 2025 | E | Direction only; sample frame and selection are not described in the summary consulted |
| Production data-contract cases at Convoy, Glassdoor, and Adevinta Spain | Sanderson, Freeman, and Schmidt, *Data Contracts*, O'Reilly, 2025 | C | That the practice has real implementations, not that it has measured returns |
| Contracts stored as YAML in GitLab, enforced in a data catalogue, with freshness and completeness SLOs and schema-drift alerting | Google Cloud customer case, Virgin Media O2, December 2025 | D | An implementation pattern; vendor-published and unaudited |
| Organisations seeking a single owner for master data struggle with collaboration; control is exercised across several areas | Gartner, *Quick Answer: Who Owns Master Data?*, September 2024 | D | The per-field source-of-record position in section 5 |
| A minimum of one year's notice for breaking changes to generally available enterprise APIs | Google Cloud API stability commitments, July 2021 | C | A benchmark deprecation window, not a universal rule |

Three observations follow.

**Entity resolution is the part you can be confident about.** The mathematics is old, the failure
modes are documented, and the tools are commodity. The uncertainty is not "does probabilistic
matching work". It is "is the labelled sample representative, and does the review queue have
staff". Those are measurement and operations questions, and they are answerable locally.

**Data contracts are the part you should treat as early practice.** The adoption figures are
self-reported, the case studies are chosen by the vendors who publish them, and none of the
sources consulted reports a controlled comparison of contract adoption against a baseline. The
practice is coherent and worth adopting. The percentages are not evidence of return.

**The repricing failure is under-documented in public sources.** No audited public study was found
that quantifies the cost of a silent semantic change in a supply chain interface. The Calder case
in section 6 is a constructed composite, not a reported incident. It is internally consistent and
its arithmetic is reproducible, and it should be labelled as a teaching case rather than as
evidence of prevalence.

What practitioners actually do differs from what the guidance recommends in predictable ways.

| Recommendation | Observed practice | Why the gap persists |
|---|---|---|
| Define completeness against a producer-reported denominator | Row-count and job-status checks | The producer is not asked, and the consumer cannot compute it alone |
| Version the semantics, not only the endpoint | Path version only, no payload version | Semantic fields were never declared part of the interface |
| Register consumers and notify them | A mailing list or a changelog page | The registry is incomplete and nobody owns it |
| Review every semantic diff before release | Review only major releases | The release is classified by the schema validator, which passes |
| Assign source of record per field | One system declared the golden record | Per-field governance is slower and requires business decisions |
| Keep a dated crosswalk | Crosswalk exists but is undated | Merges are treated as one-off cleanups, not history |

The last row is the quiet one. A crosswalk without validity dates cannot answer "which entity did
this identifier mean in March", which is exactly the question an audit, a recall, or a restated
forecast asks.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| HTTP 200 treated as completeness | No denominator in the pipeline; row counts vary but nobody checks | Success is measured per request, not per population | A stable subset is planned as if it were the whole | Producer-reported count; reconciliation per run |
| Offset pagination over a mutable source | Row count drifts by a few hundred; duplicates appear intermittently | Inserts and deletes shift page windows | Records missed and duplicated in the same run | Snapshot parameter, cursor on an immutable key, or a stable sort |
| Retry policy ignores Retry-After | Bursts of 429s followed by a gap | Fixed backoff shorter than the server's window | Truncated read, or a retry storm that extends it | Honour Retry-After; exponential backoff with jitter; cap total attempts |
| Unit change shipped as a minor release | A price or quantity series steps by a round factor | Semantics were outside the declared public API | Cost, margin, and order quantity wrong by that factor | Unit is a contract field and a contract test |
| Enum value added without consumer review | A CASE statement silently falls to its else branch | Consumers assume a closed set | Records classified as unknown or default | Publish enum policy; test exhaustive consumers |
| Null meaning changed from unknown to not applicable | Null rate unchanged, so no alert fires | The value is absent for a different reason | Averages and coverage computed on the wrong base | Declare null semantics; test the meaning, not only the rate |
| Timestamp changed from local to UTC | Events cluster at an unusual hour | Offset removed without a migration | Late or early facts; wrong daily buckets | Declare timezone; dual-write through a transition |
| Deletes invisible | Population grows monotonically; retired records persist | No tombstone or deletion endpoint | Phantom items and prices in the plan | Contract a deletion signal; reconcile counts |
| Exact match creates a new entity for every rename | New-item count rises after a supplier reorganisation | No deterministic key, or a key that changes | Split history, duplicate demand, wrong defaults | GTIN plus pack, GLN, legal identifier; review queue |
| Threshold tuned on an unbalanced sample | Reported accuracy is high while users complain about wrong merges | Accuracy is dominated by the majority non-match class | False merges at scale, invisible in the metric | Precision and recall on a stratified labelled sample |
| Transitive closure chains distinct entities | One cluster contains records that pairwise score low | A-B and B-C links imply A-C without evidence | Irreversible false merge of a cluster | Review chained edges; constrained clustering |
| Golden record overwritten by last write | Attribute values flip between reports | Survivorship unspecified | Unexplainable changes in cost or description | Per-field precedence with named owners |
| Contract exists but is not enforced | The document is older than the endpoint | No test, no gate, no owner with veto | False assurance; the contract becomes decoration | Tests under continuous integration (CI); review gate; owner sign-off |
| Consumer registry incomplete | The deprecation notice reaches three of seven consumers | Registration is voluntary | A consumer breaks on removal day | Discover consumers from logs and contracts; make registration a condition of access |

Two rows deserve expansion. **HTTP 200 treated as completeness** is the most common failure in
API-fed planning, because it is cheap: the request succeeded, the payload parsed, and the row count
is plausible. The fix is a denominator the producer states and the consumer reconciles, with a
failure that blocks the plan. **The unit change shipped as a minor release** is the most
expensive, because it combines a correct producer decision under a narrow definition of the public
API with a correct consumer decision to ignore unknown fields. Only a contract that declares units
as public API, plus a diff-triggered review, prevents it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The producer can state a population count | Completeness needs a denominator | The interface has no count endpoint; permissions hide records | Ask for a count by segment; compare with an independent extract |
| A stable identifier exists per entity | Deterministic matching and crosswalks depend on it | Supplier codes are reissued; GTINs are missing; legal names change | Measure key stability over twelve months |
| Pagination is consistent for the duration of a read | A full read is assumed to be one logical snapshot | Offset pagination on a mutable table; cursor expiry | Re-read a sample page and compare; check for a snapshot parameter |
| The contract describes runtime behaviour | Tests assert the contract | The contract is stale, or describes an intended future state | Diff the contract against observed payloads monthly |
| The consumer registry is complete | Deprecation notices need recipients | Voluntary registration; shadow consumers | Discover consumers from access logs and job metadata |
| A labelled sample exists for threshold selection | Precision and recall require ground truth | No reviewed pairs; labels only from the review queue, which is biased | Build a stratified sample of pairs and have two reviewers label it |
| False-merge and false-split costs can be estimated | Threshold choice is a cost trade-off | Costs are political or unmeasured | Use a range; test whether the chosen threshold changes across the range |
| Upstream teams have authority to enforce the contract | A contract without a veto is advisory | The producer is external or more powerful | Ask who can block a release; if nobody, negotiate a service credit |
| Event time and ingestion time are both retained | Late data and freshness cannot be distinguished otherwise | The pipeline overwrites event time with load time | Inspect the schema for two timestamps |
| Identity is stable over the planning horizon | Joins assume one entity per key | Acquisitions, renames, splits, divestitures | Monitor entity count against a commercial register |

Two limits are worth stating plainly. **This topic describes the boundary, not the entire
integration estate.** A company with two hundred interfaces cannot review every diff at this
level of detail, so it tiers interfaces by decision impact: the twenty that feed price, lead time,
capacity, and service get the full contract and review, and the rest get automated schema and
freshness tests. **The contract does not make an untrustworthy producer trustworthy.** It makes the
promise explicit and the breach detectable. If a supplier will not state a unit or a completeness
figure, that refusal is itself information: the consumer can price the risk, add a buffer, or find
another source.

## 12. Exercises

### Exercise 1 — Interface record and completeness audit (analysis, 60 minutes)

Choose an API you can inspect through its public documentation or your own organisation. Write an
interface record covering the shape, resources, identifiers, row grain, population statement, page
size, rate limit, and retention.

**Deliverable:** one page with a resource table, a population statement in four parts (in scope,
excluded, as of when, under whose permissions), and a feasibility calculation.
**Acceptance criteria:** the denominator is a count of a named entity, not "the rows"; at least two
completeness hazards are identified with the mechanism that causes each; event time, availability
time, and decision time are distinguished.

### Exercise 2 — Write a four-clause contract (design, 75 minutes)

Write a contract for a supplier price interface that returns item, price list, currency, unit,
validity dates, and a last-modified timestamp. Use YAML or a table, but include all four clauses.

**Deliverable:** a contract document of no more than one page, plus a list of the tests that would
enforce it.
**Acceptance criteria:** the unit and currency are declared fields, not comments; null meaning is
stated for at least three fields; key uniqueness is stated; freshness has a percentile and a
window; completeness has a denominator and a reconciliation method; a named role owns the
contract; every clause maps to at least one automated test.

### Exercise 3 — Classify eight changes (application, 45 minutes)

Classify each change as major, minor, or patch for a named consumer, and state whether a schema
validator would accept it: (a) add an optional field; (b) remove a deprecated field; (c) add an
enum value; (d) change a unit; (e) change null from unknown to not applicable; (f) widen a decimal
from two to four places; (g) rename a field with a six-month alias; (h) change the default filter
from all records to active records.

**Deliverable:** an eight-row table with columns change, schema-compatible, semantic class,
contract action, and consumer test.
**Acceptance criteria:** at least three rows are schema-compatible but semantically major; the
compatibility direction is stated correctly; one row explains why the classification depends on
what the contract declares as public API.

### Exercise 4 — Pagination and rate-limit arithmetic (calculation, 50 minutes)

An interface holds 6,400 records, returns 500 per page, and allows 120 requests per minute per
client, shared with two other consumers. The extraction window is 30 minutes. Separately, a
per-item endpoint would require 6,400 requests at an allocated 40 requests per minute.

Compute the page count, the feasible request budget, and the time required for each approach. Then
compute completeness if 320 records are missed and 140 are duplicated in a run that returns 6,220
distinct records.

**Deliverable:** the arithmetic with units, and a one-sentence recommendation on which approach to
use.
**Acceptance criteria:** page count is rounded up; the shared rate is divided correctly; the
per-item approach is shown to be feasible or infeasible against the window; completeness is stated
as a percentage and as a missing-record count.

### Exercise 5 — Precision and recall at three thresholds (calculation, 60 minutes)

Use these 20 scored candidate pairs. Scores are listed in descending order; the label is M for a
true match and N for a non-match.

```text
scores: 0.99 0.97 0.96 0.94 0.93 0.91 0.89 0.88 0.86 0.84
        0.82 0.79 0.76 0.74 0.71 0.68 0.65 0.61 0.58 0.55
labels:  M    M    M    N    M    M    M    N    M    N
         M    N    M    N    N    M    N    N    N    N
```

Compute true positives, false positives, false negatives, precision, and recall at thresholds
0.90, 0.80, and 0.70.

**Deliverable:** a threshold table with the counts and both metrics, and a sentence on which
threshold you would choose before knowing the error costs.
**Acceptance criteria:** precision and recall are computed with the correct denominators; the
counts sum to 20 at every threshold; the choice is stated as provisional and the reason is given.

### Exercise 6 — Choose a threshold from error costs (calculation, 40 minutes)

Using the counts from Exercise 5, assume a false merge costs EUR 4,200 and a false split costs
EUR 600. Compute the total error cost at each threshold. Then repeat with a false-split cost of
EUR 5,000 and find the crossover false-split cost at which the choice changes between 0.90 and
0.80.

**Deliverable:** two cost tables and one crossover calculation.
**Acceptance criteria:** every cost is the product of a count and a unit cost; the crossover is
solved algebraically, not guessed; the answer states that the threshold is a function of the cost
ratio, not a universal constant.

### Exercise 7 — Customer crosswalk and golden record (design, 75 minutes)

A distributor acquires a regional customer with 480 ship-to records. 312 match on GLN exactly. The
remaining 168 have names, addresses, and tax identifiers in inconsistent formats. Design the
resolution: deterministic rules, two blocking keys, a score, a threshold policy, a review queue,
a canonical identifier, and a crosswalk schema.

**Deliverable:** a two-page design with a worked example of one matched and one rejected pair.
**Acceptance criteria:** both block keys are justified against a specific error type; the review
band is non-empty and its expected size is estimated; the crosswalk has validity dates and a
reason code; the design states which error is worse for credit and rebate use and why.

### Exercise 8 — Red-team a proposed change (critique, 60 minutes)

A supplier proposes to redefine `available_qty` from "on hand" to "on hand minus allocated". The
field name, type, and nullability do not change. Prepare the contract review.

**Deliverable:** a completed eight-question review agenda, a change classification, and an impact
statement naming the plan parameters affected.
**Acceptance criteria:** the change is classified as semantically major despite schema
compatibility; at least three plan parameters are named (for example safety stock, available-to-
promise, and allocation); the rollback and the sign-off roles are stated; the review identifies
what evidence would change the classification.

## 13. Self-check answers

**Exercise 4 arithmetic.**

```text
pages              = ceil(6,400 / 500) = 13
shared rate        = 120 / 3 consumers = 40 requests per minute
page run           = (13 + 1 metadata) / 40 = 0.35 min = 21 s
per-item run       = 6,400 / 40 = 160 min, against a 30 min window -> infeasible
distinct received  = 6,220
completeness       = 6,220 / 6,400 = 97.2%
missing records    = 320
duplicate records  = 140
net shortfall      = 320 - 140 = 180 = 6,400 - 6,220
```

The recommendation is to use the paged endpoint, not the per-item endpoint. The paged run is
limited by pagination consistency, not by rate. The per-item run is infeasible without a cache, a
bulk export, or a higher allocation.

**Exercise 5 arithmetic.**

At each threshold, a pair is predicted positive when its score is greater than or equal to the
threshold. The 20 labels contain 10 true matches and 10 non-matches.

| Threshold | TP (true positive) | FP (false positive) | FN (false negative) | Precision `TP/(TP+FP)` | Recall `TP/(TP+FN)` |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 5 | 1 | 5 | 5/6 = 0.833 | 5/10 = 0.500 |
| 0.80 | 8 | 3 | 2 | 8/11 = 0.727 | 8/10 = 0.800 |
| 0.70 | 9 | 6 | 1 | 9/15 = 0.600 | 9/10 = 0.900 |

The counts reconcile: at 0.90, 6 predicted positives plus 14 predicted negatives is 20. At 0.80,
11 plus 9 is 20. At 0.70, 15 plus 5 is 20. A submission that reports accuracy alone has not answered
the question: accuracy is 0.70, 0.75, and 0.65 at the three thresholds, which hides the
precision-recall trade-off.

**Exercise 6 arithmetic.**

| Threshold | False merges | False splits | Cost at EUR 4,200 / EUR 600 | Cost at EUR 4,200 / EUR 5,000 |
|---:|---:|---:|---:|---:|
| 0.90 | 1 | 5 | 4,200 + 3,000 = 7,200 | 4,200 + 25,000 = 29,200 |
| 0.80 | 3 | 2 | 12,600 + 1,200 = 13,800 | 12,600 + 10,000 = 22,600 |
| 0.70 | 6 | 1 | 25,200 + 600 = 25,800 | 25,200 + 5,000 = 30,200 |

Crossover between 0.90 and 0.80, holding the false-merge cost at EUR 4,200:

```text
4,200 + 5C = 12,600 + 2C
3C = 8,400
C = 2,800
```

If a false split costs more than EUR 2,800, the lower threshold wins. If it costs less, the higher
threshold wins. The threshold is a function of the cost ratio and the score distribution, not a
property of the algorithm. A team that adopts 0.85 because a vendor recommended it has skipped both
inputs.

**Exercise 3 rubric.**

| Change | Schema-compatible | Semantic class | Action |
|---|---|---|---|
| Add optional field | Yes | Minor, major for strict consumers | Minor, announce |
| Remove field | No | Major | Major, deprecation |
| Add enum value | Yes | Minor syntactically, may be major | Minor, consumer test |
| Change unit | Yes | Major | Major, conversion guidance |
| Null unknown to not applicable | Yes | Major | Major, backfill |
| Widen decimal | Yes | Minor | Minor, test rounding |
| Rename with alias | Yes during alias | Minor then major | Minor, dated removal |
| Default filter changes | Yes | Major | Major, count reconciliation |

A strong answer marks at least the unit, null, and filter rows as schema-compatible and
semantically major. A weak answer marks them major only if a field was removed.

**Exercise 2 rubric.** A strong contract states the unit and currency as fields, defines null as
unknown versus not applicable, gives a key and its uniqueness claim, states freshness at p95 with a
measurement origin, states completeness with a denominator and a nightly reconciliation, names a
deprecation window and a notice channel, and names an owner role with escalation. Each clause has a
test. A weak contract lists field names and types and calls itself a contract.

**Exercise 7 rubric.** A strong design uses GLN as the deterministic rule, then blocks on
(normalised name, postal prefix) and on (tax identifier, country); scores on name similarity plus
address agreement; auto-links above 0.95, reviews 0.75 to 0.95, rejects below 0.75; estimates the
review queue from the pair-generation rate; assigns a new canonical identifier and retains the old
GLNs in a crosswalk with `valid_from`, `valid_to`, and a reason code. It states that for credit and
rebates the false merge is the expensive error, so the high threshold is deliberate. A weak design
uses name only, has no review band, and overwrites the old identifier.

**Exercise 8 rubric.** A strong answer classifies the change as semantically major, notes that
schema validation passes, and names safety stock, available-to-promise, and allocation as affected
parameters. It requires a dual-run period, a restated history, a rollback, and named sign-offs. It
states what evidence would change the classification, for example a written guarantee that
`available_qty` was already net of allocations in every prior version. A weak answer treats the
change as minor because the field name and type are unchanged.

## 14. Interview questions

| # | Question | Strong answer | Weak answer |
|---:|---|---|---|
| 1 | What is the difference between an API and a data feed? | A feed is a producer-defined snapshot; an API is a consumer-called query surface with producer-set limits, so the consumer must establish the population separately. | "An API is real time and a feed is a file." |
| 2 | Why can an HTTP 200 response still be incomplete? | Success is per request; completeness is per population. Pagination, filters, permissions, and deletes can all remove records while every response succeeds. | "It cannot; 200 means success." |
| 3 | What are the four clauses of a data contract? | Version, schema, service, and owner: what promise, what meaning, what freshness and completeness, and who is accountable. | "Schema and SLA." |
| 4 | When is a schema-compatible change still breaking? | When it changes a unit, a null meaning, a timezone, a population, or key uniqueness, because old code still parses the bytes and computes a different number. | "Only a removed field is breaking." |
| 5 | How do you choose a match threshold? | From a labelled, stratified pair sample, the precision and recall trade-off, and the relative cost of a false merge and a false split; then review the uncertain band. | "Use 0.85 because that is standard." |
| 6 | Why are false merges and false splits not symmetric? | A false merge commingles history and is often irreversible; a false split is visible and repairable if raw history is kept. The cost ratio, not accuracy, sets the threshold. | "Both are just errors; minimise the total." |
| 7 | What is blocking and what does it cost? | Partitioning records into candidate groups to avoid comparing every pair. It costs recall by construction, because a true pair with different block keys is never compared. | "It makes matching faster with no downside." |
| 8 | Why assign source of record per field rather than per system? | Different systems are authoritative for different attributes, and a single-system rule overwrites expert fields such as cost or shelf life with another function's data. | "The ERP is the golden record for everything." |
| 9 | Design a deprecation policy for a supply chain API. | A named consumer registry, a notice window tied to planning cycles, a dual-run or alias period, a sunset date, and a removal gate that checks registered consumers have migrated. | "Announce it in the changelog and remove it next quarter." |
| 10 | Walk through the review for a supplier changing available_qty to net of allocations. | Classify it as semantically major despite schema compatibility; name safety stock, available-to-promise (ATP), and allocation as affected; require a dual-run, restated history, rollback, and named sign-offs. | "The field name and type are unchanged, so it is a minor release." |

Each strong answer separates a syntactic test from a semantic one. Candidates who cannot make that
separation will pass every schema check and still ship the repricing error in section 6.

## 15. Key terms

- **API** — a defined boundary for requesting data or actions from another system.
- **REST** — a resource-oriented HTTP style; **GraphQL** — a client-specified query shape.
- **Data contract** — a producer's versioned, testable promise about schema, service, and owner.
- **ODCS** — the Open Data Contract Standard, a YAML structure for contracts.
- **Schema** — fields, types, and structure; **semantics** — what those fields mean.
- **Population** — the records the interface claims to represent, with a count.
- **Freshness** — event time to availability; **completeness** — present records over population.
- **Pagination** — splitting results into pages; **cursor** — an opaque position token.
- **Rate limit** — a request cap; **429** — too many requests; **Retry-After** — the wait signal.
- **Semantic versioning** — MAJOR.MINOR.PATCH tied to compatibility of the public API.
- **Breaking change** — a change that invalidates a consumer's assumptions, including semantic ones.
- **Deprecation window** — the announced period before removal; **sunset date** — the removal date.
- **Consumer registry** — the list of known consumers entitled to notice.
- **Contract test** — an automated assertion that a clause holds on real data.
- **Entity resolution** — deciding whether records denote the same entity.
- **Deterministic match** — exact agreement on strong identifiers; **probabilistic match** — a score.
- **Blocking** — grouping records to avoid all-pairs comparison; **blocking key** — the grouping fields.
- **Precision** — true positives over predicted positives; **recall** — true positives over actual positives.
- **False merge** — two entities treated as one; **false split** — one entity treated as two.
- **Transitive closure** — chaining pairwise links into clusters, with a risk of over-merging.
- **Canonical identifier** — the surrogate assigned to a resolved entity, never reused.
- **Crosswalk** — a dated mapping between identifiers across systems and time.
- **Source of record** — the system authoritative for a named field.
- **Survivorship** — the per-attribute rule for choosing a value when sources disagree.
- **MDM** — master data management, the process of governing and resolving master data.

## 16. Sources

**Standards and specifications**

- OpenAPI Initiative, *OpenAPI Specification 3.2.0*, published 19 September 2025; 3.2.1 patch
  release. `https://spec.openapis.org/oas/v3.2.0.html` (Grade B)
- Preston-Werner, T., *Semantic Versioning 2.0.0*. `https://semver.org/spec/v2.0.0.html` (Grade B)
- Nottingham, M., and Fielding, R., *RFC 6585: Additional HTTP Status Codes*, IETF, April 2012.
  `https://datatracker.ietf.org/doc/html/rfc6585.html` (Grade B)
- Nottingham, M., *RFC 8288: Web Linking*, IETF, October 2017.
  `https://datatracker.ietf.org/doc/html/rfc8288.html` (Grade B)
- Bitol project, *Open Data Contract Standard 3.2.0*.
  `https://bitol-io.github.io/open-data-contract-standard/v3.2.0/` (Grade D — community
  specification under LF AI and Data; useful, not a de jure standard)

**Peer-reviewed and academic**

- Fellegi, I. P., and Sunter, A. B. (1969). "A Theory for Record Linkage." *Journal of the
  American Statistical Association*, 64(328), 1183–1210. (Grade A)
- Papadakis, G., Skoutas, D., Thanos, E., and Palpanas, T. (2020). "Blocking and Filtering
  Techniques for Entity Resolution: A Survey." *ACM Computing Surveys*, 53(2), Article 31.
  (Grade A)
- Tuuto, P. (2025). "Over 50 Years of Probabilistic Record Linkage: A Journey with Three Selected
  Stops." *The Survey Statistician*, 92, 17–29. (Grade A)
- Sayers, A., Ben-Shlomo, Y., Blom, A. W., and Steele, F. (2015). "Probabilistic Record Linkage."
  *International Journal of Epidemiology*. DOI (digital object identifier): 10.1093/ije/dyv322. (Grade A)

**Industry, analyst, and practitioner**

- Sanderson, C., Freeman, M., and Schmidt, M. (2025). *Data Contracts*. O'Reilly Media. Case
  chapters on Convoy, Glassdoor, and Adevinta Spain. (Grade C)
- Atluri, S. (2025). "Data contracts in the wild: An approach for redefining trust and
  accountability in modern data pipelines." *World Journal of Advanced Research and Reviews*.
  (Grade E — self-reported adoption figures; sample frame not described in the summary consulted)
- Google Cloud, "Virgin Media O2 uses data contracts to build scalable AI and data products,"
  December 2025. (Grade D — vendor case study)
- Gartner, *Quick Answer: Who Owns Master Data?*, September 2024; *4 Essentials of Successful
  Master Data Management*, September 2025; *Prioritize Master Data Quality to Enable Your Supply
  Chain Strategy*, November 2025. (Grade D — vendor-analyst abstracts)
- Google Cloud, "New API stability tenets govern Google Enterprise APIs," July 2021, and the
  Google Cloud deprecation policy. (Grade C — company policy, publicly stated)

**Vendor documentation**

- Confluent, *Schema Evolution and Compatibility for Schema Registry*.
  `https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html`
  (Grade D — product semantics)

**Note on citations.** No audited public study was found that quantifies the cost of a silent
semantic change in a supply chain interface. The Calder Foods case is a constructed teaching
composite. Adoption percentages in the Atluri paper are self-reported and graded E; vendor and
analyst grades follow Module 28.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_gs1-epcis.md`](09_gs1-epcis.md)
- Ingestion mechanics assumed here: [`03_ingestion-patterns.md`](03_ingestion-patterns.md)
- Document exchange and versions: [`08_edi-x12-edifact.md`](08_edi-x12-edifact.md)
- Next topic: [`11_quality-quarantine-lineage.md`](11_quality-quarantine-lineage.md)
- Master data foundations: [`../02_supply_chain_data_landscape/02_master-data.md`](../02_supply_chain_data_landscape/02_master-data.md)
- Access, licensing, and permissions: [`../02_supply_chain_data_landscape/07_access-and-licensing.md`](../02_supply_chain_data_landscape/07_access-and-licensing.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
