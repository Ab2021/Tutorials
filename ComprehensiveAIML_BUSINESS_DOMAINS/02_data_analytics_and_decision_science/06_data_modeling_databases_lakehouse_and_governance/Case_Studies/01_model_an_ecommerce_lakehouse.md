---
case_id: "CASE-2026-DS06-001"
case_title: "Model an e-commerce lakehouse"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "streaming-systems"
domain: "E-commerce analytics, financial reconciliation, and data governance"
primary_pattern: "reconcilable-event-ledger"
secondary_patterns: ["contract-gated-serving", "late-data-recovery"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Director of Data Platform"
source_index: "INDEX.md"
review_trigger: "A payment provider, order platform, identity policy, close timetable, workload profile, or table-format guarantee changes."
---

# Model an E-commerce Lakehouse

> **Source declaration:** This is an illustrative design exercise, not a description of a retailer or a deployed platform. Every order, amount, workload, cost, and recovery result below is constructed for teaching. Arithmetic is checked in Section 6; technology behavior, prices, and legal obligations require local verification before adoption.

## 1. The One-Line Framing

Given one immutable commerce event at a time plus versioned order, payment, refund, product, and consent records, decide which certified sales and customer-serving tables to publish for finance, merchandising, and lifecycle marketing; the Data Platform Director must meet the daily close and privacy constraints against a simpler nightly warehouse extract baseline.

- **Task and frequency:** system-operation decision; certify an intraday operational view every hour and a finance-close view after the agreed watermark each UTC day.
- **Input grain:** one source mutation or business event, identified by `(source_system, event_id)` and carrying `event_time`, `knowledge_time`, and a payload version.
- **Output:** governed tables with an explicit as-of version, reconciliation status, and permitted-purpose tag; a dashboard query is never allowed to infer certification from freshness alone.
- **Beneficiaries:** Finance receives auditable net sales; merchandising receives item and promotion facts; marketing receives only purpose-authorized identity links.
- **Baseline:** a nightly extract/load warehouse with overwrites and manual spreadsheet reconciliation.
- **Binding constraint:** no consumer may silently include an unapproved refund, a duplicate payment replay, a deleted identity, or post-cutoff knowledge in a certified result.
- **Primary success condition:** for an agreed close cutoff, the published ledger has one admissible accounting effect per business obligation and can explain every difference from provider settlement.
- **Pattern:** `reconcilable-event-ledger`; secondary patterns are `contract-gated-serving` and `late-data-recovery`.
- **Why a simpler framing fails:** “put all tables in a lake” does not specify the facts, time boundary, delete behavior, or owner needed for a number to survive a late refund.

## 2. Context & Background

An illustrative retailer sells physical goods through web and mobile channels. Checkout creates orders, a payment provider authorizes and captures funds, fulfillment ships units, support can issue partial refunds, and customers may change consent or request deletion. The same business day therefore contains states, corrections, and reversals rather than one stable “order row.”

| Context field | Illustrative condition | Decision consequence |
| --- | --- | --- |
| Workflow | checkout → payment → fulfillment → refund/chargeback → settlement | Net sales cannot be derived from checkout alone. |
| Stakeholders | Finance Controller, Head of Merchandising, CRM lead, Privacy lead, Data Platform Director | Each needs a different serving contract and override authority. |
| Baseline | nightly CSV extracts overwrite prior rows; finance matches exceptions manually | History and replay identity are lost; close is slow. |
| Scale assumption | 2,000,000 events/day; 1 KiB compressed logical payload average | Sizing is an estimate, not a benchmark. |
| Timing | UTC event time; daily finance close at 05:00 UTC next day | Arrival after close needs a restatement policy. |
| Geography | illustrative multi-region customer base | No jurisdiction-specific conclusion follows. |
| Current weakness | customer_id reused across guest merge and marketing identifiers | Identity must be a time-bounded bridge, not a universal key. |

- The Controller owns whether a finance-close number is certified or restated.
- The Data Platform Director owns ingestion, storage contracts, and recovery runbooks.
- The payment-operations lead owns provider-to-ledger reconciliation exceptions.
- The Privacy lead owns deletion and purpose restrictions; it does not own a sales arithmetic decision.
- The CRM lead owns activation eligibility but cannot override a withdrawal of consent.
- The warehouse baseline remains credible when sources are stable, batch-only, low-volume, and audit history is handled elsewhere.
- A lakehouse is considered because immutable change history and one governed storage layer may reduce copied pipelines; it is not selected merely because it supports streaming.
- No actual retailer, provider, cost, compliance control, or performance result is claimed.

## 3. The Business Question

| Decision contract | Definition |
| --- | --- |
| Question | Should the team replace overwrite-oriented nightly extracts with a contract-gated event ledger and versioned serving tables for the named analytics consumers? |
| Decision owner | Director of Data Platform; Finance Controller has veto over close certification. |
| Trigger and deadline | Refund disputes and replay incidents make the next-quarter reporting design decision necessary; choose design before implementation funding. |
| Available actions | retain batch warehouse; improve batch with append audit tables; adopt a limited lakehouse ledger and curated tables; build a broad enterprise platform. |
| Value and downside | Faster explainable close and safer reuse versus cost, operational skill, and a larger control surface. |
| Objective | Publish a reconcilable daily net-sales view and purpose-limited customer view without silently changing prior certified snapshots. |
| Hard constraints | Ledger effects must be deduplicable; finance cutoff reproducible; deletion requests traceable; no raw PII in general analytics. |
| Scope | Orders, payments, refunds, product lines, customer identity links, consent, and four named serving contracts. |
| Out of scope | Demand forecasting, recommendation models, tax determination, payment authorization, and legal interpretation. |

The proxy “pipeline succeeded” is insufficient: a green run can calculate a wrong total. The business guardrail is key-level and value-level reconciliation against the provider settlement population, with withheld or restated certification when it fails.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Check / evidence plan | Status | Impact if wrong | Fallback / owner |
| --- | --- | --- | --- | --- | --- | --- |
| A-01 | Provider capture and refund messages expose stable provider transaction IDs. | data | Inspect contract and sample payloads. | unverified | Cannot deduplicate effects safely. | Use settlement-file key; Payment Ops. |
| A-02 | Source emits operation and source sequence or update timestamp. | data | Contract test on every feed. | unverified | State reconstruction becomes ambiguous. | Append snapshots; Platform. |
| A-03 | Event time and knowledge time can be recorded separately. | semantic | Schema gate. | proposed | Cannot reproduce close. | Use ingest-time close with explicit limitation; Controller. |
| C-01 | Finance close is at 05:00 UTC, with restatements separately labeled. | business | Controller sign-off. | review needed | Certification boundary contested. | Retain batch close; Controller. |
| C-02 | Raw PII is accessible only to identity service and controlled zone. | privacy | Access-policy review. | review needed | Unauthorized serving exposure. | Tokenize and block publication; Privacy lead. |
| C-03 | Illustrative storage budget is 3 TiB/month payload before replication and metadata. | cost | Measure representative payload and retention. | unverified | Wrong sizing or cost. | Keep warehouse baseline; Platform. |
| A-04 | A partial refund references an order line or allocation rule. | semantic | Refund contract test. | unverified | Line-level margin is misstated. | Hold line view; Finance. |
| A-05 | A deletion request is authoritative for identity serving. | governance | Privacy process review. | unverified | Customer table retains prohibited row. | Quarantine identity output; Privacy lead. |

The assumptions most likely to reverse the recommendation are stable payment identity, the close/restatement policy, and the true volume/retention cost. None is treated as verified because this is illustrative.

## 5. Data, Tools & Method

### 5a. Data Specification

| Dataset | Grain and keys | Time semantics | Allowed use / exclusion |
| --- | --- | --- | --- |
| bronze_commerce_event | one received mutation; `(source_system,event_id)`; payload hash | `event_time` from producer; `knowledge_time` when platform first accepts it | restricted raw; retain tombstones and malformed records separately |
| silver_order_version | one order version; `(order_id, source_version)` | valid-from/valid-to transaction interval; known-at time | operations and audit; not a sales effect |
| silver_payment_effect | one accounting candidate effect; `(provider, provider_txn_id, effect_type)` | provider occurred time plus knowledge time | reconciliation input; pending authorization excluded from net sales |
| silver_refund_effect | one refund allocation; `(refund_id,line_id)` or documented order allocation | refund occurred time plus knowledge time | negative effect; unallocated line views withheld |
| gold_daily_net_sales | one business-date, currency, channel, product aggregation and certified snapshot id | business date by event time; as-of knowledge cutoff | finance only when reconciliation gate passes |
| gold_customer_activation | one identity-purpose-time eligibility row | known-at cutoff and consent effective interval | marketing only; no raw PII; deleted/withdrawn suppressed |

- **Business keys:** `order_id` identifies an order business object; `order_line_id` identifies a sellable line; provider transaction ID identifies a payment rail effect; `customer_subject_id` is an internal pseudonymous subject, not an email.
- **Surrogate keys:** generated warehouse keys may speed joins but never replace deduplication or source traceability.
- **Event grain:** an event records a fact or mutation, not the current “truth.” A replay of the same event ID is not a second sale.
- **State grain:** a versioned order state records one source version. A current snapshot selects the greatest admissible source version, not the latest arrival blindly.
- **Time:** `event_time` answers when business says it occurred; `knowledge_time` answers when this platform could have known it; `ingest_time` supports operations but cannot substitute for either.
- **CDC operation:** INSERT, UPDATE, DELETE, and snapshot-marker are explicit fields. A DELETE is a state transition or tombstone, never a physical disappearance from the audit ledger.
- **Late arrival:** accepted if schema-valid; it may revise a provisional view but cannot rewrite the meaning of a previously certified snapshot without a restatement record.
- **Duplicate rule:** same source/event key with same canonical payload is idempotent; same key with different payload is quarantined as a producer conflict.
- **Join rule:** order-to-line is one-to-many; payment-to-order can be many-to-one; refunds can be many-to-many only through an explicit allocation table.
- **Currency rule:** all arithmetic in the worked ledger is USD illustrative major units; production requires currency, FX, tax, and rounding contracts.

### 5b. Tools & Stack

| Component | Proposed role | Why it exists | Not implied |
| --- | --- | --- | --- |
| Append-only object storage | durable raw events and immutable manifests | replay and forensic evidence | a particular cloud vendor |
| Transactional table format | atomic versioned curated tables | read a named snapshot; merge CDC | universal low latency |
| Stream or micro-batch processor | validate, deduplicate, and construct effects | bounded processing and checkpoints | exactly-once business semantics |
| Catalog and policy engine | schema, owner, purpose and access tags | block unauthorized serving | legal compliance by itself |
| Orchestrator | close gate and backfill workflow | named dependencies and retries | that a DAG proves correctness |
| SQL engine | certified aggregate and reconciliation query | auditable calculation | a single query engine choice |
| Secrets/key manager | provider credentials and token mapping | separate identity access | identity resolution correctness |

Names of vendors are intentionally omitted. The architecture decision concerns capabilities and contracts; implementation selection must compare locally available skills, support, interoperability, throughput, recovery behavior, and price.

### 5c. Method Summary

| Candidate | Evidence considered | Strength | Limit / avoid when | Status | Revisit if |
| --- | --- | --- | --- | --- | --- |
| Nightly overwrite warehouse | illustrative baseline only | simple operations and predictable batch cost | loses mutation lineage unless separately archived | considered | few corrections and no intraday consumers |
| Batch warehouse plus immutable audit append | illustrative design comparison | preserves history with modest change | serving reconciliation still custom | considered | daily close is sole consumer |
| Contract-gated lakehouse ledger | illustrative design | one versioned path from events to certified tables | more operating controls and skills required | proposed | late change and reuse justify controls |
| Enterprise multi-domain platform | not tested | may standardize broad governance | scope and cost exceed bounded decision | rejected for now | multiple domains share confirmed needs |

- Input → canonicalize schema and units → validate contract → deduplicate event identity → append raw ledger → derive versioned entities/effects → reconcile → publish certified snapshot or refusal.
- A successful transport acknowledgement is not a successful ledger effect; effect admission requires identity, schema, and semantic checks.
- The method does not promise exactly-once delivery. It creates exactly-once *accounting effect eligibility* only if source identity and conflict policy hold.
- Orders are stateful context; net sales is an effect ledger. Do not sum order snapshots and payment effects in one denominator.
- The simpler batch warehouse wins if the required consumer is only a daily report, inputs are append-only and stable, privacy joins are absent, and audit history already exists at the source.
- The lakehouse proposal wins only after acceptance tests demonstrate reproducible snapshots, bounded restatement, correct delete propagation, and sustainable operating cost.

### 5d. Data provenance, quality, exclusions and leakage controls

| Control | Predicate | Failure action | Owner |
| --- | --- | --- | --- |
| Schema compatibility | declared schema version is known and required fields parse | quarantine partition; block affected gold publication | Producer + Platform |
| Identity uniqueness | same event key has one canonical payload hash | quarantine conflicting duplicate; retain exact replay marker | Platform |
| Effect eligibility | capture/refund has provider ID, currency, amount, and mapped order | exclude from certified net sales; create exception | Payment Ops |
| Order allocation | refund allocation totals refund amount within currency rounding rule | withhold line-level view | Finance |
| Close watermark | all required source manifests through cutoff are present or explicitly excepted | publish provisional only or hold close | Controller |
| Settlement reconciliation | admissible provider effects match settlement key and amount within contract | block finance certification | Payment Ops |
| Consent/deletion | serving row has active purpose and no effective deletion | deny/withdraw row from activation | Privacy lead |

- Raw source payloads and restricted identity mapping are separate zones; general analysts use pseudonymous curated tables.
- A catalog tag is not protection if the engine ignores it; access policy must be enforced at the serving boundary and tested with a denied principal.
- PII minimization: use a subject token for joins, retain the reidentification mapping only where the identity service needs it, and emit purpose-specific projections.
- Deletion: write a deletion request event, identify affected identity projections, revoke them from current serving tables, and retain only a policy-approved tombstone/audit reference in restricted evidence. Exact retention is OPEN pending policy.
- Knowledge-time leakage: a report “as of 2026-11-03 05:00 UTC” can include events with older event times only if their knowledge time is no later than that cutoff.
- Backfill leakage: a corrected product category may be useful for a new analysis but must not silently alter a historical certified finance snapshot.
- Quality rates must name expected population: accepted events / offered events differs from valid effects / accepted events.
- Unavailable evidence: no real source schemas, provider contracts, identity policy, retention schedule, settlement samples, or measured latency/cost was available.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location or basis | Version/date | Purpose | Accessibility | Limitation |
| --- | --- | --- | --- | --- | --- |
| SRC-01 | This illustrative case ledger in Section 6 | 2026-11-10 | arithmetic and replay oracle | public in course | constructed; not operational evidence |
| SRC-02 | Local Data Contract Checklist | draft accessed 2026-11-10 | contract and gate vocabulary | local draft | not an external authority |
| CFG-01 | proposed `commerce-v1` schema contract | not created | field, key, and operation rules | OPEN | must be authored and approved |
| RUN-01 | manual arithmetic smoke recorded below | 2026-11-10 | net-sales/replay check | local authoring check | not a load or integration test |
| DEC-01 | Section 3 decision contract | 2026-11-10 | selection boundary | public in course | no funding decision occurred |

- A reproducible implementation would pin schema contract version, canonicalization algorithm revision, table snapshot ID, source manifest IDs, close cutoff, reconciliation query revision, and identity-policy version.
- A run must record whether each source partition was complete, late, absent, or overridden; “latest” is not reproducible.
- Random seeds are irrelevant to deterministic ledger arithmetic but clock, timezone, decimal scale, and collation settings are material.
- The method-selection ledger is intentionally a proposal. No benchmark, deployment, price quote, or vendor test was run.

## 6. Analysis Walkthrough

The decisive result is conceptual and arithmetic rather than a deployment result: the same customer-facing order can yield a provisional $90 net-sales effect at close and a $70 restated effect once a $20 late refund becomes known. A replay must not make it $160. The worked ledger below makes the time boundary and identity rule auditable.

### 6.1 Worked transaction ledger — illustrative

| Row | knowledge_time UTC | event_time UTC | event ID | Business key / operation | Amount USD | Admission effect at 05:00 cutoff |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-11-03 23:55 | 2026-11-03 23:55 | ord-100-v1 | order O100 CREATED | 0.00 | context only |
| 2 | 2026-11-03 23:56 | 2026-11-03 23:56 | cap-p1 | capture P1 for O100 | 100.00 | +100.00 |
| 3 | 2026-11-03 23:57 | 2026-11-03 23:57 | disc-o100 | order discount allocation | -10.00 | -10.00 |
| 4 | 2026-11-04 00:10 | 2026-11-04 00:10 | cap-p2 | capture P2 for O101 | 50.00 | +50.00 |
| 5 | 2026-11-04 00:12 | 2026-11-04 00:12 | ref-r1 | refund R1 for O101 | 20.00 | -20.00 |
| 6 | 2026-11-04 01:00 | 2026-11-04 01:00 | cap-p1 | replay of capture P1, same hash | 100.00 | 0.00 duplicate |
| 7 | 2026-11-04 04:40 | 2026-11-03 23:58 | ref-r2 | late-known refund R2 for O100 | 20.00 | -20.00 |
| 8 | 2026-11-04 05:10 | 2026-11-03 23:58 | ref-r3 | refund R3 for O100 after cutoff | 20.00 | not in 05:00 snapshot |

For the 05:00 UTC close snapshot, eligible unique effects known by cutoff are `+100 -10 +50 -20 -20 = +100 USD`. The replay has zero incremental effect. If the business date is based on event time, O100’s late-known refund belongs to 2026-11-03’s restatable business day; if it becomes known after 05:00 it is excluded from that certified-as-of snapshot and appears in the later restatement.

The cohort calculation uses first paid capture date, not order-created date. Cohort 2026-11-03 contains O100: `100 - 10 - 20 = 70 USD` when R2 is known. Cohort 2026-11-04 contains O101: `50 - 20 = 30 USD`. Combined illustrative net sales across both cohorts is `70 + 30 = 100 USD`. The $20 R3 is a counterexample: if known later it changes a later restatement, not the already named snapshot.

```text
FORMULA [L-01]: NetSales(c, K) = sum(e.amount for admissible unique effects e
                         where e.business_date = c and e.knowledge_time <= K)
WHERE: c is event-time business date [UTC day]; K is the snapshot knowledge cutoff [UTC timestamp];
       admissible excludes non-captured authorizations and unresolved conflicts.
CONSTRAINT: each financial effect has a stable identity; signed amounts share currency and tax basis;
            duplicate identity with equal canonical payload contributes once.
NUMERICAL EXAMPLE: at K=2026-11-04T05:00Z, effects = +100 -10 +50 -20 -20 = +100 USD.
INTERPRETATION: this is a named as-of view, not a claim that later knowledge did not exist.

FORMULA [L-02]: Incremental(e) = 0 if key(e) was previously accepted with same hash; otherwise e.amount
WHERE: key(e)=(provider, provider_txn_id, effect_type); hash is canonical payload hash.
CONSTRAINT: equal key and different hash is a conflict, not a duplicate; it must not be summed.
NUMERICAL EXAMPLE: replay cap-p1 has same key and hash, so Incremental=0 USD, not +100 USD.
INTERPRETATION: idempotency is an accounting rule enforced after receipt, not a transport promise.

FORMULA [L-03]: Storage_month = events/day * bytes/event * days
WHERE: events/day is accepted raw event count; bytes/event is compressed logical payload bytes.
CONSTRAINT: payload only; replication, manifests, indexes, deletes, and query scans excluded.
NUMERICAL EXAMPLE: 2,000,000 * 1,024 * 30 = 61,440,000,000 bytes = 57.2205 GiB.
INTERPRETATION: a 3 TiB payload budget has 53.6877 months of payload-only headroom; overhead and retention must still be measured.
```

Since `3 TiB = 3,298,534,883,328 bytes`, payload-only headroom is `3,298,534,883,328 / 61,440,000,000 = 53.6877` months before replication, metadata, and higher payloads. This is an estimate, not a storage quote.

### 6.2 Late-arrival and duplicate counterexamples

- **Late-refund counterexample:** R3 has event time before close but knowledge time after close. Including it in the 05:00 snapshot uses information unavailable then; excluding it forever is also wrong. The recovery is a later snapshot plus restatement record linking old and new versions.
- **Replay counterexample:** cap-p1 arrives twice. Summing transport messages produces $200 capture for a $100 payment. Deduplication by message offset fails after a producer replay; deduplication must use the provider transaction identity and canonical payload.
- **Conflict counterexample:** a second message uses cap-p1 but says $95. Treating it as either a duplicate or new capture destroys evidence. Quarantine both candidate interpretations, alert Payment Ops, and withhold the affected reconciliation scope.
- **Delete counterexample:** a customer requests deletion after buying. Deleting the raw payment effect would break finance evidence; retaining customer identity in marketing would violate the serving purpose. Split restricted audit retention from identity-serving removal subject to approved policy.
- **Schema counterexample:** `amount` changes from decimal dollars to cents while retaining numeric type. Schema parsing passes, semantic contract fails; block effects until unit/version is explicitly migrated.

### 6.3 Close procedure and recovery trace

| Step | State transition | Invariant / check | Failure boundary |
| --- | --- | --- | --- |
| 1 | freeze manifest list at K | every listed partition has checksum and source coverage state | missing manifest means no certified close |
| 2 | canonicalize and validate | schema/version/units known | malformed events remain recoverable in quarantine |
| 3 | deduplicate | one accepted effect per key/hash | same key/different hash is conflict |
| 4 | derive effects | effect references source event and business object | pending authorization cannot enter net sales |
| 5 | reconcile | key and value comparison to settlement population | unmatched effects prevent certification |
| 6 | materialize snapshot | snapshot id stores K, manifests, query revision | “latest” cannot replace named snapshot |
| 7 | publish or refuse | finance table marked certified/provisional/withheld | refusal must be consumable by dashboard |
| 8 | late correction | append correction; recompute later snapshot | never mutate prior certification silently |

Recovery starts from the last certified snapshot and immutable source manifests, not from a current mutable table. Re-read affected partitions, apply the same canonicalization revision, produce a candidate snapshot, compare it to the prior snapshot by keys and signed amounts, and require Controller approval if the close policy says a restatement is material. The system must expose both the corrected value and why it changed.

### 6.4 Sensitivity and decision boundary

| Stress | Original | Alternative | Arithmetic / impact | Decision consequence |
| --- | --- | --- | --- | --- |
| Duplicate replay | cap-p1 deduplicated | cap-p1 counted twice | net sales rises from $100 to $200 at 05:00 under simplistic sum | identity contract is non-negotiable |
| Late R3 | known after cutoff | known at 04:59 | 05:00 net sales becomes $80 instead of $100 | knowledge cutoff controls snapshot, not event date alone |
| Payload size | 1 KiB/event | 4 KiB/event | 57.2205 GiB/month becomes 228.882 GiB/month payload-only | measure payload before cost choice |
| Event rate | 2M/day | 10M/day | payload-only monthly storage becomes 286.102 GiB at 1 KiB | validate throughput and compaction before selection |
| Close scope | daily certified close | only weekly management report | lower value from hourly mutation path | batch/audit-table alternative may win |

The recommendation reverses toward a simple batch warehouse if the last row is the actual requirement and finance can accept a documented weekly correction process. It reverses away from a lakehouse-like design if source keys are unstable: adding storage cannot recover the ability to distinguish a replay from a second payment.

## 7. The Finding

Propose a limited, contract-gated lakehouse ledger only for the named commerce reconciliation and governed-serving scope; do not approve a generic enterprise platform. Compared with overwrite-oriented nightly extracts, its expected benefit is reproducible as-of snapshots, explicit late-data restatement, and one deduplication boundary, not a claimed speed or cost improvement. The illustrative ledger proves that a $20 late refund and a $100 replay require distinct treatment, while the sizing arithmetic shows payload storage is only a preliminary input. The proposal is conditional on verification of source identities, close policy, identity/deletion controls, operational cost, and recovery acceptance tests; no deployment or outcome has been measured.

## 8. System/Solution Design

### 8.1 Input-to-output architecture

```text
Sources: order CDC | payment events | refund events | product CDC | identity/consent/deletion
      │                 │                 │                  │                 │
      └──── contract validation + immutable receipt ledger (restricted raw zone) ────┘
                                      │
                         canonicalization + key/hash deduplication
                                      │
               versioned order state | signed payment/refund effect ledger
                                      │
          reconciliation + watermark/manifest gate + policy evaluation
                                      │
      finance certified snapshots | operational provisional views | purpose-limited activation
                                      │
       Finance Controller            Merchandising                CRM / Privacy policy
Failure path: quarantine → reason code → owner queue → replay from immutable manifests → restatement
```

### 8.2 Tables, contracts, and ownership

| Layer | Contract | Owner | Read contract | Write / failure behavior |
| --- | --- | --- | --- | --- |
| Raw receipt | one accepted transport record plus receipt metadata | Platform | restricted forensic access | append only; invalid bytes retained with reason |
| Canonical event | one decoded event with source key/hash/operation | Platform + producer | controlled engineering read | conflicting duplicate quarantined |
| Order state | one `(order_id, source_version)` | Order domain | as-of state with interval semantics | CDC delete becomes tombstone/version |
| Effect ledger | one eligible capture/discount/refund effect | Payment Ops + Platform | signed amount and source lineage | unmatched effects excluded from certified output |
| Identity bridge | one subject-to-identifier interval and purpose | Identity/Privacy | identity service only | deletion revokes serving eligibility |
| Gold finance | one aggregate + snapshot id | Controller | certification state mandatory in every query | withheld on reconciliation breach |
| Gold activation | one allowed subject-purpose snapshot | CRM + Privacy | active consent/purpose required | delete/withdraw suppresses immediately per policy |

- **Serving contract—finance:** every value has `snapshot_id`, `knowledge_cutoff`, `currency_basis`, reconciliation status, and restatement-of pointer. Consumers must refuse a missing status.
- **Serving contract—merchandising:** item revenue uses an explicit allocation rule; it cannot infer product revenue from order totals where discounts and refunds are unallocated.
- **Serving contract—CRM:** subject token and eligibility only; email, address, and raw device ID are not general lakehouse dimensions.
- **Serving contract—data science:** experimental features use an as-of snapshot; training or evaluation cannot join a later identity resolution onto an earlier decision without labeling leakage.
- **Schema evolution:** additive nullable field may be compatible only after contract review; renamed fields, changed units, changed enum semantics, key changes, or changed deletion meaning are breaking.
- **Dual read/write:** for a breaking semantic change, publish v1 and v2 with a reconciliation comparator, migrate named consumers, then retire v1 with evidence.
- **CDC deletes:** retain an operation/tombstone in restricted history; current-state tables remove eligibility according to purpose and retention rules, not by an unlogged overwrite.
- **Late arrivals:** watermark policy defines provisional versus certified; arrival time does not rewrite event time.
- **Compaction:** may rewrite physical files but must preserve logical snapshot identity and source lineage.
- **Backfill:** has its own run ID and cannot silently overwrite a certified snapshot.

### 8.3 Security, privacy, and governance boundaries

- Classify fields as direct identifier, pseudonymous identifier, transaction identifier, sensitive attribute, and non-PII operational metadata; classification is a contract field, not a column-name guess.
- Separate access by purpose: reconciliation staff may need payment references; marketing may need active audience tokens; neither implies raw identity access.
- Use short-lived credentials and audit access to restricted tables; exact products and policy configuration are local choices.
- A customer merge is not retroactive permission to market to both profiles. Identity linkage must carry an effective interval, confidence/source, and purpose rule.
- Deletion requests require a traceable request ID, effective timestamp, affected serving datasets, completion state, and exception policy. The policy basis is OPEN.
- Data retention and legal hold conflicts require approved policy and counsel; this case does not resolve them.
- Lineage must connect gold snapshot → effect rows → canonical event → raw receipt and include transformation revision.
- Governance review must test a denied read and a deleted subject’s absence from activation output, not just inspect metadata.

### 8.4 Capacity, cost, and operational readiness

The payload formula in Section 6 estimates 57.2205 GiB/month at 2,000,000 events/day and 1,024 bytes/event. At 3 replicas it becomes 171.6615 GiB/month before table metadata, compaction overlap, quarantine, indexes, query scans, and retention. A cost decision therefore needs measured storage class price, compute scan profile, egress, orchestration, on-call labor, and failed-job replay volume; none is supplied here.

For a recovery estimate, suppose a 30-minute outage queues `2,000,000 / 86,400 = 23.1481 events/s`, producing `23.1481 * 1,800 = 41,666.7` events; use 41,667 whole events. At a recovered processing rate of 100 events/s while arrivals continue at 23.1481 events/s, spare rate is 76.8519 events/s and approximate drain time is `41,667 / 76.8519 = 542.17 s`, or 9.04 minutes. This fluid estimate excludes retries, source throttling, and skew; if recovered service is at or below arrivals, the queue does not drain.

- Readiness gate: restore a named snapshot from immutable manifests in an isolated environment and compare effect keys/amounts to the expected oracle.
- Readiness gate: replay a duplicated provider event and show zero additional signed effect.
- Readiness gate: inject a same-key/different-payload conflict and show quarantine plus certification refusal for affected scope.
- Readiness gate: inject a late refund and show old snapshot unchanged, later snapshot/restatement linked.
- Readiness gate: issue a deletion request in a test tenant and show activation suppression with audit evidence.
- Readiness gate: measure actual file size, scan cost, compacted-file overlap, and worker utilization on representative workloads.
- Runbook owner: Data Platform on-call owns pipeline recovery; Controller owns close certification; Payment Ops owns unmatched settlement evidence; Privacy owns deletion policy decision.

### 8.5 Field-level and operational control register

#### Control: `event_id`

**Meaning:** Producer-scoped immutable event identity.

**Why it matters:** Required before deduplication; never generated from arrival offset.

**Boundary / action:** Quarantine missing or reused-with-different-payload values.

#### Control: `source_system`

**Meaning:** Stable producer namespace.

**Why it matters:** Included in uniqueness because event IDs can collide across systems.

**Boundary / action:** Block unknown producer namespace at receipt.

#### Control: `schema_version`

**Meaning:** Producer contract revision.

**Why it matters:** Selects parser and semantic validation rules.

**Boundary / action:** Reject undeclared versions; do not guess field meanings.

#### Control: `event_time`

**Meaning:** When the business action occurred in UTC.

**Why it matters:** Assigns business date and event ordering.

**Boundary / action:** Keep original value and reason code if producer clock is invalid.

#### Control: `knowledge_time`

**Meaning:** First time this platform accepted the decoded event.

**Why it matters:** Bounds an as-of report.

**Boundary / action:** Never overwrite it on replay or compaction.

#### Control: `ingest_time`

**Meaning:** When transport persisted bytes.

**Why it matters:** Measures operational delay and aids diagnosis.

**Boundary / action:** Do not use it as a finance business clock by default.

#### Control: `operation`

**Meaning:** INSERT, UPDATE, DELETE, or SNAPSHOT marker.

**Why it matters:** Defines state reconstruction.

**Boundary / action:** Unknown operation is a contract failure, not an UPDATE.

#### Control: `source_sequence`

**Meaning:** Monotonic source order when supplied.

**Why it matters:** Resolves same-business-key updates within one source.

**Boundary / action:** Detect gaps; do not invent missing versions.

#### Control: `payload_hash`

**Meaning:** Canonical decoded payload digest.

**Why it matters:** Distinguishes exact replay from identity conflict.

**Boundary / action:** Version canonicalization algorithm explicitly.

#### Control: `order_id`

**Meaning:** Business order identifier.

**Why it matters:** Links order context to effects; not an effect key.

**Boundary / action:** Quarantine orphan effect unless contract permits delayed link.

#### Control: `order_line_id`

**Meaning:** Stable sellable line identity.

**Why it matters:** Allows allocated discount/refund analysis.

**Boundary / action:** Withhold line metric when refund allocation is absent.

#### Control: `provider`

**Meaning:** Payment rail namespace.

**Why it matters:** Prevents transaction-ID collision across rails.

**Boundary / action:** Make provider code an enumerated contract value.

#### Control: `provider_txn_id`

**Meaning:** Payment-provider transaction identity.

**Why it matters:** Forms the financial effect key.

**Boundary / action:** Escalate missing/changed identity to Payment Ops.

#### Control: `effect_type`

**Meaning:** Capture, refund, discount, chargeback, or adjustment.

**Why it matters:** Determines sign and reconciliation population.

**Boundary / action:** Block unrecognized effect type from net-sales publication.

#### Control: `amount`

**Meaning:** Signed decimal amount in declared basis.

**Why it matters:** Feeds monetary aggregation.

**Boundary / action:** Require scale, currency, tax, and rounding semantics.

#### Control: `currency_code`

**Meaning:** ISO-like declared currency code.

**Why it matters:** Prevents addition across currencies.

**Boundary / action:** Partition/report separately until approved FX conversion.

#### Control: `tax_basis`

**Meaning:** Whether amount includes tax, shipping, fees, or discounts.

**Why it matters:** Prevents cross-table semantic join error.

**Boundary / action:** Version explicitly when basis changes.

#### Control: `allocation_id`

**Meaning:** Reference to an allocation rule/result.

**Why it matters:** Explains partial refund or discount distribution.

**Boundary / action:** Withhold item aggregation on absent/invalid allocation.

#### Control: `customer_subject_id`

**Meaning:** Pseudonymous internal subject.

**Why it matters:** Permits governed eligible joins.

**Boundary / action:** Never treat as proof of legal identity or consent.

#### Control: `identity_link_version`

**Meaning:** Temporal identity bridge revision.

**Why it matters:** Supports as-of cohort reconstruction.

**Boundary / action:** Use frozen version for historical reports.

#### Control: `purpose`

**Meaning:** Declared permitted processing purpose.

**Why it matters:** Gates activation output.

**Boundary / action:** Deny serving when purpose is unknown or inactive.

#### Control: `consent_effective_at`

**Meaning:** When consent state became effective.

**Why it matters:** Prevents use outside temporal permission.

**Boundary / action:** Use policy-defined clock and correction handling.

#### Control: `deletion_request_id`

**Meaning:** Authoritative removal request trace.

**Why it matters:** Connects serving suppression to audit.

**Boundary / action:** Do not expose it in broad analytics tables.

#### Control: `manifest_id`

**Meaning:** Immutable input partition list identity.

**Why it matters:** Lets snapshot be reproduced.

**Boundary / action:** Treat missing manifest as no-close condition.

#### Control: `snapshot_id`

**Meaning:** Published table version identity.

**Why it matters:** Makes consumer read contract explicit.

**Boundary / action:** Do not label mutable “latest” as certified snapshot.

#### Control: `reconciliation_status`

**Meaning:** Matched, excepted, withheld, or provisional.

**Why it matters:** Tells consumer whether number is decision-eligible.

**Boundary / action:** Make it mandatory in finance serving result.

#### Control: `restatement_of`

**Meaning:** Prior snapshot superseded by correction.

**Why it matters:** Preserves historical chain.

**Boundary / action:** Never silently revise certified aggregate.

#### Control: `quality_reason_code`

**Meaning:** Machine-readable exclusion/failure reason.

**Why it matters:** Routes owner and allows denominator accounting.

**Boundary / action:** Avoid free-text-only diagnosis.

#### Control: `producer_owner`

**Meaning:** Named accountable source role.

**Why it matters:** Routes schema and business meaning issue.

**Boundary / action:** Do not assign Platform as semantic owner by default.

#### Control: `consumer_contract`

**Meaning:** Named decision-specific agreement.

**Why it matters:** Prevents table-level generic promises.

**Boundary / action:** Reject anonymous dashboard dependency.

#### Control: `retention_class`

**Meaning:** Policy-defined retention category.

**Why it matters:** Separates audit and serving lifecycle.

**Boundary / action:** Keep OPEN until policy is approved.

#### Control: `classification`

**Meaning:** PII/sensitivity label.

**Why it matters:** Drives access and projection.

**Boundary / action:** Review derived fields as well as raw fields.

#### Control: `zone`

**Meaning:** Raw, restricted curated, or consumer serving boundary.

**Why it matters:** Makes control location explicit.

**Boundary / action:** Do not infer zone from table name.

#### Control: `watermark`

**Meaning:** Declared completeness/late-data boundary.

**Why it matters:** Distinguishes provisional from certified.

**Boundary / action:** Publish its value with every snapshot.

#### Control: `expected_count`

**Meaning:** Expected source population count.

**Why it matters:** Provides completeness denominator.

**Boundary / action:** Do not substitute received count for expected count.

#### Control: `accepted_count`

**Meaning:** Valid received records after identity handling.

**Why it matters:** Measures delivery/validation.

**Boundary / action:** Report alongside rejected and duplicate counts.

#### Control: `excluded_amount`

**Meaning:** Signed value withheld from serving.

**Why it matters:** Makes financial impact of errors visible.

**Boundary / action:** Do not bury it in an error log.

#### Control: `exception_owner`

**Meaning:** Role responsible for resolution.

**Why it matters:** Prevents unowned close breach.

**Boundary / action:** Escalate if owner lacks decision authority.

#### Control: `override_id`

**Meaning:** Time-bounded approved exception.

**Why it matters:** Makes manual release auditable.

**Boundary / action:** Never make override a permanent status.

#### Control: `canonicalization_revision`

**Meaning:** Parser/normalizer code identity.

**Why it matters:** Reproduces payload hash and effects.

**Boundary / action:** Pin during close and replay.

#### Control: `decimal_scale`

**Meaning:** Monetary precision contract.

**Why it matters:** Prevents rounding drift.

**Boundary / action:** Test rounding at line, order, and aggregate layers.

#### Control: `timezone`

**Meaning:** Timezone semantic for local sources.

**Why it matters:** Makes date boundary explicit.

**Boundary / action:** Convert to UTC with source-zone evidence.

#### Control: `channel`

**Meaning:** Sales channel taxonomy.

**Why it matters:** Supports channel aggregation.

**Boundary / action:** Version taxonomy and map unknowns visibly.

#### Control: `product_version`

**Meaning:** As-of product classification reference.

**Why it matters:** Avoids rewriting history with current catalog.

**Boundary / action:** Use a separate current merchandising view if needed.

#### Control: `close_calendar`

**Meaning:** Business close schedule/version.

**Why it matters:** Maps cutoffs to periods.

**Boundary / action:** Controller owns changes and exceptions.

#### Control: `settlement_batch_id`

**Meaning:** Provider settlement grouping identifier.

**Why it matters:** Defines reconciliation expected population.

**Boundary / action:** Record late or corrected settlement batches separately.

#### Control: `file_checksum`

**Meaning:** Received payload integrity digest.

**Why it matters:** Detects transfer corruption.

**Boundary / action:** A matching checksum does not prove semantic correctness.

#### Control: `checkpoint_id`

**Meaning:** Processor recovery state identifier.

**Why it matters:** Supports controlled replay.

**Boundary / action:** Never trust checkpoint without source-offset/manifest link.

#### Control: `dead_letter_reason`

**Meaning:** Why record did not enter canonical ledger.

**Why it matters:** Supports repair prioritization.

**Boundary / action:** Retain enough evidence to replay after fix.

#### Control: `contract_version`

**Meaning:** Named semantic contract version.

**Why it matters:** Binds fields to meanings and gate policies.

**Boundary / action:** Require consumer acknowledgement for breaking changes.

#### Control: `access_policy_version`

**Meaning:** Policy evaluation revision.

**Why it matters:** Makes serving authorization reproducible.

**Boundary / action:** Re-evaluate current serving eligibility after policy change.

#### Control: `test_case_id`

**Meaning:** Synthetic validation scenario identity.

**Why it matters:** Links release to acceptance evidence.

**Boundary / action:** Never label synthetic pass as production evidence.

#### Control: `run_id`

**Meaning:** Transformation/reconciliation execution identity.

**Why it matters:** Aids audit and recovery.

**Boundary / action:** Record configuration, inputs, and outcome.

#### Control: `query_revision`

**Meaning:** Version of certified aggregation logic.

**Why it matters:** Explains output changes due to code.

**Boundary / action:** Review with finance for semantic changes.

#### Control: `compaction_revision`

**Meaning:** Physical maintenance operation identity.

**Why it matters:** Allows lineage through file rewrite.

**Boundary / action:** Must preserve logical table snapshot semantics.

#### Control: `backfill_reason`

**Meaning:** Why historical data is being recomputed.

**Why it matters:** Separates repair from normal arrival.

**Boundary / action:** Requires snapshot/restatement impact review.

#### Control: `materiality_rule`

**Meaning:** Controller-approved restatement threshold/process.

**Why it matters:** Decides communication not arithmetic truth.

**Boundary / action:** Keep source/policy OPEN until approved.

#### Control: `incident_id`

**Meaning:** Reference to operational event.

**Why it matters:** Links exceptions to postmortem.

**Boundary / action:** Do not use it to suppress data-quality evidence.

#### Control: `lineage_edge`

**Meaning:** Declared input-to-output dependency.

**Why it matters:** Enables impact analysis.

**Boundary / action:** Test it against actual execution metadata.

#### Control: `dashboard_version`

**Meaning:** Consumer visualization revision.

**Why it matters:** Prevents UI changes masquerading as data change.

**Boundary / action:** Include in investigation when totals move.

#### Control: `fx_rate_version`

**Meaning:** Conversion reference when multi-currency is enabled.

**Why it matters:** Makes converted amount reproducible.

**Boundary / action:** Out of scope for worked USD ledger; add before expansion.

#### Control: `refund_reason`

**Meaning:** Business reason taxonomy.

**Why it matters:** Supports operational analysis.

**Boundary / action:** Not required for signed net-sales arithmetic.

#### Control: `chargeback_status`

**Meaning:** Dispute lifecycle state.

**Why it matters:** Prevents premature financial treatment.

**Boundary / action:** Add contract before chargeback inclusion.

#### Control: `fulfillment_state`

**Meaning:** Shipment/delivery context.

**Why it matters:** Supports margin/operational analysis.

**Boundary / action:** Do not assume shipment equals payment capture.

#### Control: `promotion_id`

**Meaning:** Offer identity.

**Why it matters:** Supports discount attribution.

**Boundary / action:** Requires allocation contract for revenue use.

#### Control: `source_clock_quality`

**Meaning:** Producer timestamp health flag.

**Why it matters:** Identifies unreliable event-time assignment.

**Boundary / action:** Route to source owner; do not silently substitute arrival time.

#### Control: `migration_window`

**Meaning:** Dates v1/v2 can coexist.

**Why it matters:** Bounds dual-publish debt.

**Boundary / action:** Block retirement until named consumers exit.

#### Control: `consumer_acknowledgement`

**Meaning:** Named consumer acceptance of contract.

**Why it matters:** Confirms semantic migration.

**Boundary / action:** Do not infer from query traffic.

#### Control: `release_decision`

**Meaning:** Certified, provisional, withheld, or restated.

**Why it matters:** Is the action output of the close gate.

**Boundary / action:** Store reason and authority with value.

#### Control: `audit_export_id`

**Meaning:** Controlled export trace.

**Why it matters:** Supports evidence without broad exposure.

**Boundary / action:** Review purpose and expiry before release.

#### Control: `replay_scope`

**Meaning:** Manifests/partitions selected for recovery.

**Why it matters:** Bounds recovery blast radius.

**Boundary / action:** Record before running, not after failure.

#### Control: `slo_measurement`

**Meaning:** Observed freshness/completeness measure.

**Why it matters:** Supports operational alerting.

**Boundary / action:** Does not prove financial correctness alone.

#### Control: `cost_measurement`

**Meaning:** Metered storage/scan/compute/labor observation.

**Why it matters:** Feeds architecture comparison.

**Boundary / action:** Separate estimate from invoice and allocation method.

#### Control: `deprecation_state`

**Meaning:** Active, dual, deprecated, retired contract status.

**Why it matters:** Prevents forever schemas.

**Boundary / action:** Retain evidence of last consumer exit.

#### Control: `open_evidence_item`

**Meaning:** Known unverified dependency.

**Why it matters:** Makes recommendation limits visible.

**Boundary / action:** Assign an owner and decision blocked by it.

## 9. Evaluation & Validation

| Metric / oracle | Numerator | Denominator / population | Window / comparator | Pass condition | Status |
| --- | --- | --- | --- | --- | --- |
| Effect duplicate rate | duplicate candidate keys | all received effect candidates | per source/day | reported; exact replay has zero incremental effect | proposed |
| Conflict rate | same-key/different-hash candidates | all received effect candidates | per source/day | all conflicts quarantined and surfaced | proposed |
| Reconciliation coverage | settlement-matched eligible amount and keys | settlement expected amount and keys | named close snapshot | threshold and exceptions approved by Controller | proposed |
| Late-arrival rate | events first known after their business close cutoff | all accepted events for business date | per source/day | used to size restatement workflow, not hidden | proposed |
| Delete propagation | eligible serving rows removed/revoked within policy target | test deletion requests | test window | 100% test requests evidenced | proposed |
| Snapshot reproducibility | identical key/value output from same manifests/config | all sampled snapshot outputs | replay test | exact match under documented decimal semantics | proposed |

Reconciliation must be both key-level and value-level. Value-only matching can hide one duplicated $100 capture offset by one missing $100 capture. Key-only matching can miss a unit or allocation error. The expected population is the provider settlement scope agreed for the close, not all messages received.

| Validation scenario | Oracle | Expected result | What it does not prove |
| --- | --- | --- | --- |
| Normal ledger | manual signed-sum oracle | Section 6 yields $100 at K=05:00 | provider integration correctness |
| Exact replay | same key and hash | incremental amount is $0 | all source replay variants |
| Conflicting replay | same key, changed amount | quarantine; no gold certification for scope | root cause resolution |
| Late refund | old event time/new knowledge time | later restatement, no prior mutation | policy-approved materiality |
| CDC delete | delete/tombstone current state | current entity absent or revoked according to contract | retention legality |
| Schema unit shift | dollars→cents semantic change | contract block before gold table | producer remediation speed |
| Identity withdrawal | consent revoked | activation exclusion; finance ledger unaffected | cross-system deletion completion |

No test has been executed beyond the arithmetic smoke documented in Section 11. Proposed acceptance tests must use synthetic, authorized data and locally approved privacy procedures.

## 10. Action Taken

No production action was taken: this is an illustrative proposed design. The bounded recommended action is to fund a discovery and acceptance-test phase, not an immediate platform migration.

| Sequence | Action | Decision gate | Accountable role |
| --- | --- | --- | --- |
| 1 | Inventory provider/order/refund identifiers, CDC operations, and source clocks. | Reject or redesign if no stable effect identity. | Data Platform Director |
| 2 | Agree close cutoff, restatement label, materiality process, and exception authority. | Do not build finance gold without Controller sign-off. | Finance Controller |
| 3 | Write v1 contracts for raw events, effects, finance snapshot, and activation eligibility. | Block unowned or semantically incomplete feeds. | Producer owners + Platform |
| 4 | Build narrow synthetic replay/recovery harness and named snapshot oracle. | Advance only if all acceptance scenarios pass. | Platform |
| 5 | Measure representative storage, scan, compaction, and operator workload. | Compare total cost to batch baseline. | Platform + Finance |
| 6 | Pilot one channel and one close cycle with parallel batch comparison. | Do not claim migration benefit before reconciliation. | Controller |
| 7 | Migrate named consumers one at a time; decommission overwrite path only after sign-off. | No dual-source ambiguity for certified reporting. | Consumer owner |

The broad “enterprise lakehouse” action is explicitly deferred. If discovery shows that the batch-plus-audit alternative meets the signed requirements at lower operational cost, choose it.

## 11. Outcome & Results

| Claim ID | Claim and scope | Evidence | Supports | Does not establish |
| --- | --- | --- | --- | --- |
| CLM-01 | Illustrative ledger produces $100 net sales at 05:00 UTC after deduplicating cap-p1 and including R2. | illustrative; Section 6 arithmetic smoke | the stated time/identity rule is internally consistent | real revenue or a deployed result |
| CLM-02 | A late R3 known after cutoff belongs in a later restatement, not the named earlier snapshot. | illustrative mechanism trace | need for knowledge-time snapshots | a specific accounting policy |
| CLM-03 | Payload-only storage estimate is 57.2205 GiB/month at stated workload. | estimated; Section 6 formula | need to measure overhead and price | actual platform cost or capacity |
| CLM-04 | No production outcome exists. | not-measured | proposal status | performance, savings, or compliance |

Outcome status is **not-measured** and implementation stage is **proposed**. The only completed authoring check is the manual arithmetic/state smoke in Section 11. Any statement that the design “improves” close, cost, quality, or privacy remains a hypothesis until a controlled local evaluation records its comparator, population, and exceptions.

## 12. Failure Modes & What Went Wrong

| Failure mode | Competing causes | Discriminating check | Bounded recovery | Verification |
| --- | --- | --- | --- | --- |
| Net sales doubled | provider replay; consumer re-query; join fan-out | compare event/effect key counts and payload hashes before aggregation | deduplicate effect identity; fix fan-out; rebuild named snapshot | signed effect count and total match oracle |
| Net sales fell | late refund; missing partition; rejected schema | inspect knowledge-time ledger and manifest completeness | restatement or hold close; repair feed | difference report explains every key |
| Order total differs from payment | partial capture; discount allocation; tax basis mismatch | compare obligation, capture, refund, and allocation tables separately | withhold line/report scope; document basis | reconciliation by key and signed value |
| Customer reappears after deletion | stale identity bridge; cache; purpose bypass | trace subject token through serving snapshots | revoke projection; invalidate cache; incident review | denied test read and absence scan |
| Historical dashboard changed | backfill overwrote “current” table; no snapshot pin | compare snapshot IDs and manifest list | restore named version; publish restatement | old snapshot still queryable |
| Schema passed but values wrong | unit shift; enum redefinition; timezone change | semantic relation tests and contract diff | block release; dual publish migration | new contract version reconciles |
| Close cannot certify | unmatched settlement; missing source watermark; conflict quarantine | exception ledger grouped by reason | publish withheld/provisional label; escalate owner | Controller approval or documented refusal |
| Recovery grows backlog | service rate below arrivals; skew; retry storm | measure input, success, retry, and dead-letter rates | throttle/repartition/add capacity only after root check | spare capacity positive and queue drains |
| Identity merge corrupts cohorts | guest-to-account fan-out; retroactive linkage | temporal bridge cardinality check | use effective interval and one policy-approved mapping | cohort recomputation matches frozen snapshot |
| PII leak through derived field | free text; untagged export; excessive role | sample scan and access-log review | revoke access; quarantine export; remediate contract | policy test denies unauthorized principal |

A generic “rerun the pipeline” is not recovery. It can repeat a duplicate or mutate an already certified history. Recovery must declare inputs, configuration, snapshot target, exception scope, approver, and verification query.

## 13. Transferable Lesson

- Choose storage architecture from a decision contract, not from a slogan: the decisive need here is reproducible accounting effects under late change.
- The unit that must be idempotent is the business effect, not merely a broker message or file.
- Event time, knowledge time, and processing time answer different questions. A close report needs an explicit knowledge boundary.
- An append-only ledger is not sufficient; semantic admission, conflict handling, and reconciliation turn bytes into a governed fact.
- A current customer dimension is unsafe for historical analysis unless identity and consent changes are time-bounded.
- Physical file compaction and logical history can coexist only if snapshot/lineage semantics remain stable.
- A finance table and a marketing table may derive from related data but need different retention, purpose, and correctness contracts.
- The warehouse baseline remains valuable. If its documented constraints satisfy the actual decision, adding a lakehouse control plane is unnecessary complexity.
- The strongest test is often a counterexample: a late refund, conflicting replay, or deletion request exposes assumptions ordinary happy-path data hides.
- Never claim exactly-once business truth without naming the identity, conflict rule, state store, and recovery oracle that make the claim meaningful.

## 14. Connections & Cross-References

Related local foundations: [Data Contract Checklist](../Cheat_Sheets/01_Data_Contract_Checklist.md), [Data Model Grain and Join Safety](../Cheat_Sheets/02_Data_Model_Grain_and_Join_Safety.md), and [Data Model Notation](../Cheat_Sheets/03_Data_Model_Notation.md). These drafts provide contract, grain, and cardinality vocabulary; this case applies them to a bounded commerce decision. Open questions include approved retention/deletion policy, payment-provider settlement semantics, actual workload distribution, and local tool selection.

### 14a. Related cases, theory, and cheat sheets

| Relationship | Actual target | Mechanism shared or changed | Boundary learned |
| --- | --- | --- | --- |
| uses-method | [Data Contract Checklist](../Cheat_Sheets/01_Data_Contract_Checklist.md) | contract gates, denominators, change control | local draft; not proof of a production policy |
| uses-method | [Data Model Grain and Join Safety](../Cheat_Sheets/02_Data_Model_Grain_and_Join_Safety.md) | stated grain and fan-out diagnostics | does not select a storage platform |
| uses-method | [Data Model Notation](../Cheat_Sheets/03_Data_Model_Notation.md) | keys, temporal relations, cardinality | notation is not an event ledger implementation |
| contrasts-with | planned `02_repair_broken_customer_360_pipeline.md` | identity may be a serving subject here, not a universal profile | no authored comparison exists yet |

### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data/test that would resolve it | Priority / trigger | Owner role |
| --- | --- | --- | --- | --- |
| Which source identifier is stable across payment correction and provider migration? | No source contracts are available. | Contract review plus synthetic correction/replay test. | Before effect ledger pilot. | Payment Ops |
| What close/restatement materiality rule is approved? | No Controller policy is provided. | Signed close-calendar and exception workflow. | Before finance consumer. | Finance Controller |
| What retention and deletion conflict policy applies? | No governing policy was accessible. | Privacy/legal review and test-tenant deletion trace. | Before identity serving. | Privacy lead |
| Does measured total cost beat batch plus audit append? | Workload and price are illustrative only. | Representative load, query, compaction, and labor measurement. | Funding decision. | Data Platform Director |

### 14.1 Interview pack — checked answers

#### Q01. Why is `order_id` not enough to deduplicate net sales?

**Checked answer:** One order can have multiple captures, refunds, and adjustments; deduplicating at order grain can discard legitimate partial effects or retain replayed ones. Use a stable provider/effect identity plus effect type and payload conflict rule.

**Follow-up:** What if the provider has no stable transaction ID?

**Follow-up answer:** Use a documented settlement key only if it is stable; otherwise retain the batch baseline or redesign the source contract.

**Plausible wrong method:** “Use Kafka exactly once.”

**Answer support:** Names effect identity, partial capture, and conflict quarantine.

**Red flag:** Claims transport semantics alone solve accounting.

**Scoring:** 2: identifies grain and recovery; 1: mentions duplicates; 0: says order ID.

#### Q02. What is the difference between event time and knowledge time in the late-refund example?

**Checked answer:** Event time says R3 economically occurred at 23:58; knowledge time says the platform learned it at 05:10. The 05:00 as-of snapshot may not use 05:10 knowledge without time travel.

**Follow-up:** Could a later report include R3?

**Follow-up answer:** Yes, as a new named snapshot/restatement with its own cutoff.

**Plausible wrong method:** “Use ingest time for everything.”

**Answer support:** Defines both clocks and applies cutoff.

**Red flag:** Treats late arrival as an event-time change.

**Scoring:** 2: correct distinction; 1: partially correct; 0: conflates times.

#### Q03. Compute the 05:00 net sales from the worked ledger.

**Checked answer:** Admissible unique signed effects are +100 -10 +50 -20 -20 = +100 USD. The repeated cap-p1 adds zero; R3 is after the knowledge cutoff.

**Follow-up:** What changes if R3 arrived at 04:59?

**Follow-up answer:** The snapshot becomes $80, assuming it passes all other admission checks.

**Plausible wrong method:** “$200 because two captures.”

**Answer support:** Shows terms and cutoff.

**Red flag:** Counts replay or omits refund.

**Scoring:** 2: $100 with reasoning; 1: right number/no reason; 0: wrong.

#### Q04. Why does a DELETE not mean erasing every historical row?

**Checked answer:** A deletion request has different obligations for identity serving and financial audit. Preserve policy-approved restricted evidence/tombstone while revoking current purpose-limited identity projections; exact policy requires approval.

**Follow-up:** Who decides retention conflict?

**Follow-up answer:** The approved privacy/legal policy owner, not a data engineer improvising.

**Plausible wrong method:** “Delete all files immediately.”

**Answer support:** Separates purpose and policy.

**Red flag:** Assumes one physical operation solves all obligations.

**Scoring:** 2: boundary stated; 1: mentions tombstone; 0: unsafe universal delete.

#### Q05. How would you detect a schema-compatible semantic break?

**Checked answer:** Test declared unit, currency, enum meaning, and arithmetic relations, not only type. A decimal dollars-to-cents change preserves parseability but violates a unit contract.

**Follow-up:** What release process follows?

**Follow-up answer:** Block, version semantics, dual-publish where needed, reconcile consumers, then retire old version.

**Plausible wrong method:** “The registry says compatible.”

**Answer support:** Names semantic predicate and migration.

**Red flag:** Equates type compatibility with meaning.

**Scoring:** 2: full; 1: detects only; 0: misses issue.

#### Q06. When does the simpler batch warehouse win?

**Checked answer:** When consumers only need documented periodic output, source changes are stable/append-only, audit history exists, correction tolerance is acceptable, and measured cost/operations favor batch.

**Follow-up:** Does batch prevent reconciliation?

**Follow-up answer:** No; it can reconcile, but must still retain keys, cutoff, and corrections.

**Plausible wrong method:** “Lakehouse is always modern.”

**Answer support:** Uses requirements and comparator.

**Red flag:** Universal recommendation.

**Scoring:** 2: conditions; 1: cost only; 0: slogan.

#### Q07. What is wrong with value-only settlement reconciliation?

**Checked answer:** Equal total values can hide a missing $100 capture offset by a duplicated $100 capture. Compare keys and signed values, then investigate unmatched scope.

**Follow-up:** What is wrong with key-only matching?

**Follow-up answer:** Amounts can differ due to units, currency, allocation, or rounding.

**Plausible wrong method:** “Totals matched, so close.”

**Answer support:** Explains both dimensions.

**Red flag:** Uses only one check.

**Scoring:** 2: both; 1: one; 0: accepts total.

#### Q08. A provider event reappears with the same key but amount $95 instead of $100. What do you do?

**Checked answer:** Quarantine it as an identity/payload conflict, retain evidence, alert Payment Ops, and withhold impacted certification. Neither summing nor silently choosing latest is defensible.

**Follow-up:** When can it be resolved?

**Follow-up answer:** After source/settlement evidence establishes an approved correction with lineage.

**Plausible wrong method:** “Last write wins.”

**Answer support:** Names conflict and hold.

**Red flag:** Overwrites evidence.

**Scoring:** 2: correct containment; 1: alert only; 0: overwrite.

#### Q09. How does a customer merge create cohort leakage?

**Checked answer:** A current merged identity can attach a later account relationship to an earlier purchase. Use effective intervals and an as-of snapshot; define cohort identity rule before aggregation.

**Follow-up:** What test catches it?

**Follow-up answer:** Compare temporal bridge cardinality and replay cohort from frozen mapping.

**Plausible wrong method:** “Use current master customer.”

**Answer support:** Uses temporal identity.

**Red flag:** Uses current view blindly.

**Scoring:** 2: complete; 1: mentions merge; 0: leakage ignored.

#### Q10. State the recovery invariant after an outage.

**Checked answer:** Reprocessing the same immutable manifests with the same contract/configuration produces the same accepted effect key/value set and named snapshot, except explicitly versioned corrections.

**Follow-up:** What if service rate equals arrival rate?

**Follow-up answer:** A positive backlog never drains under the fluid model; add surplus capacity or reduce arrivals.

**Plausible wrong method:** “Restart and see.”

**Answer support:** States inputs and oracle.

**Red flag:** No reproducibility criterion.

**Scoring:** 2: invariant + boundary; 1: replay; 0: restart.

#### Q11. How do you size the illustrative payload-only month?

**Checked answer:** `2,000,000 events/day × 1,024 bytes/event × 30 days = 61,440,000,000 bytes = 57.2205 GiB`. Replication, metadata, scans, and retention are excluded.

**Follow-up:** What if events average 4 KiB?

**Follow-up answer:** It is four times: 228.882 GiB/month payload-only.

**Plausible wrong method:** “Three TiB is enough.”

**Answer support:** Shows units and exclusions.

**Red flag:** Treats estimate as quote.

**Scoring:** 2: correct; 1: approximate; 0: no units.

#### Q12. What serving contract must a finance consumer receive?

**Checked answer:** At minimum snapshot ID, knowledge cutoff, certification/reconciliation status, currency/tax basis, lineage reference, and restatement pointer. A bare number is not auditable.

**Follow-up:** What should the dashboard do on withheld status?

**Follow-up answer:** Show refusal/provisional state according to approved design, never silently substitute latest.

**Plausible wrong method:** “A table called sales_daily.”

**Answer support:** Names control fields and consumer response.

**Red flag:** Assumes table name is contract.

**Scoring:** 2: full; 1: partial; 0: no status.

#### Q13. A leader asks for “real-time customer 360” using this ledger. What is your response?

**Checked answer:** Clarify decision, latency, purpose, identity confidence, and deletion obligations. This case’s finance ledger does not justify a universal customer profile; build a separate bounded contract if warranted.

**Follow-up:** Could the same storage be reused?

**Follow-up answer:** Possibly, under separate access and serving controls; reuse does not imply shared permissions or semantics.

**Plausible wrong method:** “Yes, add all columns.”

**Answer support:** Resists scope creep with questions.

**Red flag:** Conflates ledger and profile.

**Scoring:** 2: bounded; 1: asks latency; 0: promises universal 360.

### 14.2 Design-review evidence prompts

The following review prompts make the proposed design falsifiable before any migration. Each prompt asks for a named artifact, observation, or decision record rather than an assurance that the platform is “governed.”

#### Snapshot and accounting review

- Ask the Controller to name the exact business calendar revision that maps each event-time instant to its close date.
- Compare the immutable manifest count with the producer's expected partition count before labeling any close complete.
- Require an exception record to show both the withheld signed amount and the keys responsible for that amount.
- Sample a finance query that omits `snapshot_id`; the serving layer should reject it or make the result explicitly provisional.
- Recompute one aggregate from effect rows after reversing their presentation sort order; monetary addition must remain decimal-safe and order-independent.
- Trace a positive capture through receipt, canonical event, effect row, settlement comparison, and published aggregate without relying on a dashboard label.
- Confirm that an authorization, a capture, a refund, and a chargeback have separate admissibility predicates rather than a shared “payment” flag.
- Inspect the close gate when settlement arrives late but source events are complete; the status must distinguish settlement incompleteness from ingestion incompleteness.
- Test an empty business date and confirm that zero revenue is distinguishable from a missing date partition.
- Reconcile aggregate currency totals only inside one declared currency and tax basis; cross-currency sums require a separately versioned conversion rule.
- Record the policy decision for a settlement correction received after certification, including who can declare its restatement scope.
- Reject a proposed close query if it uses mutable `current` dimension attributes without an as-of dimension version.

#### Slowly changing dimension and conformance review

- Model product category as a type-2 interval when historical merchandising needs the category known at sale time.
- Close the old product interval at the new version's effective instant; overlapping intervals for one product key are a data-quality breach.
- Preserve the natural product identifier alongside the surrogate dimension key so a replay can rebuild a dimension assignment.
- Treat a late category correction as a separate knowledge-time revision instead of rewriting an already published finance snapshot invisibly.
- Require every fact-to-dimension lookup to declare whether it uses event time, knowledge time, or current-state time.
- Verify that one order line joins to exactly one admissible product version for its chosen temporal rule.
- Quarantine a product update whose effective interval ends before it begins; timestamp parsing alone cannot establish temporal validity.
- Use a conformed channel dimension only after web and mobile labels map through a controlled taxonomy version.
- Keep customer identity bridges outside the generic customer dimension because a merge can be many-to-one at one time and disallowed for another purpose.
- Require consent eligibility to have a start, end or open interval, policy version, and decision purpose before activation publication.
- Test a boundary instant where a consent withdrawal and campaign extraction share a timestamp; the governing clock and inclusivity rule must be explicit.
- Prevent an unknown dimension member from silently inheriting the most recent known attribute; route it to an auditable “unknown” member instead.

#### Ingestion, replay, and release review

- Store the byte receipt identifier separately from the business event identity because a retried delivery can have new transport metadata.
- Pin canonicalization revision during a replay so a later parser improvement does not alter the historical deduplication decision accidentally.
- Compare hashes only after documented normalization of field order, decimal representation, and permitted null defaults.
- Route same-key and changed-hash messages to a conflict ledger with both payload references; choosing the last arrival discards forensic evidence.
- Checkpoint progress is operational metadata, not proof that all business effects were accepted or reconciled.
- Make dead-letter re-entry create a new processing attempt linked to the original receipt rather than replacing the original failure reason.
- Require a backfill request to state the input manifest range, transform revision, business rationale, and expected snapshot impact.
- Exercise a partial partition replay and show that untouched partitions retain their prior snapshot lineage.
- Block a breaking contract release until named downstream consumers acknowledge the changed unit, key, or delete meaning.
- Dual-publish only long enough to compare concrete contract versions; a permanent dual path creates two unresolvable sources of truth.
- Make release status machine-readable so a dashboard cannot convert `withheld` into an empty but apparently valid chart.
- Retain raw malformed bytes only inside the restricted zone and attach a reason code that supports safe producer repair.

#### Cost, performance, and recovery review

- Measure file-count growth separately from payload bytes because small-file metadata and planning cost can dominate a low-volume workload.
- Include temporary compaction overlap in capacity planning; a table can require old and new files concurrently during rewrite.
- Separate storage retention cost from query scan cost because a cheap retained byte can still be expensive to scan repeatedly.
- Use observed percentile arrival rates, not only the daily mean, when setting stream capacity and watermark delay.
- Account for retry amplification in recovery estimates because one poison message can consume multiple processing attempts.
- Test partition skew using a high-volume product or promotion key; equal average rates do not guarantee equal worker load.
- Define a maximum replay scope before incident response so an operator does not recompute unrelated certified periods.
- Verify that a rollback restores both table metadata and the policy version that authorized the prior serving projection.
- Treat compaction completion as a physical-maintenance event; it must not create a new business close or alter accounting semantics.
- Estimate on-call effort from proposed alerts, exception queues, and ownership handoffs rather than assuming automation removes review work.
- Compare the bounded lakehouse proposal with batch-plus-audit using the same retention period, recovery objective, and governance obligations.
- Stop a recovery drill if the replay output cannot be compared to a declared key-and-amount oracle; faster unverified replay is not a success.

These prompts are proposed acceptance evidence, not assertions that any retailer has passed them. A design review should retain unanswered prompts as explicit adoption blockers or as approved, scoped exceptions.

## 15. Quick Reference Card

| Field | Card |
| --- | --- |
| Decision | Choose a narrow contract-gated ledger proposal only if source identity, close policy, privacy controls, and measured economics pass discovery. |
| Owner | Data Platform Director; Finance Controller certifies close; Privacy lead controls identity/deletion policy. |
| Grain | One immutable source event; one accepted accounting effect per provider/effect key and canonical payload. |
| Time | Use event time for business date and knowledge time for reproducible as-of snapshot; never collapse them. |
| Core formula | NetSales(c,K) is signed admissible unique effects for business date c known by cutoff K. |
| Worked oracle | At 05:00: +100 -10 +50 -20 -20 = $100; replay adds $0; post-cutoff R3 is later restatement. |
| Contracts | Schema/unit, effect identity, watermark, reconciliation, purpose/consent, delete propagation, snapshot ID. |
| Failure response | Quarantine same-key/different-hash; withhold affected certification; replay immutable manifests; link correction to restatement. |
| Baseline | Nightly warehouse/batch may win when periodic output and existing audit controls satisfy actual requirements. |
| Evidence/stage | Illustrative arithmetic and estimated sizing; proposed design; draft document; no production measurement. |
| Source limits | No source contracts, prices, logs, policy determinations, vendor benchmarks, or deployment evidence were available. |
| Review trigger | Any change to source identity, CDC/delete semantics, cutoff, consent policy, workload, table behavior, or cost. |

**Final check:** The case ends here. All recommendation language is conditional; all numerical workload and ledger values are illustrative except the arithmetic derived from those stated inputs.
