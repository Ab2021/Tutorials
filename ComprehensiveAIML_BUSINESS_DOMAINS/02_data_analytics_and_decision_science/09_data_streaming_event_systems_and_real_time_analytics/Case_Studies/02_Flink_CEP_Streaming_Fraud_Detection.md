---
case_id: "CASE-2026-STR002"
case_title: "Flink Event-Time Fraud Sequences with Versioned Broadcast Rules"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "streaming-systems"
domain: "Payment fraud and real-time event systems"
primary_pattern: "recoverable-stream-processing"
secondary_patterns: ["partition-aware-scaling", "versioned-rule-consistency"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Fraud platform lead"
source_index: "INDEX.md"
review_trigger: "Change in Flink version, ordering contract, rule publication, or alert-effect interface"
---

# Flink Event-Time Fraud Sequences with Versioned Broadcast Rules

## 1. The One-Line Framing

Given a stream of payment authorization events for one card token, propose review alerts when a small approved authorization precedes a large approved authorization within two minutes, for fraud operations, under event-time correctness, bounded review capacity, and recoverable delivery constraints, compared with an existing single-transaction amount rule.

- **Decision type:** continuous system operation supporting human investigation, not an automatic authorization decline.
- **Decision frequency:** each eligible event can complete a candidate sequence.
- **Primary condition:** more confirmed fraud cases recovered per day than the baseline while not exceeding the review and lateness guardrails in Section 9.
- **Primary pattern:** `recoverable-stream-processing`.
- **Secondary patterns:** `partition-aware-scaling` and `versioned-rule-consistency`.
- **Evidence:** illustrative design; all figures, names, inputs, and decisions below are constructed.
- **Stage:** proposed, not deployed or experimentally validated.
- **Source limitation:** the source inventory lists `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing/01_Real_Time_Fraud_Detection_CEP_Broadcast.md`; its contents were not supplied and are not reconstructed.
- **Framework references:** Apache documentation supports framework semantics, not the fictional workload or results.
- **Why not simply classify each payment?** The proposed signal is a relationship between two events; a single-event threshold cannot express their time-ordered association without retained state.
- **Key tradeoff:** event-time ordering and rule-consistent replay require waiting and operational controls that can reduce alert timeliness.
- **Thirty-second version:** preserve a reproducible event-time sequence, bind its rule version at its first event, and make downstream alert creation idempotent; test whether the incremental detections justify the review load.

## 2. Context & Background

### 2.1 Business workflow

- A hypothetical card issuer receives authorization decisions from a payment processor.
- The authorization decision already occurred before this stream is evaluated.
- An `APPROVED` event denotes that the authorization system approved the transaction, not that fraud was confirmed.
- The issuer wants reviewers to investigate suspicious sequences soon after the second authorization.
- A reviewer may contact the customer, inspect context, or close a case under existing policy.
- Separate systems own card blocking, customer communications, reimbursement, and dispute adjudication.
- This case does not grant the streaming job authority to perform those actions.
- A false positive consumes review time and may inconvenience a legitimate customer.
- A missed sequence may postpone investigation of a fraudulent card compromise.
- These losses are not assumed equal and require issuer-specific estimation.
- The hypothetical baseline queues approved authorizations of at least USD 500 as single-event alerts.
- Its weakness for this question is not an observed failure: it cannot use the preceding small authorization as a selection condition.
- The sequence candidate is compared against the baseline using the same eligible population and matured case definition.
- The operator may keep the baseline if the new condition misses too many useful cases.
- A sequence alert does not itself establish that the earlier transaction was a criminal test payment.

### 2.2 Stakeholders and timing

- Fraud operations owns review capacity and operational acceptance.
- The fraud strategy owner approves rule thresholds and their effective schedule.
- The stream platform owns checkpoint recovery and event-time infrastructure.
- The data producer owns transaction identity, timestamp quality, and corrections.
- The case-management team owns the durable unique alert effect.
- Security owns rule-publisher identity, access controls, and incident response.
- An independent evaluation team owns baseline comparisons and label maturity.
- Customer support owns customer-facing policies following an investigation.
- The initial decision horizon is a proposed four-week shadow evaluation, not a past project.
- Deployment geography and payment channels are intentionally unspecified; actual legal and retention requirements require local counsel and records.
- No prior attempt, incident, procurement decision, or named organization is implied.

### 2.3 Scenario scale

- The example assumes 1,000,000 eligible approved authorizations per day.
- That is a planning input, not a measured transaction volume.
- The simple daily average is 1,000,000 / 86,400 = 11.57 events per second.
- The independent peak assumption is 100 events per second for one hour.
- The peak must fit within the daily total; the example does not add a second day's traffic to the total.
- Account concentration, retry rate, and actual interarrival distributions are unknown.
- An assumed 200 reviewer slots per day is a hard planning constraint for the whole alert queue in the worked example.
- The fraud team must allocate that shared capacity across baseline and proposed alerts, rather than treating each as having its own 200 slots.
- Peak load is a workload hypothesis for later measurement, not a throughput benchmark.

## 3. The Business Question

- **Illustrative stakeholder question:** Should the issuer add an event-time sequence alert, and under what review-capacity and correctness conditions?
- **Decision owner:** fraud strategy lead, with review operations and platform sign-off.
- **Trigger:** proposed replacement or supplementation of an amount-only alert.
- **Deadline:** no real deadline is known; the proposed evaluation follows data and rule-contract approval.
- **Actions:** retain baseline, add a gated incremental queue, replace the baseline, or defer pending evidence.
- **Benefit:** timely discovery of additional independently confirmed fraud cases.
- **Downside:** false alerts, delayed review, customer harm, missed fraud, and incident recovery effort.
- **Objective:** maximize incremental confirmed fraud cases reviewed within the staffing limit over a defined mature cohort.
- **Constraint:** total delivered cases, not just false positives, must fit the available daily reviewer slots.
- **Scope:** approved USD card authorization events for the illustrative single issuer and its approved channels.
- **Exclusions:** declined transactions, non-USD currency conversion, chargeback prediction, autonomous blocking, and cross-card network detection.
- **Proxy conflict:** high alert precision can coexist with low coverage if the rule selects only easy cases.
- **Guardrail conflict:** a higher detected-fraud count can consume all capacity, delaying higher-value baseline cases.
- **No-change option:** continue the baseline while capturing equivalent events for offline assessment.
- **Decision record:** no approval is recorded; Section 10 describes a proposed evaluation only.

## 4. Assumptions & Constraints

### 4.1 Assumptions ledger

- **A-01 — Unique identities:** each authorization has an immutable issuer-scoped event ID.
- Type: data contract; status: unverified.
- Check: reconcile producer IDs against source records, retries, and correction events.
- If false: matches or alerts may be duplicated or distinct transactions collapsed.
- Fallback: obtain a stable producer-assigned logical identity before enabling effects.
- Owner and trigger: producer lead; recheck on schema or source migration.
- **A-02 — Timestamps:** `event_time` is a UTC millisecond occurrence timestamp with known clock quality.
- Type: data; status: unverified.
- Check: compare against processor receipt time, clock telemetry, and correction distributions.
- If false: order and two-minute boundaries can be wrong.
- Fallback: quarantine invalid timestamps and report excluded population.
- Owner and trigger: data lead; recheck when producer clocks or timestamp semantics change.
- **A-03 — Completeness:** the configured 10-second out-of-orderness bound makes the late fraction acceptable.
- Type: workload hypothesis; status: unverified.
- Check: measure lateness by source partition and time of day.
- If false: live output omits more qualifying sequences than the tolerance permits.
- Fallback: extend the watermark delay if latency permits, or defer rollout.
- Owner and trigger: platform lead; recheck after a late-data alert or traffic shift.
- **A-04 — Rule finality:** the publisher can commit a monotonically advancing rule-coverage frontier.
- Type: control-plane contract; status: unverified.
- Check: test gaps, replay, publisher failover, and prohibited retroactive updates.
- If false: different executions can select different rule versions for the same event.
- Fallback: hold decision processing and page the rule owner; do not silently use a default.
- Owner and trigger: strategy engineering; recheck at every publisher change.
- **A-05 — Review staffing:** 200 total alerts per day are available for this population.
- Type: business capacity assumption; status: unverified.
- Check: measure staffed shifts, actual handling time, and competing queue demand.
- If false: precision alone cannot guarantee operational viability.
- Fallback: select a smaller queue using an approved ranking and retain baseline coverage measurement.
- Owner and trigger: operations manager; recheck after scheduling changes.
- **A-06 — Labels:** case adjudication can be linked to both alert arms after a 90-day maturation period.
- Type: evaluation; status: unverified.
- Check: audit linkage, unresolved disputes, false-negative sampling, and customer reporting delays.
- If false: precision and incremental coverage may be biased.
- Fallback: estimate bounds and extend observation; do not call unresolved cases legitimate.
- Owner and trigger: evaluation lead; recheck when dispute policy changes.
- **A-07 — Alert store:** the case service can enforce an atomic unique alert key with payload conflict detection.
- Type: dependency; status: unverified.
- Check: concurrent insert, timeout-after-commit, restore, and duplicate-delivery drills.
- If false: checkpoint recovery cannot protect external effects.
- Fallback: do not enable the consumer until the service exposes an equivalent transactional contract.
- Owner and trigger: case platform; recheck after storage or API changes.
- **A-08 — Rule size:** all active and retained rule versions fit in broadcast state at every parallel task.
- Type: infrastructure; status: unverified.
- Check: measure serialized state and checkpoint/restore cost under maximum policy count.
- If false: broadcast replication threatens availability.
- Fallback: reduce the policy universe or redesign the distribution mechanism.
- Owner and trigger: platform lead; recheck at rule-count growth.

### 4.2 Constraints and invalidators

- **C-01:** No automated decline or customer contact is authorized by this design.
- Source: scenario boundary, not a claim about law or an existing policy.
- Owner: fraud operations; fallback: human review only.
- **C-02:** Review load cannot exceed allocated daily capacity.
- Source: scenario assumption; owner: operations; fallback: hold or prioritize under an approved policy.
- **C-03:** No transaction or secret is copied into unrestricted logs.
- Source: proposed security control; owner: security; fallback: access-controlled redacted audit.
- **C-04:** A late or correction replay must not silently create a second case or erase a prior decision.
- Source: proposed effect contract; owner: case-management team; fallback: reconciliation and manual adjudication.
- **C-05:** Rule selection must not depend on arrival race between a control message and a transaction.
- Source: chosen correctness contract; owner: fraud platform; fallback: stop output when rule finality is unknown.
- The most decision-reversing assumptions are A-03, A-04, and A-05.
- If lateness is high, event-time accuracy may conflict with the desired response time.
- If rule finality cannot be provided, deterministic historical rule selection is unavailable without a different architecture.
- If reviewer capacity is too small, additional alerts can worsen overall fraud handling even when many are correct.
- No jurisdiction-specific legal conclusion is made here.

## 5. Data, Tools & Method

### 5a. Data Specification

- **Event grain:** one immutable logical authorization event, not one customer or one alert.
- **Entity key:** `(issuer_id, card_token)`; card tokens are pseudonymous and issuer-scoped.
- **Partitioning:** events sharing this key must reach the same keyed state owner.
- **ID:** `event_id` is unique within the issuer and stable across delivery retries.
- **Event time:** `event_time_ms` is UTC occurrence time in integer milliseconds.
- **Ingestion time:** `ingested_at_ms` is independently recorded for latency analysis, never used to redefine event time.
- **Decision:** only `APPROVED` authorizations participate in the sequence.
- **Amount:** integer USD cents, with nonnegative amounts and explicit currency.
- **Source position:** topic, partition, offset, and producer schema version are audit metadata, not business identity.
- **Correction link:** `supersedes_event_id` identifies a later authoritative correction; corrections are not silently treated as new purchases.
- **Minimum schema:** `issuer_id`, `card_token`, `event_id`, `event_time_ms`, `amount_cents`, `currency`, `decision`, `schema_version`.
- **Input authorization:** only authenticated producer principals may append to the input topic.
- **Exclusions:** missing token, malformed ID, invalid timestamp, unsupported version, non-USD, and non-approved events are counted by reason.
- **Quarantine:** malformed records go to a restricted, durable repair stream, not to the normal pattern matcher.
- **Population denominator:** all valid, eligible authorizations in the declared issuer/channel/day, including those with no resulting alert.
- **Label grain:** confirmed fraud case at an agreed case identity, not the raw number of transactions.
- **Label source:** proposed adjudication and dispute records, reconciled by an authorized evaluation team.
- **Label maturity:** wait 90 days after the event cohort ends, then report unresolved and censored records separately.
- **Selection bias:** investigation only for alerts creates missing labels among non-alerts; sample and adjudicate non-alerts independently.
- **Join:** deterministic issuer-scoped transaction-to-case link with one-to-many case mapping explicitly resolved.
- **Leakage control:** label and post-authorization actions are never supplied to the sequence decision.
- **Data split:** use earlier dates to set thresholds and a later untouched time cohort for evaluation.
- **Entity dependence:** cards can recur across days; uncertainty calculations cluster by card and account for temporal blocks.
- **Workload trace:** a future authorized capture must include arrival order, event timestamp, source position, and rule log.
- **Snapshot:** no actual dataset, trace, state snapshot, or restricted artifact is available for this exercise.
- **Correctness oracle:** an independently implemented bounded batch sort and match over the same accepted events and rule log.
- **Oracle limitation:** compare live results only within the declared on-time population; late-event repair is a separate reconciliation output.

### 5b. Tools & Stack

- **Runtime proposal:** Apache Flink 1.20.3 DataStream API with the matching `flink-cep` dependency.
- The version is chosen for a concrete reading of the cited documentation, not as a current-version recommendation.
- Recheck compatibility, security support, and semantics before any deployment or upgrade.
- The linked Flink 1.20 Kafka documentation currently says no connector is available for that Flink version; do not infer an installable connector artifact from its generic API examples.
- Selecting and validating a genuinely compatible connector release is an explicit prerequisite for building this proposal; otherwise select a supported Flink/connector pairing and recheck all cited semantics.
- **Language proposal:** Java with explicit serializers and stable operator IDs.
- **Input proposal:** durable partitioned Kafka topics with retention sufficient for recovery and a separately durable rule log.
- **Event-time stage:** keyed broadcast process function gates events on rule completeness and produces version-stamped events.
- **Pattern stage:** a fixed Flink CEP pattern keyed by `(issuer_id, card_token, rule_version)`.
- **Durability:** checkpoints stored in an access-controlled durable object store, with an appropriate supported Kafka sink.
- **Delivery boundary:** alerts first go to a durable alert-intent topic; an idempotent consumer writes cases.
- **Operations:** telemetry on watermark progress, source lag, checkpoints, side outputs, and alert consumption.
- **Not specified:** cloud provider, hardware type, exact connector version, and state backend benchmark.
- These must be chosen and pinned before a real test, rather than inferred from this case.

### 5c. Method Summary

- Validate the record and enforce stable identity before it enters the matching contract.
- Assign the producer occurrence timestamp and event-time watermark.
- Wait for both event-time progress and a trustworthy rule-coverage frontier before assigning a version.
- Select the rule effective at the first event's timestamp.
- Feed eligible events into a fixed candidate-pair CEP pattern for each relevant retained version.
- On a candidate match, require both events to be consistent with the version captured for the first event.
- Evaluate version-specific amount thresholds on the candidate pair.
- Publish an immutable alert intent with a deterministic pair-and-rule identity.
- Insert the resulting case atomically under a unique key and audit the effect.
- Route too-late events and corrections to a separately governed reconciliation process.

| Candidate | Evidence considered | Strength | Limitation | Status / revisit |
|---|---|---|---|---|
| Single large authorization | Hypothetical current rule | Simple and low state | No small-then-large relationship | Baseline; measure same cohort |
| Event-time Flink CEP plus versioned broadcast | Apache API semantics and constructed requirements | Expresses fixed sequence and preserves auditable rule version | Latency, state, and control-plane finality | Proposed; no implementation or benchmark |
| Custom keyed state machine | Considered only | Explicit dynamic logic and ordering | More application correctness code | Prefer if dynamic pattern structure is essential |
| Offline batch review | Considered only | Simple full-history reconciliation | Not suitable for timely interventions | Use as oracle or fallback where delay is acceptable |

- The rule is a deterministic condition, not an ML classifier.
- There is no optimizer, fitted preprocessing, model seed, or training loss.
- The key invariant is that a given logical pair and starting rule version produce one stable intent identity.
- No alternative has been tested on a real workload.
- A rule that changes the sequence topology cannot be implemented merely by mutating a broadcast threshold.

### 5d. Data provenance, quality, exclusions and leakage controls

- The inventory supplies the topic and filename, not the underlying source document.
- The hypothetical source records, rule history, cases, and staffing schedules do not exist as accessible evidence here.
- Each input schema and rule version would need ownership, access approval, and retention documentation.
- Preserve the original source position alongside the logical event ID for reconciliation.
- A source retry with the same ID and payload is a duplicate, not another authorization.
- A source retry with the same ID and different payload is a conflict, not a safe overwrite.
- Quarantine conflicts until the producer supplies an authoritative correction.
- Separate invalid schema rate from excluded but valid non-USD or declined authorizations.
- Measure exclusion rates by producer, channel, and time period; check for disproportionate gaps.
- Timestamp outliers can prematurely advance watermarks; bound plausible future skew at ingestion and quarantine violations.
- The acceptance bounds must be agreed with the producer, not inferred from an example.
- Restrict raw card tokens and retain only what is needed for processing, audit, and approved investigation.
- Encryption, identity, retention duration, and deletion rights need organization-specific security review.
- Do not claim compliance with any law without jurisdiction-specific assessment.

### 5e. Reproducibility and method-selection ledger

| ID | Artifact / location | Version or date | Supports | Access / limitation |
|---|---|---|---|---|
| SRC-INV | `Train/Templates/directory.txt`, Flink stateful processing entry | Inventory as supplied | Topic mapping only | Original case content unavailable |
| SRC-CEP | [Apache Flink CEP documentation](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/libs/cep/) | Flink 1.20, consulted 2026-11-07 | CEP pattern, lateness, and skip semantics | Public; not evidence of issuer outcome |
| SRC-BROADCAST | [Apache Flink broadcast state](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/fault-tolerance/broadcast_state/) | Flink 1.20, consulted 2026-11-07 | Broadcast processing and deterministic state updates | Public; does not supply the frontier protocol below |
| SRC-TIME | [Apache Flink timely stream processing](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/concepts/time/) | Flink 1.20, consulted 2026-11-07 | Event time and watermarks | Public; does not guarantee event completeness |
| SRC-CHK | [Apache Flink stateful stream processing](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/concepts/stateful-stream-processing/) | Flink 1.20, consulted 2026-11-07 | State and checkpoint/recovery concepts | Public; no blanket guarantee for arbitrary external effects |
| SRC-KAFKA | [Apache Flink Kafka connector](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/connectors/datastream/kafka/) | Flink 1.20, consulted 2026-11-07 | Connector and Kafka sink delivery contracts | Public; validate connector artifact and broker configuration |
| RUN-01 | Arithmetic and traces in Section 6 | Constructed examples | Logic and units | No executable run or observed output exists |
| RUN-02 | Future replay and failure-injection artifacts | Not yet created | Proposed evaluation only | No result is claimed |

- Publication dates for the versioned web pages are not asserted where not stated.
- The documentation pages describe an older Flink release and direct readers to current stable documentation.
- Record an actual source commit, connector dependency, serializer snapshot, job configuration, and checkpoint ID if an implementation is built.
- A future run must retain input manifest, rule log, output IDs, sink audit, environment, run time, and oracle revision.
- No claim of numerical measurement or production performance follows from reading framework documentation.

## 6. Analysis Walkthrough

### 6.1 Decision and claim ledger

- **Decisive tradeoff:** a more selective rule can reduce workload but may miss fraud that the amount-only baseline would have reviewed.
- **CLM-01:** the proposed architecture has no measured effect on confirmed fraud, latency, or customer outcomes.
- Evidence: not measured; source: absent deployment and RUN-02; carried to Sections 7, 9, and 11.
- **CLM-02:** with the constructed confusion matrix below, 200 proposed alerts fit exactly into a 200-slot daily queue.
- Evidence: illustrative arithmetic, RUN-01; not a forecast or a capacity guarantee.
- **CLM-03:** event-time watermarks provide progress declarations, not proof that all events have arrived.
- Evidence: SRC-TIME; supports a late-data policy, not an estimate of the late fraction.
- **CLM-04:** broadcast-state updates must be deterministic across operator instances.
- Evidence: SRC-BROADCAST; does not implement our custom activation/finality protocol.
- **DEC-01:** use a rule finality frontier and version-stamped candidate events rather than current-arrival-time rule lookup.
- Evidence: proposed design logic; no experimental comparison.
- **DEC-02:** send external effects through deterministic intent IDs and an atomic case-store uniqueness constraint.
- Evidence: proposed end-to-end contract; no successful recovery drill claimed.

### 6.2 Precise rule definition

- Rule ID is `R-test-then-large`; version `v1` uses small amounts from 100 to 1,000 cents inclusive.
- The second event must be an approved authorization of at least 50,000 cents.
- The second event must have strictly later event time than the first.
- It must occur no more than 120,000 milliseconds after the first.
- Formally, for first event `a` and second event `b`, require `0 < b.t - a.t <= 120000`.
- Both events must share `issuer_id` and `card_token`.
- Both must be valid USD approved authorizations and have distinct logical IDs.
- All qualifying ordered pairs are candidates; intervening eligible events do not invalidate a pair.
- A card may therefore generate more than one pair and more than one alert intent.
- There is no hidden per-card cooldown or one-alert-per-day deduplication in this contract.
- This is a deliberate correctness definition, not a recommendation to expose all pairs to reviewers without capacity controls.
- A rule version's thresholds and active interval are immutable after its coverage frontier is committed.
- Version selection is based on the first event time, not the second event time or processing time.
- If a pair crosses a rule boundary, the first event's version governs both predicates.
- The second event is replicated to every retained candidate version whose first-event interval could still pair with it.
- Replication is bounded by versions with effective intervals intersecting `[b.t - 120000, b.t)`.
- A second event can thus complete an older rule's partial match even after a new version becomes effective.
- Candidate copies are tagged with the version, but only events actually in that version's start interval may open first positions.
- The all-pairs contract requires a tested CEP pattern with relaxed nondeterministic contiguity and no skip after a match.
- The exact Flink Pattern API construction and tie behavior must be verified against the pinned version in the planned tests.
- The proposed implementation must enforce the inclusive 120,000-ms boundary explicitly; do not assume an untested `within` boundary.
- A CEP time bound may be configured conservatively and the exact inequality applied to the emitted pair.
- If the API cannot emit every required pair at the inclusive boundary, use an explicit keyed state machine instead of changing the contract.
- SRC-CEP describes the pattern vocabulary; the versioned candidate fanout and boundary check are case-specific design work.

### 6.3 A worked event-time trace

- All times in this trace are UTC on one hypothetical day.
- Rule `v1` is effective from 09:00:00.000 inclusive until 10:00:00.000 exclusive.
- Rule `v2` is effective from 10:00:00.000 inclusive; it changes the small upper bound to 500 cents.
- Both versions have the same 50,000-cent second-event threshold and two-minute horizon.
- The publisher committed a coverage frontier beyond 10:03:00 before these records were finalized.
- Event `a`: card `K`, ID `A`, event time 09:59:30.000, amount 800 cents, approved.
- Event `b`: card `K`, ID `B`, event time 10:00:15.000, amount 60,000 cents, approved.
- Event `c`: card `K`, ID `C`, event time 10:00:20.000, amount 700 cents, approved.
- Event `d`: card `K`, ID `D`, event time 10:00:45.000, amount 70,000 cents, approved.
- The processing arrival order is `b, a, d, c`; a later watermark has not yet passed `a` when `a` arrives.
- The gate buffers and then releases records in event-time order with a deterministic ID tie-break.
- `A` starts a `v1` partial match because 800 cents is in the `v1` small range.
- `B` is 45 seconds after `A` and satisfies the `v1` large threshold.
- The pair `(A,B,v1)` is valid even though `B` occurs after `v2` became effective.
- `C` is not a `v2` start because 700 exceeds the `v2` small upper bound.
- `D` is 75 seconds after `A` and also completes `(A,D,v1)` under all-pairs semantics.
- `C` is not allowed to open a `v1` start because it occurred outside the `v1` start interval.
- Therefore `(C,D,v1)` is invalid despite 700 satisfying the old small threshold.
- `(C,D,v2)` is invalid because 700 exceeds the new small threshold.
- The two expected intent identities are based on `R-test-then-large,v1,A,B` and `R-test-then-large,v1,A,D`.
- The trace is constructed, not an observed run of Flink.
- If `A` arrived after its event-time watermark, it would be quarantined as late rather than silently inserted into the live matches.
- If the same `A` were replayed from a checkpoint, the deterministic intent ID would remain unchanged.

### 6.4 Business-capacity arithmetic

- Consider a hypothetical fully adjudicated cohort of 1,000,000 eligible events over one day.
- Suppose the baseline produces 180 cases: 36 confirmed fraud cases and 144 legitimate cases.
- Baseline precision is `36 / 180 = 20%`; baseline false-positive proportion among alerts is `144 / 180 = 80%`.
- Suppose the candidate produces 200 cases: 50 confirmed fraud cases and 150 legitimate cases.
- Candidate precision is `50 / 200 = 25%`; its false-positive proportion among alerts is `150 / 200 = 75%`.
- These counts are constructed and are not evidence that the sequence rule improves anything.
- The candidate uses all 200 daily review slots; the baseline uses 180 slots.
- On a replacement comparison, this hypothetical candidate has 14 more confirmed cases and 6 more false-positive cases.
- The difference in precision is 5 percentage points, not five percent relative.
- Its relative precision increase is `(0.25 - 0.20) / 0.20 = 25%` under these invented counts.
- The reviewer must still consider fraud cases missed by replacement and the value of each case.
- Suppose 30 confirmed cases overlap across the two sets.
- Then candidate-only confirmed cases are `50 - 30 = 20` and baseline-only confirmed cases are `36 - 30 = 6`.
- The net replacement difference is `20 - 6 = 14`, consistent with the totals.
- Alert overlap and confirmed-case overlap are not interchangeable.
- Suppose the alert sets overlap in 100 case identities, including those 30 confirmed cases.
- Their union contains `180 + 200 - 100 = 280` cases, exceeding 200 slots by 80.
- Thus adding the full candidate queue on top of the baseline is not capacity-feasible.
- A ranked or sampled 20-slot addition to the 180 baseline cases is a different policy and needs its own outcome evaluation.
- The number of legitimate cases in the union is `280 - (36 + 50 - 30) = 224` under a shared complete adjudication.
- The union contains `56` confirmed cases, but only if all 280 can be reviewed and the labels are valid.
- At 15 minutes per case, 200 reviews require 3,000 minutes, or 50 reviewer-hours, before breaks and overhead.
- If each reviewer provides 6 productive review hours per day, the arithmetic requires `50 / 6 = 8.33` reviewer equivalents.
- The operating roster would need at least 9 whole people under that simplified daily staffing model.
- The 15-minute handling time and six productive hours are assumptions, not measurements.
- This capacity exercise counts alert cases, not all underlying transactions, and assumes one slot per distinct case.

### 6.5 Sensitivity and alternative explanations

| Stress | Original assumption | Variation | Consequence | Status |
|---|---|---|---|---|
| Daily capacity | 200 slots | 150 slots | Both candidate and baseline queues exceed capacity; prioritize or defer | Illustrative |
| Candidate false positives | 150 of 200 | 170 of 200, same 200 total | Precision falls from 25% to 15% if remaining 30 are confirmed | Illustrative |
| Review duration | 15 minutes | 20 minutes | 200 reviews need 66.67 productive hours, not 50 | Illustrative |
| Late fraction | Unknown | Materially above agreed guardrail | Live recall may be unacceptable; latency/coverage decision changes | Not measured |
| Rule update availability | Frontier progresses | Frontier stalls | Correctness-preserving gate delays output and can exhaust buffers | Proposed failure test |
| Baseline overlap | 100 cases | 150 cases, same totals | Union falls from 280 to 230, still above 200 | Illustrative |

- At 170 legitimate alerts out of 200, the remaining 30 confirmed cases imply `30 / 200 = 15%` precision.
- At 20 minutes per review, `200 × 20 / 60 = 66.67` hours, requiring 12 whole reviewers at six productive hours each.
- Apparent uplift can result from different populations, duplicate alerts, unequal label maturity, or selective investigation.
- Use identical eligibility, time periods, case identities, and label windows before comparing the arms.
- What was tried first: nothing was actually tested; all candidates remain proposed alternatives.
- The highest-leverage unknown is the combination of incremental fraud coverage and reviewer capacity, not the choice of state backend.
- A reasonable alternative with a simpler control-plane contract is a scheduled job restart at each version boundary; evaluate it if rule changes are rare.

### 6.6 Boundary and ordering laboratory

The following are specified expected results, not executed tests.
Each exercise uses the `v1` thresholds unless another version is named.
All events belong to the same issuer and card unless the exercise states otherwise.
IDs represent distinct authorizations unless explicitly identified as duplicates.
Times are integer milliseconds; amounts are integer USD cents.
The oracle enumerates ordered pairs and applies the first event's effective rule.

#### Example A: minimum small amount

- `A1` occurs at 1,000 ms for 100 cents.
- `A2` occurs at 2,000 ms for 50,000 cents.
- The first amount equals the inclusive lower small bound.
- The second amount equals the inclusive large threshold.
- Their event-time difference is 1,000 ms.
- The pair qualifies and should create one intent.
- Changing `A1` to 99 cents makes the first predicate false.
- No additional currency conversion or rounding is involved.
- The evaluation oracle should retain the 99-cent record in its eligible population even though it is not a start.

#### Example B: maximum small amount

- `B1` occurs at 1,000 ms for 1,000 cents.
- `B2` occurs at 2,000 ms for 50,000 cents.
- The first amount equals the inclusive upper small bound.
- The pair qualifies because both amounts meet their respective predicates.
- Replacing the first amount with 1,001 cents makes the pair invalid.
- The start predicate must use the rule version attached to the first event.
- This comparison detects accidental exclusive upper-bound logic.
- A shadow run should compare the exact pair ID, not only the total number of matches.

#### Example C: maximum duration

- `C1` occurs at 1,000 ms for 500 cents.
- `C2` occurs at 121,000 ms for 50,000 cents.
- The difference is `121000 - 1000 = 120000` ms.
- The declared contract includes this pair.
- An implementation that clears the start before processing `C2` violates the contract.
- A 121,001-ms second event yields a 120,001-ms difference and is excluded.
- A timer test must control both event ordering and watermark advancement.
- A timer API call alone does not establish that the boundary behaves as intended.

#### Example D: equal timestamps

- `D1` and `D2` occur at 8,000 ms.
- `D1` has the small amount and `D2` has the large amount.
- Their difference is zero, outside the contract's strictly positive interval.
- Deterministic ordering by ID does not turn equal timestamps into a positive time difference.
- Neither arrival order creates a valid pair.
- If the producer can only supply coarse timestamps, measure how many potential pairs this excludes.
- Adopting a producer sequence number as a tie-break would be a business-contract change.
- Any such change requires a new oracle and reevaluation of historical comparisons.

#### Example E: different card keys

- `E1` is a 500-cent start for card `K1`.
- `E2` is a 50,000-cent end for card `K2`.
- The timestamps are one second apart, but `K1` and `K2` are different.
- The expected match count is zero.
- A partitioning bug that keys only by issuer could create a false pair.
- A join using a non-unique token can create the same error.
- Review token namespace and issuer scoping when a processor migrates its token service.
- Cross-card fraud-ring detection is a separate, explicitly excluded decision.

#### Example F: different issuers

- `F1` and `F2` carry the same token text and qualifying amounts.
- `F1` belongs to issuer `I1`; `F2` belongs to issuer `I2`.
- Their event times differ by 10 seconds.
- The pair must not match because issuer is part of the key.
- A token need not be globally unique across issuers.
- Intent identity also contains the issuer to keep case keys distinct.
- A test fixture with token collisions across tenants checks isolation.
- Access control must also prevent one issuer's analysts from retrieving another issuer's data.

#### Example G: two starts, one end

- `G1` at 1,000 ms is 300 cents.
- `G2` at 2,000 ms is 400 cents.
- `G3` at 3,000 ms is 60,000 cents.
- Both starts satisfy the same version and are within the duration bound.
- Expected pairs are `(G1,G3)` and `(G2,G3)`.
- A skip-past-last-match strategy could suppress a required result.
- The case identity includes each first event ID, so the two intents differ.
- Any reviewer-level grouping must preserve both underlying pair references.

#### Example H: one start, two ends

- `H1` at 1,000 ms is 300 cents.
- `H2` at 2,000 ms is 60,000 cents.
- `H3` at 3,000 ms is 70,000 cents.
- Both `(H1,H2)` and `(H1,H3)` qualify.
- Emitting the first pair must not consume `H1` under the all-pairs contract.
- A downstream rule that merges the two into one case needs an explicit different identity contract.
- Counting pair intents and counting reviewer cases are therefore distinct measurements.
- The baseline comparison in Section 6 assumes a case-level identity already agreed across policies.

#### Example I: irrelevant intervening event

- `I1` is a 500-cent approved start at 1,000 ms.
- `I2` is an approved 2,000-cent transaction at 2,000 ms.
- `I3` is a 60,000-cent approved end at 3,000 ms.
- `I2` is neither a small start nor a large end under `v1`.
- It must not prevent the `(I1,I3)` match.
- Strict adjacency would incorrectly discard this relationship.
- Verify the pattern uses a contiguity mode compatible with intervening records.
- The source trace must include non-matching eligible records for this test to be meaningful.

#### Example J: same logical authorization delivered twice

- `J1` is received at one source offset and repeated at another offset.
- Both copies have the same issuer, event ID, and canonical payload digest.
- The record is one logical authorization even though two transport records exist.
- The event-ID keyed stage admits one canonical record to card-keyed matching.
- A qualifying large `J2` therefore yields one `(J1,J2)` intent.
- The duplicate count is recorded as an input-quality metric.
- The second offset remains auditable even though it does not represent a new authorization.
- Restore and replay must preserve this result within the permitted deduplication horizon.

#### Example K: conflicting identity

- Two source records have the same issuer and event ID.
- Their amount fields differ and so their canonical payload digests differ.
- This is not an idempotent retry.
- Neither record should silently replace the other after the conflict is discovered.
- The producer must identify the authoritative representation under the correction process.
- If the original record already generated an alert, preserve that case and attach the correction audit.
- Report the conflict rate separately from identical duplicates.
- A zero-conflict claim requires reconciled producer records, not merely a lack of exceptions.

#### Example L: out-of-order but not late

- `L2` with event time 20 seconds arrives before `L1` at 10 seconds.
- The relevant transaction watermark is still below 10 seconds when `L1` arrives.
- The gate retains both until its ordering and rule-frontier conditions hold.
- It releases `L1` before `L2` even though arrival order was reversed.
- If the amounts qualify, the oracle and live path should emit the same pair.
- The output latency includes the wait for safe ordering.
- The selected 10-second disorder bound is a proposal and needs workload evidence.
- Reversing transport order in replay should not alter the stable intent ID.

#### Example M: record after progress

- The gate's transaction-side watermark has already reached 20 seconds.
- A record with event time 10 seconds subsequently arrives.
- The proposed gate conservatively classifies it as late and writes it to the reconciliation queue.
- It does not revise the already emitted live intent manifest.
- The reconciliation oracle can later discover a pair involving that record.
- An authorized operator decides whether a new correction case or an amendment is appropriate.
- The correction cannot make a previously missed 60-second decision timely.
- Late-record counts must be associated with the affected cohort, not only with the arrival day.

#### Example N: CEP versus gate lateness

- The custom gate intentionally treats an event at or below its transaction watermark as late.
- Flink 1.20 CEP documentation describes its own event-time behavior in terms of timestamps strictly below the last watermark.
- CEP buffers records and processes them in timestamp order when their timestamp is smaller than the watermark.
- CEP can send records it considers late to a side output; it does not process them in the normal pattern.
- This gate policy is deliberately more conservative at equality than the documented CEP late condition.
- The implementation must label which component classified each event and test equality explicitly.
- Window `allowedLateness` does not automatically cause CEP to revise a completed match.
- Source: SRC-CEP, handling lateness in event time; no correction behavior is inferred from window APIs.

#### Example O: rule-frontier equality

- Assume the current committed frontier is 10:00:00.000.
- A transaction timestamped 09:59:59.999 lies strictly below the frontier.
- Once ordering is also safe, its effective rule can be selected.
- A transaction timestamped exactly 10:00:00.000 is not yet covered by that frontier.
- It must wait until a later frontier is committed.
- A transaction after 10:00:00.000 must also wait.
- This is a boundary of the proposed control protocol, not a built-in Flink rule.
- The control-plane log must record the new frontier before the gate acts on it.

#### Example P: immutable finality violation

- A publisher commits a frontier beyond 10:05.
- It later attempts to insert a rule effective at 10:02.
- The proposed contract rejects that ordinary update because it would change finalized history.
- The event log retains the rejected request and authorized actor.
- If the change is genuinely necessary, publish an explicit correction instruction for separate reconciliation.
- Do not rewrite the original version history or erase previously produced intent identities.
- Report how many historical candidates and cases the correction affects.
- A correction requires policy approval because it can create or amend customer-facing work.

#### Example Q: temporary frontier outage

- Suppose the rule frontier stops progressing while transaction input continues.
- Events newer than its coverage boundary cannot be safely assigned a rule.
- Their buffer grows even if the transaction source watermark continues to advance.
- The frontier-age alert pages the rule owner rather than requesting a fabricated frontier.
- A configured byte limit causes backpressure or source pause before local storage exhaustion.
- An operator checks source retention against the expected repair duration.
- Recovery resumes in event-time order after the authoritative log catches up.
- A post-recovery audit compares held events, released events, and rejected records by identity.

#### Example R: rule churn and fanout

- Suppose four rule intervals each overlap the two-minute first-event lookback.
- A qualifying large event may need four version-tagged candidate copies.
- Each copy can only match starts belonging to that version's effective interval.
- Copying the end does not mean creating four alerts automatically.
- The actual number of pairs depends on the stored starts and threshold predicates.
- A rate limit on rule publication bounds the number of intervals in the lookback.
- Such a limit must be approved by the rule owner and monitored.
- If control-plane requirements exceed the limit, redesign rather than dropping candidate versions.

#### Example S: case-service commit ambiguity

- The consumer sends an intent to the case service.
- The case service commits the unique case and its outbox row.
- The response is lost before the consumer receives it.
- The consumer cannot tell whether the commit happened and retries the same intent ID.
- The service finds the existing key and matching payload digest.
- It returns the existing case without creating a second review item.
- A different payload for the same key is an incident, not a successful idempotent retry.
- This is a case-service guarantee, separate from the Flink checkpoint boundary.

#### Example T: notification after case commit

- A case and an outbox record commit in one database transaction.
- The notification worker reads the outbox and calls a third-party contact provider.
- The provider may act even if its response is lost.
- Replaying the outbox alone does not prevent a second external contact.
- The provider must support a stable idempotency key, or the workflow needs another approved safeguard.
- A successful case uniqueness test does not establish notification uniqueness.
- Store each delivery attempt and provider receipt where permitted.
- Escalate ambiguous outcomes rather than claiming exactly-once communication.

### 6.7 Repair and reconciliation worked decisions

- An ordinary duplicate creates no new case; it increments a duplicate-delivery count.
- A payload conflict pauses automatic interpretation for that identity and opens a producer data-quality investigation.
- A late valid start can produce a newly discovered historical pair in the reconciliation oracle.
- A late valid end can also complete an earlier start after the live watermark passed it.
- A correction may invalidate an existing pair, create a new pair, or change neither outcome.
- The repair service compares the original intent manifest with the corrected manifest.
- Set difference `corrected minus original` identifies newly justified pair intents.
- Set difference `original minus corrected` identifies intents requiring review or amendment.
- The intersection identifies identities still supported by the corrected history.
- An intent in the intersection can still have a payload conflict that requires separate inspection.
- Repair writes a new immutable correction record with source IDs, rule version, reason, and approver.
- A prior case is never deleted merely to make counts agree with a revised calculation.
- The case record retains the original decision and links any subsequent adjudication.
- A new correction case uses a documented identity that cannot accidentally collide with the original pair intent.
- If the original case was already reviewed, the human workflow determines whether reopening is useful.
- If the original case led to a customer action, an authorized team determines remediation.
- Reconciliation cannot infer the customer's loss or undo a prior communication.
- Metrics must report late-found cases separately from timely cases.
- The reconciliation service must have access to the same historical rule versions used for live decisions.
- If the rule history is missing, mark affected intervals unreconstructable and escalate.
- A rule version restored from an untrusted backup must be checked against the signed control log.
- Any bulk repair is staged against a manifest before effects are enabled.
- The manifest contains the old identity, proposed new action, affected data interval, and expected counts.
- Two-person approval is a proposed governance option, not an asserted existing issuer policy.
- The repair consumer must use durable idempotency just like the live consumer.
- The repair plan needs explicit authorization, rate limits, and a rollback of uncommitted work.
- Already delivered effects require compensating business actions rather than a database rollback.
- A closure report records unresolved IDs, source gaps, and the exact reconciliation cutoff.
- Evidence of a successful recovery is the reconciled identity set, not simply a green job status.
- The workflow remains untested until its run and artifacts exist.

### 6.8 Failure-window accounting

- Window 1 begins after source ingestion but before the event reaches a completed checkpoint.
- A crash in this window can cause source replay of the event.
- Identical event IDs must be admitted only once to the logical matching history.
- Window 2 begins after a candidate match is computed but before its sink transaction commits.
- A crash can discard an uncommitted intent and recompute it after restore.
- The deterministic intent ID makes the recomputed output comparable to the first attempt.
- Window 3 begins after a Kafka transaction commits but before a downstream consumer acknowledges processing.
- The consumer can receive the committed intent again after restart.
- Unique case insertion makes the retry return the existing case.
- Window 4 begins after case commit but before the API response is received.
- The consumer must retry with the same key, not mint a fresh case ID.
- Window 5 begins after the outbox worker sends a provider request but before provider acknowledgement.
- This requires provider-side idempotency or an explicit ambiguous-delivery workflow.
- Window 6 begins after a checkpoint is retained but before its source offsets are no longer available.
- If retention expires, that checkpoint may no longer be usable for recovery.
- Window 7 begins after a rule frontier has committed and before all transaction partitions have advanced.
- Historical rule versions must remain available while in-flight first events can still match.
- Window 8 begins after rollback starts and before case consumers are fenced or stopped.
- Two job versions can otherwise produce intents simultaneously.
- Shared deterministic IDs can suppress identical effects, but different rule versions remain distinct identities.
- The release procedure must explicitly decide whether those version-distinct cases are permitted.
- Window 9 begins when a partition is marked idle and ends when it resumes with old data.
- The resumed data may be late because other partitions advanced the watermark.
- Idleness therefore exchanges waiting time for an increased possibility of late reconciliation.
- Window 10 begins when review capacity is exhausted and ends when the queue is drained.
- Stream correctness can remain intact while the business deadline is missed.
- Record queue arrival time and reviewer completion time independently of event time.
- No single checkpoint, retry flag, or latency dashboard covers all ten windows.
- A production readiness review must assign an owner and observable recovery evidence to each boundary.

## 7. The Finding

Recommend a shadow evaluation of the event-time, versioned sequence rule against the current single-transaction baseline, not an immediate production replacement.
Its confirmed-fraud benefit, late-event miss rate, reviewer load, and operational reliability are **not measured** (CLM-01).
The constructed 200-alert example demonstrates that a candidate can fill the available queue and that adding it to the baseline may exceed capacity; it is not a forecast (CLM-02).
A go/no-go decision requires mature labels, matched cohorts, a rule-finality contract, failure drills, and an operations-approved allocation of the shared queue.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

- No training job or statistical model is proposed.
- A strategy analyst proposes a threshold change with rationale, effective interval, owner, and approval record.
- A validation service checks schema, interval overlap, required fields, and threshold units.
- A control-plane signer authenticates the accepted immutable rule record.
- A single ordered rule log records the version and the committed coverage frontier.
- An independent replay uses an immutable snapshot of transaction events and the same rule history.
- The replay sorts by event time and deterministic tie-break and enumerates the expected pairs.
- It produces a versioned expected-intent manifest and counts exclusions separately.
- Evaluation joins the manifest to adjudication records only after their availability date.
- The operations team compares candidates using a frozen evaluation protocol and shared capacity.
- A release decision is based on the joint correctness and business gates in Section 9.
- No actual trained artifact, build, run, or approval is claimed.

```text
[Approved rule proposal] -> [Validation + authorization] -> [Immutable ordered rule log]
                                      |                                  |
                                      v                                  v
                              [Rejected change audit]          [Version + finality frontier]
[Authorized transaction snapshot] -> [Batch oracle with same rule history] -> [Expected intent manifest]
                                                                       |
[Matured adjudications] ----------------------------------------------> [Evaluation + review decision]
```

### 8b. Online / Serving / Real-Time Path

```text
[Authenticated authorization producer] -> [Kafka transaction partitions]
      -> [validate schema / time / ID] --invalid--> [restricted quarantine + repair]
      -> [timestamps + transaction watermarks] -> [key by issuer, card]
                                                  |
[approved single ordered rule log] -> [broadcast state + finality frontier]
                                                  |
                           [event-time buffer + version-selection gate]
                                                  |
                            [version-tagged candidate copies; key by card/version]
                                                  |
                               [fixed CEP pair pattern + exact pair predicate]
                                                  |
                         [deterministic alert intent ID, transactional Kafka sink]
                                                  |
                   [read-committed consumer] -> [atomic case insert + outbox]
                                                  |
                                  [human review and authorized feedback]

[late / corrections] -> [durable reconciliation queue] -> [versioned repair decision + audit]
[checkpoints] <- source offsets, keyed state, timers, broadcast state, CEP state, sink state
```

- The diagram is a proposed logical architecture, not a deployed system.
- Transaction and rule streams are independent; broadcast alone does not align their arrival order.
- The rule source carries normal rule records followed by finality-frontier records in one ordered partition.
- Every broadcast subtask processes the same log in the same order and validates its monotonic sequence number.
- An authoritative frontier `F` promises that no future accepted rule will alter the version effective at any event time `t < F`.
- An event at `t` is eligible for version assignment only when `F > t`.
- Equality is intentionally insufficient: a frontier equal to `t` covers only times strictly below `t`.
- A publisher may advance `F` ahead of wall time only after approved future coverage is irrevocably committed.
- A failed publisher must resume from the durable log and must not retract an emitted frontier.
- A retroactive emergency change is recorded as a correction and handled by reconciliation, never disguised as an ordinary update.
- If the frontier is unavailable or a sequence gap is detected, hold affected transactions; alert the owner and apply bounded overload policy.
- Do not guess a rule or silently fall back to the last arrived version.
- The proposed source emits rule-stream watermarks consistently with frontier records so a connected operator can advance event time.
- The implementation must verify that the two-input minimum watermark and record ordering actually preserve the frontier gate in the selected connector version.
- The operator buffers events until event-time order is safe and their `t < F` condition is met.
- It releases eligible events ordered by `(event_time_ms, event_id)` per card.
- The gate must preserve the original event timestamp as the Flink record timestamp entering CEP, not merely as a payload field.
- In particular, timer-triggered output can inherit the timer timestamp; an implementation needs explicit timestamp reassignment and a tested downstream-watermark holdback so CEP never sees the gate's buffered records after their watermark.
- A standard function with automatically forwarded watermarks is not assumed to provide this custom alignment; a compatible operator/watermark design is a prerequisite.
- If that behavior cannot be demonstrated on the selected Flink release, replace this gate-plus-CEP combination with a correctly specified keyed state machine rather than claiming the diagram is deployable.
- Events at or before the current transaction-side watermark on arrival are late under this proposed policy.
- A late event is written to durable reconciliation; it is not allowed to mutate already finalized live CEP state.
- The gate must separately track the transaction watermark; using only a minimum of two inputs can obscure a stalled or advancing control stream.
- If connector/API behavior cannot expose the required independent progress safely, use an explicit progress control record or a redesigned two-stage gate.
- Buffer state, release timers, and rule state are checkpointed with input positions.
- A processing-time watchdog reports a stalled watermark or frontier; it does not advance either for correctness.
- Idle transaction partitions require an explicit idleness policy; a resumed partition can yield late records after the watermark advanced.
- Do not mark the rule stream idle merely to hide a missing frontier.
- Alert-intent output retains the original pair, rule ID/version, decision timestamps, and audit metadata.
- No synchronous remote case API call is placed inside the CEP operator's match callback.

### 8c. Key Design Decisions

| ID | Choice | Alternatives | Evidence | Tradeoff / failure boundary | Revisit |
|---|---|---|---|---|---|
| DEC-01 | Immutable versions plus rule frontier | Current arrival-time rule; scheduled restart | Proposed correctness contract | Delays under stalled publisher; depends on irreversible coverage | Publisher cannot support frontier |
| DEC-02 | Intent topic plus atomic unique case write | Direct HTTP from operator | Proposed effect contract | Extra queue and consumer; no atomic cross-system magic | Case service lacks unique key |
| DEC-03 | Fixed CEP topology with candidate version fanout | Fully dynamic pattern; custom state machine | SRC-CEP and design reasoning | Fanout/state grow with rule churn | Pattern topology or scale changes |
| DEC-04 | Event-time ordering with late side path | Processing time; infinite waiting | SRC-TIME and scenario | More latency or missed live matches | Lateness distribution changes |
| DEC-05 | All valid pairs, no implicit cooldown | First pair only; per-card collapse | Explicit business semantics | Can explode in hot keys | Review capacity or business rule changes |
| DEC-06 | Separate evaluation and action | Direct automatic blocking | Scenario risk boundary | Slower intervention | Governance and evidence justify change |

### 8d. Architecture artifacts and operational readiness

#### Interface and effect contract

- `intent_id` is a collision-resistant hash of a canonical length-prefixed encoding of issuer, rule ID, rule version, first event ID, and second event ID.
- The canonical encoding includes field names or fixed field order and a schema version to avoid ambiguous concatenation.
- Event IDs in the key are ordered by their first/second roles, not lexicographically sorted.
- A version change intentionally makes a different intent identity for the same pair.
- The output also carries a canonical payload digest to distinguish a retry from a conflicting rewrite.
- A cryptographic hash collision is treated as an incident; retain the canonical components for verification.
- The case store must atomically enforce unique `intent_id` and compare payload digest on conflict.
- Same key and same digest returns the existing case without a second review item.
- Same key and different digest is quarantined as a conflict and never silently overwritten.
- An atomic transaction inserts the case and a case-service outbox row together.
- Any notification or external communication consumes that outbox with its own durable idempotency contract.
- The identity prevents duplicate case creation; it cannot make an unrelated non-idempotent vendor notification exactly once.
- Consumer acknowledgement occurs only after the case transaction commits.
- A timeout after commit causes a retry that returns the same case, not a second case.
- Case-service retention of the key must outlive the maximum permitted replay and reconciliation horizon.
- If a historical restore can outlive that horizon, restore matching deduplication records or prohibit the restore until reviewed.

#### State, timers, and rule updates

- Keyed gate state holds event IDs, timestamps, buffered payload fields, and timers per issuer/card.
- CEP state holds pending first events and match metadata for each card/version key.
- Broadcast state holds immutable versions and the highest committed rule frontier.
- Separate duplicate state holds event ID and canonical payload digest for the deduplication retention horizon.
- The ID deduplication stage must itself be keyed by issuer/event ID before repartitioning by card.
- A conflicting event ID is quarantined even if its card token differs.
- The retention policy must cover producer retries and any permitted live restore; actual durations require contract approval.
- An unbounded duplicate history cannot be kept cheaply; older replays require a durable deduplication ledger or an explicitly bounded replay horizon.
- Event-time timers clean up first-event partial matches after the latest allowed matching timestamp has safely passed.
- Do not delete a first event at exactly `a.t + 120000` before an equal-boundary second event is processed.
- The implementation and tests must establish timer-versus-record ordering and use a later cleanup instant if needed.
- Processing-time state TTL alone is not a substitute for event-time correctness across downtime or replay.
- Version retention covers the maximum pair horizon, watermark lag, replay policy, and correction horizon.
- A rule version may be removed only when no retained start, buffered event, checkpoint restore, or permitted reconciliation requires it.
- The rule registry remains the authoritative historical ledger after a local broadcast-state version is evicted.
- Broadcast updates are deterministic and monotonically validated; no task-local clock chooses a version.
- Rescaling duplicates broadcast state across subtasks and redistributes keyed state under supported Flink restore semantics.
- Test state serializer compatibility and rule-state equality after rescaling, rather than assuming arbitrary schema changes work.
- The candidate fanout uses all versions whose start intervals overlap the possible first-event interval for the second event.
- If rapid rule churn makes this set large, reject or throttle publication under an approved limit rather than silently omit versions.
- A fixed CEP topology is appropriate only while the pattern's structure remains fixed.
- A changed topology requires a separately versioned deployment or a custom pattern engine with its own state migration contract.

#### Reliability and observability

- Proposed SLI: among valid on-time candidate pairs whose second event is in the measurement cohort, fraction whose unique case is committed within 60 seconds of second-event ingestion.
- Proposed target: at least 99% over a rolling seven-day window; this is a scenario target, not an observed SLO.
- Denominator is determined by the independently replayed oracle, not merely by emitted cases.
- Report excluded late pairs separately so the latency SLI cannot improve by dropping hard records.
- Also measure completeness: unique expected on-time intent IDs with matching committed case divided by all oracle expected on-time IDs.
- Set the acceptable completeness threshold with fraud operations before a pilot; no threshold is asserted as approved.
- Track event-time lag per input partition, watermark age, source consumer lag, and idle-partition transitions.
- Track frontier position, frontier age, invalid rule count, rule sequence gaps, and version-fanout distribution.
- Track keyed state bytes, hot-card event rate, partial-match count, checkpoint duration, failure count, and restore duration.
- Track alert-intent production, committed-case rate, retry count, duplicate-key returns, payload conflicts, and outbox backlog.
- Track late records, correction records, invalid timestamps, quarantined conflicts, and their oldest unprocessed age.
- Track reviewer queue size, age, capacity consumed, and candidate/baseline overlap without exposing raw card identifiers in dashboards.
- The stream on-call owns stalled watermarks and checkpoints; rule engineering owns frontier stalls.
- The case platform owns case insert and outbox issues; fraud operations owns queue capacity and review decisions.
- Burn of the agreed error budget should stop expansion or trigger rollback under a written operational policy.
- Latency percentiles must be computed from end-to-end observations; adding component p99s does not produce an end-to-end p99.

#### Resilience and recovery

- Enable checkpointing with durable storage and compatible source/sink semantics before relying on replay.
- Checkpoint interval, timeout, minimum pause, restart strategy, and storage retention require a load and recovery test.
- Restoring state and source positions reproduces internal state relative to a completed checkpoint, subject to source retention and deterministic logic.
- A Kafka transactional sink can provide exactly-once intent publication only with the documented connector, broker, and checkpoint configuration.
- Configure consumers to read committed records where required; verify transactional timeout against realistic checkpoint and recovery duration.
- A case creation transaction is outside the Flink checkpoint and therefore requires independent idempotency.
- Kafka transactions do not atomically include the case database or downstream customer notification.
- On a case-service outage, pause or retry consumption with bounded exponential backoff and jitter while retaining intents durably.
- Apply a retry budget and alert on oldest unprocessed intent age; do not discard alerts after the budget expires.
- If retention will expire before recovery, invoke an emergency archive or stop input under the documented business continuity plan.
- On prolonged frontier stalls, cap buffered bytes, stop consuming or backpressure upstream, and alert the control-plane owner.
- Never invent a finality frontier to clear a backlog.
- Backpressure can increase source lag and eventually exceed retention; capacity and recovery drills must include this failure.
- For poison events, quarantine with a stable reason and source position so one malformed record does not create endless restarts.
- A restore from an older checkpoint may reproduce already delivered intents; the case store deduplicates them.
- After recovery, reconcile unique source IDs, alert intents, case IDs, and late/correction queues over the affected interval.
- If checkpoint files or source history are unavailable, the design cannot promise automatic recovery; activate a separately approved rebuild process.

#### Security and deployment

- Authenticate producer, rule publisher, job identity, case consumer, and operator access independently.
- Authorize each identity to the minimum topic, state, and service resources needed.
- Sign or otherwise authenticate rule publications; log approver, version, interval, and digest.
- Store secrets in an approved secret manager rather than in job configuration or logs.
- Encrypt data in transit and at rest according to actual organizational policy.
- Pseudonymous tokens remain sensitive linkage data; restrict access and retention accordingly.
- Rate-limit rule changes and validate intervals so an attacker cannot induce unbounded version fanout.
- Reject unauthorized retroactive updates and alert the security owner.
- Shadow deployment emits to an isolated topic and cannot create operational cases.
- Canary only after correctness, load, security, and recovery gates pass.
- A canary must avoid duplicate review by tagging traffic and preserving the same case identity namespace.
- Pin job artifact, serializer schema, rule schema, connector version, and checkpoint compatibility together.
- Roll back only to a binary that can read the current state and rule schema, or use an approved compatible savepoint.
- Rollback does not undo committed cases; reconcile or resolve them explicitly through the case service.
- An incompatible rollback requires a separately tested rebuild and an agreed loss/duplication accounting process.

#### Sizing and economics

- Assume the 100 events/second peak lasts 3,600 seconds: `100 × 3,600 = 360,000` events.
- That is 36% of the assumed 1,000,000-event daily volume.
- Assume an average of 200 serialized bytes of required event data before runtime overhead.
- Raw peak input is `100 × 200 = 20,000` bytes/second, approximately 20 kB/s decimal.
- A 10-second ordering buffer has a raw lower bound of `100 × 10 × 200 = 200,000` bytes under constant peak traffic.
- This excludes metadata, indexing, copies, fanout, checkpoints, uneven partitions, and backend overhead.
- Assume 10% of peak events can open a small-amount match and all remain pending for 120 seconds.
- The simplistic raw pending count is `100 × 0.10 × 120 = 1,200` first events.
- At 200 bytes each, their minimum payload is `1,200 × 200 = 240,000` bytes before state-engine overhead.
- If one hot card receives 10 small events/second for 120 seconds, that card alone can accumulate 1,200 pending starts.
- If it then receives 10 large events/second, all-pairs semantics can emit up to 12,000 pairs per second while the 1,200 starts remain eligible.
- That burst is an upper-bound illustration under the stated constant counts, not a throughput or fraud estimate.
- The broad fanout and case load can dominate input bandwidth, so benchmark match cardinality and sink capacity.
- Replicating 1 MiB of retained broadcast rules across 16 subtasks uses at least 16 MiB of logical copies before overhead and snapshots.
- Changing parallelism from 16 to 32 approximately doubles that logical broadcast footprint to 32 MiB under the same assumption.
- One hot card remains owned by one keyed subtask; adding partitions alone does not split its exact sequence state.
- Plan additional headroom for checkpoint copies, RocksDB or other backend metadata, compaction, GC, recovery, and source lag.
- Backend choice and actual state amplification must be benchmarked on the real implementation.
- No infrastructure price or staffing wage is supplied, so a monetary return on investment is not calculated.
- A future cost model must include compute, storage, checkpoint traffic, network, queue retention, failed retries, security controls, and review labor.

## 9. Evaluation & Validation

### 9.1 Evaluation contract

- **Primary metric:** incremental confirmed fraud cases reviewed within the agreed review deadline per eligible issuer-day versus the baseline policy.
- Numerator: candidate-policy confirmed unique cases reviewed on time minus baseline-policy confirmed unique cases reviewed on time.
- Denominator: eligible issuer-days in the prespecified evaluation population; also report absolute counts and the population size.
- This is a policy-level metric, not event-level classification accuracy.
- A case is confirmed only under an independently documented adjudication policy after the 90-day maturation period.
- The comparison must account for overlapping cases and shared staffing, not simply add case counts across arms.
- **Baseline:** approved-authorization amount rule at 50,000 cents, evaluated on the same eligible time and channel population.
- **Proposed pass rule:** positive estimated incremental cases per issuer-day with a confidence interval excluding zero, while all preregistered guardrails pass.
- This is an illustrative proposed threshold; fraud operations and statistical owners must approve it before seeing the final evaluation.
- **Capacity guardrail:** no policy may schedule more reviewer cases than the allocated daily slots.
- **Correctness guardrail:** no unexplained missing or duplicate on-time intent IDs against the frozen oracle.
- **Latency guardrail:** proposed seven-day 99%-within-60-seconds SLI in Section 8, measured end to end.
- **Late-data guardrail:** set an approved maximum missed-pair rate from adjudicated late reconciliation before pilot.
- **Customer guardrail:** measure avoidable contact or adverse action, even though this design does not initiate those effects.
- **Fairness guardrail:** assess error and exclusion rates by permitted, sufficiently populated cohorts under privacy review.
- **Availability guardrail:** restore, drain, and reconcile within organization-approved recovery objectives.
- **Outcome:** not executed; no pass, fail, confidence interval, or observed lift is available.

### 9.2 Test design

- Begin with a chronological offline replay of a representative authorized trace.
- Freeze rule definitions, source snapshots, matching semantics, baseline, and metrics before final evaluation.
- Verify candidate output against an independently implemented batch oracle at the same data cut and rule frontier.
- Follow with shadow execution that emits no operational cases and compares output manifests.
- Measure arrival delay, late-event rate, watermark stalls, rule-frontier lag, and state footprint in the shadow period.
- Run a controlled canary only after case-store idempotency and operations approval.
- If evaluating causal business benefit, randomize at card or other interference-safe unit rather than at individual alert.
- Maintain consistent treatment assignment for a card throughout the experiment.
- Prevent the same card's alerts in treatment from changing baseline-arm review or card behavior without accounting for interference.
- Record eligibility before assignment, assignment ratio, actual exposure, exclusions, and sample-ratio mismatch checks.
- Define how shared reviewers are allocated across arms before launch; unbalanced staffing can confound treatment effects.
- Use a prespecified observation window and stopping rule rather than optional early stopping on favorable estimates.
- Independently sample non-alerts for fraud adjudication to estimate missed cases and verification bias.
- Preserve an untouched later time period for a final evaluation after threshold tuning.
- Cluster uncertainty by card, and use time blocks or a suitable hierarchical approach when daily shocks create dependence.
- Calculate power or minimum detectable effect using realistic baseline rates and cluster structure once data exists.
- Report unresolved labels and perform sensitivity analyses rather than treating every unresolved case as legitimate.
- If observational shadow comparison is the only feasible design, label it noncausal and discuss selection and intervention differences.
- The experiment could fail because the rule adds no incremental cases, creates excessive workload, or has unacceptable late-event loss.

### 9.3 Correctness and operational test matrix

- Test a valid small/large pair well inside the two-minute interval against the independent oracle.
- Test a second event at exactly 120,000 milliseconds and require inclusion.
- Test a second event at 120,001 milliseconds and require exclusion.
- Test identical timestamps and require exclusion because order is strictly later.
- Test a pair across a version boundary using the first event's version.
- Test a post-boundary small event and reject starting it under the old version.
- Test two small events and two large events, verifying every valid pair is emitted exactly once.
- Test event-ID duplicate with identical payload and confirm one logical input.
- Test event-ID duplicate with different payload and confirm conflict quarantine.
- Test out-of-order arrival before watermark and verify identical result to sorted input.
- Test an arrival at or behind the current watermark and verify reconciliation routing.
- Test a resumed idle partition that emits an older event and verify late routing.
- Test a rule update arriving after a transaction but before the committed frontier permits release.
- Test a rule sequence gap and confirm that processing stops rather than selecting a stale default.
- Test prohibited retroactive publication and confirm rejection and security audit.
- Test a frontier stall and measure buffer growth, backpressure, alerting, and recovery.
- Kill a task before checkpoint completion and compare restored intents to the oracle.
- Kill a task after Kafka commit but before case consumer acknowledgement and confirm one case.
- Inject case API timeout after commit and verify a retry returns the existing case.
- Inject same alert ID with conflicting digest and confirm a visible incident, not overwrite.
- Restore from an older checkpoint and reconcile duplicates and any source-retention gap.
- Rescale with active partial matches and compare match manifests before and after restore.
- Test schema-compatible and incompatible rule, event, and state changes with rollback procedures.
- Load test hot-key pair explosion and ensure declared overload behavior preserves accounting.
- Test Kafka transaction expiry against checkpoint pause, restart delay, and broker limits.
- Run the end-to-end late correction workflow without sending unapproved customer communication.
- Every row above is a proposed test, not evidence that a test has passed.

## 10. Action Taken

- **No action taken as of 2026-11-07:** this is an illustrative design exercise without an implementation or organization approval.
- **Recommendation:** obtain the input and rule contracts, implement a bounded shadow prototype, and conduct the Section 9 evaluation.
- **Approver:** proposed fraud strategy owner, operations owner, security owner, and platform owner.
- **Approval status:** not recorded; no production deployment is authorized by this document.
- **Implementer:** proposed stream platform team with case-service and rule-publisher owners.
- **Handoff:** a versioned design, data contract, runbook, replay oracle, and independently reviewed test results would be required.
- **Effective scope:** none; no region, customer, or deployment date has been approved.
- **Release:** isolated shadow first, then an approved canary only if all gates pass.
- **Stop:** pause release if the rule frontier becomes untrustworthy, reconciliation fails, reviewer capacity is exceeded, or an agreed error budget is exhausted.
- **Rollback:** stop new intents, drain or preserve durable queues, restore a compatible binary/state, and reconcile already committed cases.
- **Excluded:** automatic card block, unsupervised contact, hidden reprocessing of history, and any claim of achieved uplift.

## 11. Outcome & Results

**Source declaration:** figures in Sections 2 and 6 are constructed teaching values, not measurements of any organization or evidence of realized impact.

| Metric | Baseline | Treatment / after | Change | Uncertainty | Evidence / limitation |
|---|---|---|---|---|---|
| Confirmed fraud cases reviewed on time per issuer-day | Not measured | Not measured | Not available | Not available | CLM-01; no evaluation cohort |
| Review cases per day | Not measured | Not measured | Not available | Not available | No staffing or case audit |
| Live on-time pair completeness | Not measured | Not measured | Not available | Not available | No replay manifest or sink reconciliation |
| End-to-end 60-second SLI | Not measured | Not measured | Not available | Not available | No deployment or telemetry |
| Customer harm and late-event loss | Not measured | Not measured | Not available | Not available | No mature label or repair study |

- Outcome not yet measured.
- The proposed owner is the evaluation lead, with operations validating review capacity.
- The proposed observation is a four-week shadow cohort followed by 90 days of label maturation.
- A causal pilot, if approved, needs its own prespecified sample size and stopping rule.
- Decision trigger: review the complete Section 9 results before any production case creation.
- No adoption, cost saving, incident reduction, or sustained duration is established.

## 12. Failure Modes & What Went Wrong

No incident is asserted; all failure modes below are hypothetical failure investigations.

### F-01: Transaction watermark advances too far

- Trigger: one producer emits future-dated timestamps.
- Symptom: ordinary events from other producers become late and live matches drop.
- Diagnostic: compare event time, ingestion time, and per-partition watermark advances.
- Competing explanation: a genuine delayed source partition can produce the same late-event symptom.
- Blast radius: every key sharing the affected downstream watermark path.
- Containment: quarantine future-skewed records and stop unsafe rollout.
- Prevention: producer timestamp bounds and per-partition anomaly alerts.
- Owner: ingestion platform.
- Verification: replay skewed timestamps and confirm intended late routing.
- Limitation: an overly strict bound can exclude valid events and needs source-specific calibration.

### F-02: Rule message races a transaction

- Trigger: a transaction arrives before an effective rule update.
- Symptom: current-rule lookup yields inconsistent versions across replay.
- Diagnostic: compare rule-log sequence and committed frontier with transaction timestamp.
- Competing explanation: a missing version rather than a network race.
- Blast radius: all pairs around a version boundary.
- Containment: hold output until the coverage frontier is valid.
- Prevention: immutable rules, a single ordered log, and monotonic finality.
- Owner: rule platform.
- Verification: reverse arrival order in replay and compare rule-version manifests.
- Limitation: the frontier is a custom contract and needs implementation evidence.

### F-03: Checkpoint recovery duplicates external effects

- Trigger: case insert commits but the consumer fails before acknowledging the message.
- Symptom: the alert intent is delivered again.
- Diagnostic: inspect the case transaction, intent key, digest, and consumer offset.
- Competing explanation: two legitimately distinct matching pairs share a card.
- Blast radius: repeated customer reviews or notifications if uniqueness is absent.
- Containment: stop the consumer and compare case IDs before replay.
- Prevention: atomic unique intent key, payload conflict detection, and transactional outbox.
- Owner: case-service team.
- Verification: force a timeout after commit and verify one case and one outbox identity.
- Limitation: an external notification provider also needs its own idempotency contract.

### F-04: Late earlier event changes a sequence

- Trigger: an earlier small authorization arrives after the watermark.
- Symptom: the live rule misses a match that full historical sorting finds.
- Diagnostic: compare the late queue with the batch oracle and source offsets.
- Competing explanation: the event is invalid, corrected, or from an excluded channel.
- Blast radius: missed timely review for affected cards.
- Containment: record the missed candidate and route it to approved reconciliation.
- Prevention: tune bounded disorder against measured latency and loss tradeoffs.
- Owner: stream platform and fraud operations.
- Verification: inject a late first event and confirm no silent live-state rewrite.
- Limitation: reconciliation cannot retroactively restore the timeliness of an earlier intervention.

### F-05: Hot-card pair explosion

- Trigger: many small authorizations precede many large authorizations on one card.
- Symptom: one keyed subtask accumulates state and emits a disproportionate alert burst.
- Diagnostic: inspect per-key event rate, pending starts, candidate-pair count, and sink backlog.
- Competing explanation: connector throttling can cause similar source lag without a hot key.
- Blast radius: the affected subtask and downstream review queue.
- Containment: preserve source and intent records, backpressure, and invoke an approved operational response.
- Prevention: load-test worst-case cardinality and obtain explicit business approval before changing pair semantics.
- Owner: stream platform and fraud operations.
- Verification: generate a bounded hot-key trace and reconcile every expected pair.
- Limitation: splitting one key by arbitrary salts breaks exact per-card ordering without a new merge design.

### F-06: Review queue exceeds staffed capacity

- Trigger: candidate and baseline alert sets are unioned without accounting for overlap.
- Symptom: queue age rises despite acceptable stream latency.
- Diagnostic: compare unique case arrivals, reviewer completions, and available staffing.
- Competing explanation: unusually slow case handling rather than higher arrival rate.
- Blast radius: delayed investigation for both candidate and baseline cases.
- Containment: activate the approved queue-prioritization and escalation procedure.
- Prevention: joint capacity allocation and continuous queue-age monitoring.
- Owner: fraud operations.
- Verification: simulate a sustained arrival rate above service capacity.
- Limitation: dropping alerts for capacity reasons changes the evaluated policy and must be recorded.

### F-07: State migration cannot restore

- Trigger: deployment changes serializer or operator identity incompatibly.
- Symptom: restore fails or state cannot be mapped to the upgraded operator.
- Diagnostic: inspect savepoint compatibility, serializer snapshots, and operator IDs.
- Competing explanation: missing checkpoint storage permissions.
- Blast radius: all active card sequences and ingestion progress.
- Containment: return to a compatible binary and retained checkpoint if possible.
- Prevention: pin stable IDs and perform a savepoint restore drill before release.
- Owner: platform release engineering.
- Verification: execute both upgrade and rollback with active partial matches.
- Limitation: an old binary may not understand newly written rule versions.

### F-08: Labels appear too favorable

- Trigger: only investigated alerts receive fraud adjudication.
- Symptom: observed precision looks high while missed fraud is unknown.
- Diagnostic: compare labeling probability across alerted and non-alerted populations.
- Competing explanation: truly better selection is possible but not established.
- Blast radius: a poor release decision and misallocated reviewers.
- Containment: do not report uninvestigated cases as legitimate.
- Prevention: independent non-alert sampling and maturity tracking.
- Owner: evaluation lead.
- Verification: report weighted estimates and sensitivity to unresolved cases.
- Limitation: small or delayed samples may leave the decision inconclusive.

## 13. Transferable Lesson

- A stream-processing result is only as reproducible as its event identities, ordering rule, rule history, and recovery boundary.
- Event time and broadcast state solve different problems: one orders data by occurrence, the other distributes configuration.
- Neither mechanism alone defines which configuration was authoritative for a historical event.
- A version frontier turns a policy decision into a checkable control-plane contract, at the cost of waiting when control data stalls.
- A checkpoint makes internal processing recoverable only within its source and state assumptions.
- External human-facing side effects require their own durable identity and uniqueness guarantee.
- Reviewer capacity is part of correctness for a fraud workflow because unreviewed alerts do not deliver the intended benefit.
- This design transfers to other keyed sequences when the ordering, effective-time policy, and safe late-data action are explicit.
- It is a poor fit for a rule whose topology changes continuously unless dynamic-state semantics are separately designed.
- It is also a poor fit when the business requires a synchronous authorization decision faster than the event-time waiting budget.
- In that setting, use a separately evaluated synchronous decision path and treat this stream as delayed investigation support.
- The lesson is a design hypothesis until correctness, business outcomes, and operations are actually measured.

## 14. Connections & Cross-References

- [Collection index](INDEX.md) maps this proposal to the local streaming case collection.
- [PayPal observability migration](01_paypal_observability_dataflow.md) contrasts an attributable migration narrative with this hypothetical fraud design.
- SRC-CEP explains pattern operators, event-time lateness, and match skip strategies.
- SRC-BROADCAST explains the broadcast-state API and the requirement for deterministic update logic.
- SRC-TIME explains event time and watermark progress, including the minimum over inputs.
- SRC-CHK describes recoverable state; an external side-effect contract remains necessary.
- SRC-KAFKA describes connector behavior that must be checked against the actual connector version.
- The source inventory entry is topic provenance only; the unavailable original case is not represented as quoted or summarized.

### Interview pack: questions, answers, and follow-ups

#### Q1. What is the actual business decision?

- Decide whether a small-then-large event sequence should be added to, or replace, a large-amount review policy.
- The output is a human review case, not an automatic decline.
- Success is incremental confirmed cases reviewed on time under the shared staffing constraint.
- A follow-up should ask which fraud cases the candidate misses relative to the baseline.

#### Q2. Why is event time useful here?

- The sequence depends on when authorizations occurred rather than on when a consumer happened to process them.
- Buffering and watermarks let moderately out-of-order records be evaluated in event-time order.
- Waiting increases latency and still cannot guarantee that no older record will arrive.
- A follow-up should specify the late-record route and quantify the resulting loss of timely coverage.

#### Q3. What does a watermark mean?

- It is a progress declaration that records at or before a given timestamp are no longer expected on that stream.
- A record can violate that declaration, so the watermark is not a proof of completeness.
- Multi-input operators generally advance using the minimum input watermark.
- A follow-up should explain stalled rule-stream progress and the consequences of declaring a partition idle.

#### Q4. Does broadcast state make rule activation deterministic?

- No: transactions and broadcast updates are independent streams and can arrive in either order.
- The proposed committed frontier says which event-time intervals can no longer be changed.
- Transactions wait until their timestamp is strictly below that frontier.
- A follow-up should test publisher failover and prohibited retroactive changes.

#### Q5. Which rule applies across a version boundary?

- The version effective at the first event time governs both conditions in the pair.
- The second event is made available to the old version's in-flight candidates as well as relevant newer versions.
- A post-boundary first event cannot open a match under the old version.
- A follow-up should compare the `(A,B,v1)` and `(C,D,v1)` examples in Section 6.

#### Q6. How is the two-minute boundary defined?

- The second timestamp must be strictly later and at most 120,000 milliseconds after the first.
- Exactly 120,000 milliseconds is included; 120,001 is excluded.
- The chosen CEP implementation must be tested at that boundary rather than relying on an assumed API interpretation.
- A follow-up should explain how timers avoid removing state before an equal-boundary record is processed.

#### Q7. What does the pattern output when several starts precede several ends?

- The declared contract emits every qualifying ordered pair, subject to the same rule version.
- Relaxed nondeterministic contiguity and a compatible no-skip strategy are needed for the fixed CEP implementation.
- Candidate explosion is a real cost, not a duplicate that can be removed without changing semantics.
- A follow-up should ask whether the business would instead approve a first-match or per-card case policy.

#### Q8. What is checkpoint exactly-once, and what is it not?

- A completed checkpoint permits recovery of compatible operator state and input positions under the configured source and storage contract.
- A properly configured transactional sink may make Kafka intent publication exactly once within its documented boundaries.
- Neither automatically makes an arbitrary case database insert or customer notification exactly once.
- A follow-up should trace the timeout-after-case-commit failure.

#### Q9. How do you prevent duplicate case creation?

- Derive a stable ID from issuer, rule/version, and ordered first/second event IDs.
- Atomically enforce its uniqueness in the case store and compare a payload digest on conflict.
- Commit the case and a notification outbox row together; make further side effects independently idempotent.
- A follow-up should ask what happens when a replay occurs after deduplication retention expires.

#### Q10. How do corrections differ from ordinary duplicates?

- An identical ID and identical payload is a delivery duplicate.
- An identical ID and different payload is a conflict; it must not silently overwrite the original.
- A legitimate correction is a separately identified record linked to the original and processed by an audited repair policy.
- A follow-up should explain how an already reviewed case is amended without erasing its history.

#### Q11. What is the review-capacity conclusion in the worked example?

- The baseline has 180 cases and the candidate 200; each fits individually in 200 slots.
- With 100 overlapping cases, their union is 280, which exceeds capacity by 80.
- Therefore, deploying both without an allocation policy is infeasible in that example.
- A follow-up should distinguish a replacement policy from adding only 20 incremental slots.

#### Q12. Compute the hypothetical candidate precision and false-positive load.

- The candidate has 50 confirmed and 150 legitimate cases out of 200.
- Precision is `50 / 200 = 25%`; false positives are 150 cases, or 75% of candidate alerts.
- Baseline precision is `36 / 180 = 20%`, a five-percentage-point difference.
- A follow-up should explain why these constructed counts do not demonstrate a measured uplift.

#### Q13. How do you detect a hot key?

- Compare per-key event rate, active partial matches, output cardinality, and per-subtask lag.
- A single card can dominate one subtask even if global throughput appears small.
- Blindly salting the card key breaks the sequence unless an exact merge is separately designed.
- A follow-up should calculate the 1,200 pending starts and possible 12,000 pair-per-second burst.

#### Q14. What would falsify the design?

- Replay may show that late-event misses exceed the agreed budget or the candidate misses important baseline cases.
- Failure injection may show that rule finality or case-store idempotency does not hold.
- A capacity-controlled evaluation may show no incremental confirmed cases within the review deadline.
- A follow-up should describe which alternative would be selected under each failure.

#### Q15. How would you evaluate causality and avoid label bias?

- Freeze the policy and randomize at a card or other unit that avoids cross-arm interference where feasible.
- Allocate shared reviewers consistently and use equal label-maturity windows.
- Sample non-alerts for independent adjudication and report unresolved cases.
- A follow-up should discuss clustering, minimum detectable effect, and operational spillovers.

#### Q16. What happens during rollback after a rule schema change?

- Stop new effect creation and preserve the intent queue and current checkpoint.
- Restore only a binary compatible with the state and rule versions actually present.
- Already committed cases are reconciled or resolved, not erased by rollback.
- A follow-up should explain the rebuild path when the older binary cannot read newer state.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Payment-card fraud operations and streaming review decisions. |
| Problem type | Continuous event-sequence detection and allocation of human review capacity. |
| Method | Proposed event-time Flink CEP with versioned broadcast rules and idempotent cases; compare with a single-authorization amount rule. |
| Primary result | Not measured; no deployment, experimental effect, or operational benefit has been established. |
| Key lesson | Deterministic rule finality and external effect identity complement event-time processing, but require validated control-plane and sink contracts. |
| Tags | `recoverable-stream-processing`; `partition-aware-scaling`; `versioned-rule-consistency`. |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-2026-STR002 / 1.0 / 2026-11-07. |
| Evidence level | `not-measured`; examples and calculations are illustrative, and Apache sources support framework semantics only. |
| Implementation stage | `proposed`; document is draft and unreviewed. |
| Baseline / primary metric | Amount-only review rule; incremental confirmed fraud cases reviewed on time per eligible issuer-day over the evaluation cohort. |
| Key limitation | The original listed source contents and real workload, outcome, label, rule-finality, and recovery evidence are unavailable. |
| Review trigger / owner | Change to Flink semantics, rule publication, event-time policy, or case-effect interface / fraud platform lead. |

- **Decision:** evaluate a sequence-based fraud review alert against a single-authorization baseline.
- **Evidence:** illustrative proposal; original listed source contents unavailable; no deployment or outcome measured.
- **Key:** issuer and card token; stable issuer-scoped authorization IDs.
- **Pattern:** small approved USD authorization followed by large approved USD authorization within `(0, 120000]` milliseconds.
- **Ordering:** event time; late records go to durable reconciliation.
- **Rule selection:** first-event effective version, released only after a committed rule frontier passes its timestamp.
- **Update boundary:** second events can complete in-flight matches under an older version.
- **State:** event-ID deduplication, ordering buffer, versioned CEP partial matches, broadcast rule history, and timers.
- **Effect:** stable pair/rule intent ID, atomic case-store uniqueness, conflict detection, and transactional outbox.
- **Capacity:** 180 baseline plus 200 candidate with 100 overlaps implies 280 unique cases, not 380 or 200.
- **Risk:** source lateness, frontier stalls, hot keys, state migration, replay, review overload, and biased labels.
- **Validation:** independent oracle, rule-race and boundary tests, failure drills, mature labels, capacity-controlled comparison.
- **Decision today:** no production release; obtain contracts and measure before deciding.
