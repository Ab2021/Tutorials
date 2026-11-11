## 8. Failure Modes & Edge Cases

Each entry below is a distinct mechanism, not a variation on "monitor the model". They are ordered by
where in the corridor they bite. For every entry the useful question is the same: which observation
separates this cause from its nearest rival, and what would you look at first? All scenarios are
constructed teaching cases (HYPOTHETICAL) built from the stated contract; none asserts an observed
incident at any organisation.

### F-01 — Wildcard resolution widens the target set after approval [VIOLATED_ASSUMPTION]

**Mechanism.** The request legitimately contains a selector (`vendor_group: "SUP-*"`, `cost_centres:
["CC-1", "CC-2", ...]`, or a filter expression). The approval covers the selector as written. The
resolver, the sink, or a later state change expands that selector into a larger set than the approver
mentally enumerated, so one approved operation produces N effects, some outside the reviewed scope.

**Competing causes.** (a) The resolver expanded a legitimate wildcard. (b) The selector was legal but
matched unintended records because a naming convention changed. (c) The approval UI displayed the
selector rather than the resolved list, so nothing expanded — the approver simply never saw it. (d)
Pagination: the resolved list was longer than one page and only the first page was reviewed.

**Discriminating check.** Log both the declared selector and the canonical resolved target list at
admission, and compare cardinality with what the approval record contains. If the approval record holds
a selector string rather than an item list, cause (c) is confirmed regardless of what happened
downstream. If the approval holds a list, diff it against the dispatched list and the first divergence
identifies (a) versus (d).

**Correction.** Replace selector-shaped targets with explicit bounded lists at approval time, or make
expansion a two-phase operation: resolve, re-present the concrete list with a count, then approve. Cap
the permitted expansion cardinality per operation (for example, 50 items) and require a separate
approval above the cap.

**Prevention.** Treat any parameter whose cardinality is not 1 as a schema-level hazard. The contract test
is: submit a selector that resolves to the cap plus one item and assert the tool refuses before dispatch.

**Why it recurs.** Wildcards are convenient and every demo uses them. The convenience is real at E0 and
false at E2 and above, because the effect multiplies while the approval does not.

### F-02 — Response lost after a successful commit [NUMERICAL_EDGE]

**Mechanism.** The sink committed, the response was lost to a timeout, a proxy reset, a load balancer
idle-close, or a process death between `recv` and the durable write. The caller holds no outcome. The
natural next action — retry — is correct only if the key is reused and the ledger is consulted.

**Competing causes.** (a) The sink never received the request. (b) The sink received and is still
processing (async queue). (c) The sink committed and the response was lost. (d) The sink rejected and
the rejection was lost. All four look identical from the caller: a timeout.

**Discriminating check.** Query the sink on the key. `COMMITTED` selects (c); `NOT_FOUND` after the
sink's own processing window selects (a); `PENDING` or a queued job id selects (b); a stored rejection
with a reason selects (d). If the sink cannot be queried, the honest observation is "UNKNOWN", and the
only separating evidence is a later business-level check (does the invoice appear in the vendor's
statement next cycle).

**Correction.** Freeze the retry, promote the operation to UNKNOWN durably, and drive reconciliation.
Do not mint a new key: a new key converts an ambiguous outcome into a probable duplicate.

**Prevention.** Timeouts shorter than the sink's observed p99.9 commit time are *designed* to produce
UNKNOWN. Measure the sink's commit latency distribution and set the caller timeout above the point where
the ambiguity rate is acceptable, then make the ambiguity visible in metrics as a first-class outcome.

### F-03 — Same key, different payload [NUMERICAL_EDGE]

**Mechanism.** A retry or a replay presents key K with a payload whose canonical digest differs from the
stored one. This is not a duplicate to be suppressed and not a new operation to be created; it is
evidence that two different intentions claimed one identity.

**Competing causes.** (a) A UI edit changed the amount between attempts. (b) Canonicalisation is
non-deterministic: float formatting, key ordering, locale decimal separators, timestamp precision,
Unicode normalisation of a vendor name. (c) A default was applied by a mapper on one path and not the
other. (d) A genuine bug minted one key for two operations.

**Discriminating check.** Recompute the digest from the stored canonical payload bytes on read. If it
differs from the stored digest, the canonicalisation is broken, not the client (b). If it matches but the
incoming payload's digest differs, print both canonical payloads and diff them: a one-character diff in a
decimal field points at (b); a whole-field diff points at (a) or (c); two unrelated documents point at
(d).

**Correction.** Reject with `KEY_CONFLICT` and require a newly approved operation. Never "update" the
stored digest to match the new payload: that converts the ledger into a last-writer-wins cache and
removes the only duplicate detector you have.

**Prevention.** Canonicalise with a single serialiser used for both hashing and transport, and include a
canonicalisation version in the stored record so a format change is a migration rather than a silent
digest change. Freeze decimals as scaled integers, sort keys, and normalise strings before hashing.

### F-04 — Syntactically valid, semantically denied [CONCEPT_CONFUSION]

**Mechanism.** The call passes the schema and fails a business rule: period closed, budget exhausted,
duplicate business document, approval reference unknown, quantity below the order minimum. Callers that
see only a generic failure retry, which cannot help, and the retry load lands on the policy layer.

**Competing causes.** (a) A real policy denial. (b) A stale policy cache (the denial reflects state from
minutes ago). (c) The wrong tenant or cost centre was resolved because of an identifier confusion. (d)
A period boundary was crossed between admission and effect.

**Discriminating check.** Return and log a typed reason plus the `policy_version` and the authority time
used. Evaluate the same predicate again against current state: if the second evaluation allows it, (b) or
(d); if it denies with the same reason, (a); if it denies with a different reason, (c) is likely because
the resolved subject differs from the intended one.

**Correction.** Surface the typed reason to the caller and mark the operation non-retryable. Retry only
the class of denials that are explicitly time-based (period lock opening, budget top-up), and treat those
as scheduled re-submissions with fresh approval, not as backoff loops.

**Prevention.** Make `retryable` an explicit boolean field on the error type rather than something each
caller infers from prose or from a status code. A taxonomy where `SCHEMA_INVALID`, `AUTHORITY_DENIED`,
and `KEY_CONFLICT` are all `retryable: false` removes an entire family of client bugs.

### F-05 — Credential or data leakage through tool output [VIOLATED_ASSUMPTION]

**Mechanism.** A read tool returns a record containing a bearer token, a session cookie, a bank account
number, or a full personal record. That output enters model context, is summarised into a transcript, is
written to a debug log, and is later retrievable by an unrelated principal.

**Competing causes.** (a) The output projection includes secret-bearing fields. (b) An error path
serialises the entire request including headers and credentials. (c) A debug flag is enabled in
production. (d) A retrieved document (not tool output) contains a secret and is conflated with it.

**Discriminating check.** Scan the transcript and logs with secret-shaped patterns and compare against
the tool's declared output schema. Fields present in the payload but absent from the schema identify the
projection as the source. If the secret appears only in error records, (b) is the cause. If it appears
in a document body, (d).

**Correction.** Stop propagation: revoke anything exposed, purge the affected transcripts and logs by
policy, and narrow the projection to an allowlist of fields per tool rather than a denylist of known
secret names.

**Prevention.** Deny-by-default output projection is cheap to implement and hard to regress: a field not
listed in the output schema is not serialised at all. Pair it with a scan in CI over serialised fixtures
so that a mistakenly added field fails a test rather than a review.

### F-06 — Retry storm against a partially degraded sink [SCALE_FAILURE]

**Mechanism.** A sink slows to 60% of capacity. Clients see more timeouts, retry more, and the executor's
arrival rate rises while its service rate falls. The queue grows at the sum of both effects, and the
system enters a state where throughput is below the pre-incident level and cannot recover while retries
continue.

**Competing causes.** (a) A client retry policy without a budget. (b) Retries without jitter arriving in
phase. (c) A timeout set below the degraded service time, guaranteeing a timeout even for work that will
succeed. (d) Synchronous downstream calls inside the tool that multiply per-operation occupancy.

**Discriminating check.** Plot attempts per intent alongside successful receipts. Rising attempts with
flat receipts is (a) or (b). If attempts per intent is near 1 but queue depth still grows, the timeout
is too tight for current service times (c). If occupancy per operation rose without an attempt increase,
look for added synchronous calls in the tool implementation (d).

**Correction.** Engage a retry budget (for example, retries may not exceed 10% of prior-minute
successes), shed rather than queue beyond the budget, and raise the caller timeout for the duration of
the incident with a compensating reduction in attempts.

**Prevention.** Budget retries in the executor, not in each client SDK. A shared budget is enforceable;
five independent SDK policies are not. Add jitter by default and treat "no jitter" as a defect in review.

### F-07 — Version drift between client and server interpretations [SCALE_FAILURE]

**Mechanism.** A field changes meaning, units, or optionality. The client still sends the old shape; the
server accepts it under a permissive rule and applies a default. Both sides are internally consistent and
the effect is wrong: amounts in minor units read as major units, or an omitted `currency` defaults to the
server's home currency.

**Competing causes.** (a) A schema version mismatch during a rolling deploy. (b) A silent default applied
by the server. (c) A cached tool manifest describing a previous version. (d) Units changed without a
version bump.

**Discriminating check.** Record the negotiated schema version and the canonicalisation version on every
admission. A cluster of bad effects all reporting version v1 from a client that believes it is on v2
confirms (c). If versions agree but a field is absent from the request and present in the effect, (b).
If the magnitude is wrong by exactly 100, (d).

**Correction.** Reject unknown-version calls rather than applying defaults; replay the affected window
from the intent record after reconciliation, with a human check on the corrected amounts.

**Prevention.** Version the tool contract, not only the API. Make defaults explicit in the schema as
`required` with no fallback, and treat "server silently filled a field" as an audit event rather than a
convenience. Include the version in the canonical digest so a version change cannot reuse an in-flight
key.

### F-08 — Cross-tenant read or write through a shared cache or handle [CONCEPT_CONFUSION]

**Mechanism.** A cache key omits the tenant, a resource handle is guessable, or a "find by name" lookup
returns another tenant's record. The tool is functioning as designed; the design omitted tenancy from
the identity of the thing.

**Competing causes.** (a) Cache key missing tenant. (b) Server-side filter absent, with a client-side
filter as the only defence. (c) A shared service credential acting as a confused deputy. (d) A handle
that is globally sequential, so identifiers leak across tenants by construction.

**Discriminating check.** Issue the same logical query under two tenants and compare results and cache
hit keys. Identical objects returned for different tenants confirms (a) or (b). If the tool's own audit
shows the same principal subject for both tenants, (c). If the leak is only of existence, not content,
(d).

**Correction.** Deny and invalidate cached entries; enforce tenancy in the data access layer, not in the
tool handler, so every path inherits it.

**Prevention.** Make the tenant part of the resource identity in the schema (not an optional header) and
part of every cache key. Test with a matrix of two tenants by two tools asserting zero overlap; this is
one of the few tests that catches tenancy regressions before they reach production.

### F-09 — Approval detached from the payload that executes [VIOLATED_ASSUMPTION]

**Mechanism.** A reviewer approves an intention ("pay vendor V-88 for the Q3 order") in prose or via a
summary, and the executor runs a payload the reviewer never saw, because the model re-derived the
amount, the date, or the target between approval and dispatch.

**Competing causes.** (a) Approval stored as free text with no digest. (b) Approval bound to a digest of
an earlier payload version. (c) Re-derivation between approval and dispatch changed a field. (d) A
partial approval UI that displayed only summary fields.

**Discriminating check.** Compare the approved digest, the digest at dispatch, and the digest of the
canonical payload the sink received. Three-way mismatch isolates the stage: approval-to-dispatch
divergence is (c); approval carrying no digest at all is (a); dispatch-to-sink divergence means the
serialiser is not the canonical one.

**Correction.** Re-present the canonical payload for approval whenever the digest changes; treat a
changed digest as a new approval requirement, not an update.

**Prevention.** Bind approval to `(digest, approver, expiry, policy_version)` and verify it inside the
executor immediately before dispatch. A digest check that happens anywhere upstream of the last possible
mutation is theatre.

### F-10 — Deduplication window shorter than the retry and human horizon [NUMERICAL_EDGE]

**Mechanism.** The ledger retains keys for 24 hours. A reconciliation gap, a weekend, a support ticket,
or a delayed replay arrives on day three with the same key. The ledger says NOT_FOUND, the operation is
admitted as new, and the sink creates a second effect because its own dedup also expired.

**Competing causes.** (a) TTL shorter than the retry horizon. (b) Two different ledgers (per-region) each
retaining their own keys. (c) Clock skew making a record look expired. (d) A restored backup losing the
recent window.

**Discriminating check.** Compute the actual maximum age of any replayed key from the audit log; if it
exceeds the TTL, (a) is proven by arithmetic alone. If duplicates come from different regions, (b). If
expiry timestamps are inconsistent across replicas, (c).

**Correction.** Extend retention to the measured maximum replay age plus a margin, and for operations
whose sink retention is shorter than the human horizon, add a business-key check (vendor + amount + date
+ reference) as a second line.

**Prevention.** Derive the TTL from a stated horizon rather than a default: `W = max(retry horizon,
reconciliation lag p99, support resolution time) + margin`. Write that sentence into the tool's contract
so a future change to the retry horizon forces a TTL review.

### F-11 — Burst blindness: the average says fine, the queue says otherwise [SCALE_FAILURE]

**Mechanism.** Capacity is planned against a daily mean of 17 operations/s while the close-out hour
arrives at 25 operations/s and the recovery lane must also absorb the backlog. The plan is arithmetically
correct about the mean and structurally wrong about the shape.

**Competing causes.** (a) Under-provisioned peak capacity. (b) Correct capacity with no recovery surplus.
(c) A serialised bottleneck (one database row, one lock, one vendor API quota) capping effective μ below
the measured mean. (d) A retry amplification layered on top of the peak.

**Discriminating check.** Compare per-minute arrival and completion curves rather than daily totals. If
completion flat-tops below the provisioned rate, the bottleneck is (c) and adding workers will not help.
If completions track arrivals but the queue persists into the next hour, (b). If attempts per intent rises
during the peak, (d).

**Correction.** Create surplus during the drain (shed low-value arrivals first), and fix the serialised
bottleneck before adding workers.

**Prevention.** Plan with a load shape, not a mean: peak rate, peak duration, and drain surplus are three
numbers that belong in the launch review, and the drain formula should be recomputed whenever any of them
changes.

### F-12 — Compensation described as undo, executed as a second effect [CONCEPT_CONFUSION]

**Mechanism.** A runbook says "roll back the invoice". The tool issues a credit note. The credit note is
a new committed effect with its own approval, its own downstream consequences (the vendor reconciles it,
the tax report changes), and its own failure probability. Six sequential compensations with a 92% success
rate each leave a 0.92^6 = 0.606 chance that all succeeded, so there is a 39% chance of a partial state.

**Competing causes.** (a) The original effect was already consumed downstream, so compensation cannot
restore the prior state. (b) The compensating operation was not authorised, so it failed. (c) The
original and the compensation both succeeded but the downstream consumer applied them out of order. (d)
A compensation of a compensation was issued after a timeout, creating net drift.

**Discriminating check.** Compare the sink's transaction ledger, ordered by event time, against the
expected sequence. A missing compensation with a rejection reason is (b). Both present but the consumer's
derived balance is wrong is (c). Extra compensating entries are (d).

**Correction.** Treat compensation as a first-class operation with its own key, approval, and typed
outcomes; make partial compensation an explicit state with an owner, not an error to retry.

**Prevention.** Rename the concept in code and runbooks. A type called `Reversal` with a docstring saying
"this is a new effect and may fail" prevents the mental substitution that "Undo" invites. Model
compensation chains explicitly and alert on chains longer than two.

### F-13 — Injected tool description or retrieved text changing routing [VIOLATED_ASSUMPTION]

**Mechanism.** A tool's description, a retrieved document, or a resource annotation contains text that
changes which tool the model selects or which arguments it fills. Because descriptions are discovered at
runtime, an attacker who can influence a description or a shared resource influences routing.

**Competing causes.** (a) A malicious or compromised description in the manifest. (b) Injected text in a
retrieved document that the model treats as instruction. (c) Legitimate ambiguity in a description that
the model resolves differently across runs. (d) A stale manifest serving an older, permissive description.

**Discriminating check.** Pin the manifest digest and diff on change; an action change correlated with a
manifest digest change is (a) or (d). Replay the same request with retrieval disabled: if the action
stabilises, (b). Non-determinism with no text change points at (c) and should be measured as proposal
variance, not as an attack.

**Correction.** Allowlist the set of tools and arguments available in a given context, and ignore
description text as an authority signal. Rotate anything the description could have reached.

**Prevention.** Treat all discovered text as data with a provenance label, enforce the allowed action set
outside the model, and pin manifest versions with an explicit review on change. A description is
documentation for a human reviewer; it is not a control.

### F-14 — Typed failure collapsed into a generic error at a boundary [CONCEPT_CONFUSION]

**Mechanism.** The tool returns a precise reason, but an intermediate layer (a gateway, an adapter, an
SDK) maps everything to "tool error". Callers lose the `retryable` flag and apply a default policy, which
is almost always retry. A permanent denial becomes a retry loop, and a KEY_CONFLICT becomes a duplicate.

**Competing causes.** (a) An adapter that rethrows a generic exception. (b) An HTTP status mapping that
collapses 4xx into one code. (c) A client SDK that treats any non-200 as transient. (d) An orchestrator
that swallows the error and synthesises a new call.

**Discriminating check.** Capture the error object at each hop and compare type and fields. The first hop
where the reason becomes generic is the culprit. If the reason survives to the client but the client still
retries, the defect is in the client policy, not the boundary (c).

**Correction.** Make the error taxonomy structural: a closed enum with `retryable`, `remediation`, and
`owner` fields, carried unchanged through every hop. Enforce with a contract test that asserts the reason
code at the outermost boundary.

**Prevention.** Test the error path as carefully as the happy path. Most integration test suites exercise
success and one timeout, which is precisely why taxonomy collapse survives to production.

### F-15 — Audit record written before the effect, read as a receipt [NUMERICAL_EDGE]

**Mechanism.** The outbox row is written pre-dispatch for durability. Someone later reads
`outbox.status` and reports it as outcome evidence, or a dashboard counts outbox rows as successes. The
numbers are plausible and systematically optimistic by exactly the failure rate.

**Competing causes.** (a) Pre-dispatch and post-commit states stored in one column. (b) A dashboard
aggregating intents rather than receipts. (c) A reconciliation job that marks rows complete on timeout
"to keep the queue moving". (d) Duplicate counting because both the outbox row and the receipt are summed.

**Discriminating check.** Compare the count of `COMMITTED` rows against the sink's own count for the same
window. A constant ratio below 1 indicates (a) or (b); a ratio above 1 indicates (d). Rows with
`COMMITTED` but no sink transaction id indicate (c).

**Correction.** Separate intent state from outcome state in the schema (`dispatch_state` and
`outcome_state`), and never let a single enum span both. Recompute dashboards from outcome state only.

**Prevention.** Name fields for the rung they represent. A field called `status` invites exactly this
confusion; `dispatch_state` and `outcome_state` do not. This is a naming decision with measurable
consequences during incident review.

## 9. Complexity & Cost Analysis

Costs here are of four kinds: per-operation time added by controls, storage for dedup and audit,
cognitive cost of taxonomy and review, and the expected cost of the failures the controls prevent. The
last one is the only one that justifies the first three, and it is the one most often left unstated.

**Per-operation latency budget.** The controls are cheap individually and additive. For the E2 tool in
section 5, an ILLUSTRATIVE breakdown:

```text
COST MODEL [C-01]: added latency per operation
schema validation            2 ms
canonicalise + digest        1 ms
policy evaluation (cached)   4 ms
key/digest compare-and-insert 6 ms   (single-region durable write)
authority recheck at executor 3 ms
dispatch (network + sink)   285 ms
receipt persist             5 ms
TOTAL                       306 ms, of which 21 ms is control overhead (6.9%)
WHERE: all values are illustrative single-region measurements; cross-region durable writes multiply the
  insert cost by the round-trip, typically 30-120 ms.
INTERPRETATION: control overhead is small until the durable insert crosses a region boundary. That is the
  one line item worth engineering: co-locate the key store with the executor.
```

The design consequence is that the deduplication store should live in the same failure domain and region
as the executor, even if the audit store does not. Teams frequently share one database for both and then
pay cross-region writes on the hot path while paying for long retention they do not need on that path.

**Storage economics.** Two stores with different requirements, which is why they should not be one.

```text
COST MODEL [C-02]: dedup ledger vs audit log, one year at 17.361 ops/s mean
DEDUP: 7-day retention, 768 B/record, uniqueness index required
  records = 17.361 * 86,400 * 7 = 1,499,990 * 7 ≈ 10,500,000 (rounding to 17.361 exactly)
  bytes   = 10,500,000 * 768 = 8.06 GB
  write rate = 17.361/s = 1.5M writes/day against a unique index
AUDIT: 400-day retention, 1,200 B/record, append-only, no uniqueness
  records = 17.361 * 86,400 * 400 = 1,499,990 * 400 ≈ 600,000,000
  bytes   = 600,000,000 * 1,200 = 720 GB
INTERPRETATION: audit dominates bytes by ~90x while dedup dominates write amplification and index churn.
  Putting them in one store forces the audit volume through a unique index, which is the expensive part.
```

The operational reading is that the dedup ledger is a small, hot, indexed component and the audit log is
a large, cold, append-only one. Merging them is a common simplification that inverts the cost curve:
you pay uniqueness maintenance on 600 million rows instead of 10.5 million, and every audit insert
contends with the hot path.

**Review cost, and the throughput ceiling of a human gate.** A human gate has a service rate, and a
service rate below the arrival rate is not a control, it is a queue.

```text
COST MODEL [C-03]: human review lane
arrival needing review: 8% of 17.361 ops/s = 1.389 ops/s = 5,000 ops/hour
reviewer throughput: 4 minutes/op = 15 ops/hour/reviewer
reviewers required at parity: 5,000 / 15 = 333 reviewers
At 10 reviewers: capacity 150 ops/hour vs 5,000 arriving -> the queue grows 4,850/hour
INTERPRETATION: a "human reviews all exceptions" policy is only viable if the exception rate is far
  below 0.3% or the reviewer pool is implausible. Otherwise the gate must be selective by consequence.
```

This arithmetic is why approval thresholds exist and why they must be set by money or blast radius rather
than by "anything that feels risky". A gate at 8% of volume with four-minute reviews needs 333 reviewers;
a gate at 0.1% of volume needs about 4. The control that actually scales is the keyed executor plus
reconciliation, with humans reserved for the E3 tail.

**Expected cost of the failure the controls prevent.** Compare the cost of a control against the expected
loss it removes, and be explicit that the loss estimate is uncertain.

```text
COST MODEL [C-04]: duplicate-write exposure versus control cost
duplicates/day at c=0.985: 30 (section 5, step 6)
rework cost/duplicate: 45 EUR (illustrative blended)
annual rework: 30 * 45 * 250 = 337,500 EUR
undetected share: assume 25% of duplicates are found only at period close, at 900 EUR each
  undetected: 30 * 0.25 = 7.5/day -> 7.5 * 900 * 250 = 1,687,500 EUR/year
control cost: two engineer-weeks to raise coverage to 0.9999 + 11.6 GB storage + index write load
INTERPRETATION: the dominant term is the undetected share, and it is the term with the widest error bar.
  Even at a tenth of the assumed rate the control pays for itself in the first year.
```

The honest caveat is that the 25% undetected share and the 900 EUR figure are constructed, not measured.
The method transfers: separate detected from undetected duplicates, because detected ones have a known
unit cost and undetected ones have an unknown but larger one, and it is the undetected ones that justify
coverage investment.

**Cognitive cost of taxonomy.** A closed error taxonomy costs design time once and saves per-call
decision time forever. An open vocabulary of prose errors makes every client decide retryability by
string matching, which is exactly the kind of implicit coupling that fails silently when a message is
edited. Four reason codes with a `retryable` flag cost a few hours to agree on and remove an entire class
of duplicate-producing bugs.

**Where the complexity actually lives.** Three of the ten formulas in section 4 are single-line
arithmetic; the complexity is never in the math. It is in the placement of checks (executor versus
admission service), the durability of the record relative to dispatch, and the honesty of the outcome
states. A design that gets those three right and the arithmetic roughly right will be safe; a design that
gets the arithmetic exactly right and places the dedup store in the wrong failure domain will duplicate
under precisely the conditions it was built for.
