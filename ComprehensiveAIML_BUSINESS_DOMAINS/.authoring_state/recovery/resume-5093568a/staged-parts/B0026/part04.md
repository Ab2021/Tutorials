## 8. Failure Modes & Edge Cases

Each entry below is a HYPOTHETICAL teaching scenario derived from the contract in §4, not an observed
incident.
Entries are ordered by the plane where the defect originates; the first three are the ones that
convert an ordinary error into a money-losing one.

### ISSUE-001 - Wildcard target expansion [VIOLATED_ASSUMPTION]
**Symptom.**
A call that passed review against one resource affects a set: a delete removes 340 rows where the
request named one, or a notification sends to every subscriber on a list.
The arguments look narrow because the model supplied a friendly identifier such as `team:platform`.
**Competing causes.**
(a) The resolver expands a group identifier into its members at execution time while approval was
granted on the literal string. (b) The schema permits `["*"]` or an omitted filter that means "all".
(c) The preview shown to the approver resolved a different group version than the executor did.
**Discriminating check.**
Log the *resolved* target list with its size and hash at admission, and compare it to the size and
hash attached to the approval. If the counts differ the resolver changed; if the counts match but the
approval hash differs, the approver saw a different expansion. A single-call reproduction against a
staging sink with the same group identifier settles (a) versus (b) in one step.
**Correction.**
Reject the batch, and re-issue per-resource keys for the resources the approver actually saw.
**Prevention.**
Require an explicit bounded list in the schema (`maxItems`), forbid `"*"`, and resolve groups at
approval time rather than at execution time so the approved object contains the concrete identifiers.
**Verification.**
A contract test that submits a group identifier and asserts the admission response contains the
same expansion hash the approval flow produced.

### ISSUE-002 - Lost response coerced to failure [CONCEPT_CONFUSION]
**Symptom.**
The ledger records `REJECTED` or `FAILED`, the customer is told the refund did not happen, and the
sink shows it did. Later reconciliation finds a second refund issued with a fresh key.
**Competing causes.**
(a) The client library maps every timeout to an exception and the handler maps every exception to
`FAILED`. (b) The state enum has no `UNKNOWN`, so the code picks the nearest value. (c) A retry
policy treats transport errors as idempotent-safe and re-dispatchs under a new key minted by the
retry decorator.
**Discriminating check.**
Query the sink by the operation key after the timeout. If the sink returns a committed record, the
client's classification was wrong and cause (a) or (b) applies. If the sink has no query-by-key
endpoint, then the ledger alone cannot resolve it - that absence is the root cause and it is
architectural.
**Correction.**
Introduce `UNKNOWN` as a first-class durable state, stop the retry loop at the attempt cap, and run a
reconciler that queries by key.
**Prevention.**
Make `FAILED` reachable only from an explicit sink rejection payload, never from a transport
exception; put the mapping in one function with a table-driven test over every error class.
**Verification.**
Injected fault test: kill the response path after the sink commits and assert the terminal ledger
state is `UNKNOWN`, the caller receives a typed non-final error, and a reconciliation job resolves it
without a second effect.

### ISSUE-003 - Key minted inside the retry loop [SCALE_FAILURE]
**Symptom.**
Duplicate suppression works perfectly in unit tests and fails only under load, when the ledger shows
several committed rows for the same business intent with keys differing in a suffix.
**Competing causes.**
(a) The retry decorator generates the key per attempt. (b) The planner re-plans after a timeout and
re-emits the tool call as a new proposal, receiving a new key legitimately. (c) Two orchestrator
replicas both believe they own the intent and each mints a key.
**Discriminating check.**
Inspect the key's structure: if duplicates share a prefix and differ by attempt counter, it is (a);
if they differ entirely and separated by a planning round trip, it is (b); if they were minted within
milliseconds on different hosts, it is (c). Ledger timestamps plus host identifiers discriminate all
three without reproducing under load.
**Correction.**
Derive the key from the intent identity - `hash(intent_id, tool, canonical_payload_version)` or a
UUID minted when the intent is created - and pass it into the transport layer as an immutable field.
**Prevention.**
Make the key a required argument of the executor interface so the transport cannot invent one, and
assert in the planner that repeated planning of the same intent reuses the key.
**Verification.**
A test that fails the first two attempts at the transport layer and asserts the ledger contains
exactly one row whose `attempt_count` is 3.

### ISSUE-004 - Digest omits a field that redirects the effect [VIOLATED_ASSUMPTION]
**Symptom.**
A retry of a refund with the same key and the same amount succeeds against a *different* destination
account, because the destination was added to the schema after the digest was defined.
**Competing causes.**
(a) Digest coverage is incomplete by construction. (b) Canonicalisation normalises the destination
away (for example, lower-casing a case-sensitive account reference). (c) The destination is resolved
downstream from the order rather than passed in arguments, so two payloads genuinely have the same
effect-relevant fields.
**Discriminating check.**
Mutate each effect-relevant field in isolation and assert the ledger returns `CONFLICT`. Fields that
do not trigger a conflict are uncovered; that enumeration takes minutes and settles (a) and (b)
immediately. If all fields trigger conflicts, the change came from downstream state and is (c).
**Correction.**
Extend the canonical payload to include the resolved destination identifier, not the caller's string,
and bump the digest version with a migration that keeps old keys valid.
**Prevention.**
Derive the digest field list from the same schema fragment that the executor uses to build the sink
request, so adding a field cannot skip the digest.
**Verification.**
A generated test per schema property asserting `CONFLICT` on mutation, run in CI against the schema
registry so a new field without a digest entry fails the build.

### ISSUE-005 - Authority checked only at the host [CONCEPT_CONFUSION]
**Symptom.**
A principal whose role was revoked an hour ago still causes effects; or a caller influences the
executor to act on a resource outside the caller's tenant.
The classic shape is a service that holds broad credentials and acts on behalf of any caller whose
request reached it.
**Competing causes.**
(a) Policy caching with a long TTL at the host. (b) The executor trusts a scope claim in the request
body. (c) The host is compromised or the model is manipulated into claiming a different principal.
**Discriminating check.**
Revoke the principal and re-issue the same call directly to the executor, bypassing the host. If it
succeeds, the executor does not enforce authority (b). Then re-issue through the host with a
cache-busting policy version; if that fails, the cache TTL is (a). Compare the executor's credential
scope against the caller's grant to detect (c).
**Correction.**
Have the executor evaluate policy against the caller's forwarded identity and its own resource
resolution, with a short or event-driven cache invalidation.
**Prevention.**
Design the executor credential so it is *narrower* than any caller's rights (a per-tenant or
per-resource credential), so a confused-deputy request cannot exceed the tenant boundary.
**Verification.**
A negative test suite that calls the executor directly with a revoked identity and with a forged
scope claim, asserting denial with a typed policy error in both cases.

### ISSUE-006 - Stale snapshot admits two individually valid effects [NUMERICAL_EDGE]
**Symptom.**
A balance goes negative, a quota is exceeded, or a seat is double-sold - while every individual
decision was correct against the state it read.
**Competing causes.**
(a) Read-check-write across two workers with no conditional write. (b) The snapshot is cached with a
TTL that outlives the decision budget. (c) The sink's own check is disabled for "trusted" callers.
**Discriminating check.**
Replay the two calls concurrently against staging with a forced interleaving (a barrier between read
and write). If both commit, (a) is confirmed. Then check whether the snapshot timestamp precedes the
first commit by more than the TTL; that isolates (b). A sink-side audit log showing no condition
evaluation confirms (c).
**Correction.**
Move the invariant to the sink as a conditional write (`debit where balance >= amount`), or reserve
the amount with a TTL before dispatch.
**Prevention.**
Treat every `meaning` predicate that depends on a mutable quantity as requiring a sink-enforced
condition; document the invariant next to the schema so a reviewer can see it is not enforced in the
client.
**Verification.**
A concurrency test with N parallel debits summing to more than the balance, asserting exactly the
affordable subset commits and the rest receive a typed `INSUFFICIENT` error.

### ISSUE-007 - Retry storm against a permanent rejection [NUMERICAL_EDGE]
**Symptom.**
Queue depth climbs while error rates are flat; the same record appears at the head of the retry queue
hundreds of times, and the sink's rate limiter starts rejecting unrelated traffic.
**Competing causes.**
(a) A poison effect - the account is closed, the currency is unsupported, the record was deleted - and
the classification treats all errors as transient. (b) Backoff is exponential without jitter, so
retries synchronise into bursts. (c) A retry budget is per-attempt rather than per-intent, so the
aggregate rate grows without bound.
**Discriminating check.**
Group failures by error code and by payload hash. A single payload hash with an identical error code
across many attempts is (a). Attempt timestamps clustered at fixed multiples of the base delay
indicate (b). Attempt counts exceeding the configured per-intent cap indicate (c).
**Correction.**
Route permanent codes to a dead-letter state with the receipt, and apply full jitter to the remainder.
**Prevention.**
Classify errors at the adapter boundary into `RETRYABLE`, `PERMANENT`, `UNKNOWN` using an exhaustive
switch, and let only `RETRYABLE` consume the retry budget.
**Verification.**
A test that injects a permanent 4xx and asserts exactly one attempt, one dead-letter record, and no
further dispatch for that key.

### ISSUE-008 - Partial batch retried as a whole [NUMERICAL_EDGE]
**Symptom.**
A batch of 20 items commits 14, reports an error, is retried under the same key, and the 14 committed
items are rejected as conflicts while the 6 missing items never commit - or worse, the retry commits
all 20 under a new key and duplicates the 14.
**Competing causes.**
(a) Batch-level key with item-level failures. (b) Partial success reported as an error without the
committed item list. (c) The sink deduplicates the batch but not the items, so a new key duplicates
items.
**Discriminating check.**
Compare the sink's committed items against the batch's intended item list after the first attempt. If
the sink reports which items committed, (b) is fixable locally; if it does not, the design must
assume (a) and move to per-item keys.
**Correction.**
Split into per-item keys derived deterministically from `(batch_key, item_id)`, so a retry of the
batch replays each item's own receipt.
**Prevention.**
Specify in the contract whether a batch is an atomic semantic unit or a transport convenience; the
two require different key scopes and different error semantics.
**Verification.**
An injected partial failure at item 15 of 20, asserting that a retry produces zero duplicate effects
and that all 20 items are present exactly once.

### ISSUE-009 - Server annotation treated as an authorization fact [CONCEPT_CONFUSION]
**Symptom.**
A destructive tool executes without confirmation because the server described it as read-only, or a
tool that writes to an external system is cached as if it were a pure function.
No exception, no audit entry, no policy decision.
**Competing causes.**
(a) The client's consent logic gates on the annotation rather than on an effect class it assigned. (b)
The annotation was correct for version 1 of the tool and version 2 added a write. (c) A proxy
rewrote the tool descriptor.
**Discriminating check.**
Fetch the descriptor from the server directly and compare its annotations and version against the
pinned copy in the schema registry. A mismatch is (b) or (c). If they match, then the client's gate
reads the wrong field and (a) is confirmed by inspecting the gate's input.
**Correction.**
Assign the effect class in the registry, keyed by tool version, and treat server annotations as
advisory input to a human review, never as the gate itself.
**Prevention.**
Sign or pin descriptors and fail closed on an unrecognised tool version; require an explicit effect
class for every registered tool, with the default being W4.
**Verification.**
A registry test asserting that every tool has an explicit effect class and that changing a
descriptor's annotation does not change admission behaviour.

### ISSUE-010 - Approval replayed onto a changed payload [VIOLATED_ASSUMPTION]
**Symptom.**
An approved action executes with a different amount or recipient than the approver saw, typically
after a queue delay or a re-plan.
**Competing causes.**
(a) The approval binds to an action name rather than a digest. (b) The payload is re-derived at
dispatch from live state, so it drifts legitimately. (c) The approval store is keyed by conversation
turn and reused for a subsequent turn.
**Discriminating check.**
Compare the digest recorded at approval with the digest admitted at dispatch. A mismatch with an
identical action name is (a); a mismatch correlated with queue delay and a price or stock change is
(b); a match but an unexpected turn identifier is (c).
**Correction.**
Bind the approval to `(tool, canonical_digest, expiry)`, re-check at dispatch, and escalate on
mismatch rather than re-approving.
**Prevention.**
Make the approval token an input to the executor's admission predicate, so an effect without a
matching token cannot be admitted at all.
**Verification.**
A test that approves payload A, mutates one effect field to B, dispatches, and asserts a
`PAYLOAD_CHANGED` rejection with no sink call.

### ISSUE-011 - Federated keyspace collision across servers [SCALE_FAILURE]
**Symptom.**
A legitimate new ticket is rejected as a duplicate of a ticket created on a different tool server,
or two servers issue receipts for one key and reconciliation returns the wrong one.
**Competing causes.**
(a) The uniqueness constraint is `(tool_name, key)` while two servers expose the same name. (b) A
gateway rewrites tool names to aliases that collide. (c) Keys are truncated to fit a column, cutting
entropy below the T-05 safety line.
**Discriminating check.**
Group the ledger by tool name and inspect the server identity column: if rows share a key with
different servers, (a) applies. Compare declared names against aliases in the gateway config for (b).
Measure key length distribution - any truncation at 48 bits or below puts (c) in play.
**Correction.**
Extend the uniqueness domain to `(server_id, tool_name, key)` and preserve full key length, storing
the digest of the key rather than a prefix if column width is the constraint.
**Prevention.**
Namespace tools by server at registration, and make the alias layer explicitly map alias to canonical
`(server, name)` without collapsing distinct tools.
**Verification.**
A test with two mock servers exposing the same tool name and the same key, asserting both effects
commit and each receipt is retrievable from its own server.

### ISSUE-012 - Success returned before the receipt is durable [NUMERICAL_EDGE]
**Symptom.**
The caller is told the effect succeeded, the process crashes, and no receipt exists to reconcile with.
A retry then duplicates the effect because the ledger row was lost too.
**Competing causes.**
(a) The response is emitted before the ledger write commits. (b) The ledger write is asynchronous and
best-effort. (c) The sink commit and ledger write are in different transactions with a window between
them, and the crash lands in that window.
**Discriminating check.**
Compare the count of caller-visible successes against ledger rows with a terminal state over the same
window. A shortfall means the response preceded durability. Correlate crash timestamps with the gap
between sink commit and ledger write; a positive gap on the crash path confirms (c).
**Correction.**
Write the ledger row as `SUBMITTED` before dispatch and update it to `COMMITTED` with the receipt
before responding; on crash, `SUBMITTED` entries are the reconciliation work list.
**Prevention.**
Forbid "success" responses until a durable terminal record exists, and make the reconciler treat
stale `SUBMITTED` rows as its primary input.
**Verification.**
A crash-injection test that kills the process after the sink commits but before the response, then
asserts the reconciler finds the effect via the key and does not duplicate it.

**Cross-cutting observation.**
Nine of the twelve failures above are invisible to request-level monitoring, because the request
returned a plausible status.
The only observability that catches them is a join between the ledger, the sink's record of what it
committed, and the approval record - which means the three stores must share the operation key.
That shared key is the single most consequential design decision on this sheet.

**Takeaway.**
For each entry, the discriminating check is a comparison between two recorded things; if a design
cannot produce both records, the failure mode is undetectable by construction.

## 9. Complexity & Cost Analysis

**Control cost model.**
The following is the per-intent cost of each control, with unit prices as placeholders to be replaced
by the reader's own contracted rates.

```text
FORMULA [T-13]: c_total = c_e + c_ledger + p_retry*c_retry + p_recon*c_recon + p_appr*c_appr

WHERE  c_e       sink execution cost per intent [currency]
       c_ledger  amortised ledger write + storage per intent [currency]
       c_retry   extra sink cost of one duplicate attempt [currency]
       c_recon   cost of one reconciliation query plus operator review [currency]
       c_appr    amortised human approval cost per intent [currency]
       p_*       probabilities that the component is exercised, [0,1]
```

CONSTRAINT - costs are linear in intents and independent of payload size.
Both assumptions break for batch tools, where one admission covers many records, and for
LLM-assisted review, where cost tracks tokens rather than intents.

ILLUSTRATIVE substitution.
`c_e = 2.00`, `c_ledger = 0.000000011` (0.3411 USD/month for 14.83 GiB spread over 31.10M intents),
`c_retry = 2.00`, `p_retry = 0.087`, `c_recon = 0.15`, `p_recon = 0.0052` (11.27 terminal `UNKNOWN`
states out of 2160 intents), `c_appr = 0.00` for sub-threshold amounts.
`c_total = 2.00 + 0.000000011 + 0.174 + 0.00078 + 0 = 2.1748` USD per intent.
Against an unkeyed design at `p_retry = 0.087` with no suppression, expected sink spend is
`2.00 x 1.087 = 2.174` plus the duplicate-effect loss `0.087 x 42.50 x (1-0) = 3.6975` when every
duplicate reaches the customer, for a total of 5.872 - a factor of 2.7 on the same volume.

**Asymptotics.**
| Operation | Time complexity | Note |
|---|---|---|
| Schema validation | O(size of arguments) | linear in payload; bounded by `maxItems`/`maxLength` |
| Canonicalisation + digest | O(payload) | must not depend on map iteration order |
| Ledger claim | O(log n) | B-tree insert; hash index gives O(1) average |
| Receipt lookup by key | O(log n) | the reconcile path's hot query |
| Policy evaluation | O(rules) or O(1) with an indexed decision | resource attributes may add a join |
| Reconciliation sweep | O(stale submitted) | proportional to crash/in-flight count, not to volume |
| Retention purge | O(expired rows) | must be batched to avoid long write locks |

**Storage and retention economics.**
The ledger's dominant cost is rows, not bytes: at 512 bytes per row the storage line is trivially
small (14.83 GiB for 31.1M rows), while the *index* maintenance during purge is what hurts.
Purging 345,600 rows per day (4 intents/s) in one statement takes a write lock proportional to the
row count, so purge in chunks sized to keep lock time under the ledger's write-latency budget - a
10,000-row chunk at 8,000 rows/s is 1.25 s, which is already too long for a hot table.

**Latency budget.**
| Stage | ILLUSTRATIVE p50 | p99 | Dominated by |
|---|---|---|---|
| Schema validation | 0.2 ms | 1 ms | payload size, cold schema cache |
| Policy evaluation | 0.5 ms | 4 ms | external decision service |
| Ledger claim | 1.5 ms | 12 ms | index contention, synchronous replication |
| Sink dispatch | 120 ms | 900 ms | downstream service level |
| Receipt write | 1.5 ms | 12 ms | same as claim |
| Total | 123.7 ms | 929 ms | sink |

The lesson from the table is that the safety machinery costs low single-digit milliseconds at p50 and
about 28 ms at p99 - roughly 3 percent of the sink call - so "the ledger is too slow" is almost never
the real argument against it.

**Human approval economics.**
A reviewer at 45 seconds per decision handles 80 decisions per hour.
At 2160 intents per burst, gating all of them needs 27 reviewer-hours, which is why approval must be
thresholded by amount, novelty, or effect class.
The correct threshold is where `P(duplicate) x loss` equals `approval_cost x P(intent needs review)`;
below it, automated controls dominate; above it, review pays.

**Cost of the absence of controls.**
| Missing control | Failure rate driver | Expected loss per 2160 intents |
|---|---|---|
| No key | retries, replans | `187.83 x 42.50 = 7982.78 USD` at full loss per duplicate |
| Key with 94 percent detection | ledger gaps | 478.97 USD |
| Key plus reconciliation | residual `UNKNOWN` | 1.69 USD plus operator time |
| Key plus reconcile plus approval at W4 | residue plus review cost | 1.69 USD plus 27 reviewer-hours if unthresholded |

The two orders of magnitude between the first and third rows are the argument for the ledger, and the
last row shows why the approval threshold is the *second* decision, not the first.

**Takeaway.**
Price the controls in milliseconds and fractions of a cent, then compare to the duplicate-loss term;
the comparison usually ends the debate in one line of arithmetic.
