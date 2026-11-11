## 10. Key Theorems / Results Reference

Each result is stated in its strongest honest form, with the assumption that, when violated, makes it
false.
The "consequence" line is the design obligation the result imposes.

**R1 - No client-side exactly-once without sink cooperation.**
Statement: if the sink exposes no deduplication token and no query-by-client-key, then no client
protocol can distinguish "effect committed, response lost" from "effect never arrived".
Proof sketch: the client's only evidence is the set of responses it received; a lost response and a
lost request are observationally identical to the client, so both must be treated as `UNKNOWN`.
Consequence: the achievable property is exactly-once *where the sink stores a key*, and
at-least-once-attempt plus reconciliation elsewhere; a design claiming better is claiming information
it does not have.

**R2 - Idempotency requires a stable derived key, not a fresh identifier.**
Statement: a protocol that generates a new request identifier per attempt provides no duplicate
suppression, however many identifiers it records.
Because the ledger's uniqueness constraint is evaluated over the identifier, fresh identifiers ensure
the constraint never fires.
Consequence: the key must be derived from intent identity (or minted once at intent creation) and
passed down as an immutable argument; the transport must not be able to mint keys.

**R3 - Atomic compare-and-insert is necessary for the claim predicate.**
Statement: if the ledger claim is implemented as an upsert, `claim(k,h) = NEW` always holds, so
admission degenerates to "authority and shape", and every retry becomes a second effect.
Proof: upsert returns success regardless of prior state; the predicate's value is then independent of
history.
Consequence: implement the claim as `INSERT ... ON CONFLICT DO NOTHING RETURNING`, and treat a
zero-row result as `EXISTING`, then compare digests to separate replay from conflict.

**R4 - Recovery time is governed by surplus, not capacity.**
Statement: with a single queue, `T_drain = B/(mu_r - lambda_r)`; therefore any capacity increase that
does not enlarge `mu_r - lambda_r` leaves drain time unchanged.
Consequence: capacity planning for recovery should be expressed as surplus target (for example,
`mu_r >= 1.4 x lambda_r` at p99), not as a utilisation percentage.

**R5 - Retry inflow consumes service capacity, creating positive-feedback overload.**
Statement: if retries share the executor with first attempts and `rho` is the retry probability, the
attempt arrival rate is `lambda/(1-rho)`; the queue is stable only while `mu > lambda/(1-rho)`.
Rearranged, stability requires `rho < 1 - lambda/mu`.
At `lambda = 18`, `mu = 15` the threshold is `rho < -0.2`, i.e. *no* positive retry rate is stable
once the executor is already saturated; retries must therefore be budgeted and shed.
Consequence: impose a retry budget (for example, retries may not exceed 10 percent of first-attempt
admissions in a window) and shed retries before shedding first attempts.

**R6 - Digest coverage bounds detection multiplicatively.**
Statement: with per-field inclusion probability `c` and `k` independent effect-relevant fields,
single-field substitution detection is `c^k`; omitted fields are undetectable regardless of key
discipline.
Consequence: digest coverage is a testable property, and the test is mutation of each field.

**R7 - Approval without a digest binds a class of actions, not an action.**
Statement: an approval token `(tool, expiry)` admits any payload of that tool until expiry; the
admitted set has the cardinality of the payload space.
Consequence: approvals must carry the canonical digest, and the executor must compare it at dispatch;
this also makes the approval reusable only for replay of the identical effect, which is safe.

**R8 - Confused-deputy risk is bounded by executor credential scope, not by caller checks.**
Statement: if the executor's credential allows action A on resource set R, then any caller that can
reach the executor can cause A on any `r` in R that the executor's resource resolution yields.
Caller-side checks reduce the probability but not the bound.
Consequence: issue per-tenant or per-resource executor credentials so the bound equals the tenant
boundary; test by calling the executor directly with a forged scope.

**R9 - `UNKNOWN` must be representable, or it will be misrepresented.**
Statement: in any system where a sink commit can precede the loss of its response, the outcome set
has at least three terminal values (`COMMITTED`, `REJECTED`, `UNKNOWN`); a two-valued enum forces a
false classification with probability equal to the loss rate.
Consequence: include `UNKNOWN` in the durable state machine and give it an owner and a sweeper.

**R10 - Batch atomicity and independent-item delivery are incompatible.**
Statement: a single key over `m` items means either all items are replayed on retry (duplicating the
committed subset) or none are (dropping the failed subset); per-item keys are required for
independent delivery.
Consequence: decide the key scope from the business semantic unit, and state it in the tool contract.

**R11 - Key entropy requirements follow the birthday bound.**
Statement: with `n` keys over the retention window and `b` random bits, expected colliding pairs are
`n^2/2^(b+1)`; keeping this below 0.01 requires `b > log2(n^2/0.02)`, which for `n = 31.1M` gives
`b > 55.4` bits, so 56 bits is the first integer that meets the 0.01 target (0.0067 expected pairs);
55 bits gives 0.0134.
Consequence: 64 bits is the practical floor; 128 bits removes the question entirely at negligible
cost.

**R12 - Reconciliation is cheaper than compensation whenever the sink supports key lookup.**
Statement: reconciliation costs one read; compensation costs one new keyed write plus fees plus
operator time and may itself fail.
Consequence: build the read path first; compensation is for effects the sink cannot reverse, and its
cost should be budgeted per effect class rather than assumed away.

**Result-selection guide.**
| Question being asked | Applicable result | What it forbids |
|---|---|---|
| Can we guarantee one effect per intent? | R1, R2, R3 | claiming exactly-once without a sink key |
| Why is drain slower than planned? | R4, R5 | sizing recovery by capacity alone |
| Why do payload changes slip through? | R6 | partial digests |
| Is the approval meaningful? | R7 | name-scoped approvals |
| How bad is a leaked executor token? | R8 | trusting host-side checks |
| Why do duplicate payments appear after incidents? | R9 | two-valued outcome enums |
| Why did a batch duplicate 14 items? | R10 | one key over independent items |
| Why do legitimate calls get rejected as duplicates? | R11 | truncated keys |
| Why is reconciliation mandatory in the runbook? | R12 | compensation-first designs |

**Takeaway.**
R1, R5, and R9 are the three results that most often falsify a design in review, because each one
removes a capability the designer believed they had.

## 11. Interview / Exam Questions

Questions are ordered from definitional to diagnostic. Each has its own answer; several require
computation, and the numbers are checked against §4 and §5.

### Q01. What does a JSON Schema validate, and what does it deliberately not validate?
**Answer.** It validates the shape of the arguments against a pinned version: required properties,
types, enumerations, and numeric or array bounds. It does not validate intent, authority, freshness of
business state, or whether the effect is desirable. A refund of 4500 minor units for an order that
belongs to another tenant passes every schema check ever written.
**Why it matters.** Teams that equate "schema-valid" with "safe" place their only control at the
weakest layer, and then add a prompt instruction as the second layer, which is weaker still.

### Q02. A tool call times out with no response. Compute what the caller knows and what it must do.
**Answer.** The caller knows the request left the host and that no response arrived within the
deadline. It does not know whether the sink committed. If the ledger has a row for the key in
`SUBMITTED`, the caller must move it to `UNKNOWN` and schedule reconciliation by key; if the ledger
has no row, the caller knows nothing was dispatched and may re-admit (a fresh claim).
**Computation.** With a 3 s deadline and a cap of 4 attempts, the caller's worst-case wait is 12 s,
after which the state must be terminal - `UNKNOWN`, not `FAILED`.
**Diagnostic twist.** If the sink has no query-by-key endpoint, the answer changes from "reconcile" to
"escalate to a human with the operation key and the sink's transaction listing".

### Q03. Compute the drain time for a backlog of 360 intents with a recovery lane of 21 intents/s against 15 live intents/s, then recompute when retries raise effective arrivals to 19.565/s.
**Answer.** Naively `360/(21-15) = 60` s. Attempt-aware: arrival `19.565`, backlog becomes
`(19.565-15) x 120 = 547.83`, surplus `21 - 19.565 = 1.435`, so `547.83/1.435 = 381.8` s.
**Interpretation.** The retry inflow multiplied drain by 6.36 without changing any hardware.
**Follow-up.** At what `rho` does the lane stop draining? Surplus is zero when
`21 = 18/(1-rho)`, i.e. `rho = 1 - 18/21 = 0.1429`; above that, drain time is undefined.

### Q04. Under what conditions may two calls share an operation key?
**Answer.** Only when they are the same intended effect: same tool, same canonical payload, same
effect-relevant fields. The second call then returns the first call's receipt instead of executing.
If the digest differs, the correct answer is a `CONFLICT` error and a new approval - never a silent
overwrite.
**Counterexample.** Two refunds of 1000 units each on the same order are two intents and need two
keys, even though the amounts match; the distinguishing field is the intent identity of the planner's
decision, not the payload.

### Q05. Why must the executor re-evaluate authority rather than trust the host?
**Answer.** Because the host's decision is derived from context the model can influence: retrieved
documents, tool outputs, and prior turns. A host that trusts a scope claim in the request body has
converted a hint into a grant. Re-evaluation at the executor with server-side resource resolution
makes the executor credential - not the caller's claim - the actual bound on what can happen.
**Test.** Revoke a principal, wait past the host cache TTL, and call the executor directly with the
same request; success means R8 is violated.

### Q06. What is the difference between idempotent and replayable, and which does a client need?
**Answer.** Idempotent means the second call leaves the state unchanged after the first effect.
Replayable means the second call returns the *original receipt*. A client recovering from a lost
response needs the receipt to report an outcome; "nothing happened the second time" is not enough to
tell the customer what occurred. Implement replay by storing the receipt reference with the ledger row
and returning it on a digest match.

### Q07. When does a deterministic workflow beat an agent loop?
**Answer.** When the route and the actions are known statically - an approval flow, a nightly batch, a
fixed reconciliation - because a workflow's failure modes are enumerable and replayable, and its cost
is bounded. An agent loop wins when the route depends on discovered state (which of three systems
holds the record, which schema version the tenant exports).
**Selection test.** If you can draw the state machine before running the task, use the workflow; if
drawing it requires knowledge only the environment has, use the loop and constrain it with the
contract from §4.

### Q08. How would you test that duplicate suppression actually works under concurrency?
**Answer.** Three tests, each attacking a different mechanism. (1) Sequential replay: dispatch twice
with the same key and digest, assert one sink record and two identical receipts. (2) Concurrent
replay: two threads dispatch simultaneously, assert the ledger's unique constraint produces one
insert and one lookup, no deadlock, one effect. (3) Crash replay: kill the process after sink commit
and before the response, restart, reconcile, assert no second effect.
**Metric.** `d` from T-03 should be measured as `1 - duplicates_observed/duplicate_attempts_offered`,
over an injected-fault window, not estimated from unit tests.

### Q09. Explain confused deputy in one paragraph and give the cheapest structural fix.
**Answer.** A deputy with broad authority performs an action for a caller whose own authority is
narrower; the deputy's authority becomes the effective permission. The cheapest structural fix is to
narrow the deputy: issue the executor a credential scoped to the caller's tenant or resource, so even
a manipulated request cannot exceed that scope. The cheap-but-weak alternative - checking the caller's
rights at the host - bounds probability, not the actual permission.

### Q10. Compute the ledger size and the cost of 90-day retention at 4 intents/s with 512 bytes per row.
**Answer.** Rows `= 4 x 86400 x 90 = 31,104,000`. Bytes `= 31,104,000 x 512 = 1.5930e10`, i.e.
`14.83` GiB. At 0.023 USD per GiB-month the ledger costs 0.341 USD/month.
**Decision.** Compare with the 7982.78 USD loss of a fully unsuppressed burst: retention is not the
expensive part; the purge strategy and the index size are.

### Q11. Why is compensation not a rollback, and when is it unsafe?
**Answer.** Because the original effect is already visible to other principals - the customer saw the
refund, the warehouse received the order - so the only available action is a second effect that
offsets it, with its own key, approval, and failure modes. It is unsafe when the compensating action
is itself user-visible (a reversal notice), when fees are non-refundable, or when the offsetting
effect needs the original to still exist (cancelling a shipment that already arrived).
**Rule.** Prefer reserve-then-commit so the unsafe window is short, and use reconciliation for
`UNKNOWN` rather than compensating blindly.

### Q12. What belongs in a typed tool error taxonomy, and why does it matter to the caller?
**Answer.** At minimum: `INVALID_ARGUMENT` (client bug, no retry), `POLICY_DENIED` (with rule
identifier), `CONFLICT` (same key, different digest), `REPLAYED` (same key and digest, receipt
attached), `RETRYABLE` (transient sink condition), `UNKNOWN` (outcome undecidable), and
`DEADLINE_EXCEEDED`. The taxonomy matters because the caller's retry decision must be a function of
the code, not of the message text; a single `ERROR` code forces every caller to guess, and guesses
become duplicate effects.

### Q13. How does schema versioning interact with the digest and the ledger?
**Answer.** The digest is computed over the canonical payload as defined by a specific schema version,
so the version is part of the digest's meaning. Two options: include the version in the digest input
(a payload valid under v3 and v4 hashes differently, so a retry across a deploy conflicts), or
canonicalise to a version-independent form (safer, more work). Storing the schema version on the
ledger row makes the choice auditable.
**Failure to avoid.** Bumping the schema and recomputing digests for in-flight keys, which turns
every concurrent retry into a `CONFLICT`.

### Q14. Which audit fields make an incident review possible, and which are noise?
**Answer.** Necessary: operation key, tool name and version, principal, tenant, resolved resource list
with its hash, canonical digest, policy decision and rule identifier, attempt number, dispatch
timestamp, receipt reference, terminal state, and the reconciler's verdict. Noise: full prompt text,
model chain-of-thought, and raw payloads containing personal data.
**Test.** Can a reviewer answer "which effect, authorised by whom, against what, decided when, and
what did the sink commit?" from the audit record alone, without reading application logs? If not, a
field is missing.

### Q15. What changes invalidate this design and force a re-review?
**Answer.** A sink losing its idempotency-key support; a move to multi-region where ledger replication
lag exceeds the retry horizon; a new tool in effect class W4 or W5; a change to the approval flow's
digest binding; federation with a second server operator; and any change to the error taxonomy.
**Trigger discipline.** Each of these changes the *validity condition* of a formula in §4, not merely
a parameter, which is why they belong in an explicit re-review trigger rather than in a routine
version bump.

### Q16. A team reports 99.9 percent duplicate suppression and 30 refund duplicates per day. Diagnose the arithmetic.
**Answer.** Thirty duplicates per day at `N` intents/day implies `N x rho/(1-rho) x (1-d) = 30`. With
`rho = 0.08` and `d = 0.999`, the surplus fraction is 0.08696, so `N x 0.08696 x 0.001 = 30` gives
`N = 344,986` intents/day, i.e. 3.99 per second - plausible for a mid-size platform. If the team
believes `N` is ten times smaller (34,499 intents/day), then `d` is not 0.999 but 0.99 exactly
(`1 - 30/(34499 x 0.08696) = 1 - 0.0100`), and their
measurement is the defect: they most likely measured suppression among retries that reached the
ledger, excluding the fallback path that bypasses it.
**Discriminating check.** Count ledger rows per intent for the duplicated intents; two rows with
different keys proves the key-minting path (ISSUE-003), one row with two sink records proves the
response path (ISSUE-012).

**Takeaway.**
The questions above separate into three skills: classify the evidence available at a plane, compute
the exposure with the right formula, and choose the cheapest control whose guarantee matches the
effect class.

## 12. Quick Reference Card

| # | Rule | Number to remember | Failure it prevents |
|---|---|---|---|
| 1 | Mint one key per intent, never per attempt | `rho/(1-rho)` surplus attempts | duplicate effects on retry |
| 2 | Claim the key with compare-and-insert, keyed by `(server, tool, key)` | uniqueness over 3 fields | upsert silently re-admitting replay |
| 3 | Cover every effect-relevant field in the digest | `c^k` detection | payload substitution after approval |
| 4 | Represent `UNKNOWN` durably and reconcile by key | 3 terminal states | false `FAILED` and double refunds |
| 5 | Size recovery by surplus, not capacity | `T_drain = B/(mu_r - lambda_r)` | drain estimates off by 6x |
| 6 | Enforce authority at the executor with a narrow credential | executor scope = tenant | confused deputy |
| 7 | Bind approvals to the canonical digest with a TTL near dispatch p99 | `1 - exp(-gamma*T_tail)` | stale and replayed approvals |
| 8 | Budget retries as a fraction of first attempts | `rho < 1 - lambda/mu` | positive-feedback retry storms |
| 9 | Keep keys at least 64 bits and retain them past the retry horizon | `n^2/2^(b+1)` collisions | rejected legitimate calls |
| 10 | Choose one key per semantic unit: batch key for atomic, per-item otherwise | `(1-q)^m` batch success | partial-batch duplicates |

**Effect-class quick check.**
R0 read - no key, cache freely.
R1 PII read - audit, no key.
W2 reversible write - key, idempotent retry, receipt.
W3 compensable write - key, receipt, saga with a stated compensation.
W4 externally visible write - key, approval bound to digest, reconciliation.
W5 irreversible broadcast - approval, no automatic retry, human-verified terminal state.

**Three questions to ask any trace.**
What was the operation key, and was it minted once per intent?
What did the sink commit, according to the sink itself?
Which store proves authority was re-evaluated at the executor?

**Status.**
Draft for review; all arithmetic in §4, §5, and §9 is re-checked in this document, all workloads are
ILLUSTRATIVE, and no deployed integration, vendor behaviour, or measured company result is asserted.
Review trigger: any change to the sink's idempotency support, the error taxonomy, the approval flow's
digest binding, or the federation topology.
