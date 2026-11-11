## 10. Key Theorems / Results Reference

The results below are the ones worth committing to memory. Each is stated with its hypothesis, because a
result quoted without its condition is the most common way an experienced engineer produces a wrong
answer confidently.

**R-01 — Admission is only as strong as its weakest conjunct.** If admission is defined as the conjunction
of syntax, semantics, authority, and ledger novelty, then the probability of a wrong dispatch is at least
the probability that the weakest conjunct is evaluated against stale or absent evidence. *Hypothesis:*
the conjuncts are evaluated at the same point in time against the same version of state. *Consequence:*
when the conjuncts are spread across services with different caches, the useful strength of admission is
that of the stalest conjunct. This is the formal reason T-06 matters: an authority predicate evaluated
180 seconds before the effect contributes almost nothing when state changes every 2.5 minutes.

**R-02 — At-most-once dispatch requires durable pre-dispatch state; at-most-once effect requires sink
cooperation.** The ledger guarantees one dispatch per key. It cannot guarantee one effect per key when the
sink ignores the key, because between dispatch and sink commit there is a window the ledger does not
control. *Hypothesis:* crash-consistent durable write of the key before dispatch. *Consequence:* every
claim of "exactly once" must name which of the two guarantees it means. The honest phrasing for an opaque
sink is "at most one dispatch, at most one effect only if reconciliation confirms".

**R-03 — Duplicate effects are proportional to uncovered retries, not to retry count.** Restating T-02:
D = N q (1-c)(1-p_sink). *Hypothesis:* independent retries and uniform coverage. *Consequence:* improving
coverage has proportional effect at any retry volume; reducing retries is multiplicative on top. A design
review that discusses retry policy without discussing coverage is discussing the smaller lever.

**R-04 — A backlog drains only at the surplus, and never at the nominal rate.** From T-03, T_drain = B /
(μ_r - λ_r), which is finite only for positive surplus. *Hypothesis:* constant rates per interval.
*Consequence:* "we added capacity equal to the peak" produces zero surplus and an unbounded queue. The
correct move under sustained overload is to reduce λ_r, because increasing μ_r is bounded by the sink.

**R-05 — Tail latency is superlinear in utilisation.** P(W>t) = ρ exp(-μ(1-ρ)t) for M/M/1. Between ρ=0.8
and ρ=0.9 with μ=20 and t=1 s, the probability rises from 0.0147 to 0.1218, a factor of 8.3 for a 12.5%
throughput gain. *Hypothesis:* Poisson arrivals and exponential service. *Consequence:* utilisation
targets for interactive tools belong near 70–80%, and a system at 95% has a one-second tail of 0.35 —
which is not a tail at all but the normal case.

**R-06 — Retry amplification is bounded by the attempt cap and multiplied by client count.** E[attempts]
≤ R, and synchronised retries make the instantaneous arrival rate up to C times the average, where C is
the number of clients retrying in phase. *Hypothesis:* independent attempts with jitter. *Consequence:*
jitter is not a refinement, it is what keeps the cap meaningful. Without jitter, E[attempts] ≤ R bounds
the total work but not the peak rate.

**R-07 — Approval is a predicate over a payload, and its scope is exactly the digest's coverage.** Two
payloads sharing a digest are indistinguishable to the approval. *Hypothesis:* the digest covers every
effect-relevant field. *Consequence:* the approval's legal scope equals the set of payloads that collide
under that digest. Any effect-relevant field outside the digest is outside the approval, regardless of
what the UI displayed.

**R-08 — Key collision probability follows the birthday bound, which applies to random keys only.** For n
values in a space M, P ≈ n²/(2M). At n=2.5e6 and M=2^64, P ≈ 1.7e-7. *Hypothesis:* values are uniformly
distributed. *Consequence:* a sequential or time-based key has a collision probability that is a property
of the numbering scheme, not of the space size. Namespacing by instance id is what converts a counter
into a random-ish key.

**R-09 — A human gate is a queue with a service rate.** The gate is a control only while its capacity
exceeds the arrival rate of what requires review. *Hypothesis:* review time is roughly stationary.
*Consequence:* at 8% of 17.4 ops/s with four-minute reviews, the gate needs 333 reviewers; the control
that scales is keyed automation with humans reserved for irreversible operations.

**R-10 — Two clocks are the minimum for period-boundary correctness.** A receipt with only processing
time cannot answer whether an effect was effective before a period closed, which is the question that
matters for accruals, cut-offs, and reversals. *Hypothesis:* event time is knowable at the source.
*Consequence:* store event time, processing time, and the authority time used for the decision; the
absence of any one of the three makes a class of audit question unanswerable.

**R-11 — Compensation is a new effect with its own failure probability.** A chain of k compensations each
succeeding with probability s leaves the all-succeeded probability s^k. *Hypothesis:* independent failures.
*Consequence:* at s=0.92 and k=6, the chance that all succeeded is 0.606, so 39% of compensation chains
end in a partial state that needs a human owner. Chains longer than two deserve an explicit state machine,
not a loop.

**R-12 — Error taxonomy is an interface, and collapsing it is an interface break.** If retryability is
inferred from prose or from a generic status, then any change to a message or a status mapping is a
behavioural change for every client. *Hypothesis:* clients rely on the taxonomy for control flow, which
they always do. *Consequence:* treating the reason enum as a versioned interface with a contract test is
the only way to change it safely.

**R-13 — Retention is derived, not defaulted.** W = max(retry horizon, reconciliation lag p99, support
resolution time) + margin. *Hypothesis:* the maximum replay age is measurable from the audit log.
*Consequence:* the default 24-hour TTL in most infrastructure is shorter than a weekend, so any
Monday-morning replay of Friday's work is admitted as new. Deriving W from the horizon converts a silent
duplicate source into an explicit number.

**R-14 — Schema permissiveness transfers control of the digest to the client.** If unknown fields are
accepted and included in the digest, a client deployment can change what is hashed without a schema
review. *Hypothesis:* the digest serialises what it is given. *Consequence:* allow extra fields for
forward compatibility, exclude them from the digest, and record their presence as an audit event.

## 11. Interview / Exam Questions

Fifteen questions follow, mixing recall, calculation, and diagnosis. Each has its own answer, its own
reasoning, and its own red flag. Answers are intentionally specific: the point of the exercise is to say
which observation would change your mind, not to recite that safety matters.

### Q1. A tool call validates against its schema and the executor dispatches it. Name the three predicates the schema did not establish, and say what component must establish each.

**Answer.** The schema established syntax only. It did not establish (1) semantic legality — that the
business state permits this operation now (period open, budget sufficient, no existing document for the
business key); (2) authority — that this principal, in this tenant, may effect this resource, evaluated at
or near effect time; (3) novelty — that this intended operation has not already been admitted or committed
under its key. Semantic legality is established by a policy component that reads authoritative business
state and returns a typed reason; authority by the executor holding its own scoped credential and
rechecking; novelty by a durable compare-and-insert on the key/digest store before dispatch.

**Reasoning.** Each predicate reads different state at a different time. Collapsing them into the schema
forces the schema to encode mutable business state, which it cannot do without becoming a policy engine
with a stale cache.

**Follow-up.** Which of the three can you evaluate locally at the executor without a remote call, and what
does that cost you in freshness?

**Red flag.** "The schema is strict, so invalid calls cannot happen." Strictness bounds shape, not
legality.

### Q2. Compute expected duplicates per day: 40,000 writes/day, 3% retried at least once, ledger coverage 0.97, sink dedup absent. Then compute the same with coverage 0.999.

**Answer.** D = N q (1-c)(1-p_sink). First: 40,000 × 0.03 = 1,200 retried; 1,200 × (1-0.97) = 1,200 ×
0.03 = 36 duplicates/day. Second: 1,200 × (1-0.999) = 1,200 × 0.001 = 1.2 duplicates/day. The ratio is
30×, matching the ratio of uncovered fractions (0.03/0.001 = 30).

**Reasoning.** Only retried operations can duplicate, and only the uncovered fraction of those becomes a
second effect. Coverage improvement is proportional; retry reduction is proportional too, but you control
coverage more directly.

**Follow-up.** If the sink deduplicates on your key for 60 seconds and your retries occur within 5
seconds, what is p_sink and how does D change?

**Red flag.** Reporting 40,000 × 0.03 = 1,200 as the duplicate count, which confuses retried operations
with duplicated effects.

### Q3. Why must an idempotency key be minted before the first dispatch and persisted, rather than derived from the request?

**Answer.** If the key is derived from a request identifier, every retry produces a new request id, hence
a new key, hence no deduplication. If the key is derived from the payload by hashing, then any payload
edit produces a new key, so the ledger cannot distinguish "the user changed the amount" from "a new
operation", and the KEY_CONFLICT detector that catches exactly that case never fires. The key must
identify the intention, which exists before the bytes and survives every attempt. That means: mint at
intent formation, persist durably before the first attempt, reuse on every attempt, and keep the digest as
a separate field so a changed payload under the same key is a detectable conflict.

**Reasoning.** Keys answer "is this the same intention"; digests answer "is this the same payload". The
pair is what makes CONFLICT expressible. Derive either from the transport and you lose the distinction.

**Follow-up.** What must be true for a key to survive a process restart mid-dispatch?

**Red flag.** "We use a UUID per call." That is a request id with a nicer name.

### Q4. A backlog of 5,400 operations exists. The recovery lane completes 28 ops/s and live arrivals continue at 16.9 ops/s. Drain time? What if live arrivals rise to 28 ops/s during the drain?

**Answer.** Surplus = 28 - 16.9 = 11.1 ops/s; T = 5,400 / 11.1 = 486 s ≈ 8.1 minutes. If live arrivals
rise to 28, the surplus is 28 - 28 = 0 and the drain time is infinite: the lane is fully occupied by new
work and the backlog is conserved indefinitely. In that case no amount of waiting helps; the only fix is
to reduce live arrivals (shed low-value work) or raise μ_r above 28.

**Reasoning.** Conservation. The backlog changes only by the difference between completion and arrival,
so a lane at parity adds capacity without adding recovery.

**Follow-up.** Which lever can you pull within one minute during an incident, and what does it cost in
user-visible failure?

**Red flag.** "We will add more workers" without checking whether the surplus is positive after the new
arrivals are counted.

### Q5. At μ = 20 ops/s, what is the probability a request waits more than one second at ρ = 0.8, and at ρ = 0.9? What do you conclude about utilisation targets?

**Answer.** P = ρ exp(-μ(1-ρ)t). At ρ=0.8: 0.8 × exp(-20 × 0.2 × 1) = 0.8 × exp(-4) = 0.8 × 0.0183 =
0.0147. At ρ=0.9: 0.9 × exp(-20 × 0.1 × 1) = 0.9 × exp(-2) = 0.9 × 0.1353 = 0.1218. A 12.5% throughput
increase multiplies the one-second tail by 8.3. Conclusion: for interactive tools, target 70–80%, and
treat 90% as an incident condition rather than an efficiency goal.

**Reasoning.** The exponential term dominates because (1-ρ) appears in the exponent. The intuition "we
have 10% spare, so we are fine" reads the wrong variable.

**Follow-up.** What does the model assume about service time, and how does a heavy-tailed downstream call
change the true tail?

**Red flag.** Computing ρ and stopping there, without the tail.

### Q6. Distinguish UNKNOWN from REJECTED and from NOT_FOUND, and say which one justifies a new key.

**Answer.** UNKNOWN means the outcome is not yet knowable: the request was dispatched and the response
was lost, or the sink is still processing. REJECTED means the sink or a gate decided not to effect the
operation, with a reason. NOT_FOUND means an authoritative query found no record for the key, which is
evidence of non-effect only if the query is authoritative and the record would be durable. None of the
three justifies a new key for a retried intention. A new key is justified only when the intention itself
changes — a different amount, a different target, a different effective date — and then it is a new
operation requiring its own approval.

**Reasoning.** A new key restarts the duplicate-detection story. Minting one to escape an ambiguous
outcome is the canonical way to convert one unknown into two effects.

**Follow-up.** What must be true about the query for NOT_FOUND to mean "not effected"?

**Red flag.** Treating a timeout as REJECTED, which makes it retryable-by-default.

### Q7. Design the rejection reasons for a write tool. How many, and what fields does each carry?

**Answer.** Four to six reason codes are enough for almost every write tool: `SCHEMA_INVALID`,
`SEMANTIC_DENIED`, `AUTHORITY_DENIED`, `KEY_CONFLICT`, `SINK_REJECTED`, `RATE_LIMITED`. Each carries
`retryable` (boolean), `remediation` (what the caller should do instead), `owner` (caller, operator, or
reviewer), and a correlation key plus `policy_version` where a policy was involved. `SCHEMA_INVALID`,
`SEMANTIC_DENIED`, `AUTHORITY_DENIED`, and `KEY_CONFLICT` are `retryable: false`; `RATE_LIMITED` is
retryable with backoff; `SINK_REJECTED` is retryable only if the sink's own reason says so.

**Reasoning.** The taxonomy exists so clients stop inferring control flow from status codes and prose.
Every field is a decision the client would otherwise have to guess.

**Follow-up.** Where must the reason be asserted in tests, and why at the outermost boundary?

**Red flag.** A single `ERROR` code with a message string, which forces every client to parse English.

### Q8. A digest over seven of nine effect-relevant fields is used to bind an approval. Compute the unbound fraction, and say why the number alone is not the risk measure.

**Answer.** U = (9-7)/9 = 0.222, so about 22% of the effect-relevant field set is outside the approval.
The number alone misleads because fields are not equal: omitting `currency` changes the effect's meaning
by orders of magnitude, while omitting a display name changes nothing. Rank omissions by consequence and
treat any missing amount, currency, target, or verb as a defect regardless of U.

**Reasoning.** Coverage is a count; risk is a weighted sum with heavy tails. The count is a screening
test that catches gross negligence, not a risk score.

**Follow-up.** Which omitted field would you approve shipping for one sprint, and what compensating check
would you add?

**Red flag.** "U is 0.22, which is below our 0.25 threshold, so it ships."

### Q9. Explain why a ledger with a 24-hour TTL produced duplicates on a Monday morning.

**Answer.** The maximum replay age exceeded the TTL. Friday's operations that timed out were reconciled on
Monday, 60+ hours later; the ledger had expired their keys, so the reconciliation presented them as new
intentions and the sink created second effects. The failure is arithmetic, not subtle: TTL < maximum
replay age. The fix is to derive W from max(retry horizon, reconciliation lag p99, support resolution
time) plus a margin, and — when the sink's own dedup window is shorter than that horizon — to add a
business-key check as a second line of defence.

**Reasoning.** Retention is a claim about the future: no replay will arrive older than W. When the claim
is violated, the ledger stops being a detector and becomes a cache.

**Follow-up.** How would you measure the maximum replay age from existing data?

**Red flag.** "We extended it to 48 hours", which is still shorter than a weekend.

### Q10. A client retries with attempt cap 4 and per-attempt success probability 0.75. Compute expected attempts and the load multiplier at 17.4 intents/s. What changes if the sink degrades to s = 0.25?

**Answer.** E = (1 - 0.75^4)/(1 - 0.75) = (1 - 0.3164)/0.25 = 0.6836/0.25 = 2.734 attempts. Load =
17.4 × 2.734 = 47.6 ops/s at the sink. At s = 0.25 with the same cap: 0.25^4 = 0.003906; (1 - 0.003906)
/0.75 = 0.996094/0.75 = 1.328 attempts, so load is 17.4 × 1.328 = 23.1 ops/s — but successful completions
fall to a quarter, so the *useful* throughput collapses while attempt load rises above intent rate. That
asymmetry is the storm: the sink sees more work and delivers less.

**Reasoning.** The cap bounds work per intent but not the aggregate when many intents retry
simultaneously. A retry budget expressed against recent successes bounds the aggregate instead.

**Follow-up.** At what per-attempt success rate does a retry budget of 10% get exhausted, and what
happens then?

**Red flag.** Reporting the expected-attempts number as throughput without netting successes.

### Q11. An outbox row is written before dispatch and marked `COMMITTED` after the sink responds. Under what condition is it a valid receipt, and how do you make the condition true?

**Answer.** Valid only if the `COMMITTED` transition happens in a transaction that also persists the
sink's own transaction identifier, and only if a row that never receives that identifier is distinguishable
from one that did. Otherwise the row is evidence of dispatch, not of effect. Make it true by separating
two fields: `dispatch_state` (written pre-dispatch) and `outcome_state` with `outcome_evidence` (written
only from an authoritative response, carrying the sink's id). A timeout leaves `dispatch_state =
DISPATCHED` and `outcome_state = UNKNOWN`, which is the truth.

**Reasoning.** One enum spanning two rungs of the evidence ladder guarantees that some reader will
interpret the top rung as reached. Two fields make the ambiguity unrepresentable.

**Follow-up.** What query resolves UNKNOWN, and what makes that query authoritative?

**Red flag.** "The row says COMMITTED, so it is committed."

### Q12. A tool's description is fetched at runtime from a shared registry. What can change routing without any schema change, and what do you pin?

**Answer.** The description text, the parameter descriptions, and any examples are model-visible and
influence tool selection and argument filling. An attacker who can edit a description or a shared resource
that the description references can change which tool runs or which arguments are used, with no schema
diff. Pin the manifest by digest, review changes explicitly, keep the allowed tool-and-argument set
enforced outside the model, and treat description text as data with provenance rather than authority.

**Reasoning.** Discovery is a data path into the model's context, and the model's output is the proposal.
Anything that can write to a data path can influence the proposal; only components outside the model can
constrain the effect.

**Follow-up.** What is the smallest allowlist that still permits useful adaptation, and who owns it?

**Red flag.** "The descriptions come from our own internal registry, so they are trusted."

### Q13. Price a compensation plan: six sequential compensation steps, each succeeding with probability 0.92. What is the probability of a fully compensated outcome, and how do you handle the rest?

**Answer.** 0.92^6 = 0.606, so there is a 39.4% chance that at least one step failed and the entity is in a
partial state. Handle it by modelling compensation as a first-class operation with its own key, approval,
and typed outcomes, and by making "partially compensated" an explicit state with a named owner and a
follow-up task, rather than an error to retry. Alert on chains longer than two and on any entity that
remains partially compensated past its SLA.

**Reasoning.** Compensation is a sequence of independent effects, so its reliability multiplies down.
Retrying a failed compensation without a new key risks a double compensation, which is a worse state
than the one being fixed.

**Follow-up.** Which of the six steps is irreversible, and why is it in the chain at all?

**Red flag.** "We re-run the compensation job until it shows green."

### Q14. What single audit field, if missing, makes period-boundary disputes unanswerable, and why?

**Answer.** Event time — the time the business fact took effect, as distinct from the time your system
processed the record. Without it, a receipt cannot be placed relative to a period close, so a late-posted
invoice cannot be attributed to the correct period, and a reversal cannot be distinguished from a
different-period correction. The complete minimum is three stamps: event time, processing time, and the
authority time used for the decision.

**Reasoning.** Period boundaries are cut in event time, while most systems store processing time because
that is what their logging framework provides. The mismatch is invisible until an auditor asks which
period the effect belongs to.

**Follow-up.** Where does event time come from when the source system only sends processing time?

**Red flag.** "We log the timestamp", where the timestamp is when the row was inserted.

### Q15. Your tool passes all contract tests and duplicates appear only during downstream incidents. What is the most likely structural cause, and which measurement confirms it?

**Answer.** The dedup store and the executor are not in the same failure domain — typically the key store
is remote or shared, and during a downstream incident the key store is slow or unavailable while dispatch
still proceeds (or dispatch is retried from a path that skips the store). Coverage effectively drops to
zero for the affected interval while retries continue. Confirmation: measure key-store availability and
latency during the incident window and compare with the duplicate cohort's timestamps; if duplicates
cluster exactly where key-store latency exceeded the executor's timeout, the structural cause is
confirmed. The correction is to fail closed — refuse dispatch when the key store is unavailable — and to
co-locate the store with the executor.

**Reasoning.** A detector that is unavailable during the failure it exists to detect is not a detector.
Fail-closed converts an availability problem into an explicit outage instead of a silent data problem.

**Follow-up.** What is the business cost of failing closed for ten minutes versus duplicating for ten
minutes, and who decides?

**Red flag.** "The contract tests pass, so the ledger works."

## 12. Quick Reference Card

**Three sentences to keep.** A schema grants shape, never authority. A receipt comes from the sink, never
from your own optimism. A key identifies an intention, so it outlives the request that carried it.

**Status.** Draft. Invariants in sections 4 and 10 are DERIVED; every parameter in sections 5, 9, and 11
is an ILLUSTRATIVE construction whose arithmetic was recomputed by hand. No deployed integration, vendor
behaviour, or measured incident is asserted, and protocol details are version-specific and must be
rechecked against the current specification.

| Situation | Rule | Critical assumption | Avoid | Consequence if wrong |
|---|---|---|---|---|
| Designing a write tool | classify effect class first (E0–E4) | sink reversibility is known | treating a create as an upsert | duplicate documents that look like edits |
| Before dispatch | durable compare-and-insert on key+digest | store is crash-consistent and local | writing the record after the response | effects with no ledger entry |
| Timeout received | promote to UNKNOWN, reconcile on the key | an authoritative query exists | retrying with a new key | two effects, one intention |
| Payload differs under same key | reject as KEY_CONFLICT | digest covers all effect fields | updating the stored digest | ledger becomes last-writer-wins |
| Approval flow | bind digest + approver + expiry + policy version | executor verifies at dispatch | free-text approval | approval scope silently grows |
| Capacity plan | compute peak, backlog, and drain surplus separately | rates constant per interval | planning against the daily mean | queue grows during the peak hour |
| Utilisation target | 70–80% for interactive tools | service times roughly exponential | running at 90%+ for efficiency | one-second tail ~0.12 at ρ=0.9 |
| Dedup retention | W = max(retry, reconciliation p99, support) + margin | maximum replay age is measurable | a default 24-hour TTL | weekend replay duplicates |
| Sink ignores your key | treat as E3 and reconcile | sink state is queryable | claiming exactly-once | unprovable duplicate suppression |
| Incident with growth | check whether surplus is positive before adding lanes | arrival rate known | adding capacity at parity | no drain, ever |
