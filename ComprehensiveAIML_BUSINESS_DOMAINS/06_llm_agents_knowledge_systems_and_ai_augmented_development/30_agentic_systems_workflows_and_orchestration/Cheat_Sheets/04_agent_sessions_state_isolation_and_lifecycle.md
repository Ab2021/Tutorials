---
title: "Agent Sessions, State Ownership, Isolation and Run Lifecycle"
topic: "Agent sessions, state ownership, isolation and run lifecycle"
collection: "30_agentic_systems_workflows_and_orchestration"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Agent Sessions, State Ownership, Isolation and Run Lifecycle

## 1. File Header

> **Purpose:** decide, for a given request, who owns the state, which copy is authoritative, and what a restart is allowed to redo.
> **Prerequisites:** HTTP identity, database transactions and isolation levels, at-least-once delivery, filesystem and object-store permissions.
> **Key outcome:** trace ownership from caller to external effect, choose an isolation mechanism per boundary, and size retained state.
> **Not covered:** model training, one vendor's agent SDK, jurisdictional legal advice on retention.
> **Level:** advanced; assumes production incident experience.
> **Evidence:** invariants are DERIVED from stated preconditions; every numeric example is ILLUSTRATIVE and marked; no production benchmark is claimed.
> **Retrieval cues:** duplicate side effects → F-08, F-09, F-15; sizing → §4.3 variant B and §9.2; stale worker after failover → §4.3 variant D and F-10.

The engineering decision in this sheet is narrow and repeatable: given a session that survived a crash, choose resume, retry, cancel or reconcile.
Everything else — token budgets, prompt layout, framework choice — is downstream of that choice.
Two facts make the choice hard. First, the model context is derived data that may contain untrusted text; treating it as authority converts a prompt injection into an authorization decision.
Second, a database commit and an external API call cannot be made atomic without a contract the external system honours.
So the design separates three layers that beginners conflate: the ownership record (who may act), the durable execution record (what happened), and the conversation artefacts (what was said).
A session outlives a run; a run outlives an attempt; an external effect outlives all three.
**Takeaway:** the control plane decides authority, the store decides truth, the worker decides nothing durable.

Reading order for a practitioner under pressure: §2 for the mental model in one diagram, §4.1 for the base predicate, §8 for the symptom you are actually seeing.
Reading order for a designer: §4.3 for variants, §7 for the selection table, §9 for the cost model, then §11 to test yourself.

## 2. Concept Map / Mental Model

```text
caller (verified principal, tenant claim already resolved)
   |
   | 1. admission: authenticate, resolve tenant, apply quota
   v
control plane ============================ policy bundle (versioned)
   |            \                              |
   | 2. create   \ 3. state transition         | 4. decide authority per action
   v              v                            v
run state machine --> durable store (session, run, version, checkpoint)
   |                        ^
   | 5. dispatch with scoped, short-lived capability
   v                        | 6. checkpoint commit (before effect)
worker ---------------------+
   |            \
   | 7. build    \ 8. propose action
   v              v
model context   action intent (canonical payload + effect key)
(untrusted text)     |
                     | 9. authority re-check at effect boundary
                     v
                tool adapter --> external system of record
                     |                   |
                     | 10. receipt       | 11. effect key retained by receiver
                     v                   v
                durable run history   remote dedup window
```

Read the diagram as two loops, not one pipeline. The upper loop is authority and versioning; it is cheap, synchronous and must be correct.
The lower loop is execution; it is expensive, asynchronous and is allowed to fail and repeat.
A restart re-enters at step 3 using committed rows only. It never re-enters by trusting what a worker held in memory, because that memory died with the process.
The dashed conceptual line between "model context" and "action intent" is where most production incidents happen: a proposal acquires authority it was never granted.
Retrieved documents, tool outputs, and prior user turns all reach the model as data. None of them can widen a capability; only the control plane can.
An approval is a first-class state, not a message: it has an actor, a canonical payload digest, a policy version and an expiry.
A cancellation is also a state, not a signal: it requests a transition and the transition is completed only when in-flight effects are dispositioned.
**Takeaway:** authority flows down through the diagram; evidence flows up. Neither direction is allowed to skip a version check.

Three common misreadings of the diagram, each with the correction that matters:

- "The worker owns the session while it runs." Correction: the worker holds a lease on a version; the store holds ownership.
- "The queue is the state." Correction: the queue holds deliveries; replaying a delivery must be harmless, so state lives elsewhere.
- "The model context is the transcript." Correction: context is a per-call selection of transcript plus retrieved material, with its own retention.

## 3. Foundational Definitions

Each term below gets the same treatment: what it is, what it is not, and the check that distinguishes them at 3 a.m.

### Session
- **Definition:** an owned, durable interaction container scoped to one tenant and one conversation identity.
- **Not:** a connection, a process, or a token window.
- **Check:** does a row exist with tenant, owner, policy version and creation time before any run starts?

### Run
- **Definition:** one bounded execution that advances a session toward a goal, with its own budget and terminal state.
- **Not:** a retry, and not the whole conversation.
- **Check:** every effect carries a run identifier and a version; two runs on one session must not interleave writes silently.

### Attempt
- **Definition:** a single dispatch of a run to a worker; attempts are numbered and each has its own start, heartbeat and outcome.
- **Not:** a user-visible retry.
- **Check:** the attempt table count for a run equals the number of dispatches, including ones that timed out without a response.

### State owner
- **Definition:** the single component permitted to write a given state fragment, enforced by credential scope rather than convention.
- **Not:** the component that reads it most often.
- **Check:** can a worker write the terminal run state directly? If yes, ownership is not enforced.

### Checkpoint
- **Definition:** a committed snapshot of enough state to continue without redoing committed work.
- **Not:** a log of intentions, and not a save of the model context.
- **Check:** does resuming from checkpoint N ever re-emit an effect already recorded as committed after N?

### Transcript
- **Definition:** the append-ordered record of interaction events, retained under a stated policy.
- **Not:** the model's context window, and not an authority record.
- **Check:** is the transcript readable only under the same tenant predicate as the session?

### Context
- **Definition:** the specific sequence of material sent in one model call, assembled from transcript, retrieval and instructions.
- **Not:** durable, not complete, not trusted.
- **Check:** can you reconstruct which documents and tool outputs were in the context for a given call?

### Lease
- **Definition:** a time-bounded claim by a worker that it may act on a version; expiry allows another worker to act.
- **Not:** a lock that stops a paused process from running.
- **Check:** after expiry, does the previous holder's next write get rejected?

### Fencing token
- **Definition:** a monotonically increasing generation value attached to protected writes so receivers can reject stale holders.
- **Not:** a timestamp, and not a substitute for authorization.
- **Check:** does the storage layer compare the token, or only log it?

### Idempotency key
- **Definition:** a stable identifier for one intended effect, used by a receiver or adapter to collapse repeats.
- **Not:** a request identifier that changes on each retry, and not a cache key.
- **Check:** does the key survive process restart and retry, computed from canonical action content?

### Approval
- **Definition:** a recorded decision by an authorized actor permitting a specific canonical action within a validity window.
- **Not:** a chat message, and not a blanket permission for a run.
- **Check:** change one material field of the payload; is the approval invalidated?

### Reconciliation
- **Definition:** the process of resolving an unknown external outcome by querying, comparing receipts and writing a disposition.
- **Not:** a blind retry, and not a manual database edit.
- **Check:** is there a durable queue of unresolved effects with an owner and an age metric?

### Isolation
- **Definition:** the property that no read or write performed under one tenant or principal scope can observe or alter another's state.
- **Not:** a UI filter, and not "we use separate schemas".
- **Check:** does the enforcement point sit below the code that could forget to add a predicate?

### Retention
- **Definition:** the rule that determines when state, transcript, checkpoints and audit records become unreadable or are destroyed.
- **Not:** a backup policy, and not "we keep everything for debugging".
- **Check:** can you delete one tenant's session and prove no copy survives in caches, exports or replicas?

### Terminal state
- **Definition:** a run or session state with no outgoing transitions; further work requires a new run or an explicit reopen with audit.
- **Not:** "quiet" or "no active worker".
- **Check:** is the transition table the only writer of terminal states, and is it a single conditional update?

### Effect boundary
- **Definition:** the point where a proposal becomes an irreversible external change; the last place authority and deduplication can be enforced together.
- **Not:** the API client call site inside business logic.
- **Check:** is every outbound irreversible call funnelled through one adapter that checks policy, key and destination?

### Disposition
- **Definition:** the recorded judgement about an ambiguous effect: committed, not committed, or unknown-pending-reconciliation.
- **Not:** an inference from a timeout.
- **Check:** does the run history contain a disposition row for every timed-out irreversible call?

## 4. Core Formulas with Full Derivation

Each block uses the same anatomy: FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION. Numbers are ILLUSTRATIVE and chosen so the arithmetic can be re-done by hand.

### 4.1 Base case

```text
FORMULA
  allow_read(u, s) = [tenant(u) = tenant(s)]
                     AND [principal(u) in owners(s)]
                     AND [policy_version(u) >= min_version(s)]
                     AND [state(s) not in DELETED]

WHERE
  u   = verified request identity    (tuple: tenant, subject, roles, policy_version)
  s   = target session record        (tuple: tenant, owners, min_version, state)
  allow_read is boolean; there is no partial credit and no "log only" mode

CONSTRAINT
  tenant(u) must come from the authentication layer, not from a header,
  not from a query parameter, and not from the model's output.
  owners(s) must be membership records, not free-text strings.

NUMERICAL EXAMPLE
  request: tenant A, subject sam, roles {support}, policy_version 7
  session: tenant A, owners {sam, dara}, min_version 5, state active
  [A = A] true; [sam in {sam, dara}] true; [7 >= 5] true; [active not deleted] true
  -> allow_read = true

  same request, session owners {dara}: [sam in {dara}] false
  -> allow_read = false, and the record must not be returned even as an empty stub

INTERPRETATION
  The predicate is conjunctions of independently checkable facts.
  Its weakness is that a single missing term is easy to hide: with four terms,
  forgetting the tenant term fails only for cross-tenant traffic, which happy-path
  tests do not generate.
```

```text
FORMULA (retained state, base case)
  L = S * D * B * M

WHERE
  S = new sessions per day          (sessions/day)
  D = retention window              (days)
  B = logical bytes retained per session-day (bytes)
  M = physical multiplier           (dimensionless: indexes, WAL, replicas, snapshots)
  L = steady-state logical-then-physical footprint (bytes)

CONSTRAINT
  valid only for a stationary arrival cohort: daily new sessions and per-session
  byte growth must both be roughly constant across the retention window.
  If transcripts grow superlinearly per session, S*B is not a constant and the
  product understates the tail.

NUMERICAL EXAMPLE
  S = 2,400 sessions/day, D = 30 days, B = 12,000 bytes, M = 2.4
  session-days   = 2,400 * 30            = 72,000
  logical bytes  = 72,000 * 12,000       = 864,000,000
  physical bytes = 864,000,000 * 2.4     = 2,073,600,000
  decimal GB     = 2.0736 GB
  GiB            = 2,073,600,000 / 1,073,741,824 = 1.9312 GiB

INTERPRETATION
  Per retained session the footprint is 12,000 * 2.4 = 28,800 bytes, which is the
  most useful derived quantity: it converts a capacity question into a per-session
  cost. At 1.9 GiB this design fits a modest instance; the same formula at
  D = 365 days gives 25.2 GB and forces partitioning by tenant or by month.
```

### 4.2 General case

The general case replaces "one session, one owner, one write" with a versioned machine that several workers may touch over time.

```text
FORMULA (conditional transition)
  T(s, v, to) = update session set state = to
                where id = s and version = v and state in ALLOWED_FROM(to)
  rows_affected = 1  -> transition committed, version becomes v + 1
  rows_affected = 0  -> either version moved or the current state forbids the edge

WHERE
  ALLOWED_FROM(to) is the set of predecessor states for the target state
  version is an integer column incremented in the same statement
  the caller must not read version and then write it back in a second statement

CONSTRAINT
  only works if the store evaluates the predicate and the increment atomically,
  and only if every writer uses this statement rather than a read-modify-write.

NUMERICAL EXAMPLE
  two workers both hold version 4 and both attempt completed
  worker one: update ... where version = 4 -> 1 row, version now 5
  worker two: update ... where version = 4 -> 0 rows, must reload
  without the predicate both updates succeed and the second silently overwrites
  whatever worker one wrote, including its effect receipt

INTERPRETATION
  The conditional statement converts a race into a retry. It says nothing about
  the external effect performed under version 4; that effect still needs an
  effect key, because the database never saw it.
```

```text
MECHANISM BLOCK (state machine with absorbing terminals)
  preconditions:
    finite state set Q, declared transition relation ALLOWED subset of Q x Q
    one writer component (the transition service), credentialed to update state
  state transition:
    apply T(s, v, to) from 4.2; on 0 rows, reload and re-decide
  invariant:
    terminal states have out-degree zero, and every non-terminal state is
    reachable from created through allowed edges only
  justification:
    a run that can leave a terminal state makes "finished" unreliable and lets
    duplicate work appear legal; a disconnected non-terminal state is a run that
    no operator can ever advance or close
  worked trace:
    Q = {created, admitted, running, waiting_approval, paused, cancelling,
         completed, failed, cancelled, expired}         (10 states)
    allowed edges = 17
    ordered pairs possible = 10 * 9 = 90
    density = 17 / 90 = 0.1889
    terminals: completed, failed, cancelled, expired -> out-degree 0 each
  failure boundary:
    a compensating action (refund, ticket close) needs a new run against the
    session; reopening a terminal run would make audit counts non-monotonic
```

```text
FORMULA (isolation defect rate under independent layers)
  P(leak per unauthorised request) = PRODUCT over l in 1..L of (1 - c_l)

WHERE
  L   = number of independent enforcement layers on the read path
  c_l = probability that layer l blocks an unauthorised read (0..1)

CONSTRAINT
  independence is the whole argument, and it is usually false: layers that share
  one defaults file, one ORM session or one middleware skip together.

NUMERICAL EXAMPLE
  three layers, c = 0.9 each, independent
    P(leak) = 0.1 * 0.1 * 0.1 = 0.001
  same layers, single shared default that disables all three at once
    P(leak) = 1 - 0.9 = 0.1
  probes per week = 2,000 requests/day * 7 = 14,000
    independent:      14,000 * 0.001 = 14 leaked responses
    correlated:       14,000 * 0.1   = 1,400 leaked responses (100x)
  per-tenant cache key omission with 3 tenants and 1 shared cache:
    collision exposure = 3 * 2 = 6 ordered tenant pairs, not 3

INTERPRETATION
  Defence in depth buys one order of magnitude only when the layers fail
  independently. Architecturally the layer that matters is the one that fails
  closed by default, which is why the enforcement point belongs in storage.
```

### 4.3 Variants

```text
VARIANT A (attempts with a retry cap)
  P(success within n) = 1 - (1 - s)^n
  E[attempts, capped at n] = SUM over k = 1..n of (1 - s)^(k-1)

WHERE
  s = per-attempt success probability, assumed constant and independent
  n = attempt cap from the run budget

NUMERICAL EXAMPLE (s = 0.85, n = 3)
  P(success) = 1 - 0.15^3 = 1 - 0.003375 = 0.996625
  E[attempts] = 1 + 0.15 + 0.0225 = 1.1725
  escalation rate = 0.003375; at 2,400 sessions/day that is 8.1 sessions/day
  cost multiple versus single attempt = 1.1725 / 1.0 = 1.1725

INTERPRETATION
  The cap dominates the tail: raising n from 3 to 5 lowers the escalation rate to
  0.15^5 = 0.0000759, a 44.5x reduction, while lifting the attempt expectation from
  1.1725 to 1 + 0.15 + 0.0225 + 0.003375 + 0.00050625 = 1.17638125, which is a
  0.33% increase in model spend. The arithmetic looks free; it is not, because an
  attempt is not a fixed cost - it re-sends the accumulated context, so attempt 4
  carries more tokens than attempt 1 and the expectation understates the spend.
```

```text
VARIANT B (checkpoint interval and redo)
  redo_expected = C / 2            (crash time uniform within an interval)
  redo_fraction = C / (2 * T)
  checkpoints = floor(T / C)
  checkpoint_bytes_per_run = checkpoints * K

WHERE
  T = nominal run duration (s), C = checkpoint interval (s), K = checkpoint size (B)

NUMERICAL EXAMPLE
  T = 1,800 s, C = 90 s, K = 40 KiB
  checkpoints        = 1,800 / 90 = 20
  redo_expected      = 45 s
  redo_fraction      = 45 / 1,800 = 0.025 (2.5%)
  bytes per run      = 20 * 40 * 1,024 = 819,200 B = 800 KiB
  long runs per day  = 0.08 * 2,400 = 192
  checkpoint bytes   = 192 * 819,200 = 157,286,400 B/day
  session bytes/day  = 2,400 * 12,000 = 28,800,000 B/day
  ratio              = 157,286,400 / 28,800,000 = 5.46

INTERPRETATION
  Halving C to 45 s doubles checkpoint bytes to 314.6 MB/day and halves redo to
  1.25%. The exchange rate is exact here: 1% of redo avoided costs 2% more
  checkpoint storage, so the decision belongs to recovery-time budget, not storage.
```

```text
VARIANT C (transcript compaction)
  tokens(t) = t * R                        before compaction
  compaction turn = ceil(K / R)
  post_compaction = H + w * R

WHERE
  R = mean tokens added per turn, K = context budget, H = summary size,
  w = turns retained verbatim after the summary

NUMERICAL EXAMPLE
  R = 350, K = 24,000, H = 2,000, w = 6
  compaction turn = ceil(24,000 / 350) = ceil(68.57) = 69
  post-compaction tokens = 2,000 + 6 * 350 = 4,100
  tokens saved at that point = 69 * 350 - 4,100 = 24,150 - 4,100 = 20,050
  turns between compactions = 69 - 6 = 63

INTERPRETATION
  Compaction is lossy by construction: the summary is generated material, so it
  must never carry an approval, a capability or a number whose correctness is
  audited. Keep those in structured state, which does not get summarised.
```

```text
VARIANT D (fencing window)
  overlap = max(0, pause_max - ttl)
  writes_at_risk = overlap * write_rate

WHERE
  ttl = lease time-to-live (s), pause_max = longest tolerated holder pause (s)
  write_rate = protected writes per second by the stale holder

NUMERICAL EXAMPLE
  ttl = 30 s, pause_max = 45 s (stop-the-world GC plus network partition)
  overlap = 45 - 30 = 15 s
  write_rate = 3 writes/s
  writes_at_risk = 15 * 3 = 45
  with a monotonic fence token, rejected = 45 of 45
  without fencing, corrupted = 45 of 45

INTERPRETATION
  Fencing does not shorten the overlap; it makes the overlap harmless where the
  receiver enforces the token. A receiver that only logs the token converts the
  problem into a detective control with a 45-write detection lag.
```

### 4.4 Limits and counterexamples

The base storage formula breaks at exactly the moment transcripts stop being stationary: sessions that append tool output without bound.
Concretely, let each session accumulate g = 6,000 bytes/day instead of holding B = 12,000 bytes flat.
A session that is d days old holds g·d bytes, so the cohort in the retention window holds an average of g·D/2 bytes rather than g·D.
With g = 6,000, D = 30 and M = 2.4 the true steady state is 2,400 × 6,000 × 2.4 × (30 × 31 / 2) = 2,400 × 6,000 × 2.4 × 465 = 16,070,400,000 bytes = 16.07 GB.
The flat formula applied with the day-30 value instead gives 2,400 × 180,000 × 2.4 × 30 = 31,104,000,000 bytes = 31.1 GB, and applied with the day-1 value gives 2,400 × 6,000 × 2.4 × 30 = 1,036,800,000 bytes = 1.04 GB.
So the flat product can overstate by 1.94× or understate by 15.5× depending on which sample day you chose, and neither error is visible in the result.
The general statement: a formula that multiplies a per-day constant by D treats a flow as a stock, and the error factor is D/2 divided by the measurement day.
```text
FORMULA (growth stock)
  L_growth = S * g * M * D * (D + 1) / 2

WHERE
  S = 2,400 sessions/day, g = 6,000 B/day/session, M = 2.4, D = 30 days
  the sum d = 1..D equals D(D+1)/2 = 465, the cohort's total session-days

CONSTRAINT
  requires linear per-session growth and a stationary arrival cohort; exponential
  growth invalidates it entirely and needs a compounding term instead.

NUMERICAL EXAMPLE
  L_growth = 2,400 * 6,000 * 2.4 * 465 = 16,070,400,000 B = 16.07 GB
  flat with day-30 sample  = 31,104,000,000 B -> ratio 1.935
  flat with day-1 sample   =  1,036,800,000 B -> ratio 0.0645, understates 15.5x

INTERPRETATION
  Both the flat and the growth formula are defensible; choosing without stating
  which one you used is not, because the two answers differ by a factor of 30.
```
The independence assumption in the leakage formula fails in a way that is easy to see in code review: one helper that builds queries with an optional tenant filter, called from three services, collapses three layers into one.
The retry-cap formula fails when attempts are correlated. If one expired credential makes every attempt fail, s is not constant, the geometric model promises 99.66% recovery at n = 3, and the observed recovery is 0%.
The checkpoint redo formula assumes uniform crash time inside the interval; when crashes correlate with activity, such as memory pressure during a large tool response, they cluster just before the next checkpoint and expected redo approaches C = 90 s rather than C/2 = 45 s.
The fencing formula assumes a monotonic token the receiver enforces; against a third-party API with no such parameter the overlap arithmetic is irrelevant and only key-based deduplication or single-writer serialisation is safe.
The transition-machine invariant fails if operators can edit state directly: once a terminal state is reachable by hand, "terminal" is a convention and out-degree zero is a claim about code that no longer describes the system.

## 5. Worked Numerical Example

**ILLUSTRATIVE end-to-end trace.** One tenant, one session, one run that crosses an approval, a worker crash and an ambiguous external call. All numbers are constructed for teachability.

Setup used throughout the trace:

- Tenant T1, principal `sam`, roles {support}, policy version 12.
- Session `S-4471` created at 09:00:00Z, retention 30 days.
- Run `R-118` admitted at 09:00:04Z with attempt cap 3 and token budget 200,000.
- External action: issue a refund of 4,200 minor units to a stored destination.
- Checkpoint interval 90 s; lease TTL 30 s; worker write rate 3 writes/s.

Step 1, admission arithmetic. Admission performs 4 predicate checks (tenant binding, quota, policy version, session state) in one statement.
At 2,400 sessions per day and 1.0 run per session, that is 9,600 checks/day, or 9,600 / 86,400 = 0.111 checks/s — irrelevant as load, and therefore safe to make synchronous and strict rather than eventually consistent.

Step 2, storage for the session. Logical bytes 12,000, physical multiplier 2.4, so 28,800 bytes retained per session.
1,000 sessions of this shape in the 30-day window occupy 1,000 × 28,800 = 28,800,000 bytes (28.8 MB).
That equals one day of full arrivals (2,400 × 12,000 = 28,800,000) only because 1,000 × 2.4 = 2,400; the match is an identity in the parameters, not a validation of the model, and treating a parameter coincidence as a check is how wrong multipliers survive review.

Step 3, run progress. Attempt 1 starts at 09:00:05Z and reaches a refund proposal at 09:02:35Z, 150 s later.
Checkpoints fall at multiples of 90 s from attempt start: 09:01:35 (first, committed) and 09:03:05 (never reached).
So exactly floor(150 / 90) = 1 checkpoint is committed, at 09:01:35Z, and at 40 KiB it holds 40 × 1,024 = 40,960 bytes.

Step 4, approval. The proposal pauses into `waiting_approval` with a canonical payload digest covering tenant, destination, amount 4,200, currency and expiry at 09:17:35Z (900 s validity).
The state transition is one conditional update: version 3 → 4, one row affected.

Step 5, crash. At 09:06:00Z the worker is killed. Lease expires 30 s later at 09:06:30Z. A second worker acquires fence generation 8 and resumes from the checkpoint committed at 09:01:35Z.
Expected redo under uniform crash timing is C/2 = 45 s, but the crash landed 09:06:00 − 09:01:35 = 265 s after the checkpoint, so actual redo is 265 s, worse than the mean by 265 / 45 = 5.89×.

Step 6, effect attempt. At 09:07:10Z worker two dispatches the refund under effect key `S-4471/R-118/refund/v1`, derived from the canonical payload hash rather than from request time.
The provider times out at 09:07:40Z after 30 s. Disposition is `unknown-pending-reconciliation`, not `failed`.
Note the crash and the timeout are separate events 70 s apart; a single "the call failed" narrative would merge them and lose the fact that the retry after the crash carried a key that predated it.

Step 7, reconciliation arithmetic. Reconciliation polls by effect key at 30 s intervals with a cap of 10 polls, so the window is 10 × 30 = 300 s and the poll cost is 10 requests per unresolved effect.
If the provider's deduplication window is 24 h, a duplicate dispatch inside it is collapsed; if it retains keys for 600 s, any retry after 600 s risks a second refund, and the 300 s poll window sits safely inside it by 300 s.
The decision therefore depends on one provider number that this sheet cannot derive: OPEN until the provider contract is read.

Step 8, resolution. The poll at 09:08:10Z returns the provider receipt with its own identifier, so disposition becomes `committed`.
The run writes terminal `completed` at 09:08:12Z, giving total wall clock of 09:08:12 − 09:00:04 = 488 s = 8 min 8 s from admission.

**Independent arithmetic re-check of the headline quantities.**

```text
check 1 (per-session footprint)
  12,000 * 2.4 = 28,800
  division cross-check: 28,800 / 2.4 = 12,000  -> recovers B
  ratio cross-check:    28,800 / 12,000 = 2.4  -> recovers M

check 2 (wall clock)
  09:00:04 -> 09:08:12 is 8 min 8 s
  seconds = 8 * 60 + 8 = 488
  segment audit: proposal at 09:02:35 (151 s), crash at 09:06:00 (356 s),
  dispatch 09:07:10 (426 s), timeout 09:07:40 (456 s), receipt 09:08:10 (486 s),
  terminal 09:08:12 (488 s) -> every offset is within the total, monotone

check 3 (checkpoints and redo)
  floor(150 / 90) = 1 committed checkpoint at 09:01:35
  redo = 09:06:00 - 09:01:35 = 4 min 25 s = 265 s
  mean redo at C = 90 s is 45 s; ratio 265 / 45 = 5.888... -> 5.89x
  checkpoint bytes = 1 * 40 * 1,024 = 40,960 B

check 4 (approval validity)
  09:02:35 + 900 s = 09:02:35 + 15 min = 09:17:35  -> matches stated expiry
  the run completed at 09:08:12; from 09:08:12 to 09:17:35 is 9 min 23 s = 563 s,
  so the approval was consumed 563 s before expiry and no renewal was required
```

Every offset in check 2 was recomputed from the clock times rather than from the narrative, which is the only way the timeline stays internally consistent when a step is later moved.

**Sensitivity.** Four parameters move the outcome, and only one of them is under the team's control cheaply.

- Checkpoint interval: 90 s → 45 s lowers expected redo to 22.5 s but doubles checkpoint bytes from 157.3 MB/day to 314.6 MB/day at the observed long-run share of 8%.
- Attempt cap: 3 → 2 reduces worst-case model spend but raises escalation from 0.3375% to 15% = 0.0225, i.e. from 8.1 to 54 escalations per day at 2,400 sessions/day; the escalation queue, not the spend, is the constraint.
- Lease TTL: 30 s → 15 s shortens crash recovery by 15 s but tolerates only 15 s of pause; the same 45 s GC pause then yields 30 s of overlap instead of 15 s, doubling writes at risk from 45 to 90.
- Provider retention: 24 h versus 600 s changes the safe retry horizon by a factor of 144; it is the only parameter here whose uncertainty can produce a duplicated irreversible effect.

## 6. Visual / Geometric Intuition

Think of the run as a narrow corridor with three widths.

The first width is authority: how many identities may act at once. It is ideally one. When two workers hold live leases, the corridor has two travellers and the collision is not visible until both write.
The second width is time: the interval between the last committed checkpoint and the crash. Recovery cost is proportional to the area of that gap, not to its height.
The third width is scope: how many resources a compromised capability can touch. Narrowing scope does not slow anything down, which is why it is the cheapest control in the sheet.

```text
timeline (trace from Section 5, offsets in seconds from admission at 09:00:04)

t+0      admission, run created, version 0
t+91     09:01:35  cp1 committed (40,960 B); it is the last durable point
t+151    09:02:35  proposal formed, transition paused -> waiting_approval (v3 -> v4)
t+356    09:06:00  worker killed; nothing after cp1 survives
t+386    09:06:30  lease expires; fence generation 8 issued to worker two
t+426    09:07:10  effect dispatched under key S-4471/R-118/refund/v1
t+456    09:07:40  provider timeout -> disposition unknown
t+486    09:08:10  receipt received -> disposition committed
t+488    09:08:12  terminal completed

redo duration = 265 s (t+356 minus t+91), not an event on the timeline
redo gap compared with the 45 s mean at C = 90 s -> 265/45 = 5.89x worse
```

Two geometric readings follow, and both are decisions.

First, area under the gap is what you pay in repeated model calls. The replay after checkpoint 1 re-sends the whole assembled context, not the elapsed time: replaying 9,500 tokens costs the same whether 3 s or 265 s elapsed, which is why a slow crash is cheaper per second than a fast one.
Second, the corridor is not a queue. Adding workers does not widen authority; it only shortens the time to the next attempt. If the bottleneck is a 30 s provider timeout, adding workers raises the number of ambiguous effects per minute without reducing the 30 s.

A third reading concerns the transcript. Picture the context window as a fixed-height window sliding over an unbounded transcript tape; compaction replaces the tape behind the window with a label.
The window position is not the session position, and confusing them produces the classic bug where a resumed run re-asks a question the user already answered three turns before the compaction boundary.

## 7. Comparison / Selection Table

### 7.1 Concurrency control for session state

| Mechanism | Guarantees | Fails when | Operate cost | Choose when |
|---|---|---|---|---|
| Optimistic version predicate | one winner per version | writer ignores rows-affected | low | default for session rows |
| Single-writer queue per session | serialised transitions | failover creates two consumers | medium | ordering within a session matters |
| Advisory lock held across a call | mutual exclusion | long call outlives connection | high | never across a network call |
| Lease plus fence token | stale holder rejected by receiver | receiver ignores token | medium | multiple workers, shared external store |
| Database serialisable isolation | no interleaved anomalies | retry storms under contention | high | money-adjacent multi-row updates |

Rule of selection: pick the cheapest mechanism whose failure mode is detectable by a metric you already collect. A fence token whose rejections are not counted is theatre.

### 7.2 Isolation enforcement points

| Point | Blocks | Misses | Test that proves it |
|---|---|---|---|
| Authentication claim resolution | forged tenant | a legitimate user in the wrong tenant | two synthetic tenants, cross reads |
| Query predicate in repository | cross-tenant rows | raw SQL and admin tools | direct SQL as app role |
| Row-level security in database | anything bypassing the repository | table owner and bypass roles | connect as owner, retry |
| Per-tenant key prefix in object store | cross-prefix reads | signed URL reuse | replay a URL as another tenant |
| Cache key including tenant and version | private cache bleed | pre-warmed shared entries | prime as A, fetch as B |
| Egress allowlist on tool adapter | exfiltration to unknown hosts | an allowlisted host that proxies | resolve destination at call time |

Two rows are usually enough in production: the repository predicate plus row-level security. The others exist because those two are configuration, and configuration drifts.

### 7.3 Recovery strategy selection

| Situation | Evidence available | Action | Forbidden action | Residual risk |
|---|---|---|---|---|
| Crash before any dispatch | checkpoint, no effect rows | resume from checkpoint | none | duplicated model spend only |
| Crash after commit, before receipt | effect row with key, no receipt | reconcile by key | retry as new effect | provider lacks key lookup |
| Timeout, unknown remote outcome | outbound request id, no response | park as unknown, poll | mark failed and move on | double effect if retried |
| Duplicate delivery of same event | two deliveries, same event id | collapse by id, ack both | process second as new | receiver dedup window |
| Cancel requested mid-effect | cancel state, in-flight marker | finish disposition, then cancel | mark cancelled immediately | user sees later reversal |
| Stale worker after lease expiry | old fence generation | reject writes at receiver | trust heartbeat freshness | receiver without fencing |

Every "forbidden action" cell is a rule that exists because the alternative looks correct in code review and is wrong under failure.

### 7.4 Runtime identity patterns

| Pattern | Narrowing effect | Blind spot | Deployment note |
|---|---|---|---|
| Long-lived service account | none | no tenant binding at all | only for platform-level reads |
| Token exchange per run | audience and scope | exchange service is a deputy | needs policy at exchange |
| Per-tool scoped credential | one resource family | aggregation across tools | issue at dispatch, expire at TTL |
| Signed single-use URL | one object, one verb | long TTL silently widens | cap TTL below incident response time |
| Impersonation with audit actor | attribution kept | escalation if unapproved | require two-person rule |

## 8. Failure Modes & Edge Cases

Each entry states the symptom an operator would actually see, the mechanism, a competing cause, and a check that separates them. All are HYPOTHETICAL teaching cases; none is a reported incident.

### F-01 — Session ID treated as a capability
- **Symptom:** a support user with a session URL reads a conversation from another tenant.
- **Mechanism:** the read path uses the identifier to locate the row and never re-checks tenant.
- **Competing cause:** a routing bug that sends the query to the wrong shard also returns a foreign row.
- **Discriminating check:** re-issue the exact read under tenant B's identity with tenant A's ID; a routing bug fails for both identities, an authorization gap fails only for the wrong one.
- **Numbers:** with 2 tenants and 1 missing predicate, exposure is 1 ordered pair out of 2; with 40 tenants it is 39 pairs.
- **Containment:** disable the deep-link route; rotate any secret embedded in affected transcripts.
- **Fix:** resolve tenant from the authenticated principal and add it to the query predicate; add row-level security beneath it.
- **Verification:** cross-tenant test matrix of 2 tenants × 3 roles × 5 routes = 30 assertions, all denials required.
- **Provenance:** HYPOTHETICAL; unmeasured.

### F-02 — Tenant claim taken from a request header
- **Symptom:** a client-sent `X-Tenant` value selects storage with no membership check.
- **Mechanism:** the gateway forwards the header and the service trusts it after authentication.
- **Competing cause:** a legitimate SSO user whose membership was revoked an hour ago; identity sync lag mimics forgery.
- **Discriminating check:** compare the signed token's tenant claim with the header value; equality means the defect is elsewhere, mismatch is decisive.
- **Numbers:** header trust converts 1 authenticated credential into access to as many tenants as existence checks permit, potentially all 40.
- **Containment:** strip the header at the edge and reject requests carrying it.
- **Fix:** derive tenant from verified claims and load membership server-side at admission.
- **Verification:** send a valid token with a foreign tenant header and assert a 403 at the service, not only at the gateway.
- **Provenance:** HYPOTHETICAL; teaching scenario.

### F-03 — Cache key omits tenant and policy version
- **Symptom:** an answer generated under role `analyst` appears for role `auditor` in the same tenant, or across tenants.
- **Mechanism:** key is `hash(prompt)` while the rendered result depends on authorization-filtered retrieval.
- **Competing cause:** stale entry from before a policy change; both produce "wrong answer" but different blast radius.
- **Discriminating check:** inspect the exact key material and the cached value's source identity; a key without tenant discriminates immediately.
- **Numbers:** 40 tenants × 3 roles = 120 distinct scopes collapsing into 1 key space; the collision probability for a random pair is 1 − 1/120 ≈ 0.9917.
- **Containment:** flush the affected keyspace and serve from the retrieval path until fixed.
- **Fix:** include tenant, role digest and retrieval corpus version in the key; never cache material assembled under a filter without the filter in the key.
- **Verification:** prime as tenant A role analyst, fetch as tenant B; assert miss, then assert the second fetch is correct.
- **Provenance:** HYPOTHETICAL; unmeasured.

### F-04 — Export path traversal
- **Symptom:** an export request returns an object whose key belongs to a different tenant prefix.
- **Mechanism:** client-supplied path segments are concatenated without normalisation, so `../` escapes the prefix.
- **Competing cause:** a signed URL issued earlier for the other tenant, replayed inside its validity window.
- **Discriminating check:** log the canonicalised object key and the authorization decision; traversal shows a rejected key, URL replay shows an authorised key.
- **Numbers:** a 900 s URL TTL means a leaked link is exploitable for 15 minutes; dropping the TTL to 60 s cuts the window by 15×.
- **Containment:** disable exports, revoke outstanding URLs, audit access logs for prefix violations.
- **Fix:** construct keys server-side from opaque identifiers; enforce object-store policy in addition to application checks.
- **Verification:** replay encoded separators (`%2e%2e%2f`), double-encoded variants, and absolute keys; all must 404 at the object store.
- **Provenance:** HYPOTHETICAL; constructed example.

### F-05 — Resume trusts the original authorization
- **Symptom:** a paused run resumes after the initiating user's access was revoked.
- **Mechanism:** resume validates run ownership as recorded at admission but not current membership.
- **Competing cause:** a delegated approval granted later under a new policy version.
- **Discriminating check:** compare current membership and policy version with the values stamped at admission; divergence is the defect, equality points to delegation.
- **Numbers:** approval queue at p95 240 s means a revocation has a 4-minute window on each paused run; 60 paused runs per day gives 240 run-minutes of exposure.
- **Containment:** pause all runs whose initiator lost access; revoke outstanding tool capabilities.
- **Fix:** reauthorize at resume and again immediately before the privileged effect; record both policy versions.
- **Verification:** in a controlled test, revoke membership between pause and resume; assert the transition to running is refused with an audit row.
- **Provenance:** HYPOTHETICAL; no incident record.

### F-06 — Approval bound to a run, not to a payload
- **Symptom:** the approved amount is 4,200 but the submitted amount is 42,000.
- **Mechanism:** the approval row references the run id; the worker recomputes the payload after approval.
- **Competing cause:** canonicalisation difference producing a hash mismatch on an identical action, which fails safe and looks similar in logs.
- **Discriminating check:** recompute the digest of the intended payload and compare with the digest stored at approval time; a mismatch with identical semantics is a canonicalisation bug, a mismatch with different numbers is substitution.
- **Numbers:** with 6 material fields, one unprotected field is 1/6 of the payload surface but potentially 10× the value at risk.
- **Containment:** halt the adapter for that action type; re-approve pending items.
- **Fix:** bind approval to a digest over tenant, actor, destination, amount, currency, action type and expiry; verify at dispatch.
- **Verification:** mutate each material field in turn, 6 cases, each must invalidate the approval.
- **Provenance:** HYPOTHETICAL; illustrative.

### F-07 — Expired approval executed
- **Symptom:** a refund is issued 20 minutes after the approver's window closed.
- **Mechanism:** resume checks that a decision exists and not whether its validity covers the current time.
- **Competing cause:** clock skew between the approval service and the worker makes a valid approval appear expired or vice versa.
- **Discriminating check:** compare trusted server time, stored expiry and measured skew; a consistent 2 s skew cannot explain a 1,200 s gap.
- **Numbers:** TTL 900 s; a 20-minute delay is 1,200 s, exceeding validity by 300 s (33.3% beyond the window).
- **Containment:** freeze the action type, reconcile everything approved in the last hour.
- **Fix:** evaluate expiry at the effect boundary using a single trusted clock; treat expiry as a state transition to `expired`, not as a silent discard.
- **Verification:** resume at TTL ± 5 s; assert refuse-then-require-new-decision on both sides of the boundary.
- **Provenance:** HYPOTHETICAL; constructed.

### F-08 — Duplicate effect after timeout
- **Symptom:** two refunds, 60 s apart, for the same session.
- **Mechanism:** the timeout is interpreted as failure and the retry generates a new key from the current timestamp.
- **Competing cause:** the provider's own retry policy re-sent after its 30 s timeout.
- **Discriminating check:** list both requests with their effect keys and the provider receipt ids; identical keys with two receipts is a provider bug, different keys is an adapter bug.
- **Numbers:** retry cap 3 at 30 s timeouts creates up to 3 sends in 90 s; if the provider dedup window is 600 s, the safe retry horizon is 600 s and 90 s is inside it.
- **Containment:** stop the adapter, enumerate unresolved effects, reverse duplicates externally.
- **Fix:** derive the key from canonical action content and the run id, persist it before the first send, and reuse it across attempts.
- **Verification:** kill the worker between send and response in a fault-injection test; assert one external effect and two recorded attempts.
- **Provenance:** HYPOTHETICAL; teaching case.

### F-09 — Idempotency key collision across payloads
- **Symptom:** a second, different action is silently absorbed and never executed.
- **Mechanism:** the key is built from session id plus action type, so two distinct refunds on one session share a key.
- **Competing cause:** a receiver that treats any key replay as success without comparing payload digest.
- **Discriminating check:** send two different payloads with one key; if the second returns the first's receipt, the key space is too coarse.
- **Numbers:** one session with 4 distinct refunds collapses to 1 executed effect; the loss rate is 3/4 = 75%.
- **Containment:** audit all effects for the affected action type in the retention window.
- **Fix:** include a canonical payload digest or a monotonic intent ordinal in the key; document the key's scope.
- **Verification:** property test that distinct canonical payloads never map to one key, over a corpus of 1,000 examples.
- **Provenance:** HYPOTHETICAL; unmeasured.

### F-10 — Stale worker writes after lease expiry
- **Symptom:** a run's state goes backwards from `waiting_approval` to `running` minutes after failover.
- **Mechanism:** the paused worker's process resumes and writes with a read-modify-write path that ignores version.
- **Competing cause:** two consumers on one partition after a rebalance, both alive and healthy.
- **Discriminating check:** compare lease generation in the write with the store's current generation; a lower generation is decisive, equal generations points to duplicate consumers.
- **Numbers:** 45 s pause against a 30 s TTL yields a 15 s overlap; at 3 writes/s that is 45 writes at risk, all rejected with a fence check.
- **Containment:** fence and block the stale worker's credential; replay the transition table for the last hour.
- **Fix:** require the version predicate on every write and enforce a monotonic fence at the storage layer.
- **Verification:** freeze a worker process for 60 s with SIGSTOP-style tooling, then thaw; assert zero accepted writes from generation 7.
- **Provenance:** HYPOTHETICAL; constructed.

### F-11 — Checkpoint committed after the effect
- **Symptom:** after restart, the run re-executes an effect that already succeeded.
- **Mechanism:** checkpoint ordering is written after the tool call returns, so the crash window excludes the effect record.
- **Competing cause:** asynchronous checkpointing that returns before durable write.
- **Discriminating check:** compare checkpoint commit timestamp with the effect receipt timestamp; commit-after-receipt means ordering, commit-before but missing effect row means loss.
- **Numbers:** at 90 s interval and 30 s effect duration, up to 33.3% of a run's effects can sit in the uncommitted gap.
- **Containment:** pause resumes for the affected run family; reconcile by effect key.
- **Fix:** write intent (key plus canonical payload) durably before dispatch, then commit the receipt; resume treats intent-without-receipt as unknown.
- **Verification:** inject a crash exactly between dispatch and receipt; assert the resumed run reconciles rather than re-sends.
- **Provenance:** HYPOTHETICAL; illustrative.

### F-12 — Replay executes side effects
- **Symptom:** rebuilding a run from the event log creates new tickets.
- **Mechanism:** the replay handler calls the same tool adapter as live execution.
- **Competing cause:** a duplicate event delivered twice by the broker, which also creates two effects but with identical ids.
- **Discriminating check:** compare event ids; a distinct id with an identical payload is a replay, a repeated id is a duplicate delivery.
- **Numbers:** a 7-day log with 40,000 run events replayed naively produces 40,000 dispatches; with an effect-key guard it produces 0 new effects and 40,000 suppressed attempts.
- **Containment:** revoke the replay service's write credential for external adapters.
- **Fix:** separate replay into a pure state-reconstruction mode with no adapter credentials; make suppression observable as a counter.
- **Verification:** replay a recorded day into a sandbox provider and assert zero new provider objects and a nonzero suppression counter.
- **Provenance:** HYPOTHETICAL; no measurement.

### F-13 — Poison event blocks the session
- **Symptom:** one session never advances and its queue depth grows by one per minute.
- **Mechanism:** a malformed event fails deserialisation and is retried indefinitely in place, blocking its partition.
- **Competing cause:** a downstream timeout that also causes repeated retries but advances the offset.
- **Discriminating check:** inspect per-partition offset lag versus per-session attempt counts; a fixed offset with a rising attempt count is poison, a moving offset is a timeout.
- **Numbers:** 1 poison event at 60 retries/hour accumulates 1,440 attempts/day and blocks every session behind it on that partition.
- **Containment:** move the event to a quarantine topic after a bounded attempt count.
- **Fix:** bound attempts per event, record the failure, and continue; a session-level dead-letter state is a better outcome than a stalled partition.
- **Verification:** publish a corrupt payload and assert offset advance within the attempt bound plus one dead-letter record.
- **Provenance:** HYPOTHETICAL; teaching case.

### F-14 — Out-of-order tool results overwrite state
- **Symptom:** an older price quote replaces a newer one in the session state.
- **Mechanism:** results are applied in arrival order with no sequence comparison.
- **Competing cause:** a retried request that legitimately returns a newer value, which arrival order would handle correctly.
- **Discriminating check:** compare tool request sequence numbers with the applied state sequence; a lower applied sequence is decisive.
- **Numbers:** two in-flight calls with 400 ms spread and 2 s response variance reorder with probability near 0.5 on a busy link, so half of all such pairs land wrong.
- **Containment:** freeze state application for the affected tool, rebuild from the authoritative store.
- **Fix:** tag each result with (call id, sequence, observed-at) and apply only if newer per field, not per message.
- **Verification:** invert response order in a test harness; assert the final state reflects the higher sequence.
- **Provenance:** HYPOTHETICAL; unmeasured.

### F-15 — Cancellation reported before effects settle
- **Symptom:** the UI says cancelled while the provider confirms a refund ten minutes later.
- **Mechanism:** the cancel handler transitions to `cancelled` on request receipt rather than on disposition of in-flight effects.
- **Competing cause:** a provider that retried internally after the client abandoned the request.
- **Discriminating check:** compare cancel timestamp with the last effect receipt time; a receipt after cancellation means the terminal state was premature.
- **Numbers:** cancel at 09:06:00Z with a 30 s in-flight timeout and a 10 s poll interval gives up to 40 s of undetermined effects, i.e. 4 poll cycles.
- **Containment:** reclassify affected runs to `cancelling` and notify the requester of the discrepancy.
- **Fix:** model cancellation as `cancelling` with a disposition wait; only the disposition writes `cancelled`.
- **Verification:** cancel during a known slow tool and assert two states observed in order with exactly one terminal write.
- **Provenance:** HYPOTHETICAL; constructed.

### F-16 — Overbroad worker identity
- **Symptom:** a defect in one tool handler can read every tenant's object store.
- **Mechanism:** the worker runs with a platform-wide service account; tenant scoping happens only in application code.
- **Competing cause:** an admin-only tool exposed to general runs, which is a policy error rather than an identity error.
- **Discriminating check:** attempt a cross-tenant read from a run's own credential; success means the identity is overbroad, refusal with a policy-denied code means scoping works.
- **Numbers:** 40 tenant prefixes reachable instead of 1 gives a blast-radius ratio of 40:1; narrowing to a per-run token reduces it to 1:1.
- **Containment:** rotate the service credential, revoke long-lived tokens, review access logs for cross-prefix reads.
- **Fix:** issue a per-run capability scoped to tenant, action family and TTL; keep the platform identity for control-plane tasks only.
- **Verification:** an automated assertion that every outbound call carries a token whose scope set is a subset of the run's granted scopes.
- **Provenance:** HYPOTHETICAL; illustrative.

### F-17 — Secret persisted in a checkpoint
- **Symptom:** a database backup contains a bearer token in cleartext.
- **Mechanism:** the checkpoint serialises the whole worker state, including the credential object used for a tool call.
- **Competing cause:** transcript logging capturing the same token through a debug header.
- **Discriminating check:** grep the checkpoint payload and the transcript independently; presence in both indicates two distinct leaks needing two fixes.
- **Numbers:** a 40 KiB checkpoint per 90 s across 192 long runs/day yields 157.3 MB/day of payload; a single leaked token in that stream is enough for compromise.
- **Containment:** revoke the token, re-encrypt or purge affected objects according to retention rules.
- **Fix:** keep credentials out of serialised state; resolve them at call time from a broker, and scan checkpoint payloads in CI.
- **Verification:** a schema-level test that serialised state matches a whitelist of fields, failing on any unknown key.
- **Provenance:** HYPOTHETICAL; teaching scenario.

### F-18 — Retention deletion leaves copies
- **Symptom:** a deletion request completes but the tenant's conversation is still retrievable from a cache or export bucket.
- **Mechanism:** the delete path covers primary rows only; caches, replicas, exports and search indexes have independent lives.
- **Competing cause:** a soft-delete flag the API honours but background jobs ignore.
- **Discriminating check:** enumerate every store that received the session and query each with the deleted id; a hit in one store localises the gap.
- **Numbers:** 6 stores (primary, replica, cache, search, export, audit) means 6 checks per deletion; covering 5 of 6 leaves 16.7% of the surface unverified.
- **Containment:** purge confirmed copies, extend the deletion runbook, notify the requester per policy.
- **Fix:** maintain a per-session store inventory at write time so deletion can enumerate rather than guess.
- **Verification:** a deletion test that asserts absence in all 6 stores and fails on store count drift.
- **Provenance:** HYPOTHETICAL; unmeasured.

## 9. Complexity & Cost Analysis

### 9.1 Asymptotic behaviour of the operations that matter

- Conditional state transition: one indexed row update, effectively O(1) per transition with a B-tree lookup of depth about 3–4 for millions of rows.
- Isolation check per read: O(1) predicate evaluation at the storage layer, but O(k) in the number of joins if ownership is checked through a membership table of k rows.
- Transcript append: O(1) amortised with an append-only structure, O(n) if each append rewrites a document, which is the usual cause of quadratic session writes.
- Context assembly: O(c) in context tokens c per call, since every token must be serialised; this dominates all other per-call costs by an order of magnitude.
- Retention sweep: O(sessions deleted) per sweep; a full-table scan is O(total sessions) and is the reason sweeps must be keyed by deletion time.
- Reconciliation poll: O(unresolved effects × polls) with a hard cap; unbounded polling converts a storage problem into a rate-limit problem.

### 9.2 Back-of-envelope sizing

**ILLUSTRATIVE.** Keep the Section 5 parameters and vary one at a time.

```text
base case (2,400 sessions/day, 30 days, 12,000 B, 2.4x)
  logical per day   = 28,800,000 B  = 28.8 MB
  physical per day  = 69,120,000 B  = 69.12 MB
  steady state      = 2,073,600,000 B = 2.0736 GB
  per session       = 28,800 B

checkpoints (192 long runs/day, 20 per run, 40 KiB)
  per day           = 157,286,400 B = 157.29 MB
  share of total    = 157,286,400 / (157,286,400 + 28,800,000) = 0.8453

storage price probe (0.12 currency units per GB-month, ILLUSTRATIVE rate)
  monthly for base  = 2.0736 * 0.12 = 0.2488 per month
  monthly with cp   = 0.2488 + 0.1573 * 0.12 = 0.2677 per month
```

Checkpoint payloads dominate at 84.5% of stored bytes, which inverts the usual intuition that transcripts are the storage problem. The reason is arithmetic, not opinion: 20 snapshots per run of 40 KiB exceeds 12,000 bytes of session state per day by more than an order of magnitude for the 8% of runs that are long.

Two consequences that change design:

- Deduplicate checkpoints by content hash; repeated snapshots of an unchanged state are common when a run waits on an approval, and a wait of 900 s produces 10 identical snapshots at 90 s intervals.
- Shorten checkpoint interval only for the expensive part of a run. A single global interval forces transcript snapshots at tool-call frequency, which is the expensive choice.

### 9.3 Cost per completed run

Model spend dominates and is driven by replayed context, so the useful unit is cost per accepted outcome, not cost per call.

```text
ILLUSTRATIVE unit economics for the Section 5 run shape
  model calls per attempt        = 6
  mean context per call          = 9,500 tokens (2,000 summary + 6 turns + retrieval)
  tokens per attempt             = 6 * 9,500 = 57,000
  attempts per run               = 1.1725 (Section 4.3 variant A)
  tokens per run                 = 57,000 * 1.1725 = 66,832.5
  replayed tokens after crash    = 9,500 (one call re-issued from checkpoint)
  total incl. crash replay       = 66,832.5 + 9,500 = 76,332.5
  storage per run (30 day share) = 28,800 B
```

The recovery overhead is 9,500/76,332.5 = 12.4% of token spend, which is large enough that checkpoint placement (before expensive calls, not after) is a cost decision and not only a correctness one.

Environment for any real measurement remains OPEN: database engine and version, broker, hardware class, region and autoscaling behaviour are all unspecified here.
Benchmark plan if the design must be validated: replay a realistic mix where 8% of runs are long, inject one worker crash per 500 runs, one provider timeout per 200 effects, and one revoked-membership resume per 1,000 runs; report p50/p95/p99 for resume latency, duplicate suppression counts and unresolved-effect age.
Report sample size, run dates, software versions and price date with any cost figure; a cost without a price date is not reproducible.

## 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Use | Does not imply | Source |
|---|---|---|---|---|
| Atomic conditional update | one predicate-guarded update changes at most one row per version | exactly one winner per transition | external effects are atomic with it | DERIVED §4.2 |
| Little's law | L = λW for stable long-run averages | cross-check concurrency against measured latency | tail latency or behaviour under growing backlog | Little (1961) |
| Geometric attempt model | P(success ≤ n) = 1 − (1−s)^n for independent constant s | set attempt caps and escalation budgets | independence when failures share a cause | DERIVED §4.3 |
| Row security policies | server-side policies apply to statements subject to role and bypass rules | second enforcement layer below application code | table owners are always constrained | PostgreSQL documentation |
| Idempotent method semantics | HTTP defines idempotence in terms of intended effect of repeats | reason about transport retries | any POST is safe to repeat | RFC 9110 §9.2.2 |
| Lease plus fencing | monotonic token lets a receiver reject a stale holder | safe failover on shared storage | fencing works where the receiver ignores the token | DERIVED §4.3 variant D |
| Zero-trust resource focus | authorization is evaluated per resource and request, not per network location | placement of enforcement points | a specific product satisfies it | NIST SP 800-207 |

### Source and evidence registry

- **S1 — PostgreSQL row security:** <https://www.postgresql.org/docs/current/ddl-rowsecurity.html>, current docs; supports only the qualified statement about policy application and bypass roles; the deployed version and role configuration must be pinned.
- **S2 — Little, J. D. C. (1961):** "A Proof for the Queuing Formula L = λW", Operations Research 9(3); supports the stable long-run identity, not any workload in this sheet.
- **S3 — RFC 9110 §9.2.2:** <https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2>; supports the definition of idempotent method semantics, not a vendor's deduplication behaviour.
- **S4 — NIST SP 800-207 (2020):** Zero Trust Architecture; informs per-resource authorization, not a certification of this design.
- **S5 — OWASP LLM Top 10, prompt injection:** <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>; supports treating retrieved text as untrusted input, not a claim that any single mitigation is sufficient.
- **S6 — Internal scope crosswalk:** unit `INDEX.md` topic list; inventory evidence only, not implementation evidence.

All illustrative numbers in §§4, 5 and 9 were derived here from the stated parameters; none was measured on a deployed system.
Provider-specific behaviour — idempotency key retention, lookup-by-key support, cancellation semantics — is OPEN and requires the provider's primary contract before any retry policy is fixed.
Organisation-specific rules — data classification, deletion deadlines, legal hold, cross-region residency — are OPEN and require owner confirmation.

## 11. Interview / Exam Questions

Scoring per question: 0 unsafe or wrong; 1 partially correct with a critical gap; 2 correct with stated assumptions; 3 correct, defends a tradeoff and names a verification. Answers below are independent; the numbers differ deliberately.

### Q01. What distinguishes a session from a run, and why does the distinction change recovery?
**Answer:** a session is the durable, owned container; a run is one bounded execution with its own budget and terminal state. Recovery operates on runs and must not delete or reopen the session.
**Reasoning:** conflating them makes "retry" look like "restart the conversation", so retries re-ask the user and duplicate effects that belong to an earlier run.
**Worked check:** one session with 3 runs at 1.1725 attempts each implies 3.5 attempts total; the effect keys must embed the run id, giving 3 distinct key families, not 1.
**Boundary:** a compensating action is a new run, never a transition out of `completed`.

### Q02. A caller knows a valid session id. What stops them reading it?
**Answer:** the read predicate binds tenant from verified claims and checks membership; the id only locates a candidate row.
**Reasoning:** identifiers are not secrets and leak through logs, URLs and support tickets, so authorization must not depend on their secrecy.
**Worked check:** with 40 tenants, a missing predicate exposes 39 ordered pairs; a test matrix of 2 tenants × 3 roles × 5 routes gives 30 assertions, all requiring denial.
**Boundary:** privileged database roles bypass application checks, so the database layer needs its own policy and its bypass roles must be enumerated.

### Q03. Two workers read version 8 and both write. What exactly happens under optimistic control?
**Answer:** both issue `where version = 8`; the first affects 1 row and moves to 9; the second affects 0 and must reload and re-decide.
**Reasoning:** the predicate is the concurrency control; a read-modify-write in two statements has no such property.
**Worked check:** at 5 conflicting writes/s against one hot session with 0.4 s service time, Little's law gives L = 5 × 0.4 = 2 writers in flight on average; every overlapping pair yields one row-affected-0, so reloads approach the arrival rate rather than staying a rare path, and the retry loop itself becomes the load.
**Boundary:** the guard protects the row, not an external call made under that version — that still needs an effect key.

### Q04. What does a 30-second lease fail to guarantee once a worker pauses for 45 seconds?
**Answer:** it does not stop the paused worker from executing. It only lets another worker take over, creating a 15-second overlap.
**Reasoning:** expiry is a permission change, not a process signal, and a partitioned process cannot be told it lost the lease.
**Worked check:** overlap 45 − 30 = 15 s; at 3 writes/s, 45 writes are at risk; a monotonic fence rejects 45 of 45.
**Boundary:** if the external receiver ignores the fence token, only key-based deduplication or single-writer serialisation is safe.

### Q05. An effect call times out after the request was sent. Enumerate the dispositions.
**Answer:** committed, not committed, or unknown. Only the provider's receipt or a key lookup can distinguish them; a timeout by itself says nothing about remote state.
**Reasoning:** the response is an observation about the network path, not about the remote transaction.
**Worked check:** 3 sends at 30 s timeouts inside a 600 s dedup window are safe; the same 3 sends with a 90 s window are not, and the failure appears only after 600 s expires.
**Boundary:** with no lookup capability, park as unknown and reconcile out-of-band; never map unknown to failed.

### Q06. How is an approval made non-transferable?
**Answer:** bind it to actor, tenant, canonical payload digest, policy version and expiry, and re-verify all five at dispatch.
**Reasoning:** an approval keyed only to a run is a capability to do whatever the worker later decides.
**Worked check:** with 6 material fields, mutating each in turn gives 6 invalidation tests; a design passing 5 of 6 leaves 16.7% of the surface unprotected.
**Boundary:** canonicalisation must cover semantically relevant fields; a digest over a lossy rendering fails safe but blocks legitimate work.

### Q07. Can a local transaction make an external effect exactly-once?
**Answer:** no, not across an independent service without a shared protocol or receiver-supported key. The achievable guarantee is at-most-once per key at the receiver plus reconciliation.
**Reasoning:** two-phase commit needs both participants; a provider API is not a participant.
**Worked check:** persisting intent before dispatch and receipt after reduces the unknown set to intents without receipts; at 1 timeout per 200 effects and 2,400 effects/day that is 12 unknowns/day needing reconciliation.
**Boundary:** dedup is bounded by the receiver's key retention, so the retry horizon must be shorter than that retention.

### Q08. Why is the transcript not the model context, and where do approvals live?
**Answer:** the transcript is the durable event record; context is a per-call selection that may be summarised or truncated. Approvals live in structured state, never only inside the context.
**Reasoning:** anything summarised away is not auditable, and a summariser is a generative component whose output is not evidence.
**Worked check:** 350 tokens/turn against a 24,000-token budget compacts at turn 69, discarding 20,050 tokens of detail; if an approval were only in the discarded region, it would vanish.
**Boundary:** retrieval may re-insert approved content, but the authority record must exist independently of retrieval success.

### Q09. Cancel is requested while an effect is in flight. What is the correct sequence?
**Answer:** record `cancelling`, stop new dispatches, wait for disposition of in-flight effects, then write `cancelled`.
**Reasoning:** a terminal state that precedes disposition makes the system's own record wrong, which is worse than a slow cancellation.
**Worked check:** 30 s effect timeout plus 10 s poll interval bounds the disposition wait at 40 s for one in-flight call.
**Boundary:** if the effect is irreversible and confirmed, cancellation becomes a reversal run, not a state edit.

### Q10. Size retained state for 2,400 sessions/day, 30 days, 12,000 B, multiplier 2.4 — and say what breaks first.
**Answer:** 2,400 × 30 × 12,000 × 2.4 = 2,073,600,000 bytes (1.9312 GiB); what breaks first is checkpoint storage, at 157.3 MB/day.
**Reasoning:** 20 snapshots × 40 KiB per long run exceeds per-session state for the 8% of runs that are long, so checkpoints reach 84.5% of bytes.
**Worked check:** per session 12,000 × 2.4 = 28,800 B; 72,000 session-days × 28,800 = 2,073,600,000 B, agreeing with the product form.
**Boundary:** the formula assumes stationary cohorts; accumulating transcripts makes it understate by an order of magnitude.

### Q11. Policy tightens while a run waits in `waiting_approval`. What must change?
**Answer:** re-evaluate authorization and the payload against the current policy at resume and again at the effect boundary; if it no longer qualifies, transition to `paused` or request a fresh approval.
**Reasoning:** the approval was issued under a policy version, and the policy version is part of the authority tuple.
**Worked check:** with a 900 s TTL and a p95 queue of 240 s, 26.7% of the validity window is consumed waiting; a policy change in that window affects every queued run at once.
**Boundary:** historic approvals remain valid records; they are not valid authority after a tightening.

### Q12. Session p99 latency rose from 3 s to 30 s; hot-key contention or slow tool?
**Answer:** separate the signals: per-session attempt concurrency and lock wait time indicate contention; tool latency distribution and error rate indicate a downstream problem.
**Reasoning:** both produce queue growth, so queue depth alone is not discriminating.
**Worked check:** Little's law with λ = 12 runs/s and W = 40 s gives L = 480 in system; if measured L is 480 while tool p99 stays at 800 ms, the delay is internal.
**Boundary:** a throttled downstream can present as internal queueing when the adapter's retry loop holds the worker slot.

### Q13. Why does an append-only log alone not make recovery safe?
**Answer:** it needs durable ordering, version-compatible readers, a checkpoint boundary and a rule that replay reconstructs state without invoking adapters.
**Reasoning:** the log records intent; it cannot prove what a remote system committed, so replay of effects repeats them.
**Worked check:** 40,000 events replayed naively produce 40,000 dispatches; with an effect-key guard, 0 new effects and 40,000 suppressed attempts.
**Boundary:** suppression counts must be observable, otherwise the guard's failure is invisible until a duplicate appears.

### Q14. Write the tenant isolation test matrix and say what a green run proves.
**Answer:** 2 synthetic tenants × 3 roles × 7 paths (read, write, list, retrieve, export, resume, tool call) = 42 cases; green proves denial on those paths under those identities only.
**Reasoning:** isolation is a property of enforcement points, so the matrix must name points; unlisted paths and privileged roles remain unproven.
**Worked check:** covering 5 of 6 stores leaves 16.7% of the deletion surface unverified, which is the same arithmetic as path coverage.
**Boundary:** background jobs, admin routes and database owner roles must be added explicitly or the result overstates coverage.

### Q15. What evidence makes this design production-ready?
**Answer:** measured resume latency distribution, duplicate suppression counts, unresolved-effect age, cross-tenant denial tests including privileged roles, deletion proofs across every store, and cost per accepted outcome.
**Reasoning:** each metric maps to a specific invariant in §4; an invariant without a metric is an intention.
**Worked check:** with 2,400 runs/day and a 1-in-500 crash rate, 4.8 crashes/day give about 144 resume observations a month — enough for a p95, not for a p99.
**Boundary:** any figure quoted without workload, versions and dates is not evidence.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Core predicate | tenant from verified claims ∧ membership ∧ policy version ∧ not deleted |
| Core storage formula | L = S × D × B × M; per session = B × M = 28,800 B at the worked values |
| Transition rule | one conditional update on `version`; 0 rows means reload, never overwrite |
| Lease rule | expiry permits takeover; fencing makes the overlap harmless where enforced |
| Checkpoint rule | persist intent before dispatch; a checkpoint after the effect loses the window |
| Effect rule | stable key from canonical payload + run id, persisted before first send |
| Terminal rule | only the transition service writes terminals; no operator edits |
| Isolation rule | enforcement below the code that could forget it, tested as a matrix |
| Sizing warning | checkpoints, not transcripts, dominated at 84.5% of stored bytes here |
| Evidence | all numbers ILLUSTRATIVE; provider semantics OPEN; nothing measured |
