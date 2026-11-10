---
title: "25 Common Issues — diagnostic catalog for async Python, HTTP/WebSocket I/O, LLM inference, containers, and SQL analytics"
topic: "Recurring engineering defects: symptom discrimination, containment, correction, and replay verification"
collection: "02_python_sql_linux_and_engineering_basics"
level: "core"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 25 Common Issues — 02_python_sql_linux_and_engineering_basics

> **Evidence stance.** Every scenario in this file is a constructed teaching case and is labelled ILLUSTRATIVE. What is *not* constructed is the arithmetic: each number printed in Sections 4, 5, 6, 8, 9 and 11 is derived from the stated inputs by the formulas in Section 4 and was recomputed twice, once forward from the inputs and once backward from the claimed result. No organisation, vendor, model deployment or incident is being reported. Where a mechanism is a documented property of CPython, the SQL standard, the HTTP cache/status model, cgroup v2 or the Docker image format, the mechanism is stated as a property of the software, not as a measurement of anyone's system.

> **What this file is.** A triage instrument. It converts one vague complaint into one catalogued defect with a named mechanism, at least two competing mechanisms, a check whose outcome differs between them, a bounded containment that preserves evidence, a correction that changes a contract, and a replay that proves the correction held.

> **What this file is not.** It is not an incident log, not a runbook for one organisation, not a survey of best practice, and not a substitute for the six sheets that own the derivations. Each issue names the sheet that carries its mathematics.

## 1. File Header

**Decision this file supports.** Given an observed defect in a workflow built from this unit's material, choose a first probe that takes under ten minutes, read the result into one of four defect categories, and pick a containment that does not destroy the evidence needed for the correction. The file is written for the person already on call: the symptom is known and the cause is not.

**Why a catalog rather than a checklist.** A checklist fails exactly at the discrimination step. "Check the logs" is not a probe, because a saturated queue, a retry storm and a genuinely slow downstream all produce a latency rise in the same log lines. A catalog earns its place by pairing every symptom with a competing mechanism and a check whose outcome separates them. The check is decisive when, after running it, one row of this file is eliminated.

**The four categories, operationally defined.**
- `VIOLATED_ASSUMPTION` — the code or query is correct against an assumption that no longer holds. The contract drifted; the algorithm did not. Ten issues in this catalog are of this kind.
- `NUMERICAL_EDGE` — a boundary value produced a finite, plausible, wrong answer: zero, NULL, a tie, a rounding step, a frame boundary, a clamp, a unit change. Thirteen issues here are boundary defects.
- `SCALE_FAILURE` — the mechanism is right at low volume and its cost curve turns vertical: unbounded memory, linear-in-N work per request, synchronised retries, duplicate-key fanout. Six issues.
- `CONCEPT_CONFUSION` — two quantities that share a name were swapped: concurrency for parallelism, per-token loss for per-byte loss, mitigation for correction, a score for an authority, image size for cold-start time. Seven issues.

**Distribution, stated so the counts can be audited.** The catalog carries 36 issues, which is above the collection's minimum of 25 because each of the unit's six sheets contributes at least four defects a practitioner actually meets. By layer: async and Python 9, HTTP/WebSocket 7, LLM inference 7, containers 6, SQL analytics 7. By category: 10 violated assumption, 13 numerical edge, 6 scale failure, 7 concept confusion. The file name retains the collection's `25` by convention; the number is a floor, not a cap.

**Severity discipline.** Severity is stated as consequence inside a declared scope — one nightly batch, one tenant's dashboard, one validator path, one deploy — never as an incident claim. A defect that silently corrupts a published number outranks one that fails loudly, because the loud one is already contained by its own error and has therefore already produced its evidence.

**How to read an entry.** Seven fields per entry: symptom (what a monitor or a human reports), mechanism (why the code produces it), competing causes (the look-alikes, at least two), discriminating check (the probe that separates them), correction (the contract change), prevention (the check that keeps it fixed), verification (the replay that proves it). Section 8 gives the scannable table; the entries that carry numbers get prose as well, because a table cell cannot hold a derivation.

**Assumed environment.** CPython 3.11 or newer with `asyncio`; an HTTP/1.1 and WebSocket client; a chat-completion LLM served behind an HTTP API; containers under a modern engine with cgroup v2; a columnar SQL engine whose window functions follow the SQL standard's default frame. Where a behaviour is version-specific the version is named, because several of these defects change shape between versions.

**Scope boundaries.** Out of scope and deliberately so: security exploitation, cluster capacity planning, model training defects, and anything requiring a specific vendor's console. Capacity questions enter only where a queueing number is needed to separate causes, and then only the standard formulas of Section 4.

**Review trigger.** Re-read the entry for any issue whose dependent tool has a major-version change: `asyncio` cancellation semantics, HTTP client defaults, the tokenizer used for cost arithmetic, the container runtime's signal handling, and the SQL engine's default window frame.

Takeaway: the value of this file is not the list of defects; it is the pairing of each defect with a probe that a responder can finish before the incident call ends.

## 2. Concept Map / Mental Model

Every entry is reached by the same five-stage path, and the order is not negotiable. Containment before diagnosis is how incidents become outages; diagnosis before containment is how a two-minute defect becomes a two-hour one. The diagram puts containment third because the first two stages are read-only and cheap.

```text
  observed symptom ("slow", "wrong number", "stuck", "duplicate")
        |
        v
  (1) CLASSIFY   shape of the wrongness: arithmetic / contract /
        |        scale / naming  -> decides which probe set applies
        v
  (2) PROBE      one decisive measurement, recorded with its timestamp
        |        and the exact input that produced the complaint
        v
  (3) CONTAIN    bounded, reversible, evidence-preserving:
        |        defer, cap, disable one path, serve a prior result
        v
  (4) CORRECT    change the contract or the arithmetic, not the symptom
        |
        v
  (5) REPLAY     triggering record + nearest boundary + one neighbour
                   pass -> close with a reason code
                   fail -> the mechanism was misidentified; back to (2)
```

Two rules make the path work. Stage 2 must produce a *number or a diff*, never an opinion: "it looks like a retry storm" is a hypothesis, "attempts per logical request moved from 1.02 to 1.24" is a probe. Stage 3 must leave the triggering record intact; a containment that truncates the payload, rotates the log or overwrites the failed row deletes the evidence that stages 4 and 5 depend on.

**The second axis: where the defect lives.** Category says what kind of wrongness it is; layer says where to look first. Crossing the two turns a catalog into a triage instrument, because some cells are nearly empty and can be skipped under time pressure.

```text
layer            VIOLATED_ASSUMPTION   NUMERICAL_EDGE        SCALE_FAILURE        CONCEPT_CONFUSION
async/Python     cancellation contract  timeout budget        unbounded gather     parallel vs concurrent
HTTP/WebSocket   status-code contract   Retry-After units     retry amplification  liveness definition
LLM inference    template identity      clamp at 1e-5         context cost         token vs byte loss
container        PID 1 signal path      grace vs drain time   layer cache churn    image size vs start
SQL analytics    join-key uniqueness    boundary inclusion    hot-key scan         percentile averaging
```

The sparse cells are the useful part. A `CONCEPT_CONFUSION` in the async layer is almost always the words "parallel" and "concurrent" used interchangeably, so it resolves with one question rather than a probe. A `NUMERICAL_EDGE` in the SQL layer is almost always a boundary timestamp or a NULL, so it resolves with one count.

**Why misclassification costs time.** A `SCALE_FAILURE` treated as a `VIOLATED_ASSUMPTION` sends the responder to read a request handler that is correct code operating outside its envelope; twenty minutes disappear before anyone plots volume against latency. The reverse error is worse: a contract violation treated as a capacity problem invites extra workers, which multiplies the wrong effect — with an idempotency defect, doubling workers roughly doubles the duplicate side effects.

**The three dominant look-alike pairs, stated once.** A retry storm and a downstream slowdown both raise p99 and lower throughput, and only attempts-per-request separates them. A truncated response and a wrong answer both present as "the model did not answer", and only a substring assertion against the exact prompt separates them. A fanout join and genuine growth both change revenue, and only the row-count ratio before and after the join separates them.

**Cost of a probe, stated so probes get budgeted.** A probe is worth running when the expected cost of running it is below the expected cost of guessing. With a 25 % prior that a given mechanism is present, a probe with 0.90 sensitivity and 0.95 specificity has a posterior of 0.857 when positive (Section 4.3), and a probe costing 6 minutes against 45 minutes of rework has expected cost 12.43 minutes versus 22.5 minutes for the blind guess. Probes are cheap; rework is not; the arithmetic is in Section 5.

Takeaway: classify first, probe with a number, contain without destroying evidence, correct the contract, replay at the boundary — and use the layer to decide which single question to ask before reaching for a tool.

## 3. Foundational Definitions

**Symptom.** The externally observable behaviour that generated the complaint, stated with no cause attached: "p99 rose from 180 ms to 940 ms between 02:10 and 02:40 UTC". A symptom that already names a cause is a hypothesis in disguise and must be rewritten before classification.

**Mechanism.** The causal chain inside the system that produces the symptom in the vocabulary of the component: "each retry re-sends a 22,000-token prompt, so three attempts triple prefill work per logical request". A mechanism is falsifiable; "the service was overloaded" is the symptom with different words.

**Competing cause.** A second mechanism that would produce the same symptom. The catalog requires at least two per entry because a single-cause entry cannot be discriminated: it can only be confirmed by acting and seeing the symptom stop, which is the slowest and most destructive test available.

**Discriminating check.** A measurement, diff or count whose outcome differs between the competing causes. It must be cheap enough to run during the incident, specific enough that its result eliminates a row, and recorded together with the input that produced it. "Inspect the logs" is not a check; "count validator rejections per 1,000 requests, split by prompt-template hash" is.

**Containment.** A bounded, reversible action that limits harm while the cause is unknown: deferring items, capping intake, disabling one code path, serving a previously computed result. Containment is judged on whether it preserves evidence and can be undone, never on whether it fixes anything.

**Correction.** The change that removes the mechanism: a contract clause, a precondition check, a corrected formula, an ownership assignment. A correction is falsifiable because the replay can fail.

**Verification by replay.** Re-running the triggering record plus its nearest boundary case plus one ordinary neighbour after the correction, comparing output, reason code, side-effect count and owner action against expected values. Boundary inclusion matters: a fix for `<` versus `<=` replayed only on interior cases is not verified.

**Reason code.** A short, stable token attached to each replay outcome (`OK`, `DUPLICATE_SUPPRESSED`, `BOUNDARY_EXCLUDED`, `CAP_REACHED`, `DEFERRED`). Reason codes make replay results comparable across runs; a bare pass/fail cannot distinguish a fix from a skipped case.

**Idempotency key.** A caller-supplied value identifying a logical operation across attempts, so a repeated delivery returns the first result instead of performing the effect twice. The key must exist before the first attempt and be persisted with the effect in one transaction; a key generated inside the retry loop identifies attempts, not operations.

**At-least-once delivery.** A transport guarantee that every message arrives one or more times. Every retry mechanism and every queue in this unit is at-least-once, which is why the effect path — not the transport — carries the deduplication duty.

**Backpressure.** A signal from consumer to producer that its rate exceeds sustainable service, expressed as a bounded queue that blocks or rejects rather than growing. A system without backpressure converts a rate problem into a memory problem, which is strictly harder to diagnose because the memory problem outlives the rate spike.

**Concurrency versus parallelism.** Concurrency is interleaved progress on one or more execution units; parallelism is simultaneous execution on several. Forty coroutines against eight permits are concurrently issued and serially executed in five waves, and calling that "parallel" predicts a speedup eight permits cannot deliver.

**Fanout.** The multiplication of rows when a join key is not unique on the right side. One parent row joined to 1.30 children on average yields 1.30 rows per parent, and every downstream `SUM` is inflated by that factor — a numerical defect produced by a data-shape assumption.

**Three-valued logic.** SQL's treatment of `NULL` as unknown, so `NULL = NULL` evaluates to unknown and a predicate on a nullable column silently drops rows from an inner join or a `WHERE`. It is the most common `NUMERICAL_EDGE` in the unit's SQL sheets.

**Window frame.** The set of rows a window function sees for each row, defined by `ROWS` (physical offsets) or `RANGE` (value offsets, with peers treated together). The default frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes every peer of the current row and therefore differs from a `ROWS` running total whenever the ordering key ties.

**Censoring.** Reporting only the part of a distribution that was observed, for example a log-probability truncated to the top five candidates and clamped at a floor. Censored values substitute a bound for a measurement, so a mean over them is biased toward the floor, not merely noisier.

**Per-token versus per-byte loss.** Two normalisations of the same total surprisal, divided by tokens or by bytes. They differ by the tokenizer's bytes-per-token ratio, so a per-token comparison across tokenizers measures the tokenizers as much as the models.

**Percentile pooling.** Combining latency samples from several shards before taking a percentile. A percentile is a rank, not an average, so pooling changes which request occupies the rank; averaging per-shard p99 values produces a number that describes no request at all.

**Cache invalidation point.** The layer in a container build after which every later layer must be recomputed because an earlier input changed. Copying source before installing dependencies moves the invalidation point to the top of the build and turns a cached rebuild into a full one.

**Signal forwarding.** The path by which a stop signal reaches the process that owns the work. A shell started as PID 1 does not forward signals to its children by default, so the container receives the stop and the workload never sees it.

**Token budget.** The allocation of a model call's context window across system prompt, retrieved documents, conversation history, tool schemas and reserved output. A budget is a constraint on retrieval depth, not a suggestion, and the sum of the parts is what the bill is computed from.

**Denominator.** The count in the bottom of any rate. Monthly and 30-day denominators differ by 24/23 in a 31-day month — a 4.35 % difference that is large enough to be reported as a trend and small enough to be missed in review.

**Connection pool.** A bounded set of reusable transport handles shared by coroutines. It is a permit system that most teams forget they configured, and when it is smaller than the semaphore the pool, not the semaphore, is the capacity contract; a semaphore of 40 in front of a pool of 8 gives a throughput of 8 service times.

**Timeout versus deadline.** A timeout bounds one attempt; a deadline bounds the whole logical operation and propagates downward so each layer can abandon work that can no longer be delivered. A retry loop with a per-attempt timeout and no deadline has no upper bound on elapsed time, which is the defect behind CI-05.

**Retry budget.** A cap expressed as a ratio of retries to successes rather than as a count of attempts. At 10 % of 40 successes per second the added load cannot exceed 4 attempts per second however many requests fail, which is why a budget bounds a failure mode that an attempt cap cannot.

**Poison message.** A message that fails deterministically, so each redelivery fails identically and consumes a full attempt. Without a per-message attempt counter it is indistinguishable from a transient failure and will loop until somebody drains the queue by hand.

**Dead-letter queue.** The destination for messages that exhausted their attempts, holding the original payload and the failure reason. Its entire value is that the payload survives; a dead-letter path that stores only an identifier has preserved the count and destroyed the evidence.

**Cursor pagination.** Paging by a stable key instead of an offset. An offset page over a table receiving inserts skips rows when something is added before the cursor and repeats rows when something is deleted, and both errors are silent in the page contents.

**Upsert.** Insert-or-update against a unique constraint, so replaying a record converges on one row. An upsert keyed on a mutable natural key is not idempotent, because correcting the record changes the key and produces a second row.

**Layer digest.** The content hash that identifies a cached filesystem layer. Deploying by digest is what makes "the tested artifact" a well-defined phrase; deploying by tag ships whatever the tag resolved to at pull time, which is the defect in CI-29.

**Readiness versus liveness.** Liveness asks whether a process should be restarted; readiness asks whether it should receive traffic. Using liveness for readiness restarts healthy processes during a slow start, and using readiness for liveness sends traffic to a process whose loop has already died.

**Attempt identifier.** A value naming one transport attempt, distinct from the operation's idempotency key. Logs carrying only one of the two cannot separate retry amplification from duplicate effects, because both manifest as more records than operations.

**Backlog age.** The age of the oldest unprocessed item, which is the only queue metric that reflects user-visible delay. Depth alone is meaningless without the service rate: 10,000 pending items at 5,000 per second is two seconds of work and a non-event.

**Cardinality explosion.** Growth in distinct label values on a metric, which multiplies series and cost. A request identifier used as a label turns one counter into millions and is the most common self-inflicted monitoring outage.

**Index selectivity.** The share of rows a predicate keeps. A join key with 4.2 % NULLs and 1.30 children per parent has an effective selectivity that neither statistic states alone, which is why CI-30 and CI-31 are tested as a pair rather than separately.

Takeaway: define the eight operational terms — symptom, mechanism, competing cause, check, containment, correction, replay, reason code — before using them, because a team that conflates symptom and mechanism will spend the incident arguing about definitions.

## 4. Core Formulas with Full Derivation

Five results do most of the work in this catalog. Each is derived in the five-part anatomy: the formula, what each symbol means and its units, the conditions under which it holds, a numerical example with intermediates, and the interpretation that a responder acts on. Section 4.4 states where each one breaks.

### 4.1 Base case

The base case is a single logical request that may be attempted more than once. Everything else in this catalog that mentions load is a perturbation of this result.

```text
FORMULA [CI-F1]  expected attempts per logical request
    A(r, n) = 1 + r + r^2 + ... + r^(n-1) = (1 - r^n) / (1 - r)

WHERE
    r   independent per-attempt failure probability       [dimensionless, 0 <= r < 1]
    n   maximum attempts per logical request (1 <= n)     [count]
    A   mean transport attempts per success or final fail [attempts per operation]

CONSTRAINT
    attempts are independent; the cap n is enforced; no attempt is
    retried for a non-retryable error; r does not change between attempts

NUMERICAL EXAMPLE  r = 0.20, n = 3
    A = (1 - 0.2^3) / (1 - 0.2)
      = (1 - 0.008) / 0.8
      = 0.992 / 0.8
      = 1.2400 attempts per logical request

INTERPRETATION
    0.24 of every logical request is extra load caused by retrying.
    At 40 logical requests/s that is 9.6 additional attempts/s, so
    the downstream sees 49.6 attempts/s, not 40.
```

The second base result converts attempts into a capacity fraction. It is the reason a retry storm is a `SCALE_FAILURE` and not merely a latency problem: the extra attempts consume the same permits as useful work.

```text
FORMULA [CI-F2]  utilisation after retry amplification
    rho = lambda_0 * A(r, n) * S / c

WHERE
    lambda_0  logical arrival rate                        [requests/s]
    S         mean service time per attempt               [s/attempt]
    c         permits or workers available to attempts    [count]
    rho       fraction of capacity consumed               [dimensionless, 0..1 for stability]

CONSTRAINT
    rho < 1; S is the mean of the same distribution the permits serve;
    every attempt obtains exactly one permit for the whole of S

NUMERICAL EXAMPLE  lambda_0 = 40, A = 1.24, S = 0.180, c = 10
    offered  = 40 * 1.24            = 49.6 attempts/s
    ceiling  = c / S = 10 / 0.180   = 55.56 attempts/s
    rho      = 49.6 / 55.56         = 0.893

INTERPRETATION
    89.3 % of capacity is consumed, and 10.7 % is left for the tail.
    The same workload without retries is rho = 40/55.56 = 0.720.
    Retries alone moved the system from comfortable to critical.
```

### 4.2 General case

The general case replaces "a permit is either busy or free" with a queue in front of the permits. The standard result is Erlang C for the probability that an arriving attempt must wait, and the derived mean wait.

```text
FORMULA [CI-F3]  Erlang C wait for c identical permits
    a      = lambda_eff * S                      offered load      [Erlangs]
    rho    = a / c                               utilisation       [0..1]
    P_wait = (a^c / (c! (1 - rho))) / (sum_{k=0..c-1} a^k/k! + a^c/(c!(1-rho)))
    W_q    = P_wait * S / (c * (1 - rho))        mean queue wait   [s]
    W      = W_q + S                             mean time in system [s]
    L      = lambda_eff * W                      mean in system    [attempts]

WHERE
    c, S, lambda_eff as above; a is dimensionless but named Erlangs

CONSTRAINT
    rho < 1 strictly; arrivals Poisson; service exponential; c parallel
    identical permits; single queue, no priority classes, no abandonment

NUMERICAL EXAMPLE  c = 10, S = 0.180, lambda_eff = 49.6
    a      = 49.6 * 0.180        = 8.928
    rho    = 8.928 / 10          = 0.8928
    P_wait = 0.648
    W_q    = 0.648 * 0.180 / (10 * 0.1072) = 0.109 s
    W      = 0.109 + 0.180       = 0.289 s
    L      = 49.6 * 0.289        = 14.33 attempts in system

INTERPRETATION
    The wait is 60 % of the service time at 89 % utilisation, and it
    grew from 16 ms at 40 requests/s. Queueing, not service, owns the
    tail; the fix is fewer attempts or more permits, never a longer timeout.
```

The general-case counterpart for work admitted in fixed waves (a batch of N items against k permits, each item independent) is exact rather than asymptotic, and is the one formula a responder can compute on a whiteboard.

```text
FORMULA [CI-F4]  wall time for a finite batch under k permits
    T = ceil(N / k) * L_unit + R

WHERE
    N       items in the batch                              [count]
    k       permits that may run at once                    [count]
    L_unit  mean latency of one item                        [s]
    R       residual work that is serial (setup, commit)     [s]

CONSTRAINT
    every item needs exactly one permit for its whole duration;
    items are independent; the batch is fully materialised before dispatch

NUMERICAL EXAMPLE  N = 40, k = 8, L_unit = 0.180, R = 0
    waves = ceil(40 / 8) = 5
    T     = 5 * 0.180    = 0.900 s
    ideal with k = 40    = 0.180 s

INTERPRETATION
    The claim "40 concurrent tasks finish in 0.18 s" is false by 5x.
    It is only true if k >= N, and k is set by the smallest of
    permits, connection pool, downstream quota and file descriptors.
```

### 4.3 Variants

**Variant 1: a probe is a test, so it has sensitivity and specificity.** Choosing a probe is a decision under uncertainty, and the arithmetic is Bayes.

```text
FORMULA [CI-F5]  posterior that the mechanism is present, given a positive probe
    PPV = (sens * p) / (sens * p + (1 - spec) * (1 - p))
    NPV = (spec * (1 - p)) / (spec * (1 - p) + (1 - sens) * p)

WHERE
    p     prior probability the mechanism is present     [0..1]
    sens  P(probe positive | mechanism present)          [0..1]
    spec  P(probe negative | mechanism absent)           [0..1]
    PPV   P(mechanism present | probe positive)          [0..1]

CONSTRAINT
    the probe's error rates were estimated on cases resembling this one;
    the prior is honest; the probe is not run twice and counted twice

NUMERICAL EXAMPLE  p = 0.25, sens = 0.90, spec = 0.95
    PPV = (0.90 * 0.25) / (0.90 * 0.25 + 0.05 * 0.75)
        = 0.225 / (0.225 + 0.0375)
        = 0.225 / 0.2625
        = 0.8571
    NPV = (0.95 * 0.75) / (0.95 * 0.75 + 0.10 * 0.25)
        = 0.7125 / 0.7375 = 0.9661

INTERPRETATION
    A positive probe is right 85.7 % of the time and a negative one
    96.6 % of the time. A negative result is the stronger signal, so
    design the probe to be run first on cases where the prior is low.
```

**Variant 2: fanout and coverage are the same multiplication seen from two sides.** Fanout multiplies rows; coverage divides them; both are ratios that must be stated with their denominator.

```text
FORMULA [CI-F6]  row multiplication across a join, and revenue inflation
    f_row  = rows(after join) / rows(parent)
    S_bad  = f_row * S_good
    delta  = S_bad - S_good = (f_row - 1) * S_good

WHERE
    f_row    mean children per parent on the joined key   [rows per parent]
    S_good   sum over parent rows only                    [currency]
    S_bad    sum after the join                           [currency]

CONSTRAINT
    the parent side is unique on the join key; the child side is not;
    every child joins to exactly one parent; no filter is applied after

NUMERICAL EXAMPLE  parents = 4,800,000 rows, children = 6,240,000 rows
    f_row = 6,240,000 / 4,800,000 = 1.30
    S_good = 4,800,000 units at 1.00 per row = 4,800,000
    S_bad  = 1.30 * 4,800,000      = 6,240,000
    delta  = 0.30 * 4,800,000      = 1,440,000  (+30.0 %)

INTERPRETATION
    A 30 % revenue jump with flat order counts is a join defect until
    proven otherwise; the discriminator is COUNT(*) versus
    COUNT(DISTINCT parent_key) over the same window.
```

**Variant 3: censoring biases a mean by a computable factor.** When values below a floor are replaced by the floor, the mean is not noisy — it is shifted, and the shift is bounded by the floor and the censored share.

```text
FORMULA [CI-F7]  censored mean and its bias factor
    mean_cens = (sum_{i not censored} s_i + m * ln(floor)) / N
    bias      = mean_cens / mean_true

WHERE
    N       tokens in the sequence                              [count]
    m       tokens whose value was replaced by the floor        [count]
    floor   clamp value in probability space (e.g. 1e-5)        [probability]
    s_i     natural-log surprisal of token i                    [nats]
    ln(floor) = ln(1e-5) = -11.5129                             [nats]

CONSTRAINT
    the floor is applied per token before the mean; N includes the
    censored tokens; the true values are below the floor, not above it

NUMERICAL EXAMPLE  N = 14, m = 1, floor = 1e-5, s_i = -0.12 for 13 tokens
    sum      = 13 * (-0.12) + (-11.5129) = -1.56 - 11.5129 = -13.0729
    mean_cens = -13.0729 / 14            = -0.9338 nats/token
    mean_true = -0.12 nats/token
    bias      = -0.9338 / -0.12          = 7.78x

INTERPRETATION
    One clamped token in fourteen inflates the reported loss 7.8x.
    Two clamped tokens give -1.7476 and 14.6x. A loss that moves by
    a factor when the sequence length changes is a censoring artefact,
    not a model change.
```

**Variant 4: cold start is a sum of transfer and initialisation, not a function of image size alone.** The variant matters because teams optimise the additive term they can see.

```text
FORMULA [CI-F8]  cold start with a pull and an init phase
    T_cold = B_needed / beta + m + t_init

WHERE
    B_needed  bytes that must be fetched before first request   [MB]
    beta      effective pull bandwidth per replica              [MB/s]
    m         engine overhead before layers are usable          [s]
    t_init    process start, import, connection warm-up         [s]

CONSTRAINT
    layers are fetched on demand and not already on the node;
    decompression is included in beta; no registry rate limiting

NUMERICAL EXAMPLE  B_needed = 1,200 MB, beta = 180 MB/s, m = 0, t_init = 9 s
    pull   = 1,200 / 180 = 6.67 s
    T_cold = 6.67 + 9    = 15.67 s
    warm   = 1.2 s
    ratio  = 15.67 / 1.2 = 13.1x

INTERPRETATION
    43 % of the cold start is transfer and 57 % is initialisation.
    Halving the image saves 3.3 s; halving imports saves 4.5 s.
    Neither alone reaches the 1.2 s warm number.
```

**Variant 5: backlog age is the queue's integral, and it is the only queue metric with a clock.** Depth says how much work is waiting; age says how long the oldest item has waited, and the two diverge exactly when the service rate is close to the arrival rate.

```text
FORMULA [CI-F9]  backlog age under a rate deficit
    B(t) = B0 + (lambda_in - mu_out) * t          while lambda_in > mu_out
    age(t) = B(t) / mu_out                        as a lower bound on the oldest item

WHERE
    B0         backlog at the moment the deficit opened          [items]
    lambda_in  arrival rate                                      [items/s]
    mu_out     sustainable service rate                          [items/s]
    age        time the oldest unprocessed item has waited        [s]

CONSTRAINT
    the deficit is sustained; service is FIFO; mu_out does not
    degrade as the backlog grows (a queue that slows as it fills
    diverges faster than the linear term predicts)

NUMERICAL EXAMPLE  B0 = 0, lambda_in = 12, mu_out = 9.5, t = 40 s
    B(40)  = 0 + (12 - 9.5) * 40 = 100 items
    age    = 100 / 9.5           = 10.5 s

INTERPRETATION
    A depth of 100 items reads as "fine" in a dashboard that shows
    depth against a 500-item limit. The age says the oldest item has
    waited 10.5 s, which is the number a user experiences, and it is
    already past most interactive budgets.
```

**Variant 6: the connection pool is the ceiling nobody sized.** The permit arithmetic in CI-F2 and CI-F3 uses `c`, and `c` is the minimum over every shared resource. Substituting the semaphore alone is the most common arithmetic error in capacity work because the semaphore is the visible number.

```text
FORMULA [CI-F10]  effective concurrency and its binding resource
    c_eff = min(c_sem, c_pool, q_down * S, c_fd)
    ceiling = c_eff / S

WHERE
    c_sem   semaphore permits                                   [count]
    c_pool  connection handles                                  [count]
    q_down  downstream quota                                    [requests/s]
    c_fd    file descriptors available to the process           [count]
    S       mean service time per attempt                        [s]

CONSTRAINT
    each request holds exactly one unit of each resource it uses;
    the quota is expressed per second and must be multiplied by S
    to be comparable with a count

NUMERICAL EXAMPLE  c_sem = 40, c_pool = 8, q_down = 100, c_fd = 1024, S = 0.180
    quota in count units = 100 * 0.180 = 18
    c_eff = min(40, 8, 18, 1024) = 8
    ceiling = 8 / 0.180 = 44.44 attempts/s

INTERPRETATION
    The semaphore claims 40 permits and the throughput is 44.44/s,
    not 222.2/s. Any capacity plan built on 40 is wrong by 5x, and
    the discrepancy shows up as queue wait with the semaphore
    visibly not saturated — the signature in CI-01 and CI-09.
```

### 4.4 Limits and counterexamples

**Limit of CI-F1.** The geometric sum assumes the cap is enforced and that `r` is constant across attempts. Remove the cap and `A(0.2, infinity) = 1/(1-0.2) = 1.25`: a 24 % overhead becomes a 25 % overhead, which is why "we cap retries" is worth only about one percentage point of load at `r = 0.2`. At `r = 0.4`, though, the cap matters much more: `A(0.4, 3) = 1.56` versus `1/(1-0.4) = 1.667`. The formula also fails when failures are correlated — a single downstream outage makes every attempt fail together, and the independence assumption then understates the retry load by a large factor.

**Limit of CI-F3.** Erlang C requires `rho < 1` strictly. Substitute `lambda_eff = 55.56` at `c = 10, S = 0.18`: `a = 10.0`, `rho = 1.0`, and `W_q` diverges. The formula returns nothing useful at the boundary, which is itself the diagnostic: a queue whose wait is undefined is a queue at its ceiling, and no timeout increase changes that. The formula also assumes exponential service; a service with a heavier tail produces a larger wait at the same utilisation, so the computed wait is a lower bound when the downstream is bursty.

**Limit of CI-F4.** The wave formula assumes every item needs one permit for its whole life. If an item releases its permit while awaiting I/O — normal in `asyncio` where the permit guards only the connection — the effective concurrency is the connection pool, not the semaphore. Substituting `k = 40` while the pool holds 8 connections returns 0.18 s and is wrong by 5x; the correct substitution is `k = min(semaphore, pool, quota)`.

**Limit of CI-F5.** Bayes needs calibrated error rates. If the "probe" is a human reading logs, `sens` and `spec` are unknown and the arithmetic collapses to the prior. A second probe on the same evidence is not independent: applying the formula twice with the posterior as the new prior overstates confidence, which is why a second test only earns its cost when its errors are independent of the first.

**Limit of CI-F6.** Fanout is a mean. When the child table has a skewed distribution, the mean ratio hides local multiplication: a key with 40 children inflates that key's contribution 40x while the global ratio stays 1.30. The formula also breaks with many-to-many joins, where the ratio equals the mean children per parent on both sides and the sum is inflated by the product.

**Limit of CI-F7.** The bias formula assumes the floor is below the true values it replaces. If the floor is *above* the true value — for instance clamping at `1e-5` when the true probability was `1e-7` — the substituted value is an upper bound and the mean is biased toward zero, not away. The sign of the bias depends on which side the clamp sits, so the check is the direction of the movement, not its size.

**Limit of CI-F8.** The pull term assumes layers are fetched on demand from a cold node. On a node with a warm layer cache, `B_needed` is only the changed layers, and `T_cold` collapses toward `t_init`. This is why the same image shows 15.67 s on a fresh node and 2.1 s on a warm one, and why a p50 cold-start improvement can coexist with an unchanged p99.

**Limit of CI-F9.** The linear term holds only while the service rate is independent of the backlog. A queue whose per-item cost grows with depth — retries against a saturated downstream, a cache that thrashes, a database whose working set no longer fits — diverges faster than `(lambda_in - mu_out) * t`, and the age formula understates. The formula also fails at `lambda_in = mu_out` exactly: the backlog is then a random walk, depth stays bounded in expectation but the age grows without bound because the oldest item is never guaranteed service.

**Limit of CI-F10.** The minimum is over resources held for the whole of `S`. A resource held for part of the request — a database connection released before the response is written — has an effective count of `c_res * S / S_held`, so substituting its raw count overstates capacity. The formula also assumes no resource is held across two requests; a pooled connection reused while the first request is still writing breaks the one-unit-per-request premise and makes the quota term non-linear.

Takeaway: every formula in this catalog is a comparison instrument with a stated validity condition; when the condition fails, the formula's silence is itself the diagnosis.

## 5. Worked Numerical Example

**The complaint.** An internal analytics API reports p99 latency up by a factor of five and model spend up 24 % overnight. No deploy is recorded in the window. This is an ILLUSTRATIVE case built from the unit's arithmetic; the inputs are chosen to make each step checkable.

**Inputs, stated before any computation.**
- Baseline logical arrivals `lambda_0 = 40` requests/s; a retry cap of `n = 3` is configured with no jitter and no budget.
- Per-attempt failure probability `r = 0.20`, taken from the validator-rejection counter (one probe, run first).
- Service time per attempt `S = 0.180` s; permits `c = 10` (the semaphore), connection pool 32, downstream token quota 40 attempts/s.
- A summariser call carries 22,000 input tokens and 800 output tokens; input priced 3.00 per million tokens, output 15.00 per million.
- Eight of the forty coroutines perform a synchronous parse of about 0.25 s inside the loop.

**Step 1 — attempts per logical request.** Probe: divide transport attempts by logical requests over the same minute. The counter reports 62,000 attempts against 50,000 logical requests.

```text
attempts/request = 62,000 / 50,000 = 1.24
predicted        = (1 - 0.2^3) / (1 - 0.2) = 1.2400
```

Agreement to three decimals means the retry cap and `r` explain the transport arithmetic exactly. This eliminates "the client is duplicating requests somewhere else" and leaves retry amplification as the mechanism.

**Step 2 — offered load and utilisation.** The downstream does not see 40 requests/s; it sees 49.6.

```text
offered to downstream = 40 * 1.24     = 49.6 attempts/s
permit ceiling        = 10 / 0.180    = 55.56 attempts/s
rho                   = 49.6 / 55.56  = 0.8928
a (Erlangs)           = 49.6 * 0.180  = 8.928
P_wait (c = 10)       = 0.648
W_q                   = 0.648 * 0.180 / (10 * (1 - 0.8928)) = 0.109 s
W                     = 0.109 + 0.180 = 0.289 s
```

Compare with the same system at `lambda_0 = 40` and no retries: `a = 7.2`, `rho = 0.720`, `P_wait = 0.254`, `W_q = 0.016` s, `W = 0.196` s. Queueing went from 16 ms to 109 ms — a 6.8x increase in wait — while service time did not move at all. That explains a 47 % rise in the mean and, through the queue's tail, a large part of the p99 rise.

**Step 3 — the residual.** The reported p99 is five times 180 ms, so about 900 ms. The queue model gives a mean of 289 ms; the gap needs a second mechanism. Eight coroutines each block the loop for 0.25 s, so one pass through the batch stalls every other coroutine for `8 * 0.25 = 2.0` s. A request arriving at the start of a stall waits up to 2.0 s; the p99 of 900 ms falls inside that window. The discriminating check is the loop-lag histogram: a blocking call produces a lag distribution with a spike at the block duration and near-zero elsewhere, whereas pure queueing produces a smooth lag distribution. The spike is present.

**Step 4 — the spend.** Spend is a product, so it can be checked independently of latency.

```text
per call     = 22,000 * 3.00 / 1e6 + 800 * 15.00 / 1e6
             = 0.066 + 0.012
             = 0.078 currency units
baseline     = 40.0 * 0.078 * 3600 = 11,232.00 per hour
after retry  = 49.6 * 0.078 * 3600 = 13,927.68 per hour
increase     = 13,927.68 / 11,232.00 - 1 = 0.2400  (+24.0 %)
```

The predicted 24.0 % matches the reported 24 % exactly, which confirms that the retries are re-sending the full prompt rather than a cached prefix: a cached retry would have cost about a third as much and the increase would have been near 8 % rather than 24 %.

**Step 5 — independent re-check of the queue arithmetic.** Little's law gives a second route to the same in-system population, and the decomposition into busy plus queued is a third.

```text
L from Little        = 49.6 * 0.289        = 14.33
L from busy + queued = 49.6 * 0.180 + 49.6 * 0.109
                     = 8.928 + 5.406       = 14.33
```

The two agree exactly because both are the same identity rearranged; the value of computing both is that a transcription error in `W` or `W_q` shows up as a mismatch. If the two disagree, the error is in the arithmetic, not in the model, and no operational decision should be taken until they agree.

**Step 6 — correction, with the load recomputed.** Three changes, each independently justified: cap attempts at 2 (`A = 1 + 0.2 = 1.20`), add full jitter to the retry delay, and move the parse to a thread (`asyncio.to_thread` with a bounded executor).

```text
after cap 2 : offered = 40 * 1.20 = 48.0 attempts/s
              rho     = 48.0 / 55.56 = 0.8640
              W_q     ~ 0.070 s (Erlang C at c = 10, a = 8.64)
              W       ~ 0.250 s
```

Then the prompt-prefix cache: 19,800 of the 22,000 input tokens are stable, priced at 0.30 per million.

```text
cached input = 19,800 * 0.30 / 1e6 + 2,200 * 3.00 / 1e6
             = 0.00594 + 0.0066
             = 0.01254
per call     = 0.01254 + 0.012 = 0.02454
saving       = 1 - 0.02454 / 0.078 = 68.5 %
spend at 40/s = 40 * 0.02454 * 3600 = 3,533.76 per hour
spend at 48/s = 48 * 0.02454 * 3600 = 4,240.51 per hour
```

**Sensitivity note.** The design is close to a wall, and the distance to it is the number worth publishing. With `r = 0.20` and cap 2 the offered load is 48.0 against a ceiling of 55.56, a margin of 13.6 %. If `r` rises to 0.35 — entirely plausible during a downstream degradation — the offered load becomes `40 * 1.35 = 54.0`, `rho = 0.972`, and the Erlang C wait at `rho = 0.972` is several times the service time. The system therefore has to be protected by a retry budget proportional to successes, not by the cap alone: at a 10 % budget the offered load cannot exceed 44.0 whatever `r` does, and `rho` stays at or below 0.792.

**What the example demonstrates.** The latency number and the spend number were explained by two mechanisms that the same probe set separated: attempts-per-request proved retries, the lag histogram proved blocking, and the spend arithmetic proved the retries were uncached. No single metric would have done it, and neither of the two mechanisms alone accounts for both symptoms.

Takeaway: a defensible triage ends with three independent numbers agreeing — an attempts ratio, a queue decomposition, and a cost product — and with a stated margin to the ceiling.

## 6. Visual / Geometric Intuition

**The knee.** Wait time against arrival rate, for `c = 10` permits and `S = 0.180` s, computed with Erlang C at each point. The curve is flat for the first two-thirds of the range and then vertical; the vertical part is not a gradual degradation but a cliff.

```text
W_q (ms)
1735 |                                                          *
     |                                                         /
 219 |                                            *
 109 |                                *
  34 |                       *
  16 |                 *
   8 |            *
   2 |      *
     +----+----+----+----+----+----+----+----+----+----+----+----
        30   36   40   44   49.6 52   55          lambda (attempts/s)
             rho .54 .65 .72 .79 .893 .936 .99
```

The operational reading: at 44 attempts/s the wait is 34 ms and nobody complains; at 49.6 it is 109 ms; at 55 it is 1,735 ms. The change from "fine" to "outage" spans 20 % of the arrival range. This is why a retry storm that adds 24 % of load is not a mild degradation.

**Amplification against failure rate.** `A(0.2, 3) = 1.24` is the catalog's running example, but the shape matters more than the point: for small `r` the overhead is nearly linear, and the cap is what stops it.

```text
A
1.60 |                                        *  r = 0.40, cap 3
1.50 |
1.40 |
1.30 |                    *  r = 0.30
1.24 |           *  r = 0.20, cap 3
1.05 |  *  r = 0.05
     +----+----+----+----+----+----+----+
      0.05 0.10 0.15 0.20 0.25 0.30 0.40
```

Two curves, not one: the geometric sum with a cap of 3, and the asymptote `1/(1-r)` without a cap. They differ by only 0.01 at `r = 0.2` and by 0.107 at `r = 0.4`. So "we cap retries" buys little at low failure rates and a great deal at high ones — the opposite of the intuition that a cap is a small safety measure.

**Memory against admitted concurrency.** Unbounded task creation turns a rate into retained bytes. At 19 KiB of frame, buffer and captured locals per task, the line crosses a 1 GB container at about 52,600 tasks.

```text
RSS (MB)
1200 |                                              /
1000 |                                        *    972.8 MB at 50,000 tasks
 750 |                              /
 500 |                    /
 250 |        /
   0 +--+----+----+----+----+----+----+----+----+----
      5k  10k  20k  30k  40k  50k  60k    admitted tasks
```

The line is straight because per-task state is roughly constant. A straight line is the signature of bounded per-item state; a curve that keeps rising after the backlog drains is the signature of a leak. Those two are separated by one measurement: does RSS fall when admissions stop?

**Fanout as a multiplicative shadow.** Revenue before and after a join on a non-unique key.

```text
rows        parents only        after join (f = 1.30)
4,800,000   ####################
6,240,000   ##########################  (+1,440,000 rows, +30.0 %)
revenue     4,800,000 units -> 6,240,000 units
            30 % of the reported number is duplication, not growth
```

The visual point is that the inflation is uniform: every parent contributes 1.30 rows on average, so the chart looks like growth everywhere at once rather than a spike in one segment. Uniform inflation across all segments is a data-shape signature, not a business event.

**Censoring as a floor.** A distribution of per-token surprisal where one token is clamped at `ln(1e-5) = -11.5129`:

```text
surprisal
  0.00 |  ##############################  13 tokens at 0.12 nats
 -0.12 |-------------------------------------- true mean
 -0.93 |  X                                   reported mean after 1 clamp
-11.51 |                                      X  clamped token (1 of 14)
```

The reported mean sits at 7.8 times the true mean because a single point mass at the floor pulls it down. The shape also explains why the artefact is length-dependent: the same single clamped token in a 100-token sequence moves the mean by only 7 %, which is why the defect appears when sequences get short.

Takeaway: queue waits are a cliff, retry overhead is nearly linear in the failure rate, memory under unbounded admission is a ramp, fanout is a uniform shadow, and censoring is a floor that drags a mean — each shape names its own probe.

## 7. Comparison / Selection Table

Choosing a probe is a selection problem with disqualifiers, not a preference. The rows below are the probes this catalog relies on, ordered by the time they take.

| Probe | Answers | Resolves | Cost | Disqualified when |
|---|---|---|---|---|
| attempts / logical requests | retries versus slowdown | CI-06, CI-10, CI-15 | under 2 min if the counter exists | retries are issued by a client you cannot count |
| loop-lag histogram | blocking call versus queue tail | CI-09, CI-06 | 5 min to enable, then continuous | the loop runs in a process pool by design |
| queue decomposition `L = busy + Lq` | queueing versus service time | CI-01, CI-05, CI-06 | 10 min with `S` and `c` known | permits are not the binding limit |
| attempts per idempotency key | duplicate effects versus provider fault | CI-07 | 3 min with a key-indexed table | keys are not persisted with the effect |
| retry timestamp modulo delay | jitter presence | CI-08 | 2 min on recorded attempts | attempt start times are not logged |
| substring assertion on the exact prompt | truncation versus wrong answer | CI-12, CI-13, CI-18 | 6 min, one request per template | the prompt is assembled client-side and not logged |
| prompt-template hash diff | template drift | CI-18 | 4 min if both paths hash | hashes are not stored with scores |
| token count versus `len(text)/4` | budget mis-sizing | CI-23, CI-20 | 1 min with a tokenizer | no tokenizer matching the model is available |
| bytes per token on both sides | token versus byte normalisation | CI-17 | 5 min from stored counts | tokenizer identity was not recorded |
| clamp-count per sequence | censored loss | CI-19 | 4 min from the dump | the dump stores only the mean |
| `COUNT(*)` versus `COUNT(DISTINCT key)` | fanout versus growth | CI-30 | 2 min per query | the distinct key is not defined |
| NULL share on the join key | inner-join drop | CI-31 | 1 min per column | the column is declared NOT NULL but was widened |
| `ROWS` versus `RANGE` on the same window | frame semantics at ties | CI-32 | 3 min per query | the ordering key is unique by construction |
| rows at exactly the boundary timestamp | interval inclusion | CI-33 | 1 min per boundary | the timestamp has sub-second precision and no exact hits |
| pooled samples versus averaged percentiles | percentile aggregation | CI-34 | 4 min with shard samples | only per-shard aggregates were retained |
| signal receipt inside the workload | PID 1 forwarding | CI-24 | 5 min, one stop with work in flight | the workload is a single binary already |
| drain p99 versus grace period | deploy-time request loss | CI-25 | 3 min from the shutdown log | the workload is stateless and rejects instantly |
| pull bytes versus needed bytes | cold-start attribution | CI-26, CI-27 | 10 min with layer sizes | the node cache state is unknown |

**Selection rule, stated as an algorithm.** Start at the top of the table and stop at the first row whose disqualifier is false *and* whose outcome differs between the surviving hypotheses. That is the whole method; the table is ordered by cost because cost is the only variable the responder controls.

**Three choices the table settles.**
- To separate retry load from a slow downstream, count attempts per logical request before touching latency at all. Latency percentiles cannot separate the two, because both raise p99.
- To separate truncation from a wrong answer, assert on the presence of a sentinel string that appears only in the complete answer. Comparing the answer to an expected value mixes two questions.
- To separate fanout from growth, compute the row ratio and the distinct-key ratio over the same window. If the ratio is 1.30 and the distinct count is flat, no amount of business explanation will remove the duplication.

**What the table deliberately omits.** Profilers, distributed traces and log search all answer *where time went* but not *which mechanism*, and they cost minutes rather than seconds. They belong after a row of this table has narrowed the hypotheses to one, at which point a profiler confirms a mechanism already named.

**Comparison of containment options.** Containment is a separate choice with its own trade-offs, and the two axes are reversibility and evidence preservation.

| Containment | Stops harm by | Reversible | Preserves evidence | Cost of being wrong |
|---|---|---|---|---|
| cap intake at a fixed rate | shedding excess load | yes, one config change | yes, shed count is a counter | users see errors under load |
| disable the retry path | removing amplification | yes, feature flag | yes, attempts collapse visibly | failures surface as single errors |
| serve a prior computed result | decoupling from the downstream | yes, TTL expires | no, hides the current failure from users | staleness becomes invisible |
| raise permits | adding capacity | yes, one config change | yes, counters continue | multiplies the wrong effect |
| extend timeouts | letting work finish | yes | yes, but the window blurs | converts a slow failure into a longer one |
| drain and restart | clearing bad in-process state | yes, one restart | no, in-flight records lost unless flushed | recurrence is likely |

The rule that follows: prefer the containment that leaves a counter behind. "Serve a prior result" is the only row that destroys the signal, so it belongs to customer-facing paths with a hard availability requirement and never to the path under investigation.

Takeaway: probe by cost, stop at the first decisive row, and contain with something that leaves a counter — a containment you cannot see afterwards has removed the evidence for its own correction.

## 8. Failure Modes & Edge Cases

The catalog is below in scannable form, 36 issues with seven fields each. The prose entries that follow carry the mechanisms that need numbers, because a table cell cannot hold a derivation.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-01 | CONCEPT_CONFUSION | 40 tasks take 0.90 s where "40-way parallel" predicted 0.18 s | slow downstream; CPU saturation | compute `ceil(N/k)*L` with the real `k` | publish `N`, `k`, `L` beside every speedup claim | state the permit count in the API docstring | replay 40 tasks, assert 5 waves and 0.90 s |
| CI-02 | SCALE_FAILURE | RSS climbs with backlog, container killed | library leak; large payloads | peak RSS against `N * 19 KiB` | bound admissions with a semaphore or chunking | cap admitted tasks per unit time | replay 50,000 tasks, assert RSS below 1 GB |
| CI-03 | VIOLATED_ASSUMPTION | `await` returns but the work continues and the side effect lands | slow code; swallowed exception | live-task census before and after the timeout | cancel the inner task and await it in `finally` | cancellation owned by the caller of `create_task` | cancel mid-flight, assert zero live tasks |
| CI-04 | SCALE_FAILURE | intermittent "Task was destroyed but it is pending" | real timeout; loop shutdown | live-task set against retained references | retain a strong reference until completion | register tasks in a supervised set | shutdown test with 1,000 tasks, assert none dropped |
| CI-05 | NUMERICAL_EDGE | worst-case elapsed 7.5 s against a 5 s caller deadline | slow server; retry delay | recompute `n*t + sum(backoff)` | derive the budget from the worst case | assert budget < caller deadline in CI | replay a forced-timeout request, assert elapsed |
| CI-06 | SCALE_FAILURE | downstream load 24 % above request rate | traffic growth; bigger queries | attempts per logical request | retry budget as a fraction of successes | cap attempts and add full jitter | replay 1,000 requests at `r=0.2`, assert 1.24 |
| CI-07 | VIOLATED_ASSUMPTION | duplicate effects after a timeout | provider fault; double submit | effects per idempotency key | persist the key with the effect, return the earlier result | key created before the first attempt | replay the same key twice, assert one effect |
| CI-08 | NUMERICAL_EDGE | retries arrive as one synchronised wave | genuine outage; capacity drop | retry timestamps modulo the delay | add full jitter, honour `Retry-After` | assert jitter is non-zero in a config test | replay 500 rejections, assert spread over the window |
| CI-09 | VIOLATED_ASSUMPTION | p99 has a spike at a fixed value, mean barely moves | queueing tail; GC pause | loop-lag histogram against block duration | move blocking work to a bounded executor | ban synchronous I/O in coroutine bodies | replay 8 blocking calls, assert lag below 5 ms |
| CI-10 | VIOLATED_ASSUMPTION | 429s keep arriving despite backoff | genuine quota exhaustion; retry storm | request rate against the declared quota | honour `Retry-After` as the sole retry clock | parse and log the header value | replay a 429 with `Retry-After: 2`, assert 2 s gap |
| CI-11 | NUMERICAL_EDGE | retries fire 1,000x earlier than intended | server slowness; clock skew | compare wait to the header value's unit | convert units explicitly at the boundary | type the delay as milliseconds everywhere | replay `Retry-After: 2`, assert no attempt before 2,000 ms |
| CI-12 | VIOLATED_ASSUMPTION | streamed answers end at exactly 60 s with partial text | model stop condition; client cancel | end-time clustering at a constant | raise the idle timeout or send keepalives | pin every hop's idle timeout | replay a 90 s stream, assert completion |
| CI-13 | NUMERICAL_EDGE | request body silently shortened | client bug; wrong encoding | declared `Content-Length` against bytes sent | reject or chunk rather than truncate | validate declared against actual length | replay a 48 MB body, assert rejection |
| CI-14 | CONCEPT_CONFUSION | healthy peers dropped as dead while data flows | real packet loss; proxy timeout | last inbound frame versus last pong | treat any inbound frame as liveness | document the liveness definition once | replay a slow-but-alive peer, assert kept |
| CI-15 | SCALE_FAILURE | gateway memory grows with stalled consumers | memory leak; large frames | queue depth per connection | bounded queue, drop-oldest or close | per-connection cap and depth metric | replay a stalled consumer, assert close |
| CI-16 | VIOLATED_ASSUMPTION | dashboard shows success while the payload reports failure | client bug; stale dashboard | parse the body, not only the status | fail on the documented error field | contract test per endpoint | replay an error body with status 200, assert failure |
| CI-17 | CONCEPT_CONFUSION | candidate model "12 % better" per token, equal per byte | real quality gain; prompt drift | convert both to bits per byte | report bpb for cross-tokenizer claims | ban per-token comparisons across tokenizers | recompute from raw surprisals, assert 0.5 bpb both |
| CI-18 | VIOLATED_ASSUMPTION | eval and serving scores differ on identical text | model nondeterminism; sampling | dump the templated prefix ids from both paths | pin template hash and tokenizer version | manifest hash in every score record | replay one item on both paths, diff ids |
| CI-19 | NUMERICAL_EDGE | mean loss lower than any single token's loss | arithmetic bug; weighting | recompute from the dump and count clamps | store total surprisal and `N` beside the mean | assert `min(s) <= mean <= max(s)` | replay 14 tokens with one clamp, assert 7.78x |
| CI-20 | SCALE_FAILURE | context fits but latency and spend triple | slower hardware; longer answers | tokens per component, plotted by `k` | size retrieval by `k`, not by relevance score | per-component token budget in the builder | replay `k=4` versus `k=16` |
| CI-21 | NUMERICAL_EDGE | 12 % of outputs hit `max_new_tokens` exactly | verbose prompts; model change | share of outputs at the cap | raise the cap or add a stop sequence | alert on cap-hit share | replay a capped request, assert stop reason |
| CI-22 | CONCEPT_CONFUSION | per-request cost falls while total spend rises | price change; more traffic | cost per request against cost per thousand tokens | price the unit you bill | one cost model per surface | replay cached and uncached calls, assert 68.5 % |
| CI-23 | NUMERICAL_EDGE | budget sized on `len(text)/4` overruns by 30 % | longer answers; bigger documents | tokenizer count against the estimate | estimate with the model's own tokenizer | assert estimate error below 3 % | replay 22,000-token prompt, assert no overflow |
| CI-24 | VIOLATED_ASSUMPTION | workload killed with work in flight at stop | slow handler; queue loss | process tree at stop, signal receipt | exec the workload as PID 1 | assert a single-process tree in an image test | stop with 30 s of work, assert drain |
| CI-25 | NUMERICAL_EDGE | a share of requests lost on every deploy | client timeout; pod crash | drain p99 against the grace period | set grace above drain p99 with margin | derive grace from the drain histogram in CI | rolling restart, assert zero dropped |
| CI-26 | CONCEPT_CONFUSION | image halved, cold start barely moved | slow runtime; CPU limit | pull time against init time | attack the larger term | report both terms per release | rebuild and compare, assert attribution |
| CI-27 | SCALE_FAILURE | every deploy re-runs the full dependency install | registry slowness; bigger deps | rebuild time with the cache warm | copy lockfile and install before source | assert cache hit in CI | rebuild from cache, assert under 20 s |
| CI-28 | NUMERICAL_EDGE | throttled at 40 % CPU usage | slow code; memory pressure | `nr_throttled` over `nr_periods` | raise the quota above observed demand | alert on throttle ratio, not usage | replay a burst, assert throttle ratio |
| CI-29 | VIOLATED_ASSUMPTION | works locally, differs in the cluster | code difference; missing env var | image digest local versus deployed | deploy the tested digest, not the tag | promote by digest only | compare digests, assert equal |
| CI-30 | VIOLATED_ASSUMPTION | revenue 30 % high after a join change | real growth; price change | `COUNT(*)` against `COUNT(DISTINCT parent)` | deduplicate the right side before aggregating | assert rows per parent in a test | replay the query, assert the 1.30 ratio |
| CI-31 | NUMERICAL_EDGE | 4.2 % of rows vanish from an inner join | genuinely missing keys; filters | NULL share on both sides of the key | left join or coalesce the key | profile NULL share on join keys | replay the query, assert the 5,040 rows return |
| CI-32 | NUMERICAL_EDGE | running total jumps by a whole peer group at a tie | duplicate events; bad ordering | `ROWS` versus `RANGE` output on the same window | state the frame explicitly | forbid implicit frames in review | replay a tied key, assert both outputs named |
| CI-33 | NUMERICAL_EDGE | daily total double-counts the boundary row | late events; duplicate ids | rows at exactly the boundary timestamp | use half-open `>=` and `<` | lint for `BETWEEN` on timestamps | replay a midnight event, assert one count |
| CI-34 | CONCEPT_CONFUSION | dashboard p99 higher than any shard's p99 | slow shard; bad timer | recompute the percentile on pooled rows | pool samples, then take the percentile | ban averaging percentiles | replay three shards, assert 1,980th of 2,000 |
| CI-35 | NUMERICAL_EDGE | a ratio reports 3 instead of 3.5 | rounding; wrong numerator | recompute in floating point | cast before dividing | unit test the ratio expression | replay 7/2, assert 3.5 |
| CI-36 | CONCEPT_CONFUSION | monthly rate differs from the 30-day rate by 4.35 % | seasonality; data gap | denominator days per period | normalise to a fixed unit | document the denominator per metric | replay both windows, assert 24/23 |

### 8.1 The issues that need arithmetic

**CI-01 — Concurrency mistaken for parallelism.** Mechanism: the semaphore admits eight items and the other 32 wait, so elapsed time is `ceil(40/8) * 0.180 = 5 * 0.180 = 0.90` s while a reader who assumed 40-way parallelism predicts 0.18 s. The "slow downstream" hypothesis fails a specific test: a slower downstream raises `L_unit` for every item including the first, so the first wave would also take longer than 180 ms, and it does not. Correction: publish `N`, `k` and `L_unit` beside any speedup claim. Verification: replay 40 items and assert five distinct completion clusters.

**CI-02 — Unbounded gather.** Mechanism: each admitted coroutine retains a frame, captured locals and a response buffer, about 19 KiB in this constructed case, so 50,000 simultaneous tasks retain `50,000 * 19 KiB = 972.8 MB` before any payload is counted. The leak hypothesis fails one test: RSS that tracks backlog size and falls after the backlog drains is bounded per-item state; a leak's RSS stays high after the drain. Correction: size the semaphore from the memory budget — at 19 KiB per task and a 1 GB container, at most 52,600 concurrent tasks, and a working limit near 16,000 leaves 70 % of headroom.

**CI-03 — Cancellation that only stops the waiter.** Mechanism: `asyncio.wait_for` cancels the awaiting coroutine, not a task started with `create_task` inside it, so the timeout returns while the work continues and the side effect still lands. The elapsed time of the caller is already correct, which is why the "slow code" hypothesis fails: only the side-effect count is wrong. Check: a task census immediately after the timeout minus the census before it, and the delta is the number that matters. Correction: cancel the inner task and await it inside `finally`, so the caller cannot return before the work has stopped.

**CI-05 — Timeout budget derived from one attempt.** Mechanism: three attempts at 2 s plus 1.5 s of backoff sum to `3 * 2 + 1.5 = 7.5` s of worst-case elapsed time, against a caller deadline of 5 s. The caller gives up at 5 s while the callee is still inside attempt three, so the retry that was meant to improve reliability produces a guaranteed timeout. Correction: derive the budget from the sum, and assert in CI that total budget is below the caller's deadline. Note the boundary: with jitter, the backoff sum is a distribution, and the worst case is the sum of the maxima, not the maximum of the sum.

**CI-06 — Retry amplification as load.** Mechanism: `A(0.2, 3) = 1.24`, so 40 logical requests/s become 49.6 attempts/s. Against `c = 10` permits at 0.180 s the ceiling is 55.56 attempts/s, so `rho` moves from 0.720 to 0.893 and the Erlang C wait from 16 ms to 109 ms. The "traffic growth" hypothesis fails because the logical request counter is flat while the attempt counter rises; the two counters are separate series precisely so this comparison is possible.

**CI-07 — Idempotency key created inside the retry loop.** Mechanism: with `r = 0.2` and three attempts, 100 logical operations generate 124 attempts, and a key minted per attempt makes each attempt a distinct operation, so up to 24 duplicate effects are admissible. The "provider fault" hypothesis fails because every duplicate carries a key the client itself generated and can list. Correction: create the key before the first attempt and persist it in the same transaction as the effect. Verification: replay the same key twice and assert exactly one effect and one `DUPLICATE_SUPPRESSED` reason code.

**CI-09 — Blocking call inside the loop.** Mechanism: `time.sleep`, `requests.get` or a synchronous parse blocks the whole event loop, so eight coroutines that each block for 0.25 s stall every other coroutine for `8 * 0.25 = 2.0` s per pass. The signature is a lag histogram with a spike at the block duration and near-zero elsewhere; pure queueing produces a smooth lag distribution instead. Correction: `asyncio.to_thread` with an explicitly sized executor, so the thread count is a stated capacity rather than an implicit one.

**CI-11 — `Retry-After` read in the wrong unit.** Mechanism: the header carries seconds as an integer, and a client that treats it as milliseconds retries after 2 ms instead of 2,000 ms — a 1,000x early retry, which restarts the storm the header exists to prevent. Competing cause: clock skew, which produces a *late* retry, never an early one, so the direction of the error discriminates without any extra telemetry. Correction: parse into a named duration type at the boundary, and log the interpreted value next to the raw header.

**CI-15 — Slow consumer with an unbounded per-connection queue.** Mechanism: 200 connections at 200 messages/s of 8 KiB with a 30 s write backlog hold `200 * 200 * 8192 * 30 = 9.83 GB` of pending bytes, which the gateway discovers as an out-of-memory kill rather than as a slow consumer. The "memory leak" hypothesis fails because the bytes are attributable to named connections and fall when those connections close. Correction: a bounded queue per connection with a stated policy — drop-oldest for lossy feeds, close for stateful ones — and a depth metric per connection.

**CI-19 — Censored mean loss.** Mechanism: 13 of 14 tokens at `-0.12` nats and one clamped at `ln(1e-5) = -11.5129` give `(13 * -0.12 - 11.5129)/14 = -0.9338`, which is 7.78 times the true `-0.12`. Two clamped tokens give `-1.7476`, 14.6 times. The bound worth asserting is `min(s) <= mean <= max(s)`, because a mean outside the range of its own terms is arithmetic proof of a substitution, and no model change can produce it.

**CI-21 — Outputs truncated at the token cap.** Mechanism: 12 % of outputs end exactly at `max_new_tokens`, so the cap, not the model, decides where those answers stop. The "verbose prompt" hypothesis is testable: a prompt effect raises length across the distribution, while a cap produces a point mass at exactly one value. Correction: raise the cap only after checking the p99 of the natural stop-length distribution, and add a stop sequence where one exists; otherwise a raised cap converts truncation into spend.

**CI-25 — Grace period shorter than the drain.** Mechanism: the grace period is 10 s and the drain p99 is 14 s, so the stop signal kills one in a hundred requests at the tail — every deploy, invisibly, because the client sees a connection reset rather than an error the service records. Check: compare the grace constant to the drain p99 from the shutdown log; the two numbers live in different files, which is why the defect survives review. Correction: grace above drain p99 with margin, derived in CI from the histogram rather than chosen.

**CI-26 — Image size confused with cold-start time.** Mechanism: at 1,200 MB and 180 MB/s the pull is 6.67 s and initialisation is 9 s, so `T_cold = 15.67` s against a 1.2 s warm start — 13.1x. Halving the image removes 3.3 s; halving imports removes 4.5 s. The "slow runtime" hypothesis is answered by the attribution itself: report both terms per release, because a team optimising whichever term it can see will otherwise report a 21 % win as a breakthrough.

**CI-30 — Fanout in a join.** Mechanism: 4,800,000 parent rows joined to 6,240,000 child rows give `f = 1.30`, so a revenue sum built after the join reports 6,240,000 where the parent-only sum is 4,800,000 — a 30.0 % inflation that looks like growth in every segment simultaneously. Discriminating check: `COUNT(*)` against `COUNT(DISTINCT parent_key)` over the same window. If the row count rises 30 % while the distinct count is flat, the join is the mechanism and no business explanation survives.

**CI-31 — NULL keys dropped by an inner join.** Mechanism: 4.2 % of 120,000 rows carry a NULL in the join key, so an inner join silently drops `120,000 * 0.042 = 5,040` rows. On the inflated 6.24 M-row result of CI-30 the same share is 262,080 revenue units — and the two defects compose, which is why they are tested together. Check: NULL share on both sides of the key before the join, not after, because after the join the evidence has been discarded.

**CI-32 — `RANGE` versus `ROWS` at a tie.** Mechanism: the default frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes every peer of the current row, so a running total over four tied rows advances by the whole peer group at once rather than row by row. Both answers are defensible; what is not defensible is not knowing which one the query produced. Correction: state the frame explicitly in the query and in the review checklist, then verify by running both frames on a tied key and asserting the difference equals the peer-group sum.

**CI-34 — Averaging percentiles.** Mechanism: three shards report p99 of 180, 940 and 210 ms; the arithmetic mean is 443.3 ms, which describes no request in any shard. Pooling 2,000 samples and taking rank `ceil(0.99 * 2000) = 1,980` gives the actual tail request. The defect is a `CONCEPT_CONFUSION` because a percentile is a rank and an average of ranks is not a rank. Correction: retain samples or a mergeable sketch per shard; if only aggregates remain, publish a maximum-of-p99 as a lower bound and label it as one.

**CI-04 — Task garbage-collected while pending.** Mechanism: `asyncio` holds only a weak reference to a task, so a task created and not retained can be collected mid-flight and its result never retrieved; the symptom is CPython's "Task exception was never retrieved" or "Task was destroyed but it is pending" rather than a wrong answer. The "real timeout" hypothesis fails because a timeout cancels deterministically and logs a cancellation, whereas this arrives with no cancellation record at all. Correction: keep every task in a supervised set until it completes, and remove it in a `finally`. Verification: shutdown with 1,000 tasks and assert the set is empty and no warnings were emitted.

**CI-08 — Synchronised retry wave.** Mechanism: 500 clients that failed together all sleep 1 s and retry at the same instant, so the downstream sees a spike of 500 attempts where it could absorb the same load spread over the window. The peak is `500` rather than the sustainable rate, and the second failure produces a second identical spike. The "genuine outage" hypothesis is excluded by the shape: an outage raises load continuously, while synchronised retries produce a periodic pattern whose period equals the delay. Correction: full jitter, `delay = uniform(0, base)`, so the 500 attempts spread across 1,000 ms.

**CI-10 — Exponential backoff while ignoring `Retry-After`.** Mechanism: the server states when to return (2 s) and the client applies its own schedule (0.5 s doubling). The client therefore retries 4x too early at the first step and stays early until its schedule exceeds 2 s, which extends the throttling window it was trying to escape. The "genuine quota exhaustion" hypothesis is distinguishable by the request rate itself: honouring the header gives `20 + 20/2 = 30` attempts/s, ignoring it gives `40 + 20/0.5 = 80` — the higher rate is self-inflicted and visible in the attempt counter.

**CI-12 — Stream cut at the proxy idle timeout.** Mechanism: each hop has its own idle timeout and the shortest wins; a 60 s proxy against a 90 s generation ends the response at 60 s with a complete-looking prefix. The "model stop condition" hypothesis fails a clustering test: a model's stop reasons distribute over time, while a proxy cut produces a point mass at exactly the configured value. Correction: send keepalive tokens or comments during generation so the connection is never idle for the full period, and pin every hop's idle timeout in one configuration file with the shortest value documented.

**CI-13 — Body truncated at a transport limit.** Mechanism: a body of `48 * 1024 * 1024 = 50,331,648` bytes exceeds a configured limit and is silently cut rather than rejected, so the receiver parses valid JSON that stops early and reports a field as missing rather than the request as invalid. Competing cause: a client serialisation bug, which produces a body whose declared length matches its content — the discriminator is declared `Content-Length` against bytes actually received, and the two differ only in the truncation case.

**CI-14 — Pong liveness killing a healthy peer.** Mechanism: the liveness rule is "a pong within one interval", the interval is 30 s, and the peer's pong is delayed by a busy thread for 90 s while it streams data continuously. The rule drops a peer that is demonstrably alive. The "real packet loss" hypothesis is excluded by the inbound frame counter, which is still advancing at the moment of the drop. Correction: any inbound frame refreshes liveness, with an application heartbeat reserved for protocols where silence truly means dead.

**CI-16 — Status code trusted over the body.** Mechanism: the endpoint returns 200 with an error object in the body — a documented behaviour — and the client checks only the status, so a failed operation is recorded as a success. The "client bug" hypothesis is excluded by the response bytes, which are identical to a successful call's bytes apart from the error field. Correction: parse the documented field and fail on it; keep a contract test per endpoint that feeds a known error body and asserts the client raises.

**CI-18 — Template drift between scoring paths.** Mechanism: the evaluation script prepends three sentences and the serving path prepends two, so identical text produces different token ids and scores differ by more than sampling noise. The "model nondeterminism" hypothesis fails because the difference persists at temperature 0 and reproduces byte-for-byte on the same path. Correction: hash the rendered prefix, store the hash with every score, and fail the comparison when the hashes differ rather than attributing the gap to the model.

**CI-20 — Retrieval depth as the cost driver.** Mechanism: retrieval returns `k` documents of about 900 tokens each, so `k = 4` sends 3,600 input tokens and `k = 16` sends 14,400 — a 4x increase in prefill work for the same question. At 3.00 per million tokens and 800 output tokens, per-call cost moves from `0.0108 + 0.012 = 0.0228` to `0.0432 + 0.012 = 0.0552`, a 2.42x rise that is not a 4x rise because the output term is fixed. Correction: a per-component token budget in the request builder, with `k` chosen by marginal answer quality rather than by retrieval score.

**CI-22 — Unit confusion in unit economics.** Mechanism: cost per request falls because a cache now absorbs most input, while total spend rises because traffic doubled; both statements are true and the dashboard shows only the first. The discriminator is to price the unit being billed: cost per thousand tokens is what the invoice computes, and cost per request is a derived ratio whose denominator can move independently. Correction: one cost model per surface, with cached and uncached tokens priced separately, and both totals published.

**CI-23 — Token estimate from character count.** Mechanism: `len(text)/4` is a heuristic that underestimates code, tables and non-Latin scripts; an 88,000-character prompt estimated at 22,000 tokens is 28,600 tokens under a real tokenizer, a 23.1 % undercount that turns a budget into an overflow. The "longer answers" hypothesis is excluded because output tokens are counted exactly at generation and are not subject to the estimate at all. Correction: estimate with the model's own tokenizer and assert the estimate error stays below 3 % in CI.

**CI-27 — Dependency install re-run on every build.** Mechanism: source is copied before the lockfile, so any source change invalidates the dependency layer and the install repeats; at 240 s of install per build and 18 builds a day that is 72 minutes daily. Competing cause: a larger dependency set, which would show a rising install time at constant build count rather than a constant install time at rising build count. Correction: copy the lockfile, install, then copy source, so the invalidation point sits below the install layer.

**CI-28 — CPU quota mistaken for CPU usage.** Mechanism: the quota is 0.4 cores against a 0.5-core demand, so `20 %` of the demand cannot be served and the container is throttled whenever it bursts, while average usage reads 40 % — the quota itself. The "slow code" hypothesis is excluded by `nr_throttled/nr_periods`, which rises whenever demand exceeds the quota regardless of mean usage. Correction: raise the quota above observed burst demand, and alert on the throttle ratio rather than on usage.

**CI-29 — Tag deploys the wrong artifact.** Mechanism: an image is built, tested, and then a mutable tag is re-resolved at deploy time, so the deployed digest differs from the tested one; the difference is usually invisible until behaviour differs. The "missing environment variable" hypothesis is excluded by comparing digests, which is a string equality rather than an investigation. Correction: promote by digest, and print the digest in the deploy log next to the test log's digest.

**CI-33 — Interval inclusion at the day boundary.** Mechanism: `WHERE ts BETWEEN '2026-11-09' AND '2026-11-10'` includes midnight of the 10th, so the first event of the next day is counted twice across two runs; with one event per 5,000 rows at the boundary, the error is 0.02 % — small enough to pass review and large enough to break a reconciliation. Correction: half-open intervals, `ts >= start AND ts < start + 1 day`, and a lint rule banning `BETWEEN` on timestamp columns.

**CI-35 — Integer division in a ratio.** Mechanism: `7/2` in integer arithmetic is 3, and a percentage computed as `count/total*100` with both operands integral returns a coarse number that looks like a rounding decision rather than a truncation. The "rounding" hypothesis is distinguishable by recomputing in floating point: rounding moves the value by at most 0.5, truncation by up to 1.0 and always downward. Correction: cast before dividing, and unit-test the expression with a known non-integral result.

**CI-36 — Denominator switched between periods.** Mechanism: a monthly rate computed over 31 days and a 30-day rate computed over 30 days differ by `31/30 - 1 = 3.33 %`, and the same pair over a 24-hour day against a DST-shortened 23-hour day differs by `24/23 - 1 = 4.35 %`; both are large enough to be reported as a trend. The "seasonality" hypothesis is excluded because a denominator artefact is exactly reproducible from the calendar while seasonality is not. Correction: normalise to a fixed unit and print the denominator beside every rate.

### 8.2 The triage tree

The tree encodes the order in which the probes of Section 7 are worth running. Every branch ends in an issue ID, so the tree is a routing table rather than advice.

```text
complaint
 |
 +-- "it is wrong" ---------> recompute the identity by hand
 |      |                        disagree with the system -> arithmetic (CI-19, CI-35)
 |      |                        agree, but the input was wrong -> contract (CI-18, CI-29)
 |      +-- number too large ---> rows per key > 1?  --yes--> fanout (CI-30)
 |                                NULL share on key? --yes--> dropped rows (CI-31)
 |                                boundary row?      --yes--> interval (CI-33)
 |
 +-- "it is slow" ----------> attempts/request > 1.05? --yes--> retries (CI-06, CI-10)
 |      |                     in_flight ~ pool size?   --yes--> saturation (CI-05)
 |      |                     p99 spike at a constant? --yes--> blocking (CI-09)
 |      |                     wait grows, service flat?--yes--> queue (CI-01)
 |      +-- only at a deploy --> grace < drain p99?  --yes--> signal (CI-24, CI-25)
 |
 +-- "it is stuck" ---------> live tasks > expected?  --yes--> cancellation (CI-03, CI-04)
 |                            queue depth rising?     --yes--> backpressure (CI-15)
 |                            stream ends at 60 s?    --yes--> idle timeout (CI-12)
 |
 +-- "it is doubled" -------> effects per key > 1?    --yes--> idempotency (CI-07)
 |                            attempts per key > 1?   --yes--> transport retry (CI-08)
 |
 +-- "it costs more" -------> input tokens up?        --yes--> context (CI-20, CI-23)
                             attempts up?             --yes--> retries (CI-06)
                             cached share down?       --yes--> prefix cache (CI-22)
                             instances up?            --yes--> cold start (CI-26, CI-27)
```

Two properties make the tree usable under pressure. First, no branch requires more than one counter, so the first probe can be run with what exists. Second, the branches are ordered by how often they terminate: "wrong" resolves fastest because arithmetic can be recomputed by hand, and "slow" resolves second because attempts-per-request usually exists.

### 8.3 Four failure mechanisms that recur across layers

**Mechanism A — an unbounded accumulator behind a bounded entry point.** The entry point enforces a rate, and a queue or collection behind it grows without limit. Instances: CI-02 (task list), CI-15 (per-connection write buffer), CI-20 (retrieved context). Discriminator: compare the accumulator's size to `rate * retention` and check whether size falls when the input stops. If it falls, the accumulator is bounded state under load; if it does not, the retention is a leak.

**Mechanism B — a constant that lives in two files.** Two components each hold a related constant and nobody compares them: the 10 s grace period against the 14 s drain p99 (CI-25), the 60 s proxy idle timeout against a 90 s generation (CI-12), the caller's 5 s deadline against a 7.5 s retry budget (CI-05). Discriminator: print both constants from configuration in one place. This is the cheapest check in the catalog and the one most often skipped, because each file looks correct in isolation.

**Mechanism C — a summary statistic computed before the grouping is complete.** The aggregate is right for its input and the input is the wrong population: per-shard p99 averaged across shards (CI-34), a mean computed over censored values (CI-19), a rate whose denominator is the wrong number of days (CI-36), a count taken after a fanout join (CI-30). Discriminator: recompute the statistic from the finest retained grain and compare; if the coarse number cannot be reproduced from the fine one, the grouping is wrong.

**Mechanism D — a boundary expressed in the wrong convention.** The code and the data disagree about whether an interval includes its endpoint: `BETWEEN` on a timestamp (CI-33), `<=` in a pagination cursor, `RANGE` treated as `ROWS` (CI-32), a byte count compared before encoding (CI-13). Discriminator: construct one record exactly on the boundary and one a nanosecond either side, then count. This check costs a minute and is the only one that distinguishes inclusion from exclusion.

Takeaway: the catalog is long because defects are local, but the four mechanisms behind it are general — unbounded accumulators, duplicated constants, premature aggregation, and boundary conventions — and a team that tests those four catches most of the 36.

## 9. Complexity & Cost Analysis

**Cost of a probe.** A probe's cost is the responder-minutes to run it plus the probability it fails to discriminate multiplied by the cost of falling back to a guess. The table below uses 6 minutes per probe and 45 minutes of rework for a wrong first containment, with the Bayes posterior of Section 4.3.

| Probe | Minutes | Detects | Misses | Expected cost of using it |
|---|---|---|---|---|
| attempts per request | 2 | retries present | retries absent | 2 + 0.14 * 45 = 8.3 |
| loop-lag histogram | 5 | blocking call | pure queueing | 5 + 0.14 * 45 = 11.3 |
| idempotency-key effect count | 3 | duplicate effects | single effect | 3 + 0.10 * 45 = 7.5 |
| prompt-hash diff | 4 | template drift | genuine model difference | 4 + 0.20 * 45 = 13.0 |
| pooled percentile recompute | 4 | aggregation defect | real shard skew | 4 + 0.12 * 45 = 9.4 |
| full trace review | 40 | nothing specific | most mechanisms | 40 + 0.50 * 45 = 62.5 |

The last row is the one that changes behaviour: a thorough trace review costs more than the rework it is meant to prevent unless the prior is very high. Probes win because they are cheap and specific, not because they are thorough.

**Cost of a false positive.** A probe that fires on the wrong mechanism costs more than no probe, because it also consumes the containment. Raising permits on a queueing defect costs idle capacity; raising them on an idempotency defect multiplies duplicate effects by the worker increase. At `c = 10` to `c = 16` with a duplicate-effect rate of 1 per 100 requests, the extra workers raise the duplicate rate proportionally — from 0.0625 per 1,000 to 0.1 per 1,000 — before the fix is in.

**Retry cost, in currency.** At 49.6 attempts/s and 0.078 per call, the retried portion costs `9.6 * 0.078 * 3600 = 2,695.68` per hour, against a base of `40 * 0.078 * 3600 = 11,232.00`. The retries are 19.4 % of spend for 0 % of additional successful answers, since the successes are the same 40 per second. With the prefix cache in place the same retries cost `9.6 * 0.02454 * 3600 = 848.10` per hour, so caching reduces the absolute cost of the defect by 68.5 % while leaving the defect in place — a reminder that cost reduction and defect repair are separate work items.

**Memory cost of admission.** At 19 KiB per task, the memory cost of concurrency `k` is `19 KiB * k`. A 1 GB container with a 70 % working budget holds 37,800 tasks; a 16,000-task limit costs 304 MB and is the largest round number that leaves room for payloads. The cost of choosing `k` too small is throughput, which is visible; the cost of choosing it too large is an out-of-memory kill, which is not attributable after the fact.

**SQL cost.** A hot key scanning 45 % of a 12 M-row table touches 5.4 M rows per query. At 40 queries/minute that is 216 M row-touches per minute against whatever the engine's scan rate is, and the cost is not the mean but the contention. Pre-aggregation trades storage for scan: a daily rollup keyed by that key reduces the scan to the number of days in the window, at the price of a rebuild path that must be verified against the base table.

**Cold-start cost.** A 15.67 s cold start on a 1.2 s warm baseline means any autoscaling event pays 13.1x the warm request time for the first requests on each new replica. With 20 replicas rolling, the cost is 20 * 15.67 = 313.4 s of degraded capacity per scale-out event; that number, not the image size, is what justifies the work of moving initialisation into a readiness gate.

**Cost of the catalog itself.** Every row costs a maintainer about 20 minutes a year to re-verify, because a formula's inputs drift. Thirty-six rows is 12 hours a year, which is justified only if the catalog is used: the test of that is whether the issue IDs appear in incident notes and in replay names. An unused catalog is documentation debt with a maintenance bill.

**Replay sample size, which is the cost nobody budgets.** A replay that proves a defect is gone must be large enough that the defect's expected count under the null is comfortably above zero. For a defect with pre-fix rate `p` per operation, the number of replayed operations needed to expect three occurrences is `n = 3/p`. At `p = 0.01` that is 300 operations — routine. At `p = 1/5000`, the boundary-double-count rate of CI-33, it is 15,000 operations, which is why boundary defects are verified by constructing the boundary record explicitly rather than by volume. The rule that follows: volume replays verify common defects, and constructed boundary records verify rare ones; a team that replays only at volume will close a boundary defect on evidence that could not have detected it.

Takeaway: probes are cheap because they are specific, false positives are expensive because they consume containment, and the cost that decides most designs is not the mean but the tail event that autoscaling, deploys and retries all amplify.

## 10. Key Theorems / Results Reference

Each result below is stated with its condition and the sheet that owns its derivation. The catalog cites these by ID; a responder who needs the derivation goes to the owning sheet rather than re-deriving under pressure.

**R-01 — Retry amplification (CI-F1).** For independent attempts with failure probability `r` and cap `n`, mean attempts per operation is `(1-r^n)/(1-r)`, and the uncapped limit is `1/(1-r)`. Condition: independence, enforced cap, retryable errors only. Owned by `01_Python_Async_and_Concurrency.md` as `R-01`.

**R-02 — Utilisation after amplification (CI-F2).** `rho = lambda_0 * A * S / c`. Condition: every attempt holds one permit for the whole of `S`. The consequence used throughout is that a 24 % retry overhead consumes 24 % of capacity, not 24 % of the margin.

**R-03 — Erlang C wait (CI-F3).** For `c` identical permits, Poisson arrivals and exponential service, mean queue wait is `P_wait * S / (c * (1 - rho))` with `P_wait` the standard Erlang C probability. Condition: `rho < 1`. Owned by `01_Python_Async_and_Concurrency.md` as `Q-01` and by `02_REST_APIs_and_WebSockets.md` as `Q-02`.

**R-04 — Little's law (L-01).** `L = lambda * W` for a system in steady state with a well-defined boundary. Condition: means exist; window long enough for end effects to vanish. It is a conservation identity, not a provisioning rule, and it does not bound anything.

**R-05 — Wave time under `k` permits (CI-F4).** `T = ceil(N/k) * L_unit + R`. Condition: one permit per item for its whole life; `k = min(semaphore, pool, quota, fds)`.

**R-06 — Probe posterior (CI-F5).** `PPV = sens*p / (sens*p + (1-spec)*(1-p))`. Condition: calibrated error rates and an honest prior. The operating consequence: a negative probe is usually the stronger signal at low prior, so probes are designed to be cheap and frequently negative.

**R-07 — Fanout inflation (CI-F6).** `S_bad = f_row * S_good`. Condition: unique parent side, non-unique child side, no post-join filter. Consequence: uniform inflation across segments is a data-shape signature.

**R-08 — Censoring bias (CI-F7).** A per-token floor at `f` with `m` clamped tokens shifts the mean to `(sum + m*ln f)/N`; the bias factor is that quotient over the true mean. Condition: the floor sits below the true values, so the substitution is a lower bound on surprisal.

**R-09 — Cold-start attribution (CI-F8).** `T_cold = B_needed/beta + m + t_init`. Condition: cold node, on-demand pulls. Consequence: attribute before optimising, because the two terms respond to different work.

**R-10 — Half-open interval.** A daily bucket `[d, d+1)` counts each event once; `BETWEEN` over timestamps counts boundary events twice, at a rate of one per boundary per bucket — 0.02 % of a 5,000-row hourly bucket, which is invisible in review and material in a 12-month sum.

**R-11 — Percentile pooling.** The p-th percentile of a pooled sample is the value at rank `ceil(p * n)` of the pooled order statistic. It is not the mean of per-shard percentiles, because the mean of ranks is not a rank. Condition: samples or a mergeable sketch retained per shard.

**R-12 — Three-valued logic.** `NULL` compares as unknown, so a predicate on a nullable column removes rows from inner joins and `WHERE` clauses without error. The check is a NULL share before the join; after the join the evidence is gone.

**R-13 — Amdahl-style ceiling for permit-bound work.** With serial fraction `s`, speedup from `k`-fold concurrency is bounded by `1/(s + (1-s)/k)`. At `s = 0.1` and `k = 8` the ceiling is `1/(0.1 + 0.1125) = 4.7`, not 8 — which is why "eight permits" never yields eight-fold speedup on real work.

**R-14 — Cache saving identity.** With stable prefix share `h` priced at rate `pc` and the rest at `p`, per-call cost is `h*cached_rate + (1-h)*p` for input plus output. At `h = 0.90`, `cached_rate = 0.30/M` and `p = 3.00/M`, input cost falls from 0.066 to 0.01254 — a 68.5 % total saving once output is included, not the 90 % that `h` suggests.

Takeaway: fourteen results cover the whole catalog, and each one is a comparison instrument with a named condition — cite the ID, substitute the numbers, and check the condition before believing the answer.

## 11. Interview / Exam Questions

Fifteen questions, half of them requiring computation or diagnosis. Each answer stands on its own; the follow-up is the question an interviewer actually asks next.

**Q1 — 40 tasks run against 8 permits at 180 ms each. What is the elapsed time, and what is the most common wrong answer?**
Answer: 0.90 s, computed as `ceil(40/8) * 0.180 = 5 * 0.180`. The common wrong answer is 0.18 s, from assuming 40-way parallelism.
Reasoning: the permit count is a capacity contract; the work is concurrent but not parallel, so it proceeds in five serial waves.
Follow-up: which single number would you publish to prevent the error recurring? The triple `N`, `k`, `L_unit`, because the claim is only checkable when all three are visible.

**Q2 — A service reports a 24 % rise in downstream load with flat request rates. Give the first probe and the two candidate mechanisms.**
Answer: count attempts per logical request. If it is 1.24, retry amplification with `r = 0.2` and a cap of 3 explains the load exactly; if it is 1.00, the extra load is coming from another client.
Reasoning: the logical counter and the attempt counter are separate series, so the ratio discriminates without any latency data.
Follow-up: what would make the ratio 1.24 but the mechanism something else? A client that retries outside the counted path — which is why the attempt counter must sit at the last hop before the downstream.

**Q3 — You have a 25 % prior that a defect is a retry storm. Your probe has sensitivity 0.90 and specificity 0.95. It comes back positive. What is the probability you are right, and what would a negative have told you?**
Answer: positive gives `0.225 / 0.2625 = 0.857`; negative would have given a 0.966 probability of absence.
Reasoning: at low prior, false positives from the 5 % specificity gap are 0.0375 against 0.225 of true positives, so three in twenty positives are wrong.
Follow-up: the probe costs 6 minutes and rework costs 45. Is it worth running? Expected cost with the probe is `6 + 0.143*45 = 12.43`; without it, a coin-flip guess costs 22.5, so yes.

**Q4 — A retry is configured with a 1 s fixed delay after 500 simultaneous rejections. Describe what the downstream sees, and the fix.**
Answer: a single wave of 500 attempts at `t + 1 s`, then another at `t + 2 s` if those fail. The fix is full jitter, so delays spread over `[0, 1]` s and the wave becomes a ramp.
Reasoning: a fixed delay synchronises clients that failed together, which is exactly the population that failed together.
Follow-up: does jitter reduce total load? No — it reduces peak load and leaves the mean unchanged, which is why the cap and the budget still matter.

**Q5 — A caller has a 5 s deadline. The callee retries three times at 2 s with 1.5 s of backoff. What is the worst-case elapsed time and what breaks?**
Answer: `3*2 + 1.5 = 7.5` s, so the caller abandons at 5 s while the callee is inside attempt three; the final attempt is spent on a response nobody will read.
Reasoning: the retry budget must be derived from the caller's deadline, not from the per-attempt timeout.
Follow-up: how would you state the invariant in CI? `n*t + sum(backoff) < deadline`, with backoff evaluated at its maxima and the assertion failing the build.

**Q6 — 100 logical operations produce 124 attempts, and the idempotency key is generated inside the retry loop. How many duplicate effects are admissible and how do you prove the count?**
Answer: up to 24, one per retry, because each retry presents a new key and is therefore a new operation. Prove it by counting effects grouped by the *operation* id: the distribution shows operations with two and three effects.
Reasoning: the key must be minted before the first attempt and committed in the same transaction as the effect; anything else identifies attempts.
Follow-up: what does the replay assert? Two deliveries of the same key produce one effect and one `DUPLICATE_SUPPRESSED` reason code.

**Q7 — p99 rises five-fold with a flat mean service time and a lag histogram showing a spike at 250 ms. Name the mechanism and the check that excludes queueing.**
Answer: eight coroutines each blocking the loop for 0.25 s, producing a 2.0 s stall per pass. Queueing is excluded because queueing produces a smooth lag distribution with no spike at a fixed duration.
Reasoning: a blocking call stalls every coroutine, so the p99 is bounded by the block duration and the arrival phase, not by permit contention.
Follow-up: why is the mean almost unchanged? Only requests that arrive inside the stall are delayed, and 2 s of stall in a 60 s window touches 3 % of arrivals.

**Q8 — A WebSocket peer sends frames but its pong is 90 s late. Your liveness rule drops it. What did you confuse, and what is the safer rule?**
Answer: pong liveness with transport liveness. The safer rule is "any inbound frame counts as liveness", with a separate application-level heartbeat for the cases where silence must be detected.
Reasoning: a peer streaming data is demonstrably alive; requiring a pong as well makes the check stricter than the property it claims to measure.
Follow-up: when is pong-only correct? When the protocol is request/response and silence really does mean dead — and then the interval must be documented as a contract.

**Q9 — Two tokenizers report 2.05 and 1.80 nats per token on the same text, with 4.1 and 3.6 bytes per token. Which model is better?**
Answer: neither — both are `0.5` nats per byte. The per-token difference of `1.80/2.05 - 1 = -12.2 %` is entirely the tokenizer's bytes-per-token ratio.
Reasoning: per-token loss divides the same total surprisal by a different count; per-byte loss divides by a count that is a property of the text.
Follow-up: when is per-token legitimate? Within one tokenizer, where the ratio is constant, and then only for comparisons at equal token counts.

**Q10 — A 14-token sequence has 13 tokens at 0.12 nats and one clamped at 1e-5. Compute the reported mean and explain the size of the error.**
Answer: `(13*(-0.12) + ln(1e-5))/14 = -13.0729/14 = -0.9338` nats, which is 7.78 times the true mean of 0.12.
Reasoning: the clamp substitutes `-11.5129` for a value that is lower still, so the mean is pulled toward the floor by `(11.3929)/14 = 0.8137` nats per token.
Follow-up: what assertion catches it? `min(s) <= mean <= max(s)`, plus a stored clamp count per sequence.

**Q11 — 4.8 M parent rows join to 6.24 M child rows and a revenue sum rises 30 %. Give the two checks and say why order matters.**
Answer: `COUNT(*)` against `COUNT(DISTINCT parent_key)` before aggregation, and NULL share on the key on both sides. Order matters because a NULL key drops rows from an inner join, so a fanout defect and a NULL defect can cancel or compound, and testing after the join has discarded the evidence.
Reasoning: fanout multiplies `4.8 M` by `1.30` to `6.24 M`; a 4.2 % NULL drop on the inflated set removes 262,080 units, which is small enough to hide inside the inflation.
Follow-up: how do you prove the fix? The replay must return the parent count for `COUNT(*)` and the same sum for both the pre-join and post-join queries.

**Q12 — Three shards report p99 of 180, 940 and 210 ms. Your dashboard says 443 ms. What is wrong and what is the right number?**
Answer: the dashboard averaged three percentiles, which describes no request. Pool the samples: with 2,000 pooled samples the p99 is the 1,980th order statistic, which lies in the shard that produced the 940 ms values.
Reasoning: a percentile is a rank; the mean of ranks is not a rank, and pooling changes which request occupies the rank.
Follow-up: if only aggregates survive, what do you publish? A maximum of per-shard p99 as a labelled lower bound, because the pooled percentile cannot be smaller than the largest shard percentile.

**Q13 — A grace period is 10 s and the drain p99 is 14 s. What exactly is lost per deploy, and why does nobody see it?**
Answer: about 1 % of in-flight requests at the tail, terminated by the stop signal. Nobody sees it because the client observes a connection reset and the service records no error — the record is destroyed with the process.
Reasoning: the two numbers live in different files, so each looks correct in isolation; the invariant is `grace > drain_p99 + margin`.
Follow-up: how would you test it? Roll one replica with 30 s of work in flight and assert zero dropped requests and a logged drain count.

**Q14 — A join change doubles the row count with the distinct key count flat. A colleague says growth is broad-based. What is the diagnosis and the cheapest disproof?**
Answer: fanout. The cheapest disproof is the ratio `rows/parent = 1.30` together with a flat distinct count — broad-based growth moves both.
Reasoning: uniform inflation across every segment is a data-shape signature; genuine growth concentrates somewhere and usually moves the distinct count as well.
Follow-up: what if the distinct count also rises 30 %? Then you have both new parents and new children, and the join still needs deduplication before the sum.

**Q15 — An image falls from 1.2 GB to 900 MB and cold start moves from 15.67 s to 12.3 s. Was the work worth it, and what would you do next?**
Answer: the pull fell from 6.67 s to 5.0 s, saving 1.67 s, so the remaining cold start is dominated by 9 s of initialisation. The next unit of work is initialisation, not bytes.
Reasoning: `T_cold = B/beta + t_init`, and after the change the second term is 73 % of the total, so its elasticity is higher.
Follow-up: how do you cut initialisation without hiding the cost? Move imports behind first use and gate readiness on a warm-up request, so the cost moves into the readiness window rather than the request path.

Takeaway: the questions that separate candidates are the ones that ask for a number and then for the condition under which the number is wrong — the second half is where the mechanism shows.

## 12. Quick Reference Card

| If you see | First probe | Likely issue | Containment that keeps the counter |
|---|---|---|---|
| elapsed 5x the predicted wave time | `ceil(N/k)` with the real `k` | CI-01 | none needed; publish `N`, `k`, `L` |
| RSS rising with backlog | size against `N * 19 KiB` | CI-02 | cap admitted tasks |
| timeout returns, effect still lands | live-task census delta | CI-03 | disable the side effect, not the timer |
| load 24 % above request rate | attempts per logical request | CI-06 | retry budget as a fraction of successes |
| duplicate effects after a timeout | effects per operation id | CI-07 | disable retries for that path |
| p99 spike at a fixed duration | loop-lag histogram | CI-09 | move the block off the loop |
| retries in one synchronised wave | timestamps modulo delay | CI-08 | add jitter, honour `Retry-After` |
| stream ends at exactly 60 s | end-time histogram | CI-12 | raise the idle timeout |
| "12 % better" per token | convert to bits per byte | CI-17 | none; report bpb |
| loss below the minimum token loss | clamp count per sequence | CI-19 | none; recompute from the dump |
| outputs ending at the cap | share at `max_new_tokens` | CI-21 | lower the cap, add a stop sequence |
| killed with work in flight | signal receipt inside the workload | CI-24 | extend grace, do not restart faster |
| request loss on every deploy | drain p99 versus grace | CI-25 | grace above drain p99 |
| revenue up 30 % with flat orders | `COUNT(*)` versus distinct key | CI-30 | deduplicate the right side |
| 4.2 % of rows missing | NULL share on the key | CI-31 | left join, keep the rows |
| dashboard p99 above every shard | pooled rank, not mean of ranks | CI-34 | publish a labelled lower bound |

**The four categories, one line each.** `VIOLATED_ASSUMPTION` — the contract drifted. `NUMERICAL_EDGE` — a boundary produced a finite wrong answer. `SCALE_FAILURE` — the cost curve turned vertical. `CONCEPT_CONFUSION` — two quantities sharing a name were swapped.

**The five stages, in order.** Classify, probe, contain, correct, replay. Never contain before the probe, and never close without a replay at the boundary.

**The four recurring mechanisms.** Unbounded accumulator, duplicated constant, premature aggregation, boundary convention. Testing these four catches most of the catalog.

**The condition to state before believing any number.** Every formula here has one — `rho < 1`, independence, a calibrated error rate, a unique parent key, a floor below the true value — and a number quoted without its condition is a guess with decimals.
