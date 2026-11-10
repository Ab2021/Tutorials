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

> **Purpose:** Turn a vague complaint ("the API got slow", "the numbers changed", "the queue is stuck") into one catalogued defect with a named mechanism, a check that separates it from its look-alikes, a bounded containment, a correction that changes a contract, and a replay that proves the correction held.
> **Evidence:** Every scenario below is a constructed teaching case, labelled ILLUSTRATIVE. Arithmetic in Sections 4, 5, 9 and 11 is DERIVED from the stated inputs and was recomputed with a script. No incident, vendor, model or company result is being reported.
> **What this file is not:** It is not an incident log, not a runbook for one organisation, and not a substitute for the primary sheet that owns each mechanism. Each issue names the sheet that carries its derivation.
> **Title note:** The filename's `25` is the collection's minimum issue count, not a cap. This catalog carries 32 issues so that each of the unit's six sheets contributes at least four defects that practitioners actually meet.

## 1. File Header

**Decision this file supports.** Given an observed defect in a workflow built from the unit's six topics, choose a first probe that can be run in minutes, read its result into one of four defect categories, and pick a containment that does not destroy the evidence needed for the correction. The file is written for the person who is already on call: it assumes the symptom is known and the cause is not.

**Why a catalog rather than a checklist.** Checklists fail at the discrimination step. "Check the logs" is not a probe, because logs from a saturated queue, a retry storm and a slow downstream all show a latency rise. A catalog earns its place by pairing every symptom with at least two competing mechanisms and a check whose outcome differs between them. The check must be decisive: after running it, one row of this file is eliminated.

**The four categories, operationally defined.** `VIOLATED_ASSUMPTION` means the code or query is correct against an assumption that no longer holds — the contract drifted, not the algorithm. `NUMERICAL_EDGE` means a boundary value (zero, NULL, a tie, a rounding step, a frame boundary, a clamp) produced a finite answer that is wrong. `SCALE_FAILURE` means the mechanism is right at low volume and its cost curve turns vertical — unbounded memory, linear-in-N work per request, synchronised retries, fanout from duplicate keys. `CONCEPT_CONFUSION` means two quantities that share a name were swapped: per-token loss for per-byte loss, a mitigation for a correction, a planned test for a result, a score for an authority.

**Severity discipline.** Severity is stated as consequence inside the declared scope (a nightly batch, one tenant's dashboard, one validator path), never as an incident claim. An issue that silently corrupts a published number outranks one that fails loudly, because the loud one is already contained by its own error.

**How to read an entry.** Each entry has seven fields: symptom (what a monitor or a human reports), mechanism (why the code produces it), competing causes (the look-alikes), discriminating check (the probe that separates them), correction (the contract change), prevention (the check that keeps it fixed), verification (the replay that proves it). The catalog table in Section 8 repeats the same fields in one line per issue, so a responder can scan first and read second.

**Provenance rule.** HYPOTHETICAL marks a constructed scenario. Where an entry quotes a formula from a sibling sheet, the formula is cited and the arithmetic recomputed here; the scenario stays constructed. No entry claims that any organisation experienced it.

**Assumed environment.** CPython 3.11+ with `asyncio`, an HTTP/1.1 and WebSocket client, an LLM served behind a chat API, containers run under a modern engine with cgroup v2, and a columnar SQL engine whose window functions follow the SQL standard's default frame. Version-specific behaviour is flagged where it changes a fix.

Takeaway: the file's value is not the list of defects; it is the pairing of each defect with a probe that a responder can finish before the incident call ends.

## 2. Concept Map / Mental Model

Every entry in this catalog is reached by the same five-stage path, and the order is not negotiable. Containment before diagnosis is how incidents become outages; diagnosis before containment is how a two-minute defect becomes a two-hour one. The diagram puts containment third because the first two stages are read-only and fast.

```text
  observed symptom
        |
        v
  (1) CLASSIFY  --- is the shape of the wrongness arithmetic, contract,
        |             scale, or naming?  (category decides the probe set)
        v
  (2) PROBE     --- one decisive measurement, recorded with its timestamp
        |             and the exact input that triggered the complaint
        v
  (3) CONTAIN   --- bounded, reversible, and evidence-preserving:
        |             defer, cap, disable one path, return a prior result
        v
  (4) CORRECT   --- change the contract or the arithmetic, not the symptom
        |
        v
  (5) REPLAY    --- original record + nearest boundary + one neighbour
                    pass -> close with a reason code
                    fail -> the mechanism was misidentified; back to (2)
```

Two rules make the path work. First, stage 2 produces a *number or a diff*, never an opinion; "it looks like a retry storm" is a hypothesis, "attempts per request moved from 1.02 to 1.24" is a probe. Second, stage 3 must leave the triggering record intact. A containment that truncates the payload, rotates the log, or overwrites the failed row removes the evidence that stages 4 and 5 depend on.

**A second axis: where the defect lives.** The category says what kind of wrongness it is; the layer says where to look first. Crossing the two is what turns a catalog into a triage instrument, because some cells are nearly empty and can be skipped under time pressure.

```text
layer            VIOLATED_ASSUMPTION   NUMERICAL_EDGE        SCALE_FAILURE        CONCEPT_CONFUSION
client/async     cancellation semantics zero permits            unbounded gather     "parallel" vs "concurrent"
network/HTTP     status-code contract   Retry-After seconds    retry amplification  mitigation vs correction
stream/WS        frame atomicity        heartbeat interval     slow-consumer queue  ping failure vs slow peer
model/inference  template identity      clamp at 1e-5          context cost         loss unit vs task claim
data/SQL         NULL three-valued      boundary inclusion     join fanout          count of rows vs count of keys
container        PID 1 signal path      grace vs drain time    layer cache churn    image size vs cold start
```

The empty-looking cells are the useful part: a `CONCEPT_CONFUSION` in the async layer is almost always the words "parallel" and "concurrent" being used interchangeably, and a `NUMERICAL_EDGE` in the SQL layer is almost always a boundary timestamp or a NULL, so those two cells resolve with one question rather than a probe.

**Why misclassification costs time.** A `SCALE_FAILURE` treated as `VIOLATED_ASSUMPTION` sends the responder to read the request handler, which is correct code operating outside its envelope; an hour disappears before anyone plots volume against latency. The reverse error is worse: a contract violation treated as a capacity problem invites the responder to add workers, which multiplies the wrong effect and typically doubles the side effects.

**The look-alike problem, stated once.** Three pairs dominate the catalog. A retry storm and a downstream slowdown both raise p99 and reduce throughput, and only the attempts-per-request series separates them. A truncated response and a wrong answer both present as "the model did not answer", and only a substring assertion on the exact prompt text separates them. A fanout join and a genuinely higher metric both change revenue, and only the row-count ratio before and after the join separates them.

Takeaway: classify first, probe with a number, contain without destroying evidence, correct the contract, replay at the boundary — and remember that the layer tells you which one question to ask before reaching for a probe.

## 3. Foundational Definitions

**Symptom.** The externally observable behaviour that generated the complaint, stated without a cause attached: "p99 rose from 180 ms to 940 ms between 02:10 and 02:40 UTC". A symptom that already names a cause is a hypothesis in disguise and must be rewritten before classification.

**Mechanism.** The causal chain inside the system that produces the symptom, in the vocabulary of the component: "each retry re-sends a 22,000-token prompt, so three attempts triple prefill work per logical request". A mechanism is falsifiable; a cause phrased as "the service was overloaded" is the symptom again with a different word.

**Competing cause.** A second mechanism that would produce the same symptom. The catalog requires at least two per entry, because a single-cause entry cannot be discriminated: it can only be confirmed by acting and seeing whether the symptom stops, which is the slowest possible test.

**Discriminating check.** A measurement, diff, or count whose outcome differs between the competing causes. It must be cheap enough to run during the incident, specific enough that its result eliminates a row, and recorded with the input that produced it. "Inspect the logs" is not a check; "count validator rejections per 1,000 requests, split by prompt-template hash" is.

**Containment.** A bounded, reversible action that limits harm while the cause is still unknown: deferring items, capping intake, disabling one code path, returning a previously computed result. Containment is judged on whether it preserves the evidence and whether it can be undone, not on whether it fixes anything.

**Correction.** The change that removes the mechanism: a contract clause, a precondition check, a corrected formula, an ownership assignment. A correction is falsifiable because the replay in the next definition can fail.

**Verification by replay.** Re-running the triggering record plus its nearest boundary case plus one ordinary neighbour, after the correction, and comparing output, reason code, side-effect count, and owner action against the expected values. Boundary inclusion matters: a fix for `<` versus `<=` that only replays interior cases is not verified.

**Idempotency key.** A caller-supplied value that identifies a logical operation across attempts, so a repeated delivery returns the first result instead of performing the effect again. The key must be created before the first attempt and persisted with the effect in one transaction; a key generated inside the retry loop identifies attempts, not operations.

**At-least-once delivery.** A transport guarantee that every message is delivered one or more times. Every retry mechanism and every queue in this unit is at-least-once, which is why the effect path, not the transport, carries the deduplication duty.

**Backpressure.** A signal from a consumer to a producer that its rate exceeds sustainable service, expressed as a bounded queue that blocks or rejects rather than growing. A system without backpressure converts a rate problem into a memory problem, which is strictly harder to diagnose.

**Concurrency versus parallelism.** Concurrency is interleaved progress on one or more execution units; parallelism is simultaneous execution on several. Forty coroutines against eight permits are concurrently issued and serially executed in waves, and calling that "parallel" predicts a speedup that eight permits cannot deliver.

**Fanout.** The multiplication of rows when a join key is not unique on the right side. One parent row joined to 1.3 children on average yields 1.3 rows per parent, and every downstream `SUM` is inflated by that factor — a numerical defect produced by a data-shape assumption.

**Three-valued logic.** SQL's treatment of `NULL` as unknown, so `NULL = NULL` evaluates to unknown and a predicate on a nullable column silently drops rows from an `INNER JOIN` or a `WHERE`. It is the most common source of `NUMERICAL_EDGE` in the unit's SQL sheets.

**Window frame.** The set of rows a window function sees for each row, defined by `ROWS` (physical offsets) or `RANGE` (value offsets, with peers treated together). The default frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes every peer of the current row and therefore differs from a `ROWS` running total whenever the ordering key ties.

**Censoring.** Reporting only the part of a distribution that was observed, for example a log-probability truncated to the top five candidates and clamped at a floor. Censored values substitute a bound for a measurement, and a mean over them is biased toward the floor, not merely noisier.

**Per-token versus per-byte loss.** Two normalisations of the same total surprisal, divided by tokens or by bytes. They differ by the tokenizer's bytes-per-token ratio, so a per-token comparison across tokenizers measures the tokenizers as much as the models.

**Percentile pooling.** Combining latency samples from several shards before taking a percentile. Because a percentile is a rank, not an average, pooling changes which request occupies the rank: the pooled p99 is dominated by whichever shard contributes the most long requests, and averaging per-shard p99 values produces a number that describes no request at all.

**Cache invalidation point.** The layer in a container build after which every subsequent layer must be recomputed because an earlier input changed. Placing a source copy before a dependency install moves the invalidation point to the top of the build and turns a cached rebuild into a full one.

**Signal forwarding.** The path by which a stop signal reaches the process that owns the work. A shell started as PID 1 does not forward signals to its children by default, so the container receives the stop and the workload never sees it.

**Reason code.** A short, stable token attached to a deferral or rejection so that later analysis can group decisions without parsing prose. Reason codes are the join key between an operational decision and its evidence.

**Owner.** The named role accountable for a contract's behaviour and for the replay that closes an issue. An entry without an owner is a note, not a correction, because nothing prevents the defect from returning.

Takeaway: the vocabulary exists to keep four things apart that incident language fuses — what was observed, why it happened, what was done to stop the bleeding, and what changed so it cannot recur.

## 4. Core Formulas with Full Derivation

The catalog's arithmetic is small: five results cover almost every quantitative issue in the file. Each is derived here in the order a responder needs it — an imbalance, its persistence under retry, its serialisation under a permit count, its inflation under a duplicate key, and its distortion under censoring — followed by the boundary where each breaks.

Every result below shares one precondition that is worth stating once: the inputs are measured over the same window and the same population as the symptom. A rate computed over a quiet hour and applied to a peak hour produces a number that is arithmetically right and operationally meaningless.

### 4.1 Base case: conservation of an unbalanced queue

Derivation. Consider one work-conserving server with arrival rate `λ` and sustainable service rate `μ`, both constant, no loss and no retries. In an interval of length `t`, arrivals add `λ·t` items and completions remove `μ·t`, so the backlog changes by `(λ − μ)·t`. Because backlog cannot be negative under this model, the content is the positive part of the integral, plus whatever was already queued at the start.

```text
FORMULA [CI-F1]: B(t) = B0 + max(0, (lambda - mu) * t)        while lambda > mu
                 B(t) = max(0, B0 - (mu - lambda) * t)        while lambda < mu
WHERE: B  [items] backlog at the end of the interval;
       B0 [items] backlog at the start, B0 >= 0;
       lambda [items/s] arrival rate; mu [items/s] sustainable service rate;
       t [s] interval length; all rates measured over the same window.
CONSTRAINT: single work-conserving server, constant rates, no loss, no retry,
       no priority classes, and mu measured at the deployed configuration
       rather than at the vendor's headline figure.
NUMERICAL EXAMPLE: lambda = 120 items/s, mu = 100 items/s, t = 360 s.
       (120 - 100) * 360 = 20 * 360 = 7,200 items of backlog.
INTERPRETATION: the defect is a rate imbalance, not a slow handler. No amount
       of per-item optimisation changes the slope while lambda exceeds mu; only
       demand shaping, added service rate, or shed load does.
```

**Why this is the base case.** It converts "the queue is growing" into a slope with units. A responder who measures the slope has already localised the defect to capacity or demand, and has already ruled out the whole `NUMERICAL_EDGE` family, because a numerical defect does not produce a linear backlog.

**Validity condition.** The formula holds only while both rates stay constant. A backlog that grows for forty minutes and then plateaus indicates that `λ` fell or `μ` rose, not that the model was wrong; recompute with the observed segment boundaries rather than averaging across them.

### 4.2 General case: retry multiplicity and amplified load

Derivation. Let `r` be the probability that one attempt fails and is retried, assumed independent across attempts, and let `n` be the maximum attempt count. The expected number of attempts per logical request is the sum of the probabilities of needing each successive attempt: `1 + r + r² + … + r^{n-1}`, a geometric series with ratio `r`.

```text
FORMULA [CI-F2]: A(r, n) = (1 - r**n) / (1 - r)          for r < 1, n >= 1
                 P_success(n) = 1 - r**n
                 load multiplier = A(r, n) applied to downstream capacity
WHERE: r is a probability in [0, 1); n is an integer attempt cap >= 1;
       A [-] is expected attempts per logical request, dimensionless;
       P_success [-] is the probability the request completes inside n attempts.
CONSTRAINT: failure independence across attempts (violated by a persistent
       outage or a shared rate limit), constant r during the retry window,
       and a linear cost per attempt.
NUMERICAL EXAMPLE: r = 0.20, n = 3.
       A = (1 - 0.008) / 0.80 = 0.992 / 0.80 = 1.24 attempts per request.
       P_success = 1 - 0.008 = 0.992, so 0.8% of requests exhaust three attempts.
INTERPRETATION: a 20% failure rate raises downstream load by 24%, not by 20%,
       because retries are themselves requests. Raising n from 3 to 4 adds
       0.0016 attempts per request: the marginal retry buys a 0.16% success
       gain and is almost never worth its tail latency.
```

**The variant that matters operationally.** When `r` is not constant but rises with load — the usual case for a rate-limited downstream — the series diverges in effect even though it converges in form: retries add load, load raises `r`, and the fixed point is reached only when the client's attempt cap or the server's shedding stops the loop. That is why the retry cap and the retry budget are capacity controls, not politeness.

**Validity condition.** Independence fails for a retry storm caused by one persistent fault: if the fault persists across the window, `r` is effectively 1 and `A = n` exactly. The probe that distinguishes the two is the distribution of final failure reasons: an independent-failure regime shows a mix of causes across attempts, a persistent-fault regime shows the same cause every time.

### 4.3 Variants: permit waves, saturation, and in-flight count

Derivation, permit waves. With `N` independent tasks each taking `L` seconds of wall time and a pool that admits at most `k` at once, the pool processes `ceil(N/k)` waves of `L` seconds, so elapsed time is `ceil(N/k)·L`. The coroutines are concurrent; the execution is serial in waves, and the speedup over a serial loop is exactly `ceil(N/k)` at best.

```text
FORMULA [CI-F3]: T_elapsed = ceil(N / k) * L           permits are the limit
                 S_speedup = (N * L) / (ceil(N / k) * L) = N / ceil(N / k)
WHERE: N [-] task count, integer >= 1; k [-] permits, integer >= 1;
       L [s] per-task wall time, assumed uniform; T_elapsed [s].
CONSTRAINT: tasks are independent and identical in duration; the limit is the
       permit count, not I/O latency, CPU, or a downstream connection pool.
NUMERICAL EXAMPLE: N = 40, k = 8, L = 0.250 s.
       waves = ceil(40/8) = 5;  T = 5 * 0.250 = 1.25 s;  serial = 10.0 s.
       speedup = 40/5 = 8.0x, which is exactly k, not N.
INTERPRETATION: if a responder expects 40x and observes 8x, nothing is broken;
       the expectation was "parallel" while the configuration is "concurrent
       with 8 permits". Raising N without raising k moves elapsed time linearly
       in ceil(N/k) and looks exactly like a slow downstream.
```

Derivation, saturation. Little's law states that the average number of items in a system equals the arrival rate times the average time in system: `L_inflight = λ·W`. It holds for any stable system under weak conditions, which is what makes it a diagnostic rather than a model: measure two of the three quantities and the third is determined, so a disagreement means the measurement windows differ.

```text
FORMULA [CI-F4]: L_inflight = lambda * W
WHERE: L_inflight [items] average concurrent items; lambda [items/s];
       W [s] average time from entry to exit, measured on the same population.
CONSTRAINT: steady state over the measurement window; a system that is filling
       (arrivals exceed departures) is not in steady state and Little's law
       describes no single instant.
NUMERICAL EXAMPLE: lambda = 300 requests/s, W = 1.5 s gives L_inflight = 450.
       At a 400-connection pool, 450 in flight means 50 waiters, so queueing --
       not handler speed -- accounts for the latency above service time.
INTERPRETATION: in-flight count and pool size are compared, never estimated.
       A pool smaller than L_inflight is a queue; a pool much larger than
       L_inflight is unused capacity being paid for.
```

**Validity condition for both.** Uniform `L` is the simplification that fails first: with a heavy-tailed duration distribution, `ceil(N/k)·L` understates elapsed time and the wave model must be replaced by a queueing calculation. The tell is a wave count that matches while elapsed time exceeds it.

### 4.4 Limits, counterexamples, and the two distortions that survive every fix

The first distortion is fanout. A left join to a right table whose key is not unique produces `Σ_j c_j` rows for `n` parents, where `c_j` is the child count for parent `j`, so the multiplier is the mean child count.

```text
FORMULA [CI-F5]: F = (1/n) * sum over parents of c_j = rows_after / rows_before
                 SUM_after = F * SUM_before + (contribution of the extra rows)
WHERE: n [-] parent row count; c_j [-] child rows matching parent j, c_j >= 0;
       F [-] fanout multiplier, dimensionless, F >= 0.
CONSTRAINT: the child table's key is unique per (parent, event) pair; a child
       table with its own duplicates multiplies F by that further multiplicity.
NUMERICAL EXAMPLE: n = 1,000 parents, 1,300 child rows, so F = 1.30.
       A revenue SUM of 84,000 currency units per 1,000 orders becomes 109,200
       if every parent is multiplied by 1.3 identically, an inflation of 30.0%.
INTERPRETATION: the defect is invisible in row-level spot checks, because each
       row is correct. Only the count ratio before and after the join exposes
       it, which is why the count belongs in the query's assertions, not the
       reviewer's memory.
```

The second distortion is censoring. If a scored token's probability falls below a floor `p_floor` and the harness reports `log(p_floor)` instead of the true value, the reported surprisal is *smaller* than the true surprisal, so the reported mean loss and perplexity are biased downward. The bias is one-sided and grows with the share of censored targets.

```text
FORMULA [CI-F6]: L_reported = (1/D) * [ sum over observed of s_i
                                        + m * (-ln p_floor) ]
WHERE: D [-] scored token count; s_i [nats] true surprisal of an observed token;
       m [-] count of censored targets, m <= D; p_floor [-] clamp probability.
CONSTRAINT: the clamp is a lower bound on probability and hence an upper bound
       on surprisal per censored target; the true value may be larger, never smaller.
NUMERICAL EXAMPLE: D = 1,000, true total surprisal 1,200.0 nats, m = 40,
       mean true surprisal of those 40 assumed 12.0 nats, p_floor = 1e-5.
       reported sum = (1,200.0 - 480.0) + 40 * 11.512925 = 720.0 + 460.517 = 1,180.52
       L_reported = 1.18052 nats/token versus L_true = 1.20000 nats/token.
INTERPRETATION: 4% censoring understates the mean by 1.62% and perplexity by
       1.93% (3.25606 versus 3.32012). The direction is guaranteed by the clamp;
       only the size depends on the assumption about the hidden 40.
```

**The counterexample that breaks the obvious fix.** The natural response to censoring is to drop censored targets, which changes `D` as well as the numerator and therefore compares two different populations. Dropping 40 targets from a 1,000-token sample and reporting the mean of the remaining 960 yields a number that is unbiased for a *different* question — "how well does the model predict tokens it found easy" — and that question is not the one a promotion decision asks.

**Limits, stated as a list.** `CI-F1` fails under non-constant rates and priority classes. `CI-F2` fails under correlated failures. `CI-F3` fails under non-uniform durations. `CI-F4` fails outside steady state. `CI-F5` fails when the child table itself contains duplicates of the join key. `CI-F6` fails when the clamp floor is set below the tokenizer's true minimum probability, in which case no censoring occurs and the correction is unnecessary.

Takeaway: five formulas and one inequality cover the catalog's arithmetic, and each carries an explicit condition under which it stops being true — which is the only reason to trust it during an incident.

## 5. Worked Numerical Example

**Scenario (ILLUSTRATIVE).** A document-enrichment service reads 1,200 PDFs per hour from a queue, converts each page to text, calls an LLM to extract two fields, and writes a row per document into an analytics table that a revenue dashboard sums. Between 02:10 and 02:40 UTC the dashboard's daily revenue figure rises 30% with no new orders, and the enrichment service's p99 rises from 180 ms to 940 ms. Nobody has changed a model, a prompt, or a container image; one deploy changed a join in the dashboard query and another changed a retry constant in the enrichment client.

**What the responder must decide.** Which of the two deploys explains which symptom, whether either symptom is a real business change, and what to roll back first. The example computes the three quantities that decide it: the fanout multiplier on the revenue sum, the retry-driven load multiplier on the model path, and the permit-wave elapsed time that explains the p99 if the retry story is false.

### Step 1 — Test the revenue figure for fanout

Rule: `CI-F5`. The dashboard joins a 1,000-row order table to an event table and sums `order_amount`; the new join key is `(customer_id, day)` rather than `order_id`.

```text
rows before join, order table            = 1,000
rows after join, event table matched     = 1,300
fanout multiplier F = 1,300 / 1,000      = 1.30          [-]

sum of order_amount before               = 84,000 currency units
predicted sum after, if only F changed   = 84,000 * 1.30 = 109,200 currency units
observed sum after                       = 109,200 currency units
residual after removing fanout           = 0 currency units
```

The prediction matches the observation exactly, which is strong evidence but not proof: a genuine 30% revenue increase would produce the same total if it were distributed proportionally across customers. The discriminating check is the count ratio, and it is decisive here because a real increase changes `rows before` while fanout changes only `rows after`.

```text
check: rows_after / rows_before = 1.30 on a day with no new orders
     => the row count grew without any new parent row, so the extra rows are
        join artefacts, and every additive measure over them is inflated.
```

**Independent re-check.** Sum the 300 extra rows directly: if the extra rows carry the same `order_amount` distribution as the originals, their total is `84,000 × 0.30 = 25,200`, and `84,000 + 25,200 = 109,200`. The two routes agree, and they must, because both are the same identity rearranged. A disagreement would mean the extra rows are not a proportional sample, in which case the multiplier alone is insufficient and the sum must be recomputed on deduplicated keys.

**Boundary case where the check breaks.** If the order table itself gained rows during the window — a backfill, for instance — `rows_before` is measured on a different population than `rows_after`, and the ratio conflates fanout with growth. The fix is to freeze the parent set: count `DISTINCT order_id` before and after the join. If `COUNT(DISTINCT order_id)` is unchanged while `COUNT(*)` rose, it is fanout, with no ambiguity.

### Step 2 — Test the latency for retry amplification

Rule: `CI-F2`. The enrichment client retries on a 429 response up to three attempts, with no jitter and a fixed 500 ms delay. The downstream reports 20% of first attempts rejected during the window.

```text
r = 0.20, n = 3
A = (1 - 0.20**3) / (1 - 0.20) = (1 - 0.008) / 0.80 = 0.992 / 0.80 = 1.24

baseline downstream load   = 1,200 documents/hour * 1 attempt = 1,200 attempts/hour
observed downstream load   = 1,200 * 1.24 = 1,488 attempts/hour
added load                 = 288 attempts/hour = 24.0% above baseline
```

Now the second-order effect, which is the part that produces the latency symptom: the 240 documents that fail twice contribute a delay of `2 × 0.5 = 1.0 s` of retry delay each, and the 48 that fail three times contribute `2 × 0.5 = 1.0 s` as well (two waits between three attempts, with the third attempt succeeding). Total retry delay per hour is `288 × 0.5 = 144 s`, spread across 1,200 documents, which is 120 ms of added average latency — not 760 ms.

```text
documents with >=1 retry = 1,200 * 0.20 = 240
documents with >=2 retry = 1,200 * 0.20**2 = 48
retry waits per hour     = 240 + 48 = 288
total retry delay        = 288 * 0.5 s = 144 s
mean delay per document  = 144 / 1,200 = 0.120 s = 120 ms
```

**Interpretation.** 120 ms of added mean latency cannot explain a p99 that rose by 760 ms. So either the retry delay is not the mechanism, or the failures are correlated so that `r` is far higher inside the window than the 20% reported for the whole hour. That is the decisive fork, and Step 3 tests it.

**Independent re-check.** The load multiplier from the closed form (`1.24`) must equal the direct sum `1 + 0.20 + 0.04 = 1.24`. It does. If a responder instead computed `1.20` by adding only the first retry, downstream capacity would be under-provisioned by 3.3% of that load, which is small but compounding when the same arithmetic appears in a capacity plan.

### Step 3 — Test the tail for permit-wave serialisation

Rule: `CI-F3`. The client's concurrency limit is 8 permits and the observed service time per document is 0.250 s against the model endpoint. In a 30-second burst the queue delivered 40 documents at once — a redelivery after a consumer restart, which is exactly the kind of event that produces a short, sharp tail.

```text
N = 40, k = 8, L = 0.250 s
waves   = ceil(40 / 8) = 5
elapsed = 5 * 0.250 = 1.25 s  = 1,250 ms
serial  = 40 * 0.250 = 10.0 s
speedup = 40 / 5 = 8.0x
```

A document at the back of the burst waits four waves, i.e. `4 × 0.250 = 1.0 s = 1,000 ms` of queueing inside the client, before its own 250 ms of service begins. That is the p99 mechanism: not a slow downstream, not retries, but a burst admitted into a pool of eight and serialised in waves. The observed p99 of 940 ms sits between the fourth-wave start (750 ms) and the fifth-wave start (1,000 ms), consistent with a burst slightly smaller than 40 arriving.

**Independent re-check.** In-flight count from Little's law with `λ = 40/30 = 1.333 documents/s` and `W = 0.250 s` gives `L_inflight = 0.333`, so the steady state needs one permit and the pool of eight is ample. The burst, not the pool size, is the defect: a redelivery of 40 documents at once is a `SCALE_FAILURE` in the consumer's batch admission, and no permit count that also fits the memory budget would absorb it.

**Sensitivity note.** If `L` is 0.250 s on average but heavy-tailed with a p99 of 1.2 s, four waves of a heavy tail produce a much longer tail than the wave arithmetic suggests, and the wave count matching while elapsed time exceeds prediction is the tell. Conversely, if the burst had been spread over 3 seconds rather than delivered at once, `ceil(40/8)` would still be 5 but the walls between waves would be filled, and the tail would collapse to roughly `L` plus a small queueing term.

### Step 4 — Cost the two corrections and pick the order

The revenue defect is a publication defect: a wrong number is on a dashboard that someone may act on. The latency defect is a throughput defect that has not yet caused a missed SLA. The order follows from that asymmetry, not from which fix is easier.

```text
revenue exposure  = 25,200 currency units overstated per day
                   (the fanout delta, recomputed from Step 1)
rollback cost     = one query revert + one refresh; ~6 minutes of dashboard staleness
latency exposure  = 1 burst per consumer restart, ~1,200 ms of p99 for ~30 s
                   with 1,488 - 1,200 = 288 retries/hour * 0.5 s = 144 s/hour of
                   added downstream occupancy, i.e. 4.0% of one worker
```

**Verdict.** Revert the join first, because it changes a published number and the revert is cheap and exact. Fix the burst admission second, with a bounded batch size and a per-document deduplication key so the redelivery cannot multiply attempts. Fix the retry policy third: jittered backoff plus a retry budget, since the 24% load multiplier is a real capacity cost even though it is not the p99 mechanism.

**Wrong path, and what it would have concluded.** A responder who saw the p99 first and added four workers to the pool would have spent money and changed nothing, because the pool was never the binding constraint (`L_inflight = 0.333` against eight permits). A responder who saw the revenue rise and attributed it to a promotion would have shipped a 30% overstatement into a planning cycle. The step that catches both is the count ratio in Step 1 and the in-flight comparison in Step 3 — two probes, each under a minute, each decisive.

**Remaining uncertainty.** The 20% rejection rate is an hourly average and the burst happened inside a thirty-second sub-window, so the effective `r` during the burst is unknown and could have been 100%. Under `r = 1.0` the retry term becomes `n = 3` attempts per document, tripling load, and the 144 s of retry occupancy becomes 360 s. Nothing in the example measures whether the dashboard's freshness SLA allows a 6-minute rollback, and nothing measures whether the burst arrived from redelivery or from the producer — that requires the consumer's offset log.

Takeaway: compute the multiplier before choosing the fix, because a 1.30 fanout and a 1.24 retry multiplier look like the same "30% worse" symptom and require opposite corrections.

## 6. Visual / Geometric Intuition

Four pictures carry the catalog's intuition: the backlog trajectory that separates a rate defect from a burst, the retry tree that shows where the extra attempts come from, the wave diagram that explains a p99 nobody can reproduce, and the decision tree that orders the probes. Each is schematic; every label is an exact value from Section 5 or Section 4.

### 6.1 Backlog trajectory: rate defect versus burst

The two shapes below are the fastest way to tell capacity from burst, and the test takes two samples one interval apart. A rate defect has a slope that persists; a burst has a step that either drains or does not, depending only on whether `μ > λ` in the quiet period.

```text
backlog
1,800 |                                              *  rate defect (CI-08)
      |                                        *
1,200 |                                  *
      |                            *
 600  |                      *                        ____  burst (CI-03)
      |                *                    _______/
   0  +----*----*----*--------*--------*---/------------------> t (min)
          0    5   10   15   20   25   30   35
              ^                            ^
        backlog = 0                 drain begins; slope
        at 02:10                   becomes -60 items/min

  rate defect: lambda 150/s, mu 120/s -> +30/s = +1,800/min, no drain
  burst:       +40 documents at once, drained at 8 permits * 4/s = 32/s
               => 40 / 32 = 1.25 s to clear, so the step disappears by 02:11
```

Axes: minutes since the first symptom on the horizontal, items of backlog on the vertical. Key point: the two curves differ in *derivative after the event*, not in peak height. A responder who only sees the peak cannot distinguish them; a responder who takes two samples can, and the second sample costs one query.

### 6.2 The retry tree and the attempt budget

Each retry is a new request, so the tree below is also a load diagram. The widths are the attempt counts from `CI-F2` at `r = 0.20`, and the sum of the widths is `1.24` attempts per logical request.

```text
1,000 logical requests
  |
  +-- 800 succeed on attempt 1                     (0 retries)
  |
  +-- 200 fail attempt 1
        |
        +-- 160 succeed on attempt 2               (1 retry  = 160 requests)
        |
        +-- 40 fail attempt 2
              |
              +-- 32 succeed on attempt 3          (2 retries =  64 requests)
              |
              +--  8 fail all three                (3 attempts, 24 requests)

  total attempts = 1,000 + 160 + 64 + 24 = 1,248  -> 1.248 per logical request
  closed form    = 1 + 0.2 + 0.04 + 0.008 = 1.248 attempts
```

Axes: vertical depth is attempt number, horizontal width is request count. Key point: the third attempt contributes 64 requests to load and rescues only 32, so its load-to-rescue ratio is 2:1 — every attempt level has ratio `1/r = 5:1` at `r = 0.2`, which is why a retry cap of two is usually where the marginal value collapses.

**Caption.** A retry storm is not a tree that grows deeper; it is a tree whose branching factor rises. At `r = 0.5` the same cap produces `1 + 0.5 + 0.25 = 1.75` attempts per request, a 75% load increase from a failure rate that a dashboard shows as "half the requests retried", which reads as benign.

### 6.3 Permit waves: where the p99 comes from

The wave diagram answers the question "why is the slowest request slow when nothing is slow". Each horizontal bar is one wave; the numbers inside are document indices in arrival order.

```text
time (ms)   0        250       500       750      1000      1250
            |---------|---------|---------|---------|---------|
wave 1      [ 1..8   ]
wave 2                [ 9..16  ]
wave 3                          [17..24  ]
wave 4                                    [25..32  ]
wave 5                                              [33..40  ]

p50 of the burst: document 20 -> starts at 500 ms, ends 750 ms   = 750 ms total
p90 of the burst: document 36 -> starts at 1000 ms, ends 1250 ms = 1250 ms total
p99 of steady traffic (1 doc at a time): 250 ms

  queueing delay = floor((i-1)/k) * L      i = arrival index, k = permits
  total latency  = queueing delay + L
```

Axes: time in milliseconds; each bar is a wave of at most `k` documents. Key point: latency in a saturated pool is a step function of arrival index, so the distribution is multimodal with modes at `L, 2L, 3L, …`. A histogram with evenly spaced humps is diagnostic of a permit pool; a smooth long tail is diagnostic of a slow downstream.

**Caption.** This is why "we cannot reproduce the p99" is a normal report: the p99 was produced by a burst that lasted thirty seconds, and replaying single documents never occupies more than one wave. Reproducing it requires replaying the burst shape, which is what the consumer's offset log preserves.

### 6.4 The probe decision tree

The catalog is large, so the first probe must be chosen by symptom shape rather than by topic familiarity. The tree below is the one a responder can run before the incident call starts.

```text
symptom?
  |
  +-- "a number is wrong" ---------> is rows_after/rows_before != 1 on unchanged
  |                                  parents?  --yes--> fanout (CI-05, CI-19)
  |                                             --no --> NULL/boundary (CI-21, CI-22)
  |
  +-- "it is slow" ----------------> attempts/request > 1.05? --yes--> retries (CI-06)
  |                                  in_flight > pool size?   --yes--> saturation (CI-08)
  |                                  p99 multimodal?          --yes--> waves (CI-03)
  |
  +-- "it stopped" ----------------> last successful event time vs now
  |                                  -- gap at a deploy --> signal/contract (CI-11, CI-26)
  |                                  -- gap at a clock edge --> timezone (CI-18)
  |
  +-- "it is expensive" -----------> spend / request, split by component
                                     -- input tokens up --> context (CI-13, CI-14)
                                     -- attempts up     --> retries (CI-06)
                                     -- instances up    --> cache miss (CI-27)
```

Key point: the tree's first question is about the *shape* of the complaint, because shape determines which family of probe can be decisive. "It is slow" with a multimodal p99 and "it is slow" with a smooth tail are two different investigations, and the histogram is cheaper than either.

Takeaway: draw the backlog before arguing about capacity, draw the retry tree before adding workers, draw the waves before hunting a slow dependency, and choose the first probe from the symptom's shape rather than from the component you know best.

## 7. Comparison / Selection Table

The catalog's recurring decision is *which probe to run first*. A probe is judged on three properties: how fast it returns, whether its result eliminates a row rather than merely raising suspicion, and whether it is safe to run in production. The table ranks the unit's probes on those properties so a responder under time pressure can pick without re-deriving the reasoning.

| Probe | Returns in | Eliminates | Safe in prod | Cost | Fails when | Owner sheet |
|---|---|---|---|---|---|---|
| Count ratio before/after a join (`COUNT(*)` vs `COUNT(DISTINCT key)`) | seconds | fanout, duplicate-key inflation | yes, read-only | one extra aggregate per query | parent population changed in the window | SQL analytics |
| Attempts per logical request | minutes | retry storm, client-side duplication | yes, from existing counters | a counter and a ratio | counters are per attempt and unlabeled | REST APIs |
| In-flight count versus pool size | seconds | queueing versus slow handler | yes | one gauge comparison | the system is filling, not steady | Async Python |
| p99 histogram, not scalar | minutes | wave serialisation, bimodal tails | yes | a histogram metric | buckets are too coarse to show modes | Async Python |
| `EXPLAIN` row estimate versus actual rows | minutes | bad statistics, join order, spill | yes, on a replica | one plan plus one run | the plan cache hides the chosen plan | SQL analytics |
| KV-cache bytes per request | seconds | context-length cost, concurrency loss | yes, arithmetic only | tokens × bytes/token | attention is sliding-window, not full | LLM fundamentals |
| Template hash diff | seconds | prompt drift, duplicated system block | yes | one hash comparison | the template is built at runtime from config | LLM fundamentals |
| Full-vocabulary log-probability check | minutes | censored loss, clamped floor | yes, on a replay sample | a replay of 50 items | the endpoint only returns top-k | LLM fundamentals |
| `nr_throttled` and `nr_periods` in cgroup | seconds | CPU throttling versus slow code | yes | two counters | the workload is IO-bound | Docker |
| Layer `CACHED` markers in build output | minutes | cache invalidation position | yes, in CI | one build log read | the builder rewrites timestamps | Docker |
| Digest comparison local versus registry | seconds | "works locally" image drift | yes | two digests | the tag is rebuilt in place | Docker |
| `Retry-After` versus client backoff | seconds | synchronised retry waves | yes | read one header, one config | the server omits the header | REST APIs |
| Reason-code distribution | minutes | policy misroute, silent deferral | yes | a group-by on the ledger | reason codes are free text | All |
| Offset-log replay of the burst shape | tens of minutes | burst admission, redelivery | yes, on a copy | a replay harness | the log has short retention | Async Python |

**How to read the ranking.** The first three columns decide the choice: a probe that returns in seconds and eliminates a category beats a probe that returns in minutes and merely narrows it. The `Cost` column exists to stop a responder from reaching for the expensive probe first out of familiarity — the plan-and-run probe is the most informative item in the table and almost never the first one to run.

**Selection rules, stated as three sentences.** If the symptom is a wrong number, the count ratio runs first because it is a read-only aggregate and it separates data shape from data change. If the symptom is latency, the histogram runs before any single dependency is instrumented, because a multimodal histogram removes three hypotheses at once. If the symptom is a stop, the gap timestamp is compared against the deploy timeline before any component is restarted, because a restart destroys the state that would have explained the gap.

**When the table is the wrong instrument.** Under a data-loss symptom, no probe in this table is safe until writes are frozen: a count ratio on a table still receiving writes measures the writes. Under a security symptom — a leaked credential, an unexpected egress — the correct first action is containment by revocation, and every probe here becomes secondary because it reads the very data whose exposure is the incident.

Takeaway: rank probes by decisiveness and speed, not by how much instrumentation they use; the cheapest decisive probe is the professional choice, and the most informative probe is usually the fourth one you run.

## 8. Failure Modes & Edge Cases

The catalog is below: 32 issues, ordered by the layer in which they are first detectable. Every row carries the seven required fields; the prose under each row carries the mechanism and the number that makes the entry checkable. The ordering is deliberate — a responder reads the layer of the symptom, not the whole file — and the category counts are shown at the end of the section so a reader can see which families are dense and which are thin.

### 8.1 Async Python: scheduling, cancellation, and pools

The five defects here share a property that makes them hard: the code is *correct* for the workload it was written against, and the defect appears only when the workload changes shape. None of them raises an exception on the happy path.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verify |
|---|---|---|---|---|---|---|---|
| CI-01 | CONCEPT_CONFUSION | 40 tasks take 1.25 s where 8 permits should give 0.25 s | slow downstream; missing permits | compare elapsed to `ceil(N/k)*L` | state the permit count as the capacity contract | publish `N`, `k`, `L` beside any speedup claim | replay 40 tasks, assert 5 waves |
| CI-02 | SCALE_FAILURE | RSS climbs with backlog size, then the process is killed | leak in a library; large payloads | peak RSS versus `N * per-task bytes` | bound the fan-out with a semaphore or chunking | cap admitted tasks per unit time | replay 50,000 tasks, assert RSS bound |
| CI-03 | VIOLATED_ASSUMPTION | `await` returns but the work keeps running after cancel | slow code; ignored exception | task count after cancel versus before | await cancellation and propagate it | cancellation owned by the caller, checked in `finally` | cancel mid-flight, assert zero live tasks |
| CI-04 | SCALE_FAILURE | intermittent "Task was destroyed but it is pending" | real timeout; loop shutdown | live-task set versus retained references | retain a strong reference until completion | register every task in a supervised set | shutdown test with 1,000 tasks, assert none dropped |
| CI-05 | NUMERICAL_EDGE | worst-case latency double the expected timeout | slow server; retry delay | recompute `n*t + sum(backoff)` | derive the budget from the worst case, not the per-attempt value | assert budget < caller's deadline in CI | replay a forced-timeout request, assert elapsed |

**CI-01 — Concurrency mistaken for parallelism.** Mechanism: the pool admits eight tasks and the other 32 wait, so elapsed time is `ceil(40/8) × 0.250 = 1.25 s` while a responder who expected 40-way parallelism predicts 0.25 s. The competing explanation — a slow downstream — fails the same arithmetic: a slow downstream raises `L` for every task including the first, so the first task's completion time would also exceed 250 ms, and it does not.

**CI-02 — Unbounded `gather`.** Mechanism: each admitted task holds a response buffer and a coroutine frame, about 1 KB per task in the constructed case, so 50,000 tasks at once retain roughly 51.2 MB of state before any payload. The leak hypothesis fails a specific test: RSS that tracks backlog size and falls after the backlog drains is bounded state, not a leak, whereas a leak's RSS stays high after the drain. Correction is a semaphore sized by the memory budget, not by a round number.

**CI-03 — Cancellation that only stops the waiter.** Mechanism: `asyncio.wait_for` cancels the *awaiting* coroutine; a task started with `create_task` inside it is not cancelled unless the code cancels it, so the timeout returns while the work continues and the side effect still lands. The "slow code" hypothesis fails because the elapsed time of the *caller* is already correct; only the side-effect count is wrong. The check is a task census before and after the timeout, and the number that matters is the delta, not the total.

**CI-04 — Fire-and-forget tasks collected mid-flight.** Mechanism: the event loop holds only weak references to tasks, so a task whose only reference is the loop's can be garbage collected while pending — roughly 4 ms of work lost per dropped task in the constructed case, 240 ms across an hour of 1,000 drops. The "real timeout" hypothesis fails because this symptom appears only at shutdown or under memory pressure, never in steady state. Correction is a supervised set that holds strong references and is drained at exit.

**CI-05 — Timeout budget arithmetic.** Mechanism: three attempts at 3 s each plus backoff waits of 0.5 s and 1.0 s give a worst case of `9 + 1.5 = 10.5 s`, while the caller's deadline is 6 s, so the client cannot honour the contract it advertises. The "slow server" hypothesis fails a boundary test: a request that fails instantly on all three attempts still consumes 1.5 s of backoff, which no server speed can remove. Prevention is an assertion in CI that `n·t + Σbackoff ≤ deadline`.

### 8.2 HTTP and WebSocket: retries, backpressure, and stream termination

The defects here live in the contract between client and server: status semantics, retry eligibility, and the assumption that a stream that has not ended is still alive. Each has a look-alike in the transport layer, and the discriminator is almost always a header, a counter, or an offset rather than a log line.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verify |
|---|---|---|---|---|---|---|---|
| CI-06 | SCALE_FAILURE | downstream load 24% above request rate | traffic growth; bigger queries | attempts per logical request | retry budget as a fraction of successes | cap attempts and add full jitter | replay 1,000 requests at `r=0.2`, assert 1.24 |
| CI-07 | VIOLATED_ASSUMPTION | duplicate charges after a timeout | payment provider fault; double submit | count effects per idempotency key | persist the key with the effect, return the prior result | key created before the first attempt | replay same key twice, assert one effect |
| CI-08 | NUMERICAL_EDGE | retries pile up in a synchronised wave | genuine outage; capacity drop | retry timestamps modulo the delay | honour `Retry-After`, add full jitter | assert jitter is non-zero in config test | replay 500 rejections, assert spread |
| CI-09 | SCALE_FAILURE | gateway memory grows with slow consumers | memory leak; large frames | queue depth per connection | bounded queue, drop-oldest or close | per-connection queue cap and metric | replay a stalled consumer, assert close |
| CI-10 | CONCEPT_CONFUSION | connections dropped as "dead" while data flows | real packet loss; proxy timeout | timestamp of last data versus last pong | treat any inbound frame as liveness | document the liveness definition | replay slow-but-alive peer, assert kept |
| CI-11 | VIOLATED_ASSUMPTION | stream ends at exactly 60 s with partial output | model stop condition; client cancel | end time clustered at a constant | raise proxy idle timeout or send keepalives | pin every hop's idle timeout | replay a 90 s stream, assert completion |
| CI-12 | NUMERICAL_EDGE | request body silently shortened | client bug; wrong encoding | byte length sent versus `Content-Length` | reject or chunk instead of truncating | validate declared against actual length | replay 48 MB body, assert rejection |

**CI-06 — Retry amplification.** Mechanism: at `r = 0.20` and three attempts, `A = 1.24`, so a downstream sized for 1,200 requests/minute receives 1,488. The "traffic growth" hypothesis fails because request *rate* and attempt *rate* diverge by exactly the multiplier, and the multiplier is computable from the failure rate the server already reports. Note the direction of blame: the server reports a 20% rejection rate, which reads as a server problem, while the 24% load increase is the client's contribution.

**CI-07 — Retry of a non-idempotent effect.** Mechanism: the client times out after the server committed, retries, and commits again; the ledger shows two effects for one logical operation. The "provider fault" hypothesis fails a key-based count: a provider fault produces duplicate effects with *different* keys only from different callers, whereas this defect produces duplicates under one key. Correction is not "retry less" but "make the key persistent and check it inside the same transaction as the effect".

**CI-08 — Synchronised retry wave.** Mechanism: a fixed 500 ms backoff has a period the downstream can predict, so 500 rejections in one second become 500 retries 500 ms later — a spike of 5,000 requests/s against a 500/s capacity, a tenfold overshoot. The "capacity drop" hypothesis fails a timestamp histogram: synchronised retries cluster in a narrow band whose width equals the jitter, which is zero here. Correction is full jitter in `[0, base·2^attempt]`, plus honouring `Retry-After` where present.

**CI-09 — Slow consumer with an unbounded queue.** Mechanism: a client that reads at 3,000 frames/s while the server produces 5,000/s accumulates 2,000 frames/s; at 1 KB per frame that is 2.048 MB/s and 7.37 GB/hour of retained memory per stalled connection. The "memory leak" hypothesis fails a per-connection profile: leak memory is not released when the connection closes, queue memory is. Correction is a bounded per-connection queue with an explicit policy — close, or drop oldest with a counter.

**CI-10 — Liveness confused with throughput.** Mechanism: the implementation resets its liveness timer only on data frames, so a peer that sends heartbeats but no application frames is declared dead after the idle timeout even though the socket is healthy. The "packet loss" hypothesis fails because packet loss drops *all* frame types, including pongs, and the check is whether any inbound frame arrived inside the window. Correction is to touch the liveness timer on every inbound frame, not on data frames only.

**CI-11 — Stream cut by an intermediate idle timeout.** Mechanism: the model produces its first token in 2 s and then a 45-second gap during a long prefill or a tool call; the proxy's idle timeout is 60 s, but the gap plus 18 s of prior decode reaches it. The tell is that end times cluster at one constant, which no model stop condition produces, because stop conditions cluster at token counts. Correction is a keepalive comment frame or a raised idle timeout at the hop that owns the constant.

**CI-12 — Body truncated at a size limit.** Mechanism: a proxy with a 48 MB body limit reads 48 MB and forwards it, so the client's declaration of 48 MB succeeds while the actual 50 MB fails mid-stream and the server sees a short body. The "wrong encoding" hypothesis fails a length comparison: UTF-8 mis-encoding changes the byte count predictably, a truncation leaves a body shorter than the declared length with no encoding error. Correction is a 413 with the limit stated, never a silent trim.

### 8.3 LLM inference: units, identity, and the boundary of a claim

These six defects are the most expensive in the catalog, because each produces a *correct-looking number* that is then used to justify a decision. None of them fails loudly. The discriminator for all six is the same move: recompute the quantity from raw counts in a unit that the comparison cannot flatter.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verify |
|---|---|---|---|---|---|---|---|
| CI-13 | CONCEPT_CONFUSION | candidate model "8% better" but users see no change | real quality gain; prompt drift | convert both to bits/byte | report bpb for cross-tokenizer claims | ban per-token comparisons across tokenizers | recompute from raw surprisals, assert bpb |
| CI-14 | VIOLATED_ASSUMPTION | scores differ between eval and serving on identical text | model nondeterminism; sampling | dump the exact templated prefix ids | pin template hash and tokenizer version | manifest hash in every score record | replay one item both paths, diff ids |
| CI-15 | NUMERICAL_EDGE | mean loss lower than any single token's loss | arithmetic bug; weighting | recompute `Σs/D` from the dump | store total surprisal and `D` beside the mean | assert `min(s) <= mean <= max(s)` | replay 100 tokens, assert bound |
| CI-16 | SCALE_FAILURE | context fits but latency and spend triple | slower hardware; longer answers | tokens per component, plotted | size retrieval by `k`, not by relevance | budget per component in the request builder | replay with `k=4` versus `k=16` |
| CI-17 | NUMERICAL_EDGE | 12% of outputs hit `max_new_tokens` exactly | verbose prompts; model change | share of outputs at the cap | raise cap or add a stop sequence | alert on cap-hit share | replay capped request, assert stop reason |
| CI-18 | CONCEPT_CONFUSION | "yesterday's" totals shift every night | data correction; late events | compare event time to process time | define the cut-off in UTC and fix the window | one timezone constant per pipeline | replay a DST boundary event |

**CI-13 — Per-token loss used across tokenizers.** Mechanism: a coarser tokenizer groups more bytes per token, so each prediction is mechanically easier and per-token loss falls without any improvement per byte. In the sibling sheet's constructed example the same response text scores 7.19% better in nats/token, 8.20% better in perplexity, and 3.33% better in bits/byte — three correct numbers, one defensible claim. The "prompt drift" hypothesis is eliminated because drift changes the *content* being scored, which moves per-byte loss too; a tokenizer difference moves only per-token loss.

**CI-14 — Chat template mismatch.** Mechanism: evaluation scores raw user text while serving sends a templated prefix with role markers, so the model conditions on a different distribution than the one measured. The effect is not small: a duplicated system header adds 28 template tokens per request and can shift mean NLL by an amount comparable to a real quality difference. "Sampling nondeterminism" fails the id-diff test, because a nondeterministic sampler changes the *output* while a template mismatch changes the *input ids*, and the input ids are comparable without any sampling.

**CI-15 — A mean that cannot be a mean.** Mechanism: the harness averages a per-token loss computed from a clamped floor and a censored top-k, so the reported mean sits below the minimum of the observed surprisals. The check is an invariant, not a measurement: `min_i s_i ≤ mean ≤ max_i s_i` must hold for any weighted mean of non-negative surprisals. An arithmetic bug and a censoring bug both violate it; they are separated by whether the individual surprisals in the dump are themselves clamped.

**CI-16 — A context that fits is not a context that is affordable.** Mechanism: retrieval `k` raised from 4 to 16 adds 12 passages; at 1,400 tokens each that is 16,800 extra prompt tokens, so prefill compute and KV-cache bytes per request rise by 16,800 × 131,072 = 2.1 GiB of cache on the usual shape, and a 4× longer prompt costs 4× the prefill. The "slower hardware" hypothesis fails because TTFT scales with prompt tokens while inter-token latency does not, so a hardware regression would move both.

**CI-17 — Silent truncation at the output cap.** Mechanism: 12% of responses end at exactly `max_new_tokens`, so the completion is cut mid-sentence and a downstream parser sees an incomplete JSON object. The share at the cap is a direct measurement, and it separates truncation from verbosity: verbose answers have a smooth length distribution, truncated ones have a point mass at the cap. Correction is a raised cap *plus* a stop-sequence audit, because a cap raise alone moves the pile-up.

**CI-18 — Event time mistaken for process time.** Mechanism: a nightly job filters `WHERE created_at >= CURRENT_DATE - 1` in server-local time while the source timestamps are UTC, so during the five hours when local time is behind UTC the job reads a window that overlaps yesterday and re-counts rows. The "late events" hypothesis is eliminated by checking whether the affected rows have event times inside both windows; late-arriving events move once, a timezone defect moves them every night at the same hour.

### 8.4 Containers: identity, signals, and cache position

Container defects are almost all identity defects: the thing that ran is not the thing that was tested, or the stop that was sent is not the stop that was received. Three of the five are visible in a digest or a counter, which makes them cheap to prove and expensive to guess.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verify |
|---|---|---|---|---|---|---|---|
| CI-19 | VIOLATED_ASSUMPTION | workload killed with work in flight at stop | slow handler; queue loss | process tree at stop, signal receipt | exec the workload as PID 1 | assert single-process tree in image test | send stop with 30 s of work, assert drain |
| CI-20 | NUMERICAL_EDGE | a share of requests lost on every deploy | client timeout; pod crash | drain p99 versus grace period | set grace above drain p99 | budget = drain p99 + margin, in CI | rolling restart, assert zero dropped |
| CI-21 | CONCEPT_CONFUSION | works locally, not in the cluster | code difference; missing env | digest local versus deployed | deploy the tested digest, not the tag | promote by digest only | compare digests, assert equal |
| CI-22 | SCALE_FAILURE | cold start 24 s, not 1.2 s | slow runtime; CPU limit | image bytes pulled versus bytes needed | reorder layers, shrink base | assert image size budget in CI | rebuild from cache, assert pull time |
| CI-23 | NUMERICAL_EDGE | throttled at 40% CPU usage | slow code; memory pressure | `nr_throttled` over `nr_periods` | raise quota or lower period sensitivity | alert on throttle ratio, not usage | replay burst, assert throttle ratio |

**CI-19 — PID 1 that swallows the signal.** Mechanism: `ENTRYPOINT ["sh", "-c", "..."]` makes the shell PID 1, and the shell does not forward `SIGTERM` to its child, so the workload runs until `SIGKILL` at the end of the grace period — a 45 s grace minus 30 s of unforwarded drain work means the last in-flight batch is lost on every stop. The "slow handler" hypothesis fails because a slow handler still *finishes* within the grace period if the signal arrives; here the signal never arrives at the process that could react.

**CI-20 — Grace period shorter than the drain tail.** Mechanism: on stop, a server drains in-flight requests; if the drain p99 is 12 s and the grace period is 10 s, then with a uniform drain time the share of stops that lose work is `(12 − 10)/12 = 16.7%`. Note the arithmetic: the tail is what matters, so a mean drain of 2 s with a 12 s p99 loses the same share. The "client timeout" hypothesis fails because client-side loss correlates with *deploy events*, while a timeout correlates with load.

**CI-21 — Tag identity drift.** Mechanism: the cluster pulls `app:latest`, which was rebuilt after the test run, so the deployed bytes differ from the tested bytes with no build failure and no version change. The "missing env var" hypothesis fails a digest comparison: an env difference shows identical digests with different configuration, a tag drift shows different digests for the same tag. Correction is promotion by digest, which makes the tested artefact the deployed artefact by construction.

**CI-22 — Cold start dominated by transfer.** Mechanism: at 100 MB/s an extra layer of 2,400 MB costs 24 s of pull against a 1.2 s boot, so cold start is 95% network and 5% runtime — and reordering a dependency install above a source copy invalidates every later layer on each commit, which is how the layer becomes large. The "slow runtime" hypothesis fails because runtime time is unchanged between a warm and a cold start while pull time differs; the check compares a cached start to an uncached one.

**CI-23 — Throttled while idle-looking.** Mechanism: a CPU quota of 0.2 cores with a 100 ms period gives a 20 ms budget per period; a burst of 4 ms work every 20 ms consumes the budget exactly, so `nr_throttled/nr_periods` rises while the averaged usage reads 40%. The "slow code" hypothesis fails because the same code at the same rate runs faster with a larger quota; the counter ratio is the discriminator, not the usage percentage.

### 8.5 SQL analytics: three-valued logic, frame semantics, and fanout

The nine defects here are the unit's most reproduced errors because they are *silent*: a query with a NULL predicate or a wrong frame returns fewer rows or a different number, never an error. Every row's discriminator is a count, and counts are cheap. The first two are fanout variants that differ in where the duplicate originates.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verify |
|---|---|---|---|---|---|---|---|
| CI-24 | VIOLATED_ASSUMPTION | revenue 30% high after a join change | real growth; price change | `COUNT(*)` versus `COUNT(DISTINCT key)` | deduplicate the right side before aggregating | assert row count per join in tests | replay the query, assert 1.30 ratio gone |
| CI-25 | NUMERICAL_EDGE | 4% of rows vanish from an inner join | genuinely missing keys; filters | count NULL keys on both sides | use a left join or coalesce the key | profile NULL share on join keys | replay the query, assert row count |
| CI-26 | NUMERICAL_EDGE | running total jumps by a peer-group at a tie | duplicate events; bad ordering | compare `ROWS` and `RANGE` output | state the frame explicitly | forbid implicit frames in review | replay a tied key, assert equality |
| CI-27 | NUMERICAL_EDGE | `COUNT(col)` understates total rows | filters; join loss | compare `COUNT(*)` and `COUNT(col)` | use `COUNT(*)` for row counts | document nullability per column | replay on a 4%-NULL column, assert delta |
| CI-28 | NUMERICAL_EDGE | daily totals double-count a boundary row | late events; duplicate ids | rows with timestamp exactly at midnight | use half-open `>=` and `<` | lint for `BETWEEN` on timestamps | replay a midnight event, assert one count |
| CI-29 | NUMERICAL_EDGE | integer division gives a percentage of 3 | rounding; wrong numerator | recompute in floating point | cast before dividing | unit test the ratio expression | replay 7/2, assert 3.5 |
| CI-30 | CONCEPT_CONFUSION | dashboard p99 higher than any shard's p99 | slow shard; bad timer | recompute the percentile on pooled rows | pool samples, then take the percentile | ban averaging percentiles | replay three shards, assert pooled |
| CI-31 | SCALE_FAILURE | one key's window scans 45% of the table | missing index; small table | rows per key versus total | pre-aggregate or partition by key | alert on skew ratio | replay hot key, assert scan share |
| CI-32 | CONCEPT_CONFUSION | monthly rate differs from a 30-day rate by 3.2% | real seasonality; data gap | denominator days per period | normalise to a fixed unit | document the denominator per metric | replay both windows, assert ratio |

**CI-24 — Fanout from a non-unique join key.** Mechanism: joining orders to events on `(customer, day)` instead of `order_id` multiplies each order by its event count, so a `SUM` over 1,000 orders with 1,300 matching events inflates by `1.30`. It is `VIOLATED_ASSUMPTION` rather than `NUMERICAL_EDGE` because the arithmetic is right: the assumption that the right side is unique per left row is what failed. The "real growth" hypothesis is eliminated by the distinct-key count, which stays at 1,000.

**CI-25 — NULL keys dropped by an inner join.** Mechanism: 4% of rows carry a NULL join key; `NULL = NULL` evaluates to unknown, so an inner join drops them silently — 48 of 1,200 rows, and 3,360 currency units of a 84,000 total. The "missing keys" hypothesis fails a two-sided count: if the keys are genuinely absent from the right table, the left table's non-null keys are unmatched, whereas NULL-keyed rows are unmatched on both sides and countable in the left table alone.

**CI-26 — `RANGE` versus `ROWS` at a tie.** Mechanism: the default frame includes all *peers* of the current row, so with key values `[10, 10, 10, 20]` a running sum reports `30, 30, 30, 50` under `RANGE` and `10, 20, 30, 50` under `ROWS`. For the first row the difference is 20 units, a 100% error at the start of the window, and it is invisible in the total. The "duplicate events" hypothesis fails because duplicates change the row count, while peer inclusion leaves the count unchanged and changes only the frame membership.

**CI-27 — `COUNT(col)` is not a row count.** Mechanism: `COUNT(col)` counts non-NULL values, so a column with 4% NULLs reports 1,152 for a 1,200-row table — a 4% understatement that propagates into every per-row denominator. The "join loss" hypothesis fails because the discrepancy is exactly the NULL count, which is measurable independently with `COUNT(*) - COUNT(col)`.

**CI-28 — `BETWEEN` on a timestamp boundary.** Mechanism: `BETWEEN '2026-01-01' AND '2026-01-02'` includes `2026-01-02 00:00:00`, so the row at midnight is counted in both days — one duplicated row per day boundary, 365 per year on a daily-grain table. The check is a targeted count of rows exactly at midnight, which is one row per day when the defect exists and zero when the query is correct.

**CI-29 — Integer division in a ratio.** Mechanism: `successes / total` with both integers truncates, so `7/2` reports `3` instead of `3.5` — a 14.29% understatement — and the error is largest for small counts, which is exactly where a small sample needs the most care. The "wrong numerator" hypothesis fails because the numerator is verifiable by counting rows; the tell is that the result is always an integer.

**CI-30 — Percentiles averaged across shards.** Mechanism: with three shards of 100, 1,000 and 100 rows whose p99 values are 500 ms, 28.8 ms and 700 ms, the mean of per-shard p99 is 409.6 ms while the pooled p99 over all 1,200 rows is 700 ms — a 70.9% understatement. The "slow shard" hypothesis fails because the pooled value is computed from the same rows; the defect is the aggregation, not the data.

**CI-31 — Skew that defeats a per-key window.** Mechanism: 1% of keys hold 45% of rows, so 540 of 1,200 rows sit in twelve keys; a per-key window over the hot key scans 45% of the table while the median key scans a handful of rows. The "missing index" hypothesis fails because an index on the hot key does not reduce the rows the window must read — the frame itself is large.

**CI-32 — Denominators that differ by calendar length.** Mechanism: a monthly rate divides by 31 days in one period and 30 in the next, so a flat daily volume produces a 3.23% difference with no change in behaviour. The "seasonality" hypothesis fails a fixed-window recomputation: a 30-day rolling rate is identical across both months when the underlying daily volume is flat.

### 8.6 Category census and the distribution of the catalog

The census below exists because a catalog with a lopsided category distribution misleads: a responder who reads many `SCALE_FAILURE` entries learns to suspect volume, which is the wrong prior for a wrong number.

| Category | Count | Share | Most common layer | Typical discriminator |
|---|---|---|---|---|
| VIOLATED_ASSUMPTION | 7 | 21.9% | SQL and HTTP contracts | a count, ratio, or digest comparison |
| NUMERICAL_EDGE | 10 | 31.3% | SQL boundaries, LLM units | an invariant the reported number must obey |
| SCALE_FAILURE | 8 | 25.0% | async pools, containers, LLM context | a plotted cost curve against volume |
| CONCEPT_CONFUSION | 7 | 21.9% | LLM claims, percentiles, parallelism | a recomputation in an invariant unit |

**Reading the census.** `NUMERICAL_EDGE` is the largest family at 31.3% (10 of 32), which matches the observation that boundary semantics generate more silent defects than any other mechanism: `<` versus `<=`, `ROWS` versus `RANGE`, integer versus float, NULL versus empty. `CONCEPT_CONFUSION` is the smallest at 21.9%, but it carries the highest cost per instance, because a confused comparison produces a defensible-looking claim rather than a visibly wrong number.

**What the census does not say.** It is not a frequency estimate from any real system. The distribution reflects which defects are teachable with a decisive probe, not which are common. A production telemetry study would very likely show a different mix, dominated by configuration drift, and this file makes no claim about that mix.

**Cross-layer issues.** Four defects appear in more than one layer and are filed by first detectability, which is why the counts do not sum to the layer table sizes: duplicate effects (CI-07) also appear as a SQL count anomaly; censored loss (CI-15) also appears as an unexplained LLM spend drop; the fanout multiplier (CI-24) also appears as a retry-driven load figure; and grace-period loss (CI-20) also appears as a client timeout.
