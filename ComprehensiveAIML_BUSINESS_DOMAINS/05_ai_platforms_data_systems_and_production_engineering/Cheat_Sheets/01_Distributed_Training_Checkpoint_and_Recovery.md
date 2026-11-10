---
title: "Distributed Training Checkpoint and Recovery"
topic: "Checkpoint state, consistent publication, restore, elastic reshard, and storage economics for distributed training"
collection: "ai-platforms-data-systems-production-engineering"
level: "core"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Distributed Training Checkpoint and Recovery

## 1. File Header

**Outcome:** size, publish, validate, and restore a checkpoint that actually resumes training, and diagnose the
cases where a resume succeeds technically but silently changes the optimisation trajectory.

**Audience:** training-platform engineers who operate data-, tensor-, pipeline-, or optimizer-sharded jobs and
who are on call when a long run dies at hour nine of a forty-hour budget.

**Scope, inside:** the mutable state a restart consumes; the publication protocol that makes a generation
discoverable; the arithmetic of interval, cost, and lost work; reshard rules; storage tiers and their cost.

**Scope, outside:** model architecture, hyperparameter choice, scheduler policy beyond its state, and cluster
scheduler internals. Those are inputs to this sheet, not its subject.

**Evidence status:** every byte count, bandwidth, price, and interval below is constructed for teaching and is
labelled ILLUSTRATIVE. The arithmetic is derived here and re-checked by two independent routes; the numbers are
not measurements of any deployment, vendor, or published run.

**Source limitation:** the workspace directory inventory names
`finetuning_course/Module_19_Distributed_Training_Infrastructure/19.5_Fault_Tolerance_and_Elastic_Training.md`
and `finetuning_course/Module_14_MLOps_Practical_Workflows/14.3_Checkpointing_and_Fault_Tolerance.md`, but their
bodies are not present here. This sheet is therefore organised around mechanisms and derivations rather than a
summary of those files, and no framework behaviour is asserted as guaranteed for any particular release.

**Prerequisites:** global optimizer step versus microbatch; gradient accumulation; the difference between a
gradient buffer and an optimizer moment; distributed ranks and process groups; durable storage primitives.

**Companion artifacts:** `./02_25_Common_Issues.md` carries the diagnostic catalog that cites the identifiers
used here (CI-01 torn step, CI-04 weight export treated as a restart, CI-06 GiB versus GB). The unit topic map
is `../INDEX.md`; the training-scale material lives under `../24_distributed_training_data_pipelines_and_ml_compute/`.

**Notation:** plain text, one convention per file. `GiB = 2^30` bytes, `GB = 10^9` bytes, `TiB = 1024 GiB`.
Bandwidth is payload throughput in GiB/s unless a line says otherwise. Time is seconds unless marked.

**How to read this sheet:** sections 2 to 4 build the model, section 5 runs one number all the way through,
sections 6 and 7 turn it into a picture and a selection rule, sections 8 to 10 cover diagnosis, cost, and the
results worth memorising, sections 11 and 12 are for rehearsal.

**The one-sentence version:** a checkpoint is a published, validated, self-describing prefix of training, and
every property in that sentence is separately breakable.

## 2. Concept Map / Mental Model

```text
step k completes  ->  quiesce or snapshot at a safe point
                  ->  write immutable shards (rank-local or resharded)
                  ->  write manifest: names, sizes, hashes, schema, cursor
                  ->  validate manifest + referenced objects
                  ->  commit: publish pointer / latest marker
failure          ->  read pointer  ->  pick newest committed generation
                  ->  validate   ->  load   ->  rebuild topology   ->  resume
```

The diagram shows dependencies, not atomicity. No arrow in it is free, and three of them are failure surfaces:
the write (partial objects), the commit (two writers, one pointer), and the rebuild (topology change).

Three properties are routinely conflated and must be separated, because each has its own test.

**Logical consistency** asks whether the collected tensors and counters describe one prefix of training. A
generation can be perfectly durable and completely inconsistent: eight ranks wrote at eight different steps,
each write succeeded, and the manifest is honest about it.

**Physical durability** asks whether the bytes survive the loss of the node, disk, or zone that received them.
A consistent generation held only in page cache is neither durable nor recoverable, and a test that reads it
back from the same node cannot tell the difference.

**Discoverability** asks whether recovery can find the generation without knowing its name. A directory full of
perfectly valid generations with a broken or absent latest pointer is not a recovery point; it is an inventory.

A fourth property, **sufficiency**, is the one that separates a checkpoint from an export. Sufficiency asks
whether every mutable input to the next logical update is present: parameters, optimizer moments, step counter,
scheduler state, gradient-scaler history, RNG streams, accumulation phase, and the input cursor with its dataset
version. A weight export has the first and none of the rest, which is why it initialises well and resumes badly.

The mental model that survives contact with incidents is a **cut**, borrowed from distributed systems. A cut is
consistent when no message is recorded as received before it was sent. In a training loop the "messages" are the
gradient all-reduce and the parameter broadcast; a consistent cut is one taken while no such collective is in
flight. Practically this means the safe point is a completed optimizer update with no rank still inside a
backward pass. Anything else yields tensors from different updates, and the artifact will load without complaint.

Publication is a second cut: the set of objects must become visible to readers in one step, or a reader must be
able to distinguish a complete set from an incomplete one. Since most object stores offer only per-object
atomicity, the usual construction is immutable generation objects plus one small mutable pointer updated
conditionally, with the rule that readers never trust a generation the pointer has not named.

Two consequences follow immediately and are the source of most operational surprise.

**An old generation is worth more than a new one.** Publication should be strictly monotone and readers should
fall back rather than guess. A run that cannot publish generation 41 should resume from 40, not from a directory
that happens to have 41 shards of 64.

**Restore is not the inverse of write.** Writing is embarrassingly parallel across ranks because each rank owns
its shard. Restoring into a different world size requires a gather whose cost is governed by the interconnect,
not by the object store, and that asymmetry is where recovery-time objectives are usually lost.

The last element of the mental model is a written **recovery contract**, because the unwritten version differs
between the person who built the path and the person who is paged at 03:00. It has six lines and each line is
falsifiable by a drill: what counts as a committed generation; which generation is eligible after a partial
publication; the maximum tolerable loss in steps; the maximum tolerable restore in wall clock; who may declare a
generation unusable and how that decision is recorded; and what happens to the last uncommitted interval. Every
line is short, and every line has been the subject of an argument during an incident. Writing them converts a
disagreement about facts into a disagreement about policy, which is a much cheaper argument to have.

**Takeaway:** name which of the four properties a given test actually proves before trusting it.

## 3. Foundational Definitions

| Term | Meaning, with unit or type | Boundary or confusion |
|---|---|---|
| Generation | Immutable logical snapshot with a unique identifier, e.g. `gen-000041` | Not a directory listing; directories can be partial |
| Commit | The act of making a generation eligible for recovery, usually one conditional pointer write | A successful shard write is not a commit |
| Manifest | Immutable metadata: expected object names, byte sizes, digests, schema version, step | Not a substitute for reading the objects it names |
| Global optimizer step | Integer count of completed parameter updates | Not a microbatch count; not a forward pass count |
| Microbatch | Input subset for one forward/backward pass | May not trigger an update under accumulation |
| Accumulation phase | Integer count of completed microbatches inside the current update, range `0..k-1` | Must be saved, or reset only at an update boundary |
| Rank | Process identity inside a process group, `0..N-1` | Identity is not stable across elastic restarts |
| World size | Count of participating processes in a group | Tensor, data, and pipeline groups can differ |
| Shard | Physical partition of model or optimizer state on one rank | A rank's shard is not necessarily reusable by that rank |
| Safe point | Instant with no collective in flight and a well-defined step boundary | "Between two Python lines" is not automatically a safe point |
| Weight export | Parameters in inference format, no optimizer or cursor state | Sufficient for serving; insufficient for resumption |
| RPO | Recovery point objective: tolerable lost progress in time, steps, or samples | An objective, not a measured worst case |
| RTO | Recovery time objective: tolerable elapsed time until useful work resumes | Composition matters more than the target number |
| RNG state | Generator states for CPU, accelerator, and worker streams, including counter position | A seed is not a state; it is the state's constructor |
| Global sample cursor | Next logical training position in a versioned sampling plan | A prefetch cursor is not consumed progress |
| Effective bandwidth | Payload bytes completed per elapsed second for the stated workload | Link peak, or a sum of per-rank rates, is not effective bandwidth |
| Write amplification | Bytes written to storage per byte of unique state produced | Erasure coding, replication, and re-slicing all inflate it |
| Delta checkpoint | A generation containing only state that changed since a named base | Its restore cost includes the whole chain |
| Base chain | Ordered list of generations a delta depends on | An unbounded chain converts storage savings into RTO |
| Reshard | Loading a generation written at one world size into another | Supported only if sharding is expressed by identity |
| Retention | Policy that decides which generations remain readable | Deleting a base breaks every delta above it |
| Drill | Deliberate fault injection against the checkpoint path with recorded timing | A restore nobody timed is a hypothesis |
| Goodput | Fraction of wall-clock time spent on training that will survive a failure | Distinct from hardware utilisation |
| Checkpoint cost, C | Elapsed seconds of training time consumed to produce and publish one generation | Includes stalls, not just transfer |

Three definitions deserve a warning because they are the ones operators get wrong under pressure.

A **step number** has at least three conventions in circulation: microbatches consumed, optimizer updates
completed, and scheduled steps including skipped ones. A checkpoint that records "step 12000" without saying
which convention is a latent off-by-a-thousand. The safe practice is to store the update count, the microbatch
count, and the cursor together, and to treat disagreement between them as a defect.

**Effective bandwidth** is measured, never summed. Eight ranks each reporting 1.0 GiB/s against a shared
600 Gbit/s backend have not demonstrated 8 GiB/s; they have demonstrated contention. The number that predicts
checkpoint cost is the aggregate payload rate observed during a real write, measured at the store.

**RPO** is a promise about loss, and it must be expressed in the unit the business cares about. "Twenty
minutes" is 20 minutes of GPU time, which is a different quantity at 512 accelerators than at 8. Converting to
samples or tokens is what makes an RPO comparable to a data-reprocessing cost.

**Takeaway:** every counter in a checkpoint needs a stated convention, and every bandwidth needs a measurement.

Two further definitions are needed before the formulas, because they decide whether a restore is idempotent.

A **consumption cursor** is the logical position of the *last fully consumed* sample, not the position the
loader has read. With `p` prefetch workers and a per-worker prefetch depth `d`, the loader may be `p * d` batches
ahead of consumption. A checkpoint that stores the loader's position therefore skips `p * d` batches on resume.
With 8 workers and depth 4, that is 32 batches — at a batch of 1,024 sequences and 4,096 tokens per sequence,
32 * 1,024 * 4,096 = 134.2 million tokens never trained on. That is not a rounding error; it is a data hole
whose size is a configuration parameter nobody set deliberately.

A **training-time side effect** is any external state the loop mutates outside its tensors: a counter pushed to
a metrics service, a sample marked as used in a feature store, a message published to a queue. These are not
captured by any tensor snapshot, so a restore replays the interval while the external world has already
advanced. The rule that keeps this tractable is that side effects must be keyed by `(run_id, step)` and be
idempotent on that key, which turns "exactly once" from an impossible requirement into a deduplication on read.

## 4. Core Formulas with Full Derivation

Convention for this section: each result is stated as formula, where it applies, what must hold, one numeric
instance, and an interpretation. Numbers are ILLUSTRATIVE and re-derived in section 5.

### 4.1 Base case

The base case is one writer, one durable store, immutable objects, and a single conditional pointer write. The
only question it answers is how long the state transfer takes, which sets the floor for every later model.

```text
FORMULA [F1]:  C = S / B          and          s_r = S / N_r
WHERE: C is checkpoint cost in seconds; S is total state bytes; B is aggregate effective throughput in bytes/s;
       s_r is per-rank payload in bytes; N_r is the number of ranks that each hold a disjoint shard.
CONSTRAINT: shards are disjoint and roughly equal; the write is bandwidth-bound, not request-bound;
            no rank is idle waiting for the slowest one by more than a small factor.
NUMERICAL EXAMPLE: S = 980e9 B, B = 8 GiB/s = 8 * 2^30 = 8.590e9 B/s, so C = 980e9 / 8.590e9 = 114.1 s.
       With N_r = 512, s_r = 980e9 / 512 = 1.914e9 B = 1.783 GiB per rank.
INTERPRETATION: cost is linear in state and inverse in achieved throughput; halving S halves C exactly, while
       doubling B halves C only if the store and the fabric both scale, which they usually do not.
```

Two corrections to F1 matter in practice and are the same correction twice: the maximum replaces the mean.

```text
FORMULA [F1b]:  C = max over r of ( s_r / b_r )      and      C >= N_objects / q
WHERE: b_r is rank r's achieved write rate; N_objects is the number of objects written; q is the store's
       sustained request rate in requests/s.
CONSTRAINT: s_r and b_r are measured for the same window; q is measured, not the vendor's burst figure.
NUMERICAL EXAMPLE: 512 ranks of 1.783 GiB at 16 MiB/s each gives 1.783 * 1024 / 16 = 114.1 s, matching F1.
       If one rank achieves 4 MiB/s instead, that rank alone needs 456 s and the whole step waits 456 s.
INTERPRETATION: the mean is a description of the fabric; the maximum is the delay the optimizer actually pays.
```

The request-rate term is the reason small-shard schemes disappoint. A 512-shard checkpoint at one object per
shard is 512 requests, which no store notices. An 8,192-shard optimizer checkpoint at 24 requests per shard is
196,608 requests; at a sustained 2,000 requests/s that is 98 s of pure request overhead before a byte is
counted, which is comparable to the transfer itself.

### 4.2 General case

The general case adds failure and recovery to the transfer, and produces the only formula most teams actually
need: how often to checkpoint.

```text
FORMULA [F2]:  phi(T) = C/T + (T/2 + R)/M
WHERE: phi is the fraction of wall-clock time lost to checkpointing and to replay; T is the interval between
       commits in seconds; C is cost per commit from F1; R is the mean restart time in seconds (detect +
       schedule + fetch + load + warm-up); M is the mean time between failures *for the whole run* in seconds.
CONSTRAINT: failures arrive with a constant hazard 1/M; a failure inside a cycle loses on average T/2 of work
       plus R of restart; the checkpoint itself is not interrupted by the failure it is racing.
NUMERICAL EXAMPLE: C = 114.1 s, R = 572 s, M = 10800 s, T = 1570 s:
       phi = 114.1/1570 + (785 + 572)/10800 = 0.0727 + 0.1256 = 0.1983, i.e. 19.8% of the run is not progress.
INTERPRETATION: the two terms trade off. Frequent commits raise the first term; rare commits raise the second.
```

Differentiating gives the interval that minimises the approximate overhead.

```text
FORMULA [F3]:  T* = sqrt(2 * C * M)          with      phi(T*) ≈ 2 * sqrt(C / (2 M)) = sqrt(2 C / M)
WHERE: T* is the optimal interval in seconds; symbols as in F2; R does not appear because it is additive and
       constant in T, so it shifts the floor of phi but not the location of the minimum.
CONSTRAINT: the same constant-hazard assumption as F2, plus C << M so that at most a handful of commits are
       lost per failure; the approximation drops terms of order (C/M) and (T/M)^2.
NUMERICAL EXAMPLE: T* = sqrt(2 * 114.1 * 10800) = sqrt(2.465e6) = 1570 s = 26.2 min.
       Check against F2 at T = 600 and T = 3600: phi = 0.271 and 0.251, both above 0.198.
INTERPRETATION: the optimum is flat, not sharp — 0.5 percentage points separate 20 minutes from the optimum.
```

The exact statement of the optimum, which is the result usually attributed to Young and Daly, minimises the
same ratio without the small-term approximation.

```text
FORMULA [F3b]:  minimise  (C + T/2 + R) / (T + C)      =>      T* = sqrt(2 * C * (M + R)) - C
WHERE: the numerator is expected non-progress time per cycle plus restart; the denominator is useful work.
CONSTRAINT: R and C are treated as constants; the failure process is memoryless so the age of the current
       interval does not inform the next failure.
NUMERICAL EXAMPLE: sqrt(2 * 114.1 * 11372) - 114.1 = sqrt(2.595e6) - 114.1 = 1611 - 114 = 1497 s.
INTERPRETATION: the exact form lands 5% from the simple form; either is adequate because the valley is flat.
```

The flatness is the operationally important part, and it is worth a third route so that nobody has to trust
algebra under pressure.

```text
FORMULA [F3c]:  T* = 3.6 * sqrt(C)      (minutes, when M = 3 h)
WHERE: T* in minutes, C in seconds, valid only for M = 10800 s; derived by substituting sqrt(2*10800) = 147.0
       and dividing by 60: 147/60 = 2.45, then multiplying by sqrt(C) gives minutes.
CONSTRAINT: M fixed at 3 h and C in seconds; a different M rescales by sqrt(M/10800).
NUMERICAL EXAMPLE: C = 114 s: T* = 3.6 * 10.68 = 38 min? No — recompute: 2.45 * 10.68 = 26.2 min, so the
       constant is 2.45, not 3.6. The corrected rule is T* = 2.45 * sqrt(C) minutes.
INTERPRETATION: a hand rule is only safe if the constant is re-derived; the mis-stated 3.6 would over-wait by 45%.
```

That last block is deliberately kept with its failed first constant, because the failure is the lesson: a
memorised constant is a formula with an invisible validity condition, and the check that catches it takes ten
seconds of arithmetic.

### 4.3 Variants

Four variants cover nearly every real schedule: restart time that is composed rather than measured, shard
counts large enough to hit request limits, delta chains, and replicated durability.

```text
FORMULA [F4]:  R = t_detect + t_sched + t_fetch + t_load + t_warm      (overlap only where stated)
WHERE: t_detect is failure detection latency; t_sched is queue plus image pull plus container start;
       t_fetch is the transfer from the store to the node; t_load is deserialise plus reshard plus all-gather;
       t_warm is compile, capture, and first-step latency before steady-state throughput.
CONSTRAINT: fetch and load are serial for the same bytes unless the loader can stream; t_warm cannot begin
       before parameters exist; detection cannot be zero because health checks take time to notice.
NUMERICAL EXAMPLE: 30 + 60 + 152 + 240 + 90 = 572 s. Note t_load (240 s) exceeds t_fetch (152 s), so buying a
       faster store cannot bring R below 420 s (30+60+240+90) — the other terms dominate any storage upgrade.
INTERPRETATION: optimise the largest addend, and check whether it is even addressable by the intended change.
```

```text
FORMULA [F5]:  C = max( S/B , N_obj/q )      and      N_obj = N_shards * k
WHERE: N_shards is the number of state partitions; k is objects per partition (data file, index, metadata).
CONSTRAINT: the store's request concurrency limit is shared with readers and other jobs; q is the fair share.
NUMERICAL EXAMPLE: S = 980e9 B at B = 8 GiB/s needs 114.1 s. With N_shards = 8192, k = 3, q_fair = 1500/s:
       N_obj = 24576, N_obj/q = 16.4 s, so the write stays bandwidth-bound with 7x headroom on requests.
       At k = 20 and 32768 shards: 655360 objects / 1500 = 437 s, and the checkpoint becomes request-bound.
INTERPRETATION: shard count is chosen for reshard flexibility and memory, but it sets a hidden request budget.
```

```text
FORMULA [F6]:  bytes_chain(n) = S + n * (1 - f) * S        restore(n) = bytes_chain(n) / B_read
WHERE: n is the number of deltas above the base; f is the fraction of state that is frozen or unchanged
       (0 <= f < 1); B_read is restore throughput.
CONSTRAINT: deltas apply in order; the base must remain readable; f is stable across the chain, which is false
       when the frozen set changes (unfreezing an embedding table invalidates the estimate).
NUMERICAL EXAMPLE: S = 912.7 GiB, f = 0.30, n = 7: chain = 912.7 + 7 * 638.9 = 5385 GiB versus 8 full
       generations at 7301.6 GiB, a 26% storage saving. Restore of the last delta = 5385 / 6 = 897 s, not 152 s.
INTERPRETATION: incremental checkpoints trade storage for RTO at roughly 6x; bound n explicitly.
```

```text
FORMULA [F7]:  P(lose all copies) = p^r      only if failures are independent across copies
WHERE: p is the per-copy annual loss probability; r is the number of replicas or the erasure-code tolerance + 1.
CONSTRAINT: independence. Copies in one account, one region, one credential scope, or one bucket policy are
       not independent, so the correct model is p, not p^r.
NUMERICAL EXAMPLE: p = 0.02 per copy-year, r = 3: 0.02^3 = 8e-6 per year under independence — but a single
       credential revocation with r = 3 dependent copies has probability 0.02, i.e. 2,500x higher.
INTERPRETATION: replica count buys durability against hardware, not against operational or policy failure.
```

### 4.4 Limits and counterexamples

**Limit 1: T tending to zero.** As T shrinks, C/T grows without bound while the loss term shrinks. The model
predicts phi > 1 for T < C, which is arithmetic honesty: committing faster than the commit costs means the run
never trains. The practical floor is that the checkpoint window must be a small fraction of the interval,
conventionally below 15 percent.

**Limit 2: correlated failure.** The uniform-hazard assumption fails when a single event removes many workers.
A rack-level power loss that kills 64 of 512 ranks is one event with 64 casualties; the loss is still
approximately T/2, but the restart now includes waiting for 64 new nodes, which inflates R and can exceed M.
When R approaches M, F2's denominator model breaks and no interval restores goodput.

**Limit 3: the run-level MTBF shrinks with scale.** Per-node MTBF is not run MTBF. With independent nodes and
per-node MTBF `M_n`, a run over `n` nodes has `M_run = M_n / n`. At `M_n` = 30 days and n = 512, `M_run` =
720 h / 512 = 1.41 h = 5062 s, not 30 days. Re-running F3 with M = 5062 gives T* = sqrt(2*114.09*5062) = 1075 s
= 17.9 min and phi(1075) = 0.1061 + 0.1119 = 21.8%? No — recompute with R = 572: phi(1075) = 114.09/1075 +
(537.5 + 572)/5062 = 0.1061 + 0.2192 = 32.5%. The error in that first attempt is dropping R, which is 11.3% of
M at this scale and only 5.3% at M = 10800; the restart term does not shrink with the cluster, so it becomes the
dominant term as the run gets larger.

The operational consequence is sharper than the arithmetic: at M = 5062 the floor of phi is `R/M` = 11.3%
before any training-time cost is counted, so no interval choice can get overhead below about 11%. Keeping the
26-minute interval derived from the wrong M gives 34.1%, so the correct interval is worth 1.6 points — small in
isolation, but it also halves the worst-case replay window, which is the number the RPO promised.

**Limit 4: metadata-dominated cost.** F1 assumes the payload dominates. For a run whose state is mostly a giant
sparse embedding table written as millions of tiny rows, the fixed per-object cost dominates the transfer and the
formula understates C by an order of magnitude. The diagnostic is the ratio of object count to bytes: below
about 1 MiB per object, assume request-bound.

**Limit 5: the formula cannot price the thing that actually hurts.** Lost work is not uniformly painful. Losing
the last 13 minutes before a scheduled evaluation wastes the evaluation; losing 13 minutes at hour two of forty
is noise. A schedule with a checkpoint pinned before every irreversible downstream step beats a schedule chosen
purely by F3, and the difference is invisible in phi.

**Validity condition for the whole section:** constant hazard, measured C and R, one run, one storage tier,
failures independent of progress, and no irreversible external side effects. Every one of those has been false
in some real incident, so treat phi as a planning number and never as an availability guarantee.

## 5. Worked Numerical Example

ILLUSTRATIVE scenario: a 70e9-parameter dense model trained on 512 accelerators across 64 nodes, 8 ranks per
node, bf16 compute with an fp32 master copy, Adam with two fp32 moments, fully sharded optimizer state.

### 5.1 State inventory

| Component | Per-parameter bytes | Total bytes | GiB | Share |
|---|---:|---:|---:|---:|
| bf16 parameters | 2 | 140e9 | 130.4 | 14.3% |
| fp32 master copy | 4 | 280e9 | 260.8 | 28.6% |
| Adam first moment | 4 | 280e9 | 260.8 | 28.6% |
| Adam second moment | 4 | 280e9 | 260.8 | 28.6% |
| Scalars, RNG, cursor, schema | — | < 1e6 | 0.001 | <0.1% |
| Total | 14 | 980e9 | 912.7 | 100% |

Check the arithmetic two ways. Per parameter: 2 + 4 + 4 + 4 = 14 bytes, so 14 * 70e9 = 980e9 bytes. In GiB:
980e9 / 2^30 = 980e9 / 1.0737e9 = 912.7 GiB. In decimal GB the same state is 980.0 GB — the 7.4% gap between
those two numbers is exactly CI-06, and it is large enough to change a capacity decision near a quota.

The structural fact worth memorising: optimizer moments are 57.1% of state (2 * 260.8 / 912.7). Any scheme
that skips them saves the majority of bytes and destroys resumption, which is why a checkpoint that is "small
and fast" deserves suspicion before praise.

### 5.2 Primary calculation

Write at an achieved 8 GiB/s aggregate: C = 912.7 / 8 = 114.09 s. Restore at 6 GiB/s (readers are usually
slower than writers because of fan-in and validation): 912.7 / 6 = 152.12 s.

Compose the restart time: detect 30 s, schedule and image 60 s, fetch 152 s, load and reshard 240 s, warm-up
90 s, total R = 572 s. The load term exceeds the fetch term, so a faster store would move less than half the
total.

Take M = 10800 s (three hours, the run-level figure for a mid-size cluster) and compute the optimum:
T* = sqrt(2 * 114.09 * 10800) = sqrt(2.464e6) = 1569.8 s = 26.2 min. At that interval:
phi = 114.09/1569.8 + (784.9 + 572)/10800 = 0.0727 + 0.1256 = 0.1983.

So about one hour in five is not spent advancing the model. That is the number to compare against the cost of
any proposed fix, and it decomposes cleanly: 0.073 of it is writing, 0.126 of it is replay and restart.

### 5.3 Independent re-check, by two other routes

**Route A, per-hour accounting.** Failures per hour = 3600/10800 = 0.3333. Lost work per failure = T/2 + R =
784.9 + 572 = 1356.9 s. Loss per hour = 0.3333 * 1356.9 = 452.3 s. Commits per hour = 3600/1569.8 = 2.293;
write time per hour = 2.293 * 114.09 = 261.6 s. Total non-progress = 452.3 + 261.6 = 713.9 s of 3600 = 19.83%.
Same answer, different decomposition, no shared algebra.

**Route B, per-parameter cost.** Cost per commit per parameter = 14 bytes / (8 * 2^30 bytes/s) = 1.6297e-9 s.
Times 70e9 parameters = 114.08 s per commit. This confirms C independently of the GiB conversion, and it also
shows the sensitivity: dropping the fp32 master copy (12 bytes instead of 14) would cut C by 14.3% to 97.8 s.

Three routes agreeing to three significant figures is the standard this sheet holds itself to; a single route
with a unit conversion is where CI-06 lives.

### 5.4 Sensitivity

| Interval T (s) | Write term C/T | Loss term (T/2+R)/M | Total phi | Verdict |
|---:|---:|---:|---:|---|
| 300 | 38.03% | 6.69% | 44.72% | catastrophic, commits faster than they pay |
| 600 | 19.02% | 8.07% | 27.09% | too frequent |
| 1200 | 9.51% | 10.85% | 20.36% | near-optimal, half the exposure |
| 1570 | 7.27% | 12.56% | 19.83% | optimum under stated assumptions |
| 1800 | 6.34% | 13.63% | 19.97% | indistinguishable from optimum |
| 2700 | 4.23% | 17.80% | 22.02% | 2.2 points worse, doubles RPO |
| 3600 | 3.17% | 21.96% | 25.13% | one-hour interval costs 5.3 points |

Read the table as a shape, not a ranking. Between 1200 s and 1800 s the total varies by 0.5 percentage points,
which is smaller than the error in either C or R. Choosing 1200 s instead of the computed optimum in exchange
for a 20-minute RPO is therefore rational and should be recorded as a deliberate decision, not a deviation.

Now change R and watch the location move. With R = 300 s: T* = sqrt(2*114.09*10800) = 1570 s, unchanged, but
phi(1570) = 0.0727 + (784.9+300)/10800 = 0.0727 + 0.1005 = 17.3%. Halving restart time is worth 2.5 points at
the same interval. Halving C (to 57 s) moves T* to sqrt(2*57*10800) = 1110 s and phi to 0.0514 + (555+572)/10800
= 0.0514 + 0.1044 = 15.6%. So the two levers are not interchangeable: C improvement shifts the schedule and pays
twice, R improvement lowers the floor without shifting the interval.

### 5.5 What the example establishes, and what it does not

It establishes that the model is internally consistent across three routes, that the interval choice is
insensitive within a factor of 1.5, that optimizer state dominates bytes, and that load and warm-up dominate
restart time rather than transfer.

It does not establish that any particular framework can restart in 572 s, nor that 8 GiB/s is achievable on any
specific fabric, nor that a three-hour MTBF applies to another cluster. The example is a template: substitute
measured C, R, and M, and the same arithmetic gives an answer for the real system.

## 6. Visual / Geometric Intuition

```text
wall clock ->
[==== train 1569.8 s ====][ckpt 114 s][== train ==][ X failure ]
                                                     |
                                                     +-- replay from last commit
```

The picture to hold in mind is a sawtooth where the vertical drop is `u` (time since the last commit) plus `R`.

```text
progress
  ^        /|        /|         /|
  |      /  |      /  |       /  |
  |    /    |    /    |     /    |
  |  /      |  /      |   /      |
  +----------------------------------> time
      114   114        114              each vertical face is a commit, each drop is R + u
```

Two geometric readings follow. The area under the sawtooth divided by the total area is goodput, and the
sawtooth's vertical faces are not the expensive part unless they are dense — which is why the loss term wins at
long intervals and the write term wins at short ones.

The overhead curve is a shallow valley, not a spike:

```text
phi
45% |*
27% |  *            <- 600 s
20% |     *  *  *   <- 1200, 1570, 1800 s
22% |           *
25% |              *  <- 3600 s
    +------------------> T (log scale, 300..3600 s)
```

The flat floor of that valley is the single most useful operational fact in this sheet. It means a schedule
chosen for reasons other than phi — a checkpoint before evaluation, before a data-shard boundary, before a
manual intervention — costs almost nothing, and a schedule defended as "optimal" to four significant figures is
false precision.

A third picture explains why restore feels disproportionate to write. Writing is a fan-out with each rank
owning a disjoint shard; the store aggregates the streams. Reading into a different world size is a fan-in
followed by a gather, so the same bytes cost more time because the fabric, not the store, becomes the limit.

```text
write:   rank 0..511  --> [store]                 (fan-out, disjoint, B_store limited)
restore: [store] --> rank 0..511 --> all-gather   (fan-in, then B_fabric limited)
```

The memory picture is the fourth. A checkpoint needs a staging buffer unless the writer can stream from device
memory, and the staging choice decides whether a commit stalls the optimizer.

```text
device memory:  [ params | grads | moments | activations | workspace | free ]
                                    ^ if free < staging, the writer must serialise with the step
host memory:    [ pinned staging | page cache | ... ]
```

The ceiling is not the sum of tensors; it is the sum plus the largest transient the serialiser allocates. A run
that fits with 30 GiB spare and then OOMs at commit time is not misconfigured, it is measuring the transient.

**Takeaway:** the shape is a shallow valley plus a sawtooth; both are dominated by terms other than transfer.

## 7. Comparison / Selection Table

### 7.1 Checkpoint strategies

| Strategy | State captured | Publication unit | RTO driver | Storage per generation | Reshard | Typical misuse |
|---|---|---|---|---|---|---|
| Full synchronous, rank-local shards | Everything | All shards then pointer | Fetch + load | S | Requires identity-based mapping | Assuming the pointer write is atomic without a conditional |
| Full synchronous, resharded writer | Everything, already repartitioned | Coordinator-written set | Fetch + load, less gather | S | Built in | Coordinator becomes a serialisation bottleneck |
| Async staged (device to host to store) | Everything, snapshot at safe point | Pointer after upload completes | Fetch + load | S | Same as writer | Training continues and mutates the buffer being uploaded |
| Incremental delta | Changed state only | Base plus delta list | Whole chain read | (1-f)S per delta | Same as base | Unbounded chain discovered only during an incident |
| Weight export | Parameters only | Single file | Load only | 0.14S for this model | Not applicable | Presented as a checkpoint |
| Optimizer-state-only | Moments and master weights | Shard set | Fetch + load | 0.86S | Same as base | Useless alone; needs a matching parameter set |

The column that decides most arguments is RTO driver, not size. A delta scheme that halves storage and
quintuples restore time has traded the objective the business actually named for one it did not.

### 7.2 Storage tiers

| Tier | Effective write rate | Effective read rate | Cost shape | Failure mode | Use |
|---|---|---|---|---|---|
| Node-local NVMe | 3-7 GiB/s per node | 3-7 GiB/s per node | Included in node price | Dies with the node; invisible to recovery | Fast staging with async upload only |
| Shared POSIX filesystem | 1-4 GiB/s aggregate | 1-4 GiB/s | Provisioned capacity | Metadata server contention at high object counts | Small clusters, single-generation retention |
| Object store, standard | 5-12 GiB/s aggregate | 5-12 GiB/s | Per GB-month plus requests | Request throttling; eventual listing consistency | Default for large runs |
| Object store, infrequent access | Same write | Same read, higher latency | Cheaper storage, retrieval fee | Surprise retrieval cost during a restore storm | Generations older than the last two |
| Cold archive | Minutes to hours to restore | Hours | Cheapest storage | Restore exceeds RTO by design | Compliance retention only |

### 7.3 Retention policy

| Generations kept | Storage for this model | Restore choice | Risk |
|---|---:|---|---|
| 1 | 912.7 GiB | None if the only copy is corrupt | No fallback |
| 2 | 1.78 TiB | Previous generation | Minimum defensible for a long run |
| 4 | 3.57 TiB | Three steps back | Covers a bad data batch discovered late |
| 8 | 7.13 TiB | Seven steps back | Covers a bad config discovered across a day |
| 8 plus cold archive | 7.13 TiB hot | Any, slowly | Compliance without hot cost |

At an ILLUSTRATIVE $0.023 per GB-month on the decimal definition, one generation costs 980 * 0.023 = $22.54 per
month; eight cost $180.32; twenty concurrent runs cost $3,606 per month. Under the GiB reading the same line is
$20.99, and the 7% difference compounds across a fleet — which is why the unit must be written into the policy.

**Takeaway:** choose the strategy by the RTO and RPO it delivers, then the tier by measured rates, then the
retention by the recovery story you can actually execute.

## 8. Failure Modes & Edge Cases

Each entry gives the mechanism, competing causes that produce the same symptom, the check that separates them,
and the correction that survives the next incident.

### 8.1 Torn step: the generation mixes two updates

Mechanism: the writer reads parameter buffers while rank 3 is still inside the backward pass, so its shard
reflects update k while rank 0's reflects k+1. Every object is internally valid and every digest matches.

Competing causes: (a) no barrier before the read; (b) a barrier that is per-group and misses the pipeline
stage group; (c) asynchronous checkpointing that snapshots a buffer the optimizer then mutates.

Discriminating check: record the step counter inside each shard's metadata and compare across shards. A torn
step shows two or more distinct counters; a genuine single-step generation shows one. This is cheap and
conclusive, which is why it belongs in the manifest rather than in a log line.

Correction: insert an explicit global barrier and re-take the generation; then re-run two steps from it and
compare loss against the pre-failure trace.

Prevention: make the step counter part of the object name, so a torn set produces two directory names that
cannot be assembled into one manifest.

### 8.2 Mixed-generation shards in one manifest

Mechanism: a retry writes generation 41's shard 12 over generation 40's shard 12 because the writer reused a
path template with a stale variable. The manifest for 40 now names a shard from 41.

Competing causes: (a) non-immutable paths; (b) a retry that recomputed the path from a mutable global;
(c) a copy step that overwrote rather than created.

Discriminating check: compare the manifest's recorded digest for shard 12 against the stored object's digest.
Mismatch confirms overwrite; if digests match but the step counters differ, the manifest itself was assembled
from the wrong generation.

Correction: restore the previous generation and rebuild the writer path as a pure function of
`(run_id, generation, shard_id)`.

Prevention: write-once semantics at the store level, so an overwrite attempt fails loudly instead of silently.

### 8.3 The pointer moves but the objects are not durable

Mechanism: the writer completes uploads into a caching layer, publishes, and the cache evicts before the bytes
reach durable storage. Recovery finds a pointer to objects that no longer exist.

Competing causes: (a) upload completion reported before durability; (b) a bucket lifecycle rule that expires
objects early; (c) an eventual-consistency read that returned stale listing data at publish time.

Discriminating check: after publishing, list the manifest's objects from a fresh client with no cache, and
compare sizes as well as existence. Repeat after a short delay to separate propagation from durability.

Correction: require the store's own durability acknowledgement before the pointer write, and treat a listing
mismatch as a failed publication.

Prevention: a publication gate that refuses the pointer unless the manifest's object count and byte total
validate from an independent client.

### 8.4 Concurrent coordinators publish different generations

Mechanism: a preemption retry starts a new coordinator while the old one is still alive, both write, and the
pointer lands on the older generation. Training then resumes from an earlier point than the operator believes.

Competing causes: (a) no fencing token on the pointer write; (b) two runs sharing a prefix; (c) clock skew
making the older generation appear newer.

Discriminating check: the pointer's conditional write must carry the expected previous value. Read the
pointer's version history: a losing writer will have attempted a conditional write that failed, and that
attempt is visible in store logs even though no data changed.

Correction: use compare-and-set on the pointer, and make the generation identifier monotone so a stale writer's
value is rejected rather than accepted-by-timestamp.

Prevention: one writer identity per run, expressed as a lease, with the pointer conditional on the lease holder.

### 8.5 Resume succeeds but the loss jumps

Mechanism: the checkpoint omitted a state element — most often the gradient scaler's overflow history, the
learning-rate scheduler's internal counter, or the accumulation phase — so the first steps after resume use
different effective dynamics.

Competing causes: (a) genuinely missing state; (b) scheduler restored but stepped once at load; (c) data order
changed because the sampler was reseeded, so the loss difference is data, not state.

Discriminating check: run the resumed job for zero optimizer steps and compare the first forward loss against
the recorded loss at the checkpointed step on the same batch. A pure state gap shows a discontinuity in loss,
not in the batch. If the batch differs, fix the cursor first and re-measure.

Correction: restore the missing element and replay from the generation before the failure.

Prevention: a resume test that asserts the first-step loss is within a stated tolerance of the recorded value,
run in CI on a tiny model where the whole path executes in seconds.

### 8.6 Restore is fast but the run is slower afterwards

Mechanism: the generation loaded into a different world size produced a different reduction order, so the
numerics changed slightly; or the shard mapping preserved tensor shapes but changed which parameters share a
rank, breaking a fused kernel's assumption.

Competing causes: (a) legitimate numerics change from a new topology; (b) a kernel falling back to a slow path
because a shard is no longer contiguous; (c) one rank carrying a disproportionate share of the state.

Discriminating check: compare per-rank state bytes after reshard. Unequal bytes with equal tensor counts points
to (c); equal bytes with lower step throughput points to (b); neither, with a small loss offset that decays,
points to (a) and is acceptable.

Correction: constrain reshard to topology changes that keep shards balanced, or accept and document the numeric
change.

Prevention: record per-rank byte totals in the manifest so the imbalance is visible before the load finishes.

### 8.7 The checkpoint window stalls the fabric for other tenants

Mechanism: a 512-rank write at 8 GiB/s saturates a shared backend, so an unrelated inference service sees p99
latency triple for two minutes. The training job is behaving correctly and is still the cause.

Competing causes: (a) no bandwidth reservation; (b) burst credits exhausted by a previous write; (c) an
unrelated job's traffic coinciding.

Discriminating check: correlate the service's latency spike with the write window using byte counters per
tenant. A coincident-only relationship shows no per-tenant byte increase; a causal one shows the training
tenant consuming the shared pool.

Correction: rate-limit the checkpoint writer or move it to a dedicated endpoint; re-measure both workloads.

Prevention: schedule commits with a token bucket whose rate is derived from the tenant's fair share, and alert
on the checkpoint writer's share rather than on total bytes.

### 8.8 Restore time exceeds the objective and nobody noticed

Mechanism: RTO was defined as "time to start loading", while the business meant "time to useful throughput".
Compile, graph capture, and warm-up add minutes that the measurement excluded.

Competing causes: (a) definition gap; (b) a cold node image that must be pulled during recovery; (c) the load
itself regressed because the object count grew.

Discriminating check: time each phase separately during a drill and compare with the objective's wording. If
the phases sum above the objective but the measured metric is below it, the metric is measuring the wrong thing.

Correction: restate RTO as the sum from failure detection to steady-state throughput, and measure it end to end.

Prevention: an automated drill at the retention interval that fails the build when the measured RTO exceeds the
stated objective.

### 8.9 A drill passes because it uses the wrong generation

Mechanism: the drill restores the newest generation in a test bucket, which is always the most recently written
and therefore the one most likely to be cached. Real recovery reads a generation written hours ago, from a
different path, possibly after a lifecycle transition.

Competing causes: (a) cache warmth; (b) a lifecycle rule that moved the older object to a slower tier;
(c) permissions differing between the drill identity and the production identity.

Discriminating check: run the drill against the second-newest generation and with the production identity. A
pass on the newest with the test identity proves almost nothing about either.

Correction: fix the drill, not the objective; then re-baseline the RTO number.

Prevention: the drill selects its generation by policy (oldest retained, or a random one from the last 24 h),
never by "whatever is newest".

### 8.10 Retention deletes a base that deltas depend on

Mechanism: the retention rule keeps the newest eight generations by creation time. A delta chain's base is nine
generations old, so it is deleted while seven deltas still reference it, and every one of them becomes
unrestorable.

Competing causes: (a) count-based retention on a chain; (b) a manual cleanup; (c) a delta that silently
promoted itself to a base without recording the new chain.

Discriminating check: resolve the dependency closure of every retained generation and compare with the retained
set. A closure that references a deleted object is the defect, and it is computable before any incident.

Correction: restore the base from an archive if one exists; if not, treat the chain as lost and record which
generations became unrestorable.

Prevention: retention operates on chain closure, never on individual objects, and refuses to delete a base with
live dependents.

### 8.11 The checkpoint is a credential

Mechanism: the serialised object contains the optimizer state plus an environment snapshot, including tokens.
Anyone with read access to the bucket has the credentials, and checkpoints are usually readable by every job in
the namespace.

Competing causes: (a) secrets captured in the state dict; (b) an access policy broader than the run; (c) an
untrusted serialiser allowing code execution on load.

Discriminating check: scan a generation for credential patterns and deserialise it in a sandbox with no network;
an outbound connection attempt during load identifies (c), a pattern match identifies (a).

Correction: strip non-model state from the serialised payload, rotate anything found, and restrict the bucket to
the run's identity.

Prevention: an allow-list serialiser that can only emit tensors and declared scalars, plus a load path that
never enables arbitrary code execution.

### 8.12 Every component is healthy and the run still loses an hour a day

Mechanism: no single failure is large. Warm-up after each commit costs 20 s, the writer stalls the optimizer
for 114 s, and detection takes 30 s more than anyone assumed. Individually unremarkable, together they account
for the loss, and the phi model in F2 predicts them only if each term is measured.

Competing causes: (a) unmeasured commit stall; (b) per-commit warm-up after a graph re-capture; (c) a scheduler
that preempts at a fixed interval regardless of checkpoint age.

Discriminating check: instrument the loop so that every second is attributed to exactly one of: forward,
backward, optimize, commit, replay, restart. Unattributed time is the defect, and it is usually the largest
bucket before instrumentation.

Correction: pick the largest attributed bucket; if it is commit stall, move to async staging; if it is warm-up,
stop re-capturing graphs at commit boundaries.

Prevention: keep the attribution permanently, as a counter rather than a log, and review its shape weekly.

## 9. Complexity & Cost Analysis

### 9.1 Scaling of the write path

Cost per commit is O(S) in bytes when bandwidth-bound and O(N_shards * k) in requests when request-bound. The
crossover is the design question.

| Ranks | Per-rank payload | Per-rank write time at 16 MiB/s | Objects at 8,192 shards, k = 3 | Binding term |
|---:|---:|---:|---:|---|
| 64 | 14.26 GiB | 912.7 s | 24,576 → 16 s at 1,500/s | per-rank latency |
| 128 | 7.13 GiB | 456.4 s | 24,576 → 16 s | per-rank latency |
| 512 | 1.78 GiB | 114.1 s | 24,576 → 16 s | balanced |
| 2048 | 0.45 GiB | 28.5 s | 24,576 → 16 s | requests begin to matter |
| 8192 | 0.11 GiB | 7.1 s | 24,576 → 16 s | requests dominate |

Read the table by column rather than by row. Aggregate throughput is the per-rank rate times the rank count, so
it rises linearly while per-rank time falls: 64 ranks at 16 MiB/s is 1.0 GiB/s and 912.7 s, while 512 ranks is
8.0 GiB/s and 114.1 s. The object count does not move with rank count at all, because it is set by the shard
count and the objects per shard, which is why the last column flips from latency-bound to request-bound as ranks
increase. At 8,192 ranks each rank's shard is small enough that the per-object overhead — request, metadata,
checksum — exceeds the payload transfer, and the checkpoint stops scaling even though no single rank is slow.

This is the mechanism behind CI-07. A team raising shard count to reduce per-rank memory sees checkpoint time
rise, concludes the store degraded, and the store is behaving exactly as it did at 512 shards.

The aggregate figure is idealised: real aggregate improves sublinearly as concurrency rises, flattens at the
backend's knee, then falls when request overhead and retries dominate. The measured shape is an S-curve, and the
position of the knee is a property of the store and the fabric rather than of the checkpoint code.

### 9.2 Restore sequence and its budget

| Phase | Illustrative duration | Scales with | Addressable by storage upgrade |
|---|---:|---|---|
| Detect failure | 30 s | health-check interval | No; shorten the interval instead |
| Schedule replacement nodes | 60 s | queue depth, image size | No |
| Fetch from store | 152 s | bytes / read rate | Yes |
| Load, deshard, all-gather | 240 s | fabric bandwidth, shard imbalance | Partly |
| Warm-up to steady state | 90 s | compile, capture, first-step effects | No |
| Total | 572 s | — | Only 152 s of it |

Only 27% of the restart budget in this instance responds to a faster store. That ratio is the argument against
buying storage to fix RTO, and it holds for most configurations because loading is CPU- and fabric-bound while
fetching is store-bound.

### 9.3 Cost model

| Line item | Formula | Value at 912.7 GiB per generation |
|---|---|---|
| Hot storage, one generation | S_gb * price | 980 * 0.023 = $22.54 per month |
| Hot storage, eight generations | 8 * 22.54 | $180.32 per month |
| Across twenty concurrent runs | 20 * 180.32 | $3,606 per month |
| Request cost, 24.6k objects, 8 generations | 196.6k * price per 1k | $0.98 at $0.005/1k, negligible |
| Egress on restore | S_gb * egress price | $0 if in-region, material if not |
| Compute lost to phi = 19.8% | 0.198 * 512 accelerators * rate | the largest line by an order of magnitude |

The last row is the point. Storage cost scales with bytes; lost compute scales with the square root of bytes
through F3, and a 19.8% overhead on 512 accelerators for a month dwarfs the $180 of storage it protects. Teams
optimise the visible invoice and lose the invisible one, which is why phi belongs on the same dashboard as
spend.

### 9.4 Drill catalog

A drill is a deliberate fault injected against the checkpoint path, with a recorded outcome. These are the
sixteen that have the highest information per minute of effort.

| # | Injection | Expected behaviour | Pass criterion |
|---:|---|---|---|
| 1 | Kill a writer before its first byte lands | No commit; previous generation still selected | Restore from n-1 in under RTO |
| 2 | Kill a writer at 50% of its shard | Manifest absent; partial object not discoverable | Pointer unchanged |
| 3 | Kill the coordinator after the last shard, before the pointer | Generation orphaned but complete | Recovery selects n-1 without operator action |
| 4 | Kill the coordinator immediately after the pointer write | New generation selected | Restore from n, first-step loss within tolerance |
| 5 | Corrupt one shard's bytes, leave the size | Digest mismatch detected | Load refuses, falls back to n-1 |
| 6 | Delete one named object from the manifest | Validation fails before load | Error names the object, not a generic failure |
| 7 | Truncate the manifest itself | Unparseable manifest is not a commit | Fallback path exercised |
| 8 | Roll the pointer back one generation | Recovery follows the pointer, not modification time | Restore from n-1, and the rollback is visible in logs |
| 9 | Expire the object early via lifecycle rule | Publication or restore fails loudly | Alert fires before the run resumes from nothing |
| 10 | Restore into half the world size | Reshard path exercises identity mapping | First-step loss within tolerance, no shape error |
| 11 | Restore into double the world size | Same as above, opposite direction | Per-rank bytes balanced within 5% |
| 12 | Revoke the read credential mid-restore | Failure is explicit, retry is bounded | No partial load accepted |
| 13 | Fill the staging filesystem | Writer fails cleanly, trainer unaffected | No commit, no corruption |
| 14 | Throttle the store to 10% of normal | Commit window grows; trainer still correct | Timeout policy behaves as documented |
| 15 | Simulate a preemption with 60 s notice | Commit completes if it fits, aborts atomically if not | Aborted commit leaves no visible generation |
| 16 | Restore the oldest retained generation | Chain closure resolves | Restore completes; time recorded as the worst case |

Each drill has an observable pass criterion, which is what separates a drill from a hope. The ones teams skip
are 3, 5, and 8, and those are precisely the ones that decide whether recovery is a protocol or a habit.

### 9.5 Complexity summary

| Operation | Time complexity | Space complexity | Practical limit |
|---|---|---|---|
| Serialise state | O(S) bytes | O(S_staging) transient | Staging buffer vs free device memory |
| Publish | O(1) conditional write | O(1) | Requires a CAS primitive |
| Validate | O(S) bytes read | O(1) | Hashing cost, parallelisable |
| Restore | O(S + gather) | O(S) resident | Fabric bandwidth, shard balance |
| Retention closure | O(G * d) | O(G) | G generations, d chain depth |
| Drill | O(RTO) wall clock | one bucket prefix | Requires isolation from production |

**Takeaway:** the cost that matters is not the storage invoice; it is the fraction of accelerator-hours spent
not advancing the model, and only measurement attributes that fraction.

## 10. Key Theorems / Results Reference

**R1 — Consistent cut condition.** A checkpoint is logically consistent if and only if no collective operation
is in flight at the moment of the read and every rank's state belongs to the same completed update. Precondition:
the loop has an identifiable step boundary. Violation consequence: silent mixing of two updates, detectable only
by comparing per-shard step counters.

**R2 — Atomic publication.** If the set of objects is written before the pointer and the pointer update is a
conditional write on the previous value, then a reader that follows the pointer observes either the old complete
generation or the new complete generation, never a mixture. Precondition: objects are immutable and the pointer
is the only discovery mechanism. Violation consequence: readers assemble partial generations.

**R3 — Optimal interval (Young/Daly form).** Under constant failure hazard, the interval minimising
`(C + T/2 + R)/(T + C)` is `sqrt(2 C (M + R)) - C`, and the simpler `sqrt(2 C M)` is within about 10% whenever
`C, R << M`. Precondition: memoryless failure and measured C, R. Violation consequence: intervals chosen by
habit are typically too long, because habit inherits a per-node MTBF.

**R4 — Run MTBF scaling.** For `n` independent nodes with per-node MTBF `M_n`, the run's MTBF is `M_n / n`.
Precondition: failures independent and node-replacing. Violation consequence: at 512 nodes a 30-day node figure
becomes 1.4 h, and every interval derived from the node figure is 4.4x too long. This is the single most common
arithmetic error in checkpoint scheduling.

**R5 — RTO composition.** `RTO = detect + schedule + fetch + load + warm`, and no term can be reduced by
optimising a different term. Precondition: the terms measured separately. Violation consequence: storage
investment that cannot move the objective, because fetch is a minority of the sum.

**R6 — Delta chain bound.** If each delta restores in `d` seconds and a base in `b`, then restoring the head of
a chain of length `n` costs `b + n*d`, which grows linearly in `n` while storage grows sublinearly. Precondition:
deltas are applied in order. Violation consequence: an unbounded chain converts a storage win into an RTO
breach, discovered during the incident that needed it.

**R7 — Replication independence.** `P(total loss) = p^r` holds only for independent copies; copies sharing an
account, region, credential, or lifecycle policy behave as one copy. Precondition: independence by construction.
Violation consequence: a durability claim that is wrong by orders of magnitude exactly in the scenarios that
matter, such as a credential rotation or a bucket policy change.

**R8 — Reshard equivalence.** A generation can be restored at a different world size if and only if the shard
mapping is expressed by tensor identity and the loader can all-gather before use. Precondition: no rank-local
state outside the declared mapping. Violation consequence: shape-correct restores with silently permuted
parameters, which load without error and train badly.

**R9 — Validation asymmetry.** Digest verification proves integrity, not consistency or sufficiency. It detects
corruption and mixed generations only if each shard carries its generation identifier. Precondition: digests
computed over the bytes as stored. Violation consequence: an all-clear from validation on a torn step.

**R10 — Stall attribution.** Total non-progress equals the sum of commit stall, replay, restart, and unattributed
time; the model in R3 accounts for the first three but never the fourth. Precondition: per-second attribution
exists. Violation consequence: an overhead that is visible in wall clock and invisible in every dashboard.

**Takeaway:** nine of these ten fail silently; only the drill program converts them into observable behaviour.

## 11. Interview / Exam Questions

### Q1. Why is a weight export not a checkpoint?

**Answer:** It omits optimizer moments, step and scheduler state, scaler history, RNG positions, and the input
cursor, so a resume re-initialises the second-order moments and the schedule.

**Reasoning:** For this model the moments and master copy are 85.7% of state (781.6 of 912.7 GiB); a 130.4 GiB
export has captured a sixth of what a restart needs. Adam's update magnitude depends on running moment
estimates, so resetting them changes the effective step size for hundreds of updates.

**Follow-up:** If you must restart from weights, what else has to be re-derived, and how long does that take?

**Red flag:** "The parameters are the model, so the parameters are the state."

### Q2. Compute the optimal interval for C = 200 s, R = 400 s, M = 4 h.

**Answer:** T* = sqrt(2 * 200 * 14400) = sqrt(5.76e6) = 2400 s = 40 min, giving phi = 200/2400 + (1200+400)/14400
= 8.33% + 11.11% = 19.4%.

**Reasoning:** M = 4 h = 14400 s. The two terms are nearly balanced at the optimum, which is expected: at T*, C/T
equals T/(2M), the two halves of the trade.

**Follow-up:** What if the run actually spans 1024 nodes with per-node MTBF of 60 days?

**Red flag:** Using minutes in one term and seconds in the other, or skipping the R term entirely.

### Q3. A 512-node run has per-node MTBF of 30 days. What is the run MTBF, and why does it change the schedule?

**Answer:** 720 h / 512 = 1.41 h = 5062 s. With C = 114 s, T* = sqrt(2*114*5062) = 1074 s = 17.9 min, not the
26.2 min that the three-hour assumption gave.

**Reasoning:** Independent node failures aggregate as a rate sum, so the run hazard is the node hazard times the
node count. The optimal interval scales as sqrt(M), so a 2.13x smaller M gives a 0.69x interval.

**Follow-up:** What if the failures are not independent — say, a top-of-rack switch serving 64 nodes?

**Red flag:** Treating vendor node MTBF as a property of the job.

### Q4. Where does a restored run lose time if the store is upgraded from 4 to 12 GiB/s?

**Answer:** Only the fetch term changes: 912.7/4 = 228.2 s becomes 912.7/12 = 76.1 s, a saving of 152.1 s of a
572 s restore, taking R to 420 s and phi at T = 1570 to 0.0727 + (784.9+420)/10800 = 0.0727 + 0.1116 = 18.4%,
about 1.4 points.

**Reasoning:** The load, warm-up, detection, and scheduling terms are unaffected. The saving is real but bounded
by the share of R that the store actually serves.

**Follow-up:** Which single change would buy more, and what would you measure to decide?

**Red flag:** Quoting the throughput improvement as if the whole restore were store-bound.

### Q5. Design the publication protocol for an object store with only per-object atomicity.

**Answer:** Write immutable objects under a generation-scoped prefix, write the manifest last inside that prefix,
then commit by a compare-and-set on a single pointer object that names the generation. Readers follow only the
pointer and validate the named manifest before loading.

**Reasoning:** Immutability removes the overwrite class of defect; the CAS makes concurrent coordinators safe;
validation turns a corrupt generation into a fallback rather than a failed restore.

**Follow-up:** What does the reader do if the pointer names a generation whose manifest is missing?

**Red flag:** Publishing by writing a `latest` file with unconditional overwrite, or by directory modification
time.

### Q6. A resume completes and the loss jumps by 0.4 for fifty steps. Diagnose.

**Answer:** Compare the first forward loss on the identical batch before and after resume. A jump on the identical
batch is a state gap; a differently ordered batch is a cursor gap; a jump that decays within a few steps with
identical state is a legitimate numerics change from a new topology.

**Reasoning:** The three candidates are separable by holding the batch fixed, which removes data order as a
variable. Only then does the loss difference mean anything.

**Follow-up:** Which state element would you inspect first, and what would its absence look like numerically?

**Red flag:** Re-running with a fresh seed and concluding the checkpoint is fine because the loss returns.

### Q7. Estimate the storage cost of keeping eight generations for twenty runs.

**Answer:** 912.7 GiB = 980.0 GB per generation. Eight generations = 7.84 TB; at $0.023/GB-month that is
$180.32 per run-month, so twenty runs cost $3,606 per month.

**Reasoning:** The GiB-to-GB conversion is explicit because the price is quoted per decimal GB. Using 912.7 as
GB instead of 980.0 understates by 7%.

**Follow-up:** What does the same line cost if four of the eight generations move to an archive tier?

**Red flag:** Multiplying GiB by a per-GB price without stating the convention.

### Q8. What makes async checkpointing unsafe, and what makes it safe?

**Answer:** Unsafe when the trainer mutates the buffer being copied; safe when the snapshot is taken at a safe
point into a staging region that the trainer never writes again, and publication waits for the upload.

**Reasoning:** Async means "the trainer continues", not "the state is frozen". If those two are conflated, the
writer reads a buffer mid-update and produces a torn step with a perfectly valid digest.

**Follow-up:** How would you prove no mutation, cheaply, in a test?

**Red flag:** "It's async, so it can't stall training" — with no statement about buffers.

### Q9. A checkpoint takes 114 s and the interval is 600 s. Is that acceptable?

**Answer:** No. The write consumes 19% of wall clock (114/600), and phi = 0.190 + (300+572)/10800 = 0.190 +
0.081 = 27.1%. Moving to 1200 s cuts phi to 20.4% while halving the RPO.

**Reasoning:** The commit-to-interval ratio is the first-order term in F2; anything above roughly 15% means the
interval is chosen for loss tolerance rather than for overhead.

**Follow-up:** What would have to be true for 600 s to be right anyway?

**Red flag:** Accepting it because "we want a short RPO" without pricing the overhead.

### Q10. What is wrong with keeping checkpoints in node-local NVMe for speed?

**Answer:** They die with the node, which is the failure they were meant to survive; and a restore from a
different node cannot see them at all.

**Reasoning:** The recovery path selects generations by pointer from a shared store. A local generation is
discoverable only by the node that wrote it, so its effective availability is the node's availability.

**Follow-up:** What role can local NVMe legitimately play?

**Red flag:** "We replicate by using RAID on the node."

### Q11. Your delta chain is 30 deep and storage is fine. What is the risk?

**Answer:** Restore time grows linearly: 30 deltas at 6 GiB/s over 638.9 GiB each is 3,194 s of reads plus the
base, so RTO is breached even though bytes are saved.

**Reasoning:** Applying deltas is sequential and cannot be parallelised across the chain, so the cost is a sum,
not a maximum. Compaction into a new base is the only way to bound it.

**Follow-up:** What chain depth would you allow, and what enforces it?

**Red flag:** Measuring only storage saved, never the restore time of the deepest chain.

### Q12. A drill restores in 5 minutes; the incident takes 40. What did the drill miss?

**Answer:** Likely a cold image pull, a different identity's permissions, an older generation on a slower tier,
or contention with other tenants during the real event.

**Reasoning:** All four are absent from a warm, isolated, most-recent-generation drill. The drill measured the
protocol's happy path, which is the part least likely to be wrong.

**Follow-up:** Design the drill that would have predicted the 40 minutes.

**Red flag:** Concluding that the incident was an outlier rather than the drill being unrepresentative.

### Q13. How do you detect a torn step without per-shard metadata?

**Answer:** You mostly cannot, cheaply. You can compare a checksum of the concatenated state against a
recomputed value only if you have a reference, which you do not. The practical answer is to add the step counter
to the manifest and to each shard.

**Reasoning:** Digests prove bytes are as written, not that they belong together. Detecting a mixed set requires
an identifier common to the shards, recorded at write time.

**Follow-up:** Which is cheaper, adding metadata or replaying from the previous generation after every failure?

**Red flag:** "We would see it in the loss curve" — sometimes, hours later.

### Q14. Compute the checkpoint overhead if the run has M = 2 h, C = 90 s, T = 900 s, R = 300 s.

**Answer:** phi = 90/900 + (450 + 300)/7200 = 0.1000 + 0.1042 = 20.4%. The optimum would be
sqrt(2*90*7200) = 1138 s.

**Reasoning:** M = 7200 s. At T = 900 the write term and the loss term happen to be nearly equal (10.0% against
10.4%), which is the signature of an interval close to optimal; 1138 s gives 0.0791 + 0.1207 = 20.0%.

**Follow-up:** Given a gap of only 0.4 points, what argument would you actually make for changing the interval?

**Red flag:** Computing the loss term as T/M rather than (T/2 + R)/M.

## 12. Quick Reference Card

| Item | Rule | Number to remember |
|---|---|---|
| State per parameter | bf16 weights + fp32 master + two Adam moments = 14 bytes | 70e9 parameters → 980e9 B = 912.7 GiB |
| Write cost | C = S / B, but take the max over ranks | 912.7 GiB at 8 GiB/s = 114 s |
| Interval | T* = sqrt(2 C M); flat within ±50% | C = 114 s, M = 3 h → 26 min |
| Run MTBF | per-node MTBF divided by node count | 30 days at 512 nodes = 1.41 h |
| Restart time | detect + schedule + fetch + load + warm | 572 s here; only 152 s is storage |
| Publication | immutable objects, manifest last, CAS pointer | one conditional write decides recovery |
| Delta chains | bytes grow sublinearly, restore grows linearly | cap depth; 7 deltas = 897 s restore |
| Validation | digest ≠ consistency ≠ sufficiency | compare step counters across shards |
| Retention | delete by chain closure, never by count | base with live deltas is undeletable |
| Overhead | phi = C/T + (T/2 + R)/M, on the dashboard with spend | 19.8% here = the largest cost line |
