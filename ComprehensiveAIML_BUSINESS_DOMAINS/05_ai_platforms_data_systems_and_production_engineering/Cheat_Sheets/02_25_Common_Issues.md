---
title: "25 Common Issues — AI Platforms, Data Systems and Production Engineering"
topic: "Diagnostic catalog for training, serving, data, evaluation, and platform failures"
collection: "ai-platforms-data-systems-production-engineering"
level: "advanced"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 25 Common Issues — AI Platforms, Data Systems and Production Engineering

## 1. File Header

**Outcome:** turn a symptom report from a training, serving, data, or platform system into a bounded test that
separates competing causes, then into a correction that removes the mechanism rather than the observation.

**Audience:** platform and reliability engineers on call for AI systems; the reader is assumed to know what a
gradient is, what an optimizer step is, and how to read a latency histogram without being told.

**Scope:** the seven modules of this part — accelerator hardware, distributed training and compute, serving and
inference cost, MLOps and feature platforms, data quality and reliability, evaluation and monitoring, and API
platform engineering with FinOps. The catalog deliberately excludes model-quality statistics that belong to an
evaluation sheet, except where a platform failure masquerades as one.

**What this file is not.** It is not a runbook with commands. A runbook encodes one team's tooling; this encodes
the discriminating observation, which survives a tooling change. Where a command would be needed, the entry says
which observable to collect instead.

**Evidence status:** every configuration, byte count, rate, and price is constructed for teaching and labelled
ILLUSTRATIVE. Arithmetic is derived and re-checked here. No entry describes a real incident at a named company,
and no measured result is claimed. Where an entry depends on provider behaviour, it states the behaviour as an
assumption to verify rather than a fact.

**Identifier stability:** `CI-01` through `CI-30` are stable. A later revision may sharpen an entry but must not
renumber it, because sheets and exercises cite these identifiers. New entries append above 30; they never
renumber the existing twenty-five-plus catalog.

**Count:** the file carries thirty issues, exceeding the twenty-five minimum, because three of the seven modules
in this part produce distinct failure classes that a shorter catalog would have to merge — and merging
`GiB versus GB` with `mean of ratios` would hide that one is a unit defect and the other an aggregation defect.
Merging them is exactly the compression that makes a catalog useless.

**The single sentence worth memorising:** almost every entry here is a case where two different quantities share
one name, and the fix is to name both.

**Companions:** `./01_Distributed_Training_Checkpoint_and_Recovery.md` supplies the checkpoint model that entries
CI-01, CI-02, CI-08, CI-09, CI-20, CI-21, CI-22, CI-23, and CI-26 extend. `../INDEX.md` maps the seven modules.
Cross-references use `CI-nn` so a case can cite a diagnostic without copying it.

**Conventions:** `GiB = 2^30` bytes, `GB = 10^9` bytes, `MiB = 2^20` bytes. Rates are per second. Percentages are
of the stated denominator, always stated. Time is seconds unless marked. Every constructed figure is ILLUSTRATIVE.

## 2. Concept Map / Mental Model

```text
symptom
  -> which quantity is misnamed?        (identity layer)
  -> which scope is being summed?       (aggregation layer)
  -> which assumption about the input?  (assumption layer)
  -> which resource actually binds?     (capacity layer)
  -> discriminating observation
  -> correction at the authoritative layer
  -> verification on a boundary record
```

The five-layer ladder is the triage order, and the order matters because the layers contaminate downward. A
misnamed quantity (identity) produces a wrong aggregation even when the aggregation code is correct. A wrong
aggregation produces a false capacity conclusion even when the capacity model is correct. Reading the ladder from
the top is therefore cheaper than starting from the symptom's most visible layer, which is almost always the
capacity layer.

**Layer 1, identity.** Two numbers share a name and differ in unit, scope, or definition. GiB against GB; p50 of
per-minute means against p50 of requests; weights against restartable state; a prefetch cursor against a
consumption cursor. No amount of careful downstream arithmetic repairs this, and the downstream work usually
looks correct.

**Layer 2, aggregation.** The quantity is well named per item but is combined incorrectly across items. Mean of
per-rank rates against ratio of summed bytes and summed time; mean of per-tenant cache hit rates against a
pooled rate; per-node MTBF summed against a run. The tell is that the answer moves when the grouping changes,
which it should not for an intensive quantity.

**Layer 3, assumption.** The mechanism is arithmetically sound but one of its premises was false in this
instance: the failure process is not memoryless, the copies are not independent, the feature is not available at
request time, the shard mapping is not rank-invariant. These are the entries where the formula is right and the
world is different.

**Layer 4, capacity.** Something saturated: requests, fabric, memory, queue. This layer is the most visible and
the least explanatory, because saturation is a consequence. A saturated store during a checkpoint is layer 4;
the layer-1 defect was choosing a shard count whose request budget exceeded the fair share.

**Layer 5, observation.** The instrument is wrong. Percentiles computed from pre-averaged data, counters that
never emit, timers that exclude warm-up. Diagnosing at layer 5 feels like progress because the numbers start
moving, which is precisely the danger.

```text
   identity  ----->  aggregation  ----->  assumption
        |                  |                  |
        v                  v                  v
   wrong unit        wrong scope        false premise
        \                 |                 /
         +--------> capacity saturation <--+
                          |
                          v
                   observation defect  (hides all of the above)
```

**Takeaway:** a symptom names a layer; evidence must name a mechanism, and the two are never the same thing.

## 3. Foundational Definitions

| Term | Meaning, with unit | Boundary that trips people |
|---|---|---|
| Intensive quantity | Value invariant to the size of the group it is computed over, e.g. a rate, a ratio, a temperature | A mean of means is not intensive; a pooled ratio is |
| Extensive quantity | Value that scales with group size, e.g. bytes, tokens, requests | Summing is correct; averaging across unequal groups is not |
| Effective batch | Samples whose gradients contribute to one optimizer update, `N_ranks * batch_per_rank * accumulation` | Not the per-rank batch; not the microbatch |
| Goodput | Share of wall clock spent on work that survives a failure | Not utilisation, not throughput |
| Token | One unit of model input or output as tokenised | Not a word; not a character; not a fixed byte count |
| Prefill | Phase that processes the prompt and populates the cache | Cost scales with prompt length, once |
| Decode | Phase that emits one token per sequence per step | Cost scales with output length and batch |
| KV cache | Per-token stored keys and values, `2 * L * n_kv * d_head * bytes` | Grows with sequence length and concurrency, not with weights |
| Padding | Tokens added to make a batch rectangular | Paid for in compute, not in information |
| p99 | Value below which 99% of a stated population falls | Depends entirely on the population and window |
| Window | The interval over which a rate or percentile is computed | An empty window has no rate; zero is not the answer |
| Cut-off | The instant at which a decision's inputs are frozen | Event time and processing time are different clocks |
| Feature freshness | Age of the newest value available to a decision | A "real-time" feature can still be hours stale |
| Prevalence | Share of positives in the population a model scores | Not the share in the training set |
| PPV | Positive predictive value, `TP / (TP + FP)` at a stated threshold and prevalence | Falls as prevalence falls, at constant sensitivity |
| RTO | Elapsed time until useful work resumes | Measured end to end or not at all |
| Idempotency key | Identifier that makes a repeated effect collapse to one | Requires the effect to be keyed, not merely retried |
| Warm pool | Pre-initialised capacity outside the critical path | Only helps if the pool is larger than the burst |
| Cost per token | Charge divided by tokens actually billed | Input and output prices differ; padding is billed |
| Shard | Partition of state or data across ranks | Partition count is not free; it buys requests |
| Rank-invariant mapping | Shard assignment expressed by tensor identity rather than process index | The only basis on which reshard can work |

Four definitions carry most of the diagnostic weight and deserve their own paragraph.

**Intensive versus extensive** is the whole of layer 2. Bytes are extensive: 512 ranks holding 1.78 GiB each hold
912.7 GiB in total, and summing is right. Rate per rank is intensive: the correct aggregate is total bytes over
total time, not the mean of the per-rank rates, because the ranks did not each run for the same duration. The
difference is arithmetic, not philosophy, and CI-06 shows the error at 3.8% when exposures differ by 2x and at
unbounded magnitude as they diverge further.

**p99 is a property of a population, not of a service.** Two teams can report different p99s for the same service
and both be right, if one computes over requests and the other over one-minute averages of requests. Averaging
first destroys the tail by construction: a one-minute mean of a thousand requests cannot show that eleven of them
took four seconds. CI-19 quantifies the collapse.

**Prevalence is not a property of the model.** Every threshold-based quality number is conditional on the
population it was measured in. A model that is excellent at 12% prevalence can be useless at 1.2%, and the
arithmetic that shows this is three lines long (CI-16). Deployment changes prevalence even when the model does
not change, which is why an unchanged model can have a changed quality number.

**Idempotency requires a key, not a retry.** Retrying a request that increments a counter increments it twice.
Retrying a request that writes `(run_id, step)` collapses. The distinction is about the effect's identity, and no
amount of client-side retry logic substitutes for it. CI-25 is the training-side instance: a checkpoint restore
replays an interval while the external world has already advanced.

**Takeaway:** decide first whether the quantity is intensive or extensive; most scope errors are answered by
that one question.

## 4. Core Formulas with Full Derivation

The formulas below are the ones cited by more than three catalog entries. Each is stated with its validity
condition, because every one of them is applied outside its range somewhere in the catalog.

### 4.1 Base case: the pooled-versus-averaged ratio

```text
FORMULA [F1]:  r_pooled = sum(x_i) / sum(n_i)          r_mean = (1/k) * sum(x_i / n_i)
WHERE: x_i is a count of events in group i; n_i is that group's exposure (requests, seconds, bytes);
       k is the number of groups; r is a rate with units events per unit exposure.
CONSTRAINT: r_pooled equals the rate of the union of groups and is therefore independent of grouping;
            r_mean equals r_pooled only when all n_i are equal.
NUMERICAL EXAMPLE: group A = 1024 events over 512 s, group B = 640 over 512 s.
       r_pooled = 1664/1024 = 1.625 events/s. r_mean = (2.0 + 1.25)/2 = 1.625 — equal here because n_i match.
       Now change B's exposure to 256 s: r_pooled = 1664/768 = 2.167; r_mean = (2.0 + 2.5)/2 = 2.25, 3.8% high.
INTERPRETATION: equality holds exactly when exposures match, which is why the defect hides in symmetric tests.
```

The asymmetry of the error is the useful part. When one group has much more exposure than the other, the mean of
rates over-weights the small group, and it over-weights it in whichever direction that group happens to lean. A
small, slow, or failing group is therefore systematically over-represented in the average — which is exactly the
group an operator is looking for, making the defect look like a real signal.

### 4.2 General case: prevalence and predictive value

```text
FORMULA [F2]:  PPV(pi) = pi * se / (pi * se + (1 - pi) * (1 - sp))
WHERE: pi is prevalence as a proportion; se is sensitivity; sp is specificity; PPV is the share of flagged
       cases that are truly positive.
CONSTRAINT: se and sp are fixed properties of the model at a fixed threshold; pi is a property of the scored
       population. Changing the threshold changes both se and sp, so this formula holds only at one operating point.
NUMERICAL EXAMPLE: se = 0.90, sp = 0.95. At pi = 0.12: PPV = 0.108 / (0.108 + 0.044) = 0.710.
       At pi = 0.012: PPV = 0.0108 / (0.0108 + 0.0494) = 0.179.
INTERPRETATION: a tenfold prevalence drop takes precision from 71% to 18% with an unchanged model and threshold.
```

The denominator structure explains why the collapse is so violent: the false-positive term `(1-pi)(1-sp)` barely
moves as prevalence falls, because it is dominated by the large negative class. The true-positive term shrinks
with `pi`. At low prevalence the model's precision is governed almost entirely by its specificity, which is why
"we need 95% precision at 1% prevalence" is a statement about `sp`, not about ranking quality.

### 4.3 Variants

```text
FORMULA [F3]:  KV_bytes_per_token = 2 * L * n_kv * d_head * b
WHERE: L is layer count; n_kv is key/value heads per layer (1 for multi-query, L for multi-head); d_head is head
       dimension; b is bytes per element; the 2 is for one key tensor and one value tensor.
CONSTRAINT: no paging, no quantisation, no sharing across sequences; the product must be per *token*, so a
       request's cache is this value times its sequence length.
NUMERICAL EXAMPLE: L = 80, n_kv = 8, d_head = 128, b = 2: 2*80*8*128*2 = 327,680 B = 320 KiB per token.
       A 2,048-token sequence therefore holds 640 MiB; 32 concurrent sequences hold 20.0 GiB.
INTERPRETATION: concurrency times length, not parameter count, decides whether the cache fits.
```

```text
FORMULA [F4]:  padding_utilisation = L_mean / L_max          cost_multiplier = L_max / L_mean
WHERE: L_mean is mean sequence length in the batch; L_max is the padded length the kernel executes.
CONSTRAINT: static padding to a power of two or to a fixed maximum; continuous batching and length bucketing
       change the formula by changing L_max per step.
NUMERICAL EXAMPLE: L_mean = 300, L_max = 4096: utilisation = 7.32%, multiplier = 13.65x.
INTERPRETATION: a 13.65x multiplier on the prefill term is invisible in a per-token bill and obvious in GPU time.
```

```text
FORMULA [F5]:  M_run = M_node / n          T* = sqrt(2 * C * M_run)
WHERE: M_node is per-node mean time between failures; n is independent node count; C is checkpoint cost;
       T* is the overhead-minimising interval under a constant hazard.
CONSTRAINT: independent failures; node-replacing recovery; C << M_run. Correlated failures (rack, switch,
       power domain) invalidate the division and must be modelled as a separate hazard with its own n.
NUMERICAL EXAMPLE: M_node = 720 h, n = 512: M_run = 1.406 h = 5063 s. With C = 114 s, T* = 1075 s = 17.9 min.
INTERPRETATION: the interval shrinks as sqrt(n); a schedule derived from the node figure is 1.46x too long here.
```

```text
FORMULA [F6]:  cost_per_1k = (n_in * p_in + n_out * p_out) / 1000
WHERE: n_in and n_out are token counts actually billed, including padded prompt tokens; p_in and p_out are
       prices per token; the divisor converts to the vendor's quoting unit.
CONSTRAINT: cache reads, batch discounts, and reasoning tokens may be priced differently and must be added as
       separate terms rather than folded into either price.
NUMERICAL EXAMPLE: p_in = $3 per 1M tokens, p_out = $12 per 1M; 1,000 in and 300 out: (0.003 + 0.0036) = $0.0066.
       Per 1k *requests* that is $6.60. With L_max = 4096 padding on a 300-token prompt: in = 0.012288 + 0.0036.
INTERPRETATION: prompt padding moves the input term by 4x on a short prompt; the bill looks like a throughput bug.
```

### 4.4 Limits and counterexamples

**Where F1 breaks.** With one group, `r_mean = r_pooled` trivially, so a unit test on a single group cannot
detect the defect. A test that uses equal exposures also passes. The only test that discriminates uses two groups
with deliberately unequal `n_i` and a known answer; anything else is a test of the test.

**Where F2 breaks.** When the threshold is chosen per deployment — the common case — `se` and `sp` are not
constants across the two prevalences, and the formula's answer is a lower bound on the collapse, because
thresholds tuned at low prevalence usually raise `sp` further. Conversely, when prevalence shifts and the
threshold is retuned, the comparison of PPV across deployments is not a measurement of model change at all.

**Where F3 breaks.** Paged or quantised caches, prefix sharing across requests, and sliding-window attention each
change the per-token product by a factor. A model with grouped-query attention at `n_kv = 8` of 64 heads stores
an eighth of what the multi-head formula implies; a model with a sliding window stores a constant rather than a
linear amount once the window is full. Applying F3 across architectures is the most common capacity error in
serving reviews.

**Where F4 breaks.** Continuous batching removes static padding from the decode phase but not from prefill, so a
figure computed from total latency mixes a padded term with an unpadded one. The correct statement names the
phase: prefill padding utilisation 7.32%, decode utilisation near 100% in occupancy terms.

**Where F5 breaks.** Correlated failure. A fabric partition removing 64 nodes at once is one event, so the rate
model needs a second hazard with its own rate; `M_node / n` then understates the run hazard while overstating
independence. The diagnostic is a failure log grouped by minute: correlated events cluster, independent ones do
not.

**Where F6 breaks.** When the provider bills per request rather than per token, or applies a committed-throughput
discount, the per-token formula describes a price list rather than an invoice. The invoice is the authority, and
the formula's job is to predict it, not to replace it.

**Validity condition for the section:** each formula assumes the population and the unit stated in its `WHERE`
line. Every catalog entry below that cites a formula also names the premise that failed, because a formula is
never wrong on its own.

## 5. Worked Numerical Example

ILLUSTRATIVE composite: a 70e9-parameter model served on 8 accelerators, 512 training ranks, a feature platform
with three online stores, and a 3-hour run-level MTBF. The point is to run one number through several catalog
entries and show that they are the same defect at different layers.

### 5.1 The reported symptom

An operator reports that "checkpointing is eating the cluster": the training run is losing about 20% of wall
clock, and the storage invoice rose 8x in the same week. Two separate claims, and the catalog says they belong to
different layers.

### 5.2 Layer 1 first: are the bytes the bytes?

Per parameter the state is 2 (bf16 weights) + 4 (fp32 master) + 4 + 4 (two moments) = 14 bytes, so
14 * 70e9 = 980e9 bytes. As GiB: 980e9 / 1.0737e9 = 912.7 GiB. As GB: 980.0 GB.

The invoice is quoted per GB-month at an ILLUSTRATIVE $0.023: 980.0 * 0.023 = $22.54 per generation-month. The
capacity plan was computed in GiB, so the plan said 912.7 units and the invoice charged 980 — a 7.4% gap on every
line, which is CI-03 and not a rate change. With eight generations and twenty runs the gap is 0.074 * $3,606 =
$267 per month of pure unit confusion.

### 5.3 Layer 2: is the rate aggregated correctly?

The checkpoint writer reports per-rank rates of 16 MiB/s on 480 ranks and 4 MiB/s on 32 ranks. Compute both ways.

Mean of rates: (480 * 16 + 32 * 4) / 512 = (7680 + 128) / 512 = 15.25 MiB/s. Times 512 ranks = 7,808 MiB/s
aggregate, so the predicted cost is 912.7 * 1024 / 7808 = 119.7 s.

Pooled: total bytes 980e9 B; time is governed by the slowest rank in each stage, but the store sees the sum of
completed bytes. If the 32 slow ranks take 1.783 * 1024 / 4 = 456.4 s and the rest take 114.1 s, the *commit*
completes at 456.4 s regardless of the aggregate rate. Pooled throughput over that window is
980e9 / 456.4 = 2.147e9 B/s = 2.0 GiB/s, not 7.6 GiB/s.

So the arithmetic that "proves" a 120-second checkpoint predicts a 456-second stall, a 3.8x error, and the
difference is not an aggregation bug alone — it is the max-versus-mean defect. The correct statement is:
aggregate is pooled over completed bytes, but latency is the maximum over ranks.

### 5.4 Layer 3: which premise failed?

The overhead of 20% is consistent with the model in the companion sheet: with C = 114 s, R = 572 s, M = 10800 s,
T = 1570 s, phi = 0.0727 + 0.1256 = 0.1983. But the observed 20% includes a 456-second stall on the slow ranks,
so the premises to test are (a) the hazard is constant, (b) C is the 114-second figure, (c) R is measured end to
end. A drill answers all three; the invoice answers none.

Recompute with the run-level MTBF instead of the node figure. If the run actually spans 512 nodes with per-node
MTBF 720 h, then M_run = 5063 s and T* = 1075 s, and phi at T = 1570 becomes
114/1570 + (785 + 572)/5063 = 0.0726 + 0.2680 = 34.1%. The 20% the operator sees is *better* than the model
predicts, which sends the investigation to detection and scheduling rather than to storage.

### 5.5 Layer 4: what actually binds?

Request count: 8,192 optimizer shards at 3 objects each = 24,576 objects. At a fair share of 1,500 requests/s
that is 16.4 s — not binding. Bandwidth: not binding at 8 GiB/s aggregate. The slow ranks are binding, and they
are binding because of a per-node disk that is shared with a data loader. That is a capacity answer, and it is
the fourth layer, which is why it took four steps to reach.

### 5.6 The two-route re-check

Route A, wall clock: 456.4 s of stall per commit at T = 1570 s means 3,600 / 1570 = 2.293 commits per hour, so
stall per hour = 2.293 * 456.4 = 1,046.5 s = 29.1% — worse than the reported 20%, which tells us the operator's
figure was taken during a period without the slow ranks.

Route B, per-token state: 14 bytes per parameter at an aggregate 2.0 GiB/s = 2.147e9 B/s gives
14 / 2.147e9 = 6.52e-9 s per parameter; times 70e9 = 456.4 s. Two independent routes, same answer.

### 5.7 Sensitivity

| Slow-rank rate | Slow-rank time | phi at T = 1570, M = 10800 | phi at T = 1075, M = 5063 |
|---:|---:|---:|---:|
| 16 MiB/s (uniform) | 114.1 s | 19.8% | 32.5% |
| 8 MiB/s | 228.2 s | 27.1% | 43.1% |
| 4 MiB/s | 456.4 s | 41.6% | 64.4% |
| 2 MiB/s | 912.7 s | 70.7% | 106.8% |

Two things move between the columns, and the example's point depends on separating them. The right column uses
the run-level hazard from 512 nodes, which lowers the optimal interval from 1570 s to 1075 s; that change alone
raises overhead from 19.8% to 32.5% at a uniform 114 s commit cost. The rows then add the slow-rank effect, and
at 2 MiB/s the model returns 106.8%, which is arithmetically honest and operationally meaningless: overhead over
100% means the run makes no forward progress at all, because a 912.7 s commit inside a 1075 s cycle leaves
nothing.

That last row is why the ladder in section 2 puts capacity fourth. Optimising T while one rank is 4x slower tunes
a term that is not binding, and the only correct action is to find why the rank is slow.

### 5.8 What the example establishes

It establishes that the same symptom — "checkpointing costs 20%" — has a unit error, an aggregation error, a
premise error, and a capacity cause underneath it, in that order, and that the cost of diagnosing in ladder order
is four small computations rather than one large rewrite.

It does not establish that 456 s is the right number for any system, nor that the storage invoice rose for the
reason assumed. The invoice is a separate investigation with its own layer-1 question.

## 6. Visual / Geometric Intuition

```text
         reported number
               |
        [unit identity]  -- mismatch? --> stop, fix units (CI-03, CI-04)
               |
        [aggregation]    -- moves when regrouped? --> ratio of sums (CI-06, CI-19, CI-24)
               |
        [premise]        -- fails a drill? --> assumption entry (CI-16, CI-27, CI-29)
               |
        [capacity]       -- saturates? --> find binding resource (CI-07, CI-12, CI-18)
               |
        [instrument]     -- disagrees with a second source? --> fix measurement (CI-19, CI-20)
```

The vertical order is a cost gradient: each layer down requires more instrumentation and more time. The
horizontal question at each layer is the discriminator, and it is chosen so that the answer is a *different
observation*, not a stronger opinion.

A second picture explains the p99 collapse, because it is the entry operators most often dispute:

```text
requests:   |1s|1s|1s|1s|1s|1s|1s|1s|1s|4s|      true p99 (of 10) = 4 s
per-minute: |  1.3 s mean  |                      reported p99 = 1.3 s
```

The mean is not lying; it is answering a different question. Any percentile computed on pre-aggregated data is a
percentile of the aggregates, and aggregates are less extreme than their members by construction. The geometric
statement is that averaging is a contraction toward the centre, so it can only shrink tails.

A third picture, for the capacity layer, is a stacked bar per phase where the tallest segment is not the one
under discussion:

```text
restore budget, 572 s total
[detect 30][sched 60][fetch 152][load 240][warm 90]
                          ^ storage's whole share
```

The bar makes the argument that no storage purchase can move 73% of the budget. It is the cheapest possible
counter to a proposal that starts with "we should upgrade the store", and it takes one paragraph to draw.

A fourth, for prevalence, is the confusion matrix scaled by population:

```text
at pi = 12%:   TP 108 | FP  44      PPV 71%
at pi = 1.2%:  TP  11 | FP  49      PPV 18%
```

The false-positive cell barely moves while the true-positive cell collapses by ten, and that asymmetry is the
entire mechanism of CI-16. Drawing it once prevents the argument about whether the model "got worse" — it did
not; the population changed.

**Takeaway:** the ladder orders the work by cost, and the pictures show which cell or segment is doing the damage.

## 7. Comparison / Selection Table

### 7.1 Which check to run first

| Symptom family | First discriminator | Why it is first | Entry |
|---|---|---|---|
| Numbers do not reconcile with an invoice or quota | Recompute in both units and compare | Cheapest, and unit errors invalidate everything downstream | CI-03 |
| Aggregate disagrees with per-group reports | Recompute as ratio of sums | Two lines of arithmetic, conclusive | CI-06 |
| Tail latency differs between teams | Ask what population the percentile was taken over | Distinguishes instrument from service | CI-19 |
| Quality fell with no model change | Compute prevalence in the scored population | Separates population change from model change | CI-16 |
| OOM under load, not at rest | Compute cache from concurrency times length | Separates weights from activations and cache | CI-12 |
| Restore misses its objective | Time the five phases separately | Identifies the addressable term | CI-20 |
| Duplicate side effects after a retry | Check whether the effect carries a key | Separates retry from non-idempotency | CI-25 |
| Throughput fell after a topology change | Compare per-rank bytes and kernel selection | Separates numeric change from load imbalance | CI-23 |

### 7.2 Discriminating checks and what they cost

| Check | Evidence produced | Cost | Fails to discriminate when |
|---|---|---|---|
| Recompute in both units | Which unit the source used | Seconds | Both readings agree by coincidence near 1.07x |
| Ratio of sums | Whether aggregation is the defect | Seconds | Groups have equal exposure (defect invisible) |
| Replay one record at a boundary | Whether behaviour differs at the edge | Minutes | The boundary case is not exercised by the path |
| Two-source comparison | Whether the instrument is wrong | Hours, needs a second source | Both sources share a library |
| Fault injection | Whether the protocol is a protocol | Hours, needs isolation | Injection changes the timing being measured |
| Population recount | Whether prevalence moved | Hours, needs labels | Labels arrive late and are themselves shifted |

### 7.3 Correction scope

| Correction | Removes | Leaves | When to choose |
|---|---|---|---|
| Rename the field or unit | Layer 1 confusion | The wrong arithmetic, now visible | Always, first |
| Change aggregation to pooled | Layer 2 defect | Layer 3 premises untested | Whenever regroup changes the answer |
| Add a drill | Layer 3 blind spot | Nothing about capacity | Before any RTO commitment |
| Add capacity or admission control | Layer 4 saturation | The demand that caused it | After the binding resource is named |
| Replace the instrument | Layer 5 defect | The original question, still open | When a second source disagrees |

**Takeaway:** pick the check whose failure is informative; a check that passes under both the right and the wrong
implementation has cost time and bought nothing.

## 8. Failure Modes & Edge Cases

### 8.0 Catalog summary

Thirty entries, ordered by layer and then by module, because the layer order is the triage order and the module
order keeps related symptoms adjacent. Categories: VIOLATED_ASSUMPTION (the premise was false),
NUMERICAL_EDGE (the arithmetic or unit breaks at a boundary), SCALE_FAILURE (behaviour changes with size), and
CONCEPT_CONFUSION (two quantities share a name).

| ID | Category | Symptom in one line |
|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | A generation mixes two optimizer updates and every digest matches |
| CI-02 | VIOLATED_ASSUMPTION | The latest pointer names objects that no longer exist |
| CI-03 | NUMERICAL_EDGE | A quota or invoice disagrees by 7% because GiB and GB were swapped |
| CI-04 | CONCEPT_CONFUSION | A weight export is consumed as a restartable checkpoint |
| CI-05 | SCALE_FAILURE | Run MTBF is taken from the per-node figure and the schedule is 1.5x too long |
| CI-06 | NUMERICAL_EDGE | Mean of per-rank rates replaces ratio of sums and overstates throughput |
| CI-07 | SCALE_FAILURE | A shard count chosen for flexibility makes the write request-bound |
| CI-08 | CONCEPT_CONFUSION | A digest match is accepted as proof of consistency |
| CI-09 | VIOLATED_ASSUMPTION | The prefetch cursor is saved as the consumption cursor |
| CI-10 | NUMERICAL_EDGE | Three step conventions coexist and the resume is off by an accumulation window |
| CI-11 | CONCEPT_CONFUSION | Tokens per second is quoted without saying which tokens were counted |
| CI-12 | SCALE_FAILURE | KV cache is sized from mean sequence length and OOMs at the batch tail |
| CI-13 | NUMERICAL_EDGE | Padding multiplies prefill cost and vanishes from the per-token bill |
| CI-14 | VIOLATED_ASSUMPTION | An online feature is computed differently from its offline definition |
| CI-15 | CONCEPT_CONFUSION | Offline AUC is reported as an online uplift estimate |
| CI-16 | NUMERICAL_EDGE | Precision collapses when prevalence falls at a fixed threshold |
| CI-17 | SCALE_FAILURE | A retry storm during failover multiplies load on the recovering tier |
| CI-18 | VIOLATED_ASSUMPTION | The autoscaler tracks CPU while the accelerator is saturated |
| CI-19 | NUMERICAL_EDGE | p99 is computed from per-minute means and the tail disappears |
| CI-20 | NUMERICAL_EDGE | RTO is measured to first byte of load rather than to steady state |
| CI-21 | VIOLATED_ASSUMPTION | Count-based retention deletes a delta chain's base |
| CI-22 | SCALE_FAILURE | Cold image pull and queue wait land inside the restore budget |
| CI-23 | NUMERICAL_EDGE | Learning rate is not rescaled after reshard changes effective batch |
| CI-24 | CONCEPT_CONFUSION | Cache hit rate is averaged across tenants with unequal traffic |
| CI-25 | VIOLATED_ASSUMPTION | Replay duplicates an external side effect that had already advanced |
| CI-26 | SCALE_FAILURE | The checkpoint writer saturates a shared fabric for other tenants |
| CI-27 | NUMERICAL_EDGE | Loss scale sits at the fp16 overflow boundary and clips silently |
| CI-28 | CONCEPT_CONFUSION | Accelerator memory is planned from parameter bytes alone |
| CI-29 | VIOLATED_ASSUMPTION | Evaluation data overlaps the training shard and nothing flags it |
| CI-30 | NUMERICAL_EDGE | Cost per token mixes input and output prices and pads the prompt |

Two conventions hold for every entry. The discriminating check must produce a *different observable* under each
competing cause; where that is impossible, the entry says so and recommends escalation instead of inventing a
check. And every verification names a boundary record next to the triggering record, because a correction
validated only on the trigger has usually been narrowed to fit it.

### 8.1 CI-01 — a generation mixes two optimizer updates

| Field | Content |
|---|---|
| ID | CI-01 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A resume from a fresh generation produces a loss curve that diverges from the pre-failure trace |
| Competing causes | (a) no global barrier before the read; (b) a barrier per group missing the pipeline stage; (c) async writer reading a buffer the optimizer then mutates |
| Discriminating check | Compare the step counter recorded inside each shard's metadata across all shards; two or more distinct values is a torn set |
| Correction | Re-take the generation behind an explicit global barrier, then replay two updates and compare loss against the recorded values |
| Prevention | Put the step number in the object path so a torn set produces two names that cannot assemble into one manifest |
| Verification | Repeat with a writer deliberately delayed by 200 ms on one rank; the guard must refuse to publish |

The mechanism is a read without a happens-before edge. The optimizer's update on rank 0 completes, the writer
reads rank 0's parameters, and rank 3's update lands while rank 1's shard is being read. Nothing in the object
store objects: each shard is a valid serialisation of a valid tensor.

The subtlety that makes (c) different from (a) and (b) is that an async writer can be *correctly* synchronised
and still torn, because the barrier only guarantees that copies start at the same step. If the copy takes 40 s and
the trainer advances during it, the buffer has moved on. The discriminator is whether the writer's own copy took
any time: a copy that completes in one call with no trainer progress cannot be (c).

Correction note: re-taking the generation is the visible fix, but the real fix is the barrier's scope. A barrier
on the data-parallel group does not include the pipeline group, and a pipeline stage holds a distinct set of
layers, so a torn read across stages produces a model whose forward pass mixes two updates. The check that finds
this is to compare step counters across *stages*, not across ranks within a stage — a distinction that costs one
extra query and finds a defect most teams ship.

Verification boundary: the delayed-writer drill is the important half. A guard tested only on a clean write can
be a guard that checks the manifest after the fact, which is exactly the case that passes under both the right
and the wrong implementation.

### 8.2 CI-02 — the pointer names objects that no longer exist

| Field | Content |
|---|---|
| ID | CI-02 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | Recovery starts, then fails partway through loading with a not-found on a named shard |
| Competing causes | (a) upload completion reported before durability; (b) a lifecycle rule expiring objects early; (c) an eventual-consistency listing at publish time |
| Discriminating check | From a fresh client with no cache, list every object the manifest names and compare count and total bytes; repeat after a delay |
| Correction | Require the store's durability acknowledgement before the pointer write; republish from a generation that validates |
| Prevention | A publication gate that validates the manifest from an independent client before the pointer moves |
| Verification | Expire one object via a lifecycle rule in a test bucket and confirm the gate refuses rather than publishes |

Mechanism (a) is the one that surprises people, because the writer did nothing wrong by its own contract: it
awaited a completion callback, and the callback fired when the bytes reached a cache. The gap between "written"
and "durable" is provider-specific and must be read from the provider's consistency documentation rather than
assumed, which is why this entry states the behaviour as an assumption to verify.

Distinguishing (b) from (a) is cheap: (b) leaves a lifecycle event in the bucket's audit log with a rule name,
and (a) leaves nothing. Distinguishing (c) is the delay retest — a listing that becomes correct after a short
wait is propagation, not loss. All three produce identical restore errors, and the audit log is what separates
them.

Correction detail worth keeping: the gate should validate the *manifest's* claims rather than the *bucket's*
listing, because the manifest is what recovery will read. A gate that checks "the bucket has 8,192 objects" and
a recovery that reads "the 8,192 objects the manifest names" can disagree, and the disagreement is the defect.

### 8.3 CI-03 — GiB and GB swapped near a boundary

| Field | Content |
|---|---|
| ID | CI-03 |
| Category | NUMERICAL_EDGE |
| Symptom | A checkpoint exceeds a quota that the capacity plan said it fitted under, or a bill is 7% above forecast |
| Competing causes | (a) binary units in the plan, decimal in the quota; (b) the quota counts replicas; (c) the quota counts snapshots and metadata |
| Discriminating check | Convert the same state to both units and compare against the quota's own documentation of its unit |
| Correction | State the unit next to every capacity number and convert at the boundary, once, in one place |
| Prevention | A units-aware type or a lint rule that rejects a bare number in a capacity config |
| Verification | Re-run the size report and the quota check on a generation sized just under the limit |

The magnitude is 7.4% per boundary, which is small enough to look like noise and large enough to cross a hard
limit. 912.7 GiB and 980.0 GB are the same bytes, and a 950 GB quota accepts one and rejects the other depending
on which arithmetic the quota applies.

Cause (b) is worth separating because it is not a units defect at all: three replicas of 912.7 GiB is 2.67 TiB,
and a plan that priced one generation has understated by 3x, not by 7%. The discriminator is to divide the
observed usage by the single-generation size: a ratio near 1.07 points to units, a ratio near an integer points
to copies.

Cause (c) is the one that bites in object stores: the manifest lists every object with its own metadata, and
8,192 shards at 3 objects each is 24,576 metadata entries whose names and sizes are themselves stored. The
overhead is small in bytes and not small in object count, so a quota expressed in objects and a plan expressed in
bytes cannot be compared at all.

Verification boundary: size the test generation to within 2% of the limit. A test at half the limit passes under
both unit conventions and proves nothing.

### 8.4 CI-04 — a weight export is consumed as a restartable checkpoint

| Field | Content |
|---|---|
| ID | CI-04 |
| Category | CONCEPT_CONFUSION |
| Symptom | A resumed run trains, but its loss is offset from the original trace and the schedule restarts |
| Competing causes | (a) the artifact genuinely lacks optimizer and cursor state; (b) the state exists but the loader skips it; (c) the scheduler is restored but stepped once at load |
| Discriminating check | Diff the keys present in the artifact against the keys the trainer consumes; count tensors and scalars per key group |
| Correction | Publish a state-complete generation for resume and keep the export as a separate, clearly named artifact |
| Prevention | Two artifact types with different names and different schemas, so the wrong one cannot be loaded by accident |
| Verification | Load both artifacts into a test trainer and confirm the state-complete one yields a first-step loss within tolerance |

For the model in the companion sheet, parameters are 130.4 GiB of 912.7 GiB, so a weight export carries 14.3% of
the state. The missing 85.7% is not padding: Adam's update divides by the square root of a running second moment,
and initialising that moment to zero turns the first updates after resume into a different optimisation problem.

Distinguishing (b) from (a) is a key count, and it is decisive: if the artifact contains `optimizer.state.*`
keys and the loader does not request them, the defect is in the load path — a one-line fix with a large effect.
Cause (c) is distinguishable by exactly one step of difference in the schedule, which shows up as a small
learning-rate discontinuity rather than a loss offset, and it is found by printing the scheduler state before and
after the first resumed step.

The naming point matters more than it sounds. Teams that keep one artifact type end up with a loader that
"mostly" works and a runbook that says "restore from the latest checkpoint, unless it is a small one". A second
name costs nothing and removes the conditional from the runbook.

### 8.5 CI-05 — run MTBF taken from the per-node figure

| Field | Content |
|---|---|
| ID | CI-05 |
| Category | SCALE_FAILURE |
| Symptom | The observed failure rate is far above the rate used to choose the checkpoint interval |
| Competing causes | (a) hazards add across independent nodes; (b) failures are correlated within a rack or power domain; (c) the count includes preemptions, which are not failures |
| Discriminating check | Group failure events by minute and by node; clustering indicates (b), uniform spacing indicates (a), and event metadata identifies (c) |
| Correction | Recompute the interval from the measured run-level hazard, then re-derive the schedule |
| Prevention | Store the run-level hazard as a monitored quantity, not a copied constant from a vendor sheet |
| Verification | Compare predicted against observed failure count over one week at the new interval |

The arithmetic is a division: 720 h of per-node MTBF across 512 nodes is 1.406 h of run MTBF, and the interval
scales as the square root of that, so the correct interval is 1.46x shorter than the one derived from the node
number. An interval that is 1.46x too long costs roughly two percentage points of goodput at the sizes in the
worked example, and considerably more when the hazard is higher.

Cause (c) is a genuine confusion rather than a scale effect, and it is worth separating because preemption is
scheduled, not stochastic: a cluster that preempts at a fixed 24-hour boundary produces one large loss per day at
a predictable time. The correct response is a checkpoint pinned before the boundary, not a smaller interval.

Correlated failures (b) break the model rather than the arithmetic. When 64 nodes fail together, the count of
events is one and the count of lost ranks is 64; a rate computed from events understates the recovery cost,
while a rate computed from lost ranks overstates the frequency. The right pair of statistics is events per hour
and ranks lost per event, and they answer different questions.

### 8.6 CI-06 — mean of per-rank rates replaces the pooled rate

| Field | Content |
|---|---|
| ID | CI-06 |
| Category | NUMERICAL_EDGE |
| Symptom | A throughput figure predicts a transfer that takes several times longer than forecast |
| Competing causes | (a) averaging intensive quantities over unequal exposure; (b) the max-over-ranks latency governs completion; (c) units differ between the per-rank and aggregate figures |
| Discriminating check | Sum bytes, sum time, divide; then compare with the mean of rates. A gap proves (a) |
| Correction | Report total bytes over total elapsed time for throughput, and max over ranks for latency |
| Prevention | Two named metrics with different units, so one cannot be substituted for the other |
| Verification | Recompute on a deliberately skewed two-group case where the answer is known by hand |

The worked example makes the size of the error concrete: 480 ranks at 16 MiB/s and 32 at 4 MiB/s average to
15.25 MiB/s, implying 119.7 s, while the commit actually completes when the slowest rank finishes at 456.4 s.
The mean is 4.6x optimistic about the slow group's share.

Cause (b) is the deeper one, because even a correctly pooled throughput does not predict commit latency. A store
that ingests 8 GiB/s aggregate can still have one rank 200 ms behind, and the commit is a barrier, so latency is
a maximum. The pair of statements — pooled for capacity, max for schedule — is the whole correction, and neither
number alone answers both questions.

Cause (c) appears when per-rank figures are quoted in MiB/s and the aggregate in Gbit/s: 16 MiB/s times 512
ranks is 8,192 MiB/s = 8.0 GiB/s = 68.7 Gbit/s, and a 100 Gbit/s link budget looks adequate right up until the
protocol overhead and the shared tenant are counted.

### 8.7 CI-07 — the shard count makes the write request-bound

| Field | Content |
|---|---|
| ID | CI-07 |
| Category | SCALE_FAILURE |
| Symptom | Checkpoint time grows after a shard-count increase even though bytes did not change |
| Competing causes | (a) per-object request cost exceeds the fair share; (b) metadata operations scale with object count; (c) the store throttles per prefix |
| Discriminating check | Compute objects divided by observed requests per second for the window and compare with bytes divided by bandwidth |
| Correction | Reduce objects per shard, or batch small tensors into fewer objects, or raise the fair share deliberately |
| Prevention | A shard-count review that computes the request budget before approving the count |
| Verification | Time a write at the new count and confirm the request term is under 50% of the byte term |

The arithmetic is a ratio of two floors: 24,576 objects at 1,500 requests/s is 16.4 s, negligible against
114 s of transfer. At 32,768 shards and 20 objects each, 655,360 objects at the same share is 437 s, and the
checkpoint becomes dominated by request handling that transfers no payload worth counting.

Cause (c) is distinct and common: many stores partition throttling by key prefix, so writing all shards under
one prefix serialises against a single partition's limit even when the account's total limit is far higher. The
discriminator is a test that writes the same object count under one prefix and then under 64 prefixes; a large
difference confirms prefix-level throttling, and the correction is a hashed prefix, not a smaller object count.

The reason this is a scale failure rather than a numerical edge is that the defect appears only above a count.
At 512 shards the design is fine and the arithmetic is fine; at 8,192 it is neither, and no single-object test
will show it.

### 8.8 CI-08 — a digest match is accepted as proof of consistency

| Field | Content |
|---|---|
| ID | CI-08 |
| Category | CONCEPT_CONFUSION |
| Symptom | Validation reports success and the resumed model behaves as if restored from a different step |
| Competing causes | (a) digests prove integrity, not membership; (b) the digest was computed over a re-serialised object; (c) the manifest itself was assembled from two generations |
| Discriminating check | Compare the step counter inside each shard against the manifest's declared step; digests cannot answer this |
| Correction | Add a generation and step field to each shard and validate membership, not only bytes |
| Prevention | Compute and validate digests and membership as two separate gate steps, both required |
| Verification | Inject a shard from the previous generation with a matching digest and confirm the membership gate rejects it |

The conceptual point is that three properties are often collapsed into the word "valid": the bytes are as
written (integrity), the set is the intended set (membership), and the state is sufficient for the next update
(sufficiency). A digest answers the first only, and it answers it perfectly, which is why the other two get
skipped.

Cause (b) is a real trap for objects that are re-encoded on read: a digest computed at write time over a
compressed stream does not match a digest computed at read time over the decompressed payload. The correct
practice is to digest the stored bytes, not the logical tensor, and to say so in the manifest's schema.

Cause (c) is the manifest assembly path, which is code most teams write once and never test. The check is to
assemble a manifest from a deliberately mixed directory and require the assembler to fail; an assembler that
succeeds is the defect, and it is found in seconds with a fixture.

### 8.9 CI-09 — the prefetch cursor is saved as the consumption cursor

| Field | Content |
|---|---|
| ID | CI-09 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A resumed run skips a contiguous block of data that no one can find in the loss curve |
| Competing causes | (a) loader position saved instead of consumed position; (b) the sampler reseeded, changing order entirely; (c) shard assignment changed, so a different rank reads a different range |
| Discriminating check | Record the consumed index at the checkpoint and compare with the restored start index; a constant offset equals workers times depth |
| Correction | Save the last fully consumed index and re-derive loader positions at restore |
| Prevention | The cursor is written by the consumer, not by the loader, so the two cannot be confused |
| Verification | Restore with prefetch depth doubled and confirm the resume start is unchanged |

The size of the hole is a configuration product, not a bug in the ordinary sense: 8 workers at depth 4 with
1,024 sequences per batch and 4,096 tokens per sequence is 134.2 million tokens that were fetched, never
consumed, and never trained on. Nothing errors, nothing warns, and the model simply has a gap.

Distinguishing (a) from (b) is the offset's shape. A prefetch error produces a constant, reproducible skip; a
reseed produces a completely different order with no structure, visible as a loss discontinuity that does not
repeat when the same restore is attempted twice. Distinguishing (c) is per-rank: compare the index range each
rank reads before and after; a permutation with the same union points to (c), a truncated union points to (a).

The prevention is structural rather than procedural: if the consumer writes the cursor, there is no loader
position to accidentally save. Any design in which the writer can see the loader's internal state will
eventually save it.

### 8.10 CI-10 — three step conventions coexist

| Field | Content |
|---|---|
| ID | CI-10 |
| Category | NUMERICAL_EDGE |
| Symptom | A resume lands a few hundred steps away from where the run stopped, or the schedule restarts mid-decay |
| Competing causes | (a) microbatches counted as steps; (b) skipped updates counted; (c) accumulation phase lost so the first update after resume uses a partial window |
| Discriminating check | Print all three counters — microbatches, updates, scheduled steps — at the checkpoint and after the first resumed update; exactly one should advance by one |
| Correction | Store all three with explicit names and assert the expected relations at load |
| Prevention | A single `Progress` record type with named fields and a validation function |
| Verification | Restore mid-accumulation and confirm the next update consumes the remaining microbatches |

The arithmetic of cause (a) is a multiplication: with effective batch 16,384 = 512 ranks * 8 per rank * 4
accumulation, a counter in microbatches is 4x the update count, so "step 12,000" is update 3,000 under one
convention and update 12,000 under another. Every downstream schedule position differs by 4x, and the symptom is
a learning rate that jumps.

Cause (c) is the subtler one, and it is why the phase belongs in the checkpoint. If the checkpoint lands after
two of four microbatches, the accumulated gradients are either saved (and the phase resumes at 2) or discarded
(and the phase resets at 0). Saving them costs state; discarding them costs a partial window's work and slightly
changes the effective batch for one update. Both are defensible, and neither is defensible by accident.

Verification boundary: test at phase 3 of 4, not at phase 0. A checkpoint taken only at update boundaries never
exercises the accumulation logic, which is the same as not having it.

### 8.11 CI-11 — tokens per second without a token definition

| Field | Content |
|---|---|
| ID | CI-11 |
| Category | CONCEPT_CONFUSION |
| Symptom | Two teams report throughput for the same service that differ by 3x and both are defensible |
| Competing causes | (a) prompt tokens counted in one figure and not the other; (b) per-request versus per-token framing; (c) padding and cache reads billed differently than counted |
| Discriminating check | Recompute from raw counters: sum prompt tokens, sum generated tokens, sum requests, and divide by the same elapsed window |
| Correction | Report prompt and generated tokens separately, per second, with the window stated |
| Prevention | The metric name carries the definition, e.g. `generated_tokens_per_s` |
| Verification | Recompute a second day from raw counters and confirm the published figure reproduces |

A service that processes 1,000-token prompts and emits 100 tokens does 1,100 token-units of work per request but
produces only 100 generated tokens, and the prefill work is far more parallel than decode. A figure that blends
them into "tokens per second" cannot be compared with a figure that counts only decode, and the ratio between the
two is roughly the prompt-to-output ratio.

Cause (b) is often a units-of-business mismatch rather than an error: a product team counts requests because
that is what it sells, and a platform team counts tokens because that is what it costs. Both numbers are
correct, and a dashboard that puts them in one column invites a false comparison. The fix is two columns with
different names.

Cause (c) is where padding enters: a 4,096-token padded batch with 300 real tokens moves 13.65x the prefill
compute of the real tokens, and if the provider bills the padded length, the cost per useful token is 13.65x the
cost per billed token. CI-13 is the same mechanism seen from the cost side.

### 8.12 CI-12 — KV cache sized from mean sequence length

| Field | Content |
|---|---|
| ID | CI-12 |
| Category | SCALE_FAILURE |
| Symptom | The serving tier OOMs under a load it handled the previous week, with no configuration change |
| Competing causes | (a) cache sized on mean length while admission uses fixed concurrency; (b) fragmentation from variable lengths; (c) cache not freed on early termination |
| Discriminating check | Compute 2 * L * n_kv * d_head * b * length * concurrency for the observed maximum, not the mean |
| Correction | Admit by token budget rather than by request count, and preempt or queue when the budget is exhausted |
| Prevention | Admission control expressed in cache bytes, so the limit is the quantity that actually runs out |
| Verification | Drive the maximum observed length and concurrency simultaneously and confirm no OOM |

The arithmetic from F3: 320 KiB per token at 80 layers, 8 KV heads, head dimension 128, and 2 bytes. A mean
length of 800 tokens at 16 concurrent sequences needs 4.0 GiB, and the same service at 2,048 tokens and 32
sequences needs 20.0 GiB. The change is entirely in the input distribution, which is why "no configuration
change" is true and irrelevant.

Cause (b) is a different mechanism with the same symptom: a paged allocator fragments less than a contiguous
one, but a cache with mixed 200-token and 4,000-token sequences leaves unusable gaps if blocks are fixed-size and
large. The discriminator is the ratio of allocated to used bytes; a high ratio with modest load points to
fragmentation rather than demand.

Cause (c) is the cheapest to fix and the easiest to miss: a cancelled request that leaves its cache entry behind
consumes capacity until eviction, and a client that retries on timeout doubles the leak. The check is cache bytes
at steady state versus cache bytes after a burst of cancellations, with no traffic in between.

### 8.13 CI-13 — padding multiplies prefill cost invisibly

| Field | Content |
|---|---|
| ID | CI-13 |
| Category | NUMERICAL_EDGE |
| Symptom | GPU time per request is far above the estimate derived from the token bill |
| Competing causes | (a) static padding to a maximum length; (b) attention is quadratic in padded length; (c) the bill is per generated token while compute is per padded token |
| Discriminating check | Compare L_mean with the L_max the kernel executes for the same requests; compute the multiplier |
| Correction | Bucket by length or use continuous batching so L_max tracks the batch, not the configured maximum |
| Prevention | The utilisation ratio is a serving metric, reviewed with the latency dashboard |
| Verification | Re-measure time per request after bucketing at the same throughput and confirm the improvement |

With L_mean = 300 and L_max = 4,096 the utilisation is 7.32% and the multiplier 13.65. Applied to a prefill
term of 0.3 s that is 4.1 s of compute for 0.3 s of work — a number that looks like a hardware problem and is a
batching problem.

Cause (b) compounds it, because attention cost includes a term in the square of the sequence length. Padding from
300 to 4,096 scales that term by 186x, not 13.65x, and the quadratic term is a minority at short lengths and a
majority at long ones. The diagnostic is to plot time against L_max at fixed L_mean; a superlinear slope identifies
the quadratic term.

Cause (c) is a billing-versus-compute mismatch worth stating plainly: if the provider bills generated tokens,
padding is free to the invoice and expensive to the GPU, so a cost review based on the invoice will not see it.
The metric that reveals it is GPU-seconds per generated token, which is why that metric belongs in the FinOps
review rather than only in the performance review.

### 8.14 CI-14 — an online feature differs from its offline definition

| Field | Content |
|---|---|
| ID | CI-14 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | Offline validation is strong and online performance is at chance |
| Competing causes | (a) the online transform computes a different statistic; (b) the online value is stale beyond the training window's assumption; (c) the training join used a later partition than serving can see |
| Discriminating check | Log the raw inputs and the computed feature for 1,000 requests, then recompute offline from the same raw inputs and diff |
| Correction | Serve from the same transform code path, or accept and quantify the divergence |
| Prevention | A parity test that runs the offline and online transforms over identical inputs in CI |
| Verification | Replay a day of logged raw inputs through both paths and require exact agreement on the sampled rows |

Mechanism (a) is the classic: a training pipeline computes a 30-day average with a window boundary at midnight
UTC, and the serving path computes a 30-day average with a boundary at request time. Both are "30-day averages",
the values differ by up to a day of data, and the difference is largest exactly where the feature matters.

Cause (b) is a freshness assumption rather than a formula: a feature labelled "real time" may be refreshed every
fifteen minutes, and if the model's training data assumed per-event updates, the served distribution is shifted
in a way no aggregate monitor sees. The check is the age distribution of the feature at decision time, not its
availability.

Cause (c) is a temporal join defect on the platform side: if the training join takes yesterday's partition
because today's was late, the model has learned from values that serving cannot reproduce. The discriminator is
the cut-off comparison in the feature's own metadata, which is why that metadata is not optional.

### 8.15 CI-15 — offline AUC reported as an online uplift

| Field | Content |
|---|---|
| ID | CI-15 |
| Category | CONCEPT_CONFUSION |
| Symptom | A model with a strong ranking metric produces no measurable business change in a live test |
| Competing causes | (a) ranking quality is not the decision the product makes; (b) the threshold and action are fixed, so only calibration matters; (c) the live population is shifted from the validation population |
| Discriminating check | Compute the metric at the deployed threshold and action, on the deployed population, then compare with the business decision's own counterfactual |
| Correction | Report the metric the action depends on, next to the ranking metric, both at the operating point |
| Prevention | Every model card names the metric, threshold, population, and the decision it informs |
| Verification | Re-run the evaluation at a threshold perturbed by one step and confirm the metric moves as the action does |

AUC is a property of the ranking over all thresholds. A product that flags the top 1% by score never uses the
ranking below its cut, and its outcome depends on precision in that band — a quantity AUC does not report. A
model can improve AUC while worsening precision at the operating point, and that is not a paradox; it is two
different quantities with one nickname.

Cause (b) is the calibration version: if the action is "call everyone above 0.5", then the ordering above 0.5 is
irrelevant and the probability at 0.5 decides. A model with excellent ordering and poor calibration at 0.5 has
no effect, and the metric to look at is reliability in the band, not discrimination overall.

Cause (c) is CI-16 arriving through a different door, and the correction is the same: state the population.

### 8.16 CI-16 — precision collapses when prevalence falls

| Field | Content |
|---|---|
| ID | CI-16 |
| Category | NUMERICAL_EDGE |
| Symptom | Post-launch precision is a fraction of validation precision with an unchanged model and threshold |
| Competing causes | (a) prevalence shifted; (b) labels are delayed and the early window is biased; (c) the threshold was retuned implicitly by a preprocessing change |
| Discriminating check | Compute prevalence in the scored population and in the validation population; apply F2 to predict PPV at the new prevalence |
| Correction | Retune the threshold to the operating prevalence, or change the action to tolerate low precision |
| Prevention | Precision is reported as a function of prevalence, not as a single number |
| Verification | Recompute PPV on a stratified sample that restores the validation prevalence |

From F2, sensitivity 0.90 and specificity 0.95 give PPV 0.710 at 12% prevalence and 0.179 at 1.2%. The model did
not change; the population did, and the collapse is arithmetic rather than empirical. Predicting it in advance
takes one line and prevents a month of model debugging.

Cause (b) is a real confound and must be separated before concluding (a): if labels arrive 14 days late, the
early post-launch window contains only fast-labelled cases, which are systematically easier or harder than
average. The check is to compare the prevalence in the labelled subset against the prevalence in the full scored
set; a gap identifies the bias.

Cause (c) is a deployment defect: a change in a preprocessing step can shift the score distribution so that a
fixed threshold selects a different share. The discriminator is the flag rate — unchanged prevalence with a
changed flag rate is (c), and unchanged flag rate with a changed prevalence is (a).

### 8.17 CI-17 — a retry storm multiplies load on the recovering tier

| Field | Content |
|---|---|
| ID | CI-17 |
| Category | SCALE_FAILURE |
| Symptom | A brief failure becomes a long outage; the recovering service is immediately at several times its normal load |
| Competing causes | (a) unbounded client retries with no backoff; (b) synchronized retry timers; (c) load balancer health checks that amplify |
| Discriminating check | Plot request rate during recovery against the pre-failure baseline; a multiple above 1 proves amplification, and its timing pattern separates (a) from (b) |
| Correction | Add jittered exponential backoff, a retry budget, and admission control that rejects rather than queues |
| Prevention | Retry policy is a platform default, not a per-client choice, and is load-tested |
| Verification | Kill one replica under full load and confirm total request rate after recovery stays within 1.2x baseline |

The amplification factor is `1 / (1 - p)` for a single retry layer with failure probability `p`: at `p = 0.5`
that is 2x, at `p = 0.8` it is 5x. Two layers multiply, so a client retrying at 3 attempts against a service that
itself retries downstream turns one request into nine at the deepest tier — and the deepest tier is the one least
able to absorb it.

Cause (b) is the one that turns a short failure into a long one: if every client retries at the same 1-second
boundary, the recovering service sees a synchronized pulse that fails again, because the pulse exceeds capacity
by construction. Jitter is not a refinement; it is the mechanism that converts a pulse into a ramp.

Cause (c) is often overlooked: a health check with a short timeout and a low failure threshold removes a
recovering instance before it finishes initialising, which increases load on the remaining ones, which increases
their latency, which triggers more removals. The discriminator is whether removed instances were actually healthy
a second later.

### 8.18 CI-18 — the autoscaler tracks CPU while the accelerator saturates

| Field | Detail |
|---|---|
| ID | CI-18 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | Latency degrades while the autoscaler reports healthy utilisation and does not scale |
| Competing causes | (a) the scaling metric is CPU but the bottleneck is accelerator memory or compute; (b) queue depth is the real signal and is not exported; (c) scaling is fast but capacity acquisition is slow |
| Discriminating check | Compare the scaling metric against the resource that saturates first, measured in the same window |
| Correction | Scale on the binding signal — queue wait or accelerator memory — with CPU retained as a secondary |
| Prevention | The scaling policy names the saturating resource and is reviewed when the workload changes |
| Verification | Drive load to the point of degradation and confirm a scale-out event fires before latency breaches the objective |

The mechanism is a metric that is a proxy for the workload at one stage of its life and not at another. A
preprocessing-heavy service is CPU-bound and scales sensibly on CPU; the same service after a kernel upgrade does
its work on the accelerator and shows 15% CPU at saturation. The autoscaler is not broken; it is measuring a
resource that stopped being the constraint.

Cause (b) is about signal availability rather than choice: queue wait is the honest signal for a batched service,
and many stacks do not export it, so the policy uses what exists. The correction is to export the signal, which is
work, and the entry is a scale failure because the absence of the signal becomes visible only under load.

Cause (c) is a timing mismatch: if scaling decisions take 30 s and new capacity takes four minutes to become
ready, the policy's benefit arrives long after the burst. The fix is a warm pool sized to the observed burst
shape, and the diagnostic is the distribution of burst durations — bursts shorter than the acquisition time
cannot be served by scaling at all.

### 8.19 CI-19 — p99 computed from per-minute means

| Field | Content |
|---|---|
| ID | CI-19 |
| Category | NUMERICAL_EDGE |
| Symptom | A dashboard reports a p99 well below the slowest requests that users describe |
| Competing causes | (a) percentile of pre-aggregated means; (b) histogram buckets too coarse at the tail; (c) a different population, e.g. server-side only, excluding queue time |
| Discriminating check | Recompute the percentile from raw request records for the same window and the same population definition |
| Correction | Compute percentiles from raw observations or from a fine-grained histogram, never from means |
| Prevention | The metric name states the aggregation level, and the raw path exists for arbitration |
| Verification | Inject a known 0.5% of 4-second requests and confirm the reported p99 approaches 4 s |

The collapse is structural. Ten requests, nine at 1 s and one at 4 s, have a true p99 of 4 s; the per-minute mean
is 1.3 s, and a p99 of one-minute means is 1.3 s. The tail is not lost by a bug but by an operation that is
mathematically a contraction toward the centre.

Cause (b) is a coarser version of the same defect: a histogram with buckets at 1, 2, 4, 8 s cannot report a p99
better than the bucket width, and interpolation within a bucket is an assumption, not a measurement. For an
objective stated as "p99 under 800 ms", a histogram whose buckets are 1 s apart cannot demonstrate compliance at
all.

Cause (c) is a population question and is often the actual disagreement between teams: server-side latency
excludes queueing and client retries, and a user's experience includes both. Two correct numbers describing
different intervals is a definition gap, and the fix is to publish both under different names.

### 8.20 CI-20 — RTO measured to first byte instead of steady state

| Field | Content |
|---|---|
| ID | CI-20 |
| Category | NUMERICAL_EDGE |
| Symptom | Drills report a comfortable restore time and real incidents take several times longer |
| Competing causes | (a) the timer stops at first load byte; (b) warm-up and compile are excluded; (c) the drill used a warm cache and the production path did not |
| Discriminating check | Time the five phases — detect, schedule, fetch, load, warm — separately and sum them; compare with the objective's wording |
| Correction | Redefine the measurement to end at steady-state throughput and re-baseline |
| Prevention | The drill emits per-phase timings as structured events, not one total |
| Verification | Re-run the drill against the second-newest generation with the production identity |

The composition matters more than the total: in the worked example, fetch is 152 s of 572 s, so a measurement
that stops at first byte reports 242 s and a storage upgrade that halves fetch takes the reported number to
166 s while the operational number moves to 496 s. The metric improves 31% and the system improves 13%.

Cause (b) is where graph capture and kernel selection live, and they are not small: a 90 s warm-up on a 572 s
restore is 16%, and on a 200 s restore it is 45%. Excluding it makes the objective look met by a system that
resumes training at a third of its steady-state rate for the first minute.

Cause (c) is a drill-validity issue rather than a definition one, and it is the most common reason a drill
number is optimistic. The correction is to select the generation by policy — oldest retained, or a random
generation from the last day — never by recency.

### 8.21 CI-21 — count-based retention deletes a delta chain's base

| Field | Content |
|---|---|
| ID | CI-21 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | A restore fails on a single missing object, and several generations are all unusable |
| Competing causes | (a) count-based retention on a chain; (b) a manual cleanup that ignored dependencies; (c) a delta that silently promoted itself to a base without recording the new chain |
| Discriminating check | Resolve the dependency closure of every retained generation and diff against the retained set |
| Correction | Restore the base from an archive if available; otherwise mark the affected generations unusable and record the loss |
| Prevention | Retention operates on chain closure and refuses to delete an object with live dependents |
| Verification | Run retention against a synthetic chain of 12 deltas and confirm the closure is preserved |

The mechanism is a policy that reasons about objects while the system's guarantee is about chains. Keeping the
newest eight generations by creation time is a clear, auditable rule that is wrong the moment deltas exist,
because the eighth-newest generation may be a delta whose base is the twelfth-newest object.

Cause (c) is the one that hides: a compaction job that rewrites a chain into a new base must record the new base
in every child's metadata. If it records the new base in the base's own metadata only, the children still point
at the deleted parent, and the failure appears weeks later at the next restore.

The cost of the defect is asymmetric in a way worth stating: storage saved is linear in chain depth, while
generations lost are all of them above the deleted base. A retention policy that saves 26% of storage and
occasionally loses a day of training has a poor expected value, and the arithmetic that shows it is the closure
computation, not an argument about storage prices.

### 8.22 CI-22 — cold image pull and queue wait inside the restore budget

| Field | Content |
|---|---|
| ID | CI-22 |
| Category | SCALE_FAILURE |
| Symptom | A node replacement that should take a minute takes eight, and the cause is not the checkpoint |
| Competing causes | (a) image pull from a cold registry cache; (b) scheduler queue when the cluster is full; (c) container start plus driver initialisation |
| Discriminating check | Time scheduling separately from container start and from image pull using the platform's own events |
| Correction | Pre-pull images on a warm pool, or cache them on the node image |
| Prevention | Recovery capacity is reserved, and images are pre-pulled as part of pool maintenance |
| Verification | Drain a node and measure a replacement's time to a ready process with the store idle |

This is a scale failure because the pull cost is shared: 64 nodes pulling a 12 GiB image through one registry
endpoint is not 64 independent pulls, it is a fan-in against a bandwidth limit. At 2 GiB/s of shared registry
egress, 64 * 12 GiB = 768 GiB takes 384 s of pure transfer, and that time sits inside the RTO whether or not it
is labelled as restore.

Cause (b) is the honest version of the same problem: when every node in the cluster is occupied, a replacement
waits, and the wait is a scheduling decision that the training team does not control. The mitigation is reserved
capacity for recovery, which is a policy purchase and should be priced as such rather than discovered during an
incident.

Cause (c) is small individually and additive across phases, which is why the drill must time phases rather than
totals. A 20 s driver initialisation on each of 64 nodes that all start together is not 20 s of wall clock if
the initialisation contends for a shared resource; the aggregate is what the incident shows.

### 8.23 CI-23 — learning rate not rescaled after effective batch changes

| Field | Content |
|---|---|
| ID | CI-23 |
| Category | NUMERICAL_EDGE |
| Symptom | A run resharded to a different rank count trains unstably, or converges more slowly, with the same config |
| Competing causes | (a) effective batch changed with rank count; (b) a scheduler stepped by rank count rather than update; (c) gradient accumulation changed simultaneously |
| Discriminating check | Compute effective batch = ranks * per-rank * accumulation before and after; compare with the update count per wall-clock hour |
| Correction | Hold effective batch constant by adjusting accumulation, or rescale the learning rate by the square root of the ratio |
| Prevention | The launcher asserts effective batch against the config, and reshard is a reviewed change |
| Verification | Run 200 updates at both configurations from the same generation and compare loss trajectories |

The arithmetic is a multiplication that changes silently. 512 ranks at 8 per rank and 4 accumulation is 16,384;
halving ranks with everything else fixed is 8,192, so each update sees half the gradient signal and the noise in
the update doubles as a fraction of its mean. The square-root rule `lr_new = lr_old * sqrt(0.5) = 0.707 * lr_old`
is the common heuristic, and it is a heuristic — the entry's point is that the batch changed, not that 0.707 is
right.

Cause (b) is a defect rather than a decision: a scheduler stepped once per rank-advance increments 512 times per
update before the reshard and 256 after, so the learning rate decays twice as fast in wall-clock terms and the
resumed run is effectively on a different schedule. The discriminator is steps-per-update, which should always
be one.

Cause (c) is the combination failure: changing ranks and accumulation together can leave effective batch
unchanged, which is correct, while changing only one is a silent change to the optimisation problem. The check
is the product, not either factor.

### 8.24 CI-24 — cache hit rate averaged across tenants

| Field | Content |
|---|---|
| ID | CI-24 |
| Category | CONCEPT_CONFUSION |
| Symptom | The reported hit rate is high while the largest tenant reports poor latency |
| Competing causes | (a) unweighted mean of per-tenant rates; (b) per-tenant rates are correct but the tenant's traffic changed; (c) eviction is unfair across key populations |
| Discriminating check | Recompute as total hits over total requests across tenants; compare with the mean of rates |
| Correction | Report the pooled rate for capacity and a per-tenant table for fairness |
| Prevention | The metric name distinguishes pooled from per-tenant |
| Verification | Recompute on a two-tenant case with deliberately unequal volume |

With ten small tenants at 90% and one large tenant at 40%, the unweighted mean is (10*0.90 + 0.40)/11 = 0.855,
which reads as healthy. If the large tenant carries 80% of requests, the pooled rate is
(0.8*0.40 + 0.2*0.90) = 0.50, and the difference between 85.5% and 50% is the difference between an incident and
a footnote.

Cause (c) is a mechanism rather than an aggregation defect: a shared cache with a least-recently-used policy
serves the tenant with the highest request rate best, so a large tenant with a diverse key set can have a low hit
rate while small tenants with narrow key sets have high ones. The pooled statistic then depends on the mix,
which is exactly what makes it move without any change to the cache.

The correction is two numbers, not one, and the reason is that they answer different questions: pooled rate
answers "is the cache doing its job for the system", per-tenant rate answers "is the cache fair". A single number
cannot answer both, and a dashboard with one number will be used to answer both anyway.

### 8.25 CI-25 — replay duplicates an external side effect

| Field | Content |
|---|---|
| ID | CI-25 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | After a restore, downstream counters or records show more work than the run actually performed |
| Competing causes | (a) side effects are not keyed, so replay repeats them; (b) the key exists but the sink ignores it; (c) the effect is genuinely at-least-once and the consumer assumes exactly-once |
| Discriminating check | Inspect an effect record for a key derived from run and step; absence proves (a), presence with duplicates proves (b) |
| Correction | Key every effect by `(run_id, step, effect_name)` and make the sink idempotent on that key |
| Prevention | The side-effect client requires a key at the type level, so an unkeyed call cannot compile |
| Verification | Kill a run immediately after a side effect and confirm the replay produces exactly one record |

The mechanism is that a checkpoint restores tensors, not the external world. The interval between the last
commit and the failure is replayed, and every effect emitted during it is emitted twice. For a metrics counter
the cost is a wrong dashboard; for a publish to a queue or a write to a billing system the cost is real.

Cause (b) is a common half-fix: a key is emitted but the sink appends rather than upserts. The discriminator is to
send the same keyed event twice deliberately and observe the result; a sink that stores two rows has answered the
question permanently, and the test belongs in the sink's own suite.

Cause (c) is a contract problem: if the transport is at-least-once, the consumer must be idempotent, and a design
that assumes exactly-once has a defect that appears under retry rather than under load. The honest statement is
that exactly-once delivery is a property of a keyed sink, not of a queue.

### 8.26 CI-26 — the checkpoint writer saturates a shared fabric

| Field | Content |
|---|---|
| ID | CI-26 |
| Category | SCALE_FAILURE |
| Symptom | An unrelated service's p99 triples for two minutes at regular intervals |
| Competing causes | (a) no bandwidth reservation for the writer; (b) burst credits exhausted by an earlier write; (c) a coincident job with the same period |
| Discriminating check | Correlate the latency spikes with per-tenant byte counters; a causal relation shows a byte increase for the writer |
| Correction | Rate-limit the writer to a fair share, or isolate it onto a dedicated endpoint |
| Prevention | Commit scheduling uses a token bucket derived from the tenant's share, and alerts on the writer's share |
| Verification | Run the periodic commit against a loaded test service and confirm its p99 stays within 10% of baseline |

The mechanism is contention that is completely correct behaviour from the training job's point of view: it was
told to write 912.7 GiB, it wrote at the maximum rate available, and the maximum rate available was shared. The
defect is in the scheduling, not in the writer.

Cause (b) is a credits-versus-reservation distinction: a burst allowance lets a job exceed its steady share for a
bounded period, so a checkpoint that starts after a long idle period can consume the whole burst and then be
throttled mid-write, which lengthens the stall and produces a second latency spike. The discriminator is the
write's rate profile — a plateau followed by a cliff is credits, a flat low rate is a hard limit.

Cause (c) is the control experiment, and it must be run before the mitigation: a coincident job with the same
period produces the same correlation. The discriminator is to shift the commit phase by 90 s and see whether the
spike follows; if it does not, the correlation was coincidental.

### 8.27 CI-27 — the loss scale sits at the fp16 overflow boundary

| Field | Content |
|---|---|
| ID | CI-27 |
| Category | NUMERICAL_EDGE |
| Symptom | Loss occasionally becomes NaN or the scale collapses to a small value and training stalls |
| Competing causes | (a) the scale is too high for the observed gradient magnitudes; (b) an inf in one rank's gradient poisons the all-reduce for all; (c) the scale history is not restored, so the first steps after resume overflow |
| Discriminating check | Log the scale and the number of skipped updates per step; a sawtooth with periodic skips is (a), a single catastrophic event is (b) |
| Correction | Reduce the initial scale, add per-rank inf detection before the reduction, and restore the scale with the checkpoint |
| Prevention | The scaler's state is part of the checkpoint schema, and the resume test asserts no skip in the first ten updates |
| Verification | Restore mid-run and confirm the first ten updates complete with zero skips |

The numeric boundary is concrete: fp16's largest finite value is 65,504, so a gradient whose scaled magnitude
exceeds that becomes inf, and one inf in a summed reduction makes the whole reduction inf for every rank. The
"periodic skips" pattern is the scaler doing its job, halving the scale each time until it finds a working value;
the failure is that the halving costs an update each time.

Cause (b) is worth separating because the fix is different and cheap: detecting non-finite values per rank before
the all-reduce identifies which rank produced it, and reduces the blast radius from a lost update on every rank to
a skipped contribution from one. Without it, the log shows an inf with no owner.

Cause (c) is the resume-specific instance: a scale restored to a default rather than to its trained value
overflows on the first step, halves, overflows again, and costs a slow ramp. The check is the skip count in the
first ten updates after restore, which is a one-line assertion in a resume test.

### 8.28 CI-28 — accelerator memory planned from parameter bytes alone

| Field | Content |
|---|---|
| ID | CI-28 |
| Category | CONCEPT_CONFUSION |
| Symptom | A configuration that fits on paper OOMs at the first long sequence or the first large batch |
| Competing causes | (a) activations and workspace omitted; (b) cache or staging buffer omitted; (c) fragmentation prevents a contiguous allocation |
| Discriminating check | Instrument peak allocated versus reserved memory and attribute it by phase: forward, backward, optimizer, transfer |
| Correction | Budget from measured peak per phase, with the largest single allocation reserved explicitly |
| Prevention | The capacity plan lists every term, and a test runs the longest sequence in CI |
| Verification | Run the maximum sequence length and batch in a test and confirm peak stays under the limit by a margin |

The parameter term is the smallest one for many models: 70e9 parameters in bf16 is 130.4 GiB, and the training
state is 912.7 GiB, so a plan built from "the model is 130 GB" understates by 7x before activations are counted.
For inference, a 70e9 model at 2 bytes is 130.4 GiB of weights and, at 320 KiB per cached token, another 20 GiB
for 32 concurrent 2,048-token sequences — a 15% addition that is not in a parameter-only plan.

Cause (c) is the one that produces OOMs with free memory visible: the allocator has enough total bytes and no
contiguous block large enough. The discriminator is the gap between reserved and allocated; a large persistent
gap points to fragmentation, and the correction is a different allocator strategy or a smaller peak allocation.

The general lesson is that memory is a set of terms with different scaling behaviours: weights constant, state
constant, activations linear in batch and quadratic in sequence for attention, cache linear in concurrency. A
plan that records one number cannot be checked, and a plan that records five can be.

### 8.29 CI-29 — evaluation data overlaps the training shard

| Field | Content |
|---|---|
| ID | CI-29 |
| Category | VIOLATED_ASSUMPTION |
| Symptom | Evaluation is strong offline and degrades immediately after deployment, with a large gap between slices |
| Competing causes | (a) literal overlap between evaluation and training records; (b) near-duplicate content across shards; (c) the evaluation set is drawn from a different time window with a different distribution |
| Discriminating check | Hash normalised record content and intersect the two sets; report overlap rate and the top overlapping keys |
| Correction | Deduplicate at the record level before splitting, then re-split and re-evaluate |
| Prevention | The split is produced by the platform, not by the modelling team, and its overlap check is a gate |
| Verification | Recompute the overlap rate on the new split and confirm it is zero by exact hash |

Cause (b) is the hard version: near-duplicates survive exact hashing, so a document with one changed sentence
appears in both sets. A shingle-based or embedding-based near-duplicate check finds them, and the rate is often
several percent in corpora assembled from the web.

Cause (c) is not leakage and should not be corrected by deduplication: a time-based split deliberately evaluates
on a later window, and the gap is the honest estimate of drift. The distinguishing observation is whether the
overlap check finds shared records; if it finds none and the gap persists, the split is doing its job and the
question moves to distribution shift.

The verification deserves emphasis because it is the one teams weaken: a hash-based overlap check returning zero
on a new split is a fact, while "we removed the overlapping documents" is a recollection.

### 8.30 CI-30 — cost per token mixes prices and pads the prompt

| Field | Content |
|---|---|
| ID | CI-30 |
| Category | NUMERICAL_EDGE |
| Symptom | A unit-cost figure is 3-4x below the invoice, and the gap is blamed on pricing changes |
| Competing causes | (a) input and output tokens priced identically; (b) output tokens counted per request rather than per token; (c) padded prompt tokens billed but not counted in the denominator |
| Discriminating check | Recompute from the invoice's own line items: billed input tokens, billed output tokens, and the unit prices on the invoice |
| Correction | Apply separate prices to separate counts, and include padding in the input count |
| Prevention | The cost model is derived from the price list at build time and reconciled monthly against the invoice |
| Verification | Reconcile one day's invoice against the model, requiring agreement within 2% |

From F6 with `p_in = $3` and `p_out = $12` per million tokens, 1,000 input and 300 output tokens cost $0.003 +
$0.0036 = $0.0066. Using a single $3 price for both gives $0.0039, a 41% understatement, and using the per-request
count in the denominator compounds it.

Cause (c) is the one that produces a 4x gap: a 300-token prompt padded to 4,096 is billed at 4,096 input tokens,
so the input term becomes $0.0123 instead of $0.0009, a 13.65x increase on that term. The denominator, if it
counts generated tokens only, does not move at all — so the unit cost rises by exactly the padding multiplier and
the model that ignored padding reports the pre-padding number.

The reconciliation discipline is what keeps this honest: a model reconciled monthly against an invoice within 2%
is a model, and a model that has never been reconciled is a hypothesis. The reconciliation also catches the
pricing changes that cause (a) and (b) name, which is why it is the prevention rather than a nice-to-have.

## 9. Complexity & Cost Analysis

### 9.1 Cost of a wrong diagnosis

| Failure | Time to detect unaided | Time with the ladder | Reason |
|---|---|---|---|
| Unit confusion near a quota | Hours to days | Minutes | The arithmetic is two conversions |
| Aggregation defect | Days, or never | Minutes | Regrouping changes the answer |
| Prevalence shift blamed on the model | Weeks | Hours | Needs a population recount, not a model change |
| Instrument defect | Weeks | Hours to days | Needs a second source to disagree |
| Correlated-failure blind spot | Until the next incident | One drill | Needs injection, not observation |

The asymmetry is the argument for the ladder: the top two layers cost minutes and account for a large share of
disputed numbers, while the bottom two layers cost weeks and are often discovered by accident.

### 9.2 Instrumentation cost

| Signal | Collection cost | Retention cost | Prevents |
|---|---|---|---|
| Raw request records with latency | High volume, sampled | Moderate | CI-19, CI-11 |
| Per-phase timings for restore | Trivial | Trivial | CI-20, CI-22 |
| Step counter in each shard | Trivial | Negligible | CI-01, CI-08, CI-10 |
| Feature age at decision time | Low | Low | CI-14 |
| Prevalence in the scored population | Needs labels | Moderate | CI-16, CI-15 |
| Per-tenant cache counters | Low | Low | CI-24 |
| Effective batch in the launcher config | Trivial | Negligible | CI-23 |

The table's point is that four of the seven signals cost almost nothing and prevent the highest-frequency
entries. Teams tend to invest in the expensive ones, because the expensive ones produce dashboards.

### 9.3 Correction cost versus defect cost

| Correction | Engineering effort | Risk of the change | Defect cost if unaddressed |
|---|---|---|---|
| Rename a unit in a config | Minutes | None | One quota breach per quarter |
| Pool a ratio | Hours | Low, changes reported numbers | Persistent 5-50% reporting error |
| Add step metadata to shards | Hours | Schema migration for old generations | Silent torn resumes |
| Admission control in cache bytes | Days | Can increase queueing | OOM under peak load |
| Key every side effect | Days | Sink changes | Duplicate effects, unrecoverable without reconciliation |
| Chain-closure retention | Days | Refuses some deletions | Loss of all generations above a base |

### 9.4 Where the effort actually goes

The distribution is worth stating because it inverts the catalog's ordering, and this file does not have a
measured one: no incident corpus was counted here, so any split would be invented. What can be said without
inventing a number is structural. Layers 1 and 2 are decidable by arithmetic on data the team already has, so
they are cheap to test and cheap to be wrong about. Layer 3 needs a drill or a second data source, so it is
priced in hours. Layer 4 needs a purchase or a design change, so it is priced in weeks and is the layer where a
wrong diagnosis is most expensive to act on. The ladder therefore orders work by the cost of the evidence, and
a team should exhaust the cheap layers before buying the expensive evidence — not because cheap layers are more
frequent, but because the cheap test either resolves the question or eliminates a layer for free.

**Takeaway:** the ladder is ordered by cost of evidence, not by frequency of cause, and the two orders differ.

## 10. Key Theorems / Results Reference

**R1 — Pooled versus averaged ratio.** For groups with unequal exposure, the mean of per-group rates differs
from the rate of the union; they coincide only when exposures are equal. Consequence: any rate reported without
its grouping is ambiguous, and CI-06 quantifies the error at 53.8% on a two-group case.

**R2 — Percentile non-composition.** A percentile of a set of aggregates is not the percentile of the underlying
population, and is bounded above by the largest aggregate. Consequence: tails cannot survive pre-aggregation,
so a p99 from per-minute means is a statement about minutes.

**R3 — Bayes at fixed operating point.** PPV falls monotonically with prevalence for fixed sensitivity and
specificity, and the fall is steep because the false-positive term is dominated by the negative class.
Consequence: an unchanged model has a changed precision whenever the population changes (CI-16).

**R4 — Hazard addition under independence.** For `n` independent components, the aggregate hazard is the sum of
component hazards, so the mean time to first failure is the harmonic-scale division `M_node / n`. Consequence:
checkpoint intervals must be derived from run-level hazards, and correlated failures break the model entirely.

**R5 — Max-plus latency.** A barrier's completion time is the maximum over participants, not the mean;
throughput is the sum over participants. Consequence: capacity needs the pooled rate and schedule needs the
maximum, and one number cannot serve both (CI-06).

**R6 — Request-rate floor.** A write of `N` objects against a sustained request allowance `q` cannot complete
faster than `N/q`, independent of bytes. Consequence: shard count sets a hidden latency floor that appears only
above a threshold, which is why CI-07 is a scale failure and not an arithmetic error.

**R7 — Padding multiplier.** With static padding, executed length `L_max` against useful length `L_mean` gives a
multiplier `L_max / L_mean` on linear terms and `(L_max / L_mean)^2` on quadratic ones. Consequence: a 13.65x
linear multiplier and a 186x quadratic multiplier from the same configuration.

**R8 — Cache growth.** KV cache bytes are linear in concurrency and length and independent of parameter count.
Consequence: a capacity plan built from weights omits the term that grows with traffic (CI-12, CI-28).

**R9 — Idempotency requires a key.** A repeated effect collapses to one only if the effect is identified by a
value stable across attempts. Consequence: retry policy cannot substitute for a key, and exactly-once is a
property of a keyed sink rather than of a transport.

**R10 — Integrity, membership, sufficiency.** Digest verification establishes integrity only; membership needs a
shared generation identifier and sufficiency needs a state inventory. Consequence: a green validation is
compatible with a torn generation (CI-08).

**R11 — Retention closure.** Deleting an object with live dependents removes every generation that depends on it,
so count-based retention is unsafe in the presence of chains. Consequence: deleting one base destroys every
generation above it, so the loss is superlinear in the depth of the chain (CI-21).

**R12 — Composition of restart time.** `RTO = detect + schedule + fetch + load + warm` with a maximum rather than
a mean across ranks. Consequence: the addressable share of the objective is one addend, and CI-20 shows the
measured share at 27%.

**Takeaway:** eleven of these twelve are statements about which quantity is being measured, not about which
algorithm is running.

## 11. Interview / Exam Questions

### Q1. A checkpoint write takes 456 s while the per-rank rates average to 15.25 MiB/s over 512 ranks. Which number is wrong?

**Answer:** The 15.25 MiB/s is a correct mean of an intensive quantity and the wrong predictor; completion is
governed by the slowest rank. The 456 s corresponds to 1.783 GiB at 4 MiB/s on 32 ranks.

**Reasoning:** Throughput across participants is a sum; latency across participants is a maximum. A barrier waits
for the maximum, so a schedule derived from a mean understates whenever ranks are heterogeneous.

**Follow-up:** What would you report to a capacity review, and what would you report to an on-call engineer?

**Red flag:** Averaging again, or concluding the store is slow.

### Q2. Compute the checkpoint interval for C = 90 s, R = 300 s, M = 2 h.

**Answer:** T* = sqrt(2 * 90 * 7200) = sqrt(1.296e6) = 1138 s = 19.0 min. At that interval phi = 90/1138 +
(569 + 300)/7200 = 0.0791 + 0.1207 = 20.0%.

**Reasoning:** M must be in seconds and C in seconds. The R term shifts the floor without moving the minimum,
because it is constant in T.

**Follow-up:** What does the interval become if the run spans 400 independent nodes with 30-day per-node MTBF?

**Red flag:** Using 2 hours as 120 in the formula, or dropping the R term.

### Q3. A model's precision fell from 71% to 18% with no change. What is the first computation?

**Answer:** Prevalence in the scored population. With sensitivity 0.90 and specificity 0.95, PPV at 12% is 0.710
and at 1.2% is 0.179, so the observed fall is fully explained by a tenfold prevalence drop.

**Reasoning:** PPV is prevalence-dependent while sensitivity and specificity are not. Computing PPV at the new
prevalence takes one line and either explains the gap or excludes the mechanism.

**Follow-up:** If prevalence is unchanged, what are the next two candidates and how do you separate them?

**Red flag:** Retraining, or comparing AUC across populations.

### Q4. A dashboard shows p99 of 1.3 s while users report 4 s. Both are computed from the same traffic.

**Answer:** The dashboard computes p99 over one-minute means; the user experiences individual requests. Ten
requests with nine at 1 s and one at 4 s have a true p99 of 4 s and a mean of 1.3 s.

**Reasoning:** Averaging is a contraction toward the centre, so a percentile of aggregates cannot exceed the
largest aggregate and cannot represent a tail.

**Follow-up:** What change to the metric pipeline fixes it without changing the underlying collection?

**Red flag:** Blaming clock skew, or concluding users are wrong.

### Q5. Why does a 4,096-token padded batch with 300 real tokens cost 13.65x in compute but look free on the invoice?

**Answer:** Because the invoice bills generated tokens while compute runs over padded prompt length. The
utilisation is 300/4096 = 7.32% and the multiplier 13.65x applies to the prefill term.

**Reasoning:** Billing and compute measure different quantities. The metric that reveals the gap is GPU-seconds
per generated token, which no invoice contains.

**Follow-up:** What does bucketing by length recover, and what does it cost in scheduling complexity?

**Red flag:** Accepting that padding is free because the bill did not change.

### Q6. A serving tier OOMs at 32 concurrent 2,048-token requests with 80 layers, 8 KV heads, head dimension 128, bf16.

**Answer:** Cache is 2*80*8*128*2 = 327,680 B = 320 KiB per token, so 2,048 tokens is 640 MiB per sequence and 32
sequences need 20.0 GiB, on top of the weights.

**Reasoning:** The 2 accounts for key and value tensors; 8 KV heads rather than 64 is grouped-query attention and
already reduces the cache eightfold. Admission must be expressed in cache bytes.

**Follow-up:** What admission policy would you replace a fixed concurrency limit with?

**Red flag:** Sizing from parameter bytes, or assuming a sliding window when the model has none.

### Q7. A resumed run skips data and no error was logged. Where do you look?

**Answer:** At the cursor definition. If the loader's position was saved instead of the consumed position, the
skip equals workers times prefetch depth, which is 32 batches for 8 workers at depth 4.

**Reasoning:** The loader is ahead of consumption by construction, so saving its position saves the future. The
check is the restored start index against the recorded consumed index.

**Follow-up:** Why is the fix structural rather than a corrected constant?

**Red flag:** Reseeding the sampler, which changes order and hides the defect.

### Q8. Estimate the monthly cost of eight generations at 912.7 GiB each, priced at $0.023 per GB-month.

**Answer:** 912.7 GiB = 980.0 GB, so one generation-month is $22.54; eight is $180.32; twenty runs is $3,606.

**Reasoning:** The price is per decimal GB, so the GiB figure must be converted before multiplying. Using 912.7
as GB understates by 7.4%.

**Follow-up:** Compare that with the cost of 19.8% lost accelerator time and say which you would optimise first.

**Red flag:** Multiplying GiB by a per-GB price.

### Q9. A drill restores in 5 minutes and the real recovery takes 40. Name three mechanisms and their discriminators.

**Answer:** Cold image pull (measured pull time in platform events), a stale generation on a slower tier (the
generation's tier in its metadata), and contention with other tenants (per-tenant byte counters during the
window). A fourth is a different identity's permissions (authorisation errors in the audit log).

**Reasoning:** All four are absent from a warm, isolated, most-recent-generation drill, so none can be found by
repeating that drill. Each leaves a distinct record.

**Follow-up:** Which single change to the drill would have caught two of the four?

**Red flag:** Concluding the incident was exceptional.

### Q10. How does a 512-node run with 30-day per-node MTBF change a 3-hour checkpoint schedule?

**Answer:** Run MTBF becomes 720/512 = 1.406 h = 5063 s. With C = 114 s, T* = 1075 s rather than 1570 s, and
overhead at 1570 s rises from 19.8% to 34.1%.

**Reasoning:** Independent hazards add, so the mean time to first failure divides by the node count. The interval
scales as the square root, giving the 1.46x factor.

**Follow-up:** What evidence would tell you failures are not independent?

**Red flag:** Treating the vendor number as a property of the job.

### Q11. A digest check passes and the restored model trains badly. Why can both be true?

**Answer:** A digest establishes integrity, not membership or sufficiency. A set of shards from two generations,
each internally consistent, passes every digest and is still torn.

**Reasoning:** Membership needs a generation identifier recorded per shard; sufficiency needs a state inventory
compared with what the trainer consumes. Neither is derivable from a hash.

**Follow-up:** What is the cheapest field you can add that closes the larger of the two gaps?

**Red flag:** "The checksum passed, so the file is fine."

### Q12. During recovery, requests to the recovering service triple and it fails again. What is the mechanism and the fix?

**Answer:** Retry amplification. A single retry layer with failure probability `p` multiplies load by
`1/(1-p)`: 2x at p = 0.5 and 5x at p = 0.8. The fix is jittered backoff with a retry budget and admission control
that rejects rather than queues.

**Reasoning:** The amplification is arithmetic, not misbehaviour; every client is doing what it was told. The
synchronised-timer case turns a short failure into a long one because the pulse re-exceeds capacity.

**Follow-up:** Why is jitter the load-bearing part rather than the backoff?

**Red flag:** Raising the retry limit, or scaling out as the primary fix.

### Q13. A cache reports 85% hit rate and the largest tenant reports poor latency. Reconcile.

**Answer:** 85% is an unweighted mean across tenants. With ten small tenants at 90% and one large at 40%, and the
large one carrying 80% of requests, the pooled rate is 50%.

**Reasoning:** Hit rate is a ratio, so it pools as total hits over total requests. Averaging across unequal
volumes over-weights small tenants.

**Follow-up:** Which of the two numbers belongs on the capacity dashboard, and why?

**Red flag:** Asserting the tenant is misconfigured because its rate is below the mean.

### Q14. Why does a weight export resume badly, in one computation?

**Answer:** It carries parameters only: 130.4 GiB of 912.7 GiB, or 14.3%. The missing moments and master copy are
85.7% of state, and Adam's update divides by the square root of a moment that was reset to zero.

**Reasoning:** The loss offset after such a resume is not noise around the same trajectory; it is a different
optimisation problem for the first few hundred updates.

**Follow-up:** Which single missing element explains a loss offset that persists for a thousand updates rather
than a few?

**Red flag:** "It loaded without error, so it resumed."

### Q15. An autoscaler reports 15% CPU at saturation and does not scale. What do you change?

**Answer:** The scaling signal. CPU stopped being the binding resource after work moved to the accelerator; the
signals that bind are queue wait or accelerator memory, and the policy should name the saturating resource.

**Reasoning:** A proxy metric is valid only while the proxy remains the constraint, and nothing in a workload
change announces that it has stopped. The check is to measure every candidate resource under peak load and see
which one saturates first.

**Follow-up:** What is the failure mode of switching to queue depth as the sole signal?

**Red flag:** Lowering the CPU threshold, which scales on a metric that is 15% at the true limit.

## 12. Quick Reference Card

| Symptom | First discriminator | Likely entry |
|---|---|---|
| Numbers disagree with an invoice or quota by about 7% | Convert to both units and compare with the quota's documented unit | CI-03, CI-30 |
| Aggregate contradicts per-group reports | Recompute as ratio of sums; check whether regrouping moves the answer | CI-06, CI-24 |
| p99 far below the worst user experience | Ask what population the percentile was taken over | CI-19 |
| Quality fell with no model change | Compute prevalence in the scored population, then apply PPV | CI-16, CI-15 |
| Resume succeeds but loss is offset | Diff artifact keys against consumed keys; check first-step loss | CI-04, CI-09, CI-10 |
| Validation green, resume wrong | Compare per-shard step counters, not digests | CI-01, CI-08 |
| Restore misses its objective | Time the five phases separately | CI-20, CI-22 |
| OOM under load, not at rest | Compute cache from concurrency times length | CI-12, CI-28 |
| Failure rate far above plan | Divide per-node MTBF by node count | CI-05 |
| Checkpoint slows after a shard increase | Objects divided by request allowance versus bytes over bandwidth | CI-07, CI-17 |
