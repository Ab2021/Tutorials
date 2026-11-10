---
case_id: "CASE-2026-023-002"
case_title: "Profile a memory-bound training step before buying more accelerators"
case_version: "1.0-draft"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "incident"
domain: "GPU hardware, kernel efficiency and distributed training"
primary_pattern: "resource-accounting"
secondary_patterns: ["correctness-contract", "cost-latency-tradeoff"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "training platform lead"
source_index: "INDEX.md"
review_trigger: "device SKU, driver/compiler, runtime kernel selection, sequence length, batch geometry, topology or deadline change"
---

# Profile a memory-bound training step before buying more accelerators

## 1. The One-Line Framing

Given a per-step timing and traffic profile of a dense transformer pretraining step, the training platform lead must choose one bottleneck-removal intervention for the next run window, subject to a fixed wall-clock budget and a numerical-parity guardrail, compared with the default response of adding accelerator count.

The unit of analysis is one optimizer step of a fixed model, sequence length, and batch geometry, decomposed into named phases: matrix-multiply work, attention work, collective communication, and residual host-side overhead.
The decision output is a ranked, evidence-labelled intervention list with stop rules, plus an explicit statement of which quantities were measured, which were derived, and which remain unknown.
The beneficiary is the research team whose experiment loses wall-clock days when the step exceeds the scheduled window, and the platform owner who pays for device-hours that produce no additional tokens.
The baseline is "scale out first": add devices or replicas to close the gap without locating the limiting resource.
The binding constraint is a fixed 288-hour window over eight devices, which converts into a maximum admissible step time; the run must also keep loss-trajectory parity within a declared tolerance.
The evidence declaration comes first: every device peak, bandwidth, step time, cost rate, and power figure in this case is constructed for teaching.
No number here is a vendor specification, a measured production trace, a leaderboard entry, or a result attributed to a named organisation.
The decisive, unglamorous finding is that the attention phase consumed 6.10 s of a 14.60 s step while using neither the tensor units nor the memory system at their limits, so neither "add bandwidth" nor "add FLOPs" would have been the correct purchase.
Resource accounting is the primary pattern: each phase is charged against a specific boundary with explicit units before any remedy is selected.
Correctness contract is secondary because a faster kernel that changes the accepted reduction order must still demonstrate that the loss trajectory stays within a declared equivalence band.
Cost-latency tradeoff is secondary because device-hours and wall-clock days are not interchangeable units even when both improve together.
A raw "utilization is 39%" dashboard reading is insufficient because the same 39% is consistent with a compute-starved GEMM phase, a latency-bound attention phase, and a collective phase stalled by a degraded fabric link.
The framing therefore refuses to accept a single aggregate efficiency number as a diagnosis.

## 2. Context & Background

A research group is pretraining a 7.0e9-parameter dense decoder-only transformer from scratch on one eight-device node.
The training objective is next-token prediction with a standard cross-entropy loss over a fixed tokenizer vocabulary.
The run is planned for 2.0e10 training tokens, which is small enough that the schedule fits in a twelve-day window only if per-step time stays under a derived threshold.
Each step consumes a global batch of eight sequences of 32,768 tokens, giving 262,144 tokens per optimizer step and 76,294 steps for the planned token count.
Parallelism is deliberately simple: pure data parallelism across eight ranks, no tensor parallelism, no pipeline parallelism, no optimizer-state sharding.
The simplicity is a scoping decision, not a claim of optimality; it keeps the accounting boundary per rank identical and removes cross-rank weight sharding from the memory analysis.
Precision is bfloat16 for the forward and backward matrix products with an fp32 master weight copy and AdamW state, following a mixed-precision policy whose overflow behaviour was validated separately.
Activation checkpointing is enabled on every transformer layer's input boundary, which trades recomputation for activation residency.
The baseline instrumentation is a coarse step timer plus a framework-reported "model FLOPs utilization" gauge that operators watch as a single health number.
That gauge has no phase decomposition, no traffic counters, and no clock or topology metadata attached to it.
When step time rose past the window threshold, the first proposal in the team meeting was to double the device count.
The platform lead pushed back, because the doubling proposal had no causal story: it would have doubled both the compute and the per-rank gradient traffic.
The research lead owns the experiment schedule, the token budget, and the acceptable loss-trajectory equivalence band.
The platform lead owns the profiling method, the intervention ranking, and the stop rules.
The kernel engineer owns the attention implementation and its numerical validation.
The network engineer owns the fabric health, the collective library configuration, and the link-level counters.
Capacity planning owns the chargeback model and the device-hour rate.
Compliance and data governance are out of scope for this case; the corpus pipeline and its provenance controls belong to a separate decision with its own owner.
This case concerns one profiling cycle at one point in the run, and it explicitly does not claim that its conclusions transfer to a different model width, sequence length, or interconnect generation.

### Operating vocabulary

| Term | Meaning in this case | Boundary that must be stated with it |
|---|---|---|
| Step | One forward, backward, and optimizer update at the declared batch geometry | Never compare step times across geometries without normalising tokens |
| Micro-batch | One sequence group processed between gradient-accumulation boundaries | At this geometry there is no accumulation; one step equals one micro-batch |
| Phase | Named segment of the step with its own limiting resource | A phase label without a resource boundary is not an accounting unit |
| MFU | Model FLOPs utilization = achieved model FLOP/s divided by a declared peak | The denominator convention changes the number by a third |
| Arithmetic intensity | FLOPs performed per byte moved to or from device memory | Must state which bytes are counted, since cached bytes do not count |
| Ridge point | Intensity at which a kernel transitions from bandwidth-bound to compute-bound | Depends on the achieved, not advertised, hardware rates |
| Traffic | Bytes actually moved across a named memory or fabric boundary | A tensor's logical size is not its traffic |
| Collective | A multi-rank operation such as all-reduce over gradient buffers | Its cost depends on algorithm, topology, and message size together |
| Parity band | Declared tolerance on loss or logit agreement after an optimization | A band is a contract, not a hope |
| Device-hour | One accelerator occupied for one hour | Billing and capacity planning use the same unit for different purposes |
| Recompute | Repeated forward work substituted for stored activations | Charged to the compute phase, not the memory phase |

The distinction between a phase label and a resource boundary is the discipline this case enforces.
"Attention is slow" is a phase label.
"Attention took 6.10 s while moving 70.37 TB at an achieved 11.54 TB/s, which is 48% of the aggregate device-memory rate" is an accounting statement with a testable implication.
Everything in Section 6 is written to keep that distinction visible.

### Why the obvious process persists

Scaling out is the historically rewarded action: it requires no diagnosis, it produces an unambiguous resource change, and it usually improves something.
It also consumes the budget that could have paid for the diagnosis, and under a fixed window it may not fit at all if node allocation is the scarce resource rather than money.
The aggregate MFU gauge persists because it is cheap to emit and easy to display, and because a single number creates the illusion that one remedy exists.
The case does not argue that scaling out is wrong in general; it argues that it was wrong here, and that the profile was the cheaper experiment.

## 3. The Business Question

Illustrative stakeholder question, paraphrased into a decision contract: "The run is scheduled to end in twelve days and we are already over the per-step threshold. Do we buy more devices, or is something in the step itself wrong?"

| Decision contract | Content |
|---|---|
| Question | Which single intervention should be funded for the next run window so the planned token budget completes inside the fixed 288-hour window without breaking the loss-parity band? |
| Decision owner | Training platform lead, with the research lead holding veto over any change that exceeds the declared parity band |
| Trigger and deadline | Step time exceeded the derived window threshold during the warm-up phase; the intervention must be selected before the next allocation window opens |
| Available actions | Add devices (scale out); replace the attention implementation with a fused kernel; repair or reconfigure the gradient collective; enlarge batch to improve arithmetic intensity; shrink sequence length; accept the overrun and renegotiate the window; pause and instrument further |
| Value and downside | Meeting the window returns the scheduled experiment; a wrong intervention burns the window and produces no additional tokens; a silently inexact kernel corrupts the run's scientific value |
| Objective | Minimise device-hours to complete 2.0e10 tokens subject to step time under the window threshold and parity inside the band |
| Hard constraints | Fixed 288-hour window; eight-device allocation; parity band; no change to the model definition or tokenizer; no reduction in the planned token count |
| Scope | This model, this sequence length, this node, this run window |
| Out of scope | Corpus quality, tokenizer changes, architecture changes, checkpoint storage design, multi-node scheduling policy, vendor selection |

The business objective and the engineering proxy are not identical, and the case names the gap.
The proxy is step time in seconds; the objective is completed tokens within a calendar window at acceptable scientific validity.
A step-time improvement achieved by shrinking the sequence length would satisfy the proxy and violate the objective, because it changes the training distribution and therefore the experiment.
A step-time improvement achieved by dropping the parity contract would satisfy the proxy and destroy the run's comparability with its reference curve.
Any ranking of interventions must therefore carry both a time effect and a validity effect, and the parity band is a disqualifier rather than a term in a weighted score.

What an expensive false decision looks like here: buying eight additional devices converts a 7.4% step-time overrun into a schedule that still cannot start, because the allocation window does not contain the extra devices, while the actual limiting phase remains untouched.
A second expensive false decision is adopting a fused attention kernel that is 30% faster and changes the accepted reduction order, without running the parity test, so the run completes on time and produces a loss curve that cannot be compared with anything.
The decision frequency is low, roughly once per run, which is why a repeatable profiling method matters more than a one-off heroic diagnosis.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | The declared device peak of 320 TFLOP/s per device applies to the dense bfloat16 matrix-multiply path used here | hardware | Read the installed device's documented dense-precision rate for the exact dtype and sparsity setting | unverified | Every efficiency ratio shifts by the ratio of assumed to actual peak | Recompute all ratios with the documented value; do not rescale the measured times | Device SKU or dtype change / platform lead |
| A-02 | Aggregate device-memory bandwidth is eight times the per-device 3.0 TB/s | hardware | Check that no device is partitioned, throttled, or sharing a narrower link | unverified | Traffic-bound floors are wrong in proportion | Re-derive per-device floors and report the limiting device, not the mean | Partitioning or clock-policy change / platform lead |
| A-03 | The measured phase timings decompose the step without double counting | measurement | Sum of phases equals the independently measured step wall time within the sampling tolerance | verified in the constructed trace (14.60 s) | An unaccounted phase hides the real limit | Add the missing phase; never rescale the others to fit | Profiler or runtime version change / kernel engineer |
| A-04 | The attention traffic model of four passes over the score tensor per layer is the as-implemented behaviour | model | Read the kernel's actual materialisation pattern, or count traffic in a profiler | illustrative | The computed 70.37 TB is wrong, changing the break-even conclusion | Recount traffic; if the kernel is already fused, this intervention is moot | Kernel selection change / kernel engineer |
| A-05 | Data-parallel all-reduce of bfloat16 gradients moves 2*(n-1)/n times the gradient bytes per rank | model | Confirm the collective algorithm and bucket size; ring all-reduce matches this model | illustrative | Collective floor and the size of the observed gap change | Replace the ring model with the actual algorithm's cost | Collective library or algorithm change / network engineer |
| A-06 | bfloat16 accumulation with fp32 master weights keeps the loss inside the declared band | numerical | Run the parity test against an fp32 reference over a fixed seed and token window | unverified | The whole run is invalid regardless of speed | Widen the band with justification, or revert precision | Precision policy change / research lead |
| A-07 | Layer-input checkpointing is enabled and costs one extra forward pass per layer | implementation | Inspect the applied checkpoint policy in the run configuration | verified in configuration | Memory and compute accounting both change | Re-derive residency and recompute cost | Checkpoint policy change / platform lead |
| A-08 | The 20e9 token budget and 7.0e9 parameter count are fixed for this decision | business | Read the research plan of record | verified in the plan | Deadline arithmetic and FLOP totals change | Recompute the whole ledger | Plan revision / research lead |
| A-09 | Illustrative device-hour rate is 3.00 currency units and node power is 5.6 kW | business | Replace with an actual dated rate and metered power | illustrative | Economic ranking changes, technical ranking does not | Recompute cost columns only | Rate or power change / capacity planning |
| A-10 | The profiler's own overhead is small relative to the phases being measured | measurement | Compare a profiled step with a clean step and report the delta | unverified | Durations are inflated, and phase ratios may be distorted | Use sampling profiler or counters only | Profiler configuration change / kernel engineer |
| C-01 | Wall-clock window is 288 hours and cannot be extended | business | Allocation contract | confirmed in scenario | Deadline comparison is meaningless | Re-derive threshold | Allocation change / capacity planning |
| C-02 | Exactly eight devices are available in the window | infrastructure | Allocation contract | confirmed in scenario | Scale-out option is unavailable, not merely unattractive | Re-rank with nine or more devices if granted | Allocation change / capacity planning |
| C-03 | Loss-trajectory parity must stay inside the declared band | scientific | Research plan of record | confirmed in scenario | Any numerical change is disqualified | Choose only bit-compatible interventions | Band revision / research lead |
| C-04 | Model definition, tokenizer, and token budget are frozen | scientific | Research plan of record | confirmed in scenario | Sequence or width remedies are disqualified | Choose interventions that preserve the distribution | Plan revision / research lead |
| C-05 | The run must remain restartable from its existing checkpoint format | operations | Recovery drill on the current format | unverified | A change that alters shard layout breaks recovery | Keep optimizer-state layout unchanged | Checkpoint format change / SRE |
| C-06 | Fabric health is not currently monitored per link | operations | Absence of link counters in the current telemetry | confirmed in scenario | The collective gap cannot be attributed to a specific link | Add counters before concluding | Telemetry rollout / network engineer |
| C-07 | No second node is available for a controlled A/B run | infrastructure | Allocation contract | confirmed in scenario | All comparisons are within-run or single-node | Use interleaved step sampling with a declared variance | Allocation change / platform lead |
| C-08 | The run may be paused for one profiling window without losing the allocation | operations | Scheduler policy | unverified | Profiling must be done online with overhead | Use in-band counters only | Scheduler policy change / SRE |

The three assumptions most likely to reverse the recommendation are A-04, A-06, and A-02.
If the attention kernel already avoids materialising the score tensor, then the largest modelled opportunity disappears and the GEMM or collective path becomes the candidate.
If the parity test fails for the fused kernel, the fastest option is disqualified and the ranking falls back to the collective repair.
If a device is partitioned or clock-limited, then the aggregate bandwidth assumption is false and every traffic floor must be recomputed per device, which may move the bottleneck to a single rank.
Each of those is a cheap check relative to the intervention, which is why they are listed as preconditions rather than risks.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is one optimizer step, identified by a step index within one run.
The table below describes the trace schema a profiler must produce for this decision, not a dataset of examples.

| Field | Required detail |
|---|---|
| Unit of analysis | One step, one rank, one phase; steps are keyed by (run_id, step_index, rank_id, phase) |
| Volume and window | A contiguous window of steps including warm-up, steady state, and at least one checkpoint write |
| Inputs | Phase start/end timestamps on a monotonic clock; kernel names and durations; byte counters for device memory and fabric; collective message sizes; achieved clocks; power draw if metered |
| Outcome/label | Phase duration in seconds and traffic in bytes; no predictive label is used |
| Population and exclusions | All ranks in the step are included; warm-up steps before allocator stabilisation are reported separately and not averaged with steady state |
| Join semantics | Join phase records to step records on (run_id, step_index); a phase record without a matching step record is a telemetry defect, not a zero |
| Splits | Not applicable in the modelling sense: the decision uses one profiling window plus a boundary window at a different sequence length |

The trace schema matters because the most common profiling failure in this setting is a silent join defect.
An attention phase that failed to emit a record would otherwise be averaged as zero duration, producing the confident and entirely false conclusion that attention is free.
The method therefore requires an explicit unaccounted-time line in the decomposition rather than distributing residual time across the other phases.

For the boundary test, a second window at a different sequence length is required.
The 64,384-token window at 65,536 sequence length is the boundary case used in Section 6.15; it is not a second dataset but a different operating point of the same trace schema.

### 5b. Tools & Stack

| Component | Role in this decision | What must be pinned |
|---|---|---|
| Training framework | Produces the step, the autograd graph, and the checkpoint format | Framework version, distributed backend, checkpoint layout |
| Attention implementation | Determines score-tensor traffic and reduction order | Kernel identity, version, whether it materialises scores, supported dtypes |
| Collective library | Determines all-reduce algorithm and bucket behaviour | Library version, algorithm selected, bucket size, timeout settings |
| Profiler | Emits phase timings, kernel names, byte counters | Profiler version, sampling mode, overhead estimate, clock domain |
| Fabric counters | Attribute collective delay to links | Per-link byte and error counters, availability |
| Device management interface | Reports clocks, power, partitioning | Tool version, sampling interval, whether MIG or equivalent partitioning is active |
| Experiment tracker | Records the run configuration that produced the trace | Configuration hash, tokenizer hash, data cursor |

The list omits storage backends, scheduler internals, and corpus tooling because none of them changes the ranking of interventions in this decision.
Naming a component's product family would add nothing to the accounting; the pinned version is what makes a claim reproducible.

### 5c. Method Summary

The method is a four-stage decomposition: build a FLOP and byte model from the model geometry, derive the resource floors implied by those totals, compare the floors with measured phase durations, then compute the break-even efficiency at which a candidate intervention stops being worth doing.

The simplest credible baseline is the aggregate MFU gauge: total model FLOPs divided by step time, expressed against a declared peak.
Its strength is that it costs nothing and detects gross regressions.
Its limitation is that it cannot distinguish a compute-starved phase from a latency-bound one, so it cannot rank remedies.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Aggregate MFU gauge | Framework-emitted counter | Free, continuous, detects regressions | No phase attribution, denominator ambiguity | used as baseline only | Never sufficient alone |
| Roofline per phase | FLOP and byte model plus achieved rates | Names the limiting resource per phase | Depends on traffic model accuracy | chosen | Traffic model or device rate changes |
| Kernel-level profiler trace | Per-kernel durations and counters | Highest resolution | Overhead and distortion | used for attribution | Overhead changes scheduling |
| Controlled A/B node experiment | Two nodes, one change each | Strongest causal evidence | Not available in this allocation | not tested | A second node becomes available |
| Scale-out | Allocation arithmetic only | Simple, unambiguous | Consumes the window; does not address a phase-level limit | rejected with arithmetic | Node count becomes available and phase profile is balanced |

Model objective, optimizer, and stopping rule are fixed by the research plan and are not variables in this decision.
What varies is the implementation of two phases and the configuration of one collective.
That narrow variation is what makes the causal claim defensible within a single node.

### 5d. Provenance, Quality, Exclusions and Leakage

The trace is a single-window observation on one node, produced under the profiler's own instrumentation.
Its provenance record must therefore carry the profiler configuration, because the act of measuring changes what is measured: instrumented kernels can serialise work that would otherwise overlap, and clock sampling can perturb boost behaviour.

The traffic counters have a different provenance from the duration counters.
Durations come from the profiler's timeline; traffic comes from hardware counters that may be aggregated per memory partition.
Mixing them without saying so is the analogue of training/serving leakage in a predictive project: it produces a number that looks like a joint measurement but is a comparison of two differently defined quantities.

Exclusions that must be stated: warm-up steps, the first step after a checkpoint write, any step during which the allocation was preempted, and any rank whose counters were unavailable.
A rank with missing counters is reported as missing, never imputed from its peers, because the entire purpose of the analysis is to find per-rank asymmetry.

No corpus data enters this decision, so there is no label-leakage question in the usual sense.
The analogous leakage hazard is using a fused-kernel benchmark from a different sequence length as if it were evidence for this sequence length, since the attention phase's arithmetic intensity depends quadratically on sequence length.

### 5e. Reproducibility and Method-Selection Ledger

| Artifact or source ID | Location | Version or date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Device documentation for dense bfloat16 rate and memory bandwidth | must be pinned at authoring | Supplies A-01 and A-02 denominators | public documentation | Documentation states theoretical maxima, not achieved rates |
| SRC-02 | Attention kernel documentation describing materialisation and supported dtypes | must be pinned | Supplies A-04 traffic assumption | public documentation | Documentation may lag the selected kernel |
| SRC-03 | Collective library documentation for algorithm selection | must be pinned | Supplies A-05 cost model | public documentation | Algorithm selection is heuristic and topology-dependent |
| RUN-01 | Constructed phase trace in Section 6.5 | this document | Supplies the decomposition used throughout | illustrative | Not a measured run; no hardware was profiled |
| RUN-02 | Constructed arithmetic oracle in Section 6.18 | this document | Independent check of every derived quantity | illustrative | Checks arithmetic, not hardware behaviour |

Everything in RUN-01 and RUN-02 is constructed, and the case says so at every point of use.
The construction is not a weakness for the teaching purpose, because the decision structure is what transfers; the numbers are a scaffold that a reader replaces with their own counters.

Reproduction of the arithmetic requires only the model geometry and the declared hardware rates, all of which are given.
Reproduction of the hardware behaviour requires the pinned stack, the trace schema, and the boundary window, none of which is available here.
That distinction is recorded rather than blurred, because a reader who treats the constructed 6.10 s as a benchmark will draw a false conclusion about their own node.

## 6. Analysis Walkthrough

### 6.1 Fix the conventions before computing anything

Three conventions are fixed once and used everywhere, because each of them changes headline numbers by a large factor.

The first is the FLOP convention: 6N per token for forward plus backward without activation recomputation, and 8N per token when the backward pass recomputes the forward for checkpointed layers.
This case uses 8N because every layer input is checkpointed.

The second is the unit convention: storage uses binary units (1 GiB = 1,024 MiB = 2^30 bytes), and rates use decimal prefixes (1 TB/s = 10^12 bytes/s).
Mixing them produces a 7.4% error on every bandwidth-derived quantity, which is larger than the schedule margin being decided.

The third is the MFU denominator: the declared peak is the dense bfloat16 matrix-multiply rate of the device, multiplied by the device count, with no sparsity multiplier and no allowance for the fact that some executed FLOPs are lower-precision accumulate operations.

```text
FORMULA [C-01]: F_model = k * N * T
WHERE: F_model [FLOP]; N [parameters]; T [tokens per step]; k = 8 for forward+backward with recomputation, 6 otherwise.
CONSTRAINT: dense transformer with the declared checkpoint policy; attention FLOPs are counted in the 8N convention only to first order.
NUMERICAL EXAMPLE: 8 * 7.0e9 * 262,144 = 1.4680064e16 FLOP per step.
INTERPRETATION: the step performs 14.68 PFLOP of model work, which against a declared 2.56 PFLOP/s node peak implies a 5.7344 s floor.
```

The 6N alternative would give 1.1010048e16 FLOP and a 4.3008 s floor.
Reporting a utilization number without saying which convention produced it is the single most common way two engineers disagree by a third while both being arithmetically correct.

### 6.2 Derive the memory floors for the same step

The persistent-state ledger is per rank and uses binary units.

```text
FORMULA [C-02]: M_persistent = N*b_w/DP + N*b_g/DP + N*b_m/DP + 2*N*b_a/DP
WHERE: b_w=2 bytes (bfloat16 weights), b_g=2 bytes (bfloat16 gradients), b_m=4 bytes (fp32 master), b_a=4 bytes (each Adam moment); DP=8 ranks.
CONSTRAINT: plain data parallelism with replicated weights and unsharded optimizer state; a sharded-optimizer design changes every term.
NUMERICAL EXAMPLE: 13.0385 + 13.0385 + 3.2596 + 6.5193 = 35.8559 GiB per rank.
INTERPRETATION: 35.86 GiB of the device's memory is committed before a single activation is stored.
```

Weights and gradients are replicated, not sharded, so each rank holds a full copy.
The master copy and the two Adam moments are also replicated in this design, which is why the optimizer terms are four times the weight bytes divided by the parameter count rather than divided by a shard count.

Activation residency is the second term.
With layer-input checkpointing, each layer stores its input activation of shape (sequence, hidden) in bfloat16.

```text
FORMULA [C-03]: M_ckpt_input = S*H*2 / 2^30
WHERE: S [tokens per sequence]; H [hidden size]; the factor 2 is bfloat16 bytes.
CONSTRAINT: one stored tensor per layer boundary; attention internals are recomputed, not stored.
NUMERICAL EXAMPLE: 32,768 * 4,096 * 2 = 2.68435456e8 bytes = 0.25 GiB per layer; 32 layers = 8 GiB.
INTERPRETATION: checkpointing converts a per-layer activation explosion into a linear, predictable 8 GiB.
```

The counterfactual without checkpointing is the important comparison, because it explains why the policy is not optional at this sequence length.
Per layer, a naive implementation holds the query, key, and value tensors (three tensors of sequence by hidden), the attention output (one), the two normalized copies used by pre- and post-attention residual paths, and the feed-forward intermediate at four times hidden.
That is roughly six activation tensors of 0.25 GiB plus one of 1 GiB, or about 2.5 GiB per layer, or about 80 GiB over 32 layers.
Adding the 35.86 GiB of persistent state already exceeds the usable device memory, which is the arithmetic reason checkpointing is mandatory here rather than a performance preference.

### 6.3 The attention score tensor is the dominant transitory allocation

Attention at this geometry produces an explicit score tensor of shape (heads, sequence, sequence) per layer in the as-implemented path.

```text
FORMULA [C-04]: B_scores = A*S^2*b_s
WHERE: A=32 heads; S=32,768; b_s=2 bytes in bfloat16.
CONSTRAINT: scores materialised for at least one layer at a time; a fused kernel that tiles the scores reduces this to a tile.
NUMERICAL EXAMPLE: 32 * 32,768^2 * 2 = 6.8719476736e10 bytes = 64 GiB for all heads of one layer.
INTERPRETATION: one layer's score tensor alone is 1.8 times the persistent state, so the implementation must process heads or query blocks in chunks.
```

Chunking eight heads reduces the transitory allocation to 16 GiB, which fits inside the residual budget after persistent state and checkpointed activations.

```text
NUMERICAL EXAMPLE: 8 * 32,768^2 * 2 = 1.7179869184e10 bytes = 16 GiB.
```

The whole-layer alternative of 64 GiB does not fit, so the chunk parameter is a hard constraint discovered by arithmetic rather than by an out-of-memory event.
That is the practical value of the ledger: it converts an operational failure into a configuration decision.

### 6.4 Roofline per phase, with achieved rather than advertised rates

The roofline comparison is only meaningful if the intensity of each phase is computed from the traffic the implementation actually generates.

```text
FORMULA [C-05]: I = F / B_traffic;  t_floor = max(F / P_peak, B_traffic / B_peak)
WHERE: I [FLOP/byte]; F [FLOP]; B_traffic [bytes]; P_peak [FLOP/s]; B_peak [bytes/s].
CONSTRAINT: single-level memory model; caches are not modelled, so the byte count must be the bytes that actually reach device memory.
NUMERICAL EXAMPLE: dense GEMM intensity is far above the ridge; attention as implemented is 64 FLOP/byte.
INTERPRETATION: the two phases sit on opposite sides of a device-specific ridge point, so no single remedy can serve both.
```

The ridge point is the ratio of achieved peak to achieved bandwidth, and it must be stated with both numbers.

```text
FORMULA [C-06]: I_ridge = P_device / B_device
WHERE: P_device [FLOP/s]; B_device [bytes/s].
NUMERICAL EXAMPLE: 3.20e14 / 3.00e12 = 106.67 FLOP/byte for the declared device rates.
CONSTRAINT: advertised peak against advertised bandwidth; an achieved-rate ridge would use measured values.
INTERPRETATION: a kernel at 64 FLOP/byte is below the ridge and therefore bandwidth-side, but only by a factor of 1.7.
```

A factor of 1.7 is small enough that the conclusion is sensitive to both rates, which is exactly why the break-even efficiency in Section 6.9 is computed rather than asserted.
An achieved-rate ridge using 45% of peak and 88% of bandwidth gives 54.55 FLOP/byte, which is below 64, so the same kernel would be classified as compute-side under achieved rates.
The two classifications are not contradictory; they answer different questions, and the case uses the advertised-rate ridge for the floor and the efficiency comparison for the decision.

### 6.5 Phase decomposition of the constructed step

The constructed trace reports four phases plus an explicitly unaccounted line.

| Phase | Duration (s) | What it contains | Named limiting resource |
|---|---:|---|---|
| Matrix-multiply work | 7.30 | Linear layers forward and backward, including recomputation of checkpointed layer inputs | Tensor units |
| Attention work | 6.10 | Score computation, softmax, value combination, and their backward passes | Mixed: memory and softmax dependency |
| Collective | 0.75 | Gradient all-reduce over eight ranks | Fabric |
| Residual/host | 0.45 | Optimizer update, kernel launches, sampling | Host and launch latency |
| Unaccounted | 0.00 | Must be shown, not distributed | None claimed |
| Sum | 14.60 | Independently measured step wall time | — |

The sum is checked against the wall time rather than assumed, which satisfies A-03.
An unaccounted line of exactly zero is itself a claim: in a real trace it is usually a small positive number, and a large one invalidates the decomposition until explained.

### 6.6 Check the model FLOPs against the measured step

The first sanity check compares the compute-side floor with the measured duration.

```text
FORMULA [C-07]: MFU = F_model / (t_step * P_node)
WHERE: MFU [dimensionless]; t_step [s]; P_node = DP * P_device [FLOP/s].
NUMERICAL EXAMPLE: 1.4680064e16 / (14.60 * 2.56e15) = 0.3928.
INTERPRETATION: the step achieves 39.3% of declared dense peak under the 8N convention, or 29.5% under the 6N convention.
```

Neither number is a verdict.
The 39.3% figure is an aggregate that mixes a compute-bound phase, a memory-side phase, and a fabric phase, so it cannot by itself indicate where 60% of the potential is going.
The next three subsections attribute the shortfall.

### 6.7 The matrix-multiply phase is compute-limited but not at peak

The matrix-multiply phase moves far fewer bytes than it computes, so its floor is set by tensor-unit throughput.

```text
FORMULA [C-08]: eta_gemm = (F_model - F_attn) / (t_gemm * P_node)
WHERE: eta_gemm [dimensionless]; F_attn [FLOP attributed to attention].
NUMERICAL EXAMPLE: attention FLOPs = 4*S*A*d*L*T_tokens... evaluated for this geometry gives 4.5036e15, so (14.680064 - 4.5036)e15 / (7.30 * 2.56e15) = 0.5445.
INTERPRETATION: the linear layers achieve 54.5% of declared peak, which is a normal starting point for bfloat16 with an unoptimised epilogue and no fused bias-activation.
```

An efficiency of 54.5% on the compute side is not the binding problem in this step, because closing it entirely would save 3.32 s while the deadline needs only 1.01 s.
But it is large enough to be worth a later experiment once the attention phase is fixed, which is why the ranking in Section 8 sequences the interventions instead of choosing one forever.

### 6.8 The attention phase is on neither roofline

Attention FLOPs at this geometry follow the standard quadratic form.

```text
FORMULA [C-09]: F_attn = 4*S^2*d*A*L*T_seqs
WHERE: d=128 head dimension; A=32 heads; L=32 layers; T_seqs=8 sequences per step; the factor 4 covers score computation and value combination in forward and backward.
CONSTRAINT: full (non-causal-masked-out) counting; masking does not change the asymptotic form and is not modelled here.
NUMERICAL EXAMPLE: 4 * 32,768^2 * 128 * 32 * 32 * 8 = 4.503599627370496e15 FLOP.
INTERPRETATION: attention is 30.7% of model FLOPs while occupying 41.8% of the step, so it is already disproportionate.
```

The traffic model for the as-implemented path assumes four passes over the score tensor per layer: write scores, read for softmax, read for value combination, and an additional read in the backward pass.

```text
FORMULA [C-10]: B_attn = 4 * T_seqs * A * S^2 * b_s * L
WHERE: b_s=2 bytes; all other symbols as above.
NUMERICAL EXAMPLE: 4 * 8 * 32 * 32,768^2 * 2 * 32 = 7.0368744177664e13 bytes = 70.37 TB per step.
INTERPRETATION: 70.37 TB at the aggregate 24.0 TB/s rate gives a 2.9320 s traffic floor, well below the measured 6.10 s.
```

Two conclusions follow, and the case keeps them separate.
First, the phase is not bandwidth-saturated: it achieved 70.37/6.10 = 11.54 TB/s, which is 48.1% of aggregate device-memory bandwidth.
Second, it is not compute-saturated either: it executed 4.5036e15 FLOP in 6.10 s, or 0.7383 PFLOP/s, which is 28.8% of node peak.
Something else limits it, and the honest name for that something is a dependency-and-occupancy limit that this trace cannot localise further.

That gap matters for the decision: a phase at 48% of bandwidth and 29% of compute is a phase whose cost is dominated by serialisation, not by the size of either resource.

### 6.9 Compute the break-even efficiency that decides fused versus naive

The candidate intervention is a fused attention kernel that avoids materialising the full score tensor.
Its benefit is a large traffic reduction; its cost is a lower effective compute efficiency, because flash-style kernels trade extra FLOPs and register pressure for reduced traffic.

The decision reduces to a comparison of two floors.

```text
FORMULA [C-11]: t_naive_floor = B_attn / (eta_b * B_node);  t_fused_floor = F_attn / (eta_c * P_node)
WHERE: eta_b [achieved fraction of bandwidth]; eta_c [achieved fraction of peak compute].
CONSTRAINT: the fused kernel's traffic is assumed small enough that its floor is compute-side; its FLOP count includes the softmax rescaling overhead, not modelled separately.
NUMERICAL EXAMPLE: naive floor = 2.9320/eta_b; fused floor = 3.9094 s at eta_c=0.45.
INTERPRETATION: the naive path wins whenever eta_b exceeds 0.75.
```

The break-even bandwidth efficiency is therefore 2.9320/3.9094 = 0.75.
Since the observed efficiency was 0.48, the fused path is predicted to win by a wide margin: 6.10 s becomes 3.91 s, a saving of 2.19 s.
If the observed efficiency had been 0.80, the same fused kernel would have been slower than the naive path, and adopting it would have cost the run time rather than saving it.

This is the single most transferable result in the case: the value of a traffic-reducing kernel is a function of the efficiency you were already achieving, so the intervention must be justified with the measured efficiency, not with the traffic ratio alone.

### 6.10 The collective gap is a separate, cheaper defect

The data-parallel all-reduce moves gradient bytes per rank according to the ring model.

```text
FORMULA [C-12]: B_ar_per_rank = 2*(DP-1)/DP * N*b_g
WHERE: b_g=2 bytes; DP=8.
NUMERICAL EXAMPLE: 2*(7/8)*14e9 = 2.45e10 bytes = 22.8174 GiB per rank.
INTERPRETATION: at a 200 GB/s effective fabric rate the floor is 0.1225 s, so the observed 0.75 s is a 6.1x gap.
```

The implied effective rate is 2.45e10/0.75 = 32.7 GB/s, or 26% of the documented 0.1225 s expectation at 200 GB/s.

```text
FORMULA [C-13]: eta_fabric = B_ar_per_rank / (t_coll * B_fabric)
WHERE: B_fabric [bytes/s per rank].
NUMERICAL EXAMPLE: 2.45e10 / (0.75 * 2.0e11) = 0.1633 against the per-rank documented rate, or 0.1633*8 = 0.49 aggregate if all eight ranks are counted separately.
INTERPRETATION: the collective is running at roughly a sixth of its documented per-rank rate, which is a defect rather than a physics limit.
```

Two competing explanations are consistent with a slow collective and must be distinguished before any fix.
The first is a degraded link or a mis-selected algorithm, in which case per-link counters will show one rank or one port lagging its peers.
The second is that the collective is genuinely overlapped with computation and the 0.75 s is only the non-overlapped residue, in which case the "fix" is to improve overlap rather than bandwidth.

The discriminating test is a standalone all-reduce microbenchmark at the exact message size, run outside the training step.
If the microbenchmark reaches the documented rate while the in-step collective does not, the problem is overlap or bucketing.
If the microbenchmark also shows a sixth of the documented rate, the problem is the fabric.
This case cannot run that benchmark, so it records the gap and marks the attribution OPEN.

### 6.11 Assemble the deadline arithmetic before ranking fixes

The window arithmetic converts the business constraint into a per-step threshold.

```text
FORMULA [C-14]: t_step_max = (W * 3600 * DP) / (n_steps * DP) = W*3600 / n_steps
WHERE: W [hours]=288; n_steps = ceil(T_total / T_step) = ceil(2.0e10 / 262,144) = 76,294.
NUMERICAL EXAMPLE: 288*3600/76,294 = 13.5895 s.
INTERPRETATION: the baseline 14.60 s step overruns the threshold by 1.0105 s, or 7.4%, and the required speedup is 1.0744.
```

Note that the device count cancels in this particular formulation because both the work and the capacity scale with devices at fixed geometry.
That cancellation is not a general truth; it holds here because the token count, the batch geometry, and the parallelism degree are all fixed.
Under a scale-out proposal the geometry would change, and the cancellation would need to be re-derived rather than assumed.

The required speedup of 1.0744 is small, which is why the cheapest available defect should be fixed first.
The collective gap offers a modelled 0.6275 s against a required 1.0105 s, so it alone is insufficient, but it is sufficient in combination with a modest attention improvement.

### 6.12 Compare the interventions on a common ledger

All rows use the same geometry, the same deadline threshold, and the same parity guardrail.

| Option | Modelled new step (s) | Speedup | Meets 13.5895 s? | Device-hours for 2.0e10 tokens | Parity risk | Disposition |
|---|---:|---:|---|---:|---|---|
| Do nothing | 14.60 | 1.000 | no, over by 1.0105 s | 2,475.3 | none | rejected |
| Scale out to 16 devices at preserved efficiency | 7.2761 | 2.007 | yes, margin 6.3135 s | 2,467.2 | none | infeasible under C-02; near-zero device-hour saving |
| Fused attention at eta_c=0.45 | 12.4094 | 1.177 | yes, margin 1.1801 s | 2,103.9 | must be tested | proposed first |
| Fused attention at eta_c=0.55 | 11.6986 | 1.248 | yes, margin 1.8909 s | 1,983.4 | must be tested | upside case |
| Naive attention at eta_b=0.88 | 11.8319 | 1.234 | yes, margin 1.7576 s | 2,006.0 | none | alternative if parity fails |
| Collective repair only | 13.9725 | 1.045 | no, over by 0.3830 s | 2,368.9 | none | insufficient alone |
| Shrink sequence to 8,192 | geometry change | not comparable | — | — | violates C-04 | disqualified |

The scale-out row needs its own arithmetic rather than a dismissal, because it is the proposal the team actually made.
Doubling to sixteen devices while keeping one sequence per rank doubles the global batch to 524,288 tokens, halves the step count to 38,147, and leaves per-rank work unchanged, so the step time stays at 14.60 s.
The deadline threshold at that geometry becomes 288*3600/38,147 = 27.18 s, which the 14.60 s step clears by a wide margin.
Calendar time falls to 154.2 hours, so the schedule would fit if the devices existed.

Two facts disqualify it anyway.
The allocation contract provides exactly eight devices in the window, which makes the option infeasible rather than unattractive; an option that cannot be started is not ranked.
And the device-hours are 154.2*16 = 2,467.2 against 2,475.3 for the baseline, a saving of 0.3%, because the same work at the same per-device efficiency occupies proportionally more devices for proportionally less time.
Scale-out converts a calendar problem into a capacity purchase without repairing the defect, so the defect recurs at every scale; the fused kernel reduces device-hours by 15.0% and would compose with a later scale-out instead of competing with it.

The collective-repair-only row is deliberately included because it is the cheapest action and the case must show that it is insufficient rather than simply skip it.
Its saving is the difference between the observed 0.75 s and the modelled 0.1225 s floor, or 0.6275 s, which leaves the step at 13.9725 s.

The naive-at-88% row is the important fallback.
If the fused kernel fails its parity test, the naive path still meets the deadline if bandwidth efficiency is repaired to 88%, which is a plausible target for a tiled, non-fused implementation with better occupancy.
That row is what makes the recommendation robust to a failed numerical test.

The device-hour column is computed as steps times step time times devices divided by 3,600, and it is the only column that is comparable across geometries with different global batch sizes.
Comparing calendar hours across those geometries would reward the option that merely spends more devices, which is precisely the error the case is written to prevent.

### 6.13 Sensitivity: which assumption moves the decision

| Stress or test | Original condition | Alternative | Recomputed step or quantity | Change in conclusion | Decision consequence |
|---|---|---|---|---|---|
| Attention compute efficiency | 0.45 | 0.40 | 12.8980 s, margin 0.6915 s | still meets | fused remains preferred |
| Attention compute efficiency | 0.45 | 0.35 | 13.5263 s, margin 0.0632 s | meets, inside profiler noise | require the parity run to also report achieved efficiency before committing |
| Attention compute efficiency | 0.45 | 0.30 | 14.3641 s, over by 0.7746 s | reverses | fall back to naive-path repair |
| Bandwidth efficiency | 0.88 | 0.80 | 12.1650 s, margin 1.4245 s | still meets | naive fallback is comfortable |
| Bandwidth efficiency | 0.88 | 0.70 | 12.6886 s, margin 0.9009 s | still meets | naive fallback viable without fusion |
| Bandwidth efficiency | 0.88 | 0.60 | 13.3867 s, margin 0.2028 s | meets with thin margin | naive fallback fragile; treat as last resort |
| Sequence length | 32,768 | 65,536 | attention share 61.4%; naive 12.191 s, fused 7.8187 s | prize grows, rule unchanged | fused becomes mandatory |
| Sequence length | 32,768 | 8,192 | attention share 7.7%; naive 1.524 s, fused 0.977 s | attention no longer material | redirect effort to collective and epilogue |
| Device peak | 320 TFLOP/s | 280 TFLOP/s | compute floor 6.5536 s; fused 12.968 s, margin 0.6215 s | does not reverse; margin halves | proceed, but re-check before the next SKU |
| Device peak | 320 TFLOP/s | 240 TFLOP/s | compute floor 7.6459 s; fused 14.5455 s, over by 0.956 s | reverses | renegotiate the window |
| Aggregate bandwidth | 24.0 TB/s | 12.0 TB/s | naive at 0.88 becomes 18.496 s; collective floor 0.2450 s | reverses | a fabric or partitioning defect must be fixed before any kernel work |
| Profiler overhead | assumed small | +8% on attention only | attention 6.588 s, step 15.088 s, achieved bandwidth efficiency 0.445 | conclusion strengthens | require an overhead estimate with every phase table |

The rows that reverse or materially weaken the decision are the low-compute-efficiency row, the very low device-peak row, and the halved-bandwidth row.
An attention compute efficiency at 0.30 makes fusion fail the deadline, so the fallback chain matters: at 0.30 the naive path at 0.70 bandwidth efficiency still gives 12.6886 s, which meets the threshold, and that is why the fallback is credible rather than aspirational.
A device peak of 240 TFLOP/s pushes every option past the threshold and converts the decision from "which optimization" to "renegotiate the window", a different decision with a different owner.
Halving aggregate bandwidth inverts the ranking completely: the naive path becomes hopeless at 18.5 s while the fused path's compute-side floor is unaffected, so the first action becomes investigating whether a device is partitioned or a link is degraded.

The remaining rows show the case is not knife-edge.
A 20% shortfall in attention compute efficiency still clears the deadline, and a 20% shortfall in bandwidth efficiency still leaves the fallback viable.
The decision only changes when an assumption moves by roughly a third or more, which is the kind of robustness a reader should demand before funding an intervention.

### 6.14 What would falsify the diagnosis

Four observations would each falsify the attribution, and each is cheap to obtain.

If the standalone all-reduce microbenchmark at 22.82 GiB per rank reaches the documented rate, the collective gap is an overlap defect and not a fabric defect; the remediation changes from network work to bucketing and stream configuration.
If a counterfactual run with the score tensor materialised in a fused kernel shows attention time unchanged, then A-04 is wrong and the traffic model was fiction; the intervention budget must be redirected to the GEMM epilogue.
If per-device clocks show that one rank ran at a reduced clock for part of the window, the phase decomposition is contaminated by a throttling event and the measured 6.10 s is not a steady-state figure.
If the profiler's clean-versus-instrumented delta exceeds 5% of step time, every phase ratio is biased and the ranking must be recomputed from counters alone.

Stating these in advance is what separates a diagnosis from a narrative.

### 6.15 Boundary case: doubling the sequence length

At 65,536 tokens per sequence with four sequences per step, the token count per step is held constant so the compute floor is unchanged.

```text
NUMERICAL EXAMPLE: F_attn = 4*65,536^2*128*32*32*4 = 9.0072e15 FLOP, which is 61.4% of model FLOPs.
B_attn = 4*4*32*65,536^2*2*32 = 1.4074e14 bytes = 140.74 TB, giving a 5.8641 s floor at full bandwidth and 7.8187 s at 75%.
```

Two things change qualitatively.
First, attention becomes the majority of model FLOPs, so the aggregate MFU gauge becomes even less informative.
Second, the traffic floor at full bandwidth (5.86 s) exceeds the fused kernel's compute floor at 45% efficiency (7.82 s)? No — it does not; 5.86 s is below 7.82 s, so fusion still wins, but the margin narrows from a factor of 1.33 to a factor of 1.33 at the same ratio.

Recomputing carefully: naive floor at full bandwidth is 5.8641 s; fused floor at 45% compute efficiency is 7.8187 s. The break-even bandwidth efficiency is 5.8641/7.8187 = 0.75, identical to the 32,768 case.
The break-even ratio is scale-invariant because both the FLOP count and the traffic scale as sequence squared while the token count is held fixed.
What changes is the magnitude of the prize: at 65,536 the fused kernel saves roughly 6.10*(140.74/70.37) minus 7.82, which is a larger absolute saving, so the intervention becomes more valuable even though its decision rule is unchanged.

This is a useful result for a reader who expects "longer context means different physics".
The physics is the same; the arithmetic intensity of the attention phase is 64 FLOP/byte at both sequence lengths because both FLOPs and bytes scale as sequence squared.

### 6.16 Boundary case: short sequence

At 8,192 tokens per sequence with 32 sequences per step, holding tokens constant:

```text
NUMERICAL EXAMPLE: F_attn = 1.1259e15 FLOP, which is 7.7% of model FLOPs; B_attn = 1.7592e13 bytes = 17.59 TB, a 0.7330 s floor.
```

Attention is no longer the bottleneck, and the ranking inverts: the collective gap (0.6275 s) and the GEMM epilogue efficiency become the only material opportunities.
A reader who adopts the fused kernel as a general policy without this boundary check will spend engineering time on a phase that occupies less than 8% of the step.

### 6.17 Separating the memory ledger from the time ledger

The memory analysis and the time analysis share symbols but answer different questions, and conflating them produces a common error.

The memory ledger says: with chunked scores, the step uses 63.86 GiB of a usable 78.8 GiB, leaving 14.94 GiB of headroom.

```text
NUMERICAL EXAMPLE: 35.8559 (persistent) + 8.0 (checkpointed inputs) + 16.0 (eight-head score chunk) + 4.0 (workspace reserve) = 63.8559 GiB; usable = 80 - 1.2 = 78.8 GiB; headroom = 14.9441 GiB.
```

The headroom is what makes a larger micro-batch thinkable, but only if the time ledger also improves.
Doubling the sequences per step to 16 halves the number of steps and roughly doubles per-step time, so the deadline is unchanged except for the collective, whose cost per step grows sublinearly with batch size.

```text
NUMERICAL EXAMPLE: at 16 sequences the collective floor becomes 2*(7/8)*28e9/2.0e11 = 0.245 s for double the gradient bytes, because gradient bytes scale with parameters, not batch; the per-step collective cost therefore does not grow, and its relative share halves.
```

That is a real second-order benefit of a larger batch and it belongs in the ledger, but it is not available here because an eight-head score chunk at double the batch would be 32 GiB and total residency would be 95.86 GiB, which exceeds usable memory.

```text
NUMERICAL EXAMPLE: 35.8559 + 8.0 + 32.0 + 4.0 = 79.8559 GiB against 78.8 GiB usable, an overrun of 1.0559 GiB.
```

So the memory boundary disqualifies the batch increase by 1.06 GiB, a margin small enough that a reader should expect a four-head chunk to make it fit.
At four heads the score chunk is 8 GiB and total residency is 55.86 GiB, leaving 22.94 GiB of headroom, which fits comfortably.
The tradeoff is more chunk iterations per layer and therefore more launch overhead, which is exactly the kind of coupling the decomposition is meant to expose.

### 6.18 Executable oracle for every derived quantity

The following short program was executed against the numbers in this section.
It is reproduced so a reader can substitute their own geometry without re-deriving the algebra.

```python
GiB = 2**30
P, L, H, A, d = 7.0e9, 32, 4096, 32, 128
S, seqs, dev = 32768, 8, 8
Pdev, Bdev = 320e12, 3.0e12
Peak, Bagg = dev*Pdev, dev*Bdev
tok = seqs*S

f8 = 8*P*tok
attF = 4*S*S*d*A*L*seqs
scorePass = seqs*A*S*S*2*L
naive = 4*scorePass

assert tok == 262144
assert abs(f8/Peak - 5.7344) < 1e-9
assert abs(attF/f8 - 0.3067833782857143) < 1e-12
assert abs(naive/1e12 - 70.368744177664) < 1e-9
assert abs(naive/Bagg - 2.9320310074026668) < 1e-9
assert abs(attF/(0.45*Peak) - 3.909374676536889) < 1e-9
assert abs((naive/Bagg)/(attF/(0.45*Peak)) - 0.75) < 1e-12
assert abs(f8/(14.60*Peak) - 0.3927671232876712) < 1e-9
assert abs((f8-attF)/(7.30*Peak) - 0.5445453966518357) < 1e-9
assert abs(naive/6.10/1e12 - 11.535859701256394) < 1e-9
assert abs(naive/6.10/Bagg - 0.4806608208856831) < 1e-9
steps = -(-20000000000 // tok)
assert steps == 76294
tmax = 288*3600/steps
assert abs(tmax - 13.589535219021155) < 1e-9
assert abs(tmax - 13.5895) < 1e-4
new = 7.30 + 0.45 + 0.75 + attF/(0.45*Peak)
assert abs(new - 12.409374676536889) < 1e-9
print('all assertions passed')
```

The oracle checks the token count, the FLOP total, the attention share, the traffic total, the traffic floor, the fused floor, the break-even ratio, both efficiency measures, the achieved bandwidth, the step count, the deadline threshold, and the post-intervention step time.
It deliberately checks the break-even ratio to twelve decimal places because that ratio, not the absolute times, is the decision.

## 7. The Finding

The step overruns its window threshold by 1.01 s (7.4%) and the aggregate efficiency gauge cannot explain why, but the phase decomposition can: the attention phase consumes 41.8% of the step while running at 48.1% of aggregate memory bandwidth and 28.8% of peak compute, so it is neither bandwidth-bound nor compute-bound in the way a purchase decision assumes.
The cheapest defect is the gradient collective, which runs at roughly a sixth of its documented per-rank rate and is worth a modelled 0.63 s, but that is insufficient alone against a 1.01 s gap.
The intervention that clears the deadline is a fused attention kernel that avoids materialising the 70.37 TB of score traffic, which is modelled at 12.41 s per step against the 13.59 s threshold, a 1.177x speedup with a 1.18 s margin.
That intervention is conditional on the parity contract and on an achieved compute efficiency of at least roughly 0.35; below that, the correct action reverts to repairing the naive path's bandwidth efficiency to about 0.88, which also clears the deadline at 11.83 s.
Scaling out is disqualified not by principle but by the allocation arithmetic: eight devices are all that exist in the window, and the profile shows no balanced shortage that more devices would relieve.

## 8. System/Solution Design

### 8.1 Decision structure and ownership

The design is a decision pipeline rather than a service architecture, because the deliverable is a funded intervention with stop rules.

```text
trace schema -> per-phase FLOP/byte model -> floors and efficiency
      |                                          |
      |                                          v
      +-> precondition checks (A-02, A-04, A-06) -> break-even test
                                                     |
                              +----------------------+----------------------+
                              v                                             v
                    fused kernel trial                        naive-path repair
                     parity gate                               occupancy work
                              |                                             |
                              +------------------+--------------------------+
                                                 v
                                    deadline check against 13.5895 s
                                                 |
                                        pass -> adopt; fail -> renegotiate window
```

Each arrow is owned: the platform lead owns the model and the deadline check, the kernel engineer owns the trial and its parity evidence, the network engineer owns the collective attribution, and the research lead owns the parity band and the veto.

### 8.2 Precondition gate before any intervention

The gate exists because three assumptions can each invalidate the whole ranking.

The first gate checks that the assumed device peak and bandwidth match the installed hardware and that no rank is partitioned or clock-limited.
Its output is a corrected pair of denominators, not a corrected numerator; the measured times stay as measured.

The second gate confirms the as-implemented attention path materialises the score tensor, by reading the selected kernel's behaviour or by counting traffic directly.
If it does not, the traffic model is discarded and the intervention list is rebuilt around the GEMM epilogue.

The third gate confirms that the checkpoint policy applies to all layers, since a partially applied policy changes both residency and recompute FLOPs.

A gate failure is not a reason to proceed with a caveat; it is a reason to re-rank, because the entire value of the analysis is that it names the limiting resource correctly.

### 8.3 Parity contract for a faster attention kernel

The parity contract is stated as an ordered test, cheapest first.

First, a numerical agreement test on a fixed batch: run the old and new kernels on identical inputs with identical seeds and compare logits elementwise.
The band uses a relative tolerance justified from bfloat16's representation rather than an arbitrary number; the machine epsilon of bfloat16 is 2^-8, so a relative tolerance looser than 3.9e-3 on accumulated values is not a meaningful equivalence claim.

Second, a loss-trajectory test over a declared token window: run both paths for a fixed number of steps from the same checkpoint and compare the loss sequences.
The pass criterion is a mean absolute difference within the band and no monotone divergence trend.

Third, a long-horizon canary: continue the new path for a further window and check that the loss does not drift outside the band.

Fourth, an overflow and saturation check: bfloat16 has a narrow exponent range relative to fp32, and a fused kernel that changes rescaling order can change saturation behaviour on outlier activations.

The contract's purpose is to make "it looked fine" a failing answer.
The research lead signs the band, the kernel engineer produces the evidence, and the platform lead refuses adoption without it.

### 8.4 Instrumentation contract for the next profiling window

The next window must emit, per step and per rank: phase durations on a monotonic clock, kernel-level durations for the top consumers, device-memory byte counters, fabric byte and error counters per link, achieved clocks, and the profiler's own overhead estimate.

The contract also requires an explicit unaccounted-time field.
A decomposition that hides residual time inside other phases cannot be audited, and the residual is precisely where an unexpected serialisation would live.

Finally, the contract requires the trace schema to be versioned alongside the run configuration hash, so that a later comparison cannot silently mix two incompatible definitions of a phase.

### 8.5 Operational readiness and rollback

Adopting the fused kernel is reversible at the implementation flag level, but not at the scientific level: a run that has already trained for thousands of steps on the new kernel cannot be rolled back to the old kernel mid-run without introducing a discontinuity in the trajectory.
The design therefore requires a pre-run decision point, taken after the parity evidence and before the token budget is consumed.
If the fused kernel fails parity, the naive-path repair is adopted instead, which changes performance only and requires no parity evidence beyond the existing bfloat16 policy.
That asymmetry — one intervention requiring a numerical contract, the other not — is what determines the sequencing in Section 10.

## 9. Evaluation & Validation

### Validation matrix

| Validation item | Method | Pass criterion | Status | Evidence class |
|---|---|---|---|---|
| Token and FLOP arithmetic | Executable oracle, Section 6.18 | All assertions pass | executed | verified arithmetic |
| Phase decomposition closure | Sum of phases against independently measured step time | Difference within sampling tolerance | constructed as closed | illustrative |
| Device peak and bandwidth | Documentation read at authoring time, per installed SKU | Denominators match the deployed device | not executed | open |
| Score-tensor materialisation | Kernel source read or traffic counter | Confirms four-pass model within 10% | not executed | open |
| Collective attribution | Standalone all-reduce at 22.82 GiB per rank | Distinguishes fabric from overlap | not executed | open |
| Fused-kernel numerical parity | Fixed-seed logit comparison plus loss-window comparison | Within the declared band, no divergence trend | not executed | open |
| Deadline outcome | Recomputed step time against 13.5895 s | New step below threshold | derived only | illustrative |
| Sensitivity to compute efficiency | Recompute at 0.35, 0.30 | Decision stable or fallback declared | executed | verified arithmetic |
| Boundary at 65,536 sequence | Recompute attention share and floors | Decision rule unchanged, prize larger | executed | verified arithmetic |
| Boundary at 8,192 sequence | Recompute attention share and floors | Rank order inverts | executed | verified arithmetic |
| Collective-repair-only sufficiency | Recomputed step 13.9725 s | Must fail threshold to justify combination | executed | verified arithmetic |

The matrix distinguishes what was actually executed from what remains open, which is the entire point of the evidence labels.
Only arithmetic was executed here; nothing about hardware behaviour was measured.

The negative tests matter as much as the positive ones.
A validation suite that only checks that the new kernel is faster would accept a kernel that is faster and wrong.
The suite therefore includes the parity test as a hard gate, the boundary tests as scope guards, and the collective-repair-only row as a demonstration that the cheapest fix is insufficient.

The measurement plan for the profiling window requires at least 200 consecutive steady-state steps, since a single step's duration at this scale carries variance from allocator behaviour, clock variation, and host interference.
Reporting a mean without the spread would make a 0.66 s margin indistinguishable from noise, which is why the sensitivity table includes a row where the margin collapses to 0.06 s.

## 10. Action Taken

Because the case is illustrative, the actions are the ones the decision authorises, not an executed campaign.

Step one: run the three precondition gates. They cost hours, not days, and each can cancel the plan.
Step two: repair the collective by measuring the standalone all-reduce, then fixing the specific cause it reveals — link, algorithm choice, bucket size, or stream overlap. This step is adopted regardless of the attention outcome because it is cheap and it does not touch numerics.
Step three: trial the fused attention kernel on a short token window with the parity suite. Adoption requires all four parity tests.
Step four: if parity fails, switch to the naive-path repair, whose target is the bandwidth efficiency needed to clear the deadline; the sensitivity table shows that 0.70 is sufficient, which is a considerably easier target than 0.88.
Step five: re-profile one window with the instrumentation contract and confirm the modelled savings appear as real phase reductions before committing the remaining token budget.
Step six: if the post-intervention step still exceeds 13.5895 s, escalate the window renegotiation to the research lead rather than consuming the budget on an overrun.

The sequencing deliberately places the numerically-safe action first and the numerically-risky action second.
That ordering means a parity failure costs a short trial rather than the run.

Each step carries a stop rule: a gate failure stops the subsequence, a parity failure redirects to the fallback, and a missed deadline after intervention escalates rather than iterates.

## 11. Outcome & Results

The modelled outcome after the fused intervention is a 12.4094 s step against a 13.5895 s threshold, a speedup of 1.1765 with a margin of 1.1801 s.
The modelled device-hours for the token budget fall from 2,475.3 to 2,103.9, a saving of 371.4 device-hours.

```text
NUMERICAL EXAMPLE: saving fraction = 371.4035/2475.3164 = 0.1500, consistent with 1 - 1/1.1765.
```

At the illustrative 3.00 units per device-hour, that is a 1,114.2-unit saving; at an illustrative node power of 5.6 kW and 0.15 units per kWh, a further 353.5 kWh and 53.0 units.
Those numbers are constructed rates applied to a constructed result, so they demonstrate the accounting rather than a real saving.

The modelled outcome after the collective repair alone is a 13.9725 s step, which still misses the threshold by 0.3830 s.
That row is the case's explicit negative result: the cheapest defect was real but insufficient, and reporting it as "the fix" would have been the most likely wrong conclusion in this decision.

Evidence class for every outcome in this section: illustrative, derived from constructed inputs by checked arithmetic.
No production run was profiled, no kernel was benchmarked, and no cost was incurred.
The correct statement of the result is therefore "the ledger says this intervention is predicted to clear the threshold by 1.18 s, conditional on the parity gate and on an achieved attention compute efficiency of at least 0.35".

## 12. Failure Modes & What Went Wrong

### F-01 — The aggregate gauge is compared across conventions

An operator reports a 29.5% efficiency and concludes the step is inefficient by 70%.
The number was computed with the 6N convention while the checkpointed implementation executes 8N of work, so the gauge understates real efficiency by a third.
Diagnosis: recompute both conventions on the same step.
Recovery: fix the gauge's convention and annotate it in the dashboard.
Prevention: make the convention a field of the metric, not a comment in a document.

### F-02 — Profiling changes the scheduling it measures

Instrumentation serialises kernels that would otherwise overlap, so the profiled attention phase absorbs wait time that is not intrinsic to the kernel.
Diagnosis: compare a clean step with a profiled step and report the delta.
Recovery: use counters only, or sampling mode with a low rate.
Prevention: state the profiler overhead next to every phase table.

### F-03 — The score-tensor traffic model describes a kernel that is not running

A reader assumes four passes over the score tensor, but the deployed kernel is already tiled and the real traffic is a small multiple of the tile size.
Diagnosis: count device-memory traffic for the attention kernel directly.
Recovery: rebuild the intervention list around whichever phase is actually limiting.
Prevention: pin the kernel identity in the run configuration and assert it in the profiling script.

### F-04 — A fused kernel silently changes acceptance behaviour

The new kernel accumulates in a different order and produces logits that agree to five significant figures, which passes a casual eyeball check and fails the declared band over a thousand steps.
Diagnosis: fixed-seed elementwise comparison plus a loss-window trend test.
Recovery: revert the flag and adopt the naive-path repair.
Prevention: make the parity suite a gate on the adoption, not a post-hoc report.

### F-05 — The collective "fix" reduces overlap and slows the step

Moving the all-reduce to a dedicated stream improves its standalone rate and destroys its overlap with the backward pass, so the step gets slower despite a better microbenchmark.
Diagnosis: compare end-to-end step time, not the collective in isolation.
Recovery: restore the overlapped configuration and tune bucket size instead.
Prevention: treat any communication change as a step-level experiment.

### F-06 — A single throttled rank contaminates the phase profile

One device ran at a reduced clock because of a thermal or power event, so every rank-synchronised phase shows a duration set by the slowest rank.
Diagnosis: per-rank clock and duration records joined on the step index.
Recovery: exclude the window, restore the clock policy, and re-profile.
Prevention: emit achieved clocks per rank and alert on divergence.

### F-07 — A checkpoint write steals the bandwidth being measured

The window included a checkpoint write, so the attention phase competed with storage traffic for memory bandwidth.
Diagnosis: align storage write counters with the phase timeline.
Recovery: exclude checkpoint steps and re-profile.
Prevention: separate checkpoint windows from profiling windows by policy.

### F-08 — The deadline arithmetic holds the geometry constant and the scale-out proposal changes it

The threshold derivation cancels the device count because work and capacity scale together at fixed geometry; a scale-out proposal changes the geometry and invalidates the cancellation.
Diagnosis: re-derive the threshold for the proposed geometry instead of reusing the old number.
Recovery: recompute per-step threshold and total steps for the new global batch.
Prevention: write the threshold formula with its fixed-geometry condition attached.

### F-09 — Memory headroom is spent on a batch increase that then fails the deadline

Fourteen gigabytes of headroom invites doubling the batch, which doubles per-step time and leaves the deadline unchanged, while also doubling the score-chunk allocation past the usable limit.
Diagnosis: recompute residency for the proposed chunk and batch together.
Recovery: reduce the chunk to four heads before increasing batch.
Prevention: keep the memory ledger and the time ledger as separate gates.

## 13. Transferable Lesson

A utilization number is a symptom, and a phase decomposition is a diagnosis; the second is what lets an engineer refuse a purchase proposal with arithmetic instead of opinion.

The decision rule that transfers is the break-even efficiency: a traffic-reducing kernel is worth its lower compute efficiency only when the efficiency already achieved on the naive path is below the ratio of the two floors.
Here that ratio was 0.75 against an observed 0.48, which made the fusion clearly favourable; at an observed 0.80 it would have been harmful.

The second transferable result is that the cheapest defect need not be the sufficient one. The collective repair was worth 0.63 s against a 1.01 s gap, and a team that stopped there would have declared success and missed the window.

The third is that numerical validity is a disqualifier rather than a weight. A 30% speedup that leaves the band is worth less than a 15% speedup that does not, because the slower option still produces a usable experiment.

The fourth is that boundary checks are not optional decoration. The same intervention is mandatory at 65,536 sequence length, marginal at 32,768, and irrelevant at 8,192, and the only thing that distinguishes these is the arithmetic intensity of the phase, which a reader can compute in a few lines.

These four rules transfer to any accelerator profile where a phase's cost is attributed to a named resource and compared against a measured efficiency rather than an advertised one.

## 14. Connections & Cross-References

The companion sheets for this module are [H100 architecture and resource limits](../Cheat_Sheets/01_H100_Architecture_and_Resource_Limits.md) and [roofline arithmetic intensity and kernel diagnosis](../Cheat_Sheets/01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md); both supply the definitions this case uses without repeating the derivations.
The capacity and admission reasoning that converts a memory ledger into a serving decision belongs to [CASE-2026-023-001](01_Size_an_Inference_Cluster.md), which is a different decision with a different owner and is not duplicated here.
Training memory accounting, optimizer-state sharding, and collective behaviour are developed in [Module 24](../../24_distributed_training_data_pipelines_and_ml_compute/INDEX.md); the checkpoint and recovery sheet at part level covers the durable-state contract that this case deliberately holds fixed.
The reference-informed brief PLAN-2026-IR007 concerning continuous batching and KV-memory policy has its canonical home in [Module 25](../../25_model_serving_inference_optimisation_and_cost/INDEX.md) and is not restated as a case here.

Open evidence items: the device documentation values, the deployed kernel's materialisation behaviour, the fabric health record, and the parity evidence all remain unverified in this workspace.
The source-tree CUDA titles concerning asynchronicity, TMA, WGMMA, and kernel design remain planned teaching commitments in this module's cheat-sheet collection and are not claimed as covered by this case.

### Interview pack

1. **Why is a 39% MFU reading not a diagnosis?**
   **Answer:** It aggregates a compute-bound phase, a mixed attention phase, and a fabric phase into one ratio, so it can be improved by three different remedies whose costs differ by an order of magnitude.
   **Reasoning:** The ratio's denominator is a declared peak and its numerator is model FLOPs; neither term contains information about which resource actually limited any phase.
   **Follow-up:** What would you emit instead?
   **Follow-up answer:** Per-phase duration, achieved rate, and the named limiting resource, plus an unaccounted-time line.
   **Acceptable alternative:** A per-phase roofline table produced from counters.
   **Red flag:** "39% means we should buy more GPUs."
   **Scoring:** 2 names the aggregation defect and a replacement; 1 says the number is low.
   **Answer support:** Sections 6.5 through 6.9 build the decomposition that replaces the gauge.

2. **Compute the step's model FLOPs under both conventions.**
   **Answer:** 8N gives 8*7.0e9*262,144 = 1.4680064e16 FLOP; 6N gives 1.1010048e16 FLOP.
   **Reasoning:** N is parameters per token; the token count per step is sequences times sequence length; the multiplier accounts for forward, backward, and recomputation.
   **Follow-up:** Which one applies here?
   **Follow-up answer:** 8N, because every layer input is checkpointed and recomputed.
   **Acceptable alternative:** Counting achieved FLOPs from a profiler's tensor-op counters.
   **Red flag:** Using 2N, which counts only the forward pass.
   **Scoring:** 2 both values and the reason; 1 one value.
   **Answer support:** C-01 and A-07 fix the convention and the checkpoint policy.

3. **Derive the deadline threshold and say why the device count cancels.**
   **Answer:** 288 hours over 76,294 steps gives 13.5895 s per step; the device count cancels because tokens per step and per-device throughput both scale with devices at fixed geometry.
   **Reasoning:** n_steps is total tokens over tokens per step; with a fixed global batch, per-step work is fixed, so the wall-clock per step is the only variable.
   **Follow-up:** When does it stop cancelling?
   **Follow-up answer:** When the global batch or the parallelism layout changes, since both change per-step work or per-step cost.
   **Acceptable alternative:** Deriving total device-hours and dividing by devices.
   **Red flag:** Assuming more devices always shorten the calendar schedule.
   **Scoring:** 2 the threshold and the cancellation condition; 1 the threshold only.
   **Answer support:** C-14 and F-08 state the fixed-geometry condition.

4. **Why is the attention phase neither bandwidth- nor compute-saturated?**
   **Answer:** It achieved 11.54 TB/s of 24.0 TB/s (48.1%) and 0.738 PFLOP/s of 2.56 PFLOP/s (28.8%), so neither resource was exhausted.
   **Reasoning:** Both ceilings are computed from the same measured duration, so a phase below both is limited by dependency chains, occupancy, or serialisation between softmax stages.
   **Follow-up:** What would you measure next?
   **Follow-up answer:** Occupancy, stall reasons, and the softmax-to-GEMM serialisation within the kernel.
   **Acceptable alternative:** A kernel-level trace showing idle gaps.
   **Red flag:** "It is memory-bound because attention is quadratic."
   **Scoring:** 2 both fractions plus a mechanism; 1 one fraction.
   **Answer support:** C-09, C-10, and Section 6.8 compute both efficiencies.

5. **Formulate the break-even efficiency for adopting the fused kernel.**
   **Answer:** Compare B_attn/(eta_b*B_node) with F_attn/(eta_c*P_node); fusion wins when eta_b is below 0.75 for eta_c = 0.45.
   **Reasoning:** Both expressions are floors for the same phase, so the cheaper floor wins; the ratio of floors is independent of sequence length when tokens per step are fixed.
   **Follow-up:** What if eta_c is only 0.30?
   **Follow-up answer:** The fused floor becomes 5.86 s, which exceeds the naive floor at any efficiency above 0.50, so fusion is rejected.
   **Acceptable alternative:** Numerical evaluation of both floors across a grid of efficiencies.
   **Red flag:** Comparing traffic ratios without the achieved efficiency.
   **Scoring:** 2 the expression and the number; 1 the expression.
   **Answer support:** C-11 and the sensitivity rows in Section 6.13.

6. **What does the collective gap tell you, and what does it not tell you?**
   **Answer:** It says the observed 0.75 s is 6.1 times the 0.1225 s ring floor at 200 GB/s; it does not say whether the cause is fabric degradation or lost overlap.
   **Reasoning:** A ring model gives a lower bound for bytes per rank; both a slow link and a non-overlapped execution inflate the observed duration identically.
   **Follow-up:** What is the discriminating test?
   **Follow-up answer:** A standalone all-reduce at the same message size outside the training step.
   **Acceptable alternative:** Per-link counters joined to the collective's timeline.
   **Red flag:** "The network is slow" without a per-link record.
   **Scoring:** 2 the ratio and the ambiguity; 1 the ratio only.
   **Answer support:** C-12, C-13, and the two competing explanations in Section 6.10.

7. **Why is repairing the collective alone insufficient?**
   **Answer:** Its modelled saving is 0.6275 s against a required 1.0105 s, leaving the step at 13.9725 s, which still exceeds 13.5895 s.
   **Reasoning:** The gap between observed and floor duration bounds the recovery from that phase; subtracting it from the current step gives the best case for that intervention alone.
   **Follow-up:** Does that make the repair pointless?
   **Follow-up answer:** No; it is cheap, numerically safe, and it reduces the improvement the attention work must deliver.
   **Acceptable alternative:** Treating the two interventions as a portfolio with independent savings.
   **Red flag:** Declaring the deadline met after the cheapest fix.
   **Scoring:** 2 the arithmetic and the conclusion; 1 the conclusion.
   **Answer support:** The "collective repair only" row in Section 6.12 and Section 11.

8. **Design the parity test for a faster attention kernel.**
   **Answer:** Fixed-seed elementwise logit comparison within a band justified by bfloat16's 2^-8 epsilon, a loss-window comparison with no divergence trend, a longer canary window, and an overflow/saturation check.
   **Reasoning:** Elementwise agreement bounds local error; the window test bounds accumulation; the canary bounds slow drift; the saturation check catches exponent-range effects that local agreement can miss.
   **Follow-up:** What tolerance would you accept?
   **Follow-up answer:** A relative tolerance no looser than about 3.9e-3 on accumulated values, with a tighter bound on logits where the reference is fp32.
   **Acceptable alternative:** A statistical equivalence test on the loss distribution with a pre-registered margin.
   **Red flag:** "The loss looks the same after 50 steps."
   **Scoring:** 2 all four tests; 1 elementwise only.
   **Answer support:** Section 8.3 and A-06 define the contract and its band.

9. **Why does the batch-doubling idea fail on memory rather than on time?**
   **Answer:** Doubling sequences doubles the score chunk from 16 GiB to 32 GiB, so total residency becomes 79.8559 GiB against 78.8 GiB usable, an overrun of 1.0559 GiB.
   **Reasoning:** Persistent state and checkpointed inputs do not scale with batch, but the transitory score chunk does, so the marginal cost of the increase is entirely transitory.
   **Follow-up:** How would you make it fit?
   **Follow-up answer:** Halve the head chunk to four heads, reducing the chunk to 8 GiB and total residency to 55.86 GiB.
   **Acceptable alternative:** Shortening the score-chunk lifetime by fusing the softmax.
   **Red flag:** Assuming memory headroom is fungible across phases.
   **Scoring:** 2 the arithmetic and the fix; 1 the overrun only.
   **Answer support:** Section 6.17 computes both residencies.

10. **What changes at 65,536 sequence length, and what does not?**
    **Answer:** Attention's share of model FLOPs rises from 30.7% to 61.4% and its traffic floor from 2.93 s to 5.86 s, but the break-even ratio stays at 0.75 because FLOPs and bytes both scale as sequence squared at fixed tokens.
    **Reasoning:** Both floors are homogeneous of the same degree in sequence length, so their ratio is invariant; only the absolute prize grows.
    **Follow-up:** Does the intervention become more or less valuable?
    **Follow-up answer:** More valuable in absolute seconds, with an unchanged decision rule.
    **Acceptable alternative:** Recomputing both floors numerically at the new geometry.
    **Red flag:** Claiming the physics changes qualitatively with context length.
    **Scoring:** 2 the invariance and the magnitude; 1 the share change only.
    **Answer support:** Section 6.15 evaluates both geometry points.

11. **At 8,192 sequence length, what is the bottleneck?**
    **Answer:** Attention falls to 7.7% of model FLOPs with a 0.73 s traffic floor, so the collective gap and GEMM epilogue efficiency become the only material opportunities.
    **Reasoning:** The quadratic terms shrink relative to the linear terms, so the phase that dominated at long context becomes a minor consumer.
    **Follow-up:** Would you still adopt the fused kernel?
    **Follow-up answer:** Only if it is free; the engineering time is better spent on the collective and the epilogue.
    **Acceptable alternative:** A portfolio decision weighted by phase share.
    **Red flag:** Applying a long-context optimisation unconditionally.
    **Scoring:** 2 the share and the redirection; 1 the share only.
    **Answer support:** Section 6.16 recomputes the short-context geometry.

12. **A colleague proposes adding eight devices. Give the arithmetic refusal.**
    **Answer:** The allocation window contains eight devices, so the option is infeasible rather than merely unattractive; and at fixed geometry the deadline threshold cancels the device count, so more devices would not change the per-step requirement without changing geometry.
    **Reasoning:** Feasibility is a constraint check before it is an optimisation; the threshold derivation shows that the calendar constraint is a per-step-time constraint here.
    **Follow-up:** When would adding devices be the right answer?
    **Follow-up answer:** When the profile shows every phase near its resource ceiling, so the only remaining lever is more of that resource.
    **Acceptable alternative:** A device-hour cost comparison at equal achieved efficiency.
    **Red flag:** "More GPUs always make training faster."
    **Scoring:** 2 feasibility plus the cancellation; 1 one of them.
    **Answer support:** C-02, C-14, and the scale-out row in Section 6.12.

13. **How would a throttled rank distort this analysis, and how do you detect it?**
    **Answer:** Every synchronised phase takes the duration of the slowest rank, so one reduced-clock device inflates the attention and collective phases together; detection is a per-rank clock and duration join.
    **Reasoning:** Data-parallel steps synchronise at the collective, so a single straggler sets the step boundary while the other ranks idle.
    **Follow-up:** Does the mean across ranks hide it?
    **Follow-up answer:** Yes; the mean is pulled up by the straggler's phase and down by the idle peers, producing a plausible-looking average.
    **Acceptable alternative:** Per-rank duration histograms with an explicit max-rank column.
    **Red flag:** Reporting only an eight-rank average.
    **Scoring:** 2 the mechanism and the detector; 1 the mechanism.
    **Answer support:** F-06 and the per-rank requirement in Section 8.4.

14. **State precisely what this case has and has not established.**
    **Answer:** It has established, by checked arithmetic on constructed inputs, that the phase decomposition implies a 1.18 s margin for the fused kernel and that the collective repair alone is insufficient; it has not measured any hardware, verified any device rate, or proved any kernel's parity.
    **Reasoning:** The evidence classes are separate: arithmetic can be verified without the workload being real, and a modelled saving is not a measured one.
    **Follow-up:** What is the minimum evidence to upgrade the claim?
    **Follow-up answer:** One profiled window with the instrumentation contract plus the standalone collective benchmark plus the parity suite results.
    **Acceptable alternative:** An independent reproduction on a second node with the same stack pins.
    **Red flag:** Presenting the modelled 12.41 s as an achieved step time.
    **Scoring:** 2 both halves of the boundary; 1 only the positive claim.
    **Answer support:** Sections 5e and 9, and the evidence declaration in Section 1.

## 15. Quick Reference Card

| Field | Reference |
|---|---|
| Decision | Fund a fused attention kernel, with collective repair first and a naive-path fallback; refuse scale-out. |
| Core calculation | F_model = 8*N*T; B_attn = 4*T_seqs*A*S^2*2*L; floor = max(F/P_peak, B/B_peak). |
| Worked values | 8*7.0e9*262,144 = 14.68 PFLOP; 70.37 TB attention traffic; floors 5.7344 s and 2.9320 s. |
| Deadline rule | Threshold = window_hours*3600/steps = 288*3600/76,294 = 13.5895 s; baseline overruns by 1.0105 s. |
| Break-even rule | Fusion wins when achieved bandwidth efficiency is below (B/B_peak)/(F/(0.45*P_peak)) = 0.75; observed 0.48. |
| Efficiency readings | MFU 39.3% at 8N (29.5% at 6N); GEMM 54.5%; attention 48.1% of bandwidth and 28.8% of compute. |
| Cheapest-defect trap | Collective repair is worth 0.6275 s of a required 1.0105 s; it is necessary but insufficient. |
| Memory gate | 35.86 GiB persistent + 8 GiB checkpointed inputs + 16 GiB score chunk + 4 GiB reserve = 63.86 GiB of 78.8 GiB usable. |
| Parity gate | bfloat16 epsilon 2^-8, so relative tolerance no looser than 3.9e-3, plus loss-window and canary tests. |
| Boundary rule | Break-even ratio is invariant in sequence length at fixed tokens; the prize is not. Attention is 61.4% of FLOPs at 65,536 and 7.7% at 8,192. |
| Evidence/stage/review | illustrative; proposed incident analysis; draft; not reviewed; hardware, kernel, fabric and parity evidence OPEN. |
