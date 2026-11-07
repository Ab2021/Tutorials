---
lesson_id: "FT-M01-04"
lesson_title: "Mixture of Experts"
version: "1.0-draft"
created: "2026-11-07"
reviewed: null
status: "draft"
evidence_basis: "primary research papers and constructed routing examples"
owner_role: "curriculum maintainer"
review_trigger: "Change to routing, capacity, expert parallelism, or token-overflow behavior"
---

# 1.4 — Mixture of Experts

> **Course location:** Module 19, Transformers, LLMs and Foundation-Model Internals. This lesson maps to `finetuning_course/Module_01_Transformer_Architecture/1.4_Mixture_of_Experts.md` in [`Train/Templates/directory.txt`](../../../Templates/directory.txt). The source filename is a scope cue; the original lesson body was not inspected.
>
> **Status:** Draft. All counts and routing examples below are illustrative; no model, router, accelerator, or distributed workload was executed.

## Learning outcomes

Learners can explain token-routed sparse feed-forward layers; distinguish expert count, total parameters, active parameters and total system work; calculate simple routing and capacity quantities; identify why routing imbalance undermines expected efficiency; and specify correctness/performance tests for a MoE implementation.

## 1. Dense feed-forward layer versus routed experts

A dense transformer FFN applies the same feed-forward weights to every token. A mixture-of-experts (MoE) layer replaces or supplements that FFN with multiple expert networks and a router. The router scores candidate experts for each token; a selection rule chooses one or more experts; the token is dispatched; expert outputs are combined and returned to the original token position.

A simplified token path is:

```text
hidden state for token
      ↓
router scores over E experts
      ↓ top-k selection and routing weights
dispatch token representation to selected experts
      ↓
expert-specific FFN computations
      ↓ weighted combination
return output to token's sequence position
```

MoE is called sparse when only a subset of expert computations is used for each token. The full expert bank still exists as model capacity. Sparse compute does not automatically imply low checkpoint storage, low communication, low latency, or better quality.

MoE is an architectural family, not one fixed algorithm. Designs differ in router function, top-k, capacity policy, auxiliary losses, expert structure, shared experts, token dropping, routing granularity, communication strategy, and inference batching. A model name or “top-2” statement is insufficient to infer these details.

## 2. Router scoring and top-k selection

Let a token hidden vector be `h ∈ R^d` and let there be `E` experts. A simple linear router computes logits `r = h W_r + b_r`, with `W_r` shaped `[d,E]` under this row-vector convention. A softmax can transform router logits into expert probabilities `g_e`; top-k selection selects the largest `k` experts under a specified tie-breaking and normalization policy.

One common conceptual output is a weighted combination of selected expert outputs:

```text
y(h) = sum_{e in S_k(h)} w_e(h) * Expert_e(h)
```

`S_k(h)` is the selected expert set; `w_e` are routing weights, which may be raw selected probabilities or renormalized over selected experts depending on the implementation. The weighting convention is part of the architecture contract.

A deterministic top-k router can route identical inputs identically in inference, but training may include noise or other balancing mechanisms. Router computation is small relative to a large expert MLP in some designs, but dispatch, sorting, communication, capacity management, and synchronization can dominate end-to-end latency.

### Router weight example

For `d=512` and `E=8`, a bias-free linear router has `512*8=4,096` weights. Adding an 8-element bias yields 4,104 parameters. This count excludes auxiliary router projections, noise parameters, expert networks, and implementation metadata. It does not establish that routing is computationally free.

## 3. Total experts, selected experts, and parameter count

Assume each expert is a conventional two-matrix FFN with hidden width `d` and intermediate width `d_ff`. One expert has `2*d*d_ff` matrix weights, excluding biases. With `E` independent experts, total expert matrix weights are `E*2*d*d_ff`. If top-k `k` experts run for each token, the selected expert matrix weights used on one token are nominally `k*2*d*d_ff` (again excluding router and other shared components).

Validity: independent equal-sized experts, fixed top-k execution, conventional two-matrix FFN, and no shared expert or conditional substructure.

Derivation: count expansion and contraction matrices in one expert; multiply by expert count for total bank; multiply by selected count for the expert-path weights invoked per token. This invocation count is not the amount of memory resident and not a complete FLOP count.

```text
FORMULA [MOE-01]: P_expert,total = E * 2*d*d_ff
                 P_expert,active/token ≈ k * 2*d*d_ff
WHERE: E = expert count; k = selected experts/token; d = hidden width;
       d_ff = expert intermediate width; P = parameter count.
CONSTRAINT: independent equal experts and conventional two-matrix FFNs; excludes router, shared experts, biases, norms and other model weights.
NUMERICAL EXAMPLE: E=8, k=2, d=512, d_ff=2048.
                   One expert=2*512*2048=2,097,152 weights.
                   Total expert bank=8*2,097,152=16,777,216 weights.
                   Selected expert weights/token=2*2,097,152=4,194,304.
INTERPRETATION: each token invokes a nominal quarter of the expert bank, but all eight experts remain part of total capacity and typically must be stored or sharded.
```

Independent check: `k/E=2/8=0.25`; `4,194,304/16,777,216=0.25`. The active expert path is twice the matrix count of one dense FFN of the same `d` and `d_ff`, before router, dispatch and communication costs. Therefore “only top-2 of 8” does not mean half the dense FFN work or one quarter of total system cost.

## 4. Routing load and capacity

Across a batch of `T` tokens with top-k `k`, the number of token-to-expert assignments before capacity limits is `T*k`. If assignments were perfectly balanced over `E` experts, the average load would be `T*k/E`. Real routing can be skewed; the average does not bound the maximum expert load.

Implementations may provision per-expert capacity, often based on a capacity factor `C` times the expected average. A generic illustrative form is:

```text
capacity_per_expert = ceil(C * T * k / E)
```

Exact conventions differ: implementations may pad capacities, use per-device token counts, route second-choice experts, drop excess assignments, reroute tokens, or use different capacity definitions. Consult the pinned implementation and test actual semantics.

Illustrative values: `T=128` tokens, `k=2`, `E=8`, `C=1.25` yield average assignments `128*2/8=32` per expert and nominal capacity `ceil(1.25*32)=40` assignments per expert. Aggregate capacity is `8*40=320`, while there are 256 assignments. The 64 extra slots are aggregate headroom, not a guarantee that no expert overflows: one expert could receive more than 40 assignments while others are underused.

Load metrics should include assignment counts, capacity utilization, fraction of tokens dropped or rerouted, router probability distribution, per-expert compute, and imbalance over time and across devices. An average load alone can hide an overloaded expert.

## 5. Balancing objectives and tradeoffs

A router trained only to minimize the main task loss may collapse onto a small subset of experts, leaving other experts underused. Auxiliary balancing losses or routing constraints can encourage more even assignment. They introduce a tradeoff: overly strong balancing can force tokens to experts that are less useful for their content or destabilize task optimization.

Balancing is not synonymous with equal utilization at every instant. The desired behavior depends on capacity, topology, batch, sequence, data mixture, and expert specialization. A diagnostic should compare routing distributions with quality and compute, not optimize a histogram in isolation.

Some MoE systems use noisy or stochastic routing during training to increase exploration; inference routing may be deterministic. This creates a train/serve contract to validate. Reproducing only the inference router does not reproduce training dynamics; reproducing only training-time load metrics does not prove serving latency.

## 6. Dispatch, combine, and distributed placement

A routing implementation commonly groups token assignments by expert, transfers or gathers token states to expert computation locations, evaluates experts, and scatters/combine outputs back to token positions. If experts are distributed across devices, routing can require all-to-all communication. Token ordering and inverse mapping must be correct even when tokens are padded, dropped, rerouted, or sent to multiple experts.

Mechanism invariants:

- Every admitted token has an explicit routing outcome under the configured policy.
- Every selected expert assignment is associated with the original token ID and sequence position.
- The combine stage uses the intended router weights and returns each result to the correct token.
- Capacity handling is deterministic or documented for replay; overflow behavior is observable.
- Distributed dispatch preserves exactly the configured top-k assignment semantics across ranks.
- A failed rank or communication operation does not silently produce a partial output treated as valid.

A small reference implementation can act as an oracle: represent tokens as integer IDs, compute simple expert functions, route a tiny batch, then verify selected experts, assignment counts, returned ordering, weighted combination, and overflow policy. Production tests must also cover distributed ranks, uneven loads, empty expert batches, and request/session isolation.

## 7. Compute, memory, and communication

For a token assigned to `k` experts, expert arithmetic is approximately `k` expert-FFN computations, plus router and dispatch overhead. If every expert resides on a separate device, activations may need to move between devices; communication volume scales with routed token representations and number of assignments, subject to topology and implementation.

Total expert parameter storage scales with `E`. Active per-token expert compute scales approximately with `k`, but this comparison assumes equal experts and ignores router and infrastructure work. Expert parallelism can increase memory capacity by distributing weights, while increasing network traffic and synchronization. Replication may lower communication but uses more memory. Capacity padding can waste compute; token dropping can change model output and task quality.

A systems comparison should report at least:

- total and active parameter definitions;
- weight bytes and expert placement;
- selected assignments per token;
- maximum and percentile expert load;
- token-drop/reroute rates;
- all-to-all bytes and time;
- expert compute time and idle time;
- end-to-end throughput and latency under realistic batch/concurrency;
- quality at matched evaluation data, token budget and compute/cost comparator.

A FLOP estimate derived only from active experts ignores the cost of moving tokens and coordinating routing. A memory estimate derived only from active experts ignores the rest of the expert bank.

## 8. Dense versus MoE selection

| Constraint or goal | Dense FFN tendency | MoE tendency | Evidence required |
|---|---|---|---|
| Simplicity and portability | Fewer routing/runtime contracts | More dispatch and placement complexity | Operational skill, framework support and failure drills |
| Total weight memory | Scales with one FFN per layer | Scales with all expert weights unless offloaded/quantized | Actual checkpoint and residency measurements |
| Per-token expert arithmetic | Executes shared FFN | Executes selected experts plus router | Matched-token profiler results |
| Communication | Often simpler under tensor/data parallelism | May require expert all-to-all | Network topology and communication profiling |
| Specialization capacity | Shared transformation | Multiple conditional transformations | Quality slices and expert-use analysis |
| Load sensitivity | Dense work more uniform | Router skew can overload a subset | Tail expert loads and overflow/reroute rates |
| Latency at low batch | Often fewer coordination stages | Dispatch overhead may dominate | Single-request and low-concurrency tests |
| High-throughput serving | Efficient dense kernels may perform well | Sparse compute may help if dispatch is efficient | Offered/accepted load, p50/p95/p99 latency and cost |

Select MoE only when measured benefit on the target objective justifies total storage, routing, communication, operational, and evaluation costs. Compare against a dense baseline with equivalent data, tokenizer, training budget definition and release criteria. Do not infer superiority from parameter count or active count alone.

## 9. Failure modes and diagnostic procedures

### Failure A: expert collapse

**Symptom:** a small fraction of experts receives most tokens; other experts are nearly idle.

**Competing causes:** router initialization; balancing loss too weak; data mixture skew; mask/dispatch bug; accidental expert parameter sharing.

**Discriminating check:** inspect assignment histograms by training step, domain, device and token position; verify expert weights are distinct; compare router logits and auxiliary-loss gradients.

**Bounded fix:** correct confirmed routing/parameter defect or run a controlled balancing-strength experiment. Do not declare success solely because counts become uniform.

**Verification:** repeat routing and quality checks on held-out data; measure expert utilization and task loss with a fixed seed/configuration.

### Failure B: capacity overflow or token dropping

**Symptom:** outputs differ as batch composition changes, or tokens are dropped/rerouted at high load.

**Competing causes:** capacity factor too low; skewed router; padding counted as real tokens; per-device capacity mismatch; second-choice routing behavior misunderstood.

**Discriminating check:** log token IDs before/after dispatch, per-expert counts, overflow decisions and masks; reproduce one overflow batch with a small simulator.

**Bounded fix:** correct accounting or select a justified capacity/routing policy; account for increased memory/compute if capacity rises.

**Verification:** demonstrate no silent loss under the accepted workload or quantify and evaluate the explicit overflow behavior.

### Failure C: routing output order corruption

**Symptom:** quality collapse despite reasonable load; output values appear associated with wrong token positions.

**Competing causes:** incorrect permutation inverse; duplicate top-k assignment mishandling; padding slots included; cross-rank scatter index error.

**Discriminating check:** use unique token IDs and expert functions that return recognizable constants; trace dispatch and combine indices end-to-end.

**Bounded fix:** repair indexing and add deterministic routing parity tests before performance optimization.

**Verification:** test all selected experts, top-k combinations, empty expert bins, overflow and cross-device transfers against a reference.

### Failure D: MoE slower than dense baseline

**Symptom:** nominally sparse model has worse throughput or tail latency.

**Competing causes:** all-to-all overhead; low batch; imbalanced expert work; sorting/dispatch kernels; capacity padding; poor expert placement; synchronization.

**Discriminating check:** profile router, dispatch, communication, expert compute, combine and queue time separately under production-like concurrency.

**Bounded fix:** address the measured dominant component or reject the architecture under this workload. Do not treat increased batch as a free fix if it violates latency or memory constraints.

**Verification:** compare end-to-end latency and cost against a dense baseline at matched quality and offered load.

### Failure E: active parameter claim misleads resource planning

**Symptom:** device memory exceeds the “active parameters” estimate or all experts cannot be loaded.

**Competing causes:** total experts must be resident; expert sharding or replication differs; quantization metadata omitted; optimizer/activation memory included in training.

**Discriminating check:** enumerate total expert tensors, placement, dtype, metadata and simultaneous live allocations.

**Bounded fix:** recompute storage from total parameters and plan placement, offload or quantization explicitly.

**Verification:** actual load and peak-memory test for the intended concurrency and checkpoint format.

## 10. Worked routing exercise

Assume a batch contains 128 real tokens, 8 experts, top-2 routing, equal expert FFNs with 2,097,152 matrix weights, and capacity factor 1.25 under the simplified per-expert formula.

1. Assignments: `128*2=256` token-expert assignments.
2. Balanced average: `256/8=32` assignments per expert.
3. Per-expert capacity: `ceil(1.25*32)=40` assignments.
4. Total nominal expert slots: `8*40=320`.
5. Total expert matrix parameters: `8*2,097,152=16,777,216`.
6. Selected expert matrix weights per token: `2*2,097,152=4,194,304`.

These values do not show actual router balance. For example, one expert could receive 60 assignments and another 10; the average remains 32, but the first expert exceeds capacity 40. Depending on implementation, some assignments are dropped, rerouted, or cause a different capacity policy. The observable consequence must be tested.

If the router selects an expert with probability `0.7` and another with `0.2`, a top-2 implementation may use raw weights summing to `0.9` or renormalize them to `0.7/0.9` and `0.2/0.9`, depending on the contract. The output differs. Parameter count and expert selection alone do not determine the combine semantics.

## 11. Interview and self-check questions

### Q1. What is sparse in sparse MoE?
**Answer:** The computation path per token activates only a subset of experts under the routing rule.
**Reasoning:** The total expert bank remains part of model capacity and storage.
**Follow-up:** What metric clarifies storage?
**Answer:** Total distinct stored parameters and bytes by placement/dtype.

### Q2. With eight experts and top-2 routing, what fraction of experts are selected per token?
**Answer:** Two of eight, or 25% by expert count.
**Reasoning:** `2/8=0.25`.
**Follow-up:** Does this mean 25% of dense-model FLOPs?
**Answer:** No; expert sizes, router, communication, baseline dense structure and execution overhead matter.

### Q3. For `d=512`, `d_ff=2048`, how many matrix weights in one two-layer expert?
**Answer:** `2*512*2048=2,097,152`.
**Reasoning:** Expansion and contraction matrices each have `512*2048` elements.
**Follow-up:** How many in eight equal experts?
**Answer:** `16,777,216`, excluding router and other components.

### Q4. For 128 tokens, top-2, eight experts, what is balanced average assignments?
**Answer:** `128*2/8=32` assignments per expert.
**Reasoning:** Total assignment count is 256.
**Follow-up:** Does average 32 guarantee no capacity overflow at 40?
**Answer:** No; load can be skewed across experts.

### Q5. What does a capacity factor change?
**Answer:** It changes the capacity allocation relative to a nominal expected load under a stated formula.
**Reasoning:** Larger capacity can reduce overflow but consumes memory/compute and does not eliminate arbitrary skew.
**Follow-up:** Is capacity formula universal?
**Answer:** No; verify per-device/global counting, padding and overflow semantics in the implementation.

### Q6. Why can a balanced router still be low quality?
**Answer:** Uniform utilization does not guarantee that tokens are assigned to experts that produce useful outputs.
**Reasoning:** A balancing objective trades off against task loss and expert specialization.
**Follow-up:** What should be evaluated together?
**Answer:** Load/overflow and task quality by relevant data slices.

### Q7. Why might MoE have worse latency than dense?
**Answer:** Router, token grouping, all-to-all, synchronization, padding, or low utilization can outweigh reduced active expert arithmetic.
**Reasoning:** End-to-end path includes dispatch and combine.
**Follow-up:** What is the right benchmark?
**Answer:** Matched end-to-end workload, concurrency, quality, offered load, latency distribution and cost.

### Q8. How do you test dispatch correctness?
**Answer:** Use unique token IDs and simple distinguishable expert functions, then verify routing, assignment, returned order and weighted combination against a reference.
**Reasoning:** End-to-end accuracy alone may not localize mapping errors.
**Follow-up:** What boundary cases matter?
**Answer:** Empty expert bins, top-k ties, capacity overflow, padding and distributed rank transfers.

### Q9. What is the difference between selected parameters and resident memory?
**Answer:** Selected parameters are those invoked on a token path; resident memory includes weights stored or loaded even when not selected for that token.
**Reasoning:** Conditional execution does not automatically unload other experts.
**Follow-up:** What changes the residency calculation?
**Answer:** Sharding, offload, caching, replication, quantization, and runtime load policy.

### Q10. What information is needed before comparing dense and MoE models?
**Answer:** Architecture, total/active parameters, data, token budget, training/inference compute, quality, hardware, topology, latency, memory, and cost.
**Reasoning:** A comparison with unequal resources or objectives is confounded.
**Follow-up:** What is a defensible comparator?
**Answer:** A dense baseline and MoE candidate evaluated under a clearly matched budget relevant to the decision.

## 12. Quick reference card

| Item | Rule |
|---|---|
| Routing | Router scores tokens; top-k assignment selects expert paths; combine contract must be explicit. |
| Total expert weights | `E * weights_per_expert`. |
| Selected expert weights/token | Approximately `k * weights_per_expert`; not total memory. |
| Assignment count | `T*k` before capacity handling. |
| Balanced average | `T*k/E`; does not bound maximum load. |
| Capacity | Implementation-specific; verify factor, counting scope, padding, drop/reroute policy. |
| Systems cost | Include dispatch, all-to-all, imbalance, padding, synchronization and expert weight residency. |
| Correctness oracle | Tiny token-ID reference trace through route, expert, combine, and return order. |
| Comparison | Match task/data/budget/workload; report total and active parameters separately. |
| Evidence | Illustrative calculations only; no measured model result. |

## Sources and boundaries

1. Shazeer et al., [“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”](https://arxiv.org/abs/1701.06538), 2017. Primary paper introducing a sparsely gated MoE layer and associated routing/capacity concerns in its setup.
2. Fedus, Zoph, and Shazeer, [“Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity”](https://arxiv.org/abs/2101.03961), 2021. Primary paper on top-1 routing, capacity and balancing in its experimental architecture; not evidence that all MoE systems use identical policies.
3. [Module 01 source coverage plan](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-transformers): includes the illustrative eight-expert, top-2 exercise and router-load validation target.
4. [`Train/Templates/directory.txt`](../../../Templates/directory.txt), lines 860–864: visible Module 01 source filenames only; original source bodies remain unavailable.

The lesson gives a generic routing model for reasoning. Before implementing a particular MoE, inspect the exact framework version, source revision, dispatch backend, capacity policy, expert parallelism and checkpoint format; then run correctness and workload-specific tests.
