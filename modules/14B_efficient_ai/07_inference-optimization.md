# Inference Optimization

> Module 14B — Efficient AI and On-Device Inference
> Topic 07 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 02, 05
> Version 2026.3

## 1. Why this topic exists

A trained model is a graph of mathematical operations. A deployed model is a program that executes those operations on a specific processor, with a specific memory layout, under a specific runtime. The gap between the two is where inference optimization lives.

The gap is often larger than the compression gains. A model can be quantized, pruned, and distilled, and still miss its deadline because three operators fall back to the CPU and force four partition boundaries. A model can be unchanged in arithmetic and twice as fast after fusion, because the intermediate tensors no longer travel to memory. A model can fit in flash and fail at runtime because the memory planner could not find a layout that fits the arena.

Three industry problems motivate this topic.

**The operator-support surprise.** A neural processing unit (NPU) supports convolution, depthwise convolution, and ReLU, but not Gaussian error linear unit (GELU), layer normalization, or non-maximum suppression. The unsupported operators run on the CPU, and each boundary between accelerators costs a copy and a synchronization. The latency penalty of one unsupported operator can exceed the benefit of the whole accelerator.

**The memory-traffic surprise.** Fusion does not reduce arithmetic; it reduces trips to memory. On a memory-bound device, that is the whole game. A convolution followed by batch normalization followed by ReLU writes two intermediate tensors that nobody needs; fusing them removes both writes and both reads. The arithmetic is identical, and the latency is lower.

**The graph-transform surprise.** The model that the training framework exports is not the model the runtime executes. Constant folding, layout conversion, operator fusion, and quantization all rewrite the graph. A model that is correct in PyTorch can be numerically different after export, and a model that is fast in the interpreter can be slow after conversion because a preferred fusion did not apply.

Inference optimization is the discipline of managing these gaps. It is less glamorous than architecture design and often more valuable: a 20% latency reduction from fusion costs less than a 20% reduction from a new model, and it does not require new data or a new clinical validation of accuracy, though it does require a change assessment.

### 1.1 What each optimization saves

| Optimization | Arithmetic | Memory traffic | Launch overhead | Typical gain |
|---|---|---|---|---|
| Constant folding | Removes work | Removes reads | Removes launches | Small to moderate |
| Dead node elimination | Removes work | Removes reads and writes | Removes launches | Small |
| Conv-BN fusion | Same | Removes one pass | Removes one launch | Small to moderate |
| Conv-BN-ReLU fusion | Same | Removes two passes | Removes two launches | Moderate |
| Layout conversion | Same | Adds a transpose if mixed | Adds a launch | Negative unless it enables a fast kernel |
| Memory planning | Same | Same | Same | None; reduces peak memory |
| KV cache | Removes recomputation | Adds cache reads | Same | Large for decode |
| Speculative decoding | Changes the schedule | Adds draft reads | Same | Workload-dependent |

The table is a reminder that "optimization" is not one thing. A pass that reduces memory traffic does nothing for a compute-bound model, and a pass that reduces launches does nothing for a model with few nodes.

### 1.2 The two failure signatures

Inference optimization fails in two recognizable ways.

- **No change.** The pass was not applied, or the runtime chose a different kernel, or the model is bound by something the pass does not touch. Inspect the exported graph before tuning.
- **Worse.** The pass added a transpose, forced a fallback, changed the layout for a kernel that did not benefit, or increased peak memory beyond the arena. Revert and bisect the passes.

Both signatures are visible in the graph and in the per-node timing. Neither is visible in the model's accuracy, which is why an accuracy-only validation misses them.

### 1.3 What this topic assumes

- The model has been trained and, optionally, compressed. This topic does not choose the compression method.
- The target runtime and board are known. A runtime-agnostic optimization guide would be a list of maybes.
- The clinical latency budget exists and includes non-model terms. Topic 01 explains why.

If any assumption fails, return to the topic that establishes it before optimizing.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** graph optimizations into semantics-preserving, arithmetic-changing, and layout-changing categories.
2. **Compute** the memory traffic saved by fusing a chain of operators.
3. **Estimate** the latency cost of an operator fallback, including partition boundaries.
4. **Choose** a layout and explain its effect on kernel selection.
5. **Plan** memory for a fixed arena and identify the peak live set.
6. **Explain** key-value (KV) cache growth and the options for reducing it in on-device language models.
7. **Validate** that an optimized graph produces the same clinical decisions as the reference.
8. **Document** an optimization pass so that the deployed graph is reproducible.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A classification table with examples |
| 2 | Exercise 2 | Byte arithmetic for a fused chain |
| 3 | Exercise 3 | A fallback cost estimate |
| 4 | Exercise 4 | A layout choice with a kernel rationale |
| 5 | Exercise 5 | A peak live-set calculation |
| 6 | Exercise 6 | KV cache bytes and options |
| 7 | Exercise 7 | A numerical equivalence check |
| 8 | Exercise 8 | A reproducible optimization record |

## 3. The core mental model

An inference graph is a sequence of operations connected by tensors. Optimization changes either the operations, the tensors, or the mapping of operations to processors.

```text
   input
     │
     ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ conv    │───►│ batch   │───►│ relu    │───► ...
   │         │    │ norm    │    │         │
   └─────────┘    └─────────┘    └─────────┘
        │              │              │
        ▼              ▼              ▼
     tensor A       tensor B       tensor C
     (write+read)   (write+read)   (write+read)

   fused:
   ┌──────────────────────────────┐
   │ conv + batch norm + relu     │───► tensor C
   └──────────────────────────────┘
        one kernel, one write, one read
```

The fused form performs the same arithmetic and moves fewer bytes. On a memory-bound device, the reduction in bytes is the reduction in time. On a compute-bound device, the gain is smaller, because the arithmetic dominates.

### 3.1 Three categories of optimization

| Category | What changes | Examples | Risk |
|---|---|---|---|
| Semantics-preserving | Neither arithmetic nor layout | Constant folding, dead node elimination, identity removal | Low |
| Arithmetic-changing | Rounding, fusion order, approximation | Conv-BN fusion, GELU approximation, fast math | Numerical difference |
| Layout-changing | Tensor order and stride | NCHW to NHWC, tiling, packing | Wrong results if a kernel expects another layout |

The categories matter because they carry different validation burdens. A semantics-preserving pass needs a diff test, not a clinical re-evaluation. An arithmetic-changing pass needs a numerical comparison and, if the difference is material, a clinical evaluation. A layout-changing pass needs a correctness test on the target, because a layout mismatch can produce wrong results rather than an error.

### 3.2 The cost of a graph boundary

When an operator runs on a different processor from its neighbours, the runtime inserts a boundary: a copy of the tensor, a synchronization, and a dispatch. The cost is roughly:

```text
boundary_cost = copy_bytes / bandwidth + sync_overhead + dispatch_overhead
```

On a microcontroller, the copy may be a few microseconds and the dispatch a few more. On an applications processor with an NPU, the copy may go through shared memory and the synchronization may cost tens of microseconds. The boundary cost is often larger than the arithmetic cost of the operator that caused it.

This is why operator support is a first-order design concern, not a post-hoc detail. A model with three unsupported operators can have five partitions and four boundaries, and the boundaries can double the latency even if each unsupported operator is cheap.

### 3.3 Memory planning

A runtime must assign a memory address to every tensor. The peak memory is the maximum over time of the sum of live tensors. Two tensors are live simultaneously if one is produced before the other is consumed.

```text
   time ──────────────────────────────────────►
   A  ████████
   B      ████████
   C           ████████
   D                ████████
   peak live = 2 tensors (A+B, B+C, C+D)
```

Fusion reduces the number of live tensors by removing intermediates. Memory planning reduces the arena by reusing buffers for tensors whose lifetimes do not overlap. TensorFlow Lite for Microcontrollers documents a greedy planner that does exactly this, with a single arena split into a head for shared tensors, a temporary region, and a tail for persistent allocations.

The practical consequence is that peak memory is not the sum of all tensors and not the size of the model file. It is the high-water mark of the live set, plus the runtime's own allocations and alignment padding.

### 3.4 Layout

A tensor's logical shape is not its memory layout. NCHW and NHWC store the same values in different orders, and a kernel written for one is slow or incorrect on the other.

| Layout | Order | Common on | Consequence |
|---|---|---|---|
| NCHW | batch, channel, height, width | PyTorch, some GPUs | Channel stride is large |
| NHWC | batch, height, width, channel | TensorFlow, mobile CPUs, many NPUs | Channel stride is 1 |
| Blocked | channel blocks interleaved | Vendor accelerators | Requires a packing step |

A layout conversion is a transpose, which costs a full read and write of the tensor. A runtime that converts layout at every boundary can spend more time transposing than computing. The fix is to keep one layout through the graph, or to let the compiler insert conversions only where the cost is amortized.

### 3.5 The language-model case

A decoder-only language model has two phases with different bottlenecks.

- **Prefill:** the model processes the prompt in parallel. The bottleneck is compute, because the matrix multiplications are large.
- **Decode:** the model generates one token at a time. The bottleneck is memory bandwidth, because each token requires reading the weights and the KV cache.

The KV cache stores the keys and values of previous tokens so that they are not recomputed. Its size grows linearly with context:

```text
KV bytes = 2 × layers × heads × head_dim × sequence_length × bytes_per_value
```

For 12 layers, 12 heads, head dimension 64, sequence 512, and FP16:

```text
2 × 12 × 12 × 64 × 512 × 2 = 18,874,368 B = 18.0 MiB
```

That is larger than many on-device models. Quantizing the cache to INT8 halves it to 9.0 MiB, at an accuracy cost that must be evaluated separately from weight quantization.

PagedAttention, described by Kwon et al. (SOSP 2023), manages the cache in fixed-size blocks with a block table, reducing fragmentation and enabling sharing. The paper reports that only 20.4–38.2% of KV cache memory held actual token states in prior systems, and that paged allocation achieves about 96.3% effective usage with a default block size of 16 tokens. Those are the authors' measurements on their serving workloads, not on a medical device.

### 3.6 The optimization pipeline

A typical export pipeline applies passes in a fixed order, and the order matters.

```text
   trained graph
        │
        ▼
   constant folding ──► smaller graph, fewer constants
        │
        ▼
   dead code elimination ──► smaller graph
        │
        ▼
   operator fusion ──► fewer launches, less traffic
        │
        ▼
   layout assignment ──► kernel-friendly tensors
        │
        ▼
   quantization ──► lower precision, new scales
        │
        ▼
   partitioning ──► backend assignment, boundaries
        │
        ▼
   memory planning ──► arena layout
```

Two ordering rules matter.

- **Fusion before quantization.** Fusion changes the weight distribution and the accumulation; quantization must see the fused graph. Quantizing first and fusing second invalidates the scales.
- **Quantization before partitioning.** The partitioner must know which operators have quantized kernels. Partitioning first can assign a node to a backend that cannot execute its quantized form.

A pipeline that reverses either rule produces a graph that loads and runs but is slower or less accurate than expected.

### 3.7 What optimization cannot fix

- **A model that is too large for the memory.** Fusion reduces intermediates, not weights.
- **A clinical accuracy shortfall.** Graph optimization does not add information.
- **An unsupported operator with no equivalent.** The fallback remains.
- **A sensor pipeline that is slower than the model.** The bottleneck is outside the graph.
- **A thermal design that cannot sustain the load.** The clock will drop regardless of the graph.

The list is useful in design reviews, where optimization is sometimes proposed as a remedy for problems it cannot touch.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Graph optimization | Rewriting a computation graph to improve execution | Confused with quantization | Transform |
| Constant folding | Evaluating constant subgraphs at compile time | Confused with fusion | Transform |
| Dead node elimination | Removing nodes whose outputs are unused | Confused with pruning | Transform |
| Identity removal | Removing no-op nodes | Assumed to change semantics | Transform |
| Operator fusion | Combining nodes into one kernel | Assumed to reduce arithmetic | Transform |
| Conv-BN fusion | Folding batch normalization into convolution weights | Assumed exact in quantized arithmetic | Transform |
| Activation fusion | Combining an activation with the preceding operator | Confused with approximation | Transform |
| Layout optimization | Converting tensor order for kernel efficiency | Assumed free | Transform |
| NCHW | Batch, channel, height, width layout | Confused with NHWC | Layout |
| NHWC | Batch, height, width, channel layout | Confused with NCHW | Layout |
| Tiling | Splitting a tensor into tiles for cache residency | Confused with blocking | Transform |
| Memory planning | Assigning buffers to tensors to minimize peak memory | Confused with model size | Transform |
| Arena | A single preallocated memory region for tensors | Confused with heap | Buffer |
| Peak live set | Maximum simultaneously live tensors | Confused with total tensor count | Bytes |
| Kernel autotuning | Selecting the fastest kernel for a shape | Assumed portable | Process |
| Execution provider | A runtime backend for a device | Confused with a device | Component |
| Delegate | A backend that executes a subgraph | Confused with an operator | Component |
| Partition | A subgraph assigned to one processor | Confused with a graph cut | Subgraph |
| Boundary | A copy and synchronization between partitions | Assumed free | Cost |
| Operator fallback | Running an unsupported operator on another processor | Assumed cheap | Mechanism |
| Operator support list | The set of operators a backend implements | Assumed complete | Document |
| Graph partitioner | The component that assigns nodes to backends | Confused with the scheduler | Component |
| Quantization | Reducing numeric precision | Confused with graph optimization | Transform |
| Requantization | Converting accumulators to output precision | Ignored in fusion analysis | Step |
| KV cache | Stored keys and values for autoregressive decoding | Confused with model weights | Buffer |
| PagedAttention | Block-based KV cache management | Confused with attention itself | Method |
| Block table | Mapping from logical to physical KV blocks | Confused with a page table in an OS | Structure |
| Prefill | Processing the prompt in parallel | Confused with training | Phase |
| Decode | Generating one token at a time | Confused with prefill | Phase |
| Speculative decoding | A draft model proposes tokens, a larger model verifies | Assumed always faster | Method |
| Continuous batching | Adding requests to a running batch | Confused with static batching | Method |
| Prompt caching | Reusing computed prefix states | Confused with KV cache | Method |
| Throughput | Tokens or inferences per second | Confused with latency | Rate |
| Time to first token | Latency until the first output token | Confused with total latency | ms |
| Inter-token latency | Time between output tokens | Confused with time to first token | ms |
| Soak test | Sustained load test | Confused with a benchmark | Test |
| Numerical equivalence | Same outputs within a stated tolerance | Assumed exact | Property |
| Determinism | Same output on every run | Assumed for all kernels | Property |
| Fallback path | A simpler rule or model when the main path fails | Confused with operator fallback | Architecture |
| Graph capture | Recording a graph for replay | Confused with tracing | Process |

Two terms are frequently confused.

**Operator fallback versus fallback path.** Operator fallback is a runtime decision to execute an unsupported node on another processor. A fallback path is a product-level design in which a simpler rule or model takes over when the main model is unavailable. Both are called "fallback", and they have nothing in common.

**Fusion versus approximation.** Fusion combines operations without changing the mathematical result in exact arithmetic. Approximation changes the result, for example by using a faster GELU. A fused GELU may be both, and the validation must distinguish which.

### 4.1 Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| L | Latency | ms |
| T_op | Time for one operator | ms |
| N_nodes | Number of graph nodes | count |
| N_boundaries | Number of partition boundaries | count |
| B | Bytes copied at a boundary | byte |
| BW | Memory bandwidth | byte/s |
| C_sync | Fixed synchronization cost | ms |
| S_kv | KV cache size | byte |
| L_layers | Number of layers | count |
| H | Number of heads | count |
| d_head | Head dimension | count |
| n_ctx | Context length | tokens |

The two formulas used most often:

```text
L_fused = Σ T_op - Σ (T_fused_group - max(T_members)) + N_launches × T_launch
L_partitioned = L_fused + Σ (T_cpu - T_accel) + N_boundaries × (B/BW + C_sync)
```

## 5. The mechanics, step by step

The procedure produces an optimized graph with evidence. It assumes a trained, possibly compressed model and a target runtime.

### Step 1. Capture the reference graph and its outputs

**Input:** The trained model and a fixed set of reference inputs.

**Output:** A reference graph, a set of reference outputs, and a baseline latency.

**Rationale:** Every optimization must be compared with a reference. Without one, a numerical difference cannot be attributed and a latency gain cannot be measured.

The reference set should include:

- Typical inputs from the deployment distribution.
- Edge cases: the smallest and largest shapes, the noisiest signal, the rarest class.
- Inputs that exercise every branch in the graph.
- A fixed seed for any stochastic component, or a deterministic mode.

**Branch point:** If the model has dynamic shapes, capture a reference for each shape the device will see. A single-shape reference does not validate the others.

**Edge case:** A model with dropout or random augmentation in the export path is not deterministic. Remove those layers before capture, or fix the seed and record it.

### Step 2. Apply semantics-preserving optimizations

**Input:** The reference graph.

**Output:** A graph with constants folded, dead nodes removed, and identities eliminated.

**Rationale:** These passes are safe and free. They reduce graph size and sometimes remove whole subgraphs that the training framework left behind.

The passes:

- **Constant folding:** evaluate subgraphs whose inputs are all constants.
- **Dead node elimination:** remove nodes whose outputs do not reach the graph outputs.
- **Identity removal:** remove no-op reshapes, casts, and padding where the runtime can prove they are no-ops.
- **Common subexpression elimination:** compute a repeated expression once.

**Branch point:** If a pass changes the graph's outputs beyond floating-point tolerance, it is not semantics-preserving for this model. Treat it as arithmetic-changing and validate accordingly.

**Edge case:** Constant folding can increase model size if it materializes a large constant that was previously computed at runtime. Check the file size after folding.

### Step 3. Fuse operators

**Input:** The graph and the runtime's supported fusion patterns.

**Output:** A graph with fused nodes and a list of the fusions applied.

**Rationale:** Fusion reduces memory traffic and launch overhead. It does not reduce arithmetic, except when the fused kernel uses a faster implementation.

The common patterns:

| Pattern | Fused into | Saves | Risk |
|---|---|---|---|
| Conv + BN | Conv with folded weights | Two passes over the tensor | Numerical difference |
| Conv + BN + ReLU | One kernel | Two passes and two launches | Numerical difference |
| MatMul + Add | GEMM with beta | One pass | Low |
| MatMul + Add + GELU | Fused GEMM epilogue | One pass | Approximation if GELU is approximated |
| LayerNorm + residual | Fused normalization | One pass | Low |
| Attention (QKV + softmax + AV) | Fused attention | Several passes | Layout and mask complexity |

**Branch point:** If the runtime does not support a fusion, the graph keeps the unfused form. Forcing a fusion by rewriting the graph can produce a node the runtime does not recognize, which is worse than the unfused version.

**Edge case:** Conv-BN fusion changes the weight range. If the model is quantized, the scales must be recomputed after fusion, not before.

### Step 4. Choose the layout

**Input:** The graph, the runtime's preferred layout, and the target's kernel support.

**Output:** A layout assignment and a list of conversions.

**Rationale:** A layout mismatch forces transposes, which cost a full read and write of the tensor. A consistent layout avoids them.

The decision:

| Target | Preferred layout | Reason |
|---|---|---|
| Mobile CPU with SIMD | NHWC | Channel-contiguous access |
| Many NPUs | Vendor-specific blocked layout | Tensor core tiling |
| Desktop GPU | NCHW | Library convention |
| Microcontroller | NHWC or channel-last | CMSIS-NN convention |

**Branch point:** If the graph mixes operators with different preferred layouts, let the compiler insert conversions at the cheapest points, or split the graph into layout-homogeneous regions.

**Edge case:** A conversion inserted inside a loop or a fused region can negate the fusion benefit. Check the final graph for conversions.

### Step 5. Plan memory

**Input:** The optimized graph and the runtime's allocator.

**Output:** A memory plan with peak usage and buffer assignments.

**Rationale:** Peak memory is a hard constraint on a microcontroller. A plan that exceeds the arena fails at initialization, not at inference, and the failure is better found early.

The procedure:

1. Compute the lifetime of every tensor.
2. Compute the peak live set.
3. Reuse buffers whose lifetimes do not overlap.
4. Add alignment padding and runtime overhead.
5. Compare with the arena size.
6. If it does not fit, reduce the input window, fuse more, or tile.

**Branch point:** If the peak live set is close to the arena, reserve headroom for the runtime's temporary allocations. A plan that fits exactly will fail when the runtime allocates a scratch buffer.

**Edge case:** A tensor that is used by an operator on another processor may need a separate buffer for the copy, which increases the peak.

### Step 6. Partition the graph for accelerators

**Input:** The graph and the backend support lists.

**Output:** A partition assignment and a list of boundaries.

**Rationale:** A partition determines which processor executes each node. The boundaries between partitions are the cost of heterogeneity.

The procedure:

1. List every operator and its supported backends.
2. Assign each node to the fastest backend that supports it.
3. Merge adjacent nodes on the same backend.
4. Count the boundaries.
5. Estimate the boundary cost.
6. If the cost exceeds the benefit, change the assignment or replace the operator.

**Branch point:** If a single unsupported operator splits a large supported region into two, it may be cheaper to run the whole region on the slower backend than to pay two boundaries.

**Edge case:** A backend may support an operator only for certain shapes or precisions. The support list is necessary but not sufficient; the specific configuration must be tested.

### Step 7. Handle dynamic shapes and control flow

**Input:** The graph and the range of shapes the device will see.

**Output:** A plan for shape specialization or dynamic execution.

**Rationale:** Many accelerators require static shapes. A dynamic shape can force a fallback or a recompilation, and the cost is paid at runtime.

Options:

- **Specialize:** compile one graph per shape.
- **Pad:** pad inputs to a fixed shape.
- **Dynamic:** use a runtime that supports dynamic shapes, usually at a performance cost.
- **Bucket:** group similar shapes and compile for the bucket maximum.

**Branch point:** If the shape range is wide, specialization can produce many binaries and a large flash footprint. Bucketing trades some performance for size.

**Edge case:** A model with data-dependent control flow may not be compilable at all. Check the runtime's support for the control-flow operators before designing around them.

### Step 8. Optimize the language-model path if present

**Input:** A decoder-only model and its deployment constraints.

**Output:** A plan for KV cache, batching, and decoding.

**Rationale:** Language models have different bottlenecks from convolutional models, and the optimizations are different.

The options:

- **KV cache quantization:** halve the cache with INT8, at an accuracy cost that must be evaluated.
- **PagedAttention:** reduce fragmentation and enable sharing.
- **Speculative decoding:** a small draft model proposes tokens and the large model verifies them. The speedup depends on the acceptance rate.
- **Continuous batching:** improve throughput on a server, not latency on a single device.
- **Prompt caching:** reuse prefix states, with attention to privacy and staleness.

**Branch point:** If the device serves one user at a time, throughput optimizations do not help latency and can hurt it by adding queueing.

**Edge case:** A quantized KV cache can change the model's outputs more than weight quantization, because the cache accumulates over the sequence. Evaluate long-context behavior, not only the first few tokens.

### Step 9. Validate numerical equivalence and latency

**Input:** The optimized graph, the reference outputs, and the target.

**Output:** A comparison table and a target measurement.

**Rationale:** An optimized graph is a different program. It must produce the same decisions within a stated tolerance and meet the latency budget.

The validation includes:

- Output differences on the reference set: maximum absolute difference, relative difference, and agreement of the final decision.
- Accuracy, calibration, and subgroup results on the held-out set.
- Latency at p50, p95, and p99 on the target.
- Peak memory against the arena.
- Energy per inference.
- Determinism across runs.

**Branch point:** If the decision agreement is below 100% on the reference set, inspect the cases where it differs. A single differing case can be a genuine numerical sensitivity or a bug.

**Edge case:** A graph can be numerically equivalent on average and different on the tail, which is where the clinical decisions are. Include the tail cases in the reference set.

### Step 10. Record the pass and classify the change

**Input:** All artifacts.

**Output:** An optimization record and a change classification.

**Rationale:** The deployed graph is the result of a sequence of transformations. Without a record, it cannot be reproduced or defended.

The record includes:

- The reference graph and its hash.
- The optimization passes applied, in order, with their parameters.
- The runtime and compiler versions.
- The fusion and partition lists.
- The layout assignment.
- The memory plan and peak usage.
- The numerical comparison and the target measurements.
- The change classification.

**Branch point:** If the change alters clinical outputs beyond the stated tolerance, it is not an optimization; it is a model change and must follow the change-control pathway.

**Edge case:** A runtime upgrade that changes fusion or kernel selection is a change even if the model file is identical. Record the runtime version and re-validate after upgrades.

### 5.1 Fusion patterns and when they fail

Fusion is not automatic. A runtime fuses only patterns it recognizes, and a pattern that is *almost* right is not fused at all.

| Pattern | Fused by | Fails when |
|---|---|---|
| Conv + BN | Most runtimes | BN has a non-constant statistic, or training mode |
| Conv + BN + ReLU | Most runtimes | Activation is not exactly ReLU, or layout differs |
| MatMul + Add | Most runtimes | Add has a non-broadcastable shape |
| MatMul + Add + GELU | Some runtimes | GELU is approximated, or the epilogue is unsupported |
| LayerNorm + residual | Some runtimes | The residual has a different layout |
| Attention | Some runtimes | Mask type or head layout is unsupported |

When a fusion fails, the graph keeps the unfused form, and the cost is the extra passes and launches. The failure is silent: the model runs, produces correct outputs, and is slower. Inspecting the exported graph is the only way to know.

### 5.2 A partition cost model

A partition's cost has four terms:

```text
cost = operator_time_on_backend
     + copy_bytes / bandwidth
     + sync_overhead
     + lost_fusion_cost
```

The last term is the one teams forget. A boundary prevents fusion across it, so the nodes on either side cannot share a kernel even if they would otherwise be fused.

A simple estimate:

```text
boundary_cost ≈ copy_bytes / bandwidth + fixed_sync
```

For an 8 KiB tensor over a 1 GB/s link, the copy is 8 µs; a fixed synchronization of 0.5–1 ms dominates on many mobile systems. That is why small tensors at boundaries are disproportionately expensive.

### 5.3 Optimization checklist

- The reference graph and reference outputs are stored.
- The passes are applied in the documented order.
- The exported graph is inspected, not only the training graph.
- The partition list is known, with the backend for each node.
- The boundary count and cost are estimated.
- The layout is consistent, or the conversions are counted.
- The peak live set fits the arena with headroom.
- Numerical equivalence is checked on tails and borderline cases.
- Latency is measured on the target at p50, p95, and p99.
- The runtime and toolchain versions are recorded.
- The change is classified.

### 5.4 Determinism and reproducibility

Two runs of the same optimized graph should produce the same outputs. In practice, three things can break this.

- **Parallel reductions.** Floating-point addition is not associative, and a kernel that reduces in a different order produces different last bits.
- **Nondeterministic kernel selection.** An autotuner may choose different kernels on different runs, especially under varying thermal conditions.
- **Asynchronous execution.** A runtime that overlaps operations may change the timing and, in rare cases, the order of accumulation.

For a regulated device, determinism is a requirement for the evidence, not necessarily for every run. The practical approach is to fix the kernel selection, disable autotuning in the validated build, and record the runtime configuration.

### 5.5 Decision table for optimization outcomes

| Observation | Likely cause | Next action |
|---|---|---|
| Latency improves, accuracy unchanged | Fusion or layout | Accept; record the pass |
| Latency improves, accuracy shifts | Arithmetic-changing fusion | Quantify; validate clinically |
| No latency change after fusion | Compute-bound or fusion not applied | Inspect the graph |
| Latency worse after conversion | Layout transposes or fallback | Inspect conversions and partitions |
| Peak memory higher than expected | Live set or copies | Recompute lifetimes; check boundaries |
| Model fails to load | Unsupported operator or shape | Check support list and shapes |
| Outputs differ on a few inputs | Numerical sensitivity | Inspect the cases; tighten tolerance |
| Outputs differ widely | Layout or fusion bug | Revert the pass; bisect |
| p99 much worse than p50 | Fallback or scheduling | Trace per-inference partitions |
| Energy worse despite lower latency | Different kernel or clock | Measure power, not only time |

### 5.6 Edge cases in inference optimization

**Outlier channels.** A graph optimization does not change the weights, but fusion can change the accumulation order and the effective rounding. A layer that was just above the accuracy floor can fall below it after fusion. Re-evaluate the quantized model after fusion.

**Per-tensor versus per-channel scales.** Layout conversion can change the axis along which per-channel scales are applied. A scale vector attached to the wrong axis produces wrong results, often without an error. Verify the axis after conversion.

**Activation clipping.** Fusion can remove an intermediate tensor that was used to calibrate activation ranges. If the range was computed on the unfused tensor, the fused kernel may see a different distribution. Recalibrate after fusion.

**Operator fallback.** A fallback is not just a slower operator; it is a boundary. The cost is the copy plus the synchronization plus the loss of fusion across the boundary. Estimate the boundary cost, not only the operator cost.

**Unsupported operators.** A runtime may support an operator in float but not in INT8, or at one shape but not another. The support list must be read at the precision and shape level.

**Zero-point drift.** Fusion changes the accumulation and the requantization. A zero point that was correct before fusion may produce a small offset after it. Compare the outputs on a fixed input.

**Calibration-set bias.** Calibration after fusion must use data that represents the deployment population. A fused graph that is calibrated on one site inherits the same bias as an unfused one.

**Thermal throttling.** Fusion can increase the instantaneous power draw by keeping more units busy. A faster kernel can trigger throttling sooner. Measure sustained behavior.

## 6. Worked example

This is a constructed case. The graph, timings, and accuracy values are illustrative and chosen so that the arithmetic closes.

### 6.1 The graph and the baseline

The ECG classifier from earlier topics is exported as a 12-node graph: three convolution blocks (conv, batch norm, ReLU), a pooling node, a global average pool, and a fully connected layer. On the Cortex-M4F at 64 MHz, the measured latency is 45.7 ms at p99, and the model consumes 1.124 mJ per inference at 3.0 V and 8.2 mA.

The per-node times are:

| Node | Time (ms) | Backend |
|---|---:|---|
| conv1 | 8.0 | NPU |
| bn1 | 0.5 | NPU |
| relu1 | 0.3 | NPU |
| conv2 | 14.0 | NPU |
| bn2 | 0.5 | NPU |
| relu2 | 0.3 | NPU |
| pool | 2.0 | NPU |
| conv3 | 16.0 | NPU |
| bn3 | 0.5 | NPU |
| relu3 | 0.3 | NPU |
| gap | 0.8 | NPU |
| fc | 2.5 | NPU |
| **Total** | **45.7** | |

The dispatch overhead is included in these times. An unfused graph pays 12 launches; a fused graph pays 6.

### 6.2 Fusion arithmetic

Fusing conv + batch norm + ReLU into one node saves the batch-norm pass and the ReLU pass, but not the convolution. The fused time is approximately the convolution time plus a small epilogue:

```text
conv1 + bn1 + relu1 = 8.0 + 0.5 + 0.3 = 8.8 ms unfused
fused               = 8.4 ms
saving              = 0.4 ms

conv2 block: 14.8 unfused -> 14.4 fused, saving 0.4 ms
conv3 block: 16.8 unfused -> 16.4 fused, saving 0.4 ms
```

Total arithmetic saving: 1.2 ms. Launch saving: 6 fewer launches at 0.05 ms each = 0.3 ms. Total fusion saving: 1.5 ms, which brings the latency from 45.7 to 44.2 ms.

The memory saving is larger than the time saving. Each unfused block writes two intermediate tensors of 8 KiB each. Fusing removes 6 intermediates, or 48 KiB of writes and the corresponding reads. On a memory-bound device, that would matter more; on this compute-bound model, it is a secondary benefit.

### 6.3 Memory before and after

```text
Unfused: 12 tensors × 8 KiB = 96 KiB of intermediates
Fused:    6 tensors × 8 KiB = 48 KiB of intermediates
Saving:   48 KiB
```

With four partition boundaries, each requiring a host copy of an 8 KiB tensor:

```text
Peak with boundaries = 48 KiB + 4 × 8 KiB = 80 KiB
```

Against a 256 KiB SRAM budget, all three configurations fit. On a 64 KiB device, only the fused configuration without boundaries would fit. Memory planning is therefore a feasibility question before it is a performance question.

### 6.4 Operator fallback

Three operators are unsupported by the NPU backend: GELU, layer normalization, and softmax. The runtime partitions the graph into five regions and inserts four boundaries.

```text
CPU time per unsupported operator = 0.80 ms
NPU time for the same operator   = 0.05 ms
Extra compute = 3 × (0.80 - 0.05) = 2.25 ms
Boundary cost = 4 × 0.80 = 3.20 ms
Total fallback cost = 5.45 ms

Latency = 44.2 + 5.45 = 49.65 ms
```

The fallback adds 5.45 ms, or 12.3% of the fused latency, and pushes the model over the 40 ms budget. The unsupported operators' arithmetic is only 2.25 ms of that; the boundaries are 3.20 ms, or 59% of the cost.

This is the characteristic signature of a fallback problem: the penalty is dominated by the boundaries, not by the operators.

### 6.5 Latency against the budget

| Configuration | Latency (ms) | Against 40 ms |
|---|---:|---|
| Unfused, all NPU | 45.7 | +5.7 |
| Fused, all NPU | 44.2 | +4.2 |
| Fused, 3 fallbacks | 49.65 | +9.65 |
| Fused, operators replaced | 44.2 | +4.2 |
| Fused, operators replaced, more fusion | 43.8 | +3.8 |

Replacing the three unsupported operators with supported equivalents removes the fallback and recovers 5.45 ms, but the model still misses the 40 ms budget by 4.2 ms. The remaining gap must be closed by reducing arithmetic: distillation or structured pruning, as in Topics 03 and 06.

### 6.6 Energy

```text
E_base     = 3.0 × 0.0082 × 0.0457 = 1.124 mJ
E_fused    = 3.0 × 0.0082 × 0.0442 = 1.087 mJ
E_fallback = 3.0 × 0.0082 × 0.04965 = 1.221 mJ
```

The fallback costs 0.134 mJ per inference relative to the fused graph, or 12.3%, matching the latency ratio because the board current is assumed constant.

### 6.7 Accuracy and calibration

| Configuration | Sensitivity | ECE | Slope | Decision agreement |
|---|---:|---:|---:|---:|
| Reference (float) | 0.942 | 0.021 | 0.97 | 100% |
| Fused, INT8 | 0.938 | 0.029 | 0.93 | 99.8% |
| Fused, INT8, fallbacks | 0.938 | 0.029 | 0.93 | 99.8% |
| Fused, INT8, operators replaced | 0.937 | 0.030 | 0.92 | 99.7% |

The decision agreement is the fraction of held-out inputs on which the optimized graph produces the same class as the reference at the operating threshold. A 0.2–0.3% disagreement is small in aggregate and must be inspected case by case: if the disagreements concentrate in the positive class, the sensitivity changes more than the agreement suggests.

The operator replacement changes the model slightly, which is why the sensitivity falls by 0.1 point relative to the fused version with fallbacks. The replacement is an arithmetic change, not a pure optimization.

### 6.8 Diagnostic one: outlier-driven scale collapse

The fully connected layer's 32,768 weights have a minimum of -1.029, a maximum of 1.2912, and a median absolute value of 0.0610. The symmetric per-tensor scale is 0.01016693, which is 16.7% of the median.

```text
S = 1.2912 / 127 = 0.01016693
step / median = 0.1667
weights rounded to zero = 1,411 of 32,768 (4.3%)
distinct levels used = 102 of 255
SQNR = 30.00 dB
```

The aggregate SQNR looks acceptable and the zeroed fraction is small, but the level count shows that most of the integer range is unused. The remedy is per-channel quantization, which raises SQNR to 44.67 dB at a metadata cost of 2,560 bytes.

### 6.9 Diagnostic two: per-channel skew

The 512 channels have ranges from 0.2756 to 2.3202, a ratio of 8.4. Under a per-tensor scale, a typical channel receives about 28 levels. Per-channel scales give every channel the full 255 levels.

The skew is a property of the weights, not of the graph optimization. Fusion does not change it, and layout conversion does not change it. The diagnostic is the channel range ratio; above about 4, per-tensor quantization costs material accuracy.

### 6.10 Varying one assumption

**Remove the fallbacks by replacing the operators.** Latency falls from 49.65 to 44.2 ms, a 5.45 ms recovery. The replacement is an arithmetic change: GELU becomes a supported approximation, layer normalization becomes a supported variant, and softmax becomes a sigmoid. The sensitivity falls by 0.1 point, which is a model change, not an optimization.

**Fuse more aggressively.** Fusing the pooling node into the preceding convolution and the global average pool into the last convolution saves another 0.4 ms, bringing the latency to 43.8 ms. The saving is small because the fused nodes are already cheap; the exercise shows diminishing returns.

**Batch size 1 to 8.** Throughput rises by roughly 6×, but p99 latency for the first item in the batch rises because the batch must be filled before the kernel runs. On a single-patient device, batching is not a latency optimization.

**KV cache quantization for a language model.** For a model with 18.0 MiB of FP16 KV cache, INT8 quantization reduces it to 9.0 MiB. The memory saving is real; the accuracy cost must be evaluated on long contexts, because the cache accumulates error over the sequence.

**Runtime upgrade.** A new runtime version changes kernel selection and fusion. The same model file can be 10% faster or slower. Re-measure after every upgrade; do not assume the previous result carries over.

### 6.11 The decision

The fused graph with fallbacks runs at 49.65 ms and fails the 40 ms budget by 9.65 ms. Replacing the three unsupported operators recovers 5.45 ms and brings the graph to 44.2 ms, still 4.2 ms over. Fusing the pooling nodes recovers another 0.4 ms, leaving 3.8 ms.

The remaining gap is arithmetic, not graph structure. The options are distillation, structured pruning, or a faster clock. The graph optimization has done what it can: it removed 1.5 ms of fusion overhead and 5.45 ms of fallback, and it has identified the residual as a model-size problem.

This is the normal outcome of an optimization pass: it rarely closes a gap on its own, and its value is partly in what it proves about the remaining bottleneck.

The decision also has a regulatory dimension. Replacing three operators changes the model's arithmetic, so the optimized graph is a model change, not a pure optimization. If the change is not covered by the approved plan, the graph must be reverted and the arithmetic reduction pursued instead. The cheapest technical fix is not always the shippable one.

A third consideration is maintainability. A graph with four boundaries is harder to debug than a graph with none, because a latency regression can originate in any partition or in the copies between them. The operational cost of heterogeneity should be weighed against its performance benefit. For a device that will be maintained for a decade, the simpler graph is often the better product decision.

The residual 3.8 ms is not a graph problem, and continuing to tune the graph would be a misallocation of effort. The next step is a model-level reduction, measured against the same 40 ms budget.

## 7. Code, query, or calculation

The script models a graph's latency with fusion and fallback, computes memory traffic and KV cache size, and implements the quantization core with error metrics. It reproduces Section 6.

```python
def fused_latency(nodes, fusion_groups, launch_ms=0.05):
    """Sum the per-node times, then replace fused groups by their max time."""
    times = {name: t for name, t, _ in nodes}
    total = sum(times.values())
    for group in fusion_groups:
        group_time = max(times[n] for n in group)
        total -= sum(times[n] for n in group) - group_time
    return total + launch_ms * (len(nodes) - sum(len(g) - 1 for g in fusion_groups))


def partition_cost(nodes, unsupported, cpu_ms=0.80, npu_ms=0.05, boundary_ms=0.80):
    extra_compute = sum(cpu_ms - npu_ms for n in unsupported)
    boundaries = len(unsupported) + 1
    return extra_compute + boundaries * boundary_ms, boundaries
```

Compute the baseline and the fused graph:

```python
nodes = [
    ("conv1", 8.0, "npu"), ("bn1", 0.5, "npu"), ("relu1", 0.3, "npu"),
    ("conv2", 14.0, "npu"), ("bn2", 0.5, "npu"), ("relu2", 0.3, "npu"),
    ("pool", 2.0, "npu"), ("conv3", 16.0, "npu"), ("bn3", 0.5, "npu"),
    ("relu3", 0.3, "npu"), ("gap", 0.8, "npu"), ("fc", 2.5, "npu"),
]
groups = [("conv1", "bn1", "relu1"), ("conv2", "bn2", "relu2"),
          ("conv3", "bn3", "relu3")]

print("unfused sum =", sum(t for _, t, _ in nodes))
print("fused =", round(fused_latency(nodes, groups), 2))

cost, boundaries = partition_cost(nodes, ["gelu", "layernorm", "softmax"])
print("fallback cost =", cost, "boundaries =", boundaries)
print("fused + fallback =", round(fused_latency(nodes, groups) + cost, 2))
```

Expected output:

```text
unfused sum = 45.7
fused = 44.2
fallback cost = 5.45 boundaries = 4
fused + fallback = 49.65
```

Memory and KV cache:

```python
intermediates = 12
tensor_kib = 8
print("unfused intermediates KiB =", intermediates * tensor_kib)
print("fused intermediates KiB =", 6 * tensor_kib)
print("with 4 boundaries KiB =", 6 * tensor_kib + 4 * tensor_kib)

layers, heads, head_dim, seq, bytes_per = 12, 12, 64, 512, 2
kv = 2 * layers * heads * head_dim * seq * bytes_per
print("kv bytes =", kv, "MiB =", round(kv / 1024 / 1024, 2))
print("kv int8 MiB =", round(kv / 2 / 1024 / 1024, 2))
```

Expected output:

```text
unfused intermediates KiB = 96
fused intermediates KiB = 48
with 4 boundaries KiB = 80
kv bytes = 18874368 MiB = 18.0
kv int8 MiB = 9.0
```

Quantization diagnostics:

```python
import math
import random


def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def symmetric_scale(values, bits=8):
    return max(abs(v) for v in values) / (2 ** (bits - 1) - 1)


def quantize_symmetric(values, scale, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    return [min(qmax, max(qmin, round_half_away(v / scale))) for v in values]


def dequantize(codes, scale):
    return [q * scale for q in codes]


def sqnr(original, reconstructed):
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    return 10 * math.log10(signal / noise) if noise else float("inf")


random.seed(11)
weights = [[round(random.gauss(0, 0.09), 4) for _ in range(64)]
           for _ in range(512)]
weights[7] = [round(v * 6.0, 4) for v in weights[7]]
weights[7][3] = 0.62
flat = [v for row in weights for v in row]

scale = symmetric_scale(flat, 8)
codes = quantize_symmetric(flat, scale, 8)
median = sorted(abs(v) for v in flat)[len(flat) // 2]
print("scale %.8f step/median %.4f" % (scale, scale / median))
print("zeros", sum(1 for q in codes if q == 0), "levels", len(set(codes)))
print("sqnr %.2f" % sqnr(flat, dequantize(codes, scale)))
```

Expected output:

```text
scale 0.01016693 step/median 0.1667
zeros 1411 levels 102
sqnr 30.00
```

Assumptions: per-node times include dispatch; fusion replaces a group's time with the maximum of its members plus no epilogue cost; each unsupported operator costs one boundary on each side; boundary cost is 0.80 ms; the KV cache uses FP16 and the model has 12 layers, 12 heads, and head dimension 64; and the quantization is symmetric per-tensor.

## 8. Data, standards, and regulatory context

Inference optimization is constrained by runtime specifications and by the same regulatory framework as the other compression methods. The runtime specifications define what the graph means; the regulations define what evidence is needed when the graph changes.

### 8.1 Runtime documentation as the operative specification

ONNX Runtime documents three graph optimization levels. Basic optimizations are semantics-preserving and include constant folding, redundant node elimination, and node fusions such as Conv-Add, Conv-Mul, Conv-BatchNorm, and Relu-Clip. Extended optimizations include GEMM activation fusion, MatMul-Add fusion, GELU fusion, layer normalization fusion, attention fusion, and skip-layer normalization fusion, and apply only to specific execution providers. Layout optimizations, such as the NCHWc optimizer, are CPU-only.

The documentation also states that offline-optimized models must match the target hardware and session options. A model optimized for one execution provider can be slower or invalid on another. This is a runtime constraint, not a regulatory one, but it determines whether the deployed artifact works.

ExecuTorch documents a partitioner-and-delegate architecture. A backend's partitioner assigns nodes to the backend; the delegate's `preprocess` produces a compiled blob that is serialized into the `.pte` file; unsupported nodes fall back to the portable CPU implementation. Each `.pte` file is backend-specialized, so a model must be exported once per target, and the partition must be inspected to know what runs where.

TensorFlow Lite for Microcontrollers documents a single tensor arena with a head, a temporary region, and a tail, and a greedy memory planner. The runtime fits in about 16 KiB on a Cortex-M3. The arena size is a build-time constant determined by experiment, and the documentation provides recording APIs to print the allocation plan.

### 8.2 IEC 62304 and the software item

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79, requires the runtime and the optimization toolchain to be configuration items if their output is shipped. The optimized graph is the software item; the training graph is not.

This has a practical consequence: a runtime upgrade that changes fusion or kernel selection is a change to a configuration item, and it requires re-verification even if the model file is unchanged.

### 8.3 ISO 14971 and numerical change

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified and controlled. An arithmetic-changing optimization, such as a fused GELU approximation, can change outputs. The risk file should record:

- The optimization and the magnitude of the numerical change.
- The inputs on which the change is largest.
- The clinical consequence if the decision changes.
- The validation that bounds the risk.

A semantics-preserving pass needs a diff test. An arithmetic-changing pass needs a clinical evaluation proportional to the change.

### 8.4 FDA PCCP and runtime changes

The FDA's final guidance on Predetermined Change Control Plans for AI-enabled device software functions was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022.

A runtime upgrade or a graph optimization is a software change. Whether it is covered depends on the plan and on whether the change alters the device's outputs or its performance envelope. A plan that anticipated "runtime updates that do not change outputs beyond a stated tolerance" may cover a fusion pass; a plan that did not mention runtime changes may not.

### 8.5 EU framework

In the European Union, Regulation (EU) 2017/745 governs the device and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk systems. The Annex I applicability date was originally 2 August 2027; the AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves it to 2 August 2028. Verify the current date before relying on it.

### 8.6 What the standards do not say

- No standard defines "optimized graph" or requires a specific pass order.
- No standard defines a numerical tolerance for equivalence; the team must justify one.
- No standard requires a partition report, although IEC 62304's design documentation implies it.
- No standard says whether a runtime upgrade is a new device; that depends on the change and the plan.

The absence means the evidence must be designed, not looked up.

### 8.8 A validation protocol for an optimized graph

A protocol that a reviewer can follow has six parts.

1. **Reference:** the unoptimized graph, its hash, and its outputs on a frozen input set.
2. **Tolerance:** the maximum absolute difference and the minimum decision agreement, justified clinically.
3. **Coverage:** typical inputs, tails, borderline cases, and every branch.
4. **Target measurement:** p50, p95, and p99 on the production board, with the runtime version.
5. **Memory:** peak high-water mark against the arena, with headroom.
6. **Change classification:** whether the pass is semantics-preserving, arithmetic-changing, or layout-changing, and the corresponding evidence.

A protocol without the tolerance is an opinion. A protocol without the target measurement is a desktop result.

### 8.7 What is binding and what is convention

| Item | Status | Consequence |
|---|---|---|
| ONNX Runtime optimization levels | Runtime documentation | Determines which passes apply |
| ExecuTorch delegate contract | Runtime documentation | Determines partition and fallback |
| TFLM arena model | Runtime documentation | Determines memory planning |
| IEC 62304 | Binding where required | Runtime and toolchain are configuration items |
| ISO 14971 | Binding where required | Numerical changes enter the risk file |
| FDA PCCP guidance | Non-binding; statute binding | Runtime change may need pre-authorization |
| "Optimized model" | Convention | Must state the passes and the runtime version |
| Numerical tolerance | Convention | Must be stated and justified |

## 9. Industry practice and case evidence

### 9.1 What runtimes document

The runtime documentation is the best public evidence about what optimization actually does, because it describes the passes that exist and the conditions under which they apply.

- ONNX Runtime separates basic, extended, and layout optimizations, and states that extended fusions apply only to specific execution providers. This is an admission that the same model can have different graphs on different backends.
- ExecuTorch states that each `.pte` file is backend-specialized and that unsupported operations fall back to the portable CPU. This is an admission that portability requires per-target export.
- TFLM states that the tensor arena is a single buffer with a greedy planner and that the runtime fits in about 16 KiB on a Cortex-M3. This is an admission that memory is planned, not incidental.

These are documented facts about the runtimes. The inference for a deployment is that the runtime version and the backend are part of the model definition for validation purposes.

### 9.2 The KV cache evidence

Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023, arXiv:2309.06180), reports that prior serving systems used only 20.4–38.2% of their KV cache memory for actual token states, because of reserved slots, internal fragmentation, and external fragmentation. PagedAttention allocates the cache in fixed-size blocks with a block table, achieving about 96.3% effective usage with a default block size of 16 tokens, and enables sharing through copy-on-write for parallel sampling and beam search.

The paper reports 2–4× throughput improvements over FasterTransformer and Orca at equal latency on its workloads. Those are serving-system measurements on data-center GPUs, not on-device measurements. The transferable idea is the block-based allocation and the fragmentation analysis, not the throughput number.

### 9.3 Speculative decoding

Speculative decoding uses a small draft model to propose several tokens and a larger model to verify them in one pass. The speedup depends on the acceptance rate: if the draft model's proposals are accepted often, the large model does fewer sequential passes; if they are rejected often, the draft work is wasted.

The method is documented in the literature and implemented in several serving systems. The quantitative benefit is workload- and model-pair-specific, and it should be measured rather than assumed. For a medical device, the draft model is an additional model in the evidence chain, with its own failure modes.

### 9.4 What the evidence supports

| Claim | Status |
|---|---|
| Fusion reduces memory traffic without changing arithmetic | Established; standard compiler technique |
| Conv-BN fusion changes numerical results slightly | Established; accumulation order and rounding |
| Operator fallback costs more than the operator's arithmetic | Consequence of boundary copies and synchronization |
| NHWC is preferred on many mobile CPUs and NPUs | Vendor documentation |
| PagedAttention improves KV cache utilization | Kwon et al., serving workloads |
| Speculative decoding speeds up generation | Workload-dependent; measured per pair |
| A 20% latency reduction from fusion is typical | Not established; model- and runtime-specific |
| An optimized graph is numerically identical | Not supported; tolerance must be stated |

### 9.5 What is not established

Several common claims are weaker than they sound.

- **"Fusion gives 2×."** Fusion gives whatever the memory-traffic reduction gives on a specific model and device. On a compute-bound model, it may give 3%.
- **"The NPU is 10× faster."** The NPU is faster on supported operators and shapes; the end-to-end gain depends on partitions and fallbacks.
- **"Quantization and fusion compose."** They compose only if the scales are recomputed after fusion.
- **"The optimized model is equivalent."** It is equivalent within a tolerance that must be stated and tested.

Each claim is a hypothesis about a specific configuration, not a general property.

### 9.6 Three deployment patterns

**Single-backend, fused.** The graph runs entirely on one accelerator, with fusion applied by the vendor compiler. This is the simplest and most predictable pattern. The risk is vendor lock-in and a graph that must be re-exported for each target.

**Heterogeneous, partitioned.** The graph is split between an accelerator and a CPU, usually because of unsupported operators. The risk is boundary cost and jitter. The mitigation is to minimize boundaries, not to minimize the number of unsupported operators.

**CPU-only, optimized.** The graph runs on the CPU with SIMD kernels, fusion, and quantization. This is the fallback when no accelerator is available, and it is often sufficient for small models. The risk is that it consumes CPU cycles needed by the rest of the product.

Do not choose the pattern by accident. A graph that becomes heterogeneous because of one operator is a design decision made by omission.

The pattern should be chosen at design time, not discovered at integration.

### 9.7 Why medical inference optimization differs

**The decision, not the output, is the unit of validation.** A 0.1% change in logits is irrelevant if it does not change a decision at the threshold, and a 0.01% change is critical if it flips a borderline case. The reference set must include the borderline cases.

**The runtime is part of the device.** A model file that runs correctly on a workstation runtime and incorrectly on the device runtime is a device defect, not a build issue. The runtime version belongs in the design record.

**The change is regulated.** A fusion that changes outputs is a model change. The regulatory pathway depends on the change, not on the intent.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unsupported operator | Partition count high | No kernel for the op | Latency spike | Operator audit before design |
| Boundary cost ignored | Fallback "only one op" | Copies and sync | Larger than expected | Estimate boundary cost |
| Fusion not applied | Graph unchanged after export | Runtime lacks the pattern | No gain | Inspect the graph |
| Layout transposes | Extra nodes after conversion | Mismatched layout | Slower than baseline | Fix layout assignment |
| Numerical drift after fusion | Outputs differ on a few inputs | Accumulation order | Decision changes | Reference set with tails |
| Peak memory exceeds arena | Initialization failure | Live set or copies | Device fails to boot | Plan memory; reserve headroom |
| Constant folding enlarges model | File size grows | Materialized constants | Flash overrun | Check size after folding |
| Dynamic shapes force fallback | Shape-dependent latency | Recompilation or CPU path | Tail latency | Specialize or bucket |
| KV cache overflow | Context truncation | Linear growth | Wrong answers | Bound context; quantize cache |
| Speculative decoding rejected | Draft acceptance low | Mismatched draft model | Slower than baseline | Measure acceptance; disable |
| Runtime upgrade changes behavior | Different latency or outputs | Kernel or fusion change | Unvalidated change | Pin and re-validate |
| Determinism lost | Different outputs across runs | Parallel reduction order | Unreproducible evidence | Test determinism; fix seeds |
| Fusion disabled by a near-miss pattern | Graph unchanged | Pattern mismatch | No gain | Inspect and rewrite the graph |
| Boundary tensor too small to amortize | High boundary count | Fixed sync dominates | Latency regression | Merge regions |
| Fallback path not tested | Product behavior unknown | Main path only | Unsafe failure | Test the fallback |
| Reference set misses the tail | Equivalence looks perfect | Unrepresentative inputs | Clinical miss | Add tail cases |
| Cache layout mismatch | Extra transposes | Vendor layout requirement | Slower than baseline | Use the vendor layout |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Per-node times are additive | Latency estimate | Overlap, pipelining | Measure the graph |
| Fusion time equals the max of members | Simple model | Epilogue cost, register pressure | Microbenchmark the fused kernel |
| Boundary cost is 0.80 ms | Fallback estimate | Different bus, copy size | Measure a boundary |
| Unsupported operators cost 0.80 ms on CPU | Fallback estimate | Operator and shape | Measure each operator |
| Launch overhead is 0.05 ms | Fusion estimate | Runtime and device | Measure with an empty kernel |
| The reference set covers the decision boundary | Validation | Missing borderline cases | Add tail cases |
| Numerical tolerance is adequate | Equivalence | Sensitive inputs | Inspect disagreements |
| The runtime version is fixed | Reproducibility | Upgrade | Pin and re-baseline |
| The layout is consistent | Performance | Mixed operators | Inspect conversions |
| The arena has headroom | Feasibility | Runtime scratch allocations | Measure high-water mark |
| KV cache is the dominant memory | LLM sizing | Weights or activations dominate | Profile |
| Determinism is guaranteed | Evidence | Parallel reductions | Run repeatedly |
| The fusion pattern is recognized | Predicts gain | Near-miss pattern | Inspect the graph |
| Boundary tensors are large enough | Predicts gain | Fixed sync dominates | Measure a boundary |
| The vendor layout is used | Predicts speed | Layout conversion | Inspect conversions |
| The fallback path is specified | Product safety | Main path only | Test the fallback |
| The reference set covers the tail | Equivalence | Unrepresentative inputs | Add tail cases |

The central limit is that graph optimization is runtime-specific. A pass that helps on one backend can hurt on another, and a graph that is optimal for one device is not optimal for a second. The measurement must be repeated per target, and the optimization record must name the runtime.

A second limit is that the boundary between optimization and model change is not always clear. A fused GELU approximation is an optimization in intent and a model change in effect. The classification should follow the effect on outputs, not the label the team gives the pass.

## 12. Exercises

**Exercise 1 — Optimization categories (recall, 25 minutes).**
Classify constant folding, Conv-BN fusion, NCHW-to-NHWC conversion, GELU approximation, and dead node elimination into semantics-preserving, arithmetic-changing, or layout-changing.
*Deliverable:* a five-row table.
*Acceptance criteria:* each classification justified; at least one item belongs to two categories and is noted.
*Time:* 25 minutes.

**Exercise 2 — Memory traffic (calculation, 35 minutes).**
A chain of three operators each reads and writes an 8 KiB tensor. Compute the bytes moved unfused and fused, assuming the input and output are counted once.
*Deliverable:* a byte calculation.
*Acceptance criteria:* unfused = 6 tensor accesses; fused = 2; saving stated in KiB.
*Time:* 35 minutes.

**Exercise 3 — Fallback cost (calculation, 30 minutes).**
A graph has two unsupported operators, each 0.6 ms on CPU and 0.04 ms on the accelerator. Boundary cost is 0.5 ms. Compute the total fallback cost and the number of boundaries.
*Deliverable:* a calculation.
*Acceptance criteria:* extra compute = 1.12 ms; boundaries = 3; boundary cost = 1.5 ms; total = 2.62 ms.
*Time:* 30 minutes.

**Exercise 4 — Layout (analysis, 30 minutes).**
A model runs on a mobile CPU with SIMD and on a desktop GPU. Choose a layout for each and explain the consequence of using the other.
*Deliverable:* a short recommendation.
*Acceptance criteria:* NHWC for mobile, NCHW for GPU; transposes and kernel selection explained.
*Time:* 30 minutes.

**Exercise 5 — Peak memory (calculation, 40 minutes).**
A graph has tensors A (8 KiB), B (8 KiB), C (16 KiB), and D (8 KiB), with lifetimes A→B, B→C, C→D. Compute the peak live set and the arena size with 10% overhead.
*Deliverable:* a lifetime diagram and a calculation.
*Acceptance criteria:* peak = 24 KiB (B+C); arena = 26.4 KiB with overhead.
*Time:* 40 minutes.

**Exercise 6 — KV cache (calculation, 30 minutes).**
A model has 24 layers, 16 heads, head dimension 64, context 1,024, and FP16 cache. Compute the cache size and the size after INT8 quantization.
*Deliverable:* a calculation.
*Acceptance criteria:* 2 × 24 × 16 × 64 × 1,024 × 2 = 100,663,296 B = 96 MiB; INT8 = 48 MiB.
*Time:* 30 minutes.

**Exercise 7 — Numerical equivalence (design, 45 minutes).**
Design a reference set and a tolerance for validating a fused graph. State the inputs, the metrics, and the acceptance rule.
*Deliverable:* a one-page protocol.
*Acceptance criteria:* tails included; decision agreement metric; tolerance justified; disagreement inspection procedure.
*Time:* 45 minutes.

**Exercise 8 — Optimization record (design, 50 minutes).**
Write the record for the Section 6 fused graph with fallbacks, including passes, runtime version, partitions, numerical results, and change classification.
*Deliverable:* a two-page record.
*Acceptance criteria:* reproducible; partitions listed; numerical comparison included; change classified.
*Time:* 50 minutes.

## 13. Self-check answers

**A1. Classifications for Exercise 1.**
Constant folding: semantics-preserving. Conv-BN fusion: arithmetic-changing, because the accumulation and rounding order change, even though the exact-arithmetic result is the same. NCHW-to-NHWC: layout-changing. GELU approximation: arithmetic-changing. Dead node elimination: semantics-preserving. Conv-BN fusion is also layout-relevant if the kernel changes, which is why it belongs to two categories in practice.

**A2. Memory traffic for Exercise 2.**
Unfused: op1 reads input and writes A (2 accesses), op2 reads A and writes B (2), op3 reads B and writes output (2) = 6 accesses × 8 KiB = 48 KiB. Fused: reads input, writes output = 2 accesses × 8 KiB = 16 KiB. Saving = 32 KiB, or 67%.

**A3. Fallback cost for Exercise 3.**
Extra compute = 2 × (0.6 - 0.04) = 1.12 ms. Boundaries = 3 (before, between, after). Boundary cost = 3 × 0.5 = 1.5 ms. Total = 2.62 ms. The boundaries are 57% of the cost.

**A4. Layout for Exercise 4.**
Mobile CPU with SIMD: NHWC, because the channel dimension is contiguous and vectorizes across channels. Desktop GPU: NCHW, because the libraries and memory access patterns are optimized for it. Using NHWC on the GPU causes transposes or slow kernels; using NCHW on the mobile CPU causes strided access and poor SIMD utilization.

**A5. Peak memory for Exercise 5.**
Lifetimes: A ends when B begins, B ends when C begins, C ends when D begins. Peak live set is B + C = 8 + 16 = 24 KiB. With 10% overhead, 26.4 KiB.

**A6. KV cache for Exercise 6.**
2 × 24 × 16 × 64 × 1,024 × 2 = 100,663,296 bytes = 96 MiB. INT8 halves it to 48 MiB.

**A7. Why operator fallback is expensive.**
Because the cost is dominated by the boundary: a copy of the tensor, a synchronization, and a dispatch, plus the loss of fusion across the boundary. The unsupported operator's arithmetic is often a small fraction of the penalty.

**A8. Why fusion can change accuracy.**
Because the accumulation order and the rounding change. In exact arithmetic the result is the same; in floating point or fixed point, the last bits differ. For a model near a decision boundary, the difference can flip a decision.

**A9. Rubric for Exercise 7 (equivalence protocol).**
A strong protocol includes typical inputs, tails, borderline cases, and every branch; states maximum absolute difference, decision agreement, and subgroup agreement; justifies the tolerance clinically; and describes how disagreements are inspected. A weak protocol uses random inputs and a single average difference.

**A10. Rubric for Exercise 8 (record).**
A strong record lists the pass sequence, runtime and compiler versions, fusion and partition lists, layout, memory plan, numerical comparison, target measurements, and change classification. A weak record names the runtime and the final latency only.

**A11. Why does fusion before quantization matter?**
Fusion changes the weights (Conv-BN folding) and the accumulation order. Quantization computes scales from the fused weights and activations. Quantizing first and fusing second leaves scales that no longer match the graph, so the model must be recalibrated and may lose accuracy in the interim.

**A12. Why can a runtime upgrade invalidate a validation?**
Because the runtime selects kernels, applies fusions, and plans memory. A new version can change any of these, so the same model file can produce different outputs or latency. The runtime is a configuration item, and the validation must be repeated or the runtime pinned.

## 14. Interview questions

**Q1. What does operator fusion actually save?**
A strong answer says memory traffic and launch overhead, not arithmetic, and gives the conv-BN-ReLU example. A weak answer says it makes the model faster without a mechanism.

**Q2. Why can one unsupported operator double latency?**
A strong answer explains partitioning, the copy and synchronization at each boundary, and the loss of fusion. A weak answer says the operator runs on a slower processor.

**Q3. What is the difference between NCHW and NHWC?**
A strong answer defines the orders, names the hardware that prefers each, and explains the transpose cost of mixing them. A weak answer says one is faster.

**Q4. How do you compute peak memory?**
A strong answer computes tensor lifetimes, finds the maximum live set, adds runtime overhead and alignment, and compares with the arena. A weak answer sums all tensors or quotes the model size.

**Q5. What is a KV cache and why does it matter on device?**
A strong answer explains the stored keys and values, the linear growth with context, the memory formula, and the quantization option. A weak answer calls it a cache without the size implication.

**Q6. What is speculative decoding?**
A strong answer describes the draft model, the verification pass, the acceptance rate, and the workload dependence. A weak answer says it makes generation faster.

**Q7. When is an optimization a model change?**
A strong answer says when it changes outputs beyond the stated tolerance, regardless of intent, and describes the validation and change-control consequence. A weak answer calls all optimizations build settings.

**Q8. How do you validate an optimized graph?**
A strong answer uses a reference set with tails and borderline cases, reports output differences and decision agreement, and measures on the target. A weak answer compares average accuracy.

**Q9. Why does a runtime upgrade require re-validation?**
A strong answer says the runtime is a configuration item, kernel selection and fusion can change, and the same model file can behave differently. A weak answer assumes the model file determines behavior.

**Q10. What is the difference between throughput and latency for a medical device?**
A strong answer says latency is the time for one patient's inference and throughput is the rate across patients, and that batching improves the latter while worsening the former. A weak answer uses the terms interchangeably.

**Q11. How would you reduce the number of partitions?**
A strong answer replaces unsupported operators, chooses a backend with broader support, or accepts a slower backend for a region to avoid boundaries. A weak answer ignores the problem.

**Q12. Why is a boundary between partitions expensive?**
A strong answer names the copy, the synchronization, the dispatch, and the lost fusion across the boundary, and notes that fixed synchronization often dominates for small tensors. A weak answer says the CPU is slower.

**Q13. How do you keep an optimized graph reproducible?**
A strong answer pins the runtime and toolchain, records the pass order and parameters, stores the reference outputs, and fixes kernel selection. A weak answer stores the model file only.

**Q14. What would you record about an optimized graph?**
A strong answer lists passes, versions, fusions, partitions, layouts, memory plan, numerical results, and change classification. A weak answer records the final latency.

## 15. Key terms

- **Arena** — preallocated memory region for tensors.
- **Boundary** — copy and synchronization between partitions.
- **Constant folding** — evaluating constant subgraphs at compile time.
- **Continuous batching** — adding requests to a running batch.
- **Conv-BN fusion** — folding batch normalization into convolution weights.
- **Dead node elimination** — removing nodes whose outputs are unused.
- **Decode** — generating one token at a time.
- **Delegate** — backend that executes a subgraph.
- **Determinism** — same output on every run.
- **Execution provider** — runtime backend for a device.
- **Fallback path** — product-level alternative when the model is unavailable.
- **Fusion** — combining nodes into one kernel.
- **Graph capture** — recording a graph for replay.
- **Graph optimization** — rewriting a graph to improve execution.
- **Inter-token latency** — time between output tokens.
- **KV cache** — stored keys and values for autoregressive decoding.
- **Kernel autotuning** — selecting the fastest kernel for a shape.
- **Layout optimization** — converting tensor order for efficiency.
- **Memory planning** — assigning buffers to minimize peak memory.
- **NCHW** — batch, channel, height, width layout.
- **NHWC** — batch, height, width, channel layout.
- **Numerical equivalence** — same outputs within a stated tolerance.
- **Operator fallback** — running an unsupported operator elsewhere.
- **Operator support list** — operators a backend implements.
- **PagedAttention** — block-based KV cache management.
- **Partition** — subgraph assigned to one processor.
- **Peak live set** — maximum simultaneously live tensors.
- **Prefill** — processing the prompt in parallel.
- **Prompt caching** — reusing computed prefix states.
- **Requantization** — converting accumulators to output precision.
- **Soak test** — sustained load test.
- **Speculative decoding** — draft proposes, larger model verifies.
- **Throughput** — inferences or tokens per second.
- **Tiling** — splitting a tensor into cache-sized tiles.
- **Time to first token** — latency until the first output token.
- **Graph partitioner** — component that assigns nodes to backends.
- **Operator support list** — operators a backend implements.
- **Requantization** — converting accumulators to output precision.
- **Numerical tolerance** — permitted output difference for equivalence.

## 16. Sources

All URLs accessed 25 October 2026.

**Runtime documentation**
- ONNX Runtime, "Graph optimizations". https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- PyTorch, "Understanding backends and delegates", ExecuTorch. https://docs.pytorch.org/executorch/main/compiler-delegate-and-partitioner.html
- PyTorch, "ExecuTorch backends". https://docs.pytorch.org/executorch/main/backends-section.html
- Google AI Edge, "LiteRT for Microcontrollers overview". https://developers.google.com/edge/litert/microcontrollers/overview
- TensorFlow, "Memory management in TFLM". https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md
- NVIDIA, "TensorRT documentation". https://docs.nvidia.com/deeplearning/tensorrt/latest/

**Language-model inference**
- Kwon, W., et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. https://arxiv.org/abs/2309.06180

**Quantization interaction**
- Jacob, B., et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017/2018. https://arxiv.org/abs/1712.05877
- Nagel, M., et al., "A White Paper on Neural Network Quantization", 2021. https://arxiv.org/abs/2106.08295

**Standards and regulation**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

> **Note:** The Section 6 graph and timings are constructed for teaching. PagedAttention figures are the authors' measurements on serving workloads, not on-device results.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`06_pruning-and-sparsity.md`](06_pruning-and-sparsity.md)
- Next topic: [`08_on-device-architecture.md`](08_on-device-architecture.md)
- Related topic: [`02_target-hardware-and-performance.md`](02_target-hardware-and-performance.md)
- Related topic: [`05_quantization.md`](05_quantization.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
