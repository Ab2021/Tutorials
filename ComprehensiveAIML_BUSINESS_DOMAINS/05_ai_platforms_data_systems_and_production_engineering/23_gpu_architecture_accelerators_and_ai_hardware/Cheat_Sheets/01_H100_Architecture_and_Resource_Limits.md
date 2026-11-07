---
collection: gpu_architecture_accelerators_and_ai_hardware
file_id: "01"
topic: H100 architecture and resource limits
level: 1
status: draft
version: "1.0"
created: 2026-11-07
reviewed: null
review_trigger: GPU SKU, CUDA compiler, driver, profiler, or workload changes
notation_style: plain-text
prerequisites: [CUDA thread/block and basic memory concepts]
source_status: OPEN
owner_role: GPU platform engineering
source_index: INDEX.md
---
# H100 Architecture and Resource Limits

## 1. File Header

> **Purpose:** Connect H100 architecture to resource-constrained kernel placement and an evidence-driven production diagnosis.
> **Prerequisites:** CUDA threads, blocks, warps, bytes, division and hardware/software version recording.
> **Key outcome:** Calculate simplified resident-block limits, identify missing allocation rules, and plan a discriminating measurement.
> **This file does NOT cover:** A complete TMA programming tutorial, a WGMMA instruction guide, or multi-node collective implementation.
> **Level:** 1 Foundations, with production diagnosis.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** Sources below support architectural statements; constructed examples are ILLUSTRATIVE, not device specifications or measured benchmarks.
> **Retrieval cue:** For an occupancy arithmetic problem start at Section 5; for an incident start at Section 8.

This sheet follows the H100 architecture inventory in `Train/Templates/directory.txt`, module 01, without claiming the inventory itself is a hardware source.
**SOURCED:** NVIDIA's Hopper tuning guide describes compute capability 9.0 and CUDA-specific architectural changes [S1].
An H100 PCIe card, H100 SXM device, and H100 NVL system are not interchangeable benchmark configurations.
Installed device properties, topology, clocks, firmware, driver, runtime, compiler and kernel launch parameters define the actual experiment.
No real workload throughput, latency, accuracy, price, or energy result is claimed here.
**Takeaway:** A device name is a starting point for evidence, not a complete execution environment.

## 2. Concept Map / Mental Model

```text
CPU launches grid of thread blocks
          | block placement depends on resources and device constraints
          v
GPU -> GPCs -> SMs -> warp schedulers -> eligible warp instructions
                  |                   |-> scalar/vector execution
                  |                   |-> Tensor Core operations
                  | registers and shared memory: residency limits
                  v
   shared/L1 <-> L2 <-> HBM; asynchronous TMA moves selected tiles
                         |
                         +-> NVLink / PCIe -> peer device or host
```
The hierarchy is schematic and is not an exact wiring diagram.
Arrows show organization or data movement, not guaranteed direct paths or equal bandwidth.
Block residency affects the number of available warps; dependencies and memory stalls affect whether those warps can issue.
The grid may execute in waves if all blocks cannot be resident at once.
NVLink connects supported devices and topology components; its presence does not imply every peer has a direct link.
**Takeaway:** Separate resource admission, instruction readiness, and actual data movement before attributing a performance limit.

## 3. Foundational Definitions

| Term | Meaning and units | Operational distinction |
|---|---|---|
| GPU | Accelerator executing CUDA grids | Product SKU and node topology must be identified separately. |
| GPC | Graphics processing cluster containing GPU processing resources | Do not equate a GPC to one SM or one CUDA block. |
| SM | Streaming multiprocessor, a scheduling and execution resource | Resident block resources are constrained per SM. |
| Warp | Group of 32 CUDA threads executing as a scheduling unit | A partially populated warp still consumes scheduling resources. |
| Block | Cooperating CUDA threads with block-scoped resources | A block is admitted to one SM; scheduling is not a performance guarantee. |
| Register | Per-thread logical storage allocated from physical SM resources | Allocations may be rounded; source variable count is not allocated registers. |
| Shared memory | Explicitly managed on-chip storage, typically shared within a block | Capacity and per-block limits are configuration-dependent. |
| Occupancy | Active resident warps divided by the supported maximum resident warps | Occupancy is not utilization or instruction throughput. |
| Eligible warp | Resident warp ready to issue a suitable instruction | A resident warp waiting for data is not necessarily eligible. |
| HBM | High-bandwidth device memory | Advertised bandwidth is not measured kernel bandwidth. |
| L2 | Device cache between SM-side requests and memory/system paths | Cache hits change HBM traffic and the relevant roofline. |
| Tensor Core | Specialized matrix arithmetic execution resource | Supported types, instruction forms, layouts and accumulation matter. |
| TMA | Tensor Memory Accelerator for supported asynchronous bulk tensor transfers | Transfer completion and visibility require correct synchronization. |
| NVLink | NVIDIA high-speed interconnect technology | Peer-access behavior and aggregate bandwidth depend on actual topology. |

A scheduler selects among eligible warps; it does not make all resident warps issue every cycle.
The compiler can trade registers against instruction count, spills, and occupancy.
The maximum block count is not necessarily achieved if any independent resource or placement condition binds first.

### From grid to execution

A host launch specifies a grid of blocks and the number of threads in each block.
The runtime and device schedule blocks subject to hardware and launch constraints.
The programmer must not assume that all blocks start together or that block execution order matches block index.
Within a conventional block, threads can cooperate using supported block-scoped synchronization and shared memory.
Some architectures and launch modes provide additional cluster-level facilities; those are separate contracts and are not assumed in the occupancy example.
On an SM, resident warps compete for scheduling opportunities according to the hardware's rules.
A warp can be resident but temporarily unable to issue because its next instruction depends on unfinished work.
Even an eligible warp may not issue immediately if another warp is selected or the required execution resource is unavailable.
The profiler's definitions and scope are essential when comparing any of these states.
Consequently, adding resident warps may hide some latency without increasing the amount of useful work per instruction.

### Registers, local memory and shared memory

Source variables do not map one-for-one to allocated hardware registers.
Compilation, optimization, unrolling, live ranges and target architecture affect the final allocation.
Inspect the compiled kernel rather than assuming the number of variables predicts occupancy.
If the compiler cannot keep all needed values in registers, some accesses can use local memory.
Local memory is a per-thread address space, not a promise that the data resides in on-chip storage.
Spill behavior and cache effects depend on the generated code and actual workload.
Shared memory is explicitly allocated for a block and can support cooperation and data reuse.
Its footprint includes both static and requested dynamic shared memory, subject to implementation-specific limits and rounding.
Shared memory does not automatically synchronize threads or validate address calculations.
When changing a tile layout, verify both address mapping and the required synchronization.

### Distinguishing caches and memory boundaries

L1 and L2 are caching levels, whereas HBM is device memory; they do not represent interchangeable traffic counts.
A source-level load may hit in cache and therefore need not generate the same number of HBM bytes.
Conversely, imperfect access patterns and repeated loads can move more data than a simple payload estimate.
For a roofline calculation, define whether bytes cross the SM-to-cache, cache-to-cache or device-memory boundary.
Use a bandwidth ceiling compatible with the chosen boundary and precision.
If the workload fits a cache in a repeated benchmark but not in production, the benchmark may conceal device-memory pressure.
If one workload reuses inputs while another streams new data, the same kernel can have different effective bottlenecks.
Compare cold and warm conditions deliberately instead of assuming either is representative.
The companion roofline sheet develops the corresponding arithmetic and measurement limits.

### Tensor operations and arithmetic correctness

Tensor Core availability does not mean every matrix operation or data type automatically uses the same execution path.
The compiler, library, target architecture, matrix dimensions, alignment and chosen numerical types all matter.
Inputs may require conversion or scaling; accumulation precision can differ from input precision.
The correct tolerance depends on the downstream decision and reference, not on a generic percentage.
For a reduction, error can change with the order of operations and the length of the reduction dimension.
Compare representative values, extremes, zeros and edge shapes against a suitable reference.
If quantization or lower precision is introduced, evaluate quality as well as latency and memory.
An optimization is acceptable only when its output meets the documented application contract.

### Asynchronous movement and buffer reuse

An asynchronous operation separates issue time from completion time.
The issuing thread or warp may proceed while a transfer is in progress, subject to the documented API.
Before consuming a destination buffer, the program must establish the required completion and visibility relationship.
Before reusing a buffer, it must also ensure that prior consumers are finished.
Double buffering can overlap movement and compute but increases the live storage footprint.
More buffers can reduce the number of blocks that fit on an SM.
Insufficient buffering can leave computation waiting for data; excessive buffering can worsen resource pressure.
Neither outcome follows from buffer count alone, so compare correct variants on the actual workload.
The full TMA descriptor, barrier and instruction contracts require their own version-specific guide.

### Device-to-device connectivity

The existence of NVLink in a product family does not specify the route between any two visible GPUs.
Record the actual platform, device pairing, topology and any partitioning before selecting a communication strategy.
Peer-access support, permissions and process placement must be checked for the deployed environment.
For a transfer benchmark, define payload size, direction, synchronization, warm-up and concurrent traffic.
One-way bandwidth is not the same quantity as summed bidirectional or fabric-wide bandwidth.
An isolated point-to-point measurement also does not predict all-reduce performance.
Collective behavior depends on algorithm, participating devices, payload, topology and contention.
Model and measure the communication pattern used by the application.
**Takeaway:** Resident, eligible, issued, and completed are different states and require different measurements.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Validity: a simplified homogeneous SM with a register pool R, identical blocks, and exact unrounded allocation of r registers per thread for T threads per block.
Derivation: each block consumes r × T registers; n resident blocks require n × r × T ≤ R; the largest integer n is the floor of R/(r × T).
**FORMULA [E1]:** register-only block bound = floor(R / (r × T)).
**WHERE:** R is registers per SM; r is registers per thread; T is threads per block.
**CONSTRAINT:** Ignore allocation granularity and all other limits; use actual hardware allocation rules for deployment.
**NUMERICAL EXAMPLE:** R = 65,536, r = 64, T = 256; each block consumes 16,384 registers; floor(65,536/16,384) = 4 blocks.
**INTERPRETATION:** Under this simplified register constraint alone, at most four such blocks fit.
Boundary: if a block requests more than the allowed per-block registers, launch feasibility must be checked independently.

### 4.2 General Case

Validity: resources are nonnegative capacities and each candidate block consumes known rounded allocations; all included limits apply to the same launch and SM configuration.
Derivation: each independent limit yields an upper bound on the same resident-block count; satisfying all of them requires their minimum.
**FORMULA [E2]:** B = min(B_registers, B_shared, B_threads, B_warps, B_blocks, other applicable placement bounds).
**WHERE:** B is the maximum candidate resident blocks per SM under the modeled constraints; each term is a nonnegative block count.
**CONSTRAINT:** Exact register and shared-memory allocation granularity, block limits, cluster placement and architecture-specific requirements must be incorporated.
**NUMERICAL EXAMPLE:** If only the register bound 4 and shared-memory bound 2 are considered, min(4,2) = 2 blocks per SM.
**INTERPRETATION:** Shared memory is the tighter of these two illustrative constraints.
Boundary: the minimum of incomplete bounds is only an upper bound on actual residency.

Validity: each block has a known actual allocated shared-memory footprint S and the modeled SM has capacity M available to that kernel.
Derivation: n × S ≤ M, so the greatest integer n is floor(M/S).
**FORMULA [E3]:** shared-memory block bound = floor(M/S).
**WHERE:** M and S are in the same byte units; S must be positive.
**CONSTRAINT:** Account for allocation rounding, reserved memory, static plus dynamic allocations, per-block opt-in, and configuration.
**NUMERICAL EXAMPLE:** M = 96 KiB and S = 48 KiB/block, so floor(96/48) = 2 blocks.
**INTERPRETATION:** Two blocks fit by this shared-memory constraint alone.
Boundary: a zero shared-memory request makes this resource nonbinding rather than permitting division by zero.

### 4.3 Key Special Cases / Variants

| Variant | Change | When useful | Disqualifier |
|---|---|---|---|
| Register-only | E1 | Explain one resource bound | Rounded allocations or other limits unknown |
| Shared-only | E3 | Compare tile sizes | Shared-memory carveout or overhead unknown |
| Combined | E2 | Identify the tightest modeled limit | Missing cluster or execution restrictions |
| Occupancy API | Device and kernel-aware estimate | Check a compiled launch | Runtime and profiler must match deployment |
| Profiler | Observe execution behavior | Differentiate bound from stalls | Profiling perturbation or counter scope unknown |

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

Increasing register allocation per thread cannot increase E1 when all other inputs stay fixed.
This monotonic bound does not establish that execution time rises: more registers can reduce spills or redundant computation.
Increasing block size can increase per-block register consumption and warp count simultaneously.
A kernel with fewer active warps can outperform one with more if the former issues useful work more effectively.
A large grid does not overcome a resource restriction on each SM; it only supplies more blocks to schedule in later waves.
**Takeaway:** Use the minimum of correctly rounded constraints, then measure performance rather than optimizing occupancy in isolation.

## 5. Worked Numerical Example

**ILLUSTRATIVE, ARITHMETIC CHECKED:** These values are assumptions for an exercise, not H100 specifications.
Decision: determine the joint resident-block ceiling under two deliberately simplified constraints.
Input register capacity: 65,536 registers per SM.
Input block size: 256 threads per block.
Input allocation: 64 registers per thread, with no rounding.
Input shared memory: 48 KiB per block.
Input available shared memory: 96 KiB per SM.
Omitted deliberately: block, thread, warp, allocation-granularity, reservation, cluster, carveout, and launch-validity limits.
Step 1: 256 threads/block × 64 registers/thread = 16,384 registers/block.
Step 2: 65,536 registers/SM ÷ 16,384 registers/block = 4 blocks/SM.
Step 3: 96 KiB/SM ÷ 48 KiB/block = 2 blocks/SM.
Step 4: min(4, 2) = 2 blocks/SM under the listed assumptions.
Independent check: 2 blocks use 32,768 registers and 96 KiB shared memory.
A third block would use 144 KiB shared memory and exceed the assumed 96 KiB limit.
Four blocks would fit the assumed registers exactly, but demand 192 KiB shared memory.
Wrong path: call four blocks the result after checking only registers; this ignores the tighter shared-memory bound.
The result is an upper bound when omitted constraints are restored, not evidence of an actual H100 occupancy value.
Production-scale interpretation: query the actual device, compile the real kernel, inspect resource usage and use the installed occupancy API.
Then measure achieved occupancy, eligible warps, instruction mix, memory activity and end-to-end latency under representative input.

### Sensitivity A: reducing shared memory without changing work

Keep the register assumptions fixed and suppose an implementation uses 32 KiB of shared memory per block instead of 48 KiB.
Assume, for this calculation only, that the change preserves the same mathematical output and does not alter any other resource.
The shared-memory ceiling is now floor(96/32) = 3 blocks per SM.
The register ceiling remains 4 blocks per SM.
Considering only these two constraints, the combined ceiling is min(4, 3) = 3 blocks.
The additional resident block is a capacity prediction, not a throughput prediction.
If the smaller tile increases global-memory traffic or synchronization, elapsed time could increase despite the higher ceiling.
Check that the proposed tile covers edge elements correctly and compare output with a reference implementation before measuring performance.
An implementation that achieves only two blocks may be limited by another constraint omitted from this exercise.

### Sensitivity B: trading shared memory for registers

Suppose the reduced shared-memory design also raises the compiled register allocation to 96 registers per thread.
The simplified register cost becomes 256 × 96 = 24,576 registers per block.
The register-only ceiling becomes floor(65,536/24,576) = 2 blocks.
The shared-memory ceiling remains 3 blocks at 32 KiB each.
The joint simplified ceiling is therefore min(2, 3) = 2 blocks.
Independently, two blocks consume 49,152 registers; three would consume 73,728, exceeding the illustrative register pool.
This example shows why optimizing shared-memory usage in isolation can move the bottleneck rather than increase residency.
Real compiler allocations may be rounded by architecture-specific units, so use the actual allocation rather than treating this estimate as a device measurement.

### Sensitivity C: comparing a resource ceiling to warp occupancy

Assume the warp size is 32 threads and each 256-thread block contains eight full warps.
At two resident blocks, there are 16 resident warps per SM under the simplified example.
If a separate illustrative limit is 64 resident warps per SM, the resulting theoretical occupancy is 16/64 = 25%.
This 64-warp limit is an exercise assumption, not a claim about a particular H100 configuration.
If the assumed limit were 48 warps instead, the same 16 resident warps would correspond to approximately 33.3%.
Both calculations describe resident-warps divided by an assumed maximum, not the fraction of cycles issuing useful instructions.
Partial warps, hardware scheduling partitions, launch limits, and actual resource granularity need separate treatment.
The number of blocks may also be constrained by limits that the two-resource estimate omits.

### Capacity check before a performance experiment

1. Record the exact GPU SKU, partitioning mode, driver and toolkit versions.
2. Compile the candidate kernel and retain the binary and compiler resource report.
3. Identify static and dynamic shared-memory use separately, then apply the documented allocation rules.
4. Use the compiled kernel, intended launch dimensions and dynamic shared-memory request with the installed occupancy API.
5. Confirm launch feasibility and correctness against a reference across ordinary and boundary-shaped inputs.
6. Measure unprofiled latency with a stable workload and an explicit warm-up and repetition protocol.
7. Profile the same workload and distinguish active warps from eligible warps, issue activity and memory behavior.
8. Compare candidates on useful throughput and correctness, not occupancy alone.
9. Investigate spills, traffic, synchronization and instruction changes if the predicted occupancy improvement fails to improve runtime.
10. Record tool versions, metric definitions and uncertainty so that another engineer can reproduce the comparison.

None of these steps has been performed on an actual device for this illustrative example.
The exercises do not establish an H100 benchmark, an approved launch configuration or a universal tuning rule.
The arithmetic is hand checked by multiplication and division; no device execution is claimed.
**Takeaway:** Four register-limited blocks become two combined blocks when the illustrative shared-memory limit is included.

## 6. Visual / Geometric Intuition

```text
Illustrative capacity per SM, not to scale:
register pool: [block A 16384][block B 16384][block C 16384][block D 16384]
shared pool:   [block A 48 KiB             ][block B 48 KiB             ]
                                                   ^ shared capacity ends
```
The same block needs both resources simultaneously.
The register pool admits four blocks in isolation; the shared pool admits two.
Two blocks consume all assumed shared capacity even though half the modeled register pool remains unused.
This diagram is not a physical floorplan or a model of allocation granularity.
**Takeaway:** Unused capacity in one resource cannot compensate for an exhausted independent resource.

## 7. Comparison / Selection Table

| Choice | Preconditions | Benefit | Cost / boundary | Evidence |
|---|---|---|---|---|
| Simple direct loads | Correct addressing and synchronization | Straightforward baseline | Can underuse data reuse | HEURISTIC; benchmark required |
| Shared-memory tiling | Reuse amortizes staging | May reduce repeated global accesses | Shared footprint can reduce residency | HEURISTIC; workload dependent |
| Asynchronous tensor transfer | Supported target, layout, descriptor and synchronization | Can overlap transfer and computation | Setup and barrier overhead; incorrect completion corrupts results | SOURCED mechanism [S1,S2]; performance NOT MEASURED |
| Tensor Core path | Supported arithmetic, layout and accuracy | Specialized matrix operation | Conversion, padding and numerical constraints | SOURCED [S1]; performance NOT MEASURED |
| More blocks per SM | Valid resource configuration | More potential latency hiding | More occupancy is not automatically faster | DERIVED from eligibility distinction |
| Peer GPU communication | Supported peer topology and permissions | Avoids some host-mediated paths | Link contention, NUMA and topology matter | SOURCED [S3]; workload rate NOT MEASURED |

First reject any choice that fails correctness, supported instruction, numerical tolerance, or synchronization requirements.
Next compare an uncomplicated baseline against a candidate on representative shapes and realistic concurrency.
If the memory footprint fails, reduce tile or staging footprint before attempting a larger launch.
If numerical quality fails, use a more appropriate precision or accumulation strategy rather than accepting a fast wrong answer.
If evidence conflicts, prioritize correctness and service-level constraints, then repeat controlled measurements.
Record shapes, dtype, clock and power conditions, kernel version, warm-up, repetitions and uncertainty.
No empirical comparison is presented; performance of each variant is NOT MEASURED.
**Takeaway:** Choose a correct baseline first and optimize the demonstrated bottleneck, not the architecture label.

## 8. Failure Modes & Edge Cases

All examples below are HYPOTHETICAL, not observed incidents.
| ID | Symptom / competing hypotheses | Discriminating check | Bounded action and verification |
|---|---|---|---|
| F01 | Low occupancy: registers, shared memory, block size, or placement | Inspect compiled resource usage and occupancy limiting factors | Change the actual binding resource; rerun correctness and timing. |
| F02 | High occupancy but slow kernel: dependencies, memory stalls, or instruction throughput | Compare eligible warps, issue activity, and stall reasons | Test one targeted change; do not infer causality from occupancy alone. |
| F03 | Spills after register cap: occupancy improved but local traffic increased | Compare compiler spill diagnostics and memory counters | Relax cap or change live ranges; compare speed and correctness. |
| F04 | TMA output stale: completion/barrier error versus wrong descriptor | Validate descriptor and completion protocol with small known tensor | Correct synchronization and retest under race-checking tools. |
| F05 | Tensor output differs: precision, layout, overflow or actual bug | Compare higher-precision reference and inspect errors by input range | Adjust accumulation/scaling or correct layout; enforce tolerances. |
| F06 | Multi-GPU bandwidth low: topology, unsupported peer access or contention | Inspect topology, peer access and isolated versus concurrent transfers | Choose supported path or placement; measure under production concurrency. |
| F07 | Unexpected HBM pressure: low cache reuse or excessive rereads | Compare L2 and device-memory traffic at matching boundaries | Improve locality only if counters and timing support the hypothesis. |
| F08 | Launch fails: invalid shared-memory request, block size or device support | Capture launch error and query kernel/device attributes | Fix unsupported launch and check every asynchronous error boundary. |
| F09 | Results intermittent: data race, missing synchronization, or uninitialized buffer | Reproduce with a small fixed seed and CUDA correctness tools | Fix ordering/ownership and rerun stress plus reference comparisons. |
| F10 | Slow after deployment: clock/power, driver change, workload or contention | Compare full environment and isolate the kernel from co-tenants | Roll back or mitigate the confirmed change; retain incident evidence. |
| F11 | Profiler and production disagree: replay, sampling, or changed clocks | Compare unprofiled repetitions with profiler configuration and scope | Reduce perturbation and corroborate with another signal. |
| F12 | One partition behaves differently: isolation profile or shared resources | Record GPU instance configuration and per-instance telemetry | Reproduce on matched partition; avoid generalizing across profiles. |

### F01 investigation: resource-bound residency

**Scenario:** A tile-size increase improves arithmetic reuse but decreases predicted resident blocks.
Capture the compiler's register and shared-memory report for both binaries rather than comparing source declarations alone.
Check whether dynamic shared memory was included in the launch and occupancy calculation.
Compute separate bounds using rounded allocations when the architecture documentation provides the necessary rules.
If the binding limit is shared memory, compare a smaller tile without also changing precision or input layout.
If registers bind, inspect live ranges and spill behavior before imposing a register limit.
Test the same shapes and correctness criteria for both versions.
Escalate if neither reported resource explains the difference; another launch, warp, block or placement constraint may bind.

### F03 investigation: occupancy gained at the cost of spills

**Scenario:** A register cap increases theoretical occupancy while kernel latency worsens.
Keep block size, input shape, clock policy and data layout fixed across the comparison.
Record compiler spill information for both binaries.
Compare device-memory and cache traffic, bearing in mind that local-memory accesses may be cached.
If the capped kernel has more spill traffic and worse unprofiled latency, the cap is a plausible cause.
Remove or relax the cap and repeat; a correlation in one profile alone is not proof.
If traffic is unchanged, investigate instruction count, dependencies, launch configuration and timing noise.
Never deploy the capped variant merely because its occupancy percentage is larger.

### F04 investigation: asynchronous copy completion

**Scenario:** A consumer sometimes observes incorrect values in a tile populated by an asynchronous transfer.
Start with a small known tensor and a deterministic scalar or ordinary-copy reference.
Record the descriptor, shapes, strides, destination region and all participating threads.
Check the version-specific requirements for issuing, completing and consuming that transfer.
Verify that the synchronization object is initialized and used at the correct scope and phase.
An issued operation is not equivalent to completed, visible data.
Run an appropriate correctness tool where its documented coverage includes the relevant operation.
Then stress repeated transfers and buffer reuse; a single passing execution does not establish absence of races.
If the descriptor is wrong, fixing synchronization alone will not correct the address calculation.

### F05 investigation: numerical discrepancy

**Scenario:** A matrix kernel differs from a reference on a subset of inputs.
First check whether both paths compute the same operation, including transposes, padding, scales and accumulation type.
Compare against a higher-precision reference when that reference is appropriate for the problem.
Inspect absolute and relative errors together; relative error is unstable near a zero reference.
Group discrepancies by input magnitude, matrix shape and reduction length.
If only extreme magnitudes fail, investigate overflow, underflow and scaling.
If only edge tiles fail, investigate masking, strides and out-of-bounds access before changing numerical tolerance.
Agree an application-specific error budget with the consumer; a fast result outside that budget is not a valid optimization.

### F06 investigation: topology-dependent transfers

**Scenario:** A multi-GPU transfer is slower between some device pairs.
Record the exact device IDs, process placement, NUMA affinity and visible topology.
Check whether peer access is supported and enabled for the particular pair and runtime configuration.
Measure a transfer-size sweep in both directions with an isolated workload and explicit synchronization.
Repeat under representative concurrent traffic to reveal contention.
Report payload size, elapsed time and whether bandwidth is one-way or aggregate.
Do not infer collective performance from a single peer copy: collective algorithm, topology and synchronization differ.
If pairwise access is unavailable, choose a documented supported transfer path and measure its impact.

### F09 investigation: intermittent incorrect output

**Scenario:** The same input occasionally produces different incorrect results.
Preserve the exact binary, launch configuration, input and output before changing the system.
Repeat with a fixed input and compare against an independent reference.
Check initialization of device buffers and whether their lifetime extends through asynchronous use.
Audit ownership and ordering between producers and consumers, including cross-stream dependencies.
Use the installed correctness tools only within their documented detection scope.
If the issue disappears under instrumentation, treat timing sensitivity as evidence to investigate, not proof of a race.
After a fix, test boundary shapes and repeated runs under representative concurrency.

### F10 investigation: regression after deployment

**Scenario:** End-to-end latency increases after a driver, binary, workload or infrastructure change.
Compare workload distributions and input shapes before attributing the change to the GPU.
Record GPU SKU, partitioning, driver, toolkit, clocks, power policy and co-tenant activity.
Separate data transfer, queueing, kernel execution and postprocessing in the measurement.
Run old and new binaries under matched conditions where rollback and compatibility permit.
Use repeated unprofiled timings and compare with the deployed service metric.
If the regression follows a configuration change, preserve the evidence and roll back only through the approved operational procedure.
Monitor correctness and customer-facing latency after mitigation; a kernel-level improvement alone may not resolve the service issue.

A negative register-spill test does not prove shared memory is the bottleneck.
A peer link test alone does not establish collective efficiency or application throughput.
Escalate recurring correctness faults, hardware errors, and unbounded contention to the owning platform and reliability teams.
A standalone collection-wide Common Issues artifact is planned, not written.
**Takeaway:** Treat the first symptom as a hypothesis and require a discriminating observation before changing production configuration.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let N be the number of data elements, K the reduction dimension, and b the byte size per element.
For a simple elementwise pass with one read and one write per element, work is O(N) and modeled data traffic is approximately 2Nb bytes, excluding caching and overhead.
A dense matrix multiply of dimensions M × K and K × P has O(MKP) scalar arithmetic; actual instruction counts depend on implementation and precision.
A tile's shared storage scales with its staged tile dimensions and number of buffers; padding, swizzling and alignment change allocated size.
A full diagnostic includes compilation and preprocessing, kernel launches, transfers, synchronization, warm-up and correctness checks.
Distributed workloads add topology-dependent communication, collectives, synchronization and straggler costs.
Big-O describes growth, not an actual runtime or the attainable hardware bandwidth.

### 9b. Back-of-Envelope Sizing

**ILLUSTRATIVE:** A kernel stages two 48 KiB buffers per block.
The staged payload is 2 × 48 = 96 KiB per block.
Under the example's assumed 96 KiB per SM and no overhead, shared memory admits floor(96/96) = 1 block per SM.
If one-buffer staging were valid, its 48 KiB footprint would admit two blocks under the same assumptions.
Double buffering may overlap transfer and compute, but no speedup follows without timing and correct completion handling.
Actual allocation rounding, pipeline metadata, per-block limits, registers and other restrictions are excluded.
A production budget must include model weights, activations, workspace, allocator fragmentation, graph capture, concurrency and safety headroom.
For economic comparison report cost per successful workload outcome, not GPU-seconds alone, and state currency, date, utilization and excluded charges.
**Takeaway:** A larger pipeline can improve overlap while reducing residency; measure the net effect under the real memory budget.

## 10. Key Theorems / Results Reference

| Result | Conditions | What follows | What does not follow |
|---|---|---|---|
| Integer capacity bound | Identical blocks, known resource consumption | At most floor(capacity/consumption) blocks fit | Actual allocation ignores no other resource. |
| Minimum of upper bounds | Each bound applies to the same launch | The tightest bound constrains combined residency | The bound guarantees issue throughput. |
| Occupancy definition | Active and maximum warp counts use same SM configuration | Their ratio describes resident-warp fraction | A higher ratio guarantees higher performance. |
| Asynchronous completion | Correct documented completion and visibility protocol | A consumer can use completed data within the documented scope | Issuing a transfer alone guarantees data are ready. |

The first two rows are elementary derived bounds, not empirical H100 measurements.
CUDA architecture and synchronization details are version-specific; consult the current installed toolkit documentation [S1,S2].
**Takeaway:** Mathematical resource bounds and documented execution guarantees are conditional, not performance predictions.

## 11. Interview / Exam Questions

All interview scenarios are constructed; none are attributed to an employer.
Scoring for each: 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct with a defensible tradeoff or verification plan.

### Level 1 — Definition / Recall

**Q1. What is an SM, and is it the same as a GPC?**
Answer: An SM is a streaming multiprocessor that hosts resident thread blocks and executes warp instructions; a GPC is a broader grouping of processing resources.
A block's per-SM resource requirements do not become a per-GPC register pool.
Trap: treating all SM resources inside a GPC as fungible per-block capacity.
Source: Section 3 and [S1].

**Q2. Distinguish active, eligible and issued warps.**
Answer: Active means resident; eligible means ready to issue under scheduler constraints; issued means an instruction was selected for dispatch.
A memory-stalled active warp may not be eligible.
Trap: asserting that high active occupancy proves high issue efficiency.
Source: Sections 2–3.

**Q3. What distinguishes HBM, L2, shared memory and registers?**
Answer: HBM is device memory, L2 is a cache, shared memory is explicit on-chip block-oriented storage, and registers serve thread-local values.
Scope, ownership, allocation and access behavior differ.
Trap: assuming every shared-memory access reaches HBM.
Source: [S1,S2].

### Level 2 — Derivation / Apply

**Q4. Derive the register-only resident-block bound.**
Answer: If n identical blocks each use rT registers and the pool contains R, then nrT ≤ R; hence n ≤ floor(R/(rT)).
The derivation assumes exact allocation and no other constraint; actual allocation granularity must be considered.
Trap: rounding the quotient upward or treating the register-only answer as full occupancy.
Source: E1.

**Q5. Compute the source-plan example.**
Answer: 256 × 64 = 16,384 registers per block; 65,536 / 16,384 = 4 register-limited blocks.
Shared memory permits 96 / 48 = 2 blocks; the combined simplified bound is min(4,2) = 2.
Check: three blocks require 144 KiB, exceeding the illustrative 96 KiB.
Trap: labeling these hypothetical capacities as actual H100 specifications.
Source: Section 5.

**Q6. What happens if the illustrative block uses 96 KiB shared memory?**
Answer: Under the unchanged 96 KiB assumed SM limit and no overhead, the shared bound becomes floor(96/96) = 1 block.
This is a resource bound, not a claim that one block achieves better overlap or throughput.
Trap: ignoring per-block feasibility and actual allocation rounding.
Source: E3 and Section 9.

**Q7. What does a 50% theoretical occupancy mean?**
Answer: Under the particular launch and hardware configuration, modeled resident warps are half the modeled maximum resident warps.
It does not mean half of instructions issue, half of time is useful, or half of peak throughput is achieved.
Trap: using it as a speedup prediction without checking actual issue and stalls.
Source: Section 3.

### Level 3 — Judgement / Design / Debug

**Q8. A change raises occupancy but slows the kernel. Explain.**
Answer: Lower register limits can cause spills or more instructions; smaller tiles can reduce reuse; extra resident warps may not resolve the actual bottleneck.
Compare compiler resource reports, local-memory traffic, instruction counts and repeated unprofiled elapsed time.
Trap: reverting solely because occupancy rose; identify the mechanism and preserve correctness.
Source: F02–F03.

**Q9. How do you diagnose a stale TMA tile?**
Answer: Check tensor-map dimensions and addressing, producer completion, barrier initialization and phase, consumer wait, and buffer reuse ordering.
Use a tiny deterministic tensor and compare every element to a CPU or trusted GPU reference.
Trap: assuming asynchronous transfer initiation means data are already visible to all consumers.
Source: [S2] and F04.

**Q10. Why can two H100 installations have different results?**
Answer: SKU, HBM configuration, host and link topology, clocks, cooling, partitioning, software, precision and workload shape can differ.
Capture the exact environment before comparing, and use repeated same-workload tests.
Trap: using a marketing peak as a measured application result.
Source: [S3].

**Q11. How would you select between direct loads and staged tensor transfer?**
Answer: Start with a correct direct-load baseline; verify supported tensor map, alignment and synchronization; benchmark representative tile shapes and reuse.
Prefer the candidate only if the production metric improves without breaking correctness or resource budgets.
Trap: assuming a newer mechanism automatically wins on small or irregular transfers.
Source: Section 7.

**Q12. An HBM counter falls but latency rises. What next?**
Answer: Inspect L2 traffic, extra instructions, synchronization, occupancy and profiler perturbation; a reduction in one counter is not a complete causal explanation.
Repeat an unprofiled run and compare the same input distribution and software configuration.
Trap: declaring victory from bandwidth alone.
Source: Sections 7–8.

**Q13. How would you check whether NVLink is the limiting factor?**
Answer: Record topology and peer access, measure relevant directional transfers in isolation and during the real collective, and compare link activity with application timing.
Also examine synchronization, compute imbalance and host staging before attributing the bottleneck to the fabric.
Trap: summing advertised bidirectional rates as though they were achieved one-way application bandwidth.
Source: [S3] and F06.

**Q14. What evidence is required to claim a Tensor Core optimization is safe?**
Answer: Show supported instruction and datatype, matching input/output layout, numerical error across representative and edge-case inputs, race-free execution, and repeated performance under the deployment configuration.
A higher precision fallback may be appropriate where the error budget is exceeded.
Trap: equating successful execution with acceptable model accuracy.
Source: [S1,S2] and F05.

**Q15. How do you handle an ambiguous production regression?**
Answer: Preserve release and hardware metadata, isolate workload changes from platform changes, establish a baseline, then vary one plausible cause at a time.
Mitigate service impact with a safe rollback where appropriate, and confirm both correctness and the service metric after the intervention.
Trap: changing register caps, clocks and batch size together and attributing the result to one of them.
Source: F10–F11.

### Primary Sources and Reproducibility Record

- [S1] NVIDIA, [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/), CUDA documentation; architecture and CUDA tuning, version is the live documentation version; review trigger: CUDA release or GPU architecture change.
- [S2] NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), CUDA programming semantics and asynchronous operations; live version; use documentation matching the installed toolkit.
- [S3] NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/), product overview; model-form-factor and interconnect context, not a measurement of a particular installation.
- [S4] NVIDIA, [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), metric and profiling interpretation; definitions vary by release and GPU.
- [S5] NVIDIA, [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/), compiler binary inspection; match tool version to generated binary.
- [S6] NVIDIA, [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html), supported correctness checking; consult version-specific coverage and limitations.

No external benchmark has been reproduced in this sheet.
Source links identify primary documentation; exact toolkit and document versions must be captured at use time.
Review status is open; this draft is not certified for deployment or safety-critical decisions.
Record GPU SKU and UUID, node topology, partitions, driver, toolkit, compiler flags, binary hash, clock and power settings, workload shapes, dtype, warm-up and repetitions.
Collect both unprofiled timing and profiler output, and retain error tolerances and reference results with the experiment.
Profiling metric names and availability vary by tool and GPU version; do not hard-code a counter name as universal.
**Takeaway:** Defensible answers separate arithmetic, documented mechanism, and measured behavior, and preserve enough context to reproduce each.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | H100 architecture and resource-constrained kernel placement |
| Core formula | Resident blocks are bounded by the minimum of all applicable, correctly rounded per-SM resource limits. |
| Key assumption | Resource allocations and available capacities refer to the same actual kernel, configuration and device. |
| Complexity | For an elementwise pass over N items, O(N) work and modeled O(N) data movement; actual traffic depends on caching. |
| Use when | Diagnosing placement constraints or deciding which architecture signal to profile next. |
| Avoid when | Exact allocation, launch feasibility, correctness or topology is unknown and a firm deployment claim is needed. |
| Go deeper | [Roofline arithmetic intensity and kernel diagnosis](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md); full TMA and multi-GPU instruction guides are not yet written in this collection. |
