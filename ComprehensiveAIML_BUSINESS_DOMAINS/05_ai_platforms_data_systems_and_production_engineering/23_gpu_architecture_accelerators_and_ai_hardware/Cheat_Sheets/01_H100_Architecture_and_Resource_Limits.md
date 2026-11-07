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

### Architecture and resource clinic: versioned vendor evidence

The following vendor statements are **SOURCED**, not measurements by this author.
The reference is NVIDIA's [Hopper Tuning Guide, CUDA 12.4.1 archive, revision 1.0](https://docs.nvidia.com/cuda/archive/12.4.1/hopper-tuning-guide/index.html), sections 1.4.1–1.4.3 [S7].
This versioned source is historical documentation, not a promise that every H100 deployment has identical usable resources.
Query the installed device and match documentation to the actual toolkit before making a launch decision.
The guide uses KB for shared-memory capacities; binary KiB is used in the exercises below.

#### SM capacity versus admission

The guide states that compute capability 9.0 supports a maximum of 64 concurrent warps per SM.
It reports 64K 32-bit registers per SM and up to 255 registers per thread.
It also reports a maximum of 32 resident thread blocks per SM.
Those are separate ceilings: reaching one does not mean the others have been reached.
The guide lists 228 KB of shared memory per SM and a maximum 227 KB per block.
Its 1 KB per-block CUDA reservation helps explain why per-block addressable shared memory is smaller than the SM maximum.
The shared-memory limit available for a particular launch also depends on the selected configuration and kernel.
The documented static shared-memory limit is 48 KB for architectural compatibility.
Dynamic allocations above that limit require explicit opt-in under the documented rules.
An opt-in request is not evidence that a particular launch will fit or succeed.
The guide lists selectable shared-memory carveout capacities, from 0 through 228 KB.
A requested preferred carveout is not a substitute for inspecting the effective configuration.
Increasing the shared-memory portion changes the capacity available for the unified L1/shared-memory arrangement.
Whether a larger carveout helps depends on the workload's shared-memory need and cache behavior.
For thread-block clusters, the guide recommends `cudaOccupancyMaxActiveClusters`.
Ordinary block occupancy alone does not account for cluster placement.

#### Comparing Hopper and A100 without manufacturing a benchmark

The same versioned guide compares H100's 228 KB shared-memory capacity per SM with A100's 164 KB.
The arithmetic difference is 64 KB per SM under those documented figures.
The ratio is 228/164, approximately 1.3902, or approximately 39.0% more.
That is a capacity comparison, not a prediction that a kernel is 39% faster.
The guide compares the combined H100 L1/texture/shared capacity of 256 KB with 192 KB for A100.
The arithmetic difference is 64 KB, and 256/192 is approximately 1.333.
The guide describes 50 MB H100 L2 versus 40 MB A100 L2.
That ratio is 1.25; it does not imply a 25% larger hit rate.
Actual cache usefulness depends on reuse distance, traffic, competing workloads and policy.
The guide reports twice the FP32 operations per cycle per SM for compute capability 9.0 versus 8.0.
This comparison is per cycle per SM, not per device and not an application benchmark.
Device SM counts, sustained clocks, instruction mix and dependencies must be considered separately.
The same guide describes H100 support for HBM3 and HBM2e, depending on product.
It describes an HBM3 bandwidth of up to 3 TB/s in its historical product context.
An “up to” figure is neither the specification of every H100 SKU nor a measured workload rate.
The guide compares fourth-generation NVLink's aggregate bidirectional 900 GB/s with A100's 600 GB/s.
Those are stated aggregate bidirectional product figures, not a one-way bandwidth guarantee between a selected pair.
NVSwitch routing, link connectivity, partitions and other traffic alter the applicable path.
The NVLink comparison must not be substituted for a PCIe bandwidth figure.

#### Architectural mechanisms and non-equivalences

The guide describes TMA as a Hopper extension of asynchronous copies available in Ampere.
It describes transfers of one- through five-dimensional tensors between global and shared memory.
It also describes supported shared-memory transfers between SMs in a cluster.
The relevant copy form, layout restrictions and synchronization rules depend on the actual API and toolkit.
Avoid equating a TMA transfer's issuance with completion or visibility to consumers.
The guide states that the portable maximum cluster size is eight blocks.
It notes an H100-specific nonportable size of 16 when the documented opt-in is enabled.
That nonportable capability is not a portability promise to other devices or partition configurations.
Distributed shared memory permits cluster blocks to access one another's shared-memory regions under its programming contract.
It is not an ordinary device-wide shared address space.
Each block's shared-memory allocation remains associated with that block.
Remote access therefore needs correct peer-block lifetime and cluster synchronization.
The guide describes inline compression for eligible allocations through the driver API.
Compression can reduce physical data transfer when the contents are compressible.
It does not reduce the allocation's reserved footprint according to the guide.
Compression support must be queried and requested; do not assume all allocations use it.
Compare physical traffic and application-visible bytes only when their definitions are clear.

#### Evidence audit before using a number

Identify whether a statement is a device maximum, an API constraint, a measured counter or an assumption.
Identify whether its units are bytes, binary multiples, decimal multiples, instructions or operations.
Confirm whether the source describes one SM, one GPU, one direction, or an entire fabric.
Record the source version, section, device SKU, and any relevant opt-in or configuration.
Separate historical A100/H100 architectural comparisons from present system availability.
Do not apply an H100 guide's aggregate number to a MIG instance without checking that instance.
Do not treat a source's marketing maximum as a measured ceiling for an individual kernel.
Report vendor figures and the conditions under which the proposed deployment differs.

### Numerical laboratory: independent resource and performance questions

All cases in this laboratory are **ILLUSTRATIVE** and arithmetically checked.
Their inputs are constructed unless explicitly identified as values taken from [S7].
They do not claim a measured H100 execution, validated kernel, price or deployment result.
Every resource calculation is an upper bound subject to the explicitly omitted hardware constraints.
For these examples, 1 KiB means 1,024 bytes and 1 MiB means 1,024 KiB.

#### Case A — partial warps and block count

Assume a 70-thread block, a 32-thread warp, and a modeled limit of 64 resident warps per SM.
The block uses ceil(70/32) = 3 warp slots.
Two of the three warps have 32 threads, and the third has six participating threads.
One block therefore has 70/96, approximately 72.92%, populated thread slots.
Assume all other resource limits allow 20 such blocks.
Twenty blocks require 20 × 3 = 60 resident warp slots.
The corresponding modeled occupancy is 60/64 = 93.75%.
Twenty-one blocks would require 63 warp slots, but this example caps residency at 20 for another stated reason.
The number of participating threads is 20 × 70 = 1,400.
The number of represented thread slots is 20 × 96 = 1,920.
High warp occupancy can therefore coexist with partially populated warps.
The arithmetic does not measure lane utilization, divergence or actual issue efficiency.
Boundary: if the hardware's thread or allocation limit is tighter, fewer blocks can reside.

#### Case B — rounded register allocation changes a conclusion

Assume a hypothetical register pool of 65,536 32-bit registers.
Assume a 256-thread block whose compiler report states 33 registers per thread.
Without rounding, one would estimate 256 × 33 = 8,448 registers per block.
That naive estimate gives floor(65,536/8,448) = 7 blocks.
Now assume, only for this exercise, an actual allocation of 9,216 registers per block after rounding.
The modeled rounded bound becomes floor(65,536/9,216) = 7 blocks.
Seven blocks consume 64,512 registers and leave 1,024.
Eight blocks would consume 73,728 and do not fit.
Change only the hypothetical rounded allocation to 9,472 registers per block.
The bound then falls to floor(65,536/9,472) = 6 blocks.
Six blocks consume 56,832 registers; seven require 66,304.
This comparison shows a discrete admission cliff rather than a smooth proportional change.
The invented rounded allocations are not a claim about H100 allocation granularity.
Obtain the real allocation behavior from the matching toolchain and documentation.

#### Case C — static plus dynamic shared memory

Assume a modeled 128 KiB shared-memory budget per SM.
The block has 12 KiB static shared memory and requests 28 KiB dynamically.
For this exercise, assume no reservation, rounding or other allocation overhead.
The combined footprint is 12 + 28 = 40 KiB per block.
The shared-only bound is floor(128/40) = 3 blocks.
Three blocks use 120 KiB; a fourth would need 160 KiB.
If dynamic shared memory rises to 36 KiB, the footprint becomes 48 KiB.
The bound becomes floor(128/48) = 2 blocks.
Two blocks use 96 KiB; three require 144 KiB.
Comparing dynamic requests alone would miss the static component.
Per-block feasibility, real reservations and actual carveout remain separate checks.
Do not treat the 128 KiB budget as a documented H100 carveout.

#### Case D — when a register cap introduces traffic

Assume a baseline moves 800 MiB across a particular measured boundary per invocation.
Assume the capped variant moves 1,000 MiB at the same boundary.
The increase is 200 MiB, or 200/800 = 25%.
Suppose both perform the same logical work and the additional traffic is confirmed to be spill-related.
At an illustrative effective 1,000 MiB/ms boundary bandwidth, the extra traffic alone has a 0.2 ms service-time equivalent.
That is 200 MiB divided by 1,000 MiB/ms.
It is not necessarily 0.2 ms of additional elapsed time because traffic can overlap other work.
Suppose unprofiled latency rises from 1.6 ms to 1.9 ms.
The absolute difference is 0.3 ms and the relative increase is 0.3/1.6 = 18.75%.
If the extra traffic is local-memory traffic but hits L2, it need not all cross the HBM boundary.
Measure the same counter boundary in both variants.
Compiler spill reports plus traffic and controlled reversal provide stronger evidence than occupancy alone.

#### Case E — waves and underfilled final wave

Assume 80 SMs, two resident blocks per SM and a grid of 350 blocks.
The idealized device-wide simultaneous capacity is 80 × 2 = 160 blocks.
The idealized minimum number of waves is ceil(350/160) = 3.
The first two full waves contain 320 blocks in total.
The remaining wave contains 350 − 320 = 30 blocks.
Its fraction of the 160-block capacity is 30/160 = 18.75%.
This is a scheduling-capacity sketch, not a guarantee of synchronized waves.
Blocks can have different durations and can be admitted as resources become free.
If the grid instead has 320 blocks, the same idealized model has two full waves.
That does not prove 320 blocks finish sooner if each block performs more work.
Wave estimates are useful for recognizing small-grid underfill, not predicting latency exactly.
Cluster constraints and partitioning can invalidate the assumed device-wide capacity.

#### Case F — tensor tile storage

Assume a matrix multiply stages one 64 × 32 input tile and one 32 × 64 input tile.
Assume both inputs use two bytes per element.
The first tile contains 2,048 elements and requires 4,096 bytes.
The second tile also requires 4,096 bytes.
One pair of staged tiles therefore requires 8,192 bytes, or 8 KiB.
Two staged pairs require 16 KiB before padding, swizzling, barriers or metadata.
At a hypothetical 96 KiB SM budget, the shared-only double-buffer bound is floor(96/16) = 6 blocks.
The single-buffer bound is floor(96/8) = 12 blocks.
These are only resource bounds; registers, warps and block limits can be tighter.
If double buffering saves waits but doubles staging, measure whether overlap offsets reduced residency.
Accumulator storage is not included in this shared-memory estimate.
The arithmetic assumes dense tiles; edge masking and packed layouts can change the actual footprint.

#### Case G — arithmetic intensity at two boundaries

Assume an operation performs 12 billion floating-point operations.
At the HBM boundary, suppose measured traffic is 3 billion bytes.
Its HBM arithmetic intensity is 12/3 = 4 FLOP/byte.
At a cache boundary, suppose traffic is 12 billion bytes.
Its cache-boundary arithmetic intensity is 12/12 = 1 FLOP/byte.
The same operation can therefore have two different correct intensities.
Assume compatible illustrative bandwidth ceilings of 2 TB/s at HBM and 8 TB/s at the cache boundary.
Both products produce 8 TFLOP/s: 4 × 2 and 1 × 8.
Assume a compatible compute ceiling of 20 TFLOP/s.
The combined modeled roof is min(8, 8, 20) = 8 TFLOP/s.
This is an upper-bound model under the chosen traffic and ceilings.
Mixing 4 FLOP/byte with the 8 TB/s cache ceiling would incorrectly produce 32 TFLOP/s.
State whether FLOPs include multiply-add as two operations and keep the convention consistent.
If a cache or bandwidth counter includes unrelated traffic, the computed intensity is not kernel-specific.

#### Case H — transfer size, direction and rate

Assume a one-way payload of 4 GiB and an elapsed transfer time of 40 ms.
Using binary units, the rate is 4/0.040 = 100 GiB/s.
The corresponding decimal payload is 4 × 1,073,741,824 bytes.
Dividing by 0.040 seconds gives approximately 107.37 GB/s.
This is a constructed calculation, not an H100 NVLink measurement.
If the reverse direction simultaneously transfers the same payload in the same interval, the summed bidirectional rate is 200 GiB/s.
The individual directional rate is still 100 GiB/s.
Do not compare the one-way value directly to a summed bidirectional headline.
Synchronization, measurement scope and whether both directions overlap must be reported.
An application may have smaller messages and dependencies that make its effective rate lower.

#### Case I — matrix arithmetic and numerical tolerance

Assume dense matrices A with shape 128 × 64 and B with shape 64 × 256.
The output has shape 128 × 256, or 32,768 elements.
Under the convention of two FLOPs per multiply-add, the modeled work is 2 × 128 × 64 × 256.
The resulting count is 4,194,304 FLOPs.
This convention ignores the small difference between K multiplications and K − 1 additions per exact dot product.
Suppose a reference value is 0.002 and a candidate returns 0.0023.
The absolute error is 0.0003.
The relative error against the nonzero reference is 0.0003/0.002 = 15%.
At a reference of zero, this relative-error definition is undefined.
Use an application-approved absolute and relative tolerance and document the zero-reference rule.
The worked values do not justify accepting or rejecting any particular model.

#### Case J — Amdahl-style kernel optimization

Assume a service stage takes 10 ms, of which the target kernel takes 4 ms.
Other work takes 6 ms and is assumed unchanged.
Suppose an optimization halves only that kernel's execution time.
The new total is 6 + 4/2 = 8 ms.
The stage speedup is 10/8 = 1.25 times, not two times.
The latency reduction is (10 − 8)/10 = 20%.
If batching, transfer overlap or queueing changes, this simple additive model is invalid.
An isolated kernel result must be checked against the actual end-to-end metric.
The 10 ms and 4 ms inputs are illustrative, not observed production measurements.

#### Case K — GPU memory budget

Assume an illustrative usable device-memory budget of 70 GiB.
Weights occupy 32 GiB, persistent runtime allocations 4 GiB, and workspace 8 GiB.
The fixed subtotal is 32 + 4 + 8 = 44 GiB.
Reserve 10 GiB for fragmentation, transients and operational headroom.
The remaining budget for concurrent requests is 70 − 44 − 10 = 16 GiB.
If each request consumes 0.5 GiB under the same conditions, the arithmetic ceiling is 32 requests.
At 0.75 GiB per request, floor(16/0.75) = 21 requests.
Twenty-one requests use 15.75 GiB; twenty-two use 16.5 GiB.
This capacity arithmetic does not establish acceptable latency or safe request admission.
The real service needs measured peaks, variable sequence lengths and allocator behavior.
MIG partitions and other processes change the available memory budget.
The assumed 70 GiB is not a statement about the usable memory of any specific H100.

#### Case L — cost per successful result

Assume a device rental rate of 6 currency units per hour and 3,600 successful requests per hour.
The device-only cost per successful request is 6/3,600 = 0.001667 currency units.
Suppose a faster variant completes 4,000 attempts per hour but 200 fail the quality contract.
Its successful rate is 3,800 per hour.
At the same illustrative device rate, device-only cost per success is 6/3,800, approximately 0.001579.
The improvement is approximately 5.26% relative to the baseline cost per success.
This is not a real cloud price, a benchmark, or a full cost of ownership.
Include host, network, storage, idle time and retries before making an economic decision.
Quality failures can have costs beyond the device-time arithmetic.

### Profiling field guide: from launch to causality

This is an operational recipe, not a claim that these commands were executed here.
Use the installed tool's help and matching documentation because options and counters change by release.
Obtain permission before profiling a shared production system or accessing customer inputs.
Sanitize traces and filenames if workload metadata is sensitive.

#### 1. Establish the experiment

Write the expected input, output, tolerance and service-level decision before changing a kernel.
Select representative shapes plus small, boundary, sparse and unusually large cases.
Record the exact GPU model, UUID, partition profile, driver and runtime.
Record toolkit, compiler, flags, binary hash and library versions.
Record host CPU placement, NUMA topology, device pairings and competing workloads.
Retain the dataset generation seed or a privacy-safe reproducible input.
Separate warm-up from timed trials, and record how each is performed.
Measure the end-to-end service path as well as the isolated kernel.
Keep a rollback-capable baseline and define the acceptance threshold in advance.
Treat thermal and power behavior as experimental variables rather than invisible constants.

#### 2. Inspect launch and compilation

Compile with the toolchain's supported verbose resource reporting, such as `nvcc --ptxas-options=-v` when applicable.
Capture the compiler version, target architecture and full command with its output.
Read register count, static shared memory and any spill report for the particular kernel.
Record runtime dynamic shared-memory bytes separately from compiler-reported static allocation.
Confirm the grid, block dimensions and, if used, cluster dimensions.
Check the returned launch error and the later synchronization error at appropriate boundaries.
Remember that a successful launch does not establish numerical correctness.
Use device and function attribute queries to discover limits relevant to the actual process.
Do not infer occupancy from source-level local variables.
The compiler may specialize, inline or eliminate work under different build flags.

#### 3. Establish correct timing

Use a synchronization method appropriate to the work being timed.
For device execution on a stream, CUDA events can bound work ordered in that stream.
Include any cross-stream dependencies required to represent the operation.
Host wall-clock timing is appropriate for a broader host-to-result path when synchronization is explicit.
Do not mix launch-only host timing with completed-kernel device timing.
Run enough independent repetitions to describe variability rather than reporting only the best run.
Report median and a relevant tail percentile with sample count and environment.
Randomize or interleave variants when drift could favor the order of execution.
Investigate outliers instead of deleting them without a predeclared rule.
Profiling runs are diagnostic; keep a separate unprofiled baseline for the performance decision.

#### 4. Inspect scheduling and occupancy

Use Nsight Compute's version-matched occupancy and scheduler sections [S4].
Distinguish theoretical occupancy, achieved active warps and eligible warps.
Inspect the reported limiting resources rather than guessing from a single percentage.
Low achieved occupancy with adequate theoretical capacity can indicate a small grid, short duration or uneven work.
High active warps with few eligible warps suggests investigating dependencies or waiting, not automatically adding blocks.
High eligibility with poor issue behavior requires examining scheduler and execution-pipeline constraints.
Counter definitions, sampling and scope must match the version being used.
For clusters, use the cluster-aware occupancy API and consider cluster scheduling granularity.
Compare the profile against unprofiled runtime to identify profiler perturbation.

#### 5. Inspect memory at the correct boundary

Start with a traffic hypothesis: redundant loads, poor coalescing, spills or weak reuse.
Inspect the memory workload sections for the installed profiler and GPU.
Compare requested bytes and transferred sectors only using the metric definitions for that release.
A high sector-to-request ratio can signal waste, but alignment and access shape matter.
Separate local-memory traffic from explicit shared-memory staging and ordinary global loads.
Inspect cache and device-memory traffic separately.
If traffic is small but latency remains high, investigate dependencies and instruction throughput.
If the cache hit rate changes between candidates, explain why HBM bytes may differ.
Use a controlled one-change experiment to distinguish layout effects from tile-size effects.
Do not assume a bandwidth counter alone proves a memory bottleneck.

#### 6. Inspect instructions and arithmetic

Use generated-code inspection tools appropriate to the compiled binary [S5].
Compare instruction types, conversions, address arithmetic and branch behavior across candidates.
Determine whether the intended matrix instructions are actually present when this is relevant.
Disassembly is evidence of code generation, not proof that a particular path dominates runtime.
Use profiler instruction and pipeline activity to connect generated code to execution.
Distinguish input precision, intermediate conversion and accumulation precision.
Validate extreme values and cancellation against a suitable reference.
If the input workload varies in shape, inspect each materially different execution path.
Do not compare a sparse peak with dense work unless the operation and counting convention match.

#### 7. Synchronization and correctness

Use Compute Sanitizer tools only within their documented supported coverage [S6].
Select the relevant memory, race, initialization or synchronization checks for the issue.
Instrumentation can alter scheduling and may not expose every race or unsupported instruction path.
Review producer-to-consumer ownership and barrier scope even when a tool reports no error.
For asynchronous copies, check initialization, arrival, completion, wait and buffer reuse.
For cluster communication, check that remote shared-memory access completes before the owning block exits.
For cross-stream work, record the event or other dependency connecting producer and consumer.
Use edge sizes that cross tile and alignment boundaries.
Compare results repeatedly under representative concurrency.
Correctness precedes latency comparison.

#### 8. Identify the causal lever

Write the predicted observation for each competing explanation before running a variant.
For register pressure, expect a change in allocation or spills when the live range changes.
For shared capacity, expect a change in the corresponding occupancy limit when the footprint changes.
For memory reuse, expect boundary-specific traffic changes when reuse is improved.
For launch underfill, expect sensitivity to grid size or work partitioning at fixed per-item work.
For synchronization stalls, expect a change in wait behavior when a correct pipeline is adjusted.
Reject an explanation when its predicted signal is absent under a valid test.
One correlated counter is not sufficient to establish a causal mechanism.
If multiple changes are unavoidable, document the confounding and run a targeted follow-up.
Keep both failed and successful experiments to avoid selecting only favorable noise.

#### 9. Production deployment decision

Roll out a correct candidate gradually using the established release process.
Track successful throughput, end-to-end latency, numerical quality and error rate together.
Track device memory headroom and relevant clock, power, thermal and hardware-error signals.
Inspect workload shape drift and co-tenant changes after deployment.
Compare against a concurrent control when practical, not only a historical best run.
Define a rollback trigger that does not require interpreting a single noisy sample.
Retain the previous binary and its compatible runtime configuration.
Escalate hardware faults and recurring data corruption to the responsible platform owner.
Repeat qualification after driver, toolkit, SKU, topology or model changes.
An optimization is not approved solely because a profiler reports a larger percentage.

### Additional interview practice: reasoning under constraints

**Q16. A 70-thread block runs on 32-thread warps. How many warps are allocated?**
Answer: Three warp slots are needed because ceil(70/32) = 3.
There are 26 unpopulated thread slots in the final modeled group of 96.
The result says nothing by itself about branch divergence among the 70 active threads.
Trap: using floor(70/32) and losing the final partial warp.

**Q17. Can a shared-memory carveout increase while performance falls?**
Answer: Yes; the candidate can gain shared capacity while changing cache capacity or increasing staging.
Compare the actual launch configuration, cache traffic, occupancy and unprofiled latency.
Trap: treating the carveout as free extra on-chip memory.

**Q18. Why does a 50 MB L2 not imply every 45 MB workload hits in cache?**
Answer: Access patterns, competing traffic, replacement, metadata and configuration affect effective reuse.
Measure hit behavior at the relevant boundary with a representative workload.
Trap: treating nominal cache size as a guaranteed per-kernel reservation.

**Q19. Why is 900 GB/s bidirectional not automatically a 900 GB/s one-way pairwise rate?**
Answer: The quoted [S7] number aggregates directions and links for a product configuration.
The chosen device pair may use only a subset of links or a shared path.
Trap: comparing an aggregate specification to a single-direction transfer measurement.

**Q20. What does a register spill mean for an HBM counter?**
Answer: Spilled values use local-memory accesses, but those accesses can interact with caches.
Some may reach HBM and some may not; inspect compiler evidence and relevant traffic counters.
Trap: equating each spilled byte with one observed HBM byte.

**Q21. What is the benefit and cost of double buffering?**
Answer: It can overlap movement with computation when completion and reuse are correctly synchronized.
It also increases storage and can lower residency or raise setup overhead.
Compare output and unprofiled time at representative sizes.
Trap: claiming double buffering is always faster.

**Q22. How does a cluster change ordinary occupancy reasoning?**
Answer: Blocks have cluster-level placement and lifetime constraints, so independent per-SM block bounds are incomplete.
Use the applicable cluster-aware occupancy API and check the requested cluster size and portability.
Trap: multiplying a per-SM estimate by the number of SMs as a guaranteed cluster count.

**Q23. Why can a smaller tile improve occupancy but worsen execution time?**
Answer: It can reduce shared or register pressure while increasing redundant loads, instructions or synchronization.
Verify correctness and compare traffic, issue behavior and repeated timings.
Trap: choosing a tile solely by theoretical occupancy.

**Q24. How would you compare H100 with A100 fairly?**
Answer: Match workload, dtype, correctness, software, power policy and measurement boundary.
Report exact SKUs, costs, device counts and relevant topology.
Per-SM architectural differences are not whole-system benchmark results.
Trap: extrapolating the documented per-cycle FP32 ratio directly to application speedup.

**Q25. A 350-block grid runs on an illustrative 80-SM device with two blocks per SM. How many idealized waves?**
Answer: ceil(350/(80 × 2)) = 3, with 30 blocks remaining after two full capacity waves.
Actual scheduling can overlap waves and uneven block runtimes affect the timeline.
Trap: asserting that the last 30 blocks all start simultaneously.

**Q26. A kernel moves 3 billion HBM bytes for 12 billion FLOPs. What is its intensity?**
Answer: 4 FLOP/byte at the specified HBM boundary.
Use an HBM-compatible bandwidth ceiling and the same FLOP convention.
Trap: substituting L2 bytes or a different-precision peak without relabeling the calculation.

**Q27. How should a team react to a profiler-only speedup?**
Answer: Repeat the comparison without profiling and examine replay, sampling and clock differences.
Check the production service metric and numerical correctness before rollout.
Trap: assuming instrumented timing equals customer latency.

**Q28. What is required before accepting a lower-precision Tensor Core path?**
Answer: Demonstrate supported instructions, valid layout and acceptable error on representative and edge inputs.
Also test downstream model or business quality, not merely elementwise closeness.
Trap: using a generic relative tolerance near zero.

**Q29. What does inline compression save?**
Answer: For eligible compressible data it can reduce physical traffic according to [S7].
The guide explicitly says it does not shrink the reserved allocation footprint.
Trap: counting compressed traffic savings as extra device-memory capacity.

**Q30. How would you determine whether TMA is appropriate?**
Answer: Verify supported descriptor, layout, synchronization and target architecture first.
Compare a correct ordinary-copy baseline against TMA for representative tile sizes and concurrency.
Trap: assuming one issuing thread eliminates all setup, barrier or resource costs.

**Q31. A service spends 4 ms of 10 ms in the optimized kernel; the kernel becomes twice as fast. What is the total speedup?**
Answer: With all other work unchanged and additive, total time falls to 8 ms.
The total speedup is 10/8 = 1.25 times.
Trap: promising a twofold service speedup from a twofold kernel speedup.

**Q32. What does an occupancy estimate fail to prove?**
Answer: It does not prove launch correctness, eligible-warp availability, throughput or numerical accuracy.
It is an admission prediction under a particular kernel and device configuration.
Trap: reporting theoretical occupancy as observed utilization.

**Q33. Why can two H100 partitions differ?**
Answer: Their instance profiles and resource allocations may differ, and contention or workload can also differ.
Record and query each partition rather than applying full-device figures blindly.
Trap: treating identical product names as identical available capacity.

**Q34. How do you verify an optimization that changes output only for edge tiles?**
Answer: Check bounds, masks, strides and padding against a trusted reference on sizes around tile boundaries.
Then check numerical precision if the indexing is correct.
Trap: immediately widening tolerances to hide an address error.

**Q35. When should a kernel improvement be rejected despite better throughput?**
Answer: Reject it when correctness, security, latency objectives or memory headroom fail the deployment contract.
Compare cost per successful result, not raw attempts, and preserve rollback capability.
Trap: accepting a fast result outside the application's quality budget.

### Resource-accounting workshop: allocation and feasibility

All capacities and timings in this workshop are **ILLUSTRATIVE**, not H100 specifications or measured outcomes.
The calculations use binary KiB, where 1 KiB is 1,024 bytes.
They describe a single homogeneous SM unless a different scope is explicitly stated.
Actual allocations, legal launches and placement must be checked with the target device and toolkit.

**Case A — register allocation rounds up.**
Suppose a fictional SM has 65,536 registers available to resident blocks.
Each block has 256 threads, or eight 32-thread warps.
Suppose the compiler reports 65 registers per thread.
Assume, for this example only, that each warp's register allocation rounds up to a multiple of 256 registers.
The unrounded allocation per warp is 32 × 65 = 2,080 registers.
Rounding 2,080 up to the next multiple of 256 gives 2,304 registers per warp.
Eight warps therefore consume 8 × 2,304 = 18,432 registers per block.
The register bound is floor(65,536 / 18,432) = 3 blocks.
Three blocks use 55,296 registers, leaving 10,240 registers unused.
Four blocks would need 73,728 registers and do not fit this register pool.
The naïve unrounded estimate is floor(65,536 / 16,640) = 3 blocks here.
The matching result is coincidental; it does not establish that rounding can be ignored.
At 64 registers per thread the assumed rounded allocation is 16,384 registers per block.
The corresponding bound is four blocks, assuming all other limits admit four.
One additional register per thread therefore crosses a capacity threshold in this constructed case.
This example does not assert that the assumed register granularity applies to H100.
Consult the installed occupancy calculator and compiled-kernel information for the real allocation.

**Case B — shared-memory allocation rounds up.**
Assume a fictional SM provides 100 KiB of shared memory to blocks.
Each block requests 32.1 KiB including static and dynamic allocations.
Assume allocation is rounded up to a 1 KiB unit for this exercise.
Each resident block consumes 33 KiB under that fictional rule.
The shared-memory bound is floor(100 / 33) = 3 blocks.
Four blocks would consume 132 KiB, exceeding the assumed capacity.
If a layout change raises the unrounded request to 33.1 KiB, allocation becomes 34 KiB.
Then floor(100 / 34) = 2 blocks, with 32 KiB left over.
The unused capacity does not allow a third 34 KiB block.
Check per-block maximums separately: an allocation can fit the SM pool yet be illegal for one block.
The requested dynamic shared-memory amount and any opt-in attribute must agree with the launch.
The effective shared/L1 configuration can change available capacity.
The allocation unit in this example is deliberately invented and must not be used as a hardware rule.

**Case C — multiple independent residency limits.**
Assume a block has 192 threads and uses six warps.
Suppose the rounded resource accounting gives a register limit of four blocks.
Suppose shared memory permits three blocks, thread capacity permits five, and warp capacity permits six.
Suppose the architectural block count permits eight.
The resulting independent-resource upper bound is min(4, 3, 5, 6, 8) = 3 blocks per SM.
Three blocks contribute 18 resident warps.
If the fictional maximum is 48 resident warps, theoretical occupancy is 18/48 = 37.5%.
The number 48 is an example assumption, not a device claim.
Reducing register use alone cannot raise this particular bound while shared memory remains the tightest limit.
Reducing shared use to admit four blocks changes the upper bound to four.
Four blocks contribute 24 warps, or 50% of the assumed 48-warp maximum.
The new active-warp fraction is not evidence that execution time improved.
If the kernel depends on special launch placement or clusters, the independent bound may still be unattainable.

**Case D — a partial warp consumes capacity.**
Assume a 100-thread block and a 32-thread warp.
The warp count is ceil(100/32) = 4, not 100/32 rounded down.
The final warp has four participating thread indices and 28 unused lane positions.
With a fictional limit of 32 resident warps, the warp-only bound is floor(32/4) = 8 blocks.
If the same SM admits at most 768 threads, the thread bound is floor(768/100) = 7 blocks.
The combined bound cannot exceed seven blocks before checking registers, shared memory and block limits.
Seven blocks contain 700 requested threads and occupy 28 warp slots.
Their theoretical warp occupancy under the assumed limit is 28/32 = 87.5%.
The fraction of useful lanes in the allocated warp slots is 700/(28 × 32) = 78.125%.
Lane utilization and occupancy describe different denominators and must not be multiplied into a throughput prediction.
Partial warps also do not imply all participating lanes take the same branch.

### Concurrency workshop: grids, streams and cooperative placement

**Case E — waves and a long tail.**
Assume 12 SMs and an independent-block limit of three blocks per SM.
The instantaneous modeled capacity is 12 × 3 = 36 blocks.
For a grid of 100 blocks, ceil(100/36) = 3 idealized capacity waves.
Two full capacity waves account for 72 blocks, leaving 28 blocks.
This is a count of capacity-sized groups, not a promise of a global wave barrier.
If every block took exactly 10 arbitrary time units and all slots stayed busy, the idealized duration would be 30 units.
Unequal block durations, launch overhead and runtime scheduling invalidate that simple duration.
If just one block takes 25 units while others take 10, its placement affects the observed tail.
Measure per-block work imbalance or representative input skew before blaming occupancy.
Do not infer that 28 blocks in the last group occupy every SM uniformly.

**Case F — two kernels share a device.**
Assume a fictional SM has 64 KiB of shared memory and enough registers, warp slots and block slots for the following blocks.
Kernel A requests 40 KiB per block; kernel B requests 24 KiB per block.
One A and one B block sum to 64 KiB and are resource-compatible in this simplified model.
Two A blocks would need 80 KiB and cannot coexist on that SM under the assumption.
Two B blocks need 48 KiB, leaving 16 KiB.
Compatibility is necessary but not sufficient for observed concurrent execution.
Grid arrival, stream dependencies, scheduler decisions and other resource allocations also matter.
Concurrency can increase contention for execution units, cache, memory bandwidth or interconnect.
Separate streams permit ordering independence where the API contract allows it; they do not guarantee simultaneous kernel execution.
Record each stream's dependencies and buffer ownership before interpreting a timeline.
Compare concurrent completion time with the same work serialized under matched conditions.
Also compare individual latency and total successful throughput: they answer different objectives.
Never overlap writes to the same memory unless the program supplies a correct synchronization and ownership protocol.

**Case G — concurrent kernels compete for registers.**
Assume a fictional 65,536-register SM, ignoring rounding and all other limits.
An A block uses 32,768 registers; a B block uses 24,576 registers.
One of each uses 57,344 registers, leaving 8,192.
A second B block would raise consumption to 81,920 registers, which does not fit.
Two A blocks exactly consume the assumed register pool.
Two B blocks consume 49,152 registers and leave 16,384.
This arithmetic establishes possible combinations, not a guarantee that any specific mixture is scheduled.
If either kernel has a large shared-memory footprint, repeat the calculation for that resource as well.
For heterogeneous blocks, do not simply take the minimum of each kernel's standalone block counts.
Instead evaluate the resource sum for each proposed mixture.
On the real system account for allocation units, per-block feasibility and any scheduler or partition restrictions.

**Case H — producer and consumer across streams.**
Suppose stream P launches a kernel that writes buffer X.
Suppose stream C launches a kernel that reads X.
Independent stream order alone does not establish that P's write finishes before C reads.
Record an event after the producer in P and make C wait on that event before launching the consumer.
Use the correct device, event and stream semantics for the installed runtime.
Keep X allocated until all asynchronous uses have completed.
If X is reused for the next request, establish a separate dependency on the previous consumer's completion.
If producer and consumer run in the same stream, their stream order supplies the relevant execution order.
This example is an ordering sketch, not a substitute for checking API-specific memory visibility rules.
Profile the resulting timeline only after validating the output for repeated and overlapping requests.
A fast consumer reading stale data is a correctness failure, not useful concurrency.

**Case I — cluster placement is not ordinary per-SM placement.**
Assume an illustrative four-block cluster and 20 blocks in the grid.
The launch contains five complete clusters if the applicable launch contract permits those dimensions.
An ordinary per-SM estimate of two blocks does not establish how many such clusters can reside concurrently.
Cluster placement must satisfy the architecture's co-scheduling and resource rules.
Use the cluster-aware occupancy interface supported by the installed CUDA version.
Check permitted cluster dimensions and any device-specific or portability restrictions.
If blocks access distributed shared memory, the owner block must remain alive for the required access lifetime.
Use the documented cluster synchronization and completion protocol rather than inventing an inter-block global barrier.
The example says nothing about the cluster size supported by a specific H100 configuration.

### Profiling workshop: distinguish causes with matched evidence

**Experiment 1 — occupancy versus issue readiness.**
Choose a correct baseline and record its binary, launch, input shapes and resource usage.
Create a variant that changes register pressure without changing the mathematical operation.
Use the compiler report and occupancy tooling to confirm whether the limiting resource actually changed.
Inspect eligible-warp and scheduler observations using definitions for the installed profiler.
If modeled occupancy rises but eligible warps do not, investigate dependencies, synchronization and instruction mix.
If eligible warps rise but elapsed time does not improve, inspect execution-pipeline or memory saturation.
If spills appear, inspect their local-memory traffic and whether the data reaches the measured memory boundary.
Keep unprofiled timing separate from instrumented timing.
Repeat across representative shapes; one shape cannot characterize a variable workload.
No particular observation is guaranteed by the proposed change.

**Experiment 2 — HBM traffic versus cache reuse.**
Suppose the operation logically reads 128 MiB and writes 64 MiB.
The minimal logical payload in this model is 192 MiB, excluding all extra operations.
Suppose an illustrative counter reports 256 MiB crossing the selected HBM boundary.
The measured-to-logical ratio is 256/192 = 4/3, approximately 1.33.
This hypothetical ratio can motivate investigation of rereads, writes, access granularity and other traffic.
It does not identify which mechanism caused the difference.
Counter semantics, compression and the capture range can also affect the comparison.
Suppose a second run reports 128 MiB at the same boundary.
That smaller figure could reflect cache reuse or different accounting; it does not prove either explanation.
Compare warm and cold protocols with identical work and explicitly state the cache state.
Use a boundary-compatible bandwidth calculation and report bytes as decimal or binary units consistently.
These numbers are invented teaching inputs, not H100 counter measurements.

**Experiment 3 — asynchronous pipeline depth.**
Construct a correct single-buffer baseline with a documented completion wait.
Construct a double-buffer version with independent producer and consumer ownership for each buffer.
Account for the second buffer, barrier state, alignment and any additional registers.
Verify that each buffer is not overwritten until its previous consumer has finished.
Check correctness on zero-length or minimal cases where supported and on edge tiles.
Compare the shared-memory and register admission limits before measuring time.
Inspect a timeline and relevant wait behavior to test the overlap hypothesis.
If occupancy falls, determine whether additional overlap compensates using unprofiled timing.
If runtime improves only for large tiles, retain shape-specific evidence rather than declaring a universal result.
If runtime worsens, investigate setup and synchronization overhead as competing hypotheses.
Preserve the baseline for rollback and for future toolkit changes.

**Experiment 4 — underfilled grid versus resource pressure.**
For an illustrative 120-SM device with 60 independent blocks, no more than 60 SMs can host a block at once.
That statement assumes each block occupies only one SM and excludes cluster-specific placement.
More per-SM occupancy cannot create more than 60 independent blocks from that grid.
Increasing parallel decomposition may expose more blocks but can add work and coordination.
Compare an equivalent finer decomposition only after validating its output and accounting for overhead.
If the grid instead has 1,200 blocks, block scarcity is less likely to explain the initial launch underfill.
That larger grid still does not guarantee balanced work, high occupancy or sustained throughput.
Check block duration distribution and resource limits to distinguish the alternatives.
The 120-SM device is fictional and is not a specification for any H100 SKU.

**Experiment 5 — instrumentation and replay.**
First collect repeated unprofiled end-to-end and kernel timings with the same workload.
Then collect a limited profiler set sufficient to test the leading hypothesis.
Check whether the profiler replays the kernel or changes the cache state for the requested metrics.
State whether the workload is deterministic and whether replay preserves its inputs and dependencies.
For stateful kernels, ensure replay cannot silently alter the workload or output interpretation.
If metrics conflict with unprofiled behavior, investigate instrumentation and sampling scope before changing code.
Compare timing distributions, not only the fastest individual run.
Record co-tenancy, power and clock conditions as potential confounders.
Do not treat a profiler percentage as a direct customer-facing performance metric.
Choose rollout based on correctness and representative unprofiled service behavior.

### Interview drill: capacity, bandwidth and latency

**Q36. A block uses 20 KiB shared memory and the assumed per-SM budget is 96 KiB. What is the shared bound?**
Answer: floor(96/20) = 4 blocks before allocation rounding and all other constraints.
Four blocks use 80 KiB; five would use 100 KiB and exceed the assumed budget.
Trap: calling the remaining 16 KiB enough for a fifth block.

**Q37. A 160-thread block runs on 32-thread warps; three blocks fit on a fictional 40-warp SM. What occupancy results?**
Answer: Five warps per block yield 15 active warps and 15/40 = 37.5% theoretical occupancy.
The 40-warp maximum is an illustrative premise, not an H100 claim.
Trap: dividing 480 active threads by a warp count rather than a thread count.

**Q38. A modeled kernel performs 24 billion operations and transfers 6 billion bytes across HBM. What intensity results?**
Answer: 24/6 = 4 operations per byte under the stated counting convention.
Compare only with a compute ceiling and HBM bandwidth ceiling that use compatible units and precision.
Trap: using logical source bytes if the stated traffic is measured at HBM.

**Q39. A fictional bandwidth ceiling is 2 TB/s and the intensity is 4 operations per byte. What is the bandwidth roof?**
Answer: 2 TB/s × 4 operations/byte = 8 tera-operations/s.
This decimal-unit result is an upper model under the stated assumptions, not a measured H100 rate.
The effective roof is also constrained by the relevant compute ceiling.
Trap: claiming the kernel necessarily reaches either ceiling.

**Q40. An illustrative request spends 3 ms queued, 2 ms transferring and 5 ms computing, without overlap. What is latency?**
Answer: 3 + 2 + 5 = 10 ms for this explicitly serial request path.
If compute falls to 2.5 ms and all other terms stay fixed, latency is 7.5 ms.
The ratio 10/7.5 is approximately 1.33, not a twofold end-to-end speedup.
Trap: adding durations that actually overlap or ignoring changed queueing under load.

**Q41. Can two streams with no dependency still serialize?**
Answer: Yes; stream independence permits overlap but resource limits, implementation and workload may prevent it.
Confirm with a timeline, then inspect resource usage and configuration.
Trap: interpreting API concurrency as a guarantee of physical simultaneity.

**Q42. A 256-thread block uses 80 registers per thread without rounding. How many registers does it request?**
Answer: 256 × 80 = 20,480 registers in this simplified arithmetic.
A fictional 65,536-register pool permits floor(65,536/20,480) = 3 such blocks by registers alone.
Trap: asserting that real allocation ignores granularity or that all other limits permit three.

**Q43. How do you compare a full GPU and a partition?**
Answer: Query the resources visible to each instance and identify its actual partition profile and workload isolation.
Use compatible software, workload, correctness checks and measurement boundaries.
Trap: scaling full-device specifications linearly without checking the partition's assigned resources.

**Q44. Why can a kernel with fewer HBM bytes be slower?**
Answer: It may execute more instructions, use more synchronization, reduce occupancy or stress another boundary.
Test these competing mechanisms with matched counters and repeated unprofiled timings.
Trap: assuming HBM is the limiting resource without evidence.

**Q45. What evidence is needed for a hardware-specific claim?**
Answer: Identify the exact SKU, configuration and primary source for a specification.
For a performance claim, additionally provide the workload, setup, measurement procedure and result provenance.
Trap: treating a vendor peak or an invented example as an observed application result.

**SOURCED:** [S7] NVIDIA, *Hopper Tuning Guide*, CUDA 12.4.1 archive, guide revision 1.0, sections 1.4.1–1.4.3; consulted 2026-11-07.
Link: https://docs.nvidia.com/cuda/archive/12.4.1/hopper-tuning-guide/index.html .
Scope: documented H100/compute-capability-9.0 characteristics and explicitly stated A100 comparisons.
Limitations: historical guide, vendor figures, not independently reproduced and not a substitute for installed-device queries.
Review trigger: new GPU SKU, driver, CUDA toolkit, profiler, runtime configuration or workload.

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
