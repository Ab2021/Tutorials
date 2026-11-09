---
collection: "gpu-architecture-accelerators-and-ai-hardware"
file_id: "01"
topic: "CUDA roofline arithmetic intensity and kernel performance diagnosis"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "GPU, compiler, profiler, precision, traffic definition, or workload changes"
notation_style: "plain-text"
prerequisites: ["CUDA threads, blocks, memory hierarchy, floating-point arithmetic, dimensional analysis"]
pattern_tags: [roofline, measurement, kernel-diagnosis]
source_status: "OPEN"
owner_role: "GPU curriculum maintainer"
source_index: "INDEX.md"
---
# CUDA Roofline Arithmetic Intensity and Kernel Diagnosis

## 1. File Header

**Outcome:** Calculate a boundary-specific arithmetic intensity, construct a conditional roof, and choose the next discriminating experiment.
**Teaching home:** [module 23](../INDEX.md); [CUDA kernel design plan](../../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-kernel).
**Source inventory:** [directory.txt](../../../../Templates/directory.txt), especially line 391, lists `CUDA/Module_08_Kernel_Design/01_roofline_model_and_arithmetic_intensity.md`; the [exact source crosswalk](../../../00_SOURCE_TOPIC_CROSSWALK.md) records its teaching destination. This is syllabus provenance, not access to the original source body.
**Source scope:** `CUDA/Module_08_Kernel_Design/01_roofline_model_and_arithmetic_intensity.md` is a visible crosswalk topic; its contents are not assumed available.
**Audience:** engineers who can read CUDA code and use dimensional analysis.
**Evidence:** DERIVED means consequence of stated assumptions; ILLUSTRATIVE means constructed, not measured.
**Evidence:** HEURISTIC means a testable prioritization, not a guarantee; OPEN identifies evidence still needed.
**Scope:** kernel and pipeline diagnosis, not a specification of any particular GPU.
**Prerequisites:** know threads, warps, blocks, basic matrix multiplication, floating-point formats and timing.
**Takeaway:** A roof is a conditional upper bound, not a forecast or a substitute for a correctness test.

## 2. Concept Map / Mental Model

Start with the output contract and one precisely timed workload.
Count arithmetic work using a declared operation convention.
Choose a memory boundary and count bytes crossing that boundary.
Divide work by those bytes to obtain arithmetic intensity.
Use a matching bandwidth and a matching compute ceiling.
The lower of compute and bandwidth ceilings limits a steady-state rate in the simplified model.
Compare achieved rate with the bound under the same measurement scope.
If near the sloped roof, investigate bytes or the matching bandwidth.
If near the horizontal roof, investigate the relevant compute pipeline or algorithmic work.
If far below both, investigate latency, launch overhead, occupancy, synchronization, imbalance and instruction throughput.
Repeat with L1, L2, device memory, or interconnect boundaries when each matters.
A kernel can be device-memory-light but L2-heavy when cached data is repeatedly accessed.
A workload can hit neither roof when it does not expose enough parallelism.
A change in measured intensity can come from cache, shape, kernel fusion or counting conventions.
**Takeaway:** Diagnosis is a sequence of falsifiable comparisons, not a label based on one chart.

## 3. Foundational Definitions

| Symbol or term | Meaning | Units and domain | Boundary |
|---|---|---|---|
| W | Defined floating-point work for one selected workload | FLOP, nonnegative scalar | State whether useful or executed operations are counted. |
| Q_h | Actual traffic across hierarchy boundary h | bytes, nonnegative scalar | Reads and writes at the same boundary and time scope. |
| I_h | Arithmetic intensity W/Q_h | FLOP/byte, Q_h > 0 | Different boundaries give different values. |
| t | Elapsed time of the same workload | seconds, positive scalar | Device kernel, range, or end-to-end must be named. |
| P | Achieved throughput W/t | FLOP/s | Numerator must match the compute roof convention. |
| B_h | Sustainable or assumed boundary bandwidth | byte/s, nonnegative | Read/write mix, data size and environment matter. |
| C | Compatible compute throughput ceiling | FLOP/s, nonnegative | Precision, instruction family and clock assumptions matter. |
| I_star | Balance point C/B_h | FLOP/byte, B_h > 0 | Not a universal device constant. |
| FMA | Fused multiply-add | conventionally two FLOP | One instruction need not mean one FLOP. |
| useful work | Operations required by the specified mathematical result | FLOP | Excludes padding and redundant operations. |
| executed work | Operations actually performed | FLOP | Can include padding, recomputation and masked work. |
| transaction | A hardware transfer unit at a named boundary | bytes | May exceed the requested useful bytes. |
| occupancy | Active warps relative to the supported maximum | dimensionless | Does not itself measure scheduler issue efficiency. |
| cache residency | Data retained at a named cache level | qualitative or bytes | It can change between cold and warm trials. |
| latency | Time to complete an operation or workload | seconds | Not interchangeable with throughput. |
| FLOP convention | Mapping from arithmetic operations to count | operations | Count an FMA consistently as two if using that convention. |
| tensor operation | Matrix instruction operating on a tile | operations | Peak depends on data types and instruction shape. |

Byte units here use decimal TB = 10^12 bytes and TFLOP = 10^12 FLOP.
MiB means 2^20 bytes; do not silently mix these unit systems.
A zero-work workload gives zero useful throughput but may still incur launch time.
If Q_h is zero, I_h is undefined in this simple quotient; report boundary traffic and compute separately.
**Takeaway:** Every number must carry its population, time interval, units and memory boundary.

## 4. Core Formulas with Full Derivation

### 4.1 Base: intensity and performance

Validity: W and Q_h refer to the same workload and Q_h is strictly positive.
Derivation: intensity is the defined work divided by transferred bytes.
```text
FORMULA [R1]: I_h = W / Q_h
WHERE: W [FLOP]; Q_h [byte]; I_h [FLOP/byte].
CONSTRAINT: same work and hierarchy boundary, Q_h > 0.
NUMERICAL EXAMPLE: 2e12 FLOP / 0.5e12 byte = 4 FLOP/byte.
INTERPRETATION: each transferred byte supports four counted FLOP on average.
```
This is an aggregate ratio; individual operations need not have this intensity.
Validity: elapsed time t is positive and covers the counted work and bytes.
Derivation: divide both W and Q_h by t and substitute R1.
```text
FORMULA [R2]: P = W/t = I_h * (Q_h/t)
WHERE: P [FLOP/s]; t [s]; Q_h/t [byte/s].
CONSTRAINT: all quantities cover the same execution interval.
NUMERICAL EXAMPLE: W=2e12 FLOP, Q_h=0.5e12 byte, t=0.5 s;
                   P=4e12 FLOP/s and Q_h/t=1e12 byte/s.
INTERPRETATION: 4 FLOP/byte times 1 TB/s gives 4 TFLOP/s.
```
The equality is accounting, not evidence that bandwidth causes the latency.

### 4.2 General: conditional roof and balance point

Validity: C bounds the selected operation rate and B_h bounds transfers at h over the same time scope.
Derivation: P <= C and Q_h/t <= B_h; R2 implies P <= I_h*B_h.
Both inequalities hold, so their minimum is an upper bound.
```text
FORMULA [R3]: P <= min(C, B_h * I_h)
WHERE: C [FLOP/s], B_h [byte/s], I_h [FLOP/byte].
CONSTRAINT: comparable work, traffic, precision, interval and valid ceilings.
NUMERICAL EXAMPLE: C=60 TFLOP/s, B_h=2 TB/s, I_h=4 FLOP/byte;
                   P <= min(60, 8) = 8 TFLOP/s.
INTERPRETATION: the idealized bandwidth roof is below the compute roof.
```
A measured microbenchmark bandwidth is an empirical comparison ceiling, not necessarily a strict physical bound for every read/write mix.
Validity: B_h is strictly positive and both roofs use the same FLOP convention.
Derivation: set C equal to B_h*I_h and solve for I_h.
```text
FORMULA [R4]: I_star = C / B_h
WHERE: I_star [FLOP/byte], C [FLOP/s], B_h [byte/s].
CONSTRAINT: B_h > 0; use matching instruction and memory boundaries.
NUMERICAL EXAMPLE: 60 TFLOP/s / 2 TB/s = 30 FLOP/byte.
INTERPRETATION: intensity below 30 is on this model's bandwidth side.
```
This is not a universal threshold even for a single named GPU.

### 4.3 Variants: hierarchies and time lower bound

Validity: each boundary has an independently valid bandwidth bound for the same operation count and elapsed interval.
Derivation: apply R3 separately, then take the minimum of all applicable bounds.
```text
FORMULA [R5]: P <= min(C, B_1*I_1, B_2*I_2, ...)
WHERE: I_h=W/Q_h; h indexes distinct transfer boundaries.
CONSTRAINT: Q_h > 0; ceilings must be valid for the chosen work and time.
NUMERICAL EXAMPLE: C=60, B_1*I_1=12, B_2*I_2=8 TFLOP/s;
                   P <= 8 TFLOP/s.
INTERPRETATION: the tightest valid roof dominates this simple bound.
```
Overlaps, contention and dependencies can make actual performance lower than every roof.
Validity: W is positive and the R3 bound is positive.
Derivation: P=W/t and rearrange P <= min(C,B_h*I_h).
```text
FORMULA [R6]: t >= max(W/C, Q_h/B_h)
WHERE: t [s], W [FLOP], C [FLOP/s], Q_h [byte], B_h [byte/s].
CONSTRAINT: positive ceilings; same interval and compatible ceilings.
NUMERICAL EXAMPLE: W=2 TFLOP, Q_h=0.5 TB, C=60 TFLOP/s, B_h=2 TB/s;
                   t >= max(0.0333, 0.25) s = 0.25 s.
INTERPRETATION: idealized device time cannot be below the larger term.
```
Do not add the two times unless an explicit non-overlap model justifies addition.

### 4.4 Limits and counterexamples

A hardware-advertised peak can be unattainable at a constrained clock or precision.
A byte count based on source-level loads may differ from transactions at DRAM.
A kernel below a sloped line may be latency-limited rather than bandwidth-saturated.
A high arithmetic intensity cannot make a serial dependency chain parallel.
An increased intensity due solely to redundant FLOPs does not imply faster useful work.
Changing W between versions invalidates a naive achieved-FLOP/s comparison.
For Q_h approaching zero, intensity grows without bound but throughput remains bounded by compute and other resources.
For zero bandwidth with positive required traffic, the assumed workload cannot complete in finite time under this model.
**Takeaway:** The algebra is simple only after compatible counts and ceilings are established.

## 5. Worked Numerical Example

**ILLUSTRATIVE:** One processing range performs 2 TFLOP and moves 0.5 TB across device memory.
Assume 60 TFLOP/s compatible compute and 2 TB/s compatible device-memory bandwidth.
These are invented modeling inputs, not measured specifications of a named accelerator.
The byte total includes all reads and writes at that one device-memory boundary.
Assume its count and the work count refer to the same range and precision.
Using R1, intensity is 2/0.5 = 4 FLOP/byte.
Using R4, the balance point is 60/2 = 30 FLOP/byte.
Using R3, the device-memory roof is 2*4 = 8 TFLOP/s.
The compute roof remains 60 TFLOP/s.
Therefore the conditional bound is 8 TFLOP/s, not the expected achieved rate.
Using R6, the lower time bound is max(2/60, 0.5/2) seconds.
That equals max(0.0333..., 0.25) = 0.25 seconds.
Independent arithmetic check: 8 TFLOP/s * 0.25 s = 2 TFLOP.
Independent traffic check: 2 TB/s * 0.25 s = 0.5 TB.
Now consider a hypothetical fusion that eliminates 0.1 TB of traffic at the same boundary.
Assume output semantics and the 2 TFLOP useful work are unchanged.
Assume fusion adds no other transferred bytes or extra execution work for this ideal comparison.
The new traffic is 0.5 - 0.1 = 0.4 TB.
The new intensity is 2/0.4 = 5 FLOP/byte.
The new roof is min(60, 2*5) = 10 TFLOP/s.
The new time lower bound is max(2/60, 0.4/2) = 0.2 seconds.
The theoretical roof ratio is 10/8 = 1.25; this is not a measured speedup.
The ideal time reduction is 0.05 seconds, or 20% of 0.25 seconds.
Cross-check: 0.1 TB / 2 TB/s = 0.05 s, assuming the same bandwidth roof binds both variants.
Sensitivity: if effective bandwidth is only 1 TB/s, the original roof is 4 TFLOP/s and the ideal lower bound is 0.5 s.
Sensitivity: if eliminated intermediate accesses hit cache, device-memory savings can be less than 0.1 TB.
Sensitivity: if fusion adds spills, its actual Q_h may even increase.
Sensitivity: if changed numerical order violates the output tolerance, the optimization is invalid regardless of throughput.
OPEN: no GPU, compiler, kernel, runtime, power state or experiment has been measured for this example.
**Takeaway:** The traffic reduction changes a conditional bound; measuring correctness and actual time decides the optimization.

## 6. Visual / Geometric Intuition

```text
P [FLOP/s] on log scale
  ^                      ---------------- C, compute roof
  |                    /
  |                  /  slope 1: B_h * I_h
  |                /
  |              /
  +-------------+----------------------> I_h [FLOP/byte], log scale
              C / B_h
```
The sloped line intersects the horizontal line at R4's balance point.
On logarithmic axes, multiplying intensity by two doubles the bandwidth-bound roof.
After the intersection, further intensity increase does not lift the compute roof.
A point below the sloped roof is not necessarily bandwidth-saturated.
The vertical gap indicates headroom under the model, not a guaranteed optimization opportunity.
A point above a purported strict roof indicates mismatched counts, scope, units or an invalid ceiling.
Separate hierarchy roofs require separate intensity coordinates because Q_h differs by level.
A high-level multi-roof diagram must label the boundary of every diagonal line.
**Takeaway:** Read the roof and the coordinate together, not a visually appealing slope in isolation.

## 7. Comparison / Selection Table

| Choice | Use when | Tradeoff | Disqualifier | Measurement |
|---|---|---|---|---|
| Improve coalescing | transaction overhead is visible | layout changes can cost conversion | no compatible access pattern | boundary bytes and time |
| Tile for reuse | reuse is available within a tile | shared memory and registers rise | residency or synchronization dominates | traffic, occupancy, elapsed time |
| Fuse stages | an intermediate crosses the selected boundary | greater resource pressure | numerical or resource constraint violated | end-to-end correctness and time |
| Split stages | fusion causes spills or poor occupancy | extra launches and intermediate traffic | bandwidth or launch cost dominates | compare matched whole ranges |
| Recompute | cheaper arithmetic can replace an expensive load | increased compute work | compute pipeline already saturated | useful work and latency |
| Reduce precision | error budget and hardware support allow it | conversion and accuracy costs | output fails tolerance | error distribution and matched time |
| Increase parallelism | shortfall arises from insufficient active work | more scheduling and launch overhead | insufficient independent work | wave distribution and timing |
| Optimize instruction mix | relevant pipeline is saturated | alternate instructions may reduce accuracy | incompatible precision or semantics | generated instructions and error |
| Leave unchanged | bottleneck lies elsewhere in the application | local headroom remains | overall objective still unmet | end-to-end baseline |

**HEURISTIC:** Prefer the experiment that can distinguish competing causes at lowest correctness risk.
**Takeaway:** The right action depends on the measured bottleneck and the output contract.

## 8. Failure Modes & Edge Cases

All scenarios below are HYPOTHETICAL and all checks are proposed, not observed production incidents.
Each scenario separates a competing explanation, a discriminating signal, a bounded intervention and a verification.

### D01. Uncoalesced input
- **Symptom:** adjacent lanes request scattered words.
- **Competing cause:** poor spatial locality rather than bandwidth saturation.
- **Discriminator:** compare requested bytes with transactions at the relevant level.
- **Bounded intervention:** reorder thread-to-data mapping without changing values.
- **Verification:** check output and device-memory traffic across aligned and offset inputs.

### D02. Strided stores
- **Symptom:** adjacent output lanes write distant addresses.
- **Competing cause:** write policy or cache effects instead of stride.
- **Discriminator:** compare store transaction efficiency over two layouts.
- **Bounded intervention:** choose a contiguous output mapping where consumers permit.
- **Verification:** validate downstream layout and whole-range time.

### D03. Redundant global loads
- **Symptom:** multiple threads reload the same value.
- **Competing cause:** cache hits may already suppress device-memory traffic.
- **Discriminator:** compare cache-level and device-memory bytes.
- **Bounded intervention:** share or tile reused values only when reuse is confirmed.
- **Verification:** measure all relevant traffic and shared-memory pressure.

### D04. Cache warmup bias
- **Symptom:** a repeated launch fits in cache.
- **Competing cause:** the first launch has a colder working set.
- **Discriminator:** compare cold and warm access protocols.
- **Bounded intervention:** represent the intended production reuse explicitly.
- **Verification:** report both distributions and hierarchy traffic.

### D05. Write allocation overhead
- **Symptom:** stored values induce more boundary traffic than payload.
- **Competing cause:** the profiler may count a different boundary.
- **Discriminator:** inspect read and write traffic independently.
- **Bounded intervention:** adjust store strategy only if supported and safe.
- **Verification:** confirm output visibility and actual traffic.

### D06. Register spilling
- **Symptom:** local memory traffic rises after fusion.
- **Competing cause:** large input footprint alone can increase traffic.
- **Discriminator:** compare compiler resource report and local-memory counters.
- **Bounded intervention:** reduce live ranges or split the stage.
- **Verification:** recheck correctness and end-to-end latency.

### D07. Insufficient blocks
- **Symptom:** small grids do not cover the device.
- **Competing cause:** a serial dependency may also suppress utilization.
- **Discriminator:** compare grid size and active wave distribution.
- **Bounded intervention:** batch independent inputs when semantics allow.
- **Verification:** check latency and throughput at production batch sizes.

### D08. Uneven tail blocks
- **Symptom:** partial tiles do less useful work.
- **Competing cause:** a shape-specific branch may cause divergence.
- **Discriminator:** inspect work distribution and edge fraction.
- **Bounded intervention:** choose shape-aware tiling or a separate edge path.
- **Verification:** test odd sizes and compare useful output.

### D09. Long dependency chain
- **Symptom:** few warps are eligible despite work in flight.
- **Competing cause:** memory stalls can produce a similar signal.
- **Discriminator:** inspect scheduler eligibility with instruction dependencies.
- **Bounded intervention:** increase independent work without increasing spills excessively.
- **Verification:** check generated instructions and execution time.

### D10. Barrier imbalance
- **Symptom:** some warps arrive early and wait.
- **Competing cause:** slow memory access may delay all participants.
- **Discriminator:** correlate barrier wait with producer completion.
- **Bounded intervention:** balance work before the synchronization point.
- **Verification:** run race checks and measure barrier stalls.

### D11. Shared-memory bank conflicts
- **Symptom:** shared accesses serialize within a warp.
- **Competing cause:** barrier or occupancy issues can look similar.
- **Discriminator:** inspect shared access transactions against requested accesses.
- **Bounded intervention:** change layout or padding with a proven index mapping.
- **Verification:** test boundaries and shared transaction counts.

### D12. Host launch overhead
- **Symptom:** many short kernels dominate host-visible latency.
- **Competing cause:** device synchronization may be the real cost.
- **Discriminator:** inspect timeline and per-launch device durations.
- **Bounded intervention:** batch or fuse only when outputs remain equivalent.
- **Verification:** measure host-to-result latency without profiler overhead.

### D13. Synchronization on host
- **Symptom:** a host wait interrupts overlap.
- **Competing cause:** copy-engine saturation can also limit throughput.
- **Discriminator:** compare timeline with and without the specific wait.
- **Bounded intervention:** defer only waits not required by data dependencies.
- **Verification:** test cross-stream ordering and end-to-end results.

### D14. Transfer bottleneck
- **Symptom:** host-device copies dominate the request.
- **Competing cause:** a long kernel can hide the copy in a timeline.
- **Discriminator:** measure transfer bytes and end-to-end critical path.
- **Bounded intervention:** reduce copies or overlap independent stages.
- **Verification:** validate buffer lifetime and latency distribution.

### D15. Wrong precision roof
- **Symptom:** single-precision work is compared with a tensor peak.
- **Competing cause:** generated code may not use the expected instruction family.
- **Discriminator:** inspect precision and executed instruction mix.
- **Bounded intervention:** choose a roof matching executed operations.
- **Verification:** recalculate intensity and report error tolerance.

### D16. FMA miscount
- **Symptom:** one instruction is counted as one operation.
- **Competing cause:** tensor tile accounting may also be incomplete.
- **Discriminator:** derive the count from the exact arithmetic convention.
- **Bounded intervention:** use two operations per scalar fused multiply-add consistently.
- **Verification:** reconcile static work and profiler convention.

### D17. Mixed useful and executed FLOPs
- **Symptom:** padding increases the reported operation rate.
- **Competing cause:** the faster rate could reflect actual useful output.
- **Discriminator:** compare useful output per second and executed operations.
- **Bounded intervention:** label both numerators separately.
- **Verification:** measure latency and numerical agreement.

### D18. Wrong bandwidth unit
- **Symptom:** bytes are mixed with bits or binary units.
- **Competing cause:** wrong interval can also distort the ratio.
- **Discriminator:** write dimensional conversions explicitly.
- **Bounded intervention:** normalize every quantity before plotting.
- **Verification:** recompute with an independent unit check.

### D19. Arithmetic overflow
- **Symptom:** integer work-count multiplication wraps.
- **Competing cause:** a format conversion may truncate the result.
- **Discriminator:** calculate dimensions in a wide integer type.
- **Bounded intervention:** use checked multiplication before conversion.
- **Verification:** test dimensions near the representable boundary.

### D20. Missing write bytes
- **Symptom:** only input reads appear in the model.
- **Competing cause:** cache behavior may also alter transfer volume.
- **Discriminator:** separate input, output and intermediate traffic.
- **Bounded intervention:** include writes at the selected boundary.
- **Verification:** compare totals with the profiler definition.

### D21. Implicit workspace
- **Symptom:** library kernels move an unmodeled workspace.
- **Competing cause:** hidden initialization may have a separate launch.
- **Discriminator:** trace all kernels and allocations in the measured range.
- **Bounded intervention:** include workspace traffic and time in range accounting.
- **Verification:** compare the complete call against the baseline.

### D22. Cold-start compilation
- **Symptom:** first invocation includes code generation.
- **Competing cause:** first invocation may also initialize context and memory.
- **Discriminator:** separate compilation, initialization and steady-state samples.
- **Bounded intervention:** report cold and warmed paths separately.
- **Verification:** repeat on a clean process and a warmed process.

### D23. Clock variation
- **Symptom:** effective compute and bandwidth change over time.
- **Competing cause:** a competing process can produce similar variance.
- **Discriminator:** record clocks, power and other GPU activity.
- **Bounded intervention:** control or stratify environmental state.
- **Verification:** compare distributions under representative conditions.

### D24. Concurrent workloads
- **Symptom:** another process consumes memory or compute resources.
- **Competing cause:** thermal changes can produce a similar slowdown.
- **Discriminator:** compare isolated and shared-device traces.
- **Bounded intervention:** schedule or budget contention explicitly.
- **Verification:** measure service latency with realistic concurrency.

### D25. Profiler replay changes cache
- **Symptom:** multiple passes observe a different cache state.
- **Competing cause:** nondeterministic input can also change counters.
- **Discriminator:** compare replay settings with unprofiled behavior.
- **Bounded intervention:** use a replay mode appropriate to the workload.
- **Verification:** validate timing outside the profiler and compare metrics.

### D26. Profiling short kernel
- **Symptom:** instrumentation dominates the operation being studied.
- **Competing cause:** launch overhead may also dominate the real workload.
- **Discriminator:** compare unprofiled batched timing with profile time.
- **Bounded intervention:** collect a minimal relevant metric set.
- **Verification:** repeat unprofiled timings and report dispersion.

### D27. Tensor padding
- **Symptom:** tile-aligned execution computes extra elements.
- **Competing cause:** masked edge traffic may be the main overhead.
- **Discriminator:** compare useful dimensions with executed tile shape.
- **Bounded intervention:** select a compatible tile or fallback path.
- **Verification:** check odd shapes and useful throughput.

### D28. Precision conversion
- **Symptom:** format conversion consumes instructions or bandwidth.
- **Competing cause:** compute may still be limited by the main operation.
- **Discriminator:** inspect conversion instruction mix and associated bytes.
- **Bounded intervention:** retain a compatible representation across stages.
- **Verification:** validate error budget and whole-pipeline timing.

### D29. Sparse claim mismatch
- **Symptom:** the count assumes hardware sparsity acceleration.
- **Competing cause:** irregularity or unsupported encoding prevents that path.
- **Discriminator:** inspect generated instruction family and actual format.
- **Bounded intervention:** use the applicable dense roof when sparse support is absent.
- **Verification:** check values, executed code and matched latency.

### D30. Memory latency not bandwidth
- **Symptom:** low bandwidth accompanies a long kernel.
- **Competing cause:** low work volume can make utilization misleading.
- **Discriminator:** inspect eligible warps and outstanding requests.
- **Bounded intervention:** increase independent requests or improve locality.
- **Verification:** ensure speedup at the same output and workload.

### D31. Instruction issue limit
- **Symptom:** a pipeline is busy while arithmetic rate is low.
- **Competing cause:** a memory bottleneck can also reduce issue.
- **Discriminator:** inspect instruction mix and scheduler issue activity.
- **Bounded intervention:** reduce unnecessary address or control instructions.
- **Verification:** compare generated code and useful throughput.

### D32. Branch divergence
- **Symptom:** threads in a warp take different paths.
- **Competing cause:** data-dependent memory accesses can dominate instead.
- **Discriminator:** compare branch distribution and memory activity.
- **Bounded intervention:** reorganize data or work only if ordering permits.
- **Verification:** test all branches and measure distribution-sensitive time.

### D33. Atomic contention
- **Symptom:** many threads update the same location.
- **Competing cause:** insufficient bandwidth can slow noncontended atomics too.
- **Discriminator:** compare contended and distributed-key workloads.
- **Bounded intervention:** aggregate locally before the global update.
- **Verification:** validate ordering requirements and numerical tolerance.

### D34. Reduction order
- **Symptom:** parallel reassociation changes floating-point results.
- **Competing cause:** an indexing bug can cause much larger errors.
- **Discriminator:** compare against a higher-precision reference by shape.
- **Bounded intervention:** set an explicit tolerance or preserve required order.
- **Verification:** test cancellation and extreme values.

### D35. NaN propagation
- **Symptom:** a fast path changes exceptional-value behavior.
- **Competing cause:** input preprocessing may have removed NaNs.
- **Discriminator:** run explicit NaN infinity and signed-zero cases.
- **Bounded intervention:** choose operations meeting the stated numerical contract.
- **Verification:** compare bitwise or tolerance-based reference as required.

### D36. Occupancy overfocus
- **Symptom:** more active warps do not improve time.
- **Competing cause:** register reduction may introduce extra instructions.
- **Discriminator:** compare eligible issue and runtime before changing resources.
- **Bounded intervention:** change occupancy only when latency hiding is limiting.
- **Verification:** confirm no spills and compare performance distributions.

### D37. Resource-limited tiling
- **Symptom:** larger tiles reduce nominal bytes but lower residency.
- **Competing cause:** edge inefficiency may explain the slowdown.
- **Discriminator:** inspect registers shared memory and active blocks.
- **Bounded intervention:** try a smaller tile while retaining enough reuse.
- **Verification:** measure traffic and time across representative shapes.

### D38. Misaligned access
- **Symptom:** unaligned base pointers require extra transactions.
- **Competing cause:** stride can independently waste transactions.
- **Discriminator:** compare aligned and intentionally shifted allocations.
- **Bounded intervention:** align allocations or adapt the mapping safely.
- **Verification:** test real offsets and memory bounds.

### D39. Irregular gather
- **Symptom:** useful bytes are much smaller than fetched sectors.
- **Competing cause:** cache reuse can reduce the physical transfer count.
- **Discriminator:** compare sector requests and cache hit behavior.
- **Bounded intervention:** reorder indices if application semantics allow.
- **Verification:** include reorder cost and validate permutation inversion.

### D40. Repeated output writes
- **Symptom:** an intermediate is written and read between stages.
- **Competing cause:** the intermediate might remain cache resident.
- **Discriminator:** compare device-memory and cache traffic for the full range.
- **Bounded intervention:** fuse only if resource usage and semantics allow.
- **Verification:** measure complete range and intermediate elimination.

### D41. Pipeline imbalance
- **Symptom:** a producer is slower than its consumer.
- **Competing cause:** synchronization protocol can also stall both.
- **Discriminator:** inspect stage timing and buffer occupancy if available.
- **Bounded intervention:** rebalance work or buffer depth under resource limits.
- **Verification:** test ordering and end-to-end throughput.

### D42. Wrong time window
- **Symptom:** traffic comes from one launch but time from a range.
- **Competing cause:** asynchronous timing can omit device execution.
- **Discriminator:** align profiler launch identity and timed interval.
- **Bounded intervention:** use a correctly synchronized same-scope timer.
- **Verification:** cross-check range and per-kernel durations.

### D43. Concurrent kernel timing
- **Symptom:** isolated kernel time does not reproduce overlap.
- **Competing cause:** contention can offset overlap gains.
- **Discriminator:** inspect the application timeline with concurrent kernels.
- **Bounded intervention:** evaluate the full range rather than summing isolated times.
- **Verification:** measure wall time under representative concurrency.

### D44. Division by zero
- **Symptom:** no bytes cross a chosen boundary.
- **Competing cause:** a missing or unsupported counter can report zero too.
- **Discriminator:** inspect counter validity and cache state.
- **Bounded intervention:** report the undefined intensity instead of dividing.
- **Verification:** recollect with a supported metric and known input.

### D45. Sampling noise
- **Symptom:** small observed differences vary between runs.
- **Competing cause:** a deterministic performance regression may coexist.
- **Discriminator:** compare repeated paired timings and distributions.
- **Bounded intervention:** retain baseline until effect exceeds measurement noise.
- **Verification:** repeat under controlled representative conditions.

### D46. Shape distribution shift
- **Symptom:** an optimization favors a rare large shape.
- **Competing cause:** the dominant shapes may have changed since baseline.
- **Discriminator:** weight measurements by current workload distribution.
- **Bounded intervention:** dispatch by shape only if complexity is justified.
- **Verification:** measure aggregate and tail latency under a holdout mix.

### D47. Tightened error budget
- **Symptom:** lower precision no longer meets downstream tolerance.
- **Competing cause:** a separate logic bug may mimic numerical drift.
- **Discriminator:** compare against a high-precision reference on stress inputs.
- **Bounded intervention:** restore precision or modify the accumulation strategy.
- **Verification:** validate downstream metrics and edge cases.

### D48. Wrong memory boundary
- **Symptom:** L2 traffic is compared to device-memory bandwidth.
- **Competing cause:** bytes from a different launch can look similar.
- **Discriminator:** inspect counter names and hierarchy definitions.
- **Bounded intervention:** pair each byte counter with its own roof.
- **Verification:** document the level and replot the workload.

### D49. Racing shared memory
- **Symptom:** a faster variant produces intermittent errors.
- **Competing cause:** uninitialized memory can create similar symptoms.
- **Discriminator:** use targeted race and initialization diagnostics.
- **Bounded intervention:** add the synchronization required by the data dependency.
- **Verification:** stress repeated varied schedules and compare outputs.

### D50. Excessive recomputation
- **Symptom:** reported intensity rises while latency worsens.
- **Competing cause:** a simultaneous clock change can also slow execution.
- **Discriminator:** compare useful FLOPs, executed instructions and time.
- **Bounded intervention:** retain recomputation only if it saves a limiting transfer.
- **Verification:** validate useful throughput and resource counters.

### D51. Insufficient benchmark duration
- **Symptom:** timer resolution obscures small kernel differences.
- **Competing cause:** actual performance may be noisy at larger durations.
- **Discriminator:** time batches and divide by the exact number of launches.
- **Bounded intervention:** amortize timing overhead without changing data reuse.
- **Verification:** compare single-launch and batched conditions.

### D52. Wrong bandwidth reference
- **Symptom:** a read-only microbenchmark is used for mixed reads and writes.
- **Competing cause:** cache-resident traffic can also inflate apparent bandwidth.
- **Discriminator:** measure a reference with comparable direction and footprint.
- **Bounded intervention:** label the reference empirical rather than an absolute law.
- **Verification:** repeat for the real access mix and device configuration.

### D53. Dynamic allocation in hot path
- **Symptom:** allocation or initialization dominates a request.
- **Competing cause:** a synchronization operation can account for the delay.
- **Discriminator:** trace host and device activity around allocation.
- **Bounded intervention:** reuse buffers if ownership and lifetime permit.
- **Verification:** test concurrency and peak memory as well as time.

### D54. Missing tail work
- **Symptom:** a vectorized kernel skips final elements.
- **Competing cause:** the reference output may itself be incorrectly shaped.
- **Discriminator:** test lengths around each vector and tile boundary.
- **Bounded intervention:** add a bounds-safe tail path.
- **Verification:** compare every output element for small adversarial lengths.

### D55. Incorrect overlap assumption
- **Symptom:** adding compute and transfer ceilings exaggerates lower bound.
- **Competing cause:** hidden dependencies may actually serialize phases.
- **Discriminator:** inspect the timeline and dependencies.
- **Bounded intervention:** model maximum for overlap and sum only for proven serialization.
- **Verification:** compare predictions with observed range duration.

### D56. Compiler version change
- **Symptom:** a rebuild changes register use or instructions.
- **Competing cause:** a driver or power-state change can also alter time.
- **Discriminator:** record binary hash compiler flags and generated code.
- **Bounded intervention:** pin the toolchain during a controlled comparison.
- **Verification:** rerun correctness and representative timings on both builds.

### D57. Missing error check
- **Symptom:** an asynchronous failure surfaces after the measured kernel.
- **Competing cause:** the numerical mismatch may arise in a later consumer.
- **Discriminator:** check launch status and synchronize at the diagnosis boundary.
- **Bounded intervention:** propagate errors without changing production synchronization unnecessarily.
- **Verification:** exercise an invalid-input path and the normal workload.

### D58. Output not consumed
- **Symptom:** a benchmark omits necessary downstream work.
- **Competing cause:** compiler elimination or lazy evaluation may also alter work.
- **Discriminator:** verify output use and the actual launched kernels.
- **Bounded intervention:** benchmark the same observable output contract.
- **Verification:** compare complete application results and end-to-end latency.

### D59. Underestimated output size
- **Symptom:** temporary outputs exceed the assumed traffic model.
- **Competing cause:** compression or caching can move fewer physical bytes.
- **Discriminator:** derive sizes from actual shapes dtypes and layout.
- **Bounded intervention:** include intermediate and metadata transfers where relevant.
- **Verification:** compare byte estimates against boundary counters.

### D60. Incorrect comparison baseline
- **Symptom:** variants use different shapes or numeric accuracy.
- **Competing cause:** one variant may include extra initialization.
- **Discriminator:** hold input outputs warmup and scope constant.
- **Bounded intervention:** choose a reference implementing identical semantics.
- **Verification:** report any intentional difference and its effect.

**Takeaway:** Diagnose with a competing explanation and a falsifiable check before changing code.

### 8.1 Numerical diagnosis laboratories

All laboratories are constructed examples, not measurements or device specifications.
For every bandwidth limit below, assume the indicated ceiling is valid for the access mix and interval.
Rates use decimal units; a bound is not an expected runtime.
The proposed counterfactuals are tests of hypotheses, not promises of improvement.

#### L01. Vector addition and transaction waste
- Contract: `c[i] = a[i] + b[i]` for N FP32 elements, with distinct arrays.
- Useful arithmetic is N additions; useful payload is 8N input bytes and 4N output bytes.
- Under one read per input and one write per output, intensity is N/(12N) = 1/12 FLOP/byte.
- At an assumed 1.2 TB/s, the corresponding roof is 0.1 TFLOP/s.
- For N = 1,000,000, compulsory payload is 12 MB and ideal transfer time is 10 microseconds.
- A measured 18 MB at the same boundary implies 1/18 FLOP/byte, not 1/12.
- If time is 20 microseconds, achieved traffic is 0.9 TB/s and useful rate is 0.05 TFLOP/s.
- The extra 6 MB might reflect transactions, write handling, or other traffic in the range.
- Compare aligned contiguous input against an offset view while holding N fixed.
- Reject alignment as the sole cause if transaction bytes and latency do not respond.
- Check aliasing and tail handling before treating either kernel as equivalent.

#### L02. SAXPY and scalar reuse
- Contract: `y[i] = a*x[i] + y[i]` with FP32 arrays and a scalar `a`.
- Count an FMA as two FLOPs, including when the implementation uses separate multiply and add.
- Ignoring the scalar's amortized load, payload is 4N for x, 4N for old y, and 4N for new y.
- Intensity is 2/12 = 1/6 FLOP/byte under those payload assumptions.
- At 1.2 TB/s, this gives a 0.2 TFLOP/s bandwidth ceiling.
- One million elements require 12 MB and at least 10 microseconds at that ceiling.
- If a scalar is redundantly fetched across the selected boundary, include those actual bytes.
- The same input and output pointer is required for the in-place operation, but x/y aliasing needs a separate contract.
- A low FLOP rate can be fully consistent with high bandwidth because useful arithmetic per byte is small.
- Compare data sizes below and above cache capacity to distinguish cache residency from DRAM traffic.
- Verify whether FMA contraction changes the permitted numerical result.

#### L03. Dot product with partial reductions
- Contract: sum N products of FP32 vectors and return one FP32 scalar.
- Useful work is N multiplications and N-1 additions, or 2N-1 FLOPs.
- Input payload is 8N bytes; a single scalar result adds four bytes.
- Ignoring partial-reduction traffic, intensity approaches 0.25 FLOP/byte as N grows.
- At N = 1024, work is 2047 FLOPs and payload is 8196 bytes.
- The resulting intensity is approximately 0.2498 FLOP/byte.
- At an assumed 1 TB/s, the transfer-only time lower bound is 8.196 nanoseconds.
- This tiny lower bound does not account for launch latency or a dependency chain in the reduction.
- A two-pass implementation writes and rereads partial sums; include both directions.
- To distinguish launch overhead, vary N while keeping the implementation and timing scope fixed.
- Check cancellation-heavy data because reassociation changes floating-point rounding.

#### L04. Matrix-vector multiplication
- Contract: dense FP32 matrix A of M rows and K columns times vector x.
- Use 2MK FLOPs as the conventional FMA count and ignore final-add convention differences.
- Under ideal reuse of x, payload is 4MK + 4K + 4M bytes for A, x and y.
- At M = K = 1024, work is 2,097,152 FLOPs.
- The ideal payload is 4,202,496 bytes and intensity is about 0.499 FLOP/byte.
- At 2 TB/s, the bandwidth roof is about 0.998 TFLOP/s.
- If x is reread from device memory for every row, add approximately 4MK rather than 4K bytes.
- The resulting payload is 8,392,704 bytes and intensity is about 0.250 FLOP/byte.
- A cache-level traffic counter can increase while device-memory traffic remains nearly unchanged.
- Compare vector footprints that fit and do not fit in the relevant cache.
- A packed matrix format may change both traffic and indexing cost, so compare whole-operation time.

#### L05. Square matrix multiply, compulsory data
- Contract: C = A*B for three distinct FP32 square matrices of order n; beta is zero.
- Approximate useful work by 2n^3 FLOPs under the FMA convention.
- If each input is read once and output is written once, payload is 12n^2 bytes.
- The corresponding optimistic intensity is n/6 FLOP/byte.
- At n = 1024, work is 2,147,483,648 FLOPs and payload is 12,582,912 bytes.
- This gives about 170.67 FLOP/byte before any extra traffic.
- At 2 TB/s and 60 TFLOP/s, the diagonal is 341.33 TFLOP/s and the compute roof is lower.
- The assumed compulsory traffic is an idealization, not a guarantee of attainable reuse.
- If beta is nonzero, reading old C adds 4n^2 bytes and changes the ideal intensity to n/8.
- If tensor operations use lower-precision operands or padded tiles, adjust both byte and operation counts.
- Verify numerical tolerance and the actual instruction family before selecting a compute ceiling.

#### L06. Tile-level reuse in matrix multiply
- Consider one output tile of 32 by 32 and one reduction slice of length 32.
- It performs 2*32*32*32 = 65,536 conventional FLOPs.
- A 32 by 32 FP32 A tile and a same-size B tile contain 8,192 bytes together.
- With no output traffic in this slice accounting, intensity is 8 FLOP/byte.
- The complete kernel must also account for output and any reread across reduction slices.
- Reusing each input tile across multiple outputs is what raises intensity above scalar multiply-add.
- Doubling the tile dimension can increase reuse but also increases shared-memory demand quadratically.
- If two buffers hold both 32 by 32 input tiles, their FP32 footprint is 16,384 bytes.
- Compare this footprint with the per-block shared-memory limit and occupancy allocation granularity.
- If the same device-memory bytes remain after tiling, the original cache may already supply the reuse.
- Test nonmultiples of tile sizes to expose edge predicates and extra executed work.

#### L07. Seven-point stencil
- Contract: one interior output is the weighted sum of its center and six neighbors.
- Assume seven multiplications and six additions, giving 13 useful FLOPs per output.
- With no inter-output reuse, seven FP32 reads plus one FP32 write require 32 bytes.
- The no-reuse intensity is 13/32 = 0.40625 FLOP/byte.
- Under ideal large-grid reuse with one input read and one output write per interior point, it approaches 13/8 = 1.625.
- At 1.6 TB/s the corresponding diagonal roofs are 0.65 and 2.6 TFLOP/s.
- These two cases are traffic scenarios, not guaranteed lower and upper timings for a real boundary policy.
- Halo loads, boundary conditions, write policies and finite dimensions alter the actual count.
- Compare L2 and DRAM traffic when tiling to see which hierarchy level gains reuse.
- A shared-memory tile must load its halo and synchronize before consuming neighboring values.
- Validate boundary cells separately from the interior because their mathematical contract can differ.

#### L08. Sparse matrix-vector multiplication
- Contract: CSR matrix with 1,000,000 nonzeros and 100,000 rows times an FP32 vector.
- Use two conventional FLOPs per nonzero, for 2,000,000 useful FLOPs.
- FP32 values and 32-bit column indices contribute 8,000,000 bytes.
- The 100,001 32-bit row offsets contribute 400,004 bytes.
- Writing 100,000 FP32 outputs adds 400,000 bytes.
- If every referenced x value is loaded once per nonzero, x contributes another 4,000,000 bytes.
- Total modeled payload is 12,800,004 bytes, giving about 0.15625 FLOP/byte.
- At 1 TB/s, this payload gives a 12.8 microsecond transfer lower bound.
- Repeated x indices may hit cache, while random accesses may fetch more than their four-byte payload.
- Long rows can create load imbalance even when total bandwidth is low.
- Compare row-length distribution, transaction amplification and cache hits before choosing a row mapping.

#### L09. Histogram and atomic contention
- Contract: map N input values to bins and increment one integer counter per value.
- Floating-point work may be zero, so a FLOP roofline is not a useful primary representation.
- Instead define useful work as N updates and report updates per second.
- If each input is four bytes, input payload alone is 4N bytes.
- Actual atomic read-modify-write traffic depends on cache, aggregation and hardware semantics.
- For N = 1,000,000 and an input bandwidth limit of 1 TB/s, reading input takes at least 4 microseconds.
- That lower bound says nothing about a single hot bin's serialization.
- Hold N and the bin count fixed while comparing uniform and all-to-one key distributions.
- A slowdown concentrated in the all-to-one case supports contention rather than input bandwidth alone.
- Per-block aggregation trades extra local state and merging for fewer global atomic updates.
- Check counter overflow and ensure the sum of bins equals the number of valid inputs.

#### L10. Gather and sector utilization
- Contract: 32 lanes each read one FP32 element from a source array.
- The requested payload for one warp is 32*4 = 128 bytes.
- Assume an illustrative transfer sector of 32 bytes at the selected boundary.
- If all words occupy four distinct sectors, 128 bytes transfer and utilization is 100%.
- If every lane requires a different sector, 1,024 bytes transfer and utilization is 12.5%.
- These counts assume no overlap or hits from previous requests at that boundary.
- The 8-fold transaction difference does not guarantee an 8-fold latency difference.
- Sorting indices might improve locality but adds sorting cost and may alter output order.
- Compare requested payload, sector count and downstream permutation cost for the complete operation.
- Repeated addresses can be served differently from 32 distinct addresses and need their own case.
- Confirm that reordering preserves duplicate-index and output-order semantics.

#### L11. Shared-memory transpose
- Contract: transpose a tile without changing its element values.
- For a 32 by 32 FP32 tile, load and store payload together total 8,192 bytes.
- A transpose has no inherent floating-point arithmetic, so count elements or bytes per second.
- Assume 32 banks and one 32-bit word per bank mapping in the illustrative model.
- A row-major shared array of width 32 maps a column's successive words to the same bank.
- Padding the row stride to 33 maps successive column words to successive banks in this model.
- The padded shared tile occupies 32*33*4 = 4,224 bytes instead of 4,096 bytes.
- Hardware bank organization and instruction width must be checked for the actual target.
- A conflict counter drop without a time drop can mean another stage is dominant.
- Include synchronization between tile loads and transposed stores.
- Test rectangular and partial tiles to detect out-of-bounds accesses and missing elements.

#### L12. Reduction tree versus serial chain
- Contract: sum 1,024 values, ignoring differences in floating-point association for the work count.
- A serial chain performs 1,023 adds with a dependency depth of 1,023.
- An ideal balanced tree performs the same 1,023 adds with depth 10.
- The tree exposes parallel work but requires communication and synchronization between levels.
- If an illustrative dependent-add latency is four cycles, the serial chain has a 4,092-cycle dependency bound.
- The same simplistic latency model gives a 40-cycle arithmetic depth for the ideal tree.
- Neither number includes instruction issue, communication, data movement or scheduling.
- A block-wide reduction may require additional stages after individual warp reductions.
- Compare eligible warps and dependency stalls, not occupancy alone.
- Test exact integers and cancellation-prone floating values against the stated error tolerance.
- Nondeterministic atomic reductions may additionally violate a reproducibility requirement.

#### L13. Fusion and a cached intermediate
- Contract: two elementwise stages produce and consume an N-element FP32 intermediate.
- If the intermediate crosses DRAM in both directions, eliminating it saves 8N DRAM bytes.
- At N = 10,000,000 this hypothetical saving is 80 MB.
- At a valid 2 TB/s bandwidth limit, that byte difference represents 40 microseconds of ideal transfer time.
- If the intermediate stays in cache, actual DRAM savings can be much smaller.
- A fused version can increase register lifetime and introduce local-memory spill traffic.
- Suppose fusion removes 80 MB but adds 20 MB of spills at the same boundary.
- The net traffic reduction is then 60 MB, corresponding to 30 microseconds at 2 TB/s.
- These are differences in bandwidth terms, not guaranteed differences in measured execution time.
- Measure the full two-stage range and inspect both device-memory and cache-level traffic.
- Validate any changed rounding caused by keeping intermediates in registers.

#### L14. Recompute instead of load
- Suppose one output requires either loading a four-byte value or recomputing it with eight extra FLOPs.
- Let the original workload have 100 million useful FLOPs and transfer 100 million bytes.
- Its useful intensity is 1 FLOP/byte and, at 1 TB/s, its bandwidth roof is 1 TFLOP/s.
- For ten million eliminated loads, traffic falls by 40 million bytes under the assumed boundary behavior.
- The new useful intensity is 100/60 = 1.667 useful FLOP/byte.
- Executed arithmetic rises to 180 million FLOPs; executed intensity is 180/60 = 3 FLOP/byte.
- At a compatible 2 TFLOP/s executed-compute ceiling, the added compute has a 90 microsecond bound.
- The new transfer bound is 60 microseconds, while the original transfer bound was 100 microseconds.
- The simplified lower bound therefore moves from 100 to 90 microseconds, not to 60.
- If the eliminated load hit in cache, the assumed 40 MB DRAM saving is invalid.
- Compare useful outputs per second and correctness, not executed FLOP/s alone.

#### L15. Tensor padding overhead
- Contract: an M by K times K by N product with dimensions M=N=K=100.
- Suppose the chosen implementation executes dimensions padded independently to 128.
- Useful conventional work is 2*100^3 = 2,000,000 FLOPs.
- Executed tile work under this simplified padding model is 2*128^3 = 4,194,304 FLOPs.
- Useful work is about 47.68% of executed work.
- If execution takes 100 microseconds, executed throughput is about 41.94 GFLOP/s.
- The same interval yields only 20 GFLOP/s of useful throughput.
- A second algorithm with less padding can win despite a lower executed-operation rate.
- Padding may also change traffic, masking, workspace size and epilogue cost.
- Compare complete output-equivalent kernels for the actual shape distribution.
- Check that padded values cannot contribute to valid outputs.

#### L16. Mixed-precision input and accumulation
- Contract: a dot product reads two FP16 vectors and accumulates in FP32.
- Assuming no conversion traffic and one read of each element, input payload is 4N bytes.
- Conventional useful work approaches 2N FLOPs, giving approximately 0.5 FLOP/byte.
- The corresponding FP32-input calculation reads 8N bytes and approaches 0.25 FLOP/byte.
- At 1 TB/s the respective bandwidth roofs are approximately 0.5 and 0.25 TFLOP/s.
- These examples assume the same selected boundary and omit the final scalar write.
- Conversion instructions and accumulation precision must match the chosen compute ceiling.
- Lower input precision can change underflow, overflow and cancellation behavior.
- If conversion is performed in a separate kernel, count its reads, writes and launch.
- Measure numerical error on representative and adversarial input distributions.
- A speedup that violates the error contract is not an acceptable optimization.

#### L17. Cache-specific intensity
- A workload performs 400 GFLOP while 100 GB crosses L2 and 20 GB crosses DRAM.
- Its L2 intensity is 4 FLOP/byte, while its DRAM intensity is 20 FLOP/byte.
- Assume L2 bandwidth 8 TB/s, DRAM bandwidth 2 TB/s and compatible compute 80 TFLOP/s.
- The L2 diagonal is 32 TFLOP/s and the DRAM diagonal is 40 TFLOP/s.
- The tightest of these assumed roofs is 32 TFLOP/s.
- The corresponding lower time bounds are 12.5 ms for L2 and 10 ms for DRAM.
- Compute contributes a 5 ms lower bound, so the combined maximum is 12.5 ms.
- Multiplying DRAM intensity by L2 bandwidth would mix boundaries and incorrectly produce 160 TFLOP/s.
- A working set may have a high DRAM intensity because cache reuse filters repeated accesses.
- Inspect both counter definitions and ensure the traffic covers the same range as the FLOP count.
- Neither diagonal proves that the actual workload is close to saturating its corresponding interface.

#### L18. Latency and outstanding requests
- Suppose a memory interface can sustain 1 TB/s and each request moves 32 bytes.
- In an illustrative model, average request latency is 400 ns and the pipeline is otherwise unconstrained.
- Little's law implies 1e12*400e-9 = 400,000 bytes must be in flight to sustain that rate.
- At 32 bytes per request, this corresponds to 12,500 concurrent requests across the modeled interface.
- If only 2,500 requests can be outstanding, the latency-limited rate is at most 200 GB/s in that model.
- This is 20% of the assumed bandwidth ceiling despite the workload's low intensity.
- Real requests may be merged, sized differently or served from different hierarchy levels.
- More independent warps or independent loads per warp may increase concurrency.
- Unrolling can also increase register use and reduce the number of resident warps.
- Vary independent work while monitoring eligible warps and outstanding transactions.
- A flat response suggests another limitation or an already-saturated concurrency resource.

#### L19. Tile edge utilization
- Suppose an output matrix is 100 by 100 and is covered with 32 by 32 output tiles.
- Four tiles are needed in each dimension, giving 16 tiles and 16,384 tile positions.
- Only 10,000 output positions are valid, so positional utilization is about 61.04%.
- The remaining 6,384 positions must be masked or excluded from execution.
- A 16 by 16 tile requires seven tiles per dimension, or 12,544 positions.
- Its positional utilization is about 79.72%, but it may have less data reuse.
- The fraction of valid positions is not automatically the fraction of useful instructions.
- Some hardware instructions execute complete tiles even when stores are masked.
- Smaller tiles can require more scheduling, more halo loads or more reduction slices.
- Compare executed work, byte traffic, register use and latency across the same matrix shapes.
- Test dimensions just below and above tile boundaries for indexing and performance cliffs.

#### L20. Pipeline overlap and lower bounds
- Suppose a range needs 4 ms of transfer and 6 ms of computation under their respective ceilings.
- With ideal overlap, the lower bound is max(4, 6) = 6 ms.
- With strictly sequential phases and the same phase durations, their sum is 10 ms.
- Partial overlap can produce any duration between those two idealized cases, before other overhead.
- A measured 8 ms is consistent with 2 ms of effective overlap in this simplified example.
- But a timeline is needed to establish overlap rather than altered phase durations.
- Chunking can introduce startup and drain bubbles even when steady-state stages overlap.
- Double buffering consumes additional memory and needs correct producer-consumer synchronization.
- Concurrent stages can contend for the same bandwidth, changing both durations.
- Compare the whole range, not the isolated stage times alone.
- Include data-dependency events and output visibility in the correctness contract.

#### L21. Amdahl limit for kernel optimization
- Suppose a request takes 10 ms, of which the target kernel accounts for 2 ms.
- If that kernel is made twice as fast and the rest is unchanged, total time becomes 9 ms.
- Overall speedup is 10/9, approximately 1.111, rather than 2.
- Even eliminating the entire 2 ms kernel would leave 8 ms and cap speedup at 1.25.
- If stages overlap, their isolated durations cannot be summed as independent request fractions.
- Use critical-path timing to identify the fraction that actually contributes to end-to-end latency.
- For a batch-throughput objective, queueing and pipeline utilization may change the relevant denominator.
- A local roofline improvement can be real while the customer-visible result remains negligible.
- Compare complete request latency under the same batching and concurrency.
- Check whether optimization changes accuracy, allocation or preprocessing in other stages.
- Report the timing distribution rather than extrapolating from a single fast sample.

#### L22. Bandwidth comparison and uncertainty
- Suppose an illustrative mixed-access reference transfers 120 GB in 0.06 s.
- Its empirical bandwidth is 2 TB/s for that particular footprint and access mix.
- A kernel transferring 60 GB in 0.04 s achieves 1.5 TB/s.
- The ratio to the reference is 75%, not proof that the theoretical hardware maximum is 75% utilized.
- If the 60 GB byte count has a 5% uncertainty, its implied bandwidth interval is 1.425 to 1.575 TB/s.
- This interval holds time fixed and is not a statistical confidence interval.
- A measured rate above the reference can be legitimate if the access mix differs.
- Cache hits can also make source-level requested bytes exceed physical DRAM traffic.
- Record read/write direction, working-set size, concurrency and clock state for the reference.
- Repeat paired baseline and variant trials to reduce sensitivity to drift.
- Do not mix profiler-replay duration with unprofiled application time.

#### L23. Work inflation and useful throughput
- Suppose two kernels compute the same 1 billion useful FLOPs.
- Variant A executes 1 billion FLOPs in 2 ms, yielding 0.5 TFLOP/s useful and executed.
- Variant B executes 2 billion FLOPs in 3 ms, yielding 0.667 TFLOP/s executed.
- Variant B's useful rate is only 0.333 TFLOP/s and its latency is 50% higher.
- Its larger executed rate therefore does not indicate a better implementation.
- Redundant work can be introduced by padding, recomputation or alternative algorithms.
- If extra arithmetic reduces traffic enough, it can still be worthwhile on a different workload.
- Apply the compute ceiling to the operation convention used by that ceiling.
- Apply the decision metric to output-equivalent work and its complete elapsed time.
- For changed algorithms, confirm mathematical equivalence and error tolerance first.
- Keep the same input distribution and result-consumption requirements for both variants.

#### L24. Clock and precision sensitivity
- Let a modeled kernel require 120 GFLOPs and 30 GB of traffic.
- With C = 60 TFLOP/s and B = 2 TB/s, compute and memory bounds are 2 ms and 15 ms.
- The maximum is 15 ms, so changing only C cannot lower this particular bound.
- If traffic falls to 3 GB, the memory bound becomes 1.5 ms and compute becomes the tighter bound.
- Halving C to 30 TFLOP/s then raises the compute bound to 4 ms.
- These are sensitivity calculations, not claims about a named accelerator's clock behavior.
- Different precision or instruction families require their own compatible compute ceilings.
- A clock reduction can also change bandwidth, so holding B fixed is a modeling assumption.
- Compare actual code generation and clock state before attributing a shift to arithmetic throughput.
- On a short kernel, fixed launch overhead can dominate either lower bound.
- Verify the precision change against range, rounding and exceptional-value requirements.

#### L25. Persistent versus ordinary scheduling
- Suppose 10,000 independent tasks each require 1,000 units of useful work.
- Total useful work is 10 million units regardless of whether blocks persist.
- Ordinary scheduling distributes tasks through multiple blocks; a persistent kernel loops over tasks.
- Persistent blocks can reduce repeated setup but occupy resources for longer.
- A global queue may introduce atomics, contention and unpredictable task ordering.
- For task durations of 1 and 9 microseconds, equal task counts do not imply equal work.
- Dynamic assignment can improve balance if queue overhead is smaller than the avoided idle time.
- Compare per-block completion times and queue contention, not just the grid dimension.
- Ensure enough persistent blocks exist to cover the device without blocking dependent work.
- A cooperative synchronization scheme must respect its actual residency and launch requirements.
- Check that all tasks execute exactly once and that any ordering constraints remain valid.

#### L26. Circular-buffer pipeline depth
- Suppose loading a tile takes 8 time units and consuming it takes 4 time units.
- An ideal steady-state pipeline cannot finish tiles faster than one per 8 units.
- If instead loading takes 4 and compute takes 8, the compute stage sets the same 8-unit interval.
- More buffers cannot overcome the slower stage's sustained service time.
- Additional buffering can hide latency when one buffer would otherwise leave a stage idle.
- If one tile buffer costs 16 KiB, double buffering requires 32 KiB per block.
- Triple buffering would require 48 KiB before other shared-memory allocations.
- Larger per-block allocation can reduce active blocks and offset improved overlap.
- Startup, drain and synchronization costs matter especially for few tiles.
- Inspect producer and consumer waits and compare several depths for representative K sizes.
- Verify that a buffer is not overwritten before all consumers finish reading it.

#### L27. Warp specialization tradeoff
- Divide a hypothetical block into producer and consumer warps with explicit tile handoff.
- A producer-only warp does not directly contribute consumer arithmetic while it is producing data.
- If a block has four warps and one is reserved for production, three remain for the consumer stage.
- That 3/4 share is a scheduling description, not a throughput prediction.
- Specialization can help when overlapping stages outweighs synchronization and reduced consumer capacity.
- It can hurt when all warps would otherwise have sufficient independent memory and arithmetic work.
- Compare stage-level issue activity and wait time for specialized and unspecialized implementations.
- Changing the producer count also changes register and shared-memory pressure.
- Correct handoff requires visibility and ordering supported by the actual synchronization primitive.
- A producer that exits early can strand consumers at a barrier or leave a tile uninitialized.
- Test short reductions and irregular final tiles as well as long steady-state runs.

#### L28. Host-device transfer as an independent boundary
- Suppose a request transfers 400 MB to a device over an assumed 25 GB/s link.
- The ideal transfer bound is 0.4/25 = 0.016 s, or 16 ms.
- A 2 ms device kernel cannot reduce the whole request below the link-bound term by optimizing compute alone.
- Returning another 100 MB costs at least 4 ms if the same bandwidth applies and phases do not overlap.
- With mandatory serial transfer directions, the two modeled transfer terms total 20 ms.
- Concurrent bidirectional bandwidth can differ from the separate one-direction reference.
- Pinned memory, staging, setup overhead and transfer granularity affect actual behavior.
- Batching can amortize overhead but may increase latency and memory footprint.
- Reusing device-resident inputs can eliminate transfers only when application semantics permit it.
- Measure end-to-end synchronization, not just the asynchronous copy enqueue duration.
- Check that the result is visible on the host before recording request completion.

#### L29. Interpreting an apparent roof violation
- Suppose work is 1 TFLOP and a range takes 10 ms, giving 100 TFLOP/s.
- A proposed compute roof is 60 TFLOP/s, so the point appears above the roof.
- If the 1 TFLOP count includes ten identical invocations while time covers one, scope is inconsistent.
- If the count is useful work but the ceiling applies to a different precision, the comparison is also invalid.
- If 60 TFLOP/s is merely a microbenchmark reference, a different instruction mix may exceed it.
- A bit-per-second bandwidth figure mistaken for bytes per second introduces an eightfold unit error.
- Asynchronous host timing can omit most of the device execution interval.
- Check operation count, launch count, precision, interval, units and reference construction in that order.
- Do not clamp the point to the roof; resolve the inconsistent assumption.
- A strict compatible upper bound genuinely exceeded means at least one premise is false.
- Preserve the raw counter and timing records for independent reconciliation.

#### L30. Shape-weighted objective
- Suppose 90% of requests use shape A and 10% use shape B, with baseline latencies 1 ms and 10 ms.
- The arithmetic mean latency under this request mix is 0.9*1 + 0.1*10 = 1.9 ms.
- A variant changes A to 0.8 ms but B to 15 ms.
- Its mean becomes 0.9*0.8 + 0.1*15 = 2.22 ms, a regression in this objective.
- A large improvement on frequent small shapes does not automatically offset a rare large regression.
- The 90th, 95th or 99th percentile cannot be inferred from these means alone.
- Batch composition, concurrency and queueing can change the actual latency distribution.
- Use the observed workload mix rather than a convenient single matrix size.
- Compare accuracy and output semantics for both shapes, including padding and tails.
- Report per-shape timings alongside the aggregate so the tradeoff remains visible.
- Reevaluate weights when the workload distribution changes.

#### L31. Layer normalization with separate passes
- Contract: normalize each row of 1,024 FP32 elements using its mean and population variance.
- The mathematical mean is the sum divided by 1,024; variance is the mean squared deviation.
- A naive three-pass implementation reads the row once for each statistic and once to normalize.
- Ignoring parameter vectors and scalar statistics, those reads total 12,288 bytes per row.
- Writing 1,024 FP32 normalized values adds 4,096 bytes, for 16,384 payload bytes.
- An implementation that retains the row and writes once needs 8,192 payload bytes in the ideal case.
- At an assumed 1 TB/s, the respective payload transfer terms are 16.384 ns and 8.192 ns per row.
- Register spills or shared-memory staging can erode that theoretical traffic saving.
- A cache may already service later reads, so a lower source-level load count need not halve DRAM traffic.
- Compare DRAM bytes, cache-level bytes and spill instructions for the complete normalization.
- Ensure epsilon placement, accumulation precision and variance definition match the reference.
- Test constant rows, extreme magnitudes and N=1 rather than only random typical inputs.

#### L32. Softmax and stable exponentials
- Contract: for each row, compute exp(x_i - max(x)) divided by the sum of those exponentials.
- The subtraction of the maximum prevents positive exponential overflow for finite row inputs.
- A row of 2,048 FP32 values occupies 8,192 bytes; output occupies another 8,192 bytes.
- Reading the row twice and writing the result gives 24,576 bytes of modeled payload.
- Reading it once while retaining all necessary values would lower the ideal payload to 16,384 bytes.
- At an assumed 2 TB/s, those payloads correspond to 12.288 ns and 8.192 ns respectively.
- Exponential, division, reduction and synchronization costs can dominate those transfer terms.
- FLOP conventions rarely price transcendental instructions consistently with a simple FMA ceiling.
- Inspect the instruction mix and use an appropriate throughput reference for the exponential path.
- Compare long and short rows to separate launch and reduction costs from steady-state traffic.
- Define behavior explicitly for rows containing infinities, NaNs or only masked positions.
- Verify both output tolerance and the intended normalization over unmasked elements.

#### L33. Split-K matrix multiplication
- Contract: partition the K dimension into four independent partial matrix products.
- For M=N=1,024, each FP32 partial output contains 4,194,304 bytes.
- Writing and rereading four partial matrices contributes 33,554,432 bytes before the final output write.
- Writing the final 1,024 by 1,024 FP32 matrix adds 4,194,304 bytes.
- The partial-output plus final-output payload is therefore 37,748,736 bytes.
- At an assumed 2 TB/s, this payload alone implies at least 18.874368 microseconds of transfer time.
- The unsplit ideal output write alone costs 4,194,304 bytes when beta is zero.
- Split-K may expose more blocks when M and N are too small to occupy the device.
- It also introduces a reduction, extra traffic and possibly an additional kernel launch.
- Compare device occupancy and elapsed time at several K values with M and N fixed.
- Include all partial-workspace traffic at the same boundary as the bandwidth reference.
- Reassociation across split partitions can change results, requiring numerical validation.

#### L34. Matrix multiplication epilogue
- Contract: compute an FP32 matrix product followed by bias addition and an activation.
- For a 1,024 by 1,024 output, one FP32 matrix occupies 4,194,304 bytes.
- A separate activation that rereads and rewrites the output adds 8,388,608 payload bytes.
- If the intermediate crosses DRAM in both directions, this is 8.388608 MB of additional DRAM traffic.
- At an assumed 2 TB/s, the additional traffic term is 4.194304 microseconds.
- A fused epilogue can avoid that intermediate transfer but may increase accumulator live ranges.
- Bias traffic depends on layout, reuse, cache behavior and whether the bias is per row or column.
- The activation's comparison, exponential or polynomial instructions require their own accounting.
- An activation applied before the accumulator is rounded can produce a different numerical result.
- Check register use and spill traffic before concluding that fusion reduced total movement.
- Compare the entire product-plus-activation range rather than the GEMM kernel in isolation.
- Test negative zero, infinities and NaNs if the activation contract specifies their behavior.

#### L35. Batched small matrix products
- Contract: multiply 1,000 independent pairs of 16 by 16 FP32 matrices with beta zero.
- Each product has approximately 2*16^3 = 8,192 conventional useful FLOPs.
- The batch therefore contains 8,192,000 useful FLOPs.
- Reading both inputs and writing one output costs 3*16^2*4 = 3,072 bytes per product.
- The batch payload is 3,072,000 bytes, and ideal intensity is 8/3 FLOP/byte.
- At 1.5 TB/s, the bandwidth diagonal is 4 TFLOP/s under these assumptions.
- The corresponding ideal payload transfer time is 2.048 microseconds for the full batch.
- Launching 1,000 separate kernels instead of one batched operation can change the timing regime.
- A batched library may use padding or workspace, so count its actual executed work and traffic.
- Compare one batched launch with separate launches under the same output and synchronization contract.
- Check strides and inter-matrix gaps; contiguous matrices are not guaranteed by the word batched.
- Include irregular batch counts and nonstandard leading dimensions in correctness tests.

#### L36. Vectorized memory operations and alignment
- Consider 32 lanes each loading four consecutive FP32 values, for 512 requested bytes.
- Assume the warp's requested region is contiguous and the modeled transfer unit is 32 bytes.
- A 32-byte-aligned region occupies 16 sectors, or 512 transferred bytes.
- Shifting that region by four bytes makes it span 17 sectors, or 544 transferred bytes.
- The modeled transaction amplification is 544/512 = 1.0625, or 6.25%.
- This sector calculation assumes no sector was already resident at the selected boundary.
- Vectorized instructions can reduce instruction count without changing the physical payload.
- A misaligned vector instruction may require a different implementation or violate its alignment contract.
- Compare generated load instructions and sector counts for aligned and offset pointers.
- Handle the remaining elements explicitly when N is not divisible by the vector width.
- Verify that vector loads do not cross an allocation boundary, even when stores are masked.
- Check aliasing assumptions before changing scalar code to a wider vectorized form.

#### L37. Atomic aggregation by warp
- Suppose a warp of 32 lanes increments counters for 32 keys.
- If all lanes share one key, naive execution issues 32 logical increments to that key.
- A correct warp aggregation can combine them into one atomic add of 32.
- If all keys are distinct, aggregation cannot reduce the number of logical key updates.
- For eight distinct keys each repeated four times, ideal grouping needs eight atomic adds.
- The actual gain depends on grouping instructions, contention and hardware atomic behavior.
- A count of issued atomic operations is not the same thing as DRAM transaction count.
- Compare all-equal, eight-key and all-distinct input distributions with identical N.
- If only the all-equal case improves, contention reduction is a plausible explanation.
- Grouping must respect active-lane masks for partial warps and divergent control flow.
- Integer addition can overflow; use the counter width and overflow semantics required by the contract.
- For floating-point updates, reassociation can change the final value or reproducibility.

#### L38. Convolution halo cost
- Contract: compute a 16 by 16 output tile using a 3 by 3 stencil with stride one.
- With no padding inside the input tile, the tile needs an 18 by 18 input region.
- That region contains 324 input values for 256 output values.
- FP32 input payload is 1,296 bytes and FP32 output payload is 1,024 bytes.
- If each input tile value is loaded once, total payload is 2,320 bytes per tile.
- Nine multiplications and eight additions per output give 4,352 useful FLOPs.
- The ideal tile-local intensity is 4,352/2,320 ≈ 1.876 FLOP/byte.
- Neighboring output tiles may reload overlapping halos, while caches may suppress DRAM reloads.
- The count excludes filter coefficients; add them at the boundary where they actually cross.
- A larger tile reduces halo fraction but consumes more shared memory and may lower residency.
- Compare the 16 by 16 case with actual edge tiles and padding rules.
- Validate boundary handling and coefficient layout against a direct reference implementation.

#### L39. Hierarchy counter reconciliation
- Suppose a range requests 8 GB of useful load payload at the instruction level.
- The profiler reports 12 GB of L1-to-L2 request traffic and 4 GB of DRAM read traffic.
- The 12/8 ratio is 1.5, but it is meaningful only if counter definitions cover the same loads.
- The DRAM-to-L2 ratio is 4/12 = 1/3 under the simplified matching-scope assumption.
- It is not automatically an L2 miss rate: sectors, writes, prefetches and aggregation can differ.
- If the range also writes 2 GB to DRAM, total modeled DRAM traffic is 6 GB.
- For 24 GFLOPs in 2 ms, achieved throughput is 12 TFLOP/s.
- Its DRAM intensity is 24/6 = 4 FLOP/byte and achieved DRAM bandwidth is 3 TB/s.
- Their product, 4 FLOP/byte times 3 TB/s, reproduces the 12 TFLOP/s throughput.
- Compare read and write counters independently before interpreting an aggregate.
- Check units, counter replay behavior and inclusion of other kernels in the range.
- If the reported counters are sampled or estimated, retain their uncertainty with the result.

#### L40. Interview: distinguishing two competing hypotheses
- Question: a gather kernel reaches 15% of a matched DRAM reference; what should be changed first?
- A correct answer does not infer saturation merely from a low arithmetic intensity.
- Hypothesis one is insufficient memory-level parallelism due to dependent index loads.
- Hypothesis two is excessive transferred sectors caused by scattered lane addresses.
- For the first hypothesis, vary independent gathers per thread while tracking registers and eligible warps.
- For the second, hold index values constant but reorder independent work to improve spatial locality.
- If sector bytes fall and time falls while outstanding requests stay similar, locality gains support hypothesis two.
- If eligible requests rise and time falls with similar bytes, greater concurrency supports hypothesis one.
- Either experiment can change cache residency, so inspect both cache and DRAM boundary counters.
- A third possibility is that the launch is too short for a meaningful steady-state comparison.
- Reordering must preserve duplicate-index semantics and restore the requested output order.
- Full credit explains what observation would falsify the proposed mechanism.

**Primary source:** Williams, Waterman and Patterson, [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785), *Communications of the ACM*, 2009.
The original paper defines operational intensity using DRAM traffic after cache filtering; the hierarchy-specific variants here explicitly name other boundaries.
The directory crosswalk also lists CUDA kernel-design topics on warp specialization, pipelining, circular buffering and persistent scheduling; filenames alone do not establish their unseen contents.

## 9. Complexity & Cost Analysis

For a dense matrix product with dimensions M by K and K by N, the mathematical inner products require approximately 2*M*N*K FLOP under the two-FLOP FMA convention.
Validity: dense arithmetic, ignoring a possible difference in the initial accumulation and ignoring epilogue work.
The exact count depends on the implementation and whether initialization is counted.
```text
FORMULA [R7]: W_gemm approximately 2*M*N*K
WHERE: M, N, K positive integer dimensions; W_gemm [FLOP].
CONSTRAINT: dense multiply-add work; no padding or epilogue included.
NUMERICAL EXAMPLE: M=N=K=1024 gives 2,147,483,648 FLOP.
INTERPRETATION: the cubic work count is a useful-work estimate, not an instruction count.
```
For one ideal read of each FP32 input and one write of a new FP32 output, the compulsory payload model is as follows.
Validity: contiguous dense matrices, four-byte elements, no preexisting output read, and each input crosses the chosen boundary once.
```text
FORMULA [R8]: Q_ideal = 4*(M*K + K*N + M*N)
WHERE: Q_ideal [bytes]; dimensions as in R7; each element is 4 bytes.
CONSTRAINT: one input read and one output write, no workspace or extra transactions.
NUMERICAL EXAMPLE: M=N=K=1024 gives 12,582,912 bytes = 12 MiB.
INTERPRETATION: this is ideal payload accounting, not measured traffic.
```
For that illustrative shape, R1 gives 2,147,483,648 / 12,582,912 = 170.666... FLOP/byte.
Check: 12,582,912 * 170.666... = 2,147,483,648.
The ideal calculation excludes reloading, caches, padding, workspace, beta-times-existing-output and epilogue operations.
At a particular memory boundary, data already resident there may make its traffic lower than the cold input/output payload.
Conversely, reloading or inefficient transactions may increase observed traffic.
With beta nonzero, reading the previous output must be considered in the selected boundary accounting.
With smaller or rectangular matrices, the work-to-payload ratio changes.
For M=N=K=1, R7 gives 2 FLOP and R8 gives 12 bytes, or 1/6 FLOP/byte.
A single-element workload is also highly sensitive to launch and latency costs.
The multiplication needs storage for the input and output matrices; extra space depends on the algorithm and implementation.
Tiling can reduce repeated transfers but consumes per-block shared memory and registers.
Larger tiles can reduce active blocks and impair latency hiding.
Fusion can eliminate intermediate transfers while increasing live state.
Profiler collection costs time and can perturb cache and scheduling.
Total engineering cost also includes correctness, maintenance, deployment, and device-specific tuning.
For a pipeline, count each stage's launch, intermediate bytes and synchronization along the critical path.
Do not infer end-to-end cost from a single isolated kernel's roofline point.
**Takeaway:** Complexity describes scaling; only scoped measurement establishes delivered latency or throughput.

## 10. Key Theorems / Results Reference

- **DERIVED R1:** intensity is work divided by bytes at a named boundary.
- **DERIVED R2:** throughput equals intensity times achieved boundary bandwidth for matching intervals.
- **DERIVED R3:** the minimum of compatible compute and bandwidth ceilings bounds throughput.
- **DERIVED R4:** the balance point is the compatible compute ceiling divided by bandwidth.
- **DERIVED R5:** multiple independently valid hierarchy bounds can be intersected.
- **DERIVED R6:** the ideal execution time is bounded below by the larger compute or transfer term.
- **DERIVED R7–R8:** dense GEMM work and ideal payload follow from element and inner-product counts.
- **SOURCED:** NVIDIA [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), v2026.4.0, accessed 2026-11-07: tool sections include hierarchy roofline charts, scheduler and memory analysis.
- The same source documents collection overhead and kernel/application/range replay limitations.
- The guide is tool documentation, not evidence for the illustrative hardware ceilings or benchmark outcomes here.
- **OPEN:** exact metrics, replay support and profiler UI must be checked for the installed profiler, device, driver and toolkit version.
- **OPEN:** no device-specific roof, peak, achieved result, workload or numerical tolerance has been measured here.
- Recheck the guide on profiler upgrade or when a claimed metric or collection mode changes.
**Takeaway:** Source tool behavior separately from mathematical derivations and unmeasured workload assumptions.

## 11. Interview / Exam Questions

All numerical inputs below are **ILLUSTRATIVE**. They test the stated conventions; none is a device measurement.

### Q01 / Level 1 / boundary definition
**Prompt and given facts:** Define arithmetic intensity for a kernel that performs work while moving data through a cache hierarchy.
**Clarifications and acceptable assumptions:** Name one boundary, such as HBM, before reporting bytes.
**Answer:** `I_h = W / Q_h`: useful work `W` divided by bytes `Q_h` crossing named boundary `h`. A source-level load is not automatically an HBM byte because it can hit in cache. A divide-by-zero byte count is invalid, not “infinite performance.”
**Alternative answers and when defensible:** Executed rather than useful FLOPs are defensible only if explicitly reported as an instruction-efficiency analysis.
**Follow-up:** How does a warm cache change the statement? It can change `Q_h`, so retain the cache-state protocol.
**Trap/red flag and why it is wrong:** “FLOPs divided by input size” silently substitutes a payload estimate for a measured boundary.
**Scoring:** 0 undefined; 1 ratio without boundary; 2 correct definition; 3 explains cache distinction.
**Answer source:** Sections 3 and equation R1.

### Q02 / Level 1 / compatible roof
**Prompt and given facts:** State the roofline bound and its most important validity condition.
**Clarifications and acceptable assumptions:** Compute and bandwidth ceilings must use the same precision, operation convention and workload interval.
**Answer:** `P <= min(C, B_h * I_h)`. `C` is a compatible compute ceiling and `B_h` is bandwidth at the same boundary. Dimensional analysis gives `(byte/s)*(FLOP/byte)=FLOP/s`.
**Alternative answers and when defensible:** Multiple hierarchy roofs are useful when each has separately measured traffic and a compatible ceiling.
**Follow-up:** Does touching a roof prove output correctness? No; throughput and numerical correctness are separate contracts.
**Trap/red flag and why it is wrong:** Combining an FP16 tensor ceiling with FP32-counted work creates a meaningless comparison.
**Scoring:** 0 wrong relation; 1 names two roofs; 2 states minimum and compatibility; 3 gives units and boundary condition.
**Answer source:** Section 4, equations R2–R3.

### Q03 / Level 2 / checked calculation
**Prompt and given facts:** A fictional kernel performs 24 GFLOP and transfers 6 GB across its stated HBM boundary in 2 ms. Find intensity, achieved throughput and achieved bandwidth.
**Clarifications and acceptable assumptions:** Use decimal giga units consistently; the traffic includes reads and writes for the measured range.
**Answer:** `I=24/6=4 FLOP/byte`; `P=24 GFLOP/0.002 s=12 TFLOP/s`; `B=6 GB/0.002 s=3 TB/s`. Sanity check: `I*B=4*3=12 TFLOP/s`.
**Alternative answers and when defensible:** Binary units are acceptable only if every input and resulting rate is converted consistently.
**Follow-up:** If traffic is actually L2 bytes, what changes? Label the intensity L2-specific and compare it with an L2-compatible ceiling.
**Trap/red flag and why it is wrong:** Dividing 24 GFLOP by 2 ms and calling it bandwidth confuses work rate with byte rate.
**Scoring:** 0 incorrect units; 1 one correct ratio; 2 all three values; 3 includes the independent product check.
**Answer source:** Sections 4–5 and laboratory L39.

### Q04 / Level 2 / balance point
**Prompt and given facts:** With a fictional compatible compute ceiling of 60 TFLOP/s and bandwidth ceiling of 2 TB/s, find the balance intensity and classify a 4 FLOP/byte kernel.
**Clarifications and acceptable assumptions:** Ceilings use decimal units and the same operation count.
**Answer:** `I_balance=C/B=60/2=30 FLOP/byte`. The 4 FLOP/byte point has a bandwidth roof of 8 TFLOP/s, below 60, so the *model* is bandwidth-limited.
**Alternative answers and when defensible:** “Potentially bandwidth-limited” is preferable when the byte count or attainable bandwidth has not been established.
**Follow-up:** What if achieved bandwidth is only 0.5 TB/s? The attained transfer rate predicts 2 TFLOP/s, which calls for diagnosis rather than a new theoretical roof.
**Trap/red flag and why it is wrong:** Declaring 4 FLOP/byte universally memory-bound ignores the device, precision and boundary.
**Scoring:** 0 wrong division; 1 correct number only; 2 correct classification; 3 states its conditional nature.
**Answer source:** Equation R4 and Section 5.

### Q05 / Level 2 / traffic model
**Prompt and given facts:** Derive the ideal FP32 payload for dense `M×K` by `K×N` GEMM with one read of each input and one new output write.
**Clarifications and acceptable assumptions:** Four bytes per value; exclude workspace, padding and beta-times-existing-output.
**Answer:** Element count is `MK+KN+MN`; multiplying by four yields `Q_ideal=4(MK+KN+MN)` bytes. For `M=N=K=1024`, this is `12,582,912` bytes or 12 MiB. The work estimate is `2MNK=2,147,483,648 FLOP`.
**Alternative answers and when defensible:** Including a C read is correct when the epilogue computes nonzero beta; state that changed contract.
**Follow-up:** Why can measured HBM bytes exceed this? Reloads, transaction inefficiency, workspace and other operations can cross the boundary.
**Trap/red flag and why it is wrong:** Calling the payload a benchmark bandwidth measurement turns an ideal lower-bound model into evidence.
**Scoring:** 0 omits an operand; 1 correct count; 2 correct bytes and qualification; 3 names beta/caching boundary.
**Answer source:** Section 9, equations R7–R8.

### Q06 / Level 3 / ambiguous diagnosis
**Prompt and given facts:** A gather kernel is at 15% of a matched DRAM reference and has low intensity. Choose the first diagnostic experiment.
**Clarifications and acceptable assumptions:** The kernel is correct; both dependent-index latency and scattered sectors are plausible.
**Answer:** Do not tune from intensity alone. Hold logical work fixed and separately vary independent gathers per thread (tests memory-level parallelism) and index ordering (tests locality). Record eligible work, sector bytes, cache/DRAM bytes and unprofiled time.
**Alternative answers and when defensible:** A timeline/latency experiment is defensible if it preserves the workload and exposes dependency chains.
**Follow-up:** What signal supports locality? Sector traffic and time fall while outstanding-request opportunity remains similar.
**Trap/red flag and why it is wrong:** “It is bandwidth-bound, so vectorize” assumes saturation and may violate alignment or fail to address dependencies.
**Scoring:** 0 one-cause assertion; 1 proposes profiling; 2 separates hypotheses; 3 states falsifying signals.
**Answer source:** Section 8, laboratory L40.

### Q07 / Level 3 / fusion selection
**Prompt and given facts:** A candidate fusion removes a 64 MiB intermediate but increases register use and changes reduction order. Should it ship?
**Clarifications and acceptable assumptions:** No runtime or parity result has been measured.
**Answer:** Not yet. First define an output tolerance/reference and test ordinary, extreme and edge shapes. Then compare full-range unprofiled latency, resources, spill/local traffic and boundary bytes under identical inputs. Reduced bytes may be offset by spills, lower residency or synchronization.
**Alternative answers and when defensible:** Retaining separate kernels is defensible when parity, maintainability or shape variability dominates a speculative traffic saving.
**Follow-up:** If parity passes but latency worsens at small shapes? Keep the baseline or use a size-gated dispatch only after validating the dispatcher.
**Trap/red flag and why it is wrong:** “Fewer launches always wins” ignores increased live state and launch overhead’s workload dependence.
**Scoring:** 0 ships from byte estimate; 1 requests benchmark; 2 includes correctness and resources; 3 gives a changed-shape decision.
**Answer source:** Sections 7–8.

### Q08 / Level 3 / impossible point
**Prompt and given facts:** A plotted point lies above a selected roof. What must be checked before calling it a breakthrough?
**Clarifications and acceptable assumptions:** The roof was built from a microbenchmark; the application range includes several kernels.
**Answer:** Reconcile FLOP convention, byte boundary, time range, dtype, workload, cache state, counter semantics and whether the empirical roof applies. A point can exceed an irrelevant microbenchmark ceiling without violating physics.
**Alternative answers and when defensible:** Repeating the measurement is useful after scope reconciliation, not as a substitute for it.
**Follow-up:** What if all definitions match? Audit timing/counter collection and revise the ceiling only with documented evidence.
**Trap/red flag and why it is wrong:** Treating every above-roof point as faster-than-possible mistakes a scoped model for a universal law.
**Scoring:** 0 celebrates result; 1 says “units”; 2 lists scope reconciliation; 3 distinguishes theoretical and empirical ceilings.
**Answer source:** Sections 4 and 8.

### Q09 / Level 3 / profiler versus service timing
**Prompt and given facts:** A profiled kernel is 20% faster than the baseline, but unprofiled end-to-end latency has not changed.
**Clarifications and acceptable assumptions:** The profiler may replay work and the service includes queueing, transfers and neighboring kernels.
**Answer:** Preserve the unprofiled baseline as the service evidence. Inspect range coverage, replay behavior, changed cache state and critical-path time; then measure the complete representative request distribution. The kernel observation is a hypothesis, not a deployment result.
**Alternative answers and when defensible:** Profiling a narrower counter set is defensible if it reduces perturbation and tests a named mechanism.
**Follow-up:** If only long requests improve? Report that slice and decide whether its volume justifies complexity.
**Trap/red flag and why it is wrong:** Replacing customer latency with a profiler percentage confuses instrumentation output with the decision metric.
**Scoring:** 0 deploys on profiler result; 1 asks for end-to-end timing; 2 identifies perturbation; 3 ties slice result to rollout decision.
**Answer source:** Sections 8–10.

### Q10 / Level 3 / verification boundary
**Prompt and given facts:** A team claims a tile increase “improved the model” because a roofline estimate rose. What evidence is required to correct the claim?
**Clarifications and acceptable assumptions:** The tile changes memory layout and numerical reduction order.
**Answer:** Separate three claims: the estimate predicts a changed upper bound; correctness requires comparison with a reference and declared tolerance; improvement requires representative unprofiled workload measurement with setup and uncertainty. A higher roof alone proves none of the latter two.
**Alternative answers and when defensible:** If output is bitwise required, use that contract instead of a tolerance; otherwise justify a numerical error metric and acceptance limit.
**Follow-up:** What if the new tile is faster but fails tolerance? Reject it or redesign accumulation; performance cannot waive the output contract.
**Trap/red flag and why it is wrong:** Calling a model bound an achieved benchmark collapses derived, illustrative and measured evidence.
**Scoring:** 0 accepts roofline as proof; 1 asks for timing; 2 separates performance and correctness; 3 states evidence labels and rollback.
**Answer source:** Sections 5, 7 and 10.

**Takeaway:** A defensible roofline answer states a measurable boundary, checks units, proposes a falsifiable diagnosis, and refuses to turn an upper bound into a benchmark.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | CUDA roofline arithmetic intensity and performance diagnosis |
| Core formula | `I_h = W/Q_h`; `P <= min(C, B_h*I_h)` for compatible valid ceilings |
| Key assumption | Work, time, traffic boundary, precision and ceilings describe the same workload |
| Complexity | Work and traffic depend on the kernel; dense matrix multiplication's operation count is described in Section 9 |
| Use when | A compatible bandwidth and compute ceiling can bound the measured workload |
| Avoid when | Byte counts or ceilings refer to different boundaries, formats or workloads |
| Go deeper | [Collection index](INDEX.md); benchmark the actual kernel and device |
| Diagnostic | Below both roofs, investigate parallelism, latency, issue rate and launch costs |
| Evidence | Constructed examples are illustrative, not measured performance |
