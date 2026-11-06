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
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D02. Strided stores
- **Symptom:** adjacent output lanes write distant addresses.
- **Competing cause:** write policy or cache effects instead of stride.
- **Discriminator:** compare store transaction efficiency over two layouts.
- **Bounded intervention:** choose a contiguous output mapping where consumers permit.
- **Verification:** validate downstream layout and whole-range time.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D03. Redundant global loads
- **Symptom:** multiple threads reload the same value.
- **Competing cause:** cache hits may already suppress device-memory traffic.
- **Discriminator:** compare cache-level and device-memory bytes.
- **Bounded intervention:** share or tile reused values only when reuse is confirmed.
- **Verification:** measure all relevant traffic and shared-memory pressure.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D04. Cache warmup bias
- **Symptom:** a repeated launch fits in cache.
- **Competing cause:** the first launch has a colder working set.
- **Discriminator:** compare cold and warm access protocols.
- **Bounded intervention:** represent the intended production reuse explicitly.
- **Verification:** report both distributions and hierarchy traffic.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D05. Write allocation overhead
- **Symptom:** stored values induce more boundary traffic than payload.
- **Competing cause:** the profiler may count a different boundary.
- **Discriminator:** inspect read and write traffic independently.
- **Bounded intervention:** adjust store strategy only if supported and safe.
- **Verification:** confirm output visibility and actual traffic.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D06. Register spilling
- **Symptom:** local memory traffic rises after fusion.
- **Competing cause:** large input footprint alone can increase traffic.
- **Discriminator:** compare compiler resource report and local-memory counters.
- **Bounded intervention:** reduce live ranges or split the stage.
- **Verification:** recheck correctness and end-to-end latency.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D07. Insufficient blocks
- **Symptom:** small grids do not cover the device.
- **Competing cause:** a serial dependency may also suppress utilization.
- **Discriminator:** compare grid size and active wave distribution.
- **Bounded intervention:** batch independent inputs when semantics allow.
- **Verification:** check latency and throughput at production batch sizes.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D08. Uneven tail blocks
- **Symptom:** partial tiles do less useful work.
- **Competing cause:** a shape-specific branch may cause divergence.
- **Discriminator:** inspect work distribution and edge fraction.
- **Bounded intervention:** choose shape-aware tiling or a separate edge path.
- **Verification:** test odd sizes and compare useful output.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D09. Long dependency chain
- **Symptom:** few warps are eligible despite work in flight.
- **Competing cause:** memory stalls can produce a similar signal.
- **Discriminator:** inspect scheduler eligibility with instruction dependencies.
- **Bounded intervention:** increase independent work without increasing spills excessively.
- **Verification:** check generated instructions and execution time.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D10. Barrier imbalance
- **Symptom:** some warps arrive early and wait.
- **Competing cause:** slow memory access may delay all participants.
- **Discriminator:** correlate barrier wait with producer completion.
- **Bounded intervention:** balance work before the synchronization point.
- **Verification:** run race checks and measure barrier stalls.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D11. Shared-memory bank conflicts
- **Symptom:** shared accesses serialize within a warp.
- **Competing cause:** barrier or occupancy issues can look similar.
- **Discriminator:** inspect shared access transactions against requested accesses.
- **Bounded intervention:** change layout or padding with a proven index mapping.
- **Verification:** test boundaries and shared transaction counts.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D12. Host launch overhead
- **Symptom:** many short kernels dominate host-visible latency.
- **Competing cause:** device synchronization may be the real cost.
- **Discriminator:** inspect timeline and per-launch device durations.
- **Bounded intervention:** batch or fuse only when outputs remain equivalent.
- **Verification:** measure host-to-result latency without profiler overhead.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D13. Synchronization on host
- **Symptom:** a host wait interrupts overlap.
- **Competing cause:** copy-engine saturation can also limit throughput.
- **Discriminator:** compare timeline with and without the specific wait.
- **Bounded intervention:** defer only waits not required by data dependencies.
- **Verification:** test cross-stream ordering and end-to-end results.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D14. Transfer bottleneck
- **Symptom:** host-device copies dominate the request.
- **Competing cause:** a long kernel can hide the copy in a timeline.
- **Discriminator:** measure transfer bytes and end-to-end critical path.
- **Bounded intervention:** reduce copies or overlap independent stages.
- **Verification:** validate buffer lifetime and latency distribution.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D15. Wrong precision roof
- **Symptom:** single-precision work is compared with a tensor peak.
- **Competing cause:** generated code may not use the expected instruction family.
- **Discriminator:** inspect precision and executed instruction mix.
- **Bounded intervention:** choose a roof matching executed operations.
- **Verification:** recalculate intensity and report error tolerance.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D16. FMA miscount
- **Symptom:** one instruction is counted as one operation.
- **Competing cause:** tensor tile accounting may also be incomplete.
- **Discriminator:** derive the count from the exact arithmetic convention.
- **Bounded intervention:** use two operations per scalar fused multiply-add consistently.
- **Verification:** reconcile static work and profiler convention.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D17. Mixed useful and executed FLOPs
- **Symptom:** padding increases the reported operation rate.
- **Competing cause:** the faster rate could reflect actual useful output.
- **Discriminator:** compare useful output per second and executed operations.
- **Bounded intervention:** label both numerators separately.
- **Verification:** measure latency and numerical agreement.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D18. Wrong bandwidth unit
- **Symptom:** bytes are mixed with bits or binary units.
- **Competing cause:** wrong interval can also distort the ratio.
- **Discriminator:** write dimensional conversions explicitly.
- **Bounded intervention:** normalize every quantity before plotting.
- **Verification:** recompute with an independent unit check.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D19. Arithmetic overflow
- **Symptom:** integer work-count multiplication wraps.
- **Competing cause:** a format conversion may truncate the result.
- **Discriminator:** calculate dimensions in a wide integer type.
- **Bounded intervention:** use checked multiplication before conversion.
- **Verification:** test dimensions near the representable boundary.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D20. Missing write bytes
- **Symptom:** only input reads appear in the model.
- **Competing cause:** cache behavior may also alter transfer volume.
- **Discriminator:** separate input, output and intermediate traffic.
- **Bounded intervention:** include writes at the selected boundary.
- **Verification:** compare totals with the profiler definition.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D21. Implicit workspace
- **Symptom:** library kernels move an unmodeled workspace.
- **Competing cause:** hidden initialization may have a separate launch.
- **Discriminator:** trace all kernels and allocations in the measured range.
- **Bounded intervention:** include workspace traffic and time in range accounting.
- **Verification:** compare the complete call against the baseline.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D22. Cold-start compilation
- **Symptom:** first invocation includes code generation.
- **Competing cause:** first invocation may also initialize context and memory.
- **Discriminator:** separate compilation, initialization and steady-state samples.
- **Bounded intervention:** report cold and warmed paths separately.
- **Verification:** repeat on a clean process and a warmed process.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D23. Clock variation
- **Symptom:** effective compute and bandwidth change over time.
- **Competing cause:** a competing process can produce similar variance.
- **Discriminator:** record clocks, power and other GPU activity.
- **Bounded intervention:** control or stratify environmental state.
- **Verification:** compare distributions under representative conditions.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D24. Concurrent workloads
- **Symptom:** another process consumes memory or compute resources.
- **Competing cause:** thermal changes can produce a similar slowdown.
- **Discriminator:** compare isolated and shared-device traces.
- **Bounded intervention:** schedule or budget contention explicitly.
- **Verification:** measure service latency with realistic concurrency.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D25. Profiler replay changes cache
- **Symptom:** multiple passes observe a different cache state.
- **Competing cause:** nondeterministic input can also change counters.
- **Discriminator:** compare replay settings with unprofiled behavior.
- **Bounded intervention:** use a replay mode appropriate to the workload.
- **Verification:** validate timing outside the profiler and compare metrics.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D26. Profiling short kernel
- **Symptom:** instrumentation dominates the operation being studied.
- **Competing cause:** launch overhead may also dominate the real workload.
- **Discriminator:** compare unprofiled batched timing with profile time.
- **Bounded intervention:** collect a minimal relevant metric set.
- **Verification:** repeat unprofiled timings and report dispersion.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D27. Tensor padding
- **Symptom:** tile-aligned execution computes extra elements.
- **Competing cause:** masked edge traffic may be the main overhead.
- **Discriminator:** compare useful dimensions with executed tile shape.
- **Bounded intervention:** select a compatible tile or fallback path.
- **Verification:** check odd shapes and useful throughput.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D28. Precision conversion
- **Symptom:** format conversion consumes instructions or bandwidth.
- **Competing cause:** compute may still be limited by the main operation.
- **Discriminator:** inspect conversion instruction mix and associated bytes.
- **Bounded intervention:** retain a compatible representation across stages.
- **Verification:** validate error budget and whole-pipeline timing.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D29. Sparse claim mismatch
- **Symptom:** the count assumes hardware sparsity acceleration.
- **Competing cause:** irregularity or unsupported encoding prevents that path.
- **Discriminator:** inspect generated instruction family and actual format.
- **Bounded intervention:** use the applicable dense roof when sparse support is absent.
- **Verification:** check values, executed code and matched latency.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D30. Memory latency not bandwidth
- **Symptom:** low bandwidth accompanies a long kernel.
- **Competing cause:** low work volume can make utilization misleading.
- **Discriminator:** inspect eligible warps and outstanding requests.
- **Bounded intervention:** increase independent requests or improve locality.
- **Verification:** ensure speedup at the same output and workload.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D31. Instruction issue limit
- **Symptom:** a pipeline is busy while arithmetic rate is low.
- **Competing cause:** a memory bottleneck can also reduce issue.
- **Discriminator:** inspect instruction mix and scheduler issue activity.
- **Bounded intervention:** reduce unnecessary address or control instructions.
- **Verification:** compare generated code and useful throughput.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D32. Branch divergence
- **Symptom:** threads in a warp take different paths.
- **Competing cause:** data-dependent memory accesses can dominate instead.
- **Discriminator:** compare branch distribution and memory activity.
- **Bounded intervention:** reorganize data or work only if ordering permits.
- **Verification:** test all branches and measure distribution-sensitive time.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D33. Atomic contention
- **Symptom:** many threads update the same location.
- **Competing cause:** insufficient bandwidth can slow noncontended atomics too.
- **Discriminator:** compare contended and distributed-key workloads.
- **Bounded intervention:** aggregate locally before the global update.
- **Verification:** validate ordering requirements and numerical tolerance.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D34. Reduction order
- **Symptom:** parallel reassociation changes floating-point results.
- **Competing cause:** an indexing bug can cause much larger errors.
- **Discriminator:** compare against a higher-precision reference by shape.
- **Bounded intervention:** set an explicit tolerance or preserve required order.
- **Verification:** test cancellation and extreme values.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D35. NaN propagation
- **Symptom:** a fast path changes exceptional-value behavior.
- **Competing cause:** input preprocessing may have removed NaNs.
- **Discriminator:** run explicit NaN infinity and signed-zero cases.
- **Bounded intervention:** choose operations meeting the stated numerical contract.
- **Verification:** compare bitwise or tolerance-based reference as required.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D36. Occupancy overfocus
- **Symptom:** more active warps do not improve time.
- **Competing cause:** register reduction may introduce extra instructions.
- **Discriminator:** compare eligible issue and runtime before changing resources.
- **Bounded intervention:** change occupancy only when latency hiding is limiting.
- **Verification:** confirm no spills and compare performance distributions.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D37. Resource-limited tiling
- **Symptom:** larger tiles reduce nominal bytes but lower residency.
- **Competing cause:** edge inefficiency may explain the slowdown.
- **Discriminator:** inspect registers shared memory and active blocks.
- **Bounded intervention:** try a smaller tile while retaining enough reuse.
- **Verification:** measure traffic and time across representative shapes.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D38. Misaligned access
- **Symptom:** unaligned base pointers require extra transactions.
- **Competing cause:** stride can independently waste transactions.
- **Discriminator:** compare aligned and intentionally shifted allocations.
- **Bounded intervention:** align allocations or adapt the mapping safely.
- **Verification:** test real offsets and memory bounds.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D39. Irregular gather
- **Symptom:** useful bytes are much smaller than fetched sectors.
- **Competing cause:** cache reuse can reduce the physical transfer count.
- **Discriminator:** compare sector requests and cache hit behavior.
- **Bounded intervention:** reorder indices if application semantics allow.
- **Verification:** include reorder cost and validate permutation inversion.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D40. Repeated output writes
- **Symptom:** an intermediate is written and read between stages.
- **Competing cause:** the intermediate might remain cache resident.
- **Discriminator:** compare device-memory and cache traffic for the full range.
- **Bounded intervention:** fuse only if resource usage and semantics allow.
- **Verification:** measure complete range and intermediate elimination.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D41. Pipeline imbalance
- **Symptom:** a producer is slower than its consumer.
- **Competing cause:** synchronization protocol can also stall both.
- **Discriminator:** inspect stage timing and buffer occupancy if available.
- **Bounded intervention:** rebalance work or buffer depth under resource limits.
- **Verification:** test ordering and end-to-end throughput.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D42. Wrong time window
- **Symptom:** traffic comes from one launch but time from a range.
- **Competing cause:** asynchronous timing can omit device execution.
- **Discriminator:** align profiler launch identity and timed interval.
- **Bounded intervention:** use a correctly synchronized same-scope timer.
- **Verification:** cross-check range and per-kernel durations.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D43. Concurrent kernel timing
- **Symptom:** isolated kernel time does not reproduce overlap.
- **Competing cause:** contention can offset overlap gains.
- **Discriminator:** inspect the application timeline with concurrent kernels.
- **Bounded intervention:** evaluate the full range rather than summing isolated times.
- **Verification:** measure wall time under representative concurrency.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D44. Division by zero
- **Symptom:** no bytes cross a chosen boundary.
- **Competing cause:** a missing or unsupported counter can report zero too.
- **Discriminator:** inspect counter validity and cache state.
- **Bounded intervention:** report the undefined intensity instead of dividing.
- **Verification:** recollect with a supported metric and known input.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D45. Sampling noise
- **Symptom:** small observed differences vary between runs.
- **Competing cause:** a deterministic performance regression may coexist.
- **Discriminator:** compare repeated paired timings and distributions.
- **Bounded intervention:** retain baseline until effect exceeds measurement noise.
- **Verification:** repeat under controlled representative conditions.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D46. Shape distribution shift
- **Symptom:** an optimization favors a rare large shape.
- **Competing cause:** the dominant shapes may have changed since baseline.
- **Discriminator:** weight measurements by current workload distribution.
- **Bounded intervention:** dispatch by shape only if complexity is justified.
- **Verification:** measure aggregate and tail latency under a holdout mix.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D47. Tightened error budget
- **Symptom:** lower precision no longer meets downstream tolerance.
- **Competing cause:** a separate logic bug may mimic numerical drift.
- **Discriminator:** compare against a high-precision reference on stress inputs.
- **Bounded intervention:** restore precision or modify the accumulation strategy.
- **Verification:** validate downstream metrics and edge cases.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D48. Wrong memory boundary
- **Symptom:** L2 traffic is compared to device-memory bandwidth.
- **Competing cause:** bytes from a different launch can look similar.
- **Discriminator:** inspect counter names and hierarchy definitions.
- **Bounded intervention:** pair each byte counter with its own roof.
- **Verification:** document the level and replot the workload.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D49. Racing shared memory
- **Symptom:** a faster variant produces intermittent errors.
- **Competing cause:** uninitialized memory can create similar symptoms.
- **Discriminator:** use targeted race and initialization diagnostics.
- **Bounded intervention:** add the synchronization required by the data dependency.
- **Verification:** stress repeated varied schedules and compare outputs.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D50. Excessive recomputation
- **Symptom:** reported intensity rises while latency worsens.
- **Competing cause:** a simultaneous clock change can also slow execution.
- **Discriminator:** compare useful FLOPs, executed instructions and time.
- **Bounded intervention:** retain recomputation only if it saves a limiting transfer.
- **Verification:** validate useful throughput and resource counters.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D51. Insufficient benchmark duration
- **Symptom:** timer resolution obscures small kernel differences.
- **Competing cause:** actual performance may be noisy at larger durations.
- **Discriminator:** time batches and divide by the exact number of launches.
- **Bounded intervention:** amortize timing overhead without changing data reuse.
- **Verification:** compare single-launch and batched conditions.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D52. Wrong bandwidth reference
- **Symptom:** a read-only microbenchmark is used for mixed reads and writes.
- **Competing cause:** cache-resident traffic can also inflate apparent bandwidth.
- **Discriminator:** measure a reference with comparable direction and footprint.
- **Bounded intervention:** label the reference empirical rather than an absolute law.
- **Verification:** repeat for the real access mix and device configuration.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D53. Dynamic allocation in hot path
- **Symptom:** allocation or initialization dominates a request.
- **Competing cause:** a synchronization operation can account for the delay.
- **Discriminator:** trace host and device activity around allocation.
- **Bounded intervention:** reuse buffers if ownership and lifetime permit.
- **Verification:** test concurrency and peak memory as well as time.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D54. Missing tail work
- **Symptom:** a vectorized kernel skips final elements.
- **Competing cause:** the reference output may itself be incorrectly shaped.
- **Discriminator:** test lengths around each vector and tile boundary.
- **Bounded intervention:** add a bounds-safe tail path.
- **Verification:** compare every output element for small adversarial lengths.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D55. Incorrect overlap assumption
- **Symptom:** adding compute and transfer ceilings exaggerates lower bound.
- **Competing cause:** hidden dependencies may actually serialize phases.
- **Discriminator:** inspect the timeline and dependencies.
- **Bounded intervention:** model maximum for overlap and sum only for proven serialization.
- **Verification:** compare predictions with observed range duration.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D56. Compiler version change
- **Symptom:** a rebuild changes register use or instructions.
- **Competing cause:** a driver or power-state change can also alter time.
- **Discriminator:** record binary hash compiler flags and generated code.
- **Bounded intervention:** pin the toolchain during a controlled comparison.
- **Verification:** rerun correctness and representative timings on both builds.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D57. Missing error check
- **Symptom:** an asynchronous failure surfaces after the measured kernel.
- **Competing cause:** the numerical mismatch may arise in a later consumer.
- **Discriminator:** check launch status and synchronize at the diagnosis boundary.
- **Bounded intervention:** propagate errors without changing production synchronization unnecessarily.
- **Verification:** exercise an invalid-input path and the normal workload.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D58. Output not consumed
- **Symptom:** a benchmark omits necessary downstream work.
- **Competing cause:** compiler elimination or lazy evaluation may also alter work.
- **Discriminator:** verify output use and the actual launched kernels.
- **Bounded intervention:** benchmark the same observable output contract.
- **Verification:** compare complete application results and end-to-end latency.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D59. Underestimated output size
- **Symptom:** temporary outputs exceed the assumed traffic model.
- **Competing cause:** compression or caching can move fewer physical bytes.
- **Discriminator:** derive sizes from actual shapes dtypes and layout.
- **Bounded intervention:** include intermediate and metadata transfers where relevant.
- **Verification:** compare byte estimates against boundary counters.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

### D60. Incorrect comparison baseline
- **Symptom:** variants use different shapes or numeric accuracy.
- **Competing cause:** one variant may include extra initialization.
- **Discriminator:** hold input outputs warmup and scope constant.
- **Bounded intervention:** choose a reference implementing identical semantics.
- **Verification:** report any intentional difference and its effect.
- **Scope:** compare the same shape, precision, boundary and output contract.
- **Evidence:** HYPOTHETICAL; no production failure or optimization result is asserted.
- **Guardrail:** revert if the correctness contract fails or representative latency worsens.
- **Decision:** accept the hypothesis only if the predicted signal changes in the proposed experiment.
- **Escalation:** if neither explanation fits, inspect the full range and revisit the measurement boundary.
- **Record:** retain input, version, timing scope, counter definition and uncertainty with the result.

**Takeaway:** Diagnose with a competing explanation and a falsifiable check before changing code.

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

All numerical inputs below are ILLUSTRATIVE; answers follow the assumptions and counting conventions in Sections 3–5.

### Q01. What is arithmetic intensity?
- **Answer:** Work divided by bytes transferred across a named memory boundary.
- **Reasoning:** The boundary matters because cache traffic and device-memory traffic can differ.
- **Follow-up:** What if the byte count is zero?
- **Follow-up answer:** The quotient is undefined; report the boundary and diagnose counter validity.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q02. What is the roofline bound?
- **Answer:** The minimum of compatible compute throughput and bandwidth times intensity.
- **Reasoning:** Each term is an independent bound on the same workload rate.
- **Follow-up:** Does touching the bound guarantee correctness?
- **Follow-up answer:** No; output correctness is independent of throughput.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q03. Count one fused multiply-add.
- **Answer:** Under the declared convention it is two floating-point operations.
- **Reasoning:** One multiply and one add are counted although the operation is fused.
- **Follow-up:** Does one instruction always equal two FLOP?
- **Follow-up answer:** No; vector and matrix instructions cover multiple arithmetic operations.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q04. For 2 TFLOP and 0.5 TB, find intensity.
- **Answer:** 4 FLOP/byte.
- **Reasoning:** Divide 2 by 0.5; the decimal tera factors cancel.
- **Follow-up:** What if bytes fall to 0.4 TB?
- **Follow-up answer:** Intensity becomes 5 FLOP/byte if work stays fixed.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q05. With 60 TFLOP/s compute and 2 TB/s bandwidth, find the balance point.
- **Answer:** 30 FLOP/byte.
- **Reasoning:** Divide the compatible compute ceiling by bandwidth.
- **Follow-up:** Is this a universal GPU threshold?
- **Follow-up answer:** No; both ceilings vary with precision, boundary and operating conditions.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q06. For intensity 4 and bandwidth 2 TB/s, what is the bandwidth roof?
- **Answer:** 8 TFLOP/s.
- **Reasoning:** Multiplying FLOP/byte by byte/s cancels bytes.
- **Follow-up:** Is 8 TFLOP/s an observed result?
- **Follow-up answer:** No; it is a conditional bound under illustrative assumptions.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q07. A point is far under both roofs. Is it bandwidth-bound?
- **Answer:** Not established.
- **Reasoning:** Insufficient parallelism, dependencies, issue limits or launch overhead may dominate.
- **Follow-up:** What is a useful next step?
- **Follow-up answer:** Inspect scheduler eligibility, timeline and resource utilization before choosing a fix.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q08. Why can fusion fail despite reducing an intermediate?
- **Answer:** It may increase register pressure, spills or reduce occupancy.
- **Reasoning:** The change can add traffic or reduce attainable issue rate.
- **Follow-up:** How do you decide?
- **Follow-up answer:** Measure full-range latency, boundary bytes, resources and correctness.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q09. A roofline point is above its roof. What does that mean?
- **Answer:** The comparison or assumed ceiling needs investigation.
- **Reasoning:** Units, byte boundary, FLOP convention, timing scope or empirical ceiling may be mismatched.
- **Follow-up:** Could a microbenchmark ceiling be exceeded legitimately?
- **Follow-up answer:** Yes; a different access mix can exceed that empirical reference.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q10. How do you compare useful and executed FLOPs?
- **Answer:** Report both explicitly and use consistent throughput denominators.
- **Reasoning:** Padding and recomputation can inflate executed FLOP/s without improving useful output.
- **Follow-up:** What is the decision metric?
- **Follow-up answer:** Output-equivalent latency or useful throughput under the workload objective.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q11. What is the ideal time bound in the worked example?
- **Answer:** 0.25 seconds.
- **Reasoning:** The maximum of 2/60 seconds and 0.5/2 seconds is 0.25 seconds.
- **Follow-up:** After 0.1 TB is eliminated under unchanged assumptions?
- **Follow-up answer:** The bound becomes 0.2 seconds; this is not a measured result.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q12. Does a larger tile always help?
- **Answer:** No.
- **Reasoning:** It can improve reuse while consuming more shared memory and registers.
- **Follow-up:** What changed constraint could reverse the decision?
- **Follow-up answer:** A smaller workload or tighter register budget can make reduced occupancy dominate.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

### Q13. How do you compare profiler and production timing?
- **Answer:** Measure an unprofiled baseline with the same output and workload.
- **Reasoning:** Profiler replay and instrumentation can alter overhead and cache conditions.
- **Follow-up:** Which result is the production latency?
- **Follow-up answer:** The representative unprofiled timing, subject to its sampling and environment limitations.
- **Evidence class:** DERIVED from the stated model; no benchmark is implied.
- **Scoring:** full credit requires the answer and its boundary or justification.
- **Partial credit:** a correct result without its key assumption.
- **Red flag:** treating an illustrative ceiling as measured production performance.
- **Support:** use the definitions, formulas and examples in Sections 3–5.

**Takeaway:** A strong answer explains the boundary, the calculation, and how the conclusion could fail.

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
