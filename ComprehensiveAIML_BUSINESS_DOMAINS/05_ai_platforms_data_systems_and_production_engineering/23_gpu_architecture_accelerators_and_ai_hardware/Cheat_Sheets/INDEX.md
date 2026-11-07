# GPU Architecture, Accelerators and AI Hardware — Cheat Sheets

**Status:** The roofline and H100 sheets are unreviewed drafts; remaining entries are planned. The H100 sheet exceeds 1,000 lines but has not been technically reviewed or validated on a device. Follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master. Hypothetical values are not hardware benchmarks.

## 1. Overview

This collection supports engineers reasoning about GPU performance and resource constraints. The current draft connects arithmetic intensity, memory traffic, compatible ceilings and kernel diagnosis. Readiness means choosing the measured boundary and workload, computing a bound with consistent units, and checking whether an optimization actually changes the bottleneck.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Practitioner | [Roofline arithmetic intensity and kernel diagnosis](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md) | Compute a boundary-specific bound and diagnose limits; requires arithmetic, data movement and GPU fundamentals | Draft; unreviewed |
| Foundations | [H100 architecture and resource limits](01_H100_Architecture_and_Resource_Limits.md) | Explain resource-limited residency and architecture-specific profiling | Draft; unreviewed |
| Planned | Precision guide | Compare numerical and hardware precision tradeoffs | Not written |
| Planned | Hardware sizing worksheet | Size compute, memory, networking and workload headroom | Not written |
| Planned | Common Issues | Distinct collection-wide diagnostic cases | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Is a kernel likely compute- or bandwidth-limited? | [Formula section](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#4-core-formulas-with-full-derivation) | FLOP count, traffic boundary and ceilings must describe the same operation and precision. |
| How do I calculate a simple roofline? | [Worked example](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#5-worked-numerical-example) | Results use illustrative ceilings and are not measurements. |
| Why is measured throughput below the roof? | [Failure modes](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#8-failure-modes--edge-cases) | A ceiling is an upper-bound model, not a guarantee of achieved performance. |
| How do launch resources constrain H100 residency? | [Worked example](01_H100_Architecture_and_Resource_Limits.md#5-worked-numerical-example) | Example capacities are illustrative; query the actual device and launch. |

## 4. Reading paths

- **Beginner:** Review FLOPs, bytes, elapsed time, GPU memory hierarchy and dimensional analysis before reading the draft's definitions.
- **Practitioner:** Read the complete roofline draft, reproduce its calculation and compare a measured kernel with compatible compute and bandwidth ceilings.
- **Advanced:** Examine multiple memory boundaries, cache effects, precision, occupancy and concurrency; design controlled profiling and a correctness-preserving optimization. The precision and sizing references remain planned.

## 5. Prerequisite skills

Calculate rates and ratios with consistent decimal or binary units, describe GPU compute and memory hierarchies, identify the target kernel and precision, and interpret profiling counters for the installed tools and hardware.

## 6. What this collection does not cover

A roofline model alone does not determine numerical accuracy, end-to-end latency, utilization of every execution unit, or a real system's price-performance. Precision-specific guidance, full hardware sizing and the Common Issues artifact remain planned here.

## 7. Master symbol glossary

| Symbol | Meaning and units | Scope / warning | Reference |
|---|---|---|---|
| FLOP | Floating-point operation | State counting convention and precision | [Definitions](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#3-foundational-definitions) |
| Arithmetic intensity | FLOPs per byte crossing a specified boundary | Different cache or memory boundaries yield different values | [Formula section](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#4-core-formulas-with-full-derivation) |
| Compute ceiling | Compatible peak or measured attainable FLOP/s | Must match operation, precision and assumed hardware | [Formula section](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#4-core-formulas-with-full-derivation) |
| Bandwidth | Bytes per second at the chosen boundary | Nominal link speed is not an observed application rate | [Worked example](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#5-worked-numerical-example) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [Roofline](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#12-quick-reference-card) | Performance is bounded by the lesser of compatible compute and intensity-times-bandwidth ceilings. | All terms must refer to the same workload, precision and traffic boundary. |
| [H100 resources](01_H100_Architecture_and_Resource_Limits.md#12-quick-reference-card) | Residency is bounded by the minimum of applicable resource limits. | Use actual rounded allocations, device constraints and cluster placement; do not infer performance from occupancy alone. |

## 9. Common Issues locator

A standalone Common Issues artifact is **planned, not written**; no collection-wide issue total is claimed. Use the draft's [failure modes](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md#8-failure-modes--edge-cases) for competing causes and proposed discriminators.

## 10. Sources, versions and maintenance

The roofline and H100 examples are illustrative; they do not establish measured performance. The H100 draft cites the archived CUDA 12.4.1 Hopper guide for versioned vendor claims; its review and device validation remain open. Review assumptions, counter definitions, hardware capability and precision when device, driver, compiler, workload or profiling-tool version changes. The original visible course tree is a topic inventory, not proof of measured results.

**Planned reference-informed brief:** [PLAN-2026-IR007](../../../00_REFERENCE_INFORMED_CASE_PLAN.md) covers continuous batching, chunked prefill, paged KV storage, prefix authorization and speculative-decoding correctness. Measure time to first token, inter-token latency, offered versus accepted load and cache headroom; speculative decoding can be slower. Its [interview example](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md) is reference material, not verified workload evidence.

## 11. Collection acceptance checklist

- [ ] Planned GPU topics and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each sheet retains all 12 sections, checked calculations or traces, diagnostic failures, answered interviews and its final card.
- [ ] Workload, precision, measurement boundary, versions and source limitations are explicit.
- [ ] Links, glossary, paths, issue locations and editorial statuses match the actual files.
