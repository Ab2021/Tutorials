# Module 35 Cheat Sheets — Edge AI, Embedded Systems and TinyML

## 1. Overview

This collection teaches resource-bounded acquisition, inference, scheduling, power and memory decisions for microcontrollers and small edge computers. Readiness means a learner can state buffer ownership, derive an explicit capacity or timing calculation, select a bounded execution path, diagnose a competing failure cause, and name evidence still needed on the target. It does not establish medical, safety or device certification.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Key skill | Status |
|---|---|---|---|---|---|
| 1 | [01 Embedded Acquisition, Firmware and TinyML](01_Embedded_Acquisition_Firmware_and_TinyML.md) | Budget a timestamped DMA-to-inference-to-UART path and diagnose its contracts. | rates, C ownership | `R=f_s*b_s+(f_s/N)h`; buffer invariant | draft |
| 1 | MCU model budget | Select model/arena/stack allocation under target constraints. | File 01 | peak liveness and flash/RAM accounting | planned |
| 1 | Sensor sampling guide | Choose acquisition/filter/decimation boundaries. | rates, signal basics | sampling, window and timestamp reasoning | planned |
| 2 | TinyML deployment checklist | Defend conversion, test, update and rollback evidence. | Files 01–03 | release manifest and target validation | planned |
| Final | Common Issues | Diagnose at least 25 distinct local failure conditions. | all completed sheets | symptom-to-contract workflow | not authored |

## 3. Quick topic lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| Can UART carry this sensor stream? | 01 §4–5 | fixed framing stated | baud is not bytes/s |
| Is a DMA block safe to read? | 01 §4.4 | completion semantics known | `volatile` alone may not order memory |
| Why does inference miss a deadline? | 01 §4.2, §8 | execution/interference bounded | average latency is not a bound |
| Does the model fit RAM? | planned MCU model budget | linker and liveness evidence available | parameter bytes are not peak arena |
| Is an update safe to deploy? | planned deployment checklist | rollback/authenticity requirements known | signing does not prove compatibility |

## 4. Reading paths — three options

**Beginner:** 01, then planned sensor sampling guide, then planned MCU model budget. Objective: build a bounded sensor-to-output mental model.

**Practitioner:** 01, sensor sampling guide, MCU model budget, TinyML deployment checklist, then the final Common Issues file when authored. Objective: review a target integration without treating illustrative estimates as measurements.

**Advanced:** follow the practitioner route, read the [module cases](../Case_Studies/INDEX.md) when authored, then the Part 07 [cross-boundary issues](../../../Cheat_Sheets/99_35_Common_Issues.md). Optional forward path: Module 36 runtimes/quantisation.

## 5. Prerequisite skills

- Convert Hz, seconds, bytes, baud, amps and joules; distinguish a count from a rate.
- Read C ownership/lifetime and basic interrupt/task scheduling.
- Interpret a linker map, timestamped counter trace and protocol frame.
- Use exact device documentation rather than extrapolating across MCU variants.

## 6. What this collection does NOT cover

It hands off clinical signal interpretation to module/domain-specific material; complete model compression and runtime selection to Module 36; sensor-fusion estimation to Module 37; and functional safety lifecycle evidence to Module 41. It does not claim a real target benchmark, certification or proprietary implementation.

## 7. Master symbol glossary

| Symbol/term | Meaning | Units/type | Valid range | First use |
|---|---|---|---|---|
| `f_s` | sample rate per channel | samples/s | positive | [01 §3](01_Embedded_Acquisition_Firmware_and_TinyML.md#3-foundational-definitions) |
| `N` | samples/block | integer samples | positive | 01 §3 |
| `T_w` | block duration | seconds | uniform sampling assumption | 01 §4 |
| `C`, `I`, `D` | execution, interference, deadline | seconds | stated workload only | 01 §4 |
| `K` | reusable buffers | integer | implementation-defined | 01 §4 |
| `B` | baud | symbols/s | framing-dependent | 01 §4 |
| `E`, `P_avg` | energy, average power | J, W | state model stated | 01 §4 |

## 8. Quick Reference Card directory

| File/card | Rule | Use condition | Critical assumption | Avoid when |
|---|---|---|---|---|
| [01 §12](01_Embedded_Acquisition_Firmware_and_TinyML.md#12-quick-reference-card) | publish complete blocks; validate freshness | bounded local path | sequence/timestamp are trustworthy | target semantics unknown |

## 9. Common Issues locator

A distinct Module 35 Common Issues artifact is **not authored**. The part-level [35 cross-boundary issues](../../../Cheat_Sheets/99_35_Common_Issues.md) is supplemental and does not satisfy this local collection requirement. Required local issue areas: DMA ownership, priority inversion, UART framing, stack/arena collision, stale decision, tensor metadata, DSP state, and wake transients.

## 10. Empirical sources, versions and collection maintenance

| File | Evidence/status | Sources and review trigger | Open limitation |
|---|---|---|---|
| 01 | draft; DERIVED/ILLUSTRATIVE/OPEN | ST, CMSIS-NN and FreeRTOS links in §10; review on target/config change | no target execution or original source body |
| planned files | planned | obtain primary documentation before authoring | not teaching artifacts |

The visible source inventory is [directory.txt lines 1094–1110](../../../../Templates/directory.txt); its mapping is the [edge-arm-firmware syllabus](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-arm-firmware). Source titles establish scope, not source-body access.

## 11. Collection acceptance checklist

- [x] Original three reference-family rows retained in the ladder.
- [x] One draft has all 12 sections, hand-checked arithmetic, failures and answered questions.
- [ ] Each remaining planned sheet is authored to the individual-file contract.
- [ ] Distinct local Common Issues has at least 25 complete issues.
- [ ] All target measurements, versions and primary claims are reviewed.
- [ ] Local cases are authored and cross-linked.
