# Module 56 Cheat Sheets — End-to-End Capstones and Portfolio

## 1. Overview

This collection teaches how to scope, evidence, evaluate, and present an end-to-end capstone without overstating a prototype. It assumes prior data, ML, systems, and governance skills. Readiness means a learner can trace one input through a bounded architecture, calculate a declared budget, identify an oracle, state a limitation, and defend a release gate. Visible source titles are inventory only; their bodies and hardware evidence are unavailable.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Status |
|---|---|---|---|---|
| Core | [01 Capstone Evidence Rubric](01_Capstone_Evidence_Rubric.md) | Bound claims, calculate frame/queue feasibility, and define evidence gates | module index; measurement boundaries | draft |
| Core | Portfolio checklist | Create artifact/claim/reproduction checklist | 01 | planned |
| Advanced | Architecture review agenda | Run a decision and failure-path review | 01 | planned |
| Final | Common Issues | Diagnose capstone evidence and integration failures | all sheets | planned; no file |

## 3. Quick topic lookup

| Question | File | Section | Use condition | Failure warning |
|---|---|---|---|---|
| Can I call this end-to-end? | [01](01_Capstone_Evidence_Rubric.md) | 1–2 | a demo claim needs scope | a diagram is not an execution trace |
| Does the timing budget close? | [01](01_Capstone_Evidence_Rubric.md) | 4–5 | batch/rate inputs known | component means are not p99 |
| How do I diagnose stale display? | [01](01_Capstone_Evidence_Rubric.md) | 8 | timestamps exist | UI age alone does not identify cause |

## 4. Reading paths — three options

**Beginner:** module index → 01 Sections 1–5 → Section 12; learn claim tiers and one checked calculation.

**Practitioner:** 01 in order → record an input/output trace → prepare the planned portfolio checklist; do not call an unexecuted plan a result.

**Advanced:** 01 → planned architecture agenda → HIL/trace validation → planned Common Issues; add only version-pinned, primary evidence for device/framework claims.

## 5. Prerequisite skills

Readers need dimensional arithmetic, timestamp semantics, basic queues, schema validation, version control, and the ability to define an evaluation denominator. Relevant methods are in the course authoring standard and Module 56 index. Hardware-specific wiring, framework APIs, clinical rules, and model benchmarks require their own primary references.

## 6. What this collection does NOT cover

It does not certify medical devices, diagnose patients, provide a complete STM32/Pi5 build, or establish performance of a specific framework/model. Separate work must cover clinical validation, threat modeling, device electrical safety, licensing, and any deployment approval.

## 7. Master symbol glossary

| Symbol/term | Meaning | Units/type | Valid range | First use |
|---|---|---|---|---|
| `n` | samples per frame | samples/frame | positive integer | [01 §4](01_Capstone_Evidence_Rubric.md#4-core-formulas-with-full-derivation) |
| `f_s` | sampling rate | samples/s | positive | [01 §4](01_Capstone_Evidence_Rubric.md#4-core-formulas-with-full-derivation) |
| `T_frame` | source interval per frame | s or ms | nonnegative | [01 §4](01_Capstone_Evidence_Rubric.md#4-core-formulas-with-full-derivation) |
| `lambda`, `mu` | arrival/service rate | frames/s | nonnegative | [01 §4](01_Capstone_Evidence_Rubric.md#4-core-formulas-with-full-derivation) |
| `B` | queued frames | frames | nonnegative | [01 §4](01_Capstone_Evidence_Rubric.md#4-core-formulas-with-full-derivation) |

## 8. Quick Reference Card directory

| File/card | Core rule | Use condition | Critical assumption | Avoid condition |
|---|---|---|---|---|
| [01 §12](01_Capstone_Evidence_Rubric.md#12-quick-reference-card) | claim tier must match linked evidence | any portfolio assertion | scope is explicit | borrowed or absent evidence |

## 9. Common Issues locator

No Common Issues file exists yet. The final artifact must retain all 12 cheat-sheet headings and contain at least 25 distinct complete issues with ID, category, provenance, competing causes, diagnostic, mitigation risk, correction, prevention, and verification. The six diagnostics in 01 are not a substitute.

## 10. Sources, versions and maintenance

| Source ID | Source | Supported scope | Status / limitation | Recheck trigger |
|---|---|---|---|---|
| SRC-56-01 | [`directory.txt` lines 1147–1164](../../../../Templates/directory.txt) | visible Module08 names | inventory only; no bodies | source export changes |
| SRC-56-02 | [edge-capstone brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-capstone) | syllabus requirements | planned, not empirical evidence | source/body becomes available |
| SRC-56-03 | [edge-source brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-source) | clipped-source boundary | unresolved-source remains unresolved | exact paths are recovered |

## 11. Collection acceptance checklist

- [x] One real sheet has a 12-section sequence, source boundary, formula/invariant, calculation, diagnostics, and answered interview questions.
- [ ] Every indexed topic has its own substantive artifact.
- [ ] Each individual artifact meets the course 1,000-substantive-line gate.
- [ ] Portfolio checklist, architecture agenda, and final Common Issues artifact exist.
- [ ] Primary sources, versions, runnable traces, HIL evidence, and links have been independently reviewed.
