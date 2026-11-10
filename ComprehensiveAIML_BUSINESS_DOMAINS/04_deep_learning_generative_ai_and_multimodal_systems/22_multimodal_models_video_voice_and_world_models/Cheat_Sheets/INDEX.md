# Module 22 Cheat Sheets — Multimodal Models, Video, Voice and World Models

## 1. Overview

This draft collection will teach selection and diagnosis for visual-text, video, voice, and
vision-language-action systems. It assumes neural-network and evaluation basics. No individual
sheet or final Common Issues artifact exists yet; all topics below are planned, not completed.
Visible source titles are inventory commitments, not inspected source bodies or performance evidence.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Key skill | State |
|---|---|---|---|---|---|
| Foundation | `01_Multimodal_Evaluation.md` (planned) | Define modality-specific and joint task contracts and slice metrics. | Classification metrics, train/test separation | Error decomposition | planned |
| Core | `02_Voice_Agent_Latency_Budget.md` (planned) | Allocate a latency budget across capture, ASR, reasoning, TTS, and network steps. | Rates, percentiles, queue basics | Budget and boundary arithmetic | planned |
| Advanced | `03_VLA_Risk_Checklist.md` (planned) | Decide when perception/action evidence requires abstention or human control. | Coordinate/time contracts, safety workflow | State/invariant diagnosis | planned |
| Final | `04_25_Common_Issues.md` (planned) | Diagnose at least 25 distinct multimodal failures. | Files 01–03 | Symptom-to-recovery trace | planned |

## 3. Quick topic lookup

| Question | Planned file / section | Use condition | Failure warning |
|---|---|---|---|
| Which metric exposes a cross-modal failure? | 01 / evaluation slices | Inputs and ground truth are time-aligned | Aggregate accuracy can hide modality slices. |
| Where did a voice interaction exceed its budget? | 02 / critical path | Timestamps use one clock contract | Mean latency is not a tail guarantee. |
| May an action suggestion be automated? | 03 / authority and safe state | Actuator, frame, and approval scope are known | Model confidence is not authority. |
| How do I triage a recurring symptom? | 04 / issue index | A concrete symptom is observed | Similar symptoms can have competing causes. |

## 4. Reading paths — three options

| Path | Sequence | Objective | Assumptions / optional extension |
|---|---|---|---|
| Beginner | 01 → 02 → 04 | Measure a bounded multimodal workflow. | Knows basic classification metrics; 03 is optional. |
| Practitioner | 01 → 02 → 03 → 04 | Select an architecture boundary and a recovery plan. | Can define time, data, and authority contracts. |
| Advanced | 01 → 02 → 03 → 04 plus Module 18/20 sheets | Defend cross-modal evaluation and operational limits. | External modules are optional handoffs, not substitutes. |

## 5. Prerequisite skills

| Area | Required skill | Boundary / reference |
|---|---|---|
| ML evaluation | Define population, label, split, metric numerator, denominator, and slice. | Module 17 evaluation material is a prerequisite, not proof of multimodal validity. |
| Systems | Distinguish event time, processing time, rate, queue, and percentile. | A latency budget is conditional on its clock and workload model. |
| Modality handling | Describe image coordinates, audio sample rate, video frame rate, and text tokenisation. | Conversions do not recreate information missing at capture. |
| Operations | Identify a human owner, escalation path, and safe state. | No model score grants operational authority. |

## 6. What this collection does NOT cover

It does not certify medical, safety, legal, employment, or insurance decisions; those require
applicable domain owners and evidence. It does not replace sensor calibration, control theory,
privacy/access governance, ASR training, or video-annotation practice. It also does not treat a
document VLM extraction case as a substitute for video or action-system validation.

## 7. Master symbol glossary

| Symbol / term | Meaning | Units / type | Valid range | First planned use |
|---|---|---|---|---|
| `t_e` | Event timestamp | seconds, UTC contract | Monotone only within stated source | 01–03 |
| `L` | End-to-end latency | milliseconds | Defined start/end clocks | 02 |
| `L_i` | Component latency on critical path | milliseconds | Sum only for serial components | 02 |
| `r` | Input rate | items/s | Declared modality and population | 02 |
| `p` | Calibrated probability or confidence score | dimensionless | `[0,1]`; not authority | 01, 03 |
| `a` | Proposed action | typed command | Only within approved action set | 03 |

## 8. Quick Reference Card directory

| Planned file / card | Core rule | Use condition | Critical assumption | Avoid when |
|---|---|---|---|---|
| 01 / Section 12 | Measure joint outcome and modality slices. | Labels and alignment are defined. | Evaluation split is independent. | Labels or timestamps are ambiguous. |
| 02 / Section 12 | `L = sum(L_i)` only on a serial critical path. | Same-clock component boundaries exist. | No unmodelled queue/retry path. | Components overlap or retry. |
| 03 / Section 12 | Escalate when evidence or authority is absent. | A safe fallback exists. | Human owner can act. | Routing has no response contract. |
| 04 / Section 12 | Diagnose competing causes before correction. | Symptom is observable. | Diagnostic signals are retained. | No provenance or version record exists. |

## 9. Common Issues locator

`04_25_Common_Issues.md` is planned and therefore has no working link or issue anchors.
It must contain 25 distinct, complete issues with stable IDs, category, provenance, competing
causes, expected diagnostic signal, mitigation risk, correction, prevention, verification, and
related sheet. The count is currently **0 authored issues**, not 25.

## 10. Empirical sources, versions and collection maintenance

| Item | Evidence state | Limitation | Review trigger |
|---|---|---|---|
| Local catalog topics | Visible-name inventory | Original lesson bodies unavailable | Source body or corrected crosswalk becomes available |
| PLAN-2026-IR011 | Planning reference only | Its workload and results are not transferable | A bounded independent manuscript is authored |
| Any future numerical example | ILLUSTRATIVE unless source/run is recorded | Arithmetic alone is not performance evidence | Recompute and attach source/run scope |

## 11. Collection acceptance checklist

- [ ] Four real canonical files exist in the stated order, including a final 25-issue artifact.
- [ ] Each individual sheet has metadata, 12 ordered sections, a checked example and boundary, four diagnostics, 10–15 answered questions, and a final card.
- [ ] The Common Issues file has all 12 headings and at least 25 complete distinct entries.
- [ ] Links and anchors target real files; source, evidence, implementation stage, and editorial status remain separate.
- [ ] Collection glossary, paths, issue locator, source limits, and acceptance state are reconciled after each manuscript.
