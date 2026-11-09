# Module 42 Cheat Sheets — Healthcare, Life Sciences and Biomedical AI

## 1. Overview

This collection teaches bounded, evidence-aware healthcare AI decisions: clinical assistance, biomedical data pipelines, human factors, imaging/triage, discovery, and validation. A reader is ready only when they can state the decision-time data boundary, calculate a metric with its denominator, diagnose a failure, and name the accountable human/workflow. Named organizations in source inventory are research contexts, not deployment claims.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Status |
|---|---|---|---|---|
| 1 | [01 Clinical Decision Support: Evidence and Handoff](01_Clinical_Decision_Support_Evidence_and_Handoff.md) | define time-safe, non-binding decision support and calculate sensitivity | confusion-matrix arithmetic | draft; below course 1,000-line publication requirement |
| 1 | [02 Biomedical Signal Pipeline and Validation](02_Biomedical_Signal_Pipeline_and_Validation.md) | specify acquisition, preprocessing, segmentation and validity | 01 and sampling basics | draft; below course 1,000-line publication requirement |
| 2 | [03 Human Factors: Alert Design and Recovery](03_Human_Factors_Alert_Design_and_Recovery.md) | select review, alert and recovery controls | 01 workflow contract | draft; below course 1,000-line publication requirement |
| 2 | imaging and triage | define a triage evaluation and handoff | diagnostic-study design | planned |
| 3 | drug discovery and validation | separate scientific prediction from experimental validation | statistics and domain review | planned |
| 3 | regulation and post-market monitoring | construct evidence and change-control ledger | prior files | planned |
| final | Common Issues | diagnose 25+ complete healthcare-AI issues | all applicable sheets | planned |

## 3. Quick topic lookup

| Question | File/section | Use condition | Failure warning |
|---|---|---|---|
| What data may a support system use? | 01 §3–4 | index-time review | event time is not availability time |
| How do I calculate sensitivity? | 01 §4–5 | adjudicated positive target | sensitivity is not utility |
| What should abstention do? | 01 §2, §7–8 | missing/stale/out-of-scope input | silent negative output |
| How do I size review work? | 01 §4.2, §9 | explicit alert capacity | omit false-positive workload |

## 4. Reading paths — three options

- **Beginner:** 01, then planned biomedical signal pipeline and human-factors checklist; objective: define a bounded assistive workflow.
- **Practitioner:** beginner path, then planned imaging/triage, drug discovery/validation, and regulation/monitoring; objective: defend an evaluation and operating boundary.
- **Advanced:** all files plus the final Common Issues artifact; objective: diagnose time, label, handoff, and monitoring failure under changed constraints.

## 5. Prerequisite skills

Use a confusion matrix; distinguish counts, rates, and conditional probabilities; record event/availability/decision timestamps; map a clinical workflow without assigning clinical authority; and read an approved local policy. This collection does not provide medical training or policy determinations.

## 6. What this collection does NOT cover

It does not supply diagnosis or treatment advice, claim device clearance, establish legal/regulatory applicability, or reproduce unavailable source bodies. Robotics, generic ML methods, and signal-processing foundations hand off to their respective course modules. Primary studies and local governance artifacts remain required for real implementation claims.

## 7. Master symbol glossary

| Symbol/term | Meaning | Units/type | Valid range | First use |
|---|---|---|---|---|
| `TP` | positive target units flagged | count | non-negative integer | [01 §4](01_Clinical_Decision_Support_Evidence_and_Handoff.md#4-core-formulas-with-full-derivation) |
| `FN` | positive target units not flagged | count | non-negative integer | 01 §4 |
| `FP` | negative target units flagged | count | non-negative integer | 01 §4 |
| `t_d` | decision time | timestamp | stated zone | 01 §2 |
| `t_a` | feature availability time | timestamp | stated zone | 01 §3 |
| `N`, `A` | eligible units, abstentions | count | `0<=A<=N` | 01 §4.3 |

## 8. Quick Reference Card directory

| File | Card | Core rule | Critical assumption | Avoid when |
|---|---|---|---|---|
| [01](01_Clinical_Decision_Support_Evidence_and_Handoff.md#12-quick-reference-card) | clinical support | `t_a <= t_d`; clinician owns action | authorized data and handoff exist | outcome/provenance unavailable |

## 9. Common Issues locator

`07_25_Common_Issues.md` is planned, not created. It must contain at least 25 distinct complete issues with category, provenance, competing causes, diagnostic, correction, prevention, and verification. Current authored issue examples are in [01 §8](01_Clinical_Decision_Support_Evidence_and_Handoff.md#8-failure-modes--edge-cases) and are not a completed collection artifact.

## 10. Sources, versions and maintenance

| Source ID | Source/version | Supported scope | Reviewed | Limitation / trigger |
|---|---|---|---|---|
| SRC-M42-01 | [directory inventory](../../../../Templates/directory.txt) | visible source labels only | 2026-11-09 | original bodies unavailable; recheck if supplied |
| SRC-M42-02 | [clinical decision-support brief](../../../00_SOURCE_COVERAGE_AGENTS.md#scenario-07) | curriculum obligations | 2026-11-09 | planned, not empirical evidence |
| SRC-M42-03 | [payer research brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#deep-payer) | payer scope and evidence boundary | 2026-11-09 | named-company claims need primary sources |

## 11. Collection acceptance checklist

- [ ] Every planned topic has a substantive 12-section sheet meeting the course 1,000-line requirement.
- [ ] Every sheet has checked topic-specific arithmetic/trace, four diagnostic failures, and 10–15 answered questions.
- [ ] The final Common Issues file has at least 25 complete issues and Section 12 last.
- [ ] Primary sources support every empirical or company claim.
- [ ] All paths, prerequisites, glossary entries, cards, evidence labels, and statuses are reconciled.
